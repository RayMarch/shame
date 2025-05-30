use std::fmt::Display;
use std::num::NonZeroU64;

use crate::any::layout::{repr, GpuTypeLayout, LayoutableType, TypeRepr};
use crate::backend::language::Language;
use crate::call_info;
use crate::common::po2::U32PowerOf2;
use crate::frontend::any::Any;
use crate::frontend::any::{record_node, InvalidReason};
use crate::frontend::encoding::{EncodingErrorKind, EncodingGuard};
use crate::frontend::error::InternalError;
use crate::frontend::rust_types::type_layout::layoutable;
use crate::frontend::rust_types::type_layout::layoutable::ir_compat::IRConversionError;
use crate::ir::expr::Binding;
use crate::ir::expr::Expr;
use crate::ir::ir_type::{AccessModeReadable, HandleType, SamplesPerPixel};
use crate::ir::pipeline::{PipelineError, StageMask, WipBinding, WipPushConstantsField};
use crate::ir::recording::{Context, MemoryRegion};
use crate::ir::{self, AddressSpace, StoreType, TextureFormatWrapper, TextureSampleUsageType, Type};
use crate::ir::{ir_type::TextureShape, AccessMode};
use std::collections::btree_map::Entry;
use thiserror::Error;

use super::expr::{PipelineIo, PushConstantsField};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// `BindPath(bind_group, binding)`: a pair of bind-group index and binding index
pub struct BindPath(pub u32, pub u32);

impl std::fmt::Display for BindPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bind-group: {}, binding: {}", self.0, self.1)
    }
}

/// (no documentation yet)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BindingType {
    /// (no documentation yet)
    Buffer {
        /// (no documentation yet)
        ty: BufferBindingType,
        /// (no documentation yet)
        has_dynamic_offset: bool,
    },
    /// (no documentation yet)
    Sampler(SamplingMethod),
    /// (no documentation yet)
    SampledTexture {
        /// (no documentation yet)
        shape: TextureShape,
        /// (no documentation yet)
        sample_type: TextureSampleUsageType,
        /// (no documentation yet)
        samples_per_pixel: SamplesPerPixel,
    },
    /// (no documentation yet)
    StorageTexture {
        /// (no documentation yet)
        shape: TextureShape,
        /// (no documentation yet)
        format: TextureFormatWrapper,
        /// (no documentation yet)
        access: AccessMode,
    },
}

impl BindingType {
    /// (no documentation yet)
    pub fn max_supported_stage_visibility(&self, vertex_writable_storage_supported: bool) -> StageMask {
        let is_writeable_storage = match self {
            BindingType::Buffer { ty, has_dynamic_offset } => match ty {
                BufferBindingType::Uniform => false,
                BufferBindingType::Storage(access) => match access {
                    AccessModeReadable::Read => false,
                    AccessModeReadable::ReadWrite => true,
                },
            },
            BindingType::Sampler(_) => false,
            BindingType::SampledTexture { .. } => false,
            BindingType::StorageTexture { .. } => true,
        };

        if is_writeable_storage {
            if vertex_writable_storage_supported {
                StageMask::all()
            } else {
                StageMask::all() & !StageMask::vert()
            }
        } else {
            StageMask::all()
        }
    }

    /// whether interactions with this binding (e.g. writes) can produce
    /// observable side effects when running a shader
    pub(crate) fn can_produce_side_effects(&self) -> bool {
        match self {
            BindingType::Buffer { ty, has_dynamic_offset } => match ty {
                BufferBindingType::Uniform => false,
                BufferBindingType::Storage(access) => AccessMode::from(*access).is_writeable(),
            },
            BindingType::Sampler(_) => false,
            BindingType::SampledTexture {
                shape: _,
                sample_type: _,
                samples_per_pixel: _,
            } => false,
            BindingType::StorageTexture {
                shape: _,
                format: _,
                access,
            } => access.is_writeable(),
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferBindingType {
    Uniform,
    Storage(AccessModeReadable),
}

/// a sampler's texture sampling method
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplingMethod {
    /// filtering (bilinear, trilinear, anisotropic etc.)
    #[default]
    Filtering,
    /// nearest-neightbor sampling
    NonFiltering,
    /// comparison sampling (for depth textures)
    Comparison,
}

impl Display for SamplingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SamplingMethod::Filtering => "filtering",
            SamplingMethod::NonFiltering => "non-filtering",
            SamplingMethod::Comparison => "comparison",
        })
    }
}

#[allow(missing_docs)]
#[derive(Debug, Error, Clone)]
pub enum BindingError {
    #[error("the type `{0:?}` cannot be used for a binding of kind `{1:?}` because of:\n{0}")]
    TypeHasInvalidLayoutForBinding(LayoutableType, BindingType, IRConversionError),
    #[error("a non-reference buffer (non `BufferRef`) must be both read-only and constructible")]
    NonRefBufferRequiresReadOnlyAndConstructible,
}

fn layout_to_store_type<T: TypeRepr>(
    layout: GpuTypeLayout<T>,
    binding_ty: &BindingType,
) -> Result<StoreType, EncodingErrorKind> {
    let store_type: ir::StoreType = layout.layoutable_type().clone().try_into().map_err(|e| {
        BindingError::TypeHasInvalidLayoutForBinding(layout.layoutable_type().clone(), binding_ty.clone(), e)
    })?;

    // https://www.w3.org/TR/WGSL/#host-shareable-types
    if !store_type.is_host_shareable() {
        return Err(InternalError::new(
            true,
            format!(
                "LayoutableType to StoreType conversion did not result in a host-shareable type. LayoutableType:\n{}",
                layout.layoutable_type(),
            ),
        )
        .into());
    }

    Ok(store_type)
}

fn record_and_register_binding(
    ctx: &Context,
    path: BindPath,
    visibility: StageMask,
    binding_ty: BindingType,
    store_type: StoreType,
    ty: Type,
) -> Result<Any, EncodingErrorKind> {
    let any = record_node(
        ctx.latest_user_caller(),
        Expr::PipelineIo(PipelineIo::Binding(Binding { bind_path: path, ty })),
        &[],
    );

    match ctx.pipeline_layout_mut().bindings.entry(path) {
        Entry::Occupied(entry) => Err(PipelineError::DuplicateBindPath(path, entry.get().shader_ty.clone()).into()),
        Entry::Vacant(entry) => match any.node() {
            Some(node) => {
                entry.insert(WipBinding {
                    call_info: call_info!(),
                    user_defined_visibility: visibility,
                    binding_ty,
                    shader_ty: store_type,
                    node,
                });
                Ok(any)
            }
            None => Err(InternalError::new(true, format!("binding at `{path}` produced invalid Any object")).into()),
        },
    }
}

#[track_caller]
fn create_any_catch_errors(create_any: impl FnOnce(&Context) -> Result<Any, EncodingErrorKind>) -> Any {
    Context::try_with(call_info!(), |ctx| match create_any(ctx) {
        Ok(any) => any,
        Err(e) => {
            ctx.push_error(e);
            Any::new_invalid(InvalidReason::ErrorThatWasPushed)
        }
    })
    .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
}

impl Any {
    /// Creates a storage buffer binding at the specified bind path.
    #[track_caller]
    pub fn storage_buffer_binding(
        bind_path: BindPath,
        visibility: StageMask,
        layout: GpuTypeLayout<repr::Storage>,
        access: AccessModeReadable,
        buffer_binding_as_ref: bool,
        has_dynamic_offset: bool,
    ) -> Any {
        let create_any = |ctx: &Context| -> Result<Any, EncodingErrorKind> {
            let binding_type = BindingType::Buffer {
                ty: BufferBindingType::Storage(access),
                has_dynamic_offset,
            };
            let store_type = layout_to_store_type(layout, &binding_type)?;

            let ty = match (buffer_binding_as_ref, access) {
                (true, _) => Type::Ref(
                    MemoryRegion::new(
                        ctx.latest_user_caller(),
                        store_type.clone(),
                        None,
                        None,
                        access.into(),
                        AddressSpace::Storage,
                    )?,
                    store_type.clone(),
                    access.into(),
                ),
                (false, AccessModeReadable::ReadWrite) => {
                    return Err(BindingError::NonRefBufferRequiresReadOnlyAndConstructible.into());
                }
                (false, AccessModeReadable::Read) => {
                    if !store_type.is_constructible() {
                        return Err(BindingError::NonRefBufferRequiresReadOnlyAndConstructible.into());
                    }
                    Type::Store(store_type.clone())
                }
            };

            record_and_register_binding(ctx, bind_path, visibility, binding_type, store_type, ty)
        };

        create_any_catch_errors(create_any)
    }

    /// Creates a uniform buffer binding at the specified bind path.
    #[track_caller]
    pub fn uniform_buffer_binding(
        bind_path: BindPath,
        visibility: StageMask,
        layout: GpuTypeLayout<repr::Uniform>,
        has_dynamic_offset: bool,
    ) -> Any {
        let create_any = |ctx: &Context| -> Result<Any, EncodingErrorKind> {
            let binding_type = BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset,
            };
            let store_type = layout_to_store_type(layout, &binding_type)?;
            if !store_type.is_constructible() {
                return Err(BindingError::NonRefBufferRequiresReadOnlyAndConstructible.into());
            }

            let ty = Type::Store(store_type.clone());

            record_and_register_binding(ctx, bind_path, visibility, binding_type, store_type, ty)
        };

        create_any_catch_errors(create_any)
    }

    /// Creates a handle binding at the specified bind path.
    #[track_caller]
    pub fn handle_binding(bind_path: BindPath, visibility: StageMask, handle_type: HandleType) -> Any {
        let create_any = |ctx: &Context| -> Result<Any, EncodingErrorKind> {
            let binding_type = match &handle_type {
                HandleType::Sampler(s) => BindingType::Sampler(*s),
                HandleType::SampledTexture(s, t, p) => BindingType::SampledTexture {
                    shape: *s,
                    sample_type: *t,
                    samples_per_pixel: *p,
                },
                HandleType::StorageTexture(s, f, a) => BindingType::StorageTexture {
                    shape: *s,
                    format: f.clone(),
                    access: *a,
                },
            };

            let store_type = StoreType::Handle(handle_type);
            let ty = Type::Store(store_type.clone());

            record_and_register_binding(ctx, bind_path, visibility, binding_type, store_type, ty)
        };

        create_any_catch_errors(create_any)
    }

    /// get the next field of type `ty` in the push-constants struct.
    ///
    /// consecutive fields are assumed to be aligned in the same manner as
    /// fields in a `#[derive(GpuLayout)]` struct, which correspond to the WGSL
    /// struct layout rules: https://www.w3.org/TR/WGSL/#structure-member-layout
    ///
    /// The push constant visibility ranges of individual shader stages (e.g. vertex vs fragment)
    /// which are returned at the end of the pipeline encoding
    /// are inferred at the granularity of the fields returned by subsequent calls
    /// to this function.
    /// > Note:
    /// > This means calling this function once to return a single
    /// > [`Any`] instance of type [`SizedType::Structure`]
    /// > will always result in either the entire
    /// > struct being visible or invisible in a given shader stage.
    #[track_caller]
    pub fn next_push_constants_field(
        ty: layoutable::SizedType,
        custom_min_size: Option<u64>,
        custom_min_align: Option<U32PowerOf2>,
    ) -> Any {
        Context::try_with(call_info!(), |ctx| {
            let mut push_constants = &mut ctx.pipeline_layout_mut().push_constants;
            let field_index = push_constants.len();

            // This is dead code, but makes sure that if we ever decide that a SizedType
            // is not trivially layoutable as repr::Storage, it gets caught here.
            let _ = GpuTypeLayout::<repr::Storage>::new(ty.clone());

            let ty = match ir::SizedType::try_from(ty) {
                Ok(ty) => ty,
                Err(e) => {
                    ctx.push_error(e.into());
                    return Any::new_invalid(InvalidReason::ErrorThatWasPushed);
                }
            };

            let any = record_node(
                ctx.latest_user_caller(),
                // TODO(release) remove this redundancy of `ty`
                Expr::PipelineIo(PipelineIo::PushConstantsField(PushConstantsField {
                    field_index,
                    ty: ty.clone(),
                })),
                &[],
            );

            if let Some(node) = any.node() {
                push_constants.push(WipPushConstantsField::new(
                    ty,
                    custom_min_size,
                    custom_min_align,
                    ctx.first_user_caller(),
                    node,
                ));
            } else {
                // error was already pushed by `record_node` if any is invalid
            }

            any
        })
        .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
    }
}
