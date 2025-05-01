use std::fmt::Display;
use std::num::NonZeroU64;

use crate::backend::language::Language;
use crate::call_info;
use crate::common::po2::U32PowerOf2;
use crate::frontend::any::Any;
use crate::frontend::any::{record_node, InvalidReason};
use crate::frontend::encoding::{EncodingErrorKind, EncodingGuard};
use crate::frontend::error::InternalError;
use crate::ir::expr::Binding;
use crate::ir::expr::Expr;
use crate::ir::ir_type::{
    check_layout, get_type_for_buffer_binding_type, AccessModeReadable, HandleType, LayoutConstraints, LayoutError,
    LayoutErrorContext, SamplesPerPixel,
};
use crate::ir::pipeline::{PipelineError, StageMask, WipBinding, WipPushConstantsField};
use crate::ir::recording::Context;
use crate::ir::{self, StoreType, TextureFormatWrapper, TextureSampleUsageType, Type};
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
    #[error("invalid type `{0:?}` for binding of kind `{1:?}`")]
    InvalidTypeForBinding(ir::StoreType, BindingType),
    #[error("the type `{0:?}` cannot be used for a binding of kind `{1:?}` because of its layout.\n{2}")]
    TypeHasInvalidLayoutForBinding(ir::StoreType, BindingType, LayoutError),
}

impl Any {
    /// import the resource bound at `path` of kind `binding_ty`.
    /// the shader type `ty` must correspond to the `binding_ty` according to https://www.w3.org/TR/WGSL/#var-decls
    /// (paragraphs on uniform buffer, storage buffer etc.)
    ///
    /// -----
    ///
    /// `buffer_binding_as_ref`: how the resulting `Any` is returned.
    /// - `true`: returns `Ref<ty, ...>` (`binding_ty` must be `BindingType::Buffer`)
    /// - `false`: returns `ty` (`binding_ty` must be a uniform or read-only storage buffer, `ty` must be constructible)
    ///
    /// if the conditions mentioned above are not met, an `EncodingError` is pushed and an invalid `Any` is returned.
    #[track_caller]
    pub fn binding(
        path: BindPath,
        visibility: StageMask,
        ty: ir::StoreType,
        binding_ty: BindingType,
        buffer_binding_as_ref: bool,
    ) -> Any {
        let record_handle_node = |ty, call_info| {
            record_node(
                call_info,
                Expr::PipelineIo(PipelineIo::Binding(Binding { bind_path: path, ty })),
                &[],
            )
        };
        // this closure checks whether `binding_ty` and `ty` are compatible
        let create_any = |ctx: &Context| -> Result<Any, EncodingErrorKind> {
            ctx.push_error_if_outside_encoding_scope("import of bindings");
            let call_info = ctx.latest_user_caller();

            if buffer_binding_as_ref && !matches!(binding_ty, BindingType::Buffer { .. }) {
                return Err(LayoutError::NonRefBufferRequiresReadOnlyAndConstructible.into());
            }

            let any = match &binding_ty {
                BindingType::Buffer {
                    ty: buffer_ty,
                    has_dynamic_offset,
                } => {
                    let ref_or_value_ty =
                        get_type_for_buffer_binding_type(&ty, *buffer_ty, buffer_binding_as_ref, ctx)?;
                    Ok(record_handle_node(ref_or_value_ty, ctx.latest_user_caller()))
                }
                BindingType::Sampler(s1) => match &ty {
                    StoreType::Handle(HandleType::Sampler(s2)) if s1 == s2 => {
                        Ok(record_handle_node(Type::Store(ty.clone()), call_info))
                    }
                    _ => Err(BindingError::InvalidTypeForBinding(ty.clone(), binding_ty.clone())),
                },
                BindingType::SampledTexture {
                    shape: s0,
                    sample_type: t0,
                    samples_per_pixel: spp0,
                } => match &ty {
                    StoreType::Handle(HandleType::SampledTexture(s1, t1, spp1))
                        if (t0 == t1) && (s0 == s1) && (spp0 == spp1) =>
                    {
                        Ok(record_handle_node(Type::Store(ty.clone()), call_info))
                    }

                    _ => Err(BindingError::InvalidTypeForBinding(ty.clone(), binding_ty.clone())),
                },
                BindingType::StorageTexture {
                    shape: s0,
                    format: f0,
                    access: a0,
                } => match &ty {
                    StoreType::Handle(HandleType::StorageTexture(s1, f1, a1))
                        if (a0 == a1) && (f0 == f1) && (s0 == s1) =>
                    {
                        Ok(record_handle_node(Type::Store(ty.clone()), call_info))
                    }

                    _ => Err(BindingError::InvalidTypeForBinding(ty.clone(), binding_ty.clone())),
                },
            };

            match ctx.pipeline_layout_mut().bindings.entry(path) {
                Entry::Occupied(entry) => {
                    Err(PipelineError::DuplicateBindPath(path, entry.get().shader_ty.clone()).into())
                }
                Entry::Vacant(entry) => match any {
                    Ok(any) => match any.node() {
                        Some(node) => {
                            entry.insert(WipBinding {
                                call_info,
                                user_defined_visibility: visibility,
                                binding_ty,
                                shader_ty: ty,
                                node,
                            });
                            Ok(any)
                        }
                        None => Err(InternalError::new(
                            true,
                            format!("binding at `{path}` produced invalid Any object"),
                        )
                        .into()),
                    },
                    Err(err) => Err(err.into()),
                },
            }
        };
        Context::try_with(call_info!(), |ctx| match create_any(ctx) {
            Ok(any) => any,
            Err(e) => {
                ctx.push_error(e);
                Any::new_invalid(InvalidReason::ErrorThatWasPushed)
            }
        })
        .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
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
        ty: ir::SizedType,
        custom_min_size: Option<u64>,
        custom_min_align: Option<U32PowerOf2>,
    ) -> Any {
        Context::try_with(call_info!(), |ctx| {
            let mut push_constants = &mut ctx.pipeline_layout_mut().push_constants;
            let field_index = push_constants.len();

            let store_ty = StoreType::Sized(ty.clone());
            if let Err(e) = check_layout(
                &LayoutErrorContext {
                    binding_type: BufferBindingType::Storage(AccessModeReadable::Read),
                    expected_constraints: LayoutConstraints::Wgsl(ir::ir_type::WgslBufferLayout::StorageAddressSpace),
                    top_level_type: store_ty.clone(),
                    use_color: ctx.settings().colored_error_messages,
                },
                &store_ty.clone(),
            ) {
                ctx.push_error(e.into());
            }

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
