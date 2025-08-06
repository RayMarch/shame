#![allow(clippy::let_and_return)]
use std::{cell::Cell, iter, marker::PhantomData, rc::Rc};

use crate::{
    call_info,
    common::integer::post_inc_u32,
    frontend::{
        any::{
            render_io::{Attrib, Location, VertexAttribFormat, VertexBufferLayout},
            shared_io::{BindPath, BindingType},
            Any, InvalidReason,
        },
        error::InternalError,
        rust_types::{
            error::FrontendError,
            layout_traits::{
                cpu_type_name_and_layout, get_layout_compare_with_cpu_push_error, ArrayElementsUnsizedError, FromAnys,
                GpuLayout, VertexLayout,
            },
            reference::AccessMode,
            struct_::SizedFields,
            type_traits::{BindingArgs, GpuSized, GpuStore, GpuStoreImplCategory, NoAtomics, NoBools},
            GpuType,
        },
        texture::{
            texture_array::{StorageTextureArray, TextureArray},
            texture_traits::{
                LayerCoords, SamplingFormat, SamplingMethod, Spp, StorageTextureCoords, StorageTextureFormat,
                SupportsCoords, SupportsSpp, TextureCoords,
            },
            Sampler, Texture, TextureKind,
        },
    },
    ir::{
        self,
        ir_type::{Field, LayoutError},
        pipeline::{PipelineError, StageMask},
        recording::Context,
        TextureFormatWrapper,
    },
};

use super::{binding::Binding, rasterizer::VertexIndex};

/// an iterator over the draw command's bound vertex buffers, which also
/// allows random access
///
/// use `.next()` or `.at(...)`/`.index(...)` to access individual vertex buffers
pub struct VertexBufferIter {
    next_slot: u32,
    location_counter: Rc<LocationCounter>,
    private_ctor: (),
}

impl VertexBufferIter {
    pub(crate) fn new() -> Self {
        Self {
            next_slot: 0,
            location_counter: Rc::new(0.into()),
            private_ctor: (),
        }
    }

    /// access the `i`th vertex buffer
    /// that was bound when the draw command was scheduled and interpret its
    /// type as `T`.
    #[track_caller]
    pub fn at<T: VertexLayout>(&mut self, i: u32) -> VertexBuffer<T> {
        self.next_slot = i + 1;
        VertexBuffer::new(i, self.location_counter.clone())
    }

    /// access the `i`th vertex buffer
    /// that was bound when the draw command was scheduled and interpret its
    /// type as `T`.
    #[track_caller]
    pub fn index<T: VertexLayout>(&mut self, i: u32) -> VertexBuffer<T> {
        // just to be consistent with the other `at` functions, where the
        // `shame::Index` trait provides the `index` alternative, we offer both as
        // type associated functions here. Choose which one you like better.
        self.at(i)
    }

    /// access the next vertex buffer (or the first if no buffer was imported yet)
    /// that was bound when the draw command was scheduled and interpret its
    /// type as `T`.
    #[allow(clippy::should_implement_trait)] // not fallible
    #[track_caller]
    pub fn next<T: VertexLayout>(&mut self) -> VertexBuffer<T> {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.at(slot)
    }
}

/// a buffer containing an array of `T` where `T` has special layout rules (= may only contain vectors and scalars) and can only
/// be looked up once by a [`VertexIndex`] of a render pipeline.
pub struct VertexBuffer<'a, T: VertexLayout> {
    slot: u32,
    attribs_and_stride: Result<(Box<[Attrib]>, u64), InvalidReason>,
    phantom: PhantomData<&'a [T]>,
}

impl<T: VertexLayout> VertexBuffer<'_, T> {
    #[track_caller]
    fn new(slot: u32, location_counter: Rc<LocationCounter>) -> Self {
        let call_info = call_info!();
        let attribs_and_stride = Context::try_with(call_info, |ctx| {
            let skip_stride_check = false; // it is implied that T is in an array, the strides must match
            let layout = get_layout_compare_with_cpu_push_error::<T>(ctx, skip_stride_check);

            let attribs_and_stride = Attrib::get_attribs_and_stride(&layout, &location_counter).ok_or_else(|| {
                ctx.push_error(FrontendError::MalformedVertexBufferLayout(layout).into());
                InvalidReason::ErrorThatWasPushed
            });

            if let Ok((new_attribs, _)) = &attribs_and_stride {
                let rp = ctx.render_pipeline();
                if let Err(e) = ensure_locations_are_unique(slot, ctx, &rp, new_attribs) {
                    ctx.push_error(e.into());
                }
            }

            attribs_and_stride
        })
        .unwrap_or(Err(InvalidReason::CreatedWithNoActiveEncoding));

        Self {
            slot,
            attribs_and_stride,
            phantom: PhantomData,
        }
    }
}

/// checks that there are no duplicate vertex attribute locations and vertex buffer slots
fn ensure_locations_are_unique(
    slot: u32,
    ctx: &Context,
    rp: &ir::pipeline::WipRenderPipelineDescriptor,
    new_attribs: &[Attrib],
) -> Result<(), PipelineError> {
    for vbuf in &rp.vertex_buffers {
        if vbuf.index == slot {
            return Err(PipelineError::DuplicateVertexBufferImport(slot));
        }
        for existing_attrib in &vbuf.attribs {
            if new_attribs.iter().any(|a| a.location == existing_attrib.location) {
                return Err(PipelineError::DuplicateAttribLocation {
                    location: existing_attrib.location,
                    buffer_a: vbuf.index,
                    buffer_b: slot,
                });
            }
        }
    }
    Ok(())
}

pub struct LocationCounter(Cell<u32>);

impl LocationCounter {
    pub(crate) fn next(&self) -> Location {
        let i = self.0.get();
        self.0.set(i + 1);
        Location(i)
    }
}

impl From<u32> for LocationCounter {
    fn from(value: u32) -> Self { Self(Cell::new(value)) }
}

impl<T: VertexLayout> VertexBuffer<'_, T> {
    #[track_caller]
    /// perform a vertex buffer lookup with a special [`VertexIndex`]
    ///
    /// Vertex buffers are special arrays that can only be indexed **once** by the vertex-index
    /// or by the instance-index associated with every vertex.
    /// (see geometry instancing: https://sotrh.github.io/learn-wgpu/beginner/tutorial7-instancing/).
    ///
    /// after indexing the `VertexBuffer` object is consumed and cannot be used for
    /// another index lookup.
    ///
    /// There are only two [`VertexIndex`] objects in `shame`:
    /// - `draw_context.vertices.index` the per-vertex index as defined by the `sm::Indexing` sequence
    /// - `draw_context.vertices.instance_index` the instance index that each vertex belongs to
    ///
    /// ## Example
    ///
    /// ```
    /// #[derive(sm::GpuLayout)]
    /// // vertex buffers support #[gpu_repr(packed)] which avoids padding bytes
    /// struct MyVertex {
    ///     pos: f32x3,
    ///     nor: f32x3,
    /// }
    /// let vb: sm::VertexBuffer<MyVertex> = drawcall.vertices.buffers.next();
    /// let vertex = vb.index(drawcall.vertices.index);
    /// ```
    ///
    /// see "Fixed Function Vertex Processing" https://docs.vulkan.org/spec/latest/chapters/fxvertex.html
    /// for more information
    pub fn index(self, index: VertexIndex) -> T {
        // just to be consistent with the other `at` functions, where the
        // `shame::Index` trait provides the `index` alternative, we offer both as
        // type associated functions here. Choose which one you like better.
        self.at(index)
    }

    /// perform a vertex buffer lookup with a special [`VertexIndex`]
    ///
    /// Vertex buffers are special arrays that can only be indexed **once** by the vertex-index
    /// or by the instance-index associated with every vertex.
    /// (see geometry instancing: https://sotrh.github.io/learn-wgpu/beginner/tutorial7-instancing/).
    ///
    /// after indexing the `VertexBuffer` object is consumed and cannot be used for
    /// another index lookup.
    ///
    /// There are only two [`VertexIndex`] objects in `shame`:
    /// - `draw_context.vertices.index` the per-vertex index as defined by the `sm::Indexing` sequence
    /// - `draw_context.vertices.instance_index` the instance index that each vertex belongs to
    ///
    /// ## Example
    ///
    /// ```
    /// #[derive(sm::GpuLayout)]
    /// // vertex buffers support #[gpu_repr(packed)] which avoids padding bytes
    /// struct MyVertex {
    ///     pos: f32x3,
    ///     nor: f32x3,
    /// }
    /// let vb: sm::VertexBuffer<MyVertex> = drawcall.vertices.buffers.next();
    /// let vertex = vb.at(drawcall.vertices.index);
    /// ```
    ///
    /// see "Fixed Function Vertex Processing" https://docs.vulkan.org/spec/latest/chapters/fxvertex.html
    /// for more information
    #[track_caller]
    pub fn at(self, index: VertexIndex) -> T {
        let lookup = index.0;

        let invalid_with_reason =
            |reason| T::from_anys(std::iter::repeat_n(Any::new_invalid(reason), T::expected_num_anys()));

        Context::try_with(call_info!(), |ctx| match self.attribs_and_stride {
            Ok((attribs, stride)) => T::from_anys(
                Any::vertex_buffer(
                    self.slot,
                    VertexBufferLayout {
                        stride,
                        lookup,
                        attribs,
                    },
                )
                .into_iter(),
            ),
            Err(reason) => invalid_with_reason(reason),
        })
        .unwrap_or_else(|| invalid_with_reason(InvalidReason::CreatedWithNoActiveEncoding))
    }
}

/// access to buffers, textures, etc. that were bound in groups
///
/// ## example usage:
///
/// ```
/// use shame as sm;
/// // access the next group (0..)
/// let mut group0 = bind_groups.next();
/// let mut group1 = bind_groups.next();
/// // access a specific group
/// let mut group4 = bind_groups.at(4);
///
/// // downcast its bindings
/// let buf: sm::Buffer<sm::Array<sm::f32x4>> = group0.next();
/// ```
/// see the documentation of [`BindingIter::next`] for
/// more binding-type examples
///
/// [`BindingIter::next`]: crate::BindingIter::next
pub struct BindGroupIter<'a> {
    next: u32,
    private_ctor: (),
    phantom: PhantomData<&'a ()>,
}

impl<'a> BindGroupIter<'a> {
    pub(crate) fn new() -> Self {
        Self {
            next: 0,
            private_ctor: (),
            phantom: PhantomData,
        }
    }

    /// access the `i`th bind group that is bound during the drawcall/compute dispatch.
    ///
    /// returns a binding iterator that allows iteration over/ or random access
    /// to the bindings of that bind group.
    #[track_caller]
    pub fn at(&mut self, group: u32) -> BindingIter {
        self.next = group + 1;
        BindingIter {
            next: BindPath(group, 0),
            phantom: PhantomData,
        }
    }

    /// access the `i`th bind group that is bound during the drawcall/compute dispatch.
    ///
    /// returns a binding iterator that allows iteration over/ or random access
    /// to the bindings of that bind group.
    #[track_caller]
    pub fn index(&mut self, i: u32) -> BindingIter { self.at(i) }

    /// access the next bind group that is bound during the drawcall/compute dispatch.
    ///
    /// returns the first bind group (index = 0) if no other bind group was accessed
    /// before, otherwise the bind group with index one higher than the last one.
    ///
    /// returns a binding iterator that allows iteration over/ or random access
    /// to the bindings of that bind group.
    #[track_caller]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> BindingIter<'a> {
        BindingIter {
            next: BindPath(post_inc_u32(&mut self.next), 0),
            phantom: PhantomData,
        }
    }
}

/// iterator over the bindings of a bind group that also allows random access.
///
/// see [`BindingIter::next`] and [`BindingIter::at`]/[`BindingIter::index`]
pub struct BindingIter<'a> {
    next: BindPath,
    phantom: PhantomData<&'a ()>,
}

impl BindingIter<'_> {
    /// access the binding at index `index` within the bind group.
    ///
    /// ```
    /// let b: sm::Buffer<T> = bind_group.at(4);
    /// ```
    ///
    /// see the documentation of [`BindingIter::next`] for more info on `T`
    /// and more examples.
    ///
    /// [`BindingIter::next`]: crate::BindingIter::next
    #[track_caller]
    pub fn at<T: Binding>(&mut self, index: u32) -> T {
        Context::try_with(call_info!(), |ctx| {
            let vert_write_storage = ctx.settings().vertex_writable_storage_by_default;
            let max_visibility = T::binding_type().max_supported_stage_visibility(vert_write_storage);
            self.at_with_visibility(index, max_visibility)
        })
        .unwrap_or_else(|| {
            self.next = BindPath(self.next.0, index + 1);
            Binding::new_binding(Err(InvalidReason::CreatedWithNoActiveEncoding))
        })
    }

    /// access the binding at index `index` within the bind group.
    /// with a custom shader stage visibility mask
    ///
    /// see the documentation of [`BindingIter::next_with_visibility`] for more info on `T`
    /// and more examples.
    ///
    /// [`BindingIter::next_with_visibility`]: crate::BindingIter::next_with_visibility
    #[track_caller]
    pub fn at_with_visibility<T: Binding>(&mut self, index: u32, stages: StageMask) -> T {
        self.next = BindPath(self.next.0, index + 1);
        Binding::new_binding(Ok(BindingArgs {
            path: BindPath(self.next.0, index),
            visibility: stages,
        }))
    }

    /// access the binding at index `index` within the bind group.
    ///
    /// ```
    /// let b: sm::Buffer<T> = bind_group.index(4);
    /// ```
    ///
    /// see the documentation of [`BindingIter::next`] for more info on `T`
    /// and more examples.
    ///
    /// [`BindingIter::next`]: crate::BindingIter::next
    #[track_caller]
    pub fn index<T: Binding>(&mut self, index: u32) -> T { self.at(index) }

    /// access the next binding of the bind group. (last index + 1)
    ///
    /// # usage
    ///
    /// ## Storage Buffers
    /// ```
    /// let b: sm::Buffer   <T> = bind_group.next();
    /// let b: sm::BufferRef<T> = bind_group.next();
    /// ```
    /// where `T` is a `#[derive(GPULayout)]` struct, or `sm::array`, `sm::vec`, `sm::mat`, ...
    /// (use `BufferRef` for mutable data, atomics and runtime-sized `GpuLayout` structs)
    ///
    /// ---
    /// ## Uniform Buffers
    /// ```
    /// use shame::mem;
    /// let b: sm::Buffer   <T, mem::Uniform> = bind_group.next();
    /// let b: sm::BufferRef<T, mem::Uniform> = bind_group.next();
    /// ```
    /// where `T` is a `#[derive(GPULayout)]` struct `S`, or `sm::array`, `sm::vec`, `sm::mat` ...
    /// (use `BufferRef` for mutable data, atomics and runtime-sized `GpuLayout` structs)
    ///
    /// ---
    /// ## Samplers
    /// ```
    /// let b: sm::Sampler<sm::Nearest>    = bind_group.next();
    /// let b: sm::Sampler<sm::Filtering>  = bind_group.next();
    /// let b: sm::Sampler<sm::Comparison> = bind_group.next();
    /// ```
    ///
    /// ---
    /// ## Textures with known format
    /// ```
    /// // non exhaustive examples:
    /// let texrg  : sm::Texture<sm::tf::Rg8Unorm> = bind_group.next();
    /// let texrgba: sm::Texture<sm::tf::Rgba8Unorm> = bind_group.next();
    /// let tex3d  : sm::Texture<sm::tf::Rgba8Unorm, f32x3> = bind_group.next();
    /// let msaa2d : sm::Texture<sm::tf::Rgba8Unorm, f32x2, Multi> = bind_group.next();
    /// let texcube: sm::Texture<sm::tf::Rgba8Unorm, sm::CubeDir> = bind_group.next();
    /// let depth  : sm::Texture<sm::tf::Depth24Plus> = bind_group.next();
    /// ```
    ///
    /// ---
    /// ## Textures with unknown format
    /// (the compiler cannot help as much with invalid usage when the format is unknown)
    /// ```
    /// // non exhaustive examples:
    /// let filt_rgba: sm::Texture<sm::Filterable<f32x4>> = bind_group.next();
    /// let filt_rgb : sm::Texture<sm::Filterable<f32x3>> = bind_group.next();
    /// let filt_rg  : sm::Texture<sm::Filterable<f32x2>> = bind_group.next();
    /// let filt_r   : sm::Texture<sm::Filterable<f32x1>> = bind_group.next();
    /// let filt_rgba: sm::Texture<sm::NonFilterable<f32x4>> = bind_group.next();
    /// let depth_tex: sm::Texture<sm::Depth> = bind_group.next();
    /// ```
    ///
    /// ---
    /// ## Arrays of textures
    /// ```
    /// let texarr: sm::TextureArray<sm::tf::Rgba8Unorm, 4> = bind_group.next();
    /// let texarr: sm::TextureArray<sm::Filterable<f32x4>, 4> = bind_group.next();
    /// ```
    /// ---
    /// ## storage textures
    /// ```
    /// let texsto: sm::StorageTexture<sm::tf::Rgba8Unorm> = bind_group.next();
    /// let texsto: sm::StorageTexture<sm::tf::Rgba8Unorm, u32x2> = bind_group.next();
    /// ```
    /// ---
    /// ## Arrays of storage textures
    /// ```
    /// let texstoarr: sm::StorageTextureArray<sm::tf::Rgba8Unorm, 4> = bind_group.next();
    /// let texstoarr: sm::StorageTextureArray<sm::tf::Rgba8Unorm, 4, u32x2, sm::Write> = bind_group.next();
    /// ```
    ///
    //TODO(release) mention/explain default visibility
    #[track_caller]
    #[allow(clippy::should_implement_trait)]
    pub fn next<T: Binding>(&mut self) -> T {
        let path = self.post_inc_path();
        Context::try_with(call_info!(), |ctx| {
            let vert_write_storage = ctx.settings().vertex_writable_storage_by_default;
            let max_visibility = T::binding_type().max_supported_stage_visibility(vert_write_storage);
            Binding::new_binding(Ok(BindingArgs {
                path,
                visibility: max_visibility,
            }))
        })
        .unwrap_or_else(|| Binding::new_binding(Err(InvalidReason::CreatedWithNoActiveEncoding)))
    }

    /// access the next binding in the bind group with a custom shader stage
    /// visibility mask.
    ///
    /// by default `shame` chooses the maximum viable visibility, that is
    /// every shader stage (except for storage bindings the vertex stage is excluded
    /// if vertex-writable storage is not explicitly enabled in the pipeline encoding
    /// settings)
    ///
    /// see [`BindingIter::next`] for more info on the `T` parameter
    #[track_caller]
    pub fn next_with_visibility<T: Binding>(&mut self, stages: StageMask) -> T {
        Binding::new_binding(Ok(BindingArgs {
            path: self.post_inc_path(),
            visibility: stages,
        }))
    }

    fn post_inc_path(&mut self) -> BindPath {
        let temp = self.next;
        let BindPath(group, binding) = &mut self.next;
        *binding += 1;
        temp
    }
}

/// push-constant values that were set before the current draw command/dispatch.
///
/// use [`PushConstants::get`] to downcast into a specific [`GpuLayout`]
pub struct PushConstants<'a> {
    phantom: PhantomData<&'a ()>,
}

impl PushConstants<'_> {
    pub(crate) fn new() -> Self { Self { phantom: PhantomData } }


    /// interpret the push constants as an instance of `T`.
    ///
    /// This function can only be called once per pipeline.
    ///
    /// ## examples:
    /// vectors/matrices
    /// ```
    /// let pc: shame::f32x4 = push_constants.get();
    /// ```
    /// ---
    /// [`GpuSized`] arrays
    /// ```
    /// let pc: shame::Array<shame::f32x4, shame::Size<4>> = push_constants.get();
    /// ```
    /// ---
    /// [`GpuLayout`] structs, if they are [`GpuSized`] and match the [`mem::Storage`] layout
    /// ```
    /// #[derive(shame::GpuLayout)]
    /// struct Things {
    ///     a: shame::f32x4,
    ///     b: shame::u32x2,
    /// }
    /// let pc: Things = push_constants.get();
    /// ```
    ///
    /// ## [`mem::Storage`] layout requirements
    /// `T` needs to conform to the layout requirements of the [`mem::Storage`] address space.
    /// This is the default layout when using `#[derive(GpuLayout)]`, so no further adjustments should
    /// be needed. In case they are, see the *Storage* layout requirements at
    /// https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    ///
    /// > maintainer note:
    /// > sources for the layout requirements of push constants:
    /// > https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GL_KHR_vulkan_glsl.txt#L216
    ///
    /// [`mem::Storage`]: crate::mem::Storage
    #[track_caller]
    pub fn get<T>(self) -> T
    where
        T: GpuStore + GpuSized + NoAtomics + NoBools + GpuLayout,
    {
        let _caller_scope = Context::call_info_scope();

        // the push constants structure as a whole doesn't need to have the same stride
        let skip_stride_check = true;
        Context::try_with(call_info!(), |ctx| {
            let _ = get_layout_compare_with_cpu_push_error::<T>(ctx, skip_stride_check);

            match T::impl_category() {
                GpuStoreImplCategory::Fields(buffer_block) => match buffer_block.last_unsized_field() {
                    None => {
                        assert_eq!(buffer_block.sized_fields().len(), buffer_block.fields().count());
                        let fields = buffer_block.sized_fields().iter().map(|f| {
                            Any::next_push_constants_field(f.ty.clone(), f.custom_min_size, f.custom_min_align)
                        });
                        T::from_anys(fields)
                    }
                    Some(_) => {
                        let msg = format!(
                            "Push constant type `{}` contains unsized last field",
                            buffer_block.name()
                        );
                        let any = ctx.push_error_get_invalid_any(InternalError::new(true, msg).into());
                        T::from_anys(std::iter::repeat_n(any, T::expected_num_anys()))
                    }
                },
                GpuStoreImplCategory::GpuType(ty) => {
                    let any = match ty {
                        ir::StoreType::Sized(sized_type) => Any::next_push_constants_field(sized_type, None, None),
                        _ => {
                            let err = InternalError::new(
                                true,
                                "Unable to obtain sized-type from push constant type, \
                                    even though trait bound `GpuSized` should ensure that"
                                    .into(),
                            );
                            ctx.push_error_get_invalid_any(err.into())
                        }
                    };
                    T::from_anys(std::iter::once(any))
                }
            }
        })
        .unwrap_or_else(|| {
            T::from_anys(std::iter::once(Any::new_invalid(
                InvalidReason::CreatedWithNoActiveEncoding,
            )))
        })
    }
}
