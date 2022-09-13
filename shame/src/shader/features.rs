use crate::assert;
use crate::rec::fields::Fields;
use crate::rec::sampler_texture::{Sampler, TexSampleType, Texture};
use crate::rec::{float4, uint, uint3, TexCoordType};
use crate::rec::{
    texture_combined_sampler::*, AnyDowncast, AsTen, DType, IsShapeScalarOrVec, Rec, Shape, Stage, Ten, WriteOnly,
};
use crate::wrappers::UnsafeAccess;
use shame_graph::{Any, Context, LocationIter};
use std::marker::PhantomData;
use std::ops::Range;

use super::*;
use shame_graph::ShaderKind;

/// panics if there is no recording happening
pub fn current_shader() -> shame_graph::ShaderKind {
    shame_graph::Context::try_with(|ctx| ctx.shader_kind())
        .expect("cannot get current shader while no active recording is happening")
}

pub fn is_fragment_shader() -> bool { current_shader() == ShaderKind::Fragment }
pub fn is_vertex_shader() -> bool { current_shader() == ShaderKind::Vertex }
pub fn is_compute_shader() -> bool { current_shader() == ShaderKind::Compute }
pub fn is_not_fragment_shader() -> bool { current_shader() != ShaderKind::Fragment }
pub fn is_not_vertex_shader() -> bool { current_shader() != ShaderKind::Vertex }
pub fn is_not_compute_shader() -> bool { current_shader() != ShaderKind::Compute }

/// whether there is an active recording happening on the current thread
pub fn is_recording() -> bool { shame_graph::Context::try_with(|_| ()).is_some() }

/// returns a (vertex_shader_glsl, fragment_shader_glsl) pair
pub fn record_render_shaders(mut f: impl FnMut(RenderShaderFeatures)) -> (String, String) {
    let on_error = shame_graph::ErrorBehavior::Panic; //Result error behavior is not fully implemented yet

    let mut record = |shader_kind| {
        let ctx = shame_graph::Context::with_thread_local_context_enabled(shader_kind, on_error, || {
            shame_graph::Context::with(|ctx| {
                ctx.record_shader_main(|| f(RenderShaderFeatures::new()));
            });
        });

        ctx.generate_glsl().expect("error while generating glsl")
    };

    (
        record(shame_graph::ShaderKind::Vertex),
        record(shame_graph::ShaderKind::Fragment),
    )
}

/// returns a glsl compute shader string
pub fn record_compute_shader(f: impl FnOnce(ComputeShaderFeatures)) -> String {
    let on_error = shame_graph::ErrorBehavior::Panic; //Result error behavior is not fully implemented yet
    let shader_kind = shame_graph::ShaderKind::Compute;

    let ctx = shame_graph::Context::with_thread_local_context_enabled(shader_kind, on_error, || {
        shame_graph::Context::with(|ctx| {
            ctx.record_shader_main(|| f(ComputeShaderFeatures::new()));
        });
    });

    let glsl = ctx.generate_glsl().expect("error while generating glsl");

    glsl
}

/// features available in a render shaders recording
pub struct RenderShaderFeatures<'a> {
    /// rasterizer functionality
    pub raster: Raster<'a>,
    /// configure shader inputs and outputs
    pub io: RenderIO<'a>,
}

/// features available in a compute shaders recording
pub struct ComputeShaderFeatures<'a> {
    /// set the work group size and gain access to invocation related ids
    pub dispatch: WorkGroupSetup<'a>,
    /// configure shader inputs and outputs
    pub io: ComputeIO<'a>,
}

/// configure shader inputs and outputs for render shaders
pub struct RenderIO<'a> {
    _phantom: PhantomData<&'a ()>, // prevents constructor from being public
}

/// configure shader inputs and outputs for compute shaders
pub struct ComputeIO<'a> {
    _phantom: PhantomData<&'a ()>, // prevents constructor from being public
}

impl RenderShaderFeatures<'_> {
    pub(crate) fn new() -> Self {
        Self {
            raster: Raster::new(),
            io: RenderIO { _phantom: PhantomData },
        }
    }
}

impl ComputeShaderFeatures<'_> {
    pub(crate) fn new() -> Self {
        Self {
            dispatch: WorkGroupSetup { _phantom: PhantomData },
            io: ComputeIO { _phantom: PhantomData },
        }
    }
}

impl ComputeIO<'_> {
    /// adds a bind group to the currently recorded shader's inputs.
    /// The `binding_index_iterator` specifies how the binding indices should
    /// be counted as bindings are added to the bindgroup.
    #[must_use]
    pub fn group<It>(&self, group_index: u32, binding_index_iterator: It) -> Group<It>
    where
        It: Iterator<Item = u32>,
    {
        Group::new(group_index, binding_index_iterator)
    }
}

impl RenderIO<'_> {
    /// adds a bind group to the currently recorded shader's inputs.
    /// The `binding_index_iterator` specifies how the binding indices should
    /// be counted as bindings are added to the bindgroup.
    #[must_use]
    pub fn group<It>(&self, group_index: u32, binding_index_iterator: It) -> Group<It>
    where
        It: Iterator<Item = u32>,
    {
        Group::new(group_index, binding_index_iterator)
    }

    /// creates a vertex stream builder for adding vertex attributes to the
    /// inputs of the currently recorded shader.
    ///
    /// the attribute location iterator will skip locations that are overlapping
    /// with previously used locations
    /// e.g. if the iterator provides [0, 2, 4..] and the user binds a mat4 and
    /// then a vec4, the mat4 will be assigned location 0, and the vec4 will be
    /// assigned location 4, since the mat4 occupies locations 0, 1, 2, 3 for
    /// its 4 column vectors.
    #[must_use]
    pub fn vertex<It>(&self, attribute_location_iterator: It) -> VertexStreamBuilder<It>
    where
        It: Iterator<Item = u32>,
    {
        VertexStreamBuilder {
            used_attribute_locations: Vec::new(),
            attribute_location_iterator: LocationIter::new(attribute_location_iterator),
        }
    }

    /// create a fragment outputs builder that lets you add color attachments to
    /// the currently recorded shader and write to them
    #[must_use]
    pub fn pixel<It>(&self, output_location_iterator: It) -> FragmentOutputsBuilder<It>
    where
        It: Iterator<Item = u32>,
    {
        FragmentOutputsBuilder {
            output_location_iterator,
        }
    }

    /// not yet supported!
    /// individual uniforms for oldschool OpenGL usage
    pub fn uniform<T: AsTen>(&self, _location: u32) -> Ten<T::S, T::D> { unimplemented!() }
}

/// Group builder that lets you add bindings to a bindgroup
pub struct Group<It: Iterator<Item = u32> = std::ops::RangeFrom<u32>> {
    group_index: u32,
    bindings: Vec<(shame_graph::Loc, shame_graph::Binding)>,
    binding_index_iterator: It,
}

impl<It: Iterator<Item = u32>> Drop for Group<It> {
    fn drop(&mut self) {
        Context::with(|ctx| {
            let mut shader = ctx.shader_mut();
            let bindings = std::mem::take(&mut self.bindings);
            shader
                .side_effects
                .push_bind_group(self.group_index, shame_graph::BindGroup::new(bindings));
        })
    }
}

/// add vertex attribute inputs to the currently recorded shader
pub struct VertexStreamBuilder<It: Iterator<Item = u32>> {
    #[allow(unused)]
    used_attribute_locations: Vec<u32>, //for detection of overlap errors //TODO: check that overlaps are still detected despite this not being used
    attribute_location_iterator: LocationIter<It>,
}

/// add fragment outputs (color attachments) inputs to the currently recorded
/// shader
pub struct FragmentOutputsBuilder<It: Iterator<Item = u32>> {
    output_location_iterator: It,
}

fn next_index(it: &mut impl Iterator<Item = u32>, index_name: &str) -> u32 {
    match it.next() {
        Some(index) => index,
        None => panic!("provided {} iterator ran out of indices", index_name), //TODO: this should not be a panic but rather whatever the user selected as their error behavior (Result or Panic)
    }
}

impl<It: Iterator<Item = u32>> Group<It> {
    fn new(group_index: u32, binding_index_iterator: It) -> Self
    where
        It: Iterator<Item = u32>,
    {
        Group {
            group_index,
            bindings: Vec::new(),
            binding_index_iterator,
        }
    }

    /// add a read-only storage buffer binding to this group
    pub fn storage<T: Fields>(&mut self) -> T {
        use shame_graph::*;
        let (t, block) = T::new_as_interface_block(Access::Const);
        let index = self.next_index();
        self.bindings.push((index, Binding::Storage(block)));
        t
    }

    /// add a mutable storage buffer binding to this group
    ///
    /// mutable storage buffers allow for all kinds of memory race conditions,
    /// until we figure out how to expose them through safe highlevel sync
    /// primitives, using them will remain unsafe
    pub fn storage_mut<T: Fields>(&mut self) -> UnsafeAccess<T> {
        use shame_graph::*;
        let (t, block) = T::new_as_interface_block(Access::CopyOnWrite); //TODO: double check if copy on write access is still correct here
        let index = self.next_index();
        self.bindings.push((index, Binding::StorageMut(block)));
        UnsafeAccess::new(t)
    }

    fn next_index(&mut self) -> u32 { next_index(&mut self.binding_index_iterator, "binding index") }

    /// add a uniform block binding to this group
    pub fn uniform_block<T: Fields>(&mut self) -> T {
        use shame_graph::*;
        let (t, block) = T::new_as_interface_block(Access::Const);
        let index = self.next_index();
        self.bindings.push((index, Binding::UniformBlock(block)));
        t
    }

    /// add a texture-combined-sampler binding to this group
    pub fn combine_sampler<Out: TexSampleType, In: TexCoordType>(&mut self) -> CombineSampler<Out, In> {
        let tcsampler = CombineSampler::new();
        let opaque_ty = CombineSampler::<Out, In>::opaque_ty();
        let index = self.next_index();
        self.bindings
            .push((index, shame_graph::Binding::Opaque(opaque_ty, tcsampler.any())));
        tcsampler
    }

    /// add a sampler binding to this group
    pub fn sampler(&mut self) -> Sampler {
        use shame_graph::*;
        let sampler = Sampler::new();
        let index = self.next_index();
        self.bindings
            .push((index, Binding::Opaque(Sampler::opaque_ty(), sampler.any())));
        sampler
    }

    /// add a texture binding to this group
    pub fn texture<Out: TexSampleType, In: TexCoordType>(&mut self) -> Texture<Out, In> {
        use shame_graph::*;
        let texture = Texture::new();
        let index = self.next_index();
        self.bindings
            .push((index, Binding::Opaque(Texture::<Out, In>::opaque_ty(), texture.any())));
        texture
    }
}

impl<It: Iterator<Item = u32>> VertexStreamBuilder<It> {
    /// add `T`'s fields as individual attributes to the vertex inputs of the
    /// current shader recording.
    pub fn attributes<T: Fields>(&mut self) -> T { self.attributes_detailed().0 }

    /// returns an empty vector when called from a non-vertex stage
    pub fn attributes_detailed<T: Fields>(&mut self) -> (T, Vec<(shame_graph::Tensor, Range<u32>)>) {
        use shame_graph::*;
        let stage = crate::Stage::Vertex;
        let mut loc_iter = &mut self.attribute_location_iterator;

        let mut attribute_for_tensor = |ctx: &Context, ten: Tensor, name: &str| -> Option<(Any, Tensor, Range<u32>)> {
            let mut shader = ctx.shader_mut();
            match &mut shader.stage_interface {
                StageInterface::Vertex { inputs, .. } => {
                    let (any, range) =
                        inputs.push_vertex_attribute_with_location_iter(&mut loc_iter, ten, Some(name.to_string()));
                    Some((any, ten, range))
                }
                _ => None,
            }
        };

        let non_tensor_error = |ctx: &Context, ty, name| -> Any {
            let fields_of = T::parent_type_name().map(|p| format!(" of `{p}`")).unwrap_or_default();
            let err_text = format!("cannot interpret field `{}: {}`{} as a vertex attribute, only tensor types `Ten<S, D>` allowed (e.g. `floatN`, `floatMxN`)", name, &ty, fields_of);
            ctx.push_error(Error::TypeError(err_text));
            Any::not_available()
        };

        let mut ranges = Vec::new();

        let t = Context::with(|ctx| {
            T::from_fields_downcast(None, &mut |ty, name| -> (Any, crate::Stage) {
                let any = match ty.kind {
                    TyKind::Tensor(ten) => match attribute_for_tensor(ctx, ten, name) {
                        Some((any, ten, range)) => {
                            //we're in a vertex stage recording
                            ranges.push((ten, range));
                            any
                        }
                        None => Any::not_available(),
                    },
                    _ => non_tensor_error(ctx, ty, name),
                };
                (any, stage)
            })
        });
        (t, ranges)
    }
}

impl<It: Iterator<Item = u32>> FragmentOutputsBuilder<It> {
    fn next_location(&mut self) -> u32 {
        next_index(&mut self.output_location_iterator, "rendertarget output location")
    }

    /// add a color target to the currently recorded shader
    pub fn color<Ten: AsTen>(&mut self) -> WriteOnly<Ten::S, Ten::D>
    where
        Ten::S: IsShapeScalarOrVec,
    {
        self.color_with_ident::<Ten>(Some("color_out".to_string()))
    }

    /// add a color target to the currently recorded shader.
    /// The color target will be called `ident` in the generated shader code
    /// if possible.
    pub fn color_with_ident<Ten: AsTen>(&mut self, ident: Option<String>) -> WriteOnly<Ten::S, Ten::D>
    where
        Ten::S: IsShapeScalarOrVec,
    {
        //TODO: replace with better writeonly type
        let location = self.next_location();

        Context::with(|ctx| {
            let mut shader = ctx.shader_mut();
            let tensor = shame_graph::Tensor::new(Ten::S::SHAPE, Ten::D::DTYPE);

            let any = match &mut shader.stage_interface {
                shame_graph::StageInterface::Fragment { outputs, .. } => {
                    outputs.push_color_attachment(location, tensor, ident)
                }
                _ => Any::not_available(),
            };

            WriteOnly::<Ten::S, Ten::D>::new(any, Stage::Fragment)
        })
    }
}

/// access to rasterizer functionality
pub struct Raster<'a> {
    needs_to_be_used: bool,
    phantom: PhantomData<&'a ()>,
}

impl<'a> Raster<'a> {
    pub(crate) fn new() -> Self {
        use shame_graph::ShaderKind::*;
        match crate::current_shader() {
            Vertex => Self {
                needs_to_be_used: true,
                phantom: PhantomData,
            },
            Fragment => Self {
                needs_to_be_used: false,
                phantom: PhantomData,
            },
            Compute => panic!("raster object constructed in compute shader"),
        }
    }

    /// rasterize primitives at the provided clip_space positions.
    /// The clip space positions are combined to primitives according to the
    /// primitive topology specified elsewhere in your application.
    pub fn rasterize(mut self, clip_space_position: float4) -> Primitive<'a> {
        assert::assert_string(
            !Context::with(|ctx| ctx.inside_branch().is_some()),
            "rasterize cannot be called from within conditional blocks such as if-then/if-then-else/for/while.",
        );
        Any::v_position().assign(clip_space_position.as_any());
        self.needs_to_be_used = false;
        Primitive { _phantom: PhantomData }
    }
}

impl Drop for Raster<'_> {
    fn drop(&mut self) {
        assert::assert_string(
            !self.needs_to_be_used,
            "rasterizer was dropped before being used. A vertex stage recording must call the rasterizer via `features.raster.rasterize(...)`."
        );
    }
}

/// functions for setting up the workgroup size and gaining access to invocation
/// ids
pub struct WorkGroupSetup<'a> {
    pub(crate) _phantom: PhantomData<&'a ()>,
}

impl WorkGroupSetup<'_> {
    /// sets the `local_size_*` and provides access to invocation ids/sizes.
    ///
    /// produces the following glsl:
    /// ```text
    /// layout(local_size_x = X​, local_size_y = Y​, local_size_z = Z​) in;
    /// ```
    /// where `[X, Y, Z] = workgroup_size`
    ///
    pub fn work_group(self, workgroup_size: [usize; 3]) -> Ids {
        Context::with(|ctx| {
            let mut shader = ctx.shader_mut();

            match &mut shader.stage_interface {
                shame_graph::StageInterface::Compute { workgroup_size: w } => {
                    *w = Some(workgroup_size);
                }
                _ => unreachable!(),
            };

            Ids { _phantom: PhantomData }
        })
    }
}

/// access to shader invocation related ids and sizes
pub struct Ids {
    pub(crate) _phantom: PhantomData<()>,
}

impl Ids {
    /// glsl: `gl_NumWorkGroups`
    pub fn total_work_groups_dispatched(&self) -> uint3 { Any::c_num_work_groups().downcast(Stage::Uniform) }

    /// glsl: `gl_WorkGroupID`
    pub fn work_group3(&self) -> uint3 { Any::c_work_group_id().downcast(Stage::Uniform) }

    /// glsl: `gl_LocalInvocationID`
    pub fn local3(&self) -> uint3 { Any::c_local_invocation_id().downcast(Stage::Uniform) }

    /// glsl: `gl_LocalInvocationIndex`
    /// ```text
    ///   gl_LocalInvocationIndex =
    ///       gl_LocalInvocationID.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y +
    ///       gl_LocalInvocationID.y * gl_WorkGroupSize.x +
    ///       gl_LocalInvocationID.x;
    /// ```
    pub fn local(&self) -> uint { Any::c_local_invocation_index().downcast(Stage::Uniform) }

    /// glsl: `gl_GlobalInvocationID`
    ///
    /// `gl_GlobalInvocationID = gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID`
    pub fn global3(&self) -> uint3 { Any::c_global_invocation_id().downcast(Stage::Uniform) }

    /// glsl: `gl_WorkGroupSize`
    pub fn work_group_size(&self) -> uint3 { Any::c_work_group_size().downcast(Stage::Uniform) }
}
