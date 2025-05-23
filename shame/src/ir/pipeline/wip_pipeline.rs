use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    fmt::{Debug, Display},
    ops::Range,
    rc::Rc,
};

use thiserror::Error;

use super::{PossibleStages, ShaderStage, StageMask};
use crate::{
    any::layout::Repr,
    call_info,
    common::{
        integer::post_inc_usize,
        po2::U32PowerOf2,
        pool::{Key, PoolRef},
    },
    frontend::{
        any::{
            render_io::{Attrib, ColorTarget, Location, VertexBufferLayoutRecorded},
            shared_io::{BindPath, BindingType},
        },
        encoding::{
            features::{DrawContext, Indexing},
            fragment_test::{DepthTest, StencilState, StencilTest},
            mask::BitVec64,
            pipeline_info::Dict,
            rasterizer::{Draw, FragmentQuad, FragmentStage, PrimitiveAssembly},
            EncodingError, EncodingErrorKind, EncodingGuard,
        },
        error::InternalError,
        rust_types::{
            len::x3,
            type_layout::{self, layoutable, StructLayout},
        },
    },
    ir::{
        self,
        expr::{BuiltinShaderIo, Interpolator, ShaderIo},
        recording::{BuiltinTemplateStructs, CallInfo, Context, NodeRecordingError},
        FragmentShadingRate, Node, SizedField, SizedStruct, SizedType, StoreType, StructureDefinitionError,
        StructureFieldNamesMustBeUnique, TextureFormatWrapper, Type,
    },
    results::DepthStencilState,
    BindingIter, DepthLhs, StencilMasking, Test, TypeLayout,
};


/// like `std::stringify!(x)` but it won't compile if `x` is not a valid expression or type
#[doc(hidden)]
#[macro_export]
macro_rules! stringify_checked {
    (ty: $t: ty) => {
        {
            if false {
                let _: $t = unreachable!(); // make sure the type is valid
            }
            std::stringify!($t)
        }
    };
    (expr: $ex: expr) => {
        {
            if false {
                let _ = $ex; // make sure the expression is valid
            }
            std::stringify!($ex)
        }
    };
}

#[derive(Error, Debug, Clone)]
pub enum PipelineError {
    #[error("Missing pipeline specialization. Use either the `{}` or `{}` method to start a compute- or render-pipeline encoding.",
        stringify_checked!(expr: EncodingGuard::new_compute_pipeline::<3>).replace("3", "_"),
        stringify_checked!(expr: EncodingGuard::new_render_pipeline),
    )]
    MissingSpecialization,
    #[error("render-pipelines must use the rasterizer.\
        \nCall the `.vertices.assemble(...)` method on your `{}` to specify what kind of primitives you want to draw.\
        \nOn the returned `{}` call one of the `.rasterize*` methods to use the rasterizer and obtain the `{}` object.",
        stringify_checked!(ty: DrawContext),
        stringify_checked!(ty: PrimitiveAssembly<false>).replace("false", "_"),
        stringify_checked!(ty: FragmentStage),
    )]
    UnusedRasterizer,
    #[error("Invalid vertex input type: {0:?}. vertex inputs must be of scalar or vector type.")]
    InvalidAttributeType(Type),
    #[error("no vertex attribute found with location {0}")]
    AttribLocationNotFound(Location),
    #[error("duplicate vertex attribute location {location} in vertex buffers #{buffer_a} and #{buffer_b}")]
    DuplicateAttribLocation {
        location: Location,
        buffer_a: u32,
        buffer_b: u32,
    },
    #[error("trying to import vertex buffer #{0} twice. previous import at {1}")]
    DuplicateVertexBufferImport(u32, CallInfo),
    #[error("vertex buffer was created while no pipeline encoding was active")]
    VertexBufferCreatedOutsideOfActiveEncoding,
    #[error("trying to access color target #{0} twice")]
    DuplicateColorTargetAccess(u32),
    #[error("Invalid interpolator type: {0:?}. primitive fragments can only be filled with scalar or vector types.")]
    InvalidInterpolatorType(Type),
    #[error("no vertex output interpolator found with location {0}")]
    InterpolatorLocationNotFound(Location),
    #[error("Unable to import binding {} in bindgroup {} multiple times. This binding was already imported as a `{}` before.", .0 .0, .0 .1, .1)]
    DuplicateBindPath(BindPath, StoreType),
    #[error("{0}")]
    InsufficientVisibility(InsufficientVisibilityError),
    #[error("duplicate vertex output interpolator location {0}")]
    DuplicateInterpolatorLocation(Location),
    #[error("no color target found at index {0}")]
    ColorTargetNotFoundAtSlot(u32),
    #[error("the texture format {0:?} does not support blending")]
    FormatDoesNotSupportBlending(TextureFormatWrapper),
    #[error("only the color target 0 supports alpha to coverage. It was called on color target {0}.")]
    AlphaToCoverageUnsupportedForSlot(u32),
    #[error(
        "alpha to coverage functionality requires a color target with 4 color channels (i.e. it must have an alpha channel). Target format `{0}` has {1} color channels."
    )]
    AlphaToCoverageRequires4Channels(String, u32),
    #[error("single sample rasterization requires a sample mask with a single sample, not {0} samples")]
    SingleSamplingSampleMaskMustHave1Sample(usize),
    #[error(
        "multisample/supersample rasterization requires a sampling mask with at least 2 samples. Only {0} sample(s) was/were requested"
    )]
    TooFewSamplesInMultisampleMask(usize),
    #[error(
        "A nonzero depth bias requires triangle topology. For more information on this restriction wrt. different APIs see https://github.com/gpuweb/gpuweb/issues/4729"
    )]
    NonZeroDepthBiasRequiresTriangles,
}

#[derive(Error, Debug, Clone)]
pub struct InsufficientVisibilityError {
    pub path: BindPath,
    pub user_defined_visibility: StageMask,
    pub required_visibility: StageMask,
    pub vertex_writable_storage_by_default_enabled: bool,
}

impl Display for InsufficientVisibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Insufficient shader stage visibility of binding.")?;
        writeln!(
            f,
            "{} is only visible in the stages:\n{}, ",
            self.path,
            self.user_defined_visibility.to_string_verbose()
        )?;
        writeln!(
            f,
            "but the binding is used in the stages:\n{}.",
            self.required_visibility.to_string_verbose()
        )?;

        let vert_visible = self.user_defined_visibility.contains_stage(ShaderStage::Vert);
        let vert_access = self.required_visibility.contains_stage(ShaderStage::Vert);
        use crate::Settings;
        let Settings {
            vertex_writable_storage_by_default, // just to grab the identifier of this field for a correct error message
            ..
        } = Settings::default();

        if !vert_visible && vert_access && !self.vertex_writable_storage_by_default_enabled {
            writeln!(
                f,
                "(for writeable storage bindings, vertex stage visibility is not enabled by default, because \
            some devices don't support it. However this default can be changed by setting `{}::{} = true` \
            at the beginning of the pipeline encoding. Ideally this is coupled with the corresponding feature in the Graphics API)",
                stringify_checked!(ty: Settings),
                stringify_checked!(expr: vertex_writable_storage_by_default)
            )?;
        }
        type T = crate::Buffer<crate::f32x1>; // dummy type for error message
        write!(
            f,
            "For custom visibility per binding, use the `{}` function",
            stringify_checked!(expr: BindingIter::next_with_visibility::<T>)
        )?;
        Ok(())
    }
}

impl WipPipeline {
    /// whether this pipeline has a fragment stage that, when run, can produce
    /// observable effects, such as changes to storage textures/buffers or color/depth targets
    pub fn may_have_fragment_effects(&self) -> bool {
        match self.kind {
            PipelineKind::Render => (),
            PipelineKind::Compute => return false,
        };

        for binding in self.layout.borrow().bindings.values() {
            if binding.binding_ty.can_produce_side_effects() &&
                binding.user_defined_visibility.contains_stage(ShaderStage::Frag)
            {
                return true;
            }
        }

        let rp = self.special.render.borrow();
        if !rp.color_targets.is_empty() {
            return true;
        }
        if rp.depth_stencil.get().is_some() {
            return true;
        }
        false
    }

    pub fn register_shader_io(&self, io: &ShaderIo, node: Key<Node>, ctx: &Context) {
        let stages_within_pipeline = StageMask::pipeline(self.kind) & io.possible_stages().can_appear_in();

        if stages_within_pipeline.is_empty() {
            ctx.push_error(NodeRecordingError::WrongPipelineKindForShaderIo(io.clone(), self.kind).into());
            return;
        }
        match io {
            ShaderIo::Builtin(builtin) => {
                let mut builtins = self.special.builtin_io.borrow_mut();
                let previous = builtins.insert(*builtin, (node, ctx.latest_user_caller()));
                if let Some((node, caller)) = previous {
                    ctx.push_error(NodeRecordingError::SameShaderIoSetMultipleTimes(io.clone(), caller).into());
                }
            }
            ShaderIo::Interpolate(_) => (),     // registered at Any::fill_primitive
            ShaderIo::GetInterpolated(_) => (), // registered at Any::fill_primitive
            ShaderIo::WriteToColorTarget { .. } => (), // registered at Any::color_target_write
            ShaderIo::GetVertexInput { .. } => (), // registered at Any::vertex_buffer
        }
    }
}

#[derive(Debug)]
pub struct WipPipeline {
    pub label: Option<String>,
    pub kind: PipelineKind,
    pub layout: RefCell<WipPipelineLayoutDescriptor>,
    pub builtin_template_structs: RefCell<BuiltinTemplateStructs>,
    pub special: WipSpecialization,
}

impl WipPipeline {
    pub(crate) fn new(kind: PipelineKind) -> Self {
        Self {
            label: None,
            kind,
            layout: Default::default(),
            builtin_template_structs: Default::default(),
            special: WipSpecialization {
                builtin_io: Default::default(),
                render: Default::default(),
                compute: Default::default(),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct WipBinding {
    pub call_info: CallInfo,
    /// the visibility set by the user for this binding (or the default, if the user chose to not specify it)
    pub user_defined_visibility: StageMask,
    pub binding_ty: BindingType,
    pub shader_ty: StoreType,
    pub node: Key<Node>,
}

#[derive(Debug, Clone)]
pub struct WipPushConstantsField {
    pub call_info: CallInfo,
    pub custom_min_size: Option<u64>,
    pub custom_min_align: Option<U32PowerOf2>,
    pub ty: SizedType,
    pub node: Key<Node>,
}

#[derive(Default, Debug, Clone)]
pub struct ByteRangesPerStage {
    pub comp: Option<Range<u32>>,
    pub task: Option<Range<u32>>,
    pub mesh: Option<Range<u32>>,
    pub vert: Option<Range<u32>>,
    pub frag: Option<Range<u32>>,
}

impl ByteRangesPerStage {
    fn range_for_stage_mut(&mut self, stage: ShaderStage) -> &mut Option<Range<u32>> {
        match stage {
            ShaderStage::Comp => &mut self.comp,
            ShaderStage::Task => &mut self.task,
            ShaderStage::Mesh => &mut self.mesh,
            ShaderStage::Vert => &mut self.vert,
            ShaderStage::Frag => &mut self.frag,
        }
    }

    fn range_for_stage(&self, stage: ShaderStage) -> &Option<Range<u32>> {
        match stage {
            ShaderStage::Comp => &self.comp,
            ShaderStage::Task => &self.task,
            ShaderStage::Mesh => &self.mesh,
            ShaderStage::Vert => &self.vert,
            ShaderStage::Frag => &self.frag,
        }
    }

    fn stages_with_some(&self) -> StageMask {
        ShaderStage::all()
            .into_iter()
            .filter(|stage| self.range_for_stage(*stage).is_some())
            .fold(StageMask::empty(), |mask, stage| mask | stage.into())
    }
}

impl WipPushConstantsField {
    pub(crate) fn new(
        ty: SizedType,
        custom_min_size: Option<u64>,
        custom_min_align: Option<U32PowerOf2>,
        call_info: CallInfo,
        node: Key<Node>,
    ) -> Self {
        Self {
            custom_min_size,
            custom_min_align,
            ty,
            call_info,
            node,
        }
    }

    #[rustfmt::skip]
    fn extend_range(src_range: &mut Option<Range<u32>>, extend_by: Range<u32>) {
        let range = src_range.get_or_insert(extend_by.clone());
        range.start = range.start.min(extend_by.start);
        range.end   = range.end  .max(extend_by.end  );
    }

    /// returns a pair of `Ok((extract_ranges_per_stage, byte_size))` where
    /// `byte_size` is the size of the struct of all push constant fields.
    pub(crate) fn extract_ranges_per_stage(
        allowed_stages: StageMask,
        fields: &[WipPushConstantsField],
        nodes: &PoolRef<Node>,
    ) -> Result<(ByteRangesPerStage, u64), InternalError> {
        let sized_struct = match fields {
            [] => return Ok((ByteRangesPerStage::default(), 0)),
            [first @ .., last] => Self::fields_as_sized_struct_nonempty(first, last),
        }?;

        let byte_size = sized_struct.byte_size();

        // TODO(release) the `.expect()` calls here can be removed by building a `std::alloc::Layout`-like builder for struct layouts.
        let sized_struct: layoutable::SizedStruct = sized_struct
            .try_into()
            .expect("push constants are NoBools and NoHandles");
        let layout = TypeLayout::new_layout_for(&sized_struct.into(), Repr::Storage);
        let layout = match &layout.kind {
            type_layout::TypeLayoutSemantics::Structure(layout) => &**layout,
            _ => unreachable!("expected struct layout for type layout of struct"),
        };

        let mut ranges = ByteRangesPerStage::default();

        for (field, node) in layout.fields.iter().zip(fields.iter().map(|f| f.node)) {
            let stages = nodes[node].stages.must_appear_in();
            let field_size = field
                .field
                .ty
                .byte_size()
                .expect("SizedStruct type enforces Some(size)");
            let start = field.rel_byte_offset;
            let end = start + field_size;

            let start = u32::try_from(start)
                .map_err(|e| InternalError::new(true, format!("push constant field start {start} out of u32 range")))?;
            let end = u32::try_from(end)
                .map_err(|e| InternalError::new(true, format!("push constant field start {end} out of u32 range")))?;

            for stage in stages {
                Self::extend_range(ranges.range_for_stage_mut(stage), start..end);
            }
        }

        let disallowed_stages = StageMask::all() & !allowed_stages;
        let occured_stages = ranges.stages_with_some();
        if (occured_stages & disallowed_stages) != StageMask::empty() {
            return Err(InternalError::new(
                true,
                format!(
                    "disallowed stage in inferred `push constant` ranges for pipeline.\nallowed stages: {allowed_stages},\noccured stages: {occured_stages}"
                ),
            ));
        }

        Ok((ranges, byte_size))
    }

    fn fields_as_sized_struct_nonempty(
        fields: &[WipPushConstantsField],
        last: &WipPushConstantsField,
    ) -> Result<SizedStruct, InternalError> {
        // TODO(release) we only do this because we need the layout info, which could be gotten with zero allocations and
        // without string uniqueness checks instead.
        // Create a layout builder similar to `std::alloc::Layout` to replace this instead.

        let mut i = 0;
        let mut to_sized_field = |f: &WipPushConstantsField| SizedField {
            name: format!("pushc{}", post_inc_usize(&mut i)).into(),
            custom_min_size: f.custom_min_size,
            custom_min_align: f.custom_min_align,
            ty: f.ty.clone(),
        };

        // here we have to allocate unique name strings for each field,
        // so we don't fail the name uniqueness check, even though we don't need those names.
        SizedStruct::new_nonempty("PushConstants".into(),
            fields.iter().map(&mut to_sized_field).collect(),
            to_sized_field(last)
        ).map_err(|err| match err {
            StructureFieldNamesMustBeUnique => {
                InternalError::new(true, format!("intermediate push constants structure field names are not unique. fields: {fields:?}, last: {last:?}"))
            }
        })
    }
}

#[derive(Default, Debug, Clone)]
pub struct WipPipelineLayoutDescriptor {
    pub(crate) bindings: BTreeMap<BindPath, WipBinding>,
    /// the fields of the push constants #[derive(GpuLayout)] rust-struct, in order.
    ///
    /// single field for non-struct types.
    pub(crate) push_constants: Vec<WipPushConstantsField>,
}

#[derive(Default, Debug)]
pub struct WipSpecialization {
    pub(crate) builtin_io: RefCell<BTreeMap<BuiltinShaderIo, (Key<Node>, CallInfo)>>,
    pub(crate) render: RefCell<WipRenderPipelineDescriptor>,
    pub(crate) compute: RefCell<WipComputePipelineDescriptor>,
}

#[derive(Default, Debug)]
pub struct WipRenderPipelineDescriptor {
    pub(crate) vertex_buffers: Vec<RecordedWithIndex<VertexBufferLayoutRecorded>>,
    // TODO(release) test if color target multisampling works decoupled from depth/stencil buffer multisampling https://registry.khronos.org/vulkan/specs/1.2-extensions/html/chap8.html#VUID-VkRenderingInfo-imageView-06858
    pub(crate) color_targets: Vec<RecordedWithIndex<ColorTarget>>,
    pub(crate) depth_stencil: LateRecorded<WipDepthStencilState>,
    pub(crate) interpolators: Vec<(Interpolator, CallInfo)>,
    pub(crate) vertex_id_order: LateRecorded<Indexing>,
    pub(crate) sample_mask: LateRecorded<BitVec64>,
    pub(crate) fragment_quad: LateRecorded<FragmentQuad>,
    pub(crate) fragment_shading_rate: LateRecorded<FragmentShadingRate>,
    /// glsl `invariant`, wgsl `invariant`, hlsl `precise`
    /// (precise provides stronger guarantees than invariant in both hlsl and glsl and can therefore also be used to implement this)
    pub(crate) deterministic_clip_pos: LateRecorded<bool>,
    pub(crate) color_target0_alpha_to_coverage: LateRecorded<bool>,
    pub(crate) draw: LateRecorded<Draw>,
}

#[derive(Debug)]
pub struct WipDepthStencilState {
    pub format: TextureFormatWrapper,
    pub depth_test: Option<DepthTest>,
    pub stencil_test: Option<StencilTest>,
}

#[derive(Default, Debug)]
pub struct WipComputePipelineDescriptor {
    pub(crate) thread_grid_size_within_workgroup: LateRecorded<[u32; 3]>,
    pub(crate) expected_threads_per_wave: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PipelineKind {
    Render,
    Compute,
    //Mesh,
}

impl PipelineKind {
    pub fn write_requires_pipeline_kind_error(current: PipelineKind, required: PipelineKind) -> String {
        match required {
            PipelineKind::Render => format!(
                "this operation is only accessible in render pipelines. The current `Encoding` was specialized as a {current}."
            ),
            PipelineKind::Compute => format!(
                "this operation is only accessible in compute pipelines. The current `Encoding` was specialized as a {current}."
            ),
        }
    }

    pub fn stages(self) -> &'static [ShaderStage] {
        use ShaderStage::*;
        match self {
            PipelineKind::Render => &[Vert, Frag],
            PipelineKind::Compute => &[Comp],
        }
    }
}

impl Display for PipelineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            PipelineKind::Render => "render pipeline",
            PipelineKind::Compute => "compute pipeline",
        })
    }
}

/// a required value that is recorded during the encoding. A glorified Option<T> which also
/// stores the code location where the information came from.
#[derive(Debug, Clone, Copy)]
pub struct LateRecorded<T> {
    value: Option<(T, CallInfo)>,
}

impl<T> Default for LateRecorded<T> {
    fn default() -> Self {
        Self {
            value: Default::default(),
        }
    }
}

impl<T: Debug> LateRecorded<T> {
    #[track_caller]
    pub(crate) fn set(&mut self, value: T) {
        self.value = Context::try_with(call_info!(), |ctx| (value, ctx.latest_user_caller()));
    }

    pub(crate) fn set_once(&mut self, value: T) {
        let is_none = self.value.is_none();
        self.value = Context::try_with(call_info!(), |ctx| {
            if !is_none {
                ctx.push_error(
                    InternalError::new(
                        true,
                        format!(
                            "set_once called multiple times. most recently with value of {:?}",
                            value
                        ),
                    )
                    .into(),
                );
            }
            (value, ctx.latest_user_caller())
        });
    }

    pub(crate) fn get_value(self) -> Option<T> { self.value.map(|t| t.0) }

    pub(crate) fn get(&self) -> Option<&(T, CallInfo)> { self.value.as_ref() }

    #[track_caller]
    pub(crate) fn try_get(
        self,
        info_name: &str,
        assemble_error: &impl Fn(CallInfo, EncodingErrorKind) -> EncodingError,
    ) -> Result<(T, CallInfo), EncodingError> {
        match self.value {
            Some((t, call_info)) => Ok((t, call_info)),
            None => {
                let str = format!("information tagged `{info_name}` was missed by the recording mechanism");
                Err(assemble_error(call_info!(), InternalError::new(true, str).into()))
            }
        }
    }
}

impl WipDepthStencilState {
    pub fn finalize(&self, call_info: CallInfo) -> Result<DepthStencilState, EncodingError> {
        let depth_write_enabled;
        let depth_compare;
        let bias;
        match self.depth_test {
            Some(d) => {
                depth_write_enabled = d.replace_on_pass;
                depth_compare = d.test;
                bias = match d.operand {
                    DepthLhs::Explicit(_, bias) => bias,
                    DepthLhs::FragmentZ(bias) => bias,
                }
            }
            None => {
                depth_write_enabled = false;
                depth_compare = Test::Never;
                bias = crate::DepthBias::default();
            }
        }

        let noop = crate::StencilBranch::Test {
            test: Test::Never,
            on_pass: crate::StencilOp::Keep,
            on_fail: crate::StencilOp::Keep,
        };

        let (masking, cw, ccw) = match self.stencil_test {
            None => (
                StencilMasking::Masked(0x00), /*wgpu expects 0 mask for disabled stencil test*/
                noop,
                noop,
            ),
            Some(s) => match s {
                StencilTest::PerWinding { masking, ccw, cw } => (masking, ccw, cw),
                StencilTest::Single(winding, masking, branch) => match winding {
                    crate::Winding::Ccw => (masking, branch, noop),
                    crate::Winding::Cw => (masking, noop, branch),
                    crate::Winding::Either => (masking, branch, branch),
                },
            },
        };
        let (rw_mask, w_mask) = match masking {
            StencilMasking::Unmasked => (!0, !0),
            StencilMasking::Masked(mask) => (mask, mask),
            StencilMasking::PerAccess { read_write, write } => (read_write, write),
        };

        let to_stencil_face = |branch: crate::StencilBranch| {
            let (test, on_pass_depth_pass, on_pass_depth_fail, on_fail) = match branch {
                crate::StencilBranch::Test { test, on_pass, on_fail } => (test, on_pass, on_pass, on_fail),
                crate::StencilBranch::TestConsiderDepth {
                    test,
                    on_pass_depth_pass,
                    on_pass_depth_fail,
                    on_fail,
                } => (test, on_pass_depth_pass, on_pass_depth_fail, on_fail),
            };
            crate::StencilFace {
                compare: test,
                on_pass_depth_fail,
                on_fail,
                on_pass_depth_pass,
            }
        };

        Ok(DepthStencilState {
            format: self.format.clone(),
            depth_write_enabled,
            depth_compare,
            bias,
            stencil: StencilState {
                ccw: to_stencil_face(ccw),
                cw: to_stencil_face(cw),
                rw_mask: rw_mask as u32,
                w_mask: w_mask as u32,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RecordedWithIndex<T> {
    pub call_info: CallInfo,
    pub index: u32,
    t: T,
}

impl<T> RecordedWithIndex<T> {
    pub fn into_inner(self) -> T { self.t }
}

impl<T> RecordedWithIndex<T> {
    pub fn new(value: T, index: u32, call_info: CallInfo) -> Self {
        Self {
            call_info,
            index,
            t: value,
        }
    }
}

impl<T> std::ops::Deref for RecordedWithIndex<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target { &self.t }
}

impl<T> std::ops::DerefMut for RecordedWithIndex<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.t }
}

impl<T: PartialEq> PartialEq<RecordedWithIndex<T>> for RecordedWithIndex<T> {
    fn eq(&self, other: &RecordedWithIndex<T>) -> bool { self.index == other.index && self.t == other.t }
}

impl<T: Eq> Eq for RecordedWithIndex<T> {}

impl<T: std::hash::Hash> std::hash::Hash for RecordedWithIndex<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.t.hash(state);
    }
}

impl WipRenderPipelineDescriptor {
    pub fn find_vertex_attrib(&self, location: Location) -> Result<&Attrib, PipelineError> {
        self.vertex_buffers
            .iter()
            .find_map(|vb| vb.attribs.iter().find(|attrib| attrib.location == location))
            .map(|attrib| &attrib.t)
            .ok_or(PipelineError::AttribLocationNotFound(location))
    }

    pub fn find_interpolator(&self, location: Location) -> Result<&Interpolator, PipelineError> {
        self.interpolators
            .iter()
            .find_map(|(t, _)| (t.location == location).then_some(t))
            .ok_or(PipelineError::InterpolatorLocationNotFound(location))
    }

    pub fn find_color_target(&self, slot: u32) -> Result<&ColorTarget, PipelineError> {
        self.color_targets
            .iter()
            .find_map(|c| (c.index == slot).then_some(&**c))
            .ok_or(PipelineError::ColorTargetNotFoundAtSlot(slot))
    }

    pub fn is_supersampling(&self) -> Option<(bool, CallInfo)> {
        let mask = self.sample_mask.get()?.0;
        let (shading_rate, call_info) = self.fragment_shading_rate.get()?;
        Some((
            match (shading_rate, mask.len()) {
                (FragmentShadingRate::PerSample, 2..) => true,
                _ => false,
            },
            *call_info,
        ))
    }
}
