use std::{collections::BTreeMap, num::IntErrorKind, ops::Range, sync::Arc};

use crate::{
    backend::shader_code::ShaderCode,
    common::small_vec::SmallVec,
    frontend::any::{
        render_io::{ColorTarget, VertexBufferLayoutRecorded},
        shared_io::BindingType,
    },
    ir::{
        pipeline::{ShaderStage, StageMask},
        StoreType, TextureFormatWrapper,
    },
    results::LanguageCode,
    Winding,
};

use super::{features::Indexing, fragment_test::DepthStencilState, mask::BitVec64, rasterizer::Draw, IsPipelineKind};

/// shaders and pipeline info for creating a render pipeline using a graphics api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderPipeline {
    /// debug label of the render pipeline
    pub label: Option<String>,
    /// vertex and fragment shader, as well as source spans
    pub shaders: RenderPipelineShaders,
    /// additional info required to initialize a render pipeline
    pub pipeline: RenderPipelineInfo,
}

/// shader and pipeline info for creating a compute pipeline using a graphics api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputePipeline {
    /// debug label of the compute pipeline
    pub label: Option<String>,
    /// compute shader, as well as source spans
    pub shader: ComputeShader,
    /// additional info required to initialize a compute pipeline
    pub pipeline: ComputePipelineInfo,
}

pub(crate) enum PipelineDefinition {
    Render(RenderPipeline),
    Compute(ComputePipeline),
}

pub type Dict<K, V> = BTreeMap<K, V>;

/// info required to initialize a render pipeline in addition to shaders
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct RenderPipelineInfo {
    /// vertex buffer memory layouts
    pub vertex_buffers: Dict<u32, VertexBufferLayoutRecorded>,
    /// type information of each bind group / descriptorset
    pub bind_groups: Dict<u32, BindGroupLayout>,
    /// byte ranges of push constants per shader stage
    pub push_constants: RenderPipelinePushConstantRanges,
    /// information about primitive assembly and rasterization
    pub rasterizer: RasterizerState,
    /// information about fragment depth/stencil tests
    pub depth_stencil: Option<DepthStencilState>,
    /// color attachments that this render pipeline writes to
    pub color_targets: Dict<u32, ColorTarget>,
    /// whether the fragment stage can be skipped entirely because it is known
    /// to have no effect.
    /// (this is `true` if no writeable bindings or color/depth/stencil attachments are
    /// being accessed in the fragment shader)
    pub skippable_fragment_stage: bool,
}

/// info required to initialize a compute pipeline in addition to shaders
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ComputePipelineInfo {
    /// compute thread-grid hierarchy setup information
    pub grid_info: ComputeGridInfo,
    /// type information of each bind group / descriptorset
    pub bind_groups: Dict<u32, BindGroupLayout>,
    /// range of the bytes of push constant memory that is visible to the compute shader
    pub push_constant_range: Option<Range<u32>>,
}

/// compute thread-grid hierarchy setup information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ComputeGridInfo {
    /// the `[x, y, z]` dimensions of the thread grid that makes up a workgroup
    pub thread_grid_size_per_workgroup: [u32; 3],
    /// whether memory from the [`mem::WorkGroup`] address space should be zero-initialized
    /// before the compute shader runs.
    ///
    /// [`mem::WorkGroup`]: crate::mem::WorkGroup
    pub zero_init_workgroup_memory: bool,
    /// the expected amount of threads in a [`Wave`].
    ///
    /// `None` if the pipeline makes no assumptions about [`Wave`] thread count.
    ///
    /// [`Wave`]: crate::Wave
    pub expected_threads_per_wave: Option<u32>,
}

/// vertex and fragment shader code, as well as meta information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderPipelineShaders {
    /// the vertex shader code
    ///
    /// for some target languages, `vert_code` and `frag_code` are identical
    /// (`Arc` points to the same object). Use `self.get_shared_shader_code`
    /// to find out if this is the case, and if yes, only compile the shader
    /// once for both vertex and fragment stages.
    pub vert_code: Arc<LanguageCode>,
    /// the entry point function name of the vertex shader
    pub vert_entry_point: &'static str,
    /// the fragment shader code
    ///
    /// for some target languages, `vert_code` and `frag_code` are identical
    /// (`Arc` points to the same object). Use `self.get_shared_shader_code`
    /// to find out if this is the case, and if yes, only compile the shader
    /// once for both vertex and fragment stages.
    pub frag_code: Arc<LanguageCode>,
    /// the entry point function name of the fragment shader
    pub frag_entry_point: &'static str,
}

impl RenderPipelineShaders {
    /// returns `Some(&code)` if both vertex and fragment shader use the same
    /// shared [`ShaderCode`] object, `None` otherwise
    pub fn get_shared_shader_code(&self) -> Option<&Arc<LanguageCode>> {
        Arc::ptr_eq(&self.vert_code, &self.frag_code).then_some(&self.vert_code)
    }

    /// converts `self` into `Ok(code)` if both vertex and fragment shader use the same
    /// shared [`LanguageCode`] object, otherwise two separate [`LanguageCode`] objects are returned
    pub fn into_shared_shader_code(self) -> Result<LanguageCode, (LanguageCode, LanguageCode)> {
        let into_inner_or_else_clone = |arc| match Arc::try_unwrap(arc) {
            Ok(code) => code,
            // some other reference exists, so we need to clone
            Err(code) => LanguageCode::clone(&code).clone(),
        };
        if Arc::ptr_eq(&self.vert_code, &self.frag_code) {
            drop(self.frag_code); // reduce refcount to 1 (if no other ref was created by the user)
            Ok(into_inner_or_else_clone(self.vert_code))
        } else {
            Err((
                into_inner_or_else_clone(self.vert_code),
                into_inner_or_else_clone(self.frag_code),
            ))
        }
    }
}

/// compute shader code and meta information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputeShader {
    /// compute shader code and span information by target language
    pub code: LanguageCode,
    /// compute shader entry point function name
    pub entry_point: &'static str,
}

/// byte-slice ranges of push constants that each shader stage uses.
/// These may overlap. Also a `Some(range)` is never empty.
///
/// important: generated wgsl vertex and fragment shaders share a single shader
/// module that declares a single `var<push_constant>` module-scope variable.
///
/// related: the `push_constant_ranges` field at https://docs.rs/wgpu/latest/wgpu/struct.PipelineLayoutDescriptor.html
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderPipelinePushConstantRanges {
    /// the byte size of the push constants. Serves as an upper bound for
    /// the ranges.
    ///
    /// for wgsl shaders, this is the byte size of the `var<push_constant>` variable's type
    pub push_constants_byte_size: u64,
    /// byte range used by the vertex shader
    ///
    /// for wgsl shaders, this is the range within the single `var<push_constant>`
    /// declaration that is in the shared shader code string of vertex and fragment shader.
    pub vert: Option<Range<u32>>,
    /// byte range used by the fragment shader
    ///
    /// for wgsl shaders, this is the range within the single `var<push_constant>`
    /// declaration that is in the shared shader code string of vertex and fragment shader.
    pub frag: Option<Range<u32>>,
}

/// render pipeline stages expressed as single range per stage
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RenderPipelinePushConstantsSingleRangePerStage {
    /// only the vertex stage uses push constants
    VertexOnly(Range<u32>),
    /// only the fragment stage uses push constants
    FragmentOnly(Range<u32>),
    /// the two stages use separate ranges
    Separate {
        /// vertex stage range
        vert: Range<u32>,
        /// vertex stage range
        frag: Range<u32>,
    },
    /// the two stages overlap, and since each stage can only be
    /// served by one range,
    Overlapping {
        /// single range for both ranges
        vert_frag: Range<u32>,
    },
}

impl RenderPipelinePushConstantRanges {
    /// returns a structure that contains only one push constant range per
    /// shader stage.
    ///
    /// the resulting type offers a [`as_slice()`] method
    ///
    /// [`as_slice()`]: RenderPipelinePushConstantRangesSeparate::as_slice()
    pub(crate) fn into_single_range_per_stage(self) -> Option<RenderPipelinePushConstantsSingleRangePerStage> {
        use RenderPipelinePushConstantsSingleRangePerStage as R;
        let non_empty = |r: &Range<u32>| !r.is_empty();
        match (self.vert.filter(non_empty), self.frag.filter(non_empty)) {
            (None, None) => None,
            (Some(v), None) => Some(R::VertexOnly(v)),
            (None, Some(f)) => Some(R::FragmentOnly(f)),
            (Some(v), Some(f)) => {
                let full_range = v.start.min(f.start)..v.end.max(f.end);
                let is_overlap = (v.start < f.end) && (v.end > f.start);
                match is_overlap {
                    true => Some(R::Overlapping { vert_frag: full_range }),
                    false => Some(R::Separate { vert: v, frag: f }),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_constant_ranges() {
        let ranges = |vert: Range<u32>, frag: Range<u32>| {
            RenderPipelinePushConstantRanges {
                vert: (!vert.is_empty()).then_some(vert),
                frag: (!frag.is_empty()).then_some(frag),
                push_constants_byte_size: 0, //unused in this test
            }
            .into_single_range_per_stage()
        };

        use RenderPipelinePushConstantsSingleRangePerStage as S;

        let separate = |vert: Range<u32>, frag: Range<u32>| Some(S::Separate { vert, frag });
        let overlap = |vert_frag: Range<u32>| Some(S::Overlapping { vert_frag });
        let vert_only = |vert: Range<u32>| Some(S::VertexOnly(vert));
        let frag_only = |frag: Range<u32>| Some(S::FragmentOnly(frag));

        assert_eq!(ranges(0..4, 4..8), separate(0..4, 4..8));
        assert_eq!(ranges(2..4, 4..6), separate(2..4, 4..6));
        assert_eq!(ranges(2..3, 5..6), separate(2..3, 5..6));
        assert_eq!(ranges(5..6, 2..3), separate(5..6, 2..3));
        assert_eq!(ranges(2..4, 3..5), overlap(2..5));
        assert_eq!(ranges(3..5, 2..4), overlap(2..5));
        assert_eq!(ranges(0..0, 1..1), None);
        assert_eq!(ranges(0..3, 1..2), overlap(0..3));
        assert_eq!(ranges(1..2, 0..3), overlap(0..3));
        assert_eq!(ranges(0..3, 1..1), vert_only(0..3));
        assert_eq!(ranges(1..1, 0..3), frag_only(0..3));
    }
}

/// information about primitive assembly and rasterization setup
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct RasterizerState {
    /// Which sequence of vertex-indices are assigned to the threads of a drawcall
    ///
    /// contains information about index-buffer usage and format
    pub vertex_indexing: Indexing,
    /// How the vertex sequence should be assembled into primitives,
    /// as well as other details about culling, rasterization etc.
    /// (no documentation yet)
    pub draw_info: Draw,
    /// which primitive winding order is considered front facing
    pub front_face: Winding,
    /// number of rasterized samples per pixel, as well as masking information
    ///
    /// the bit count in `self.samples` specifies the sample count,
    /// use `self.samples.len()` to obtain that value.
    ///
    /// the values of the bits in `self.samples` are sample mask
    pub samples: BitVec64,
    /// whether the alpha value written to color target #0 should be
    /// interpreted as an additional coverage mask, on top of the `self.samples` mask
    pub color_target0_alpha_to_coverage: bool,
}

/// (no documentation yet)
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupLayout {
    // TODO(release) low prio: add debug label
    /// (no documentation yet)
    pub bindings: BTreeMap<u32, BindingLayout>,
}

/// (no documentation yet)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindingLayout {
    /// (no documentation yet)
    pub visibility: StageMask,
    /// (no documentation yet)
    pub binding_ty: BindingType,
    /// (no documentation yet)
    pub shader_ty: StoreType,
}

impl BindingLayout {
    /// returns a `BindingLayout` corresponding to `T` with the maximum shader stage visibility
    pub fn from_ty_with_max_visibility<T: crate::Binding + ?Sized>(vertex_writable_storage_supported: bool) -> Self {
        let binding_ty = T::binding_type();
        BindingLayout {
            visibility: binding_ty.max_supported_stage_visibility(vertex_writable_storage_supported),
            binding_ty,
            shader_ty: T::store_ty(),
        }
    }
}
