//! render pipeline features for recording vertex, fragment shaders and pipeline
//! info
use super::culling::Cull;
use super::pixel_format::{IsColorFormat, IsDepthFormat};
use super::render_pipeline_info::{
    AttributeInfo, ColorTargetInfo, VertexBufferInfo, VertexStepMode,
};
use super::topology::{IndexFormat, PrimitiveIndex};
use super::{instantiate_push_constant, target, with_thread_render_pipeline_info_mut};
use crate::rec::fields::Fields;
use crate::record_render_shaders;
use crate::shader::{FragmentOutputsBuilder, Primitive, VertexStreamBuilder};
use crate::{rec::*, PrimitiveTopology};
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::RangeFrom;

use super::render_pipeline_info::RenderPipelineInfo;

/// results of successfully recording a render pipeline
#[derive(Debug, Clone)]
pub struct RenderPipelineRecording {
    /// a pair of glsl (vertex_shader, fragment_shader) strings
    pub shaders_glsl: (String, String),
    /// additional pipeline info for creating a render pipeline layout
    pub info: RenderPipelineInfo,
}

impl Display for RenderPipelineRecording {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("vertex-shader:\n{}\n", self.shaders_glsl.0))?;
        f.write_fmt(format_args!("fragment-shader:\n{}\n", self.shaders_glsl.1))?;
        f.write_fmt(format_args!("render-pipeline-info:\n{}\n", self.info))
    }
}

/// shame features available in render pipeline recordings
pub struct RenderFeatures<'a> {
    /// rasterization functionality, turning per-vertex data into primitives
    pub raster: Raster<'a>,
    /// pipeline io, used to add bind groups which contain textures, uniform/
    /// storage buffer bindings etc.
    pub io: IO<'a>,
    phantom: PhantomData<()>,
}

/// access to rasterization functionality (drawing primitives on the pixel
/// grid). Must be used for a render pipeline to be valid, otherwise the
/// recording will yield errors.
pub struct Raster<'a> {
    inner: crate::shader::Raster<'a>,
}

/// pipeline io, used to add bind groups which contain textures, uniform/
/// storage buffer bindings etc.
pub struct IO<'a> {
    inner: crate::shader::RenderIO<'a>,
    vertex_stream_builder: VertexStreamBuilder<RangeFrom<u32>>,
    fragment_outputs_builder: FragmentOutputsBuilder<RangeFrom<u32>>,
    group_counter: RangeFrom<u32>,
}

impl RenderPipelineRecording {
    /// convenience function for turing the struct into nested tuples
    pub fn unpack(self) -> ((String, String), RenderPipelineInfo) {
        (self.shaders_glsl, self.info)
    }
}

impl IO<'_> {
    /// instantiates T as interleaved attributes from a single vertex buffer
    pub fn vertex_buffer<T: Fields>(&mut self) -> T {
        self.vertex_buffer_detailed(VertexStepMode::Vertex)
    }

    /// instantiates T as interleaved attributes from a single vertex buffer
    /// which provides values per-instance
    pub fn instance_buffer<T: Fields>(&mut self) -> T {
        self.vertex_buffer_detailed(VertexStepMode::Instance)
    }

    fn vertex_buffer_detailed<T: Fields>(&mut self, step_mode: VertexStepMode) -> T {
        let (t, details) = self.vertex_stream_builder.attributes_detailed();

        let attributes = details
            .into_iter()
            .map(|(tensor, loc_range)| AttributeInfo {
                location: loc_range.start,
                type_: tensor,
            })
            .collect::<Vec<_>>();

        if let shame_graph::ShaderKind::Vertex = crate::current_shader() {
            with_thread_render_pipeline_info_mut(|r| {
                r.vertex_buffers.push(VertexBufferInfo {
                    step_mode,
                    attributes,
                })
            });
        }

        t
    }

    /// instantiates an index buffer type for use with [`Raster::rasterize`]
    pub fn index_buffer<T: PrimitiveIndex>(&mut self) -> T {
        T::new(&mut self.inner)
        // recording happens when the index buffer is used in Raster::rasterize (may be changed in the future)
    }

    /// access a color target of format `T` as render pipeline output.
    /// Theres an alternative [`IO::color_ms`] call for multisampled color
    /// targets
    pub fn color<T: IsColorFormat>(&mut self) -> target::Color<T>
    where
        <<T>::Item as AsTen>::S: crate::rec::IsShapeScalarOrVec,
    {
        let color_target_index = with_thread_render_pipeline_info_mut(|r| {
            let index = r.color_targets.len();
            r.color_targets.push(ColorTargetInfo {
                sample_count: 1,
                color_format: <T as IsColorFormat>::ENUM,
                blending: None,
            });
            index
        });

        let value: WriteOnly<<T::Item as AsTen>::S, <T::Item as AsTen>::D> = self
            .fragment_outputs_builder
            .color_with_ident::<Ten<<T::Item as AsTen>::S, <T::Item as AsTen>::D>>(Some(
                "color_out".to_string(),
            ));
        target::Color {
            _phantom: PhantomData,
            value,
            color_target_index,
        }
    }

    /// access a depth target of format `T` as the depth buffer used by the
    /// current pipeline
    pub fn depth<T: IsDepthFormat>(&mut self) -> target::Depth<T> {
        with_thread_render_pipeline_info_mut(|r| {
            assert!(
                r.depth_stencil_target.is_none(),
                "only one depth stencil target supported per pipeline"
            ); //TODO: this could be handled nicer, maybe calling this consumes self to assure its only called once
            r.depth_stencil_target = Some(<T as IsDepthFormat>::ENUM);
        });

        target::Depth {
            _phantom: PhantomData,
        }
    }

    /// access the push constant as a tensor of given `S` `D`.
    ///
    /// usage:
    /// ```text
    /// let p: float4 = io.push_constant();
    /// ```
    pub fn push_constant<S: Shape, D: DType>(&mut self) -> Ten<S, D> {
        instantiate_push_constant()
    }

    /// access a multisampled color target of format `T` as render pipeline
    /// output. `SAMPLES` must be a valid sample count (2, 4, 8, 16, 32, 64)
    pub fn color_ms<T: IsColorFormat, const SAMPLES: u8>(&mut self) -> target::ColorMS<T, SAMPLES>
    where
        <<T>::Item as AsTen>::S: crate::rec::IsShapeScalarOrVec,
    {
        let valid_sample_counts = [2, 4, 8, 16, 32, 64];
        assert!(
            valid_sample_counts.contains(&SAMPLES),
            "multisampling sample count must be one of {:?}. got {SAMPLES}",
            valid_sample_counts
        );

        let color_target_index = with_thread_render_pipeline_info_mut(|r| {
            let index = r.color_targets.len();
            r.color_targets.push(ColorTargetInfo {
                sample_count: SAMPLES,
                color_format: <T as IsColorFormat>::ENUM,
                blending: None,
            });
            index
        });

        let value: WriteOnly<<T::Item as AsTen>::S, <T::Item as AsTen>::D> = self
            .fragment_outputs_builder
            .color_with_ident::<Ten<<T::Item as AsTen>::S, <T::Item as AsTen>::D>>(Some(
                "color_out".to_string(),
            ));
        target::ColorMS {
            _phantom: PhantomData,
            value,
            color_target_index,
        }
    }

    /// creates access to a new bind group, which can then be used to access
    /// bindings such as textures, buffer bindings, samplers, ...
    pub fn group(&mut self) -> crate::shader::Group<RangeFrom<u32>> {
        self.inner.group(
            self.group_counter
                .next()
                .expect("rangefrom iterator terminated"),
            0..,
        )
    }
}

impl<'a> Raster<'a> {
    /// rasterize primitives at the provided clip_space positions.
    /// The clip space positions are combined to primitives according to the
    /// primitive topology specified in the index buffer.
    /// If the indexing happens in the winding order described in
    /// `primitive_culling`, the primitive is not rasterized
    /// (see [face culling](https://learnopengl.com/Advanced-OpenGL/Face-culling)
    /// ).
    pub fn rasterize<P: PrimitiveIndex>(
        self,
        clip_space_position: impl AsFloat4,
        primitive_culling: Cull,
        _index_buffer: P,
    ) -> Primitive<'a> {
        with_thread_render_pipeline_info_mut(|r| {
            r.cull = Some(primitive_culling);
            r.primitive_topology = Some(P::TOPOLOGY);
            r.index_dtype = Some(<P::Format as IndexFormat>::INDEX_DTYPE);
        });
        self.inner.rasterize(clip_space_position.as_ten())
    }

    /// same as `rasterize` except without using an index buffer
    pub fn rasterize_indexless(
        self,
        clip_space_position: impl AsFloat4,
        primitive_culling: Cull,
        primitive_topology: PrimitiveTopology,
    ) -> Primitive<'a> {
        with_thread_render_pipeline_info_mut(|r| {
            r.cull = Some(primitive_culling);
            r.primitive_topology = Some(primitive_topology);
            r.index_dtype = None;
        });
        self.inner.rasterize(clip_space_position.as_ten())
    }
}

/// turn a rust render pipeline function into shaders and pipeline info by
/// executing it.
/// The provided function is called multiple times to record different shader
/// stages.
pub fn record_render_pipeline(mut f: impl FnMut(RenderFeatures)) -> RenderPipelineRecording {
    //in a render pipeline, the provided `f` is recorded twice, which yields
    //two shaders and two pipeline infos. Since the two (vertex and fragment) shaders
    //actually share the pipeline, the two RenderPipelineInfo recordings are later
    //merged into one before being returned to the user
    let mut pipeline_infos = vec![];

    let (vert, frag) = record_render_shaders(|feat| {
        shame_graph::Context::with(|ctx| {
            //store render pipeline info in misc
            *ctx.misc_mut() = Box::new(RenderPipelineInfo::default())
        });

        let pixel_builder = feat.io.pixel(0..);
        let vertex_builder = feat.io.vertex(0..);
        let features = RenderFeatures {
            raster: Raster { inner: feat.raster },
            io: IO {
                inner: feat.io,
                vertex_stream_builder: vertex_builder,
                fragment_outputs_builder: pixel_builder,
                group_counter: 0..,
            },
            phantom: PhantomData,
        };
        f(features);
        super::record_groups_into_context();
        let render_pipeline_info = with_thread_render_pipeline_info_mut(|r| std::mem::take(r));
        pipeline_infos.push(render_pipeline_info);
    });

    assert!(pipeline_infos.len() == 2);
    let vert_info = pipeline_infos.swap_remove(0);
    let frag_info = pipeline_infos.swap_remove(0);

    RenderPipelineRecording {
        shaders_glsl: (vert, frag),
        info: RenderPipelineInfo::merge_individual_stage_recordings(vert_info, frag_info),
    }
}
