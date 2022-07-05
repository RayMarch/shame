//! compute pipeline features for recording compute shaders and pipeline info
use std::{ops::RangeFrom, fmt::Display, marker::PhantomData};

use crate::{record_compute_shader, rec::{Shape, DType}, Ten, shader::Ids};
use super::{instantiate_push_constant, with_thread_compute_pipeline_info_mut};

use super::compute_pipeline_info::ComputePipelineInfo;

/// shame features available in compute pipeline recordings
pub struct ComputeFeatures<'a> {
    /// used to set up the work group dimensions and obtain invocation ids
    pub dispatch: WorkGroupSetup<'a>,
    /// pipeline io, used to add bind groups which contain textures, uniform/
    /// storage buffer bindings etc. 
    pub io: IO<'a>,
    phantom: PhantomData<()>,
}

/// pipeline io, used to add bind groups which contain textures, uniform/
/// storage buffer bindings etc.
pub struct IO<'a> {
    inner: crate::shader::ComputeIO<'a>,
    group_counter: RangeFrom<u32>,
}

/// used to specify the dimensions of workgroups in the current compute pipeline
/// and access various invocation related ids.
pub struct WorkGroupSetup<'a> {
    inner: crate::shader::WorkGroupSetup<'a>,
}

impl WorkGroupSetup<'_> {

    /*priv*/ fn new() -> Self {
        Self {
            inner: crate::shader::WorkGroupSetup {_phantom: PhantomData},
        }
    }

    /// specify the dimensions of workgroups in the current compute pipeline
    /// and access various invocation related ids.
    pub fn work_group(self, work_group_size: [usize; 3]) -> Ids {
        with_thread_compute_pipeline_info_mut(|c| {
            c.work_group_size = Some(work_group_size)
        });
        self.inner.work_group(work_group_size)
    }

}

impl IO<'_> {
    /// creates access to a new bind group, which can then be used to access 
    ///bindings such as textures, buffer bindings, samplers, ...
    pub fn group(&mut self) -> crate::shader::Group<RangeFrom<u32>>{
        self.inner.group(self.group_counter.next().expect("rangefrom iterator terminated"), 0..)
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
}

/// results of successfully recording a compute pipeline
#[derive(Debug)]
pub struct ComputePipelineRecording {
    /// the recorded shader as a glsl compute shader string
    pub shader_glsl: String,
    /// additional pipeline info for creating a compute pipeline layout
    pub info: ComputePipelineInfo,
}

impl Display for ComputePipelineRecording {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("shader:\n{}\n", self.shader_glsl))?;
        f.write_fmt(format_args!("compute-pipeline-info:\n{}\n", self.info))
    }
}

/// turn a rust compute pipeline function into a shader + pipeline info by 
/// executing it.
pub fn record_compute_pipeline(f: impl FnOnce(ComputeFeatures)) -> ComputePipelineRecording {

    let mut info = ComputePipelineInfo::default();

    let shader_glsl = record_compute_shader(|feat| {

        shame_graph::Context::with(|ctx| {
            //store render pipeline info in misc
            *ctx.misc_mut() = Box::new(ComputePipelineInfo::default())
        });
        
        let features = ComputeFeatures {
            dispatch: WorkGroupSetup::new(),
            io: IO {
                inner: feat.io,
                group_counter: 0..,
            },
            phantom: PhantomData,
        };
        f(features);
        super::record_groups_into_context();
        info = with_thread_compute_pipeline_info_mut(|c| std::mem::take(c));
    });
    
    ComputePipelineRecording {
        shader_glsl,
        info,
    }
}
