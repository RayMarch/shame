//! pipeline io traits to wrap [render_pipeline::IO] and [compute_pipeline::IO]
use super::*;
use crate::shader::Group;
use std::ops::RangeFrom;

/// a trait to wrap [render_pipeline::IO] and [compute_pipeline::IO]
pub trait GenericPipelineIO {
    /// same as the `group` function of the respective io objects, different
    /// name to avoid name collisions
    fn next_group(&mut self) -> Group<RangeFrom<u32>>;
    /// same as the `push_constant` function of the respective io objects
    fn push_constant<S: Shape, D: DType>(&mut self) -> Ten<S, D>;
}

impl GenericPipelineIO for compute_pipeline::IO<'_> {
    fn next_group(&mut self) -> Group<RangeFrom<u32>> {
        self.group()
    }

    fn push_constant<S: Shape, D: DType>(&mut self) -> Ten<S, D> {
        self.push_constant()
    }
}

impl GenericPipelineIO for render_pipeline::IO<'_> {
    fn next_group(&mut self) -> Group<RangeFrom<u32>> {
        self.group()
    }

    fn push_constant<S: Shape, D: DType>(&mut self) -> Ten<S, D> {
        self.push_constant()
    }
}
