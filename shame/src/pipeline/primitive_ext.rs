//! Extends the [`Primitive`] type from the shader api
use super::target::DepthWrite;
use crate::shader::Primitive;

impl Primitive<'_> {
    /// the primitive's DepthWrite value based on the clip space position
    /// input that was given to the rasterizer.
    pub fn depth(&self) -> DepthWrite { DepthWrite::PrimitiveZ }
}
