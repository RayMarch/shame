//! Extends the [`Primitive`] type from the shader api
use crate::shader::Primitive;
use super::target::DepthWrite;

impl Primitive<'_> {
    /// the primitive's DepthWrite value based on the clip space position
    /// input that was given to the rasterizer.
    pub fn depth(&self) -> DepthWrite {
        DepthWrite::PrimitiveZ
    }
}