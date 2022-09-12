//! Index buffer configuration types
use std::marker::PhantomData;
use crate::{shader::RenderIO, int, rec::AnyDowncast};
use crate::rec::Stage;

/// datatypes representing the individual indices of index buffers
pub trait IndexFormat {
    /// enum version of `Self`
    const INDEX_DTYPE: IndexDType;
}
impl IndexFormat for u32 {const INDEX_DTYPE: IndexDType = IndexDType::U32;}
impl IndexFormat for u16 {const INDEX_DTYPE: IndexDType = IndexDType::U16;}

/// datatypes representing the individual indices of index buffers
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexDType {
    U32,
    U16,
}

/// How consecutive indices of the depth buffer are interpreted to form
/// primitive shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    /// Indices that form triangles: {1, 2, 3} {4, 5, 6} {7, 8, 9} ...
    TriangleList,
    /// Indices that form triangles: {1, 2, 3} {3, 2, 4} {3, 4, 5} ... (default)
    TriangleStrip,
}

impl Default for PrimitiveTopology {
    fn default() -> Self {
        PrimitiveTopology::TriangleList
    }
}

/// layout of the index buffer
pub trait PrimitiveIndex {
    /// datatype of individual indices
    type Format: IndexFormat;
    /// Describes how primitives are assembled from a stream of indices
    const TOPOLOGY: PrimitiveTopology;

    /// creates an object of `Self` but doesn't necessarily use it in the
    /// recorded shader. If you want to use it in your shader to form primitives
    /// pass it into the [`Raster::rasterize`] function
    fn new(io: &mut RenderIO) -> Self;

    /// glsl: `gl_VertexID`
    /// returns a per-vertex variable
    fn vertex_index(&self) -> int {
        shame_graph::Any::v_vertex_id_vk().downcast(Stage::Vertex)
    }
}

/// Indices that form triangles: {1, 2, 3} {4, 5, 6} {7, 8, 9} ...
pub struct TriangleList<T: IndexFormat> {
    t: PhantomData<T>,
}

impl<T: IndexFormat> PrimitiveIndex for TriangleList<T> {
    type Format = T;
    const TOPOLOGY: PrimitiveTopology = PrimitiveTopology::TriangleList;

    fn new(_io: &mut RenderIO) -> Self {
        Self {t: PhantomData}
    }
}

/// Indices that form triangles: {1, 2, 3} {3, 2, 4} {3, 4, 5} ...
pub struct TriangleStrip<T: IndexFormat> {
    t: PhantomData<T>,
}

impl<T: IndexFormat> PrimitiveIndex for TriangleStrip<T> {
    type Format = T;
    const TOPOLOGY: PrimitiveTopology  = PrimitiveTopology::TriangleStrip;

    fn new(_io: &mut RenderIO) -> Self {
        Self {t: PhantomData}
    }
}