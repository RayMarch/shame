use std::{marker::PhantomData, ops::Deref};

use super::{
    io_iter::{BindGroupIter, PushConstants, VertexBufferIter},
    pipeline_kind::{Compute, Render},
    rasterizer::{PrimitiveAssembly, VertexStage},
    EncodingGuard,
};
use crate::frontend::rust_types::vec::vec;
use crate::{
    call_info,
    common::marker::*,
    frontend::{
        any::Any,
        rust_types::{
            len::{x1, x2, x3, GridDim},
            reference::{AccessModeReadable, Ref},
            type_traits::GpuStore,
            AsAny, GpuType,
        },
    },
    ir::recording::Context,
    mem, u32x1,
};

/// The sequence of vertex-indices as they are assigned to each thread of a drawcall
///
/// use [`Indexing::Incremental`] if no index buffer is used and
/// `Buffer*` variants if you are using an index buffer.
///
/// the vertex-index of thread #`t` is the `t % i`'th value in the [`VertexIdSequence`],
/// where `i` is the number of vertices per instance.
///
/// The [`Indexing`] sequence may contain duplicate ids.
///
/// the vertex-index can be accessed via `pipeline.vertex.index` where `pipeline`
/// is the object returned from [`EncodingGuard::new_render_pipeline`]
///
/// The [`Indexing`] sequence corresponds to `WebGPU`'s `vertexIndexList`
/// see https://www.w3.org/TR/webgpu/#vertex-processing
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Indexing {
    /// counting up incrementally,
    /// like 0, 1, 2, 3, 4...
    ///
    /// this sequence may not start at 0 if special draw commands are used
    /// that allow the user to set a starting id.
    ///
    /// no index-buffer is used.
    #[default]
    Incremental,
    /// the sequence is provided by a `[u8]` index buffer
    ///
    /// e.g.
    /// `let index_buffer: [u8; 6] = [0, 1, 2, 2, 3, 0]`
    ///
    /// if an id appears multiple times (like `2` in this example), the runtime
    /// is allowed to reuse the results of an earlier thread with the same id,
    /// as opposed to running another thread with that id.
    BufferU8,
    /// the sequence is provided by a `[u16]` index buffer
    ///
    /// e.g.
    /// `let index_buffer: [u16; 6] = [0, 1, 2, 2, 3, 0]`
    ///
    /// if an id appears multiple times (like `2` in this example), the runtime
    /// is allowed to reuse the results of an earlier thread with the same id,
    /// as opposed to running another thread with that id.
    BufferU16,
    /// the sequence is provided by a `[u32]` index buffer
    ///
    /// e.g.
    /// `let index_buffer: [u32; 6] = [0, 1, 2, 2, 3, 0]`
    ///
    /// if an id appears multiple times (like `2` in this example), the runtime
    /// is allowed to reuse the results of an earlier thread with the same id,
    /// as opposed to running another thread with that id.
    BufferU32,
}

/// The context of a draw command received from the Cpu.
///
/// This includes the bound resources and buffers at the time of issuing the command,
/// as well as access to render features.
pub struct DrawContext<'a> {
    /// access to bound vertex buffers as well as primitive assembly and indexing functionality
    ///
    /// ## example usage:
    ///
    /// ### vertex buffers
    /// ```
    /// use shame as sm;
    ///
    /// let vertex_uv: sm::f32x2 = self.vertices.buffers.next().at(self.vertices.index);
    /// // same as
    /// let uv_buffer: sm::VertexBuffer<sm::f32x2> = self.vertices.buffers.next();
    /// let uv = uv_buffer.at(self.vertices.index); // or self.vertices.instance_index
    /// ```
    /// ### primitive assembly
    /// ```
    /// use shame as sm;
    /// let vertex_pos: sm::f32x2 = self.vertices.buffers.next().at(self.vertices.index);
    /// // assemble triangles, keep only the counter-clockwise ones
    /// let primitive = self.vertices.assemble(vertex_pos, sm::Draw::triangle_list(sm::Ccw));
    /// ```
    /// ### vertex index arithmetic
    /// ```
    /// let vert_idx = self.vertices.index;
    /// let instance_idx = self.vertices.instance_index;
    /// /// cast for usage in arithmetic
    /// let even = self.vertices.index.to_u32() % 2;
    /// let x = self.vertices.index.to_f32();
    /// ```
    pub vertices: VertexStage<'a>,
    /// access to buffers, textures, etc. that were bound in groups
    ///
    /// ## example usage:
    ///
    /// ```
    /// use shame as sm;
    /// // access the next group (0..)
    /// let mut group0 = self.bind_groups.next();
    /// let mut group1 = self.bind_groups.next();
    /// // access a specific group
    /// let mut group4 = self.bind_groups.at(4);
    ///
    /// // downcast its bindings
    /// let buf: sm::Buffer<sm::Array<sm::f32x4>> = group0.next();
    /// ```
    /// see the documentation of [`BindingIter::next`] for
    /// more binding-type examples
    ///
    /// [`BindingIter::next`]: crate::BindingIter::next
    pub bind_groups: BindGroupIter<'a>,
    /// access to push-constant memory that was set before the drawcall
    ///
    /// ## example usage:
    /// ```
    /// let p: f32x4 = self.push_constants.get();
    /// ```
    /// see documentation of [`PushConstants::get`] for more examples.
    pub push_constants: PushConstants<'a>,
    pub(super) encoding: &'a EncodingGuard<Render>,
    pub(super) phantom: PhantomData<(Unsend, Unsync)>,
}

/// The context of a compute dispatch command received from the Cpu.
///
/// This includes the bound resources and buffers at the time of issuing the command.
pub struct DispatchContext<'a, Dim: GridDim> {
    /// access to buffers, textures, etc. that are bound to the pipeline in groups
    pub bind_groups: BindGroupIter<'a>,
    /// access to push-constant memory that was set before the dispatch
    pub push_constants: PushConstants<'a>,
    pub(super) grid: ComputeGrid<Dim>,
    pub(super) encoding: &'a EncodingGuard<Compute>,
    pub(super) phantom: PhantomData<(Unsend, Unsync, Dim)>,
}

impl<Dim: GridDim> Deref for DispatchContext<'_, Dim> {
    type Target = ComputeGrid<Dim>;

    fn deref(&self) -> &Self::Target { &self.grid }
}

/// arrays defining the dimensions of the workgroup's thread grid.
///
/// implemented by `[u32; N]` where `N` is in 1, 2 or 3.
///
/// - `[u32; 3]` for 3D grids
/// - `[u32; 2]` for 2D grids
/// - `[u32; 1]` for 1D number lines
///
/// examples:
/// - `[64]` for 1D workloads with 64 threads per workgroup
/// - `[8, 8]` for 2D workloads with 8x8 threads per workgroup
/// - `[4, 4, 2]` for 3D workloads with 4x4x2 threads per workgroup
pub trait GridSize {
    /// amount of dimensions of the workgroup's thread grid,
    /// either `x1` `x2` or `x3` depending on the length of the `Self` array
    type Dim: GridDim;

    /// interprets `self` as a 3D grid and returns its dimensions as
    /// `[x_len, y_len, z_len]`
    ///
    /// for example:
    /// - `[64] => [64, 1, 1]`
    /// - `[8, 8] => [8, 8, 1]`
    /// - `[4, 4, 2] => [4, 4, 2]`
    fn as_3d(&self) -> [u32; 3];
}
#[rustfmt::skip] impl GridSize for [u32; 1] { type Dim = x1; fn as_3d(&self) -> [u32; 3] {[self[0], 1, 1]} }
#[rustfmt::skip] impl GridSize for [u32; 2] { type Dim = x2; fn as_3d(&self) -> [u32; 3] {[self[0], self[1], 1]} }
#[rustfmt::skip] impl GridSize for [u32; 3] { type Dim = x3; fn as_3d(&self) -> [u32; 3] {[self[0], self[1], self[2]]} }

/// the grid hierarchy, consisting of a grid of workgroups which in turn consist
/// of a fixed (pipeline-defined) size grid of threads which execute the compute pipeline
#[allow(clippy::manual_non_exhaustive)]
pub struct ComputeGrid<Dim: GridDim> {
    /// A group of threads that works on a subset of the whole compute workload.
    /// The size of every Workgroup in a pipeline is the same, but can be chosen
    /// freely when creating the pipeline.
    pub workgroup: WorkGroup<Dim>,
    /// A per-workgroup value that represents the position of that workgroup
    /// within the workgroup grid of the current compute dispatch.
    ///
    /// It lies in the range `shame::zero()..self.workgroup_grid_size`
    pub workgroup_pos: vec<u32, Dim>,
    /// the dimensions of the grid of workgroups of the current compute dispatch.
    /// This value is provided to the graphics api when dispatching a compute workload.
    ///
    /// A per-dispatch value.
    pub workgroup_grid_size: vec<u32, Dim>,
    /// small group of threads that run in lockstep on the gpu hardware.
    /// Use this field to access wave intrinsics, which allow fast communication
    /// between threads of a wave.
    pub wave: Wave,
    //pub thread: Thread<Dim>,
    /// the position of the thread within the compute dispatch grid
    ///
    /// Identical to `self.workgroup_pos * self.workgroup_grid_size + self.workgroup.thread_pos`.
    ///
    /// A per-thread value
    pub thread_pos: vec<u32, Dim>,
    private_ctor: (),
}

impl<Dim: GridDim> ComputeGrid<Dim> {
    #[track_caller]
    pub(crate) fn new() -> Self {
        let dim = Dim::LEN;
        Self {
            workgroup: WorkGroup {
                thread_pos: Any::shrink_vector(dim, Any::new_thread_pos_within_workgroup()).into(),
                thread_grid_size: Any::thread_grid_size_within_workgroup(dim)
                    .suggest_ident("workgroup_size")
                    .into(),
                thread_id: Any::new_thread_id_within_workgroup().into(),
                private_ctor: (),
            },
            workgroup_pos: Any::shrink_vector(dim, Any::new_workgroup_pos_within_dispatch()).into(),
            workgroup_grid_size: Any::shrink_vector(dim, Any::new_workgroup_grid_size_within_dispatch()).into(),
            wave: Wave {
                thread_id: Any::new_thread_id_within_wave().into(),
                thread_count: Any::new_thread_count_within_wave().into(),
                private_ctor: (),
            },
            thread_pos: Any::shrink_vector(dim, Any::new_thread_pos_within_dispatch()).into(),
            private_ctor: (),
        }
    }
}

/// A group of threads that works on a subset of the whole compute workload.
/// The size of every Workgroup in a pipeline is the same, but can be chosen
/// freely when creating the pipeline.
///
/// A single compute dispatch starts one or many workgroups at once. The number
/// of Workgroups can be chosen differently for each dispatch to adjust for
/// different amounts of work at runtime.
///
/// Threads within a Workgroup can share memory and use synchronization barriers
/// in the compute shader to synchronize memory accesses with each other.
///
/// see https://www.w3.org/TR/WGSL/#compute-shader-workgroups
#[allow(clippy::manual_non_exhaustive)]
pub struct WorkGroup<Dim: GridDim> {
    /// the position of the thread within its Workgroups thread grid.
    ///
    /// this value is different per thread
    pub thread_pos: vec<u32, Dim>,
    /// the dimensions of the workgroup's thread grid. Equal for every workgroup of a pipeline.
    ///
    /// A constant defined via [`EncodingGuard::new_compute_pipeline`]
    pub thread_grid_size: vec<u32, Dim>, // user defined [u32; N] fed into sm::vec::new
    /// the 1-dimensional numeric index of a thread within its workgroup.
    ///
    /// It lies in the range `shame::zero()..self.thread_grid_size`
    ///
    /// this value is different per thread
    pub thread_id: u32x1,
    private_ctor: (),
}

/// the *Wave*, also known as *Warp* (cuda), *Subgroup* (vulkan, webgpu), *SimdGroup* (metal)
/// is a small group of threads that run in lockstep on the gpu hardware.
/// They are able to share information at very fast speeds via wave intrinsics,
/// which are available as associated functions on `Wave`
#[allow(clippy::manual_non_exhaustive)]
pub struct Wave {
    /// the 1-dimensional numeric index of a thread within its wave.
    ///
    /// It lies in the range `shame::zero()..self.thread_count`
    ///
    /// this value is different per thread
    pub thread_id: u32x1,
    /// the maximum amount of threads in a wave
    ///
    /// - A *per-workgroup* value in compute pipelines
    /// - A *per-wave* value in render pipelines
    ///
    /// see https://www.w3.org/TR/WGSL/#subgroup-size-builtin-value
    pub thread_count: u32x1,
    private_ctor: (),
}

// pub struct Thread<Dim: GridDim> {
//     pub(super) phantom: PhantomData<(Dim)>,
// }
