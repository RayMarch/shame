use shame::prelude::*;

pub type Index = u32;

/// this struct can be auto-generated from Vertex, see the "mirror" feature example
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct VertexCpu {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

/// this struct can be auto-generated from VertexCpu, see the "mirror" feature example
#[derive(shame::Fields)]
struct Vertex {
    pub pos: float3,
    pub color: float3,
}

// try editing the shader and running the `hot_reload_shaders` binary.
// It will update the shader while the `hot_reload_engine` keeps running.
// On my machine the entire process takes about 1 second for incremental builds.
// If you have any ideas how to speed this up further, feel free to contact me!
pub fn pipeline(mut f: RenderFeatures) {
    let vertex: Vertex = f.io.vertex_buffer();
    let index: TriangleList<Index> = f.io.index_buffer();
    let transform: float4x4 = f.io.group().uniform_block();

    let time: float = f.io.push_constant();
    let pos = (vertex.pos.xy().rotate_2d(time * 1.0), 0.0, 1.0);

    let poly = f.raster.rasterize(transform * pos, Cull::CW, index);
    f.io.color::<RGBA_Surface>().set((poly.lerp(vertex.color), 1.0));
}
