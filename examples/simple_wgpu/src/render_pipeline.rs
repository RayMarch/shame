use shame::prelude::*;

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

pub fn pipeline(mut f: RenderFeatures) {
    let vertex: Vertex = f.io.vertex_buffer();
    let index: TriangleList<super::Index> = f.io.index_buffer();
    let transform: float4x4 = f.io.group().uniform_block();

    let pos = (vertex.pos.xy().rotate_2d(f.io.push_constant() as float), 0.0, 1.0);

    let poly = f.raster.rasterize(transform * pos, Cull::CW, index);

    // in wgpu we only know the window's color target format at runtime.
    // in shame, "RGBA_Surface" is a way of saying
    // "whatever RGBA format the render surface ends up having".
    // When converting from a `shame::RenderPipelineRecording` to a `wgpu::RenderPipeline`
    // the specific `wgpu::TextureFormat` is provided.
    f.io.color::<RGBA_Surface>().set((poly.lerp(vertex.color), 1.0));
}
