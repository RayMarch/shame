
use shame::prettify::syntax_highlight_glsl;
use shame::prelude::*;

fn my_render_pipeline(mut f: RenderFeatures) {
    // use `f` to build your shader

    // define a vertex layout
    #[derive(shame::Fields)]
    struct MyVertex {
        pos: float3,
        color: float4,
    }

    // use the vertex layout in this shader
    let vert: MyVertex = f.io.vertex_buffer();
    // use an index buffer containing u32s
    let topology: TriangleList<u32> = f.io.index_buffer();

    // read from the 0th bind group
    let mut group0 = f.io.group();
    // which has a uniform block at binding #0
    let matrix: float4x4 = group0.uniform_block();

    // use the uniform data in calculations
    let clip_xyzw = matrix * (vert.pos, 1.0);

    // rasterize triangles at clip space positions with clockwise culling
    let polygon = f.raster.rasterize(
        clip_xyzw, Cull::CW, topology,
    );

    // vertex -> fragment interpolation
    let mut frag_color = polygon.lerp(vert.color);

    // read the push constant as a float
    let time: float = f.io.push_constant();
    frag_color += time.sin() * 0.1;

    // write to an sRGB render-target with alpha.
    // use alpha blending with frag_color as src color
    f.io.color::<RGBA_8888_sRGB>().blend(
        Blend::alpha(), 
        frag_color
    );
}

pub fn main() {
    // generate the shaders and pipeline layout
    let out = shame::record_render_pipeline(my_render_pipeline);
    
    let (vertex_shader, fragment_shader) = &out.shaders_glsl;
    let info = &out.info;

    println!("{}", out.to_string_colored());
}