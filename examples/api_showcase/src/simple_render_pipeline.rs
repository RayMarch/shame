use shame::*;

pub fn main() {
    let out = shame::record_render_pipeline(simple_render_pipeline);
    println!("{}", out.to_string_colored());
}

/// this serves as an overview, for more details, look at [`crate::render_pipeline`]
fn simple_render_pipeline(mut feat: shame::RenderFeatures) {
    // create three individual vertex buffers as inputs for position, normal and uv.
    // Every call to `feat.io.vertex_buffer()` creates a separate vertex buffer.
    // for interleaved vertex attributes within a single buffer, see `structs_example.rs`
    let position: float3 = feat.io.vertex_buffer();
    let normal: float3 = feat.io.vertex_buffer();
    let uv: float2 = feat.io.vertex_buffer();

    // create a new bind group builder
    // bind group indices in shame's render/compute pipeline api count like 0, 1, 2,..
    let mut bind_group0 = feat.io.group();

    // define binding 0 of bind group 0 to be a float4x4 uniform block.
    let transform: float4x4 = bind_group0.uniform_block();
    // define binding 1 of bind group 0 to be a Sampler2D
    let sampler: CombineSamplerRGBA = bind_group0.combine_sampler();

    // expand the position with a 1.0 value to make it `float4`.
    // this is the same as `(position, 1.0)`
    let raster_position = position.xyz1();

    // calling rasterize gives us the `polygon` object, which allows us to interpolate
    // per-vertex values across the polygon's surface. This gives us per-fragment values.
    let polygon = feat.raster.rasterize_indexless(
        raster_position,
        Cull::default(), //defaults to culling Clockwise triangles
        PrimitiveTopology::TriangleList,
    );

    // interpolate the vertex `normal` across the rasterized polygon's surface
    // this takes care of declaring a vertex shader output and a fragment shader input for `normal`
    let fragment_normal = polygon.plerp(normal);

    // turn a tuple into a float3 by calling `rec()`
    let light_direction = (1.0, 1.0, 0.0).rec();

    // n dot l lighting
    let light_intensity = fragment_normal.dot(light_direction.normalize());

    // interpolate the vertex `uv` coordinates to get per-fragment UVs, then
    // use them to sample from sampler.
    let texture_color = sampler.sample(polygon.plerp(uv)).xyz();

    let color = texture_color * light_intensity;

    use pipeline::pixel_format::*;

    // we want to output to a color and a depth rendertarget
    let mut color_out = feat.io.color::<RGBA_8888_sRGB>();
    let mut depth_out = feat.io.depth::<Depth32>();

    // extend color by (color, 1.0) to have an alpha component
    color_out.set(color.xyz1());

    // perform "less" depth testing and write the new polygon's depth values
    depth_out.test_write(DepthTest::Less, polygon.depth());
}
