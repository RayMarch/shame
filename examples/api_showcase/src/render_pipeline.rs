
use shame::*;

pub fn main() {
    let out = shame::record_render_pipeline(my_render_pipeline);
    println!("{}", out.to_string_colored());
}

fn my_render_pipeline(mut feat: RenderFeatures) {

    // vertex inputs consist of vertex attributes, which can be stored
    // interleaved in one buffer, in separate vertex buffers, or in a mixture of
    // both

    // describe a vertex layout and `derive(shame::Fields)` to make it usable
    // within shame. only tensor types, such as `float`, `floatN` and `floatMxN`
    // are valid vertex attributes.
    #[derive(shame::Fields)]
    struct Vertex {
        pos: float3,
        nor: float3,
        uv: float2,
    }
    // read `Vertex` as interleaved vertex attributes from a
    // common vertex buffer...
    let vertex: Vertex = feat.io.vertex_buffer(); // vertex buffer #0
    // ...and access individual vertex attributes like this:
    vertex.pos;
    vertex.nor;
    vertex.uv;

    // if you want each vertex attribute to be in its own vertex buffer,
    // do this:
    let v_pos: float3 = feat.io.vertex_buffer(); // vertex buffer #1
    let v_nor: float3 = feat.io.vertex_buffer(); // vertex buffer #2
    let v_uv : float2 = feat.io.vertex_buffer(); // vertex buffer #3

    // the same rules apply to the `instance_buffer` function, which is used
    // to describe per-instance vertex attributes.
    let inst_color: float4 = feat.io.instance_buffer(); // vertex buffer #4
    let inst_pos_nor_uv: Vertex = feat.io.instance_buffer(); // vertex buffer #5

    // choose an index buffer structure.
    // both primitive topology and index datatype need to be specified
    // e.g. TriangleList<T>, TriangleStrip<T>
    let topo: TriangleList<u32> = feat.io.index_buffer();

    // additional inputs can be added via bind groups
    // create bind groups like this:
    let mut group0 = feat.io.group(); // bind group #0
    let mut group1 = feat.io.group(); // bind group #1
    // you can create bindgroups any time, they don't have to be created in one
    // block.

    // you can add uniform blocks and read-only storage buffers to a group like
    // so:
    // define the buffer layout
    #[derive(shame::Fields)]
    struct Transforms {
        world: float4x4,
        view: float4x4,
        projection: float4x4,
    }

    // add a uniform block binding to group 0, which has `Transforms` inside
    let tfs: Transforms = group0.uniform_block(); // bind group #0 binding #0

    // you can also use tensors, arrays etc. directly as bindings
    let tfs2: float4x4 = group0.uniform_block(); // bind group #0 binding #1
    let tfs2: Array<float4x4> = group0.storage(); // bind group #0 binding #2

    // if you want to add an array of structs, it works as follows:
    let tfs2: Array<Struct<Transforms>> = group0.storage(); // bind group #0 binding #3
    //for more info on Arrays and storage, see the compute pipeline example

    // matrix/vector multiplication works as expected
    let clip_pos = tfs.projection * tfs.view * tfs.world * (vertex.pos, 1.0);

    // there's also a shorthand for appending a 1.0 to a float3
    let clip_pos = tfs.projection * tfs.view * tfs.world * vertex.pos.xyz1();
    // see the vec matrix example for more shorthands etc.

    let culling = Cull::CW; //choose the winding order of the triangles you want
    //to cull, or `Cull::Off` for disabling face culling

    let use_index_buffer = true;

    if !use_index_buffer {
        // if you don't want to use an index buffer, the primitive topology for
        // the vertex inputs has to be specified separately.
        let primitive_topology = PrimitiveTopology::TriangleList;

        // call the rasterizer without index buffer
        let primitive = feat.raster.rasterize_indexless(clip_pos, culling, primitive_topology);
        return; // lets continue this example with an index buffer instead.
    }

    // rasterize at the clip space position, with the specified culling and
    // index buffer.
    let primitive = feat.raster.rasterize(clip_pos, culling, topo);

    // interpolate the per-vertex position across the primitive to obtain a
    // per-fragment position
    let frag_pos = primitive.lerp(v_pos);

    // there are different ways to interpolate across the primitive.
    let frag_uv = primitive.lerp(v_uv); // linear interpolation
    let frag_uv = primitive.flat(v_uv); // flat interpolation (takes the value of the "provoking" vertex)
    let frag_uv = primitive.plerp(v_uv); // perspective aware interpolation (takes clip_pos.w into account)
    // of those three, `plerp` is the most commonly used.

    // read push constants of a certain type (float2 in this case)
    // shame does not support different push constants for vertex and fragment
    // stage so far.
    let push_constant: float2 = feat.io.push_constant();

    // tuples can be interpolated with a single call
    let (frag_pos, frag_nor, frag_uv) = primitive.plerp((v_pos, v_nor, v_uv));

    // for texture sampling we need a sampler and a texture
    let sampler = group1.sampler(); // group #1 binding #0

    // lets import a texture that takes a float2 as sampling input (=texture coordinates)
    // and returns a float4
    let texture0: Texture<float4, float2> = group1.texture(); // group #1 binding #1

    // there are also type aliases for the first generic argument if you prefer that
    let texture1: TextureRGBA<float2> = group1.texture(); // group #1 binding #2

    // the texture coordinate generic argument defaults to `float2` so you can omit it
    let texture2: Texture<float4> = group1.texture(); // group #1 binding #3

    // a cubemap texture is just a texture that takes a CubeDir as sampling input
    let texture_cube: TextureRGBA<CubeDir> = group1.texture(); // group #1 binding #4

    // sampling with the per-vertex value `v_uv` will result in per-vertex color values in `v_sample`
    let v_sample = texture0.sample(sampler, v_uv);

    // sampling with the per-fragment value `frag_uv` instead will output per-fragment color values in `f_sample`
    let f_sample = texture0.sample(sampler, frag_uv);

    // you can also just interpolate the vertex uvs in place when you need them
    let f_sample = texture0.sample(sampler, primitive.plerp(v_uv));

    // this is how you sample from a cubemap texture. The CubeDir newtype is used to differentiate between cube map and 3d textures.
    let f_sample = texture_cube.sample(sampler, CubeDir(frag_nor));

    // alternatively to separate texture and sampler, theres also a combined texture + sampler object
    // which is supported by some graphics apis
    let combine_sampler: CombineSampler<float4, float2> = group1.combine_sampler();
    let v_sample = combine_sampler.sample(v_uv);

    // per fragment values have partial derivatives that describe their
    // difference to the neighboring fragment in a local 2x2 fragment grid
    let sample_dx = f_sample.dx(); //x partial derivative
    let sample_dy = f_sample.dy(); //y partial derivative
    let sample_dxy: (float4, float4) = f_sample.dxy(); //or just both at the same time

    // add a depth buffer (there can only be one per pipeline) of a certain
    // format
    use shame::pixel_format::*; //import all the pixel format types
    let mut depth_buffer = feat.io.depth::<Depth32>();

    // there are different ways to interact with the depth buffer, but after
    // you interact with it, it is consumed.
    let way = 1;
    match way {
        1 => {
            // this is the most common way to interact with a depth buffer
            // if the fragment depth from the rasterized clip position is
            // "less or equal" the existing depth at that depth buffer
            // pixel, the fragment colors and depth get written.
            depth_buffer.test_write(DepthTest::LessOrEqual, primitive.depth());
            // alternatively to `primitive.depth()` you can write
            // `DepthWrite::PrimitiveZ`
        }
        2 => {
            // if the fragment depth from the rasterized clip position is
            // "less or equal" the existing depth at that depth buffer
            // pixel, the fragment colors get written, but not the depth.
            depth_buffer.test_write(DepthTest::LessOrEqual, DepthWrite::Off);
        }
        3 => {
            // if the fragment depth from the rasterized clip position is
            // "less or equal" than `frag_nor.x()`
            // pixel, the fragment colors get written, and `frag_nor.x()` gets
            // written to the depth buffer.
            depth_buffer.test_write(
                DepthTest::LessOrEqual,
                DepthWrite::Write(frag_nor.x())
            );
        }
        _ => {
            // Always write `frag_nor.x()` to the depth buffer, as well as
            // the fragment colors to their respective targets
            depth_buffer.test_write(
                DepthTest::Always,
                DepthWrite::Write(frag_nor.x())
            );
        }
    }

    let result = f_sample;
    use shame::pixel_format::*; //import all the pixel format types

    // write `result` to an `RGBA_8888_sRGB` color target
    feat.io.color::<RGBA_8888_sRGB>().set(result); // color target #0

    // writing `result` to an `RGB_888` color target causes a compiler error!
    // RGB only has 3 components, result has 4!
    //feat.io.color::<RGB_888>().set(result); // error: expected struct `shame::vec3`, found struct `shame::vec4`

    feat.io.color::<RGB_888>().set(result.xyz()); // color target #1

    // add a 4x multisampling color target
    feat.io.color_ms::<RGBA_8888_sRGB, 4>().set(result); // color target #2

    // alpha blend `result` onto an `RGBA_8888_sRGB` color target
    feat.io.color::<RGBA_8888_sRGB>().blend(
        Blend::alpha(), //see the `Blend` type for more blend equations
        result
    ); // color target #3

    // `RGBA_Surface` can be used if the surface format is not yet known.
    // It can be replaced after recording.
    // (see `simple_wgpu` example for more details)
    feat.io.color::<RGBA_Surface>().set(result);

}



