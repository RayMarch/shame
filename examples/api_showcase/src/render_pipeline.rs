
use shame::{*, prelude::Depth32};

pub fn main() {
    let out = shame::record_render_pipeline(my_render_pipeline);
    println!("{}", out.to_string_colored());
}

fn my_render_pipeline(mut feat: RenderFeatures) {

    // describe a vertex layout and `derive(shame::Fields)` to make it usable within shame.
    // only tensor types, such as `float`, `floatN` and `floatMxN` are valid vertex attributes.
    #[derive(shame::Fields)]
    struct Vertex {
        pos: float4x4,
        nor: float3,
        uv: float2,
    }
    let vertex: Vertex = feat.io.vertex_buffer();
    panic!("TODO: continue interleaved vs non-interleaved");

    // vertex inputs consist of vertex attributes, which can be read from
    // individual vertex buffers
    let v_pos: float3 = feat.io.vertex_buffer(); // 1st vertex buffer
    let v_nor: float3 = feat.io.vertex_buffer(); // 2nd vertex buffer
    let v_uv : float2 = feat.io.vertex_buffer(); // 3rd vertex buffer

    // or interleaved within one shared vertex buffer

    // derive shame::Fields will make this struct's layout usable in many shame functions.
    #[derive(shame::Fields)]
    struct PosNorUv {
        pos: float3,
        nor: float3,
        uv: float2,
    }

    // we can now use `PosNorUv` to describe a single vertex buffer with interleaved attributes pos, nor and uv.
    let vertex: PosNorUv = feat.io.vertex_buffer(); // 4th vertex buffer
    vertex.pos; // read from 4th vertex buffer
    vertex.nor; // also read from 4th vertex buffer
    vertex.uv;  // also read from 4th vertex buffer

    // choose an index buffer structure. 
    // both primitive topology and index datatype need to be specified
    // e.g. TriangleList<T>, TriangleStrip<T>
    let topo: TriangleList<u32> = feat.io.index_buffer();

    // additional inputs can be added via bind groups
    // create bind groups like this:
    let mut group0 = feat.io.group(); // bind group #0
    let mut group1 = feat.io.group(); // bind group #1
    // you can create bindgroups any time, they don't have to be created in one block.

    // you can add uniform blocks and read-only storage buffers to a group like so
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

    let culling = Cull::CW; //choose the winding order of the triangles you want to cull, or `Off` for disabling face culling
    
    let use_index_buffer = true;

    if !use_index_buffer {
        // if you don't want to use an index buffer, the primitive topology for the vertex inputs has to be 
        // specified separately.
        let primitive_topology = PrimitiveTopology::TriangleList;

        // call the rasterizer without index buffer
        let primitive = feat.raster.rasterize_indexless(clip_pos, culling, primitive_topology);
        return; // lets continue this example with an index buffer instead.
    }
    
    // rasterize at the clip space position, with the specified culling and index buffer.
    let primitive = feat.raster.rasterize(clip_pos, culling, topo);
    
    // interpolate the per-vertex position across the primitive to obtain a per-fragment position
    let frag_pos = primitive.lerp(v_pos);

    // there are different ways to interpolate across the primitive. 
    let frag_uv = primitive.lerp(v_uv); // linear interpolation
    let frag_uv = primitive.flat(v_uv); // flat interpolation (takes the value of the "provoking" vertex)
    let frag_uv = primitive.plerp(v_uv); // perspective aware interpolation (takes clip_pos.w into account)
    // of those three, `plerp` is the most commonly used.

    todo!("TODO: push constants");

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

    panic!("TODO: render targets");
    panic!("TODO: z-test");

}



