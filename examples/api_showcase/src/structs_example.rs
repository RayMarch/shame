
use shame::*;

pub fn structs_example(mut feat: shame::RenderFeatures) {

    #[derive(shame::Fields)]
    struct Foo {
        first: float4,
        second: float3,
        third: float2,
    }

    // use Foo to describe an interleaved vertex buffer with 3 attributes
    // location 0: first
    // location 1: second
    // location 2: third
    let vertex: Foo = feat.io.vertex_buffer();

    // create a bind group
    let mut group0 = feat.io.group();

    // use Foo to describe the contents of a uniform block
    let uniform_block: Foo = group0.uniform_block();

    // use Foo to describe the contents of a storage buffer
    let storage: Foo = group0.storage();

    // use Foo to describe the contents of a mutable storage buffer
    // accessing a mutable storage buffer is unsafe, therefore it is wrapped.
    let mutable_storage: UnsafeAccess<Foo> = group0.storage_mut();

    // this struct does not show up in the shader code as such. To the shader
    // only the creation of the individual fields as variables is visible.
    // therefore first, second and third are 3 separate vectors in the resulting shader code.
    let invisible_struct = Foo {
        first: float4::one(),
        second: float3::one(),
        third: float2::one(),
    };

    // calling .rec() turns our value into a shame::Struct<Foo> which is visible
    // in the resulting shader code
    let visible_struct = Foo {
        first: float4::one(),
        second: float3::one(),
        third: float2::one(),
    }.rec();

    //alternative, more explicit way of creating a shame::Struct<Foo>.
    //identical to the example above, but will yield better error messages when done wrong.
    let visible_struct = shame::Struct::new(Foo {
        first: float4::one(),
        second: float3::one(),
        third: float2::one(),
    });

    //flat composition feature of shame::Fields
    #[derive(shame::Fields)]
    struct PositionUvAnd<T: shame::Fields> {
        position: float4,
        t: T, //`t` itself is not visible in shader. Instead it behaves as if T's fields were "copy-pasted" in here directly
        uv: float2,
    }

    #[derive(shame::Fields)]
    struct NormalColor {
        normal: float3,
        color: float4,
    }

    #[derive(shame::Fields)]
    struct TBNColor {
        tangent_binormal_normal: float3x3,
        color: float4,
    }

    // interleaved: position, normal, color, uv
    let interleaved: PositionUvAnd<NormalColor> = feat.io.vertex_buffer();

    // interleaved: position, tangent_binormal_normal, color, uv
    let interleaved: PositionUvAnd<TBNColor> = feat.io.vertex_buffer();

    // interleaved: position, unnamed_float3, uv
    let interleaved: PositionUvAnd<float3> = feat.io.vertex_buffer();

    // interleaved: position, position2, unnamed_float4, uv, uv2
    let interleaved: PositionUvAnd<PositionUvAnd<float4>> = feat.io.vertex_buffer();

    // (calling rasterizer just so that this is a valid render pipeline)
    feat.raster.rasterize_indexless(float4::default(), Default::default(), Default::default());
}

pub fn main() {
    let out = shame::record_render_pipeline(structs_example);
    println!("{}", out.to_string_colored());
}