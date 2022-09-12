
use shame::*;

fn vec_matrix_example(mut feat: shame::RenderFeatures) {

    // turn a tuple into a float2 via .rec()
    let a = (1.0, 2.0).rec();

    // turn a (float2, float2) into a float4
    let b = (a, a).rec();

    // you can also mix floatN and f32
    let c = (1.0, a, 2.0).rec();

    //functions that accept `impl AsFloat4` can accept tuples directly, like the first argument of `rasterize_indexless`
    feat.raster.rasterize_indexless((1.0, 2.0, a), Default::default(), Default::default());

    //many uses of operators such as + - * / accept tuples or f32
    let d = a * (1.0, 2.0);
    let d = a + (1.0, 2.0);
    let d = a - (1.0, 2.0);
    let d = a / (1.0, 2.0);

    //access vector components via .x(), .y(), .z(), .w()
    let d = a.x() + a.y();

    //access parts of vectors with subsets of .xyzw()
    let d = c.xy();
    let d = c.yz();
    let d = c.xyz();
    let d = c.yzw();

    let mut e = (1.0, 2.0, 3.0).rec();
    //shorthands for appending a zero or one component to a vector
    let d = e.xyz0(); //same as `(e, 0.0).rec()`
    let d = e.xyz1(); //same as `(e, 1.0).rec()`
    let d = a.xy01(); //same as `(a, 0.0, 1.0).rec()`

    //generic zero()/one() function, will take the required shape (scalar, vec2, vec3, vec4)
    //TODO: fix the import. this shouldn't be exclusively accessible through prelude
    use shame::prelude::*;
    e = zero();
    e = one();
    e = id(); //identity wrt * operator (same as one() in the case of scalar/vector)

    //specific zero()/one() function, for when you want to add more type information
    let f = float3::one();
    let f = float4::zero();

    //create a float4x3 matrix (3 rows, 4 columns)
    let row_matrix = (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, 3.0),
    ).mat_rows();

    //create a float3x4 matrix (4 rows, 3 columns)
    let col_matrix = (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, 3.0),
    ).mat_cols();

    //mix f32 with float2, float4 etc.
    let mut row_matrix = (
        c,
        (0.0, 1.0, a),
        (a, a),
    ).mat_rows();

    row_matrix = zero(); //fill with zeroes
    row_matrix = one(); //fill with ones
    row_matrix = id(); //identity

}

pub fn main() {
    let out = shame::record_render_pipeline(vec_matrix_example);
    println!("{}", out.to_string_colored());
}