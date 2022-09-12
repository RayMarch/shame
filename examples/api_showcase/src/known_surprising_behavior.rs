
use shame::prettify::syntax_highlight_glsl;
use shame::prelude::*;

#[derive(shame::Fields, Default)]
struct MyStruct {
    a: float4,
    b: float4,
}

#[derive(shame::Fields, Default)]
struct Vertex {
    pos: float4,
    nor: float4,
}

/// this example shows surprising behavior that is known
/// and either about to be fixed in future versions or just
/// inherent to how shame works. You might want to check this file
/// if you encounter strange behavior or just post in our discord.
#[keep_idents]
pub fn main() {

    // mutating a struct member or array element lvalue does not
    // narrow the stage of the struct/array
    //
    // this is planned to be fixed in a future version
    // it requires a larger refactor of the stage mechanism
    let out = shame::record_render_pipeline(|mut f| {

        let vertex: Vertex = f.io.vertex_buffer();
        let topo: TriangleList<u32> = f.io.index_buffer();
        let poly = f.raster.rasterize(vertex.pos.xyz1(), Cull::default(), topo);

        let fragment_pos = poly.plerp(vertex.pos);

        // create uniform value `my_struct`
        let mut foo = MyStruct::default().rec();

        foo.a += vertex.pos;   // `foo.a` becomes `Stage::Vertex`
        foo.b += fragment_pos; // `foo.b` becomes `Stage::Fragment`
        // the rust code implies that both modify the same struct, but they don't.
        // in the resulting shader code there are two `foo`, each with only one
        // of the mutations.
        // possible solution: the first assignment to `foo.a` should also affect
        // `foo` itself and turn it into a `Stage::Vertex` value. Then the 2nd
        // assignment could yield a nice error message. This is not possible
        // with the current implementation and requires some refactoring.

        let mut array = [float4::zero(); 5].rec();
        *array.at_const_mut(0) += vertex.pos;   //array[0] becomes `stage::Vertex`
        *array.at_const_mut(0) += fragment_pos; //array[0] becomes `stage::Fragment`
        // since the intermediate variable representing array[0] gets re-created
        // for each `at_const_mut` call, the stage information gets lost.
        // possible solution: the first assignment should also affect `array` itself
        // and change it to `Stage::Vertex`, that way the 2nd assignment could
        // yield a nice error message.
    });

    print_stages_main_only(&out);
}

fn print_stages_main_only(out: &RenderPipelineRecording) {
    let (vertex_shader, fragment_shader) = &out.shaders_glsl;

    let vertex_shader = vertex_shader.split_once("void main").unwrap().1;
    let fragment_shader = fragment_shader.split_once("void main").unwrap().1;

    println!("vertex_main {}", syntax_highlight_glsl(vertex_shader));
    println!("fragment_main {}", syntax_highlight_glsl(fragment_shader));
}