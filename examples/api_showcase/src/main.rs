#![allow(unused, clippy::no_effect)]
use shame as sm;
use shame::prelude::*;
use shame::aliases::*;

#[rustfmt::skip]
fn make_pipeline(some_param: u32) -> Result<sm::results::RenderPipeline, sm::EncodingErrors> {

    // start a pipeline encoding with the default settings.
    // (in the `shame_wgpu` examples, this is wrapped by the `Gpu` object)
    //
    // compared to earlier versions of `shame`, pipeline
    // encoding is no longer based on a closure, but based on a
    // RAII guard `sm::EncodingGuard<...>` instead.
    // That way you can use the `?` operator for your own
    // non-shame errors that occur during the pipeline encoding.
    // If you like the closure-style better you can still wrap this
    // RAII guard in a higher order function.
    let mut encoder = sm::start_encoding(sm::Settings::default())?;

    // choose either a `render` or `compute` pipeline
    // the returned object represents the GPU's state when the drawcall happens.
    let mut drawcall = encoder.new_render_pipeline(sm::Indexing::BufferU16);

    // compute pipelines are created like so: (a compute example can be found further down)
    // let mut dispatch = encoder.new_compute_pipeline([8, 4]);

    &drawcall.vertices; // access to vertex-shader related functionality
    &drawcall.bind_groups; // access to bind groups (descriptor-sets)
    &drawcall.push_constants; // access to push constant data

    // import vertex buffers with the `vertices.buffers` iterator, or by index via `.index(_)`
    let vbuf: sm::VertexBuffer<f32x3> = drawcall.vertices.buffers.next();

    // A vertex buffer can be looked up exactly once (consuming) by a
    // vertex index, which is either `vertices.id` or `vertices.instance_id`
    let pos = vbuf.index(drawcall.vertices.index);

    // or as a one-liner
    let uv: f32x2 = drawcall.vertices.buffers.next().index(drawcall.vertices.index);

    // a single "derive" implements all relevant traits (if they apply)
    // e.g.
    //    `GpuLayout` - for byte layout compatibility checks
    //     `GpuSized` - if all fields are `GpuSized`
    //   `GpuAligned` - if all fields are `GpuAligned`
    // `VertexLayout` - if all fields are vec or packed-vec
    //      `NoBools` - if there are no bools (see WGSL "HostShareable")
    //    `NoAtomics` - if there are no atomics (for instantiable structs)
    // etc...
    #[derive(sm::GpuLayout)]
    #[cpu(TransformsOnCpu)] // (optional) specify a corresponding CPU type for layout checks
    struct Transforms {
        world: f32x4x4,
        view: f32x4x4,
        proj: f32x4x4,
    }

    // this struct contains "packed" vectors (snorm, unorm etc.) which are
    // only supported in vertex buffers, not storage/uniform buffers.
    // This is reflected in the traits that are derived for `MyVertexFormat`.
    //
    // note: vertex layouts support #[gpu_repr(packed)] to prevent padding
    // between fields of a struct, which often happens with 3 dimensional vectors
    #[derive(sm::GpuLayout)]
    struct MyVertexFormat {
        nor: sm::packed::snorm16x2,
        uv: sm::packed::unorm8x2,
        pos: f32x3,
    }
    let vertex_data: MyVertexFormat = drawcall.vertices.buffers.next().index(drawcall.vertices.index);

    // iterator-based import of bind groups. Alternatively by index.
    let mut group0 = drawcall.bind_groups.next();

    // import storage or uniform buffers via the `group0` iterator
    //
    // these iterators exist so that you can abstract them away in your own
    // api-specific layer that suits your needs such that you can represent
    // bind groups as types.
    //
    // `Transforms` is checked for compatibility with `TransformsOnCpu` here.
    // The check happens at shader-generation time, so that a nice error
    // message can be generated, pointing to the field that doesn't match.
    // (once rusts const-generics are more powerful this may be moved to compile-time)
    let xforms_sto: sm::Buffer<Transforms, sm::mem::Storage> = group0.next();
    let xforms_uni: sm::Buffer<Transforms, sm::mem::Uniform> = group0.next();

    // conditional code generation based on pipeline parameter
    if some_param > 0 {
        // if not further specified, defaults to `sm::mem::Storage`
        let xforms_sto2: sm::Buffer<Transforms> = group0.next();
    }

    // result types of matrix multiplications are inferred
    let xform = xforms_sto.proj * xforms_sto.view * xforms_sto.world;

    // here are some examples of how vector and matrix types behave
    let my_vec3 = sm::vec!(1.0, 2.0, 3.0);
    let my_vec4 = sm::vec!(my_vec3, 0.0); // component concatenation, like usual in shaders
    let my_vec4 = my_vec3.extend(0.0); // or like this

    let my_normal = sm::vec!(1.0, 1.0, 0.0).normalize();
    let rgb = my_normal.remap(-1.0..=1.0, 0.0..=1.0); // remap linear ranges (instead of " * 0.5 + 0.5")

    let alpha = 0.4.to_gpu(); // convert from rust to `shame` types (also works for arrays and structs)
    let smooth: f32x1 = alpha.smoothstep(0.4..0.8);

    // clamp as generalized min, max, clamp via half open ranges
    let upper = alpha.clamp(..=0.8);
    let lower = alpha.clamp(0.1..);
    let both  = alpha.clamp(0.1..=0.8);

    // reverse subtraction
    let k = (1.0 - (1.0 - alpha).sqrt());
    let k = alpha.rsub(1.0).sqrt().rsub(1.0); // same as above
    let k = alpha.rsub1().sqrt().rsub1(); // same as above

    // iterate over components of vec
    let sum: f32x1 = my_vec4.into_iter().map(|x| x * x).sum();

    let linear = xform.resize() as f32x3x3; // generic matrix resize
    let linear = linear * sm::mat::id(); // generic identity matrix

    // generic zero, one
    let z: f32x3 = sm::zero();
    let z: f32x3 = sm::one();

    // generic unit vectors for coordinate axes x, y, z, w
    let to_light_3d: f32x3 = sm::vec::y();
    let to_light_2d: f32x2 = sm::vec::y();

    // matrix constructors
    let basis = sm::mat::from_cols([
        sm::vec!(2.0, 0.0, 0.0), // scale x by 2.0
        sm::vec!(0.0, 1.0, 0.0),
        sm::vec!(0.0, 0.0, 1.0),
        sm::vec!(4.0, 5.0, 6.0), // translate by (4, 5, 6)
    ]);

    let basis = sm::mat::from_rows([
        sm::vec!(2.0, 0.0, 0.0, 4.0),
        sm::vec!(0.0, 1.0, 0.0, 5.0),
        sm::vec!(0.0, 0.0, 1.0, 6.0),
    ]);

    // from elements (column major)
    let basis = sm::f32x4x3::new([
        2.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        4.0, 5.0, 6.0,
    ]);

    // mutable state requires interior mutability via `sm::Cell`
    let k = sm::Cell::new(1i32);
    k.set(4);
    let l = k.get();

    // unlike `std::cell::Cell`, `sm::Cell` lets you mutate fields individually
    let my_vec = sm::Cell::new((1.0, 2.0, 3.0));
    my_vec.y.set(3.0);

    // rusts `if/while/for` means conditional code generation and loop-unrolling,
    //
    // therefore shader control flow uses closures.
    // The api is designed after rusts `bool::then` and the likes:

    let rust_bool = true;
    // condition doesn't appear in the shader, only `k = 1`
    rust_bool.then(|| k.set(1));

    let shame_bool = true.to_gpu();
    // condition appears in the shader as `if true { k = 1 }`
    // (the `move` keyword is required for safety,
    // but can be turned off via a shame crate-feature)
    shame_bool.then(move || k.set(1));

    // a variety of different shader-loop functions also exist

    // for loop doesn't appear in the shader, only `k = 1; k = 2; k = 3; ...`
    for i in 0..=10 {
        k.set(i)
    }

    // for loop appears in the shader
    sm::for_range(0..=10, move |i| {
        k.set(i)
    });

    // `.to_gpu()` can also be used as a vec constructor (used by sm::vec! macro internally)
    let clip_pos = xform * (pos, 0.0).to_gpu();

    let primitive = drawcall.vertices.assemble(
        clip_pos,
        // the Draw enum offers some convenience ctors for common seetings
        // draw only the counter clockwise triangles:
        sm::Draw::triangle_list(sm::Winding::Ccw),
    );

    let mirror_distances = [pos.x - 4.0, -pos.y].to_gpu();

    // clipping planes
    let primitive = primitive.clip(mirror_distances);

    // use `rasterize`, `rasterize_multisample` or `rasterize_supersample`.
    // Rasterization gives access to the fragment-stage api via `frag`
    let frag = primitive.rasterize(sm::Accuracy::Relaxed);

    // vertex/fragment perspective interpolators for `pos` and `uv`
    // (other calls exist for flat/linear interpolation, see `fill` documentation)
    let (pos, uv) = frag.fill((pos, uv));

    // samplers are generic over their capabilities (Nearest/Filtering/Comparison)
    let sampler: sm::Sampler<sm::Filtering> = group0.next();

    // textures are generic too:
    // Texture<Format, Coords = f32x2, SPP = Single>
    // (spp = samples per pixel is either `sm::Single` or `sm::Multi`)
    let texture: sm::Texture<sm::tf::Rg8Unorm> = group0.next();

    // fragment-quad based derivative of `uv`
    let duv = frag.quad.grad(uv, sm::GradPrecision::Fine);

    // individual partial derivatives are accessible as fields.
    duv.dx;
    duv.dy;

    // `fwidth` was renamed to `dxy_manhattan` because "width" is misleading
    frag.quad.dxy_manhattan(uv, sm::GradPrecision::Coarse);

    // sampling functions in `shame` require
    // - sampler
    // - texture
    // - a mipmap function `MipFn` which selects the mipmap level
    // - texture coordinates

    // sample mipmap level zero at uv.
    // returns f32x2 because of the RG texture format.
    let rg = sampler.sample(texture, sm::MipFn::zero(), uv);

    // use the fragment-quad based derivative for mipmap calculation.
    // (this is the default in many shading langs)
    let rg = sampler.sample(texture, sm::MipFn::Quad(frag.quad), uv);
    // or a shorthand
    let rg = sampler.sample(texture, frag.quad.into(), uv);

    // whether to replace the existing depth values if the depth test passes
    let replace_depth_on_pass = true;

    // the fragment object can be used for depth/stencil testing once.
    // tests can also be skipped by calling `attachments.color_iter()` instead.
    // Access to the color target iterator is returned.
    let mut targets = frag
        .attachments
        .depth_test::<sm::tf::Depth24Plus>(sm::DepthTest::less_equal(replace_depth_on_pass));

    // the `targets` iterator is also something you probably want to
    // abstract in your api specific layer, so that a "Framebuffer-struct"
    // can be conveniently used and re-used for every pipeline that shares it.
    let albedo: sm::ColorTarget<sm::tf::Rgba8Unorm> = targets.next();
    // blending op is specified here, as well as the source-color for blending
    albedo.blend(sm::Blend::alpha(), (rg, 0.0, 1.0));

    // here we don't blend, and write to an RG target, which takes a f32x2
    targets.next::<sm::tf::Rg8Unorm>().set(rg);

    // `targets` is a `ColorTargetIter<Single>` because we chose single-sample rasterization.
    // features like alpha-to-coverage are exposed on multisample color targets only.
    // `shame` makes it impossible to use these features if they don't apply to the situation.

    // this causes a compiler error, because Rg8Unorm has no alpha channel:
    // targets.next::<tfmt::Rg8Unorm>().set_with_alpha_to_coverage(rg);

    // finish the encoding and obtain the pipeline setup info + shader code.
    encoder.finish()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // create your pipeline with `some_param = 4`
    let my_pipeline = make_pipeline(4);

    // note: shame will not panic.
    // If shame ever panics that is a bug, please report that as a github issue.
    //
    // shame is designed to be part of a game engine, and as such it would be
    // very disruptive, even in the case of internal errors, if the program panicked.
    //
    // Therefore shame even treats its own internal errors as runtime errors and
    // propagates them via `Result`, so you can fallback to a previous, working shader
    // or display a pink texture.

    // normally you would feed `my_pipeline` directly into your graphics api layer
    // to automatically generate e.g. a `wgpu::RenderPipeline` for you.
    // but lets take a look whats inside:
    match my_pipeline {
        Err(e) => {
            // `e` contains encoding errors. They are similar to shader compiler errors,
            // Some things cannot be validated in the rust typesystem, so reporting
            // them like this is the next best thing.

            // by default, the `Display` trait of encoding errors will access
            // your rust files to show a small excerpt of the rust code that caused
            // the error.
            // If shame doesn't find the .rs file, it will display only the error
            // without an excerpt.
            // To prevent `shame` from even trying to read the .rs file (for example
            // because you're shipping your binary to users) you can set
            // `sm::Settings::error_excerpt` to `false` for your release builds.
            Err(e)
        }
        Ok(p) => {
            // lets look at the shaders
            // `syntax_highlight` makes it nicer to read on the terminal
            println!(
                "vertex shader: {}\n fragment shader:{}",
                p.shaders.vert_code.syntax_highlight(),
                p.shaders.frag_code.syntax_highlight(),
            );

            // certain target languages, such as WGSL support a single shader
            // string that contains both vertex and fragment shader. To avoid
            // having to compile the same code twice, use `into_shared_shader_code`
            // which outputs a single code object if vertex and fragment shaders
            // share the same code string, and two objects if they don't.

            // match p.shaders.into_shared_shader_code() {
            //     Ok(both_code) => {}
            //     Err((frag_code, vert_code)) => {}
            // }

            // `Debug` will print the shader code interleaved with links to
            // the rust-code location for every expression
            dbg!(&p.shaders.vert_code);

            // shader stage entry point function names
            &p.shaders.vert_entry_point;
            &p.shaders.frag_entry_point;

            // "origin_spans" contains a mapping from
            // shader file/line/column to your rust code's file/line/column.
            //
            // you can hook this up to your shader compiler or other tools,
            // so you can jump right into the originating rust line of code
            // in case of a later error, instead of having to read generated shader code.
            p.shaders.vert_code.origin_spans();
            p.shaders.frag_code.origin_spans();
            // ideally this never happens, `shame` tries its best to always
            // generate valid shader code or give you nice encoding errors,
            // but in the unfortunate case where this does happen you can use
            // the spans to quickly find the cause.

            // the additional pipeline information is here
            &p.pipeline.vertex_buffers;
            &p.pipeline.bind_groups;
            &p.pipeline.color_targets;
            &p.pipeline.rasterizer;
            //...
            Ok(p)
        }
    }?;
    Ok(())
}

fn make_compute_pipeline(side_len: u32) -> Result<sm::results::ComputePipeline, sm::EncodingErrors> {
    let mut encoder = sm::start_encoding(sm::Settings::default())?;

    /// create a compute pipeline, where each workgroup consists of
    /// `side_len * side_len` threads, arranged as a 2D square.
    let dispatch = encoder.new_compute_pipeline([side_len, side_len]);

    // alternatively, compute dispatches can also be 1D or 3D, depending on
    // the amount of dimensions you provide:
    //
    // encoder.new_compute_pipeline([32]);      // 1D dispatch
    // encoder.new_compute_pipeline([8, 4]);    // 2D dispatch
    // encoder.new_compute_pipeline([4, 4, 2]); // 3D dispatch

    // all the compute-grid positions and sizes are of the appropriate
    // dimensionality right out of the box, no need for `.xy()` to discard the
    // z component
    let position: u32x2 = dispatch.thread_pos;
    // if you're interested, this gist explains the reasoning behind the
    // naming scheme of `shame` dispatch ids: https://gist.github.com/RayMarch/8e258008211e408a6cf73b63c46cc97b
    // and why it doesn't use terms like "invocation", "subgroup" and "global"

    // access to workgroup and wave functionality is given by the respective
    // fields in `dispatch`:
    let workgroup = &dispatch.workgroup;
    let wave = &dispatch.wave;

    // vectors allow infallible reduce on their components
    let num_threads_per_wg = workgroup.thread_grid_size.reduce(|a, b| a * b);

    // addition and multiplication reduce has shorthands
    let num_threads_per_wg = workgroup.thread_grid_size.comp_product();
    let manhattan_length = position.comp_sum();

    // access to bind groups and push constants works the same as in a render
    // pipeline, see the render pipeline example above for more info.
    &dispatch.bind_groups;
    &dispatch.push_constants;

    // barriers for synchronization
    sm::barrier::storage();
    sm::barrier::texture();
    sm::barrier::workgroup();

    // wave (=subgroup) functions are accessible via the wave object
    dispatch.wave.thread_count; // likely 32
    dispatch.wave.thread_id; // 1D id from 0 to wave.thread_count

    // soon there will be more wave intrinsics accessible via the `wave` object
    // (or maybe they're already supported and i forgot to update this comment)
    //
    // dispatch.wave.max(position)

    // mutable state uses interior mutability via `sm::Cell`
    // see more examples of this in the render pipeline example above
    let per_thread_i32 = sm::Cell::new(1i32);

    // define a struct layout
    // (for more info on structs and `GpuLayout`, see the render pipeline
    // example above, this example will go into the specifics of how atomics
    // are handled)
    #[derive(sm::GpuLayout)]
    struct DataWithAtomics {
        a: f32x2,
        b: sm::Atomic<u32>, // `Atomic<T>` has the same layout as `vec<T, x1>`
        c: sm::Atomic<i32>,
    }

    // workgroup local mutable memory is obtained like this.
    // Here there's one u32 per workgroup, so all threads that read/write to
    // `per_workgroup_u32` will access the same single 4-byte u32 memory cell.
    let per_workgroup_u32 = sm::mem::workgroup_local::<u32x1>();

    // workgroup shared memory can contain atomics, or structs that contain atomics
    // within them.
    // here is how to create workgroup local memory that contains atomics via
    // default construction, which uses zeroed values (see https://www.w3.org/TR/WGSL/#zero-value-builtin-function)
    let wg_atomic_u32 = sm::mem::workgroup_local::<sm::Atomic<u32>>();

    // load and store similar to rust's `std::sync::atomic`s
    let val = wg_atomic_u32.load();
    wg_atomic_u32.store(4u32);
    let before = wg_atomic_u32.fetch_add(1u32);

    let wg_data = sm::mem::workgroup_local::<sm::Struct<DataWithAtomics>>();
    wg_data.b.store(1u32);

    // rusts `if/while/for` means conditional code generation and loop-unrolling,
    //
    // therefore shader control flow uses closures.
    // The api is designed after rusts `bool::then` and the likes:

    let rust_bool = true;
    // condition doesn't appear in the shader, only `per_thread_data = 1`
    rust_bool.then(|| per_thread_i32.set(1));

    // .to_gpu() converts to gpu-equivalent type, see render pipeline example.
    let shame_bool = true.to_gpu();
    // condition appears in the shader as `if true { per_thread_data = 1 }`
    // (the `move` keyword is required for safety,
    // but can be turned off via a shame crate-feature)
    shame_bool.then(move || per_thread_i32.set(1));

    // for more information on for/while loops that appear in the shader,
    // see the render pipeline example above.

    encoder.finish()
}

// a struct on the cpu which can be layout-checked against a GpuLayout type
#[derive(sm::CpuLayout)]
#[repr(C)] // enforced by derive macro
struct TransformsOnCpu {
    world: Mat4,
    view: Mat4,
    proj: Mat4,
}

#[repr(C, align(8))]
struct Mat2([[f32; 2]; 2]);

// tell `shame` about the layout semantics of your cpu types
// Mat2::layout() == sm::f32x2x2::layout()
impl sm::CpuLayout for Mat2 {
    fn cpu_layout() -> sm::TypeLayout { sm::gpu_layout::<sm::f32x2x2>() }
}

#[repr(C, align(16))]
struct Mat4([[f32; 4]; 4]);
impl sm::CpuLayout for Mat4 {
    fn cpu_layout() -> sm::TypeLayout { sm::gpu_layout::<sm::f32x4x4>() }
}

// using "duck-traiting" allows you to define layouts for foreign cpu-types,
// sidestepping the orphan-rule:

// use glam::Mat4;

// // declare your own trait with a `layout()` function like this
// // This function will be used by the `derive(GpuLayout)` proc macro
// pub trait MyCpuLayoutTrait {
//     fn layout() -> shame::TypeLayout;
// }

// // tell `shame` about the layout semantics of `glam` types
// impl MyCpuLayoutTrait for glam::Mat4 {
//     fn layout() -> shame::TypeLayout { sm::gpu_layout::<sm::f32x4x4>() }
// }
