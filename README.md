<p align="center">
  <img width="270" src="https://github.com/RayMarch/shame_assets/blob/main/readme/beta_logo.png?raw=true" alt="logo"/>
</p>

# shame

## Shader metaprogramming in Rust

A lightweight [DSL] for writing GPU-pipelines in Rust

`shame` is designed for use in conjunction with the Metal/Vulkan/WebGPU generation of graphics APIs or [wgpu]/[SDL3] style libraries.

> This is a _beta_ release. Please open an issue if you encounter a problem. [Join our discord] and we will help you get started!

`shame` eliminates the need for a separate shader language and verbose descriptors,
without compromising on functionality, by fully embedding all shader/pipeline features into the Rust type system.

<p align="center">
  <img width="690" alt="a rust pipeline function is compiled to shaders and pipeline info" src="https://github.com/RayMarch/shame_assets/blob/main/readme/beta_codegen.png?raw=true">
</p>

By bridging the gap between CPU and GPU without leaving the Rust type system, `shame` empowers you to build concise graphics programs with a refreshing level of type safety.

[Metal]: https://developer.apple.com/metal/
[Vulkan]: https://www.vulkan.org
[WebGPU]: https://www.w3.org/TR/webgpu/
[SDL3]: https://wiki.libsdl.org/SDL3/CategoryGPU

## Motivation

Modern graphics APIs often require us filling out descriptors, matching up their contents with other descriptors, and restating the same facts in slightly different syntax many times. Doing this should be a compiler's job, but for the Rust compiler to be able to help us here, we first need to teach it how the GPU works. Unfortunately, most of the interesting information about the GPU's work is locked behind the shader language. To unlock it, we need to turn Rust itself into a shader language, but thats not enough. The real plumbing happens at the interface between CPU and GPU, which includes [fixed function] pipeline stages, [memory layouts] and [binding types]. Each of these has intricate interactions that we want Rust to understand, check, and ideally infer. Thankfully, Rust's features such as  ownership, traits and associated types allow us to express most of these interactions cleanly (and some not so cleanly). This leaves us with a thin Rust layer that faithfully models the GPU as Vulkan/Metal/WebGPU sees it, and that lets us write graphics pipelines in a more concise and modular way than ever before.

[fixed function]: https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions
[binding types]: https://docs.rs/wgpu/latest/wgpu/enum.BindingType.html
[memory layouts]: https://docs.vulkan.org/guide/latest/shader_memory_layout.html

<p align="center">
  <img width="430" alt="shame is a layer between your application and the graphics api's shader language/pipeline descriptors" src="https://github.com/RayMarch/shame_assets/blob/main/readme/beta_layers.png?raw=true">
</p>

## Getting started

1. set up an application using the graphics API of your choice (we recommend [wgpu])
2. add `shame` to your `Cargo.toml` via 
  ```
  cargo add --git https://github.com/raymarch/shame shame
  ```
3. convert `shame`'s output (shader code and pipeline info) to your graphics API
    > A [wgpu] example of this can be found at [examples/shame_wgpu/src/conversion.rs], which is used by the example applications. If you intend on targeting [wgpu] you can also use the [examples/shame_wgpu] crate directly. If you target a different API, that example can still be a useful reference.

4. check out the [api overview](examples/api_showcase/src/main.rs) of  `shame` and have fun writing pipelines in Rust!

[examples/api_showcase/src/main.rs]: examples/api_showcase/src/main.rs

[examples/shame_wgpu/src/conversion.rs]: examples/shame_wgpu/src/conversion.rs
[examples/shame_wgpu]: examples/shame_wgpu

## Code example (more in examples folder)

[(or click here for a more comprehensive overview of the api)](examples/api_showcase/src/main.rs)

```rust
use shame as sm;
use shame::aliases::*; // vector/matrix shorthands
use shame::texture_formats::*;

// define a render pipeline as a rust function
fn my_render_pipeline(mut drawcall: sm::DrawContext) {
    // Use `drawcall` to access gpu bound resources
    // and pipeline functionality.

    #[derive(sm::GpuLayout)]
    struct MyVertex {
        pos: f32x3,
        nor: f32x3,
    }
    let vb: sm::VertexBuffer<MyVertex> = drawcall.vertices.buffers.next();

    // fixed function vertex buffer lookup
    let vertex = vb.at(drawcall.vertices.index);

    // inferred push constant ranges from usage
    let matrix: f32x4x4 = drawcall.push_constants.get();

    // matrix and vector arithmetic
    let clip_pos = matrix * sm::vec!(vertex.pos, 1.0);

    // primitive assembly, then rasterize to fragments
    let fragments = drawcall
        .vertices
        .assemble(clip_pos, sm::Draw::triangle_strip(sm::Ccw))
        .rasterize(sm::Accuracy::default());

    // interpolate vertex normals for every fragment
    let frag_nor = fragments.fill(vertex.nor);

    // per fragment calculations - based on `frag_nor`
    let lighting = frag_nor.dot(f32x3::z()).clamp(0..);
    let color = sm::vec!(1.0, 0.5, 0.7, 1.0) * lighting;

    // access color attachments
    // (hide this in your own reusable "framebuffer"
    // structure which uses your specific target formats)
    let mut targets = fragments.attachments.color_iter();

    // alpha blending on the first color target.
    // Traits ensure that `Rgba8Unorm` is a valid
    // color target format which supports blending
    targets.next::<Rgba8Unorm>().blend(
        sm::Blend::alpha(), 
        color
    );

    // Rg8Unorm has only two channels.
    // Remap normals to the 0..1 range and drop the z coordinate
    targets.next::<Rg8Unorm>().set(
        frag_nor.xy().remap(-1..1, 0..1)
    );
}

fn main() -> Result<(), sm::EncodingErrors> {
    // use finished pipeline with your graphics api
    let result = {
        // `sm::Settings` configures various aspects of shader generation
        let mut encoder = sm::start_encoding(sm::Settings::default())?;
        my_render_pipeline(encoder.new_render_pipeline(sm::Indexing::default()));
        encoder.finish()?
    };

    println!("vertex shader: {}", result.shaders.vert_code);
    println!("fragment shader: {}", result.shaders.frag_code);
    println!("pipeline descriptor: {:?}", result.pipeline);

    // for debugging purposes print shader span information 
    // (= which line of rust code generated which line of 
    // shader code)
    dbg!(result.shaders.vert_code);
    Ok(())
}
// more examples in the examples folder!
```

Feature                  | Status |           |
------------------------ | ------ | --------- |
WGSL out                 | ✅     | supported |
Spir-V out               | ✖️     | currently unsupported, use [naga] to convert WGSL output to Spir-V 
render pipeline          | ✅     | supported
compute pipeline         | ✅     | supported
mesh pipeline            | ✖️     | curently unsupported 
vertex shaders           | ✅     | supported
fragment shaders         | ✅     | supported
compute shaders          | ✅     | supported
geometry shaders         | ✖️     | unsupported, not planned
tesselation shaders      | ✖️     | unsupported, not planned
mutable state            | ✅     | supported
runtime control flow     | ✅     | supported
memory layout validation | ✅     | supported
push constants           | ✅     | supported
atomics                  | ✅     | supported
writable storage         | ✅     | supported
wave/quad intrinsics     | ✖️     | currently unsupported, work in progress
🔥 hot reloading         | ✅     | supported

## Panic behavior
`shame` is intended for use in engine projects where panicking is unaccepable, as it
makes the runtime unable to respond to invalid pipelines (e.g. by displaying a 
fallback error texture). Great effort has been put into making `shame` never panic.
Both internal and API usage errors are communicated via `Result::Err`. 
Please open an issue if `shame` ever panics for you.

> note: The example executables in the `examples` folder may have `panic`s originating from wgpu if async validation errors happen.

## Community
We have a discord server where you can ask questions, give feedback, contribute or show your creations!
https://discord.gg/Xm5Ck7CCJk

## you might also like

- [wgpu](https://github.com/gfx-rs/wgpu): recommended graphics library for use with `shame`
- [naga]: if you want to convert `shame`'s WGSL output to SPIR-V etc.
- other ways to write shaders in Rust:
  - [rust-gpu](https://github.com/EmbarkStudios/rust-gpu): Rust as a first-class language and ecosystem for GPU graphics & compute shaders
  - [posh](https://github.com/leod/posh): OpenGL ES 3.0/WebGL 2.0 focused shader EDSL + graphics library
  - [cubecl](https://github.com/tracel-ai/cubecl): multi-platform high-performance compute language extension for Rust
  - [rasen](https://github.com/leops/rasen): generates SPIR-V bytecode from an operation graph (+ DSL)
  - [shades](https://github.com/phaazon/shades): a different approach of a shader EDSL in Rust
  - [rendiation](https://github.com/mikialex/rendiation): Rendiation Rendering Framework

[DSL]: https://en.wikipedia.org/wiki/Domain-specific_language#eDSL
[Join our discord]: https://discord.gg/eVkkxXgGcJ
[wgpu]: https://crates.io/crates/wgpu
[naga]: https://github.com/gfx-rs/naga

### license

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in shame by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
</sub>
