| <img src="https://user-images.githubusercontent.com/20972939/226926834-855b6d21-9241-47a4-8e1f-f6b62b97dac4.png" width="60" height="60" alt="exclamation point sign"/> | *A new major version is in development!* |
| :---: | --- |

<p align="center">
<img style="align: center" width="330" src="https://github.com/RayMarch/shame_assets/blob/main/readme/logo_with_bg.png?raw=true#gh-light-mode-only" alt="logo"/>
<img style="align: center" width="330" src="https://github.com/RayMarch/shame_assets/blob/main/readme/logo_with_bg_dark.png?raw=true#gh-dark-mode-only" alt="logo"/>
</p>

# **shame**

## shader metaprogramming in **rust**

`shame` lets you write **shader recipes** in rust, which can be turned into shaders and pipelines at runtime!

You can generate parts of the shaders based on runtime conditions, use the rust type system, output entire pipeline layouts and more!

`shame` is very experimental!

![a single rust function generates vertex/fragment shaders and pipeline information](https://github.com/RayMarch/shame_assets/blob/main/readme/overview_with_bg.png?raw=true#gh-light-mode-only)
![a single rust function generates vertex/fragment shaders and pipeline information](https://github.com/RayMarch/shame_assets/blob/main/readme/overview_with_bg_dark.png?raw=true#gh-dark-mode-only)

`shame` pipelines are written as small rust functions which are "recorded" and return...

- ...shader `String`s
  - vertex/fragment or compute shaders
- ...pipeline layout information
  - face culling,
  - z-test/z-write,
  - blending,
  - index format/primitive topology,
  - vertex attributes and buffer layouts,
  - color/depth targets and formats,
  - bind group layouts,
  - push constant type

## Code example (more in examples folder)

```rust
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

fn main() {
    // generate the shaders and pipeline layout
    let out = shame::record_render_pipeline(my_render_pipeline);
    
    let (vertex_shader, fragment_shader) = &out.shaders_glsl;
    let pipeline_info = &out.info;

    println!("{out}");
}

// more examples in the examples folder!
```


Feature       |     Status         |  |
--------------- | ------------------ | ---- |
GLSL (Vulkan) out | ✅🚧 | supported, not feature complete yet
GLSL (OpenGL) out | ✖️✋  | will probably implement if people ask for it |
WGSL out           | 🗓️  | planned |
render pipeline layout | ✅ | supported, not feature complete yet
compute pipeline layout | ✅ | supported
vertex shaders | ✅ | supported
fragment shaders | ✅ | supported
compute shaders | ✅ | supported
geometry shaders | ✖️ | unsupported, not planned
hull/domain shaders | ✖️ | unsupported, not planned
🔥 hot reloading | ✅ | supported, see examples

## Community
We have a discord server where you can ask questions, give feedback, contribute or show your creations!
https://discord.gg/eVkkxXgGcJ

## you might also like

- [wgpu](https://github.com/gfx-rs/wgpu): recommended graphics library for use with `shame`
- [naga](https://github.com/gfx-rs/naga): if you want to convert `shame`'s GLSL output to SPIR-V etc.
- other ways to write shaders in rust:
  - [rust-gpu](https://github.com/EmbarkStudios/rust-gpu): rust as a first-class language and ecosystem for GPU graphics & compute shaders
  - [rasen](https://github.com/leops/rasen): generates SPIR-V bytecode from an operation graph (+ DSL)
  - [shades](https://github.com/phaazon/shades): a different approach of a shader EDSL in rust
  - [rendiation](https://github.com/mikialex/rendiation): Rendiation Rendering Framework

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
