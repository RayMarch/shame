//#![deny(missing_docs)] //TODO: reenable
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::bare_urls)]

//! [`shame`] - shader metaprogramming
//!
//! an EDSL for writing render and compute pipelines in rust.
//!
//! see the examples folder for how to use [`shame`].

/// how `shame`'s proc macros access `shame_graph`
pub mod shame_reexports { //required for proc macros to access shame_graph
    pub use shame_graph;
    pub use crate as shame;
}

pub mod rec;
pub mod prelude;
pub mod keep_idents;
pub mod shader;
pub mod pipeline;

mod assert;
mod wrappers;
pub mod prettify;

pub use shame_graph::Any;

pub use shader::Primitive;

pub use shader::current_shader;
pub use shader::record_render_shaders;
pub use shader::record_compute_shader;

pub use pipeline::render_pipeline::record_render_pipeline;
pub use pipeline::compute_pipeline::record_compute_pipeline;

pub use pipeline::render_pipeline::RenderPipelineRecording;
pub use pipeline::compute_pipeline::ComputePipelineRecording;

pub use pipeline::topology::TriangleList;
pub use pipeline::topology::TriangleStrip;

pub use pipeline::topology::PrimitiveIndex;
pub use pipeline::topology::IndexFormat;
pub use pipeline::pixel_format::PixelFormat;
pub use pipeline::pixel_format::DepthFormat;
pub use pipeline::pixel_format::ColorFormat;
pub use pipeline::pixel_format::IsPixelFormat;
pub use pipeline::pixel_format::IsDepthFormat;
pub use pipeline::pixel_format::IsColorFormat;

pub use rec::fields::Fields;
pub use rec::MatCtor;
pub use rec::IntoRec;

pub use rec::Struct;
pub use rec::Array;
pub use rec::Ten;
pub use rec::aliases::*;

pub use rec::sampler_texture::{
    Sampler,
    Texture,

    TextureR,
    TextureRG,
    TextureRGB,
    TextureRGBA,

    TexSampleType,
};

pub use rec::texture_combined_sampler::{
    CombineSampler,

    CombineSamplerR,
    CombineSamplerRG,
    CombineSamplerRGB,
    CombineSamplerRGBA,

    TexCoordType,
};

pub use wrappers::UnsafeAccess;

pub use shader::ComputeShaderFeatures;
pub use shader::RenderShaderFeatures;

pub use shame_derive::Fields;
pub use shame_derive::host;
pub use shame_derive::device;
pub use shame_derive::keep_idents;

#[cfg(feature = "mirror")] mod mirror;

pub use pipeline::compute_pipeline::ComputeFeatures;
pub use pipeline::render_pipeline::RenderFeatures;

pub use pipeline::render_pipeline_info::RenderPipelineInfo;
pub use pipeline::compute_pipeline_info::ComputePipelineInfo;
pub use pipeline::render_pipeline_info::BindGroupInfo;
pub use pipeline::BindingInfo;
pub use pipeline::BindingType;
pub use pipeline::render_pipeline_info::PushConstantInfo;
pub use pipeline::StageFlags;
pub use pipeline::render_pipeline_info::VertexBufferInfo;
pub use pipeline::render_pipeline_info::VertexStepMode;
pub use pipeline::render_pipeline_info::AttributeInfo;

pub use pipeline::GenericPipelineIO;

pub use rec::*;

pub mod tensor_type {
    //!additional module for those imports to avoid confusion with Tensor trait
    pub use shame_graph::Tensor;
    pub use shame_graph::Shape;
    pub use shame_graph::DType;
}

pub use pipeline::topology::PrimitiveTopology;
pub use pipeline::topology::IndexDType;
pub use pipeline::culling::Cull;
pub use pipeline::target::DepthTest;
pub use pipeline::target::DepthWrite;
pub use pipeline::blending::{
    Blend,
    BlendEquation,
    BlendFactor,
    BlendOp,
};

pub use pipeline::pixel_format;
pub use shame_reexports::shame_graph::ShaderKind;
pub use control_flow::{
    for_range,
    for_range_step,
    continue_,
    continue_if,
    break_,
    break_if,
    discard_fragment,
    discard_fragment_if,
};