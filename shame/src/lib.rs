//#![deny(missing_docs)] //TODO: reenable
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::bare_urls)]

//! [`shame`] - shader metaprogramming
//!
//! an EDSL for writing render and compute pipelines in rust.
//!
//! see the examples folder for how to use [`shame`].

/// how `shame`'s proc macros access `shame_graph`
pub mod shame_reexports {
    //required for proc macros to access shame_graph
    pub use crate as shame;
    pub use shame_graph;
}

pub mod keep_idents;
pub mod pipeline;
pub mod prelude;
pub mod rec;
pub mod shader;

mod assert;
pub mod prettify;
mod wrappers;

pub use shame_graph::Any;

pub use shader::Primitive;

pub use shader::current_shader;
pub use shader::record_compute_shader;
pub use shader::record_render_shaders;

pub use pipeline::compute_pipeline::record_compute_pipeline;
pub use pipeline::render_pipeline::record_render_pipeline;

pub use pipeline::compute_pipeline::ComputePipelineRecording;
pub use pipeline::render_pipeline::RenderPipelineRecording;

pub use pipeline::topology::TriangleList;
pub use pipeline::topology::TriangleStrip;

pub use pipeline::pixel_format::ColorFormat;
pub use pipeline::pixel_format::DepthFormat;
pub use pipeline::pixel_format::IsColorFormat;
pub use pipeline::pixel_format::IsDepthFormat;
pub use pipeline::pixel_format::IsPixelFormat;
pub use pipeline::pixel_format::PixelFormat;
pub use pipeline::topology::IndexFormat;
pub use pipeline::topology::PrimitiveIndex;

pub use rec::fields::Fields;
pub use rec::IntoRec;
pub use rec::MatCtor;

pub use rec::aliases::*;
pub use rec::Array;
pub use rec::Struct;
pub use rec::Ten;

pub use rec::sampler_texture::{Sampler, TexSampleType, Texture, TextureR, TextureRG, TextureRGB, TextureRGBA};

pub use rec::texture_combined_sampler::{
    CombineSampler, CombineSamplerR, CombineSamplerRG, CombineSamplerRGB, CombineSamplerRGBA, TexCoordType,
};

pub use wrappers::UnsafeAccess;

pub use shader::ComputeShaderFeatures;
pub use shader::RenderShaderFeatures;

pub use shame_derive::device;
pub use shame_derive::host;
pub use shame_derive::keep_idents;
pub use shame_derive::Fields;

#[cfg(feature = "mirror")]
mod mirror;

pub use pipeline::compute_pipeline::ComputeFeatures;
pub use pipeline::render_pipeline::RenderFeatures;

pub use pipeline::compute_pipeline_info::ComputePipelineInfo;
pub use pipeline::render_pipeline_info::AttributeInfo;
pub use pipeline::render_pipeline_info::BindGroupInfo;
pub use pipeline::render_pipeline_info::PushConstantInfo;
pub use pipeline::render_pipeline_info::RenderPipelineInfo;
pub use pipeline::render_pipeline_info::VertexBufferInfo;
pub use pipeline::render_pipeline_info::VertexStepMode;
pub use pipeline::BindingInfo;
pub use pipeline::BindingType;
pub use pipeline::StageFlags;

pub use pipeline::GenericPipelineIO;

pub use rec::*;

pub mod tensor_type {
    //!additional module for those imports to avoid confusion with Tensor trait
    pub use shame_graph::DType;
    pub use shame_graph::Shape;
    pub use shame_graph::Tensor;
}

pub use pipeline::blending::{Blend, BlendEquation, BlendFactor, BlendOp};
pub use pipeline::culling::Cull;
pub use pipeline::target::DepthTest;
pub use pipeline::target::DepthWrite;
pub use pipeline::topology::IndexDType;
pub use pipeline::topology::PrimitiveTopology;

pub use control_flow::{
    break_, break_if, continue_, continue_if, discard_fragment, discard_fragment_if, for_range, for_range_step,
};
pub use pipeline::pixel_format;
pub use shame_reexports::shame_graph::ShaderKind;
