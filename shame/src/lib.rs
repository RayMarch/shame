#![doc = include_str!("../../README.md")]
#![forbid(unsafe_code)]
//#![warn(clippy::cast_lossless)]
#![deny(missing_docs)]
#![allow(clippy::match_like_matches_macro, clippy::diverging_sub_expression)]
#![allow(unused)]

mod backend;
mod common;
mod frontend;
mod ir;

// public interface starts here:
pub mod prelude;

// pipeline encoding
pub use frontend::encoding::start_encoding;
pub use frontend::encoding::Settings;
pub use backend::language::Language;

pub use frontend::encoding::EncodingGuard;

/// marker types for different pipeline kinds.
///
/// Used in [`crate::EncodingGuard`]
pub mod pipeline_kind {
    pub use crate::frontend::encoding::pipeline_kind::IsPipelineKind;
    pub use crate::frontend::encoding::pipeline_kind::Render;
    pub use crate::frontend::encoding::pipeline_kind::Compute;
}
pub use frontend::encoding::EncodingError;
pub use frontend::encoding::EncodingErrors;
pub use frontend::encoding::ThreadIsAlreadyEncoding;
pub use frontend::encoding::features::DrawContext;
pub use frontend::encoding::features::DispatchContext;

// # render pipeline

// vertex stage
pub use frontend::encoding::features::Indexing;
pub use frontend::encoding::rasterizer::VertexStage;
pub use frontend::encoding::rasterizer::VertexIndex;
pub use frontend::encoding::io_iter::VertexBuffer;
pub use frontend::encoding::io_iter::VertexBufferIter;

// primitive assembly
pub use frontend::encoding::rasterizer::PrimitiveAssembly;
pub use frontend::encoding::rasterizer::Draw;
pub use frontend::encoding::rasterizer::Winding;
pub use frontend::encoding::rasterizer::Winding::Ccw;
pub use frontend::encoding::rasterizer::Winding::Cw;
pub use frontend::encoding::rasterizer::ZClip;

// rasterize
pub use frontend::encoding::rasterizer::Accuracy;
pub use frontend::encoding::mask::BitVec64;
pub use frontend::encoding::mask::BitVec64Iter;

// fragment stage
pub use frontend::encoding::rasterizer::FragmentStage;
pub use frontend::encoding::fill::PrimitiveInterpolatable;

pub use frontend::encoding::rasterizer::FragmentQuad;
pub use frontend::encoding::rasterizer::Gradient;
pub use ir::GradPrecision;
pub use frontend::encoding::rasterizer::GradUnit;

pub use frontend::encoding::fill::Fill;
pub use frontend::encoding::fill::PickVertex;

pub use frontend::encoding::flow::discard;
pub use frontend::encoding::flow::discard_if;

// fragment tests
pub use frontend::texture::texture_traits::DepthStencilFormat;
pub use frontend::texture::texture_traits::DepthFormat;
pub use frontend::texture::texture_traits::StencilFormat;
pub use frontend::texture::texture_traits::Aspect;

pub use frontend::encoding::fragment_test::Test;

pub use frontend::encoding::fragment_test::DepthTest;
pub use frontend::encoding::fragment_test::DepthBias;
pub use frontend::encoding::fragment_test::DepthLhs;

pub use frontend::encoding::fragment_test::StencilTest;
pub use frontend::encoding::fragment_test::StencilBranch;
pub use frontend::encoding::fragment_test::StencilFace;
pub use frontend::encoding::fragment_test::StencilMasking;
pub use frontend::encoding::fragment_test::StencilOp;

// color attachments
pub use frontend::encoding::rasterizer::RenderPassAttachments;

pub use frontend::encoding::color_target::ColorTargetIter;
pub use frontend::encoding::color_target::ColorTarget;
pub use frontend::texture::texture_traits::ColorTargetFormat;

pub use frontend::texture::texture_traits::Spp;
pub use frontend::texture::texture_traits::SupportsSpp;
pub use frontend::texture::texture_traits::Single;
pub use frontend::texture::texture_traits::Multi;

pub use frontend::texture::texture_traits::Blendable;
pub use frontend::any::blend::Blend;
pub use frontend::any::blend::BlendComponent;
pub use frontend::any::blend::BlendFactor;
pub use frontend::any::blend::BlendOperation;

// # compute pipeline
pub use frontend::encoding::features::ComputeGrid;
pub use frontend::encoding::features::WorkGroup;
pub use frontend::encoding::features::Wave;

pub use frontend::encoding::features::GridSize;
pub use frontend::rust_types::len::GridDim;

/// synchronization functions (barriers) for compute pipelines
pub mod barrier {
    pub use crate::frontend::rust_types::barrier::storage;
    pub use crate::frontend::rust_types::barrier::texture;
    pub use crate::frontend::rust_types::barrier::workgroup;
    pub use crate::frontend::rust_types::barrier::workgroup_uniform_load;
}

// # pipeline layout
pub use frontend::encoding::io_iter::BindGroupIter;
pub use frontend::encoding::io_iter::BindingIter;
pub use frontend::encoding::io_iter::PushConstants;

pub use ir::pipeline::StageMask;
pub use ir::pipeline::StageMaskIter;
pub use ir::pipeline::ShaderStage;

// # control flow
pub use frontend::encoding::flow::if_;
pub use frontend::encoding::flow::if_else;
pub use frontend::encoding::flow::for_count;
pub use frontend::encoding::flow::for_range;
pub use frontend::encoding::flow::while_;
pub use frontend::encoding::flow::loop_;
pub use frontend::encoding::flow::break_;
pub use frontend::encoding::flow::break_if;
pub use frontend::encoding::flow::continue_;
pub use frontend::encoding::flow::continue_if;

// # `GpuType`s
pub use frontend::rust_types::GpuType;
pub use frontend::rust_types::ToGpuType;
pub use frontend::rust_types::To; // shorthand for `ToGpuType`

// `vec`
pub use frontend::rust_types::vec::vec;
pub use frontend::rust_types::len::{x1, x2, x3, x4};
pub use frontend::rust_types::len::Len;
pub use common::floating_point::f16;
pub use frontend::rust_types::vec::ToScalar; // .splat()

pub use frontend::rust_types::vec::zero;
pub use frontend::rust_types::vec::one;
pub use frontend::rust_types::vec::ComponentIter;

pub use frontend::rust_types::scalar_type::ScalarType;
pub use frontend::rust_types::scalar_type::ScalarTypeNumber;
pub use frontend::rust_types::scalar_type::ScalarTypeFp;
pub use frontend::rust_types::scalar_type::ScalarTypeInteger;
pub use frontend::rust_types::scalar_type::ScalarTypeSigned;
pub use frontend::rust_types::scalar_type::ScalarType32Bit;

pub use frontend::rust_types::vec_range_traits::VecRange;
pub use frontend::rust_types::vec_range_traits::VecRangeBounds;
pub use frontend::rust_types::vec_range_traits::VecRangeInclusive;
pub use frontend::rust_types::vec_range_traits::VecRangeBoundsInclusive;

/// [`vec`] and [`mat`] aliases in the style of rust's `std::simd`.
///
/// e.g.
/// - `f32x4` for `vec<f32, x4>`,
/// - `i32x1` for `vec<i32, x1>`,
/// - `boolx3` for `vec<bool, x3>`,
/// - `f32x2x2` for `mat<f32, x2, x2>`,
///
/// [`vec`]: crate::vec
pub mod aliases {
    use crate::frontend::rust_types::aliases;

    #[rustfmt::skip]
    pub use aliases::rust_simd::{
        f16x1, f32x1, f64x1, u32x1, i32x1, boolx1,
        f16x2, f32x2, f64x2, u32x2, i32x2, boolx2,
        f16x3, f32x3, f64x3, u32x3, i32x3, boolx3,
        f16x4, f32x4, f64x4, u32x4, i32x4, boolx4,
    };

    #[rustfmt::skip]
    pub use aliases::rust_simd::{
        f16x2x2, f32x2x2, f64x2x2,
        f16x2x3, f32x2x3, f64x2x3,
        f16x2x4, f32x2x4, f64x2x4,

        f16x3x2, f32x3x2, f64x3x2,
        f16x3x3, f32x3x3, f64x3x3,
        f16x3x4, f32x3x4, f64x3x4,

        f16x4x2, f32x4x2, f64x4x2,
        f16x4x3, f32x4x3, f64x4x3,
        f16x4x4, f32x4x4, f64x4x4,
    };
}
pub use aliases::*;

// `mat`
pub use frontend::rust_types::mat::mat;

// `Struct`
pub use frontend::rust_types::struct_::Struct;

// `Array`
pub use frontend::rust_types::array::Array;
pub use frontend::rust_types::array::ArrayLen;
pub use frontend::rust_types::array::Size;
pub use frontend::rust_types::array::RuntimeSize;
pub use frontend::rust_types::index::GpuIndex;

// `Atomic`
pub use frontend::rust_types::atomic::Atomic;
pub use frontend::rust_types::atomic::AtomicU32;
pub use frontend::rust_types::atomic::AtomicI32;

// `Ref` / `Cell`
pub use frontend::rust_types::reference::Ref;
pub use frontend::rust_types::mem::Cell;
pub use frontend::rust_types::reference::Read;
pub use frontend::rust_types::reference::Write;
pub use frontend::rust_types::reference::ReadWrite;
pub use frontend::rust_types::reference::AccessMode;
pub use crate::frontend::rust_types::reference::AccessModeReadable;
pub use crate::frontend::rust_types::reference::AccessModeWritable;
pub use frontend::rust_types::reference::ReadableRef;
pub use frontend::rust_types::reference::WritableRef;

/// traits, types and functions related to address spaces and memory allocation
pub mod mem {
    use crate::frontend::rust_types::mem;

    // `AddressSpace`
    pub use mem::AddressSpace;
    pub use mem::SupportsAccess;

    // `impl AddressSpace`
    pub use mem::Thread;
    pub use mem::Fn;
    pub use mem::Uniform;
    pub use mem::Storage;
    pub use mem::WorkGroup;

    // allocation functions
    //pub use frontend::rust_types::mem::alloc;
    //pub use crate::frontend::rust_types::mem::thread_local;
    pub use crate::frontend::rust_types::mem::workgroup_local;
}

// # Binding
pub use frontend::encoding::binding::Binding;

// Buffer
pub use frontend::encoding::buffer::Buffer;
pub use frontend::encoding::buffer::BufferRef;
pub use frontend::encoding::buffer::BufferAddressSpace;

// `Sampler`
pub use frontend::texture::Sampler;
pub use frontend::texture::texture_traits::SamplingMethod;
pub use frontend::texture::texture_traits::SupportsSampler;
pub use frontend::texture::texture_traits::Filtering;
pub use frontend::texture::texture_traits::Nearest;
pub use frontend::texture::texture_traits::Comparison;

// `Texture`
pub use frontend::texture::Texture;
pub use frontend::texture::texture_array::TextureArray;
pub use frontend::texture::MipFn;
pub use frontend::texture::texture_traits::SamplingFormat;
pub use frontend::texture::texture_traits::TextureFormat;
pub use frontend::texture::texture_traits::TextureCoords;
pub use frontend::texture::texture_traits::LayerCoords;
pub use frontend::texture::texture_traits::RegularGrid;
pub use frontend::texture::texture_traits::SupportsCoords;
pub use frontend::texture::texture_traits::CubeDir;

pub use frontend::texture::texture_traits::Filterable;
pub use frontend::texture::texture_traits::NonFilterable;
pub use frontend::texture::texture_traits::Depth;
pub use frontend::texture::texture_traits::TexelShaderType;

pub use common::integer::i4; // constant texel offset type
pub use common::integer::I4ConversionError;

// `StorageTexture`
pub use frontend::texture::storage_texture::StorageTexture;
pub use frontend::texture::texture_array::StorageTextureArray;
pub use frontend::texture::texture_traits::StorageTextureCoords;
pub use frontend::texture::texture_traits::StorageTextureFormat;

pub use ir::TextureFormatId;
pub use ir::ir_type::TextureFormatWrapper;
pub use ir::TextureSampleUsageType;
pub use ir::TextureAspect;

// TODO(release) try not to expose the entire texture_formats module here
pub use frontend::texture::texture_formats;
pub use frontend::texture::texture_formats as tf;

// # derive macros
pub use shame_derive::CpuLayout;
pub use shame_derive::GpuLayout;
pub use frontend::rust_types::layout_traits::GpuLayout;
pub use frontend::rust_types::layout_traits::CpuLayout;
pub use frontend::rust_types::type_layout;
pub use frontend::rust_types::type_layout::TypeLayout;
pub use frontend::rust_types::type_layout::TypeLayoutError;
pub use frontend::rust_types::type_layout::cpu_shareable;
pub use frontend::rust_types::layout_traits::ArrayElementsUnsizedError;

// derived traits
pub use frontend::rust_types::type_traits::GpuStore;
pub use frontend::rust_types::type_traits::GpuSized;
pub use frontend::rust_types::type_traits::GpuAligned;

pub use frontend::rust_types::struct_::BufferFields;
pub use frontend::rust_types::struct_::SizedFields;

pub use frontend::rust_types::type_traits::NoBools;
pub use frontend::rust_types::type_traits::NoAtomics;
pub use frontend::rust_types::type_traits::NoHandles;
pub use frontend::rust_types::type_traits::VertexAttribute;

pub use frontend::rust_types::layout_traits::VertexLayout;

/// vector types [`PackedVec`] and scalar types ([`unorm8`], [`snorm8`], ...),
/// with a smaller memory footprint than regular [`vec`] vectors,
/// which can only be used in vertex layouts
pub mod packed {
    #[rustfmt::skip]
    pub use super::frontend::rust_types::packed_vec::{
        PackedVec,
        unorm8, snorm8,

        unorm16, snorm16,

        u8x2, unorm8x2, snorm8x2,
        u8x4, unorm8x4, snorm8x4,

        u16x2, unorm16x2, snorm16x2,
        u16x4, unorm16x4, snorm16x4,

        u32_bits_of_snorm8x4_from_f32x4,
        u32_bits_of_unorm8x4_from_f32x4,
        u32_bits_of_i8x4_from_i32x4,
        u32_bits_of_u8x4_from_u32x4,
        u32_bits_of_i8x4_clamp_from_i32x4,
        u32_bits_of_u8x4_clamp_from_u32x4,
        u32_bits_of_snorm16x2_from_f32x2,
        u32_bits_of_unorm16x2_from_f32x2,
        u32_bits_of_f16x2_from_f32x2,
        u32_bits_of_snorm8x4_to_f32x4,
        u32_bits_of_unorm8x4_to_f32x4,
        u32_bits_of_i8x4_to_i32x4,
        u32_bits_of_u8x4_to_u32x4,
        u32_bits_of_snorm16x2_to_f32x2,
        u32_bits_of_unorm16x2_to_f32x2,
        u32_bits_of_f16x2_to_f32x2
    };
}

/// results of pipeline encoding
pub mod results {
    use crate::any::ColorTarget;
    use crate::frontend::encoding::pipeline_info;

    pub use pipeline_info::RenderPipeline;
    pub use pipeline_info::RenderPipelineShaders;
    pub use pipeline_info::RenderPipelineInfo;
    pub use crate::frontend::any::render_io::VertexBufferLayout;
    pub use pipeline_info::RenderPipelinePushConstantRanges;
    pub use pipeline_info::RasterizerState;

    pub use pipeline_info::ComputePipeline;
    pub use pipeline_info::ComputeShader;
    pub use pipeline_info::ComputePipelineInfo;
    pub use pipeline_info::ComputeGridInfo;

    pub use pipeline_info::BindGroupLayout;
    pub use pipeline_info::BindingLayout;

    pub use crate::frontend::any::shared_io::BindingType;
    pub use crate::frontend::any::shared_io::BufferBindingType;
    pub use crate::frontend::any::shared_io::SamplingMethod;
    pub use crate::ir::ir_type::AccessModeReadable;
    pub use crate::ir::ir_type::TextureShape;
    pub use crate::ir::ChannelFormatShaderType;
    pub use crate::ir::SamplesPerPixel;
    pub use crate::ir::AccessMode;
    pub use crate::any::VertexBufferLookupIndex;
    pub use crate::any::VertexAttribFormat;
    pub use crate::any::Len;
    pub use crate::any::ScalarType;
    pub use crate::any::LenEven;
    pub use crate::any::PackedBitsPerComponent;
    pub use crate::any::PackedScalarType;
    pub use crate::any::PackedFloat;
    pub use crate::any::ChannelWrites;
    pub use crate::frontend::encoding::fragment_test::DepthStencilState;

    pub use crate::backend::shader_code::ShaderCode;
    pub use crate::backend::shader_code::LanguageCode;

    /// a key-value store
    pub type Dict<K, V> = std::collections::BTreeMap<K, V>;
}

// #[doc(hidden)] interface starts here
// (not part of the public api)

#[doc(hidden)]
#[allow(missing_docs)]
pub mod any {
    use crate::frontend::any;
    use crate::ir;

    // type erased
    pub use any::Any;
    pub use crate::frontend::rust_types::AsAny;

    // runtime types
    pub use crate::ir::ir_type::AccessMode;
    pub use crate::ir::ir_type::AccessModeReadable;
    pub use crate::ir::ir_type::AddressSpace;
    pub use crate::ir::ir_type::AlignedType;
    pub use crate::ir::ir_type::HandleType;
    pub use crate::ir::ir_type::Len;
    pub use crate::ir::ir_type::Len2;
    pub use crate::ir::ir_type::LenEven;
    pub use crate::ir::ir_type::ScalarConstant;
    pub use crate::ir::ir_type::ScalarType;
    pub use crate::ir::ir_type::ScalarTypeFp;
    pub use crate::ir::ir_type::ScalarTypeInteger;
    pub use crate::ir::ir_type::SizedType;
    pub use crate::ir::ir_type::StoreType;
    pub use crate::ir::ir_type::Type;

    pub use crate::common::po2::U32PowerOf2;

    pub use crate::ir::ir_type::PackedBitsPerComponent;
    pub use crate::ir::ir_type::PackedFloat;
    pub use crate::ir::ir_type::PackedScalarType;
    pub use crate::ir::ir_type::PackedVector;

    pub use crate::ir::ir_type::BufferBlock;
    pub use crate::ir::ir_type::RuntimeSizedArrayField;
    pub use crate::ir::ir_type::SizedField;
    pub use crate::ir::ir_type::SizedStruct;
    pub use crate::ir::ir_type::Struct;
    pub use crate::ir::ir_type::StructKind;
    pub use crate::ir::ir_type::StructureDefinitionError;
    pub use crate::ir::ir_type::StructureFieldNamesMustBeUnique;

    // runtime binding api
    pub use any::shared_io::BindPath;
    pub use any::shared_io::BindingType;
    pub use any::shared_io::BufferBindingType;
    pub use any::shared_io::SamplingMethod;

    // runtime any api
    pub use crate::ir::expr::AtomicModify;
    pub use crate::ir::expr::Comp4;
    pub use crate::ir::expr::CompoundOp;
    pub use crate::ir::expr::VectorAccess;
    pub use crate::ir::expr::DataPackingFn;

    // runtime texture api
    pub use crate::ir::ir_type::ChannelFormatShaderType;
    pub use crate::ir::ir_type::FragmentShadingRate;
    pub use crate::ir::ir_type::SamplesPerPixel;
    pub use crate::ir::ir_type::TextureAspect;
    pub use crate::ir::ir_type::TextureSampleUsageType;

    // runtime pipeline recording api
    pub use any::render_io::Attrib;
    pub use any::render_io::ColorTarget;
    pub use any::render_io::ChannelWrites;
    pub use any::render_io::FragmentSampleMethod;
    pub use any::render_io::FragmentSamplePosition;
    pub use any::render_io::VertexBufferLookupIndex;
    pub use any::render_io::Location;
    pub use any::render_io::VertexAttribFormat;
    pub use any::render_io::VertexBufferLayout;

    #[allow(missing_docs)]
    pub mod control_flow {
        pub use crate::frontend::any::flow_builders::ForRecorder;
        pub use crate::frontend::any::flow_builders::IfRecorder;
        pub use crate::frontend::any::flow_builders::WhileRecorder;

        // TODO(release) untested
        pub use crate::frontend::any::fn_builder::FnRecorder;
        pub use crate::frontend::any::fn_builder::Param;
        pub use crate::frontend::any::fn_builder::PassAs;
    }
}

#[doc(hidden)]
#[allow(missing_docs)]
pub mod __private {
    pub const DEBUG_PRINT_ENABLED: bool = cfg!(feature = "debug_print");

    pub use super::common::proc_macro_reexports;
    pub use crate::common::small_vec_actual::SmallVec;
    pub use crate::ir::CallInfo;
}
