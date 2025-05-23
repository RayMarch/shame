//! this module is doing the verbose, but trivial conversion from the `shame`
//! output structs/enums to their wgpu counterparts. Features that are unsupported
//! by wgpu are returned as [`ShameToWgpuError`]s

use shame::{self as sm, TextureFormatId};
use shame::__private::SmallVec; // just to avoid some .collect() allocations.
use shame::results as smr;

use crate::surface_format::ExtendedTextureFormats;

pub enum ShameToWgpuError {
    UnsupportedShaderStage(sm::ShaderStage),
    UnsupportedTextureFormat(String),
    UnsupportedIndexBufferFormat(sm::Indexing),
    UnsupportedVertexAttribFormat(smr::VertexAttribFormat),
    MustStartAtIndexZero(&'static str, u32),
    MustHaveConsecutiveIndices(&'static str),
    FragmentStageNeedsAttachmentInteraction,
    RuntimeSurfaceFormatNotProvided,
}

impl std::error::Error for ShameToWgpuError {}

impl std::fmt::Debug for ShameToWgpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self}") }
}

impl std::fmt::Display for ShameToWgpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ShameToWgpuError as E;
        match self {
            E::UnsupportedShaderStage(mask) => write!(f, "mask `{mask}` contains stages unsupported by wgpu"),
            E::UnsupportedTextureFormat(fmt) => write!(f, "texture format `{fmt:?}` is unsupported by wgpu"),
            E::MustStartAtIndexZero(thing, index) => {
                write!(f, "the first {thing} (index={index}) must have index zero in wgpu")
            }
            E::MustHaveConsecutiveIndices(thing) => write!(f, "{thing}s must have consecutive indices in wgpu"),
            E::UnsupportedIndexBufferFormat(fmt) => {
                write!(f, "wgpu does not support the index buffer format `{fmt:?}`")
            }
            E::UnsupportedVertexAttribFormat(fmt) => {
                write!(f, "wgpu does not support the vertex attribute format `{fmt:?}`")
            }
            E::FragmentStageNeedsAttachmentInteraction => {
                write!(
                    f,
                    "wgpu requires render pipelines to access at least one color or depth/stencil attachment"
                )
            }
            E::RuntimeSurfaceFormatNotProvided => write!(
                f,
                "trying to convert runtime surface-format to wgpu, but `surface_format` is not available (`None`) in this context"
            ),
        }
    }
}

pub fn render_pipeline_push_constant_ranges(
    pcr: smr::RenderPipelinePushConstantRanges,
) -> Option<wgpu::PushConstantRange> {
    let var_declaration_range = 0..pcr.push_constants_byte_size as u32;

    match (pcr.vert, pcr.frag) {
        (None, None) => None,
        (Some(_v), None) => Some(wgpu::ShaderStages::VERTEX),
        (None, Some(_f)) => Some(wgpu::ShaderStages::FRAGMENT),
        (Some(_v), Some(_f)) => Some(wgpu::ShaderStages::VERTEX_FRAGMENT),
    }
    .map(|stages| wgpu::PushConstantRange {
        stages,
        range: var_declaration_range,
    })
}

pub fn stage_mask(sm: sm::StageMask) -> Result<wgpu::ShaderStages, ShameToWgpuError> {
    let mut mask = wgpu::ShaderStages::empty();

    for stage in sm {
        mask |= match stage {
            sm::ShaderStage::Comp => wgpu::ShaderStages::COMPUTE,
            sm::ShaderStage::Vert => wgpu::ShaderStages::VERTEX,
            sm::ShaderStage::Frag => wgpu::ShaderStages::FRAGMENT,
            sm::ShaderStage::Task | sm::ShaderStage::Mesh => {
                return Err(ShameToWgpuError::UnsupportedShaderStage(stage));
            }
        };
    }
    Ok(mask)
}

pub fn sample_type(st: sm::TextureSampleUsageType) -> wgpu::TextureSampleType {
    use sm::TextureSampleUsageType as T;
    match st {
        T::FilterableFloat { len: _ } => wgpu::TextureSampleType::Float { filterable: true },
        T::Nearest { len: _, channel_type } => match channel_type {
            smr::ChannelFormatShaderType::F32 => wgpu::TextureSampleType::Float { filterable: false },
            smr::ChannelFormatShaderType::I32 => wgpu::TextureSampleType::Sint,
            smr::ChannelFormatShaderType::U32 => wgpu::TextureSampleType::Uint,
        },
        T::Depth => wgpu::TextureSampleType::Depth,
    }
}

/// converts `tf` into a `wgpu::TextureFormat` if supported.
/// If `tf` is `ExtraTextureFormats::SurfaceFormat`, then the provided `surface_format` argument
/// is returned if it is `Some`. Otherwise an error is returned.
#[rustfmt::skip]
pub fn texture_format(tf: &dyn sm::TextureFormatId, surface_format: Option<wgpu::TextureFormat>) -> Result<wgpu::TextureFormat, ShameToWgpuError> {
    let fmt = match tf.as_builtin() {
        Some(fmt) => Ok(fmt),
        None => {
            if tf.to_binary_repr() == ExtendedTextureFormats::SurfaceFormat.to_binary_repr() {
                return surface_format.ok_or(ShameToWgpuError::RuntimeSurfaceFormatNotProvided)
            } else {
                Err(ShameToWgpuError::UnsupportedTextureFormat(format!("{tf:?}")))
            }
        }
    }?;

    use sm::tf::BuiltinTextureFormatId as SmTf;
    use sm::tf::astc::AstcBlock as SmASTCb;
    use sm::tf::astc::AstcChannel as SmASTCc;
    let wtf = match fmt {
        SmTf::R8Unorm              => wgpu::TextureFormat::R8Unorm,
        SmTf::R8Snorm              => wgpu::TextureFormat::R8Snorm,
        SmTf::R8Uint               => wgpu::TextureFormat::R8Uint,
        SmTf::R8Sint               => wgpu::TextureFormat::R8Sint,
        SmTf::R16Uint              => wgpu::TextureFormat::R16Uint,
        SmTf::R16Sint              => wgpu::TextureFormat::R16Sint,
        SmTf::R16Unorm             => wgpu::TextureFormat::R16Unorm,
        SmTf::R16Snorm             => wgpu::TextureFormat::R16Snorm,
        SmTf::R16Float             => wgpu::TextureFormat::R16Float,
        SmTf::Rg8Unorm             => wgpu::TextureFormat::Rg8Unorm,
        SmTf::Rg8Snorm             => wgpu::TextureFormat::Rg8Snorm,
        SmTf::Rg8Uint              => wgpu::TextureFormat::Rg8Uint,
        SmTf::Rg8Sint              => wgpu::TextureFormat::Rg8Sint,
        SmTf::R32Uint              => wgpu::TextureFormat::R32Uint,
        SmTf::R32Sint              => wgpu::TextureFormat::R32Sint,
        SmTf::R32Float             => wgpu::TextureFormat::R32Float,
        SmTf::Rg16Uint             => wgpu::TextureFormat::Rg16Uint,
        SmTf::Rg16Sint             => wgpu::TextureFormat::Rg16Sint,
        SmTf::Rg16Unorm            => wgpu::TextureFormat::Rg16Unorm,
        SmTf::Rg16Snorm            => wgpu::TextureFormat::Rg16Snorm,
        SmTf::Rg16Float            => wgpu::TextureFormat::Rg16Float,
        SmTf::Rgba8Unorm           => wgpu::TextureFormat::Rgba8Unorm,
        SmTf::Rgba8UnormSrgb       => wgpu::TextureFormat::Rgba8UnormSrgb,
        SmTf::Rgba8Snorm           => wgpu::TextureFormat::Rgba8Snorm,
        SmTf::Rgba8Uint            => wgpu::TextureFormat::Rgba8Uint,
        SmTf::Rgba8Sint            => wgpu::TextureFormat::Rgba8Sint,
        SmTf::Bgra8Unorm           => wgpu::TextureFormat::Bgra8Unorm,
        SmTf::Bgra8UnormSrgb       => wgpu::TextureFormat::Bgra8UnormSrgb,
        SmTf::Rgb9e5Ufloat         => wgpu::TextureFormat::Rgb9e5Ufloat,
        SmTf::Rgb10a2Uint          => wgpu::TextureFormat::Rgb10a2Uint,
        SmTf::Rgb10a2Unorm         => wgpu::TextureFormat::Rgb10a2Unorm,
        SmTf::Rg32Uint             => wgpu::TextureFormat::Rg32Uint,
        SmTf::Rg32Sint             => wgpu::TextureFormat::Rg32Sint,
        SmTf::Rg32Float            => wgpu::TextureFormat::Rg32Float,
        SmTf::Rgba16Uint           => wgpu::TextureFormat::Rgba16Uint,
        SmTf::Rgba16Sint           => wgpu::TextureFormat::Rgba16Sint,
        SmTf::Rgba16Unorm          => wgpu::TextureFormat::Rgba16Unorm,
        SmTf::Rgba16Snorm          => wgpu::TextureFormat::Rgba16Snorm,
        SmTf::Rgba16Float          => wgpu::TextureFormat::Rgba16Float,
        SmTf::Rgba32Uint           => wgpu::TextureFormat::Rgba32Uint,
        SmTf::Rgba32Sint           => wgpu::TextureFormat::Rgba32Sint,
        SmTf::Rgba32Float          => wgpu::TextureFormat::Rgba32Float,
        SmTf::Stencil8             => wgpu::TextureFormat::Stencil8,
        SmTf::Depth16Unorm         => wgpu::TextureFormat::Depth16Unorm,
        SmTf::Depth24Plus          => wgpu::TextureFormat::Depth24Plus,
        SmTf::Depth24PlusStencil8  => wgpu::TextureFormat::Depth24PlusStencil8,
        SmTf::Depth32Float         => wgpu::TextureFormat::Depth32Float,
        SmTf::Depth32FloatStencil8 => wgpu::TextureFormat::Depth32FloatStencil8,
        SmTf::NV12                 => wgpu::TextureFormat::NV12,
        SmTf::Bc1RgbaUnorm         => wgpu::TextureFormat::Bc1RgbaUnorm,
        SmTf::Bc1RgbaUnormSrgb     => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
        SmTf::Bc2RgbaUnorm         => wgpu::TextureFormat::Bc2RgbaUnorm,
        SmTf::Bc2RgbaUnormSrgb     => wgpu::TextureFormat::Bc2RgbaUnormSrgb,
        SmTf::Bc3RgbaUnorm         => wgpu::TextureFormat::Bc3RgbaUnorm,
        SmTf::Bc3RgbaUnormSrgb     => wgpu::TextureFormat::Bc3RgbaUnormSrgb,
        SmTf::Bc4RUnorm            => wgpu::TextureFormat::Bc4RUnorm,
        SmTf::Bc4RSnorm            => wgpu::TextureFormat::Bc4RSnorm,
        SmTf::Bc5RgUnorm           => wgpu::TextureFormat::Bc5RgUnorm,
        SmTf::Bc5RgSnorm           => wgpu::TextureFormat::Bc5RgSnorm,
        SmTf::Bc6hRgbUfloat        => wgpu::TextureFormat::Bc6hRgbUfloat,
        SmTf::Bc6hRgbFloat         => wgpu::TextureFormat::Bc6hRgbFloat,
        SmTf::Bc7RgbaUnorm         => wgpu::TextureFormat::Bc7RgbaUnorm,
        SmTf::Bc7RgbaUnormSrgb     => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        SmTf::Etc2Rgb8Unorm        => wgpu::TextureFormat::Etc2Rgb8Unorm,
        SmTf::Etc2Rgb8UnormSrgb    => wgpu::TextureFormat::Etc2Rgb8UnormSrgb,
        SmTf::Etc2Rgb8A1Unorm      => wgpu::TextureFormat::Etc2Rgb8A1Unorm,
        SmTf::Etc2Rgb8A1UnormSrgb  => wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb,
        SmTf::Etc2Rgba8Unorm       => wgpu::TextureFormat::Etc2Rgba8Unorm,
        SmTf::Etc2Rgba8UnormSrgb   => wgpu::TextureFormat::Etc2Rgba8UnormSrgb,
        SmTf::EacR11Unorm          => wgpu::TextureFormat::EacR11Unorm,
        SmTf::EacR11Snorm          => wgpu::TextureFormat::EacR11Snorm,
        SmTf::EacRg11Unorm         => wgpu::TextureFormat::EacRg11Unorm,
        SmTf::EacRg11Snorm         => wgpu::TextureFormat::EacRg11Snorm,
        SmTf::Astc { block, channel } => wgpu::TextureFormat::Astc {
            block: match block {
                SmASTCb::B4x4   => wgpu::AstcBlock::B4x4,
                SmASTCb::B5x4   => wgpu::AstcBlock::B5x4,
                SmASTCb::B5x5   => wgpu::AstcBlock::B5x5,
                SmASTCb::B6x5   => wgpu::AstcBlock::B6x5,
                SmASTCb::B6x6   => wgpu::AstcBlock::B6x6,
                SmASTCb::B8x5   => wgpu::AstcBlock::B8x5,
                SmASTCb::B8x6   => wgpu::AstcBlock::B8x6,
                SmASTCb::B8x8   => wgpu::AstcBlock::B8x8,
                SmASTCb::B10x5  => wgpu::AstcBlock::B10x5,
                SmASTCb::B10x6  => wgpu::AstcBlock::B10x6,
                SmASTCb::B10x8  => wgpu::AstcBlock::B10x8,
                SmASTCb::B10x10 => wgpu::AstcBlock::B10x10,
                SmASTCb::B12x10 => wgpu::AstcBlock::B12x10,
                SmASTCb::B12x12 => wgpu::AstcBlock::B12x12,
            },
            channel: match channel {
                SmASTCc::Unorm     => wgpu::AstcChannel::Unorm,
                SmASTCc::UnormSrgb => wgpu::AstcChannel::UnormSrgb,
                SmASTCc::Hdr       => wgpu::AstcChannel::Hdr,
            }
        },
    };
    Ok(wtf)
}

pub fn binding_layout(index: u32, bl: &smr::BindingLayout) -> Result<wgpu::BindGroupLayoutEntry, ShameToWgpuError> {
    let tex_shape_to_count_dim = |shape| match shape {
        smr::TextureShape::_1D => (None, wgpu::TextureViewDimension::D1),
        smr::TextureShape::_2D => (None, wgpu::TextureViewDimension::D2),
        smr::TextureShape::_3D => (None, wgpu::TextureViewDimension::D3),
        smr::TextureShape::Cube => (None, wgpu::TextureViewDimension::Cube),
        smr::TextureShape::_2DArray(count) => (Some(count), wgpu::TextureViewDimension::D2Array),
        smr::TextureShape::CubeArray(count) => (Some(count), wgpu::TextureViewDimension::CubeArray),
    };

    let (ty, count) = {
        match &bl.binding_ty {
            smr::BindingType::Buffer { ty, has_dynamic_offset } => {
                let ty = match ty {
                    smr::BufferBindingType::Uniform => wgpu::BufferBindingType::Uniform,
                    smr::BufferBindingType::Storage(am) => wgpu::BufferBindingType::Storage {
                        read_only: match am {
                            smr::AccessModeReadable::Read => true,
                            smr::AccessModeReadable::ReadWrite => false,
                        },
                    },
                };
                (
                    wgpu::BindingType::Buffer {
                        ty,
                        has_dynamic_offset: *has_dynamic_offset,
                        min_binding_size: bl.shader_ty.min_byte_size(),
                    },
                    None,
                )
            }
            smr::BindingType::Sampler(method) => (
                wgpu::BindingType::Sampler(match method {
                    smr::SamplingMethod::Filtering => wgpu::SamplerBindingType::Filtering,
                    smr::SamplingMethod::NonFiltering => wgpu::SamplerBindingType::NonFiltering,
                    smr::SamplingMethod::Comparison => wgpu::SamplerBindingType::Comparison,
                }),
                None,
            ),
            smr::BindingType::SampledTexture {
                shape,
                sample_type: st,
                samples_per_pixel: spp,
            } => {
                let (count, dim) = tex_shape_to_count_dim(*shape);

                let ty = wgpu::BindingType::Texture {
                    sample_type: sample_type(*st),
                    view_dimension: dim,
                    multisampled: match spp {
                        smr::SamplesPerPixel::Single => false,
                        smr::SamplesPerPixel::Multi => true,
                    },
                };

                (ty, count)
            }
            smr::BindingType::StorageTexture { shape, format, access } => {
                let (count, dim) = tex_shape_to_count_dim(*shape);

                let ty = wgpu::BindingType::StorageTexture {
                    access: match access {
                        smr::AccessMode::Read => wgpu::StorageTextureAccess::ReadOnly,
                        smr::AccessMode::Write => wgpu::StorageTextureAccess::WriteOnly,
                        smr::AccessMode::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
                    },
                    format: texture_format(format.as_dyn(), None)?,
                    view_dimension: dim,
                };

                (ty, count)
            }
        }
    };

    let supported_stages = sm::StageMask::pipeline_render() | sm::StageMask::pipeline_compute();

    let visibility = match stage_mask(bl.visibility) {
        Ok(v) => v,
        Err(unsupported_stage) => {
            // the mask only contains unsupported stages, (likely mesh/task shaders)
            // this may be fine, e.g. if the user chose a blanket StageMask::all() visibility
            //
            // in that case we use the remaining stages as visibility
            //
            // if there are no remaining stages however, we return the error.
            match bl.visibility & supported_stages {
                mask if mask.is_empty() => return Err(unsupported_stage),
                supported_subset => stage_mask(supported_subset)?,
            }
        }
    };

    Ok(wgpu::BindGroupLayoutEntry {
        binding: index,
        visibility,
        ty,
        count,
    })
}

pub fn bind_group_layout(
    gpu: &wgpu::Device,
    bgl: &smr::BindGroupLayout,
) -> Result<wgpu::BindGroupLayout, ShameToWgpuError> {
    Ok(gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &bgl
            .bindings
            .iter()
            .map(|(i, binding)| binding_layout(*i, binding))
            .collect::<Result<SmallVec<_, 8>, _>>()?,
    }))
}

fn pipeline_layout(
    gpu: &wgpu::Device,
    bind_groups: &smr::Dict<u32, smr::BindGroupLayout>,
    push_consts: &[wgpu::PushConstantRange],
) -> Result<Option<wgpu::PipelineLayout>, ShameToWgpuError> {
    if bind_groups.is_empty() && push_consts.is_empty() {
        return Ok(None);
    }
    const EST_NUM_BIND: usize = 8;
    let bgls = {
        let mut vec = SmallVec::<_, EST_NUM_BIND>::default();
        for (i, bgl) in bind_groups {
            let len = vec.len() as u32;
            match len {
                _ if len == *i => Ok(()),
                0 => Err(ShameToWgpuError::MustStartAtIndexZero("bind group", *i)),
                _ => Err(ShameToWgpuError::MustHaveConsecutiveIndices("bind group")),
            }?;
            vec.push(bind_group_layout(gpu, bgl)?)
        }
        vec
    };
    let bgl_refs: SmallVec<_, EST_NUM_BIND> = bgls.iter().collect();

    let pl = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &bgl_refs,
        push_constant_ranges: push_consts,
    });

    Ok(Some(pl))
}

fn shader_module(gpu: &wgpu::Device, lang_code: smr::LanguageCode) -> wgpu::ShaderModule {
    gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: match lang_code {
            smr::LanguageCode::Wgsl(code) => wgpu::ShaderSource::Wgsl(code.into()),
        },
    })
}

pub fn blend(blend: sm::Blend) -> wgpu::BlendState {
    let blend_factor = |factor: sm::BlendFactor| match factor {
        sm::BlendFactor::Zero => wgpu::BlendFactor::Zero,
        sm::BlendFactor::One => wgpu::BlendFactor::One,
        sm::BlendFactor::Src => wgpu::BlendFactor::Src,
        sm::BlendFactor::OneMinusSrc => wgpu::BlendFactor::OneMinusSrc,
        sm::BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        sm::BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        sm::BlendFactor::Dst => wgpu::BlendFactor::Dst,
        sm::BlendFactor::OneMinusDst => wgpu::BlendFactor::OneMinusDst,
        sm::BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
        sm::BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        sm::BlendFactor::SrcAlphaSaturated => wgpu::BlendFactor::SrcAlphaSaturated,
        sm::BlendFactor::Constant => wgpu::BlendFactor::Constant,
        sm::BlendFactor::OneMinusConstant => wgpu::BlendFactor::OneMinusConstant,
    };
    let blend_op = |op: sm::BlendOperation| match op {
        sm::BlendOperation::Add => wgpu::BlendOperation::Add,
        sm::BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
        sm::BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        sm::BlendOperation::Min => wgpu::BlendOperation::Min,
        sm::BlendOperation::Max => wgpu::BlendOperation::Max,
    };
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: blend_factor(blend.color.src_factor),
            dst_factor: blend_factor(blend.color.dst_factor),
            operation: blend_op(blend.color.operation),
        },
        alpha: wgpu::BlendComponent {
            src_factor: blend_factor(blend.alpha.src_factor),
            dst_factor: blend_factor(blend.alpha.dst_factor),
            operation: blend_op(blend.alpha.operation),
        },
    }
}

#[derive(Clone)]
struct WgpuVertexBufferLayout {
    pub array_stride: wgpu::BufferAddress,
    pub step_mode: wgpu::VertexStepMode,
    pub attributes: SmallVec<wgpu::VertexAttribute, 16>,
}

impl WgpuVertexBufferLayout {
    pub fn as_ref(&self) -> wgpu::VertexBufferLayout {
        wgpu::VertexBufferLayout {
            array_stride: self.array_stride,
            step_mode: self.step_mode,
            attributes: &self.attributes,
        }
    }
}

fn color_writes(write_mask: smr::ChannelWrites) -> wgpu::ColorWrites {
    let mut mask = wgpu::ColorWrites::empty();
    for (write, bit) in [
        (write_mask.r, wgpu::ColorWrites::RED),
        (write_mask.g, wgpu::ColorWrites::GREEN),
        (write_mask.b, wgpu::ColorWrites::BLUE),
        (write_mask.a, wgpu::ColorWrites::ALPHA),
    ] {
        if write {
            mask |= bit
        }
    }
    mask
}

#[rustfmt::skip]
fn vertex_format(format: smr::VertexAttribFormat) -> Result<wgpu::VertexFormat, ShameToWgpuError> {
    use shame::any::layout::ScalarType as S;
    use smr::Len as L;
    use wgpu::VertexFormat as W;
    let unsupported = Err(ShameToWgpuError::UnsupportedVertexAttribFormat(format));
    Ok(match format {
        smr::VertexAttribFormat::Fine(v) => match (v.scalar, v.len) {
            (S::F16, L::X1) => W::Float16,
            (S::F16, L::X2) => W::Float16x2,
            (S::F16, L::X3) => return unsupported,
            (S::F16, L::X4) => W::Float16x4,

            (S::F32, L::X1) => W::Float32,
            (S::F32, L::X2) => W::Float32x2,
            (S::F32, L::X3) => W::Float32x3,
            (S::F32, L::X4) => W::Float32x4,

            (S::F64, L::X1) => W::Float64,
            (S::F64, L::X2) => W::Float64x2,
            (S::F64, L::X3) => W::Float64x3,
            (S::F64, L::X4) => W::Float64x4,

            (S::U32, L::X1) => W::Uint32,
            (S::U32, L::X2) => W::Uint32x2,
            (S::U32, L::X3) => W::Uint32x3,
            (S::U32, L::X4) => W::Uint32x4,

            (S::I32, L::X1) => W::Sint32,
            (S::I32, L::X2) => W::Sint32x2,
            (S::I32, L::X3) => W::Sint32x3,
            (S::I32, L::X4) => W::Sint32x4,
        },

        smr::VertexAttribFormat::Coarse(p) => {
            use smr::PackedScalarType as PS;
            use smr::PackedFloat as Norm;
            use smr::PackedBitsPerComponent as Bits;
            use smr::LenEven as L;

            match (p.scalar_type, p.bits_per_component, p.len) {
            (PS::Float(Norm::Unorm), Bits::_8 , L::X2) => W::Unorm8x2,
            (PS::Float(Norm::Unorm), Bits::_8 , L::X4) => W::Unorm8x4,
            (PS::Float(Norm::Unorm), Bits::_16, L::X2) => W::Unorm16x2,
            (PS::Float(Norm::Unorm), Bits::_16, L::X4) => W::Unorm16x4,

            (PS::Float(Norm::Snorm), Bits::_8 , L::X2) => W::Snorm8x2,
            (PS::Float(Norm::Snorm), Bits::_8 , L::X4) => W::Snorm8x4,
            (PS::Float(Norm::Snorm), Bits::_16, L::X2) => W::Snorm16x2,
            (PS::Float(Norm::Snorm), Bits::_16, L::X4) => W::Snorm16x4,

            (PS::Int , Bits::_8 , L::X2) => W::Sint8x2,
            (PS::Int , Bits::_8 , L::X4) => W::Sint8x4,
            (PS::Int , Bits::_16, L::X2) => W::Sint16x2,
            (PS::Int , Bits::_16, L::X4) => W::Sint16x4,

            (PS::Uint, Bits::_8 , L::X2) => W::Uint8x2,
            (PS::Uint, Bits::_8 , L::X4) => W::Uint8x4,
            (PS::Uint, Bits::_16, L::X2) => W::Uint16x2,
            (PS::Uint, Bits::_16, L::X4) => W::Uint16x4,
        }},
    })
}

/// returns (array_stride, step_mode, attributes) tuple
fn vertex_buffer_layout(vbuf: smr::VertexBufferLayoutRecorded) -> Result<WgpuVertexBufferLayout, ShameToWgpuError> {
    let layout = WgpuVertexBufferLayout {
        array_stride: vbuf.stride,
        step_mode: match vbuf.lookup {
            smr::VertexBufferLookupIndex::VertexIndex => wgpu::VertexStepMode::Vertex,
            smr::VertexBufferLookupIndex::InstanceIndex => wgpu::VertexStepMode::Instance,
        },
        attributes: {
            let mut vec = SmallVec::default();
            for att in vbuf.attribs {
                vec.push(wgpu::VertexAttribute {
                    shader_location: att.location.0,
                    offset: att.offset,
                    format: vertex_format(att.format)?,
                })
            }
            vec
        },
    };
    Ok(layout)
}

#[rustfmt::skip]
fn depth_stencil(ds: smr::DepthStencilState, front_face: wgpu::FrontFace) -> Result<wgpu::DepthStencilState, ShameToWgpuError> {
    let compare = |test| match test {
        sm::Test::Never        => wgpu::CompareFunction::Never,
        sm::Test::Less         => wgpu::CompareFunction::Less,
        sm::Test::Equal        => wgpu::CompareFunction::Equal,
        sm::Test::LessEqual    => wgpu::CompareFunction::LessEqual,
        sm::Test::Greater      => wgpu::CompareFunction::Greater,
        sm::Test::NotEqual     => wgpu::CompareFunction::NotEqual,
        sm::Test::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        sm::Test::Always       => wgpu::CompareFunction::Always,
    };
    let stencil_op = |op: sm::StencilOp| match op {
        sm::StencilOp::Keep     => wgpu::StencilOperation::Keep,
        sm::StencilOp::Zero     => wgpu::StencilOperation::Zero,
        sm::StencilOp::Replace  => wgpu::StencilOperation::Replace,
        sm::StencilOp::Invert   => wgpu::StencilOperation::Invert,
        sm::StencilOp::IncClamp => wgpu::StencilOperation::IncrementClamp,
        sm::StencilOp::DecClamp => wgpu::StencilOperation::DecrementClamp,
        sm::StencilOp::IncWrap  => wgpu::StencilOperation::IncrementWrap,
        sm::StencilOp::DecWrap  => wgpu::StencilOperation::DecrementWrap,
    };
    let stencil_face = |face: sm::StencilFace| {
        wgpu::StencilFaceState {
            compare: compare(face.compare),
            fail_op: stencil_op(face.on_fail),
            depth_fail_op: stencil_op(face.on_pass_depth_fail),
            pass_op: stencil_op(face.on_pass_depth_pass),
        }
    };
    let (front, back) = match front_face {
        wgpu::FrontFace::Ccw => (ds.stencil.ccw, ds.stencil.cw),
        wgpu::FrontFace::Cw  => (ds.stencil.cw, ds.stencil.ccw),
    };
    Ok(wgpu::DepthStencilState {
        format: texture_format(ds.format.as_dyn(), None)?,
        depth_write_enabled: ds.depth_write_enabled,
        depth_compare: match ds.depth_compare {
            sm::Test::Never        => wgpu::CompareFunction::Never,
            sm::Test::Less         => wgpu::CompareFunction::Less,
            sm::Test::Equal        => wgpu::CompareFunction::Equal,
            sm::Test::LessEqual    => wgpu::CompareFunction::LessEqual,
            sm::Test::Greater      => wgpu::CompareFunction::Greater,
            sm::Test::NotEqual     => wgpu::CompareFunction::NotEqual,
            sm::Test::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
            sm::Test::Always       => wgpu::CompareFunction::Always,
        },
        stencil: wgpu::StencilState {
            front: stencil_face(front),
            back:  stencil_face(back),
            read_mask:  ds.stencil.rw_mask,
            write_mask: ds.stencil.w_mask,
        },
        bias: wgpu::DepthBiasState {
            constant:    ds.bias.constant,
            slope_scale: ds.bias.slope_scale,
            clamp:       ds.bias.clamp,
        },
    })
}

#[allow(unused)]
pub fn render_pipeline(
    gpu: &wgpu::Device,
    pdef: smr::RenderPipeline,
    surface_format: Option<wgpu::TextureFormat>,
) -> Result<wgpu::RenderPipeline, ShameToWgpuError> {
    enum Modules {
        Separate(wgpu::ShaderModule, wgpu::ShaderModule),
        Shared(wgpu::ShaderModule),
    }

    let (modules, v_entry, f_entry) = {
        let v = pdef.shaders.vert_entry_point;
        let f = pdef.shaders.frag_entry_point;
        let modules = match pdef.shaders.into_shared_shader_code() {
            Ok(shader) => Modules::Shared(shader_module(gpu, shader)),
            Err((vert, frag)) => Modules::Separate(shader_module(gpu, vert), shader_module(gpu, frag)),
        };
        (modules, v, f)
    };

    let compilation_options = wgpu::PipelineCompilationOptions {
        constants: &[],
        zero_initialize_workgroup_memory: false, // render pipelines have no workgroups
    };

    let front_face = match pdef.pipeline.rasterizer.front_face {
        sm::Winding::Ccw => wgpu::FrontFace::Ccw,
        sm::Winding::Cw => wgpu::FrontFace::Cw,
        sm::Winding::Either => wgpu::FrontFace::Ccw,
    };

    let vertex_buffers = {
        let mut vec = SmallVec::<WgpuVertexBufferLayout, 8>::default();
        for (i, vbuf) in pdef.pipeline.vertex_buffers {
            let len = vec.len() as u32;
            match len {
                _ if len == i => Ok(()),
                0 => Err(ShameToWgpuError::MustStartAtIndexZero("vertex buffer", i)),
                _ => Err(ShameToWgpuError::MustHaveConsecutiveIndices("vertex buffer")),
            }?;
            vec.push(vertex_buffer_layout(vbuf)?)
        }
        vec
    };
    let mut vertex_buffers =
        SmallVec::<wgpu::VertexBufferLayout, 8>::from_iter(vertex_buffers.iter().map(|x| x.as_ref()));

    let no_targets = pdef.pipeline.color_targets.is_empty() && pdef.pipeline.depth_stencil.is_none();
    if no_targets {
        return Err(ShameToWgpuError::FragmentStageNeedsAttachmentInteraction);
    }
    let targets = {
        let mut targets = pdef.pipeline.color_targets;
        let mut vec = SmallVec::<Option<wgpu::ColorTargetState>, 8>::default();
        let max_target = targets.keys().copied().max().unwrap_or(0);
        for i in 0..=max_target {
            vec.push(match targets.remove(&i) {
                None => None,
                Some(col) => Some(wgpu::ColorTargetState {
                    format: texture_format(col.format.as_dyn(), surface_format)?,
                    blend: col.blend.map(blend),
                    write_mask: color_writes(col.write_mask),
                }),
            })
        }

        vec
    };

    let render = gpu.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: pdef.label.as_ref().map(|s| s.as_ref()),
        layout: pipeline_layout(
            gpu,
            &pdef.pipeline.bind_groups,
            render_pipeline_push_constant_ranges(pdef.pipeline.push_constants).as_slice(),
        )?
        .as_ref(),
        vertex: {
            wgpu::VertexState {
                module: match &modules {
                    Modules::Separate(v, f) => v,
                    Modules::Shared(s) => s,
                },
                entry_point: Some(v_entry),
                compilation_options: compilation_options.clone(),
                buffers: &vertex_buffers,
            }
        },
        primitive: {
            let draw = pdef.pipeline.rasterizer.draw_info;
            wgpu::PrimitiveState {
                topology: match draw {
                    sm::Draw::Point => wgpu::PrimitiveTopology::PointList,
                    sm::Draw::Line {
                        strip,
                        triangle_topology,
                    } => match (strip, triangle_topology) {
                        (true, true) => wgpu::PrimitiveTopology::TriangleStrip,
                        (false, true) => wgpu::PrimitiveTopology::TriangleList,
                        (true, false) => wgpu::PrimitiveTopology::LineStrip,
                        (false, false) => wgpu::PrimitiveTopology::LineList,
                    },
                    sm::Draw::Triangle { strip, .. } => match strip {
                        true => wgpu::PrimitiveTopology::TriangleStrip,
                        false => wgpu::PrimitiveTopology::TriangleList,
                    },
                },
                polygon_mode: match draw {
                    shame::Draw::Point => wgpu::PolygonMode::Point,
                    shame::Draw::Line {
                        strip,
                        triangle_topology,
                    } => match triangle_topology {
                        true => wgpu::PolygonMode::Line,
                        // "line"-ness is achieved via `wgpu::PrimitiveTopology` already.
                        // that way we don't require `wgpu::FeaturesWGPU::POLYGON_MODE_LINE`
                        // in this particular case
                        false => wgpu::PolygonMode::Fill,
                    },
                    shame::Draw::Triangle { .. } => wgpu::PolygonMode::Fill,
                },
                strip_index_format: match pdef.pipeline.rasterizer.vertex_indexing {
                    sm::Indexing::Incremental => None,
                    sm::Indexing::BufferU8 => match draw.is_strip() {
                        true => return Err(ShameToWgpuError::UnsupportedIndexBufferFormat(sm::Indexing::BufferU8)),
                        false => None,
                    },
                    sm::Indexing::BufferU16 => Some(wgpu::IndexFormat::Uint16),
                    sm::Indexing::BufferU32 => Some(wgpu::IndexFormat::Uint32),
                },
                front_face,
                cull_mode: match draw {
                    sm::Draw::Point => None,
                    sm::Draw::Line { .. } => None,
                    sm::Draw::Triangle { winding, .. } => {
                        match (winding, front_face) {
                            (sm::Winding::Either, _) => None,
                            // front face is visible, so back face is culled
                            (sm::Winding::Ccw, wgpu::FrontFace::Ccw) | (sm::Winding::Cw, wgpu::FrontFace::Cw) => {
                                Some(wgpu::Face::Back)
                            }
                            // front face is not visible, so front face is culled
                            (sm::Winding::Ccw, wgpu::FrontFace::Cw) | (sm::Winding::Cw, wgpu::FrontFace::Ccw) => {
                                Some(wgpu::Face::Front)
                            }
                        }
                    }
                },
                unclipped_depth: match draw {
                    shame::Draw::Point | shame::Draw::Line { .. } => false, // not a polygon
                    shame::Draw::Triangle {
                        strip,
                        conservative,
                        winding,
                        z_clip,
                    } => match z_clip {
                        shame::ZClip::NearFar => false,
                        shame::ZClip::Off => true,
                    },
                },
                conservative: match draw {
                    shame::Draw::Point | shame::Draw::Line { .. } => false,
                    shame::Draw::Triangle { conservative, .. } => conservative,
                },
            }
        },
        depth_stencil: pdef
            .pipeline
            .depth_stencil
            .map(|ds| depth_stencil(ds, front_face))
            .transpose()?,
        multisample: wgpu::MultisampleState {
            count: pdef.pipeline.rasterizer.samples.len() as u32,
            mask: pdef.pipeline.rasterizer.samples.as_u64(),
            alpha_to_coverage_enabled: pdef.pipeline.rasterizer.color_target0_alpha_to_coverage,
        },
        fragment: match pdef.pipeline.skippable_fragment_stage {
            true => return Err(ShameToWgpuError::FragmentStageNeedsAttachmentInteraction),
            false => Some(wgpu::FragmentState {
                module: match &modules {
                    Modules::Separate(v, f) => f,
                    Modules::Shared(s) => s,
                },
                entry_point: Some(f_entry),
                compilation_options,
                targets: &targets,
            }),
        },
        multiview: None,
        cache: None,
    });
    Ok(render)
}

#[allow(unused)]
pub fn compute_pipeline(
    gpu: &wgpu::Device,
    pdef: smr::ComputePipeline,
) -> Result<wgpu::ComputePipeline, ShameToWgpuError> {
    let layout = {
        let pushc = pdef.pipeline.push_constant_range.map(|range| wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range,
        });
        pipeline_layout(gpu, &pdef.pipeline.bind_groups, pushc.as_slice())?
    };
    let compute = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: pdef.label.as_ref().map(|s| s.as_ref()),
        layout: layout.as_ref(),
        module: &shader_module(gpu, pdef.shader.code),
        entry_point: Some(pdef.shader.entry_point),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[],
            zero_initialize_workgroup_memory: pdef.pipeline.grid_info.zero_init_workgroup_memory,
        },
        cache: None,
    });
    Ok(compute)
}
