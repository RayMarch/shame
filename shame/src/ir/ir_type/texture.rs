//! Texture formats, largely copied or modified from the `wgpu` crate

use crate::backend::language::Language;

use super::{AccessMode, Len, ScalarType, SizedType, StoreType};
use std::any::Any;
use std::borrow::Cow;
use std::cmp::PartialEq;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelFormatShaderType {
    F32,
    I32,
    U32,
}

impl From<ChannelFormatShaderType> for ScalarType {
    fn from(value: ChannelFormatShaderType) -> Self {
        match value {
            ChannelFormatShaderType::F32 => ScalarType::F32,
            ChannelFormatShaderType::I32 => ScalarType::I32,
            ChannelFormatShaderType::U32 => ScalarType::U32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureSampleShaderType {
    FilterableFloat {
        len: Len,
    },
    Nearest {
        len: Len,
        channel_type: ChannelFormatShaderType,
    },
    Depth,
}

/// taken from the `wgpu` crate
///
/// If there is a conversion in the format (such as srgb -> linear), the conversion listed here is for
/// loading from texture in a shader. When writing to the texture, the opposite conversion takes place.
///
/// Corresponds to [WebGPU `GPUTextureFormat`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gputextureformat).
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[repr(u8)]
pub enum BuiltinTextureFormatId {
    // Normal 8 bit formats
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,

    // Normal 16 bit formats
    R16Uint,
    R16Sint,
    R16Unorm,
    R16Snorm,
    R16Float,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,

    // Normal 32 bit formats
    R32Uint,
    R32Sint,
    R32Float,
    Rg16Uint,
    Rg16Sint,
    Rg16Unorm,
    Rg16Snorm,
    Rg16Float,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,
    Bgra8UnormSrgb,

    // Packed 32 bit formats
    Rgb9e5Ufloat,
    Rgb10a2Uint,
    Rgb10a2Unorm,
    Rg11b10Float,

    // Normal 64 bit formats
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Unorm,
    Rgba16Snorm,
    Rgba16Float,

    // Normal 128 bit formats
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,

    // Depth and stencil formats
    Stencil8,
    Depth16Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,

    Depth32FloatStencil8,

    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc4RSnorm,
    Bc5RgUnorm,
    Bc5RgSnorm,
    Bc6hRgbUfloat,
    Bc6hRgbFloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,
    Etc2Rgb8Unorm,
    Etc2Rgb8UnormSrgb,
    Etc2Rgb8A1Unorm,
    Etc2Rgb8A1UnormSrgb,
    Etc2Rgba8Unorm,
    Etc2Rgba8UnormSrgb,
    EacR11Unorm,
    EacR11Snorm,
    EacRg11Unorm,
    EacRg11Snorm,
    Astc {
        /// compressed block dimensions
        block: AstcBlock,
        /// ASTC RGBA channel
        channel: AstcChannel,
    },
}

/// ASTC block dimensions
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum AstcBlock {
    B4x4,
    B5x4,
    B5x5,
    B6x5,
    B6x6,
    B8x5,
    B8x6,
    B8x8,
    B10x5,
    B10x6,
    B10x8,
    B10x10,
    B12x10,
    B12x12,
}

/// ASTC RGBA channel
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum AstcChannel {
    Unorm,
    UnormSrgb,
    Hdr,
}

/// Kind of data the texture holds.
///
/// Corresponds to [WebGPU `GPUTextureAspect`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gputextureaspect).
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum TextureAspect {
    #[default]
    Color,
    Stencil,
    Depth,
}

/// Kind of data the texture holds.
///
/// Corresponds to [WebGPU `GPUTextureAspect`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gputextureaspect).
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum TextureAspectOld {
    /// Depth, Stencil, and Color.
    #[default]
    All,
    /// Stencil.
    StencilOnly,
    /// Depth.
    DepthOnly,
}

/// Specific type of a sample in a texture binding.
///
/// Corresponds to [WebGPU `GPUTextureSampleType`](https://gpuweb.github.io/gpuweb/#enumdef-gputexturesampletype).
// TODO(release) low prio: should this be removed?
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TextureSampleTypeOld {
    /// Sampling returns floats.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<f32>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2D t;
    /// ```
    Float {
        /// If this is `false`, the texture can't be sampled with
        /// a filtering sampler.
        ///
        /// Even if this is `true`, it's possible to sample with
        /// a **non-filtering** sampler.
        filterable: bool,
    },
    /// Sampling does the depth reference comparison.
    ///
    /// This is also compatible with a non-filtering sampler.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_depth_2d;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2DShadow t;
    /// ```
    Depth,
    /// Sampling returns signed integers.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<i32>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform itexture2D t;
    /// ```
    Sint,
    /// Sampling returns unsigned integers.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<u32>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform utexture2D t;
    /// ```
    Uint,
}

/// taken from the `wgpu` crate
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct TextureFormatFeatures(u32);

/// If not present, the texture can't be sampled with a filtering sampler.
/// This may overwrite TextureSampleType::Float.filterable
const FILTERABLE: TextureFormatFeatures = TextureFormatFeatures(1 << 0);
/// Texture format supports a per-pixel sample amount of 2
const MULTISAMPLE_X2: TextureFormatFeatures = TextureFormatFeatures(1 << 1);
/// Texture format supports a per-pixel sample amount of 4
const MULTISAMPLE_X4: TextureFormatFeatures = TextureFormatFeatures(1 << 2);
/// Texture format supports a per-pixel sample amount of 8
const MULTISAMPLE_X8: TextureFormatFeatures = TextureFormatFeatures(1 << 3);
/// Texture format supports a per-pixel sample amount of 16
const MULTISAMPLE_X16: TextureFormatFeatures = TextureFormatFeatures(1 << 4);
/// Textures of this format can be used as a read or readwrite storage texture
const STORAGE_READ_OR_READWRITE: TextureFormatFeatures = TextureFormatFeatures(1 << 6);
/// If not present, the texture can't be blended into the render target.
const BLENDABLE: TextureFormatFeatures = TextureFormatFeatures(1 << 7);

/// Allows a texture to be a sampleable texture in a bind group.
const TEXTURE_BINDING: TextureFormatFeatures = TextureFormatFeatures(1 << 22);
/// Allows a texture to be a storage texture in a bind group.
const STORAGE_BINDING: TextureFormatFeatures = TextureFormatFeatures(1 << 23);
/// Allows a texture to be an output attachment of a render pass.
const RENDER_ATTACHMENT: TextureFormatFeatures = TextureFormatFeatures(1 << 24);

impl TextureFormatFeatures {
    fn all() -> Self {
        FILTERABLE |
            MULTISAMPLE_X2 |
            MULTISAMPLE_X4 |
            MULTISAMPLE_X8 |
            MULTISAMPLE_X16 |
            STORAGE_READ_OR_READWRITE |
            BLENDABLE |
            TEXTURE_BINDING |
            STORAGE_BINDING |
            RENDER_ATTACHMENT
    }
}

impl std::ops::BitOr for TextureFormatFeatures {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output { Self(self.0 | rhs.0) }
}

impl BuiltinTextureFormatId {
    /// taken from the `wgpu` crate
    ///
    /// Returns `true` if the format is a depth and/or stencil format
    ///
    /// see <https://gpuweb.github.io/gpuweb/#depth-formats>
    pub fn is_depth_stencil_format(&self) -> bool {
        match *self {
            Self::Stencil8 |
            Self::Depth16Unorm |
            Self::Depth24Plus |
            Self::Depth24PlusStencil8 |
            Self::Depth32Float |
            Self::Depth32FloatStencil8 => true,
            _ => false,
        }
    }

    /// taken from the `wgpu` crate
    ///
    /// Returns `true` if the format is a combined depth-stencil format
    ///
    /// see <https://gpuweb.github.io/gpuweb/#combined-depth-stencil-format>
    pub fn is_combined_depth_stencil_format(&self) -> bool {
        match *self {
            Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8 => true,
            _ => false,
        }
    }

    /// taken from the `wgpu` crate
    ///
    /// Returns `true` if the format has a color aspect
    pub fn has_color_aspect(&self) -> bool { !self.is_depth_stencil_format() }

    /// taken from the `wgpu` crate
    ///
    /// Returns `true` if the format has a depth aspect
    pub fn has_depth_aspect(&self) -> bool {
        match *self {
            Self::Depth16Unorm |
            Self::Depth24Plus |
            Self::Depth24PlusStencil8 |
            Self::Depth32Float |
            Self::Depth32FloatStencil8 => true,
            _ => false,
        }
    }

    /// taken from the `wgpu` crate
    ///
    /// Returns `true` if the format has a stencil aspect
    pub fn has_stencil_aspect(&self) -> bool {
        match *self {
            Self::Stencil8 | Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8 => true,
            _ => false,
        }
    }

    /// modified from the `wgpu` crate
    ///
    /// Returns the format features guaranteed by the WebGPU spec.
    ///
    /// Additional features are available if `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` is enabled.
    pub fn format_features_webgpu(&self) -> TextureFormatFeatures {
        // Multisampling
        let noaa = TextureFormatFeatures(0);
        let msaa = MULTISAMPLE_X4;
        let msaa_resolve = msaa;

        // Flags
        let basic = TEXTURE_BINDING;
        let attachment = basic | RENDER_ATTACHMENT;
        let storage = basic | STORAGE_BINDING;
        let all_flags = TextureFormatFeatures::all();
        let rg11b10f = basic;
        let bgra8unorm = attachment;

        #[rustfmt::skip] // lets make a nice table
        let (
            mut flags,
            allowed_usages,
        ) = match *self {
            Self::R8Unorm =>              (msaa_resolve, attachment),
            Self::R8Snorm =>              (        noaa,      basic),
            Self::R8Uint =>               (        msaa, attachment),
            Self::R8Sint =>               (        msaa, attachment),
            Self::R16Uint =>              (        msaa, attachment),
            Self::R16Sint =>              (        msaa, attachment),
            Self::R16Float =>             (msaa_resolve, attachment),
            Self::Rg8Unorm =>             (msaa_resolve, attachment),
            Self::Rg8Snorm =>             (        noaa,      basic),
            Self::Rg8Uint =>              (        msaa, attachment),
            Self::Rg8Sint =>              (        msaa, attachment),
            Self::R32Uint =>              (        noaa,  all_flags),
            Self::R32Sint =>              (        noaa,  all_flags),
            Self::R32Float =>             (        msaa,  all_flags),
            Self::Rg16Uint =>             (        msaa, attachment),
            Self::Rg16Sint =>             (        msaa, attachment),
            Self::Rg16Float =>            (msaa_resolve, attachment),
            Self::Rgba8Unorm =>           (msaa_resolve,  all_flags),
            Self::Rgba8UnormSrgb =>       (msaa_resolve, attachment),
            Self::Rgba8Snorm =>           (        noaa,    storage),
            Self::Rgba8Uint =>            (        msaa,  all_flags),
            Self::Rgba8Sint =>            (        msaa,  all_flags),
            Self::Bgra8Unorm =>           (msaa_resolve, bgra8unorm),
            Self::Bgra8UnormSrgb =>       (msaa_resolve, attachment),
            Self::Rgb10a2Uint =>          (        msaa, attachment),
            Self::Rgb10a2Unorm =>         (msaa_resolve, attachment),
            Self::Rg11b10Float =>         (        msaa,   rg11b10f),
            Self::Rg32Uint =>             (        noaa,  all_flags),
            Self::Rg32Sint =>             (        noaa,  all_flags),
            Self::Rg32Float =>            (        noaa,  all_flags),
            Self::Rgba16Uint =>           (        msaa,  all_flags),
            Self::Rgba16Sint =>           (        msaa,  all_flags),
            Self::Rgba16Float =>          (msaa_resolve,  all_flags),
            Self::Rgba32Uint =>           (        noaa,  all_flags),
            Self::Rgba32Sint =>           (        noaa,  all_flags),
            Self::Rgba32Float =>          (        noaa,  all_flags),

            Self::Stencil8 =>             (        msaa, attachment),
            Self::Depth16Unorm =>         (        msaa, attachment),
            Self::Depth24Plus =>          (        msaa, attachment),
            Self::Depth24PlusStencil8 =>  (        msaa, attachment),
            Self::Depth32Float =>         (        msaa, attachment),
            Self::Depth32FloatStencil8 => (        msaa, attachment),

            Self::R16Unorm =>             (        msaa,    storage),
            Self::R16Snorm =>             (        msaa,    storage),
            Self::Rg16Unorm =>            (        msaa,    storage),
            Self::Rg16Snorm =>            (        msaa,    storage),
            Self::Rgba16Unorm =>          (        msaa,    storage),
            Self::Rgba16Snorm =>          (        msaa,    storage),

            Self::Rgb9e5Ufloat =>         (        noaa,      basic),

            Self::Bc1RgbaUnorm =>         (        noaa,      basic),
            Self::Bc1RgbaUnormSrgb =>     (        noaa,      basic),
            Self::Bc2RgbaUnorm =>         (        noaa,      basic),
            Self::Bc2RgbaUnormSrgb =>     (        noaa,      basic),
            Self::Bc3RgbaUnorm =>         (        noaa,      basic),
            Self::Bc3RgbaUnormSrgb =>     (        noaa,      basic),
            Self::Bc4RUnorm =>            (        noaa,      basic),
            Self::Bc4RSnorm =>            (        noaa,      basic),
            Self::Bc5RgUnorm =>           (        noaa,      basic),
            Self::Bc5RgSnorm =>           (        noaa,      basic),
            Self::Bc6hRgbUfloat =>        (        noaa,      basic),
            Self::Bc6hRgbFloat =>         (        noaa,      basic),
            Self::Bc7RgbaUnorm =>         (        noaa,      basic),
            Self::Bc7RgbaUnormSrgb =>     (        noaa,      basic),

            Self::Etc2Rgb8Unorm =>        (        noaa,      basic),
            Self::Etc2Rgb8UnormSrgb =>    (        noaa,      basic),
            Self::Etc2Rgb8A1Unorm =>      (        noaa,      basic),
            Self::Etc2Rgb8A1UnormSrgb =>  (        noaa,      basic),
            Self::Etc2Rgba8Unorm =>       (        noaa,      basic),
            Self::Etc2Rgba8UnormSrgb =>   (        noaa,      basic),
            Self::EacR11Unorm =>          (        noaa,      basic),
            Self::EacR11Snorm =>          (        noaa,      basic),
            Self::EacRg11Unorm =>         (        noaa,      basic),
            Self::EacRg11Snorm =>         (        noaa,      basic),

            Self::Astc { .. } =>          (        noaa,      basic),
        };

        if matches!(self.sample_type(None), Some(TextureSampleTypeOld::Float { .. })) {
            flags = flags | FILTERABLE;
            flags = flags | BLENDABLE;
        }

        flags | allowed_usages
    }

    /// Returns the sample type compatible with this format and aspect
    ///
    /// Returns `None` only if the format is combined depth-stencil
    /// and `TextureAspect::All` or no `aspect` was provided
    pub fn sample_type(&self, aspect: Option<TextureAspect>) -> Option<TextureSampleTypeOld> {
        let float = TextureSampleTypeOld::Float { filterable: true };
        let unfilterable_float = TextureSampleTypeOld::Float { filterable: false };
        let depth = TextureSampleTypeOld::Depth;
        let uint = TextureSampleTypeOld::Uint;
        let sint = TextureSampleTypeOld::Sint;

        match *self {
            Self::R8Unorm |
            Self::R8Snorm |
            Self::Rg8Unorm |
            Self::Rg8Snorm |
            Self::Rgba8Unorm |
            Self::Rgba8UnormSrgb |
            Self::Rgba8Snorm |
            Self::Bgra8Unorm |
            Self::Bgra8UnormSrgb |
            Self::R16Float |
            Self::Rg16Float |
            Self::Rgba16Float |
            Self::Rgb10a2Unorm |
            Self::Rg11b10Float => Some(float),

            Self::R32Float | Self::Rg32Float | Self::Rgba32Float => Some(unfilterable_float),

            Self::R8Uint |
            Self::Rg8Uint |
            Self::Rgba8Uint |
            Self::R16Uint |
            Self::Rg16Uint |
            Self::Rgba16Uint |
            Self::R32Uint |
            Self::Rg32Uint |
            Self::Rgba32Uint |
            Self::Rgb10a2Uint => Some(uint),

            Self::R8Sint |
            Self::Rg8Sint |
            Self::Rgba8Sint |
            Self::R16Sint |
            Self::Rg16Sint |
            Self::Rgba16Sint |
            Self::R32Sint |
            Self::Rg32Sint |
            Self::Rgba32Sint => Some(sint),

            Self::Stencil8 => Some(uint),
            Self::Depth16Unorm | Self::Depth24Plus | Self::Depth32Float => Some(depth),
            Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8 => None,

            Self::R16Unorm |
            Self::R16Snorm |
            Self::Rg16Unorm |
            Self::Rg16Snorm |
            Self::Rgba16Unorm |
            Self::Rgba16Snorm => Some(float),

            Self::Rgb9e5Ufloat => Some(float),

            Self::Bc1RgbaUnorm |
            Self::Bc1RgbaUnormSrgb |
            Self::Bc2RgbaUnorm |
            Self::Bc2RgbaUnormSrgb |
            Self::Bc3RgbaUnorm |
            Self::Bc3RgbaUnormSrgb |
            Self::Bc4RUnorm |
            Self::Bc4RSnorm |
            Self::Bc5RgUnorm |
            Self::Bc5RgSnorm |
            Self::Bc6hRgbUfloat |
            Self::Bc6hRgbFloat |
            Self::Bc7RgbaUnorm |
            Self::Bc7RgbaUnormSrgb => Some(float),

            Self::Etc2Rgb8Unorm |
            Self::Etc2Rgb8UnormSrgb |
            Self::Etc2Rgb8A1Unorm |
            Self::Etc2Rgb8A1UnormSrgb |
            Self::Etc2Rgba8Unorm |
            Self::Etc2Rgba8UnormSrgb |
            Self::EacR11Unorm |
            Self::EacR11Snorm |
            Self::EacRg11Unorm |
            Self::EacRg11Snorm => Some(float),

            Self::Astc { .. } => Some(float),
        }
    }

    pub(crate) fn color_type_in_shader(self) -> Option<SizedType> {
        use super::Len::*;
        use super::ScalarType::*;
        self.sample_type(None).map(|sample_type| {
            SizedType::Vector(
                X4,
                match sample_type {
                    TextureSampleTypeOld::Float { filterable } => F32,
                    TextureSampleTypeOld::Depth => F32,
                    TextureSampleTypeOld::Sint => I32,
                    TextureSampleTypeOld::Uint => U32,
                },
            )
        })
    }
}
