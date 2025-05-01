use thiserror::Error;

use super::{Len, ScalarType};
use crate::{
    frontend::{any::shared_io::SamplingMethod, texture::texture_formats::BuiltinTextureFormatId},
    ir::SizedType,
};
use std::{fmt::Display, hash::Hash, num::NonZeroU32, sync::Arc};

/// (no documentation yet)
#[derive(Clone)]
pub struct TextureFormatWrapper(Arc<dyn TextureFormatId>);

impl std::fmt::Debug for TextureFormatWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{:?}", &*self.0) }
}

impl PartialEq for TextureFormatWrapper {
    fn eq(&self, other: &Self) -> bool {
        match (self.0.as_builtin(), other.0.as_builtin()) {
            (Some(a), Some(b)) => a == b,
            (None, None) => self.0.to_binary_repr() == self.0.to_binary_repr(),
            _ => false,
        }
    }
}

impl Eq for TextureFormatWrapper {}

impl Hash for TextureFormatWrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.0.as_builtin() {
            Some(builtin) => builtin.hash(state),
            None => self.0.to_binary_repr().hash(state),
        }
    }
}

impl<T: TextureFormatId> From<T> for TextureFormatWrapper {
    fn from(format: T) -> Self { Self(Arc::new(format)) }
}

impl TextureFormatWrapper {
    /// type erase `TextureFormatId`
    pub fn new<T: TextureFormatId>(format: T) -> Self { format.into() }

    /// access the inner `dyn TextureFormatId` reference
    pub fn as_dyn(&self) -> &dyn TextureFormatId { &*self.0 }

    pub(crate) fn sample_type(&self) -> Option<TextureSampleUsageType> { self.0.sample_type() }

    pub(crate) fn sample_type_as_sized_type(&self) -> Option<SizedType> { self.0.sample_type().map(SizedType::from) }

    // either vec4 or scalar(for depth-textures)
    pub(crate) fn sample_type_in_wgsl(&self) -> Option<SizedType> { self.sample_type().map(|x| x.type_in_wgsl()) }

    pub(crate) fn scalar_type_in_shader(&self) -> Option<ScalarType> {
        self.0.sample_type().map(|x| x.shader_scalar_ty())
    }

    pub(crate) fn has_aspect(&self, aspect: TextureAspect) -> bool { self.0.has_aspect(aspect) }

    pub(crate) fn is_color_format(&self) -> bool { self.has_aspect(TextureAspect::Color) }

    pub(crate) fn is_sampleable(&self) -> bool { self.sample_type().is_some() }

    pub(crate) fn is_blenable(&self) -> bool { self.0.is_blendable() }

    pub(crate) fn to_wgsl_repr(&self) -> Option<std::borrow::Cow<str>> { self.0.to_wgsl_repr() }

    /// if exists, return the corresponding shame builtin texture format id
    pub fn as_builtin(&self) -> Option<BuiltinTextureFormatId> { self.0.as_builtin() }
}

/// (no documentation yet)
pub trait TextureFormatId: std::fmt::Debug + 'static {
    /// convert `Self` into a binary representation that allows for comparison.
    /// identical binary representations are assumed to be equal.
    /// unidentical binary representations are assumed to be not equal. ([`std::cmp::Eq`])
    ///
    /// this exists to facilitate comparsion in a dyn-compatible way
    fn to_binary_repr(&self) -> std::borrow::Cow<[u8]>;

    /// the wgsl shading language requires storage texture formats to be written
    /// down in the shader code.
    /// This function returns that string representation.
    ///
    /// If the texture format has no such representation, because it cannot
    /// be used in this context, `None` should be returned, which causes an
    /// encoding error if the texture format is used in a way that would require
    /// a string representation in wgsl.
    ///
    /// for example:
    /// ```
    /// assert_eq!(Rgba8Unorm.to_wgsl_repr(), Some("rgba8unorm"));
    /// assert_eq!(Rgba8UnormSrgb.to_wgsl_repr(), None);
    /// ```
    fn to_wgsl_repr(&self) -> Option<std::borrow::Cow<str>>;

    /// the implementation of this function is entirely optional, it will only
    /// be used for preventing calls to `to_binary_repr` when hashing pipelines.
    ///
    /// If unsure just return `None`.
    fn as_builtin(&self) -> Option<BuiltinTextureFormatId> { None }

    /// result type when sampling from a `Texture` that uses this texture format
    ///
    /// shame only allows sampling from single-aspect formats.
    /// If you want to sample the stencil aspect of a `Depth24Stencil8` texture
    /// use the `Stencil8` format for the `Texture`
    fn sample_type(&self) -> Option<TextureSampleUsageType>;

    /// whether a given texture aspect such as color/depth/stencil is present in
    /// this format
    fn has_aspect(&self, aspect: TextureAspect) -> bool;

    /// whether this texture format is blendable
    fn is_blendable(&self) -> bool;
}

/// Kind of data the texture holds.
///
/// Corresponds to [WebGPU `GPUTextureAspect`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gputextureaspect).
#[allow(missing_docs)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum TextureAspect {
    #[default]
    Color,
    Stencil,
    Depth,
}

/// (no documentation yet)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSampleUsageType {
    /// doesn't have to be filtered, can also be nearest sampled
    FilterableFloat {
        /// the amount of channels in a texel
        len: Len,
    },
    /// must be nearest sampled. Not filterable!
    Nearest {
        /// the amount of channels in a texel
        len: Len,
        /// the datatype of every texel channel
        channel_type: ChannelFormatShaderType,
    },
    /// can be nearest sampled (returns float) and comparison-sampled
    Depth,
}

impl TextureSampleUsageType {
    /// multisampling makes filterable float textures unfilterable
    ///
    // TODO(release): this "restrict_with_spp" is enforcing an invariant of "SampledTexture" BindingType and ir::Type.
    // (the invariant being that multisampling turns filterable-float formats into nearest-float)
    // this should be done in a proper type with private fields and getters to prevent
    // the invariant being broken, and DRY wrt BindingType should be achieved
    // (e.g. by implementing the invariant within BindingType and offering a `BindingType -> ir::Type` conversion,
    // and always calling that one)
    pub fn restrict_with_spp(self, spp: SamplesPerPixel) -> TextureSampleUsageType {
        match (self, spp) {
            (TextureSampleUsageType::FilterableFloat { len }, SamplesPerPixel::Multi) => {
                TextureSampleUsageType::Nearest {
                    len,
                    channel_type: ChannelFormatShaderType::F32,
                }
            }
            (x, _) => x,
        }
    }
}

impl std::fmt::Display for TextureSampleUsageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextureSampleUsageType::FilterableFloat { len } => {
                write!(f, "{}ch filterable-float", u32::from(*len))
            }
            TextureSampleUsageType::Nearest { len, channel_type } => {
                write!(f, "{}ch nearest {}", u32::from(*len), channel_type)
            }
            TextureSampleUsageType::Depth => write!(f, "depth"),
        }
    }
}

impl TextureSampleUsageType {
    /// the scalar type of vectors returned by sampling a texture with this type
    pub fn shader_scalar_ty(&self) -> ScalarType { ScalarType::from(self.channel_ty()) }

    /// whether textures of this type can be sampled with a filtering sampler
    pub fn is_filterable(&self) -> bool {
        match self {
            TextureSampleUsageType::FilterableFloat { len } => true,
            TextureSampleUsageType::Nearest {
                len: _,
                channel_type: _,
            } |
            TextureSampleUsageType::Depth => false,
        }
    }

    pub(crate) fn type_in_wgsl(&self) -> SizedType {
        let sty = self.shader_scalar_ty();
        let len = if self.is_depth() { Len::X1 } else { Len::X4 };
        SizedType::Vector(len, sty)
    }

    /// whether `self` is equal to `Depth`
    pub fn is_depth(&self) -> bool {
        match self {
            TextureSampleUsageType::FilterableFloat { len } => false,
            TextureSampleUsageType::Nearest { len, channel_type } => false,
            TextureSampleUsageType::Depth => true,
        }
    }

    /// whether textures of this type can be sampled with a sampler that uses `method`
    pub fn is_compatible_with_sampler(&self, method: SamplingMethod) -> bool {
        match method {
            SamplingMethod::Filtering => matches!(self, Self::FilterableFloat { len: _ }),
            SamplingMethod::NonFiltering => true,
            SamplingMethod::Comparison => matches!(self, Self::Depth),
        }
    }

    /// the channel type. removes component count information from `self`
    pub fn channel_ty(&self) -> ChannelFormatShaderType {
        use TextureSampleUsageType as ST;
        match self {
            ST::FilterableFloat { len } => ChannelFormatShaderType::F32,
            ST::Nearest { len, channel_type } => *channel_type,
            ST::Depth => ChannelFormatShaderType::F32,
        }
    }

    /// the amount of components in a sampling result
    pub fn len(&self) -> Len {
        match self {
            TextureSampleUsageType::FilterableFloat { len } | TextureSampleUsageType::Nearest { len, .. } => *len,
            TextureSampleUsageType::Depth => Len::X1,
        }
    }
}

impl From<TextureSampleUsageType> for SizedType {
    fn from(value: TextureSampleUsageType) -> Self { SizedType::Vector(value.len(), value.channel_ty().into()) }
}

/// the returned type per texture channel after being sampled in a shader
/// (e.g. `Rgba8Unorm` stores 8-bit `unorm8` channels, but after sampling they are `f32`s)
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelFormatShaderType {
    F32,
    I32,
    U32,
}

impl std::fmt::Display for ChannelFormatShaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ChannelFormatShaderType::F32 => "f32",
            ChannelFormatShaderType::I32 => "i32",
            ChannelFormatShaderType::U32 => "u32",
        })
    }
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

impl ScalarType {
    pub(crate) const fn as_channel_format_shader_type(&self) -> Option<ChannelFormatShaderType> {
        match self {
            ScalarType::F16 => None,
            ScalarType::Bool => None,
            ScalarType::F64 => None,
            ScalarType::F32 => Some(ChannelFormatShaderType::F32),
            ScalarType::U32 => Some(ChannelFormatShaderType::U32),
            ScalarType::I32 => Some(ChannelFormatShaderType::I32),
        }
    }
}

impl TryFrom<ScalarType> for ChannelFormatShaderType {
    type Error = ();

    fn try_from(value: ScalarType) -> Result<Self, Self::Error> { value.as_channel_format_shader_type().ok_or(()) }
}

/// (no documentation yet)
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureShape {
    _1D,
    _2D,
    /// 2D texture array, with nonzero amount of elements
    _2DArray(NonZeroU32),
    _3D,
    /// cubemap texture
    Cube,
    /// 2D texture array, with nonzero amount of elements
    CubeArray(NonZeroU32),
}

impl Display for TextureShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TextureShape::_1D => "1D",
            TextureShape::_2D => "2D",
            TextureShape::_3D => "3D",
            TextureShape::Cube => "cube",
            TextureShape::_2DArray(n) => return write!(f, "2D-array<{n}>"),
            TextureShape::CubeArray(n) => return write!(f, "cube-array<{n}>"),
        })
    }
}

impl TextureShape {
    /// returns the array version of `self`  with the provided `num_elements` if it exists
    pub fn array_version(&self, num_elements: NonZeroU32) -> Option<TextureShape> {
        match self {
            TextureShape::_1D | TextureShape::_3D | TextureShape::_2DArray(_) | TextureShape::CubeArray(_) => None,
            TextureShape::_2D => Some(TextureShape::_2DArray(num_elements)),
            TextureShape::Cube => Some(TextureShape::CubeArray(num_elements)),
        }
    }
}

impl TextureShape {
    /// whether `self` is a texture array shape
    pub fn is_array(&self) -> bool {
        match self {
            TextureShape::_1D | TextureShape::_2D | TextureShape::_3D | TextureShape::Cube => false,
            TextureShape::_2DArray(_) | TextureShape::CubeArray(_) => true,
        }
    }
}

/// amount of samples per pixel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplesPerPixel {
    /// one sample per pixel
    Single,
    /// more than one sample per pixel
    Multi,
}

/// the sampling-rate at which fragment shaders need to execute
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentShadingRate {
    /// one fragment shader thread per pixel
    PerPixel,
    /// one fragment shader thread per (single/multisampling/supersampling) sample.
    PerSample,
}
