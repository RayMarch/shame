use std::{marker::PhantomData, num::NonZeroU32};

use crate::{
    common::integer::i4,
    f32x2, f32x3,
    frontend::{
        encoding::rasterizer::FragmentStage,
        rust_types::{
            layout_traits::GetAllFields,
            len::*,
            reference::AccessMode,
            scalar_type::{ScalarType, ScalarTypeInteger},
            vec::{vec, IsVec},
            AsAny, GpuType, To,
        },
        texture::texture_formats::BuiltinTextureFormatId,
    },
    ir::{
        self,
        ir_type::{TextureFormatWrapper, TextureShape},
        FragmentShadingRate, GradPrecision, SamplesPerPixel, TextureFormatId, TextureSampleUsageType,
    },
};

use crate::frontend::any::shared_io;
use crate::frontend::any::Any;

#[diagnostic::on_unimplemented(
    message = "No sampled texture type exists that uses texture coordinates of type `{Self}`"
)]
/// a [`sm::vec`] type (or [`sm::CubeDir`]) that can be used as positions within the
/// texture coordinate space.
///
/// These are often called "UV"s if the texture is 2 dimensional or
/// "UVW"s for 3d textures.
///
/// [`sm::CubeDir`] is used for cube textures.
///
/// [`sm::CubeDir`]: crate::CubeDir
/// [`sm::vec`]: crate::vec
pub trait TextureCoords: SupportsSpp<Single> + Copy {
    /// (no documentation yet)
    type Len: Len;
    #[doc(hidden)] // runtime api
    const SHAPE: TextureShape;
    #[doc(hidden)] // runtime api
    fn to_inner_any(coords: Self) -> Any;
    /// returns inner vec type for e.g. CubeDir
    fn to_inner(coords: Self) -> vec<f32, Self::Len> { Self::to_inner_any(coords).into() }
}

#[diagnostic::on_unimplemented(
    message = "No storage texture type exists that uses texture coordinates of type `{Self}`"
)]
/// (no documentation yet)
pub trait StorageTextureCoords {
    /// (no documentation yet)
    type Len: Len;
    /// (no documentation yet)
    const SHAPE: TextureShape;
    /// internal function to extract the type erased any
    #[doc(hidden)] // runtime api
    fn texcoord_as_any(coord: Self) -> Any;
}
#[diagnostic::on_unimplemented(message = "No array texture type exists that uses texture coordinates of type `{Self}`")]
/// whether `Coords` can be used in a [`TextureArray`]
/// where `Coords` implements either [`TextureCoords`] or [`StorageTextureCoords`]
///
/// [`TextureArray`]: crate::TextureArray
pub trait LayerCoords {
    /// the corresponding texture-array shape, provided the array element count
    const ARRAY_SHAPE: fn(NonZeroU32) -> TextureShape;
}

/// texture coordinate types which are not cubemap directions.
/// which support a constant offset of `i4` values to be
/// applied before sampling
///
/// Not implemented for [`CubeDir`]
#[diagnostic::on_unimplemented(
    message = "Textures with coordinates of type `{Self}` are not regular grid shaped and do not support constant offsets"
)]
pub trait RegularGrid {
    /// a `[shame::i4; N]` array of 4 bit signed integers that describe a constant texel offset
    type Offset;

    /// internal function to extract uvw offset
    #[doc(hidden)]
    fn texcoord_const_offset_as_uvw_offset(offs: Self::Offset) -> [i4; 3];
}
impl<T: ScalarType> RegularGrid for vec<T, x1> {
    type Offset = [i4; 1];
    fn texcoord_const_offset_as_uvw_offset([u]: [i4; 1]) -> [i4; 3] { [u, i4::clamped(0), i4::clamped(0)] }
}
impl<T: ScalarType> RegularGrid for vec<T, x2> {
    type Offset = [i4; 2];
    fn texcoord_const_offset_as_uvw_offset([u, v]: [i4; 2]) -> [i4; 3] { [u, v, i4::clamped(0)] }
}
impl<T: ScalarType> RegularGrid for vec<T, x3> {
    type Offset = [i4; 3];
    fn texcoord_const_offset_as_uvw_offset([u, v, w]: [i4; 3]) -> [i4; 3] { [u, v, w] }
}

// TODO(release) seal this trait
/// implemented by `f32x2` and `CubeDir`
#[diagnostic::on_unimplemented(
    message = "Textures with coordinates of type `{Self}` cannot be unfolded as a 2D surface. Only 2D and cube textures support this."
)]
pub trait Coords2dProjection: TextureCoords {}
impl Coords2dProjection for f32x2 {}
impl Coords2dProjection for CubeDir {}

// 1D texture
impl SupportsSpp<Single> for vec<f32, x1> {}
impl TextureCoords for vec<f32, x1> {
    type Len = x1;
    const SHAPE: TextureShape = TextureShape::_1D;
    fn to_inner_any(s: Self) -> Any { s.as_any() }
}
impl StorageTextureCoords for vec<u32, x1> {
    type Len = x1;
    const SHAPE: TextureShape = TextureShape::_1D;
    fn texcoord_as_any(s: Self) -> Any { s.as_any() }
}
impl StorageTextureCoords for vec<i32, x1> {
    type Len = x1;
    const SHAPE: TextureShape = TextureShape::_1D;
    fn texcoord_as_any(s: Self) -> Any { s.as_any() }
}
// 2D texture
impl<SPP: Spp> SupportsSpp<SPP> for vec<f32, x2> {}
impl TextureCoords for vec<f32, x2> {
    type Len = x2;
    const SHAPE: TextureShape = TextureShape::_2D;
    fn to_inner_any(s: Self) -> Any { s.as_any() }
}

impl StorageTextureCoords for vec<u32, x2> {
    type Len = x2;
    const SHAPE: TextureShape = TextureShape::_2D;
    fn texcoord_as_any(s: Self) -> Any { s.as_any() }
}
impl StorageTextureCoords for vec<i32, x2> {
    type Len = x2;
    const SHAPE: TextureShape = TextureShape::_2D;
    fn texcoord_as_any(s: Self) -> Any { s.as_any() }
}
// 3D texture
impl SupportsSpp<Single> for vec<f32, x3> {}
impl TextureCoords for vec<f32, x3> {
    const SHAPE: TextureShape = TextureShape::_3D;
    type Len = x3;
    fn to_inner_any(s: Self) -> Any { s.as_any() }
}
impl StorageTextureCoords for vec<u32, x3> {
    type Len = x3;
    const SHAPE: TextureShape = TextureShape::_3D;
    fn texcoord_as_any(s: Self) -> Any { s.as_any() }
}
impl StorageTextureCoords for vec<i32, x3> {
    type Len = x3;
    const SHAPE: TextureShape = TextureShape::_3D;
    fn texcoord_as_any(s: Self) -> Any { s.as_any() }
}
// Cube Texture
#[derive(Clone, Copy)]
/// A 3d direction vector pointing from the center of a cube to a point on a
/// cube-map face. Any non-zero 3d vector of float numbers is a valid direction.
///
/// To define a Cube-map texture, choose this type as the [`TextureCoords`] of
/// the [`Texture`].
///
/// During sampling the vector is scaled to hit the cube-map face exactly, so
/// no normalization is required before constructing an instance of [`CubeDir`]
///
/// [`Texture`]: crate::Texture
pub struct CubeDir(pub vec<f32, x3>);
impl SupportsSpp<Single> for CubeDir {}
impl TextureCoords for CubeDir {
    const SHAPE: TextureShape = TextureShape::Cube;
    type Len = x3;
    fn to_inner_any(s: Self) -> Any { s.0.as_any() }
}
// Array Texture
impl LayerCoords for CubeDir {
    const ARRAY_SHAPE: fn(NonZeroU32) -> TextureShape = TextureShape::CubeArray;
}
impl LayerCoords for vec<f32, x2> {
    const ARRAY_SHAPE: fn(NonZeroU32) -> TextureShape = TextureShape::_2DArray;
}
impl LayerCoords for vec<u32, x2> {
    const ARRAY_SHAPE: fn(NonZeroU32) -> TextureShape = TextureShape::_2DArray;
}
impl LayerCoords for vec<i32, x2> {
    const ARRAY_SHAPE: fn(NonZeroU32) -> TextureShape = TextureShape::_2DArray;
}

impl From<vec<f32, x3>> for CubeDir {
    fn from(value: vec<f32, x3>) -> Self { Self(value) }
}

impl CubeDir {
    /// create a cubemap direction from a `f32x3` pointing from the center of the cube towards the cube faces (any length except 0 is valid)
    pub fn new(dir: impl To<f32x3>) -> Self { Self(dir.to_gpu()) }
}

/// implemented for
/// - any sampleable texture format type (e.g. `Rgba8Unorm`, `Depth24plus`, ...).
///   you can browse texture formats by typing `shame::texture_formats::browse::`
/// - `shame::Filterable<T>` where `T` is a `f32` scalar/vector
/// - `shame::NonFilterable<T>` where `T` is a `f32`, `i32` or `u32` scalar/vector
/// - `shame::Depth`
pub trait SamplingFormat: SupportsSpp<Single> {
    // TODO(release) actually, instead of doing the thing mentioned in the TODO below, the
    // SamplingFormats (which are either TextureFormat or "pure SamplingFormat", such as filterable<float4>)
    // should all map to a "pure samplingformat" (via associated type) which then specifies the SampleType (via associated type).
    // TODO(release) validate that a user has defined compatible SampleType and SAMPLE_TYPE when a texture is constructed
    /// (no documentation yet)
    type SampleType: TexelShaderType;

    /// the sample type of this format, filterable float might still become unfilterable if this format is paired with Multisampling
    const SAMPLE_TYPE: TextureSampleUsageType;
}

/// a tag type that describes a [`SamplingFormat`] that can be sampled
/// while using texture filtering (`Sampler<Filtering>`)
///
/// can only be instantiated with `f32` scalars/vectors like so: `Filterable<vec<f32, _>>`
pub struct Filterable<OutputFloatVec: TexelShaderType<T = f32>>(PhantomData<OutputFloatVec>);

/// a tag type that describes a [`SamplingFormat`] that can only be sampled
/// by using a non-filtering sampler (`Sampler<Nearest>`)
///
/// can be instantiated with any `f32`, `i32` or `u32` vectors or scalars:
/// - `NonFilterable<vec<f32, _>>`
/// - `NonFilterable<vec<i32, _>>`
/// - `NonFilterable<vec<u32, _>>`
///
/// or generically as
/// - `NonFilterable<vec<T, _>> where T: ChannelFormatShaderType`
pub struct NonFilterable<OutputVec: TexelShaderType>(PhantomData<OutputVec>);

/// a tag type that describes the depth component of depth textures
///
/// `Texture<Depth>`
pub struct Depth;

impl<Vec: TexelShaderType<T = f32>> SamplingFormat for Filterable<Vec> {
    type SampleType = Vec;
    const SAMPLE_TYPE: TextureSampleUsageType = TextureSampleUsageType::FilterableFloat {
        len: <Vec as IsVec>::L::LEN,
    };
}

impl<Vec: TexelShaderType> SamplingFormat for NonFilterable<Vec> {
    type SampleType = Vec;
    const SAMPLE_TYPE: TextureSampleUsageType = TextureSampleUsageType::Nearest {
        len: <Vec as IsVec>::L::LEN,
        channel_type: match <Vec as IsVec>::T::SCALAR_TYPE.as_channel_format_shader_type() {
            Some(t) => t,
            None => panic!("const panic"),
        },
    };
}

impl SamplingFormat for Depth {
    type SampleType = vec<f32, x1>;
    const SAMPLE_TYPE: TextureSampleUsageType = TextureSampleUsageType::Depth;
}

impl<Vec: TexelShaderType<T = f32>> SupportsSampler<Filtering> for Filterable<Vec> {}
impl<Vec: TexelShaderType<T = f32>> SupportsSampler<Nearest> for Filterable<Vec> {}
impl<Vec: TexelShaderType> SupportsSampler<Nearest> for NonFilterable<Vec> {}
impl SupportsSampler<Comparison> for Depth {}


/// implemented by
/// - [`sm::Filtering`]
/// - [`sm::Nearest`]
/// - [`sm::Comparison`]
///
/// [`sm::Filtering`]: crate::Filtering
/// [`sm::Nearest`]: crate::Nearest
/// [`sm::Comparison`]: crate::Comparison
pub trait SamplingMethod: Copy {
    #[allow(missing_docs)]
    const SAMPLING_METHOD: shared_io::SamplingMethod;
}

/// (no documentation yet)
#[derive(Clone, Copy)]
pub struct Filtering;
impl SamplingMethod for Filtering {
    const SAMPLING_METHOD: shared_io::SamplingMethod = shared_io::SamplingMethod::Filtering;
}

/// (no documentation yet)
#[derive(Clone, Copy)]
pub struct Comparison;
impl SamplingMethod for Comparison {
    const SAMPLING_METHOD: shared_io::SamplingMethod = shared_io::SamplingMethod::Comparison;
}

/// (no documentation yet)
#[derive(Clone, Copy)]
pub struct Nearest;
impl SamplingMethod for Nearest {
    const SAMPLING_METHOD: shared_io::SamplingMethod = shared_io::SamplingMethod::NonFiltering;
}

#[diagnostic::on_unimplemented(
    message = "Format `{Self}` does not support a sampler of type `shame::Sampler<{S}>`. To find the list of formats that do, use `shame::texture_formats::browse::`."
)]
/// (no documentation yet)
pub trait SupportsSampler<S: SamplingMethod>: SamplingFormat {}

/// a [`SamplingMethod`] that isn't [`Comparison`].
///
///  implemented by [`Filtering`] and [`Nearest`].
pub trait NonComparison: SamplingMethod {}
impl NonComparison for Filtering {}
impl NonComparison for Nearest {}

#[diagnostic::on_unimplemented(
    message = "Format `{Self}` does not support blending. To find the list of formats that do, use `shame::texture_formats::browse::blending::`"
)]
/// (no documentation yet)
pub trait Blendable {}

/// implemented by texture formats that have only one texture aspect (color / depth / stencil)
pub trait Aspect {
    /// (no documentation yet)
    type TexelShaderType: TexelShaderType;
}

/// (no documentation yet)
pub trait TextureFormat {
    /// (no documentation yet)
    fn id() -> impl crate::TextureFormatId;
}

/// (no documentation yet)
pub trait StorageTextureFormat<A: AccessMode>: TextureFormat + Aspect {}
/// (no documentation yet)
pub trait ColorTargetFormat: TextureFormat + Aspect {}

/// (no documentation yet)
pub trait DepthFormat: TextureFormat {
    /// (no documentation yet)
    type DepthShaderType: TexelShaderType;
}

/// (no documentation yet)
pub trait StencilFormat: TextureFormat {
    /// (no documentation yet)
    type StencilShaderType: TexelShaderType;
}

/// (no documentation yet)
pub trait DepthStencilFormat: TextureFormat {
    /// (no documentation yet)
    type Depth: DepthFormat;
    /// (no documentation yet)
    type Stencil: StencilFormat;
}

// TODO(release) this trait might be superfluous
#[diagnostic::on_unimplemented(message = "`{Self}` is not a type that can result from sampling a texture")]
/// (no documentation yet)
pub trait TexelShaderType: GpuType + IsVec {}

impl<L: Len, T: ChannelFormatShaderType> TexelShaderType for vec<T, L> {}

#[diagnostic::on_unimplemented(message = "`{Self}` textures do not support `{T}`-sampling")]
/// implemented by texture parameters that support a certain [`Spp`] (samples per pixel).
///
/// - every `SamplingFormat` supports at least [`Single`] sample per pixel, only some support [`Multi`].
/// - every texture-coordinate type supports at least [`Single`] sample per pixel, only some support [`Multi`].
///
pub trait SupportsSpp<T: Spp> {}
impl<F: SamplingFormat> SupportsSpp<Single> for F {}

#[diagnostic::on_unimplemented(message = "`{Self}` textures do not support `{T}` texture coordinates")]
/// whether `Self` supports texture coordinates of type `T`
///
/// some texture formats do not support certain coordinate systems (such as cube maps, 3d textures)
pub trait SupportsCoords<T /*T is constrained in the TextureCoords traits*/> {}
impl<F: TextureFormat, T: ScalarType> SupportsCoords<vec<T, x2>> for F {}
impl<F: TextureFormat> SupportsCoords<CubeDir> for F {}
impl<T: TexelShaderType<T = f32>, S: ScalarType> SupportsCoords<vec<S, x2>> for Filterable<T> {}
impl<T: TexelShaderType<T = f32>> SupportsCoords<CubeDir> for Filterable<T> {}
impl<T: TexelShaderType, S: ScalarType> SupportsCoords<vec<S, x2>> for NonFilterable<T> {}
impl<T: TexelShaderType> SupportsCoords<CubeDir> for NonFilterable<T> {}
impl<T: ScalarType> SupportsCoords<vec<T, x2>> for Depth {}
impl SupportsCoords<CubeDir> for Depth {}

/// see https://www.w3.org/TR/WGSL/#texel-formats
pub trait ChannelFormat {
    const BYTE_SIZE: u32;
    type ShaderType: ChannelFormatShaderType;
}

pub trait ChannelFormatShaderType: ScalarType {
    const SHADER_TYPE: ir::ChannelFormatShaderType;
}
impl ChannelFormatShaderType for f32 {
    const SHADER_TYPE: ir::ChannelFormatShaderType = ir::ChannelFormatShaderType::F32;
}
impl ChannelFormatShaderType for i32 {
    const SHADER_TYPE: ir::ChannelFormatShaderType = ir::ChannelFormatShaderType::I32;
}
impl ChannelFormatShaderType for u32 {
    const SHADER_TYPE: ir::ChannelFormatShaderType = ir::ChannelFormatShaderType::U32;
}

/// samples per pixel (either [`Single`] or [`Multi`] for multisampling/supersampling)
pub trait Spp {
    #[doc(hidden)] // runtime api
    const SAMPLES_PER_PIXEL: SamplesPerPixel;
}

/// a single sample per pixel
///
/// used in the trait [`Spp`] (samples per pixel)
#[derive(Clone, Copy)]
pub struct Single;
/// multiple samples per pixel
///
/// used in the trait [`Spp`] (samples per pixel)
#[derive(Clone, Copy)]
pub struct Multi;

impl Spp for Single {
    const SAMPLES_PER_PIXEL: SamplesPerPixel = SamplesPerPixel::Single;
}
impl Spp for Multi {
    const SAMPLES_PER_PIXEL: SamplesPerPixel = SamplesPerPixel::Multi;
}

impl TextureFormatId for BuiltinTextureFormatId {
    fn to_binary_repr(&self) -> std::borrow::Cow<[u8]> {
        Vec::from(format!("{self:?}")).into() // TODO(release) low prio: consider picking a better binary repr
    }

    fn to_wgsl_repr(&self) -> Option<std::borrow::Cow<str>> {
        crate::backend::wgsl::wgsl_builtin_texture_format_repr(*self).map(Into::into)
    }

    fn as_builtin(&self) -> Option<BuiltinTextureFormatId> { Some(*self) }

    fn sample_type(&self) -> Option<crate::ir::TextureSampleUsageType> {
        let filterable = self.is_filterable();
        let has_depth = self.has_depth_aspect();
        let sampling_result = self.sampling_result_type();

        use crate::frontend::rust_types::len::Len;
        use crate::ir::{ChannelFormatShaderType as CF, TextureAspect, TextureSampleUsageType::*};

        sampling_result.map(|(len, channel_format)| {
            let sample_type = match channel_format {
                CF::F32 => {
                    if filterable {
                        FilterableFloat { len }
                    } else {
                        Nearest {
                            len,
                            channel_type: CF::F32,
                        }
                    }
                }
                CF::I32 => Nearest {
                    len,
                    channel_type: CF::I32,
                },
                CF::U32 => Nearest {
                    len,
                    channel_type: CF::U32,
                },
            };

            if has_depth { Depth } else { sample_type }
        })
    }

    fn has_aspect(&self, aspect: ir::TextureAspect) -> bool {
        match aspect {
            ir::TextureAspect::Color => self.has_color_aspect(),
            ir::TextureAspect::Stencil => self.has_stencil_aspect(),
            ir::TextureAspect::Depth => self.has_depth_aspect(),
        }
    }

    fn is_blendable(&self) -> bool { self.is_blendable_impl() }
}
