use std::{default, marker::PhantomData};

use texture_traits::{Comparison, Coords2dProjection, CubeDir, Filtering, Multi, Nearest};

use crate::backend::wgsl::WgslErrorKind;
use crate::frontend::rust_types::vec::vec;
use crate::ir::recording::{Context, NodeRecordingError};
use crate::results::TextureShape;
use crate::{call_info, f32x2, f32x4, i32x1, u32x1, ScalarTypeInteger, StorageTexture, TextureSampleUsageType};
use crate::{
    common::{integer::i4, small_vec_actual::SmallVec},
    f32x1,
    frontend::rust_types::{scalar_type::ScalarType, vec::IsVec},
    ir::{self, Comp4},
};

use self::texture_traits::{
    NonComparison, RegularGrid, SamplingFormat, SamplingMethod, Single, Spp, StorageTextureCoords,
    StorageTextureFormat, SupportsCoords, SupportsSampler, SupportsSpp, TexelShaderType, TextureCoords,
};
use crate::frontend::rust_types::len::Len;

use super::{
    any::{shared_io::BindPath, Any, InvalidReason},
    encoding::rasterizer::{mip_bias_for_supersampling, FragmentQuad, FragmentStage, Gradient},
    rust_types::{len::x2, reference::AccessMode, vec::ToInteger, GpuType, To, ToGpuType},
};

pub mod storage_texture;
pub mod texture_array;
pub mod texture_formats;
pub mod texture_traits;

/// A handle to a samplable texture
///
/// ## Example Instantiations
/// ```
/// use shame as sm;
/// use sm::aliases::*;
///
/// // specific format (features are inferred)
/// sm::Texture<sm::tfRgba8Unorm, f32x2, Single> // rgba, filterable, 2D, 1-spp
/// sm::Texture<sm::tfRgba8Unorm> // same as above
/// // explicit features (compiler can help less with valid usage checks)
/// sm::Texture<Filterable<f32x4>, f32x2, Multi> // rgba, filterable, 2D, multiple spp
/// sm::Texture<NonFilterable<f32x3>, f32x3, Single> // rgb, not filterable, 3D, 1-spp
/// ```
///
/// ## Generic Parameters
///
/// `Format` can be one of
/// - any sampleable [`TextureFormat`] (e.g. `shame::tf::Rgba8Unorm`). This is recommended
///   for when the exact format of the texture is known at compile time. Filtering
///   support and texel types can be inferred.
/// - [`sm::Filterable<T>`], [`sm::NonFilterable<T>`] or [`sm::Depth`] if the texture format is
///   not known at compile time. `T` is the return type when sampling from this texture.
///     - [`Filterable<T>`] for any filterable texture, where `T` can be any `f32` scalar/vector
///     - [`NonFilterable<T>`] for any nearest-samplable texture, where `T` can be any `f32`, `u32` or `i32` scalar/vector
///     - [`Depth`] for any depth textures
///
/// `Coords` represents a sampling position on the texture. It can be one of the following:
/// - [`f32x1`] for 1D textures
/// - [`f32x2`] for 2D textures
/// - [`f32x3`] for 3D textures
/// - [`CubeDir`] for cube textures
/// - `_` to infer the shape of the texture (for example, from the `sampler.sample(...)` call)
///
/// `SPP` the samples-per-pixel of the texture. It can be either
/// - [`sm::Single`] for textures with 1 sample per pixel (default)
/// - [`sm::Multi`] for textures with more than 1 sample per pixel
///
/// > note: the chosen `Format` limits the choices of `Coords` and `SPP` since
/// > not all formats support all texture shapes and multisampling. This is
/// > checked via their trait bounds
///
/// for arrays of textures, use the [`TextureArray`] type
///
/// [`TextureFormat`]: crate::TextureFormat
/// [`f32x1`]: crate::f32x1
/// [`f32x2`]: crate::f32x2
/// [`f32x3`]: crate::f32x3
/// [`CubeDir`]: crate::CubeDir
/// [`sm::Single`]: crate::Single
/// [`sm::Multi`]: crate::Multi
/// [`TextureArray`]: crate::TextureArray
/// [`Filterable<T>`]: crate::Filterable
/// [`NonFilterable<T>`]: crate::NonFilterable
/// [`Depth`]: crate::Depth
pub struct Texture<Format, Coords = vec<f32, x2>, SPP = Single>
where
    Coords: TextureCoords + SupportsSpp<SPP>, // TODO(release) SupportsFormat<Format> (example: 3d cube textures exist)
    Format: SamplingFormat + SupportsSpp<SPP> + SupportsCoords<Coords>,
    SPP: Spp,
{
    inner: TextureKind,
    phantom: PhantomData<(Coords, Format, SPP)>,
}

#[derive(Clone, Copy)]
pub(crate) enum TextureKind {
    Standalone(Any),
    ArrayLayer {
        texture: Any,
        layer: Any,
        shape: TextureShape,
    },
}

/// a sampler, required to sample from `Texture<Fmt, _, _> where Fmt: SupportsSampler<Method>`
///
/// `Method` can be one of:
/// - [`sm::Filtering`] for bilinear, anisotropic etc. sampling
/// - [`sm::Nearest`] for unfiltered sampling
/// - [`sm::Comparison`] for depth-texture comparison samplers
///
/// [`sm::Filtering`]: crate::Filtering
/// [`sm::Nearest`]: crate::Nearest
/// [`sm::Comparison`]: crate::Comparison
#[derive(Clone, Copy)]
pub struct Sampler<Method: SamplingMethod> {
    any: Any,
    phantom: PhantomData<Method>,
}

impl<Method: SamplingMethod> Sampler<Method> {
    pub(super) fn from_inner(any: Any) -> Self {
        Self {
            any,
            phantom: PhantomData,
        }
    }
}

impl<Format, Coords, SPP> Texture<Format, Coords, SPP>
where
    Coords: TextureCoords + SupportsSpp<SPP>,
    Format: SamplingFormat + SupportsSpp<SPP> + SupportsCoords<Coords>,
    SPP: Spp,
{
    pub(super) fn from_inner(inner: TextureKind) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<Format, Coords, SPP> Texture<Format, Coords, SPP>
where
    Coords: TextureCoords + SupportsSpp<SPP> + RegularGrid,
    Format: SamplingFormat + SupportsSpp<SPP> + SupportsCoords<Coords>,
    SPP: Spp,
{
    /// (no documentation yet)
    pub fn size(&self) -> vec<u32, Coords::Len> {
        let (TextureKind::Standalone(texture) |
        TextureKind::ArrayLayer {
            texture,
            layer: _,
            shape: _,
        }) = self.inner;
        texture.texture_dimensions(None).into()
    }
}

impl<Format, Coords> Texture<Format, Coords, Single>
where
    Coords: TextureCoords + SupportsSpp<Single> + RegularGrid,
    Format: SamplingFormat + SupportsSpp<Single> + SupportsCoords<Coords>,
{
    /// (no documentation yet)
    pub fn size_at_mip_level(&self, level: impl ToInteger) -> vec<u32, Coords::Len> {
        let (TextureKind::Standalone(texture) |
        TextureKind::ArrayLayer {
            texture,
            layer: _,
            shape: _,
        }) = self.inner;
        texture.texture_dimensions(Some(level.to_any())).into()
    }

    /// (no documentation yet)
    pub fn mip_level_count(&self) -> u32x1 {
        let (TextureKind::Standalone(texture) |
        TextureKind::ArrayLayer {
            texture,
            layer: _,
            shape: _,
        }) = self.inner;
        texture.texture_num_levels().into()
    }
}

impl<Format, Coords> Texture<Format, Coords, Multi>
where
    Coords: TextureCoords + SupportsSpp<Multi>,
    Format: SamplingFormat + SupportsSpp<Multi> + SupportsCoords<Coords>,
{
    /// (no documentation yet)
    pub fn size_at_mip_level(&self, level: impl ToInteger) -> vec<u32, Coords::Len> {
        let (TextureKind::Standalone(texture) |
        TextureKind::ArrayLayer {
            texture,
            layer: _,
            shape: _,
        }) = self.inner;
        texture.texture_dimensions(Some(level.to_any())).into()
    }

    /// (no documentation yet)
    pub fn sample_count(&self) -> u32x1 {
        let (TextureKind::Standalone(texture) |
        TextureKind::ArrayLayer {
            texture,
            layer: _,
            shape: _,
        }) = self.inner;
        texture.texture_num_samples().into()
    }
}

impl<Format: SamplingFormat + SupportsCoords<CubeDir>> Texture<Format, CubeDir, Single> {
    /// size in texels (width, height) of a single cube face,
    /// where width and height are always equal
    pub fn face_size(&self) -> vec<u32, x2> {
        let (TextureKind::Standalone(texture) |
        TextureKind::ArrayLayer {
            texture,
            layer: _,
            shape: _,
        }) = self.inner;
        texture.texture_dimensions(None).into()
    }

    /// (no documentation yet)
    pub fn face_size_at_mip_level(&self, level: impl ToInteger) -> vec<u32, x2> {
        let (TextureKind::Standalone(texture) |
        TextureKind::ArrayLayer {
            texture,
            layer: _,
            shape: _,
        }) = self.inner;
        texture.texture_dimensions(Some(level.to_any())).into()
    }

    /// (no documentation yet)
    pub fn mip_level_count(&self) -> u32x1 {
        let (TextureKind::Standalone(texture) |
        TextureKind::ArrayLayer {
            texture,
            layer: _,
            shape: _,
        }) = self.inner;
        texture.texture_num_levels().into()
    }
}

#[derive(Clone, Copy)]
/// mip-map level calculation functions available for [`Filtering`] or [`Nearest`] samplers
///
/// this function is directly describing the function used in the shader, where the
/// supersampling mip bias has already been applied.
///
/// for example, when supersampling,
/// - `MipFn::QuadDdxy` becomes `AdjustedMipFn::QuadDdxyBias(- log4(sample_count))`
/// - `MipFn::QuadDdxyBias(b)` becomes `AdjustedMipFn::QuadDdxyBias(b - log4(sample_count))`
///
/// see the docs of `mip_bias_for_supersampling`
/// see also https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#3.5.7%20Pixel%20Shader%20Derivatives
#[doc(hidden)] // runtime api
pub(crate) enum RateUnadjustedMipFn<Coords: TextureCoords> {
    Quad,
    QuadBias(f32x1),
    Level(f32x1),
    Grad([Coords; 2]),
}

/// ## Mipmap coordinates calculation function
/// The function that a `Sampler` uses to calculate the
/// mipmap level and texel coordinates.
///
/// If unsure use `MipFn::Quad(fragment.quad)` or `MipFn::zero()`.
///
/// When using a `shame::Sampler`, providing the texture coordinates is not
/// sufficient for sampling texture data. The mipmap level(s) need to be decided as well.
///
/// The different ways of how mipmap levels can be calculated are the variants of this enum.
///
/// > note: `MipFn::Quad` is the method that most shader languages use
/// > in their default texture sampling functions. They use separate sampling functions
/// > to express the other `MipFn`s:
/// >
/// > | wgsl                 | | shame `MipFn`         |
/// > |----------------------|-|-----------------------|
/// > | `textureSample`      | | `MipFn::Quad` |
/// > | `textureSampleBias`  | | `MipFn::QuadBiased` |
/// > | `textureSampleGrad`  | | `MipFn::Grad`     |
/// > | `textureSampleLevel` | | `MipFn::Level` |
///
/// (no documentation yet)
#[derive(Clone, Copy)]
pub enum MipFn<Coords: TextureCoords> {
    /// usage: `MipFn::Quad(fragment.quad)` which is equivalent to `fragment.quad.into()`
    ///
    /// ## Use the `Coords` gradient to decide the mip level.
    ///
    /// This is the default way that texture sampling works in most shader languages.
    ///
    /// This variant uses the fragment quad to calculate the implicit derivatives
    /// of `Coords` when the texture sampling happens (not when this constructor
    /// is called).
    ///
    /// ## The Fragment Quad?
    /// Fragment shaders are divided into 2x2 grids (so called "quads"
    /// see https://www.w3.org/TR/WGSL/#fragment-shaders-helper-invocations for
    /// more info). In `shame` the [`FragmentQuad`] is obtained by rasterizing
    /// a primitive to obtain a [`FragmentStage<...>`] and accessing its `quad` field:
    /// `fragment.quad`.
    ///
    /// `MipFn::Quad` is functionally equivalent to
    /// ```
    /// MipFn::Grad(fragment.quad.grad(coords))
    /// ```
    /// however, `MipFn::Quad` is said to be more performant on some gpus.
    /// > note: Using this variant yields `textureSample` calls when targeting WGSL.
    /// > see: https://www.w3.org/TR/WGSL/#texturesample
    ///
    Quad(FragmentQuad),
    /// usage:
    ///
    /// ```
    /// MipFn::QuadBiased(fragment.quad, -1.0.to_gpu())
    /// ```
    ///
    /// ## Use the `Coords` gradient to decide the mip level.
    /// same as [`Gradient`] except a `f32x1` argument is added to the computed
    /// mip level as an offset.
    ///
    /// > Using this variant yields `textureSampleBias` calls when targeting WGSL.
    /// > see: https://www.w3.org/TR/WGSL/#texturesamplebias
    QuadBiased(FragmentQuad, f32x1),
    /// ## Choose the mip level directly
    /// chooses the mipmap level explicitly without any gradient calculation.
    /// A mip level of `0` corresponds to the highest resolution mipmap.
    ///
    /// ```
    /// sampler.sample(texture, MipFn::Level(2.0.to_gpu()), uv);
    /// // or shorter
    /// sampler.sample(texture, MipFn::level(2.0), uv)
    /// ```
    ///
    /// > Using this variant yields `textureSampleGrad` calls when targeting WGSL.
    /// > see: https://www.w3.org/TR/WGSL/#texturesamplelevel
    Level(f32x1),
    /// provide a custom gradient for the mip level calculation.
    ///
    /// usage:
    /// ```
    /// sampler.sample(texture, MipFn::Grad(Gradient { dx, dy }), uv);
    /// // or shorter
    /// sampler.sample(texture, Gradient { dx, dy }.into(), uv)
    /// ```
    ///
    /// > Using this variant yields `textureSampleGrad` calls when targeting WGSL.
    /// > see: https://www.w3.org/TR/WGSL/#texturesamplegrad
    Grad(Gradient<Coords>),
}

impl<Coords: TextureCoords> Default for MipFn<Coords> {
    fn default() -> Self { MipFn::zero() }
}

fn level_for_ddxy<L: Len>(Gradient { dx, dy }: Gradient<vec<f32, L>>) -> f32x1 {
    (dx.square_length()).max(dy.square_length()).log2() * 0.5
}

impl<Coords: TextureCoords> MipFn<Coords> {
    // it may be helpful if the user is able to call a mipfn manually. consider exposing this in the future
    // #[rustfmt::skip]
    // pub fn call(&self, coords: Coords) -> f32x1 {
    //     match self {
    //         MipFn::Level   { at, level }    => *level,
    //         MipFn::Grad    { at, ddxy  }    => level_for_ddxy(ddxy.map(Coords::to_inner)),
    //         MipFn::Quad    (uv, quad)       => level_f(quad.d(Coords::to_inner(*uv))),
    //         MipFn::QuadBias(uv, quad, bias) => levor_ddxy(quad.d(Coords::to_inner(*uv))) + *bias,
    //     }
    // }


    /// A [`MipFn`] that always chooses mip level zero
    pub fn zero() -> Self { MipFn::level(0.0) }

    /// A [`MipFn`] that always chooses a specific mip level
    pub fn level(mip_level: impl To<f32x1>) -> Self { MipFn::Level(mip_level.to_gpu()) }

    /// (no documentation yet)
    pub fn quad_biased(quad: FragmentQuad, level_bias: impl To<f32x1>) -> Self {
        MipFn::QuadBiased(quad, level_bias.to_gpu())
    }

    pub(crate) fn apply_rate_adjustment(&self) -> RateUnadjustedMipFn<Coords> {
        // see docs of [`mip_bias_for_supersampling`] for justification why this bias is applied when supersampling is active
        let ss_bias = |quad: &FragmentQuad| mip_bias_for_supersampling(quad.num_spp).to_gpu();
        match self {
            MipFn::Quad(quad) => match (quad.rate, quad.num_spp) {
                (ir::FragmentShadingRate::PerSample, 2..) => RateUnadjustedMipFn::QuadBias(ss_bias(quad)),
                _ => RateUnadjustedMipFn::Quad,
            },
            MipFn::QuadBiased(quad, bias) => match (quad.rate, quad.num_spp) {
                (ir::FragmentShadingRate::PerSample, 2..) => RateUnadjustedMipFn::QuadBias(*bias + ss_bias(quad)),
                _ => RateUnadjustedMipFn::QuadBias(*bias),
            },
            MipFn::Level(level) => RateUnadjustedMipFn::Level(*level),
            MipFn::Grad(Gradient { dx, dy }) => RateUnadjustedMipFn::Grad([*dx, *dy]),
        }
    }
}

impl<Coords: TextureCoords> RateUnadjustedMipFn<Coords> {
    pub fn zero() -> Self { Self::Level(0.0.to_gpu()) }
}

impl<Method: NonComparison> Sampler<Method> {
    /// (no documentation yet)
    #[track_caller]
    pub fn sample<Format, Coords, SPP>(
        &self,
        texture: Texture<Format, Coords, SPP>,
        mip_fn: MipFn<Coords>,
        coords: Coords,
    ) -> Format::SampleType
    where
        Coords: TextureCoords + SupportsSpp<SPP>,
        Format: SupportsSampler<Method> + SupportsSpp<SPP> + SupportsCoords<Coords>,
        SPP: Spp,
    {
        let (texture, array_index) = match texture.inner {
            TextureKind::Standalone(t) => (t, None),
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape: _,
            } => (t, Some(layer)),
        };
        let mip_level = mip_fn.apply_rate_adjustment();
        let coords = Coords::to_inner_any(coords);
        let any = Any::texture_sample(self.any, texture, coords, array_index, None, mip_level.into());
        any.texture_sample_vec_shrink(Format::SAMPLE_TYPE).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn sample_with_offset<Format, Coords, SPP>(
        &self,
        texture: Texture<Format, Coords, SPP>,
        mip_fn: MipFn<Coords>,
        coords: Coords,
        offset: Coords::Offset,
    ) -> Format::SampleType
    where
        Coords: TextureCoords + RegularGrid + SupportsSpp<SPP>,
        Format: SupportsSampler<Method> + SupportsSpp<SPP> + SupportsCoords<Coords>,
        SPP: Spp,
    {
        let (texture, array_index) = match texture.inner {
            TextureKind::Standalone(t) => (t, None),
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape: _,
            } => (t, Some(layer)),
        };
        let mip_level = mip_fn.apply_rate_adjustment();
        let coords = Coords::to_inner_any(coords);
        let offset = Coords::texcoord_const_offset_as_uvw_offset(offset);
        let any = Any::texture_sample(self.any, texture, coords, array_index, Some(offset), mip_level.into());
        any.texture_sample_vec_shrink(Format::SAMPLE_TYPE).into()
    }

    /// (no documentation yet)
    ///
    /// note: this function does not support textures that are elements of texture arrays.
    #[track_caller]
    pub fn sample_edge_clamp_level0<Format>(
        &self,
        texture: Texture<Format, f32x2, Single>,
        coords: f32x2,
    ) -> Format::SampleType
    where
        Format: SupportsSampler<Method> + SupportsSpp<Single> + SupportsCoords<f32x2>,
        Format::SampleType: IsVec<T = f32>,
    {
        let texture = match texture.inner {
            TextureKind::Standalone(t) => t,
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape: _,
            } => Context::try_with(call_info!(), |ctx| {
                ctx.push_error_get_invalid_any(
                    NodeRecordingError::TextureArrayElementsCannotSampleEdgeClampLevel0.into(),
                )
            })
            .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding)),
        };
        let coords = f32x2::to_inner_any(coords);

        let any = texture.texture_sample_base_clamp_to_edge(self.any, coords);
        any.texture_sample_vec_shrink(Format::SAMPLE_TYPE).into()
    }

    /// (no documentation yet)
    /// TODO: mention the arg order is indexable as `[u][v]`
    #[track_caller]
    pub fn gather<Format, Coords>(
        &self,
        texture: Texture<Format, Coords, Single>,
        coords: Coords,
    ) -> [[Format::SampleType; 2]; 2]
    where
        Coords: Coords2dProjection + SupportsSpp<Single>,
        Format: SupportsSampler<Method> + SupportsSpp<Single> + SupportsCoords<Coords>,
    {
        gather_transpose::<Format::SampleType>(
            Format::SAMPLE_TYPE.is_depth(),
            texture.inner,
            self.any,
            Coords::to_inner_any(coords),
            None,
        )
        .map(|col| col.map(Into::into))
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn gather_with_offset<Format>(
        &self,
        texture: Texture<Format, f32x2, Single>,
        coords: f32x2,
        offset: [i4; 2],
    ) -> [[Format::SampleType; 2]; 2]
    where
        f32x2: TextureCoords + RegularGrid<Offset = [i4; 2]> + SupportsSpp<Single>,
        Format: SupportsSampler<Method> + SupportsSpp<Single> + SupportsCoords<f32x2>,
    {
        gather_transpose::<Format::SampleType>(
            Format::SAMPLE_TYPE.is_depth(),
            texture.inner,
            self.any,
            f32x2::to_inner_any(coords),
            Some(f32x2::texcoord_const_offset_as_uvw_offset(offset)),
        )
        .map(|col| col.map(Into::into))
    }
}

/// uses the usual shader gather functions (which returns all texel corners of the same channel),
/// but reorganizes the results into 4 vectors representing the 4 texels
#[track_caller]
fn gather_transpose<SampleType: IsVec>(
    is_depth: bool,
    texture_kind: TextureKind,
    sampler: Any,
    coords: Any,
    offset_uvw: Option<[i4; 3]>,
) -> [[Any; 2]; 2] {
    let (texture, array_index) = match texture_kind {
        TextureKind::Standalone(t) => (t, None),
        TextureKind::ArrayLayer {
            texture: t,
            layer,
            shape: _,
        } => (t, Some(layer)),
    };

    let sample_type_len: ir::Len = SampleType::L::LEN;
    let sample_scalar_type: ir::ScalarType = <SampleType::T as ScalarType>::SCALAR_TYPE;
    let num_channels = usize::from(sample_type_len);

    let channel_ident_hint = |comp: Comp4| match comp {
        X => "gather_rrrr",
        Y => "gather_gggg",
        Z => "gather_bbbb",
        W => "gather_aaaa",
    };

    let gather_4_corners_of_one_channel = |channel| {
        texture
            .texture_gather(sampler, coords, array_index, (!is_depth).then_some(channel), None)
            .suggest_ident(channel_ident_hint(channel))
    };

    use Comp4::*;
    // vec![vec4(r0, r1, r2, r3), vec4(g0, g1, g2, g3)...]
    let rrrr_gggg_bbbb = (0..num_channels)
        .map(|ch| gather_4_corners_of_one_channel([X, Y, Z, W][ch]))
        .collect::<Vec<_>>();

    let corner_ident_hint = |comp: Comp4| match comp {
        X => "gather_umin_vmax",
        Y => "gather_umax_vmax",
        Z => "gather_umax_vmin",
        W => "gather_umin_vmin",
    };

    let anys = [X, Y, Z, W].map(|corner| {
        let rgb = (0..num_channels)
            .map(|ch| rrrr_gggg_bbbb[ch].get_component(corner))
            .collect::<Vec<_>>();
        Any::new_vec(sample_type_len, sample_scalar_type, &rgb[0..num_channels])
            .suggest_ident(corner_ident_hint(corner))
    });

    //   ^
    //   |  [0][1]
    // V |  [3][2]
    //    ---------->
    //       U
    let [umin_vmax, umax_vmax, umax_vmin, umin_vmin] = anys;

    // [u][v] with u in 0..=1, v in 0..=1 indexable 2D array
    [[umin_vmin, umin_vmax], [umax_vmin, umax_vmax]]
}

impl Sampler<Comparison> {
    /// (no documentation yet)
    #[track_caller]
    pub fn sample<Format, Coords>(
        &self,
        texture: Texture<Format, Coords>,
        coords: Coords,
        threshold: impl To<f32x1>,
    ) -> f32x1
    where
        Coords: TextureCoords,
        Format: SupportsSampler<Comparison> + SupportsCoords<Coords>,
    {
        let (texture, array_index) = match texture.inner {
            TextureKind::Standalone(t) => (t, None),
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape: _,
            } => (t, Some(layer)),
        };
        let coords = Coords::to_inner_any(coords);
        let any = Any::texture_sample_compare_level_0(self.any, texture, coords, array_index, threshold.to_any(), None);
        f32x1::from(any)
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn sample_with_offset<Format, Coords>(
        &self,
        texture: Texture<Format, Coords>,
        coords: Coords,
        threshold: impl To<f32x1>,
        offset: Coords::Offset,
    ) -> f32x1
    where
        Coords: TextureCoords + RegularGrid,
        Format: SupportsSampler<Comparison> + SupportsCoords<Coords>,
    {
        let (texture, array_index) = match texture.inner {
            TextureKind::Standalone(t) => (t, None),
            // unreachable, is caught by `SupportsSampler` trait, and later in `TypeCheck`
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape: _,
            } => (t, Some(layer)),
        };
        let coords = Coords::to_inner_any(coords);
        let offset = Coords::texcoord_const_offset_as_uvw_offset(offset);
        let any = Any::texture_sample_compare_level_0(
            self.any,
            texture,
            coords,
            array_index,
            threshold.to_any(),
            Some(offset),
        );
        f32x1::from(any)
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn gather<Format, Coords>(
        &self,
        texture: Texture<Format, Coords>,
        coords: Coords,
        threshold: impl To<f32x1>,
    ) -> [[f32x1; 2]; 2]
    where
        Coords: TextureCoords,
        Format: SupportsSampler<Comparison> + SupportsCoords<Coords>,
    {
        let (texture, array_index) = match texture.inner {
            TextureKind::Standalone(t) => (t, None),
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape: _,
            } => (t, Some(layer)),
        };

        let corners = texture.texture_gather_compare(
            self.any,
            Coords::to_inner_any(coords),
            threshold.to_any(),
            array_index,
            None,
        );
        Self::arrange_gather_compare(corners)
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn gather_with_offset<Format, Coords>(
        &self,
        texture: Texture<Format, Coords>,
        coords: Coords,
        threshold: impl To<f32x1>,
        offset: Coords::Offset,
    ) -> [[f32x1; 2]; 2]
    where
        Coords: TextureCoords + RegularGrid,
        Format: SupportsSampler<Comparison> + SupportsCoords<Coords>,
    {
        let (texture, array_index) = match texture.inner {
            TextureKind::Standalone(t) => (t, None),
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape: _,
            } => (t, Some(layer)),
        };

        let [u, v, w] = Coords::texcoord_const_offset_as_uvw_offset(offset);

        let corners = texture.texture_gather_compare(
            self.any,
            Coords::to_inner_any(coords),
            threshold.to_any(),
            array_index,
            Some([u, v]),
        );
        Self::arrange_gather_compare(corners)
    }

    fn arrange_gather_compare(corners_f32x4: Any) -> [[f32x1; 2]; 2] {
        use Comp4::*;
        let texels = [X, Y, Z, W].map(|comp| {
            corners_f32x4
                .get_component(comp)
                .suggest_ident(match comp {
                    X => "cmp_umin_vmax",
                    Y => "cmp_umax_vmax",
                    Z => "cmp_umax_vmin",
                    W => "cmp_umin_vmin",
                })
                .into()
        });

        let [umin_vmax, umax_vmax, umax_vmin, umin_vmin] = texels;
        [[umin_vmin, umin_vmax], [umax_vmin, umax_vmax]]
    }
}

impl<Format, Coords, SPP> Clone for Texture<Format, Coords, SPP>
where
    Coords: TextureCoords + SupportsSpp<SPP>,
    Format: SamplingFormat + SupportsSpp<SPP> + SupportsCoords<Coords>,
    SPP: Spp,
{
    fn clone(&self) -> Self { *self }
}

impl<Format, Coords, SPP> Copy for Texture<Format, Coords, SPP>
where
    Coords: TextureCoords + SupportsSpp<SPP>,
    Format: SamplingFormat + SupportsSpp<SPP> + SupportsCoords<Coords>,
    SPP: Spp,
{
}

impl<Format, Coords> Texture<Format, Coords, Single>
where
    Coords: TextureCoords + RegularGrid + SupportsSpp<Single>,
    Format: SamplingFormat + SupportsSpp<Single> + SupportsCoords<Coords>,
{
    /// (no documentation yet)
    // TODO(release) wgpu rejects `mip_level` u32x1 even though the wgsl spec says it is legal,
    // only calling naga separately does not cause this issue. Once its resolve, turn mip_level into an `impl ToInteger` again
    #[track_caller]
    pub fn load<Int: ScalarTypeInteger>(
        &self,
        int_coords: vec<Int, Coords::Len>,
        mip_level: impl To<i32x1>,
    ) -> Format::SampleType {
        let (texture, array_index, shape) = match self.inner {
            TextureKind::Standalone(t) => (t, None, Coords::SHAPE),
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape,
            } => (t, Some(layer), shape),
        };
        let mip_level = mip_level.to_any();
        let coords = int_coords.to_any();
        let any = texture.texture_load(shape, coords, Some(mip_level), array_index);
        any.texture_sample_vec_shrink(Format::SAMPLE_TYPE).into()
    }
}

impl<Format, Coords> Texture<Format, Coords, Multi>
where
    Coords: TextureCoords + RegularGrid + SupportsSpp<Multi>,
    Format: SamplingFormat + SupportsSpp<Multi> + SupportsCoords<Coords>,
{
    /// (no documentation yet)
    #[track_caller]
    pub fn load<Int: ScalarTypeInteger>(
        &self,
        int_coords: vec<Int, Coords::Len>,
        mip_level: impl ToInteger,
        sample_index: impl ToInteger,
    ) -> Format::SampleType {
        let (texture, array_index, shape) = match self.inner {
            TextureKind::Standalone(t) => (t, None, Coords::SHAPE),
            TextureKind::ArrayLayer {
                texture: t,
                layer,
                shape,
            } => (t, Some(layer), shape),
        };
        let mip_level = mip_level.to_any();
        let coords = int_coords.to_any();
        let sample_index = sample_index.to_any();
        let any = texture.texture_load_multisample(shape, coords, sample_index);
        any.texture_sample_vec_shrink(Format::SAMPLE_TYPE).into()
    }
}
