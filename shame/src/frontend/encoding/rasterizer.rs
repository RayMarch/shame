use crate::frontend::any::render_io::{VertexBufferLookupIndex, Location};
use crate::frontend::any::{Any, InvalidReason};
use crate::frontend::rust_types::array::{Array, Size, UpTo8};
use crate::frontend::rust_types::error::FrontendError;
use crate::frontend::rust_types::len::{x1, x2, x3, x4, Len};
use crate::frontend::rust_types::vec::IsVec;
use crate::frontend::rust_types::{AsAny, GpuType, To, ToGpuType};
use crate::frontend::texture::texture_traits::{
    CubeDir, DepthFormat, DepthStencilFormat, Multi, Single, StencilFormat, TextureCoords, TextureFormat,
};
use crate::frontend::texture::MipFn;
use crate::ir::pipeline::{PipelineError, WipDepthStencilState};
use crate::ir::recording::{BlockError, Context};
use crate::ir::{self, CallInfo, GradPrecision, FragmentShadingRate, SamplesPerPixel, TextureFormatWrapper};
use crate::{
    call_info,
    frontend::{
        any::record_node, encoding::EncodingErrorKind, error::InternalError, rust_types::scalar_type::ScalarType,
        texture::texture_traits::Spp,
    },
    ir::{
        expr::{BuiltinShaderOut, Expr, NoMatchingSignature, ShaderIo},
        recording::NodeRecordingError,
    },
    sig, try_ctx_track_caller,
};

use crate::frontend::rust_types::vec::vec;
use crate::{boolx1, f32x1, f32x2, f32x4, f64x1, i32x1, u32x1, u32x2, DepthBias, DepthLhs};
use std::cell::Cell;
use std::marker::PhantomData;
use std::num::{NonZeroU8, NonZeroUsize};
use std::ops::Deref;

use super::color_target::ColorTargetIter;
use super::features::Wave;
use super::fragment_test::{DepthTest, StencilTest};
use super::io_iter::VertexBufferIter;
use super::mask::BitVec64;

/// (no documentation yet)
pub struct VertexStage<'a> {
    /// The index used to perform lookups into vertex-buffers and define primitives,
    /// as defined by the [`Indexing`] sequence.
    ///
    /// use `.to_u32()` or `.to_i32()` or `*`(deref) to use it for arithmetic.
    ///
    /// A [`VertexIndex`] is a numeric index that can be used for vertex-buffer lookups.
    ///
    /// [`Indexing`]: crate::Indexing
    pub index: VertexIndex,
    /// The index of the instance that a given vertex belongs to.
    ///
    /// use `.to_u32()` or `.to_i32()` or `*`(deref) to use it for arithmetic.
    ///
    /// A [`VertexIndex`] is a special numeric index that can be used for vertex-buffer lookups.
    pub instance_index: VertexIndex,
    /// an iterator over the draw command's bound vertex buffers, which also
    /// allows random access
    ///
    /// use `.next()` or `.at(...)`/`.index(...)` to access individual vertex buffers
    pub buffers: VertexBufferIter,
    phantom: PhantomData<&'a ()>,
    private_ctor: (),
}

impl VertexStage<'_> {
    /// this ctor cannot be public, since it would allow creation of vertices in a compute pipeline
    #[track_caller]
    pub(crate) fn new() -> Self {
        VertexStage {
            index: VertexIndex::new(VertexBufferLookupIndex::VertexIndex),
            instance_index: VertexIndex::new(VertexBufferLookupIndex::InstanceIndex),
            buffers: VertexBufferIter::new(),
            phantom: PhantomData,
            private_ctor: (),
        }
    }

    /// assemble the per-vertex clip-space coordinates into primitives as described
    /// by the `draw` enum.
    ///
    /// call `.rasterize_*` on the returned object to rasterize the primitive.
    ///
    /// * `clip_space_position` can be a 2D, 3D or 4D vector of `f32` for rasterizing
    ///   2D or 3D shapes with or without perspective. If `clip_space_position` is a
    ///   4D vector it is treated as homogenous coordinates, which is useful for
    ///   perspective projection.
    ///
    /// see <https://gpuweb.github.io/gpuweb/#clip-space-coordinates>
    #[track_caller]
    pub fn assemble<Dim: Len>(self, clip_space_position: vec<f32, Dim>, draw: Draw) -> PrimitiveAssembly<false> {
        Context::try_with(call_info!(), |ctx| {
            ctx.push_error_if_outside_encoding_scope("primitive assembly");
        });
        record_node(
            call_info!(),
            BuiltinShaderOut::Position.into(),
            &[clip_space_position.to_gpu().ext_homo().as_any()],
        );
        PrimitiveAssembly { draw }
    }
}

/// a numeric index that can be used for vertex-buffer lookups.
///
/// In order to use it for arithmetic it needs to be either dereferenced (`*` operator),
/// converted to a `u32x1`, `i32x1` or `f32x1`
/// via the [`From`] trait, or converted via `.to_u32()`, `.to_i32()`, `.to_f32()` associated functions.
#[derive(Clone, Copy)]
pub struct VertexIndex(pub(crate) VertexBufferLookupIndex, pub(crate) u32x1);

impl VertexIndex {
    #[track_caller]
    pub(crate) fn new(lookup: VertexBufferLookupIndex) -> Self {
        Self(
            lookup,
            match lookup {
                VertexBufferLookupIndex::VertexIndex => Any::new_vertex_id().into(),
                VertexBufferLookupIndex::InstanceIndex => Any::new_instance_id().into(),
            },
        )
    }
}

impl std::ops::Deref for VertexIndex {
    type Target = u32x1;

    fn deref(&self) -> &Self::Target { &self.1 }
}

impl ToGpuType for VertexIndex {
    type Gpu = u32x1;

    #[track_caller]
    fn to_gpu(&self) -> Self::Gpu {
        let VertexIndex(lookup, value) = self;
        match *lookup {
            VertexBufferLookupIndex::VertexIndex => *value,
            VertexBufferLookupIndex::InstanceIndex => *value,
        }
    }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { None }
}

impl From<VertexIndex> for u32x1 {
    fn from(vi: VertexIndex) -> Self { vi.to_gpu() }
}

impl From<VertexIndex> for i32x1 {
    fn from(vi: VertexIndex) -> Self { Self::from(vi.to_gpu()) }
}

impl From<VertexIndex> for f32x1 {
    fn from(vi: VertexIndex) -> Self { Self::from(vi.to_gpu()) }
}

impl From<VertexIndex> for f64x1 {
    fn from(vi: VertexIndex) -> Self { Self::from(vi.to_gpu()) }
}

impl From<VertexIndex> for VertexBufferLookupIndex {
    fn from(vi: VertexIndex) -> Self { vi.0 }
}

/// (no documentation yet)
pub struct FragmentStage<SPP: Spp = Single> {
    // TODO(release) low prio: store a generation here, or add lifetime of the encoding
    /// access to [fragment quad] operations such as partial derivatives ([`FragmentQuad::grad`])
    ///
    /// see https://www.w3.org/TR/WGSL/#quad
    ///
    /// [fragment quad]: https://www.w3.org/TR/WGSL/#quad
    pub quad: FragmentQuad,
    /// the fragment position consisting of
    ///
    /// - `x = self.pixel_pos.x` (see [`FragmentStage::pixel_pos`])
    /// - `y = self.pixel_pos.y` (see [`FragmentStage::pixel_pos`])
    /// - `z = self.depth` (see [`FragmentStage::depth`])
    /// - `w = self.perspective_divisor` (see [`FragmentStage::perspective_divisor`])
    pub pos: f32x4,
    /// precise position of the fragment sample within the pixel grid
    ///
    /// values are in the range range `(0, 0)..color_targets_dimensions_in_pixels`
    ///
    /// see `fp.xy` at https://www.w3.org/TR/WGSL/#position-builtin-value
    pub pixel_pos: f32x2,
    /// the viewport depth
    ///
    /// by default this value is in the `0.0 ..= 1.0` range, however
    /// the graphics api can set a custom min and max depth which overrides these
    /// bounds.
    ///
    /// see `fp.z` at https://www.w3.org/TR/WGSL/#position-builtin-value
    pub depth: f32x1,
    /// the perspective divisor
    ///
    /// the per-fragment interpolated value of `1.0 / clip_space_position.w`
    ///
    /// see `fp.w` at https://www.w3.org/TR/WGSL/#position-builtin-value
    pub perspective_divisor: f32x1,
    /// the sample index within a rasterized pixel
    ///
    /// when using multisampling or supersampling,
    /// the value is least 0 and at most `sample_mask.len() - 1`, where `sample_mask`
    /// refers to the `sample_mask` argument passed to the `rasterize_*` function.
    ///
    /// `sample_index` is always 0 when not using multisampling or supersampling rasterization
    pub sample_index: u32x1,
    /// whether a given fragment is part of a counter clockwise primitive.
    ///
    /// line rasterization with non-triangle topology and point primitives are never
    /// considered counter clockwise.
    ///
    pub is_ccw_primitive: boolx1,
    /// The sample coverage mask per fragment.
    /// It contains a bitmask indicating which samples of a fragment are covered
    /// by the primitive being rendered.
    ///
    /// see https://www.w3.org/TR/WGSL/#sample-mask-builtin-value
    pub sample_mask: u32x1,
    /// access to depth-test, stencil-test and color attachments
    ///
    /// - `self.color_iter()` skips the depth/stencil tests and returns access
    ///   to the color target iterator
    /// - `self.attachments.depth_test<DepthFormat>(...)` performs depth-test and
    ///   returns access to the color target iterator. Alternative functions
    ///   for stencil test exist (`stencil_test`, `stencil_and_depth_test`)
    pub attachments: RenderPassAttachments<SPP>,
    fill_location_counter: Cell<u32>,
    phantom: PhantomData<(SPP)>,
    private_ctor: (),
}

/// (no documentation yet)
#[derive(Debug, Clone, Copy)]
pub struct FragmentQuad {
    pub(crate) rate: FragmentShadingRate,
    pub(crate) num_spp: usize,
}

/// (no documentation yet)
#[derive(Clone, Copy)]
pub struct Gradient<T> {
    /// (no documentation yet)
    pub dx: T,
    /// (no documentation yet)
    pub dy: T,
}

impl<T> Gradient<T> {
    /// (no documentation yet)
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Gradient<U> {
        Gradient {
            dx: f(self.dx),
            dy: f(self.dy),
        }
    }
}

fn rate_adjust_derivative_if_needed<L: Len>(t: vec<f32, L>, quad: FragmentQuad, distance: GradUnit) -> vec<f32, L> {
    let factor = derivative_rate_adjustment_factor(quad.num_spp);
    match distance {
        GradUnit::Sample if quad.is_supersample() => t * factor,
        _ => t,
    }
}

impl<T: TextureCoords> From<Gradient<T>> for MipFn<T> {
    fn from(g: Gradient<T>) -> Self { MipFn::Grad(g) }
}

impl<T: TextureCoords> From<FragmentQuad> for MipFn<T> {
    fn from(q: FragmentQuad) -> Self { MipFn::Quad(q) }
}

impl FragmentQuad {
    /// calculates `self.dx.abs() + self.dy.abs()` (also known as `fwidth`)
    ///
    /// which is the "manhattan length" of the vector `(self.dx, self.dy)`
    ///
    /// this function is called `fwidth` in common shader languages.
    /// It was renamed to be less confusing wrt. "width" vs "height" implying that
    /// only the `abs(self.dx)` component is used (which is not the case).
    /// see https://www.w3.org/TR/WGSL/#fwidth-builtin
    #[track_caller]
    pub fn dxy_manhattan<L: Len>(&self, value: vec<f32, L>, precision: GradPrecision) -> vec<f32, L> {
        let manhattan = value.as_any().fwidth(precision).into();
        rate_adjust_derivative_if_needed(manhattan, *self, GradUnit::default())
    }

    /// alternative naming for the `dxy_manhattan` function.
    ///
    /// `fwidth` is a common function in shader languages that computes
    /// `abs(self.dx) + abs(self.dy)`.
    /// the name `fwidth` is misleading wrt. "width" vs "height" implying that
    /// only the `abs(self.dx)` component is used (which is not the case).
    ///
    /// see https://www.w3.org/TR/WGSL/#fwidth-builtin.
    ///
    /// Use `dxy_manhattan` instead.
    #[deprecated = "this function has been renamed to `dxy_manhattan`"]
    #[doc(alias = "dxy_manhattan")]
    pub fn fwidth<L: Len>(&self, value: vec<f32, L>, precision: GradPrecision) -> vec<f32, L> {
        self.dxy_manhattan(value, precision)
    }
}

pub(crate) fn derivative_rate_adjustment_factor(num_spp: usize) -> f32 { 1.0 / f64::sqrt(num_spp as f64) as f32 }

/// supersampling in the target shader languages does not adjust the mip bias
/// of the texture sampling functions accordingly.
/// For example, in common shader languages, if sample-rate shading and
/// multisampling is enabled, while every fragment shader invocation runs per
/// sample, the implicit derivatives are still calculated with regard to
/// a 1 pixel distance
/// (source: https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#3.5.7%20Pixel%20Shader%20Derivatives)
/// and not adjusted for the higher sampling rate.
///
/// This function returns the additional mip bias required to counteract
/// that.
pub(crate) fn mip_bias_for_supersampling(sample_count: usize) -> f32 {
    match sample_count {
        0 => f32::NEG_INFINITY,
        // special cases for guaranteed precision
        1 => 0.0,
        4 => -1.0,
        16 => -2.0,
        64 => -3.0,
        256 => -4.0,
        n => -f32::log(n as f32, 4.0),
    }
}

impl ToGpuType for Gradient<vec<f32, x1>> {
    type Gpu = vec<f32, x2>;

    fn to_gpu(&self) -> Self::Gpu { (self.dx, self.dy).to_gpu() }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { None }
}

impl ToGpuType for Gradient<vec<f32, x2>> {
    type Gpu = vec<f32, x4>;

    fn to_gpu(&self) -> Self::Gpu { (self.dx, self.dy).to_gpu() }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { None }
}

impl<T> From<Gradient<T>> for [T; 2] {
    fn from(dt: Gradient<T>) -> Self { [dt.dx, dt.dy] }
}

impl<T> Gradient<T> {
    /// returns `[self.dx, self.dy]`
    pub fn into_array(self) -> [T; 2] { self.into() }
}

/// Whether partial derivatives should be adjusted to reflect the resolution
/// increase when using multisampling or supersampling
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradUnit {
    /// partial derivatives are adjusted according to the multisampling or supersampling
    /// resolution increase
    #[default]
    Sample,
    /// partial derivatives are calculated wrt. pixel distances even if multisampling
    /// or supersampling is used
    Pixel,
}

impl From<FragmentShadingRate> for GradUnit {
    fn from(value: FragmentShadingRate) -> Self {
        match value {
            FragmentShadingRate::PerPixel => GradUnit::Pixel,
            FragmentShadingRate::PerSample => GradUnit::Sample,
        }
    }
}

impl FragmentQuad {
    pub(crate) fn is_supersample(&self) -> bool {
        matches!((self.rate, self.num_spp), (ir::FragmentShadingRate::PerSample, 2..))
    }
}

#[track_caller]
pub(crate) fn new_sample_id(rate: FragmentShadingRate, spp: SamplesPerPixel) -> vec<u32, x1> {
    // TODO(release) the 0-value returned for single sample rasterization is not forced to be per-fragment.
    // that on its own doesn't cause any problems, but it means sample_index could technically be used
    // as a per-vertex value, and then once the shader gets switched to multisampling the stage-solver
    // would reject it, which would be confusing behavior.
    match (spp, rate) {
        (SamplesPerPixel::Multi, FragmentShadingRate::PerSample) => Any::new_sample_id().into(),
        _ => 0u32.to_gpu(),
    }
}

impl<SPP: Spp> FragmentStage<SPP> {
    #[track_caller]
    pub(crate) fn new(num_samples_per_pixel: usize, rate: FragmentShadingRate, draw: Draw) -> Self {
        let quad = FragmentQuad {
            rate,
            num_spp: num_samples_per_pixel,
        };
        let call_info = call_info!();
        Context::try_with(call_info!(), |ctx| {
            ctx.render_pipeline_mut().fragment_quad.set_once(quad);
        });

        // see https://www.w3.org/TR/WGSL/#position-builtin-value
        let pos: f32x4 = Any::new_fragment_position().into();
        let depth = pos.z;
        let pixel_pos = pos.xy();
        let perspective_divisor = pos.w;

        let is_ccw_primitive: boolx1 = match draw {
            Draw::Point => false.to_gpu(), // TODO(release) these should be per-fragment, but are not forced to be
            Draw::Line {
                strip,
                triangle_topology,
            } => match triangle_topology {
                true => Any::new_front_facing().into(),
                false => false.to_gpu(),
            },
            Draw::Triangle {
                strip: _,
                conservative: _,
                winding,
                z_clip: _,
            } => match winding {
                Winding::Ccw => true.to_gpu(),
                Winding::Cw => false.to_gpu(),
                Winding::Either => {
                    // `shame` forces `front_face` to be `Ccw`
                    Any::new_front_facing().into()
                }
            },
        };

        Self {
            quad,
            sample_index: new_sample_id(rate, SPP::SAMPLES_PER_PIXEL),
            phantom: PhantomData,
            pos,
            pixel_pos,
            depth,
            perspective_divisor,
            is_ccw_primitive,
            sample_mask: Any::new_sample_mask().into(),
            fill_location_counter: Cell::new(0),
            attachments: RenderPassAttachments { phantom: PhantomData },
            private_ctor: (),
        }
    }

    pub(crate) fn next_location(&self) -> Location {
        let mut i = self.fill_location_counter.get();
        self.fill_location_counter.set(i + 1);
        Location(i)
    }
}

/// access to depth-test, stencil-test and color attachments
///
/// - `self.color_iter()` skips the depth/stencil tests and returns access
///   to the color target iterator
/// - `self.attachments.depth_test<DepthFormat>(...)` performs depth-test and
///   returns access to the color target iterator. Alternative functions
///   for stencil test exist (`stencil_test`, `stencil_and_depth_test`)
pub struct RenderPassAttachments<SPP: Spp> {
    phantom: PhantomData<SPP>,
}

impl<SPP: Spp> RenderPassAttachments<SPP> {
    // skips the [`DepthTest`] or [`StencilTest`] and returns access to
    // the color targets via [`ColorTargetIter`]
    //
    // alternatively, call
    // - `self.depth_test`
    // - `self.stencil_test`
    // - `self.stencil_and_depth_test`
    //
    // to filter out fragments before writing to the color buffer.
    // those functions return the same [`ColorTargetIter`]
    /// (no documentation yet)
    #[allow(clippy::drop_non_drop)] // must consume
    #[track_caller]
    pub fn color_iter(self) -> ColorTargetIter<SPP> {
        drop(self);
        ColorTargetIter::new()
    }

    /// performs the [`DepthTest`] and returns access to
    /// the color targets via [`ColorTargetIter`]
    ///
    /// alternatively, call
    /// - `self.color_iter` (skips depth/stencil tests)
    /// - `self.stencil_test`
    /// - `self.stencil_and_depth_test`
    ///
    /// those functions return the same [`ColorTargetIter`]
    #[must_use]
    #[track_caller]
    pub fn depth_test<DepthBufferFormat: DepthFormat>(self, test: DepthTest) -> ColorTargetIter<SPP> {
        let fmt: TextureFormatWrapper = DepthBufferFormat::id().into();
        self.depth_stencil_test_internal(fmt.clone(), Some((fmt, test)), None)
    }

    // performs the [`StencilTest`] and returns access to
    // the color targets via [`ColorTargetIter`]
    //
    // alternatively, call
    // - `self.color_iter` (skips depth/stencil tests)
    // - `self.depth_test`
    // - `self.stencil_and_depth_test`
    //
    // those functions return the same [`ColorTargetIter`]
    /// (no documentation yet)
    #[must_use]
    #[track_caller]
    pub fn stencil_test<StencilBufferFormat: StencilFormat>(self, test: StencilTest) -> ColorTargetIter<SPP> {
        let fmt: TextureFormatWrapper = StencilBufferFormat::id().into();
        self.depth_stencil_test_internal(fmt.clone(), None, Some((fmt, test)))
    }

    // performs the [`StencilTest`] and [`DepthTest`] and returns access to
    // the color targets via [`ColorTargetIter`]
    //
    // alternatively, call
    // - `self.color_iter` (skips depth/stencil tests)
    // - `self.depth_test`
    // - `self.stencil_test`
    //
    // those functions return the same [`ColorTargetIter`]
    /// (no documentation yet)
    #[must_use]
    #[track_caller]
    pub fn stencil_and_depth_test<DepthStencilBufferFormat: DepthStencilFormat>(
        self,
        stencil_test: StencilTest,
        depth_test: DepthTest,
    ) -> ColorTargetIter<SPP> {
        self.depth_stencil_test_internal(
            DepthStencilBufferFormat::id().into(),
            // TODO(release) i do not remember why i separated it into Depth and Stencil aspect here
            Some((DepthStencilBufferFormat::Depth::id().into(), depth_test)),
            Some((DepthStencilBufferFormat::Stencil::id().into(), stencil_test)),
        )
    }

    #[allow(clippy::drop_non_drop)]
    #[track_caller]
    fn depth_stencil_test_internal(
        self,
        combined_fmt: TextureFormatWrapper,
        depth: Option<(TextureFormatWrapper, DepthTest)>,
        stencil: Option<(TextureFormatWrapper, StencilTest)>,
    ) -> ColorTargetIter<SPP> {
        drop(self);
        Context::try_with(call_info!(), |ctx| {
            ctx.push_error_if_outside_encoding_scope("depth/stencil test");

            if let Some((_, dt)) = depth {
                match dt.operand {
                    DepthLhs::FragmentZ(_) => (),
                    DepthLhs::Explicit(depth_val, _) => {
                        record_node(
                            ctx.latest_user_caller(),
                            BuiltinShaderOut::FragDepth.into(),
                            &[depth_val.as_any()],
                        );
                    }
                }
            }

            let mut p = ctx.render_pipeline_mut();

            let draw_supports_depth_bias = p
                .draw
                .get()
                .map(|(draw, _)| draw.is_triangle_topology())
                .unwrap_or(true);
            if let Some((_, depth_test)) = &depth {
                let bias = match depth_test.operand {
                    DepthLhs::Explicit(_, bias) => bias,
                    DepthLhs::FragmentZ(bias) => bias,
                };
                if !draw_supports_depth_bias && bias != DepthBias::zero() {
                    ctx.push_error(PipelineError::NonZeroDepthBiasRequiresTriangles.into());
                }
            }

            p.depth_stencil.set(WipDepthStencilState {
                format: combined_fmt,
                depth_test: depth.map(|d| d.1),
                stencil_test: stencil.map(|d| d.1),
            });
        });
        ColorTargetIter::new()
    }
}

pub(crate) enum DeltaComp {
    X,
    Y,
}

#[allow(private_interfaces)]
impl<L: Len> vec<f32, L> {
    #[track_caller]
    fn partial_derivative(
        &self,
        cmp: DeltaComp,
        precision: GradPrecision,
        quad: FragmentQuad,
        distance: GradUnit,
    ) -> Self {
        let any = self.as_any();
        rate_adjust_derivative_if_needed(
            match cmp {
                DeltaComp::X => any.ddx(precision),
                DeltaComp::Y => any.ddy(precision),
            }
            .into(),
            quad,
            distance,
        )
    }
}

impl FragmentQuad {
    /// approximate gradient of `value` in [framebuffer coordinates]
    ///
    /// This calculation uses the individual values of `value` in the
    /// [derivative group], which in most cases is comprised of a quad of
    /// four shader threads to estimate partial derivatives wrt. [framebuffer coordinates].
    ///
    /// for more details see https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#shaders-derivative-operations
    ///
    /// > an alternative function called [FragmentQuad::grad_rate] exists, which
    /// > allows adjustment of unit and precision
    ///
    /// ## Example
    /// ```
    /// use shame as sm;
    /// let uv: sm::f32x2 = ...;
    /// let duv = self.quad.grad(uv, sm::GradPrecision::Fine);
    ///
    /// // access the individual partial derivatives
    /// let something = duv.dx + duv.dy;
    /// ```
    ///
    /// note: the sign of the returned [`Gradient::dy`] component is flipped
    /// from what you might expect when passing in clip space coordinates.
    /// This is because the framebuffer coordinate system has a different
    /// y-axis orientation than clip-space.
    ///
    /// see <https://www.w3.org/TR/webgpu/#coordinate-systems>
    ///
    /// [derivative group]: https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#shaders-derivative-operations
    /// [dpdx](https://www.w3.org/TR/WGSL/#dpdx-builtin)
    /// [dpdy](https://www.w3.org/TR/WGSL/#dpdy-builtin)
    /// [framebuffer coordinates](https://www.w3.org/TR/webgpu/#coordinate-systems)
    #[track_caller]
    pub fn grad<L: Len>(&self, value: vec<f32, L>, precision: GradPrecision) -> Gradient<vec<f32, L>> {
        self.grad_rate(value, precision, GradUnit::default())
    }

    /// approximate gradient of `value` in [framebuffer coordinates]
    ///
    /// This calculation uses the individual values of `value` in the
    /// [derivative group], which in most cases is comprised of a quad of
    /// four shader threads to estimate partial derivatives wrt. [framebuffer coordinates].
    ///
    /// for more details see https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#shaders-derivative-operations
    ///
    /// > an alternative shorthand called [FragmentQuad::grad] exists
    ///
    /// ## Example
    /// ```
    /// use shame as sm;
    /// let uv: sm::f32x2 = ...;
    /// let duv = self.quad.grad_rate(uv, sm::DxyPrecision::Fine, sm::DxyUnit::Sample);
    ///
    /// // access the individual partial derivatives
    /// let something = duv.dx + duv.dy;
    /// ```
    ///
    /// note: the sign of the returned [`Gradient::dy`] component is flipped
    /// from what you might expect when passing in clip space coordinates.
    /// This is because the framebuffer coordinate system has a different
    /// y-axis orientation than clip-space.
    ///
    /// see <https://www.w3.org/TR/webgpu/#coordinate-systems>
    ///
    /// [derivative group]: https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#shaders-derivative-operations
    /// [dpdx](https://www.w3.org/TR/WGSL/#dpdx-builtin)
    /// [dpdy](https://www.w3.org/TR/WGSL/#dpdy-builtin)
    /// [framebuffer coordinates](https://www.w3.org/TR/webgpu/#coordinate-systems)
    #[track_caller]
    pub fn grad_rate<L: Len>(
        &self,
        value: vec<f32, L>,
        precision: GradPrecision,
        distance: GradUnit,
    ) -> Gradient<vec<f32, L>> {
        Gradient {
            dx: value.partial_derivative(DeltaComp::X, precision, *self, distance),
            dy: value.partial_derivative(DeltaComp::Y, precision, *self, distance),
        }
    }
}

/// (no documentation yet)
pub struct PrimitiveAssembly<const CLIP: bool> {
    draw: Draw,
}

impl<const CLIP: bool> PrimitiveAssembly<CLIP> {
    /// obtain fragments by rasterizing all assembled primitives.
    /// see <https://en.wikipedia.org/wiki/Rasterization>.
    ///
    /// * `accuracy`: whether certain floating point operations are allowed to
    ///   speed up the calculation of the clip space coordinates
    ///
    /// alternative functions exist for multisampling/supersampling
    /// - [`rasterize_multisample`] for "MSAA" (multisample anti aliasing)
    /// - [`rasterize_supersample`] for "SSAA" (supersample anti aliasing)
    ///
    /// [`rasterize_multisample`]: PrimitiveAssembly::rasterize_multisample
    /// [`rasterize_supersample`]: PrimitiveAssembly::rasterize_supersample
    #[track_caller]
    pub fn rasterize(self, accuracy: Accuracy) -> FragmentStage<Single> {
        self.rasterize_internal(BitVec64::full(1), accuracy, FragmentShadingRate::PerPixel)
    }

    #[track_caller]
    /// obtain fragments by rasterizing all assembled primitives.
    /// One fragment is generated per pixel, but each fragment has a sample bitmask
    /// that represents the covered parts (samples) of that pixel.
    /// see <https://docs.vulkan.org/tutorial/latest/10_Multisampling.html>.
    ///
    /// * the `sample_mask` argument defines which samples are rasterized (1-bit) and which
    ///   ones are ignored (0-bit). The length of the `sample_mask` defines how many
    ///   samples every pixel contains. This length must be at least 2, otherwise
    ///   the pipeline encoding fails with an [`EncodingError`]
    ///
    /// * `accuracy`: whether certain floating point operations are allowed to
    ///   speed up the calculation of the clip space coordinates
    ///
    /// alternative functions exist for single-sampling/supersampling
    /// - [`rasterize`] for regular (single sample per pixel) rasterization
    /// - [`rasterize_supersample`] for "SSAA" (supersample anti aliasing)
    ///
    /// [`rasterize`]: PrimitiveAssembly::rasterize
    /// [`rasterize_supersample`]: PrimitiveAssembly::rasterize_supersample
    pub fn rasterize_multisample(self, sample_mask: BitVec64, accuracy: Accuracy) -> FragmentStage<Multi> {
        self.rasterize_internal(sample_mask, accuracy, FragmentShadingRate::PerPixel)
    }

    /// obtain fragments by rasterizing all assembled primitives.
    /// One fragment is generated per sample, where one pixel has multiple samples.
    /// see <https://en.wikipedia.org/wiki/Supersampling>.
    ///
    /// * the `sample_mask` argument defines which samples are rasterized (1-bit) and which
    ///   ones are ignored (0-bit). The length of the `sample_mask` defines how many
    ///   samples every pixel contains. This length must be at least 2, otherwise
    ///   the pipeline encoding fails with an [`EncodingError`]
    ///
    /// * `accuracy`: whether certain floating point operations are allowed to
    ///   speed up the calculation of the clip space coordinates
    ///
    /// alternative functions exist for single-sampling/multisampling
    /// - [`rasterize`] for regular (single sample per pixel) rasterization
    /// - [`rasterize_multisample`] for "MSAA" (multisample anti aliasing)
    ///
    /// > note: the following document describes how supersampling is achieved on the
    /// > different API targets:
    /// > https://gist.github.com/RayMarch/b9c302155bd405d45ddd7740697485c4
    ///
    /// [`EncodingError`]: crate::EncodingError
    /// [`rasterize`]: PrimitiveAssembly::rasterize
    /// [`rasterize_supersample`]: PrimitiveAssembly::rasterize_supersample
    #[track_caller]
    pub fn rasterize_supersample(self, sample_mask: BitVec64, accuracy: Accuracy) -> FragmentStage<Multi> {
        if cfg!(feature = "debug_print") && cfg!(debug_assertions) {
            println!(
                "TODO(release) write an integration test that checks if wgsl honors the spec even if @interpolate(sample) __supersample is unused"
            );
        }
        self.rasterize_internal(sample_mask, accuracy, FragmentShadingRate::PerSample)
    }

    #[track_caller]
    fn rasterize_internal<SPP: Spp>(
        self,
        sample_mask: BitVec64,
        accuracy: Accuracy,
        shading_rate: FragmentShadingRate,
    ) -> FragmentStage<SPP> {
        let PrimitiveAssembly { draw } = self;
        let call_info = call_info!();
        let num_spp = sample_mask.len();
        Context::try_with(call_info, |ctx| {
            ctx.push_error_if_outside_encoding_scope("rasterization");

            let mut p = ctx.render_pipeline_mut();
            p.draw.set_once(draw);
            p.deterministic_clip_pos.set(matches!(accuracy, Accuracy::Reproducible));
            p.fragment_shading_rate.set(shading_rate);
            p.sample_mask.set(match SPP::SAMPLES_PER_PIXEL {
                ir::SamplesPerPixel::Single => match sample_mask.len() {
                    1 => sample_mask,
                    n => {
                        ctx.push_error(PipelineError::TooFewSamplesInMultisampleMask(n).into());
                        BitVec64::full(1)
                    }
                },
                ir::SamplesPerPixel::Multi => match sample_mask.len() {
                    2.. => sample_mask,
                    n => {
                        ctx.push_error(PipelineError::TooFewSamplesInMultisampleMask(n).into());
                        BitVec64::full(4)
                    }
                },
            })
        });
        FragmentStage::new(
            match SPP::SAMPLES_PER_PIXEL {
                ir::SamplesPerPixel::Multi => num_spp,
                ir::SamplesPerPixel::Single => 1,
            },
            shading_rate,
            draw,
        )
    }
}

impl PrimitiveAssembly<false> {
    /// (no documentation yet)
    #[track_caller]
    pub fn clip<const N: usize>(mut self, signed_clip_distances: Array<f32x1, Size<N>>) -> PrimitiveAssembly<true>
    where
        Size<N>: UpTo8,
    {
        Context::try_with(call_info!(), |ctx| {
            ctx.push_error_if_outside_encoding_scope("primitive clipping");
        });
        record_node(
            call_info!(),
            BuiltinShaderOut::ClipDistances {
                count: Size::<N>::nonzero(),
            }
            .into(),
            &[signed_clip_distances.to_any()],
        );
        PrimitiveAssembly { draw: self.draw }
    }
}

/// controls whether certain floating point optimizations can be performed
/// that might lead to a slightly different set of pixels/samples being rasterized.
///
/// in the target language this describes the presence of
/// - WGSL: `@invariant` attribute
/// - GLSL: `invariant` qualifier
/// - HLSL: `precise` qualifier
///
/// on the clip space position.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accuracy {
    #[default]
    /// enables floating point optimizations that can lead to minimally
    /// different pixels/samples being rasterized, even if two different pipelines
    /// perform the exact same floating point calculations.
    ///
    /// This does not cause "cracks" between neighboring polygons that use the
    /// same pipeline.
    ///
    /// Considered to be the faster option.
    Relaxed, // consider renaming to "Optimize"?
    /// ensures that two different render pipelines that perform the exact
    /// same floating point calculations leading to the clip space position
    /// also result in the exact same pixels/samples being rasterized.
    /// Some floating point optimizations will be disabled to achieve this.
    ///
    /// This is useful for techniques like "depth prepass".
    ///
    /// ---
    /// corresponds to
    /// - WGSL: `@invariant` attribute
    /// - GLSL: `invariant` qualifier
    /// - HLSL: `precise` qualifier
    ///
    /// applied to the clip space position vertex shader output.
    Reproducible,
}

/// whether primitives are clipped when extending beyond the near/far plane of
/// the clip space.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZClip {
    /// clip polygons that penetrate the near or far plane at the intersection line.
    /// The parts of polygons in front of the near plane and behind the far plane will
    /// not be rendered.
    #[default]
    NearFar,
    /// keep the parts of polygons that penetrate the near or far plane.
    /// These parts will be drawn with depth=0.0 (near-plane) and depth=1.0 (far-plane)
    /// respectively
    Off,
}

/// ### Winding of polygon faces
/// A triangle's vertices are arranged either clockwise `Cw` or counter-clockwise `Ccw`
/// when iterating over them in [`Indexing`] sequence order.
/// You can decide to only rasterize `Cw` or `Ccw` triangles respectively,
/// this feature is commonly referred to as "backface culling".
///
/// ### why is there no "back face" in shame?
/// this question is answered in detail here:
/// <https://gist.github.com/RayMarch/045f92dee5d911e144f8dd7fece219a2>
///
/// In the future this enum may be replaced by something that does not rely
/// on a definition of "up/down/left/right", such as the signed triangle area:
/// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFrontFace.html
/// (no documentation yet)
#[derive(Default, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Winding {
    #[default]
    /// (default) Counter clockwise, the winding order usually used for front facing polygons.
    Ccw,
    /// Clockwise, the winding order usually used for back facing polygons.
    Cw,
    /// either counter clockwise and clockwise
    Either,
}

/// How the vertex sequence should be assembled into primitives,
/// as well as other details about culling, rasterization etc.
///
/// see https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-primitive-topologies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Draw {
    /// draw points at the clip space positions provided by each thread
    ///
    /// https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-point-lists
    Point,
    /// draw lines between the clip space positions provided by each thread
    Line {
        /// whether to use "strip" or "list" primitive topology
        ///
        /// - `true`: interpret the vertex sequence as
        ///   [triangle strips](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-triangle-strips)
        ///   or [line strips](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-line-strips)
        ///   depending on the value of [`Draw::Line::triangle_topology`].
        ///
        /// - `false`: interpret the vertex sequence as
        ///   [triangle lists](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-triangle-lists)
        ///   or [line lists](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-line-lists)
        ///   depending on the value of [`Draw::Line::triangle_topology`].
        ///
        /// when using strip topology, the strips can be separated by using a
        /// special index value in the index buffer. For `[u8]`/`[u16]`/`[u32]` index buffers,
        /// strips can be separated with the value `u8::MAX` or `u16::MAX` or `u32::MAX` respectively.
        ///
        strip: bool,
        /// whether to use "triangle" or "line" primitive topology
        ///
        /// - `true` interpret the vertex sequence as
        ///   [triangle strips](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-triangle-strips)
        ///   or [triangle lists](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-triangle-lists)
        ///   depending on the value of [`Draw::Line::strip`].
        ///
        /// - `false` interpret the vertex sequence as  
        ///   [line strips](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-line-strips)
        ///   or [line lists](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-line-lists)
        ///   depending on the value of [`Draw::Line::strip`].
        triangle_topology: bool,
    },
    /// draw a filled triangle formed by the clip space positions of three threads each.
    ///
    /// If you want to draw wireframe triangles, choose `Line { triangle_topology: true, .. }`
    Triangle {
        /// whether to use triangle strip or triangle list topology
        ///
        /// - `true`: use [triangle strip](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-triangle-strips)
        ///   topology.
        ///
        /// - `false`: use [triangle list](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-triangle-lists)
        ///   topology.
        ///
        /// when using [triangle strip](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#drawing-triangle-strips)
        /// topology, the strips an be separated by using a special index value
        /// in the index buffer. For `[u8]`/`[u16]`/`[u32]` index buffers,
        /// strips can be separated with the value `u8::MAX` or `u16::MAX` or `u32::MAX` respectively.
        strip: bool,
        /// how to decide if a pixel is filled
        ///
        /// - `true`: if the triangle overlaps the pixel square
        /// - `false`: if the pixel's center point is inside the triangle
        conservative: bool,
        /// determines which triangles are drawn based on winding order
        ///
        /// when following the clip space positions provided by thread #0, #1, #2, ...
        /// in ascending order, they form triangles in either a clockwise
        /// or counter clockwise way.
        ///
        /// - `Winding::Ccw`: only draw "counter clockwise" triangles
        /// - `Winding::Cw`: only draw "clockwise" triangles
        /// - `Winding::Either`: all triangles should be drawn regardless of winding order
        ///
        /// see https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#primsrast-polygons-basic
        winding: Winding,
        /// whether to "clip" away the part of the triangle that passes through the near/far plane
        ///
        /// see "depth clipping" at https://www.w3.org/TR/webgpu/#primitive-clipping
        z_clip: ZClip,
    },
}

impl Default for Draw {
    fn default() -> Self {
        Draw::Triangle {
            strip: false,
            conservative: false,
            winding: Winding::default(),
            z_clip: ZClip::NearFar,
        }
    }
}

impl Draw {
    pub(crate) fn is_triangle_topology(&self) -> bool {
        match self {
            Draw::Point => false,
            Draw::Line {
                strip,
                triangle_topology,
            } => *triangle_topology,
            Draw::Triangle {
                strip,
                conservative,
                winding,
                z_clip,
            } => true,
        }
    }

    /// (no documentation yet)
    pub fn triangle_list(winding: Winding) -> Self {
        Draw::Triangle {
            strip: false,
            conservative: false,
            winding,
            z_clip: ZClip::NearFar,
        }
    }

    /// (no documentation yet)
    pub fn triangle_strip(winding: Winding) -> Self {
        Draw::Triangle {
            strip: true,
            conservative: false,
            winding,
            z_clip: ZClip::NearFar,
        }
    }

    /// (no documentation yet)
    pub fn line_list() -> Self {
        Draw::Line {
            strip: false,
            triangle_topology: false,
        }
    }

    /// (no documentation yet)
    pub fn line_strip() -> Self {
        Draw::Line {
            strip: true,
            triangle_topology: false,
        }
    }

    /// (no documentation yet)
    pub fn points() -> Self { Draw::Point }

    /// whether `self` has strip topology, where every vertex in the sequence starts a new primitive
    pub fn is_strip(&self) -> bool {
        match self {
            Draw::Point => false,
            Draw::Line {
                strip,
                triangle_topology,
            } => *strip,
            Draw::Triangle {
                strip,
                conservative,
                winding,
                z_clip,
            } => *strip,
        }
    }
}
