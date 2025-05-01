use crate::{
    call_info,
    frontend::{
        any::{
            render_io::{FragmentSampleMethod, FragmentSamplePosition, Location},
            Any, InvalidReason,
        },
        error::InternalError,
        rust_types::{layout_traits::VertexLayout, len::Len, scalar_type::ScalarType, vec::vec, AsAny, GpuType},
        texture::texture_traits::{Multi, Single, Spp},
    },
    ir::{
        self,
        recording::{CallInfoScope, Context},
        FragmentShadingRate,
    },
    Ref,
};

use super::rasterizer::FragmentStage;

/// Whether to fill the fragments of a primitive by using
/// lienar interpolation or perspective-correct interpolation.
///
/// see https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#shaders-interpolation-decorations
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    /// perform perspective-division on the per-vertex values, using the `w` coordinate
    /// of the clip-space position provided to the [`rasterize`] function.
    ///
    /// This achieves perspective-correct interpolation results.
    ///
    /// see https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#vertexpostproc-coord-transform
    ///
    /// [`rasterize`] = crate::PrimitiveAssembly::rasterize
    #[default]
    Perspective,
    /// Use linear interpolation of the per-vertex values to determine the per-fragment values.
    ///
    /// This yields non perspective-correct results for 3D objects in non-orthographic projections.
    ///
    /// see https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#shaders-interpolation-decorations
    Linear,
}

/// Which of the primitive's vertices values is read and assigned to all
/// fragments when using [`FragmentStage::fill_flat`] interpolation.
///
/// see https://www.w3.org/TR/WGSL/#interpolation
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PickVertex {
    /// pick the first vertex of the triangle.
    ///
    /// after turning the [`Indexing`] sequence into tuples of N (3 for triangles, 2 for lines, ...)
    /// each of which represent the vertex-indices of a primitive, the
    /// first vertex in each tuple provides its value to all fragments that lie
    /// inside the primitive.
    ///
    /// [`Indexing`]: crate::VertexIndexing
    First,
    /// either the first or last vertex of the primitive is chosen.
    ///
    /// Whether first or last is picked is decided by the implementation.
    ///
    /// after turning the [`Indexing`] sequence into tuples of N (3 for triangles, 2 for lines, ...)
    /// each of which represent the vertex-indices of a primitive, the
    /// first or last vertex in each tuple provides its value to all fragments that lie
    /// inside the primitive.
    ///
    /// [`Indexing`]: crate::VertexIndexing
    ///
    /// see https://www.w3.org/TR/WGSL/#interpolation
    #[default]
    Either,
}

impl<SPP: Spp> FragmentStage<SPP> {
    #[track_caller]
    #[must_use]
    /// fill by picking the value of one of the primitive's vertices, assign
    /// that value to every fragment of the primitive.
    ///
    /// `pick` specifies which of the primitive's vertices should be chosen.
    /// see [`PickVertex`].
    ///
    /// see the [`FragmentStage::fill`] documentation for an explanation on how
    /// fill works in general
    pub fn fill_flat<T: PrimitiveInterpolatable>(&self, pick: PickVertex, per_vertex: T) -> T::Output {
        per_vertex.interpolate_across_primitive(&PrimitiveInterpolation::Flat(pick, self), &|| self.next_location())
    }

    #[track_caller]
    #[must_use]
    /// fill by sampling the primitive according to the fragment's shading rate.
    /// This is the default way of interpolating per-vertex values to obtain per-fragment values.
    ///
    /// the fragment shading rate depends on the chosen rasterization function:
    /// - [`rasterize`]: per-pixel shading (interpolate at pixel-center)
    /// - [`rasterize_multisample`]: per-pixel shading (interpolate at pixel-center)
    /// - [`rasterize_supersample`]: per-sample shading (interpolate at supersampling sample position)
    ///
    /// see the [`FragmentStage::fill`] documentation for an explanation on how
    /// primitive interpolation works in general
    ///
    /// [`rasterize`]: crate::PrimitiveAssembly::rasterize
    /// [`rasterize_multisample`]: crate::PrimitiveAssembly::rasterize_multisample
    /// [`rasterize_supersample`]: crate::PrimitiveAssembly::rasterize_supersample
    pub fn fill_rate<T: PrimitiveInterpolatable>(&self, fill: Fill, per_vertex: T) -> T::Output {
        per_vertex.interpolate_across_primitive(&PrimitiveInterpolation::Default(fill, self), &|| self.next_location())
    }

    /// fill all fragments of the primitive with the provided values
    ///
    /// This function takes per-vertex values, interpolates them
    /// at a different sampling position for every fragment in the primitive
    /// and returns per-fragment values.
    ///
    /// > note: In the generated shader code, this function's `per_vertex`
    /// > argument becomes a vertex-stage output, and the returned value is the
    /// > corresponding fragment stage input.
    ///
    /// `fill` is a shorthand for the most common primitive interpolation,
    /// which is perspective-aware per-fragment sampling of the primitive:
    /// ```
    /// let fragment_uv = frag.fill(vertex_uv);
    /// // is the same as:
    /// let fragment_uv = frag.fill_rate(shame::Fill::Perspective, vertex_uv);
    /// // where "rate" refers to the fragment's shading rate (per-pixel vs per-sample)
    /// ```
    /// > see [`Fill::Perspective`] for more info on how perspective is handled.
    ///
    /// There are multiple different [`FragmentStage`]`::fill_*` functions, each
    /// of which allow different kinds of sampling:
    /// ```
    /// let frag_uv = frag.fill_rate(linear_or_perspective, vert_uv);
    /// let frag_uv = frag.fill_pixel_center(linear_or_perspective, vert_uv);
    /// let frag_uv = frag.fill_centroid(linear_or_perspective, vert_uv);
    /// let frag_uv = frag.fill_flat(first_or_either, vert_uv);
    /// ```
    /// see their respective documentation for more info.
    /// - [FragmentStage::fill_rate]
    /// - [FragmentStage::fill_pixel_center]
    /// - [FragmentStage::fill_centroid]
    /// - [FragmentStage::fill_flat]
    ///
    /// for more info on primitive interpolation in general see:
    /// https://www.w3.org/TR/WGSL/#interpolation
    ///
    /// [`FragmentStage::fill_*`]: FragmentStage::fill_rate
    #[track_caller]
    #[must_use]
    pub fn fill<T: PrimitiveInterpolatable>(&self, per_vertex: T) -> T::Output {
        self.fill_rate(Fill::Perspective, per_vertex)
    }
}

impl FragmentStage<Multi> {
    /// fill by sampling once per pixel at the centroids.
    ///
    /// see the [`FragmentStage::fill`] documentation for an explanation on how
    /// primitive interpolation works in general, the following text focuses on
    /// centroid sampling.
    ///
    /// The centroid is
    /// - the pixel center if the pixel is fully covered by the primitive
    /// - A point that lies both within the pixel square and the primitive.
    ///   
    /// > Some Graphics APIs define the centroid is the average
    /// > of all covered samples in a pixel, which leads to the unclear case
    /// > in conservative rasterization, when no sample is covered but the primitive
    /// > overlaps the pixel square.
    ///
    /// the intention of `fill_centroid` is to prevent situations in which per-vertex values
    /// are interpolated at a pixel-center that lies barely outside of the polygon,
    /// by slightly moving those pixel-centers into the covered region of that pixel.
    ///
    /// The downside of this approach is that [`FragmentQuad`] based `[Gradient]`s
    /// get broken results at partially covered pixels, since the distances between
    /// quad fragments becomes irregular at those pixels.
    ///
    #[track_caller]
    #[must_use]
    pub fn fill_centroid<T: PrimitiveInterpolatable>(&self, fill: Fill, per_vertex: T) -> T::Output {
        per_vertex.interpolate_across_primitive(
            &PrimitiveInterpolation::MultisampleCentroid::<Multi>(fill, self),
            &|| self.next_location(),
        )
    }

    /// fill by sampling at the pixel centers.
    /// (only relevant for supersampling rasterization)
    ///
    /// see the [`FragmentStage::fill`] documentation for an explanation on how
    /// primitive interpolation works in general, the following text focuses on
    /// pixel-center sampling.
    ///
    /// for every fragment that belongs to a pixel, that pixel's center is chosen
    /// as the interpolation position of the per-vertex values.
    /// When using supersampling rasterization, this means that all fragments of
    /// a pixel get the identical value.
    ///
    /// If this function is called in a non-supersampling context, it has the same
    /// effect as the [`FragmentStage::fill_rate`] interpolation,
    /// since the shading rate of the fragment is already "per-pixel"
    #[track_caller]
    #[must_use]
    pub fn fill_pixel_center<T: PrimitiveInterpolatable>(&self, per_vertex: T, fill: Fill) -> T::Output {
        per_vertex.interpolate_across_primitive(
            &PrimitiveInterpolation::SupersampleForcePixelCenter::<Multi>(fill, self),
            &|| self.next_location(),
        )
    }
}

#[track_caller]
fn vec_fill_any<T: ScalarType, L: Len>(v: &vec<T, L>, method: FragmentSampleMethod, loc: Location) -> Any {
    match T::SCALAR_TYPE {
        ir::ScalarType::F16 | ir::ScalarType::F32 | ir::ScalarType::F64 => v.as_any(),
        ir::ScalarType::U32 | ir::ScalarType::I32 | ir::ScalarType::Bool => {
            // convert to vec<f32, L> then fill.
            Any::new_vec(L::LEN, ir::ScalarType::F32, &[v.as_any()])
        }
    }
    .fill_fragments(loc, method)
}

pub enum PrimitiveInterpolation<'a, SPP: Spp> {
    Default(Fill, &'a FragmentStage<SPP>),
    MultisampleCentroid(Fill, &'a FragmentStage<Multi>),
    SupersampleForcePixelCenter(Fill, &'a FragmentStage<Multi>),
    Flat(PickVertex, &'a FragmentStage<SPP>),
}

/// (no documentation yet)
/// TODO(docs) mention tuples and vecs, probably best to make a list of examples
pub trait PrimitiveInterpolatable {
    /// the per-fragment output type after interpolation
    type Output;

    #[doc(hidden)] // internal
    fn interpolate_across_primitive<SPP: Spp>(
        &self,
        interpolation: &PrimitiveInterpolation<SPP>,
        next_location: &impl Fn() -> Location,
    ) -> Self::Output;
}

impl<T: ScalarType, L: Len> PrimitiveInterpolatable for vec<T, L> {
    type Output = vec<T::LerpOutput, L>;

    #[track_caller]
    fn interpolate_across_primitive<SPP: Spp>(
        &self,
        interpolation: &PrimitiveInterpolation<SPP>,
        next_location: &impl Fn() -> Location,
    ) -> Self::Output {
        use ir::FragmentShadingRate as Rate;
        use FragmentSampleMethod as F;
        use PrimitiveInterpolation as P;

        let shading_rate = Context::try_with(call_info!(), |ctx| {
            let p = ctx.render_pipeline();
            match p.fragment_quad.get() {
                Some((quad, _)) => Ok(quad.rate),
                None => Err(ctx.push_error_get_invalid_any(
                    InternalError::new(
                        true,
                        "`interpolate_across_primitive` fragment quad information not recorded yet".into(),
                    )
                    .into(),
                )),
            }
        })
        .unwrap_or(Err(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding)));

        let shading_rate = match shading_rate {
            Ok(r) => r,
            Err(invalid) => return invalid.into(),
        };

        let default_positions = match (SPP::SAMPLES_PER_PIXEL, shading_rate) {
            // single-sampling has one fragment per pixel
            (ir::SamplesPerPixel::Single, Rate::PerPixel | Rate::PerSample) => FragmentSamplePosition::PixelCenter,
            // multisampling has one fragment per pixel
            (ir::SamplesPerPixel::Multi, Rate::PerPixel) => FragmentSamplePosition::PixelCenter,
            // supersampling has one fragment per sample
            (ir::SamplesPerPixel::Multi, Rate::PerSample) => FragmentSamplePosition::PerSample,
        };
        let method = match interpolation {
            P::Default(fill, _) => F::Interpolated(*fill, default_positions),
            P::MultisampleCentroid(fill, _) => F::Interpolated(*fill, FragmentSamplePosition::Centroid),
            P::SupersampleForcePixelCenter(fill, _) => F::Interpolated(*fill, FragmentSamplePosition::PixelCenter),
            P::Flat(pick, _) => F::Flat(*pick),
        };
        vec_fill_any(self, method, next_location()).into()
    }
}

impl<T: ScalarType, L: Len> PrimitiveInterpolatable for Ref<vec<T, L>> {
    type Output = vec<T::LerpOutput, L>;

    #[track_caller]
    fn interpolate_across_primitive<SPP: Spp>(
        &self,
        interpolation: &PrimitiveInterpolation<SPP>,
        next_location: &impl Fn() -> Location,
    ) -> Self::Output {
        self.get().interpolate_across_primitive(interpolation, next_location)
    }
}

macro_rules! impl_tuple_interpolation {
    (
        $($($A: ident),*;)*
    ) => {
        $(
            impl<$($A: PrimitiveInterpolatable),*> PrimitiveInterpolatable for ($($A,)*) {
                type Output = ($($A::Output,)*);

                #[track_caller]
                fn interpolate_across_primitive<SPP: Spp>(&self, interpolation: &PrimitiveInterpolation<SPP>, next_location: &impl Fn() -> Location)
                -> Self::Output {
                    #[allow(non_snake_case)]
                    let ($($A,)*) = self;
                    ($($A.interpolate_across_primitive(interpolation, next_location) ,)*)
                }
            }
        )*
    };
}

impl PrimitiveInterpolatable for () {
    type Output = ();

    fn interpolate_across_primitive<SPP: Spp>(
        &self,
        _: &PrimitiveInterpolation<SPP>,
        _: &impl Fn() -> Location,
    ) -> Self::Output {
    }
}

impl_tuple_interpolation! {
    A;
    A, B;
    A, B, C;
    A, B, C, D;
    A, B, C, D, E;
    A, B, C, D, E, F;
    A, B, C, D, E, F, G;
    A, B, C, D, E, F, G, H;
}

impl<T: PrimitiveInterpolatable, const N: usize> PrimitiveInterpolatable for [T; N] {
    type Output = [T::Output; N];

    #[track_caller]
    fn interpolate_across_primitive<SPP: Spp>(
        &self,
        interpolation: &PrimitiveInterpolation<SPP>,
        next_location: &impl Fn() -> Location,
    ) -> Self::Output {
        let scope = CallInfoScope::new(call_info!());
        self.each_ref()
            .map(|x| x.interpolate_across_primitive(interpolation, next_location))
    }
}
