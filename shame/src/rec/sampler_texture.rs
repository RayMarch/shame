//! sampler and texture types
use super::{narrow_stages_or_push_error, AnyDowncast, AsTen, IsShapeScalarOrVec, Shape, Stage};
use super::{DType, Ten, TexCoordType};
use crate::{float, float2, float3, float4};
use shame_graph::TexDtypeDimensionality;
use shame_graph::{Any, OpaqueTy};
use std::marker::PhantomData;

/// the output type when sampling from a texture
pub trait TexSampleType: AsTen {}
impl<S: IsShapeScalarOrVec> TexSampleType for Ten<S, f32> {}
impl<S: IsShapeScalarOrVec> TexSampleType for Ten<S, i32> {}
impl<S: IsShapeScalarOrVec> TexSampleType for Ten<S, u32> {}

/// alias for textures that return a `float4` when sampled
pub type TextureRGBA<In = float2> = Texture<float4, In>;
/// alias for textures that return a `float3` when sampled
pub type TextureRGB<In = float2> = Texture<float3, In>;
/// alias for textures that return a `float2` when sampled
pub type TextureRG<In = float2> = Texture<float2, In>;
/// alias for textures that return a `float` when sampled
pub type TextureR<In = float2> = Texture<float, In>;

/// a texture binding, which can be sampled with
/// texture coordinates of type `In` to obtain a sample of type `Out`.
///
/// examples:
/// - `Texture<float4, float2>`: 2D RGBA texture:
/// - `Texture<float , float3>`: 3D (voxel) R (1 color channel) texture
/// - `Texture<float3, CubeDir>`: a RGB cubemap texture
pub struct Texture<Out: TexSampleType, In: TexCoordType = float2> {
    any: Any,
    phantom: PhantomData<(In, Out)>,
}

/// a sampler binding. Contains parametrization for sampling from textures
#[derive(Copy, Clone)]
pub struct Sampler {
    pub(crate) any: Any,
}

impl Sampler {
    /// the [`Stage`] of the sampler binding
    pub fn stage(&self) -> Stage { Stage::Uniform }
}

impl Sampler {
    pub(crate) fn new() -> Self {
        Self {
            any: Any::global_interface(Self::ty(), Some("sampler".to_string())),
        }
    }

    /// type erased sampler value
    pub fn any(&self) -> Any { self.any }

    /// runtime type struct of the sampler
    pub fn ty() -> shame_graph::Ty { shame_graph::Ty::new(shame_graph::TyKind::Opaque(Self::opaque_ty())) }

    /// opaque type struct of the sampler
    pub(crate) fn opaque_ty() -> OpaqueTy { OpaqueTy::Sampler }

    /// take a `Out` sample of `tex` at `tex_coords`
    pub fn sample<In, Out>(&self, tex: &Texture<Out, In>, tex_coords: In) -> Ten<Out::S, Out::D>
    where
        In: TexCoordType,
        Out: TexSampleType,
    {
        tex.sample(*self, tex_coords)
    }
}

impl<Out: TexSampleType, In: TexCoordType> Texture<Out, In> {
    pub(crate) fn new() -> Self {
        Self {
            any: Any::global_interface(Self::ty(), Some("texture".to_string())),
            phantom: PhantomData,
        }
    }

    /// type erased texture value
    pub fn any(&self) -> Any { self.any }

    /// runtime type struct of the texture
    pub fn ty() -> shame_graph::Ty { shame_graph::Ty::new(shame_graph::TyKind::Opaque(Self::opaque_ty())).as_const() }

    /// opaque type struct of the texture
    pub(crate) fn opaque_ty() -> OpaqueTy { OpaqueTy::Texture(TexDtypeDimensionality(Out::D::DTYPE, In::DIM)) }

    /// take a `Out` sample of `self` at `tex_coords` using `sampler`
    pub fn sample(&self, sampler: Sampler, tex_coords: In) -> Ten<Out::S, Out::D> {
        let sampler_any = sampler.any;

        let kind = TexDtypeDimensionality(Out::D::DTYPE, In::DIM);
        let tcsampler = Any::texture_combined_sampler(kind, self.any, sampler_any);

        let (tex_coords_any, tex_coords_stage) = tex_coords.tex_coord_any();
        let sample = tcsampler.sample(tex_coords_any, None);

        use shame_graph::Shape::*;
        //apply output shape by using the tensor constructor if necessary
        let channels = match Out::S::SHAPE {
            Scalar | Vec(2) | Vec(3) => {
                let dst_tensor = shame_graph::Tensor::new(Out::S::SHAPE, Out::D::DTYPE);
                Any::new_tensor(dst_tensor, &[sample])
            }
            Vec(4) => sample, //no need for conversion
            s => panic!("invalid shape for texture channels: {}", s),
        };

        channels.downcast(narrow_stages_or_push_error([sampler.stage(), tex_coords_stage]))
    }
}
