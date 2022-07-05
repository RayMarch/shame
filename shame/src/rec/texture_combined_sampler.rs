//! type representing glsl's Sampler2D-like types, which combined samplers and
//! textures inside a single object
use std::marker::PhantomData;

use crate::TexSampleType;

use super::*;
use shame_graph::{Any, TexDimensionality, TexDtypeDimensionality};

/// a type that describes where a texture is sampled. Which type depends on the
/// kind of texture.
/// 
/// - float : 1d texture
/// - float2: 2d texture
/// - float3: 3d (voxel) texture
/// - CubeDir: cubemap texture
pub trait TexCoordType { 
    /// describes the sampling domain of the texture
    const DIM: TexDimensionality; 
    /// type erases the texture coordinates
    fn tex_coord_any(&self) -> (Any, Stage); 
}
impl TexCoordType for float   { const DIM: TexDimensionality = TexDimensionality::Tex1d;      fn tex_coord_any(&self) -> (Any, Stage) {(self.any, self.stage)} }
impl TexCoordType for float2  { const DIM: TexDimensionality = TexDimensionality::Tex2d;      fn tex_coord_any(&self) -> (Any, Stage) {(self.any, self.stage)} }
impl TexCoordType for float3  { const DIM: TexDimensionality = TexDimensionality::Tex3d;      fn tex_coord_any(&self) -> (Any, Stage) {(self.any, self.stage)} }
impl TexCoordType for CubeDir { const DIM: TexDimensionality = TexDimensionality::TexCubeMap; fn tex_coord_any(&self) -> (Any, Stage) {(self.0.any, self.0.stage)} }

/// cube direction 
/// using this as the `TexCoord` generic argument of `Texture` means the texture is considered a cube map.
/// 
/// When specifying the inner `float3` value for sampling, it is interpreted as a direction which is projected onto the surface of unit cube `[-1..1]^3`.
/// TODO: find a proper source to link here which discribes how cubemap sampling works
#[derive(Clone, Copy)]
pub struct CubeDir(pub float3);

/// alias for texture/sampler pairs that return a `float4` when sampled
pub type CombineSamplerRGBA<In = float2> = CombineSampler<float4, In>;
/// alias for texture/sampler pairs that return a `float3` when sampled
pub type CombineSamplerRGB <In = float2> = CombineSampler<float3, In>;
/// alias for texture/sampler pairs that return a `float2` when sampled
pub type CombineSamplerRG  <In = float2> = CombineSampler<float2, In>;
/// alias for texture/sampler pairs that return a `float` when sampled
pub type CombineSamplerR   <In = float2> = CombineSampler<float , In>;

/// a sampler and texture together. 
/// glsl calls this a "texture combined sampler".
pub struct CombineSampler<Out: TexSampleType, In: TexCoordType = float2> {
    any: Any,
    _phantom: PhantomData<(Out, In)>,
}


impl<Out: TexSampleType, In: TexCoordType> CombineSampler<Out, In> {
    pub(crate) fn new() -> Self {
        Self {
            any: shame_graph::Any::global_interface(Self::ty(), Some("tc_sampler".to_string())),
            _phantom: PhantomData,
        }
    }

    pub(crate) fn stage(&self) -> Stage {
        Stage::Uniform
    }

    /// samples at the coordinates `coords` in their respective stage.
    /// 
    /// if `coords` is a vertex-stage value, the resulting sample is also a 
    /// vertex-stage value.
    /// 
    /// if `coords` is a fragment-stage value, the resulting sample is also a 
    /// fragment-stage value.
    /// 
    /// if `coords` is a uniform-stage value, the resulting sample is also a 
    /// uniform-stage value. The sample will be taken in both vertex and 
    /// fragment stage (keep in mind that unused samples will most likely just 
    /// be optimized away by your shader compiler though.)
    pub fn sample(&self, tex_coords: In) -> Ten<Out::S, Out::D> { //TODO: remove duplication with the other sample functions
        let tcsampler = self.any;

        let (tex_coords_any, tex_coords_stage) = tex_coords.tex_coord_any();
        let sample = tcsampler.sample(tex_coords_any, None);
        
        use shame_graph::Shape::*;
        //apply output shape by using the tensor constructor if necessary
        let channels = match Out::S::SHAPE {
            Scalar |
            Vec(2) |
            Vec(3) => {
                let dst_tensor = shame_graph::Tensor::new(Out::S::SHAPE, Out::D::DTYPE);
                Any::new_tensor(dst_tensor, &[sample])
            },
            Vec(4) => sample, //no need for conversion
            s => panic!("invalid shape for texture channels: {}", s)
        };

        channels.downcast(narrow_stages_or_push_error([self.stage(), tex_coords_stage]))
    }

    /// type erased recording type of this [`CombineSampler`]
    pub fn any(&self) -> Any {
        self.any
    }

    /// runtime type struct of `Self`
    pub fn ty() -> shame_graph::Ty {
        shame_graph::Ty::new(shame_graph::TyKind::Opaque(Self::opaque_ty())).as_const()
    }

    pub(crate) fn opaque_ty() -> shame_graph::OpaqueTy {
        let dtype_dim = TexDtypeDimensionality(shame_graph::DType::F32, In::DIM);
        shame_graph::OpaqueTy::TextureCombinedSampler(dtype_dim)
    }
}