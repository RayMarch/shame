
use super::{DType, TexDimensionality, Shape, Ty};
use DType::*;
use crate::expr::{OpaqueTy, TexDtypeDimensionality, ShadowSamplerKind, TyKind};

//glsl generics syntax used in Chapter 8 of https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.pdf
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub enum GlslGeneric {
    //Chapter 8. Built-In Functions
    int,
    uint,
    float,
    double,
    bool,
    vec2,
    vec3,
    vec4,
    dvec2,
    dvec3,
    dvec4,

    genType, //this is no longer used in glsl spec 4.6, they use genFType instead.
    genFType,
    genDType,
    genBType,
    genUType,
    genIType,

    mat,
    dmat,

    //8.7. Vector Relational Functions
    bvec,
    ivec,
    uvec,
    vec,

    //8.9. Texture Functions
    gvec4,
    gsampler1D,
    gsampler2D,
    gsampler3D,
    gsamplerCube,
    gsampler2DRect,
    gsampler1DArray,
    gsampler2DArray,
    gsamplerCubeArray,
    gsamplerBuffer,
    gsampler2DMS,
    gsampler2DMSArray,

    sampler1DShadow,
    sampler2DShadow,
    samplerCubeShadow,
    sampler2DRectShadow,
    sampler1DArrayShadow,
    sampler2DArrayShadow,
    samplerCubeArrayShadow,
}

#[derive(Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GlslGenericInstantiation {
    specific,
    gen(Shape), //the "gen" in genFType ...
    vecN(u8), //the components of bvec, ivec, uvec, vec,
    matMxN(u8, u8), //the dimensions of mat, dmat,
    g(DType), //the "g" in gvec4, gsampler... where g in ["i", "u", ""]
}

impl HighLevelGlslGeneric {

    pub fn new(glsl: GlslGeneric) -> Self {
        use GlslGeneric::*;
        let shadow = |k: ShadowSamplerKind| {
            Ty::new(TyKind::Opaque(OpaqueTy::ShadowSampler(k)))
        };

        match glsl {
            int    => Specific(Ty::int()),
            uint   => Specific(Ty::uint()),
            float  => Specific(Ty::float()),
            double => Specific(Ty::double()),
            bool   => Specific(Ty::bool()),
            vec2   => Specific(Ty::vec2()),
            vec3   => Specific(Ty::vec3()),
            vec4   => Specific(Ty::vec4()),
            dvec2  => Specific(Ty::dvec2()),
            dvec3  => Specific(Ty::dvec3()),
            dvec4  => Specific(Ty::dvec4()),

            //shared "gen" == shape when used in decls
            genType  => GenXType(F32),
            genFType => GenXType(F32),
            genDType => GenXType(F64),
            genBType => GenXType(Bool),
            genUType => GenXType(U32),
            genIType => GenXType(I32),

            mat  => MatMxN(F32),
            dmat => MatMxN(F64),

            //shared shape when used in decls
            bvec => VecN(Bool),
            ivec => VecN(I32),
            uvec => VecN(U32),
            vec  => VecNF32F64, //matches both vecN and dvecN where N in 0..4

            //shared "g" == DType when used in decls
            gvec4 => GVec4F32I32U32,
            gsampler1D        => GSampler(TexDimensionality::Tex1d),
            gsampler2D        => GSampler(TexDimensionality::Tex2d),
            gsampler3D        => GSampler(TexDimensionality::Tex3d),
            gsamplerCube      => GSampler(TexDimensionality::TexCubeMap),
            gsampler2DRect    => GSampler(TexDimensionality::TexRectangle),
            gsampler1DArray   => GSampler(TexDimensionality::Tex1dArray),
            gsampler2DArray   => GSampler(TexDimensionality::Tex2dArray),
            gsamplerCubeArray => GSampler(TexDimensionality::TexCubeMapArray),
            gsamplerBuffer    => GSampler(TexDimensionality::TexBuffer),
            gsampler2DMS      => GSampler(TexDimensionality::Tex2dMultisample),
            gsampler2DMSArray => GSampler(TexDimensionality::Tex2dMultisampleArray),

            sampler1DShadow        => Specific(shadow(ShadowSamplerKind::Tex1d)),
            sampler2DShadow        => Specific(shadow(ShadowSamplerKind::Tex2d)),
            samplerCubeShadow      => Specific(shadow(ShadowSamplerKind::TexCubeMap)),
            sampler2DRectShadow    => Specific(shadow(ShadowSamplerKind::TexRectangle)),
            sampler1DArrayShadow   => Specific(shadow(ShadowSamplerKind::Tex1dArray)),
            sampler2DArrayShadow   => Specific(shadow(ShadowSamplerKind::Tex1dArray)),
            samplerCubeArrayShadow => Specific(shadow(ShadowSamplerKind::TexCubeMapArray)),
        }
    }

    pub fn try_instantiate(&self, ty: &Ty) -> Option<GlslGenericInstantiation> {
        match (self, &ty.kind) {

            (Specific(x), _) => (ty.eq_ignore_access(x))
                .then(|| GlslGenericInstantiation::specific),

            (GenXType(x), TyKind::Tensor(ten)) => //GLSL 4.60 spec: "[...] where the input arguments (and corresponding output) can be float, vec2, vec3, or vec4, genFType is used"
                (&ten.dtype == x && matches!(ten.shape, Shape::Vec(_) | Shape::Scalar))
                .then(|| GlslGenericInstantiation::gen(ten.shape)),

            (MatMxN(x), TyKind::Tensor(ten)) => match ten.shape {
                Shape::Mat(m, n) if &ten.dtype == x => Some(GlslGenericInstantiation::matMxN(m, n)),
                _ => None
            }

            (VecN(x), TyKind::Tensor(ten)) => match ten.shape {
                Shape::Vec(n) if &ten.dtype == x => Some(GlslGenericInstantiation::vecN(n)),
                _ => None,
            }

            (VecNF32F64, TyKind::Tensor(ten)) => match ten.shape {
                Shape::Vec(n) if [F32, F64].contains(&ten.dtype) => Some(GlslGenericInstantiation::vecN(n)),
                _ => None,
            }

            (GVec4F32I32U32, TyKind::Tensor(ten)) =>
                ([F32, I32, U32].contains(&ten.dtype) && matches!(ten.shape, Shape::Vec(4)))
                .then(|| GlslGenericInstantiation::g(ten.dtype)),

            (GSampler(x), TyKind::Opaque(OpaqueTy::TextureCombinedSampler(TexDtypeDimensionality(dtype, sampler_kind)))) =>
                (x == sampler_kind)
                .then(|| GlslGenericInstantiation::g(*dtype)),

            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum HighLevelGlslGeneric {
    Specific(Ty),
    GenXType(DType),//Vec or scalar
    MatMxN(DType),
    VecN(DType),  //Vec only (num components > 1)
    VecNF32F64,
    GVec4F32I32U32, //"gvec4" is either vec4, ivec4 or uvec4 (section 8.9. Texture Functions)
    GSampler(TexDimensionality),
}
use HighLevelGlslGeneric::*;
