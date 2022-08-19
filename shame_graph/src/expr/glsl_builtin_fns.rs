use super::*;
use crate::{error::Error, glsl_generic_function_decls};
use glsl_generics::GlslGeneric::*;

pub fn try_deduce_builtin_fn(kind: &super::BuiltinFn, args: &[Ty]) -> Result<Ty, Error> {
    use super::BuiltinFn::*;

    let apply = |args: &[Ty], decls: &[GlslGenericFunctionDecl]| {
        decls.iter().find_map(|decl| decl.deduce_return_type(args))
    };

    let deduced_ty = match kind {
        Dfdx => apply(
            args,
            &glsl_generic_function_decls! {
                genType dFdx(genType p);
            },
        ),
        Dfdy => apply(
            args,
            &glsl_generic_function_decls! {
                genType dFdy(genType p);
            },
        ),
        DfdxCoarse => apply(
            args,
            &glsl_generic_function_decls! {
                genType dFdxCoarse(genType p);
            },
        ),
        DfdyCoarse => apply(
            args,
            &glsl_generic_function_decls! {
                genType dFdyCoarse(genType p);
            },
        ),
        DfdxFine => apply(
            args,
            &glsl_generic_function_decls! {
                genType dFdxFine(genType p);
            },
        ),
        DfdyFine => apply(
            args,
            &glsl_generic_function_decls! {
                genType dFdyFine(genType p);
            },
        ),
        Round => apply(
            args,
            &glsl_generic_function_decls! {
                genType  round(genType  x);
                genDType round(genDType x);
            },
        ),
        Floor => apply(
            args,
            &glsl_generic_function_decls! {
                genType  floor(genType  x);
                genDType floor(genDType x);
            },
        ),
        Ceil => apply(
            args,
            &glsl_generic_function_decls! {
                genType  ceil(genType x);
                genDType ceil(genDType x);
            },
        ),
        Equal => apply(
            args,
            &glsl_generic_function_decls! {
                bvec equal( vec x,  vec y);
                bvec equal(ivec x, ivec y);
                bvec equal(uvec x, uvec y);
            },
        ),
        NotEqual => apply(
            args,
            &glsl_generic_function_decls! {
                bvec notEqual( vec x,  vec y);
                bvec notEqual(ivec x, ivec y);
                bvec notEqual(uvec x, uvec y);
            },
        ),
        LessThan => apply(
            args,
            &glsl_generic_function_decls! {
                bvec lessThan( vec x,  vec y);
                bvec lessThan(ivec x, ivec y);
                bvec lessThan(uvec x, uvec y);
            },
        ),
        LessThanEqual => apply(
            args,
            &glsl_generic_function_decls! {
                bvec lessThanEqual( vec x,  vec y);
                bvec lessThanEqual(ivec x, ivec y);
                bvec lessThanEqual(uvec x, uvec y);
            },
        ),
        GreaterThan => apply(
            args,
            &glsl_generic_function_decls! {
                bvec greaterThan( vec x,  vec y);
                bvec greaterThan(ivec x, ivec y);
                bvec greaterThan(uvec x, uvec y);
            },
        ),
        GreaterThanEqual => apply(
            args,
            &glsl_generic_function_decls! {
                bvec greaterThanEqual( vec x,  vec y);
                bvec greaterThanEqual(ivec x, ivec y);
                bvec greaterThanEqual(uvec x, uvec y);
            },
        ),
        All => apply(
            args,
            &glsl_generic_function_decls! {
                bool all(bvec x);
            },
        ),
        Any => apply(
            args,
            &glsl_generic_function_decls! {
                bool all(bvec x);
            },
        ),
        Not => apply(
            args,
            &glsl_generic_function_decls! {
                bvec not(bvec x);
            },
        ),
        Atan => apply(
            args,
            &glsl_generic_function_decls! {
                genType atan(genType y, genType x);
                genType atan(genType y_over_x);
            },
        ),
        Sign => apply(
            args,
            &glsl_generic_function_decls! {
                genType sign(genType x);
                genIType sign(genIType x);
                genDType sign(genDType x);
            },
        ),
        Pow => apply(
            args,
            &glsl_generic_function_decls! {
                genType pow(genType x, genType y);
            },
        ),
        Sqrt => apply(
            args,
            &glsl_generic_function_decls! {
                genType sqrt(genType x);
                genDType sqrt(genDType x);
            },
        ),
        Fract => apply(
            args,
            &glsl_generic_function_decls! {
                genType fract(genType x);
                genDType fract(genDType x);
            },
        ),
        Length => apply(
            args,
            &glsl_generic_function_decls! {
                float length(genType x);
                double length(genDType x);
            },
        ),
        Sin => apply(
            args,
            &glsl_generic_function_decls! {
                genType sin(genType angle);
            },
        ),
        Cos => apply(
            args,
            &glsl_generic_function_decls! {
                genType sin(genType angle);
            },
        ),
        Dot => apply(
            args,
            &glsl_generic_function_decls! {
                float dot(genType x,  genType y);
                double dot(genDType x,  genDType y);
            },
        ),
        Cross => apply(
            args,
            &glsl_generic_function_decls! {
                vec3 cross(vec3 x, vec3 y);
                dvec3 cross(dvec3 x, dvec3 y);
            },
        ),
        Min => apply(
            args,
            &glsl_generic_function_decls! {
                genType min(genType x, genType y);
                genType min(genType x, float y);
                genDType min(genDType x, genDType y);
                genDType min(genDType x, double y);
                genIType min(genIType x, genIType y);
                genIType min(genIType x, int y);
                genUType min(genUType x, genUType y);
                genUType min(genUType x, uint y);
            },
        ),
        Max => apply(
            args,
            &glsl_generic_function_decls! {
                genType max(genType x, genType y);
                genType max(genType x, float y);
                genDType max(genDType x, genDType y);
                genDType max(genDType x, double y);
                genIType max(genIType x, genIType y);
                genIType max(genIType x, int y);
                genUType max(genUType x, genUType y);
                genUType max(genUType x, uint y);
            },
        ),
        Clamp => apply(
            args,
            &glsl_generic_function_decls! {
                genType clamp(genType x, genType minVal, genType maxVal);
                genType clamp(genType x, float minVal, float maxVal);
                genDType clamp(genDType x, genDType minVal, genDType maxVal);
                genDType clamp(genDType x, double minVal, double maxVal);
                genIType clamp(genIType x, genIType minVal, genIType maxVal);
                genIType clamp(genIType x, int minVal, int maxVal);
                genUType clamp(genUType x, genUType minVal, genUType maxVal);
                genUType clamp(genUType x, uint minVal, uint maxVal);
            },
        ),
        Smoothstep => apply(
            args,
            &glsl_generic_function_decls! {
                genType smoothstep(genType edge0, genType edge1, genType x);
                genType smoothstep(float edge0, float edge1, genType x);
                genDType smoothstep(genDType edge0, genDType edge1, genDType x);
                genDType smoothstep(double edge0, double edge1, genDType x);
            },
        ),
        Mix => apply(
            args,
            &glsl_generic_function_decls! {
                genType mix(genType x, genType y, genType a);
                genType mix(genType x, genType y, float a);
                genDType mix(genDType x, genDType y, genDType a);
                genDType mix(genDType x, genDType y, double a);
                genType mix(genType x, genType y, genBType a);
                genDType mix(genDType x, genDType y, genBType a);
                genIType mix(genIType x, genIType y, genBType a);
                genUType mix(genUType x, genUType y, genBType a);
                genBType mix(genBType x, genBType y, genBType a);
            },
        ),
        Abs => apply(
            args,
            &glsl_generic_function_decls! {
                genType abs(genType x);
                genIType abs(genIType x);
                genDType abs(genDType x);
            },
        ),
        Normalize => apply(
            args,
            &glsl_generic_function_decls! {
                genType normalize(genType v);
                genDType normalize(genDType v);
            },
        ),
        Texture => apply(
            args,
            &glsl_generic_function_decls! {
                gvec4 texture(gsampler1D sampler, float P, [float bias]);
                gvec4 texture(gsampler2D sampler, vec2 P, [float bias]);
                gvec4 texture(gsampler3D sampler, vec3 P, [float bias]);
                gvec4 texture(gsamplerCube sampler, vec3 P, [float bias]);
                float texture(sampler1DShadow sampler, vec3 P, [float bias]);
                float texture(sampler2DShadow sampler, vec3 P, [float bias]);
                float texture(samplerCubeShadow sampler, vec4 P, [float bias]);
                gvec4 texture(gsampler1DArray sampler, vec2 P, [float bias]);
                gvec4 texture(gsampler2DArray sampler, vec3 P, [float bias]);
                gvec4 texture(gsamplerCubeArray sampler, vec4 P, [float bias]);
                float texture(sampler1DArrayShadow sampler, vec3 P, [float bias]);
                float texture(sampler2DArrayShadow sampler, vec4 P, [float bias]);
                gvec4 texture(gsampler2DRect sampler, vec2 P);
                float texture(sampler2DRectShadow sampler, vec3 P);
                float texture(samplerCubeArrayShadow sampler, vec4 P, float compare);
            },
        ),
    };

    deduced_ty.ok_or_else(|| invalid_arguments(kind, args))
}
