//! convenience aliases for tensor types, e.g. [`float4`] for `Ten<vec4; f32>`
use super::*;

macro_rules! impl_rec_aliases {
    ($($shape: ident: ($floatN: ident, $doubleN: ident, $intN: ident, $uintN: ident, $boolN: ident);)*) => {$(
        #[allow(non_camel_case_types, missing_docs)] pub type $floatN  = Ten<$shape, f32>;
        #[allow(non_camel_case_types, missing_docs)] pub type $doubleN = Ten<$shape, f64>;
        #[allow(non_camel_case_types, missing_docs)] pub type $intN    = Ten<$shape, i32>;
        #[allow(non_camel_case_types, missing_docs)] pub type $uintN   = Ten<$shape, u32>;
        #[allow(non_camel_case_types, missing_docs)] pub type $boolN   = Ten<$shape,bool>;
    )*};
}

impl_rec_aliases!{
    scal  : (float   , double   , int   , uint   , boolean);
    vec2  : (float2  , double2  , int2  , uint2  , bool2  );
    vec3  : (float3  , double3  , int3  , uint3  , bool3  );
    vec4  : (float4  , double4  , int4  , uint4  , bool4  );
    mat2  : (float2x2, double2x2, int2x2, uint2x2, bool2x2);
    mat2x3: (float2x3, double2x3, int2x3, uint2x3, bool2x3);
    mat2x4: (float2x4, double2x4, int2x4, uint2x4, bool2x4);
    mat3x2: (float3x2, double3x2, int3x2, uint3x2, bool3x2);
    mat3  : (float3x3, double3x3, int3x3, uint3x3, bool3x3);
    mat3x4: (float3x4, double3x4, int3x4, uint3x4, bool3x4);
    mat4x2: (float4x2, double4x2, int4x2, uint4x2, bool4x2);
    mat4x3: (float4x3, double4x3, int4x3, uint4x3, bool4x3);
    mat4  : (float4x4, double4x4, int4x4, uint4x4, bool4x4);
}

macro_rules! trait_alias {
    ($alias: ident = $original: path) => {
        /// types that can be converted to this tensor
        pub trait $alias: $original {}
        impl<T: $original> $alias for T {}
    };
}

trait_alias!(AsFloat4 = AsTen<S=vec4, D=f32>);
trait_alias!(AsFloat3 = AsTen<S=vec3, D=f32>);
trait_alias!(AsFloat2 = AsTen<S=vec2, D=f32>);
trait_alias!(AsFloat  = AsTen<S=scal, D=f32>);

trait_alias!(AsDouble4 = AsTen<S=vec4, D=f64>);
trait_alias!(AsDouble3 = AsTen<S=vec3, D=f64>);
trait_alias!(AsDouble2 = AsTen<S=vec2, D=f64>);
trait_alias!(AsDouble  = AsTen<S=scal, D=f64>);

trait_alias!(AsInt4 = AsTen<S=vec4, D=i32>);
trait_alias!(AsInt3 = AsTen<S=vec3, D=i32>);
trait_alias!(AsInt2 = AsTen<S=vec2, D=i32>);
trait_alias!(AsInt  = AsTen<S=scal, D=i32>);

trait_alias!(AsUint4 = AsTen<S=vec4, D=u32>);
trait_alias!(AsUint3 = AsTen<S=vec3, D=u32>);
trait_alias!(AsUint2 = AsTen<S=vec2, D=u32>);
trait_alias!(AsUint  = AsTen<S=scal, D=u32>);

trait_alias!(AsBool4   = AsTen<S=vec4, D=bool>);
trait_alias!(AsBool3   = AsTen<S=vec3, D=bool>);
trait_alias!(AsBool2   = AsTen<S=vec2, D=bool>);
trait_alias!(AsBoolean = AsTen<S=scal, D=bool>);