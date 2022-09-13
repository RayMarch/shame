//! trait for datatype of a tensor's components such as `f32`, `bool`, `i32`...,
use super::*;
use shame_graph::Any;

/// datatype of a tensor's components
///
/// e.g.
/// - `float, float2, float3, float4` are of [`DType`] `f32`
/// - `double, double2, double3, double4` are of [`DType`] `f64`
/// - `int, int2, int3, int4` are of [`DType`] `i32`
/// - `uint, uint2, uint3, uint4` are of [`DType`] `u32`
/// - `boolean, bool2, bool3, bool4` are of [`DType`] `bool`
pub trait DType: Copy + 'static {
    /// enum representation of `Self`
    const DTYPE: shame_graph::DType;

    /// turn `Self` into a shader literal (type erased)
    fn new_literal_any(&self) -> shame_graph::Any;

    /// turn `Self` into a shader literal
    fn new_literal(&self) -> Ten<scal, Self> { Ten::from_downcast(self.new_literal_any(), Stage::Uniform) }

    /// convert a f32 to this type
    fn from_f32(val: f32) -> Self;
}

macro_rules! impl_dtypes {
    ($(($fXX: ident, $dtype_enum: expr) -> as literal: $literal: ident; from f32: $from_f32: expr;)*) => {$(
        impl DType for $fXX {

            const DTYPE: shame_graph::DType = $dtype_enum;

            fn new_literal_any(&self) -> shame_graph::Any {
                shame_graph::Any::$literal((*self as $fXX) as _)
            }

            fn from_f32(val: f32) -> Self {
                $from_f32(val)
            }
        }
    )*};
}

impl_dtypes! {
    (f32 , shame_graph::DType::F32 ) -> as literal: float;  from f32: |x: f32| x as Self;
    (f64 , shame_graph::DType::F64 ) -> as literal: double; from f32: |x: f32| x as Self;
    (i32 , shame_graph::DType::I32 ) -> as literal: int;    from f32: |x: f32| x as Self;
    (u32 , shame_graph::DType::U32 ) -> as literal: uint;   from f32: |x: f32| x as Self;
    (bool, shame_graph::DType::Bool) -> as literal: bool;   from f32: |x: f32| x != 0.0;
}

macro_rules! rust_primitive_types_as_ten {
    ($(($fXX: ty) as ten -> $ten_ty: ident;)*) => {$(

        impl AsTen for $fXX {
            type S = scal;
            type D = <$ten_ty as AsTen>::D;

            fn as_ten(&self) -> Ten<Self::S, Self::D> {
                Self::D::new_literal(&(*self as <$ten_ty as AsTen>::D))
            }
        }

        impl IntoRec for $fXX {
            type Rec = Ten<scal, <$ten_ty as AsTen>::D>;
            fn rec(self) -> Self::Rec {
                <$ten_ty as AsTen>::D::new_literal(&(self as <$ten_ty as AsTen>::D))
            }

            fn into_any(self) -> Any {self.rec().into_any()}
            fn stage(&self) -> Stage {Stage::Uniform}
        }
    )*};
}

rust_primitive_types_as_ten! {
    (f32)  as ten -> float;
    (i32)  as ten -> int;
    (bool) as ten -> boolean;
    (u32)  as ten -> uint;
}

/// implemented for `f64` and `f32`, ensures `IsDTypeNumber`
pub trait IsDTypeFloatingPoint: IsDTypeNumber {}
impl IsDTypeFloatingPoint for f32 {}
impl IsDTypeFloatingPoint for f64 {}

/// implemented for `i32`, `u32`, `bool`,
pub trait IsDtypeNonFloatingPoint: DType {}
impl IsDtypeNonFloatingPoint for i32 {}
impl IsDtypeNonFloatingPoint for u32 {}
impl IsDtypeNonFloatingPoint for bool {}

/// implemented for `i32` and `u32`, ensures `IsDTypeNumber`
pub trait IsDTypeInteger: IsDTypeNumber {}
impl IsDTypeInteger for i32 {}
impl IsDTypeInteger for u32 {}

/// implemented for `f32`, `f64`, `i32`, `u32`. ensures 'DType'
pub trait IsDTypeNumber: DType {}
impl IsDTypeNumber for f32 {}
impl IsDTypeNumber for f64 {}
impl IsDTypeNumber for i32 {}
impl IsDTypeNumber for u32 {}
