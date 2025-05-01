use crate::frontend::any::Any;
use crate::{common::floating_point::f16, ir};

/// (no documentation yet)
#[diagnostic::on_unimplemented(message = "`{Self}` is not a shader scalar type")]
pub trait ScalarType: Copy {
    #[doc(hidden)] // runtime api
    const SCALAR_TYPE: ir::ScalarType;
    /// the `ScalarType` of the result vector of `t.lerp(a, b)` where `a` and `b` are `vec<Self, _>`
    type LerpOutput: ScalarType;
}

#[rustfmt::skip] impl ScalarType for f16 {const SCALAR_TYPE: ir::ScalarType = ir::ScalarType::F16; type LerpOutput = f16;}
#[rustfmt::skip] impl ScalarType for f32 {const SCALAR_TYPE: ir::ScalarType = ir::ScalarType::F32; type LerpOutput = f32;}
#[rustfmt::skip] impl ScalarType for f64 {const SCALAR_TYPE: ir::ScalarType = ir::ScalarType::F64; type LerpOutput = f64;}
#[rustfmt::skip] impl ScalarType for u32 {const SCALAR_TYPE: ir::ScalarType = ir::ScalarType::U32; type LerpOutput = f32;}
#[rustfmt::skip] impl ScalarType for i32 {const SCALAR_TYPE: ir::ScalarType = ir::ScalarType::I32; type LerpOutput = f32;}
#[rustfmt::skip] impl ScalarType for bool {const SCALAR_TYPE: ir::ScalarType = ir::ScalarType::Bool;type LerpOutput = f32;}

#[allow(clippy::unnecessary_cast)]
#[track_caller]
pub(crate) fn dtype_as_scalar_from_f64<T: ScalarType>(num: f64) -> Any {
    use ir::ScalarType::*;
    match T::SCALAR_TYPE {
        F16 => Any::new_scalar(ir::ScalarConstant::F16(f16::from(num))),
        F32 => Any::new_scalar(ir::ScalarConstant::F32(num as f32)),
        F64 => Any::new_scalar(ir::ScalarConstant::F64(num as f64)),
        I32 => Any::new_scalar(ir::ScalarConstant::I32(num as i32)),
        U32 => Any::new_scalar(ir::ScalarConstant::U32(num as u32)),
        Bool => Any::new_scalar(ir::ScalarConstant::Bool(num != 0.0f64)),
    }
}

/// (no documentation yet)
pub trait ScalarTypeNumber: ScalarType {}
impl ScalarTypeNumber for f16 {}
impl ScalarTypeNumber for f32 {}
impl ScalarTypeNumber for f64 {}
impl ScalarTypeNumber for u32 {}
impl ScalarTypeNumber for i32 {}

/// (no documentation yet)
pub trait ScalarTypeSigned: ScalarType {}
impl ScalarTypeSigned for f16 {}
impl ScalarTypeSigned for f32 {}
impl ScalarTypeSigned for f64 {}
impl ScalarTypeSigned for i32 {}

/// a signed or unsigned integer [`ScalarType`]
pub trait ScalarTypeInteger: ScalarTypeNumber {
    #[doc(hidden)] // runtime api
    const SCALAR_TYPE_INTEGER: ir::ScalarTypeInteger;
}
impl ScalarTypeInteger for u32 {
    const SCALAR_TYPE_INTEGER: ir::ScalarTypeInteger = ir::ScalarTypeInteger::U32;
}
impl ScalarTypeInteger for i32 {
    const SCALAR_TYPE_INTEGER: ir::ScalarTypeInteger = ir::ScalarTypeInteger::I32;
}

/// Floating point scalar types
///
/// implemented by [`f32`], [`f16`] and [`f64`]
///
/// > note: [`f16`] and [`f64`] may not supported
/// > by some target languages / configurations.
pub trait ScalarTypeFp: ScalarTypeNumber + ScalarTypeSigned {
    #[doc(hidden)] // runtime api
    const SCALAR_TYPE_FP: ir::ScalarTypeFp;
}
impl ScalarTypeFp for f16 {
    const SCALAR_TYPE_FP: ir::ScalarTypeFp = ir::ScalarTypeFp::F16;
}
impl ScalarTypeFp for f32 {
    const SCALAR_TYPE_FP: ir::ScalarTypeFp = ir::ScalarTypeFp::F32;
}
impl ScalarTypeFp for f64 {
    const SCALAR_TYPE_FP: ir::ScalarTypeFp = ir::ScalarTypeFp::F64;
}

/// a [`ScalarType`]  that is 32 bits in size
pub trait ScalarType32Bit: ScalarType {}
impl ScalarType32Bit for f32 {}
impl ScalarType32Bit for i32 {}
impl ScalarType32Bit for u32 {}
