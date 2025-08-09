#![allow(clippy::identity_op)]

use super::{
    reference::{AccessModeReadable, ReadWrite, ReadableRef},
    scalar_type::{ScalarTypeInteger, ScalarTypeNumber, ScalarTypeSigned},
    type_traits::NoBools,
    vec::ToVec,
    AsAny, GpuType, To, ToGpuType,
};
use crate::common::floating_point::f16;
use crate::frontend::rust_types::reference::Ref;
use crate::frontend::rust_types::vec::vec;
use crate::{
    frontend::{
        any::Any,
        rust_types::{len::*, scalar_type::ScalarType},
    },
    impl_ops, ir,
    mem::AddressSpace,
};
use std::ops::*;

// TODO(release) write operator tests for all combinations of scalar/vector, vextor/matrix, scalar/matrix
//      - every combination should exist (forward and backward, mat * vec vs vec * mat)
//      - create a "must not compile" test for those who are disallowed, so that this is exhaustive
//          otherwise this is predestined to have some cases that are not properly represented

impl_ops! {
<L: Len, N: ScalarTypeSigned> Neg !op: neg(x: vec<N, L>) -> vec<N, L>: {op(x.as_any()).into()};
}

impl_ops! {
<L: Len, N: ScalarTypeNumber > Add !op: add(l: vec<N, L>, r: vec<N, L>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
<L: Len, N: ScalarTypeNumber > Sub !op: sub(l: vec<N, L>, r: vec<N, L>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
<L: Len, N: ScalarTypeNumber > Div !op: div(l: vec<N, L>, r: vec<N, L>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
<L: Len, N: ScalarTypeInteger> Rem !op: rem(l: vec<N, L>, r: vec<N, L>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};

// vec vs scalar for + and - is disabled (commented out) because it can easily be fixed via .splat() but causes ambiguity in type inference in some cases, which forces the user to specify Len. e.g. vec2 + sm::vec::x()
//<L: Len2, N: ScalarTypeNumber > Add !op: add(l: vec<N, L>, r: vec<N, x1>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
//<L: Len2, N: ScalarTypeNumber > Sub !op: sub(l: vec<N, L>, r: vec<N, x1>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
<L: Len2, N: ScalarTypeNumber > Div !op: div(l: vec<N, L>, r: vec<N, x1>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
<L: Len2, N: ScalarTypeInteger> Rem !op: rem(l: vec<N, L>, r: vec<N, x1>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};

// to prevent overlapping impls, this is how i chose to implement "scalar * vec" and "vec * scalar".
// the latter version is usable for generic programming. the former not. (luckily multiplication is commutative)
<L: Len, N: ScalarTypeNumber> Mul !op: mul(l: vec<N, L>, r: vec<N, x1>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
<N: ScalarTypeNumber> Mul !op: mul(l: vec<N, x1>, r: vec<N, x2>) -> vec<N, x2>: {op(l.as_any(), r.as_any()).into()};
<N: ScalarTypeNumber> Mul !op: mul(l: vec<N, x1>, r: vec<N, x3>) -> vec<N, x3>: {op(l.as_any(), r.as_any()).into()};
<N: ScalarTypeNumber> Mul !op: mul(l: vec<N, x1>, r: vec<N, x4>) -> vec<N, x4>: {op(l.as_any(), r.as_any()).into()};
}

impl<L: Len, N: ScalarTypeNumber> vec<N, L> {
    /// component-wise multiplication, also known as hadamard product
    #[track_caller]
    pub fn mul_each(self, rhs: impl ToVec<L = L, T = N>) -> vec<N, L> { self.as_any().mul(rhs.to_any()).into() }
}

impl_ops! { + also implements lhs <-> rhs swapped
    // f64 impls are commented out as they cause `1.0` generic float literals to be turned into f64 instead of f32 sometimes. Users can still manually call f64x1::from
    <L: Len> Add !op: add(l: vec<f16, L>, r: f16) -> vec<f16, L>: {op(l.to_any(), r.to_any()).into()};
    <L: Len> Add !op: add(l: vec<f32, L>, r: f32) -> vec<f32, L>: {op(l.to_any(), r.to_any()).into()};
    //<L: Len> Add !op: add(l: vec<f64, L>, r: f64) -> vec<f64, L>: {op(l.to_any(), Any::new_scalar(ir::ScalarConstant::F64(r))).into()};
    <L: Len> Add !op: add(l: vec<i32, L>, r: i32) -> vec<i32, L>: {op(l.to_any(), r.to_any()).into()};
    <L: Len> Add !op: add(l: vec<u32, L>, r: u32) -> vec<u32, L>: {op(l.to_any(), r.to_any()).into()};

    <L: Len> Sub !op: sub(l: vec<f16, L>, r: f16) -> vec<f16, L>: {op(l.to_any(), r.to_any()).into()};
    <L: Len> Sub !op: sub(l: vec<f32, L>, r: f32) -> vec<f32, L>: {op(l.to_any(), r.to_any()).into()};
    //<L: Len> Sub !op: sub(l: vec<f64, L>, r: f64) -> vec<f64, L>: {op(l.to_any(), Any::new_scalar(ir::ScalarConstant::F64(r))).into()};
    <L: Len> Sub !op: sub(l: vec<i32, L>, r: i32) -> vec<i32, L>: {op(l.to_any(), r.to_any()).into()};
    <L: Len> Sub !op: sub(l: vec<u32, L>, r: u32) -> vec<u32, L>: {op(l.to_any(), r.to_any()).into()};

    <L: Len> Mul !op: mul(l: vec<f16, L>, r: f16) -> vec<f16, L>: {op(l.to_any(), r.to_any()).into()};
    <L: Len> Mul !op: mul(l: vec<f32, L>, r: f32) -> vec<f32, L>: {op(l.to_any(), r.to_any()).into()};
    //<L: Len> Mul !op: mul(l: vec<f64, L>, r: f64) -> vec<f64, L>: {op(l.to_any(), Any::new_scalar(ir::ScalarConstant::F64(r))).into()};
    <L: Len> Mul !op: mul(l: vec<i32, L>, r: i32) -> vec<i32, L>: {op(l.to_any(), r.to_any()).into()};
    <L: Len> Mul !op: mul(l: vec<u32, L>, r: u32) -> vec<u32, L>: {op(l.to_any(), r.to_any()).into()};

    <L: Len> Div !op: div(l: vec<f16, L>, r: f16) -> vec<f16, L>: {op(l.to_any(), r.to_any()).into()};
    <L: Len> Div !op: div(l: vec<f32, L>, r: f32) -> vec<f32, L>: {op(l.to_any(), r.to_any()).into()};
    //<L: Len> Div !op: div(l: vec<f64, L>, r: f64) -> vec<f64, L>: {op(l.to_any(), Any::new_scalar(ir::ScalarConstant::F64(r))).into()};
    <L: Len> Div !op: div(l: vec<i32, L>, r: i32) -> vec<i32, L>: {op(l.to_any(), r.to_any()).into()};
    <L: Len> Div !op: div(l: vec<u32, L>, r: u32) -> vec<u32, L>: {op(l.to_any(), r.to_any()).into()};

    <L: Len> Rem !op: rem(l: vec<i32, L>, r: i32) -> vec<i32, L>: {op(l.to_any(), r.to_any()).into()};
    <L: Len> Rem !op: rem(l: vec<u32, L>, r: u32) -> vec<u32, L>: {op(l.to_any(), r.to_any()).into()};
}

// Ref impls

// commented out all Ref *op*_assign implementations, since they are replaced by the
// more ergonomic `set_*op*` variants.
/*
impl_ops! {
    <L: Len, N: ScalarTypeNumber, AS: AddressSpace> AddAssign !op: add_assign(l: &mut Ref<vec<N, L>, AS, ReadWrite>, r: vec<N, L>): {op(&mut l.as_any(), r.as_any())};
    <L: Len, N: ScalarTypeNumber, AS: AddressSpace> SubAssign !op: sub_assign(l: &mut Ref<vec<N, L>, AS, ReadWrite>, r: vec<N, L>): {op(&mut l.as_any(), r.as_any())};
    <L: Len, N: ScalarTypeNumber, AS: AddressSpace> DivAssign !op: div_assign(l: &mut Ref<vec<N, L>, AS, ReadWrite>, r: vec<N, L>): {op(&mut l.as_any(), r.as_any())};

    // to prevent overlapping impls, this is how i chose to implement "scalar * vec" and "vec * scalar".
    // the latter version is usable for generic programming. the former not. (luckily multiplication is commutative)
    <L: Len, N: ScalarTypeNumber, AS: AddressSpace> MulAssign !op: mul_assign(l: &mut Ref<vec<N,  L>, AS, ReadWrite>, r: vec<N, x1>): {op(&mut l.as_any(), r.as_any())};

    // everything below here is implicit splat of rhs
    <        N: ScalarTypeNumber, AS: AddressSpace> MulAssign !op: mul_assign(l: &mut Ref<vec<N, x1>, AS, ReadWrite>, r: vec<N, x2>): {op(&mut l.as_any(), r.as_any())};
    <        N: ScalarTypeNumber, AS: AddressSpace> MulAssign !op: mul_assign(l: &mut Ref<vec<N, x1>, AS, ReadWrite>, r: vec<N, x3>): {op(&mut l.as_any(), r.as_any())};
    <        N: ScalarTypeNumber, AS: AddressSpace> MulAssign !op: mul_assign(l: &mut Ref<vec<N, x1>, AS, ReadWrite>, r: vec<N, x4>): {op(&mut l.as_any(), r.as_any())};

    <L: Len, AS: AddressSpace> AddAssign !op: add_assign(l: &mut Ref<vec<f16, L>, AS, ReadWrite>, r: f16): {op(&mut l.as_any(), r.to_any().splat(L::LEN, f16::SCALAR_TYPE))};
    <L: Len, AS: AddressSpace> AddAssign !op: add_assign(l: &mut Ref<vec<f32, L>, AS, ReadWrite>, r: f32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, f32::SCALAR_TYPE))};
    //<L: Len, AS: AddressSpace> AddAssign !op: add_assign(l: &mut Ref<vec<f64, L>, AS, ReadWrite>, r: f64): {op(&mut l.as_any(), Any::new_scalar(ir::ScalarConstant::F64(r)))};
    <L: Len, AS: AddressSpace> AddAssign !op: add_assign(l: &mut Ref<vec<i32, L>, AS, ReadWrite>, r: i32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, i32::SCALAR_TYPE))};
    <L: Len, AS: AddressSpace> AddAssign !op: add_assign(l: &mut Ref<vec<u32, L>, AS, ReadWrite>, r: u32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, u32::SCALAR_TYPE))};

    <L: Len, AS: AddressSpace> SubAssign !op: sub_assign(l: &mut Ref<vec<f16, L>, AS, ReadWrite>, r: f16): {op(&mut l.as_any(), r.to_any().splat(L::LEN, f16::SCALAR_TYPE))};
    <L: Len, AS: AddressSpace> SubAssign !op: sub_assign(l: &mut Ref<vec<f32, L>, AS, ReadWrite>, r: f32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, f32::SCALAR_TYPE))};
    //<L: Len, AS: AddressSpace> SubAssign !op: sub_assign(l: &mut Ref<vec<f64, L>, AS, ReadWrite>, r: f64): {op(&mut l.as_any(), Any::new_scalar(ir::ScalarConstant::F64(r)))};
    <L: Len, AS: AddressSpace> SubAssign !op: sub_assign(l: &mut Ref<vec<i32, L>, AS, ReadWrite>, r: i32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, i32::SCALAR_TYPE))};
    <L: Len, AS: AddressSpace> SubAssign !op: sub_assign(l: &mut Ref<vec<u32, L>, AS, ReadWrite>, r: u32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, u32::SCALAR_TYPE))};

    <L: Len, AS: AddressSpace> MulAssign !op: mul_assign(l: &mut Ref<vec<f16, L>, AS, ReadWrite>, r: f16): {op(&mut l.as_any(), r.to_any().splat(L::LEN, f16::SCALAR_TYPE))};
    <L: Len, AS: AddressSpace> MulAssign !op: mul_assign(l: &mut Ref<vec<f32, L>, AS, ReadWrite>, r: f32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, f32::SCALAR_TYPE))};
    //<L: Len, AS: AddressSpace> MulAssign !op: mul_assign(l: &mut Ref<vec<f64, L>, AS, ReadWrite>, r: f64): {op(&mut l.as_any(), Any::new_scalar(ir::ScalarConstant::F64(r)))};
    <L: Len, AS: AddressSpace> MulAssign !op: mul_assign(l: &mut Ref<vec<i32, L>, AS, ReadWrite>, r: i32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, i32::SCALAR_TYPE))};
    <L: Len, AS: AddressSpace> MulAssign !op: mul_assign(l: &mut Ref<vec<u32, L>, AS, ReadWrite>, r: u32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, u32::SCALAR_TYPE))};

    <L: Len, AS: AddressSpace> DivAssign !op: div_assign(l: &mut Ref<vec<f16, L>, AS, ReadWrite>, r: f16): {op(&mut l.as_any(), r.to_any().splat(L::LEN, f16::SCALAR_TYPE))};
    <L: Len, AS: AddressSpace> DivAssign !op: div_assign(l: &mut Ref<vec<f32, L>, AS, ReadWrite>, r: f32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, f32::SCALAR_TYPE))};
    //<L: Len, AS: AddressSpace> DivAssign !op: div_assign(l: &mut Ref<vec<f64, L>, AS, ReadWrite>, r: f64): {op(&mut l.as_any(), Any::new_scalar(ir::ScalarConstant::F64(r)))};
    <L: Len, AS: AddressSpace> DivAssign !op: div_assign(l: &mut Ref<vec<i32, L>, AS, ReadWrite>, r: i32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, i32::SCALAR_TYPE))};
    <L: Len, AS: AddressSpace> DivAssign !op: div_assign(l: &mut Ref<vec<u32, L>, AS, ReadWrite>, r: u32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, u32::SCALAR_TYPE))};

    <L: Len, AS: AddressSpace> RemAssign !op: rem_assign(l: &mut Ref<vec<i32, L>, AS, ReadWrite>, r: i32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, i32::SCALAR_TYPE))};
    <L: Len, AS: AddressSpace> RemAssign !op: rem_assign(l: &mut Ref<vec<u32, L>, AS, ReadWrite>, r: u32): {op(&mut l.as_any(), r.to_any().splat(L::LEN, u32::SCALAR_TYPE))};
}
*/

impl<T: ScalarTypeNumber, L: Len, AS: AddressSpace> Ref<vec<T, L>, AS, ReadWrite> {
    /// compound assignment using component-wise multiplication, also known as hadamard product
    ///
    /// equivalent to
    ///
    /// `self.set(self.get().mul_each(rhs))`
    #[track_caller]
    pub fn set_mul_each(&mut self, rhs: vec<T, L>) {
        let op = MulAssign::mul_assign;
        op(&mut self.as_any(), rhs.to_any())
    }
}

#[rustfmt::skip]
impl<T: ScalarType, L: Len> vec<T, L> {
    /// component wise "less than" operation (`<`)
    #[track_caller] pub fn lt(&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().lt(rhs.to_any()).into()}
    /// component wise "less than or equal" operation (`<=`)
    #[track_caller] pub fn le(&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().le(rhs.to_any()).into()}
    /// component wise "greater than" operation (`>`)
    #[track_caller] pub fn gt(&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().gt(rhs.to_any()).into()}
    /// component wise "greater than or equal" operation (`>=`)
    #[track_caller] pub fn ge(&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().ge(rhs.to_any()).into()}
    /// component wise equality comparison operation (`==`)
    #[track_caller] pub fn eq(&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().eq(rhs.to_any()).into()}
    /// component wise "not equal" comparison operation (`!=`)
    #[track_caller] pub fn ne(&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().ne(rhs.to_any()).into()}

    /// component wise "less than" operation (`<`)
    #[track_caller] pub fn less_than   (&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().lt(rhs.to_any()).into()}
    /// component wise "less than or equal" operation (`<=`)
    #[track_caller] pub fn less_eq     (&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().le(rhs.to_any()).into()}
    /// component wise "greater than" operation (`>`)
    #[track_caller] pub fn greater_than(&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().gt(rhs.to_any()).into()}
    /// component wise "greater than or equal" operation (`>=`)
    #[track_caller] pub fn greater_eq  (&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().ge(rhs.to_any()).into()}
    /// component wise equality comparison operation (`==`)
    #[track_caller] pub fn equals       (&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().eq(rhs.to_any()).into()}
    /// component wise "not equal" comparison operation (`!=`)
    #[track_caller] pub fn not_equal   (&self, rhs: impl To<vec<T, L>>) -> vec<bool, L> {self.as_any().ne(rhs.to_any()).into()}
}


// Bit::BitwiseComplement       =>  [Vec(n, t)]              if t.is_integer() => Vec(n, t)
// Bit::Or | Bit::And | Bit::XOr  =>  [Vec(n, t), Vec(n, t)  ] if t.is_integer() => Vec(n, t)
// Bit::Shl | Bit::Shr           =>  [Vec(n, t), Vec(n, U32)] if t.is_integer() => Vec(n, t)

// And= | Or= | Xor= => [ Ref(Vec(n, t), RW), Vec(n, t)   ] if t.is_integer() => Unit,
// Shr= | Shl=       => [ Ref(Vec(n, t), RW), Vec(n, U32) ] if t.is_integer() => Unit,

impl_ops! {
    <L: Len> Not !op: not(x: vec<bool, L>) -> vec<bool, L>: {op(x.as_any()).into()};
    <L: Len, T: ScalarTypeInteger> Not !op: not(x: vec<T, L>) -> vec<T, L>: {op(x.as_any()).into()};
}

impl_ops! { + also implements lhs <-> rhs swapped
    <> BitAnd !op: bitand(l: vec<u32, x1>, r: u32) -> vec<u32, x1>: {op(l.to_any(), r.to_any()).into()};
    <> BitOr  !op: bitor (l: vec<u32, x1>, r: u32) -> vec<u32, x1>: {op(l.to_any(), r.to_any()).into()};
    <> BitXor !op: bitxor(l: vec<u32, x1>, r: u32) -> vec<u32, x1>: {op(l.to_any(), r.to_any()).into()};
    <> BitAnd !op: bitand(l: vec<i32, x1>, r: i32) -> vec<i32, x1>: {op(l.to_any(), r.to_any()).into()};
    <> BitOr  !op: bitor (l: vec<i32, x1>, r: i32) -> vec<i32, x1>: {op(l.to_any(), r.to_any()).into()};
    <> BitXor !op: bitxor(l: vec<i32, x1>, r: i32) -> vec<i32, x1>: {op(l.to_any(), r.to_any()).into()};
}

impl_ops! {
    <L: Len, N: ScalarTypeInteger> BitAnd !op: bitand(l: vec<N, L>, r: vec<N, L>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
    <L: Len, N: ScalarTypeInteger> BitOr  !op: bitor (l: vec<N, L>, r: vec<N, L>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
    <L: Len, N: ScalarTypeInteger> BitXor !op: bitxor(l: vec<N, L>, r: vec<N, L>) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};

    <L: Len , N: ScalarTypeInteger> Shl !op: shl(l: vec<N, L>, r: vec<u32, L >) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
    <L: Len , N: ScalarTypeInteger> Shr !op: shr(l: vec<N, L>, r: vec<u32, L >) -> vec<N, L>: {op(l.as_any(), r.as_any()).into()};
    <L: Len2, N: ScalarTypeInteger> Shl !op: shl(l: vec<N, L>, r: vec<u32, x1>) -> vec<N, L>: {op(l.as_any(), r.as_any().splat(L::LEN, u32::SCALAR_TYPE)).into()};
    <L: Len2, N: ScalarTypeInteger> Shr !op: shr(l: vec<N, L>, r: vec<u32, x1>) -> vec<N, L>: {op(l.as_any(), r.as_any().splat(L::LEN, u32::SCALAR_TYPE)).into()};
    <L: Len , N: ScalarTypeInteger> Shl !op: shl(l: vec<N, L>, r: u32)          -> vec<N, L>: {op(l.as_any(), r.to_any().splat(L::LEN, u32::SCALAR_TYPE)).into()};
    <L: Len , N: ScalarTypeInteger> Shr !op: shr(l: vec<N, L>, r: u32)          -> vec<N, L>: {op(l.as_any(), r.to_any().splat(L::LEN, u32::SCALAR_TYPE)).into()};
}

impl_ops! {
    <L: Len, N: ScalarTypeInteger, AS: AddressSpace> BitAndAssign !op: bitand_assign(l: &mut Ref<vec<N  , L >, AS, ReadWrite>, r: vec<N  , L >): {op(&mut l.as_any(), r.as_any())};
    <L: Len, N: ScalarTypeInteger, AS: AddressSpace> BitOrAssign  !op: bitor_assign (l: &mut Ref<vec<N  , L >, AS, ReadWrite>, r: vec<N  , L >): {op(&mut l.as_any(), r.as_any())};
    <L: Len, N: ScalarTypeInteger, AS: AddressSpace> BitXorAssign !op: bitxor_assign(l: &mut Ref<vec<N  , L >, AS, ReadWrite>, r: vec<N  , L >): {op(&mut l.as_any(), r.as_any())};
    <AS: AddressSpace>                               BitAndAssign !op: bitand_assign(l: &mut Ref<vec<u32, x1>, AS, ReadWrite>, r: u32): {op(&mut l.as_any(), r.to_any())};
    <AS: AddressSpace>                               BitOrAssign  !op: bitor_assign (l: &mut Ref<vec<u32, x1>, AS, ReadWrite>, r: u32): {op(&mut l.as_any(), r.to_any())};
    <AS: AddressSpace>                               BitXorAssign !op: bitxor_assign(l: &mut Ref<vec<u32, x1>, AS, ReadWrite>, r: u32): {op(&mut l.as_any(), r.to_any())};
    <AS: AddressSpace>                               BitAndAssign !op: bitand_assign(l: &mut Ref<vec<i32, x1>, AS, ReadWrite>, r: i32): {op(&mut l.as_any(), r.to_any())};
    <AS: AddressSpace>                               BitOrAssign  !op: bitor_assign (l: &mut Ref<vec<i32, x1>, AS, ReadWrite>, r: i32): {op(&mut l.as_any(), r.to_any())};
    <AS: AddressSpace>                               BitXorAssign !op: bitxor_assign(l: &mut Ref<vec<i32, x1>, AS, ReadWrite>, r: i32): {op(&mut l.as_any(), r.to_any())};
}

impl_ops! {
    <L: Len , N: ScalarTypeInteger, AS: AddressSpace> ShlAssign !op: shl_assign(l: &mut Ref<vec<N, L>, AS, ReadWrite>, r: vec<u32, L >): {op(&mut l.as_any(), r.as_any())};
    <L: Len , N: ScalarTypeInteger, AS: AddressSpace> ShrAssign !op: shr_assign(l: &mut Ref<vec<N, L>, AS, ReadWrite>, r: vec<u32, L >): {op(&mut l.as_any(), r.as_any())};
    <L: Len2, N: ScalarTypeInteger, AS: AddressSpace> ShlAssign !op: shl_assign(l: &mut Ref<vec<N, L>, AS, ReadWrite>, r: vec<u32, x1>): {op(&mut l.as_any(), r.as_any().splat(L::LEN, u32::SCALAR_TYPE))};
    <L: Len2, N: ScalarTypeInteger, AS: AddressSpace> ShrAssign !op: shr_assign(l: &mut Ref<vec<N, L>, AS, ReadWrite>, r: vec<u32, x1>): {op(&mut l.as_any(), r.as_any().splat(L::LEN, u32::SCALAR_TYPE))};
    <L: Len , N: ScalarTypeInteger, AS: AddressSpace> ShlAssign !op: shl_assign(l: &mut Ref<vec<N, L>, AS, ReadWrite>, r: u32         ): {op(&mut l.as_any(), r.to_any().splat(L::LEN, u32::SCALAR_TYPE))};
    <L: Len , N: ScalarTypeInteger, AS: AddressSpace> ShrAssign !op: shr_assign(l: &mut Ref<vec<N, L>, AS, ReadWrite>, r: u32         ): {op(&mut l.as_any(), r.to_any().splat(L::LEN, u32::SCALAR_TYPE))};
}

#[allow(non_camel_case_types)]
#[rustfmt::skip]
impl<T: ScalarType, L: Len, AS: AddressSpace> Ref<vec<T, L>, AS, ReadWrite> {
    /// (no documentation yet)
    #[track_caller] pub fn set_add<x1_or_L: OneOr<L>>(mut self, rhs: impl To<vec<T, x1_or_L>>) where T: ScalarTypeNumber { self.as_any().add_assign(rhs.to_any().splat(L::LEN, T::SCALAR_TYPE)) }
    /// (no documentation yet)
    #[track_caller] pub fn set_sub<x1_or_L: OneOr<L>>(mut self, rhs: impl To<vec<T, x1_or_L>>) where T: ScalarTypeNumber { self.as_any().sub_assign(rhs.to_any().splat(L::LEN, T::SCALAR_TYPE)) }
    /// (no documentation yet)
    #[track_caller] pub fn set_mul<x1_or_L: OneOr<L>>(mut self, rhs: impl To<vec<T, x1_or_L>>) where T: ScalarTypeNumber { self.as_any().mul_assign(rhs.to_any().splat(L::LEN, T::SCALAR_TYPE)) }
    /// (no documentation yet)
    #[track_caller] pub fn set_div<x1_or_L: OneOr<L>>(mut self, rhs: impl To<vec<T, x1_or_L>>) where T: ScalarTypeNumber { self.as_any().div_assign(rhs.to_any().splat(L::LEN, T::SCALAR_TYPE)) }
    /// (no documentation yet)
    #[track_caller] pub fn set_rem<x1_or_L: OneOr<L>>(mut self, rhs: impl To<vec<T, x1_or_L>>) where T: ScalarTypeNumber { self.as_any().rem_assign(rhs.to_any().splat(L::LEN, T::SCALAR_TYPE)) }
    
    /// (no documentation yet)
    #[track_caller] pub fn set_bitand(mut self, rhs: impl To<vec<T, L>>) where T: ScalarTypeInteger {self.bitand_assign(rhs.to_gpu())}
    /// (no documentation yet)
    #[track_caller] pub fn set_bitor (mut self, rhs: impl To<vec<T, L>>) where T: ScalarTypeInteger {self.bitor_assign (rhs.to_gpu())}
    /// (no documentation yet)
    #[track_caller] pub fn set_bitxor(mut self, rhs: impl To<vec<T, L>>) where T: ScalarTypeInteger {self.bitxor_assign(rhs.to_gpu())}
    /// (no documentation yet)
    #[track_caller] pub fn set_shl   (mut self, rhs: impl To<vec<T, L>>) where T: ScalarTypeInteger {self.as_any().shl_assign(rhs.to_any())}
    /// (no documentation yet)
    #[track_caller] pub fn set_shr   (mut self, rhs: impl To<vec<T, L>>) where T: ScalarTypeInteger {self.as_any().shr_assign(rhs.to_any())}

    /// (no documentation yet)
    #[track_caller] pub fn set_not(mut self) where vec<T, L>: Not<Output = vec<T, L>> { self.set(!self.get()) }
    /// (no documentation yet)
    #[track_caller] pub fn set_neg(mut self) where T: ScalarTypeSigned { self.set(-self.get()) }
}
