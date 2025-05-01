use super::{
    mat::mat,
    reference::{AccessModeReadable, ReadWrite, ReadableRef},
    scalar_type::{ScalarTypeFp, ScalarTypeInteger, ScalarTypeNumber, ScalarTypeSigned},
    vec::ToVec,
    AsAny, GpuType, ToGpuType,
};
use crate::{
    common::floating_point::f16,
    frontend::rust_types::{len::*, scalar_type::ScalarType},
    impl_ops, ir,
    mem::AddressSpace,
};
use std::ops::*;

use crate::frontend::any::Any;
use crate::frontend::rust_types::vec::vec;

impl_ops! {
    <T: ScalarTypeFp, C: Len2, R: Len2> Add !op: add(l: mat<T, C, R>, r: mat<T, C, R>) -> mat<T, C, R>: {op(l.as_any(), r.as_any()).into()};
    <T: ScalarTypeFp, C: Len2, R: Len2> Sub !op: sub(l: mat<T, C, R>, r: mat<T, C, R>) -> mat<T, C, R>: {op(l.as_any(), r.as_any()).into()};

    <T: ScalarTypeFp, C: Len2, R: Len2> Mul !op: mul(l: mat<T, C, R>, r: vec<T, x1>) -> mat<T, C, R>: {op(l.as_any(), r.as_any()).into()};
    <T: ScalarTypeFp, C: Len2, R: Len2> Mul !op: mul(l: vec<T, x1>, r: mat<T, C, R>) -> mat<T, C, R>: {op(l.as_any(), r.as_any()).into()};

    <T: ScalarTypeFp, C: Len2, R: Len2> Mul !op: mul(l: mat<T, C, R>, r: vec<T, C>) -> vec<T, R>: {op(l.as_any(), r.as_any()).into()};
    <T: ScalarTypeFp, C: Len2, R: Len2> Mul !op: mul(l: vec<T, R>, r: mat<T, C, R>) -> vec<T, C>: {op(l.as_any(), r.as_any()).into()};

    <C: Len2, R: Len2, K: Len2, T: ScalarTypeFp>
    Mul !op: mul(l: mat<T, K, R>, r: mat<T, C, K>) -> mat<T, C, R>: {op(l.as_any(), r.as_any()).into()};
}
