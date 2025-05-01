use super::{len::*, scalar_type::ScalarType, vec::vec, vec::ToVec, To, ToGpuType};
use crate::common::floating_point::f16;
use crate::frontend::any::Any;
use crate::ir::ScalarConstant;
use crate::{boolx1, f16x1, f32x1, f64x1, i32x1, u32x1};

impl<A, B, T: ScalarType> ToGpuType for (A, B)
where
    A: ToVec<T = T>,
    B: ToVec<T = T>,
    (A::L, B::L): LenSum,
{
    type Gpu = vec<T, <(A::L, B::L) as LenSum>::Sum>;


    #[track_caller]
    fn to_gpu(&self) -> Self::Gpu {
        let (a, b) = self;
        let len = <Self::Gpu as ToVec>::L::LEN;
        let sty = <Self::Gpu as ToVec>::T::SCALAR_TYPE;
        Any::new_vec(len, sty, &[a.to_any(), b.to_any()]).into()
    }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { None }
}

impl<A, B, C, T: ScalarType> ToGpuType for (A, B, C)
where
    A: ToVec<T = T>,
    B: ToVec<T = T>,
    C: ToVec<T = T>,
    (A::L, B::L, C::L): LenSum,
{
    type Gpu = vec<T, <(A::L, B::L, C::L) as LenSum>::Sum>;

    #[track_caller]
    fn to_gpu(&self) -> Self::Gpu {
        let (a, b, c) = self;
        let len = <Self::Gpu as ToVec>::L::LEN;
        let sty = <Self::Gpu as ToVec>::T::SCALAR_TYPE;
        Any::new_vec(len, sty, &[a.to_any(), b.to_any(), c.to_any()]).into()
    }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { None }
}

impl<A, B, C, D, T: ScalarType> ToGpuType for (A, B, C, D)
where
    A: ToVec<L = x1, T = T>,
    B: ToVec<L = x1, T = T>,
    C: ToVec<L = x1, T = T>,
    D: ToVec<L = x1, T = T>,
{
    type Gpu = vec<T, x4>;


    #[track_caller]
    fn to_gpu(&self) -> Self::Gpu {
        let (a, b, c, d) = self;
        let len = <Self::Gpu as ToVec>::L::LEN;
        let sty = <Self::Gpu as ToVec>::T::SCALAR_TYPE;
        Any::new_vec(len, sty, &[a.to_any(), b.to_any(), c.to_any(), d.to_any()]).into()
    }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { None }
}

macro_rules! rust_primitive_types_as_scalars {
    ($($FXX: ident ($fXX: ty) -> $scalar_ty: ident;)*) => {$(

        impl ToGpuType for $fXX {
            type Gpu = vec<$fXX, x1>;

            #[track_caller]
            fn to_gpu(&self) -> Self::Gpu {
                Any::new_scalar(ScalarConstant:: $FXX (*self)).into()
            }

            fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> {None}
        }

    )*};
}

rust_primitive_types_as_scalars! {
    F16(f16)  -> f16x1;
    F32(f32)  -> f32x1;
    // use 1.0.to_f64x1() for f64 literals
    //TODO(release) find a way to add f64 that doesn't screw with rust literals being not default-interpreted as f32
    I32(i32)  -> i32x1;
    U32(u32)  -> u32x1;
    Bool(bool) -> boolx1;
}

pub trait ExplicitFloatingPointConversion {
    fn to_f16x1(&self) -> f16x1;
    fn to_f64x1(&self) -> f64x1;
}

impl ExplicitFloatingPointConversion for f64 {
    fn to_f16x1(&self) -> f16x1 { Any::new_scalar(ScalarConstant::F16(f16::from(*self))).into() }

    fn to_f64x1(&self) -> f64x1 { Any::new_scalar(ScalarConstant::F64(*self)).into() }
}

impl ExplicitFloatingPointConversion for f32 {
    fn to_f16x1(&self) -> f16x1 { Any::new_scalar(ScalarConstant::F16(f16::from(*self))).into() }

    fn to_f64x1(&self) -> f64x1 { Any::new_scalar(ScalarConstant::F64(*self as f64)).into() }
}

impl ExplicitFloatingPointConversion for f16 {
    fn to_f16x1(&self) -> f16x1 { Any::new_scalar(ScalarConstant::F16(*self)).into() }

    fn to_f64x1(&self) -> f64x1 { Any::new_scalar(ScalarConstant::F16(f32::from(*self).into())).into() }
}
