// convenience implementations of `std::ops` (+ - * ...) for
// `Buffer<vec>` and `Buffer<mat>`

use std::ops::{Add, BitAnd, BitOr, BitXor, Deref, Div, Mul, Rem, Shl, Shr, Sub};

use super::buffer::Buffer;
use crate::frontend::rust_types::vec::vec;
use crate::frontend::rust_types::{
    len::{x1, Len, Len2},
    mat::mat,
    scalar_type::{ScalarType, ScalarTypeFp, ScalarTypeInteger, ScalarTypeNumber},
    type_traits::{NoAtomics, NoBools, NoHandles},
};

// // Buf<mat> * vec
// impl<T: ScalarTypeFp, C: Len2, R: Len2, L: Len> Mul<vec<T, L>> for Buffer<mat<T, C, R>>
// where
//     Buffer<mat<T, C, R>>: Deref<Target = mat<T, C, R>>,
//     mat<T, C, R>: Mul<vec<T, L>>,
// {
//     type Output = <mat<T, C, R> as Mul<vec<T, L>>>::Output;

//     fn mul(self, rhs: vec<T, L>) -> Self::Output { *self * rhs }
// }

// // vec * Buf<mat>
// impl<T: ScalarTypeFp, C: Len2, R: Len2, L: Len> Mul<Buffer<mat<T, C, R>>> for vec<T, L>
// where
//     vec<T, L>: Mul<mat<T, C, R>>,
//     Buffer<mat<T, C, R>>: Deref<Target = mat<T, C, R>>,
// {
//     type Output = <Self as Mul<mat<T, C, R>>>::Output;

//     fn mul(self, rhs: Buffer<mat<T, C, R>>) -> Self::Output { self * *rhs.deref() }
// }


macro_rules! impl_binops_for_buffer {
    (
        $(
            impl<$($gen: ident: $bound: ident),*>
            $Lhs: ty, $Mul: ident :: $mul: ident, $Rhs: ty;
        )*
    ) => {
        // Buffer<A> x B
        $(
        impl<$($gen: $bound),*> $Mul<$Rhs> for Buffer<$Lhs>
        where
            $Lhs: NoBools,
            Buffer<$Lhs>: Deref<Target = $Lhs>,
            $Lhs: Mul<$Rhs>,
        {
            type Output = <$Lhs as $Mul<$Rhs>>::Output;

            fn $mul(self, rhs: $Rhs) -> Self::Output { (*self).$mul(rhs) }
        }

        // A x Buffer<B>
        impl<$($gen: $bound),*> $Mul<Buffer<$Rhs>> for $Lhs
        where
            $Rhs: NoBools,
            Buffer<$Rhs>: Deref<Target = $Rhs>,
            $Lhs: Mul<$Rhs>,
        {
            type Output = <$Lhs as $Mul<$Rhs>>::Output;

            fn $mul(self, rhs: Buffer<$Rhs>) -> Self::Output { self.$mul(*rhs) }
        }

        // Buffer<A> x Buffer<B>
        impl<$($gen: $bound),*> $Mul<Buffer<$Rhs>> for Buffer<$Lhs>
        where
            $Lhs: NoBools,
            $Rhs: NoBools,
            Buffer<$Lhs>: Deref<Target = $Lhs>,
            Buffer<$Rhs>: Deref<Target = $Rhs>,
            $Lhs: Mul<$Rhs>,
        {
            type Output = <$Lhs as $Mul<$Rhs>>::Output;

            fn $mul(self, rhs: Buffer<$Rhs>) -> Self::Output { (*self).$mul(*rhs) }
        }
        )*
    };
}

impl_binops_for_buffer! {
    // scalar * mat
    impl<T: ScalarTypeFp, C: Len2, R: Len2> vec<T, x1>  , Mul::mul, mat<T, C, R>;
    impl<T: ScalarTypeFp, C: Len2, R: Len2> mat<T, C, R>, Mul::mul, vec<T, x1>;

    // vec * mat
    impl<T: ScalarTypeFp, C: Len2, R: Len2, L: Len2> mat<T, C, R>, Mul::mul, vec<T, L>;
    impl<T: ScalarTypeFp, C: Len2, R: Len2, L: Len2> vec<T, L>   , Mul::mul, mat<T, C, R>;

    // mat * mat
    impl<C: Len2, R: Len2,          T: ScalarTypeFp> mat<T, C, R>, Mul::mul, mat<T, C, R>;

    // vec + - * / vec
    impl<T: ScalarTypeNumber, L: Len> vec<T, L>, Add::add, vec<T, L>;
    impl<T: ScalarTypeNumber, L: Len> vec<T, L>, Sub::sub, vec<T, L>;
    impl<T: ScalarTypeNumber, L: Len> vec<T, L>, Mul::mul, vec<T, L>;
    impl<T: ScalarTypeNumber, L: Len> vec<T, L>, Div::div, vec<T, L>;
    impl<T: ScalarTypeInteger, L: Len> vec<T, L>, Rem::rem, vec<T, L>;
    impl<T: ScalarTypeInteger, L: Len> vec<T, L>, BitAnd::bitand, vec<T, L>;
    impl<T: ScalarTypeInteger, L: Len> vec<T, L>, BitOr::bitor, vec<T, L>;
    impl<T: ScalarTypeInteger, L: Len> vec<T, L>, BitXor::bitxor, vec<T, L>;

    // mat * mat
    impl<T: ScalarTypeFp, C: Len2, R: Len2> mat<T, C, R>, Add::add, mat<T, C, R>;
}
