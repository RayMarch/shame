//! convenience functions on tuples of types that implement certain traits
use shame_graph::Any;

use super::*;

///tuples of `impl` [`IntoRec`] types that can be converted to tuples of `impl` [`Rec`] via `(a, b, …, c).into_recs()`
pub trait IntoRecsTuple {
    /// Output tuple of [`Rec`] types
    type Output;
    /// Output tuple of [`Any`] values
    type OutputAnys;

    /// calls `into_rec` on every tuple element
    fn into_recs(self) -> Self::Output;
    /// calls `into_any` on every tuple element
    fn into_anys(self) -> Self::OutputAnys;
}

impl<A: IntoRec, B: IntoRec, C: IntoRec, D: IntoRec> IntoRecsTuple for (A, B, C, D) {
    type Output = (A::Rec, B::Rec, C::Rec, D::Rec);
    type OutputAnys = [Any; 4];

    fn into_recs(self) -> Self::Output {
        let (a, b, c, d) = self;
        (a.rec(), b.rec(), c.rec(), d.rec())
    }

    fn into_anys(self) -> Self::OutputAnys {
        let (a, b, c, d) = self;
        [a.into_any(), b.into_any(), c.into_any(), d.into_any()]
    }
}

impl<A: IntoRec, B: IntoRec, C: IntoRec> IntoRecsTuple for (A, B, C) {
    type Output = (A::Rec, B::Rec, C::Rec);
    type OutputAnys = [Any; 3];

    fn into_recs(self) -> Self::Output {
        let (a, b, c) = self;
        (a.rec(), b.rec(), c.rec())
    }

    fn into_anys(self) -> Self::OutputAnys {
        let (a, b, c) = self;
        [a.into_any(), b.into_any(), c.into_any()]
    }
}

impl<A: IntoRec, B: IntoRec> IntoRecsTuple for (A, B) {
    type Output = (A::Rec, B::Rec);
    type OutputAnys = [Any; 2];

    fn into_recs(self) -> Self::Output {
        let (a, b) = self;
        (a.rec(), b.rec())
    }

    fn into_anys(self) -> Self::OutputAnys {
        let (a, b) = self;
        [a.into_any(), b.into_any()]
    }
}

impl<A: IntoRec> IntoRecsTuple for (A,) {
    type Output = (A::Rec,);
    type OutputAnys = [Any; 1];

    fn into_recs(self) -> Self::Output {
        let (a,) = self;
        (a.rec(),)
    }

    fn into_anys(self) -> Self::OutputAnys {
        let (a,) = self;
        [a.into_any()]
    }
}

impl<A: IntoRec> IntoRecsTuple for std::ops::Range<A> {
    type Output = (A::Rec, A::Rec);
    type OutputAnys = [Any; 2];

    fn into_recs(self) -> Self::Output {
        (self.start.rec(), self.end.rec())
    }

    fn into_anys(self) -> Self::OutputAnys {
        [self.start.into_any(), self.end.into_any()]
    }
}

/// downcast tuples of [`Any`] to tuples of `impl`[`Rec`] types
pub trait FromAnys {
    /// tuple of [`Any`]
    type Input;
    #[track_caller]
    /// downcast tuples of [`Any`] to tuples of `impl`[`Rec`] types
    fn from_downcast(anys: &Self::Input, stage: Stage) -> Self;
}

impl<AS: Shape, AD: DType, BS: Shape, BD: DType> FromAnys for (Ten<AS, AD>, Ten<BS, BD>) {
    type Input = [Any; 2];
    #[track_caller]
    fn from_downcast([a, b]: &Self::Input, stage: Stage) -> Self {
        (a.downcast(stage), b.downcast(stage))
    }
}

impl<AS: Shape, AD: DType, BS: Shape, BD: DType, CS: Shape, CD: DType> FromAnys
    for (Ten<AS, AD>, Ten<BS, BD>, Ten<CS, CD>)
{
    type Input = [Any; 3];
    #[track_caller]
    fn from_downcast([a, b, c]: &Self::Input, stage: Stage) -> Self {
        (a.downcast(stage), b.downcast(stage), c.downcast(stage))
    }
}

impl<AS: Shape, AD: DType, BS: Shape, BD: DType, CS: Shape, CD: DType, DS: Shape, DD: DType>
    FromAnys for (Ten<AS, AD>, Ten<BS, BD>, Ten<CS, CD>, Ten<DS, DD>)
{
    type Input = [Any; 4];
    #[track_caller]
    fn from_downcast([a, b, c, d]: &Self::Input, stage: Stage) -> Self {
        (
            a.downcast(stage),
            b.downcast(stage),
            c.downcast(stage),
            d.downcast(stage),
        )
    }
}

/// call min and max on mixed tuples of tensors or types that can be turned into
/// that tensor, e.g `(_float, _f32, _float).min()`
pub trait MinMaxTuple {
    /// output tuple of `impl`[`Rec`] types
    type Output;

    /// returns the minimum of all tuple members
    fn min(self) -> Self::Output;
    /// returns the maximum of all tuple members
    fn max(self) -> Self::Output;
}

macro_rules! impl_MinMaxTuple {
    ($A: ident $(, $Tail: ident)*) => {
        impl<$A: AsFloat, $($Tail: AsFloat),*> MinMaxTuple for ($A, $($Tail),*) {
            type Output = float;

            fn min(self) -> Self::Output {
                #[allow(non_snake_case)] let ($A, $($Tail),*) = self;
                #[allow(non_snake_case)] let ($A, $($Tail),*) = ($A.as_ten(), $($Tail.as_ten()),*);
                let val = $A;
                $(
                    let val = val.min($Tail);
                )*
                val
            }

            fn max(self) -> Self::Output {
                #[allow(non_snake_case)] let ($A, $($Tail),*) = self;
                #[allow(non_snake_case)] let ($A, $($Tail),*) = ($A.as_ten(), $($Tail.as_ten()),*);
                let val = $A;
                $(
                    let val = val.max($Tail);
                )*
                val
            }
        }
    };
}

impl_MinMaxTuple!(A, B);
impl_MinMaxTuple!(A, B, C);
impl_MinMaxTuple!(A, B, C, D);
impl_MinMaxTuple!(A, B, C, D, E);
impl_MinMaxTuple!(A, B, C, D, E, F);
