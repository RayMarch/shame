use super::{len::x1, vec::ToVec, vec_range::*};
use crate::frontend::rust_types::vec::vec;
use crate::{
    call_info,
    frontend::{
        any::{Any, InvalidReason},
        encoding::EncodingErrorKind,
        rust_types::{
            len::{Len, OneOr},
            scalar_type::ScalarType,
        },
    },
    ir::{self, recording::Context},
};

//[old-doc] N-dimensional numeric value ranges.
//[old-doc]
//[old-doc] This trait allows for vector/scalar ranges to have a more concise notation than `(a0,a1,a2)..=(b0,b1,b2)` or similar.
//[old-doc]
//[old-doc] `IntoRange` for N-dimensional ranges is implemented for the following expressions:
//[old-doc] - `vector..=vector`, if they match the dimension N.
//[old-doc] - `scalar..=scalar`, which are implicitly converted to N-dimensional ranges as needed.
//[old-doc] - `(scalar, vector)` or `(vector, scalar)` pairs, which are interpreted as (from, to) `Range`s
//[old-doc]
//[old-doc] where `vector` is `vec<T, _>` and `scalar` is `vec<T, x1>`
//[old-doc]
//[old-doc] implicit conversion of rust's primitive types is performed via the [`ToGpuType`] trait.
//[old-doc]
//[old-doc] for example, `IntoRange<float3>` is implemented by
//[old-doc] - `(0.0, 0.0, 0.0)..=(1.0, 1.0, 1.0)`
//[old-doc] - `0.0..=1.0`, which is implicitly expanded to the above
//[old-doc] - `my_float3..=my_float3`
//[old-doc] - `my_float..=my_float`, which is implicitly expanded to 3 dimensions
//[old-doc] - `(my_float3, 1.0)` for heterogenous types. The pair is interpreted as (start, end) of `Range`. The scalar is expanded to the amount of dimensions of the non-scalar.
//[old-doc]
/// (no documentation yet)
// inclusive and exclusive ranges
pub trait VecRange<T: ScalarType, L: Len> {
    /// (no documentation yet)
    fn get_bounds(&self) -> VecBounds;
}

#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be interpreted as a closed, end-inclusive range of scalars or vectors"
)]
// inclusive ranges only
// (for clamp)
/// (no documentation yet)
pub trait VecRangeInclusive<T: ScalarType, L: Len> {
    /// (no documentation yet)
    fn get_bounds_inclusive(&self) -> VecBoundsInclusive;
}

#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be interpreted as a closed or half-open range of scalars or vectors"
)]
// inclusive, exclusive, unbounded, half-open ranges
/// (no documentation yet)
pub trait VecRangeBounds<T: ScalarType, L: Len> {
    /// (no documentation yet)
    fn get_opt_bounds(&self) -> VecOptBounds;
}

#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be interpreted as an end-inclusive, closed or half-open range of scalars or vectors"
)]
// inclusive-only: full, half-open and closed ranges
/// (no documentation yet)
pub trait VecRangeBoundsInclusive<T: ScalarType, L: Len> {
    /// (no documentation yet)
    fn get_opt_bounds_inclusive(&self) -> VecOptBoundsInclusive;
}

impl<Vec, Lor1, L: Len> VecRangeInclusive<Vec::T, L> for std::ops::RangeInclusive<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    fn get_bounds_inclusive(&self) -> VecBoundsInclusive {
        VecBoundsInclusive([self.start().into(), self.end().into()])
    }
}

impl<Vec, Lor1, L: Len> VecRange<Vec::T, L> for std::ops::RangeInclusive<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    #[track_caller]
    fn get_bounds(&self) -> VecBounds {
        VecBounds([
            VecBound::new_inclusive(self.start()),
            VecBound::new_inclusive(self.end()),
        ])
    }
}

impl<Vec, Lor1, L: Len> VecRange<Vec::T, L> for std::ops::Range<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    fn get_bounds(&self) -> VecBounds {
        VecBounds([VecBound::new_inclusive(&self.start), VecBound::new_excl(&self.end)])
    }
}

impl<Vec, Lor1, L: Len> VecRangeBounds<Vec::T, L> for std::ops::RangeInclusive<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    fn get_opt_bounds(&self) -> VecOptBounds {
        VecOptBounds([
            Some(VecBound::new_inclusive(self.start())),
            Some(VecBound::new_inclusive(self.end())),
        ])
    }
}

impl<Vec, Lor1, L: Len> VecRangeBounds<Vec::T, L> for std::ops::Range<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    fn get_opt_bounds(&self) -> VecOptBounds {
        VecOptBounds([
            Some(VecBound::new_inclusive(&self.start)),
            Some(VecBound::new_excl(&self.end)),
        ])
    }
}

impl<Vec, Lor1, L: Len> VecRangeBounds<Vec::T, L> for std::ops::RangeFrom<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    fn get_opt_bounds(&self) -> VecOptBounds { VecOptBounds([Some(VecBound::new_inclusive(&self.start)), None]) }
}

impl<Vec, Lor1, L: Len> VecRangeBounds<Vec::T, L> for std::ops::RangeTo<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    fn get_opt_bounds(&self) -> VecOptBounds { VecOptBounds([None, Some(VecBound::new_excl(&self.end))]) }
}

impl<Vec, Lor1, L: Len> VecRangeBounds<Vec::T, L> for std::ops::RangeToInclusive<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    fn get_opt_bounds(&self) -> VecOptBounds { VecOptBounds([None, Some(VecBound::new_inclusive(&self.end))]) }
}

impl<T: ScalarType, L: Len> VecRangeBounds<T, L> for std::ops::RangeFull {
    fn get_opt_bounds(&self) -> VecOptBounds { VecOptBounds([None, None]) }
}

impl<Vec, Lor1, L: Len> VecRangeBoundsInclusive<Vec::T, L> for std::ops::RangeInclusive<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    fn get_opt_bounds_inclusive(&self) -> VecOptBoundsInclusive {
        VecOptBoundsInclusive([Some(self.start().into()), Some(self.end().into())])
    }
}

impl<Vec, Lor1, L: Len> VecRangeBoundsInclusive<Vec::T, L> for std::ops::RangeFrom<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    fn get_opt_bounds_inclusive(&self) -> VecOptBoundsInclusive {
        VecOptBoundsInclusive([Some((&self.start).into()), None])
    }
}

impl<Vec, Lor1, L: Len> VecRangeBoundsInclusive<Vec::T, L> for std::ops::RangeToInclusive<Vec>
where
    Vec: ToVec<L = Lor1>,
    Lor1: OneOr<L>,
{
    fn get_opt_bounds_inclusive(&self) -> VecOptBoundsInclusive {
        VecOptBoundsInclusive([None, Some((&self.end).into())])
    }
}

impl<T: ScalarType, L: Len> VecRangeBoundsInclusive<T, L> for std::ops::RangeFull {
    fn get_opt_bounds_inclusive(&self) -> VecOptBoundsInclusive { VecOptBoundsInclusive([None, None]) }
}

#[cfg(feature = "shorthand_float_ranges")]
mod shorthand_float_ranges {
    use super::*;

    impl<L: Len> VecRangeInclusive<f32, L> for std::ops::RangeInclusive<i32> {
        fn get_bounds_inclusive(&self) -> VecBoundsInclusive {
            VecBoundsInclusive([(&(*self.start() as f32)).into(), (&(*self.end() as f32)).into()])
        }
    }

    impl<L: Len> VecRange<f32, L> for std::ops::RangeInclusive<i32> {
        fn get_bounds(&self) -> VecBounds {
            VecBounds([
                VecBound::new_inclusive(&(*self.start() as f32)),
                VecBound::new_inclusive(&(*self.end() as f32)),
            ])
        }
    }

    impl<L: Len> VecRange<f32, L> for std::ops::Range<i32> {
        fn get_bounds(&self) -> VecBounds {
            VecBounds([
                VecBound::new_inclusive(&(self.start as f32)),
                VecBound::new_excl(&(self.end as f32)),
            ])
        }
    }

    impl<L: Len> VecRangeBounds<f32, L> for std::ops::RangeInclusive<i32> {
        fn get_opt_bounds(&self) -> VecOptBounds {
            VecOptBounds([
                Some(VecBound::new_inclusive(&(*self.start() as f32))),
                Some(VecBound::new_inclusive(&(*self.end() as f32))),
            ])
        }
    }

    impl<L: Len> VecRangeBounds<f32, L> for std::ops::Range<i32> {
        fn get_opt_bounds(&self) -> VecOptBounds {
            VecOptBounds([
                Some(VecBound::new_inclusive(&(self.start as f32))),
                Some(VecBound::new_excl(&(self.end as f32))),
            ])
        }
    }

    impl<L: Len> VecRangeBounds<f32, L> for std::ops::RangeFrom<i32> {
        fn get_opt_bounds(&self) -> VecOptBounds {
            VecOptBounds([Some(VecBound::new_inclusive(&(self.start as f32))), None])
        }
    }

    impl<L: Len> VecRangeBounds<f32, L> for std::ops::RangeTo<i32> {
        fn get_opt_bounds(&self) -> VecOptBounds { VecOptBounds([None, Some(VecBound::new_excl(&(self.end as f32)))]) }
    }

    impl<L: Len> VecRangeBounds<f32, L> for std::ops::RangeToInclusive<i32> {
        fn get_opt_bounds(&self) -> VecOptBounds {
            VecOptBounds([None, Some(VecBound::new_inclusive(&(self.end as f32)))])
        }
    }

    impl<L: Len> VecRangeBoundsInclusive<f32, L> for std::ops::RangeInclusive<i32> {
        fn get_opt_bounds_inclusive(&self) -> VecOptBoundsInclusive {
            VecOptBoundsInclusive([
                Some((&(*self.start() as f32)).into()),
                Some((&(*self.end() as f32)).into()),
            ])
        }
    }

    impl<L: Len> VecRangeBoundsInclusive<f32, L> for std::ops::RangeFrom<i32> {
        fn get_opt_bounds_inclusive(&self) -> VecOptBoundsInclusive {
            VecOptBoundsInclusive([Some((&(self.start as f32)).into()), None])
        }
    }

    impl<L: Len> VecRangeBoundsInclusive<f32, L> for std::ops::RangeToInclusive<i32> {
        fn get_opt_bounds_inclusive(&self) -> VecOptBoundsInclusive {
            VecOptBoundsInclusive([None, Some((&(self.end as f32)).into())])
        }
    }
}
