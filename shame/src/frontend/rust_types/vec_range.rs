use thiserror::Error;

use super::{len::x1, vec::ToVec, vec_range_traits::*};
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

fn with_sides<T>([start, end]: [T; 2]) -> [(T, BoundSide); 2] { [(start, BoundSide::Start), (end, BoundSide::End)] }

impl VecBoundsInclusive {
    fn iter(&self) -> impl Iterator<Item = (AnyVec, BoundSide)> { with_sides(self.0).into_iter() }
}
impl VecBounds {
    fn iter(&self) -> impl Iterator<Item = (VecBound, BoundSide)> { with_sides(self.0).into_iter() }
}
impl VecOptBounds {
    fn iter(&self) -> impl Iterator<Item = (Option<VecBound>, BoundSide)> { with_sides(self.0).into_iter() }
}

#[derive(Clone, Copy, Debug)]
pub enum BoundSide {
    Start,
    End,
}

#[derive(Clone, Copy, Debug)]
pub enum Inclusivity {
    Incl,
    Excl,
}

#[derive(Clone, Copy, Debug)]
pub struct AnyVec {
    len: ir::Len,
    t: ir::ScalarType,
    any: Any,
}

impl<V: ToVec> From<&V> for AnyVec {
    #[track_caller]
    fn from(vec: &V) -> Self {
        AnyVec {
            len: V::L::LEN,
            t: V::T::SCALAR_TYPE,
            any: vec.to_any(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VecBound(AnyVec, Inclusivity);

impl VecBound {
    #[track_caller]
    pub(super) fn new_excl(v: &impl ToVec) -> Self { VecBound(v.into(), Inclusivity::Excl) }
    #[track_caller]
    pub(super) fn new_inclusive(v: &impl ToVec) -> Self { VecBound(v.into(), Inclusivity::Incl) }
}

pub struct VecBoundsInclusive(pub(super) [AnyVec; 2]);
pub struct VecBounds(pub(super) [VecBound; 2]);
pub struct VecOptBounds(pub(super) [Option<VecBound>; 2]);
pub struct VecOptBoundsInclusive(pub(super) [Option<AnyVec>; 2]);

#[derive(Clone, Copy)]
pub enum VecBoundsByLen<T: ScalarType, L: Len> {
    X1([(vec<T, x1>, Inclusivity); 2]),
    L([(vec<T, L>, Inclusivity); 2]),
}

impl<T: ScalarType, L: Len> VecBoundsByLen<T, L> {
    #[track_caller]
    pub fn push_error_if_needed(result: Result<Self, VecRangeError>) -> Self {
        Context::try_with(call_info!(), |ctx| match result {
            Ok(x) => x,
            Err(e) => {
                ctx.push_error(EncodingErrorKind::FrontendError(e.into()));
                Self::new_invalid(InvalidReason::ErrorThatWasPushed)
            }
        })
        .unwrap_or_else(|| Self::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
    }

    fn new_invalid(reason: InvalidReason) -> Self {
        let any = Any::new_invalid(reason);
        Self::L([(any.into(), Inclusivity::Incl), (any.into(), Inclusivity::Incl)])
    }
}

#[derive(Error, Debug, Clone)]
pub enum VecRangeError {
    #[error("vector range error: incompatible vector dimensions: {0} vs {1}")]
    IncompatibleLenPair(ir::Len, ir::Len),
}

impl VecBounds {
    pub fn by_len<T: ScalarType, L: Len>(self) -> VecBoundsByLen<T, L> {
        VecBoundsByLen::push_error_if_needed(self.try_by_len())
    }

    pub fn scalar<T: ScalarType>(self) -> [(vec<T, x1>, Inclusivity); 2] {
        match self.by_len::<T, x1>() {
            VecBoundsByLen::X1(x) => x,
            VecBoundsByLen::L(x) => x,
        }
    }

    pub fn try_by_len<T: ScalarType, L: Len>(self) -> Result<VecBoundsByLen<T, L>, VecRangeError> {
        let t = T::SCALAR_TYPE;
        let len = L::LEN;

        let [VecBound(av, ai), VecBound(bv, bi)] = self.0;

        let max_len = av.len.max(bv.len);
        for len in [av.len, bv.len] {
            match len {
                ir::Len::X1 => Ok(()),
                _ if len == max_len => Ok(()),
                other => Err(VecRangeError::IncompatibleLenPair(av.len, bv.len)),
            }?;
        }

        let by_len = match max_len {
            ir::Len::X1 => VecBoundsByLen::X1([(av.any.into(), ai), (bv.any.into(), bi)]),
            _ => VecBoundsByLen::L({
                let [av, bv] = [av, bv].map(|vec| {
                    // splat only if needed
                    match vec.len {
                        ir::Len::X1 => vec.any.splat(max_len, vec.t),
                        _ => vec.any,
                    }
                    .into()
                });
                [(av, ai), (bv, bi)]
            }),
        };
        Ok(by_len)
    }
}

impl VecOptBoundsInclusive {
    pub fn into_full_len_vecs<T: ScalarType, L: Len>(self) -> [Option<vec<T, L>>; 2] {
        let result = self.try_as_typed_vecs();
        Context::try_with(call_info!(), |ctx| match result {
            Ok(x) => x,
            Err(e) => {
                ctx.push_error(EncodingErrorKind::FrontendError(e.into()));
                let invalid = Any::new_invalid(InvalidReason::ErrorThatWasPushed).into();
                [Some(invalid); 2]
            }
        })
        .unwrap_or_else(|| {
            let invalid = Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding).into();
            [Some(invalid); 2]
        })
    }

    pub fn try_as_typed_vecs<T: ScalarType, L: Len>(self) -> Result<[Option<vec<T, L>>; 2], VecRangeError> {
        let splat_if_needed = |vec: AnyVec| -> Result<vec<T, L>, VecRangeError> {
            match vec.len {
                len if len == L::LEN => Ok(vec.any.into()),
                ir::Len::X1 => Ok(vec.any.splat(L::LEN, vec.t).into()),
                len => Err(VecRangeError::IncompatibleLenPair(len, L::LEN)),
            }
        };

        let result = match self.0 {
            [None, None] => [None, None],
            [None, Some(b)] => [None, Some(splat_if_needed(b)?)],
            [Some(a), None] => [Some(splat_if_needed(a)?), None],
            [Some(a), Some(b)] => [Some(splat_if_needed(a)?), Some(splat_if_needed(b)?)],
        };
        Ok(result)
    }
}
