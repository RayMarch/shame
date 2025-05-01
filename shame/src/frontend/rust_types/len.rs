#![allow(non_camel_case_types)]
use std::marker::PhantomData;

use super::{
    layout_traits::FromAnys, mem::AddressSpace, reference::AccessMode, scalar_type::ScalarType,
    type_traits::EmptyRefFields, vec::Components,
};
use crate::ir;

/// component count for [`vec`]
///
/// implemented by the marker types [`x1`], [`x2`], [`x3`] and [`x4`]
/// to represent 1, 2, 3 or 4 dimensional vectors respectively
///
/// ## Examples
/// ```
/// let _: vec<f32, x1>; // `f32x1` scalar
/// let _: vec<u32, x3>; // `u32x3` 3 component vector
/// ```
///
/// [`mat`]: crate::mat
/// [`vec`]: crate::vec
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a component count (`Len`). `Len` is only implemented by the marker types `x1`, `x2`, `x3` and `x4`"
)]
pub trait Len: Copy {
    /// `Self` as a `usize` number
    const USIZE: usize;
    /// `Self` as an enum for the runtime api
    const LEN: ir::Len;
    #[doc(hidden)] // internal
    #[allow(private_bounds)]
    type VecComponents<T: ScalarType>: Components;
    #[doc(hidden)] // internal
    #[allow(private_bounds)]
    type VecComponentsRef<T: ScalarType, AS: AddressSpace, AM: AccessMode>: FromAnys + Copy;
}
use crate::frontend::rust_types::reference::Ref;

/// A [`Len`] that is greater or equal to 2
///
/// used as a component count for [`mat`]
///
/// implemented by the marker types [`x2`], [`x3`] and [`x4`]
/// to represent 2, 3 or 4 columns or rows respectively
///
/// ## Examples
/// ```
/// let _: mat<f32, x2, x3>; // 2 columns, 3 rows
/// //let _: mat<f32, x1, x1>; // error: `x1` does not implement `Len2`
/// ```
///
/// [`mat`]: crate::mat
/// [`vec`]: crate::vec
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a component count of at least 2. The `Len2` trait is only implemented by the marker types `x2`, `x3` and `x4`"
)]
pub trait Len2: Len + AtLeastLen<x2> {
    const LEN2: ir::Len2;
}

/// a count ([`Len`]) of 1
#[derive(Clone, Copy)]
pub struct x1;
/// a count ([`Len`]) of 2
#[derive(Clone, Copy)]
pub struct x2;
/// a count ([`Len`]) of 3
#[derive(Clone, Copy)]
pub struct x3;
/// a count ([`Len`]) of 4
#[derive(Clone, Copy)]
pub struct x4;

#[rustfmt::skip] impl Len for x1 {const USIZE: usize = 1; const LEN: ir::Len = ir::Len::X1; type VecComponents<T: ScalarType> = super::vec::EmptyComponents<T>; type VecComponentsRef<T: ScalarType, AS: AddressSpace, AM: AccessMode> = EmptyRefFields;}
#[rustfmt::skip] impl Len for x2 {const USIZE: usize = 2; const LEN: ir::Len = ir::Len::X2; type VecComponents<T: ScalarType> = super::vec::Xy  <T>; type VecComponentsRef<T: ScalarType, AS: AddressSpace, AM: AccessMode> = super::vec::RefXy  <T, AS, AM>; }
#[rustfmt::skip] impl Len for x3 {const USIZE: usize = 3; const LEN: ir::Len = ir::Len::X3; type VecComponents<T: ScalarType> = super::vec::Xyz <T>; type VecComponentsRef<T: ScalarType, AS: AddressSpace, AM: AccessMode> = super::vec::RefXyz <T, AS, AM>; }
#[rustfmt::skip] impl Len for x4 {const USIZE: usize = 4; const LEN: ir::Len = ir::Len::X4; type VecComponents<T: ScalarType> = super::vec::Xyzw<T>; type VecComponentsRef<T: ScalarType, AS: AddressSpace, AM: AccessMode> = super::vec::RefXyzw<T, AS, AM>; }

#[rustfmt::skip] impl Len2 for x2 { const LEN2: ir::Len2 = ir::Len2::X2; }
#[rustfmt::skip] impl Len2 for x3 { const LEN2: ir::Len2 = ir::Len2::X3; }
#[rustfmt::skip] impl Len2 for x4 { const LEN2: ir::Len2 = ir::Len2::X4; }

#[rustfmt::skip] impl From<x1> for ir::Len { fn from(_: x1) -> Self { ir::Len::X1 }}
#[rustfmt::skip] impl From<x2> for ir::Len { fn from(_: x2) -> Self { ir::Len::X2 }}
#[rustfmt::skip] impl From<x3> for ir::Len { fn from(_: x3) -> Self { ir::Len::X3 }}
#[rustfmt::skip] impl From<x4> for ir::Len { fn from(_: x4) -> Self { ir::Len::X4 }}

/// dimension count of a compute pipeline thread grid.
///
/// implemented by the marker types [`x1`], [`x2`] and [`x3`]
/// to represent 1, 2 or 3 dimensional compute grids respectively.
#[diagnostic::on_unimplemented(message = "`{Self}` is not a `Len` less than or equal to `x4`")]
pub trait GridDim: Len {}
impl GridDim for x1 {}
impl GridDim for x2 {}
impl GridDim for x3 {}


#[diagnostic::on_unimplemented(message = "`{Self}` is not a `Len` greater or equal to `{T}`")]
pub trait AtLeastLen<T: Len>: Len {}

impl<T: Len> AtLeastLen<x1> for T {}
impl AtLeastLen<x2> for x2 {}
impl AtLeastLen<x2> for x3 {}
impl AtLeastLen<x2> for x4 {}

impl AtLeastLen<x3> for x3 {}
impl AtLeastLen<x3> for x4 {}

impl AtLeastLen<x4> for x4 {}

#[diagnostic::on_unimplemented(message = "`{Self}` is not an even number length. Use `x2` or `x4`.")]
/// A [`Len`] that is a multiple of 2 (used for packed vectors).
///
/// implemented by the marker types [`x2`] and [`x4`]
pub trait LenEven: Len {
    const LEN_EVEN: ir::LenEven;
}
impl LenEven for x2 {
    const LEN_EVEN: ir::LenEven = ir::LenEven::X2;
}
impl LenEven for x4 {
    const LEN_EVEN: ir::LenEven = ir::LenEven::X4;
}


#[diagnostic::on_unimplemented(
    message = "cannot extend a `{Self}` vector by another `{T}` vector. Vectors can only have up to 4 components."
)]
pub trait ExtendBy<T: Len>: Len {
    type Len: Len;
}
#[rustfmt::skip] impl ExtendBy<x1> for x1 { type Len = x2; }
#[rustfmt::skip] impl ExtendBy<x1> for x2 { type Len = x3; }
#[rustfmt::skip] impl ExtendBy<x1> for x3 { type Len = x4; }
#[rustfmt::skip] impl ExtendBy<x2> for x1 { type Len = x3; }
#[rustfmt::skip] impl ExtendBy<x2> for x2 { type Len = x4; }
#[rustfmt::skip] impl ExtendBy<x3> for x1 { type Len = x4; }

#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a tuple of types implementing `Len` that sum up to less than or equal 4"
)]
/// implemented for tuples of [`Len`] types. The associated type [`LenSum::Sum`] is the
/// sum of tuple elements lengths
///
/// e.g. `<(x1, x2, x1) as LenSum>::Sum` is `x4`
pub trait LenSum {
    /// sum of the tuple elements' lengths
    type Sum: Len;
}

macro_rules! impl_len_sum {
    ($(($($shape: ty),*) -> $output: ty;)*) => {
        $(
            impl LenSum for ($($shape),*) {
                type Sum = $output;
            }
        )*
    };
}

impl_len_sum! {
    (x1, x1, x1, x1) -> x4;
    (x2, x1, x1) -> x4;
    (x1, x2, x1) -> x4;
    (x1, x1, x2) -> x4;
    (x2, x2) -> x4;
    (x1, x3) -> x4;
    (x3, x1) -> x4;

    (x1, x1, x1) -> x3;
    (x2, x1) -> x3;
    (x1, x2) -> x3;

    (x1, x1) -> x2;
}

#[diagnostic::on_unimplemented(message = "vector component count `{Self}` is neither `x1` nor `{L}`")]
pub trait OneOr<L: Len>: Len {}
impl<L: Len> OneOr<L> for L {}
impl OneOr<x2> for x1 {}
impl OneOr<x3> for x1 {}
impl OneOr<x4> for x1 {}
