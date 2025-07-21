use std::{borrow::Cow, marker::PhantomData, ops::Deref};

use super::{
    index::GpuIndex,
    layout_traits::{ArrayElementsUnsizedError, FromAnys, GetAllFields, GpuLayout},
    len::{x1, x2, x3, x4, AtLeastLen, Len, Len2},
    mem::AddressSpace,
    reference::{AccessMode, AccessModeReadable},
    scalar_type::{ScalarType, ScalarTypeFp},
    type_layout::{
        self,
        layoutable::{self},
        TypeLayout,
    },
    type_traits::{
        BindingArgs, EmptyRefFields, GpuAligned, GpuSized, GpuStore, GpuStoreImplCategory, NoAtomics, NoBools,
        NoHandles,
    },
    vec::{scalar, vec, ToInteger},
    AsAny, GpuType, To, ToGpuType,
};
use crate::{frontend::rust_types::reference::Ref, ir::recording::CallInfoScope};
use crate::{
    call_info,
    frontend::{
        any::{
            shared_io::{BindPath, BindingType},
            Any, InvalidReason,
        },
        encoding::buffer::{BufferAddressSpace, BufferInner, BufferRefInner},
    },
    ir::{self, pipeline::StageMask, recording::Context, Comp4, VectorAccess},
};

/// A column major matrix with between 2 and 4 columns/rows
///
/// see <https://www.w3.org/TR/WGSL/#matrix-types>
///
/// for memory layout information see <https://www.w3.org/TR/WGSL/#alignment-and-size>
#[allow(non_camel_case_types)] // primitive type in shader
#[derive(Clone, Copy)]
pub struct mat<T: ScalarTypeFp, Cols: Len2, Rows: Len2> {
    any: Any,
    phantom: PhantomData<(Cols, Rows, T)>,
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> Default for mat<T, C, R> {
    #[track_caller]
    fn default() -> Self {
        Self::from(
            Context::try_with(call_info!(), |_| Any::new_mat(C::LEN2, R::LEN2, T::SCALAR_TYPE_FP, &[]))
                .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding)),
        )
    }
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> GpuLayout for mat<T, C, R> {
    fn layout_recipe() -> layoutable::LayoutableType {
        layoutable::Matrix {
            columns: C::LEN2,
            rows: R::LEN2,
            scalar: T::SCALAR_TYPE_FP,
        }
        .into()
    }

    fn cpu_type_name_and_layout() -> Option<Result<(Cow<'static, str>, TypeLayout), ArrayElementsUnsizedError>> { None }
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> FromAnys for mat<T, C, R> {
    fn expected_num_anys() -> usize { 1 }

    #[track_caller]
    fn from_anys(mut anys: impl Iterator<Item = Any>) -> Self { super::layout_traits::from_single_any(anys).into() }
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> GpuSized for mat<T, C, R> {
    fn sized_ty() -> ir::SizedType { ir::SizedType::Matrix(C::LEN2, R::LEN2, T::SCALAR_TYPE_FP) }
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> GpuType for mat<T, C, R> {
    fn ty() -> ir::Type { ir::Type::Store(Self::store_ty()) }

    fn from_any_unchecked(any: Any) -> Self {
        Self {
            any,
            phantom: PhantomData,
        }
    }
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> NoHandles for mat<T, C, R> {}
impl<T: ScalarTypeFp, C: Len2, R: Len2> NoAtomics for mat<T, C, R> {}
impl<T: ScalarTypeFp, C: Len2, R: Len2> NoBools for mat<T, C, R> {}

impl<T: ScalarTypeFp, C: Len2, R: Len2> AsAny for mat<T, C, R> {
    fn as_any(&self) -> Any { self.any }
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> GpuStore for mat<T, C, R> {
    type RefFields<AS: AddressSpace, AM: AccessMode> = EmptyRefFields;
    fn store_ty() -> ir::StoreType { ir::StoreType::Sized(<Self as GpuSized>::sized_ty()) }

    fn instantiate_buffer_inner<AS: BufferAddressSpace>(
        args: Result<BindingArgs, InvalidReason>,
        bind_ty: BindingType,
    ) -> BufferInner<Self, AS>
    where
        Self: NoAtomics + NoBools,
    {
        BufferInner::new_plain(args, bind_ty)
    }

    fn instantiate_buffer_ref_inner<AS: BufferAddressSpace, AM: AccessModeReadable>(
        args: Result<BindingArgs, InvalidReason>,
        bind_ty: BindingType,
    ) -> BufferRefInner<Self, AS, AM>
    where
        Self: NoBools,
    {
        BufferRefInner::new_plain(args, bind_ty)
    }

    fn impl_category() -> GpuStoreImplCategory { GpuStoreImplCategory::GpuType(Self::store_ty()) }
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> GpuAligned for mat<T, C, R> {
    fn aligned_ty() -> ir::AlignedType { ir::AlignedType::Sized(<Self as GpuSized>::sized_ty()) }
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> ToGpuType for mat<T, C, R> {
    type Gpu = Self;

    fn to_gpu(&self) -> Self::Gpu { *self }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { Some(self) }
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> From<Any> for mat<T, C, R> {
    #[track_caller]
    fn from(any: Any) -> Self { GpuType::from_any(any) }
}

impl<R: Len2, T: ScalarTypeFp> From<[vec<T, R>; 2]> for mat<T, x2, R> {
    #[rustfmt::skip]    fn from(cols: [vec<T, R>; 2]) -> Self {
        Any::new_mat(<x2 as Len2>::LEN2, R::LEN2, T::SCALAR_TYPE_FP, &cols.map(|col| col.as_any()),).into()
    }
}

impl<R: Len2, T: ScalarTypeFp> From<[vec<T, R>; 3]> for mat<T, x3, R> {
    #[rustfmt::skip]    fn from(cols: [vec<T, R>; 3]) -> Self {
        Any::new_mat(<x3 as Len2>::LEN2, R::LEN2, T::SCALAR_TYPE_FP, &cols.map(|col| col.as_any()),).into()
    }
}

impl<R: Len2, T: ScalarTypeFp> From<[vec<T, R>; 4]> for mat<T, x4, R> {
    #[rustfmt::skip]    fn from(cols: [vec<T, R>; 4]) -> Self {
        Any::new_mat(<x4 as Len2>::LEN2, R::LEN2, T::SCALAR_TYPE_FP, &cols.map(|col| col.as_any()),).into()
    }
}

impl<Cols: Len2, Rows: Len2, T: ScalarTypeFp> mat<T, Cols, Rows> {
    /// initializes a matrix from column vectors
    ///
    /// alternatively, the constructors [`mat::new`] and [`mat::from_rows`]
    /// can be used to initialize a matrix given its columns or rows respectively
    ///
    /// ## Example
    /// ```
    /// use shame as sm;
    ///
    /// // the matrix
    /// //
    /// // m: | 0.0  3.0 |
    /// //    | 1.0  4.0 |
    /// //    | 2.0  5.0 |
    /// //
    /// // can be constructed via either of:
    /// let m: f32x2x3 = sm::mat::new([
    ///     0.0, 1.0, 2.0, // column 0, top to bottom
    ///     3.0, 4.0, 5.0, // column 1, top to bottom
    /// ]);
    ///
    /// let m = sm::mat::from_cols([
    ///     sm::vec!(0.0, 1.0, 2.0) // column 0, top to bottom
    ///     sm::vec!(3.0, 4.0, 5.0) // column 1, top to bottom
    /// ]);
    ///
    /// let m = sm::mat::from_rows([
    ///     sm::vec!(0.0, 3.0) // row 1
    ///     sm::vec!(1.0, 4.0) // row 2
    ///     sm::vec!(2.0, 5.0) // row 3
    /// ])
    ///
    /// let m: f32x3x2 = sm::mat::new([
    ///     0.0, 3.0 // column 0, becomes row 0
    ///     1.0, 4.0 // column 1, becomes row 1
    ///     2.0, 5.0 // column 2, becomes row 2
    /// ]).transpose();
    ///
    /// // `From::from` does `from_cols`
    /// let m: f32x2x3 = [
    ///     sm::vec!(0.0, 1.0, 2.0) // column 0, top to bottom
    ///     sm::vec!(3.0, 4.0, 5.0) // column 1, top to bottom
    /// ].into();
    /// ```
    pub fn from_cols<const N: usize>(cols: [vec<T, Rows>; N]) -> Self
    where
        [vec<T, Rows>; N]: Into<Self>,
    {
        cols.into()
    }

    /// initializes a matrix from row vectors
    ///
    /// > note: the code generator is internally accessing the row components and
    /// > rearranging them to a call of [`mat::new`].
    ///
    /// alternatively, the constructors [`mat::new`] and [`mat::from_cols`]
    /// can be used to initialize a matrix given its columns or rows respectively
    ///
    /// ## Example
    /// ```
    /// use shame as sm;
    ///
    /// // the matrix
    /// //
    /// // m: | 0.0  3.0 |
    /// //    | 1.0  4.0 |
    /// //    | 2.0  5.0 |
    /// //
    /// // can be constructed via either of:
    /// let m: f32x2x3 = sm::mat::new([
    ///     0.0, 1.0, 2.0, // column 0, top to bottom
    ///     3.0, 4.0, 5.0, // column 1, top to bottom
    /// ]);
    ///
    /// let m = sm::mat::from_cols([
    ///     sm::vec!(0.0, 1.0, 2.0) // column 0, top to bottom
    ///     sm::vec!(3.0, 4.0, 5.0) // column 1, top to bottom
    /// ]);
    ///
    /// let m = sm::mat::from_rows([
    ///     sm::vec!(0.0, 3.0) // row 1
    ///     sm::vec!(1.0, 4.0) // row 2
    ///     sm::vec!(2.0, 5.0) // row 3
    /// ])
    ///
    /// let m: f32x3x2 = sm::mat::new([
    ///     0.0, 3.0 // column 0, becomes row 0
    ///     1.0, 4.0 // column 1, becomes row 1
    ///     2.0, 5.0 // column 2, becomes row 2
    /// ]).transpose();
    ///
    /// // `From::from` does `from_cols`
    /// let m: f32x2x3 = [
    ///     sm::vec!(0.0, 1.0, 2.0) // column 0, top to bottom
    ///     sm::vec!(3.0, 4.0, 5.0) // column 1, top to bottom
    /// ].into();
    /// ```
    pub fn from_rows<const N: usize>(rows: [vec<T, Cols>; N]) -> Self
    where
        [vec<T, Cols>; N]: Into<mat<T, Rows, Cols>>, // transposed trait bound from column constructor
    {
        use ir::Comp4::*;
        let mut column_maj = Vec::with_capacity(4 * 4);
        for col in &[X, Y, Z, W][0..Cols::USIZE] {
            for row in rows {
                column_maj.push(row.as_any().get_component(*col))
            }
        }
        Any::new_mat(Cols::LEN2, Rows::LEN2, T::SCALAR_TYPE_FP, &column_maj).into()
    }

    /// get the `i`'th column vector of `self`
    ///
    /// produces an indeterminate value if `i` is out of bounds, but does not cause undefined behavior
    /// (WGSL).
    ///
    /// see https://www.w3.org/TR/WGSL/#indeterminate-values
    pub fn col(&self, i: impl ToInteger) -> vec<T, Rows> { self.any.matrix_index(i.to_any()).into() }

    /// construct a matrix filled with zeroes
    ///
    /// ## Examples
    /// ```
    /// let x: mat<f32, x3, x2> = mat::zero();
    /// let x: mat<f32, x4, x4> = mat::zero();
    /// ```
    #[track_caller]
    pub fn zero() -> Self { Default::default() }

    /// resizes `self` to a different matrix size by either removing columns/rows
    /// or adding zero-filled columns/rows.
    // (no test case yet)
    #[track_caller]
    pub fn resize<NewCols: Len2, NewRows: Len2>(&self) -> mat<T, NewCols, NewRows> {
        use ir::VectorAccess::*;
        use std::cmp::Ordering::*;
        use Comp4::*;
        let call_info_scope = CallInfoScope::new(call_info!());

        let zero = || {
            Any::new_scalar(match T::SCALAR_TYPE_FP {
                ir::ScalarTypeFp::F16 => ir::ScalarConstant::F16(f16::from(0.0)),
                ir::ScalarTypeFp::F32 => ir::ScalarConstant::F32(0.0f32),
                ir::ScalarTypeFp::F64 => ir::ScalarConstant::F64(0.0f64),
            })
        };

        let idx = |i: u32| Any::new_scalar(ir::ScalarConstant::U32(i));

        let swizzle = [
            Swizzle1([X]), // unreachable, since matrices have at least 2 rows
            Swizzle2([X, Y]),
            Swizzle3([X, Y, Z]),
            Swizzle4([X, Y, Z, W]),
        ];

        let col_gain = NewCols::USIZE as i32 - Cols::USIZE as i32;
        let row_gain = NewRows::USIZE as i32 - Rows::USIZE as i32;

        let resize_column_vec: &dyn Fn(Any) -> Any = match row_gain {
            ..0 => &|col: Any| col.swizzle(swizzle[NewRows::USIZE - 1]),
            0 => &|col: Any| col,
            1 => &|col: Any| Any::new_vec(NewRows::LEN, T::SCALAR_TYPE, &[col, zero()]),
            2 => &|col: Any| Any::new_vec(NewRows::LEN, T::SCALAR_TYPE, &[col, zero(), zero()]),
            3.. => unreachable!("matrices can only gain up to 2 rows (from 2 to 4)"),
        };

        let m = self.any;

        let args: &[_] = match NewCols::LEN2 {
            ir::Len2::X2 => &[
                resize_column_vec(m.matrix_index(idx(0))),
                resize_column_vec(m.matrix_index(idx(1))),
            ],
            ir::Len2::X3 => &[
                resize_column_vec(m.matrix_index(idx(0))),
                resize_column_vec(m.matrix_index(idx(1))),
                resize_column_vec(m.matrix_index(idx(2))),
            ],
            ir::Len2::X4 => &[
                resize_column_vec(m.matrix_index(idx(0))),
                resize_column_vec(m.matrix_index(idx(1))),
                resize_column_vec(m.matrix_index(idx(2))),
                resize_column_vec(m.matrix_index(idx(3))),
            ],
        };
        Any::new_mat(NewCols::LEN2, NewRows::LEN2, T::SCALAR_TYPE_FP, args).into()
    }

    /// identical to `Into::into(_)` except that it is also usable in generic
    /// contexts. If unsure use `Into::into()` instead of this.
    ///
    /// ## why does this exist? why not use `Into::into`
    /// implementing matrix conversion from any floating point type to any other
    /// floating point type generically via the `From` trait is impossible
    /// because rusts `core` implementation of `From<T> for T` blocks this more
    /// generic conversion. Therefore the `From` implementation is split up into
    /// all the specific permutations.
    #[track_caller]
    pub fn into_generic<T1: ScalarTypeFp>(self) -> mat<T1, Cols, Rows> { <mat<T1, Cols, Rows>>::from_generic(self) }

    /// identical to `From::from()` except that it is also usable in generic
    /// contexts. If unsure use `From::from()` instead of this.
    ///
    /// ## why does this exist? why not use `From::from`
    /// implementing matrix conversion from any floating point type to any other
    /// floating point type generically via the `From` trait is impossible
    /// because rusts `core` implementation of `From<T> for T` blocks this more
    /// generic conversion. Therefore the `From` implementation is split up into
    /// all the specific permutations.
    #[track_caller]
    pub fn from_generic<T1: ScalarTypeFp>(value: mat<T1, Cols, Rows>) -> mat<T, Cols, Rows> {
        Any::new_mat(Cols::LEN2, Rows::LEN2, T::SCALAR_TYPE_FP, &[value.any]).into()
    }

    /// matrix transposition
    ///
    /// flips columns and rows
    ///
    #[track_caller]
    pub fn transpose(self) -> mat<T, Rows, Cols> { self.any.transpose().into() }
}

impl<C: Len2, T: ScalarTypeFp> mat<T, C, C> {
    /// the [determinant](https://en.wikipedia.org/wiki/Determinant) of the matrix
    ///
    /// see <https://www.w3.org/TR/WGSL/#determinant-builtin>
    #[track_caller]
    pub fn determinant(self) -> vec<T, x1> { self.as_any().determinant().into() }
}

macro_rules! impl_from {
    (
        $(From<mat<T, C, R>> for mat<$t: ty, C, R> where T in [$($t_from: ty),*];)*
    ) => {$($(
        impl<C: Len2, R: Len2> From<mat<$t_from, C, R>> for mat<$t, C, R> {
            #[track_caller]
            fn from(other: mat<$t_from, C, R>) -> Self {
                Self::from_generic(other)
            }
        }
    )*)*};
}

impl<T: ScalarTypeFp, L: Len2> mat<T, L, L> {
    /// constructs the identity transformation
    ///
    /// only implemented for square matrices:
    ///
    /// ----------------------
    /// `mat<_, x2, x2>::id()`:
    /// ```
    ///     1, 0
    ///     0, 1
    /// ```
    /// ----------------------
    /// `mat<_, x3, x3>::id()`:
    /// ```
    ///     1, 0, 0
    ///     0, 1, 0
    ///     0, 0, 1
    /// ```
    /// ----------------------
    /// `mat<_, x4, x4>::id()`:
    /// ```
    ///     1, 0, 0, 0
    ///     0, 1, 0, 0
    ///     0, 0, 1, 0
    ///     0, 0, 0, 1
    /// ```
    #[rustfmt::skip]
    pub fn id() -> Self {
        let fp_scalar = |n: u8| {
            Any::new_scalar(match T::SCALAR_TYPE_FP {
                ir::ScalarTypeFp::F16 => ir::ScalarConstant::F16(f16::from(n as f32)),
                ir::ScalarTypeFp::F32 => ir::ScalarConstant::F32(n as f32),
                ir::ScalarTypeFp::F64 => ir::ScalarConstant::F64(n as f64),
            })
        };

        let anys: &[_] = match L::LEN2 {
            ir::Len2::X2 => &[
                1, 0,
                0, 1,
            ].map(fp_scalar),
            ir::Len2::X3 => &[
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ].map(fp_scalar),
            ir::Len2::X4 => &[
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ].map(fp_scalar),
        };
        Any::new_mat(L::LEN2, L::LEN2, T::SCALAR_TYPE_FP, anys).into()
    }
}

pub trait NumComponents<const N: usize> {}
#[rustfmt::skip] impl<T: ScalarTypeFp> NumComponents<{2 * 2}> for mat<T, x2, x2> {}
#[rustfmt::skip] impl<T: ScalarTypeFp> NumComponents<{2 * 3}> for mat<T, x2, x3> {}
#[rustfmt::skip] impl<T: ScalarTypeFp> NumComponents<{2 * 4}> for mat<T, x2, x4> {}
#[rustfmt::skip] impl<T: ScalarTypeFp> NumComponents<{3 * 2}> for mat<T, x3, x2> {}
#[rustfmt::skip] impl<T: ScalarTypeFp> NumComponents<{3 * 3}> for mat<T, x3, x3> {}
#[rustfmt::skip] impl<T: ScalarTypeFp> NumComponents<{3 * 4}> for mat<T, x3, x4> {}
#[rustfmt::skip] impl<T: ScalarTypeFp> NumComponents<{4 * 2}> for mat<T, x4, x2> {}
#[rustfmt::skip] impl<T: ScalarTypeFp> NumComponents<{4 * 3}> for mat<T, x4, x3> {}
#[rustfmt::skip] impl<T: ScalarTypeFp> NumComponents<{4 * 4}> for mat<T, x4, x4> {}

impl<T: ScalarTypeFp, C: Len2, R: Len2> mat<T, C, R> {
    /// initializes a column major matrix from components, in the order in which
    /// they appear in memory
    ///
    /// alternatively, the constructors [`mat::from_cols`] and [`mat::from_rows`]
    /// can be used to initialize a matrix given its columns or rows respectively
    ///
    /// ## Example
    /// ```
    /// use shame as sm;
    ///
    /// // the matrix
    /// //
    /// // m: | 0.0  3.0 |
    /// //    | 1.0  4.0 |
    /// //    | 2.0  5.0 |
    /// //
    /// // can be constructed via either of:
    /// let m: f32x2x3 = sm::mat::new([
    ///     0.0, 1.0, 2.0, // column 0, top to bottom
    ///     3.0, 4.0, 5.0, // column 1, top to bottom
    /// ]);
    ///
    /// let m = sm::mat::from_cols([
    ///     sm::vec!(0.0, 1.0, 2.0) // column 0, top to bottom
    ///     sm::vec!(3.0, 4.0, 5.0) // column 1, top to bottom
    /// ]);
    ///
    /// let m = sm::mat::from_rows([
    ///     sm::vec!(0.0, 3.0) // row 1
    ///     sm::vec!(1.0, 4.0) // row 2
    ///     sm::vec!(2.0, 5.0) // row 3
    /// ])
    ///
    /// let m: f32x3x2 = sm::mat::new([
    ///     0.0, 3.0 // column 0, becomes row 0
    ///     1.0, 4.0 // column 1, becomes row 1
    ///     2.0, 5.0 // column 2, becomes row 2
    /// ]).transpose();
    ///
    /// // `From::from` does `from_cols`
    /// let m: f32x2x3 = [
    ///     sm::vec!(0.0, 1.0, 2.0) // column 0, top to bottom
    ///     sm::vec!(3.0, 4.0, 5.0) // column 1, top to bottom
    /// ].into();
    /// ```
    pub fn new<const N: usize>(components: [impl To<scalar<T>>; N]) -> Self
    where
        Self: NumComponents<N>,
    {
        Any::new_mat(C::LEN2, R::LEN2, T::SCALAR_TYPE_FP, &components.map(|x| x.to_any())).into()
    }
}

use crate::common::floating_point::f16;

#[rustfmt::skip]
impl_from! {
    // this is necessary to avoid colliding implementations with rusts `core` `impl From<T> for T`
    // use `from_generic` and `into_generic` if you are in a generic context instead.
    From<mat<T, C, R>> for mat<f16, C, R> where T in [     f32, f64];
    From<mat<T, C, R>> for mat<f32, C, R> where T in [f16,      f64];
    From<mat<T, C, R>> for mat<f64, C, R> where T in [f16, f32     ];
}

impl<Cols: Len2, Rows: Len2, T: ScalarTypeFp, Idx> GpuIndex<Idx> for mat<T, Cols, Rows>
where
    Idx: ToInteger,
{
    type Output = vec<T, Rows>;

    fn index(&self, i: Idx) -> Self::Output { self.col(i) }
}

impl<Cols: Len2, Rows: Len2, T: ScalarTypeFp, Idx, AM: AccessMode, AS: AddressSpace> GpuIndex<Idx>
    for Ref<mat<T, Cols, Rows>, AS, AM>
where
    Idx: ToInteger,
{
    type Output = Ref<vec<T, Rows>, AS, AM>;

    fn index(&self, i: Idx) -> Self::Output { self.col(i) }
}

impl<Cols: Len2, Rows: Len2, T: ScalarTypeFp, AM, AS> Ref<mat<T, Cols, Rows>, AS, AM>
where
    AM: AccessMode,
    AS: AddressSpace,
{
    /// get the `i`'th column vector of `self`
    ///
    /// produces an indeterminate value if `i` is out of bounds, but does not cause undefined behavior
    /// (WGSL).
    ///
    /// see https://www.w3.org/TR/WGSL/#indeterminate-values
    pub fn col(&self, i: impl ToInteger) -> Ref<vec<T, Rows>, AS, AM> { self.as_any().matrix_index(i.to_any()).into() }
}

impl<Cols: Len2, Rows: Len2, T: ScalarTypeFp> GetAllFields for mat<T, Cols, Rows> {
    fn fields_as_anys_unchecked(self_as_any: Any) -> impl std::borrow::Borrow<[Any]> { [] }
}
