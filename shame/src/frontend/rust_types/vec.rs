use super::{
    error::FrontendError,
    index::GpuIndex,
    ir,
    layout_traits::from_single_any,
    len::*,
    mem::AddressSpace,
    reference::{AccessMode, AccessModeReadable},
    scalar_type::{dtype_as_scalar_from_f64, ScalarType, ScalarTypeInteger, ScalarTypeNumber},
    type_layout::repr,
    type_traits::{BindingArgs, GpuAligned, GpuStoreImplCategory, NoAtomics, NoHandles, VertexAttribute},
    AsAny, GpuType, To, ToGpuType,
};
use crate::{
    any::{
        layout::{self, Layoutable, LayoutableSized},
        BufferBindingType,
    },
    call_info,
    common::{
        proc_macro_utils::{collect_into_array_exact, push_wrong_amount_of_args_error},
        small_vec::SmallVec,
    },
    frontend::encoding::{
        buffer::{BufferAddressSpace, BufferInner, BufferRefInner},
        rasterizer::Gradient,
    },
    ir::{
        pipeline::StageMask,
        recording::{CallInfoScope, Context, NodeRecordingError},
        Comp4, GradPrecision, VectorAccess,
    },
};

use super::{
    layout_traits::{CpuLayout, FromAnys, GetAllFields, GpuLayout},
    type_layout::TypeLayout,
    type_traits::{GpuSized, GpuStore, NoBools},
};
use crate::frontend::any::shared_io::{BindPath, BindingType};
use crate::frontend::rust_types::reference::Ref;
use crate::{
    common::floating_point::f16,
    frontend::any::{render_io::VertexAttribFormat, Any, InvalidReason},
};

use ir::VectorAccess::*;
use std::{borrow::Borrow, marker::PhantomData, ops::Deref};
use Comp4::*;

#[allow(non_camel_case_types)] // primitive type in shader
pub type scalar<T> = vec<T, x1>;

#[allow(non_camel_case_types)] // primitive type in shader
#[derive(Clone, Copy)]
/// an `L` dimensional scalar or vector.
///
/// where `L` is one of [`sm::x1`], [`sm::x2`], [`sm::x3`] or [`sm::x4`]
///
/// ## Memory Layout
/// the memory layout of `sm::vec` corresponds to the scalar and vector layouts
/// described in https://www.w3.org/TR/WGSL/#alignment-and-size .
///
/// ## Examples
/// ```
/// use shame as sm;
/// use sm::prelude::*; // trait functions for constructing `vec` from rust types
/// use sm::aliases::*; // type aliases like `f32x4`, `u32x1` etc.
///
/// let scalar: sm::vec<f32, x1> = 1.0.to_gpu(); // a scalar value (using the `sm::ToGpuType` trait)
///
/// let my_vec3 = sm::vec!(1.0, 2.0, 3.0);
/// let my_vec4 = sm::vec!(my_vec3, 0.0); // component concatenation, like usual in shaders
/// let my_vec4 = my_vec3.extend(0.0); // or like this
///
/// let my_normal = sm::vec!(1.0, 1.0, 0.0).normalize();
/// let rgb = my_normal.remap(-1.0..=1.0, 0.0..=1.0); // remap linear ranges (instead of " * 0.5 + 0.5")
///
/// let alpha = 0.4.to_gpu(); // convert from rust to `shame` types (also works for arrays and structs)
/// let smooth: f32x1 = alpha.smootcstep(0.4..0.8);
///
/// // clamp as generalized min, max, clamp via half open ranges
/// let upper = alpha.clamp(..=0.8);
/// let lower = alpha.clamp(0.1..);
/// let both  = alpha.clamp(0.1..=0.8);
///
/// // reverse subtraction
/// let k = (1.0 - (1.0 - alpha).sqrt());
/// let k = alpha.rsub(1.0).sqrt().rsub(1.0); // same as above
/// let k = alpha.rsub1().sqrt().rsub1(); // same as above
///
/// // iterate over components of vec
/// let sum: f32x1 = my_vec4.into_iter().map(|x| x * x).sum();
///
/// let linear = xform.resize() as f32x3x3; // generic matrix resize
/// let linear = linear * sm::mat::id(); // generic identity matrix
///
/// // generic zero, one
/// let z: f32x3 = sm::zero();
/// let z: f32x3 = sm::one();
///
/// // generic unit vectors for coordinate axes x, y, z, w
/// let to_light_3d: f32x3 = sm::vec::y();
/// let to_light_2d: f32x2 = sm::vec::y();
/// ```
///
/// [`sm::x1`]: crate::x1
/// [`sm::x2`]: crate::x2
/// [`sm::x3`]: crate::x3
/// [`sm::x4`]: crate::x4
pub struct vec<T: ScalarType, L: Len> {
    components: L::VecComponents<T>,
    any: Any,
}

pub mod macros {
    /// create a new [`vec`] from scalar components or by concatenating vectors/scalars
    ///
    /// examples:
    /// ```
    /// let v2 = vec!(1.0, 2.0);
    /// let v4 = vec!(0.0, v2, 3.0);
    /// let v4 = vec!(v2, v2);
    /// let v4 = vec!(v2.x, v2.y, v2);
    /// ```
    /// integer [`vec`] require verbose literals for now:
    /// ```
    /// let v3i = vec!(1_i32, 2_i32, 3_i32);
    /// let v3u = vec!(1_u32, 2_u32, 3_u32);
    /// let v4u = vec!(v3u.x, v3u);
    /// let v3u = vec!(v2.to_u32(), v3u.x); // cast scalar types via `.to_*`
    /// ```
    /// we are searching for a solution to make this less verbose
    ///
    /// [`vec`]: super::vec
    #[macro_export]
    macro_rules! vec {
        ($($components: expr),* $(,)?) => {
            $crate::vec::new(($($components),*))
        };
    }
    pub(crate) use vec;

    #[doc(hidden)] // unstable
    #[macro_export]
    macro_rules! vecu {
        //TODO(release) make this macro capable of vec concatenation, and remove doc(hidden) afterwards
        ($($components: expr),* $(,)?) => {
            $crate::vec::new(($($components as u32),*))
        };
    }
    pub(crate) use vecu;

    #[doc(hidden)] // unstable
    #[macro_export]
    macro_rules! veci {
        //TODO(release) make this macro capable of vec concatenation, and remove doc(hidden) afterwards
        ($($components: expr),* $(,)?) => {
            $crate::vec::new(($($components as i32),*))
        };
    }
    pub(crate) use veci;

    #[doc(hidden)] // unstable
    #[macro_export]
    macro_rules! vecf {
        //TODO(release) make this macro capable of vec concatenation, and remove doc(hidden) afterwards
        ($($components: expr),* $(,)?) => {
            $crate::vec::new(($($components as f32),*))
        };
    }
    pub(crate) use vecf;

    #[doc(hidden)] // unstable
    #[macro_export]
    macro_rules! vech {
        //TODO(release) make this macro capable of vec concatenation, and remove doc(hidden) afterwards
        ($($components: expr),* $(,)?) => {
            $crate::vec::new(($($crate::f16::from($components as f32)),*))
        };
    }
    pub(crate) use vech;
}

impl<T: ScalarType, L: Len> vec<T, L> {
    //[old-doc] ## creating vectors or scalars
    //[old-doc] In `shame` vectors and scalars are both represented through the `vec<T, L>` type.
    //[old-doc] Scalars are just one-dimensional vectors `scalar<T> = vec<T, x1>`.
    //[old-doc]
    //[old-doc] Vectors are initialized via tuples, where `vec::new` is equivalent to `.to_gpu()`
    //[old-doc] from the `ToGpuType` trait.
    //[old-doc] ```
    //[old-doc] let v = vec::new((1.0, 2.0));
    //[old-doc] let v = (1.0, 2.0).to_gpu();
    //[old-doc] let s = 1.0.to_gpu();
    //[old-doc] ```
    //[old-doc] however, `.to_gpu()` can convert many rust types to their `shame` equivalents,
    //[old-doc] not just vectors and scalars.
    //[old-doc]
    //[old-doc] ### combining multiple vectors/scalars to form larger vectors
    //[old-doc] like many shading languages, `shame` supports constructors that
    //[old-doc] combine vectors and scalars to larger vectors by concatenating their
    //[old-doc] components
    //[old-doc] ```
    //[old-doc] let v2 = (1.0, 2.0).to_gpu();
    //[old-doc] let v3 = (v2, 3.0).to_gpu();
    //[old-doc] let v4 = (v2, v2).to_gpu();
    //[old-doc] let v4 = (0.0, v2, 3.0).to_gpu();
    //[old-doc] ```
    //[old-doc]
    //[old-doc] ### zero/one initialization
    //[old-doc] to generically fill a `vec<T, L>` with zeroes use `zero()` or `vec::zero()`
    //[old-doc] ```
    //[old-doc] // vec `L` and `T` will be inferred, TODO(release) may support matrices in the future
    //[old-doc] let v0 = zero();
    //[old-doc] let v1 = one();
    //[old-doc]
    //[old-doc] // same as above but constrained to `vec`s only
    //[old-doc] let v0 = vec::zero();
    //[old-doc] let v1 = vec::one();
    //[old-doc]
    //[old-doc] // help the compiler figure out parts of the type if necessary
    //[old-doc] let v0 = vec::<i32, _>::zero();
    //[old-doc] let v0 = vec::<_, x4>::zero();
    //[old-doc] ```
    //[old-doc]
    /// (no documentation yet)
    #[track_caller]
    pub fn new<TupleOrVec>(init: TupleOrVec) -> Self
    where
        TupleOrVec: To<vec<T, L>>,
    {
        init.to_gpu()
    }
    /// (no documentation yet)
    #[track_caller]
    pub fn zero() -> Self { super::vec::zero() }
    /// (no documentation yet)
    #[track_caller]
    pub fn one() -> Self { super::vec::one() }

    #[track_caller]
    fn standard_basis_column(column: Comp4) -> Self {
        let new_literal = dtype_as_scalar_from_f64::<T>;

        let mut args = SmallVec::<Any, 4>::default();
        // non-functional style to not mess up `CallInfo` of `new_literal`
        for component in L::LEN.iter_components() {
            args.push(new_literal(if component == column { 1.0 } else { 0.0 }));
        }
        Any::new_vec(L::LEN, T::SCALAR_TYPE, &args).into()
    }

    /// A vector with all zeroes except the x component which is one.
    ///
    /// the x vector of the [standard basis](https://en.wikipedia.org/wiki/Standard_basis)
    ///
    /// depending on the dimensionality `L` of the vector returns:
    /// - 2D: (1, 0)
    /// - 3D: (1, 0, 0)
    /// - 4D: (1, 0, 0, 0)
    #[track_caller]
    pub fn x() -> Self
    where
        L: AtLeastLen<x2>,
    {
        Self::standard_basis_column(Comp4::X)
    }

    /// A vector with all zeroes except the y component which is one.
    ///
    /// the y vector of the [standard basis](https://en.wikipedia.org/wiki/Standard_basis)
    ///
    /// depending on the dimensionality `L` of the vector returns:
    /// - 2D: (0, 1)
    /// - 3D: (0, 1, 0)
    /// - 4D: (0, 1, 0, 0)
    #[track_caller]
    pub fn y() -> Self
    where
        L: AtLeastLen<x2>,
    {
        Self::standard_basis_column(Comp4::Y)
    }

    /// A vector with all zeroes except the z component which is one.
    ///
    /// the z vector of the [standard basis](https://en.wikipedia.org/wiki/Standard_basis)
    ///
    /// depending on the dimensionality `L` of the vector returns:
    /// - 3D: (0, 0, 1)
    /// - 4D: (0, 0, 1, 0)
    #[track_caller]
    pub fn z() -> Self
    where
        L: AtLeastLen<x3>,
    {
        Self::standard_basis_column(Comp4::Z)
    }

    /// the number of components in this vector.
    ///
    /// If you want the euclidean length instead, use `.length()`
    ///
    /// ### why i32?
    /// Returns an `i32` instead of `usize` because `i32` is the default
    /// type for indexing in `shame` (mostly due to the way rusts integer
    /// literals interact with type inference, otherwise doing basic indexing
    /// with constants would be much more verbose (e.g. require `as u32` `.to_gpu()` etc.)
    #[allow(clippy::len_without_is_empty)] // cannot be empty
    pub const fn len(&self) -> i32 { L::USIZE as i32 }

    /// (no documentation yet)
    #[track_caller]
    pub fn at(&self, i: impl ToInteger) -> vec<T, x1> { self.any.vector_index(i.to_any()).into() }

    /// (no documentation yet)
    #[track_caller]
    pub fn extend<Ext: Len>(&self, extension: impl To<vec<T, Ext>>) -> vec<T, <L as ExtendBy<Ext>>::Len>
    where
        L: ExtendBy<Ext>,
    {
        Any::new_vec(
            <L as ExtendBy<Ext>>::Len::LEN,
            T::SCALAR_TYPE,
            &[self.any, extension.to_any()],
        )
        .into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn xy(&self) -> vec<T, x2>
    where
        L: AtLeastLen<x2>,
    {
        self.any.swizzle(Swizzle2([X, Y])).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn yz(&self) -> vec<T, x2>
    where
        L: AtLeastLen<x3>,
    {
        self.any.swizzle(Swizzle2([Y, Z])).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn zw(&self) -> vec<T, x2>
    where
        L: AtLeastLen<x4>,
    {
        self.any.swizzle(Swizzle2([Z, W])).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn xyz(&self) -> vec<T, x3>
    where
        L: AtLeastLen<x3>,
    {
        self.any.swizzle(Swizzle3([X, Y, Z])).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn yzw(&self) -> vec<T, x3>
    where
        L: AtLeastLen<x4>,
    {
        self.any.swizzle(Swizzle3([Y, Z, W])).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn swizzle<const N: usize>(&self, comps: [Comp4; N]) -> vec<T, <() as Swizzle<N>>::Len>
    where
        (): Swizzle<N>,
    {
        match push_swizzle_error_if_needed(L::LEN, &comps) {
            Some(invalid_any) => invalid_any.into(),
            None => {
                let xyzw = <() as Swizzle<N>>::to_vector_access(comps);
                self.as_any().swizzle(xyzw).into()
            }
        }
    }
}

impl<T: ScalarType> vec<T, x2> {
    /// split vector into `[self.x, self.y]` (useful for pattern matching and `.map(_)`)
    pub fn comps(&self) -> [vec<T, x1>; 2] { [self.x, self.y] }

    /// split vector into `(self.x, self.y)` (useful for pattern matching)
    pub fn tuple(&self) -> (vec<T, x1>, vec<T, x1>) { (self.x, self.y) }
}

impl<T: ScalarType> vec<T, x3> {
    /// split vector into `[self.x, self.y, self.z]` (useful for pattern matching and `.map(_)`)
    pub fn array(&self) -> [vec<T, x1>; 3] { [self.x, self.y, self.z] }

    /// split vector into `(self.x, self.y, self.z)` (useful for pattern matching)
    pub fn tuple(&self) -> (vec<T, x1>, vec<T, x1>, vec<T, x1>) { (self.x, self.y, self.z) }
}

#[allow(clippy::type_complexity)]
impl<T: ScalarType> vec<T, x4> {
    /// split vector into `[self.x, self.y, self.z, self.w]` (useful for pattern matching and `.map(_)`)
    pub fn array(&self) -> [vec<T, x1>; 4] { [self.x, self.y, self.z, self.w] }

    /// split vector into `(self.x, self.y, self.z, self.w)` (useful for pattern matching)
    pub fn tuple(&self) -> (vec<T, x1>, vec<T, x1>, vec<T, x1>, vec<T, x1>) { (self.x, self.y, self.z, self.w) }

    /// split a 4 component vector into the xyz and the w component
    pub fn xyz_w(&self) -> (vec<T, x3>, vec<T, x1>) { (self.xyz(), self.w) }

    /// A vector with all zeroes except the w component which is one.
    ///
    /// the w vector of the [standard basis](https://en.wikipedia.org/wiki/Standard_basis)
    ///
    /// returns (0, 0, 0, 1)
    #[track_caller]
    pub fn w() -> Self { vec::standard_basis_column(Comp4::W) }
}

/// implemented only for [`vec`]
///
/// this trait is mostly a workaround since some rust trait features are not
/// stable yet, such as associated type defaults (see `TextureFormat`)
///
/// [`vec`]: crate::vec
// TODO(release) seal this trait
pub trait IsVec:
    GpuType + ToGpuType<Gpu = Self> + Into<vec<Self::T, Self::L>> + From<vec<Self::T, Self::L>> + Copy
{
    type L: Len;
    type T: ScalarType;
}

impl<T: ScalarType, L: Len> IsVec for vec<T, L> {
    type L = L;
    type T = T;
}

pub trait ToVec: ToGpuType<Gpu = vec<Self::T, Self::L>> {
    type L: Len;
    type T: ScalarType;
}

impl<V: ToGpuType<Gpu = vec<T, L>>, T: ScalarType, L: Len> ToVec for V {
    type L = L;
    type T = T;
}

/// convert rust literals/scalars ([`f32`], [`i32`], ...) into their respective
/// shame scalars ([`f32x1`], [`i32x1`], ..)
///
/// also implemented by shame scalars, which, when converted just stay scalars (identity).
///
/// this trait is implemented by
/// * [`f32`]
/// * not [`f64`], use `shame::f64x1::from(_)` instead
/// * [`u32`]
/// * [`i32`]
/// * [`bool`]
///
/// but also implemented by their corresponding shame scalar types:
/// * [`shame::f16`]
/// * [`f16x1`]
/// * [`f32x1`]
/// * [`f64x1`]
/// * [`i32x1`]
/// * [`u32x1`]
/// * [`boolx1`]
///
/// [`shame::f16`]: crate::f16
/// [`f16x1`]: crate::aliases::f16x1
/// [`f32x1`]: crate::aliases::f32x1
/// [`f64x1`]: crate::aliases::f64x1
/// [`i32x1`]: crate::aliases::i32x1
/// [`u32x1`]: crate::aliases::u32x1
/// [`boolx1`]: crate::aliases::boolx1
pub trait ToScalar: ToVec<L = x1> {
    /// creates a vector with `L` components, each of which are filled with the value of `self`
    ///
    /// ## Example
    /// ```
    /// use shame as sm;
    /// use sm::ToScalar as _; // or `use sm::prelude::*`
    ///
    /// let q: f32x3 = 3.0.splat();
    /// // same as
    /// let q: f32x3 = sm::vec!(3.0, 3.0, 3.0);
    ///
    /// let p: f32x4 = q.x.splat();
    /// // same as
    /// let p: f32x4 = sm::vec!(q.x, q.x, q.x, q.x);
    /// ```
    #[track_caller]
    fn splat<L: Len>(&self) -> vec<Self::T, L> {
        match L::LEN {
            ir::Len::X1 => self.to_any().into(), // x1 => x1 noop
            _ => self.to_any().splat(L::LEN, Self::T::SCALAR_TYPE).into(),
        }
    }
}
impl<V: ToGpuType<Gpu = vec<T, x1>>, T: ScalarType> ToScalar for V {}

/// a type that can be converted to [`vec<i32, x1>`] or [`vec<u32, x1>`] scalars
/// such as:
///
/// - `u32`
/// - `i32`
/// - [`vec<u32, x1>`]
/// - [`vec<i32, x1>`]
pub trait ToInteger: ToScalar {
    const SCALAR_TYPE_INTEGER: ir::ScalarTypeInteger;
}
impl ToInteger for crate::u32x1 {
    const SCALAR_TYPE_INTEGER: ir::ScalarTypeInteger = ir::ScalarTypeInteger::U32;
}
impl ToInteger for crate::i32x1 {
    const SCALAR_TYPE_INTEGER: ir::ScalarTypeInteger = ir::ScalarTypeInteger::I32;
}
impl ToInteger for u32 {
    const SCALAR_TYPE_INTEGER: ir::ScalarTypeInteger = ir::ScalarTypeInteger::U32;
}
impl ToInteger for i32 {
    const SCALAR_TYPE_INTEGER: ir::ScalarTypeInteger = ir::ScalarTypeInteger::I32;
}
impl ToInteger for crate::VertexIndex {
    const SCALAR_TYPE_INTEGER: ir::ScalarTypeInteger = ir::ScalarTypeInteger::U32;
}
//impl<V: ToVec<L = x1>> ToInteger for V where V::T: ScalarTypeInteger {}

impl<T: ScalarType, L: Len> Deref for vec<T, L> {
    type Target = L::VecComponents<T>;

    fn deref(&self) -> &Self::Target { &self.components }
}

impl<T: ScalarType, L: Len> GpuSized for vec<T, L> {
    fn sized_ty() -> ir::SizedType { ir::SizedType::Vector(L::LEN, T::SCALAR_TYPE) }
}

impl<T: ScalarType, L: Len> GpuAligned for vec<T, L> {
    fn aligned_ty() -> ir::AlignedType { ir::AlignedType::Sized(<Self as GpuSized>::sized_ty()) }
}

impl<T: ScalarType, L: Len> GpuStore for vec<T, L> {
    type RefFields<AS: AddressSpace, AM: AccessMode> = L::VecComponentsRef<T, AS, AM>;
    fn store_ty() -> ir::StoreType { ir::StoreType::Sized(<Self as GpuSized>::sized_ty()) }

    fn instantiate_buffer_inner<AS: BufferAddressSpace>(
        args: Result<BindingArgs, InvalidReason>,
        bind_ty: BufferBindingType,
        has_dynamic_offset: bool,
    ) -> BufferInner<Self, AS>
    where
        Self: NoAtomics + NoBools,
    {
        BufferInner::new_plain(args, bind_ty, has_dynamic_offset)
    }

    fn instantiate_buffer_ref_inner<AS: BufferAddressSpace, AM: AccessModeReadable>(
        args: Result<BindingArgs, InvalidReason>,
        bind_ty: BufferBindingType,
        has_dynamic_offset: bool,
    ) -> BufferRefInner<Self, AS, AM>
    where
        Self: NoBools,
    {
        BufferRefInner::new_plain(args, bind_ty, has_dynamic_offset)
    }

    fn impl_category() -> GpuStoreImplCategory { GpuStoreImplCategory::GpuType(Self::store_ty()) }
}


impl<T: ScalarType, L: Len> LayoutableSized for vec<T, L>
where
    vec<T, L>: NoBools,
{
    fn layoutable_type_sized() -> layout::SizedType {
        layout::Vector::new(T::SCALAR_TYPE.try_into().expect("no bools"), L::LEN).into()
    }
}
impl<T: ScalarType, L: Len> Layoutable for vec<T, L>
where
    vec<T, L>: NoBools,
{
    fn layoutable_type() -> layout::LayoutableType { Self::layoutable_type_sized().into() }
}

impl<T: ScalarType, L: Len> GpuLayout for vec<T, L>
where
    vec<T, L>: NoBools,
{
    type GpuRepr = repr::Storage;

    fn cpu_type_name_and_layout()
    -> Option<Result<(std::borrow::Cow<'static, str>, TypeLayout), super::layout_traits::ArrayElementsUnsizedError>>
    {
        None
    }
}

impl<T: ScalarType, L: Len> GpuType for vec<T, L> {
    fn ty() -> ir::Type { ir::Type::Store(Self::store_ty()) }

    #[track_caller]
    fn from_any_unchecked(any: Any) -> Self {
        Self {
            components: <_>::new(any),
            any,
        }
    }
}

impl<T: ScalarType, L: Len> AsAny for vec<T, L> {
    fn as_any(&self) -> Any { self.any }
}

impl<T: ScalarType, L: Len> ToGpuType for vec<T, L> {
    type Gpu = Self;



    fn to_gpu(&self) -> Self::Gpu { *self }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { Some(self) }
}

impl<T: ScalarType, L: Len> From<Any> for vec<T, L> {
    #[track_caller]
    fn from(any: Any) -> Self { GpuType::from_any(any) }
}

#[derive(Clone, Copy)]
/// empty deref type for the `Deref::Target` of single component `vec`.
/// Larger `vec` deref to `Xy`, `Xyz`, `Xyzw` depending on their `Len`
pub struct EmptyComponents<T: ScalarType> {
    pub phantom: PhantomData<T>,
}

#[derive(Clone, Copy)]
pub struct Xy<T: ScalarType> {
    pub x: vec<T, x1>,
    pub y: vec<T, x1>,
}

#[derive(Clone, Copy)]
pub struct Xyz<T: ScalarType> {
    pub x: vec<T, x1>,
    pub y: vec<T, x1>,
    pub z: vec<T, x1>,
    // pub xy: vec<T, x2>,
    // pub yz: vec<T, x2>,
}

#[derive(Clone, Copy)]
pub struct Xyzw<T: ScalarType> {
    pub x: vec<T, x1>,
    pub y: vec<T, x1>,
    pub z: vec<T, x1>,
    pub w: vec<T, x1>,
    // pub xy: vec<T, x2>,
    // pub yz: vec<T, x2>,
    // pub zw: vec<T, x2>,

    // pub xyz: vec<T, x3>,
    // pub yzw: vec<T, x3>,
}

#[derive(Clone, Copy)]
pub struct RefXy<T: ScalarType, AS: AddressSpace, AM: AccessMode> {
    pub x: Ref<vec<T, x1>, AS, AM>,
    pub y: Ref<vec<T, x1>, AS, AM>,
}

#[derive(Clone, Copy)]
pub struct RefXyz<T: ScalarType, AS: AddressSpace, AM: AccessMode> {
    pub x: Ref<vec<T, x1>, AS, AM>,
    pub y: Ref<vec<T, x1>, AS, AM>,
    pub z: Ref<vec<T, x1>, AS, AM>,
}

#[derive(Clone, Copy)]
pub struct RefXyzw<T: ScalarType, AS: AddressSpace, AM: AccessMode> {
    pub x: Ref<vec<T, x1>, AS, AM>,
    pub y: Ref<vec<T, x1>, AS, AM>,
    pub z: Ref<vec<T, x1>, AS, AM>,
    pub w: Ref<vec<T, x1>, AS, AM>,
}

pub(crate) trait Components: Copy {
    #[track_caller]
    fn new(parent: Any) -> Self;
}

impl<T: ScalarType> Components for EmptyComponents<T> {
    fn new(parent: Any) -> Self { EmptyComponents { phantom: PhantomData } }
}

impl<T: ScalarType> Components for Xy<T> {
    #[track_caller]
    fn new(any: Any) -> Self {
        match any.inner() {
            Ok(_) => Self {
                x: any.swizzle(Swizzle1([X])).into(),
                y: any.swizzle(Swizzle1([Y])).into(),
            },
            Err(reason) => Self {
                x: Any::new_invalid(reason).into(),
                y: Any::new_invalid(reason).into(),
            },
        }
    }
}

impl<T: ScalarType> Components for Xyz<T> {
    #[track_caller]
    fn new(any: Any) -> Self {
        match any.inner() {
            Ok(_) => Self {
                x: any.swizzle(Swizzle1([X])).into(),
                y: any.swizzle(Swizzle1([Y])).into(),
                z: any.swizzle(Swizzle1([Z])).into(),
                // xy: any.swizzle(Swizzle2([X, Y])).into(),
                // yz: any.swizzle(Swizzle2([Y, Z])).into(),
            },
            Err(reason) => Self {
                x: Any::new_invalid(reason).into(),
                y: Any::new_invalid(reason).into(),
                z: Any::new_invalid(reason).into(),
                // xy: Any::new_invalid(reason).into(),
                // yz: Any::new_invalid(reason).into(),
            },
        }
    }
}

impl<T: ScalarType> Components for Xyzw<T> {
    #[track_caller]
    fn new(any: Any) -> Self {
        use Comp4::*;
        match any.inner() {
            Ok(_) => Self {
                x: any.swizzle(Swizzle1([X])).into(),
                y: any.swizzle(Swizzle1([Y])).into(),
                z: any.swizzle(Swizzle1([Z])).into(),
                w: any.swizzle(Swizzle1([W])).into(),
                // xy: any.swizzle(Swizzle2([X, Y])).into(),
                // yz: any.swizzle(Swizzle2([Y, Z])).into(),
                // zw: any.swizzle(Swizzle2([Z, W])).into(),

                // xyz: any.swizzle(Swizzle3([X, Y, Z])).into(),
                // yzw: any.swizzle(Swizzle3([Y, Z, W])).into(),
            },
            Err(reason) => Self {
                x: Any::new_invalid(reason).into(),
                y: Any::new_invalid(reason).into(),
                z: Any::new_invalid(reason).into(),
                w: Any::new_invalid(reason).into(),
                // xy: Any::new_invalid(reason).into(),
                // yz: Any::new_invalid(reason).into(),
                // zw: Any::new_invalid(reason).into(),

                // xyz: Any::new_invalid(reason).into(),
                // yzw: Any::new_invalid(reason).into(),
            },
        }
    }
}

const XY_LEN: usize = 2;
impl<T: ScalarType, AS: AddressSpace, AM: AccessMode> FromAnys for RefXy<T, AS, AM> {
    fn expected_num_anys() -> usize { XY_LEN }

    #[track_caller]
    fn from_anys(anys: impl Iterator<Item = Any>) -> Self {
        const EXPECTED_LEN: usize = XY_LEN;
        let [x, y] = match collect_into_array_exact::<Any, EXPECTED_LEN>(anys) {
            Ok(t) => t,
            Err(actual_len) => {
                push_wrong_amount_of_args_error(actual_len, EXPECTED_LEN, call_info!());
                [Any::new_invalid(InvalidReason::ErrorThatWasPushed); EXPECTED_LEN]
            }
        };
        Self {
            x: From::from(x),
            y: From::from(y),
        }
    }
}

const XYZ_LEN: usize = 3;
impl<T: ScalarType, AS: AddressSpace, AM: AccessMode> FromAnys for RefXyz<T, AS, AM> {
    fn expected_num_anys() -> usize { XYZ_LEN }
    #[track_caller]
    fn from_anys(anys: impl Iterator<Item = Any>) -> Self {
        const EXPECTED_LEN: usize = XYZ_LEN;
        let [x, y, z] = match collect_into_array_exact::<Any, EXPECTED_LEN>(anys) {
            Ok(t) => t,
            Err(actual_len) => {
                push_wrong_amount_of_args_error(actual_len, EXPECTED_LEN, call_info!());
                [Any::new_invalid(InvalidReason::ErrorThatWasPushed); EXPECTED_LEN]
            }
        };
        Self {
            x: From::from(x),
            y: From::from(y),
            z: From::from(z),
        }
    }
}

const XYZW_LEN: usize = 4;
impl<T: ScalarType, AS: AddressSpace, AM: AccessMode> FromAnys for RefXyzw<T, AS, AM> {
    fn expected_num_anys() -> usize { XYZW_LEN }

    #[track_caller]
    fn from_anys(anys: impl Iterator<Item = Any>) -> Self {
        const EXPECTED_LEN: usize = XYZW_LEN;
        let [x, y, z, w] = match collect_into_array_exact::<Any, EXPECTED_LEN>(anys) {
            Ok(t) => t,
            Err(actual_len) => {
                push_wrong_amount_of_args_error(actual_len, EXPECTED_LEN, call_info!());
                [Any::new_invalid(InvalidReason::ErrorThatWasPushed); EXPECTED_LEN]
            }
        };
        Self {
            x: From::from(x),
            y: From::from(y),
            z: From::from(z),
            w: From::from(w),
        }
    }
}

impl<T: ScalarType, L: Len> vec<T, L> {
    /// (no documentation yet)
    pub fn to_f16(self) -> vec<f16, L> { self.into_generic() }
    /// (no documentation yet)
    pub fn to_f32(self) -> vec<f32, L> { self.into_generic() }
    /// (no documentation yet)
    pub fn to_f64(self) -> vec<f64, L> { self.into_generic() }
    /// (no documentation yet)
    pub fn to_u32(self) -> vec<u32, L> { self.into_generic() }
    /// (no documentation yet)
    pub fn to_i32(self) -> vec<i32, L> { self.into_generic() }
    /// (no documentation yet)
    pub fn to_bool(self) -> vec<bool, L> { self.into_generic() }

    /// identical to `Into::into(_)` except that it is also usable in generic
    /// contexts. If unsure use `Into::into()` instead of this.
    ///
    /// ## why does this exist? why not use `Into::into`
    /// implementing `vec` conversion from any `ScalarType` to any other
    /// `ScalarType` generically via the `From` trait is impossible
    /// because rusts `core` implementation of `From<T> for T` blocks this more
    /// generic conversion. Therefore the `From` implementation is split up into
    /// all the specific permutations.
    #[track_caller]
    pub fn into_generic<T1: ScalarType>(self) -> vec<T1, L> { vec::<T1, L>::from_generic(self) }

    /// identical to `From::from()` except that it is also usable in generic
    /// contexts. If unsure use `From::from()` instead of this.
    ///
    /// ## why does this exist? why not use `From::from`
    /// implementing `vec` conversion from any `ScalarType` to any other
    /// `ScalarType` generically via the `From` trait is impossible
    /// because rusts `core` implementation of `From<T> for T` blocks this more
    /// generic conversion. Therefore the `From` implementation is split up into
    /// all the specific permutations.
    #[track_caller]
    pub fn from_generic<T1: ScalarType>(value: vec<T1, L>) -> vec<T, L> {
        Any::new_vec(L::LEN, T::SCALAR_TYPE, &[value.any]).into()
    }
}

macro_rules! impl_from {
    (
        $(From<vec<T, L>> for vec<$t: ty, L> where T in [$($t_from: ty),*];)*
    ) => {$($(
        impl<L: Len> From<vec<$t_from, L>> for vec<$t, L> {
            fn from(other: vec<$t_from, L>) -> Self {
                Self::from_generic(other)
            }
        }
    )*)*};
}


#[rustfmt::skip]
impl_from! {
    // this is necessary to avoid colliding implementations with rusts `core` `impl From<T> for T`
    // use `from_generic` and `into_generic` if you are in a generic context instead.
    From<vec<T, L>> for vec<f16, L> where T in [       f32, f64, u32, i32, bool];
    From<vec<T, L>> for vec<f32, L> where T in [  f16,      f64, u32, i32, bool];
    From<vec<T, L>> for vec<f64, L> where T in [  f16, f32,      u32, i32, bool];
    From<vec<T, L>> for vec<u32, L> where T in [  f16, f32, f64,      i32, bool];
    From<vec<T, L>> for vec<i32, L> where T in [  f16, f32, f64, u32,      bool];
    From<vec<T, L>> for vec<bool, L> where T in [  f16, f32, f64, u32, i32      ];
}

/// (no documentation yet)
#[track_caller]
pub fn zero<T: ScalarType, L: Len>() -> vec<T, L> {
    dtype_as_scalar_from_f64::<T>(0.0).splat(L::LEN, T::SCALAR_TYPE).into()
}

/// (no documentation yet)
#[track_caller]
pub fn one<T: ScalarType, L: Len>() -> vec<T, L> {
    dtype_as_scalar_from_f64::<T>(1.0).splat(L::LEN, T::SCALAR_TYPE).into()
}

/// (no documentation yet)
#[derive(Clone)]
pub struct ComponentIter<T: ScalarType> {
    components: std::slice::Iter<'static, ir::Comp4>,
    any: Any,
    phantom: PhantomData<T>,
}

impl<T: ScalarType> Iterator for ComponentIter<T> {
    type Item = vec<T, x1>;

    #[track_caller]
    fn next(&mut self) -> Option<Self::Item> { Some(self.any.get_component(*self.components.next()?).into()) }
}

impl<T: ScalarType, L: Len> IntoIterator for vec<T, L> {
    type Item = vec<T, x1>;

    type IntoIter = ComponentIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        use ir::Comp4::*;
        const COMPS: &[ir::Comp4] = &[X, Y, Z, W];
        ComponentIter {
            components: COMPS[0..L::USIZE].iter(),
            any: self.any,
            phantom: PhantomData,
        }
    }
}

impl<T: ScalarTypeNumber, L: Len> std::iter::Sum for vec<T, L> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self { iter.reduce(|l, r| l + r).unwrap_or_else(|| vec::zero()) }
}

impl<T: ScalarTypeNumber> std::iter::Product for vec<T, x1> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self { iter.reduce(|l, r| l * r).unwrap_or_else(|| vec::zero()) }
}

impl<T: ScalarType, L: Len, Idx: ToInteger> GpuIndex<Idx> for vec<T, L> {
    type Output = vec<T, x1>;

    fn index(&self, index: Idx) -> Self::Output { self.any.vector_index(index.to_any()).into() }
}

impl<T: ScalarType, L: Len> Default for vec<T, L> {
    #[track_caller]
    fn default() -> Self {
        Self::from(
            Context::try_with(call_info!(), |_| Any::new_vec(L::LEN, T::SCALAR_TYPE, &[]))
                .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding)),
        )
    }
}

impl<T: ScalarType, L: Len> NoHandles for vec<T, L> {}
impl<T: ScalarType, L: Len> NoAtomics for vec<T, L> {}
impl<L: Len> NoBools for vec<f16, L> {}
impl<L: Len> NoBools for vec<f32, L> {}
impl<L: Len> NoBools for vec<f64, L> {}
impl<L: Len> NoBools for vec<u32, L> {}
impl<L: Len> NoBools for vec<i32, L> {}

impl<T: ScalarType, L: Len, AS: AddressSpace, AM: AccessMode, Idx: ToInteger> GpuIndex<Idx> for Ref<vec<T, L>, AS, AM> {
    type Output = Ref<vec<T, x1>, AS, AM>;

    fn index(&self, i: Idx) -> Ref<vec<T, x1>, AS, AM> { self.as_any().vector_index(i.to_any()).into() }
}

impl<T: ScalarType, L: Len, AS: AddressSpace, AM: AccessMode> Ref<vec<T, L>, AS, AM> {
    /// (no documentation yet)
    #[track_caller]
    pub fn at(&self, i: impl ToInteger) -> Ref<vec<T, x1>, AS, AM> {
        match L::LEN {
            ir::Len::X1 => self.as_any().into(),
            _ => self.as_any().vector_index(i.to_any()).into(),
        }
    }

    // 2 component swizzle

    /// (no documentation yet)
    #[track_caller]
    pub fn xy(&self) -> Ref<vec<T, x2>, AS, AM>
    where
        L: AtLeastLen<x2>,
    {
        self.as_any().swizzle(Swizzle2([X, Y])).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn yz(&self) -> Ref<vec<T, x2>, AS, AM>
    where
        L: AtLeastLen<x3>,
    {
        self.as_any().swizzle(Swizzle2([Y, Z])).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn zw(&self) -> Ref<vec<T, x2>, AS, AM> { self.as_any().swizzle(Swizzle2([Z, W])).into() }

    // 3 component swizzle

    /// (no documentation yet)
    #[track_caller]
    pub fn xyz(&self) -> Ref<vec<T, x3>, AS, AM>
    where
        L: AtLeastLen<x3>,
    {
        self.as_any().swizzle(Swizzle3([X, Y, Z])).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn yzw(&self) -> Ref<vec<T, x3>, AS, AM>
    where
        L: AtLeastLen<x4>,
    {
        self.as_any().swizzle(Swizzle3([Y, Z, W])).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn swizzle<const N: usize>(&self, comps: [Comp4; N]) -> Ref<vec<T, <() as Swizzle<N>>::Len>, AS, AM>
    where
        (): Swizzle<N>,
    {
        match push_swizzle_error_if_needed(L::LEN, &comps) {
            Some(invalid_any) => invalid_any.into(),
            None => {
                let xyzw = <() as Swizzle<N>>::to_vector_access(comps);
                self.as_any().swizzle(xyzw).into()
            }
        }
    }
}

#[track_caller]
/// returns an invalid `Any` if an error was pushed
fn push_swizzle_error_if_needed(len: ir::Len, comps: &[Comp4]) -> Option<Any> {
    let call_info = call_info!();
    comps.iter().find(|c| !c.is_contained_in(len)).map(|c| {
        Context::try_with(call_info!(), |ctx| {
            ctx.push_error_get_invalid_any(
                NodeRecordingError::InvalidSwizzleComponent {
                    len: u64::from(len) as u8,
                    component: c.as_str(),
                }
                .into(),
            )
        })
        .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
    })
}

pub trait Swizzle<const N: usize> {
    type Len: Len;

    fn to_vector_access(comps: [Comp4; N]) -> VectorAccess;
}

impl Swizzle<1> for () {
    type Len = x1;
    fn to_vector_access(comps: [Comp4; 1]) -> VectorAccess { VectorAccess::Swizzle1(comps) }
}
impl Swizzle<2> for () {
    type Len = x2;
    fn to_vector_access(comps: [Comp4; 2]) -> VectorAccess { VectorAccess::Swizzle2(comps) }
}
impl Swizzle<3> for () {
    type Len = x3;
    fn to_vector_access(comps: [Comp4; 3]) -> VectorAccess { VectorAccess::Swizzle3(comps) }
}
impl Swizzle<4> for () {
    type Len = x4;
    fn to_vector_access(comps: [Comp4; 4]) -> VectorAccess { VectorAccess::Swizzle4(comps) }
}

impl<T: ScalarType, L: Len> GetAllFields for vec<T, L> {
    #[rustfmt::skip]
    fn fields_as_anys_unchecked(any: Any) -> impl std::borrow::Borrow<[Any]> {
        use crate::common::small_vec::SmallVec;
        use Comp4::*;
        let swizzle = |c: Comp4| any.swizzle(VectorAccess::Swizzle1([c]));

        let smallvec: SmallVec<_, 4> = match L::LEN {
            ir::Len::X1 => [].as_slice().into(), // empty, instead of [X].map(swizzle), because a scalar has no .x component
            ir::Len::X2 => [X, Y]      .map(swizzle).as_slice().into(),
            ir::Len::X3 => [X, Y, Z]   .map(swizzle).as_slice().into(),
            ir::Len::X4 => [X, Y, Z, W].map(swizzle).as_slice().into(),
        };

        smallvec
    }
}

impl<T: ScalarType, L: Len> VertexAttribute for vec<T, L>
where
    Self: NoBools,
{
    fn vertex_attrib_format() -> VertexAttribFormat {
        VertexAttribFormat::Fine(L::LEN, T::SCALAR_TYPE.try_into().expect("no bools vec"))
    }
}

impl<T: ScalarType, L: Len> FromAnys for vec<T, L> {
    #[track_caller]
    fn from_anys(anys: impl Iterator<Item = Any>) -> Self { from_single_any(anys).into() }

    fn expected_num_anys() -> usize { 1 }
}
