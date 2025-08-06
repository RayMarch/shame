use super::atomic::Atomic;
use super::index::GpuIndex;
use super::layout_traits::{ArrayElementsUnsizedError, FromAnys, GetAllFields, GpuLayout};
use super::len::x1;
use super::mem::AddressSpace;
use super::reference::{AccessMode, AccessModeReadable, AccessModeWritable, Read};
use super::scalar_type::ScalarTypeInteger;
use super::type_layout::{self, recipe, TypeLayout, ArrayLayout};
use super::type_traits::{
    BindingArgs, EmptyRefFields, GpuAligned, GpuSized, GpuStore, GpuStoreImplCategory, NoAtomics, NoBools, NoHandles,
};
use super::vec::{ToInteger, ToVec};
use super::{AsAny, GpuType};
use super::{To, ToGpuType};
use crate::common::small_vec::SmallVec;
use crate::frontend::any::shared_io::{BindPath, BindingType};
use crate::frontend::any::Any;
use crate::frontend::any::InvalidReason;
use crate::frontend::encoding::buffer::{Buffer, BufferAddressSpace, BufferInner, BufferRef, BufferRefInner};
use crate::frontend::encoding::flow::{for_range_impl, FlowFn};
use crate::frontend::error::InternalError;
use crate::frontend::rust_types::reference::Ref;
use crate::frontend::rust_types::vec::vec;
use crate::ir::ir_type::stride_of_array_from_element_align_size;
use crate::ir::pipeline::StageMask;
use crate::ir::recording::Context;
use crate::{call_info, for_count, ir};
use std::borrow::Cow;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::rc::Rc;

/// Amount of elements in an [`Array`]
///
/// implemented by
/// - [`Size<N>`] for arrays with statically known non-zero length
/// - [`RuntimeSize`] for arrays with unknown length before shader runtime
pub trait ArrayLen: Copy {
    /// the amount of elements, or `None` if `Self` is [`RuntimeSize`]
    const LEN: Option<NonZeroU32>;
}

/// for arrays with unknown element count before shader runtime.
///
/// Arrays which use this length are not [`GpuSized`]
///
/// May be used in [`Buffer`] or [`BufferRef`] bindings
///
/// see https://www.w3.org/TR/WGSL/#array-types
///
/// [`Buffer`]: crate::Buffer
/// [`BufferRef`]: crate::BufferRef
/// [`GpuSized`]: crate::GpuSized
#[derive(Clone, Copy)]
pub struct RuntimeSize;
impl ArrayLen for RuntimeSize {
    const LEN: Option<NonZeroU32> = None;
}

/// for arrays with known element count at compile time
#[derive(Clone, Copy)]
pub struct Size<const N: usize>; // usize to support conversion from [T; N]
impl<const N: usize> ArrayLen for Size<N> {
    const LEN: Option<NonZeroU32> = Some(
        match NonZeroU32::new(match N {
            0..=4294967295 => N as u32,
            _ => panic!("N must be between 0 and u32::MAX"),
        }) {
            Some(n) => n,
            None => panic!("N must be > 0"),
        },
    );
}

// ## Array (compile time or runtime sized)
//
// | rust     | shame                             | `GpuSized`
// |----------|-----------------------------------|--------------
// | `[T]`    | `shame::Array<T>`                 | no
// | `[T; N]` | `shame::Array<T, shame::Size<N>>` | yes
// |          |                                   |
/// (no documentation yet)
#[derive(Clone, Copy)]
pub struct Array<T: GpuType + GpuSized, N: ArrayLen = RuntimeSize> {
    any: Any,
    phantom: PhantomData<(T, N)>,
}

impl<T: GpuType + GpuStore + GpuSized, N: ArrayLen> GpuType for Array<T, N> {
    fn ty() -> ir::Type { ir::Type::Store(Self::store_ty()) }

    fn from_any_unchecked(any: Any) -> Self {
        Self {
            any,
            phantom: PhantomData,
        }
    }
}

impl<T: GpuType + GpuSized, N: ArrayLen> AsAny for Array<T, N> {
    fn as_any(&self) -> Any { self.any }
}

impl<const N: usize> Size<N> {
    pub(crate) const fn nonzero() -> NonZeroU32 {
        <Self as ArrayLen>::LEN.expect("nonzero validated at build time by impl ArrayLen for Size<N>")
    }
}

impl<T: GpuType + GpuSized + GpuStore, N: ArrayLen> GpuStore for Array<T, N> {
    type RefFields<AS: AddressSpace, AM: AccessMode> = EmptyRefFields;
    fn store_ty() -> ir::StoreType { Self::array_store_ty() }

    fn instantiate_buffer_inner<AS: BufferAddressSpace>(
        args: Result<BindingArgs, InvalidReason>,
        bind_ty: BindingType,
    ) -> BufferInner<Self, AS>
    where
        Self: NoAtomics + NoBools,
    {
        BufferInner::new_array(args, bind_ty)
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

impl<T: GpuType + GpuSized, N: ArrayLen> GpuAligned for Array<T, N> {
    fn aligned_ty() -> ir::AlignedType {
        match N::LEN {
            Some(len) => ir::AlignedType::Sized(ir::SizedType::Array(Rc::new(T::sized_ty()), len)),
            None => ir::AlignedType::RuntimeSizedArray(T::sized_ty()),
        }
    }
}

impl<T: GpuType + GpuSized, const N: usize> GpuSized for Array<T, Size<N>> {
    fn sized_ty() -> ir::SizedType {
        ir::SizedType::Array(
            Rc::new(T::sized_ty()),
            Size::<N>::LEN.expect("known length at compile time"),
        )
    }
}

#[rustfmt::skip] impl<T: GpuType + GpuSized + NoHandles, N: ArrayLen> NoHandles for Array<T, N> {}
#[rustfmt::skip] impl<T: GpuType + GpuSized + NoAtomics, N: ArrayLen> NoAtomics for Array<T, N> {}
#[rustfmt::skip] impl<T: GpuType + GpuSized + NoBools  , N: ArrayLen> NoBools   for Array<T, N> {}

impl<T: GpuType + GpuStore + GpuSized, N: ArrayLen> ToGpuType for Array<T, N> {
    type Gpu = Self;

    fn to_gpu(&self) -> Self::Gpu { self.clone() }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { Some(self) }
}

impl<T: GpuType + GpuSized + GpuLayout, N: ArrayLen> GpuLayout for Array<T, N> {
    fn layout_recipe() -> recipe::TypeLayoutRecipe {
        match N::LEN {
            Some(n) => recipe::SizedArray::new(Rc::new(T::layout_recipe_sized()), n).into(),
            None => recipe::RuntimeSizedArray::new(T::layout_recipe_sized()).into(),
        }
    }

    fn cpu_type_name_and_layout() -> Option<Result<(Cow<'static, str>, TypeLayout), ArrayElementsUnsizedError>> {
        let (t_cpu_name, t_cpu_layout) = match T::cpu_type_name_and_layout()? {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        let t_cpu_size = match t_cpu_layout.byte_size() {
            Some(size) => size,
            None => return Some(Err(ArrayElementsUnsizedError { elements: t_cpu_layout })),
        };
        let name = match N::LEN {
            Some(n) => format!("[{t_cpu_name}; {n}]"),
            None => format!("[{t_cpu_name}]"),
        };

        let result = (
            name.into(),
            ArrayLayout {
                byte_size: N::LEN.map(|n| n.get() as u64 * t_cpu_size),
                align: t_cpu_layout.align().into(),
                // array stride is element size according to
                // https://doc.rust-lang.org/reference/type-layout.html#r-layout.properties.size
                byte_stride: t_cpu_size,
                element_ty: t_cpu_layout,
                len: N::LEN.map(NonZeroU32::get),
            }
            .into(),
        );

        Some(Ok(result))

        // old comment below, TODO(release) check if still relevant or can be deleted.
        //
        // implementing this requires knowledge about size and alignment of TypeLayout on the rust side,
        // which is currently not part of `TypeLayout`. if `byte_size` and `align` information is stored
        // explicitly in `TypeLayout` then shame::Array::cpu_type_name_and_layout can be implemented.
        //
        // a specific example that is currently not possible to decide is:
        // Array<vec<f32, x3>>. It is unclear if the rust type used for vec3 by the user has a 16 byte alignment
        // and therefore (according to repr(C) rules) a 16 byte size.
        // whether the cpu type used for vec3f uses 4 byte alignment or 16 byte is unclear, and therefore
        // the array stride is unclear, unlike shame(=WGSL) where the stride is round_up(16, 12) = 16.
    }
}

impl<T: GpuType + GpuSized, N: ArrayLen> FromAnys for Array<T, N> {
    fn expected_num_anys() -> usize { 1 }

    #[track_caller]
    fn from_anys(mut anys: impl Iterator<Item = Any>) -> Self { super::layout_traits::from_single_any(anys).into() }
}

impl<T: GpuType + GpuSized, N: ArrayLen> Array<T, N> {
    fn array_store_ty() -> ir::StoreType {
        let element_ty = <T as GpuSized>::sized_ty();
        match N::LEN {
            Some(n) => ir::StoreType::Sized(ir::SizedType::Array(Rc::new(element_ty), n)),
            None => ir::StoreType::RuntimeSizedArray(element_ty),
        }
    }
}

impl<T: GpuType + GpuSized, N: ArrayLen> From<Any> for Array<T, N> {
    fn from(any: Any) -> Self {
        super::typecheck_downcast(any, ir::Type::Store(Self::array_store_ty()), |any| Self {
            any,
            phantom: PhantomData,
        })
    }
}

impl<T: GpuSized + GpuType + NoAtomics, N: ArrayLen> Array<T, N> {
    /// (no documentation yet)
    #[track_caller]
    pub fn at(&self, index: impl ToInteger) -> T { self.any.array_index(index.to_any()).into() }

    /// (no documentation yet)
    #[track_caller]
    pub fn len(&self) -> vec<u32, x1> {
        match N::LEN {
            Some(len) => len.get().to_gpu(),
            None => self.as_any().address().array_length().into(),
        }
    }
}

impl<Idx: ToInteger, T: GpuType + GpuSized + NoAtomics, N: ArrayLen> GpuIndex<Idx> for Array<T, N> {
    type Output = T;

    fn index(&self, index: Idx) -> T { self.any.array_index(index.to_any()).into() }
}

impl<Idx, T, AS, AM, N> GpuIndex<Idx> for Ref<Array<T, N>, AS, AM>
where
    Idx: ToInteger,
    T: GpuType + GpuSized + GpuStore + 'static,
    AS: AddressSpace + 'static,
    AM: AccessModeReadable + 'static,
    N: ArrayLen,
{
    type Output = Ref<T, AS, AM>;

    #[track_caller]
    fn index(&self, index: Idx) -> Ref<T, AS, AM> { self.at(index) }
}

impl<T: ToGpuType, const N: usize> ToGpuType for [T; N]
where
    T::Gpu: GpuStore + GpuSized,
{
    type Gpu = Array<T::Gpu, Size<N>>;

    fn to_gpu(&self) -> Self::Gpu {
        let anys: [Any; N] = std::array::from_fn(|i| self[i].to_any());
        Any::new_array(Rc::new(T::Gpu::sized_ty()), &anys).into()
    }

    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { None }
}

impl<T: GpuType + GpuStore + GpuSized + NoAtomics, const N: usize> Array<T, Size<N>> {
    /// (no documentation yet)
    #[track_caller]
    pub fn new(fields: [impl To<T>; N]) -> Self { fields.to_gpu() }

    /// zero initialize, see https://www.w3.org/TR/WGSL/#zero-value-builtin-function
    #[track_caller]
    pub fn zero() -> Self { Any::new_default(Self::sized_ty()).into() }

    /// create a new array by transforming the elements of `self`
    ///
    /// like [`[T; N]::map`]
    #[track_caller]
    pub fn map<R>(self, f: impl FnOnce(T) -> R + FlowFn) -> Array<R, Size<N>>
    where
        T: 'static,
        R: GpuType + GpuStore + GpuSized + NoAtomics + 'static,
    {
        let result = Array::<R, _>::zero().cell();
        for_range_impl(0..N as u32, move |i| {
            result.at(i).set(f(self.at(i)));
        });
        result.get()
    }
}

impl<T: GpuType + GpuSized, N: ArrayLen> GetAllFields for Array<T, N> {
    fn fields_as_anys_unchecked(self_as_any: Any) -> impl std::borrow::Borrow<[Any]> { [] }
}

#[track_caller]
fn push_buffer_of_array_has_wrong_variant_error(is_ref: bool, expected_variant: &str) -> Any {
    Context::try_with(call_info!(), |ctx| {
        ctx.push_error_get_invalid_any(
            InternalError::new(
                true,
                format!(
                    "Buffer{} inner is not `{expected_variant}` during indexing.",
                    if is_ref { "Ref" } else { "" }
                ),
            )
            .into(),
        )
    })
    .unwrap_or(Any::new_invalid(
        crate::frontend::any::InvalidReason::CreatedWithNoActiveEncoding,
    ))
}

impl<Idx, T, AS, const DYN_OFFSET: bool> GpuIndex<Idx> for Buffer<Array<T>, AS, DYN_OFFSET>
where
    Idx: ToInteger,
    T: GpuType + GpuSized + NoAtomics + NoHandles + GpuStore + 'static,
    Array<T>: GpuStore + NoAtomics + NoBools + NoHandles,
    AS: BufferAddressSpace + 'static,
{
    type Output = T;

    #[track_caller]
    fn index(&self, index: Idx) -> T {
        if let BufferInner::RuntimeSizedArray(arr) = &self.inner {
            arr.at(index).get()
        } else {
            push_buffer_of_array_has_wrong_variant_error(false, std::stringify!(BufferInner::RuntimeSizedArray)).into()
        }
    }
}

impl<Idx, T, AS, AM, const DYN_OFFSET: bool> GpuIndex<Idx> for BufferRef<Array<T>, AS, AM, DYN_OFFSET>
where
    Idx: ToInteger,
    T: GpuType + GpuSized + GpuStore + 'static,
    Array<T>: GpuStore + NoBools + NoHandles,
    AS: BufferAddressSpace + 'static,
    AM: AccessModeReadable + 'static,
{
    type Output = Ref<T, AS, AM>;

    #[track_caller]
    fn index(&self, index: Idx) -> Ref<T, AS, AM> {
        if let BufferRefInner::Plain(arr) = &self.inner {
            arr.at(index)
        } else {
            push_buffer_of_array_has_wrong_variant_error(true, std::stringify!(BufferRefInner::Plain)).into()
        }
    }
}

impl<T: GpuType + GpuSized, AS: BufferAddressSpace + 'static, const DYN_OFFSET: bool> Buffer<Array<T>, AS, DYN_OFFSET>
where
    Array<T>: GpuStore + NoAtomics + NoBools + NoHandles,
    T: GpuType + GpuSized + NoAtomics + NoHandles + GpuStore + 'static,
{
    /// (no documentation yet)
    #[track_caller]
    pub fn len(&self) -> vec<u32, x1> {
        if let BufferInner::RuntimeSizedArray(arr) = &self.inner {
            arr.as_any().address().array_length().into()
        } else {
            push_buffer_of_array_has_wrong_variant_error(false, std::stringify!(BufferInner::RuntimeSizedArray)).into()
        }
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn at(&self, index: impl ToInteger) -> T {
        if let BufferInner::RuntimeSizedArray(arr) = &self.inner {
            arr.at(index).get()
        } else {
            push_buffer_of_array_has_wrong_variant_error(false, std::stringify!(BufferInner::RuntimeSizedArray)).into()
        }
    }
}

impl<T, AS, AM, const DYN_OFFSET: bool> BufferRef<Array<T>, AS, AM, DYN_OFFSET>
where
    Array<T>: GpuStore + NoBools + NoHandles,
    T: GpuStore + GpuType + GpuSized + 'static,
    AS: BufferAddressSpace + 'static,
    AM: AccessModeReadable + 'static,
{
    /// (no documentation yet)
    #[track_caller]
    pub fn len(&self) -> vec<u32, x1> {
        if let BufferRefInner::Plain(arr) = &self.inner {
            arr.as_any().address().array_length().into()
        } else {
            push_buffer_of_array_has_wrong_variant_error(false, std::stringify!(BufferRefInner::Plain)).into()
        }
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn at(&self, index: impl ToInteger) -> Ref<T, AS, AM> {
        if let BufferRefInner::Plain(arr) = &self.inner {
            arr.at(index)
        } else {
            push_buffer_of_array_has_wrong_variant_error(true, std::stringify!(BufferRefInner::Plain)).into()
        }
    }
}

#[diagnostic::on_unimplemented(message = "array size <= 8 is required")]
pub trait UpTo8 {}
impl UpTo8 for Size<1> {}
impl UpTo8 for Size<2> {}
impl UpTo8 for Size<3> {}
impl UpTo8 for Size<4> {}
impl UpTo8 for Size<5> {}
impl UpTo8 for Size<6> {}
impl UpTo8 for Size<7> {}
impl UpTo8 for Size<8> {}
