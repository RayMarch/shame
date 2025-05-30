use std::marker::PhantomData;

use super::{
    layout_traits::{ArrayElementsUnsizedError, FromAnys, GetAllFields, GpuLayout},
    len::x1,
    mem::{AddressSpace, AddressSpaceAtomic},
    reference::{AccessMode, AccessModeReadable, ReadWrite},
    scalar_type::{ScalarType, ScalarTypeInteger},
    type_layout::{
        self,
        layoutable::{self, LayoutableSized},
        repr, TypeLayout,
    },
    type_traits::{
        BindingArgs, EmptyRefFields, GpuAligned, GpuSized, GpuStore, GpuStoreImplCategory, NoAtomics, NoBools,
        NoHandles,
    },
    vec::vec,
    AsAny, GpuType, To, ToGpuType,
};
use crate::{
    any::{layout::Layoutable, BufferBindingType},
    frontend::rust_types::reference::Ref,
};
use crate::{
    boolx1,
    frontend::{
        any::{
            shared_io::{BindPath, BindingType},
            Any, InvalidReason,
        },
        encoding::buffer::{BufferAddressSpace, BufferInner, BufferRefInner},
    },
    ir::{self, pipeline::StageMask, recording::AtomicCompareExchangeWeakGenerics},
};

/// see https://www.w3.org/TR/WGSL/#atomic-types
///
/// [`Atomic<T>`] is the type of a memory cell, not a value. Therefore it is only accessible through references.
///
/// [`Atomic<T>`] can be used via:
/// - storage buffer bindings: [`BufferRef<_, mem::Storage>`]
/// - workgroup memory allocations
///
/// [`BufferRef<_, mem::Storage>`]: crate::BufferRef<_, crate::mem::Storage>
/// [`BufferRef`]: crate::BufferRef
/// [`mem`]: crate::mem
/// (no documentation yet)
#[derive(Clone, Copy)]
pub struct Atomic<T: ScalarTypeInteger> {
    any: Any,
    phantom: PhantomData<T>,
}

/// see [`Atomic`]
pub type AtomicU32 = Atomic<u32>;
/// see [`Atomic`]
pub type AtomicI32 = Atomic<i32>;

impl<T: ScalarTypeInteger> GpuType for Atomic<T> {
    fn ty() -> ir::Type { ir::Type::Store(Self::store_ty()) }

    fn from_any_unchecked(any: Any) -> Self {
        Atomic {
            any,
            phantom: PhantomData,
        }
    }
}

impl<T: ScalarTypeInteger> ToGpuType for Atomic<T> {
    type Gpu = Self;
    fn to_gpu(&self) -> Self::Gpu { *self }
    fn to_any(&self) -> Any { self.any }
    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { Some(self) }
}

impl<T: ScalarTypeInteger> GpuSized for Atomic<T> {
    fn sized_ty() -> ir::SizedType { ir::SizedType::Atomic(T::SCALAR_TYPE_INTEGER) }
}

impl<T: ScalarTypeInteger> GpuAligned for Atomic<T> {
    fn aligned_ty() -> ir::AlignedType { ir::AlignedType::Sized(<Self as GpuSized>::sized_ty()) }
}

impl<T: ScalarTypeInteger> GpuStore for Atomic<T> {
    type RefFields<AS: AddressSpace, AM: AccessMode> = EmptyRefFields;
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

impl<T: ScalarTypeInteger> NoBools for Atomic<T> {}
impl<T: ScalarTypeInteger> NoHandles for Atomic<T> {}

impl<T: ScalarTypeInteger> AsAny for Atomic<T> {
    fn as_any(&self) -> Any { self.any }
}

impl<T: ScalarTypeInteger> From<Any> for Atomic<T> {
    fn from(any: Any) -> Self {
        super::typecheck_downcast(any, <Self as GpuSized>::sized_ty().into(), |any| Self {
            any,
            phantom: PhantomData,
        })
    }
}

impl<T: ScalarTypeInteger> FromAnys for Atomic<T> {
    fn expected_num_anys() -> usize { 1 }

    #[track_caller]
    fn from_anys(mut anys: impl Iterator<Item = Any>) -> Self { super::layout_traits::from_single_any(anys).into() }
}

impl<T: ScalarTypeInteger> GetAllFields for Atomic<T> {
    fn fields_as_anys_unchecked(self_as_any: Any) -> impl std::borrow::Borrow<[Any]> { [] }
}

impl<T: ScalarTypeInteger> LayoutableSized for Atomic<T> {
    fn layoutable_type_sized() -> layoutable::SizedType {
        layoutable::Atomic {
            scalar: T::SCALAR_TYPE_INTEGER,
        }
        .into()
    }
}
impl<T: ScalarTypeInteger> Layoutable for Atomic<T> {
    fn layoutable_type() -> layoutable::LayoutableType { Self::layoutable_type_sized().into() }
}

impl<T: ScalarTypeInteger> GpuLayout for Atomic<T> {
    type GpuRepr = repr::Storage;

    fn cpu_type_name_and_layout()
    -> Option<Result<(std::borrow::Cow<'static, str>, TypeLayout), ArrayElementsUnsizedError>> {
        None
    }
}

impl<Int, AS> Ref<Atomic<Int>, AS, ReadWrite>
where
    Int: ScalarTypeInteger,
    AS: AddressSpaceAtomic,
{
    /// (no documentation yet)
    #[track_caller]
    pub fn load(&self) -> vec<Int, x1> { self.as_any().address().atomic_load().into() }

    /// (no documentation yet)
    #[track_caller]
    pub fn store(&self, value: impl To<vec<Int, x1>>) { self.as_any().address().atomic_store(value.to_any()); }

    /// (no documentation yet)
    #[track_caller]
    pub fn swap(&self, value: impl To<vec<Int, x1>>) -> vec<Int, x1> {
        self.as_any().address().atomic_exchange(value.to_any()).into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn fetch_add(&self, value: impl To<vec<Int, x1>>) -> vec<Int, x1> {
        self.as_any()
            .address()
            .atomic_read_modify_write(ir::AtomicModify::Add, value.to_any())
            .into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn fetch_sub(&self, value: impl To<vec<Int, x1>>) -> vec<Int, x1> {
        self.as_any()
            .address()
            .atomic_read_modify_write(ir::AtomicModify::Sub, value.to_any())
            .into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn fetch_max(&self, value: impl To<vec<Int, x1>>) -> vec<Int, x1> {
        self.as_any()
            .address()
            .atomic_read_modify_write(ir::AtomicModify::Max, value.to_any())
            .into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn fetch_min(&self, value: impl To<vec<Int, x1>>) -> vec<Int, x1> {
        self.as_any()
            .address()
            .atomic_read_modify_write(ir::AtomicModify::Min, value.to_any())
            .into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn fetch_and(&self, value: impl To<vec<Int, x1>>) -> vec<Int, x1> {
        self.as_any()
            .address()
            .atomic_read_modify_write(ir::AtomicModify::And, value.to_any())
            .into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn fetch_or(&self, value: impl To<vec<Int, x1>>) -> vec<Int, x1> {
        self.as_any()
            .address()
            .atomic_read_modify_write(ir::AtomicModify::Or, value.to_any())
            .into()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn fetch_xor(&self, value: impl To<vec<Int, x1>>) -> vec<Int, x1> {
        self.as_any()
            .address()
            .atomic_read_modify_write(ir::AtomicModify::Xor, value.to_any())
            .into()
    }


    /// If `self` is equal to `current` replaces the contents of `self` with `new`
    /// and returns the previous value.
    ///
    /// returns a `(previous_value, success)` tuple, where `success` is `true` if the
    /// contents of `self` were replaced.
    ///
    /// ### spurious fail:
    /// important: The equality comparison may spuriously fail on some implementations. That is, the
    /// second component of the returned pair may be false even if the first component of the pair
    /// equals `current`.
    ///
    /// see
    /// - WGSL `atomicCompareExchangeWeak` https://www.w3.org/TR/WGSL/#atomic-compare-exchange-weak
    /// - Rust equivalent [`std::sync::atomic::AtomicU32::compare_exchange_weak`]
    #[track_caller]
    pub fn compare_exchange_weak(
        &self,
        current: impl To<vec<Int, x1>>,
        new: impl To<vec<Int, x1>>,
    ) -> (vec<Int, x1>, boolx1) // this return value should become a shame::Option in the future, once we have iterators figured out
    {
        let acew = self.as_any().address().atomic_compare_exchange_weak(
            AtomicCompareExchangeWeakGenerics(AS::ADDRESS_SPACE, Int::SCALAR_TYPE_INTEGER),
            current.to_any(),
            new.to_any(),
        );
        (acew.old_value.into(), acew.exchanged.into())
    }
}
