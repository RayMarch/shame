use std::marker::PhantomData;

use super::{
    array::{Array, ArrayLen},
    atomic::Atomic,
    error::FrontendError,
    layout_traits::{FromAnys, GetAllFields},
    mem::{self, AddressSpace},
    scalar_type::ScalarTypeInteger,
    struct_::{SizedFields, Struct},
    type_traits::{GpuSized, GpuStore, NoAtomics},
    typecheck_downcast,
    vec::ToInteger,
    AsAny, GpuType, To,
};
use crate::{any::layout::LayoutableSized, frontend::any::Any, GpuLayout};
use crate::frontend::rust_types::len::x1;
use crate::frontend::rust_types::vec::vec;
use crate::{
    boolx1, call_info,
    frontend::any::InvalidReason,
    frontend::error::InternalError,
    ir::{
        self,
        recording::{AtomicCompareExchangeWeakGenerics, Context},
        Type,
    },
    mem::SupportsAccess,
};
use std::borrow::Borrow;


/// A memory operation can read, write, or both read and write.
/// Memory locations may support only some of these accesses.
///
/// this trait is implemented by the following marker types
/// - [`Read`] for read-only access
/// - [`Write`] for write-only access
/// - [`ReadWrite`] for both read and write access
///
/// see https://www.w3.org/TR/WGSL/#memory-access-mode
pub trait AccessMode: Copy {
    #[doc(hidden)] // runtime api
    const ACCESS: ir::AccessMode;
}

/// read-only [`AccessMode`] marker type
#[derive(Clone, Copy)]
pub struct Read;
impl AccessMode for Read {
    const ACCESS: ir::AccessMode = ir::AccessMode::Read;
}

/// write-only [`AccessMode`] marker type
#[derive(Clone, Copy)]
pub struct Write;
impl AccessMode for Write {
    const ACCESS: ir::AccessMode = ir::AccessMode::Write;
}

/// read and write [`AccessMode`] marker type
#[derive(Clone, Copy)]
pub struct ReadWrite;
impl AccessMode for ReadWrite {
    const ACCESS: ir::AccessMode = ir::AccessMode::ReadWrite;
}

/// An [`AccessMode`] that is either [`Read`] or [`ReadWrite`]
pub trait AccessModeReadable: AccessMode {
    #[doc(hidden)] // runtime api
    const ACCESS_MODE_READABLE: ir::AccessModeReadable;
} // TODO(release) seal this trait
impl AccessModeReadable for ReadWrite {
    const ACCESS_MODE_READABLE: ir::AccessModeReadable = ir::AccessModeReadable::ReadWrite;
}
impl AccessModeReadable for Read {
    const ACCESS_MODE_READABLE: ir::AccessModeReadable = ir::AccessModeReadable::Read;
}

/// An [`AccessMode`] that is either [`Write`] or [`ReadWrite`]
pub trait AccessModeWritable: AccessMode {} // TODO(release) seal this trait
impl AccessModeWritable for ReadWrite {}
impl AccessModeWritable for Write {}


#[diagnostic::on_unimplemented(
    message = "cannot read from a `shame::Ref<_, _, Write>` which only provides `Write` access"
)]
/// A [`Ref`] with an [`AccessMode`] that is either [`Read`] or [`ReadWrite`]
pub trait ReadableRef {} // TODO(release) seal this trait
impl<T: GpuStore, AS: AddressSpace> ReadableRef for Ref<T, AS, ReadWrite> {}
impl<T: GpuStore, AS: AddressSpace> ReadableRef for Ref<T, AS, Read> {}

#[diagnostic::on_unimplemented(
    message = "cannot write to a `shame::Ref<_, _, Read>` which only provides `Read` access"
)]
/// A [`Ref`] with an [`AccessMode`] that is either [`Read`] or [`ReadWrite`]
pub trait WritableRef {} // TODO(release) seal this trait
impl<T: GpuStore, AS: AddressSpace> WritableRef for Ref<T, AS, ReadWrite> {}
impl<T: GpuStore, AS: AddressSpace> WritableRef for Ref<T, AS, Write> {}

// TODO(docs) Docs: mention that this has broadly same interface as `Cell`
/// (no documentation yet)
pub struct Ref<T, AS = mem::Fn, AM = ReadWrite>
where
    T: GpuStore,
    AS: AddressSpace,
    AM: AccessMode,
{
    any: Any,
    fields_as_refs: T::RefFields<AS, AM>,
}

impl<T: GpuStore, AS: AddressSpace, AM: AccessMode> Copy for Ref<T, AS, AM> {}

impl<T: GpuStore, AS: AddressSpace, AM: AccessMode> Clone for Ref<T, AS, AM> {
    fn clone(&self) -> Self { *self }
}


impl<T, AS, AM> AsAny for Ref<T, AS, AM>
where
    T: GpuStore,
    AS: AddressSpace,
    AM: AccessMode,
{
    fn as_any(&self) -> Any { self.any }
}

impl<T, AS, AM> Ref<T, AS, AM>
where
    T: GpuType + GpuStore + NoAtomics,
    AS: AddressSpace,
    AM: AccessMode,
    Self: ReadableRef,
{
    /// (no documentation yet)
    #[track_caller]
    pub fn get(&self) -> T
    where
        T: GpuSized, // value must be constructible
    {
        self.deref()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn deref(&self) -> T
    where
        T: GpuSized, // value must be constructible
    {
        self.any.ref_load().into()
    }
}

impl<T, AS, AM> std::ops::Deref for Ref<T, AS, AM>
where
    T: GpuStore,
    AS: AddressSpace,
    AM: AccessMode,
{
    type Target = T::RefFields<AS, AM>;

    fn deref(&self) -> &Self::Target { &self.fields_as_refs }
}

impl<T, AS, AM> Ref<T, AS, AM>
where
    T: GpuType + GpuStore + NoAtomics,
    AS: AddressSpace,
    AM: AccessMode,
    Self: WritableRef,
{
    /// (no documentation yet)
    #[track_caller]
    pub fn set(&self, value: impl To<T>)
    where
        T: GpuSized, // value must be constructible
    {
        self.any.set(value.to_any())
    }
}

impl<T, AS, AM> From<Any> for Ref<T, AS, AM>
where
    T: GpuType + GpuStore,
    AS: AddressSpace,
    AM: AccessMode,
{
    #[track_caller]
    fn from(any: Any) -> Self {
        let from_any_unchecked = |any| Self {
            any,
            fields_as_refs: {
                let field_anys = <T as GetAllFields>::fields_as_anys_unchecked(any);
                FromAnys::from_anys((field_anys.borrow() as &[Any]).iter().copied())
            },
        };
        Context::with(call_info!(), |ctx| {
            let invalid = |e| from_any_unchecked(ctx.push_error_get_invalid_any(e));
            let expected_store_ty = T::store_ty();

            match any.ty() {
                Some(Type::Ref(alloc, store_ty, access)) => match alloc.address_space == AS::ADDRESS_SPACE {
                    true => {
                        typecheck_downcast(any, Type::Ref(alloc, expected_store_ty, AM::ACCESS), from_any_unchecked)
                    }
                    false => invalid(
                        FrontendError::DowncastWithInvalidAddressSpace {
                            dynamic_as: alloc.address_space,
                            rust_as: AS::ADDRESS_SPACE,
                        }
                        .into(),
                    ),
                },
                Some(ty) => invalid(
                    FrontendError::DowncastNonRefToRef {
                        dynamic_type: ty,
                        rust_type: expected_store_ty,
                    }
                    .into(),
                ),
                None => from_any_unchecked(any), // already invalid.
            }
        })
    }
}

impl<T, AS, AM, N> Ref<Array<T, N>, AS, AM>
where
    T: GpuType + GpuSized + GpuStore + 'static + GpuLayout + LayoutableSized,
    AS: AddressSpace + 'static,
    AM: AccessModeReadable + 'static,
    N: ArrayLen,
{
    /// (no documentation yet)
    #[track_caller]
    pub fn at(&self, index: impl ToInteger) -> Ref<T, AS, AM> { self.as_any().array_index(index.to_any()).into() }
}

impl<T, AM> Ref<T, mem::WorkGroup, AM>
where
    T: GpuType + GpuStore + NoAtomics,
    AM: AccessMode,
    Self: ReadableRef,
{
    // see WGSL https://www.w3.org/TR/WGSL/#workgroupUniformLoad-builtin
    /// (no documentation yet)
    pub fn uniform_load(&self) -> T { self.as_any().address().workgroup_uniform_load().into() }
}
