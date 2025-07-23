use crate::common::small_vec::SmallVec;
use crate::frontend::any::shared_io::{BindPath, BindingType};
use crate::frontend::any::{Any, InvalidReason};
use crate::frontend::encoding::buffer::BufferRefInner;
use crate::{
    call_info,
    frontend::{
        encoding::{
            buffer::{BufferAddressSpace, BufferInner},
            EncodingErrorKind,
        },
        error::InternalError,
    },
    ir::{self, pipeline::StageMask, recording::Context, SizedStruct, StructureDefinitionError},
};

use std::{
    array::from_ref,
    borrow::{Borrow, Cow},
    ops::Deref,
    rc::Rc,
};

use super::layout_traits::{GetAllFields, GpuLayout};
use super::type_layout::{self, recipe, TypeLayout};
use super::type_traits::{GpuAligned, GpuSized, GpuStore, GpuStoreImplCategory, NoBools};
use super::{
    error::FrontendError,
    layout_traits::{ArrayElementsUnsizedError, FromAnys},
    mem::AddressSpace,
    reference::{AccessMode, AccessModeReadable},
    type_traits::{BindingArgs, NoAtomics, NoHandles},
    typecheck_downcast, AsAny,
};
use super::{GpuType, ToGpuType};

#[diagnostic::on_unimplemented(
    message = "`{Self}` contains fields whose size on the gpu is not known at rust compile-time."
)]
// implemented by `derive(shame::GpuLayout)` if all struct fields are `GpuSized`
// (the fields of a `shame::Struct`)
// TODO(release) consider renaming to SizedStoreFields
/// (no documentation yet)
pub trait SizedFields: BufferFields + GpuSized /*not `NoAtomics`, since it can be part of a larger `Buffer`*/ {
    #[doc(hidden)]
    fn get_sizedstruct_type() -> ir::SizedStruct;
}

// the fields of a buffer binding, vertex buffer or struct (each of those options has their own additional bounds).
// May contain atomics, packed vectors, or a runtime-sized `Array<T>` at the last field
// TODO(release) consider renaming to StoreFields
/// (no documentation yet)
pub trait BufferFields: GpuStore + GpuAligned + GpuLayout + NoHandles + FromAnys + GetAllFields {
    /// (no documentation yet)
    fn as_anys(&self) -> impl Borrow<[Any]>;

    /// behaves just like `Clone::clone`.
    /// this exists only to not require #[derive(Clone)] on every #[derive(GpuLayout)]
    #[doc(hidden)]
    fn clone_fields(&self) -> Self;

    #[doc(hidden)]
    fn get_bufferblock_type() -> ir::BufferBlock;
}

#[derive(Copy)]
// a [`GpuSized`] struct containing the fields of the rust struct `Layout`.
// To create a struct within a shader, you first need to define its layout via
// `derive(GpuLayout)`.
// TODO(docs) example code
// TODO(docs) explain why this exists, and what it means for `Fields` to be a generic parameter here
/// (no documentation yet)
pub struct Struct<Layout: SizedFields + GpuStore> {
    fields: Layout,
    any: Any,
}

impl<T: SizedFields + GpuStore + NoAtomics> Struct<T> {
    /// (no documentation yet)
    #[track_caller]
    pub fn new(fields: T) -> Self { fields.to_gpu() }
}

impl<T: SizedFields + GpuStore> Clone for Struct<T> {
    fn clone(&self) -> Self {
        Self {
            fields: self.fields.clone_fields(),
            any: self.any,
        }
    }
}

impl<T: SizedFields + GpuStore> GpuStore for Struct<T> {
    type RefFields<AS: AddressSpace, AM: AccessMode> = T::RefFields<AS, AM>;
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

impl<T: SizedFields + GpuStore> GpuAligned for Struct<T> {
    fn aligned_ty() -> ir::AlignedType { ir::AlignedType::Sized(<Self as GpuSized>::sized_ty()) }
}

impl<T: SizedFields + GpuStore> GpuSized for Struct<T> {
    fn sized_ty() -> ir::SizedType { ir::SizedType::Structure(T::get_sizedstruct_type()) }
}

impl<T: SizedFields + GpuStore + NoBools> NoBools for Struct<T> {}
impl<T: SizedFields + GpuStore + NoAtomics> NoAtomics for Struct<T> {}
impl<T: SizedFields + GpuStore + NoHandles> NoHandles for Struct<T> {}

impl<T: SizedFields + GpuStore> Deref for Struct<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target { &self.fields }
}

impl<T: SizedFields + GpuStore + NoBools + GpuLayout> GpuLayout for Struct<T> {
    fn layout_recipe() -> recipe::TypeLayoutRecipe { T::layout_recipe() }

    fn cpu_type_name_and_layout() -> Option<Result<(Cow<'static, str>, TypeLayout), ArrayElementsUnsizedError>> {
        T::cpu_type_name_and_layout().map(|x| x.map(|(name, l)| (format!("Struct<{name}>").into(), l)))
    }
}

impl<T: SizedFields + GpuStore> FromAnys for Struct<T> {
    fn expected_num_anys() -> usize { 1 }

    fn from_anys(anys: impl Iterator<Item = Any>) -> Self { super::layout_traits::from_single_any(anys).into() }
}

impl<T: SizedFields + GpuStore> GpuType for Struct<T>
where
    Struct<T>: Sized,
{
    fn ty() -> ir::Type { <Self as GpuSized>::sized_ty().into() }

    #[track_caller]
    fn from_any_unchecked(any: Any) -> Self {
        let ty = T::get_sizedstruct_type();
        let t = Context::try_with(call_info!(), |ctx| {
            T::from_anys(ty.fields().map(|field| any.get_field(field.name.clone())))
        })
        .unwrap_or_else(|| {
            T::from_anys(
                ty.fields()
                    .map(|field| Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding)),
            )
        });
        Self { fields: t, any }
    }
}

impl<T: SizedFields + GpuStore> ToGpuType for Struct<T> {
    type Gpu = Self;
    fn to_gpu(&self) -> Self::Gpu { self.clone() }
    fn to_any(&self) -> Any { self.any }
    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { Some(self) }
}

impl<T: SizedFields + GpuStore> AsAny for Struct<T> {
    fn as_any(&self) -> Any { self.any }
}

impl<T: SizedFields + GpuStore> From<Any> for Struct<T> {
    #[track_caller]
    fn from(any: Any) -> Self {
        let _call_info_scope = Context::call_info_scope();
        typecheck_downcast(
            any,
            ir::Type::Store(<Self as GpuStore>::store_ty()),
            Self::from_any_unchecked,
        )
    }
}

impl<T: SizedFields + GpuStore> GetAllFields for Struct<T> {
    fn fields_as_anys_unchecked(self_as_any: Any) -> impl Borrow<[Any]> { T::fields_as_anys_unchecked(self_as_any) }
}
