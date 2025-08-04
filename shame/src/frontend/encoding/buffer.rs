use crate::common::proc_macro_reexports::GpuStoreImplCategory;
use crate::frontend::any::shared_io::{BindPath, BindingType, BufferBindingType};
use crate::frontend::any::{Any, InvalidReason};
use crate::frontend::rust_types::array::{Array, ArrayLen, RuntimeSize, Size};
use crate::frontend::rust_types::atomic::Atomic;
use crate::frontend::rust_types::layout_traits::{get_layout_compare_with_cpu_push_error, FromAnys, GetAllFields};
use crate::frontend::rust_types::len::{Len, Len2};
use crate::frontend::rust_types::mem::{self, AddressSpace, SupportsAccess};
use crate::frontend::rust_types::reference::Ref;
use crate::frontend::rust_types::reference::{AccessMode, Read, ReadWrite};
use crate::frontend::rust_types::scalar_type::{ScalarType, ScalarTypeFp, ScalarTypeNumber};
use crate::frontend::rust_types::struct_::{BufferFields, SizedFields, Struct};
use crate::frontend::rust_types::type_traits::{BindingArgs, GpuSized, GpuStore, NoAtomics, NoBools, NoHandles};
use crate::frontend::rust_types::vec::vec;
use crate::frontend::rust_types::{mat::mat, GpuType};
use crate::frontend::rust_types::{reference::AccessModeReadable, scalar_type::ScalarTypeInteger};
use crate::ir::pipeline::StageMask;
use crate::ir::recording::{Context, MemoryRegion};
use crate::ir::Type;
use crate::{self as shame, call_info, ir, GpuLayout};

use std::borrow::Borrow;
use std::marker::PhantomData;
use std::ops::{Deref, Mul};

use super::binding::Binding;

/// Address spaces used for [`Buffer`] and [`BufferRef`] bindings.
///
/// Implemented by the marker types
/// - [`mem::Uniform`]
/// - [`mem::Storage`]
pub trait BufferAddressSpace: AddressSpace + SupportsAccess<Read> {
    /// Either Storage or Uniform address space.
    const BUFFER_ADDRESS_SPACE: BufferAddressSpaceEnum;
}
/// Either Storage or Uniform address space.
#[derive(Debug, Clone, Copy)]
pub enum BufferAddressSpaceEnum {
    /// Storage address space
    Storage,
    /// Uniform address space
    Uniform,
}
impl BufferAddressSpace for mem::Uniform {
    const BUFFER_ADDRESS_SPACE: BufferAddressSpaceEnum = BufferAddressSpaceEnum::Uniform;
}
impl BufferAddressSpace for mem::Storage {
    const BUFFER_ADDRESS_SPACE: BufferAddressSpaceEnum = BufferAddressSpaceEnum::Storage;
}
impl std::fmt::Display for BufferAddressSpaceEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferAddressSpaceEnum::Storage => write!(f, "storage address space"),
            BufferAddressSpaceEnum::Uniform => write!(f, "uniform address space"),
        }
    }
}

/// A read-only buffer binding, for writeable buffers and atomics use [`BufferRef`] instead.
///
/// Buffer contents are accessible via [`std::ops::Deref`] `*`.
///
/// ## Generic arguments
/// - `Content`: the buffer content. May not contain atomics or bools
/// - `DYNAMIC_OFFSET`: whether an offset into the bound buffer can be specified when binding its bind-group in the graphics api.
/// - `AS`: the address space can be either of
///     - `mem::Uniform`
///         - has special memory layout requirements, see https://www.w3.org/TR/WGSL/#address-space-layout-constraints
///         - check the uniform buffer size limitations via the graphics api
///     - `mem::Storage`
///         - for large buffers
///
/// ## Example
/// ```
/// use shame as sm;
///
/// // storage buffers
/// let buffer: sm::Buffer<sm::f32x4, sm::mem::Storage> = bind_group_iter.next();
/// // same as above, since `mem::Storage` is the default
/// let buffer: sm::Buffer<sm::f32x4> = bind_group_iter.next();
///
/// // access via `std::ops::Deref` `*`
/// let value = *buffer + sm::vec!(1.0, 2.0, 3.0, 4.0);
///
/// // uniform buffers
/// let buffer: sm::Buffer<sm::f32x4, sm::mem::Uniform> = bind_group_iter.next();
///
/// // runtime-sized array buffer
/// let buffer: sm::Buffer<sm::Array<sm::f32x4>> = bind_group_iter.next();
///
/// // fixed size array buffer
/// let buffer: sm::Buffer<sm::Array<sm::f32x4, sm::Size<512>>> = bind_group_iter.next();
///
/// // custom struct type buffer
/// #[derive(sm::GpuLayout)]
/// struct Transforms {
///     world: f32x4x4,
///     view: f32x4x4,
///     proj: f32x4x4,
/// }
/// let buffer: sm::Buffer<Transforms> = bind_group_iter.next();
/// // equivalent to
/// let buffer: sm::Buffer<sm::Struct<Transforms>> = bind_group_iter.next();
///
/// // array of structs
/// let buffer: sm::Buffer<sm::Array<sm::Struct<Transforms>>> = bind_group_iter.next();
/// ```
///
/// > maintainer note:
/// > the precise trait bounds of buffer bindings are found in the `Binding` impl blocks.
#[derive(Clone, Copy)]
pub struct Buffer<Content, AS = mem::Storage, const DYNAMIC_OFFSET: bool = false>
where
    Content: GpuStore + NoHandles + NoAtomics + NoBools,
    AS: BufferAddressSpace,
{
    pub(crate) inner: BufferInner<Content, AS>,
}

impl<T, AS, const DYN_OFFSET: bool> Buffer<T, AS, DYN_OFFSET>
where
    T: GpuStore + NoHandles + NoAtomics + NoBools + GpuLayout,
    AS: BufferAddressSpace,
{
    #[track_caller]
    fn new(args: Result<BindingArgs, InvalidReason>) -> Self {
        let skip_stride_check = true; // not a vertex buffer
        Context::try_with(call_info!(), |ctx| {
            get_layout_compare_with_cpu_push_error::<T>(ctx, skip_stride_check)
        });
        Self {
            inner: T::instantiate_buffer_inner(args, BufferInner::<T, AS>::binding_type(DYN_OFFSET)),
        }
    }
}

impl<T, AS, AM, const DYN_OFFSET: bool> BufferRef<T, AS, AM, DYN_OFFSET>
where
    T: GpuStore + NoHandles + NoBools + GpuLayout,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
{
    #[track_caller]
    fn new(args: Result<BindingArgs, InvalidReason>) -> Self {
        let skip_stride_check = true; // not a vertex buffer
        Context::try_with(call_info!(), |ctx| {
            get_layout_compare_with_cpu_push_error::<T>(ctx, skip_stride_check)
        });
        Self {
            inner: T::instantiate_buffer_ref_inner(args, BufferRefInner::<T, AS, AM>::binding_type(DYN_OFFSET)),
        }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum BufferInner<T: GpuStore, AS: BufferAddressSpace> {
    FieldsSized(T),
    /// unused, `BufferRef` must be used for this case
    FieldsRuntimeSized(T::RefFields<AS, Read>),
    PlainSized(T),
    RuntimeSizedArray(Ref<T, AS, Read>),
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum BufferRefInner<T: GpuStore, AS: BufferAddressSpace, AM: AccessModeReadable> {
    Fields(T::RefFields<AS, AM>),
    Plain(Ref<T, AS, AM>),
}

impl<T: GpuType + GpuStore + GpuSized + NoBools, AS: BufferAddressSpace> BufferInner<T, AS> {
    #[track_caller]
    pub(crate) fn new_plain(args: Result<BindingArgs, InvalidReason>, bind_ty: BindingType) -> Self {
        let as_ref = false;
        let any = match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => {
                Any::binding(path, visibility, <T as GpuStore>::store_ty(), bind_ty, as_ref)
            }
        };
        BufferInner::PlainSized(any.into())
    }
}

impl<T: BufferFields + NoAtomics + NoBools, AS: BufferAddressSpace> BufferInner<T, AS> {
    #[track_caller]
    #[doc(hidden)]
    pub fn new_fields(args: Result<BindingArgs, InvalidReason>, bind_ty: BindingType) -> Self {
        let init_any = |as_ref, store_ty| match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => Any::binding(path, visibility, store_ty, bind_ty, as_ref),
        };

        let block = T::get_bufferblock_type();
        match ir::SizedStruct::try_from(block.clone()) {
            Ok(struct_) => {
                let store_ty = ir::StoreType::Sized(ir::SizedType::Structure(struct_));
                let block_any = init_any(false, store_ty);
                let fields_anys = <T as GetAllFields>::fields_as_anys_unchecked(block_any);
                let fields_anys = (fields_anys.borrow() as &[Any]).iter().cloned();
                let fields = <T as FromAnys>::from_anys(fields_anys);
                BufferInner::FieldsSized(fields)
            }
            Err(_) => {
                // TODO(release) test this! A `Buffer<Foo> where Foo's last field is a runtime sized array`
                let store_ty = ir::StoreType::BufferBlock(block);
                let block_any_ref = init_any(true, store_ty);
                let fields_anys_refs = <T as GetAllFields>::fields_as_anys_unchecked(block_any_ref);
                let fields_anys_refs = (fields_anys_refs.borrow() as &[Any]).iter().cloned();
                let fields_refs = <T::RefFields<AS, Read> as FromAnys>::from_anys(fields_anys_refs);
                BufferInner::FieldsRuntimeSized(fields_refs)
            }
        }
    }
}

impl<T: GpuType + GpuStore + GpuSized, AS: BufferAddressSpace, L: ArrayLen> BufferInner<Array<T, L>, AS> {
    #[track_caller]
    #[doc(hidden)]
    pub(crate) fn new_array(args: Result<BindingArgs, InvalidReason>, bind_ty: BindingType) -> Self {
        let call_info = call_info!();

        let init_any = |ty, as_ref| match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => Any::binding(path, visibility, ty, bind_ty, as_ref),
        };

        let store_ty = <Array<T, L> as GpuStore>::store_ty();
        match L::LEN {
            // GpuSized
            Some(_) => {
                let as_ref = false;
                let any = init_any(store_ty, as_ref);
                BufferInner::PlainSized(any.into())
            }
            // RuntimeSize
            None => {
                let as_ref = true;
                let any_ref = init_any(store_ty, as_ref);
                BufferInner::RuntimeSizedArray(any_ref.into())
            }
        }
    }
}

impl<T: GpuStore + NoBools, AS: BufferAddressSpace, AM: AccessModeReadable> BufferRefInner<T, AS, AM> {
    #[track_caller]
    pub(crate) fn new_plain(args: Result<BindingArgs, InvalidReason>, bind_ty: BindingType) -> Self
    where
        T: GpuType,
    {
        let as_ref = true;
        let store_ty = <T as GpuStore>::store_ty();
        let any = match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => Any::binding(path, visibility, store_ty, bind_ty, as_ref),
        };
        BufferRefInner::Plain(any.into())
    }

    #[track_caller]
    #[doc(hidden)]
    pub fn new_fields(args: Result<BindingArgs, InvalidReason>, bind_ty: BindingType) -> Self
    where
        T: BufferFields,
    {
        let as_ref = true;
        let block_any_ref = match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => Any::binding(
                path,
                visibility,
                ir::StoreType::BufferBlock(T::get_bufferblock_type()),
                bind_ty,
                as_ref,
            ),
        };
        let fields_anys_refs = <T as GetAllFields>::fields_as_anys_unchecked(block_any_ref);
        let fields_anys_refs = (fields_anys_refs.borrow() as &[Any]).iter().cloned();
        let fields_refs = <T::RefFields<AS, AM> as FromAnys>::from_anys(fields_anys_refs);
        BufferRefInner::Fields(fields_refs)
    }
}

impl<T: GpuStore, AS: BufferAddressSpace> BufferInner<T, AS> {
    fn binding_type(has_dynamic_offset: bool) -> BindingType {
        let ty = match AS::ADDRESS_SPACE {
            ir::AddressSpace::Uniform => BufferBindingType::Uniform,
            ir::AddressSpace::Storage => BufferBindingType::Storage(Read::ACCESS_MODE_READABLE),
            _ => unreachable!("AS: BufferAddressSpace"),
        };
        BindingType::Buffer { ty, has_dynamic_offset }
    }
}

impl<T: GpuStore, AS: BufferAddressSpace, AM: AccessModeReadable> BufferRefInner<T, AS, AM> {
    fn binding_type(has_dynamic_offset: bool) -> BindingType {
        let ty = match AS::ADDRESS_SPACE {
            ir::AddressSpace::Uniform => BufferBindingType::Uniform,
            ir::AddressSpace::Storage => BufferBindingType::Storage(AM::ACCESS_MODE_READABLE),
            _ => unreachable!("AS: BufferAddressSpace"),
        };
        BindingType::Buffer { ty, has_dynamic_offset }
    }
}

// maintainer note:
//
// ## Why the `GpuSized` requirement?
//
// in shame a buffer-as-value (= `Buffer`) may not contain **fields** of
// Array<_, RuntimeSize>/Atomic<_> since in WGSL this type
// cannot exist outside of a reference.
// however, as an exception to that rule, it seems like we can allow `Buffer<Array<T, RuntimeSize>>` directly
#[rustfmt::skip] impl<T: GpuStore + NoHandles + NoAtomics + NoBools, AS: BufferAddressSpace, const DYN_OFFSET: bool>
Binding for Buffer<T, AS, DYN_OFFSET>
where
    T: GpuSized+ GpuLayout
{
    fn binding_type() -> BindingType { BufferInner::<T, AS>::binding_type(DYN_OFFSET) }
    #[track_caller]
    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self { Buffer::new(args) }

    fn store_ty() -> ir::StoreType {
        store_type_from_impl_category(T::impl_category())
    }
}

fn store_type_from_impl_category(category: GpuStoreImplCategory) -> ir::StoreType {
    match category {
        GpuStoreImplCategory::GpuType(ty) => ty,
        GpuStoreImplCategory::Fields(buffer_block) => match buffer_block.clone().try_into() {
            Ok(sized) => ir::StoreType::Sized(ir::SizedType::Structure(sized)),
            Err(_) => ir::StoreType::BufferBlock(buffer_block),
        },
    }
}

#[rustfmt::skip] impl<T: GpuStore + NoHandles + NoAtomics + NoBools, AS: BufferAddressSpace, const DYN_OFFSET: bool>
Binding for Buffer<Array<T>, AS, DYN_OFFSET>
where
    T: GpuType + GpuSized + GpuLayout
{
    fn binding_type() -> BindingType { BufferInner::<T, AS>::binding_type(DYN_OFFSET) }
    #[track_caller]
    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self { Buffer::new(args) }

    fn store_ty() -> ir::StoreType {
        ir::StoreType::RuntimeSizedArray(T::sized_ty())
    }
}

impl<T: GpuStore + NoHandles + NoAtomics + NoBools, AS, const DYN_OFFSET: bool> std::ops::Deref
    for Buffer<T, AS, DYN_OFFSET>
where
    T: GpuSized,
    AS: BufferAddressSpace,
{
    type Target = T;

    fn deref(&self) -> &T {
        use BufferInner as B;
        match &self.inner {
            B::FieldsRuntimeSized(_) | B::RuntimeSizedArray(_) => unreachable!("T: GpuSized"),
            B::FieldsSized(t) | B::PlainSized(t) => t,
        }
    }
}

macro_rules! impl_deref_for_bufferref_of_gputypes {
    (
        $(impl<$(
            $gen: ident : $bound: ident $(+ $bounds: ident)*
        ),*> Deref for BufferRef<$type_name: ty, ...>;)*
    ) => {
        $(
            impl<
                $($gen: $bound $(+ $bounds)*,)*
                AS, AM, const DYN_OFFSET: bool> std::ops::Deref for BufferRef<$type_name, AS, AM, DYN_OFFSET>
            where
                $type_name: GpuStore + NoHandles + NoBools,
                AS: BufferAddressSpace,
                AM: AccessModeReadable
            {
                type Target = Ref<$type_name, AS, AM>;

                fn deref(&self) -> &Self::Target {
                    use BufferRefInner as B;
                    match &self.inner {
                        B::Fields(_) => unreachable!("`Self: GpuType` + the `GpuStore` impl of all `GpuType` types sets this up as `Plain`"),
                        B::Plain(ref_t) => &ref_t,
                    }
                }
            }
        )*
    };
}

impl_deref_for_bufferref_of_gputypes! {
    impl<T: SizedFields>                     Deref for BufferRef<Struct<T>   , ...>;
    impl<T: GpuType + GpuSized, L: ArrayLen> Deref for BufferRef<Array<T, L> , ...>;
    impl<T: ScalarType, L: Len>              Deref for BufferRef<vec<T, L>   , ...>;
    impl<T: ScalarTypeFp, C: Len2, R: Len2>  Deref for BufferRef<mat<T, C, R>, ...>;
    impl<T: ScalarTypeInteger>               Deref for BufferRef<Atomic<T>   , ...>;
}

impl<T: GpuStore + NoHandles + NoBools, AS, AM, const DYN_OFFSET: bool> std::ops::Deref
    for BufferRef<T, AS, AM, DYN_OFFSET>
where
    T: BufferFields,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
{
    type Target = T::RefFields<AS, AM>;

    fn deref(&self) -> &Self::Target {
        use BufferRefInner as B;
        match &self.inner {
            B::Fields(t_ref_fields) => t_ref_fields,
            B::Plain(_) => unreachable!("T: BufferFields + creation in derive macro `Store` impl have the same bound"),
        }
    }
}

// TODO(release) targeting WebGPU, push error if T's last field is an `Array<_, RuntimeSize>` in a `Buffer<T, mem::Unifrom>`
// TODO(release) targeting WebGPU, push an error if `Buffer<Array<_, RuntimeSize>, mem::Uniform>` is created

/// A read-write or read-only buffer binding.
///
/// [`BufferRef`] is required for writeable buffers and buffers containing [`Atomic`]s.
/// For read-only buffers [`Buffer`] may be more ergonomic, as it don't requires `.get()`.
///
/// only the [`mem::Storage`] address space supports [`ReadWrite`] access.
///
/// ## Generic arguments
/// - `Content`: the buffer content. May not contain bools
/// - `DYNAMIC_OFFSET`: whether an offset into the bound buffer can be specified when binding its bind-group in the graphics api.
/// - `AS`: the address space can be either of
///     - `mem::Uniform`
///         - read only access
///         - has special memory layout requirements, see https://www.w3.org/TR/WGSL/#address-space-layout-constraints
///         - check the uniform buffer size limitations via the graphics api
///     - `mem::Storage`
///         - readwrite or read-only access
///         - for large buffers
///
/// # Example
/// ```
/// use shame as sm;
/// use sm::f32x4x4;
/// use sm::f32x4;
///
/// // storage buffers
/// let buffer: sm::BufferRef<f32x4, sm::mem::Storage, sm::ReadWrite> = bind_group_iter.next();
/// // same as above, since `mem::Storage` and `ReadWrite` is the default
/// let buffer: sm::BufferRef<f32x4> = bind_group_iter.next();
///
/// // read access via `.get()`
/// let value = buffer.get() + sm::vec!(1.0, 2.0, 3.0, 4.0);
///
/// // write access via `.set()`
/// buffer.set(sm::vec!(1.0, 2.0, 3.0, 4.0));
///
/// // uniform buffers
/// let buffer: sm::BufferRef<f32x4, sm::mem::Uniform, sm::Read> = bind_group_iter.next();
///
/// // fixed size array buffer
/// let buffer: sm::BufferRef<sm::Array<f32x4, sm::Size<512>>> = bind_group_iter.next();
///
/// // runtime-sized array buffer
/// let buffer: sm::BufferRef<sm::Array<f32x4>> = bind_group_iter.next();
///
/// // array lookup returns reference
/// let element: sm::Ref<f32x4> = buffer.at(4u32);
/// buffer.at(8u32).set(sm::vec!(1.0, 2.0, 3.0, 4.0));
///
/// // custom struct type buffer
/// #[derive(sm::GpuLayout)]
/// struct Transforms {
///     world: f32x4x4,
///     view: f32x4x4,
///     proj: f32x4x4,
/// }
/// let buffer: sm::BufferRef<Transforms> = bind_group_iter.next();
///
/// // field access returns references
/// let world: sm::Ref<f32x4x4> = buffer.world;
///
/// // get fields via `.get()`
/// let matrix: f32x4x4 = buffer.world.get();
///
/// // write to fields via `.set(_)`
/// buffer.world.set(mat::zero())
///
/// // array of structs
/// let buffer: sm::Buffer<sm::Array<sm::Struct<Transforms>>> = bind_group_iter.next();
///
/// ```
///
/// > maintainer note:
/// > the precise trait bounds of buffer bindings are found in the `Binding` impl blocks.
#[derive(Clone, Copy)]
pub struct BufferRef<Content, AS = mem::Storage, AM = ReadWrite, const DYNAMIC_OFFSET: bool = false>
where
    Content: GpuStore + NoHandles + NoBools,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
{
    pub(crate) inner: BufferRefInner<Content, AS, AM>,
}

#[rustfmt::skip] impl<T: GpuStore + NoBools + NoHandles + GpuLayout, AS, AM, const DYN_OFFSET: bool>
Binding for BufferRef<T, AS, AM, DYN_OFFSET>
where
    AS: BufferAddressSpace + SupportsAccess<AM>,
    AM: AccessModeReadable + AtomicsRequireWriteable<T>
{
    fn binding_type() -> BindingType { BufferRefInner::<T, AS, AM>::binding_type(DYN_OFFSET) }
    #[track_caller]
    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self { BufferRef::new(args) }

    fn store_ty() -> ir::StoreType {
        store_type_from_impl_category(T::impl_category())
    }
}

#[diagnostic::on_unimplemented(
    message = "atomics can only be used in read-write storage buffers. Use `ReadWrite` instead of `Read`."
)]
trait AtomicsRequireWriteable<T> {}
impl<T> AtomicsRequireWriteable<T> for ReadWrite {}
impl<T> AtomicsRequireWriteable<T> for Read where T: NoAtomics {}
