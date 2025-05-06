use super::{
    array::{Array, RuntimeSize},
    layout_traits::{FromAnys, GetAllFields, GpuLayout},
    mem::{self, AddressSpace},
    reference::{AccessMode, AccessModeReadable},
    AsAny, GpuType, ToGpuType,
};
use crate::frontend::any::shared_io::{BindPath, BindingType};
use crate::{
    call_info,
    common::proc_macro_utils::push_wrong_amount_of_args_error,
    frontend::{
        any::{render_io::VertexAttribFormat, Any, InvalidReason},
        encoding::buffer::{BufferAddressSpace, BufferInner, BufferRefInner},
        error::InternalError,
    },
    ir::{
        self,
        ir_type::{align_of_array, AlignedType},
        pipeline::StageMask,
        recording::Context,
    },
};

/// marker type for `impl Store` to specify that `Self` has no particular `Store::RefFields`
#[derive(Clone, Copy)]
pub struct EmptyRefFields;

impl FromAnys for EmptyRefFields {
    fn expected_num_anys() -> usize { 0 }

    #[track_caller]
    fn from_anys(anys: impl Iterator<Item = Any>) -> Self {
        match anys.count() {
            0 => (),
            n => {
                let err = format!(
                    "trying to instantiate {} (which takes 0 arguments) with {n} argument.",
                    std::stringify!(EmptyRefFields)
                );
                Context::try_with(call_info!(), |ctx| {
                    ctx.push_error_get_invalid_any(InternalError::new(true, err).into())
                });
            }
        };
        Self
    }
}

#[doc(hidden)]
pub struct BindingArgs {
    pub path: BindPath,
    pub visibility: StageMask,
}

//[old-doc] A type whose values can be stored in a shader memory cell
//[old-doc] (for example via buffer bindings or by allocating via [`shame::alloc`])
//[old-doc] and manipulated/accessed via
//[old-doc] a [`shame::Ref`].
//[old-doc]
//[old-doc] corresponds to WGSL "Storable type" https://www.w3.org/TR/WGSL/#storable-types
/// (no documentation yet)
pub trait GpuStore: GpuAligned + GetAllFields + FromAnys {
    /// the type whose public immutable interface is exposed by [`shame::Ref<Self>`]:
    ///
    /// `<shame::Ref<Self, _, _> as std::ops::Deref>::Target`
    ///
    /// Since `shame` cannot make use of rusts builtin `&`/`&mut` propagation mechanism,
    /// it needs to be imitated by generating reference versions of [`GpuLayout`]
    /// structs.
    ///
    /// for example, for a type
    /// ```
    /// #[derive(shame::GpuLayout)]
    /// struct Foo {
    ///     a: float,
    ///     b: float,
    /// }
    /// ```
    /// the `shame::GpuLayout` derive macro generates a type
    /// ```
    /// struct Foo_ref<AM, AS> where ... {
    ///     a: Ref<float, AM, AS>,
    ///     b: Ref<float, AM, AS>,
    /// }
    /// ```
    /// which is the [`std::ops::Deref`] target of
    /// `Ref<Foo, _, _>` such that
    /// `foo.a` and `foo.b` resolve to the appropriate `Ref<float, _, _>`.
    ///
    /// if a type `Self` does not represent a composite type (like `vec<_, x1>`), it may set
    /// `RefFields` = [`EmptyRefFields`]
    ///
    /// [`Ref<Foo, _, _>`]: crate::Ref
    /// [`shame::Ref<Self>`]: crate::Ref
    type RefFields<AS: AddressSpace, AM: AccessMode>: FromAnys + Copy;

    /// internal function that aids in the construction of `Buffer` as a `Binding`.
    ///
    /// constructs an object containing an invalid `Any` if args is `Err`
    #[doc(hidden)]
    fn instantiate_buffer_inner<AS: BufferAddressSpace>(
        args: Result<BindingArgs, InvalidReason>,
        ty: BindingType,
    ) -> BufferInner<Self, AS>
    where
        Self: std::marker::Sized /*not GpuSized, this is deliberate*/ + NoAtomics + NoBools;

    /// internal function that aids in the construction of `BufferRef` as a `Binding`
    ///
    /// constructs an object containing an invalid `Any` if args is `Err`
    #[doc(hidden)]
    fn instantiate_buffer_ref_inner<AS: BufferAddressSpace, AM: AccessModeReadable>(
        args: Result<BindingArgs, InvalidReason>,
        ty: BindingType,
    ) -> BufferRefInner<Self, AS, AM>
    where
        Self: std::marker::Sized /*not GpuSized, this is deliberate*/ + NoBools;

    #[doc(hidden)] // runtime api
    fn store_ty() -> ir::StoreType
    where
        Self: GpuType;

    #[doc(hidden)] // unstable
    #[track_caller]
    /// forces `self` to appear in the generated shader code. No dead code elimination can remove it.
    fn show(&self) -> &Self
    where
        Self: GpuType,
    {
        self.as_any().show();
        self
    }

    #[doc(hidden)] // proc macro detail
    fn impl_category() -> GpuStoreImplCategory;
}

#[doc(hidden)] // proc macro detail
pub enum GpuStoreImplCategory {
    GpuType(ir::StoreType),
    Fields(ir::BufferBlock),
}

#[diagnostic::on_unimplemented(message = "the size of `{Self}` on the gpu is not known at rust compile-time")]
/// ## known byte-size on the gpu
/// types whose byte-size on the graphics device is known at rust compile-time
///
/// This is also implemented for non-[`GpuType`]s like structs which derive [`GpuLayout`]
/// which contain only fields that are [`GpuSized`]
///
/// note: [`GpuSized`] does not imply [`GpuStore`], because [`Atomic<T>`] is [`GpuSized`] but `!GpuStore`
///
/// [`Atomic<T>`]: crate::Atomic
pub trait GpuSized: GpuAligned {
    // `GpuSized` does not imply `GpuStore`, because `Atomic<T>` is `GpuSized` but `!GpuStore`

    /// returns the `ir::SizedType` that `Self` corresponds to inside the shader type system.
    #[doc(hidden)] // runtime api
    fn sized_ty() -> ir::SizedType
    where
        Self: GpuType;
}

#[diagnostic::on_unimplemented(
    message = "the memory alignment of `{Self}` on the gpu is not known at rust compile-time"
)]
/// ## known byte-alignment on the gpu
/// types that have a byte-alignment on the graphics device that is known at rust compile-time
pub trait GpuAligned: GpuLayout {
    #[doc(hidden)] // runtime api
    fn aligned_ty() -> AlignedType
    where
        Self: GpuType;
}

#[diagnostic::on_unimplemented(
    message = "`{Self}` may contain `bool`s, which have an unspecified memory footprint on the graphics device."
)]
// implementor note:
// NoXYZ traits must require GpuLayout or some other base trait, so that the
// error message isn't misleading for user provided types `T`. Those types will show
// the base trait diagnostic, instead of "`T` contains `XYZ`" which it doesn't.
/// types that don't contain booleans at any nesting level
///
/// boolean types do not have a defined size on gpus.
/// You may want to use unsigned integers for transferring boolean data instead.
pub trait NoBools: GpuLayout {}

/// (no documentation yet)
#[diagnostic::on_unimplemented(
    message = "`{Self}` may be or contain a `shame::Atomic` type. Atomics are usable via `shame::BufferRef<_, Storage, ReadWrite>` or via allocations in workgroup memory"
)]
// implementor note:
// NoXYZ traits must require GpuLayout or some other base trait, so that the
// error message isn't misleading for user provided types `T`. Those types will show
// the base trait diagnostic, instead of "`T` contains `XYZ`" which it doesn't.
/// types that don't contain atomics at any nesting level
pub trait NoAtomics: GpuLayout {}

// implementor note:
// NoXYZ traits must require GpuLayout or some other base trait, so that the
// error message isn't misleading for user provided types `T`. Those types will show
// the base trait diagnostic, instead of "`T` contains `XYZ`" which it doesn't.
#[diagnostic::on_unimplemented(
    message = "`{Self}` may be or contain a handle type such as `Texture`, `Sampler`, `StorageTexture`."
)]
/// Implemented by types that aren't/contain no textures, storage textures, their array variants or samplers
pub trait NoHandles: GpuLayout {}

/// this trait is only implemented by:
///
/// * `sm::vec`s of non-boolean type (e.g. `sm::f32x4`)
/// * `sm::packed::PackedVec`s (e.g. `sm::packed::unorm8x4`)
pub trait VertexAttribute: GpuLayout + FromAnys {
    #[doc(hidden)] // runtime api
    fn vertex_attrib_format() -> VertexAttribFormat;
}

/// Trait that the fields of a derived `GpuLayout` type must implement.
/// This is used for showing a more helpful error message when trying to use
/// #[derive(GpuLayout)]
/// struct A { ... }
/// in
/// #[derive(GpuLayout)]
/// struct B { a: A }
/// directly. This should instead be
/// #[derive(GpuLayout)]
/// struct B { a: shame::Struct<A> }
/// which the error message points out.
#[diagnostic::on_unimplemented(
    message = "Try using shame::Struct<{Self}> instead. If that doesn't work {Self} can not be used as the field of a type deriving shame::GpuLayout."
)]
pub trait FromAny {
    /// Constructs Self from Any
    fn from_any(any: Any) -> Self;
}

impl<T: From<Any>> FromAny for T {
    fn from_any(any: Any) -> Self { T::from(any) }
}
