use layout_traits::GpuLayout;
use mem::Cell;
use reference::{Read, ReadWrite};
use type_traits::{GpuSized, GpuStore};

use self::error::FrontendError;
use super::any::{Any, InvalidReason};
use crate::frontend::rust_types::reference::Ref;
use crate::{
    call_info,
    ir::{self, recording::Context, Type},
};

pub mod aliases;
pub mod array;
pub mod atomic;
pub mod barrier;
pub mod constructors;
pub mod error;
pub mod index;
pub mod layout_traits;
pub mod len;
pub mod mat;
pub mod mat_operators;
pub mod mem;
pub mod packed_vec;
pub mod reference;
pub mod scalar_type;
pub mod struct_;
pub mod type_layout;
pub mod type_traits;
pub mod vec;
pub mod vec_functions;
pub mod vec_operators;
pub mod vec_range;
pub mod vec_range_traits;

//[old-doc] A type that records the expressions that it is part of into the `shame`
//[old-doc] expression graph,
//[old-doc] so that shader code can be generated from it.
//[old-doc] implemented by
//[old-doc] - `vec<T, L>` (but not `PackedVec<L, T>`)
//[old-doc] - `mat<L, T>`
//[old-doc] - `Struct<T>`
//[old-doc] - `Array<T, Size<N>>`
//[old-doc] - `Array<T>`
//[old-doc] - `Atomic<T>`
//[old-doc] - `Ref<T, …>`
//[old-doc] - `Sampler<…>`
//[old-doc] - `Texture<…>`
//[old-doc] - `StorageTexture<…>`
/// (no documentation yet)
///
pub trait GpuType: ToGpuType<Gpu = Self> + From<Any> + AsAny + Clone {
    /// (no documentation yet)
    #[doc(hidden)] // returns a type from the `any` api
    fn ty() -> ir::Type;

    /// (no documentation yet)
    #[track_caller]
    fn from_any_unchecked(any: Any) -> Self;

    /// (no documentation yet)
    #[track_caller]
    fn from_any(any: Any) -> Self { typecheck_downcast(any, Self::ty(), Self::from_any_unchecked) }
}

/// (no documentation yet)
pub trait AsAny {
    /// (no documentation yet)
    fn as_any(&self) -> Any;
}

#[track_caller]
pub(crate) fn typecheck_downcast<T>(any: Any, expected_ty: Type, from_any_unchecked: impl Fn(Any) -> T) -> T {
    let call_info = call_info!();
    match any.ty() {
        Some(dynamic_type) => match dynamic_type == expected_ty {
            true => from_any_unchecked(any),
            false => {
                let invalid = Context::try_with(call_info, |ctx| {
                    ctx.push_error(
                        FrontendError::InvalidDowncast {
                            dynamic_type,
                            rust_type: expected_ty,
                        }
                        .into(),
                    );
                    Any::new_invalid(InvalidReason::ErrorThatWasPushed)
                })
                .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding));
                from_any_unchecked(invalid)
            }
        },
        None => {
            // a `None` typed value means `any` is "invalid" which is the result
            // of previous shader errors. Turning invalid `Any`s into `Value`
            // types is ok, think of this like `.and_then` with the `Option` monad.
            from_any_unchecked(any)
        }
    }
}

/// (no documentation yet)
#[diagnostic::on_unimplemented(message = "`{Self}` cannot be converted to a [`GpuType`]")]
pub trait ToGpuType {
    /// (no documentation yet)
    type Gpu: GpuType;

    /// turn `self` into its corresponding [`GpuType`]
    #[track_caller]
    fn to_gpu(&self) -> Self::Gpu;

    /// (no documentation yet)
    #[track_caller]
    fn to_any(&self) -> Any { self.to_gpu().as_any() }

    /// (no documentation yet)
    #[track_caller]
    fn as_gpu_type_ref(&self) -> Option<&Self::Gpu> { None }

    /// convenience function for [`shame::Cell::new(...)`]
    ///
    /// ## mutable state in shaders
    /// allocates space in the shader's function address space [`mem::Fn`],
    /// initializes it with `init` and returns a read-write reference to that space.
    ///
    /// ## performance
    /// This kind of memory allocation is not like `Box::new` on the CPU.
    /// It is not using a heap.
    /// Its closest counterpart on the CPU would be a stack allocation.
    ///
    /// This allocation in the [`mem::Fn`] address space corresponds to
    /// - `GLSL`: variable declared in function
    /// - `Spir-V`: "Function" storage class
    /// - `WGSL`: "function" address space/variable declared in function
    ///
    /// [`shame::Cell::new(...)`]: crate::Cell::new
    #[track_caller]
    fn cell(self) -> Ref<Self::Gpu, mem::Fn>
    where
        Self::Gpu: GpuStore + GpuSized,
        Self: std::marker::Sized,
    {
        Cell::new(self)
    }
}

/// shorthand for [`ToGpuType<Gpu = T>`]
// TODO(docs) copy the documentation of `ToGpuType` in here
#[diagnostic::on_unimplemented(
    message = "A `{T}` shader value cannot be created from a `{Self}`",
    label = "the trait `To<{T}>` is not implemented for `{Self}`",
    note = "`To<{T}>` is a shorthand for `ToGpuType<Gpu={T}>`"
)]
pub trait To<T: GpuType>: ToGpuType<Gpu = T> {}
impl<G: GpuType, T: ToGpuType<Gpu = G>> To<G> for T {}
