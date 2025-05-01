use super::mem;
use super::reference::*;
use super::type_traits::GpuSized;
use super::type_traits::GpuStore;
use crate::frontend::any;
use crate::frontend::any::Any;
use crate::ir;
use std::marker::PhantomData;

/// ### the function's address space, memory is reclaimed after the block scope ends.
///
/// the closest CPU equivalent to this address space is the CPU call stack
///
/// in terms of other graphics libraries/languages
/// - `glsl`: local variables in funcitons
/// - `wgsl`: "function" address space
/// - `spir-v`: "function" storage class
#[derive(Clone, Copy)]
pub struct Fn(());
/// ### the thread's address space
///
/// memory in this address space is visible only to a single thread
///
/// in terms of other graphics libraries/languages
/// - `glsl`: non-uniform non-shared global variables
/// - `wgsl`: private address space
/// - `spir-v`: private storage class
#[derive(Clone, Copy)]
pub struct Thread(());
/// ### the workgroup's address space
///
/// only usable in compute shaders
///
/// memory in this address space is visible to all threads of a workgroup.
///
/// in terms of other graphics libraries/languages
/// - `glsl`: shared type qualifier https://www.khronos.org/opengl/wiki/Type_Qualifier_(GLSL)#Shared
/// - `wgsl`: workgroup address space
/// - `spir-v`: workgroup storage class
#[derive(Clone, Copy)]
pub struct WorkGroup(());
/// ### the address space for push constants
///
/// read-only, visible across all threads of a dispatch/drawcall
///
/// in terms of other graphics libraries/languages
/// - `glsl`: uniforms with `layout(push_constant)`
/// - `wgsl`: `var<push_constant>` (`wgpu` only)
/// - `spir-v`: PushConstant storage class
#[derive(Clone, Copy)]
pub struct PushConstant(());
/// ### the address space of uniform buffer bindings
///
/// read-only, visible across all threads of a dispatch/drawcall
///
/// This address space is called "uniform" for historic reasons, it is not
/// the only source of uniform values, and reading in it does not necessarily
/// produce uniform values (i.e. during array lookup, if the array index is
/// not uniform).
#[derive(Clone, Copy)]
pub struct Uniform(());
/// ### the address space of storage buffer bindings
///
/// readable and writeable, visible across all threads of a dispatch/drawcall
#[derive(Clone, Copy)]
pub struct Storage(());
/// ### the address space of texture-/sampler bindings
///
/// visible across all threads of a dispatch/drawcall
#[derive(Clone, Copy)]
pub struct Handle(());
/// ### the address space of pipeline outputs (color targets)
///
/// write-only
#[derive(Clone, Copy)]
pub struct Output(());

// TODO(release) seal trait
/// Marker-types representing address spaces on the Gpu.
///
/// must be one of:
/// - [mem::Fn] (allocable)
/// - [mem::Thread] (allocable)
/// - [mem::WorkGroup] (allocable, compute pipelines only)
/// - [mem::PushConstant]
/// - [mem::Uniform] (for buffer bindings)
/// - [mem::Storage] (for buffer bindings)
/// - [mem::Handle] (for texture/sampler bindings)
/// - [mem::Output] (for color targets)
pub trait AddressSpace: Copy {
    /// enum variant representing `Self`
    const ADDRESS_SPACE: ir::AddressSpace;
    /// the default [`AccessMode`] of memory in this address space
    type DefaultAccess: AccessMode;
}

use ir::AddressSpace as Enum;

use super::{reference::AccessMode, GpuType, ToGpuType};
#[rustfmt::skip] impl AddressSpace for Fn           {const ADDRESS_SPACE: Enum = Enum::Function    ; type DefaultAccess = ReadWrite;}
#[rustfmt::skip] impl AddressSpace for Thread      {const ADDRESS_SPACE: Enum = Enum::Thread     ; type DefaultAccess = ReadWrite;}
#[rustfmt::skip] impl AddressSpace for WorkGroup    {const ADDRESS_SPACE: Enum = Enum::WorkGroup   ; type DefaultAccess = ReadWrite;}
#[rustfmt::skip] impl AddressSpace for PushConstant {const ADDRESS_SPACE: Enum = Enum::PushConstant; type DefaultAccess = Read;}
#[rustfmt::skip] impl AddressSpace for Uniform      {const ADDRESS_SPACE: Enum = Enum::Uniform     ; type DefaultAccess = Read;}
#[rustfmt::skip] impl AddressSpace for Storage      {const ADDRESS_SPACE: Enum = Enum::Storage     ; type DefaultAccess = Read;}
#[rustfmt::skip] impl AddressSpace for Handle       {const ADDRESS_SPACE: Enum = Enum::Handle      ; type DefaultAccess = Read;}
#[rustfmt::skip] impl AddressSpace for Output       {const ADDRESS_SPACE: Enum = Enum::Output      ; type DefaultAccess = Write;}

/// [`AddressSpace`]s that support access mode `A`
#[diagnostic::on_unimplemented(message = "address space `{Self}` does not support `{A}` access")]
pub trait SupportsAccess<A: AccessMode> {}
impl<A: AccessMode> SupportsAccess<A> for Fn {}
impl<A: AccessMode> SupportsAccess<A> for Thread {}
impl<A: AccessMode> SupportsAccess<A> for WorkGroup {}
impl SupportsAccess<Read> for PushConstant {}
impl SupportsAccess<Read> for Uniform {}
impl<A: AccessMode> SupportsAccess<A> for Storage {}
impl<A: AccessMode> SupportsAccess<A> for Handle {}
impl SupportsAccess<Write> for Output {}

/// [`AddressSpace`]s which allow memory allocations in the pipeline encoding
pub trait Allocable: AddressSpace + SupportsAccess<ReadWrite> + SupportsAccess<Read> + SupportsAccess<Write> {}
impl Allocable for Fn {}
impl Allocable for Thread {}
impl Allocable for WorkGroup {} // (glsl "shared" - only allocatable in compute pipelines)

/// [`AddressSpace`]s which can contain atomic types and support atomic operations
#[diagnostic::on_unimplemented(message = "address space `{Self}` does not allow atomic operations.")]
pub trait AddressSpaceAtomic: AddressSpace {}
impl AddressSpaceAtomic for Storage {}
impl AddressSpaceAtomic for WorkGroup {}

// ## mutable state in shaders
// allocates space in the shader address space `AS`, initializes it with `init`
// and returns a read-write reference to that space. The returned references
// cannot dangle. Any `shame` code that would make them dangle results in a pipeline recording error.
//
// note for CPU programmers: these allocations are not comparable to slow heap
// allocations, the individual [`AddressSpace`]s can be roughly compared
// to rust counterparts like so:
// - [`mem::Fn`]: stack allocations
// - [`mem::Private`]: thread local memory
// - [`mem::WorkGroup`]: static global variables (except only visible to threads within the same workgroup)
//
// TODO(release) clean this up, `init` is only supported for non-workgroup allocs,
// so this interface is only valid for Fn (and private?) see https://www.w3.org/TR/WGSL/#var-and-value
// #[track_caller]
// fn alloc_impl<T: ToGpuType, AS: AddressSpace + Allocable>(init: T) -> Ref<T::Gpu, AS, AS::DefaultAccess>
// where
//     T::Gpu: GpuStore + GpuSized, //TODO(release) it appears WGSL accepts unsized array vars in function address space, investigate if this bound can be removed
// {
//     Any::alloc_in(AS::ADDRESS_SPACE, init.to_any()).into()
// }

//[old-doc] convenience function for `alloc_in::<_, mem::Fn>(...)`
//[old-doc]
//[old-doc] ## mutable state in shaders
//[old-doc] allocates space in the shader's function address space [`mem::Fn`],
//[old-doc] initializes it with `init` and returns a read-write reference to that space.
//[old-doc]
//[old-doc] ## performance
//[old-doc] This kind of memory allocation is not like `Box::new` on the CPU.
//[old-doc] It is not using a heap.
//[old-doc] Its closest counterpart on the CPU would be a stack allocation.
//[old-doc]
//[old-doc] This allocation in the [`mem::Fn`] address space corresponds to
//[old-doc] - `GLSL`: variable declared in function
//[old-doc] - `Spir-V`: "Function" storage class
//[old-doc] - `WGSL`: "function" address space/variable declared in function
/// (no documentation yet)
#[track_caller]
pub(crate) fn alloc<T: ToGpuType>(init: T) -> Ref<T::Gpu, Fn>
where
    T::Gpu: GpuStore + GpuSized,
{
    Any::alloc(T::Gpu::sized_ty(), init.to_any()).into()
}

// TODO(release) clean this up (delete it or keep it)
// (no documentation yet)
// #[track_caller]
// pub fn workgroup_local<T: ToGpuType>(init: T) -> Ref<T::Gpu, mem::WorkGroup>
// where
//     T::Gpu: GpuStore + GpuSized,
// {
//     alloc_impl::<_, WorkGroup>(init)
// }

/// allocate per-workgroup memory. Whether the memory returned is zero initialized
/// can be controlled via the [`Settings`] provided to the pipeline encoding guard.
///
/// Zero initialization is defined at https://www.w3.org/TR/WGSL/#zero-value-builtin-function
///
/// [`Settings`]: crate::Settings
#[track_caller]
pub fn workgroup_local<T: GpuType + GpuStore + GpuSized>() -> Ref<T, mem::WorkGroup> {
    // workgroup allocations are default initialized (or uninitialized), since they cannot take
    // an inintializer value if `T` contains atomics. Atomics are not values but
    // memory cells, so we would require some sort of mapping from composit types
    // that contain atomics to composit types that have a regular u32 or i32 in that
    // place. At the time of writing it appears that WGSL does not support this either.
    Any::alloc_default_in(ir::AddressSpace::WorkGroup, T::sized_ty()).into()
}

#[doc(hidden)] // TODO(release) not necessary when function recording doesn't exist yet, so we keep it private api until then
#[track_caller]
pub fn thread_local<T: ToGpuType>(init: T) -> Ref<T::Gpu, mem::Thread>
where
    T::Gpu: GpuStore + GpuSized,
{
    Any::alloc_explicit(ir::AddressSpace::Thread, T::Gpu::sized_ty(), Some(init.to_any())).into()
}

/// (no documentation yet)
pub struct Cell {
    private_ctor: (),
}

#[allow(clippy::new_ret_no_self)]
impl Cell {
    // TODO(docs) docs, mention that the cell only lives in the shader scope it was created,
    //       so passing it outside of a `shame::boolx1::then` closure for example makes it invalid
    /// (no documentation yet)
    #[track_caller]
    pub fn new<T: ToGpuType>(init: T) -> Ref<T::Gpu, Fn>
    where
        T::Gpu: GpuStore + GpuSized,
    {
        alloc(init)
    }
}
