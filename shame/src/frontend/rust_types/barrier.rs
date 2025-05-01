use crate::frontend::any::Any;
use crate::frontend::rust_types::reference::Ref;

use super::{
    mem,
    reference::{AccessMode, AccessModeReadable},
    type_traits::{GpuSized, GpuStore, NoAtomics},
    AsAny, GpuType,
};

/// synchronize memory and atomic operations in the [`mem::WorkGroup`] address space
///
/// see https://www.w3.org/TR/WGSL/#workgroupBarrier-builtin
#[track_caller]
pub fn workgroup() { Any::workgroup_barrier() }

/// synchronize memory operations in the [`mem::Handle`] address space
///
/// see https://www.w3.org/TR/WGSL/#textureBarrier-builtin
#[track_caller]
pub fn texture() { Any::texture_barrier() }

/// synchronize memory and atomic operations in the [`mem::Storage`] address space
///
/// see https://www.w3.org/TR/WGSL/#storageBarrier-builtin
#[track_caller]
pub fn storage() { Any::storage_barrier() }

// TODO(release) use this a bunch of times to see if this `Ref`-based interface even makes sense
/// returns `src`'s value per workgroup. `src` must be a per-workgroup value or coarser.
///
/// this function will cause thread synchronization
///
/// see https://www.w3.org/TR/WGSL/#workgroupUniformLoad-builtin
#[track_caller]
pub fn workgroup_uniform_load<T, AM>(src: Ref<T, mem::WorkGroup, AM>) -> T
where
    AM: AccessModeReadable,
    T: GpuType + GpuStore + GpuSized + NoAtomics,
{
    src.as_any().address().workgroup_uniform_load().into()
}
