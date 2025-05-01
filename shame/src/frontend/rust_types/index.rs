/// Used for indexing operations on [`GpuType`]s.
///
/// This replaces rusts [`std::ops::Index`] trait, which cannot be used
/// for code generation, as it requires the return type to be a shared reference.
/// Therefore the `array[i]` syntax cannot be used.
///
/// # Example
/// `array.index(4u32)`
///
/// To reduce verbosity, a shorthand function `array.at(i)` exists for most
/// containers that implement [`GpuIndex`], which is equivalent to `array.index(i)`.
///
/// > maintainer note:
/// > We couldn't decide on an indexing naming scheme, so we expose both `.at(i)` and `.index(i)`.
/// > `.at(i)` naming has no precedent in the rust standard library, but is often used in other languages.
/// > `.index(i)` is used by rust, but never really written this way.
/// > `.get(i)` is used in too many contexts without arguments as we already have a
/// > lot of `.get()` functions required for interior mutability and push constants.
/// > example: `array2_ref.get().get(i).get(j)`, `push_constants.get()`
///
/// [`GpuType`]: crate::GpuType
pub trait GpuIndex<Idx> {
    /// The returned type after indexing.
    type Output;

    /// look up the `index`th element.
    ///
    /// when used with either of
    /// - [`Array`]
    /// - [`vec`]
    /// - [`mat`] (access the `index`'th column vector)
    /// - [`TextureArray`]
    /// - [`StorageTextureArray`]
    ///
    /// out of bounds accesses will result in indeterminate values, but won't
    /// trigger undefined behavior (WGSL).
    ///
    /// [`Array`]: crate::Array
    /// [`vec`]: crate::vec
    /// [`mat`]: crate::mat
    /// [`TextureArray`]: crate::TextureArray
    /// [`StorageTextureArray`]: crate::StorageTextureArray
    #[track_caller]
    fn index(&self, index: Idx) -> Self::Output;
}
