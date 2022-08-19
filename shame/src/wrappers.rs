use crate::rec::Rec;

/// A wrapper type that requires the user to open up an `unsafe` block to access the
/// inner type via `access()` or `.access_mut()`. Incorrect usage of the wrapped
/// type can result in a shader that creates memory race conditions on the GPU.
pub struct UnsafeAccess<T>(T);

impl<T> UnsafeAccess<T> {
    #[cfg(feature = "enable_unsafe_features")]
    /// obtain mutable access to the inner `T`
    ///
    /// while technically `unsafe` is not required here, the intention of this
    /// wrapper is to make the user aware that the contained type `T` can, when
    /// used incorrectly, create memory race conditions on the GPU if the final
    /// shader is being compiled and run.
    ///
    /// This code has no risk of creating any unsoundness at all in the usual
    /// sense, as it is not using any `unsafe` features.
    pub unsafe fn access_mut(&mut self) -> &mut T {
        &mut self.0
    }

    #[cfg(feature = "enable_unsafe_features")]
    /// obtain access to the inner `T`
    ///
    /// while technically `unsafe` is not required here, the intention of this
    /// wrapper is to make the user aware that the contained type `T` can, when
    /// used incorrectly, create memory race conditions on the GPU if the final
    /// shader is being compiled and run.
    ///
    /// This code has no risk of creating any unsoundness at all in the usual
    /// sense, as it is not using any `unsafe` features.
    pub unsafe fn access(&self) -> &T {
        &self.0
    }

    /// wrap `t` to be only accessible via an unsafe block
    pub fn new(t: T) -> Self {
        Self(t)
    }
}

impl<T: Rec> UnsafeAccess<T> {
    /// assign an identifier to the contained `T` value, which will be used if
    /// it shows up in the generated shader code (if possible)
    pub fn aka(&self, name: &str) {
        self.0.as_any().aka(name);
    }
}
