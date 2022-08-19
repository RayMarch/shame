//! helper types for the `keep_idents` proc macro
use super::rec::*;
use crate::{rec::fields::Fields, wrappers::UnsafeAccess, TexSampleType};

/// helper type which is necessary for the `keep_idents` proc macro
pub struct TryKeepIdent<'a, T>(pub &'a T);

impl<Out: TexSampleType, In: TexCoordType> TryKeepIdent<'_, CombineSampler<Out, In>> {
    #[allow(missing_docs)]
    pub fn store_ident(&self, name: &str) {
        self.0.any().aka(name);
    }
}

impl<T: Rec> TryKeepIdent<'_, UnsafeAccess<T>> {
    #[allow(missing_docs)]
    pub fn store_ident(&self, name: &str) {
        self.0.aka(name);
    }
}

impl<T: Rec, const N: usize> TryKeepIdent<'_, Array<T, Size<N>>> {
    #[allow(missing_docs)]

    pub fn store_ident(&self, name: &str) {
        self.0.as_any().aka(name);
    }
}

impl<T: Rec> TryKeepIdent<'_, Array<T, Unsized>> {
    #[allow(missing_docs)]
    pub fn store_ident(&self, name: &str) {
        self.0.as_any().aka(name);
    }
}

impl<T: Fields> TryKeepIdent<'_, Struct<T>> {
    #[allow(missing_docs)]
    pub fn store_ident(&self, name: &str) {
        self.0.as_any().aka(name);
    }
}

impl<S: Shape, D: DType> TryKeepIdent<'_, Ten<S, D>> {
    #[allow(missing_docs)]
    pub fn store_ident(&self, name: &str) {
        self.0.aka(name);
    }
}

impl<S: Shape, D: DType> TryKeepIdent<'_, WriteOnly<S, D>> {
    #[allow(missing_docs)]
    pub fn store_ident(&self, name: &str) {
        self.0.aka(name);
    }
}

/// helper trait used by the `keep_idents` proc macro
pub trait TryKeepIdentTrait {
    /// the default implementation will be applied to all types that aren't
    /// related to shader recording at all.
    fn store_ident(&self, _name: &str) {
        //noop
    }
}

impl<T> TryKeepIdentTrait for TryKeepIdent<'_, T> {}
