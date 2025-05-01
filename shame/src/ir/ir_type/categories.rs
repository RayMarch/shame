use crate::{frontend::any::shared_io::BufferBindingType, ir::Type};

use super::{ScalarType, SizedType, StoreType};

impl Type {
    #[allow(missing_docs)] // runtime api
    pub fn is_host_shareable(&self) -> bool {
        match self {
            Type::Store(store) => store.is_host_shareable(),
            _ => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_constructible(&self) -> bool {
        match self {
            Type::Store(store) => store.is_constructible(),
            _ => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_creation_fixed_footprint(&self) -> bool {
        match self {
            Type::Store(store) => store.is_creation_fixed_footprint(),
            _ => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_plain_and_fixed_footprint(&self) -> bool {
        match self {
            Type::Store(store) => store.is_plain_and_fixed_footprint(),
            _ => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn contains_atomics(&self) -> bool {
        match self {
            Type::Store(store) => store.contains_atomics(),
            _ => false,
        }
    }
}

impl StoreType {
    // https://www.w3.org/TR/WGSL/#host-shareable-types
    // every `SizedType` or `RuntimeSizedArray`
    // that doesn't contain any `bool`s.
    #[allow(missing_docs)] // runtime api
    pub fn is_host_shareable(&self) -> bool {
        use StoreType as T;
        match self {
            T::Sized(s) | T::RuntimeSizedArray(s) => s.is_host_shareable(),
            T::BufferBlock(def) => def.is_host_shareable(),
            T::Handle(_) => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_creation_fixed_footprint(&self) -> bool {
        use StoreType::*;
        match self {
            Sized(_) => true,
            RuntimeSizedArray(_) | Handle(_) => false,
            BufferBlock(block) => block.is_creation_fixed_footprint(),
        }
    }

    /// see https://www.w3.org/TR/WGSL/#fixed-footprint-types
    pub fn is_plain_and_fixed_footprint(&self) -> bool {
        use StoreType::*;
        match self {
            Sized(_) => true,
            RuntimeSizedArray(_) | Handle(_) => false,
            BufferBlock(blk) => blk.is_fixed_footprint(),
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn contains_atomics(&self) -> bool {
        match self {
            StoreType::Sized(s) | StoreType::RuntimeSizedArray(s) => s.contains_atomics(),
            StoreType::Handle(_) => false,
            StoreType::BufferBlock(blk) => blk.contains_atomics(),
        }
    }

    /// see https://www.w3.org/TR/WGSL/#constructible-types
    pub fn is_constructible(&self) -> bool {
        use StoreType::*;
        match self {
            Sized(sized) => sized.is_constructible(),
            RuntimeSizedArray(_) | Handle(_) => false,
            BufferBlock(blk) => blk.is_constructible(),
        }
    }
}

impl SizedType {
    #[allow(missing_docs)] // runtime api
    pub fn is_host_shareable(&self) -> bool {
        use SizedType as T;
        match self {
            T::Vector(_, s) => s.is_host_shareable(),
            T::Matrix(_, _, s) => ScalarType::from(*s).is_host_shareable(),
            T::Atomic(s) => ScalarType::from(*s).is_host_shareable(),
            T::Array(e, _) => e.is_host_shareable(),
            T::Structure(s) => s.fields().all(|t| t.ty().is_host_shareable()),
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn contains_atomics(&self) -> bool {
        use SizedType as T;
        match self {
            T::Atomic(_) => true,
            T::Vector(_, _) | T::Matrix(_, _, _) => false,
            T::Array(e, _) => e.contains_atomics(),
            T::Structure(s) => s.contains_atomics(),
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_constructible(&self) -> bool {
        match self {
            SizedType::Vector(_, _) => true,
            SizedType::Matrix(_, _, _) => true,
            SizedType::Array(elem, _) => elem.is_constructible(),
            SizedType::Atomic(_) => false,
            SizedType::Structure(sized_struct) => sized_struct.fields().all(|f| f.ty.is_constructible()),
        }
    }
}

impl ScalarType {
    #[allow(missing_docs)] // runtime api
    pub fn is_host_shareable(&self) -> bool {
        use ScalarType as T;
        match self {
            T::F16 | T::F32 | T::F64 | T::U32 | T::I32 => true,
            T::Bool => false,
        }
    }
}
