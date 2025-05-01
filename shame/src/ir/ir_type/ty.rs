use self::struct_::SizedStruct;
use crate::{
    frontend::{any::shared_io, rust_types::type_layout::TypeLayoutRules},
    ir::recording::MemoryRegion,
};

use super::*;
use std::{
    fmt::Display,
    num::{NonZeroU32, NonZeroU64},
    rc::Rc,
};

#[derive(Clone, PartialEq, Eq)]
// Types according to the WebGPU type system,
// slightly modified to include
// - `StoreType::BufferBlock` for potential OpenGL/GLSL compatibility
// - `Rc<Allocation>` for tracking memory cell accesses (useful for the stage solver)
#[doc(hidden)] // runtime api
pub enum Type {
    Unit,
    Ptr(Rc<MemoryRegion>, StoreType, AccessMode),
    Ref(Rc<MemoryRegion>, StoreType, AccessMode),
    Store(StoreType),
}

/// types that pointers/reference can point to.
#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StoreType {
    /// WGSL "creation-fixed-footprint"
    Sized(SizedType),
    Handle(HandleType),
    RuntimeSizedArray(SizedType),
    BufferBlock(BufferBlock),
}

/// types that have a size which is known at shader creation time.
/// WGSL "creation-fixed-footprint"
#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SizedType {
    /// Scalar or Vector
    ///
    /// A Scalar is represented as `Vector(Len::X1, _)`
    Vector(Len, ScalarType),
    Matrix(Len2, Len2, ScalarTypeFp),
    Array(Rc<SizedType>, NonZeroU32),
    Atomic(ScalarTypeInteger),
    Structure(SizedStruct),
}

/// types that represent handles to resources (Textures and Samplers).
#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HandleType {
    SampledTexture(TextureShape, TextureSampleUsageType, SamplesPerPixel),
    StorageTexture(TextureShape, TextureFormatWrapper, AccessMode),
    Sampler(shared_io::SamplingMethod),
}

impl From<SizedType> for StoreType {
    fn from(value: SizedType) -> Self { StoreType::Sized(value) }
}

impl From<SizedType> for Type {
    fn from(value: SizedType) -> Self { Type::Store(StoreType::Sized(value)) }
}

impl From<ScalarType> for Type {
    fn from(value: ScalarType) -> Self { Type::Store(StoreType::Sized(SizedType::Vector(Len::X1, value))) }
}

impl Type {
    pub fn is_ref(&self) -> bool { matches!(self, Type::Ref { .. }) }
}

impl std::fmt::Debug for Type {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// custom debug implementation to hide details of the allocation, and instead print refs and ptrs
        /// as expected from wgsl
        match self {
            Self::Unit => write!(f, "Unit"),
            Self::Ptr(arg0, arg1, arg2) => f.debug_tuple("Ptr").field(&arg0.address_space).field(arg1).field(arg2).finish(),
            Self::Ref(arg0, arg1, arg2) => f.debug_tuple("Ref").field(&arg0.address_space).field(arg1).field(arg2).finish(),
            Self::Store(arg0) => f.debug_tuple("Store").field(arg0).finish(),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_view_str = match self {
            Type::Ptr(..) => "Ptr",
            Type::Ref(..) => "Ref",
            _ => "",
        };
        match self {
            Type::Unit => f.write_str("()"),
            Type::Ptr(a, s, am) | Type::Ref(a, s, am) => write!(f, "{mem_view_str}<{s}, {}, {am}>", a.address_space),
            Type::Store(s) => write!(f, "{s}"),
        }
    }
}

impl Display for StoreType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreType::Sized(x) => write!(f, "{x}"),
            StoreType::Handle(x) => write!(f, "{x}"),
            StoreType::RuntimeSizedArray(x) => write!(f, "Array<{x}>"),
            StoreType::BufferBlock(x) => write!(f, "{}", x.name()),
        }
    }
}

impl Display for SizedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SizedType::Vector(l, t) => write!(f, "{t}{l}"),
            SizedType::Matrix(c, r, t) => {
                write!(f, "mat<{}, {}, {}>", ScalarType::from(*t), Len::from(*c), Len::from(*r))
            }
            SizedType::Array(t, n) => write!(f, "array<{t}, {n}>"),
            SizedType::Atomic(t) => write!(f, "atomic<{}>", ScalarType::from(*t)),
            SizedType::Structure(s) => write!(f, "{}", s.name()),
        }
    }
}

impl Display for HandleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandleType::SampledTexture(shape, fmt, spp) => write!(f, "Texture<{fmt}, {shape}, {spp:?}>"),
            HandleType::StorageTexture(shape, fmt, access) => write!(f, "StorageTexture<{fmt:?}, {shape}, {access}>"),
            HandleType::Sampler(s) => write!(f, "Sampler<{s}>"),
        }
    }
}

impl StoreType {
    pub fn min_byte_size(&self) -> Option<NonZeroU64> {
        match self {
            StoreType::Sized(sized_type) => NonZeroU64::new(sized_type.byte_size()),
            StoreType::Handle(handle_type) => None,
            StoreType::RuntimeSizedArray(sized_type) => NonZeroU64::new(sized_type.byte_size()),
            StoreType::BufferBlock(buffer_block) => NonZeroU64::new(buffer_block.min_byte_size()),
        }
    }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AlignedType {
    Sized(SizedType),
    RuntimeSizedArray(SizedType),
}

impl AlignedType {
    pub fn align(&self) -> u64 {
        match self {
            AlignedType::Sized(sized) => sized.align(),
            AlignedType::RuntimeSizedArray(sized) => align_of_array(sized),
        }
    }
}
