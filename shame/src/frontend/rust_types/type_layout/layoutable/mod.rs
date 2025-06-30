//! This module defines types that can be laid out in memory.

use std::{fmt::Formatter, num::NonZeroU32, rc::Rc};

use crate::{
    any::U32PowerOf2,
    call_info,
    common::prettify::set_color,
    ir::{self, ir_type::BufferBlockDefinitionError, recording::Context, StructureFieldNamesMustBeUnique},
};

pub use crate::ir::{Len, Len2, PackedVector, ScalarTypeFp, ScalarTypeInteger, ir_type::CanonName};
use super::{construction::StructKind};

pub(crate) mod align_size;
pub(crate) mod builder;
pub(crate) mod ir_compat;

pub use align_size::{FieldOffsets, MatrixMajor, LayoutCalculator, array_size, array_stride, array_align};
pub use builder::{SizedOrArray, FieldOptions};

/// Types that can be layed out in memory.
///
/// `LayoutableType` does not contain any layout information itself, but a layout
/// can be assigned to it using [`GpuTypeLayout`] according to one of the available layout rules:
/// storage, uniform or packed.
#[derive(Debug, Clone)]
pub enum LayoutableType {
    /// A type with a known size.
    Sized(SizedType),
    /// A struct with a runtime sized array as it's last field.
    UnsizedStruct(UnsizedStruct),
    /// An array whose size is determined at runtime.
    RuntimeSizedArray(RuntimeSizedArray),
}

/// Types that have a size which is known at shader creation time.
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub enum SizedType {
    Vector(Vector),
    Matrix(Matrix),
    Array(SizedArray),
    Atomic(Atomic),
    PackedVec(PackedVector),
    Struct(SizedStruct),
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vector {
    pub scalar: ScalarType,
    pub len: Len,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Matrix {
    pub scalar: ScalarTypeFp,
    pub columns: Len2,
    pub rows: Len2,
}

#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct SizedArray {
    pub element: Rc<SizedType>,
    pub len: NonZeroU32,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy)]
pub struct Atomic {
    pub scalar: ScalarTypeInteger,
}

#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct RuntimeSizedArray {
    pub element: SizedType,
}

/// Scalar types with known memory layout.
///
/// Same as `ir::ScalarType`, but without `ScalarType::Bool` since booleans
/// don't have a standardized memory representation.
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    F16,
    F32,
    U32,
    I32,
    F64,
}

/// A struct with a known fixed size.
#[derive(Debug, Clone)]
pub struct SizedStruct {
    /// The canonical name of the struct.
    pub name: CanonName,
    // This is private to ensure a `SizedStruct` always has at least one field.
    fields: Vec<SizedField>,
}

/// A struct whose size is not known before shader runtime.
///
/// This struct has a runtime sized array as it's last field.
#[derive(Debug, Clone)]
pub struct UnsizedStruct {
    /// The canonical name of the struct.
    pub name: CanonName,
    /// Fixed-size fields that come before the unsized field
    pub sized_fields: Vec<SizedField>,
    /// Last runtime sized array field of the struct.
    pub last_unsized: RuntimeSizedArrayField,
}

#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct SizedField {
    pub name: CanonName,
    pub custom_min_size: Option<u64>,
    pub custom_min_align: Option<U32PowerOf2>,
    pub ty: SizedType,
}

#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct RuntimeSizedArrayField {
    pub name: CanonName,
    pub custom_min_align: Option<U32PowerOf2>,
    pub array: RuntimeSizedArray,
}

/// Trait for types that have a well-defined memory layout.
pub trait Layoutable {
    /// Returns the `LayoutableType` representation for this type.
    fn layoutable_type() -> LayoutableType;
}
/// Trait for types that have a well-defined memory layout and statically known size.
pub trait LayoutableSized: Layoutable {
    /// Returns the `SizedType` representation for this type.
    fn layoutable_type_sized() -> SizedType;
}

//   Conversions to ScalarType, SizedType and LayoutableType   //

macro_rules! impl_into_sized_type {
    ($($ty:ident -> $variant:path),*) => {
       $(
           impl $ty {
               /// Const conversion to [`SizedType`]
               pub const fn into_sized_type(self) -> SizedType { $variant(self) }
               /// Const conversion to [`LayoutableType`]
               pub const fn into_layoutable_type(self) -> LayoutableType {
                   LayoutableType::Sized(self.into_sized_type())
               }
           }

           impl From<$ty> for SizedType {
               fn from(v: $ty) -> Self { v.into_sized_type() }
           }
       )*
    };
}
impl_into_sized_type!(
    Vector       -> SizedType::Vector,
    Matrix       -> SizedType::Matrix,
    SizedArray   -> SizedType::Array,
    Atomic       -> SizedType::Atomic,
    SizedStruct  -> SizedType::Struct,
    PackedVector -> SizedType::PackedVec
);

impl<T> From<T> for LayoutableType
where
    SizedType: From<T>,
{
    fn from(value: T) -> Self { LayoutableType::Sized(SizedType::from(value)) }
}

impl From<UnsizedStruct> for LayoutableType {
    fn from(s: UnsizedStruct) -> Self { LayoutableType::UnsizedStruct(s) }
}
impl From<RuntimeSizedArray> for LayoutableType {
    fn from(a: RuntimeSizedArray) -> Self { LayoutableType::RuntimeSizedArray(a) }
}

impl ScalarTypeInteger {
    pub const fn as_scalar_type(self) -> ScalarType {
        match self {
            ScalarTypeInteger::I32 => ScalarType::I32,
            ScalarTypeInteger::U32 => ScalarType::U32,
        }
    }
}
impl From<ScalarTypeInteger> for ScalarType {
    fn from(int: ScalarTypeInteger) -> Self { int.as_scalar_type() }
}
impl ScalarTypeFp {
    pub const fn as_scalar_type(self) -> ScalarType {
        match self {
            ScalarTypeFp::F16 => ScalarType::F16,
            ScalarTypeFp::F32 => ScalarType::F32,
            ScalarTypeFp::F64 => ScalarType::F64,
        }
    }
}
impl From<ScalarTypeFp> for ScalarType {
    fn from(int: ScalarTypeFp) -> Self { int.as_scalar_type() }
}

// Display impls

impl std::fmt::Display for LayoutableType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutableType::Sized(s) => s.fmt(f),
            LayoutableType::RuntimeSizedArray(a) => a.fmt(f),
            LayoutableType::UnsizedStruct(s) => s.fmt(f),
        }
    }
}

impl std::fmt::Display for SizedType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SizedType::Vector(v) => v.fmt(f),
            SizedType::Matrix(m) => m.fmt(f),
            SizedType::Array(a) => a.fmt(f),
            SizedType::Atomic(a) => a.fmt(f),
            SizedType::PackedVec(p) => p.fmt(f),
            SizedType::Struct(s) => s.fmt(f),
        }
    }
}

impl std::fmt::Display for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}{}", self.scalar, self.len) }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mat<{}, {}, {}>",
            self.scalar,
            Len::from(self.columns),
            Len::from(self.rows)
        )
    }
}

impl std::fmt::Display for SizedArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "Array<{}, {}>", &*self.element, self.len) }
}

impl std::fmt::Display for Atomic {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "Atomic<{}>", ScalarType::from(self.scalar)) }
}

impl std::fmt::Display for SizedStruct {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.name) }
}

impl std::fmt::Display for UnsizedStruct {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.name) }
}

impl std::fmt::Display for RuntimeSizedArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "Array<{}>", self.element) }
}

impl std::fmt::Display for ScalarType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ScalarType::F16 => "f16",
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
            ScalarType::U32 => "u32",
            ScalarType::I32 => "i32",
        })
    }
}
