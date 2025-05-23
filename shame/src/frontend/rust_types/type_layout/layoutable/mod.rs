//! This module defines types that can be laid out in memory.

use std::{num::NonZeroU32, rc::Rc};

use crate::{
    any::U32PowerOf2,
    ir::{self, ir_type::BufferBlockDefinitionError, StructureFieldNamesMustBeUnique},
};

pub use crate::ir::{Len, Len2, PackedVector, ScalarTypeFp, ScalarTypeInteger, ir_type::CanonName};
use super::FieldOptions;

pub(crate) mod align_size;
pub(crate) mod builder;

pub use align_size::{FieldOffsets, MatrixMajor, array_size, array_stride, array_align};
pub use builder::SizedOrArray;

/// Types that have a defined memory layout.
///
/// `LayoutableType` does not contain any layout information itself, but a layout
/// can be assigned to it using [`TypeLayout`] according to one of the available layout rules:
/// storage, uniform or packed.
#[derive(Debug, Clone)]
pub enum LayoutableType {
    /// A type with a statically known size.
    Sized(SizedType),
    /// A struct with a runtime sized array as it's last field.
    UnsizedStruct(UnsizedStruct),
    /// An array whose size is determined at runtime.
    RuntimeSizedArray(RuntimeSizedArray),
}

/// Types with a statically known size.
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

/// A struct whose size is not known at compile time.
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

impl SizedField {
    /// Creates a new `SizedField`.
    pub fn new(options: impl Into<FieldOptions>, ty: impl Into<SizedType>) -> Self {
        let options = options.into();
        Self {
            name: options.name,
            custom_min_size: options.custom_min_size,
            custom_min_align: options.custom_min_align,
            ty: ty.into(),
        }
    }
}

impl RuntimeSizedArrayField {
    /// Creates a new `RuntimeSizedArrayField` given it's field name,
    /// an optional custom minimum align and it's element type.
    pub fn new(
        name: impl Into<CanonName>,
        custom_min_align: Option<U32PowerOf2>,
        element_ty: impl Into<SizedType>,
    ) -> Self {
        Self {
            name: name.into(),
            custom_min_align,
            array: RuntimeSizedArray {
                element: element_ty.into(),
            },
        }
    }
}

impl SizedArray {
    /// Creates a new `RuntimeSizedArray` from it's element type and length.
    pub fn new(element_ty: impl Into<SizedType>, len: NonZeroU32) -> Self {
        Self {
            element: Rc::new(element_ty.into()),
            len,
        }
    }
}

impl RuntimeSizedArray {
    /// Creates a new `RuntimeSizedArray` from it's element type.
    pub fn new(element_ty: impl Into<SizedType>) -> Self {
        RuntimeSizedArray {
            element: element_ty.into(),
        }
    }
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


impl std::fmt::Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ScalarType::F16 => "f16",
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
            ScalarType::U32 => "u32",
            ScalarType::I32 => "i32",
        })
    }
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


//     Conversions to ir types     //

impl From<LayoutableType> for ir::StoreType {
    fn from(host: LayoutableType) -> Self {
        match host {
            LayoutableType::Sized(s) => ir::StoreType::Sized(s.into()),
            LayoutableType::RuntimeSizedArray(s) => ir::StoreType::RuntimeSizedArray(s.element.into()),
            LayoutableType::UnsizedStruct(s) => ir::StoreType::BufferBlock(s.into()),
        }
    }
}

impl From<SizedType> for ir::SizedType {
    fn from(host: SizedType) -> Self {
        match host {
            SizedType::Vector(v) => ir::SizedType::Vector(v.len, v.scalar.into()),
            SizedType::Matrix(m) => ir::SizedType::Matrix(m.columns, m.rows, m.scalar),
            SizedType::Array(a) => ir::SizedType::Array(Rc::new(Rc::unwrap_or_clone(a.element).into()), a.len),
            SizedType::Atomic(i) => ir::SizedType::Atomic(i.scalar),
            SizedType::PackedVec(v) => SizedType::Vector(match v.byte_size() {
                ir::ir_type::PackedVectorByteSize::_2 => Vector::new(ScalarType::F16, Len::X1),
                ir::ir_type::PackedVectorByteSize::_4 => Vector::new(ScalarType::U32, Len::X1),
                ir::ir_type::PackedVectorByteSize::_8 => Vector::new(ScalarType::U32, Len::X2),
            })
            .into(),
            SizedType::Struct(s) => ir::SizedType::Structure(s.into()),
        }
    }
}

impl From<ScalarType> for ir::ScalarType {
    fn from(scalar_type: ScalarType) -> Self {
        match scalar_type {
            ScalarType::F16 => ir::ScalarType::F16,
            ScalarType::F32 => ir::ScalarType::F32,
            ScalarType::F64 => ir::ScalarType::F64,
            ScalarType::U32 => ir::ScalarType::U32,
            ScalarType::I32 => ir::ScalarType::I32,
        }
    }
}

impl From<SizedStruct> for ir::ir_type::SizedStruct {
    fn from(host: SizedStruct) -> Self {
        let mut fields: Vec<ir::ir_type::SizedField> = host.fields.into_iter().map(Into::into).collect();
        // has at least one field
        let last_field = fields.pop().unwrap();

        // Note: This might throw an error in real usage if the fields aren't valid
        // We're assuming they're valid in the conversion
        match ir::ir_type::SizedStruct::new_nonempty(host.name, fields, last_field) {
            Ok(s) => s,
            Err(StructureFieldNamesMustBeUnique) => {
                // TODO(chronicl) this isn't true
                unreachable!("field names are unique for `LayoutType`")
            }
        }
    }
}

impl From<UnsizedStruct> for ir::ir_type::BufferBlock {
    fn from(host: UnsizedStruct) -> Self {
        let sized_fields: Vec<ir::ir_type::SizedField> = host.sized_fields.into_iter().map(Into::into).collect();

        let last_unsized = host.last_unsized.into();

        // TODO(chronicl)
        // Note: This might throw an error in real usage if the struct isn't valid
        // We're assuming it's valid in the conversion
        match ir::ir_type::BufferBlock::new(host.name, sized_fields, Some(last_unsized)) {
            Ok(b) => b,
            Err(BufferBlockDefinitionError::FieldNamesMustBeUnique) => {
                // TODO(chronicl) this isn't true
                unreachable!("field names are unique for `UnsizedStruct`")
            }
            Err(BufferBlockDefinitionError::MustHaveAtLeastOneField) => {
                unreachable!("`UnsizedStruct` has at least one field.")
            }
        }
    }
}

impl From<RuntimeSizedArray> for ir::StoreType {
    fn from(array: RuntimeSizedArray) -> Self { ir::StoreType::RuntimeSizedArray(array.element.into()) }
}

impl From<SizedField> for ir::ir_type::SizedField {
    fn from(f: SizedField) -> Self { ir::SizedField::new(f.name, f.custom_min_size, f.custom_min_align, f.ty.into()) }
}

impl From<RuntimeSizedArrayField> for ir::ir_type::RuntimeSizedArrayField {
    fn from(f: RuntimeSizedArrayField) -> Self {
        ir::RuntimeSizedArrayField::new(f.name, f.custom_min_align, f.array.element.into())
    }
}


//     Conversions from ir types     //

/// Type contains bools, which doesn't have a known layout.
#[derive(thiserror::Error, Debug)]
#[error("Type contains bools, which doesn't have a known layout.")]
pub struct ContainsBoolsError;

impl TryFrom<ir::ScalarType> for ScalarType {
    type Error = ContainsBoolsError;

    fn try_from(value: ir::ScalarType) -> Result<Self, Self::Error> {
        Ok(match value {
            ir::ScalarType::F16 => ScalarType::F16,
            ir::ScalarType::F32 => ScalarType::F32,
            ir::ScalarType::F64 => ScalarType::F64,
            ir::ScalarType::U32 => ScalarType::U32,
            ir::ScalarType::I32 => ScalarType::I32,
            ir::ScalarType::Bool => return Err(ContainsBoolsError),
        })
    }
}

impl TryFrom<ir::SizedType> for SizedType {
    type Error = ContainsBoolsError;

    fn try_from(value: ir::SizedType) -> Result<Self, Self::Error> {
        Ok(match value {
            ir::SizedType::Vector(len, scalar) => SizedType::Vector(Vector {
                scalar: scalar.try_into()?,
                len,
            }),
            ir::SizedType::Matrix(columns, rows, scalar) => SizedType::Matrix(Matrix { scalar, columns, rows }),
            ir::SizedType::Array(element, len) => SizedType::Array(SizedArray {
                element: Rc::new((*element).clone().try_into()?),
                len,
            }),
            ir::SizedType::Atomic(scalar_type) => SizedType::Atomic(Atomic { scalar: scalar_type }),
            ir::SizedType::Structure(structure) => SizedType::Struct(structure.try_into()?),
        })
    }
}

impl TryFrom<ir::ir_type::SizedStruct> for SizedStruct {
    type Error = ContainsBoolsError;

    fn try_from(structure: ir::ir_type::SizedStruct) -> Result<Self, Self::Error> {
        let mut fields = Vec::new();

        for field in structure.sized_fields() {
            fields.push(SizedField {
                name: field.name.clone(),
                custom_min_size: field.custom_min_size,
                custom_min_align: field.custom_min_align,
                ty: field.ty.clone().try_into()?,
            });
        }

        Ok(SizedStruct {
            name: structure.name().clone(),
            fields,
        })
    }
}

/// Errors that can occur when converting IR types to layoutable types.
#[derive(thiserror::Error, Debug)]
pub enum LayoutableConversionError {
    /// Type contains bools, which don't have a standardized memory layout.
    #[error("Type contains bools, which don't have a standardized memory layout.")]
    ContainsBool,
    /// Type is a handle, which don't have a standardized memory layout.
    #[error("Type is a handle, which don't have a standardized memory layout.")]
    IsHandle,
}

impl From<ContainsBoolsError> for LayoutableConversionError {
    fn from(_: ContainsBoolsError) -> Self { Self::ContainsBool }
}

impl TryFrom<ir::StoreType> for LayoutableType {
    type Error = LayoutableConversionError;

    fn try_from(value: ir::StoreType) -> Result<Self, Self::Error> {
        Ok(match value {
            ir::StoreType::Sized(sized_type) => LayoutableType::Sized(sized_type.try_into()?),
            ir::StoreType::RuntimeSizedArray(element) => LayoutableType::RuntimeSizedArray(RuntimeSizedArray {
                element: element.try_into()?,
            }),
            ir::StoreType::BufferBlock(buffer_block) => buffer_block.try_into()?,
            ir::StoreType::Handle(_) => return Err(LayoutableConversionError::IsHandle),
        })
    }
}

impl TryFrom<ir::ir_type::BufferBlock> for LayoutableType {
    type Error = ContainsBoolsError;

    fn try_from(buffer_block: ir::ir_type::BufferBlock) -> Result<Self, Self::Error> {
        let mut sized_fields = Vec::new();

        for field in buffer_block.sized_fields() {
            sized_fields.push(SizedField {
                name: field.name.clone(),
                custom_min_size: field.custom_min_size,
                custom_min_align: field.custom_min_align,
                ty: field.ty.clone().try_into()?,
            });
        }

        let last_unsized = if let Some(last_field) = buffer_block.last_unsized_field() {
            RuntimeSizedArrayField {
                name: last_field.name.clone(),
                custom_min_align: last_field.custom_min_align,
                array: RuntimeSizedArray {
                    element: last_field.element_ty.clone().try_into()?,
                },
            }
        } else {
            return Ok(SizedStruct {
                name: buffer_block.name().clone(),
                fields: sized_fields,
            }
            .into());
        };

        Ok(UnsizedStruct {
            name: buffer_block.name().clone(),
            sized_fields,
            last_unsized,
        }
        .into())
    }
}

// Display impls

impl std::fmt::Display for LayoutableType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutableType::Sized(s) => write!(f, "{s}"),
            LayoutableType::RuntimeSizedArray(a) => write!(f, "Array<{}>", a.element),
            LayoutableType::UnsizedStruct(s) => write!(f, "{}", s.name),
        }
    }
}

impl std::fmt::Display for SizedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SizedType::Vector(v) => write!(f, "{}{}", ir::ScalarType::from(v.scalar), v.len),
            SizedType::Matrix(m) => {
                write!(
                    f,
                    "mat<{}, {}, {}>",
                    ir::ScalarType::from(m.scalar),
                    Len::from(m.columns),
                    Len::from(m.rows)
                )
            }
            SizedType::Array(a) => write!(f, "Array<{}, {}>", &*a.element, a.len),
            SizedType::Atomic(a) => write!(f, "Atomic<{}>", ir::ScalarType::from(a.scalar)),
            // TODO(chronicl) figure out scalar type display
            SizedType::PackedVec(p) => write!(f, "PackedVec<{:?}, {}>", p.scalar_type, Len::from(p.len)),
            SizedType::Struct(s) => write!(f, "{}", s.name),
        }
    }
}
