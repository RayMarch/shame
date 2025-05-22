#![allow(missing_docs)]
use std::{num::NonZeroU32, rc::Rc};

use crate::{
    any::U32PowerOf2,
    ir::{self, ir_type::BufferBlockDefinitionError, StructureFieldNamesMustBeUnique},
    GpuAligned, GpuSized, GpuStore, GpuType, NoBools, NoHandles,
};

// TODO(chronicl)
// - Consider moving this module into ir_type?
// - We borrow these types from `StoreType` currently. Maybe it would be better the other
//   way around - `StoreType` should borrow from `HostShareableType`.
pub use crate::ir::{Len, Len2, LenEven, PackedVector, ScalarTypeFp, ScalarTypeInteger, ir_type::CanonName};
use super::{constraint, FieldOptions};

mod align_size;
mod builder;

pub use align_size::*;
pub use builder::*;

// TODO(chronicl) rewrite
/// This reprsents a wgsl spec compliant host-shareable type with the addition
/// that f64 is a supported scalar type.
///
/// https://www.w3.org/TR/WGSL/#host-shareable-types
#[derive(Debug, Clone)]
pub enum LayoutType {
    Sized(SizedType),
    UnsizedStruct(UnsizedStruct),
    RuntimeSizedArray(RuntimeSizedArray),
}

#[derive(Debug, Clone)]
pub enum SizedType {
    Vector(Vector),
    Matrix(Matrix),
    Array(SizedArray),
    Atomic(Atomic),
    PackedVec(PackedVector),
    Struct(SizedStruct),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vector {
    pub scalar: ScalarType,
    pub len: Len,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Matrix {
    pub scalar: ScalarTypeFp,
    pub columns: Len2,
    pub rows: Len2,
}

#[derive(Debug, Clone)]
pub struct SizedArray {
    pub element: Rc<SizedType>,
    pub len: NonZeroU32,
}

#[derive(Debug, Clone, Copy)]
pub struct Atomic {
    pub scalar: ScalarTypeInteger,
}

#[derive(Debug, Clone)]
pub struct RuntimeSizedArray {
    pub element: SizedType,
}

/// Same as `ir::ScalarType`, but without `ScalarType::Bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    F16,
    F32,
    U32,
    I32,
    F64,
}

#[derive(Debug, Clone)]
pub struct SizedStruct {
    pub name: CanonName,
    // This is private to ensure a `SizedStruct` always has at least one field.
    fields: Vec<SizedField>,
}

#[derive(Debug, Clone)]
pub struct UnsizedStruct {
    pub name: CanonName,
    pub sized_fields: Vec<SizedField>,
    pub last_unsized: RuntimeSizedArrayField,
}

#[derive(Debug, Clone)]
pub struct SizedField {
    pub name: CanonName,
    pub custom_min_size: Option<u64>,
    pub custom_min_align: Option<U32PowerOf2>,
    pub ty: SizedType,
}

#[derive(Debug, Clone)]
pub struct RuntimeSizedArrayField {
    pub name: CanonName,
    pub custom_min_align: Option<U32PowerOf2>,
    pub array: RuntimeSizedArray,
}

impl SizedField {
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
    pub fn new(element_ty: impl Into<SizedType>, len: NonZeroU32) -> Self {
        Self {
            element: Rc::new(element_ty.into()),
            len,
        }
    }
}

impl RuntimeSizedArray {
    pub fn new(element_ty: impl Into<SizedType>) -> Self {
        RuntimeSizedArray {
            element: element_ty.into(),
        }
    }
}

pub enum SizedOrArray {
    Sized(SizedType),
    RuntimeSizedArray(RuntimeSizedArray),
}

#[derive(thiserror::Error, Debug)]
#[error("`LayoutType` is `UnsizedStruct`, which is not a variant of `SizedOrArray`")]
pub struct IsUnsizedStruct;
impl TryFrom<LayoutType> for SizedOrArray {
    type Error = IsUnsizedStruct;

    fn try_from(value: LayoutType) -> Result<Self, Self::Error> {
        match value {
            LayoutType::Sized(sized) => Ok(SizedOrArray::Sized(sized)),
            LayoutType::RuntimeSizedArray(array) => Ok(SizedOrArray::RuntimeSizedArray(array)),
            LayoutType::UnsizedStruct(_) => Err(IsUnsizedStruct),
        }
    }
}

// TODO(chronicl) documentation
pub trait BinaryRepr {
    fn layout_type() -> LayoutType;
}
pub trait BinaryReprSized: BinaryRepr {
    fn layout_type_sized() -> SizedType;
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

//   Conversions to ScalarType, SizedType and HostshareableType   //

macro_rules! impl_into_sized_type {
    ($($ty:ident -> $variant:path),*) => {
       $(
           impl $ty {
               /// Const conversion to [`SizedType`]
               pub const fn into_sized_type(self) -> SizedType { $variant(self) }
               /// Const conversion to [`HostshareableType`]
               pub const fn into_cpu_shareable(self) -> LayoutType {
                   LayoutType::Sized(self.into_sized_type())
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

impl<T> From<T> for LayoutType
where
    SizedType: From<T>,
{
    fn from(value: T) -> Self { LayoutType::Sized(SizedType::from(value)) }
}

impl From<UnsizedStruct> for LayoutType {
    fn from(s: UnsizedStruct) -> Self { LayoutType::UnsizedStruct(s) }
}
impl From<RuntimeSizedArray> for LayoutType {
    fn from(a: RuntimeSizedArray) -> Self { LayoutType::RuntimeSizedArray(a) }
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

impl From<LayoutType> for ir::StoreType {
    fn from(host: LayoutType) -> Self {
        match host {
            LayoutType::Sized(s) => ir::StoreType::Sized(s.into()),
            LayoutType::RuntimeSizedArray(s) => ir::StoreType::RuntimeSizedArray(s.element.into()),
            LayoutType::UnsizedStruct(s) => ir::StoreType::BufferBlock(s.into()),
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
            // TODO(chronicl) check if this should be decompressed
            SizedType::PackedVec(v) => v.decompressed_ty(),
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

#[derive(thiserror::Error, Debug)]
#[error("Type contains bools, which isn't cpu shareable.")]
pub struct ContainsBools;

impl TryFrom<ir::ScalarType> for ScalarType {
    type Error = ContainsBools;

    fn try_from(value: ir::ScalarType) -> Result<Self, Self::Error> {
        Ok(match value {
            ir::ScalarType::F16 => ScalarType::F16,
            ir::ScalarType::F32 => ScalarType::F32,
            ir::ScalarType::F64 => ScalarType::F64,
            ir::ScalarType::U32 => ScalarType::U32,
            ir::ScalarType::I32 => ScalarType::I32,
            ir::ScalarType::Bool => return Err(ContainsBools),
        })
    }
}

impl ir::ScalarType {
    // TODO(chronicl) remove
    pub fn as_host_shareable_unchecked(self) -> ScalarType { self.try_into().unwrap() }
}

impl TryFrom<ir::SizedType> for SizedType {
    type Error = ContainsBools;

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
    type Error = ContainsBools;

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

#[derive(thiserror::Error, Debug)]
pub enum CpuShareableConversionError {
    #[error("Type contains bools, which isn't cpu shareable.")]
    ContainsBool,
    #[error("Type is a handle, which isn't cpu shareable.")]
    IsHandle,
}

impl From<ContainsBools> for CpuShareableConversionError {
    fn from(_: ContainsBools) -> Self { Self::ContainsBool }
}

impl TryFrom<ir::StoreType> for LayoutType {
    type Error = CpuShareableConversionError;

    fn try_from(value: ir::StoreType) -> Result<Self, Self::Error> {
        Ok(match value {
            ir::StoreType::Sized(sized_type) => LayoutType::Sized(sized_type.try_into()?),
            ir::StoreType::RuntimeSizedArray(element) => LayoutType::RuntimeSizedArray(RuntimeSizedArray {
                element: element.try_into()?,
            }),
            ir::StoreType::BufferBlock(buffer_block) => buffer_block.try_into()?,
            ir::StoreType::Handle(_) => return Err(CpuShareableConversionError::IsHandle),
        })
    }
}

impl TryFrom<ir::ir_type::BufferBlock> for LayoutType {
    type Error = ContainsBools;

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
