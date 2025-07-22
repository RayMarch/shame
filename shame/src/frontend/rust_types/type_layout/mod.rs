#![allow(missing_docs)]
//! Everything related to type layouts.

use std::{
    fmt::{Debug, Display, Write},
    hash::Hash,
    marker::PhantomData,
    rc::Rc,
};

use crate::{
    any::U32PowerOf2,
    call_info,
    common::{
        ignore_eq::IgnoreInEqOrdHash,
        prettify::{set_color, UnwrapOrStr},
    },
    ir::{
        self,
        ir_type::{round_up, CanonName},
        recording::Context,
        Len,
    },
};
use layoutable::{
    align_size::{StructLayoutCalculator, PACKED_ALIGN},
    LayoutableType, Matrix, Vector, PackedVector,
};

pub(crate) mod compatible_with;
pub(crate) mod eq;
pub(crate) mod layoutable;

pub const DEFAULT_REPR: Repr = Repr::Storage;

/// The memory layout of a type.
///
/// This models only the layout, not other characteristics of the types.
/// For example an `Atomic<vec<u32, x1>>` is treated like a regular `vec<u32, x1>` layout wise.
///
/// ### Layout comparison
///
/// The `PartialEq + Eq` implementation of `TypeLayout` is designed to answer the question
/// "do these two types have the same layout" so that uploading a type to the gpu
/// will result in no memory errors.
///
/// a layout comparison looks like this:
/// ```
/// use shame as sm;
/// assert_eq!(sm::cpu_layout::<f32>(), sm::gpu_layout<sm::vec<f32, sm::x1>>());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeLayout {
    /// `vec<T, L>`
    Vector(VectorLayout),
    /// special compressed vectors for vertex attribute types
    ///
    /// see the [`crate::packed`] module
    PackedVector(PackedVectorLayout),
    /// `mat<T, Cols, Rows>`, first `Len2` is cols, 2nd `Len2` is rows
    Matrix(MatrixLayout),
    /// `Array<T>` and `Array<T, Size<N>>`
    Array(Rc<ArrayLayout>),
    /// structures which may be empty and may have an unsized last field
    Struct(Rc<StructLayout>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorLayout {
    pub byte_size: u64,
    pub align: IgnoreInEqOrdHash<U32PowerOf2>,
    pub ty: Vector,

    // debug information
    pub debug_is_atomic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackedVectorLayout {
    pub byte_size: u64,
    pub align: IgnoreInEqOrdHash<U32PowerOf2>,
    pub ty: PackedVector,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatrixLayout {
    pub byte_size: u64,
    pub align: IgnoreInEqOrdHash<U32PowerOf2>,
    pub ty: Matrix,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayLayout {
    pub byte_size: Option<u64>,
    pub align: IgnoreInEqOrdHash<U32PowerOf2>,
    pub byte_stride: u64,
    pub element_ty: TypeLayout,
    // not NonZeroU32, since for rust `CpuLayout`s the array size may be 0.
    pub len: Option<u32>,
}

/// a sized or unsized struct type with 0 or more fields
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructLayout {
    pub byte_size: Option<u64>,
    pub align: IgnoreInEqOrdHash<U32PowerOf2>,
    /// The canonical name of the structure type, ignored in equality/hash comparisons
    pub name: IgnoreInEqOrdHash<CanonName>,
    /// The fields of the structure with their memory offsets
    pub fields: Vec<FieldLayout>,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldLayout {
    /// The relative byte offset of this field from the start of its containing structure
    pub rel_byte_offset: u64,
    pub name: CanonName,
    pub ty: TypeLayout,
}

/// Enum of layout rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Repr {
    /// Wgsl storage address space layout
    /// https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    Storage,
    /// Wgsl uniform address space layout
    /// https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    Uniform,
    /// Packed layout. Vertex buffer only.
    Packed,
}

impl std::fmt::Display for Repr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Repr::Storage => write!(f, "storage"),
            Repr::Uniform => write!(f, "uniform"),
            Repr::Packed => write!(f, "packed"),
        }
    }
}

impl ArrayLayout {
    pub fn short_name(&self) -> String {
        match self.len {
            Some(n) => format!("array<{}, {n}>", self.element_ty.short_name()),
            None => format!("array<{}, runtime-sized>", self.element_ty.short_name()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayoutInfo(u8);
#[rustfmt::skip]
impl LayoutInfo {
    pub const NONE:   Self    = Self(0);
    pub const OFFSET: Self    = Self(1 << 0);
    pub const ALIGN:  Self    = Self(1 << 1);
    pub const SIZE:   Self    = Self(1 << 2);
    pub const STRIDE: Self    = Self(1 << 3);
    pub const ALL:    Self    = Self(Self::OFFSET.0 | Self::ALIGN.0 | Self::SIZE.0 | Self::STRIDE.0);
}
impl std::ops::BitOr for LayoutInfo {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output { LayoutInfo(self.0 | rhs.0) }
}
impl LayoutInfo {
    pub fn contains(&self, other: Self) -> bool { (self.0 & other.0) == other.0 }

    pub fn header(&self) -> String {
        let mut parts = Vec::with_capacity(4);
        for (info, info_str) in [
            (Self::OFFSET, "offset"),
            (Self::ALIGN, "align"),
            (Self::SIZE, "size"),
            (Self::STRIDE, "stride"),
        ] {
            if self.contains(info) {
                parts.push(info_str);
            }
        }
        parts.join(" ")
    }

    pub fn format(&self, offset: Option<u64>, align: U32PowerOf2, size: Option<u64>, stride: Option<u64>) -> String {
        let infos: [(Self, &'static str, &dyn Display); 4] = [
            (Self::OFFSET, "offset", &UnwrapOrStr(offset, "")),
            (Self::ALIGN, "align", &align.as_u32()),
            (Self::SIZE, "size", &UnwrapOrStr(size, "")),
            (Self::STRIDE, "stride", &UnwrapOrStr(stride, "")),
        ];
        let mut parts = Vec::with_capacity(4);
        for (info, info_str, value) in infos {
            if self.contains(info) {
                parts.push(format!("{:>info_width$}", value, info_width = info_str.len()));
            }
        }
        parts.join(" ")
    }
}

pub struct StructWriter<'a> {
    s: &'a StructLayout,
    tab: &'static str,
    layout_info: LayoutInfo,
    layout_info_offset: usize,
}

impl<'a> StructWriter<'a> {
    pub fn new(s: &'a StructLayout, layout_info: LayoutInfo) -> Self {
        let mut this = Self {
            s,
            tab: "    ",
            layout_info,
            layout_info_offset: 0,
        };
        this.set_layout_info_offset_auto(None);
        this
    }

    pub(crate) fn layout_info_offset(&self) -> usize { self.layout_info_offset }

    /// By setting `max_fields` to `Some(n)`, the writer will adjust Self::layout_info_offset
    /// to only take into account the first `n` fields of the struct.
    pub(crate) fn set_layout_info_offset_auto(&mut self, max_fields: Option<usize>) {
        let fields = match max_fields {
            Some(n) => n.min(self.s.fields.len()),
            None => self.s.fields.len(),
        };
        let layout_info_offset = (0..fields)
            .map(|i| self.field_declaration(i).len())
            .max()
            .unwrap_or(0)
            .max(self.struct_declaration().len());
        self.layout_info_offset = layout_info_offset;
    }

    pub(crate) fn ensure_layout_info_offset(&mut self, min_layout_info_offset: usize) {
        self.layout_info_offset = self.layout_info_offset.max(min_layout_info_offset)
    }

    pub(crate) fn tab(&self) -> &'static str { self.tab }

    fn struct_declaration(&self) -> String { format!("struct {} {{", self.s.name) }

    fn field_declaration(&self, field_index: usize) -> String {
        match self.s.fields.get(field_index) {
            Some(field) => format!("{}{}: {},", self.tab, field.name, field.ty.short_name()),
            None => String::new(),
        }
    }

    pub(crate) fn write_header<W: Write>(&self, f: &mut W) -> std::fmt::Result {
        if self.layout_info != LayoutInfo::NONE {
            let info_offset = self.layout_info_offset();
            write!(f, "{:info_offset$}{}", "", self.layout_info.header())?;
        }
        Ok(())
    }

    pub(crate) fn write_struct_declaration<W: Write>(&self, f: &mut W) -> std::fmt::Result {
        let info = self.layout_info.format(
            None, // offset is not applicable for structs
            *self.s.align,
            self.s.byte_size,
            None, // stride is not applicable for structs
        );
        let info_offset = self.layout_info_offset();
        write!(f, "{:info_offset$}{info}", self.struct_declaration())
    }

    pub(crate) fn write_field<W: Write>(&self, f: &mut W, field_index: usize) -> std::fmt::Result {
        use TypeLayout::*;

        let field = &self.s.fields[field_index];
        let info = self.layout_info.format(
            Some(field.rel_byte_offset),
            field.ty.align(),
            field.ty.byte_size(),
            match &field.ty {
                Array(array) => Some(array.byte_stride),
                Vector(_) | PackedVector(_) | Matrix(_) | Struct(_) => None,
            },
        );
        let info_offset = self.layout_info_offset();
        write!(f, "{:info_offset$}{info}", self.field_declaration(field_index))
    }

    pub(crate) fn write_struct_end<W: Write>(&self, f: &mut W) -> std::fmt::Result { write!(f, "}}") }

    pub(crate) fn writeln_header<W: Write>(&self, f: &mut W) -> std::fmt::Result {
        if self.layout_info != LayoutInfo::NONE {
            self.write_header(f)?;
            writeln!(f)?;
        }
        Ok(())
    }

    pub(crate) fn writeln_struct_declaration<W: Write>(&self, f: &mut W) -> std::fmt::Result {
        self.write_struct_declaration(f)?;
        writeln!(f)
    }

    pub(crate) fn writeln_field<W: Write>(&self, f: &mut W, field_index: usize) -> std::fmt::Result {
        self.write_field(f, field_index)?;
        writeln!(f)
    }

    pub(crate) fn writeln_struct_end<W: Write>(&self, f: &mut W) -> std::fmt::Result {
        self.write_struct_end(f)?;
        writeln!(f)
    }
}

impl StructLayout {
    pub fn short_name(&self) -> String { self.name.to_string() }

    pub(crate) fn to_string_with_layout_info(&self, layout_info: LayoutInfo) -> Result<String, std::fmt::Error> {
        let mut s = String::new();
        self.write(&mut s, layout_info)?;
        Ok(s)
    }

    pub(crate) fn writer(&self, layout_info: LayoutInfo) -> StructWriter<'_> { StructWriter::new(self, layout_info) }

    pub(crate) fn write<W: Write>(&self, f: &mut W, layout_info: LayoutInfo) -> std::fmt::Result {
        use TypeLayout::*;

        let mut writer = self.writer(layout_info);
        writer.writeln_header(f)?;
        writer.writeln_struct_declaration(f)?;
        for i in 0..self.fields.len() {
            writer.writeln_field(f, i)?;
        }
        writer.writeln_struct_end(f)
    }
}

impl TypeLayout {
    /// Returns the byte size of the represented type.
    ///
    /// For sized types, this returns Some(size), while for unsized types
    /// (like runtime-sized arrays), this returns None.
    pub fn byte_size(&self) -> Option<u64> {
        match self {
            TypeLayout::Vector(v) => Some(v.byte_size),
            TypeLayout::PackedVector(p) => Some(p.byte_size),
            TypeLayout::Matrix(m) => Some(m.byte_size),
            TypeLayout::Array(a) => a.byte_size,
            TypeLayout::Struct(s) => s.byte_size,
        }
    }

    /// Returns the alignment requirement of the represented type.
    pub fn align(&self) -> U32PowerOf2 {
        match self {
            TypeLayout::Vector(v) => *v.align,
            TypeLayout::PackedVector(p) => *p.align,
            TypeLayout::Matrix(m) => *m.align,
            TypeLayout::Array(a) => *a.align,
            TypeLayout::Struct(s) => *s.align,
        }
    }

    /// If self is sized and `byte_size` is None, the size is not overwritten.
    pub fn set_byte_size(&mut self, byte_size: Option<u64>) {
        match self {
            TypeLayout::Vector(v) => {
                if let Some(size) = byte_size {
                    v.byte_size = size;
                }
            }
            TypeLayout::Matrix(m) => {
                if let Some(size) = byte_size {
                    m.byte_size = size;
                }
            }
            TypeLayout::PackedVector(v) => {
                if let Some(size) = byte_size {
                    v.byte_size = size;
                }
            }
            TypeLayout::Array(a) => {
                let mut array = (**a).clone();
                array.byte_size = byte_size;
                *a = Rc::new(array);
            }
            TypeLayout::Struct(s) => {
                let mut struct_ = (**s).clone();
                struct_.byte_size = byte_size;
                *s = Rc::new(struct_);
            }
        }
    }

    pub fn set_align(&mut self, align: U32PowerOf2) {
        let align = align.into();
        match self {
            TypeLayout::Vector(v) => {
                v.align = align;
            }
            TypeLayout::Matrix(m) => {
                m.align = align;
            }
            TypeLayout::PackedVector(v) => {
                v.align = align;
            }
            TypeLayout::Array(a) => {
                let mut array = (**a).clone();
                array.align = align;
                *a = Rc::new(array);
            }
            TypeLayout::Struct(s) => {
                let mut struct_ = (**s).clone();
                struct_.align = align;
                *s = Rc::new(struct_);
            }
        }
    }

    /// a short name for this `TypeLayout`, useful for printing inline
    pub fn short_name(&self) -> String {
        use TypeLayout::*;

        match &self {
            Vector(v) => v.ty.to_string(),
            PackedVector(v) => v.ty.to_string(),
            Matrix(m) => m.ty.to_string(),
            Array(a) => a.short_name(),
            Struct(s) => s.short_name(),
        }
    }

    pub(crate) fn to_string_with_layout_information(&self, layout_info: LayoutInfo) -> Result<String, std::fmt::Error> {
        let mut s = String::new();
        self.write(&mut s, layout_info)?;
        Ok(s)
    }

    pub(crate) fn write<W: Write>(&self, f: &mut W, layout_info: LayoutInfo) -> std::fmt::Result {
        use TypeLayout::*;

        match self {
            Vector(_) | PackedVector(_) | Matrix(_) | Array(_) => {
                let plain = self.short_name();

                let stride = match self {
                    Array(a) => Some(a.byte_stride),
                    Vector(_) | PackedVector(_) | Matrix(_) | Struct(_) => None,
                };
                let info_offset = plain.len() + 1;

                // Write header if some layout information is requested
                if layout_info != LayoutInfo::NONE {
                    writeln!(f, "{:info_offset$}{}", "", layout_info.header())?;
                }

                // Write the type name and layout information
                let info = layout_info.format(None, self.align(), self.byte_size(), stride);
                writeln!(f, "{plain:info_offset$}{info}")?;
            }
            Struct(s) => s.write(f, layout_info)?,
        };

        Ok(())
    }

    pub(crate) fn first_line_of_display_with_ellipsis(&self) -> String {
        let string = format!("{}", self);
        string.split_once('\n').map(|(s, _)| format!("{s}…")).unwrap_or(string)
    }
}

impl TypeLayout {
    // TODO(chronicl) this should be removed with improved any api for storage/uniform bindings
    pub(crate) fn from_store_ty(
        store_type: ir::StoreType,
    ) -> Result<Self, layoutable::ir_compat::LayoutableConversionError> {
        let t: layoutable::LayoutableType = store_type.try_into()?;
        Ok(t.layout())
    }
}

impl From<VectorLayout> for TypeLayout {
    fn from(layout: VectorLayout) -> Self { TypeLayout::Vector(layout) }
}

impl From<PackedVectorLayout> for TypeLayout {
    fn from(layout: PackedVectorLayout) -> Self { TypeLayout::PackedVector(layout) }
}

impl From<MatrixLayout> for TypeLayout {
    fn from(layout: MatrixLayout) -> Self { TypeLayout::Matrix(layout) }
}

impl From<ArrayLayout> for TypeLayout {
    fn from(layout: ArrayLayout) -> Self { TypeLayout::Array(Rc::new(layout)) }
}

impl From<StructLayout> for TypeLayout {
    fn from(layout: StructLayout) -> Self { TypeLayout::Struct(Rc::new(layout)) }
}


impl Display for TypeLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.write(f, LayoutInfo::ALL) }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        any::U32PowerOf2,
        frontend::rust_types::type_layout::{
            layoutable::{*},
            Repr, *,
        },
    };
    use std::{rc::Rc, num::NonZeroU32};

    #[test]
    fn test_array_alignment() {
        let array: LayoutableType = SizedArray::new(
            Rc::new(Vector::new(ScalarType::F32, Len::X1).into()),
            NonZeroU32::new(1).unwrap(),
        )
        .into();

        // To change the top level arrays repr, we need to set the default repr,
        // because non-structs inherit repr.
        let storage = array.layout_with_default_repr(Repr::Storage);
        let uniform = array.layout_with_default_repr(Repr::Uniform);
        let packed = array.layout_with_default_repr(Repr::Packed);

        assert_eq!(storage.align(), U32PowerOf2::_4);
        assert_eq!(uniform.align(), U32PowerOf2::_16);
        assert_eq!(packed.align(), U32PowerOf2::_1);

        assert_eq!(storage.byte_size(), Some(4));
        assert_eq!(uniform.byte_size(), Some(16));
        assert_eq!(packed.byte_size(), Some(4));

        match (storage, uniform, packed) {
            (TypeLayout::Array(storage), TypeLayout::Array(uniform), TypeLayout::Array(packed)) => {
                assert_eq!(storage.len, Some(1));
                assert_eq!(uniform.len, Some(1));
                assert_eq!(packed.len, Some(1));
                assert_eq!(storage.byte_stride, 4);
                assert_eq!(uniform.byte_stride, 16);
                assert_eq!(packed.byte_stride, 4);
            }
            _ => panic!("Unexpected layout kind"),
        }
    }

    #[test]
    fn test_struct_alignment() {
        let s =
            |repr| -> LayoutableType { SizedStruct::new("A", "a", Vector::new(ScalarType::F32, Len::X1), repr).into() };

        let storage = s(Repr::Storage).layout();
        let uniform = s(Repr::Uniform).layout();
        let packed = s(Repr::Packed).layout();

        assert_eq!(storage.align(), U32PowerOf2::_4);
        assert_eq!(uniform.align(), U32PowerOf2::_16);
        assert_eq!(packed.align(), U32PowerOf2::_1);

        assert_eq!(storage.byte_size(), Some(4));
        assert_eq!(uniform.byte_size(), Some(16));
        assert_eq!(packed.byte_size(), Some(4));
    }

    #[test]
    fn test_nested_struct_field_offset() {
        let s = |repr| -> LayoutableType {
            let a = SizedStruct::new("A", "a", Vector::new(ScalarType::F32, Len::X1), repr);
            SizedStruct::new("B", "a", Vector::new(ScalarType::F32, Len::X1), repr)
                .extend("b", a) // offset 4 for storage and packed, offset 16 for uniform
                .into()
        };

        let storage = s(Repr::Storage).layout();
        let uniform = s(Repr::Uniform).layout();
        let packed = s(Repr::Packed).layout();

        assert_eq!(storage.align(), U32PowerOf2::_4);
        assert_eq!(uniform.align(), U32PowerOf2::_16);
        assert_eq!(packed.align(), U32PowerOf2::_1);

        assert_eq!(storage.byte_size(), Some(8));
        // field b is bytes 16..=19 and struct size must be a multiple of the struct align (16)
        assert_eq!(uniform.byte_size(), Some(32));
        assert_eq!(packed.byte_size(), Some(8));

        match (storage, uniform, packed) {
            (TypeLayout::Struct(storage), TypeLayout::Struct(uniform), TypeLayout::Struct(packed)) => {
                assert_eq!(storage.fields[1].rel_byte_offset, 4);
                assert_eq!(uniform.fields[1].rel_byte_offset, 16);
                assert_eq!(packed.fields[1].rel_byte_offset, 4);
            }
            _ => panic!("Unexpected layout kind"),
        }
    }

    #[test]
    fn test_array_in_struct_field_offset() {
        let s = |repr| -> LayoutableType {
            SizedStruct::new("B", "a", Vector::new(ScalarType::F32, Len::X1), repr)
                .extend(
                    "b",
                    SizedArray::new(
                        Rc::new(Vector::new(ScalarType::F32, Len::X1).into()),
                        NonZeroU32::new(1).unwrap(),
                    ),
                ) // offset 4 for storage and packed, offset 16 for uniform
                .into()
        };

        let storage = s(Repr::Storage).layout();
        let uniform = s(Repr::Uniform).layout();
        let packed = s(Repr::Packed).layout();

        assert_eq!(storage.align(), U32PowerOf2::_4);
        assert_eq!(uniform.align(), U32PowerOf2::_16);
        assert_eq!(packed.align(), U32PowerOf2::_1);

        assert_eq!(storage.byte_size(), Some(8));
        // field b is bytes 16..=19 and struct size must be a multiple of the struct align (16)
        assert_eq!(uniform.byte_size(), Some(32));
        assert_eq!(packed.byte_size(), Some(8));

        match (storage, uniform, packed) {
            (TypeLayout::Struct(storage), TypeLayout::Struct(uniform), TypeLayout::Struct(packed)) => {
                assert_eq!(storage.fields[1].rel_byte_offset, 4);
                assert_eq!(uniform.fields[1].rel_byte_offset, 16);
                assert_eq!(packed.fields[1].rel_byte_offset, 4);
            }
            _ => panic!("Unexpected layout kind"),
        }
    }

    #[test]
    fn test_unsized_struct_layout() {
        let mut unsized_struct = UnsizedStruct {
            name: CanonName::from("TestStruct"),
            repr: Repr::Storage,
            sized_fields: vec![
                SizedField {
                    name: CanonName::from("field1"),
                    custom_min_size: None,
                    custom_min_align: None,
                    ty: Vector::new(ScalarType::F32, Len::X2).into(),
                },
                SizedField {
                    name: CanonName::from("field2"),
                    custom_min_size: None,
                    custom_min_align: None,
                    ty: Vector::new(ScalarType::F32, Len::X1).into(),
                },
            ],
            last_unsized: RuntimeSizedArrayField {
                name: CanonName::from("dynamic_array"),
                custom_min_align: None,
                array: RuntimeSizedArray {
                    element: Vector::new(ScalarType::F32, Len::X1).into(),
                },
            },
        };
        let recipe: LayoutableType = unsized_struct.clone().into();

        let layout = recipe.layout();
        assert_eq!(layout.byte_size(), None);
        assert!(layout.align().as_u64() == 8); // align of vec2<f32>
        match &layout {
            TypeLayout::Struct(struct_layout) => {
                assert_eq!(struct_layout.fields.len(), 3);
                assert_eq!(struct_layout.fields[0].name, CanonName::from("field1"));
                assert_eq!(struct_layout.fields[1].name, CanonName::from("field2"));
                assert_eq!(struct_layout.fields[2].name, CanonName::from("dynamic_array"));

                assert_eq!(struct_layout.fields[0].rel_byte_offset, 0); // vec2<f32>
                assert_eq!(struct_layout.fields[1].rel_byte_offset, 8); // f32
                assert_eq!(struct_layout.fields[2].rel_byte_offset, 12); // Array<f32>
                // The last field should be an unsized array
                match &struct_layout.fields[2].ty {
                    TypeLayout::Array(array) => {
                        assert_eq!(array.byte_size, None);
                        assert_eq!(array.byte_stride, 4)
                    }
                    _ => panic!("Expected runtime-sized array for last field"),
                }
            }
            _ => panic!("Expected structure layout"),
        }

        // Testing uniform representation
        unsized_struct.repr = Repr::Uniform;
        let recipe: LayoutableType = unsized_struct.into();
        println!("{:#?}", recipe);
        let layout = recipe.layout();
        assert_eq!(layout.byte_size(), None);
        // Struct alignmment has to be a multiple of 16, but the runtime sized array
        // also has an alignment of 16, which transfers to the struct alignment.
        assert!(layout.align().as_u64() == 16);
        match &layout {
            TypeLayout::Struct(struct_layout) => {
                assert_eq!(struct_layout.fields[0].rel_byte_offset, 0); // vec2<f32>
                assert_eq!(struct_layout.fields[1].rel_byte_offset, 8); // f32
                // array has alignment of 16, so offset should be 16
                assert_eq!(struct_layout.fields[2].rel_byte_offset, 16); // Array<f32>
                match &struct_layout.fields[2].ty {
                    // Stride has to be a multiple of 16 in uniform address space
                    TypeLayout::Array(array) => {
                        assert_eq!(array.byte_size, None);
                        assert_eq!(array.byte_stride, 16);
                    }
                    _ => panic!("Expected runtime-sized array for last field"),
                }
            }
            _ => panic!("Expected structure layout"),
        }
    }
}
