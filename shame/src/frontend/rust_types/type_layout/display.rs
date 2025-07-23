//! This module provides the `Display` implementation for `TypeLayout` and also contains
//! a `StructWriter` which can be used to write the layout of a struct piece by piece
//! with configurable layout information.

use std::fmt::{Display, Write};

use crate::{
    any::{
        layout::{ArrayLayout, StructLayout},
        U32PowerOf2,
    },
    common::prettify::UnwrapOrStr,
    TypeLayout,
};

impl Display for TypeLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.write(f, LayoutInfo::ALL) }
}

impl TypeLayout {
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

impl ArrayLayout {
    pub fn short_name(&self) -> String {
        match self.len {
            Some(n) => format!("array<{}, {n}>", self.element_ty.short_name()),
            None => format!("array<{}, runtime-sized>", self.element_ty.short_name()),
        }
    }
}

/// A bitmask that indicates which layout information should be displayed.
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
            // Could make this configurable
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
        let info = self.layout_info.format(None, *self.s.align, self.s.byte_size, None);
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
