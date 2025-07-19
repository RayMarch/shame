use std::rc::Rc;

use crate::{
    any::layout::{ElementLayout, FieldLayout, FieldLayoutWithOffset, Repr, StructLayout, TypeLayoutSemantics},
    frontend::rust_types::type_layout::{ArrayLayout, DEFAULT_REPR},
    ir, TypeLayout,
};

use super::{
    Atomic, LayoutableType, Matrix, PackedVector, RuntimeSizedArray, SizedArray, SizedField, SizedStruct, SizedType,
    UnsizedStruct, Vector,
};

impl LayoutableType {
    pub fn layout(&self) -> TypeLayout { self.layout_with_default_repr(DEFAULT_REPR) }

    pub fn layout_with_default_repr(&self, default_repr: Repr) -> TypeLayout {
        match self {
            LayoutableType::Sized(ty) => ty.layout(default_repr),
            LayoutableType::UnsizedStruct(ty) => ty.layout(default_repr),
            LayoutableType::RuntimeSizedArray(ty) => ty.layout(default_repr),
        }
    }
}

impl SizedType {
    pub fn layout(&self, parent_repr: Repr) -> TypeLayout {
        match &self {
            SizedType::Vector(v) => v.layout(parent_repr),
            SizedType::Atomic(a) => a.layout(parent_repr),
            SizedType::Matrix(m) => m.layout(parent_repr),
            SizedType::Array(a) => a.layout(parent_repr),
            SizedType::PackedVec(v) => v.layout(parent_repr),
            SizedType::Struct(s) => s.layout(parent_repr),
        }
    }
}

impl Vector {
    pub fn layout(&self, parent_repr: Repr) -> TypeLayout {
        TypeLayout::new(
            self.byte_size(parent_repr).into(),
            self.align(parent_repr),
            TypeLayoutSemantics::Vector(*self),
        )
    }
}

impl Matrix {
    pub fn layout(&self, parent_repr: Repr) -> TypeLayout {
        TypeLayout::new(
            self.byte_size(parent_repr).into(),
            self.align(parent_repr),
            TypeLayoutSemantics::Matrix(*self),
        )
    }
}

impl Atomic {
    pub fn layout(&self, parent_repr: Repr) -> TypeLayout {
        // Atomic types are represented as vectors of length 1.
        let vector = Vector::new(self.scalar.into(), ir::Len::X1);
        TypeLayout::new(
            vector.byte_size(parent_repr).into(),
            vector.align(parent_repr),
            TypeLayoutSemantics::Vector(vector),
        )
    }
}

impl SizedArray {
    pub fn layout(&self, parent_repr: Repr) -> TypeLayout {
        TypeLayout::new(
            self.byte_size(parent_repr).into(),
            self.align(parent_repr),
            TypeLayoutSemantics::Array(Rc::new(ArrayLayout {
                byte_stride: self.byte_stride(parent_repr),
                element_ty: self.element.layout(parent_repr),
                len: Some(self.len.get()),
            })),
        )
    }
}

impl PackedVector {
    pub fn layout(&self, parent_repr: Repr) -> TypeLayout {
        TypeLayout::new(
            self.byte_size().as_u64().into(),
            self.align(parent_repr),
            TypeLayoutSemantics::PackedVector(*self),
        )
    }
}

impl SizedStruct {
    pub fn layout(&self, parent_repr: Repr) -> TypeLayout {
        let mut field_offsets = self.field_offsets();
        let fields = (&mut field_offsets)
            .zip(self.fields())
            .map(|(offset, field)| sized_field_to_field_layout(field, offset, parent_repr))
            .collect::<Vec<_>>();

        let (byte_size, align) = field_offsets.struct_byte_size_and_align();

        TypeLayout::new(
            Some(byte_size),
            align,
            TypeLayoutSemantics::Structure(Rc::new(StructLayout {
                name: self.name.clone().into(),
                fields,
            })),
        )
    }
}

fn sized_field_to_field_layout(field: &SizedField, offset: u64, repr: Repr) -> FieldLayoutWithOffset {
    let mut ty = field.ty.layout(repr);
    // VERY IMPORTANT: TypeLayout::from_sized_type does not take into account
    // custom_min_align and custom_min_size, but field.byte_size and field.align do.
    ty.byte_size = Some(field.byte_size(repr));
    ty.align = field.align(repr).into();
    FieldLayoutWithOffset {
        rel_byte_offset: offset,
        field: FieldLayout {
            name: field.name.clone(),
            ty,
        },
    }
}

impl UnsizedStruct {
    pub fn layout(&self, parent_repr: Repr) -> TypeLayout {
        let mut field_offsets = self.field_offsets();
        let mut fields = (&mut field_offsets.sized_field_offsets())
            .zip(self.sized_fields.iter())
            .map(|(offset, field)| sized_field_to_field_layout(field, offset, parent_repr))
            .collect::<Vec<_>>();

        let (field_offset, align) = field_offsets.last_field_offset_and_struct_align();

        let mut ty = self.last_unsized.array.layout(parent_repr);
        // VERY IMPORTANT: TypeLayout::from_runtime_sized_array does not take into account
        // custom_min_align, but s.last_unsized.align does.
        ty.align = self.last_unsized.align(parent_repr).into();

        fields.push(FieldLayoutWithOffset {
            rel_byte_offset: field_offset,
            field: FieldLayout {
                name: self.last_unsized.name.clone(),
                ty,
            },
        });

        TypeLayout::new(
            None,
            align,
            TypeLayoutSemantics::Structure(Rc::new(StructLayout {
                name: self.name.clone().into(),
                fields,
            })),
        )
    }
}

impl RuntimeSizedArray {
    pub fn layout(&self, parent_repr: Repr) -> TypeLayout {
        TypeLayout::new(
            None,
            self.align(parent_repr),
            TypeLayoutSemantics::Array(Rc::new(ArrayLayout {
                byte_stride: self.byte_stride(parent_repr),
                element_ty: self.element.layout(parent_repr),
                len: None,
            })),
        )
    }
}
