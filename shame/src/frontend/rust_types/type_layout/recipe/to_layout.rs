use std::rc::Rc;
use crate::{
    frontend::rust_types::type_layout::{
        ArrayLayout, FieldLayout, MatrixLayout, PackedVectorLayout, Repr, StructLayout, VectorLayout,
    },
    ir, TypeLayout,
};
use super::{
    Atomic, TypeLayoutRecipe, Matrix, PackedVector, RuntimeSizedArray, SizedArray, SizedField, SizedStruct, SizedType,
    UnsizedStruct, Vector,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecipeContains {
    CustomFieldAlign,
    CustomFieldSize,
    PackedVector,
}

impl std::fmt::Display for RecipeContains {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecipeContains::CustomFieldAlign => write!(f, "custom field alignment attribute"),
            RecipeContains::CustomFieldSize => write!(f, "custom field size attribute"),
            RecipeContains::PackedVector => write!(f, "packed vector"),
        }
    }
}

impl TypeLayoutRecipe {
    /// Returns the layout of this type recipe using `Repr::default` for top level types
    /// that aren't structs.
    pub fn layout(&self) -> TypeLayout { self.layout_with_default_repr(Repr::default()) }

    /// Returns the layout of this type recipe using the given `default_repr` for top level types
    /// that aren't structs.
    pub fn layout_with_default_repr(&self, default_repr: Repr) -> TypeLayout {
        match self {
            TypeLayoutRecipe::Sized(ty) => ty.layout(default_repr),
            TypeLayoutRecipe::UnsizedStruct(ty) => ty.layout().into(),
            TypeLayoutRecipe::RuntimeSizedArray(ty) => ty.layout(default_repr).into(),
        }
    }

    /// Checks whether `RecipeContains` is contained in this type recipe.
    pub fn contains(&self, c: RecipeContains) -> bool {
        match self {
            TypeLayoutRecipe::Sized(ty) => ty.contains(c),
            TypeLayoutRecipe::UnsizedStruct(ty) => ty.contains(c),
            TypeLayoutRecipe::RuntimeSizedArray(ty) => ty.contains(c),
        }
    }
}

#[allow(missing_docs)]
impl SizedType {
    pub fn layout(&self, parent_repr: Repr) -> TypeLayout {
        match &self {
            SizedType::Vector(v) => v.layout(parent_repr).into(),
            SizedType::Atomic(a) => a.layout(parent_repr).into(),
            SizedType::Matrix(m) => m.layout(parent_repr).into(),
            SizedType::Array(a) => a.layout(parent_repr).into(),
            SizedType::PackedVec(v) => v.layout(parent_repr).into(),
            SizedType::Struct(s) => s.layout().into(),
        }
    }

    pub fn contains(&self, c: RecipeContains) -> bool {
        match c {
            RecipeContains::CustomFieldAlign | RecipeContains::CustomFieldSize | RecipeContains::PackedVector => {
                match self {
                    SizedType::Vector(_) => false,
                    SizedType::Atomic(_) => false,
                    SizedType::Matrix(_) => false,
                    SizedType::Array(a) => a.contains(c),
                    SizedType::PackedVec(_) => c == RecipeContains::PackedVector,
                    SizedType::Struct(s) => s.contains(c),
                }
            }
        }
    }
}

#[allow(missing_docs)]
impl Vector {
    pub fn layout(&self, parent_repr: Repr) -> VectorLayout {
        VectorLayout {
            byte_size: self.byte_size(parent_repr),
            align: self.align(parent_repr).into(),
            ty: *self,
            debug_is_atomic: false,
        }
    }
}

#[allow(missing_docs)]
impl Matrix {
    pub fn layout(&self, parent_repr: Repr) -> MatrixLayout {
        MatrixLayout {
            byte_size: self.byte_size(parent_repr),
            align: self.align(parent_repr).into(),
            ty: *self,
        }
    }
}

#[allow(missing_docs)]
impl Atomic {
    pub fn layout(&self, parent_repr: Repr) -> VectorLayout {
        // Atomic types are represented as vectors of length 1.
        let vector = Vector::new(self.scalar.into(), ir::Len::X1);
        let mut layout = vector.layout(parent_repr);
        layout.debug_is_atomic = true;
        layout
    }
}

impl PackedVector {
    pub fn layout(&self, parent_repr: Repr) -> PackedVectorLayout {
        PackedVectorLayout {
            byte_size: self.byte_size().as_u64(),
            align: self.align(parent_repr).into(),
            ty: *self,
        }
    }
}

#[allow(missing_docs)]
impl SizedArray {
    pub fn layout(&self, parent_repr: Repr) -> ArrayLayout {
        ArrayLayout {
            byte_size: self.byte_size(parent_repr).into(),
            align: self.align(parent_repr).into(),
            byte_stride: self.byte_stride(parent_repr),
            element_ty: self.element.layout(parent_repr),
            len: Some(self.len.get()),
        }
    }

    pub fn contains(&self, c: RecipeContains) -> bool {
        match c {
            RecipeContains::CustomFieldAlign | RecipeContains::CustomFieldSize | RecipeContains::PackedVector => {
                self.element.contains(c)
            }
        }
    }
}

#[allow(missing_docs)]
impl SizedStruct {
    pub fn layout(&self) -> StructLayout {
        let mut field_offsets = self.field_offsets();
        let fields = (&mut field_offsets)
            .zip(self.fields())
            .map(|(offset, field)| sized_field_to_field_layout(field, offset, self.repr))
            .collect::<Vec<_>>();

        let (byte_size, align) = field_offsets.struct_byte_size_and_align();

        StructLayout {
            byte_size: Some(byte_size),
            align: align.into(),
            name: self.name.clone().into(),
            fields,
        }
    }

    pub fn contains(&self, c: RecipeContains) -> bool {
        let mut contains = false;
        match c {
            RecipeContains::CustomFieldAlign => {
                contains |= self.fields.iter().any(|f| f.custom_min_align.is_some());
            }
            RecipeContains::CustomFieldSize => {
                contains |= self.fields.iter().any(|f| f.custom_min_size.is_some());
            }
            RecipeContains::PackedVector => {}
        }
        for field in self.fields.iter() {
            contains |= field.ty.contains(c);
        }
        contains
    }
}

fn sized_field_to_field_layout(field: &SizedField, offset: u64, repr: Repr) -> FieldLayout {
    let mut ty = field.ty.layout(repr);
    // VERY IMPORTANT: TypeLayout::from_sized_type does not take into account
    // custom_min_align and custom_min_size, but field.byte_size and field.align do.
    ty.set_byte_size(Some(field.byte_size(repr)));
    ty.set_align(field.align(repr));

    FieldLayout {
        rel_byte_offset: offset,
        name: field.name.clone(),
        ty,
    }
}

#[allow(missing_docs)]
impl UnsizedStruct {
    pub fn layout(&self) -> StructLayout {
        let mut field_offsets = self.field_offsets();
        let mut fields = (&mut field_offsets.sized_field_offsets())
            .zip(self.sized_fields.iter())
            .map(|(offset, field)| sized_field_to_field_layout(field, offset, self.repr))
            .collect::<Vec<_>>();

        let (field_offset, align) = field_offsets.last_field_offset_and_struct_align();

        let mut ty = self.last_unsized.array.layout(self.repr);
        // VERY IMPORTANT: TypeLayout::from_runtime_sized_array does not take into account
        // custom_min_align, but s.last_unsized.align does.
        ty.align = self.last_unsized.align(self.repr).into();

        fields.push(FieldLayout {
            rel_byte_offset: field_offset,
            name: self.last_unsized.name.clone(),
            ty: ty.into(),
        });

        StructLayout {
            byte_size: None,
            align: align.into(),
            name: self.name.clone().into(),
            fields,
        }
    }

    pub fn contains(&self, c: RecipeContains) -> bool {
        let mut contains = false;
        match c {
            RecipeContains::CustomFieldAlign => {
                contains |= self.sized_fields.iter().any(|f| f.custom_min_align.is_some());
                contains |= self.last_unsized.custom_min_align.is_some();
            }
            RecipeContains::CustomFieldSize => {
                contains |= self.sized_fields.iter().any(|f| f.custom_min_size.is_some());
            }
            RecipeContains::PackedVector => {}
        }
        for field in self.sized_fields.iter() {
            contains |= field.ty.contains(c);
        }
        contains
    }
}

#[allow(missing_docs)]
impl RuntimeSizedArray {
    pub fn layout(&self, parent_repr: Repr) -> ArrayLayout {
        ArrayLayout {
            byte_size: None,
            align: self.align(parent_repr).into(),
            byte_stride: self.byte_stride(parent_repr),
            element_ty: self.element.layout(parent_repr),
            len: None,
        }
    }

    pub fn contains(&self, c: RecipeContains) -> bool {
        match c {
            RecipeContains::CustomFieldAlign | RecipeContains::CustomFieldSize | RecipeContains::PackedVector => {
                self.element.contains(c)
            }
        }
    }
}
