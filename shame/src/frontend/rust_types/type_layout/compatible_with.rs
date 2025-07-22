use std::fmt::Write;

use crate::{
    any::layout::StructLayout,
    common::prettify::{set_color, UnwrapOrStr},
    frontend::rust_types::type_layout::{ArrayLayout, LayoutInfo},
    ir::ir_type::max_u64_po2_dividing,
    TypeLayout,
};

use super::{layoutable::LayoutableType, Repr, DEFAULT_REPR};

pub struct TypeLayoutCompatibleWith<AddressSpace> {
    recipe: LayoutableType,
    _phantom: std::marker::PhantomData<AddressSpace>,
}

impl<AS: AddressSpace> TypeLayoutCompatibleWith<AS> {
    pub fn try_from(recipe: LayoutableType) -> Result<Self, AddressSpaceError> {
        let address_space = AS::ADDRESS_SPACE;
        let layout = recipe.layout();

        match (address_space, layout.byte_size()) {
            // Must be sized in wgsl's uniform address space
            (AddressSpaceEnum::WgslUniform, None) => return Err(AddressSpaceError::MustBeSized(recipe, address_space)),
            (AddressSpaceEnum::WgslUniform, Some(_)) | (AddressSpaceEnum::WgslStorage, _) => {}
        }

        // Check for layout errors
        let recipe_unified = recipe.to_unified_repr(address_space.matching_repr());
        let layout_unified = recipe_unified.layout();
        if layout != layout_unified {
            match try_find_mismatch(&layout, &layout_unified) {
                Some(mismatch) => {
                    return Err(LayoutError {
                        recipe,
                        address_space,
                        mismatch,
                    }
                    .into());
                }
                None => return Err(AddressSpaceError::UnknownLayoutError(recipe, address_space)),
            }
        }

        Ok(Self {
            recipe,
            _phantom: std::marker::PhantomData,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AddressSpaceEnum {
    WgslStorage,
    WgslUniform,
}

impl std::fmt::Display for AddressSpaceEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddressSpaceEnum::WgslStorage => f.write_str("wgsl's storage address space"),
            AddressSpaceEnum::WgslUniform => f.write_str("wgsl's uniform address space"),
        }
    }
}

pub trait AddressSpace {
    const ADDRESS_SPACE: AddressSpaceEnum;
}

pub struct WgslStorage;
pub struct WgslUniform;
impl AddressSpace for WgslStorage {
    const ADDRESS_SPACE: AddressSpaceEnum = AddressSpaceEnum::WgslStorage;
}
impl AddressSpace for WgslUniform {
    const ADDRESS_SPACE: AddressSpaceEnum = AddressSpaceEnum::WgslUniform;
}

impl AddressSpaceEnum {
    fn matching_repr(&self) -> Repr {
        match self {
            AddressSpaceEnum::WgslStorage => Repr::Storage,
            AddressSpaceEnum::WgslUniform => Repr::Uniform,
        }
    }
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum AddressSpaceError {
    #[error("{0}")]
    LayoutError(#[from] LayoutError),
    #[error("Unknown layout error occured for {0} in {1}.")]
    UnknownLayoutError(LayoutableType, AddressSpaceEnum),
    #[error(
        "The size of `{0}` on the gpu is not known at compile time. {1} \
     requires that the size of {0} on the gpu is known at compile time."
    )]
    MustBeSized(LayoutableType, AddressSpaceEnum),
    #[error("{0} contains a `PackedVector`, which are not allowed in {1}.")]
    MayNotContainPackedVec(LayoutableType, AddressSpaceEnum),
}

#[derive(Debug, Clone)]
pub struct LayoutError {
    recipe: LayoutableType,
    address_space: AddressSpaceEnum,
    mismatch: LayoutMismatch,
}

impl std::error::Error for LayoutError {}
impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TypeLayout::*;

        let colored = todo!();

        let types_are_the_same = "The LayoutError is produced by comparing two semantically equivalent TypeLayouts, so all (nested) types are the same";
        match &self.mismatch {
            LayoutMismatch::TopLevel {
                layout_left: layout1,
                layout_right: layout2,
                mismatch,
            } => match mismatch {
                TopLevelMismatch::Type => unreachable!("{}", types_are_the_same),
                TopLevelMismatch::ArrayStride {
                    array_left,
                    array_right,
                } => {
                    writeln!(
                        f,
                        "`{}` requires a stride of {} in {}, but has a stride of {}.",
                        array_left.short_name(),
                        array_right.byte_stride,
                        self.address_space,
                        array_left.byte_stride
                    )?;
                }
            },
            LayoutMismatch::Struct {
                struct_left,
                struct_right,
                mismatch,
            } => {
                match mismatch {
                    StructMismatch::FieldArrayStride {
                        field_index,
                        array_left,
                        array_right,
                    } => {
                        writeln!(
                            f,
                            "`{}` in `{}` requires a stride of {} in {}, but has a stride of {}.",
                            array_left.short_name(),
                            struct_left.short_name(),
                            array_right.byte_stride,
                            self.address_space,
                            array_left.byte_stride
                        )?;
                        writeln!(f, "The full layout of `{}` is:", struct_left.short_name())?;
                        write_struct(f, struct_left, Some(*field_index), colored)?;
                    }
                    StructMismatch::FieldByteSize { field_index } => {
                        let field_left = struct_left.fields[*field_index];
                        let field_right = struct_right.fields[*field_index];

                        writeln!(
                            f,
                            "Field `{}` in `{}` requires a byte size of {} in {}, but has a byte size of {}",
                            field_left.name,
                            struct_left.name,
                            UnwrapOrStr(field_right.ty.byte_size(), ""),
                            self.address_space,
                            UnwrapOrStr(field_left.ty.byte_size(), "")
                        )?;
                        writeln!(f, "The full layout of `{}` is:", struct_left.short_name())?;
                        write_struct(f, struct_left, Some(*field_index), colored)?;
                    }
                    StructMismatch::FieldOffset { field_index } => {
                        let field_left = struct_left.fields[*field_index];
                        let field_right = struct_right.fields[*field_index];
                        let field_name = &field_left.name;
                        let offset = field_left.rel_byte_offset;
                        let expected_align = field_right.ty.align().as_u64();
                        let actual_align = max_u64_po2_dividing(field_left.rel_byte_offset);

                        writeln!(
                            f,
                            "Field `{}` in `{}` needs to be {} byte aligned in {}, but has a byte-offset of {} which is only {} byte aligned",
                            field_name, struct_left.name, expected_align, self.address_space, offset, actual_align
                        )?;
                        writeln!(f, "The full layout of `{}` is:", struct_left.short_name())?;
                        write_struct(f, struct_left, Some(*field_index), colored)?;

                        writeln!(f, "Potential solutions include:")?;

                        writeln!(
                            f,
                            "- add an #[align({})] attribute to the definition of `{}`",
                            field_right.ty.align().as_u32(),
                            field_name
                        )?;
                        writeln!(
                            f,
                            "- increase the offset of `{field_name}` until it is divisible by {expected_align} by making previous fields larger or adding fields before it"
                        )?;
                        writeln!(
                            f,
                            "- if you are using the uniform address space, use the storage address space instead"
                        )?;
                        writeln!(f)?;


                        match self.address_space {
                            AddressSpaceEnum::WgslUniform => writeln!(
                                f,
                                "In the {}, structs, arrays and array elements must be at least 16 byte aligned.",
                                self.address_space
                            )?,
                            AddressSpaceEnum::WgslStorage => {}
                        }

                        match self.address_space {
                            (AddressSpaceEnum::WgslUniform | AddressSpaceEnum::WgslStorage) => writeln!(
                                f,
                                "More info about the {} can be found at https://www.w3.org/TR/WGSL/#address-space-layout-constraints",
                                self.address_space
                            )?,
                        }
                    }
                    StructMismatch::FieldCount |
                    StructMismatch::FieldName { .. } |
                    StructMismatch::FieldType { .. } => {
                        unreachable!("{}", types_are_the_same)
                    }
                };
            }
        }
        todo!()
    }
}

fn write_struct<W>(f: &mut W, s: &StructLayout, highlight_field: Option<usize>, colored: bool) -> std::fmt::Result
where
    W: Write,
{
    let use_256_color_mode = false;

    let mut writer = s.writer(LayoutInfo::ALL);
    writer.writeln_header(f);
    writer.writeln_struct_declaration(f);
    for field_index in 0..s.fields.len() {
        if Some(field_index) == highlight_field {
            if colored {
                set_color(f, Some("#508EE3"), use_256_color_mode)?;
            }
            writer.write_field(f, field_index)?;
            writeln!(f, " <--")?;
            if colored {
                set_color(f, None, use_256_color_mode)?;
            }
        } else {
            writer.writeln_field(f, field_index);
        }
    }
    writer.writeln_struct_end(f)?;

    Ok(())
}

#[derive(Debug, Clone)]
pub enum LayoutMismatch {
    /// `layout1` and `layout2` are always top level layouts.
    ///
    /// For example, in case of `Array<Array<f32x3>>`, it's always the layout
    /// of `Array<Array<f32x3>>` even if the array stride mismatch
    /// may be happening for the inner `Array<f32x3>`.
    TopLevel {
        layout_left: TypeLayout,
        layout_right: TypeLayout,
        mismatch: TopLevelMismatch,
    },
    Struct {
        struct_left: StructLayout,
        struct_right: StructLayout,
        mismatch: StructMismatch,
    },
}

#[derive(Debug, Clone)]
pub enum TopLevelMismatch {
    Type,
    /// This contains the array layout where the mismatch is happening,
    /// which is not necessarily the top level array.
    /// For example, in case of an array stride mismatch of the inner
    /// array in Array<Array<f32x3>>, this is Array<f32x3>'s layout.
    ArrayStride {
        array_left: ArrayLayout,
        array_right: ArrayLayout,
    },
}

/// Returns the first mismatch found in the struct fields.
///
/// Field count is checked last.
#[derive(Debug, Clone)]
pub enum StructMismatch {
    FieldName {
        field_index: usize,
    },
    FieldType {
        field_index: usize,
    },
    FieldOffset {
        field_index: usize,
    },
    FieldByteSize {
        field_index: usize,
    },
    FieldArrayStride {
        field_index: usize,
        array_left: ArrayLayout,
        array_right: ArrayLayout,
    },
    FieldCount,
}

/// Find the first depth first layout mismatch
pub(crate) fn try_find_mismatch(layout1: &TypeLayout, layout2: &TypeLayout) -> Option<LayoutMismatch> {
    use TypeLayout::*;

    // First check if the kinds are the same type
    match (&layout1, &layout2) {
        (Vector(v1), Vector(v2)) => {
            if v1.ty != v2.ty {
                return Some(LayoutMismatch::TopLevel {
                    layout_left: layout1.clone(),
                    layout_right: layout2.clone(),
                    mismatch: TopLevelMismatch::Type,
                });
            }
        }
        (PackedVector(p1), PackedVector(p2)) => {
            if p1.ty != p2.ty {
                return Some(LayoutMismatch::TopLevel {
                    layout_left: layout1.clone(),
                    layout_right: layout2.clone(),
                    mismatch: TopLevelMismatch::Type,
                });
            }
        }
        (Matrix(m1), Matrix(m2)) => {
            if m1.ty != m2.ty {
                return Some(LayoutMismatch::TopLevel {
                    layout_left: layout1.clone(),
                    layout_right: layout2.clone(),
                    mismatch: TopLevelMismatch::Type,
                });
            }
        }
        (Array(a1), Array(a2)) => {
            // Recursively check element types
            match try_find_mismatch(&a1.element_ty, &a2.element_ty) {
                // In case the mismatch isn't related to a struct field mismatch,
                // we propagate the error upwards to this array. For example
                // Array<Array<f32x1>> vs Array<Array<u32x1>> comparison returns a
                // Array<Array<f32x1>> vs Array<Array<u32x1>> type mismatch and not a
                // Array<f32x1>        vs Array<u32x1>        mismatch
                // Note that we don't change the TopLevelMismatch::ArrayStride fields though.
                Some(LayoutMismatch::TopLevel {
                    layout_left: layout1,
                    layout_right: layout2,
                    mismatch,
                }) => match mismatch {
                    TopLevelMismatch::Type | TopLevelMismatch::ArrayStride { .. } => {
                        return Some(LayoutMismatch::TopLevel {
                            layout_left: layout1.clone(),
                            layout_right: layout2.clone(),
                            mismatch,
                        });
                    }
                },
                m @ Some(LayoutMismatch::Struct { .. }) => return m,
                None => return None,
            }

            // Check array sizes match
            if a1.len != a2.len {
                return Some(LayoutMismatch::TopLevel {
                    layout_left: layout1.clone(),
                    layout_right: layout2.clone(),
                    mismatch: TopLevelMismatch::Type,
                });
            }

            // Check array stride
            if a1.byte_stride != a2.byte_stride {
                return Some(LayoutMismatch::TopLevel {
                    layout_left: layout1.clone(),
                    layout_right: layout2.clone(),
                    mismatch: TopLevelMismatch::ArrayStride {
                        array_left: (**a1).clone(),
                        array_right: (**a2).clone(),
                    },
                });
            }
        }
        (Struct(s1), Struct(s2)) => {
            return try_find_struct_mismatch(s1, s2);
        }
        // Different kinds entirely. Matching exhaustively, so that changes to TypeLayout lead us here.
        (Vector(_) | PackedVector(_) | Matrix(_) | Array(_) | Struct(_), _) => {
            return Some(LayoutMismatch::TopLevel {
                layout_left: layout1.clone(),
                layout_right: layout2.clone(),
                mismatch: TopLevelMismatch::Type,
            });
        }
    }

    None
}

fn try_find_struct_mismatch(struct1: &StructLayout, struct2: &StructLayout) -> Option<LayoutMismatch> {
    for (field_index, (field1, field2)) in struct1.fields.iter().zip(struct2.fields.iter()).enumerate() {
        // Order of checks is important here. We check in order
        // - field name
        // - field type and if the field is/contains a struct, recursively check its fields
        // - field offset
        // - field byte size
        if field1.name != field2.name {
            return Some(LayoutMismatch::Struct {
                struct_left: struct1.clone(),
                struct_right: struct2.clone(),
                mismatch: StructMismatch::FieldName { field_index },
            });
        }

        // Recursively check field types
        if let Some(inner_mismatch) = try_find_mismatch(&field1.ty, &field2.ty) {
            match inner_mismatch {
                // If it's a top-level mismatch, convert it to a field mismatch
                LayoutMismatch::TopLevel {
                    mismatch: TopLevelMismatch::Type,
                    ..
                } => {
                    return Some(LayoutMismatch::Struct {
                        struct_left: struct1.clone(),
                        struct_right: struct2.clone(),
                        mismatch: StructMismatch::FieldType { field_index },
                    });
                }
                LayoutMismatch::TopLevel {
                    mismatch:
                        TopLevelMismatch::ArrayStride {
                            array_left,
                            array_right,
                        },
                    ..
                } => {
                    return Some(LayoutMismatch::Struct {
                        struct_left: struct1.clone(),
                        struct_right: struct2.clone(),
                        mismatch: StructMismatch::FieldArrayStride {
                            field_index,
                            array_left,
                            array_right,
                        },
                    });
                }
                // Pass through nested struct mismatches
                struct_mismatch @ LayoutMismatch::Struct { .. } => return Some(struct_mismatch),
            }
        }

        // Check field offset
        if field1.rel_byte_offset != field2.rel_byte_offset {
            return Some(LayoutMismatch::Struct {
                struct_left: struct1.clone(),
                struct_right: struct2.clone(),
                mismatch: StructMismatch::FieldOffset { field_index },
            });
        }

        // Check field byte size
        if field1.ty.byte_size() != field2.ty.byte_size() {
            return Some(LayoutMismatch::Struct {
                struct_left: struct1.clone(),
                struct_right: struct2.clone(),
                mismatch: StructMismatch::FieldByteSize { field_index },
            });
        }
    }

    // Check field count
    if struct1.fields.len() != struct2.fields.len() {
        return Some(LayoutMismatch::Struct {
            struct_left: struct1.clone(),
            struct_right: struct2.clone(),
            mismatch: StructMismatch::FieldCount,
        });
    }

    None
}
