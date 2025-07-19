use std::fmt::Write;

use crate::{
    any::layout::{ElementLayout, StructLayout},
    common::prettify::set_color,
    frontend::rust_types::type_layout::ArrayLayout,
    TypeLayout,
};

use super::{layoutable::LayoutableType, Repr, DEFAULT_REPR, TypeLayoutSemantics};

pub struct TypeLayoutCompatibleWith<AddressSpace> {
    recipe: LayoutableType,
    _phantom: std::marker::PhantomData<AddressSpace>,
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
        use TypeLayoutSemantics::*;

        let types_are_the_same = "The LayoutError is produced by comparing two semantically equivalent TypeLayout, so all (nested) types are the same";
        match &self.mismatch {
            LayoutMismatch::TopLevel {
                layout1,
                layout2,
                mismatch,
            } => match mismatch {
                TopLevelMismatch::Type => unreachable!("{}", types_are_the_same),
                TopLevelMismatch::ArrayStride => {
                    let (element1, element2) = match (&layout1.kind, &layout2.kind) {
                        (Array(e1, _), Array(e2, _)) => (e1, e2),
                        _ => unreachable!("Array stride error can only occur if the layouts are arrays"),
                    };

                    writeln!(
                        f,
                        "`{}` requires a stride of {} in {}, but has a stride of {}.",
                        layout1.to_string_plain(),
                        element2.byte_stride,
                        self.address_space,
                        element1.byte_stride
                    )?;
                }
            },
            LayoutMismatch::Struct {
                layout1,
                layout2,
                mismatch,
            } => {
                let field_index = match mismatch {
                    StructMismatch::FieldArrayStride {
                        field_index,
                        array_left,
                        array_right,
                    } => {
                        writeln!(
                            f,
                            "`{}` in {} requires a stride of {} in {}, but has a stride of {}.",
                            *layout1.name, array_left.byte_stride, self.address_space, array_right.byte_stride
                        )?;
                    }
                    StructMismatch::FieldCount => unreachable!("{}", types_are_the_same),
                };
            }
        }
        todo!()
    }
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

#[derive(Debug, Clone)]
pub enum LayoutMismatch {
    /// `layout1` and `layout2` are always top level layouts.
    ///
    /// For example, in case of `Array<Array<f32x3>>`, it's always the layout
    /// of `Array<Array<f32x3>>` even if the array stride mismatch
    /// may be happening for the inner `Array<f32x3>`.
    TopLevel {
        layout1: TypeLayout,
        layout2: TypeLayout,
        mismatch: TopLevelMismatch,
    },
    Struct {
        layout1: StructLayout,
        layout2: StructLayout,
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

#[derive(Debug, Clone)]
pub enum StructMismatch {
    FieldCount,
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
}

fn try_find_mismatch(layout1: &TypeLayout, layout2: &TypeLayout) -> Option<LayoutMismatch> {
    use TypeLayoutSemantics::*;

    // First check if the kinds are the same type
    match (&layout1.kind, &layout2.kind) {
        (Vector(v1), Vector(v2)) => {
            if v1 != v2 {
                return Some(LayoutMismatch::TopLevel {
                    layout1: layout1.clone(),
                    layout2: layout2.clone(),
                    mismatch: TopLevelMismatch::Type,
                });
            }
        }
        (PackedVector(p1), PackedVector(p2)) => {
            if p1 != p2 {
                return Some(LayoutMismatch::TopLevel {
                    layout1: layout1.clone(),
                    layout2: layout2.clone(),
                    mismatch: TopLevelMismatch::Type,
                });
            }
        }
        (Matrix(m1), Matrix(m2)) => {
            if m1 != m2 {
                return Some(LayoutMismatch::TopLevel {
                    layout1: layout1.clone(),
                    layout2: layout2.clone(),
                    mismatch: TopLevelMismatch::Type,
                });
            }
        }
        (Array(a1), Array(a2)) => {
            // Check array sizes match
            if a1.len != a2.len {
                return Some(LayoutMismatch::TopLevel {
                    layout1: layout1.clone(),
                    layout2: layout2.clone(),
                    mismatch: TopLevelMismatch::Type,
                });
            }

            // Check array stride
            if a1.byte_stride != a2.byte_stride {
                return Some(LayoutMismatch::TopLevel {
                    layout1: layout1.clone(),
                    layout2: layout2.clone(),
                    mismatch: TopLevelMismatch::ArrayStride {
                        array_left: (**a1).clone(),
                        array_right: (**a2).clone(),
                    },
                });
            }

            // Recursively check element types
            match try_find_mismatch(&a1.element_ty, &a2.element_ty) {
                // In case the mismatch isn't related to a struct field mismatch,
                // we propagate the error upwards to this array. For example
                // Array<Array<f32x1>> vs Array<Array<u32x1>> comparison returns a
                // Array<Array<f32x1>> vs Array<Array<u32x1>> type mismatch and not a
                // Array<f32x1>        vs Array<u32x1>        mismatch
                // Note that we don't change the TopLevelMismatch::ArrayStride fields though.
                Some(LayoutMismatch::TopLevel {
                    layout1,
                    layout2,
                    mismatch,
                }) => match mismatch {
                    TopLevelMismatch::Type | TopLevelMismatch::ArrayStride { .. } => {
                        return Some(LayoutMismatch::TopLevel {
                            layout1: layout1.clone(),
                            layout2: layout2.clone(),
                            mismatch,
                        });
                    }
                },
                m @ Some(LayoutMismatch::Struct { .. }) => return m,
                None => return None,
            }
        }
        (Structure(s1), Structure(s2)) => {
            return try_find_struct_mismatch(s1, s2);
        }
        // Different kinds entirely. Matching exhaustively, so that changes to TypeLayout lead us here.
        (Vector(_) | PackedVector(_) | Matrix(_) | Array(_) | Structure(_), _) => {
            return Some(LayoutMismatch::TopLevel {
                layout1: layout1.clone(),
                layout2: layout2.clone(),
                mismatch: TopLevelMismatch::Type,
            });
        }
    }

    None
}

fn try_find_struct_mismatch(struct1: &StructLayout, struct2: &StructLayout) -> Option<LayoutMismatch> {
    // Check field count
    if struct1.fields.len() != struct2.fields.len() {
        return Some(LayoutMismatch::Struct {
            layout1: struct1.clone(),
            layout2: struct2.clone(),
            mismatch: StructMismatch::FieldCount,
        });
    }

    for (field_index, (field1, field2)) in struct1.fields.iter().zip(struct2.fields.iter()).enumerate() {
        // Check field offset
        if field1.rel_byte_offset != field2.rel_byte_offset {
            return Some(LayoutMismatch::Struct {
                layout1: struct1.clone(),
                layout2: struct2.clone(),
                mismatch: StructMismatch::FieldOffset { field_index },
            });
        }

        // Check field byte size
        if field1.field.ty.byte_size != field2.field.ty.byte_size {
            return Some(LayoutMismatch::Struct {
                layout1: struct1.clone(),
                layout2: struct2.clone(),
                mismatch: StructMismatch::FieldByteSize { field_index },
            });
        }

        // Recursively check field types
        if let Some(inner_mismatch) = try_find_mismatch(&field1.field.ty, &field2.field.ty) {
            match inner_mismatch {
                // If it's a top-level mismatch, convert it to a field mismatch
                LayoutMismatch::TopLevel {
                    mismatch: TopLevelMismatch::Type,
                    ..
                } => {
                    return Some(LayoutMismatch::Struct {
                        layout1: struct1.clone(),
                        layout2: struct2.clone(),
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
                        layout1: struct1.clone(),
                        layout2: struct2.clone(),
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
    }

    None
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiffLine {
    Shared(String),
    Left(String),
    Right(String),
}

pub fn diff_lines(left: &str, right: &str) -> Vec<DiffLine> {
    let left_lines: Vec<&str> = left.lines().collect();
    let right_lines: Vec<&str> = right.lines().collect();

    let mut result = Vec::new();
    let mut left_idx = 0;
    let mut right_idx = 0;

    while left_idx < left_lines.len() && right_idx < right_lines.len() {
        if left_lines[left_idx] == right_lines[right_idx] {
            result.push(DiffLine::Shared(left_lines[left_idx].to_string()));
            left_idx += 1;
            right_idx += 1;
        } else {
            // Find the next matching line
            let mut found_match = false;
            for i in (left_idx + 1)..left_lines.len() {
                if let Some(j) = right_lines[(right_idx + 1)..]
                    .iter()
                    .position(|&line| line == left_lines[i])
                {
                    let right_match_idx = right_idx + 1 + j;

                    // Add differing lines before the match
                    for k in left_idx..i {
                        result.push(DiffLine::Left(left_lines[k].to_string()));
                    }
                    for k in right_idx..right_match_idx {
                        result.push(DiffLine::Right(right_lines[k].to_string()));
                    }

                    left_idx = i;
                    right_idx = right_match_idx;
                    found_match = true;
                    break;
                }
            }

            if !found_match {
                result.push(DiffLine::Left(left_lines[left_idx].to_string()));
                result.push(DiffLine::Right(right_lines[right_idx].to_string()));
                left_idx += 1;
                right_idx += 1;
            }
        }
    }

    // Add remaining lines
    while left_idx < left_lines.len() {
        result.push(DiffLine::Left(left_lines[left_idx].to_string()));
        left_idx += 1;
    }

    while right_idx < right_lines.len() {
        result.push(DiffLine::Right(right_lines[right_idx].to_string()));
        right_idx += 1;
    }

    result
}
