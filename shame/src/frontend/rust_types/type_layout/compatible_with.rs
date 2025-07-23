use std::fmt::Write;

use crate::{
    any::layout::StructLayout,
    call_info,
    common::prettify::{set_color, UnwrapOrStr},
    frontend::rust_types::type_layout::{display::LayoutInfo, ArrayLayout},
    ir::{ir_type::max_u64_po2_dividing, recording::Context},
    TypeLayout,
};

use super::{recipe::TypeLayoutRecipe, Repr};

/// `TypeLayoutCompatibleWith<AddressSpace>` is a `TypeLayoutRecipe` with the additional
/// guarantee that the resulting `TypeLayout` is useable in the specified `AddressSpace`.
///
/// The address spaces are language specific. For example, in WGSL there are two address spaces:
/// [`WgslStorage`] and [`WgslUniform`].
///
/// To be "useable" or "compatible with" an address space means that the type layout
/// - satisfies the layout requirements of the address space
/// - is representable in the target language
///
/// Wgsl has only one representation of types - there is no choice between std140 and std430
/// like in glsl - so to be representable in wgsl means that the type layout produced by
/// the recipe is the same as the one produced by the same recipe but with all structs
/// in the recipe using Repr::Wgsl, which is what shame calls wgsl's representation/layout algorithm.
pub struct TypeLayoutCompatibleWith<AddressSpace> {
    recipe: TypeLayoutRecipe,
    _phantom: std::marker::PhantomData<AddressSpace>,
}

impl<AS: AddressSpace> TypeLayoutCompatibleWith<AS> {
    pub fn try_from(recipe: TypeLayoutRecipe) -> Result<Self, AddressSpaceError> {
        let address_space = AS::ADDRESS_SPACE;
        let layout = recipe.layout();

        match (address_space, layout.byte_size()) {
            // Must be sized in wgsl's uniform address space
            (AddressSpaceEnum::WgslUniform, None) => return Err(AddressSpaceError::MustBeSized(recipe, address_space)),
            (AddressSpaceEnum::WgslUniform, Some(_)) | (AddressSpaceEnum::WgslStorage, _) => {}
        }

        // Check that the type layout is representable in the target language
        match address_space {
            AddressSpaceEnum::WgslStorage | AddressSpaceEnum::WgslUniform => {
                // Wgsl has only one type representation: Repr::Wgsl, so the layout produced by the recipe
                // is representable in wgsl iff the layout produced by the same recipe but with
                // all structs in the recipe using Repr::Wgsl is the same.
                let recipe_unified = recipe.to_unified_repr(Repr::Wgsl);
                let layout_unified = recipe_unified.layout();
                if layout != layout_unified {
                    match try_find_mismatch(&layout, &layout_unified) {
                        Some(mismatch) => {
                            return Err(AddressSpaceError::NotRepresentable(LayoutError {
                                recipe,
                                address_space,
                                mismatch,
                                colored: Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages)
                                    .unwrap_or(false),
                            }));
                        }
                        None => return Err(AddressSpaceError::UnknownLayoutError(recipe, address_space)),
                    }
                }
            }
        }

        // Check that the type layout satisfies the requirements of the address space
        match address_space {
            AddressSpaceEnum::WgslStorage => {
                // As long as the recipe is representable in wgsl, it satifies the storage address space requirements.
                // We already checked that the recipe is representable in wgsl above.
            }
            AddressSpaceEnum::WgslUniform => {
                // Repr::WgslUniform is made for exactly this purpose: to check that the type layout
                // satisfies the requirements of wgsl's uniform address space.
                let recipe_unified = recipe.to_unified_repr(Repr::WgslUniform);
                let layout_unified = recipe_unified.layout();
                if layout != layout_unified {
                    match try_find_mismatch(&layout, &layout_unified) {
                        Some(mismatch) => {
                            return Err(AddressSpaceError::DoesntSatisfyRequirements(LayoutError {
                                recipe,
                                address_space,
                                mismatch,
                                colored: Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages)
                                    .unwrap_or(false),
                            }));
                        }
                        None => return Err(AddressSpaceError::UnknownLayoutError(recipe, address_space)),
                    }
                }
            }
        }


        Ok(Self {
            recipe,
            _phantom: std::marker::PhantomData,
        })
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
impl AddressSpaceEnum {
    pub fn language(&self) -> &'static str {
        match self {
            AddressSpaceEnum::WgslStorage => "wgsl",
            AddressSpaceEnum::WgslUniform => "wgsl",
        }
    }
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum AddressSpaceError {
    #[error("Address space requirements not satisfied:\n{0}")]
    DoesntSatisfyRequirements(LayoutError),
    #[error("{} is not representable in {}:\n{0}", .0.recipe, .0.address_space.language())]
    NotRepresentable(LayoutError),
    #[error("Unknown layout error occured for {0} in {1}.")]
    UnknownLayoutError(TypeLayoutRecipe, AddressSpaceEnum),
    #[error(
        "The size of `{0}` on the gpu is not known at compile time. {1} \
     requires that the size of {0} on the gpu is known at compile time."
    )]
    MustBeSized(TypeLayoutRecipe, AddressSpaceEnum),
    #[error("{0} contains a `PackedVector`, which are not allowed in {1}.")]
    MayNotContainPackedVec(TypeLayoutRecipe, AddressSpaceEnum),
}

#[derive(Debug, Clone)]
pub struct LayoutError {
    recipe: TypeLayoutRecipe,
    address_space: AddressSpaceEnum,
    mismatch: LayoutMismatch,
    colored: bool,
}

impl std::error::Error for LayoutError {}
impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let types_are_the_same = "The LayoutError is produced by comparing two semantically equivalent TypeLayouts, so all (nested) types are the same";
        match &self.mismatch {
            LayoutMismatch::TopLevel {
                layout_left,
                layout_right,
                mismatch,
            } => match mismatch {
                TopLevelMismatch::Type => unreachable!("{}", types_are_the_same),
                TopLevelMismatch::ArrayStride {
                    array_left,
                    array_right,
                } => {
                    if array_left.short_name() == array_right.short_name() {
                        writeln!(
                            f,
                            "`{}` requires a stride of {} in {}, but has a stride of {}.",
                            array_left.short_name(),
                            array_right.byte_stride,
                            self.address_space,
                            array_left.byte_stride
                        )?;
                    } else {
                        writeln!(
                            f,
                            "`{}` in `{}` requires a stride of {} in {}, but has a stride of {}.",
                            array_left.short_name(),
                            layout_left.short_name(),
                            array_right.byte_stride,
                            self.address_space,
                            array_left.byte_stride
                        )?;
                    }
                }
                // TODO(chronicl) fix byte size message for when the byte size mismatch is happening
                // in a nested array
                TopLevelMismatch::ByteSize { .. } => {
                    writeln!(
                        f,
                        "`{}` has a byte size of {} in {}, but has a byte size of {}.",
                        layout_left.short_name(),
                        UnwrapOrStr(layout_left.byte_size(), ""),
                        self.address_space,
                        UnwrapOrStr(layout_right.byte_size(), "")
                    )?;
                }
            },
            LayoutMismatch::Struct {
                struct_left,
                struct_right,
                mismatch,
            } => {
                match mismatch {
                    StructMismatch::FieldLayout {
                        field_index,
                        mismatch:
                            TopLevelMismatch::ArrayStride {
                                array_left,
                                array_right,
                            },
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
                        write_struct(f, struct_left, Some(*field_index), self.colored)?;
                    }
                    StructMismatch::FieldLayout {
                        field_index,
                        mismatch: TopLevelMismatch::ByteSize { left, right },
                    } => {
                        let field_left = &struct_left.fields[*field_index];
                        let field_right = &struct_right.fields[*field_index];

                        // TODO(chronicl) fix byte size message for when the byte size mismatch is happening
                        // in a nested array
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
                        write_struct(f, struct_left, Some(*field_index), self.colored)?;
                    }
                    StructMismatch::FieldOffset { field_index } => {
                        let field_left = &struct_left.fields[*field_index];
                        let field_right = &struct_right.fields[*field_index];
                        let field_name = &field_left.name;
                        let offset = field_left.rel_byte_offset;
                        let expected_align = field_right.ty.align().as_u64();
                        let actual_align = max_u64_po2_dividing(field_left.rel_byte_offset);

                        writeln!(
                            f,
                            "Field `{}` in `{}` needs to be {} byte aligned in {}, but has a byte-offset of {}, which is only {} byte aligned",
                            field_name, struct_left.name, expected_align, self.address_space, offset, actual_align
                        )?;
                        writeln!(f, "The full layout of `{}` is:", struct_left.short_name())?;
                        write_struct(f, struct_left, Some(*field_index), self.colored)?;

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
                            AddressSpaceEnum::WgslUniform | AddressSpaceEnum::WgslStorage => writeln!(
                                f,
                                "More info about the {} can be found at https://www.w3.org/TR/WGSL/#address-space-layout-constraints",
                                self.address_space
                            )?,
                        }
                    }
                    StructMismatch::FieldCount |
                    StructMismatch::FieldName { .. } |
                    StructMismatch::FieldLayout {
                        mismatch: TopLevelMismatch::Type,
                        ..
                    } => {
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

/// Contains information about the layout mismatch between two `TypeLayout`s.
///
/// The type layouts are traversed depth-first and the first mismatch encountered
/// is reported at its deepest level.
///
/// In case of nested structs this means that if the field `a` in
/// ```
/// struct A {
///     a: struct B { ... }
/// }
/// struct AOther {
///     a: struct BOther { ... }
/// }
/// ```
/// mismatches, because `B` and `BOther` don't match, then the exact mismatch
/// between `B` and `BOther` is reported and not the field mismatch of `a` in `A` and `AOther`.
///
/// Nested arrays are reported in two levels: `LayoutMismatch::TopLevel` contains the
/// top level / outer most array layout and a  `TopLevelMismatch`, which contains the
/// inner type layout where the mismatch is happening.
/// For example if there is an array stride mismatch of the inner array of `Array<Array<f32x3>>`,
/// then `LayoutMismatch::TopLevel` contains the layout of `Array<Array<f32x3>>` and
/// a `TopLevelMismatch::ArrayStride` with the layout of `Array<f32x3>`.
///
/// A field of nested arrays in a struct is handled in the same way by
/// `LayoutMismatch::Struct` containing the field index, which let's us access the outer
/// most array, and a `TopLevelMismatch`, which let's us access the inner type layout
/// where the mismatch is happening.
#[derive(Debug, Clone)]
pub enum LayoutMismatch {
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
    ByteSize {
        left: TypeLayout,
        right: TypeLayout,
    },
    ArrayStride {
        array_left: ArrayLayout,
        array_right: ArrayLayout,
    },
}

/// Field count is checked last.
#[derive(Debug, Clone)]
pub enum StructMismatch {
    FieldName {
        field_index: usize,
    },
    FieldLayout {
        field_index: usize,
        mismatch: TopLevelMismatch,
    },
    FieldOffset {
        field_index: usize,
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
                // Update the top level layouts and propagate the LayoutMismatch
                Some(LayoutMismatch::TopLevel {
                    layout_left: layout1,
                    layout_right: layout2,
                    mismatch,
                }) => {
                    return Some(LayoutMismatch::TopLevel {
                        layout_left: layout1.clone(),
                        layout_right: layout2.clone(),
                        mismatch,
                    });
                }
                // Struct mismatch, so it's not a top-level mismatch anymore
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

    // Check byte size
    if layout1.byte_size() != layout2.byte_size() {
        return Some(LayoutMismatch::TopLevel {
            layout_left: layout1.clone(),
            layout_right: layout2.clone(),
            mismatch: TopLevelMismatch::ByteSize {
                left: layout1.clone(),
                right: layout2.clone(),
            },
        });
    }

    None
}

fn try_find_struct_mismatch(struct1: &StructLayout, struct2: &StructLayout) -> Option<LayoutMismatch> {
    for (field_index, (field1, field2)) in struct1.fields.iter().zip(struct2.fields.iter()).enumerate() {
        // Order of checks is important here. We check in order
        // - field name
        // - field type, byte size and if the field is/contains a struct, recursively check its fields
        // - field offset
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
                LayoutMismatch::TopLevel { mismatch, .. } => {
                    return Some(LayoutMismatch::Struct {
                        struct_left: struct1.clone(),
                        struct_right: struct2.clone(),
                        mismatch: StructMismatch::FieldLayout { field_index, mismatch },
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
