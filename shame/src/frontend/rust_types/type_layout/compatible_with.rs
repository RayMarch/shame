use std::fmt::Write;

use crate::{
    any::layout::StructLayout,
    call_info,
    common::prettify::{set_color, UnwrapOrStr},
    frontend::rust_types::type_layout::{
        display::LayoutInfo,
        eq::{try_find_mismatch, LayoutMismatch, StructMismatch, TopLevelMismatch},
        ArrayLayout,
    },
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
/// - is representable in the target language
/// - satisfies the layout requirements of the address space
///
/// Wgsl has only one representation of types - there is no choice between std140 and std430
/// like in glsl - so to be representable in wgsl means that the type layout produced by
/// the recipe is the same as the one produced by the same recipe but with all structs
/// in the recipe using the Repr::Wgsl layout algorithm. Aditionally, all custom attributes used
/// by the recipe need to be support by wgsl, which are only the struct field attributes
/// `#[align(N)]` and `#[size(N)]` currently.
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
                // all structs in the recipe using Repr::Wgsl is the same and all custom attributes
                // used by the recipe are supported by wgsl, which is checked in `TypeLayoutRecipe::layout`
                // TODO(chronicl) line above
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
                            return Err(AddressSpaceError::RequirementsNotSatisfied(LayoutError {
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

#[derive(Debug, Clone, Copy, PartialEq)]
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
    #[error("{} is not representable in {}:\n{0}", .0.recipe, .0.address_space.language())]
    NotRepresentable(LayoutError),
    #[error("Address space requirements not satisfied:\n{0}")]
    RequirementsNotSatisfied(LayoutError),
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
                        writeln!(f, "The full layout of `{}` is:\n", struct_left.short_name())?;
                        write_struct(f, struct_left, Some(*field_index), self.colored)?;

                        writeln!(f, "\nPotential solutions include:")?;

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
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline_kind::Render;
    use crate::{self as shame, EncodingGuard, ThreadIsAlreadyEncoding};
    use shame as sm;
    use shame::{aliases::*, GpuLayout};

    const PRINT: bool = true;

    macro_rules! is_struct_mismatch {
        ($result:expr, $as_error:ident, $mismatch:pat) => {
            {
                if let Err(e) = &$result && PRINT {
                    println!("{e}");
                }
                matches!(
                    $result,
                    Err(AddressSpaceError::$as_error(LayoutError {
                        mismatch: LayoutMismatch::Struct {
                            mismatch: $mismatch,
                            ..
                        },
                        ..
                    }))
                )
            }
        };
    }

    fn enable_color() -> Result<EncodingGuard<Render>, ThreadIsAlreadyEncoding> {
        sm::start_encoding(sm::Settings::default())
    }

    #[test]
    fn test_field_offset_error_not_representable() {
        let _guard = enable_color();

        #[derive(sm::GpuLayout)]
        #[gpu_repr(packed)]
        struct A {
            a: f32x1,
            // has offset 4, but in wgsl's storage/uniform address space, it needs to be 16 byte aligned
            b: f32x3,
        }

        // The error variant is NotRepresentable, because there is no way to represent it in wgsl,
        // because an offset of 4 is not possible for f32x3, because it is 16 byte aligned.
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<WgslStorage>::try_from(A::layout_recipe()),
            NotRepresentable,
            StructMismatch::FieldOffset { .. }
        ));
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<WgslUniform>::try_from(A::layout_recipe()),
            NotRepresentable,
            StructMismatch::FieldOffset { .. }
        ));

        #[derive(sm::GpuLayout)]
        #[gpu_repr(wgsl)]
        struct B {
            a: f32x1,
            // offset 4, but wgsl's uniform address space requires 16 byte alignment
            b: sm::Array<f32x1>,
        }

        // The error variant is RequirementsNotSatisfied, because B is representable in wgsl,
        // because it's Repr::Wgsl, but it does not satisfy the requirements of wgsl's uniform address space,
        // because the array has an align of 4 in Repr::Wgsl, but an align of 16 in Repr::WgslUniform.
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<WgslUniform>::try_from(B::layout_recipe()),
            RequirementsNotSatisfied,
            StructMismatch::FieldOffset { .. }
        ));
    }
}
