use std::fmt::Write;

use crate::{
    any::layout::StructLayout,
    call_info,
    common::prettify::{set_color, UnwrapOrStr},
    frontend::{
        encoding::buffer::BufferAddressSpaceEnum,
        rust_types::type_layout::{
            display::LayoutInfo,
            eq::{try_find_mismatch, LayoutMismatch, StructMismatch, TopLevelMismatch},
            recipe::to_layout::RecipeContains,
            ArrayLayout,
        },
    },
    ir::{ir_type::max_u64_po2_dividing, recording::Context},
    mem, BufferAddressSpace, Language, TypeLayout,
};

use super::{recipe::TypeLayoutRecipe, Repr};

pub use mem::{Storage, Uniform};

/// `TypeLayoutCompatibleWith<AddressSpace>` is a [`TypeLayoutRecipe`] with the additional
/// guarantee that the [`TypeLayout`] it produces is compatible with the specified `AddressSpace`.
///
/// Address space requirements are language specific, which is why `TypeLayoutCompatibleWith` constructors
/// additionally take a [`Language`] parameter.
///
/// To be "compatible with" an address space means that
/// - the recipe is **valid** ([`TypeLayoutRecipe::layout`] succeeds)
/// - the type layout **satisfies the layout requirements** of the address space
/// - the type layout recipe is **representable** in the target language
///
/// To be representable in a language means that the type layout recipe, can be expressed in the
/// language's type system:
/// 1. all types in the recipe can be expressed in the target language (for example `bool` or `PackedVector` can't be expressed in wgsl)
/// 2. the available layout algorithms in the target language can produce the same layout as the one produced by the recipe
/// 3. support for the custom attributes the recipe uses, such as `#[align(N)]` and `#[size(N)]`.
///    Custom attributes may be rejected by the target language itself (NotRepresentable error)
///    or by the layout algorithms specified in the recipe (InvalidRecipe error during `TypeLayoutRecipe -> TypeLayout` conversion).
///
/// For example for wgsl we have
/// 1. PackedVector can be part of a recipe, but can not be expressed in wgsl,
///    so a recipe containing a PackedVector is not representable in wgsl.
/// 2. Wgsl has only one layout algorithm (`Repr::Wgsl`) - there is no choice between std140 and std430
///    like in glsl - so to be representable in wgsl the type layout produced by the recipe
///    has to be the same as the one produced by the same recipe but using exclusively the Repr::Wgsl
///    layout algorithm instead of the layout algorithms specified in the recipe.
/// 3. Wgsl only supports custom struct field attributes `#[align(N)]` and `#[size(N)]` currently.
#[derive(Debug, Clone)]
pub struct TypeLayoutCompatibleWith<AddressSpace> {
    recipe: TypeLayoutRecipe,
    _phantom: std::marker::PhantomData<AddressSpace>,
}

impl<AS: BufferAddressSpace> TypeLayoutCompatibleWith<AS> {
    pub fn try_from(language: Language, recipe: TypeLayoutRecipe) -> Result<Self, AddressSpaceError> {
        let address_space = AS::BUFFER_ADDRESS_SPACE;
        let layout = recipe.layout();

        match (language, address_space, layout.byte_size()) {
            // Must be sized in wgsl's uniform address space
            (Language::Wgsl, BufferAddressSpaceEnum::Uniform, None) => {
                return Err(RequirementsNotSatisfied::MustBeSized(recipe, language, address_space).into());
            }
            (Language::Wgsl, BufferAddressSpaceEnum::Uniform, Some(_))
            | (Language::Wgsl, BufferAddressSpaceEnum::Storage, _) => {}
        }

        // Check that the recipe is representable in the target language.
        // See `TypeLayoutCompatibleWith` docs for more details on what it means to be representable.
        match (language, address_space) {
            (Language::Wgsl, BufferAddressSpaceEnum::Storage | BufferAddressSpaceEnum::Uniform) => {
                // We match like this, so that future additions to `RecipeContains` lead us here.
                match RecipeContains::CustomFieldAlign {
                    // supported in wgsl
                    RecipeContains::CustomFieldAlign | RecipeContains::CustomFieldSize |
                   // not supported in wgsl
                    RecipeContains::PackedVector => {
                        if recipe.contains(RecipeContains::PackedVector) {
                            return Err(NotRepresentable::MayNotContain(
                                recipe,
                                language,
                                address_space,
                                RecipeContains::PackedVector,
                            )
                            .into());
                        }
                    }
                }

                // Wgsl has only one layout algorithm
                let recipe_wgsl = recipe.to_unified_repr(Repr::Wgsl);
                let layout_wgsl = recipe_wgsl.layout_with_default_repr(Repr::Wgsl);
                if layout != layout_wgsl {
                    match try_find_mismatch(&layout, &layout_wgsl) {
                        Some(mismatch) => {
                            return Err(NotRepresentable::LayoutError(LayoutError {
                                recipe,
                                kind: LayoutErrorKind::NotRepresentable,
                                language,
                                address_space,
                                mismatch,
                                colored: Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages)
                                    .unwrap_or(false),
                            })
                            .into());
                        }
                        None => return Err(NotRepresentable::UnknownLayoutError(recipe, address_space).into()),
                    }
                }
            }
        }

        // Check that the type layout satisfies the requirements of the address space
        match (language, address_space) {
            (Language::Wgsl, BufferAddressSpaceEnum::Storage) => {
                // As long as the recipe is representable in wgsl, it satifies the storage address space requirements.
                // We already checked that the recipe is representable in wgsl above.
            }
            (Language::Wgsl, BufferAddressSpaceEnum::Uniform) => {
                // Repr::WgslUniform is made for exactly this purpose: to check that the type layout
                // satisfies the requirements of wgsl's uniform address space.
                let recipe_unified = recipe.to_unified_repr(Repr::WgslUniform);
                let layout_unified = recipe_unified.layout();
                if layout != layout_unified {
                    match try_find_mismatch(&layout, &layout_unified) {
                        Some(mismatch) => {
                            return Err(RequirementsNotSatisfied::LayoutError(LayoutError {
                                recipe,
                                kind: LayoutErrorKind::RequirementsNotSatisfied,
                                language,
                                address_space,
                                mismatch,
                                colored: Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages)
                                    .unwrap_or(false),
                            })
                            .into());
                        }
                        None => return Err(RequirementsNotSatisfied::UnknownLayoutError(recipe, address_space).into()),
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

#[derive(thiserror::Error, Debug, Clone)]
pub enum AddressSpaceError {
    #[error("{0}")]
    NotRepresentable(#[from] NotRepresentable),
    #[error("{0}")]
    RequirementsNotSatisfied(#[from] RequirementsNotSatisfied),
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum NotRepresentable {
    #[error("{0}")]
    LayoutError(LayoutError),
    #[error("{0} contains a {3}, which is not allowed in {1}'s {2}.")]
    MayNotContain(TypeLayoutRecipe, Language, BufferAddressSpaceEnum, RecipeContains),
    #[error("Unknown layout error occured for {0} in {1}.")]
    UnknownLayoutError(TypeLayoutRecipe, BufferAddressSpaceEnum),
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum RequirementsNotSatisfied {
    #[error("{0}")]
    LayoutError(LayoutError),
    #[error(
        "The size of `{0}` on the gpu is not known at compile time. {1}'s {2} \
     requires that the size of {0} on the gpu is known at compile time."
    )]
    MustBeSized(TypeLayoutRecipe, Language, BufferAddressSpaceEnum),
    #[error("Unknown layout error occured for {0} in {1}.")]
    UnknownLayoutError(TypeLayoutRecipe, BufferAddressSpaceEnum),
}

#[derive(Debug, Clone)]
pub struct LayoutError {
    recipe: TypeLayoutRecipe,
    mismatch: LayoutMismatch,

    /// Used to adjust the error message
    /// to fit `NotRepresentable` or `RequirementsNotSatisfied`.
    kind: LayoutErrorKind,
    language: Language,
    address_space: BufferAddressSpaceEnum,

    colored: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum LayoutErrorKind {
    NotRepresentable,
    RequirementsNotSatisfied,
}

impl LayoutError {
    /// Returns the context the error occurred in. The "{language}" in case of a `NotRepresentable` error,
    /// or the "{language}'s {address_space}" in case of a `RequirementsNotSatisfied` error.
    fn context(&self) -> &'static str {
        match self.kind {
            LayoutErrorKind::NotRepresentable => match self.language {
                Language::Wgsl => "wgsl",
            },
            LayoutErrorKind::RequirementsNotSatisfied => match (self.language, self.address_space) {
                (Language::Wgsl, BufferAddressSpaceEnum::Storage) => "wgsl's storage address space",
                (Language::Wgsl, BufferAddressSpaceEnum::Uniform) => "wgsl's uniform address space",
            },
        }
    }
}

impl std::error::Error for LayoutError {}
impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            LayoutErrorKind::NotRepresentable => {
                writeln!(f, "`{}` is not representable in {}:", self.recipe, self.language)?;
            }
            LayoutErrorKind::RequirementsNotSatisfied => {
                writeln!(
                    f,
                    "`{}` does not satisfy the layout requirements of {}:",
                    self.recipe,
                    self.context()
                )?;
            }
        }

        match &self.mismatch {
            LayoutMismatch::TopLevel {
                layout_left,
                layout_right,
                mismatch,
            } => write_top_level_mismatch(f, self, layout_left, layout_right, mismatch),
            LayoutMismatch::Struct {
                struct_left,
                struct_right,
                mismatch,
            } => write_struct_mismatch(f, self, struct_left, struct_right, mismatch),
        }
    }
}

fn write_top_level_mismatch(
    f: &mut std::fmt::Formatter<'_>,
    error: &LayoutError,
    layout_left: &TypeLayout,
    layout_right: &TypeLayout,
    mismatch: &TopLevelMismatch,
) -> Result<(), std::fmt::Error> {
    match mismatch {
        TopLevelMismatch::Type => unreachable!(
            "The LayoutError is produced by comparing two semantically equivalent TypeLayouts, so all (nested) types are the same"
        ),
        TopLevelMismatch::ArrayStride {
            array_left,
            array_right,
        } => {
            let outer_most_array_has_mismatch = array_left.short_name() == layout_left.short_name();
            if outer_most_array_has_mismatch {
                writeln!(
                    f,
                    "`{}` requires a stride of {} in {}, but has a stride of {}.",
                    array_left.short_name(),
                    array_right.byte_stride,
                    error.context(),
                    array_left.byte_stride
                )?;
            } else {
                writeln!(
                    f,
                    "`{}` in `{}` requires a stride of {} in {}, but has a stride of {}.",
                    array_left.short_name(),
                    layout_left.short_name(),
                    array_right.byte_stride,
                    error.context(),
                    array_left.byte_stride
                )?;
            }
        }
        TopLevelMismatch::ByteSize { left, right } => {
            let outer_most_array_has_mismatch = left.short_name() == layout_left.short_name();
            if outer_most_array_has_mismatch {
                writeln!(
                    f,
                    "`{}` has a byte size of {} in {}, but has a byte size of {}.",
                    layout_left.short_name(),
                    UnwrapOrStr(layout_left.byte_size(), ""),
                    error.context(),
                    UnwrapOrStr(layout_right.byte_size(), "")
                )?;
            } else {
                writeln!(
                    f,
                    "`{}` in `{}` has a byte size of {} in {}, but has a byte size of {}.",
                    layout_left.short_name(),
                    layout_left.short_name(),
                    UnwrapOrStr(layout_left.byte_size(), ""),
                    error.context(),
                    UnwrapOrStr(layout_right.byte_size(), "")
                )?;
            }
        }
    }
    Ok(())
}

fn write_struct_mismatch(
    f: &mut std::fmt::Formatter<'_>,
    error: &LayoutError,
    struct_left: &StructLayout,
    struct_right: &StructLayout,
    mismatch: &StructMismatch,
) -> Result<(), std::fmt::Error> {
    match mismatch {
        StructMismatch::FieldLayout {
            field_index,
            field_left,
            mismatch:
                TopLevelMismatch::ArrayStride {
                    array_left,
                    array_right,
                },
            ..
        } => {
            let outer_most_array_has_mismatch = field_left.ty.short_name() == array_left.short_name();
            let layout_info = if outer_most_array_has_mismatch {
                LayoutInfo::STRIDE
            } else {
                // if an inner array has the stride mismatch, showing the outer array's stride could be confusing
                LayoutInfo::NONE
            };

            writeln!(
                f,
                "`{}` in `{}` requires a stride of {} in {}, but has a stride of {}.",
                array_left.short_name(),
                struct_left.short_name(),
                array_right.byte_stride,
                error.context(),
                array_left.byte_stride
            )?;
            writeln!(f, "The full layout of `{}` is:\n", struct_left.short_name())?;
            write_struct(f, struct_left, layout_info, Some(*field_index), error.colored)?;
        }
        StructMismatch::FieldLayout {
            field_index,
            field_left,
            field_right,
            mismatch: TopLevelMismatch::ByteSize { left, right },
        } => {
            let outer_most_array_has_mismatch = field_left.ty.short_name() == left.short_name();
            let layout_info = if outer_most_array_has_mismatch {
                LayoutInfo::SIZE
            } else {
                // if an inner array has the byte size mismatch, showing the outer array's byte size could be confusing
                LayoutInfo::NONE
            };

            if !outer_most_array_has_mismatch {
                write!(f, "`{}` in field", left.short_name())?;
            } else {
                write!(f, "Field")?;
            }
            writeln!(
                f,
                " `{}` of `{}` requires a byte size of {} in {}, but has a byte size of {}",
                field_left.name,
                struct_left.name,
                UnwrapOrStr(field_right.ty.byte_size(), ""),
                error.context(),
                UnwrapOrStr(field_left.ty.byte_size(), "")
            )?;
            writeln!(f, "The full layout of `{}` is:", struct_left.short_name())?;
            write_struct(f, struct_left, layout_info, Some(*field_index), error.colored)?;
        }
        StructMismatch::FieldOffset {
            field_index,
            field_left,
            field_right,
        } => {
            let field_name = &field_left.name;
            let offset = field_left.rel_byte_offset;
            let expected_align = field_right.ty.align().as_u64();
            let actual_align = max_u64_po2_dividing(field_left.rel_byte_offset);

            writeln!(
                f,
                "Field `{}` of `{}` needs to be {} byte aligned in {}, but has a byte-offset of {}, which is only {} byte aligned",
                field_name,
                struct_left.name,
                expected_align,
                error.context(),
                offset,
                actual_align
            )?;

            writeln!(f, "The full layout of `{}` is:\n", struct_left.short_name())?;
            write_struct(
                f,
                struct_left,
                LayoutInfo::OFFSET | LayoutInfo::ALIGN | LayoutInfo::SIZE,
                Some(*field_index),
                error.colored,
            )?;

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
            match (error.kind, error.language, error.address_space) {
                (LayoutErrorKind::RequirementsNotSatisfied, Language::Wgsl, BufferAddressSpaceEnum::Uniform) => {
                    writeln!(f, "- use a storage binding instead of a uniform binding")?;
                }
                (
                    LayoutErrorKind::NotRepresentable | LayoutErrorKind::RequirementsNotSatisfied,
                    Language::Wgsl,
                    BufferAddressSpaceEnum::Storage | BufferAddressSpaceEnum::Uniform,
                ) => {}
            }
            writeln!(f)?;

            match error.language {
                Language::Wgsl => {
                    match error.address_space {
                        BufferAddressSpaceEnum::Uniform => writeln!(
                            f,
                            "In {}, structs, arrays and array elements must be at least 16 byte aligned.",
                            error.context()
                        )?,
                        BufferAddressSpaceEnum::Storage => {}
                    }

                    match (error.kind, error.address_space) {
                        (
                            LayoutErrorKind::RequirementsNotSatisfied,
                            BufferAddressSpaceEnum::Uniform | BufferAddressSpaceEnum::Storage,
                        ) => writeln!(
                            f,
                            "More info about the wgsl's {} can be found at https://www.w3.org/TR/WGSL/#address-space-layout-constraints",
                            error.address_space
                        )?,
                        (LayoutErrorKind::NotRepresentable, _) => writeln!(
                            f,
                            "More info about the wgsl's layout algorithm can be found at https://www.w3.org/TR/WGSL/#alignment-and-size"
                        )?,
                    }
                }
            }
        }
        StructMismatch::FieldCount
        | StructMismatch::FieldName { .. }
        | StructMismatch::FieldLayout {
            mismatch: TopLevelMismatch::Type,
            ..
        } => {
            unreachable!(
                "The LayoutError is produced by comparing two semantically equivalent TypeLayouts, so all (nested) types are the same"
            )
        }
    }
    Ok(())
}

fn write_struct<W>(
    f: &mut W,
    s: &StructLayout,
    layout_info: LayoutInfo,
    highlight_field: Option<usize>,
    colored: bool,
) -> std::fmt::Result
where
    W: Write,
{
    let use_256_color_mode = false;

    let mut writer = s.writer(layout_info);
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

    const PRINT: bool = false;

    macro_rules! is_struct_mismatch {
        ($result:expr, $as_error:ident, $mismatch:pat) => {{
            if let Err(e) = &$result
                && PRINT
            {
                println!("{e}");
            }
            matches!(
                $result,
                Err(AddressSpaceError::$as_error($as_error::LayoutError(LayoutError {
                    mismatch: LayoutMismatch::Struct {
                        mismatch: $mismatch,
                        ..
                    },
                    ..
                })))
            )
        }};
    }

    fn enable_color() -> Option<Result<EncodingGuard<Render>, ThreadIsAlreadyEncoding>> {
        PRINT.then(|| sm::start_encoding(sm::Settings::default()))
    }

    #[test]
    fn field_offset_error_not_representable() {
        let _guard = enable_color();

        #[derive(sm::GpuLayout)]
        #[gpu_repr(packed)]
        struct A {
            a: f32x1,
            // has offset 4, but in wgsl's storage/uniform address space, it needs to be 16 byte aligned
            b: f32x3,
        }

        // The error variant is NotRepresentable, because there is no way to represent it in wgsl,
        // because an offset of 4 is not possible for f32x3, because needs to be 16 byte aligned.
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<Storage>::try_from(Language::Wgsl, A::layout_recipe()),
            NotRepresentable,
            StructMismatch::FieldOffset { .. }
        ));
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<Uniform>::try_from(Language::Wgsl, A::layout_recipe()),
            NotRepresentable,
            StructMismatch::FieldOffset { .. }
        ));
    }

    #[test]
    fn wgsl_uniform_array_stride_requirements_not_satisfied() {
        let _guard = enable_color();

        #[derive(sm::GpuLayout)]
        struct A {
            a: f32x1,
            // has stride 4, but wgsl's uniform address space requires a stride of 16.
            // also, has wrong offset, because array align is multiple of 16 in wgsl's uniform address space.
            // array stride error has higher priority than field offset error,
            b: sm::Array<f32x1, sm::Size<1>>,
        }

        // The error variant is RequirementsNotSatisfied, because the array has a stride of 4 in Repr::Packed,
        // but wgsl's uniform address space requires a stride of 16.
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<Uniform>::try_from(Language::Wgsl, A::layout_recipe()),
            RequirementsNotSatisfied,
            StructMismatch::FieldLayout {
                mismatch: TopLevelMismatch::ArrayStride { .. },
                ..
            }
        ));

        // Testing that the error remains the same when nested in another struct
        #[derive(sm::GpuLayout)]
        struct B {
            a: sm::Struct<A>,
        }
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<Uniform>::try_from(Language::Wgsl, B::layout_recipe()),
            RequirementsNotSatisfied,
            StructMismatch::FieldLayout {
                mismatch: TopLevelMismatch::ArrayStride { .. },
                ..
            }
        ));

        // Testing that the error remains the same when nested in an array
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<Uniform>::try_from(
                Language::Wgsl,
                <sm::Array<sm::Struct<A>, sm::Size<1>>>::layout_recipe()
            ),
            RequirementsNotSatisfied,
            StructMismatch::FieldLayout {
                mismatch: TopLevelMismatch::ArrayStride { .. },
                ..
            }
        ));
    }

    #[test]
    fn wgsl_uniform_field_offset_requirements_not_satisfied() {
        let _guard = enable_color();

        #[derive(sm::GpuLayout)]
        struct A {
            a: f32x1,
            b: sm::Struct<B>,
        }
        #[derive(sm::GpuLayout)]
        struct B {
            a: f32x1,
        }

        // The error variant is RequirementsNotSatisfied, because the array has a stride of 4 in Repr::Packed,
        // but wgsl's uniform address space requires a stride of 16.
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<Uniform>::try_from(Language::Wgsl, A::layout_recipe()),
            RequirementsNotSatisfied,
            StructMismatch::FieldOffset { .. }
        ));

        // Testing that the error remains the same when nested in another struct
        #[derive(sm::GpuLayout)]
        struct C {
            a: sm::Struct<A>,
        }
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<Uniform>::try_from(Language::Wgsl, C::layout_recipe()),
            RequirementsNotSatisfied,
            StructMismatch::FieldOffset { .. }
        ));

        // Testing that the error remains the same when nested in an array
        assert!(is_struct_mismatch!(
            TypeLayoutCompatibleWith::<Uniform>::try_from(
                Language::Wgsl,
                <sm::Array<sm::Struct<A>, sm::Size<1>>>::layout_recipe()
            ),
            RequirementsNotSatisfied,
            StructMismatch::FieldOffset { .. }
        ));
    }

    #[test]
    fn wgsl_uniform_must_be_sized() {
        let _guard = enable_color();

        #[derive(sm::GpuLayout)]
        struct A {
            a: sm::Array<f32x1>,
        }

        let e = TypeLayoutCompatibleWith::<Uniform>::try_from(Language::Wgsl, A::layout_recipe()).unwrap_err();
        if PRINT {
            println!("{e}");
        }
        assert!(matches!(
            e,
            AddressSpaceError::RequirementsNotSatisfied(RequirementsNotSatisfied::MustBeSized(
                _,
                Language::Wgsl,
                BufferAddressSpaceEnum::Uniform
            ))
        ));

        // Storage address space should allow unsized types
        assert!(TypeLayoutCompatibleWith::<Storage>::try_from(Language::Wgsl, A::layout_recipe()).is_ok());
    }

    #[test]
    fn wgsl_storage_may_not_contain_packed_vec() {
        let _guard = enable_color();

        #[derive(sm::GpuLayout)]
        #[gpu_repr(packed)]
        struct A {
            a: sm::packed::snorm16x2,
        }
        let e = TypeLayoutCompatibleWith::<Storage>::try_from(Language::Wgsl, A::layout_recipe()).unwrap_err();
        if PRINT {
            println!("{e}");
        }
        assert!(matches!(
            e,
            AddressSpaceError::NotRepresentable(NotRepresentable::MayNotContain(
                _,
                Language::Wgsl,
                BufferAddressSpaceEnum::Storage,
                RecipeContains::PackedVector
            ))
        ));
    }
}
