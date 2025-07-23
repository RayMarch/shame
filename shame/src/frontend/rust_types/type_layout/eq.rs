use crate::common::prettify::UnwrapOrStr;
use crate::frontend::rust_types::type_layout::compatible_with::{StructMismatch, TopLevelMismatch};
use crate::frontend::rust_types::type_layout::display::LayoutInfo;

use super::compatible_with::try_find_mismatch;
use super::*;

/// Error of two layouts mismatching. Implements Display for a visualization of the mismatch.
#[derive(Clone)]
pub struct LayoutMismatch {
    /// 2 (name, layout) pairs
    layouts: [(String, TypeLayout); 2],
    colored_error: bool,
}


impl std::fmt::Debug for LayoutMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self}") }
}

impl Display for LayoutMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [(a_name, a), (b_name, b)] = &self.layouts;
        let layouts = [(a_name.as_str(), a), (b_name.as_str(), b)];
        match LayoutMismatch::write(f, layouts, self.colored_error)? {
            Mismatch::Found => {}
            Mismatch::NotFound => {
                writeln!(
                    f,
                    "<internal error while writing layout mismatch: no difference found, please report this error>"
                )?;
                writeln!(f, "the full type layouts in question are:")?;
                for (name, layout) in layouts {
                    writeln!(f, "`{name}`:")?;
                    writeln!(f, "{layout}")?;
                }
            }
        }

        Ok(())
    }
}

/// Whether the mismatch was found or not.
pub(crate) enum Mismatch {
    Found,
    NotFound,
}

impl LayoutMismatch {
    #[allow(clippy::needless_return)]
    pub(crate) fn write<W: Write>(
        f: &mut W,
        layouts: [(&str, &TypeLayout); 2],
        colored: bool,
    ) -> Result<Mismatch, std::fmt::Error> {
        let [(a_name, a), (b_name, b)] = layouts;

        let use_256_color_mode = false;
        let enable_color = |f_: &mut W, hex| match colored {
            true => set_color(f_, Some(hex), use_256_color_mode),
            false => Ok(()),
        };
        let reset_color = |f_: &mut W| match colored {
            true => set_color(f_, None, use_256_color_mode),
            false => Ok(()),
        };
        let hex_left = "#DF5853"; // red
        let hex_right = "#9A639C"; // purple

        // Using try_find_mismatch to find the first mismatching type / struct field.
        let Some(mismatch) = try_find_mismatch(a, b) else {
            return Ok(Mismatch::NotFound);
        };

        use compatible_with::LayoutMismatch::{TopLevel, Struct};
        match mismatch {
            TopLevel {
                layout_left,
                layout_right,
                mismatch,
            } => match mismatch {
                TopLevelMismatch::Type => writeln!(
                    f,
                    "The layouts of `{}` and `{}` do not match, because their types are semantically different.",
                    layout_left.short_name(),
                    layout_right.short_name()
                )?,
                TopLevelMismatch::ArrayStride {
                    array_left,
                    array_right,
                } => {
                    writeln!(
                        f,
                        "The layouts of `{}` and `{}` do not match.",
                        layout_left.short_name(),
                        layout_right.short_name()
                    )?;
                    writeln!(
                        f,
                        "`{}` has a stride of {}, while `{}` has a stride of {}.",
                        array_left.short_name(),
                        array_left.byte_stride,
                        array_right.short_name(),
                        array_right.byte_stride
                    )?;
                }
                TopLevelMismatch::ByteSize { left, right } => {
                    writeln!(
                        f,
                        "The layouts of `{}` and `{}` do not match.",
                        layout_left.short_name(),
                        layout_right.short_name()
                    )?;
                    writeln!(
                        f,
                        "`{}` has a byte size of {}, while `{}` has a byte size of {}.",
                        left.short_name(),
                        UnwrapOrStr(left.byte_size(), "runtime-sized"),
                        right.short_name(),
                        UnwrapOrStr(right.byte_size(), "runtime-sized")
                    )?;
                }
            },
            Struct {
                struct_left,
                struct_right,
                mismatch,
            } => {
                let field_name = |field_index: usize| {
                    if let Some(name) = struct_left.fields.get(field_index).map(|f| &f.name) {
                        format!("field `{name}`")
                    } else {
                        // Should never happen, but just in case we fall back to this
                        format!("field {field_index}")
                    }
                };

                writeln!(
                    f,
                    "The layouts of `{}` and `{}` do not match, because the",
                    struct_left.name, struct_right.name
                );
                let (mismatch_field_index, layout_info) = match &mismatch {
                    StructMismatch::FieldName { field_index } => {
                        writeln!(f, "names of field {field_index} are different.")?;
                        (Some(field_index), LayoutInfo::NONE)
                    }
                    StructMismatch::FieldLayout {
                        field_index,
                        mismatch: TopLevelMismatch::Type,
                    } => {
                        writeln!(f, "type of {} is different.", field_name(*field_index))?;
                        (Some(field_index), LayoutInfo::NONE)
                    }
                    // TODO(chronicl) fix byte size message for when the byte size mismatch is happening
                    // in a nested array
                    StructMismatch::FieldLayout {
                        field_index,
                        mismatch: TopLevelMismatch::ByteSize { left, right },
                    } => {
                        match struct_left.fields.get(*field_index) {
                            // Inner type in (nested) array has mismatching byte size
                            Some(field) if &field.ty != left => {
                                writeln!(
                                    f,
                                    "byte size of `{}` is {} in `{}` and {} in `{}`.",
                                    left.short_name(),
                                    UnwrapOrStr(left.byte_size(), "runtime-sized"),
                                    struct_left.name,
                                    UnwrapOrStr(right.byte_size(), "runtime-sized"),
                                    struct_right.name,
                                )?;
                                // Not showing byte size info, because it can be misleading since
                                // the inner type is the one that has mismatching byte size.
                                (Some(field_index), LayoutInfo::NONE)
                            }
                            _ => {
                                writeln!(f, "byte size of {} is different.", field_name(*field_index))?;
                                (Some(field_index), LayoutInfo::SIZE)
                            }
                        }
                    }
                    StructMismatch::FieldLayout {
                        field_index,
                        mismatch:
                            TopLevelMismatch::ArrayStride {
                                array_left,
                                array_right,
                            },
                    } => {
                        match struct_left.fields.get(*field_index) {
                            // Inner type in (nested) array has mismatching stride
                            Some(field) if field.ty.short_name() != array_left.short_name() => {
                                writeln!(
                                    f,
                                    "stride of `{}` is {} in `{}` and {} in `{}`.",
                                    array_left.short_name(),
                                    array_left.byte_stride,
                                    struct_left.name,
                                    array_right.byte_stride,
                                    struct_right.name,
                                )?;
                                // Not showing stride info, because it can be misleading since
                                // the inner type is the one that has mismatching stride.
                                (Some(field_index), LayoutInfo::NONE)
                            }
                            _ => {
                                writeln!(f, "array stride of {} is different.", field_name(*field_index))?;
                                (Some(field_index), LayoutInfo::STRIDE)
                            }
                        }
                    }
                    StructMismatch::FieldOffset { field_index } => {
                        writeln!(f, "offset of {} is different.", field_name(*field_index))?;
                        (
                            Some(field_index),
                            LayoutInfo::OFFSET | LayoutInfo::ALIGN | LayoutInfo::SIZE,
                        )
                    }
                    StructMismatch::FieldCount => {
                        writeln!(f, "number of fields is different.")?;
                        (None, LayoutInfo::NONE)
                    }
                };
                writeln!(f)?;

                let fields_without_mismatch = match mismatch_field_index {
                    Some(index) => struct_left.fields.len().min(struct_right.fields.len()).min(*index),
                    None => struct_left.fields.len().min(struct_right.fields.len()),
                };

                // Start writing the structs with the mismatch highlighted
                let mut writer_left = struct_left.writer(layout_info);
                let mut writer_right = struct_right.writer(layout_info);

                // Make sure layout info offset only takes account the fields before and including the mismatch,
                // because those are the only fields that will be written below.
                if let Some(mismatch_field_index) = mismatch_field_index {
                    let max_fields = Some(mismatch_field_index + 1);
                    writer_left.set_layout_info_offset_auto(max_fields);
                    writer_right.set_layout_info_offset_auto(max_fields);
                }
                // Make sure layout info offset is large enough to fit the custom struct declaration
                let struct_declaration = format!("struct {a_name} / {b_name} {{");
                let layout_info_offset = writer_left
                    .layout_info_offset()
                    .max(writer_right.layout_info_offset())
                    .max(struct_declaration.len());
                writer_left.ensure_layout_info_offset(layout_info_offset);
                writer_right.ensure_layout_info_offset(layout_info_offset);

                // Write header
                writer_left.writeln_header(f)?;

                // Write custom struct declaration
                write!(f, "struct ")?;
                enable_color(f, hex_left)?;
                write!(f, "{}", struct_left.name)?;
                reset_color(f)?;
                write!(f, " / ")?;
                enable_color(f, hex_right)?;
                write!(f, "{}", struct_right.name)?;
                reset_color(f)?;
                writeln!(f, " {{")?;

                // Write matching fields
                for field_index in 0..fields_without_mismatch {
                    writer_left.writeln_field(f, field_index)?;
                }

                match mismatch {
                    StructMismatch::FieldName { field_index } |
                    StructMismatch::FieldLayout { field_index, .. } |
                    StructMismatch::FieldOffset { field_index } => {
                        // Write mismatching field
                        enable_color(f, hex_left)?;
                        writer_left.write_field(f, field_index)?;
                        writeln!(f, "  <-- {a_name}")?;
                        reset_color(f)?;
                        enable_color(f, hex_right)?;
                        writer_right.write_field(f, field_index)?;
                        writeln!(f, "  <-- {b_name}")?;
                        reset_color(f)?;
                        if struct_left.fields.len() > field_index + 1 || struct_right.fields.len() > field_index + 1 {
                            // Write ellipsis if there are more fields after the mismatch
                            writeln!(f, "{}...", writer_left.tab())?;
                        }
                    }
                    StructMismatch::FieldCount => {
                        // Write the remaining fields of the larger struct
                        let (writer, len, hex) = match struct_left.fields.len() > struct_right.fields.len() {
                            true => (&mut writer_left, struct_left.fields.len(), hex_left),
                            false => (&mut writer_right, struct_right.fields.len(), hex_right),
                        };

                        enable_color(f, hex)?;
                        for field_index in fields_without_mismatch..len {
                            writer.writeln_field(f, field_index)?;
                        }
                        reset_color(f)?;
                    }
                }

                // Write closing bracket
                writer_left.writeln_struct_end(f)?;
            }
        }

        Ok(Mismatch::Found)
    }
}

/// takes two pairs of `(debug_name, layout)` and compares them for equality.
///
/// if the two layouts are not equal it uses the debug names in the returned
/// error to tell the two layouts apart.
pub(crate) fn check_eq(a: (&str, &TypeLayout), b: (&str, &TypeLayout)) -> Result<(), LayoutMismatch>
where
    TypeLayout: PartialEq<TypeLayout>,
{
    match a.1 == b.1 {
        true => Ok(()),
        false => Err(LayoutMismatch {
            layouts: [(a.0.into(), a.1.to_owned()), (b.0.into(), b.1.to_owned())],
            colored_error: Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages)
                .unwrap_or(false),
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate as shame;
    use shame as sm;
    use shame::{CpuLayout, GpuLayout, gpu_layout, cpu_layout};
    use crate::aliases::*;

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct f32x3_align4(pub [f32; 4]);

    impl CpuLayout for f32x3_align4 {
        fn cpu_layout() -> shame::TypeLayout {
            let mut layout = gpu_layout::<f32x3>();
            layout.set_align(shame::any::U32PowerOf2::_4);
            layout
        }
    }

    fn print_mismatch<T: GpuLayout, TCpu: CpuLayout>() {
        println!(
            "{}",
            super::check_eq(("gpu", &gpu_layout::<T>()), ("cpu", &cpu_layout::<TCpu>())).unwrap_err()
        );
    }

    #[test]
    fn test_layout_mismatch_nested() {
        // For colored output
        let mut encoder = sm::start_encoding(sm::Settings::default()).unwrap();
        let _ = encoder.new_render_pipeline(sm::Indexing::BufferU16);

        // field name mismatch
        #[derive(GpuLayout)]
        pub struct A {
            a: u32x1,
        }
        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct ACpu {
            b: u32,
        }
        print_mismatch::<A, ACpu>();

        // field type mismatch
        #[derive(GpuLayout)]
        pub struct B {
            a: f32x1,
        }
        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct BCpu {
            a: u32,
        }
        print_mismatch::<B, BCpu>();

        // field offset mismatch
        #[derive(GpuLayout)]
        pub struct C {
            a: f32x1,
            b: f32x3,
        }
        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct CCpu {
            a: f32,
            b: f32x3_align4,
        }
        print_mismatch::<C, CCpu>();
    }
}
