use crate::frontend::rust_types::type_layout::compatible_with::TopLevelMismatch;

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
        let tab = "  ";
        // TODO(chronicl) include names somehow
        let [(a_name, a), (b_name, b)] = layouts;

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
            } => {
                match mismatch {
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
                        let array_type_layout: TypeLayout = array_left.clone().into();
                        writeln!(
                            f,
                            "The layouts of `{}` and `{}` do not match.",
                            layout_left.short_name(),
                            layout_right.short_name()
                        )?;
                        // Using array_left and array_right here, because those are the layouts
                        // of the deepest array stride mismatch. For example it can be that
                        // layout_left = Array<Array<f32x1>>
                        // array_left = Array<f32x1>
                        // because the deepest array stride mismatch is happening on the inner Array<f32x1>.
                        writeln!(
                            f,
                            "`{}` has a stride of {}, while `{}` has a stride of {}.",
                            array_left.short_name(),
                            array_left.byte_stride,
                            array_right.short_name(),
                            array_right.byte_stride
                        )?;
                    }
                }
            }
            Struct {
                struct_left,
                struct_right,
                mismatch,
            } => {
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

                let mut writer_left = struct_left.writer(true);
                let mut writer_right = struct_right.writer(true);
                // Making sure that both writer have the same layout info offset, the max of both.
                writer_left.ensure_layout_info_offset(writer_right.layout_info_offset());
                writer_right.ensure_layout_info_offset(writer_left.layout_info_offset());

                writer_left.writeln_header(f)?;
                enable_color(f, hex_left)?;
                writer_left.writeln_struct_declaration(f)?;
                reset_color(f)?;
                enable_color(f, hex_right)?;
                writer_right.writeln_struct_declaration(f)?;
                reset_color(f)?;
                for field_index in 0..struct_left.fields.len() {
                    let field_left = &struct_left.fields[field_index];
                    let field_right = &struct_right.fields[field_index];

                    // We are checking every field for equality, instead of using the singular found mismatch from try_find_mismatch
                    if field_left.ty == field_right.ty && field_left.rel_byte_offset == field_right.rel_byte_offset {
                        writer_left.writeln_field(f, field_index)?;
                    } else {
                        enable_color(f, hex_left)?;
                        writer_left.writeln_field(f, field_index)?;
                        reset_color(f)?;
                        enable_color(f, hex_right)?;
                        writer_right.writeln_field(f, field_index)?;
                        reset_color(f)?;
                    }
                }
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

#[test]
fn test_layout_mismatch() {
    use crate as shame;
    use shame as sm;
    use shame::CpuLayout;
    use crate::aliases::*;

    #[derive(shame::GpuLayout)]
    #[cpu(ACpu)]
    pub struct A {
        a: u32x1,
        c: sm::Array<f32x3, sm::Size<2>>,
        b: f32x1,
    }

    #[derive(shame::CpuLayout)]
    #[repr(C)]
    pub struct ACpu {
        d: u32,
        c: [[f32; 3]; 2],
        b: u32,
    }

    let mut encoder = sm::start_encoding(sm::Settings::default()).unwrap();
    let mut drawcall = encoder.new_render_pipeline(sm::Indexing::BufferU16);
    let mut group0 = drawcall.bind_groups.next();
    let a: sm::Buffer<A, sm::mem::Storage> = group0.next();

    let primitive = drawcall
        .vertices
        .assemble(f32x3::zero(), sm::Draw::triangle_list(sm::Winding::Ccw));
    let frag = primitive.rasterize(sm::Accuracy::Relaxed);
    frag.fill(f32x3::zero());
    encoder.finish().unwrap();
}


#[test]
fn test_layout_mismatch_nested() {
    use crate as shame;
    use shame as sm;
    use shame::CpuLayout;
    use crate::aliases::*;

    #[derive(shame::GpuLayout)]
    #[cpu(ACpu)]
    pub struct A {
        a: u32x1,
        c: sm::Array<f32x3, sm::Size<2>>,
        b: f32x1,
    }

    #[derive(shame::CpuLayout)]
    #[repr(C)]
    pub struct ACpu {
        d: u32,
        c: [[f32; 3]; 2],
        b: u32,
    }

    let mut encoder = sm::start_encoding(sm::Settings::default()).unwrap();
    let mut drawcall = encoder.new_render_pipeline(sm::Indexing::BufferU16);
    let mut group0 = drawcall.bind_groups.next();
    let a: sm::Buffer<A, sm::mem::Storage> = group0.next();

    let primitive = drawcall
        .vertices
        .assemble(f32x3::zero(), sm::Draw::triangle_list(sm::Winding::Ccw));
    let frag = primitive.rasterize(sm::Accuracy::Relaxed);
    frag.fill(f32x3::zero());
    encoder.finish().unwrap();
}
