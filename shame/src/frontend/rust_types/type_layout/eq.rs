use super::*;

/// Error of two layouts mismatching. Implements Display for a visualization of the mismatch.
#[derive(Clone)]
pub struct LayoutMismatch {
    /// 2 (name, layout) pairs
    layouts: [(String, TypeLayout<constraint::Plain>); 2],
    colored_error: bool,
}


impl std::fmt::Debug for LayoutMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self) // use Display
    }
}

impl LayoutMismatch {
    fn pad_width(name_a: &str, name_b: &str) -> usize { name_a.chars().count().max(name_b.chars().count()) + SEP.len() }
}

impl Display for LayoutMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let colored = self.colored_error;
        let [(a_name, a), (b_name, b)] = &self.layouts;
        write!(f, "{:width$}", ' ', width = Self::pad_width(a_name, b_name))?;
        let layouts = [(a_name.as_str(), a), (b_name.as_str(), b)];
        match LayoutMismatch::write("", layouts, colored, f) {
            Err(MismatchWasFound) => Ok(()),
            Ok(KeepWriting) => {
                writeln!(
                    f,
                    "<internal error while writing layout mismatch: no difference found, please report this error>"
                )?;
                writeln!(f, "the full type layouts in question are:")?;
                for (name, layout) in layouts {
                    writeln!(f, "`{}`:", name)?;
                    writeln!(f, "{}", layout)?;
                }
                Ok(())
            }
        }
    }
}

/// layout mismatch diff name separator
/// used in the display impl of LayoutMismatch to show the actual place where the layouts mismatch
/// ```
/// layout mismatch:
/// cpu{SEP} f32
/// gpu{SEP} i32
/// ```
const SEP: &str = ": ";

/// whether the mismatching part of the TypeLayouts in a LayoutMismatch was already expressed via writes.
/// indicates that the `write` function should stop writing.
pub(crate) struct MismatchWasFound;
pub(crate) struct KeepWriting;

impl LayoutMismatch {
    //TODO(low prio) try to figure out a cleaner way of writing these.

    /// this function uses the `Err(MismatchWasFound)` to halt traversing the typelayout.
    /// It does not constitute an error of this function, it is just so the ? operator can be used to propagate the abort.
    #[allow(clippy::needless_return)]
    pub(crate) fn write<W: Write>(
        indent: &str,
        layouts: [(&str, &TypeLayout<constraint::Plain>); 2],
        colored: bool,
        f: &mut W,
    ) -> Result<KeepWriting, MismatchWasFound> {
        let tab = "  ";
        let [(a_name, a), (b_name, b)] = layouts;

        if a == b {
            a.write(indent, colored, f);
            return Ok(KeepWriting);
        }

        let use_256_color_mode = false;
        let hex_color = |f_: &mut W, hex| match colored {
            true => set_color(f_, Some(hex), use_256_color_mode),
            false => Ok(()),
        };
        let color_reset = |f_: &mut W| match colored {
            true => set_color(f_, None, use_256_color_mode),
            false => Ok(()),
        };

        let color_a_hex = "#DF5853";
        let color_b_hex = "#9A639C";
        let color_a = |f| hex_color(f, color_a_hex);
        let color_b = |f| hex_color(f, color_b_hex);

        let pad_width = Self::pad_width(a_name, b_name);

        use TypeLayoutSemantics as S;
        match (&a.kind, &b.kind) {
            (S::Structure(sa), S::Structure(sb)) => {
                let max_fields = sa.all_fields().len().max(sb.all_fields().len());
                {
                    write!(f, "struct ");
                    hex_color(f, color_a_hex);
                    write!(f, "{}", sa.name);
                    color_reset(f);
                    write!(f, " / ");
                    hex_color(f, color_b_hex);
                    write!(f, "{}", sb.name);
                    color_reset(f);
                    writeln!(f, " {{");
                }

                let mut sa_fields = sa.all_fields().iter();
                let mut sb_fields = sb.all_fields().iter();

                loop {
                    //TODO(low prio) get a hold of the code duplication here
                    match (sa_fields.next(), sb_fields.next()) {
                        (Some(a_field), Some(b_field)) => {
                            let offsets_match = a_field.rel_byte_offset == b_field.rel_byte_offset;
                            let types_match = a_field.field.ty == b_field.field.ty;
                            if !offsets_match && types_match {
                                // only write this mismatch if the types are also the same, otherwise display the detailed type mismatch further below
                                let a_ty_string = a_field.field.ty.first_line_of_display_with_ellipsis();
                                let b_ty_string = b_field.field.ty.first_line_of_display_with_ellipsis();
                                color_a(f);
                                writeln!(
                                    f,
                                    "{a_name}{SEP}{indent}{:3} {}: {a_ty_string} align={}",
                                    a_field.rel_byte_offset,
                                    a_field.field.name,
                                    a_field.field.byte_align().as_u32()
                                );
                                color_b(f);
                                writeln!(
                                    f,
                                    "{b_name}{SEP}{indent}{:3} {}: {b_ty_string} align={}",
                                    b_field.rel_byte_offset,
                                    b_field.field.name,
                                    b_field.field.byte_align().as_u32()
                                );
                                color_reset(f);
                                writeln!(
                                    f,
                                    "field offset is different on {a_name} ({}) and {b_name} ({}).",
                                    a_field.rel_byte_offset, b_field.rel_byte_offset
                                );
                                return Err(MismatchWasFound);
                            }
                            let offset = a_field.rel_byte_offset;
                            let a_field = &a_field.field;
                            let b_field = &b_field.field;
                            let field = &a_field;

                            if offsets_match {
                                write!(f, "{:width$}{indent}{offset:3} ", ' ', width = pad_width);
                            } else {
                                write!(f, "{:width$}{indent}  ? ", ' ', width = pad_width);
                            }
                            if a_field.name != b_field.name {
                                writeln!(f);
                                color_a(f);
                                writeln!(f, "{a_name}{SEP}{indent}    {}: …", a_field.name);
                                color_b(f);
                                writeln!(f, "{b_name}{SEP}{indent}    {}: …", b_field.name);
                                color_reset(f);
                                writeln!(
                                    f,
                                    "identifier mismatch, either\nfield '{}' is missing on {a_name}, or\nfield '{}' is missing on {b_name}.",
                                    b_field.name, a_field.name
                                );
                                return Err(MismatchWasFound);
                            }
                            write!(f, "{}: ", field.name);
                            if a_field.ty != b_field.ty {
                                Self::write(
                                    &format!("{indent}{tab}"),
                                    [(a_name, &a_field.ty), (b_name, &b_field.ty)],
                                    colored,
                                    f,
                                )?;
                                return Err(MismatchWasFound);
                            }
                            write!(f, "{}", field.ty.first_line_of_display_with_ellipsis());
                            if a_field.byte_size() != b_field.byte_size() {
                                writeln!(f);
                                color_a(f);
                                writeln!(
                                    f,
                                    "{a_name}{SEP}{indent} size={}",
                                    a_field
                                        .byte_size()
                                        .as_ref()
                                        .map(|x| x as &dyn Display)
                                        .unwrap_or(&"?" as _)
                                );
                                color_b(f);
                                writeln!(
                                    f,
                                    "{b_name}{SEP}{indent} size={}",
                                    b_field
                                        .byte_size()
                                        .as_ref()
                                        .map(|x| x as &dyn Display)
                                        .unwrap_or(&"?" as _)
                                );
                                color_reset(f);
                                return Err(MismatchWasFound);
                            }

                            write!(
                                f,
                                " size={}",
                                field
                                    .byte_size()
                                    .as_ref()
                                    .map(|x| x as &dyn Display)
                                    .unwrap_or(&"?" as _)
                            );
                            writeln!(f, ",");
                        }
                        (Some(a_field), None) => {
                            let offset = a_field.rel_byte_offset;
                            let a_field = &a_field.field;
                            let field = &a_field;
                            color_a(f);
                            write!(f, "{a_name}{SEP}{indent}{offset:3} ");
                            write!(f, "{}: ", field.name);
                            write!(f, "{}", field.ty.first_line_of_display_with_ellipsis());
                            write!(
                                f,
                                " size={}",
                                field
                                    .byte_size()
                                    .as_ref()
                                    .map(|x| x as &dyn Display)
                                    .unwrap_or(&"?" as _)
                            );
                            writeln!(f, ",");
                            color_b(f);
                            writeln!(f, "{b_name}{SEP}{indent}<missing field '{}'>", a_field.name);
                            color_reset(f);
                            return Err(MismatchWasFound);
                        }
                        (None, Some(b_field)) => {
                            let offset = b_field.rel_byte_offset;
                            let b_field = &b_field.field;
                            color_a(f);
                            writeln!(f, "{a_name}{SEP}{indent}<missing field '{}'>", b_field.name);
                            let field = &b_field;
                            color_b(f);
                            write!(f, "{b_name}{SEP}{indent}{offset:3} ");
                            write!(f, "{}: ", field.name);
                            write!(f, "{}", field.ty.first_line_of_display_with_ellipsis());
                            write!(
                                f,
                                " size={}",
                                field
                                    .byte_size()
                                    .as_ref()
                                    .map(|x| x as &dyn Display)
                                    .unwrap_or(&"?" as _)
                            );
                            writeln!(f, ",");
                            color_reset(f);
                            return Err(MismatchWasFound);
                        }
                        (None, None) => break,
                    }
                }

                write!(f, "{:width$}{indent}}}", ' ', width = pad_width);
                let align_matches = a.byte_align() == b.byte_align();
                let size_matches = a.byte_size() == b.byte_size();
                if !align_matches && size_matches {
                    writeln!(f);
                    color_a(f);
                    writeln!(f, "{a_name}{SEP}{indent}align={}", a.byte_align().as_u32());
                    color_b(f);
                    writeln!(f, "{b_name}{SEP}{indent}align={}", b.byte_align().as_u32());
                    color_reset(f);
                    return Err(MismatchWasFound);
                } else {
                    match align_matches {
                        true => write!(f, " align={}", a.byte_align().as_u32()),
                        false => write!(f, " align=?"),
                    };
                }
                if !size_matches {
                    writeln!(f);
                    color_a(f);
                    writeln!(
                        f,
                        "{a_name}{SEP}{indent}size={}",
                        a.byte_size().as_ref().map(|x| x as &dyn Display).unwrap_or(&"?" as _)
                    );
                    color_b(f);
                    writeln!(
                        f,
                        "{b_name}{SEP}{indent}size={}",
                        b.byte_size().as_ref().map(|x| x as &dyn Display).unwrap_or(&"?" as _)
                    );
                    color_reset(f);
                    return Err(MismatchWasFound);
                } else {
                    match a.byte_size() {
                        Some(size) => write!(f, " size={size}"),
                        None => write!(f, " size=?"),
                    };
                }
                // this should never happen, returning Ok(KeepWriting) will trigger the internal error in the Display impl
                return Ok(KeepWriting);
            }
            (S::Array(ta, na), S::Array(tb, nb)) => {
                if na != nb {
                    writeln!(f);
                    color_a(f);
                    write!(f, "{a_name}{SEP}");

                    write!(
                        f,
                        "array<…, {}>",
                        match na {
                            Some(n) => n as &dyn Display,
                            None => (&"runtime-sized") as &dyn Display,
                        }
                    );

                    //a.writeln(indent, colored, f);
                    writeln!(f);
                    color_b(f);
                    write!(f, "{b_name}{SEP}");

                    write!(
                        f,
                        "array<…, {}>",
                        match nb {
                            Some(n) => n as &dyn Display,
                            None => (&"runtime-sized") as &dyn Display,
                        }
                    );

                    //b.writeln(indent, colored, f);
                    color_reset(f);
                    Err(MismatchWasFound)
                } else {
                    write!(f, "array<");
                    //ta.ty.write(&(indent.to_string() + tab), colored, f);

                    Self::write(
                        &format!("{indent}{tab}"),
                        [(a_name, &ta.ty), (b_name, &tb.ty)],
                        colored,
                        f,
                    )?;

                    if let Some(na) = na {
                        write!(f, ", {na}");
                    }
                    write!(f, ">");

                    if ta.byte_stride != tb.byte_stride {
                        writeln!(f);
                        color_a(f);
                        writeln!(f, "{a_name}{SEP}{indent}>  stride={}", ta.byte_stride);
                        color_b(f);
                        writeln!(f, "{b_name}{SEP}{indent}>  stride={}", tb.byte_stride);
                        color_reset(f);
                        Err(MismatchWasFound)
                    } else {
                        // this should never happen, returning Ok(KeepWriting) will trigger the internal error in the Display impl
                        write!(f, ">  stride={}", ta.byte_stride);
                        return Ok(KeepWriting);
                    }
                }
            }
            (S::Vector(_), S::Vector(_)) => {
                writeln!(f);
                color_a(f);
                write!(f, "{a_name}{SEP}");
                a.writeln(indent, colored, f);
                color_b(f);
                write!(f, "{b_name}{SEP}");
                b.writeln(indent, colored, f);
                color_reset(f);
                Err(MismatchWasFound)
            }
            (S::Matrix(_), S::Matrix(_)) => {
                writeln!(f);
                color_a(f);
                write!(f, "{a_name}{SEP}");
                a.writeln(indent, colored, f);
                color_b(f);
                write!(f, "{b_name}{SEP}");
                b.writeln(indent, colored, f);
                color_reset(f);
                Err(MismatchWasFound)
            }
            (S::PackedVector(p), S::PackedVector(p1)) => {
                writeln!(f);
                color_a(f);
                write!(f, "{a_name}{SEP}");
                a.writeln(indent, colored, f);
                color_b(f);
                write!(f, "{b_name}{SEP}");
                b.writeln(indent, colored, f);
                color_reset(f);
                Err(MismatchWasFound)
            }
            (
                // its written like this so that exhaustiveness checks lead us to this match statement if a type is added
                S::Structure { .. } | S::Array { .. } | S::Vector { .. } | S::Matrix { .. } | S::PackedVector { .. },
                _,
            ) => {
                // TypeLayoutSemantics mismatch
                writeln!(f);
                color_a(f);
                write!(f, "{a_name}{SEP}");
                a.writeln(indent, colored, f);
                color_b(f);
                write!(f, "{b_name}{SEP}");
                b.writeln(indent, colored, f);
                color_reset(f);
                Err(MismatchWasFound)
            }
        }
    }
}

/// takes two pairs of `(debug_name, layout)` and compares them for equality.
///
/// if the two layouts are not equal it uses the debug names in the returned
/// error to tell the two layouts apart.
pub(crate) fn check_eq<L: TypeConstraint, R: TypeConstraint>(
    a: (&str, &TypeLayout<L>),
    b: (&str, &TypeLayout<R>),
) -> Result<(), LayoutMismatch>
where
    TypeLayout<L>: PartialEq<TypeLayout<R>>,
{
    match a.1 == b.1 {
        true => Ok(()),
        false => Err(LayoutMismatch {
            layouts: [
                (a.0.into(), a.1.to_owned().into_plain()),
                (b.0.into(), b.1.to_owned().into_plain()),
            ],
            colored_error: Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages)
                .unwrap_or(false),
        }),
    }
}
