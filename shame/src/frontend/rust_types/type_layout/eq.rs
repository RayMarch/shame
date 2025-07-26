use crate::common::prettify::UnwrapOrStr;
use crate::frontend::rust_types::type_layout::display::LayoutInfo;

use super::*;

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
/// mismatches, because `B` and `BOther` don't match, then the exact mismatch (some field mismatch)
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
/// `LayoutMismatch::Struct` containing the field index and `FieldLayout`, which let's us access the outer
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
        field_left: FieldLayout,
        field_right: FieldLayout,
    },
    FieldLayout {
        field_index: usize,
        field_left: FieldLayout,
        field_right: FieldLayout,
        mismatch: TopLevelMismatch,
    },
    FieldOffset {
        field_index: usize,
        field_left: FieldLayout,
        field_right: FieldLayout,
    },
    FieldCount,
}

/// Find the first depth first layout mismatch
pub(crate) fn try_find_mismatch(layout1: &TypeLayout, layout2: &TypeLayout) -> Option<LayoutMismatch> {
    use TypeLayout::*;

    let make_mismatch = |mismatch: TopLevelMismatch| LayoutMismatch::TopLevel {
        layout_left: layout1.clone(),
        layout_right: layout2.clone(),
        mismatch,
    };

    // First check if the kinds are the same type
    match (&layout1, &layout2) {
        (Vector(v1), Vector(v2)) => {
            if v1.ty != v2.ty {
                return Some(make_mismatch(TopLevelMismatch::Type));
            }
        }
        (PackedVector(p1), PackedVector(p2)) => {
            if p1.ty != p2.ty {
                return Some(make_mismatch(TopLevelMismatch::Type));
            }
        }
        (Matrix(m1), Matrix(m2)) => {
            if m1.ty != m2.ty {
                return Some(make_mismatch(TopLevelMismatch::Type));
            }
        }
        (Array(a1), Array(a2)) => {
            // Recursively check element types
            match try_find_mismatch(&a1.element_ty, &a2.element_ty) {
                // Update the top level layouts and propagate the LayoutMismatch
                Some(LayoutMismatch::TopLevel { mismatch, .. }) => {
                    return Some(make_mismatch(mismatch));
                }
                // Struct mismatch, so it's not a top-level mismatch
                m @ Some(LayoutMismatch::Struct { .. }) => return m,
                None => {}
            }

            // Check array lengths, which are a type mismatch if they differ
            if a1.len != a2.len {
                return Some(make_mismatch(TopLevelMismatch::Type));
            }

            // Check array stride
            if a1.byte_stride != a2.byte_stride {
                return Some(make_mismatch(TopLevelMismatch::ArrayStride {
                    array_left: (**a1).clone(),
                    array_right: (**a2).clone(),
                }));
            }
        }
        (Struct(s1), Struct(s2)) => {
            return try_find_struct_mismatch(s1, s2);
        }
        // Different kinds entirely. Matching exhaustively, so that changes to TypeLayout lead us here.
        (Vector(_) | PackedVector(_) | Matrix(_) | Array(_) | Struct(_), _) => {
            return Some(make_mismatch(TopLevelMismatch::Type));
        }
    }

    // Check byte size.
    // We do this at the end, because type mismatches should have priority over byte size mismatches.
    if layout1.byte_size() != layout2.byte_size() {
        return Some(make_mismatch(TopLevelMismatch::ByteSize {
            left: layout1.clone(),
            right: layout2.clone(),
        }));
    }

    None
}

fn try_find_struct_mismatch(struct1: &StructLayout, struct2: &StructLayout) -> Option<LayoutMismatch> {
    let make_mismatch = |mismatch: StructMismatch| LayoutMismatch::Struct {
        struct_left: struct1.clone(),
        struct_right: struct2.clone(),
        mismatch,
    };

    for (field_index, (field1, field2)) in struct1.fields.iter().zip(struct2.fields.iter()).enumerate() {
        // Order of checks is important here. We check in order
        // - field name
        // - field inner mismatch
        // - field offset
        if field1.name != field2.name {
            return Some(make_mismatch(StructMismatch::FieldName {
                field_index,
                field_left: field1.clone(),
                field_right: field2.clone(),
            }));
        }

        // Recursively check field types
        if let Some(inner_mismatch) = try_find_mismatch(&field1.ty, &field2.ty) {
            match inner_mismatch {
                // If it's a top-level mismatch, convert it to a field mismatch
                LayoutMismatch::TopLevel { mismatch, .. } => {
                    return Some(make_mismatch(StructMismatch::FieldLayout {
                        field_index,
                        field_left: field1.clone(),
                        field_right: field2.clone(),
                        mismatch,
                    }));
                }
                // Pass through struct mismatches
                struct_mismatch @ LayoutMismatch::Struct { .. } => return Some(struct_mismatch),
            }
        }

        // Check field offset
        if field1.rel_byte_offset != field2.rel_byte_offset {
            return Some(make_mismatch(StructMismatch::FieldOffset {
                field_index,
                field_left: field1.clone(),
                field_right: field2.clone(),
            }));
        }
    }

    // Check field count.
    // We do this at the end, because fields are checked in order and a field count mismatch
    // can be viewed as a field mismatch of one field beyond the last field of the smaller struct.
    if struct1.fields.len() != struct2.fields.len() {
        return Some(make_mismatch(StructMismatch::FieldCount));
    }

    None
}

/// Error of two layouts mismatching. Implements Display for a visualization of the mismatch.
#[derive(Clone)]
pub struct CheckEqLayoutMismatch {
    /// 2 (name, layout) pairs
    layouts: [(String, TypeLayout); 2],
    colored_error: bool,
}

impl std::fmt::Debug for CheckEqLayoutMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self}") }
}

impl Display for CheckEqLayoutMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [(a_name, a), (b_name, b)] = &self.layouts;
        let layouts = [(a_name.as_str(), a), (b_name.as_str(), b)];
        match CheckEqLayoutMismatch::write(f, layouts, self.colored_error) {
            Ok(()) => {}
            Err(DisplayMismatchError::FmtError(e)) => return Err(e),
            Err(DisplayMismatchError::NotFound) => {
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

pub(crate) enum DisplayMismatchError {
    NotFound,
    FmtError(std::fmt::Error),
}
impl From<std::fmt::Error> for DisplayMismatchError {
    fn from(err: std::fmt::Error) -> Self { DisplayMismatchError::FmtError(err) }
}

impl CheckEqLayoutMismatch {
    #[allow(clippy::needless_return)]
    pub(crate) fn write<W: Write>(
        f: &mut W,
        layouts: [(&str, &TypeLayout); 2],
        colored: bool,
    ) -> Result<(), DisplayMismatchError> {
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
            return Err(DisplayMismatchError::NotFound);
        };

        match mismatch {
            LayoutMismatch::TopLevel {
                layout_left,
                layout_right,
                mismatch,
            } => match mismatch {
                TopLevelMismatch::Type => writeln!(
                    f,
                    "The layouts of `{}` ({a_name}) and `{}` ({b_name}) do not match, because their types are semantically different.",
                    layout_left.short_name(),
                    layout_right.short_name()
                )?,
                TopLevelMismatch::ArrayStride {
                    array_left,
                    array_right,
                } => {
                    writeln!(
                        f,
                        "The layouts of `{}` ({a_name}) and `{}` ({b_name}) do not match.",
                        layout_left.short_name(),
                        layout_right.short_name()
                    )?;
                    writeln!(
                        f,
                        "`{}` ({a_name}) has a stride of {}, while `{}` ({b_name}) has a stride of {}.",
                        array_left.short_name(),
                        array_left.byte_stride,
                        array_right.short_name(),
                        array_right.byte_stride
                    )?;
                }
                TopLevelMismatch::ByteSize { left, right } => {
                    writeln!(
                        f,
                        "The layouts of `{}` ({a_name}) and `{}` ({b_name}) do not match.",
                        layout_left.short_name(),
                        layout_right.short_name()
                    )?;
                    writeln!(
                        f,
                        "`{}` ({a_name}) has a byte size of {}, while `{}` ({b_name}) has a byte size of {}.",
                        left.short_name(),
                        UnwrapOrStr(left.byte_size(), "runtime-sized"),
                        right.short_name(),
                        UnwrapOrStr(right.byte_size(), "runtime-sized")
                    )?;
                }
            },
            LayoutMismatch::Struct {
                struct_left,
                struct_right,
                mismatch,
            } => {
                let is_top_level = &TypeLayout::Struct(Rc::new(struct_left.clone())) == a;
                if is_top_level {
                    writeln!(
                        f,
                        "The layouts of `{}` and `{}` do not match, because the",
                        struct_left.name, struct_right.name
                    )?;
                } else {
                    writeln!(
                        f,
                        "The layouts of `{}` and `{}`, contained in `{}` and `{}` respectively, do not match, because the",
                        struct_left.name,
                        struct_right.name,
                        a.short_name(),
                        b.short_name()
                    )?;
                }
                let (mismatch_field_index, layout_info) = match &mismatch {
                    StructMismatch::FieldName { field_index, .. } => {
                        writeln!(f, "names of field {field_index} are different.")?;
                        (Some(field_index), LayoutInfo::NONE)
                    }
                    StructMismatch::FieldLayout {
                        field_index,
                        field_left,
                        mismatch: TopLevelMismatch::Type,
                        ..
                    } => {
                        writeln!(f, "type of `{}` is different.", field_left.name)?;
                        (Some(field_index), LayoutInfo::NONE)
                    }
                    StructMismatch::FieldLayout {
                        field_index,
                        field_left,
                        mismatch: TopLevelMismatch::ByteSize { left, right },
                        ..
                    } => {
                        // Inner type in (nested) array has mismatching byte size
                        if &field_left.ty != left {
                            writeln!(
                                f,
                                "byte size of `{}` is {} in `{}` and the byte size of `{}` is {} in `{}`.",
                                left.short_name(),
                                UnwrapOrStr(left.byte_size(), "runtime-sized"),
                                struct_left.name,
                                right.short_name(),
                                UnwrapOrStr(right.byte_size(), "runtime-sized"),
                                struct_right.name,
                            )?;
                            // Not showing byte size info, because it can be misleading since
                            // the inner type is the one that has mismatching byte size.
                            (Some(field_index), LayoutInfo::NONE)
                        } else {
                            writeln!(f, "byte size of `{}` is different.", field_left.name)?;
                            (Some(field_index), LayoutInfo::SIZE)
                        }
                    }
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
                        // Inner type in (nested) array has mismatching stride
                        if field_left.ty.short_name() != array_left.short_name() {
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
                        } else {
                            writeln!(f, "array stride of {} is different.", field_left.name)?;
                            (Some(field_index), LayoutInfo::STRIDE)
                        }
                    }
                    StructMismatch::FieldOffset {
                        field_index,
                        field_left,
                        ..
                    } => {
                        writeln!(f, "offset of {} is different.", field_left.name)?;
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
                    StructMismatch::FieldName { field_index, .. } |
                    StructMismatch::FieldLayout { field_index, .. } |
                    StructMismatch::FieldOffset { field_index, .. } => {
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

        Ok(())
    }
}

/// takes two pairs of `(debug_name, layout)` and compares them for equality.
///
/// if the two layouts are not equal it uses the debug names in the returned
/// error to tell the two layouts apart.
pub(crate) fn check_eq(a: (&str, &TypeLayout), b: (&str, &TypeLayout)) -> Result<(), CheckEqLayoutMismatch>
where
    TypeLayout: PartialEq<TypeLayout>,
{
    match a.1 == b.1 {
        true => Ok(()),
        false => Err(CheckEqLayoutMismatch {
            layouts: [(a.0.into(), a.1.to_owned()), (b.0.into(), b.1.to_owned())],
            colored_error: Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages)
                .unwrap_or(false),
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::pipeline_kind::Render;
    use crate::{self as shame, EncodingGuard, ThreadIsAlreadyEncoding};
    use shame as sm;
    use shame::{CpuLayout, GpuLayout, gpu_layout, cpu_layout};
    use crate::aliases::*;

    const PRINT: bool = true;

    #[derive(Clone, Copy)]
    #[repr(C)]
    #[allow(non_camel_case_types)]
    struct f32x3_align4(pub [f32; 3]);

    impl CpuLayout for f32x3_align4 {
        fn cpu_layout() -> shame::TypeLayout {
            let mut layout = gpu_layout::<f32x3>();
            layout.set_align(shame::any::U32PowerOf2::_4);
            layout
        }
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    #[allow(non_camel_case_types)]
    struct f32x3_size16(pub [f32; 4]);

    impl CpuLayout for f32x3_size16 {
        fn cpu_layout() -> shame::TypeLayout {
            let mut layout = gpu_layout::<f32x3>();
            layout.set_byte_size(Some(16));
            layout
        }
    }

    fn check_mismatch<T: GpuLayout, TCpu: CpuLayout>() {
        let mismatch = super::check_eq(("gpu", &gpu_layout::<T>()), ("cpu", &cpu_layout::<TCpu>())).unwrap_err();
        if PRINT {
            println!("{mismatch}");
        }
    }

    fn enable_color() -> Option<Result<EncodingGuard<Render>, ThreadIsAlreadyEncoding>> {
        PRINT.then(|| sm::start_encoding(sm::Settings::default()))
    }

    #[test]
    fn test_field_name_mismatch() {
        let _guard = enable_color();

        #[derive(GpuLayout)]
        pub struct A {
            a: u32x1,
        }
        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct ACpu {
            b: u32,
        }
        check_mismatch::<A, ACpu>();
    }

    #[test]
    fn test_field_type_mismatch() {
        let _guard = enable_color();

        if PRINT {
            println!("The error also shows how \"...\" is used if there are more fields after the mismatching field\n");
        }
        #[derive(GpuLayout)]
        pub struct B {
            a: f32x1,
            b: f32x1,
        }
        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct BCpu {
            a: u32,
            b: f32,
        }
        check_mismatch::<B, BCpu>();
    }

    #[test]
    fn test_field_offset_mismatch() {
        let _guard = enable_color();

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
        check_mismatch::<C, CCpu>();
    }

    #[test]
    fn test_field_byte_size_mismatch() {
        let _guard = enable_color();

        #[derive(GpuLayout)]
        pub struct D {
            a: f32x3,
        }
        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct DCpu {
            a: f32x3_size16,
        }
        check_mismatch::<D, DCpu>();
    }

    #[test]
    fn test_field_nested_byte_size_mismatch() {
        let _guard = enable_color();

        if PRINT {
            println!(
                "The error does not show the `size` column, because it could be confusing, since the type in the array is where the mismatch happens:\n"
            );
        }
        #[derive(GpuLayout)]
        pub struct E {
            a: sm::Array<f32x3, sm::Size<4>>,
        }
        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct ECpu {
            a: [f32x3_size16; 4],
        }
        check_mismatch::<E, ECpu>();
    }

    #[test]
    fn test_field_stride_mismatch() {
        let _guard = enable_color();

        #[derive(GpuLayout)]
        pub struct F {
            a: sm::Array<f32x3, sm::Size<4>>,
        }
        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct FCpu {
            a: [f32x3_align4; 4],
        }
        check_mismatch::<F, FCpu>();
    }

    #[test]
    fn test_stride_mismatch() {
        let _guard = enable_color();
        check_mismatch::<sm::Array<f32x3, sm::Size<4>>, [f32x3_align4; 4]>();
    }

    #[test]
    fn test_nested_stride_mismatch() {
        let _guard = enable_color();
        check_mismatch::<sm::Array<sm::Array<f32x3, sm::Size<4>>, sm::Size<2>>, [[f32x3_align4; 4]; 2]>();
    }

    #[test]
    fn test_nested_stride_in_struct_mismatch() {
        let _guard = enable_color();

        if PRINT {
            println!(
                "The error does not show the `stride` column, because it could be confusing, since the type in the array is where the mismatch happens:\n"
            );
        }
        #[derive(GpuLayout)]
        pub struct G {
            a: sm::Array<sm::Array<f32x3, sm::Size<4>>, sm::Size<2>>,
        }
        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct GCpu {
            a: [[f32x3_align4; 4]; 2],
        }
        check_mismatch::<G, GCpu>();
    }

    #[test]
    fn test_struct_in_struct_mismatch() {
        let _guard = enable_color();

        #[derive(GpuLayout)]
        pub struct Inner {
            x: f32x1,
            y: f32x1,
        }

        #[derive(GpuLayout)]
        pub struct Outer {
            inner: sm::Struct<Inner>,
            z: u32x1,
        }

        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct InnerCpu {
            x: f32,
            y: u32, // Type mismatch here
        }

        #[derive(CpuLayout)]
        #[repr(C)]
        pub struct OuterCpu {
            inner: InnerCpu,
            z: u32,
        }

        check_mismatch::<Outer, OuterCpu>();
    }
}
