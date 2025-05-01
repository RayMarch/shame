use std::{
    fmt::{Debug, Display, Write},
    num::NonZeroU32,
    ops::Deref,
    rc::Rc,
};

use crate::{
    call_info,
    common::{
        ignore_eq::{IgnoreInEqOrdHash, InEqOrd},
        prettify::set_color,
    },
    ir::{
        self,
        ir_type::{
            align_of_array, byte_size_of_array, round_up, stride_of_array, AlignedType, CanonName, LenEven,
            PackedVectorByteSize, ScalarTypeFp, ScalarTypeInteger,
        },
        recording::Context,
        Len, SizedType, Type,
    },
};
use thiserror::Error;

/// The type contained in the bytes of a `TypeLayout`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeLayoutSemantics {
    /// `vec<T, L>`
    Vector(ir::Len, ir::ScalarType),
    /// special compressed vectors for vertex attribute types
    ///
    /// see the [`crate::packed`] module
    PackedVector(ir::PackedVector),
    /// `mat<T, Cols, Rows>`, first `Len2` is cols, 2nd `Len2` is rows
    Matrix(ir::Len2, ir::Len2, ScalarTypeFp),
    /// `Array<T>` and `Array<T, Size<N>>`
    Array(Rc<ElementLayout>, Option<u32>), // not NonZeroU32, since for rust `CpuLayout`s the array size may be 0.
    /// structures which may be empty and may have an unsized last field
    Structure(Rc<StructLayout>),
}

/// The memory layout of a type.
///
/// This models only the layout, not other characteristics of the types.
/// For example an `Atomic<vec<u32, x1>>` is treated like a regular `vec<u32, x1>` layout wise.
///
/// The `PartialEq + Eq` implementation of `TypeLayout` is designed to answer the question
/// "do these two types have the same layout" so that uploading a type to the gpu
/// will result in no memory errors.
///
/// a layout comparison looks like this:
/// ```
/// assert!(f32::cpu_layout() == vec<f32, x1>::gpu_layout());
/// // or, more explicitly
/// assert_eq!(
///     <f32 as CpuLayout>::cpu_layout(),
///     <vec<f32, x1> as GpuLayout>::gpu_layout(),
/// );
/// ```
///
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TypeLayout {
    /// size in bytes (Some), or unsized (None)
    pub byte_size: Option<u64>,
    /// the byte alignment
    ///
    /// top level alignment is not considered relevant in some checks, but relevant in others (vertex array elements)
    pub byte_align: IgnoreInEqOrdHash<u64>,
    /// the type contained in the bytes of this type layout
    pub kind: TypeLayoutSemantics,
}

impl TypeLayout {
    pub(crate) fn new(byte_size: Option<u64>, byte_align: u64, kind: TypeLayoutSemantics) -> Self {
        Self {
            byte_size,
            byte_align: byte_align.into(),
            kind,
        }
    }

    pub(crate) fn from_rust_sized<T: Sized>(kind: TypeLayoutSemantics) -> Self {
        Self::new(Some(size_of::<T>() as u64), align_of::<T>() as u64, kind)
    }

    fn first_line_of_display_with_ellipsis(&self) -> String {
        let string = format!("{}", self);
        string.split_once('\n').map(|(s, _)| format!("{s}…")).unwrap_or(string)
    }

    /// a short name for this `TypeLayout`, useful for printing inline
    pub fn short_name(&self) -> String {
        match &self.kind {
            TypeLayoutSemantics::Vector { .. } |
            TypeLayoutSemantics::PackedVector { .. } |
            TypeLayoutSemantics::Matrix { .. } => format!("{}", self),
            TypeLayoutSemantics::Array(element_layout, n) => match n {
                Some(n) => format!("array<{}, {n}>", element_layout.ty.short_name()),
                None => format!("array<{}, runtime-sized>", element_layout.ty.short_name()),
            },
            TypeLayoutSemantics::Structure(s) => s.name.to_string(),
        }
    }
}

/// a sized or unsized struct type with 0 or more fields
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructLayout {
    pub name: IgnoreInEqOrdHash<CanonName>,
    pub fields: Vec<FieldLayoutWithOffset>,
}

impl StructLayout {
    /// this exists, because if in the future a refactor happens that separates
    /// fields into sized and unsized fields, the intention of this function is
    /// clear
    fn all_fields(&self) -> &[FieldLayoutWithOffset] { &self.fields }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldLayoutWithOffset {
    pub field: FieldLayout,
    pub rel_byte_offset: u64, // this being relative is used in TypeLayout::byte_size
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ElementLayout {
    pub byte_stride: u64,
    pub ty: TypeLayout,
}

/// the layout rules used when calculating the byte offsets and alignment of a type
#[derive(Debug, Clone, Copy)]
pub enum TypeLayoutRules {
    /// wgsl type layout rules, see https://www.w3.org/TR/WGSL/#memory-layouts
    Wgsl,
    // reprC,
    // Std140,
    // Std430,
    // Scalar,
}

#[allow(missing_docs)]
#[derive(Error, Debug, Clone)]
pub enum TypeLayoutError {
    #[error("An array cannot contain elements of an unsized type {elements}")]
    ArrayOfUnsizedElements { elements: TypeLayout },
    #[error("the type `{0}` has no defined {1:?} layout in shaders")]
    LayoutUndefined(Type, TypeLayoutRules),
    #[error("in `{parent_name}` at field `{field_name}`: {error}")]
    AtField {
        parent_name: CanonName,
        field_name: CanonName,
        error: Rc<TypeLayoutError>,
    },
    #[error("in element of array: {0}")]
    InArrayElement(Rc<TypeLayoutError>),
}

#[allow(missing_docs)] // TODO(docs) low priority docs, add after release
impl TypeLayout {
    pub fn from_ty(rules: TypeLayoutRules, ty: &ir::Type) -> Result<TypeLayout, TypeLayoutError> {
        match ty {
            Type::Unit | Type::Ptr(_, _, _) | Type::Ref(_, _, _) => {
                Err(TypeLayoutError::LayoutUndefined(ty.clone(), rules))
            }
            Type::Store(ty) => Self::from_store_ty(rules, ty),
        }
    }

    pub fn from_store_ty(rules: TypeLayoutRules, ty: &ir::StoreType) -> Result<TypeLayout, TypeLayoutError> {
        match ty {
            ir::StoreType::Sized(sized) => Ok(TypeLayout::from_sized_ty(rules, sized)),
            ir::StoreType::Handle(handle) => Err(TypeLayoutError::LayoutUndefined(ir::Type::Store(ty.clone()), rules)),
            ir::StoreType::RuntimeSizedArray(element) => Ok(Self::from_array(rules, element, None)),
            ir::StoreType::BufferBlock(s) => Ok(Self::from_struct(rules, s)),
        }
    }

    pub fn from_sized_ty(rules: TypeLayoutRules, ty: &ir::SizedType) -> TypeLayout {
        pub use TypeLayoutSemantics as Sem;
        let size = ty.byte_size();
        let align = ty.align();
        match ty {
            ir::SizedType::Vector(l, t) =>
            // we treat bool as a type that has a layout to allow for an
            // `Eq` operator on `TypeLayout` that behaves intuitively.
            // The layout of `bool`s is not actually observable in any part of the api.
            {
                TypeLayout::new(Some(size), align, Sem::Vector(*l, *t))
            }
            ir::SizedType::Matrix(c, r, t) => TypeLayout::new(Some(size), align, Sem::Matrix(*c, *r, *t)),
            ir::SizedType::Array(sized, l) => Self::from_array(rules, sized, Some(*l)),
            ir::SizedType::Atomic(t) => Self::from_sized_ty(rules, &ir::SizedType::Vector(ir::Len::X1, (*t).into())),
            ir::SizedType::Structure(s) => Self::from_struct(rules, s),
        }
    }

    pub fn from_array(rules: TypeLayoutRules, element: &ir::SizedType, len: Option<NonZeroU32>) -> TypeLayout {
        TypeLayout::new(
            len.map(|n| byte_size_of_array(element, n)),
            align_of_array(element),
            TypeLayoutSemantics::Array(
                Rc::new(ElementLayout {
                    byte_stride: match rules {
                        TypeLayoutRules::Wgsl => stride_of_array(element),
                    },
                    ty: Self::from_sized_ty(rules, element),
                }),
                len.map(NonZeroU32::get),
            ),
        )
    }

    pub fn from_struct(rules: TypeLayoutRules, s: &ir::Struct) -> TypeLayout {
        let (size, align, struct_layout) = StructLayout::from_ir_struct(rules, s);
        TypeLayout::new(size, align, TypeLayoutSemantics::Structure(Rc::new(struct_layout)))
    }

    pub fn struct_from_parts(
        rules: TypeLayoutRules,
        packed: bool,
        name: CanonName,
        fields: impl ExactSizeIterator<Item = FieldLayout>,
    ) -> Result<TypeLayout, StructLayoutError> {
        let (byte_size, byte_align, struct_) = StructLayout::new(rules, packed, name, fields)?;
        let layout = TypeLayout::new(byte_size, byte_align, TypeLayoutSemantics::Structure(Rc::new(struct_)));
        Ok(layout)
    }

    pub fn from_aligned_type(rules: TypeLayoutRules, ty: &AlignedType) -> TypeLayout {
        match ty {
            AlignedType::Sized(sized) => Self::from_sized_ty(rules, sized),
            AlignedType::RuntimeSizedArray(element) => Self::from_array(rules, element, None),
        }
    }

    pub(crate) fn writeln<W: Write>(&self, indent: &str, colored: bool, f: &mut W) -> std::fmt::Result {
        self.write(indent, colored, f)?;
        writeln!(f)
    }

    //TODO(low prio) try to figure out a cleaner way of writing these.
    pub(crate) fn write<W: Write>(&self, indent: &str, colored: bool, f: &mut W) -> std::fmt::Result {
        let tab = "  ";
        let use_256_color_mode = false;
        let color = |f_: &mut W, hex| match colored {
            true => set_color(f_, Some(hex), use_256_color_mode),
            false => Ok(()),
        };
        let reset = |f_: &mut W| match colored {
            true => set_color(f_, None, use_256_color_mode),
            false => Ok(()),
        };

        use TypeLayoutSemantics as Sem;

        match &self.kind {
            Sem::Vector(l, t) => match l {
                Len::X1 => write!(f, "{t}")?,
                l => write!(f, "{t}x{}", u64::from(*l))?,
            },
            Sem::PackedVector(c) => write!(f, "{}", c)?,
            Sem::Matrix(c, r, t) => write!(f, "{}", ir::SizedType::Matrix(*c, *r, *t))?,
            Sem::Array(t, n) => {
                let stride = t.byte_stride;
                write!(f, "array<")?;
                t.ty.write(&(indent.to_string() + tab), colored, f)?;
                if let Some(n) = n {
                    write!(f, ", {n}")?;
                }
                write!(f, ">  stride={stride}")?;
            }
            Sem::Structure(s) => {
                writeln!(f, "struct {} {{", s.name)?;
                {
                    let indent = indent.to_string() + tab;
                    for field in &s.fields {
                        let offset = field.rel_byte_offset;
                        let field = &field.field;
                        write!(f, "{indent}{offset:3} {}: ", field.name)?;
                        field.ty.write(&(indent.to_string() + tab), colored, f)?;
                        if let Some(size) = field.ty.byte_size {
                            let size = size.max(field.custom_min_size.unwrap_or(0));
                            write!(f, " size={size}")?;
                        } else {
                            write!(f, " size=?")?;
                        }
                        writeln!(f, ",")?;
                    }
                }
                write!(f, "{indent}}}")?;
                write!(f, " align={}", self.byte_align)?;
                if let Some(size) = self.byte_size {
                    write!(f, " size={size}")?;
                } else {
                    write!(f, " size=?")?;
                }
            }
        };
        Ok(())
    }

    pub fn align(&self) -> u64 { *self.byte_align }

    pub fn byte_size(&self) -> Option<u64> { self.byte_size }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldLayout {
    pub name: CanonName,
    pub custom_min_size: IgnoreInEqOrdHash<Option<u64>>, // whether size/align is custom doesn't matter for the layout equality.
    pub custom_min_align: IgnoreInEqOrdHash<Option<u64>>,
    pub ty: TypeLayout,
}

impl FieldLayout {
    fn byte_size(&self) -> Option<u64> {
        self.ty
            .byte_size()
            .map(|byte_size| byte_size.max(self.custom_min_size.unwrap_or(0)))
    }
    fn align(&self) -> u64 { self.ty.align().max(self.custom_min_align.map(u64::from).unwrap_or(1)) }
}

#[allow(missing_docs)]
#[derive(Error, Debug)]
pub enum StructLayoutError {
    #[error(
        "field #{unsized_field_index} in struct `{struct_name}` with {num_fields} is unsized. Only the last field may be unsized."
    )]
    UnsizedFieldMustBeLast {
        struct_name: CanonName,
        unsized_field_index: usize,
        num_fields: usize,
    },
}

impl StructLayout {
    /// returns a `(byte_size, byte_alignment, struct_layout)` tuple or an error
    ///
    /// this was created for the `#[derive(GpuLayout)]` macro to support the
    /// non-GpuType `PackedVec` for gpu_repr(packed) and non-packed.
    ///
    // TODO(low prio) find a way to merge all struct layout calculation functions in this codebase. This is very redundand.
    pub(crate) fn new(
        rules: TypeLayoutRules,
        packed: bool,
        name: CanonName,
        fields: impl ExactSizeIterator<Item = FieldLayout>,
    ) -> Result<(Option<u64>, u64, StructLayout), StructLayoutError> {
        let mut total_byte_size = None;
        let mut total_align = 1;
        let num_fields = fields.len();
        let struct_layout = StructLayout {
            name: name.clone().into(),
            fields: {
                let mut offset_so_far = 0;
                let mut fields_with_offset = Vec::new();
                for (i, field) in fields.enumerate() {
                    let is_last = i + 1 == num_fields;
                    fields_with_offset.push(FieldLayoutWithOffset {
                        field: field.clone(),
                        rel_byte_offset: match rules {
                            TypeLayoutRules::Wgsl => {
                                let field_offset = match (packed, *field.custom_min_align) {
                                    (true, None) => offset_so_far,
                                    (true, Some(custom_align)) => round_up(custom_align, offset_so_far),
                                    (false, _) => round_up(field.align(), offset_so_far),
                                };
                                match (field.byte_size(), is_last) {
                                    (Some(field_size), _) => {
                                        offset_so_far = field_offset + field_size;
                                        Ok(())
                                    }
                                    (None, true) => Ok(()),
                                    (None, false) => Err(StructLayoutError::UnsizedFieldMustBeLast {
                                        struct_name: name.clone(),
                                        unsized_field_index: i,
                                        num_fields,
                                    }),
                                }?;
                                field_offset
                            }
                        },
                    });
                    total_align = total_align.max(field.align());
                    if is_last {
                        // wgsl spec:
                        //   roundUp(AlignOf(S), justPastLastMember)
                        //   where justPastLastMember = OffsetOfMember(S,N) + SizeOfMember(S,N)

                        // if the last field size is None (= unsized), just_past_last is None (= unsized)
                        let just_past_last = field.byte_size().map(|_| offset_so_far);
                        total_byte_size = just_past_last.map(|just_past_last| round_up(total_align, just_past_last));
                    }
                }
                fields_with_offset
            },
        };
        Ok((total_byte_size, total_align, struct_layout))
    }

    /// returns a `(byte_size, byte_alignment, struct_layout)` tuple
    #[doc(hidden)]
    pub fn from_ir_struct(rules: TypeLayoutRules, s: &ir::Struct) -> (Option<u64>, u64, StructLayout) {
        let mut total_byte_size = None;
        let struct_layout = StructLayout {
            name: s.name().clone().into(),
            fields: {
                let mut offset = 0;
                let mut fields = Vec::new();
                for field in s.sized_fields() {
                    fields.push(FieldLayoutWithOffset {
                        field: FieldLayout {
                            name: field.name.clone(),
                            ty: TypeLayout::from_sized_ty(rules, &field.ty),
                            custom_min_size: field.custom_min_size.into(),
                            custom_min_align: field.custom_min_align.map(u64::from).into(),
                        },
                        rel_byte_offset: match rules {
                            TypeLayoutRules::Wgsl => {
                                let rel_byte_offset = round_up(field.align(), offset);
                                offset = rel_byte_offset + field.byte_size();
                                rel_byte_offset
                            }
                        },
                    })
                }
                if let Some(unsized_array) = s.last_unsized_field() {
                    fields.push(FieldLayoutWithOffset {
                        field: FieldLayout {
                            name: unsized_array.name.clone(),
                            custom_min_align: unsized_array.custom_min_align.map(u64::from).into(),
                            custom_min_size: None.into(),
                            ty: TypeLayout::from_array(rules, &unsized_array.element_ty, None),
                        },
                        rel_byte_offset: round_up(unsized_array.align(), offset),
                    })
                } else {
                    total_byte_size = Some(s.min_byte_size());
                }
                fields
            },
        };
        (total_byte_size, s.align(), struct_layout)
    }
}

impl Display for TypeLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let colored = Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages).unwrap_or(false);
        self.write("", colored, f)
    }
}

impl Debug for TypeLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.write("", false, f) }
}

#[derive(Clone)]
pub struct LayoutMismatch {
    /// 2 (name, layout) pairs
    layouts: [(String, TypeLayout); 2],
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
        layouts: [(&str, &TypeLayout); 2],
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
                                    a_field.field.align()
                                );
                                color_b(f);
                                writeln!(
                                    f,
                                    "{b_name}{SEP}{indent}{:3} {}: {b_ty_string} align={}",
                                    b_field.rel_byte_offset,
                                    b_field.field.name,
                                    b_field.field.align()
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
                let align_matches = a.align() == b.align();
                let size_matches = a.byte_size() == b.byte_size();
                if !align_matches && size_matches {
                    writeln!(f);
                    color_a(f);
                    writeln!(f, "{a_name}{SEP}{indent}align={}", a.align());
                    color_b(f);
                    writeln!(f, "{b_name}{SEP}{indent}align={}", b.align());
                    color_reset(f);
                    return Err(MismatchWasFound);
                } else {
                    match align_matches {
                        true => write!(f, " align={}", a.align()),
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
            (S::Vector(na, ta), S::Vector(nb, tb)) => {
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
            (S::Matrix(c, r, t), S::Matrix(c1, r1, t1)) => {
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

impl TypeLayout {
    /// takes two pairs of `(debug_name, layout)` and compares them for equality.
    ///
    /// if the two layouts are not equal it uses the debug names in the returned
    /// error to tell the two layouts apart.
    pub(crate) fn check_eq(a: (&str, &TypeLayout), b: (&str, &TypeLayout)) -> Result<(), LayoutMismatch> {
        match a.1 == b.1 {
            true => Ok(()),
            false => Err(LayoutMismatch {
                layouts: [(a.0.into(), a.1.clone()), (b.0.into(), b.1.clone())],
                colored_error: Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages)
                    .unwrap_or(false),
            }),
        }
    }
}
