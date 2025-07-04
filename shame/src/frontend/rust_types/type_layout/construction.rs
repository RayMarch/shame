use std::fmt::Formatter;

use crate::{
    __private::SmallVec,
    ir::{ir_type::max_u64_po2_dividing, StoreType},
};
use super::{
    layoutable::{
        FieldOffsets, MatrixMajor, RuntimeSizedArray, RuntimeSizedArrayField, SizedField, SizedStruct, SizedType,
        UnsizedStruct,
    },
    *,
};
use TypeLayoutSemantics as TLS;

impl TypeLayout {
    /// Returns the type layout of the `LayoutableType`
    /// layed out according to the `repr`.
    pub fn new_layout_for(ty: &LayoutableType, repr: Repr) -> Self {
        match ty {
            LayoutableType::Sized(ty) => Self::from_sized_type(ty, repr),
            LayoutableType::UnsizedStruct(ty) => Self::from_unsized_struct(ty, repr),
            LayoutableType::RuntimeSizedArray(ty) => Self::from_runtime_sized_array(ty, repr),
        }
    }

    fn from_sized_type(ty: &SizedType, repr: Repr) -> Self {
        let (size, align, tls) = match &ty {
            SizedType::Vector(v) => (v.byte_size(repr), v.align(repr), TLS::Vector(*v)),
            SizedType::Atomic(a) => (
                a.byte_size(),
                a.align(repr),
                TLS::Vector(Vector::new(a.scalar.into(), ir::Len::X1)),
            ),
            SizedType::Matrix(m) => (m.byte_size(repr), m.align(repr), TLS::Matrix(*m)),
            SizedType::Array(a) => (
                a.byte_size(repr),
                a.align(repr),
                TLS::Array(
                    Rc::new(ElementLayout {
                        byte_stride: a.byte_stride(repr),
                        ty: Self::from_sized_type(&a.element, repr),
                    }),
                    Some(a.len.get()),
                ),
            ),
            SizedType::PackedVec(v) => (v.byte_size().as_u64(), v.align(repr), TLS::PackedVector(*v)),
            SizedType::Struct(s) => {
                let mut field_offsets = s.field_offsets(repr);
                let fields = (&mut field_offsets)
                    .zip(s.fields())
                    .map(|(offset, field)| sized_field_to_field_layout(field, offset, repr))
                    .collect();

                let (byte_size, align) = field_offsets.struct_byte_size_and_align();
                (
                    byte_size,
                    align,
                    TLS::Structure(Rc::new(StructLayout {
                        name: s.name.clone().into(),
                        fields,
                    })),
                )
            }
        };

        TypeLayout::new(Some(size), align, tls)
    }

    fn from_unsized_struct(s: &UnsizedStruct, repr: Repr) -> Self {
        let mut field_offsets = s.field_offsets(repr);
        let mut fields = (&mut field_offsets.sized_field_offsets())
            .zip(s.sized_fields.iter())
            .map(|(offset, field)| sized_field_to_field_layout(field, offset, repr))
            .collect::<Vec<_>>();

        let (field_offset, align) = field_offsets.last_field_offset_and_struct_align();

        let mut ty = Self::from_runtime_sized_array(&s.last_unsized.array, repr);
        // VERY IMPORTANT: TypeLayout::from_runtime_sized_array does not take into account
        // custom_min_align, but s.last_unsized.align does.
        ty.align = s.last_unsized.align(repr).into();

        fields.push(FieldLayoutWithOffset {
            rel_byte_offset: field_offset,
            field: FieldLayout {
                name: s.last_unsized.name.clone(),
                ty,
            },
        });

        TypeLayout::new(
            None,
            align,
            TLS::Structure(Rc::new(StructLayout {
                name: s.name.clone().into(),
                fields,
            })),
        )
    }

    fn from_runtime_sized_array(ty: &RuntimeSizedArray, repr: Repr) -> Self {
        Self::new(
            None,
            ty.align(repr),
            TLS::Array(
                Rc::new(ElementLayout {
                    byte_stride: ty.byte_stride(repr),
                    ty: Self::from_sized_type(&ty.element, repr),
                }),
                None,
            ),
        )
    }
}

fn sized_field_to_field_layout(field: &SizedField, offset: u64, repr: Repr) -> FieldLayoutWithOffset {
    let mut ty = TypeLayout::from_sized_type(&field.ty, repr);
    // VERY IMPORTANT: TypeLayout::from_sized_type does not take into account
    // custom_min_align and custom_min_size, but field.byte_size and field.align do.
    ty.byte_size = Some(field.byte_size(repr));
    ty.align = field.align(repr).into();
    FieldLayoutWithOffset {
        rel_byte_offset: offset,
        field: FieldLayout {
            name: field.name.clone(),
            ty,
        },
    }
}

impl<T: DerivableRepr> GpuTypeLayout<T> {
    /// Creates a new `GpuTypeLayout<T>` where `T` implements [`DerivableRepr`].
    ///
    /// All LayoutableType's can be layed out according to storage and packed layout rules,
    /// which corresponds to the available #[gpu_repr(packed)] and #[gpu_repr(storage)]
    /// attributes when deriving `GpuLayout` for a struct.
    ///
    /// `GpuTypeLayout<Uniform>` can be acquired via
    /// `GpuTypeLayout::<Uniform>::try_from(gpu_type_layout_storage)`.
    pub fn new(ty: impl Into<LayoutableType>) -> Self {
        Self {
            ty: ty.into(),
            _repr: PhantomData,
        }
    }
}

impl TryFrom<GpuTypeLayout<repr::Storage>> for GpuTypeLayout<repr::Uniform> {
    type Error = LayoutError;

    fn try_from(ty: GpuTypeLayout<repr::Storage>) -> Result<Self, Self::Error> {
        check_repr_equivalence_for_type(ty.layoutable_type(), Repr::Storage, Repr::Uniform)?;
        Ok(GpuTypeLayout {
            ty: ty.ty,
            _repr: PhantomData,
        })
    }
}

/// Checks whether the layout of `ty` as `actual_repr` and as `expected_repr` are compatible.
/// Compatible means that all field offsets of structs and all strides of arrays are the same.
///
/// Another way to say this is, that we are laying `ty` out according to two different
/// layout rules and checking whether the byte representation of those layouts is the same.
fn check_repr_equivalence_for_type(
    ty: &LayoutableType,
    actual_repr: Repr,
    expected_repr: Repr,
) -> Result<(), LayoutError> {
    let ctx = LayoutContext {
        top_level_type: ty,
        actual_repr,
        expected_repr,
        use_color: Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages).unwrap_or(false),
    };
    let is_sized = matches!(ctx.top_level_type, LayoutableType::Sized(_));
    if ctx.expected_repr.is_uniform() && !is_sized {
        return Err(LayoutError::UniformBufferMustBeSized(
            "wgsl",
            ctx.top_level_type.clone(),
        ));
    }

    match &ctx.top_level_type {
        LayoutableType::Sized(s) => check_compare_sized(ctx, s),
        LayoutableType::UnsizedStruct(s) => {
            let mut actual_offsets = s.field_offsets(ctx.actual_repr);
            let mut expected_offsets = s.field_offsets(ctx.expected_repr);
            check_sized_fields(
                ctx,
                s,
                s.sized_fields.iter().zip(
                    actual_offsets
                        .sized_field_offsets()
                        .zip(expected_offsets.sized_field_offsets()),
                ),
            )?;

            let (actual_last_offset, _) = actual_offsets.last_field_offset_and_struct_align();
            let (expected_last_offset, _) = expected_offsets.last_field_offset_and_struct_align();

            if actual_last_offset != expected_last_offset {
                let field = &s.last_unsized;
                let field_index = s.sized_fields.len();
                return Err(StructFieldOffsetError {
                    ctx: ctx.into(),
                    struct_type: s.clone().into(),
                    field_name: field.name.clone(),
                    field_index,
                    actual_offset: actual_last_offset,
                    expected_alignment: field.align(ctx.expected_repr),
                }
                .into());
            }

            Ok(())
        }
        LayoutableType::RuntimeSizedArray(a) => {
            let actual_stride = a.byte_stride(ctx.actual_repr);
            let expected_stride = a.byte_stride(ctx.expected_repr);
            match actual_stride == expected_stride {
                false => Err(ArrayStrideError {
                    ctx: ctx.into(),
                    actual_stride,
                    expected_stride,
                    element_ty: a.element.clone(),
                }
                .into()),
                true => Ok(()),
            }
        }
    }
}

fn check_compare_sized(ctx: LayoutContext, ty: &SizedType) -> Result<(), LayoutError> {
    match ty {
        SizedType::Struct(s) => {
            let mut actual_offsets = s.field_offsets(ctx.actual_repr);
            let mut expected_offsets = s.field_offsets(ctx.expected_repr);
            check_sized_fields(
                ctx,
                s,
                s.fields()
                    .into_iter()
                    .zip((&mut actual_offsets).zip(&mut expected_offsets)),
            )
        }
        SizedType::Array(a) => {
            let actual_stride = a.byte_stride(ctx.actual_repr);
            let expected_stride = a.byte_stride(ctx.expected_repr);
            match actual_stride == expected_stride {
                false => Err(ArrayStrideError {
                    ctx: ctx.into(),
                    actual_stride,
                    expected_stride,
                    element_ty: (*a.element).clone(),
                }
                .into()),
                true => Ok(()),
            }
        }
        SizedType::Vector(_) | SizedType::Matrix(_) | SizedType::Atomic(_) | SizedType::PackedVec(_) => {
            assert_eq!(ty.byte_size(ctx.actual_repr), ty.byte_size(ctx.expected_repr));
            Ok(())
        }
    }
}

fn check_sized_fields<'a>(
    ctx: LayoutContext,
    s: &(impl Into<StructKind> + Clone),
    fields_actual_and_expected_offsets: impl Iterator<Item = (&'a SizedField, (u64, u64))>,
) -> Result<(), LayoutError> {
    for (i, (field, (actual_offset, expected_offset))) in fields_actual_and_expected_offsets.enumerate() {
        if actual_offset != expected_offset {
            return Err(StructFieldOffsetError {
                ctx: ctx.into(),
                struct_type: s.clone().into(),
                field_name: field.name.clone(),
                field_index: i,
                actual_offset,
                expected_alignment: field.align(ctx.expected_repr),
            }
            .into());
        }

        check_compare_sized(ctx, &field.ty)?;
    }

    Ok(())
}

/// Enum of possible errors during comparison of two layouts for the same `LayoutableType`.
#[derive(thiserror::Error, Debug, Clone)]
pub enum LayoutError {
    #[error("{0}")]
    ArrayStride(#[from] ArrayStrideError),
    #[error("{0}")]
    StructureFieldOffset(#[from] StructFieldOffsetError),
    #[error(
        "The size of `{1}` on the gpu is not known at compile time. `{0}` \
    requires that the size of uniform buffers on the gpu is known at compile time."
    )]
    UniformBufferMustBeSized(&'static str, LayoutableType),
    #[error("{0} contains a `PackedVector`, which are not allowed in {1} memory layouts ")]
    MayNotContainPackedVec(LayoutableType, Repr),
}


#[derive(Debug, Clone, Copy)]
pub struct LayoutContext<'a> {
    top_level_type: &'a LayoutableType,
    actual_repr: Repr,
    expected_repr: Repr,
    use_color: bool,
}

#[derive(Debug, Clone)]
pub struct LayoutErrorContext {
    top_level_type: LayoutableType,
    actual_repr: Repr,
    expected_repr: Repr,
    use_color: bool,
}

impl From<LayoutContext<'_>> for LayoutErrorContext {
    fn from(ctx: LayoutContext) -> Self {
        LayoutErrorContext {
            top_level_type: ctx.top_level_type.clone(),
            actual_repr: ctx.actual_repr,
            expected_repr: ctx.expected_repr,
            use_color: ctx.use_color,
        }
    }
}

impl Repr {
    fn more_info_at(&self) -> &str { "https://www.w3.org/TR/WGSL/#memory-layouts" }
}

#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct ArrayStrideError {
    ctx: LayoutErrorContext,
    expected_stride: u64,
    actual_stride: u64,
    element_ty: SizedType,
}

impl std::error::Error for ArrayStrideError {}
impl Display for ArrayStrideError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let top_level = &self.ctx.top_level_type;
        writeln!(
            f,
            "array elements within type `{}` do not satisfy {} layout requirements.",
            top_level, self.ctx.expected_repr,
        )?;
        writeln!(
            f,
            "The array with `{}` elements requires stride {}, but has stride {}.",
            self.element_ty, self.expected_stride, self.actual_stride
        )?;
        writeln!(f, "The full layout of `{}` is:", top_level)?;
        let layout = TypeLayout::new_layout_for(&self.ctx.top_level_type, self.ctx.actual_repr);
        layout.write("", self.ctx.use_color, f)?;
        writeln!(f)?;
        writeln!(
            f,
            "\nfor more information on the layout rules, see {}",
            self.ctx.expected_repr.more_info_at()
        )?;
        Ok(())
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct StructFieldOffsetError {
    pub ctx: LayoutErrorContext,
    pub struct_type: StructKind,
    pub field_name: CanonName,
    pub field_index: usize,
    pub actual_offset: u64,
    pub expected_alignment: U32PowerOf2,
}

#[derive(Debug, Clone)]
pub enum StructKind {
    Sized(SizedStruct),
    Unsized(UnsizedStruct),
}

impl From<SizedStruct> for StructKind {
    fn from(value: SizedStruct) -> Self { StructKind::Sized(value) }
}
impl From<UnsizedStruct> for StructKind {
    fn from(value: UnsizedStruct) -> Self { StructKind::Unsized(value) }
}

impl std::error::Error for StructFieldOffsetError {}
impl Display for StructFieldOffsetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let top_level_type = &self.ctx.top_level_type;
        let top_level_name = match top_level_type {
            LayoutableType::Sized(SizedType::Struct(s)) => Some(&s.name),
            LayoutableType::UnsizedStruct(s) => Some(&s.name),
            _ => None,
        };

        let struct_name = match &self.struct_type {
            StructKind::Sized(s) => &s.name,
            StructKind::Unsized(s) => &s.name,
        };

        let is_top_level = match &self.struct_type {
            StructKind::Sized(s) => top_level_type == &LayoutableType::Sized(SizedType::Struct(s.clone())),
            StructKind::Unsized(s) => top_level_type == &LayoutableType::UnsizedStruct(s.clone()),
        };

        let structure_def_location = Context::try_with(call_info!(), |ctx| -> Option<_> {
            match &self.struct_type {
                StructKind::Sized(s) => {
                    let s: ir::SizedStruct = s.clone().try_into().ok()?;
                    ctx.struct_registry().get(&s).map(|def| def.call_info())
                }
                StructKind::Unsized(s) => {
                    let s: ir::BufferBlock = s.clone().try_into().ok()?;
                    ctx.struct_registry().get(&s).map(|def| def.call_info())
                }
            }
        })
        .flatten();


        writeln!(
            f,
            "The type `{top_level_type}` cannot be layed out according to {} layout rules.",
            self.ctx.expected_repr
        )?;
        match is_top_level {
            true => write!(f, "Struct `{}`", struct_name)?,
            false => write!(f, "It contains a struct `{}`, which", struct_name)?,
        }
        writeln!(
            f,
            " does not satisfy the {} memory layout requirements.",
            self.ctx.expected_repr
        )?;
        writeln!(f)?;

        if let Some(call_info) = structure_def_location {
            writeln!(f, "Definition at {call_info}")?;
        }

        write_struct_layout(
            &self.struct_type,
            self.ctx.actual_repr,
            self.ctx.use_color,
            Some(self.field_index),
            f,
        )?;

        let actual_align = max_u64_po2_dividing(self.actual_offset);
        let expected_alignment = self.expected_alignment.as_u32();
        writeln!(f)?;
        set_color(f, Some("#508EE3"), false)?;
        writeln!(
            f,
            "Field `{}` needs to be {} byte aligned, but has a byte-offset of {} which is only {actual_align} byte aligned.",
            self.field_name, expected_alignment, self.actual_offset
        )?;
        set_color(f, None, false)?;
        writeln!(f)?;

        writeln!(f, "Potential solutions include:")?;
        writeln!(
            f,
            "- add an #[align({})] attribute to the definition of `{}` (not supported by OpenGL/GLSL pipelines)",
            expected_alignment, self.field_name
        )?;
        match (self.ctx.actual_repr, self.ctx.expected_repr) {
            (Repr::Storage, Repr::Uniform) => writeln!(
                f,
                "- use the storage address space instead of the uniform address space"
            )?,
            _ => {}
        }

        writeln!(
            f,
            "- increase the offset of `{}` until it is divisible by {} by making previous fields larger or adding fields before it",
            self.field_name, expected_alignment
        )?;
        writeln!(f)?;

        writeln!(
            f,
            "In the {} address space, structs, arrays and array elements must be at least 16 byte aligned.",
            self.ctx.expected_repr
        )?;
        writeln!(
            f,
            "More info about the {} address space layout can be found at https://www.w3.org/TR/WGSL/#address-space-layout-constraints",
            self.ctx.expected_repr
        )?;
        Ok(())
    }
}

fn write_struct_layout<F>(
    struct_type: &StructKind,
    repr: Repr,
    colored: bool,
    highlight_field: Option<usize>,
    f: &mut F,
) -> std::fmt::Result
where
    F: Write,
{
    let use_256_color_mode = false;
    let color = |f_: &mut F, hex, field_index| match colored && Some(field_index) == highlight_field {
        true => set_color(f_, Some(hex), use_256_color_mode),
        false => Ok(()),
    };
    let reset = |f_: &mut F, field_index| match colored && Some(field_index) == highlight_field {
        true => set_color(f_, None, use_256_color_mode),
        false => Ok(()),
    };

    let (struct_name, sized_fields, mut field_offsets) = match struct_type {
        StructKind::Sized(s) => (&s.name, s.fields(), s.field_offsets(repr).into_inner()),
        StructKind::Unsized(s) => (&s.name, s.sized_fields.as_slice(), s.field_offsets(repr).into_inner()),
    };

    let indent = "  ";
    let field_decl_line = |field: &SizedField| format!("{indent}{}: {},", field.name, field.ty);
    let header = format!("struct {} {{", struct_name);
    let table_start_column = 1 + sized_fields
        .iter()
        .map(field_decl_line)
        .map(|s| s.len())
        .max()
        .unwrap_or(0)
        .max(header.chars().count());
    f.write_str(&header)?;
    for i in header.len()..table_start_column {
        f.write_char(' ')?
    }
    writeln!(f, "offset align size")?;
    for (field_index, (field, field_offset)) in sized_fields.iter().zip(&mut field_offsets).enumerate() {
        color(f, "#508EE3", field_index)?;

        let (align, size) = (field.align(repr), field.byte_size(repr));
        let decl_line = field_decl_line(field);
        f.write_str(&decl_line)?;
        // write spaces to table on the right
        for _ in decl_line.len()..table_start_column {
            f.write_char(' ')?
        }
        writeln!(f, "{:6} {:5} {:4}", field_offset, align.as_u32(), size)?;

        reset(f, field_index)?;
    }
    if let StructKind::Unsized(s) = struct_type {
        let field_index = sized_fields.len();
        color(f, "#508EE3", field_index)?;

        let last_field = &s.last_unsized;
        let (last_field_offset, _) = s.field_offsets(repr).last_field_offset_and_struct_align();

        let decl_line = format!("{indent}{}: {},", last_field.name, last_field.array);
        f.write_str(&decl_line)?;
        // write spaces to table on the right
        for _ in decl_line.len()..table_start_column {
            f.write_char(' ')?
        }
        write!(f, "{:6} {:5}", last_field_offset, last_field.align(repr).as_u64())?;


        reset(f, field_index)?;
    }
    writeln!(f, "}}")?;
    Ok(())
}
