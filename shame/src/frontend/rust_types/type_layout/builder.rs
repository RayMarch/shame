use crate::{
    __private::SmallVec,
    cpu_shareable::{self as cs, Major, Repr, SizedField, UnsizedStruct},
    ir::{ir_type::max_u64_po2_dividing, StoreType},
};
use super::*;
use TypeLayoutSemantics as TLS;

impl TypeLayout<constraint::Storage> {
    pub fn new_storage_layout_for(ty: impl Into<LayoutType>) -> Self {
        type_layout_internal::cast_unchecked(TypeLayout::new_layout_for(ty, Repr::Storage))
    }

    /// Get the `CpuShareableType` this layout is based on.
    pub fn cpu_shareable(&self) -> &LayoutType {
        self.cpu_shareable
            .as_ref()
            .expect("constraint::Storage always contains a cpu-shareable")
    }
}

impl TypeLayout<constraint::Uniform> {
    /// Takes a `SizedType`, because wgsl only supports sized uniform buffers.
    /// Use `new_layout_for_unchecked`to obtain the the uniform layout of an unsized host-shareable. /// Using the unsized layout with wgsl as your target language will cause an error.
    pub fn new_uniform_layout_for(ty: impl Into<LayoutType>) -> Self {
        type_layout_internal::cast_unchecked(TypeLayout::new_layout_for(ty, Repr::Uniform))
    }

    /// Get the `CpuShareableType` this layout is based on.
    pub fn cpu_shareable(&self) -> &LayoutType {
        self.cpu_shareable
            .as_ref()
            .expect("constraint::Uniform always contains a cpu-shareable")
    }
}

impl TypeLayout<constraint::Packed> {
    pub fn new_packed_layout_for(ty: impl Into<LayoutType>) -> Self {
        type_layout_internal::cast_unchecked(TypeLayout::new_layout_for(ty, Repr::Packed))
    }

    /// Get the `CpuShareableType` this layout is based on.
    pub fn cpu_shareable(&self) -> &LayoutType {
        self.cpu_shareable
            .as_ref()
            .expect("constraint::Storage always contains a cpu-shareable")
    }
}

impl TypeLayout {
    pub fn new_layout_for(ty: impl Into<LayoutType>, repr: Repr) -> Self {
        match ty.into() {
            LayoutType::Sized(ty) => Self::from_sized_type(ty, repr),
            LayoutType::UnsizedStruct(ty) => Self::from_unsized_struct(ty, repr),
            LayoutType::RuntimeSizedArray(ty) => Self::from_runtime_sized_array(ty, repr),
        }
    }

    fn from_sized_type(ty: cs::SizedType, repr: Repr) -> Self {
        let (size, align, tls) = match &ty {
            cs::SizedType::Vector(v) => (v.byte_size(), v.align(), TLS::Vector(*v)),
            cs::SizedType::Atomic(a) => (
                a.byte_size(),
                a.align(),
                TLS::Vector(Vector::new(a.scalar.into(), ir::Len::X1)),
            ),
            cs::SizedType::Matrix(m) => (m.byte_size(Major::Row), m.align(repr, Major::Row), TLS::Matrix(*m)),
            cs::SizedType::Array(a) => (
                a.byte_size(repr),
                a.align(repr),
                TLS::Array(
                    Rc::new(ElementLayout {
                        byte_stride: a.byte_stride(repr),
                        ty: Self::from_sized_type((*a.element).clone(), repr),
                    }),
                    Some(a.len.get()),
                ),
            ),
            cs::SizedType::PackedVec(v) => (v.byte_size().as_u64(), v.align(), TLS::PackedVector(*v)),
            cs::SizedType::Struct(s) => {
                let mut field_offsets = s.field_offsets(repr);
                let fields = (&mut field_offsets)
                    .zip(s.fields())
                    .map(|(offset, field)| sized_field_to_field_layout(field, offset, repr))
                    .collect();

                (
                    field_offsets.byte_size(),
                    field_offsets.align(),
                    TLS::Structure(Rc::new(StructLayout {
                        name: s.name.clone().into(),
                        fields,
                    })),
                )
            }
        };

        TypeLayout::new(Some(size), align, tls, Some(ty.into()))
    }

    fn from_unsized_struct(s: UnsizedStruct, repr: Repr) -> Self {
        let mut field_offsets = s.sized_field_offsets(repr);
        let mut fields = (&mut field_offsets)
            .zip(s.sized_fields.iter())
            .map(|(offset, field)| sized_field_to_field_layout(field, offset, repr))
            .collect::<Vec<_>>();

        let (field_offset, align) = s.last_field_offset_and_struct_align(field_offsets);
        fields.push(FieldLayoutWithOffset {
            rel_byte_offset: field_offset,
            field: FieldLayout {
                name: s.last_unsized.name.clone(),
                custom_min_size: None.into(),
                custom_min_align: s.last_unsized.custom_min_align.into(),
                ty: Self::from_runtime_sized_array(s.last_unsized.array.clone(), repr),
            },
        });

        TypeLayout::new(
            None,
            align,
            TLS::Structure(Rc::new(StructLayout {
                name: s.name.clone().into(),
                fields,
            })),
            Some(s.into()),
        )
    }


    fn from_runtime_sized_array(ty: cs::RuntimeSizedArray, repr: Repr) -> Self {
        Self::new(
            None,
            ty.byte_align(repr),
            TLS::Array(
                Rc::new(ElementLayout {
                    byte_stride: ty.byte_stride(repr),
                    ty: Self::from_sized_type(ty.element.clone(), repr),
                }),
                None,
            ),
            Some(ty.into()),
        )
    }
}

fn sized_field_to_field_layout(field: &SizedField, offset: u64, repr: Repr) -> FieldLayoutWithOffset {
    FieldLayoutWithOffset {
        rel_byte_offset: offset,
        field: FieldLayout {
            name: field.name.clone(),
            custom_min_size: field.custom_min_size.into(),
            custom_min_align: field.custom_min_align.into(),
            ty: TypeLayout::from_sized_type(field.ty.clone(), repr),
        },
    }
}

impl<'a> TryFrom<&'a TypeLayout<constraint::Storage>> for TypeLayout<constraint::Uniform> {
    type Error = UniformLayoutError;

    fn try_from(s_layout: &'a TypeLayout<constraint::Storage>) -> Result<Self, Self::Error> {
        let cpu_shareable = s_layout.cpu_shareable().clone();

        // Checking whether it's sized
        let sized = match cpu_shareable {
            LayoutType::Sized(sized) => sized,
            _ => {
                return Err(UniformLayoutError::MustBeSized("wgsl", cpu_shareable.into()));
            }
        };

        let u_layout = TypeLayout::new_uniform_layout_for(sized);

        // Checking fields
        fn check_layout<S: TypeConstraint, U: TypeConstraint>(
            s_layout: &TypeLayout<S>,
            u_layout: &TypeLayout<U>,
            is_top_level: bool,
        ) -> Result<(), Mismatch> {
            // kinds are the same, because the type layouts are based on the same cpu shareable
            match (&s_layout.kind, &u_layout.kind) {
                (TLS::Structure(s), TLS::Structure(u)) => {
                    for (i, (s_field, u_field)) in s.fields.iter().zip(u.fields.iter()).enumerate() {
                        // Checking field offset
                        if s_field.rel_byte_offset != u_field.rel_byte_offset {
                            let (sized_fields, last_unsized) = match s_layout.get_cpu_shareable().unwrap() {
                                LayoutType::Sized(cs::SizedType::Struct(s)) => (s.fields().to_vec(), None),
                                LayoutType::UnsizedStruct(s) => (s.sized_fields.clone(), Some(s.last_unsized.clone())),
                                _ => unreachable!("is struct, because tls is struct"),
                            };

                            return Err(Mismatch::StructureFieldOffset(StructureFieldOffsetError {
                                struct_layout: (**s).clone(),
                                sized_fields,
                                last_unsized,
                                field_name: s_field.field.name.clone(),
                                field_index: i,
                                actual_offset: s_field.rel_byte_offset,
                                expected_alignment: u_field.field.byte_align(),
                                is_top_level,
                            }));
                        }

                        check_layout(&s_field.field.ty, &u_field.field.ty, false)?;
                    }
                }
                (TLS::Array(s_ele, _), TLS::Array(u_ele, _)) => {
                    // As long as the strides are the same, the size must be the same and
                    // the element align must divide the stride (uniform requirement), because
                    // u_layout is a valid uniform layout.
                    if s_ele.byte_stride != u_ele.byte_stride {
                        let element_ty = match u_ele.ty.get_cpu_shareable().unwrap() {
                            LayoutType::Sized(s) => s.clone(),
                            _ => {
                                unreachable!("elements of an array are always sized for TypeLayout<Storage | Uniform>")
                            }
                        };
                        return Err(Mismatch::ArrayStride(ArrayStrideError {
                            expected: u_ele.byte_stride,
                            actual: s_ele.byte_stride,
                            element_ty,
                        }));
                    }

                    check_layout(&s_ele.ty, &u_ele.ty, false)?;
                }
                _ => {}
            }

            Ok(())
        }


        match check_layout(s_layout, &u_layout, true) {
            Ok(()) => Ok(u_layout),
            Err(e) => {
                let ctx = LayoutErrorContext {
                    s_layout: s_layout.clone(),
                    u_layout,
                    // TODO(chronicl) default shouldn't be true?
                    use_color: Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages)
                        .unwrap_or(true),
                };

                use UniformLayoutError as U;
                let e = match e {
                    Mismatch::ArrayStride(e) => U::ArrayStride(WithContext::new(ctx, e)),
                    Mismatch::StructureFieldOffset(e) => U::StructureFieldOffset(WithContext::new(ctx, e)),
                };
                Err(e)
            }
        }
    }
}

type DepthIndex = SmallVec<usize, 5>;

#[derive(thiserror::Error, Debug, Clone)]
pub enum UniformLayoutError {
    #[error("{0}")]
    ArrayStride(WithContext<ArrayStrideError>),
    #[error("{0}")]
    StructureFieldOffset(WithContext<StructureFieldOffsetError>),
    // TODO(chronicl) consider using CpuShareableType and implementing Display for it instead
    // of converting to StoreType
    #[error(
        "The size of `{1}` on the gpu is not known at compile time. `{0}` \
    requires that the size of uniform buffers on the gpu is known at compile time."
    )]
    MustBeSized(&'static str, StoreType),
}

#[derive(Debug, Clone)]
pub enum Mismatch {
    ArrayStride(ArrayStrideError),
    StructureFieldOffset(StructureFieldOffsetError),
}


#[derive(Debug, Clone)]
pub struct LayoutErrorContext {
    s_layout: TypeLayout<constraint::Storage>,
    u_layout: TypeLayout<constraint::Uniform>,
    use_color: bool,
}

#[derive(Debug, Clone)]
pub struct WithContext<T> {
    ctx: LayoutErrorContext,
    inner: T,
}

impl<T> std::ops::Deref for WithContext<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target { &self.inner }
}

impl<T> WithContext<T> {
    fn new(ctx: LayoutErrorContext, inner: T) -> Self { Self { ctx, inner } }
}


#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct ArrayStrideError {
    expected: u64,
    actual: u64,
    element_ty: cs::SizedType,
}

impl std::error::Error for WithContext<ArrayStrideError> {}
impl Display for WithContext<ArrayStrideError> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let top_level: StoreType = self.ctx.s_layout.cpu_shareable().clone().into();
        writeln!(
            f,
            "array elements within type `{}` do not satisfy uniform layout requirements.",
            top_level
        )?;
        writeln!(
            f,
            "The array with `{}` elements requires stride {}, but has stride {}.",
            Into::<ir::SizedType>::into(self.element_ty.clone()),
            self.expected,
            self.actual
        )?;
        writeln!(f, "The full layout of `{}` is:", top_level)?;
        self.ctx.s_layout.write("", self.ctx.use_color, f)?;
        writeln!(f)?;
        writeln!(
            f,
            "\nfor more information on the layout rules, see https://www.w3.org/TR/WGSL/#address-space-layout-constraints",
        )?;
        Ok(())
    }
}


#[derive(Debug, Clone)]
pub struct StructureFieldOffsetError {
    pub struct_layout: StructLayout,
    pub sized_fields: Vec<cs::SizedField>,
    pub last_unsized: Option<cs::RuntimeSizedArrayField>,
    pub field_name: CanonName,
    pub field_index: usize,
    pub actual_offset: u64,
    pub expected_alignment: U32PowerOf2,
    pub is_top_level: bool,
}

impl std::error::Error for WithContext<StructureFieldOffsetError> {}
impl Display for WithContext<StructureFieldOffsetError> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO(chronicl) remove conversion here if CpuShareableType implements Display
        let top_level_type: StoreType = self.ctx.s_layout.cpu_shareable().clone().into();
        writeln!(
            f,
            "The type `{top_level_type}` cannot be used as a uniform buffer binding."
        )?;

        match self.is_top_level {
            true => write!(f, "Struct `{}`", &*self.struct_layout.name)?,
            false => write!(f, "It contains a struct `{}`, which", &*self.struct_layout.name)?,
        }
        writeln!(f, " does not satisfy the uniform memory layout requirements.",)?;
        writeln!(f)?;
        // TODO(chronicl) structure_def_location
        // if let Some(call_info) = structure_def_location {
        //     writeln!(f, "Definition at {call_info}")?;
        // }

        write_struct_layout(
            &self.struct_layout,
            &self.sized_fields,
            self.last_unsized.as_ref(),
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

        writeln!(f, "Potential solutions include:");
        writeln!(
            f,
            "- add an #[align({})] attribute to the definition of `{}` (not supported by OpenGL/GLSL pipelines)",
            expected_alignment, self.field_name
        );
        writeln!(f, "- use a storage binding instead of a uniform binding");
        writeln!(
            f,
            "- increase the offset of `{}` until it is divisible by {} by making previous fields larger or adding fields before it",
            self.field_name, expected_alignment
        );
        writeln!(f)?;

        writeln!(
            f,
            "In the uniform address space, structs, arrays and array elements must be at least 16 byte aligned.\nMore info about the uniform address space layout can be found at https://www.w3.org/TR/WGSL/#address-space-layout-constraints"
        )?;
        Ok(())
    }
}

/// Panics if s is not the layout of a struct and doesn't contain a cpu-shareable.
fn write_struct_layout<F>(
    struct_layout: &StructLayout,
    sized_fields: &[cs::SizedField],
    last_unsized: Option<&cs::RuntimeSizedArrayField>,
    colored: bool,
    highlight_field: Option<usize>,
    f: &mut F,
) -> std::fmt::Result
where
    F: Write,
{
    let use_256_color_mode = false;
    let color = |f_: &mut F, hex| match colored {
        true => set_color(f_, Some(hex), use_256_color_mode),
        false => Ok(()),
    };
    let reset = |f_: &mut F| match colored {
        true => set_color(f_, None, use_256_color_mode),
        false => Ok(()),
    };

    let struct_name = &*struct_layout.name;

    let indent = "  ";
    let field_decl_line = |field: &cs::SizedField| {
        let sized: ir::SizedType = field.ty.clone().into();
        format!("{indent}{}: {},", field.name, sized)
    };
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
    for (i, (field, layout)) in sized_fields.iter().zip(struct_layout.fields.iter()).enumerate() {
        if Some(i) == highlight_field {
            color(f, "#508EE3")?;
        }
        let (align, size) = (layout.ty.byte_align(), layout.ty.byte_size().expect("is sized field"));
        let decl_line = field_decl_line(field);
        f.write_str(&decl_line)?;
        // write spaces to table on the right
        for _ in decl_line.len()..table_start_column {
            f.write_char(' ')?
        }
        writeln!(f, "{:6} {:5} {:4}", layout.rel_byte_offset, align.as_u32(), size)?;
        if Some(i) == highlight_field {
            reset(f)?;
        }
    }
    if let Some(last_field) = last_unsized {
        let layout = struct_layout.fields.last().expect("structs have at least one field");

        let store_type: ir::StoreType = last_field.array.clone().into();
        let decl_line = format!("{indent}{}: {},", last_field.name, store_type);
        f.write_str(&decl_line)?;
        // write spaces to table on the right
        for _ in decl_line.len()..table_start_column {
            f.write_char(' ')?
        }
        write!(f, "{:6} {:5}", layout.rel_byte_offset, layout.ty.byte_align().as_u32())?;
    }
    writeln!(f, "}}")?;
    Ok(())
}
