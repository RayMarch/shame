use super::*;

//     Conversions to ir types     //

/// Errors that can occur when converting IR types to layoutable types.
#[derive(thiserror::Error, Debug)]
pub enum IRConversionError {
    /// Packed vectors do not exist in the shader type system.
    #[error("Type is or contains a packed vector, which does not exist in the shader type system.")]
    ContainsPackedVector,
    /// Struct field names must be unique in the shader type system.
    #[error("{0}")]
    DuplicateFieldName(#[from] DuplicateFieldNameError),
}

#[derive(Debug)]
pub struct DuplicateFieldNameError {
    pub struct_type: StructKind,
    pub first_occurence: usize,
    pub second_occurence: usize,
    pub use_color: bool,
}

impl std::error::Error for DuplicateFieldNameError {}

impl std::fmt::Display for DuplicateFieldNameError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let (struct_name, sized_fields, last_unsized) = match &self.struct_type {
            StructKind::Sized(s) => (&s.name, s.fields(), None),
            StructKind::Unsized(s) => (&s.name, s.sized_fields.as_slice(), Some(&s.last_unsized)),
        };

        let indent = "  ";
        let is_duplicate = |i| self.first_occurence == i || self.second_occurence == i;
        let arrow = |i| match is_duplicate(i) {
            true => " <--",
            false => "",
        };
        let color = |f: &mut Formatter<'_>, i| {
            if (self.use_color && is_duplicate(i)) {
                set_color(f, Some("#508EE3"), false)
            } else {
                Ok(())
            }
        };
        let color_reset = |f: &mut Formatter<'_>, i| {
            if self.use_color && is_duplicate(i) {
                set_color(f, None, false)
            } else {
                Ok(())
            }
        };

        writeln!(
            f,
            "Type contains or is a struct with duplicate field names.\
            Field names must be unique in the shader type system.\n\
            The following struct contains duplicate field names:"
        )?;
        let header = writeln!(f, "struct {} {{", struct_name);
        for (i, field) in sized_fields.iter().enumerate() {
            color(f, i)?;
            writeln!(f, "{indent}{}: {},{}", field.name, field.ty, arrow(i))?;
            color_reset(f, i)?;
        }
        if let Some(field) = last_unsized {
            let i = sized_fields.len();
            color(f, i)?;
            writeln!(f, "{indent}{}: {},{}", field.name, field.array, arrow(i))?;
            color_reset(f, i)?;
        }
        writeln!(f, "}}")?;

        Ok(())
    }
}

fn check_for_duplicate_field_names(
    sized_fields: &[SizedField],
    last_unsized: Option<&RuntimeSizedArrayField>,
) -> Option<(usize, usize)> {
    // Brute force search > HashMap for the amount of fields
    // we'd usually deal with.
    let mut duplicate_fields = None;
    for (i, field1) in sized_fields.iter().enumerate() {
        for (j, field2) in sized_fields.iter().enumerate().skip(i + 1) {
            if field1.name == field2.name {
                duplicate_fields = Some((i, j));
                break;
            }
        }
        if let Some(last_unsized) = last_unsized {
            if field1.name == last_unsized.name {
                duplicate_fields = Some((i, sized_fields.len()));
                break;
            }
        }
    }
    duplicate_fields
}

#[track_caller]
fn use_color() -> bool { Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages).unwrap_or(false) }

impl TryFrom<LayoutableType> for ir::StoreType {
    type Error = IRConversionError;

    fn try_from(ty: LayoutableType) -> Result<Self, Self::Error> {
        match ty {
            LayoutableType::Sized(s) => Ok(ir::StoreType::Sized(s.try_into()?)),
            LayoutableType::RuntimeSizedArray(s) => Ok(ir::StoreType::RuntimeSizedArray(s.element.try_into()?)),
            LayoutableType::UnsizedStruct(s) => Ok(ir::StoreType::BufferBlock(s.try_into()?)),
        }
    }
}

impl TryFrom<SizedType> for ir::SizedType {
    type Error = IRConversionError;

    fn try_from(host: SizedType) -> Result<Self, Self::Error> {
        Ok(match host {
            SizedType::Vector(v) => ir::SizedType::Vector(v.len, v.scalar.into()),
            SizedType::Matrix(m) => ir::SizedType::Matrix(m.columns, m.rows, m.scalar),
            SizedType::Array(a) => {
                let element = Rc::unwrap_or_clone(a.element);
                let converted_element = element.try_into()?;
                ir::SizedType::Array(Rc::new(converted_element), a.len)
            }
            SizedType::Atomic(i) => ir::SizedType::Atomic(i.scalar),
            SizedType::PackedVec(_) => return Err(IRConversionError::ContainsPackedVector),
            SizedType::Struct(s) => ir::SizedType::Structure(s.try_into()?),
        })
    }
}

impl From<ScalarType> for ir::ScalarType {
    fn from(scalar_type: ScalarType) -> Self {
        match scalar_type {
            ScalarType::F16 => ir::ScalarType::F16,
            ScalarType::F32 => ir::ScalarType::F32,
            ScalarType::F64 => ir::ScalarType::F64,
            ScalarType::U32 => ir::ScalarType::U32,
            ScalarType::I32 => ir::ScalarType::I32,
        }
    }
}

impl TryFrom<SizedStruct> for ir::ir_type::SizedStruct {
    type Error = IRConversionError;

    fn try_from(ty: SizedStruct) -> Result<Self, Self::Error> {
        if let Some((first, second)) = check_for_duplicate_field_names(ty.fields(), None) {
            return Err(IRConversionError::DuplicateFieldName(DuplicateFieldNameError {
                struct_type: StructKind::Sized(ty),
                first_occurence: first,
                second_occurence: second,
                use_color: use_color(),
            }));
        }

        let mut fields: Vec<ir::ir_type::SizedField> = Vec::new();
        for field in ty.fields {
            fields.push(field.try_into()?);
        }
        let last_field = fields.pop().unwrap();

        match ir::ir_type::SizedStruct::new_nonempty(ty.name, fields, last_field) {
            Ok(s) => Ok(s),
            Err(StructureFieldNamesMustBeUnique) => unreachable!("checked above"),
        }
    }
}

impl TryFrom<UnsizedStruct> for ir::ir_type::BufferBlock {
    type Error = IRConversionError;

    fn try_from(ty: UnsizedStruct) -> Result<Self, Self::Error> {
        if let Some((first, second)) = check_for_duplicate_field_names(&ty.sized_fields, Some(&ty.last_unsized)) {
            return Err(IRConversionError::DuplicateFieldName(DuplicateFieldNameError {
                struct_type: StructKind::Unsized(ty),
                first_occurence: first,
                second_occurence: second,
                use_color: use_color(),
            }));
        }

        let mut sized_fields: Vec<ir::ir_type::SizedField> = Vec::new();

        for field in ty.sized_fields {
            sized_fields.push(field.try_into()?);
        }

        let last_unsized = ty.last_unsized.try_into()?;

        match ir::ir_type::BufferBlock::new(ty.name, sized_fields, Some(last_unsized)) {
            Ok(b) => Ok(b),
            Err(BufferBlockDefinitionError::FieldNamesMustBeUnique) => unreachable!("checked above"),
            Err(BufferBlockDefinitionError::MustHaveAtLeastOneField) => {
                unreachable!("last_unsized is at least one field")
            }
        }
    }
}

impl TryFrom<RuntimeSizedArray> for ir::StoreType {
    type Error = IRConversionError;

    fn try_from(array: RuntimeSizedArray) -> Result<Self, Self::Error> {
        Ok(ir::StoreType::RuntimeSizedArray(array.element.try_into()?))
    }
}

impl TryFrom<SizedField> for ir::ir_type::SizedField {
    type Error = IRConversionError;

    fn try_from(f: SizedField) -> Result<Self, Self::Error> {
        Ok(ir::SizedField::new(
            f.name,
            f.custom_min_size,
            f.custom_min_align,
            f.ty.try_into()?,
        ))
    }
}

impl TryFrom<RuntimeSizedArrayField> for ir::ir_type::RuntimeSizedArrayField {
    type Error = IRConversionError;

    fn try_from(f: RuntimeSizedArrayField) -> Result<Self, Self::Error> {
        Ok(ir::RuntimeSizedArrayField::new(
            f.name,
            f.custom_min_align,
            f.array.element.try_into()?,
        ))
    }
}


//     Conversions from ir types     //

/// Type contains bools, which doesn't have a known layout.
#[derive(thiserror::Error, Debug)]
#[error("Type contains bools, which doesn't have a known layout.")]
pub struct ContainsBoolsError;

/// Errors that can occur when converting IR types to layoutable types.
#[derive(thiserror::Error, Debug)]
pub enum LayoutableConversionError {
    /// Type contains bools, which don't have a standardized memory layout.
    #[error("Type contains bools, which don't have a standardized memory layout.")]
    ContainsBool,
    /// Type is a handle, which don't have a standardized memory layout.
    #[error("Type is a handle, which don't have a standardized memory layout.")]
    IsHandle,
}

impl TryFrom<ir::ScalarType> for ScalarType {
    type Error = ContainsBoolsError;

    fn try_from(value: ir::ScalarType) -> Result<Self, Self::Error> {
        Ok(match value {
            ir::ScalarType::F16 => ScalarType::F16,
            ir::ScalarType::F32 => ScalarType::F32,
            ir::ScalarType::F64 => ScalarType::F64,
            ir::ScalarType::U32 => ScalarType::U32,
            ir::ScalarType::I32 => ScalarType::I32,
            ir::ScalarType::Bool => return Err(ContainsBoolsError),
        })
    }
}

impl TryFrom<ir::SizedType> for SizedType {
    type Error = ContainsBoolsError;

    fn try_from(value: ir::SizedType) -> Result<Self, Self::Error> {
        Ok(match value {
            ir::SizedType::Vector(len, scalar) => SizedType::Vector(Vector {
                scalar: scalar.try_into()?,
                len,
            }),
            ir::SizedType::Matrix(columns, rows, scalar) => SizedType::Matrix(Matrix { scalar, columns, rows }),
            ir::SizedType::Array(element, len) => SizedType::Array(SizedArray {
                element: Rc::new((*element).clone().try_into()?),
                len,
            }),
            ir::SizedType::Atomic(scalar_type) => SizedType::Atomic(Atomic { scalar: scalar_type }),
            ir::SizedType::Structure(structure) => SizedType::Struct(structure.try_into()?),
        })
    }
}

impl TryFrom<ir::ir_type::SizedStruct> for SizedStruct {
    type Error = ContainsBoolsError;

    fn try_from(structure: ir::ir_type::SizedStruct) -> Result<Self, Self::Error> {
        let mut fields = Vec::new();

        for field in structure.sized_fields() {
            fields.push(SizedField {
                name: field.name.clone(),
                custom_min_size: field.custom_min_size,
                custom_min_align: field.custom_min_align,
                ty: field.ty.clone().try_into()?,
            });
        }

        Ok(SizedStruct {
            name: structure.name().clone(),
            fields,
        })
    }
}

impl From<ContainsBoolsError> for LayoutableConversionError {
    fn from(_: ContainsBoolsError) -> Self { Self::ContainsBool }
}

impl TryFrom<ir::StoreType> for LayoutableType {
    type Error = LayoutableConversionError;

    fn try_from(value: ir::StoreType) -> Result<Self, Self::Error> {
        Ok(match value {
            ir::StoreType::Sized(sized_type) => LayoutableType::Sized(sized_type.try_into()?),
            ir::StoreType::RuntimeSizedArray(element) => LayoutableType::RuntimeSizedArray(RuntimeSizedArray {
                element: element.try_into()?,
            }),
            ir::StoreType::BufferBlock(buffer_block) => buffer_block.try_into()?,
            ir::StoreType::Handle(_) => return Err(LayoutableConversionError::IsHandle),
        })
    }
}

impl TryFrom<ir::ir_type::BufferBlock> for LayoutableType {
    type Error = ContainsBoolsError;

    fn try_from(buffer_block: ir::ir_type::BufferBlock) -> Result<Self, Self::Error> {
        let mut sized_fields = Vec::new();

        for field in buffer_block.sized_fields() {
            sized_fields.push(SizedField {
                name: field.name.clone(),
                custom_min_size: field.custom_min_size,
                custom_min_align: field.custom_min_align,
                ty: field.ty.clone().try_into()?,
            });
        }

        let last_unsized = if let Some(last_field) = buffer_block.last_unsized_field() {
            RuntimeSizedArrayField {
                name: last_field.name.clone(),
                custom_min_align: last_field.custom_min_align,
                array: RuntimeSizedArray {
                    element: last_field.element_ty.clone().try_into()?,
                },
            }
        } else {
            return Ok(SizedStruct {
                name: buffer_block.name().clone(),
                fields: sized_fields,
            }
            .into());
        };

        Ok(UnsizedStruct {
            name: buffer_block.name().clone(),
            sized_fields,
            last_unsized,
        }
        .into())
    }
}


#[test]
fn test_ir_conversion_error() {
    use crate::{f32x1, packed::unorm8x2};

    let ty: LayoutableType = SizedStruct::new("A", "a", f32x1::layoutable_type_sized())
        .extend("b", f32x1::layoutable_type_sized())
        .extend("a", f32x1::layoutable_type_sized())
        .into();
    let result: Result<ir::StoreType, _> = ty.try_into();
    assert!(matches!(
        result,
        Err(IRConversionError::DuplicateFieldName(DuplicateFieldNameError {
            struct_type: StructKind::Sized(_),
            first_occurence: 0,
            second_occurence: 2,
            ..
        }))
    ));

    let ty: LayoutableType = SizedStruct::new("A", "a", unorm8x2::layoutable_type_sized()).into();
    let result: Result<ir::StoreType, _> = ty.try_into();
    assert!(matches!(result, Err(IRConversionError::ContainsPackedVector)));
}
