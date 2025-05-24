use super::*;

//     Conversions to ir types     //

/// Errors that can occur when converting IR types to layoutable types.
#[derive(thiserror::Error, Debug)]
pub enum IRConversionError {
    /// Packed vectors do not exist in the shader type system.
    #[error("{0}")]
    ContainsPackedVector(#[from] ContainsPackedVectorError),
    /// Struct field names must be unique in the shader type system.
    #[error("{0}")]
    DuplicateFieldName(#[from] DuplicateFieldNameError),
}

#[derive(Debug)]
pub enum ContainsPackedVectorError {
    /// Top level type is a packed vector.
    SelfIsPackedVector(LayoutableType),
    /// Top level nested Array packed vector, for example Array<Array<Array<unorm8x2>>>.
    InArray(LayoutableType),
    /// A struct field is a packed vector or a struct field is a nested Array packed vector.
    InStruct {
        struct_type: StructKind,
        packed_vector_field: usize,
        top_level_type: Option<LayoutableType>,
        use_color: bool,
    },
}

impl std::error::Error for ContainsPackedVectorError {}

impl std::fmt::Display for ContainsPackedVectorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ending =
            ", which does not exist in the shader type system.\nPacked vectors may only be used in vertex buffers.";
        let (struct_name, sized_fields, last_unsized, packed_vector_field, use_color) = match self {
            ContainsPackedVectorError::SelfIsPackedVector(t) => return writeln!(f, "{t} is a packed vector{ending}"),
            ContainsPackedVectorError::InArray(a) => return writeln!(f, "{a} contains a packed vector{ending}"),
            ContainsPackedVectorError::InStruct {
                struct_type,
                packed_vector_field,
                top_level_type,
                use_color,
            } => {
                let (struct_name, sized_fields, last_unsized) = match struct_type {
                    StructKind::Sized(s) => (&s.name, s.fields(), None),
                    StructKind::Unsized(s) => (&s.name, s.sized_fields.as_slice(), Some(&s.last_unsized)),
                };

                if let Some(top_level_type) = top_level_type {
                    writeln!(
                        f,
                        "The struct {struct_name} in {top_level_type} contains a packed vector{ending}"
                    )?;
                } else {
                    writeln!(f, "{struct_name} contains a packed vector{ending}")?;
                }

                (
                    struct_name,
                    sized_fields,
                    last_unsized,
                    *packed_vector_field,
                    *use_color,
                )
            }
        };



        let indent = "  ";
        let is_packed_vec = |i| packed_vector_field == i;
        let arrow = |i| match is_packed_vec(i) {
            true => " <--",
            false => "",
        };
        let color = |f: &mut Formatter<'_>, i| {
            if (use_color && is_packed_vec(i)) {
                set_color(f, Some("#508EE3"), false)
            } else {
                Ok(())
            }
        };
        let color_reset = |f: &mut Formatter<'_>, i| {
            if use_color && is_packed_vec(i) {
                set_color(f, None, false)
            } else {
                Ok(())
            }
        };

        writeln!(f, "The following struct contains a packed vector:")?;
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

#[derive(Debug)]
pub struct DuplicateFieldNameError {
    pub struct_type: StructKind,
    pub first_field: usize,
    pub second_field: usize,
    pub is_top_level: bool,
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
        let is_duplicate = |i| self.first_field == i || self.second_field == i;
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
    struct_type: StructKind,
    is_top_level: bool,
) -> Result<StructKind, DuplicateFieldNameError> {
    let (sized_fields, last_unsized) = match &struct_type {
        StructKind::Sized(s) => (s.fields(), None),
        StructKind::Unsized(s) => (s.sized_fields.as_slice(), Some(&s.last_unsized)),
    };

    // Brute force search > HashMap for the amount of fields
    // we'd usually deal with.
    let mut duplicate_fields = None;
    for (i, field1) in sized_fields.iter().enumerate() {
        for (j, field2) in sized_fields[i..].iter().enumerate() {
            if field1.name == field2.name {
                duplicate_fields = Some((i, j));
                break;
            }
        }
    }

    match duplicate_fields {
        Some((first_field, second_field)) => Err(DuplicateFieldNameError {
            struct_type,
            first_field,
            second_field,
            is_top_level,
            use_color: use_color(),
        }),
        None => Ok(struct_type),
    }
}

#[track_caller]
fn use_color() -> bool { Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages).unwrap_or(false) }

impl TryFrom<LayoutableType> for ir::StoreType {
    type Error = IRConversionError;

    fn try_from(ty: LayoutableType) -> Result<Self, Self::Error> {
        // Checking for top level packed vec.
        if matches!(ty, LayoutableType::Sized(SizedType::PackedVec(_))) {
            return Err(IRConversionError::ContainsPackedVector(
                ContainsPackedVectorError::SelfIsPackedVector(ty),
            ));
        }
        // Checking for nested array packed vec.
        if is_packed_vec(&ty) {
            return Err(IRConversionError::ContainsPackedVector(
                ContainsPackedVectorError::InArray(ty),
            ));
        }

        fn convert

        todo!()
    }
}

/// Returns true if `ty` is a packed vector or a nesting of Arrays
/// with the last Array directly containing a packed vector.
fn is_packed_vec(ty: &LayoutableType) -> bool {
    let mut ty = match ty {
        LayoutableType::Sized(s) => s,
        LayoutableType::RuntimeSizedArray(a) => &a.element,
        LayoutableType::UnsizedStruct(_) => return false,
    };

    loop {
        match ty {
            SizedType::PackedVec(_) => return true,
            SizedType::Array(a) => ty = &a.element,
            _ => return false,
        }
    }
}

// impl TryFrom<SizedType> for ir::SizedType {
//     type Error = IRConversionError;

//     fn try_from(host: SizedType) -> Result<Self, Self::Error> {
//         // Ok(match host {
//         //     SizedType::Vector(v) => ir::SizedType::Vector(v.len, v.scalar.into()),
//         //     SizedType::Matrix(m) => ir::SizedType::Matrix(m.columns, m.rows, m.scalar),
//         //     SizedType::Array(a) => {
//         //         let element = Rc::unwrap_or_clone(a.element);
//         //         let converted_element = element.try_into()?;
//         //         ir::SizedType::Array(Rc::new(converted_element), a.len)
//         //     }
//         //     SizedType::Atomic(i) => ir::SizedType::Atomic(i.scalar),
//         //     SizedType::PackedVec(_) => return Err(IRConversionError::ContainsPackedVector),
//         //     SizedType::Struct(s) => ir::SizedType::Structure(s.try_into()?),
//         // })
//         todo!()
//     }
// }

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

// impl TryFrom<SizedStruct> for ir::ir_type::SizedStruct {
//     type Error = IRConversionError;

//     fn try_from(host: SizedStruct) -> Result<Self, Self::Error> {
//         let mut fields: Vec<ir::ir_type::SizedField> = Vec::new();

//         for field in host.fields {
//             fields.push(field.try_into()?);
//         }

//         // has at least one field
//         if fields.is_empty() {
//             // This should not happen based on SizedStruct's implementation
//             return Err(IRConversionError::DuplicateFieldName);
//         }

//         let last_field = fields.pop().unwrap();

//         match ir::ir_type::SizedStruct::new_nonempty(host.name, fields, last_field) {
//             Ok(s) => Ok(s),
//             Err(StructureFieldNamesMustBeUnique) => Err(IRConversionError::DuplicateFieldName),
//         }
//     }
// }

// impl TryFrom<UnsizedStruct> for ir::ir_type::BufferBlock {
//     type Error = IRConversionError;

//     fn try_from(host: UnsizedStruct) -> Result<Self, Self::Error> {
//         let mut sized_fields: Vec<ir::ir_type::SizedField> = Vec::new();

//         for field in host.sized_fields {
//             sized_fields.push(field.try_into()?);
//         }

//         let last_unsized = host.last_unsized.try_into()?;

//         match ir::ir_type::BufferBlock::new(host.name, sized_fields, Some(last_unsized)) {
//             Ok(b) => Ok(b),
//             Err(BufferBlockDefinitionError::FieldNamesMustBeUnique) => Err(IRConversionError::DuplicateFieldName),
//             Err(BufferBlockDefinitionError::MustHaveAtLeastOneField) => {
//                 // This should not happen based on UnsizedStruct's implementation
//                 Err(IRConversionError::DuplicateFieldName)
//             }
//         }
//     }
// }

// impl TryFrom<RuntimeSizedArray> for ir::StoreType {
//     type Error = IRConversionError;

//     fn try_from(array: RuntimeSizedArray) -> Result<Self, Self::Error> {
//         Ok(ir::StoreType::RuntimeSizedArray(array.element.try_into()?))
//     }
// }

// impl TryFrom<SizedField> for ir::ir_type::SizedField {
//     type Error = IRConversionError;

//     fn try_from(f: SizedField) -> Result<Self, Self::Error> {
//         Ok(ir::SizedField::new(
//             f.name,
//             f.custom_min_size,
//             f.custom_min_align,
//             f.ty.try_into()?,
//         ))
//     }
// }

// impl TryFrom<RuntimeSizedArrayField> for ir::ir_type::RuntimeSizedArrayField {
//     type Error = IRConversionError;

//     fn try_from(f: RuntimeSizedArrayField) -> Result<Self, Self::Error> {
//         Ok(ir::RuntimeSizedArrayField::new(
//             f.name,
//             f.custom_min_align,
//             f.array.element.try_into()?,
//         ))
//     }
// }


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
