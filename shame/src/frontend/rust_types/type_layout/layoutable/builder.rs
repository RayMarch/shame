use super::*;

impl LayoutableType {
    /// Fallibly creates a new `LayoutableType` of a struct.
    ///
    /// An error is returned if the following rules aren't followed:
    /// - There must be at least one field.
    /// - None of the fields must be an `UnsizedStruct`.
    /// - Only the last field may be unsized (a runtime sized array).
    pub fn struct_from_parts(
        struct_name: impl Into<CanonName>,
        fields: impl IntoIterator<Item = (FieldOptions, LayoutableType)>,
    ) -> Result<Self, StructFromPartsError> {
        use StructFromPartsError::*;

        enum Field {
            Sized(SizedField),
            Unsized(RuntimeSizedArrayField),
        }

        let mut fields = fields
            .into_iter()
            .map(|(options, ty)| {
                Ok(match ty {
                    LayoutableType::Sized(s) => Field::Sized(SizedField::new(options, s)),
                    LayoutableType::RuntimeSizedArray(a) => Field::Unsized(RuntimeSizedArrayField::new(
                        options.name,
                        options.custom_min_align,
                        a.element,
                    )),
                    LayoutableType::UnsizedStruct(_) => return Err(MustNotHaveUnsizedStructField),
                })
            })
            .peekable();

        let mut sized_fields = Vec::new();
        let mut last_unsized = None;
        while let Some(field) = fields.next() {
            let field = field?;
            match field {
                Field::Sized(sized) => sized_fields.push(sized),
                Field::Unsized(a) => {
                    last_unsized = Some(a);
                    if fields.peek().is_some() {
                        return Err(OnlyLastFieldMayBeUnsized);
                    }
                }
            }
        }

        let field_count = sized_fields.len() + last_unsized.is_some() as usize;
        if field_count == 0 {
            return Err(MustHaveAtLeastOneField);
        }

        if let Some(last_unsized) = last_unsized {
            Ok(UnsizedStruct {
                name: struct_name.into(),
                sized_fields,
                last_unsized,
            }
            .into())
        } else {
            Ok(SizedStruct::from_parts(struct_name, sized_fields).into())
        }
    }
}

#[allow(missing_docs)]
#[derive(thiserror::Error, Debug)]
pub enum StructFromPartsError {
    #[error("Struct must have at least one field.")]
    MustHaveAtLeastOneField,
    #[error("Only the last field of a struct may be unsized.")]
    OnlyLastFieldMayBeUnsized,
    #[error("A field of the struct is an unsized struct, which isn't allowed.")]
    MustNotHaveUnsizedStructField,
}

impl SizedStruct {
    /// Creates a new `SizedStruct` with one field.
    ///
    /// To add additional fields to it, use [`SizedStruct::extend`] or [`SizedStruct::extend_unsized`].
    pub fn new(name: impl Into<CanonName>, field_options: impl Into<FieldOptions>, ty: impl Into<SizedType>) -> Self {
        Self {
            name: name.into(),
            fields: vec![SizedField::new(field_options, ty)],
        }
    }

    /// Adds a sized field to the struct.
    pub fn extend(mut self, field_options: impl Into<FieldOptions>, ty: impl Into<SizedType>) -> Self {
        self.fields.push(SizedField::new(field_options, ty));
        self
    }

    /// Adds a runtime sized array field to the struct. This can only be the last
    /// field of a struct, which is ensured by transitioning to an UnsizedStruct.
    pub fn extend_unsized(
        self,
        name: impl Into<CanonName>,
        custom_min_align: Option<U32PowerOf2>,
        element_ty: impl Into<SizedType>,
    ) -> UnsizedStruct {
        UnsizedStruct {
            name: self.name,
            sized_fields: self.fields,
            last_unsized: RuntimeSizedArrayField::new(name, custom_min_align, element_ty),
        }
    }

    /// Adds either a `SizedType` or a `RuntimeSizedArray` field to the struct.
    ///
    /// Returns a `LayoutableType`, because the `Self` may either stay
    /// a `SizedStruct` or become an `UnsizedStruct` depending on the field's type.
    pub fn extend_sized_or_array(self, field_options: impl Into<FieldOptions>, field: SizedOrArray) -> LayoutableType {
        let options = field_options.into();
        match field {
            SizedOrArray::Sized(ty) => self.extend(options, ty).into(),
            SizedOrArray::RuntimeSizedArray(a) => self
                .extend_unsized(options.name, options.custom_min_align, a.element)
                .into(),
        }
    }

    /// The fields of this struct.
    pub fn fields(&self) -> &[SizedField] { &self.fields }

    pub(crate) fn from_parts(name: impl Into<CanonName>, fields: Vec<SizedField>) -> Self {
        Self {
            name: name.into(),
            fields,
        }
    }
}

#[allow(missing_docs)]
pub enum SizedOrArray {
    Sized(SizedType),
    RuntimeSizedArray(RuntimeSizedArray),
}

#[allow(missing_docs)]
#[derive(thiserror::Error, Debug)]
#[error("`LayoutType` is `UnsizedStruct`, which is not a variant of `SizedOrArray`")]
pub struct IsUnsizedStructError;
impl TryFrom<LayoutableType> for SizedOrArray {
    type Error = IsUnsizedStructError;

    fn try_from(value: LayoutableType) -> Result<Self, Self::Error> {
        match value {
            LayoutableType::Sized(sized) => Ok(SizedOrArray::Sized(sized)),
            LayoutableType::RuntimeSizedArray(array) => Ok(SizedOrArray::RuntimeSizedArray(array)),
            LayoutableType::UnsizedStruct(_) => Err(IsUnsizedStructError),
        }
    }
}

impl SizedField {
    /// Creates a new `SizedField`.
    pub fn new(options: impl Into<FieldOptions>, ty: impl Into<SizedType>) -> Self {
        let options = options.into();
        Self {
            name: options.name,
            custom_min_size: options.custom_min_size,
            custom_min_align: options.custom_min_align,
            ty: ty.into(),
        }
    }
}

impl RuntimeSizedArrayField {
    /// Creates a new `RuntimeSizedArrayField` given it's field name,
    /// an optional custom minimum align and it's element type.
    pub fn new(
        name: impl Into<CanonName>,
        custom_min_align: Option<U32PowerOf2>,
        element_ty: impl Into<SizedType>,
    ) -> Self {
        Self {
            name: name.into(),
            custom_min_align,
            array: RuntimeSizedArray {
                element: element_ty.into(),
            },
        }
    }
}

impl SizedArray {
    /// Creates a new `SizedArray` from it's element type and length.
    pub fn new(element_ty: Rc<SizedType>, len: NonZeroU32) -> Self {
        Self {
            element: element_ty,
            len,
        }
    }
}

impl RuntimeSizedArray {
    /// Creates a new `RuntimeSizedArray` from it's element type.
    pub fn new(element_ty: impl Into<SizedType>) -> Self {
        RuntimeSizedArray {
            element: element_ty.into(),
        }
    }
}

/// Options for the field of a struct.
///
/// If you only want to customize the field's name, you can convert most string types
/// to `FieldOptions` using `Into::into`. For most methods that take `impl Into<FieldOptions>`
/// parameters you can just pass the string type directly.  
#[derive(Debug, Clone)]
pub struct FieldOptions {
    /// Name of the field
    pub name: CanonName,
    /// Custom minimum align of the field.
    pub custom_min_align: Option<U32PowerOf2>,
    /// Custom mininum size of the field.
    pub custom_min_size: Option<u64>,
}

impl FieldOptions {
    /// Creates new `FieldOptions`.
    ///
    /// If you only want to customize the field's name, you can convert most string types
    /// to `FieldOptions` using `Into::into`, but most methods take `impl Into<StructOptions>`,
    /// meaning you can just pass the string type directly.
    pub fn new(
        name: impl Into<CanonName>,
        custom_min_align: Option<U32PowerOf2>,
        custom_min_size: Option<u64>,
    ) -> Self {
        Self {
            name: name.into(),
            custom_min_align,
            custom_min_size,
        }
    }
}

impl<T: Into<CanonName>> From<T> for FieldOptions {
    fn from(name: T) -> Self { Self::new(name, None, None) }
}
