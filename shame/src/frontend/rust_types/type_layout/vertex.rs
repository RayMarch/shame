use crate::{any::VertexAttribFormat, cpu_shareable::Repr};
use super::{*};

impl TypeLayout<constraint::Vertex> {
    pub fn from_vertex_attribute(attribute: VertexAttribFormat) -> TypeLayout<constraint::Vertex> {
        let (size, align, kind) = match attribute {
            VertexAttribFormat::Fine(len, scalar) => {
                let sized = cpu_shareable::Vector::new(scalar, len);
                (sized.byte_size(), sized.align(), TypeLayoutSemantics::Vector(sized))
            }
            VertexAttribFormat::Coarse(packed) => (
                u8::from(packed.byte_size()) as u64,
                U32PowerOf2::try_from(packed.align() as u32).unwrap(),
                TypeLayoutSemantics::PackedVector(packed),
            ),
        };

        type_layout_internal::cast_unchecked(TypeLayout::new(Some(size), align, kind, None))
    }

    /// Creates a new builder for a `TypeLayout<constraint::Vectex>`. Takes the first
    /// attribute immediately, because at least one attribut needs to exist.
    ///
    /// `rules` determines whether the layout is packed or not. The `Uniform` and `Storage`
    /// are equivalent for vertex layouts.
    pub fn vertex_builder(
        struct_name: impl Into<CanonName>,
        field_options: impl Into<FieldOptions>,
        attribute: VertexAttribFormat,
        rules: Repr,
    ) -> VertexLayoutBuilder {
        VertexLayoutBuilder::new(struct_name, field_options, attribute, rules)
    }
}

pub struct VertexLayoutBuilder {
    name: CanonName,
    attributes: Vec<FieldLayout>,
    rules: Repr,
}

impl VertexLayoutBuilder {
    /// Creates a new builder for a `TypeLayout<constraint::Vectex>`. Takes the first
    /// attribute immediately, because at least one attribut needs to exist.
    ///
    /// `rules` determines whether the layout is packed or not. The `Uniform` and `Storage`
    /// are equivalent for vertex layouts.
    pub fn new(
        struct_name: impl Into<CanonName>,
        field_options: impl Into<FieldOptions>,
        attribute: VertexAttribFormat,
        rules: Repr,
    ) -> Self {
        let this = VertexLayoutBuilder {
            name: struct_name.into(),
            attributes: Vec::new(),
            rules,
        };
        this.extend(field_options, attribute)
    }

    pub fn extend(mut self, field_options: impl Into<FieldOptions>, attribute: VertexAttribFormat) -> Self {
        let layout = TypeLayout::from_vertex_attribute(attribute);
        let options = field_options.into();
        self.attributes.push(FieldLayout::new(
            options.name,
            options.custom_min_size,
            options.custom_min_align,
            layout.into(),
        ));
        self
    }

    pub fn finish(self) -> TypeLayout<constraint::Vertex> {
        let mut calc = LayoutCalculator::new(matches!(self.rules, Repr::Packed));
        let fields = self
            .attributes
            .into_iter()
            .map(|field| {
                let rel_byte_offset = calc.extend(
                    field.byte_size().unwrap(), // attributes are always sized
                    field.byte_align(),
                    *field.custom_min_size,
                    *field.custom_min_align,
                );
                FieldLayoutWithOffset { field, rel_byte_offset }
            })
            .collect::<Vec<_>>();

        type_layout_internal::cast_unchecked(TypeLayout::new(
            Some(calc.byte_size()),
            calc.align(),
            TypeLayoutSemantics::Structure(Rc::new(StructLayout {
                name: self.name.into(),
                fields,
            })),
            None,
        ))
    }
}
