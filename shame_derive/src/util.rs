use proc_macro2::Span;
use syn::{*, spanned::Spanned};
use syn::{DataStruct, DeriveInput, FieldsNamed};

#[allow(unused)]
pub fn map_fields<'a, R>(fields: &'a FieldsNamed, mut func: impl 'a + FnMut(Span, &Option<Ident>, &Type) -> R) 
-> impl Iterator<Item=R> + 'a {
    fields.named.iter().map(move |f| {
        func(f.span(), &f.ident, &f.ty)
    })
}

#[allow(unused)]
pub fn map_fields_enumerate<'a, R>(fields: &'a FieldsNamed, mut func: impl 'a + FnMut(usize, Span, &Option<Ident>, &Type) -> R) 
-> impl Iterator<Item=R> + 'a {
    fields.named.iter().enumerate().map(move |(i, f)| {
        func(i, f.span(), &f.ident, &f.ty)
    })
}

pub enum DeriveData<'a> {
    /// A struct with named fields
    Struct(&'a DeriveInput, &'a DataStruct, &'a FieldsNamed),
    /// A tuple struct
    Tuple(&'a DeriveInput, &'a DataStruct, &'a FieldsUnnamed),
    /// A unit struct
    Unit(&'a DeriveInput, &'a DataStruct),
    /// An enum
    Enum(&'a DeriveInput, &'a DataEnum),
    /// A nondiscriminant union
    Union(&'a DeriveInput, &'a DataUnion),
}

#[allow(unused)]
pub enum VariantData<'a> {
    /// A discriminant variant e.g. `Foo = 1`
    Discriminant(/*index*/ usize, &'a Variant, &'a Expr),
    /// A variant with named fields similar to a struct
    NamedFields(/*index*/ usize, &'a Variant, &'a FieldsNamed),
    /// A variant with unnamed fields similar to a tuple struct
    UnnamedFields(/*index*/ usize, &'a Variant, &'a FieldsUnnamed),
    /// A variant without fields similar to a unit struct
    Unit(/*index*/ usize, &'a Variant),
}

pub fn parse(input: & DeriveInput) -> DeriveData {
    match &input.data {
        Data::Struct(s) => {
            match &s.fields {
                Fields::Named(n) => DeriveData::Struct(input, s, n),
                Fields::Unnamed(u) => DeriveData::Tuple(input, s, u),
                Fields::Unit => DeriveData::Unit(input, s),
            }
        },
        Data::Enum(e) => DeriveData::Enum(input, e),
        Data::Union(u) => DeriveData::Union(input, u),
    }
}

#[allow(unused)]
pub fn parse_variants(input: &DataEnum) -> impl Iterator<Item=VariantData> {
    input.variants.iter().enumerate().map(|(index, variant)| {
        match variant {
            Variant { discriminant: Some((_, expression)), .. } => VariantData::Discriminant(index, variant, expression),
            Variant { fields: Fields::Named(named), .. } => VariantData::NamedFields(index, variant, named),
            Variant { fields: Fields::Unnamed(unnamed), ..} => VariantData::UnnamedFields(index, variant, unnamed),
            variant => VariantData::Unit(index, variant),
        }
    })
}