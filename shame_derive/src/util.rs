use proc_macro2::Span;
use proc_macro2::TokenStream as TokenStream2;
use punctuated::Punctuated;
use quote::quote;
use syn::{spanned::Spanned, *};
use token::Comma;

/// tries to find `#[ident(literal)]`
///
/// returns
/// - `Ok(None)` if an attribute with ident `ident` wasn't found among attribs
/// - `Err(_)` if an attribute with ident `ident` was found, but doesn't have form `ident(literal)`
/// - `Ok(Some(Literal))` otherwise
pub fn find_literal_list_attr<T: syn::parse::Parse>(
    ident: &str,
    attribs: &[syn::Attribute],
) -> Result<Option<(Span, T)>> {
    for a in attribs {
        return Ok(match a.path().is_ident(ident) {
            false => continue,
            true => {
                let list = a.meta.require_list()?;
                Some((list.span(), list.parse_args::<T>()?))
            }
        });
    }
    Ok(None)
}

pub enum Repr {
    Packed,
    Storage,
    Uniform,
}

pub fn determine_gpu_repr(attribs: &[syn::Attribute]) -> Result<Option<(Span, Repr)>> {
    let mut repr = Repr::Storage;
    for a in attribs {
        if a.path().is_ident("gpu_repr") {
            a.parse_nested_meta(|meta| {
                if meta.path.is_ident("packed") {
                    repr = Repr::Packed;
                    return Ok(());
                } else if meta.path.is_ident("storage") {
                    repr = Repr::Storage;
                    return Ok(());
                } else if meta.path.is_ident("uniform") {
                    repr = Repr::Uniform;
                    return Ok(());
                }

                Err(meta.error("unrecognized `gpu_repr`. Did you mean `gpu_repr(packed)`?"))
            })?;

            return Ok(Some((a.span(), repr)));
        }
    }
    Ok(None)
}

pub struct Generics<'a> {
    /// <A: Trait, B: Trait>
    pub decl: &'a Punctuated<GenericParam, Comma>,
    /// (A, B): Trait, C: Trait
    pub where_clause_predicates: Option<&'a Punctuated<WherePredicate, Comma>>,
    pub idents: Vec<TokenStream2>,
}

impl<'a> Generics<'a> {
    pub fn from_input(input: &'a DeriveInput) -> Self {
        let generics_decl = &input.generics.params;
        let where_clause_predicates = input.generics.where_clause.as_ref().map(|wc| &wc.predicates);
        let idents_of_generics: Vec<TokenStream2> = input
            .generics
            .params
            .iter()
            .map(|param| match param {
                syn::GenericParam::Type(syn::TypeParam { ident, .. }) |
                syn::GenericParam::Const(syn::ConstParam { ident, .. }) => quote!(#ident),
                syn::GenericParam::Lifetime(syn::LifetimeParam { lifetime, .. }) => quote!(#lifetime),
            })
            .collect();

        Self {
            decl: generics_decl,
            where_clause_predicates,
            idents: idents_of_generics,
        }
    }
}

#[allow(unused)]
pub enum ReprAttr {
    Other {
        repr_c: bool,
        align: Option<usize>,
        packed: Option<usize>,
    },
    Transparent,
}

/// modified from `syn` docs example
pub fn try_parse_repr(attribs: &[syn::Attribute]) -> Result<Option<(Span, ReprAttr)>> {
    use syn::{parenthesized, token, LitInt};

    let mut repr_c = false;
    let mut repr_transparent = false;
    let mut repr_align = None::<usize>;
    let mut repr_packed = None::<usize>;
    let mut span = None::<Span>;

    for attr in attribs {
        if attr.path().is_ident("repr") {
            span = Some(attr.span());
            attr.parse_nested_meta(|meta| {
                // #[repr(C)]
                if meta.path.is_ident("C") {
                    repr_c = true;
                    return Ok(());
                }

                // #[repr(transparent)]
                if meta.path.is_ident("transparent") {
                    repr_transparent = true;
                    return Ok(());
                }

                // #[repr(align(N))]
                if meta.path.is_ident("align") {
                    let content;
                    parenthesized!(content in meta.input);
                    let lit: LitInt = content.parse()?;
                    let n: usize = lit.base10_parse()?;
                    repr_align = Some(n);
                    return Ok(());
                }

                // #[repr(packed)] or #[repr(packed(N))], omitted N means 1
                if meta.path.is_ident("packed") {
                    if meta.input.peek(token::Paren) {
                        let content;
                        parenthesized!(content in meta.input);
                        let lit: LitInt = content.parse()?;
                        let n: usize = lit.base10_parse()?;
                        repr_packed = Some(n);
                    } else {
                        repr_packed = Some(1);
                    }
                    return Ok(());
                }

                Err(meta.error("unrecognized repr"))
            })?;
        }
    }

    match span {
        None => Ok(None),
        Some(span) => Ok(match repr_transparent {
            true => Some((span, ReprAttr::Transparent)),
            false => Some((
                span,
                ReprAttr::Other {
                    repr_c,
                    align: repr_align,
                    packed: repr_packed,
                },
            )),
        }),
    }
}
