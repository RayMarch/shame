#![allow(non_snake_case, clippy::match_like_matches_macro)]
mod keep_idents;
mod derive_fields;
mod mirror;

use keep_idents::*;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{ItemFn, parse_macro_input, parse_quote};
use syn::fold::Fold;
use syn::Error;

use syn::spanned::*;

mod util;

#[proc_macro_attribute]
pub fn keep_idents(_args: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemFn);

    // Use a syntax tree traversal to transform the function body.
    let mut output = State.fold_item_fn(input);
    output.block.stmts.push(
        parse_quote!(use shame_reexports::shame::keep_idents::TryKeepIdentTrait;)
    );
    // Hand the resulting function body back to the compiler.
    TokenStream::from(quote!(#output))
}

#[proc_macro_derive(Fields)]
pub fn derive_fields(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let r = generate_fields_tokens(input);

    r.unwrap_or_else(|err| {
        err.to_compile_error()
    }).into()
}

fn generate_fields_tokens(input: syn::DeriveInput) -> Result<TokenStream2, Error> {
    match util::parse(&input) {
        util::DeriveData::Struct(input, struct_data, named_fields)
            => Ok(derive_fields::impl_for_struct(input, struct_data, named_fields)),
        _ => Err(syn::Error::new(input.span(), "Must be used on a struct with named fields")),
    }
}

#[cfg(feature = "mirror")]
#[proc_macro_attribute]
#[allow(unused)]
pub fn host(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as mirror::Args);
    let as_derive = parse_macro_input!(input as syn::DeriveInput);
    let r = generate_mirror_tokens(mirror::Kind::Host, args, as_derive);

    r.unwrap_or_else(|err| err.to_compile_error()).into()
}

#[cfg(feature = "mirror")]
#[proc_macro_attribute]
#[allow(unused)]
pub fn device(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as mirror::Args);
    let as_derive = parse_macro_input!(input as syn::DeriveInput);
    let r = generate_mirror_tokens(mirror::Kind::Device, args, as_derive);

    r.unwrap_or_else(|err| err.to_compile_error()).into()
}

fn generate_mirror_tokens(kind: mirror::Kind, args: mirror::Args, input: syn::DeriveInput) -> Result<TokenStream2, Error> {
    match util::parse(&input) {
        util::DeriveData::Struct(input, struct_data, named_fields)
            => Ok(mirror::impl_for_struct(kind, args, input, struct_data, named_fields)),
        _ => Err(syn::Error::new(input.span(), "Must be used on a struct with named fields")),
    }
}