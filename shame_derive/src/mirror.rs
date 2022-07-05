use syn::PathSegment;
use syn::{Ident, Result, Token, Path, DataStruct, FieldsNamed, DeriveInput, punctuated::Punctuated};
use syn::parse::{Parse, ParseStream};
use proc_macro2::TokenStream as TokenStream2;
use proc_macro2::Span;
use quote::{quote, quote_spanned};
use super::util::*;

pub struct Args {
    pub name_ext: NameExtension,
    pub module_path: Option<syn::Path>,
}

pub enum NameExtension {
    Prefix(Ident),
    Postfix(Ident),
    Replace(Ident),
}

pub enum Kind {
    Host, Device,
}

impl Parse for Args {
    fn parse(args: ParseStream) -> Result<Self> {

        let pre_star: Option<Token![*]> = args.parse().ok();
        let name: syn::Ident = args.parse()?;
        let post_star: Option<Token![*]> = args.parse().ok();
        
        let name_ext = match (pre_star, post_star) {
            (None, None) => NameExtension::Replace(name),
            (None, Some(_)) => NameExtension::Prefix(name),
            (Some(_), None) => NameExtension::Postfix(name),
            (Some(_), Some(_)) => {
                return Err(syn::Error::new(args.span(), "only one asterisk allowed either before or after the identifier"))
            },
        };
        
        let maybe_comma: Option<Token![,]> = args.parse().ok();
        let module_path: Option<syn::Path> = maybe_comma.and_then(|_| args.parse().ok());

        Ok(Args {
            name_ext, module_path
        })
    }
}

fn make_mirror_type_ident(other: &Ident, ext: NameExtension) -> Ident {
    match ext {
        NameExtension:: Prefix(ext) => Ident::new(&format!("{ext}{other}"), Span::call_site()),
        NameExtension::Postfix(ext) => Ident::new(&format!("{other}{ext}"), Span::call_site()),
        NameExtension::Replace(ident) => ident,
    }
}

pub fn push_ident_onto_path(path: &Option<Path>, ident: Ident) -> Path {
    match path {
        None => Path::from(ident),
        Some(path) => {
            Path {
                leading_colon: path.leading_colon,
                segments: Punctuated::from_iter(
                    path.segments.iter().chain(std::iter::once(&PathSegment::from(ident))).cloned()
                )
            }
        }
    }
}

pub fn impl_for_struct(kind: Kind, args: Args, input: &DeriveInput, _struct_data: &DataStruct, fields: &FieldsNamed) -> TokenStream2 {
    let host_trait   = push_ident_onto_path(&args.module_path, Ident::new("Host", Span::call_site()));
    let device_trait = push_ident_onto_path(&args.module_path, Ident::new("Device", Span::call_site()));
    
    match kind {
        Kind::Host => {
            let host_type = &input.ident;
            let device_type = make_mirror_type_ident(host_type, args.name_ext);

            let host_fields = map_fields(fields, |span, ident, ty| quote_spanned! {span => 
                #ident: #ty
            });

            let device_fields = map_fields(fields, |span, ident, ty| quote_spanned! {span => 
                #ident: <#ty as #host_trait>::Device
            });

            quote!{
                #[repr(C)]
                #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
                struct #host_type {
                    #(#host_fields),*
                }
            
                #[derive(shame::Fields)]
                #[allow(non_camel_case_types)] 
                struct #device_type {
                    #(#device_fields),*
                }
            
                impl #device_trait for #device_type {
                    type Host = #host_type;
                }
            
                impl #host_trait for #host_type {
                    type Device = #device_type;
                    fn as_bytes(&self) ->  &[u8] {
                        bytemuck::cast_slice(std::slice::from_ref(self))
                    }
                }
            } //quote
        }
        Kind::Device => {
            let device_type = &input.ident;
            let host_type = make_mirror_type_ident(device_type, args.name_ext);
        
            let host_fields = map_fields(fields, |span, ident, ty| quote_spanned! {span => 
                #ident: <#ty as #device_trait>::Host
            });
        
            let device_fields = map_fields(fields, |span, ident, ty| quote_spanned! {span => 
                #ident: #ty
            });
        
            quote!{
                #[repr(C)]
                #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
                #[allow(non_camel_case_types)] 
                struct #host_type {
                    #(#host_fields),*
                }
            
                #[derive(shame::Fields)]
                struct #device_type {
                    #(#device_fields),*
                }
            
                impl #device_trait for #device_type {
                    type Host = #host_type;
                }
            
                impl #host_trait for #host_type {
                    type Device = #device_type;
                    fn as_bytes(&self) ->  &[u8] {
                        bytemuck::cast_slice(std::slice::from_ref(self))
                    }
                }
            } //quote
        }
    }

}