use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, quote_spanned};
use syn::{DataStruct, DeriveInput, FieldsNamed};

use crate::util::map_fields;

pub fn impl_for_struct(
    input: &DeriveInput,
    _struct_data: &DataStruct,
    fields: &FieldsNamed,
) -> TokenStream2 {
    let derive_ty = &input.ident;

    let generics_decl = &input.generics.params; //<A: Trait, B: Trait>
    let where_clause = &input.generics.where_clause; //(A, B): Trait, C: Trait

    let generic_args = input.generics.params.iter().map(|params| {
        //<A, B>
        match params {
            syn::GenericParam::Type(x) => &x.ident,
            syn::GenericParam::Lifetime(x) => &x.lifetime.ident,
            syn::GenericParam::Const(x) => &x.ident,
        }
    });
    let generic_args2 = generic_args.clone();

    let shame = quote!(shame);
    let shame_graph = quote!(#shame::shame_reexports::shame_graph);
    let Fields = quote!(#shame::rec::fields::Fields);
    let Rec = quote!(#shame::rec::Rec);
    let IntoRec = quote!(#shame::rec::IntoRec);
    let Struct = quote!(#shame::rec::struct_::Struct);
    let Stage = quote!(#shame::rec::Stage);
    let Ty = quote!(#shame_graph::Ty);
    let Any = quote!(#shame_graph::Any);

    let fields_init = map_fields(fields, |span, ident, _| {
        quote_spanned! {span =>
            #ident: #Fields::from_fields_downcast(Some(std::stringify!(#ident)), f)
        }
    });

    let fields_vec_extend = map_fields(fields, |span, ident, _| {
        quote_spanned! {span =>
            vec.extend(self.#ident.collect_fields());
        }
    });

    quote! {
        impl<#generics_decl> #Fields for #derive_ty<#(#generic_args),*> where #where_clause {

            fn parent_type_name() -> Option<&'static str> {
                Some(std::stringify!(#derive_ty))
            }

            fn from_fields_downcast(name: Option<&'static str>, f: &mut impl FnMut(#Ty, &'static str) -> (#Any, #Stage)) -> Self {
                Self {
                    #(#fields_init),*
                }
            }

            fn collect_fields(&self) -> Vec<(#Any, #Stage)> {
                let mut vec = Vec::new();
                #(#fields_vec_extend)*
                vec
            }
        }

        impl<#generics_decl> #IntoRec for #derive_ty<#(#generic_args2),*> where #where_clause {
            type Rec = #Struct<Self>;

            fn rec(self) -> Self::Rec {
                #Struct::<Self>::new(self)
            }

            fn into_any(self) -> #Any {
                #Rec::as_any(&self.rec())
            }

            fn stage(&self) -> #Stage {
                #shame::rec::narrow_stages_or_push_error(
                    #Fields::collect_fields(self).into_iter().map(|(_, stage)| stage)
                )
            }
        }
    } //quote!
}
