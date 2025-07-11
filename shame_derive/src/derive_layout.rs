use proc_macro2::Span;
use proc_macro2::TokenStream as TokenStream2;
use quote::format_ident;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::Field;
use syn::LitInt;
use syn::{DataStruct, DeriveInput, FieldsNamed};

use crate::util;
use crate::util::Repr;

macro_rules! bail {
    ($span: expr, $display: expr) => {return Err(syn::Error::new($span, $display,))};
}

#[derive(Debug)]
pub enum WhichDerive {
    GpuLayout,
    CpuLayout,
}

pub fn impl_for_struct(
    which_derive: WhichDerive,
    input: &DeriveInput,
    span: &Span,
    _data_struct: &DataStruct,
    fields: &FieldsNamed,
) -> Result<TokenStream2, syn::Error> {
    let derive_struct_ident = &input.ident;
    let vis = &input.vis;
    let derive_struct_ref_ident = format_ident!("{derive_struct_ident}_ref");

    let re: TokenStream2 = quote!(shame::__private::proc_macro_reexports);

    if fields.named.is_empty() {
        return Err(syn::Error::new_spanned(
            fields,
            format!("`derive({which_derive:?})` does not support empty structs"),
        ));
    }
    let num_fields = fields.named.len();

    // TODO(release) test all the different cases of generics and bounds
    let util::Generics {
        decl: generics_decl,
        where_clause_predicates,
        idents: idents_of_generics,
    } = util::Generics::from_input(input);

    if let Some(first) = idents_of_generics.first() {
        bail!(
            first.span(),
            format!("`derive({which_derive:?})` currently does not support generics")
        )
    }

    if let Some(where_) = where_clause_predicates {
        bail!(
            where_.span(),
            format!("`derive({which_derive:?})` currently does not support where clauses")
        )
    }



    // we do lots of `Vec<_>` collection here, because `&Vec<_>` is copy and supports `quote` repetition,
    // if we find another way of getting this without requiring `collect` replace all the vecs in here.
    let field_vec = |f: fn(&Field) -> _| fields.named.iter().map(f).collect::<Vec<_>>();

    // &vecs for repetitions
    let field_vis = &field_vec(|f @ Field { vis, .. }| quote_spanned!(f.span() => #vis));
    let field_ident = &field_vec(|f @ Field { ident, .. }| quote_spanned!(f.span() => #ident));
    let field_type = &field_vec(|f @ Field { ty, .. }| quote_spanned!(f.span() => #ty   ));

    // parse/validate attributes
    // #[cpu(T)]
    let cpu_attr = util::find_literal_list_attr::<syn::Type>("cpu", &input.attrs)?;
    let cpu_equivalent_type = cpu_attr
        .clone()
        .map(|(span, ty)| quote_spanned! { span => #ty })
        .into_iter();
    let none_if_no_cpu_equivalent_type = cpu_attr.is_none().then_some(quote! { None }).into_iter();

    // #[gpu_repr(packed | storage)]
    let gpu_repr = util::try_find_gpu_repr(&input.attrs)?;
    if let (Some((span, _)), WhichDerive::CpuLayout) = (&gpu_repr, &which_derive) {
        bail!(*span, "`gpu_repr` attribute is only supported by `derive(GpuLayout)`")
    }
    // if no `#[gpu_repr(_)]` attribute was explicitly specified, we default to `Repr::Storage`
    let gpu_repr = gpu_repr.map(|(_, repr)| repr).unwrap_or(util::Repr::Storage);
    let gpu_repr_shame = match gpu_repr {
        Repr::Packed => quote!( #re::repr::Packed ),
        Repr::Storage => quote!( #re::repr::Storage ),
    };

    // #[repr(...)]
    let repr_c_attr = util::try_parse_repr(&input.attrs)?;
    let mut repr_align_attr = None;
    let add_repr_c_message = "`derive(CPULayout)` requires an additional `#[repr(C)]` attribute";
    match repr_c_attr {
        None => {
            if let WhichDerive::CpuLayout = which_derive {
                bail!(*span, add_repr_c_message);
            }
        }
        Some((span, util::ReprAttr::Other { repr_c, align, packed })) => {
            repr_align_attr = align;
            match which_derive {
                WhichDerive::GpuLayout => {
                    bail!(
                        span,
                        "#[derive(GPULayout)] does not support #[repr(...)] attributes. \
                    Modification of the GPU layout is limited to #[gpu_repr(packed)] on structs and #[align(N)], #[size(N)] attributes \
                    in front of struct fields.\n see the WGSL spec on those attributes: \n https://www.w3.org/TR/WGSL/#align-attr \n https://www.w3.org/TR/WGSL/#size-attr"
                    );
                }
                WhichDerive::CpuLayout => {
                    if !repr_c {
                        bail!(span, add_repr_c_message);
                    }
                    if packed.is_some() {
                        bail!(span, "`derive(CPULayout)` does not support `repr(packed(_))`");
                    }
                }
            }
        }
        Some((span, util::ReprAttr::Transparent)) => {
            bail!(span, "``repr(transparent)` is not supported by `derive(GPULayout)`");
        }
    }

    if let Some((attr_span, _)) = util::find_literal_list_attr::<LitInt>("size", &input.attrs)? {
        bail!(
            attr_span,
            "custom `size` attributes are only supported on struct fields"
        )
    }

    if let Some((attr_span, _)) = util::find_literal_list_attr::<LitInt>("align", &input.attrs)? {
        bail!(
            attr_span,
            "custom `align` attributes are only supported on struct fields"
        )
    }

    struct FieldAttrs {
        size: Option<(Span, LitInt)>,
        align: Option<(Span, LitInt)>,
    }

    let mut fields_with_attrs = Vec::with_capacity(num_fields);
    for field in fields.named.iter() {
        let fwa = FieldAttrs {
            size: util::find_literal_list_attr::<syn::LitInt>("size", &field.attrs)?,
            align: util::find_literal_list_attr::<syn::LitInt>("align", &field.attrs)?,
        };

        match gpu_repr {
            Repr::Packed => {
                if fwa.align.is_some() {
                    bail!(
                        field.span(),
                        "`#[gpu_repr(packed)]` structs do not support `#[align(N)]` attributes"
                    );
                }
                // TODO(chronicl) decide on whether size attribute is allowed. Will have to be adjusted in
                // LayoutCalculator too!
                // if fwa.size.is_some() {
                //     bail!(
                //         field.span(),
                //         "`#[gpu_repr(packed)]` structs do not support `#[size(N)]` attributes"
                //     );
                // }
            }
            Repr::Storage => {}
        }

        if let Some((span, align_lit)) = &fwa.align {
            match align_lit.base10_parse().map(u32::is_power_of_two) {
                Ok(true) => (),
                Ok(false) => bail!(*span, "alignment attribute must be a power of two"),
                Err(_) => bail!(
                    *span,
                    "alignment attribute must lie inside the u32 value range and be a power of two"
                ),
            }
        }
        fields_with_attrs.push(fwa);
    }

    // turns None/Some((span, x)) into quote_spanned!(span => None/Some(x))
    let quote_option = |opt| match opt {
        None => quote_spanned!(*span => None),
        Some((span, x)) => quote_spanned!(span => Some(#x)),
    };

    let field_size = &fields_with_attrs
        .iter()
        .map(|f| quote_option(f.size.clone()))
        .collect::<Vec<_>>();
    let field_align = &fields_with_attrs
        .iter()
        .map(|f| quote_option(f.align.clone()))
        .collect::<Vec<_>>();

    let (last_field_ident, first_fields_ident) = field_ident.split_last().expect("checked above");
    let (last_field_type, first_fields_type) = field_type.split_last().expect("checked above");
    let (last_field_size, first_fields_size) = field_size.split_last().expect("checked above");
    let (last_field_align, first_fields_align) = field_align.split_last().expect("checked above");

    let enable_if_last_field_has_size_attribute = fields_with_attrs
        .last()
        .and_then(|l| l.size.clone())
        .map(|_| quote!(()))
        .into_iter();

    // trick for achieving https://doc.rust-lang.org/beta/unstable-book/language-features/trivial-bounds.html
    let triv = quote!(for<'trivial_bound>);

    match which_derive {
        WhichDerive::GpuLayout => {
            let impl_layoutable = quote! {
                impl<#generics_decl> #re::Layoutable for #derive_struct_ident<#(#idents_of_generics),*>
                where
                    // These NoBools and NoHandle bounds are only for better diagnostics, Layoutable already implies them
                    #(#first_fields_type: #re::NoBools + #re::NoHandles + #re::Layoutable + #re::GpuSized,)*
                    #last_field_type: #re::NoBools + #re::NoHandles + #re::Layoutable,
                    #where_clause_predicates
                {
                    fn layoutable_type() -> #re::LayoutableType {
                        let result = #re::LayoutableType::struct_from_parts(
                            std::stringify!(#derive_struct_ident),
                            [
                                #((
                                    #re::FieldOptions::new(
                                        std::stringify!(#field_ident),
                                        #field_align.map(|align: u32| TryFrom::try_from(align).expect("power of two validated during codegen")).into(),
                                        #field_size.into(),
                                    ),
                                    <#field_type as #re::Layoutable>::layoutable_type()
                                ),)*
                            ]
                        );

                        match result {
                            Ok(layoutable_type) => layoutable_type,
                            Err(#re::StructFromPartsError::MustHaveAtLeastOneField) => unreachable!("checked above"),
                            Err(#re::StructFromPartsError::OnlyLastFieldMayBeUnsized) => unreachable!("ensured by field trait bounds"),
                            // GpuType is not implemented for derived structs directly, so they can't be used
                            // as the field of another struct, instead shame::Struct<T> has to be used, which
                            // only accepts sized structs.
                            Err(#re::StructFromPartsError::MustNotHaveUnsizedStructField) => unreachable!("GpuType bound  for fields makes this impossible"),
                        }
                    }
                }
            };

            let impl_gpu_layout = quote! {
                impl<#generics_decl> #re::GpuLayout for #derive_struct_ident<#(#idents_of_generics),*>
                where
                    #(#first_fields_type: #re::Layoutable + #re::GpuSized,)*
                    #last_field_type: #re::Layoutable,
                    #where_clause_predicates
                {
                    type GpuRepr = #gpu_repr_shame;

                    fn cpu_type_name_and_layout() -> Option<Result<(std::borrow::Cow<'static, str>, #re::TypeLayout), #re::ArrayElementsUnsizedError>> {
                        use #re::CpuLayout as _;
                        #(
                            Some(Ok((
                                std::stringify!(#cpu_equivalent_type).into(),
                                <#cpu_equivalent_type>::cpu_layout(),
                            )))
                        )*
                        #(#none_if_no_cpu_equivalent_type /*yields `None`*/)*
                    }

                }
            };

            let impl_vertex_buffer_layout = quote! {
                impl<#generics_decl> #re::VertexLayout for #derive_struct_ident<#(#idents_of_generics),*>
                where
                    #(#triv #field_type: #re::VertexAttribute,)*
                    #where_clause_predicates
                { }
            };



            let impl_from_anys = quote! {
                impl<#generics_decl> #re::FromAnys for #derive_struct_ident<#(#idents_of_generics),*>
                where #where_clause_predicates {
                    fn expected_num_anys() -> usize {#num_fields}

                    #[track_caller]
                    fn from_anys(mut anys: impl Iterator<Item = #re::Any>) -> Self {
                        use #re::{
                            collect_into_array_exact,
                            push_wrong_amount_of_args_error
                        };

                        const EXPECTED_LEN: usize = #num_fields;
                        let [#(#field_ident),*] = match collect_into_array_exact::<#re::Any, EXPECTED_LEN>(anys) {
                            Ok(t) => t,
                            Err(actual_len) => {
                                let any = push_wrong_amount_of_args_error(actual_len, EXPECTED_LEN, #re::call_info!());
                                [any; EXPECTED_LEN]
                            }
                        };

                        Self {
                            #(#field_ident: <#field_type as #re::GpuLayoutField>::from_any(#field_ident)),*
                        }
                    }
                }
            };

            let impl_fake_auto_traits = quote! {
                // impl fake auto traits via `for<'trivial_bound>` trick:

                impl<#generics_decl> #re::NoAtomics for #derive_struct_ident<#(#idents_of_generics),*>
                where
                    #(#triv #field_type: #re::NoAtomics,)*
                    #where_clause_predicates
                {}

                impl<#generics_decl> #re::NoBools for #derive_struct_ident<#(#idents_of_generics),*>
                where
                    #(#triv #field_type: #re::NoBools,)*
                    #where_clause_predicates
                {}

                // NoHandles does not use the 'trivial_bound trick, because it is a requirement for #[derive(GpuLayout)] and an implication of `GpuAligned`
                impl<#generics_decl> #re::NoHandles for #derive_struct_ident<#(#idents_of_generics),*>
                where
                    #(#field_type: #re::NoHandles,)*
                    #where_clause_predicates
                {}

                impl<#generics_decl> #re::GpuSized for #derive_struct_ident<#(#idents_of_generics),*>
                where
                    #(#triv #field_type: #re::GpuSized,)*
                    #where_clause_predicates
                {
                    fn sized_ty() -> #re::ir::SizedType where
                    #triv Self: #re::GpuType {
                        unreachable!("Self: !GpuType")
                    }
                }

                impl<#generics_decl> #re::GpuAligned for #derive_struct_ident<#(#idents_of_generics),*>
                where
                    #(#triv #field_type: #re::GpuAligned,)*
                    #where_clause_predicates
                {
                    fn aligned_ty() -> #re::ir::AlignedType where
                    #triv Self: #re::GpuType {
                        unreachable!("Self: !GpuType")
                    }
                }
            };

            match gpu_repr {
                Repr::Packed =>
                // this is basically only for vertex buffers, so
                // we only implement `GpuLayout` and `VertexLayout`, as well as their implied traits
                {
                    Ok(quote! {
                        #impl_layoutable
                        #impl_gpu_layout
                        #impl_vertex_buffer_layout
                        #impl_fake_auto_traits
                        #impl_from_anys
                    })
                }
                Repr::Storage => {
                    // non gpu_repr(packed)
                    let struct_ref_doc = format!(
                        r#"This struct was generated by `#[derive(shame::GpuLayout)]`
            as a version of `{derive_struct_ident}` which holds references to its fields. It is used as
            the `std::ops::Deref` target of `shame::Ref<{derive_struct_ident}>`"#
                    );

                    Ok(quote! {
                        #impl_layoutable
                        #impl_gpu_layout
                        #impl_vertex_buffer_layout
                        #impl_fake_auto_traits
                        #impl_from_anys

                        #[doc = #struct_ref_doc]
                        #[allow(non_camel_case_types)]
                        #[derive(Clone, Copy)]
                        #vis struct #derive_struct_ref_ident<_AS: #re::AddressSpace, _AM: #re::AccessMode>
                        where #(
                            #triv #field_type: #re::GpuStore + #re::GpuType,
                        )*
                        {
                            #(
                                #field_vis #field_ident: #re::Ref<#field_type, _AS, _AM>,
                            )*
                        }

                        impl<#generics_decl> #re::BufferFields for #derive_struct_ident<#(#idents_of_generics),*>
                        where
                            #(#triv #field_type: #re::GpuStore + #re::GpuType,)*
                            #(#triv #first_fields_type: #re::GpuSized,)*
                            #triv #last_field_type:     #re::GpuAligned,
                            #where_clause_predicates
                        {
                            fn as_anys(&self) -> impl std::borrow::Borrow<[#re::Any]> {
                                use #re::AsAny;
                                [
                                    #(self.#field_ident.as_any()),*
                                ]
                            }

                            fn clone_fields(&self) -> Self {
                                Self {
                                    #(#field_ident: std::clone::Clone::clone(&self.#field_ident)),*
                                }
                            }

                            fn get_bufferblock_type() -> #re::ir::BufferBlock {
                                let mut fields = std::vec::Vec::from([
                                    #(
                                        #re::ir::SizedField {
                                            name: std::stringify!(#first_fields_ident).into(),
                                            custom_min_size: #first_fields_size,
                                            custom_min_align: #first_fields_align.map(|align: u32| TryFrom::try_from(align).expect("power of two validated during codegen")),
                                            ty: <#first_fields_type as #re::GpuSized>::sized_ty(),
                                        }
                                    ),*
                                ]);

                                let mut last_unsized = None::<#re::ir::RuntimeSizedArrayField>;
                                #[allow(clippy::no_effect)]
                                {
                                    // this part is only here to force a compiler error if the last field
                                    // uses the #[size(n)] attribute but the type is not shame::GpuSized.
                                    #(#enable_if_last_field_has_size_attribute; // only generate the line below if the last field has a #[size(n)] attribute
                                        // compiler error if not shame::GpuSized
                                        fn __() where #last_field_type: #re::GpuSized {}
                                    )*

                                    match <#last_field_type as #re::GpuAligned>::aligned_ty() {
                                        #re::ir::AlignedType::Sized(ty) =>
                                            fields.push(#re::ir::SizedField {
                                                name: std::stringify!(#last_field_ident).into(),
                                                custom_min_size: #last_field_size,
                                                custom_min_align: #last_field_align.map(|align: u32| TryFrom::try_from(align).expect("power of two validated during codegen")),
                                                ty
                                            }),
                                        #re::ir::AlignedType::RuntimeSizedArray(element_ty) =>
                                            last_unsized = Some(#re::ir::RuntimeSizedArrayField {
                                                name: std::stringify!(#last_field_ident).into(),
                                                custom_min_align: #last_field_align.map(|align: u32| TryFrom::try_from(align).expect("power of two validated during codegen")),
                                                element_ty
                                            }),
                                    }
                                }

                                use #re::BufferBlockDefinitionError as E;
                                match #re::ir::BufferBlock::new(
                                    std::stringify!(#derive_struct_ident).into(),
                                    fields,
                                    last_unsized
                                ) {
                                    Ok(t) => t,
                                    Err(e) => match e {
                                        E::MustHaveAtLeastOneField => unreachable!(">= 1 field is ensured by derive macro"),
                                        E::FieldNamesMustBeUnique(_) => unreachable!("unique field idents are ensured by rust struct definition"),
                                    }
                                }
                            }
                        }

                        impl<#generics_decl> #re::GpuStore for #derive_struct_ident<#(#idents_of_generics),*>
                        where
                            #(#triv #field_type: #re::GpuStore + #re::GpuType,)*
                            #where_clause_predicates
                        {
                            type RefFields<AS: #re::AddressSpace, AM: #re::AccessMode> = #derive_struct_ref_ident<AS, AM>;

                            fn store_ty() -> #re::ir::StoreType where
                            #triv Self: #re::GpuType {
                                unreachable!("Self: !GpuType")
                            }

                            fn instantiate_buffer_inner<AS: #re::BufferAddressSpace>(
                                args: Result<#re::BindingArgs, #re::InvalidReason>,
                                bind_ty: #re::BindingType
                            ) -> #re::BufferInner<Self, AS>
                            where
                                #triv Self:
                                    #re::NoAtomics +
                                    #re::NoBools
                            {
                                #re::BufferInner::new_fields(args, bind_ty)
                            }

                            fn instantiate_buffer_ref_inner<AS: #re::BufferAddressSpace, AM: #re::AccessModeReadable>(
                                args: Result<#re::BindingArgs, #re::InvalidReason>,
                                bind_ty: #re::BindingType
                            ) -> #re::BufferRefInner<Self, AS, AM>
                            where
                                #triv Self: #re::NoBools,
                            {
                                #re::BufferRefInner::new_fields(args, bind_ty)
                            }

                            fn impl_category() -> #re::GpuStoreImplCategory {
                                #re::GpuStoreImplCategory::Fields(<Self as #re::BufferFields>::get_bufferblock_type())
                            }
                        }

                        impl<#generics_decl> #re::SizedFields for #derive_struct_ident<#(#idents_of_generics),*>
                        where
                            #(#triv #field_type:  #re::GpuSized + #re::GpuStore + #re::GpuType,)*
                            #where_clause_predicates
                        {
                            fn get_sizedstruct_type() -> #re::ir::SizedStruct {
                                let struct_ = #re::ir::SizedStruct::new_nonempty(
                                    std::stringify!(#derive_struct_ident).into(),
                                    std::vec::Vec::from([
                                        #(
                                            #re::ir::SizedField {
                                                name: std::stringify!(#first_fields_ident).into(),
                                                custom_min_size: #first_fields_size,
                                                custom_min_align: #first_fields_align.map(|align: u32| TryFrom::try_from(align).expect("power of two validated during codegen")),
                                                ty: <#first_fields_type as #re::GpuSized>::sized_ty(),
                                            }
                                        ),*
                                    ]),
                                    #re::ir::SizedField {
                                        name: std::stringify!(#last_field_ident).into(),
                                        custom_min_size: #last_field_size,
                                        custom_min_align: #last_field_align.map(|align: u32| TryFrom::try_from(align).expect("power of two validated during codegen")),
                                        ty: <#last_field_type as #re::GpuSized>::sized_ty(),
                                    }
                                );
                                match struct_ {
                                    Ok(s) => s,
                                    Err(#re::ir::StructureFieldNamesMustBeUnique { .. }) => unreachable!("field name uniqueness is checked by rust"),
                                }
                            }
                        }

                        impl<#generics_decl> #re::GetAllFields for #derive_struct_ident<#(#idents_of_generics),*>
                        where
                            #(#triv #first_fields_type: #re::GpuSized,)*
                            #triv #last_field_type: #re::GpuAligned,
                            #where_clause_predicates
                        {
                            fn fields_as_anys_unchecked(self_: #re::Any) -> impl std::borrow::Borrow<[#re::Any]> {
                                [
                                    #(self_.get_field(std::stringify!(#field_ident).into())),*
                                ]
                            }
                        }

                        impl<AS: #re::AddressSpace, AM: #re::AccessMode> #re::FromAnys for #derive_struct_ref_ident<AS, AM>
                        where #(
                            #triv #field_type: #re::GpuStore + #re::GpuType,
                        )* {
                            fn expected_num_anys() -> usize {#num_fields}

                            #[track_caller]
                            fn from_anys(mut anys: impl Iterator<Item = #re::Any>) -> Self {
                                use #re::{
                                    collect_into_array_exact,
                                    push_wrong_amount_of_args_error
                                };
                                const EXPECTED_LEN: usize = #num_fields;
                                let [#(#field_ident),*] = match collect_into_array_exact::<#re::Any, EXPECTED_LEN>(anys) {
                                    Ok(t) => t,
                                    Err(actual_len) => {
                                        let any = push_wrong_amount_of_args_error(actual_len, EXPECTED_LEN, #re::call_info!());
                                        [any; EXPECTED_LEN]
                                    }
                                };
                                Self {
                                    #(#field_ident: From::from(#field_ident)),*
                                }
                            }
                        }
                    })
                }
            }
        }
        WhichDerive::CpuLayout => {
            let align_attr_or_none = match repr_align_attr {
                None => quote!(None),
                Some(n) => quote!(
                    Some(#re::U32PowerOf2::try_from_usize(#n)).expect("rust checks that N in repr(C, align(N)) is a power of 2.")
                ),
            };

            Ok(quote! {

            impl<#generics_decl> #re::CpuLayout for #derive_struct_ident<#(#idents_of_generics),*>
            where
                #(#first_fields_type: ::std::marker::Sized,)*
                #(#field_type: #re::CpuAligned,)*
                #where_clause_predicates
            {
                fn cpu_layout() -> #re::TypeLayout {
                    //use #re::CpuLayout // using `use` instead of `as #re::CpuAligned` allows for duck-traits to circumvent the orphan rule
                    use #re::CpuAligned;
                    use #re::CpuLayout as _;
                    let layout = #re::repr_c_struct_layout(
                        #align_attr_or_none,
                        std::stringify!(#derive_struct_ident).into(),
                        &[
                            #((
                                #re::ReprCField {
                                    name: std::stringify!(#first_fields_ident),
                                    alignment: <#first_fields_type>::CPU_ALIGNMENT,
                                    layout: <#first_fields_type>::cpu_layout(), // DO NOT refactor to `as #re::CpuLayout`, that would prevent the duck-trait trick for circumventing the orphan rule
                                },
                                std::mem::offset_of!(#derive_struct_ident, #first_fields_ident),
                                std::mem::size_of::<#first_fields_type>(),
                            )),*
                        ],
                        #re::ReprCField {
                            name: std::stringify!(#last_field_ident),
                            alignment: <#last_field_type>::CPU_ALIGNMENT,
                            layout: <#last_field_type>::cpu_layout(), // DO NOT refactor to `as #re::CpuLayout`, that would prevent the duck-trait trick for circumventing the orphan rule
                        },
                        <#last_field_type>::CPU_SIZE
                    );

                    match layout {
                        Ok(l) => l,
                        Err(e) => match e {
                            #re::ReprCError::SecondLastElementIsUnsized => unreachable!("`offset_of` was called on this elmement, so it must be sized")
                        }
                    }
                }

                // fn gpu_type_layout() -> Option<Result<#re::TypeLayout, #re::ArrayElementsUnsizedError>> {
                //     panic!("gpu_type_layout on `CpuLayout` types is currently unsupported")
                // }
            }
                        })
        }
    }
}
