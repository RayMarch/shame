
use smallvec::SmallVec;
use crate::common::IteratorExt;
use super::{EST_ARGS, Ty};
use HighLevelGlslGeneric::*;
use GlslGenericInstantiation::*;
use super::*;

#[macro_export]
macro_rules! glsl_generic_function_decls {
    (
        $(
            $ret_type: ident $fn_name: ident (
                $($arg_type: ident $arg_name: ident),*
                $(, [$($opt_arg_type: ident $opt_arg_name: ident),*])?
            );
        )*
    ) => {
        
        [$(GlslGenericFunctionDecl {
            ret: $ret_type,
            name: stringify!($fn_name),
            args:          smallvec::SmallVec::from_slice(&[  $((    $arg_type, stringify!(    $arg_name))),*  ]),
            optional_args: smallvec::SmallVec::from_slice(&[$($(($opt_arg_type, stringify!($opt_arg_name))),*)?]),
        }),*]
        
    };
}

pub struct GlslGenericFunctionDecl {
    pub ret: GlslGeneric,
    pub name: &'static str,
    pub args:          SmallVec<[(GlslGeneric, &'static str); EST_ARGS]>,
    pub optional_args: SmallVec<[(GlslGeneric, &'static str); 2]>,
}

impl GlslGenericFunctionDecl {

    fn instantiate_return_type(generic_return: GlslGeneric, instantiation: GlslGenericInstantiation) -> Ty {
        let generic_return = HighLevelGlslGeneric::new(generic_return);

        match (generic_return, instantiation) {
            (Specific(ty), _) => ty,
            (GenXType(dtype), gen(shape)) => Ty::tensor(shape, dtype),
            (MatMxN(dtype), matMxN(m, n)) => Ty::tensor(Shape::Mat(m, n), dtype),
            (VecN(dtype), vecN(n))        => Ty::tensor(Shape::Vec(n)   , dtype),
            (GVec4F32I32U32, g(dtype))    => Ty::tensor(Shape::Vec(4)   , dtype),
            (GSampler(kind), g(dtype))    => Ty::texture_combined_sampler(dtype, kind),
            (VecNF32F64, _) => panic!("trying to deduce return type of generic type 'vec' for glsl vector-relational-function. Such a function does not exist according to the glsl 4.6 specification."),
            _ => panic!("trying to deduce return type for generic glsl function that uses two different kinds of generic instantiation")
        }
    }

    pub fn deduce_return_type(&self, args: &[Ty]) -> Option<Ty> {

        let allowed_arg_range = self.args.len()..=(self.args.len() + self.optional_args.len());
        
        match allowed_arg_range.contains(&args.len()) {
            false => None,
            true => {

                let args_zipped = self.args.iter()
                .chain(self.optional_args.iter())
                .zip(args.iter())
                .map(|((generic_arg, _), ty)| {
                    (HighLevelGlslGeneric::new(*generic_arg), ty)
                });

                let mut specifics = args_zipped.clone().filter_map(|(generic, ty)| match &generic {
                    Specific(spec) => Some((spec.clone(), ty)),
                    _ => None,
                });

                let generics = args_zipped.filter_map(|(generic, ty)| match &generic {
                    Specific(_) => None,
                    _ => Some((generic, ty)),
                });
            
                let result_ty = generics.all_same(|(generic_arg, arg)| { //all have to resolve to the same instanciation
                    generic_arg.try_instantiate(arg)
                })
                .flatten()
                .map(|inst| Self::instantiate_return_type(self.ret, inst));

                match specifics.all(|(spec, ty)| spec.eq_ignore_access(ty)) {
                    true => result_ty,
                    false => None,
                }
            },
        }
    }
}