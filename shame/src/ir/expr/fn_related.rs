use std::{fmt::Display, rc::Rc};

use crate::{
    call_info,
    common::{pool::Key, small_vec_actual::SmallVec},
    frontend::any::fn_builder::PassAs,
    ir::{
        self,
        expr::type_check::SignatureStrings,
        recording::{Context, FunctionDef, MemoryRegion},
        StoreType, Type,
    },
    sig,
};

use super::{NoMatchingSignature, TypeCheck};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgViewKind {
    Ref,
    Ptr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FnRelated {
    /// creation of a function parameter that exists as a placeholder for any
    /// later function call to the recorded function. this expression takes 0
    /// arguments and produces a value
    FnParamValue(StoreType),
    /// creation of a function parameter that exists as a placeholder for any
    /// later function call to the recorded function. this expression takes 0
    /// arguments and produces a Ref or Pointer depending on `ArgViewKind`
    FnParamMemoryView(Rc<MemoryRegion>, ArgViewKind),
    /// call operation of the function defined in .0
    Call(Key<FunctionDef>),
}

impl Display for FnRelated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FnRelated::FnParamValue(store_type) => write!(f, "fn-param: {store_type}"),
            FnRelated::FnParamMemoryView(memory_region, arg_view_kind) => match arg_view_kind {
                ArgViewKind::Ref => write!(f, "fn-param-ref: {}", &memory_region.ty),
                ArgViewKind::Ptr => write!(f, "fn-param-ptr: {}", &memory_region.ty),
            },
            FnRelated::Call(key) => Context::try_with(call_info!(), |ctx| -> Option<_> {
                let fn_defs = ctx.try_pool()?;
                let fn_def = fn_defs.get(*key)?;

                Some((|| -> Result<_, _> {
                    write!(f, "call fn{}(", key.index());
                    if !fn_def.params.is_empty() {
                        write!(f, "_")?;
                        for i in 1..fn_def.params.len() {
                            write!(f, ", _")?;
                        }
                    }
                    if fn_def.return_.is_some() {
                        write!(f, "-> _")?;
                    }
                    Ok(())
                })())
            })
            .flatten()
            .unwrap_or_else(|| write!(f, "call fn{}", key.index())),
        }
    }
}

impl TypeCheck for FnRelated {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        use Type::*;
        (match self {
            FnRelated::FnParamValue(t) => sig!(
                [] => t
            )(self, args),
            FnRelated::FnParamMemoryView(allocation, kind) => {
                let allocation = allocation.clone();
                let t = allocation.ty.clone();
                let access = allocation.allowed_access;
                match kind {
                    ArgViewKind::Ptr => sig!([] => Ptr(allocation, t, access))(self, args),
                    ArgViewKind::Ref => sig!([] => Ref(allocation, t, access))(self, args),
                }
            }
            FnRelated::Call(def_key) => {
                Context::try_with(call_info!(), |ctx| {
                    let defs = ctx.pool();
                    let nodes = ctx.pool();
                    let def = &defs[*def_key];

                    let same_len = args.len() == def.params.len();
                    let all_types_match = def.params.iter().map(|arg| nodes[*arg].ty()).zip(args).all(|(param, arg)| {
                        param == arg
                    });

                    if same_len && all_types_match {
                        let return_ty = def.return_.map(|key|
                            match nodes[key].ty().clone() {
                                Unit => Unit,
                                Ptr(a, store, access) => Ptr(a.new_return_value_dependent(ctx), store, access),
                                Ref(a, store, access) => Ref(a.new_return_value_dependent(ctx), store, access),
                                Store(store) => {
                                    if cfg!(feature = "debug_print") && cfg!(debug_assertions) {
                                        println!("TODO(release) push error for types that cannot be returned, such as StorageBlock, Handles, etc.");
                                    }
                                    Store(store)
                                },
                            }
                        ).unwrap_or(Unit);
                        Ok(return_ty)
                    } else {

                        let signature_string = {
                            use std::fmt::Write;
                            let mut s = String::new();
                            write!(s, "[");
                            let mut param_tys = def.params.iter().map(|arg| nodes[*arg].ty());
                            if let Some(first) = param_tys.next() {
                                write!(s, "{first:?}");
                            }
                            for param in param_tys {
                                write!(s, ", {param:?}");
                            }
                            write!(s, "] => ");
                            let return_ty = def.return_.map(|key| nodes[key].ty()).unwrap_or(&Unit);
                            write!(s, "{return_ty:?}");
                            s
                        };

                        Err(NoMatchingSignature {
                            expression_name: "Call".into(),
                            arguments: args.iter().cloned().collect(),
                            allowed_signatures: SignatureStrings::Dynamic(vec![signature_string]),
                            shorthand_level: Default::default(),
                            signature_formatting: None,
                            comment: None,
                        })
                    }
                }).unwrap_or_else(|| Err(NoMatchingSignature {
                    expression_name: "Call".into(),
                    arguments: args.iter().cloned().collect(),
                    allowed_signatures: SignatureStrings::Static(&[]),
                    shorthand_level: Default::default(),
                    signature_formatting: None,
                    comment: Some("the function call's type was inferred outside of an active Encoding. The function definition could not be fetched.".to_string()),
                }))
            },
        })
    }
}
