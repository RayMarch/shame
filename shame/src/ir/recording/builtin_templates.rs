use crate::frontend::any::Any;
use crate::{
    call_info,
    frontend::any::record_node,
    frontend::encoding::pipeline_info::Dict,
    ir::{
        self,
        expr::{type_check::SigFormatting, AtomicFn, BuiltinFn, Expr, NoMatchingSignature, NumericFn},
        recording::Context,
        SizedStruct, SizedType, StoreType, StructureFieldNamesMustBeUnique, Type,
    },
    same, sig,
};
use std::{path::Display, rc::Rc};

use super::{InteractionKind, MemoryRegion};

#[derive(Debug, Default)]
pub struct BuiltinTemplateStructs {
    instantiations: Dict<TemplateStructParams, ir::SizedStruct>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TemplateStructParams {
    Frexp(FrexpGenerics),
    Modf(ModfGenerics),
    AtomicCompareExchangeWeak(AtomicCompareExchangeWeakGenerics),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrexpGenerics(pub ir::ScalarTypeFp, pub ir::Len);
pub struct FrexpInstance {
    pub fract: Any, // vec<.0, .1>
    pub exp: Any,   // vec<i32, .1>
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModfGenerics(pub ir::ScalarTypeFp, pub ir::Len);
pub struct ModfInstance {
    pub fract: Any, // vec<.0, .1>
    pub whole: Any, // vec<.0, .1>
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AtomicCompareExchangeWeakGenerics(pub ir::AddressSpace, pub ir::ScalarTypeInteger);
pub struct AtomicCompareExchangeWeakInstance {
    pub old_value: Any, // vec<.0, x1> (= u32x1 or i32x1)
    pub exchanged: Any, // bool
}

impl std::fmt::Display for TemplateStructParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TemplateStructParams as P;
        match self {
            P::Frexp(FrexpGenerics(fp, len)) => write!(f, "frexp<{fp}{len}>"),
            P::Modf(ModfGenerics(fp, len)) => write!(f, "modf<{fp}{len}>"),
            P::AtomicCompareExchangeWeak(AtomicCompareExchangeWeakGenerics(addr, int)) => write!(
                f,
                "atomic_compare_exchange_weak<{addr}, {}>",
                ir::ScalarType::from(*int)
            ),
        }
    }
}

impl BuiltinTemplateStructs {
    pub fn instantiations(&self) -> &Dict<TemplateStructParams, ir::SizedStruct> { &self.instantiations }

    /// infers the type of the corresponding `Expr` that uses `params`,
    pub(crate) fn infer_type(args: &[Type], params: TemplateStructParams) -> Result<Type, NoMatchingSignature> {
        let result: Result<Type, NoMatchingSignature> = Context::try_with(call_info!(), |ctx| {
            let mut self_ = ctx.pipeline().builtin_template_structs.borrow_mut();
            let template_struct = SizedType::Structure(self_.instantiate_if_needed(params));
            use ir::SizedType::Vector;
            match params {
                TemplateStructParams::Frexp(FrexpGenerics(fp, len)) => {
                    let t = ir::ScalarType::from(fp);
                    struct Frexp((), ()); // struct declaration only used for "name" below, so the generics make sense in the generated error message
                    sig!(
                        {
                            name: Frexp(len, t),
                            fmt: SigFormatting::RemoveAsterisksAndClone,
                        },
                        [Vector(l0, t0)] if *l0 == len && *t0 == t => template_struct,
                    )(&params, args)
                }
                TemplateStructParams::Modf(ModfGenerics(fp, len)) => {
                    let t = ir::ScalarType::from(fp);
                    struct Modf((), ()); // struct declaration only used for "name" below, so the generics make sense in the generated error message
                    sig!(
                        {
                            name: Modf(len, t),
                            fmt: SigFormatting::RemoveAsterisksAndClone,
                        },
                        [Vector(l0, t0)] if *l0 == len && *t0 == t => template_struct,
                    )(&params, args)
                }
                TemplateStructParams::AtomicCompareExchangeWeak(AtomicCompareExchangeWeakGenerics(
                    address_space,
                    int,
                )) => {
                    let t = &ir::ScalarType::from(int);
                    /*
                        fn atomicCompareExchangeWeak(
                            atomic_ptr: ptr<AS, atomic<T>, read_write>,
                            cmp: T,
                            v: T
                        ) -> __atomic_compare_exchange_result<T>
                    */
                    use ir::AccessMode::*;
                    use ir::Len::*;
                    use ir::SizedType::*;
                    use ir::StoreType::*;
                    use Type::*;

                    struct AtomicCompareExchangeWeak((), ()); // struct declaration only used for "name" below, so the generics make sense in the generated error message
                    sig!(
                        {
                            name: AtomicCompareExchangeWeak(address_space, t),
                            fmt: SigFormatting::RemoveAsterisksAndClone,
                        },
                        [
                            Ptr(region, Sized(Atomic(t0)), ReadWrite),
                            Store(Sized(Vector(X1, t1))),
                            Store(Sized(Vector(X1, t2))),
                        ]
                        if region.address_space == address_space &&
                            same!(t t0 t1 t2)
                        => template_struct
                    )(&params, args)
                }
            }
        })
        .unwrap_or_else(|| {
            Err(NoMatchingSignature::empty_with_name_and_comment(
                format!("{params}").into(),
                "(cannot infer return type, expr was called without an active encoding on this thread)".into(),
                args,
            ))
        });
        result
    }
}

fn new_field(ident: &'static str, ty: SizedType) -> ir::SizedField { ir::SizedField::new(ident.into(), None, None, ty) }

impl BuiltinTemplateStructs {
    fn instantiate_if_needed(&mut self, params: TemplateStructParams) -> ir::SizedStruct {
        self.instantiations
            .entry(params)
            .or_insert_with(|| match params {
                TemplateStructParams::Frexp(FrexpGenerics(fp, len)) => {
                    let struc = SizedStruct::new_nonempty(
                        format!("frexp_{fp}{len}_t").into(),
                        [new_field("fract", SizedType::Vector(len, fp.into()))].into(),
                        new_field("exp", SizedType::Vector(len, ir::ScalarType::I32)),
                    );
                    match struc {
                        Err(StructureFieldNamesMustBeUnique { .. }) => unreachable!("field names above are unique"),
                        Ok(s) => s,
                    }
                }
                TemplateStructParams::Modf(ModfGenerics(fp, len)) => {
                    let struc = SizedStruct::new_nonempty(
                        format!("modf_{fp}{len}_t").into(),
                        [new_field("fract", SizedType::Vector(len, fp.into()))].into(),
                        new_field("whole", SizedType::Vector(len, fp.into())),
                    );
                    match struc {
                        Err(StructureFieldNamesMustBeUnique { .. }) => unreachable!("field names above are unique"),
                        Ok(s) => s,
                    }
                }
                TemplateStructParams::AtomicCompareExchangeWeak(AtomicCompareExchangeWeakGenerics(addr, int)) => {
                    let int = ir::ScalarType::from(int);
                    let struc = SizedStruct::new_nonempty(
                        format!("atomicCmpExWk_{int}_{}_t", addr.ident_suffix()).into(),
                        [new_field("old_value", SizedType::Vector(ir::Len::X1, int))].into(),
                        new_field("exchanged", SizedType::Vector(ir::Len::X1, ir::ScalarType::Bool)),
                    );
                    match struc {
                        Err(StructureFieldNamesMustBeUnique { .. }) => unreachable!("field names above are unique"),
                        Ok(s) => s,
                    }
                }
            })
            .clone()
    }
}

impl Any {
    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn frexp(&self, genr: FrexpGenerics) -> FrexpInstance {
        let anonymous_struct = record_node(
            call_info!(),
            NumericFn::Exponent(ir::expr::ExponentFn::Frexp(genr)).into(),
            &[*self],
        );
        FrexpInstance {
            fract: anonymous_struct.get_field("fract".into()),
            exp: anonymous_struct.get_field("exp".into()),
        }
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn modf(&self, genr: ModfGenerics) -> ModfInstance {
        let anonymous_struct = record_node(
            call_info!(),
            NumericFn::Discontinuity(ir::expr::DiscontinuityFn::Modf(genr)).into(),
            &[*self],
        );
        ModfInstance {
            fract: anonymous_struct.get_field("fract".into()),
            whole: anonymous_struct.get_field("whole".into()),
        }
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn atomic_compare_exchange_weak(
        &mut self,
        genr: AtomicCompareExchangeWeakGenerics,
        cmp: Any,
        val: Any,
    ) -> AtomicCompareExchangeWeakInstance {
        let anonymous_struct = record_node(
            call_info!(),
            BuiltinFn::Atomic(ir::expr::AtomicFn::AtomicCompareExchangeWeak(genr)).into(),
            &[*self, cmp, val],
        );
        MemoryRegion::record_interaction(anonymous_struct, *self, InteractionKind::ReadWrite);
        AtomicCompareExchangeWeakInstance {
            old_value: anonymous_struct.get_field("old_value".into()),
            exchanged: anonymous_struct.get_field("exchanged".into()),
        }
    }
}
