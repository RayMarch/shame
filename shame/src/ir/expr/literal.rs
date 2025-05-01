use std::fmt::Display;

use super::{type_check::TypeCheck, BuiltinFn, Constructor};
use crate::frontend::any::Any;
use crate::{
    impl_track_caller_fn_any, ir, ir::expr::type_check::NoMatchingSignature, ir::expr::Expr, ir::ir_type, ir::Type, sig,
};
use ir_type::{Len::*, SizedType::*, StoreType::*};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Literal(pub ir_type::ScalarConstant);

impl Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //write!(f, "literal ");
        match self.0 {
            ir::ScalarConstant::F16(f16) => write!(f, "{}h", f32::from(f16)),
            ir::ScalarConstant::F32(x) => write!(f, "{}f", x),
            ir::ScalarConstant::F64(x) => write!(f, "{}d", x),
            ir::ScalarConstant::U32(x) => write!(f, "{}u", x),
            ir::ScalarConstant::I32(x) => write!(f, "{}i", x),
            ir::ScalarConstant::Bool(x) => write!(f, "{x}"),
        }
    }
}

impl TypeCheck for Literal {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        let t = self.0.ty();
        sig!({name: Literal(t),},
            [] => Vector(X1, t),
        )(self, args)
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn new_scalar(constant: ir::ScalarConstant) -> Any => [] Expr::Literal(Literal(constant));
        pub fn new_default(ty: ir::SizedType) -> Any => [] BuiltinFn::Constructor(Constructor::Default(ty));
    }
}
