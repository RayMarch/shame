use super::{operation::*, Decomposition, Expr, NoMatchingSignature, TypeCheck};
use crate::frontend::any::Any;
use crate::{
    call_info,
    frontend::{any::record_node, error::InternalError},
    impl_track_caller_fn_any,
    ir::{
        self,
        expr::type_check::SignatureStrings,
        recording::{Context, NodeRecordingError},
    },
};
use std::{fmt::Display, ops::*};

/// wgsl operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    /// !a
    // (no test case yet)
    Not,
    /// ~a
    // (no test case yet)
    Complement,
    /// -a
    Neg,
    /// a & b
    // (no test case yet)
    And,
    /// a && b
    // (no test case yet)
    AndAnd,
    /// a | b
    // (no test case yet)
    Or,
    /// a || b
    // (no test case yet)
    OrOr,
    /// a ^ b
    // (no test case yet)
    Xor,
    /// a + b
    Add,
    /// a - b
    Sub,
    /// a * b
    Mul,
    /// a / b
    // (no test case yet)
    Div,
    /// a % b
    // (no test case yet)
    Rem,
    /// &a
    // (no test case yet)
    AddressOf,
    /// *a
    // (no test case yet)
    Indirection,
    /// a << b
    // (no test case yet)
    Shl,
    /// a >> b
    // (no test case yet)
    Shr,
    /// a == b
    // (no test case yet)
    Equal,
    /// a != b
    // (no test case yet)
    NotEqual,
    /// a > b
    // (no test case yet)
    GreaterThan,
    /// a < b
    LessThan,
    /// a >= b
    // (no test case yet)
    GeraterThanOrEqual,
    /// a <= b
    // (no test case yet)
    LessThanOrEqual,
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Operator::Not => "!",
            Operator::Complement => "~",
            Operator::Neg => "-",
            Operator::And => "&",
            Operator::AndAnd => "&&",
            Operator::Or => "|",
            Operator::OrOr => "||",
            Operator::Xor => "^",
            Operator::Add => "+",
            Operator::Sub => "-",
            Operator::Mul => "*",
            Operator::Div => "/",
            Operator::Rem => "%",
            Operator::AddressOf => "AddressOf (&)",
            Operator::Indirection => "Indirection (*)",
            Operator::Shl => "<<",
            Operator::Shr => ">>",
            Operator::Equal => "==",
            Operator::NotEqual => "!=",
            Operator::GreaterThan => ">",
            Operator::LessThan => "<",
            Operator::GeraterThanOrEqual => ">=",
            Operator::LessThanOrEqual => "<=",
        };
        write!(f, "operator {str}")
    }
}

impl From<Operator> for Expr {
    fn from(x: Operator) -> Self { Expr::Operator(x) }
}

impl Operator {
    #[rustfmt::skip]
    const fn get_operations(self) -> &'static [Operation] {
        match self {
            Operator::Not => &[
                Operation::Logical(Logical::LogicNot)
            ],
            Operator::Neg => &[
                Operation::Arithmetic(Arithmetic::Negation)
            ],
            Operator::Complement => &[
                Operation::Bit(Bit::BitwiseComplement)
            ],
            Operator::And => &[
                Operation::Logical(Logical::LogicAnd),
                Operation::Bit(Bit::BitwiseAnd)
            ],
            Operator::AndAnd => &[
                Operation::Logical(Logical::ShortCircuitAnd)
            ],
            Operator::Or => &[
                Operation::Logical(Logical::LogicOr),
                Operation::Bit(Bit::BitwiseOr)
            ],
            Operator::OrOr => &[
                Operation::Logical(Logical::ShortCircuitOr)
            ],
            Operator::Xor => &[
                Operation::Bit(Bit::BitwiseExclusiveOr)
            ],
            Operator::Add => &[
                Operation::Arithmetic(Arithmetic::Addition),
                Operation::MixedSVArithmetic(MixedSVArithmetic::Addition),
                Operation::MatrixArithmetic(MatrixArithmetic::MatrixAddition),
            ],
            Operator::Sub => &[
                Operation::Arithmetic(Arithmetic::Subtraction),
                Operation::MixedSVArithmetic(MixedSVArithmetic::Subtraction),
                Operation::MatrixArithmetic(MatrixArithmetic::MatrixSubtraction),
            ],
            Operator::Mul => &[
                Operation::Arithmetic(Arithmetic::Multiplication),
                Operation::MixedSVArithmetic(MixedSVArithmetic::Multiplication),
                Operation::MatrixArithmetic(MatrixArithmetic::ComponentWiseScaling),
                Operation::MatrixArithmetic(MatrixArithmetic::LinearAlgebraMatrixColumnVectorProduct),
                Operation::MatrixArithmetic(MatrixArithmetic::LinearAlgebraRowVectorMatrixProduct),
                Operation::MatrixArithmetic(MatrixArithmetic::LinearAlgebraMatrixProduct),
            ],
            Operator::Div => &[
                Operation::Arithmetic(Arithmetic::Division),
                Operation::MixedSVArithmetic(MixedSVArithmetic::Division),
            ],
            Operator::Rem => &[
                Operation::Arithmetic(Arithmetic::Division),
                Operation::MixedSVArithmetic(MixedSVArithmetic::Division),
            ],
            Operator::AddressOf => &[
                Operation::AddressOf
            ],
            Operator::Indirection => &[
                Operation::Indirection
            ],
            Operator::Shl => &[
                Operation::Bit(Bit::ShiftLeft)
            ],
            Operator::Shr => &[
                Operation::Bit(Bit::ShiftRight)
            ],
            Operator::Equal => &[
                Operation::Comparison(Comparison::Equality)
            ],
            Operator::NotEqual => &[
                Operation::Comparison(Comparison::Inequality)
            ],
            Operator::GreaterThan => &[
                Operation::Comparison(Comparison::GreaterThan)
            ],
            Operator::LessThan => &[
                Operation::Comparison(Comparison::LessThan)
            ],
            Operator::GeraterThanOrEqual => &[
                Operation::Comparison(Comparison::GreaterThanOrEqual)
            ],
            Operator::LessThanOrEqual => &[
                Operation::Comparison(Comparison::LessThanOrEqual)
            ],
        }
    }
}

impl TypeCheck for Operator {
    #[rustfmt::skip]
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        // this function tries all of the `Operation`s for this `Operator`.
        // If none of them has a signature that matches `args`, 
        // all the mismatching signatures are combined into one large
        // `NoMatchingSignature` error.

        let push_conflict_err = |op, args, a, b| Context::try_with(call_info!(), |ctx|
            ctx.push_error(InternalError::new(true,
                format!("at least two matching operations for operator `{:?}` with arguments {:?} \
                which have conflicting inferred return types ({:?} vs {:?}).", 
                self, args, a, b)
            ).into()));

        let ops = self.get_operations();

        match ops.iter().find_map(|op| op.infer_type(args).ok()) {
            // early out, no error concat
            Some(ty) => Ok(ty),
            // concat all mismatching signatures
            None => ops.iter().map(|op| op.infer_type(args))
            .reduce(|e0, e1| match (e0, e1) {
                (Err(a), Err(b)) => Err(a.concat(b)),
                (Err(_), x) | (x, Err(_)) => x,
                (Ok(a), Ok(b)) => {
                    if a != b {push_conflict_err(self, args, a.clone(), b);}
                    Ok(a)
                }
            }).unwrap_or_else(|| Err(NoMatchingSignature::empty_with_name(format!("{:?}", self).into(), args)))
        }
    }
}

macro_rules! impl_any_unary_operators {
    ($($Not: ident, $not: ident, $op: expr;)*) => {
        $(impl $Not for Any {
            type Output = Any;
            #[track_caller]
            fn $not(self) -> Self::Output {
                $crate::frontend::any::record_node($crate::call_info!(), $crate::ir::expr::Expr::Operator($op), &[self])
            }
        })*
    };
}

macro_rules! impl_any_binary_operators {
    ($($Add: ident, $add: ident, $op: expr;)*) => {
        $(impl $Add for Any {
            type Output = Any;
            #[track_caller]
            fn $add(self, rhs: Self) -> Self::Output {
                $crate::frontend::any::record_node($crate::call_info!(), $crate::ir::expr::Expr::Operator($op), &[self, rhs])
            }
        })*
    };
}

impl_any_unary_operators! {
    Not, not, Operator::Not;
    Neg, neg, Operator::Neg;
}

impl_any_binary_operators! {
    BitAnd, bitand, Operator::And;
    BitOr , bitor , Operator::Or;
    BitXor, bitxor, Operator::Xor;
    Add, add, Operator::Add;
    Sub, sub, Operator::Sub;
    Mul, mul, Operator::Mul;
    Div, div, Operator::Div;
    Rem, rem, Operator::Rem;
    Shl, shl, Operator::Shl;
    Shr, shr, Operator::Shr;
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn bitwise_complement(&self) -> Any => [*self] Operator::Complement;

        pub fn or_short_circuit(&self, rhs: Any) -> Any => [*self, rhs] Operator::OrOr;
        pub fn and_short_circuit(&self, rhs: Any) -> Any => [*self, rhs] Operator::AndAnd;

        pub fn indirection(&self) -> Any => [*self] Operator::Indirection;

        pub fn eq(&self, rhs: Any) -> Any => [*self, rhs] Operator::Equal;
        pub fn ne(&self, rhs: Any) -> Any => [*self, rhs] Operator::NotEqual;
        pub fn lt(&self, rhs: Any) -> Any => [*self, rhs] Operator::LessThan;
        pub fn gt(&self, rhs: Any) -> Any => [*self, rhs] Operator::GreaterThan;
        pub fn le(&self, rhs: Any) -> Any => [*self, rhs] Operator::LessThanOrEqual;
        pub fn ge(&self, rhs: Any) -> Any => [*self, rhs] Operator::GeraterThanOrEqual;
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn address(&self) -> Any {
        use Decomposition as D;
        let call_info = call_info!();
        if let Some(node) = self.node() {
            Context::try_with(call_info, |ctx| {
                if let Expr::Decomposition(D::VectorAccess(_) | D::VectorIndex | D::VectorIndexConst(_)) =
                    &ctx.pool()[node].expr
                {
                    ctx.push_error(NodeRecordingError::CannotTakeAddressOfVectorComponents.into())
                }
            });
        }
        record_node(call_info, Expr::Operator(Operator::AddressOf), &[*self])
    }
}
