use super::{NoMatchingSignature, TypeCheck};
use crate::{
    ir::{self, expr::type_check::SigFormatting, AddressSpace},
    same, sig,
};
use ir::Len::*;
use ir::ScalarType::*;
use ir::SizedType::*;
use ir::StoreType::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// the `Operator`s' expressions, as listed in the WGSL spec.
pub enum Operation {
    Logical(Logical),
    Arithmetic(Arithmetic),
    /// Binary arithmetic expressions with mixed scalar and vector operands
    MixedSVArithmetic(MixedSVArithmetic),
    MatrixArithmetic(MatrixArithmetic),
    Comparison(Comparison),
    Bit(Bit),
    AddressOf,
    Indirection,
}

impl TypeCheck for Operation {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        use ir::Type::*;
        match self {
            Operation::Logical(x) => x.infer_type(args),
            Operation::Arithmetic(x) => x.infer_type(args),
            Operation::MixedSVArithmetic(x) => x.infer_type(args),
            Operation::MatrixArithmetic(x) => x.infer_type(args),
            Operation::Comparison(x) => x.infer_type(args),
            Operation::Bit(x) => x.infer_type(args),
            Operation::AddressOf => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Ref(a, t, am)] if a.address_space != AddressSpace::Handle => Ptr(a.clone(), t.clone(), *am)
            )(self, args),
            Operation::Indirection => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Ptr(a, t, am)] => Ref(a.clone(), t.clone(), *am)
            )(self, args),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// https://www.w3.org/TR/WGSL/#logical-expr
pub enum Logical {
    /// !a
    LogicNot,
    /// a || b
    ShortCircuitOr,
    /// a | b
    LogicOr,
    /// a && b
    ShortCircuitAnd,
    /// a & b
    LogicAnd,
}

impl TypeCheck for Logical {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        match self {
            Logical::LogicNot => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n, Bool)] => Vector(*n, Bool)
            )(self, args),
            Logical::ShortCircuitOr | Logical::ShortCircuitAnd => sig!(
                [Vector(X1, Bool), Vector(X1, Bool)] => Vector(X1, Bool)
            )(self, args),
            Logical::LogicOr | Logical::LogicAnd => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n0, Bool), Vector(n1, Bool)] if same!(n0 n1) => Vector(*n1, Bool)
            )(self, args),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// https://www.w3.org/TR/WGSL/#arithmetic-expr
pub enum Arithmetic {
    Negation,
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Remainder,
}

impl TypeCheck for Arithmetic {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        match self {
            Arithmetic::Negation => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n, t)] if t.is_numeric() && t.is_signed() => Vector(*n, *t)
            )(self, args),
            Arithmetic::Addition |
            Arithmetic::Subtraction |
            Arithmetic::Multiplication |
            Arithmetic::Division |
            Arithmetic::Remainder => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n0, t0), Vector(n1, t1)] if t0.is_numeric() && same!(n0 n1; t0 t1) => Vector(*n0, *t0)
            )(self, args),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// the table "Binary arithmetic expressions with mixed scalar and vector operands"
/// in https://www.w3.org/TR/WGSL/#arithmetic-expr
///
/// - vector x scalar
/// - scalar x vector
pub enum MixedSVArithmetic {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Remainder,
}

impl TypeCheck for MixedSVArithmetic {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        match self {
            MixedSVArithmetic::Addition |
            MixedSVArithmetic::Subtraction |
            MixedSVArithmetic::Multiplication |
            MixedSVArithmetic::Division |
            MixedSVArithmetic::Remainder => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n, t0), Vector(X1, t1)] if t0.is_numeric() && same!(t0 t1) => Vector(*n, *t0),
                [Vector(X1, t0), Vector(n, t1)] if t0.is_numeric() && same!(t0 t1) => Vector(*n, *t0),
            )(self, args),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// the table "Matrix arithmetic" in https://www.w3.org/TR/WGSL/#arithmetic-expr
pub enum MatrixArithmetic {
    MatrixAddition,
    MatrixSubtraction,
    ComponentWiseScaling,
    LinearAlgebraMatrixColumnVectorProduct,
    LinearAlgebraRowVectorMatrixProduct,
    LinearAlgebraMatrixProduct,
}

impl TypeCheck for MatrixArithmetic {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        match self {
            MatrixArithmetic::MatrixAddition | MatrixArithmetic::MatrixSubtraction => sig!(
                [mat0, mat1 @ Matrix(_, _, t)] if mat0 == mat1 && t.is_floating_point() => mat0,
            )(self, args),
            MatrixArithmetic::ComponentWiseScaling => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(X1, t0), Matrix(c, r, t1)] if t0 == t1 && t0.is_floating_point() => Matrix(*c, *r, *t1),
                [Matrix(c, r, t0), Vector(X1, t1)] if t0 == t1 && t0.is_floating_point() => Matrix(*c, *r, *t0),
            )(self, args),
            MatrixArithmetic::LinearAlgebraMatrixColumnVectorProduct => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Matrix(c, r, t0), Vector(c1, t1)] if same!(t0 t1; c c1) && t0.is_floating_point() => Vector((*r).into(), *t1),
            )(self, args),
            MatrixArithmetic::LinearAlgebraRowVectorMatrixProduct => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(r1, t1), Matrix(c, r, t0)] if same!(t0 t1; r r1) && t0.is_floating_point() => Vector((*c).into(), *t1),
            )(self, args),
            MatrixArithmetic::LinearAlgebraMatrixProduct => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Matrix(k0, r, t0), Matrix(c, k1, t1)] if same!(t0 t1; k0 k1) && t0.is_floating_point() => Matrix(*c, *r, *t0),
            )(self, args),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// https://www.w3.org/TR/WGSL/#comparison-expr
pub enum Comparison {
    Equality,
    Inequality,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl TypeCheck for Comparison {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        match self {
            Comparison::Equality | Comparison::Inequality => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n0, t0), Vector(n1, t1)] if same!(n0 n1; t0 t1) => Vector(*n0, Bool)
            )(self, args),
            Comparison::LessThan |
            Comparison::LessThanOrEqual |
            Comparison::GreaterThan |
            Comparison::GreaterThanOrEqual => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n0, t0), Vector(n1, t1)] if t0.is_numeric() && same!(n0 n1; t0 t1) => Vector(*n0, Bool)
            )(self, args),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// https://www.w3.org/TR/WGSL/#bit-expr
pub enum Bit {
    BitwiseComplement,
    BitwiseOr,
    BitwiseAnd,
    BitwiseExclusiveOr,
    ShiftLeft,
    ShiftRight,
}

impl TypeCheck for Bit {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        match self {
            Bit::BitwiseComplement => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n, t)] if t.is_integer() => Vector(*n, *t)
            )(self, args),
            Bit::BitwiseOr | Bit::BitwiseAnd | Bit::BitwiseExclusiveOr => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n0, t0), Vector(n1, t1)] if same!(n0 n1; t0 t1) && t0.is_integer() => Vector(*n0, *t0)
            )(self, args),
            Bit::ShiftLeft | Bit::ShiftRight => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n0, t), Vector(n1, U32)] if n0 == n1 && t.is_integer() => Vector(*n0, *t)
            )(self, args),
        }
    }
}
