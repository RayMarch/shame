use std::fmt::Display;

use super::{
    type_check::{NoMatchingSignature, SigFormatting, TypeCheck},
    BuiltinFn, Expr,
};
use crate::frontend::any::Any;
use crate::{
    impl_track_caller_fn_any,
    ir::{
        ir_type::{
            AccessMode, AddressSpace, Indirection,
            Len::*,
            Len2,
            ScalarType::{self, *},
            SizedType::*,
            StoreType::*,
        },
        recording::{BuiltinTemplateStructs, FrexpGenerics, ModfGenerics},
    },
};
use crate::{ir, ir::ir_type::StoreType, same};
use crate::{ir::Type, sig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// the Numeric functions from the WGSL spec, split up into subcategories
pub enum NumericFn {
    TrigonometryFn(TrigonometryFn),
    LinearAlgebra(LinearAlgebraFn),
    Discontinuity(DiscontinuityFn),
    Exponent(ExponentFn),
    // (no test case yet)
    Bit(BitFn),
}

impl Display for NumericFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumericFn::TrigonometryFn(x) => write!(f, "{x:?}"),
            NumericFn::LinearAlgebra(x) => write!(f, "{x:?}"),
            NumericFn::Discontinuity(x) => write!(f, "{x:?}"),
            NumericFn::Exponent(x) => write!(f, "{x}"),
            NumericFn::Bit(x) => write!(f, "{x:?}"),
        }
    }
}

impl From<NumericFn> for Expr {
    fn from(value: NumericFn) -> Self { Expr::BuiltinFn(BuiltinFn::Numeric(value)) }
}

impl TypeCheck for NumericFn {
    #[rustfmt::skip]
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        match self {
            NumericFn::TrigonometryFn(x) => x.infer_type(args),
            NumericFn::LinearAlgebra(x)  => x.infer_type(args),
            NumericFn::Discontinuity(x)  => x.infer_type(args),
            NumericFn::Exponent(x)       => x.infer_type(args),
            NumericFn::Bit(x)            => x.infer_type(args),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigonometryFn {
    Sin,
    Sinh,
    Asin,
    Asinh,
    Cos,
    Cosh,
    Acos,
    Acosh,
    Tan,
    Tanh,
    Atan,
    Atanh,
    Atan2,
    Degrees,
    Radians,
}

impl From<TrigonometryFn> for Expr {
    fn from(x: TrigonometryFn) -> Self { Expr::BuiltinFn(super::BuiltinFn::Numeric(NumericFn::TrigonometryFn(x))) }
}

impl TypeCheck for TrigonometryFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        match self {
            TrigonometryFn::Sin |
            TrigonometryFn::Sinh |
            TrigonometryFn::Asin |
            TrigonometryFn::Asinh |
            TrigonometryFn::Cos |
            TrigonometryFn::Cosh |
            TrigonometryFn::Acos |
            TrigonometryFn::Acosh |
            TrigonometryFn::Tan |
            TrigonometryFn::Tanh |
            TrigonometryFn::Atan |
            TrigonometryFn::Atanh => sig! (
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n, t @ (F16 | F32 | F64))] => Vector(*n, *t),
            )(self, args),
            TrigonometryFn::Atan2 => sig! (
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n, t @ (F16 | F32 | F64)), Vector(n1, t1)] if same!(n n1; t t1) => Vector(*n, *t),
            )(self, args),
            TrigonometryFn::Degrees | TrigonometryFn::Radians => sig! (
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Vector(n, t @ (F16 | F32 | F64))] => Vector(*n, *t),
            )(self, args),
        }
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn sin    (&self) -> Any => [*self] TrigonometryFn::Sin;
        pub fn sinh   (&self) -> Any => [*self] TrigonometryFn::Sinh;
        pub fn asin   (&self) -> Any => [*self] TrigonometryFn::Asin;
        pub fn asinh  (&self) -> Any => [*self] TrigonometryFn::Asinh;
        pub fn cos    (&self) -> Any => [*self] TrigonometryFn::Cos;
        pub fn cosh   (&self) -> Any => [*self] TrigonometryFn::Cosh;
        pub fn acos   (&self) -> Any => [*self] TrigonometryFn::Acos;
        pub fn acosh  (&self) -> Any => [*self] TrigonometryFn::Acosh;
        pub fn tan    (&self) -> Any => [*self] TrigonometryFn::Tan;
        pub fn tanh   (&self) -> Any => [*self] TrigonometryFn::Tanh;
        pub fn atan   (&self) -> Any => [*self] TrigonometryFn::Atan;
        pub fn atanh  (&self) -> Any => [*self] TrigonometryFn::Atanh;
        pub fn atan2  (y: Any, x: Any) -> Any => [y, x] TrigonometryFn::Atan2;
        pub fn degrees(&self) -> Any => [*self] TrigonometryFn::Degrees;
        pub fn radians(&self) -> Any => [*self] TrigonometryFn::Radians;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearAlgebraFn {
    Cross,
    Determinant,
    Distance,
    Dot,
    FaceForward,
    InverseSqrt,
    Length,
    Normalize,
    Reflect,
    Refract,
    Transpose,
    Fma,
}

impl From<LinearAlgebraFn> for Expr {
    fn from(f: LinearAlgebraFn) -> Self { Expr::BuiltinFn(super::BuiltinFn::Numeric(NumericFn::LinearAlgebra(f))) }
}

impl TypeCheck for LinearAlgebraFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        match self {
            LinearAlgebraFn::Cross => {
                sig!([Vector(X3, t0), Vector(X3, t1)] if same!(t0 t1) && t0.is_floating_point() => Vector(X3, *t0))(
                    self, args,
                )
            }
            LinearAlgebraFn::Determinant => {
                sig!([Matrix(c, r, t0)] if c == r && t0.is_floating_point() => Vector(X1, (*t0).into()))(self, args)
            }
            LinearAlgebraFn::Distance => {
                sig!([Vector(n0, t0), Vector(n1, t1)] if same!(n0 n1; t0 t1) && t0.is_floating_point() => Vector(X1, *t0))(
                    self, args,
                )
            }
            LinearAlgebraFn::Dot => {
                sig!([v0 @ Vector(n0 @ (X2 | X3 | X4), t0), Vector(n1, t1)] if same!(n0 n1; t0 t1) && t0.is_numeric() => Vector(X1, *t0))(
                    self, args,
                )
            }
            LinearAlgebraFn::FaceForward => {
                sig!([Vector(n0 @ (X2 | X3 | X4), t0), Vector(n1, t1), Vector(n2, t2)] if same!(n0 n1 n2; t0 t1 t2) && t0.is_floating_point() => Vector(*n0, *t0))(
                    self, args,
                )
            }
            LinearAlgebraFn::InverseSqrt => sig!([Vector(n, t)] if t.is_floating_point() => Vector(*n, *t))(self, args),
            LinearAlgebraFn::Length => sig!([Vector(n, t)] if t.is_floating_point() => Vector(X1, *t))(self, args),
            LinearAlgebraFn::Normalize => {
                sig!([Vector(n @ (X2 | X3 | X4), t)] if t.is_floating_point() => Vector(*n, *t))(self, args)
            }
            LinearAlgebraFn::Reflect => {
                sig!([Vector(n0 @ (X2 | X3 | X4), t0), Vector(n1, t1)] if same!(n0 n1; t0 t1) && t0.is_floating_point() => Vector(*n0, *t0))(
                    self, args,
                )
            }
            LinearAlgebraFn::Refract => {
                sig!([Vector(n0 @ (X2 | X3 | X4), t0), Vector(n1, t1), Vector(X1, t2)] if same!(n0 n1; t0 t1 t2) && t0.is_floating_point() => Vector(*n0, *t0))(
                    self, args,
                )
            }
            LinearAlgebraFn::Transpose => {
                sig!([Matrix(c, r, t)] if t.is_floating_point() => Matrix(*r, *c, *t))(self, args)
            }
            LinearAlgebraFn::Fma => {
                sig!([Vector(n0, t0), Vector(n1, t1), Vector(n2, t2)] if same!(n0 n1 n2; t0 t1 t2) && t0.is_floating_point() => Vector(*n0, *t0))(
                    self, args,
                )
            }
        }
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn cross        (&self, rhs: Any) -> Any => [*self, rhs]            LinearAlgebraFn::Cross;
        pub fn determinant  (&self) -> Any => [*self]                           LinearAlgebraFn::Determinant;
        pub fn distance     (&self, other: Any) -> Any => [*self, other]        LinearAlgebraFn::Distance;
        pub fn dot          (&self, rhs: Any) -> Any => [*self, rhs]            LinearAlgebraFn::Dot;
        pub fn face_forward (&self, e2: Any, e3: Any) -> Any => [*self, e2, e3] LinearAlgebraFn::FaceForward;
        pub fn inverse_sqrt (&self) -> Any => [*self]                           LinearAlgebraFn::InverseSqrt;
        pub fn length       (&self) -> Any => [*self]                           LinearAlgebraFn::Length;
        pub fn normalize    (&self) -> Any => [*self]                           LinearAlgebraFn::Normalize;
        pub fn reflect      (self: Any, surface_orientation: Any) -> Any => [self, surface_orientation]  LinearAlgebraFn::Reflect;
        pub fn refract      (self: Any, normal: Any, ior_ratio: Any) -> Any => [self, normal, ior_ratio] LinearAlgebraFn::Refract;
        pub fn transpose    (&self) -> Any => [*self]                               LinearAlgebraFn::Transpose;
        pub fn fma          (&self, mul: Any, add: Any) -> Any => [*self, mul, add] LinearAlgebraFn::Fma;
    }

    #[doc(hidden)] // runtime api
    pub fn square_length(&self) -> Any { self.dot(*self) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscontinuityFn {
    Max,
    Min,
    Mix, // not actually a "Discontinuity"Fn
    Clamp,
    Floor,
    Ceil,
    Fract,
    Modf(ModfGenerics),
    Abs,
    Sign,
    Round,
    Saturate,
    Step,
    Smoothstep,
    Trunc,
    QuantizeToF16,
}

impl From<DiscontinuityFn> for Expr {
    fn from(f: DiscontinuityFn) -> Self { Expr::BuiltinFn(super::BuiltinFn::Numeric(NumericFn::Discontinuity(f))) }
}

impl TypeCheck for DiscontinuityFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        match self {
            DiscontinuityFn::Max | DiscontinuityFn::Min => sig!(
                [v0 @ Vector(n0, t0), v1 @ Vector(n1, t1)] if v0 == v1 && t0.is_numeric() => v0
            )(self, args),
            DiscontinuityFn::Mix => sig!(
                [v0 @ Vector(n0, t0), v1 @ Vector(n1, t1), Vector(X1, ta)] if same!(t0 t1 ta; v0 v1) && t0.is_floating_point() => v0,
                [v0 @ Vector(_, t0), v1 @ Vector(_, _), va @ Vector(_, _)] if same!(v0 v1 va) && t0.is_floating_point() => v0
            )(self, args),
            DiscontinuityFn::Clamp => sig!(
                [va @ Vector(_, t0), v0 @ Vector(_, _), v1 @ Vector(_, _)] if same!(va v0 v1) && t0.is_numeric() => v0
            )(self, args),
            DiscontinuityFn::Floor |
            DiscontinuityFn::Ceil |
            DiscontinuityFn::Fract |
            DiscontinuityFn::Round |
            DiscontinuityFn::Saturate |
            DiscontinuityFn::Trunc => sig!([v @ Vector(_, t)] if t.is_floating_point() => v)(self, args),
            DiscontinuityFn::Modf(modf_generics) => {
                BuiltinTemplateStructs::infer_type(args, ir::recording::TemplateStructParams::Modf(*modf_generics))
            }
            DiscontinuityFn::Sign => sig!([v @ Vector(_, I32 | F16 | F32 | F64)] => v)(self, args),
            DiscontinuityFn::Abs => sig!([v @ Vector(_, t)] if t.is_numeric() => v)(self, args),
            DiscontinuityFn::Step => {
                sig!([edge @ Vector(_, t), x @ Vector(_, _)] if edge == x && t.is_floating_point() => x)(self, args)
            }
            DiscontinuityFn::Smoothstep => sig!(
                [v0 @ Vector(_, t0), v1 @ Vector(_, _), va @ Vector(_, _)] if same!(v0 v1 va) && t0.is_floating_point() => v0
            )(self, args),
            DiscontinuityFn::QuantizeToF16 => sig!([v @ Vector(_, F32)] => v)(self, args),
        }
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn max        (&self, other: Any) -> Any => [*self, other] DiscontinuityFn::Max;
        pub fn min        (&self, other: Any) -> Any => [*self, other] DiscontinuityFn::Min;
        pub fn mix        (&self, a: Any, b: Any) -> Any => [a, b, *self] DiscontinuityFn::Mix;
        pub fn clamp      (&self, min: Any, max: Any) -> Any => [*self, min, max] DiscontinuityFn::Clamp;
        pub fn floor      (&self) -> Any => [*self] DiscontinuityFn::Floor;
        pub fn ceil       (&self) -> Any => [*self] DiscontinuityFn::Ceil;
        pub fn fract      (&self) -> Any => [*self] DiscontinuityFn::Fract;
        //pub fn modf       (&self) -> Any => [..] DiscontinuityFn::Modf; // returns exotic `__modf_result_vecN_abstract` type
        pub fn abs        (&self) -> Any => [*self] DiscontinuityFn::Abs;
        pub fn sign       (&self) -> Any => [*self] DiscontinuityFn::Sign;
        pub fn round_ties_even (&self) -> Any => [*self] DiscontinuityFn::Round;
        pub fn saturate   (&self) -> Any => [*self] DiscontinuityFn::Saturate;
        pub fn step       (&self, threshold: Any) -> Any => [threshold, *self] DiscontinuityFn::Step;
        pub fn smoothstep (&self, from: Any, to: Any) -> Any => [from, to, *self] DiscontinuityFn::Smoothstep;
        pub fn trunc      (&self) -> Any => [*self] DiscontinuityFn::Trunc;
        pub fn quantize_to_f16 (&self) -> Any => [*self] DiscontinuityFn::QuantizeToF16;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitFn {
    CountLeadingZeros,
    CountOneBits,
    CountTrailingZeros,
    ExtractBits,
    FirstLeadingBit,
    FirstTrailingBit,
    InsertBits,
    ReverseBits,
}

impl From<BitFn> for Expr {
    fn from(f: BitFn) -> Self { Expr::BuiltinFn(super::BuiltinFn::Numeric(NumericFn::Bit(f))) }
}

impl TypeCheck for BitFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        match self {
            BitFn::InsertBits => {
                sig!([t0 @ Vector(_, U32 | I32), t1, Vector(X1, U32), Vector(X1, U32)] if t0 == t1 => t0)(self, args)
            }
            BitFn::ReverseBits |
            BitFn::FirstLeadingBit |
            BitFn::FirstTrailingBit |
            BitFn::CountLeadingZeros |
            BitFn::CountOneBits |
            BitFn::CountTrailingZeros => sig!([v @ Vector(_, U32 | I32)] => v)(self, args),
            BitFn::ExtractBits => sig!([v @ Vector(_, U32 | I32), Vector(X1, U32), Vector(X1, U32)] => v)(self, args),
        }
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn count_leading_zeros  (&self) -> Any => [*self] BitFn::CountLeadingZeros;
        pub fn count_one_bits       (&self) -> Any => [*self] BitFn::CountOneBits;
        pub fn count_trailing_zeros (&self) -> Any => [*self] BitFn::CountTrailingZeros;
        pub fn extract_bits         (&self, offset: Any, count: Any) -> Any => [*self, offset, count] BitFn::ExtractBits;
        pub fn first_leading_bit    (&self) -> Any => [*self] BitFn::FirstLeadingBit;
        pub fn first_trailing_bit   (&self) -> Any => [*self] BitFn::FirstTrailingBit;
        pub fn insert_bits          (&self, newbits: Any, offset: Any, count: Any) -> Any => [*self, newbits, offset, count] BitFn::InsertBits;
        pub fn reverse_bits         (&self) -> Any => [*self] BitFn::ReverseBits;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExponentFn {
    Exp,
    Exp2,
    Log,
    Log2,
    Frexp(FrexpGenerics),
    Ldexp,
    Pow,
    Sqrt,
}

impl From<ExponentFn> for Expr {
    fn from(f: ExponentFn) -> Self { Expr::BuiltinFn(super::BuiltinFn::Numeric(NumericFn::Exponent(f))) }
}

impl TypeCheck for ExponentFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        match self {
            ExponentFn::Sqrt | ExponentFn::Exp | ExponentFn::Exp2 | ExponentFn::Log | ExponentFn::Log2 => {
                sig!([v @ Vector(_, t)] if t.is_floating_point() => v)(self, args)
            }
            ExponentFn::Pow => {
                sig!([v0 @ Vector(_, t), v1 @ Vector(_, _)] if same!(v0 v1) && t.is_floating_point() => v0)(self, args)
            }
            ExponentFn::Frexp(frexp_generics) => {
                BuiltinTemplateStructs::infer_type(args, ir::recording::TemplateStructParams::Frexp(*frexp_generics))
            }
            ExponentFn::Ldexp => {
                sig!([t @ Vector(n0, s), i @ Vector(n1, I32)] if same!(n0 n1) && s.is_floating_point() => t)(self, args)
            }
        }
    }
}

impl Display for ExponentFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExponentFn::Exp |
            ExponentFn::Exp2 |
            ExponentFn::Log |
            ExponentFn::Log2 |
            ExponentFn::Ldexp |
            ExponentFn::Pow |
            ExponentFn::Sqrt => write!(f, "{self:?}"),
            ExponentFn::Frexp(FrexpGenerics(fp, len)) => {
                write!(f, "Frexp<{}{}>", ScalarType::from(*fp), len)
            }
        }
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn exp  (&self) -> Any => [*self] ExponentFn::Exp;
        pub fn exp2 (&self) -> Any => [*self] ExponentFn::Exp2;
        pub fn log  (&self) -> Any => [*self] ExponentFn::Log;
        pub fn log2 (&self) -> Any => [*self] ExponentFn::Log2;
        //pub fn frexp(&self) -> Any => [*self] ExponentFn::Frexp; // exotic return type, not implemented yet
        pub fn ldexp(self, e2: Any) -> Any => [self, e2] ExponentFn::Ldexp;
        pub fn pow  (&self, exponent: Any) -> Any => [*self, exponent] ExponentFn::Pow;
        pub fn sqrt (&self) -> Any => [*self] ExponentFn::Sqrt;
    }
}
