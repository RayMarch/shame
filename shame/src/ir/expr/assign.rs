use std::ops::*;

use super::{type_check::SigFormatting, Expr, NoMatchingSignature, TypeCheck};
use crate::frontend::any::Any;
use crate::{
    call_info,
    frontend::any::record_node,
    frontend::error::InternalError,
    impl_track_caller_fn_any,
    ir::{
        ir_type::{
            AccessMode, AddressSpace, Indirection,
            Len::*,
            Len2,
            ScalarType::{self, *},
            SizedType::*,
            StoreType::*,
            Type::Unit,
        },
        recording::{Context, InteractionKind, MemoryInteractionEvent, MemoryRegion},
    },
};
use crate::{ir, ir::ir_type::StoreType, ir::Type, same, sig};

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)] // runtime api
pub enum CompoundOp {
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    AndAssign,
    OrAssign,
    XorAssign,
    ShrAssign,
    ShlAssign,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assign {
    /// a = b
    Assign,
    /// `a $= b`, where `$` is one of `+, -, *, /, %, &, |, ^, >>, <<`
    CompoundAssignment(CompoundOp),
    /// i++
    // (no test case yet)
    Increment,
    /// i--
    // (no test case yet)
    Decrement,
}

impl std::fmt::Display for Assign {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Assign::Assign => "assign (=)",
            Assign::CompoundAssignment(op) => match op {
                CompoundOp::AddAssign => "+=",
                CompoundOp::SubAssign => "-=",
                CompoundOp::MulAssign => "*=",
                CompoundOp::DivAssign => "/=",
                CompoundOp::RemAssign => "%=",
                CompoundOp::AndAssign => "&=",
                CompoundOp::OrAssign => "|=",
                CompoundOp::XorAssign => "^=",
                CompoundOp::ShrAssign => ">>=",
                CompoundOp::ShlAssign => "<<=",
            },
            Assign::Increment => "++",
            Assign::Decrement => "--",
        };
        write!(f, "{str}")
    }
}

impl From<Assign> for Expr {
    fn from(x: Assign) -> Self { Expr::Assign(x) }
}

impl TypeCheck for Assign {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        use AccessMode::*;
        use AddressSpace::*;
        (match self {
            Assign::Assign => sig! (
                [
                    Type::Ref(allocation, t0, Write | ReadWrite),
                    Type::Store(t1)
                ]
                if allocation.is_writeable() && t0 == t1 => Unit
            )(self, args),
            Assign::CompoundAssignment(op) => op.infer_type(args),
            // https://www.w3.org/TR/WGSL/#increment-decrement
            Assign::Increment | Assign::Decrement => sig!(
                [Type::Ref(_, Sized(Vector(X1, t)), ReadWrite)] if t.is_integer() => Unit
            )(self, args),
        })
    }
}

impl TypeCheck for CompoundOp {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        use AccessMode::*;
        match self {
            CompoundOp::AddAssign |
            CompoundOp::SubAssign |
            CompoundOp::MulAssign |
            CompoundOp::DivAssign |
            CompoundOp::RemAssign => sig! (
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [
                    Type::Ref(allocation, Sized(Vector(n0, t0)), ReadWrite),
                    Type::Store(Sized(Vector(n1, t1)))
                ]
                if allocation.is_writeable() && t0.is_numeric() && same!(n0 n1; t0 t1) => Unit
            )(self, args),
            CompoundOp::AndAssign | CompoundOp::OrAssign | CompoundOp::XorAssign => sig! (
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [
                    Type::Ref(allocation, Sized(Vector(n0, t0)), ReadWrite),
                    Type::Store(Sized(Vector(n1, t1)))
                ]
                if allocation.is_writeable() && t0.is_integer() && same!(n0 n1; t0 t1) => Unit
            )(self, args),
            CompoundOp::ShrAssign | CompoundOp::ShlAssign => sig! (
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [
                    Type::Ref(allocation, Sized(Vector(n0, t)), ReadWrite),
                    Type::Store(Sized(Vector(n1, U32)))
                ]
                if allocation.is_writeable() && t.is_integer() && same!(n0 n1) => Unit
            )(self, args),
        }
    }
}

macro_rules! impl_any_assign_operators {
    ($($AddAssign: ident, $add_assign: ident, $op: expr;)*) => {
        $(impl $AddAssign for Any {
            #[track_caller]
            fn $add_assign(&mut self, rhs: Any) {
                let compound_assign = $crate::frontend::any::record_node($crate::call_info!(), $crate::ir::expr::Expr::Assign(Assign::CompoundAssignment($op)), &[*self, rhs]);
                MemoryRegion::record_interaction(compound_assign, *self, InteractionKind::ReadWrite);
            }
        })*
    };
}

impl_any_assign_operators! {
    AddAssign, add_assign, CompoundOp::AddAssign;
    SubAssign, sub_assign, CompoundOp::SubAssign;
    MulAssign, mul_assign, CompoundOp::MulAssign;
    DivAssign, div_assign, CompoundOp::DivAssign;
    RemAssign, rem_assign, CompoundOp::RemAssign;
    BitAndAssign, bitand_assign, CompoundOp::AndAssign;
    BitOrAssign, bitor_assign, CompoundOp::OrAssign;
    BitXorAssign, bitxor_assign, CompoundOp::XorAssign;
    ShrAssign, shr_assign, CompoundOp::ShrAssign;
    ShlAssign, shl_assign, CompoundOp::ShlAssign;
}

impl Any {
    #[doc(hidden)] // runtime api
    pub fn increment(&self) {
        let ref_ = *self;
        let inc = record_node(call_info!(), Assign::Increment.into(), &[ref_]);
        MemoryRegion::record_interaction(inc, ref_, InteractionKind::ReadWrite);
    }

    #[doc(hidden)] // runtime api
    pub fn decrement(&self) {
        let ref_ = *self;
        let inc = record_node(call_info!(), Assign::Increment.into(), &[ref_]);
        MemoryRegion::record_interaction(inc, ref_, InteractionKind::ReadWrite);
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn set(&self, rhs: Any) {
        let ref_ = *self;
        let assign = record_node(call_info!(), Assign::Assign.into(), &[ref_, rhs]);
        MemoryRegion::record_interaction(assign, ref_, InteractionKind::Write);
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn ref_load(&self) -> Any {
        let ref_ = *self;
        let value = record_node(call_info!(), Expr::RefLoad(RefLoad), &[ref_]);
        MemoryRegion::record_interaction(value, ref_, InteractionKind::Read);
        value
    }
}

/// WGSL: `let x = reference` or "The Load Rule"
/// see https://www.w3.org/TR/WGSL/#ref-ptr-use-case
///
/// reads the current value of the memory cell that `Ref` points to and returns that.
/// this directly corresponds to either `let` bindings or the "load-rule" in WGSL (https://www.w3.org/TR/WGSL/#load-rule).
///
/// contrary to [`CopyAssign::CopyAssign`], this only accepts a `Ref` argument, and it shows the
/// intent of "dereferencing" as opposed to "copying"
/// > note: I am considering merging these two variants in the future,
/// > but for now i want to be able to tell them apart when debugging expr graphs.
///
/// In WGSL loads are always implicitly applied in `let out = arg` or function argument contexts (= Load Rule).
/// In `shame` on the other hand, this is explicit (`shame::Ref::get`).
///
/// It is important that this expression exists especially in the context of
/// array access chains:
///
/// ```ignore
/// var i = 0;
/// let k: ref<u32> = foo[i] //(*) the `ref<u32>` type cannot be expressed in wgsl code
/// i += 1; // <- change of execution state
/// k = 4;
/// ```
/// the above code cannot be generated since the `ref<...>` type cannot be written
/// down in wgsl
///
/// if `shame` naively just treated the load-rule as implicit and stored `k` as an access chain, which will never be turned into a statement,
/// the following erroneous code would be generated:
/// ```ignore
/// var i = 0;
/// // (*) (nothing here because `ref<...>` cannot be written down)
/// i += 1; // <- change of execution state
/// foo[i] = 4; // access chain references the `i` mutable variable, which has changed since!! so this is the wrong index!
/// ```
/// instead, `shame` must emit a `RefLoad` expression at the (*) line, so that an additional `let` binding is generated:
/// ```ignore
/// var i = 0;
/// let i_ = 0; // (*) RefLoad expr turned into a `let` binding
/// i += 1; // <- change of execution state
/// foo[i_] = 4; // access chain references the `i_` value, not the mutable variable. This is correct.
/// ```
/// In `shame`, the `RefLoad` expr is emitted when calling `shame::Ref::get`, therefore loads are *explicit*.
/// That way `i_` is generated and the generated code is correct.
/// To reduce verbosity in the generated shader, such a `let` binding only needs to be
/// inserted if the execution state between an expression using a RefLoad argument, and that RefLoad argument itself is different.
///
/// for completeness sake, here's the corresponding `shame` code:
///
/// ```rust
/// let foo = sm::Cell::new([0; 10]);
/// let i = sm::Cell::new(0i32);
/// let k: Ref<sm::i32x1> = foo.at(i.get()); //(*) .get() emits `RefLoad`
/// i.set_add(1);
/// k.set(4);
/// ```
///
/// > note: As a consequence, `shame` should not
/// > auto-apply the load-rule (like in wgsl) in its type erased `Any` api
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefLoad;

impl TypeCheck for RefLoad {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        use AccessMode::*;
        sig!(
            { fmt: SigFormatting::RemoveAsterisksAndClone, },
            [Type::Ref(_, t, Read | ReadWrite)] => t,
        )(self, args)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// function that prevents dead code elimination (useful for testing)
pub struct Show;

impl TypeCheck for Show {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        sig!(
            [t @ Type::Store(_)] => Type::Unit,
            [t @ Type::Ptr(..)] => Type::Unit,
        )(self, args)
    }
}

impl Any {
    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn show(&self) -> Any { record_node(call_info!(), Expr::Show(Show), &[*self]) }
}
