mod alloc;
mod assign;
mod builtin_fn;
mod builtin_numeric_fn;
mod builtin_texture_fn;
mod decomposition;
mod fn_related;
mod literal;
mod operation;
mod operator;
mod pipeline_io_expr;
mod shader_io_expr;
pub(crate) mod type_check;

use std::fmt::Display;

use crate::{impl_track_caller_fn_any, ir::ir_type, ir::Type, sig, try_ctx_track_caller};
pub use alloc::*;
pub use assign::*;
pub use builtin_fn::*;
pub use builtin_numeric_fn::*;
pub use builtin_texture_fn::*;
pub use decomposition::*;
pub use fn_related::*;
pub use literal::*;
pub use operator::*;
pub use pipeline_io_expr::*;
pub use shader_io_expr::*;
pub use type_check::NoMatchingSignature;
pub use type_check::TypeCheck;

use super::pipeline::PipelineKind;
use super::pipeline::PossibleStages;
use super::pipeline::StageMask;
use super::recording::Stmt;


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// using an identifier of a variable associated with a memory region (variable)
    VarIdent(VarIdent),
    /// assignments are not expressions in WGSL, but they are in `shame`
    Assign(Assign),
    /// read/write to builtin variables and other shader stage io
    ShaderIo(ShaderIo),
    /// read/write to pipeline io (bind-groups/bindings)
    PipelineIo(PipelineIo),
    RefLoad(RefLoad),
    Literal(Literal),
    /// non-assignment operators (assignments are in the `Assign` variant)
    Operator(Operator),
    // Expr::Identifier // present in the WGSL spec, but in this library, values with associated identifiers are represented in the `Node` struct
    FnRelated(FnRelated),
    Decomposition(Decomposition),
    BuiltinFn(BuiltinFn),
    Show(Show),
}

impl TypeCheck for Expr {
    #[rustfmt::skip]
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        use ir_type::{Len::*, StoreType::*, *};
        match self {
            Expr::VarIdent      (x) => x.infer_type(args),
            Expr::Assign        (x) => x.infer_type(args),
            Expr::ShaderIo      (x) => x.infer_type(args),
            Expr::PipelineIo    (x) => x.infer_type(args),
            Expr::RefLoad       (x) => x.infer_type(args),
            Expr::Literal       (x) => x.infer_type(args),
            Expr::Operator      (x) => x.infer_type(args),
            Expr::FnRelated     (x) => x.infer_type(args),
            Expr::Decomposition (x) => x.infer_type(args),
            Expr::BuiltinFn     (x) => x.infer_type(args),
            Expr::Show    (x) => x.infer_type(args),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExprCategory {
    ChangesExecutionState,
    /// Values that can be 'let' bound, but either
    ///
    /// - the language backends already define
    ///   them as idents (think GLSL `gl_VertexID` global ident) and so it would
    ///   just lead to less readable code if we added another `let s_0 = gl_VertexId`
    ///   ident-binding.
    ///
    /// - they are subjectively considered noisy and inexpensive, such as `vec.x` access
    ///   which leads to more readable code even if repeated. For example we
    ///   prefer
    ///   - `something(s_0.x, s_0.x)` over
    ///   - `let s_1 = s_0.x; something(s_1, s_1)`.
    ///
    /// note: this is only a hint. the statement emitter is free to ignore this if it makes reasoning about correctness easier.
    BindToIdentUndesirable,
    /// none of the above
    Normal,
}

impl Expr {
    pub fn classify(&self) -> ExprCategory {
        match self {
            Expr::VarIdent(_) => ExprCategory::Normal, // a ref will never be bound to ident
            Expr::Assign(x) => ExprCategory::ChangesExecutionState,
            Expr::PipelineIo(f) => ExprCategory::BindToIdentUndesirable,
            Expr::ShaderIo(x) => match x.may_change_execution_state() {
                true => ExprCategory::ChangesExecutionState,
                false => ExprCategory::BindToIdentUndesirable,
            },
            Expr::RefLoad(_) => ExprCategory::Normal,
            Expr::Literal(_) => ExprCategory::Normal,
            Expr::Operator(_) => ExprCategory::Normal,
            Expr::FnRelated(FnRelated::Call(_)) => ExprCategory::ChangesExecutionState,
            Expr::FnRelated(FnRelated::FnParamMemoryView(mem, pass_as)) => match pass_as {
                ArgViewKind::Ref => ExprCategory::Normal, // a ref will never be bound to ident
                ArgViewKind::Ptr => ExprCategory::Normal,
            },
            Expr::FnRelated(FnRelated::FnParamValue(..)) => ExprCategory::Normal,
            Expr::Decomposition(d) => match d {
                Decomposition::VectorAccess(a) => match a {
                    VectorAccess::Swizzle1(_) => ExprCategory::BindToIdentUndesirable,
                    VectorAccess::Swizzle2(_) | VectorAccess::Swizzle3(_) | VectorAccess::Swizzle4(_) => {
                        ExprCategory::Normal
                    }
                },
                Decomposition::StructureAccess(canon_name) => ExprCategory::BindToIdentUndesirable,
                Decomposition::VectorIndex |
                Decomposition::VectorIndexConst(_) |
                Decomposition::MatrixIndex |
                Decomposition::MatrixIndexConst(_) |
                Decomposition::ArrayIndex |
                Decomposition::ArrayIndexConst(_) => ExprCategory::Normal,
            },
            Expr::BuiltinFn(f) => match f.may_change_execution_state() {
                true => ExprCategory::ChangesExecutionState,
                false => ExprCategory::Normal,
            },
            Expr::Show(Show) => ExprCategory::Normal,
        }
    }

    /// whether the expression can possibly change the `ExecutionState`.
    /// this is true for expressions such as mutating functions,
    /// synchronization functions (barriers), Assignment operators,...
    ///
    /// TODO(release) remove this function, replace the call sites with `classify`
    pub fn may_change_execution_state(&self) -> bool { self.classify() == ExprCategory::ChangesExecutionState }

    #[rustfmt::skip]
    pub fn possible_stages(&self) -> PossibleStages {
        let none = StageMask::empty();
        let all = StageMask::all();
        let stages = match self {
            Expr::ShaderIo(x) => x.possible_stages(),
            Expr::PipelineIo(x) => x.possible_stages(),
            Expr::BuiltinFn(x) => x.possible_stages(),
            Expr::Assign(_) => PossibleStages::new(true, none, all, false),
            Expr::VarIdent(_) |
            Expr::Literal(_) |
            Expr::Operator(_) |
            Expr::FnRelated(_) |
            Expr::Decomposition(_) => PossibleStages::all(),
            Expr::RefLoad(_) => PossibleStages::all(),
            Expr::Show(must_appear) => PossibleStages::new(true, none, all, false),
        };

        #[cfg(debug_assertions)]
        if self.classify() == ExprCategory::ChangesExecutionState {
            // TODO(low prio): this requirement may not be strictly true for e.g. Expr::Assign(...) but
            // for now we pessimistically require them to be written in the shader.
            // consider relaxing this once we have a better test suite
            if !stages.must_appear_at_all() {
                println!(
                    "shame-rs internal non-fatal error (please report): execution state changes of expressions should coincide with 'must_appear_at_all' requirement. \
                    expr {self:?} changes execution state, but does not have 'must_appear_at_all' flag"
                );
            }
        }
        stages
    }
}

impl Display for Expr {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::VarIdent(x)      => write!(f, "{x}"),
            Expr::Assign(x)        => write!(f, "{x}"),
            Expr::ShaderIo(x)      => write!(f, "{x}"),
            Expr::PipelineIo(x)    => write!(f, "{x}"),
            Expr::RefLoad(x)       => write!(f, "ref.get"),
            Expr::Literal(x)       => write!(f, "{x}"),
            Expr::Operator(x)      => write!(f, "{x}"),
            Expr::FnRelated(x)     => write!(f, "{x}"),
            Expr::Decomposition(x) => write!(f, "{x}"),
            Expr::BuiltinFn(x)     => write!(f, "{x}"),
            Expr::Show(x)    => write!(f, "must-appear"),
        }
    }
}
