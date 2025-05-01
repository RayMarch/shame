use crate::{
    common::pool::PoolRef,
    ir::{
        recording::{AllocStmt, Block, ExprStmt, MemoryRegion, Stmt, TimeInstant},
        CallInfo, Node,
    },
};

use super::{PossibleStages, StageMask};

/// trait exclusively used by the stage solver to create error messages among other things
///
/// implemented by elements whose shader stage appearance/visibility is decided by the stage solver
pub trait InferStage {
    fn possible_stages(&self) -> &PossibleStages;

    fn call_info(&self) -> CallInfo;

    /// the name of this shader element as used in a stage solver error
    fn write_name(&self, f: &mut dyn std::fmt::Write) -> Result<(), std::fmt::Error>;

    fn name_string(&self) -> String {
        let mut s = String::new();
        self.write_name(&mut s);
        s
    }

    /// the verb associated with `Self` prefixed with "be", e.g. Nodes can "be evaluated", MemoryRegions can "be accessed" etc.
    fn be_verb(&self) -> &'static str;

    /// a short written representation of the `can_appear_in` mask
    fn write_can_only_appear_in(&self, f: &mut dyn std::fmt::Write) -> Result<(), std::fmt::Error> {
        let can = self.possible_stages().can_appear_in();

        let be_evaluated = self.be_verb();

        if can.is_empty() {
            write!(f, "cannot {be_evaluated} in any shader stage")?;
            return Ok(());
        }

        match can.get_only_stage() {
            Some(stage) => match stage {
                super::ShaderStage::Vert => write!(f, "can only {be_evaluated} per-vertex")?,
                super::ShaderStage::Frag => write!(f, "can only {be_evaluated} per-fragment")?,
                _ => write!(f, "can only {be_evaluated} in the {stage} stage")?,
            },
            None => {
                if can == StageMask::pipeline_render() {
                    write!(f, "can only {be_evaluated} as either per-vertex or per-fragment")?;
                } else {
                    write!(f, "can only appear in the ")?;
                    for (i, stage) in can.into_iter().enumerate() {
                        write!(f, "{stage}")?;
                        if i != 0 {
                            write!(f, "/")?;
                        }
                    }
                    write!(f, " stages")?;
                }
            }
        }
        Ok(())
    }

    /// a short written representation of the `must_appear_in` mask
    fn write_must_appear_in(&self, f: &mut dyn std::fmt::Write) -> Result<(), std::fmt::Error> {
        let must = self.possible_stages().must_appear_in();

        let be_evaluated = self.be_verb();

        if must.is_empty() {
            write!(f, "doesn't have to {be_evaluated} in any shader stage")?;
            return Ok(());
        }

        match must.get_only_stage() {
            Some(stage) => match stage {
                super::ShaderStage::Vert => write!(f, "must {be_evaluated} per-vertex")?,
                super::ShaderStage::Frag => write!(f, "must {be_evaluated} per-fragment")?,
                _ => write!(f, "must {be_evaluated} in the {stage} stage")?,
            },
            None => {
                if must == StageMask::pipeline_render() {
                    write!(f, "must {be_evaluated} both per-vertex and per-fragment")?;
                } else {
                    write!(f, "must {be_evaluated} in the ")?;
                    for (i, stage) in must.into_iter().enumerate() {
                        write!(f, "{stage}")?;
                        if i != 0 {
                            write!(f, " and ")?;
                        }
                    }
                    write!(f, " stage")?;
                }
            }
        }
        Ok(())
    }
}

impl InferStage for Node {
    fn possible_stages(&self) -> &PossibleStages { &self.stages }

    fn call_info(&self) -> CallInfo { self.call_info }

    fn be_verb(&self) -> &'static str { "be evaluated" }

    #[allow(clippy::collapsible_match)]
    fn write_name(&self, f: &mut dyn std::fmt::Write) -> Result<(), std::fmt::Error> {
        use crate::ir::expr::Expr;
        let mut fallback = |f: &mut dyn std::fmt::Write| write!(f, "expression `{}`", self.expr);
        match &self.expr {
            Expr::ShaderIo(shader_io) => match shader_io {
                crate::ir::expr::ShaderIo::Builtin(builtin_shader_io) => match builtin_shader_io {
                    crate::ir::expr::BuiltinShaderIo::Get(s_in) => write!(f, "value `{:?}`", s_in),
                    crate::ir::expr::BuiltinShaderIo::Set(s_out) => write!(f, "variable `{:?}`", s_out),
                },
                crate::ir::expr::ShaderIo::Interpolate(location) => write!(f, "fill-source #{location}"),
                crate::ir::expr::ShaderIo::GetInterpolated(location) => write!(f, "fill-result #{location}"),
                crate::ir::expr::ShaderIo::GetVertexInput(location) => write!(f, "vertex attribute #{location}"),
                crate::ir::expr::ShaderIo::WriteToColorTarget { slot } => write!(f, "color target #{slot}"),
            },
            // Expr::VarIdent(var_ident) => (),
            // Expr::Assign(assign) => (),
            // Expr::PipelineIo(pipeline_io) => (),
            // Expr::RefLoad(ref_load) => (),
            // Expr::Literal(literal) => (),
            // Expr::Operator(operator) => (),
            // Expr::FnRelated(fn_related) => (),
            // Expr::Decomposition(decomposition) =>  (),
            Expr::BuiltinFn(builtin_fn) => match builtin_fn {
                crate::ir::expr::BuiltinFn::Derivative(derivative_fn) => match derivative_fn {
                    crate::ir::expr::DerivativeFn::Dpdx(_) => write!(f, "fragment-quad gradient (dpdx)"),
                    crate::ir::expr::DerivativeFn::Dpdy(_) => write!(f, "fragment-quad gradient (dpdy)"),
                    crate::ir::expr::DerivativeFn::Fwidth(_) => write!(f, "dxy_manhattan"),
                },
                _ => fallback(f),
            },
            // Expr::Show(show) => todo!(),
            _ => fallback(f),
        }
    }
}

impl InferStage for MemoryRegion {
    fn possible_stages(&self) -> &PossibleStages { &self.stages }

    fn call_info(&self) -> CallInfo { self.call_info }

    fn be_verb(&self) -> &'static str { "be accessed" }

    fn write_name(&self, f: &mut dyn std::fmt::Write) -> Result<(), std::fmt::Error> {
        write!(f, "memory region `{:?}`", self)
    }
}

impl InferStage for Block {
    fn possible_stages(&self) -> &PossibleStages { &self.stages }

    fn call_info(&self) -> CallInfo { self.call_info }

    fn be_verb(&self) -> &'static str { "be executed" }

    fn write_name(&self, f: &mut dyn std::fmt::Write) -> Result<(), std::fmt::Error> {
        write!(f, "block of type `{:?}`", self.kind)
    }
}

impl InferStage for StmtInfo<'_> {
    fn possible_stages(&self) -> &PossibleStages {
        match &self.kind {
            // expr statements do not have their own stage tracking, as their nodes already cover that.
            Stmt::Expr(stmt) => match stmt {
                ExprStmt::Expr(n) | ExprStmt::IntroduceIdent(n) | ExprStmt::Condition(n) => {
                    self.nodes[*n].possible_stages()
                }
            },
            Stmt::Allocate(AllocStmt {
                allocation,
                initial_value: init,
            }) => &allocation.stages,
            // flow statements such as `discard` and control-flow structures have explicit stage tracking
            Stmt::Flow(flow_stmt, stages) => stages,
        }
    }

    fn be_verb(&self) -> &'static str { "be executed" }

    fn call_info(&self) -> CallInfo { self.call_info }

    fn write_name(&self, f: &mut dyn std::fmt::Write) -> Result<(), std::fmt::Error> {
        write!(f, "statement `{}`", self.kind)
    }
}

pub struct StmtInfo<'a> {
    pub kind: &'a Stmt,
    pub call_info: CallInfo,
    pub nodes: &'a PoolRef<'a, Node>,
}
