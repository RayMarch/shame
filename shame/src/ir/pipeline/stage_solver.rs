use std::{
    borrow::BorrowMut,
    cell::{Cell, RefCell},
    collections::BTreeSet,
    fmt::{Display, Write},
    rc::Rc,
};

use thiserror::Error;

use crate::ir::{self, pipeline::PipelineKind};
use crate::{
    common::{pool::Key, prettify::set_color},
    frontend::encoding::pipeline_kind,
    ir::{
        expr::{Expr, FnRelated},
        pipeline::StageMask,
        recording::{
            AllocStmt, Block, BlockKind, CallInfo, Context, ExprStmt, FlowStmt, FunctionDef, Jump,
            MemoryInteractionEvent, MemoryRegion, Stmt,
        },
        AccessMode, AddressSpace, Node, Type,
    },
};

use super::{
    infer_stage::{InferStage, StmtInfo},
    ShaderStage,
};

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub struct StageSolverError {
    pub call_info: CallInfo,
    pub kind: StageSolverErrorKind,
}

impl Display for StageSolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.call_info, self.kind)
    }
}

const FALLBACK_ERROR_SUFFIX: &str = "\n\n(note: this is an unrefined fallback error message, please report how this error was triggered so we can improve this message)";

#[allow(clippy::enum_variant_names)]
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum StageSolverErrorKind {
    #[error("{element_name} cannot appear in required stages: `{}`{FALLBACK_ERROR_SUFFIX}", stages.to_string_verbose())]
    CannotAppearInRequiredStages { element_name: String, stages: StageMask },
    #[error(
        "cannot determine shader stage for {element_name}. This can for example happen, if it depends on values or control flow that requires both per-vertex and per-fragment evaluation.{FALLBACK_ERROR_SUFFIX}"
    )]
    CannotAppearInAnyStage { element_name: String },
    #[error("{element_name} cannot appear in more than one shader stage, but its usage requires that it appears in stages: {}.", stages.to_string_verbose())]
    CannotAppearInMultipleStages { element_name: String, stages: StageMask },
    #[error(
        "{0}\n\n(note: Most shader-stage errors happen when a per-vertex object is used in a per-fragment computation. In those cases, the per-vertex value must first be interpolated via the `fragment.fill` family of functions.)"
    )]
    Propagation(String),
}

#[allow(clippy::enum_variant_names)]
/// intermediate error type that gest converted to StageSolverErrorKind
pub enum CheckOkError {
    CannotAppearInRequiredStages { stages: StageMask },
    CannotAppearInAnyStage,
    CannotAppearInMultipleStages { stages: StageMask },
}

impl CheckOkError {
    fn with_name(self, element_name: String) -> StageSolverErrorKind {
        use CheckOkError as A;
        use StageSolverErrorKind as B;
        match self {
            A::CannotAppearInRequiredStages { stages } => B::CannotAppearInRequiredStages { element_name, stages },
            A::CannotAppearInAnyStage => B::CannotAppearInAnyStage { element_name },
            A::CannotAppearInMultipleStages { stages } => B::CannotAppearInMultipleStages { element_name, stages },
        }
    }
}

/// two-argument based stage requirement propagation, first argument is "left", second is "right"
enum Propagation {
    /// asymmetrical propagation of stage information (right variant)
    ///
    /// naming inside the function uses `dependent` and `arg` as an example, but this function is applied
    /// to other situations as well
    ///
    /// "left makes right appear" so the dead code elimination flows to the right (r)
    AsymR,
    /// asymmetrical propagation of stage information (left variant)
    ///
    /// flipped argument order of `AsymR`
    ///
    /// "right makes left appear" so the dead code elimination flows to the left (l)
    AsymL, // TODO(release): since asym R is just asym L flipped, we can get rid of one of them entirely by flipping args
    Sym,
}

/// Mark an error emission as optional
///
/// the general idea here is to cherry pick individual error situations and emit
/// a more descriptive error message than the more general error handling later would provide.
/// Removing this error emission (by setting this to false) is safe, as it would only impact the quality of error messages,
/// and not cause malformed programs to get accepted
const NICER_ERRORS: bool = true;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PossibleStages(RefCell<Inner>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inner {
    /// whether it must appear at all, or can be left out in codegen.
    ///
    /// when this is `true` and `must_appear_in` is `empty` then it means it wasn't
    /// figured out which stage it is going to show up in yet.
    ///
    /// for example, writes to a storage buffer must appear in the shader, but it
    /// is not yet decided which shader stage it will show up in.
    pub must_appear_at_all: bool,
    /// lower bound of stages, may only grow (`bitor` operator)
    pub must_appear_in: StageMask,
    /// upper bound of stages, may only shrink (`bitand` operator)
    pub can_appear_in: StageMask,

    /// whether it should not appear in multiple shader stages, for example
    /// a texture load operation may be considered so expensive that emitting
    /// it in the vertex and fragment shader is considered misleading.
    pub cannot_appear_twice: bool,
}

impl Display for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}|{}",
            match self.must_appear_at_all {
                true => '!',
                false => ' ',
            },
            {
                let d: &dyn std::fmt::Display = match self.must_appear_in {
                    x if x == StageMask::empty() => &" -",
                    x if x == StageMask::vert() => &"v ",
                    x if x == StageMask::frag() => &" f",
                    x if x == StageMask::comp() => &" c",
                    x if x == StageMask::frag() | StageMask::vert() => &"vf",
                    _ => &self.must_appear_in,
                };
                d
            },
            {
                let d: &dyn std::fmt::Display = match self.can_appear_in {
                    x if x == StageMask::empty() => &"- ",
                    x if x == StageMask::vert() => &"v ",
                    x if x == StageMask::frag() => &" f",
                    x if x == StageMask::comp() => &" c",
                    x if x == StageMask::frag() | StageMask::vert() => &"vf",
                    _ => &self.can_appear_in,
                };
                d
            }
        )?;
        let symbols = match (self.is_settled(), self.cannot_appear(), self.check_ok().is_err()) {
            (_, _, true) => " ↯",
            (true, true, _) => "∅✔",
            (true, _, _) => " ✔",
            (_, true, _) => "∅ ",
            (false, false, false) => "  ",
        };
        f.write_str(symbols);
        Ok(())
    }
}

impl Display for PossibleStages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.try_borrow() {
            Ok(inner) => write!(f, "{inner}"),
            Err(_) => write!(f, "(PossibleStages already borrowed)"),
        }
    }
}

impl Inner {
    pub fn check_ok(&self) -> Result<(), CheckOkError> {
        // TODO(release) depending on the error quality, consider switching the order of these. maybe StageContradiction is easier to understand, even if it also CannotAppear

        if self.can_appear_in.is_empty() {
            // we require all elements to be able to appear, even if they don't `must_appear_at_all`, otherwise errors would pop up just because unused variables are used
            return Err(CheckOkError::CannotAppearInAnyStage);
        }

        let stages_it_must_appear_in_but_cannot = (!self.can_appear_in & self.must_appear_in);
        if stages_it_must_appear_in_but_cannot != StageMask::empty() {
            return Err(CheckOkError::CannotAppearInRequiredStages {
                stages: stages_it_must_appear_in_but_cannot,
            });
        }

        let num_stages_it_must_appear_in = self.must_appear_in.count_ones();
        if num_stages_it_must_appear_in > 1 && self.cannot_appear_twice {
            return Err(CheckOkError::CannotAppearInMultipleStages {
                stages: self.must_appear_in,
            });
        }
        Ok(())
    }

    /// whether the only shader stages that it must appear in are also the only shader stages it can appear in
    pub fn is_settled(&self) -> bool { self.can_appear_in == self.must_appear_in }

    pub fn cannot_appear(&self) -> bool { self.can_appear_in.is_empty() }

    /// whether this element must appear in one of the stages that are set in `filter`
    pub fn is_present(&self, filter: StageMask) -> bool { (self.must_appear_in & filter) != StageMask::empty() }

    /// if `self.must_appear_at_all == true` and has only one stage in `self.can_appear_in`,
    /// that stage gets included in `must_appear_in`
    pub fn confirm_stage_if_no_alternatives(&mut self) {
        if self.must_appear_at_all && self.can_appear_in.count_ones() == 1 {
            self.must_appear_in |= self.can_appear_in
        }
    }
}

impl PossibleStages {
    pub fn new(
        must_appear_at_all: bool,
        must_appear_in: StageMask,
        can_appear_in: StageMask,
        cannot_appear_twice: bool,
    ) -> Self {
        if !must_appear_in.is_empty() {
            debug_assert!(
                must_appear_at_all,
                "if something must appear in a stage, it must appear at all"
            );
        }
        Self(RefCell::new(Inner {
            must_appear_at_all,
            must_appear_in,
            can_appear_in,
            cannot_appear_twice,
        }))
    }

    pub fn can_appear_in(&self) -> StageMask { self.0.borrow().can_appear_in }

    pub fn must_appear_in(&self) -> StageMask { self.0.borrow().must_appear_in }

    pub fn must_appear_at_all(&self) -> bool { self.0.borrow().must_appear_at_all }

    /// whether this element must appear in one of the stages that are set in `filter`
    pub fn is_present(&self, filter: StageMask) -> bool { self.0.borrow().is_present(filter) }

    pub fn new_all_in_pipeline(pipeline_kind: PipelineKind) -> Self {
        Self::new(false, StageMask::empty(), pipeline_kind.into(), false)
    }

    pub fn all() -> Self {
        Self(RefCell::new(Inner {
            must_appear_at_all: false,
            must_appear_in: StageMask::empty(),
            can_appear_in: StageMask::all(),
            cannot_appear_twice: false,
        }))
    }

    pub fn empty() -> Self {
        Self(RefCell::new(Inner {
            must_appear_at_all: false,
            must_appear_in: StageMask::empty(),
            can_appear_in: StageMask::empty(),
            cannot_appear_twice: false,
        }))
    }

    pub fn restrict(mut self, additional_restrictions: PossibleStages) -> Self {
        let other = additional_restrictions.0.into_inner();
        let self_ = self.0.get_mut();

        self_.can_appear_in &= other.can_appear_in;
        self_.must_appear_in |= other.must_appear_in;

        self_.must_appear_at_all |= other.must_appear_at_all;
        self_.cannot_appear_twice |= other.cannot_appear_twice;

        self
    }

    /// if must appear, but doesn't have a stage in `must_appear_in` yet,
    /// chooses the earliest `can_appear_in` stage
    pub fn put_into_early_stage_if_unassigned(a: &mut Inner) {
        // if it must appear, doesn't have a stage yet, but can appear
        let doesnt_have_a_stage_yet = a.must_appear_in.count_ones() == 0;
        if a.must_appear_at_all {
            if doesnt_have_a_stage_yet {
                a.must_appear_in = a.can_appear_in.earliest_stage_only();
            }
            assert!(a.must_appear_in.count_ones() > 0);
        }
    }

    /// special case for the vertex-writeable storage case
    pub fn restrict_to_nonvertex_stage_if_unassigned(a: &mut Inner) {
        // if it must appear, doesn't have a stage yet, but can appear
        let doesnt_have_a_stage_yet = a.must_appear_in.count_ones() == 0;
        if a.must_appear_at_all && doesnt_have_a_stage_yet {
            // remove vertex stage from possible stages
            a.can_appear_in &= !StageMask::vert();
        }
    }

    fn track_changes(
        a_object: &dyn InferStage,
        b_object: &dyn InferStage,
        propagation: Propagation,
    ) -> Result<u64, StageSolverError> {
        let a = a_object.possible_stages();
        let b = b_object.possible_stages();

        if !std::ptr::eq(a, b) {
            let (a_before, b_before) = {
                let mut a = a.0.borrow_mut();
                let mut b = b.0.borrow_mut();

                (a.clone(), b.clone())
            };

            pub fn sym(a_obj: &dyn InferStage, b_obj: &dyn InferStage) -> Result<(), StageSolverErrorKind> {
                let mut a = a_obj.possible_stages().0.borrow_mut();
                let mut b = b_obj.possible_stages().0.borrow_mut();

                if NICER_ERRORS {
                    if (a.can_appear_in & b.can_appear_in).is_empty() {
                        drop(a);
                        drop(b);
                        let mut s = String::new();
                        let f = &mut s;
                        a_obj.write_name(f);
                        write!(f, " ");
                        a_obj.write_can_only_appear_in(f);
                        write!(f, ", but ");
                        b_obj.write_name(f);
                        write!(f, " (at {} )", b_obj.call_info());
                        write!(f, ", which ");
                        b_obj.write_can_only_appear_in(f);
                        write!(f, ".");
                        return Err(StageSolverErrorKind::Propagation(s));
                    }

                    let a_cannot_appear_in = !a.can_appear_in;
                    let b_cannot_appear_in = !b.can_appear_in;
                    if (a.must_appear_in & b_cannot_appear_in) != StageMask::empty() {
                        drop(a);
                        drop(b);
                        let mut s = String::new();
                        let f = &mut s;
                        a_obj.write_name(f);
                        write!(f, " ");
                        a_obj.write_must_appear_in(f);
                        write!(f, ", but ");
                        b_obj.write_name(f);
                        write!(f, " (at {} )", b_obj.call_info());
                        write!(f, ", which ");
                        b_obj.write_can_only_appear_in(f);
                        write!(f, ".");
                        return Err(StageSolverErrorKind::Propagation(s));
                    }
                    // same as above but a vs b swapped
                    if (a_cannot_appear_in & b.must_appear_in) != StageMask::empty() {
                        drop(a);
                        drop(b);
                        let mut s = String::new();
                        let f = &mut s;
                        a_obj.write_name(f);
                        write!(f, " ");
                        a_obj.write_can_only_appear_in(f);
                        write!(f, ", but ");
                        b_obj.write_name(f);
                        write!(f, " (at {} )", b_obj.call_info());
                        write!(f, ", which ");
                        b_obj.write_must_appear_in(f);
                        write!(f, ".");
                        return Err(StageSolverErrorKind::Propagation(s));
                    }
                }

                {
                    let can = a.can_appear_in & b.can_appear_in;
                    a.can_appear_in = can;
                    b.can_appear_in = can;
                }
                {
                    let must = a.must_appear_in | b.must_appear_in;
                    a.must_appear_in = must;
                    b.must_appear_in = must;
                }
                {
                    let must_at_all = a.must_appear_at_all || b.must_appear_at_all;
                    a.must_appear_at_all = must_at_all;
                    b.must_appear_at_all = must_at_all;
                }
                {
                    let cannot_multistage = a.cannot_appear_twice && b.cannot_appear_twice;
                    a.cannot_appear_twice = cannot_multistage;
                    b.cannot_appear_twice = cannot_multistage;
                }
                Ok(())
            }

            enum AsymRError {}

            fn asym_r(dependent_obj: &dyn InferStage, arg_obj: &dyn InferStage) -> Result<(), StageSolverErrorKind> {
                let mut dependent = dependent_obj.possible_stages().0.borrow_mut();
                let mut arg = arg_obj.possible_stages().0.borrow_mut();

                if NICER_ERRORS {
                    if (dependent.can_appear_in & arg.can_appear_in).is_empty() {
                        drop(dependent);
                        drop(arg);
                        let mut s = String::new();
                        let f = &mut s;
                        dependent_obj.write_name(f);
                        write!(f, " ");
                        dependent_obj.write_can_only_appear_in(f);
                        write!(f, ", but depends on ");
                        arg_obj.write_name(f);
                        write!(f, " (at {} )", arg_obj.call_info());
                        write!(f, ", which ");
                        arg_obj.write_can_only_appear_in(f);
                        write!(f, ".");
                        return Err(StageSolverErrorKind::Propagation(s));
                    }

                    let arg_cannot_appear_in = !arg.can_appear_in;
                    if (dependent.must_appear_in & arg_cannot_appear_in) != StageMask::empty() {
                        drop(dependent);
                        drop(arg);
                        let mut s = String::new();
                        let f = &mut s;
                        dependent_obj.write_name(f);
                        write!(f, " ");
                        dependent_obj.write_must_appear_in(f);
                        write!(f, ", but depends on ");
                        arg_obj.write_name(f);
                        write!(f, " (at {} )", arg_obj.call_info());
                        write!(f, ", which ");
                        arg_obj.write_can_only_appear_in(f);
                        write!(f, ".");
                        return Err(StageSolverErrorKind::Propagation(s));
                    }
                }

                // if an expr's argument cannot appear in a stage, the expr itself can also not appear in that stage
                dependent.can_appear_in &= arg.can_appear_in;

                // if an expr must appear in a stage, its arguments must also appear in that stage
                arg.must_appear_in |= dependent.must_appear_in;

                // if an expr must appear, its arguments must appear too
                arg.must_appear_at_all |= dependent.must_appear_at_all;

                Ok(())
            }

            let result = match propagation {
                Propagation::AsymL => asym_r(b_object, a_object).map_err(|e| (e, b_object.call_info())),
                Propagation::AsymR => asym_r(a_object, b_object).map_err(|e| (e, a_object.call_info())),
                Propagation::Sym => sym(a_object, b_object).map_err(|e| (e, a_object.call_info())),
            };

            if let Err((kind, call_info)) = result {
                return Err(StageSolverError { kind, call_info });
            }

            let mut a = a.0.borrow_mut();
            let mut b = b.0.borrow_mut();

            a.confirm_stage_if_no_alternatives();
            b.confirm_stage_if_no_alternatives();

            if let Err(kind) = a.check_ok() {
                let call_info = a_object.call_info();
                return Err(StageSolverError {
                    call_info,
                    kind: kind.with_name(a_object.name_string()),
                });
            }
            if let Err(kind) = b.check_ok() {
                let call_info = b_object.call_info();
                return Err(StageSolverError {
                    call_info,
                    kind: kind.with_name(b_object.name_string()),
                });
            }

            Ok((*a != a_before) as u64 + (*b != b_before) as u64)
        } else {
            // object depends on itself. No stage changes can be propagated
            Ok(0)
        }
    }

    //TODO(release) low prio: reduce code duplication between this and `Self::track_changes`,
    // for example by moving the cannot_appear and is_contradiction checks into `confirm_stage_if_no_alternatives`
    fn track_changes_unary(a_obj: &dyn InferStage, mut f: impl FnMut(&mut Inner)) -> Result<u64, StageSolverError> {
        let mut a = a_obj.possible_stages().0.borrow_mut();
        let a_before = a.clone();

        f(&mut a);
        a.confirm_stage_if_no_alternatives();

        a.check_ok().map_err(|e| StageSolverError {
            call_info: a_obj.call_info(),
            kind: e.with_name(a_obj.name_string()),
        });

        Ok((*a != a_before) as u64)
    }

    /// returns the amount of [`PossibleStages`] structs that were modified
    fn propagate_stage_requirements(ctx: &Context, all_allocs: &[Rc<MemoryRegion>]) -> Result<u64, StageSolverError> {
        let nodes = ctx.pool::<Node>();
        let blocks = ctx.pool::<Block>();

        let mut total_changed = 0;

        let add_call_info = |call_info| move |kind| StageSolverError { call_info, kind };

        // nodes <=> their args
        for (node_i, node) in nodes.enumerate() {
            for arg_i in node.args.iter().cloned() {
                let arg = &nodes[arg_i];
                total_changed += Self::track_changes(node, arg, Propagation::AsymR)?;
            }
        }

        // nodes <=> the blocks they were recorded in
        // user defined function call nodes <=> funtion definition
        for node in nodes.iter() {
            let mut block = &blocks[node.block];

            // the entry point block does not propagate like the others
            if block.kind != BlockKind::EntryPoint {
                total_changed += Self::track_changes(block, node, Propagation::AsymL)?;
            }

            if let Expr::FnRelated(FnRelated::Call(key)) = &node.expr {
                let fn_def = &ctx.pool()[*key];
                let fn_block = &blocks[fn_def.body];
                total_changed += Self::track_changes(fn_block, node, Propagation::AsymL)?;
            }
        }

        // blocks => their ancestors
        for ch_key in blocks.keys() {
            let mut child = &blocks[ch_key];
            for parent in Block::iter_stack(ch_key, &blocks) {
                total_changed += Self::track_changes(child, parent, Propagation::AsymR)?;
                child = parent;
            }
        }

        // blocks => their stmts
        // alloc/return statements => their optinal arg nodes
        // control structure stmts => their blocks
        for blk_key in blocks.keys() {
            let mut block = &blocks[blk_key];
            let is_entry_point_block = matches!(block.kind, BlockKind::EntryPoint);

            // condition block and condition expr are treated as one thing (=propagate symmetrical)
            if let Some(cond) = block.get_expr_if_condition() {
                let cond = &nodes[cond];
                total_changed += Self::track_changes(block, cond, Propagation::Sym)?;
            }

            // statements are the most involved aspect of the stage solver.
            // after unsuccessfully trying to generalize their behavior, we now
            // instead write the specifics right into this for loop.
            for (kind, _, call_info) in &block.stmts {
                if let Stmt::Expr(ExprStmt::Expr(n) | ExprStmt::IntroduceIdent(n) | ExprStmt::Condition(n)) = kind {
                    // expr statements do not have their own stage tracking, as their nodes already cover that.
                    continue;
                }

                let stmt = StmtInfo {
                    kind,
                    call_info: *call_info,
                    nodes: &nodes,
                };

                if !is_entry_point_block {
                    total_changed += Self::track_changes(
                        &stmt,
                        block,
                        Propagation::AsymR, // TODO(release) test whether symmetrical propagation belongs here with some simple test case
                    )?;
                }

                // Return and Allocate have optional node "args" which require asymmetrical propagation
                if let Some(optional_arg) = stmt.kind.optional_arg_nodes() {
                    let mut optional_arg_node = &nodes[optional_arg];
                    total_changed += Self::track_changes(&stmt, optional_arg_node, Propagation::AsymR)?;
                }

                // control flow statements may consist of blocks (then/else etc blocks)
                if let Stmt::Flow(flow_stmt, stages) = stmt.kind {
                    for stmt_block_key in flow_stmt.blocks_used().iter() {
                        let mut stmt_block = &blocks[*stmt_block_key];
                        total_changed += Self::track_changes(&stmt, stmt_block, Propagation::Sym)?;
                    }
                }
            }
        }

        // memory regions and their interaction expressions (read/write/readwrite)
        for region in all_allocs {
            use AddressSpace as AS;

            for interact in region.interactions.borrow().iter() {
                let node = &nodes[interact.node];

                let propagation_kind = match (region.address_space, region.allowed_access) {
                    // readonly bindings can appear in multiple shader stages.
                    // any other allocation or binding must appear in only one shader stage, because
                    // the `shame` control flow allows that a fragment-stage-write appears _before_ a
                    // vertex-stage-write in the code.
                    // this order would be flipped around when the stages are emitted as statements
                    // in their respective shaders.
                    // With readonly bindings this is inconsequential, as the order of reads can be swapped
                    // if the memory region stays unmodified in that time period (which is the case for readonly bindings)
                    (AS::Storage | AS::Uniform | AS::Handle, AccessMode::Read) => Propagation::AsymR,
                    _ => Propagation::Sym,
                };

                total_changed += Self::track_changes(node, &**region, propagation_kind)?;
            }
        }
        Ok(total_changed)
    }


    fn remove_vertex_stage_from_unchosen_writable_bindings(ctx: &Context) -> Result<u64, StageSolverError> {
        let mut total_changed = 0;

        // TODO(release) make this not iterate over every node
        // like `for (path, wip_binding) in pipeline_layout.bindings.into_iter() {`
        for node in ctx.pool::<Node>().iter() {
            if node.is_writeable_binding() {
                // remove `vert` from possible stages if it got this far and we don't have vertex writeable storage by default
                total_changed += Self::track_changes_unary(node, Self::restrict_to_nonvertex_stage_if_unassigned)?;
                // still perform the put_into_early_stage_if_unassigned below, now that vertex is gone
            }
        }
        Ok(total_changed)
    }

    /// if something `must_appear_at_all`, but doesn't have a stage in `must_appear_in` yet,
    /// chooses the earliest `can_appear_in` stage and puts it into `must_appear_in`
    fn force_stage_for_the_unchosen(ctx: &Context, all_allocs: &[Rc<MemoryRegion>]) -> Result<u64, StageSolverError> {
        let mut total_changed = 0;
        let add_call_info = |call_info| move |kind| StageSolverError { call_info, kind };

        let nodes = ctx.pool::<Node>();
        for node in nodes.iter() {
            total_changed += Self::track_changes_unary(node, Self::put_into_early_stage_if_unassigned)?;
        }
        for block in ctx.pool::<Block>().iter() {
            total_changed += Self::track_changes_unary(block, Self::put_into_early_stage_if_unassigned)?;
            for (stmt, _, stmt_call_info) in block.stmts.iter() {
                let stmt_info = match stmt {
                    // some statements don't have their own stage tracking, as they are effectively just one expr. Exprs are handled elsewhere
                    Stmt::Expr(_) | Stmt::Allocate { .. } => continue,
                    Stmt::Flow(_, stages) => StmtInfo {
                        kind: stmt,
                        call_info: *stmt_call_info,
                        nodes: &nodes,
                    },
                };
                total_changed += Self::track_changes_unary(&stmt_info, Self::put_into_early_stage_if_unassigned)?;
            }
        }
        for alloc in all_allocs.iter() {
            total_changed += Self::track_changes_unary(&**alloc, Self::put_into_early_stage_if_unassigned)?;
        }
        Ok(total_changed)
    }
}

fn create_debug_report_string(ctx: &Context) -> String {
    let nodes = ctx.pool::<Node>();
    let blocks = ctx.pool::<Block>();
    let mut s = String::new();
    for (node_i, node) in nodes.enumerate() {
        writeln!(s, "n{: <3}  {} {}", node_i.index(), node.stages, node.expr);
    }
    writeln!(s);
    for (block_i, block) in blocks.enumerate() {
        writeln!(s, "b{: <3}  {} {:?}", block_i.index(), block.stages, block.kind);
    }
    s
}

fn print_debug_report_diff(debug_report_before: &str, debug_report_after: &str) {
    let num_lines_before = debug_report_before.lines().count();
    let num_lines_after = debug_report_after.lines().count();

    if num_lines_before != num_lines_after {
        println!("<report line count has changed from {num_lines_before} to {num_lines_after}. printing full report>");
        println!("{debug_report_after}");
    } else {
        for (before, after) in debug_report_before.lines().zip(debug_report_after.lines()) {
            let line = if before != after {
                let mut s = String::new();
                let line_color = line_color(after);
                if before.chars().count() == after.chars().count() {
                    // show only the changed chars
                    //let non_highlighted = Some("#8888FF");
                    let non_highlighted = line_color;
                    set_color(&mut s, non_highlighted, false);
                    for (b, a) in before.chars().zip(after.chars()) {
                        if b == a {
                            s.write_char(a);
                        } else {
                            set_color(&mut s, Some("#FFCC00"), false);
                            s.write_char(a);
                            set_color(&mut s, non_highlighted, false);
                        }
                    }
                    set_color(&mut s, None, false);
                    writeln!(&mut s);
                } else {
                    // show both lines
                    set_color(&mut s, Some("#DF5853"), false);
                    writeln!(&mut s, "{before}");
                    set_color(&mut s, Some("#8888FF"), false);
                    writeln!(&mut s, "{after}");
                    set_color(&mut s, None, false);
                }
                s
            } else {
                "".into() //after.into()
            };
            print!("{line}")
        }
    }
}

fn line_color(line: &str) -> Option<&'static str> {
    let vert_stage_color = Some("#FF0080"); //Some("#FF65C0"); //Some("#FF5050");
    let frag_stage_color = Some("#50B0FF"); //Some("#8888FF");
    let dead_code_color = Some("#505050");
    let all_stages_color = Some("#909060");
    let undecided_color = None; // Some("#B370A9");

    if line.contains("!v |") {
        vert_stage_color
    } else if line.contains("! f|") {
        frag_stage_color
    } else if line.contains("! -|v ") {
        vert_stage_color
    } else if line.contains("! -| f") {
        frag_stage_color
    } else if line.contains("!vf|") {
        all_stages_color
    } else if line.contains("! -|") {
        undecided_color
    } else if line.contains("  -|") {
        dead_code_color
    } else {
        undecided_color
    }
}

fn print_debug_report(report: &str) {
    let mut s = String::new();
    for line in report.lines() {
        let s = &mut s;
        set_color(s, line_color(line), false);
        writeln!(s, "{line}");
        set_color(s, None, false);
    }
    print!("{s}")
}

pub fn solve_shader_stages(ctx: &Context) -> Result<(), StageSolverError> {
    // TODO(release) lots of optimization potential here, for now i'm focusing on correctness

    // TODO(release) this "all_allocs" is almost a duplicate of the non_fn_allocaitons vec in the wgsl backend.
    //       Maybe put this `all_allocs` into `ctx` and reuse it in the wgsl backend.
    let all_allocs = {
        let mut vec = Vec::new();
        let mut dedup = BTreeSet::<_>::new();
        for node in ctx.pool::<Node>().iter() {
            match &node.ty {
                Type::Ptr(a, ..) | Type::Ref(a, ..) => {
                    if dedup.insert(Rc::as_ptr(a)) {
                        vec.push(Rc::clone(a))
                    }
                }
                Type::Unit | Type::Store(_) => (),
            }
        }
        vec
    };

    let mut debug_report_before = if cfg!(feature = "debug_print") && cfg!(debug_assertions) {
        println!(
            "stage propagation for {} nodes, {} blocks, {} allocs",
            ctx.pool::<Node>().len(),
            ctx.pool::<Block>().len(),
            all_allocs.len()
        );
        let r = create_debug_report_string(ctx);
        print_debug_report(&r);
        r
    } else {
        Default::default()
    };

    // iteratively propagate the stage requirements
    let mut rounds = 0;
    loop {
        let mut forced_choice = false;
        let mut applied_vertex_special_case_times = None;
        rounds += 1;
        let mut changes = PossibleStages::propagate_stage_requirements(ctx, &all_allocs);
        if changes == Ok(0) {
            if !ctx.settings().vertex_writable_storage_by_default {
                changes = PossibleStages::remove_vertex_stage_from_unchosen_writable_bindings(ctx);
            }
            match changes {
                Ok(0) | Err(_) => (),
                Ok(n) => applied_vertex_special_case_times = Some(n),
            }
        }
        if changes == Ok(0) {
            forced_choice = true;
            changes = PossibleStages::force_stage_for_the_unchosen(ctx, &all_allocs)
        }
        if cfg!(feature = "debug_print") && cfg!(debug_assertions) {
            print!("\nround {rounds} ");
            if let Some(n) = applied_vertex_special_case_times {
                println!(" (applied vertex shader special case {rounds} times)");
            }
            if forced_choice {
                println!(" (regular propagation settled, placing the unplaced)");
            }
            match &changes {
                Ok(0) => println!("(no changes)"),
                Ok(changes) => println!("({changes} changes):"),
                Err(e) => println!("(unsolvable)"),
            };

            let mut debug_report_after = create_debug_report_string(ctx);
            print_debug_report_diff(&debug_report_before, &debug_report_after);
            debug_report_before = debug_report_after;
        }
        if changes? == 0 {
            break;
        }
    }
    // the next iteration should not cause any changes as well
    debug_assert_eq!(
        {
            if cfg!(feature = "debug_print") && cfg!(debug_assertions) {
                println!("verifying settled stage solver (extra round)");
            }
            PossibleStages::propagate_stage_requirements(ctx, &all_allocs)
        },
        Ok(0)
    );

    if cfg!(feature = "debug_print") && cfg!(debug_assertions) {
        println!("results:");
        print_debug_report(&debug_report_before);

        println!(
            "stage propagation settled after {rounds} rounds for {} nodes, {} blocks, {} allocs\n",
            ctx.pool::<Node>().len(),
            ctx.pool::<Block>().len(),
            all_allocs.len()
        );
    }

    Ok(())
}
