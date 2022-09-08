
use crate::{pool::Key, BranchState, Stage};
use super::*;

#[derive(Copy, Clone, Debug)]
pub enum BlockKind {
    /// Block containing multiple statements. Used for:
    ///     - function body
    ///     - if body
    ///     - else body
    Body,
    /// same as Body, but in for/while loops.
    /// this is a separate variant to allow the loop init rules to be simpler
    /// also this allows simple implementation of `break` and `continue` validity checks
    LoopBody,
    /// Block representing `cond` in loops such as
    /// `for(a; cond; b)` or `while(cond)`
    /// - only expression statements allowed, which means
    ///     - no variable declarations/definitions in this block
    ///     - may not contain any blocks
    /// 
    /// contains either
    ///     - `Some(expr)` => the boolean condition expression
    ///     - `None` => infinite loop
    LoopCondition(Option<Key<Expr>>),
    /// Block representing `inc` in `for(a; b; inc)` 
    /// - only expression statements allowed, which means
    ///     - no variable declarations/definitions in this block
    ///     may not contain blocks of type: Body, LoopCondition, LoopIncrement, LoopInit
    LoopIncrement,
    /// Block representing `inc` in `for(a; b; inc)`
    /// 
    /// allowed statements are:
    /// - declarations/definitions, but only if they all declare variables of 
    ///   the same type
    /// - expression statements
    /// 
    /// may not contain blocks of type Body.
    LoopInit,
}
use BlockKind::*;

impl std::fmt::Display for BlockKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Body => "body",
            LoopBody => "loop body",
            LoopCondition(_) => "loop condition",
            LoopIncrement => "loop increment",
            LoopInit => "loop initialization",
        })
    }
}

impl BlockKind {
    pub fn may_contain_block_of_kind(&self, kind: BlockKind) -> bool {
        match (self, kind) {
            (LoopIncrement | LoopCondition(_), _) => false,

            (Body | LoopBody, Body | LoopInit) => true,
            (Body | LoopBody, LoopBody | LoopCondition(_) | LoopIncrement) => false,
            (LoopInit, Body | LoopInit) => false,
            (LoopInit, LoopBody | LoopCondition(_) | LoopIncrement) => true,
        }
    }

    pub fn may_contain_variable_decls_or_defs(&self) -> bool {
        match self {
            Body | LoopBody | LoopInit => true,
            LoopCondition(_) | LoopIncrement => false
        }
    }
}

impl Default for BlockKind {
    fn default() -> Self {BlockKind::Body}
}

/// see [`BlockKind`] for what a block can be. 
pub struct Block {
    pub(crate) kind: BlockKind,
    pub(crate) branch_info: Option<(BranchState, Stage)>,
    /// Amount of times expressions were attempted to be recorded that contained
    /// exclusively `Any` objects that had a non available expression
    /// e.g. "per-vertex" expressions during a fragment shader recording in this 
    /// block (not counting ones that happened in nested blocks).
    /// 
    /// this is a number instead of a bool to provide more info in an error message
    pub(crate) amount_of_attempts_recording_not_available_exprs: u32,
    /// amount of expressions that were successfully recorded in this block
    /// (not counting expressions recorded in nested blocks)
    /// 
    /// this is a number instead of a bool to provide more info in an error message
    pub(crate) amount_of_exprs_recorded: u32,
    pub(crate) parent: Option<Key<Block>>,
    /// item it belongs to, in order to decide whether a function boundary was 
    /// crossed during recording
    pub(crate) origin_item: Key<Item>,
    pub(crate) stmts: Vec<Stmt>,
}

impl Block {

    pub(crate) fn new(
        parent: Option<Key<Block>>, 
        origin_item: Key<Item>, 
        branch_info: Option<(BranchState, Stage)>,
        kind: BlockKind) -> Block {
        Self {
            kind,
            branch_info,
            amount_of_attempts_recording_not_available_exprs: 0,
            amount_of_exprs_recorded: 0,
            parent,
            origin_item,
            stmts: Vec::new(),
        }
    }

    pub fn add_stmt(&mut self, stmt: Stmt) {
        self.stmts.push(stmt)
    }

    // pub(crate) fn check_for_not_available_exprs(&self, ctx: &Context) {
    //     let na_amount = self.amount_of_attempts_recording_not_available_exprs;
    //     let expr_amount = self.amount_of_exprs_recorded;
    //     match self.kind {
    //         Body | LoopBody => (), //No restrictions to check for
    //         LoopCondition(_) | LoopIncrement | LoopInit => {

    //             match (na_amount, expr_amount) {
    //                 (1.., 0) => (),
                    
    //             }
    //             // ctx.push_error(Error::BlockRestrictionsViolated(
    //             //     format!("{}")
    //             // ));
    //         },
    //     };
    // }
}