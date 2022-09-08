
use crate::{pool::Key, BranchState};
use super::*;

#[derive(Copy, Clone, Debug)]
pub enum BlockKind {
    /// Block containing multiple statements. Used for:
    ///     - function body
    ///     - if body
    ///     - else body
    ///     - for loop body
    ///     - while loop body
    Body,
    /// Block representing `cond` in loops such as
    /// `for(a; cond; b)` or `while(cond)`
    /// - only expression statements allowed, which means
    ///     - no variable declarations/definitions in this block
    ///     - no nested ifs, elses, loops
    /// 
    /// contains either
    ///     - `Some(expr)` => the condition expression
    ///     - `None` => infinite loop
    LoopCondition(Option<Key<Expr>>),
    /// Block representing `inc` in `for(a; b; inc)` 
    /// - only expression statements allowed, which means
    ///     - no variable declarations/definitions in this block
    ///     - no nested ifs, elses, loops
    LoopIncrement,
    /// Block representing `inc` in `for(a; b; inc)` 
    /// 
    /// allowed statements are:
    /// - declarations/definitions, but only if they all declare variables of 
    ///   the same type
    /// - expression statements
    /// 
    /// no nested ifs, elses, loops
    LoopInit,
}
use BlockKind::*;

impl std::fmt::Display for BlockKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Body => "body",
            LoopCondition(_) => "loop condition",
            LoopIncrement => "loop increment",
            LoopInit => "loop initialization",
        })
    }
}

impl BlockKind {
    pub fn may_contain_nested_blocks(&self) -> bool {
        match self {
            Body => true,
            LoopCondition(_) | LoopIncrement | LoopInit => false
        }
    }

    pub fn may_contain_variable_decls_or_defs(&self) -> bool {
        match self {
            Body | LoopInit => true,
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
    pub(crate) is_branch: Option<BranchState>,
    /// Amount of times expressions were attempted to be recorded that contained
    /// exclusively `Any` objects that had a non available expression
    /// e.g. "per-vertex" expressions during a fragment shader recording in this 
    /// block
    /// 
    /// this is a number instead of a bool to provide more info in an error message
    pub(crate) amount_of_attempts_recording_not_available_exprs: u32,
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
        is_branch: Option<BranchState>,
        kind: BlockKind) -> Block {
        Self {
            kind,
            is_branch,
            amount_of_attempts_recording_not_available_exprs: 0,
            parent,
            origin_item,
            stmts: Vec::new(),
        }
    }

    pub fn record_stmt(&mut self, stmt: Stmt) {
        self.stmts.push(stmt)
    }
}