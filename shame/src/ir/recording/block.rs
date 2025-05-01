use std::rc::Rc;

use crate::ir::ir_type::AddressSpace;
use crate::ir::pipeline::PipelineKind;
use crate::ir::pipeline::{PossibleStages, StageMask};
use crate::{
    common::pool::{Key, Pool, PoolRef},
    ir::expr::Expr,
    ir::recording::{context::Context, stmt::Stmt},
};

/// different kinds of block that have different restrictions on what can happen
/// within them.
/// Example structure of a shame program:
/// ```text
/// EntryPoint {
///     Condition {
///     }
///     Body(If) {
///         ...
///     }
///     // function recording inside the shame entrypoint
///     Body(Fn) {
///         ...
///         // another function recording
///         Body(Fn) {
///             ...
///         }
///     }
///     // loop recordings are wrapped in loop init blocks so that the scope
///     // of varaibles introduced in the init block is reflected in the block
///     // hierarchy
///     ForInit {
///         int i = 0;
///         Condition {
///             i + j < 10
///         }
///         ForIncrement {
///             ++i
///         }
///         Body(For) {
///             ...
///             Condition {
///                 ...
///             }
///             Body(If) {
///                 break
///             }
///             ...
///             Condition {
///                 ...
///             }
///             ConditionBody {
///                 continue
///             }
///         }
///     }
/// }
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// the encoding's root block. Its statements will be turned into statements
    /// in the shader's entry-point function.
    /// This must be the current block for the user to be able to do:
    /// - bindings specialization
    /// - rasterization
    EntryPoint,
    /// Block containing multiple statements. Which statements are allowed depends
    /// on the kind.
    Body(BodyKind),
    /// Block representing `cond` in loops/if such as
    /// `for(a; cond; b)`, `if(cond)` or `while(cond)`
    /// - only condition statements allowed, which means
    ///     - no variable declarations/definitions in this block
    ///     - may not contain any blocks
    Condition(ConditionKind),
    /// Block representing `inc` in `for(a; b; inc)`
    /// - only expression statements allowed, which means
    ///     - no variable declarations/definitions in this block
    ///       may not contain blocks of type: Body, LoopCondition, ForIncrement, ForInit
    ForIncrement,
    /// Block representing `inc` in `for(a; b; inc)`
    ///
    /// allowed statements are:
    /// - declarations/definitions, but only if they all declare variables of
    ///   the same type
    /// - expression statements
    ///
    /// may not contain blocks of type Body.
    ForInit,
}
use BlockKind::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConditionKind {
    If,
    While,
    /// not allowed in `Body` blocks, only in `ForInit`
    For,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Then,
    Else,
    /// allows return statement inside of it or its nested blocks
    Function,
    /// allows continue and break statements inside of it or its nested blocks
    While,
    /// allows continue and break statements inside of it or its nested blocks
    For,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ThenOrElse {
    Then,
    Else,
}

use super::{CallInfo, ExprStmt, Node, RecordTime, TimeInstant};
use thiserror::Error;

impl BlockKind {
    pub fn has_different_execution_state(&self) -> bool {
        match self {
            EntryPoint => false,
            ForInit => false,
            // executed conditionally
            Body(_) => true,
            // the condition itself is executed conditionally after the first time
            Condition(_) => true,
            // executed conditionally
            ForIncrement => true,
        }
    }

    #[rustfmt::skip]
    pub fn may_contain_block_of_kind(&self, kind: BlockKind) -> bool {
        use ConditionKind as CK;
        use BodyKind as BK;
        match (self, kind) {
            // no block may contain an entry point block inside of it
            (_, EntryPoint) => false,
            // `ForIncrement` / `Condition` blocks may not contain nested blocks
            (ForIncrement | Condition(_), _) => false,
            // `ForInit` is the only block type that can contain other For-related block types, and may not contain anything else
            (ForInit, Body(BK::For) | Condition(CK::For) | ForIncrement) => true,
            (_       , Body(BK::For) | Condition(CK::For) | ForIncrement) => false,
            (ForInit, _) => false,
            // any `Body` block can contain any non-for-`Body` or `ForInit` or `Condition(If | While)` blocks
            (
                EntryPoint | Body(BK::Function | BK::Then | BK::Else | BK::While | BK::For),
                EntryPoint | Body(BK::Function | BK::Then | BK::Else | BK::While) |
                ForInit | Condition(CK::If | CK::While),
            ) => true,
        }
    }

    pub fn may_contain_variable_decls_or_defs(&self) -> bool {
        use BodyKind as BK;
        use ConditionKind as CK;
        match self {
            EntryPoint | Body(BK::Function | BK::Then | BK::Else | BK::While | BK::For) | ForInit => true,
            Condition(CK::If | CK::While | CK::For) | ForIncrement => false,
        }
    }

    pub fn may_allocate_in_address_space(&self, address_space: AddressSpace) -> bool {
        use crate::ir::ir_type::AddressSpace as A;
        match address_space {
            A::Thread | A::WorkGroup | A::PushConstant | A::Uniform | A::Storage | A::Handle | A::Output => {
                match self {
                    EntryPoint => true,
                    _ => false,
                }
            }
            A::Function => self.may_contain_variable_decls_or_defs(),
        }
    }
}

/// see [`BlockKind`] for what a block can be.
#[derive(Debug, PartialEq, Eq)]
pub struct Block {
    pub(crate) call_info: CallInfo,
    pub(crate) kind: BlockKind,
    pub(crate) stages: PossibleStages,
    pub(crate) parent: Option<Key<Block>>,
    pub(crate) stmts: Vec<(Stmt, TimeInstant, CallInfo)>,
}

#[derive(Debug, Error, Clone)]
pub enum BlockError {
    #[error("block recording chain resulted in unexpected amount of blocks. expected {expected}, got {actual}.")]
    UnexpectedAmountOfBlocks { expected: usize, actual: usize },
    #[error(
        "block recording was not finished properly. this happens in block-series recordings such as `if`, `for`, `while`, when the block recorder is prematurely dropped, or when two block-series recordings are being advanced in an alternating way."
    )]
    UnfinishedBlockRecording,
    #[error("ill formed block series recorder")]
    IllFormedBlockSeriesRecorder, // this error always happens because of a previous error, and thus is hidden if previous errors exist
    #[error("encoding ended while the block started at {0} was still unclosed.")]
    UnclosedBlock(CallInfo),
    #[error(
        "block recording exceeded its pipeline encoding's lifetime. It was either started before or ended after the pipeline encoding it belongs to."
    )]
    BlockOutsideOfEncoding,
    #[error(
        "the parent block has an invalid index. Did you start recording a block in a previous Encoding, then ended it in this Encoding?"
    )]
    ParentBlockIndexInvalid,
    #[error("entry point block must be created through a different block constructor")]
    WrongCtorForEntryPoint,
    #[error("attempting to create a root block that is not of kind `EntryPoint`")]
    ParentlessBlockMustHaveEntryPointKind,
    #[error("cannot create a `{nest:?}` block inside the `{:?}` block created at \'{}\'", .parent.0, .parent.1)]
    CannotNestKindInParent {
        nest: BlockKind,
        parent: (BlockKind, CallInfo),
    },
    #[error("the stmt {stmt:?} is not allowed in this block (block was started at {block_caller} )")]
    StmtNotAllowedInBlockKind { stmt: Stmt, block_caller: CallInfo },
    #[error("{0} is not allowed inside a conditional or function block recording (block was started at {1} ).")]
    MustBeInEncodingScope(&'static str, CallInfo),
}

impl Block {
    pub(crate) fn new_entry_point(call_info: CallInfo, pipeline_kind: PipelineKind) -> Block {
        Self {
            call_info,
            kind: BlockKind::EntryPoint,
            stages: PossibleStages::new_all_in_pipeline(pipeline_kind),
            parent: None,
            stmts: Default::default(),
        }
    }

    pub(crate) fn find_in_stack(
        top_of_stack: Key<Block>,
        pool: &PoolRef<Block>,
        mut pred: impl FnMut(&Block) -> bool,
    ) -> Option<Key<Block>> {
        let mut current = top_of_stack;
        loop {
            let block = &pool[current];
            match (pred(block), block.parent) {
                (true, _) => break Some(current),
                (_, None) => break None,
                (_, Some(parent)) => current = parent,
            }
        }
    }

    pub(crate) fn find_key_in_stack(
        top_of_stack: Key<Block>,
        pool: &PoolRef<Block>,
        mut pred: impl FnMut(Key<Block>) -> bool,
    ) -> Option<Key<Block>> {
        let mut current = top_of_stack;
        loop {
            let block = &pool[current];
            match (pred(current), block.parent) {
                (true, _) => break Some(current),
                (_, None) => break None,
                (_, Some(parent)) => current = parent,
            }
        }
    }

    pub(crate) fn iter_stack<'a>(top_of_stack: Key<Block>, pool: &'a PoolRef<'a, Block>) -> BlockStackIterator<'a> {
        BlockStackIterator {
            next: Some(top_of_stack),
            pool,
        }
    }

    // if `self` is a `Condition` block, gets the corresponding `Condition` statement's boolean expression,
    // if it was already recorded.
    pub(crate) fn get_expr_if_condition(&self) -> Option<Key<Node>> {
        match self.kind {
            // condition blocks usually contains just a single `Condition` statement, and if not,
            // an encoding error was produced already when the 2nd statement was inserted.
            BlockKind::Condition(_) => self.stmts.iter().find_map(|s| match s {
                (Stmt::Expr(ExprStmt::Condition(node)), _, _) => Some(*node),
                _ => None,
            }),
            _ => None,
        }
    }

    /// creates a new Block that has `parent_key` as parent.
    /// fails if `kind` is `EntryPoint`, use `new_entry_point` for that.
    pub(crate) fn new_with_parent(ctx: &Context, kind: BlockKind, parent_key: Key<Block>) -> Key<Block> {
        let mut pool = ctx.pool_mut();
        let stages = match pool.get(parent_key) {
            None => {
                ctx.push_error(BlockError::ParentBlockIndexInvalid.into());
                PossibleStages::new_all_in_pipeline(ctx.pipeline_kind())
            }
            Some(parent) => {
                if !parent.kind.may_contain_block_of_kind(kind) {
                    ctx.push_error(
                        BlockError::CannotNestKindInParent {
                            nest: kind,
                            parent: (parent.kind, parent.call_info),
                        }
                        .into(),
                    )
                }
                parent.stages.clone()
            }
        };
        pool.push(Self {
            call_info: ctx.latest_user_caller(),
            kind,
            parent: Some(parent_key),
            stages,
            stmts: Default::default(),
        })
    }
}

pub(crate) struct BlockStackIterator<'a> {
    next: Option<Key<Block>>,
    pool: &'a PoolRef<'a, Block>,
}

impl<'a> Iterator for BlockStackIterator<'a> {
    type Item = &'a Block;

    fn next(&mut self) -> Option<Self::Item> { self.next.map(|key| &self.pool[key]).inspect(|b| self.next = b.parent) }
}
