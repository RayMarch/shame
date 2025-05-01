use std::{fmt::Display, rc::Rc};

use thiserror::Error;

use super::{Block, CallInfo, Context, Ident, MemoryRegion, Node, TimeInstant};
use crate::{
    call_info,
    common::{
        pool::{Key, PoolRef, PoolRefMut},
        small_vec::SmallVec,
    },
    frontend::error::InternalError,
    ir::{
        pipeline::{PossibleStages, StageMask},
        recording::{BlockKind, BodyKind, ConditionKind},
        Len, ScalarType, SizedType, StoreType, Type,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stmt {
    /// expr statements can be created after the stage solver ran, for example
    /// to guarantee that certain expressions have identifiers
    Expr(ExprStmt),
    /// aka "variable declaration" (`var` in wgsl)
    Allocate(AllocStmt),
    /// statements that create control flow blocks
    Flow(FlowStmt, PossibleStages),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocStmt {
    pub allocation: Rc<MemoryRegion>,
    pub initial_value: Option<Key<Node>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlowStmt {
    /// constrol structures that contain blocks
    Control(Control),
    /// statements that perform control flow jumps
    Jump(Jump),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprStmt {
    /// expression statements, mostly for assignment-operators, barriers
    Expr(Key<Node>),
    /// aka "value declaration" (`let` binding in wgsl)
    IntroduceIdent(Key<Node>),
    /// effectively the same as `Expr`, but checked to
    /// - evaluate to bool and
    /// - be the only statement in a `Condition` block
    Condition(Key<Node>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Jump {
    Continue,
    Break,
    Return(Option<Key<Node>>),
    Discard,
}

impl Display for Jump {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Jump::Continue => "continue",
            Jump::Break => "break",
            Jump::Return(Some(_)) => "return <expression>",
            Jump::Return(None) => "return",
            Jump::Discard => "discard",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// control structures which contain block scopes
pub enum Control {
    IfThen {
        cond: Key<Block>,
        then: Key<Block>,
    },
    IfThenElse {
        cond: Key<Block>,
        then: Key<Block>,
        els: Key<Block>,
    },
    For {
        init: Key<Block>,
        cond: Key<Block>,
        inc: Key<Block>,
        body: Key<Block>,
    },
    While {
        cond: Key<Block>,
        body: Key<Block>,
    },
}

#[derive(Debug, Clone, Error)]
pub enum StmtError {
    #[error("expected boolean condition, found value of type `{0:?}`")]
    NonBooleanCondition(Type),
    #[error("Condition statement outside of condition block")]
    ConditionOutsideOfConditionBlock,
    #[error("non-condition statement `{0}` in a condition block")]
    NonConditionInConditionBlock(Stmt),
    #[error(
        "trying to add another statement to a block of kind `{kind:?}`, which may only contain one statement. \
    The block which was introduced at {block_location} already contains a statement which originates from {other_stmt}"
    )]
    MultipleStmtsInBlock {
        kind: BlockKind,
        block_location: CallInfo,
        other_stmt: CallInfo,
    },
    #[error("`{0}` can only be used inside a conditional control flow or function recording")]
    UnconditionalReturnOrDiscard(Jump),
    #[error("{0} statements can only be used inside a loop body recording")]
    OnlyAllowedInLoopBody(Jump),
    #[error("{0} statements can only be used inside a function body recording")]
    OnlyAllowedInFnBody(Jump),
    #[error("cannot create a new variable inside a `{0:?}` block recording")]
    CannotIntroduceIdentInBlock(BlockKind),
    #[error("discard statements are not valid in a block recording of kind `{0:?}`")]
    DiscardInvalidIn(BlockKind),
    #[error("control flow like `if`, `for`, `while`, etc. cannot be added into a block recording of kind `{0:?}`")]
    CannotIntroduceConditionalFlowIn(BlockKind),
    #[error(
        "value used out of scope.\n\
    Value of type `{value_ty}` originating from [ {value_loc} ] is used outside of its scope.\n\
    Scope defined at [ {valid_scope_defined_at} ]\n\
    but value is used at [ {invalid_usage_at} ] in statement of kind: {invalid_usage_stmt}"
    )]
    ValueUsedOutOfScope {
        value_ty: Type,
        value_loc: CallInfo,
        valid_scope_defined_at: CallInfo,
        invalid_usage_at: CallInfo,
        invalid_usage_stmt: Stmt,
    },
}

impl Display for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let write_expr = |f: &mut std::fmt::Formatter<'_>, key: Key<Node>| -> std::fmt::Result {
            let success = Context::try_with(call_info!(), |ctx| -> Option<_> {
                let nodes = ctx.try_pool()?;
                let node = nodes.get(key)?;
                write!(f, "{} {}({})", key.index(), node.expr, &node.ty);
                Some(())
            })
            .flatten();

            if success.is_none() {
                write!(f, "node #{}", key.index())?;
            }
            Ok(())
        };
        match self {
            Stmt::Expr(expr_stmt) => match expr_stmt {
                ExprStmt::Expr(key) => {
                    write!(f, "statement expr ")?;
                    write_expr(f, *key)?;
                    Ok(())
                }
                ExprStmt::IntroduceIdent(key) => {
                    write!(f, "introduce identifier for ")?;
                    write_expr(f, *key)
                }
                ExprStmt::Condition(key) => {
                    write!(f, "condition bool expr ")?;
                    write_expr(f, *key)
                }
            },
            Stmt::Allocate(alloc_stmt) => {
                let a = &alloc_stmt.allocation;
                write!(f, "alloc_in<{}, {}, {}>", a.ty, a.address_space, a.allowed_access,)
            }
            Stmt::Flow(flow_stmt, possible_stages) => match flow_stmt {
                FlowStmt::Control(control) => match control {
                    Control::IfThen { cond, then } => write!(f, "if-then"),
                    Control::IfThenElse { cond, then, els } => write!(f, "if-then-else"),
                    Control::For { init, cond, inc, body } => write!(f, "for-loop"),
                    Control::While { cond, body } => write!(f, "while-loop"),
                },
                FlowStmt::Jump(jump) => match jump {
                    Jump::Continue => write!(f, "continue"),
                    Jump::Break => write!(f, "break"),
                    Jump::Discard => write!(f, "discard"),
                    Jump::Return(return_value) => {
                        write!(f, "return")?;
                        if let Some(key) = return_value {
                            write!(f, " ")?;
                            write_expr(f, *key)?;
                        };
                        Ok(())
                    }
                },
            },
        };
        Ok(())
    }
}

impl ExprStmt {
    pub fn push_to_block(
        self,
        ctx: &Context,
        node_pool: &mut PoolRefMut<Node>, // TODO(release) this doesn't need to be Mut
        block_key: Key<Block>,
        time: TimeInstant,
        call_info: CallInfo,
    ) -> Result<(), StmtError> {
        Stmt::Expr(self).push_to_block(ctx, node_pool, block_key, time, call_info)
    }
}

impl AllocStmt {
    pub fn push_to_block(
        self,
        ctx: &Context,
        node_pool: &mut PoolRefMut<Node>, // TODO(release) this doesn't need to be Mut
        block_key: Key<Block>,
        time: TimeInstant,
        call_info: CallInfo,
    ) -> Result<(), StmtError> {
        Stmt::Allocate(self).push_to_block(ctx, node_pool, block_key, time, call_info)
    }
}

impl FlowStmt {
    pub fn push_to_block(
        self,
        ctx: &Context,
        node_pool: &mut PoolRefMut<Node>,
        block_key: Key<Block>,
        time: TimeInstant,
        call_info: CallInfo,
    ) -> Result<(), StmtError> {
        let possible_stages = match self {
            FlowStmt::Control(control) => PossibleStages::all(),
            FlowStmt::Jump(jump) => match jump {
                Jump::Continue | Jump::Break | Jump::Return(_) => PossibleStages::all(),
                Jump::Discard => PossibleStages::new(true, StageMask::frag(), StageMask::frag(), true),
            },
        };

        Stmt::Flow(self, possible_stages).push_to_block(ctx, node_pool, block_key, time, call_info)
    }

    // returns an `impl IntoIterator<Item = Key<Node>>` over all nodes used in this statement
    pub fn nodes_used(&self) -> Option<Key<Node>> {
        match self {
            FlowStmt::Control(control) => match control {
                Control::IfThen { cond: _, then: _ } |
                Control::IfThenElse {
                    cond: _,
                    then: _,
                    els: _,
                } |
                Control::For {
                    init: _,
                    cond: _,
                    inc: _,
                    body: _,
                } |
                Control::While { cond: _, body: _ } => None,
            },
            FlowStmt::Jump(jump) => match jump {
                Jump::Return(Some(key)) => Some(*key),
                Jump::Return(None) | Jump::Continue | Jump::Break | Jump::Discard => None,
            },
        }
    }

    // returns an `impl IntoIterator<Item = Key<Block>>` over all blocks used in this statement
    pub fn blocks_used(&self) -> SmallVec<Key<Block>, 4> {
        let slice: &[Key<Block>] = match self {
            FlowStmt::Control(control) => match control {
                Control::IfThen { cond, then } => &[*cond, *then],
                Control::IfThenElse { cond, then, els } => &[*cond, *then, *els],
                Control::For { init, cond, inc, body } => &[*init, *cond, *inc, *body],
                Control::While { cond, body } => &[*cond, *body],
            },
            FlowStmt::Jump(jump) => &[],
        };
        slice.into()
    }
}

impl Stmt {
    pub fn stages_cloned(&self, ctx: &Context) -> PossibleStages {
        let nodes = ctx.pool::<Node>();
        match self {
            Stmt::Expr(e) => match e {
                ExprStmt::Condition(n) | ExprStmt::Expr(n) | ExprStmt::IntroduceIdent(n) => &nodes[*n].stages,
            },
            Stmt::Allocate(AllocStmt {
                allocation,
                initial_value: _,
            }) => &allocation.stages,
            Stmt::Flow(flow_stmt, possible_stages) => possible_stages,
        }
        .clone()
    }

    /// Return and Allocate have optional node "args" such as initial value or return value
    pub fn optional_arg_nodes(&self) -> Option<Key<Node>> {
        match self {
            Stmt::Expr(_) => None,
            Stmt::Allocate(AllocStmt {
                allocation,
                initial_value,
            }) => *initial_value,
            Stmt::Flow(flow_stmt, _) => match flow_stmt {
                FlowStmt::Control(control) => match control {
                    Control::IfThen { cond, then } => None,
                    Control::IfThenElse { cond, then, els } => None,
                    Control::For { init, cond, inc, body } => None,
                    Control::While { cond, body } => None,
                },
                FlowStmt::Jump(jump) => match jump {
                    Jump::Continue | Jump::Break | Jump::Discard => None,
                    Jump::Return(key) => *key,
                },
            },
        }
    }

    #[allow(clippy::unit_arg)]
    fn push_to_block(
        self,
        ctx: &Context,
        node_pool: &mut PoolRefMut<Node>,
        block_key: Key<Block>,
        time: TimeInstant,
        call_info: CallInfo,
    ) -> Result<(), StmtError> {
        self.may_be_added_to_block(ctx, block_key, node_pool, call_info)
            .map(|()| {
                let block = &mut ctx.pool_mut()[block_key];
                if let Some(node) = self.nodes_used() {
                    node_pool[node].is_part_of_stmt = true;
                }
                block.stmts.push((self, time, call_info));
            })
    }

    pub fn nodes_used(&self) -> Option<Key<Node>> {
        match self {
            Stmt::Expr(e) => match e {
                ExprStmt::Expr(key) | ExprStmt::IntroduceIdent(key) | ExprStmt::Condition(key) => Some(*key),
            },
            Stmt::Allocate(AllocStmt {
                allocation,
                initial_value,
            }) => *initial_value,
            Stmt::Flow(flow_stmt, _) => flow_stmt.nodes_used(),
        }
    }

    /// whether `self` may be added to the `block_key` block, regardless of where it is added
    /// (it may be added somewhere between the existing statements, wrt. `RecordTime`)
    ///
    /// return value:
    /// - `Ok(())` if `self` may be added to `block_key`
    /// - `Err(reason)` otherwise
    ///
    /// > note:
    /// > also checks how many statements are in the block already for `LoopInit` and `LoopIncrement`
    pub fn may_be_added_to_block(
        &self,
        ctx: &Context,
        block_key: Key<Block>,
        nodes: &PoolRefMut<Node>,
        call_info: CallInfo,
    ) -> Result<(), StmtError> {
        use BlockKind as B;
        use BodyKind as BK;
        use StmtError as E;
        let blocks = ctx.pool();
        let block = &blocks[block_key];

        let arg_used_out_of_scope = self.nodes_used().iter().copied().find(|arg| {
            let arg_node = &nodes[*arg];
            let arg_block_ancestor = Block::find_key_in_stack(block_key, &blocks, |b| b == arg_node.block);
            arg_block_ancestor.is_none()
        });

        if let Some(key) = arg_used_out_of_scope {
            let arg_node = &nodes[key];
            return Err(StmtError::ValueUsedOutOfScope {
                value_ty: arg_node.ty.clone(),
                value_loc: arg_node.call_info,
                valid_scope_defined_at: blocks[arg_node.block].call_info,
                invalid_usage_at: call_info,
                invalid_usage_stmt: self.clone(),
            });
        }

        // this match statement is written this way so that rustc can help
        // validate that every combination of `BlockKind` and `Stmt` is handled.
        // please do not add a `_` or `(_, _)` case to this.
        match (block.kind, self) {
            // =================================================================
            // For increment and init blocks may only contain one stmt
            ((B::ForIncrement | B::ForInit), _) if !block.stmts.is_empty() => {
                match block.stmts.first() {
                    Some(first) => Err(E::MultipleStmtsInBlock {
                        kind: block.kind,
                        block_location: block.call_info,
                        other_stmt: first.2,
                    }),
                    None => Ok(()), // unreachable
                }
            }

            // =================================================================
            // condition related
            (B::Condition(_), Stmt::Expr(ExprStmt::Condition(node))) => match nodes[*node].ty() {
                Type::Store(StoreType::Sized(SizedType::Vector(Len::X1, ScalarType::Bool))) => Ok(()),
                ty => Err(E::NonBooleanCondition(ty.clone())),
            },
            (_, Stmt::Expr(ExprStmt::Condition(node))) => Err(E::ConditionOutsideOfConditionBlock),
            (B::Condition(_), stmt) => Err(E::NonConditionInConditionBlock(stmt.clone())),

            // =================================================================
            // loop-exclusive statements
            (_, Stmt::Flow(FlowStmt::Jump(j @ (Jump::Break | Jump::Continue)), _)) => {
                Block::find_in_stack(block_key, &blocks, |b| matches!(b.kind, B::Body(BK::For | BK::While)))
                    .map(|_| ())
                    .ok_or(E::OnlyAllowedInLoopBody(*j))
            }

            // =================================================================
            // function-exclusive statements
            (_, Stmt::Flow(FlowStmt::Jump(j @ Jump::Return(_)), _)) => {
                Block::find_in_stack(block_key, &blocks, |b| matches!(b.kind, B::Body(BK::Function)))
                    .map(|_| ())
                    .ok_or(E::OnlyAllowedInFnBody(*j))
            }

            // =================================================================
            // variable/value definition/declaration
            (kind, Stmt::Expr(ExprStmt::IntroduceIdent(_)) | Stmt::Allocate { .. }) => kind
                .may_contain_variable_decls_or_defs()
                .then_some(())
                .ok_or(E::CannotIntroduceIdentInBlock(kind)),

            // =================================================================
            // expr stmt
            (
                B::ForIncrement |
                B::ForInit |
                B::EntryPoint |
                B::Body(BK::Then | BK::Else | BK::For | BK::Function | BK::While),
                Stmt::Expr(ExprStmt::Expr(_)),
            ) => Ok(()),

            // =================================================================
            // discard
            (B::ForIncrement | B::ForInit, Stmt::Flow(FlowStmt::Jump(Jump::Discard), _)) => {
                Err(E::DiscardInvalidIn(block.kind))
            }
            (
                B::EntryPoint | B::Body(BK::Then | BK::Else | BK::For | BK::Function | BK::While),
                Stmt::Flow(FlowStmt::Jump(Jump::Discard), _),
            ) => Ok(()),

            // =================================================================
            // flow
            (B::ForIncrement | B::ForInit, Stmt::Flow(FlowStmt::Control(_), _)) => {
                Err(E::CannotIntroduceConditionalFlowIn(block.kind))
            }
            (
                B::EntryPoint | B::Body(BK::Then | BK::Else | BK::For | BK::Function | BK::While),
                Stmt::Flow(FlowStmt::Control(_), _),
            ) => Ok(()),
        }
    }
}
