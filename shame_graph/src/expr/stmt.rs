use std::fmt::Display;

use crate::{pool::Key, Context, Error};
use super::*;

#[derive(Clone, Debug)]
pub enum Flow {
    IfThen      {cond: Key<Expr>, then: Key<Block>},
    IfThenElse  {cond: Key<Expr>, then: Key<Block>, els: Key<Block>},
    For         {init: Key<Block>, cond: Key<Block>, inc: Key<Block>, body: Key<Block>},
    While       {cond: Key<Block>, body: Key<Block>},
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    VariableDecl(Named<Ty>),
    VariableDef(Named<Key<Expr>>),
    Expr(Key<Expr>), //mostly for assignment-operators
    Flow(Flow),
    Continue,
    Break,
    Return(Option<Key<Expr>>),
    Discard,
}

impl Display for StmtKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use StmtKind::*;
        let is_recording = Context::is_currently_recording_on_this_thread();
        match self {
            VariableDecl(Named(ty, _)) => f.write_fmt(format_args!("declaration of {ty} variable")),
            VariableDef(Named(ex, _)) => {
                match is_recording {
                    true => Context::with(|ctx| {
                        match &ctx.exprs()[*ex] {e => 
                            f.write_fmt(format_args!("definition of {} variable via {} expression", e.ty, e.kind))
                        }
                    }),
                    false => f.write_fmt(format_args!("variable definition"))
                }
            },
            Expr(ex) => {
                match is_recording {
                    true => Context::with(|ctx| {
                        match &ctx.exprs()[*ex] {e => 
                            f.write_fmt(format_args!("expression ({}) of type {}", e.kind, e.ty))
                        }
                    }),
                    false => f.write_fmt(format_args!("expression"))
                }
            },
            Flow(flow) => {
                use super::Flow::*;
                match flow {
                    IfThen    {..} => f.write_str("if-then"),
                    IfThenElse{..} => f.write_str("if-then-else"),
                    For       {..} => f.write_str("for-loop"),
                    While     {..} => f.write_str("while-loop"),
                }
            },
            Return(Some(ex)) => {
                match is_recording {
                    true => Context::with(|ctx| {
                        match &ctx.exprs()[*ex] {e => 
                            f.write_fmt(format_args!("return {} expression of type {}", e.kind, e.ty))
                        }
                    }),
                    false => f.write_fmt(format_args!("return"))
                }
            },
            Return(None) => f.write_str("return without value"),
            Discard => f.write_str("discard"),
            Continue => f.write_str("continue"),
            Break => f.write_str("break"),
        }
    }
}

#[derive(Clone)]
pub struct Stmt {
    pub(crate) time: RecordTime,
    pub(crate) kind: StmtKind,
}

impl Stmt {
    pub fn new(time: RecordTime, kind: StmtKind) -> Self {
        Self {
            time, 
            kind,
        }
    }

    fn record_stmt(kind: StmtKind) {
        Context::with(|ctx| match &mut ctx.blocks_mut()[ctx.current_block_key_unwrap()] {
            block => block.add_stmt(Stmt::new(RecordTime::next(), kind))
        });
    }

    pub fn record_break() {
        Context::with(|ctx| match ctx.is_inside_loop_body() {
            true => Stmt::record_stmt(StmtKind::Break),
            false => ctx.push_error(Error::IllegalStatement(
                format!("break statement used outside of loops"),
            ))
        })
    }

    pub fn record_continue() {
        Context::with(|ctx| match ctx.is_inside_loop_body() {
            true => Stmt::record_stmt(StmtKind::Continue),
            false => ctx.push_error(Error::IllegalStatement(
                format!("continue statement used outside of loops"),
            ))
        })
    }

    pub fn record_discard() {
        use crate::BranchState::*;
        
        let shader_kind = Context::with(|ctx| {
            let blocks = ctx.blocks();
            let mut stack = ctx.stack_blocks(&blocks);

            let is_foreign_stage_conditional_block = stack.find(|key| 
                matches!(blocks[*key].is_branch, Some(BranchWithConditionNotAvailable))
            ).is_some();

            if is_foreign_stage_conditional_block {
                ctx.push_error(Error::IllegalStatement(
                    "trying to use discard in a conditional block that does not branch on a fragment stage condition. Discard is a fragment stage statement.".to_string()
                ))
            }

            ctx.shader_kind()
            
        });

        if shader_kind == crate::ShaderKind::Fragment {
            Stmt::record_stmt(StmtKind::Discard)
        }
    }
}
