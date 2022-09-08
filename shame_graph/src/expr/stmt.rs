use std::fmt::Display;

use crate::{pool::Key, Context};
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
    Return(Option<Key<Expr>>),
    Discard,
}

impl Display for StmtKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use StmtKind::*;
        match self {
            VariableDecl(Named(ty, _)) => f.write_fmt(format_args!("declaration of {ty} variable")),
            VariableDef(Named(ex, _)) => {
                Context::with(|ctx| {
                    match &ctx.exprs()[*ex] {e => 
                        f.write_fmt(format_args!("definition of {} variable via {} expression", e.ty, e.kind))
                    }
                })
            },
            Expr(ex) => {
                Context::with(|ctx| {
                    match &ctx.exprs()[*ex] {e => 
                        f.write_fmt(format_args!("expression statement {} of type {}", e.kind, e.ty))
                    }
                })
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
                Context::with(|ctx| {
                    match &ctx.exprs()[*ex] {e => 
                        f.write_fmt(format_args!("return {} expression of type {}", e.kind, e.ty))
                    }
                })
            },
            Return(None) => f.write_str("return without value"),
            Discard => f.write_str("discard"),
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
}
