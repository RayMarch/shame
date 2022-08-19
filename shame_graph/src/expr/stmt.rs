use super::*;
use crate::pool::Key;

#[derive(Clone)]
pub enum Flow {
    IfThen {
        cond: Key<Expr>,
        then: Key<Block>,
    },
    IfThenElse {
        cond: Key<Expr>,
        then: Key<Block>,
        els: Key<Block>,
    },
    For {
        init: Key<Expr>,
        cond: Key<Expr>,
        inc: Key<Expr>,
        body: Key<Block>,
    },
    While {
        cond: Key<Expr>,
        body: Key<Block>,
    },
}

#[derive(Clone)]
pub enum StmtKind {
    VariableDecl(Named<Ty>),
    VariableDef(Named<Key<Expr>>),
    Expr(Key<Expr>), //mostly for assignment-operators
    Flow(Flow),
    Return(Option<Key<Expr>>),
    Discard,
}

#[derive(Clone)]
pub struct Stmt {
    pub(crate) time: RecordTime,
    pub(crate) kind: StmtKind,
}

impl Stmt {
    pub fn new(time: RecordTime, kind: StmtKind) -> Self {
        Self { time, kind }
    }
}
