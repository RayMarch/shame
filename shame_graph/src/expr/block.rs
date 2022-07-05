
use crate::{pool::Key, BranchState};
use super::*;

//contains item it belongs to, in order to decide whether a function boundary was crossed during 
//recording
pub struct Block {
    pub(crate) is_branch: Option<BranchState>,
    pub(crate) parent: Option<Key<Block>>,
    pub(crate) origin_item: Key<Item>,
    pub(crate) stmts: Vec<Stmt>,
}

impl Block {

    pub(crate) fn new(parent: Option<Key<Block>>, origin_item: Key<Item>, is_branch: Option<BranchState>) -> Block {
        Self {
            is_branch,
            parent,
            origin_item,
            stmts: Vec::new(),
        }
    }

    pub fn record_stmt(&mut self, stmt: Stmt) {
        self.stmts.push(stmt)
    }
}