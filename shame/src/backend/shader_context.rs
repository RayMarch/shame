use std::{cell::RefCell, collections::BTreeSet, rc::Rc};

use super::code_write_buf::{IndentTracker, Indentation};
use crate::{
    common::pool::Key,
    ir::{
        expr::Expr,
        recording::{self, AllocStmt, Block, ExprStmt, FinalIdents, Ident, MemoryRegion, Node, Stmt},
        AddressSpace, StoreType,
    },
};

pub struct ShaderContext<'a, T> {
    pub ctx: &'a recording::Context,
    pub idents: FinalIdents,
    pub indent: IndentTracker,
    /// this vec of definitions is filled up as struct names are being
    /// written to the shader code. Afterwards it contains all necessary
    /// struct definitions
    pub non_fn_allocations: Vec<Rc<MemoryRegion>>,
    pub info: T,
}

impl<'a, T> ShaderContext<'a, T> {
    #[allow(clippy::single_match)]
    pub fn new(ctx: &'a recording::Context, idents: FinalIdents, t: T) -> Self {
        let mut non_fn_allocations = vec![];
        // this iteration of the node pool can be avoided if we decide
        // to collect all this info during recording
        for block in ctx.pool::<Block>().iter() {
            for (stmt, _, _) in &block.stmts {
                if let Stmt::Allocate(AllocStmt {
                    allocation,
                    initial_value: init,
                }) = stmt
                {
                    if allocation.address_space != AddressSpace::Function {
                        non_fn_allocations.push(allocation.clone())
                    }
                }
            }
        }

        Self {
            ctx,
            idents,
            indent: Default::default(),
            non_fn_allocations,
            info: t,
        }
    }
}
