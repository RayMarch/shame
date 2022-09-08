use std::{cell::Cell};

use crate::{context::Context, error::Error, pool::{Key, PoolRef, PoolRefMut}};
use super::*;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecordTime(u32);

impl RecordTime {
    pub fn next() -> Self {
        thread_local! {static NEXT: Cell<u32> = Cell::new(0);}
        let time = NEXT.with(|x| x.get());
        NEXT.with(|x| x.set(time + 1));
        Self (time)
    }
}

pub struct Expr {
    pub(crate) ty: Ty,
    pub(crate) kind: ExprKind,
    pub(crate) ident: Option<IdentSlot>,
    pub(crate) args: Vec<Key<Expr>>,
    pub(crate) parent_block: Key<Block>,
    pub(crate) ident_req: Cell<IdentRequirement>,
    pub(crate) time: RecordTime,
}

#[derive(Debug, Clone, Copy)]
pub enum IdentRequirement {
    RefCount(u8), //refcount < 2 means we don't need to make this expression a variable (by giving it an identifier and a VariableDef statement)
    IdentNeeded, //refcount >= 2 and another condition (see update_statement_requirements) means we need an identifier
}

/// traverses upward from a given expression until a non-lvalue is found
/// if the provided expression is already not an lvalue, no traversal happens
pub(crate) fn find_closest_ancestor_non_lvalue(exprs: &PoolRef<Expr>, mut key: Key<Expr>) -> Key<Expr> {
    loop {
        let expr = &exprs[key];
        if expr.ty.access != Access::LValue {break key}
        match expr.args.as_slice() {
            [arg,..] => key = *arg,
            [] => panic!("trying to traverse upward from lvalue expression {:?} which has no parent arguments", expr.kind)
        }
    }
}

fn update_ident_requirements(kind: &ExprKind, block: Key<Block>, args: &[Key<Expr>]) {
    use IdentRequirement::*;
    Context::with(|ctx| {
        let exprs = ctx.exprs();
        for (arg_i, arg) in args.iter().enumerate() {
            let val = find_closest_ancestor_non_lvalue(&exprs, *arg);
            let val = &exprs[val];

            //increment the refcount, if we're at a count of 2 the argument requires an identifier and a VariableDef statement
            val.ident_req.set(match val.ident_req.get() {
                RefCount(0) => RefCount(1), //expr only used once. no identifier needed yet
                RefCount(1) => IdentNeeded, //would be RefCount(2) => we need an identifier for this argument
                IdentNeeded => IdentNeeded, //stays the same
                x => panic!("unexpected enum in update_statement_requirements {:?}", x)
            });

            //we always need an identifier if...
            if block != val.parent_block //...if our expression references an argument across a block boundary 
                || kind.is_mutating_arg_with_index(arg_i) //...if our expression needs the first arg to be an lvalue (e.g. +=, *= etc)
                {
                val.ident_req.set(IdentNeeded);
            }
        }
    });
}

fn validate_argument_scope(args: &[Key<Expr>]) {
    Context::with(|ctx| {
        let exprs = ctx.exprs();
        let blocks = ctx.blocks();

        let stack = ctx.stack_blocks(&blocks);

        //check whether all argument exprs were created in a block 
        //that is present in the current block stack.
        for expr in args.iter().map(|key| &exprs[*key]) {
            if !stack.clone().any(|x| x == expr.parent_block) {

                let ident = expr.ident
                .and_then(|slot| ctx.idents()[*slot].clone())
                .map(|s| format!("'{s}' "))
                .unwrap_or_else(|| "".to_string());

                let ty = &expr.ty;

                ctx.push_error(Error::ScopeError(
                    format!("value {ident}(of type: {ty}) out of scope. This happens if you use a value that was \
                    created within an if-then/if-then-else/for/while control flow recording closure \
                    outside of that closure. Maybe you have used the `=` operator (which moves the reference) \
                    as opposed to `.set(...)` which performs assignment of the underlying value. \
                    Be careful when using `=` operators in control flow recordings, their usage can sadly \
                    not be supported since rust does not allow overloading/tracing of the `=` operator.\
                    ")
                ));
                break;
            };
        }

    })
}

impl Expr {

    pub fn new(ident: Option<IdentSlot>, kind: ExprKind, args: Vec<Key<Expr>>, parent_block: Key<Block>) -> Result<Self, Error> {
        let arg_types = args_as_types(&args);

        //error out if we try to write to read-only or read from write-only args
        validate_access(&kind, arg_types.as_slice(), &args).and_then(|_| {

            try_deduce_type(&kind, arg_types.as_slice()).map(|ty| {
                Self::new_internal(ident, ty, kind, args, parent_block)
            })

        })
    }

    fn new_internal(ident: Option<IdentSlot>, ty: Ty, kind: ExprKind, args: Vec<Key<Expr>>, parent_block: Key<Block>) -> Self {
        validate_argument_scope(&args);
        update_ident_requirements(&kind, parent_block, &args);
        Self {
            ty,
            kind,
            ident,
            args,
            parent_block,
            ident_req: Cell::new(IdentRequirement::RefCount(0)),
            time: RecordTime::next(),
        }
    }

    /// forces the expression to be bound to an identifier with the given name, 
    /// or a generated name.
    /// this most likely causes a variable definition statement to be inserted.
    /// Calling this function multiple times on the same expression will yield the
    /// same generated code as calling it only the last time. 
    pub fn force_ident(&mut self, maybe_name: Option<String>) {
        Context::with(|ctx| {
            self.ident_req.set(IdentRequirement::IdentNeeded);
            match self.ident {
                Some(ident_slot) => ctx.idents_mut()[*ident_slot] = maybe_name,
                None => self.ident = Some(IdentSlot::new_in(maybe_name, &mut ctx.idents_mut())),
            }
        });
    }

    pub fn needs_variable_def_stmt(&self) -> bool {
        match self.ident_req.get() {
            IdentRequirement::IdentNeeded => match self.kind {
                ExprKind::GlobalInterface(_) => false,
                ExprKind::BuiltinVar(_) => false,
                // ExprKind::Literal(_) => false,
                ExprKind::Copy {..} => true,
                _ => true
            }
            _ => match self.kind {
                ExprKind::Copy {..} => true,
                _ => false
            }
        }
    }

    pub fn needs_expr_stmt(&self) -> bool {
        self.kind.is_mutating_any_arg() //if the expression mutates any state it needs to be put into a statement
    }

    pub fn needs_loop_condition_expr_stmt(key: Key<Expr>, expr: &Expr, blocks: &PoolRefMut<Block>) -> bool {
        match blocks[expr.parent_block].kind {
            BlockKind::LoopCondition(Some(cond_key)) => key == cond_key,
            _ => false
        }
    }

}