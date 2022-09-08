
use std::fmt::Display;
use std::{
    cell::{Cell, RefCell, Ref, RefMut},
    num::NonZeroU32,
};
use crate::*;

thread_local!(static CONTEXT: RefCell<Option<Context>> = RefCell::new(None));
thread_local!(static GENERATION: Cell<NonZeroU32> = Cell::new(NonZeroU32::new(999 /*arbitrary number*/).unwrap()));

pub(super) fn increment_thread_generation() -> NonZeroU32 {GENERATION.with(|g| g.increment_by(1))}
#[allow(unused)] 
pub(super) fn   current_thread_generation() -> NonZeroU32 {GENERATION.with(|g| g.get())}

pub struct Context {
    exprs : Pool<Expr>,
    blocks: Pool<Block>,
    items : Pool<Item>,
    idents: Pool<Option<String>>,
    
    warnings: RefCell<Vec<Warning>>,
    errors:   RefCell<Vec<Error>>,

    shader: RefCell<Shader>,
    misc: RefCell<Box<dyn std::any::Any>>,

    current_block: Cell<Option<Key<Block>>>,
    
    pub(crate) shader_kind: ShaderKind,
    error_behavior: ErrorBehavior,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderKind {
    Vertex,
    Fragment,
    Compute,
}

impl Display for ShaderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ShaderKind::Vertex => "vertex",
            ShaderKind::Fragment => "fragment",
            ShaderKind::Compute => "compute",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorBehavior {
    Panic,
    Result,
}

impl Context {

    #[track_caller] pub(crate) fn exprs_mut(&self) -> PoolRefMut<Expr> {self.exprs.borrow_mut()}
    #[track_caller] pub(crate) fn exprs  (&self) -> PoolRef   <Expr> {self.exprs.borrow  ()}

    #[track_caller] pub(crate) fn blocks_mut(&self) -> PoolRefMut<Block> {self.blocks.borrow_mut()}
    #[track_caller] pub(crate) fn blocks  (&self) -> PoolRef   <Block> {self.blocks.borrow  ()}

    #[track_caller] pub(crate) fn items_mut(&self) -> PoolRefMut<Item> {self.items.borrow_mut()}
    #[track_caller] pub(crate) fn items  (&self) -> PoolRef   <Item> {self.items.borrow  ()}

    #[track_caller] pub(crate) fn idents_mut(&self) -> PoolRefMut<Option<String>> {self.idents.borrow_mut()}
    #[track_caller] pub(crate) fn idents  (&self) -> PoolRef   <Option<String>> {self.idents.borrow  ()}

    #[track_caller] pub fn shader_mut(&self) -> RefMut<Shader> {self.shader.borrow_mut()}
    #[track_caller] pub fn shader  (&self) -> Ref   <Shader> {self.shader.borrow  ()}

    #[track_caller] pub fn misc_mut(&self) -> RefMut<Box<dyn std::any::Any>> {self.misc.borrow_mut()}
    #[track_caller] pub fn misc  (&self) -> Ref   <Box<dyn std::any::Any>> {self.misc.borrow  ()}

    #[track_caller] pub(crate) fn errors_mut(&self) -> RefMut<Vec<Error>> {self.errors.borrow_mut()}
    #[allow(unused)] #[track_caller] pub(crate) fn errors(&self) -> Ref<Vec<Error>> {self.errors.borrow  ()}

    #[allow(unused)] #[track_caller] pub(crate) fn warnings_mut(&self) -> RefMut<Vec<Warning>> {self.warnings.borrow_mut()}
    #[allow(unused)] #[track_caller] pub(crate) fn warnings  (&self) -> Ref   <Vec<Warning>> {self.warnings.borrow  ()}

    #[track_caller] pub fn shader_kind(&self) -> ShaderKind {
        self.shader_kind
    }

    pub(crate) fn new(generation: NonZeroU32, shader_kind: ShaderKind, error_behavior: ErrorBehavior) -> Self {     
        Self {
            exprs:  Pool::new(generation),
            blocks: Pool::new(generation),
            items:  Pool::new(generation),
            idents: Pool::new(generation),
            warnings: RefCell::new(Vec::new()),
            errors:   RefCell::new(Vec::new()),
            shader:   RefCell::new(Shader::new(shader_kind)),
            misc:     RefCell::new(Box::new(())),
            current_block: Cell::new(None),
            shader_kind,
            error_behavior,
        }
    }

    pub fn push_warning(&mut self, w: Warning) {
        self.warnings.borrow_mut().push(w)
    }

    #[track_caller]
    pub fn push_error(&self, e: Error) {
        match self.error_behavior {
            ErrorBehavior::Panic => panic!("{}", e),
            ErrorBehavior::Result => self.errors_mut().push(e),
        }
    }

    pub fn with_thread_local_context_enabled(shader_kind: ShaderKind, error_behavior: ErrorBehavior, f: impl FnOnce()) -> Context {
        let generation = increment_thread_generation();
        let new_ctx = Context::new(generation, shader_kind, error_behavior);

        CONTEXT.with(|thread_ctx| {
            assert!(thread_ctx.borrow().is_none(), "cannot start a new recording context, another context is still present on this thread");
            *thread_ctx.borrow_mut() = Some(new_ctx);
        });

        f();

        Context::with(|ctx| {
            ctx.post_process();

            //checks whether attribute locations overlap or bind-group indices are reused
            if let Err(e) = ctx.shader().validate_interface() {
                ctx.push_error(e)
            }
        });
        
        CONTEXT.with(|thread_ctx| {
            thread_ctx.borrow_mut().take().unwrap() 
        })
    }

    fn post_process(&self) {
        let mut blocks = self.blocks_mut();

        for (key, expr) in self.exprs_mut().enumerate() {
            if expr.needs_variable_def_stmt() {
                let ident = expr.ident.get_or_insert_with(|| IdentSlot::new_in(None, &mut self.idents_mut()));
                let stmt = Stmt::new(expr.time, StmtKind::VariableDef(Named(key, *ident)));
                blocks[expr.parent_block].stmts.push(stmt);
            }
            else if expr.needs_expr_stmt() {
                let stmt = Stmt::new(expr.time, StmtKind::Expr(key));
                blocks[expr.parent_block].stmts.push(stmt);
            }
        }

        //sort all statements within a block
        for block in blocks.iter_mut() {
            block.stmts.sort_by_key(|a| a.time);
        }
    }

    #[track_caller]
    pub fn with_mut<R>(f: impl FnOnce(&mut Context) -> R) -> R {
        CONTEXT.with(|ctx| {
            let mut ctx = ctx.borrow_mut();
            let ctx = ctx.as_mut().expect("Context::with_mut with no active recording");
            f(ctx)
        })
    }

    #[track_caller]
    pub fn with<R>(f: impl FnOnce(&Context) -> R) -> R {
        CONTEXT.with(|ctx| {
            let ctx = ctx.borrow();
            let ctx = ctx.as_ref().expect("Context::with_mut with no active recording");
            f(ctx)
        })
    }

    #[track_caller]
    pub fn try_with<R>(f: impl FnOnce(&Context) -> R) -> Option<R> {
        CONTEXT.with(|ctx| ctx.borrow().as_ref().map(f))
    }
}

#[derive(Clone, Copy)]
pub enum BranchState {
    Branch,
    BranchWithConditionNotAvailable, //e.g. when you're branching on a fragment shader value but you're recording a vertex shader
}

impl Context {

    /// whether the recording thread is currently in a closure of a branch recording
    /// such as if-then/if-then-else/for/while
    pub fn inside_branch(&self) -> Option<BranchState> {
        let blocks = self.blocks();
        let stack = self.stack_blocks(&blocks);

        // we are inside a branch if any of the blocks in the stack are part of a branch
        stack.fold(None, |acc, block_key| {
            use BranchState::*;
            let branch_state = blocks[block_key].is_branch;
            match (acc, branch_state) {
                (None, None) => None,
                (None, Some(x)) => Some(x),
                (Some(x), None) => Some(x),
                (Some(x), Some(y)) => Some(match (x, y) {
                    (Branch, Branch) => Branch,
                    (Branch, BranchWithConditionNotAvailable) => BranchWithConditionNotAvailable,
                    (BranchWithConditionNotAvailable, Branch) => BranchWithConditionNotAvailable,
                    (BranchWithConditionNotAvailable, BranchWithConditionNotAvailable) => BranchWithConditionNotAvailable,
                }),
            }
        })
    }

    pub(crate) fn stack_blocks<'a>(&'a self, blocks: &'a PoolRef<Block>) -> impl Iterator<Item=Key<Block>> + Clone + 'a {
        start_iter_from(Some(self.current_block_key_unwrap()), move |key| blocks[key].parent)
    }

    pub(crate) fn current_block_key_unwrap(&self) -> Key<Block> {
        self.current_block.get().expect("no current block exists yet to record into")
    }

    pub fn record_shader_main(&self, f: impl FnOnce()) {
        assert!(self.current_block.get().is_none());

        //establish item <-> block link
        let item  = self.items_mut().push(Item::MainFuncDef {body: Default::default()});
        let block = self.blocks_mut().push(Block::new(None, item, None, BlockKind::Body));
        unwrap_variant!(&self.items()[item], Item::MainFuncDef{body} => body.set(Some(block)));

        self.current_block.set(Some(block));
        f();
        self.current_block.set(None);
    }

    pub fn nested_record_function<R>(&self, ident: IdentSlot, f: impl FnOnce() -> R) -> R {
        assert!(self.current_block.get().is_some());

        //establish item <-> block link
        let item  = self.items_mut().push(Item::FuncDef {ident, body: Default::default(), args: Default::default()});
        
        let parent = self.current_block.take();
        let block = self.blocks_mut().push(Block::new(parent, item, None, BlockKind::Body));

        unwrap_variant!(&self.items()[item], Item::FuncDef{body, ..} => body.set(Some(block)));
        
        self.current_block.set(Some(block));
        let result = f();
        let block_ = self.current_block.replace(parent);

        assert!(block_ == Some(block));

        result
    }
    
    pub(crate) fn record_nested_block<R>(&self, is_branch: Option<BranchState>, f: impl FnOnce() -> R) -> (R, Key<Block>) {
        assert!(self.current_block.get().is_some());
        
        let mut blocks = self.blocks_mut();

        //establish item <-> block link
        let parent_block = &blocks[self.current_block_key_unwrap()];
        if !parent_block.kind.may_contain_nested_blocks() {
            self.push_error(Error::BlockRestrictionsViolated(
                format!("cannot record nested block in {} recording", parent_block.kind)
            ))
        }
        let item = blocks[self.current_block_key_unwrap()].origin_item;
        
        let parent = self.current_block.take();
        let child = blocks.push(Block::new(parent, item, is_branch, BlockKind::Body));
        drop(blocks);

        self.current_block.set(Some(child));
        let result = f();
        let child_ = self.current_block.replace(parent);

        assert!(child_ == Some(child));
        (result, child)
    }

    /// doesn't define a new struct if an identical struct was already recorded
    pub fn get_or_insert_struct(&self, name: &str, fields: &[(&str, Ty)]) -> Struct {

        //check if struct matches name and fields without creating a new one
        for item in self.items().iter() {
            if let Item::StructDef(struct_) = item {
                if struct_.eq_name_fields(name, fields) {
                    return struct_.clone()
                }
            }
        }

        let new_struct = Struct::from_name_fields(self, name, fields);

        //insert new struct definition
        self.items_mut().push(Item::StructDef(new_struct.clone()));

        new_struct
    }

}