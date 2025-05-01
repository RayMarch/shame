use node::expr::ExprCategory;
use thiserror::Error;

use super::*;
use crate::call_info;
use crate::common::pool::Key;
use crate::common::pool::PoolRef;
use crate::common::pool::PoolRefMut;
use crate::common::proc_macro_reexports::ir;
use crate::common::small_vec::SmallVec;
use crate::frontend::any::ArgumentNotAvailable;
use crate::frontend::encoding::EncodingError;
use crate::frontend::encoding::EncodingErrorKind;
use crate::frontend::encoding::EncodingErrors;
use crate::frontend::error::InternalError;
use crate::ir::expr;
use crate::ir::expr::Expr;
use crate::ir::expr::FnRelated;
use crate::ir::expr::NoMatchingSignature;
use crate::ir::expr::RefLoad;
use crate::ir::expr::ShaderIo;
use crate::ir::expr::TypeCheck;
use crate::ir::ir_type;
use crate::ir::pipeline;
use crate::ir::pipeline::PipelineKind;
use crate::ir::pipeline::PossibleStages;
use crate::ir::pipeline::StageMask;
use crate::ir::Comp4;
use crate::ir::Type;
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt::Display;
use std::ops::DerefMut;
use std::rc::Rc;
use std::slice::SliceIndex;

/// A `Node` represents a *Value* in the WGSL sense (not a *Variable*!).
///
/// see https://www.w3.org/TR/WGSL/#var-vs-value
///
/// (for variables, see [`MemoryRegion`])
#[derive(Debug, Clone)]
pub struct Node {
    pub(crate) call_info: CallInfo,
    pub(crate) args: Box<[Key<Node>]>,
    pub(crate) expr: expr::Expr,
    /// `ty` follows from `args` and `expr`.
    pub(crate) ty: ir_type::Type,
    /// If this *Value* has an associated identifier (e.g. via `let` binding)
    /// see https://www.w3.org/TR/WGSL/#value-decls
    ///
    /// `Ident` bindings may be forced by `shame` in some cases,
    /// or if the user sets a custom ident for easier debuggability of the generated code.
    pub(crate) ident: Option<Key<Ident>>,
    /// whether `self` is part of a statement, such as [`ExprStmt::Condition`], [`Jump::Return`], etc.
    ///
    /// if and only if, for any statement `stmt`,`stmt.used_nodes()` contains this node,
    /// then `self.is_part_of_stmt` is `true`, otherwise `false`
    pub(crate) is_part_of_stmt: bool,
    /// the block that `self` was recorded in. This implies the scope in which `self` is valid.
    pub(crate) block: Key<Block>,
    pub(crate) stages: PossibleStages,
    pub(crate) time: TimeInstant,
    private_ctor: (),
}

#[derive(Debug, Error, Clone)]
pub enum NodeRecordingError {
    #[error(
        "access chain/decomposition expressions cannot be bound to user defined identifiers (`{1}`). Expression: {0:?}"
    )]
    AccessChainExprCannotHaveUserIdent(Expr, String),
    #[error("empty arrays are not allowed")]
    ArraysMustBeNonEmpty,
    #[error("cannot access component {component} of a {len} component vector")]
    InvalidSwizzleComponent { len: u8, component: &'static str },
    #[error("{0}")]
    NoMatchingSignature(#[from] NoMatchingSignature),
    #[error("no matching overload found for {0:?} with the provided arguments")]
    NoOverloadFound(Expr),
    #[error("{0}")]
    ArgumentNotAvailable(#[from] ArgumentNotAvailable),
    #[error("the `AddressOf` operator cannot be used on vector components. ")]
    CannotTakeAddressOfVectorComponents,
    #[error("the shader input/output `{0}` may only be created once. It was created before at {1}")]
    SameShaderIoSetMultipleTimes(ShaderIo, CallInfo),
    #[error("barriers/synchronization functions are only allowed in compute pipelines")]
    BarrierInNonComputePipeline,
    #[error("trying to access workgroup grid size in a non-compute pipeline")]
    WorkgroupGridSizeNotAvailable,
    #[error(
        "The special edge-clmaped mip-level 0 sampling function does not support textures which are part of a texture array"
    )]
    TextureArrayElementsCannotSampleEdgeClampLevel0,
    #[error("trying to splat non-scalar value of type `{0}`")]
    TryingToSplatNonScalar(Type),
    #[error("trying to extend type `{0}` to length {0}")]
    UnableToExtendType(Type, ir::Len2),
    #[error("the shader input/output `{0}` can not be used in a {1}")]
    WrongPipelineKindForShaderIo(ShaderIo, PipelineKind),
    #[error(
        "value used out of scope.\n\
    Argument of type `{value_ty}` originating from [ {value_loc} ] is used outside of its scope.\n\
    Scope defined at [ {valid_scope_defined_at} ]\n\
    but value is used at [ {invalid_usage_at} ] in expression of kind: {invalid_usage_expr:?}"
    )]
    ValueUsedOutOfScope {
        value_ty: Type,
        value_loc: CallInfo,
        valid_scope_defined_at: CallInfo,
        invalid_usage_at: CallInfo,
        invalid_usage_expr: Expr,
    },
}

impl Node {
    pub(crate) fn new(ctx: &Context, args: Box<[Key<Node>]>, expr: expr::Expr) -> Result<Self, NodeRecordingError> {
        let nodes = ctx.pool();
        let arg_types: SmallVec<_, 4> = args.iter().map(|key| nodes[*key].ty.clone()).collect();

        let ty = expr.infer_type(&arg_types)?;
        let time_before = TimeInstant::next(); // contains the "before" execution state

        if expr.may_change_execution_state() {
            // note: the execution state is different *after* the effect of this node, so
            // the node itself still has the same exec state.
            // otherwise any arguments to a node that changes exec state would always
            // have to be introduced as `let` declarations before being fed into this node.
            ctx.increment_execution_state();
        };

        let stages = expr
            .possible_stages()
            .restrict(PossibleStages::new_all_in_pipeline(ctx.pipeline_kind()));

        let node = Node {
            call_info: ctx.latest_user_caller(),
            args,
            expr,
            ty,
            ident: None,
            is_part_of_stmt: false,
            block: ctx.current_block(),
            time: time_before,
            stages,
            private_ctor: (),
        };

        // scope check
        let blocks = &ctx.pool();
        for arg_key in &node.args {
            let arg_node = &nodes[*arg_key];
            if Block::find_key_in_stack(node.block, blocks, |b| b == arg_node.block).is_none() {
                return Err(out_of_scope_error(ctx, blocks, &node, arg_node));
            }
        }
        Ok(node)
    }

    // TODO(release) remove
    pub(crate) fn call_info(&self) -> CallInfo { self.call_info }

    // TODO(release) remove
    pub(crate) fn ty(&self) -> &Type { &self.ty }

    pub(crate) fn is_writeable_binding(&self) -> bool {
        match &self.expr {
            Expr::PipelineIo(ir::expr::PipelineIo::Binding(_)) => match &self.ty {
                Type::Ptr(_, _, access_mode) |
                Type::Ref(_, _, access_mode) |
                Type::Store(ir::StoreType::Handle(ir::HandleType::StorageTexture(_, _, access_mode))) => {
                    access_mode.is_writeable()
                }
                _ => false,
            },
            _ => false,
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        let mut args = self.args.iter();
        if let Some(arg) = args.next() {
            write!(f, "{}", arg.index())?;
        }
        for arg in args {
            write!(f, ", {}", arg.index())?;
        }
        write!(f, ") ")?;
        writeln!(f, "{} ", self.call_info)?;
        write!(f, "{}: ", self.expr)?;
        write!(f, "{} ", self.ty)?;

        Ok(())
    }
}

/// called during the `Context::finish` stage of recording when all nodes are known,
/// and when the stage-solver already ran.
pub(crate) fn create_statements_for_nodes_that_require_it(ctx: &Context) -> Result<(), EncodingError> {
    let mut node_pool_mut = ctx.pool_mut::<Node>();
    let pool_gen = node_pool_mut.generation();

    let nodes: &Vec<Node> = &node_pool_mut;

    // gather adjacency and type info about each node,
    // so we can decide which ones need which kind of statement to be emitted
    let stances = {
        let mut stances: Box<[_]> = nodes.iter().map(NodeCircumstances::init_with_node).collect();
        assert!(nodes.len() == stances.len());

        // iterate over all (must_appear-)nodes and accumulate their usage count
        for (node_i, node) in nodes.iter().enumerate() {
            let stance = &mut stances[node_i];
            // don't propagate the usage counts of refs, since long access chains (>= 2) would already
            // trigger a `let binding` stmt emission for anything at the end of the access chain.
            if stance.type_category != TypeCategory::Ref {
                propagate_node_usages(ctx, nodes, &mut stances, node)?;
            }
        }
        stances
    };

    let mut idents = ctx.pool_mut();

    for node_key in node_pool_mut.keys() {
        let stance = stances[node_key.index()];
        let node = &mut node_pool_mut[node_key];
        let is_already_part_of_stmt = node.is_part_of_stmt;

        // `reason` just captures which match arm was matched, for debugging reasons
        let (needs_stmt, _reason) = match stance {
            // fn parameter expressions are the only expressions that (at the time of writing) can be `Ref`s
            // but also emit a statement, due to how function recording is implemented right now.
            s if s.is_fn_param => (false, 1),

            // expressions returning `Ref`s (aka access chains) cannot be `let` bound to an identifier
            // since WGSL would trigger a load in that case, so it would become a value and no longer be a `Ref`.
            // In `shame`, this kind of load is explicit via the `Expr::RefLoad` (=Ref::get).
            s if s.type_category == TypeCategory::Ref && !s.is_fn_param => (false, 2),

            // expressions that alter the `ExecutionState` (think barriers, assignments) must
            // be turned into statements, so that other expressions can be placed before or after them
            // by putting their respective statements in the lines above or below this statement
            s if s.expr_category == ExprCategory::ChangesExecutionState => (true, 3),

            // expressions (for example RefLoad) whose values are used in a later execution state
            // must be bound to `let` bindings before any execution state change happens, since
            // if they slip into any later execution state, they might evaluate to something different.
            s if s.used_from_another_execution_state => (true, 4),

            // expressions which already have been given ident-hints either by the user or by
            // the rust abstraction layer for readability will need a `let` binding to introduce
            // that identifier. (unless they are fn params, which don't require let bindings)
            s if s.has_had_ident_created_during_recording && !s.is_fn_param => (true, 5),

            // expressions which are used more than once are turned into `let` bindings to
            // avoid generating the same (sometimes complex) expression twice.
            s if s.usage_count_from_nodes_that_must_appear >= 2 &&
                s.expr_category != ExprCategory::BindToIdentUndesirable =>
            {
                (true, 6)
            }

            // expressions which must appear, but are never used as args, and are not forced
            // into statements by any of the other above rules, must be turned into statements
            // to satisfy the `must_appear` requirement. (at the time of writing this affects only the `Expr::Show` expr)
            s if s.must_appear && s.usage_count_from_nodes_that_must_appear == 0 => (true, 7),
            _ => (false, 0),
        };

        if needs_stmt && !is_already_part_of_stmt {
            let stmt = match stance.type_category {
                TypeCategory::Other => {
                    node.ident.get_or_insert_with(|| idents.push(Ident::Unchosen));
                    ExprStmt::IntroduceIdent(node_key)
                }
                TypeCategory::Unit => ExprStmt::Expr(node_key),
                TypeCategory::Ref => unreachable!("corresponding arm in previous match is 'false'"),
            };

            let solved_stages = node.stages.clone();
            let (node_block, node_time, node_call_info) = (node.block, node.time.clone(), node.call_info);

            if cfg!(feature = "debug_print") && cfg!(debug_assertions) {
                //println!("emitting stmt for reason #{} expr: {}: {}", reason, node.expr, node.ty);
            }

            // push_to_block needs a shared node pool, maybe we can clean this up at some point
            stmt.push_to_block(ctx, &mut node_pool_mut, node_block, node_time, node_call_info)
                .map_err(|e| ctx.assemble_error(node_call_info, e))?;
        }
    }
    Ok(())
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TypeCategory {
    Unit,
    Ref,
    Other,
}

/// information that is derived from different results of the recording process, which is
/// relevant for deciding whether/which kind of statement needs to be emitted for this node.
#[derive(Copy, Clone, Debug)]
struct NodeCircumstances {
    is_fn_param: bool,
    expr_category: ExprCategory,
    type_category: TypeCategory,
    /// whether this node was not eliminated by dead code elimination
    must_appear: bool,
    /// whether the node already has had an ident assigned to it during recording,
    /// either via user-hint, or via auto-hint. Ident suggestions by `shame` help
    /// readability sometimes.
    has_had_ident_created_during_recording: bool,
    /// whether this node is an arg of another node that has a different execution state
    used_from_another_execution_state: bool,
    /// how often this node is an arg of another node that was not
    /// eliminated by dead-code-elimination in the stage solver
    usage_count_from_nodes_that_must_appear: u32,
}

impl NodeCircumstances {
    fn init_with_node(node: &Node) -> Self {
        Self {
            expr_category: node.expr.classify(),
            is_fn_param: match node.expr {
                Expr::FnRelated(FnRelated::FnParamMemoryView(_, _) | FnRelated::FnParamValue(_)) => true,
                Expr::FnRelated(FnRelated::Call(_)) => false,
                _ => false,
            },
            type_category: match node.ty {
                Type::Unit => TypeCategory::Unit,
                Type::Ptr { .. } => TypeCategory::Other,
                Type::Ref(_, _, _) => TypeCategory::Ref,
                Type::Store(_) => TypeCategory::Other,
            },
            has_had_ident_created_during_recording: node.ident.is_some(),
            must_appear: node.stages.must_appear_at_all(),
            used_from_another_execution_state: false,
            usage_count_from_nodes_that_must_appear: 0,
        }
    }
}

/// propagates the usage counters of a node to its (transitive (if acces chain)) args
fn propagate_node_usages(
    ctx: &Context,
    nodes: &[Node],
    nodes_stances: &mut [NodeCircumstances],
    node: &Node,
) -> Result<(), EncodingError> {
    if node.stages.must_appear_at_all() {
        for arg in node.args.iter() {
            let arg_i = arg.index();

            let (Some(arg_info), Some(arg_node)) = (nodes_stances.get_mut(arg_i), nodes.get(arg_i)) else {
                return Err(ctx.assemble_error(
                    node.call_info,
                    InternalError::new(
                        true,
                        format!(
                            "node has arg with index `{arg_i}` outside of the valid range.\narg generation: {}",
                            arg.generation()
                        ),
                    ),
                ));
            };

            // TODO(release) this is the new meaning of "access chain", consider getting rid of the classification on `Expr` that uses the same term
            let arg_is_access_chain = arg_node.ty().is_ref();

            match arg_is_access_chain {
                true => propagate_node_usages(ctx, nodes, nodes_stances, arg_node)?,
                false => {
                    arg_info.usage_count_from_nodes_that_must_appear += 1;

                    match node.time.exec_state().cmp(&arg_node.time.exec_state()) {
                        Ordering::Less => return Err(decreasing_exec_state_error(ctx, node, arg_node)),
                        Ordering::Equal => (),
                        Ordering::Greater => arg_info.used_from_another_execution_state = true,
                    }
                }
            }
        }
    }
    Ok(())
}

fn out_of_scope_error(ctx: &Context, blocks: &PoolRef<'_, Block>, node: &Node, arg_node: &Node) -> NodeRecordingError {
    NodeRecordingError::ValueUsedOutOfScope {
        value_ty: arg_node.ty.clone(),
        value_loc: arg_node.call_info,
        valid_scope_defined_at: blocks[arg_node.block].call_info,
        invalid_usage_at: node.call_info,
        invalid_usage_expr: node.expr.clone(),
    }
}

fn decreasing_exec_state_error(ctx: &Context, node: &Node, arg_node: &Node) -> EncodingError {
    ctx.assemble_error(
        node.call_info,
        InternalError::new(
            true,
            format!(
                "node {node:?} with exec state `{:?}` has argument `{arg_node:?}` with a larger exec state `{:?}`.",
                node.time.exec_state(),
                arg_node.time.exec_state()
            ),
        ),
    )
}
