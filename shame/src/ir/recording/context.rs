use super::{
    create_statements_for_nodes_that_require_it,
    pools::{GetPool, Pools},
    thread_local::CONTEXT,
    AllocStmt, Block, BlockError, BlockKind, CallInfo, CellNonZeroU32Ext, ExecutionState, ExprStmt, FlowStmt, Node,
    NodeRecordingError, RecordTime, Stmt, TimeInstant,
};
use crate::{
    frontend::any::shared_io::BindPath, ir::pipeline::InsufficientVisibilityError, results::LanguageCode, ShaderStage,
    Winding,
};
use crate::{
    backend::{self, language::Language},
    call_info,
    common::pool::{Key, PoolRef, PoolRefMut},
    frontend::{
        any::{Any, ArgumentNotAvailable, InvalidReason},
        encoding::{
            pipeline_info::{
                BindGroupLayout, BindingLayout, ComputeGridInfo, ComputePipeline, ComputePipelineInfo, ComputeShader,
                PipelineDefinition, RasterizerState, RenderPipeline, RenderPipelineInfo,
                RenderPipelinePushConstantRanges, RenderPipelineShaders,
            },
            EncodingError, EncodingErrorKind, EncodingErrors, Settings,
        },
        error::InternalError,
    },
    ir::{
        expr::{self},
        ir_type::StructRegistry,
        pipeline::{
            solve_shader_stages, PipelineError, PipelineKind, StageMask, WipComputePipelineDescriptor, WipPipeline,
            WipPipelineLayoutDescriptor, WipPushConstantsField, WipRenderPipelineDescriptor,
        },
        recording::{dedup_and_finalize_idents, FinalIdents},
    },
};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    collections::BTreeMap,
    num::NonZeroU32,
    sync::Arc,
};

/// updates `$ctx.latest_user_caller` appropriately with `$new_caller` if
/// needed, before executing `$expr_to_execute`, then restoring the previous
/// caller (if it has been updated at all).
///
/// this is a macro instead of a higher order function so that the
/// debugging callstack of the user does not get needlessly polluted.
macro_rules! with_updated_latest_user_caller {
    ($new_caller: expr, $ctx: ident, $expr_to_execute: expr) => {{
        // if `latest_user_caller` is `Some(_)`, the `$new_caller` is reentrant
        // from within the library and therefore not useful for error reporting.
        // Therefore the incoming `$new_caller` is ignored in that case.
        let latest_user_caller = $ctx.latest_user_caller_since_first.get().unwrap_or($new_caller);

        // remember whatever was in `latest_user_caller` before to 
        // restore it afterwards
        let restore = $ctx.latest_user_caller_since_first.replace(Some(latest_user_caller));
        let result = $expr_to_execute;
        $ctx.latest_user_caller_since_first.set(restore);
        result
    }};
}

pub struct Context {
    /// settings provided by the user on encoding start
    settings: Settings,
    /// the first call site recorded from the user
    first_user_caller: CallInfo,
    /// the latest call site happening in user-code. Nested context accesses within
    /// the library do not update this variable.
    latest_user_caller_since_first: Cell<Option<CallInfo>>,
    next_record_time: Cell<NonZeroU32>,
    next_execution_state: Cell<NonZeroU32>,
    current_block: Cell<Key<Block>>,
    errors: RefCell<Vec<EncodingError>>,
    struct_registry: RefCell<StructRegistry>,
    wip_pipeline: WipPipeline,
    pools: Pools,
}

impl Context {
    pub(crate) fn new(
        first_user_caller: CallInfo,
        settings: Settings,
        generation: NonZeroU32,
        kind: PipelineKind,
    ) -> Self {
        let pools = Pools::new(generation);
        let entry_point_block = Block::new_entry_point(first_user_caller, kind);
        let current_block = Block::get_pool(&pools).borrow_mut().push(entry_point_block);
        Self {
            settings,
            first_user_caller,
            latest_user_caller_since_first: Default::default(),
            wip_pipeline: WipPipeline::new(kind),
            next_record_time: Cell::new(NonZeroU32::MIN),
            next_execution_state: Cell::new(NonZeroU32::MIN),
            current_block: Cell::new(current_block),
            struct_registry: Default::default(),
            errors: Default::default(),
            pools,
        }
    }

    /// access the context mutably and register `call_info` to automatically be
    /// added to errors that are pushed within `f`'s execution.
    #[allow(unused)]
    pub(crate) fn with_mut<R>(call_info: CallInfo, f: impl FnOnce(&mut Context) -> R) -> R {
        CONTEXT.with(|ctx| {
            let mut ctx = ctx.borrow_mut();
            let ctx = ctx.as_mut().expect("Context::with_mut with no active recording");
            with_updated_latest_user_caller!(call_info, ctx, f(ctx))
        })
    }

    /// access the context and register `call_info` to automatically be
    /// added to errors that are pushed within `f`'s execution.
    pub(crate) fn with<R>(call_info: CallInfo, f: impl FnOnce(&Context) -> R) -> R {
        CONTEXT.with(|ctx| {
            let ctx = ctx.borrow();
            let ctx = ctx.as_ref().expect("Context::with_mut with no active recording");
            with_updated_latest_user_caller!(call_info, ctx, f(ctx))
        })
    }

    /// fallible version of `with`
    #[allow(clippy::manual_map)] //map would make the callstack deeper, bad for debugging experience
    pub(crate) fn try_with<R>(call_info: CallInfo, f: impl FnOnce(&Context) -> R) -> Option<R> {
        CONTEXT.with(|ctx| match ctx.borrow().as_ref() {
            Some(ctx) => Some(with_updated_latest_user_caller!(call_info, ctx, f(ctx))),
            None => None,
        })
    }

    /// returns the latest user caller info, appropriate for display in
    /// error messages or "shader code <-> rust" mapping
    pub(crate) fn latest_user_caller(&self) -> CallInfo {
        self.latest_user_caller_since_first
            .get()
            .unwrap_or(self.first_user_caller)
    }

    // TODO(release) this is mutable access across the crate, find a way to reduce this access scope
    pub(crate) fn current_block(&self) -> Key<Block> { self.current_block.get() }

    pub(crate) fn first_user_caller(&self) -> CallInfo { self.first_user_caller }

    pub(crate) fn generation(&self) -> NonZeroU32 { self.pools.generation() }

    pub(crate) fn settings(&self) -> &Settings { &self.settings }

    pub(crate) fn pipeline_kind(&self) -> PipelineKind { self.wip_pipeline.kind }

    pub(crate) fn next_record_time(&self) -> RecordTime { RecordTime::new(self.next_record_time.increment_by(1)) }

    pub(crate) fn increment_execution_state(&self) -> ExecutionState {
        if cfg!(feature = "debug_print") && cfg!(debug_assertions) {
            println!("-- inc exec state -- {}", self.latest_user_caller());
        }
        ExecutionState::new(self.next_execution_state.increment_by(1))
    }
    pub(crate) fn current_execution_state(&self) -> ExecutionState {
        ExecutionState::new(self.next_execution_state.get())
    }

    /// whether the recording is not in a nested scope such as blocks/function recordings
    ///
    /// returns the location where that scope was created for better errors
    pub(crate) fn currently_in_non_encoding_scope(&self) -> Option<CallInfo> {
        match &self.pool()[self.current_block.get()] {
            b if b.kind != BlockKind::EntryPoint => Some(b.call_info),
            _ => None,
        }
    }

    pub(crate) fn push_error_if_outside_encoding_scope(&self, action_in_error_message: &'static str) {
        if let Some(scope_call_info) = self.currently_in_non_encoding_scope() {
            self.push_error(BlockError::MustBeInEncodingScope(action_in_error_message, scope_call_info).into());
        }
    }

    #[track_caller]
    pub(crate) fn pool<T: GetPool>(&self) -> PoolRef<T> { T::get_pool(&self.pools).borrow() }

    #[track_caller]
    pub(crate) fn try_pool<T: GetPool>(&self) -> Option<PoolRef<T>> { T::get_pool(&self.pools).try_borrow() }

    #[track_caller]
    pub(crate) fn pool_mut<T: GetPool>(&self) -> PoolRefMut<T> { T::get_pool(&self.pools).borrow_mut() }

    pub(crate) fn pipeline(&self) -> &WipPipeline { &self.wip_pipeline }

    pub(crate) fn pipeline_layout_mut(&self) -> RefMut<WipPipelineLayoutDescriptor> {
        self.wip_pipeline.layout.borrow_mut()
    }

    pub(crate) fn struct_registry_mut(&self) -> RefMut<StructRegistry> { self.struct_registry.borrow_mut() }

    pub(crate) fn struct_registry(&self) -> Ref<StructRegistry> { self.struct_registry.borrow() }

    pub(crate) fn assert_pipeline_kind(&self, required: PipelineKind) {
        let current = self.pipeline_kind();
        if current != required {
            self.push_error(EncodingErrorKind::RequiresPipelineKind { current, required })
        }
    }

    #[track_caller]
    pub(crate) fn render_pipeline_mut(&self) -> RefMut<WipRenderPipelineDescriptor> {
        self.assert_pipeline_kind(PipelineKind::Render);
        self.wip_pipeline.special.render.borrow_mut()
    }

    #[track_caller]
    pub(crate) fn render_pipeline(&self) -> Ref<WipRenderPipelineDescriptor> {
        self.assert_pipeline_kind(PipelineKind::Render);
        self.wip_pipeline.special.render.borrow()
    }

    #[track_caller]
    pub(crate) fn compute_pipeline_mut(&self) -> RefMut<WipComputePipelineDescriptor> {
        self.assert_pipeline_kind(PipelineKind::Compute);
        self.wip_pipeline.special.compute.borrow_mut()
    }

    #[track_caller]
    pub(crate) fn compute_pipeline(&self) -> Ref<WipComputePipelineDescriptor> {
        self.assert_pipeline_kind(PipelineKind::Compute);
        self.wip_pipeline.special.compute.borrow()
    }

    pub(crate) fn push_error(&self, error: EncodingErrorKind) {
        self.push_assembled_error(self.assemble_error(self.latest_user_caller(), error));
    }

    // all push error functions flow through here
    pub(crate) fn push_assembled_error(&self, error: EncodingError) { self.errors.borrow_mut().push(error); }

    pub(crate) fn push_error_get_invalid_any(&self, error: EncodingErrorKind) -> Any {
        self.push_error(error);
        Any::new_invalid(InvalidReason::ErrorThatWasPushed)
    }

    pub(crate) fn assemble_error(&self, call_info: CallInfo, error: impl Into<EncodingErrorKind>) -> EncodingError {
        self.settings.assemble_error_fn()(call_info, error.into())
    }

    // TODO(release) low prio: merge these 3 functions, except the `expr` stmt one, which is used in node.rs to push statements that have no stage tracking of their own after the stage solver already ran
    pub(crate) fn push_flow_stmt_to_current_block(&self, stmt: FlowStmt, time: TimeInstant, call_info: CallInfo) {
        if let Err(error) = stmt.push_to_block(self, &mut self.pool_mut(), self.current_block(), time, call_info) {
            self.push_assembled_error(self.assemble_error(call_info, error))
        }
    }

    pub(crate) fn push_expr_stmt_to_current_block(&self, stmt: ExprStmt, time: TimeInstant, call_info: CallInfo) {
        if let Err(error) = stmt.push_to_block(self, &mut self.pool_mut(), self.current_block(), time, call_info) {
            self.push_assembled_error(self.assemble_error(call_info, error))
        }
    }

    pub(crate) fn push_alloc_stmt_to_current_block(&self, stmt: AllocStmt, time: TimeInstant, call_info: CallInfo) {
        if let Err(error) = stmt.push_to_block(self, &mut self.pool_mut(), self.current_block(), time, call_info) {
            self.push_assembled_error(self.assemble_error(call_info, error))
        }
    }

    pub(crate) fn replace_current_block(&self, replace_with: Key<Block>) -> Key<Block> {
        self.current_block.replace(replace_with)
    }

    pub(crate) fn push_node(&self, expr: expr::Expr, args: &[Any]) -> Any {
        // try accessing each argument's node and collect it into a slice.
        // if `None` not every argument is present

        let arg_nodes = match ArgumentNotAvailable::new(expr.clone(), args) {
            None => Ok(args.iter().filter_map(|a| a.node()).collect()),
            Some(arg_na) => Err(NodeRecordingError::ArgumentNotAvailable(arg_na)),
        };

        let new_node = arg_nodes.and_then(|arg_nodes| {
            // performs type checking etc.
            Node::new(self, arg_nodes, expr.clone())
        });

        match new_node {
            Ok(node) => {
                self.struct_registry_mut().find_and_register_new_structs_used_in_type(
                    &node.ty,
                    &mut self.pool_mut(),
                    node.call_info,
                );
                let key = self.pool_mut().push(node);
                if let expr::Expr::ShaderIo(io) = &expr {
                    self.wip_pipeline.register_shader_io(io, key, self);
                }
                Any::from_parts(Ok(key))
            }
            Err(e) => {
                self.push_error(e.into());
                Any::new_invalid(InvalidReason::ErrorThatWasPushed)
            }
        }
    }

    #[track_caller]
    pub(crate) fn finish(self) -> Result<PipelineDefinition, EncodingErrors> {
        if cfg!(feature = "debug_print") && cfg!(debug_assertions) {
            use crate::common::prettify::*;
            use std::fmt::Write;
            let mut s = String::new();
            for (i, node) in self.pool::<crate::ir::Node>().iter().enumerate() {
                set_color(&mut s, Some("#8888FF"), false);
                write!(s, "{i} {} {}", node.time.record_time(), node.time.exec_state());
                set_color(&mut s, Some("#505050"), true);
                writeln!(s, " <- {node}\n");
            }
            set_color(&mut s, None, true);
            print!("{s}");
        }

        {
            let blocks = self.pool();
            let current_block = &blocks[self.current_block()];
            if current_block.kind != BlockKind::EntryPoint {
                self.push_error(BlockError::UnclosedBlock(current_block.call_info).into())
            }
        }

        if let Err(e) = solve_shader_stages(&self) {
            self.push_assembled_error(self.assemble_error(e.call_info, e.kind))
        }

        // run this before the errors get extracted, since it may push new errors
        create_statements_for_nodes_that_require_it(&self)?;

        if let Some(errors) = EncodingErrors::from_vec(self.errors.take()) {
            return Err(errors);
        }

        // sort statements by record-time.
        for block in self.pool_mut::<Block>().iter_mut() {
            block.stmts.sort_by_key(|(_, t, _)| t.record_time())
        }

        let idents = dedup_and_finalize_idents(self.pool_mut(), self.settings.shader_identifier_prefix);

        self.collect_pipeline_definition(idents)
    }

    #[track_caller]
    fn collect_pipeline_definition(self, final_idents: FinalIdents) -> Result<PipelineDefinition, EncodingErrors> {
        let call_info = call_info!();

        let shader_code = match self.settings.lang {
            Language::Wgsl => backend::wgsl::generate_shader(&self, final_idents),
        }?;

        let into_err = self.settings.assemble_error_fn();
        let pipeline_kind = self.pipeline_kind();

        let pipeline_layout = &self.wip_pipeline.layout.borrow();
        let bind_groups = {
            let mut dict = BTreeMap::<u32, BindGroupLayout>::new();
            for (path, wip_binding) in pipeline_layout.bindings.iter() {
                // if we inferred that a stage is required which is not allowed by the user
                // we create an error.
                let user_vis = wip_binding.user_defined_visibility;
                let required_vis = self.pool()[wip_binding.node].stages.must_appear_in();


                if user_vis != (user_vis | required_vis) {
                    return Err(into_err(
                        wip_binding.call_info,
                        PipelineError::InsufficientVisibility(InsufficientVisibilityError {
                            path: *path,
                            user_defined_visibility: user_vis,
                            required_visibility: required_vis,
                            vertex_writable_storage_by_default_enabled: self
                                .settings()
                                .vertex_writable_storage_by_default,
                        })
                        .into(),
                    )
                    .into());
                }

                let binding = BindingLayout {
                    visibility: user_vis,
                    binding_ty: wip_binding.binding_ty.clone(),
                    shader_ty: wip_binding.shader_ty.clone(),
                };

                let BindPath(group_i, binding_i) = *path;
                let bind_group = dict.entry(group_i).or_default();
                if let Some(occupied) = bind_group.bindings.insert(binding_i, binding) {
                    return Err(into_err(
                        wip_binding.call_info,
                        PipelineError::DuplicateBindPath(*path, occupied.shader_ty).into(),
                    )
                    .into());
                }
            }
            dict
        };

        let (push_constant_ranges, push_constants_byte_size) = {
            WipPushConstantsField::extract_ranges_per_stage(
                StageMask::pipeline(pipeline_kind),
                &pipeline_layout.push_constants,
                &self.pool(),
            )
            .map_err(|err| into_err(call_info, err.into()))?
        };

        macro_rules! get_late_recorded {
            ($late_recorded: expr) => {
                $late_recorded.try_get(std::stringify!($late_recorded), &into_err)
            };
        }

        let pipeline_def = match pipeline_kind {
            PipelineKind::Render => {
                let skippable_fragment_stage = !self.wip_pipeline.may_have_fragment_effects();
                let render = self.wip_pipeline.special.render.into_inner();
                PipelineDefinition::Render(RenderPipeline {
                    label: None, //TODO(release) add api to add label
                    shaders: {
                        let shader_code = Arc::new(match self.settings.lang {
                            Language::Wgsl => LanguageCode::Wgsl(shader_code),
                        });
                        RenderPipelineShaders {
                            vert_entry_point: "vert_main",
                            frag_entry_point: "frag_main",
                            vert_code: shader_code.clone(),
                            frag_code: shader_code,
                        }
                    },
                    pipeline: RenderPipelineInfo {
                        skippable_fragment_stage,
                        vertex_buffers: render
                            .vertex_buffers
                            .into_iter()
                            .map(|vbuf| (vbuf.index, vbuf.into_inner()))
                            .collect(),
                        bind_groups,
                        push_constants: RenderPipelinePushConstantRanges {
                            vert: push_constant_ranges.vert.filter(|r| !r.is_empty()),
                            frag: push_constant_ranges.frag.filter(|r| !r.is_empty()),
                            push_constants_byte_size,
                        },
                        rasterizer: RasterizerState {
                            vertex_indexing: render
                                .vertex_id_order
                                .get_value()
                                .ok_or_else(|| into_err(call_info, PipelineError::MissingSpecialization.into()))?,
                            draw_info: render
                                .draw
                                .get_value()
                                .ok_or_else(|| into_err(call_info, PipelineError::UnusedRasterizer.into()))?,
                            samples: get_late_recorded!(render.sample_mask)?.0,
                            // we renamed WGSL `front_facing` to `is_ccw_primitive`,
                            // this means we must force `front_face` == `Ccw`.
                            //
                            // move info about this reasoning: https://gist.github.com/RayMarch/045f92dee5d911e144f8dd7fece219a2
                            front_face: Winding::Ccw,
                            color_target0_alpha_to_coverage: render
                                .color_target0_alpha_to_coverage
                                .get_value()
                                .unwrap_or(false),
                        },
                        color_targets: render
                            .color_targets
                            .into_iter()
                            .map(|tgt| (tgt.index, tgt.into_inner()))
                            .collect(),
                        depth_stencil: {
                            match render.depth_stencil.get() {
                                Some((ds, call)) => Some(ds.finalize(call_info)?),
                                None => None,
                            }
                        },
                    },
                })
            }
            PipelineKind::Compute => {
                let compute = self.wip_pipeline.special.compute.into_inner();

                PipelineDefinition::Compute(ComputePipeline {
                    label: None, //TODO(release) add api to add label
                    shader: ComputeShader {
                        code: match self.settings.lang {
                            Language::Wgsl => LanguageCode::Wgsl(shader_code),
                        },
                        entry_point: "comp_main",
                    },
                    pipeline: ComputePipelineInfo {
                        grid_info: ComputeGridInfo {
                            thread_grid_size_per_workgroup: get_late_recorded!(
                                compute.thread_grid_size_within_workgroup
                            )?
                            .0,
                            zero_init_workgroup_memory: self.settings.zero_init_workgroup_memory,
                            expected_threads_per_wave: compute.expected_threads_per_wave,
                        },
                        bind_groups,
                        push_constant_range: push_constant_ranges.comp,
                    },
                })
            }
        };
        // if errors were pushed since, for example internal errors that arise from the shader-code generator interacting with the ir module
        if let Some(errors) = EncodingErrors::from_vec(self.errors.take()) {
            return Err(errors);
        }
        Ok(pipeline_def)
    }
}

#[doc(hidden)] // internal
pub struct CallInfoScope {
    self_: CallInfo, // the latest user caller (non-reentrant library caller) from when `new` was called
    to_restore: Option<CallInfo>,
    context_was_missing: bool, // don't restore the state if the context was missing
}

impl Context {
    #[track_caller]
    pub(crate) fn call_info_scope() -> CallInfoScope { CallInfoScope::new(call_info!()) }
}

impl CallInfoScope {
    pub fn new(new_caller: CallInfo) -> Self {
        let result = CONTEXT
            .try_with(|ctx| {
                let maybe_init = ctx.try_borrow().ok()?;
                let ctx = maybe_init.as_ref()?;
                // if `latest_user_caller` is `Some(_)`, the `new_caller` is reentrant
                // from within the library and therefore not useful for error reporting.
                // Therefore the incoming `new_caller` is ignored in that case.
                let mut latest_user_caller = ctx.latest_user_caller_since_first.get().unwrap_or(new_caller);

                // remember whatever was in `latest_user_caller_since_first` before to
                // restore it afterwards
                let to_restore = ctx.latest_user_caller_since_first.replace(Some(latest_user_caller));

                Some(CallInfoScope {
                    self_: latest_user_caller,
                    to_restore,
                    context_was_missing: false,
                })
            })
            .ok()
            .flatten();

        match result {
            Some(t) => t,
            None => CallInfoScope {
                self_: new_caller,
                to_restore: None,
                context_was_missing: true,
            },
        }
    }
}

impl Drop for CallInfoScope {
    fn drop(&mut self) {
        if !self.context_was_missing {
            CONTEXT.try_with(|ctx| {
                let Ok(maybe_init) = ctx.try_borrow() else { return };
                let Some(ctx) = maybe_init.as_ref() else { return };

                let popped = ctx.latest_user_caller_since_first.replace(self.to_restore);
                if popped != Some(self.self_) {
                    ctx.push_error(
                        InternalError::new(
                            true,
                            format!(
                                "call-info scope stack-property was violated. pushed: {:?}, popped: {:?}",
                                Some(self.self_),
                                popped
                            ),
                        )
                        .into(),
                    );
                }
            });
        }
    }
}
