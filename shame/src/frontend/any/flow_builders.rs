use std::marker::PhantomData;

use crate::frontend::any::Any;
use crate::{
    call_info,
    common::{pool::Key, small_vec::SmallVec},
    frontend::error::InternalError,
    ir::recording::{
        Block, BlockKind, BlockSeriesRecorder, BodyKind,
        ConditionKind::{self, *},
        Context, Control, ExprStmt, FlowStmt, Stmt, ThenOrElse, TimeInstant,
    },
};

#[allow(missing_docs)]
pub struct Initialize;
#[allow(missing_docs)]
pub struct Condition;
#[allow(missing_docs)]
pub struct Increment;
#[allow(missing_docs)]
pub struct Body;

#[allow(missing_docs)]
pub struct ForRecorder<State> {
    start_time: TimeInstant,
    /// records the init block, which surrounds the other three (condition, increment, body)
    init_rec: BlockSeriesRecorder<1>,
    /// records the other three blocks, which are on the same level. starts on `ForRecorder<Init>::next`
    other_rec: Option<BlockSeriesRecorder<3>>,
    phantom: PhantomData<State>,
}

impl<S> ForRecorder<S> {
    #[rustfmt::skip]
    fn change_state<T>(self) -> ForRecorder<T> {
        let Self    { start_time, init_rec, other_rec, phantom } = self;
        ForRecorder { start_time, init_rec, other_rec, phantom: PhantomData}
    }
}

impl ForRecorder<Initialize> {
    #[allow(missing_docs)]
    #[allow(clippy::new_without_default)]
    #[track_caller]
    pub fn new() -> Self {
        Self {
            start_time: TimeInstant::next(),
            init_rec: BlockSeriesRecorder::new(call_info!(), BlockKind::ForInit),
            other_rec: None,
            phantom: PhantomData,
        }
    }

    #[allow(missing_docs)]
    #[track_caller]
    pub fn next(mut self) -> ForRecorder<Condition> {
        self.other_rec = Some(BlockSeriesRecorder::new(
            call_info!(),
            BlockKind::Condition(ConditionKind::For),
        ));
        self.change_state()
    }
}

impl ForRecorder<Condition> {
    #[allow(missing_docs)]
    #[track_caller]
    pub fn next(mut self, condition: Any) -> ForRecorder<Increment> {
        let call_info = call_info!();
        // the error for unavailable condition will be created after encoding.
        if let Some(node) = condition.node() {
            Context::try_with(call_info, |ctx| {
                ctx.push_expr_stmt_to_current_block(ExprStmt::Condition(node), TimeInstant::next(), call_info)
            });
        }
        let Some(other_rec) = &mut self.other_rec else {
            unreachable!("previous (1) `for` block recording state always initializes with `Some`")
        };
        other_rec.advance(call_info, BlockKind::ForIncrement);
        self.change_state()
    }
}

impl ForRecorder<Increment> {
    #[allow(missing_docs)]
    #[track_caller]
    pub fn next(mut self) -> ForRecorder<Body> {
        let Some(other_rec) = &mut self.other_rec else {
            unreachable!("previous (2) `for` block recording state always initializes with `Some`")
        };
        other_rec.advance(call_info!(), BlockKind::Body(BodyKind::For));
        self.change_state()
    }
}

impl ForRecorder<Body> {
    #[rustfmt::skip]
    #[allow(missing_docs)]
    #[track_caller]
    pub fn finish(mut self) {
        let call_info = call_info!();
        let Some(other_rec) = self.other_rec.take() else { unreachable!("previous (3) `for` block recording state always initializes with `Some`")};

        Context::try_with(call_info, |ctx| {
            let other = other_rec.finish(call_info);
            let initial_caller = self.init_rec.initial_caller();
            let init = self.init_rec.finish(call_info);
            match (init, other) {
                (Ok([init]), Ok([cond, inc, body])) => {
                    ctx.push_flow_stmt_to_current_block(
                        FlowStmt::Control(Control::For { init , cond, inc, body }),
                        self.start_time,
                        initial_caller
                    )
                },
                (e1, e0) => {
                    e0.map_err(|e| ctx.push_error(e.into()));
                    e1.map_err(|e| ctx.push_error(e.into()));
                }
            }
        });
    }
}

#[allow(missing_docs)]
pub struct Then;
#[allow(missing_docs)]
pub struct Else;

#[allow(missing_docs)]
pub struct IfRecorder<State> {
    start_time: TimeInstant,
    rec: BlockSeriesRecorder<3>,
    phantom: PhantomData<State>,
}

impl<S> IfRecorder<S> {
    #[rustfmt::skip]
    fn change_state<T>(self) -> IfRecorder<T> {
        let Self   { start_time, rec, phantom } = self;
        IfRecorder { start_time, rec, phantom: PhantomData}
    }
}

impl IfRecorder<Condition> {
    #[allow(missing_docs)]
    #[allow(clippy::new_without_default)]
    #[track_caller]
    pub fn new() -> Self {
        Self {
            start_time: TimeInstant::next(),
            rec: BlockSeriesRecorder::new(call_info!(), BlockKind::Condition(If)),
            phantom: PhantomData,
        }
    }

    #[allow(missing_docs)]
    #[track_caller]
    pub fn next(mut self, condition: Any) -> IfRecorder<Then> {
        let call_info = call_info!();
        // the error for unavailable condition will be created after encoding.
        if let Some(node) = condition.node() {
            Context::try_with(call_info, |ctx| {
                ctx.push_expr_stmt_to_current_block(ExprStmt::Condition(node), TimeInstant::next(), call_info)
            });
        }
        self.rec.advance(call_info, BlockKind::Body(BodyKind::Then));
        self.change_state()
    }
}

impl IfRecorder<Then> {
    #[allow(missing_docs)]
    #[track_caller]
    /// adds an `else` block
    pub fn next(mut self) -> IfRecorder<Else> {
        self.rec.advance(call_info!(), BlockKind::Body(BodyKind::Else));
        self.change_state()
    }

    #[rustfmt::skip]
    #[allow(missing_docs)]
    #[track_caller]
    pub fn finish(mut self) {
        let call_info = call_info!();
        let initial_caller = self.rec.initial_caller();
        Context::try_with(call_info, |ctx| match self.rec.finish(call_info) {
            Err(e) => ctx.push_error(e.into()),
            Ok([cond, then]) => ctx.push_flow_stmt_to_current_block(
                FlowStmt::Control(Control::IfThen { cond, then }),
                self.start_time,
                initial_caller
            ),
        });
    }
}

impl IfRecorder<Else> {
    #[rustfmt::skip]
    #[allow(missing_docs)]
    #[track_caller]
    pub fn finish(mut self) {
        let call_info = call_info!();
        let initial_caller = self.rec.initial_caller();
        Context::try_with(call_info, |ctx| match self.rec.finish(call_info) {
            Err(e) => ctx.push_error(e.into()),
            Ok([cond, then, els]) => ctx.push_flow_stmt_to_current_block(
                FlowStmt::Control(Control::IfThenElse { cond, then, els }),
                self.start_time,
                initial_caller
            ),
        });
    }
}

#[allow(missing_docs)]
pub struct WhileRecorder<State> {
    start_time: TimeInstant,
    rec: BlockSeriesRecorder<2>,
    phantom: PhantomData<State>,
}

impl<S> WhileRecorder<S> {
    #[rustfmt::skip]
    fn change_state<T>(self) -> WhileRecorder<T> {
        let Self   { start_time, rec, phantom } = self;
        WhileRecorder { start_time, rec, phantom: PhantomData}
    }
}

impl WhileRecorder<Condition> {
    #[allow(clippy::new_without_default)]
    #[track_caller]
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            start_time: TimeInstant::next(),
            rec: BlockSeriesRecorder::new(call_info!(), BlockKind::Condition(If)),
            phantom: PhantomData,
        }
    }

    #[allow(missing_docs)]
    #[track_caller]
    pub fn next(mut self, condition: Any) -> WhileRecorder<Body> {
        let call_info = call_info!();
        // the error for unavailable condition will be created after encoding.
        if let Some(node) = condition.node() {
            Context::try_with(call_info, |ctx| {
                ctx.push_expr_stmt_to_current_block(ExprStmt::Condition(node), TimeInstant::next(), call_info)
            });
        }
        self.rec.advance(call_info, BlockKind::Body(BodyKind::While));
        self.change_state()
    }
}

impl WhileRecorder<Body> {
    #[rustfmt::skip]
    #[allow(missing_docs)]
    #[track_caller]
    pub fn finish(mut self) {
        let call_info = call_info!();
        let initial_caller = self.rec.initial_caller();
        Context::try_with(call_info, |ctx| match self.rec.finish(call_info) {
            Err(e) => ctx.push_error(e.into()),
            Ok([cond, body]) => ctx.push_flow_stmt_to_current_block(
                FlowStmt::Control(Control::While { cond, body }),
                self.start_time,
                initial_caller
            ),
        });
    }
}
