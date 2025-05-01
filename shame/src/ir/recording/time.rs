use std::{fmt::Display, num::NonZeroU32, rc::Rc};

use crate::{
    call_info,
    common::{pool::Key, small_vec_actual::SmallVec},
};

use super::{Block, BlockKind, Context, FunctionDef};

/// A counter that represents the order of recording within an `Encoding`'s
/// lifetime.
/// Objects with a smaller value have been recorded before objects with a
/// larger value.
/// This does NOT model execution order. An expression with
/// `RecordTime(4)` may be executed after an expression with `RecordTime(5)`
/// (either because of expression reordering, or because of it simply
/// being in a loop that jumps back to the top of execution, or other reasons).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecordTime(u32);

impl Display for RecordTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "\u{23F1}({})", self.0) }
}

/// TODO(release): ask a person that knows more about compilers what this is normally called
///
/// This counter models different states of the shader's execution.
/// It is a cheap way of tracking whether exprs can be reordered, with
/// many false-negatives (= exprs that can be reordered being classified
/// as un-reorderable).
/// Every time
///     - a memory write happens
///     - a conditional/loop block gets entered or exited
///     - a barrier function is called
///     - etc.
///
/// this counter is incremented, so that the shader generator can easily decide
/// whether the results of expressions need to be stored in constants/variables
/// before that expression is being written into the shader.
///
/// # Execution States
/// Two regions of code are in two different execution states if:
/// for any possible expression/statement that is valid in both regions, the
/// expression/statement evaluates to different values/has different effects.
/// Or put differently:
/// If moving an identically-written expression/statement from one code region to
/// another changes the meaning of that expression/statement, the regions have different
/// execution state.
///
/// > note: change of meaning that results from identifier shadowing is not
/// > relevant here. it is assumed that all identifiers in a recording are unique.
///
/// example 1:
/// ```
/// // region A
/// if x < 3 {
///     //region B
/// }
/// // region C
/// ```
/// region A and B have different execution states since placing a `y += 1` statement
/// in A has a different effect than in B for threads where `x < 3`.
/// The same is true for region B vs C.
/// region A and C *likely* have different execution states since the statement
/// `x = 0` will trigger all effects in region B if placed in A, but not if
/// placed in C. Since execution states are assigned pessimistically to avoid
/// generating wrong code, A and C are modelled as having two
/// different execution states.
///
/// example 2:
/// ```
/// // region A
/// workgroupBarrier();
/// // region B
/// foo = 4;
/// // region C
/// ```
/// - placing an expression that reads from the AddressSpace::Workgroup
///   will evaluate to different values if placed in region A vs B.
/// - placing an expression reading `foo` such as `sin(foo)` will evaluate to
///   different values if placed in region B vs C.
///   therefore the execution states of A, B and C are all different.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExecutionState(u32);

impl Display for ExecutionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Exs{}", self.0) }
}

impl RecordTime {
    pub fn new(time_counter: NonZeroU32) -> Self { Self(time_counter.into()) }

    pub fn invalid() -> Self {
        Self(0) // record time is generated from a NonZeroU32,
        // so 0 is free as an invalid state
    }
}

impl ExecutionState {
    pub fn new(time_counter: NonZeroU32) -> Self { Self(time_counter.into()) }

    pub fn invalid() -> Self {
        Self(0) // exec state is generated from a NonZeroU32,
        // so 0 is free as an invalid state
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeInstant {
    record_time: RecordTime,
    exec_state: ExecutionState,
}

impl TimeInstant {
    pub fn record_time(&self) -> RecordTime { self.record_time }

    pub fn exec_state(&self) -> ExecutionState { self.exec_state }

    /// takes a snapshot of the current recording block/function hierarchy
    /// but also increments the instant counter from the last call to `next`.
    pub fn next() -> TimeInstant {
        Context::try_with(call_info!(), |ctx| TimeInstant {
            record_time: ctx.next_record_time(),
            exec_state: ctx.current_execution_state(),
        })
        .unwrap_or_else(Self::invalid)
    }

    // FIXME: we can get rid of invalid states
    //  if we require a `&Context` arg in `Self::next`
    fn invalid() -> TimeInstant {
        TimeInstant {
            record_time: RecordTime::invalid(),
            exec_state: ExecutionState::invalid(),
        }
    }
}
