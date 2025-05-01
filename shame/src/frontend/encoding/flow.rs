use crate::frontend::rust_types::len::x1;
use crate::frontend::rust_types::mem::Cell;
use crate::frontend::rust_types::reference::ReadWrite;
use crate::frontend::rust_types::vec::ToInteger;
use crate::frontend::rust_types::vec::ToVec;
use crate::frontend::rust_types::vec_range::Inclusivity;
use crate::i32x1;
use crate::ir;
use crate::ir::recording::FlowStmt;
use crate::ir::recording::StmtError;
use crate::u32x1;

use crate::boolx1;
use crate::frontend::rust_types::reference::Ref;
use crate::frontend::rust_types::scalar_type::ScalarTypeInteger;
use crate::frontend::rust_types::vec::vec;
use crate::frontend::rust_types::vec::zero;
use crate::frontend::rust_types::{AsAny, GpuType, To, ToGpuType};
use crate::VecRange;
use crate::{
    call_info,
    frontend::any::flow_builders::*,
    frontend::rust_types::reference::AccessMode,
    ir::recording::{Context, Jump, Stmt, TimeInstant},
    mem::AddressSpace,
};

use super::EncodingErrorKind;


#[cfg(not(feature = "relaxed_control_flow"))]
pub trait FlowFn: 'static {}
#[cfg(not(feature = "relaxed_control_flow"))]
impl<T: 'static> FlowFn for T {}

#[cfg(feature = "relaxed_control_flow")]
pub trait FlowFn {}
#[cfg(feature = "relaxed_control_flow")]
impl<T> FlowFn for T {}

#[track_caller]
fn record_jump_stmt(jump: Jump) {
    Context::try_with(call_info!(), |ctx| {
        match jump {
            Jump::Continue | Jump::Break | Jump::Return(None) => Ok(()),
            Jump::Discard | Jump::Return(Some(_)) => match ctx.currently_in_non_encoding_scope() {
                None => Err(StmtError::UnconditionalReturnOrDiscard(jump)),
                Some(_) => Ok(()),
            },
        }
        .map_err(|err| ctx.push_error(err.into()));

        ctx.push_flow_stmt_to_current_block(FlowStmt::Jump(jump), TimeInstant::next(), ctx.latest_user_caller());
        ctx.increment_execution_state()
    });
}

/// record an if statement, which gets turned into
/// an `if (cond) { body }` statement within the generated shader.
/// This is equivalent to [`vec::then`].
///
/// note: in the code generation step, the `then` function gets executed exactly once to
/// record everything inside it.
///
/// ## example
///
/// ```
/// use shame::{if_};
/// use shame as sm;
///
/// let i = sm::Cell::new(1u32);
///
/// let condition = vertex.index.to_u32().gt(16u32);
/// if_(condition, move || {
///     // do something
///     i.set_add(4u32);
/// });
/// ```
#[track_caller]
pub fn if_(cond: boolx1, then: impl FnOnce() + FlowFn) {
    let r = IfRecorder::new().next(cond.as_any());
    then();
    r.finish()
}

/// record an if-then-else statement, which gets turned into
/// an `if (cond) { ... } else { ... } ` statement within the generated shader.
///
/// note: in the code generation step, the `then` and `else_` functions get
/// executed exactly once to record everything inside them.
///
/// ## example
///
/// ```
/// use shame::{if_else};
/// use shame as sm;
///
/// let i = sm::Cell::new(1u32);
///
/// let condition = pipe.vertex.index.to_u32().gt(16u32);
/// sm::if_else(condition,
///     move || { // then block
///         i.set_add(1u32);
///     },
///     move || { // else block
///         i.set_sub(1u32);
///     },
/// );
/// ```
#[track_caller]
pub fn if_else(cond: boolx1, then: impl FnOnce() + FlowFn, else_: impl FnOnce() + FlowFn) {
    let r = IfRecorder::new().next(cond.as_any());
    then();
    let r = r.next();
    else_();
    r.finish()
}

/// record a while loop
///
/// break the loop with [`break_`]/[`break_if`] or use [`continue_`]/[`continue_if`] to skip iterations
///
/// the recording gets turned into
/// a while loop within the generated shader.
///
/// note: in the code generation step, the `body` gets executed exactly once to
/// record everything inside it.
///
/// ## example
///
/// ```
/// use shame::{while_};
/// use shame as sm;
///
/// let i = sm::Cell::new(0u32);
///
/// let cond = i.get().greater_than(10u32);
/// while_(move || {
///     i.set_add(1u32);
/// });
/// ```
#[track_caller]
pub fn while_(condition: impl FnOnce() -> boolx1 + FlowFn, body: impl FnOnce() + FlowFn) {
    let r = WhileRecorder::new();
    let value = condition();
    let r = r.next(value.as_any());
    body();
    r.finish()
}

/// record a range-based for loop, which gets turned into
/// a for loop within the generated shader.
///
/// break the loop with [`break_`]/[`break_if`] or use [`continue_`]/[`continue_if`] to skip iterations
///
/// note: in the code generation step, the `body` gets executed exactly once to
/// record everything inside it.
///
/// ## example
///
/// ```
/// use shame as sm;
///
/// let j = sm::Cell::new(1u32);
///
/// for_range(5..10, move |i| {
///     j.set_add(i.to_i32());
/// })
/// ```
#[track_caller]
pub fn for_range(range: impl VecRange<i32, x1>, body: impl FnOnce(i32x1) + FlowFn) { for_range_impl(range, body); }

#[track_caller]
pub(crate) fn for_range_impl<T: ScalarTypeInteger>(
    range: impl VecRange<T, x1>,
    body: impl FnOnce(vec<T, x1>) + FlowFn,
) {
    let [(start, start_bound), (end, end_bound)] = range.get_bounds().scalar();

    let start = match start_bound {
        Inclusivity::Incl => start,
        Inclusivity::Excl => start + vec::one(),
    };

    let r = ForRecorder::new();
    let mut i: Ref<vec<T, x1>> = Cell::new(start);
    let r = r.next();
    let cond = match end_bound {
        Inclusivity::Incl => i.get().less_eq(end),
        Inclusivity::Excl => i.get().less_than(end),
    };
    let r = r.next(cond.as_any());
    i.set_add(vec::<_, x1>::one());
    let r = r.next();
    body(i.get());
    r.finish()
}

/// record a range-based for loop which iterates over the
/// `0..upper_bound_exclusive` range
///
/// break the loop with [`break_`]/[`break_if`] or use [`continue_`]/[`continue_if`] to skip iterations
///
/// the recording gets turned into
/// a for loop within the generated shader.
///
/// note: in the code generation step, the `body` gets executed exactly once to
/// record everything inside it.
///
/// ## example
///
/// ```
/// use shame as sm;
///
/// let j = sm::Cell::new(0u32);
///
/// for_count(10, move |i| {
///     j.set_add(i.to_i32());
/// })
/// ```
#[track_caller]
pub fn for_count<Int>(upper_bound_exclusive: Int, body: impl FnOnce(Int::Gpu) + FlowFn)
where
    Int: ToInteger,
    <Int as ToVec>::T: ScalarTypeInteger,
{
    for_range_impl(vec::zero()..upper_bound_exclusive.to_gpu(), body)
}

/// record a loop
///
/// break the loop with [`break_`]/[`break_if`] or use [`continue_`]/[`continue_if`] to skip iterations
///
/// the recording gets turned into
/// a `while(true)` loop within the generated shader.
///
/// note: in the code generation step, the `body` gets executed exactly once to
/// record everything inside it.
///
/// ## example
///
/// ```
/// use shame::{loop_, if_, break_};
/// use shame as sm;
///
/// let i = sm::Cell::new(0u32);
///
/// loop_(move || {
///     break_if(i.get().greater_than(10u32));
///     i.set_add(1u32);
/// });
/// ```
#[track_caller]
pub fn loop_(body: impl FnOnce() + FlowFn) {
    let r = WhileRecorder::new().next(true.to_any());
    body();
    r.finish()
}

/// record a `break` statement in a loop
///
/// there exists a [`break_if`] shorthand for conditional `break`ing
///
/// if this function is called outside of a loop recording, the encoding fails
/// with an [`EncodingError`]
///
/// [`EncodingError`]: crate::EncodingError
#[track_caller]
pub fn break_() { record_jump_stmt(Jump::Break) }

/// record a `continue` statement in a loop
///
/// there exists a [`continue_if`] shorthand for conditional `continue`ing
///
/// if this function is called outside of a loop recording, the encoding fails
/// with an [`EncodingError`]
///
/// [`EncodingError`]: crate::EncodingError
#[track_caller]
pub fn continue_() { record_jump_stmt(Jump::Continue) }

/// record a `if (cond) { break; }` statement in a loop
///
/// if this function is called outside of a loop recording, the encoding fails
/// with an [`EncodingError`]
///
/// [`EncodingError`]: crate::EncodingError
#[track_caller]
pub fn break_if(cond: boolx1) {
    let r = IfRecorder::new().next(cond.as_any());
    break_();
    r.finish()
}

/// record a `if (cond) { continue; }` statement in a loop
///
/// if this function is called outside of a loop recording, the encoding fails
/// with an [`EncodingError`]
///
/// [`EncodingError`]: crate::EncodingError
#[track_caller]
pub fn continue_if(cond: boolx1) {
    let r = IfRecorder::new().next(cond.as_any());
    continue_();
    r.finish()
}

/// record a [`discard`] statement
///
/// there exists a [`discard_if`] shorthand for conditional `continue`ing
///
/// [`discard`] turns all currently active fragment threads permanently inactive
/// (that is, until the pipeline finishes executing).
/// Use [`discard`] inside an [`if_`] body (or other control flow) to selectively
/// discard fragments.
///
/// usage of [`discard`] makes all currently recording control flow blocks run
/// per-fragment, that means per-vertex values cannot be used as condition or
/// inside the block.
///
/// ## example
///
/// ```
/// let even_row = (fragment.pixel_pos.x.to_u32() % 2u32).equals(0u32);
/// if_(even_row, move || {
///     discard();
/// });
/// ```
#[track_caller]
pub fn discard() {
    // TODO(release)
    // recording an unconditional discard will cause a shader compile error
    // with the error "instructions after return" if there are any statements generated
    // after the unconditional discard.
    // possible solutions:
    // - disallow make unconditional discards (this also solves the stage problem)

    record_jump_stmt(Jump::Discard)
}

/// record a `if (cond) { discard; }` statement
///
/// [`discard`] turns all currently active fragment threads permanently inactive
/// (that is, until the pipeline finishes executing).
///
/// this conditional [`discard_if`] turns only those active threads inactive that
/// satisfy the condition `cond`.
///
/// usage of [`discard_if`] makes all currently recording control flow blocks run
/// per-fragment, that means per-vertex values cannot be used as condition or
/// inside the block.
///
/// ## example
/// ```
/// let even_row = (fragment.pixel_pos.x.to_u32() % 2u32).equals(0u32);
/// discard_if(even_row);
/// ```
#[track_caller]
pub fn discard_if(cond: boolx1) {
    let r = IfRecorder::new().next(cond.as_any());
    discard();
    r.finish()
}

#[track_caller]
pub fn return_value(value: impl ToGpuType) {
    if let Some(node) = value.to_any().node() {
        record_jump_stmt(Jump::Return(Some(node)))
    }
}

#[track_caller]
pub fn return_() {
    Context::try_with(call_info!(), |ctx| {
        ctx.push_flow_stmt_to_current_block(
            FlowStmt::Jump(Jump::Return(None)),
            TimeInstant::next(),
            ctx.latest_user_caller(),
        )
    });
}
