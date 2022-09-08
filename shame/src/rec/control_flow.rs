//! control flow functions for recording `if` `while` `for` etc.
use std::ops::Bound;
use shame_graph::Context;

use super::*;

impl Ten<scal, bool> {

    /// record an `if (self) {then_fn}` branch in the shader.
    /// 
    /// note that the `then_fn` will be executed exactly once
    /// to record which statements/expressions are required to be
    /// put into the 'if' branches' body. This may seem unintuitive wrt.
    /// any regular rust code that runs in `then_fn` as it is
    /// not conditionally executed at all. It runs either way since
    /// self's value cannot be known when recording the shader.
    pub fn then(&self, then_fn: impl FnOnce() + 'static) { //TODO: +'static is experimental to prevent variable "smuggling" out of scope
        self.into_any().record_then(then_fn)
    }

    /// record an `if (self) {then_fn} else {else_fn}` branch in the shader.
    /// 
    /// note that `then_fn` and `else_fn` will be executed exactly once
    /// in this order. They are not run conditionally at all because
    /// they are responsible for recording the statements/expressions
    /// that need to be put into the then/else branch respectively.
    /// This may seem unintuitive wrt.
    /// any regular rust code that runs in `then_fn`/`else_fn` as it is
    /// not conditionally executed at all. It runs either way since
    /// self's value cannot be known when recording the shader
    pub fn then_else(&self, then_fn: impl FnOnce(), else_fn: impl FnOnce() + 'static) { //TODO: +'static is experimental to prevent variable "smuggling" out of scope
        self.into_any().record_then_else(then_fn, else_fn)
    }

    // pub fn for_loop(&self, increment_fn: impl FnOnce() + 'static, body_fn: impl FnOnce() + 'static) { //TODO: +'static is experimental to prevent variable "smuggling" out of scope
    //     self.into_any().record_for(increment_fn, body_fn)
    // }

    // pub fn while_(evaluate_condition_fn: impl FnOnce() -> boolean, body_fn: impl FnOnce()) {
    //     Any::record_for_loop(||(), || evaluate_condition_fn().as_any(), ||(), body_fn)
    // }

    // shame::while_(|| true, || {});
    // shame::for_range(0..10, |i| {});
    // shame::for_range_step(0..10, 2, |i| {});

    

}

fn get_bound<R, T>(bound: &Bound<T>, f: impl FnOnce(&T) -> R) -> Option<R> {
    use Bound::*;
    match bound {
        Included(x) | Excluded(x) => Some(f(x)),
        Unbounded => None,
    }
}

fn map_bound<R, T>(bound: Bound<T>, f: impl FnOnce(T) -> R) -> Bound<R> {
    use Bound::*;
    match bound {
        Included(x) => Included(f(x)),
        Excluded(x) => Excluded(f(x)),
        Unbounded => Unbounded,
    }
}

pub fn for_range<D, Scalar>(
    range: impl std::ops::RangeBounds<Scalar>, 
    body_fn: impl FnOnce(Scalar::Rec)
) 
where 
    Scalar: AsTen<S=scal, D=D>,
    D: IsDtypeNonFloatingPoint // reject situations where it matters whether a float is equal to range.end
{
    for_range_step(range, || one(), body_fn)
}

pub fn for_range_step<D, Scalar, Step>(
    range: impl std::ops::RangeBounds<Scalar>, 
    step_fn: impl FnOnce() -> Step,
    body_fn: impl FnOnce(Scalar::Rec)
) 
where
    D: DType,
    Scalar: AsTen<S=scal, D=D>,
    Step: AsTen<S=scal, D=D> 
{   
    use Bound::*;
    let [start, end] = [range.start_bound(), range.end_bound()];
    let [start_stage, end_stage] = [start, end].map(
        |b| get_bound(&b, |t| t.stage()).unwrap_or(Stage::Uniform)
    );
    let _ = narrow_stages_or_push_error(
        [start_stage, end_stage]
    );

    let shader_kind = Context::with(|ctx| ctx.shader_kind());
    let available = Stage::from_shader_kind(shader_kind) != Stage::NotAvailable;
    
    use shame_graph::{Any, Error};
    let i = std::cell::Cell::new(None);

    if available {
        Any::record_for_loop(
            || { // init
                let start = map_bound(start, |x| x.into_any().aka("i"));
                match Any::lower_bound_value(start) {
                    Ok(start) => i.set(Some(start)),
                    Err(e) => Context::with(|ctx| ctx.push_error(match e {
                        Unbounded => Error::ArgumentError("for range: cannot use unbounded lower bound".to_string()),
                        Included(d) => Error::TypeError(format!("for range: cannot use {d} lower bound in inclusive range")),
                        Excluded(d) => Error::TypeError(format!("for range: cannot use {d} lower bound in inclusive range")),
                    })),
                }
            }, 
            || { // condition evaluation
                let mut i = i.get().unwrap();
                let end = map_bound(end, |x| x.into_any());
                i.is_below_upper_bound(end)
            }, 
            || { // increment
                let mut i = i.get().unwrap();
                let step = step_fn();
                i.increment_by(step.into_any())
            }, 
            || { // body
                let i = i.get().unwrap();
                let i = i.downcast(start_stage);
                body_fn(i)
            }
        );
    }
}

pub fn break_if(cond: boolean) {cond.then(|| break_())}
pub fn continue_if(cond: boolean) {cond.then(|| continue_())}
pub fn discard_if(cond: boolean) {cond.then(|| discard_fragment())}

pub fn break_() {
    shame_graph::Stmt::record_break()
}

pub fn continue_() {
    shame_graph::Stmt::record_continue();
}

pub fn discard_fragment() {
    shame_graph::Stmt::record_discard();
}