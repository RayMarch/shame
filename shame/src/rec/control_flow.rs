//! control flow functions for recording `if` `while` `for` etc.
use std::ops::Range;

use shame_graph::{Context};

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

pub fn for_range<D, I>(range: Range<I>, body_fn: impl FnOnce(I::Rec)) 
where 
    I: AsTen<S=scal, D=D>,
    D: IsDtypeNonFloatingPoint // reject situations where it matters whether a float is equal to range.end
{
    for_range_step(range, || one(), body_fn)
}

pub fn for_range_step<D: DType, I: AsTen<S=scal, D=D>, Step: AsTen<S=scal, D=D>>(
    range: Range<I>, 
    step_fn: impl FnOnce() -> Step,
    body_fn: impl FnOnce(I::Rec)
) {
    let stage = narrow_stages_or_push_error([
        range.start.stage(), 
        range.end.stage()
    ]);

    let shader_kind = Context::with(|ctx| ctx.shader_kind());
    let available = Stage::from_shader_kind(shader_kind) != Stage::NotAvailable;
    
    use shame_graph::{Any, DType::*};
    let dtype = I::D::DTYPE;
    let i = std::cell::Cell::new(None);

    if available {
        Any::record_for_loop(
            || { // init
                let start = range.start.into_any();
                i.set(Some(start.aka("i")))
            }, 
            || { // condition evaluation
                let end = range.end.into_any();
                let i = i.get().unwrap();
                match dtype {
                    Bool => (!i).logical_and(end), //DNF of "<" for booleans
                    I32 | U32 | F32 | F64 => i.lt(end),
                }
            }, 
            || { // increment
                let mut i = i.get().unwrap();
                let step = step_fn();
                i.increment_by(step.into_any())
            }, 
            || { // body
                let i = i.get().unwrap();
                let i = i.downcast(stage);
                body_fn(i)
            }
        );
    }
}