//! control flow functions for recording `if` `while` `for` etc.
use super::*;

impl Ten<scal, bool> {

    /// record an `if (self) {recording_fn}` branch in the shader.
    /// 
    /// note that the `recording_fn` will be executed exactly once
    /// to record which statements/expressions are required to be
    /// put into the 'if' branches' body. This may seem unintuitive wrt.
    /// any regular rust code that runs in `recording_fn` as it is
    /// not conditionally executed at all. It runs either way since
    /// self's value cannot be known when recording the shader.
    pub fn then(&self, recording_fn: impl FnOnce() + 'static) { //TODO: +'static is experimental to prevent variable "smuggling" out of scope
        self.into_any().record_then(recording_fn)
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

    // pub fn while_loop(&self, body_fn: impl FnOnce() + 'static) { //TODO: +'static is experimental to prevent variable "smuggling" out of scope
    //     self.into_any().record_while(body_fn)
    // }

    // pub fn for_loop(&self, increment_fn: impl FnOnce() + 'static, body_fn: impl FnOnce() + 'static) { //TODO: +'static is experimental to prevent variable "smuggling" out of scope
    //     self.into_any().record_for(increment_fn, body_fn)
    // }

}

