//! wrapper for write-only types

use super::*;
use crate::assert;
use shame_graph::{Access, Any};

/// wrapper type for write-only tensors
#[derive(Clone, Copy)]
pub struct WriteOnly<S: Shape, D: DType>(Ten<S, D>);

impl<S: Shape, D: DType> WriteOnly<S, D> {
    /// downcasts `any` and `stage` to a write only wrapped [`Ten<S, D>`]
    pub fn new(any: Any, stage: Stage) -> Self {
        let any = any
            .ty_via_thread_ctx()
            .map(|ty| {
                assert::assert_string(
                    ty.access == Access::WriteOnly,
                    format!("cannot downcast {ty} to WriteOnly type"),
                )
                .unwrap_or(any)
            })
            .unwrap_or(Any::not_available());

        WriteOnly(Ten::from_downcast(any, stage))
    }

    /// type erased wrapped tensor
    pub fn any(&mut self) -> Any { self.0.any }

    /// stage of the wrapped value
    pub fn stage(&self) -> Stage { self.0.stage }

    /// assign an identifier to the wrapped value
    pub fn aka(&self, name: &str) { self.0.any.aka(name); }

    /// assign `val` to `self`
    ///
    /// alternative naming to `write`
    pub fn set(&mut self, val: impl AsTen<S = S, D = D>) {
        let val = val.as_ten();
        self.0.set(val);
    }

    /// assign `val` to `self`
    ///
    /// alternative naming to `set`
    pub fn write(&mut self, val: impl AsTen<S = S, D = D>) { self.set(val) }
}
