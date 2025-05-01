use std::{cell::RefCell, marker::PhantomData};
use std::{
    cell::{Cell, Ref, RefMut},
    num::NonZeroU32,
};

use super::context::Context;
use crate::common::marker::{Unsend, Unsync};

thread_local! {
    pub(super) static CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) };
    static GENERATION: Cell<NonZeroU32> = const { Cell::new(NonZeroU32::new(999 /*arbitrary number*/).unwrap()) };
}

/// While alive, keeps the context in thread local memory.
/// removes it on drop.
pub struct ThreadContextGuard(PhantomData<Unsend>);

impl ThreadContextGuard {
    /// put `ctx` into the `CONTEXT` thread local variable.
    /// If that slot is already occupied, returns `Err(ctx)` back to the user.
    pub fn new(ctx: Context) -> Result<ThreadContextGuard, Context> {
        CONTEXT.with(|r| match &mut *r.borrow_mut() {
            Some(_) => Err(ctx),
            r @ None => {
                *r = Some(ctx);
                Ok(ThreadContextGuard(PhantomData))
            }
        })
    }

    pub fn into_inner(self) -> Context {
        let taken = CONTEXT.with(|ctx| ctx.borrow_mut().take());
        assert!(taken.is_some());
        taken.expect("ensured by `Self: !Send` and the `put` method")
    }
}

impl Drop for ThreadContextGuard {
    fn drop(&mut self) { CONTEXT.with(|ctx| ctx.borrow_mut().take()); }
}

pub(crate) fn next_thread_generation() -> NonZeroU32 { GENERATION.with(|g| g.increment_by(1)) }

pub trait CellNonZeroU32Ext {
    type Output;
    fn increment_by(&self, amount: u32) -> Self::Output;
}

impl CellNonZeroU32Ext for Cell<NonZeroU32> {
    type Output = NonZeroU32;
    fn increment_by(&self, amount: u32) -> NonZeroU32 {
        let next: u32 = self.get().into();
        let next = NonZeroU32::new(next + amount).unwrap();
        self.set(next);
        next
    }
}
