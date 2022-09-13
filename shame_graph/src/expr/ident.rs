use std::ops::Deref;

use crate::{
    context::Context,
    pool::{Key, PoolRefMut},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentSlot(pub(crate) Key<Option<String>>);

impl IdentSlot {
    // don't make this function public to prevent users from
    // allocating lots of ident slots. Only offer IdentSlot reusing functions (e.g. in Any)
    pub(crate) fn new_in(maybe_named: Option<String>, pool_mut: &mut PoolRefMut<Option<String>>) -> Self {
        Self(pool_mut.push(maybe_named))
    }

    pub fn rename(&self, new_maybe_name: Option<String>) {
        Context::with(|ctx| {
            let ident = &mut ctx.idents_mut()[self.0];
            *ident = new_maybe_name;
        });
    }

    pub fn eq_str(&self, str: &str) -> bool {
        Context::with(|ctx| {
            let ident = &ctx.idents()[self.0];
            match ident {
                Some(s) => s == str,
                None => false,
            }
        })
    }
}

impl Deref for IdentSlot {
    type Target = Key<Option<String>>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Named<T>(pub T, pub IdentSlot);
