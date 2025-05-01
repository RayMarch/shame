use std::{cell::Cell, collections::HashMap, fmt::Display, num::NonZeroU32, ops::Deref};

use crate::{
    call_info,
    common::pool::{Key, PoolRefMut},
    frontend::{encoding::EncodingError, error::InternalError},
    ir::expr::Expr,
    try_ctx_track_caller,
};

use super::{Context, NodeRecordingError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ident {
    /// the dedup step will choose a unique identifier
    Unchosen,
    Chosen(Priority, String),
}

macro_rules! impl_prios {
    (pub enum $name: ident {$($(#[$($attrss:tt)*])* $var: ident,)*}) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $name {$($(#[$($attrss)*])* $var),*}
        impl $name {
            pub fn all() -> &'static [Priority] {
                &[$($name::$var),*]
            }
        }
    };
}

/// priorities, lowest to highest
impl_prios! {
    pub enum Priority {
        /// prefix from settings will be added to this ident
        Auto,
        /// prefix from settings will be added to this ident
        UserHint,
        /// no prefix will be added to this ident
        UserForced,
        /// used internally for ensuring certain identifiers, for example `main`
        /// for the entry point function in glsl
        Forced,
    }
}

impl Ident {
    #[track_caller]
    pub fn try_push_to_thread_ctx(self) -> Option<Key<Ident>> {
        Context::try_with(call_info!(), |ctx| ctx.pool_mut().push(self))
    }

    /// called by the library to auto suggest a string for the identifier
    pub(crate) fn auto(ident: String) -> Option<Key<Ident>> {
        Ident::Chosen(Priority::Auto, ident).try_push_to_thread_ctx()
    }

    /// called by the library to auto suggest a string for the identifier
    pub(crate) fn auto_in_pool(ident: String, pool: &mut PoolRefMut<Ident>) -> Key<Ident> {
        pool.push(Ident::Chosen(Priority::Auto, ident))
    }

    pub fn try_change(&mut self, new: Ident) -> Result<(), Priority> {
        match (&self, &new) {
            (Ident::Unchosen, _) => Ok(()),
            (Ident::Chosen(prio, _), Ident::Unchosen) => Err(*prio),
            (Ident::Chosen(prio, _), Ident::Chosen(new_prio, _)) => match prio <= new_prio {
                true => Ok(()),
                false => Err(*prio),
            },
        }
        .map(|()| *self = new)
    }
}

impl Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Ident::Unchosen => "<anonymous>",
            Ident::Chosen(_, ident) => ident,
        })
    }
}

pub fn arbitrary_str_to_ident_string(arb: &str, prefix: &str) -> String {
    let mut chars = arb.chars();
    let mut s = format!("{prefix}{arb}");
    let mut s = s.replace(|c: char| !(c.is_ascii_alphanumeric() || c == '_'), "");
    if s.is_empty() {
        s.push('e');
    }
    if s.starts_with(|c: char| c.is_numeric()) {
        s = format!("_{s}");
    }
    while s.ends_with(|c: char| c.is_numeric()) {
        // for now characters are not allowed to end with numeric chars
        // due to the way the deduplication algorithm works.
        // TODO(release) rewrite the deduplication algorithm to deal with existing number suffixes
        s.pop();
    }
    while s.starts_with("__") {
        s = s.replacen("__", "_", 1);
    }
    if s == "_" {
        // if we started with arb consisting of only underscores
        s += "e";
    }
    while s.starts_with("gl_") | s.starts_with("sm_") | s.starts_with("Sm_") {
        s = s.replacen("gl_", "gl", 1);
        s = s.replacen("sm_", "sm", 1);
        s = s.replacen("Sm_", "Sm", 1);
    }
    //TODO(release) make this create a github issue "internal error" message
    debug_assert!(
        is_valid_cross_language_ident_str(&s),
        "internal error (please report): the identifier string `{arb}` was transformed to `{s}`, which is not considered valid across target languages"
    );
    s
}

// TODO(low prio) this can instead be associated functions of [`Language`]
pub fn is_valid_cross_language_ident_str(ident: &str) -> bool {
    match () {
        () if ident.starts_with("__") => false,  // WGSL
        () if ident == "_" => false,             // WGSL
        () if ident.starts_with("sm_") => false, // shame (hardcoded idents start with sm_)
        () if ident.starts_with("Sm_") => false, // shame (hardcoded idents start with Sm_)
        () if ident.starts_with("gl_") => false, // GLSL
        () if ident.starts_with(|c: char| c.is_numeric()) => false,
        () if ident.chars().any(|c| !(c.is_ascii_alphanumeric() || c == '_')) => false,
        () => true,
    }
}

/// final deduplicated identifiers, ready for writing into shader code.
///
/// use `Key<Ident>` to index via `std::ops::Index`
pub struct FinalIdents {
    idents: Box<[String]>,
    generation: NonZeroU32,
}

impl std::ops::Index<Key<Ident>> for FinalIdents {
    type Output = str;

    fn index(&self, key: Key<Ident>) -> &Self::Output {
        self.idents
            .get(key.index())
            .and_then(|s| (self.generation == key.generation()).then_some(s.as_str()))
            .expect("same generation means same indexing works")
    }
}

/// make all identifiers unique, except if duplicate idents are both `ForcedUnchecked` idents
pub fn dedup_and_finalize_idents(mut pool: PoolRefMut<Ident>, prefix: &str) -> FinalIdents {
    struct Entry {
        prio: Priority,
        string: String,
        suffix: Cell<Option<u32>>,
    }

    // make all idents valid
    let mut valid_idents = pool
        .iter()
        .map(|ident| {
            let (prio, string) = match ident {
                Ident::Unchosen => (Priority::Auto, ""),
                Ident::Chosen(prio, string) => (*prio, string.as_str()),
            };
            Entry {
                prio,
                string: arbitrary_str_to_ident_string(string, prefix),
                suffix: Default::default(),
            }
        })
        .collect::<Vec<_>>();

    // add number at the end of identifier names to remove collisions
    // for example:
    // `my_normal` `my_normal` `my_normal`
    // becomes
    // `my_normal` `my_normal1` `my_normal2`
    //  ^owner
    //
    // every identifier string has one owner which gets to keep the identifier
    // without a number suffix.
    // ownership is decided based on priority.

    struct Owner {
        prio: Priority,
        index: usize,
        /// counter that increments when a new suffix is given to that identifier string
        suffix_max: u32,
    }
    let mut ident_owner = HashMap::<&String, Owner>::new();

    for (i, ith) in valid_idents.iter().enumerate() {
        ident_owner
            .entry(&ith.string)
            .and_modify(|owner| {
                // find out which of the two is not the owner
                // thats the one that gets a suffix
                let non_owner = match ith.prio >= owner.prio {
                    false => i,
                    true => {
                        // `ith` becomes new owner
                        // previous owner gets a number
                        let prev_owner = owner.index;
                        owner.index = i;
                        prev_owner
                    }
                };
                valid_idents[non_owner].suffix.set(Some(owner.suffix_max));
                owner.suffix_max += 1;
            })
            .or_insert(Owner {
                prio: ith.prio,
                index: i,
                suffix_max: 0,
            });
    }

    // append number suffixes to strings to form final identifiers
    let array: Box<[_]> = valid_idents
        .into_iter()
        .map(|ith| match ith.suffix.get() {
            Some(suffix) => format!("{}{}", ith.string, suffix),
            None => ith.string,
        })
        .collect();
    FinalIdents {
        idents: array,
        generation: pool.generation(),
    }
}
