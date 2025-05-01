use crate::call_info;
use crate::common::pool::{Key, Pool};
use crate::ir::pipeline::PossibleStages;
use crate::ir::recording::Node;
use std::fmt::Display;
use std::num::NonZeroU32;

use super::{Block, Context, FunctionDef, Ident};

pub(crate) trait GetPool: Sized {
    fn get_pool(pools: &Pools) -> &Pool<Self>;
}

macro_rules! impl_pools {
    ($($index: tt: $pool_ty: ty),* $(,)?) => {
        pub(crate) struct Pools {
            generation: NonZeroU32,
            pools_per_type: ($(Pool<$pool_ty>,)*)
        }

        impl Pools {
            pub fn new(generation: NonZeroU32) -> Self {
                Self {
                    generation,
                    pools_per_type: ($(Pool::<$pool_ty>::new(generation),)*)
                }
            }
        }

        $(
        impl GetPool for $pool_ty {
            fn get_pool(pools: &Pools) -> &Pool<Self> {
                &pools.pools_per_type.$index
            }
        }
        )*
    };
}

impl_pools! {
    0: Node,
    1: Ident,
    2: Block,
    3: FunctionDef,
}

impl Pools {
    pub fn generation(&self) -> NonZeroU32 { self.generation }
}

impl<T: GetPool + Display> Display for Key<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_context(f, |f, ctx| ctx.pool()[*self].fmt(f))
    }
}

impl<T> Key<T> {
    /// only calls `func` if recording with a context of the same generation,
    /// formats differently otherwise.
    fn display_with_context<'a>(
        &self,
        f: &mut std::fmt::Formatter<'a>,
        func: impl FnOnce(&mut std::fmt::Formatter<'a>, &Context) -> std::fmt::Result,
    ) -> std::fmt::Result {
        Context::try_with(call_info!(), |ctx| match ctx.generation() == self.generation() {
            true => func(f, ctx),
            false => write!(f, "[foreign {}]", std::any::type_name::<T>()),
        })
        .unwrap_or_else(|| write!(f, "[expired {}]", std::any::type_name::<T>()))
    }
}
