/// put into phantom data this acts like an `impl !Sync` for the containing type
pub type Unsync = std::cell::Cell<()>;
/// put into phantom data this acts like an `impl !Send` for the containing type
pub type Unsend = std::sync::MutexGuard<'static, ()>;
