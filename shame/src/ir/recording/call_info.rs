use std::fmt::Display;

use super::Context;

/// information about a function caller, such as rust-code location
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallInfo {
    /// a rust code location
    pub location: &'static std::panic::Location<'static>,
}

impl std::fmt::Debug for CallInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallInfo")
            .field("location", &self.location.to_string())
            .finish()
    }
}

impl Display for CallInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.location) }
}

#[doc(hidden)]
#[macro_export]
macro_rules! call_info {
    () => {
        // uses __private import path because call_info! is used in proc macros
        $crate::__private::proc_macro_reexports::CallInfo { location: std::panic::Location::caller() }
    };
}

#[doc(hidden)]
#[macro_export]
/// access the recording context while registering the `#[track_caller]` caller
/// which will be used by error messages that happenin within `$f`
///
/// alternatively use [`ctx_explicit_caller!`] if you want to pass in the
/// call info explicitly
macro_rules! try_ctx_track_caller {
    ($f: expr) => {
        $crate::ir::recording::Context::try_with($crate::call_info!(), $f)
    };
}

impl CallInfo {
    /// returns the `CallInfo` (code location) of the caller of this function
    #[track_caller]
    pub fn caller() -> Self { call_info!() }
}

#[doc(hidden)]
#[macro_export]
/// access the recording context while registering the `$call_info`
/// which will be used by error messages that happenin within `$f`
macro_rules! try_ctx_explicit_caller {
    ($call_info: expr, $f: expr) => {
        $crate::ir::recording::Context::try_with($call_info, $f)
    };
}

/// access the recording context while registering the `#[track_caller]` caller
/// which will be used by error messages that happenin within `$f`
///
/// alternatively use [`ctx_explicit_caller!`] if you want to pass in the
/// call info explicitly
#[doc(hidden)]
#[macro_export]
macro_rules! try_ctx_track_caller_mut {
    ($f: expr) => {
        $crate::ir::recording::Context::try_with_mut($crate::call_info!(), $f)
    };
}

#[doc(hidden)]
#[macro_export]
/// access the recording context while registering the `$call_info`
/// which will be used by error messages that happenin within `$f`
macro_rules! try_ctx_explicit_caller_mut {
    ($call_info: expr, $f: expr) => {
        $crate::ir::recording::Context::try_with_mut($call_info, $f)
    };
}
