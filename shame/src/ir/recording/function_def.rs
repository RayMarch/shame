use thiserror::Error;

use crate::{
    common::{pool::Key, small_vec::SmallVec},
    frontend::any::{fn_builder::PassAs, InvalidReason},
    ir::{AccessMode, AddressSpace, StoreType, Type},
};

use super::{Block, CallInfo, Ident, Node};

#[derive(Debug, Clone, Error)]
pub enum FnError {
    #[error(
        "Wrong amount of parameters. Trying to access {expected} parameters of a function which has {got} parameters."
    )]
    ExtractingWrongAmountOfParams { expected: usize, got: usize },
    #[error("return value is unavailable, reason: {0}")]
    ReturnValueNotAvailable(InvalidReason),
    #[error("function definition is not available, reason: {0}")]
    FnDefinitionNotAvailable(InvalidReason),
    #[error("invalid type {1:?} for a {0:?} function parameter")]
    InvalidFunctionParameterType(PassAs, StoreType),
    #[error("access mode `{0}` is not supported by address space {1:?}")]
    AccessModeNotSupportedByAddressSpace(AccessMode, AddressSpace),
}

/// function definition
#[derive(Debug, PartialEq, Eq)]
pub struct FunctionDef {
    /// rust location of the function recording start
    pub(crate) call_info: CallInfo,
    /// the function name as it will be shown in the shader, and as it is required
    /// to generate call expressions
    pub(crate) ident: Key<Ident>,
    /// the function parameters from the perspective of within the function
    /// (not the function's call site)
    pub(crate) params: SmallVec<Key<Node>, 4>,
    /// the function's return value from the perspective of within the function
    /// (not the function's call site)
    pub(crate) return_: Option<Key<Node>>,
    /// the block of statements within the function
    pub(crate) body: Key<Block>,
}
