use thiserror::Error;

use super::{type_layout::TypeLayout, vec_range::VecRangeError};
use crate::ir::{self, Type};

#[derive(Error, Debug, Clone)]
pub enum FrontendError {
    #[error(
        "trying to downcast a value with dynamic type `{dynamic_type}` into rust type which expects dynamic type `{rust_type}`"
    )]
    InvalidDowncast { dynamic_type: Type, rust_type: Type },
    #[error("trying to compose a value of rust type `{rust_type}` from invalid amount of values: `{amount}`")]
    InvalidCompositeDowncast { amount: usize, rust_type: Type },
    #[error("trying to initialize a struct with {expected} fields from {actual} values")]
    InvalidCompositeDowncastAmount { expected: usize, actual: usize },
    #[error("invalid downcast from `{0}` to type `{1}` which is not available inside the shader typesystem")]
    InvalidDowncastToNonShaderType(Type, TypeLayout),
    #[error("internal u32 unpacking in packed vectors not supported yet")]
    PackedVecU32UnpackingNotSupportedYet,
    #[error("trying to initialize a single value from {amount} values")]
    InvalidDowncastAmount { amount: usize },
    //TODO(release) depending on the interpretation of "unchecked" in `from_anys_unchecked` which can provoke this error, this might be considered an internal error or not.
    #[error("vertex buffer layouts may only contain non-bool vector or scalar types. The layout is: {0}")]
    MalformedVertexBufferLayout(TypeLayout),
    #[error(
        "cannot downcast a value whose dynamic type has address space {dynamic_as} to a rust type with address space {rust_as}"
    )]
    DowncastWithInvalidAddressSpace {
        dynamic_as: ir::AddressSpace,
        rust_as: ir::AddressSpace,
    },
    #[error(
        "cannot downcast a value whose dynamic type {dynamic_type} is not a reference to a rust type that is a reference to {rust_type}"
    )]
    DowncastNonRefToRef {
        dynamic_type: ir::Type,
        rust_type: ir::StoreType,
    },
    #[error(transparent)]
    VecRangeError(#[from] VecRangeError),
}
