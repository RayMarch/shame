use crate::{context::ShaderKind, expr::Loc};
use std::{ops::Range, rc::Rc};
use thiserror::*;

#[derive(Error, Debug)]
pub enum Error {
    #[error("trying to use a {} shader expression that is not available in a {} shader context", .found, .expected)]
    NAInShaderKind{expected: ShaderKind, found: ShaderKind},
    // #[error("trying to use a {} shader expression together with a {} shader expression", .0, .1)]
    // IncompatibleShaderKind(ShaderKind, ShaderKind),
    #[error("expression not available: \n{}", .reason)]
    NA {reason: &'static str}, //value not available e.g. because it belongs to a different shader //&str instead of String becuase this happens intentionally a lot when recording multiple shader types with the same code
    #[error("expression depends on N/A value: \n{}", .0)]
    NADependent(Rc<String>), //value depends (transitively) on a not-available value
    #[error("record-time type error: \n{}", .0)]
    TypeError(String),
    #[error("record-time struct field access error: \n{}", .0)]
    FieldSelectError(String),
    #[error("record-time argument error: \n{}", .0)]
    ArgumentError(String),
    #[error("record-time out of bounds error: \n{}", .0)]
    OutOfBounds(String),
    #[error("value used outside of its valid scope: \n{}", .0)]
    ScopeError(String),
    #[error("vertex attribute location range {:?} overlapping with {:?}", .0, .1)]
    OverlappingAttributeLocation(Range<Loc>, Range<Loc>),
    #[error("fragment color output location {} is used multiple times", .duplicate_location)]
    OverlappingColorAttachmentLocation{duplicate_location: Loc},
    #[error("BindGroup index {} is used multiple times", .duplicate_index)]
    OverlappingBindGroupIndex{duplicate_index: Loc},
    #[error("In BindGroup {}: \nBinding index {} is used multiple times", .bind_group, .duplicate_index)]
    OverlappingBindingIndex{bind_group: Loc, duplicate_index: Loc},
    #[error("cannot record floating point number of category {:?}", .0)]
    UnsupportedFloadingPointCategory(std::num::FpCategory),
    #[error("assertion failed: {}", .0)]
    AssertionFailed(String),
}

#[derive(Error, Debug)]
pub enum Warning {
    #[error("assigning name '{}' to expression that already has the name '{}'", .new_name, .previous_name)]
    ForcedBindingAlreadyBound{previous_name: String, new_name: String},
}
