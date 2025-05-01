use std::{fmt::Display, rc::Rc};

use thiserror::Error;

use crate::{
    call_info, f32x2,
    frontend::{
        encoding::{EncodingError, EncodingErrorKind},
        error::InternalError,
    },
    ir::{
        self,
        expr::{Expr, ShaderIo},
        ir_type::{CanonName, Struct},
        pipeline::ShaderStage,
        recording::{CallInfo, MemoryRegion, Stmt, TemplateStructParams},
        AccessMode, AddressSpace, TextureFormatWrapper, TextureSampleUsageType, Type,
    },
    stringify_checked, MipFn,
};

use super::WgslContext;

#[derive(Clone, Copy)]
pub enum WgslErrorLevel {
    Original,
    InternalPleaseReport,
}

pub struct WgslError {
    pub call_info: CallInfo,
    pub level: WgslErrorLevel,
    pub error: WgslErrorKind,
}

impl WgslError {
    pub(super) fn into_encoding_err(self, ctx: &WgslContext) -> EncodingError {
        /// transform the error into an `EncodingErrorKind::Internal` if the
        /// level suggests it.
        let error = match self.level {
            WgslErrorLevel::Original => EncodingErrorKind::WgslError(self.error),
            WgslErrorLevel::InternalPleaseReport => InternalError::new(true, self.error.to_string()).into(),
        };
        EncodingError {
            location: self.call_info,
            use_colors: ctx.ctx.settings().colored_error_messages,
            write_excerpt: ctx.ctx.settings().error_excerpt,
            error,
        }
    }
}

impl From<std::fmt::Error> for WgslError {
    fn from(x: std::fmt::Error) -> Self {
        WgslError {
            level: WgslErrorLevel::Original,
            call_info: call_info!(),
            error: WgslErrorKind::StdFmt(x),
        }
    }
}

#[derive(Debug, Error, Clone)]
pub enum WgslErrorKind {
    #[error("std::fmt::Error occured while writing shader: {0}")]
    StdFmt(std::fmt::Error),
    #[error("f64 types are unsupported by WGSL. Another target language may support f64")]
    F64Unsupported,
    #[error("attempt to generate NaN floating point literals, NaN constants are not supported in WGSL")]
    NaNUnsupported,
    #[error("attempt to generate +Inf or -Inf floating point literals, infinity constants are not supported in WGSL")]
    InfUnsupported,
    #[error("type `{0}` may not appear in WGSL code, but the code generator is attempting to write it.")]
    TypeMayNotAppearInWrittenForm(Type),
    #[error("allocation statement expr `{0:?}` evaluates to type `{1}` which is not a reference type.")]
    AllocationExprReturnsNonRefType(Expr, Type),
    #[error("attempt to codegen a non-init alloc expr `{0:?}` of type `{1}`")]
    AttemptToWriteNonInitAllocExpr(Expr, Type),
    #[error(
        "address space `{0}` may not appear in WGSL code, but the code generator is attempting to write it. Explicit allocation in this address space is disallowed."
    )]
    AddressSpaceMayNotAppearInWrittenForm(AddressSpace),
    #[error("expression `{expr:?}` has invalid amount of arguments {actual} (expected {expected})")]
    InvalidAmountOfArguments { expr: Expr, expected: u32, actual: u32 },
    #[error("unexpected expression `{expr:?}`, expected `{expected}`")]
    UnexpectedExpression { expr: Expr, expected: &'static str },
    #[error("allocation is missing a variable identifier. Region: {0:#?}")]
    AllocationHasNoIdent(Rc<MemoryRegion>),
    #[error("io expr `{0:?}` is missing an identifier")]
    IoHasNoIdent(ShaderIo),
    #[error("missing struct definition for struct with canonical name `{0}`")]
    MissingStructDefinition(String),
    #[error("missing definition for builtin template struct of instantiation `{0}`")]
    MissingTemplateStructDefinition(TemplateStructParams),
    #[error("function argument #{0} of function {1} has no identifier assigned to it")]
    NthFunctionArgHasNoIdent(u32, String),
    #[error("{0} is missing an identifier")]
    MissingIdent(&'static str),
    #[error("access mode `{0}` is not supported by address space {1:?}")]
    AccessModeNotSupportedByAddressSpace(AccessMode, AddressSpace),
    #[error("information about {0} is missing in the pipeline recording")]
    MissingInfo(&'static str),
    #[error("invalid color target format during shader creation. color target has no corresponding shader datatype")]
    ColorTargetHasNoShaderType(TextureFormatWrapper),
    #[error("texture format unsupported by expression {0:?}")]
    TextureFormatUnsupported(Expr),
    #[error("WGSL output does not support the {0} shader stage")]
    UnsupportedStage(ShaderStage),
    #[error("trying to generate code for access of field `{1}` on non-struct type `{0}`")]
    FieldAccessOnNonStruct(Type, CanonName),
    #[error("trying to generate code for access of unknown field `{1}` on structure `{0:?}`")]
    UnknownFieldForStruct(Type, CanonName),
    #[error("`{0}` has no identifier deduplication entry")]
    UnregisteredStruct(Rc<Struct>),
    #[error("trying to generate code for texture format `{0:?}` which cannot be represented in wgsl code.")]
    UnrepresentableTextureFormat(TextureFormatWrapper),
    #[error("expression `{0}` cannot be turned into a standalone statement or phony assignment")]
    ExprCannotBeAStatement(Expr),
    #[error("invalid return type of BitCast: {0}")]
    BitcastCannotReturnType(Type),
    #[error("In WGSL 1D-textures can only be sampled with {} and no offset. Alternatively you can use the non-sampler-based access functions for 1D textures",
    stringify_checked!(expr: MipFn::<T>::Quad).to_string().replace(" ", "").replace("::<T>::", "::"))]
    Texture1DRequiresImplicitGradNoOffset,
    #[error("In WGSL depth-textures do not support {} and {}", 
        stringify_checked!(expr: MipFn::<T>::QuadBiased).to_string().replace(" ", "").replace("::<T>::", "::"), 
        stringify_checked!(expr: MipFn::<T>::Grad).to_string().replace(" ", "").replace("::<T>::", "::"))]
    DepthTexturesDontSupportBiasedOrExplicitGradientSampling,
}

// dummy type for error message
type T = f32x2;

impl WgslErrorKind {
    /// error at a given call site
    pub fn at(self, call_info: CallInfo) -> WgslError {
        WgslError {
            call_info,
            error: self,
            level: WgslErrorLevel::Original,
        }
    }

    /// error at a given call site, with given level.
    pub fn at_level(self, call_info: CallInfo, level: WgslErrorLevel) -> WgslError {
        WgslError {
            call_info,
            error: self,
            level,
        }
    }
}
