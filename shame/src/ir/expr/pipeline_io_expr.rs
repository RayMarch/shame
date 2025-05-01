use std::fmt::Display;

use crate::{call_info, StageMask};
use crate::frontend::any::shared_io::BindPath;
use crate::frontend::error::InternalError;
use crate::ir::pipeline::PossibleStages;
use crate::ir::recording::Context;
use crate::ir::{self, Type};

use super::type_check::{SignatureStrings, TypeCheck, TypeShorthandLevel};
use super::{Expr, NoMatchingSignature};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineIo {
    Binding(Binding),
    PushConstantsField(PushConstantsField),
}

impl Display for PipelineIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineIo::Binding(x) => write!(f, "{x}"),
            PipelineIo::PushConstantsField(x) => write!(f, "{x}"),
        }
    }
}

impl TypeCheck for PipelineIo {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        match self {
            PipelineIo::Binding(binding) => binding.infer_type(args),
            PipelineIo::PushConstantsField(push_constant) => push_constant.infer_type(args),
        }
    }
}

impl PipelineIo {
    pub fn possible_stages(&self) -> PossibleStages {
        match self {
            PipelineIo::Binding(binding) => {
                // non vertex-writeable storage is handled in the stage solver separately
                PossibleStages::all()
            }
            PipelineIo::PushConstantsField(push_constants_field) => PossibleStages::all(),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Binding {
    pub bind_path: BindPath,
    pub ty: Type,
}

impl std::fmt::Display for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bind_groups[{}][{}]: {}",
            self.bind_path.0, self.bind_path.1, self.ty
        )
    }
}

impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "binding {{{}, ty: {}}}", self.bind_path, self.ty)
    }
}

impl TypeCheck for Binding {
    fn infer_type(&self, args: &[Type]) -> Result<Type, super::NoMatchingSignature> {
        // the binding-expr has no arguments, the only thing checked here is
        // that the argument list is empty.
        // The structure of the return type is validated in the [`Any::binding`] calls
        match args {
            [] => Ok(self.ty.clone()),
            _ => Err(NoMatchingSignature {
                expression_name: format!("Bind<{}>", self.ty).into(),
                arguments: args.iter().cloned().collect(),
                allowed_signatures: SignatureStrings::Static(&[]),
                shorthand_level: TypeShorthandLevel::default(),
                signature_formatting: None,
                comment: Some(format!("({})", self.bind_path)),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushConstantsField {
    pub field_index: usize,
    pub ty: ir::SizedType,
}

impl Display for PushConstantsField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "push_constants._{}: {}", self.field_index, &self.ty)
    }
}

impl TypeCheck for PushConstantsField {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        // the push-constant-expr has no arguments, the only thing checked here is
        // that the argument list is empty.
        // The structure of the return type is validated in the [`Any::get_push_constants`] calls
        match args {
            [] => Ok(self.ty.clone().into()),
            _ => Err(NoMatchingSignature {
                expression_name: format!("PushConstantField<{}, {}>", self.field_index, self.ty).into(),
                arguments: args.iter().cloned().collect(),
                allowed_signatures: SignatureStrings::Static(&[]),
                shorthand_level: TypeShorthandLevel::default(),
                signature_formatting: None,
                comment: None,
            }),
        }
    }
}
