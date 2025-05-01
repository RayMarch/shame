use thiserror::Error;

pub use crate::frontend::any::Any;
use crate::{
    call_info,
    common::{pool::Key, small_vec::SmallVec},
    frontend::{encoding::EncodingErrorKind, error::InternalError},
    ir::{
        self,
        expr::{ArgViewKind, Expr, FnRelated},
        recording::{
            AllocError, BlockKind, BlockSeriesRecorder, BodyKind, CallInfo, Context, Dependence, FlowStmt, FnError,
            FunctionDef, Ident, Jump, MemoryRegion, Priority, Stmt, TimeInstant,
        },
        AccessMode, Node, SizedType, Type,
    },
};

use super::InvalidReason;

/// How a function argument is passed, and which type it has
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassAs {
    /// pass by value. If called with a `Type::Store`, passes the value.
    /// If called with a `Type::Ref`, performs a read on that reference and
    /// passes that value.
    ///
    /// callable with `Type::Store` and `Type::Ref`
    Value,
    /// *not available for GLSL targets*
    ///
    /// callable with `Ptr(a, store_type, AccessMode { write: true, .. }),`
    /// where `a` is an allocation in the specified address space
    Ptr(ir::AddressSpace, AccessMode),
    /// callable with `Ref(a, store_type, AccessMode { write: true, .. }),`
    /// where `a` is an allocation in the specified address space
    ///
    /// ### Behavior: "copy-out"
    /// the respective function argument is exposed as a `AddressSpace::Function`
    /// allocation with `AccessMode::write_only()` inside the function.
    /// Its value is read at the end of the function call and copied into
    /// the reference parameter provided at call site.
    Out,
    /// callable with `Ref(allocation, store_type, AccessMode::read_write())`.
    ///
    /// ### Behavior: "copy-in, copy-out"
    /// the parameter value is being read when the function is called and
    /// stored into a new allocation that is local to the function. This
    /// allocation is exposed as the respective argument (of type `Ref(a, store_type, AccessMode::read_write())`, where `a` has `AddressSpace::Function`)
    /// inside the function recording block.
    /// After the function's execution is finished the argument's value is written back
    /// into the parameter's referenced allocation.
    InOut,
}

#[allow(missing_docs)]
#[derive(Clone, PartialEq, Eq)]
pub struct Param(pub PassAs, pub ir::StoreType);

#[allow(missing_docs)]
pub struct AnyFunction {
    key: Result<Key<FunctionDef>, InvalidReason>,
}

impl AnyFunction {
    pub(crate) fn new_invalid(reason: InvalidReason) -> Self { Self { key: Err(reason) } }
}

#[allow(missing_docs)]
pub struct FnRecorder {
    call_info: CallInfo,
    ident: Ident,
    params: SmallVec<Any, 4>,
    rec: BlockSeriesRecorder<1>,
}

impl Any {
    #[track_caller]
    fn new_function_param(call_info: CallInfo, Param(pass_as, store_ty): Param) -> Any {
        use ir::StoreType as ST;
        use PassAs as PA;

        Context::try_with(call_info, |ctx| {
            let expr = || -> Result<_, EncodingErrorKind> {
                // validate types
                // https://www.w3.org/TR/WGSL/#function-restriction
                match (pass_as, &store_ty) {
                    // sized types
                    (PA::Value | PA::InOut | PA::Out | PA::Ptr(..), ST::Sized(_)) => Ok(()),

                    // runtime-sized array
                    (PA::Ptr(addr, am), ST::RuntimeSizedArray(_)) => Ok(()),

                    // handles (not supported for now, even though shader languages technically support this)
                    (_, ST::Handle(_)) => Err(FnError::InvalidFunctionParameterType(pass_as, store_ty.clone())),

                    _ => Err(FnError::InvalidFunctionParameterType(pass_as, store_ty.clone())),
                }?;

                // create expr
                Ok(match pass_as {
                    PassAs::Value => Expr::FnRelated(FnRelated::FnParamValue(store_ty)),
                    PassAs::Ptr(addr, am) => {
                        if !addr.supports_access(am) {
                            return Err(FnError::AccessModeNotSupportedByAddressSpace(am, addr).into());
                        }
                        let alloc =
                            MemoryRegion::new(call_info, store_ty, None, Some(Dependence::FnParameter), am, addr)?;
                        Expr::FnRelated(FnRelated::FnParamMemoryView(alloc, ArgViewKind::Ptr))
                    }
                    PassAs::Out | PassAs::InOut => {
                        // in both cases (out/inout) the access mode is read+write.
                        // "out" just means that the value is not copied in, but the
                        // variable itself has read access as well.
                        let rw = AccessMode::ReadWrite;
                        let alloc = MemoryRegion::new(
                            call_info,
                            store_ty,
                            Some(ctx.pool_mut().push(Ident::Unchosen)),
                            Some(Dependence::FnParameter),
                            rw,
                            ir::AddressSpace::Function,
                        )?;
                        Expr::FnRelated(FnRelated::FnParamMemoryView(alloc, ArgViewKind::Ref))
                    }
                })
            };
            match expr() {
                Ok(expr) => {
                    let any = ctx.push_node(expr, &[]);
                    any.try_set_ident_internal(Ident::Unchosen); // needs ident for function signature
                    any
                }
                Err(e) => {
                    ctx.push_error(e);
                    Any::new_invalid(InvalidReason::ErrorThatWasPushed)
                }
            }
        })
        .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
    }
}

impl FnRecorder {
    #[track_caller]
    #[allow(missing_docs)]
    pub fn new(suggested_ident: Option<String>, param_declarations: &[Param]) -> Self {
        let call_info = call_info!();
        // block recording is started first
        let rec = BlockSeriesRecorder::new(call_info, BlockKind::Body(BodyKind::Function));
        // then function params are created inside of that block for scope tracking reasons
        let params = param_declarations
            .iter()
            .cloned()
            .map(|arg| Any::new_function_param(call_info, arg))
            .collect();
        Self {
            call_info,
            ident: match suggested_ident {
                Some(name) => Ident::Chosen(Priority::UserHint, name),
                None => Ident::Unchosen,
            },
            params,
            rec,
        }
    }

    /// returns a sized array of the function's args.
    /// If `self.args().len() != N` the encoding fails with an error and the returned
    /// `Any` instances will not record any operations performed on them.
    #[track_caller]
    #[allow(missing_docs)]
    pub fn args_sized<const N: usize>(&self) -> [Any; N] {
        let call_info = call_info!();
        let len = self.args().len();
        <[Any; N]>::try_from(self.args()).unwrap_or_else(|_| {
            let err = FnError::ExtractingWrongAmountOfParams { expected: N, got: len }.into();
            let reason = match Context::try_with(call_info, |ctx| ctx.push_error(err)) {
                Some(()) => InvalidReason::ErrorThatWasPushed,
                None => InvalidReason::CreatedWithNoActiveEncoding,
            };
            [Any::new_invalid(reason); N]
        })
    }

    #[track_caller]
    #[allow(missing_docs)]
    pub fn record_fn<const N: usize>(
        suggested_ident: Option<String>,
        args: [Param; N],
        f: impl FnOnce([Any; N]) -> Option<Any>,
    ) -> AnyFunction {
        let rec = Self::new(suggested_ident, &args);
        let return_ = f(rec.args_sized());
        rec.finish(return_)
    }

    #[track_caller]
    #[allow(missing_docs)]
    pub fn finish(self, return_value: Option<Any>) -> AnyFunction {
        let call_info = call_info!();
        Context::try_with(call_info, |ctx| {
            let return_ = return_value.map(|any| any.inner()).transpose();
            // record the return statement if necessary
            if let Ok(maybe_return) = return_ {
                ctx.push_flow_stmt_to_current_block(
                    FlowStmt::Jump(Jump::Return(maybe_return)),
                    TimeInstant::next(),
                    call_info,
                )
            }

            let args: SmallVec<Key<Node>, 4> = self.params.iter().filter_map(|any| any.node()).collect();
            let block = self.rec.finish(call_info);

            match (args.len() == self.params.len(), return_, block) {
                (true, Ok(return_), Ok([block])) => AnyFunction {
                    key: Ok(ctx.pool_mut().push(FunctionDef {
                        call_info: self.call_info,
                        ident: ctx.pool_mut().push(self.ident),
                        params: args,
                        return_,
                        body: block,
                    })),
                },
                (true, _, Err(block_error)) => {
                    ctx.push_error(block_error.into());
                    AnyFunction::new_invalid(InvalidReason::ErrorThatWasPushed)
                }
                (false, _, _) => {
                    let msg = format!(
                        "when finishing function recording: available argument mismatch, expected {}, got {}",
                        self.params.len(),
                        args.len()
                    );
                    ctx.push_error(InternalError::new(true, msg).into());
                    AnyFunction::new_invalid(InvalidReason::ErrorThatWasPushed)
                }
                (_, Err(reason), _) => {
                    ctx.push_error(FnError::ReturnValueNotAvailable(reason).into());
                    AnyFunction::new_invalid(InvalidReason::ErrorThatWasPushed)
                }
            }
        })
        .unwrap_or(AnyFunction::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
    }

    #[allow(missing_docs)]
    pub fn args(&self) -> &[Any] { &self.params }
}

impl AnyFunction {
    #[track_caller]
    #[allow(missing_docs)]
    pub fn call(&self, args: &[Any]) -> Option<Any> {
        Context::try_with(call_info!(), |ctx| match self.key {
            Ok(key) => {
                let has_return_value = ctx.pool()[key].return_.is_some();
                let return_value = ctx.push_node(Expr::FnRelated(FnRelated::Call(key)), args);
                has_return_value.then_some(return_value)
            }
            Err(reason) => {
                ctx.push_error(FnError::FnDefinitionNotAvailable(reason).into());
                Some(Any::new_invalid(InvalidReason::ErrorThatWasPushed))
            }
        })
        .unwrap_or(Some(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding)))
    }
}
