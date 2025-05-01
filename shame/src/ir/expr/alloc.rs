use std::rc::Rc;

use crate::frontend::any::Any;
use crate::ir::recording::NodeRecordingError;
use crate::{
    call_info,
    common::option_ext::slice_from_opt,
    frontend::any::{ArgumentNotAvailable, InvalidReason},
    impl_track_caller_fn_any,
    ir::{
        expr::Expr,
        ir_type::{
            AccessMode, AddressSpace, Indirection,
            Len::*,
            Len2,
            ScalarType::{self, *},
            StoreType::*,
        },
        recording::{
            AllocError, AllocStmt, Context, ExprStmt, Ident, InteractionKind, MemoryInteractionEvent, MemoryRegion,
            Priority, Stmt, TimeInstant,
        },
        Len, SizedType,
    },
    try_ctx_track_caller,
};
use crate::{ir, ir::ir_type::StoreType, ir::Type, ir::Type::*, same, sig};

use super::{type_check::SigFormatting, NoMatchingSignature, TypeCheck};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarIdent(pub Rc<MemoryRegion>);

impl std::fmt::Display for VarIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "var-ident: {}",
            Type::Ref(self.0.clone(), self.0.ty.clone(), self.0.allowed_access)
        )
    }
}

impl TypeCheck for VarIdent {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        let region = Rc::clone(&self.0);
        let access = region.allowed_access;
        let t = region.ty.clone();
        sig!(
            { fmt: SigFormatting::RemoveAsterisksAndClone, },
            [] => Type::Ref(region, t, access),
        )(self, args)
    }
}

impl Any {
    // see `https://www.w3.org/TR/WGSL/#var-decls`
    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn alloc_explicit(address_space: AddressSpace, ty: SizedType, initial_value: Option<Any>) -> Any {
        /// only `Storage` supports two access modes, and `Storage` can only be
        /// accessed via binding, so no need to support anything but default access here
        let access = address_space.default_access_mode();
        Context::try_with(call_info!(), |ctx| {
            use AddressSpace as AS;
            let addr_check = match address_space {
                AS::Function | AS::Thread => {
                    let initial_value_type_check = match initial_value {
                        None => Ok(()),
                        Some(val) => match val.ty() {
                            None => Err(AllocError::InitWithInvalidAny(ty.clone())),
                            Some(Store(Sized(val_ty))) if val_ty == ty => Ok(()),
                            Some(val_ty) => Err(AllocError::InitWithWrongType(ty.clone(), val_ty)),
                        },
                    };

                    let constructible_check = match ty.is_constructible() {
                        true => Ok(()),
                        false => Err(AllocError::AddressSpaceRequiresConstructibleType(
                            ty.clone(),
                            address_space,
                        )),
                    };

                    initial_value_type_check.and(constructible_check)
                }
                AS::WorkGroup => {
                    if initial_value.is_some() {
                        Err(AllocError::AllocationDoesNotSupportInitialValues(address_space))
                    } else {
                        let ty = Store(Sized(ty.clone())); // this is here in case this function gets refactored to take a non-sized type in the future
                        if ty.is_plain_and_fixed_footprint() {
                            Ok(())
                        } else {
                            Err(AllocError::AddressSpaceRequiresPlainFixedFootprint(ty, address_space))
                        }
                    }
                }
                AS::Uniform | AS::Storage | AS::Handle | AS::PushConstant | AS::Output => {
                    Err(AllocError::CannotDirectlyAlloc(address_space))
                }
            };

            match addr_check.and_then(|()| {
                MemoryRegion::new(
                    ctx.latest_user_caller(),
                    StoreType::Sized(ty),
                    Some(ctx.pool_mut().push(Ident::Chosen(Priority::Auto, "c".into()))),
                    None,
                    access,
                    address_space,
                )
            }) {
                Err(e) => ctx.push_error_get_invalid_any(e.into()),
                Ok(region) => {
                    let initial_value = initial_value.and_then(|any| any.node());
                    if let Some(node) = initial_value {
                        region.add_interaction(
                            MemoryInteractionEvent {
                                kind: InteractionKind::Init,
                                node,
                            },
                            ctx,
                        );
                    }
                    ctx.push_alloc_stmt_to_current_block(
                        AllocStmt {
                            allocation: region.clone(),
                            initial_value,
                        },
                        TimeInstant::next(),
                        ctx.latest_user_caller(),
                    );

                    ctx.push_node(Expr::VarIdent(VarIdent(region)), &[])
                }
            }
        })
        .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
    }

    #[track_caller]
    #[doc(hidden)] // runtime api
    // alloc with no initializer value (initializes with default (zero-) value)
    pub fn alloc_default_in(address_space: AddressSpace, ty: SizedType) -> Any {
        Any::alloc_explicit(address_space, ty, None)
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn alloc(ty: SizedType, initial_value: Any) -> Any {
        Any::alloc_explicit(AddressSpace::Function, ty, Some(initial_value))
    }
}
