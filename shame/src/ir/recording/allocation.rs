use ir_type::AddressSpace;
use thiserror::Error;

use super::{call_info, time::TimeInstant, CallInfo, Context, Ident, Node};
use crate::frontend::any::Any;
use crate::{
    call_info,
    common::pool::Key,
    frontend::error::InternalError,
    ir::{
        self,
        ir_type::{self, StoreType},
        pipeline::{self, PossibleStages},
        Type,
    },
    mem::{self, SupportsAccess},
};
use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRegion {
    pub(crate) call_info: CallInfo,
    pub(crate) ty: StoreType,
    /// whether the allocation is not a specific allocation, but one that is a stand-in for a parameter provided
    /// at call site of a function, or one of a ptr returned from a function call.
    /// TODO(release) low prio: this might be an outdated concept
    pub(crate) dependence: Option<Dependence>,
    pub(crate) allowed_access: ir_type::AccessMode,
    pub(crate) address_space: ir_type::AddressSpace,
    pub(crate) stages: PossibleStages,
    pub(crate) ident: Option<Key<Ident>>,
    pub(crate) interactions: RefCell<Vec<MemoryInteractionEvent>>,
    /// empty field to force a private constructor, so that invariants must be checked through `new`
    private_ctor: (),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AllocError {
    #[error("allocation cannot be created because address space `{0}` does not support access mode {1:?}")]
    AddressSpaceDoesNotSupportAccessMode(ir::AddressSpace, ir::AccessMode),

    #[error(
        "memory for type `{0}` cannot be allocated in `{1}` address space. Some Address spaces have requirements wrt types containing bools/atomics, or runtime-sizedness."
    )]
    CannotAllocateTypeInAddressSpace(ir::SizedType, ir::AddressSpace),

    #[error("allocation in address space `{0}` cannot be initialized with a value. Use default initialization instead")]
    AllocationDoesNotSupportInitialValues(ir::AddressSpace),

    #[error(
        "memory in `{1}` address space must be of constructible type. `{0}` is not constructible. \
   Constructible types are types that don't contain atomics or runtime-sized arrays. \
   see https://www.w3.org/TR/WGSL/#constructible"
    )]
    AddressSpaceRequiresConstructibleType(ir::SizedType, ir::AddressSpace),

    #[error(
        "memory in `{1}` address space must be of plain fixed-foodprint type. `{0}` does \
    not satisfy this requirement. see https://www.w3.org/TR/WGSL/#plain-types-section \
    and https://www.w3.org/TR/WGSL/#fixed-footprint-types"
    )]
    AddressSpaceRequiresPlainFixedFootprint(ir::Type, ir::AddressSpace),

    #[error("memory for type `{0}` cannot be initialized with value of type `{1}`")]
    InitWithWrongType(ir::SizedType, Type),

    #[error("trying to initialize memory of type `{0}` with invalid value")]
    InitWithInvalidAny(ir::SizedType),

    #[error("direct allocation within the `{0:?}` address space is not possible. Use {} instead for this address space. \
    If you are generally unsure about which address space to choose, use `Fn`.", api_for_address_space(.0))]
    CannotDirectlyAlloc(AddressSpace),

    // TODO(release) this can potentially be relaxed to allow Workgroup and private allocs inside conditional blocks
    #[error(
        "allocation within the `{address_space:?}` address space is not possible inside block recording at {block_caller} . \
    Allocation in this address space is only possible in the \"encoding scope\", \
    which is outside of any \n\
    - conditional block recordings\n\
    - loop block recordings of any kind\n\
    - function block recording\n\
    This allocation would succeed if it was instead in the `Fn` address space."
    )]
    RequiresEncodingScope {
        address_space: AddressSpace,
        block_caller: CallInfo,
    },
    #[error(
        "memory `{action:?}` expression of `{ty}` is invalid on a memory region with `{access}` access in the `{addr}` address space."
    )]
    DisallowedInteraction {
        action: InteractionKind,
        access: ir::AccessMode,
        ty: ir::StoreType,
        addr: AddressSpace,
    },
    #[error("cannot return a memory view (i.e. a pointer or reference type) from a recorded function")]
    CannotReturnMemoryView,
}

fn api_for_address_space(addr: &AddressSpace) -> &'static str {
    use AddressSpace as AS;
    match addr {
        AS::Function => "shame::Cell::new",
        AS::Thread => "alloc_in",
        AS::WorkGroup => "alloc_in",
        AS::Uniform | AS::Storage | AS::Handle => "bind group/bindings",
        AS::PushConstant => "the push constant pipeline feature",
        AS::Output => "color/depth targets",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dependence {
    FnParameter,
    /// currently unsupported, since no target language allows returning
    /// pointers or references from a function
    ReturnedFromFn,
}

impl MemoryRegion {
    pub(crate) fn new(
        call_info: CallInfo,
        ty: StoreType,
        ident: Option<Key<Ident>>,
        dependence: Option<Dependence>,
        am: ir_type::AccessMode,
        addr: ir_type::AddressSpace,
    ) -> Result<Rc<Self>, AllocError> {
        match addr.supports_access(am) {
            false => Err(AllocError::AddressSpaceDoesNotSupportAccessMode(addr, am)),
            true => Ok(Rc::new(Self {
                call_info,
                ty,
                dependence,
                allowed_access: am,
                address_space: addr,
                stages: PossibleStages::all(),
                interactions: Default::default(),
                ident,
                private_ctor: (),
            })),
        }
    }

    /// if a memory view is returned from a call to a user-recorded function,
    /// this constructor is called to create a new allocation instance for the
    /// returned `Any`, which has `Dependence::ReturnedFromFn` in order to be
    /// tracked independently of the return values from other calls to the same
    /// user-recorded function.
    pub fn new_return_value_dependent(self: Rc<Self>, ctx: &Context) -> Rc<Self> {
        // at the time of writing, WGSL is the only target language that supports pointers.
        // The WGSL spec explicitly states functions must not return ref or ptr types.
        // In order to release this library earlier we skip the proper tracking
        // of ptr return types and always return an error here.
        ctx.push_error(AllocError::CannotReturnMemoryView.into());
        Rc::new(MemoryRegion {
            call_info: ctx.latest_user_caller(),
            ty: self.ty.clone(),
            dependence: Some(Dependence::ReturnedFromFn),
            allowed_access: self.allowed_access,
            address_space: self.address_space,
            ident: None,
            stages: PossibleStages::all(),
            interactions: Default::default(),
            private_ctor: (),
        })
    }

    pub fn is_writeable(&self) -> bool { self.address_space.is_writeable() && self.allowed_access.is_writeable() }
    pub fn is_readable(&self) -> bool { self.address_space.is_readable() && self.allowed_access.is_readable() }

    pub(crate) fn add_interaction(&self, interaction: MemoryInteractionEvent, ctx: &Context) {
        let is_access_violation = match interaction.kind {
            InteractionKind::Init | InteractionKind::Write => !self.is_writeable(),
            InteractionKind::Read => !self.is_readable(),
            InteractionKind::ReadWrite => !(self.is_readable() && self.is_writeable()),
        };
        if is_access_violation {
            ctx.push_error(
                AllocError::DisallowedInteraction {
                    action: interaction.kind,
                    access: self.allowed_access,
                    ty: self.ty.clone(),
                    addr: self.address_space,
                }
                .into(),
            );
        }

        self.interactions.borrow_mut().push(interaction)
    }

    #[track_caller]
    /// `ref_arg` can be a `Type::Ref` or `Type::Ptr`
    ///
    /// this function just calls `add_interaction` internally
    // TODO(release) for now, this is called manually in the Any api at every place
    // that constitutes a memory interaction. Instead we could match the arg types of expr-node
    // recordings in general and see if they contain memory views (= Type::Ref or Type::Ptr).
    // The problem with that approach is that we don't know if that memory view is
    // actually accessed by that expression, or if its just an "unused" argument.
    // (imagine for example a (currently non-existent) pointer arithmetic + operation.
    // that operation would not actually access the pointer)
    // Therefore the current approach was chosen, but maybe there's a better one
    pub(crate) fn record_interaction(expr: Any, ref_arg: Any, kind: InteractionKind) {
        Context::try_with(call_info!(), |ctx| {
            // Type checker already generates encoding-errors for the return cases here.
            let Some(ref_key) = ref_arg.node() else {
                return; // skip if `ref_arg` was passed in as invalid, when `expr` was recorded
            };
            let Some(interaction_node_key) = expr.node() else {
                return; // skipped if `expr`'s type check failed (which already pushed an encoding error)
            };
            let ref_node = &ctx.pool()[ref_key];
            let (Type::Ref(region, _, _) | Type::Ptr(region, _, _)) = &ref_node.ty else {
                let msg =
                    format!("memory {kind:?} expression was successfully typechecked with non-memory view argument");
                ctx.push_error(InternalError::new(true, msg).into());
                return;
            };
            region.add_interaction(
                MemoryInteractionEvent {
                    kind,
                    node: interaction_node_key,
                },
                ctx,
            );
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionKind {
    Init,
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryInteractionEvent {
    pub(crate) kind: InteractionKind,
    pub(crate) node: Key<Node>,
}
