//! type erased runtime api of the shader generator
//!
//! see the documentation of [`Any`] for more information
use std::{cell::Cell, fmt::Display};

use crate::{
    call_info,
    common::{
        format::numeral_suffix,
        pool::{Key, PoolRef},
    },
    frontend::encoding::EncodingErrorKind,
    ir::{
        self,
        expr::{self, BuiltinShaderIn, ShaderIo},
        recording::{CallInfo, Context, Ident, Node, Priority},
    },
    try_ctx_track_caller,
};

/// blend modes for use with color target writes in render pipelines.
pub mod blend;
/// control structure builder types, using the typestate pattern
pub mod flow_builders;
/// types for building functions that appear as such in the shader code
pub mod fn_builder;
/// types for render pipeline io such as vertex/fragment inputs/outputs
pub mod render_io;
/// types for generic pipeline io such as bindings, push constants, ...
pub mod shared_io;

/// reason why an `Any` instance is in an invalid state
#[derive(Debug, Clone, Copy)]
pub enum InvalidReason {
    /// the `Any` instance is invalid because it was created outside of
    /// an active encoding
    CreatedWithNoActiveEncoding,
    /// the `Any` instance is invalid because of an error that was already
    /// pushed to the active context's errors.
    ErrorThatWasPushed,
}

impl Display for InvalidReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            InvalidReason::CreatedWithNoActiveEncoding => "created while no encoding was active on this thread",
            InvalidReason::ErrorThatWasPushed => "a previous error caused this",
        })
    }
}

/// Type erased [`GpuType`] for use with the runtime-typed api.
///
/// Each [`GpuType`] instance such as [`vec`] or [`Array`] can be turned into an
/// instance of [`Any`], which exposes the functionality of all [`GpuType`]s.
/// (this can be done with the [`AsAny`] trait using ``.as_any()``)
///
/// An [`Any`] instance knows its own dynamic type during shader-generation
/// at runtime.
/// Calling its functions is perfectly safe due to the runtime typechecker.
/// If e.g. a [`vec`] function is called on an [`Any`] that has non-[`vec`]
/// dynamic type, the runtime typechecker emits an [`EncodingError`], and
/// the encoding fails.
///
/// Downcasting of [`Any`] objects back into their [`GpuType`] is done via
/// [`From::from`]/[`Into::into`]. If the dynamic type does not match the [`GpuType`],
/// an [`EncodingError`] is generated and returned at the end of pipeline creation.
/// (this can be extended to fallible `TryFrom` implementations, contact me if
/// you need this)
///
/// This "gradual typing" api is useful if you want to generate shader variants
/// that contain code that uses different types depending on a pipeline specialization
/// parameter.
///
/// ## every [`GpuType`] uses [`Any`] internally
/// no code-generation overhead is caused by using [`Any`] objects,
/// as every [`GpuType`] uses them internally to implement all their functionality.
/// The [`Any`]-API was modelled closely after the webgpu shading language `WGSL`
/// since it is the most cross compatible shader language that is human readable,
/// and since the design decisions are transparent and documented on github issue
/// discussions in the gpuweb repository.
/// Most [`Any`] functions will have their original `WGSL` names or similar.
/// The [`GpuType`]s that use the [`Any`]-API internally act like a thin layer
/// that makes the WGSL semantics understandable to rustc, but also adds more
/// expressive modelling such as truly generic vectors (`vec<T, L>`), etc.
///
/// [`GpuType`]: crate::GpuType
/// [`AsAny`]: crate::any::AsAny
/// [`vec`]: crate::vec
/// [`Array`]: crate::Array
/// [`EncodingError`]: crate::EncodingError
#[derive(Debug, Clone, Copy)]
pub struct Any {
    node: Result<Key<Node>, InvalidReason>,
}

pub(crate) fn record_node(call_info: CallInfo, expr: expr::Expr, args: &[Any]) -> Any {
    Context::try_with(call_info, |ctx| ctx.push_node(expr, args))
        .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
}

#[doc(hidden)]
#[macro_export]
/// adds `#[track_caller]`, registers the caller in the thread context and calls
/// [`crate::ir::recording::Context::push_node`] with the provided expression and its arguments
macro_rules! impl_track_caller_fn_any {
    (
        $(
            $vi: vis fn $fn_name: ident $args: tt -> $ret_ty: ty
            => [$($rec_args: expr),*] $ex: expr $(, return $ret: expr)?;
        )*
    ) => {
        $(
            #[allow(clippy::unused_unit)]
            #[track_caller]
            #[allow(missing_docs)]
            $vi fn $fn_name $args -> $ret_ty {
                $crate::frontend::any::record_node($crate::call_info!(), $crate::ir::expr::Expr::from($ex), &[$($rec_args),*])
                $(; return $ret)?
            }
        )*
    };
}

impl Any {
    /// access the dynamic type of `self` if `self` is not invalid
    ///
    /// instances of [`Any`] can be invalid if they are the result of an operation
    /// that did not succeed in the runtime type checker.
    #[track_caller]
    pub fn ty(&self) -> Option<ir::Type> {
        try_ctx_track_caller!(|ctx| self.node.ok().map(|node| ctx.pool()[node].ty().clone())).flatten()
    }

    /// choose a name for the value of `self` in the generated shader code
    ///
    /// this name may be modified or ignored if `shame` cannot guarantee that a
    /// valid shader can be generated with this string present.
    pub fn aka(self, name: &str) -> Self {
        let _ = self.try_set_ident(name.to_string());
        self
    }

    #[track_caller]
    /// tries to set the identifier associated with `self` in the generated shader code.
    /// If `self` already has an identifier associated with it that has a
    /// higher `Priority`, fails and returns that higher `Priority`.
    ///
    /// This function assigns this name with the priority `Priority::UserHint`
    ///
    /// `name` may be modified or ignored if `shame` cannot guarantee that a
    /// valid shader can be generated with this string present.
    pub fn try_set_ident(&self, name: String) -> Result<(), Priority> {
        self.try_set_ident_internal(Ident::Chosen(Priority::UserHint, name))
    }

    #[track_caller]
    pub(crate) fn try_set_ident_internal(&self, ident: Ident) -> Result<(), Priority> {
        Context::try_with(call_info!(), |ctx| -> Result<(), Priority> {
            let mut idents = ctx.pool_mut();
            let Some(node) = self.node() else { return Ok(()) };

            match &mut ctx.pool_mut()[node].ident {
                Some(key) => idents[*key].try_change(ident)?,
                x => *x = Some(idents.push(ident)),
            };
            Ok(())
        })
        .unwrap_or(Ok(()))
    }

    pub(crate) fn suggest_ident(&self, ident: impl Into<String>) -> Any {
        let _ = self.try_set_ident_internal(Ident::Chosen(Priority::Auto, ident.into()));
        *self
    }

    pub(crate) fn node(&self) -> Option<Key<Node>> { self.node.ok() }

    pub(crate) fn inner(&self) -> Result<Key<Node>, InvalidReason> { self.node }

    pub(crate) fn from_parts(key: Result<Key<Node>, InvalidReason>) -> Self { Self { node: key } }

    pub(crate) fn new_invalid(reason: InvalidReason) -> Self { Self::from_parts(Err(reason)) }

    /// create an invalid `Any` which is caused by `user_error`.
    ///
    /// Use this function if you like to use the `shame` error
    /// propagation system rather than using `Result<T, E>`.
    #[track_caller]
    pub fn new_invalid_from_err(user_error: Box<dyn std::error::Error>) -> Self {
        try_ctx_track_caller!(|ctx| { ctx.push_error(EncodingErrorKind::UserDefinedError(user_error)) });
        Self::new_invalid(InvalidReason::ErrorThatWasPushed)
    }

    pub(crate) fn try_eval_floating_point(&self, nodes: &PoolRef<Node>) -> Option<f64> {
        match self.get_if_literal(nodes)? {
            ir::ScalarConstant::F64(x) => Some(x),
            ir::ScalarConstant::F32(x) => Some(x as f64),
            ir::ScalarConstant::F16(x) => Some(f64::from(x)),
            _ => None,
        }
    }

    pub(crate) fn get_if_literal(&self, nodes: &PoolRef<Node>) -> Option<ir::ScalarConstant> {
        match nodes[self.node()?].expr {
            expr::Expr::Literal(ir::expr::Literal(scalar)) => Some(scalar),
            _ => None,
        }
    }
}

/// an argument of an expression is not a valid [`Any`] object
#[derive(Debug, Clone)]
pub struct ArgumentNotAvailable {
    /// the expression used
    pub expr: expr::Expr,
    /// the reasons why each argument was invalid, or None if that argument was valid
    pub arg_availability: Box<[Option<InvalidReason>]>,
}

impl From<ArgumentNotAvailable> for EncodingErrorKind {
    fn from(x: ArgumentNotAvailable) -> Self {
        EncodingErrorKind::NodeRecordingError(ir::recording::NodeRecordingError::ArgumentNotAvailable(x))
    }
}

impl std::error::Error for ArgumentNotAvailable {}

impl std::fmt::Display for ArgumentNotAvailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.arg_availability.iter().filter(|x| x.is_some()).count();
        let is_are = match count {
            1 => "is",
            _ => "are",
        };

        // expression `{0:?}` is invalid because one of its arguments is the result of another invalid expression.
        // This error is most likely the result of another error and will vanish once the root cause is fixed.
        write!(f, "expression `{:?}` is invalid because ", self.expr)?;
        match count {
            1 => write!(f, "one",)?,
            _ => write!(f, "{count}",)?,
        };
        writeln!(f, " of its arguments {is_are} invalid:")?;
        for (izero, reason) in self.arg_availability.iter().enumerate() {
            let i = izero + 1;
            let th = numeral_suffix(i);
            match reason {
                Some(InvalidReason::CreatedWithNoActiveEncoding) => writeln!(
                    f,
                    " - {i}{th} argument is invalid since it was created before the encoding was started."
                )?,
                Some(InvalidReason::ErrorThatWasPushed) => {
                    writeln!(f, " - {i}{th} argument is invalid because of an earlier error.")?
                }
                None => (),
            }
        }
        Ok(())
    }
}

impl ArgumentNotAvailable {
    /// None if all `Any` are available
    pub(crate) fn new(expr: expr::Expr, args: &[Any]) -> Option<Self> {
        match args.iter().all(|any| any.node.is_ok()) {
            true => None,
            false => Some(Self {
                expr,
                arg_availability: args.iter().map(|any| any.node.err()).collect(),
            }),
        }
    }
}
