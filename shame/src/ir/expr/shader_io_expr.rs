use std::fmt::Display;
use std::num::NonZeroU32;
use std::rc::Rc;

use super::{Comp4, Expr, NoMatchingSignature, TypeCheck};
use crate::frontend::any::render_io::{Attrib, FragmentSampleMethod, Location};
use crate::frontend::any::shared_io::BindPath;
use crate::frontend::any::Any;
use crate::frontend::encoding::fill::{Fill, PickVertex};
use crate::frontend::encoding::EncodingErrorKind;
use crate::frontend::error::InternalError;
use crate::frontend::texture::texture_traits::StorageTextureFormat;
use crate::ir::expr::type_check::SigFormatting;
use crate::ir::ir_type::{TextureAspect, TextureSampleUsageType};
use crate::ir::pipeline::{PipelineError, PossibleStages, ShaderStage, StageMask};
use crate::ir::recording::{Context, NodeRecordingError};
use crate::ir::Len::*;
use crate::ir::ScalarType::*;
use crate::ir::SizedType::*;
use crate::ir::StoreType::*;
use crate::ir::Type::Unit;
use crate::ir::{self, ChannelFormatShaderType, Len, ScalarConstant, ScalarType, SizedType, StoreType};
use crate::{call_info, impl_track_caller_fn_any, sig};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderIo {
    Builtin(BuiltinShaderIo),
    Interpolate(Location),
    GetInterpolated(Location),
    GetVertexInput(Location),
    WriteToColorTarget { slot: u32 },
}

impl Display for ShaderIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShaderIo::Builtin(x) => write!(f, "{x}"),
            ShaderIo::Interpolate(location) => write!(f, "set interpolator[{}]", location),
            ShaderIo::GetInterpolated(location) => write!(f, "get interpolator[{}]", location),
            ShaderIo::GetVertexInput(location) => write!(f, "get vertex_in[{}]", location),
            ShaderIo::WriteToColorTarget { slot } => write!(f, "set color_out[{}]", slot),
        }
    }
}

impl From<ShaderIo> for Expr {
    fn from(value: ShaderIo) -> Self { Expr::ShaderIo(value) }
}

impl TypeCheck for ShaderIo {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        Context::with(call_info!(), |ctx| match self.infer_shader_io_type(ctx, args) {
            Ok(result) => result,
            Err(e) => {
                ctx.push_error(e);
                Err(NoMatchingSignature::empty_with_name(format!("{self:?}").into(), args))
            }
        })
    }
}

impl ShaderIo {
    /// returns either a regular `TypeCheck` result, or
    /// an `EncodingErrorKind` if the required resources were not found in the
    /// render pipeline recording (invalid attrib locations, color target slots etc.)
    fn infer_shader_io_type(
        &self,
        ctx: &Context,
        args: &[ir::Type],
    ) -> Result<Result<ir::Type, NoMatchingSignature>, EncodingErrorKind> {
        match self {
            ShaderIo::Builtin(x) => Ok(x.infer_type(args)),
            ShaderIo::Interpolate(loc) => {
                let render_pipeline = ctx.render_pipeline();
                let interp = render_pipeline.find_interpolator(*loc)?;
                let interpolated_type = interp.get_sized_type();
                let result = sig!(
                    { name: ShaderIo::Interpolate(interpolated_type), force_in_type: SizedType, },
                    [t] if *t == interpolated_type => Unit
                )(self, args);
                Ok(result)
            }
            ShaderIo::GetInterpolated(loc) => {
                let render_pipeline = ctx.render_pipeline();
                let interp = render_pipeline.find_interpolator(*loc)?;
                let interpolated_type = interp.get_sized_type();
                let result = sig!(
                    { name: ShaderIo::GetInterpolated(interpolated_type), force_in_type: SizedType, },
                    [] => interpolated_type
                )(self, args);
                Ok(result)
            }
            ShaderIo::GetVertexInput(loc) => {
                let render_pipeline = ctx.render_pipeline();
                let Attrib { format, .. } = render_pipeline.find_vertex_attrib(*loc)?;
                Ok(sig!(
                    [] => format.type_in_shader()
                )(self, args))
            }
            ShaderIo::WriteToColorTarget { slot } => {
                use TextureSampleUsageType as TST;
                let render_pipeline = ctx.render_pipeline();
                let color_target = render_pipeline.find_color_target(*slot)?;
                Ok(match color_target.format.sample_type() {
                    Some(sample_type) => match sample_type {
                        TST::FilterableFloat {
                            len: num_color_target_components,
                        } |
                        TST::Nearest {
                            len: num_color_target_components,
                            channel_type: ChannelFormatShaderType::F32,
                        } => sig!([Vector(n, F32)] if *n == num_color_target_components => Unit)(self, args),
                        TST::Nearest {
                            len: num_color_target_components,
                            channel_type: ChannelFormatShaderType::I32,
                        } => sig!([Vector(n, I32)] if *n == num_color_target_components => Unit)(self, args),
                        TST::Nearest {
                            len: num_color_target_components,
                            channel_type: ChannelFormatShaderType::U32,
                        } => sig!([Vector(n, U32)] if *n == num_color_target_components => Unit)(self, args),
                        TST::Nearest {
                            len: num_color_target_components,
                            channel_type: color_target_channel_type,
                        } => {
                            sig!([Vector(n, t)] if *n == num_color_target_components && *t == ScalarType::from(color_target_channel_type) => Unit)(
                                self, args,
                            )
                        }
                        TST::Depth => Err(InternalError::new(
                            true,
                            format!(
                                "attempting to color write to depth-aspect texture {slot} with format {:?}",
                                color_target.format
                            ),
                        ))?,
                    },

                    None => Err(InternalError::new(
                        true,
                        format!(
                            "no sample type for color target at {slot} with format {:?}",
                            color_target.format
                        ),
                    ))?,
                })
            }
        }
    }
}

impl ShaderIo {
    pub fn may_change_execution_state(&self) -> bool {
        match self {
            ShaderIo::Builtin(x) => x.may_change_execution_state(),
            ShaderIo::GetInterpolated(_) => false,
            ShaderIo::GetVertexInput(_) => false,
            ShaderIo::Interpolate(_) => true, // unsure, chose `true` just to be on the safe side
            ShaderIo::WriteToColorTarget { .. } => true, // unsure, chose `true` just to be on the safe side
        }
    }

    #[rustfmt::skip]
    pub fn possible_stages(&self) -> PossibleStages {
        let none = StageMask::empty();
        let vert = StageMask::vert();
        let mesh = StageMask::mesh();
        let frag = StageMask::frag();
        let comp = StageMask::comp();
        let (must_appear, must_in, can_in, only_once) = match self {
            ShaderIo::Builtin(builtin_io) => return builtin_io.possible_stages(),
            ShaderIo::Interpolate(_) => (true, none, vert | mesh, true), // must appear, so that the corresponding GetInterpolated can appear.
            ShaderIo::GetInterpolated(_) => (false, none, frag, true),
            ShaderIo::GetVertexInput (_) => (false, none, vert, true),
            ShaderIo::WriteToColorTarget { slot } => (true, frag, frag, true),
        };
        PossibleStages::new(must_appear, must_in, can_in, only_once)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuiltinShaderIo {
    Get(BuiltinShaderIn),
    Set(BuiltinShaderOut),
}

impl BuiltinShaderIo {
    pub fn possible_stages(&self) -> PossibleStages {
        let none = StageMask::empty();
        let vert = StageMask::vert();
        let mesh = StageMask::mesh();
        let frag = StageMask::frag();
        let comp = StageMask::comp();
        let (must_appear, must_in, can_in, only_once) = match self {
            BuiltinShaderIo::Get(get) => match get {
                BuiltinShaderIn::VertexIndex | BuiltinShaderIn::InstanceIndex => (false, none, vert, true),

                BuiltinShaderIn::Position |
                BuiltinShaderIn::FrontFacing |
                BuiltinShaderIn::SampleIndex |
                BuiltinShaderIn::SampleMask => (false, none, frag, true),

                BuiltinShaderIn::LocalInvocationIndex |
                BuiltinShaderIn::LocalInvocationId |
                BuiltinShaderIn::GlobalInvocationId |
                BuiltinShaderIn::WorkgroupId |
                BuiltinShaderIn::NumWorkgroups => (false, none, comp, true),

                BuiltinShaderIn::SubgroupInvocationId => (false, none, frag | comp, true),
                BuiltinShaderIn::SubgroupSize => (false, none, frag | comp, false),
            },
            BuiltinShaderIo::Set(set) => match set {
                BuiltinShaderOut::Position => (true, none, vert | mesh, true),
                BuiltinShaderOut::ClipDistances { count } => (true, none, vert | mesh, true),
                BuiltinShaderOut::FragDepth => (true, frag, frag, true),
                BuiltinShaderOut::SampleMask => (true, frag, frag, true),
            },
        };
        PossibleStages::new(must_appear, must_in, can_in, only_once)
    }
}

impl Display for BuiltinShaderIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuiltinShaderIo::Get(x) => write!(f, "get {x:?}"),
            BuiltinShaderIo::Set(x) => write!(f, "set {x:?}"),
        }
    }
}

impl BuiltinShaderIo {
    fn may_change_execution_state(&self) -> bool {
        match self {
            BuiltinShaderIo::Get(_) => false,
            BuiltinShaderIo::Set(_) => true, // unsure, might be pessimistic
        }
    }
}

impl TypeCheck for BuiltinShaderIo {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        match self {
            BuiltinShaderIo::Get(x) => x.infer_type(args),
            BuiltinShaderIo::Set(x) => x.infer_type(args),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuiltinShaderIn {
    // vertex
    VertexIndex,
    InstanceIndex,
    // fragment
    // (no test case yet)
    Position,
    // (no test case yet)
    FrontFacing,
    // (no test case yet)
    SampleIndex,
    // (no test case yet)
    SampleMask,
    // compute
    // (no test case yet)
    LocalInvocationIndex,
    // (no test case yet)
    LocalInvocationId,
    // (no test case yet)
    GlobalInvocationId,
    // (no test case yet)
    WorkgroupId,
    // (no test case yet)
    NumWorkgroups,
    // (no test case yet)
    SubgroupInvocationId,
    SubgroupSize,
}

impl From<BuiltinShaderIn> for Expr {
    fn from(value: BuiltinShaderIn) -> Self { Expr::ShaderIo(ShaderIo::Builtin(BuiltinShaderIo::Get(value))) }
}

impl TypeCheck for BuiltinShaderIn {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        let bool = sig!([] => Bool);
        let u32 = sig!([] => U32);
        let vec3u = sig!([] => Vector(X3, U32));
        let vec4f = sig!([] => Vector(X4, F32));
        (match self {
            BuiltinShaderIn::VertexIndex | BuiltinShaderIn::InstanceIndex => u32,
            BuiltinShaderIn::Position => vec4f,
            BuiltinShaderIn::FrontFacing => bool,
            BuiltinShaderIn::SampleIndex | BuiltinShaderIn::SampleMask => u32,
            BuiltinShaderIn::LocalInvocationIndex => u32,
            BuiltinShaderIn::LocalInvocationId => vec3u,
            BuiltinShaderIn::GlobalInvocationId => vec3u,
            BuiltinShaderIn::WorkgroupId => vec3u,
            BuiltinShaderIn::NumWorkgroups => vec3u,
            BuiltinShaderIn::SubgroupInvocationId => u32,
            BuiltinShaderIn::SubgroupSize => u32,
        })(self, args)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuiltinShaderOut {
    // vertex
    Position,
    ClipDistances { count: NonZeroU32 },
    // fragment
    // (no test case yet)
    FragDepth,
    // (no test case yet)
    SampleMask,
}

impl From<BuiltinShaderOut> for Expr {
    fn from(value: BuiltinShaderOut) -> Self { Expr::ShaderIo(ShaderIo::Builtin(BuiltinShaderIo::Set(value))) }
}

impl TypeCheck for BuiltinShaderOut {
    fn infer_type(&self, args: &[ir::Type]) -> Result<ir::Type, NoMatchingSignature> {
        (match self {
            BuiltinShaderOut::Position => sig!([Vector(X4, F32)] => Unit),
            BuiltinShaderOut::ClipDistances { count } => {
                let distance_count = count;
                return sig!(
                    { fmt: SigFormatting::RemoveAsterisksAndClone, },
                    [Array(f32x1, n)] if n.get() <= 8 && n == distance_count && **f32x1 == Vector(X1, F32) => Unit,
                )(self, args);
            }
            BuiltinShaderOut::FragDepth => sig!([F32] => Unit),
            BuiltinShaderOut::SampleMask => sig!([U32] => Unit),
        })(self, args)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interpolator {
    pub vec_ty: (Len, ScalarType),
    pub method: FragmentSampleMethod,
    pub location: Location,
}

impl Interpolator {
    fn get_sized_type(&self) -> SizedType {
        let (len, stype) = self.vec_ty;
        SizedType::Vector(len, stype)
    }
}

impl Any {
    impl_track_caller_fn_any! {
        // shader inputs can only be instantiated once, therefore they start with "new"

        pub fn new_front_facing() -> Any => [] BuiltinShaderIn::FrontFacing;
        pub fn new_sample_mask() -> Any => [] BuiltinShaderIn::SampleMask;
        pub fn new_sample_id() -> Any => [] BuiltinShaderIn::SampleIndex;
        pub fn new_fragment_position() -> Any => [] BuiltinShaderIn::Position;

        pub fn new_thread_pos_within_dispatch() -> Any => [] BuiltinShaderIn::GlobalInvocationId;
        pub fn new_thread_pos_within_workgroup() -> Any => [] BuiltinShaderIn::LocalInvocationId;
        pub fn new_thread_id_within_workgroup() -> Any => [] BuiltinShaderIn::LocalInvocationIndex;
        pub fn new_workgroup_pos_within_dispatch() -> Any => [] BuiltinShaderIn::WorkgroupId;
        pub fn new_workgroup_grid_size_within_dispatch() -> Any => [] BuiltinShaderIn::NumWorkgroups;
        pub fn new_thread_id_within_wave() -> Any => [] BuiltinShaderIn::SubgroupInvocationId;
        pub fn new_thread_count_within_wave() -> Any => [] BuiltinShaderIn::SubgroupSize;
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn thread_grid_size_within_workgroup(dims: Len) -> Any {
        Context::try_with(call_info!(), |ctx| {
            match &ctx.compute_pipeline().thread_grid_size_within_workgroup.get() {
                Some(([x, y, z], _)) => {
                    let u32_scalar = |c| Any::new_scalar(ScalarConstant::U32(c));
                    let sty = ScalarType::U32;
                    match dims {
                        X1 => Any::new_vec(dims, sty, &[*x].map(u32_scalar)),
                        X2 => Any::new_vec(dims, sty, &[*x, *y].map(u32_scalar)),
                        X3 => Any::new_vec(dims, sty, &[*x, *y, *z].map(u32_scalar)),
                        X4 => ctx.push_error_get_invalid_any(
                            InternalError::new(true, "4D workgroup thread grid size unsupported".into()).into(),
                        ),
                    }
                }
                None => ctx.push_error_get_invalid_any(NodeRecordingError::WorkgroupGridSizeNotAvailable.into()),
            }
        })
        .unwrap_or(Any::new_invalid(
            crate::frontend::any::InvalidReason::CreatedWithNoActiveEncoding,
        ))
    }

    #[track_caller]
    pub(crate) fn shrink_vector(swizzle_len: Len, vec: Any) -> Any {
        use ir::VectorAccess::*;
        use Comp4::*;
        match swizzle_len {
            X1 => vec.swizzle(Swizzle1([X])),
            X2 => vec.swizzle(Swizzle2([X, Y])),
            X3 => vec.swizzle(Swizzle3([X, Y, Z])),
            X4 => vec,
        }
    }
}
