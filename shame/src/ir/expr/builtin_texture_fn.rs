use std::{fmt::Display, num::NonZeroU32, ops::Not, rc::Rc};

use super::{type_check::*, BuiltinFn, Comp4, ExponentFn, Expr, NumericFn};
use crate::backend::wgsl::WgslErrorKind;
use crate::frontend::any::shared_io::SamplingMethod as SamplingMethodEnum;
use crate::{Language, TextureSampleUsageType};
use crate::{
    call_info,
    common::integer::i4,
    frontend::{
        any::{record_node, Any, ArgumentNotAvailable, InvalidReason},
        encoding::EncodingErrorKind,
        rust_types::{AsAny, GpuType},
        texture::{
            texture_traits::{SamplingMethod, TextureCoords},
            RateUnadjustedMipFn,
        },
    },
    impl_track_caller_fn_any,
    ir::{
        ir_type::{
            AccessMode, AddressSpace, Indirection,
            Len::*,
            Len2,
            ScalarType::{self, *},
            ScalarTypeFp, SizedStruct,
            SizedType::*,
            StoreType::*,
            TextureShape,
            Type::Unit,
        },
        pipeline::{PossibleStages, ShaderStage, StageMask},
        recording::{Context, NodeRecordingError},
        HandleType, SamplesPerPixel, SizedType,
    },
};

use crate::{ir, ir::StoreType, ir::Type, same, sig};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
#[rustfmt::skip]
// functions with an `uv_offset` or `uvw_offset` parameter only support this offset for non-Cube textures
// 
// the 3rd component of `uvw_offset` will be ignored for textures with less than 3 dimensions
pub enum TextureFn {
    TextureDimensions,
    TextureNumLayers,
    TextureNumLevels,
    TextureNumSamples,
    TextureGather {
        /// must be None for depth textures and Some for color textures
        channel: Option<Comp4>, 
        uv_offset: Option<[i4; 2]>,
    },
    TextureGatherCompare { uv_offset: Option<[i4; 2]> },
    TextureLoad(TextureShape, SamplesPerPixel),
    TextureStore(TextureShape),
    TextureSample             { uvw_offset: Option<[i4; 3]> },
    TextureSampleBias         { uvw_offset: Option<[i4; 3]> },
    TextureSampleCompare      { uvw_offset: Option<[i4; 3]> },
    TextureSampleCompareLevel { uvw_offset: Option<[i4; 3]> },
    TextureSampleGrad         { uvw_offset: Option<[i4; 3]> },
    TextureSampleLevel        { uvw_offset: Option<[i4; 3]> },
    TextureSampleBaseClampToEdge,
}

impl From<TextureFn> for Expr {
    fn from(value: TextureFn) -> Self { Expr::BuiltinFn(BuiltinFn::Texture(value)) }
}

impl Display for TextureFn {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let write_offset3 = |f: &mut std::fmt::Formatter<'_>, name: &str, uvw: &Option<[i4; 3]>| -> std::fmt::Result {
            write!(f, "{name}")?;
            if let Some([u, v, w]) = uvw {
                write!(f, "-offset[{u}, {v}, {w}]")?;
            }
            Ok(())
        };

        match self {
            TextureFn::TextureStore(shape) => write!(f, "TextureStore<{shape}>"),
            TextureFn::TextureLoad(shape, spp) => write!(f, "TextureLoad<{shape}, {spp:?}>"),

            TextureFn::TextureDimensions |
            TextureFn::TextureNumLayers |
            TextureFn::TextureNumLevels |
            TextureFn::TextureNumSamples |
            TextureFn::TextureSampleBaseClampToEdge => write!(f, "{self:?}"),

            TextureFn::TextureGather { channel, uv_offset } => {
                write!(f, "TextureGather");
                let channel = channel.map(|c| match c {
                    Comp4::X => "r",
                    Comp4::Y => "g",
                    Comp4::Z => "b",
                    Comp4::W => "a",
                });
                match (channel, uv_offset) {
                    (None, None) => Ok(()),
                    (Some(ch), None) => write!(f, "<{ch}>"),
                    (None, Some([u, v])) => write!(f, "-offset[{u}, {v}]"),
                    (Some(ch), Some([u, v])) => write!(f, "<{ch}>-offset[{u}, {v}]"),
                }
            },
            TextureFn::TextureGatherCompare { uv_offset } => {
                write!(f, "TextureGatherCompare");
                match uv_offset {
                    Some([u, v]) => write!(f, "-offset[{u}, {v}]"),
                    None => Ok(()),
                }
            },
            TextureFn::TextureSample             { uvw_offset } => write_offset3(f, "TextureSample"            , uvw_offset),
            TextureFn::TextureSampleBias         { uvw_offset } => write_offset3(f, "TextureSampleBias"        , uvw_offset),
            TextureFn::TextureSampleCompare      { uvw_offset } => write_offset3(f, "TextureSampleCompare"     , uvw_offset),
            TextureFn::TextureSampleCompareLevel { uvw_offset } => write_offset3(f, "TextureSampleCompareLevel", uvw_offset),
            TextureFn::TextureSampleGrad         { uvw_offset } => write_offset3(f, "TextureSampleGrad"        , uvw_offset),
            TextureFn::TextureSampleLevel        { uvw_offset } => write_offset3(f, "TextureSampleLevel"       , uvw_offset),

        }
    }
}

/// the argument(s) in texture sampling calls that is/are used by the shader to decide the
/// mip map level to sample from.
///
/// type erased version of [`RateUnadjustedMipFn<Coords>`]
pub enum MipLevelAny {
    Implicit,
    BiasedImplicit(Any),
    Explicit(Any),
    FromGradient([Any; 2]),
}

impl<C: TextureCoords> From<RateUnadjustedMipFn<C>> for MipLevelAny {
    fn from(value: RateUnadjustedMipFn<C>) -> MipLevelAny {
        match value {
            RateUnadjustedMipFn::Quad => MipLevelAny::Implicit,
            RateUnadjustedMipFn::QuadBias(bias) => MipLevelAny::BiasedImplicit(bias.as_any()),
            RateUnadjustedMipFn::Level(mip_level) => MipLevelAny::Explicit(mip_level.as_any()),
            RateUnadjustedMipFn::Grad(grad) => MipLevelAny::FromGradient(grad.map(C::to_inner_any)),
        }
    }
}

impl Any {
    #[allow(clippy::single_match)]
    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn texture_sample(
        sampler: Any,
        texture: Any,
        coords: Any,
        array_index: Option<Any>,
        uvw_offset: Option<[i4; 3]>,
        mip_level: MipLevelAny,
    ) -> Any {
        let invalid = Context::try_with(call_info!(), |ctx| {
            let mut invalid = None;
            if let Some(Type::Store(StoreType::Handle(HandleType::SampledTexture(shape, sample_ty, _)))) = texture.ty()
            {
                match ctx.settings().lang {
                    Language::Wgsl => {
                        // 1D textures only support implicit derivative sampling,
                        // so they can only be sampled in the fragment shader
                        if shape == TextureShape::_1D {
                            match (&mip_level, uvw_offset) {
                                (MipLevelAny::Implicit, None) => (), // ok
                                _ => {
                                    invalid = Some(ctx.push_error_get_invalid_any(
                                        WgslErrorKind::Texture1DRequiresImplicitGradNoOffset.into(),
                                    ))
                                }
                            }
                        }

                        // depth textures don't support gradient and biased implicit in wgsl
                        match (&mip_level, sample_ty) {
                            (
                                MipLevelAny::BiasedImplicit { .. } | MipLevelAny::FromGradient(..),
                                TextureSampleUsageType::Depth,
                            ) => {
                                invalid = Some(ctx.push_error_get_invalid_any(
                                    WgslErrorKind::DepthTexturesDontSupportBiasedOrExplicitGradientSampling.into(),
                                ))
                            }
                            _ => (),
                        }
                    }
                }
            }
            invalid
        });

        if let Some(invalid) = invalid.flatten() {
            return invalid;
        }

        match mip_level {
            MipLevelAny::Implicit => {
                let expr = Expr::from(TextureFn::TextureSample { uvw_offset });
                match array_index {
                    Some(i) => record_node(call_info!(), expr, &[texture, sampler, coords, i]),
                    None => record_node(call_info!(), expr, &[texture, sampler, coords]),
                }
            }
            MipLevelAny::BiasedImplicit(bias) => {
                let expr = Expr::from(TextureFn::TextureSampleBias { uvw_offset });
                match array_index {
                    Some(i) => record_node(call_info!(), expr, &[texture, sampler, coords, i, bias]),
                    None => record_node(call_info!(), expr, &[texture, sampler, coords, bias]),
                }
            }
            MipLevelAny::Explicit(mip_level) => {
                let expr = Expr::from(TextureFn::TextureSampleLevel { uvw_offset });
                match array_index {
                    Some(i) => record_node(call_info!(), expr, &[texture, sampler, coords, i, mip_level]),
                    None => record_node(call_info!(), expr, &[texture, sampler, coords, mip_level]),
                }
            }
            MipLevelAny::FromGradient([ddx, ddy]) => {
                let expr = Expr::from(TextureFn::TextureSampleGrad { uvw_offset });
                match array_index {
                    Some(i) => record_node(call_info!(), expr, &[texture, sampler, coords, i, ddx, ddy]),
                    None => record_node(call_info!(), expr, &[texture, sampler, coords, ddx, ddy]),
                }
            }
        }
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn texture_sample_compare_level_0(
        sampler: Any,
        texture: Any,
        coords: Any,
        array_index: Option<Any>,
        depth_ref: Any,
        uvw_offset: Option<[i4; 3]>,
    ) -> Any {
        // as for why this defaults to the TextureSampleCompareLevel version
        // see discussion https://github.com/gpuweb/gpuweb/issues/1319
        let expr = Expr::from(TextureFn::TextureSampleCompareLevel { uvw_offset });
        match array_index {
            Some(i) => record_node(call_info!(), expr, &[texture, sampler, coords, i, depth_ref]),
            None => record_node(call_info!(), expr, &[texture, sampler, coords, depth_ref]),
        }
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn texture_gather(
        &self,
        sampler: Any,
        coords: Any,
        array_index: Option<Any>,
        channel: Option<Comp4>, // None for depth
        uv_offset: Option<[i4; 2]>,
    ) -> Any {
        let expr = Expr::from(TextureFn::TextureGather { channel, uv_offset });
        match array_index {
            Some(i) => record_node(call_info!(), expr, &[*self, sampler, coords, i]),
            None => record_node(call_info!(), expr, &[*self, sampler, coords]),
        }
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn texture_gather_compare(
        &self,
        sampler: Any,
        coords: Any,
        depth_ref: Any,
        array_index: Option<Any>,
        uv_offset: Option<[i4; 2]>,
    ) -> Any {
        let expr = Expr::from(TextureFn::TextureGatherCompare { uv_offset });
        match array_index {
            Some(i) => record_node(call_info!(), expr, &[*self, sampler, coords, i, depth_ref]),
            None => record_node(call_info!(), expr, &[*self, sampler, coords, depth_ref]),
        }
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn texture_load_multisample(&self, shape: TextureShape, coords: Any, sample_index: Any) -> Any {
        record_node(
            call_info!(),
            TextureFn::TextureLoad(shape, SamplesPerPixel::Multi).into(),
            &[*self, coords, sample_index],
        )
    }

    /// load texel data from a single-sampled texture, (use [`Any::texture_load_multisample`] for multi sampled textures)
    /// the `array_index` is the required for texture-arrays. For all other texture shapes this argument is ignored.
    ///
    /// see https://www.w3.org/TR/WGSL/#textureload
    #[track_caller]
    #[doc(hidden)] // runtime api
    pub fn texture_load(
        &self,
        shape: TextureShape,
        coords: Any,
        mip_level: Option<Any>, // None for storage
        array_index: Option<Any>,
    ) -> Any {
        let call_info = call_info!();
        let expr = Expr::from(TextureFn::TextureLoad(shape, ir::SamplesPerPixel::Single));
        (|| -> Option<_> {
            use TextureShape as TS;
            let expr = expr.clone();
            Some(match shape {
                TS::Cube | TS::_1D | TS::_2D | TS::_3D => match mip_level {
                    Some(mip_level) => record_node(call_info, expr, &[*self, coords, mip_level]),
                    None => record_node(call_info, expr, &[*self, coords]), // storage
                },
                TS::CubeArray(_) | TS::_2DArray(_) => match mip_level {
                    Some(mip_level) => record_node(call_info, expr, &[*self, coords, array_index?, mip_level]),
                    None => record_node(call_info, expr, &[*self, coords, array_index?]), // storage
                },
            })
        })()
        .unwrap_or_else(|| {
            Context::try_with(call_info, |ctx| {
                ctx.push_error_get_invalid_any(NodeRecordingError::NoOverloadFound(expr).into())
            })
            .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
        })
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn texture_store(&self, shape: TextureShape, coords: Any, array_index: Option<Any>, value: Any) {
        let expr = Expr::from(TextureFn::TextureStore(shape));
        match array_index {
            Some(i) => record_node(call_info!(), expr, &[*self, coords, i, value]),
            None => record_node(call_info!(), expr, &[*self, coords, value]),
        };
    }

    impl_track_caller_fn_any! {
        pub fn texture_num_layers (&self) -> Any => [*self] TextureFn::TextureNumLayers;
        pub fn texture_num_levels (&self) -> Any => [*self] TextureFn::TextureNumLevels;
        pub fn texture_num_samples(&self) -> Any => [*self] TextureFn::TextureNumSamples;
        pub fn texture_sample_base_clamp_to_edge(&self, sampler: Any, uv: Any) -> Any => [*self, sampler, uv] TextureFn::TextureSampleBaseClampToEdge;
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn texture_dimensions(&self, mip_level: Option<Any>) -> Any {
        let expr = Expr::from(TextureFn::TextureDimensions);
        match mip_level {
            Some(lv) => record_node(call_info!(), expr, &[*self, lv]),
            None => record_node(call_info!(), expr, &[*self]),
        }
    }
}

impl TypeCheck for TextureFn {
    #[rustfmt::skip]
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        use ir::HandleType::*;
        use ir::HandleType;
        use TextureShape::*;
        use ir::SamplesPerPixel;
        use ir::TextureSampleUsageType::*;
        use ir::ChannelFormatShaderType;

        let error = |comment: String| {
            NoMatchingSignature::empty_with_name_and_comment(format!("{self:?}").into(), comment, args)
        };

        // check that, if sampling happens, the samplers used is compatible with the textures
        {
            use crate::frontend::any::shared_io::SamplingMethod as SBT;

            let samplers = args.iter().filter_map(|arg| match arg {
                Type::Store(StoreType::Handle(HandleType::Sampler(s))) => Some(s),
                _ => None,
            });
            let textures = args.iter().filter_map(|arg| match arg {
                Type::Store(StoreType::Handle(HandleType::SampledTexture(shape, sample_ty, spp))) => Some((shape, sample_ty, spp)),
                _ => None,
            });
            for sampler_ty in samplers {
                for (_shape, sample_ty, _spp) in textures.clone() {
                    if !sample_ty.is_compatible_with_sampler(*sampler_ty) {
                        return Err(error(format!("a {sample_ty}-sampled texture is incompatible with a {sampler_ty} sampler")))
                    }
                }
            }
        };

        (match self {
            TextureFn::TextureDimensions => {
                sig!(
                    { fmt: SigFormatting::RemoveAsterisksAndClone, },
                    // no miplevel
                    [Handle(SampledTexture(d, _, _) | StorageTexture(d, _, _))] if *d == _1D => Vector(X1, U32),
                    [Handle(SampledTexture(d, _, _) | StorageTexture(d, _, _))] if matches!(*d, _2D | _2DArray(_) | Cube | CubeArray(_)) => Vector(X2, U32),
                    [Handle(SampledTexture(d, _, _) | StorageTexture(d, _, _))] if *d == _3D => Vector(X3, U32),

                    // 2nd arg is miplevel
                    [Handle(SampledTexture(d, _, _)), Sized(Vector(X1, I32 | U32))] if *d == _1D => Vector(X1, U32),
                    [Handle(SampledTexture(d, _, _)), Sized(Vector(X1, I32 | U32))] if matches!(*d, _2D | _2DArray(_) | Cube | CubeArray(_)) => Vector(X2, U32),
                    [Handle(SampledTexture(d, _, _)), Sized(Vector(X1, I32 | U32))] if *d == _3D => Vector(X3, U32),
                )(self, args)
                // WGSL spec: If level is outside the range [0, textureNumLevels(t)) then an indeterminate value for the return type may be returned.
                // TODO(release) consider making the array size part of the shame type, so out of bounds access can be caught
                // UPDATE: the array size was now added to the shame type (because wgpu requires it in the bindgroup layout binding entry), nothing more has been done though.
            }
            TextureFn::TextureGather {channel, uv_offset} => {
                use ir::TextureSampleUsageType::*;
                match args {
                    [Type::Store(Handle(SampledTexture(shape, st, spp))), ..] => {
                        // offsets are only allowed on non-cube textures
                        let offset_check = match (uv_offset, shape) {
                            (Some(offset), Cube | CubeArray(_) | _3D) => Err(error("offset must be `None` for cube and cube-array textures".into())),
                            _ => Ok(()),
                        };

                        let spp_check = match spp {
                            SamplesPerPixel::Single => Ok(()),
                            SamplesPerPixel::Multi => Err(error("gather requires single sample-per-pixel textures".into())),
                        };

                        // check whether the component matches the sample type and texture channel count
                        let component_check = match (st, channel) {
                            (Depth, Some(_)) => Err(error("component must be `None` for depth textures".into())),
                            (Depth, None) => Ok(()),
                            (FilterableFloat{..} | Nearest {..}, None) => Err(error("component cannot be `None` for non-depth textures".into())),
                            (FilterableFloat{..} | Nearest {..}, Some(component)) => {
                                match component.is_contained_in(st.len()) {
                                    true => Ok(()),
                                    false => Err(error(format!("cannot gather {component} component of a {}-channel texture", st.len())))
                                }
                            }
                        };

                        // actual type signatures
                        let inferred_type = sig!(
                            [Handle(SampledTexture(         _2D, sample_ty, _)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)) ] => Vector(X4, ScalarType::from(sample_ty.channel_ty())),
                            [Handle(SampledTexture(        Cube, sample_ty, _)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32))] => Vector(X4, ScalarType::from(sample_ty.channel_ty())),
                            [Handle(SampledTexture( _2DArray(_), sample_ty, _)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32))] => Vector(X4, ScalarType::from(sample_ty.channel_ty())),
                            [Handle(SampledTexture(CubeArray(_), sample_ty, _)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32))] => Vector(X4, ScalarType::from(sample_ty.channel_ty())),
                        )(self, args);

                        offset_check.and(spp_check).and(component_check).and(inferred_type)
                    }
                    _ => Err(error("first argument must be a handle to a sampled texture".into()))
                }
            },
            TextureFn::TextureGatherCompare { uv_offset } => {
                use ir::TextureSampleUsageType::*;
                use SamplingMethodEnum::*;

                match args {
                    [Type::Store(Handle(SampledTexture(shape, Depth, SamplesPerPixel::Single))), ..] => {
                        // offsets are only allowed on non-cube textures
                        let offset_check = match (uv_offset, shape) {
                            (Some(offset), Cube | CubeArray(_) | _3D) => Err(error("offset must be `None` for cube and cube-array textures".into())),
                            _ => Ok(()),
                        };

                        let inferred_type = sig!(
                            [Handle(SampledTexture(         _2D, Depth, _)), Handle(Sampler(Comparison)), /*uv:*/ Sized(Vector(X2, F32)),                                                /*depth_ref:*/ Sized(Vector(X1, F32))] => Vector(X4, F32),
                            [Handle(SampledTexture(        Cube, Depth, _)), Handle(Sampler(Comparison)), /*uv:*/ Sized(Vector(X3, F32)),                                                /*depth_ref:*/ Sized(Vector(X1, F32))] => Vector(X4, F32),
                            [Handle(SampledTexture( _2DArray(_), Depth, _)), Handle(Sampler(Comparison)), /*uv:*/ Sized(Vector(X2, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*depth_ref:*/ Sized(Vector(X1, F32))] => Vector(X4, F32),
                            [Handle(SampledTexture(CubeArray(_), Depth, _)), Handle(Sampler(Comparison)), /*uv:*/ Sized(Vector(X3, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*depth_ref:*/ Sized(Vector(X1, F32))] => Vector(X4, F32),
                        )(self, args);

                        offset_check.and(inferred_type)
                    }
                    _ => Err(error("first argument must be a handle to a single sample per pixel depth texture".into()))
                }
            },
            TextureFn::TextureLoad(shape, spp) => {
                use SamplesPerPixel::*;
                use AccessMode::*;
                // single-spp sampled texture =>    level arg: I32 | U32
                // multi-spp sampled texture  => sample_index: I32 | U32
                //                        all =>       coords: Vector(shape-based, I32 | U32)
                //              array texture =>  array_index: I32 | U32
                match spp {
                    Single => match shape {
                        _1D => sig!(
                            [Handle(SampledTexture(_1D, st, Single)), /*uv:*/ Sized(Vector(X1, U32 | I32)), /*level:*/ Sized(Vector(X1, U32 | I32))] => st.type_in_wgsl(),
                            [Handle(StorageTexture(_1D, fmt, Read | ReadWrite)), /*uv:*/ Sized(Vector(X1, U32 | I32))] if fmt.is_sampleable() => fmt.sample_type_in_wgsl().expect("storage formats are always sampleable"),
                        )(self, args),
                        _2D => sig!(
                            [Handle(SampledTexture(_2D, st, Single)), /*uv:*/ Sized(Vector(X2, U32 | I32)), /*level:*/ Sized(Vector(X1, U32 | I32))] => st.type_in_wgsl(),
                            [Handle(StorageTexture(_2D, fmt, Read | ReadWrite)), /*uv:*/ Sized(Vector(X2, U32 | I32))] if fmt.is_sampleable() => fmt.sample_type_in_wgsl().expect("storage formats are always sampleable"),        
                        )(self, args),
                        _2DArray(n) => sig!(
                            [Handle(SampledTexture(_2DArray(n), st, Single)), /*uv:*/ Sized(Vector(X2, U32 | I32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*level:*/ Sized(Vector(X1, U32 | I32))] => st.type_in_wgsl(),
                            [Handle(StorageTexture(_2DArray(n), fmt, Read | ReadWrite)), /*uv:*/ Sized(Vector(X2, U32 | I32)), /*array_index:*/ Sized(Vector(X1, U32 | I32))] if fmt.is_sampleable() => fmt.sample_type_in_wgsl().expect("storage formats are always sampleable"),        
                        )(self, args),
                        _3D => sig!(
                            [Handle(SampledTexture(_3D, st, Single)), /*uv:*/ Sized(Vector(X3, U32 | I32)), /*level:*/ Sized(Vector(X1, U32 | I32))] => st.type_in_wgsl(),
                            [Handle(StorageTexture(_3D, fmt, Read | ReadWrite)), /*uv:*/ Sized(Vector(X3, U32 | I32))] if fmt.is_sampleable() => fmt.sample_type_in_wgsl().expect("storage formats are always sampleable"),        
                        )(self, args),
                        Cube | CubeArray(_) => Err(error("unsupported texture shape".into())),
                    },
                    Multi => sig!(
                        [Handle(SampledTexture(_2D, st,  Multi)), /*uv:*/ Sized(Vector(X2, U32 | I32)), /*sample_index:*/ Sized(Vector(X1, U32 | I32))] => st.type_in_wgsl(),
                    )(self, args)
                }
            },
            TextureFn::TextureStore(shape) => {
                use AccessMode::*;
                match shape {
                    _1D => sig!(
                        [Handle(StorageTexture(_1D, fmt, Write | ReadWrite)), /*uv:*/ Sized(Vector(X1, U32 | I32)), Sized(Vector(X4, s))] if Some(*s) == fmt.scalar_type_in_shader() => Unit,
                    )(self, args),
                    _2D => sig!(
                        [Handle(StorageTexture(_2D, fmt, Write | ReadWrite)), /*uv:*/ Sized(Vector(X2, U32 | I32)), Sized(Vector(X4, s))] if Some(*s) == fmt.scalar_type_in_shader() => Unit,
                    )(self, args),
                    _2DArray(n) => sig!(
                        [Handle(StorageTexture(_2DArray(n), fmt, Write | ReadWrite)), /*uv:*/ Sized(Vector(X2, U32 | I32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), Sized(Vector(X4, s))] if Some(*s) == fmt.scalar_type_in_shader() => Unit,
                    )(self, args),
                    _3D => sig!(
                        [Handle(StorageTexture(     _3D, fmt, Write | ReadWrite)), /*uv:*/ Sized(Vector(X3, U32 | I32)), Sized(Vector(X4, s))] if Some(*s) == fmt.scalar_type_in_shader() => Unit,
                    )(self, args),
                    Cube | CubeArray(_) => Err(error("unsupported texture shape".into())),
                }
            },
            TextureFn::TextureNumLayers => sig!(
                [Handle(SampledTexture(_2DArray(_) | CubeArray(_), _, _))] => U32,
                [Handle(StorageTexture(_2DArray(_), _, _))] => U32,
            )(self, args),
            TextureFn::TextureNumLevels => sig!(
                [Handle(SampledTexture(..))] => U32,
            )(self, args),
            TextureFn::TextureNumSamples => sig!(
                [Handle(SampledTexture(_, _, SamplesPerPixel::Multi))] => U32,
            )(self, args),
            TextureFn::TextureSample { uvw_offset } => {
                use SamplesPerPixel::*;
                match args {
                    [Type::Store(Handle(SampledTexture(shape, st, Single))), ..] => {
                        let offset_check = match (uvw_offset, shape) {
                            (Some(offset), Cube | CubeArray(_)) => Err(error("offset must be `None` for cube and cube-array textures".into())),
                            _ => Ok(()),
                        };

                        let inferred_ty = sig!(
                            [Handle(SampledTexture(         _1D, sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X1, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture(         _2D, sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture( _2DArray(_), sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture(  _3D | Cube, sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture(CubeArray(_), sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32))] => sample_ty.type_in_wgsl(),
                        )(self, args);

                        offset_check.and(inferred_ty)
                    }
                    _ => Err(error("first argument must be a sampleable texture with a single sample per pixel".into()))
                }
            },
            TextureFn::TextureSampleBias { uvw_offset } => {
                use SamplesPerPixel::*;
                match args {
                    [Type::Store(Handle(SampledTexture(shape, st, Single))), ..] if *st != Depth => {
                        let offset_check = match (uvw_offset, shape) {
                            (Some(offset), Cube | CubeArray(_)) => Err(error("offset must be `None` for cube and cube-array textures".into())),
                            _ => Ok(()),
                        };

                        let inferred_ty = sig!(
                            [Handle(SampledTexture(          _2D, sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*bias:*/ Sized(Vector(X1, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture(  _2DArray(_), sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*bias:*/ Sized(Vector(X1, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture(   _3D | Cube, sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*bias:*/ Sized(Vector(X1, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture( CubeArray(_), sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*bias:*/ Sized(Vector(X1, F32))] => sample_ty.type_in_wgsl(),
                        )(self, args);

                        offset_check.and(inferred_ty)
                    }
                    _ => Err(error("first argument must be a sampleable non-depth texture with a single sample per pixel".into()))
                }
            },
            TextureFn::TextureSampleCompare { uvw_offset } |
            TextureFn::TextureSampleCompareLevel { uvw_offset }=> {
                use SamplesPerPixel::*;
                match args {
                    [Type::Store(Handle(SampledTexture(shape, Depth, Single))), ..] => {
                        let offset_check = match (uvw_offset, shape) {
                            (Some(offset), Cube | CubeArray(_)) => Err(error("offset must be `None` for cube and cube-array textures".into())),
                            _ => Ok(()),
                        };
                        let inferred_ty = sig!(
                            [Handle(SampledTexture(          _2D, Depth, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*depth_ref:*/ Sized(Vector(X1, F32))] => F32,
                            [Handle(SampledTexture(  _2DArray(_), Depth, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*depth_ref:*/ Sized(Vector(X1, F32))] => F32,
                            [Handle(SampledTexture(         Cube, Depth, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*depth_ref:*/ Sized(Vector(X1, F32))] => F32,
                            [Handle(SampledTexture( CubeArray(_), Depth, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*depth_ref:*/ Sized(Vector(X1, F32))] => F32,
                        )(self, args);

                        offset_check.and(inferred_ty)
                    }
                    _ => Err(error("first argument must be a sampleable depth texture with a single sample per pixel".into()))
                }
            },
            TextureFn::TextureSampleGrad { uvw_offset } => {
                use SamplesPerPixel::*;
                match args {
                    [Type::Store(Handle(SampledTexture(shape, st, Single))), ..] if *st != Depth => {
                        let offset_check = match (uvw_offset, shape) {
                            (Some(offset), Cube | CubeArray(_)) => Err(error("offset must be `None` for cube and cube-array textures".into())),
                            _ => Ok(()),
                        };

                        let inferred_ty = sig!(
                            [Handle(SampledTexture(          _2D, sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)),                                              /*ddx:*/ Sized(Vector(X2, F32)), /*ddy:*/ Sized(Vector(X2, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture(  _2DArray(_), sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*ddx:*/ Sized(Vector(X2, F32)), /*ddy:*/ Sized(Vector(X2, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture(   _3D | Cube, sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)),                                              /*ddx:*/ Sized(Vector(X3, F32)), /*ddy:*/ Sized(Vector(X3, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture( CubeArray(_), sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*ddx:*/ Sized(Vector(X3, F32)), /*ddy:*/ Sized(Vector(X3, F32))] => sample_ty.type_in_wgsl(),
                        )(self, args);

                        offset_check.and(inferred_ty)
                    }
                    _ => Err(error("first argument must be a sampleable non-depth texture with a single sample per pixel".into()))
                }
            },
            TextureFn::TextureSampleLevel { uvw_offset } => {
                use SamplesPerPixel::*;
                match args {
                    [Type::Store(Handle(SampledTexture(shape, st, Single))), ..] => {
                        let offset_check = match (uvw_offset, shape) {
                            (Some(offset), Cube | CubeArray(_)) => Err(error("offset must be `None` for cube and cube-array textures".into())),
                            _ => Ok(()),
                        };

                        let inferred_ty = sig!(
                            [Handle(SampledTexture(          _2D, sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*level:*/ Sized(Vector(X1, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture(  _2DArray(_), sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*level:*/ Sized(Vector(X1, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture(   _3D | Cube, sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*level:*/ Sized(Vector(X1, F32))] => sample_ty.type_in_wgsl(),
                            [Handle(SampledTexture( CubeArray(_), sample_ty, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*level:*/ Sized(Vector(X1, F32))] => sample_ty.type_in_wgsl(),

                            [Handle(SampledTexture(          _2D, Depth, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*level:*/ Sized(Vector(X1, U32 | I32))] => SizedType::from(Depth),
                            [Handle(SampledTexture(  _2DArray(_), Depth, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*level:*/ Sized(Vector(X1, U32 | I32))] => SizedType::from(Depth),
                            [Handle(SampledTexture(   _3D | Cube, Depth, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*level:*/ Sized(Vector(X1, U32 | I32))] => SizedType::from(Depth),
                            [Handle(SampledTexture( CubeArray(_), Depth, Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X3, F32)), /*array_index:*/ Sized(Vector(X1, U32 | I32)), /*level:*/ Sized(Vector(X1, U32 | I32))] => SizedType::from(Depth),
                        )(self, args);

                        offset_check.and(inferred_ty)
                    }
                    _ => Err(error("first argument must be a sampleable texture with a single sample per pixel".into()))
                }
            },
            TextureFn::TextureSampleBaseClampToEdge => sig!(
                [Handle(SampledTexture(_2D, st, SamplesPerPixel::Single)), Handle(Sampler(_)), /*uv:*/ Sized(Vector(X2, F32))] if st.shader_scalar_ty() == F32 => st.type_in_wgsl(),
            )(self, args),
        })
    }
}
