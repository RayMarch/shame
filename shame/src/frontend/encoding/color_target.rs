use std::cell::Cell;
use std::marker::PhantomData;

use render_io::ChannelWrites;

use crate::frontend::any::blend::Blend;
use crate::frontend::any::{Any, InvalidReason};
use crate::frontend::rust_types::len::x4;
use crate::frontend::rust_types::scalar_type::ScalarType;
use crate::frontend::rust_types::vec::IsVec;
use crate::frontend::rust_types::To;
use crate::ir::pipeline::PipelineError;
use crate::ir::recording::Context;
use crate::ir::{Len, TextureFormatId};
use crate::{call_info, f32x1};
use crate::{frontend::any::render_io, ir};

use crate::frontend::texture::texture_traits::{
    Blendable, ColorTargetFormat, Multi, Single, Spp, SupportsSpp, TexelShaderType, TextureFormat,
};

use super::rasterizer::FragmentStage;

/// (no documentation yet)
pub struct ColorTarget<Format: ColorTargetFormat, SPP: Spp = Single>
where
    Format: SupportsSpp<SPP>,
{
    slot: u32,
    phantom: PhantomData<(Format, SPP)>,
}

impl<T: ColorTargetFormat + SupportsSpp<SPP>, SPP: Spp> ColorTarget<T, SPP> {
    fn new(slot: u32) -> Self {
        ColorTarget {
            slot,
            phantom: PhantomData,
        }
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn blend(self, blend: Blend, src_color: impl To<T::TexelShaderType>)
    where
        T: Blendable,
    {
        self.set_explicit(Some(blend), ChannelWrites::default(), None, src_color)
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn set(self, src_color: impl To<T::TexelShaderType>) {
        self.set_explicit(None, ChannelWrites::default(), None, src_color)
    }

    // ## alpha to coverage
    //       `alpha_to_coverage` is only supported for the first (index = 0) target.
    //       the first target also needs to be multisampled and have 4 color components.
    //       Otherwise a recording error is generated.
    /// (no documentation yet)
    #[track_caller]
    pub fn set_explicit(
        self,
        blend: Option<Blend>,
        write_mask: ChannelWrites,
        alpha_to_coverage: Option<&FragmentStage<Multi>>,
        src_color: impl To<T::TexelShaderType>,
    ) {
        self.set_internal(blend, write_mask, alpha_to_coverage.is_some(), src_color);
    }

    #[track_caller]
    pub(crate) fn set_internal(
        self,
        blend: Option<Blend>,
        write_mask: ChannelWrites,
        alpha_to_coverage: bool,
        src_color: impl To<T::TexelShaderType>,
    ) {
        let fmt_id = <T as TextureFormat>::id();

        let len = fmt_id.sample_type().map(|x| x.len());

        Context::try_with(call_info!(), |ctx| {
            if ctx.render_pipeline().color_targets.iter().any(|c| c.index == self.slot) {
                ctx.push_error(PipelineError::DuplicateColorTargetAccess(self.slot).into());
            }

            if alpha_to_coverage && self.slot != 0 {
                ctx.push_error(PipelineError::AlphaToCoverageUnsupportedForSlot(self.slot).into())
            }

            let num_channels = fmt_id.sample_type().map(|x| u32::from(x.len())).unwrap_or(0);

            if alpha_to_coverage && num_channels != 4 {
                ctx.push_error(
                    PipelineError::AlphaToCoverageRequires4Channels(format!("{:?}", fmt_id), num_channels).into(),
                )
            }

            ctx.render_pipeline_mut()
                .color_target0_alpha_to_coverage
                .set(alpha_to_coverage);

            if blend.is_some() && !fmt_id.is_blendable() {
                ctx.push_error(PipelineError::FormatDoesNotSupportBlending(fmt_id.into()).into())
            }

            let stype = <<T::TexelShaderType as IsVec>::T as ScalarType>::SCALAR_TYPE;
            let target = render_io::ColorTarget::new(<T as TextureFormat>::id(), blend, write_mask);
            Any::color_target_write(self.slot, target, src_color.to_any())
        });
    }
}

impl<T: ColorTargetFormat + SupportsSpp<Multi>> ColorTarget<T, Multi> {
    // `alpha_to_coverage` is only supported for the first (index 0) color target. Otherwise a recording error is generated.
    /// (no documentation yet)
    #[track_caller]
    pub fn set_with_alpha_to_coverage(self, src_color: impl To<T::TexelShaderType>)
    where
        T::TexelShaderType: IsVec<L = x4>, // must have 4 components (i.e. must have alpha channel)
    {
        let target = render_io::ColorTarget::new(<T as TextureFormat>::id(), None, Default::default());
        self.set_internal(None, ChannelWrites::rgba(), true, src_color);
    }
}

/// (no documentation yet)
pub struct ColorTargetIter<SPP: Spp> {
    next_slot: u32,
    phantom: PhantomData<SPP>,
}

impl<SPP: Spp> ColorTargetIter<SPP> {
    pub(super) fn new() -> ColorTargetIter<SPP> {
        Self {
            next_slot: 0,
            phantom: PhantomData,
        }
    }

    /// (no documentation yet)
    #[allow(clippy::should_implement_trait)]
    pub fn next<Format>(&mut self) -> ColorTarget<Format, SPP>
    where
        Format: ColorTargetFormat + SupportsSpp<SPP>,
    {
        let slot = self.next_slot;
        self.next_slot += 1;
        ColorTarget::new(slot)
    }

    /// (no documentation yet)
    pub fn at<Format>(&mut self, slot: u32) -> ColorTarget<Format, SPP>
    where
        Format: ColorTargetFormat + SupportsSpp<SPP>,
    {
        self.next_slot = slot + 1;
        ColorTarget::new(slot)
    }

    /// (no documentation yet)
    pub fn index<Format>(&mut self, slot: u32) -> ColorTarget<Format, SPP>
    where
        Format: ColorTargetFormat + SupportsSpp<SPP>,
    {
        self.at(slot)
    }
}
