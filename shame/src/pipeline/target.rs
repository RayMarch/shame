//! Color and depth target types for writing rendertargets in render shaders
use std::{marker::PhantomData};

use super::{super::rec::*, with_thread_render_pipeline_info_mut};
use super::{blending::Blend};
use super::pixel_format::*;

/// color rendertarget (not multisampled)
pub struct Color<CF: IsColorFormat> {
    pub(crate) _phantom: PhantomData<CF>,
    pub(crate) value: WriteOnly<<CF::Item as AsTen>::S, <CF::Item as AsTen>::D>,
    pub(crate) color_target_index: usize,
}

impl<CF: IsColorFormat> Color<CF> {

    /// assign color values to the pixels in this rendertarget
    /// (same as `set`, couldn't decide on a name)
    pub fn write(mut self, color_value: impl AsTen<S=<CF::Item as AsTen>::S, D=<CF::Item as AsTen>::D>) {
        self.value.write(color_value)
    }

    /// assign color values to the pixels in this rendertarget
    /// (same as `write`, couldn't decide on a name)
    pub fn set(self, color_value: impl AsTen<S=<CF::Item as AsTen>::S, D=<CF::Item as AsTen>::D>) {
        self.write(color_value)
    }

    /// blend color values on top of the existing pixels in this rendertarget
    /// using the blend functions described in `blend`
    pub fn blend(self, blend: Blend, color_value: impl AsTen<S=<CF::Item as AsTen>::S, D=<CF::Item as AsTen>::D>) {
        with_thread_render_pipeline_info_mut(|r| {
            let target = r.color_targets.get_mut(self.color_target_index).expect("color target index out of bounds");
            target.blending = Some(blend)
        });
        self.write(color_value);
    }

}

/// multisample color rendertarget
pub struct ColorMS<CF: IsColorFormat, const SAMPLES: u8> {
    pub(crate) _phantom: PhantomData<CF>,
    pub(crate) value: WriteOnly<<CF::Item as AsTen>::S, <CF::Item as AsTen>::D>,
    pub(crate) color_target_index: usize,
}

impl<CF: IsColorFormat, const SAMPLES: u8> ColorMS<CF, SAMPLES> {

    /// assign color values to the samples in this color rendertarget
    /// (same as `set`, couldn't decide on a name)
    pub fn write(mut self, color_value: impl AsTen<S=<CF::Item as AsTen>::S, D=<CF::Item as AsTen>::D>) {
        narrow_stages_or_push_error([self.value.stage(), color_value.stage()]);
        self.value.write(color_value)
    }

    /// assign color values to the samples in this rendertarget
    /// (same as `write`, couldn't decide on a name)
    pub fn set(self, color_value: impl AsTen<S=<CF::Item as AsTen>::S, D=<CF::Item as AsTen>::D>) {
        self.write(color_value)
    }

    /// blend color values on top of the existing samples in this rendertarget
    /// using the blend functions described in `blend`
    pub fn blend(self, blend: Blend, color_value: Ten<<CF::Item as AsTen>::S, <CF::Item as AsTen>::D>) {
        with_thread_render_pipeline_info_mut(|r| {
            let target = r.color_targets.get_mut(self.color_target_index).expect("color target index out of bounds");
            target.blending = Some(blend)
        });
        self.write(color_value);
    }
}

/// depth rendertarget
pub struct Depth<DF: IsDepthFormat> { //TODO: multisampling depth buffer
    pub(crate) _phantom: PhantomData<DF>,
}

/// which kind of depth test is performed to decide whether a fragment color
/// is written or blended onto the color rendertargets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthTest {
    /// the fragment always passes the depth test
    Always,
    /// the fragment never passes the depth test
    Never,
    /// the fragment passes if its depth value is less than the value in the
    /// depth buffer
    Less,
    /// the fragment passes if its depth value is equal to the value in the
    /// depth buffer
    Equal,
    /// the fragment passes if its depth value is greater than the value in the
    /// depth buffer
    Greater,
    /// the fragment passes if its depth value is <= than the value in the depth
    /// buffer
    LessOrEqual,
    /// the fragment passes if its depth value is >= than the value in the depth
    /// buffer
    GreaterOrEqual,
    /// the fragment passes if its depth value is != than the value in the depth
    /// buffer
    NotEqual,
}

/// value that should be written to the depth buffer
pub enum DepthWrite {
    /// a specific per-fragment float value (this value is converted to the
    /// depth format of the respective depth buffer)
    Write(float),
    /// write the z value of the interpolated clip-space position that was
    /// passed to the rasterizer
    PrimitiveZ,
    /// don't write a depth value. Keep the one that is currently in the buffer
    Off,
}

impl<DF: IsDepthFormat> Depth<DF> {

    /// perform a depth test and/or write a value to the depth buffer
    ///
    /// if you don't want to depth-test, use [`DepthTest::Always`]
    ///
    /// if you don't want to depth-write, use [`DepthWrite::Off`]
    pub fn test_write(self, test: DepthTest, write: DepthWrite) {
        with_thread_render_pipeline_info_mut(|r| {
            r.depth_test = Some(test);
            r.depth_write = Some(match write {
                DepthWrite::Write(value) => {
                    shame_graph::Any::f_frag_depth().set(value.as_any());
                    true
                },
                DepthWrite::PrimitiveZ => true,
                DepthWrite::Off => false,
            });
        });
    }
}