use super::{error::WgslError, WgslContext, WgslErrorKind};
use crate::backend::language::Language;
use crate::frontend::texture::texture_formats::BuiltinTextureFormatId;
use crate::ir::TextureFormatWrapper;
use crate::{
    backend::code_write_buf::CodeWriteSpan,
    ir::{recording::CallInfo, StoreType},
};
use std::fmt::Write;

pub fn wgsl_builtin_texture_format_repr(format: BuiltinTextureFormatId) -> Option<&'static str> {
    use BuiltinTextureFormatId as TF;
    match format {
        TF::R8Unorm => Some("r8unorm"),
        TF::R8Snorm => Some("r8snorm"),
        TF::R8Uint => Some("r8uint"),
        TF::R8Sint => Some("r8sint"),
        TF::R16Uint => Some("r16uint"),
        TF::R16Sint => Some("r16sint"),
        TF::R16Unorm => Some("r16unorm"),
        TF::R16Snorm => Some("r16snorm"),
        TF::R16Float => Some("r16float"),
        TF::Rg8Unorm => Some("rg8unorm"),
        TF::Rg8Snorm => Some("rg8snorm"),
        TF::Rg8Uint => Some("rg8uint"),
        TF::Rg8Sint => Some("rg8sint"),
        TF::R32Uint => Some("r32uint"),
        TF::R32Sint => Some("r32sint"),
        TF::R32Float => Some("r32float"),
        TF::Rg16Uint => Some("rg16uint"),
        TF::Rg16Sint => Some("rg16sint"),
        TF::Rg16Unorm => Some("rg16unorm"),
        TF::Rg16Snorm => Some("rg16snorm"),
        TF::Rg16Float => Some("rg16float"),
        TF::Rgba8Unorm => Some("rgba8unorm"),
        TF::Rgba8UnormSrgb => Some("rgba8unormsrgb"),
        TF::Rgba8Snorm => Some("rgba8snorm"),
        TF::Rgba8Uint => Some("rgba8uint"),
        TF::Rgba8Sint => Some("rgba8sint"),
        TF::Bgra8Unorm => Some("bgra8unorm"),
        TF::Bgra8UnormSrgb => Some("bgra8unormsrgb"),
        TF::Rgb9e5Ufloat => Some("rgb9e5ufloat"),
        TF::Rgb10a2Uint => Some("rgb10a2uint"),
        TF::Rgb10a2Unorm => Some("rgb10a2unorm"),
        TF::Rg32Uint => Some("rg32uint"),
        TF::Rg32Sint => Some("rg32sint"),
        TF::Rg32Float => Some("rg32float"),
        TF::Rgba16Uint => Some("rgba16uint"),
        TF::Rgba16Sint => Some("rgba16sint"),
        TF::Rgba16Unorm => Some("rgba16unorm"),
        TF::Rgba16Snorm => Some("rgba16snorm"),
        TF::Rgba16Float => Some("rgba16float"),
        TF::Rgba32Uint => Some("rgba32uint"),
        TF::Rgba32Sint => Some("rgba32sint"),
        TF::Rgba32Float => Some("rgba32float"),
        TF::Stencil8 => Some("stencil8"),
        TF::Depth16Unorm => Some("depth16unorm"),
        TF::Depth24Plus => Some("depth24plus"),
        TF::Depth24PlusStencil8 => Some("depth24plusstencil8"),
        TF::Depth32Float => Some("depth32float"),
        TF::Depth32FloatStencil8 => Some("depth32floatstencil8"),
        TF::NV12 => None,
        TF::Bc1RgbaUnorm => None,
        TF::Bc1RgbaUnormSrgb => None,
        TF::Bc2RgbaUnorm => None,
        TF::Bc2RgbaUnormSrgb => None,
        TF::Bc3RgbaUnorm => None,
        TF::Bc3RgbaUnormSrgb => None,
        TF::Bc4RUnorm => None,
        TF::Bc4RSnorm => None,
        TF::Bc5RgUnorm => None,
        TF::Bc5RgSnorm => None,
        TF::Bc6hRgbUfloat => None,
        TF::Bc6hRgbFloat => None,
        TF::Bc7RgbaUnorm => None,
        TF::Bc7RgbaUnormSrgb => None,
        TF::Etc2Rgb8Unorm => None,
        TF::Etc2Rgb8UnormSrgb => None,
        TF::Etc2Rgb8A1Unorm => None,
        TF::Etc2Rgb8A1UnormSrgb => None,
        TF::Etc2Rgba8Unorm => None,
        TF::Etc2Rgba8UnormSrgb => None,
        TF::EacR11Unorm => None,
        TF::EacR11Snorm => None,
        TF::EacRg11Unorm => None,
        TF::EacRg11Snorm => None,
        TF::Astc { .. } => None,
    }
}

pub(super) fn write_texture_format(
    code: &mut CodeWriteSpan,
    fmt: &TextureFormatWrapper,
    call_info: CallInfo,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    match fmt.to_wgsl_repr() {
        Some(str) => {
            write!(code, "{str}");
            Ok(())
        }
        None => Err(WgslErrorKind::UnrepresentableTextureFormat(fmt.clone())
            .at_level(call_info, super::error::WgslErrorLevel::Original)),
    }
}
