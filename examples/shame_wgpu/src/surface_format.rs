use std::borrow::Cow;
use shame as sm;

/// the surface format is a texture format chosen at runtime, depending on the
/// window. It will most likely turn out to be `Bgra8UnormSrgb` or `Bgra8Unorm`.
/// This type is treated as a 4 component texture format (color + alpha)
/// that is blendable and supports multisampling.
///
/// > note: In this example implementation this format is not considered Sampleable,
/// > but that feature can be added by making bindgroups able to access the surface
/// > format during bindgroup layout creation. For now we recommend to just
/// > use a specific texture format like `Rgba8Unorm` for those cases.
pub struct SurfaceFormat;

impl sm::TextureFormat for SurfaceFormat {
    fn id() -> impl shame::TextureFormatId { ExtendedTextureFormats::SurfaceFormat }
}
impl sm::Aspect for SurfaceFormat {
    type TexelShaderType = sm::f32x4;
}
impl sm::ColorTargetFormat for SurfaceFormat {}
impl sm::Blendable for SurfaceFormat {}
impl sm::SupportsSpp<sm::Single> for SurfaceFormat {}
impl sm::SupportsSpp<sm::Multi> for SurfaceFormat {}

#[derive(Debug)]
#[repr(u8)]
pub enum ExtendedTextureFormats {
    SurfaceFormat, // maybe its worth adding this to the builtin shame formats
}

impl sm::TextureFormatId for ExtendedTextureFormats {
    fn to_binary_repr(&self) -> Cow<[u8]> { b"SurfaceFormat".into() }

    fn to_wgsl_repr(&self) -> Option<Cow<str>> { None }

    fn sample_type(&self) -> Option<sm::TextureSampleUsageType> {
        Some(sm::TextureSampleUsageType::FilterableFloat { len: sm::x4.into() })
    }

    fn has_aspect(&self, aspect: sm::TextureAspect) -> bool { aspect == sm::TextureAspect::Color }

    fn is_blendable(&self) -> bool { true }
}
