//! pixel formats, which includes color, depth, depthstencil formats
use crate::rec::*;

/// data layout of the color/depth component channels inside a pixel
pub enum PixelFormat {
    /// the pixel is a color
    Color(ColorFormat),
    /// the pixel is a depth component
    Depth(DepthFormat),
}

/// implemented by pixel format tag types (see `define_color_format_types` and
/// `define_depth_format_types`)
pub trait IsPixelFormat {
    /// the corresponding enum variant to `Self`
    const ENUM: PixelFormat;
}

/// implemented by color format tag types (see `define_color_format_types`)
pub trait IsColorFormat: IsPixelFormat {
    /// the tensor type used to write to a target with this color format
    type Item: AsTen;
    /// the corresponding enum variant to `Self`
    const ENUM: ColorFormat;
}

/// implemented by depth format tag types (see `define_depth_format_types`)
pub trait IsDepthFormat: IsPixelFormat {
    /// the corresponding enum variant to `Self`
    const ENUM: DepthFormat;
}

macro_rules! define_color_format_types {
    ($($format: ident: $item_type: ident, $pixel_format: expr,)*) => { //pixel_format is ignored for now

        $(
            #[allow(non_camel_case_types)]
            #[doc = std::concat!(
                std::stringify!($format),
                " is a color format which can be written via a ",
                std::stringify!($item_type)
            )]
            pub struct $format;
            impl IsColorFormat for $format {
                type Item = $item_type;
                const ENUM: ColorFormat = ColorFormat::$format;
            }

            impl IsPixelFormat for $format {
                const ENUM: PixelFormat = PixelFormat::Color(ColorFormat::$format);
            }
        )*

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[allow(non_camel_case_types, missing_docs)]
        /// describes how a up to 4 component vector is packed into a pixel
        pub enum ColorFormat {
            $($format),*
        }
    };

}

macro_rules! define_depth_format_types {
    ($($format: ident, $pixel_format: expr,)*) => { //pixel_format is ignored for now

        $(
            #[allow(non_camel_case_types)]
            #[doc = std::concat!(
                std::stringify!($format),
                " is a depth format",
            )]
            pub struct $format;
            impl IsDepthFormat for $format {
                const ENUM: DepthFormat = DepthFormat::$format;
            }

            impl IsPixelFormat for $format {
                const ENUM: PixelFormat = PixelFormat::Depth(DepthFormat::$format);
            }
        )*

        #[allow(non_camel_case_types, missing_docs)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        /// describes how depth information of a pixel is represented
        pub enum DepthFormat {
            $($format),*
        }
    };
}

define_color_format_types! {
    // Linear color formamts, pixels are read and written without conversion. Vulkan considers these as `UNORM`
    R_8: float,               PixelFormat::R8,
    RG_88: float2,            PixelFormat::R8G8,
    RGB_888: float3,          PixelFormat::R8G8B8,
    RGBA_8888: float4,        PixelFormat::R8G8B8A8,
    R_16: float,              PixelFormat::R16,
    RG_16_16: float2,         PixelFormat::R16G16,
    RGB_16_16_16: float3,     PixelFormat::R16G16B16,
    RGBA_16_16_16_16: float4, PixelFormat::R16G16B16A16,

    // Floating point textures (sFloat = signed floating point)
    R_32_sFloat: float,               PixelFormat::R32_SFLOAT,
    RG_32_32_sFloat: float2,          PixelFormat::R32G32_SFLOAT,
    RGB_32_32_32_sFloat: float3,      PixelFormat::R32G32B32_SFLOAT,
    RGBA_32_32_32_32_sFloat: float4,  PixelFormat::R32G32B32A32_SFLOAT,
    RGBA_16_16_16_16_sFloat: float4,  PixelFormat::R16G16B16A16_SFLOAT,

    // sRGB color formats, pixels are converted at read/write operations
    R_8_sRGB: float,          PixelFormat::R8_SRGB,
    RG_88_sRGB: float2,       PixelFormat::R8G8_SRGB,
    RGB_888_sRGB: float3,     PixelFormat::R8G8B8_SRGB,
    RGBA_8888_sRGB: float4,   PixelFormat::R8G8B8A8_SRGB,

    // // BGRA Formats
    BGRA_8888: float4,        PixelFormat::B8G8R8A8,
    BGR_888_sRGB: float3,     PixelFormat::B8G8R8_SRGB,
    BGRA_8888_sRGB: float4,   PixelFormat::B8G8R8A8_SRGB,

    // HDR Formats
    ABGR_2_10_10_10_Pack32: float4, PixelFormat::A2B10G10R10UnormPack32,
    ARGB_2_10_10_10_Pack32: float4, PixelFormat::A2R10G10B10UnormPack32,

    // Refer to the Format of the surface you're rendering to
    RGBA_Surface: float4, PixelFormat::RGBA_Surface,
}

define_depth_format_types! {
    Depth32, PixelFormat::Depth32,
}