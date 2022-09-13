//! Color target blend equations
use std::fmt::Display;

/// Factor that source or destination colors in a [`BlendEquation`] are multiplied
/// with before the [`BlendOp`] is applied
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum BlendFactor {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    DestinationColor,
    OneMinusDestinationColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationAlpha,
    OneMinusDestinationAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
}

/// Describes how source and destination colors are combined after having their
/// respective factors applied
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// Describes how to blend a source pixel onto a destination pixel of a certain
/// color target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendEquation {
    /// Factor that the source color is multiplied with before [`BlendOp`] is
    /// applied
    pub src_factor: BlendFactor,
    /// Factor that the destination color (the color of the pixel inside the
    /// color target) is multiplied with before [`BlendOp`] is applied
    pub dst_factor: BlendFactor,
    /// The operation
    pub op: BlendOp,
}

/// Describes blending of the incoming fragment that the shader calculated (the
/// source color) onto the color that is already present in the color target at
/// that pixel (the destination color)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blend {
    /// Applied to the rgb color components
    pub rgb: BlendEquation,
    /// Applied to the alpha component
    pub a: BlendEquation,
}

impl Blend {
    /// constructs a new [`Blend`] given two blend equations,
    /// one for the rgb components and one for the alpha components of
    /// source fragment and destination pixel
    pub fn new(rgb: BlendEquation, a: BlendEquation) -> Self { Self { rgb, a } }

    /// Convenience constructor for alpha blending
    pub fn alpha() -> Blend {
        Self {
            rgb: BlendEquation {
                src_factor: BlendFactor::SourceAlpha,
                dst_factor: BlendFactor::OneMinusSourceAlpha,
                op: BlendOp::Add,
            },
            a: BlendEquation {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                op: BlendOp::Add,
            },
        }
    }

    /// Convenience constructor for additive blending
    pub fn add() -> Blend {
        Self {
            rgb: BlendEquation {
                src_factor: BlendFactor::SourceAlpha,
                dst_factor: BlendFactor::One,
                op: BlendOp::Add,
            },
            a: BlendEquation {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                op: BlendOp::Add,
            },
        }
    }
}

impl Display for Blend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //rgb: (1-dst.rgb + src.a), a: (dst.a + src.a)
        f.write_fmt(format_args!("rgb={{ {} }}.rgb, a={{ {} }}.a", self.rgb, self.a))
    }
}

impl Display for BlendEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use BlendFactor::*;
        // (1-dst.rgb + src.a)
        let format_fac = |name: &str, fac: BlendFactor| -> String {
            let fac = match fac {
                Zero => "0",
                One => "1",
                SourceColor => "src.rgb",
                OneMinusSourceColor => "(1-src.rgb)",
                DestinationColor => "dst.rgb",
                OneMinusDestinationColor => "(1-dst.rgb)",
                SourceAlpha => "src.a",
                OneMinusSourceAlpha => "(1-src.a)",
                DestinationAlpha => "dst.a",
                OneMinusDestinationAlpha => "(1-dst.a)",
                ConstantColor => "const.rgb",
                OneMinusConstantColor => "(1-const.rgb)",
                ConstantAlpha => "const.a",
                OneMinusConstantAlpha => "(1-const.a)",
            };
            format!("{name}*{fac}")
        };

        let src = format_fac("src", self.src_factor);
        let dst = format_fac("dst", self.dst_factor);

        use BlendOp::*;
        match self.op {
            Add => f.write_fmt(format_args!("{src} + {dst}")),
            Subtract => f.write_fmt(format_args!("{src} - {dst}")),
            ReverseSubtract => f.write_fmt(format_args!("-{src} + {dst}")),
            Min => f.write_fmt(format_args!("{dst}.min({src})")),
            Max => f.write_fmt(format_args!("{dst}.max({src})")),
        }
    }
}
