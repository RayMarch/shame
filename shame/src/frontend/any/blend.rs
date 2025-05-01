/// Describes the blending operation of the per-fragment color that is calculated
/// by the pipeline (the source color) onto the color that is already present
/// in the color target at that pixel location (the destination color)
/// ## examples
/// ```
/// // convenience constructors for popular blend modes
/// let alpha = Blend::alpha()
/// let add = Blend::add()
///
/// // manual blend mode constrution, by describing rgb and alpha blend equations
/// use shame::BlendFactor::*;
/// use shame::BlendOperation::*;
///
/// let add = Blend::from_parts(
///     (SrcAlpha, Add, One), // rgb
///     (One, Add, One) // alpha
/// );
///
/// // or more verbose
/// Blend::new(
///     BlendComponent {
///         src_factor: BlendFactor::SrcAlpha,
///         operation: BlendOperation::Add,
///         dst_factor: BlendFactor::One,
///     },
///     BlendComponent {
///         src_factor: BlendFactor::One,
///         operation: BlendOperation::Add,
///         dst_factor: BlendFactor::One,
///     },
/// )
/// ```
/// for more information on how the blend equation works
/// see the documentation of [`BlendComponent`]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Blend {
    /// blend operation applied to the red, green and blue components
    pub color: BlendComponent,
    /// blend operation applied to the alpha component
    pub alpha: BlendComponent,
}

impl Blend {
    /// constructs a new [`Blend`] given two blend equations,
    /// one for the rgb components and one for the alpha components of
    /// per-fragment color (source) and color target pixel (destination) respectively
    pub fn new(rgb: BlendComponent, a: BlendComponent) -> Self { Self { color: rgb, alpha: a } }

    /// alternative constructor for [`Blend`] by providing [`BlendComponent`] as tuples
    pub fn from_parts(
        rgb: (BlendFactor, BlendOperation, BlendFactor),
        a: (BlendFactor, BlendOperation, BlendFactor),
    ) -> Self {
        Self::new(
            BlendComponent {
                src_factor: rgb.0,
                operation: rgb.1,
                dst_factor: rgb.2,
            },
            BlendComponent {
                src_factor: a.0,
                operation: a.1,
                dst_factor: a.2,
            },
        )
    }

    /// alpha blending
    pub fn alpha() -> Blend { Blend::from_parts((SrcAlpha, Add, OneMinusSrcAlpha), (One, Add, One)) }

    /// additive blending
    pub fn add() -> Blend { Blend::from_parts((SrcAlpha, Add, One), (One, Add, One)) }

    /// max blending (the "lighter" color dominates)
    pub fn max() -> Blend { Blend::from_parts((SrcAlpha, Max, One), (One, Max, One)) }

    /// min blending (the "darker" color dominates)
    pub fn min() -> Blend { Blend::from_parts((SrcAlpha, Min, One), (One, Min, One)) }
}

impl Display for Blend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //rgb: (1-dst.rgb + src.a), a: (dst.a + src.a)
        f.write_fmt(format_args!("rgb={{ {} }}.rgb, a={{ {} }}.a", self.color, self.alpha))
    }
}

impl Display for BlendComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use BlendFactor::*;
        // (1-dst.rgb + src.a)
        let format_fac = |name: &str, fac: BlendFactor| -> String {
            let fac = match fac {
                Zero => "0",
                One => "1",
                Src => "src.rgb",
                OneMinusSrc => "(1-src.rgb)",
                Dst => "dst.rgb",
                OneMinusDst => "(1-dst.rgb)",
                SrcAlpha => "src.a",
                OneMinusSrcAlpha => "(1-src.a)",
                DstAlpha => "dst.a",
                OneMinusDstAlpha => "(1-dst.a)",
                Constant => "const",
                OneMinusConstant => "(1-const)",
                SrcAlphaSaturated => "min(src.a, 1-dst.a)",
                // ConstantAlpha => "constant.a",
                // OneMinusConstantAlpha => "(1-constant.a)",
            };
            format!("{name}*{fac}")
        };

        let src = format_fac("src", self.src_factor);
        let dst = format_fac("dst", self.dst_factor);

        use BlendOperation::*;
        match self.operation {
            Add => f.write_fmt(format_args!("{src} + {dst}")),
            Subtract => f.write_fmt(format_args!("{src} - {dst}")),
            ReverseSubtract => f.write_fmt(format_args!("-{src} + {dst}")),
            Min => f.write_fmt(format_args!("{dst}.min({src})")),
            Max => f.write_fmt(format_args!("{dst}.max({src})")),
        }
    }
}

/// A parametrization of the blend equation that can be applied to
/// either the `rgb` components or the alpha components.
///
/// the equation is:
///
/// Src * `src_factor` `operation` Dst * `dst_factor`
///
/// where
/// - Src is the incoming per-fragment color calculated in the pipeline
/// - Dst is the color that is already present in the color target at the fragment position
/// - `operation` is a binary operator
///
/// ## example
/// ```
/// let eqadd = BlendComponent {
///     src_factor: BlendFactor::SrcAlpha,
///     dst_factor: BlendFactor::One,
///     operation: BlendOperation::Add,
/// }
/// ```
/// here `eqadd` describes the following equation:
///
/// Src * SrcAlpha Add Dst * One
///
/// which, if the blend equation is applied to the `rgb` components can be
/// rewritten to:
///
/// Src.rgb * Src.alpha + Dst.rgb * 1.0
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BlendComponent {
    /// factor multiplied with the per-fragment color that was calculated by
    /// the pipeline
    pub src_factor: BlendFactor,
    /// factor multiplied with the color that is already present in the color target
    /// at the fragment position
    pub dst_factor: BlendFactor,
    /// binary operator that combines the source and destination arguments
    pub operation: BlendOperation,
}


/// A factor used in the blend equation, see [`BlendComponent`]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum BlendFactor {
    /// 0.0
    Zero = 0,
    /// 1.0
    One = 1,
    /// S.component
    Src = 2,
    /// 1.0 - S.component
    OneMinusSrc = 3,
    /// S.alpha
    SrcAlpha = 4,
    /// 1.0 - S.alpha
    OneMinusSrcAlpha = 5,
    /// D.component
    Dst = 6,
    /// 1.0 - D.component
    OneMinusDst = 7,
    /// D.alpha
    DstAlpha = 8,
    /// 1.0 - D.alpha
    OneMinusDstAlpha = 9,
    /// min(S.alpha, 1.0 - D.alpha)
    SrcAlphaSaturated = 10,
    /// Constant
    Constant = 11,
    /// 1.0 - Constant
    OneMinusConstant = 12,
}
use std::fmt::Display;

use BlendFactor::*;

/// A binary operator used in the blend equation, see [`BlendComponent`]
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub enum BlendOperation {
    /// Src + Dst
    #[default]
    Add = 0,
    /// Src - Dst
    Subtract = 1,
    /// Dst - Src
    ReverseSubtract = 2,
    /// min(Src, Dst)
    Min = 3,
    /// max(Src, Dst)
    Max = 4,
}
use BlendOperation::*;
