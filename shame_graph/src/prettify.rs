use std::fmt::Write;
use Highlight::*;

macro_rules! highlights {
    ($($kind: ident => $color: expr,)+) => {
        pub enum Highlight {$($kind),+}
        const HIGHLIGHTS: &[Highlight] = &[$(Highlight::$kind),+];
        impl Highlight {
            pub fn get_color(&self) -> &'static str {
                match self {$($kind => $color),+}
            }
        }
    };
}

highlights! {
    //this is the order in which they will be attempted to parse
    //colors are converted to 256 ansi colors
    //first match counts
    Comment         => "#4B795D",
    LineComment     => "#4B795D",
    Preprocessor    => "#A26C79",
    BuiltinType     => "#30948A",
    BuiltinFunc     => "#9A639C",
    GLUnderscore    => "#9A639C",
    Keyword         => "#719DD1",
    UppercaseIdent  => "#17988B",
    Ident           => "#6F6F6F",
    Operator        => "#949494",
    Punctuation     => "#777777",
    Braces          => "#553856",
    Number          => "#778571",
    Unmatched       => "#DA929C",
}

fn try_read_str<'a>(input: &'a str, patterns: &[&str]) -> Option<(&'a str, &'a str)> {
    patterns
        .iter()
        .find(|t| input.starts_with(**t))
        .map(|t| input.split_at(t.len()))
}

fn try_read_word<'a>(input: &'a str, patterns: &[&str]) -> Option<(&'a str, &'a str)> {
    try_read_str(input, patterns)
        .and_then(|(head, tail)| (!tail.starts_with(|x: char| x.is_alphanumeric() || x == '_')).then(|| (head, tail)))
}

fn try_read_from_until<'a>(input: &'a str, [a, b]: [&str; 2]) -> Option<(&'a str, &'a str)> {
    try_read_str(input, &[a]).map(|(_head, tail)| {
        tail.find(b)
            .map(|len| input.split_at(a.len() + len + b.len()))
            .unwrap_or((input, ""))
    })
}

/// add color codes to the glsl string to highlight glsl syntax
pub fn syntax_highlight_glsl(mut code: &str) -> String {
    let mut out = "".to_string();
    let is_alphanumeric_underscore = |x: &char| x.is_alphanumeric() || *x == '_';

    loop {
        let found = HIGHLIGHTS.iter().find_map(|h| {
            match *h {
                Comment => try_read_from_until(code, ["/*", "*/"]),
                LineComment => try_read_from_until(code, ["//", "\n"]),
                Preprocessor => try_read_from_until(code, ["#", "\n"]),
                Keyword => try_read_word(
                    code,
                    &[
                        "struct",
                        "set",
                        "binding",
                        "layout",
                        "location",
                        "push_constant",
                        "flat",
                        "smooth",
                        "noperspective",
                        "packed",
                        "shared",
                        "std430",
                        "std140",
                        "readonly",
                        "buffer",
                        "uniform",
                        "in",
                        "out",
                    ],
                ),
                BuiltinType => try_read_word(
                    code,
                    &[
                        "void",
                        "double",
                        "bool",
                        "float",
                        "vec2",
                        "vec3",
                        "vec4",
                        "ivec2",
                        "ivec3",
                        "ivec4",
                        "bvec2",
                        "bvec3",
                        "bvec4",
                        "int",
                        "uint",
                        "uvec2",
                        "uvec3",
                        "uvec4",
                        "dvec2",
                        "dvec3",
                        "dvec4",
                        "mat2",
                        "mat3",
                        "mat4",
                        "mat2x2",
                        "mat2x3",
                        "mat2x4",
                        "mat3x2",
                        "mat3x3",
                        "mat3x4",
                        "mat4x2",
                        "mat4x3",
                        "mat4x4",
                        "dmat2",
                        "dmat3",
                        "dmat4",
                        "dmat2x2",
                        "dmat2x3",
                        "dmat2x4",
                        "dmat3x2",
                        "dmat3x3",
                        "dmat3x4",
                        "dmat4x2",
                        "dmat4x3",
                        "dmat4x4",
                        "sampler1D",
                        "sampler1DShadow",
                        "sampler1DArray",
                        "sampler1DArrayShadow",
                        "isampler1D",
                        "isampler1DArray",
                        "usampler1D",
                        "usampler1DArray",
                        "sampler2D",
                        "sampler2DShadow",
                        "sampler2DArray",
                        "sampler2DArrayShadow",
                        "isampler2D",
                        "isampler2DArray",
                        "usampler2D",
                        "usampler2DArray",
                        "sampler2DRect",
                        "sampler2DRectShadow",
                        "isampler2DRect",
                        "usampler2DRect",
                        "sampler2DMS",
                        "isampler2DMS",
                        "usampler2DMS",
                        "sampler2DMSArray",
                        "isampler2DMSArray",
                        "usampler2DMSArray",
                        "sampler3D",
                        "isampler3D",
                        "usampler3D",
                        "samplerCube",
                        "samplerCubeShadow",
                        "isamplerCube",
                        "usamplerCube",
                        "samplerCubeArray",
                        "samplerCubeArrayShadow",
                        "isamplerCubeArray",
                        "usamplerCubeArray",
                        "samplerBuffer",
                        "isamplerBuffer",
                        "usamplerBuffer",
                        "image1D",
                        "iimage1D",
                        "uimage1D",
                        "image1DArray",
                        "iimage1DArray",
                        "uimage1DArray",
                        "image2D",
                        "iimage2D",
                        "uimage2D",
                        "image2DArray",
                        "iimage2DArray",
                        "uimage2DArray",
                        "image2DRect",
                        "iimage2DRect",
                        "uimage2DRect",
                        "image2DMS",
                        "iimage2DMS",
                        "uimage2DMS",
                        "image2DMSArray",
                        "iimage2DMSArray",
                        "uimage2DMSArray",
                        "image3D",
                        "iimage3D",
                        "uimage3D",
                        "imageCube",
                        "iimageCube",
                        "uimageCube",
                        "imageCubeArray",
                        "iimageCubeArray",
                        "uimageCubeArray",
                        "imageBuffer",
                        "iimageBuffer",
                        "uimageBuffer",
                        "texture1D",
                        "texture1DArray",
                        "itexture1D",
                        "itexture1DArray",
                        "utexture1D",
                        "utexture1DArray",
                        "texture2D",
                        "texture2DArray",
                        "itexture2D",
                        "itexture2DArray",
                        "utexture2D",
                        "utexture2DArray",
                        "texture2DRect",
                        "itexture2DRect",
                        "utexture2DRect",
                        "texture2DMS",
                        "itexture2DMS",
                        "utexture2DMS",
                        "texture2DMSArray",
                        "itexture2DMSArray",
                        "utexture2DMSArray",
                        "texture3D",
                        "itexture3D",
                        "utexture3D",
                        "textureCube",
                        "itextureCube",
                        "utextureCube",
                        "textureCubeArray",
                        "itextureCubeArray",
                        "utextureCubeArray",
                        "textureBuffer",
                        "itextureBuffer",
                        "utextureBuffer",
                        "sampler",
                        "samplerShadow",
                    ],
                ),
                BuiltinFunc => try_read_word(
                    code,
                    &[
                        "radians",
                        "degrees",
                        "sin",
                        "cos",
                        "tan",
                        "asin",
                        "acos",
                        "atan",
                        "sinh",
                        "cosh",
                        "tanh",
                        "asinh",
                        "acosh",
                        "atanh",
                        "pow",
                        "exp",
                        "log",
                        "exp2",
                        "log2",
                        "sqrt",
                        "inversesqrt",
                        "abs",
                        "sign",
                        "floor",
                        "trunc",
                        "round",
                        "roundEven",
                        "ceil",
                        "fract",
                        "mod",
                        "modf",
                        "min",
                        "max",
                        "clamp",
                        "mix",
                        "step",
                        "smoothstep",
                        "isnan",
                        "isinf",
                        "floatBitsToInt",
                        "floatBitsToUint",
                        "intBitsToFloat",
                        "uintBitsToFloat",
                        "fma",
                        "frexp",
                        "ldexp",
                        "packUnorm2x16",
                        "packSnorm2x16",
                        "packUnorm4x8",
                        "packSnorm4x8",
                        "unpackUnorm2x16",
                        "unpackSnorm2x16",
                        "unpackUnorm4x8",
                        "unpackSnorm4x8",
                        "packHalf2x16",
                        "unpackHalf2x16",
                        "packDouble2x32",
                        "unpackDouble2x32",
                        "length",
                        "distance",
                        "dot",
                        "cross",
                        "normalize",
                        "ftransform",
                        "faceforward",
                        "reflect",
                        "refract",
                        "matrixCompMult",
                        "outerProduct",
                        "transpose",
                        "determinant",
                        "inverse",
                        "lessThan",
                        "lessThanEqual",
                        "greaterThan",
                        "greaterThanEqual",
                        "equal",
                        "notEqual",
                        "any",
                        "all",
                        "not",
                        "uaddCarry",
                        "usubBorrow",
                        "umulExtended",
                        "imulExtended",
                        "bitfieldExtract",
                        "bitfieldInsert",
                        "bitfieldReverse",
                        "bitCount",
                        "findLSB",
                        "findMSB",
                        "textureSize",
                        "textureQueryLod",
                        "textureQueryLevels",
                        "textureSamples",
                        "texelFetchOffset",
                        "textureProjOffset",
                        "textureLodOffset",
                        "textureProj",
                        "textureLod",
                        "textureOffset",
                        "texelFetch",
                        "textureProjLodOffset",
                        "textureGradOffset",
                        "textureProjGradOffset",
                        "textureProjGrad",
                        "textureGrad",
                        "textureProjLod",
                        "texture",
                        "textureGather",
                        "textureGatherOffsets",
                        "textureGatherOffset",
                        "texture1DProjLod",
                        "texture1DProj",
                        "texture1DLod",
                        "texture1D",
                        "texture2DProjLod",
                        "texture2DProj",
                        "texture2DLod",
                        "texture2D",
                        "texture3DProjLod",
                        "texture3DProj",
                        "texture3DLod",
                        "texture3D",
                        "textureCube",
                        "textureCubeLod",
                        "shadow1DProjLod",
                        "shadow2DProjLod",
                        "shadow1DLod",
                        "shadow2DLod",
                        "shadow1DProj",
                        "shadow2DProj",
                        "shadow1D",
                        "shadow2D",
                    ],
                ),
                GLUnderscore => code.starts_with("gl_").then(|| {
                    let len = code
                        .chars()
                        .take_while(is_alphanumeric_underscore)
                        .map(|x| x.len_utf8())
                        .sum();
                    code.split_at(len)
                }),
                UppercaseIdent => code.starts_with(|x: char| x.is_uppercase()).then(|| {
                    let len = code
                        .chars()
                        .take_while(is_alphanumeric_underscore)
                        .map(|x| x.len_utf8())
                        .sum();
                    code.split_at(len)
                }),
                Ident => code.starts_with(|x: char| x.is_alphabetic() || x == '_').then(|| {
                    let len = code
                        .chars()
                        .take_while(is_alphanumeric_underscore)
                        .map(|x| x.len_utf8())
                        .sum();
                    code.split_at(len)
                }),
                Punctuation => try_read_str(code, &[".", "(", ")", ",", ";", "[", "]"]),
                Braces => try_read_str(code, &["{", "}"]),
                Number => code.starts_with(|x: char| x.is_ascii_digit()).then(|| {
                    let len = code
                        .chars()
                        .take_while(|x| x.is_ascii_digit() || ['.', 'u', 'U'].contains(x))
                        .map(|x| x.len_utf8())
                        .sum();
                    code.split_at(len)
                }),
                Operator => try_read_str(
                    code,
                    &[
                        "=", "+", "-", "~", "!", "*", "/", "%", "<<", ">>", "<", ">", "<=", ">=", "==", "!=", "&", "^",
                        "|", "&&", "||",
                    ],
                ),
                Unmatched => code.chars().next().map(|c| code.split_at(c.len_utf8())), //advance by 1 char
            }
            .map(|split| (split, h))
        });

        match found {
            None => break,
            Some(((token, tail), h)) => {
                code = tail;
                set_color(&mut out, Some(h.get_color()), true);
                out.write_fmt(format_args!("{}", token)).ok();
            }
        }
    }
    set_color(&mut out, None, false);
    out
}

/// takes #RRGGBB hex codes, e.g. "#FE2215"
fn hex_to_rgb8(hexcode: &str) -> [u8; 3] {
    let (_numsign, hex) = hexcode.split_at(1);
    let invalid = || panic!("invalid #RRGGBB color hex code '{}'", hexcode);
    match hex.len() {
        6 => match [
            u8::from_str_radix(&hex[0..2], 16),
            u8::from_str_radix(&hex[2..4], 16),
            u8::from_str_radix(&hex[4..6], 16),
        ] {
            [Ok(r), Ok(g), Ok(b)] => [r, g, b],
            _ => invalid(),
        },
        _ => invalid(),
    }
}

pub fn set_color<W: std::fmt::Write>(w: &mut W, hexcode: Option<&str>, use_256_color_mode: bool) {
    match hexcode {
        None => w.write_str("\x1B[0m").ok(),
        Some(h) => {
            if use_256_color_mode {
                let [r, g, b] = hex_to_rgb8(h).map(|n8| (n8 as f32 / (256.0 / 6.0)) as u8);
                w.write_fmt(format_args!("\x1B[38;5;{}m", 16 + b + (6 * (g + 6 * r))))
                    .ok()
            } else {
                let [r, g, b] = hex_to_rgb8(h);
                w.write_fmt(format_args!("\x1B[38;2;{};{};{}m", r, g, b)).ok()
            }
        }
    };
}
