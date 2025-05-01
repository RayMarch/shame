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
    AtSignIdent     => "#9A639C",
    Keyword         => "#719DD1",
    UppercaseIdent  => "#17988B",
    Ident           => "#AAAAAA", //"#6F6F6F",
    Operator        => "#949494",
    Punctuation     => "#6F6F6F",
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
        .and_then(|(head, tail)| (!tail.starts_with(|x: char| x.is_alphanumeric() || x == '_')).then_some((head, tail)))
}

fn try_read_from_until<'a>(input: &'a str, [a, b]: [&str; 2]) -> Option<(&'a str, &'a str)> {
    try_read_str(input, &[a]).map(|(_head, tail)| {
        tail.find(b)
            .map(|len| input.split_at(a.len() + len + b.len()))
            .unwrap_or((input, ""))
    })
}

/// slow and inaccurate syntax highlighting for wgsl code.
///
/// TODO(low prio) please rewrite at some point.
#[rustfmt::skip]
pub fn syntax_highlight_wgsl(mut code: &str) -> String {
    let mut out = "".to_string();
    let is_alphanumeric_underscore = |x: &char| x.is_alphanumeric() || *x == '_';

    loop {
        let found = HIGHLIGHTS.iter().find_map(|h| match *h {
            Comment => try_read_from_until(code, ["/*", "*/"]),
            LineComment => try_read_from_until(code, ["//", "\n"]),
            Preprocessor => try_read_from_until(code, ["#", "\n"]),
            Keyword => try_read_word(code, &[
                "fn", "struct", "return", "var", "let", "if", "else", "while", "for", "true", "false",
            ]),
            BuiltinType => try_read_word(code, &[
                "vec2f", "vec2f", "vec2f", "vec3f", "vec3f", "vec3f", "vec4f", "vec4f", "vec4f", "vec2u", "vec2u", "vec2u", "vec3u", "vec3u", "vec3u", "vec4u", "vec4u", "vec4u", "vec2i", "vec2i", "vec2i", "vec3i", "vec3i", "vec3i", "vec4i", "vec4i", "vec4i", "vec2b", "vec2b", "vec2b", "vec3b", "vec3b", "vec3b", "vec4b", "vec4b", "vec4b", "vec2h", "vec2h", "vec2h", "vec3h", "vec3h", "vec3h", "vec4h", "vec4h", "vec4h",
                "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4", "mat4x2", "mat4x3", "mat4x4", "vec2", "vec3", "vec4",
                "bool", "u32", "i32", "f32", "f16", "vec2f", "vec3f", "vec4f", "atomic", "array",
                "function", "workgroup", "storage", "uniform", "push_constant", "read_write", "read", "write",
            ]),
            BuiltinFunc => try_read_word(code, &[
                "workgroupUniformLoad", "workgroupBarrier", "textureBarrier", "storageBarrier", "unpack2x16float", "unpack2x16unorm", "unpack2x16snorm", "unpack4x8unorm", "unpack4x8snorm", "pack2x16float", "pack2x16unorm", "pack2x16snorm", "pack4x8unorm", "pack4x8snorm", "atomicCompareExchangeWeak", "atomicExchange", "atomicXor", "atomicOr", "atomicAnd", "atomicMin", "atomicMax", "atomicSub", "atomicAdd", "atomicStore", "atomicLoad", "textureStore", "textureSampleBaseClampToEdge", "textureSampleLevel", "textureSampleGrad", "textureSampleCompareLevel", "textureSampleCompare", "textureSampleBias", "textureSample", "textureNumSamples", "textureNumLevels", "textureNumLayers", "textureLoad", "textureGatherCompare", "textureGather", "textureDimensions", "fwidthFine", "fwidthCoarse", "fwidth", "dpdyFine", "dpdyCoarse", "dpdy", "dpdxFine", "dpdxCoarse", "dpdx", "trunc", "transpose", "tanh", "tan", "step", "sqrt", "smoothstep", "sinh", "sin", "sign", "saturate", "round", "reverseBits", "refract", "reflect", "radians", "quantizeToF16", "pow", "normalize", "modf", "mix", "max", "log2", "log", "length", "ldexp", "inverseSqrt", "insertBits", "frexp", "fract", "fma", "floor", "firstTrailingBit", "firstLeadingBit", "faceForward", "extractBits", "exp2", "exp", "dot", "distance", "determinant", "degrees", "cross", "countTrailingZeros", "countOneBits", "countLeadingZeros", "cosh", "cos", "clamp", "ceil", "atan2", "atanh", "atan", "asinh", "asin", "acosh", "acos", "abs", "arrayLength", "select", "any", "all", "min",
            ]),
            AtSignIdent => {
                code.starts_with('@').then(|| {
                    let len = code.chars().take_while(|x| is_alphanumeric_underscore(x) || *x == '@').map(|x| x.len_utf8()).sum();
                    code.split_at(len) 
                })
            }
            UppercaseIdent => {
                code.starts_with(|x: char| x.is_uppercase()).then(|| {
                    let len = code.chars().take_while(is_alphanumeric_underscore).map(|x| x.len_utf8()).sum();
                    code.split_at(len)
                })
            },
            Ident => {
                code.starts_with(|x: char| x.is_alphabetic() || x == '_').then(|| {
                    let len = code.chars().take_while(is_alphanumeric_underscore).map(|x| x.len_utf8()).sum();
                    code.split_at(len)
                })
            }
            Punctuation => try_read_str(code, &[
                ".", "(", ")", ",", ";", "[", "]"
            ]),
            Braces => try_read_str(code, &["{", "}"]),
            Number => {
                code.starts_with(|x: char| x.is_ascii_digit()).then(|| {
                    let len = code.chars().take_while(|x| x.is_ascii_digit() || ['.', 'f', 'h', 'u', 'i'].contains(x))
                    .map(|x| x.len_utf8()).sum();
                    code.split_at(len)
                })
            },
            Operator => try_read_str(code, &[
                "=", "+", "-", "~", "!", "*", "/", "%", "<<", ">>", "<", ">", "<=", ">=","==", "!=", "&", "^", "|", "&&", "||"
            ]),
            Unmatched => code.chars().next().map(|c| code.split_at(c.len_utf8())) //advance by 1 char
        }.map(|split| (split, h)));

        match found {
            None => break,
            Some(((token, tail), h)) => {
                code = tail;
                set_color(&mut out, Some(h.get_color()), false);
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
