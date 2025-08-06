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

pub fn set_color<W: std::fmt::Write>(w: &mut W, hexcode: Option<&str>, use_256_color_mode: bool) -> std::fmt::Result {
    match hexcode {
        None => write!(w, "\x1B[0m"),
        Some(h) => {
            if use_256_color_mode {
                let [r, g, b] = hex_to_rgb8(h).map(|n8| (n8 as f32 * 5.0 / 255.0) as u8);
                write!(w, "\x1B[38;5;{}m", 16 + b + (6 * (g + 6 * r)))
            } else {
                let [r, g, b] = hex_to_rgb8(h);
                write!(w, "\x1B[38;2;{};{};{}m", r, g, b)
            }
        }
    }
}

/// Implements `Display` to print `Some(T)` as `T` and `None` as the provided &'static str.
pub(crate) struct UnwrapOrStr<T>(pub Option<T>, pub &'static str);
impl<T: std::fmt::Display> std::fmt::Display for UnwrapOrStr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnwrapOrStr(Some(s), _) => s.fmt(f),
            UnwrapOrStr(None, s) => s.fmt(f),
        }
    }
}
