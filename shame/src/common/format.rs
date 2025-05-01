use std::fmt::Write;

use crate::ir::recording::CallInfo;

/// for "1st" "2nd" "3rd", and the likes. for `1` returns `"st"`
pub fn numeral_suffix(i: usize) -> &'static str {
    match i {
        0 => "th",
        1 => "st",
        2 => "nd",
        3 => "rd",
        4..=20 => "th",
        _ => match i % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        },
    }
}

pub fn num_digits(k: usize) -> usize { k.checked_ilog10().unwrap_or(0) as usize + 1 }

#[cfg(feature = "error_excerpt")]
pub fn write_error_excerpt(f: &mut impl Write, call_info: CallInfo, use_colors: bool) -> Result<(), std::fmt::Error> {
    use crate::common::prettify::*;
    use std::fs::File;
    use std::io::{self, BufRead};
    use std::path::Path;
    let range: std::ops::RangeInclusive<i32> = (-1)..=1;

    let use_256_color_mode = false;
    let color = |f_: &mut _, hex| match use_colors {
        true => set_color(f_, Some(hex), use_256_color_mode),
        false => Ok(()),
    };
    let reset = |f_: &mut _| match use_colors {
        true => set_color(f_, None, use_256_color_mode),
        false => Ok(()),
    };

    let Ok(file) = File::open(call_info.location.file()) else {
        color(f, "#508EE3")?;
        writeln!(f, "[code excerpt not found]")?;
        reset(f)?;
        return Ok(());
    };

    let center = call_info.location.line() as i32;
    let col = call_info.location.column();

    let mut reader = io::BufReader::new(file).lines();

    let abs_range = (center + range.start())..=(center + range.end());

    let default_line_numbers_width = 2;
    let line_numbers_width = abs_range
        .clone()
        .map(|k| num_digits(k.max(0) as usize).max(default_line_numbers_width))
        .max()
        .unwrap_or(default_line_numbers_width);

    // one blank line at the start, this seems to also be what rustc does
    {
        color(f, "#508EE3")?;
        for _ in 0..line_numbers_width {
            f.write_char(' ')?
        }
        writeln!(f, " |")?;
        reset(f)?;
    }

    let write_line_number = |f: &mut dyn Write, k: usize| -> Result<(), std::fmt::Error> {
        let k_width = num_digits(k);
        write!(f, "{k}")?;
        for _ in 0..(line_numbers_width - k_width) {
            f.write_char(' ')?
        }
        Ok(())
    };

    for (i, line) in reader.enumerate() {
        let i = i + 1; //counting starts at 1

        if abs_range.contains(&(i as i32)) {
            let Ok(line) = line else {
                color(f, "#508EE3")?;
                writeln!(f, "[code line not found]")?;
                reset(f)?;
                return Ok(());
            };

            // skip blank lines at the start or end of the range
            let is_blank = line.trim().is_empty();
            if is_blank && (i as i32 == *abs_range.start() || i as i32 == *abs_range.end()) {
                continue;
            }

            if i as i32 == center {
                color(f, "#508EE3")?;
                write_line_number(f, i)?;
                write!(f, " >")?;
                reset(f)?;
                writeln!(f, "{line}");
                color(f, "#508EE3")?;
                for _ in 0..line_numbers_width {
                    f.write_char(' ')?
                }
                write!(f, " |")?;
                for _ in 1..col {
                    f.write_char(' ')?;
                }
                writeln!(f, "^")?;
            } else {
                color(f, "#508EE3")?;
                write_line_number(f, i)?;
                write!(f, " |")?;
                reset(f)?;
                writeln!(f, "{line}");
            }
        }
    }

    // one blank line at the end, this seems to also be what rustc does
    // one blank line at the start, this seems to also be what rustc does
    {
        color(f, "#508EE3")?;
        for _ in 0..line_numbers_width {
            f.write_char(' ')?
        }
        writeln!(f, " |")?;
        reset(f)?;
    }

    Ok(())
}
