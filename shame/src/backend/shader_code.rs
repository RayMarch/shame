use crate::{
    common::{format::num_digits, ignore_eq::IgnoreInEqOrdHash, prettify::*},
    ir::recording::CallInfo,
};
use std::{
    borrow::Cow,
    fmt::{Display, Write},
    ops::{Deref, Range},
    slice::SliceIndex,
};

use super::wgsl::syntax_highlight_wgsl;

/// A shader code string, as well was `self.origin_spans` meta information about
/// which characters in the string were created by which location in the rust code.
///
/// there are multiple useful formatting options available for debugging shaders:
/// ```
/// let code: ShaderCode = ...;
/// println!("{code}"); // prints code.string
/// println!("{code:?}"); // prints for every line of `code` the rust line of code that generated it
/// println!("{}", code.syntax_highlight_as_wgsl()); // prints code with WGSL syntax highlighting
/// ```
///
/// The `self.origin_spans` information can help finding the rust-code that caused issues
/// that happen after the generated code was handed down to other tools/compilers.
/// For example, you can parse line/column information provided by your
/// graphics-api shader compiler error messages and use it to automatically look up the
/// respective `self.origin_spans` entry to reveal the rust-code that generated the
/// erroneous code.
/// note: (please report any instances of ill-formed generated code, so we can improve this library)
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ShaderCode {
    /// shader code in the language chosen before recording the pipeline
    pub(super) string: String,
    /// maps byte ranges within the `self.string` to the rust-code location that
    /// is responsible for generating it.
    ///
    /// `origin_spans` do not contribute to `ShaderCode`'s
    /// `Eq`/`PartialEq`/`Ord`/`PartialOrd`/`Hash` implementations.
    /// This means hashes/ordering of shaders is not impacted by
    /// moving lines around in the rust code that generated the shader.
    pub(super) origin_spans: IgnoreInEqOrdHash<Vec<(Range<usize>, CallInfo)>>,
}

impl From<ShaderCode> for Cow<'static, str> {
    fn from(value: ShaderCode) -> Self { value.string.into() }
}

/// shader code, put in different enum variants based on the target language/bytecode
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum LanguageCode {
    /// webgpu shading language https://www.w3.org/TR/WGSL/
    Wgsl(ShaderCode), // SpirV(Rc<u32>, Vec<(Range<usize>, CallInfo)>)
}

impl Display for LanguageCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LanguageCode::Wgsl(code) => write!(f, "{code}"),
        }
    }
}

impl std::fmt::Debug for LanguageCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LanguageCode::Wgsl(code) => code.write_with_span_comments(true, f),
        }
    }
}

impl Deref for ShaderCode {
    type Target = str;

    fn deref(&self) -> &str { &self.string }
}

impl LanguageCode {
    /// shader code in the language chosen before recording the pipeline
    pub fn as_str(&self) -> &str {
        match self {
            LanguageCode::Wgsl(code) => &code.string,
        }
    }

    #[doc(hidden)]
    /// get origin spans as a slice
    ///
    /// origin spans map byte ranges within the code string to the rust-code
    /// location that is responsible for generating it.
    ///
    /// `origin_spans` do not contribute to `ShaderCode`'s
    /// `Eq`/`PartialEq`/`Ord`/`PartialOrd`/`Hash` implementations.
    /// This means hashes/ordering of shaders is not impacted by
    /// moving lines around in the rust code that generated the shader.
    /// unstable api, might replace the `Vec` with some sort of dict
    /// at some point.
    pub fn origin_spans_slice(&self) -> &[(Range<usize>, CallInfo)] {
        match self {
            LanguageCode::Wgsl(code) => &code.origin_spans,
        }
    }

    /// iterate over origin spans
    ///
    /// origin spans map byte ranges within the code string to the rust-code
    /// location that is responsible for generating it.
    ///
    /// `origin_spans` do not contribute to `ShaderCode`'s
    /// `Eq`/`PartialEq`/`Ord`/`PartialOrd`/`Hash` implementations.
    /// This means hashes/ordering of shaders is not impacted by
    /// moving lines around in the rust code that generated the shader.
    pub fn origin_spans(&self) -> impl Iterator<Item = (Range<usize>, CallInfo)> + use<'_> {
        match self {
            LanguageCode::Wgsl(code) => code.origin_spans.iter().cloned(),
        }
    }

    /// return the generated shader code as (inaccurately) syntax highlighted code.
    ///
    /// this function exists to improve basic readability, for more accurate
    /// syntax highlighting, please consider using a dedicated crate
    pub fn syntax_highlight(&self) -> String {
        match self {
            LanguageCode::Wgsl(code) => code.syntax_highlight_as_wgsl(),
        }
    }
}

impl ShaderCode {
    /// shader code in the language chosen before recording the pipeline
    pub fn as_str(&self) -> &str { &self.string }

    /// consumes self and returns the shader code as an owned [`String`]
    pub fn into_string(self) -> String { self.string }

    #[doc(hidden)]
    /// get origin spans as a slice
    ///
    /// origin spans map byte ranges within the code string to the rust-code
    /// location that is responsible for generating it.
    ///
    /// `origin_spans` do not contribute to `ShaderCode`'s
    /// `Eq`/`PartialEq`/`Ord`/`PartialOrd`/`Hash` implementations.
    /// This means hashes/ordering of shaders is not impacted by
    /// moving lines around in the rust code that generated the shader.
    /// unstable api, might replace the `Vec` with some sort of dict
    /// at some point.
    pub fn origin_spans_slice(&self) -> &[(Range<usize>, CallInfo)] { &self.origin_spans }

    /// iterate over origin spans
    ///
    /// origin spans map byte ranges within the code string to the rust-code
    /// location that is responsible for generating it.
    ///
    /// `origin_spans` do not contribute to `ShaderCode`'s
    /// `Eq`/`PartialEq`/`Ord`/`PartialOrd`/`Hash` implementations.
    /// This means hashes/ordering of shaders is not impacted by
    /// moving lines around in the rust code that generated the shader.
    pub fn origin_spans(&self) -> impl Iterator<Item = (Range<usize>, CallInfo)> + use<'_> {
        self.origin_spans.iter().cloned()
    }
}

impl Display for ShaderCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.string) }
}

impl std::fmt::Debug for ShaderCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.write_with_span_comments(true, f) }
}

impl ShaderCode {
    /// return the generated shader code as (inaccurately) syntax highlighted
    /// wgsl code.
    ///
    /// this function exists to improve basic readability, for more accurate
    /// syntax highlighting, please consider using a dedicated crate
    pub fn syntax_highlight_as_wgsl(&self) -> String { syntax_highlight_wgsl(&self.string) }

    /// return the generated shader code, interleaved with rust code location paths
    /// that describe which line was generated by which rust expression
    pub fn string_with_span_comments(&self, colored: bool) -> String {
        let mut s = String::new();
        self.write_with_span_comments(colored, &mut s);
        s
    }

    /// write the generated shader code, interleaved with rust code location paths
    /// that describe which line was generated by which rust expression
    pub fn write_with_span_comments<F: Write>(&self, colored: bool, f: &mut F) -> std::fmt::Result {
        let line_numbers_width = num_digits(self.string.lines().count() + 1);

        let use_256_color_mode = false;
        let color = |f_: &mut F, hex| match colored {
            true => set_color(f_, Some(hex), use_256_color_mode),
            false => Ok(()),
        };
        let reset = |f_: &mut F| match colored {
            true => set_color(f_, None, use_256_color_mode),
            false => Ok(()),
        };
        let write_line_number = |mut f_: &mut _, k: usize| -> std::fmt::Result {
            color(f_, "#508EE3")?;
            for _ in 0..(line_numbers_width.saturating_sub(num_digits(k))) {
                f_.write_char(' ')?;
            }
            write!(f_, "{k} |")?;
            reset(f_)?;
            Ok(())
        };
        let write_line_number_empty = |f_: &mut F, vertical: char| -> std::fmt::Result {
            color(f_, "#508EE3")?;
            for _ in 0..(line_numbers_width) {
                f_.write_char(' ')?;
            }
            write!(f_, " {vertical}")?;
            reset(f_)?;
            Ok(())
        };
        let write_n_chars = |f_: &mut F, n: usize, c: char| -> std::fmt::Result {
            for _ in 0..n {
                f_.write_char(c)?;
            }
            Ok(())
        };

        let code = self.string.as_str();
        let mut spans_for_line = Vec::new();

        for (line_num_zero, line) in self.string.lines().enumerate() {
            let line_start_i = line.as_bytes().as_ptr() as usize - self.string.as_bytes().as_ptr() as usize;
            let line_end_i = line_start_i + line.len();

            spans_for_line.clear();
            spans_for_line.extend(self.origin_spans.iter().filter_map(|(r, call_info)| {
                code.get(r.clone()).and_then(|slice| {
                    str_overlap(slice, line).then_some((
                        (r.start as isize - line_start_i as isize)..(r.end as isize - line_start_i as isize),
                        call_info,
                    ))
                })
            }));

            // we are assuming here that every char has the same console display width
            for (i, c) in line.char_indices().rev() {
                for (span, caller) in spans_for_line.iter().filter(|span| span.0.start == i as isize) {
                    let num_underlines = span.len().min(line.len() - i);
                    let full = num_underlines < span.len();

                    if full {
                        write_line_number_empty(f, '├')?;
                        write_n_chars(f, i, ' ')?;
                        color(f, "#508EE3")?;
                        writeln!(f, "--[ {caller} ]")?;
                        reset(f)?
                    }
                }
            }

            write_line_number(f, line_num_zero + 1)?;
            writeln!(f, "{line}")?;

            // we are assuming here that every char has the same console display width
            for (i, c) in line.char_indices().rev() {
                for (span, caller) in spans_for_line.iter().filter(|span| span.0.start == i as isize) {
                    let num_underlines = span.len().min(line.len() - i);
                    let full = num_underlines < span.len();

                    if !full {
                        write_line_number_empty(f, '|')?;
                        write_n_chars(f, i, ' ')?;
                        let underline_char = '-';
                        color(f, "#508EE3")?;
                        f.write_char('^')?;
                        write_n_chars(f, num_underlines.saturating_sub(1), underline_char)?;
                        writeln!(f, " {caller}")?;
                        reset(f)?
                    }
                }
            }
        }
        Ok(())
    }
}

fn str_overlap(a: &str, b: &str) -> bool {
    let a = a.as_bytes().as_ptr_range();
    let b = b.as_bytes().as_ptr_range();
    a.contains(&b.start) || b.contains(&a.start)
}

fn ptr_in_str<T>(ptr: *const T, slice: &str) -> bool { slice.as_bytes().as_ptr_range().contains(&(ptr as *const _)) }
