use std::cell::Cell;
use std::fmt::Write;
use std::ops::Range;

use super::shader_code::ShaderCode;
use crate::ir::recording::CallInfo;

pub type Result<T> = std::result::Result<T, std::fmt::Error>;
pub type Span = Range<usize>;

#[derive(Default, Clone)]
/// a string that keeps track of how many chars are inside it.
///
/// also implements an interface similar to `std::fmt::Write` but
/// returns `Span`s of the written text on every write operation
pub struct CountedString {
    full: String,
    /// precomputed `self.full.chars().count()`
    full_chars_count: usize,
    /// `write_*` calls write their chars into this `staging`, then
    /// its `.chars().count()` is calculated and added to `full_chars_count`.
    /// that way the entire `full` doesn't need to be iterated over on every write
    /// just to calculate the string's char count.
    staging: String,
}

impl CountedString {
    pub fn new() -> Self { Default::default() }

    /// like `.chars().count()` on a string, except the value is already known
    /// and requires no iteration/computation
    pub fn chars_count(&self) -> usize { self.full_chars_count }

    pub fn into_inner(self) -> String {
        debug_assert!(self.full_chars_count == self.full.chars().count());
        debug_assert!(self.staging.is_empty());
        self.full
    }

    fn consume_staging(&mut self) -> Result<()> {
        let start = self.full_chars_count;
        let end = start + self.staging.chars().count();
        self.full_chars_count = end;
        self.full.write_str(&self.staging)?;
        self.staging.clear();
        Ok(())
    }
}

impl Write for CountedString {
    fn write_str(&mut self, s: &str) -> Result<()> {
        self.staging.write_str(s)?;
        self.consume_staging()?;
        Ok(())
    }

    fn write_char(&mut self, c: char) -> Result<()> {
        self.full.push(c);
        self.full_chars_count += 1;
        Ok(())
    }

    fn write_fmt(&mut self, args: std::fmt::Arguments<'_>) -> Result<()> {
        self.staging.write_fmt(args)?;
        self.consume_staging()?;
        Ok(())
    }
}

#[derive(Default)]
pub struct CodeWriteBuf {
    str: CountedString,
    spans: Vec<(Span, CallInfo)>,
}

impl CodeWriteBuf {
    pub fn new() -> Self { Default::default() }

    pub fn finish(self) -> ShaderCode {
        ShaderCode {
            string: self.str.into_inner(),
            origin_spans: self.spans.into(),
        }
    }

    pub fn span(&mut self, call_info: CallInfo) -> CodeWriteSpan {
        let span_start = self.str.chars_count();
        CodeWriteSpan {
            root: self,
            span_start,
            same_location_as_parent: false,
            location: call_info,
        }
    }
}

impl CodeWriteSpan<'_> {
    pub fn sub_span(&mut self, call_info: CallInfo) -> CodeWriteSpan {
        let span_start = self.root.str.chars_count();
        CodeWriteSpan {
            root: self.root,
            span_start,
            same_location_as_parent: call_info == self.location,
            location: call_info,
        }
    }
}

/// everything written to an instance within its lifetime is wrapped in a span
pub struct CodeWriteSpan<'a> {
    root: &'a mut CodeWriteBuf,
    span_start: usize,
    same_location_as_parent: bool,
    pub(crate) location: CallInfo,
}

impl Drop for CodeWriteSpan<'_> {
    fn drop(&mut self) {
        // try to find spans that this span is just an extension of (= same call_info, their end == self's start)
        // heuristic: most extended predecessors are within the last 3 ranges. avoid scanning all ranges in every drop even though it would be correct.
        let search_window = 3;
        let mut is_extension = false;
        for prev in self.root.spans.iter_mut().rev().take(search_window) {
            let same_locations = self.location == prev.1;
            let prev_range = &mut prev.0;
            if same_locations && prev_range.end == self.span_start {
                // `self` is just an extension of the last span, so just extend
                // the last span's range
                prev_range.end = self.root.str.chars_count();
                is_extension = true;
            }
        }

        if !is_extension {
            // don't push the location twice if the enclosing (parent) scope
            // has the same location
            if !self.same_location_as_parent {
                self.root
                    .spans
                    .push((self.span_start..self.root.str.chars_count(), self.location))
            }
        }
    }
}

impl std::fmt::Write for CodeWriteSpan<'_> {
    fn write_str(&mut self, s: &str) -> Result<()> { self.root.str.write_str(s) }

    fn write_char(&mut self, c: char) -> Result<()> { self.root.str.write_char(c) }

    fn write_fmt(&mut self, args: std::fmt::Arguments<'_>) -> Result<()> { self.root.str.write_fmt(args) }
}

#[derive(Debug, Clone)]
pub struct IndentTracker {
    pub level: Cell<u16>,
    pub spaces_per_level: u16,
}

impl IndentTracker {
    pub fn deeper(&self) -> Indentation { Indentation::new(self, 1) }
    pub fn current(&self) -> Indentation { Indentation::new(self, 0) }
}

impl Default for IndentTracker {
    fn default() -> Self {
        Self {
            level: Cell::new(0),
            spaces_per_level: 4,
        }
    }
}

/// a RAII guard for adding indentation levels.
///
/// example:
/// ```no_run
/// writeln!(f, "{indent}struct Coords {{")?;
/// {
///     let indent = indent.deeper();
///     writeln!(f, "{indent}x: uint,")?;
///     writeln!(f, "{indent}y: uint,")?;
/// } //indent is back to how it was before
/// writeln!(f, "{indent}}}")?
/// ```
pub struct Indentation<'a> {
    depth_offset: u16,
    tracker: &'a IndentTracker,
}

impl Drop for Indentation<'_> {
    fn drop(&mut self) {
        let lv = &self.tracker.level;
        lv.set(lv.get().saturating_sub(self.depth_offset));
    }
}

impl<'a> Indentation<'a> {
    fn new(tracker: &'a IndentTracker, depth_offset: u16) -> Self {
        let lv = &tracker.level;
        lv.set(lv.get().saturating_add(depth_offset));
        Self { depth_offset, tracker }
    }

    pub fn deeper(&'a self) -> Indentation<'a> { Indentation::new(self.tracker, 1) }
}

impl std::fmt::Display for Indentation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = "                                                                ";
        let depth = (self.tracker.level.get() * self.tracker.spaces_per_level) as usize;
        let s = s.get(0..depth).unwrap_or(s);
        f.write_str(s)
    }
}
