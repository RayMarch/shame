//! printing recording results with syntax highlighting

pub use shame_graph::prettify::*;

use crate::{
    ComputePipelineInfo, ComputePipelineRecording, RenderPipelineInfo, RenderPipelineRecording,
};

fn colorize_pipeline_info_string(out: &mut String, string: &str) {
    use Highlight::*;
    fn write(out: &mut String, [l, m, r]: [&str; 3], colors: [Highlight; 3]) {
        let [a, b, c] = colors.map(|h| Some(h.get_color()));
        set_color(out, a, true);
        *out += l;
        set_color(out, b, true);
        *out += m;
        set_color(out, c, true);
        *out += r;
    }
    for line in string.lines() {
        if let Some((l, r)) = line.split_once(':') {
            write(out, [l, ":", r], [Keyword, Punctuation, UppercaseIdent]);
        } else if let Some((l, r)) = line.split_once("=>") {
            write(out, [l, "=>", r], [Number, Punctuation, UppercaseIdent]);
        } else {
            *out += line;
        }
        *out += "\n";
    }
    set_color(out, None, true);
}

impl RenderPipelineInfo {
    /// converts `self` to a string and adds coloring for printing on the console
    pub fn to_string_colored(&self) -> String {
        let mut out = String::new();
        colorize_pipeline_info_string(&mut out, &self.to_string());
        out
    }
}

impl ComputePipelineInfo {
    /// converts `self` to a string and adds coloring for printing on the console
    pub fn to_string_colored(&self) -> String {
        let mut out = String::new();
        colorize_pipeline_info_string(&mut out, &self.to_string());
        out
    }
}

impl RenderPipelineRecording {
    /// converts `self` to a string and adds coloring for printing on the console
    pub fn to_string_colored(&self) -> String {
        let (vert, frag) = &self.shaders_glsl;

        let mut out = String::new();
        let title_color = Highlight::Preprocessor.get_color();
        set_color(&mut out, Some(title_color), true);
        out += "~~~ vertex shader ~~~\n";
        out += &syntax_highlight_glsl(&vert);

        set_color(&mut out, Some(title_color), true);
        out += "\n\n~~~ fragment shader ~~~\n";
        out += &syntax_highlight_glsl(&frag);

        set_color(&mut out, Some(title_color), true);
        out += "\n\n~~~ pipeline info ~~~\n";
        out += &self.info.to_string_colored();

        out
    }
}

impl ComputePipelineRecording {
    /// converts `self` to a string and adds coloring for printing on the console
    pub fn to_string_colored(&self) -> String {
        let mut out = String::new();
        let title_color = "#A26C79";
        set_color(&mut out, Some(title_color), true);
        out += "~~~ compute shader ~~~\n";
        out += &syntax_highlight_glsl(&self.shader_glsl);

        set_color(&mut out, Some(title_color), true);
        out += "\n\n~~~ pipeline info ~~~\n";
        out += &self.info.to_string_colored();

        out
    }
}
