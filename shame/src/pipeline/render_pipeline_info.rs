//! pipeline info for creating render pipeline layouts

use super::BindingInfo;
use super::topology::{IndexDType, PrimitiveTopology};
use super::pixel_format::{ColorFormat, DepthFormat};
use super::target::DepthTest;
use super::blending::Blend;
use super::culling::Cull;
use super::StageFlags;
use std::fmt::Display;
use std::mem::take;

/// Additional info to the recorded render shaders, which is necessary to create
/// a render pipeline.
///
/// Members which are `None` or empty have not been recorded
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderPipelineInfo {
    /// face culling
    pub cull: Option<Cull>,
    /// depth test configuraiton
    pub depth_test: Option<DepthTest>,
    /// whether the depth buffer is written to
    pub depth_write: Option<bool>,
    /// the way in which consecutive indices are formed into triangles
    pub primitive_topology: Option<PrimitiveTopology>,
    /// datatype of individual index values in the index buffer
    pub index_dtype: Option<IndexDType>,
    /// describes vertex buffers and which attributes they contain
    pub vertex_buffers: Vec<VertexBufferInfo>,
    /// which color targets the render pipeline outputs to and how
    pub color_targets: Vec<ColorTargetInfo>,
    /// format of the depth or depth/stencil buffer
    pub depth_stencil_target: Option<DepthFormat>,
    /// bind group layout information of all attached bind groups
    pub bind_groups: Vec<BindGroupInfo>,
    /// push constant format
    pub push_constant: Option<PushConstantInfo>,
}

impl Display for RenderPipelineInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.push_constant {
            None => f.write_str("no push constant\n")?,
            Some(x) => f.write_fmt(format_args!("push constant: {x}\n"))?
        }
        match &self.bind_groups[..] {
            [] => f.write_str("no bind groups\n")?,
            s => {
                f.write_fmt(format_args!("bind groups:\n"))?;
                for x in s {
                    f.write_fmt(format_args!("{x}"))?;
                }
            }
        }
        match &self.vertex_buffers[..] {
            [] => f.write_str("no vertex input stream\n")?,
            s => {
                f.write_fmt(format_args!("vertex input stream:\n"))?;
                for (i, x) in s.iter().enumerate() {
                    f.write_fmt(format_args!("  vertex buffer {i}:\n"))?;
                    f.write_fmt(format_args!("{x}"))?;
                }
            }
        }
        if !self.color_targets.is_empty() || self.depth_stencil_target.is_some() {
            f.write_fmt(format_args!("render targets:\n"))?;
            for (i, x) in self.color_targets.iter().enumerate() {
                f.write_fmt(format_args!("  {i} => {x}"))?;
            }
            if let Some(x) = &self.depth_stencil_target {
                f.write_fmt(format_args!("  depth => {x:?}\n"))?;
            }

            match &self.color_targets[..] {
                [] => (),
                s => {
                    f.write_fmt(format_args!("color target blending:\n"))?;
                    for (i, x) in s.iter().enumerate() {
                        match x.blending {
                            Some(x) => {
                                f.write_fmt(format_args!("  {i} => rgb = {}\n", x.rgb))?;
                                f.write_fmt(format_args!("         a = {}\n", x.a))?;
                            },
                            None    => f.write_fmt(format_args!("  {i} => Opaque\n"))?,
                        }
                    }
                }
            }
        }
        else {
            f.write_str("no render targets\n")?;
        }

        match &self.depth_test {
            Some(x) => f.write_fmt(format_args!("depth-test: {x:?}\n"))?,
            None    => f.write_fmt(format_args!("depth-test: ?\n"))?,
        }
        match &self.depth_write {
            Some(x) => f.write_fmt(format_args!("depth-write: {x}\n"))?,
            None    => f.write_fmt(format_args!("depth-write: ?\n"))?,
        }
        match &self.primitive_topology {
            Some(x) => f.write_fmt(format_args!("primitive topology: {x:?}\n"))?,
            None    => f.write_fmt(format_args!("primitive topology: ?\n"))?,
        }
        match &self.index_dtype {
            Some(x) => f.write_fmt(format_args!("index type: {x:?}\n"))?,
            None    => f.write_fmt(format_args!("index type: ?\n"))?,
        }
        match &self.cull {
            Some(Cull::Off) => f.write_fmt(format_args!("face culling: disabled\n"))?,
            Some(x) => f.write_fmt(format_args!("face culling: {x:?}\n"))?,
            None    => f.write_fmt(format_args!("face culling: ?\n"))?,
        }
        Ok(())
    }
}

/// Description of a color rendertarget and how it is interacted with by a
/// render pipeline
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorTargetInfo {
    /// amount of color samples per pixel pixel of this rendertarget only
    /// certain values are allowed such as 1, 2, 4, 8, 16
    pub sample_count: u8,
    /// the way in which incoming fragment colors are applied to the existing
    /// pixel color (and alpha)
    pub blending: Option<Blend>,
    /// format of every pixel in this color target
    pub color_format: ColorFormat,
}

impl Display for ColorTargetInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?} {}x\n", self.color_format, self.sample_count))
    }
}

/// layout information about vertex attributes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeInfo {
    /// the shader location the attribute needs to be bound to
    pub location: u32,
    /// datatype of this vertex attribute
    /// (matrix types consume multiple attribute locations)
    pub type_: shame_graph::Tensor,
}

/// whether a given vertex attribute is provided per vertex or per instance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexStepMode {
    /// vertex buffer contains one value per vertex
    Vertex,
    /// vertex buffer contains one value per instance
    Instance,
}

/// description of one of the buffers that make up the vertex input stream
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexBufferInfo {
    /// whether a given vertex attribute is provided per vertex or per instance
    pub step_mode: VertexStepMode,
    /// interleaved vertex attributes of this vertex buffer
    pub attributes: Vec<AttributeInfo>,
}

impl Display for VertexBufferInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for a in &self.attributes {
            f.write_fmt(format_args!("    {} => {}\n", a.location, a.type_))?;
        }
        Ok(())
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindGroupInfo { //TODO: this belongs in a different module doesn't it?
    /// group index (`layout(set = index)` in glsl)
    pub index: u32,
    /// types and indices of bindings in this bind group
    pub bindings: Vec<BindingInfo>,
}

impl Display for BindGroupInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("  bind group {}:\n", self.index))?;
        for b in &self.bindings {
            f.write_fmt(format_args!("    {b}\n"))?;
        }
        Ok(())
    }
}

/// layout of push constant in the recorded pipeline
/// currently only a single tensor push constant is supported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushConstantInfo { //TODO: this belongs in a different module doesn't it?
    /// datatype of the push constant
    pub type_: shame_graph::Tensor,
    /// which shader stages the push constants are used by
    pub visibility: StageFlags,
}

impl Display for PushConstantInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{} {}", self.visibility, self.type_))?;
        Ok(())
    }
}

impl RenderPipelineInfo {

    pub(crate) fn merge_individual_stage_recordings(mut v: RenderPipelineInfo, mut f: RenderPipelineInfo) -> RenderPipelineInfo {
        //TODO: this is also a spot where i should do a different kind of error handling in the future maybe
        //first we move out the parts we expect to be different...
        let vertex_buffer        = (take(&mut v.vertex_buffers), take(&mut f.vertex_buffers));
        let color_target         = (take(&mut v.color_targets), take(&mut f.color_targets));
        let depth_stencil_target = (take(&mut v.depth_stencil_target), take(&mut f.depth_stencil_target));

        //then we compare the rest and assert on its equality
        assert!(v == f, "pipeline info was recorded differently in vertex vs fragment shader");

        let mut merged = v;
        merged.vertex_buffers = vertex_buffer.0;
        merged.color_targets = color_target.1;
        merged.depth_stencil_target = depth_stencil_target.1;
        merged
    }
}