//! parent module of the pipeline recording api
use std::fmt::Display;

use crate::{
    current_shader,
    pipeline::compute_pipeline_info::ComputePipelineInfo,
    rec::{AnyDowncast, DType, Shape},
    Ten,
};

use self::render_pipeline_info::{BindGroupInfo, PushConstantInfo, RenderPipelineInfo};

pub mod pipeline_io;
pub use pipeline_io::GenericPipelineIO;

pub mod blending;
pub mod culling;
pub mod pixel_format;
pub mod primitive_ext;
pub mod target;
pub mod topology;

pub mod compute_pipeline;
pub mod compute_pipeline_info;

pub mod render_pipeline;
pub mod render_pipeline_info;

pub(crate) fn with_thread_render_pipeline_info_mut<R>(f: impl FnOnce(&mut RenderPipelineInfo) -> R) -> R {
    shame_graph::Context::with(|ctx| {
        let mut misc = ctx.misc_mut();
        match misc.downcast_mut::<RenderPipelineInfo>() {
            Some(rp) => f(rp),
            None => panic!("trying to access render pipeline outside of render pipeline recording. Did you accidentially record render shaders, instead of a render pipeline?"),
        }
    })
}

pub(crate) fn with_thread_compute_pipeline_info_mut<R>(f: impl FnOnce(&mut ComputePipelineInfo) -> R) -> R {
    shame_graph::Context::with(|ctx| {
        let mut misc = ctx.misc_mut();
        match misc.downcast_mut::<ComputePipelineInfo>() {
            Some(rp) => f(rp),
            None => panic!("trying to access compute pipeline outside of render pipeline recording. Did you accidentially record render shaders, instead of a render pipeline?"),
        }
    })
}

pub(crate) fn with_thread_pipeline_bind_groups<R>(f: impl FnOnce(&mut Vec<BindGroupInfo>) -> R) -> R {
    shame_graph::Context::with(|ctx| {
        let mut misc = ctx.misc_mut();
        match misc.downcast_mut::<RenderPipelineInfo>() {
            Some(r) => f(&mut r.bind_groups),
            None => match misc.downcast_mut::<ComputePipelineInfo>() {
                Some(c) => f(&mut c.bind_groups),
                None => panic!("trying to access pipeline bind groups outside of a pipeline recording. Did you accidentially record render/compute shaders, instead of a render/compute pipeline?"),
            }
        }
    })
}

pub(crate) fn with_thread_pipeline_push_constant<R>(f: impl FnOnce(&mut Option<PushConstantInfo>) -> R) -> R {
    shame_graph::Context::with(|ctx| {
        let mut misc = ctx.misc_mut();
        match misc.downcast_mut::<RenderPipelineInfo>() {
            Some(r) => f(&mut r.push_constant),
            None => match misc.downcast_mut::<ComputePipelineInfo>() {
                Some(c) => f(&mut c.push_constant),
                None => panic!("trying to access pipeline push-constant outside of a pipeline recording. Did you accidentially record render/compute shaders, instead of a render/compute pipeline?"),
            }
        }
    })
}

fn record_groups_into_context() {
    use shame_graph::{Binding, OpaqueTy};

    let get_binding_type = |b: &Binding| {
        match b {
            Binding::Opaque(ty, _) => match ty {
                OpaqueTy::TextureCombinedSampler(_) => BindingType::TextureCombinedSampler,
                OpaqueTy::ShadowSampler(_)          => BindingType::ShadowSampler,
                OpaqueTy::Sampler                   => BindingType::Sampler,
                OpaqueTy::Texture(_)                => BindingType::Texture,
                OpaqueTy::Image(_) => panic!("Image bindings should not be represented through the Binding::Opaque but through the Binding::OpaqueImage enum variant"),
                OpaqueTy::AtomicCounter(_) => todo!("Atomic counter bindings not supported yet"),
            },
            Binding::UniformBlock(_) => BindingType::UniformBuffer,
            Binding::StorageMut(_)   => BindingType::ReadWriteStorageBuffer,
            Binding::Storage(_)      => BindingType::ReadOnlyStorageBuffer,
            Binding::OpaqueImage { is_read_only, .. } => {
                match is_read_only {
                    true => BindingType::ReadOnlyStorageImage,
                    false => BindingType::ReadWriteStorageImage,
                }
            },
        }
    };

    with_thread_pipeline_bind_groups(|bind_groups| {
        assert!(bind_groups.is_empty());

        shame_graph::Context::with(|ctx| {
            *bind_groups = ctx
                .shader()
                .side_effects
                .bind_groups()
                .iter()
                .map(|(group_i, group)| {
                    BindGroupInfo {
                        index: *group_i,
                        bindings: group
                            .bindings()
                            .iter()
                            .map(|(bind_i, bind)| {
                                BindingInfo {
                                    binding: *bind_i,
                                    binding_type: get_binding_type(bind),
                                    visibility: StageFlags::all_based_on_recording_kind(), //as of now stage visibility is not tracked
                                }
                            })
                            .collect(),
                    }
                })
                .collect();
        });
    })
}

fn instantiate_push_constant<S: Shape, D: DType>() -> Ten<S, D> {
    use shame_graph::Any;
    use shame_graph::ShaderKind::*;
    use shame_graph::Tensor;

    let tensor = Tensor::new(S::SHAPE, D::DTYPE);

    let visibility = StageFlags::all_based_on_recording_kind(); //TODO: we're currently not tracking where the push_constant is actually being used. improve this for better visibility flags (can reduce delays)

    with_thread_pipeline_push_constant(|p| {
        assert!(p.is_none(), "only one push_constant supported per pipeline"); //TODO: this could be handled more nicely, maybe calling this consumes self to assure its only called once
        *p = Some(PushConstantInfo {
            type_: tensor,
            visibility,
        });
    });

    shame_graph::Context::with(|ctx| {
        let needs_push_constant_decl = match current_shader() {
            Vertex if visibility.vertex => true,
            Fragment if visibility.fragment => true,
            Compute if visibility.compute => true,
            _ => false,
        };

        match needs_push_constant_decl {
            true => ctx
                .shader_mut()
                .side_effects
                .set_push_constant(tensor, Some("push_const".to_string())),
            false => Any::not_available(),
        }
        .downcast(crate::rec::Stage::Uniform)
    })
}

#[allow(missing_docs)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BindingType {
    /// A sampler (without an attached image).
    Sampler,
    /// A combination of Image and Sampler.
    TextureCombinedSampler,

    ShadowSampler,
    /// An image, without a sampler.
    Texture,

    /// A read-only uniform buffer
    UniformBuffer,
    /// A read-only uniform texel buffer
    UniformTexelBuffer,

    /// A read-write storage buffer
    ReadWriteStorageBuffer,
    /// A read-only storage buffer
    ReadOnlyStorageBuffer,

    /// A read-write storage texel buffer
    ReadWriteStorageTexelBuffer,
    /// A read-only storage texel buffer
    ReadOnlyStorageTexelBuffer,

    /// A read-write storage image
    ReadWriteStorageImage,
    /// A read-only storage image
    ReadOnlyStorageImage,
}

impl Display for BindingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_fmt(format_args!("{:?}", self)) }
}

/// relevant info about a binding inside a bindgroup for creating pipeline
/// layouts
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingInfo {
    /// glsl: `layout(binding = X)`
    pub binding: u32,
    /// type of binding, e.g. Uniform Buffer, Texture ...
    pub binding_type: BindingType,
    /// which shader stages this binding needs to be visible at
    pub visibility: StageFlags,
}

impl Display for BindingInfo {
    fn fmt(&self, format: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format.write_fmt(format_args!(
            "{} => {} {}",
            self.binding, self.visibility, self.binding_type
        ))?;
        Ok(())
    }
}

/// describe from which shader stages a bind group binding has to be visible
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct StageFlags {
    /// vertex shader
    pub vertex: bool,
    /// fragment shader
    pub fragment: bool,
    /// compute shader
    pub compute: bool,
}

impl std::fmt::Debug for StageFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Stages{{{}{}{} }}",
            if self.vertex { " Vertex" } else { "" },
            if self.fragment { " Fragment" } else { "" },
            if self.compute { " Compute" } else { "" },
        ))
    }
}

impl std::fmt::Display for StageFlags {
    fn fmt(&self, format: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // if you're wondering what
        // "vfc", "v--", "-f-", "--c", "vf-"
        // means your search probably got you here.
        // Its the Stage Visibility Flags for the vertex/fragment/compute stages
        // v-- means only vertex stage
        // vfc means all stages
        let v = if self.vertex { "v" } else { "-" };
        let f = if self.fragment { "f" } else { "-" };
        let c = if self.compute { "c" } else { "-" };
        format.write_fmt(format_args!("{v}{f}{c}"))
    }
}

impl StageFlags {
    /// all stage flags are set
    pub fn all() -> Self {
        StageFlags {
            vertex: true,
            fragment: true,
            compute: true,
        }
    }
    /// no stage flag is set
    pub fn none() -> Self {
        StageFlags {
            vertex: false,
            fragment: false,
            compute: false,
        }
    }
    /// vertex and fragment stage flags are set, no compute
    pub fn vertex_fragment() -> Self {
        StageFlags {
            vertex: true,
            fragment: true,
            compute: false,
        }
    }
    /// compute stage flag is set, no vertex or fragment
    pub fn compute() -> Self {
        StageFlags {
            vertex: false,
            fragment: false,
            compute: true,
        }
    }

    /// returns the stage flags based on the current shader recording kind
    ///
    /// - render shader/pipeline recording: vertex and fragment stage flags are set
    /// - compute shader/pipeline recording: compute stage flag is set
    pub fn all_based_on_recording_kind() -> Self {
        use shame_graph::ShaderKind::*;
        match crate::current_shader() {
            Vertex | Fragment => Self::vertex_fragment(),
            Compute => Self::compute(),
        }
    }
}
