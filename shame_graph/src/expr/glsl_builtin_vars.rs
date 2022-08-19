use super::*;
use crate::{
    context::{Context, ShaderKind},
    error::Error,
};

macro_rules! glsl_type_to_ty {
    ($inout: ident $tensor: ident) => {Ty:: $tensor ().into_access($inout.as_access())};

    ($inout: ident $tensor: ident [$($array_size: expr)?]) => {
        Ty::array(Ty:: $tensor (), [$($array_size)?].first().cloned()).into_access($inout.as_access())
    };
}

macro_rules! glsl_decl_builtin_var_enum {
    ($v:vis enum $enum_name: ident {
        $($inout: ident $tensor: ident $var_name: ident $([$($array_size: expr)?])?;)*
    }) => {
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone, Copy)]
        $v enum $enum_name {
            $($var_name),*
        }

        impl $enum_name {
            pub fn ty(&self) -> Ty {
                use $enum_name::*;
                match self {
                    $($var_name => glsl_type_to_ty!($inout $tensor $([$($array_size)?])?),)*
                }
            }

            pub fn glsl_str(&self) -> &'static str {
                use $enum_name::*;
                match self {
                    $($var_name => stringify!($var_name),)*
                }
            }
        }

    };
}

//https://www.khronos.org/opengl/wiki/Built-in_Variable_(GLSL)
glsl_decl_builtin_var_enum! {
    pub enum VertexVar {
        _in int gl_VertexID;       // only present when not targeting Vulkan
        _in int gl_InstanceID;     // only present when not targeting Vulkan
        _in int gl_VertexIndex;    // only present when targeting Vulkan
        _in int gl_InstanceIndex;  // only present when targeting Vulkan

        out vec4 gl_Position;
        out float gl_PointSize;
        out float gl_ClipDistance[];
    }
}

glsl_decl_builtin_var_enum! {
    pub enum FragmentVar {
        _in vec4 gl_FragCoord;
        _in bool gl_FrontFacing;
        _in vec2 gl_PointCoord;

        _in int gl_SampleID;
        _in vec2 gl_SamplePosition; //any usage of this will force per-sample evaluation
        _in int gl_SampleMaskIn[];  //any usage of this will force per-sample evaluation

        _in float gl_ClipDistance[];
        _in int gl_PrimitiveID;

        out float gl_FragDepth;
    }
}

glsl_decl_builtin_var_enum! {
    pub enum ComputeVar {
        _in uvec3 gl_NumWorkGroups;
        _in uvec3 gl_WorkGroupID;
        _in uvec3 gl_LocalInvocationID;
        _in uvec3 gl_GlobalInvocationID;
        _in uint  gl_LocalInvocationIndex;

        _const uvec3 gl_WorkGroupSize;
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BuiltinVar {
    VertexVar(VertexVar),
    FragmentVar(FragmentVar),
    ComputeVar(ComputeVar),
}

impl BuiltinVar {
    pub fn shader_kind(&self) -> ShaderKind {
        match self {
            BuiltinVar::VertexVar(_) => ShaderKind::Vertex,
            BuiltinVar::FragmentVar(_) => ShaderKind::Fragment,
            BuiltinVar::ComputeVar(_) => ShaderKind::Compute,
        }
    }

    pub fn glsl_str(&self) -> &'static str {
        match self {
            BuiltinVar::VertexVar(x) => x.glsl_str(),
            BuiltinVar::FragmentVar(x) => x.glsl_str(),
            BuiltinVar::ComputeVar(x) => x.glsl_str(),
        }
    }
}

pub fn try_deduce_builtin_var(kind: &super::BuiltinVar, args: &[Ty]) -> Result<Ty, Error> {
    if !args.is_empty() {
        return Err(invalid_arguments(kind, args));
    }

    let ctx_shader = Context::with(|ctx| ctx.shader_kind);
    match (kind.shader_kind(), ctx_shader) {
        (expected, found) if expected != found => Err(Error::NAInShaderKind { expected, found }),
        _ => match kind {
            BuiltinVar::VertexVar(x) => Ok(x.ty()),
            BuiltinVar::FragmentVar(x) => Ok(x.ty()),
            BuiltinVar::ComputeVar(x) => Ok(x.ty()),
        },
    }
}

#[allow(non_camel_case_types)]
enum InOut {
    _in,
    out,
    _const,
}

impl InOut {
    pub fn as_access(&self) -> Access {
        match self {
            InOut::_in => Access::Const,
            InOut::out => Access::WriteOnly,
            InOut::_const => Access::Const,
        }
    }
}
use InOut::*;
