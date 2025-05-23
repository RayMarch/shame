use super::{
    error::{WgslError, WgslErrorLevel},
    write_block_by_key,
    write_node::get_single_arg,
    WgslContext,
};
use crate::{
    frontend::any::render_io::Attrib,
    ir::pipeline::{LateRecorded, RecordedWithIndex},
};
use crate::frontend::any::render_io::FragmentSampleMethod;
use crate::frontend::any::render_io::FragmentSamplePosition;
use crate::frontend::any::render_io::Location;
use crate::{
    backend::{
        code_write_buf::CodeWriteSpan,
        wgsl::{write_node::write_node_by_key, write_type::write_type, WgslErrorKind},
    },
    call_info,
    common::pool::{Key, PoolRef},
    frontend::encoding::fill::{Fill, PickVertex},
    ir::{
        self,
        expr::*,
        pipeline::{ShaderStage, WipRenderPipelineDescriptor},
        recording::{Block, BlockKind, BodyKind, CallInfo, Context},
        Len, Node, SizedType, StoreType, Type,
    },
};
use std::fmt::Write;

/// these categories model the different codegen cases for builtin outputs/
/// interpolate expressions.
#[derive(Clone, Copy)]
enum ScopeCategory {
    /// whether the node was recorded outside of any block or function recordings
    ///
    /// yields
    /// - a `let identifier = arg;` line for builtin outputs
    /// - a `identifier,` line in the ouptut struct's constructor in the entry point fn return statement.
    EncodingScope,
    /// whether the node was recorded in a block recording but outside of any
    /// function recording
    ///
    /// yields
    /// - a `var identifier: ty;` line in the shader entry point function
    /// - a `identifier = arg0;` line at the node's code position
    /// - a `identifier,` line in the ouptut struct's constructor in the entry point fn return statement.
    Nested,
    /// whether the node was recorded inside a function recording block
    ///
    /// yields
    /// - a `var<private> identifier: ty;` line in module scope
    /// - a `identifier = arg0;` line at the node's code position
    /// - a `identifier,` line in the ouptut struct's constructor in the entry point fn return statement.
    FunctionRecording,
}

fn scope_category(block_key: Key<Block>, ctx: &Context) -> ScopeCategory {
    let blocks = ctx.pool();
    let block = &blocks[block_key];
    if block.kind == BlockKind::EntryPoint {
        return ScopeCategory::EncodingScope;
    }
    let found = Block::find_in_stack(block_key, &blocks, |b| b.kind == BlockKind::Body(BodyKind::Function));
    match found {
        None => ScopeCategory::Nested,
        Some(_) => ScopeCategory::FunctionRecording,
    }
}

#[derive(Clone, Copy)]
enum IOCategory {
    Input,
    Output,
}

const fn io_category(io: &ShaderIo) -> IOCategory {
    match io {
        ShaderIo::Builtin(x) => match x {
            BuiltinShaderIo::Get(_) => IOCategory::Input,
            BuiltinShaderIo::Set(_) => IOCategory::Output,
        },
        ShaderIo::Interpolate(_) => IOCategory::Output,
        ShaderIo::GetInterpolated(_) => IOCategory::Input,
        ShaderIo::WriteToColorTarget { slot } => IOCategory::Output,
        ShaderIo::GetVertexInput(attrib) => IOCategory::Input,
    }
}

const fn stage_prefix_pascal_case(stage: &ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Comp => "Comp",
        ShaderStage::Task => "Task",
        ShaderStage::Mesh => "Mesh",
        ShaderStage::Vert => "Vert",
        ShaderStage::Frag => "Frag",
    }
}

const fn builtin_io_name(io: BuiltinShaderIo) -> &'static str {
    match io {
        BuiltinShaderIo::Get(x) => match x {
            BuiltinShaderIn::VertexIndex => "vertex_index",
            BuiltinShaderIn::InstanceIndex => "instance_index",
            BuiltinShaderIn::Position => "position",
            BuiltinShaderIn::FrontFacing => "front_facing",
            BuiltinShaderIn::SampleIndex => "sample_index",
            BuiltinShaderIn::SampleMask => "sample_mask",

            BuiltinShaderIn::LocalInvocationIndex => "local_invocation_index",
            BuiltinShaderIn::LocalInvocationId => "local_invocation_id",
            BuiltinShaderIn::GlobalInvocationId => "global_invocation_id",
            BuiltinShaderIn::WorkgroupId => "workgroup_id",
            BuiltinShaderIn::NumWorkgroups => "num_workgroups",

            BuiltinShaderIn::SubgroupInvocationId => "subgroup_invocation_id",
            BuiltinShaderIn::SubgroupSize => "subgroup_size",
        },
        BuiltinShaderIo::Set(x) => match x {
            BuiltinShaderOut::Position => "position",
            BuiltinShaderOut::FragDepth => "frag_depth",
            BuiltinShaderOut::SampleMask => "sample_mask",
            BuiltinShaderOut::ClipDistances { count: _ } => "clip_distances",
        },
    }
}

fn write_interpolate_attribute(code: &mut CodeWriteSpan, method: FragmentSampleMethod) -> Result<(), WgslError> {
    use FragmentSampleMethod as Met;
    let sampling = match method {
        Met::Interpolated(_, s) => match s {
            FragmentSamplePosition::PixelCenter => "center",
            FragmentSamplePosition::Centroid => "centroid",
            FragmentSamplePosition::PerSample => "sample",
        },
        Met::Flat(p) => match p {
            PickVertex::First => "first",
            PickVertex::Either => "either",
        },
    };
    let interpolation = match method {
        Met::Interpolated(Fill::Perspective, _) => "perspective",
        Met::Interpolated(Fill::Linear, _) => "linear",
        Met::Flat(_) => "flat",
    };
    write!(code, " @interpolate({interpolation}, {sampling})")?;
    Ok(())
}

fn write_shader_io_attributes(code: &mut CodeWriteSpan, entry: &Entry, ctx: &WgslContext) -> Result<(), WgslError> {
    match &entry.io {
        ShaderIo::Builtin(x @ BuiltinShaderIo::Set(BuiltinShaderOut::Position)) => {
            match ctx.ctx.render_pipeline().deterministic_clip_pos.get() {
                Some((true, call_info)) => {
                    let code = &mut code.sub_span(*call_info);
                    write!(code, "@invariant ")?;
                    Ok(())
                }
                Some((false, _)) => Ok(()),
                None => Err(WgslErrorKind::MissingInfo("rasterizer accuracy/position invariant")
                    .at_level(call_info!(), WgslErrorLevel::InternalPleaseReport)),
            }?;
            write!(code, "@builtin({})", builtin_io_name(*x))?
        }
        ShaderIo::Builtin(x) => write!(code, "@builtin({})", builtin_io_name(*x))?,
        ShaderIo::Interpolate(loc) => write!(code, "@location({})", loc)?,
        ShaderIo::GetInterpolated(loc) => write!(code, "@location({})", loc)?,
        ShaderIo::WriteToColorTarget { slot } => write!(code, "@location({})", slot)?,
        ShaderIo::GetVertexInput(attrib) => write!(code, "@location({})", attrib.0)?,
    }
    match entry.fill {
        None | Some(FragmentSampleMethod::Interpolated(Fill::Perspective, FragmentSamplePosition::PixelCenter)) => {}
        Some(fill) => write_interpolate_attribute(code, fill)?,
    }
    Ok(())
}

fn write_shader_io_ident(code: &mut CodeWriteSpan, io: &ShaderIo) -> Result<(), WgslError> {
    match io {
        ShaderIo::Builtin(x) => write!(code, "sm_{}", builtin_io_name(*x))?,
        ShaderIo::Interpolate(loc) => write!(code, "sm_fill{}", loc)?,
        ShaderIo::GetInterpolated(loc) => write!(code, "fill{}", loc)?,
        ShaderIo::WriteToColorTarget { slot } => write!(code, "sm_color{}", slot)?,
        ShaderIo::GetVertexInput(attrib) => write!(code, "attr{}", attrib.0)?,
    }
    Ok(())
}

pub(super) fn write_shader_entry_point_attributes(
    code: &mut CodeWriteSpan,
    stage: ShaderStage,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    match stage {
        ShaderStage::Task | ShaderStage::Mesh => {
            return Err(WgslErrorKind::UnsupportedStage(stage)
                .at_level(ctx.ctx.latest_user_caller(), WgslErrorLevel::InternalPleaseReport));
        }
        ShaderStage::Comp => {
            writeln!(code, "@compute")?;
            match ctx.ctx.compute_pipeline_mut().thread_grid_size_within_workgroup.get() {
                None => {
                    return Err(WgslErrorKind::MissingInfo("workgroup grid size")
                        .at_level(code.location, WgslErrorLevel::InternalPleaseReport));
                }
                Some(([x, y, z], call_info)) => {
                    let mut code = code.sub_span(*call_info);
                    writeln!(&mut code, "@workgroup_size({x}, {y}, {z})")?;
                }
            }
        }
        ShaderStage::Vert => writeln!(code, "@vertex")?,
        ShaderStage::Frag => writeln!(code, "@fragment")?,
    }
    Ok(())
}

pub(super) fn write_shader_entry_point_and_io_defs(
    code: &mut CodeWriteSpan,
    stage: ShaderStage,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    let defs = prepare_io_definitions(stage, ctx)?;
    let indent = ctx.indent.current();

    let entry_point_fn_name = match stage {
        ShaderStage::Comp => "comp_main",
        ShaderStage::Task => "task_main",
        ShaderStage::Mesh => "mesh_main",
        ShaderStage::Vert => "vert_main",
        ShaderStage::Frag => "frag_main",
    };
    let stage_prefix = stage_prefix_pascal_case(&stage);

    // define sm_StageIn, sm_StageOut structs
    for (struct_fields, suffix) in [(&defs.struct_fields_in, "In"), (&defs.struct_fields_out, "Out")] {
        if !struct_fields.is_empty() {
            // empty structs are not allowed in wgsl
            writeln!(code, "struct sm_{stage_prefix}{suffix} {{")?;
            let indent = indent.deeper();
            for entry in struct_fields {
                write!(code, "{indent}")?;
                {
                    let code = &mut code.sub_span(entry.call_info);
                    write_shader_io_attributes(code, entry, ctx)?;
                    write!(code, " ")?;
                    if entry.is_supersampling_dummy {
                        write!(code, "sm_force_supersampling")?;
                    } else {
                        write_shader_io_ident(code, &entry.io)?;
                    }
                    write!(code, ": ")?;
                    write_type(code, entry.call_info, &entry.ty, ctx)?;
                }
                writeln!(code, ",")?;
            }
            drop(indent);
            writeln!(code, "}}")?;
        }
    }

    for var_private in [&defs.var_in_private, &defs.var_out_private] {
        for entry in var_private {
            {
                let code = &mut code.sub_span(entry.call_info);
                write!(code, "var<private> ")?;
                write_shader_io_ident(code, &entry.io)?;
                write!(code, ": ")?;
                write_type(code, entry.call_info, &entry.ty, ctx)?;
                write!(code, ";")?;
            }
            writeln!(code)?;
        }
    }

    write_shader_entry_point_attributes(code, stage, ctx)?;
    write!(code, "fn {entry_point_fn_name}(")?;
    if !defs.struct_fields_in.is_empty() {
        write!(code, "sm_in: sm_{stage_prefix}In")?;
    }
    write!(code, ") ")?;
    if !defs.struct_fields_out.is_empty() {
        write!(code, "-> sm_{stage_prefix}Out ")?;
    }
    writeln!(code, "{{")?;
    {
        // write var definitions for output variables that are only accessed
        // in the entry point function
        let indent = indent.deeper();
        for entry in &defs.var_out_entry_point {
            let code = &mut code.sub_span(entry.call_info);
            write!(code, "{indent}var ")?;
            write_shader_io_ident(code, &entry.io)?;
            write!(code, ": ")?;
            write_type(code, entry.call_info, &entry.ty, ctx)?;
            writeln!(code, ";")?;
        }

        // statements
        write_block_by_key(code, ctx.ctx.current_block(), stage.as_mask(), ctx)?;

        // return SmStageOut(out0, out1, ...);
        if !defs.struct_fields_out.is_empty() {
            // empty structs are not allowed in wgsl
            writeln!(code, "{indent}return sm_{stage_prefix}Out(")?;
            {
                let indent = indent.deeper();
                for entry in &defs.struct_fields_out {
                    write!(code, "{indent}")?;
                    match entry.is_supersampling_dummy {
                        false => write_shader_io_ident(&mut code.sub_span(entry.call_info), &entry.io)?,
                        true => write!(code, "0f")?,
                    }
                    writeln!(code, ",")?;
                }
            }
            writeln!(code, "{indent});")?;
        }
    }
    writeln!(code, "}}")?;
    Ok(())
}

#[derive(Clone)]
struct Entry {
    io: ShaderIo,
    ty: Type,
    fill: Option<FragmentSampleMethod>,
    io_category: IOCategory,
    scope: ScopeCategory,
    /// whether this entry was artificially added to enforce supersampling
    is_supersampling_dummy: bool,
    call_info: CallInfo,
}

#[derive(Default)]
pub(super) struct IoDefinitions {
    /// which io must be part of the input/output struct definition
    struct_fields_in: Vec<Entry>,
    struct_fields_out: Vec<Entry>,

    /// which outputs must be predeclared as a var, so that they can be written to
    /// from the scope that the writes happen in
    var_out_private: Vec<Entry>,
    var_out_entry_point: Vec<Entry>,

    /// which inputs must be written into private vars, so that they can be
    /// read from other functions
    var_in_private: Vec<Entry>,
}

/// enforcing sample-rate shading if supersampling.
///
/// see https://gist.github.com/RayMarch/b9c302155bd405d45ddd7740697485c4
fn push_supersampling_enforcing_entry_if_needed(
    rp: &WipRenderPipelineDescriptor,
    entries: &mut Vec<Entry>,
    stage: ShaderStage,
) -> Result<(), WgslError> {
    let Some((supersampling, call_info)) = rp.is_supersampling() else {
        return Ok(());
    };
    if supersampling {
        let is_sample_rate_shading_already_enabled = rp.interpolators.iter().any(|(terp, _)| {
            matches!(
                terp.method,
                FragmentSampleMethod::Interpolated(_, FragmentSamplePosition::PerSample)
            )
        });

        if !is_sample_rate_shading_already_enabled {
            let next_location = Location(
                entries
                    .iter()
                    .map(|e| match e.io {
                        ShaderIo::Interpolate(loc) | ShaderIo::GetInterpolated(loc) => *loc + 1,
                        _ => 0,
                    })
                    .max()
                    .unwrap_or(0),
            );
            let ty = Type::Store(StoreType::Sized(SizedType::Vector(Len::X1, ir::ScalarType::F32)));
            let fill = Some(FragmentSampleMethod::Interpolated(
                Fill::Perspective,
                FragmentSamplePosition::PerSample,
            ));
            let scope = ScopeCategory::EncodingScope;
            let is_supersampling_dummy = true;
            match stage {
                ShaderStage::Mesh | ShaderStage::Vert => entries.push(Entry {
                    io: ShaderIo::Interpolate(next_location),
                    io_category: IOCategory::Output,
                    ty,
                    fill,
                    is_supersampling_dummy,
                    scope,
                    call_info,
                }),
                ShaderStage::Frag => entries.push(Entry {
                    io: ShaderIo::GetInterpolated(next_location),
                    io_category: IOCategory::Input,
                    ty,
                    fill,
                    is_supersampling_dummy,
                    scope,
                    call_info,
                }),
                _ => (),
            }
        }
    }
    Ok(())
}

pub(super) fn prepare_io_definitions(stage: ShaderStage, ctx: &WgslContext) -> Result<IoDefinitions, WgslError> {
    let mut defs = IoDefinitions::default();
    let builtin_io = ctx.ctx.pipeline().special.builtin_io.borrow();
    let nodes = ctx.ctx.pool();

    for (io, (node_key, call_info)) in builtin_io.iter() {
        let node = &nodes[*node_key];
        if !node.stages.is_present(stage.into()) {
            continue;
        }
        let io = ShaderIo::Builtin(*io);
        let io_category = io_category(&io);
        let entry = Entry {
            io: io.clone(),
            ty: match io_category {
                IOCategory::Input => &node.ty,
                IOCategory::Output => &nodes[get_single_arg(node)?].ty,
            }
            .clone(),
            fill: None,
            io_category,
            is_supersampling_dummy: false,
            scope: scope_category(node.block, ctx.ctx),
            call_info: *call_info,
        };
        match entry.io_category {
            IOCategory::Input => {
                defs.struct_fields_in.push(entry.clone());
                match entry.scope {
                    ScopeCategory::EncodingScope => (),
                    ScopeCategory::Nested => (),
                    // if the input is accessed from another function
                    // that means the entry-point function's parameter is not
                    // accessible. Therefore a private var needs to be defined
                    ScopeCategory::FunctionRecording => defs.var_in_private.push(entry),
                }
            }
            IOCategory::Output => {
                defs.struct_fields_out.push(entry.clone());
                match entry.scope {
                    // outputs that are set directly in encoding scope will yield
                    // a `let` definition at the node's code location
                    // and therefore need no `var` definition
                    ScopeCategory::EncodingScope => (),
                    // if the write to the output happens in a nested block
                    // within the entry-point function, a local `var` definition is needed
                    // which is then visible from within the nested block for writing
                    // as well as the entry-point function's return statement, where the value
                    // is written into the output struct constructor.
                    ScopeCategory::Nested => defs.var_out_entry_point.push(entry),
                    // if an output is set from another function, a
                    // `var<private>` has to be defined so the output is accessible
                    // from that function as well was the entry point function
                    // return statement (where it is written into the output struct constructor)
                    ScopeCategory::FunctionRecording => defs.var_out_private.push(entry),
                }
            }
        }
    }
    match stage {
        ShaderStage::Comp => (),
        ShaderStage::Task => (),
        ShaderStage::Mesh => (),
        ShaderStage::Vert => {
            let render_pipeline = ctx.ctx.render_pipeline();
            for vertex_buffer in &render_pipeline.vertex_buffers {
                for attrib in &vertex_buffer.attribs {
                    let Attrib {
                        location,
                        offset,
                        format,
                    } = &**attrib;

                    let entry = Entry {
                        io: ShaderIo::GetVertexInput(*location),
                        ty: Type::Store(StoreType::Sized(format.type_in_shader())),
                        fill: None,
                        io_category: IOCategory::Input,
                        is_supersampling_dummy: false,
                        scope: ScopeCategory::EncodingScope,
                        call_info: attrib.call_info,
                    };
                    defs.struct_fields_in.push(entry);
                }
            }
            for (terp, call_info) in &render_pipeline.interpolators {
                let (len, stype) = terp.vec_ty;
                let entry = Entry {
                    io: ShaderIo::Interpolate(terp.location),
                    ty: Type::Store(StoreType::Sized(SizedType::Vector(len, stype))),
                    fill: Some(terp.method),
                    io_category: IOCategory::Output,
                    is_supersampling_dummy: false,
                    scope: ScopeCategory::EncodingScope,
                    call_info: *call_info,
                };
                defs.struct_fields_out.push(entry);
            }
            push_supersampling_enforcing_entry_if_needed(&render_pipeline, &mut defs.struct_fields_out, stage)?;
        }
        ShaderStage::Frag => {
            let render_pipeline = ctx.ctx.render_pipeline();
            for (terp, call_info) in &render_pipeline.interpolators {
                let (len, stype) = terp.vec_ty;
                let entry = Entry {
                    io: ShaderIo::GetInterpolated(terp.location),
                    ty: Type::Store(StoreType::Sized(SizedType::Vector(len, stype))),
                    io_category: IOCategory::Input,
                    fill: Some(terp.method),
                    is_supersampling_dummy: false,
                    scope: ScopeCategory::EncodingScope,
                    call_info: *call_info,
                };
                defs.struct_fields_in.push(entry);
            }
            push_supersampling_enforcing_entry_if_needed(&render_pipeline, &mut defs.struct_fields_in, stage)?;
            for color_target in &render_pipeline.color_targets {
                let Some(ty) = color_target.format.sample_type_as_sized_type() else {
                    return Err(WgslErrorKind::ColorTargetHasNoShaderType(color_target.format.clone())
                        .at_level(color_target.call_info, WgslErrorLevel::InternalPleaseReport));
                };
                let entry = Entry {
                    io: ShaderIo::WriteToColorTarget {
                        slot: color_target.index,
                    },
                    ty: ty.into(),
                    fill: None,
                    io_category: IOCategory::Input,
                    is_supersampling_dummy: false,
                    scope: ScopeCategory::EncodingScope,
                    call_info: color_target.call_info,
                };
                defs.struct_fields_out.push(entry.clone());
            }
        }
    }
    Ok(defs)
}

pub(super) fn write_shader_io_node(
    code: &mut CodeWriteSpan,
    io: &ShaderIo,
    node: &Node,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    match io_category(io) {
        IOCategory::Input => {
            match scope_category(node.block, ctx.ctx) {
                ScopeCategory::EncodingScope | ScopeCategory::Nested => {
                    // the entry point funciton argument `sm_in` is accessible
                    // so we just write its identifier
                    write!(code, "sm_in.")?;
                    write_shader_io_ident(code, io)?;
                }
                ScopeCategory::FunctionRecording => {
                    // the entry point function argument `sm_in` is not accessible
                    // instead a variable `var<private> sm_in_...` was declared
                    // and assigned at the entry point function preamble.
                    // we access that variable here
                    write_shader_io_ident(code, io)?;
                }
            }
        }
        IOCategory::Output => {
            // if we are in the encoding scope, then this is a `let output = arg`.
            // otherwise its only `output = arg`, because the `var output` must
            // have been defined in a scope further outside, so that it is
            // accessible by the return statement of this shader stage.
            match scope_category(node.block, ctx.ctx) {
                ScopeCategory::EncodingScope => write!(code, "let ")?,
                ScopeCategory::Nested | ScopeCategory::FunctionRecording => {}
            }
            write_shader_io_ident(code, io)?;
            write!(code, " = ")?;
            write_node_by_key(code, get_single_arg(node)?, false, ctx)?;
        }
    }
    Ok(())
}
