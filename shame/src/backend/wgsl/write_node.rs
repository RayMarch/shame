use std::{fmt::Write, ops::Deref, rc::Rc};

use crate::{
    backend::{
        code_write_buf::CodeWriteSpan,
        wgsl::{
            write_builtin_templates::write_builtin_template_wrapper_fn_name,
            write_type::{write_sized_type, write_store_type},
        },
    },
    common::pool::Key,
    frontend::any::shared_io::{BindPath, SamplingMethod},
    ir::{
        self,
        expr::{
            Assign, Binding, BuiltinFn, Decomposition, Expr, FnRelated, Literal, Operator, PipelineIo,
            PushConstantsField, RefLoad, Show, TextureFn,
        },
        ir_type::{CanonName, TextureShape},
        recording::{CallInfo, FunctionDef, TemplateStructParams},
        Comp4, CompoundOp, HandleType, Len, Node, ScalarConstant, ScalarType, SizedType, StoreType, Type,
    },
};

use super::{
    error::{WgslError, WgslErrorLevel},
    write_shader_io::write_shader_io_node,
    write_type::write_type,
    WgslContext, WgslErrorKind,
};

pub(super) fn write_node_type_by_key(
    code: &mut CodeWriteSpan,
    key: Key<Node>,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    let node = &ctx.ctx.pool()[key];
    write_type(code, node.call_info, &node.ty, ctx)?;
    Ok(())
}

pub(super) fn write_node_by_key(
    code: &mut CodeWriteSpan,
    key: Key<Node>,
    ignore_ident: bool,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    write_node(code, &ctx.ctx.pool()[key], ignore_ident, ctx)
}

pub(super) fn write_node(
    code: &mut CodeWriteSpan,
    node: &Node,
    ignore_ident: bool,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    let mut code = code.sub_span(node.call_info());
    // if this node has had an identifier introduced for itself via a statement,
    // the identifier may be used unless explicitly ignored.
    match node.ident.filter(|_| !ignore_ident) {
        None => write_expr(&mut code, node, ctx)?,
        Some(ident) => write!(&mut code, "{}", &ctx.idents[ident])?,
    }
    Ok(())
}

fn write_expr(code: &mut CodeWriteSpan, node: &Node, ctx: &WgslContext) -> Result<(), WgslError> {
    match &node.expr {
        Expr::VarIdent(region) => match region.0.ident {
            Some(key) => write!(code, "{}", &ctx.idents[key])?,
            None => {
                return Err(WgslErrorKind::MissingIdent("variable")
                    .at_level(node.call_info, WgslErrorLevel::InternalPleaseReport));
            }
        },
        Expr::RefLoad(RefLoad) => {
            let arg = get_single_arg(node)?;
            write_node_by_key(code, arg, false, ctx)?;
        }
        Expr::Assign(assign) => write_assign(code, *assign, node, ctx)?,
        Expr::Literal(Literal(scalar)) => match scalar {
            ScalarConstant::F16(x) => {
                match f32::from(*x).classify() {
                    std::num::FpCategory::Nan => Err(WgslErrorKind::NaNUnsupported.at(node.call_info)),
                    std::num::FpCategory::Infinite => Err(WgslErrorKind::InfUnsupported.at(node.call_info)),
                    _ => Ok(write!(code, "{}h", f32::from(*x))?),
                }?;
            }
            ScalarConstant::F32(x) => {
                match x.classify() {
                    std::num::FpCategory::Nan => Err(WgslErrorKind::NaNUnsupported.at(node.call_info)),
                    std::num::FpCategory::Infinite => Err(WgslErrorKind::InfUnsupported.at(node.call_info)),
                    _ => Ok(write!(code, "{}f", x)?),
                }?;
            }
            ScalarConstant::F64(x) => {
                match x.classify() {
                    std::num::FpCategory::Nan => Err(WgslErrorKind::NaNUnsupported.at(node.call_info)),
                    std::num::FpCategory::Infinite => Err(WgslErrorKind::InfUnsupported.at(node.call_info)),
                    _ => Ok(write!(code, "{}d", x)?),
                }?;
                return Err(WgslErrorKind::F64Unsupported.at(node.call_info));
            }
            ScalarConstant::U32(x) => write!(code, "{x}u")?,
            ScalarConstant::I32(x) => write!(code, "{x}i")?,
            ScalarConstant::Bool(x) => code.write_str(match x {
                true => "true",
                false => "false",
            })?,
        },
        Expr::Operator(x) => write_operator(code, *x, node, ctx)?,
        Expr::FnRelated(x) => match x {
            FnRelated::Call(def) => {
                let funcs = ctx.ctx.pool();
                write!(code, "{}", &ctx.idents[funcs[*def].ident])?;
                write_paren_arg_list(code, node, ctx)?;
            }
            // parameters have been written as part of fn signature
            FnRelated::FnParamValue(_) | FnRelated::FnParamMemoryView(_, _) => (),
        },
        Expr::Decomposition(decomp) => {
            use Decomposition as D;
            match decomp {
                D::VectorIndexConst(i) | D::MatrixIndexConst(i) | D::ArrayIndexConst(i) => {
                    let arg = get_single_arg(node)?;
                    write_node_by_key(code, arg, false, ctx)?;
                    write!(code, "[{i}]")?;
                }
                D::ArrayIndex | D::VectorIndex | D::MatrixIndex => {
                    let [arg, i] = get_n_args(node)?;
                    write_node_by_key(code, arg, false, ctx)?;
                    write!(code, "[")?;
                    write_node_by_key(code, i, false, ctx)?;
                    write!(code, "]")?;
                }
                D::VectorAccess(swizzle) => {
                    let arg = get_single_arg(node)?;
                    write_node_by_key(code, arg, false, ctx)?;
                    write!(code, ".")?;
                    for component in &**swizzle {
                        code.write_char(match component {
                            Comp4::X => 'x',
                            Comp4::Y => 'y',
                            Comp4::Z => 'z',
                            Comp4::W => 'w',
                        });
                    }
                }
                D::StructureAccess(field_canon_name) => {
                    write_field_access(code, node, field_canon_name, ctx)?;
                }
            }
        }
        Expr::BuiltinFn(builtin_fn) => {
            write_builtin_fn(code, builtin_fn, node, ctx)?;
        }
        Expr::ShaderIo(io) => write_shader_io_node(code, io, node, ctx)?,
        Expr::PipelineIo(io) => match io {
            PipelineIo::Binding(Binding { bind_path, ty }) => {
                write_binding_ident_with_ty(code, node.call_info, *bind_path, ty)?;
            }
            PipelineIo::PushConstantsField(PushConstantsField { field_index, ty }) => {
                write!(code, "sm_pushc._{field_index}")?;
            }
        },
        Expr::Show(Show) => {
            write!(code, "/*must appear:*/")?;
            let arg = get_single_arg(node)?;
            write_node_by_key(code, arg, false, ctx)?
        }
    }
    Ok(())
}


pub(super) fn write_binding_ident_with_ty(
    code: &mut CodeWriteSpan,
    call_info: CallInfo,
    path: BindPath,
    ty: &Type,
) -> Result<(), WgslError> {
    match ty {
        Type::Unit => Err(WgslErrorKind::TypeMayNotAppearInWrittenForm(ty.clone()).at(call_info)),
        Type::Ptr(mem, sty, access) | Type::Ref(mem, sty, access) => write_binding_ident(code, path, sty),
        Type::Store(sty) => write_binding_ident(code, path, sty),
    }
}

pub(super) fn write_binding_ident(
    code: &mut CodeWriteSpan,
    BindPath(group, binding): BindPath,
    sty: &StoreType,
) -> Result<(), WgslError> {
    write!(code, "sm_g{group}_{binding}_")?;
    let access_shorthand = |access| match access {
        ir::AccessMode::Read => "r",
        ir::AccessMode::Write => "w",
        ir::AccessMode::ReadWrite => "rw",
    };
    let write_shape = |code: &mut CodeWriteSpan, shape| -> Result<(), WgslError> {
        code.write_str(match shape {
            TextureShape::_1D => "1d",
            TextureShape::_2D => "",
            TextureShape::_2DArray(non_zero) => "_arr",
            TextureShape::_3D => "3d",
            TextureShape::Cube => "_cube",
            TextureShape::CubeArray(non_zero) => "_cube_arr",
        });
        Ok(())
    };
    let write_store_ty = |code: &mut CodeWriteSpan, sty: &_| -> Result<(), WgslError> {
        match sty {
            StoreType::Sized(st) => code.write_str(match st {
                SizedType::Atomic(scalar_type_integer) => "atomic",
                _ => "buffer",
            })?,
            StoreType::Handle(handle) => match handle {
                HandleType::SampledTexture(shape, ty, spp) => {
                    code.write_str("tex")?;
                    write_shape(code, *shape)?;
                }
                HandleType::StorageTexture(shape, fmt, acc) => {
                    write!(code, "{}_tex", access_shorthand(*acc))?;
                    write_shape(code, *shape)?;
                }
                HandleType::Sampler(samp) => code.write_str(match samp {
                    SamplingMethod::Filtering => "filter",
                    SamplingMethod::NonFiltering => "nearest",
                    SamplingMethod::Comparison => "compar",
                })?,
            },
            StoreType::RuntimeSizedArray(sized_type) => code.write_str("buffer")?,
            StoreType::BufferBlock(buffer_block) => code.write_str("buffer")?,
        };
        Ok(())
    };
    write_store_ty(code, sty)?;
    Ok(())
}

pub(super) fn write_field_access(
    code: &mut CodeWriteSpan,
    node: &Node,
    field_name: &CanonName,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    let arg = get_single_arg(node)?;
    let ty = &ctx.ctx.pool()[arg].ty;

    let struct_ = match ty {
        Type::Store(StoreType::Sized(SizedType::Structure(s))) => Ok(s as &Rc<ir::Struct>),
        Type::Ref(_, StoreType::BufferBlock(s), _) => Ok(s as &Rc<ir::Struct>),
        Type::Ref(_, StoreType::Sized(SizedType::Structure(s)), _) => Ok(s as &Rc<ir::Struct>),
        _ => Err(WgslErrorKind::FieldAccessOnNonStruct(ty.clone(), field_name.clone())
            .at_level(node.call_info, WgslErrorLevel::InternalPleaseReport)),
    }?;

    let registry = ctx.ctx.struct_registry();
    let def = registry.get(struct_).ok_or_else(|| {
        WgslErrorKind::UnregisteredStruct(struct_.clone())
            .at_level(node.call_info, WgslErrorLevel::InternalPleaseReport)
    })?;

    let (ident, field) = def.get_field_by_name(field_name).ok_or_else(|| {
        WgslErrorKind::UnknownFieldForStruct(ty.clone(), field_name.clone())
            .at_level(node.call_info, WgslErrorLevel::InternalPleaseReport)
    })?;

    write_node_by_key(code, arg, false, ctx)?;
    write!(code, ".{}", &ctx.idents[*ident])?;
    Ok(())
}

pub(super) fn get_single_arg(node: &Node) -> Result<Key<Node>, WgslError> { get_n_args::<1>(node).map(|[arg]| arg) }

pub(super) fn get_n_args<const N: usize>(node: &Node) -> Result<[Key<Node>; N], WgslError> {
    TryFrom::<&[Key<Node>]>::try_from(&node.args).map_err(|_| {
        WgslErrorKind::InvalidAmountOfArguments {
            expr: node.expr.clone(),
            expected: N as u32,
            actual: node.args.len() as u32,
        }
        .at_level(node.call_info, WgslErrorLevel::InternalPleaseReport)
    })
}

fn write_assign(code: &mut CodeWriteSpan, assign: Assign, node: &Node, ctx: &WgslContext) -> Result<(), WgslError> {
    use CompoundOp as Op;
    let op_str = match assign {
        Assign::Increment => "++",
        Assign::Decrement => "--",
        Assign::Assign => "=",
        Assign::CompoundAssignment(c) => match c {
            Op::AddAssign => "+=",
            Op::SubAssign => "-=",
            Op::MulAssign => "*=",
            Op::DivAssign => "/=",
            Op::RemAssign => "%=",
            Op::AndAssign => "&=",
            Op::OrAssign => "|=",
            Op::XorAssign => "^=",
            Op::ShrAssign => ">>=",
            Op::ShlAssign => "<<=",
        },
    };

    match (assign, &*node.args) {
        (Assign::Assign | Assign::CompoundAssignment(_), [lhs, rhs]) => {
            write_node_by_key(code, *lhs, false, ctx)?;
            write!(code, " {op_str} ")?;
            write_node_by_key(code, *rhs, false, ctx)?;
        }
        (Assign::Increment | Assign::Decrement, [arg]) => {
            write_node_by_key(code, *arg, false, ctx)?;
            code.write_str(op_str)?;
        }
        _ => {
            let expected = match assign {
                Assign::Assign | Assign::CompoundAssignment(_) => 2,
                Assign::Increment | Assign::Decrement => 1,
            };
            return Err(WgslErrorKind::InvalidAmountOfArguments {
                expr: Expr::Assign(assign),
                expected,
                actual: node.args.len() as u32,
            }
            .at(node.call_info));
        }
    }
    Ok(())
}

fn write_operator(code: &mut CodeWriteSpan, op: Operator, node: &Node, ctx: &WgslContext) -> Result<(), WgslError> {
    use Operator as Op;
    let op_str = operator_str(op);
    write!(code, "(")?;
    match &*node.args {
        [a] => {
            write!(code, "{op_str}")?;
            write_node_by_key(code, *a, false, ctx)?;
        }
        [a, b] => {
            write_node_by_key(code, *a, false, ctx)?;
            write!(code, " {op_str} ")?;
            write_node_by_key(code, *b, false, ctx)?;
        }
        _ => {
            return Err(WgslErrorKind::InvalidAmountOfArguments {
                expr: Expr::Operator(op),
                expected: match op {
                    Op::Not | Op::Neg | Op::Complement | Op::AddressOf | Op::Indirection => 1,
                    Op::And |
                    Op::AndAnd |
                    Op::Or |
                    Op::OrOr |
                    Op::Xor |
                    Op::Add |
                    Op::Sub |
                    Op::Mul |
                    Op::Div |
                    Op::Rem |
                    Op::Shl |
                    Op::Shr |
                    Op::Equal |
                    Op::NotEqual |
                    Op::GreaterThan |
                    Op::LessThan |
                    Op::GeraterThanOrEqual |
                    Op::LessThanOrEqual => 2,
                },
                actual: node.args.len() as u32,
            }
            .at(node.call_info));
        }
    }
    write!(code, ")")?;
    Ok(())
}

const fn operator_str(op: Operator) -> &'static str {
    match op {
        Operator::Not => "!",
        Operator::Complement => "~",
        Operator::Neg => "-",
        Operator::And => "&",
        Operator::AndAnd => "&&",
        Operator::Or => "|",
        Operator::OrOr => "||",
        Operator::Xor => "^",
        Operator::Add => "+",
        Operator::Sub => "-",
        Operator::Mul => "*",
        Operator::Div => "/",
        Operator::Rem => "%",
        Operator::AddressOf => "&",
        Operator::Indirection => "*",
        Operator::Shl => "<<",
        Operator::Shr => ">>",
        Operator::Equal => "==",
        Operator::NotEqual => "!=",
        Operator::GreaterThan => ">",
        Operator::LessThan => "<",
        Operator::GeraterThanOrEqual => ">=",
        Operator::LessThanOrEqual => "<=",
    }
}

fn write_builtin_fn(
    code: &mut CodeWriteSpan,
    builtin_fn: &BuiltinFn,
    node: &Node,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    match builtin_fn {
        BuiltinFn::Texture(f) => write_builtin_texture_fn(code, f, node, ctx)?,
        _ => {
            write_builtin_fn_name(code, builtin_fn, node, ctx)?;
            write_paren_arg_list(code, node, ctx)?;
        }
    }
    Ok(())
}

fn write_builtin_texture_fn(
    code: &mut CodeWriteSpan,
    texture_fn: &TextureFn,
    node: &Node,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    let builtin_fn = BuiltinFn::Texture(*texture_fn);
    use TextureShape as TS;

    let texture_info = (|| {
        let nodes = ctx.ctx.pool();
        for key in &*node.args {
            match &nodes[*key].ty {
                Type::Unit => (),
                Type::Ptr(_, store, _) | Type::Ref(_, store, _) | Type::Store(store) => match &store {
                    StoreType::Handle(HandleType::SampledTexture(shape, st, _)) => return (Some(*shape), Some(*st)),
                    StoreType::Handle(HandleType::StorageTexture(shape, fmt, _)) => {
                        return (Some(*shape), fmt.sample_type());
                    }
                    _ => (),
                },
            }
        }
        (None, None)
    });

    match texture_fn {
        TextureFn::TextureGather {
            channel: component,
            uv_offset,
        } => {
            write_builtin_fn_name(code, &builtin_fn, node, ctx)?;
            write!(code, "(")?;
            if let Some(component) = component {
                write!(code, "{}, ", component.as_index())?;
            }
            write_arg_list(code, node, ctx)?;
            if let Some([u, v]) = uv_offset {
                write!(code, ", vec2<i32>({u}, {v})")?;
            }
            write!(code, ")")?;
        }
        TextureFn::TextureGatherCompare { uv_offset } => {
            write_builtin_fn_name(code, &builtin_fn, node, ctx)?;
            write!(code, "(")?;
            write_arg_list(code, node, ctx)?;
            if let Some([u, v]) = uv_offset {
                write!(code, ", vec2<i32>({u}, {v})")?;
            }
            write!(code, ")")?;
        }
        TextureFn::TextureSample { uvw_offset } |
        TextureFn::TextureSampleBias { uvw_offset } |
        TextureFn::TextureSampleCompare { uvw_offset } |
        TextureFn::TextureSampleCompareLevel { uvw_offset } |
        TextureFn::TextureSampleGrad { uvw_offset } |
        TextureFn::TextureSampleLevel { uvw_offset } => {
            write_builtin_fn_name(code, &builtin_fn, node, ctx)?;
            write!(code, "(")?;
            write_arg_list(code, node, ctx)?;
            let (shape, sample_ty) = texture_info();
            if let Some([u, v, w]) = uvw_offset {
                // write appropriate subset of uvw_offset
                match shape {
                    Some(TS::_1D) => write!(code, ", {u}")?,
                    Some(TS::_2D | TS::_2DArray(_)) => write!(code, ", vec2<i32>({u}, {v})")?,
                    Some(TS::_3D) => write!(code, ", vec3<i32>({u}, {v}, {w})")?,
                    _ => (),
                }
            }
            write!(code, ")")?;
        }
        TextureFn::TextureStore(shape) => {
            write_builtin_fn_name(code, &builtin_fn, node, ctx)?;
            write_paren_arg_list(code, node, ctx)?;
        }
        TextureFn::TextureDimensions |
        TextureFn::TextureNumLayers |
        TextureFn::TextureNumLevels |
        TextureFn::TextureNumSamples |
        TextureFn::TextureLoad(_, _) |
        TextureFn::TextureSampleBaseClampToEdge => {
            write_builtin_fn_name(code, &builtin_fn, node, ctx)?;
            write_paren_arg_list(code, node, ctx)?;
        }
    };
    Ok(())
}

fn write_builtin_fn_name(
    code: &mut CodeWriteSpan,
    builtin_fn: &BuiltinFn,
    node: &Node,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    use crate::ir::expr::DerivativeFn::*;
    use crate::ir::expr::*;
    use crate::ir::GradPrecision as Prec;
    match builtin_fn {
        BuiltinFn::Constructor(x) => {
            let sized_type = match x {
                Constructor::Default(t) => t.clone(),
                Constructor::Scalar(s) => SizedType::Vector(Len::X1, *s),
                Constructor::Vector(len, s) => SizedType::Vector((*len).into(), *s),
                Constructor::Matrix(cols, rows, s) => SizedType::Matrix(*cols, *rows, *s),
                Constructor::Array(e, c) => SizedType::Array(e.clone(), *c),
                Constructor::Structure(def) => SizedType::Structure(def.clone()),
            };
            write_sized_type(code, &sized_type, node.call_info, ctx)?
        }
        BuiltinFn::Reinterpret(x) => match x {
            ReinterpretFn::Bitcast(t) => {
                let (len, stype) = match &node.ty {
                    Type::Store(ir::StoreType::Sized(ir::SizedType::Vector(len, stype))) => Ok((*len, *stype)),
                    t => Err(WgslErrorKind::BitcastCannotReturnType(t.clone())
                        .at_level(node.call_info, WgslErrorLevel::InternalPleaseReport)),
                }?;

                code.write_str("bitcast<")?;
                // the type in the angle brackets is always the return type
                write_sized_type(code, &SizedType::Vector(len, stype), node.call_info, ctx)?;
                code.write_str(">")?;
            }
        },
        BuiltinFn::Logical(x) => code.write_str(match x {
            LogicalFn::All => "all",
            LogicalFn::Any => "any",
            LogicalFn::Select => "select",
        })?,
        BuiltinFn::Array(x) => code.write_str(match x {
            ArrayFn::ArrayLength => "arrayLength",
        })?,
        BuiltinFn::Numeric(x) => code.write_str(match x {
            NumericFn::TrigonometryFn(f) => match f {
                TrigonometryFn::Sin => "sin",
                TrigonometryFn::Sinh => "sinh",
                TrigonometryFn::Asin => "asin",
                TrigonometryFn::Asinh => "asinh",
                TrigonometryFn::Cos => "cos",
                TrigonometryFn::Cosh => "cosh",
                TrigonometryFn::Acos => "acos",
                TrigonometryFn::Acosh => "acosh",
                TrigonometryFn::Tan => "tan",
                TrigonometryFn::Tanh => "tanh",
                TrigonometryFn::Atan => "atan",
                TrigonometryFn::Atanh => "atanh",
                TrigonometryFn::Atan2 => "atan2",
                TrigonometryFn::Degrees => "degrees",
                TrigonometryFn::Radians => "radians",
            },
            NumericFn::LinearAlgebra(f) => match f {
                LinearAlgebraFn::Cross => "cross",
                LinearAlgebraFn::Determinant => "determinant",
                LinearAlgebraFn::Distance => "distance",
                LinearAlgebraFn::Dot => "dot",
                LinearAlgebraFn::FaceForward => "faceForward",
                LinearAlgebraFn::InverseSqrt => "inverseSqrt",
                LinearAlgebraFn::Length => "length",
                LinearAlgebraFn::Normalize => "normalize",
                LinearAlgebraFn::Reflect => "reflect",
                LinearAlgebraFn::Refract => "refract",
                LinearAlgebraFn::Transpose => "transpose",
                LinearAlgebraFn::Fma => "fma",
            },
            NumericFn::Discontinuity(f) => match f {
                DiscontinuityFn::Max => "max",
                DiscontinuityFn::Min => "min",
                DiscontinuityFn::Mix => "mix",
                DiscontinuityFn::Clamp => "clamp",
                DiscontinuityFn::Floor => "floor",
                DiscontinuityFn::Ceil => "ceil",
                DiscontinuityFn::Fract => "fract",
                DiscontinuityFn::Modf(generics) => {
                    return write_builtin_template_wrapper_fn_name(code, &TemplateStructParams::Modf(*generics));
                }
                DiscontinuityFn::Abs => "abs",
                DiscontinuityFn::Sign => "sign",
                DiscontinuityFn::Round => "round",
                DiscontinuityFn::Saturate => "saturate",
                DiscontinuityFn::Step => "step",
                DiscontinuityFn::Smoothstep => "smoothstep",
                DiscontinuityFn::Trunc => "trunc",
                DiscontinuityFn::QuantizeToF16 => "quantizeToF16",
            },
            NumericFn::Exponent(f) => match f {
                ExponentFn::Exp => "exp",
                ExponentFn::Exp2 => "exp2",
                ExponentFn::Log => "log",
                ExponentFn::Log2 => "log2",
                ExponentFn::Frexp(generics) => {
                    return write_builtin_template_wrapper_fn_name(code, &TemplateStructParams::Frexp(*generics));
                }
                ExponentFn::Ldexp => "ldexp",
                ExponentFn::Pow => "pow",
                ExponentFn::Sqrt => "sqrt",
            },
            NumericFn::Bit(f) => match f {
                BitFn::CountLeadingZeros => "countLeadingZeros",
                BitFn::CountOneBits => "countOneBits",
                BitFn::CountTrailingZeros => "countTrailingZeros",
                BitFn::ExtractBits => "extractBits",
                BitFn::FirstLeadingBit => "firstLeadingBit",
                BitFn::FirstTrailingBit => "firstTrailingBit",
                BitFn::InsertBits => "insertBits",
                BitFn::ReverseBits => "reverseBits",
            },
        })?,
        BuiltinFn::Derivative(x @ (Dpdx(p) | Dpdy(p) | Fwidth(p))) => write!(
            code,
            "{}{}",
            match x {
                Dpdx(_) => "dpdx",
                Dpdy(_) => "dpdy",
                Fwidth(_) => "fwidth",
            },
            match p {
                Prec::DonTCare => "",
                Prec::Coarse => "Coarse",
                Prec::Fine => "Fine",
            }
        )?,
        BuiltinFn::Texture(x) => code.write_str(match x {
            TextureFn::TextureDimensions => "textureDimensions",
            TextureFn::TextureNumLayers => "textureNumLayers",
            TextureFn::TextureNumLevels => "textureNumLevels",
            TextureFn::TextureNumSamples => "textureNumSamples",
            TextureFn::TextureGather { .. } => "textureGather",
            TextureFn::TextureGatherCompare { .. } => "textureGatherCompare",
            TextureFn::TextureLoad(_, _) => "textureLoad",
            TextureFn::TextureStore(_) => "textureStore",
            TextureFn::TextureSample { .. } => "textureSample",
            TextureFn::TextureSampleBias { .. } => "textureSampleBias",
            TextureFn::TextureSampleCompare { .. } => "textureSampleCompare",
            TextureFn::TextureSampleCompareLevel { .. } => "textureSampleCompareLevel",
            TextureFn::TextureSampleGrad { .. } => "textureSampleGrad",
            TextureFn::TextureSampleLevel { .. } => "textureSampleLevel",
            TextureFn::TextureSampleBaseClampToEdge => "textureSampleBaseClampToEdge",
        })?,
        BuiltinFn::Atomic(x) => code.write_str(match x {
            AtomicFn::AtomicLoad => "atomicLoad",
            AtomicFn::AtomicStore => "atomicStore",
            AtomicFn::AtomicReadModifyWrite(x) => match x {
                AtomicModify::Add => "atomicAdd",
                AtomicModify::Sub => "atomicSub",
                AtomicModify::Max => "atomicMax",
                AtomicModify::Min => "atomicMin",
                AtomicModify::And => "atomicAnd",
                AtomicModify::Or => "atomicOr",
                AtomicModify::Xor => "atomicXor",
            },
            AtomicFn::AtomicExchange => "atomicExchange",
            AtomicFn::AtomicCompareExchangeWeak(generics) => {
                return write_builtin_template_wrapper_fn_name(
                    code,
                    &TemplateStructParams::AtomicCompareExchangeWeak(*generics),
                );
            }
        })?,
        BuiltinFn::DataPacking(x) => code.write_str(match x {
            DataPackingFn::Pack4x8snorm => "pack4x8snorm",
            DataPackingFn::Pack4x8unorm => "pack4x8unorm",
            DataPackingFn::Pack4xI8 => "pack4xI8",
            DataPackingFn::Pack4xU8 => "pack4xU8",
            DataPackingFn::Pack4xI8Clamp => "pack4xI8Clamp",
            DataPackingFn::Pack4xU8Clamp => "pack4xU8Clamp",
            DataPackingFn::Pack2x16snorm => "pack2x16snorm",
            DataPackingFn::Pack2x16unorm => "pack2x16unorm",
            DataPackingFn::Pack2x16float => "pack2x16float",

            DataPackingFn::Unpack4x8snorm => "unpack4x8snorm",
            DataPackingFn::Unpack4x8unorm => "unpack4x8unorm",
            DataPackingFn::Unpack4xI8 => "unpack4xI8",
            DataPackingFn::Unpack4xU8 => "unpack4xU8",
            DataPackingFn::Unpack2x16snorm => "unpack2x16snorm",
            DataPackingFn::Unpack2x16unorm => "unpack2x16unorm",
            DataPackingFn::Unpack2x16float => "unpack2x16float",
        })?,
        BuiltinFn::Sync(x) => code.write_str(match x {
            SyncFn::StorageBarrier => "storageBarrier",
            SyncFn::WorkgroupBarrier => "workgroupBarrier",
            SyncFn::WorkgroupUniformLoad => "workgroupUniformLoad",
            SyncFn::TextureBarrier => "textureBarrier",
        })?,
    };
    Ok(())
}

/// for example `(foo, bar(), 3, 1.0f)`
fn write_paren_arg_list(code: &mut CodeWriteSpan, node: &Node, ctx: &WgslContext) -> Result<(), WgslError> {
    write!(code, "(")?;
    write_arg_list(code, node, ctx)?;
    write!(code, ")")?;
    Ok(())
}

/// for example `foo, bar(), 3, 1.0f`
fn write_arg_list(code: &mut CodeWriteSpan, node: &Node, ctx: &WgslContext) -> Result<(), WgslError> {
    for (arg_i, arg) in node.args.iter().enumerate() {
        if arg_i != 0 {
            write!(code, ", ")?;
        }
        write_node_by_key(code, *arg, false, ctx)?;
    }
    Ok(())
}

/// whether the expression, when written as-is with a semicolon behind it, counts
/// as a valid WGSL statement. (no `let _ =`)
///
/// `expr;`
/// this is only true for some exprs, including:
/// - `a = b` assignment: https://www.w3.org/TR/WGSL/#assignment
/// - `func(args...)` function calls: https://www.w3.org/TR/WGSL/#function-call-statement
/// - `i++` increment or decrement: https://www.w3.org/TR/WGSL/#increment-decrement
///
/// expressions that are only sometimes valid as a statement return false.
///
/// This property is relevant because WGSL does not support expression-statements
/// for every type of expression.
pub(super) fn is_guaranteed_valid_as_statement(expr: &Expr) -> bool {
    use ir::expr::*;
    match expr {
        Expr::VarIdent(var_ident) => false,
        Expr::Assign(assign) => match assign {
            Assign::Assign => true,
            Assign::CompoundAssignment(compound_op) => true,
            Assign::Increment | Assign::Decrement => true,
        },
        Expr::ShaderIo(shader_io) => match shader_io {
            ShaderIo::Builtin(x) => match x {
                BuiltinShaderIo::Get(i) => match i {
                    BuiltinShaderIn::VertexIndex |
                    BuiltinShaderIn::InstanceIndex |
                    BuiltinShaderIn::Position |
                    BuiltinShaderIn::FrontFacing |
                    BuiltinShaderIn::SampleIndex |
                    BuiltinShaderIn::SampleMask |
                    BuiltinShaderIn::LocalInvocationIndex |
                    BuiltinShaderIn::LocalInvocationId |
                    BuiltinShaderIn::GlobalInvocationId |
                    BuiltinShaderIn::WorkgroupId |
                    BuiltinShaderIn::NumWorkgroups |
                    BuiltinShaderIn::SubgroupInvocationId |
                    BuiltinShaderIn::SubgroupSize => false,
                },
                BuiltinShaderIo::Set(o) => match o {
                    BuiltinShaderOut::Position => true,
                    BuiltinShaderOut::FragDepth => true,
                    BuiltinShaderOut::SampleMask => true,
                    BuiltinShaderOut::ClipDistances { count: _ } => true,
                },
            },
            ShaderIo::GetVertexInput(_) => false,
            ShaderIo::GetInterpolated(_) => false,
            ShaderIo::Interpolate(_) => true,
            ShaderIo::WriteToColorTarget { slot: _ } => true,
        },
        Expr::PipelineIo(io) => match io {
            PipelineIo::Binding(_) => false,
            PipelineIo::PushConstantsField(_) => false,
        },
        Expr::RefLoad(ref_load) => false,
        Expr::Literal(literal) => false,
        Expr::Operator(operator) => false,
        Expr::FnRelated(fn_related) => match fn_related {
            FnRelated::FnParamValue(_) => false,
            FnRelated::FnParamMemoryView(_, _) => false,
            FnRelated::Call(_) => true,
        },
        Expr::Decomposition(x) => match x {
            Decomposition::VectorAccess(_) |
            Decomposition::VectorIndex |
            Decomposition::VectorIndexConst(_) |
            Decomposition::MatrixIndex |
            Decomposition::MatrixIndexConst(_) |
            Decomposition::ArrayIndex |
            Decomposition::ArrayIndexConst(_) |
            Decomposition::StructureAccess(_) => false,
        },
        Expr::BuiltinFn(builtin_fn) => match builtin_fn {
            BuiltinFn::Constructor(_) |
            BuiltinFn::Reinterpret(_) |
            BuiltinFn::Logical(_) |
            BuiltinFn::Array(_) |
            BuiltinFn::Numeric(_) |
            BuiltinFn::Derivative(_) |
            BuiltinFn::Atomic(_) |
            BuiltinFn::DataPacking(_) |
            BuiltinFn::Sync(_) => true,
            BuiltinFn::Texture(_) => true,
        },
        Expr::Show(Show) => false,
    }
}
