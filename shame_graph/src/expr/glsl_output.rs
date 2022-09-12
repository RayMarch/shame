
use std::{cell::Cell, fmt::Display};
use super::glsl_invalid_idents::GLSL_WORDS_THAT_CANNOT_BE_IDENTIFIERS;
use crate::{any::Any, common::IteratorExt, context::*, error::Error, pool::Key};
use super::*;

fn arg_list_to_glsl(ex: &State, args: &[Key<Expr>]) -> String {
    args.iter()
    .map(|expr_key| expr_key_to_glsl(ex, *expr_key, false))
    .collect::<Vec<_>>()
    .join(", ")
}

fn operator_to_glsl(ex: &State, op: &Operator, args: &[Key<Expr>]) -> String {
    assert!(args.len() == op.argc);
    let op_str = op.glsl_str;

    let binds_stronger_than = |op: &Operator, sub: Key<Expr>, is_left_of_sub: bool,| {

        let sub_expr = &ex.ctx.exprs()[sub];
        let has_binding = sub_expr.ident.is_some();

        match (has_binding, &sub_expr.kind) {
            (false, ExprKind::Operator(sub)) => {

                //while we're here lets check that lhs lvalues do have an identifier (even though this check is unrelated to precedence)
                if sub.lhs_lvalue && is_left_of_sub {
                    assert!(matches!(sub_expr.ident, Some(_)), "left hand side expression of operator{} doesn't have a binding", sub.glsl_str);
                }

                match op.glsl_prec == sub.glsl_prec {
                    true => {
                        debug_assert!(op.glsl_assoc == sub.glsl_assoc, "same precedence must imply same associativity");
                        match sub.glsl_assoc {
                            LeftToRight =>  is_left_of_sub,
                            RightToLeft => !is_left_of_sub,
                        }
                    },
                    false => op.glsl_prec < sub.glsl_prec,
                }
            }
            _ => false,
        }
    };

    let wrap_if = |cond, x| if cond {format!("({})", x)} else {x};
    use crate::Associativity::*;

    match &args {
        [a] => {
            let is_left_of_sub = op.glsl_assoc == RightToLeft;
            let a = wrap_if(binds_stronger_than(op, *a, is_left_of_sub), expr_key_to_glsl(ex, *a, false));

            match op.glsl_assoc {
                LeftToRight => format!("{a}{op_str}"), //postfix
                RightToLeft => format!("{op_str}{a}"), //prefix
            }
        },
        [a, b] => {
            let a = wrap_if(binds_stronger_than(op, *a, false), expr_key_to_glsl(ex, *a, false));
            let b = wrap_if(binds_stronger_than(op, *b, true), expr_key_to_glsl(ex, *b, false));

            match op {
                Operator::Subscript => format!("{a}[{b}]"),
                _ => format!("{a} {op_str} {b}")
            }
        },
        [a, b, c] => {
            let a = wrap_if(ex.ctx.exprs()[*a].ident.is_none(), expr_key_to_glsl(ex, *a, false));
            let b = wrap_if(ex.ctx.exprs()[*b].ident.is_none(), expr_key_to_glsl(ex, *b, false));
            let c = wrap_if(ex.ctx.exprs()[*c].ident.is_none(), expr_key_to_glsl(ex, *c, false));
            format!("{a} {op_str} {b} : {c}")
        },
        _ => panic!("unsupported glsl operator {:?} with {} arguments", op, args.len()),
    }
}

fn swizzle_to_glsl(ex: &State, sw: &Swizzle, args: &[Key<Expr>]) -> String {
    use Swizzle::*;
    let f = |comp: &u8| match comp {
        0 => "x",
        1 => "y",
        2 => "z",
        3 => "w",
        _ => panic!("{} is invalid", sw)
    };

    let expr_str = match args {
        [arg] => expr_key_to_glsl(ex, *arg, false),
        _ => panic!("invalid amount of expression arguments ({}) for swizzle operation", args.len())
    };

    match sw {
        GetVec4([x, y, z, w]) => format!("{}.{}{}{}{}", expr_str, f(x), f(y), f(z), f(w)),
        GetVec3([x, y, z])    => format!("{}.{}{}{}",   expr_str, f(x), f(y), f(z)),
        GetVec2([x, y])       => format!("{}.{}{}",     expr_str, f(x), f(y)),
        GetScalar([x])        => format!("{}.{}",       expr_str, f(x)),
    }
}

impl FloatingPoint for f32 {fn get_fp_category(&self) -> std::num::FpCategory {self.classify()}}
impl FloatingPoint for f64 {fn get_fp_category(&self) -> std::num::FpCategory {self.classify()}}
trait FloatingPoint: Display {
    fn get_fp_category(&self) -> std::num::FpCategory;
}


fn floating_point_to_glsl<T: FloatingPoint>(t: &T) -> String {
    use std::num::FpCategory::*;
    match t.get_fp_category() {
        Normal | Zero => {
            let mut s = format!("{}", t);
            if !s.contains('.') {s += ".";}
            s
        },
        cat => panic!("cannot convert floating point number of category '{:?}' to glsl", cat)
    }
}

fn copy_expr_to_glsl(ex: &State, expr: &Expr, comment: &str) -> String {
    // the copy expression is just the identity here.
    // in `Expr::needs_variable_def_stmt` it is enforced that copy expressions will
    // result in a variable definition statement, which performs the copy implicitly.
    // this allows us to copy types which have no copy constructor in glsl.
    // while glsl types like float or ivec3 have constructors like `float(foo.x)` that would
    // satisfy our needs for an identity operation that gets rid of lvalues, such a constructor
    // is not available for all glsl types that can be copied via defining a new variable.
    match expr.args.as_slice() {
        [arg] => if comment.is_empty() {
            format!("{}", expr_key_to_glsl(ex, *arg, false))
        } else {
            format!("{}/*{comment}*/", expr_key_to_glsl(ex, *arg, false))
        },
        _ => panic!("trying to generate glsl of a copy expression with {} arguments, only 1 argument allowed.", expr.args.len())
    }
}

fn expr_kind_to_glsl(ex: &State, expr: &Expr) -> String {
    match &expr.kind {
        ExprKind::GlobalInterface(_) => panic!("global interface expr without identifier is invalid"),
        ExprKind::Copy {comment} => copy_expr_to_glsl(ex, expr, comment),
        ExprKind::Literal(x) => match x {
            Literal::Bool(x) => format!("{}", x),
            Literal::F32 (x) => floating_point_to_glsl(x),
            Literal::F64 (x) => floating_point_to_glsl(x) + "lf",
            Literal::I32 (x) => format!("{}", x),
            Literal::U32 (x) => format!("{}u", x),
        },
        ExprKind::Constructor(x) => match x {
            Constructor::Tensor(x) => {
                let arglist = arg_list_to_glsl(ex, &expr.args);
                let prefix = || dtype_to_glsl_prefix(x.dtype);
                match x.shape {
                    Shape::Scalar => format!("{}({})", dtype_to_glsl(x.dtype), arglist),
                    Shape::Vec(n) => format!("{}vec{}({})", prefix(), n, arglist),
                    Shape::Mat(m, n) => match m == n {
                        true => format!("{}mat{}({})",    prefix(),    n, arglist),
                        false => format!("{}mat{}x{}({})", prefix(), m, n, arglist),
                    },
                }
            }
            Constructor::Struct(x) => {
                let args = expr.args.iter()
                .map(|expr_key| expr_key_to_glsl(ex, *expr_key, false));

                let Struct(Named(fields, ident)) = x;
                let ident = ex.valid_ident(**ident);

                let field_idents = fields.iter()
                .map(|Named(_, id)| ex.valid_ident(**id));

                let arglist = ex.with_deeper_indent(|| {
                    let indent = ex.indent_string();

                    args.zip(field_idents)
                    .map(|(arg, ident)| format!("\n{indent}/*{ident}*/ {arg}"))
                    .collect::<Vec<_>>()
                    .join(", ")

                });
                let indent = ex.indent_string();
                format!("{ident}({arglist}\n{indent})")
            }
            Constructor::Array(Array(ty, maybe_len)) => {
                let arglist = arg_list_to_glsl(ex, &expr.args);
                let ty_str = ty_to_glsl(ex, ty);
                match maybe_len {
                    Some(len) => format!("{ty_str}[{len}]({arglist})"),
                    None => format!("{ty_str}[]({arglist})"),
                }
            },
            Constructor::TextureCombinedSampler(TexDtypeDimensionality(dtype, dims)) => {
                let arglist = arg_list_to_glsl(ex, &expr.args);
                let ty_str = ty_to_glsl(ex, &Ty::texture_combined_sampler(*dtype, *dims));
                format!("{ty_str}({arglist})")
            },
        },
        ExprKind::Swizzle(x) => swizzle_to_glsl(ex, x, &expr.args),
        ExprKind::FieldSelect(x) => field_select_to_glsl(ex, x, &expr.args),
        ExprKind::Operator(x) => operator_to_glsl(ex, x, &expr.args),
        ExprKind::BuiltinFn(x) => format!("{}({})", x.glsl_str, arg_list_to_glsl(ex, &expr.args)),
        ExprKind::BuiltinVar(x) => x.glsl_str().to_string(),
    }
}

fn field_select_to_glsl(ex: &State, field: &IdentSlot, args: &[Key<Expr>]) -> String {
    match args {
        [arg] => {

            let expr = expr_key_to_glsl(ex, *arg, false);
            let field_ident = ex.valid_ident(field.0);

            //add parentheses around expr if its an operator
            match ex.ctx.exprs()[*arg].kind {
                ExprKind::Operator(_) => format!("({expr}).{field_ident}"),
                _ => format!("{expr}.{field_ident}")
            }
        }
        _ => panic!("trying to generate glsl of a field selection with {} arguments, only 1 argument allowed.", args.len())
    }
}

fn flow_to_glsl(ex: &State, flow: &Flow) -> String {
    let blk = |block_key: &Key<Block>| block_key_to_glsl(ex, *block_key);
    let exp = |expr_key : &Key<Expr >|  expr_key_to_glsl(ex, *expr_key, false);
    let blocks = ex.ctx.blocks();
    match flow {
        Flow::IfThen { cond, then } => {
            assert!(matches!(blocks[*then].kind, BlockKind::Body));
            format!("if ({}) {}", exp(cond), blk(then))
        }
        Flow::IfThenElse { cond, then, els } => {
            assert!(matches!(blocks[*then].kind, BlockKind::Body));
            assert!(matches!(blocks[* els].kind, BlockKind::Body));
            format!("if ({}) {} else {}", exp(cond), blk(then), blk(els))
        }
        Flow::For { init, cond, inc, body } => {
            //TODO: parse don't validate. the block kind should be enforced at `Flow` creation instead
            assert!(matches!(blocks[*init].kind, BlockKind::LoopInit));
            assert!(matches!(blocks[*cond].kind, BlockKind::LoopCondition(Some(_))));
            assert!(matches!(blocks[* inc].kind, BlockKind::LoopIncrement));
            assert!(matches!(blocks[*body].kind, BlockKind::LoopBody));
            format!("for ({}; {}; {}) {}", blk(init), blk(cond), blk(inc), blk(body))
        }
        Flow::While { cond, body } => {
            //TODO: parse don't validate. the block kind should be enforced at `Flow` creation instead
            assert!(matches!(blocks[*body].kind, BlockKind::LoopBody));
            format!("if ({}) {}", blk(cond), blk(body))
        }
    }
}

fn dtype_to_glsl_prefix(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "b",
        DType::F32  => "",
        DType::F64  => "d",
        DType::I32  => "i",
        DType::U32  => "u",
    }
}

fn tex_dimensionality_to_glsl_suffix(kind: TexDimensionality) -> &'static str {
    match kind {
        TexDimensionality::Tex1d => "1D",
        TexDimensionality::Tex2d => "2D",
        TexDimensionality::Tex3d => "3D",
        TexDimensionality::TexCubeMap => "Cube",
        TexDimensionality::TexRectangle => "2DRect",
        TexDimensionality::Tex1dArray => "1DArray",
        TexDimensionality::Tex2dArray => "2DArray",
        TexDimensionality::TexCubeMapArray => "CubeArray",
        TexDimensionality::TexBuffer => "Buffer",
        TexDimensionality::Tex2dMultisample => "2DMS",
        TexDimensionality::Tex2dMultisampleArray => "2DMSArray",
    }
}

fn shadow_sampler_kind_to_glsl_suffix(kind: ShadowSamplerKind) -> &'static str {
    match kind {
        ShadowSamplerKind::Tex1d => "1D",
        ShadowSamplerKind::Tex2d => "2D",
        ShadowSamplerKind::TexCubeMap => "Cube",
        ShadowSamplerKind::TexRectangle => "2DRect",
        ShadowSamplerKind::Tex1dArray => "1DArray",
        ShadowSamplerKind::Tex2dArray => "2DArray",
        ShadowSamplerKind::TexCubeMapArray => "CubeArray",
    }
}

fn dtype_to_glsl(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::F32  => "float",
        DType::F64  => "double",
        DType::I32  => "int",
        DType::U32  => "uint",
    }
}

fn ty_to_glsl(ex: &State, ty: &Ty) -> String {
    match &ty.kind {
        TyKind::Void => "void".to_string(),
        TyKind::Tensor(Tensor{dtype, shape}) => match shape {
            Shape::Scalar => dtype_to_glsl(*dtype).to_string(),
            Shape::Vec(n) => format!("{}vec{}", dtype_to_glsl_prefix(*dtype), n),
            Shape::Mat(m, n) => {
                if let DType::U32 | DType::I32 | DType::Bool = dtype {
                    panic!("trying to generate glsl representation of an unsupported matrix type: {:?}", ty)
                }
                match m == n {
                    true => format!("{}mat{}"   , dtype_to_glsl_prefix(*dtype), m),
                    false => format!("{}mat{}x{}", dtype_to_glsl_prefix(*dtype), m, n),
                }
            }
        },
        TyKind::Struct(Struct(Named(_, ident))) => {
            ex.valid_ident(**ident).to_string()
        },
        TyKind::Callable(x) => panic!("trying to generate glsl representation of a callable type: {:?}", x),
        TyKind::Array(Array(ty, size)) => match size {
            Some(n) => format!("{}[{}]", ty_to_glsl(ex, ty), n),
            None    => format!("{}[]"  , ty_to_glsl(ex, ty)),
        }
        TyKind::Opaque(x) => match x {
            OpaqueTy::TextureCombinedSampler(TexDtypeDimensionality(dtype, kind)) =>
                format!("{}sampler{}", dtype_to_glsl_prefix(*dtype), tex_dimensionality_to_glsl_suffix(*kind)),
            OpaqueTy::ShadowSampler(kind) =>
                format!("sampler{}Shadow", shadow_sampler_kind_to_glsl_suffix(*kind)),
            OpaqueTy::Sampler => "sampler".to_string(),
            OpaqueTy::Texture(TexDtypeDimensionality(dtype, kind)) =>
                format!("{}texture{}", dtype_to_glsl_prefix(*dtype), tex_dimensionality_to_glsl_suffix(*kind)),
            OpaqueTy::Image(TexDtypeDimensionality(dtype, kind)) =>
                format!("{}image{}", dtype_to_glsl_prefix(*dtype), tex_dimensionality_to_glsl_suffix(*kind)),
            OpaqueTy::AtomicCounter(_) => todo!(),
        },
        TyKind::ArrayOfOpaque(_) => unimplemented!(),
        TyKind::InterfaceBlock(_) => unimplemented!(),
    }
}

fn expr_to_glsl(ex: &State, expr: &Expr, ignore_ident: bool) -> String {
    match (expr.ident.as_ref(), ignore_ident) {
        (Some(ident), false) => ex.valid_ident(ident.0).to_string(),
        _ => expr_kind_to_glsl(ex, expr)
    }
}

fn expr_key_to_glsl(ex: &State, expr_key: Key<Expr>, ignore_ident: bool) -> String {
    let result = expr_to_glsl(ex, &ex.ctx.exprs()[expr_key], ignore_ident);
    let index = expr_key.index();
    match ex.verbose {
        true => format!(" /*ᵉ{index}*/{result}"),
        false => result,
    }
}

fn stmt_to_glsl(ex: &State, stmt: &Stmt) -> String {
    match &stmt.kind {
        StmtKind::VariableDecl(Named(ty, ident)) => {
            format!("{} {};",
                ty_to_glsl(ex, ty),
                ex.valid_ident(ident.0)
            )
        },
        StmtKind::VariableDef(Named(expr_key, ident)) => {
            format!("{} {} = {};",
                ty_to_glsl(ex, &ex.ctx.exprs()[*expr_key].ty),
                ex.valid_ident(ident.0),
                expr_key_to_glsl(ex, *expr_key, true),
            )
        }
        StmtKind::Expr(x) => format!("{};", expr_key_to_glsl(ex, *x, false)),
        StmtKind::Flow(x) => flow_to_glsl(ex, x),
        StmtKind::Return(x) => match x {
            Some(x) => format!("return {};", expr_key_to_glsl(ex, *x, false)),
            None => "return;".to_string(),
        },
        StmtKind::Discard => "discard;".to_string(),
        StmtKind::Continue => "continue;".to_string(),
        StmtKind::Break => "break;".to_string(),
    }
}

fn block_to_glsl(ex: &State, block: &Block) -> String {
    let mut s = String::with_capacity(64);
    use StmtKind::*;
    use BlockKind::*;
    match block.kind {
        Body | LoopBody => {
            s += "{\n";
            ex.with_deeper_indent(|| {
                for stmt in block.stmts.iter() {
                    let stmt_string = stmt_to_glsl(ex, stmt);
                    s += &format!("{}{}\n", ex.indent_string(), stmt_string);
                }
            });
            s += &(ex.indent_string() + "}");
        },
        LoopCondition(loop_condition_expr) => {
            let len = block.stmts.len();
            for (i, stmt) in block.stmts.iter().enumerate() {
                let is_last = i == len-1;
                let stmt_string = match &stmt.kind {
                    Expr(key) => {
                        if let Some(loop_condition_expr) = loop_condition_expr {
                            // if the stmt is the loop condition, it must be last
                            // if the stmt is not loop condition, it can't be last
                            assert!((loop_condition_expr == *key) == is_last); //TODO: make result
                        }
                        expr_key_to_glsl(ex, *key, false)
                    }
                    kind => panic!("invalid statement of kind {kind} in loop condition") //TODO: make error
                };
                let maybe_comma = match i {0 => "", _ => ", "};
                s += &format!("{}{}", maybe_comma, stmt_string);
            }
            if loop_condition_expr.is_none() {
                // no condition implies endless loop
                // this means we need to add a true at the end
                s += match len {
                    0 => "true",
                    _ => ", true",
                }
            }
        },
        LoopIncrement => {
            for (i, stmt) in block.stmts.iter().enumerate() {
                let stmt_string = match &stmt.kind {
                    Expr(key) => expr_key_to_glsl(ex, *key, false),
                    kind => panic!("invalid statement of kind {kind} in loop increment") //TODO: make error
                };
                let maybe_comma = match i {0 => "", _ => ", "};
                s += &format!("{}{}", maybe_comma, stmt_string);
            }
        },
        LoopInit => {

            let iter = block.stmts.iter();
            let first = iter.clone().next();

            if let Some(first) = first {
                match &first.kind {
                    VariableDecl(_) | VariableDef(_) => {

                        let decl_ty = match &first.kind {
                            VariableDecl(Named(ty, _)) => ty.clone(),
                            VariableDef(Named(key, _)) => ex.ctx.exprs()[*key].ty.clone(),
                            _ => unreachable!()
                        };

                        s += &(ty_to_glsl(ex, &decl_ty) + " ");
                        let mut decls = vec![];

                        for stmt in iter {
                            let (ident, ty, expr_key) = match &stmt.kind {
                                VariableDecl(Named(ty, ident)) => (ident, ty.clone(), None),
                                VariableDef(Named(key, ident)) => (ident, ex.ctx.exprs()[*key].ty.clone(), Some(key)),
                                _ => unreachable!()
                            };

                            assert!(decl_ty.eq_ignore_access(&ty), "glsl only supports one declaration type per loop init statement. This loop init statement tries to declare at least a {decl_ty} and {ty} type");

                            let ident = ex.valid_ident(ident.0);
                            decls.push(match expr_key {
                                Some(expr_key) => format!("{} = {}", ident, expr_key_to_glsl(ex, *expr_key, true)),
                                None => ident.to_string(),
                            });
                        }
                        s += &decls.join(", ");
                    }
                    Expr(_) => {
                        s += &iter.map(|stmt| match &stmt.kind {
                            Expr(x) => expr_key_to_glsl(ex, *x, false),
                            _ => unreachable!()
                        }).collect::<Vec<_>>().join(", ");
                    }
                    kind => {
                        panic!("invalid statement of kind {kind} in loop increment") //TODO: make error
                    }
                }
            }

        },
    }
    s
}

fn block_key_to_glsl(ex: &State, block_key: Key<Block>) -> String {
    block_to_glsl(ex, &ex.ctx.blocks()[block_key])
}

fn item_to_glsl(ex: &State, item: &Item) -> String {
    match item {
        Item::FuncDef { .. } => todo!(),
        Item::MainFuncDef { body } => {
            let body = body.get().expect("main function block missing");
            format!("void main() {}", block_key_to_glsl(ex, body))
        },
        Item::StructDef(struct_) => {
            struct_def_to_glsl(ex, struct_)
        }
    }
}

fn struct_def_to_glsl(ex: &State, struct_: &Struct) -> String {
    let &Struct(Named(fields_, name_)) = &struct_;

    let fields_string = fields_.iter().map(|Named(ty, ident)| {
        let ty = ty_to_glsl(ex, ty);
        let ident_string = ex.valid_ident(**ident);
        let indent = ex.with_deeper_indent(|| ex.indent_string());
        format!("{indent}{ty} {ident_string};\n")
    })
    .collect::<String>();

    let indent = ex.indent_string();
    let ident_string = ex.valid_ident(**name_);
    format!("{indent}struct {ident_string} {{\n{fields_string}{indent}}};\n")
}

#[derive(Clone, Copy)]
pub enum Packing {
    Shared,
    Packed,
    Std140,
    Std430,
}

fn packing_to_glsl(packing: Packing) -> &'static str {match packing {
    Packing::Shared => "shared",
    Packing::Packed => "packed",
    Packing::Std140 => "std140", //TODO: implement warning from OpenGL wiki: Warning: Implementations sometimes get the std140 layout wrong for vec3 components. You are advised to manually pad your structures/arrays out and avoid using vec3 at all.
    Packing::Std430 => "std430",
}}

fn struct_layout_to_glsl(ex: &State, struct_layout: &InterfaceBlock) -> Result<String, Error> {
    let result = struct_layout.0.iter().map(|any| -> Result<String, Error> {

        let (ty, ident) = ty_ident_of_any(any, ex)
        .ok_or(Error::NA {reason: "interface block member not found"})?;

        let ty = ty_to_glsl(ex, &ty);
        let ident_string = ex.valid_ident(*ident);
        let indent = ex.with_deeper_indent(|| ex.indent_string());
        Ok(format!("{indent}{ty} {ident_string};\n"))
    })
    .collect::<Result<String, Error>>()?;

    Ok(format!("{{\n{}}}", result))
}

fn binding_to_glsl(ex: &State, set_binding_index: (u32, u32), binding: &Binding) -> Result<(String, Option<Packing>), Error> {
    use Binding::*;
    let block_ident = |(set_i, bind_i)| format!("S{}_B{}", set_i, bind_i);

    let result = match binding {
        Opaque(opaque_ty, any) => {
            assert!(!matches!(opaque_ty, OpaqueTy::Image(_)), "Image bindings are not supposed to be in Binding::Opaque, but in Binding::OpaqueImage because of their additional parameters");
            let opaque_ty = Ty::new(TyKind::Opaque(*opaque_ty));

            let (ty, ident) = ty_ident_of_any(any, ex)
            .ok_or(Error::NA {reason: "interface member not found"})?;

            assert!(opaque_ty.eq_ignore_access(&ty), "opaque-type binding has changed its type since the interface recording");

            let ty_string = ty_to_glsl(ex, &ty);
            let ident_string = ex.valid_ident(*ident);

            format!("uniform {ty_string} {ident_string};\n")
        },
        OpaqueImage { .. } => {
            unimplemented!()
        },
        UniformBlock(x) => format!("uniform {} {};\n", block_ident(set_binding_index),
            struct_layout_to_glsl(ex, x)?),
        StorageMut(x) => format!("buffer {} {};\n", block_ident(set_binding_index),
            struct_layout_to_glsl(ex, x)?),
        Storage(x) => format!("readonly buffer {} {};\n", block_ident(set_binding_index),
            struct_layout_to_glsl(ex, x)?),
    };

    let packing = match binding {
        Opaque(_, _) | OpaqueImage{..} => None,
        UniformBlock(_) | StorageMut(_) | Storage(_) => Some(Packing::Std140),
    };

    Ok((result, packing))
}

fn push_constant_str(ex: &State, sfx: &SideEffects) -> Result<String, Error> {

    match sfx.push_constant {
        Some(any) => {
            let (ty, ident) = ty_ident_of_any(&any, ex)
            .ok_or(Error::NA{reason: "unable to obtain push constant's identifier and type"})?;
            let ident_string = ex.valid_ident(*ident);
            let ty = ty_to_glsl(ex, &ty);
            let indent = ex.with_deeper_indent(|| ex.indent_string());
            let decl = format!("{}{} {};\n", indent, ty, ident_string);
            Ok(format!("layout(push_constant) uniform PushConstantBlock {{\n{}}};\n", decl))
        },
        None => Ok("".to_string()),
    }
}

fn side_effects_to_glsl(ex: &State, sfx: &SideEffects) -> Result<String, Error> {

    let groups_str = sfx.bind_groups.iter().map(|(set_i, bind_group)| {
        bind_group.0.iter().map(move |(bind_i, binding)| -> Result<String, Error> {
            let (binding_string, packing) = binding_to_glsl(ex, (*set_i, *bind_i), binding)?;
            let packing_string = packing.map(|p| format!("{}, ", packing_to_glsl(p))).unwrap_or_else(|| "".to_string());
            Ok(format!("layout({}set={}, binding={}) {}\n", packing_string, set_i, bind_i, binding_string))
        })
    })
    .flatten()
    .collect::<Result<String, Error>>()?;

    let push_constant_str = push_constant_str(ex, sfx)?;

    Ok(groups_str + &push_constant_str)
}

fn ty_ident_of_any(any: &Any, ex: &State) -> Option<(Ty, IdentSlot)> {
    let expr_key = any.expr_key?;
    let expr = &ex.ctx.exprs()[expr_key];
    let ident = expr.ident.as_ref()?;
    Some((expr.ty.clone(), *ident))
}


fn varyings_to_glsl(ex: &State, varyings: &Varyings) -> Result<String, Error> {
    let qualifier = match varyings.0 {
        InOut::In => "in",
        InOut::Out => "out",
    };

    varyings.1.iter().enumerate().map(|(index, (interp, any))| -> Result<String, Error> {

        let interp = match interp {
            Interpolation::Flat => "flat ",
            Interpolation::Linear => "noperspective ",
            Interpolation::PerspectiveLinear => "", //perspective linear corresponds to "smooth" in glsl, which is the default so it can be left out
        };

        let (ty, ident) = ty_ident_of_any(any, ex)
        .ok_or(Error::NA {reason: "interpolated variable not found"})?;

        let ty = ty_to_glsl(ex, &ty);
        let ident_string = ex.valid_ident(*ident);
        Ok(format!("layout(location={}) {}{} {} {};\n", index, interp, qualifier, ty, ident_string))
    })
    .collect::<Result<String, Error>>()
}

fn stage_interface_to_glsl(ex: &State, stage: &StageInterface) -> Result<String, Error> {
    let result = match stage {
        StageInterface::Vertex { inputs, outputs } => {
            let in_string = inputs.vertex_attributes.iter().map(|(locs, any)| -> Result<String, Error> {
                let (ty, ident) = ty_ident_of_any(any, ex)
                .ok_or(Error::NAInShaderKind {expected: ShaderKind::Vertex, found: ex.ctx.shader_kind})?;
                let ty = ty_to_glsl(ex, &ty);
                let ident_string = ex.valid_ident(*ident);
                Ok(format!("layout(location={}) in {} {};\n", locs.start, ty, ident_string))
            })
            .collect::<Result<String, Error>>()?;

            let out_string = varyings_to_glsl(ex, outputs)?;
            format!("{}\n{}", in_string, out_string)
        },
        StageInterface::Fragment { inputs, outputs } => {
            let in_string = varyings_to_glsl(ex, inputs)?;

            let out_string = outputs.color_attachments.iter().map(|(loc, any)| {
                let (ty, ident) = ty_ident_of_any(any, ex)
                .ok_or(Error::NA {reason: "interface block member not found"})?;
                let ty = ty_to_glsl(ex, &ty);
                let ident_string = ex.valid_ident(*ident);
                Ok(format!("layout(location={}) out {} {};\n", loc, ty, ident_string))
            })
            .collect::<Result<String, Error>>()?;

            format!("{}\n{}", in_string, out_string)
        },
        StageInterface::Compute { workgroup_size } => {
            let [x, y, z] = workgroup_size.unwrap_or([1, 1, 1]);
            format!("layout(\n    local_size_x = {}, \n    local_size_y = {}, \n    local_size_z = {}\n) in;\n", x, y, z)
        },
    };
    Ok(result)
}

fn is_ident(s: &str) -> bool {
    !s.is_empty() && s.chars().enumerate().all(|(i, c)|
        match i {
            0 => c.is_ascii_alphabetic(),
            _ => c.is_ascii_alphanumeric(),
        } || c == '_'
    )
}

fn ident_satisfies_glsl_constraints(s: &str) -> bool {
    !(
        s.starts_with("gl_") ||
        s.starts_with("__") ||
        GLSL_WORDS_THAT_CANNOT_BE_IDENTIFIERS.contains(&s)
    )
}

const ANONYMOUS_IDENTIFIER_PREFIX: &str = "_";

fn single_valid_ident_for_slot(slot: &Option<String>) -> String {
    match slot {
        Some(string) if is_ident(string) => {
            match ident_satisfies_glsl_constraints(string) {
                true => string.clone(),
                false => match string.starts_with("__") {
                    true => format!("{ANONYMOUS_IDENTIFIER_PREFIX}0{string}"),
                    false => ANONYMOUS_IDENTIFIER_PREFIX.to_string() + string
                },
            }
        }
        _ => ANONYMOUS_IDENTIFIER_PREFIX.to_string()
    }
}

/// takes idents with the same name and puts numbers (1, 2...) behind the duplicates to
/// make them different form each other. this may result in new name collisions, therefore
/// the function returns the amount of edited idents. Call it repeatedly until it returns
/// 0 to make sure all the identifiers are unique.
/// TODO: add tests, verify that it works, or rewrite it with a more elegant algorithm.
#[allow(clippy::needless_range_loop)]
fn deduplicate_idents_pass(idents: &mut Vec<String>) -> usize {
    let mut order = (0..idents.len()).collect::<Vec<_>>();
    order.sort_by_key(|i| &idents[*i]);
    let mut changed_counter = 0;

    for i in 0..order.len() {
        let ai = order[i];

        for j in i+1..order.len() {
            let bi = order[j];

            let a = &idents[ai];
            let b = &idents[bi];

            if a == b {
                let delta = j-i;
                let b_mut = &mut idents[bi];
                *b_mut = format!("{}{}", b_mut, delta);
                if !ident_satisfies_glsl_constraints(b_mut) {
                    *b_mut = format!("{}{}", b_mut, 0); //append another '0' if the ident became invalid glsl, (e.g. if 'vec' became 'vec2', make it 'vec20' instead)
                }
                changed_counter += 1;
            }
            else {break}
        }
    }

    changed_counter
}

fn valid_idents_for_pool(idents: &Vec<Option<String>>) -> Vec<String> {
    let mut valids = idents.iter().map(single_valid_ident_for_slot).collect();
    loop { //iteratively make duplicate idents unique to prevent name collisions
        let num_edits = deduplicate_idents_pass(&mut valids);
        if num_edits == 0 {break}
    }
    debug_assert!(valids.iter().all_unique(), "identifiers contain duplicates after deduplication: {:?}", valids);
    debug_assert!(valids.iter().all(|s| is_ident(s) && ident_satisfies_glsl_constraints(s)), "valid identifiers list contains identifiers that aren't valid: {:?}", valids);
    valids
}

struct GlslExportState<'a> {
    ctx: &'a Context,
    indent: Cell<i32>,
    valid_idents: Vec<String>,
    verbose: bool,
}

impl GlslExportState<'_> {

    fn new(ctx: &Context, verbose: bool) -> GlslExportState {
        GlslExportState {
            ctx,
            indent: Cell::new(0),
            valid_idents: valid_idents_for_pool(&*ctx.idents()),
            verbose,
        }
    }

    fn valid_ident(&self, key: Key<Option<String>>) -> &str {
        &self.valid_idents[key.index]
    }

    fn with_deeper_indent<R>(&self, f: impl FnOnce() -> R) -> R {
        let level = self.indent.get();
        self.indent.set(level + 1);
        let r = f();
        self.indent.set(level);
        r
    }

    fn indent_string(&self) -> String {
        match self.indent.get() {
            indent @ 0.. => "    ".repeat(indent as usize),
            _ => "/*negative indentation*/".to_string(),
        }
    }
}

type State<'a> = GlslExportState<'a>;

impl Context {
    pub fn generate_glsl(&self) -> Result<String, Error> {
        let ex = GlslExportState::new(self, false);
        let version = "#version 450\n".to_string();

        let struct_defs = self.items().iter()
        .filter(|item| matches!(item, Item::StructDef(_))) //pick only the struct defs
        .map(|item| item_to_glsl(&ex, item))
        .collect::<Vec<_>>()
        .join("\n");

        let side_effects =    side_effects_to_glsl(&ex, &self.shader().side_effects)?;
        let stage        = stage_interface_to_glsl(&ex, &self.shader().stage_interface)?;

        let functions = self.items().iter()
        .filter(|item| !matches!(item, Item::StructDef(_))) //filter everything except struct defs
        .map(|item| item_to_glsl(&ex, item))
        .collect::<Vec<_>>()
        .join("\n\n");

        let mut elements = vec![version];
        elements.push(struct_defs);
        if !side_effects.trim().is_empty() {elements.push(side_effects)}
        if !stage.trim().is_empty() {elements.push(stage)}
        elements.push(functions);
        let result = elements.join("\n");

        Ok(result)
    }
}