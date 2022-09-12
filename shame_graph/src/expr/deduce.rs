use smallvec::SmallVec;
use crate::{common::IteratorExt, context::Context, error::Error, expr::{DType, Shape}, pool::*, IdentSlot, Struct, Array, find_closest_ancestor_non_lvalue, OpaqueTy};

use super::{Access, Expr, ExprKind, Named, Tensor, Ty, try_deduce_builtin_fn, try_deduce_builtin_var};
use crate::expr::TyKind;

///estimated upper bound to argument count, which will be used in SmallVecs to avoid allocations
pub const EST_ARGS: usize = 8;

pub fn args_as_types(args: &[Key<Expr>]) -> SmallVec<[Ty; EST_ARGS]> {
    Context::with(|ctx| {
        args.iter().map(|&arg| ctx.exprs()[arg].ty.clone()).collect()
    })
}

pub fn types_as_tensors(types: &[Ty]) -> Option<SmallVec<[Tensor; EST_ARGS]>> {
    let result = types.iter().map(|ty| -> Option<Tensor> {
        if let TyKind::Tensor(ten) = ty.kind {
            Some(ten)
        } else {None}
    }).collect();
    result
}

pub fn arg_count_error(kind: impl std::fmt::Debug, args: &[Ty]) -> Error {
    let msg = format!("{:?} got wrong amount of arguments ({}).", kind, args.len());
    Error::ArgumentError(msg)
}

pub fn invalid_arguments(kind: impl std::fmt::Display, args: &[Ty]) -> Error {
    let arg_list = args.iter().map(|arg| arg.to_string()).collect::<Vec<String>>().join(", \n");
    let msg = match arg_list.is_empty() {
        true => format!("invalid arguments - no arguments provided for {kind}"),
        false => format!("invalid arguments for {kind}: \n{arg_list}."),
    };
    Error::ArgumentError(msg)
}

pub fn validate_access(kind: &ExprKind, arg_types: &[Ty], args: &[Key<Expr>]) -> Result<(), Error> {
    for (i, (arg_ty, arg)) in arg_types.iter().zip(args).enumerate() {

        let writing = kind.is_mutating_arg_with_index(i);
        let reading = !writing;

        let row = || format!("{:?} trying to write to argument {}, which is const: {}", kind, i, arg_types[i]);
        let wor = || format!("{:?} trying to read from argument {}, which is write-only: {}", kind, i, arg_types[i]);

        //if the argument is an lvalue, the access of the non-lvalue it refers to is relevant here
        let non_lvalue_access = match arg_ty.access {
            Access::Const | Access::WriteOnly | Access::CopyOnWrite => arg_ty.access,
            Access::LValue => Context::with(|ctx| {
                let exprs = ctx.exprs();
                let anc = find_closest_ancestor_non_lvalue(&exprs, *arg);
                exprs[anc].ty.access
            }),
        };

        match non_lvalue_access {
            Access::Const     if writing => {
                Err(Error::ArgumentError(row()))? //read-only write
            }
            Access::WriteOnly if reading => {
                Err(Error::ArgumentError(wor()))? //write-only read
            }
            //LValue, CopyOnWrite
            _ => (),
        }
    }
    Ok(())
}

pub fn try_deduce_type(kind: &ExprKind, args: &[Ty]) -> Result<Ty, Error> {
    match kind {
        ExprKind::GlobalInterface(ty) => Ok(ty.clone()),
        ExprKind::Copy{..} => try_deduce_copy(kind, args),
        ExprKind::Literal    (x) => Ok(Ty::tensor(Shape::Scalar, x.dtype)),
        ExprKind::Constructor(x) => try_deduce_constructor (x, args),
        ExprKind::Operator   (x) => try_deduce_operator    (x, args),
        ExprKind::Swizzle    (x) => try_deduce_swizzle     (x, args),
        ExprKind::FieldSelect(x) => try_deduce_field_select(x, args),
        ExprKind::BuiltinFn  (x) => try_deduce_builtin_fn  (x, args),
        ExprKind::BuiltinVar (x) => try_deduce_builtin_var (x, args),
    }
}

pub fn try_deduce_field_select(field: &IdentSlot, args: &[Ty]) -> Result<Ty, Error> {
    let result = match args {
        [ty] => {
            match &ty.kind {
                TyKind::Struct(Struct(Named(fields, _))) => {
                    // verify that our struct actually has a field with that ident
                    // and read its type
                    let result = fields.iter().find_map(|Named(ty, ident)| {
                        (ident == field).then(|| ty.clone().into_access(Access::LValue))
                    });

                    result.ok_or_else(|| {
                        Context::with(|ctx| {
                            let field = ctx.idents()[**field].clone().unwrap_or("unnamed-field".to_string());
                            Error::FieldSelectError(format!("no field {field} in struct {ty}"))
                        })
                    })
                },
                _ => Err(invalid_arguments("struct-field-select", args))
            }
        }
        _ => Err(arg_count_error("struct-field-select", args))
    };
    assert!(result.as_ref().map_or(true, |ty| {
        ty.access == Access::LValue
    }), "struct field select expressions must always return lvalues");
    result
}

pub fn try_deduce_copy(kind: &ExprKind, args: &[Ty]) -> Result<Ty, Error> {
    //copy converts lvalues and const values to CopyOnWrite, otherwise it behaves like an identity function
    //in glsl, a copy expr becomes a variable definition:
    //```type ident = arg_expr;```
    //which is enforced in `Expr::needs_variable_def_stmt`
    use TyKind::*;
    match args {
        [arg] => {
            //copy can only be called on types that can be declared as a variable
            let ty = match arg.kind {
                Tensor(_) | Struct(_) | Array(_) => {
                    arg
                }
                Void | Callable(_) | Opaque(_) | ArrayOfOpaque(_) | InterfaceBlock(_) => {
                    Err(invalid_arguments(kind, args))?
                }
            };
            Ok(ty.clone().into_access(Access::CopyOnWrite))
        }
        _ => Err(arg_count_error(kind, args))?
    }
}

pub fn try_deduce_constructor(kind: &super::Constructor, args: &[Ty]) -> Result<Ty, Error> {
    use super::Constructor;

    match kind {
        Constructor::Tensor(result_ten) => {
            use DType::*;

            let is_supported_result = match (result_ten.dtype, result_ten.shape) {
                (I32 | U32 | Bool, Shape::Mat(_, _)) => false, //glsl does not support imat*, umat*, bmat* types
                (F32 | F64       , Shape::Mat(_, _)) => true,
                (_, Shape::Vec(_) | Shape::Scalar)    => true,
            };

            match (is_supported_result, types_as_tensors(args)) {
                (true, Some(arg_tens)) => {
                    let result_type = Ty::new(TyKind::Tensor(*result_ten));

                    match arg_tens.as_slice() {
                        [_arg] => Ok(result_type), //scalar, expanding or shortening ctor
                        arg_tens => { //concat components
                            let no_mats        = arg_tens.iter().all(|t| !t.shape.is_mat());
                            let all_same_dtype = arg_tens.iter().all(|t| t.dtype == result_ten.dtype);

                            let comp_sum = arg_tens.iter().map(|t| t.shape.comps_total()).sum::<usize>();
                            let result_comps = result_ten.shape.comps_total();

                            match comp_sum {
                                1 => panic!("component sum of multiple-arg match should never be 1"), //caught in outter match
                                _ if all_same_dtype && no_mats && comp_sum == result_comps => Ok(result_type),
                                _ => Err(invalid_arguments(kind, args))
                            }
                        }
                    }
                },
                _ => Err(invalid_arguments(kind, args)),
            }
        },
        Constructor::Struct(struct_) => {
            let Struct(Named(fields, _)) = struct_;

            let all_arg_types_match = fields.iter().zip(args).all(|(Named(field_ty, _), arg_ty)| {
                field_ty.eq_ignore_access(arg_ty)
            });

            all_arg_types_match
            .then(|| Ty::new(TyKind::Struct(struct_.clone())))
            .ok_or_else(|| {
                invalid_arguments(kind, args)
            })
        },
        Constructor::Array(array) => {
            let Array(elem_ty, maybe_len) = array;
            match maybe_len {
                Some(len) if len != &args.len() => Err(invalid_arguments(kind, args)),
                Some(_) | None => Ok(
                    Ty::new(TyKind::Array(Array::new_sized((**elem_ty).clone(), args.len())))
                ),
            }
        },
        Constructor::TextureCombinedSampler(ctor_dtype_dims) => {
            // XsamplerYD(t, s)
            match args {
                [texture, sampler] => {
                    match (&texture.kind, &sampler.kind) {
                        (
                            TyKind::Opaque(OpaqueTy::Texture(arg_dtype_dims)),
                            TyKind::Opaque(OpaqueTy::Sampler),
                        ) =>
                        match ctor_dtype_dims == arg_dtype_dims {
                            true => Ok(Ty::texture_combined_sampler(ctor_dtype_dims.0, ctor_dtype_dims.1)),
                            false => Err(invalid_arguments(kind, args)),
                        }
                        _ => Err(invalid_arguments(kind, args)),
                    }
                }
                _ => Err(invalid_arguments(kind, args))
            }
        },
    }

}

pub fn try_deduce_operator(kind: &super::Operator, args: &[Ty]) -> Result<Ty, Error> {
    use super::Operator;
    use crate::Shape::*;
    use crate::DType::*;

    match args.first() {
        Some(first) => match &first.kind {
            TyKind::Tensor(_) => {
                return try_deduce_tensor_operator(kind, args)
            },
            TyKind::Struct(_) => match (kind, &args) {
                (Operator::Assign, &[lhs, rhs]) if lhs.eq_ignore_access(rhs) => {
                    return Ok(Ty::void());
                },
                _ => (),
            },
            TyKind::Array(Array(elem_ty, _len)) => match (kind, &args) {
                (Operator::Subscript, &[_, index]) => match index.kind {
                    //see https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.html#structure-and-array-operations
                    TyKind::Tensor(Tensor { dtype: U32, shape: Scalar }) |
                    TyKind::Tensor(Tensor { dtype: I32, shape: Scalar }) => {
                        return Ok((**elem_ty).clone().into_access(Access::LValue));
                    },
                    _ => ()
                }
                (Operator::Assign, &[lhs, rhs]) if lhs.eq_ignore_access(rhs) => {
                    return Ok(Ty::void());
                },
                _ => ()
            },
            _ => ()
        },
        None => () // no operator has zero args
    };
    return Err(invalid_arguments(kind, args));
}

pub fn try_deduce_swizzle(kind: &super::Swizzle, args: &[Ty]) -> Result<Ty, Error> {
    use Access::*;
    let (dtype, arg_access, num_comps) = match args {
        [ty] => match ty.kind {
            TyKind::Tensor(ten) => match ten.shape {
                Shape::Vec(num_comps) => (ten.dtype, ty.access, num_comps),
                _ => Err(invalid_arguments(kind, args))?,
            },
            _ => Err(invalid_arguments(kind, args))?,
        }
        _ => Err(arg_count_error(kind, args))?,
    };

    let swizzle = kind.inner_slice();
    let amount_ok = swizzle.len() <= num_comps as usize;
    let indices_ok = swizzle.iter().all(|i| (0..4).contains(i));
    if !(amount_ok && indices_ok) {
        Err(invalid_arguments(kind, args))?;
    }

    let shape = Shape::from_vec_len(swizzle.len())
    .expect("invalid swizzle length occured in type deduction");

    // GLSL 4.60 spec: chapter 5.8. Assignments
    // "swizzles with repeated fields [...] cannot be l-values"
    let access = match (swizzle.iter().all_unique(), arg_access) {
        (true, _) => LValue,
        (false, LValue) => CopyOnWrite,
        (false, Const | CopyOnWrite) => CopyOnWrite,
        (false, WriteOnly) => WriteOnly,
    };

    Ok(Ty::tensor(shape, dtype).into_access(access))
}

pub fn try_deduce_tensor_operator(kind: &super::Operator, args: &[Ty]) -> Result<Ty, Error> {
    use super::Operator::*;
    let tens = types_as_tensors(args).ok_or_else(|| invalid_arguments(kind, args))?;

    let result_tensor = match (tens.as_slice(), kind) {
        (&[l, r], Add | Sub | Mul | Div) => {

            if l.dtype == r.dtype {
                //https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.pdf chapter "5.9 Expressions"
                use Shape::*;
                match (kind, l.shape, r.shape) {
                    (_, Scalar, Scalar) => Ok(l),
                    (_, Vec(a), Vec(b)) if a == b => Ok(l),
                    (_, Scalar, Vec(_) | Mat(_, _)) => Ok(r),
                    (_, Vec(_) | Mat(_, _), Scalar) => Ok(l),
                    (Add | Sub | Div, Mat(a, b), Mat(x, y)) if a == x && b == y => Ok(l),

                    (Mul, _, _) => match (l.shape, r.shape, 1) { //the 1 is the missing dimension assigned to the row/col vecs
                        (Mat(l_col, l_row), Mat(r_col, r_row), _) |
                        (Mat(l_col, l_row), Vec(r_row)       , r_col) | //matrix * column-vector
                        (Vec(l_col)       , Mat(r_col, r_row), l_row)   //row-vector * matrix
                        if l_col == r_row => {
                            Ok(Tensor::new(Shape::from_dims_u8((r_col , l_row)), l.dtype))
                        }
                        _ => Err(invalid_arguments(kind, args))
                    }
                    _ => Err(invalid_arguments(kind, args))
                }
            } else {
                Err(invalid_arguments(kind, args))
            }
        },
        (&[arg], Subscript) => {
            // this is the subscript operator for tensors only
            // matrix[i] -> column vector
            // vector[i] -> scalar
            use Shape::*;
            match arg.shape {
                Scalar => Err(invalid_arguments(kind, args)),
                Vec(_v)    => Ok(Tensor::new(Scalar, arg.dtype)),
                Mat(_c, r) => Ok(Tensor::new(Vec(r), arg.dtype)),
            }
        },
        (&[arg], PostfixInc | PostfixDec | PrefixInc | PrefixDec) => Ok(arg),
        (&[arg], Positive | Negative) => Ok(arg),
        (&[arg], BitNot) if [DType::U32, DType::I32].contains(&arg.dtype) && !arg.shape.is_mat() => Ok(arg),
        (&[arg], Not) if arg == Tensor::bool() => Ok(Tensor::bool()),
        (&[l, r], ShiftL | ShiftR) => {
            use Shape::*;
            let dtype_ok = |ten: Tensor| [DType::U32, DType::I32].contains(&ten.dtype) && !ten.shape.is_mat();
            let shapes_ok = match (l.shape, r.shape) {
                (Scalar, Scalar) => true,
                (Vec(_), Scalar) => true,
                (Vec(x), Vec(y)) if x == y => true,
                _ => false,
            };

            match dtype_ok(l) && dtype_ok(r) && shapes_ok {
                true => Ok(l),
                false => Err(invalid_arguments(kind, args)),
            }
        },
        (&[l, r], Rem) => {
            use Shape::*;
            let both_int_uint = l == r && [DType::I32, DType::U32].contains(&l.dtype);
            let no_mats   = [l, r].iter().all(|t| !t.shape.is_mat());
            match both_int_uint && no_mats {
                true => match (l.shape, r.shape) {
                    (Scalar, Scalar) |
                    (Scalar, Vec(_)) => Ok(r),
                    (Vec(_), Scalar) => Ok(l),
                    (Vec(x), Vec(y)) if x == y => Ok(l),
                    _ => Err(invalid_arguments(kind, args))
                }
                false => Err(invalid_arguments(kind, args)),
            }
        },

        (&[l, r], Less | Greater | LessEqual | GreaterEqual) => {
            let both_same_scalars = l == r && l.shape == Shape::Scalar;
            let dtype_ok = [DType::U32, DType::I32, DType::F32, DType::F64].contains(&l.dtype);

            match both_same_scalars && dtype_ok {
                true => Ok(Tensor::bool()),
                false => Err(invalid_arguments(kind, args)),
            }
        },

        (&[l, r], Equal | NotEqual) if l == r => Ok(Tensor::bool()),

        (&[l, r], BitAnd | BitOr | BitXor) => { //careful if you ever implement implicit conversions! the result type depends on them
            use Shape::*;
            let dtype_ok = |ten: Tensor| [DType::U32, DType::I32].contains(&ten.dtype) && !ten.shape.is_mat();
            let dtypes_ok = dtype_ok(l) && l.dtype == r.dtype;

            match dtypes_ok {
                true => match (l.shape, r.shape) {
                    (Scalar, Scalar) => Ok(l),
                    (Vec(_), Scalar) => Ok(l),
                    (Scalar, Vec(_)) => Ok(r),
                    (Vec(x), Vec(y)) if x == y => Ok(l),
                    _ => Err(invalid_arguments(kind, args)),
                },
                false => Err(invalid_arguments(kind, args)),
            }
        },
        (&[l, r], And | Or | LogicalXor) if l == r && l == Tensor::bool() => Ok(Tensor::bool()),
        (&[exp1, exp2, exp3], TernaryIf) if exp1 == Tensor::bool() && exp2 == exp3 => Ok(exp2),
        (&[l, r], Assign) if l == r => Ok(l),
        (&[_l, _r], AddAssign | SubAssign | MulAssign | DivAssign | RemAssign | ShiftLAssign | ShiftRAssign | AndAssign | XorAssign | OrAssign) => {
            let non_assign_version = match kind {
                   AddAssign => Add,
                   SubAssign => Sub,
                   MulAssign => Mul,
                   DivAssign => Div,
                   RemAssign => Rem,
                ShiftLAssign => ShiftL,
                ShiftRAssign => ShiftR,
                   AndAssign => BitAnd,
                   XorAssign => BitXor,
                    OrAssign => BitOr,
                _ => panic!("assign-op has no corresponding non-assign-op")
            };
            try_deduce_tensor_operator(&non_assign_version, args).map(|ty| {
                ty.try_as_tensor().expect("expected tensor result from tensor operator type deduction")
            })
        }
        _ => Err(invalid_arguments(kind, args))
    };

    let access = match kind {
        Subscript => Access::LValue,
        _ => Access::CopyOnWrite,
    };

    result_tensor.map(|ten| Ty::new_access(TyKind::Tensor(ten), access))
}