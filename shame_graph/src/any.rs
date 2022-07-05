
use crate::BranchState;
use crate::common::IteratorExt;
use crate::common::new_array_enumerate;
use crate::Context;
use crate::context::ShaderKind;
use crate::error::Error;
use super::expr::*;
use super::pool::*;

///TODO: document that this has reference semantics, e.g. = operator behavior, which may be unintuitive
#[derive(Debug, Clone, Copy)]
pub struct Any {
    pub(crate) expr_key: Option<Key<Expr>>,
}

impl Any {
    
    fn by_recording_expr(kind: ExprKind, args: &[Any]) -> Self {

        let all_is_some = args.iter().all(|any| any.expr_key.is_some());

        let expr_key = all_is_some.then(|| {

            let unwrapped_args = args.iter().map(|arg| arg.expr_key.unwrap()).collect(); //allocates a Vec

            let current_block = Context::with(|ctx| ctx.current_block_key_unwrap());

            let maybe_expr = Expr::new(
                None, 
                kind, 
                unwrapped_args,
                current_block
            );

            let expr = match maybe_expr {
                Ok(expr) => expr,
                Err(e) => {Context::with(|ctx| ctx.push_error(e)); return None},
            };
            let expr_key = Context::with(|ctx| ctx.exprs_mut().push(expr));
            Some(expr_key)

        }).flatten();

        // // this commented out block below behaves wrong when recording an if(uniformval)
        // if !all_is_some { //attempt to record an expression with some expr_keys unavailable in this shader recording (stage)
        //     if let Some(BranchState::Branch) = Context::with(|ctx| ctx.inside_branch()) {
        //         //if we are in a true branch, (not a branch on a Any::not_available() condition) attempting to record anything with Any::not_available() arguments is regarded as an error,
        //         //because those expressions just won't not show up in the shader even though intentionally put there. Calling into a function that requires e.g. both vertex and fragment variables could be manipulated by calling from a `vertex_bool.then(|| {...})` branch to only record the vertex portion of that function and disable the fragment portion, which messes with the function implementors expectation.
        //         Context::with(|ctx| {
        //             let kind = ctx.shader_kind();
        //             ctx.push_error(Error::NADependent(
        //                 Rc::new(format!("attempting to record a non-{kind}-stage expression inside a conditional block with a {kind}-stage condition. (conditional blocks are if-then/if-then-else/for/while... recordings)"))
        //             ));
        //         });
        //     }
        // }

        Self {expr_key}
    }


    pub(crate) fn ty(&self, pool: &PoolRef<Expr>) -> Option<Ty> {
        self.expr_key.map(|key| pool[key].ty.clone())
    }

    pub fn ty_via_ctx(&self, ctx: &Context) -> Option<Ty> {
        self.expr_key.map(|key| ctx.exprs()[key].ty.clone())
    }

    pub fn ty_via_thread_ctx(&self) -> Option<Ty> {
        Context::with(|ctx| self.ty_via_ctx(ctx))
    }
    
    /// assign a name that will be used when the recorded expression is converted into a variable in the resulting shader code.
    /// the provided name may get changed slightly in order to not collide with keywords/other variables in the target language.
    pub fn aka(self, name: &str) -> Self {
        //will silently fail if self has no expr key
        self.aka_maybe(Some(name.to_string()));
        self
    }

    pub fn aka_maybe(self, maybe_name: Option<String>) -> Self {
        //will silently fail if self has no expr key
        Context::with(|ctx| {
            let mut exprs = ctx.exprs_mut();
            self.expr_key.map(|key| exprs[key].force_ident(maybe_name))
        });
        self
    }
    
    //
    // ctors
    //

    pub fn not_available() -> Self {
        Any {
            expr_key: None,
        }
    }

    pub fn is_available(&self) -> bool {
        self.expr_key.is_some()
    }

    /// converts lvalues to non-lvalues and removes constness. Allows for manipulation of the resulting value without influencing any other value.
    /// cannot be used on write-only values.
    /// ```ignore
    /// let a = Any::vec3(Any::f32(1.0));
    /// let xy = a.xy(); //.xy() returns an lvalue
    /// xy.set(Any::f32(2.0)); //writes into `a`'s `x` and `y` components
    /// let xy2 = a.xy().copy(); //.copy() makes xy2 a non-lvalue
    /// xy2.set(Any::)
    /// ```
    ///
    pub fn copy(&self) -> Self {
        let copied = Any::by_recording_expr(ExprKind::Copy, &[*self]);

        //if self already has an identifier, create a new slot based on that identifier
        //that will result in a `foo` getting copied into `foo_copy` or similar
        self.expr_key.map(|key| {
            Context::with(|ctx| {
                let maybe_ident = {
                    let exprs = ctx.exprs_mut();
                    let idents = ctx.idents_mut();
                    exprs[key].ident
                        .and_then(|slot| idents[*slot].as_ref())
                        .map(|name| format!("{name}_copy"))
                };

                maybe_ident.map(|ident| {
                    copied.aka(&*ident)
                })
            })
        });

        copied
    }

    /// used e.g. for uniform blocks
    pub fn global_interface(ty: Ty, ident: Option<String>) -> Self {
        Any::by_recording_expr(ExprKind::GlobalInterface(ty), &[]).aka_maybe(ident)
    }
    
    pub fn texture_combined_sampler(kind: TexDtypeDimensionality, texture: Any, sampler: Any) -> Any {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::TextureCombinedSampler(kind)), &[texture, sampler])
    }

    pub fn bool(value: bool) -> Self {
        Any::by_recording_expr(ExprKind::Literal(Literal::Bool(value)), &[])
    }

    pub fn float(value: f32) -> Self {
        use std::num::FpCategory::*;
        match value.classify() {
            Normal | Zero => 
                Any::by_recording_expr(ExprKind::Literal(Literal::F32(value)), &[]),
            cat => Context::with(|ctx| {
                ctx.push_error(Error::UnsupportedFloadingPointCategory(cat));
                Any::not_available()
            })
        }
    }

    pub fn double(value: f64) -> Self {
        use std::num::FpCategory::*;
        match value.classify() {
            Normal | Zero => {},
            cat => panic!("cannot convert f64 point number of category '{:?}' to glsl", cat)
        };
        Any::by_recording_expr(ExprKind::Literal(Literal::F64(value)), &[])
    }

    pub fn int(value: i32) -> Self {
        Any::by_recording_expr(ExprKind::Literal(Literal::I32(value)), &[])
    }

    pub fn uint(value: u32) -> Self {
        Any::by_recording_expr(ExprKind::Literal(Literal::U32(value)), &[])
    }

    pub fn cast_bool(arg: Any) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::bool())), &[arg])
    }

    pub fn cast_float(arg: Any) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::float())), &[arg])
    }

    pub fn cast_double(arg: Any) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::double())), &[arg])
    }

    pub fn cast_int(arg: Any) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::int())), &[arg])
    }

    pub fn cast_uint(arg: Any) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::uint())), &[arg])
    }

    pub fn vec2(args: &[Any]) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::vec2())), args)
    }

    pub fn vec3(args: &[Any]) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::vec3())), args)
    }

    pub fn vec4(args: &[Any]) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::vec4())), args)
    }

    pub fn ivec2(args: &[Any]) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::ivec2())), args)
    }

    pub fn ivec3(args: &[Any]) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::ivec3())), args)
    }

    pub fn ivec4(args: &[Any]) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(Tensor::ivec4())), args)
    }

    pub fn new_tensor(tensor: Tensor, args: &[Any]) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Tensor(tensor)), args)
    }

    pub fn new_matrix_from_cols(tensor: Tensor, args: &[Any]) -> Self {
        Self::new_tensor(tensor, args)
    }

    pub fn new_matrix_from_rows(tensor: Tensor, args: &[Any]) -> Self {
        let (c, r) = tensor.shape.dims_u8();
        
        match args.len() == tensor.shape.row_count() {
            false => {
                Context::with(|ctx| ctx.push_error(Error::ArgumentError(
                    format!("cannot create a matrix ({}) with {} rows", tensor.to_string(), args.len())
                )));
                Any::not_available()
            }
            true => {
                // this calls
                // matMxN(row0.x, row1.x..., row0.y, row1.y... ,...)
                let comps = 
                (0..c).map(|col_i| 
                    (0..r).map(move |row_i|
                        args[row_i as usize].swizzle(&[col_i])
                    )
                ).flatten().collect::<Vec<_>>();
                Self::new_tensor(tensor, &comps)
            }
        }
    }

    pub fn struct_initializer(struct_: Struct, args: &[Any]) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Struct(struct_)), args)
    }

    pub fn array_initializer(array: Array, args: &[Any]) -> Self {
        Any::by_recording_expr(ExprKind::Constructor(Constructor::Array(array)), args)
    }

    //
    // swizzle / component access / field select
    //

    /// this is the dot operator for struct field access e.g. `foo.member`
    /// which returns an lvalue
    pub fn field_select(&self, field: IdentSlot) -> Self {
        Any::by_recording_expr(ExprKind::FieldSelect(field), &[*self])
    }

    /// used to select all the fields in Struct related derive code
    pub fn select_first_n_fields<const N: usize>(&self) -> [Any; N] {

        Context::with(|ctx| {
            self.ty_via_ctx(ctx).map(|ty| {

                let not_available_error = || {
                    ctx.push_error(Error::FieldSelectError(format!("cannot select {N} fields from {ty}")));
                    [Any::not_available(); N]
                };

                let struct_ty = match &ty.kind {
                    TyKind::Struct(s) => s,
                    _ => return not_available_error(),
                };
                let Struct(Named(fields, _)) = struct_ty;
                if fields.len() != N {
                    return not_available_error();
                }

                new_array_enumerate(|i| {
                    let Named(_, ident) = fields[i];
                    Any::by_recording_expr(ExprKind::FieldSelect(ident), &[*self])
                })
            }).unwrap_or_else(|| {
                [Any::not_available(); N]
            })
        })
    }

    ///returns `None` if `self` is not available in the current shader stage
    pub fn subscript_len(&self) -> Option<SubscriptLen> {
        use SubscriptLen::*;
        self.ty_via_thread_ctx().map(|ty| {
            match &ty.kind {
                TyKind::Tensor(tensor) => match tensor.len_wrt_subscript_operator() {
                    Some(len) => Sized(len),
                    None => InvalidTy(ty)
                },
                TyKind::Array(Array(_, len)) => match *len {
                    Some(len) => Sized(len),
                    None => Unsized,
                },
                TyKind::ArrayOfOpaque(_) => todo!(),
                TyKind::Opaque(_) => todo!(),
                TyKind::Void |
                TyKind::Struct(_) |
                TyKind::Callable(_) |
                TyKind::InterfaceBlock(_) => InvalidTy(ty), 
            }
        })
    }

    pub fn subscript(&self, index: Any) -> Self {

        let index_literal = index.expr_key
        .and_then(|key| {
            Context::with(|ctx| { 
                let expr = &ctx.exprs()[key];
                
                match expr.kind {
                    ExprKind::Literal(lit) => match lit {
                        Literal::U32(i) => Some(i as i64),
                        Literal::I32(i) => Some(i as i64),
                        _ => None, //this will result in an error at type deduction anyways, no need to push an error here
                    },
                    _ => None,
                }
            })
        });

        let maybe_len: Option<usize> = self.subscript_len().and_then(|slen| match slen {
            SubscriptLen::Sized(len) => Some(len),
            SubscriptLen::Unsized => None,
            SubscriptLen::InvalidTy(_) => {
                //this will result in an error at type deduction anyways, no need to push an error here
                None
            },
        });

        if let (Some(ty), Some(len), Some(i)) = (self.ty_via_thread_ctx(), maybe_len, index_literal) {
            let in_bounds = 0 <= i && i < len as i64;
            if !in_bounds {
                Context::with(|ctx| { 
                    ctx.push_error(
                        Error::OutOfBounds(format!("access into {ty} out of bounds 0..{len} at index {i}"))
                    )
                });
            }
        }

        Any::by_recording_expr(ExprKind::Operator(Operator::Subscript), &[*self, index])
    }

    /// returns an lvalue
    pub fn x(&self) -> Self {self.vector_index(0)}
    /// returns an lvalue
    pub fn y(&self) -> Self {self.vector_index(1)}
    /// returns an lvalue
    pub fn z(&self) -> Self {self.vector_index(2)}
    /// returns an lvalue
    pub fn w(&self) -> Self {self.vector_index(3)}

    pub fn vector_index(&self, index: u8) -> Self {
        Any::by_recording_expr(ExprKind::Swizzle(Swizzle::GetScalar([index])), &[*self])
    }

    /// the internal swizzle implementation returns either an lvalue or a copy-on-write value depending
    /// on whether the swizzle has repeated components:
    /// `foo.xyz = bar` can be valid
    /// `foo.xxz = bar` cannot be valid
    fn swizzle_internal(&self, indices: &[u8]) -> Self {
        use Swizzle::*;
        let sw = match indices {
            [x, y, z, w] => Some(GetVec4  ([*x, *y, *z, *w])),
            [x, y, z]    => Some(GetVec3  ([*x, *y, *z])),
            [x, y]       => Some(GetVec2  ([*x, *y])),
            [x]          => Some(GetScalar([*x])),
            _ => Context::with(|ctx| {
                ctx.push_error(Error::ArgumentError(
                    format!("swizzle cannot be used to create a {}-component vector (component indices={:?})", indices.len(), indices)
                ));
                None
            }),
        };
        sw.map(|sw| {
            Any::by_recording_expr(ExprKind::Swizzle(sw), &[*self])
        }).unwrap_or_else(|| Any::not_available())
    }

    /// use this swizzle function if there are no repeated components in the swizzle.
    /// the distinction of `swizzle` and `swizzle_repeated` exists because repeated components make the resulting value an rvalue, as opposed to an lvalue.
    pub fn swizzle(&self, indices: &[u8]) -> Self {
        match !indices.iter().all_unique() {
            true => Context::with(|ctx| {
                ctx.push_error(Error::ArgumentError(
                    format!("cannot use swizzle for swizzles with repeated components such as {:?}. Use `swizzle_repeated` instead.", indices)
                ));
                Any::not_available()
            }),
            false => {
                let out = self.swizzle_internal(indices);
                debug_assert!(out.ty_via_thread_ctx().map(|x| x.access == Access::LValue).unwrap_or(true), "a swizzle type is expected to be an Lvalue");
                out
            },
        }
    }

    /// use this swizzle function only if there are repeated components in the swizzle
    /// the distinction of `swizzle` and `swizzle_repeated` exists because repeated components make the resulting value an rvalue, as opposed to an lvalue.
    pub fn swizzle_repeated(&self, indices: &[u8]) -> Self {
        match indices.iter().all_unique() {
            true => Context::with(|ctx| {
                ctx.push_error(Error::ArgumentError(
                    format!("cannot use swizzle_repeated for swizzles without repeated components such as {:?}. Use `swizzle` instead.", indices)
                ));
                Any::not_available()
            }),
            false => {
                let out = self.swizzle_internal(indices);
                debug_assert!(out.ty_via_thread_ctx().map(|x| x.access != Access::LValue).unwrap_or(true), "a swizzle_repeated type is expected to not be an Lvalue");
                out
            },
        }
    }

    pub fn swizzle_copy(&self, indices: &[u8]) -> Self {
        let out = self.swizzle_internal(indices).copy();
        debug_assert!(out.ty_via_thread_ctx().map(|x| x.access != Access::LValue).unwrap_or(true), "a swizzle_copy type is expected to not be an Lvalue");
        out
    }

    pub fn swizzle_maybe_lvalue(&self, indices: &[u8]) -> Self {
        use Swizzle::*;
        let sw = match indices {
            [x, y, z, w] => Some(GetVec4  ([*x, *y, *z, *w])),
            [x, y, z]    => Some(GetVec3  ([*x, *y, *z])),
            [x, y]       => Some(GetVec2  ([*x, *y])),
            [x]          => Some(GetScalar([*x])),
            _ => Context::with(|ctx| {
                ctx.push_error(Error::ArgumentError(
                    format!("swizzle cannot be used to create a {}-component vector (component indices={:?})", indices.len(), indices)
                ));
                None
            }),
        };
        sw.map(|sw| {
            Any::by_recording_expr(ExprKind::Swizzle(sw), &[*self])
        }).unwrap_or_else(|| Any::not_available())
    }

    //
    // assignment
    //

    pub fn set(&mut self, src: Any) {
        self.ty_via_thread_ctx().map(|ty| {
            if ty.access == Access::Const {
                panic!("you are trying to assign to a value that is internally a {} value, which can be supported in the future. Until then, you can call .copy() to create a writeable copy of that value", ty);
            }
        });
        Any::by_recording_expr(ExprKind::Operator(Operator::Assign), &[*self, src]);

        //// this implementation where copy() gets called leads to unintuitive semantics in the rust code. TODO: delete this comment block
        //needs to check if the assignee (self) is const, in that case we need to copy() first to make a writeable value.
        // let self_is_const = self.ty_via_thread_ctx().map(|ty| ty.access) == Some(Access::Const);
        // match self_is_const {
        //     true => {
        //         let mut copy = self.copy();
        //         assert!(copy.ty_via_thread_ctx().map(|ty| ty.access) != Some(Access::Const), "Any::set assumption that copy() creates writeable value is false");
        //         copy.set(src);
        //     },
        //     false => {
        //         Any::by_recording_expr(ExprKind::Operator(Operator::Assign), &[*self, src]);
        //     }
        // }
    }

    ///alternate name for Any::set, lets see which name sticks
    pub fn assign(&mut self, src: Any) {self.set(src)}
    pub fn write (&mut self, src: Any) {self.set(src)}

    //
    // methods
    //

    pub fn dot(&self, val: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Dot), &[*self, val])
    }

    pub fn cross(&self, val: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Cross), &[*self, val])
    }

    pub fn atan(&self) -> Any {
        let y_over_x = *self;
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Atan), &[y_over_x])
    }

    pub fn atan2(y: Any, x: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Atan), &[y, x])
    }

    pub fn pow(&self, val: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Pow), &[*self, val])
    }

    pub fn floor(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Floor), &[*self])
    }

    pub fn ceil(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Ceil), &[*self])
    }

    pub fn round(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Round), &[*self])
    }

    pub fn min(&self, val: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Min), &[*self, val])
    }

    pub fn max(&self, val: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Max), &[*self, val])
    }

    pub fn clamp(&self, min: Any, max: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Clamp), &[*self, min, max])
    }

    pub fn sign(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Sign), &[*self])
    }

    pub fn sqrt(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Sqrt), &[*self])
    }

    pub fn sin(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Sin), &[*self])
    }

    pub fn cos(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Cos), &[*self])
    }

    pub fn fract(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Fract), &[*self])
    }

    pub fn length(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Length), &[*self])
    }

    pub fn abs(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Abs), &[*self])
    }

    pub fn normalize(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Normalize), &[*self])
    }

    pub fn smoothstep(&self, step_interval: std::ops::Range<Any>) -> Any {
        #[allow(unused_mut)]
        let (mut edge0, mut edge1) = (step_interval.start, step_interval.end);
        #[cfg(feature = "workarounds")] {
            //naga 0.8 does not support the overloads
            //genFType smoothstep(float edge0, float edge1, genFType x)
            //genDType smoothstep(double edge0, double edge1, genDType x)
            //therefore we expand edge0 and edge1 to the type of self
            if let Some(ty) = self.ty_via_thread_ctx() {
                if let TyKind::Tensor(tensor) = ty.kind {
                    edge0 = Any::new_tensor(tensor, &[edge0]); // vecN edge0 = vecN(edge0)
                    edge1 = Any::new_tensor(tensor, &[edge1]); // vecN edge1 = vecN(edge1)
                }
            }
        }
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Smoothstep), &[edge0, edge1, *self])
    }

    pub fn mix(&self, interpolate_between: std::ops::Range<Any>) -> Any {
        let (x, y) = (interpolate_between.start, interpolate_between.end);
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Mix), &[x, y, *self])
    }

    pub fn texture(&self, uv: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Texture), &[*self, uv])
    }

    //not to be confused with ne operator
    pub fn not_equal_each(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::NotEqual), &[*self, rhs])
    }

    pub fn not_each(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Not), &[*self])
    }

    pub fn logical_and(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::Operator(Operator::And), &[*self, rhs])
    }

    pub fn logical_or(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::Operator(Operator::Or), &[*self, rhs])
    }

    pub fn logical_xor(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::Operator(Operator::LogicalXor), &[*self, rhs])
    }

    pub fn all(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::All), &[*self])
    }

    //the glsl `bool any(bvec)` function, name was changed to prevent confusion
    pub fn any_is(&self) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Any), &[*self])
    }

    pub fn partial_derivative(&self, component: u8, precision: DerivativePrecision) -> Any {
        use DerivativePrecision::*;
        use BuiltinFn::*;
        let builtin = match (component, precision) {
            (0, DontCare) => Dfdx,
            (1, DontCare) => Dfdy,
            (0, Coarse)   => DfdxCoarse,
            (1, Coarse)   => DfdyCoarse,
            (0, Fine)     => DfdxFine,
            (1, Fine)     => DfdyFine,
            _ => panic!("invalid fragment partial derivative component: {}", component)
        };

        if Context::with(|ctx| ctx.shader_kind == ShaderKind::Fragment) {
            super::Any::by_recording_expr(ExprKind::BuiltinFn(builtin), &[*self])
        } else {
            super::Any::not_available()
        }
        
    }

    /// records the pow function for integer values by repeated multiplication
    /// e.g. `Any::float(0.5).pow_unrolled(5)` becomes something like `0.5 * 0.5 * 0.5 * 0.5 * 0.5`
    pub fn pow_unrolled(self, n: u32) -> Any {
        let sanity_limit = 512;
        match n > sanity_limit {
            true => Context::with(|ctx| {
                ctx.push_error(Error::AssertionFailed(format!("pow unroll sanity check failed. trying to multiply a variable {} (> {}) times", n, sanity_limit)));
                Any::not_available()
            }),
            false => {
                (1..n).fold(self, |acc, _| acc * self)
            }
        }
    }

    //
    // sampling
    //

    /// whether the bias_or_compare argument is "bias" or "compare" depends on the overload you choose
    /// (which depends on the type of the sampler self)
    /// see <https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/texture.xhtml>
    pub fn sample(&self, tex_coords: Any, bias_or_compare: Option<Any>) -> Any {
        match bias_or_compare {
            Some(val) => Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Texture), &[*self, tex_coords, val]),
            None => Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Texture), &[*self, tex_coords]),
        }
    }

    /// operators that cannot be implemented through std::ops traits
    pub fn eq(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::Operator(Operator::Equal), &[*self, rhs])
    }

    /// for vectors only
    pub fn equal(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::Equal), &[*self, rhs])
    }

    /// for scalars only not to be confused with not_equal BuiltinFn
    pub fn ne(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::Operator(Operator::NotEqual), &[*self, rhs])
    }

    /// for vectors only
    pub fn not_equal(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::NotEqual), &[*self, rhs])
    }

    /// glsl `> < == != >= <=` operators, only valid for scalars
    pub fn scalar_comparison(&self, kind: CompareKind, rhs: Any) -> Any {
        (match kind {
            CompareKind::Equal        => Any::eq,
            CompareKind::NotEqual     => Any::ne,
            CompareKind::Less         => Any::lt,
            CompareKind::LessEqual    => Any::le,
            CompareKind::Greater      => Any::gt,
            CompareKind::GreaterEqual => Any::ge,
        })(self, rhs)
    }

    /// glsl lessThan, greaterThan, equal, notEqual etc...
    pub fn vector_comparison(&self, kind: CompareKind, rhs: Any) -> Any {
        (match kind {
            CompareKind::Equal        => Any::equal,
            CompareKind::NotEqual     => Any::not_equal,
            CompareKind::Less         => Any::less_than,
            CompareKind::LessEqual    => Any::less_than_equal,
            CompareKind::Greater      => Any::greater_than,
            CompareKind::GreaterEqual => Any::greater_than_equal,
        })(self, rhs)
    }

    /// for scalars only
    pub fn lt(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::Operator(Operator::Less), &[*self, rhs])
    }

    /// for vectors only
    pub fn less_than(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::LessThan), &[*self, rhs])
    }

    /// for scalars only
    pub fn le(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::Operator(Operator::LessEqual), &[*self, rhs])
    }

    /// for vectors only
    pub fn less_than_equal(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::LessThanEqual), &[*self, rhs])
    }

    /// for scalars only
    pub fn gt(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::Operator(Operator::Greater), &[*self, rhs])
    }

    /// for vectors only
    pub fn greater_than(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::GreaterThan), &[*self, rhs])
    }

    /// for scalars only
    pub fn ge(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::Operator(Operator::GreaterEqual), &[*self, rhs])
    }

    /// for vectors only
    pub fn greater_than_equal(&self, rhs: Any) -> Any {
        Any::by_recording_expr(ExprKind::BuiltinFn(BuiltinFn::GreaterThanEqual), &[*self, rhs])
    }

    pub fn ternary_if(&self, then_: Any, else_: Any) -> Any {
        Any::by_recording_expr(ExprKind::Operator(Operator::TernaryIf), &[*self, then_, else_])
    }

    //alt naming for ternary_if
    pub fn select(&self, then_: Any, else_: Any) -> Any {
        self.ternary_if(then_, else_)
    }

    //
    //  control flow
    //

    fn to_branch_state(&self) -> BranchState {
        self.is_available()
        .then(|| BranchState::Branch)
        .unwrap_or(BranchState::BranchWithConditionNotAvailable)
    }

    pub fn record_then(&self, f: impl FnOnce()) {

        let branch_state = Some(self.to_branch_state());
        
        match self.expr_key {
            Some(cond_key) => {
                let now = RecordTime::next();
                
                Context::with(|ctx| {
                    let ((), block_key) = ctx.record_nested_block(branch_state, f);

                    let stmt = Stmt::new(now, StmtKind::Flow(Flow::IfThen {
                        cond: cond_key,
                        then: block_key
                    }));

                    ctx.blocks_mut()[ctx.current_block_key_unwrap()].record_stmt(stmt);
                })
            },
            None => Context::with(|ctx| {
                //f still needs to be executed, due to its potential side effects
                //the resulting block will not be recorded into a statement so that
                //it doesn't actually end up in the shader code
                let _unused = ctx.record_nested_block(branch_state, f);
            }),
        }

    }

    pub fn record_then_else(&self, f_then: impl FnOnce(), f_else: impl FnOnce()) {

        let branch_state = Some(self.to_branch_state());

        Context::with(|ctx| match self.expr_key {
            Some(cond_key) => {
                let now = RecordTime::next();
                
                let ((), then_key) = ctx.record_nested_block(branch_state, f_then);
                let ((), else_key) = ctx.record_nested_block(branch_state, f_else);
                
                let stmt = Stmt::new(now, StmtKind::Flow(Flow::IfThenElse {
                    cond: cond_key, 
                    then: then_key,
                    els : else_key,
                }));
                
                ctx.blocks_mut()[ctx.current_block_key_unwrap()].record_stmt(stmt);
            },
            None => {
                //functions still need to be executed, due to their potential side effects. 
                //The resulting blocks will not be recorded into statements so that
                //they don't actually end up in the shader code
                let _unused = ctx.record_nested_block(branch_state, f_then);
                let _unused = ctx.record_nested_block(branch_state, f_else);
            }
        });
    }

    pub fn record_while(&self, #[allow(unused)]body_fn: impl FnOnce()) {
        todo!()
    }

    pub fn record_for(&self, #[allow(unused)]increment_fn: impl FnOnce(), #[allow(unused)]body_fn: impl FnOnce()) {
        todo!()
    }
    
}

macro_rules! impl_binary_operators {
    ($($Op: ident, $OpFunc: ident, $OpEnum: expr;)*) => {
        $(impl $Op<Any> for Any {
            type Output = Any;
        
            fn $OpFunc(self, rhs: Any) -> Self::Output {
                Any::by_recording_expr(ExprKind::Operator($OpEnum), &[self, rhs])
            }
        })*
    };
}

macro_rules! impl_assign_operators {
    ($($Op: ident, $OpFunc: ident, $OpEnum: expr;)*) => {
        $(impl $Op<Any> for Any {
            fn $OpFunc(&mut self, rhs: Any) {
                Any::by_recording_expr(ExprKind::Operator($OpEnum), &[*self, rhs]);
            }
        })*
    };
}

macro_rules! impl_unary_operators {
    ($($Op: ident, $OpFunc: ident, $OpEnum: expr;)*) => {
        $(impl $Op for Any {
            type Output = Any;
        
            fn $OpFunc(self) -> Self::Output {
                Any::by_recording_expr(ExprKind::Operator($OpEnum), &[self])
            }
        })*
    };
}

#[derive(Debug)]
pub enum CompareKind {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

use std::ops::*;

impl_unary_operators!{
    Neg, neg, Operator::Negative;
    Not, not, Operator::Not;
}

impl_binary_operators!{
    Mul, mul, Operator::Mul;
    Div, div, Operator::Div;
    Add, add, Operator::Add;
    Sub, sub, Operator::Sub;
    Rem, rem, Operator::Rem;

    BitAnd, bitand, Operator::BitAnd;
    BitOr , bitor , Operator::BitOr ;
    BitXor, bitxor, Operator::BitXor;

    Shl, shl, Operator::ShiftL;
    Shr, shr, Operator::ShiftR;
}

impl_assign_operators!{
    MulAssign, mul_assign, Operator::MulAssign;
    DivAssign, div_assign, Operator::DivAssign;
    AddAssign, add_assign, Operator::AddAssign;
    SubAssign, sub_assign, Operator::SubAssign;
    RemAssign, rem_assign, Operator::RemAssign;

    BitAndAssign, bitand_assign, Operator::AndAssign;
    BitOrAssign , bitor_assign , Operator::OrAssign ;
    BitXorAssign, bitxor_assign, Operator::XorAssign;

    ShlAssign, shl_assign, Operator::ShiftLAssign;
    ShrAssign, shr_assign, Operator::ShiftRAssign;
}

macro_rules! impl_builtin_var_fns {
    (   
        $builtin_var_ty: ident, $valid_shader_kind: expr, $exhaustive_check_fn: ident =>
        $(
            $fn_name: ident -> $builtin_var_enum: ident;
        )*
    ) => {

        #[allow(unused)]
        //this function only exists to cause a compile-time exhaustiveness check for the Any::builtin_var functions
        fn $exhaustive_check_fn(input: $builtin_var_ty) {
            match input {
                $($builtin_var_ty::$builtin_var_enum => (),)*
            }
        }

        $(pub fn $fn_name() -> Self {
            if Context::with(|ctx| ctx.shader_kind == $valid_shader_kind) {
                Any::by_recording_expr(ExprKind::BuiltinVar(BuiltinVar :: $builtin_var_ty($builtin_var_ty :: $builtin_var_enum)), &[])
            } else {
                Any::not_available()
            }
        })*
    };
}

impl Any {

    impl_builtin_var_fns! {VertexVar, ShaderKind::Vertex, v_exhaustive_check =>
        v_vertex_id_nonvk -> gl_VertexID;
        v_instance_id_nonvk -> gl_InstanceID;
        v_vertex_id_vk -> gl_VertexIndex;
        v_instance_id_vk -> gl_InstanceIndex;

        v_position -> gl_Position;
        v_point_size -> gl_PointSize;
        v_clip_distance -> gl_ClipDistance;
    }

    impl_builtin_var_fns! {FragmentVar, ShaderKind::Fragment, f_exhaustive_check =>
        f_frag_coord -> gl_FragCoord;
        f_front_facing -> gl_FrontFacing;
        f_point_coord -> gl_PointCoord;

        f_sample_id -> gl_SampleID; 
        f_sample_position -> gl_SamplePosition; //any usage of this will force per-sample evaluation
        f_sample_mask_in -> gl_SampleMaskIn;  //any usage of this will force per-sample evaluation

        f_clip_distance -> gl_ClipDistance;
        f_primitive_id -> gl_PrimitiveID;

        f_frag_depth -> gl_FragDepth;
    }

    impl_builtin_var_fns! {ComputeVar, ShaderKind::Compute, c_exhaustive_check =>
        c_num_work_groups -> gl_NumWorkGroups;
        c_work_group_id -> gl_WorkGroupID;

        c_local_invocation_id -> gl_LocalInvocationID;
        c_global_invocation_id -> gl_GlobalInvocationID;
        // The built-in variable gl_LocalInvocationIndex is a compute shader input variable that contains the one-dimensional representation of the gl_LocalInvocationID. This is computed as:
        // gl_LocalInvocationIndex =
        //     gl_LocalInvocationID.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y +
        //     gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        //     gl_LocalInvocationID.x;
        c_local_invocation_index -> gl_LocalInvocationIndex;
        
        c_work_group_size -> gl_WorkGroupSize;
    }
}

pub enum SubscriptLen {
    Sized(usize),
    Unsized,
    InvalidTy(Ty),
}

pub enum DerivativePrecision {
    DontCare,
    Coarse,
    Fine,
}
