use std::{fmt::Display, num::NonZeroU32, ops::Not, rc::Rc};

use super::{type_check::*, Comp4, ExponentFn, Expr, NumericFn, TextureFn};
use crate::{
    call_info,
    common::integer::i4,
    frontend::{
        any::{record_node, shared_io::SamplingMethod, Any, ArgumentNotAvailable, InvalidReason},
        encoding::EncodingErrorKind,
    },
    impl_track_caller_fn_any,
    ir::{
        ir_type::{
            AccessMode, AddressSpace, Indirection,
            Len::*,
            Len2,
            ScalarType::{self, *},
            ScalarTypeFp, SizedStruct,
            SizedType::*,
            StoreType::*,
            TextureShape,
            Type::Unit,
        },
        pipeline::{PossibleStages, ShaderStage, StageMask},
        recording::{
            AtomicCompareExchangeWeakGenerics, BuiltinTemplateStructs, Context, InteractionKind, MemoryRegion,
            NodeRecordingError, TemplateStructParams,
        },
        HandleType, SamplesPerPixel, SizedType, StructureFieldNamesMustBeUnique,
    },
};

use crate::{ir, ir::StoreType, ir::Type, same, sig};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuiltinFn {
    // (no test case yet)
    Constructor(Constructor),
    // (no test case yet)
    Reinterpret(ReinterpretFn),
    // (no test case yet)
    Logical(LogicalFn),
    // (no test case yet)
    Array(ArrayFn),
    // (no test case yet)
    Numeric(NumericFn),
    Derivative(DerivativeFn),
    Texture(TextureFn),
    // (no test case yet)
    Atomic(AtomicFn),
    // (no test case yet)
    DataPacking(DataPackingFn),
    // (no test case yet)
    Sync(SyncFn),
}

impl Display for BuiltinFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuiltinFn::Constructor(x) => write!(f, "{x}"),
            BuiltinFn::Reinterpret(x) => write!(f, "{x:?}"),
            BuiltinFn::Logical(x) => write!(f, "{x:?}"),
            BuiltinFn::Array(x) => write!(f, "{x:?}"),
            BuiltinFn::Numeric(x) => write!(f, "{x}"),
            BuiltinFn::Derivative(x) => write!(f, "{x:?}"),
            BuiltinFn::Texture(x) => write!(f, "{x}"),
            BuiltinFn::Atomic(x) => write!(f, "{x}"),
            BuiltinFn::DataPacking(x) => write!(f, "{x:?}"),
            BuiltinFn::Sync(x) => write!(f, "{x:?}"),
        }
    }
}

impl BuiltinFn {
    /// see `Expr::may_change_execution_state`
    pub(super) fn may_change_execution_state(&self) -> bool {
        match self {
            BuiltinFn::Constructor(_) => false,
            BuiltinFn::Reinterpret(_) => false,
            BuiltinFn::Logical(_) => false,
            BuiltinFn::Array(_) => false,
            BuiltinFn::Derivative(_) => false,
            BuiltinFn::DataPacking(_) => false,
            BuiltinFn::Numeric(x) => match x {
                NumericFn::TrigonometryFn(_) => false,
                NumericFn::LinearAlgebra(_) => false,
                NumericFn::Discontinuity(_) => false,
                NumericFn::Bit(_) => false,
                NumericFn::Exponent(e) => match e {
                    ExponentFn::Frexp { .. } => false,
                    ExponentFn::Exp |
                    ExponentFn::Exp2 |
                    ExponentFn::Log |
                    ExponentFn::Log2 |
                    ExponentFn::Ldexp |
                    ExponentFn::Pow |
                    ExponentFn::Sqrt => false,
                },
            },
            BuiltinFn::Texture(x) => match x {
                TextureFn::TextureStore(_) => true,
                TextureFn::TextureDimensions |
                TextureFn::TextureNumLayers |
                TextureFn::TextureNumLevels |
                TextureFn::TextureNumSamples |
                TextureFn::TextureGather { .. } |
                TextureFn::TextureGatherCompare { .. } |
                TextureFn::TextureLoad(_, _) |
                TextureFn::TextureSample { .. } |
                TextureFn::TextureSampleBias { .. } |
                TextureFn::TextureSampleCompare { .. } |
                TextureFn::TextureSampleCompareLevel { .. } |
                TextureFn::TextureSampleGrad { .. } |
                TextureFn::TextureSampleLevel { .. } |
                TextureFn::TextureSampleBaseClampToEdge => false,
            },
            BuiltinFn::Atomic(_) => true,
            BuiltinFn::Sync(_) => true,
        }
    }

    pub fn possible_stages(&self) -> PossibleStages {
        let none = StageMask::empty();
        let vert = StageMask::vert();
        let mesh = StageMask::mesh();
        let frag = StageMask::frag();
        let comp = StageMask::comp();
        let all = StageMask::all();
        let unrestricted = (false, none, all, false);

        let (must_appear_at_all, must_in, can_in, only_once) = match self {
            // "WGSL Spec: Derivative Built-in Functions: "Must only be used in a fragment shader stage."
            BuiltinFn::Derivative(_) => (false, none, frag, false),

            BuiltinFn::Constructor(_) |
            BuiltinFn::Reinterpret(_) |
            BuiltinFn::Logical(_) |
            BuiltinFn::Array(_) |
            BuiltinFn::Numeric(_) => unrestricted,

            BuiltinFn::Texture(texture_fn) => match texture_fn {
                TextureFn::TextureSample { .. } => (false, none, frag, true),
                TextureFn::TextureSampleBias { .. } => (false, none, frag, true),
                TextureFn::TextureSampleCompare { .. } => (false, none, frag, true),

                TextureFn::TextureDimensions |
                TextureFn::TextureNumLayers |
                TextureFn::TextureNumLevels |
                TextureFn::TextureNumSamples => unrestricted,

                TextureFn::TextureGather { .. } |
                TextureFn::TextureGatherCompare { .. } |
                TextureFn::TextureSampleCompareLevel { .. } |
                TextureFn::TextureSampleGrad { .. } |
                TextureFn::TextureSampleLevel { .. } |
                TextureFn::TextureSampleBaseClampToEdge => (false, none, all, true),
                TextureFn::TextureLoad(_, _) => (true, none, all, true),
                TextureFn::TextureStore(_) => (true, none, all, true),
            },
            // WGSL spec: "Atomic built-in functions must not be used in a vertex shader stage."
            BuiltinFn::Atomic(x) => match x {
                AtomicFn::AtomicLoad |
                AtomicFn::AtomicStore |
                AtomicFn::AtomicReadModifyWrite(_) |
                AtomicFn::AtomicExchange |
                AtomicFn::AtomicCompareExchangeWeak(_) => (true, none, vert.not(), true),
            },
            BuiltinFn::DataPacking(_) => unrestricted,
            // WGSL spec: "All synchronization functions must only be used in the compute shader stage"
            BuiltinFn::Sync(x) => match x {
                SyncFn::StorageBarrier |
                SyncFn::TextureBarrier |
                SyncFn::WorkgroupBarrier |
                SyncFn::WorkgroupUniformLoad => (true, comp, comp, true),
            },
        };
        PossibleStages::new(must_appear_at_all, must_in, can_in, only_once)
    }
}

impl TypeCheck for BuiltinFn {
    #[rustfmt::skip]
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        match self {
            BuiltinFn::Constructor(x) => x.infer_type(args),
            BuiltinFn::Reinterpret(x) => x.infer_type(args),
            BuiltinFn::Logical    (x) => x.infer_type(args),
            BuiltinFn::Array      (x) => x.infer_type(args),
            BuiltinFn::Numeric    (x) => x.infer_type(args),
            BuiltinFn::Derivative (x) => x.infer_type(args),
            BuiltinFn::Texture    (x) => x.infer_type(args),
            BuiltinFn::Atomic     (x) => x.infer_type(args),
            BuiltinFn::DataPacking(x) => x.infer_type(args),
            BuiltinFn::Sync       (x) => x.infer_type(args),
        }
    }
}

impl From<BuiltinFn> for Expr {
    fn from(x: BuiltinFn) -> Self { Expr::BuiltinFn(x) }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constructor {
    // (no test case yet) // TODO(release) generic zeroed constructor for constructible types
    Default(SizedType),
    // (no test case yet)
    Scalar(ScalarType),
    Vector(Len2, ScalarType),
    Matrix(Len2, Len2, ScalarTypeFp),
    Array(Rc<SizedType>, NonZeroU32),
    // (no test case yet)
    Structure(SizedStruct),
}

impl Display for Constructor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len2_to_u32 = |l| u32::from(ir::Len::from(l));
        match self {
            Constructor::Default(sized_type) => write!(f, "{sized_type}()"),
            Constructor::Scalar(scalar_type) => write!(f, "{scalar_type}(_)"),
            Constructor::Vector(len2, scalar_type) => write!(f, "vec{}<{scalar_type}>(...)", len2_to_u32(*len2)),
            Constructor::Matrix(cols, rows, scalar_type) => write!(
                f,
                "mat{}x{}<{scalar_type}>(...)",
                len2_to_u32(*cols),
                len2_to_u32(*rows)
            ),
            Constructor::Array(sized_type, non_zero) => write!(f, "array<{sized_type}, {}>(...)", non_zero.get()),
            Constructor::Structure(sized_struct) => write!(f, "struct {}(...)", sized_struct.name()),
        }
    }
}

impl TypeCheck for Constructor {
    #[allow(non_snake_case)]
    #[allow(clippy::redundant_at_rest_pattern)] // this is a clippy false positive. it won't work if the `@ ..` is removed
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        match self {
            Constructor::Default(sized_type) => sig! (
                {
                    name: Constructor::Default(sized_type),
                    comment_below: "using the definition of `is_constructible` found at https://www.w3.org/TR/WGSL/#constructible",
                },
                [] if sized_type.is_constructible() => sized_type
            )(self, args),
            Constructor::Scalar(s) => (match s {
                F16 => sig! { [Vector(X1, _)] => Vector(X1, F16)},
                F32 => sig! { [Vector(X1, _)] => Vector(X1, F32) },
                F64 => sig! { [Vector(X1, _)] => Vector(X1, F64) },
                U32 => sig! { [Vector(X1, _)] => Vector(X1, U32) },
                I32 => sig! { [Vector(X1, _)] => Vector(X1, I32) },
                Bool => sig! { [Vector(X1, _)] => Vector(X1, Bool) },
            })(self, args),
            Constructor::Vector(xn, T) => match xn {
                Len2::X2 => sig!(
                    {
                        name: Vector(X2, T),
                        fmt: SigFormatting::RemoveAsterisksAndClone,
                    },
                    [Vector(X1, t)] if t == T => Vector(X2, *T), // splat
                    [Vector(X2, s)] => Vector(X2, *T), // conversion
                    [Vector(X1, t0), Vector(X1, t1)] if same!(t0 t1 T) => Vector(X2, *T), // comp wise
                )(self, args),
                Len2::X3 => sig! (
                    {
                        name: Vector(X3, T),
                        fmt: SigFormatting::RemoveAsterisksAndClone,
                    },
                    [Vector(X1, t)] if t == T => Vector(X3, *T), // splat
                    [Vector(X3, s)] => Vector(X3, *T), // conversion
                    [Vector(X1, t0), Vector(X1, t1), Vector(X1, t2)] if same!(t0 t1 t2 T) => Vector(X3, *T), // comp wise
                    [Vector(X1, t0), Vector(X2, t1)] if same!(t0 t1 T) => Vector(X3, *T), // concat 1+2
                    [Vector(X2, t0), Vector(X1, t1)] if same!(t0 t1 T) => Vector(X3, *T), // concat 2+1
                )(self, args),
                Len2::X4 => sig! (
                    {
                        name: Vector(X4, T),
                        fmt: SigFormatting::RemoveAsterisksAndClone,
                    },
                    [Vector(X1, t)] if t == T => Vector(X4, *T), // splat
                    [Vector(X4, s)] => Vector(X4, *T), // conversion
                    [Vector(X1, t0), Vector(X1, t1), Vector(X1, t2), Vector(X1, t3)] if same!(t0 t1 t2 t3 T) => Vector(X4, *T), // comp wise
                    [Vector(X1, t0), Vector(X3, t1)] if same!(t0 t1 T) => Vector(X4, *T), // concat 1+3
                    [Vector(X3, t0), Vector(X1, t1)] if same!(t0 t1 T) => Vector(X4, *T), // concat 3+1
                    [Vector(X2, t0), Vector(X2, t1)] if same!(t0 t1 T) => Vector(X4, *T), // concat 2+2
                    [Vector(X1, t0), Vector(X1, t1), Vector(X2, t2)] if same!(t0 t1 t2 T) => Vector(X4, *T), // concat 1+1+2
                    [Vector(X1, t0), Vector(X2, t1), Vector(X1, t2)] if same!(t0 t1 t2 T) => Vector(X4, *T), // concat 1+2+1
                    [Vector(X2, t0), Vector(X1, t1), Vector(X1, t2)] if same!(t0 t1 t2 T) => Vector(X4, *T), // concat 2+1+1
                )(self, args),
            },
            Constructor::Matrix(cols, rows, T) => {
                // one ctor for
                // - scalar type conversions
                // - column vectors
                // - elements
                let rows_ = &ir::Len::from(*rows);
                let T_ = ir::ScalarType::from(*T);

                sig! (
                    {
                        name: "Matrix(cols, rows, T)",
                        comment_below: format!("where T = {}, cols = {}, rows = {}", T, cols, rows),
                        fmt: SigFormatting::RemoveAsterisksAndClone,
                    },
                    // scalar-type conversion constructor
                    [Matrix(c0, r0, t0)] if same!(rows r0; cols c0) => Matrix(*cols, *rows, *T),
                    // column vector constructors
                    [Vector(r0, t0), Vector(r1, t1)]                                 if *cols == X2 && same!(rows_ r0 r1      ; T t0 t1      ) => Matrix(*cols, *rows, *T),
                    [Vector(r0, t0), Vector(r1, t1), Vector(r2, t2)]                 if *cols == X3 && same!(rows_ r0 r1 r2   ; T t0 t1 t2   ) => Matrix(*cols, *rows, *T),
                    [Vector(r0, t0), Vector(r1, t1), Vector(r2, t2), Vector(r3, t3)] if *cols == X4 && same!(rows_ r0 r1 r2 r3; T t0 t1 t2 t3) => Matrix(*cols, *rows, *T),
                    // component constructor
                    [ref comps @ ..] if comps.iter().all(|c| **c == Vector(X1, T_)) && comps.len() as u32 == u32::from(*cols) * u32::from(*rows)
                        => Matrix(*cols, *rows, *T),
                )(self, args)
            }
            Constructor::Array(t, len) => {
                let return_ty = ir::SizedType::Array(t.clone(), *len);
                let signature_str = || {
                    use std::fmt::Write;
                    let mut sig = String::new();
                    write!(sig, "[");
                    for i in 0..len.get() {
                        write!(sig, "{}, ", t);
                    }
                    write!(sig, "] => {}", return_ty);
                    sig
                };
                let no_matching_sig = || NoMatchingSignature {
                    expression_name: std::stringify!(Constructor::Array).into(),
                    arguments: args.into(),
                    allowed_signatures: SignatureStrings::Dynamic(vec![signature_str()]),
                    shorthand_level: TypeShorthandLevel::Type,
                    signature_formatting: None,
                    comment: None,
                };
                match len.get() as usize == args.len() {
                    true => {
                        let valid = args.iter().all(|arg| match arg {
                            Type::Store(StoreType::Sized(arg)) => arg == &**t,
                            _ => false,
                        });
                        match valid {
                            true => Ok(Type::from(return_ty)),
                            false => Err(no_matching_sig()),
                        }
                    }
                    false => Err(no_matching_sig()),
                }
            }
            Constructor::Structure(s) => {
                let signature_str = || {
                    use std::fmt::Write;
                    let mut sig = String::new();
                    write!(sig, "[");
                    for s in s.sized_fields() {
                        write!(sig, "{}, ", s.ty);
                    }
                    write!(sig, "] => {}", s.name());
                    sig
                };
                let no_matching_sig = || NoMatchingSignature {
                    expression_name: std::stringify!(Constructor::Structure).into(),
                    arguments: args.into(),
                    allowed_signatures: SignatureStrings::Dynamic(vec![signature_str()]),
                    shorthand_level: TypeShorthandLevel::Type,
                    signature_formatting: None,
                    comment: None,
                };
                match s.len() == args.len() {
                    true => {
                        let valid = s.fields().map(|f| &f.ty).zip(args).all(|(field, arg)| match arg {
                            Type::Store(StoreType::Sized(arg)) => arg == field,
                            _ => false,
                        });
                        match valid {
                            true => Ok(Type::from(SizedType::Structure(s.clone()))),
                            false => Err(no_matching_sig()),
                        }
                    }
                    false => Err(no_matching_sig()),
                }
            }
        }
    }
}

impl Any {
    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn new_vec(len: ir::Len, stype: ScalarType, args: &[Any]) -> Any {
        let ctor = match ir::Len2::try_from(len) {
            Err(_) => Constructor::Scalar(stype),
            Ok(len2) => Constructor::Vector(len2, stype),
        };
        record_node(call_info!(), Expr::BuiltinFn(BuiltinFn::Constructor(ctor)), args)
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn new_mat(cols: ir::Len2, rows: ir::Len2, stype: ScalarTypeFp, args: &[Any]) -> Any {
        record_node(
            call_info!(),
            Expr::BuiltinFn(BuiltinFn::Constructor(Constructor::Matrix(cols, rows, stype))),
            args,
        )
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn splat(&self, len_after: ir::Len, stype: ScalarType) -> Any {
        // splat is just a constructor call, but in order to catch errors where the
        // arg is not a scalar, this extra typecheck is added
        Context::try_with(call_info!(), |ctx| {
            match self.ty() {
                Some(Type::Store(StoreType::Sized(SizedType::Vector(X1, t)))) => {
                    match ir::Len2::try_from(len_after) {
                        Err(_) => *self, //noop, trying to splat from scalar to scalar,
                        Ok(len2) => {
                            let ctor = Constructor::Vector(len2, stype);
                            record_node(
                                ctx.latest_user_caller(),
                                Expr::BuiltinFn(BuiltinFn::Constructor(ctor)),
                                &[*self],
                            )
                        }
                    }
                }
                Some(ty) => ctx.push_error_get_invalid_any(NodeRecordingError::TryingToSplatNonScalar(ty).into()),
                None => *self,
            }
        })
        .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
    }

    /// extend a vec of size x1, x2 or x3 to `len_after`. Fails with an encoding error if
    /// `len_after` is less than the `self`'s current `Len`
    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn extend_vec_to_len(&self, len_after: ir::Len2) -> Any {
        let call_info = call_info!();
        Context::try_with(call_info!(), |ctx| match self.ty() {
            Some(ty @ Type::Store(StoreType::Sized(SizedType::Vector(len_before, t)))) => {
                let zero = || Any::new_scalar(t.constant_from_f64(0.0));
                let expr = Expr::BuiltinFn(BuiltinFn::Constructor(Constructor::Vector(len_after, t)));
                let extra_components = u32::from(len_after) as i32 - u32::from(len_before) as i32;
                match extra_components {
                    0 => *self,
                    1 => record_node(call_info, expr, &[*self, zero()]),
                    2 => record_node(call_info, expr, &[*self, zero(), zero()]),
                    3 => record_node(call_info, expr, &[*self, zero(), zero(), zero()]),
                    _ => ctx.push_error_get_invalid_any(NodeRecordingError::UnableToExtendType(ty, len_after).into()),
                }
            }
            Some(ty) => ctx.push_error_get_invalid_any(NodeRecordingError::UnableToExtendType(ty, len_after).into()),
            None => *self,
        })
        .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn new_struct(ty: ir::SizedStruct, fields: &[Any]) -> Any {
        record_node(
            call_info!(),
            Expr::BuiltinFn(BuiltinFn::Constructor(Constructor::Structure(ty))),
            fields,
        )
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    /// create a new array with known size of `elements.len()`
    pub fn new_array(element_type: Rc<SizedType>, elements: &[Any]) -> Any {
        let call_info = call_info!();
        match NonZeroU32::new(u32::try_from(elements.len()).expect("less than u32::MAX elements")) {
            None => Context::try_with(call_info, |ctx| {
                ctx.push_error_get_invalid_any(NodeRecordingError::ArraysMustBeNonEmpty.into())
            })
            .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding)),
            Some(n) => record_node(
                call_info,
                Expr::BuiltinFn(BuiltinFn::Constructor(Constructor::Array(element_type, n))),
                elements,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReinterpretFn {
    // overloads of BitCast, taken from https://www.w3.org/TR/WGSL/#bitcast-builtin
    //
    // for all these, imagine a where clause:
    //
    // (the scalar type `T` is provided in `.0`)
    // where
    //   S: ScalarTypeNumber + Is32Bit,
    //   T: ScalarTypeNumber + Is32Bit, // .0
    //   N: Len,
    //
    // fn bitcast<  TxN>(e :   SxN) ->   TxN
    // fn bitcast<  Tx1>(e : f16x2) ->   Tx1
    // fn bitcast<  Tx2>(e : f16x4) ->   Tx2
    // fn bitcast<f16x2>(e :   Tx1) -> f16x2
    // fn bitcast<f16x4>(e :   Tx2) -> f16x4
    // (no test case yet)
    Bitcast(ScalarType),
}

impl TypeCheck for ReinterpretFn {
    #[allow(non_snake_case)]
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        match self {
            ReinterpretFn::Bitcast(T) => sig!(
                {
                    name: ReinterpretFn::Bitcast(T),
                    fmt: SigFormatting::RemoveAsterisksAndClone,
                },
                [Vector(n, t0)] if t0 == T && T.is_numeric()               => Vector(*n, *T),
                [Vector(n, s)] if T != s && T.is_32_bit() && s.is_32_bit() => Vector(*n, *T),
                [Vector(X2, F16)] if T.is_32_bit()                         => Vector(X1, *T),
                [Vector(X4, F16)] if T.is_32_bit()                         => Vector(X2, *T),
                [Vector(X1, t0)] if t0.is_32_bit() && *T == F16            => Vector(X2, *T),
                [Vector(X2, t0)] if t0.is_32_bit() && *T == F16            => Vector(X4, *T),
            )(self, args),
        }
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn bitcast(&self, stype: ScalarType) -> Any => [*self] Expr::BuiltinFn(BuiltinFn::Reinterpret(ReinterpretFn::Bitcast(stype)));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalFn {
    // (no test case yet)
    All,
    // (no test case yet)
    Any,
    // (no test case yet)
    Select,
}

impl TypeCheck for LogicalFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        (match self {
            LogicalFn::All => sig! { [Vector(_, Bool)] => Bool },
            LogicalFn::Any => sig! { [Vector(_, Bool)] => Bool },
            LogicalFn::Select => sig! {
                [v @ Vector(..), v1 @ Vector(..), Vector(X1, Bool)] if same!(v v1) => v,
                [v @ Vector(n1, t1), Vector(n2, t2), Vector(n, Bool)] if same!(n n1 n2; t1 t2) => v,
            },
        })(self, args)
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn all(&self) -> Any => [*self] BuiltinFn::Logical(LogicalFn::All);
        pub fn any(&self) -> Any => [*self] BuiltinFn::Logical(LogicalFn::Any);
        pub fn select(&self, if_true: Any, if_false: Any) -> Any => [if_false, if_true, *self] BuiltinFn::Logical(LogicalFn::Select);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayFn {
    // (no test case yet)
    ArrayLength,
}

impl TypeCheck for ArrayFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        use {AddressSpace::*, Type::*};
        (match self {
            ArrayFn::ArrayLength => sig! {
               [Ptr(alloc, RuntimeSizedArray(_), _)] if alloc.address_space == Storage => U32
            },
        })(self, args)
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn array_length(&self) -> Any => [*self] Expr::BuiltinFn(BuiltinFn::Array(ArrayFn::ArrayLength));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeFn {
    Dpdx(GradPrecision),
    Dpdy(GradPrecision),
    Fwidth(GradPrecision),
}

/// precision of fragment quad derivatives
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GradPrecision {
    /// do not require any specific precision, this lets the implementation decide
    #[default]
    DonTCare,
    /// don't read the value of every fragment in the fragment quad for computing the derivative
    Coarse,
    /// use the values of all four fragments in the fragment quad to compute the derivative
    Fine,
}

impl TypeCheck for DerivativeFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        (match self {
            DerivativeFn::Dpdx(_) | DerivativeFn::Dpdy(_) | DerivativeFn::Fwidth(_) => {
                sig! {
                    { fmt: SigFormatting::RemoveAsterisksAndClone, },
                    [Vector(n, F32)] => Vector(*n, F32)
                }
            }
        })(self, args)
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn ddx    (&self, prec: GradPrecision) -> Any => [*self] BuiltinFn::Derivative(DerivativeFn::Dpdx  (prec));
        pub fn ddy    (&self, prec: GradPrecision) -> Any => [*self] BuiltinFn::Derivative(DerivativeFn::Dpdy  (prec));
        pub fn fwidth (&self, prec: GradPrecision) -> Any => [*self] BuiltinFn::Derivative(DerivativeFn::Fwidth(prec));
    }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicModify {
    Add,
    Sub,
    Max,
    Min,
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub enum AtomicFn {
    // (no test case yet)
    AtomicLoad,
    // (no test case yet)
    AtomicStore,
    // (no test case yet)
    AtomicReadModifyWrite(AtomicModify),
    // (no test case yet)
    AtomicExchange,
    // (no test case yet)
    AtomicCompareExchangeWeak(AtomicCompareExchangeWeakGenerics),
}

impl Display for AtomicFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomicFn::AtomicLoad | AtomicFn::AtomicStore | AtomicFn::AtomicExchange => write!(f, "{self:?}"),
            AtomicFn::AtomicReadModifyWrite(atomic_modify) => write!(f, "Atomic{:?}Assign", atomic_modify),
            AtomicFn::AtomicCompareExchangeWeak(g) => {
                write!(f, "AtomicCompareExchangeWeak<{}, {}>", g.0, ScalarType::from(g.1))
            }
        }
    }
}

impl TypeCheck for AtomicFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        use AccessMode::*;
        use AddressSpace::*;
        use SizedType::*;
        use StoreType::*;
        (match self {
            AtomicFn::AtomicLoad => sig! {
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Type::Ptr(allocation, Sized(Atomic(t)), ReadWrite)]
                if matches!(allocation.address_space, Storage | WorkGroup)
                => ScalarType::from(*t)
            },
            AtomicFn::AtomicStore => sig! {
                [Type::Ptr(allocation, Sized(Atomic(t0)), ReadWrite), Type::Store(Sized(Vector(X1, t)))]
                if t0 == t && matches!(allocation.address_space, Storage | WorkGroup) => Unit
            },
            AtomicFn::AtomicReadModifyWrite(_) | AtomicFn::AtomicExchange => sig! {
                [Type::Ptr(allocation, Sized(Atomic(t0)), ReadWrite), Type::Store(Sized(Vector(X1, t)))] if t0 == t => t
            },
            AtomicFn::AtomicCompareExchangeWeak(generics) => {
                return BuiltinTemplateStructs::infer_type(
                    args,
                    TemplateStructParams::AtomicCompareExchangeWeak(*generics),
                );
            }
        })(self, args)
    }
}

impl From<AtomicFn> for Expr {
    fn from(x: AtomicFn) -> Self { Expr::BuiltinFn(BuiltinFn::Atomic(x)) }
}

impl Any {
    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn atomic_load(&self) -> Any {
        let value = record_node(call_info!(), AtomicFn::AtomicLoad.into(), &[*self]);
        MemoryRegion::record_interaction(value, *self, InteractionKind::Read);
        value
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn atomic_store(&self, value: Any) {
        let store = record_node(call_info!(), AtomicFn::AtomicStore.into(), &[*self, value]);
        MemoryRegion::record_interaction(store, *self, InteractionKind::Write);
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn atomic_read_modify_write(&self, modify: AtomicModify, value: Any) -> Any {
        let read_write = record_node(
            call_info!(),
            AtomicFn::AtomicReadModifyWrite(modify).into(),
            &[*self, value],
        );
        MemoryRegion::record_interaction(read_write, *self, InteractionKind::ReadWrite);
        read_write
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn atomic_exchange(&self, value: Any) -> Any {
        let exchange = record_node(call_info!(), AtomicFn::AtomicExchange.into(), &[*self, value]);
        MemoryRegion::record_interaction(exchange, *self, InteractionKind::ReadWrite);
        exchange
    }
}

/// see https://www.w3.org/TR/WGSL/#pack-builtin-functions
///
/// and https://www.w3.org/TR/WGSL/#unpack-builtin-functions
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataPackingFn {
    // (no test case yet)
    Pack4x8snorm,
    // (no test case yet)
    Pack4x8unorm,
    // (no test case yet)
    Pack4xI8,
    // (no test case yet)
    Pack4xU8,
    // (no test case yet)
    Pack4xI8Clamp,
    // (no test case yet)
    Pack4xU8Clamp,
    // (no test case yet)
    Pack2x16snorm,
    // (no test case yet)
    Pack2x16unorm,
    // (no test case yet)
    Pack2x16float,

    // (no test case yet)
    Unpack4xI8,
    // (no test case yet)
    Unpack4xU8,
    // (no test case yet)
    Unpack4x8snorm,
    // (no test case yet)
    Unpack4x8unorm,
    // (no test case yet)
    Unpack2x16snorm,
    // (no test case yet)
    Unpack2x16unorm,
    // (no test case yet)
    Unpack2x16float,
}

impl Any {
    /// data packing transformation
    ///
    /// see https://www.w3.org/TR/WGSL/#pack-builtin-functions
    ///
    /// and https://www.w3.org/TR/WGSL/#unpack-builtin-functions
    #[track_caller]
    pub fn pack_data(self, f: DataPackingFn) -> Any {
        record_node(call_info!(), Expr::BuiltinFn(BuiltinFn::DataPacking(f)), &[self])
    }
}

impl TypeCheck for DataPackingFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        (match self {
            //         fn pack4x8snorm           (e: vec4 <f32>) -> u32
            DataPackingFn::Pack4x8snorm => sig!([Vector(X4, F32)] => U32),
            //         fn pack4x8unorm           (e: vec4 <f32>) -> u32
            DataPackingFn::Pack4x8unorm => sig!([Vector(X4, F32)] => U32),
            //         fn pack4xI8           (e: vec4 <i32>) -> u32
            DataPackingFn::Pack4xI8 => sig!([Vector(X4, I32)] => U32),
            //         fn pack4xU8           (e: vec4 <u32>) -> u32
            DataPackingFn::Pack4xU8 => sig!([Vector(X4, U32)] => U32),
            //         fn pack4xI8Clamp           (e: vec4 <i32>) -> u32
            DataPackingFn::Pack4xI8Clamp => sig!([Vector(X4, I32)] => U32),
            //         fn pack4xU8Clamp           (e: vec4 <u32>) -> u32
            DataPackingFn::Pack4xU8Clamp => sig!([Vector(X4, U32)] => U32),
            //         fn pack2x16snorm           (e: vec2 <f32>) -> u32
            DataPackingFn::Pack2x16snorm => sig!([Vector(X2, F32)] => U32),
            //         fn pack2x16unorm           (e: vec2 <f32>) -> u32
            DataPackingFn::Pack2x16unorm => sig!([Vector(X2, F32)] => U32),
            //         fn pack2x16float           (e: vec2<f32>) -> u32
            DataPackingFn::Pack2x16float => sig!([Vector(X2, F32)] => U32),

            //         fn unpack4x8snorm      (e: u32) ->      vec4 <f32>
            DataPackingFn::Unpack4x8snorm => sig!([U32] => Vector(X4, F32)),
            //         fn unpack4x8unorm      (e: u32) ->      vec4 <f32>
            DataPackingFn::Unpack4x8unorm => sig!([U32] => Vector(X4, F32)),
            //         fn unpack4xI8      (e: u32) ->      vec4 <f32>
            DataPackingFn::Unpack4xI8 => sig!([U32] => Vector(X4, F32)),
            //         fn unpack4xU8      (e: u32) ->      vec4 <f32>
            DataPackingFn::Unpack4xU8 => sig!([U32] => Vector(X4, F32)),
            //         fn unpack2x16snorm      (e: u32) ->      vec4 <f32>
            DataPackingFn::Unpack2x16snorm => sig!([U32] => Vector(X4, F32)),
            //         fn unpack2x16unorm      (e: u32) ->      vec4 <f32>
            DataPackingFn::Unpack2x16unorm => sig!([U32] => Vector(X4, F32)),
            //         fn unpack2x16snorm      (e: u32) ->      vec2 <f32>
            DataPackingFn::Unpack2x16float => sig!([U32] => Vector(X2, F32)),
        })(self, args)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncFn {
    // (no test case yet)
    StorageBarrier,
    // (no test case yet)
    TextureBarrier,
    // (no test case yet)
    WorkgroupBarrier,
    // (no test case yet)
    WorkgroupUniformLoad,
}

impl TypeCheck for SyncFn {
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        use AddressSpace::*;
        (match self {
            SyncFn::StorageBarrier | SyncFn::WorkgroupBarrier | SyncFn::TextureBarrier => sig! {
                [] => Unit
            },
            SyncFn::WorkgroupUniformLoad => sig! {
                [Type::Ptr(allocation, t, AccessMode::ReadWrite)]
                if t.is_plain_and_fixed_footprint() && !t.contains_atomics()
                && allocation.address_space == WorkGroup
                => t
            },
        })(self, args)
    }
}

impl From<SyncFn> for Expr {
    fn from(x: SyncFn) -> Self { Expr::BuiltinFn(BuiltinFn::Sync(x)) }
}

impl Any {
    #[track_caller]
    fn sync_fn_impl(sync_fn: SyncFn, args: &[Any]) -> Any {
        let call_info = call_info!();
        Context::try_with(call_info, |ctx| {
            if ctx.pipeline_kind() != ir::pipeline::PipelineKind::Compute {
                ctx.push_error(NodeRecordingError::BarrierInNonComputePipeline.into());
            }
        });
        record_node(call_info, sync_fn.into(), args)
    }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn storage_barrier() { Any::sync_fn_impl(SyncFn::StorageBarrier, &[]); }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn workgroup_barrier() { Any::sync_fn_impl(SyncFn::WorkgroupBarrier, &[]); }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn texture_barrier() { Any::sync_fn_impl(SyncFn::TextureBarrier, &[]); }

    #[doc(hidden)] // runtime api
    #[track_caller]
    pub fn workgroup_uniform_load(&self) -> Any {
        let value = Any::sync_fn_impl(SyncFn::WorkgroupUniformLoad, &[*self]);
        MemoryRegion::record_interaction(value, *self, InteractionKind::Read);
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ir_type::Len;

    #[test]
    fn type_check_infer() {
        let ctor_vec = BuiltinFn::Constructor(Constructor::Vector(Len2::X3, ScalarType::F32));
        let bitcast = BuiltinFn::Reinterpret(ReinterpretFn::Bitcast(ScalarType::F32));
        let select = BuiltinFn::Logical(LogicalFn::Select);

        let args = &[
            Type::from(SizedType::Vector(Len::X2, ScalarType::Bool)),
            Type::from(SizedType::Vector(Len::X2, ScalarType::F16)),
            Type::from(SizedType::Vector(Len::X2, ScalarType::F16)),
        ];

        assert!(ctor_vec.infer_type(args).is_err());
        assert!(bitcast.infer_type(args).is_err());

        let args = &[
            Type::from(SizedType::Vector(Len::X2, ScalarType::F16)),
            Type::from(SizedType::Vector(Len::X2, ScalarType::F16)),
            Type::from(SizedType::Vector(Len::X2, ScalarType::Bool)),
        ];
        assert!(select.infer_type(args).is_ok());

        // use `cargo test -- --nocapture` to look at error messages

        // match ctor_vec.infer_return_type(args) {
        //     Ok(ty) => println!("\n\ninferred return type: {ty:?}\n\n"),
        //     Err(err) => println!("\n\n{err}\n\n"),
        // }
    }
}
