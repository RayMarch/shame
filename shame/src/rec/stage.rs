//! runtime type annotations for representing the shader stage in which a value
//! is available (per-vertex, per-fragment, uniform).

use super::*;
use shame_graph::Any;
use shame_graph::Ty;
use shame_graph::Error;
use crate::assert;

/// runtime type annotation for [`Rec`] types, used to tag "per-vertex",
/// "per-fragment" or unrestricted "uniform" values.
///
/// using "per-vertex" values in expressions with "per-fragment" values will
/// cause an error.
///
/// expressions that combine an argument of stage "uniform" and another argument
/// result in a value of the stage of this other argument
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    /// unrestricted stage, can be used with per-vertex, per-fragment and
    /// uniform values.
    ///
    /// in compute shaders every value has [`Stage::Uniform`]
    Uniform,
    /// per-vertex values. Can be turned into per-fragment values via
    /// [`Primitive`](crate::Primitive)'s interpolation functions
    Vertex,
    /// per-fragment values. These values cannot be converted to any other stage
    Fragment,
    /// result stage when using incompatible stages
    ///
    /// (vertex + fragment) or (X + NotAvailable)
    NotAvailable,
}

impl Stage {
    /// the shader kind that corresponds to a given stage
    pub fn to_shader_kind(&self) -> Option<shame_graph::ShaderKind> {
        match self {
            Stage::Uniform => None,
            Stage::Vertex => Some(shame_graph::ShaderKind::Vertex),
            Stage::Fragment => Some(shame_graph::ShaderKind::Fragment),
            Stage::NotAvailable => None,
        }
    }

    pub fn from_shader_kind(shader_kind: shame_graph::ShaderKind) -> Stage {
        use shame_graph::ShaderKind::*;
        match shader_kind {
            Vertex => Stage::Vertex,
            Fragment => Stage::Fragment,
            Compute => Stage::Uniform,
        }
    }
}

impl std::ops::BitAnd for Stage {
    type Output = Stage;

    fn bitand(self, rhs: Self) -> Self::Output {
        use Stage::*;
        match (self, rhs) {
            (Uniform, x) => x,
            (x, Uniform) => x,

            (NotAvailable, _) => NotAvailable,
            (_, NotAvailable) => NotAvailable,

            (Vertex,   Fragment) => NotAvailable,
            (Fragment, Vertex  ) => NotAvailable,
            (Vertex,   Vertex  ) => Vertex,
            (Fragment, Fragment) => Fragment,
        }
    }
}

impl From<shame_graph::Stage> for Stage {
    fn from(stage: shame_graph::Stage) -> Self {
        match stage {
            shame_graph::Stage::Vertex => Stage::Vertex,
            shame_graph::Stage::Fragment => Stage::Fragment,
            shame_graph::Stage::Uniform => Stage::Uniform,
            shame_graph::Stage::NotAvailable => Stage::NotAvailable,
        }
    }
}

impl From<Stage> for shame_graph::Stage {
    fn from(stage: Stage) -> Self {
        match stage {
            Stage::Vertex => shame_graph::Stage::Vertex,
            Stage::Fragment => shame_graph::Stage::Fragment,
            Stage::Uniform => shame_graph::Stage::Uniform,
            Stage::NotAvailable => shame_graph::Stage::NotAvailable,
        }
    }
}

/// adding [`Stage`] and rust compile-time type information to an `Any`, to
/// downcast it to a [`Rec`] type
pub trait AnyDowncast {
    #[track_caller]

    /// downcasts the type-erased `Any` type (which represent nodes in the
    /// expression graph) to statically typed Ten<Shape, DType> types to make
    /// rust's type system aware of them.
    ///
    /// all downcast calls lead to `check_any_type_and_stage(...)` which checks
    /// whether the requested `Ten<S, D>` type actually matches the dynamic type
    /// of the `Any` object. If it doesn't, an Error is emitted in the recording
    /// context which will be handled according to the user defined
    /// [`ErrorBehavior`]
    fn downcast<S: Shape, D: DType>(&self, stage: Stage) -> Ten<S, D>;
}

impl AnyDowncast for Any {
    #[track_caller]
    fn downcast<S: Shape, D: DType>(&self, stage: Stage) -> Ten<S, D> {
        let (any, stage) = check_any_type_and_stage::<S, D>(*self, stage);
        Ten::from_downcast(any, stage)
    }
}

#[track_caller]
fn check_any_type_and_stage<S: Shape, D: DType>(mut any: Any, mut stage: Stage) -> (Any, Stage) {
    use shame_graph::ShaderKind as Shader;
    shame_graph::Context::with(|ctx| {
        if let Some(any_ty) = any.ty_via_ctx(ctx) {

            let static_type = Ty::tensor(S::SHAPE, D::DTYPE);
            if !static_type.eq_ignore_access(&any_ty) {
                assert::rec_error(Error::TypeError(
                    format!("cannot downcast a dynamic {any_ty} to a static {static_type}")
                ));
                stage = Stage::NotAvailable;
                any = Any::not_available();
            }

            debug_assert!(any.is_available());
            //since we got a type, it means any is not NotAvailable
            //so we must check if the stage is the current shader stage, or compatible with it
            match (ctx.shader_kind(), stage) {
                (_              , Stage::Uniform     ) |
                (_              , Stage::NotAvailable) | //not available only causes an error in the moment it is narrowed, not every time it is passed on
                (Shader::Vertex  , Stage::Vertex      ) |
                (Shader::Fragment, Stage::Fragment    ) => {},

                (ex @ Shader::Vertex,   fo @ Stage::Fragment) |
                (ex @ Shader::Fragment, fo @ Stage::Vertex  ) |
                (ex @ Shader::Compute,  fo @ Stage::Vertex  ) |
                (ex @ Shader::Compute,  fo @ Stage::Fragment) => {
                    let fo = fo.to_shader_kind().expect("erroneous required stage has no shader kind"); //unreachable expect
                    assert::rec_error(Error::NAInShaderKind {expected: ex, found: fo});
                    stage = Stage::NotAvailable;
                    any = Any::not_available();
                }
            };
        }
    });
    (any, stage)
}

/// obtain the dominant stage of multiple provided stages, or
/// push an error to the [`Context`] and return [`Stage::NotAvailable`].
#[track_caller]
pub fn narrow_stages_or_push_error(stages: impl IntoIterator<Item=Stage>) -> Stage {
    stages.into_iter().fold(Stage::Uniform, |acc, x| {
        let narrowed = acc & x;
        if let Stage::NotAvailable = narrowed {
            assert::rec_error(Error::AssertionFailed(format!("trying to use a {:?} shader expression together with a {:?} shader expression", acc, x))); //TODO: improve error message (e.g. by adding arg types)
        }
        narrowed
    })
}

/// obtain the most narrow stage of multiple provided stages, or
/// [`Stage::NotAvailable`].
pub trait HasCommonStage {
    #[track_caller]
    /// obtain the dominant stage of multiple provided stages, or
    /// push an error to the [`Context`] and return [`Stage::NotAvailable`].
    fn narrow_or_push_error(&self) -> Stage;
}

impl HasCommonStage for [Stage] {
    #[track_caller]
    fn narrow_or_push_error(&self) -> Stage {
        narrow_stages_or_push_error(self.iter().map(|x| *x))
    }
}

macro_rules! impl_has_common_stage_for_intorec_tuple {
    ($($A: ident),*) => {
        impl<$($A: IntoRec),*> HasCommonStage for ($($A),*) {
            #[track_caller]
            fn narrow_or_push_error(&self) -> Stage {
                #[allow(non_snake_case)]
                let ($($A),*) = self;
                narrow_stages_or_push_error([$($A.stage()),*])
            }
        }
    };
}

impl_has_common_stage_for_intorec_tuple!(A, B);
impl_has_common_stage_for_intorec_tuple!(A, B, C);
impl_has_common_stage_for_intorec_tuple!(A, B, C, D);
impl_has_common_stage_for_intorec_tuple!(A, B, C, D, E);
impl_has_common_stage_for_intorec_tuple!(A, B, C, D, E, F);