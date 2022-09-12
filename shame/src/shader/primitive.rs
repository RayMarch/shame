//! primitive type representing the vertex/fragment boundary
use std::marker::PhantomData;
use crate::assert;

use super::super::rec::*;
use shame_graph::Context;
use shame_graph::Tensor;
use shame_graph::Any;

/// A rasterized primitive made of fragments
pub struct Primitive<'a> {pub(crate) _phantom: PhantomData<&'a ()>}

impl Primitive<'_> {

    /// inTERPolate across primitive pixels
    pub fn terp<T: Terpable>(&self, value: T, kind: TerpKind) -> T::Output {
        value.terp(kind, self) //forward to T's unique trait impl
    }

    //convenience shorthands for interpolation of vertex inputs
    /// `P`erspective correct `L`inear int`ERP`olation (see `TerpKind` or glsl "smooth")
    pub fn plerp<T: Terpable>(&self, value: T) -> T::Output {self.terp(value, TerpKind::LinearPerpectiveDivide)}

    /// `L`inear int`ERP`olation (see `TerpKind` or glsl "noperspective")
    pub fn lerp<T: Terpable>(&self, value: T) -> T::Output {self.terp(value, TerpKind::Linear)}

    /// flat interpolation (see `TerpKind` or glsl "flat")
    pub fn flat<T: Terpable>(&self, value: T) -> T::Output {self.terp(value, TerpKind::Flat)}

}

/// kind of interpolation
#[derive(Debug, Clone, Copy)]
pub enum TerpKind { //TODO: get rid of this enum and use shame_graph version instead
    /// glsl "noperspective", linear interpolation (aka Barycentric Interpolation)
    Linear,
    /// glsl "smooth", divides linear result by homogenous "w" component from rasterizer position
    LinearPerpectiveDivide,
    /// glsl "flat", no transition, just one value across the primitive. see glsl documentation to know which one
    Flat,
}

/// Interpolatable across the Primitive
pub trait Terpable {
    /// result type of the interpolation function
    type Output;
    /// inTERPolate across primitive pixels
    fn terp(self, kind: TerpKind, p: &Primitive) -> Self::Output;
}

//every per-vertex value can be interpolated to a per-fragment value
impl<S: Shape, D: IsDTypeNumber> Terpable for Ten<S, D> {
    type Output = Ten<S, D>;

    fn terp(self, kind: TerpKind, p: &Primitive) -> Self::Output {
        (&self).terp(kind, p) //call terp of &Ten<S, D> to avoid code duplication
    }
}

//also for &Ten
impl<S: Shape, D: IsDTypeNumber> Terpable for &Ten<S, D> {
    type Output = Ten<S, D>;

    fn terp(self, kind: TerpKind, _p: &Primitive) -> Self::Output {

        assert::assert_string(
            !Context::with(|ctx| ctx.inside_branch().is_some()),
            "interpolation functions cannot be called from within conditional blocks such as if-then/if-then-else/for/while."
        );

        let kind = match kind {
            TerpKind::Linear => shame_graph::Interpolation::Linear,
            TerpKind::LinearPerpectiveDivide => shame_graph::Interpolation::PerspectiveLinear,
            TerpKind::Flat => shame_graph::Interpolation::Flat,
        };

        let any = shame_graph::Context::with(|ctx| {
            match &mut ctx.shader_mut().stage_interface {
                shame_graph::StageInterface::Vertex{ outputs: f2v, .. } => {
                    let (mut out, loc) = f2v.push_interpolated(kind, Tensor::new(S::SHAPE, D::DTYPE), None);
                    out.assign(self.into_any()); //assign the value of self to the vertex output variable
                    out.aka_maybe(Some(format!("_out{loc}")));
                    Any::not_available() //the interpolated value is being ignored in future recording steps
                }
                shame_graph::StageInterface::Fragment{ inputs: f2v, .. } => {
                    let (out, loc) = f2v.push_interpolated(kind, Tensor::new(S::SHAPE, D::DTYPE), None);
                    out.aka_maybe(Some(format!("_in{loc}")))
                }
                _ => panic!("cannot interpolate vertex values in a compute shader"),
            }
        });

        any.downcast(Stage::Fragment)
    }
}

//tuple implementations for Terpable

// example:
// impl<A: Terpable> Terpable for (A,) {
//     type Output = (A::Output,);
//     fn terp(self, k: TerpKind, p: &Primitive) -> Self::Output {
//         #[allow(non_snake_case)] let (A,) = self;
//         (A.terp(k, p),)
//     }
// }

macro_rules! terpable_tuple_impl {
    ($($ts: ident),+) => {
        impl<$($ts: Terpable,)*> Terpable for ($($ts),*,) {
            type Output = ($($ts::Output,)*);
            fn terp(self, k: TerpKind, p: &Primitive) -> Self::Output {
                #[allow(non_snake_case)] let ($($ts,)+) = self;
                ($($ts.terp(k, p),)+)
            }
        }
    };
}

terpable_tuple_impl!(A);
terpable_tuple_impl!(A, B);
terpable_tuple_impl!(A, B, C);
terpable_tuple_impl!(A, B, C, D);
terpable_tuple_impl!(A, B, C, D, E);
terpable_tuple_impl!(A, B, C, D, E, F);
terpable_tuple_impl!(A, B, C, D, E, F, G);
terpable_tuple_impl!(A, B, C, D, E, F, G, H);
terpable_tuple_impl!(A, B, C, D, E, F, G, H, I);
terpable_tuple_impl!(A, B, C, D, E, F, G, H, I, J);
