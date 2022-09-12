//! [`Ten`] (short for "Tensor") is a shader type representing scalars, vectors
//! up to size 4 and matrices of up to size 4x4.
use std::marker::PhantomData;
use super::{*, fields::Fields};
use shame_graph::{Any, Ty};

#[derive(Clone, Copy)]
/// [`Ten`] (short for "Tensor") is a shader type representing scalars, vectors
/// of up to size 4 and matrices of up to size 4x4.
/// It supports many arithmetic operations commonly found in shader code.
/// The two generic args mean the following:
///
/// - `S: Shape` is the shape of the tensor, e.g. vec4, mat3x2. It describes
/// how many components/rows/columns it has
/// - `D: DType` is the data type of the individual components. a
/// `Ten<vec4, f32>` consists of 4 `float` components, while a
/// `Ten<mat3x3, i32>` consists of 9 `int` components
///
/// In your average non-generic shader code you don't need to write
/// `Ten<vec4, f32`> but instead can use the alias `float4` from the aliases
/// module, which is also `use`d in `prelude`.
/// Every specific `Ten` parametrization has such a convenience type alias
/// (e.g. `int3x3`, `float`, `boolean`, `bool2`...)
pub struct Ten<S: Shape, D: DType> {
    phantom: PhantomData<(S, D)>,
    pub(super) any: Any,
    pub(super) stage: Stage,
    pub(super) swizzle: S::SwizzleMembers,
}

impl<S: Shape, D: DType> Rec for Ten<S, D> {
    fn as_any(&self) -> Any {self.any}

    fn ty() -> Ty {Ty::tensor(S::SHAPE, D::DTYPE)}

    fn from_downcast(any: Any, stage: Stage) -> Self {
        Self{phantom: PhantomData, any, stage, swizzle: <_>::new_empty()}
    }
}

impl<S: Shape, D: DType> Ten<S, D> {

    pub(crate) fn tensor() -> shame_graph::Tensor {
        shame_graph::Tensor::new(S::SHAPE, D::DTYPE)
    }

    /// upcasts, calls f, and immediately downcasts, assuming the same stage
    pub fn with_any(&self, f: impl FnOnce(Any) -> Any) -> Self {
        let (any, stage) = (self.any, self.stage);
        let avail_before = any.is_available();
        let out = f(any);
        let avail_after = out.is_available();
        crate::assert::assert_string(
            avail_before == avail_after,
            format!("upcast Ten value has changed availability status within `with_any` call. {avail_before} => {avail_after}")
        );
        out.downcast(stage)
    }

    /// amount of scalar components in the tensor, e.g. 3 for `float3`, 16 for `int4x4`
    pub fn components_len(&self) -> usize {
        S::NUM_COMPONENTS
    }

    /// iterate over the components of the tensor
    pub fn components(&self) -> std::vec::IntoIter<Ten<scal, D>> {
        let n = self.components_len() as u8;
        match n {
            1..=4 => (0..n).map(|i| {
                let access = match (i, n) {
                    (0, 1) => self.any, // float.x is not possible
                    _ => self.any.vector_index(i),
                };
                Ten::from_downcast(access, self.stage)
            }).collect::<Vec<_>>(),
            _ => unreachable!("{}-component tensor", n)
        }.into_iter()
    }

    /// iterate over the components of the tensor
    pub fn iter(&self) -> std::vec::IntoIter<Ten<scal, D>> {
        self.components()
    }
}

impl<S: Shape, D: DType> IntoIterator for Ten<S, D> {
    type Item = Ten<scal, D>;
    type IntoIter = std::vec::IntoIter<Ten<scal, D>>;

    fn into_iter(self) -> Self::IntoIter {
        self.components()
    }
}

impl<S: Shape, D: DType> std::iter::Sum for Ten<S, D> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        match iter.next() {
            None => Self::zero(),
            Some(first) => iter.fold(first, |acc, x| {
                acc + x
            }),
        }
    }
}

impl<'a, S: Shape, D: DType> std::iter::Sum<&'a Self> for Ten<S, D> {
    fn sum<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        match iter.next() {
            None => Self::zero(),
            Some(first) => iter.fold(*first, |acc, x| {
                acc + *x
            }),
        }
    }
}

/// types that can be converted into recorded tensors `Ten<S, D>`
///
/// some conversion examples:
///
/// `f32` -> `Ten<scal, f32>` aka `float`
///
/// `(f32, f32)` -> `Ten<vec2, f32>` aka `float2`
///
/// `(f32, float2)` -> `Ten<vec3, f32>` aka `float3`
///
/// `((bool, bool), bool2)` -> `Ten<vec4, bool>` aka `bool4`
pub trait AsTen: IntoRec<Rec=Ten<Self::S, Self::D>> + Copy {
    /// shape of `Self` when converted to a tensor
    type S: Shape;
    /// dtype of `Self` when converted to a tensor
    type D: DType;

    /// convert `self` to a tensor type
    fn as_ten(&self) -> Ten<Self::S, Self::D>;

    /// conversion from one tensor dtype to another, may cause an explicit
    /// conversion in the shader.
    /// implementing this via rusts `From`/`Into` traits is not possible due
    /// to overlapping impls.
    fn cast<D: DType>(&self) -> Ten<Self::S, D> {
        let src_tensor = shame_graph::Tensor::new(Self::S::SHAPE, Self::D::DTYPE);
        let dst_tensor = shame_graph::Tensor::new(Self::S::SHAPE, D::DTYPE);

        match src_tensor == dst_tensor {
            true => self.into_any(),
            false => Any::new_tensor(dst_tensor, &[self.into_any()]),
        }.downcast(self.stage())
    }
}

impl<S: Shape, D: DType> AsTen for Ten<S, D> {
    type S=S;
    type D=D;

    fn as_ten(&self) -> Ten<S, D> {*self}
}

impl<S: Shape, D: DType> IntoRec for Ten<S, D> {
    type Rec = Self;
    fn rec(self) -> Self::Rec {self}
    fn into_any(self) -> Any {self.any}
    fn stage(&self) -> Stage {self.stage}
}

impl<S: Shape, D: DType> Fields for Ten<S, D> {

    fn parent_type_name() -> Option<&'static str> {
        None
    }

    fn from_fields_downcast(name: Option<&'static str>, f: &mut impl FnMut(Ty, &'static str) -> (Any, Stage)) -> Self {
        let (any, stage) = f(Self::ty(), name.unwrap_or("val"));
        Self::from_downcast(any, stage)
    }

    fn collect_fields(&self) -> Vec<(Any, Stage)> {
        vec![(self.any, self.stage)]
    }

}

impl<S: Shape, D: DType> Default for Ten<S, D> {
    fn default() -> Self {
        use shame_graph::Shape::*;
        match S::SHAPE {
            Scalar | Vec(_) => zero(),
            Mat(_, _) => id(),
        }
    }
}