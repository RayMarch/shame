//! constructors for tensors of different shapes
use super::*;
use shame_graph::Any;
use shame_graph::Tensor;

/// trait implemented by tuples that can be assembled as a larger tensor
///
/// e.g. `(float, float2, f32) => float4`
pub trait VecCtor: AsTen + Sized {
    /// create a large tensor that contains all the components from `self`
    fn vec(self) -> Ten<Self::S, Self::D> {
        self.as_ten()
    }
}

impl<S: IsShapeVec, T: AsTen<S = S>> VecCtor for T {}

impl<
        DT: DType,
        A: AsTen<S = scal, D = DT>,
        B: AsTen<S = scal, D = DT>,
        C: AsTen<S = scal, D = DT>,
        D: AsTen<S = scal, D = DT>,
    > IntoRec for (A, B, C, D)
{
    type Rec = Ten<vec4, DT>;

    fn rec(self) -> Self::Rec {
        Any::new_tensor(Self::Rec::tensor(), &self.into_anys())
            .downcast(self.narrow_or_push_error())
    }
    fn into_any(self) -> Any {
        self.as_ten().any
    }
    fn stage(&self) -> Stage {
        self.narrow_or_push_error()
    }
}

impl<DT: DType, A: AsTen<D = DT>, B: AsTen<D = DT>, C: AsTen<D = DT>> IntoRec for (A, B, C)
where
    (A::S, B::S, C::S): VecCtorShapes,
{
    type Rec = Ten<<(A::S, B::S, C::S) as VecCtorShapes>::Output, DT>;

    fn rec(self) -> Self::Rec {
        Any::new_tensor(Self::Rec::tensor(), &self.into_anys()).downcast(self.stage())
    }
    fn into_any(self) -> Any {
        self.as_ten().any
    }
    fn stage(&self) -> Stage {
        self.narrow_or_push_error()
    }
}

impl<DT: DType, A: AsTen<D = DT>, B: AsTen<D = DT>> IntoRec for (A, B)
where
    (A::S, B::S): VecCtorShapes,
{
    type Rec = Ten<<(A::S, B::S) as VecCtorShapes>::Output, DT>;

    fn rec(self) -> Self::Rec {
        Any::new_tensor(Self::Rec::tensor(), &self.into_anys()).downcast(self.stage())
    }
    fn into_any(self) -> Any {
        self.as_ten().any
    }
    fn stage(&self) -> Stage {
        self.narrow_or_push_error()
    }
}

impl<
        DT: DType,
        A: AsTen<S = scal, D = DT>,
        B: AsTen<S = scal, D = DT>,
        C: AsTen<S = scal, D = DT>,
        D: AsTen<S = scal, D = DT>,
    > AsTen for (A, B, C, D)
{
    type S = vec4;
    type D = DT;

    fn as_ten(&self) -> Ten<Self::S, Self::D> {
        let tensor = Tensor::new(Self::S::SHAPE, Self::D::DTYPE);
        Any::new_tensor(tensor, &self.into_anys()).downcast(self.narrow_or_push_error())
    }
}

impl<DT: DType, A: AsTen<D = DT>, B: AsTen<D = DT>, C: AsTen<D = DT>> AsTen for (A, B, C)
where
    (A::S, B::S, C::S): VecCtorShapes,
{
    type S = <(A::S, B::S, C::S) as VecCtorShapes>::Output;
    type D = DT;

    fn as_ten(&self) -> Ten<Self::S, Self::D> {
        let tensor = Tensor::new(Self::S::SHAPE, Self::D::DTYPE);
        Any::new_tensor(tensor, &self.into_anys()).downcast(self.stage())
    }
}

impl<DT: DType, A: AsTen<D = DT>, B: AsTen<D = DT>> AsTen for (A, B)
where
    (A::S, B::S): VecCtorShapes,
{
    type S = <(A::S, B::S) as VecCtorShapes>::Output;
    type D = DT;

    fn as_ten(&self) -> Ten<Self::S, Self::D> {
        let tensor = Tensor::new(Self::S::SHAPE, Self::D::DTYPE);
        Any::new_tensor(tensor, &self.into_anys()).downcast(self.stage())
    }
}

/// implemented for tuples of [`Shape`] types that can be combined to form
/// larger vectors (not matrices!)
pub trait VecCtorShapes {
    /// the output vecN shape, containing all the components of the `Self`
    /// tuple's tensors
    type Output: Shape;
}

macro_rules! impl_vec_ctor_shapes {
    ($(($($shape: ty),*) -> $output: ty;)*) => {
        $(
            impl VecCtorShapes for ($($shape),*) {
                type Output = $output;
            }
        )*
    };
}

impl_vec_ctor_shapes! {
    (scal, scal, scal, scal) -> vec4;
    (vec2, scal, scal) -> vec4;
    (scal, vec2, scal) -> vec4;
    (scal, scal, vec2) -> vec4;
    (vec2, vec2) -> vec4;
    (scal, vec3) -> vec4;
    (vec3, scal) -> vec4;

    (scal, scal, scal) -> vec3;
    (vec2, scal) -> vec3;
    (scal, vec2) -> vec3;

    (scal, scal) -> vec2;
}

/// implemented for tuples of [`Shape`] types that can be interpreted as
/// columns or rows of a matrix
pub trait MatCtorShapes {
    /// shape of the result matrix, where Self's tuple elements are
    /// interpreted as columns
    type ColMat: IsShapeMat;
    /// shape of the result matrix, where Self's tuple elements are
    /// interpreted as rows
    type RowMat: IsShapeMat;
}

/// implemented for tuples of matrix columns or row vecNs, which can be combined
/// to a matrix
pub trait MatCtor<D: DType> {
    /// the result type if the tuple elements are interpreted as matrix columns
    type ColMat: Shape;
    /// the result type if the tuple elements are interpreted as matrix rows
    type RowMat: Shape;

    /// create a matrix where the tuple elements are interpreted as matrix
    /// columns
    fn mat_cols(&self) -> Ten<Self::ColMat, D>;

    /// create a matrix where the tuple elements are interpreted as matrix
    /// rows
    fn mat_rows(&self) -> Ten<Self::RowMat, D>;
}

// impl<A: AsTen, B: AsTen> MatCtor for (A, B) where (A::S, B::S): MatCtorShapes {}
// impl<A: AsTen, B: AsTen, C: AsTen> MatCtor for (A, B, C) where (A::S, B::S, C::S): MatCtorShapes {}
// impl<A: AsTen, B: AsTen, C: AsTen, D: AsTen> MatCtor for (A, B, C, D) where (A::S, B::S, C::S, D::S): MatCtorShapes {}

macro_rules! impl_mat_ctor_shapes {
    ($(($($A: ident: $vecN: ty),*) => ($matRow: ty, $matCol: ty);)*
    ) => {$(
        impl MatCtorShapes for ($($vecN),*){
            type ColMat = $matCol;
            type RowMat = $matRow;
        }
    )*}
}

impl_mat_ctor_shapes! {
    (A: vec2, B: vec2) => (mat2  , mat2  );
    (A: vec3, B: vec3) => (mat3x2, mat2x3);
    (A: vec4, B: vec4) => (mat4x2, mat2x4);
    (A: vec2, B: vec2, C: vec2) => (mat2x3, mat3x2);
    (A: vec3, B: vec3, C: vec3) => (mat3  , mat3  );
    (A: vec4, B: vec4, C: vec4) => (mat4x3, mat3x4);
    (A: vec2, B: vec2, C: vec2, D: vec2) => (mat2x4, mat4x2);
    (A: vec3, B: vec3, C: vec3, D: vec3) => (mat3x4, mat4x3);
    (A: vec4, B: vec4, C: vec4, D: vec4) => (mat4  , mat4  );
}

macro_rules! impl_mat_ctors {
    ($(($($A: ident),*);)*
    ) => {$(

        impl<DT: IsDTypeFloatingPoint, $($A: AsTen<D=DT>),*> MatCtor<DT> for ($($A),*) where ($($A::S),*): MatCtorShapes {
            type ColMat = <($($A::S),*) as MatCtorShapes>::ColMat;
            type RowMat = <($($A::S),*) as MatCtorShapes>::RowMat;

            fn mat_cols(&self) -> Ten<Self::ColMat, DT> {
                #[allow(non_snake_case)]
                let ($($A),*) = self;

                let tensor = Tensor::new(Self::ColMat::SHAPE, DT::DTYPE);

                Any::new_matrix_from_cols(tensor, &[$($A .into_any()),*])
                .downcast([$($A .stage()),*].narrow_or_push_error())
            }

            fn mat_rows(&self) -> Ten<Self::RowMat, DT> {
                #[allow(non_snake_case)]
                let ($($A),*) = self;

                let tensor = shame_graph::Tensor::new(Self::RowMat::SHAPE, DT::DTYPE);

                Any::new_matrix_from_rows(tensor, &[$($A .into_any()),*])
                .downcast([$($A .stage()),*].narrow_or_push_error())
            }
        }

    )*};
}

impl_mat_ctors! {
    (A, B);
    (A, B, C);
    (A, B, C, D);
}
