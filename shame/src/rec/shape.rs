//! tensor shapes, such as scalar, vec2, mat4x3
use super::*;

/// implemented by tensor-shape types such as 
/// - [`scal`] for scalars
/// - [`vec2`] for 2 component vectors
/// - [`mat4`] for matrices with 4 columns and 4 rows
pub trait Shape: Copy + 'static {
    /// runtime enum representing `Self`
    const SHAPE: shame_graph::Shape;

    /// implementation detail of swizzling. 
    /// [`SwizzleMember`] or `()`
    type SwizzleMembers: IsSwizzleMembers;

    /// if `Self` is a matrix, its column type, otherwise `Self`
    type Col: Shape;
    /// if `Self` is a matrix, its row type, otherwise [`scal`]
    type Row: Shape;

    /// total amount of components in a tensor of this shape
    /// (e.g. `9` for a `mat3`)
    const NUM_COMPONENTS: usize;
}

/// implementation of [`Shape`] for those types is found in [`super::swizzle`]
/// scalar 
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct scal;
/// 2 component vector
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct vec2;
/// 3 component vector
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct vec3;
/// 4 component vector
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct vec4;
/// matrix, (⠃⠃) 2 columns, 2 rows
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct mat2;
/// matrix, (⠇⠇) 2 columns, 3 rows
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct mat2x3;
/// matrix, (⡇⡇) 2 columns, 4 rows
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct mat2x4;
/// matrix, (⠃⠃⠃) 3 columns, 2 rows
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct mat3x2;
/// matrix, (⠇⠇⠇) 3 columns, 3 rows
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct mat3;
/// matrix, (⡇⡇⡇) 3 columns, 4 rows
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct mat3x4;
/// matrix, (⠃⠃⠃⠃) 4 columns, 2 rows
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct mat4x2;
/// matrix, (⠇⠇⠇⠇) 4 columns, 3 rows
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct mat4x3;
/// matrix, (⡇⡇⡇⡇) 4 columns, 4 rows
#[derive(Copy, Clone)] #[allow(non_camel_case_types)] pub struct mat4;

/// implemented by [`scal`], `vecN`
pub trait IsShapeScalarOrVec: Shape {}
    impl IsShapeScalarOrVec for scal {}
    impl IsShapeScalarOrVec for vec2 {}
    impl IsShapeScalarOrVec for vec3 {}
    impl IsShapeScalarOrVec for vec4 {}

/// implemented by all `vecN`
pub trait IsShapeVec: IsShapeScalarOrVec {}
    impl IsShapeVec for vec2 {}
    impl IsShapeVec for vec3 {}
    impl IsShapeVec for vec4 {}

/// implemented by all `mat` types
pub trait IsShapeMat: Shape {}
    impl IsShapeMat for mat2   {}
    impl IsShapeMat for mat2x3 {}
    impl IsShapeMat for mat3x2 {}
    impl IsShapeMat for mat2x4 {}
    impl IsShapeMat for mat4x2 {}
    impl IsShapeMat for mat3   {}
    impl IsShapeMat for mat3x4 {}
    impl IsShapeMat for mat4x3 {}
    impl IsShapeMat for mat4   {}

/// whether the given vec/scalar [`Shape`] is at least as wide as `S`
/// 
/// example: `AtLeastAsWideAs<vec2>` is implemented by `vec2`, `vec3`, `vec4`
pub trait ShapeAtLeastAsWideAs<S: IsShapeScalarOrVec>: IsShapeScalarOrVec {}
    impl ShapeAtLeastAsWideAs<scal> for scal {}
    impl ShapeAtLeastAsWideAs<scal> for vec2 {}
    impl ShapeAtLeastAsWideAs<scal> for vec3 {}
    impl ShapeAtLeastAsWideAs<scal> for vec4 {}

    impl ShapeAtLeastAsWideAs<vec2> for vec2 {}
    impl ShapeAtLeastAsWideAs<vec2> for vec3 {}
    impl ShapeAtLeastAsWideAs<vec2> for vec4 {}

    impl ShapeAtLeastAsWideAs<vec3> for vec3 {}
    impl ShapeAtLeastAsWideAs<vec3> for vec4 {}

    impl ShapeAtLeastAsWideAs<vec4> for vec4 {}

/// implemented for all `vecN` of `N >= 2` 
/// 
/// `vec2, vec3, vec4`
pub trait IsVecOfAtLeast2: IsShapeScalarOrVec {}
    impl IsVecOfAtLeast2 for vec2 {}
    impl IsVecOfAtLeast2 for vec3 {}
    impl IsVecOfAtLeast2 for vec4 {}

/// implemented for all `vecN` of `N >= 3` 
/// 
/// `vec3, vec4`
pub trait IsVecOfAtLeast3: IsShapeScalarOrVec + IsVecOfAtLeast2 {}
    impl IsVecOfAtLeast3 for vec3 {}
    impl IsVecOfAtLeast3 for vec4 {}

/// implemented for all `vecN` of `N >= 4` 
/// 
/// `vec4`
pub trait IsVecOfAtLeast4: IsShapeScalarOrVec + IsVecOfAtLeast2 + IsVecOfAtLeast3 {}
    impl IsVecOfAtLeast4 for vec4 {}

/// unidirectional "same or scalar" relationship, if you want a bidirectional 
/// version of this, where `A: IsScalarOr<B>` OR `B: IsScalarOr<A>` is being 
/// checked, use `(A, B): SameOrScalar`
pub trait IsScalarOr<S: Shape>: Shape {}
    impl<X: Shape> IsScalarOr<X> for scal {}
    impl IsScalarOr<vec2> for vec2 {} 
    impl IsScalarOr<vec3> for vec3 {} 
    impl IsScalarOr<vec4> for vec4 {} 
    impl IsScalarOr<mat2  > for mat2   {}
    impl IsScalarOr<mat2x3> for mat2x3 {}
    impl IsScalarOr<mat3x2> for mat3x2 {}
    impl IsScalarOr<mat2x4> for mat2x4 {}
    impl IsScalarOr<mat4x2> for mat4x2 {}
    impl IsScalarOr<mat3  > for mat3   {}
    impl IsScalarOr<mat3x4> for mat3x4 {}
    impl IsScalarOr<mat4x3> for mat4x3 {}
    impl IsScalarOr<mat4  > for mat4   {}

/// implemented for tuple types: `(X, scal)`, `(scal, X)`, `(X, X)` where `X: Shape`
/// 
/// useful for glsl style `+*-/` operators which allow scalar + vector operations
pub trait ScalarOrSame {
    /// the non-scalar type in the tuple, or `scal` if `Self` = `(scal, scal)`
    type Widest: Shape;
} 
    impl<X: Shape> ScalarOrSame for (X, X) {type Widest = X;}

    //impl<X: Shape> ScalarOrSame for (scal, X) {}
    //impl ScalarOrSame for (scal, scal) {} //handled by (X, X)
    impl ScalarOrSame for (scal, vec2) {type Widest = vec2;}
    impl ScalarOrSame for (scal, vec3) {type Widest = vec3;}
    impl ScalarOrSame for (scal, vec4) {type Widest = vec4;}
    impl ScalarOrSame for (scal, mat2  ) {type Widest = mat2;}
    impl ScalarOrSame for (scal, mat2x3) {type Widest = mat2x3;}
    impl ScalarOrSame for (scal, mat3x2) {type Widest = mat3x2;}
    impl ScalarOrSame for (scal, mat2x4) {type Widest = mat2x4;}
    impl ScalarOrSame for (scal, mat4x2) {type Widest = mat4x2;}
    impl ScalarOrSame for (scal, mat3  ) {type Widest = mat3;}
    impl ScalarOrSame for (scal, mat3x4) {type Widest = mat3x4;}
    impl ScalarOrSame for (scal, mat4x3) {type Widest = mat4x3;}
    impl ScalarOrSame for (scal, mat4  ) {type Widest = mat4;}

    //impl<X: Shape> ScalarOrSame for (X, scal) {}
    impl ScalarOrSame for (vec2, scal) {type Widest = vec2;}
    impl ScalarOrSame for (vec3, scal) {type Widest = vec3;}
    impl ScalarOrSame for (vec4, scal) {type Widest = vec4;}
    impl ScalarOrSame for (mat2  , scal) {type Widest = mat2;}
    impl ScalarOrSame for (mat2x3, scal) {type Widest = mat2x3;}
    impl ScalarOrSame for (mat3x2, scal) {type Widest = mat3x2;}
    impl ScalarOrSame for (mat2x4, scal) {type Widest = mat2x4;}
    impl ScalarOrSame for (mat4x2, scal) {type Widest = mat4x2;}
    impl ScalarOrSame for (mat3  , scal) {type Widest = mat3;}
    impl ScalarOrSame for (mat3x4, scal) {type Widest = mat3x4;}
    impl ScalarOrSame for (mat4x3, scal) {type Widest = mat4x3;}
    impl ScalarOrSame for (mat4  , scal) {type Widest = mat4;}

/// implemented for pair tuples `(A, B)` if a tensor with shape `A` 
/// can be multiplied by a tensor with shape `B`
pub trait CanBeMultiplied {
    /// the output shape of the multiplication of an `A` tensor with a `B` 
    /// tensor
    type Output: Shape;
} 
impl CanBeMultiplied for (scal, scal) {type Output = scal;}
macro_rules! impl_can_be_multiplied_vec {
    ($($vecN: ty),*) => {
        $(
            impl CanBeMultiplied for ($vecN, scal ) {type Output = $vecN;}
            impl CanBeMultiplied for (scal , $vecN) {type Output = $vecN;}
            impl CanBeMultiplied for ($vecN, $vecN) {type Output = $vecN;}
        )*
    }
}
impl_can_be_multiplied_vec!(vec2, vec3, vec4);

macro_rules! impl_can_be_multiplied_mat_times_vec {
    ($($mat: ty),*) => {
        $(
            impl CanBeMultiplied for ($mat, <$mat as Shape>::Col) {type Output = <$mat as Shape>::Row;}
            impl CanBeMultiplied for (<$mat as Shape>::Row, $mat) {type Output = <$mat as Shape>::Col;}
        )*
    }
}

impl_can_be_multiplied_mat_times_vec!{
    mat2,
    mat2x3,
    mat3x2, 
    mat2x4, 
    mat4x2, 
    mat3,
    mat3x4,
    mat4x3,
    mat4
}


macro_rules! impl_can_be_multiplied_mat_times_mat {
    (
        $((
            $lhs: ty, *
                $(($rhs: ty => $out: ty),)*
        ),)*
    ) => {
        $($(
            impl CanBeMultiplied for ($lhs, $rhs) {type Output = $out;}
        )*)*
    }
}

impl_can_be_multiplied_mat_times_mat!(
    (mat2, * //⠃⠃
        (mat2   => mat2  ), //⠃⠃ * ⠃⠃   =  ⠃⠃
        (mat3x2 => mat3x2), //⠃⠃ * ⠃⠃⠃  =  ⠃⠃⠃
        (mat4x2 => mat4x2), //⠃⠃ * ⠃⠃⠃⠃ =  ⠃⠃⠃⠃
    ),
    (mat2x3, * //⠇⠇
        (mat2   => mat2x3), //⠇⠇ * ⠃⠃   =  ⠇⠇
        (mat3x2 => mat3  ), //⠇⠇ * ⠃⠃⠃  =  ⠇⠇⠇
        (mat4x2 => mat4x3), //⠇⠇ * ⠃⠃⠃⠃ =  ⠇⠇⠇⠇
    ),
    (mat2x4, * //⡇⡇
        (mat2   => mat2x4), //⡇⡇ * ⠃⠃   =  ⡇⡇
        (mat3x2 => mat3x4), //⡇⡇ * ⠃⠃⠃  =  ⡇⡇⡇
        (mat4x2 => mat4  ), //⡇⡇ * ⠃⠃⠃⠃ =  ⡇⡇⡇⡇
    ),
    (mat3x2, * //⠃⠃⠃
        (mat2x3 => mat2  ), //⠃⠃⠃ * ⠇⠇   =  ⠃⠃
        (mat3   => mat3x2), //⠃⠃⠃ * ⠇⠇⠇  =  ⠃⠃⠃
        (mat4x3 => mat4x2), //⠃⠃⠃ * ⠇⠇⠇⠇ =  ⠃⠃⠃⠃
    ),
    (mat3, * //⠇⠇⠇
        (mat2x3 => mat2x3), //⠇⠇⠇ * ⠇⠇   =  ⠇⠇
        (mat3   => mat3  ), //⠇⠇⠇ * ⠇⠇⠇  =  ⠇⠇⠇
        (mat4x3 => mat4x3), //⠇⠇⠇ * ⠇⠇⠇⠇ =  ⠇⠇⠇⠇
    ),
    (mat3x4, * //⡇⡇⡇
        (mat2x3 => mat2x4), //⡇⡇⡇ * ⠇⠇   =  ⡇⡇
        (mat3   => mat3x4), //⡇⡇⡇ * ⠇⠇⠇  =  ⡇⡇⡇
        (mat4x3 => mat4  ), //⡇⡇⡇ * ⠇⠇⠇⠇ =  ⡇⡇⡇⡇
    ),
    (mat4x2, * //⠃⠃⠃⠃
        (mat2x4 => mat2  ), //⠃⠃⠃⠃ * ⡇⡇   =  ⠃⠃
        (mat3x4 => mat3x2), //⠃⠃⠃⠃ * ⡇⡇⡇  =  ⠃⠃⠃
        (mat4   => mat4x2), //⠃⠃⠃⠃ * ⡇⡇⡇⡇ =  ⠃⠃⠃⠃
    ),
    (mat4x3, * //⠇⠇⠇⠇
        (mat2x4 => mat2x3), //⠇⠇⠇⠇ * ⡇⡇   =  ⠇⠇
        (mat3x4 => mat3  ), //⠇⠇⠇⠇ * ⡇⡇⡇  =  ⠇⠇⠇
        (mat4   => mat4x3), //⠇⠇⠇⠇ * ⡇⡇⡇⡇ =  ⠇⠇⠇⠇
    ),
    (mat4, * //⡇⡇⡇⡇
        (mat2x4 => mat2x4), //⡇⡇⡇⡇ * ⡇⡇   =  ⡇⡇
        (mat3x4 => mat3x4), //⡇⡇⡇⡇ * ⡇⡇⡇  =  ⡇⡇⡇
        (mat4   => mat4  ), //⡇⡇⡇⡇ * ⡇⡇⡇⡇ =  ⡇⡇⡇⡇
    ),
);
