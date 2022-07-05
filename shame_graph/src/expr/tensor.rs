
use std::{fmt::Display};

use super::Ty;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Shape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bool, F32, F64, I32, U32, 
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Scalar, Vec(u8), Mat(u8, u8),
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use DType::*;
        f.write_str(match self {
            Bool => "bool", 
            I32 => "i32", 
            U32 => "u32", 
            F32 => "f32", 
            F64 => "f64",
        })
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Shape::Scalar => f.write_str("scalar"),
            Shape::Vec(x) => f.write_fmt(format_args!("vec{}", x)),
            Shape::Mat(cols, rows) => match cols == rows {
                true => f.write_fmt(format_args!("mat{}", cols)),
                false => f.write_fmt(format_args!("mat{}x{}", cols, rows)),
            }
        }
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.shape {
            Shape::Scalar => self.dtype.fmt(f),
            _ => f.write_fmt(format_args!("{}<{}>", self.shape, self.dtype)),
        }
    }
}

impl DType {

    pub fn as_signed(&self) -> DType {
        match *self {
            DType::Bool => DType::I32,
            DType::U32 => DType::I32,
            x => x,
        }
    }

    pub fn byte_size(&self) -> usize {
        #[allow(clippy::eq_op)]
        match self {
            DType::Bool => 8/8,
            DType::F32 => 32/8,
            DType::F64 => 64/8,
            DType::I32 => 32/8,
            DType::U32 => 32/8,
        }
    }
}

impl Tensor {

    pub const fn new(shape: Shape, dtype: DType) -> Self {
        Self {shape, dtype}
    }

    pub fn into_ty(&self) -> Ty {
        (*self).into()
    }

    pub fn byte_size(&self) -> usize {
        self.shape.comps_total() * self.dtype.byte_size()
    }

    ///returns the len used for out of bounds checks when using a subscript operator on this type
    ///e.g. mat3x3 at index 3 is out of range even though it has 9 elements.
    pub fn len_wrt_subscript_operator(&self) -> Option<usize> {
        match self.shape {
            Shape::Scalar => None,
            Shape::Vec(len) | 
            Shape::Mat(len, _) => Some(len as usize),
        }
    }

}

impl From<Ty> for Option<Tensor> {
    fn from(ty: Ty) -> Self {
        match ty.kind {
            super::TyKind::Tensor(x) => Some(x),
            _ => None,
        }
    }
}

impl Shape {

    pub fn from_dims_u8(dims: (u8, u8)) -> Shape {
        match dims {
            (1, 1) => Shape::Scalar,
            (1, n) | (n, 1) => Shape::Vec(n), //column and row vectors are treated the same (as in glsl)
            (m, n) => Shape::Mat(m, n)
        }
    }

    pub fn from_vec_len(len: usize) -> Option<Shape> {
        match len {
            1 => Some(Shape::Scalar),
            2|3|4 => Some(Shape::Vec(len as u8)),
            _ => None
        }
    }

    pub fn dims_u8(self) -> (u8, u8) {match self {
        Shape::Scalar    => (1, 1),
        Shape::Vec(r)    => (1, r),
        Shape::Mat(c, r) => (c, r),
    }}

    pub fn dims(self) -> (usize, usize) {
        let (a, b) = self.dims_u8();
        (a as usize, b as usize)
    }

    pub fn col_count(self) -> usize {
        self.dims().0
    }

    pub fn row_count(self) -> usize {
        self.dims().1
    }

    pub fn comps_total(&self) -> usize {
        let (x, y) = self.dims();
        x * y
    }

    pub fn len(&self) -> usize {
        self.comps_total()
    }

    pub fn is_vec   (&self) -> bool {matches!(self, Shape::Vec(_))}
    pub fn is_mat   (&self) -> bool {matches!(self, Shape::Mat(_, _))}
    pub fn is_scalar(&self) -> bool {matches!(self, Shape::Scalar)}
    pub fn is_scalar_or_vec(&self) -> bool {matches!(self, Shape::Scalar | Shape::Vec(_))}
}

impl Tensor {
    pub const fn float() -> Self {Self::new(Shape::Scalar, DType::F32)}
    pub const fn double()-> Self {Self::new(Shape::Scalar, DType::F64)}
    pub const fn   int() -> Self {Self::new(Shape::Scalar, DType::I32)}
    pub const fn  uint() -> Self {Self::new(Shape::Scalar, DType::U32)}
    pub const fn  bool() -> Self {Self::new(Shape::Scalar, DType::Bool)}
    pub const fn  vec2() -> Self {Self::new(Shape::Vec(2), DType::F32)}
    pub const fn  vec3() -> Self {Self::new(Shape::Vec(3), DType::F32)}
    pub const fn  vec4() -> Self {Self::new(Shape::Vec(4), DType::F32)}
    pub const fn bvec2() -> Self {Self::new(Shape::Vec(2), DType::Bool)}
    pub const fn bvec3() -> Self {Self::new(Shape::Vec(3), DType::Bool)}
    pub const fn bvec4() -> Self {Self::new(Shape::Vec(4), DType::Bool)}
    pub const fn dvec2() -> Self {Self::new(Shape::Vec(2), DType::F64)}
    pub const fn dvec3() -> Self {Self::new(Shape::Vec(3), DType::F64)}
    pub const fn dvec4() -> Self {Self::new(Shape::Vec(4), DType::F64)}
    pub const fn ivec2() -> Self {Self::new(Shape::Vec(2), DType::I32)}
    pub const fn ivec3() -> Self {Self::new(Shape::Vec(3), DType::I32)}
    pub const fn ivec4() -> Self {Self::new(Shape::Vec(4), DType::I32)}
    pub const fn uvec2() -> Self {Self::new(Shape::Vec(2), DType::U32)}
    pub const fn uvec3() -> Self {Self::new(Shape::Vec(3), DType::U32)}
    pub const fn uvec4() -> Self {Self::new(Shape::Vec(4), DType::U32)}

    pub const fn  mat2() -> Self {Self::new(Shape::Mat(2, 2), DType::F32)}
    pub const fn  mat3() -> Self {Self::new(Shape::Mat(3, 3), DType::F32)}
    pub const fn  mat4() -> Self {Self::new(Shape::Mat(4, 4), DType::F32)}
    pub const fn  mat2x3() -> Self {Self::new(Shape::Mat(2, 3), DType::F32)}
    pub const fn  mat2x4() -> Self {Self::new(Shape::Mat(2, 4), DType::F32)}
    pub const fn  mat3x2() -> Self {Self::new(Shape::Mat(3, 2), DType::F32)}
    pub const fn  mat3x4() -> Self {Self::new(Shape::Mat(3, 4), DType::F32)}
    pub const fn  mat4x2() -> Self {Self::new(Shape::Mat(4, 2), DType::F32)}
    pub const fn  mat4x3() -> Self {Self::new(Shape::Mat(4, 3), DType::F32)}

    pub const fn  dmat2() -> Self {Self::new(Shape::Mat(2, 2), DType::F64)}
    pub const fn  dmat3() -> Self {Self::new(Shape::Mat(3, 3), DType::F64)}
    pub const fn  dmat4() -> Self {Self::new(Shape::Mat(4, 4), DType::F64)}
    pub const fn  dmat2x3() -> Self {Self::new(Shape::Mat(2, 3), DType::F64)}
    pub const fn  dmat2x4() -> Self {Self::new(Shape::Mat(2, 4), DType::F64)}
    pub const fn  dmat3x2() -> Self {Self::new(Shape::Mat(3, 2), DType::F64)}
    pub const fn  dmat3x4() -> Self {Self::new(Shape::Mat(3, 4), DType::F64)}
    pub const fn  dmat4x2() -> Self {Self::new(Shape::Mat(4, 2), DType::F64)}
    pub const fn  dmat4x3() -> Self {Self::new(Shape::Mat(4, 3), DType::F64)}

}