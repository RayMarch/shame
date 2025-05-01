#![allow(non_camel_case_types)]
use super::{len::*, mat::mat, vec::vec, vec::ToVec};
use crate::common::floating_point::f16;
macro_rules! define_vector_aliases {
    ($($len: ident: ($f16xN: ident, $f32xN: ident, $f64xN: ident, $i32xN: ident, $u32xN: ident, $boolN: ident);)*) => {$(
        #[allow(missing_docs)] pub type $f16xN = vec<f16, $len>;
        #[allow(missing_docs)] pub type $f32xN = vec<f32, $len>;
        #[allow(missing_docs)] pub type $f64xN = vec<f64, $len>;
        #[allow(missing_docs)] pub type $i32xN = vec<i32, $len>;
        #[allow(missing_docs)] pub type $u32xN = vec<u32, $len>;
        #[allow(missing_docs)] pub type $boolN = vec<bool, $len>;
    )*};
}

macro_rules! define_matrix_aliases {
    ($($cols: ident, $rows: ident: ($f16xMxN: ident, $f32xMxN: ident, $f64xMxN: ident);)*) => {$(
        #[doc = "a matrix (see [`mat`]) where the \n\n- first number is the amount of bits in the floating point type (16), \n\n- second is the amount of columns, \n\n- third the amount of rows"] 
        pub type $f16xMxN = mat<f16, $cols, $rows>;
        #[doc = "a matrix (see [`mat`]) where the \n\n- first number is the amount of bits in the floating point type (32), \n\n- second is the amount of columns, \n\n- third the amount of rows"] 
        pub type $f32xMxN = mat<f32, $cols, $rows>;
        #[doc = "a matrix (see [`mat`]) where the \n\n- first number is the amount of bits in the floating point type (64), \n\n- second is the amount of columns, \n\n- third the amount of rows"] 
        pub type $f64xMxN = mat<f64, $cols, $rows>;
    )*};
}

/// [`vec`] and [`mat`] aliases in the style of rust's `std::simd`.
///
/// e.g. `f32x4`, `f32x2x2`, `i32x1`, `boolx3`
pub mod rust_simd {
    use super::*;

    // "x1" suffix because we don't want names to collide with rust's primitive datatypes
    #[doc = "gpu size = `2`, gpu align = `2`, see [`vec`]"]
    pub type f16x1 = vec<f16, x1>;
    #[doc = "gpu size = `4`, gpu align = `4`, see [`vec`]"]
    pub type f32x1 = vec<f32, x1>;
    #[doc = "gpu size = `8`, gpu align = `8`, see [`vec`]"]
    pub type f64x1 = vec<f64, x1>;
    #[doc = "gpu size = `4`, gpu align = `4`, see [`vec`]"]
    pub type u32x1 = vec<u32, x1>;
    #[doc = "gpu size = `4`, gpu align = `4`, see [`vec`]"]
    pub type i32x1 = vec<i32, x1>;
    #[doc = "gpu size and align are implementation defined, see [`vec`]"]
    pub type boolx1 = vec<bool, x1>;

    #[doc = "gpu size = `4`, gpu align = `4`, see [`vec`]"]
    pub type f16x2 = vec<f16, x2>;
    #[doc = "gpu size = `8`, gpu align = `8`, see [`vec`]"]
    pub type f32x2 = vec<f32, x2>;
    #[doc = "gpu size = `16`, gpu align = `16`, see [`vec`]"]
    pub type f64x2 = vec<f64, x2>;
    #[doc = "gpu size = `8`, gpu align = `8`, see [`vec`]"]
    pub type u32x2 = vec<u32, x2>;
    #[doc = "gpu size = `8`, gpu align = `8`, see [`vec`]"]
    pub type i32x2 = vec<i32, x2>;
    #[doc = "gpu size and align are implementation defined, see [`vec`]"]
    pub type boolx2 = vec<bool, x2>;

    #[doc = "gpu size = `6`, gpu align = `8`, see [`vec`]"]
    pub type f16x3 = vec<f16, x3>;
    #[doc = "gpu size = `12`, gpu align = `16`, see [`vec`]"]
    pub type f32x3 = vec<f32, x3>;
    #[doc = "gpu size = `24`, gpu align = `32`, see [`vec`]"]
    pub type f64x3 = vec<f64, x3>;
    #[doc = "gpu size = `12`, gpu align = `16`, see [`vec`]"]
    pub type u32x3 = vec<u32, x3>;
    #[doc = "gpu size = `12`, gpu align = `16`, see [`vec`]"]
    pub type i32x3 = vec<i32, x3>;
    #[doc = "gpu size and align are implementation defined, see [`vec`]"]
    pub type boolx3 = vec<bool, x3>;

    #[doc = "gpu size = `8`, gpu align = `8`, see [`vec`]"]
    pub type f16x4 = vec<f16, x4>;
    #[doc = "gpu size = `16`, gpu align = `16`, see [`vec`]"]
    pub type f32x4 = vec<f32, x4>;
    #[doc = "gpu size = `32`, gpu align = `32`, see [`vec`]"]
    pub type f64x4 = vec<f64, x4>;
    #[doc = "gpu size = `16`, gpu align = `16`, see [`vec`]"]
    pub type u32x4 = vec<u32, x4>;
    #[doc = "gpu size = `16`, gpu align = `16`, see [`vec`]"]
    pub type i32x4 = vec<i32, x4>;
    #[doc = "gpu size and align are implementation defined, see [`vec`]"]
    pub type boolx4 = vec<bool, x4>;

    // all of these are `shame::mat<FloatingPointT, Cols, Rows>`
    define_matrix_aliases! {
        x2, x2: (f16x2x2, f32x2x2, f64x2x2);
        x2, x3: (f16x2x3, f32x2x3, f64x2x3);
        x2, x4: (f16x2x4, f32x2x4, f64x2x4);

        x3, x2: (f16x3x2, f32x3x2, f64x3x2);
        x3, x3: (f16x3x3, f32x3x3, f64x3x3);
        x3, x4: (f16x3x4, f32x3x4, f64x3x4);

        x4, x2: (f16x4x2, f32x4x2, f64x4x2);
        x4, x3: (f16x4x3, f32x4x3, f64x4x3);
        x4, x4: (f16x4x4, f32x4x4, f64x4x4);
    }
}
