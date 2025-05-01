use std::ops::*;

/// 16 bit floating point types, represented as `f32` on the cpu during
/// pipeline encoding.
///
/// for now, this type serves as a pretend- 16 bit float which is actually a
/// f32 bit float.
/// It is written this way so it can be used to generate f16 values in the
/// shader code, and later be replaced by an actual f16 implementation once
/// rust supports f16 natively.
///
/// usage example:
/// `f16::from(1.0)`
///
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct f16(f32);

impl std::fmt::Display for f16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}

impl From<f32> for f16 {
    fn from(x: f32) -> Self { f16(x) }
}

impl From<f16> for f32 {
    fn from(x: f16) -> Self { x.0 }
}

impl From<f64> for f16 {
    fn from(x: f64) -> Self { f16(x as f32) }
}

impl From<f16> for f64 {
    fn from(x: f16) -> Self { x.0 as f64 }
}

macro_rules! impl_binary_operator {
    (
        $(
            ($trait_: ident, $func: ident): ($Rhs: ty) x ($Lhs: ty) $((as $cast: ty))? -> $out: ident;
        )*
    ) => {
        $(
            impl $trait_<$Rhs> for $Lhs {
                type Output = $out;

                fn $func(self, rhs: $Rhs) -> $out {
                    $out::from(f64::from(self).$func(f64::from(rhs)) $(as $cast)?)
                }
            }
        )*
    };
}

impl_binary_operator! {
    (Add, add): (f16) x (f16) -> f16;
    (Sub, sub): (f16) x (f16) -> f16;
    (Mul, mul): (f16) x (f16) -> f16;
    (Div, div): (f16) x (f16) -> f16;

    (Add, add): (f16) x (f32) (as f32) -> f32;
    (Add, add): (f32) x (f16) (as f32) -> f32;
    (Sub, sub): (f16) x (f32) (as f32) -> f32;
    (Sub, sub): (f32) x (f16) (as f32) -> f32;
    (Mul, mul): (f16) x (f32) (as f32) -> f32;
    (Mul, mul): (f32) x (f16) (as f32) -> f16;
    (Div, div): (f16) x (f32) (as f32) -> f32;
    (Div, div): (f32) x (f16) (as f32) -> f16;

    (Add, add): (f16) x (f64) -> f64;
    (Add, add): (f64) x (f16) -> f64;
    (Sub, sub): (f16) x (f64) -> f64;
    (Sub, sub): (f64) x (f16) -> f64;
    (Mul, mul): (f16) x (f64) -> f64;
    (Mul, mul): (f64) x (f16) -> f16;
    (Div, div): (f16) x (f64) -> f64;
    (Div, div): (f64) x (f16) -> f16;
}

pub fn f32_eq_where_nans_are_equal(a: f32, b: f32) -> bool {
    match a.is_nan() && b.is_nan() {
        true => true,
        false => a == b,
    }
}

pub fn f64_eq_where_nans_are_equal(a: f64, b: f64) -> bool {
    match a.is_nan() && b.is_nan() {
        true => true,
        false => a == b,
    }
}
