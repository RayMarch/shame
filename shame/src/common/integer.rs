use std::num::NonZeroU64;

use thiserror::Error;

pub trait IntegerExt {
    type Integer;
    fn divides(self, dividend: Self::Integer) -> bool;
}

impl IntegerExt for u64 {
    type Integer = u64;

    /// whether `self` divides `dividend` without a remainder
    fn divides(self, dividend: Self::Integer) -> bool { dividend.rem_euclid(self) == 0 }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
/// 4 bit signed integer with a value range of `-8..=7`
///
/// ## usage examples
/// ```
/// let seven = shame::i4::try_from(7).expect("fits");
/// let seven = shame::i4::clamped(8);
/// let [u_off, v_off] = shame::i4::clamp_all([2, -2]);
/// ```
pub struct i4(i8);

impl i4 {
    /// takes an `i8` value and constructs an `i4` by clamping it to the valid
    /// range of `-8..=7`
    pub fn clamped(t: i8) -> Self { i4(t.clamp(-8, 7)) }

    /// takes an `[i8; N]` value and constructs an `[i4; N]` by clamping every element
    /// to the valid range of `-8..=7`
    pub fn clamp_all<const N: usize>(array: [i8; N]) -> [Self; N] { array.map(i4::clamped) }
}

impl std::fmt::Display for i4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

#[allow(missing_docs)]
#[derive(Debug, Error, Clone, Copy)]
pub enum I4ConversionError {
    #[error("failed to convert `{0}` to 4 bit signed integer (out of range `-8..=7`)")]
    OutOfRange(i64),
}

impl TryFrom<i8> for i4 {
    type Error = I4ConversionError;

    fn try_from(i: i8) -> Result<Self, Self::Error> {
        match i {
            -8..=7 => Ok(i4(i)),
            _ => Err(I4ConversionError::OutOfRange(i as i64)),
        }
    }
}

impl From<i4> for i8 {
    fn from(value: i4) -> Self { value.0 as Self }
}

pub fn post_inc_u32(value: &mut u32) -> u32 {
    let temp = *value;
    *value += 1;
    temp
}

pub fn saturating_post_dec_u8(value: &mut u8) -> u8 { std::mem::replace(value, value.saturating_sub(1)) }


pub fn post_inc_usize(value: &mut usize) -> usize {
    let temp = *value;
    *value += 1;
    temp
}
