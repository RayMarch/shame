use std::{fmt::Display, num::NonZeroU32};

use crate::{
    common::floating_point::{f16, f32_eq_where_nans_are_equal, f64_eq_where_nans_are_equal},
    ir::Comp4,
};

use super::{SizedType, Type};

/// (no documentation yet)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Len {
    /// (no documentation yet)
    X1,
    /// (no documentation yet)
    X2,
    /// (no documentation yet)
    X3,
    /// (no documentation yet)
    X4,
}

/// Length starting at 2.
/// useful for example in matrix column/row sizes
#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Len2 {
    X2,
    X3,
    X4,
}

impl Display for Len2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Len::from(*self).fmt(f) }
}

/// even length values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LenEven {
    /// (no documentation yet)
    X2,
    /// (no documentation yet)
    X4,
}

impl From<LenEven> for u64 {
    fn from(value: LenEven) -> Self {
        match value {
            LenEven::X2 => 2,
            LenEven::X4 => 4,
        }
    }
}

impl From<LenEven> for Len {
    fn from(value: LenEven) -> Self {
        match value {
            LenEven::X2 => Len::X2,
            LenEven::X4 => Len::X4,
        }
    }
}

impl PartialEq<Len2> for Len {
    fn eq(&self, other: &Len2) -> bool {
        match (self, other) {
            (Len::X2, Len2::X2) => true,
            (Len::X3, Len2::X3) => true,
            (Len::X4, Len2::X4) => true,
            _ => false,
        }
    }
}

impl From<Len2> for u64 {
    fn from(value: Len2) -> Self {
        match value {
            Len2::X2 => 2,
            Len2::X3 => 3,
            Len2::X4 => 4,
        }
    }
}

impl From<Len2> for NonZeroU32 {
    fn from(value: Len2) -> Self {
        match value {
            Len2::X2 => NonZeroU32::new(2).unwrap(),
            Len2::X3 => NonZeroU32::new(3).unwrap(),
            Len2::X4 => NonZeroU32::new(4).unwrap(),
        }
    }
}

impl PartialEq<Len> for Len2 {
    fn eq(&self, other: &Len) -> bool { other == self }
}

impl Len {
    /// an iterator that iterates over the component names of a vector of length `self`
    pub fn iter_components(self) -> impl ExactSizeIterator<Item = Comp4> {
        use Comp4::*;
        [X, Y, Z, W].into_iter().take(self.into())
    }
}

/// (no documentation yet)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// (no documentation yet)
    F16,
    /// (no documentation yet)
    F32,
    /// (no documentation yet)
    F64,
    /// (no documentation yet)
    U32,
    /// (no documentation yet)
    I32,
    /// (no documentation yet)
    Bool,
}

impl ScalarType {
    pub(crate) fn is_numeric(&self) -> bool {
        use ScalarType::*;
        matches!(self, F16 | F32 | F64 | U32 | I32)
    }

    pub(crate) fn is_floating_point(&self) -> bool {
        use ScalarType::*;
        matches!(self, F16 | F32 | F64)
    }

    pub(crate) fn is_integer(&self) -> bool {
        use ScalarType::*;
        matches!(self, U32 | I32)
    }

    pub(crate) fn is_signed(&self) -> bool {
        use ScalarType::*;
        matches!(self, F16 | F32 | F64 | I32)
    }

    pub(crate) fn is_32_bit(&self) -> bool {
        use ScalarType::*;
        matches!(self, F32 | U32 | I32)
    }

    pub(crate) fn zero(&self) -> ScalarConstant {
        match self {
            ScalarType::F16 => ScalarConstant::F16(f16::from(0.0)),
            ScalarType::F32 => ScalarConstant::F32(0.0),
            ScalarType::F64 => ScalarConstant::F64(0.0),
            ScalarType::U32 => ScalarConstant::U32(0),
            ScalarType::I32 => ScalarConstant::I32(0),
            ScalarType::Bool => ScalarConstant::Bool(false),
        }
    }

    pub(crate) fn constant_from_f64(&self, val: f64) -> ScalarConstant {
        match self {
            ScalarType::F16 => ScalarConstant::F16(f16::from(val as f32)),
            ScalarType::F32 => ScalarConstant::F32(val as f32),
            ScalarType::F64 => ScalarConstant::F64(val),
            ScalarType::U32 => ScalarConstant::U32(val as u32),
            ScalarType::I32 => ScalarConstant::I32(val as i32),
            ScalarType::Bool => ScalarConstant::Bool(val != 0.0),
        }
    }
}

impl Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ScalarType::F16 => "f16",
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
            ScalarType::U32 => "u32",
            ScalarType::I32 => "i32",
            ScalarType::Bool => "bool",
        })
    }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarTypeFp {
    F16,
    F32,
    F64,
}

impl Display for ScalarTypeFp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ScalarTypeFp::F16 => "f16",
            ScalarTypeFp::F32 => "f32",
            ScalarTypeFp::F64 => "f64",
        })
    }
}

impl ScalarTypeFp {
    pub fn is_floating_point(self) -> bool { true }
}

impl PartialEq<ScalarTypeFp> for ScalarType {
    fn eq(&self, other: &ScalarTypeFp) -> bool { *self == ScalarType::from(*other) }
}

impl PartialEq<ScalarType> for ScalarTypeFp {
    fn eq(&self, other: &ScalarType) -> bool { *other == ScalarType::from(*self) }
}

impl PartialEq<ScalarTypeInteger> for ScalarType {
    fn eq(&self, other: &ScalarTypeInteger) -> bool { *self == ScalarType::from(*other) }
}

impl PartialEq<ScalarType> for ScalarTypeInteger {
    fn eq(&self, other: &ScalarType) -> bool { *other == ScalarType::from(*self) }
}

impl From<ScalarTypeFp> for ScalarType {
    fn from(value: ScalarTypeFp) -> Self {
        match value {
            ScalarTypeFp::F16 => ScalarType::F16,
            ScalarTypeFp::F32 => ScalarType::F32,
            ScalarTypeFp::F64 => ScalarType::F64,
        }
    }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarTypeInteger {
    U32,
    I32,
}

impl From<ScalarTypeInteger> for ScalarType {
    fn from(s: ScalarTypeInteger) -> Self {
        match s {
            ScalarTypeInteger::U32 => ScalarType::U32,
            ScalarTypeInteger::I32 => ScalarType::I32,
        }
    }
}

impl From<Len2> for Len {
    fn from(x: Len2) -> Self {
        match x {
            Len2::X2 => Len::X2,
            Len2::X3 => Len::X3,
            Len2::X4 => Len::X4,
        }
    }
}

impl TryFrom<Len> for Len2 {
    type Error = ();

    fn try_from(value: Len) -> Result<Self, ()> {
        match value {
            Len::X1 => Err(()),
            Len::X2 => Ok(Len2::X2),
            Len::X3 => Ok(Len2::X3),
            Len::X4 => Ok(Len2::X4),
        }
    }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy)]
pub enum ScalarConstant {
    F16(f16),
    F32(f32),
    F64(f64),
    U32(u32),
    I32(i32),
    Bool(bool),
}

impl ScalarConstant {
    pub fn ty(&self) -> ScalarType {
        match self {
            ScalarConstant::F16(_) => ScalarType::F16,
            ScalarConstant::F32(_) => ScalarType::F32,
            ScalarConstant::F64(_) => ScalarType::F64,
            ScalarConstant::U32(_) => ScalarType::U32,
            ScalarConstant::I32(_) => ScalarType::I32,
            ScalarConstant::Bool(_) => ScalarType::Bool,
        }
    }
}

impl Eq for ScalarConstant {}
impl PartialEq for ScalarConstant {
    fn eq(&self, other: &Self) -> bool {
        use ScalarConstant::*;
        match (*self, *other) {
            (F16(a), F16(b)) => f32_eq_where_nans_are_equal(a.into(), b.into()),
            (F32(a), F32(b)) => f32_eq_where_nans_are_equal(a, b),
            (F64(a), F64(b)) => f64_eq_where_nans_are_equal(a, b),
            (U32(a), U32(b)) => a == b,
            (I32(a), I32(b)) => a == b,
            (Bool(a), Bool(b)) => a == b,
            _ => false,
        }
    }
}

impl From<Len> for usize {
    fn from(l: Len) -> Self {
        match l {
            Len::X1 => 1,
            Len::X2 => 2,
            Len::X3 => 3,
            Len::X4 => 4,
        }
    }
}

impl From<Len> for u32 {
    fn from(l: Len) -> Self {
        match l {
            Len::X1 => 1,
            Len::X2 => 2,
            Len::X3 => 3,
            Len::X4 => 4,
        }
    }
}

impl From<Len> for u64 {
    fn from(l: Len) -> Self {
        match l {
            Len::X1 => 1,
            Len::X2 => 2,
            Len::X3 => 3,
            Len::X4 => 4,
        }
    }
}

impl From<Len2> for u32 {
    fn from(l: Len2) -> Self {
        match l {
            Len2::X2 => 2,
            Len2::X3 => 3,
            Len2::X4 => 4,
        }
    }
}

impl PartialEq<Len> for u32 {
    fn eq(&self, other: &Len) -> bool { self.eq(&u32::from(*other)) }
}

impl PartialOrd<Len> for u32 {
    fn partial_cmp(&self, other: &Len) -> Option<std::cmp::Ordering> { self.partial_cmp(&u32::from(*other)) }
}

impl PartialEq<Len> for usize {
    fn eq(&self, other: &Len) -> bool { self.eq(&(u32::from(*other) as usize)) }
}

impl PartialOrd<Len> for usize {
    fn partial_cmp(&self, other: &Len) -> Option<std::cmp::Ordering> { self.partial_cmp(&(u32::from(*other) as usize)) }
}

impl PartialEq<Len2> for u32 {
    fn eq(&self, other: &Len2) -> bool { self.eq(&u32::from(*other)) }
}

impl PartialOrd<Len2> for u32 {
    fn partial_cmp(&self, other: &Len2) -> Option<std::cmp::Ordering> { self.partial_cmp(&u32::from(*other)) }
}

impl Display for Len {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Len::X1 => "x1",
            Len::X2 => "x2",
            Len::X3 => "x3",
            Len::X4 => "x4",
        })
    }
}

/// (no documentation yet)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedBitsPerComponent {
    /// (no documentation yet)
    _8,
    /// (no documentation yet)
    _16,
}

impl From<PackedBitsPerComponent> for u8 {
    fn from(value: PackedBitsPerComponent) -> Self {
        match value {
            PackedBitsPerComponent::_8 => 8,
            PackedBitsPerComponent::_16 => 16,
        }
    }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedFloat {
    /// components are `u8` or `u16` depending on `PackedBitsPerComponent`.
    /// - `u8`: `[0, 255]`
    /// - `u16`: `[0, 65535]`
    ///
    /// ranges are converted to float `[0.0, 1.0]` `f32` in shaders.
    Unorm,
    /// components are `i8` or `i16`  depending on `PackedBitsPerComponent`.
    /// - `i8`: `[-127, 127]`
    /// - `i16` `[-32767, 32767]`
    ///
    /// ranges are converted to float `[-1.0, 1.0]` `f32` in shaders.
    /// - an `i8` value of `-128` is converted to `-1.0`
    /// - an `i16` value of `-32768` is converted to `-1.0`
    Snorm,
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedScalarType {
    Float(PackedFloat),
    Int,
    Uint,
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedVector {
    pub len: LenEven,
    pub bits_per_component: PackedBitsPerComponent,
    pub scalar_type: PackedScalarType,
}

/// exhaustive list of all byte sizes a `packed_vec` can have
pub enum PackedVectorByteSize {
    _2,
    _4,
    _8,
}

impl From<PackedVectorByteSize> for u8 {
    fn from(value: PackedVectorByteSize) -> Self {
        match value {
            PackedVectorByteSize::_2 => 2,
            PackedVectorByteSize::_4 => 4,
            PackedVectorByteSize::_8 => 8,
        }
    }
}

impl Display for PackedVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stype = match self.scalar_type {
            PackedScalarType::Float(x) => match x {
                PackedFloat::Unorm => "unorm",
                PackedFloat::Snorm => "snorm",
            },
            PackedScalarType::Int => "i",
            PackedScalarType::Uint => "u",
        };
        let bits_per_c = u8::from(self.bits_per_component);
        let n = u64::from(self.len);
        write!(f, "{stype}{bits_per_c}x{n}")
    }
}

impl PackedVector {
    #[rustfmt::skip]
    pub fn byte_size(&self) -> PackedVectorByteSize {
        match (self.len, self.bits_per_component) {
            (LenEven::X2, PackedBitsPerComponent:: _8) => PackedVectorByteSize::_2, // 2 * 1
            (LenEven::X2, PackedBitsPerComponent::_16) => PackedVectorByteSize::_4, // 2 * 2
            (LenEven::X4, PackedBitsPerComponent:: _8) => PackedVectorByteSize::_4, // 4 * 1
            (LenEven::X4, PackedBitsPerComponent::_16) => PackedVectorByteSize::_8, // 4 * 2
        }
    }

    pub fn align(&self) -> u64 {
        match self.byte_size() {
            PackedVectorByteSize::_2 => SizedType::Vector(Len::X1, ScalarType::F16).align(),
            PackedVectorByteSize::_4 => SizedType::Vector(Len::X1, ScalarType::U32).align(),
            PackedVectorByteSize::_8 => SizedType::Vector(Len::X2, ScalarType::U32).align(),
        }
    }
}

impl PackedScalarType {
    pub fn decompressed_ty(&self) -> ScalarType {
        match self {
            PackedScalarType::Float(_) => ScalarType::F32,
            PackedScalarType::Int => ScalarType::I32,
            PackedScalarType::Uint => ScalarType::U32,
        }
    }
}

impl PackedVector {
    pub fn decompressed_ty(&self) -> SizedType {
        SizedType::Vector(self.len.into(), self.scalar_type.decompressed_ty())
    }
}
