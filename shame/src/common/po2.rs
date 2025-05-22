use std::fmt::Display;

/// a power of two that fits in an `u32`'s value range
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub enum U32PowerOf2 {
    _1,
    _2,
    _4,
    _8,
    _16,
    _32,
    _64,
    _128,
    _256,
    _512,
    _1024,
    _2048,
    _4096,
    _8192,
    _16384,
    _32768,
    _65536,
    _131072,
    _262144,
    _524288,
    _1048576,
    _2097152,
    _4194304,
    _8388608,
    _16777216,
    _33554432,
    _67108864,
    _134217728,
    _268435456,
    _536870912,
    _1073741824,
    _2147483648,
}

impl U32PowerOf2 {
    /// Returns the corresponding u32.
    #[rustfmt::skip]
    pub const fn as_u32(self) -> u32 {
        match self {
            U32PowerOf2::_1          => 1_u32         ,
            U32PowerOf2::_2          => 2_u32         ,
            U32PowerOf2::_4          => 4_u32         ,
            U32PowerOf2::_8          => 8_u32         ,
            U32PowerOf2::_16         => 16_u32        ,
            U32PowerOf2::_32         => 32_u32        ,
            U32PowerOf2::_64         => 64_u32        ,
            U32PowerOf2::_128        => 128_u32       ,
            U32PowerOf2::_256        => 256_u32       ,
            U32PowerOf2::_512        => 512_u32       ,
            U32PowerOf2::_1024       => 1024_u32      ,
            U32PowerOf2::_2048       => 2048_u32      ,
            U32PowerOf2::_4096       => 4096_u32      ,
            U32PowerOf2::_8192       => 8192_u32      ,
            U32PowerOf2::_16384      => 16384_u32     ,
            U32PowerOf2::_32768      => 32768_u32     ,
            U32PowerOf2::_65536      => 65536_u32     ,
            U32PowerOf2::_131072     => 131072_u32    ,
            U32PowerOf2::_262144     => 262144_u32    ,
            U32PowerOf2::_524288     => 524288_u32    ,
            U32PowerOf2::_1048576    => 1048576_u32   ,
            U32PowerOf2::_2097152    => 2097152_u32   ,
            U32PowerOf2::_4194304    => 4194304_u32   ,
            U32PowerOf2::_8388608    => 8388608_u32   ,
            U32PowerOf2::_16777216   => 16777216_u32  ,
            U32PowerOf2::_33554432   => 33554432_u32  ,
            U32PowerOf2::_67108864   => 67108864_u32  ,
            U32PowerOf2::_134217728  => 134217728_u32 ,
            U32PowerOf2::_268435456  => 268435456_u32 ,
            U32PowerOf2::_536870912  => 536870912_u32 ,
            U32PowerOf2::_1073741824 => 1073741824_u32,
            U32PowerOf2::_2147483648 => 2147483648_u32,
        }
    }

    /// Returns the corresponding u64.
    pub const fn as_u64(self) -> u64 { self.as_u32() as u64 }
}

impl From<U32PowerOf2> for u32 {
    fn from(value: U32PowerOf2) -> Self { value.as_u32() }
}

impl U32PowerOf2 {
    /// Returns the maximum between `self` and `other`.
    pub const fn max(self, other: Self) -> Self { if self as u32 > other as u32 { self } else { other } }
}

#[derive(Debug)]
pub struct NotAU32PowerOf2(u32);

impl Display for NotAU32PowerOf2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} is not a power of two in the `u32` value range", self.0)
    }
}

impl std::error::Error for NotAU32PowerOf2 {}

impl U32PowerOf2 {
    /// Tries to convert a u32 to U32PowerOf2.
    pub const fn try_from_u32(value: u32) -> Option<Self> {
        Some(match value {
            1 => U32PowerOf2::_1,
            2 => U32PowerOf2::_2,
            4 => U32PowerOf2::_4,
            8 => U32PowerOf2::_8,
            16 => U32PowerOf2::_16,
            32 => U32PowerOf2::_32,
            64 => U32PowerOf2::_64,
            128 => U32PowerOf2::_128,
            256 => U32PowerOf2::_256,
            512 => U32PowerOf2::_512,
            1024 => U32PowerOf2::_1024,
            2048 => U32PowerOf2::_2048,
            4096 => U32PowerOf2::_4096,
            8192 => U32PowerOf2::_8192,
            16384 => U32PowerOf2::_16384,
            32768 => U32PowerOf2::_32768,
            65536 => U32PowerOf2::_65536,
            131072 => U32PowerOf2::_131072,
            262144 => U32PowerOf2::_262144,
            524288 => U32PowerOf2::_524288,
            1048576 => U32PowerOf2::_1048576,
            2097152 => U32PowerOf2::_2097152,
            4194304 => U32PowerOf2::_4194304,
            8388608 => U32PowerOf2::_8388608,
            16777216 => U32PowerOf2::_16777216,
            33554432 => U32PowerOf2::_33554432,
            67108864 => U32PowerOf2::_67108864,
            134217728 => U32PowerOf2::_134217728,
            268435456 => U32PowerOf2::_268435456,
            536870912 => U32PowerOf2::_536870912,
            1073741824 => U32PowerOf2::_1073741824,
            2147483648 => U32PowerOf2::_2147483648,
            n => return None,
        })
    }
}

impl TryFrom<u32> for U32PowerOf2 {
    type Error = NotAU32PowerOf2;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        U32PowerOf2::try_from_u32(value).ok_or(NotAU32PowerOf2(value))
    }
}

impl From<U32PowerOf2> for u64 {
    fn from(value: U32PowerOf2) -> Self { u32::from(value) as u64 }
}
