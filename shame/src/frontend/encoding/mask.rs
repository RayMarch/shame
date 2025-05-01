use std::{
    array,
    fmt::{Display, Write},
    ops::{Index, IndexMut},
};

use crate::{
    common::integer::saturating_post_dec_u8,
    frontend::{any::render_io::ChannelWrites, rust_types::len},
};
type Bits = u64;

/// A [`Vec<bool>`] with a maximum capacity of 64
/// where every `bool` is internally represented as a single bit.
///
/// The bits are accessible via [`BitVec64::as_u64()`] combined with [`BitVec64::occupied_mask()`]
/// or via [`std::ops::Index`] / [`std::iter::IntoIterator`] as `bool`s
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitVec64 {
    bits: Bits,
    /// len up to 64
    len: u8,
    b_one: bool,  // constant true for Index operator
    b_zero: bool, // constant false for Index operator
}

impl Display for BitVec64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //TODO(release) test
        write!(f, "0b")?;
        for bit in (0..self.len()).rev().map(|i| self[i]) {
            f.write_char(match bit {
                true => '1',
                false => '0',
            })?;
        }
        Ok(())
    }
}

impl Index<usize> for BitVec64 {
    type Output = bool;

    fn index(&self, index: usize) -> &bool {
        if index < self.len as usize {
            match (self.bits & (1 << index)) != 0x00000000 {
                true => &self.b_one,
                false => &self.b_zero,
            }
        } else {
            // we panic here because the standard library does the same thing
            // for its index operator impls
            panic!("bitvec index {index} out of bounds 0..{}", self.len)
        }
    }
}

impl BitVec64 {
    /// maximal amount of bits an instance of `Self` can store
    pub const fn capacity() -> u32 { Bits::BITS }

    /// construct a `BitVec64` of size `min(len, Self::capacity())` with all bits set to `1`
    pub fn full(len: u8) -> Self { Self::from_bools_trunc((0..len).map(|_| true)) }

    /// creates a `BitVec64` from an array of zeroes and ones
    ///
    /// example: `BitVec64::from_bools_trunc([0, 1, 1, 0])`
    ///
    /// if an element is equal to 0, it is interpreted as a zero digit, otherwise
    /// it is interpreted as a one digit.
    ///
    /// same as `BitVec64::from_bools_trunc(digits.into_iter().map(|i| i != 0))`
    pub fn from_digits_trunc(digits: impl IntoIterator<Item = u8>) -> Self {
        BitVec64::from_bools_trunc(digits.into_iter().map(|i| i != 0))
    }

    /// creates a `BitVec64` from an array of bools
    ///
    /// example: `BitVec64::from_bools_trunc([false, true, true, false])`
    pub fn from_bools_trunc(bits_iter: impl IntoIterator<Item = bool>) -> Self {
        const CAP: u8 = Bits::BITS as u8;
        let trunc_enumerate = (0..CAP).zip(bits_iter);

        let mut len: u8 = 0;
        let mut bits_left_aligned: Bits = 0;

        for (i, bit) in trunc_enumerate {
            len += 1;
            bits_left_aligned |= ((bit as Bits) << (CAP - 1 - i));
        }

        let right_shift_amount = CAP - len;
        let bits_right_aligned = match right_shift_amount {
            right_shift_amt @ ..CAP => bits_left_aligned >> right_shift_amount,
            CAP.. => 0, // overflow
        };

        BitVec64 {
            len,
            bits: bits_right_aligned,
            b_one: true,
            b_zero: false,
        }
    }

    pub(crate) fn pop(&mut self) -> Option<bool> {
        (!self.is_empty()).then(|| {
            let bit = self[self.len() - 1];
            self.bits &= self.occupied_mask() >> 1;
            self.len -= 1;
            bit
        })
    }

    /// returns `self.len() == 0`
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// returns the amount of bits in `self`, which never exceeds `self.capacity()`
    pub fn len(&self) -> usize { self.len as usize }

    /// returns a bitmask of size 64 with `self.len()` consecutive ones
    /// starting at the least significant bit
    pub fn occupied_mask(&self) -> u64 {
        assert!(self.len <= 64);
        (0..self.len).fold(0, |acc, i| acc | (1 << i))
    }

    /// returns the bitmask extended to 64 bits (padded with 0 bits)
    pub fn as_u64(self) -> u64 {
        assert_eq!(self.bits, self.bits & self.occupied_mask());
        self.bits
    }

    /// returns the bitmask extended or truncated to 32 bits (padded with 0 bits)
    pub fn as_u32(self) -> u32 {
        assert_eq!(self.bits, self.bits & self.occupied_mask());
        self.bits as u32
    }
}

impl<const N: usize> From<[u8; N]> for BitVec64 {
    fn from(digits: [u8; N]) -> Self { BitVec64::from_digits_trunc(digits) }
}

impl<const N: usize> From<[bool; N]> for BitVec64 {
    fn from(bools: [bool; N]) -> Self { BitVec64::from_bools_trunc(bools) }
}

/// iterates over a `BitVec64`'s bits, yielding `bool`s
pub struct BitVec64Iter(pub BitVec64);

impl Iterator for BitVec64Iter {
    type Item = bool;

    fn size_hint(&self) -> (usize, Option<usize>) { (self.0.len as usize, Some(self.0.len as usize)) }

    fn next(&mut self) -> Option<Self::Item> { self.0.pop() }
}

impl ExactSizeIterator for BitVec64Iter {}

#[test]
fn bitvec64_roundtrip_test() {
    for len in 0..32 {
        for bits in 0..2_u64.pow(len as u32).min(256) {
            let vec = BitVec64 {
                len,
                bits,
                b_one: true,
                b_zero: false,
            };
            let iter = vec.into_iter().collect::<Vec<bool>>();
            let vec2 = BitVec64::from_bools_trunc(iter.clone());
            if vec != vec2 {
                panic!(
                    "
                    {vec} != {vec2}
                    -------
                    vec: {vec:?}
                    iter: {iter:?}
                    vec2: {vec2:?}
                "
                )
            }
        }
    }
}

#[test]
fn bitvec64_display_test() {
    let bitvec = BitVec64::from([0, 0]);
    assert_eq!(bitvec.to_string(), "0b00");
    assert_eq!(bitvec.as_u64(), 0b00);
    assert_eq!(bitvec.as_u32(), 0b00);

    let bitvec = BitVec64::from([1, 0, 1, 1, 0]);
    assert_eq!(bitvec.to_string(), "0b10110");
    assert_eq!(bitvec.as_u64(), 0b10110);
    assert_eq!(bitvec.as_u32(), 0b10110);
}

impl IntoIterator for BitVec64 {
    type Item = bool;

    type IntoIter = BitVec64Iter;

    fn into_iter(self) -> Self::IntoIter { BitVec64Iter(self) }
}

impl BitVec64 {
    /// convert the first 4 bits of `self` to the color write mask channel flags.
    /// If `self` contains less than 4 bits, returns `None`
    pub(crate) fn first_elements_to_color_writes(self, num_color_channels: usize) -> Option<ChannelWrites> {
        (self.len() >= num_color_channels).then(|| {
            let n = num_color_channels;
            let mut rgba = [true; 4];
            for i in 0..self.len().min(rgba.len()) {
                rgba[i] = self[i];
            }

            // if all used channels are true/false, fill the rest with that value as well
            match rgba.split_at_mut(n) {
                (used, rest) if used.iter().all(|x| *x) => rest.fill(true),
                (used, rest) if used.iter().all(|x| !x) => rest.fill(false),
                _ => (),
            }

            let [r, g, b, a] = rgba;
            ChannelWrites { r, g, b, a }
        })
    }

    /// iterate over the bits of `self` as `bool`s
    pub fn iter(&self) -> impl Iterator<Item = bool> + use<'_> { self.into_iter() }
}

impl<const N: usize> From<[i32; N]> for BitVec64 {
    fn from(ints: [i32; N]) -> Self { BitVec64::from_bools_trunc(ints.map(|bit| bit != 0)) }
}

impl<const N: usize> From<[u32; N]> for BitVec64 {
    fn from(ints: [u32; N]) -> Self { BitVec64::from_bools_trunc(ints.map(|bit| bit != 0)) }
}

impl FromIterator<u32> for BitVec64 {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self { iter.into_iter().map(|x| x != 0).collect() }
}

impl FromIterator<bool> for BitVec64 {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        BitVec64::from_bools_trunc(iter.into_iter().collect::<Vec<_>>())
    }
}

impl From<u8> for BitVec64 {
    // this is a leftover impl from a different bitvec with different internal layout
    // feel free to turn this into the straight forward version
    fn from(val: u8) -> Self { (0..u8::BITS).map(|i| val & (1 << i) != 0).collect() }
}

impl From<u16> for BitVec64 {
    // this is a leftover impl from a different bitvec with different internal layout
    // feel free to turn this into the straight forward version
    fn from(val: u16) -> Self { (0..u16::BITS).map(|i| val & (1 << i) != 0).collect() }
}

impl From<u32> for BitVec64 {
    // this is a leftover impl from a different bitvec with different internal layout
    // feel free to turn this into the straight forward version
    fn from(val: u32) -> Self { (0..u32::BITS).map(|i| val & (1 << i) != 0).collect() }
}

impl From<u64> for BitVec64 {
    // this is a leftover impl from a different bitvec with different internal layout
    // feel free to turn this into the straight forward version
    fn from(val: u64) -> Self { (0..u64::BITS).map(|i| val & (1 << i) != 0).collect() }
}

impl From<usize> for BitVec64 {
    // this is a leftover impl from a different bitvec with different internal layout
    // feel free to turn this into the straight forward version
    fn from(val: usize) -> Self { (0..usize::BITS).map(|i| val & (1 << i) != 0).collect() }
}
