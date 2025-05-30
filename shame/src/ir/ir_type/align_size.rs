use std::num::NonZeroU32;
use std::rc::Rc;

use crate::backend::language::Language;
use crate::frontend::any::shared_io::BufferBindingType;
use thiserror::Error;

use super::{CanonName, Len2, ScalarType, ScalarTypeFp, SizedType, StoreType};
use super::{Len};

pub const fn round_up(multiple_of: u64, n: u64) -> u64 {
    match multiple_of {
        0 => match n {
            0 => 0,
            n => panic!("cannot round up n to a multiple of 0"),
        },
        k @ 1.. => n.div_ceil(k) * k,
    }
}

/// the largest power of two that divides `x` without remainder
///
/// returns 2^63 for `x == 0`
pub fn max_u64_po2_dividing(x: u64) -> u64 {
    (0..u64::BITS)
        .map(|i| 1u64 << i)
        .find(|po2| (x & po2) != 0)
        .unwrap_or(1u64 << (u64::BITS - 1))
}

impl ScalarType {
    const fn align(self) -> u64 { self.byte_size() }

    const fn byte_size(self) -> u64 {
        use ScalarType as S;
        match self {
            S::F16 => 2,
            S::F32 | S::U32 | S::I32 => 4,
            S::F64 => 8, // not found in the WGSL spec, but exists in GLSL
            // not observable in WGSL,
            //
            // WGSL Naga 0.14.0 output suggests size_of(bool) == 1
            //
            // OpenGL spec 4.6, section 7.6.2.1:
            //
            // "Members of type bool, int, uint, float, and double are respectively
            // extracted from a buffer object by reading a
            // single uint, int, uint, float, or double value at the specified offset."
            //
            // unclear if this is a specification of bools having 4 byte size.
            // In other places of the spec there are conversions described
            // from float, u32, i32, so maybe this has no meaning.
            // for now, Bool sizes are also not observable within `shame`
            // so we assume 4 bytes
            S::Bool => 4,
        }
    }
}

impl SizedType {
    /// the alignment in bytes of this type
    pub fn align(&self) -> u64 {
        use SizedType as ST;
        match self {
            ST::Vector(len, stype) => (u64::from(*len) * stype.align()).next_power_of_two(),
            ST::Matrix(c, r, stype) => ST::Vector((*r).into(), (*stype).into()).align(),
            ST::Atomic(a) => ScalarType::from(*a).align(),
            ST::Array(e, _) => align_of_array(e),
            ST::Structure(s) => s.align(),
        }
    }

    /// the size in bytes of this type
    pub fn byte_size(&self) -> u64 {
        use SizedType as ST;
        match self {
            ST::Vector(len, stype) => u64::from(*len) * stype.byte_size(),
            ST::Matrix(c, r, stype) => {
                byte_size_of_array(&SizedType::Vector((*r).into(), (*stype).into()), (*c).into())
            }
            ST::Atomic(a) => ScalarType::from(*a).byte_size(),
            ST::Array(e, n) => byte_size_of_array(e, (*n)),
            ST::Structure(s) => s.byte_size(),
        }
    }
}

/// size of an `array<e, n>`
///
/// `e`is the element type, not the array in quesiton
pub(crate) fn byte_size_of_array(e: &SizedType, n: NonZeroU32) -> u64 {
    let n = n.get() as u64;
    byte_size_of_array_from_stride_len(stride_of_array(e), n)
}

/// size of an `array<e>` or `array<e, _>`
///
/// `element`is the element type, not the array in quesiton
pub fn stride_of_array(element: &SizedType) -> u64 {
    stride_of_array_from_element_align_size(element.align(), element.byte_size())
}

pub fn stride_of_array_from_element_align_size(align: u64, byte_size: u64) -> u64 { round_up(align, byte_size) }

pub fn align_of_array_from_element_alignment(element_alignment: u64) -> u64 { element_alignment }

pub fn byte_size_of_array_from_stride_len(stride: u64, len: u64) -> u64 { stride * len }


/// alignment of an `array<e>` or `array<e, _>`
///
/// `e`is the element type, not the array in quesiton
pub fn align_of_array(e: &SizedType) -> u64 { align_of_array_from_element_alignment(e.align()) }


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn scalar_and_vector_layouts() {
        use Len::*;
        use ScalarType::*;
        use SizedType::*;

        let align_and_size_of = |t: SizedType| -> (u64, u64) { (t.align(), t.byte_size()) };

        // 16 bit
        assert_eq!(align_and_size_of(Vector(X1, F16)), (2, 2));
        assert_eq!(align_and_size_of(Vector(X2, F16)), (4, 4));
        assert_eq!(align_and_size_of(Vector(X3, F16)), (8, 6));
        assert_eq!(align_and_size_of(Vector(X4, F16)), (8, 8));

        // 32 bit
        for t32 in [F32, U32, I32] {
            assert_eq!(align_and_size_of(Vector(X1, F32)), (4, 4));
            assert_eq!(align_and_size_of(Vector(X2, F32)), (8, 8));
            assert_eq!(align_and_size_of(Vector(X3, F32)), (16, 12));
            assert_eq!(align_and_size_of(Vector(X4, F32)), (16, 16));
        }

        // 64 bit
        assert_eq!(align_and_size_of(Vector(X1, F64)), (8, 8));
        assert_eq!(align_and_size_of(Vector(X2, F64)), (16, 16));
        assert_eq!(align_and_size_of(Vector(X3, F64)), (32, 24));
        assert_eq!(align_and_size_of(Vector(X4, F64)), (32, 32));

        // bool
        assert_eq!(align_and_size_of(Vector(X1, Bool)), (4, 4));
        assert_eq!(align_and_size_of(Vector(X2, Bool)), (8, 8));
        assert_eq!(align_and_size_of(Vector(X3, Bool)), (16, 12));
        assert_eq!(align_and_size_of(Vector(X4, Bool)), (16, 16));
    }

    #[test]
    fn test_wgsl_spec_fns() {
        assert_eq!(round_up(5, 6), 10);
        assert_eq!(round_up(5, 5), 5);
        assert_eq!(round_up(5, 10000001), 10000005);
        assert_eq!(round_up(5, 0), 0);
        assert_eq!(round_up(1, 0), 0);
        assert_eq!(round_up(1, 1), 1);
        assert_eq!(round_up(8, 24), 24);
        assert_eq!(round_up(16, 24), 32);

        // roundUp as defined by the wgsl spec
        let wgsl_round_up = |k: u64, n: u64| ((n as f64 / k as f64).ceil() * k as f64) as u64;

        for k in 1..100 {
            for n in 0..100 {
                assert_eq!(wgsl_round_up(k, n), round_up(k, n))
            }
        }
    }
}
