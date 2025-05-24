use super::super::{LayoutCalculator, Repr};
use super::*;

//              Size and align of layoutable types              //
// https://www.w3.org/TR/WGSL/#address-space-layout-constraints //

pub(crate) const PACKED_ALIGN: U32PowerOf2 = U32PowerOf2::_1;

impl LayoutableType {
    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the align.
    pub fn byte_size(&self, repr: Repr) -> Option<u64> {
        match self {
            LayoutableType::Sized(s) => Some(s.byte_size(repr)),
            LayoutableType::UnsizedStruct(_) | LayoutableType::RuntimeSizedArray(_) => None,
        }
    }

    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the align.
    pub fn align(&self, repr: Repr) -> U32PowerOf2 {
        match self {
            LayoutableType::Sized(s) => s.align(repr),
            LayoutableType::UnsizedStruct(s) => s.align(repr),
            LayoutableType::RuntimeSizedArray(a) => a.align(repr),
        }
    }

    /// This is expensive for structs as it calculates the byte size and align from scratch.
    pub fn byte_size_and_align(&self, repr: Repr) -> (Option<u64>, U32PowerOf2) {
        match self {
            LayoutableType::Sized(s) => {
                let (size, align) = s.byte_size_and_align(repr);
                (Some(size), align)
            }
            LayoutableType::UnsizedStruct(s) => (None, s.align(repr)),
            LayoutableType::RuntimeSizedArray(a) => (None, a.align(repr)),
        }
    }
}

impl SizedType {
    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the align.
    pub fn byte_size(&self, repr: Repr) -> u64 {
        match self {
            SizedType::Array(a) => a.byte_size(repr),
            SizedType::Vector(v) => v.byte_size(),
            SizedType::Matrix(m) => m.byte_size(repr, MatrixMajor::Row),
            SizedType::Atomic(a) => a.byte_size(),
            SizedType::PackedVec(v) => u8::from(v.byte_size()) as u64,
            SizedType::Struct(s) => s.byte_size_and_align(repr).0,
        }
    }

    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the align.
    pub fn align(&self, repr: Repr) -> U32PowerOf2 {
        match self {
            SizedType::Array(a) => a.align(repr),
            SizedType::Vector(v) => v.align(repr),
            SizedType::Matrix(m) => m.align(repr, MatrixMajor::Row),
            SizedType::Atomic(a) => a.align(repr),
            SizedType::PackedVec(v) => v.align(repr),
            SizedType::Struct(s) => s.byte_size_and_align(repr).1,
        }
    }

    /// This is expensive for structs as it calculates the byte size and align from scratch.
    pub fn byte_size_and_align(&self, repr: Repr) -> (u64, U32PowerOf2) {
        match self {
            SizedType::Struct(s) => s.byte_size_and_align(repr),
            non_struct => (non_struct.byte_size(repr), non_struct.align(repr)),
        }
    }
}


impl SizedStruct {
    /// Returns [`FieldOffsets`], which serves as an iterator over the offsets of the
    /// fields of this struct, as well as a `byte_size` and `align` calculator.
    /// See the documentation of `FieldOffsets` on how to obtain `byte_size` and `align`
    /// of this struct from it.
    pub fn field_offsets(&self, repr: Repr) -> FieldOffsets {
        FieldOffsets {
            fields: &self.fields,
            field_index: 0,
            calc: LayoutCalculator::new(repr),
            repr,
        }
    }

    /// Returns (byte_size, align)
    ///
    /// ### Careful!
    /// This is an expensive operation as it calculates byte size and align from scratch.
    /// If you also need field offsets, use [`SizedStruct::field_offsets`] instead and
    /// read the documentation of [`FieldOffsets`] on how to obtain the byte size and align from it.
    pub fn byte_size_and_align(&self, repr: Repr) -> (u64, U32PowerOf2) {
        let mut field_offsets = self.field_offsets(repr);
        (&mut field_offsets).count(); // &mut so it doesn't consume
        (field_offsets.byte_size(), field_offsets.align())
    }
}

/// An iterator over the field offsets of a `SizedStruct` or the sized fields of an `UnsizedStruct`.
///
/// `FieldOffsets::byte_size` and `FieldOffsets::byte_align` can be used to query the struct's
/// `byte_size` and `byte_align`, but only takes into account the fields that have been iterated over.
///
/// Use [`UnsizedStruct::last_field_offset_and_struct_align`] to obtain the last field's offset
/// and the struct's align for `UnsizedStruct`s.
pub struct FieldOffsets<'a> {
    fields: &'a [SizedField],
    field_index: usize,
    calc: LayoutCalculator,
    repr: Repr,
}

impl Iterator for FieldOffsets<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.field_index += 1;
        self.fields.get(self.field_index - 1).map(|field| {
            let (size, align) = field.ty.byte_size_and_align(self.repr);
            let is_struct = matches!(field.ty, SizedType::Struct(_));

            self.calc
                .extend(size, align, field.custom_min_size, field.custom_min_align, is_struct)
        })
    }
}

impl FieldOffsets<'_> {
    /// Returns the byte size of the struct, but ONLY with the fields that have been iterated over so far.
    pub const fn byte_size(&self) -> u64 { self.calc.byte_size() }
    /// Returns the byte align of the struct, but ONLY with the fields that have been iterated over so far.
    pub const fn align(&self) -> U32PowerOf2 { struct_align(self.calc.align(), self.repr) }
}


impl UnsizedStruct {
    /// Returns [`FieldOffsets`], which serves as an iterator over the offsets of the
    /// sized fields of this struct. `FieldOffsets` may be also passed to
    /// [`UnsizedStruct::last_field_offset_and_struct_align`] to obtain the last field's offset
    /// and the struct's align.
    pub fn sized_field_offsets(&self, repr: Repr) -> FieldOffsets {
        FieldOffsets {
            fields: &self.sized_fields,
            field_index: 0,
            calc: LayoutCalculator::new(repr),
            repr,
        }
    }

    /// Returns (last field offset, byte align of unsized struct)
    ///
    /// `sized_field_offsets` must be from the same `UnsizedStruct`, otherwise the returned values
    /// are not accurate.
    pub fn last_field_offset_and_struct_align(&self, sized_field_offsets: FieldOffsets) -> (u64, U32PowerOf2) {
        let mut offsets = sized_field_offsets;
        // Iterating over any remaining field offsets to update the layout calculator.
        (&mut offsets).count();

        let array_align = self.last_unsized.array.align(offsets.repr);
        let custom_min_align = self.last_unsized.custom_min_align;
        let (offset, align) = offsets.calc.extend_unsized(array_align, custom_min_align);
        (offset, struct_align(align, offsets.repr))
    }

    /// This is expensive as it calculates the byte align from scratch.
    pub fn align(&self, repr: Repr) -> U32PowerOf2 {
        let offsets = self.sized_field_offsets(repr);
        self.last_field_offset_and_struct_align(offsets).1
    }
}

const fn struct_align(align: U32PowerOf2, repr: Repr) -> U32PowerOf2 {
    match repr {
        // Packedness is ensured by the `LayoutCalculator`.
        Repr::Storage => align,
        Repr::Uniform => round_up_align(U32PowerOf2::_16, align),
        Repr::Packed => PACKED_ALIGN,
    }
}

#[allow(missing_docs)]
impl Vector {
    pub const fn new(scalar: ScalarType, len: Len) -> Self { Self { scalar, len } }

    pub const fn byte_size(&self) -> u64 { self.len.as_u64() * self.scalar.byte_size() }

    pub const fn align(&self, repr: Repr) -> U32PowerOf2 {
        if repr.is_packed() {
            return PACKED_ALIGN;
        }

        let len = match self.len {
            Len::X1 | Len::X2 | Len::X4 => self.len.as_u32(),
            Len::X3 => 4,
        };
        // len * self.scalar.align() = power of 2 * power of 2 = power of 2
        U32PowerOf2::try_from_u32(len * self.scalar.align().as_u32()).unwrap()
    }
}

#[allow(missing_docs)]
impl ScalarType {
    pub const fn byte_size(&self) -> u64 {
        match self {
            ScalarType::F16 => 2,
            ScalarType::F32 | ScalarType::U32 | ScalarType::I32 => 4,
            ScalarType::F64 => 8,
        }
    }

    pub const fn align(&self) -> U32PowerOf2 {
        match self {
            ScalarType::F16 => U32PowerOf2::_2,
            ScalarType::F32 | ScalarType::U32 | ScalarType::I32 => U32PowerOf2::_4,
            ScalarType::F64 => U32PowerOf2::_8,
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy)]
pub enum MatrixMajor {
    Row,
    Column,
}

#[allow(missing_docs)]
impl Matrix {
    pub const fn byte_size(&self, repr: Repr, major: MatrixMajor) -> u64 {
        let (vec, array_len) = self.as_vector_array(major);
        let array_stride = array_stride(vec.align(repr), vec.byte_size());
        array_size(array_stride, array_len)
    }

    pub const fn align(&self, repr: Repr, major: MatrixMajor) -> U32PowerOf2 {
        let (vec, _) = self.as_vector_array(major);
        array_align(vec.align(repr), repr)
    }

    const fn as_vector_array(&self, major: MatrixMajor) -> (Vector, NonZeroU32) {
        let (vec_len, array_len): (Len, NonZeroU32) = match major {
            MatrixMajor::Row => (self.rows.as_len(), self.columns.as_non_zero_u32()),
            MatrixMajor::Column => (self.columns.as_len(), self.rows.as_non_zero_u32()),
        };
        (
            Vector {
                len: vec_len,
                scalar: self.scalar.as_scalar_type(),
            },
            array_len,
        )
    }
}

#[allow(missing_docs)]
impl Atomic {
    pub const fn byte_size(&self) -> u64 { self.scalar.as_scalar_type().byte_size() }
    pub const fn align(&self, repr: Repr) -> U32PowerOf2 {
        if repr.is_packed() {
            return PACKED_ALIGN;
        }
        self.scalar.as_scalar_type().align()
    }
}

#[allow(missing_docs)]
impl SizedArray {
    pub fn byte_size(&self, repr: Repr) -> u64 { array_size(self.byte_stride(repr), self.len) }

    pub fn align(&self, repr: Repr) -> U32PowerOf2 { array_align(self.element.align(repr), repr) }

    pub fn byte_stride(&self, repr: Repr) -> u64 {
        let (element_size, element_align) = self.element.byte_size_and_align(repr);
        array_stride(element_align, element_size)
    }
}

/// Returns an array's size given it's stride and length.
///
/// Note, this is independent of layout rules (`Repr`).
pub const fn array_size(array_stride: u64, len: NonZeroU32) -> u64 { array_stride * len.get() as u64 }

/// Returns an array's size given the alignment of it's elements.
pub const fn array_align(element_align: U32PowerOf2, repr: Repr) -> U32PowerOf2 {
    match repr {
        // Packedness is ensured by the `LayoutCalculator`.
        Repr::Storage => element_align,
        Repr::Uniform => round_up_align(U32PowerOf2::_16, element_align),
        Repr::Packed => PACKED_ALIGN,
    }
}

/// Returns an array's size given the alignment and size of it's elements.
pub const fn array_stride(element_align: U32PowerOf2, element_size: u64) -> u64 {
    // Arrays of element type T must have an element stride that is a multiple of the
    // RequiredAlignOf(T, C) for the address space C:
    round_up(element_align.as_u64(), element_size)
}

#[allow(missing_docs)]
impl RuntimeSizedArray {
    pub fn align(&self, repr: Repr) -> U32PowerOf2 { array_align(self.element.align(repr), repr) }

    pub fn byte_stride(&self, repr: Repr) -> u64 { array_stride(self.align(repr), self.element.byte_size(repr)) }
}

#[allow(missing_docs)]
impl SizedField {
    pub fn byte_size(&self, repr: Repr) -> u64 { self.ty.byte_size(repr) }
    pub fn align(&self, repr: Repr) -> U32PowerOf2 { self.ty.align(repr) }
}

#[allow(missing_docs)]
impl RuntimeSizedArrayField {
    pub fn align(&self, repr: Repr) -> U32PowerOf2 { self.array.align(repr) }
}

pub const fn round_up(multiple_of: u64, n: u64) -> u64 {
    match multiple_of {
        0 => match n {
            0 => 0,
            _ => panic!("cannot round up n to a multiple of 0"),
        },
        k @ 1.. => n.div_ceil(k) * k,
    }
}

pub const fn round_up_align(multiple_of: U32PowerOf2, n: U32PowerOf2) -> U32PowerOf2 {
    let rounded_up = round_up(multiple_of.as_u64(), n.as_u64());
    // n <= multiple_of  ->  rounded_up = multiple_of
    // n > multiple_of   ->  rounded_up = n, since both are powers of 2, n must already
    //                                       be a multiple of multiple_of
    // In both cases rounded_up is a power of 2
    U32PowerOf2::try_from_u32(rounded_up as u32).unwrap()
}
