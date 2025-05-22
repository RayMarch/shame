use super::super::LayoutCalculator;
use super::*;

//           Size and align of host-shareable types             //
// https://www.w3.org/TR/WGSL/#address-space-layout-constraints //

#[derive(Debug, Clone, Copy)]
pub enum Repr {
    /// Wgsl storage address space layout / OpenGL std430
    /// https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    Storage,
    /// Wgsl uniform address space layout / OpenGL std140
    /// https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    Uniform,
    /// Packed layout. Vertex buffer only.
    Packed,
}

impl SizedType {
    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the align.
    pub fn byte_size(&self, repr: Repr) -> u64 {
        match self {
            SizedType::Array(a) => a.byte_size(repr),
            SizedType::Vector(v) => v.byte_size(),
            SizedType::Matrix(m) => m.byte_size(Major::Row),
            SizedType::Atomic(a) => a.byte_size(),
            SizedType::PackedVec(v) => u8::from(v.byte_size()) as u64,
            SizedType::Struct(s) => s.byte_size_and_align(repr).0,
        }
    }

    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the size.
    pub fn byte_align(&self, repr: Repr) -> U32PowerOf2 {
        match self {
            SizedType::Array(a) => a.align(repr),
            SizedType::Vector(v) => v.align(),
            SizedType::Matrix(m) => m.align(repr, Major::Row),
            SizedType::Atomic(a) => a.align(),
            SizedType::PackedVec(v) => v.align(),
            SizedType::Struct(s) => s.byte_size_and_align(repr).1,
        }
    }

    /// This is expensive for structs.
    pub fn byte_size_and_align(&self, repr: Repr) -> (u64, U32PowerOf2) {
        match self {
            SizedType::Struct(s) => s.byte_size_and_align(repr),
            non_struct => (non_struct.byte_size(repr), non_struct.byte_align(repr)),
        }
    }
}


impl SizedStruct {
    pub fn field_offsets(&self, repr: Repr) -> FieldOffsets {
        FieldOffsets {
            fields: &self.fields,
            field_index: 0,
            calc: LayoutCalculator::new(matches!(repr, Repr::Packed)),
            repr,
        }
    }

    /// Returns (byte_size, byte_align)
    ///
    /// ### Careful!
    /// This is an expensive operation as it calculates byte size and align from scratch.
    /// If you also need field offsets, use [`SizedStruct::field_offsets`] instead and
    /// read the documentation of [`FieldOffsets`] on how to obtain the byte size and align from it.
    pub fn byte_size_and_align(&self, layout: Repr) -> (u64, U32PowerOf2) {
        let mut field_offsets = self.field_offsets(layout);
        (&mut field_offsets).count(); // &mut so it doesn't consume
        (field_offsets.byte_size(), field_offsets.align())
    }
}

/// An iterator over the field offsets of a `SizedStruct`.
///
/// `FieldOffsets::byte_size` and `FieldOffsets::byte_align` can be used to query the struct's
/// `byte_size` and `byte_align`, but only takes into account the fields that have been iterated over.
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
            let (size, align) = match &field.ty {
                SizedType::Struct(s) => {
                    let (size, align) = s.byte_size_and_align(self.repr);
                    match self.repr {
                        // Packedness is ensured by the `LayoutCalculator`.
                        Repr::Storage | Repr::Packed => (size, align),
                        // https://www.w3.org/TR/WGSL/#address-space-layout-constraints
                        // The uniform address space requires that:
                        // - struct S align: roundUp(16, AlignOf(S))
                        // - If a structure member itself has a structure type S, then the number of
                        // bytes between the start of that member and the start of any following
                        // member must be at least roundUp(16, SizeOf(S)).
                        // -> We adjust size too.
                        Repr::Uniform => (round_up(16, size), round_up_align(U32PowerOf2::_16, align)),
                    }
                }
                non_struct => non_struct.byte_size_and_align(self.repr),
            };

            self.calc
                .extend(size, align, field.custom_min_size, field.custom_min_align)
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
    pub fn sized_field_offsets(&self, repr: Repr) -> FieldOffsets {
        FieldOffsets {
            fields: &self.sized_fields,
            field_index: 0,
            calc: LayoutCalculator::new(matches!(repr, Repr::Packed)),
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

        let array_align = self.last_unsized.array.byte_align(offsets.repr);
        let custom_min_align = self.last_unsized.custom_min_align;
        let (offset, align) = offsets.calc.extend_unsized(array_align, custom_min_align);
        (offset, struct_align(align, offsets.repr))
    }

    /// This is expensive as it calculates the byte align from scratch.
    pub fn byte_align(&self, repr: Repr) -> U32PowerOf2 {
        let offsets = self.sized_field_offsets(repr);
        self.last_field_offset_and_struct_align(offsets).1
    }
}

const fn struct_align(align: U32PowerOf2, repr: Repr) -> U32PowerOf2 {
    match repr {
        // Packedness is ensured by the `LayoutCalculator`.
        Repr::Storage | Repr::Packed => align,
        Repr::Uniform => round_up_align(U32PowerOf2::_16, align),
    }
}

impl Vector {
    pub const fn new(scalar: ScalarType, len: Len) -> Self { Self { scalar, len } }

    pub const fn byte_size(&self) -> u64 { self.len.as_u64() * self.scalar.byte_size() }

    pub const fn align(&self) -> U32PowerOf2 {
        let len = match self.len {
            Len::X1 | Len::X2 | Len::X4 => self.len.as_u32(),
            Len::X3 => 4,
        };

        // len * self.scalar.align() = power of 2 * power of 2 = power of 2
        U32PowerOf2::try_from_u32(len * self.scalar.align().as_u32()).unwrap()
    }
}

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

#[derive(Debug, Clone, Copy)]
pub enum Major {
    Row,
    Column,
}

impl Matrix {
    pub const fn byte_size(&self, major: Major) -> u64 {
        let (vec, array_len) = self.as_vector_array(major);
        let array_stride = array_stride(vec.align(), vec.byte_size());
        array_size(array_stride, array_len)
    }

    pub const fn align(&self, repr: Repr, major: Major) -> U32PowerOf2 {
        let (vec, _) = self.as_vector_array(major);
        array_align(vec.align(), repr)
    }

    const fn as_vector_array(&self, major: Major) -> (Vector, NonZeroU32) {
        let (vec_len, array_len): (Len, NonZeroU32) = match major {
            Major::Row => (self.rows.as_len(), self.columns.as_non_zero_u32()),
            Major::Column => (self.columns.as_len(), self.rows.as_non_zero_u32()),
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

impl Atomic {
    pub const fn byte_size(&self) -> u64 { self.scalar.as_scalar_type().byte_size() }
    pub const fn align(&self) -> U32PowerOf2 { self.scalar.as_scalar_type().align() }
}

impl SizedArray {
    pub fn byte_size(&self, repr: Repr) -> u64 { array_size(self.byte_stride(repr), self.len) }

    pub fn align(&self, repr: Repr) -> U32PowerOf2 { array_align(self.element.byte_align(repr), repr) }

    pub fn byte_stride(&self, repr: Repr) -> u64 {
        let (element_size, element_align) = self.element.byte_size_and_align(repr);
        array_stride(element_align, element_size)
    }
}

pub const fn array_size(array_stride: u64, len: NonZeroU32) -> u64 { array_stride * len.get() as u64 }

pub const fn array_align(element_align: U32PowerOf2, layout: Repr) -> U32PowerOf2 {
    match layout {
        // Packedness is ensured by the `LayoutCalculator`.
        Repr::Storage | Repr::Packed => element_align,
        Repr::Uniform => U32PowerOf2::try_from_u32(round_up(16, element_align.as_u64()) as u32).unwrap(),
    }
}

pub const fn array_stride(element_align: U32PowerOf2, element_size: u64) -> u64 {
    // Arrays of element type T must have an element stride that is a multiple of the
    // RequiredAlignOf(T, C) for the address space C:
    round_up(element_align.as_u64(), element_size)
}

impl RuntimeSizedArray {
    pub fn byte_align(&self, repr: Repr) -> U32PowerOf2 { array_align(self.element.byte_align(repr), repr) }

    pub fn byte_stride(&self, repr: Repr) -> u64 { array_stride(self.byte_align(repr), self.element.byte_size(repr)) }
}

impl SizedField {
    pub fn byte_size(&self, repr: Repr) -> u64 { self.ty.byte_size(repr) }
    pub fn byte_align(&self, repr: Repr) -> U32PowerOf2 { self.ty.byte_align(repr) }
}

impl RuntimeSizedArrayField {
    pub fn byte_align(&self, repr: Repr) -> U32PowerOf2 { self.array.byte_align(repr) }
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
