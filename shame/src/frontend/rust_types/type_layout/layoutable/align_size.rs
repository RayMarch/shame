use super::super::{Repr};
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
            SizedType::Matrix(m) => m.byte_size(repr),
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
            SizedType::Matrix(m) => m.align(repr),
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
    /// fields of this struct. `SizedStruct::byte_size_and_align_from_offsets` can be
    /// used to efficiently obtain the byte_size
    pub fn field_offsets(&self, repr: Repr) -> FieldOffsetsSized {
        FieldOffsetsSized(FieldOffsets::new(self.fields(), repr))
    }

    /// Returns (byte_size, align)
    ///
    /// ### Careful!
    /// This is an expensive operation as it calculates byte size and align from scratch.
    /// If you also need field offsets, use [`SizedStruct::field_offsets`] instead and
    /// read the documentation of [`FieldOffsets`] on how to obtain the byte size and align from it.
    pub fn byte_size_and_align(&self, repr: Repr) -> (u64, U32PowerOf2) {
        self.field_offsets(repr).struct_byte_size_and_align()
    }
}

/// An iterator over the offsets of sized fields.
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
impl<'a> FieldOffsets<'a> {
    fn new(fields: &'a [SizedField], repr: Repr) -> Self {
        Self {
            fields,
            field_index: 0,
            calc: LayoutCalculator::new(repr),
            repr,
        }
    }
}

/// Iterator over the field offsets of a `SizedStruct`.
// The difference to `FieldOffsets` is that it also offers a `struct_byte_size_and_align` method.
pub struct FieldOffsetsSized<'a>(FieldOffsets<'a>);
impl Iterator for FieldOffsetsSized<'_> {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> { self.0.next() }
}
impl<'a> FieldOffsetsSized<'a> {
    /// Consumes self and calculates the byte size and align of a struct
    /// with exactly the sized fields that this FieldOffsets was created with.
    pub fn struct_byte_size_and_align(mut self) -> (u64, U32PowerOf2) {
        // Finishing layout calculations
        // using count only to advance iterator to the end
        (&mut self.0).count();
        (self.0.calc.byte_size(), struct_align(self.0.calc.align(), self.0.repr))
    }

    /// Returns the inner iterator over sized fields.
    pub fn into_inner(self) -> FieldOffsets<'a> { self.0 }
}

/// The field offsets of an `UnsizedStruct`.
///
/// - Use [`FieldOffsetsUnsized::sized_field_offsets`] for an iterator over the sized field offsets.
/// - Use [`FieldOffsetsUnsized::last_field_offset_and_struct_align`] for the last field's offset
///   and the struct's align
pub struct FieldOffsetsUnsized<'a> {
    sized: FieldOffsets<'a>,
    last_unsized: &'a RuntimeSizedArrayField,
}

impl<'a> FieldOffsetsUnsized<'a> {
    fn new(sized_fields: &'a [SizedField], last_unsized: &'a RuntimeSizedArrayField, repr: Repr) -> Self {
        Self {
            sized: FieldOffsets::new(sized_fields, repr),
            last_unsized,
        }
    }

    /// Returns an iterator over the sized field offsets.
    pub fn sized_field_offsets(&mut self) -> &mut FieldOffsets<'a> { &mut self.sized }

    /// Returns the last field's offset and the struct's align.
    pub fn last_field_offset_and_struct_align(mut self) -> (u64, U32PowerOf2) {
        // Finishing layout calculations
        // using count only to advance iterator to the end
        (&mut self.sized).count();
        let array_align = self.last_unsized.array.align(self.sized.repr);
        let custom_min_align = self.last_unsized.custom_min_align;
        let (offset, align) = self.sized.calc.extend_unsized(array_align, custom_min_align);
        (offset, struct_align(align, self.sized.repr))
    }

    /// Returns the inner iterator over sized fields.
    pub fn into_sized_fields(self) -> FieldOffsets<'a> { self.sized }
}

impl UnsizedStruct {
    /// Returns [`FieldOffsetsUnsized`].
    ///
    /// - Use [`FieldOffsetsUnsized::sized_field_offsets`] for an iterator over the sized field offsets.
    /// - Use [`FieldOffsetsUnsized::last_field_offset_and_struct_align`] for the last field's offset
    ///   and the struct's align
    pub fn field_offsets(&self, repr: Repr) -> FieldOffsetsUnsized {
        FieldOffsetsUnsized::new(&self.sized_fields, &self.last_unsized, repr)
    }

    /// This is expensive as it calculates the byte align from scratch.
    pub fn align(&self, repr: Repr) -> U32PowerOf2 { self.field_offsets(repr).last_field_offset_and_struct_align().1 }
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
        match repr {
            Repr::Packed => return PACKED_ALIGN,
            Repr::Storage | Repr::Uniform => {}
        }

        let len = match self.len {
            Len::X1 | Len::X2 | Len::X4 => self.len.as_u32(),
            Len::X3 => 4,
        };
        U32PowerOf2::try_from_u32(len * self.scalar.align().as_u32())
            .expect("power of 2 * power of 2 = power of 2. Highest operands are around 4 * 16 so overflow is unlikely")
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
    pub const fn byte_size(&self, repr: Repr) -> u64 {
        let (vec, array_len) = self.as_vector_array();
        let array_stride = array_stride(vec.align(repr), vec.byte_size());
        array_size(array_stride, array_len)
    }

    pub const fn align(&self, repr: Repr) -> U32PowerOf2 {
        let (vec, _) = self.as_vector_array();
        // AlignOf(vecR)
        vec.align(repr)
    }

    const fn as_vector_array(&self) -> (Vector, NonZeroU32) {
        let major = MatrixMajor::Column; // This can be made a parameter in the future.
        let (vec_len, array_len): (Len, NonZeroU32) = match major {
            MatrixMajor::Column => (self.rows.as_len(), self.columns.as_non_zero_u32()),
            MatrixMajor::Row => (self.columns.as_len(), self.rows.as_non_zero_u32()),
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

/// Returns an array's size=>stride (the distance between consecutive elements) given the alignment and size of its elements.
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
    pub fn byte_size(&self, repr: Repr) -> u64 {
        LayoutCalculator::calculate_byte_size(self.ty.byte_size(repr), self.custom_min_size)
    }
    pub fn align(&self, repr: Repr) -> U32PowerOf2 {
        // In case of Repr::Packed, the field's align of 1 is overwritten here by custom_min_align.
        // This is intended!
        LayoutCalculator::calculate_align(self.ty.align(repr), self.custom_min_align)
    }
}

#[allow(missing_docs)]
impl RuntimeSizedArrayField {
    pub fn align(&self, repr: Repr) -> U32PowerOf2 {
        // In case of Repr::Packed, the field's align of 1 is overwritten here by custom_min_align.
        // This is intended!
        LayoutCalculator::calculate_align(self.array.align(repr), self.custom_min_align)
    }
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

/// `LayoutCalculator` helps calculate the size, align and the field offsets of a struct.
///
/// If `LayoutCalculator` is created with `repr == Repr::Packed`, provided `field_align`s
/// are ignored and the field is inserted directly after the previous field. However,
/// a `custom_min_align` that is `Some` overwrites the "packedness" of the field.
#[derive(Debug, Clone)]
pub struct LayoutCalculator {
    next_offset_min: u64,
    align: U32PowerOf2,
    repr: Repr,
}

impl LayoutCalculator {
    /// Creates a new `LayoutCalculator`, which calculates the size, align and
    /// the field offsets of a gpu struct.
    pub const fn new(repr: Repr) -> Self {
        Self {
            next_offset_min: 0,
            align: U32PowerOf2::_1,
            repr,
        }
    }

    /// Extends the layout by a field.
    ///
    /// `is_struct` must be true if the field is a struct.
    ///
    /// Returns the field's offset.
    pub const fn extend(
        &mut self,
        field_size: u64,
        mut field_align: U32PowerOf2,
        custom_min_size: Option<u64>,
        custom_min_align: Option<U32PowerOf2>,
        is_struct: bool,
    ) -> u64 {
        // Just in case the user didn't already do this.
        if self.repr.is_packed() {
            field_align = PACKED_ALIGN;
        }

        let size = Self::calculate_byte_size(field_size, custom_min_size);
        let align = Self::calculate_align(field_align, custom_min_align);

        let offset = self.next_field_offset(align, custom_min_align);
        self.next_offset_min = match (self.repr, is_struct) {
            // The uniform address space requires that:
            // - If a structure member itself has a structure type S, then the number of
            // bytes between the start of that member and the start of any following
            // member must be at least roundUp(16, SizeOf(S)).
            (Repr::Uniform, true) => round_up(16, offset + size),
            _ => offset + size,
        };
        self.align = self.align.max(align);

        offset
    }

    /// Extends the layout by an runtime sized array field given it's align.
    ///
    /// Returns (last field offset, align)
    ///
    /// `self` is consumed, so that no further fields may be extended, because
    /// only the last field may be unsized.
    pub const fn extend_unsized(
        mut self,
        mut field_align: U32PowerOf2,
        custom_min_align: Option<U32PowerOf2>,
    ) -> (u64, U32PowerOf2) {
        // Just in case the user didn't already do this.
        if self.repr.is_packed() {
            field_align = PACKED_ALIGN;
        }

        let align = Self::calculate_align(field_align, custom_min_align);

        let offset = self.next_field_offset(align, custom_min_align);
        self.align = self.align.max(align);

        (offset, self.align)
    }

    /// Returns the byte size of the struct.
    // wgsl spec:
    //   roundUp(AlignOf(S), justPastLastMember)
    //   where justPastLastMember = OffsetOfMember(S,N) + SizeOfMember(S,N)
    //
    // self.next_offset_min is justPastLastMember already.
    pub const fn byte_size(&self) -> u64 { round_up(self.align.as_u64(), self.next_offset_min) }

    /// Returns the align of the struct.
    pub const fn align(&self) -> U32PowerOf2 { self.align }

    /// field_align should already respect field_custom_min_align.
    /// field_custom_min_align is used to overwrite packing if self is packed.
    const fn next_field_offset(&self, field_align: U32PowerOf2, field_custom_min_align: Option<U32PowerOf2>) -> u64 {
        match (self.repr, field_custom_min_align) {
            (Repr::Packed, None) => self.next_offset_min,
            (Repr::Packed, Some(custom_align)) => round_up(custom_align.as_u64(), self.next_offset_min),
            (_, _) => round_up(field_align.as_u64(), self.next_offset_min),
        }
    }

    pub(crate) const fn calculate_byte_size(byte_size: u64, custom_min_size: Option<u64>) -> u64 {
        // const byte_size.max(custom_min_size.unwrap_or(0))
        if let Some(min_size) = custom_min_size {
            if min_size > byte_size {
                return min_size;
            }
        }
        byte_size
    }

    pub(crate) const fn calculate_align(align: U32PowerOf2, custom_min_align: Option<U32PowerOf2>) -> U32PowerOf2 {
        // const align.max(custom_min_align.unwrap_or(U32PowerOf2::_1))
        if let Some(min_align) = custom_min_align {
            align.max(min_align)
        } else {
            align
        }
    }
}
