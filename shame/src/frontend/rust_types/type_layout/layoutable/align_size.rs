use super::super::{Repr};
use super::*;

//              Size and align of layoutable types              //
// https://www.w3.org/TR/WGSL/#address-space-layout-constraints //

pub(crate) const PACKED_ALIGN: U32PowerOf2 = U32PowerOf2::_1;

impl LayoutableType {
    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the align.
    pub fn byte_size(&self, default_repr: Repr) -> Option<u64> {
        match self {
            LayoutableType::Sized(s) => Some(s.byte_size(default_repr)),
            LayoutableType::UnsizedStruct(_) | LayoutableType::RuntimeSizedArray(_) => None,
        }
    }

    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the size.
    pub fn align(&self, default_repr: Repr) -> U32PowerOf2 {
        match self {
            LayoutableType::Sized(s) => s.align(default_repr),
            LayoutableType::UnsizedStruct(s) => s.align(),
            LayoutableType::RuntimeSizedArray(a) => a.align(default_repr),
        }
    }

    /// This is expensive for structs as it calculates the byte size and align by traversing all fields recursively.
    pub fn byte_size_and_align(&self, default_repr: Repr) -> (Option<u64>, U32PowerOf2) {
        match self {
            LayoutableType::Sized(s) => {
                let (size, align) = s.byte_size_and_align(default_repr);
                (Some(size), align)
            }
            LayoutableType::UnsizedStruct(s) => (None, s.align()),
            LayoutableType::RuntimeSizedArray(a) => (None, a.align(default_repr)),
        }
    }

    pub fn to_unified_repr(&self, repr: Repr) -> Self {
        let mut this = self.clone();
        this.change_all_repr(repr);
        this
    }

    // Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) {
        match self {
            LayoutableType::Sized(s) => s.change_all_repr(repr),
            LayoutableType::UnsizedStruct(s) => s.change_all_repr(repr),
            LayoutableType::RuntimeSizedArray(a) => a.change_all_repr(repr),
        }
    }
}

impl SizedType {
    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the align.
    pub fn byte_size(&self, parent_repr: Repr) -> u64 { self.byte_size_and_align(parent_repr).0 }

    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the size.
    pub fn align(&self, parent_repr: Repr) -> U32PowerOf2 { self.byte_size_and_align(parent_repr).1 }

    /// This is expensive for structs as it calculates the byte size and align by traversing all fields recursively.
    pub fn byte_size_and_align(&self, parent_repr: Repr) -> (u64, U32PowerOf2) {
        let repr = parent_repr;
        match self {
            SizedType::Array(a) => (a.byte_size(parent_repr), a.align(parent_repr)),
            SizedType::Vector(v) => (v.byte_size(parent_repr), v.align(parent_repr)),
            SizedType::Matrix(m) => (m.byte_size(parent_repr), m.align(parent_repr)),
            SizedType::Atomic(a) => (a.byte_size(), a.align(parent_repr)),
            SizedType::PackedVec(v) => (u8::from(v.byte_size()) as u64, v.align(parent_repr)),
            SizedType::Struct(s) => s.byte_size_and_align(),
        }
    }

    // Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) {
        match self {
            SizedType::Struct(s) => s.change_all_repr(repr),
            SizedType::Atomic(_) |
            SizedType::PackedVec(_) |
            SizedType::Vector(_) |
            SizedType::Matrix(_) |
            SizedType::Array(_) => {
                // No repr to change for these types.
            }
        }
    }
}

impl SizedStruct {
    /// Returns [`FieldOffsetsSized`], which serves as an iterator over the offsets of the
    /// fields of this struct. `FieldOffsetsSized::struct_byte_size_and_align` can be
    /// used to efficiently obtain the byte_size and align.
    pub fn field_offsets(&self) -> FieldOffsetsSized { FieldOffsetsSized(FieldOffsets::new(self.fields(), self.repr)) }

    /// Returns (byte_size, align)
    ///
    /// This is expensive for structs as it calculates the byte size and align by traversing all fields recursively.
    pub fn byte_size_and_align(&self) -> (u64, U32PowerOf2) { self.field_offsets().struct_byte_size_and_align() }

    // Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) {
        self.repr = repr;
        for field in &mut self.fields {
            field.ty.change_all_repr(repr);
        }
    }
}

/// An iterator over the offsets of sized fields.
pub struct FieldOffsets<'a> {
    fields: &'a [SizedField],
    field_index: usize,
    calc: StructLayoutCalculator,
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
            calc: StructLayoutCalculator::new(repr),
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
        (self.0.calc.byte_size(), self.0.calc.align())
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
        self.sized.calc.extend_unsized(array_align, custom_min_align)
    }

    /// Returns the inner iterator over sized fields.
    pub fn into_inner(self) -> FieldOffsets<'a> { self.sized }
}

impl UnsizedStruct {
    /// Returns [`FieldOffsetsUnsized`].
    ///
    /// - Use [`FieldOffsetsUnsized::sized_field_offsets`] for an iterator over the sized field offsets.
    /// - Use [`FieldOffsetsUnsized::last_field_offset_and_struct_align`] for the last field's offset
    ///   and the struct's align
    pub fn field_offsets(&self) -> FieldOffsetsUnsized {
        FieldOffsetsUnsized::new(&self.sized_fields, &self.last_unsized, self.repr)
    }

    /// This is expensive as it calculates the byte align by traversing all fields recursively.
    pub fn align(&self) -> U32PowerOf2 { self.field_offsets().last_field_offset_and_struct_align().1 }

    // Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) {
        self.repr = repr;
        for field in &mut self.sized_fields {
            field.ty.change_all_repr(repr);
        }
        self.last_unsized.array.change_all_repr(repr);
    }
}

#[allow(missing_docs)]
impl Vector {
    pub const fn new(scalar: ScalarType, len: Len) -> Self { Self { scalar, len } }

    pub const fn byte_size(&self, repr: Repr) -> u64 {
        match repr {
            Repr::Storage | Repr::Uniform | Repr::Packed => self.len.as_u64() * self.scalar.byte_size(),
        }
    }

    pub const fn align(&self, repr: Repr) -> U32PowerOf2 {
        match repr {
            Repr::Packed => PACKED_ALIGN,
            Repr::Storage | Repr::Uniform => {
                let po2_len = match self.len {
                    Len::X1 | Len::X2 | Len::X4 => self.len.as_u32(),
                    Len::X3 => 4,
                };
                let po2_align = self.scalar.align(repr);
                U32PowerOf2::try_from_u32(po2_len * po2_align.as_u32()).expect(
                    "power of 2 * power of 2 = power of 2. Highest operands are around 4 * 16 so overflow is unlikely",
                )
            }
        }
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

    pub const fn align(&self, repr: Repr) -> U32PowerOf2 {
        match repr {
            Repr::Packed => PACKED_ALIGN,
            Repr::Storage | Repr::Uniform => match self {
                ScalarType::F16 => U32PowerOf2::_2,
                ScalarType::F32 | ScalarType::U32 | ScalarType::I32 => U32PowerOf2::_4,
                ScalarType::F64 => U32PowerOf2::_8,
            },
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
        // According to https://www.w3.org/TR/WGSL/#alignment-and-size
        // SizeOf(matCxR) = SizeOf(array<vecR, C>) = C × roundUp(AlignOf(vecR), SizeOf(vecR))
        array_len.get() as u64 * round_up(vec.align(repr).as_u64(), vec.byte_size(repr))
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
        match repr {
            Repr::Packed => return PACKED_ALIGN,
            Repr::Storage | Repr::Uniform => {}
        }

        self.scalar.as_scalar_type().align(repr)
    }
}

#[allow(missing_docs)]
impl SizedArray {
    pub fn byte_size(&self, repr: Repr) -> u64 { array_size(self.byte_stride(repr), self.len) }

    pub fn align(&self, repr: Repr) -> U32PowerOf2 { array_align(self.element.align(repr), repr) }

    pub fn byte_stride(&self, repr: Repr) -> u64 {
        let (element_size, element_align) = self.element.byte_size_and_align(repr);
        array_stride(element_align, element_size, repr)
    }

    // Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) {
        let mut element = (*self.element).clone();
        element.change_all_repr(repr);
        self.element = Rc::new(element);
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
pub const fn array_stride(element_align: U32PowerOf2, element_size: u64, repr: Repr) -> u64 {
    let element_align = match repr {
        Repr::Storage => element_align,
        // This should already be the case, but doesn't hurt to ensure.
        Repr::Packed => PACKED_ALIGN,
        // The uniform address space also requires that:
        // Array elements are aligned to 16 byte boundaries.
        // That is, StrideOf(array<T,N>) = 16 × k’ for some positive integer k'.
        // - https://www.w3.org/TR/WGSL/#address-space-layout-constraints
        Repr::Uniform => round_up_align(U32PowerOf2::_16, element_align),
    };

    round_up(element_align.as_u64(), element_size)
}

#[allow(missing_docs)]
impl RuntimeSizedArray {
    pub fn align(&self, parent_repr: Repr) -> U32PowerOf2 { array_align(self.element.align(parent_repr), parent_repr) }

    pub fn byte_stride(&self, parent_repr: Repr) -> u64 {
        array_stride(
            self.align(parent_repr),
            self.element.byte_size(parent_repr),
            parent_repr,
        )
    }

    // Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) { self.element.change_all_repr(repr); }
}

#[allow(missing_docs)]
impl SizedField {
    pub fn byte_size(&self, repr: Repr) -> u64 {
        StructLayoutCalculator::calculate_byte_size(self.ty.byte_size(repr), self.custom_min_size)
    }
    pub fn align(&self, repr: Repr) -> U32PowerOf2 {
        StructLayoutCalculator::calculate_align(self.ty.align(repr), self.custom_min_align, repr)
    }
}

#[allow(missing_docs)]
impl RuntimeSizedArrayField {
    pub fn align(&self, repr: Repr) -> U32PowerOf2 {
        StructLayoutCalculator::calculate_align(self.array.align(repr), self.custom_min_align, repr)
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
pub struct StructLayoutCalculator {
    next_offset_min: u64,
    align: U32PowerOf2,
    repr: Repr,
}

impl StructLayoutCalculator {
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
        match self.repr {
            Repr::Packed => field_align = PACKED_ALIGN,
            Repr::Storage | Repr::Uniform => {}
        }

        let size = Self::calculate_byte_size(field_size, custom_min_size);
        let align = Self::calculate_align(field_align, custom_min_align, self.repr);

        let offset = self.next_field_offset(align, custom_min_align);
        self.next_offset_min = match (self.repr, is_struct) {
            // The uniform address space requires that:
            // - If a structure member itself has a structure type S, then the number of
            // bytes between the start of that member and the start of any following
            // member must be at least roundUp(16, SizeOf(S)).
            (Repr::Uniform, true) => round_up(16, offset + size),
            (Repr::Storage | Repr::Packed, _) | (Repr::Uniform, false) => offset + size,
        };
        self.align = self.align.max(align);

        offset
    }

    /// Extends the layout by a runtime sized array field given it's align.
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
        match self.repr {
            Repr::Packed => field_align = PACKED_ALIGN,
            Repr::Storage | Repr::Uniform => {}
        }

        let align = Self::calculate_align(field_align, custom_min_align, self.repr);

        let offset = self.next_field_offset(align, custom_min_align);
        self.align = self.align.max(align);

        (offset, self.align())
    }

    /// Returns the byte size of the struct.
    // wgsl spec:
    //   roundUp(AlignOf(S), justPastLastMember)
    //   where justPastLastMember = OffsetOfMember(S,N) + SizeOfMember(S,N)
    //
    // self.next_offset_min is justPastLastMember already.
    pub const fn byte_size(&self) -> u64 { round_up(self.align().as_u64(), self.next_offset_min) }

    /// Returns the align of the struct.
    pub const fn align(&self) -> U32PowerOf2 { Self::adjust_struct_alignment_for_repr(self.align, self.repr) }

    const fn next_field_offset(&self, field_align: U32PowerOf2, field_custom_min_align: Option<U32PowerOf2>) -> u64 {
        let field_align = Self::calculate_align(field_align, field_custom_min_align, self.repr);
        match (self.repr, field_custom_min_align) {
            (Repr::Packed, None) => self.next_offset_min,
            (Repr::Packed, Some(custom_align)) => round_up(custom_align.as_u64(), self.next_offset_min),
            (Repr::Storage | Repr::Uniform, _) => round_up(field_align.as_u64(), self.next_offset_min),
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

    pub(crate) const fn calculate_align(
        align: U32PowerOf2,
        custom_min_align: Option<U32PowerOf2>,
        repr: Repr,
    ) -> U32PowerOf2 {
        match repr {
            Repr::Storage | Repr::Uniform => {
                // const align.max(custom_min_align.unwrap_or(U32PowerOf2::_1))
                if let Some(min_align) = custom_min_align {
                    align.max(min_align)
                } else {
                    align
                }
            }
            // custom_min_align is ignored in packed structs and the align is always 1.
            Repr::Packed => PACKED_ALIGN,
        }
    }

    const fn adjust_struct_alignment_for_repr(align: U32PowerOf2, repr: Repr) -> U32PowerOf2 {
        match repr {
            // Packedness is ensured by the `LayoutCalculator`.
            Repr::Storage => align,
            Repr::Uniform => round_up_align(U32PowerOf2::_16, align),
            Repr::Packed => PACKED_ALIGN,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::any::U32PowerOf2;
    use crate::frontend::rust_types::type_layout::Repr;
    use crate::ir::{Len, Len2, ScalarTypeFp, ScalarTypeInteger};
    use std::num::NonZeroU32;
    use std::rc::Rc;
    use super::super::builder::FieldOptions;

    #[test]
    fn test_primitives_layout() {
        // Testing all aligns and sizes found here (and some more that aren't in the spec like f64)
        // https://www.w3.org/TR/WGSL/#alignment-and-size

        // i32, u32, or f32: AlilgnOf(T) = 4, SizeOf(T) = 4
        assert_eq!(ScalarType::I32.byte_size(), 4);
        assert_eq!(ScalarType::I32.align(Repr::Storage), U32PowerOf2::_4);
        assert_eq!(ScalarType::U32.byte_size(), 4);
        assert_eq!(ScalarType::U32.align(Repr::Storage), U32PowerOf2::_4);
        assert_eq!(ScalarType::F32.byte_size(), 4);
        assert_eq!(ScalarType::F32.align(Repr::Storage), U32PowerOf2::_4);
        // f16: AlilgnOf(T) = 2, SizeOf(T) = 2
        assert_eq!(ScalarType::F16.byte_size(), 2);
        assert_eq!(ScalarType::F16.align(Repr::Storage), U32PowerOf2::_2);
        // not found in spec
        assert_eq!(ScalarType::F64.byte_size(), 8);
        assert_eq!(ScalarType::F64.align(Repr::Storage), U32PowerOf2::_8);

        // Test atomics
        let atomic_u32 = Atomic {
            scalar: ScalarTypeInteger::U32,
        };
        let atomic_i32 = Atomic {
            scalar: ScalarTypeInteger::I32,
        };
        // atomic<T>: AlignOf(T) = 4, SizeOf(T) = 4
        assert_eq!(atomic_u32.align(Repr::Storage), U32PowerOf2::_4);
        assert_eq!(atomic_u32.byte_size(), 4);
        assert_eq!(atomic_i32.align(Repr::Storage), U32PowerOf2::_4);
        assert_eq!(atomic_i32.byte_size(), 4);

        // Test vectors
        let vec2_f32 = Vector::new(ScalarType::F32, Len::X2);
        let vec2_f16 = Vector::new(ScalarType::F16, Len::X2);
        let vec3_f32 = Vector::new(ScalarType::F32, Len::X3);
        let vec3_f16 = Vector::new(ScalarType::F16, Len::X3);
        let vec4_f32 = Vector::new(ScalarType::F32, Len::X4);
        let vec4_f16 = Vector::new(ScalarType::F16, Len::X4);
        // vec2<T>, T is i32, u32, or f32: AlignOf(T) = 8, SizeOf(T) = 8
        assert_eq!(vec2_f32.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(vec2_f32.byte_size(Repr::Storage), 8);
        // vec2<f16>: AlignOf(T) = 4, SizeOf(T) = 4
        assert_eq!(vec2_f16.align(Repr::Storage), U32PowerOf2::_4);
        assert_eq!(vec2_f16.byte_size(Repr::Storage), 4);
        // vec3<T>, T is i32, u32, or f32: AlignOf(T) = 16, SizeOf(T) = 12
        assert_eq!(vec3_f32.align(Repr::Storage), U32PowerOf2::_16);
        assert_eq!(vec3_f32.byte_size(Repr::Storage), 12);
        // vec3<f16>: AlignOf(T) = 8, SizeOf(T) = 6
        assert_eq!(vec3_f16.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(vec3_f16.byte_size(Repr::Storage), 6);
        // vec4<T>, T is i32, u32, or f32: AlignOf(T) = 16, SizeOf(T) = 16
        assert_eq!(vec4_f32.align(Repr::Storage), U32PowerOf2::_16);
        assert_eq!(vec4_f32.byte_size(Repr::Storage), 16);
        // vec4<f16>: AlignOf(T) = 8, SizeOf(T) = 8
        assert_eq!(vec4_f16.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(vec4_f16.byte_size(Repr::Storage), 8);

        // Test matrices
        let mat2x2_f32 = Matrix {
            scalar: ScalarTypeFp::F32,
            columns: Len2::X2,
            rows: Len2::X2,
        };
        let mat2x2_f16 = Matrix {
            scalar: ScalarTypeFp::F16,
            columns: Len2::X2,
            rows: Len2::X2,
        };
        let mat3x2_f32 = Matrix {
            scalar: ScalarTypeFp::F32,
            columns: Len2::X3,
            rows: Len2::X2,
        };
        let mat3x2_f16 = Matrix {
            scalar: ScalarTypeFp::F16,
            columns: Len2::X3,
            rows: Len2::X2,
        };
        let mat4x2_f32 = Matrix {
            scalar: ScalarTypeFp::F32,
            columns: Len2::X4,
            rows: Len2::X2,
        };
        let mat4x2_f16 = Matrix {
            scalar: ScalarTypeFp::F16,
            columns: Len2::X4,
            rows: Len2::X2,
        };
        let mat2x3_f32 = Matrix {
            scalar: ScalarTypeFp::F32,
            columns: Len2::X2,
            rows: Len2::X3,
        };
        let mat2x3_f16 = Matrix {
            scalar: ScalarTypeFp::F16,
            columns: Len2::X2,
            rows: Len2::X3,
        };
        let mat3x3_f32 = Matrix {
            scalar: ScalarTypeFp::F32,
            columns: Len2::X3,
            rows: Len2::X3,
        };
        let mat3x3_f16 = Matrix {
            scalar: ScalarTypeFp::F16,
            columns: Len2::X3,
            rows: Len2::X3,
        };
        let mat4x3_f32 = Matrix {
            scalar: ScalarTypeFp::F32,
            columns: Len2::X4,
            rows: Len2::X3,
        };
        let mat4x3_f16 = Matrix {
            scalar: ScalarTypeFp::F16,
            columns: Len2::X4,
            rows: Len2::X3,
        };
        let mat2x4_f32 = Matrix {
            scalar: ScalarTypeFp::F32,
            columns: Len2::X2,
            rows: Len2::X4,
        };
        let mat2x4_f16 = Matrix {
            scalar: ScalarTypeFp::F16,
            columns: Len2::X2,
            rows: Len2::X4,
        };
        let mat3x4_f32 = Matrix {
            scalar: ScalarTypeFp::F32,
            columns: Len2::X3,
            rows: Len2::X4,
        };
        let mat3x4_f16 = Matrix {
            scalar: ScalarTypeFp::F16,
            columns: Len2::X3,
            rows: Len2::X4,
        };
        let mat4x4_f32 = Matrix {
            scalar: ScalarTypeFp::F32,
            columns: Len2::X4,
            rows: Len2::X4,
        };
        let mat4x4_f16 = Matrix {
            scalar: ScalarTypeFp::F16,
            columns: Len2::X4,
            rows: Len2::X4,
        };
        // mat2x2<f32>: AlignOf(T) = 8, SizeOf(T) = 16
        assert_eq!(mat2x2_f32.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(mat2x2_f32.byte_size(Repr::Storage), 16);
        // mat2x2<f16>: AlignOf(T) = 4, SizeOf(T) = 8
        assert_eq!(mat2x2_f16.align(Repr::Storage), U32PowerOf2::_4);
        assert_eq!(mat2x2_f16.byte_size(Repr::Storage), 8);
        // mat3x2<f32>: AlignOf(T) = 8, SizeOf(T) = 24
        assert_eq!(mat3x2_f32.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(mat3x2_f32.byte_size(Repr::Storage), 24);
        // mat3x2<f16>: AlignOf(T) = 4, SizeOf(T) = 12
        assert_eq!(mat3x2_f16.align(Repr::Storage), U32PowerOf2::_4);
        assert_eq!(mat3x2_f16.byte_size(Repr::Storage), 12);
        // mat4x2<f32>: AlignOf(T) = 8, SizeOf(T) = 32
        assert_eq!(mat4x2_f32.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(mat4x2_f32.byte_size(Repr::Storage), 32);
        // mat4x2<f16>: AlignOf(T) = 4, SizeOf(T) = 16
        assert_eq!(mat4x2_f16.align(Repr::Storage), U32PowerOf2::_4);
        assert_eq!(mat4x2_f16.byte_size(Repr::Storage), 16);
        // mat2x3<f32>: AlignOf(T) = 16, SizeOf(T) = 32
        assert_eq!(mat2x3_f32.align(Repr::Storage), U32PowerOf2::_16);
        assert_eq!(mat2x3_f32.byte_size(Repr::Storage), 32);
        // mat2x3<f16>: AlignOf(T) = 8, SizeOf(T) = 16
        assert_eq!(mat2x3_f16.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(mat2x3_f16.byte_size(Repr::Storage), 16);
        // mat3x3<f32>: AlignOf(T) = 16, SizeOf(T) = 48
        assert_eq!(mat3x3_f32.align(Repr::Storage), U32PowerOf2::_16);
        assert_eq!(mat3x3_f32.byte_size(Repr::Storage), 48);
        // mat3x3<f16>: AlignOf(T) = 8, SizeOf(T) = 24
        assert_eq!(mat3x3_f16.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(mat3x3_f16.byte_size(Repr::Storage), 24);
        // mat4x3<f32>: AlignOf(T) = 16, SizeOf(T) = 64
        assert_eq!(mat4x3_f32.align(Repr::Storage), U32PowerOf2::_16);
        assert_eq!(mat4x3_f32.byte_size(Repr::Storage), 64);
        // mat4x3<f16>: AlignOf(T) = 8, SizeOf(T) = 32
        assert_eq!(mat4x3_f16.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(mat4x3_f16.byte_size(Repr::Storage), 32);
        // mat2x4<f32>: AlignOf(T) = 16, SizeOf(T) = 32
        assert_eq!(mat2x4_f32.align(Repr::Storage), U32PowerOf2::_16);
        assert_eq!(mat2x4_f32.byte_size(Repr::Storage), 32);
        // mat2x4<f16>: AlignOf(T) = 8, SizeOf(T) = 16
        assert_eq!(mat2x4_f16.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(mat2x4_f16.byte_size(Repr::Storage), 16);
        // mat3x4<f32>: AlignOf(T) = 16, SizeOf(T) = 48
        assert_eq!(mat3x4_f32.align(Repr::Storage), U32PowerOf2::_16);
        assert_eq!(mat3x4_f32.byte_size(Repr::Storage), 48);
        // mat3x4<f16>: AlignOf(T) = 8, SizeOf(T) = 24
        assert_eq!(mat3x4_f16.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(mat3x4_f16.byte_size(Repr::Storage), 24);
        // mat4x4<f32>: AlignOf(T) = 16, SizeOf(T) = 64
        assert_eq!(mat4x4_f32.align(Repr::Storage), U32PowerOf2::_16);
        assert_eq!(mat4x4_f32.byte_size(Repr::Storage), 64);
        // mat4x4<f16>: AlignOf(T) = 8, SizeOf(T) = 32
        assert_eq!(mat4x4_f16.align(Repr::Storage), U32PowerOf2::_8);
        assert_eq!(mat4x4_f16.byte_size(Repr::Storage), 32);

        //   Testing Repr::Uniform and Repr::Packed    //

        let scalars = [
            ScalarType::F16,
            ScalarType::F32,
            ScalarType::F64,
            ScalarType::I32,
            ScalarType::U32,
        ];
        let atomics = [atomic_u32, atomic_i32];
        let vectors = [vec2_f32, vec2_f16, vec3_f32, vec3_f16, vec4_f32, vec4_f16];
        let matrices = [
            mat2x2_f32, mat2x2_f16, mat3x2_f32, mat3x2_f16, mat4x2_f32, mat4x2_f16, mat2x3_f32, mat2x3_f16, mat3x3_f32,
            mat3x3_f16, mat4x3_f32, mat4x3_f16, mat2x4_f32, mat2x4_f16, mat3x4_f32, mat3x4_f16, mat4x4_f32, mat4x4_f16,
        ];

        // Testing
        // - byte size for Storage, Uniform and Packed is the same
        // - align of Storage and Uniform is the same
        // - align of Packed is 1
        for scalar in scalars {
            // Looks silly, because byte_size doesn't have a repr argument.
            assert_eq!(scalar.byte_size(), scalar.byte_size());
            assert_eq!(scalar.align(Repr::Storage), scalar.align(Repr::Uniform));
            assert_eq!(scalar.align(Repr::Packed), U32PowerOf2::_1);
        }
        for atomic in atomics {
            // Looks silly, because byte_size doesn't have a repr argument.
            assert_eq!(atomic.byte_size(), atomic.byte_size());
            assert_eq!(atomic.align(Repr::Storage), atomic.align(Repr::Uniform));
            assert_eq!(atomic.align(Repr::Packed), U32PowerOf2::_1);
        }
        for vector in vectors {
            assert_eq!(vector.byte_size(Repr::Storage), vector.byte_size(Repr::Uniform));
            assert_eq!(vector.align(Repr::Storage), vector.align(Repr::Uniform));
            assert_eq!(vector.align(Repr::Packed), U32PowerOf2::_1);
        }
        for matrix in matrices {
            assert_eq!(matrix.byte_size(Repr::Storage), matrix.byte_size(Repr::Uniform));
            assert_eq!(matrix.align(Repr::Storage), matrix.align(Repr::Uniform));
            assert_eq!(matrix.align(Repr::Packed), U32PowerOf2::_1);
        }
    }

    #[test]
    fn test_sized_array_layout() {
        let element = SizedType::Vector(Vector::new(ScalarType::F32, Len::X2));
        let array = SizedArray {
            element: Rc::new(element),
            len: NonZeroU32::new(5).unwrap(),
        };

        // vec2<f32> is 8 bytes, aligned to 8 bytes
        assert_eq!(array.byte_stride(Repr::Storage), 8);
        assert_eq!(array.byte_size(Repr::Storage), 40); // 5 * 8
        assert_eq!(array.align(Repr::Storage), U32PowerOf2::_8);

        // Uniform requires 16-byte alignment for array elements
        assert_eq!(array.byte_stride(Repr::Uniform), 16);
        assert_eq!(array.byte_size(Repr::Uniform), 80); // 5 * 16
        assert_eq!(array.align(Repr::Uniform), U32PowerOf2::_16);

        // Packed has 1-byte alignment
        assert_eq!(array.byte_stride(Repr::Packed), 8);
        assert_eq!(array.byte_size(Repr::Packed), 40); // 5 * 8
        assert_eq!(array.align(Repr::Packed), U32PowerOf2::_1);
    }

    #[test]
    fn test_runtime_sized_array_layout() {
        let element = SizedType::Vector(Vector::new(ScalarType::F32, Len::X2));
        let array = RuntimeSizedArray { element };

        assert_eq!(array.byte_stride(Repr::Storage), 8);
        assert_eq!(array.align(Repr::Storage), U32PowerOf2::_8);

        assert_eq!(array.byte_stride(Repr::Uniform), 16);
        assert_eq!(array.align(Repr::Uniform), U32PowerOf2::_16);

        assert_eq!(array.byte_stride(Repr::Packed), 8);
        assert_eq!(array.align(Repr::Packed), U32PowerOf2::_1);
    }

    #[test]
    fn test_array_size() {
        let len = NonZeroU32::new(5).unwrap();
        assert_eq!(array_size(8, len), 40);
        assert_eq!(array_size(16, len), 80);
        assert_eq!(array_size(1, len), 5);
    }

    #[test]
    fn test_array_align() {
        let element_align = U32PowerOf2::_8;
        assert_eq!(array_align(element_align, Repr::Storage), U32PowerOf2::_8);
        assert_eq!(array_align(element_align, Repr::Uniform), U32PowerOf2::_16);
        assert_eq!(array_align(element_align, Repr::Packed), U32PowerOf2::_1);

        let small_align = U32PowerOf2::_4;
        assert_eq!(array_align(small_align, Repr::Storage), U32PowerOf2::_4);
        assert_eq!(array_align(small_align, Repr::Uniform), U32PowerOf2::_16);
        assert_eq!(array_align(small_align, Repr::Packed), U32PowerOf2::_1);
    }

    #[test]
    fn test_array_stride() {
        let element_align = U32PowerOf2::_8;
        let element_size = 12;

        // Storage: round up to element alignment
        assert_eq!(array_stride(element_align, element_size, Repr::Storage), 16);
        // Uniform: round up to 16-byte alignment
        assert_eq!(array_stride(element_align, element_size, Repr::Uniform), 16);
        // Packed: round up to 1-byte alignment (no padding)
        assert_eq!(array_stride(element_align, element_size, Repr::Packed), 12);
    }

    #[test]
    fn test_layout_calculator_basic() {
        let mut calc = StructLayoutCalculator::new(Repr::Storage);

        // Add a u32 field
        let offset1 = calc.extend(4, U32PowerOf2::_4, None, None, false);
        assert_eq!(offset1, 0);
        // Add another u32 field
        let offset2 = calc.extend(4, U32PowerOf2::_4, None, None, false);
        assert_eq!(offset2, 4);
        // Add a vec2<f32> field (8 bytes, 8-byte aligned)
        let offset3 = calc.extend(8, U32PowerOf2::_8, None, None, false);
        assert_eq!(offset3, 8);

        assert_eq!(calc.byte_size(), 16);
        assert_eq!(calc.align(), U32PowerOf2::_8);
    }

    #[test]
    fn test_layout_calculator_packed() {
        let mut calc = StructLayoutCalculator::new(Repr::Packed);

        // Add a u32 field - should be packed without padding
        let offset1 = calc.extend(4, U32PowerOf2::_4, None, None, false);
        assert_eq!(offset1, 0);
        // Add a vec2<f32> field - should be packed directly after
        let offset2 = calc.extend(8, U32PowerOf2::_8, None, None, false);
        assert_eq!(offset2, 4);

        assert_eq!(calc.byte_size(), 12);
        assert_eq!(calc.align(), U32PowerOf2::_1);

        // Add a vec2<f32> field - but with custom min align, which overwrites packed alignment
        let offset3 = calc.extend(8, U32PowerOf2::_8, None, Some(U32PowerOf2::_16), false);
        assert_eq!(offset3, 16);
        // TODO(chronicl) not sure whether the alignment should stay 1 for a packesd struct
        // with custom min align field.
        assert_eq!(calc.align(), U32PowerOf2::_1);
    }

    #[test]
    fn test_layout_calculator_uniform_struct_padding() {
        let mut calc = StructLayoutCalculator::new(Repr::Uniform);

        // Add a nested struct with size 12
        let offset1 = calc.extend(12, U32PowerOf2::_4, None, None, true);
        assert_eq!(offset1, 0);
        // Add another field - should be padded to 16-byte boundary from struct
        let offset2 = calc.extend(4, U32PowerOf2::_4, None, None, false);
        assert_eq!(offset2, 16);

        assert_eq!(calc.align(), U32PowerOf2::_16); // Uniform struct alignment is multiple of 16
        assert_eq!(calc.byte_size(), 32); // Byte size of struct is a multiple of it's align
    }

    #[test]
    fn test_layout_calculator_custom_sizes_and_aligns() {
        let mut calc = StructLayoutCalculator::new(Repr::Storage);

        // Add field with custom minimum size
        let offset1 = calc.extend(4, U32PowerOf2::_4, Some(33), None, false);
        assert_eq!(offset1, 0);
        assert_eq!(calc.byte_size(), 36); // 33 rounded up to multiple of align
        // Add field with custom minimum alignment
        let offset2 = calc.extend(4, U32PowerOf2::_4, None, Some(U32PowerOf2::_16), false);
        assert_eq!(offset2, 48);

        // 33 -> placed at 48 due to 16 align -> 64 size because rounded up to multiple of align
        assert_eq!(calc.byte_size(), 64);
        assert_eq!(calc.align(), U32PowerOf2::_16);
    }

    #[test]
    fn test_layout_calculator_extend_unsized() {
        let mut calc = StructLayoutCalculator::new(Repr::Storage);

        // Add some sized fields first
        calc.extend(4, U32PowerOf2::_4, None, None, false);
        calc.extend(8, U32PowerOf2::_8, None, None, false);
        // Add unsized field
        let (offset, align) = calc.extend_unsized(U32PowerOf2::_4, None);
        assert_eq!(offset, 16);
        assert_eq!(align, U32PowerOf2::_8);
    }

    #[test]
    fn test_sized_field_calculations() {
        // Test custom size
        let field = SizedField::new(
            FieldOptions::new("test_field", None, Some(16)),
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X2)),
        );
        // Vector is 8 bytes, but field has custom min size of 16
        assert_eq!(field.byte_size(Repr::Storage), 16);
        assert_eq!(field.align(Repr::Storage), U32PowerOf2::_8);

        // Test custom alignment
        let field2 = SizedField::new(
            FieldOptions::new("test_field2", Some(U32PowerOf2::_16), None),
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X2)),
        );
        assert_eq!(field2.align(Repr::Storage), U32PowerOf2::_16);
    }

    #[test]
    fn test_runtime_sized_array_field_align() {
        let field = RuntimeSizedArrayField::new(
            "test_array",
            Some(U32PowerOf2::_16),
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X2)),
        );

        // Array has 8-byte alignment, but field has custom min align of 16
        assert_eq!(field.align(Repr::Storage), U32PowerOf2::_16);
        // Custom min align is ignored by packed
        assert_eq!(field.align(Repr::Packed), U32PowerOf2::_1);
    }

    #[test]
    fn test_sized_struct_layout() {
        // Create a struct with mixed field types
        let sized_struct = SizedStruct::new(
            "TestStruct",
            "field1",
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X1)), // 4 bytes, 4-byte aligned
            Repr::Storage,
        )
        .extend(
            "field2",
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X2)), // 8 bytes, 8-byte aligned
        )
        .extend(
            "field3",
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X1)), // 4 bytes, 4-byte aligned
        );

        // Test field offsets
        let mut field_offsets = sized_struct.field_offsets();

        // Field 1: offset 0
        assert_eq!(field_offsets.next(), Some(0));
        // Field 2: offset 8 (aligned to 8-byte boundary)
        assert_eq!(field_offsets.next(), Some(8));
        // Field 3: offset 16 (directly after field 2)
        assert_eq!(field_offsets.next(), Some(16));
        // No more fields
        assert_eq!(field_offsets.next(), None);

        // Test struct size and alignment
        let (size, align) = sized_struct.byte_size_and_align();
        assert_eq!(size, 24); // Round up to 8-byte alignment: round_up(8, 20) = 24
        assert_eq!(align, U32PowerOf2::_8);
    }

    #[test]
    fn test_uniform_struct_alignment() {
        let sized_struct = SizedStruct::new(
            "TestStruct",
            "field1",
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X2)), // 8 bytes, 8-byte aligned
            Repr::Uniform,
        );

        let (size, align) = sized_struct.byte_size_and_align();

        assert_eq!(align, U32PowerOf2::_16); // Alignment adjusted for uniform to multiple of 16
        assert_eq!(size, 16); // Byte size of struct is a multiple of it's alignment
    }

    #[test]
    fn test_unsized_struct_layout() {
        // Test UnsizedStruct with sized fields and a runtime sized array
        let mut unsized_struct = SizedStruct::new(
            "UnsizedStruct",
            "field1",
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X2)), // 4 bytes, 4-byte aligned
            Repr::Storage,
        )
        .extend(
            "field2",
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X1)), // 8 bytes, 8-byte aligned
        )
        .extend_unsized(
            "runtime_array",
            None,
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X1)), // 4 bytes per element
        );

        // Test field offsets
        let mut field_offsets = unsized_struct.field_offsets();
        let sized_offsets: Vec<u64> = field_offsets.sized_field_offsets().collect();
        assert_eq!(sized_offsets, vec![0, 8]); // First field at 0, second at 8

        // Test last field offset and struct alignment
        let (last_offset, struct_align) = field_offsets.last_field_offset_and_struct_align();
        assert_eq!(last_offset, 12); // Runtime array starts at offset 12
        assert_eq!(struct_align, U32PowerOf2::_8); // Struct alignment is 8

        // Test struct alignment method
        assert_eq!(unsized_struct.align(), U32PowerOf2::_8);

        // Test with different repr
        unsized_struct.change_all_repr(Repr::Uniform);
        let mut field_offsets_uniform = unsized_struct.field_offsets();
        let (last_offset_uniform, struct_align_uniform) = field_offsets_uniform.last_field_offset_and_struct_align();
        assert_eq!(last_offset_uniform, 16); // Different offset in uniform, because array's alignment is 16
        assert_eq!(struct_align_uniform, U32PowerOf2::_16); // Uniform struct alignment
    }

    #[test]
    fn test_packed_struct_layout() {
        let sized_struct = SizedStruct::new(
            "TestStruct",
            "field1",
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X3)), // 8 bytes, 16 align (when not packed)
            Repr::Packed,
        )
        .extend(
            "field2",
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X1)), // 4 bytes
        );

        let mut field_offsets = sized_struct.field_offsets();

        // Field 1: offset 0
        assert_eq!(field_offsets.next(), Some(0));
        // Field 2: offset 12, packed directly after field 1, despite 16 alignment of field1, because packed
        assert_eq!(field_offsets.next(), Some(12));

        let (size, align) = sized_struct.byte_size_and_align();
        assert_eq!(size, 16);
        assert_eq!(align, U32PowerOf2::_1);
    }

    #[test]
    fn test_packed_ignores_custom_min_align() {
        let mut calc = StructLayoutCalculator::new(Repr::Packed);

        // Add a u32 field with custom min align of 16
        let offset1 = calc.extend(4, U32PowerOf2::_4, None, Some(U32PowerOf2::_16), false);
        assert_eq!(offset1, 0);
        // Add a vec2<f32> field - should be packed directly after
        let offset2 = calc.extend(8, U32PowerOf2::_8, None, None, false);
        assert_eq!(offset2, 4);

        assert_eq!(calc.byte_size(), 12);
        assert_eq!(calc.align(), U32PowerOf2::_1); // Packed structs always have align of 1

        let s = SizedStruct::new(
            "TestStruct",
            "field1",
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X2)), // 8 bytes, 8-byte aligned
            Repr::Packed,
        )
        .extend(
            FieldOptions::new("field2", Some(U32PowerOf2::_16), None),
            SizedType::Vector(Vector::new(ScalarType::F32, Len::X1)), // 4 bytes, 4-byte aligned
        );

        // The custom min align is ignored in packed structs
        assert_eq!(s.byte_size_and_align().1, U32PowerOf2::_1);
        let mut offsets = s.field_offsets();
        assert_eq!(offsets.next(), Some(0)); // field1 at offset 0
        assert_eq!(offsets.next(), Some(8)); // field2 at offset 8, because min align is ignored
    }
}
