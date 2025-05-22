#![allow(missing_docs)] // TODO(chronicl) remove
//! Everything related to type layouts.



use std::{
    fmt::{Debug, Display, Write},
    hash::Hash,
    marker::PhantomData,
    rc::Rc,
};

use crate::{
    any::U32PowerOf2,
    call_info,
    common::{ignore_eq::IgnoreInEqOrdHash, prettify::set_color},
    ir::{
        self,
        ir_type::{round_up, CanonName, ScalarTypeFp},
        recording::Context,
        Len, SizedType, Type,
    },
};
use cpu_shareable::{LayoutType, Matrix, Vector};
use thiserror::Error;

mod builder;
pub mod cpu_shareable;
mod eq;

pub use builder::*;
pub(crate) use eq::*;

/// The type contained in the bytes of a `TypeLayout`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeLayoutSemantics {
    // TODO(chronicl) consider replacing this scalar type with the host shareable version.
    /// `vec<T, L>`
    Vector(Vector),
    /// special compressed vectors for vertex attribute types
    ///
    /// see the [`crate::packed`] module
    PackedVector(ir::PackedVector),
    /// `mat<T, Cols, Rows>`, first `Len2` is cols, 2nd `Len2` is rows
    Matrix(Matrix),
    /// `Array<T>` and `Array<T, Size<N>>`
    Array(Rc<ElementLayout>, Option<u32>), // not NonZeroU32, since for rust `CpuLayout`s the array size may be 0.
    /// structures which may be empty and may have an unsized last field
    Structure(Rc<StructLayout>),
}

/// The memory layout of a type.
///
/// This models only the layout, not other characteristics of the types.
/// For example an `Atomic<vec<u32, x1>>` is treated like a regular `vec<u32, x1>` layout wise.
///
/// ### Constraint generic
///
/// `TypeLayout` has a generic `T: TypeConstraint`, which guarantees that the layout
/// follows certain layout rules. The following layout rules exist and are captured by
/// the [`Repr`] enum:
/// ```
/// pub enum Repr {
///     Storage, /// wgsl storage address space layout / OpenGL std430
///     Uniform, /// wgsl uniform address space layout / OpenGL std140
///     Packed,  /// packed layout, vertex buffer only
/// }
/// ```
///
/// More information on the exact details of these layout rules is available here
///
/// https://www.w3.org/TR/WGSL/#address-space-layout-constraints
///
/// The following types implementing `TypeConstraint` exist and can be found in [`constraint`]:
///
/// ```
/// struct Plain;   /// May not follow any layout rules.
/// struct Storage; /// Follows `Repr::Storage` layout rules.
/// struct Uniform; /// Follows `Repr::Uniform` layout rules.
/// struct Vertex;  /// Follows either `Repr::Packed` or both
///                 /// `Repr::Storage` and `Repr::Uniform`.
/// ```
/// `Vertex` also guarantees that it is either the layout of a [`VertexAttribute`] or the layout
/// of a struct with fields that are all `VertexAttribute`s, which is why `Repr::Storage`
/// and `Repr::Uniform` are equivalent for it.
///
/// `Storage` and `Uniform` are always based on a [`CpuShareableType`], which can be accessed
/// using [`TypeLayout::cpu_shareable`]. They also guarantee, that all nested `TypeLayout<Plain>`s
/// (for example struct fields) are also based on a `CpuShareableType`, which can only be accessed
/// through `TypeLayout::get_cpu_shareable`.
///
/// TODO(chronicl) make sure these are caught:
/// There is some target language specific constraints that `TypeLayout<T>` does not capture:
/// - `TypeLayout<Uniform>` may be unsized, which is not allowed for wgsl.
/// - `custom_min_align` and `custom_min_size` are not supported for glsl.
///
/// ### Layout comparison
///
/// The `PartialEq + Eq` implementation of `TypeLayout` is designed to answer the question
/// "do these two types have the same layout" so that uploading a type to the gpu
/// will result in no memory errors.
///
/// A layout comparison looks like this:
/// ```
/// assert!(f32::cpu_layout() == vec<f32, x1>::gpu_layout().into_unconstraint());
/// // or, more explicitly
/// assert_eq!(
///     <f32 as CpuLayout>::cpu_layout(),
///     <vec<f32, x1> as GpuLayout>::gpu_layout(),
/// );
/// ```
///
#[derive(Clone)]
pub struct TypeLayout<T: TypeConstraint = constraint::Plain> {
    /// size in bytes (Some), or unsized (None)
    pub byte_size: Option<u64>,
    /// the byte alignment
    ///
    /// top level alignment is not considered relevant in some checks, but relevant in others (vertex array elements)
    pub align: U32PowerOf2,
    /// the type contained in the bytes of this type layout
    pub kind: TypeLayoutSemantics,

    /// Is some for `constraint::Storage` and `constraint::Uniform`. Can be converted to a `StoreType`.
    pub(crate) cpu_shareable: Option<LayoutType>,
    _phantom: PhantomData<T>,
}

// PartialEq, Eq, Hash for TypeLayout
impl<L: TypeConstraint, R: TypeConstraint> PartialEq<TypeLayout<R>> for TypeLayout<L> {
    fn eq(&self, other: &TypeLayout<R>) -> bool { self.byte_size() == other.byte_size() && self.kind == other.kind }
}
impl<T: TypeConstraint> Eq for TypeLayout<T> {}
impl<T: TypeConstraint> Hash for TypeLayout<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.byte_size.hash(state);
        self.kind.hash(state);
    }
}

use constraint::TypeConstraint;
/// Module for all restrictions on `TypeLayout<T: TypeRestriction>`.
pub mod constraint {
    use super::*;

    /// Type restriction of a `TypeLayout<T: TypeRestriction>`. This allows type layouts
    /// to have guarantees based on their type restriction. For example a
    /// `TypeLayout<restriction::Vertex> can be used in vertex buffers. For more
    /// details see the documentation of [`TypeLayout`].
    pub trait TypeConstraint: Clone + PartialEq + Eq {}

    macro_rules! type_restriction {
        ($($constraint:ident),*) => {
            $(
                /// A type restriction for `TypeLayout<T: TypeRestriction>`.
                /// See [`TypeRestriction`] documentation for TODO(chronicl)
                #[derive(Clone, PartialEq, Eq, Hash)]
                pub struct $constraint;
                impl TypeConstraint for $constraint {}
            )*
        };
    }

    type_restriction!(Plain, Storage, Uniform, Packed);

    macro_rules! impl_from_into {
        ($from:ident -> $($into:ident),*) => {
            $(
                impl From<TypeLayout<$from>> for TypeLayout<$into> {
                    fn from(layout: TypeLayout<$from>) -> Self { type_layout_internal::cast_unchecked(layout) }
                }
            )*
        };
    }

    impl_from_into!(Storage -> Plain);
    impl_from_into!(Uniform -> Plain, Storage);
    impl_from_into!(Packed  -> Plain);
}

impl TypeLayout {
    pub(crate) fn new(
        byte_size: Option<u64>,
        byte_align: U32PowerOf2,
        kind: TypeLayoutSemantics,
        hostshareable: Option<LayoutType>,
    ) -> Self {
        TypeLayout {
            byte_size,
            align: byte_align,
            kind,
            cpu_shareable: hostshareable,
            _phantom: PhantomData,
        }
    }
}

/// This module offers helper methods that do not adhere to the restriction of `TypeLayout<T: TypeRestriction>.
/// The caller must uphold these restrictions themselves.
/// The main purpose of this module is to avoid repetition.
pub(in super::super::rust_types) mod type_layout_internal {
    use super::*;

    pub fn cast_unchecked<From: TypeConstraint, Into: TypeConstraint>(layout: TypeLayout<From>) -> TypeLayout<Into> {
        TypeLayout {
            byte_size: layout.byte_size,
            align: layout.align,
            kind: layout.kind,
            cpu_shareable: layout.cpu_shareable,
            _phantom: PhantomData,
        }
    }
}

/// `LayoutCalculator` helps calculate the size, align and the field offsets of a gpu struct.
///
/// If `LayoutCalculator` is created with `packed = true`, provided `field_align`s
/// are ignored and the field is directly after the previous field. However,
/// a `custom_min_align` that is `Some` overwrites the `packedness` of the field.
#[derive(Debug, Clone)]
pub struct LayoutCalculator {
    next_offset_min: u64,
    align: U32PowerOf2,
    packed: bool,
}

impl LayoutCalculator {
    /// Creates a new `LayoutCalculator`, which calculates the size, align and
    /// the field offsets of a gpu struct.
    pub const fn new(packed: bool) -> Self {
        Self {
            next_offset_min: 0,
            align: U32PowerOf2::_1,
            packed,
        }
    }

    /// Extends the layout by a field given it's size and align.
    ///
    /// Returns the field's offset.
    pub const fn extend(
        &mut self,
        field_size: u64,
        field_align: U32PowerOf2,
        custom_min_size: Option<u64>,
        custom_min_align: Option<U32PowerOf2>,
    ) -> u64 {
        let size = FieldLayout::calculate_byte_size(field_size, custom_min_size);
        let align = FieldLayout::calculate_align(field_align, custom_min_align);

        let offset = self.next_field_offset(align, custom_min_align);
        self.next_offset_min = offset + size;
        self.align = self.align.max(align);

        offset
    }

    /// Extends the layout by a field given it's size and align. If the field
    /// is unsized, pass `None` as it's size.
    ///
    /// Returns (byte size, byte align, last field offset).
    ///
    /// `self` is consumed, so that no further fields may be extended, because
    /// only the last field may be unsized.
    pub const fn extend_maybe_unsized(
        mut self,
        field_size: Option<u64>,
        field_align: U32PowerOf2,
        custom_min_size: Option<u64>,
        custom_min_align: Option<U32PowerOf2>,
    ) -> (Option<u64>, U32PowerOf2, u64) {
        if let Some(size) = field_size {
            let offset = self.extend(size, field_align, custom_min_size, custom_min_align);
            (Some(self.byte_size()), self.align(), offset)
        } else {
            let (offset, align) = self.extend_unsized(field_align, custom_min_align);
            (None, align, offset)
        }
    }


    /// Extends the layout by an unsized field given it's align.
    ///
    /// Returns (last field offset, align)
    ///
    /// `self` is consumed, so that no further fields may be extended, because
    /// only the last field may be unsized.
    pub const fn extend_unsized(
        mut self,
        field_align: U32PowerOf2,
        custom_min_align: Option<U32PowerOf2>,
    ) -> (u64, U32PowerOf2) {
        let align = FieldLayout::calculate_align(field_align, custom_min_align);

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
        match (self.packed, field_custom_min_align) {
            (true, None) => self.next_offset_min,
            (true, Some(custom_align)) => round_up(custom_align.as_u64(), self.next_offset_min),
            (false, _) => round_up(field_align.as_u64(), self.next_offset_min),
        }
    }
}

/// a sized or unsized struct type with 0 or more fields
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructLayout {
    pub name: IgnoreInEqOrdHash<CanonName>,
    pub fields: Vec<FieldLayoutWithOffset>,
}

impl StructLayout {
    /// this exists, because if in the future a refactor happens that separates
    /// fields into sized and unsized fields, the intention of this function is
    /// clear
    fn all_fields(&self) -> &[FieldLayoutWithOffset] { &self.fields }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldLayoutWithOffset {
    pub field: FieldLayout,
    pub rel_byte_offset: u64, // this being relative is used in TypeLayout::byte_size
}

impl std::ops::Deref for FieldLayoutWithOffset {
    type Target = FieldLayout;
    fn deref(&self) -> &Self::Target { &self.field }
}

/// Describes the layout of the elements of an array.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ElementLayout {
    /// Stride of the elements
    pub byte_stride: u64,
    /// The elment layout must be constraint::Plain, because it's shared by all constraints.
    // `ElementLayout` could possibly be made generic too, but it would complicate a lot.
    pub ty: TypeLayout<constraint::Plain>,
}

/// the layout rules used when calculating the byte offsets and alignment of a type
#[derive(Debug, Clone, Copy)]
pub enum TypeLayoutRules {
    /// wgsl type layout rules, see https://www.w3.org/TR/WGSL/#memory-layouts
    Wgsl,
    // reprC,
    // Std140,
    // Std430,
    // Scalar,
}

#[allow(missing_docs)]
#[derive(Error, Debug, Clone)]
pub enum TypeLayoutError {
    #[error("An array cannot contain elements of an unsized type {elements}")]
    ArrayOfUnsizedElements { elements: TypeLayout },
    #[error("the type `{0}` has no defined {1:?} layout in shaders")]
    LayoutUndefined(Type, TypeLayoutRules),
    #[error("in `{parent_name}` at field `{field_name}`: {error}")]
    AtField {
        parent_name: CanonName,
        field_name: CanonName,
        error: Rc<TypeLayoutError>,
    },
    #[error("in element of array: {0}")]
    InArrayElement(Rc<TypeLayoutError>),
}

impl<T: TypeConstraint> TypeLayout<T> {
    /// Byte size of the represented type.
    pub fn byte_size(&self) -> Option<u64> { self.byte_size }

    /// Align of the represented type.
    pub fn byte_align(&self) -> U32PowerOf2 { self.align }

    /// Although all TypeLayout<T> always implement Into<TypeLayout<marker::MaybeInvalid>, this method
    /// is offered to avoid having to declare that as a bound when handling generic TypeLayout<T>.
    pub fn into_plain(self) -> TypeLayout<constraint::Plain> { type_layout_internal::cast_unchecked(self) }

    /// Get the `CpuShareableType` this layout is based on.
    ///
    /// This may be `None` for type layouts
    /// that aren't `TypeLayout<Storage>` or `TypeLayout<Uniform>`. Those two also offer direct
    /// access via [`TypeLayout::cpu_shareable`].
    pub fn get_cpu_shareable(&self) -> Option<&LayoutType> { self.cpu_shareable.as_ref() }

    /// a short name for this `TypeLayout`, useful for printing inline
    pub fn short_name(&self) -> String {
        match &self.kind {
            TypeLayoutSemantics::Vector { .. } |
            TypeLayoutSemantics::PackedVector { .. } |
            TypeLayoutSemantics::Matrix { .. } => format!("{}", self),
            TypeLayoutSemantics::Array(element_layout, n) => match n {
                Some(n) => format!("array<{}, {n}>", element_layout.ty.short_name()),
                None => format!("array<{}, runtime-sized>", element_layout.ty.short_name()),
            },
            TypeLayoutSemantics::Structure(s) => s.name.to_string(),
        }
    }

    pub(crate) fn first_line_of_display_with_ellipsis(&self) -> String {
        let string = format!("{}", self);
        string.split_once('\n').map(|(s, _)| format!("{s}…")).unwrap_or(string)
    }

    pub(crate) fn writeln<W: Write>(&self, indent: &str, colored: bool, f: &mut W) -> std::fmt::Result {
        self.write(indent, colored, f)?;
        writeln!(f)
    }

    //TODO(low prio) try to figure out a cleaner way of writing these.
    pub(crate) fn write<W: Write>(&self, indent: &str, colored: bool, f: &mut W) -> std::fmt::Result {
        let tab = "  ";
        let use_256_color_mode = false;
        let color = |f_: &mut W, hex| match colored {
            true => set_color(f_, Some(hex), use_256_color_mode),
            false => Ok(()),
        };
        let reset = |f_: &mut W| match colored {
            true => set_color(f_, None, use_256_color_mode),
            false => Ok(()),
        };

        use TypeLayoutSemantics as Sem;

        match &self.kind {
            Sem::Vector(Vector { len: l, scalar: t }) => match l {
                Len::X1 => write!(f, "{t}")?,
                l => write!(f, "{t}x{}", u64::from(*l))?,
            },
            Sem::PackedVector(c) => write!(f, "{}", c)?,
            Sem::Matrix(m) => write!(f, "{}", ir::SizedType::Matrix(m.columns, m.rows, m.scalar))?,
            Sem::Array(t, n) => {
                let stride = t.byte_stride;
                write!(f, "array<")?;
                t.ty.write(&(indent.to_string() + tab), colored, f)?;
                if let Some(n) = n {
                    write!(f, ", {n}")?;
                }
                write!(f, ">  stride={stride}")?;
            }
            Sem::Structure(s) => {
                writeln!(f, "struct {} {{", s.name)?;
                {
                    let indent = indent.to_string() + tab;
                    for field in &s.fields {
                        let offset = field.rel_byte_offset;
                        let field = &field.field;
                        write!(f, "{indent}{offset:3} {}: ", field.name)?;
                        field.ty.write(&(indent.to_string() + tab), colored, f)?;
                        if let Some(size) = field.ty.byte_size {
                            let size = size.max(field.custom_min_size.unwrap_or(0));
                            write!(f, " size={size}")?;
                        } else {
                            write!(f, " size=?")?;
                        }
                        writeln!(f, ",")?;
                    }
                }
                write!(f, "{indent}}}")?;
                write!(f, " align={}", self.align.as_u64())?;
                if let Some(size) = self.byte_size {
                    write!(f, " size={size}")?;
                } else {
                    write!(f, " size=?")?;
                }
            }
        };
        Ok(())
    }
}

impl TypeLayout {
    pub(crate) fn from_rust_sized<T: Sized>(kind: TypeLayoutSemantics) -> Self {
        Self::new(
            Some(size_of::<T>() as u64),
            // align is always a power of 2:
            // https://doc.rust-lang.org/reference/type-layout.html#r-layout.properties.align
            U32PowerOf2::try_from(align_of::<T>() as u32).unwrap(),
            kind,
            None,
        )
    }

    // TODO(chronicl) this should be removed with improved any api for storage/uniform bindings
    pub(crate) fn from_store_ty(store_type: ir::StoreType) -> Result<Self, cpu_shareable::CpuShareableConversionError> {
        let t: cpu_shareable::LayoutType = store_type.try_into()?;
        Ok(TypeLayout::new_storage_layout_for(t).into())
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldLayout {
    pub name: CanonName,
    // whether size/align is custom doesn't matter for the layout equality.
    pub custom_min_size: IgnoreInEqOrdHash<Option<u64>>,
    pub custom_min_align: IgnoreInEqOrdHash<Option<U32PowerOf2>>,
    /// The fields layout must be constraint::Plain,
    /// because that's the most it can be while supporting all possible constraints.
    pub ty: TypeLayout<constraint::Plain>,
}

impl FieldLayout {
    pub fn new(
        name: CanonName,
        custom_min_size: Option<u64>,
        custom_min_align: Option<U32PowerOf2>,
        ty: TypeLayout<constraint::Plain>,
    ) -> Self {
        Self {
            name,
            custom_min_size: custom_min_size.into(),
            custom_min_align: custom_min_align.into(),
            ty,
        }
    }

    fn byte_size(&self) -> Option<u64> {
        self.ty
            .byte_size()
            .map(|byte_size| Self::calculate_byte_size(byte_size, self.custom_min_size.0))
    }

    /// The alignment of the field with `custom_min_align` taken into account.
    fn byte_align(&self) -> U32PowerOf2 { Self::calculate_align(self.ty.byte_align(), self.custom_min_align.0) }

    const fn calculate_byte_size(byte_size: u64, custom_min_size: Option<u64>) -> u64 {
        // const byte_size.max(custom_min_size.unwrap_or(0))
        if let Some(min_size) = custom_min_size {
            if min_size > byte_size {
                return min_size;
            }
        }
        byte_size
    }

    const fn calculate_align(align: U32PowerOf2, custom_min_align: Option<U32PowerOf2>) -> U32PowerOf2 {
        // const align.max(custom_min_align.unwrap_or(U32PowerOf2::_1) as u32 as u64)
        if let Some(min_align) = custom_min_align {
            align.max(min_align)
        } else {
            align
        }
    }
}

/// Options for the field of a struct.
///
/// If you only want to customize the field's name, you can convert most string types
/// to `FieldOptions` using `Into::into`, but most methods take `impl Into<StructOptions>`,
/// meaning you can just pass the string type directly.
#[derive(Debug, Clone)]
pub struct FieldOptions {
    /// Name of the field
    pub name: CanonName,
    /// Custom minimum align of the field.
    pub custom_min_align: Option<U32PowerOf2>,
    /// Custom mininum size of the field.
    pub custom_min_size: Option<u64>,
}

impl FieldOptions {
    /// Creates new `FieldOptions`.
    ///
    /// If you only want to customize the field's name, you can convert most string types
    /// to `FieldOptions` using `Into::into`, but most methods take `impl Into<StructOptions>`,
    /// meaning you can just pass the string type directly.
    pub fn new(
        name: impl Into<CanonName>,
        custom_min_align: Option<U32PowerOf2>,
        custom_min_size: Option<u64>,
    ) -> Self {
        Self {
            name: name.into(),
            custom_min_align,
            custom_min_size,
        }
    }
}

impl<T: Into<CanonName>> From<T> for FieldOptions {
    fn from(name: T) -> Self { Self::new(name, None, None) }
}

impl<T: TypeConstraint> Display for TypeLayout<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let colored = Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages).unwrap_or(false);
        self.write("", colored, f)
    }
}

impl<T: TypeConstraint> Debug for TypeLayout<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.write("", false, f) }
}
