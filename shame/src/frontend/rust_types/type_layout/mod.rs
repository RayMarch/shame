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
        ir_type::{round_up, CanonName},
        recording::Context,
        Len,
    },
};
use layoutable::{
    align_size::{LayoutCalculator, PACKED_ALIGN},
    LayoutableType, Matrix, Vector,
};

pub(crate) mod construction;
pub(crate) mod eq;
pub(crate) mod layoutable;

/// The type contained in the bytes of a `TypeLayout`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeLayoutSemantics {
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
/// ### Generic
///
/// `TypeLayout` has a generic `T: TypeRepr`, which is used to statically guarantee that
/// it follows specific layout rules.
///
/// The following types implementing `TypeRepr` exist and can be found in [`shame::any::repr`]:
///
/// ```
/// struct Plain;   /// May not follow any layout rules. This is the default.
/// struct Storage; /// wgsl storage address space layout / OpenGL std430
/// struct Uniform; /// wgsl uniform address space layout / OpenGL std140
/// struct Packed;  /// Packed layout
/// ```
///
/// More information on the exact details of these layout rules is available here
///
/// https://www.w3.org/TR/WGSL/#address-space-layout-constraints
///
/// The following methods exist for creating new type layouts based on a [`LayoutableType`]
/// ```
/// let layout_type: LayoutableType = f32x1::layoutable_type();
/// let _ = TypeLayout::new_layout_for(layout_type);
/// let _ = TypeLayout::new_storage_layout_for(layout_type);
/// let _ = TypeLayout::new_uniform_layout_for(layout_type);
/// let _ = TypeLayout::new_packed_layout_for(layout_type);
/// ```
///
/// Non-`Plain` layouts are always based on a [`LayoutableType`], which can be accessed
/// using [`TypeLayout::layoutable_type`]. They also guarantee, that all nested `TypeLayout<Plain>`s
/// (for example struct fields) are also based on a `LayoutableType`, which can only be accessed
/// through `TypeLayout::get_layoutable_type`.
///
/// ### Layout comparison
///
/// The `PartialEq + Eq` implementation of `TypeLayout` is designed to answer the question
/// "do these two types have the same layout" so that uploading a type to the gpu
/// will result in no memory errors.
///
/// a layout comparison looks like this:
/// ```
/// assert!(f32::cpu_layout() == vec<f32, x1>::gpu_layout());
/// // or, more explicitly
/// assert_eq!(
///     <f32 as CpuLayout>::cpu_layout(),
///     <vec<f32, x1> as GpuLayout>::gpu_layout(),
/// );
/// ```
#[derive(Clone)]
pub struct TypeLayout {
    /// size in bytes (Some), or unsized (None)
    pub byte_size: Option<u64>,
    /// the byte alignment
    ///
    /// top level alignment is not considered relevant in some checks, but relevant in others (vertex array elements)
    pub align: U32PowerOf2,
    /// the type contained in the bytes of this type layout
    pub kind: TypeLayoutSemantics,
}

// PartialEq, Eq, Hash for TypeLayout
impl PartialEq<TypeLayout> for TypeLayout {
    fn eq(&self, other: &TypeLayout) -> bool { self.byte_size() == other.byte_size() && self.kind == other.kind }
}
impl Eq for TypeLayout {}
impl Hash for TypeLayout {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.byte_size.hash(state);
        self.kind.hash(state);
    }
}

/// TODO(chronicl)
#[derive(Debug, Clone)]
pub struct GpuTypeLayout<T: TypeRepr = repr::Storage> {
    ty: LayoutableType,
    _repr: PhantomData<T>,
}

impl<T: TypeRepr> GpuTypeLayout<T> {
    /// TODO(chronicl)
    pub fn layout(&self) -> TypeLayout { TypeLayout::new_layout_for(&self.ty, T::REPR) }
    /// TODO(chronicl)
    pub fn layoutable_type(&self) -> &LayoutableType { &self.ty }
}

use repr::TypeReprStorageOrPacked;
pub use repr::{TypeRepr, Repr};
/// Module for all restrictions on `TypeLayout<T: TypeRestriction>`.
pub mod repr {
    use super::*;

    /// Type representation used by `TypeLayout<T: TypeRepr>`. This provides guarantees
    /// about the alignment rules that the type layout adheres to.
    /// See [`TypeLayout`] documentation for more details.
    pub trait TypeRepr: Clone + PartialEq + Eq {
        /// The corresponding enum variant of `Repr`.
        const REPR: Repr;
    }
    /// TODO(chronicl)
    pub trait TypeReprStorageOrPacked: TypeRepr {}
    impl TypeReprStorageOrPacked for Storage {}
    impl TypeReprStorageOrPacked for Packed {}

    /// Enum of layout rules.
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

    impl Repr {
        /// True if `Repr::Storage`
        pub const fn is_storage(self) -> bool { matches!(self, Repr::Storage) }
        /// True if `Repr::Uniform`
        pub const fn is_uniform(self) -> bool { matches!(self, Repr::Uniform) }
        /// True if `Repr::Packed`
        pub const fn is_packed(self) -> bool { matches!(self, Repr::Packed) }
    }

    impl std::fmt::Display for Repr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Repr::Storage => write!(f, "storage"),
                Repr::Uniform => write!(f, "uniform"),
                Repr::Packed => write!(f, "packed"),
            }
        }
    }

    macro_rules! type_repr {
        ($($repr:ident),*) => {
            $(
                /// A type representation used by `TypeLayout<T: TypeRepr>`.
                /// See [`TypeLayout`] documentation for more details.
                #[derive(Clone, PartialEq, Eq, Hash)]
                pub struct $repr;
                impl TypeRepr for $repr {
                    const REPR: Repr = Repr::$repr;
                }
            )*
        };
    }
    type_repr!(Storage, Uniform, Packed);
}

impl TypeLayout {
    pub(crate) fn new(byte_size: Option<u64>, byte_align: U32PowerOf2, kind: TypeLayoutSemantics) -> Self {
        TypeLayout {
            byte_size,
            align: byte_align,
            kind,
        }
    }
}

/// a sized or unsized struct type with 0 or more fields
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructLayout {
    /// The canonical name of the structure type, ignored in equality/hash comparisons
    pub name: IgnoreInEqOrdHash<CanonName>,
    /// The fields of the structure with their memory offsets
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
    /// The layout information for the field
    pub field: FieldLayout,
    /// The relative byte offset of this field from the start of its containing structure
    pub rel_byte_offset: u64,
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
    /// The type layout of each element in the array.
    ///
    /// The element layout must be constraint::Plain because it's shared by all constraints.
    /// `ElementLayout` could possibly be made generic too, but it would complicate a lot.
    pub ty: TypeLayout,
}

impl TypeLayout {
    /// Returns the byte size of the represented type.
    ///
    /// For sized types, this returns Some(size), while for unsized types
    /// (like runtime-sized arrays), this returns None.
    pub fn byte_size(&self) -> Option<u64> { self.byte_size }

    /// Returns the alignment requirement of the represented type.
    pub fn align(&self) -> U32PowerOf2 { self.align }

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
        )
    }

    // TODO(chronicl) this should be removed with improved any api for storage/uniform bindings
    pub(crate) fn from_store_ty(
        store_type: ir::StoreType,
    ) -> Result<Self, layoutable::ir_compat::LayoutableConversionError> {
        let t: layoutable::LayoutableType = store_type.try_into()?;
        Ok(TypeLayout::new_layout_for(&t, Repr::Storage).into())
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldLayout {
    pub name: CanonName,
    pub ty: TypeLayout,
}

impl FieldLayout {
    fn byte_size(&self) -> Option<u64> { self.ty.byte_size() }

    /// The alignment of the field with `custom_min_align` taken into account.
    fn align(&self) -> U32PowerOf2 { self.ty.align() }
}

// TODO(chronicl) move into layoutable/builder.rs
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

impl Display for TypeLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let colored = Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages).unwrap_or(false);
        self.write("", colored, f)
    }
}

impl Debug for TypeLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.write("", false, f) }
}
