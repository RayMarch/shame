use std::{
    fmt::{Display, Write},
    num::NonZeroU32,
    rc::Rc,
};

use thiserror::Error;

use crate::{common::proc_macro_reexports::TypeLayoutRules, frontend::rust_types::type_layout::TypeLayout};
use crate::{
    backend::language::Language,
    call_info,
    common::prettify::set_color,
    frontend::{
        any::shared_io::BufferBindingType, encoding::EncodingErrorKind, rust_types::type_layout::LayoutMismatch,
    },
    ir::{
        ir_type::{max_u64_po2_dividing, AccessModeReadable},
        recording::{Context, MemoryRegion},
        AccessMode,
    },
};

use super::{
    align_of_array, round_up, stride_of_array, AddressSpace, BufferBlock, CanonName, Field, SizedField, SizedStruct,
    SizedType, StoreType, Type,
};
use crate::common::integer::IntegerExt;

/// (slightly modified quote from OpenGL 4.6 spec core, section 7.6.2.2 )
/// (replaced "basic machine units" with "bytes")
/// rules for std140:
///
/// 1. If the member is a scalar consuming N bytes, the base alignment is N.
/// 2. If the member is a 2 or 4 component vector with components consuming N bytes the base alignment is 2N or 4N respectively
/// 3. If the member is a 3 component vector with components consuming n bytes, the base alignment is 4N
/// 4. If the member is an array of scalars or vectors, the base alignment and array
///    stride are set to match the base alignment of a single array element according to
///    rules 1. 2. and 3., and rounded up to the base alignment of a `vec4`. The array
///    may have padding at the end; the base offset of the member following the array is
///    rounded up to the next multiple of the base alignment (edit: of the array (this is not explicitly written in the spec, but a best guess based on the end of rule 9))
/// 5. If the member is a column-major matrix with C columns and R rows, the
///    matrix is stored identically to an array of C column vectors with R components each
///    according to rule 4. .
/// 6. If the member is an array of S column-major matrices with C columns and R rows,
///    the matrix is stored identically to a row(=array???) of S * C column vectors with R
///    components each, according to rule 4. .
/// 7. If the member is a row-major matrix with C columns and R rows, the matrix
///    is stored identically to an array of R row vectors with C components each,
///    according to rules 4. .
/// 8. If the member is an array of S row-major matrices with C columns and R rows,
///    the matrix is stored identically to a row(=array???) of S * R row vectors with C
///    components each, according to rule 4. .
/// 9. If the member is a structure, the base alignment of the structure is N, where
///    N is the largest base alignment value of any of its members and rounded
///    up to the base alignment of a `vec4`. the individual members of this
///    sub-structure are then assigned offset by applying this set of rules
///    recursively, where the base offset of the first member of the sub-structure
///    is equal to the aligned offset of the structure.
///    The structure may have padding at the end; the base offset of the member
///    following the sub-structure is rounded up to the next multiple of the
///    base alignment of the structure.
/// 10. If the member ois an array of S structures, the S elements of the array are
///     laid out in order according to rules 9. .
///
/// Shader storage blocks also support the std140 layout qualifier as well as a std430
/// qualifier not supported for uniform blocks. When using the std430 storage layout,
/// shader storage blocks will be laid out in buffer storage identically to uniform
/// and shader storage blocks using the 140 layout except that the base alignment and
/// stride of arrays of scalars and vectors in rule 4 and of structures in rule 9 are
/// not rounded up a multiple of the base alignment of a vec4.
///
/// summary:
///
/// SCALAR AND VECTOR RULES:
/// 1.  base_align_of `ScalarType`s is its byte-size
/// 2.  base_align_of `Vector(len @ (X2 | X4), s)` is `len * base_align_of(s)`
/// 3.  base_align_of `Vector(X3, s)`              is `  4 * base_align_of(s)`
///
/// ARRAY RULES:
/// 4.a(@ std140) base_align_of `Array(e, n)` is `round_up(base_align_of(fvec4), base_align_of(e))`
/// 4.a(@ std430) base_align_of `Array(e, n)` is                                `base_align_of(e)`
/// 4.b stride_of `Array(e, n)` is round_up(base_align_of `Array(e, n)`, size_of(e)) // this is my interpretation of the word "match" in the spec.
/// 4.c base offset of the member after the array is `offset_of(Array(e, n) + n * stride_of(Array(e, n)))`
///   ^ the empty part after the last element's size end that fills up the stride is referred to as "padding"
///
/// COLUMN MAJOR TO ARRAY RULES:
/// 5.        `col-major Matrix(C, R, t)`     stored like `Array(Vector(R, t),   C)`
/// 6.  `Array(col-major Matrix(C, R, t), S)` stored like `Array(Vector(R, t), S*C)`
///
/// ROW MAJOR TO ARRAY RULES:
/// 7.        `row-major Matrix(C, R, t)`     stored like `Array(Vector(C, t),   R)`
/// 8.  `Array(row-major Matrix(C, R, t), S)` stored like `Array(Vector(C, t), S*R)`
///
/// STRUCTURE RULES:
/// 9.a(@std140) base_align_of `Struct(fields)` is `round_up(base_align_of(fvec4), base_align_of(fields.map(base_align_of).max()))`
/// 9.a(@std430) base_align_of `Struct(fields)` is                                `base_align_of(fields.map(base_align_of).max())`
///
/// 9. If the member is a structure, the base alignment of the structure is N, where
///    N is the largest base alignment value of any of its members and rounded
///    up to the base alignment of a `vec4`.
///
/// the individual members of this
///     sub-structure are then assigned offset by applying this set of rules
///     recursively, where the base offset of the first member of the sub-structure
///     is equal to the aligned offset of the (outer)structure.
///
///     The structure may have padding at the end; the base offset of the member
///     following the sub-structure is rounded up to the next multiple of the
///     base alignment of the (outer)structure.
///
/// 10. If the member is an array of S structures, the S elements of the array are
///     laid out in order according to rules 9. .
///
/// - storage blocks support std140 and std430
/// - uniform blocks support std140 only
///

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Std {
    _140,
    _430,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgslBufferLayout {
    UniformAddressSpace,
    StorageAddressSpace,
}

#[derive(Debug, Clone, Copy)]
pub enum LayoutConstraints {
    OpenGL(Std),
    Wgsl(WgslBufferLayout),
}

impl Display for LayoutConstraints {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            LayoutConstraints::OpenGL(std) => match std {
                Std::_140 => "std140",
                Std::_430 => "std430",
            },
            LayoutConstraints::Wgsl(w) => match w {
                WgslBufferLayout::UniformAddressSpace => "uniform address-space",
                WgslBufferLayout::StorageAddressSpace => "storage address-space",
            },
        })
    }
}

impl LayoutConstraints {
    fn more_info_at(&self) -> &'static str {
        match self {
            LayoutConstraints::OpenGL(_) => {
                "section 7.6.2.2 in https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf"
            }
            LayoutConstraints::Wgsl(_) => "https://www.w3.org/TR/WGSL/#memory-layouts",
        }
    }
}

pub fn get_type_for_buffer_binding_type(
    ty: &StoreType,
    binding_type: BufferBindingType,
    as_ref: bool,
    ctx: &Context,
) -> Result<Type, EncodingErrorKind> {
    use AccessModeReadable as AM;
    use AddressSpace as AS;

    let lang = ctx.settings().lang;

    let access = match binding_type {
        BufferBindingType::Uniform => AccessMode::Read,
        BufferBindingType::Storage(am) => am.into(),
    };

    if !ty.is_host_shareable() {
        return Err(LayoutError::NotHostShareable(ty.clone()).into());
    }

    let result = match as_ref {
        true => {
            // `binding_ty` must be `BindingType::Buffer`
            Type::Ref(
                MemoryRegion::new(call_info!(), ty.clone(), None, None, access, AddressSpace::Storage)?,
                ty.clone(),
                access,
            )
        }
        false => {
            // `binding_ty` must be a uniform or read-only storage buffer, `ty` must be constructible
            match binding_type {
                BufferBindingType::Uniform => Ok(()),
                BufferBindingType::Storage(AccessModeReadable::Read) => Ok(()),
                BufferBindingType::Storage(AccessModeReadable::ReadWrite) => {
                    Err(LayoutError::NonRefBufferRequiresReadOnlyAndConstructible)
                }
            }?;
            if !ty.is_constructible() {
                return Err(LayoutError::NonRefBufferRequiresReadOnlyAndConstructible.into());
            }
            Type::Store(ty.clone())
        }
    };

    let constraint = match lang {
        Language::Wgsl => LayoutConstraints::Wgsl(match binding_type {
            BufferBindingType::Uniform => WgslBufferLayout::UniformAddressSpace,
            BufferBindingType::Storage(_) => WgslBufferLayout::StorageAddressSpace,
        }),
        // Language::Glsl => LayoutConstraints::OpenGL(match binding_type {
        //     BufferBindingType::Uniform => Std::_140,
        //     BufferBindingType::Storage(_) => Std::_430,
        // }),
    };

    check_layout(
        &LayoutErrorContext {
            binding_type,
            expected_constraints: constraint,
            top_level_type: ty.clone(),
            use_color: ctx.settings().colored_error_messages,
        },
        ty,
    )?;
    Ok(result)
}

pub fn check_layout(ctx: &LayoutErrorContext, ty: &StoreType) -> Result<(), LayoutError> {
    if !ty.is_creation_fixed_footprint() {
        match ctx.expected_constraints {
            LayoutConstraints::OpenGL(std) => Ok(()),
            LayoutConstraints::Wgsl(l) => match l {
                WgslBufferLayout::UniformAddressSpace => Err(LayoutError::UniformBufferMustBeSized("wgsl", ty.clone())),
                WgslBufferLayout::StorageAddressSpace => Ok(()),
            },
        }?;
        // WGSL does not allow uniform buffers to have unsized types
    }
    match ty {
        StoreType::Sized(t) => check_sized_type_layout(ctx, t),
        StoreType::Handle(_) => Err(LayoutError::NotHostShareable(ty.clone())),
        StoreType::RuntimeSizedArray(e) => check_array_layout(ctx, e),
        StoreType::BufferBlock(s) => check_structure_layout(ctx, &LayoutStructureKind::BufferBlock(s.clone())),
    }
}

pub fn check_sized_type_layout(ctx: &LayoutErrorContext, ty: &SizedType) -> Result<(), LayoutError> {
    match &ty {
        SizedType::Vector(_, _) => Ok(()),
        SizedType::Matrix(_, _, _) => Ok(()),
        SizedType::Atomic(_) => Ok(()),
        SizedType::Structure(s) => check_structure_layout(ctx, &LayoutStructureKind::Structure(s.clone())),
        SizedType::Array(e, n) => {
            let expected_align = match ctx.expected_constraints {
                LayoutConstraints::OpenGL(std) => match std {
                    Std::_140 => round_up(16, align_of_array(e)),
                    Std::_430 => align_of_array(e),
                },
                LayoutConstraints::Wgsl(wbl) => match wbl {
                    WgslBufferLayout::UniformAddressSpace => round_up(16, align_of_array(e)),
                    WgslBufferLayout::StorageAddressSpace => align_of_array(e),
                },
            };
            let actual_align = align_of_array(e);
            if actual_align != expected_align {
                Err(LayoutError::ArrayAlignmentError(ArrayAlignmentError {
                    ctx: ctx.clone(),
                    expected: expected_align,
                    actual: actual_align,
                    element_ty: (**e).clone(),
                }))
            } else {
                check_array_layout(ctx, e)
            }
        }
    }
}

pub fn check_array_layout(ctx: &LayoutErrorContext, elem: &SizedType) -> Result<(), LayoutError> {
    // wgsl spec: Arrays of element type `elem` must have an element
    // stride that is a multiple of the RequiredAlignOf(elem, addr_space)
    let actual_stride = stride_of_array(elem);
    match ctx.expected_constraints {
        LayoutConstraints::OpenGL(std) => {
            // using wording of the opengl spec
            let base_align_of_array = match std {
                Std::_140 => round_up(16, elem.align()),
                Std::_430 => elem.align(),
            };
            let expected_stride = round_up(base_align_of_array, elem.byte_size());
            if actual_stride != expected_stride {
                return Err(LayoutError::ArrayStrideError(ArrayStrideError {
                    ctx: ctx.clone(),
                    expected: expected_stride,
                    actual: actual_stride,
                    element_ty: elem.clone(),
                }));
            }
        }
        LayoutConstraints::Wgsl(wbl) => {
            let required_element_align = match wbl {
                WgslBufferLayout::UniformAddressSpace => round_up(16, align_of_array(elem)),
                WgslBufferLayout::StorageAddressSpace => align_of_array(elem),
            };
            if !required_element_align.divides(actual_stride) {
                return Err(LayoutError::ArrayStrideAlignmentError(ArrayStrideAlignmentError {
                    ctx: ctx.clone(),
                    expected_align: required_element_align,
                    actual_stride,
                    element_ty: elem.clone(),
                }));
            }
        }
    };
    check_sized_type_layout(ctx, elem)?;
    Ok(())
}

pub fn check_structure_layout(ctx: &LayoutErrorContext, s: &LayoutStructureKind) -> Result<(), LayoutError> {
    let structure_name = match s {
        LayoutStructureKind::Structure(s) => s.name(),
        LayoutStructureKind::BufferBlock(s) => s.name(),
    };
    let (sized_fields, last_field) = &match s {
        LayoutStructureKind::Structure(s) => (s.sized_fields(), &None),
        LayoutStructureKind::BufferBlock(s) => (s.sized_fields(), s.last_unsized_field()),
    };

    let mut offset = 0;
    for field in sized_fields.iter() {
        // return errors if custom align or size are used in OpenGL constraints.
        // warning: removing this check will make the surrounding code wrong.
        // for example: the OpenGL spec has a wording which defines std140/std430
        // offsets of elements after arrays directly via their stride.
        // this means even if GLSL supports custom size and align in the future,
        // the wording of the std140 definition of that situation would need to
        // be checked for changes, which would likely justify a rewrite of this
        // entire module.
        match ctx.expected_constraints {
            LayoutConstraints::OpenGL(_) => {
                if let Some(custom_align) = field.custom_min_align() {
                    return Err(LayoutError::LayoutConstriantsDoNotSupportCustomFieldAlign {
                        ctx: ctx.clone(),
                        struct_or_block_name: structure_name.clone(),
                        field_name: field.name().clone(),
                        custom_align: u64::from(custom_align),
                    });
                }
                if let Some(custom_size) = field.custom_min_size() {
                    return Err(LayoutError::LayoutConstriantsDoNotSupportCustomFieldSize {
                        ctx: ctx.clone(),
                        struct_or_block_name: structure_name.clone(),
                        field_name: field.name().clone(),
                        custom_size,
                    });
                }
            }
            LayoutConstraints::Wgsl(_) => (),
        }

        // the align and size which takes custom user attributes for align and size into account
        let (align, size) = (field.align(), field.byte_size());
        // the regular align and size of field.ty
        let (ty_align, ty_size) = (field.ty().align(), field.ty().byte_size());
        offset = round_up(field.align(), offset);

        let required_align_of_field = match field.ty() {
            SizedType::Vector(_, _) | SizedType::Matrix(_, _, _) | SizedType::Atomic(_) => ty_align,
            SizedType::Structure(_) | SizedType::Array(_, _) => match ctx.expected_constraints {
                LayoutConstraints::OpenGL(std) => match std {
                    Std::_140 => round_up(16, ty_align),
                    Std::_430 => ty_align,
                },
                LayoutConstraints::Wgsl(wbl) => match wbl {
                    WgslBufferLayout::UniformAddressSpace => round_up(16, ty_align),
                    WgslBufferLayout::StorageAddressSpace => ty_align,
                },
            },
        };

        if !required_align_of_field.divides(offset) {
            return Err(LayoutError::Structure(StructureLayoutError {
                context: ctx.clone(),
                structure: s.clone(),
                field_name: field.name().clone(),
                actual_offset: offset,
                expected_alignment: required_align_of_field,
            }));
        }

        check_sized_type_layout(ctx, field.ty())?;

        offset += field.byte_size()
    }

    if let Some(last_field) = last_field {
        let ty_align = align_of_array(last_field.element_ty());
        let ty = StoreType::RuntimeSizedArray(last_field.element_ty().clone());

        // TODO(low prio) refactor this function so that theres no code duplication here
        let required_align_of_array = match ctx.expected_constraints {
            LayoutConstraints::OpenGL(std) => match std {
                Std::_140 => round_up(16, ty_align),
                Std::_430 => ty_align,
            },
            LayoutConstraints::Wgsl(wbl) => match wbl {
                WgslBufferLayout::UniformAddressSpace => round_up(16, ty_align),
                WgslBufferLayout::StorageAddressSpace => ty_align,
            },
        };

        if !required_align_of_array.divides(offset) {
            return Err(LayoutError::Structure(StructureLayoutError {
                context: ctx.clone(),
                structure: s.clone(),
                field_name: last_field.name().clone(),
                actual_offset: offset,
                expected_alignment: required_align_of_array,
            }));
        }
        check_layout(ctx, &ty)?;
    }
    Ok(())
}

#[derive(Error, Debug, Clone)]
pub enum LayoutError {
    #[error("array elements may not be runtime-sized types. found array of: {0}")]
    ArrayElementsAreUnsized(TypeLayout),
    #[error("{0}")]
    Structure(#[from] StructureLayoutError),
    #[error(
        "The size of `{1}` on the gpu is now known at compile time. `{0}` \
    requires that the size of uniform buffers on the gpu is known at compile time."
    )]
    UniformBufferMustBeSized(&'static str, StoreType),
    #[error("{0}")]
    ArrayAlignmentError(ArrayAlignmentError),
    #[error("{0}")]
    ArrayStrideError(ArrayStrideError),
    #[error("{0}")]
    ArrayStrideAlignmentError(ArrayStrideAlignmentError),
    #[error("{} layout constraints do not allow custom struct field byte-alignments. \
Type `{}` contains type `{struct_or_block_name}` which has a custom byte-alignment of {custom_align} at field `{field_name}`.", ctx.expected_constraints, ctx.top_level_type)]
    LayoutConstriantsDoNotSupportCustomFieldAlign {
        ctx: LayoutErrorContext,
        struct_or_block_name: CanonName,
        field_name: CanonName,
        custom_align: u64,
    },
    #[error("{} layout constraints do not allow custom struct field byte-sizes. \
    Type `{}` contains type `{struct_or_block_name}` which has a custom byte-size of {custom_size} at field `{field_name}`.", ctx.expected_constraints, ctx.top_level_type)]
    LayoutConstriantsDoNotSupportCustomFieldSize {
        ctx: LayoutErrorContext,
        struct_or_block_name: CanonName,
        field_name: CanonName,
        custom_size: u64,
    },
    #[error("a non-reference buffer (non `BufferRef`) must be both read-only and constructible")]
    NonRefBufferRequiresReadOnlyAndConstructible,
    #[error(
        "type {0} does not match the requirements for host-shareable types. See https://www.w3.org/TR/WGSL/#host-shareable-types (In most cases, this is caused by the type containing booleans)"
    )]
    NotHostShareable(StoreType),
    #[error("custom alignment of {custom} is too small. `{ty}` must have an alignment of at least {required}")]
    CustomAlignmentTooSmall { custom: u64, required: u64, ty: StoreType },
    #[error("custom size of {custom} is too small. `{ty}` must have a size of at least {required}")]
    CustomSizeTooSmall { custom: u64, required: u64, ty: Type },
    #[error("memory layout mismatch:\n{0}\n{}", if let Some(comment) = .1 {comment.as_str()} else {""})]
    LayoutMismatch(LayoutMismatch, Option<String>),
    #[error("runtime-sized type {name} cannot be element in an array buffer")]
    UnsizedStride { name: String },
    #[error(
        "stride mismatch:\n{cpu_name}: {cpu_stride} bytes offset between elements,\n{gpu_name}: {gpu_stride} bytes offset between elements"
    )]
    StrideMismatch {
        cpu_name: String,
        cpu_stride: u64,
        gpu_name: String,
        gpu_stride: u64,
    },
}

#[allow(missing_docs)]
#[derive(Error, Debug, Clone)]
pub struct ArrayStrideAlignmentError {
    ctx: LayoutErrorContext,
    expected_align: u64,
    actual_stride: u64,
    element_ty: SizedType,
}

impl Display for ArrayStrideAlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "array elements within type `{}` do not satisfy {} layout requirements.",
            self.ctx.top_level_type, self.ctx.expected_constraints
        );
        let expected_align = self.expected_align;
        let actual_stride = self.actual_stride;
        writeln!(
            f,
            "The array with `{}` elements requires that every element is {expected_align}-byte aligned, but the array has a stride of {actual_stride} bytes, which means subsequent elements are not {expected_align}-byte aligned.",
            self.element_ty
        );
        if let Ok(layout) = TypeLayout::from_store_ty(self.ctx.top_level_type.clone()) {
            writeln!(f, "The full layout of `{}` is:", self.ctx.top_level_type);
            layout.write("", self.ctx.use_color, f)?;
            writeln!(f);
        };
        writeln!(
            f,
            "\nfor more information on the layout rules, see {}",
            self.ctx.expected_constraints.more_info_at()
        )?;
        Ok(())
    }
}

#[allow(missing_docs)]
#[derive(Error, Debug, Clone)]
pub struct ArrayStrideError {
    ctx: LayoutErrorContext,
    expected: u64,
    actual: u64,
    element_ty: SizedType,
}

impl Display for ArrayStrideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "array elements within type `{}` do not satisfy {} layout requirements.",
            self.ctx.top_level_type, self.ctx.expected_constraints
        );
        writeln!(
            f,
            "The array with `{}` elements requires stride {}, but has stride {}.",
            self.element_ty, self.expected, self.actual
        );
        if let Ok(layout) = TypeLayout::from_store_ty(self.ctx.top_level_type.clone()) {
            writeln!(f, "The full layout of `{}` is:", self.ctx.top_level_type);
            layout.write("", self.ctx.use_color, f)?;
            writeln!(f);
        };
        writeln!(
            f,
            "\nfor more information on the layout rules, see {}",
            self.ctx.expected_constraints.more_info_at()
        )?;
        Ok(())
    }
}

#[allow(missing_docs)]
#[derive(Error, Debug, Clone)]
pub struct ArrayAlignmentError {
    ctx: LayoutErrorContext,
    expected: u64,
    actual: u64,
    element_ty: SizedType,
}

impl Display for ArrayAlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "array elements within type `{}` do not satisfy {} layout requirements.",
            self.ctx.top_level_type, self.ctx.expected_constraints
        );
        writeln!(
            f,
            "The array with `{}` elements requires alignment {}, but has alignment {}.",
            self.element_ty, self.expected, self.actual
        );
        if let Ok(layout) = TypeLayout::from_store_ty(self.ctx.top_level_type.clone()) {
            writeln!(f, "The full layout of `{}` is:", self.ctx.top_level_type);
            layout.write("", self.ctx.use_color, f)?;
            writeln!(f);
        };
        writeln!(
            f,
            "\nfor more information on the layout rules, see {}",
            self.ctx.expected_constraints.more_info_at()
        )?;
        Ok(())
    }
}



#[derive(Debug, Clone)]
pub struct LayoutErrorContext {
    pub binding_type: BufferBindingType,
    pub expected_constraints: LayoutConstraints,
    pub top_level_type: StoreType,
    /// whether the error message should be output using console colors
    pub use_color: bool,
}

#[derive(Debug, Clone)]
//TODO(release) this type is obsolete. SizedStruct and BufferBlock now both Deref to `Struct`, use that instead
pub enum LayoutStructureKind {
    Structure(SizedStruct),
    BufferBlock(BufferBlock),
}

#[derive(Debug, Clone)]
pub struct StructureLayoutError {
    pub context: LayoutErrorContext,
    pub structure: LayoutStructureKind,
    pub field_name: CanonName,
    pub actual_offset: u64,
    pub expected_alignment: u64,
}

impl std::error::Error for StructureLayoutError {}

impl Display for StructureLayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let custom_align_or_size_supported = match self.context.expected_constraints {
            LayoutConstraints::OpenGL(_) => false,
            LayoutConstraints::Wgsl(_) => true,
        };

        let structure_name: &CanonName = match &self.structure {
            LayoutStructureKind::Structure(s) => s.name(),
            LayoutStructureKind::BufferBlock(s) => s.name(),
        };

        let structure_def_location = Context::try_with(call_info!(), |ctx| {
            let s = match &self.structure {
                LayoutStructureKind::Structure(s) => &**s,
                LayoutStructureKind::BufferBlock(s) => &**s,
            };
            ctx.struct_registry().get(s).map(|def| def.call_info())
        })
        .flatten();

        let top_level_type = &self.context.top_level_type;
        let binding_type_str = match self.context.binding_type {
            BufferBindingType::Storage(_) => "storage",
            BufferBindingType::Uniform => "uniform",
        };
        writeln!(
            f,
            "the type `{top_level_type}` cannot be used as a {binding_type_str} buffer binding."
        )?;
        let ((constraints, short_summary), link) = match self.context.expected_constraints {
            LayoutConstraints::OpenGL(std) => (
                match std {
                    Std::_140 => (
                        "std140",
                        Some(
                            "In std140 buffers the alignment of structs, arrays and array elements must be at least 16.",
                        ),
                    ),
                    Std::_430 => ("std430", None),
                },
                "https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout",
            ),
            LayoutConstraints::Wgsl(x) => (
                match x {
                    WgslBufferLayout::UniformAddressSpace => (
                        "uniform address space",
                        Some(
                            "In the uniform address space, structs, arrays and array elements must be at least 16 byte aligned.",
                        ),
                    ),
                    WgslBufferLayout::StorageAddressSpace => ("storage address space", None),
                },
                "https://www.w3.org/TR/WGSL/#address-space-layout-constraints",
            ),
        };
        let actual_align = max_u64_po2_dividing(self.actual_offset);
        let nested = self.context.top_level_type.to_string() != **structure_name;
        if nested {
            write!(f, "It contains a struct `{}`, which", structure_name)?;
        } else {
            write!(f, "Struct `{}`", structure_name)?;
        }
        writeln!(f, " does not satisfy the {constraints} memory layout requirements.",)?;
        writeln!(f)?;
        if let Some(call_info) = structure_def_location {
            writeln!(f, "Definition at {call_info}")?;
        }
        write_struct_layout(&self.structure, self.context.use_color, Some(&self.field_name), f)?;

        writeln!(f)?;
        set_color(f, Some("#508EE3"), false)?;
        writeln!(
            f,
            "Field `{}` needs to be {} byte aligned, but has a byte-offset of {} which is only {actual_align} byte aligned.",
            self.field_name, self.expected_alignment, self.actual_offset
        )?;
        set_color(f, None, false)?;
        writeln!(f)?;

        writeln!(f, "Potential solutions include:");
        writeln!(
            f,
            "- add an #[align({})] attribute to the definition of `{}` (not supported by OpenGL/GLSL pipelines)",
            self.expected_alignment, self.field_name
        );
        writeln!(f, "- use a storage binding instead of a uniform binding");
        writeln!(
            f,
            "- increase the offset of `{}` until it is divisible by {} by making previous fields larger or adding fields before it",
            self.field_name, self.expected_alignment
        );
        writeln!(f)?;


        if let Some(summary) = short_summary {
            writeln!(f, "{summary}");
        }
        //writeln!(f, "Address space alignment restrictions enable use of more efficient hardware instructions for accessing the values, or satisfy more restrictive hardware requirements.")?;
        writeln!(f, "More info about the {constraints} layout can be found at {link}")?;
        Ok(())
    }
}

fn write_struct_layout<F>(
    s: &LayoutStructureKind,
    colored: bool,
    highlight_field: Option<&str>,
    f: &mut F,
) -> std::fmt::Result
where
    F: Write,
{
    let use_256_color_mode = false;
    let color = |f_: &mut F, hex| match colored {
        true => set_color(f_, Some(hex), use_256_color_mode),
        false => Ok(()),
    };
    let reset = |f_: &mut F| match colored {
        true => set_color(f_, None, use_256_color_mode),
        false => Ok(()),
    };

    let structure_name = match s {
        LayoutStructureKind::Structure(s) => s.name(),
        LayoutStructureKind::BufferBlock(s) => s.name(),
    };

    let (sized_fields, last_field) = match s {
        LayoutStructureKind::Structure(s) => (s.sized_fields(), &None),
        LayoutStructureKind::BufferBlock(s) => (s.sized_fields(), s.last_unsized_field()),
    };

    let indent = "  ";
    let field_decl_line = |field: &SizedField| format!("{indent}{}: {},", field.name(), field.ty());
    let header = format!("struct {} {{", structure_name);
    let table_start_column = 1 + sized_fields
        .iter()
        .map(field_decl_line)
        .map(|s| s.len())
        .max()
        .unwrap_or(0)
        .max(header.chars().count());
    f.write_str(&header)?;
    for i in header.len()..table_start_column {
        f.write_char(' ')?
    }
    writeln!(f, "offset align size")?;
    let mut offset = 0;
    for field in sized_fields.iter() {
        if Some(&**field.name()) == highlight_field {
            color(f, "#508EE3")?;
        }
        let (align, size) = (field.align(), field.byte_size());
        offset = round_up(field.align(), offset);
        let decl_line = field_decl_line(field);
        f.write_str(&decl_line)?;
        // write spaces to table on the right
        for i in decl_line.len()..table_start_column {
            f.write_char(' ')?
        }
        writeln!(f, "{:6} {:5} {:4}", offset, align, size)?;
        if Some(&**field.name()) == highlight_field {
            reset(f);
        }
        offset += field.byte_size()
    }
    if let Some(last_field) = last_field {
        offset = round_up(last_field.align(), offset);

        let decl_line = format!(
            "{indent}{}: {},",
            last_field.name(),
            StoreType::RuntimeSizedArray(last_field.element_ty().clone())
        );
        f.write_str(&decl_line)?;
        // write spaces to table on the right
        for i in decl_line.len()..table_start_column {
            f.write_char(' ')?
        }
        write!(f, "{:6} {:5}", offset, last_field.align())?;
    }
    writeln!(f, "}}")?;
    Ok(())
}
