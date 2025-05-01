use super::{
    error::{WgslError, WgslErrorLevel},
    write_texture_format::write_texture_format,
    WgslContext, WgslErrorKind,
};
use crate::ir::{
    self,
    ir_type::{ChannelFormatShaderType, SamplesPerPixel, TextureSampleUsageType, TextureShape},
    AccessMode,
};
use crate::{
    backend::code_write_buf::CodeWriteSpan,
    frontend::any::shared_io::SamplingMethod,
    ir::{
        ir_type::{HandleType, ScalarTypeInteger},
        recording::CallInfo,
        AddressSpace, Len, ScalarType, SizedType, StoreType, Type,
    },
};
use core::fmt;
use std::fmt::Write;

const WRITE_UNWRITEABLE_TYPES: bool = false;

pub(super) fn write_type(
    code: &mut CodeWriteSpan,
    call_info: CallInfo,
    ty: &Type,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    match ty {
        Type::Unit => {
            if WRITE_UNWRITEABLE_TYPES {
                write!(code, "💥unit")?;
                Ok(())
            } else {
                Err(WgslErrorKind::TypeMayNotAppearInWrittenForm(ty.clone())
                    .at_level(call_info, WgslErrorLevel::InternalPleaseReport))
            }
        }
        Type::Ref(a, sty, am) => {
            if WRITE_UNWRITEABLE_TYPES {
                write_memory_view_type(code, MemoryViewKind::Ref, a.address_space, sty, *am, call_info, ctx)
            } else {
                Err(WgslErrorKind::TypeMayNotAppearInWrittenForm(ty.clone())
                    .at_level(call_info, WgslErrorLevel::InternalPleaseReport))
            }
        }
        Type::Ptr(a, sty, am) => {
            write_memory_view_type(code, MemoryViewKind::Ptr, a.address_space, sty, *am, call_info, ctx)
        }
        Type::Store(sty) => write_store_type(code, sty, call_info, ctx),
    }
}

pub(super) enum MemoryViewKind {
    Ptr,
    Ref,
}

pub(super) fn write_memory_view_type(
    code: &mut CodeWriteSpan,
    kind: MemoryViewKind,
    address_space: AddressSpace,
    sty: &StoreType,
    access: AccessMode,
    call_info: CallInfo,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    let access_mode_str = match access {
        AccessMode::Read => "read",
        AccessMode::Write => "write",
        AccessMode::ReadWrite => "read_write",
    };
    code.write_str(match kind {
        MemoryViewKind::Ptr => "ptr",
        MemoryViewKind::Ref => "💥ref", // this is only used for debugging purposes, since `ref` types may never appear in written form in wgsl code
    })?;
    write!(code, "<")?;
    let mut address_space_str = address_space_ptr_qualifier_str(address_space, call_info)?;
    match kind {
        MemoryViewKind::Ptr => write!(code, "{address_space_str}, ")?,
        MemoryViewKind::Ref => {
            if address_space != ir::AddressSpace::Function {
                // less verbose if we omit function address space during debugging
                write!(code, "{address_space_str}, ")?;
            }
        }
    }
    write_store_type(code, sty, call_info, ctx)?;

    if !address_space.supports_access(access) && !WRITE_UNWRITEABLE_TYPES {
        return Err(WgslErrorKind::TypeMayNotAppearInWrittenForm(Type::Store(sty.clone()))
            .at_level(call_info, WgslErrorLevel::InternalPleaseReport));
    }

    //  https://www.w3.org/TR/WGSL/#address-space
    // quote wgsl spec:
    // When writing a variable declaration or a pointer type in WGSL source:
    // - For the storage address space, the access mode is optional, and defaults to read.
    // - For other address spaces, the access mode must not be written.

    if address_space.default_access_mode() != access {
        match address_space {
            ir::AddressSpace::Storage => {
                write!(code, ", {}", access_mode_str)?;
            }
            _ => {
                if address_space.default_access_mode() != access {
                    if !WRITE_UNWRITEABLE_TYPES {
                        return Err(
                            WgslErrorKind::TypeMayNotAppearInWrittenForm(Type::Store(sty.clone())).at(call_info)
                        );
                    } else {
                        write!(code, ", {}", access_mode_str)?;
                    }
                }
            }
        }
    }
    write!(code, ">")?;
    Ok(())
}

fn scalar_type_suffix(stype: ScalarType) -> Result<&'static str, WgslErrorKind> {
    Ok(match stype {
        ScalarType::F16 => "h",
        ScalarType::F32 => "f",
        ScalarType::F64 => return Err(WgslErrorKind::F64Unsupported),
        ScalarType::U32 => "u",
        ScalarType::I32 => "i",
        ScalarType::Bool => "",
    })
}

fn scalar_type_str(stype: ScalarType) -> Result<&'static str, WgslErrorKind> {
    Ok(match stype {
        ScalarType::F16 => "f16",
        ScalarType::F32 => "f32",
        ScalarType::F64 => return Err(WgslErrorKind::F64Unsupported),
        ScalarType::U32 => "u32",
        ScalarType::I32 => "i32",
        ScalarType::Bool => "bool",
    })
}

pub(super) fn write_store_type(
    code: &mut CodeWriteSpan,
    store: &StoreType,
    call_info: CallInfo,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    match store {
        StoreType::Sized(sized) => write_sized_type(code, sized, call_info, ctx),
        StoreType::RuntimeSizedArray(elem) => {
            write!(code, "array<")?;
            write_sized_type(code, elem, call_info, ctx)?;
            write!(code, ">")?;
            Ok(())
        }
        StoreType::Handle(handle) => {
            let suffix_for_shape = |shape: &_| match shape {
                TextureShape::_1D => "_1d",
                TextureShape::_2D => "_2d",
                TextureShape::_2DArray(_) => "_2d_array",
                TextureShape::_3D => "_3d",
                TextureShape::Cube => "_cube",
                TextureShape::CubeArray(_) => "_cube_array",
            };
            match handle {
                HandleType::SampledTexture(shape, sample, spp) => {
                    use ChannelFormatShaderType as ST;

                    write!(code, "texture")?;

                    if let TextureSampleUsageType::Depth = sample {
                        write!(code, "_depth")?;
                    }

                    match spp {
                        SamplesPerPixel::Single => (),
                        SamplesPerPixel::Multi => write!(code, "_multisampled")?,
                    };

                    write!(code, "{}", suffix_for_shape(shape))?;

                    match sample {
                        TextureSampleUsageType::FilterableFloat { len } |
                        TextureSampleUsageType::Nearest {
                            len,
                            channel_type: ST::F32,
                        } => write!(code, "<f32>")?,
                        TextureSampleUsageType::Nearest {
                            len,
                            channel_type: ST::I32,
                        } => write!(code, "<i32>")?,
                        TextureSampleUsageType::Nearest {
                            len,
                            channel_type: ST::U32,
                        } => write!(code, "<u32>")?,
                        TextureSampleUsageType::Depth => (),
                    };
                    Ok(())
                }
                HandleType::StorageTexture(shape, fmt, access) => {
                    //texture_storage_2d_array<Format, Access>
                    write!(code, "texture_storage{}<", suffix_for_shape(shape))?;
                    write_texture_format(code, fmt, call_info, ctx)?;
                    write!(code, ", ")?;
                    code.write_str(match access {
                        AccessMode::Read => "read",
                        AccessMode::Write => "write",
                        AccessMode::ReadWrite => "read_write",
                    });
                    write!(code, ">")?;
                    Ok(())
                }
                HandleType::Sampler(sbt) => {
                    write!(
                        code,
                        "{}",
                        match sbt {
                            SamplingMethod::Filtering | SamplingMethod::NonFiltering => "sampler",
                            SamplingMethod::Comparison => "sampler_comparison",
                        }
                    )?;
                    Ok(())
                }
            }
        }
        StoreType::BufferBlock(block) => {
            let ident = match ctx.ctx.struct_registry().get(block) {
                Some(def) => Ok(def.ident()),
                None => Err(WgslErrorKind::MissingStructDefinition(block.name().to_string())
                    .at_level(call_info, WgslErrorLevel::InternalPleaseReport)),
            }?;
            write!(code, "{}", &ctx.idents[ident])?;
            Ok(())
        }
    }
}

pub(super) fn write_sized_type(
    code: &mut CodeWriteSpan,
    sized: &SizedType,
    call_info: CallInfo,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    use SizedType as T;
    let at_caller = |e: WgslErrorKind| e.at(call_info);
    match sized {
        T::Vector(len, stype) => match len {
            Len::X1 => write!(code, "{}", scalar_type_str(*stype).map_err(at_caller)?)?,
            xn => write!(
                code,
                "vec{}{}",
                u32::from(*xn),
                scalar_type_suffix(*stype).map_err(at_caller)?
            )?,
        },
        T::Matrix(c, r, stype) => write!(
            code,
            "mat{}x{}{}",
            u32::from(*c),
            u32::from(*r),
            scalar_type_suffix((*stype).into()).map_err(at_caller)?
        )?,
        T::Atomic(t) => write!(
            code,
            "atomic<{}>",
            match *t {
                ScalarTypeInteger::U32 => "u32",
                ScalarTypeInteger::I32 => "i32",
            }
        )?,
        T::Array(elem, count) => {
            write!(code, "array<")?;
            write_store_type(code, &StoreType::from((**elem).clone()), call_info, ctx)?;
            write!(code, ", {count}>")?;
        }
        T::Structure(struct_) => {
            let ident = match ctx.ctx.struct_registry().get(struct_) {
                Some(def) => Ok(def.ident()),
                None => Err(WgslErrorKind::MissingStructDefinition(struct_.name().to_string())
                    .at_level(call_info, WgslErrorLevel::InternalPleaseReport)),
            }?;
            write!(code, "{}", &ctx.idents[ident])?;
        }
    }
    Ok(())
}

/// - `Err(e)` if the address space may not appear in a `var<...>` location
/// - `Ok(None)` if this address space is represented by `var` without angle brackets
/// - `Ok(Some(s))` if the address space needs to be written down as `var<s>`
pub(super) fn address_space_var_qualifier_str(
    a: AddressSpace,
    call_info: CallInfo,
) -> Result<Option<&'static str>, WgslError> {
    match a {
        AddressSpace::Function => Ok(None),
        a => address_space_ptr_qualifier_str(a, call_info).map(Some),
    }
}

pub(super) fn address_space_ptr_qualifier_str(a: AddressSpace, call_info: CallInfo) -> Result<&'static str, WgslError> {
    let str = match a {
        AddressSpace::Function => "function",
        AddressSpace::Thread => "private",
        AddressSpace::WorkGroup => "workgroup",
        AddressSpace::Uniform => "uniform",
        AddressSpace::Storage => "storage",
        AddressSpace::PushConstant => "push_constant",
        AddressSpace::Handle => "handle",
        AddressSpace::Output => "output",
    };

    match a {
        AddressSpace::Function |
        AddressSpace::Thread |
        AddressSpace::WorkGroup |
        AddressSpace::Uniform |
        AddressSpace::Storage |
        AddressSpace::PushConstant => Ok(str),
        AddressSpace::Handle | AddressSpace::Output => {
            if WRITE_UNWRITEABLE_TYPES {
                Ok(str)
            } else {
                Err(WgslErrorKind::AddressSpaceMayNotAppearInWrittenForm(a).at(call_info))
            }
        }
    }
}
