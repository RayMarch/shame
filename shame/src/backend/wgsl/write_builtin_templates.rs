use super::{error::WgslError, WgslContext};
use crate::{
    backend::{
        code_write_buf::CodeWriteSpan,
        wgsl::{
            address_space_var_qualifier_str,
            error::WgslErrorLevel,
            write_sized_type,
            write_type::{self, address_space_ptr_qualifier_str, write_memory_view_type, MemoryViewKind},
            WgslErrorKind,
        },
    },
    call_info,
    ir::{
        self,
        recording::{AtomicCompareExchangeWeakGenerics, FrexpGenerics, ModfGenerics, TemplateStructParams},
        SizedStruct, SizedType,
    },
};
use std::fmt::Write;

/// functions like `frexp` return values whose type that cannot be explicitly written down in wgsl code.
///
/// To avoid having to implement types that cannot be written down, and all the implications this
/// has on codegen, we instead define wrapper functions that wrap those structs into
/// explicitly redeclared versions of those structs, whose type can be written down in wgsl code
/// without restriction.
pub(super) fn write_builtin_template_wrapper_functions(
    code: &mut CodeWriteSpan,
    params: &TemplateStructParams,
    struct_: &ir::SizedStruct,
    ctx: &WgslContext,
) -> Result<(), WgslError> {
    use TemplateStructParams as P;

    let (struct_ident_key, call_info) = {
        let struct_registry = ctx.ctx.struct_registry();
        let Some(struct_def) = struct_registry.get(struct_) else {
            return Err(WgslErrorKind::MissingTemplateStructDefinition(*params)
                .at_level(call_info!(), WgslErrorLevel::InternalPleaseReport));
        };
        (struct_def.ident(), struct_def.call_info())
    };

    let struct_ident = &ctx.idents[struct_ident_key];
    let mut code = code.sub_span(call_info);

    write!(code, "fn ")?;
    write_builtin_template_wrapper_fn_name(&mut code, params)?;
    match params {
        P::Frexp(FrexpGenerics(fp, len)) => {
            let t = ir::ScalarType::from(*fp);

            write!(code, "(e: ")?;
            write_sized_type(&mut code, &SizedType::Vector(*len, t), call_info, ctx)?;
            writeln!(
                code,
                ") -> {struct_ident} {{
    let r = frexp(e);
    return {struct_ident}(r.fract, r.exp);
}}"
            )?;
        }
        P::Modf(ModfGenerics(fp, len)) => {
            let t = ir::ScalarType::from(*fp);

            write!(code, "(e: ")?;
            write_sized_type(&mut code, &SizedType::Vector(*len, t), call_info, ctx)?;
            writeln!(
                code,
                ") -> {struct_ident} {{
    let r = modf(e);
    return {struct_ident}(r.fract, r.whole);
}}"
            )?;
        }
        P::AtomicCompareExchangeWeak(AtomicCompareExchangeWeakGenerics(addr, int)) => {
            let t = ir::ScalarType::from(*int);

            let arg_indent = "        ";
            write!(code, "(\n{arg_indent}atomic_ptr: ")?;
            write_memory_view_type(
                &mut code,
                MemoryViewKind::Ptr,
                *addr,
                &ir::StoreType::Sized(SizedType::Atomic(*int)),
                ir::AccessMode::ReadWrite,
                call_info,
                ctx,
            )?;
            write!(code, ",\n{arg_indent}cmp: ")?;
            write_sized_type(&mut code, &ir::SizedType::Vector(ir::Len::X1, t), call_info, ctx)?;
            write!(code, ",\n{arg_indent}v: ")?;
            write_sized_type(&mut code, &ir::SizedType::Vector(ir::Len::X1, t), call_info, ctx)?;
            writeln!(
                code,
                "
) -> {struct_ident} {{
    let r = atomicCompareExchangeWeak(atomic_ptr, cmp, v);
    return {struct_ident}(r.old_value, r.exchanged);
}}"
            )?;
        }
    }
    Ok(())
}

pub(super) fn write_builtin_template_wrapper_fn_name(
    code: &mut CodeWriteSpan,
    params: &TemplateStructParams,
) -> Result<(), WgslError> {
    use TemplateStructParams as P;
    match params {
        P::Frexp(FrexpGenerics(fp, len)) => {
            let t = ir::ScalarType::from(*fp);
            write!(code, "sm_frexp_{t}{len}")?
        }
        P::Modf(ModfGenerics(fp, len)) => {
            let t = ir::ScalarType::from(*fp);
            write!(code, "sm_modf_{t}{len}")?
        }
        P::AtomicCompareExchangeWeak(AtomicCompareExchangeWeakGenerics(addr, int)) => {
            let t = ir::ScalarType::from(*int);
            let addr_shorthand = addr.ident_suffix();
            write!(code, "sm_atomicCompareExchangeWeak_{t}_{addr_shorthand}")?
        }
    };
    Ok(())
}
