use std::rc::Rc;

use crate::{
    call_info,
    frontend::{
        any::{Any, InvalidReason},
        rust_types::{
            error::FrontendError,
            type_layout::{FieldLayout, StructLayout, TypeLayout},
        },
    },
    ir::{
        ir_type::{round_up, CanonName},
        recording::{CallInfo, CallInfoScope, Context},
        AlignedType, SizedType,
    },
};

use super::po2::U32PowerOf2;

pub fn push_wrong_amount_of_args_error(amount: usize, expected_amount: usize, call_info: CallInfo) -> Any {
    Context::try_with(call_info, |ctx| {
        ctx.push_error_get_invalid_any(
            FrontendError::InvalidCompositeDowncastAmount {
                expected: expected_amount,
                actual: amount,
            }
            .into(),
        )
    })
    .unwrap_or(Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
}

/// fails if the iterator doesn't yield exactly `N` elements.
/// (fails even if there are more than `N` elements in `it`).
///
/// if collecting fails, returns the actual amount of elements the iterator had.
pub fn collect_into_array_exact<T, const N: usize>(mut it: impl Iterator<Item = T>) -> Result<[T; N], usize> {
    let mut arr: [Option<T>; N] = [(); N].map(|()| None);

    let mut i = 0;
    for t in it {
        if i < N {
            arr[i] = Some(t);
        }
        i += 1;
    }

    let iterator_len = i;

    if iterator_len == N {
        Ok(arr.map(|opt| opt.expect("iterator had the required length")))
    } else {
        Err(iterator_len)
    }
}

#[derive(Clone)]
pub struct ReprCField {
    pub name: &'static str,
    pub alignment: U32PowerOf2,
    pub layout: TypeLayout,
}

pub enum ReprCError {
    SecondLastElementIsUnsized,
}

/// created when a cpu layout is observed to not match the values of `std::mem::size_of`, `std::mem::align_of`, `CpuAligned::CPU_SIZE`, and `CpuAligned::CPU_ALIGNMENT`
#[derive(Debug, thiserror::Error, Clone)]
pub enum CpuLayoutImplMismatch {
    UnexpectedSize {
        type_name: String,
        struct_name: &'static str,
        /// size that was expected by `std::mem::size_of` / `CpuAligned::CPU_SIZE`
        expected: Option<u64>,
        /// size that was provided by the (maybe user defined) impl of `CpuLayout`
        cpu_layout_provided: Option<u64>,
    },
}

impl std::fmt::Display for CpuLayoutImplMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpuLayoutImplMismatch::UnexpectedSize {
                type_name: t,
                struct_name,
                expected: std_mem_size,
                cpu_layout_provided: cpu_layout_impl_size,
            } => {
                write!(f, "The `CpuLayout` implementation of `{t}` claims that `{t}` ")?;
                match cpu_layout_impl_size {
                    Some(s) => write!(f, "has a compile-time known size of {s},")?,
                    None => write!(f, "is unsized at compile-time,")?,
                };
                write!(f, "\nbut a `{t}` within `{struct_name}` was just observed ")?;
                match std_mem_size {
                    Some(s) => write!(f, "having a compile-time known size of {s}")?,
                    None => write!(f, "being unsized")?,
                };
                writeln!(f, ".")?;
                writeln!(
                    f,
                    "This is most likely caused by a mistake in the `shame::CpuLayout` implementation of {t}
                or in the implementation of one of the types it is composed of. 
                The size must be equal to what `std::mem::size_of` returns."
                )?;
            }
        }
        Ok(())
    }
}

#[track_caller]
fn try_report_cpu_layout_impl_mismatch(err: CpuLayoutImplMismatch) {
    let caller = call_info!();
    let success = Context::try_with(caller, |ctx| {
        ctx.push_error(crate::frontend::encoding::EncodingErrorKind::LayoutError(
            err.clone().into(),
        ));
    })
    .unwrap_or_else(|| {
        if crate::__private::DEBUG_PRINT_ENABLED {
            println!("`shame` warning @ {caller}:\n{err}");
        } else {
            // unable to report assumed implementation mistake of `CpuLayout` for a given type
        }
    });
}

#[track_caller]
pub fn repr_c_struct_layout(
    repr_c_align_attribute: Option<U32PowerOf2>,
    struct_name: &'static str,
    first_fields_with_offsets_and_sizes: &[(ReprCField, usize, usize)],
    mut last_field: ReprCField,
    // the size of the last field according to the `CpuAligned` trait's associated constant
    last_field_trait_size: Option<usize>,
) -> Result<TypeLayout, ReprCError> {
    let last_field_offset = match first_fields_with_offsets_and_sizes.last() {
        None => 0,
        Some((_2nd_last_field, _2nd_last_offset, _2nd_last_size)) => {
            let Some(_) = _2nd_last_field.layout.byte_size() else {
                return Err(ReprCError::SecondLastElementIsUnsized);
            };
            round_up(
                last_field.alignment.as_u64(),
                *_2nd_last_offset as u64 + *_2nd_last_size as u64,
            )
        }
    };

    let struct_alignment = {
        let max_alignment = first_fields_with_offsets_and_sizes
            .iter()
            .map(|(f, _, _)| f.alignment)
            .fold(last_field.alignment, ::std::cmp::max);
        match repr_c_align_attribute {
            Some(repr_c_align_attribute) => max_alignment.max(repr_c_align_attribute),
            None => max_alignment,
        }
    };

    /// the size of the last field according to the `CpuAligned` trait's associated constant
    let last_field_trait_size = last_field_trait_size.map(|s| s as u64);

    let total_struct_size =
        last_field_trait_size.map(|last_size| round_up(struct_alignment.as_u64(), last_field_offset + last_size));

    let mut fields = first_fields_with_offsets_and_sizes
        .iter()
        .map(|(field, std_mem_offset_of, std_mem_size_of)| (field, *std_mem_offset_of as u64, *std_mem_size_of as u64))
        .map(|(mut field, std_mem_offset_of, std_mem_size_of)| {
            let mut layout = field.layout.clone();
            // here `std::mem::size_of` is prioritized over `<#field_type>::cpu_layout().byte_size()`.
            // They can disagree if the user-driven `cpu_layout()` implementation is broken. TODO(release) reconsider this, especially in the case of f32x3
            layout.set_byte_size(std_mem_size_of);
            FieldLayout {
                rel_byte_offset: std_mem_offset_of,
                name: field.name.into(),
                ty: layout,
            }
        })
        .chain(std::iter::once({
            if last_field.layout.byte_size() != last_field_trait_size {
                try_report_cpu_layout_impl_mismatch(CpuLayoutImplMismatch::UnexpectedSize {
                    struct_name,
                    type_name: last_field.layout.short_name(),
                    expected: last_field_trait_size,
                    cpu_layout_provided: last_field.layout.byte_size(),
                });
            }

            // here `<#last_field_type as CpuAligned>::CPU_SIZE` is prioritized over `<#field_type>::cpu_layout().byte_size()`.
            // if the reporting above failed. The two can disagree if the user-driven `cpu_layout()` implementation is
            // broken.
            //
            // If the error could not be reported above, the user cannot be informed and they
            // have to accept the consequences of their broken `CpuLayout` impl.
            match (last_field.layout.removable_byte_size_mut(), last_field_trait_size) {
                (Ok(layout_maybe_size), trait_maybe_size) => *layout_maybe_size = trait_maybe_size,
                (Err(layout_size), Some(trait_size)) => *layout_size = trait_size,
                (Err(layout_size), None) => {
                    // in this case the rust type is an always-sized type like vector/matrix/packedvec,
                    // but the `CpuLayout` impl claims it is an unsized struct or array.
                }
            };

            FieldLayout {
                rel_byte_offset: last_field_offset,
                name: last_field.name.into(),
                ty: last_field.layout,
            }
        }))
        .collect::<Vec<_>>();

    Ok(StructLayout {
        byte_size: total_struct_size,
        align: struct_alignment.into(),
        name: struct_name.into(),
        fields,
    }
    .into())
}

#[track_caller]
pub fn call_info_scope() -> CallInfoScope { Context::call_info_scope() }
