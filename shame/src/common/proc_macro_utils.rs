use std::rc::Rc;

use crate::{
    frontend::{
        any::{Any, InvalidReason},
        rust_types::{
            error::FrontendError,
            type_layout::{
                FieldLayout, FieldLayoutWithOffset, StructLayout, TypeLayout, TypeLayoutRules, TypeLayoutSemantics,
            },
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

pub struct ReprCField {
    pub name: &'static str,
    pub alignment: usize,
    pub layout: TypeLayout,
}

pub enum ReprCError {
    SecondLastElementIsUnsized,
}

pub fn repr_c_struct_layout(
    repr_c_align_attribute: Option<u64>,
    struct_name: &'static str,
    first_fields_with_offsets_and_sizes: &[(ReprCField, usize, usize)],
    last_field: ReprCField,
    last_field_size: Option<usize>,
) -> Result<TypeLayout, ReprCError> {
    let last_field_offset = match first_fields_with_offsets_and_sizes.last() {
        None => 0,
        Some((_2nd_last_field, _2nd_last_offset, _2nd_last_size)) => {
            let Some(_) = _2nd_last_field.layout.byte_size() else {
                return Err(ReprCError::SecondLastElementIsUnsized);
            };
            round_up(
                last_field.alignment as u64,
                *_2nd_last_offset as u64 + *_2nd_last_size as u64,
            )
        }
    };

    let max_alignment = first_fields_with_offsets_and_sizes
        .iter()
        .map(|(f, _, _)| f.alignment)
        .fold(last_field.alignment, ::std::cmp::max) as u64;

    let struct_alignment = match repr_c_align_attribute {
        Some(repr_c_align) => max_alignment.max(repr_c_align),
        None => max_alignment,
    };
    let last_field_size = last_field_size.map(|s| s as u64);

    let total_struct_size = last_field_size.map(|last_size| round_up(struct_alignment, last_field_offset + last_size));

    let mut fields = first_fields_with_offsets_and_sizes
        .iter()
        .map(|(field, offset, size)| (field, *offset as u64, *size as u64))
        .map(|(field, offset, size)| FieldLayoutWithOffset {
            field: FieldLayout {
                custom_min_align: None.into(),
                custom_min_size: (field.layout.byte_size() != Some(size)).then_some(size).into(),
                name: field.name.into(),
                ty: field.layout.clone(),
            },
            rel_byte_offset: offset,
        })
        .chain(std::iter::once(FieldLayoutWithOffset {
            field: FieldLayout {
                custom_min_align: None.into(),
                custom_min_size: (last_field.layout.byte_size() != last_field_size)
                    .then_some(last_field_size)
                    .flatten()
                    .into(),
                name: last_field.name.into(),
                ty: last_field.layout,
            },
            rel_byte_offset: last_field_offset,
        }))
        .collect::<Vec<_>>();

    Ok(TypeLayout::new(
        total_struct_size,
        struct_alignment,
        TypeLayoutSemantics::Structure(Rc::new(StructLayout {
            name: struct_name.into(),
            fields,
        })),
    ))
}

#[track_caller]
pub fn call_info_scope() -> CallInfoScope { Context::call_info_scope() }
