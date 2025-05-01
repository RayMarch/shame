use core::slice::from_ref;

pub fn slice_from_opt<T>(opt: &Option<T>) -> &[T] { opt.as_ref().map(from_ref).unwrap_or_default() }
