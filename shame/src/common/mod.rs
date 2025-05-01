pub mod floating_point;
pub mod format;
pub mod ignore_eq;
pub mod integer;
pub mod iterator_ext;
pub mod marker;
pub mod ops;
pub mod option_ext;
pub mod po2;
pub mod pool;
pub mod prettify;
#[allow(missing_docs)]
pub mod proc_macro_reexports;
#[allow(missing_docs)]
pub mod proc_macro_utils;

pub mod small_vec_actual; // smallvec implementation
pub mod small_vec_dummy; // dummy implementation that just uses a std::vec::Vec under the hood

// which smallvec impl to actually use
pub use small_vec_actual as small_vec;
