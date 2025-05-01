//! a set of common imports, useful when writing pipelines.
//!
//! intended to be used in files where pipelines are defined like this:
//!
//! `use shame::prelude::*`
use crate as shame;

// make these traits usable
pub use shame::GpuIndex as _; // gpu_array.index(4u32)
pub use shame::GpuLayout as _; // t.gpu_layout()
pub use shame::CpuLayout as _; // t.cpu_layout()
pub use shame::ToGpuType as _; // t.to_gpu()
pub use shame::ToScalar as _; // 1.0_f32.splat()

pub use shame::{x1, x2, x3, x4};
pub use shame::Size;

pub use shame::GradPrecision::Coarse;
pub use shame::GradPrecision::Fine;
