//! use `glam` types in `shame`
//!
//! implement `shame::CpuLayout` for some glam types.

use shame as sm;
use sm::GpuLayout;

/// circumventing the orphan rule by defining our own trait.
///
/// this trait must be imported (`use CpuLayoutExt as _`) at the usage site
pub trait CpuLayoutExt {
    fn cpu_layout() -> sm::TypeLayout;
}

// glam::Vec4 matches sm::f32x4 in size and alignment
impl CpuLayoutExt for glam::Vec4 {
    fn cpu_layout() -> sm::TypeLayout { sm::gpu_layout::<sm::f32x4>() }
}

// glam::Vec2 only matches sm::f32x2 if it has 8 byte alignment
impl CpuLayoutExt for glam::Vec2 {
    fn cpu_layout() -> sm::TypeLayout {
        if align_of::<Self>() == 8 {
            sm::gpu_layout::<sm::f32x2>()
        } else {
            panic!("glam needs to use the `cuda` crate feature for Vec2 to be 8 byte aligned");
        }
    }
}

// glam::Mat4 matches sm::f32x4x4 in size and alignment
impl CpuLayoutExt for glam::Mat4 {
    fn cpu_layout() -> sm::TypeLayout { sm::gpu_layout::<sm::f32x4x4>() }
}
