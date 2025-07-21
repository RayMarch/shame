// #![allow(non_camel_case_types, unused)]
// use pretty_assertions::{assert_eq, assert_ne};

// use shame::{self as sm, cpu_layout, gpu_layout};
// use sm::{aliases::*, CpuLayout, GpuLayout};

// #[test]
// fn basic_layout_eq() {
//     #[derive(sm::GpuLayout)]
//     struct OnGpu {
//         a: f32x1,
//         b: u32x1,
//         c: i32x1,
//     }

//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu {
//         a: f32,
//         b: u32,
//         c: i32,
//     }

//     assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
// }

// #[test]
// fn attributes_dont_contribute_to_eq() {
//     #[derive(sm::GpuLayout)]
//     struct OnGpuA {
//         a: f32x1,
//         #[align(4)] // attribute doesn't change layout, u32 is already 4 byte aligned
//         b: u32x1,
//         c: i32x1,
//     }

//     #[derive(sm::GpuLayout)]
//     struct OnGpuB {
//         a: f32x1,
//         #[size(4)] // attribute doesn't change layout, u32 is already 4 bytes in size
//         b: u32x1,
//         c: i32x1,
//     }

//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu {
//         a: f32,
//         b: u32,
//         c: i32,
//     }

//     assert_eq!(gpu_layout::<OnGpuA>(), cpu_layout::<OnCpu>());
//     assert_eq!(gpu_layout::<OnGpuB>(), cpu_layout::<OnCpu>());
//     assert_eq!(gpu_layout::<OnGpuA>(), gpu_layout::<OnGpuB>());
// }

// #[test]
// fn fixed_by_align_size_attribute() {
//     {
//         #[derive(sm::GpuLayout)]
//         struct OnGpu {
//             a: f32x1,
//             #[size(32)]
//             b: f32x3,
//             c: i32x1,
//         }

//         #[derive(sm::CpuLayout)]
//         #[repr(C)]
//         struct OnCpu {
//             a: f32,
//             b: f32x3_size32,
//             c: i32,
//         }

//         assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
//     }

//     {
//         #[derive(sm::GpuLayout)]
//         struct OnGpu {
//             a: f32x1,
//             b: i32x1,
//             #[size(32)]
//             c: f32x3,
//         }

//         #[derive(sm::CpuLayout)]
//         #[repr(C)]
//         struct OnCpu {
//             a: f32,
//             b: i32,
//             c: f32x3_size32,
//         }

//         assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
//     }
// }

// #[test]
// fn different_align_struct_eq() {
//     #[derive(sm::GpuLayout)]
//     struct OnGpu {
//         a: f32x1,
//         b: u32x1,
//         c: i32x1,
//     }

//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu {
//         a: f32,
//         b: u32,
//         c: i32,
//     }

//     assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
// }

// #[test]
// fn unsized_struct_layout_eq() {
//     #[derive(sm::GpuLayout)]
//     struct OnGpu {
//         a: f32x1,
//         b: u32x1,
//         c: sm::Array<i32x1>,
//     }

//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu {
//         a: f32,
//         b: u32,
//         c: [i32],
//     }

//     assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
// }

// #[derive(Clone, Copy)]
// #[repr(C, align(16))]
// struct f32x4_cpu(pub [f32; 4]);

// #[derive(Clone, Copy)]
// #[repr(C, align(16))]
// struct f32x3_cpu(pub [f32; 3]);

// impl CpuLayout for f32x3_cpu {
//     fn cpu_layout() -> shame::TypeLayout { gpu_layout::<f32x3>() }
// }

// #[derive(Clone, Copy)]
// #[repr(C, align(8))]
// struct f32x2_cpu(pub [f32; 2]);
// impl CpuLayout for f32x2_cpu {
//     fn cpu_layout() -> shame::TypeLayout { gpu_layout::<f32x2>() }
// }

// #[derive(Clone, Copy)]
// #[repr(C)]
// struct f32x2_align4(pub [f32; 2]);
// impl CpuLayout for f32x2_align4 {
//     fn cpu_layout() -> shame::TypeLayout {
//         let mut layout = gpu_layout::<f32x2>();
//         layout.align = shame::any::U32PowerOf2::_4.into();
//         layout
//     }
// }

// #[derive(Clone, Copy)]
// #[repr(C)]
// struct f32x4_align4(pub [f32; 4]);

// impl CpuLayout for f32x4_align4 {
//     fn cpu_layout() -> shame::TypeLayout {
//         let mut layout = gpu_layout::<f32x4>();
//         layout.align = shame::any::U32PowerOf2::_4.into();
//         layout
//     }
// }

// #[derive(Clone, Copy)]
// #[repr(C)]
// struct f32x3_align4(pub [f32; 3]);

// // the tests assume that this is the alignment of glam vecs.
// static_assertions::assert_eq_align!(glam::Vec2, f32x2_align4);
// static_assertions::assert_eq_align!(glam::Vec3, f32x3_align4);
// static_assertions::assert_eq_align!(glam::Vec4, f32x4_cpu);

// impl CpuLayout for f32x3_align4 {
//     fn cpu_layout() -> shame::TypeLayout {
//         let mut layout = gpu_layout::<f32x3>();
//         layout.align = shame::any::U32PowerOf2::_4.into();
//         layout
//     }
// }

// #[derive(Clone, Copy)]
// #[repr(C, align(16))]
// struct f32x3_size32(pub [f32; 3], [u8; 20]);

// impl CpuLayout for f32x3_size32 {
//     fn cpu_layout() -> shame::TypeLayout { gpu_layout::<f32x3>() }
// }



// #[test]
// fn unsized_struct_vec3_align_layout_eq() {
//     #[derive(sm::GpuLayout)]
//     struct OnGpu {
//         a: f32x1,
//         b: u32x1,
//         c: sm::Array<f32x3>,
//     }

//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu {
//         a: f32,
//         b: u32,
//         c: [f32x3_cpu],
//     }

//     assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
// }

// #[test]
// #[rustfmt::skip] fn top_level_align_ignore() {
//     #[derive(sm::GpuLayout)]
//     struct OnGpu { // size=16, align=16
//         a: f32x4, // size=16, align=16
//     }

//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu {       // size=16, align=4
//         a: f32x4_align4, // size=16, align=4
//     }

//     // the alignment on the top level of the layout doesn't matter.
//     // two layouts are only considered different if an alignment mismatch
//     // leads to different offsets of fields or array elements
//     assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
// }

// #[test]
// #[rustfmt::skip] fn struct_align_round_up() {
//     #[derive(sm::GpuLayout)]
//     struct OnGpu { // size=round_up(16, 12)=16, align=16
//         a: f32x3, // size=12, align=16
//     }

//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu { // size=12, align=4
//         a: f32x3_align4,
//     }
//     assert_ne!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
//     assert!(gpu_layout::<OnGpu>().byte_size() == Some(16));
//     assert!(gpu_layout::<OnGpu>().align().as_u32() == 16);
//     assert!(cpu_layout::<OnCpu>().byte_size() == Some(12));
//     assert!(cpu_layout::<OnCpu>().align().as_u32() == 4);
// }

// #[test]
// fn unsized_struct_nested_vec3_align_layout_eq() {
//     #[derive(sm::GpuLayout)]
//     struct InnerGpu {
//         a: f32x1,
//         b: u32x1,
//     }

//     #[derive(sm::CpuLayout, Clone)]
//     #[repr(C)]
//     struct InnerCpu {
//         a: f32,
//         b: u32,
//     }

//     #[derive(sm::GpuLayout)]
//     struct OnGpu {
//         a: f32x1,
//         b: u32x1,
//         c: sm::Array<sm::Struct<InnerGpu>>,
//     }

//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu {
//         a: f32,
//         b: u32,
//         c: [InnerCpu],
//     }

//     assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
// }

// #[test]
// fn unsized_array_layout_eq() {
//     assert_eq!(gpu_layout::<sm::Array<f32x1>>(), cpu_layout::<[f32]>());
//     assert_eq!(gpu_layout::<sm::Array<f32x3>>(), cpu_layout::<[f32x3_cpu]>());
//     assert_ne!(gpu_layout::<sm::Array<f32x3>>(), cpu_layout::<[f32x3_align4]>());
//     assert_ne!(gpu_layout::<sm::Array<f32x3>>(), cpu_layout::<[f32x3_size32]>());
// }

// #[test]
// fn layouts_mismatch() {
//     #[derive(sm::GpuLayout)]
//     struct OnGpuMore {
//         a: f32x1,
//         b: u32x1,
//         c: i32x1,
//         d: i32x1,
//     }

//     #[derive(sm::GpuLayout)]
//     struct OnGpuLess {
//         a: f32x1,
//         b: u32x1,
//     }

//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu {
//         a: f32,
//         b: u32,
//         c: i32,
//     }

//     assert_ne!(gpu_layout::<OnGpuLess>(), cpu_layout::<OnCpu>());
//     assert_ne!(gpu_layout::<OnGpuMore>(), cpu_layout::<OnCpu>());
// }

// #[test]
// fn external_vec_type() {
//     // using duck-traiting just so that the proc-macro uses `CpuLayoutExt::layout()`
//     pub mod my_mod {
//         use shame::gpu_layout;
//         use shame as sm;
//         use sm::aliases::*;
//         use sm::GpuLayout as _;

//         pub trait CpuLayoutExt {
//             fn cpu_layout() -> shame::TypeLayout;
//         }

//         impl CpuLayoutExt for glam::Vec4 {
//             fn cpu_layout() -> shame::TypeLayout { gpu_layout::<f32x4>() }
//         }

//         impl CpuLayoutExt for glam::Vec3 {
//             fn cpu_layout() -> shame::TypeLayout {
//                 let mut layout = gpu_layout::<f32x3>();
//                 layout.align = sm::any::U32PowerOf2::_4.into();
//                 layout
//             }
//         }
//     }

//     #[derive(sm::GpuLayout)]
//     struct OnGpu {
//         a: f32x4,
//         b: f32x4,
//     }

//     use my_mod::CpuLayoutExt as _; // makes `glam::Vec4::layout()` compile in the derive generated code.
//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu {
//         a: glam::Vec4,
//         b: glam::Vec4,
//     }

//     assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());

//     #[derive(sm::GpuLayout)]
//     struct OnGpu2 {
//         a: f32x3,
//         b: f32x3,
//         #[align(16)]
//         c: f32x4,
//     }

//     #[derive(sm::CpuLayout)]
//     #[repr(C)]
//     struct OnCpu2 {
//         a: glam::Vec3,
//         b: glam::Vec3,
//         c: glam::Vec4,
//     }

//     assert_ne!(gpu_layout::<OnGpu2>(), cpu_layout::<OnCpu2>());

//     // TODO: delete or use compile fail test crate like trybuild to make
//     // sure that align and size attributes aren't allowed on packed structs.
//     // #[derive(sm::GpuLayout)]
//     // #[gpu_repr(packed)]
//     // struct OnGpu2Packed {
//     //     a: f32x3,
//     //     b: f32x3,
//     //     #[align(16)]
//     //     c: f32x4,
//     // }

//     // assert_eq!(gpu_layout::<OnGpu2Packed>(), cpu_layout::<OnCpu2>());
// }

// #[test]
// #[rustfmt::skip] fn gpu_repr_packed_test() {
//     {
//         #[derive(sm::GpuLayout)]
//         #[gpu_repr(packed)]
//         struct OnGpu {
//             pos: f32x3,
//             nor: f32x3,
//             uv : f32x2,
//         }

//         #[derive(sm::CpuLayout)]
//         #[repr(C)]
//         struct OnCpu {
//             pos: f32x3_align4,
//             nor: f32x3_align4,
//             uv : f32x2_align4,
//         }

//         assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
//     }
//     {
//         // TODO: delete or use compile fail test crate like trybuild to make
//         // sure that align and size attributes aren't allowed on packed structs.
//         // #[derive(sm::GpuLayout)]
//         // #[gpu_repr(packed)]
//         // struct OnGpu {
//         //     pos: f32x3,
//         //     nor: f32x3,
//         //     #[align(8)] uv : f32x2,
//         // }

//         // #[derive(sm::CpuLayout)]
//         // #[repr(C)]
//         // struct OnCpu {
//         //     pos: f32x3_align4,
//         //     nor: f32x3_align4,
//         //     uv : f32x2_cpu,
//         // }

//         // assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
//         // enum __ where OnGpu: sm::VertexLayout {}
//     }
// }
