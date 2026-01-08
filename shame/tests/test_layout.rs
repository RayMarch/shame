#![allow(non_camel_case_types, unused)]
use pretty_assertions::{assert_eq, assert_ne};
use sm::__private::proc_macro_reexports::CpuAligned;

use shame::{self as sm, cpu_layout, gpu_layout};
use sm::{aliases::*, CpuLayout, GpuLayout};

#[test]
fn basic_layout_eq() {
    #[derive(sm::GpuLayout)]
    struct OnGpu {
        a: f32x1,
        b: u32x1,
        c: i32x1,
    }

    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu {
        a: f32,
        b: u32,
        c: i32,
    }

    assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
}

#[test]
fn attributes_dont_contribute_to_eq() {
    #[derive(sm::GpuLayout)]
    struct OnGpuA {
        a: f32x1,
        #[align(4)] // attribute doesn't change layout, u32 is already 4 byte aligned
        b: u32x1,
        c: i32x1,
    }

    #[derive(sm::GpuLayout)]
    struct OnGpuB {
        a: f32x1,
        #[size(4)] // attribute doesn't change layout, u32 is already 4 bytes in size
        b: u32x1,
        c: i32x1,
    }

    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu {
        a: f32,
        b: u32,
        c: i32,
    }

    assert_eq!(gpu_layout::<OnGpuA>(), cpu_layout::<OnCpu>());
    assert_eq!(gpu_layout::<OnGpuB>(), cpu_layout::<OnCpu>());
    assert_eq!(gpu_layout::<OnGpuA>(), gpu_layout::<OnGpuB>());
}

#[test]
fn fixed_by_align_size_attribute() {
    {
        #[derive(sm::GpuLayout)]
        struct OnGpu {
            a: f32x1,
            #[size(32)]
            b: f32x3,
            c: i32x1,
        }

        #[derive(sm::CpuLayout)]
        #[repr(C)]
        struct OnCpu {
            a: f32,
            b: f32x3_size32,
            c: i32,
        }

        assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
    }

    {
        #[derive(sm::GpuLayout)]
        struct OnGpu {
            a: f32x1,
            b: i32x1,
            #[size(32)]
            c: f32x3,
        }

        #[derive(sm::CpuLayout)]
        #[repr(C)]
        struct OnCpu {
            a: f32,
            b: i32,
            c: f32x3_size32,
        }

        assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
    }

    {
        #[derive(sm::GpuLayout)]
        struct OnGpu {
            a: f32x1,
            #[size(16)]
            b: f32x3,
            c: i32x1,
        }

        #[derive(sm::CpuLayout)]
        #[repr(C)]
        struct OnCpu {
            a: f32,
            b: f32x3_size16,
            c: i32,
        }

        assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
    }

    {
        #[derive(sm::GpuLayout)]
        struct OnGpu {
            a: f32x1,
            b: i32x1,
            #[size(16)] 
            c: f32x3, // TODO(release) this should work even without #[size(16)], no?
        }

        #[derive(sm::CpuLayout)]
        #[repr(C)]
        struct OnCpu {
            a: f32,
            b: i32,
            c: f32x3_size16,
        }
        
        assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
    }

    {
        #[derive(sm::GpuLayout)]
        struct OnGpu {
            a: f32x4,
            b: f32x3, // align 16
            c: i32x1,
        }

        #[derive(sm::CpuLayout)]
        #[repr(C)]
        struct OnCpu {
            a: f32x4_cpu,
            b: f32x3_align4, // de-facto 16 aligned
            c: i32,
        }

        assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
    }

    {
        // this is the case where rust's idea that `size` must be multiple of `align`
        // clashes with wgsl's `vec3f`

        #[derive(sm::GpuLayout)]
        struct OnGpu {
            a: f32x1,
            b: f32x3,
            c: i32x1,
        }

        #[derive(sm::CpuLayout)]
        #[repr(C)]
        struct OnCpu {
            a: f32,
            b: f32x3_cpu,
            c: i32,
        }

        assert_ne!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
    }
}

#[test]
fn different_align_struct_eq() {
    #[derive(sm::GpuLayout)]
    struct OnGpu {
        a: f32x1,
        b: u32x1,
        c: i32x1,
    }

    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu {
        a: f32,
        b: u32,
        c: i32,
    }

    assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
}

#[test]
fn unsized_struct_layout_eq() {
    #[derive(sm::GpuLayout)]
    struct OnGpu {
        a: f32x1,
        b: u32x1,
        c: sm::Array<i32x1>,
    }

    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu {
        a: f32,
        b: u32,
        c: [i32],
    }

    assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct f32x4_cpu(pub [f32; 4]);

impl CpuLayout for f32x4_cpu {
    fn cpu_layout() -> shame::TypeLayout {
        rust_layout_with_shame_semantics::<Self, f32x4>()
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct f32x3_cpu(pub [f32; 3]);

impl CpuLayout for f32x3_cpu {
    fn cpu_layout() -> shame::TypeLayout {
        println!("this impl of `CpuLayout` is wrong, do not copy-paste it into your application");
        // This impl of `CpuLayout` is wrong. It claims that `Self` has size 12
        // and not size 16, as `std::mem::size_of::<Self>()` would return. 
        // It still represents a way a user might try to replicate wgsl's vec3f.
        // At the time of writing it is undecided how we want to deal with this.
        // The user can have an align4 and an align16 implementation of Vec3, similar
        // to `glam`. The user could then choose depending on the desired packing.
        // Atm this does not cause an actual memory bug, because actual offsets and
        // sizes are used in cpu-layout checks.

        // TODO(release): decide on the above issue and, depending on decision, remove `f32x3_cpu` entirely
        let mut layout = gpu_layout::<f32x3>(); // size 12
        *layout.align_mut() = Self::CPU_ALIGNMENT; // align 16
        layout
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(8))]
struct f32x2_cpu(pub [f32; 2]);
impl CpuLayout for f32x2_cpu {
    fn cpu_layout() -> shame::TypeLayout { rust_layout_with_shame_semantics::<Self, f32x2>() }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct f32x2_align4(pub [f32; 2]);
impl CpuLayout for f32x2_align4 {
    fn cpu_layout() -> shame::TypeLayout { rust_layout_with_shame_semantics::<Self, f32x2>() }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct f32x4_align4(pub [f32; 4]);

impl CpuLayout for f32x4_align4 {
    fn cpu_layout() -> shame::TypeLayout { rust_layout_with_shame_semantics::<Self, f32x4>() }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct f32x3_align4(pub [f32; 3]);

// the tests assume that this is the alignment of glam vecs.
static_assertions::assert_eq_align!(glam::Vec2, f32x2_align4);
static_assertions::assert_eq_align!(glam::Vec3, f32x3_align4);
static_assertions::assert_eq_align!(glam::Vec4, f32x4_cpu);

impl CpuLayout for f32x3_align4 {
    fn cpu_layout() -> shame::TypeLayout {
        // TODO(release): replace this with `rust_layout_with_shame_semantics::<Self, f32x3>()`
        // and find a proper solution to the consequences. Its size is 16, and not 12.
        let mut layout = gpu_layout::<f32x3>();
        *layout.align_mut() = Self::CPU_ALIGNMENT;
        layout
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct f32x3_size16(pub [f32; 3], [u8; 4]);

impl CpuLayout for f32x3_size16 {
    fn cpu_layout() -> shame::TypeLayout {
        rust_layout_with_shame_semantics::<Self, sm::f32x3>()
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct f32x3_size32(pub [f32; 3], [u8; 20]);

impl CpuLayout for f32x3_size32 {
    fn cpu_layout() -> shame::TypeLayout {
        rust_layout_with_shame_semantics::<Self, sm::f32x3>()
    }
}



#[test]
fn unsized_struct_vec3_align_layout_eq() {
    #[derive(sm::GpuLayout)]
    struct OnGpu {
        a: f32x1,
        b: u32x1,
        c: sm::Array<f32x3>,
    }

    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu {
        a: f32,
        b: u32,
        c: [f32x3_cpu],
    }

    assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
}

#[test]
#[rustfmt::skip] fn top_level_align_ignore() {
    #[derive(sm::GpuLayout)]
    struct OnGpu { // size=16, align=16
        a: f32x4, // size=16, align=16
    }

    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu {       // size=16, align=4
        a: f32x4_align4, // size=16, align=4
    }

    // the alignment on the top level of the layout doesn't matter.
    // two layouts are only considered different if an alignment mismatch
    // leads to different offsets of fields or array elements
    assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
}

#[test]
#[rustfmt::skip] fn struct_align_round_up() {
    #[derive(sm::GpuLayout)]
    struct OnGpu { // size=round_up(16, 12)=16, align=16
        a: f32x3, // size=12, align=16
    }

    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu { // size=12, align=4
        a: f32x3_align4,
    }
    assert_ne!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
    assert!(gpu_layout::<OnGpu>().byte_size() == Some(16));
    assert!(gpu_layout::<OnGpu>().align().as_u32() == 16);
    assert!(cpu_layout::<OnCpu>().byte_size() == Some(12));
    assert!(cpu_layout::<OnCpu>().align().as_u32() == 4);
}

#[test]
fn unsized_struct_nested_vec3_align_layout_eq() {
    #[derive(sm::GpuLayout)]
    struct InnerGpu {
        a: f32x1,
        b: u32x1,
    }

    #[derive(sm::CpuLayout, Clone)]
    #[repr(C)]
    struct InnerCpu {
        a: f32,
        b: u32,
    }

    #[derive(sm::GpuLayout)]
    struct OnGpu {
        a: f32x1,
        b: u32x1,
        c: sm::Array<sm::Struct<InnerGpu>>,
    }

    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu {
        a: f32,
        b: u32,
        c: [InnerCpu],
    }

    assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
}

#[test]
fn unsized_array_layout_eq() {
    assert_eq!(gpu_layout::<sm::Array<f32x1>>(), cpu_layout::<[f32]>());
    assert_eq!(gpu_layout::<sm::Array<f32x3>>(), cpu_layout::<[f32x3_cpu]>());
    assert_ne!(gpu_layout::<sm::Array<f32x3>>(), cpu_layout::<[f32x3_align4]>());
    assert_ne!(gpu_layout::<sm::Array<f32x3>>(), cpu_layout::<[f32x3_size16]>());
    assert_ne!(gpu_layout::<sm::Array<f32x3>>(), cpu_layout::<[f32x3_size32]>());
}

#[test]
fn layouts_mismatch() {
    #[derive(sm::GpuLayout)]
    struct OnGpuMore {
        a: f32x1,
        b: u32x1,
        c: i32x1,
        d: i32x1,
    }

    #[derive(sm::GpuLayout)]
    struct OnGpuLess {
        a: f32x1,
        b: u32x1,
    }

    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu {
        a: f32,
        b: u32,
        c: i32,
    }

    assert_ne!(gpu_layout::<OnGpuLess>(), cpu_layout::<OnCpu>());
    assert_ne!(gpu_layout::<OnGpuMore>(), cpu_layout::<OnCpu>());
}

#[test]
fn external_vec_type() {
    // using duck-traiting just so that the proc-macro uses `CpuLayoutExt::layout()`
    pub mod my_mod {
        use super::rust_layout_with_shame_semantics;
        use shame::gpu_layout;
        use shame as sm;
        use sm::aliases::*;
        use sm::GpuLayout as _;

        pub trait CpuLayoutExt {
            fn cpu_layout() -> shame::TypeLayout;
        }

        impl CpuLayoutExt for glam::Vec4 {
            fn cpu_layout() -> shame::TypeLayout { gpu_layout::<f32x4>() }
        }

        impl CpuLayoutExt for glam::Vec3 {
            fn cpu_layout() -> shame::TypeLayout { rust_layout_with_shame_semantics::<Self, f32x3>() }
        }
    }

    #[derive(sm::GpuLayout)]
    struct OnGpu {
        a: f32x4,
        b: f32x4,
    }

    use my_mod::CpuLayoutExt as _; // makes `glam::Vec4::layout()` compile in the derive generated code.
    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu {
        a: glam::Vec4,
        b: glam::Vec4,
    }

    assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());

    #[derive(sm::GpuLayout)]
    struct OnGpu2 {
        a: f32x3,
        b: f32x3,
        #[align(16)]
        c: f32x4,
    }

    #[derive(sm::CpuLayout)]
    #[repr(C)]
    struct OnCpu2 {
        a: glam::Vec3,
        b: glam::Vec3,
        c: glam::Vec4,
    }

    assert_ne!(gpu_layout::<OnGpu2>(), cpu_layout::<OnCpu2>());

    // TODO: delete or use compile fail test crate like trybuild to make
    // sure that align and size attributes aren't allowed on packed structs.
    // #[derive(sm::GpuLayout)]
    // #[gpu_repr(packed)]
    // struct OnGpu2Packed {
    //     a: f32x3,
    //     b: f32x3,
    //     #[align(16)]
    //     c: f32x4,
    // }

    // assert_eq!(gpu_layout::<OnGpu2Packed>(), cpu_layout::<OnCpu2>());
}

#[test]
#[rustfmt::skip] fn gpu_repr_packed_test() {
    {
        #[derive(sm::GpuLayout)]
        #[gpu_repr(packed)]
        struct OnGpu {
            pos: f32x3,
            nor: f32x3,
            uv : f32x2,
        }

        #[derive(sm::CpuLayout)]
        #[repr(C)]
        struct OnCpu {
            pos: f32x3_align4,
            nor: f32x3_align4,
            uv : f32x2_align4,
        }

        assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
    }
    {
        // TODO: delete or use compile fail test crate like trybuild to make
        // sure that align and size attributes aren't allowed on packed structs.
        // #[derive(sm::GpuLayout)]
        // #[gpu_repr(packed)]
        // struct OnGpu {
        //     pos: f32x3,
        //     nor: f32x3,
        //     #[align(8)] uv : f32x2,
        // }

        // #[derive(sm::CpuLayout)]
        // #[repr(C)]
        // struct OnCpu {
        //     pos: f32x3_align4,
        //     nor: f32x3_align4,
        //     uv : f32x2_cpu,
        // }

        // assert_eq!(gpu_layout::<OnGpu>(), cpu_layout::<OnCpu>());
        // enum __ where OnGpu: sm::VertexLayout {}
    }
}

#[rustfmt::skip]
#[test]
fn test_set_align_size() {
    #[derive(sm::GpuLayout)]
    struct OnGpu {
        a: f32x1,
        b: u32x1,
        c: sm::Array<i32x1, sm::Size<4>>,
    }

    let mut layouts = [
        gpu_layout::<f32x1>(),
        gpu_layout::<f32x4>(),
        gpu_layout::<f32x4x4>(),
        gpu_layout::<sm::packed::snorm16x2>(),
        gpu_layout::<sm::Array<f32x3>>(),
        gpu_layout::<sm::Struct<OnGpu>>(),
    ];

    for (i, lay) in layouts.iter_mut().enumerate() {
        let new_align = sm::U32PowerOf2::_128;
        assert_ne!(lay.align(), new_align, "#{i}: align of {lay} is already {new_align:?}, change new_align to make the test work");
        *lay.align_mut() = new_align;
        assert_eq!(lay.align(), new_align, "#{i}: align of {lay} is not {new_align:?}");

        let new_size = 128;
        assert_ne!(lay.byte_size(), Some(new_size), "#{i}: size of {lay} is already {new_size}, change new_size to make the test work");
        match lay.removable_byte_size_mut() {
            Ok(removable) => *removable = Some(new_size),
            Err(fixed) => *fixed = new_size,
        };
        assert_eq!(lay.byte_size(), Some(new_size), "#{i}: size of {lay} is not {new_size:?}");

        if let Ok(removable) = lay.removable_byte_size_mut() {
            *removable = None;
            assert_eq!(lay.byte_size(), None, "#{i}: size of {lay} is not None");
        };
    }
}

/// helper for defining a `TypeLayout` for a cpu type that
/// represents a `GpuSemantics` on the Gpu, but has alignment and size of `Layout` on the Cpu
pub fn rust_layout_with_shame_semantics<CpuType, GpuSemantics: sm::GpuLayout>() -> sm::TypeLayout {
    let mut layout = sm::gpu_layout::<GpuSemantics>();

    *layout.align_mut() = CpuType::CPU_ALIGNMENT;
    layout.set_byte_size(size_of::<CpuType>() as u64);

    // these are just here because we are testing
    assert_eq!(layout.align().as_u32(), align_of::<CpuType>() as u32);
    assert_eq!(layout.byte_size().map(|x| x as _), CpuType::CPU_SIZE);

    layout
}
