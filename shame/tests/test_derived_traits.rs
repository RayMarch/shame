#![allow(non_camel_case_types, unused)]
use pretty_assertions::{assert_eq, assert_ne};

use shame::{self as sm, NoBools};
use sm::{f32x1, i32x1, u32x1, CpuLayout, GpuLayout};
use static_assertions::{assert_impl_all, assert_impl_one, assert_not_impl_all, assert_not_impl_any};

macro_rules! assert_impls {
    (
        $(
            impl
            $(($trait:path) for $($type: ty),*;)?
            $(!($ntrait:path) for $($ntype: ty),*;)?
        )+
    ) => {
        $(
            $($(assert_not_impl_all!($ntype: $ntrait);)?)*
            $($(assert_impl_all!($type: $trait);)?)*
        )*
    };
}

#[test]
fn assert_derived_traits() {
    {
        #[derive(sm::GpuLayout)]
        struct T {
            a: f32x1,
            b: u32x1,
            c: i32x1,
        }

        assert_impls!(
            impl (sm::GpuLayout   ) for T;

            impl (sm::GpuStore    ) for T; // no packed vec, or gpu_repr(packed)

            impl (sm::VertexLayout) for T; // only vecs
            impl (sm::BufferFields) for T; // support buffer bindings
            impl (sm::SizedFields ) for T; // BufferFields + all fields are sized

            impl (sm::GpuSized    ) for T; // GpuAligned + no unsized arrays
            impl (sm::GpuAligned  ) for T; // no handles/ptrs/refs

            impl (sm::NoBools     ) for T;
            impl (sm::NoAtomics   ) for T;
            impl (sm::NoHandles   ) for T;
        );
    }

    {
        #[derive(sm::GpuLayout)]
        #[gpu_repr(packed)]
        struct T {
            a: f32x1,
            b: u32x1,
            c: i32x1,
        }

        #[derive(sm::GpuLayout)]
        #[gpu_repr(packed)]
        struct R {
            a: sm::packed::unorm8x4,
            b: sm::packed::u8x2,
            c: sm::packed::u16x2,
        }

        assert_impls!(
            impl  (sm::GpuLayout   ) for T, R;

            impl !(sm::GpuStore    ) for T, R;

            impl  (sm::VertexLayout) for T, R;
            impl !(sm::BufferFields) for T, R;
            impl !(sm::SizedFields ) for T, R;

            impl  (sm::GpuSized    ) for T, R;
            impl  (sm::GpuAligned  ) for T, R;

            impl  (sm::NoBools     ) for T, R;
            impl  (sm::NoAtomics   ) for T, R;
            impl  (sm::NoHandles   ) for T, R;
        );
    }

    {
        #[derive(sm::GpuLayout)]
        struct T {
            a: sm::packed::unorm8x4,
            b: u32x1,
            c: i32x1,
        }

        #[derive(sm::GpuLayout)]
        struct R {
            b: u32x1,
            c: i32x1,
            a: sm::packed::unorm8x4,
        }

        assert_impls!(
            impl  (sm::GpuLayout   ) for T, R;

            impl !(sm::GpuStore    ) for T, R;

            impl  (sm::VertexLayout) for T, R;
            impl !(sm::BufferFields) for T, R;
            impl !(sm::SizedFields ) for T, R;

            impl  (sm::GpuSized    ) for T, R;
            impl  (sm::GpuAligned  ) for T, R;

            impl  (sm::NoBools     ) for T, R;
            impl  (sm::NoAtomics   ) for T, R;
            impl  (sm::NoHandles   ) for T, R;
        );
    }

    {
        #[derive(sm::GpuLayout)]
        struct T {
            a: f32x1,
            b: u32x1,
            c: sm::Array<i32x1>,
        }

        #[derive(sm::GpuLayout)]
        struct R {
            c: sm::Array<i32x1>,
        }

        assert_impls!(
            impl  (sm::GpuLayout   ) for T, R;

            impl  (sm::GpuStore    ) for T, R;

            impl !(sm::VertexLayout) for T, R;
            impl  (sm::BufferFields) for T, R;
            impl !(sm::SizedFields ) for T, R;

            impl !(sm::GpuSized    ) for T, R;
            impl  (sm::GpuAligned  ) for T, R;

            impl  (sm::NoBools     ) for T, R;
            impl  (sm::NoAtomics   ) for T, R;
            impl  (sm::NoHandles   ) for T, R;
        );
    }
}
