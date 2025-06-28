#![allow(dead_code, unused)]
//! Demonstration of the TypeLayout and TypeLayout Builder API.

use layout::{repr, Repr, SizedStruct};
use shame::{
    any::{
        self,
        layout::{
            self, FieldOptions, LayoutableSized, Len, RuntimeSizedArrayField, ScalarType, SizedField, SizedType,
            UnsizedStruct, Vector, GpuTypeLayout,
        },
        U32PowerOf2,
    },
    boolx1, f32x1, f32x2, f32x3, f32x4, gpu_layout, Array, GpuLayout, GpuSized, TypeLayout, VertexAttribute,
    VertexLayout,
};

fn main() {
    // We'll start by replicating this struct using `any::layout` types.
    #[derive(GpuLayout)]
    struct Vertex {
        position: f32x3,
        normal: f32x3,
        uv: f32x2,
    }

    // SizedStruct::new immediately takes the first field of the struct, because
    // structs need to have at least one field.
    let sized_struct = SizedStruct::new("Vertex", "position", f32x3::layoutable_type_sized())
        .extend("normal", f32x3::layoutable_type_sized())
        .extend("uv", f32x1::layoutable_type_sized());

    let storage = GpuTypeLayout::<repr::Storage>::new(sized_struct.clone());
    let packed = GpuTypeLayout::<repr::Packed>::new(sized_struct);
    assert_ne!(storage.layout(), packed.layout());

    // Does not exist:
    // let uniform = GpuTypeLayout::<repr::Uniform>::new(sized_struct.clone());

    // However we can try to upgrade a GpuTypeLayout::<repr::Storage>
    let uniform = GpuTypeLayout::<repr::Uniform>::try_from(storage.clone()).unwrap();

    // Which if it succeeds, guarantees:
    assert_eq!(storage.layout(), uniform.layout());

    // // Let's end on a pretty error message
    let mut sized_struct = SizedStruct::new("D", "a", f32x2::layoutable_type_sized())
    // This has align of 4, which is ok for putting into `Storage` but `Uniform` memory requires 16 byte alignment of arrays
        .extend("b", Array::<f32x1, shame::Size<1>>::layoutable_type_sized());

    let storage = GpuTypeLayout::<repr::Storage>::new(sized_struct.clone());
    let uniform_result = GpuTypeLayout::<repr::Uniform>::try_from(storage.clone());
    match uniform_result {
        Err(e) => println!("This error is a showcase:\n{}", e),
        Ok(u_layout) => println!("It unexpectedly worked, ohh no."),
    }
}
