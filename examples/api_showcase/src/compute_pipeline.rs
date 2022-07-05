
use shame::prelude::*;
use shame::prettify::syntax_highlight_glsl;

pub fn main() {
    let out = shame::record_compute_pipeline(simple_compute_pipeline);
    println!("{}", out.to_string_colored());
}


pub fn simple_compute_pipeline(mut feat: shame::ComputeFeatures) {

    //in order to access thread IDs of different kinds, first set the workgroup size
    let ids = feat.dispatch.work_group([64, 64, 1]);

    ids.work_group_size(); // same as (64, 64, 1).rec()
    ids.total_work_groups_dispatched(); // how many 64x64x1 groups got dispatched

    // a "3" at the end of the function suggests it is a 3 component coordinate vector
    ids.local3(); //coords of this invocation within [0, 0, 0]..[64, 64, 1] 
    ids.global3(); // same as ids.work_group3() * ids.work_group_size() + ids.local3();
    ids.local(); // local3 but stretched out into a one dimensional index

    // create bind groups like this 
    // (not to be confused with work groups! Bind groups are called `sets` in glsl)
    let mut group0 = feat.io.group(); // bind group #0
    let mut group1 = feat.io.group(); //bind group #1
    // you can create bindgroups any time, they don't have to be created in one block.

    // you can add uniform blocks and read-only storage buffers like this
    let matrix: float4x4 = group0.uniform_block(); //group #0 binding #0
    let matrix: float4x4 = group0.storage(); //group #0 binding #1
    
    // shame encourages you to write your own additional wrappers around the group/binding 
    // calls, which also take care of some of your graphics-api specific setup. That way
    // layouts/buffer types can be inferred when creating pipelines and recording binding 
    // of resources into command buffers.
    // it is up to you to decide how far you let shame into your setup.

    //read/write storage buffers are unsafe to access. 
    let mut matrix: UnsafeAccess<float4x4> = group0.storage_mut(); //binding #2
    unsafe {
        let matrix = matrix.access_mut();
        //use `.set` instead of the `=` operator, as the latter will just replace what the `matrix` variable is referencing.
        matrix.set(id()); //id() is the identity matrix in this case
        *matrix += 5.0; // the `+=` `-=` `*=` etc... operators are fine to use though.
    }

    // derive shame::Fields will make this struct's layout usable in many shame functions.
    #[derive(shame::Fields, Clone, Copy)]
    struct Foo {
        a: float4,
        b: float2,
        c: uint2,
    }

    // `foo0` is 3 individual variables. The struct is invisible in the resulting shader code
    let foo0 = Foo {
        a: (1.0, 2.0, 3.0, 4.0).rec(),
        b: (5.0, 6.0).rec(),
        c: zero(),
    };

    // `foo1` is a `shame::Struct<Foo>` which is visible as a struct in the shader code.
    // it can be assigned with a single assignment operation, be put in arrays, etc.
    let foo1 = Foo {
        a: (1.0, 2.0, 3.0, 4.0).rec(),
        b: (5.0, 6.0).rec(),
        c: zero(),
    }.rec(); //<-- this .rec() turns the Foo into a `shame::Struct<Foo>`

    // alternatively you can explicitly call `shame::Struct::new`
    let foo2 = Struct::new(Foo {
        a: (1.0, 2.0, 3.0, 4.0).rec(),
        b: (5.0, 6.0).rec(),
        c: zero(),
    });

    // below is NOT how you make an array of `Foo`.
    //let foo_array: Array<Foo, _> = ...;
    
    //instead do this
    let foo_array: Array<Struct<Foo>, Size<16>> = Array::new([foo0; 16]);

    //or just this
    let foo_array = Array::new([foo0; 16]);

    //or just this
    let foo_array = [foo0; 16].rec();

    // Bar is a runtime-sized type, because it has a runtime-sized array at the end.
    #[derive(shame::Fields)]
    struct Bar {
        head: float4, 
        ///this is a compile time sized array, it has no restrictions of where it can be used
        sized_array: Array<uint, Size<128>>,
        ///this is a runtime-sized array. it can only be used as the last field of a storage buffer layout
        runtime_sized_array: Array<float4x4>,
    }

    // runtime-sized types can only be used in storage buffer bindings, not in uniform block bindings
    let bar0: Bar = group1.storage(); // group #1 binding #0
    let bar1: UnsafeAccess<Bar> = group1.storage_mut(); // group #1 binding #1
    //let bar2: Bar = group1.uniform_block(); //error
    
    // runtime-sized array bindings don't need to be declared as rust structs
    // their type can be described right in the binding call
    let bar3: Array<float4x4> = group1.storage(); // group #1 binding #2
    let bar4: Array<Struct<Foo>> = group1.storage(); // group #1 binding #3

    //indexing arrays is unsafe because shaders have no bounds checks
    let i = ids.local();
    unsafe {
        let matrix = bar3.at(i);
    }

    // the only exception is indexing shader compile-time sized arrays with
    // shader constants, which can be bounds checked when the shader is recorded.
    // use `at_const` insted of `at`
    bar0.sized_array.at_const(127); 

    // note how the index only needs to be constant for the shader, the
    // rust value doesn't need to be constant

    for i in 0..128 {
        bar0.sized_array.at_const(i); // this generates 128 subscript operations in the shader
        // or in other words, the loop gets unrolled.
    }

    // if the behavior of unrolling feels unintuitive to you, you can always double check your 
    // generated shader code every now and then. This way you can get a feel for which 
    // operations are unrolled and which aren't.
    // shame encourages you to write your own abstractions on top of the shame types, which
    // can "unroll" all kinds of interesting shader behavior!
}
