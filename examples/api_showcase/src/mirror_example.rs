use std::marker::PhantomData;

/// this is an example of the `mirror` feature.
///
/// Make sure to add "mirror" to the shame features in your Cargo.toml file (e.g.: `shame = { ..., features = ["mirror"] }`)
/// if you want to use this feature
///
/// also add `bytemuck` to your dependencies, //TODO: figure out how to make this work without the user having to add bytemuck
///
/// You can use this feature in addition to your shame pipeline code to
/// generate:
///  - a `shame::Fields` struct from a [Pod](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html) struct that uses your preferred vector/matrix types
///  - a [Pod](Pod) struct that uses your preferred vector/matrix types from a `shame::Fields` struct
///
/// [Pod][https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html]
pub fn main() {
    // first we need to do some preparations:

    // call the `define_mirror_module` macro with the desired name of the module at the desired place
    // this will expand to something like:
    //
    // ```
    // mod my_mirror_mod {
    //     pub trait Host   { type Device: ... }
    //     pub trait Device { type Host: ... }
    // }
    // ```
    //
    // note: This is done to circumvent the orphan rule
    shame::define_mirror_module!(my_mirror_mod);

    // implement the Host or Device traits to establish relationships between types
    // - `Host` means non-shader representation of the type, such as `f32`
    // - `Device`means shader representation of the type, such as `shame::float`

    // f32 => shame::float
    impl my_mirror_mod::Host for f32 {
        type Device = shame::float;

        fn as_bytes(&self) -> &[u8] { bytemuck::cast_slice(std::slice::from_ref(self)) }
    }

    // u32 => shame::uint
    impl my_mirror_mod::Host for u32 {
        type Device = shame::uint;

        fn as_bytes(&self) -> &[u8] { bytemuck::cast_slice(std::slice::from_ref(self)) }
    }

    // [f32; 4] => shame::float4
    impl my_mirror_mod::Host for [f32; 4] {
        type Device = shame::float4;

        fn as_bytes(&self) -> &[u8] { bytemuck::cast_slice(std::slice::from_ref(self)) }
    }

    // now that the preparations are done, we can use the mirror feature's "host" macro
    // to generate a pair of Host/Device types.
    // #[shame::host(DeviceTypeName, path::to::mirror::module (optional))]

    // this will define two types `FooCpu` and `FooGpu`
    #[shame::host(FooGpu, my_mirror_mod)]
    struct FooCpu {
        a: f32,
        b: u32,
    }

    // alternatively, the path to the mirror module can be omitted if it is visible
    // from within the current module.
    use my_mirror_mod::*;
    #[shame::host(BarGpu)]
    struct BarCpu {
        a: f32,
        b: u32,
    }

    //now we can use FooCpu and FooGpu
    let foo_cpu = FooCpu { a: 0.0, b: 0 };

    let _ = shame::record_compute_pipeline(|mut f| {
        let foo_gpu: FooGpu = f.io.group().uniform_block();
    });

    // you can also define the name of the generated struct by adding a prefix or postfix to
    // the existing struct's name by providing a `*` at the start or end
    #[shame::host(*Gpu)]
    struct Baz {
        a: f32,
        b: u32,
    }

    let baz_cpu = Baz { a: 0.0, b: 0 };

    let _ = shame::record_compute_pipeline(|mut f| {
        let baz_gpu: BazGpu = f.io.group().uniform_block();
    });

    // The relation between FooCpu and FooGpu is expressed via the Host trait
    // it can be used in generic code to bridge the host/device gap with larger abstractions

    struct MyCrossCpuGpuAbstraction<T: Host>(PhantomData<T>);

    impl<T: Host> MyCrossCpuGpuAbstraction<T> {
        fn use_in_shader(&self, io: &mut impl shame::GenericPipelineIO) -> T::Device {
            //do something...
            io.next_group().storage() //just an example
        }

        fn new() -> Self { Self(PhantomData) }
    }

    let thing = MyCrossCpuGpuAbstraction::<FooCpu>::new();

    shame::record_compute_pipeline(|mut f| {
        let foo_gpu = thing.use_in_shader(&mut f.io);
    });
    // this can be used e.g. to hide the fact that there are two different
    // views on the data from the user entirely.

    //The `shame::device` attribute macro and the `my_mirror_mod::Device` trait work
    //exactly the opposite way. Specify a type for a shader and generate the host type.

    impl my_mirror_mod::Device for shame::uint {
        type Host = u32;
    }

    impl my_mirror_mod::Device for shame::float {
        type Host = f32;
    }

    #[shame::device(QuxCpu)]
    struct QuxGpu {
        a: shame::float,
        b: shame::uint,
    }

    let qux_cpu = QuxCpu { a: 0.0, b: 0 };

    let _ = shame::record_compute_pipeline(|mut f| {
        let qux_gpu: QuxGpu = f.io.group().uniform_block();
    });
}
