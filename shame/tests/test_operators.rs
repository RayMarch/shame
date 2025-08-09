mod common;
use common::*;
use shame::ToGpuType as _;
use pretty_assertions::assert_eq;

#[test]
#[allow(non_snake_case)]
fn operator_arithmetic_f32_test() {
    let gpu = &basic_test_setup().unwrap().gpu;

    let values = [-1.5, -1.0, -0.75, -0.1, 0.0, 0.1, 0.75, 1.0, 1.5];

    // for every pair
    for a in values {
        for b in values {
            // fill `cpu_results` during recording, then compare the buffer
            // once the gpu result was downloaded
            let mut cpu_results = [0.0; 4 * 3];
            let gpu_results = init_array_via_gpu_compute(gpu, [1], [1], |_, gpu_results| {
                let mut it = 0..;

                let mut f32_assume_eq = |cpu_val, gpu_val| {
                    let i = it.next().unwrap();
                    cpu_results[i] = cpu_val;
                    gpu_results.at(i as u32).set(gpu_val);
                };

                let A = a.to_gpu();
                let B = b.to_gpu();

                // test whether the different combinations of gpu_type and rust_type
                // result in the same values as when doing the computation on the cpu

                // gpu_type x gpu_type
                f32_assume_eq(a + b, A + B);
                f32_assume_eq(a - b, A - B);
                f32_assume_eq(a / b, A / B);
                f32_assume_eq(a * b, A * B);

                // rust_type x gpu_type
                f32_assume_eq(a + b, a + B);
                f32_assume_eq(a - b, a - B);
                f32_assume_eq(a / b, a / B);
                f32_assume_eq(a * b, a * B);

                // gpu_type x rust_type
                f32_assume_eq(a + b, A + b);
                f32_assume_eq(a - b, A - b);
                f32_assume_eq(a / b, A / b);
                f32_assume_eq(a * b, A * b);
            });

            let nan_to_none = |x: f32| (!x.is_nan()).then_some(x);
            let cpu_results = cpu_results.map(nan_to_none);
            let gpu_results = gpu_results.map(nan_to_none);

            assert_eq!(cpu_results, gpu_results);
        }
    }
}
