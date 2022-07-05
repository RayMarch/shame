
use shame;
use shame::aliases::*; // `float4` etc
use shame::rec::IntoRec as _; // `.rec()`
use shame::functions::*;
// alternatively just use
// use shame::prelude::*;

pub fn functions_example(mut feat: shame::ComputeFeatures) {

    // generic zero-to-one range for different tensor shapes (scalar, vec2, vec3, vec4)
    let zto1: std::ops::Range<float3> = zero_to_one();

    let alpha = 0.5f32.rec();

    let x = alpha.smoothstep(0.0, 1.0);
    let x = alpha.smoothrange(0.0..1.0);

    let x = alpha.smoothrange(0.5.plus_minus(0.1));
    let x = (alpha, alpha).rec().lerp((0.0, 1.0), (3.0, 8.0));

    //remap is useful for changing the domain of a value via scaling + translation.
    //commonly used when sampling from normal maps. The following is equivalent to "* 2.0 - 1.0"
    let x = alpha.remap(0..1, -1..1);

    //limit is a generalizied min/max/clamp which can be used with half open or closed ranges.
    //it generates the identical code to min/max/clamp
    let x = alpha.max(1.0);
    let x = alpha.limit(1.0..);

    let x = alpha.min(1.0);
    let x = alpha.limit(..1.0);

    let x = alpha.clamp(0.0, 1.0);
    let x = alpha.limit(0.0..1.0);

}


