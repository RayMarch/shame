use shame::{shader::is_fragment_shader, *};
use std::cell::Cell;

pub fn main() {
    let out = shame::record_render_pipeline(control_flow_example);
    println!("{}", out.to_string_colored());
}

#[shame::keep_idents]
fn control_flow_example(mut feat: RenderFeatures) {
    let condition = false;
    let mut a = 1.0.rec();
    let mut b = 1.0.rec();

    // the if statement below is evaluated at rust runtime, and doesn't make it
    // into the recorded shader. This can be thought of as conditional
    // compilation of the inside shader code
    if condition {
        //this code will not show up in the shader because `condition` is false
        a += 1.0;
    }

    // the code below does the same thing as the one above
    condition.then(|| {
        a += 1.0;
    });

    // let's convert the condition from a bool to a shame::boolean, which
    // is a type that can be influenced at shader runtime.
    // `.rec()` is the universal way to convert regular rust types to their
    // shader-recording counterparts.
    let condition = condition.rec();

    // this records an `if (condition) { }` statement in the shader
    condition.then(|| {
        // the shader code generated from the code below runs only if
        // `condition` is true in the shader,
        // however the *rust* code below runs *always*! This closure gets
        // run exactly once per shader recording.

        a += 1.0;
    });

    // this records an `if (condition) { } else { }` statement
    condition.then_else(
        || {
            // records the shader code for when `condition` is true
            a += 1.0;
        },
        || {
            // records the shader code for when `condition` is false
            b -= 1.0;
        },
    );

    // referencing the same variable in both blocks mutably is not possible,
    // because the borrow checker doesn't know that these blocks are executed
    // sequentially at rust-runtime. However since shame's `Rec` type instances
    // have reference-like semantics and `impl Copy`, you can just use the
    // `move` keyword to silence the error.
    condition.then_else(
        move || {
            //<--
            a += 1.0;
        },
        || {
            a -= 1.0;
        },
    );

    // this loop gets executed at rust-runtime. it cannot have `Rec` variables
    // as the loop bounds, and the loop will not show up as such in the final
    // shader
    for i in 0..10 {
        // the shader recording code below will execute 10 times and the
        // `+=` expression will appear 10 times in the shader, effectively
        // "unrolling" the for loop.
        a += i as f32;
    }

    // this loop accepts a range of `Rec` or values that can be turned into
    // `Rec` values, then executes the closure once to record its contents.
    // `for_range` supports only non-floating-point ranges. for floating point
    // ranges use `for_range_step` instead
    for_range(0..10, |i: int| {
        // the shader recording code below becomes the loop body
        a += i.cast();
    });

    // `for_range_step` takes an additional closure for the step amount.
    // Every recorded expression inside the step closure ends up in the
    // "increment" part of the resulting shader for loop.
    for_range_step(
        0.0..10.0,
        || 1.0,
        |i: float| {
            // the shader recording code below becomes the loop body
            a += i;
        },
    );

    // Only a copy of the range is used in the shader's for loop statement.
    for_range_step(
        a..b,
        || 1.0,
        |_| {
            b += 1.0; //incrementing b does not change the upper bound of the loop!
        },
    );

    // dynamic break conditions can instead be expressed via shame::break_()
    for_range_step(
        a..,
        || 1.0,
        |_| {
            b.ge(&10.0).then(|| break_());
            b += 1.0;
        },
    );

    // since `if (cond) break;` is a common pattern, there is a shorthand function
    // for that. Users can also build their own shorthands containing `break_()`
    // since it can be called from within other functions.
    for_range_step(
        a..,
        || 1.0,
        |_| {
            break_if(b.ge(&10.0));
            b += 1.0;
        },
    );

    // similarly, `continue_()` records a continue statement.
    // `continue_if` also exists.
    // calling any of these loop control flow statement recording functions
    // outside of a loop will cause an error.
    for_range_step(
        a..,
        || 1.0,
        |_| {
            continue_if(b.ge(&10.0));
            continue_();
            break_if(b.ge(&100.0));
            break_();
        },
    );
    //break_() //error

    // if the current available high-level for loop recording tools are not
    // fit for your needs, you can try using `Any::record_for_loop`, to build
    // your own abstractions on top of it, however be warned that it is not
    // trivial to get the semantics right!
    // see the definition of `from_3_to_8_in_tenths` for an example.
    from_3_to_8_in_tenths(|i| {
        b += i;
    });

    let poly = feat
        .raster
        .rasterize_indexless(float4::default(), Default::default(), Default::default());
    let frag_value = poly.plerp(1.234.rec());
    // at the time of writing, block-stage tracking is not fully finalized.
    // You can encounter situations where loops are recorded in too many shader
    // stages. You can fix it by adding a conditional code generation `if` that
    // restricts the shader stage of the contained code.
    // The code below shows a situation where this is still necessary.
    // (in the future this will be automatically fixed by making blocks more
    // stage-aware)
    if is_fragment_shader() {
        // loop range `0..` is not a per-fragment value, meaning
        // this loop shows up in both the vertex and fragment shader.
        // this wouldn't be a problem if the shader compiler could optimize the
        // entire loop away due to lack of side-effects, however this is an
        // infinite loop.
        for_range(0.., |_| {
            // break is only recorded in the fragment shader, because it depends on
            // a per-fragment condition.
            // this means the vertex-shader version of this loop never
            // terminates.
            break_if(frag_value.lt(&2.0))
        });
    }
}

/// example custom for loop abstraction
fn from_3_to_8_in_tenths(body: impl FnOnce(float)) {
    let i = Cell::new(None); // store i outside of the closures so it can
                             // be accessed in either closure

    Any::record_for_loop(
        || {
            //init block (may only record variable decls of same type)
            let val = 3.0.rec();
            val.aka("val"); // choose the name that this value should have
                            // in the generated shader
                            // (it will automatically avoid collisions with other previously
                            // defined variables by adding numbers behind the name)
            i.set(Some(val));
        },
        || {
            //condition block (must return type erased boolean)
            let i = i.get().unwrap(); //move i from init block closure to this closure
            let cond = i.lt(&8.0); //actual condition
            (cond.as_any(), cond.stage().into()) //type erase
        },
        || {
            //increment block (may only record expression statements)
            let mut i = i.get().unwrap(); //move i from init block closure to this closure
            i += 0.1;
        },
        || {
            //body block
            let mut i = i.get().unwrap(); //move i from init block closure to this closure
            body(i)
        },
    );
}
