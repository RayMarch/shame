mod common;
use common::*;
use crate::common::{test_image::TestImage2D};
use shame_wgpu as sm;

#[test]
fn basic_compute_pipeline_test() {
    let gpu = &basic_test_setup().unwrap().gpu;

    assert_eq!(
        [1.0; 64],
        init_array_via_gpu_compute(gpu, [1, 1], [8, 8], |dispatch, arr| {
            arr.at(dispatch.workgroup.thread_id).set(1.0);
        })
    );
}

#[test]
fn basic_render_pipeline_test() {
    let gpu = &basic_test_setup().unwrap().gpu;

    let expected = TestImage2D::<sm::tf::Rg8Uint, 8, 8>::try_from_str(
        "
            R                G
            ░░░░░░░░▓▓░░░░░░ ░░░░░░░░▒▒░░░░░░
            ░░░░░░░░▓▓▓▓░░░░ ░░░░░░░░▒▒▒▒░░░░
            ░░░░░░░░▓▓▓▓▓▓░░ ░░░░░░░░▒▒▒▒▒▒░░
            ░░░░░░░░▓▓▓▓▓▓▓▓ ░░░░░░░░▒▒▒▒▒▒▒▒
            ░░░░░░░░░░░░░░░░ ░░░░░░░░░░░░░░░░
            ░░░░░░░░░░░░░░░░ ░░░░░░░░░░░░░░░░
            ░░░░░░░░░░░░░░░░ ░░░░░░░░░░░░░░░░
            ░░░░░░░░░░░░░░░░ ░░░░░░░░░░░░░░░░
        ",
    )
    .unwrap();

    let actual = TestImage2D::render_on_gpu(gpu, /*indices*/ 0..3, /*instances*/ 0..1, |drawcall| {
        let vi = drawcall.vertices.index;
        let triangle = sm::Array::new([(0.0, 0.0), (1.1, 0.0), (0.0, 1.1)]);

        let fragments = drawcall
            .vertices
            .assemble(triangle.at(vi), sm::Draw::triangle_list(sm::Ccw))
            .rasterize(sm::Accuracy::Reproducible);

        fragments
            .attachments
            .color_iter()
            .next::<sm::tf::Rg8Uint>()
            .set((255_u32, 128_u32));
    });

    assert_eq!(expected, actual);
}
