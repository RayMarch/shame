
use shame::*;
mod common;

#[test]
fn minimal_compute_pipeline() {
    record_compute_pipeline(|_| ());
}

#[test]
fn minimal_compute_pipeline_glsl() {
    let out = record_compute_pipeline(|_| ());
    let csh = "
        #version 450

        layout(
            local_size_x = 1, 
            local_size_y = 1, 
            local_size_z = 1
        ) in;

        void main() {
        }
    ";

    assert_eq_code!(csh, &out.shader_glsl);

}

#[test]
fn compute_pipeline_work_group() {
    let out = record_compute_pipeline(|feat| {
        feat.dispatch.work_group([2, 3, 4]);
    });
    let csh = "
        #version 450

        layout(
            local_size_x = 2, 
            local_size_y = 3, 
            local_size_z = 4
        ) in;

        void main() {
        }
    ";

    assert_eq_code!(csh, &out.shader_glsl);
    assert_eq!(out.info, ComputePipelineInfo {
        work_group_size: Some([2, 3, 4]),
        ..Default::default()
    });
}
