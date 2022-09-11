
use shame::*;
use shame::tensor_type::Tensor;

mod common;

const EMPTY_VSH: &str = "#version 450


void main() {
    gl_Position = vec4(0.);
}";

const EMPTY_FSH: &str = "#version 450


void main() {
}";

#[test]
fn minimal_render_pipeline_record() {
    shame::record_render_pipeline(|feat: RenderFeatures| {
        feat.raster.rasterize_indexless(float4::default(), Default::default(), Default::default());
    });
}

#[test]
fn minimal_render_pipeline() {
    let out = record_render_pipeline(|feat: RenderFeatures| {
        feat.raster.rasterize_indexless(float4::default(), Default::default(), Default::default());
    });
    assert_eq_code!(&out.shaders_glsl.0, EMPTY_VSH);
    assert_eq_code!(&out.shaders_glsl.1, EMPTY_FSH);

    let info = RenderPipelineInfo {
        cull: Some(Default::default()),
        primitive_topology: Some(Default::default()),
        ..Default::default()
    };
    assert_eq!(out.info, info);
}

#[test]
fn vertex_buffer_added() {
    let out = record_render_pipeline(|mut feat: RenderFeatures| {

        #[derive(Fields)]
        struct Vertex {
            a: float4,
        }

        let _: Vertex = feat.io.vertex_buffer();

        feat.raster.rasterize_indexless(float4::default(), Default::default(), Default::default());
    });
    assert_eq_code!(&out.shaders_glsl.0, "
        #version 450

        layout(location=0) in vec4 a;

        void main() {
            gl_Position = vec4(0.);
        }
    ");
    assert_eq_code!(&out.shaders_glsl.1, EMPTY_FSH);

    let info = RenderPipelineInfo {
        vertex_buffers: vec![
            VertexBufferInfo { 
                step_mode: VertexStepMode::Vertex, 
                attributes: vec![
                    AttributeInfo { location: 0, type_: Tensor::vec4() },
                ]
            },
        ],
        cull: Some(Default::default()),
        primitive_topology: Some(Default::default()),
        ..Default::default()
    };
    assert_eq!(out.info, info);
}

#[test]
fn instance_buffer_added() {
    let out = record_render_pipeline(|mut feat: RenderFeatures| {
        let _: float4 = feat.io.instance_buffer();
        feat.raster.rasterize_indexless(float4::default(), Default::default(), Default::default());
    });
    assert_eq_code!(&out.shaders_glsl.0, "
        #version 450


        layout(location=0) in vec4 val;


        void main() {
            gl_Position = vec4(0.);
        }
    ");
    assert_eq_code!(&out.shaders_glsl.1, EMPTY_FSH);

    let info = RenderPipelineInfo {
        vertex_buffers: vec![
            VertexBufferInfo { 
                step_mode: VertexStepMode::Instance, 
                attributes: vec![
                    AttributeInfo { location: 0, type_: Tensor::vec4() },
                ]
            },
        ],
        cull: Some(Default::default()),
        primitive_topology: Some(Default::default()),
        ..Default::default()
    };
    assert_eq!(out.info, info);
}

#[test]
fn vertex_buffer_mixed_interleaved() {
    let out = record_render_pipeline(|mut feat: RenderFeatures| {

        #[derive(Fields)]
        struct Vertex {
            a: float2,
            b: float3,
        }

        let _: Vertex = feat.io.vertex_buffer();
        let _: float4 = feat.io.vertex_buffer();

        feat.raster.rasterize_indexless(float4::default(), Default::default(), Default::default());
    });
    assert_eq_code!(&out.shaders_glsl.0, "
        #version 450

        layout(location=0) in vec2 a;
        layout(location=1) in vec3 b;
        layout(location=2) in vec4 val;

        void main() {
            gl_Position = vec4(0.);
        }
    ");
    assert_eq_code!(&out.shaders_glsl.1, EMPTY_FSH);

    let info = RenderPipelineInfo {
        vertex_buffers: vec![
            VertexBufferInfo { 
                step_mode: VertexStepMode::Vertex, 
                attributes: vec![
                    AttributeInfo { location: 0, type_: Tensor::vec2() },
                    AttributeInfo { location: 1, type_: Tensor::vec3() },
                ]
            },
            VertexBufferInfo { 
                step_mode: VertexStepMode::Vertex, 
                attributes: vec![
                    AttributeInfo { location: 2, type_: Tensor::vec4() },
                ]
            },
        ],
        cull: Some(Default::default()),
        primitive_topology: Some(Default::default()),
        ..Default::default()
    };
    assert_eq!(out.info, info);
}

#[test]
fn vertex_attribute_matrix_locations() {
    let out = record_render_pipeline(|mut feat: RenderFeatures| {

        #[derive(Fields)]
        struct Vertex {
            a: float4x4,
            b: float4,
        }

        let _: Vertex = feat.io.vertex_buffer();

        feat.raster.rasterize_indexless(float4::default(), Default::default(), Default::default());
    });
    assert_eq_code!(&out.shaders_glsl.0, "
        #version 450

        layout(location=0) in mat4 a;
        layout(location=4) in vec4 b;

        void main() {
            gl_Position = vec4(0.);
        }
    "    
    
);
    assert_eq_code!(&out.shaders_glsl.1, EMPTY_FSH);

    let info = RenderPipelineInfo {
        vertex_buffers: vec![
            VertexBufferInfo { 
                step_mode: VertexStepMode::Vertex, 
                attributes: vec![
                    AttributeInfo { location: 0, type_: Tensor::mat4() },
                    AttributeInfo { location: 4, type_: Tensor::vec4() },
                ]
            }
        ],
        cull: Some(Default::default()),
        primitive_topology: Some(Default::default()),
        ..Default::default()
    };
    assert_eq!(out.info, info);
}

#[test]
fn index_buffer_added() {
    let out = record_render_pipeline(|mut feat: RenderFeatures| {
        let ib: TriangleList<u32> = feat.io.index_buffer();
        feat.raster.rasterize(float4::default(), Default::default(), ib);
    });
    assert_eq_code!(&out.shaders_glsl.0, EMPTY_VSH);
    assert_eq_code!(&out.shaders_glsl.1, EMPTY_FSH);

    let info = RenderPipelineInfo {
        index_dtype: Some(IndexDType::U32),
        primitive_topology: Some(PrimitiveTopology::TriangleList),
        cull: Some(Default::default()),
        ..Default::default()
    };
    assert_eq!(out.info, info);
}

#[test]
fn empty_bind_group_added() {
    let out = record_render_pipeline(|mut feat: RenderFeatures| {
        feat.io.group();
        feat.raster.rasterize_indexless(float4::default(), Default::default(), Default::default());
    });
    assert_eq_code!(&out.shaders_glsl.0, EMPTY_VSH);
    assert_eq_code!(&out.shaders_glsl.1, EMPTY_FSH);

    let info = RenderPipelineInfo {
        bind_groups: vec![
            BindGroupInfo { index: 0, bindings: vec![] }
        ],
        cull: Some(Default::default()),
        primitive_topology: Some(Default::default()),
        ..Default::default()
    };
    assert_eq!(out.info, info);
}

#[test]
fn uniform_buffer_added() {
    let out = record_render_pipeline(|mut feat: RenderFeatures| {
        let mut group0 = feat.io.group();
        let _: float4 = group0.uniform_block();
        feat.raster.rasterize_indexless(float4::default(), Default::default(), Default::default());
    });

    let vsh = "
        #version 450

        layout(std140, set=0, binding=0) uniform S0_B0 {
            vec4 val;
        };

        void main() {
            gl_Position = vec4(0.);
        }
    ";

    let fsh = "
        #version 450

        layout(std140, set=0, binding=0) uniform S0_B0 {
            vec4 val;
        };

        void main() {
        }
    ";
    assert_eq_code!(&out.shaders_glsl.0, vsh);
    assert_eq_code!(&out.shaders_glsl.1, fsh);

    let info = RenderPipelineInfo {
        bind_groups: vec![
            BindGroupInfo { index: 0, bindings: vec![
                BindingInfo { 
                    binding: 0, 
                    binding_type: BindingType::UniformBuffer, 
                    visibility: StageFlags::vertex_fragment(), //no fine grained stage tracking implemented yet 
                }
            ] }
        ],
        cull: Some(Default::default()),
        primitive_topology: Some(Default::default()),
        ..Default::default()
    };
    assert_eq!(out.info, info);
}

#[test]
#[should_panic]
fn panics_on_unused_rasterizer() {
    shame::record_render_pipeline(|_| {});
}

#[test]
fn mutating_expr_propagates_not_availableness() {

    macro_rules! test_tensor_types {
        (
            $(
                $dtype: ty, $shape: ty => ($($op_assign: ident,)+); 
            )*
        ) => {$(
            shame::record_render_pipeline(|mut feat| {
                type TenT = Ten<$shape, $dtype>;
                
                let i: TriangleList<u32> = feat.io.index_buffer();
                let v = feat.io.vertex_buffer::<TenT>().copy();
        
                let poly = feat.raster.rasterize(float4::default(), Default::default(), i);
        
                let one = Ten::<$shape, f32>::one();
                let f: TenT = poly.flat(one).cast();
                
                let u = TenT::one();
                assert_eq!(u.stage(), Stage::Uniform);
                assert!(u.as_any().is_available());
                
                $({
                    let mut u = TenT::one();
                    u.$op_assign(f);
                    assert_eq!(u.stage(), Stage::Fragment);
                    assert_eq!(shame::shader::is_fragment_shader(), u.as_any().is_available());
            
                    let mut u = TenT::one();
                    u.$op_assign(v);
                    assert_eq!(u.stage(), Stage::Vertex);
                    assert_eq!(shame::shader::is_vertex_shader(), u.as_any().is_available());
                })*
            });
        )*};
    }

    // invokes the macro above for certain `DType`s + `Shape`s
    macro_rules! test_tensor_dtypes {
        ($($dtype: ty => ($($op_assign: ident,)+); )*) => {
            test_tensor_types!{$($dtype, scal => ($($op_assign,)+);)*}
            test_tensor_types!{$($dtype, vec2 => ($($op_assign,)+);)*}
            test_tensor_types!{$($dtype, vec3 => ($($op_assign,)+);)*}
            test_tensor_types!{$($dtype, vec4 => ($($op_assign,)+);)*}
        }
    }

    use std::ops::*;
    // invokes the macro above for certain `DType`s
    test_tensor_dtypes!{
        f32  => (add_assign, sub_assign, mul_assign, div_assign,);
        i32  => (add_assign, sub_assign, mul_assign, div_assign, rem_assign, bitor_assign, bitand_assign,);
        bool => (add_assign, sub_assign, mul_assign, div_assign,);
    }

    
}