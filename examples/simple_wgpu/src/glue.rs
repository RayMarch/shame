use std::ops::Index;

pub fn index_format_of<T: shame::IndexFormat>() -> wgpu::IndexFormat {
    match T::INDEX_DTYPE {
        shame::IndexDType::U32 => wgpu::IndexFormat::Uint32,
        shame::IndexDType::U16 => wgpu::IndexFormat::Uint16,
    }
}

fn from_binding_info(shame: &shame::BindingInfo) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: shame.binding,
        visibility: make_shader_stage_flags(shame.visibility),
        ty: match shame.binding_type {
            shame::BindingType::Sampler              => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            shame::BindingType::TextureCombinedSampler => panic!("texture-combined samplers are not supported. Use separate sampler/texture bindings instead."),
            shame::BindingType::ShadowSampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
            shame::BindingType::UniformBuffer => wgpu::BindingType::Buffer { 
                ty: wgpu::BufferBindingType::Uniform, 
                has_dynamic_offset: false, //TODO: Add
                min_binding_size: None 
            },
            shame::BindingType::ReadWriteStorageBuffer => wgpu::BindingType::Buffer { 
                ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                has_dynamic_offset: false, //TODO: Add
                min_binding_size: None 
            },
            shame::BindingType::ReadOnlyStorageBuffer => wgpu::BindingType::Buffer { 
                ty: wgpu::BufferBindingType::Storage { read_only: true }, 
                has_dynamic_offset: false, //TODO: Add
                min_binding_size: None 
            },
            shame::BindingType::Texture => wgpu::BindingType::Texture { 
                sample_type: wgpu::TextureSampleType::Float { filterable: true }, 
                view_dimension: wgpu::TextureViewDimension::D2, 
                multisampled: false //TODO: Add
            },
            shame::BindingType::UniformTexelBuffer => unimplemented!(),
            shame::BindingType::ReadWriteStorageTexelBuffer => unimplemented!(),
            shame::BindingType::ReadOnlyStorageTexelBuffer => unimplemented!(),
            shame::BindingType::ReadWriteStorageImage => unimplemented!(),
            shame::BindingType::ReadOnlyStorageImage => unimplemented!(),
        },
        count: None,
    }
}

fn make_bind_group_layout(shame: &shame::BindGroupInfo, device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = shame.bindings.iter().map(from_binding_info).collect();

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None, //TODO: add
        entries: &entries,
    })
}

fn make_shader_stage_flags(shame: shame::StageFlags) -> wgpu::ShaderStages {
    let mut stages = wgpu::ShaderStages::empty();
    if shame.vertex   {stages |= wgpu::ShaderStages::VERTEX  };
    if shame.fragment {stages |= wgpu::ShaderStages::FRAGMENT};
    if shame.compute  {stages |= wgpu::ShaderStages::COMPUTE };
    stages
}

fn make_push_constant_range(shame: &shame::PushConstantInfo) -> wgpu::PushConstantRange {
    wgpu::PushConstantRange {
        stages: make_shader_stage_flags(shame.visibility),
        range: 0..(shame.type_.byte_size() as u32),
    }
}

fn empty_push_constant_range() -> wgpu::PushConstantRange {
    wgpu::PushConstantRange {
        stages: wgpu::ShaderStages::empty(), 
        range: 0..0
    }
}

fn make_render_pipeline_layout(shame: &shame::RenderPipelineInfo, device: &wgpu::Device) -> wgpu::PipelineLayout {
    //TODO: i think these layouts should already exist and be passed in here... not sure though
    let layouts: Vec<_> = shame.bind_groups.iter().map(|x| make_bind_group_layout(x, device)).collect();

    let range = shame.push_constant.as_ref()
    .map(make_push_constant_range);

    let bind_group_layouts: Vec<&_> = layouts.iter().map(|x| x).collect();

    // Some => &[T], None => &[]
    let slice = range.as_ref().map_or([].as_slice(), std::slice::from_ref);

    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &bind_group_layouts,
        push_constant_ranges: slice,
    })
}

fn make_compute_pipeline_layout(shame: &shame::ComputePipelineInfo, device: &wgpu::Device) -> wgpu::PipelineLayout {
    //TODO: i think these layouts should already exist and be passed in here... not sure though
    let layouts: Vec<_> = shame.bind_groups.iter().map(|x| make_bind_group_layout(x, device)).collect();

    let range = shame.push_constant.as_ref()
    .map(make_push_constant_range)
    .unwrap_or_else(|| empty_push_constant_range());

    let bind_group_layouts: Vec<&_> = layouts.iter().map(|x| x).collect();

    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &bind_group_layouts,
        push_constant_ranges: &[range],
    })
}

fn make_vertex_format(shame: &shame::tensor_type::Tensor) -> wgpu::VertexFormat {
    use shame::tensor_type::*;
    use wgpu::VertexFormat as wgpu;
    let Tensor {dtype, shape} = shame;
    match dtype {
        DType::Bool => panic!("bool-tensor vertex attributes are not supported"),
        DType::F32 => match shape {
            Shape::Scalar => wgpu::Float32,
            Shape::Vec(2) => wgpu::Float32x2,
            Shape::Vec(3) => wgpu::Float32x3,
            Shape::Vec(4) => wgpu::Float32x4,
            Shape::Mat(_, _) => panic!("matrix vertex-attributes not supported by wgpu at the moment"),
            _ => panic!("malformed vertex attribute: {shame}")
        },
        DType::F64 => match shape {
            Shape::Scalar => wgpu::Float64,
            Shape::Vec(2) => wgpu::Float64x2,
            Shape::Vec(3) => wgpu::Float64x3,
            Shape::Vec(4) => wgpu::Float64x4,
            Shape::Mat(_, _) => panic!("matrix vertex-attributes not supported by wgpu at the moment"),
            _ => panic!("malformed vertex attribute: {shame}")
        },
        DType::I32 => match shape {
            Shape::Scalar => wgpu::Sint32,
            Shape::Vec(2) => wgpu::Sint32x2,
            Shape::Vec(3) => wgpu::Sint32x3,
            Shape::Vec(4) => wgpu::Sint32x4,
            Shape::Mat(_, _) => panic!("matrix vertex-attributes not supported by wgpu at the moment"),
            _ => panic!("malformed vertex attribute: {shame}")
        },
        DType::U32 => match shape {
            Shape::Scalar => wgpu::Uint32,
            Shape::Vec(2) => wgpu::Uint32x2,
            Shape::Vec(3) => wgpu::Uint32x3,
            Shape::Vec(4) => wgpu::Uint32x4,
            Shape::Mat(_, _) => panic!("matrix vertex-attributes not supported by wgpu at the moment"),
            _ => panic!("malformed vertex attribute: {shame}")
        },
    }
}

fn make_vertex_buffer_layouts<'a>(infos: &[shame::VertexBufferInfo], scratch: &'a mut Vec<wgpu::VertexAttribute>) -> Vec<wgpu::VertexBufferLayout<'a>> {
    assert!(scratch.is_empty());
    let mut ranges_strides = vec![];

    for info in infos {
        let mut stride: u64 = 0;
        let range_start = scratch.len();

        for shame::AttributeInfo {location, type_} in &info.attributes {
            let offset = stride;
            stride += type_.byte_size() as u64;
            scratch.push(wgpu::VertexAttribute {
                format: make_vertex_format(type_),
                offset,
                shader_location: *location,
            });
        }
        ranges_strides.push((range_start..scratch.len(), stride));
    }

    assert!(infos.len() == ranges_strides.len());

    infos.iter().zip(ranges_strides).map(|(info, (range, stride))| {
        let step_mode = match info.step_mode {
            shame::VertexStepMode::Vertex   => wgpu::VertexStepMode::Vertex,
            shame::VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
        };

            wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode,
                attributes: scratch.index(range),
            }
        }).collect()
}

fn make_vertex_state<'a, 'b>(shame: &shame::RenderPipelineInfo, module: &'a wgpu::ShaderModule, 
    scratch0: &'b mut Vec<wgpu::VertexAttribute>, 
    scratch1: &'a mut Vec<wgpu::VertexBufferLayout<'b>>
) -> wgpu::VertexState<'a> {
    assert!(scratch0.is_empty());
    assert!(scratch1.is_empty());

    let layouts = make_vertex_buffer_layouts(&shame.vertex_buffers, scratch0);
    *scratch1 = layouts;

    wgpu::VertexState {
        module,
        entry_point: "main",
        buffers: scratch1,
    }
}

fn make_primitive_state(shame: &shame::RenderPipelineInfo) -> wgpu::PrimitiveState {

    let mut is_strip = false;
    let topology = match shame.primitive_topology {
        Some(x) => match x {
            shame::PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
            shame::PrimitiveTopology::TriangleStrip => {
                is_strip = true;
                wgpu::PrimitiveTopology::TriangleStrip
            },
        },
        None => panic!("no primitive topology recorded"),
    };

    let front_face = wgpu::FrontFace::Ccw;
    let cull_mode = shame.cull.and_then(|cull| match cull {
        shame::Cull::Off => None,
        shame::Cull::CCW => Some(wgpu::Face::Front),
        shame::Cull::CW  => Some(wgpu::Face::Back),
    });

    wgpu::PrimitiveState {
        topology,
        strip_index_format: is_strip.then(|| match shame.index_dtype.expect("no index dtype recorded"){
            shame::IndexDType::U32 => wgpu::IndexFormat::Uint32,
            shame::IndexDType::U16 => wgpu::IndexFormat::Uint16
        }),
        front_face,
        cull_mode,
        unclipped_depth: false,
        polygon_mode: wgpu::PolygonMode::Fill,
        conservative: false,
    }
}

fn make_depth_stencil_state(shame: &shame::RenderPipelineInfo) -> Option<wgpu::DepthStencilState> {
    shame.depth_stencil_target.map(|target| {
        wgpu::DepthStencilState {
            format: match target {
                shame::DepthFormat::Depth32 => wgpu::TextureFormat::Depth32Float,
            },
            depth_write_enabled: shame.depth_write.expect("no information about whether depth write is enabled was recorded"),
            depth_compare: match shame.depth_test.expect("no information about depth test recorded") {
                shame::DepthTest::Always         => wgpu::CompareFunction::Always,
                shame::DepthTest::Never          => wgpu::CompareFunction::Never,
                shame::DepthTest::Less           => wgpu::CompareFunction::Less,
                shame::DepthTest::Equal          => wgpu::CompareFunction::Equal,
                shame::DepthTest::Greater        => wgpu::CompareFunction::Greater,
                shame::DepthTest::LessOrEqual    => wgpu::CompareFunction::LessEqual,
                shame::DepthTest::GreaterOrEqual => wgpu::CompareFunction::GreaterEqual,
                shame::DepthTest::NotEqual       => wgpu::CompareFunction::NotEqual,
            },
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }
    })
}

fn make_multisample_state(shame: &shame::RenderPipelineInfo) -> wgpu::MultisampleState {
    let mut sample_count = None;
    for target in &shame.color_targets {
        match sample_count {
            None => sample_count = Some(target.sample_count as u32),
            Some(count) => if target.sample_count as u32 != count {
                panic!("all color targets must have the same sample counts");
            },
        }
    }

    wgpu::MultisampleState {
        count: sample_count.unwrap_or(1),
        mask: !0,
        alpha_to_coverage_enabled: false,
    }
}

fn make_color_texture_format(shame: &shame::ColorFormat, known_surface_format: &Option<wgpu::TextureFormat>) -> wgpu::TextureFormat {
    use shame::ColorFormat as SC;
    use wgpu::TextureFormat as WTF;
    match shame {
        SC::R_8 => WTF::R8Unorm,
        SC::RG_88 => WTF::Rg8Unorm,
        SC::RGB_888 => unimplemented!(),
        SC::RGBA_8888 => WTF::Rgba8Unorm,
        SC::R_16 => WTF::R16Unorm,
        SC::RG_16_16 => WTF::Rg16Unorm,
        SC::RGB_16_16_16 => unimplemented!(),
        SC::RGBA_16_16_16_16 => WTF::Rgba16Unorm,
        SC::R_32_sFloat => WTF::R32Float,
        SC::RG_32_32_sFloat => WTF::Rg32Float,
        SC::RGB_32_32_32_sFloat => unimplemented!(),
        SC::RGBA_32_32_32_32_sFloat => WTF::Rgba32Float,
        SC::RGBA_16_16_16_16_sFloat => WTF::Rgba16Float,
        SC::R_8_sRGB => unimplemented!(),
        SC::RG_88_sRGB => unimplemented!(),
        SC::RGB_888_sRGB => unimplemented!(),
        SC::RGBA_8888_sRGB => WTF::Rgba8UnormSrgb,
        SC::BGRA_8888 => WTF::Bgra8Unorm,
        SC::BGR_888_sRGB => unimplemented!(),
        SC::BGRA_8888_sRGB => WTF::Bgra8UnormSrgb,
        SC::ABGR_2_10_10_10_Pack32 => unimplemented!(),
        SC::ARGB_2_10_10_10_Pack32 => unimplemented!(),
        SC::RGBA_Surface => known_surface_format.expect("surface format not provided, but RGBA_Surface requested by shader"),
    }
}

fn make_blend_state(shame: &Option<shame::Blend>) -> Option<wgpu::BlendState> {
    shame.map(|blend| {

        let convert_factor = |factor: shame::BlendFactor| -> wgpu::BlendFactor {
            use shame::BlendFactor as SF;
            use wgpu::BlendFactor as WF;
            match factor {
                SF::Zero => WF::Zero,
                SF::One => WF::One,
                SF::SourceColor => WF::Src,
                SF::OneMinusSourceColor => WF::OneMinusSrc,
                SF::DestinationColor => WF::Dst,
                SF::OneMinusDestinationColor => WF::OneMinusDst,
                SF::SourceAlpha => WF::SrcAlpha,
                SF::OneMinusSourceAlpha => WF::OneMinusSrcAlpha,
                SF::DestinationAlpha => WF::DstAlpha,
                SF::OneMinusDestinationAlpha => WF::OneMinusDstAlpha,
                SF::ConstantColor => WF::Constant,
                SF::OneMinusConstantColor => WF::OneMinusConstant,
                SF::ConstantAlpha => unimplemented!(), //TODO: this might be wrong to include altogether
                SF::OneMinusConstantAlpha => unimplemented!(), //TODO: this might be wrong to include altogether
            }
        };
        
        let convert_op = |op: shame::BlendOp| -> wgpu::BlendOperation {
            match op {
                shame::BlendOp::Add => wgpu::BlendOperation::Add,
                shame::BlendOp::Subtract => wgpu::BlendOperation::Subtract,
                shame::BlendOp::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
                shame::BlendOp::Min => wgpu::BlendOperation::Min,
                shame::BlendOp::Max => wgpu::BlendOperation::Max,
            }
        };

        wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: convert_factor(blend.rgb.src_factor),
                dst_factor: convert_factor(blend.rgb.dst_factor),
                operation: convert_op(blend.rgb.op),
            },
            alpha: wgpu::BlendComponent {
                src_factor: convert_factor(blend.a.src_factor),
                dst_factor: convert_factor(blend.a.dst_factor),
                operation: convert_op(blend.a.op),
            },
        }
    })
}

fn make_fragment_state<'a>(shame: &shame::RenderPipelineInfo, module: &'a wgpu::ShaderModule, scratch: &'a mut Vec<wgpu::ColorTargetState>, known_surface_format: &Option<wgpu::TextureFormat>) -> wgpu::FragmentState<'a> {

    for target in &shame.color_targets {
        scratch.push(wgpu::ColorTargetState {
            format: make_color_texture_format(&target.color_format, &known_surface_format),
            blend: make_blend_state(&target.blending),
            write_mask: wgpu::ColorWrites::default(), //TODO: add
        })
    }

    wgpu::FragmentState {
        module,
        entry_point: "main",
        targets: scratch,
    }
}

#[allow(unused)]
pub fn make_render_pipeline(shame: &shame::RenderPipelineRecording, device: &wgpu::Device, surface_format: Option<wgpu::TextureFormat>) -> wgpu::RenderPipeline {
    let layout = make_render_pipeline_layout(&shame.info, device);

    let (vsh, fsh) = &shame.shaders_glsl;

    let vsh_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("vertex shader"),
        source: wgpu::ShaderSource::Glsl {
            shader: vsh.into(),
            stage: naga::ShaderStage::Vertex,
            defines: Default::default(),
        },
    });

    let fsh_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("fragment shader"),
        source: wgpu::ShaderSource::Glsl {
            shader: fsh.into(),
            stage: naga::ShaderStage::Fragment,
            defines: Default::default(),
        },
    });

    //just store stuff in these until its no longer needed by RenderPipelineDescirptor
    let (mut scratch0, mut scratch1, mut scratch2) = (vec![], vec![], vec![]);

    let fragment = (!shame.info.color_targets.is_empty()).then(|| {
        make_fragment_state(&shame.info, &fsh_module, &mut scratch2, &surface_format)
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None, //TODO: add
        layout: Some(&layout),
        vertex: make_vertex_state(&shame.info, &vsh_module, &mut scratch0, &mut scratch1),
        primitive: make_primitive_state(&shame.info),
        depth_stencil: make_depth_stencil_state(&shame.info),
        multisample: make_multisample_state(&shame.info),
        fragment,
        multiview: None,
    })
}

#[allow(unused)]
pub fn make_compute_pipeline(shame: &shame::ComputePipelineRecording, device: &wgpu::Device) -> wgpu::ComputePipeline {
    let layout = make_compute_pipeline_layout(&shame.info, device);

    let shader_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("compute shader"),
        source: wgpu::ShaderSource::Glsl {
            shader: (&shame.shader_glsl).into(),
            stage: naga::ShaderStage::Compute,
            defines: Default::default(),
        },
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"), //TODO: add
        layout: Some(&layout),
        module: &shader_module,
        entry_point: "main",
    })
}
