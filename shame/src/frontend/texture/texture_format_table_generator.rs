
use wgpu::{AstcBlock::*, AstcChannel, Features, TextureFormat::{self, *}, TextureFormatFeatureFlags, TextureSampleType, TextureUsages};

/// excludes ASTC
const NON_GENERIC_FMTS: &[TextureFormat] = &[ 
    R8Unorm, R8Snorm, R8Uint, R8Sint, R16Uint, R16Sint, R16Unorm, R16Snorm, R16Float, Rg8Unorm, Rg8Snorm, Rg8Uint, Rg8Sint, R32Uint, R32Sint, R32Float, Rg16Uint, Rg16Sint, Rg16Unorm, Rg16Snorm, Rg16Float, Rgba8Unorm, Rgba8UnormSrgb, Rgba8Snorm, Rgba8Uint, Rgba8Sint, Bgra8Unorm, Bgra8UnormSrgb, Rgb9e5Ufloat, Rgb10a2Uint, Rgb10a2Unorm, Rg32Uint, Rg32Sint, Rg32Float, Rgba16Uint, Rgba16Sint, Rgba16Unorm, Rgba16Snorm, Rgba16Float, Rgba32Uint, Rgba32Sint, Rgba32Float, Stencil8, Depth16Unorm, Depth24Plus, Depth24PlusStencil8, Depth32Float, Depth32FloatStencil8, NV12, Bc1RgbaUnorm, Bc1RgbaUnormSrgb, Bc2RgbaUnorm, Bc2RgbaUnormSrgb, Bc3RgbaUnorm, Bc3RgbaUnormSrgb, Bc4RUnorm, Bc4RSnorm, Bc5RgUnorm, Bc5RgSnorm, Bc6hRgbUfloat, Bc6hRgbFloat, Bc7RgbaUnorm, Bc7RgbaUnormSrgb, Etc2Rgb8Unorm, Etc2Rgb8UnormSrgb, Etc2Rgb8A1Unorm, Etc2Rgb8A1UnormSrgb, Etc2Rgba8Unorm, Etc2Rgba8UnormSrgb, EacR11Unorm, EacR11Snorm, EacRg11Unorm, EacRg11Snorm,
];

const _ASTC_BLOCKS: &[wgpu::AstcBlock] = &[ 
    B4x4, B5x4, B5x5, B6x5, B6x6, B8x5, B8x6, B8x8, B10x5, B10x6, B10x8, B10x10, B12x10, B12x12,
];

const _ASTC_CHANNELS: &[wgpu::AstcChannel] = &[AstcChannel::Unorm, AstcChannel::UnormSrgb, AstcChannel::Hdr];

fn device_features(fmt: TextureFormat) -> wgpu::Features {
    Features::all_webgpu_mask() | 
    fmt.required_features() | 
    Features::BGRA8UNORM_STORAGE | 
    Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
}

fn main() {
    println!("impl_texture_formats! {{");
    let format_name_width = NON_GENERIC_FMTS.iter().map(|fmt| format!("{fmt:?}").chars().count()).max().unwrap_or(0);

    let sample_type_to_scalar_type = |sample_ty| {
        match sample_ty {
            TextureSampleType::Float { filterable: true } => "f32",
            TextureSampleType::Float { filterable: false } => "f32",
            TextureSampleType::Uint => "u32",
            TextureSampleType::Sint => "i32",
            TextureSampleType::Depth => "f32",
        }
    };

    let print_row = |fmt: TextureFormat| {
        print!("    {}: ", left_pad(&format!("{fmt:?}"), format_name_width));

        let device_features = device_features(fmt);

        let features = fmt.guaranteed_format_features(device_features);
        let usages = features.allowed_usages;
        let flags = features.flags;

        let sample_type_stype = fmt.sample_type(Some(wgpu::TextureAspect::All), Some(device_features));
        let sample_type_len = fmt.components_with_aspect(wgpu::TextureAspect::All);

        // (mod rgba, mod uint)
        print!("(");
        if fmt.has_color_aspect() && !sample_type_stype.is_none() {
            print!("{} mod, {} mod, ", match sample_type_len {
                1 => "r   ",
                2 => "rg  ",
                3 => "rgb ",
                4 => "rgba",
                _ => panic!("color format with {sample_type_len} color components")
            }, left_pad(match sample_type_stype {
                Some(TextureSampleType::Float { filterable: _ }) => "float",
                Some(TextureSampleType::Depth) => panic!("color format with depth sample type"),
                Some(TextureSampleType::Sint) => "int",
                Some(TextureSampleType::Uint) => "uint",
                None => ""
            }, "float".len()));
        } else {
            print!("{}", blank_n("rgba mod, float mod, ".len()));
        }
        if usages.contains(TextureUsages::RENDER_ATTACHMENT) && flags.contains(TextureFormatFeatureFlags::BLENDABLE) {
            print!("ba mod");
        } else {
            print!("{}", blank_n("ba mod".len()));
        }
        print!("), ");

        // SupportsCoords<vec<_, x1> + vec<_, x2> + CubeDir + vec<_, x3>>
        {
            print!("SupportsCoords<");
            print_if("vec<_, x1>", !fmt.has_depth_aspect() && !fmt.has_stencil_aspect() && !fmt.is_compressed() );
            print!(" + ");
            print_if("vec<_, x2>", true);
            print!(" + ");
            print_if("CubeDir", true);
            print!(" + ");
            print_if("vec<_, x3>", !fmt.has_depth_aspect() && !fmt.has_stencil_aspect() && !fmt.is_compressed() );
            print!(" >");
        }
        print!(", ");

        // Sampler
        if let Some(sample_type_stype) = sample_type_stype {
            print!("Sampler<");
            print_if("Filtering", matches!(sample_type_stype, wgpu::TextureSampleType::Float { filterable: true }));
            print!(" + ");
            print_if("Nearest", matches!(sample_type_stype, _));
            print!(" + ");
            print_if("Comparison", matches!(sample_type_stype, wgpu::TextureSampleType::Depth));
            print!(">");
        } else {
            print!("{}", blank_n("Sampler<Filtering + Nearest + Comparison>".len()));
        }
        print!(", ");
        // SampleType
        if let Some(sample_type_stype) = sample_type_stype {
            print!("SampleType = vec<");
            print!("{}, {}", 
                left_pad(&match sample_type_stype {
                    wgpu::TextureSampleType::Float { filterable: true } => "FilterableFloat",
                    wgpu::TextureSampleType::Float { filterable: false } => "NearestFloat",
                    wgpu::TextureSampleType::Uint => "NearestUint",
                    wgpu::TextureSampleType::Sint => "NearestInt",
                    wgpu::TextureSampleType::Depth => "Depth",
                }, "FilterableFloat".len()),
                match sample_type_len {
                    0 => "  ".to_string(),
                    n => format!("x{n}"),
                }
            );
            print!(">");
        } else {
            print!("{}", blank_n("SampleType = vec<FilterableFloat, x4>".len()));
        }
        print!(", ");
        
        // TexelShaderType
        match sample_type_stype {
            Some(sample_ty) if sample_type_len != 0 => {
                print!("TexelShaderType = ");
                print!("vec<{}, x{sample_type_len}>", sample_type_to_scalar_type(sample_ty))
            }
            _ => print!("{}", blank_n("TexelShaderType = vec<f32, x4>".len()))
        }
        print!(", ");

        // MSAA
        print!("SupportsSpp<");
        let msaa_flags = 
            TextureFormatFeatureFlags::MULTISAMPLE_X2 | 
            TextureFormatFeatureFlags::MULTISAMPLE_X4 |
            TextureFormatFeatureFlags::MULTISAMPLE_X8 |
            TextureFormatFeatureFlags::MULTISAMPLE_X16;
        print_if("Single", true);
        print!(", ");
        print_if("Multi", flags.intersects(msaa_flags));
        print!(">");
        print!(", ");

        // Storage
        if usages.contains(TextureUsages::STORAGE_BINDING) {
            print!("Storage<");
            print_if("Write", usages.contains(TextureUsages::STORAGE_BINDING));
            print!(" + ");
            print_if("Read", flags.contains(TextureFormatFeatureFlags::STORAGE_READ_WRITE));
            print!(" + ");
            print_if("ReadWrite", flags.contains(TextureFormatFeatureFlags::STORAGE_READ_WRITE));
            print!(">");
        } else {
            print!("{}", blank_n("Storage<Read + ReadWrite + Write>".len()));
        }
        print!(", ");

        // Blendable
        print_if("Blendable", flags.contains(TextureFormatFeatureFlags::BLENDABLE));
        print!(", ");

        // Target
        if usages.contains(TextureUsages::RENDER_ATTACHMENT) {
            print!("Target(");
            print_if("Color", fmt.has_color_aspect());
            print!(" + ");
            print_if("Depth", fmt.has_depth_aspect());
            print!(" + ");
            print_if("Stencil", fmt.has_stencil_aspect());
            print!(")");
        }
        else {
            print!("{}", blank_n("Target(Color + Depth + Stencil)".len()));
        }
    
        print!(", ");

        // Aspect
        if !(fmt.has_color_aspect() && fmt.sample_type(Some(wgpu::TextureAspect::All), Some(device_features)).is_none()) {
            
            let print_aspect_ty = |prefix: &str, has_aspect: bool, aspect: wgpu::TextureAspect| {
                let sample_ty = fmt.sample_type(Some(aspect), Some(device_features));
                if has_aspect && sample_ty.is_some() {
                    print!("{prefix}");
                    print!("(");
                    let len = fmt.components_with_aspect(aspect);

                    let stype = match sample_ty {
                        Some(sample_ty) => sample_type_to_scalar_type(sample_ty),
                        None => panic!("format {fmt:?} when trying to write aspect {aspect:?} the `fmt.sample_type` is None"),
                    };
                    print!("vec<");
                    print!("{stype}");
                    print!(", ");
                    match len {
                        0 => panic!("format {fmt:?} when trying to write aspect {aspect:?} the `fmt.components_with_aspect` are {len}"),
                        n => print!("x{n}")
                    }
                    print!(">");
                    print!(")");
                } else {
                    print!("{}", blank_n(prefix.len() + "(vec<f32, x1>)".len()));
                }
            };
            print!("Aspect<");
            print_aspect_ty("Color", fmt.has_color_aspect(), wgpu::TextureAspect::All);
            print!(" + ");
            print_aspect_ty("Depth", fmt.has_depth_aspect(), wgpu::TextureAspect::DepthOnly);
            print!(" + ");
            print_aspect_ty("Stencil", fmt.has_stencil_aspect(), wgpu::TextureAspect::StencilOnly);
            print!(">");
        }
        else {
            print!("{}", blank_n("Aspect<Color(vec<f32, x4>) + Depth(vec<f32, x1>) + Stencil(vec<u32, x1>)>".len()))
        }
        print!(", ");
        
        // CombinedDepthStencil
        match (fmt.has_stencil_aspect(), fmt.has_depth_aspect(), fmt.aspect_specific_format(wgpu::TextureAspect::DepthOnly), fmt.aspect_specific_format(wgpu::TextureAspect::StencilOnly)) {
            (true, true, Some(depth), Some(stencil)) => {
                print!("CombinedDepthStencil({depth:?}, {stencil:?})")
            }
            _ => (),
        }
        println!(";");
    };

    for fmt in NON_GENERIC_FMTS {
        print_row(*fmt);
    }
    println!();
    // for channel in ASTC_CHANNELS.iter().cloned() {
    //     for block in ASTC_BLOCKS.iter().cloned() {
    //         print_row(TextureFormat::Astc { block, channel });
    //     }
    // }

    println!("}}");
}

const BLANK: &str = "                                                                            ";

fn blank_n(n: usize) -> &'static str {
    &BLANK[0..BLANK.len().min(n)]
}

fn left_pad(s: &str, n: usize) -> String {
    s.to_string() + blank_n(n.saturating_sub(s.chars().count()))
}

fn print_if(str: &str, condition: bool) {
    print!("{}", match condition {
        true => str,
        false => blank_n(str.len()),
    })
}