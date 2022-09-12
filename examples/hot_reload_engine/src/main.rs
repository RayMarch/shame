use shame::{RenderPipelineRecording};
use wgpu::util::DeviceExt;
use std::time::{Duration, Instant};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use wgpu::util::*;


pub type Index = u32;

use hot_reload_shaders_lib::render_pipeline::*;
use simple_wgpu_lib::glue;
mod listen_for_shader;

async fn run(event_loop: EventLoop<()>, window: Window) {

    println!("this example demonstrates how to update the shaders at runtime, \
while still using their types in the engine crate. The shaders for this example are \
located in the `hot_reload_shaders` crate.\n
Try modifying the pipeline function in `hot_reload_shaders/src/render_pipeline.rs` and\
then execute `cargo run --bin hot_reload_shaders` in another terminal. \
It will re-record the pipeline and send it back here!\n
I recommend binding the cargo command to a hotkey or setting up a watch process.");

    let size = window.inner_size();
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty() | wgpu::Features::PUSH_CONSTANTS,
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                limits: {
                    let mut limits = wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits());
                    limits.max_push_constant_size = 4;
                    limits
                },
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let swapchain_format = surface.get_supported_formats(&adapter)[0];

    let poll_shader = listen_for_shader::new_shader_poller();

    let recording = shame::record_render_pipeline(pipeline);

    let mut render_pipeline = glue::make_render_pipeline(
        &recording,
        &device,
        Some(swapchain_format)
    );

    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&[
            VertexCpu {
                pos: [0.866, -0.5, 0.0],
                color: [1.0, 0.0, 0.0],
            },
            VertexCpu {
                pos: [0.0, 1.0, 0.0],
                color: [0.0, 1.0, 0.0],
            },
            VertexCpu {
                pos: [-0.866, -0.5, 0.0],
                color: [0.0, 0.0, 1.0],
            },
        ]),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer_source = [
        0, 1, 2 as Index
    ];
    let index_buffer_len = index_buffer_source.len() as u32;

    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&index_buffer_source),
        usage: wgpu::BufferUsages::INDEX,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let xform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&[
            0.6, 0.0, 0.0, 0.0,
            0.0, 0.6, 0.0, 0.0,
            0.0, 0.0, 0.6, 0.0,
            0.0, 0.0, 0.0, 1.0_f32,
        ]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: xform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
    };

    surface.configure(&device, &config);

    let (
        mut last_update_inst,
        mut last_frame_inst,
        mut frame_count,
        mut _accum_time,
    ) = (Instant::now(), Instant::now(), 0, 0.0);

    let mut num_shader_updates = 0;

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        //let _ = (&instance, &adapter, &shader, &pipeline_layout);
        let _ = (&instance, &adapter);

        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Reconfigure the surface with the new size
                config.width = size.width;
                config.height = size.height;
                surface.configure(&device, &config);
                // On macos the window needs to be redrawn manually after resizing
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {

                _accum_time += last_frame_inst.elapsed().as_secs_f32();
                last_frame_inst = Instant::now();
                frame_count += 1;
                let step = 120;
                if (frame_count % step) == 0 {
                    // println!(
                    //     "Avg frame time {}ms",
                    //     accum_time * 1000.0 / step as f32
                    // );
                    _accum_time = 0.0;
                }

                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(_) => {
                        surface.configure(&device, &config);
                        surface
                            .get_current_texture()
                            .expect("Failed to acquire next surface texture!")
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });

                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, bytemuck::bytes_of(
                        &(frame_count as f32 / 60.0)
                    ));
                    rpass.set_index_buffer(index_buffer.slice(..), glue::index_format_of::<Index>());
                    rpass.draw_indexed(0..index_buffer_len, 0, 0..1);
                }

                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Event::RedrawEventsCleared => {
                // Clamp to some max framerate to avoid busy-looping too much
                // (we might be in wgpu::PresentMode::Mailbox, thus discarding superfluous frames)
                //
                // winit has window.current_monitor().video_modes() but that is a list of all full screen video modes.
                // So without extra dependencies it's a bit tricky to get the max refresh rate we can run the window on.
                // Therefore we just go with 60fps - sorry 120hz+ folks!
                let target_frametime = Duration::from_secs_f64(1.0 / 60.0);
                let time_since_last_frame = last_update_inst.elapsed();
                let wake_up_dur_estimate = Duration::from_millis(2);
                if time_since_last_frame >= target_frametime {
                    window.request_redraw();
                    last_update_inst = Instant::now();
                } else {
                    *control_flow = ControlFlow::WaitUntil(
                        Instant::now() + target_frametime - time_since_last_frame
                        - wake_up_dur_estimate,
                    );
                }

                if let Some(string) = poll_shader() {
                    let split = string.split("$split_here$").collect::<Vec<_>>();
                    match &split[..] {
                        [vert, frag, info] => {
                            let existing_info = &recording.info.to_string();
                            if info != existing_info {
                                println!("EXPECTED LAYOUT:\n{existing_info}");
                                println!("GOT LAYOUT\n{info}");
                                println!("incoming shaders have incompatible pipeline layout");
                            } else {
                                let new_recording = RenderPipelineRecording {
                                    shaders_glsl: (vert.to_string(), frag.to_string()),
                                    info: recording.info.clone(),
                                };
                                render_pipeline = glue::make_render_pipeline(
                                    &new_recording,
                                    &device,
                                    Some(swapchain_format)
                                );
                                num_shader_updates += 1;
                                println!("updated shader! (#{num_shader_updates})");
                            }
                        }
                        slice => {
                            println!("{}\nerror: received a shader update that has wrong amount of parts ({})", &string, slice.len());
                        }
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
    .with_inner_size(winit::dpi::LogicalSize {
        width:  512,
        height: 512,
    })
    .build(&event_loop)
    .unwrap();

    env_logger::init();
    // Temporarily avoid srgb formats for the swapchain on the web
    pollster::block_on(run(event_loop, window));
}
