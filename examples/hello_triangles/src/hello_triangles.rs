#![allow(clippy::collapsible_match)]

use std::f32::consts::TAU;
use thiserror::Error;

use wgpu::util::DeviceExt as _;
use wgpu::{LoadOp::*, StoreOp::*};

use shame_wgpu as sm;
use sm::texture_view::TextureViewExt;
use sm::aliases::*;
use sm::prelude::*;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ShameWgpu(#[from] shame_wgpu::Error),
}

pub struct HelloTriangle {
    pipeline: wgpu::RenderPipeline,
    time: f32,
}

impl HelloTriangle {
    pub fn new(gpu: &sm::Gpu) -> Result<Self, Error> {
        let pipeline = {
            let mut enc = gpu.create_pipeline_encoder(Default::default())?;
            let mut drawcall = enc.new_render_pipeline(sm::Indexing::Incremental);

            let colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
                // convert rust array to sm::Array, which can be indexed with
                // shader runtime index
                .to_gpu();

            let index = drawcall.vertices.index;
            let vert_color = colors.at(index);

            let time: f32x1 = drawcall.push_constants.get();
            let angle_offset = drawcall.vertices.instance_index.to_f32() * TAU / 5.0;

            // calculate equilateral triangle corner positions
            let equilateral_triangle = [0, 1, 2]
                .map(|corner| (corner as f32 / 3.0) * TAU)
                .to_gpu()
                .map(move |a| a + time + angle_offset) // rotate via `time` push constant
                .map(|a| sm::vec!(a.cos(), a.sin()));

            let vert_pos = equilateral_triangle.at(index);

            // combine vertices to form triangles and rasterize
            let frag = drawcall
                .vertices
                .assemble(vert_pos * 0.7, sm::Draw::triangle_list(sm::Winding::Ccw))
                .rasterize(sm::Accuracy::default());

            // interpolate vertex colors for every fragment
            let frag_color = frag.fill(vert_color);

            // additively blend instanced triangles to form star shapes
            frag.attachments
                .color_iter()
                .next::<sm::SurfaceFormat>()
                .blend(sm::Blend::add(), frag_color.extend(1.0));

            enc.finish()?
        };

        Ok(Self { pipeline, time: 0.0 })
    }

    pub fn submit_render_commands_to_gpu(&mut self, gpu: &sm::Gpu, surface: &wgpu::TextureView) -> Result<(), Error> {
        self.time += 1.0 / 60.0;

        let mut cmd = gpu.create_command_encoder(&Default::default());
        {
            let mut pass = cmd.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[surface.attach_as_color(Clear(wgpu::Color::BLACK), Store)],
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);

            pass.set_push_constants(
                // in the future shame_wgpu can add a wgpu pipeline wrapper that
                // automatically sets this correctly, since shame returns all the
                // relevant info.
                // For now a wgpu validation error is triggered if the stages don't match
                wgpu::ShaderStages::VERTEX,
                0,
                bytemuck::bytes_of(&self.time),
            );
            pass.draw(0..3, 0..5);
        }

        let _ticket = gpu.queue().submit([cmd.finish()]);

        gpu.poll(wgpu::PollType::Poll);
        Ok(())
    }

    pub fn window_event(&mut self, event: &winit::event::WindowEvent) -> Result<(), Error> { Ok(()) }
}
