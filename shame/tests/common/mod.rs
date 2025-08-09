pub mod buffer_mapping;
pub mod gpu;
pub mod test_image;
pub mod wgpu_error_scope;
use shame_wgpu as sm;
use sm::prelude::*;
use wgpu::util::{DeviceExt};
use pollster::block_on;

use crate::common::{
    buffer_mapping::download_buffer_from_gpu,
    test_image::{TestImage2D, TestImageFormat},
    wgpu_error_scope::WgpuErrorScope,
};

/// calls `gpu::Setup::new` with reasonable default parameters.
/// If your test requires some specific capabilities, you can call `gpu::Setup`
/// yourself and skip the test based on the `Err` variant instead.
pub fn basic_test_setup() -> Result<gpu::Setup, gpu::Error> {
    let features = wgpu::Features::PUSH_CONSTANTS;

    let limits = wgpu::Limits {
        max_push_constant_size: 4,
        ..Default::default()
    };

    gpu::Setup::new(features, limits)
}

/// initialize a `[T; N]` array on the gpu with the given function `F`.
/// This is intended for tests, as it will compile a compute pipeline from `F`
/// only for this one call and then discard it.
///
/// * `dispatch_wgs`: the dimensions of the grid of workgroups that are being dispatched
/// * `wg_dims`: the dimensions of the thread-grid that makes up each individual workgroup
#[track_caller]
pub fn init_array_via_gpu_compute<const N: usize, const D: usize, T, F>(
    gpu: &sm::Gpu,
    dispatch_wgs: [u32; D],
    wg_dims: [u32; D],
    f: F,
) -> [T; N]
where
    T: sm::ScalarTypeNumber + std::fmt::Debug + bytemuck::Pod,
    sm::vec<T, x1>: sm::NoBools,
    [u32; D]: sm::GridSize,
    F: FnOnce(
        sm::DispatchContext<<[u32; D] as sm::GridSize>::Dim>,
        sm::BufferRef<sm::Array<sm::vec<T, x1>, sm::Size<N>>>,
    ),
{
    let caller = sm::__private::CallInfo::caller();

    let catch = WgpuErrorScope::new(&gpu);

    let pipeline: wgpu::ComputePipeline = {
        let mut enc = gpu.create_pipeline_encoder(Default::default()).unwrap();
        let mut pipe = enc.new_compute_pipeline(wg_dims);
        let buffer = pipe.bind_groups.next().next();
        f(pipe, buffer);
        enc.finish().unwrap()
    };

    block_on(catch.next()).unwrap();

    let storage_buffer_cpu = [T::zeroed(); N];
    let bytes = bytemuck::cast_slice(&storage_buffer_cpu);

    let storage_buffer = gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });

    block_on(catch.next()).unwrap();

    let staging_buffer = gpu.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes.len() as _,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    block_on(catch.next()).unwrap();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = gpu.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        use sm::GridSize as _;
        let [x, y, z] = dispatch_wgs.as_3d();
        pass.dispatch_workgroups(x, y, z);
    }

    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, bytes.len() as _);

    gpu.queue().submit([encoder.finish()]);
    gpu.poll(wgpu::PollType::Wait).unwrap();
    let result = download_buffer_from_gpu(&gpu, staging_buffer.slice(..)).unwrap();

    block_on(catch.finish()).unwrap();
    result.try_into().unwrap() // Vec<T> -> [T; N]
}

pub trait VertexIndexing {
    const SM: sm::Indexing;
    fn indexing(&self) -> IndexRepr;
}

pub enum IndexRepr<'a> {
    Range(std::ops::Range<u32>),
    Buffer(&'a [u8], usize),
}

impl VertexIndexing for std::ops::Range<u32> {
    const SM: sm::Indexing = sm::Indexing::Incremental;
    fn indexing(&self) -> IndexRepr { IndexRepr::Range(self.clone()) }
}

impl VertexIndexing for &[u32] {
    const SM: sm::Indexing = sm::Indexing::BufferU32;
    fn indexing(&self) -> IndexRepr { IndexRepr::Buffer(bytemuck::cast_slice(self), self.len()) }
}

impl VertexIndexing for &[u16] {
    const SM: sm::Indexing = sm::Indexing::BufferU16;
    fn indexing(&self) -> IndexRepr { IndexRepr::Buffer(bytemuck::cast_slice(self), self.len()) }
}

impl<Fmt: TestImageFormat, const W: usize, const H: usize> TestImage2D<Fmt, W, H> {
    #[track_caller]
    pub fn render_on_gpu<Idx, F>(gpu: &sm::Gpu, indexing: Idx, instances: std::ops::Range<u32>, f: F) -> Self
    where
        Idx: VertexIndexing,
        Fmt: TestImageFormat + sm::ColorTargetFormat,
        F: FnOnce(sm::DrawContext),
    {
        let caller = sm::__private::CallInfo::caller();

        let catch = WgpuErrorScope::new(&gpu);

        let pipeline: wgpu::RenderPipeline = {
            let mut enc = gpu.create_pipeline_encoder(Default::default()).unwrap();
            let draw = enc.new_render_pipeline(Idx::SM);
            println!("test @ {caller}");
            f(draw);
            enc.finish().unwrap()
        };

        block_on(catch.next()).unwrap();

        let test_image: TestImage2D<Fmt, W, H> = Default::default();

        let texture = test_image.upload_to_gpu(&gpu, &gpu.queue());
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("test view"),
            format: Some(sm::conversion::texture_format(&Fmt::id(), None).unwrap()),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let index_buffer = match indexing.indexing() {
            IndexRepr::Range(_) => None,
            IndexRepr::Buffer(data, _len) => Some((
                gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("index buffer"),
                    contents: data,
                    usage: wgpu::BufferUsages::INDEX,
                }),
                match Idx::SM {
                    shame::Indexing::Incremental => unreachable!(),
                    shame::Indexing::BufferU8 => panic!("unsupported index format"),
                    shame::Indexing::BufferU16 => wgpu::IndexFormat::Uint16,
                    shame::Indexing::BufferU32 => wgpu::IndexFormat::Uint32,
                },
            )),
        };

        let mut encoder = gpu.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.insert_debug_marker("test");
            if let Some((buffer, fmt)) = index_buffer {
                pass.set_index_buffer(buffer.slice(..), fmt);
            }

            match indexing.indexing() {
                IndexRepr::Range(range) => pass.draw(range, instances),
                IndexRepr::Buffer(_, len) => pass.draw_indexed(0..len as u32, 0, instances),
            }
        }

        gpu.queue().submit([encoder.finish()]);

        block_on(catch.next()).unwrap();

        let _ = gpu.poll(wgpu::PollType::Wait);

        let test_image = TestImage2D::download_from_gpu(&gpu, &texture);

        block_on(catch.err_or(test_image)).unwrap()
    }
}
