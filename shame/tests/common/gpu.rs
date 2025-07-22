use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    RequestAdapter(#[from] wgpu::RequestAdapterError),
    #[error(transparent)]
    RequestDevice(#[from] wgpu::RequestDeviceError),
}

#[allow(unused)]
pub struct Setup {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    pub gpu: shame_wgpu::Gpu,
}

impl Setup {
    pub fn new(features: wgpu::Features, limits: wgpu::Limits) -> Result<Setup, Error> {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&Default::default());
        let adapter = pollster::block_on(adapter)?;

        let device_queue = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        });
        let (device, queue) = pollster::block_on(device_queue)?;

        let setup = Setup {
            instance,
            adapter,
            gpu: shame_wgpu::Gpu::new(device, queue, None),
        };

        Ok(setup)
    }
}
