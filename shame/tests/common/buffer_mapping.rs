#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MappingStatus {
    NotMappedYet,
    Error(wgpu::BufferAsyncError),
}

use std::sync::{Arc, Mutex};

use bytemuck::AnyBitPattern;

/// convenient buffer mapping guard for writing tests
struct BufferMapping<'a> {
    slice: wgpu::BufferSlice<'a>,
    status: Arc<Mutex<Result<(), MappingStatus>>>,
}

impl<'a> BufferMapping<'a> {
    pub fn announce_read(slice: wgpu::BufferSlice<'a>) -> Self {
        let mode = wgpu::MapMode::Read;

        let status = Arc::new(Mutex::new(Err(MappingStatus::NotMappedYet)));

        slice.map_async(mode, {
            let status = status.clone();
            move |result| {
                let mut status = status.lock().unwrap();
                *status = result.map_err(MappingStatus::Error)
            }
        });

        Self { slice, status }
    }

    pub fn block_on_read<T: AnyBitPattern>(self, gpu: &wgpu::Device) -> Result<Vec<T>, wgpu::BufferAsyncError> {
        // since gpu.poll is a noop when targeting WebGPU, it is not guaranteed
        // that calling gpu.poll accomplishes anything, so if the buffer
        // was not mapped, we effectively spin here until it has been done.
        loop {
            match self.try_read() {
                Err(MappingStatus::NotMappedYet) => match gpu.poll(wgpu::PollType::Poll) {
                    Ok(wgpu::PollStatus::Poll | wgpu::PollStatus::QueueEmpty) => std::hint::spin_loop(),
                    x => unreachable!("PollType::Poll returns PollType::Wait-related value: {:?}", x),
                },
                Err(MappingStatus::Error(e)) => break Err(e),
                Ok(t) => break Ok(t),
            };
        }
    }

    pub fn try_read<T: AnyBitPattern>(&self) -> Result<Vec<T>, MappingStatus> {
        self.status.lock().unwrap().clone()?;
        let slice_guard = self.slice.get_mapped_range();
        Ok(bytemuck::cast_slice(&slice_guard).into())
    }

    #[allow(unused)]
    pub fn try_read_once<T: AnyBitPattern>(self) -> Result<Vec<T>, MappingStatus> {
        self.try_read()
        // implicit drop(self)
    }
}

impl Drop for BufferMapping<'_> {
    fn drop(&mut self) { self.slice.buffer().unmap(); }
}

/// load the data from a buffer with appropriate usage flags and
/// reinterpret it as a slice of `T`
pub fn download_buffer_from_gpu<T: AnyBitPattern>(
    gpu: &wgpu::Device,
    slice: wgpu::BufferSlice,
) -> Result<Vec<T>, wgpu::BufferAsyncError> {
    let mapping = BufferMapping::announce_read(slice);
    mapping.block_on_read(&gpu)
}
