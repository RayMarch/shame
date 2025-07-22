use std::{
    cell::Cell,
    collections::VecDeque,
    fmt::{Debug, Display},
};

const ALL_ERROR_FILTERS: [wgpu::ErrorFilter; 3] = [
    wgpu::ErrorFilter::Validation,
    wgpu::ErrorFilter::OutOfMemory,
    wgpu::ErrorFilter::Internal,
];

/// raii guard for pushing/popping of error scopes
///
/// one `WgpuErrorScope` creates multiple error scopes internally, one for
/// each of [`ALL_ERROR_FILTERS`]
pub struct WgpuErrorScope<'a> {
    device: &'a wgpu::Device,
    unpopped_scopes: bool,
    location: Cell<&'static std::panic::Location<'static>>,
}

impl<'a> WgpuErrorScope<'a> {
    #[track_caller]
    pub fn new(device: &'a wgpu::Device) -> Self {
        push_all_filters_error_scopes(device);
        Self {
            device,
            unpopped_scopes: true,
            location: Cell::new(std::panic::Location::caller()),
        }
    }

    /// collect wgpu errors and return them, destroy self.
    pub async fn finish(mut self) -> Result<(), WgpuErrors> {
        self.unpopped_scopes = false;
        pop_all_filters_error_scopes(self.device).await
    }

    /// collects wgpu errors and returns them as `Err` if there are any.
    /// Otherwise returns `Ok(t)`
    #[allow(unused)]
    pub async fn err_or<T>(self, t: T) -> Result<T, WgpuErrors> {
        self.finish().await?;
        Ok(t)
    }

    pub async fn next(&self) -> Result<(), WgpuErrors> {
        let result = pop_all_filters_error_scopes(self.device);
        push_all_filters_error_scopes(self.device);
        self.location.set(std::panic::Location::caller());
        result.await // await only after the error scopes have been popped and pushed
    }
}

fn push_all_filters_error_scopes(device: &wgpu::Device) {
    ALL_ERROR_FILTERS.map(|filter| device.push_error_scope(filter));
}

async fn pop_all_filters_error_scopes(device: &wgpu::Device) -> Result<(), WgpuErrors> {
    // only await after all scopes have been popped.
    let pending_results = ALL_ERROR_FILTERS.map(|_| device.pop_error_scope());

    let mut errors = None;
    for pending_result in pending_results {
        if let Some(err) = pending_result.await {
            push_error(&mut errors, err)
        }
    }
    match errors {
        Some(errors) => Err(errors),
        None => Ok(()),
    }
}

impl Drop for WgpuErrorScope<'_> {
    #[track_caller]
    fn drop(&mut self) {
        // you should call `.finish()` or other consuming functions on
        // a `WgpuErrorScope` before you destroy it.
        if self.unpopped_scopes {
            if cfg!(debug_assertions) {
                let loc = self.location.get();
                println!(
                    "uncaught wgpu error scope in:\n{}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
            }
            let _ = pollster::block_on(pop_all_filters_error_scopes(self.device));
        }
    }
}

pub struct WgpuErrors {
    pub first: wgpu::Error,
    pub rest: VecDeque<wgpu::Error>,
}

impl std::error::Error for WgpuErrors {}

impl Debug for WgpuErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self}") }
}

impl Display for WgpuErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.rest.len() {
            0 => write!(f, "{}", self.first),
            n => {
                writeln!(f, "{n} wgpu errors:")?;
                writeln!(f, "{}", self.first)?;
                for e in self.rest.iter() {
                    writeln!(f, "{e}")?;
                }
                Ok(())
            }
        }
    }
}

impl From<wgpu::Error> for WgpuErrors {
    fn from(first: wgpu::Error) -> Self {
        WgpuErrors {
            first,
            rest: Default::default(),
        }
    }
}

fn push_error(errors: &mut Option<WgpuErrors>, error: wgpu::Error) {
    match errors {
        Some(errors) => errors.rest.push_back(error),
        None => {
            *errors = Some(WgpuErrors {
                first: error,
                rest: Default::default(),
            })
        }
    }
}
