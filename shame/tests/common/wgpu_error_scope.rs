use std::{
    cell::Cell,
    collections::VecDeque,
    fmt::{Debug, Display},
};

type Location = &'static std::panic::Location<'static>;

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
    location: Cell<Location>,
}

impl<'a> WgpuErrorScope<'a> {
    /// note: this is expensive. Every creation of a new scope causes allocations,
    /// awaiting the returned error futures also may take a while
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
    ///
    /// note: awaiting this error may take long
    #[track_caller]
    pub fn finish(mut self) -> impl Future<Output = Result<(), WgpuErrors>> {
        self.unpopped_scopes = false;
        pop_all_filters_error_scopes(self.device, std::panic::Location::caller())
    }

    /// collects wgpu errors and returns them as `Err` if there are any.
    /// Otherwise returns `Ok(t)`
    ///
    /// note: awaiting this error may take long
    #[allow(unused)]
    pub fn err_or<T>(self, t: T) -> impl Future<Output = Result<T, WgpuErrors>> {
        use futures::FutureExt as _;
        self.finish().then(async |e| {
            e?;
            Ok(t)
        })
    }

    /// note: awaiting this error may take long
    #[track_caller]
    pub fn next(&self) -> impl Future<Output = Result<(), WgpuErrors>> {
        let caller = std::panic::Location::caller();
        let result = pop_all_filters_error_scopes(self.device, caller);
        push_all_filters_error_scopes(self.device);
        self.location.set(caller);
        result // await only after the error scopes have been popped and pushed
    }
}

fn push_all_filters_error_scopes(device: &wgpu::Device) {
    ALL_ERROR_FILTERS.map(|filter| device.push_error_scope(filter));
}

async fn pop_all_filters_error_scopes(device: &wgpu::Device, location: Location) -> Result<(), WgpuErrors> {
    // only await after all scopes have been popped.
    let pending_results = ALL_ERROR_FILTERS.map(|_| device.pop_error_scope());

    let mut errors = None;
    for pending_result in pending_results {
        if let Some(err) = pending_result.await {
            push_error(&mut errors, err, location)
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
            let _ = pollster::block_on(pop_all_filters_error_scopes(self.device, self.location.get()));
        }
    }
}

pub struct WgpuErrors {
    pub location: Option<Location>,
    pub first: wgpu::Error,
    pub rest: VecDeque<wgpu::Error>,
}

impl std::error::Error for WgpuErrors {}

impl Debug for WgpuErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self}") }
}

impl Display for WgpuErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(loc) = self.location {
            writeln!(f, "\n--> {loc}")?;
        }
        match self.rest.len() {
            0 => {
                write!(f, "{}", self.first)
            }
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
            location: None,
            first,
            rest: Default::default(),
        }
    }
}

fn push_error(errors: &mut Option<WgpuErrors>, error: wgpu::Error, location: Location) {
    match errors {
        Some(errors) => errors.rest.push_back(error),
        None => {
            *errors = Some(WgpuErrors {
                location: Some(location),
                first: error,
                rest: Default::default(),
            })
        }
    }
}
