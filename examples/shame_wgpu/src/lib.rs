//! shame wgpu integration
//!
//! bind-group and pipeline glue code
pub use shame::*;
pub mod bind_group;
pub mod binding;
pub mod conversion;
mod surface_format;
pub mod texture_view;

#[doc(hidden)]
pub mod __reexport {
    pub use concat_idents::concat_idents; // used by `bind_group!`
}

use conversion::ShameToWgpuError;
use pipeline_kind::*;

use thiserror::Error;
pub use surface_format::SurfaceFormat;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ShameToWgpu(#[from] ShameToWgpuError),
    #[error(transparent)]
    ThreadAlreadyEncoding(#[from] ThreadIsAlreadyEncoding),
    #[error(transparent)]
    Encoding(#[from] EncodingErrors),
}

/// a wrapper type around the `wgpu::Device` and `wgpu::Queue`, which also
/// contains some runtime information about the window surface format.
///
/// > note: if your implementation needs multiple windows with separate surface
/// > formats, this could be replaced with a key-value store from winit::WindowId
/// > to the respective surface formats
#[derive(Clone)]
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_format: Option<wgpu::TextureFormat>,
}

impl Gpu {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, surface_format: Option<wgpu::TextureFormat>) -> Self {
        Gpu {
            device,
            queue,
            surface_format,
        }
    }

    pub fn queue(&self) -> &wgpu::Queue { &self.queue }
    pub fn surface_format(&self) -> Option<wgpu::TextureFormat> { self.surface_format }
}

impl std::ops::Deref for Gpu {
    type Target = wgpu::Device;
    fn deref(&self) -> &Self::Target { &self.device }
}

/// this struct corresponds to the [`shame::Settings`].
/// It implements
///
/// [`shame::Settings`]: shame::Settings
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineEncoderOptions {
    pub colored_error_messages: bool,
    /// ### code excerpts in [`EncodingError`] messages
    /// *requires the cargo feature `error_excerpt`*
    ///
    /// allow reading the rust code files at runtime to output better
    /// error messages.
    /// If `true`, an excerpt of the rust code causing the error is
    /// copy pasted into the error message (similar to rustc errors).
    ///
    /// example:
    /// ```
    ///     shame encoding error:
    ///     --> path/to/file.rs:52:15
    ///      |
    ///   51 |
    ///   52 >    let x = a.atan();
    ///      |              ^
    ///   53 |    let y = sm::Any::atan2(a, x);
    ///      |
    ///   no matching function call for `Atan` with argument types...
    /// ```
    ///
    /// > note: failure to read the file will not cause a panic or error, but
    /// > will display a small message in the error output.
    pub error_excerpt: bool,
    /// all identifiers in the generated shader code are preceded by this prefix
    /// string (except those that are forced).
    ///
    /// WGSL example for `shader_identifier_prefix = "s_"` :
    /// ```
    /// let s_position: vec4<f32>;
    /// let s_uv: vec4<f32>;
    /// ```
    /// to prevent collisions with language specific keywords/builtins.
    /// If the prefix is `s_` and the identifier is `return`, the identifier will
    /// show up in the shader as `s_return`
    /// and won't cause problems regarding the WGSL `return` keyword.
    ///
    /// You may choose an empty string as a prefix to make the generated code more concise
    /// at the risk of causing name collisions when compiling your shader/pipeline.
    ///
    /// Some prefixes such as `__` will automatically get modified since some output languages
    /// do not allow identifiers starting with `__`.
    ///
    /// If in doubt, use `..Settings::default()`
    pub shader_identifier_prefix: &'static str,
    /// whether it is possible to write per-vertex values into storage buffers
    /// by default.
    ///
    /// Normally it is required to set the shader stage visibility explicitly for
    /// vertex-stage writeable storage buffers
    /// (via
    ///     [`BindingIter::next_with_visibility`] or
    ///     [`BindingIter::at_with_visibility`]
    /// )
    /// but with this feature enabled, the regular `.next()` and `.at(i)` [`BindingIter`]
    /// functions will assume vertex stage visibility.
    /// > maintainer note: related: https://github.com/KhronosGroup/Vulkan-Docs/issues/1790
    ///
    /// [`BindingIter`]: crate::BindingIter
    /// [`BindingIter::next_with_visibility`]: crate::BindingIter::next_with_visibility
    /// [`BindingIter::at_with_visibility`]: crate::BindingIter::next_with_visibility
    pub vertex_writable_storage_by_default: bool,
    /// whether to zero initialize workgroup memory before the pipeline starts executing
    pub zero_init_workgroup_memory: bool,
}

impl Default for PipelineEncoderOptions {
    fn default() -> Self {
        Self {
            colored_error_messages: true,
            error_excerpt: true,
            shader_identifier_prefix: "s_",
            vertex_writable_storage_by_default: false,
            zero_init_workgroup_memory: false,
        }
    }
}

impl Gpu {
    /// Wgpu-specific wrapper around [`shame::start_encoding`]
    ///
    /// (the documentation of [`shame::start_encoding`] may be more up-to-date)
    ///
    /// start a pipeline encoding on the current thread. The encoding is in progress
    /// for as long as the returned [`EncodingGuard`] object is alive. The encoding
    /// is concluded by calling `encoding_guard.finish()`, which gives access to the
    /// assembled `shame` pipeline.
    ///
    /// This function returns an `Err(ThreadIsAlreadyEncoding)` if another [`EncodingGuard`]
    /// is alive on this same thread.
    ///
    /// ## Example
    /// ```
    /// fn main() -> Result<(), sm::EncodingErrors> {
    ///
    ///     let pipeline = {
    ///         // `PipelineEncoderOptions` is a struct that allows further configuration
    ///         // of your pipeline encoding and error reporting.
    ///         let mut enc = gpu.create_pipeline_encoder(PipelineEncoderOptions::default())?;
    ///         // `enc` is generic over the pipeline kind, which decided by calling
    ///         // either `enc.new_render_pipeline` or `enc.new_compute_pipeline`.
    ///         // Without this additional call, there will be a compiler error.
    ///        
    ///         let mut drawcall = enc.new_render_pipeline(sm::Indexing::Incremental);
    ///
    ///         // ... use `drawcall` to build your pipeline
    ///
    ///         // finish the encoding and obtain the pipeline info/shader code
    ///         enc.finish()?
    ///     };
    ///
    ///     // use the generated `pipeline`
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// ## Integration
    /// If you intend to integrate `shame` into your own renderer, you likely want
    /// to wrap this function and return the actual Graphics API pipeline as a result.
    /// You can check out the applications in the `examples` folder and how they use
    /// the `shame_wgpu` crate as a reference.
    ///
    /// [`Render`]: crate::pipeline_kind::Render
    /// [`Compute`]: crate::pipeline_kind::Compute
    #[track_caller]
    pub fn create_pipeline_encoder<P: IsPipelineKind>(
        &self,
        desc: PipelineEncoderOptions,
    ) -> Result<PipelineEncoder<P>, Error> {
        Ok(PipelineEncoder {
            gpu: wgpu::Device::clone(self),
            enc_guard: start_encoding(Settings {
                lang: Language::Wgsl,
                colored_error_messages: desc.colored_error_messages,
                error_excerpt: desc.error_excerpt,
                shader_identifier_prefix: desc.shader_identifier_prefix,
                vertex_writable_storage_by_default: desc.vertex_writable_storage_by_default,
                zero_init_workgroup_memory: desc.zero_init_workgroup_memory,
            })?,
            surface_format: self.surface_format,
        })
    }
}

pub struct PipelineEncoder<P: IsPipelineKind> {
    gpu: wgpu::Device,
    enc_guard: EncodingGuard<P>,
    surface_format: Option<wgpu::TextureFormat>,
}

impl PipelineEncoder<Render> {
    /// Wgpu-specific wrapper around [`shame::EncodingGuard::new_render_pipeline`]
    ///
    /// (the documentation of [`shame::EncodingGuard::new_render_pipeline`] may be
    /// more up-to-date)
    ///
    /// create a render pipeline (see https://gpuweb.github.io/gpuweb/#render-pipeline
    /// for more info on render pipelines)
    ///
    /// * The `vertex_indexing` argument controls whether an index-buffer is used
    ///   with this pipeline, or whether vertices are numbered incrementally without
    ///   an index-buffer.
    ///
    /// * This function returns a [`DrawContext`] object that gives access to
    ///   the Gpu's state at the time a draw command is dispatched.
    ///   For a successful pipeline encoding, the rasterizer must be used. See
    ///   The [`README.md`] example or [`examples/api_showcase/src/main.rs`] for
    ///   a full example.
    ///
    /// ## Example
    /// ```
    /// let pipeline = {
    ///     let mut enc = gpu.create_pipeline_encoder(Default::default())?;
    ///     let mut drawcall = enc.new_render_pipeline(sm::Indexing::Incremental);
    ///
    ///     &drawcall.vertices; // access to vertex-shader related functionality
    ///     &drawcall.bind_groups; // access to bind groups (descriptor-sets)
    ///     &drawcall.push_constants; // access to push constant data
    ///     
    ///     let fragments = drawcall.vertices.assemble(...).rasterize(...);
    ///     
    ///     // use fragments object for per-fragment computation and io
    ///
    ///     enc.finish()?
    /// };
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new_render_pipeline(&mut self, vertex_indexing: Indexing) -> DrawContext {
        self.enc_guard.new_render_pipeline(vertex_indexing)
    }
}

impl PipelineEncoder<Compute> {
    /// Wgpu-specific wrapper around [`shame::EncodingGuard::new_compute_pipeline`]
    ///
    /// (the documentation of [`shame::EncodingGuard::new_compute_pipeline`] may
    /// be more up-to-date)
    ///
    /// create a compute pipeline (see https://gpuweb.github.io/gpuweb/#compute-pipeline
    /// for more info on compute pipelines)
    ///
    /// * The `thread_grid_size_per_workgroup` argument controls the size of each
    ///   individual workgroup in threads. This size can be 1, 2 or 3 dimensional.
    ///   for example:
    ///
    ///   - `[64]` each workgroup has 64 threads.
    ///     The compute grid is 1D and all thread positions are 1D scalars.
    ///
    ///   - `[8, 4]` each workgroup has 8x4 threads = 32 threads total.
    ///     The compute grid is 2D and all thread positions are 2D vectors.
    ///     Thread indices are still 1D scalars.
    ///
    ///   - `[4, 4, 4]` each workgroup has 4x4x4 threads = 64 threads total.
    ///     The compute grid is 3D and all thread positions are 3D vectors.
    ///     Thread indices are still 1D scalars.
    ///
    ///   - `[8, 4, 1]` each workgroup has 8x4x1 threads = 32 threads total.
    ///     The compute grid is 3D and all thread positions are 3D vectors
    ///     even though the workgroup is only a flat 2D 8x4 slice.
    ///     Thread indices are still 1D scalars.
    ///     
    ///   The amount of workgroups dispatched is controlled at runtime by the
    ///   dispatch command.
    ///
    /// * This function returns a [`Dispatch`] object that gives access to the
    ///   Gpu's state at the time the work is dispatched.
    ///
    /// ## Example
    /// ```
    /// let pipeline = {
    ///     let mut enc = gpu.create_pipeline_encoder(Default::default())?;
    ///     let mut dispatch = enc.new_compute_pipeline([8, 4]);
    ///
    ///     &dispatch.bind_groups; // access to bind groups (descriptor-sets)
    ///     &dispatch.push_constants; // access to push constant data
    ///
    ///     enc.finish()?
    /// };
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new_compute_pipeline<const N: usize>(
        &mut self,
        thread_grid_size_per_workgroup: [u32; N],
    ) -> DispatchContext<<[u32; N] as GridSize>::Dim>
    where
        [u32; N]: GridSize,
    {
        self.enc_guard.new_compute_pipeline(thread_grid_size_per_workgroup)
    }
}

impl PipelineEncoder<Compute> {
    /// completes the pipeline encoding and starts code generation.
    ///
    /// returns the pipeline object, or encoding errors that happened during encoding
    /// or code generation.
    #[track_caller]
    pub fn finish(self) -> Result<wgpu::ComputePipeline, Error> {
        let pdef = self.enc_guard.finish()?;

        if shame::__private::DEBUG_PRINT_ENABLED {
            //println!("code spans:\n{:?}", pdef.shader.code);
            println!("pipeline info: {:#?}", pdef.pipeline);
            println!("generated code:\n{}", pdef.shader.code.syntax_highlight());
        }

        Ok(conversion::compute_pipeline(&self.gpu, pdef)?)
    }
}

impl PipelineEncoder<Render> {
    /// completes the pipeline encoding and starts code generation.
    ///
    /// returns the pipeline object, or encoding errors that happened during encoding
    /// or code generation.
    #[track_caller]
    pub fn finish(self) -> Result<wgpu::RenderPipeline, Error> {
        let pdef = self.enc_guard.finish()?;

        if shame::__private::DEBUG_PRINT_ENABLED {
            //println!("code spans:\n{:?}", pdef.shader.code);
            println!("pipeline info: {:#?}", pdef.pipeline);
            println!("generated code:\n{}", pdef.shaders.vert_code.syntax_highlight());
        }

        Ok(conversion::render_pipeline(&self.gpu, pdef, self.surface_format)?)
    }
}
