use features::{ComputeGrid, WorkGroup, GridSize};
use io_iter::PushConstants;
use pipeline_info::{ComputePipeline, PipelineDefinition, RenderPipeline};

use self::{
    features::{DispatchContext, DrawContext, Indexing},
    pipeline_kind::*,
};
use crate::{
    backend::{language::Language, shader_code::ShaderCode, wgsl::WgslErrorKind},
    call_info,
    common::marker::{Unsend, Unsync},
    frontend::encoding::{
        io_iter::{BindGroupIter, VertexBufferIter},
        rasterizer::{PrimitiveAssembly, VertexStage},
    },
    ir::{
        ir_type::LayoutError,
        pipeline::{PipelineError, PipelineKind, StageSolverErrorKind},
        recording::{
            next_thread_generation, AllocError, BlockError, CallInfo, Context, FnError, NodeRecordingError, StmtError,
            ThreadContextGuard,
        },
    },
    try_ctx_track_caller,
};

use std::{cell::Cell, fmt::Display, marker::PhantomData, rc::Rc};

use super::{
    any::{render_io::VertexLayoutError, shared_io::BindingError, ArgumentNotAvailable, InvalidReason},
    error::InternalError,
    rust_types::{error::FrontendError, len::x3},
};

pub mod binding;
pub mod buffer;
pub mod buffer_op;
pub mod color_target;
pub mod features;
pub mod fill;
pub mod flow;
pub mod fragment_test;
pub mod io_iter;
pub mod io_iter_dynamic;
pub mod mask;
pub mod pipeline_info;
pub mod rasterizer;

pub mod pipeline_kind {
    use crate::ir::pipeline::PipelineKind;

    /// Marker type for `EncodingGuard`'s `Render`-pipeline instantiation
    #[derive(Debug)]
    pub struct Render;

    /// Marker type for `EncodingGuard`'s `Compute`-pipeline instantiation
    #[derive(Debug)]
    pub struct Compute;

    /// marker types that represent pipeline kinds
    ///
    /// either [`Render`] or [`Compute`]
    pub trait IsPipelineKind {
        /// `Self` as a runtime enum
        const PIPELINE_KIND: PipelineKind;
    }
    impl IsPipelineKind for Render {
        const PIPELINE_KIND: PipelineKind = PipelineKind::Render;
    }
    impl IsPipelineKind for Compute {
        const PIPELINE_KIND: PipelineKind = PipelineKind::Compute;
    }
}
use crate as shame;


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
/// use shame as sm;
///
/// fn main() -> Result<(), sm::EncodingErrors> {
///
///     let pipeline = {
///         // `sm::Settings` is a struct that allows further configuration
///         // of your pipeline encoding and error reporting.
///         let mut enc = sm::start_encoding(sm::Settings::default())?;
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
pub fn start_encoding<P: IsPipelineKind>(
    settings: shame::Settings,
) -> Result<EncodingGuard<P>, ThreadIsAlreadyEncoding> {
    EncodingGuard::new(settings)
}

/// An `EncodingGuard` represents an active `shame` pipeline encoding on the current thread
/// Once this objects lifetime ends (either via drop or `.finish()`) a new
/// pipeline encoding can be started on this thread.
pub struct EncodingGuard<P> {
    thread_context: ThreadContextGuard,
    dispatched: bool,
    phantom: PhantomData<(P, Unsend, Unsync)>,
}

/// A runtime error that happened during pipeline encoding and is reported
/// at the end of the encoding when calling [`EncodingGuard::finish`] as part of
/// [`EncodingErrors`]
///
/// These errors can be customized in the [`Settings`] passed into [`start_encoding`]
/// to contain terminal colors and excerpts of the rust-code that caused the error.
pub struct EncodingError {
    /// Location of the rust code that called the function that caused the error.
    pub location: CallInfo,
    /// whether this error should use console colors in its `std::ops::Display` formatting
    pub use_colors: bool,
    /// whether this error should attempt to read the rust file at [`EncodingError::location`]
    /// to provide a little excerpt of the code that caused the error in the
    /// `std::ops::Display` formatting
    /// (no documentation yet)
    pub write_excerpt: bool,
    /// details about the error that happened
    /// (no documentation yet)
    pub error: EncodingErrorKind,
}

impl std::fmt::Debug for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self) }
}

impl std::error::Error for EncodingError {}

impl std::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::common::format::*;
        use crate::common::prettify::*;
        let use_colors = self.use_colors;
        let use_256_color_mode = false;
        let color = |f_: &mut _, hex| match use_colors {
            true => set_color(f_, Some(hex), use_256_color_mode),
            false => Ok(()),
        };
        let reset = |f_: &mut _| match use_colors {
            true => set_color(f_, None, use_256_color_mode),
            false => Ok(()),
        };

        color(f, "#DF5853")?;
        write!(f, "shame encoding error")?;
        reset(f)?;
        writeln!(f, ":")?;
        color(f, "#508EE3")?;
        #[cfg(feature = "error_excerpt")]
        write!(f, "  ")?;
        write!(f, "-->")?;
        reset(f)?;
        writeln!(f, " {}", self.location)?;
        #[cfg(feature = "error_excerpt")]
        if self.write_excerpt {
            write_error_excerpt(f, self.location, use_colors)?;
        }
        reset(f)?;
        write!(f, "{}", self.error)
    }
}

/// Errors that happened during pipeline encoding, which caused the
/// pipeline creation to fail.
///
/// Contains one or more [`EncodingError`]s.
/// This type is a [`std::error::Error`] as well, unpacking the individual errors
/// is not necessary for propagating errors.
///
/// `self.first` contains the first and most relevant error.
pub struct EncodingErrors {
    /// the first error that caused the pipeline encoding to fail
    pub first: EncodingError,
    /// subsequent errors, which may or may not be caused by the first one
    pub rest: Box<[EncodingError]>,
}

impl std::fmt::Debug for EncodingErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self) }
}

impl std::error::Error for EncodingErrors {}

impl EncodingErrors {
    /// create a [`EncodingErrors`] instance from a [`Vec<EncodingError>`].
    ///
    /// Returns
    /// - `None` if `errors` is empty,
    /// - `Some` otherwise.
    pub fn from_vec(errors: Vec<EncodingError>) -> Option<Self> {
        let mut it = errors.into_iter();
        it.next().map(|first| Self {
            first,
            // filter the errors that are only present because of previous errors
            rest: it.filter(|e| !e.error.is_because_of_previous_error()).collect(),
        })
    }
}

impl IntoIterator for EncodingErrors {
    type Item = EncodingError;

    type IntoIter = std::iter::Chain<std::iter::Once<EncodingError>, std::vec::IntoIter<EncodingError>>;

    fn into_iter(self) -> Self::IntoIter { std::iter::once(self.first).chain(self.rest.into_vec()) }
}

impl From<EncodingError> for EncodingErrors {
    fn from(first: EncodingError) -> Self {
        Self {
            first,
            rest: Box::new([]),
        }
    }
}

impl std::fmt::Display for EncodingErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.first)?;
        for e in &*self.rest {
            writeln!(f, "{e}")?;
            writeln!(
                f,
                "(this error may have been caused by one of the previous errors above)"
            )?
        }
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum EncodingErrorKind {
    #[error("{0}")]
    StartEncodingError(#[from] ThreadIsAlreadyEncoding),
    #[error(
        "Dispatch called twice on the same encoding instance. Any render or compute pipeline can only dispatch vertices/threads once."
    )]
    MultipleDispatches,
    #[error("Provided work group size `[{}, {}, {}]` is zero. For a work group size [x, y, z] the product x*y*z must be non zero.", .0[0], .0[1], .0[2], )]
    WorkGroupSizeZero([u32; 3]),
    #[error("{0}")]
    FrontendError(#[from] FrontendError),
    #[error("{}", PipelineKind::write_requires_pipeline_kind_error(*.current, *.required))]
    RequiresPipelineKind {
        current: PipelineKind,
        required: PipelineKind,
    },
    #[error("`shame::any::Any` instance is not available. reason: {0}")]
    ValueUnavailable(InvalidReason),
    #[error("{0}")]
    NodeRecordingError(#[from] NodeRecordingError),
    #[error("{0}")]
    AllocError(#[from] AllocError),
    #[error("{0}")]
    UserDefinedError(Box<dyn std::error::Error>),
    #[error("{0}")]
    PipelineError(#[from] PipelineError),
    #[error("{0}")]
    LayoutError(#[from] LayoutError),
    #[error("{0}")]
    BindingError(#[from] BindingError),
    #[error("{0}")]
    StageSolverError(#[from] StageSolverErrorKind),
    #[error("{0}")]
    VertexLayoutError(#[from] VertexLayoutError),
    #[error("{0}")]
    WgslError(#[from] WgslErrorKind),
    #[error("{0}")]
    BlockError(#[from] BlockError),
    #[error("{0}")]
    StmtError(#[from] StmtError),
    #[error("{0}")]
    FnError(#[from] FnError),
    #[error("{0}")]
    Internal(#[from] InternalError),
}

impl EncodingErrorKind {
    pub fn is_because_of_previous_error(&self) -> bool {
        match self {
            EncodingErrorKind::ValueUnavailable(reason) => matches!(reason, InvalidReason::ErrorThatWasPushed),
            EncodingErrorKind::NodeRecordingError(NodeRecordingError::ArgumentNotAvailable(na)) => na
                .arg_availability
                .iter()
                .filter_map(|reason| *reason)
                .all(|reason| matches!(reason, InvalidReason::ErrorThatWasPushed)),
            EncodingErrorKind::BlockError(BlockError::IllFormedBlockSeriesRecorder) => true,
            EncodingErrorKind::FnError(FnError::ReturnValueNotAvailable(InvalidReason::ErrorThatWasPushed)) => true,
            EncodingErrorKind::WgslError(WgslErrorKind::TypeMayNotAppearInWrittenForm(_)) => true,
            _ => false,
        }
    }
}

/// The error that occurs when a new encoding is started on a thread, but on
/// the same thread there is already an encoding happening with an [`EncodingGuard`]
/// that has neither been `.finish()`ed or dropped.
#[derive(thiserror::Error, Debug, Clone, Copy)]
#[error(
    "Another [`EncodingGuard`] instance is present on this thread. Make sure to finish (via `.finish()`) or drop any active `Encoding` before starting a new one."
)]
pub struct ThreadIsAlreadyEncoding;

impl From<ThreadIsAlreadyEncoding> for EncodingError {
    #[track_caller]
    fn from(err: ThreadIsAlreadyEncoding) -> Self {
        EncodingError {
            location: call_info!(),
            use_colors: false,
            write_excerpt: false,
            error: err.into(),
        }
    }
}

impl From<ThreadIsAlreadyEncoding> for EncodingErrors {
    #[track_caller]
    fn from(first: ThreadIsAlreadyEncoding) -> Self {
        EncodingErrors {
            first: first.into(),
            rest: Box::new([]),
        }
    }
}

impl<P: IsPipelineKind> EncodingGuard<P> {
    // TODO(docs) mention error condition in docs
    /// (no documentation yet)
    #[track_caller]
    pub fn new(settings: Settings) -> Result<Self, ThreadIsAlreadyEncoding> {
        let context = Context::new(call_info!(), settings, next_thread_generation(), P::PIPELINE_KIND);
        Ok(Self {
            phantom: PhantomData,
            dispatched: false,
            thread_context: ThreadContextGuard::new(context).map_err(|_ctx| ThreadIsAlreadyEncoding)?,
        })
    }
}

fn err_wrong_pipeline_kind(expected_kind: &str) -> EncodingError {
    EncodingError {
        location: call_info!(),
        use_colors: false,
        write_excerpt: false,
        error: InternalError::new(
            true,
            format!("wrong pipeline definition kind. expected {expected_kind} pipeline"),
        )
        .into(),
    }
}

impl EncodingGuard<Render> {
    /// completes the pipeline encoding and starts code generation.
    ///
    /// returns the pipeline object, or encoding errors that happened during encoding
    /// or code generation.
    #[track_caller]
    pub fn finish(self) -> Result<RenderPipeline, EncodingErrors> {
        match self.thread_context.into_inner().finish()? {
            PipelineDefinition::Render(pipeline_def) => Ok(pipeline_def),
            _ => Err(err_wrong_pipeline_kind("render").into()),
        }
    }
}

impl EncodingGuard<Compute> {
    /// completes the pipeline encoding and starts code generation.
    ///
    /// returns the pipeline object, or encoding errors that happened during encoding
    /// or code generation.
    #[track_caller]
    pub fn finish(self) -> Result<ComputePipeline, EncodingErrors> {
        match self.thread_context.into_inner().finish()? {
            PipelineDefinition::Compute(pipeline_def) => Ok(pipeline_def),
            _ => Err(err_wrong_pipeline_kind("compute").into()),
        }
    }
}

impl EncodingGuard<Render> {
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
    ///     let mut enc = sm::start_encoding(sm::Settings::default())?;
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
        try_ctx_track_caller!(|ctx| {
            let already_dispatched = std::mem::replace(&mut self.dispatched, true);
            if already_dispatched {
                try_ctx_track_caller!(|ctx| ctx.push_error(EncodingErrorKind::MultipleDispatches));
            }
            ctx.render_pipeline_mut().vertex_id_order.set(vertex_indexing);
        });
        DrawContext {
            vertices: VertexStage::new(),
            bind_groups: BindGroupIter::new(),
            push_constants: PushConstants::new(),
            encoding: self,
            phantom: PhantomData,
        }
    }
}

impl EncodingGuard<Compute> {
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
    ///     let mut enc = sm::start_encoding(sm::Settings::default())?;
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
        let mut grid_size3 = [1; 3];
        grid_size3
            .iter_mut()
            .zip(thread_grid_size_per_workgroup.iter())
            .for_each(|(a, b)| *a = *b);

        try_ctx_track_caller!(|ctx| {
            let already_dispatched = std::mem::replace(&mut self.dispatched, true);
            if already_dispatched {
                ctx.push_error(EncodingErrorKind::MultipleDispatches);
            }
            match grid_size3 {
                [x, y, z] if x * y * z == 0 => ctx.push_error(EncodingErrorKind::WorkGroupSizeZero(grid_size3)),
                _ => (),
            }
            let mut p = ctx.compute_pipeline_mut();
            p.thread_grid_size_within_workgroup.set(grid_size3);
        });
        DispatchContext {
            bind_groups: BindGroupIter::new(),
            push_constants: PushConstants::new(),
            encoding: self,
            phantom: PhantomData,
            grid: ComputeGrid::new(),
        }
    }
}

/// ## configure the shame encoding
/// this struct only has public fields and is expected to be filled out via
/// its constructor.
///
/// ## example
/// ```
/// shame::Settings {
///     colored_error_messages: false,
///     ..Default::default()
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Settings {
    /// the target shader language / graphics api
    pub lang: Language,
    /// whether the [`std::fmt::Display`] impl of [`EncodingError`]s returned
    /// at the end of encoding should use terminal colors.
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
    #[cfg(feature = "error_excerpt")]
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

impl Default for Settings {
    fn default() -> Self {
        Settings::default() // call const version
    }
}

/// returns the output `Language` of this threads active encoding.
/// This can be useful for conditionally generating code for certain target
/// apis/languages, or generating custom errors for features that are unsupported
/// for certain target apis/languages (see [`crate::any::Any::new_invalid_from_err`]).
///
/// If no encoding is active on this thread, output is unspecified.
#[track_caller]
pub fn language() -> Language { Context::try_with(call_info!(), |ctx| ctx.settings().lang).unwrap_or(Language::Wgsl) }

impl Settings {
    pub(crate) fn assemble_error_fn(&self) -> impl Fn(CallInfo, EncodingErrorKind) -> EncodingError {
        let use_colors = self.colored_error_messages;
        let write_excerpt = self.error_excerpt;
        move |call_info, error| EncodingError {
            location: call_info,
            use_colors,
            write_excerpt,
            error,
        }
    }

    /// the default pipeline-encoding settings.
    ///
    /// Same as [`Default::default()`], except this function is `const`
    pub const fn default() -> Self {
        Self {
            lang: Language::Wgsl,
            colored_error_messages: true,
            #[cfg(feature = "error_excerpt")]
            error_excerpt: true,
            shader_identifier_prefix: "s_",
            vertex_writable_storage_by_default: false,
            zero_init_workgroup_memory: false,
        }
    }
}
