#![allow(clippy::option_map_unit_fn)]
use std::marker::PhantomData;

use thiserror::Error;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, StartCause, WindowEvent},
    event_loop::ActiveEventLoop,
    window::WindowId,
};

/// this is a wrapper around the winit application handler
/// which allows full RAII-style construction of the application
/// with the [`winit::event_loop::ActiveEventLoop`]
/// instead of late `Option` based initialization
pub trait ApplicationHandlerNew: Sized {
    type Args;
    type InitError: std::error::Error;
    type Error: std::error::Error;
    type UserEvent: 'static;

    fn new(args: Self::Args, event_loop: &winit::event_loop::ActiveEventLoop) -> Result<Self, Self::InitError>;

    /// see [`winit::application::ApplicationHandler::new_events`]
    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) -> Result<(), Self::Error> {
        let _ = (event_loop, cause);
        Ok(())
    }

    /// see [`winit::application::ApplicationHandler::resumed`]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Self::Error>;

    /// see [`winit::application::ApplicationHandler::user_event`]
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: Self::UserEvent) -> Result<(), Self::Error> {
        let _ = (event_loop, event);
        Ok(())
    }

    /// see [`winit::application::ApplicationHandler::window_event`]
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) -> Result<(), Self::Error>;

    /// see [`winit::application::ApplicationHandler::device_event`]
    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) -> Result<(), Self::Error> {
        let _ = (event_loop, device_id, event);
        Ok(())
    }

    /// see [`winit::application::ApplicationHandler::about_to_wait`]
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Self::Error> {
        let _ = event_loop;
        Ok(())
    }

    /// see [`winit::application::ApplicationHandler::suspended`]
    fn suspended(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Self::Error> {
        let _ = event_loop;
        Ok(())
    }

    /// see [`winit::application::ApplicationHandler::exiting`]
    fn exiting(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Self::Error> {
        let _ = event_loop;
        Ok(())
    }

    /// see [`winit::application::ApplicationHandler::memory_warning`]
    fn memory_warning(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Self::Error> {
        let _ = event_loop;
        Ok(())
    }
}

/// same as the regular `winit::EventLoop::run_app` except it
/// allows construction of `A` with an `ActiveEventLoop` (useful for surface creation)
/// as well as fallible event handlers via `Result`
pub fn run_app<A: ApplicationHandlerNew>(
    event_loop: winit::event_loop::EventLoop<A::UserEvent>,
    init_args: A::Args,
) -> Result<(), ApplicationHandlerError<A>> {
    let mut handler = Handler::<A> {
        init_args: Some(init_args),
        error: None,
        app: None,
    };

    let result = event_loop
        .run_app(&mut handler)
        .map_err(ApplicationHandlerError::EventLoop);

    match handler.error {
        Some(e) => Err(e),
        None => result,
    }
}

#[derive(Error)]
pub enum ApplicationHandlerError<A: ApplicationHandlerNew> {
    #[error("error when initializing application: {0}")]
    Init(A::InitError),
    #[error("winit event loop error: {0}")]
    EventLoop(winit::error::EventLoopError),
    #[error("winit handler error: {0}")]
    Handler(A::Error),
    #[error("repeated application initialization error. This happens when event loop exit() did not actually exit.")]
    RepeatedInitialization,
}

impl<A: ApplicationHandlerNew> std::fmt::Debug for ApplicationHandlerError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self) }
}

struct Handler<A: ApplicationHandlerNew> {
    init_args: Option<A::Args>,
    error: Option<ApplicationHandlerError<A>>,
    app: Option<A>,
}

impl<A: ApplicationHandlerNew> Handler<A> {
    fn init_app(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) -> Result<A, ApplicationHandlerError<A>> {
        self.init_args
            .take()
            .ok_or(ApplicationHandlerError::RepeatedInitialization)
            .and_then(|args| A::new(args, event_loop).map_err(ApplicationHandlerError::Init))
    }

    fn abort_loop_on_error(
        &mut self,
        event_loop: &ActiveEventLoop,
        f: impl FnOnce(&mut A, &ActiveEventLoop) -> Result<(), A::Error>,
    ) {
        if let Some(app) = &mut self.app {
            if let Err(e) = f(app, event_loop) {
                self.error = Some(ApplicationHandlerError::Handler(e));
                event_loop.exit();
            }
        }
    }
}

impl<A: ApplicationHandlerNew> ApplicationHandler<A::UserEvent> for Handler<A> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.app.is_none() {
            match self.init_app(event_loop) {
                Ok(app) => self.app = Some(app),
                Err(e) => {
                    self.error = Some(e);
                    event_loop.exit();
                }
            }
        }
        self.abort_loop_on_error(event_loop, |app, l| app.resumed(l));
    }

    fn window_event(
        &mut self,
        l: &winit::event_loop::ActiveEventLoop,
        w: winit::window::WindowId,
        e: winit::event::WindowEvent,
    ) {
        self.app.as_mut().map(|app| app.window_event(l, w, e));
    }

    fn new_events(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, cause: winit::event::StartCause) {
        self.abort_loop_on_error(event_loop, |app, l| app.new_events(l, cause));
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: A::UserEvent) {
        self.abort_loop_on_error(event_loop, |app, l| app.user_event(l, event));
    }

    fn device_event(
        &mut self,
        l: &winit::event_loop::ActiveEventLoop,
        d: winit::event::DeviceId,
        e: winit::event::DeviceEvent,
    ) {
        self.abort_loop_on_error(l, |app, l| app.device_event(l, d, e));
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.abort_loop_on_error(event_loop, |app, l| app.about_to_wait(l));
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.abort_loop_on_error(event_loop, |app, l| app.suspended(l));
    }

    fn exiting(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.abort_loop_on_error(event_loop, |app, l| app.exiting(l));
    }

    fn memory_warning(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.abort_loop_on_error(event_loop, |app, l| app.memory_warning(l));
    }
}
