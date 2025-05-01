use std::sync::Arc;
use thiserror::Error;

use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Icon, Window};
use winit::application::ApplicationHandler;

use crate::util::winit_helpers::ApplicationHandlerNew;
use crate::{hello_triangles, gpu};
use crate::hello_triangles::HelloTriangle;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    WindowCreation(winit::error::OsError),
    #[error(transparent)]
    Gpu(#[from] gpu::Error),
    #[error(transparent)]
    Scene(#[from] hello_triangles::Error),
}

type UserEvent = ();

pub struct App {
    window: Arc<Window>,
    gpu_setup: gpu::Setup,
    scene: HelloTriangle,
}

pub struct Args {
    pub title: String,
}

impl ApplicationHandlerNew for App {
    type Args = Args;
    type InitError = Error;
    type Error = Error;
    type UserEvent = UserEvent;

    fn new(args: Args, event_loop: &winit::event_loop::ActiveEventLoop) -> Result<Self, Error> {
        let attrs = Window::default_attributes()
            .with_title(args.title)
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 800));

        let window = Arc::new(event_loop.create_window(attrs).map_err(Error::WindowCreation)?);
        let gpu_setup = pollster::block_on(gpu::Setup::new(&window))?;

        Ok(Self {
            window,
            scene: HelloTriangle::new(&gpu_setup.gpu)?,
            gpu_setup,
        })
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Error> { Ok(()) }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: winit::event::StartCause) -> Result<(), Error> {
        Ok(())
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) -> Result<(), Error> {
        use winit::event::WindowEvent as E;
        self.scene.window_event(&event)?;
        match event {
            E::RedrawRequested => {
                let (surface, view) = self.gpu_setup.try_acquire_surface()?;

                self.scene.submit_render_commands_to_gpu(&self.gpu_setup.gpu, &view)?;

                self.window.pre_present_notify();
                surface.present();

                self.window.request_redraw();
            }
            E::CloseRequested => {
                event_loop.exit();
            }
            E::Resized(size) => {
                self.gpu_setup.resize(size.into());
                self.window.request_redraw();
            }
            _ => (),
        }
        Ok(())
    }
}
