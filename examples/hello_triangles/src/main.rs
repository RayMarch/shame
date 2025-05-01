#![allow(unused)]

mod app;
mod gpu;
mod hello_triangles;
mod util;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new()?;

    // "Poll" is recommended for graphics applications to avoid missing frame deadlines
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let args = app::Args {
        title: "shame example".into(),
    };

    util::winit_helpers::run_app::<app::App>(event_loop, args)?;
    Ok(())
}
