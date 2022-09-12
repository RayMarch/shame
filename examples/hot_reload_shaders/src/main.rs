
use std::io::Write;
use std::net::TcpStream;
use shame::prettify::syntax_highlight_glsl;
mod render_pipeline;
use render_pipeline::*;

fn main() -> std::io::Result<()> {
    println!("recording shaders...");
    let recording = shame::record_render_pipeline(pipeline);
    let ((vert, frag), info) = recording.unpack();

    let result = TcpStream::connect("127.0.0.1:32202")
    .and_then(|mut stream| stream.write_fmt(format_args!("{vert}$split_here${frag}$split_here${info}")));

    match result {
        Err(e) => {
            println!("error sending shader to engine. The hot_reload_engine binary must be running.");
            Err(e)
        }
        Ok(()) => {
            //lets do the syntax highlighting after the shader is sent, so we don't
            //cause any additional latency
            println!("vertex shader: {}\n",   syntax_highlight_glsl(&vert));
            println!("fragment shader: {}\n", syntax_highlight_glsl(&frag));
            println!("recording shaders...done!");
            Ok(())
        }
    }
}
