
fn start_listening() -> (std::sync::mpsc::Receiver<String>, std::thread::JoinHandle<std::io::Result<()>>) {
    use std::{net::TcpListener, io::Read};

    let (send, recv) = std::sync::mpsc::channel();

    let thread = std::thread::spawn(move || -> std::io::Result<()> {
        let listener = TcpListener::bind("127.0.0.1:32202")?;
        
        println!("listening to incoming shaders...");
        for stream in listener.incoming() {
            let mut string = String::new();
            stream?.read_to_string(&mut string)?;
            let _ = send.send(string)
            .map_err(|e| println!("send error: {e}"));
        }
        println!("no longer listening to incoming shaders.");
        Ok(())
    });

    (recv, thread)
}

pub fn new_shader_poller() -> impl Fn() -> Option<String> {
    let (recv, thread) = start_listening();
    move || {
        let _ = &thread; //capture thread
        recv.try_recv().ok()
    }
}