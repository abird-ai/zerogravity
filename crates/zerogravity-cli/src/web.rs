//! Loopback-only embedded control transport. The process host, not a browser
//! connection, owns the runtime and the shared action slot.
mod actions;
mod http;
mod wire;

use crate::control_host::{self, Actions, Document, Selection, Stop};
use std::{
    fmt::Write as _,
    net::{SocketAddr, TcpListener},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use zerogravity::{Error, Result, Runtime};

const CONNECTIONS: usize = 8;
const EVENT_STREAMS: usize = 4;

struct Shared {
    address: SocketAddr,
    authority: String,
    key: String,
    stop: AtomicBool,
    streams: AtomicUsize,
    selection: Selection,
    actions: Actions,
}

/// Separate listener lifetime. Drop/close never retires inference owners.
pub(crate) struct Host {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<Result<()>>>,
}
impl Host {
    pub fn start(
        runtime: &Runtime,
        address: SocketAddr,
        selection: Selection,
        actions: Actions,
    ) -> Result<Self> {
        require_loopback(address)?;
        let mut random = [0u8; 32];
        getrandom::fill(&mut random)
            .map_err(|_| Error::State("OS control credential generation failed".into()))?;
        let mut key = String::with_capacity(64);
        for byte in random {
            write!(key, "{byte:02x}").expect("String write");
        }
        let listener = TcpListener::bind(address)?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let authority = if address.port() == 80 {
            match address.ip() {
                std::net::IpAddr::V4(ip) => ip.to_string(),
                std::net::IpAddr::V6(ip) => format!("[{ip}]"),
            }
        } else {
            address.to_string()
        };
        let shared = Arc::new(Shared {
            address,
            authority,
            key,
            stop: AtomicBool::new(false),
            streams: AtomicUsize::new(0),
            selection,
            actions,
        });
        let runtime = runtime.clone();
        let state = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("zg-web-host".into())
            .spawn(move || accept(listener, &runtime, &state))?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    /// Intentional user-visible capability handoff, never an HTTP request
    /// target, event payload, diagnostic log, or inference API credential.
    pub fn private_url(&self) -> String {
        format!(
            "http://{}/ui/#control={}",
            self.shared.authority, self.shared.key
        )
    }

    pub fn finished(&self) -> bool {
        self.worker
            .as_ref()
            .is_none_or(|worker| worker.is_finished())
    }

    pub fn close(mut self) -> Result<()> {
        self.join()
    }

    fn join(&mut self) -> Result<()> {
        self.shared.stop.store(true, Ordering::Release);
        self.worker.take().map_or(Ok(()), |worker| {
            worker
                .join()
                .map_err(|_| Error::State("Web control host panicked".into()))?
        })
    }
}
impl Drop for Host {
    fn drop(&mut self) {
        let _ = self.join();
    }
}

fn require_loopback(address: SocketAddr) -> Result<()> {
    if !address.ip().is_loopback() {
        return Err(Error::Unsupported(
            "Web control listener must be numeric loopback".into(),
        ));
    }
    Ok(())
}

fn accept(listener: TcpListener, runtime: &Runtime, shared: &Shared) -> Result<()> {
    thread::scope(|scope| {
        let mut clients = Vec::<thread::ScopedJoinHandle<'_, ()>>::new();
        let result = (|| {
            while !shared.stop.load(Ordering::Acquire) {
                let mut index = 0;
                while index < clients.len() {
                    if clients[index].is_finished() {
                        // Connection failure is independent of the host.
                        let _ = clients.swap_remove(index).join();
                    } else {
                        index += 1;
                    }
                }
                match listener.accept() {
                    Ok((socket, peer))
                        if peer.ip().is_loopback() && clients.len() < CONNECTIONS =>
                    {
                        let worker = thread::Builder::new()
                            .name("zg-web-client".into())
                            .spawn_scoped(scope, move || {
                                let _ = http::connection(socket, runtime, shared);
                            })?;
                        clients.push(worker);
                    }
                    Ok(_) => {} // bounded; drop excess sockets rather than queue
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(20))
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(error) => return Err(error.into()),
                }
            }
            Ok(())
        })();
        // Close listener before joining every connection, including idle SSE.
        shared.stop.store(true, Ordering::Release);
        drop(listener);
        for client in clients {
            let _ = client.join();
        }
        result
    })
}

fn listen(args: &[String]) -> Result<SocketAddr> {
    let raw = match args {
        [] => "127.0.0.1:0",
        [flag, value] if flag == "--listen" => value.as_str(),
        [option] if option.starts_with("--listen=") => &option["--listen=".len()..],
        _ => {
            return Err(Error::Malformed(
                "web accepts only --listen IP:PORT; see web --help".into(),
            ));
        }
    };
    let address = raw
        .parse::<SocketAddr>()
        .map_err(|_| Error::Malformed("web --listen requires numeric IP:PORT".into()))?;
    require_loopback(address)?;
    Ok(address)
}

pub(super) fn run(args: &[String], selection: Option<(&str, &str)>, offline: bool) -> Result<()> {
    let address = listen(args)?;
    let document = Selection::new(Document::new(selection, offline)?);
    let runtime = control_host::runtime()?;
    let stop = Stop::install()?;
    let actions = Actions::default();
    let result = (|| {
        let host = Host::start(&runtime, address, document, actions.clone())?;
        use std::io::Write;
        let mut output = std::io::stdout().lock();
        writeln!(output, "Web UI: {}", host.private_url())?;
        output.flush()?;
        drop(output);
        while !stop.requested() && !host.finished() {
            thread::sleep(Duration::from_millis(100));
        }
        host.close()
    })();
    control_host::combine(
        control_host::combine(result, actions.shutdown()),
        control_host::retire(&runtime),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_loopback_only_before_host_effects() {
        assert_eq!(listen(&[]).unwrap(), "127.0.0.1:0".parse().unwrap());
        assert!(listen(&["--listen=[::1]:0".into()]).is_ok());
        for input in [
            "--listen=0.0.0.0:0",
            "--listen=localhost:0",
            "--listen=10.0.0.1:80",
            "--listen=::1:0",
            "--api-key=x",
        ] {
            assert!(listen(&[input.into()]).is_err(), "{input}");
        }
    }
}
