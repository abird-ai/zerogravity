//! Small read-only HTTP/1.1 boundary, not the inference wire parser. Explicit
//! route allowlist, numeric Host, same-origin bearer, finite socket owners.
use super::{EVENT_STREAMS, Shared, wire};
use std::{
    io::{self, Read, Write},
    net::TcpStream,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};
use zerogravity::{Runtime, control::ControlPoll};

const HEADERS: usize = 8192;
const MAX_RESPONSE: usize = 256 * 1024;
const IO_BOUND: Duration = Duration::from_secs(2);
const POLL: Duration = Duration::from_millis(250);
const EVENT_LIFETIME: Duration = Duration::from_secs(60);
const INDEX: &[u8] = include_bytes!("../../web/dist/index.html");
const SCRIPT: &[u8] = include_bytes!("../../web/dist/app.js");
const STYLE: &[u8] = include_bytes!("../../web/dist/style.css");
const LICENSES: &[u8] = include_bytes!("../../web/dist/.vite/license.md");

const SECURITY: &str = concat!(
    "Cache-Control: no-store\r\n",
    "Referrer-Policy: no-referrer\r\n",
    "X-Content-Type-Options: nosniff\r\n",
    "X-Frame-Options: DENY\r\n",
    "Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; ",
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'\r\n",
    "Cross-Origin-Resource-Policy: same-origin\r\n",
    "Connection: close\r\n",
);

fn response(
    socket: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> io::Result<()> {
    if body.len() > MAX_RESPONSE {
        return error(socket, "503 Service Unavailable");
    }
    write!(
        socket,
        "HTTP/1.1 {status}\r\n{SECURITY}Content-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )?;
    socket.write_all(body)
}

fn error(socket: &mut TcpStream, status: &str) -> io::Result<()> {
    response(
        socket,
        status,
        "text/plain; charset=utf-8",
        status.as_bytes(),
    )
}

fn header<'a>(headers: &'a [httparse::Header<'a>], name: &str) -> Result<Option<&'a [u8]>, ()> {
    let mut values = headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case(name));
    let value = values.next().map(|header| header.value);
    if values.next().is_some() {
        Err(())
    } else {
        Ok(value)
    }
}

fn bearer(value: Option<&[u8]>, key: &str) -> bool {
    let Some(value) = value.and_then(|value| value.strip_prefix(b"Bearer ")) else {
        return false;
    };
    value.len() == key.len()
        && value
            .iter()
            .zip(key.bytes())
            .fold(0u8, |difference, (a, b)| difference | (a ^ b))
            == 0
}

pub(super) fn connection(
    mut socket: TcpStream,
    runtime: &Runtime,
    shared: &Shared,
) -> io::Result<()> {
    socket.set_write_timeout(Some(IO_BOUND))?;
    let mut buffer = [0u8; HEADERS];
    let mut used = 0;
    let deadline = Instant::now() + IO_BOUND;
    let end = loop {
        if shared.stop.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut headers = [httparse::EMPTY_HEADER; 32];
        match httparse::Request::new(&mut headers).parse(&buffer[..used]) {
            Ok(httparse::Status::Complete(end)) => break end,
            Ok(httparse::Status::Partial) if used < HEADERS => {}
            _ => return error(&mut socket, "400 Bad Request"),
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(());
        }
        socket.set_read_timeout(Some(remaining.min(POLL)))?;
        match socket.read(&mut buffer[used..]) {
            Ok(0) => return Ok(()),
            Ok(bytes) => used += bytes,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                ) => {}
            Err(error) => return Err(error),
        }
    };
    // No request body, chunked input or pipelining accepted in this slice.
    if used != end {
        return error(&mut socket, "400 Bad Request");
    }
    let mut headers = [httparse::EMPTY_HEADER; 32];
    let mut request = httparse::Request::new(&mut headers);
    request.parse(&buffer[..end]).map_err(io::Error::other)?;
    let authority = shared.address.to_string();
    let host = header(request.headers, "Host");
    if request.version != Some(1)
        || !matches!(host, Ok(Some(value)) if value == shared.authority.as_bytes() || value == authority.as_bytes())
        || header(request.headers, "Transfer-Encoding") != Ok(None)
        || !matches!(
            header(request.headers, "Content-Length"),
            Ok(None | Some(b"0"))
        )
    {
        return error(&mut socket, "400 Bad Request");
    }
    if request.method != Some("GET") {
        return error(&mut socket, "405 Method Not Allowed");
    }
    match request.path {
        Some("/") => response(&mut socket, "200 OK", "text/html; charset=utf-8", INDEX),
        Some("/ui/" | "/ui/index.html") => {
            response(&mut socket, "200 OK", "text/html; charset=utf-8", INDEX)
        }
        Some("/ui/app.js") => response(
            &mut socket,
            "200 OK",
            "text/javascript; charset=utf-8",
            SCRIPT,
        ),
        Some("/ui/licenses.txt") => {
            response(&mut socket, "200 OK", "text/plain; charset=utf-8", LICENSES)
        }
        Some("/ui/style.css") => response(&mut socket, "200 OK", "text/css; charset=utf-8", STYLE),
        Some("/zg/control/snapshot" | "/zg/events") => {
            // Static assets contain no credential or state. All owner data
            // requires a distinct random capability plus same-origin policy.
            let origin = format!("http://{}", shared.authority);
            if !header(request.headers, "Origin")
                .is_ok_and(|value| value.is_none_or(|value| value == origin.as_bytes()))
            {
                return error(&mut socket, "403 Forbidden");
            }
            if !matches!(
                header(request.headers, "Sec-Fetch-Site"),
                Ok(None | Some(b"same-origin") | Some(b"none"))
            ) {
                return error(&mut socket, "403 Forbidden");
            }
            if !matches!(header(request.headers, "Authorization"), Ok(value) if bearer(value, &shared.key))
            {
                return error(&mut socket, "401 Unauthorized");
            }
            if request.path == Some("/zg/events") {
                events(&mut socket, runtime, shared)
            } else {
                match wire::snapshot(runtime, &shared.selection).and_then(|snapshot| {
                    serde_json::to_vec(&snapshot)
                        .map_err(|_| zerogravity::Error::State("snapshot encoding".into()))
                }) {
                    Ok(bytes) => response(&mut socket, "200 OK", "application/json", &bytes),
                    Err(_) => error(&mut socket, "503 Service Unavailable"),
                }
            }
        }
        _ => error(&mut socket, "404 Not Found"),
    }
}

struct StreamSlot<'a>(&'a std::sync::atomic::AtomicUsize);
impl Drop for StreamSlot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

fn events(socket: &mut TcpStream, runtime: &Runtime, shared: &Shared) -> io::Result<()> {
    if shared
        .streams
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            (n < EVENT_STREAMS).then_some(n + 1)
        })
        .is_err()
    {
        return error(socket, "503 Service Unavailable");
    }
    let _slot = StreamSlot(&shared.streams);
    let control = runtime.control();
    let mut cursor = control.subscribe();
    let mut selection = shared.selection.revision();
    write!(
        socket,
        "HTTP/1.1 200 OK\r\n{SECURITY}Content-Type: text/event-stream\r\n\r\nevent: refresh\ndata: {{}}\n\n"
    )?;
    let deadline = Instant::now() + EVENT_LIFETIME;
    let mut heartbeat = Instant::now();
    while !shared.stop.load(Ordering::Acquire) && Instant::now() < deadline {
        let mut changed = false;
        let mut resync = false;
        for i in 0..64 {
            match cursor.poll(if i == 0 { POLL } else { Duration::ZERO }) {
                ControlPoll::Pending => break,
                ControlPoll::Event(_) => changed = true,
                ControlPoll::Resync { .. } => {
                    resync = true;
                    break;
                }
            }
        }
        if shared.stop.load(Ordering::Acquire) {
            break;
        }
        let observed_selection = shared.selection.revision();
        changed |= observed_selection != selection;
        selection = observed_selection;
        if changed || resync {
            // Invalidation only. Clients refresh the same owner snapshot;
            // never treat transient notification delivery as state authority.
            socket.write_all(if resync {
                b"event: resync\ndata: {}\n\n"
            } else {
                b"event: refresh\ndata: {}\n\n"
            })?;
            heartbeat = Instant::now();
            thread_pause();
        } else if heartbeat.elapsed() >= Duration::from_secs(5) {
            socket.write_all(b": keepalive\n\n")?;
            heartbeat = Instant::now();
        }
    }
    Ok(())
}

fn thread_pause() {
    // Coalesce bursts of notifications; no inference thread or producer waits.
    std::thread::sleep(POLL);
}
