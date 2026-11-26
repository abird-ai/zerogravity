//! Small HTTP/1.1 control boundary, not the inference wire parser. Explicit
//! route allowlist, numeric Host, same-origin bearer, finite socket owners.
use super::{EVENT_STREAMS, Shared, actions, wire};
use crate::control_host::inventory;
use std::{
    io::{self, Read, Write},
    net::TcpStream,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};
use zerogravity::{Runtime, control::ControlPoll};

const HEADERS: usize = 8192;
// Bounded 64 KiB display prefix can expand up to 6x in JSON.
const MAX_RESPONSE: usize = 512 * 1024;
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
    let mut headers = [httparse::EMPTY_HEADER; 32];
    let mut request = httparse::Request::new(&mut headers);
    request.parse(&buffer[..end]).map_err(io::Error::other)?;
    let authority = shared.address.to_string();
    let host = header(request.headers, "Host");
    if request.version != Some(1)
        || !matches!(host, Ok(Some(value)) if value == shared.authority.as_bytes() || value == authority.as_bytes())
        || header(request.headers, "Transfer-Encoding") != Ok(None)
        || header(request.headers, "Expect") != Ok(None)
    {
        return error(&mut socket, "400 Bad Request");
    }
    let length = match header(request.headers, "Content-Length") {
        Ok(None) => 0,
        Ok(Some(value)) => match std::str::from_utf8(value)
            .ok()
            .and_then(|text| text.parse::<usize>().ok().filter(|n| n.to_string() == text))
        {
            Some(length) => length,
            None => return error(&mut socket, "400 Bad Request"),
        },
        Err(()) => return error(&mut socket, "400 Bad Request"),
    };
    if length > actions::MAX_BODY {
        return error(&mut socket, "413 Payload Too Large");
    }
    if used - end > length {
        return error(&mut socket, "400 Bad Request");
    }
    let path = request.path.unwrap_or("");
    let post = request.method == Some("POST");
    if post {
        if path != "/zg/control/actions" {
            return error(&mut socket, "405 Method Not Allowed");
        }
    } else if request.method != Some("GET") {
        return error(&mut socket, "405 Method Not Allowed");
    } else if length != 0 {
        return error(&mut socket, "400 Bad Request");
    }
    // Explicit route allowlist. Static files are public, never a directory.
    let action_id = path.strip_prefix("/zg/control/actions/");
    let control = matches!(
        path,
        "/zg/control/snapshot"
            | "/zg/events"
            | "/zg/control/actions"
            | "/zg/control/inventory/models"
            | "/zg/control/inventory/devices"
    ) || action_id.is_some();
    if control {
        let origin = format!("http://{}", shared.authority);
        let origin_ok = header(request.headers, "Origin").is_ok_and(|value| {
            if post {
                value == Some(origin.as_bytes())
            } else {
                value.is_none_or(|value| value == origin.as_bytes())
            }
        });
        if !origin_ok
            || !matches!(
                header(request.headers, "Sec-Fetch-Site"),
                Ok(None | Some(b"same-origin") | Some(b"none"))
            )
        {
            return error(&mut socket, "403 Forbidden");
        }
        if !matches!(header(request.headers, "Authorization"), Ok(value) if bearer(value, &shared.key))
        {
            return error(&mut socket, "401 Unauthorized");
        }
        if post {
            if length == 0
                || header(request.headers, "Content-Type") != Ok(Some(b"application/json"))
            {
                return error(&mut socket, "400 Bad Request");
            }
            // Only authenticated, same-origin POST can allocate bounded body
            // storage. One absolute header+body deadline; no chunking.
            let mut body = vec![0u8; length];
            body[..used - end].copy_from_slice(&buffer[end..used]);
            let mut offset = used - end;
            while offset < length {
                if shared.stop.load(Ordering::Acquire) {
                    return Ok(());
                }
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return error(&mut socket, "408 Request Timeout");
                }
                socket.set_read_timeout(Some(remaining.min(POLL)))?;
                match socket.read(&mut body[offset..]) {
                    Ok(0) => return error(&mut socket, "400 Bad Request"),
                    Ok(bytes) => offset += bytes,
                    Err(error)
                        if matches!(
                            error.kind(),
                            io::ErrorKind::WouldBlock
                                | io::ErrorKind::TimedOut
                                | io::ErrorKind::Interrupted
                        ) => {}
                    Err(error) => return Err(error),
                }
            }
            return match actions::submit(runtime, shared, &body) {
                Ok((status, value)) => json_response(&mut socket, status, &value),
                Err(status) => error(&mut socket, status),
            };
        }
        return match path {
            "/zg/events" => events(&mut socket, runtime, shared),
            "/zg/control/snapshot" => {
                match wire::snapshot(runtime, &shared.selection, &shared.actions) {
                    Ok(value) => json_response(&mut socket, "200 OK", &value),
                    Err(_) => error(&mut socket, "503 Service Unavailable"),
                }
            }
            "/zg/control/inventory/models" | "/zg/control/inventory/devices" => {
                // Bound cold IO across Web clients independently of the
                // action slot. Native and Web borrow the same original owners.
                let Ok(_read) = shared.inventory_read.try_lock() else {
                    return error(&mut socket, "503 Service Unavailable");
                };
                if path.ends_with("/models") {
                    json_response(&mut socket, "200 OK", &wire::models(runtime))
                } else {
                    match wire::devices(runtime) {
                        Ok(value) => json_response(&mut socket, "200 OK", &value),
                        Err(_) => error(&mut socket, "503 Service Unavailable"),
                    }
                }
            }
            _ if action_id.is_some() => {
                match actions::result(shared, action_id.expect("action route")) {
                    Ok(value) => json_response(&mut socket, "200 OK", &value),
                    Err(status) => error(&mut socket, status),
                }
            }
            _ => error(&mut socket, "405 Method Not Allowed"),
        };
    }
    match path {
        "/" | "/ui/" | "/ui/index.html" => {
            response(&mut socket, "200 OK", "text/html; charset=utf-8", INDEX)
        }
        "/ui/app.js" => response(
            &mut socket,
            "200 OK",
            "text/javascript; charset=utf-8",
            SCRIPT,
        ),
        "/ui/licenses.txt" => {
            response(&mut socket, "200 OK", "text/plain; charset=utf-8", LICENSES)
        }
        "/ui/style.css" => response(&mut socket, "200 OK", "text/css; charset=utf-8", STYLE),
        _ => error(&mut socket, "404 Not Found"),
    }
}

fn json_response(
    socket: &mut TcpStream,
    status: &str,
    value: &serde_json::Value,
) -> io::Result<()> {
    match serde_json::to_vec(value) {
        Ok(bytes) => response(socket, status, "application/json", &bytes),
        Err(_) => error(socket, "503 Service Unavailable"),
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
    let mut action = shared.actions.observe().0;
    write!(
        socket,
        "HTTP/1.1 200 OK\r\n{SECURITY}Content-Type: text/event-stream\r\n\r\nevent: refresh\ndata: {{}}\n\n"
    )?;
    let deadline = Instant::now() + EVENT_LIFETIME;
    let mut heartbeat = Instant::now();
    while !shared.stop.load(Ordering::Acquire) && Instant::now() < deadline {
        let mut changed = false;
        let mut resync = false;
        let mut models = false;
        let mut devices = false;
        for i in 0..64 {
            match cursor.poll(if i == 0 { POLL } else { Duration::ZERO }) {
                ControlPoll::Pending => break,
                ControlPoll::Event(event) => {
                    changed = true;
                    models |= inventory::models_changed(event.kind);
                    devices |= inventory::devices_changed(event.kind);
                }
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
        let observed_action = shared.actions.observe().0;
        changed |= observed_action != action;
        action = observed_action;
        if changed || resync {
            // Invalidation only. Clients refresh the same owner snapshot;
            // never treat transient notification delivery as state authority.
            socket.write_all(if resync {
                b"event: resync\ndata: {}\n\n"
            } else {
                b"event: refresh\ndata: {}\n\n"
            })?;
            if !resync {
                if models {
                    socket.write_all(b"event: models\ndata: {}\n\n")?;
                }
                if devices {
                    socket.write_all(b"event: devices\ndata: {}\n\n")?;
                }
            }
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
