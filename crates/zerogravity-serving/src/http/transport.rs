//! Owned HTTP/1 transport. Hyper owns protocol framing; the existing service
//! owns all model admission, events and cancellation. No detached tasks.
use super::{ErrorReporting, bind_token_metrics, body, ingress_body, parse_body, websocket};
use crate::{TextService, ingress::IngressLimits, metrics::Counters};
use hyper::{
    Method, Request, Response, StatusCode, Version,
    body::Incoming,
    header::{self, HeaderMap},
    server::conn::http1,
    service::service_fn,
};
use hyper_util::rt::{TokioIo, TokioTimer};
use serde_json::json;
use std::{
    convert::Infallible,
    future::Future,
    io::{self, IoSlice},
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncWrite, AsyncWriteExt, Interest, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::Notify,
    task::JoinSet,
    time,
};
use zerogravity_engine::{
    router::{ServiceError, pending::ServingCharge},
    state_cache::PrefixSharingScope,
};

const MAX_HEADERS: usize = 8 * 1024;
const MAX_PARSED_HEADERS: usize = 32;
const MAX_REQUESTS_PER_CONNECTION: usize = 256;
const READ_TIMEOUT: Duration = Duration::from_secs(10);
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
const REFUSAL_TIMEOUT: Duration = Duration::from_millis(50);

// Bounded external-runtime allowance, not an exact Tokio/Hyper heap census.
// This covers the task, header parser/maps, bounded read/write buffers and
// their growth overlap, and the one optional outstanding token write receipt.
// Native stacks, kernel socket buffers and runtime internals are external
// headroom. Retained application JSON/input/output allocations are still
// priced through the original serving ledger, before publication.
pub(crate) const SERVER_HEADROOM: u64 = 64 * 1024;
pub(super) const CONNECTION_HEADROOM: u64 = 64 * 1024;
const HTTP_HEADER_ENVELOPE: u64 = (MAX_HEADERS
    + 4 * std::mem::size_of::<ServingCharge>()
    + 4 * std::mem::size_of::<usize>()) as u64;

pub(crate) struct ServeConfig {
    pub api_key: Option<String>,
    pub enable_per_request_metrics: bool,
    pub errors: ErrorReporting,
    pub prefix_sharing_scope: Option<PrefixSharingScope>,
    pub limits: IngressLimits,
    pub realtime: Option<websocket::Admission>,
}

#[derive(Default)]
pub(crate) struct Shutdown {
    stopped: AtomicBool,
    wake: Notify,
}
impl Shutdown {
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        self.wake.notify_one();
    }
    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }
    async fn requested(&self) {
        if !self.is_stopped() {
            self.wake.notified().await;
        }
    }
    fn check(&self) -> Result<(), ServiceError> {
        if self.is_stopped() {
            Err(ServiceError::new(
                503,
                "shutting_down",
                "service is stopping",
            ))
        } else {
            Ok(())
        }
    }
}

/// Hard lifetime wire budget, including Hyper's response headers, all
/// persistent responses and chunk framing. A stalled write has a progress
/// deadline; actual successful writes debit only the bytes accepted by TCP.
struct Socket {
    inner: TcpStream,
    remaining: usize,
    blocked: Option<Pin<Box<time::Sleep>>>,
}
impl Socket {
    fn new(inner: TcpStream, remaining: usize) -> Self {
        Self {
            inner,
            remaining,
            blocked: None,
        }
    }
    fn check_write(&self, offered: usize) -> io::Result<()> {
        if offered > self.remaining {
            Err(io::Error::other("HTTP output byte limit reached"))
        } else {
            Ok(())
        }
    }
    fn written(
        &mut self,
        cx: &mut Context<'_>,
        result: Poll<io::Result<usize>>,
    ) -> Poll<io::Result<usize>> {
        match result {
            Poll::Ready(Ok(n)) => {
                self.remaining -= n;
                if n != 0 {
                    self.blocked = None;
                }
                Poll::Ready(Ok(n))
            }
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Pending => {
                let timer = self
                    .blocked
                    .get_or_insert_with(|| Box::pin(time::sleep(WRITE_TIMEOUT)));
                if timer.as_mut().poll(cx).is_ready() {
                    Poll::Ready(Err(io::ErrorKind::TimedOut.into()))
                } else {
                    Poll::Pending
                }
            }
        }
    }
}
impl AsyncRead for Socket {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}
impl AsyncWrite for Socket {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.check_write(buf.len())?;
        let result = Pin::new(&mut self.inner).poll_write(cx, buf);
        self.written(cx, result)
    }
    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let offered = bufs
            .iter()
            .try_fold(0usize, |total, buf| total.checked_add(buf.len()))
            .ok_or_else(|| io::Error::other("HTTP output length overflow"))?;
        self.check_write(offered)?;
        let result = Pin::new(&mut self.inner).poll_write_vectored(cx, bufs);
        self.written(cx, result)
    }
    fn is_write_vectored(&self) -> bool {
        true
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

fn header_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
}

/// Some(length) is fixed length, None is strictly decoded chunked input.
pub(super) fn validate_headers(
    parts: &hyper::http::request::Parts,
    limit: usize,
    accepts_chunked: bool,
) -> Result<Option<usize>, ServiceError> {
    let path = parts.uri.path_and_query().map_or("", |uri| uri.as_str());
    if parts.version != Version::HTTP_11 {
        return Err(ServiceError::new(
            400,
            "bad_request",
            "only bounded HTTP/1.1 is supported",
        ));
    }
    if path.len() > 1024
        || !path.starts_with('/')
        || path.contains('#')
        || parts.uri.scheme().is_some()
        || parts.uri.authority().is_some()
    {
        return Err(ServiceError::new(400, "bad_path", "invalid request target"));
    }
    // The pinned Hyper server parser rejects mixed framing and repeated
    // lengths before normalization. Apply duplicate-header policy here to
    // the remaining exposed fields.
    let mut bytes = parts.method.as_str().len() + path.len() + 16;
    for name in parts.headers.keys() {
        if parts.headers.get_all(name).iter().count() != 1 {
            return Err(ServiceError::new(
                400,
                "duplicate_header",
                "duplicate HTTP headers forbidden",
            ));
        }
        let value = &parts.headers[name];
        if value.to_str().is_err() {
            return Err(ServiceError::new(
                400,
                "bad_header",
                "invalid header encoding",
            ));
        }
        bytes += name.as_str().len() + value.as_bytes().len() + 4;
    }
    if bytes > MAX_HEADERS || parts.headers.len() > MAX_PARSED_HEADERS {
        return Err(ServiceError::new(
            431,
            "header_too_large",
            "HTTP headers too large",
        ));
    }
    if parts.headers.contains_key(header::EXPECT) {
        return Err(ServiceError::new(
            400,
            "unsupported_transfer",
            "Expect is not supported",
        ));
    }
    if !parts.headers.contains_key(header::HOST) {
        return Err(ServiceError::new(
            400,
            "bad_request",
            "HTTP/1.1 Host header required",
        ));
    }
    if let Some(encoding) = header_value(&parts.headers, "transfer-encoding") {
        // The selected route explicitly admits chunked input. No other
        // transfer codings or trailers, and no body on an upgrade.
        if !encoding.eq_ignore_ascii_case("chunked")
            || parts.headers.contains_key(header::CONTENT_LENGTH)
            || parts.headers.contains_key(header::TRAILER)
            || parts.headers.contains_key(header::UPGRADE)
            || parts.method != Method::POST
            || !accepts_chunked
        {
            return Err(ServiceError::new(
                400,
                "unsupported_transfer",
                "only chunked JSON text requests are supported",
            ));
        }
        return Ok(None);
    }
    let length = match header_value(&parts.headers, "content-length") {
        Some(value) => {
            if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return Err(ServiceError::new(
                    400,
                    "bad_length",
                    "invalid Content-Length",
                ));
            }
            value
                .parse::<usize>()
                .map_err(|_| ServiceError::new(413, "bad_length", "Content-Length overflow"))?
        }
        None if parts.method == Method::POST => {
            return Err(ServiceError::new(
                411,
                "length_required",
                "POST requires Content-Length or chunked encoding",
            ));
        }
        None => 0,
    };
    if length > limit {
        return Err(ServiceError::new(
            413,
            "body_too_large",
            "JSON body too large",
        ));
    }
    Ok(Some(length))
}

pub(super) fn authorize(headers: &HeaderMap, api_key: Option<&str>) -> Result<(), ServiceError> {
    if let Some(expected) = api_key {
        let received = header_value(headers, "authorization")
            .and_then(|value| value.strip_prefix("Bearer "))
            .unwrap_or("")
            .as_bytes();
        let expected = expected.as_bytes();
        let mut difference = expected.len() ^ received.len();
        for i in 0..expected.len().max(received.len()) {
            difference |= (expected.get(i).copied().unwrap_or(0)
                ^ received.get(i).copied().unwrap_or(0)) as usize;
        }
        if difference != 0 {
            return Err(ServiceError::new(
                401,
                "unauthorized",
                "Bearer token required",
            ));
        }
    }
    Ok(())
}

struct Host {
    service: Arc<dyn TextService>,
    config: ServeConfig,
    shutdown: Arc<Shutdown>,
    metrics: Arc<Counters>,
}
struct Connection {
    host: Arc<Host>,
    requests: AtomicUsize,
    upgrade: Mutex<Option<websocket::Pending>>,
}

impl Connection {
    async fn request(&self, incoming: Request<Incoming>) -> Response<body::Reply> {
        let host = &self.host;
        let number = self.requests.fetch_add(1, Ordering::Relaxed) + 1;
        let mut authorized = false;
        let result = async {
            host.shutdown.check()?;
            let (parts, incoming) = incoming.into_parts();
            let path = parts.uri.path_and_query().map_or("", |uri| uri.as_str());
            let accepts_chunked = matches!(path, "/v1/completions" | "/v1/chat/completions");
            let length = validate_headers(&parts, host.config.limits.body_bytes, accepts_chunked)?;
            // Auth before any body read, allocation or model admission.
            host.metrics.accept_request();
            authorize(&parts.headers, host.config.api_key.as_deref())?;
            // Structural ingress errors never receive diagnostic details.
            let header_charge = host
                .service
                .charge_ingress("http-headers", HTTP_HEADER_ENVELOPE)?;
            let mut ingress = Vec::with_capacity(4);
            ingress.push(header_charge);
            if parts.uri.path() == "/v1/realtime" {
                let admission =
                    host.config.realtime.as_ref().ok_or_else(|| {
                        ServiceError::new(404, "not_found", "unsupported endpoint")
                    })?;
                let request = Request::from_parts(parts, incoming);
                let (response, pending) = websocket::prepare(
                    request,
                    length.ok_or_else(|| {
                        ServiceError::new(400, "bad_upgrade", "upgrade body is not supported")
                    })?,
                    host.config.limits.body_bytes,
                    host.service.as_ref(),
                    admission,
                )?;
                host.shutdown.check()?;
                // The admitted continuation is retained by this connection,
                // not by a spawned upgrade task.
                *self.upgrade.lock().expect("upgrade ownership") = Some(pending);
                return Ok(response);
            }
            if parts.headers.contains_key(header::UPGRADE) {
                return Err(ServiceError::new(
                    400,
                    "unsupported_upgrade",
                    "upgrade not supported on this endpoint",
                ));
            }
            let reader = ingress_body::Reader::new(
                incoming,
                length,
                host.config.limits.body_bytes,
                READ_TIMEOUT,
            )?;
            let (raw, used, body_charge) = ingress_body::collect(reader, host.service.as_ref())
                .await?
                .into_parts();
            ingress.push(body_charge);
            authorized = host.config.api_key.is_some();
            host.shutdown.check()?;
            let path = parts.uri.path_and_query().map_or("", |uri| uri.as_str());
            if parts.method == Method::GET && path == "/health" {
                return Ok(body::json_response(
                    200,
                    &json!({
                        "status":"ok","admission":host.service.admission()?,
                        "per_request_metrics_enabled":host.config.enable_per_request_metrics,
                    }),
                    None,
                ));
            }
            if parts.method == Method::GET && path == "/v1/models" {
                let data = host
                    .service
                    .model_ids()?
                    .into_iter()
                    .map(
                        |id| json!({"id":id,"object":"model","created":0,"owned_by":"zerogravity"}),
                    )
                    .collect::<Vec<_>>();
                return Ok(body::json_response(
                    200,
                    &json!({"object":"list","data":data}),
                    None,
                ));
            }
            let chat = path == "/v1/chat/completions";
            if parts.method != Method::POST || !(chat || path == "/v1/completions") {
                return Err(ServiceError::new(404, "not_found", "unsupported endpoint"));
            }
            if !header_value(&parts.headers, "content-type").is_some_and(|t| {
                t.eq_ignore_ascii_case("application/json")
                    || t.to_ascii_lowercase().starts_with("application/json;")
            }) {
                return Err(ServiceError::new(
                    415,
                    "unsupported_media_type",
                    "JSON Content-Type required",
                ));
            }
            let (mut task, usage) = parse_body(
                &raw[..used],
                chat,
                Some(host.service.as_ref()),
                &mut ingress,
            )?;
            if chat {
                host.service.validate_chat(&task.model)?;
            }
            bind_token_metrics(
                &mut task,
                header_value(&parts.headers, "x-request-id"),
                host.config.enable_per_request_metrics,
            )?;
            let stream = task.stream;
            // Request-owned ingress credits transfer into the original queue.
            drop(raw);
            drop(parts);
            host.shutdown.check()?;
            let events = match host.config.prefix_sharing_scope {
                Some(scope) => host.service.request_admitted_scoped(task, ingress, scope)?,
                None => host.service.request_admitted(task, ingress)?,
            };
            if stream {
                Ok(body::streaming(
                    events,
                    chat,
                    usage,
                    host.config.errors.clone(),
                    authorized,
                ))
            } else {
                body::completion(events, chat).await
            }
        }
        .await;
        let (mut response, close) = match result {
            Ok(response) => (response, false),
            Err(error) => (
                body::json_response(
                    error.status,
                    &host.config.errors.report(&error, authorized),
                    None,
                ),
                true,
            ),
        };
        if response.status() != StatusCode::SWITCHING_PROTOCOLS
            && (close || number >= MAX_REQUESTS_PER_CONNECTION || host.shutdown.is_stopped())
        {
            response.headers_mut().insert(
                header::CONNECTION,
                header::HeaderValue::from_static("close"),
            );
        }
        response
    }
}

/// Hyper intentionally stops reading once a pipelined head is buffered behind
/// an active response. Observe peer closure even then, without consuming any
/// bytes or introducing a second HTTP parser. The bounded duplicate socket is
/// the same physical connection, not another connection/task. Normally this
/// parks on readiness/peek; retry only while actual unread bytes remain.
async fn peer_closed(peer: &TcpStream) -> io::Result<()> {
    loop {
        let ready = peer.ready(Interest::READABLE).await?;
        if ready.is_read_closed() || ready.is_error() {
            return Ok(());
        }
        if peer.peek(&mut [0u8; 1]).await? == 0 {
            return Ok(());
        }
        time::sleep(Duration::from_millis(50)).await;
    }
}

async fn connection(stream: TcpStream, peer: TcpStream, host: Arc<Host>, _charge: ServingCharge) {
    let _active = host.metrics.connection();
    let socket = Socket::new(stream, host.config.limits.output_bytes);
    let state = Arc::new(Connection {
        host: host.clone(),
        requests: AtomicUsize::new(0),
        upgrade: Mutex::new(None),
    });
    let request_state = state.clone();
    let service = service_fn(move |request| {
        let state = request_state.clone();
        async move { Ok::<_, Infallible>(state.request(request).await) }
    });
    // Explicit queue/writev is also the token-delivery observation contract:
    // Body Data advances after successful socket writes, not on buffer copy.
    let connection = http1::Builder::new()
        .keep_alive(true)
        .half_close(false)
        .timer(TokioTimer::new())
        .header_read_timeout(READ_TIMEOUT)
        .max_headers(MAX_PARSED_HEADERS)
        .max_header_size(MAX_HEADERS)
        .max_buf_size(MAX_HEADERS)
        .writev(true)
        .title_case_headers(true)
        .auto_date_header(false)
        .serve_connection(TokioIo::new(socket), service)
        .with_upgrades();
    let work = async {
        if connection.await.is_ok() {
            let pending = state.upgrade.lock().expect("upgrade ownership").take();
            if let Some(pending) = pending {
                let _ = pending.run().await;
            }
        }
    };
    tokio::select! {
        _ = work => {},
        _ = peer_closed(&peer) => {},
    }
}

/// Register one bounded, read-only peer-close observer. Setup failure follows
/// the same listener-failure join path as the former interrupt-socket setup.
fn split_peer(stream: TcpStream) -> io::Result<(TcpStream, TcpStream)> {
    stream.set_nodelay(true)?;
    let stream = stream.into_std()?;
    let peer = stream.try_clone()?;
    Ok((TcpStream::from_std(stream)?, TcpStream::from_std(peer)?))
}

/// One short overload response in the accept owner, with no new handler.
/// The bound and deadline apply even to a peer that never sends/reads data.
async fn refuse(mut stream: TcpStream, host: &Host, error: ServiceError) {
    let body = host.config.errors.report(&error, false).to_string();
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        status.as_u16(),
        status.canonical_reason().unwrap_or("Error"),
        body.len()
    );
    if header.len() + body.len() <= host.config.limits.output_bytes {
        let _ = time::timeout(REFUSAL_TIMEOUT, async {
            stream.write_all(header.as_bytes()).await?;
            stream.write_all(body.as_bytes()).await
        })
        .await;
    }
}

async fn accept_loop(listener: TcpListener, host: Arc<Host>) -> io::Result<()> {
    let mut tasks = JoinSet::new();
    let result = loop {
        if host.shutdown.is_stopped() {
            break Ok(());
        }
        tokio::select! {
            biased;
            _ = host.shutdown.requested() => break Ok(()),
            joined = tasks.join_next(), if !tasks.is_empty() => {
                if let Some(Err(error)) = joined {
                    break Err(io::Error::other(error));
                }
            }
            next = listener.accept() => {
                let (stream, _) = match next {
                    Ok(accepted) => accepted,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => break Err(error),
                };
                if host.shutdown.is_stopped() { break Ok(()); }
                if tasks.len() >= host.config.limits.connections {
                    refuse(stream, &host, ServiceError::new(503, "connections_full", "connection admission capacity reached")).await;
                    continue;
                }
                let charge = match host.service.charge_ingress("http-connection", CONNECTION_HEADROOM) {
                    Ok(charge) => charge,
                    Err(error) => {
                        refuse(stream, &host, error).await;
                        continue;
                    }
                };
                let (stream, peer) = match split_peer(stream) {
                    Ok(pair) => pair,
                    Err(error) => break Err(error),
                };
                tasks.spawn(connection(stream, peer, host.clone(), charge));
            }
        }
    };
    // Stop ingress first; cancel only protocol-owned requests and join all
    // tasks. Aborting a connection drops its body/future/receipt, including
    // idle/partial reads and backpressured writes. No spawn_blocking work.
    drop(listener);
    host.shutdown.stop();
    tasks.abort_all();
    let mut joined = Ok(());
    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result
            && !error.is_cancelled()
        {
            joined = Err(io::Error::other(error));
        }
    }
    result.and(joined)
}

pub(crate) fn serve(
    listener: std::net::TcpListener,
    service: Arc<dyn TextService>,
    shutdown: Arc<Shutdown>,
    config: ServeConfig,
    metrics: Arc<Counters>,
) -> io::Result<()> {
    // Single owned async runtime. Dropping it on unwind also drops all its
    // tasks; no model work is ever executed on the protocol runtime.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let listener = TcpListener::from_std(listener)?;
        accept_loop(
            listener,
            Arc::new(Host {
                service,
                config,
                shutdown,
                metrics,
            }),
        )
        .await
    })
}
