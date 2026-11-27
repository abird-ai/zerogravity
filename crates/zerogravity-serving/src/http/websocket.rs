//! Transport only. No realtime model/session semantics are inferred from a
//! valid WebSocket handshake. An actual adapter must admit a continuation
//! before a connection may upgrade.
use super::body::Reply;
use crate::TextService;
use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::{SinkExt, StreamExt};
use hyper::{
    Request, Response,
    body::Incoming,
    upgrade::{OnUpgrade, Upgraded},
};
use hyper_util::rt::TokioIo;
use std::{future::Future, io, pin::Pin, sync::Arc, time::Duration};
use tokio::time::{self, Instant};
use tokio_tungstenite::{
    WebSocketStream,
    tungstenite::{
        Error, Message,
        handshake::server::create_response_with_body,
        protocol::{Role, WebSocketConfig},
    },
};
use zerogravity_engine::router::{ServiceError, pending::ServingCharge};

const MAX_MESSAGE: usize = 64 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(30);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

type SessionFuture = Pin<Box<dyn Future<Output = io::Result<()>> + Send>>;
pub(crate) type Session = Box<dyn FnOnce(Socket) -> SessionFuture + Send>;
pub(crate) type Admission = Arc<dyn Fn(&str) -> Result<Session, ServiceError> + Send + Sync>;

pub(super) struct Pending {
    upgrade: OnUpgrade,
    session: Session,
    limit: usize,
    _charge: ServingCharge,
}

pub(super) fn prepare(
    mut request: Request<Incoming>,
    length: usize,
    limit: usize,
    service: &dyn TextService,
    admission: &Admission,
) -> Result<(Response<Reply>, Pending), ServiceError> {
    let invalid = || ServiceError::new(400, "bad_upgrade", "invalid WebSocket handshake");
    if length != 0 || request.method() != hyper::Method::GET {
        return Err(invalid());
    }
    // The mature handshake helper validates the method/version/upgrade
    // fields; the RFC key must additionally decode to exactly 16 bytes.
    let key = request
        .headers()
        .get("sec-websocket-key")
        .ok_or_else(invalid)?;
    let mut decoded = [0u8; 32];
    if STANDARD.decode_slice(key.as_bytes(), &mut decoded).ok() != Some(16) {
        return Err(invalid());
    }
    let response =
        create_response_with_body(&request, || Reply::Full(None)).map_err(|_| invalid())?;
    let limit = limit.min(MAX_MESSAGE);
    // Connection allowance covers HTTP; separately reserve bounded framing,
    // reassembly and write buffers before admission or upgrade. The adapter
    // owns charges for any messages/state it retains past a call.
    let charge = service.charge_ingress("http-websocket", (64 * 1024 + 8 * limit) as u64)?;
    let session = admission(request.uri().path_and_query().map_or("", |v| v.as_str()))?;
    let upgrade = hyper::upgrade::on(&mut request);
    Ok((
        response,
        Pending {
            upgrade,
            session,
            limit,
            _charge: charge,
        },
    ))
}

impl Pending {
    pub(super) async fn run(self) -> io::Result<()> {
        let io = time::timeout(HANDSHAKE_TIMEOUT, self.upgrade)
            .await
            .map_err(|_| io::ErrorKind::TimedOut)?
            .map_err(io::Error::other)?;
        let config = WebSocketConfig::default()
            .read_buffer_size(4096)
            .write_buffer_size(0)
            .max_write_buffer_size(self.limit + 1024)
            .max_message_size(Some(self.limit))
            .max_frame_size(Some(self.limit));
        // Upgraded retains the original Socket, including its remaining
        // wire budget and stalled-write deadline. The enclosing HTTP task
        // owns this future and joins it on peer-close or listener shutdown.
        let stream =
            WebSocketStream::from_raw_socket(TokioIo::new(io), Role::Server, Some(config)).await;
        (self.session)(Socket {
            stream,
            limit: self.limit,
        })
        .await
    }
}

/// Direct message IO; no unbounded send queue.
pub(crate) struct Socket {
    stream: WebSocketStream<TokioIo<Upgraded>>,
    limit: usize,
}
impl Socket {
    pub(crate) fn limit(&self) -> usize {
        self.limit
    }
    pub(crate) async fn receive(&mut self) -> io::Result<Option<Message>> {
        self.receive_until(Instant::now() + IO_TIMEOUT).await
    }
    pub(crate) async fn receive_until(&mut self, deadline: Instant) -> io::Result<Option<Message>> {
        loop {
            if Instant::now() >= deadline {
                return Err(io::ErrorKind::TimedOut.into());
            }
            let next = time::timeout_at(deadline, self.stream.next())
                .await
                .map_err(|_| io::ErrorKind::TimedOut)?;
            match next {
                Some(Ok(message @ (Message::Text(_) | Message::Binary(_)))) => {
                    return Ok(Some(message));
                }
                Some(Ok(Message::Ping(_))) => {
                    // Tungstenite queues the protocol pong automatically.
                    time::timeout_at(deadline, self.stream.flush())
                        .await
                        .map_err(|_| io::ErrorKind::TimedOut)?
                        .map_err(io::Error::other)?;
                }
                Some(Ok(Message::Pong(_))) => {}
                Some(Ok(Message::Close(_))) => {
                    match time::timeout_at(deadline, self.stream.flush()).await {
                        Ok(Ok(()) | Err(Error::ConnectionClosed)) => return Ok(None),
                        Ok(Err(error)) => return Err(io::Error::other(error)),
                        Err(_) => return Err(io::ErrorKind::TimedOut.into()),
                    }
                }
                None | Some(Err(Error::ConnectionClosed)) => return Ok(None),
                Some(Err(error)) => return Err(io::Error::other(error)),
                Some(Ok(Message::Frame(_))) => {
                    return Err(io::Error::other("unexpected raw WebSocket frame"));
                }
            }
        }
    }
    pub(crate) async fn send(&mut self, message: Message) -> io::Result<()> {
        if !matches!(message, Message::Text(_) | Message::Binary(_)) || message.len() > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebSocket message limit or type",
            ));
        }
        time::timeout(IO_TIMEOUT, self.stream.send(message))
            .await
            .map_err(|_| io::ErrorKind::TimedOut)?
            .map_err(io::Error::other)
    }
}
