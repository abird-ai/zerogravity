//! Direct Hyper body over the original charged request receipt. No secondary
//! event queue, blocking receiver, per-token task, or detached producer.
use super::{ErrorReporting, completion_response, epoch};
use bytes::{Buf, Bytes};
use futures_core::Stream;
use hyper::{
    Response, StatusCode,
    body::{Body, Frame, SizeHint},
    header,
};
use serde_json::{Value, json};
use std::{
    future::poll_fn,
    io,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::{sync::oneshot, time};
use zerogravity_engine::router::{
    RequestEvents, ResponseEvent, ServiceError, pending::ServingCharge, token_metrics::TokenMetrics,
};

pub(super) const RESPONSE_TIMEOUT: Duration = Duration::from_secs(180);
pub(super) const KEEPALIVE: Duration = Duration::from_secs(10);

/// Hyper HTTP/1 queue mode advances this buffer only after the socket's
/// successful write. The optional token receipt measures completed SSE
/// payload handoff (including its event delimiter), never body polling or
/// client receipt. It is deliberately absent on ordinary responses.
pub(super) struct Data {
    bytes: Bytes,
    written: Option<oneshot::Sender<Instant>>,
    // Keep output admission and cancellation alive while Hyper retains bytes.
    _owner: Option<Arc<RequestEvents>>,
    _projection: Option<Arc<ServingCharge>>,
}
impl Data {
    fn new(bytes: Vec<u8>, owner: Option<Arc<RequestEvents>>) -> Self {
        Self {
            bytes: bytes.into(),
            written: None,
            _owner: owner,
            _projection: None,
        }
    }
    pub(super) fn projected(
        bytes: Vec<u8>,
        owner: &Arc<RequestEvents>,
        charge: &Arc<ServingCharge>,
    ) -> Self {
        let mut data = Self::new(bytes, Some(owner.clone()));
        data._projection = Some(charge.clone());
        data
    }
    fn sse(value: &Value, owner: &Arc<RequestEvents>) -> Self {
        Self::new(
            format!("data: {value}\n\n").into_bytes(),
            Some(owner.clone()),
        )
    }
    pub(super) fn literal(value: &'static str, owner: &Arc<RequestEvents>) -> Self {
        Self {
            bytes: Bytes::from_static(value.as_bytes()),
            written: None,
            _owner: Some(owner.clone()),
            _projection: None,
        }
    }
}
impl Buf for Data {
    fn remaining(&self) -> usize {
        self.bytes.remaining()
    }
    fn chunk(&self) -> &[u8] {
        self.bytes.chunk()
    }
    fn advance(&mut self, cnt: usize) {
        self.bytes.advance(cnt);
        if !self.bytes.has_remaining()
            && let Some(written) = self.written.take()
        {
            let _ = written.send(Instant::now());
        }
    }
}

pub(super) enum Reply {
    Full(Option<Data>),
    Stream(Pin<Box<dyn Stream<Item = io::Result<Frame<Data>>> + Send>>),
}
impl Body for Reply {
    type Data = Data;
    type Error = io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<io::Result<Frame<Data>>>> {
        match &mut *self {
            Self::Full(data) => Poll::Ready(data.take().map(|data| Ok(Frame::data(data)))),
            Self::Stream(stream) => stream.as_mut().poll_next(cx),
        }
    }
    fn size_hint(&self) -> SizeHint {
        match self {
            Self::Full(data) => {
                SizeHint::with_exact(data.as_ref().map_or(0, |data| data.remaining() as u64))
            }
            Self::Stream(_) => SizeHint::new(),
        }
    }
    fn is_end_stream(&self) -> bool {
        matches!(self, Self::Full(None))
    }
}
/// The Bytes owner can itself retain a protocol record; projected lists
/// carry their independent serialization reservation until the last write.
pub(super) fn retained_json(bytes: Bytes, charge: Option<Arc<ServingCharge>>) -> Response<Reply> {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CACHE_CONTROL, "no-store")
        .header("x-content-type-options", "nosniff")
        .body(Reply::Full(Some(Data {
            bytes,
            written: None,
            _owner: None,
            _projection: charge,
        })))
        .expect("fixed JSON headers")
}

pub(super) fn json_response(
    status: u16,
    value: &Value,
    owner: Option<Arc<RequestEvents>>,
) -> Response<Reply> {
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CACHE_CONTROL, "no-store")
        .header("x-content-type-options", "nosniff")
        .body(Reply::Full(Some(Data::new(
            serde_json::to_vec(value).expect("Value encoding"),
            owner,
        ))))
        .expect("fixed HTTP response headers")
}

pub(super) async fn next(
    events: &RequestEvents,
) -> Result<ResponseEvent, std::sync::mpsc::RecvError> {
    poll_fn(|cx| events.events.poll_recv(cx)).await
}
pub(super) async fn completion(
    events: RequestEvents,
    chat: bool,
) -> Result<Response<Reply>, ServiceError> {
    let mut metrics = events
        .token_metrics
        .clone()
        .map(|request| TokenMetrics::new(request, format!("chatcmpl-zg-{}", events.id)));
    let deadline = time::Instant::now() + RESPONSE_TIMEOUT;
    loop {
        let event = time::timeout_at(deadline, next(&events))
            .await
            .map_err(|_| ServiceError::new(504, "timeout", "generation deadline reached"))?
            .map_err(|_| {
                ServiceError::new(503, "worker_unavailable", "model worker disconnected")
            })?;
        match event {
            ResponseEvent::TokenText { token, .. } => {
                metrics
                    .as_mut()
                    .ok_or_else(|| {
                        ServiceError::new(
                            500,
                            "unexpected_token_observation",
                            "unrequested token observation",
                        )
                    })?
                    .record(token)?;
            }
            ResponseEvent::Text(_) => {}
            ResponseEvent::Done {
                output,
                prompt_tokens,
                completion_tokens,
                finish_reason,
                ..
            } => {
                let mut value = completion_response(
                    events.id,
                    &events.model,
                    chat,
                    &output,
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                );
                if let Some(observations) = &mut metrics {
                    let (aggregate, records) = observations.finish(completion_tokens)?;
                    value["choices"][0]["token_ids"] = json!(observations.token_ids());
                    value["metrics"] = aggregate;
                    value["zerogravity_token_metrics"] = records;
                }
                return Ok(json_response(200, &value, Some(Arc::new(events))));
            }
            ResponseEvent::Error(error) => return Err(error),
        }
    }
}

pub(super) fn streaming(
    request: RequestEvents,
    chat: bool,
    include_usage: bool,
    errors: ErrorReporting,
    authorized: bool,
) -> Response<Reply> {
    let id = format!("chatcmpl-zg-{}", request.id);
    let mut metrics = request
        .token_metrics
        .clone()
        .map(|m| TokenMetrics::new(m, id.clone()));
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache, no-transform")
        .header("x-accel-buffering", "no");
    if let Some(m) = &metrics {
        response = response.header("x-request-id", m.wire_request_id());
    }
    let owner = Arc::new(request);
    let stream = async_stream::try_stream! {
        let created = epoch();
        let kind = if chat { "chat.completion.chunk" } else { "text_completion" };
        if chat {
            yield Frame::data(Data::sse(&json!({
                "id":id,"object":kind,"created":created,"model":owner.model,
                "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]
            }), &owner));
        }
        let deadline = time::Instant::now() + RESPONSE_TIMEOUT;
        loop {
            let now = time::Instant::now();
            if now >= deadline {
                yield Frame::data(Data::sse(&errors.report(&ServiceError::new(
                    504,"timeout","stream generation deadline reached"), authorized), &owner));
                break;
            }
            let event = match time::timeout_at(deadline.min(now + KEEPALIVE), next(&owner)).await {
                Ok(Ok(event)) => event,
                Ok(Err(_)) => {
                    yield Frame::data(Data::sse(&errors.report(&ServiceError::new(
                        503,"worker_unavailable","worker disconnected before completion"), authorized), &owner));
                    break;
                }
                Err(_) if time::Instant::now() < deadline => {
                    yield Frame::data(Data::literal(": keep-alive\n\n", &owner));
                    continue;
                }
                Err(_) => continue,
            };
            match event {
                ResponseEvent::TokenText { text, token } => {
                    let token_id = token.id;
                    let Some(observations) = metrics.as_mut() else {
                        yield Frame::data(Data::sse(&errors.report(&ServiceError::new(
                            500,"unexpected_token_observation","unrequested token observation"), authorized), &owner));
                        break;
                    };
                    if let Err(error) = observations.record(token) {
                        yield Frame::data(Data::sse(&errors.report(&error, authorized), &owner));
                        break;
                    }
                    let choice = if chat {
                        json!({"index":0,"delta":{"content":text},"token_ids":[token_id],"finish_reason":null})
                    } else {
                        json!({"index":0,"text":text,"token_ids":[token_id],"finish_reason":null})
                    };
                    let mut data = Data::sse(&json!({
                        "id":id,"object":kind,"created":created,"model":owner.model,"choices":[choice]
                    }), &owner);
                    let (written, observed) = oneshot::channel();
                    data.written = Some(written);
                    yield Frame::data(data);
                    // Do not record the next observation or build terminal
                    // metrics before the actual preceding token handoff.
                    let written = observed.await.map_err(|_| io::Error::other("token frame not written"))?;
                    observations.mark_written_at(written);
                }
                ResponseEvent::Text(text) => {
                    let choice = if chat {
                        json!({"index":0,"delta":{"content":text},"finish_reason":null})
                    } else {
                        json!({"index":0,"text":text,"finish_reason":null})
                    };
                    yield Frame::data(Data::sse(&json!({
                        "id":id,"object":kind,"created":created,"model":owner.model,"choices":[choice]
                    }), &owner));
                }
                ResponseEvent::Done { prompt_tokens, completion_tokens, finish_reason, .. } => {
                    let observed = if let Some(observations) = metrics.as_mut() {
                        match observations.finish(completion_tokens) {
                            Ok(value) => Some(value),
                            Err(error) => {
                                yield Frame::data(Data::sse(&errors.report(&error, authorized), &owner));
                                break;
                            }
                        }
                    } else { None };
                    let choice = if chat {
                        json!({"index":0,"delta":{},"finish_reason":finish_reason})
                    } else {
                        json!({"index":0,"text":"","finish_reason":finish_reason})
                    };
                    yield Frame::data(Data::sse(&json!({
                        "id":id,"object":kind,"created":created,"model":owner.model,"choices":[choice]
                    }), &owner));
                    if include_usage || observed.is_some() {
                        let mut usage = json!({
                            "id":id,"object":kind,"created":created,"model":owner.model,"choices":[],
                            "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,
                                     "total_tokens":prompt_tokens+completion_tokens}
                        });
                        if let Some((aggregate, records)) = observed {
                            usage["metrics"] = aggregate;
                            usage["zerogravity_token_metrics"] = records;
                        }
                        yield Frame::data(Data::sse(&usage, &owner));
                    }
                    break;
                }
                ResponseEvent::Error(error) => {
                    yield Frame::data(Data::sse(&errors.report(&error, authorized), &owner));
                    break;
                }
            }
        }
        yield Frame::data(Data::literal("data: [DONE]\n\n", &owner));
    };
    response
        .body(Reply::Stream(Box::pin(stream)))
        .expect("fixed SSE headers")
}
