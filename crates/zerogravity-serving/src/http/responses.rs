//! OpenAI Responses projection of the original typed text event receipt.
//! Optional protocol record; no secondary scheduler, producer task or event queue.
mod background;
mod input;
mod store;
pub(super) use background::{Jobs, start as start_background};
pub(super) use input::{count, parse};
pub(super) use store::{Pending, Store, conversation_route, route as resource_route};

use super::{
    ErrorReporting,
    body::{self, Data, Reply},
    epoch,
};
use crate::TextService;
use hyper::{Response, StatusCode, body::Frame, header};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::{self, Write},
    sync::Arc,
};
use tokio::time;
use zerogravity_engine::router::{
    RequestEvents, ResponseEvent, ServiceError, pending::ServingCharge,
};

type Result<T> = std::result::Result<T, ServiceError>;

pub(super) struct Options {
    instructions: Option<String>,
    metadata: BTreeMap<String, String>,
    max_tokens: usize,
    tool_choice: &'static str,
    parallel_tool_calls: bool,
    store: bool,
    pub(super) background: bool,
    previous: Option<String>,
    conversation: Option<String>,
    _charge: Arc<ServingCharge>,
}

// One bounded projection reservation, held by the producer and any retained
// wire buffers. Borrow text in serde views rather than clone it into JSON
// trees. Geometric capacity, old/new overlap, partial-output accumulation and
// small metadata/container headroom are priced conservatively.
pub(super) struct Wire {
    remaining: usize,
    charge: Arc<ServingCharge>,
}
impl Wire {
    fn new(service: &dyn TextService, limit: usize, metadata_bytes: usize) -> Result<Self> {
        let bytes = limit
            .checked_mul(6)
            .and_then(|n| n.checked_add(metadata_bytes))
            .and_then(|n| n.checked_add(64 * 1024))
            .ok_or_else(|| {
                ServiceError::new(400, "size_overflow", "Responses projection bound overflow")
            })?;
        let charge = service.charge_ingress("http-responses-projection", bytes as u64)?;
        Ok(Self {
            remaining: limit,
            charge: Arc::new(charge),
        })
    }

    fn encode<T: Serialize>(&mut self, owner: &Arc<RequestEvents>, value: &T) -> io::Result<Data> {
        let mut buffer = BoundedWriter {
            bytes: Vec::new(),
            limit: self.remaining,
        };
        serde_json::to_writer(&mut buffer, value).map_err(io::Error::other)?;
        self.remaining -= buffer.bytes.len();
        Ok(Data::projected(buffer.bytes, owner, &self.charge))
    }
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let end = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|end| *end <= self.limit)
            .ok_or_else(|| io::Error::other("Responses output byte limit"))?;
        if end > self.bytes.capacity() {
            let capacity = self
                .bytes
                .capacity()
                .saturating_mul(2)
                .max(end)
                .min(self.limit);
            self.bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .map_err(io::Error::other)?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct Identity {
    response: String,
    message: String,
    created: u64,
}
impl Identity {
    fn new(request_id: &str) -> Self {
        let suffix = request_id.strip_prefix("req_").unwrap_or(request_id);
        Self {
            response: format!("resp_{suffix}"),
            message: format!("msg_{suffix}"),
            created: epoch(),
        }
    }
}

#[derive(Serialize)]
struct TextPart<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
    annotations: [(); 0],
    logprobs: [(); 0],
}
impl<'a> TextPart<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            kind: "output_text",
            text,
            annotations: [],
            logprobs: [],
        }
    }
}
#[derive(Serialize)]
struct Message<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    status: &'static str,
    content: &'a [TextPart<'a>],
}
impl<'a> Message<'a> {
    fn new(identity: &'a Identity, status: &'static str, content: &'a [TextPart<'a>]) -> Self {
        Self {
            id: &identity.message,
            kind: "message",
            role: "assistant",
            status,
            content,
        }
    }
}

#[derive(Serialize)]
struct InputDetails {
    cached_tokens: usize,
}
#[derive(Serialize)]
struct OutputDetails {
    reasoning_tokens: usize,
}
#[derive(Serialize)]
struct Usage {
    input_tokens: usize,
    input_tokens_details: InputDetails,
    output_tokens: usize,
    output_tokens_details: OutputDetails,
    total_tokens: usize,
}
impl Usage {
    fn new(prompt: usize, cached: usize, output: usize) -> Result<Self> {
        if cached > prompt {
            return Err(ServiceError::new(
                500,
                "invalid_usage",
                "restored prefix exceeds prompt",
            ));
        }
        let total = prompt
            .checked_add(output)
            .ok_or_else(|| ServiceError::new(500, "invalid_usage", "token usage overflow"))?;
        Ok(Self {
            input_tokens: prompt,
            input_tokens_details: InputDetails {
                cached_tokens: cached,
            },
            output_tokens: output,
            // No separate reasoning output capability is advertised.
            output_tokens_details: OutputDetails {
                reasoning_tokens: 0,
            },
            total_tokens: total,
        })
    }
}
#[derive(Serialize)]
struct Format {
    #[serde(rename = "type")]
    kind: &'static str,
}
#[derive(Serialize)]
struct TextConfig {
    format: Format,
}
#[derive(Serialize)]
struct Incomplete {
    reason: &'static str,
}

#[derive(Serialize)]
struct ConversationRef<'a> {
    id: &'a str,
}

#[derive(Serialize)]
struct View<'a> {
    id: &'a str,
    object: &'static str,
    created_at: u64,
    completed_at: Option<u64>,
    status: &'static str,
    error: Option<&'a Value>,
    incomplete_details: Option<Incomplete>,
    instructions: Option<&'a str>,
    max_output_tokens: usize,
    model: &'a str,
    output: &'a [Message<'a>],
    parallel_tool_calls: bool,
    previous_response_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    conversation: Option<ConversationRef<'a>>,
    reasoning: Option<()>,
    store: bool,
    temperature: f64,
    top_p: f64,
    text: TextConfig,
    tool_choice: &'static str,
    tools: [(); 0],
    truncation: &'static str,
    usage: Option<Usage>,
    user: Option<()>,
    metadata: &'a BTreeMap<String, String>,
    background: bool,
    service_tier: &'static str,
    top_logprobs: usize,
}
impl Options {
    fn view<'a>(
        &'a self,
        identity: &'a Identity,
        model: &'a str,
        status: &'static str,
        output: &'a [Message<'a>],
        usage: Option<Usage>,
        error: Option<&'a Value>,
    ) -> View<'a> {
        View {
            id: &identity.response,
            object: "response",
            created_at: identity.created,
            completed_at: (status == "completed").then(epoch),
            status,
            error,
            incomplete_details: (status == "incomplete").then_some(Incomplete {
                reason: "max_output_tokens",
            }),
            instructions: self.instructions.as_deref(),
            max_output_tokens: self.max_tokens,
            model,
            output,
            parallel_tool_calls: self.parallel_tool_calls,
            previous_response_id: self.previous.as_deref(),
            conversation: self
                .conversation
                .as_deref()
                .map(|id| ConversationRef { id }),
            reasoning: None,
            store: self.store,
            temperature: 0.0,
            top_p: 1.0,
            text: TextConfig {
                format: Format { kind: "text" },
            },
            tool_choice: self.tool_choice,
            tools: [],
            truncation: "disabled",
            usage,
            user: None,
            metadata: &self.metadata,
            background: self.background,
            service_tier: "default",
            top_logprobs: 0,
        }
    }
}

fn terminal_status(reason: &str) -> Result<&'static str> {
    match reason {
        "stop" => Ok("completed"),
        "length" => Ok("incomplete"),
        _ => Err(ServiceError::new(
            500,
            "invalid_finish_reason",
            "unexpected text finish reason",
        )),
    }
}
fn response_error(errors: &ErrorReporting, failure: &ServiceError, authorized: bool) -> Value {
    let mut reported = errors.report(failure, authorized);
    let mut failure_object = reported["error"].take();
    failure_object
        .as_object_mut()
        .expect("fixed error envelope")
        .remove("type");
    failure_object
        .as_object_mut()
        .expect("fixed error envelope")
        .remove("param");
    // Response.error uses a narrower canonical code vocabulary than the
    // HTTP error envelope. Original typed details remain opt-in.
    failure_object["code"] = Value::String(
        match failure.code {
            "invalid_messages" | "invalid_prompt" | "context_overflow" => "invalid_prompt",
            "queue_full" | "backpressure" | "connections_full" => "rate_limit_exceeded",
            _ => "server_error",
        }
        .into(),
    );
    failure_object
}

fn wire_error(_: io::Error) -> ServiceError {
    ServiceError::new(
        503,
        "output_limit",
        "Responses projection exceeds output capacity",
    )
}

struct Terminal {
    output: String,
    status: &'static str,
    usage: Usage,
}

async fn terminal(events: &RequestEvents) -> Result<Terminal> {
    let deadline = time::Instant::now() + body::RESPONSE_TIMEOUT;
    loop {
        let event = time::timeout_at(deadline, body::next(events))
            .await
            .map_err(|_| ServiceError::new(504, "timeout", "generation deadline reached"))?
            .map_err(|_| {
                ServiceError::new(503, "worker_unavailable", "model worker disconnected")
            })?;
        match event {
            ResponseEvent::Text(_) => {}
            ResponseEvent::TokenText { .. } => {
                return Err(ServiceError::new(
                    500,
                    "unexpected_token_observation",
                    "unrequested token observation",
                ));
            }
            ResponseEvent::Error(error) => return Err(error),
            ResponseEvent::Done {
                output,
                prompt_tokens,
                cached_prompt_tokens,
                completion_tokens,
                finish_reason,
            } => {
                return Ok(Terminal {
                    output,
                    status: terminal_status(finish_reason)?,
                    usage: Usage::new(prompt_tokens, cached_prompt_tokens, completion_tokens)?,
                });
            }
        }
    }
}

pub(super) async fn completion(
    request: RequestEvents,
    options: Options,
    mut wire: Wire,
    request_id: &str,
    pending: Option<Pending>,
) -> Result<Response<Reply>> {
    let identity = Identity::new(request_id);
    let owner = Arc::new(request);
    let Terminal {
        output,
        status,
        usage,
    } = terminal(&owner).await?;
    let content = [TextPart::new(&output)];
    let items = [Message::new(&identity, status, &content)];
    let view = options.view(&identity, &owner.model, status, &items, Some(usage), None);
    let data = wire.encode(&owner, &view).map_err(wire_error)?;
    if let Some(pending) = pending {
        pending.finish(&view, &output, Some(status))?;
    }
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CACHE_CONTROL, "no-store")
        .header("x-content-type-options", "nosniff")
        .body(Reply::Full(Some(data)))
        .expect("fixed Responses headers"))
}

#[derive(Serialize)]
struct Event<P> {
    #[serde(rename = "type")]
    kind: &'static str,
    sequence_number: usize,
    #[serde(flatten)]
    payload: P,
}
#[derive(Serialize)]
struct ResponsePayload<'a> {
    response: &'a View<'a>,
}
#[derive(Serialize)]
struct ItemPayload<'a> {
    output_index: usize,
    item: Message<'a>,
}
#[derive(Serialize)]
struct PartPayload<'a> {
    item_id: &'a str,
    output_index: usize,
    content_index: usize,
    part: TextPart<'a>,
}
#[derive(Serialize)]
struct TextPayload<'a> {
    item_id: &'a str,
    output_index: usize,
    content_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    logprobs: [(); 0],
}
impl<'a> TextPayload<'a> {
    fn new(identity: &'a Identity, delta: Option<&'a str>, text: Option<&'a str>) -> Self {
        Self {
            item_id: &identity.message,
            output_index: 0,
            content_index: 0,
            delta,
            text,
            logprobs: [],
        }
    }
}
/// The same typed event grammar for direct foreground and retained
/// background projection. Serialization does not own an engine receipt.
enum StreamEvent<'a> {
    Response(&'static str, &'a View<'a>),
    Item(&'static str, Message<'a>),
    Part(&'static str, PartPayload<'a>),
    Text(&'static str, TextPayload<'a>),
}
impl StreamEvent<'_> {
    fn encode(self, sequence: usize, limit: usize) -> io::Result<Vec<u8>> {
        let mut buffer = BoundedWriter {
            bytes: Vec::new(),
            limit,
        };
        self.encode_into(sequence, &mut buffer)?;
        Ok(buffer.bytes)
    }
    fn encode_into(self, sequence: usize, buffer: &mut BoundedWriter) -> io::Result<()> {
        fn encode<T: Serialize>(
            kind: &'static str,
            sequence: usize,
            payload: T,
            buffer: &mut BoundedWriter,
        ) -> io::Result<()> {
            write!(buffer, "event: {kind}\ndata: ")?;
            serde_json::to_writer(
                &mut *buffer,
                &Event {
                    kind,
                    sequence_number: sequence,
                    payload,
                },
            )
            .map_err(io::Error::other)?;
            buffer.write_all(b"\n\n")
        }
        match self {
            Self::Response(kind, view) => {
                encode(kind, sequence, ResponsePayload { response: view }, buffer)
            }
            Self::Item(kind, item) => encode(
                kind,
                sequence,
                ItemPayload {
                    output_index: 0,
                    item,
                },
                buffer,
            ),
            Self::Part(kind, part) => encode(kind, sequence, part, buffer),
            Self::Text(kind, text) => encode(kind, sequence, text, buffer),
        }
    }
}
fn initial_events<'a>(identity: &'a Identity, view: &'a View<'a>) -> [StreamEvent<'a>; 4] {
    [
        StreamEvent::Response("response.created", view),
        StreamEvent::Response("response.in_progress", view),
        StreamEvent::Item(
            "response.output_item.added",
            Message::new(identity, "in_progress", &[]),
        ),
        StreamEvent::Part(
            "response.content_part.added",
            PartPayload {
                item_id: &identity.message,
                output_index: 0,
                content_index: 0,
                part: TextPart::new(""),
            },
        ),
    ]
}
fn delta_event<'a>(identity: &'a Identity, text: &'a str) -> StreamEvent<'a> {
    StreamEvent::Text(
        "response.output_text.delta",
        TextPayload::new(identity, Some(text), None),
    )
}
fn done_events<'a>(
    identity: &'a Identity,
    view: &'a View<'a>,
    output: &'a str,
    status: &'static str,
    content: &'a [TextPart<'a>],
) -> [StreamEvent<'a>; 4] {
    [
        StreamEvent::Text(
            "response.output_text.done",
            TextPayload::new(identity, None, Some(output)),
        ),
        StreamEvent::Part(
            "response.content_part.done",
            PartPayload {
                item_id: &identity.message,
                output_index: 0,
                content_index: 0,
                part: TextPart::new(output),
            },
        ),
        StreamEvent::Item(
            "response.output_item.done",
            Message::new(identity, status, content),
        ),
        StreamEvent::Response(
            if status == "completed" {
                "response.completed"
            } else {
                "response.incomplete"
            },
            view,
        ),
    ]
}
impl Wire {
    fn event(
        &mut self,
        owner: &Arc<RequestEvents>,
        sequence: &mut usize,
        event: StreamEvent<'_>,
    ) -> io::Result<Frame<Data>> {
        let bytes = event.encode(*sequence, self.remaining)?;
        self.remaining -= bytes.len();
        *sequence = sequence
            .checked_add(1)
            .ok_or_else(|| io::Error::other("Responses sequence exhausted"))?;
        Ok(Frame::data(Data::projected(bytes, owner, &self.charge)))
    }
}

pub(super) fn streaming(
    request: RequestEvents,
    options: Options,
    mut wire: Wire,
    request_id: &str,
    errors: ErrorReporting,
    authorized: bool,
    mut pending: Option<Pending>,
) -> Response<Reply> {
    let identity = Identity::new(request_id);
    let owner = Arc::new(request);
    let stream = async_stream::try_stream! {
        let mut sequence = 0;
        let mut partial = String::new();
        let initial = options.view(&identity, &owner.model, "in_progress", &[], None, None);
        for event in initial_events(&identity, &initial) {
            yield wire.event(&owner, &mut sequence, event)?;
        }
        let deadline = time::Instant::now() + body::RESPONSE_TIMEOUT;
        let failure = loop {
            let now = time::Instant::now();
            if now >= deadline {
                break ServiceError::new(504, "timeout", "stream generation deadline reached");
            }
            let event = match time::timeout_at(deadline.min(now + body::KEEPALIVE), body::next(&owner)).await {
                Ok(Ok(event)) => event,
                Ok(Err(_)) => break ServiceError::new(503, "worker_unavailable", "worker disconnected before completion"),
                Err(_) if time::Instant::now() < deadline => {
                    // Still subject to the original socket's cumulative wire
                    // budget. Literal heartbeat retains the inference receipt.
                    yield Frame::data(Data::literal(": keep-alive\n\n", &owner));
                    continue;
                }
                Err(_) => continue,
            };
            match event {
                ResponseEvent::Text(text) => {
                    let frame = wire.event(&owner, &mut sequence, delta_event(&identity, &text))?;
                    // Successful bounded serialization establishes a bound on
                    // the accumulated decoded delta bytes as well.
                    partial.push_str(&text);
                    yield frame;
                }
                ResponseEvent::TokenText { .. } => break ServiceError::new(500, "unexpected_token_observation", "unrequested token observation"),
                ResponseEvent::Done { output, prompt_tokens, cached_prompt_tokens, completion_tokens, finish_reason } => {
                    if partial != output { break ServiceError::new(500, "invalid_stream", "text deltas differ from terminal output"); }
                    let status = match terminal_status(finish_reason) { Ok(status) => status, Err(error) => break error };
                    let usage = match Usage::new(prompt_tokens, cached_prompt_tokens, completion_tokens) { Ok(usage) => usage, Err(error) => break error };
                    let content = [TextPart::new(&output)];
                    let items = [Message::new(&identity, status, &content)];
                    let view = options.view(&identity, &owner.model, status, &items, Some(usage), None);
                    let [text, part, item, terminal] = done_events(&identity, &view, &output, status, &content);
                    for event in [text, part, item] {
                        yield wire.event(&owner, &mut sequence, event)?;
                    }
                    // Encode the final event before publishing the record.
                    let frame = wire.event(&owner, &mut sequence, terminal)?;
                    if let Some(pending) = pending.take() {
                        pending.finish(&view, &output, Some(status)).map_err(io::Error::other)?;
                    }
                    yield frame;
                    return;
                }
                ResponseEvent::Error(error) => break error,
            }
        };
        owner.cancel();
        // Observe/report the original typed error once, but use the canonical
        // Response failure object rather than a Chat error chunk.
        let failure_object = response_error(&errors, &failure, authorized);
        let part = [TextPart::new(&partial)];
        let items = [Message::new(&identity, "incomplete", &part)];
        let view = options.view(&identity, &owner.model, "failed", &items, None, Some(&failure_object));
        let frame = wire.event(&owner, &mut sequence, StreamEvent::Response("response.failed", &view))?;
        if let Some(pending) = pending.take() {
            // Do not persist opt-in arbitrary diagnostics with a resource.
            let mut safe_error = failure_object;
            safe_error.as_object_mut().expect("fixed error envelope").remove("details");
            let view = options.view(&identity, &owner.model, "failed", &items, None, Some(&safe_error));
            pending.finish(&view, &partial, None).map_err(io::Error::other)?;
        }
        yield frame;
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache, no-transform")
        .header("x-accel-buffering", "no")
        .body(Reply::Stream(Box::pin(stream)))
        .expect("fixed Responses SSE headers")
}
