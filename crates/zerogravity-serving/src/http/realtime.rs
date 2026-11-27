//! Bounded text-only Realtime sessions. A session holds protocol text, not
//! a prepared model or another execution session. Every turn uses the
//! original semantic admission and committed event receipt.
use super::{
    TextUsage,
    body::{BoundedWriter, RESPONSE_TIMEOUT},
    error_json, parse_json,
    transport::request_id,
    validate_finish_reason,
    websocket::{Admission, Socket},
};
use crate::TextService;
use serde_json::{Map, Value, json};
use std::{future::poll_fn, io, sync::Arc};
use tokio::time::{self, Instant};
use tokio_tungstenite::tungstenite::Message;
use zerogravity_engine::{
    router::{
        ChatMessage, ChatRole, RequestEvents, ResponseEvent, SemanticRequest, ServiceError,
        TextInput, pending::ServingCharge, validate_alias,
    },
    state_cache::PrefixSharingScope,
};

type Result<T> = std::result::Result<T, ServiceError>;
const HISTORY_BYTES: usize = 32 * 1024;
const MAX_ITEMS: usize = 64;
const MAX_TEXT: usize = 8 * 1024;
const MAX_EVENTS: usize = 256;
// Retained history, settings, bounded output, JSON projection and growth
// overlap. Ingress parsing and native request copies have separate credit.
const SESSION_BYTES: u64 = 512 * 1024;

fn invalid(message: &'static str) -> ServiceError {
    ServiceError::new(400, "invalid_realtime_event", message)
}
fn unsupported() -> ServiceError {
    ServiceError::new(
        400,
        "unsupported_parameter",
        "unsupported realtime parameter or event",
    )
}
fn limit() -> ServiceError {
    ServiceError::new(413, "realtime_limit", "realtime session bound exceeded")
}
fn object(value: &Value) -> Result<&Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| invalid("expected an object"))
}
fn keys(map: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    if map.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(unsupported());
    }
    Ok(())
}
fn id(prefix: &str) -> Result<String> {
    Ok(format!(
        "{prefix}{}",
        request_id()?.trim_start_matches("req_")
    ))
}
fn text_event(message: Message, service: &dyn TextService) -> Result<(Value, Vec<ServingCharge>)> {
    let Message::Text(text) = message else {
        return Err(unsupported());
    };
    let mut charges = Vec::with_capacity(3);
    let value = parse_json(text.as_bytes(), Some(service), &mut charges)?;
    let map = object(&value)?;
    if let Some(event) = map.get("event_id")
        && event.as_str().is_none_or(|s| s.is_empty() || s.len() > 128)
    {
        return Err(invalid("invalid event ID"));
    }
    Ok((value, charges))
}
async fn send(socket: &mut Socket, kind: &str, mut value: Value) -> io::Result<()> {
    let map = value.as_object_mut().expect("event object");
    map.insert("type".into(), json!(kind));
    map.insert(
        "event_id".into(),
        json!(id("event_").map_err(io::Error::other)?),
    );
    let mut buffer = BoundedWriter {
        bytes: Vec::new(),
        limit: socket.limit(),
    };
    serde_json::to_writer(&mut buffer, &value).map_err(io::Error::other)?;
    let text = String::from_utf8(buffer.bytes).map_err(io::Error::other)?;
    socket.send(Message::Text(text.into())).await
}
async fn error(
    socket: &mut Socket,
    e: &ServiceError,
    client_event: Option<&str>,
) -> io::Result<()> {
    let mut value = error_json(e, false);
    if let Some(id) = client_event {
        value["error"]["event_id"] = json!(id);
    }
    send(socket, "error", value).await
}
fn text_controls(map: &Map<String, Value>, maximum: usize, current: usize) -> Result<usize> {
    if map
        .get("output_modalities")
        .is_some_and(|v| v != &json!(["text"]))
        || map
            .get("tools")
            .is_some_and(|v| v.as_array().is_none_or(|a| !a.is_empty()))
        || map
            .get("tool_choice")
            .is_some_and(|v| v.as_str() != Some("none"))
    {
        return Err(unsupported());
    }
    match map.get("max_output_tokens") {
        None => Ok(current),
        Some(value) => value
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| (1..=maximum).contains(n))
            .ok_or_else(|| invalid("invalid max_output_tokens")),
    }
}
fn item(id: &str, role: ChatRole, text: &str, status: &str) -> Value {
    json!({"id":id,"object":"realtime.item","type":"message","role":role.as_str(),
        "status":status,"content":[{"type":if role == ChatRole::Assistant {"output_text"} else {"input_text"},"text":text}]})
}
async fn item_events(socket: &mut Socket, item: Value, previous: Option<String>) -> io::Result<()> {
    send(
        socket,
        "conversation.item.added",
        json!({"item":item,"previous_item_id":previous}),
    )
    .await?;
    send(
        socket,
        "conversation.item.done",
        json!({"item":item,"previous_item_id":previous}),
    )
    .await
}

pub(crate) fn admission(
    service: Arc<dyn TextService>,
    sharing: Option<PrefixSharingScope>,
    body_limit: usize,
) -> Admission {
    Arc::new(move |path| {
        let alias = path
            .strip_prefix("/v1/realtime?model=")
            .ok_or_else(|| invalid("exact model query required"))?;
        validate_alias(alias)?;
        let registration = service.realtime_text_registration(alias)?;
        let maximum = service.max_output()?;
        if maximum == 0 || body_limit < 4096 {
            return Err(limit());
        }
        let charge = service.charge_ingress("realtime-session", SESSION_BYTES)?;
        let session = Session {
            service: service.clone(),
            alias: alias.to_owned(),
            registration,
            sharing,
            conversation: id("conv_")?,
            history: Vec::with_capacity(MAX_ITEMS),
            history_bytes: 0,
            last_item: None,
            instructions: String::new(),
            maximum,
            max_tokens: maximum,
            max_text: 0,
            events: 0,
            _charge: charge,
        };
        Ok(Box::new(move |socket| Box::pin(session.run(socket))))
    })
}
struct Session {
    service: Arc<dyn TextService>,
    alias: String,
    registration: u64,
    sharing: Option<PrefixSharingScope>,
    conversation: String,
    history: Vec<ChatMessage>,
    history_bytes: usize,
    last_item: Option<String>,
    instructions: String,
    maximum: usize,
    max_tokens: usize,
    max_text: usize,
    events: usize,
    _charge: ServingCharge,
}
enum Action {
    Updated,
    Item(Value, Option<String>),
    Respond(usize),
}
impl Session {
    fn configuration(&self) -> Value {
        json!({"type":"realtime","model":self.alias,"output_modalities":["text"],
            "instructions":self.instructions,"max_output_tokens":self.max_tokens,
            "tools":[],"tool_choice":"none","truncation":"disabled"})
    }
    fn count_event(&mut self) -> io::Result<()> {
        self.events += 1;
        if self.events > MAX_EVENTS {
            return Err(io::Error::other("realtime event limit"));
        }
        Ok(())
    }
    fn append(&mut self, id: String, role: ChatRole, text: String) -> (Value, Option<String>) {
        let value = item(&id, role, &text, "completed");
        let previous = self.last_item.replace(id);
        self.history_bytes += text.len();
        self.history.push(ChatMessage { role, text });
        (value, previous)
    }
    fn handle(&mut self, event: &Value) -> Result<Action> {
        let map = object(event)?;
        match map.get("type").and_then(Value::as_str) {
            Some("session.update") => {
                keys(map, &["type", "event_id", "session"])?;
                let settings = object(
                    map.get("session")
                        .ok_or_else(|| invalid("session required"))?,
                )?;
                keys(
                    settings,
                    &[
                        "type",
                        "model",
                        "instructions",
                        "max_output_tokens",
                        "output_modalities",
                        "tools",
                        "tool_choice",
                        "truncation",
                    ],
                )?;
                if settings
                    .get("type")
                    .is_some_and(|v| v.as_str() != Some("realtime"))
                    || settings
                        .get("model")
                        .is_some_and(|v| v.as_str() != Some(&self.alias))
                    || settings
                        .get("truncation")
                        .is_some_and(|v| v.as_str() != Some("disabled"))
                {
                    return Err(unsupported());
                }
                let max = text_controls(settings, self.maximum, self.max_tokens)?;
                let instructions = settings
                    .get("instructions")
                    .map(|v| {
                        v.as_str()
                            .filter(|s| s.len() <= self.max_text)
                            .ok_or_else(limit)
                    })
                    .transpose()?;
                if let Some(text) = instructions {
                    self.instructions = text.to_owned();
                }
                self.max_tokens = max;
                Ok(Action::Updated)
            }
            Some("conversation.item.create") => {
                keys(map, &["type", "event_id", "item"])?;
                let message = object(map.get("item").ok_or_else(|| invalid("item required"))?)?;
                keys(message, &["type", "role", "content"])?;
                if message.get("type").and_then(Value::as_str) != Some("message")
                    || message.get("role").and_then(Value::as_str) != Some("user")
                {
                    return Err(unsupported());
                }
                let parts = message
                    .get("content")
                    .and_then(Value::as_array)
                    .filter(|p| p.len() == 1)
                    .ok_or_else(unsupported)?;
                let part = object(&parts[0])?;
                keys(part, &["type", "text"])?;
                if part.get("type").and_then(Value::as_str) != Some("input_text") {
                    return Err(unsupported());
                }
                let text = part
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| invalid("text required"))?;
                if text.len() > self.max_text
                    || self.history.len() >= MAX_ITEMS
                    || self.history_bytes + text.len() > HISTORY_BYTES
                {
                    return Err(limit());
                }
                let (item, previous) = self.append(id("item_")?, ChatRole::User, text.to_owned());
                Ok(Action::Item(item, previous))
            }
            Some("response.create") => {
                keys(map, &["type", "event_id", "response"])?;
                let max = if let Some(value) = map.get("response") {
                    let response = object(value)?;
                    keys(
                        response,
                        &[
                            "conversation",
                            "max_output_tokens",
                            "output_modalities",
                            "tools",
                            "tool_choice",
                        ],
                    )?;
                    if response
                        .get("conversation")
                        .is_some_and(|v| v.as_str() != Some("auto"))
                    {
                        return Err(unsupported());
                    }
                    text_controls(response, self.maximum, self.max_tokens)?
                } else {
                    self.max_tokens
                };
                if self.history.last().is_none_or(|m| m.role != ChatRole::User) {
                    return Err(invalid("a user message is required"));
                }
                if self.history.len() >= MAX_ITEMS
                    || self.history_bytes + self.max_text > HISTORY_BYTES
                {
                    return Err(limit());
                }
                Ok(Action::Respond(max))
            }
            Some("response.cancel") => Err(invalid("no active response")),
            _ => Err(unsupported()),
        }
    }
    fn submit(&self, max_tokens: usize) -> Result<RequestEvents> {
        // Separate credit for the native owned copy; the socket keeps its
        // own history charge even after the original worker is admitted.
        let bytes = 2 * (self.history_bytes + self.instructions.len()) + 8192;
        let charge = self.service.charge_ingress("realtime-turn", bytes as u64)?;
        let mut turns = Vec::with_capacity(self.history.len() + 1);
        if !self.instructions.is_empty() {
            turns.push(ChatMessage {
                role: ChatRole::System,
                text: self.instructions.clone(),
            });
        }
        turns.extend_from_slice(&self.history);
        self.service.request_realtime_text_admitted(
            SemanticRequest {
                model: self.alias.clone(),
                input: TextInput::Conversation(turns),
                max_tokens,
                stream: true,
                token_metrics: None,
            },
            self.registration,
            vec![charge],
            self.sharing,
        )
    }
    async fn run(mut self, mut socket: Socket) -> io::Result<()> {
        // Account worst-case JSON escaping and fixed event envelope.
        self.max_text = MAX_TEXT.min((socket.limit() - 2048) / 6);
        send(
            &mut socket,
            "session.created",
            json!({"session":self.configuration()}),
        )
        .await?;
        while let Some(message) = socket.receive().await? {
            self.count_event()?;
            let (event, charges) = match text_event(message, self.service.as_ref()) {
                Ok(parsed) => parsed,
                Err(e) => {
                    error(&mut socket, &e, None).await?;
                    continue;
                }
            };
            let client_id = event
                .get("event_id")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let action = self.handle(&event);
            drop((event, charges));
            match action {
                Ok(Action::Updated) => {
                    send(
                        &mut socket,
                        "session.updated",
                        json!({"session":self.configuration()}),
                    )
                    .await?
                }
                Ok(Action::Item(item, previous)) => {
                    item_events(&mut socket, item, previous).await?
                }
                Ok(Action::Respond(max)) => match self.submit(max) {
                    Ok(receipt) => {
                        time::timeout(RESPONSE_TIMEOUT, self.respond(&mut socket, receipt, max))
                            .await
                            .map_err(|_| io::ErrorKind::TimedOut)??;
                    }
                    Err(e) => error(&mut socket, &e, client_id.as_deref()).await?,
                },
                Err(e) => error(&mut socket, &e, client_id.as_deref()).await?,
            }
        }
        Ok(())
    }
    async fn respond(
        &mut self,
        socket: &mut Socket,
        receipt: RequestEvents,
        max: usize,
    ) -> io::Result<()> {
        let response_id = id("resp_").map_err(io::Error::other)?;
        let item_id = id("item_").map_err(io::Error::other)?;
        let conversation = self.conversation.clone();
        let response = |status: &str, output: Value, usage: Value, details: Value| {
            json!({
                "id":response_id,"object":"realtime.response","conversation_id":conversation,
                "status":status,"status_details":details,"output":output,
                "output_modalities":["text"],"max_output_tokens":max,"usage":usage
            })
        };
        send(
            socket,
            "response.created",
            json!({"response":response("in_progress", json!([]), Value::Null, Value::Null)}),
        )
        .await?;
        send(socket, "response.output_item.added", json!({"response_id":response_id,"output_index":0,"item":item(&item_id, ChatRole::Assistant, "", "in_progress")})).await?;
        send(socket, "response.content_part.added", json!({"response_id":response_id,"item_id":item_id,"output_index":0,"content_index":0,"part":{"type":"text","text":""}})).await?;
        let mut text = String::new();
        let deadline = Instant::now() + RESPONSE_TIMEOUT;
        let (status, usage, details) = loop {
            tokio::select! {
                biased;
                message = socket.receive_until(deadline) => {
                    let Some(message) = message? else { return Err(io::ErrorKind::ConnectionAborted.into()); };
                    self.count_event()?;
                    let (event, _charges) = match text_event(message, self.service.as_ref()) {
                        Ok(parsed) => parsed,
                        Err(e) => { error(socket, &e, None).await?; continue; }
                    };
                    let cancel = object(&event).and_then(|map| {
                        keys(map, &["type", "event_id", "response_id"])?;
                        if map.get("type").and_then(Value::as_str) != Some("response.cancel")
                            || map.get("response_id").is_some_and(|v| v.as_str() != Some(&response_id)) {
                            return Err(invalid("active response accepts only its own cancellation"));
                        }
                        Ok(())
                    });
                    match cancel {
                        Ok(()) => break ("cancelled", Value::Null, json!({"type":"cancelled","reason":"client_cancelled"})),
                        Err(e) => error(socket, &e, event.get("event_id").and_then(Value::as_str)).await?,
                    }
                }
                event = poll_fn(|cx| receipt.events.poll_recv(cx)) => {
                    match event {
                        Ok(ResponseEvent::Text(delta) | ResponseEvent::TokenText {text:delta,..}) => {
                            if delta.len() > self.max_text - text.len() {
                                break failed("output_too_large");
                            }
                            text.push_str(&delta);
                            send(socket, "response.output_text.delta", json!({"response_id":response_id,"item_id":item_id,"output_index":0,"content_index":0,"delta":delta})).await?;
                        }
                        Ok(ResponseEvent::Done {output, prompt_tokens, cached_prompt_tokens, completion_tokens, finish_reason}) => {
                            if output != text || completion_tokens > max || validate_finish_reason(finish_reason).is_err() {
                                break failed("invalid_terminal");
                            }
                            let Ok(usage) = TextUsage::new(prompt_tokens, cached_prompt_tokens, completion_tokens) else {
                                break failed("invalid_usage");
                            };
                            let usage = json!({"input_tokens":usage.prompt,"output_tokens":usage.output,"total_tokens":usage.total,
                                "input_token_details":{"cached_tokens":usage.cached,"text_tokens":usage.prompt},
                                "output_token_details":{"text_tokens":usage.output}});
                            if finish_reason == "length" {
                                break ("incomplete", usage, json!({"type":"incomplete","reason":"max_output_tokens"}));
                            }
                            break ("completed", usage, Value::Null);
                        }
                        Ok(ResponseEvent::Error(e)) => break failure(e.code, if e.status >= 500 { "server_error" } else { "invalid_request_error" }),
                        Err(_) => break failed("inference_error"),
                    }
                }
            }
        };
        drop(receipt); // Cancel immediately, before any terminal socket write.
        let complete = status == "completed";
        let output = item(
            &item_id,
            ChatRole::Assistant,
            &text,
            if complete { "completed" } else { "incomplete" },
        );
        send(socket, "response.output_text.done", json!({"response_id":response_id,"item_id":item_id,"output_index":0,"content_index":0,"text":text})).await?;
        send(socket, "response.content_part.done", json!({"response_id":response_id,"item_id":item_id,"output_index":0,"content_index":0,"part":{"type":"text","text":text}})).await?;
        send(
            socket,
            "response.output_item.done",
            json!({"response_id":response_id,"output_index":0,"item":output}),
        )
        .await?;
        if complete {
            let (item, previous) = self.append(item_id, ChatRole::Assistant, text);
            item_events(socket, item, previous).await?;
        }
        send(
            socket,
            "response.done",
            json!({"response":response(status, json!([output]), usage, details)}),
        )
        .await?;
        Ok(())
    }
}
fn failed(code: &'static str) -> (&'static str, Value, Value) {
    failure(code, "server_error")
}
fn failure(code: &'static str, kind: &'static str) -> (&'static str, Value, Value) {
    (
        "failed",
        Value::Null,
        json!({"type":"failed","error":{"type":kind,"code":code}}),
    )
}
