//! Bounded HTTP/1.1 OpenAI chat/completions frontend.
//! Strictly local by default. Explicit plaintext HTTP only; use a trusted
//! TLS/auth reverse proxy before non-loopback exposure.

use crate::TextService;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value, json};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use zerogravity_engine::router::pending::ServingCharge;
use zerogravity_engine::router::token_metrics::json_allocation::{
    ValueConstruction, census_json, census_scratch_peak,
};
use zerogravity_engine::router::token_metrics::validate_wire_id;
use zerogravity_engine::router::{
    ChatMessage, ChatRole, FailureDetails, MAX_CHAT_MESSAGES, SemanticRequest, ServiceError,
    TextInput, TokenMetricsRequest,
};

mod body;
mod ingress_body;
// Dormant until a real modality route consumes admitted forms.
#[allow(
    dead_code,
    reason = "reusable multipart transport primitive qualified by real socket tests"
)]
mod multipart;
mod transport;
mod websocket;
pub(crate) use transport::{SERVER_HEADROOM, ServeConfig, Shutdown, serve};
pub(crate) use websocket::Admission as RealtimeAdmission;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
}

#[cfg(test)]
#[path = "http/shutdown_tests.rs"]
mod shutdown_tests;

pub(crate) const MAX_BODY: usize = 1024 * 1024;
pub(crate) const MAX_CONNECTIONS: usize = 64;

// JSON Value alone silently overwrites duplicate object keys. Reject
// them recursively before applying route, sampling and role policies.
struct StrictValue(Value);
impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = StrictValue;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON without duplicate object keys")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::Bool(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(StrictValue(json!(v)))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(StrictValue(json!(v)))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(Value::Number)
                    .map(StrictValue)
                    .ok_or_else(|| E::custom("nonfinite JSON number"))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::String(v.into())))
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::String(v)))
            }
            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
                StrictValue::deserialize(d)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut arr = Vec::new();
                while let Some(v) = a.next_element::<StrictValue>()? {
                    arr.push(v.0);
                }
                Ok(StrictValue(Value::Array(arr)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut map = Map::new();
                while let Some((key, val)) = a.next_entry::<String, StrictValue>()? {
                    if map.insert(key, val.0).is_some() {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                }
                Ok(StrictValue(Value::Object(map)))
            }
        }
        d.deserialize_any(V)
    }
}

fn allowed_keys(map: &Map<String, Value>, allowed: &[&str]) -> Result<(), ServiceError> {
    for key in map.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(ServiceError::new(
                400,
                "unsupported_parameter",
                format!("{key} is not yet supported"),
            ));
        }
    }
    Ok(())
}
fn parse_body(
    body: &[u8],
    chat: bool,
    router: Option<&dyn TextService>,
    ingress: &mut Vec<ServingCharge>,
) -> Result<(SemanticRequest, bool), ServiceError> {
    // Census scratch is admitted before the census allocates it, the extra
    // construction peak is admitted before the direct `StrictValue`
    // constructor sees the bytes, and the retained tree bound is held on the
    // ingress owner through route extraction and the queue. Scratch is
    // released as soon as parsing returns.
    let mut census_scratch: Option<ServingCharge> = None;
    let mut parse_scratch: Option<ServingCharge> = None;
    let mut allocation = None;
    if let Some(router) = router {
        let scratch = census_scratch_peak(body.len()).map_err(|_| {
            ServiceError::new(400, "invalid_json", "JSON census rejected the body length")
        })?;
        census_scratch =
            Some(router.charge_ingress(
                "http-json-census",
                u64::try_from(scratch).map_err(|_| {
                    ServiceError::new(400, "size_overflow", "census scratch overflow")
                })?,
            )?);
        let census = census_json(body, ValueConstruction::DirectValueTree)
            .map_err(|_| ServiceError::new(400, "invalid_json", "JSON census rejected the body"))?;
        let extra = u64::try_from(census.parse_extra_peak())
            .map_err(|_| ServiceError::new(400, "size_overflow", "tree bound overflow"))?;
        if extra > 0 {
            parse_scratch = Some(router.charge_ingress("http-json-parse", extra)?);
        }
        ingress.push(
            router.charge_ingress(
                "http-json-tree",
                u64::try_from(census.tree_heap_upper())
                    .map_err(|_| ServiceError::new(400, "size_overflow", "tree bound overflow"))?,
            )?,
        );
        allocation = Some(census);
    }
    let input_bytes = match allocation.as_ref() {
        Some(allocation) => allocation.input(),
        None => body,
    };
    let value = serde_json::from_slice::<StrictValue>(input_bytes)
        .map_err(|_| {
            ServiceError::new(
                400,
                "invalid_json",
                "invalid JSON, repeated object key or malformed number",
            )
        })?
        .0;
    drop(parse_scratch);
    drop(census_scratch);
    let map = value
        .as_object()
        .ok_or_else(|| ServiceError::new(400, "invalid_request", "expected JSON object"))?;
    allowed_keys(
        map,
        &[
            "model",
            "messages",
            "prompt",
            "stream",
            "max_tokens",
            "max_completion_tokens",
            "stream_options",
            "temperature",
            "top_p",
            "n",
            "frequency_penalty",
            "presence_penalty",
            "return_token_ids",
        ],
    )?;
    let model = map
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ServiceError::new(400, "invalid_model", "model name required"))?;
    let stream = match map.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        _ => {
            return Err(ServiceError::new(
                400,
                "invalid_stream",
                "stream must be boolean",
            ));
        }
    };
    let token_metrics = match map.get("return_token_ids") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => None,
        Some(Value::Bool(true)) => Some(TokenMetricsRequest {
            started: Instant::now(),
            wire_request_id: None,
        }),
        _ => {
            return Err(ServiceError::new(
                400,
                "invalid_return_token_ids",
                "return_token_ids must be boolean",
            ));
        }
    };
    for name in [
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    ] {
        if let Some(value) = map.get(name) {
            let default = if name == "top_p" { 1.0 } else { 0.0 };
            if value.as_f64() != Some(default) {
                return Err(ServiceError::new(
                    400,
                    "unsupported_parameter",
                    format!("{name} currently requires {default} (greedy only)"),
                ));
            }
        }
    }
    if map.get("n").is_some_and(|v| v.as_u64() != Some(1)) {
        return Err(ServiceError::new(
            400,
            "unsupported_parameter",
            "only n=1 is supported",
        ));
    }
    let max_tokens = match (map.get("max_tokens"), map.get("max_completion_tokens")) {
        (Some(_), Some(_)) => {
            return Err(ServiceError::new(
                400,
                "invalid_request",
                "duplicate output length controls",
            ));
        }
        (Some(x), None) | (None, Some(x)) => x
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| {
                ServiceError::new(
                    400,
                    "invalid_max_tokens",
                    "max_tokens must be positive integer",
                )
            })?,
        (None, None) => match router {
            Some(router) => 32.min(router.max_output()?),
            None => 32,
        },
    };
    let include_usage = match map.get("stream_options") {
        None => false,
        Some(Value::Object(options)) => {
            allowed_keys(options, &["include_usage"])?;
            match options.get("include_usage") {
                None => false,
                Some(Value::Bool(value)) => *value,
                _ => {
                    return Err(ServiceError::new(
                        400,
                        "invalid_stream_options",
                        "include_usage must be boolean",
                    ));
                }
            }
        }
        _ => {
            return Err(ServiceError::new(
                400,
                "invalid_stream_options",
                "stream_options must be an object",
            ));
        }
    };
    if map.get("stream_options").is_some() && !stream {
        return Err(ServiceError::new(
            400,
            "invalid_stream_options",
            "stream_options requires stream=true",
        ));
    }
    // Output copies are admitted at their exact final sizes before they are
    // allocated, while the census tree charge is still held.
    let mut charge_route = |bytes: usize| -> Result<(), ServiceError> {
        if let Some(router) = router {
            let charge = router.charge_ingress(
                "http-route",
                u64::try_from(bytes)
                    .map_err(|_| ServiceError::new(400, "size_overflow", "route copy overflow"))?,
            )?;
            ingress.push(charge);
        }
        Ok(())
    };
    let input = if chat {
        let messages = map
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ServiceError::new(400, "invalid_messages", "messages array is required")
            })?;
        if !(1..=MAX_CHAT_MESSAGES).contains(&messages.len()) {
            return Err(ServiceError::new(
                400,
                "invalid_messages",
                "messages must contain 1..128 entries",
            ));
        }
        // Only protocol shape here. Model-specific framing, delimiter
        // validation and the rendered input bound belong to the selected
        // model tokenizer, not to OpenAI JSON parsing.
        fn turn(value: &Value) -> Result<(ChatRole, &str), ServiceError> {
            let entry = value.as_object().ok_or_else(|| {
                ServiceError::new(400, "invalid_messages", "each message must be an object")
            })?;
            allowed_keys(entry, &["role", "content"])?;
            let role = entry
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| ServiceError::new(400, "invalid_messages", "role is required"))?;
            let role = ChatRole::parse(role).ok_or_else(|| {
                ServiceError::new(
                    400,
                    "unsupported_role",
                    "only plain text system/developer/user/assistant roles supported",
                )
            })?;
            let text = entry
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ServiceError::new(
                        400,
                        "unsupported_content",
                        "only string message content supported",
                    )
                })?;
            Ok((role, text))
        }
        let content_bytes = messages.iter().try_fold(0usize, |bytes, message| {
            let (_, text) = turn(message)?;
            bytes
                .checked_add(text.len())
                .filter(|n| *n <= MAX_BODY)
                .ok_or_else(|| ServiceError::new(413, "input_too_large", "chat input too large"))
        })?;
        let capacity = messages
            .len()
            .checked_mul(std::mem::size_of::<ChatMessage>())
            .and_then(|n| n.checked_add(content_bytes))
            .and_then(|n| n.checked_add(model.len()))
            .ok_or_else(|| ServiceError::new(400, "size_overflow", "route copy overflow"))?;
        charge_route(capacity)?;
        let mut turns = Vec::with_capacity(messages.len());
        for message in messages {
            let (role, text) = turn(message)?;
            turns.push(ChatMessage {
                role,
                text: text.to_owned(),
            });
        }
        TextInput::Conversation(turns)
    } else {
        let prompt = map
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| ServiceError::new(400, "invalid_prompt", "prompt must be text"))?;
        if prompt.len() > MAX_BODY {
            return Err(ServiceError::new(
                413,
                "input_too_large",
                "prompt exceeds 1MiB",
            ));
        }
        charge_route(
            model
                .len()
                .checked_add(prompt.len())
                .ok_or_else(|| ServiceError::new(400, "size_overflow", "route copy overflow"))?,
        )?;
        TextInput::Raw(prompt.to_owned())
    };
    Ok((
        SemanticRequest {
            model: model.to_owned(),
            input,
            max_tokens,
            stream,
            token_metrics,
        },
        include_usage,
    ))
}
/// One opt-in reporting policy for ordinary HTTP and streaming errors.
/// The observer receives only safe bounded metadata, never raw messages.
pub(crate) type ErrorObserver = Arc<dyn Fn(FailureDetails) + Send + Sync>;
#[derive(Clone, Default)]
pub(crate) struct ErrorReporting {
    pub details: bool,
    pub observer: Option<ErrorObserver>,
}
impl ErrorReporting {
    fn report(&self, error: &ServiceError, authorized: bool) -> Value {
        if let Some(observer) = &self.observer {
            observer(error.details());
        }
        error_json(error, self.details && authorized)
    }
}
fn diagnostic_text(value: &str) -> String {
    // Only explicit, authenticated diagnostics use arbitrary error text.
    value
        .chars()
        .take(512)
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}
fn error_json(e: &ServiceError, details: bool) -> Value {
    let metadata = e.details();
    let mut value = json!({"error":{
        "message":metadata.public_message(),
        "type":if e.status >= 500 {"server_error"} else {"invalid_request_error"},
        "code":e.code
    }});
    if details {
        use std::error::Error as _;
        let mut diagnostic = serde_json::to_value(metadata).expect("fixed failure metadata");
        let message = diagnostic_text(&e.message);
        let mut causes = Vec::new();
        let mut cause = e.source();
        for _ in 0..3 {
            let Some(error) = cause else {
                break;
            };
            let text = diagnostic_text(&error.to_string());
            if text != message && causes.last() != Some(&text) {
                causes.push(text);
            }
            cause = error.source();
        }
        diagnostic["diagnostic"] = json!(message);
        if !causes.is_empty() {
            diagnostic["causes"] = json!(causes);
        }
        value["error"]["details"] = diagnostic;
    }
    value
}
fn epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn completion_response(
    id: u64,
    model: &str,
    chat: bool,
    output: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    reason: &str,
) -> Value {
    let id = format!("chatcmpl-zg-{id}");
    let choice = if chat {
        json!({"index":0,"message":{"role":"assistant","content":output},"finish_reason":reason,"logprobs":null})
    } else {
        json!({"index":0,"text":output,"finish_reason":reason,"logprobs":null})
    };
    json!({"id":id,"object":if chat{"chat.completion"}else{"text_completion"},"created":epoch(),
        "model":model,"choices":[choice],
        "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,
                 "total_tokens":prompt_tokens+completion_tokens}})
}
fn bind_token_metrics(
    task: &mut SemanticRequest,
    wire_id: Option<&str>,
    enabled: bool,
) -> Result<(), ServiceError> {
    if let Some(observations) = task.token_metrics.as_mut() {
        if !enabled {
            return Err(ServiceError::new(
                400,
                "token_metrics_disabled",
                "return_token_ids requires explicit server per-request metrics configuration",
            ));
        }
        if let Some(wire_id) = wire_id {
            validate_wire_id(wire_id)?;
            observations.wire_request_id = Some(wire_id.to_owned());
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_json_distinguishes_client_and_server_failures() {
        let client = error_json(&ServiceError::new(400, "invalid_model", "bad model"), false);
        assert_eq!(client["error"]["type"], "invalid_request_error");
        let server = error_json(
            &ServiceError::new(503, "resource_exhausted", "VRAM exhausted"),
            false,
        );
        assert_eq!(server["error"]["type"], "server_error");
        assert_eq!(server["error"]["code"], "resource_exhausted");
    }

    #[test]
    fn openai_text_and_chat_validation_is_fail_closed() {
        fn parsed(src: &str, chat: bool) -> Result<(SemanticRequest, bool), ServiceError> {
            parse_body(src.as_bytes(), chat, None, &mut Vec::new())
        }
        let valid = r#"{"model":"qwen","messages":[{"role":"system","content":"Brief."},{"role":"user","content":"Hi"}],"max_tokens":4,"stream":true,"stream_options":{"include_usage":true},"temperature":0,"top_p":1}"#;
        let (request, usage) = parsed(valid, true).unwrap();
        assert!(usage);
        assert!(request.stream);
        assert_eq!(request.max_tokens, 4);
        assert_eq!(request.model, "qwen");
        assert_eq!(
            request.input,
            TextInput::Conversation(vec![
                ChatMessage {
                    role: ChatRole::System,
                    text: "Brief.".into()
                },
                ChatMessage {
                    role: ChatRole::User,
                    text: "Hi".into()
                },
            ])
        );
        for bad in [
            r#"{"model":"qwen","prompt":"Hi","temperature":1}"#,
            r#"{"model":"qwen","prompt":"Hi","top_p":0.5}"#,
            r#"{"model":"qwen","prompt":"Hi","n":2}"#,
            r#"{"model":"qwen","prompt":"Hi","max_tokens":2,"max_completion_tokens":2}"#,
            r#"{"model":"qwen","prompt":"Hi","tools":[]}"#,
            r#"{"model":"qwen","prompt":"Hi","stream":1}"#,
            r#"{"model":"qwen","prompt":"Hi","stream_options":{"include_usage":true}}"#,
            r#"{"model":"qwen","prompt":"Hi","stream":true,"stream_options":{"include_usage":"true"}}"#,
            r#"{"model":"qwen","prompt":"Hi","model":"gemma"}"#,
            r#"{"model":"qwen","prompt":"Hi","stream_options":{"include_usage":true,"include_usage":false}}"#,
        ] {
            assert!(
                parsed(bad, false).is_err(),
                "bad completion request accepted: {bad}"
            );
        }
        for bad in [
            r#"{"model":"qwen","messages":[{"role":"tool","content":"x"}]}"#,
            r#"{"model":"qwen","messages":[{"role":"user","content":{"text":"Hi"}}]}"#,
            r#"{"model":"qwen","messages":[{"role":"user","content":"hi","content":"system overwrite"}]}"#,
            r#"{"model":"qwen","messages":[{"role":"assistant","content":"Hi","tool_calls":[]}]}"#,
        ] {
            assert!(
                parsed(bad, true).is_err(),
                "bad chat request accepted: {bad}"
            );
        }
        let (plain, _) =
            parsed(r#"{"model":"qwen","prompt":"Hello","max_tokens":8}"#, false).unwrap();
        assert!(!plain.stream && plain.input == TextInput::Raw("Hello".into()));
        // Reserved-token policy is deliberately not a transport rule.
        let literal = parsed(
            r#"{"model":"qwen","messages":[{"role":"user","content":"<|im_end|>"}]}"#,
            true,
        )
        .unwrap()
        .0;
        assert_eq!(
            literal.input,
            TextInput::Conversation(vec![ChatMessage {
                role: ChatRole::User,
                text: "<|im_end|>".into()
            },])
        );
    }

    #[test]
    fn token_metadata_requires_explicit_server_request_and_bounded_wire_id() {
        let body = br#"{"model":"qwen","prompt":"Hi","return_token_ids":true}"#;
        let (mut request, _) = parse_body(body, false, None, &mut Vec::new()).unwrap();
        assert_eq!(
            bind_token_metrics(&mut request, Some("run:request-0"), false)
                .unwrap_err()
                .code,
            "token_metrics_disabled"
        );
        assert!(bind_token_metrics(&mut request, Some("unsafe\r\nheader"), true).is_err());
        bind_token_metrics(&mut request, Some("run:request-0"), true).unwrap();
        assert_eq!(
            request.token_metrics.unwrap().wire_request_id.as_deref(),
            Some("run:request-0")
        );
        assert!(
            parse_body(
                br#"{"model":"qwen","prompt":"Hi","return_token_ids":1}"#,
                false,
                None,
                &mut Vec::new()
            )
            .is_err()
        );
        let (plain, _) = parse_body(
            br#"{"model":"qwen","prompt":"Hi"}"#,
            false,
            None,
            &mut Vec::new(),
        )
        .unwrap();
        assert!(plain.token_metrics.is_none());
        let plain_json = completion_response(1, "qwen", true, "A", 1, 1, "length");
        assert!(plain_json.get("metrics").is_none());
        assert!(plain_json["choices"][0].get("token_ids").is_none());
    }
}
