//! Bounded HTTP/1.1 OpenAI chat/completions frontend.
//! Strictly local by default. Explicit plaintext HTTP only; use a trusted
//! TLS/auth reverse proxy before non-loopback exposure.

use crate::TextService;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value, json};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use zerogravity_engine::router::pending::ServingCharge;
use zerogravity_engine::router::token_metrics::json_allocation::{
    ValueConstruction, census_json, census_scratch_peak,
};
use zerogravity_engine::router::token_metrics::validate_wire_id;
use zerogravity_engine::router::{FailureDetails, SemanticRequest, ServiceError};

mod body;
mod completions;
mod embeddings;
use completions::{completion_response, parse_body};
mod ingress_body;
// Dormant until a real modality route consumes admitted forms.
#[allow(
    dead_code,
    reason = "reusable multipart transport primitive qualified by real socket tests"
)]
mod multipart;
mod responses;
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

/// Shared charged strict JSON ingress. Each protocol adapter validates its
/// own schema without reparsing or lowering through another wire format.
fn parse_json(
    body: &[u8],
    router: Option<&dyn TextService>,
    ingress: &mut Vec<ServingCharge>,
) -> Result<Value, ServiceError> {
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
    Ok(value)
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
    let message = if let Some(param) = e.parameter() {
        if e.code == "unsupported_parameter" {
            format!("Unsupported parameter: {param}.")
        } else {
            format!("{} Parameter: {param}.", metadata.public_message())
        }
    } else {
        metadata.public_message().to_owned()
    };
    let mut value = json!({"error":{
        "message":message,
        "type":if e.status >= 500 {"server_error"} else {"invalid_request_error"},
        "code":e.code,
        "param":e.parameter()
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
/// One checked accounting projection from the original terminal receipt.
struct TextUsage {
    prompt: usize,
    cached: usize,
    output: usize,
    total: usize,
}
impl TextUsage {
    fn new(prompt: usize, cached: usize, output: usize) -> Result<Self, ServiceError> {
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
            prompt,
            cached,
            output,
            total,
        })
    }
}
fn validate_finish_reason(reason: &str) -> Result<(), ServiceError> {
    match reason {
        "stop" | "length" => Ok(()),
        _ => Err(ServiceError::new(
            500,
            "invalid_finish_reason",
            "unexpected text finish reason",
        )),
    }
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
    use zerogravity_engine::router::{ChatMessage, ChatRole, TextInput};

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
            r#"{"model":"qwen","messages":[{"role":"assistant","content":"Hi","tool_calls":[{}]}]}"#,
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
        let plain_json = completion_response(
            1,
            "qwen",
            true,
            "A",
            TextUsage::new(1, 0, 1).unwrap(),
            "length",
        )
        .unwrap();
        assert!(plain_json.get("metrics").is_none());
        assert!(plain_json["choices"][0].get("token_ids").is_none());
    }
}
