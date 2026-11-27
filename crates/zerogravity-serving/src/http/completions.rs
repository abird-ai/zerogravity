//! Chat and legacy Completions lower directly into the shared text request.
//! Only the supported greedy, single-choice text semantics are admitted.
use super::{MAX_BODY, TextUsage, epoch, parse_json, validate_finish_reason};
use crate::TextService;
use serde_json::{Map, Value, json};
use std::time::Instant;
use zerogravity_engine::router::{
    ChatMessage, ChatRole, MAX_CHAT_MESSAGES, SemanticRequest, ServiceError, TextInput,
    TokenMetricsRequest, pending::ServingCharge, validate_alias,
};

type Result<T> = std::result::Result<T, ServiceError>;

fn optional<'a>(map: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    map.get(key).filter(|v| !v.is_null())
}
fn invalid(code: &'static str, parameter: &'static str, message: &'static str) -> ServiceError {
    ServiceError::new(400, code, message).with_parameter(parameter)
}
fn unsupported(parameter: &'static str) -> ServiceError {
    invalid(
        "unsupported_parameter",
        parameter,
        "unsupported completion capability",
    )
}
fn neutral(
    map: &Map<String, Value>,
    parameter: &'static str,
    accepts: impl FnOnce(&Value) -> bool,
) -> Result<()> {
    if optional(map, parameter).is_some_and(|v| !accepts(v)) {
        return Err(unsupported(parameter));
    }
    Ok(())
}
fn empty_array(v: &Value) -> bool {
    v.as_array().is_some_and(Vec::is_empty)
}
fn empty_object(v: &Value) -> bool {
    v.as_object().is_some_and(Map::is_empty)
}
fn boolean(map: &Map<String, Value>, name: &'static str, code: &'static str) -> Result<bool> {
    optional(map, name)
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| invalid(code, name, "expected boolean"))
        })
        .transpose()
        .map(|v| v.unwrap_or(false))
}

const COMMON: &[&str] = &[
    "model",
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
    "stop",
    "logit_bias",
    "logprobs",
    "user",
    "seed",
];
const CHAT: &[&str] = &[
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "modalities",
    "store",
    "top_logprobs",
    "service_tier",
    "metadata",
    "audio",
    "functions",
    "function_call",
    "reasoning_effort",
    "prediction",
    "web_search_options",
    "verbosity",
    "prompt_cache_key",
    "prompt_cache_retention",
    "prompt_cache_options",
    "safety_identifier",
];
const LEGACY: &[&str] = &["prompt", "best_of", "echo", "suffix"];

/// Validate each route's controls independently. Null optional values are
/// omission; only exact no-op values of unimplemented features are accepted.
fn controls(map: &Map<String, Value>, chat: bool) -> Result<()> {
    for key in map.keys() {
        if COMMON.contains(&key.as_str())
            || (chat && CHAT.contains(&key.as_str()))
            || (!chat && LEGACY.contains(&key.as_str()))
        {
            continue;
        }
        if let Some(&name) = CHAT.iter().chain(LEGACY).find(|&&name| name == key) {
            return Err(unsupported(name));
        }
        return Err(ServiceError::new(
            400,
            "unsupported_parameter",
            "unknown completion parameter",
        ));
    }
    for name in [
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    ] {
        let default = if name == "top_p" { 1.0 } else { 0.0 };
        neutral(map, name, |v| v.as_f64() == Some(default))?;
    }
    neutral(map, "n", |v| v.as_u64() == Some(1))?;
    neutral(map, "logit_bias", empty_object)?;
    for name in ["stop", "user", "seed"] {
        neutral(map, name, |_| false)?;
    }
    if chat {
        neutral(map, "logprobs", |v| v.as_bool() == Some(false))?;
        neutral(map, "top_logprobs", |v| v.as_u64() == Some(0))?;
        neutral(map, "tools", empty_array)?;
        neutral(map, "tool_choice", |v| {
            matches!(v.as_str(), Some("auto" | "none"))
        })?;
        neutral(map, "parallel_tool_calls", Value::is_boolean)?;
        neutral(map, "response_format", |v| {
            v.as_object().is_some_and(|o| {
                o.len() == 1 && o.get("type").and_then(Value::as_str) == Some("text")
            })
        })?;
        neutral(map, "modalities", |v| {
            v.as_array()
                .is_some_and(|a| a.len() == 1 && a[0].as_str() == Some("text"))
        })?;
        neutral(map, "store", |v| v.as_bool() == Some(false))?;
        neutral(map, "metadata", empty_object)?;
        neutral(map, "service_tier", |v| {
            matches!(v.as_str(), Some("auto" | "default"))
        })?;
        for name in [
            "audio",
            "functions",
            "function_call",
            "reasoning_effort",
            "prediction",
            "web_search_options",
            "verbosity",
            "prompt_cache_key",
            "prompt_cache_retention",
            "prompt_cache_options",
            "safety_identifier",
        ] {
            neutral(map, name, |_| false)?;
        }
    } else {
        neutral(map, "best_of", |v| v.as_u64() == Some(1))?;
        neutral(map, "echo", |v| v.as_bool() == Some(false))?;
        // Legacy logprobs=0 still asks for the sampled token's logprob.
        for name in ["logprobs", "suffix"] {
            neutral(map, name, |_| false)?;
        }
    }
    Ok(())
}

/// Borrow the exact text fragments. Model-specific template rendering and
/// delimiter/role validation remain with the selected tokenizer.
fn fragments(content: &Value, mut visit: impl FnMut(&str) -> Result<()>) -> Result<()> {
    if let Some(text) = content.as_str() {
        return visit(text);
    }
    let invalid_content = || {
        invalid(
            "unsupported_content",
            "messages",
            "expected string or typed text parts",
        )
    };
    let parts = content
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(invalid_content)?;
    for part in parts {
        let part = part.as_object().ok_or_else(invalid_content)?;
        if part.len() != 2 || part.get("type").and_then(Value::as_str) != Some("text") {
            return Err(invalid_content());
        }
        visit(
            part.get("text")
                .and_then(Value::as_str)
                .ok_or_else(invalid_content)?,
        )?;
    }
    Ok(())
}
fn turn(value: &Value) -> Result<(ChatRole, &Value)> {
    let invalid_message = || invalid("invalid_messages", "messages", "expected role and content");
    let entry = value.as_object().ok_or_else(invalid_message)?;
    let role = entry
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(invalid_message)?;
    let role = ChatRole::parse(role).ok_or_else(|| {
        invalid(
            "unsupported_role",
            "messages",
            "only text system/developer/user/assistant roles supported",
        )
    })?;
    for (key, value) in entry {
        match key.as_str() {
            "role" | "content" => {}
            // SDK replay of an ordinary assistant message may contain
            // absent optional tool/audio/refusal fields.
            "audio" | "function_call" | "refusal"
                if role == ChatRole::Assistant && value.is_null() => {}
            "tool_calls"
                if role == ChatRole::Assistant && (value.is_null() || empty_array(value)) => {}
            _ => return Err(unsupported("messages")),
        }
    }
    Ok((role, entry.get("content").ok_or_else(invalid_message)?))
}

fn content_size(content: &Value) -> Result<usize> {
    let mut bytes = 0usize;
    fragments(content, |text| {
        bytes = bytes
            .checked_add(text.len())
            .filter(|n| *n <= MAX_BODY)
            .ok_or_else(|| {
                ServiceError::new(413, "input_too_large", "chat input too large")
                    .with_parameter("messages")
            })?;
        Ok(())
    })?;
    Ok(bytes)
}

pub(super) fn parse_body(
    body: &[u8],
    chat: bool,
    service: Option<&dyn TextService>,
    ingress: &mut Vec<ServingCharge>,
) -> Result<(SemanticRequest, bool)> {
    let value = parse_json(body, service, ingress)?;
    let map = value
        .as_object()
        .ok_or_else(|| ServiceError::new(400, "invalid_request", "expected JSON object"))?;
    controls(map, chat)?;
    let model = map
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("invalid_model", "model", "model name required"))?;
    validate_alias(model).map_err(|e| e.with_parameter("model"))?;
    let stream = boolean(map, "stream", "invalid_stream")?;
    let token_metrics = boolean(map, "return_token_ids", "invalid_return_token_ids")?.then(|| {
        TokenMetricsRequest {
            started: Instant::now(),
            wire_request_id: None,
        }
    });
    let max = service.map_or(Ok(zerogravity_engine::router::MAX_OUTPUT_TOKENS), |s| {
        s.max_output()
    })?;
    let length = match (
        optional(map, "max_tokens"),
        optional(map, "max_completion_tokens"),
    ) {
        (Some(_), Some(_)) => {
            return Err(invalid(
                "invalid_request",
                "max_completion_tokens",
                "duplicate output length controls",
            ));
        }
        (Some(value), None) => Some(("max_tokens", value)),
        (None, Some(value)) => Some(("max_completion_tokens", value)),
        (None, None) => None,
    };
    let max_tokens = if let Some((parameter, value)) = length {
        value
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| (1..=max).contains(n))
            .ok_or_else(|| {
                invalid(
                    "invalid_max_tokens",
                    parameter,
                    "output limit outside configured text capacity",
                )
            })?
    } else {
        if max == 0 {
            return Err(ServiceError::new(
                503,
                "worker_unavailable",
                "no configured text output capacity",
            ));
        }
        max.min(32)
    };
    let include_usage = if let Some(options) = optional(map, "stream_options") {
        let invalid_options = || {
            invalid(
                "invalid_stream_options",
                "stream_options",
                "stream_options requires a streaming options object",
            )
        };
        let options = options
            .as_object()
            .filter(|_| stream)
            .ok_or_else(invalid_options)?;
        if options
            .keys()
            .any(|key| !matches!(key.as_str(), "include_usage" | "include_obfuscation"))
        {
            return Err(unsupported("stream_options"));
        }
        if optional(options, "include_obfuscation").is_some_and(|v| v.as_bool() != Some(false)) {
            return Err(unsupported("stream_options"));
        }
        optional(options, "include_usage")
            .map(|v| v.as_bool().ok_or_else(invalid_options))
            .transpose()?
            .unwrap_or(false)
    } else {
        false
    };
    // Measure and admit the final typed copies before allocating, while the
    // original strict JSON tree remains charged.
    let overflow = || ServiceError::new(400, "size_overflow", "route copy overflow");
    let mut charge = |bytes: usize| -> Result<()> {
        if let Some(service) = service {
            ingress.push(
                service
                    .charge_ingress("http-route", u64::try_from(bytes).map_err(|_| overflow())?)?,
            );
        }
        Ok(())
    };
    let input = if chat {
        let messages = map
            .get("messages")
            .and_then(Value::as_array)
            .filter(|m| (1..=MAX_CHAT_MESSAGES).contains(&m.len()))
            .ok_or_else(|| {
                invalid(
                    "invalid_messages",
                    "messages",
                    "messages must contain 1..128 entries",
                )
            })?;
        let mut sizes = [0usize; MAX_CHAT_MESSAGES];
        let mut bytes = 0usize;
        for (index, message) in messages.iter().enumerate() {
            let (_, content) = turn(message)?;
            sizes[index] = content_size(content)?;
            bytes = bytes
                .checked_add(sizes[index])
                .filter(|n| *n <= MAX_BODY)
                .ok_or_else(|| {
                    ServiceError::new(413, "input_too_large", "chat input too large")
                        .with_parameter("messages")
                })?;
        }
        charge(
            messages
                .len()
                .checked_mul(std::mem::size_of::<ChatMessage>())
                .and_then(|n| n.checked_add(bytes))
                .and_then(|n| n.checked_add(model.len()))
                .ok_or_else(overflow)?,
        )?;
        let mut turns = Vec::with_capacity(messages.len());
        for (message, size) in messages.iter().zip(sizes) {
            let (role, content) = turn(message)?;
            let mut text = String::with_capacity(size);
            fragments(content, |part| {
                text.push_str(part);
                Ok(())
            })?;
            turns.push(ChatMessage { role, text });
        }
        TextInput::Conversation(turns)
    } else {
        let prompt = map
            .get("prompt")
            .and_then(Value::as_str)
            .filter(|p| !p.is_empty())
            .ok_or_else(|| invalid("invalid_prompt", "prompt", "nonempty text prompt required"))?;
        if prompt.len() > MAX_BODY {
            return Err(
                ServiceError::new(413, "input_too_large", "prompt exceeds 1MiB")
                    .with_parameter("prompt"),
            );
        }
        charge(model.len().checked_add(prompt.len()).ok_or_else(overflow)?)?;
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

impl TextUsage {
    pub(super) fn completion_json(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt,
            "prompt_tokens_details": {"cached_tokens": self.cached},
            "completion_tokens": self.output,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": self.total
        })
    }
}

pub(super) fn completion_response(
    id: u64,
    model: &str,
    chat: bool,
    output: &str,
    usage: TextUsage,
    reason: &str,
) -> Result<Value> {
    validate_finish_reason(reason)?;
    let id = format!("chatcmpl-zg-{id}");
    let choice = if chat {
        json!({"index":0,"message":{"role":"assistant","content":output},"finish_reason":reason,"logprobs":null})
    } else {
        json!({"index":0,"text":output,"finish_reason":reason,"logprobs":null})
    };
    Ok(
        json!({"id":id,"object":if chat{"chat.completion"}else{"text_completion"},"created":epoch(),
        "model":model,"choices":[choice],"usage":usage.completion_json()}),
    )
}
