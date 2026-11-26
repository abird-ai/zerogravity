//! Bounded text-only Responses input. No translation through Chat JSON.
use super::{Options, Wire, store::Store};
use crate::TextService;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use zerogravity_engine::router::{
    ChatMessage, ChatRole, MAX_CHAT_MESSAGES, SemanticRequest, ServiceError, TextInput,
    pending::ServingCharge,
};

type Result<T> = std::result::Result<T, ServiceError>;

fn invalid(param: &'static str, message: &'static str) -> ServiceError {
    ServiceError::new(400, "invalid_request", message).with_parameter(param)
}
fn unsupported(param: &'static str) -> ServiceError {
    ServiceError::new(
        400,
        "unsupported_parameter",
        "Responses capability is not supported",
    )
    .with_parameter(param)
}
fn optional<'a>(map: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    map.get(key).filter(|value| !value.is_null())
}
fn boolean(map: &Map<String, Value>, key: &'static str, default: bool) -> Result<bool> {
    optional(map, key)
        .map(|v| v.as_bool().ok_or_else(|| invalid(key, "expected boolean")))
        .unwrap_or(Ok(default))
}
fn only(map: &Map<String, Value>, allowed: &[&str], param: &'static str) -> Result<()> {
    if map.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(unsupported(param));
    }
    Ok(())
}
fn neutral(
    map: &Map<String, Value>,
    key: &'static str,
    accepts: impl FnOnce(&Value) -> bool,
) -> Result<()> {
    if optional(map, key).is_some_and(|v| !accepts(v)) {
        return Err(unsupported(key));
    }
    Ok(())
}
fn empty_array(value: &Value) -> bool {
    value.as_array().is_some_and(Vec::is_empty)
}
fn plain_text(value: &Value) -> Result<()> {
    let map = value
        .as_object()
        .ok_or_else(|| invalid("text", "text must be an object"))?;
    only(map, &["format"], "text")?;
    if let Some(format) = optional(map, "format") {
        let format = format
            .as_object()
            .ok_or_else(|| invalid("text", "text.format must be an object"))?;
        only(format, &["type"], "text")?;
        if format.get("type").and_then(Value::as_str) != Some("text") {
            return Err(unsupported("text"));
        }
    }
    Ok(())
}

// Borrowed input content; measure once before copying. Text parts are
// concatenated in-order within their original role, with no new separator.
fn fragments(
    value: &Value,
    role: ChatRole,
    mut visit: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    if let Some(text) = value.as_str() {
        return visit(text);
    }
    let parts = value
        .as_array()
        .ok_or_else(|| invalid("input", "message content must be text or text parts"))?;
    if parts.is_empty() {
        return Err(invalid("input", "message content parts must not be empty"));
    }
    for part in parts {
        let part = part
            .as_object()
            .ok_or_else(|| invalid("input", "content part must be an object"))?;
        let kind = part.get("type").and_then(Value::as_str);
        match kind {
            Some("input_text") => only(part, &["type", "text"], "input")?,
            Some("output_text") if role == ChatRole::Assistant => {
                only(part, &["type", "text", "annotations", "logprobs"], "input")?;
                neutral(part, "annotations", empty_array)?;
                neutral(part, "logprobs", empty_array)?;
            }
            _ => return Err(unsupported("input")),
        }
        visit(
            part.get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("input", "content text is required"))?,
        )?;
    }
    Ok(())
}

fn turns(input: &Value, mut visit: impl FnMut(ChatRole, &Value) -> Result<()>) -> Result<()> {
    if input.is_string() {
        return visit(ChatRole::User, input);
    }
    let items = input
        .as_array()
        .ok_or_else(|| invalid("input", "input must be text or message items"))?;
    if items.is_empty() || items.len() > MAX_CHAT_MESSAGES {
        return Err(invalid("input", "invalid message count"));
    }
    for item in items {
        let item = item
            .as_object()
            .ok_or_else(|| invalid("input", "message item must be an object"))?;
        // Full explicit assistant text can be replayed; resource references,
        // tools and non-text content need their own semantic capabilities.
        only(item, &["type", "role", "content", "status", "id"], "input")?;
        neutral(item, "type", |v| v.as_str() == Some("message"))?;
        neutral(item, "status", |v| v.as_str() == Some("completed"))?;
        let role = item
            .get("role")
            .and_then(Value::as_str)
            .and_then(ChatRole::parse)
            .ok_or_else(|| invalid("input", "unsupported message role"))?;
        if optional(item, "id").is_some_and(|v| {
            role != ChatRole::Assistant
                || v.as_str().is_none_or(|id| id.is_empty() || id.len() > 128)
        }) {
            return Err(unsupported("input"));
        }
        visit(
            role,
            item.get("content")
                .ok_or_else(|| invalid("input", "message content required"))?,
        )?;
    }
    Ok(())
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| ServiceError::new(400, "size_overflow", "Responses size overflow"))
}

pub(in crate::http) fn parse(
    body: &[u8],
    service: &dyn TextService,
    ingress: &mut Vec<ServingCharge>,
    output_limit: usize,
    resources: &Store,
) -> Result<(SemanticRequest, Options, Wire)> {
    let value = super::super::parse_json(body, Some(service), ingress)?;
    let map = value
        .as_object()
        .ok_or_else(|| invalid("input", "expected JSON object"))?;
    for key in map.keys() {
        match key.as_str() {
            "model"
            | "input"
            | "instructions"
            | "max_output_tokens"
            | "stream"
            | "store"
            | "temperature"
            | "top_p"
            | "text"
            | "tools"
            | "tool_choice"
            | "parallel_tool_calls"
            | "truncation"
            | "include"
            | "background"
            | "background_options"
            | "previous_response_id"
            | "conversation"
            | "reasoning"
            | "metadata"
            | "service_tier"
            | "top_logprobs"
            | "stream_options" => (),
            // Named refusal for known request capabilities, even when a
            // client sends a neutral-looking value we cannot honor.
            "prompt" => return Err(unsupported("prompt")),
            "prompt_cache_key" => return Err(unsupported("prompt_cache_key")),
            "prompt_cache_retention" => return Err(unsupported("prompt_cache_retention")),
            "safety_identifier" => return Err(unsupported("safety_identifier")),
            "user" => return Err(unsupported("user")),
            "max_tool_calls" => return Err(unsupported("max_tool_calls")),
            "context_management" => return Err(unsupported("context_management")),
            _ => {
                return Err(ServiceError::new(
                    400,
                    "unsupported_parameter",
                    "unknown Responses parameter",
                ));
            }
        }
    }
    let model = map
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("model", "model name required"))?;
    zerogravity_engine::router::validate_alias(model).map_err(|e| e.with_parameter("model"))?;
    let stream = boolean(map, "stream", false)?;
    // Foreground storage is opt-in. Background always needs a temporary
    // polling record, but store=false never permits durable continuation.
    let store = boolean(map, "store", false)?;
    let background = boolean(map, "background", false)?;
    if background && stream {
        return Err(unsupported("stream"));
    }
    if let Some(options) = optional(map, "background_options") {
        if !background {
            return Err(unsupported("background_options"));
        }
        let options = options
            .as_object()
            .ok_or_else(|| invalid("background_options", "expected object"))?;
        only(options, &["completion_window"], "background_options")?;
        if optional(options, "completion_window").is_some_and(|v| v.as_str() != Some("soonest")) {
            return Err(unsupported("background_options"));
        }
    }
    for name in ["conversation", "reasoning"] {
        neutral(map, name, |_| false)?;
    }
    let previous = optional(map, "previous_response_id")
        .map(|v| {
            v.as_str()
                .filter(|id| id.len() <= 128)
                .ok_or_else(|| invalid("previous_response_id", "response ID required"))
        })
        .transpose()?;
    let parent = previous
        .map(|id| {
            let parent = resources
                .get(id)
                .map_err(|e| e.with_parameter("previous_response_id"))?;
            if !parent.continuable {
                return Err(invalid(
                    "previous_response_id",
                    "previous response has no complete text output",
                ));
            }
            Ok(parent)
        })
        .transpose()?;
    neutral(map, "temperature", |v| v.as_f64() == Some(0.0))?;
    neutral(map, "top_p", |v| v.as_f64() == Some(1.0))?;
    neutral(map, "tools", empty_array)?;
    neutral(map, "include", empty_array)?;
    neutral(map, "truncation", |v| v.as_str() == Some("disabled"))?;
    neutral(map, "service_tier", |v| {
        matches!(v.as_str(), Some("auto" | "default"))
    })?;
    neutral(map, "top_logprobs", |v| v.as_u64() == Some(0))?;
    let tool_choice = match optional(map, "tool_choice").map(Value::as_str) {
        None | Some(Some("auto")) => "auto",
        Some(Some("none")) => "none",
        _ => return Err(unsupported("tool_choice")),
    };
    let parallel_tool_calls = boolean(map, "parallel_tool_calls", true)?;
    if let Some(text) = optional(map, "text") {
        plain_text(text)?;
    }
    if let Some(options) = optional(map, "stream_options") {
        let options = options
            .as_object()
            .ok_or_else(|| invalid("stream_options", "expected object"))?;
        only(options, &["include_obfuscation"], "stream_options")?;
        if !stream || boolean(options, "include_obfuscation", false)? {
            return Err(unsupported("stream_options"));
        }
    }
    let max = service.max_output()?;
    let max_tokens = match optional(map, "max_output_tokens") {
        None => max.min(32),
        Some(value) => value
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| *v > 0 && *v <= max)
            .ok_or_else(|| {
                invalid(
                    "max_output_tokens",
                    "output limit outside configured text capacity",
                )
            })?,
    };
    if max_tokens == 0 {
        return Err(invalid("max_output_tokens", "no output capacity"));
    }
    let instructions = optional(map, "instructions")
        .map(|v| {
            v.as_str()
                .ok_or_else(|| invalid("instructions", "instructions must be text"))
        })
        .transpose()?;
    let input = map
        .get("input")
        .ok_or_else(|| invalid("input", "input required"))?;
    let mut count = usize::from(instructions.is_some());
    let mut bytes = instructions.map_or(0, str::len);
    if let Some(parent) = &parent {
        count = add(count, add(parent.input.len(), 1)?)?;
        for turn in parent.input.iter().chain(std::iter::once(&parent.output)) {
            bytes = add(bytes, turn.text.len())?;
        }
    }
    turns(input, |role, content| {
        count = add(count, 1)?;
        fragments(content, role, |text| {
            bytes = add(bytes, text.len())?;
            Ok(())
        })
    })?;
    if count > MAX_CHAT_MESSAGES || bytes > super::super::MAX_BODY {
        return Err(
            ServiceError::new(413, "input_too_large", "Responses input exceeds limits")
                .with_parameter("input"),
        );
    }
    let metadata = optional(map, "metadata")
        .map(|v| {
            let metadata = v
                .as_object()
                .ok_or_else(|| invalid("metadata", "metadata must be an object"))?;
            if metadata.len() > 16 {
                return Err(invalid("metadata", "too many metadata entries"));
            }
            for (key, value) in metadata {
                if key.chars().count() > 64
                    || value.as_str().is_none_or(|s| s.chars().count() > 512)
                {
                    return Err(invalid("metadata", "metadata key/value exceeds limits"));
                }
            }
            Ok(metadata)
        })
        .transpose()?;
    // Price the typed input before copying, independently of response
    // metadata, which must survive retirement of the admitted input.
    let route_bytes = add(
        add(bytes, model.len())?,
        count * std::mem::size_of::<ChatMessage>(),
    )?;
    ingress.push(service.charge_ingress("http-responses-input", route_bytes as u64)?);
    let metadata_bytes = metadata.map_or(0, |m| {
        m.iter()
            .map(|(k, v)| k.len() + v.as_str().unwrap().len())
            .sum()
    });
    let wire = Wire::new(
        service,
        if background {
            // Pending owns every serialized polling snapshot. Only retain
            // the options/metadata allowance, not a second wire budget.
            0
        } else if store {
            output_limit.min(super::super::MAX_BODY)
        } else {
            output_limit
        },
        add(
            add(metadata_bytes, instructions.map_or(0, str::len))?,
            previous.map_or(0, str::len),
        )?,
    )?;
    let mut messages = Vec::with_capacity(count);
    if let Some(text) = instructions {
        messages.push(ChatMessage {
            role: ChatRole::Developer,
            text: text.into(),
        });
    }
    // Only this request's instructions precede the replay. The previous
    // response's instructions were never put in its continuation history.
    if let Some(parent) = parent {
        messages.extend(parent.input.iter().cloned());
        messages.push(parent.output.clone());
    }
    turns(input, |role, content| {
        let mut len = 0;
        fragments(content, role, |text| {
            len = add(len, text.len())?;
            Ok(())
        })?;
        let mut text = String::with_capacity(len);
        fragments(content, role, |s| {
            text.push_str(s);
            Ok(())
        })?;
        messages.push(ChatMessage { role, text });
        Ok(())
    })?;
    let metadata: BTreeMap<_, _> = metadata
        .into_iter()
        .flatten()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
        .collect();
    Ok((
        SemanticRequest {
            model: model.to_owned(),
            input: TextInput::Conversation(messages),
            max_tokens,
            stream,
            token_metrics: None,
        },
        Options {
            instructions: instructions.map(str::to_owned),
            metadata,
            max_tokens,
            tool_choice,
            parallel_tool_calls,
            store,
            background,
            previous: previous.map(str::to_owned),
            _charge: wire.charge.clone(),
        },
        wire,
    ))
}
