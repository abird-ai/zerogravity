//! One bounded nonstreaming projection of the original encoder receipt.
//! No model mathematics, queue, tokenizer or detached task in HTTP.
use super::{
    body::{self, BoundedWriter, Reply},
    parse_json,
};
use crate::TextService;
use base64::{Engine, engine::general_purpose::STANDARD};
use hyper::Response;
use serde::{Serialize, Serializer};
use serde_json::Value;
use std::sync::Arc;
use zerogravity_engine::{
    embedding::{EmbeddingInput, EmbeddingReceipt, EmbeddingRequest, MAX_DIMENSIONS, MAX_INPUTS},
    router::{
        ServiceError,
        pending::{ServingCharge, checked_sum, vec_bytes},
        validate_alias,
    },
};

type Result<T> = std::result::Result<T, ServiceError>;

fn invalid_input() -> ServiceError {
    ServiceError::new(
        400,
        "invalid_input",
        "expected nonempty string, strings, token IDs or token-ID arrays",
    )
    .with_parameter("input")
}
fn unsupported(parameter: &'static str) -> ServiceError {
    ServiceError::new(
        400,
        "unsupported_parameter",
        "unsupported embedding control",
    )
    .with_parameter(parameter)
}
fn overflow() -> ServiceError {
    ServiceError::new(400, "size_overflow", "embedding projection extent overflow")
}
pub(super) fn parameter_error(error: ServiceError) -> ServiceError {
    match error.code {
        "model_not_found" | "unsupported_task" | "model_changed" => error.with_parameter("model"),
        "invalid_input" | "input_too_large" | "context_overflow" => error.with_parameter("input"),
        _ => error,
    }
}
#[derive(Clone, Copy)]
enum Borrowed<'a> {
    Text(&'a str),
    Tokens(&'a [Value]),
}
impl Borrowed<'_> {
    fn bytes(self) -> Result<u64> {
        match self {
            Self::Text(text) if !text.is_empty() => Ok(text.len() as u64),
            Self::Tokens(ids)
                if !ids.is_empty()
                    && ids
                        .iter()
                        .all(|id| id.as_u64().and_then(|n| u32::try_from(n).ok()).is_some()) =>
            {
                vec_bytes::<u32>(ids.len())
            }
            _ => Err(invalid_input()),
        }
    }
    fn owned(self) -> EmbeddingInput {
        match self {
            Self::Text(text) => EmbeddingInput::Text(text.to_owned()),
            Self::Tokens(ids) => EmbeddingInput::Tokens(
                ids.iter()
                    .map(|id| id.as_u64().expect("validated token ID") as u32)
                    .collect(),
            ),
        }
    }
}
#[derive(Clone, Copy)]
enum Encoding {
    Float,
    Base64,
}

pub(super) struct Wire {
    dimensions: usize,
    count: usize,
    encoding: Encoding,
    limit: usize,
    charge: Arc<ServingCharge>,
}
impl Wire {
    pub(super) fn dimensions(&self) -> usize {
        self.dimensions
    }
    fn new(
        service: &dyn TextService,
        dimensions: usize,
        count: usize,
        encoding: Encoding,
        limit: usize,
    ) -> Result<Self> {
        if !(1..=MAX_DIMENSIONS).contains(&dimensions) || !(1..=MAX_INPUTS).contains(&count) {
            return Err(ServiceError::new(
                500,
                "invalid_embedding",
                "unbounded encoder capability",
            ));
        }
        // Finite f32 JSON needs at most 32 bytes/coordinate including comma.
        // Bound by both actual declared width and configured transport cap.
        let row = match encoding {
            Encoding::Float => dimensions.checked_mul(32),
            Encoding::Base64 => dimensions
                .checked_mul(4)
                .and_then(|n| n.div_ceil(3).checked_mul(4)),
        }
        .ok_or_else(overflow)?;
        let bound = row
            .checked_add(128)
            .and_then(|n| n.checked_mul(count))
            .and_then(|n| n.checked_add(512))
            .ok_or_else(overflow)?;
        let limit = limit.min(bound);
        // Writer growth overlap, row/base64 temporaries, and small serde
        // descriptors; independent projection credit follows the bytes.
        let bytes = limit
            .checked_mul(2)
            .and_then(|n| n.checked_add(dimensions * 12 + 64 * 1024))
            .ok_or_else(overflow)?;
        let charge = Arc::new(service.charge_ingress("http-embedding-projection", bytes as u64)?);
        Ok(Self {
            dimensions,
            count,
            encoding,
            limit,
            charge,
        })
    }
}

pub(super) fn parse(
    body: &[u8],
    service: &dyn TextService,
    ingress: &mut Vec<ServingCharge>,
    limit: usize,
) -> Result<(EmbeddingRequest, Wire)> {
    let value = parse_json(body, Some(service), ingress)?;
    let map = value.as_object().ok_or_else(invalid_input)?;
    if map.keys().any(|key| {
        !matches!(
            key.as_str(),
            "model" | "input" | "dimensions" | "encoding_format" | "user"
        )
    }) {
        return Err(ServiceError::new(
            400,
            "unsupported_parameter",
            "unknown embedding parameter",
        ));
    }
    let model = map.get("model").and_then(Value::as_str).ok_or_else(|| {
        ServiceError::new(400, "invalid_model", "model required").with_parameter("model")
    })?;
    validate_alias(model).map_err(|e| e.with_parameter("model"))?;
    let dimensions = service
        .embedding_dimensions(model)
        .map_err(parameter_error)?;
    if let Some(v) = map.get("dimensions").filter(|v| !v.is_null())
        && v.as_u64() != Some(dimensions as u64)
    {
        return Err(unsupported("dimensions"));
    }
    if map.get("user").is_some_and(|v| !v.is_null()) {
        return Err(unsupported("user"));
    }
    let encoding = match map
        .get("encoding_format")
        .filter(|v| !v.is_null())
        .map(Value::as_str)
    {
        None | Some(Some("float")) => Encoding::Float,
        Some(Some("base64")) => Encoding::Base64,
        _ => return Err(unsupported("encoding_format")),
    };
    let input = map.get("input").ok_or_else(invalid_input)?;
    let mut rows = [None; MAX_INPUTS];
    let count = if let Some(text) = input.as_str() {
        rows[0] = Some(Borrowed::Text(text));
        1
    } else if let Some(array) = input.as_array().filter(|a| !a.is_empty()) {
        if array[0].is_number() {
            rows[0] = Some(Borrowed::Tokens(array));
            1
        } else {
            if array.len() > MAX_INPUTS {
                return Err(invalid_input());
            }
            let strings = array[0].is_string();
            for (slot, value) in rows.iter_mut().zip(array) {
                *slot = Some(if strings {
                    Borrowed::Text(value.as_str().ok_or_else(invalid_input)?)
                } else {
                    Borrowed::Tokens(value.as_array().ok_or_else(invalid_input)?)
                });
            }
            array.len()
        }
    } else {
        return Err(invalid_input());
    };
    let inputs = &rows[..count];
    let mut bytes = checked_sum(&[model.len() as u64, vec_bytes::<EmbeddingInput>(count)?])?;
    for row in inputs {
        bytes = checked_sum(&[bytes, row.expect("classified input").bytes()?])?;
    }
    ingress.push(service.charge_ingress("http-embedding-input", bytes)?);
    let wire = Wire::new(service, dimensions, count, encoding, limit)?;
    let request = EmbeddingRequest {
        model: model.to_owned(),
        inputs: inputs
            .iter()
            .map(|row| row.expect("classified input").owned())
            .collect(),
    };
    Ok((request, wire))
}

struct Vector<'a>(&'a [f32], Encoding);
impl Serialize for Vector<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self.1 {
            Encoding::Float => self.0.serialize(serializer),
            Encoding::Base64 => {
                let mut raw = Vec::with_capacity(self.0.len() * 4);
                for value in self.0 {
                    raw.extend(value.to_le_bytes());
                }
                serializer.serialize_str(&STANDARD.encode(&raw))
            }
        }
    }
}
#[derive(Serialize)]
struct Row<'a> {
    object: &'static str,
    index: usize,
    embedding: Vector<'a>,
}
#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}
#[derive(Serialize)]
struct Projection<'a> {
    object: &'static str,
    model: &'a str,
    data: Vec<Row<'a>>,
    usage: Usage,
}

pub(super) async fn completion(events: EmbeddingReceipt, wire: Wire) -> Result<Response<Reply>> {
    let result = tokio::time::timeout(body::RESPONSE_TIMEOUT, body::next(&events))
        .await
        .map_err(|_| ServiceError::new(504, "timeout", "embedding deadline reached"))?
        .map_err(|_| ServiceError::new(503, "worker_unavailable", "encoder worker disconnected"))?
        .map_err(parameter_error)?;
    if result.dimensions() != wire.dimensions
        || result.len() != wire.count
        || result.input_tokens() == 0
        || result.values().iter().any(|v| !v.is_finite())
    {
        return Err(ServiceError::new(
            500,
            "invalid_embedding",
            "unexpected encoder terminal",
        ));
    }
    let data = result
        .values()
        .chunks_exact(wire.dimensions)
        .enumerate()
        .map(|(index, vector)| Row {
            object: "embedding",
            index,
            embedding: Vector(vector, wire.encoding),
        })
        .collect();
    let usage = Usage {
        prompt_tokens: result.input_tokens(),
        total_tokens: result.input_tokens(),
    };
    let mut buffer = BoundedWriter {
        bytes: Vec::new(),
        limit: wire.limit,
    };
    serde_json::to_writer(
        &mut buffer,
        &Projection {
            object: "list",
            model: &events.model,
            data,
            usage,
        },
    )
    .map_err(|_| {
        ServiceError::new(
            413,
            "output_too_large",
            "embedding projection exceeds output limit",
        )
    })?;
    // The actual vector result and original receipt retire; only serialized
    // bytes and their independent physical credit survive socket writes.
    Ok(body::retained_json(buffer.bytes.into(), Some(wire.charge)))
}
