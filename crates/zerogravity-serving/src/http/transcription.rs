//! Bounded multipart projection of the original native speech receipt.
//! WAV transport only; no resampling, codec execution, filesystem or model math.
use super::{
    body::{self, BoundedWriter, Reply},
    multipart::{Form, Part},
};
use crate::TextService;
use hyper::Response;
use serde::Serialize;
use std::{io::Write, sync::Arc, time::Duration};
use zerogravity_engine::{
    router::{
        ServiceError,
        pending::{ServingCharge, vec_bytes},
        validate_alias,
    },
    transcription::{
        MAX_TRANSCRIPT_BYTES, SAMPLE_RATE, TranscriptionFinish, TranscriptionLimits,
        TranscriptionReceipt, TranscriptionRequest,
    },
};

type Result<T> = std::result::Result<T, ServiceError>;
// The reference speech adapter is finite but not interactive-speed. Use a
// separate hard deadline for this route; existing text deadlines stay intact.
const SPEECH_TIMEOUT: Duration = Duration::from_secs(600);

fn invalid(parameter: &'static str, message: &'static str) -> ServiceError {
    ServiceError::new(400, "invalid_input", message).with_parameter(parameter)
}
fn unsupported(parameter: &'static str) -> ServiceError {
    ServiceError::new(
        400,
        "unsupported_parameter",
        "unsupported transcription control",
    )
    .with_parameter(parameter)
}
fn audio_error(message: &'static str) -> ServiceError {
    invalid("file", message)
}
pub(super) fn parameter_error(error: ServiceError) -> ServiceError {
    match error.code {
        "model_not_found" | "model_changed" | "unsupported_task" => error.with_parameter("model"),
        "invalid_input" | "input_too_large" | "context_overflow" => error.with_parameter("file"),
        _ => error,
    }
}

/// Exact RIFF PCM16 mono16k, little endian. Skip bounded ancillary chunks,
/// including their required padding; never trust file suffix or a claimed
/// duration. Reject duplicate format/data, truncated or trailing bytes.
fn pcm16_wav(bytes: &[u8], maximum: usize) -> Result<&[u8]> {
    let bad = || audio_error("expected complete RIFF/WAVE PCM16 mono 16000 Hz");
    fn le16(v: &[u8]) -> u16 {
        u16::from_le_bytes(v.try_into().expect("two bytes"))
    }
    fn le32(v: &[u8]) -> u32 {
        u32::from_le_bytes(v.try_into().expect("four bytes"))
    }
    if bytes.len() < 12
        || &bytes[..4] != b"RIFF"
        || &bytes[8..12] != b"WAVE"
        || (le32(&bytes[4..8]) as usize).checked_add(8) != Some(bytes.len())
    {
        return Err(bad());
    }
    let mut at = 12;
    let mut format = false;
    let mut data = None;
    let mut chunks = 0;
    while at < bytes.len() {
        chunks += 1;
        if chunks > 64 || bytes.len() - at < 8 {
            return Err(bad());
        }
        let kind = &bytes[at..at + 4];
        let size = le32(&bytes[at + 4..at + 8]) as usize;
        let start = at + 8;
        let end = start
            .checked_add(size)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(bad)?;
        let payload = &bytes[start..end];
        match kind {
            b"fmt " => {
                if format
                    || !(size == 16 || (size == 18 && le16(&payload[16..18]) == 0))
                    || le16(&payload[..2]) != 1
                    || le16(&payload[2..4]) != 1
                    || le32(&payload[4..8]) != SAMPLE_RATE
                    || le32(&payload[8..12]) != SAMPLE_RATE * 2
                    || le16(&payload[12..14]) != 2
                    || le16(&payload[14..16]) != 16
                {
                    return Err(bad());
                }
                format = true;
            }
            b"data" => {
                if !format || data.is_some() || size == 0 || !size.is_multiple_of(2) {
                    return Err(bad());
                }
                if size / 2 > maximum {
                    return Err(ServiceError::new(
                        413,
                        "input_too_large",
                        "audio samples exceed admitted model limit",
                    )
                    .with_parameter("file"));
                }
                data = Some(payload);
            }
            _ => {}
        }
        at = end
            .checked_add(size % 2)
            .filter(|&at| at <= bytes.len())
            .ok_or_else(bad)?;
    }
    data.ok_or_else(bad)
}
fn scalar<'a>(part: &'a Part, parameter: &'static str) -> Result<&'a str> {
    if part.filename.is_some()
        || part.bytes().len() > 1024
        || part.content_type.as_ref().is_some_and(|v| {
            !v.split(';')
                .next()
                .is_some_and(|t| t.trim().eq_ignore_ascii_case("text/plain"))
        })
    {
        return Err(invalid(parameter, "expected bounded text form field"));
    }
    std::str::from_utf8(part.bytes()).map_err(|_| invalid(parameter, "invalid UTF-8 form field"))
}
#[derive(Clone, Copy)]
enum Format {
    Json,
    Text,
}
pub(super) struct Wire {
    format: Format,
    limit: usize,
    text_bytes: usize,
    charge: Arc<ServingCharge>,
}
pub(super) fn parse(
    form: &Form,
    service: &dyn TextService,
    ingress: &mut Vec<ServingCharge>,
    output_limit: usize,
) -> Result<(TranscriptionRequest, TranscriptionLimits, Wire)> {
    let mut file = None;
    let mut model = None;
    let mut format = None;
    let mut temperature = false;
    let mut stream = false;
    for part in form.parts() {
        match part.name.as_str() {
            "file" => {
                if file.replace(part).is_some() {
                    return Err(invalid("file", "duplicate file field"));
                }
                if part.filename.as_deref().is_none_or(str::is_empty)
                    || part.content_type.as_ref().is_some_and(|v| {
                        !matches!(
                            v.to_ascii_lowercase().as_str(),
                            "audio/wav"
                                | "audio/x-wav"
                                | "audio/wave"
                                | "audio/vnd.wave"
                                | "application/octet-stream"
                        )
                    })
                {
                    return Err(audio_error("expected WAV audio file"));
                }
            }
            "model" => {
                if model.replace(scalar(part, "model")?).is_some() {
                    return Err(invalid("model", "duplicate model field"));
                }
            }
            "response_format" => {
                if format.is_some() {
                    return Err(invalid("response_format", "duplicate response format"));
                }
                format = Some(match scalar(part, "response_format")? {
                    "json" => Format::Json,
                    "text" => Format::Text,
                    _ => return Err(unsupported("response_format")),
                });
            }
            "temperature" => {
                if temperature {
                    return Err(invalid("temperature", "duplicate temperature"));
                }
                temperature = true;
                if scalar(part, "temperature")?.parse::<f64>().ok() != Some(0.0) {
                    return Err(unsupported("temperature"));
                }
            }
            "stream" => {
                if stream {
                    return Err(invalid("stream", "duplicate stream"));
                }
                stream = true;
                if scalar(part, "stream")? != "false" {
                    return Err(unsupported("stream"));
                }
            }
            "language" => return Err(unsupported("language")),
            "prompt" => return Err(unsupported("prompt")),
            "chunking_strategy" => return Err(unsupported("chunking_strategy")),
            "include" | "include[]" => return Err(unsupported("include")),
            "timestamp_granularities" | "timestamp_granularities[]" => {
                return Err(unsupported("timestamp_granularities"));
            }
            "known_speaker_names" | "known_speaker_names[]" => {
                return Err(unsupported("known_speaker_names"));
            }
            "known_speaker_references" | "known_speaker_references[]" => {
                return Err(unsupported("known_speaker_references"));
            }
            _ => {
                return Err(ServiceError::new(
                    400,
                    "unsupported_parameter",
                    "unknown transcription parameter",
                ));
            }
        }
    }
    let model = model.ok_or_else(|| invalid("model", "model required"))?;
    validate_alias(model).map_err(|e| e.with_parameter("model"))?;
    let file = file.ok_or_else(|| audio_error("file required"))?;
    let limits = service
        .transcription_limits(model)
        .map_err(parameter_error)?;
    if limits.max_samples == 0
        || limits.max_tokens == 0
        || !(1..=MAX_TRANSCRIPT_BYTES).contains(&limits.max_output_bytes)
    {
        return Err(ServiceError::new(
            500,
            "invalid_model",
            "unbounded speech capability",
        ));
    }
    let raw = pcm16_wav(file.bytes(), limits.max_samples)?;
    // The original multipart backing stays charged while decoding. Admit
    // the owned PCM before allocating; native request admission retains its
    // own exact backing credit, then temporary transport credit retires.
    let input_bytes = vec_bytes::<f32>(raw.len() / 2)?
        .checked_add(model.len() as u64)
        .ok_or_else(|| audio_error("audio allocation extent"))?;
    ingress.push(service.charge_ingress("http-transcription-pcm", input_bytes)?);
    let format = format.unwrap_or(Format::Json);
    let bound = match format {
        Format::Json => limits.max_output_bytes * 6 + 256,
        Format::Text => limits.max_output_bytes,
    };
    let limit = output_limit.min(bound);
    let charge = Arc::new(
        service.charge_ingress("http-transcription-projection", (limit * 2 + 4096) as u64)?,
    );
    let samples = raw
        .chunks_exact(2)
        .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.)
        .collect();
    Ok((
        TranscriptionRequest {
            model: model.to_owned(),
            samples,
            sample_rate: SAMPLE_RATE,
            max_tokens: limits.max_tokens,
        },
        limits,
        Wire {
            format,
            limit,
            text_bytes: limits.max_output_bytes,
            charge,
        },
    ))
}

#[derive(Serialize)]
struct Usage {
    #[serde(rename = "type")]
    kind: &'static str,
    input_tokens: usize,
    output_tokens: usize,
    total_tokens: usize,
}
#[derive(Serialize)]
struct Projection<'a> {
    text: &'a str,
    usage: Usage,
}

pub(super) async fn completion(
    receipt: TranscriptionReceipt,
    wire: Wire,
) -> Result<Response<Reply>> {
    let transcript = tokio::time::timeout(SPEECH_TIMEOUT, body::next(&receipt))
        .await
        .map_err(|_| ServiceError::new(504, "timeout", "transcription deadline reached"))?
        .map_err(|_| ServiceError::new(503, "worker_unavailable", "speech worker disconnected"))?
        .map_err(parameter_error)?;
    let usage = transcript.usage();
    // The OpenAI transcription shape has no finish_reason. Do not expose
    // a length-limited partial string as a successful complete transcript.
    if usage.finish == TranscriptionFinish::Length {
        return Err(ServiceError::new(
            422,
            "transcription_incomplete",
            "transcription reached its admitted token limit",
        ));
    }
    let total = usage
        .input_tokens
        .checked_add(usage.output_tokens)
        .ok_or_else(|| {
            ServiceError::new(500, "invalid_transcript", "transcription usage overflow")
        })?;
    if transcript.text().len() > wire.text_bytes
        || usage.input_tokens == 0
        || usage.output_tokens == 0
    {
        return Err(ServiceError::new(
            500,
            "invalid_transcript",
            "unexpected speech terminal",
        ));
    }
    let mut buffer = BoundedWriter {
        bytes: Vec::new(),
        limit: wire.limit,
    };
    let encoded = match wire.format {
        Format::Json => serde_json::to_writer(
            &mut buffer,
            &Projection {
                text: transcript.text(),
                usage: Usage {
                    kind: "tokens",
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    total_tokens: total,
                },
            },
        )
        .map_err(std::io::Error::other),
        Format::Text => buffer.write_all(transcript.text().as_bytes()),
    };
    encoded.map_err(|_| {
        ServiceError::new(
            413,
            "output_too_large",
            "transcription projection exceeds output limit",
        )
    })?;
    let content_type = match wire.format {
        Format::Json => "application/json",
        Format::Text => "text/plain; charset=utf-8",
    };
    Ok(body::retained_body(
        buffer.bytes.into(),
        Some(wire.charge),
        content_type,
    ))
}

#[cfg(test)]
mod tests;
