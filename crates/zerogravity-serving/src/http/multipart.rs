//! Bounded form-data decoding, after route authorization. No media semantics,
//! file paths, persistent upload store, or detached parser tasks.
use super::ingress_body::{Collected, Reader};
use crate::TextService;
use multer::{Constraints, Multipart, SizeLimit};
use tokio::time::{self, Instant};
use zerogravity_engine::router::{ServiceError, pending::ServingCharge};

const MAX_PARTS: usize = 32;
const MAX_NAME: usize = 128;
const MAX_FILENAME: usize = 255;
const MAX_CONTENT_TYPE: usize = 256;

#[derive(Clone, Copy)]
pub(super) struct Limits {
    pub parts: usize,
    pub part_bytes: usize,
}

pub(super) struct Part {
    pub name: String,
    // Client metadata only: never interpreted as a filesystem path.
    pub filename: Option<String>,
    pub content_type: Option<String>,
    backing: Collected,
}
impl Part {
    pub(super) fn bytes(&self) -> &[u8] {
        self.backing.bytes()
    }
}

pub(super) struct Form {
    parts: Vec<Part>,
    _metadata: ServingCharge,
}

impl Form {
    pub(super) fn parts(&self) -> &[Part] {
        &self.parts
    }
}

fn invalid() -> ServiceError {
    ServiceError::new(400, "invalid_multipart", "invalid multipart form")
}
fn timeout() -> ServiceError {
    ServiceError::new(408, "timeout", "HTTP body deadline reached")
}
fn parser_error(error: multer::Error) -> ServiceError {
    match error {
        multer::Error::StreamReadFailed(cause) => cause
            .downcast::<ServiceError>()
            .map(|error| *error)
            .unwrap_or_else(|_| invalid()),
        multer::Error::FieldSizeExceeded { .. } => {
            ServiceError::new(413, "part_too_large", "multipart part too large")
        }
        multer::Error::StreamSizeExceeded { .. } => {
            ServiceError::new(413, "body_too_large", "HTTP body too large")
        }
        multer::Error::NoMultipart => ServiceError::new(
            415,
            "unsupported_media_type",
            "multipart/form-data required",
        ),
        multer::Error::LockFailure => {
            ServiceError::new(500, "multipart_state", "multipart decoder state error")
        }
        _ => invalid(),
    }
}

fn boundary(content_type: &str) -> Result<String, ServiceError> {
    // RFC 2046 bchars, 1..=70 bytes, no trailing space. MIME syntax itself
    // is parsed by Multer, not a second local Content-Type parser.
    let boundary = multer::parse_boundary(content_type).map_err(parser_error)?;
    if boundary.is_empty()
        || boundary.len() > 70
        || boundary.ends_with(' ')
        || !boundary
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"'()+_,-./:=? ".contains(&b))
    {
        return Err(invalid());
    }
    Ok(boundary)
}

fn metadata(value: &str, limit: usize) -> Result<String, ServiceError> {
    if value.len() > limit || value.chars().any(char::is_control) {
        return Err(invalid());
    }
    Ok(value.to_owned())
}

/// Consume one complete form using the original decoded-body deadline.
/// Returned bytes and bounded metadata stay charged until the owner drops.
pub(super) async fn collect(
    reader: Reader,
    content_type: &str,
    service: &dyn TextService,
    limits: Limits,
) -> Result<Form, ServiceError> {
    let deadline = reader.deadline();
    let result = time::timeout_at(deadline, decode(reader, content_type, service, limits))
        .await
        .map_err(|_| timeout())?;
    if Instant::now() >= deadline {
        return Err(timeout());
    }
    result
}

async fn decode(
    mut reader: Reader,
    content_type: &str,
    service: &dyn TextService,
    limits: Limits,
) -> Result<Form, ServiceError> {
    let cap = reader.limit();
    if limits.parts == 0 || limits.parts > MAX_PARTS || limits.part_bytes == 0 {
        return Err(ServiceError::new(
            500,
            "multipart_limits",
            "invalid multipart limits",
        ));
    }
    // Conservative external-parser allowance, not an allocator census:
    // geometric buffer growth/overlap, parsed headers and scratch strings.
    // The decoded aggregate cap remains authoritative even for unknown
    // lengths. The output copies have their own exact backing charges.
    let _parser = service.charge_ingress("http-multipart-parser", 64 * 1024 + 8 * cap as u64)?;
    let boundary = boundary(content_type)?;
    let metadata_bytes =
        limits.parts * (std::mem::size_of::<Part>() + MAX_NAME + MAX_FILENAME + MAX_CONTENT_TYPE);
    let charge = service.charge_ingress("http-multipart-metadata", metadata_bytes as u64)?;
    let mut parts = Vec::new();
    parts.try_reserve_exact(limits.parts).map_err(|_| {
        ServiceError::new(503, "host_memory", "multipart metadata allocation failed")
    })?;
    let mut form = Form {
        parts,
        _metadata: charge,
    };
    let stream = async_stream::stream! {
        loop {
            match reader.next().await {
                Ok(Some(bytes)) => {
                    yield Ok(bytes);
                    // Multer pulls ready input until Pending. Give it a
                    // cooperative boundary between frames to decode parts
                    // before fetching the rest of a ready upload.
                    tokio::task::yield_now().await;
                }
                Ok(None) => break,
                Err(error) => { yield Err(error); break; }
            }
        }
    };
    let part_cap = cap.min(limits.part_bytes);
    let constraints = Constraints::new().size_limit(
        SizeLimit::new()
            .whole_stream(cap as u64)
            .per_field(part_cap as u64),
    );
    let mut parser = Multipart::with_constraints(stream, boundary, constraints);
    while let Some(mut field) = parser.next_field().await.map_err(parser_error)? {
        if form.parts.len() >= limits.parts {
            return Err(ServiceError::new(
                413,
                "too_many_parts",
                "too many multipart parts",
            ));
        }
        let headers = field.headers();
        // A file's name and media type never grant a filesystem or decoder
        // capability. Nested content remains opaque; no transfer decoding.
        if !headers
            .get("content-disposition")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(';').next())
            .is_some_and(|v| v.trim().eq_ignore_ascii_case("form-data"))
            || headers.contains_key("content-transfer-encoding")
        {
            return Err(invalid());
        }
        let name = metadata(
            field.name().filter(|n| !n.is_empty()).ok_or_else(invalid)?,
            MAX_NAME,
        )?;
        let filename = field
            .file_name()
            .map(|v| metadata(v, MAX_FILENAME))
            .transpose()?;
        let content_type = match headers.get("content-type") {
            Some(value) => {
                if field.content_type().is_none() {
                    return Err(invalid());
                }
                Some(metadata(
                    value.to_str().map_err(|_| invalid())?,
                    MAX_CONTENT_TYPE,
                )?)
            }
            None => None,
        };
        let mut backing = Collected::new(0, service, "http-multipart-part")?;
        while let Some(chunk) = field.chunk().await.map_err(parser_error)? {
            backing.append(&chunk, part_cap, service, "http-multipart-part")?;
        }
        form.parts.push(Part {
            name,
            filename,
            content_type,
            backing,
        });
    }
    // The patched parser reaches success only at decoded HTTP EOF, including
    // the original reader's length, trailer and deadline checks.
    drop(parser);
    Ok(form)
}

#[cfg(test)]
mod tests;
