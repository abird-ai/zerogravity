//! Typed in-process failures and bounded, payload-free diagnostic observations.
use std::sync::Arc;
use zerogravity_core::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCategory {
    Artifact,
    Input,
    Unsupported,
    Resource,
    Limit,
    State,
    Io,
}
impl FailureCategory {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Artifact => "artifact",
            Self::Input => "input",
            Self::Unsupported => "unsupported",
            Self::Resource => "resource",
            Self::Limit => "limit",
            Self::State => "state",
            Self::Io => "io",
        }
    }
}
/// Safe to retain in an operational ring or forward as structured metadata.
/// No arbitrary message, resource scope, pathname, input or credentials.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct FailureDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    pub code: &'static str,
    pub category: FailureCategory,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os_error: Option<i32>,
}
impl FailureDetails {
    fn new(category: FailureCategory, code: &'static str) -> Self {
        Self {
            status: None,
            code,
            category,
            requested_bytes: None,
            available_bytes: None,
            os_error: None,
        }
    }
    pub fn core(error: &Error) -> Self {
        use FailureCategory as C;
        let (category, code) = match error {
            Error::Malformed(_) => (C::Artifact, "invalid_artifact"),
            Error::InvalidInput(_) => (C::Input, "invalid_input"),
            Error::Unsupported(_) => (C::Unsupported, "unsupported"),
            Error::Resource(_) => (C::Resource, "resource_exhausted"),
            Error::Limit(_) => (C::Limit, "limit_exceeded"),
            Error::State(_) => (C::State, "inference_error"),
            Error::Io(_) => (C::Io, "io_error"),
        };
        let mut details = Self::new(category, code);
        match error {
            Error::Resource(limit) => {
                details.requested_bytes = Some(limit.requested);
                details.available_bytes = Some(limit.available);
            }
            Error::Io(error) => details.os_error = error.raw_os_error(),
            _ => {}
        }
        details
    }
    pub fn public_message(self) -> &'static str {
        match self.code {
            "backend_not_compiled" => "The requested execution backend is not compiled.",
            "cuda_toolkit_missing" | "cuda_cccl_missing" => {
                "The required CUDA toolkit components are unavailable."
            }
            "unauthorized" => "Bearer token required.",
            "remote_bind_unsupported" => "The requested listener binding is not supported.",
            "model_not_found" | "missing_model" => "The requested model is not available.",
            "invalid_model" => "Invalid model selection.",
            "invalid_json" => "Invalid JSON request.",
            "invalid_messages" => "Invalid chat messages.",
            "invalid_prompt" => "A text prompt is required.",
            "invalid_config" | "config_invalid" => "Invalid configuration.",
            "unsupported_chat" => "The requested model does not support this chat route.",
            "unsupported_media_type" => "JSON Content-Type required.",
            "not_found" => "Endpoint not found.",
            "queue_full" | "connections_full" | "backpressure" => {
                "Service capacity is exhausted; retry later."
            }
            "timeout" | "backpressure_timeout" => "The request deadline was exceeded.",
            "runtime_closed" | "runtime_failed" | "runtime_unconfigured" | "shutting_down" => {
                "The runtime is not available."
            }
            _ => match self.category {
                FailureCategory::Artifact => "Model artifact validation failed.",
                FailureCategory::Input => "Invalid request or configuration.",
                FailureCategory::Unsupported => "The requested execution route is unsupported.",
                FailureCategory::Resource => "The requested resource capacity is unavailable.",
                FailureCategory::Limit => {
                    "The request exceeds an execution or configuration limit."
                }
                FailureCategory::State => {
                    "The operation could not complete in the current runtime state."
                }
                FailureCategory::Io => "An input/output operation failed.",
            },
        }
    }
}

fn service_category(status: u16, code: &str) -> FailureCategory {
    use FailureCategory as C;
    match code {
        "invalid_artifact" => C::Artifact,
        "backend_not_compiled"
        | "runtime_unconfigured"
        | "cuda_toolkit_missing"
        | "cuda_cccl_missing"
        | "unsupported" => C::Unsupported,
        _ if code.starts_with("unsupported_") || code.ends_with("_unsupported") => C::Unsupported,
        "resource_exhausted" | "host_budget" | "source_capacity" | "cohort_capacity"
        | "queue_allocation" | "connections_full" | "queue_full" => C::Resource,
        "limit_exceeded" | "size_overflow" | "context_overflow" | "config_limit"
        | "id_exhausted" => C::Limit,
        "io_error" | "read_failed" | "config_unreadable" | "bind_failed" | "worker_spawn" => C::Io,
        _ if matches!(status, 413 | 429 | 431) => C::Limit,
        _ if matches!(status, 400 | 401 | 403 | 404 | 411 | 415) => C::Input,
        _ => C::State,
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ServiceError {
    pub status: u16,
    pub code: &'static str,
    /// Trusted in-process diagnostic. HTTP must use its explicit public projection.
    pub message: String,
    #[serde(skip)]
    origin: Option<Arc<Error>>,
    #[serde(skip)]
    parameter: Option<&'static str>,
}
impl ServiceError {
    pub fn new(status: u16, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            origin: None,
            parameter: None,
        }
    }
    pub fn inference(error: Error) -> Self {
        let details = FailureDetails::core(&error);
        let status = match details.category {
            FailureCategory::Resource => 503,
            FailureCategory::State | FailureCategory::Io => 500,
            _ => 400,
        };
        Self {
            status,
            code: details.code,
            message: error.to_string(),
            origin: Some(Arc::new(error)),
            parameter: None,
        }
    }
    /// Only static protocol parameter names, never arbitrary request values.
    pub fn with_parameter(mut self, parameter: &'static str) -> Self {
        self.parameter = Some(parameter);
        self
    }
    pub fn parameter(&self) -> Option<&'static str> {
        self.parameter
    }
    pub fn details(&self) -> FailureDetails {
        let mut details = self
            .origin
            .as_deref()
            .map(FailureDetails::core)
            .unwrap_or_else(|| {
                FailureDetails::new(service_category(self.status, self.code), self.code)
            });
        details.code = self.code;
        details.status = Some(self.status);
        details
    }
    pub fn into_core(self) -> Error {
        if let Some(origin) = self.origin {
            return Arc::try_unwrap(origin).unwrap_or_else(shared_core_error);
        }
        // An HTTP status is not evidence of artifact corruption. A
        // synthetic capacity error has no measured resource reservation.
        match service_category(self.status, self.code) {
            FailureCategory::Artifact => Error::Malformed(self.message),
            FailureCategory::Input => Error::InvalidInput(self.message),
            FailureCategory::Unsupported => Error::Unsupported(self.message),
            FailureCategory::Resource | FailureCategory::Limit => Error::Limit(self.message),
            FailureCategory::State => Error::State(self.message),
            FailureCategory::Io => Error::Io(std::io::Error::other(self.message)),
        }
    }
}
/// A service error can be cloned for multiple observers. Preserve the original
/// typed failure in that case too, including OS codes and custom I/O causes.
fn shared_core_error(origin: Arc<zerogravity_core::Error>) -> zerogravity_core::Error {
    use zerogravity_core::Error;
    match origin.as_ref() {
        Error::Malformed(message) => Error::Malformed(message.clone()),
        Error::InvalidInput(message) => Error::InvalidInput(message.clone()),
        Error::Unsupported(message) => Error::Unsupported(message.clone()),
        Error::Limit(message) => Error::Limit(message.clone()),
        Error::Resource(limit) => Error::Resource(limit.clone()),
        Error::State(message) => Error::State(message.clone()),
        Error::Io(error) => {
            let error = if let Some(code) = error.raw_os_error() {
                std::io::Error::from_raw_os_error(code)
            } else {
                std::io::Error::new(error.kind(), SharedIoCause(origin))
            };
            Error::Io(error)
        }
    }
}
#[derive(Debug)]
struct SharedIoCause(Arc<zerogravity_core::Error>);
impl std::fmt::Display for SharedIoCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.as_ref() {
            zerogravity_core::Error::Io(error) => std::fmt::Display::fmt(error, f),
            other => std::fmt::Display::fmt(other, f),
        }
    }
}
impl std::error::Error for SharedIoCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}
impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}: {}", self.status, self.code, self.message)
    }
}
impl std::error::Error for ServiceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.origin
            .as_deref()
            .map(|origin| origin as &dyn std::error::Error)
    }
}
impl From<zerogravity_core::Error> for ServiceError {
    fn from(error: zerogravity_core::Error) -> Self {
        Self::inference(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn core_errors_keep_stable_public_classification_and_clean_messages() {
        let resource = ServiceError::inference(zerogravity_core::Error::resource(
            "device memory / VRAM budget",
            "model admission",
            12,
            8,
        ));
        assert_eq!(
            (resource.status, resource.code),
            (503, "resource_exhausted")
        );
        assert!(
            resource
                .message
                .contains("requested 12 bytes, available 8 bytes")
        );

        let limit = ServiceError::inference(zerogravity_core::Error::Limit("context bound".into()));
        assert_eq!((limit.status, limit.code), (400, "limit_exceeded"));

        let state = ServiceError::inference(zerogravity_core::Error::State("worker state".into()));
        assert_eq!((state.status, state.code), (500, "inference_error"));

        assert!(matches!(resource.clone().into_core(),
            zerogravity_core::Error::Resource(limit)
                if limit.resource == "device memory / VRAM budget"
                    && limit.operation == "model admission"
                    && limit.requested == 12 && limit.available == 8));
        assert_eq!(
            resource.into_core().to_string(),
            "device memory / VRAM budget exhausted during model admission: requested 12 bytes, available 8 bytes"
        );
        assert_eq!(
            limit.into_core().to_string(),
            "limit exceeded: context bound"
        );
        assert_eq!(
            state.into_core().to_string(),
            "invalid session state: worker state"
        );

        let core = ServiceError::new(503, "resource_exhausted", "VRAM exhausted").into_core();
        assert!(matches!(
            core,
            zerogravity_core::Error::Limit(message) if message == "VRAM exhausted"
        ));
    }

    #[test]
    fn service_translation_preserves_cause_and_never_infers_corruption_from_status() {
        use std::error::Error as _;
        use zerogravity_core::Error;
        let cases = [
            Error::Malformed("bad tensor header".into()),
            Error::InvalidInput("bad setting".into()),
            Error::Unsupported("unsupported device".into()),
            Error::Limit("context bound".into()),
            Error::State("worker retired".into()),
            Error::Io(std::io::Error::from_raw_os_error(2)),
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "source denied",
            )),
        ];
        for error in cases {
            let message = error.to_string();
            let service = ServiceError::inference(error);
            assert_eq!(service.source().unwrap().to_string(), message);
            let wire = serde_json::to_value(&service).unwrap();
            assert_eq!(wire.as_object().unwrap().len(), 3);
            assert_eq!(wire["message"], message);
            // Shared and unique conversions have identical category/text.
            assert_eq!(service.clone().into_core().to_string(), message);
            assert_eq!(service.into_core().to_string(), message);
        }
        let io = ServiceError::inference(Error::Io(std::io::Error::from_raw_os_error(2)));
        assert!(matches!(io.clone().into_core(), Error::Io(e) if e.raw_os_error()==Some(2)));
        let io = ServiceError::inference(Error::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "source denied",
        )));
        assert!(
            matches!(io.clone().into_core(), Error::Io(e) if e.kind()==std::io::ErrorKind::PermissionDenied && e.get_ref().is_some())
        );
        assert!(
            matches!(ServiceError::new(400,"backend_not_compiled","CUDA or ROCm required").into_core(),
            Error::Unsupported(message) if message=="CUDA or ROCm required")
        );
        assert!(
            matches!(ServiceError::new(400,"invalid_config","invalid profile").into_core(),
            Error::InvalidInput(message) if message=="invalid profile")
        );
        assert!(matches!(
            ServiceError::new(400, "invalid_artifact", "corrupt header").into_core(),
            Error::Malformed(_)
        ));
        assert!(matches!(
            ServiceError::new(503, "runtime_closed", "retired").into_core(),
            Error::State(_)
        ));
        assert!(matches!(
            ServiceError::new(400, "limit_exceeded", "context bound").into_core(),
            Error::Limit(_)
        ));
    }

    #[test]
    fn operational_details_are_structured_not_raw_diagnostics() {
        let e = ServiceError::inference(Error::Resource(
            zerogravity_core::ResourceLimit::new("PRIVATE-PATH", "PRIVATE-OP", 12, 8)
                .with_scope("SECRET-TOKEN"),
        ));
        let details = e.details();
        assert_eq!(details.status, Some(503));
        assert_eq!(details.code, "resource_exhausted");
        assert_eq!(details.category, FailureCategory::Resource);
        assert_eq!(
            (details.requested_bytes, details.available_bytes),
            (Some(12), Some(8))
        );
        let wire = serde_json::to_string(&details).unwrap();
        assert!(!wire.contains("PRIVATE") && !wire.contains("SECRET"));
        assert!(!details.public_message().contains("PRIVATE"));
    }
}
