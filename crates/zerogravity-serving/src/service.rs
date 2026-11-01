//! Transport seam over the caller's existing scheduler; never owns shutdown.
use serde_json::{Value, json};
pub use zerogravity_engine::router::{
    Request, RequestEvents, ServiceError, pending::ServingCharge,
};
use zerogravity_engine::{Runtime, router::Router};

type Result<T> = std::result::Result<T, ServiceError>;

/// Short admission/snapshot operations only. Returned events retain their
/// original cancellation and physical owners; transport IO happens after return.
pub trait TextService: Send + Sync + 'static {
    fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents>;
    fn charge_ingress(&self, session: &str, bytes: u64) -> Result<ServingCharge>;
    fn model_ids(&self) -> Result<Vec<String>>;
    fn max_output(&self) -> Result<usize>;
    fn admission(&self) -> Result<Value>;
    /// Cold validation; an unconfigured runtime may still be configured later.
    fn require_open(&self) -> crate::Result<()> {
        Ok(())
    }
}

impl TextService for Runtime {
    fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents> {
        if request.input.len() > self.limits().max_input_bytes {
            return Err(ServiceError::new(
                400,
                "invalid_request",
                "request exceeds input/output bounds",
            ));
        }
        self.with_text_router(|router| router.request_admitted(request, ingress))
    }
    fn charge_ingress(&self, session: &str, bytes: u64) -> Result<ServingCharge> {
        self.with_text_router(|router| router.charge_ingress(session, bytes))
    }
    fn model_ids(&self) -> Result<Vec<String>> {
        self.with_text_router(TextService::model_ids)
    }
    fn max_output(&self) -> Result<usize> {
        self.with_text_router(TextService::max_output)
    }
    fn admission(&self) -> Result<Value> {
        self.with_text_router(TextService::admission)
    }
    fn require_open(&self) -> crate::Result<()> {
        Runtime::require_open(self)
    }
}

impl TextService for Router {
    fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents> {
        Router::request_admitted(self, request, ingress)
    }
    fn charge_ingress(&self, session: &str, bytes: u64) -> Result<ServingCharge> {
        Router::charge_ingress(self, session, bytes)
    }
    fn model_ids(&self) -> Result<Vec<String>> {
        Ok(Router::model_ids(self).map(str::to_owned).collect())
    }
    fn max_output(&self) -> Result<usize> {
        Ok(self.max_output)
    }
    fn admission(&self) -> Result<Value> {
        let (limit, reserved, remaining, serving) = self.memory_snapshot().ok_or_else(|| {
            ServiceError::new(500, "host_state", "host admission snapshot unavailable")
        })?;
        let model = reserved.checked_sub(serving);
        let queue = self.queue_snapshot();
        Ok(json!({"kind":"configured_host_cpu",
            "host_limit_bytes":limit,
            "total_host_limit_bytes":limit,
            "total_host_reserved_bytes":reserved,
            "reserved_bytes":reserved,
            "remaining_bytes":remaining,
            "model_host_reserved_bytes":model,
            "backend_host_reserved_bytes":0,
            "serving_host_reserved_bytes":serving,
            "queued_requests":queue.pending,
            "pending_admission_in_progress":queue.in_progress,
            "pending_queue_capacity":queue.capacity,
            "pending_queue_charged_bytes":queue.backing_bytes,
            "pending_queue_consumers":queue.consumers,
            "queued_requests_limit":queue.limit,
            "active_sessions":self.active_sessions(),
            "peak_active_sessions":self.peak_active_sessions(),
            "active_sessions_limit":self.active_limit()}))
    }
}
