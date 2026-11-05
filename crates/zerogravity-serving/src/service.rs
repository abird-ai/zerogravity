//! Transport seam over the caller's existing scheduler; never owns shutdown.
use serde_json::Value;
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
    /// Preserve source endpoint policy even when HTTP has already framed chat.
    fn validate_chat(&self, _model: &str) -> Result<()> {
        Ok(())
    }
    /// Cold validation; an unconfigured runtime may still be configured later.
    fn require_open(&self) -> crate::Result<()> {
        Ok(())
    }
}

impl TextService for Runtime {
    fn validate_chat(&self, model: &str) -> Result<()> {
        self.with_text_router(|router| router.validate_chat(model))
    }
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
    fn validate_chat(&self, model: &str) -> Result<()> {
        Router::validate_chat(self, model)
    }
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
        Router::admission(self)
    }
}
