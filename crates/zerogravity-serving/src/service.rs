//! Transport seam over the caller's existing scheduler; never owns shutdown.
use serde_json::Value;
pub use zerogravity_engine::router::{
    Request, RequestEvents, SemanticRequest, ServiceError, pending::ServingCharge,
};
use zerogravity_engine::{Runtime, router::Router, state_cache::PrefixSharingScope};

type Result<T> = std::result::Result<T, ServiceError>;

/// Short admission/snapshot operations only. Returned events retain their
/// original cancellation and physical owners; transport IO happens after return.
pub trait TextService: Send + Sync + 'static {
    fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents>;
    fn request_admitted_scoped(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
        sharing: PrefixSharingScope,
    ) -> Result<RequestEvents> {
        let _ = (request, ingress, sharing);
        Err(ServiceError::new(
            400,
            "prefix_sharing_unsupported",
            "text service does not support scoped prefix sharing",
        ))
    }
    /// Protocol-neutral ingress. Legacy embedders retain their single-text
    /// API; a conversation requires an actual template-capable adapter.
    fn request_semantic_admitted(
        &self,
        request: SemanticRequest,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents> {
        if request.input.is_chat() {
            self.validate_chat(&request.model)?;
        }
        self.request_admitted(request.into_legacy()?, ingress)
    }
    fn request_semantic_admitted_scoped(
        &self,
        request: SemanticRequest,
        ingress: Vec<ServingCharge>,
        sharing: PrefixSharingScope,
    ) -> Result<RequestEvents> {
        if request.input.is_chat() {
            self.validate_chat(&request.model)?;
        }
        self.request_admitted_scoped(request.into_legacy()?, ingress, sharing)
    }
    fn charge_ingress(&self, session: &str, bytes: u64) -> Result<ServingCharge>;
    fn model_ids(&self) -> Result<Vec<String>>;
    fn max_output(&self) -> Result<usize>;
    fn admission(&self) -> Result<Value>;
    /// Capability check for legacy embedding adapters.
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
        self.request_semantic_admitted(request.into(), ingress)
    }
    fn request_admitted_scoped(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
        sharing: PrefixSharingScope,
    ) -> Result<RequestEvents> {
        self.request_semantic_admitted_scoped(request.into(), ingress, sharing)
    }
    fn request_semantic_admitted(
        &self,
        request: SemanticRequest,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents> {
        self.request_semantic_ingress(request, ingress, None)
    }
    fn request_semantic_admitted_scoped(
        &self,
        request: SemanticRequest,
        ingress: Vec<ServingCharge>,
        sharing: PrefixSharingScope,
    ) -> Result<RequestEvents> {
        self.request_semantic_ingress(request, ingress, Some(sharing))
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
    fn request_admitted_scoped(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
        sharing: PrefixSharingScope,
    ) -> Result<RequestEvents> {
        Router::request_admitted_scoped(self, request, ingress, sharing)
    }
    fn request_semantic_admitted(
        &self,
        request: SemanticRequest,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents> {
        Router::request_semantic_admitted(self, request, ingress)
    }
    fn request_semantic_admitted_scoped(
        &self,
        request: SemanticRequest,
        ingress: Vec<ServingCharge>,
        sharing: PrefixSharingScope,
    ) -> Result<RequestEvents> {
        Router::request_semantic_admitted_scoped(self, request, ingress, sharing)
    }
    fn charge_ingress(&self, session: &str, bytes: u64) -> Result<ServingCharge> {
        Router::charge_ingress(self, session, bytes)
    }
    fn model_ids(&self) -> Result<Vec<String>> {
        Ok(Router::model_ids(self).map(str::to_owned).collect())
    }
    fn max_output(&self) -> Result<usize> {
        Ok(Router::max_output(self))
    }
    fn admission(&self) -> Result<Value> {
        Router::admission(self)
    }
}
