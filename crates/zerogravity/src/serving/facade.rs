//! Facade view of the original engine service; no registry, queue or transport IO.
use super::{
    TextService,
    service::{Request, RequestEvents, SemanticRequest, ServiceError, ServingCharge},
};
type Result<T> = std::result::Result<T, ServiceError>;

impl TextService for crate::Runtime {
    fn embedding_dimensions(&self, alias: &str) -> Result<usize> {
        self.engine.embedding_dimensions(alias)
    }
    fn request_embeddings_admitted(
        &self,
        request: crate::embedding::EmbeddingRequest,
        dimensions: usize,
        ingress: Vec<ServingCharge>,
    ) -> Result<crate::embedding::EmbeddingReceipt> {
        self.engine
            .request_embeddings_admitted(request, dimensions, ingress)
    }
    fn validate_chat(&self, model: &str) -> Result<()> {
        self.engine.validate_chat(model)
    }
    fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents> {
        self.engine.request_admitted(request, ingress)
    }
    fn request_semantic_admitted(
        &self,
        request: SemanticRequest,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents> {
        self.engine.request_semantic_admitted(request, ingress)
    }
    fn charge_ingress(&self, session: &str, bytes: u64) -> Result<ServingCharge> {
        self.engine.charge_ingress(session, bytes)
    }
    fn model_ids(&self) -> Result<Vec<String>> {
        self.engine.model_ids()
    }
    fn max_output(&self) -> Result<usize> {
        self.engine.max_output()
    }
    fn admission(&self) -> Result<serde_json::Value> {
        self.engine.admission()
    }
    fn require_open(&self) -> crate::Result<()> {
        self.engine.require_open()
    }
}
