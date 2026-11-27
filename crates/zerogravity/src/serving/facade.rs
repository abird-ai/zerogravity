//! Facade view of the original engine service; no registry, queue or transport IO.
use super::{
    TextService,
    service::{Request, RequestEvents, SemanticRequest, ServiceError, ServingCharge},
};
type Result<T> = std::result::Result<T, ServiceError>;

impl TextService for crate::Runtime {
    fn realtime_text_registration(&self, alias: &str) -> Result<u64> {
        self.engine.realtime_text_registration(alias)
    }
    fn request_realtime_text_admitted(
        &self,
        request: SemanticRequest,
        registration: u64,
        ingress: Vec<ServingCharge>,
        sharing: Option<zerogravity_engine::state_cache::PrefixSharingScope>,
    ) -> Result<RequestEvents> {
        self.engine
            .request_realtime_text_admitted(request, registration, ingress, sharing)
    }
    fn translation_limits(
        &self,
        alias: &str,
    ) -> Result<zerogravity_serving::service::TranscriptionLimits> {
        self.engine.translation_limits(alias)
    }
    fn request_translation_admitted(
        &self,
        request: crate::transcription::TranscriptionRequest,
        limits: zerogravity_serving::service::TranscriptionLimits,
        ingress: Vec<ServingCharge>,
    ) -> Result<crate::transcription::TranscriptionReceipt> {
        self.engine
            .request_translation_admitted(request, limits, ingress)
    }
    fn transcription_limits(
        &self,
        alias: &str,
    ) -> Result<zerogravity_serving::service::TranscriptionLimits> {
        self.engine.transcription_limits(alias)
    }
    fn request_transcription_admitted(
        &self,
        request: crate::transcription::TranscriptionRequest,
        limits: zerogravity_serving::service::TranscriptionLimits,
        ingress: Vec<ServingCharge>,
    ) -> Result<crate::transcription::TranscriptionReceipt> {
        self.engine
            .request_transcription_admitted(request, limits, ingress)
    }
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
