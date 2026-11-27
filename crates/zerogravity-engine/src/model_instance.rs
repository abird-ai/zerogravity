//! Coarse source-owned text adapter; family math and publication stay in the adapter.
use crate::Result;
pub mod execution;
use std::sync::Arc;
use zerogravity_core::identity::SourceIdentity;

/// Measured selected-model input and its tokenizer allocation envelope.
/// Legacy single-user adapters price their historical text bound; structured
/// conversations must report the exact rendered bytes.
pub struct PromptBounds {
    pub input_bytes: usize,
    pub tokens: usize,
    pub scratch_bytes: u64,
}

/// Tokenizer handles must retain the source's original inventory admission.
pub trait TextTokenizer: Send + Sync {
    fn eos(&self) -> u32;
    fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String>;
    fn prompt_tokens(&self, input: &str, chat: bool) -> Result<Vec<u32>>;
    fn prompt_token_bound(&self, input_bytes: usize, chat: bool) -> usize;
    fn admitted_prompt_scratch_bound(&self, input_bytes: usize, chat: bool) -> Result<u64>;
    fn decode_scratch_bound(&self, tokens: usize) -> usize;

    /// Model-owned template measurement and tokenizer scratch bounds.
    /// Existing single-text adapters remain valid; structured conversations
    /// require an explicit model implementation rather than HTTP framing.
    fn semantic_prompt_bound(
        &self,
        input: &zerogravity_core::task::TextInput,
    ) -> Result<PromptBounds> {
        let (text, chat) = input.single_text().ok_or_else(|| {
            crate::Error::Unsupported("model has no structured conversation template".into())
        })?;
        Ok(PromptBounds {
            input_bytes: text.len(),
            tokens: self.prompt_token_bound(text.len(), chat),
            scratch_bytes: self.admitted_prompt_scratch_bound(text.len(), chat)?,
        })
    }
    fn semantic_prompt_tokens(
        &self,
        input: &zerogravity_core::task::TextInput,
    ) -> Result<Vec<u32>> {
        let (text, chat) = input.single_text().ok_or_else(|| {
            crate::Error::Unsupported("model has no structured conversation template".into())
        })?;
        self.prompt_tokens(text, chat)
    }
}

/// Successful calls return only after the original session's publication.
/// The engine schedules these operations; it never creates another publisher.
pub trait TextSession: Send {
    fn prefill_chunk_without_logits(&mut self, tokens: &[u32]) -> Result<()>;
    fn prefill_transaction(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;
    fn forward(&mut self, token: u32) -> Result<Vec<f32>>;
}

/// One immutable, admitted source shared by aliases and independent sessions.
pub trait PreparedModel: Send + Sync {
    fn source(&self) -> SourceIdentity;
    fn tokenizer(&self) -> Arc<dyn TextTokenizer>;
    fn max_context(&self) -> usize;
    fn max_token_bytes(&self) -> usize;
    fn vocabulary_size(&self) -> u64;
    /// Cold metadata only: no device IO, new source readers, or execution.
    /// Called while the original catalog admission lock is held.
    fn describe(&self) -> Result<crate::diagnostics::ModelDetails> {
        Ok(crate::diagnostics::ModelDetails::default())
    }
    /// Model capability, independent of protocol or response presentation.
    fn validate_chat(&self) -> std::result::Result<(), crate::router::ServiceError> {
        Ok(())
    }
    /// Explicit qualification for bounded text-only realtime conversation turns.
    fn public_realtime_text(&self) -> bool {
        false
    }
    /// Source-specific validation after count and host admission, before enqueue.
    fn validate_request(
        &self,
        _request: &crate::router::SemanticRequest,
    ) -> std::result::Result<(), crate::router::ServiceError> {
        Ok(())
    }
}

/// Host-logit execution; device feedback/cohorts use their typed executor instead.
pub trait LogitModel: PreparedModel {
    fn max_session_bytes(&self) -> u64;
    fn session_scratch_bytes(&self) -> u64;
    fn new_session(self: Arc<Self>) -> Result<Box<dyn TextSession>>;
}

/// A source-bound, stateless sequence encoder. A call returns only after
/// actual completion and never retains the caller's output slice. The
/// adapter owns tokenizer/scratch admission and validates the selected
/// context before executing; the scheduler owns batch output and lifetime.
pub trait EmbeddingEncoder: Send + Sync {
    fn source(&self) -> SourceIdentity;
    fn max_context(&self) -> usize;
    fn vocabulary_size(&self) -> u64;
    fn dimensions(&self) -> usize;
    /// Trusted adapter declaration of a qualified serving capability.
    /// Captured once at registration. Candidates and native-only encoders
    /// do not become public merely because they can calculate vectors.
    fn public_embeddings(&self) -> bool {
        false
    }
    fn validate_admission(
        &self,
        admission: &zerogravity_core::resource_arbiter::TransferAdmission,
    ) -> Result<()>;
    fn describe(&self) -> Result<crate::diagnostics::ModelDetails> {
        Ok(crate::diagnostics::ModelDetails::default())
    }
    fn embed_into(
        &self,
        input: &crate::embedding::EmbeddingInput,
        max_tokens: usize,
        output: &mut [f32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<usize>;
}

#[derive(Clone)]
pub struct ModelInstance {
    model: Arc<dyn PreparedModel>,
}
impl ModelInstance {
    pub fn new(model: Arc<dyn PreparedModel>) -> Self {
        Self { model }
    }
    pub fn source(&self) -> SourceIdentity {
        self.model.source()
    }
    pub fn prepared(&self) -> &Arc<dyn PreparedModel> {
        &self.model
    }
    pub fn shares_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.model, &other.model)
    }
}
