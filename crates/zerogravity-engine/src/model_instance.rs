//! Coarse source-owned text adapter; family math and publication stay in the adapter.
use crate::Result;
use std::sync::Arc;
use zerogravity_core::identity::SourceIdentity;

/// Tokenizer handles must retain the source's original inventory admission.
pub trait TextTokenizer: Send + Sync {
    fn eos(&self) -> u32;
    fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String>;
    fn prompt_tokens(&self, input: &str, chat: bool) -> Result<Vec<u32>>;
    fn prompt_token_bound(&self, input_bytes: usize, chat: bool) -> usize;
    fn admitted_prompt_scratch_bound(&self, input_bytes: usize, chat: bool) -> Result<u64>;
    fn decode_scratch_bound(&self, tokens: usize) -> usize;
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
    fn max_session_bytes(&self) -> u64;
    fn session_scratch_bytes(&self) -> u64;
    fn max_token_bytes(&self) -> usize;
    fn vocabulary_size(&self) -> u64;
    fn new_session(self: Arc<Self>) -> Result<Box<dyn TextSession>>;
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
