//! Cold protocol-family availability, not loaded-model eligibility.
//! Admission always resolves the actual registered model and task through
//! the original runtime. A supported adapter is not a capability grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    Health,
    Models,
    Completions,
    ChatCompletions,
    Responses,
    Conversations,
    Embeddings,
    Transcriptions,
    Translations,
    Speech,
    Images,
    Realtime,
}
impl Endpoint {
    pub fn require_implemented(self) -> crate::Result<()> {
        match self {
            Self::Health
            | Self::Models
            | Self::Completions
            | Self::ChatCompletions
            | Self::Responses
            | Self::Conversations
            | Self::Embeddings
            | Self::Transcriptions
            | Self::Translations
            | Self::Realtime => Ok(()),
            Self::Speech | Self::Images => Err(crate::Error::Unsupported(format!(
                "Fabric endpoint {self:?} is not implemented"
            ))),
        }
    }
}
