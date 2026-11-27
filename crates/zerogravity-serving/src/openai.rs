//! Legacy cold endpoint markers do not assert loaded-model capability.
//! Actual embedding eligibility is checked on the original registry at ingress/admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    Health,
    Models,
    Completions,
    ChatCompletions,
    Embeddings,
    Responses,
}
impl Endpoint {
    pub fn require_implemented(self) -> crate::Result<()> {
        match self {
            Self::Health | Self::Models | Self::Completions | Self::ChatCompletions => Ok(()),
            Self::Embeddings | Self::Responses => Err(crate::Error::Unsupported(format!(
                "Fabric endpoint {self:?} is not implemented"
            ))),
        }
    }
}
