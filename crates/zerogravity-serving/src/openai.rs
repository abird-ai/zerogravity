//! Endpoint names do not imply protocol or model implementation.
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
