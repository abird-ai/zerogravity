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
        Err(crate::Error::Unsupported(format!(
            "Fabric endpoint {self:?} awaits protocol migration"
        )))
    }
}
