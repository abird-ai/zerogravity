//! Responses is independently unsupported, not an alias for chat completions.
pub fn require_implemented() -> crate::Result<()> {
    super::openai::Endpoint::Responses.require_implemented()
}
