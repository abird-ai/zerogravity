//! Native Responses protocol adapter, independent of Chat Completions.
//! Cold availability does not assert that a model is loaded or eligible.
pub fn require_implemented() -> crate::Result<()> {
    super::openai::Endpoint::Responses.require_implemented()
}
