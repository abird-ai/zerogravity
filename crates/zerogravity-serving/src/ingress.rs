//! Transport limits are distinct from model/device admission.
use crate::{Error, Result};
#[derive(Clone, Copy, Debug)]
pub struct IngressLimits {
    /// Concurrent retained HTTP connection tasks (1..=64).
    pub connections: usize,
    /// Decoded body limit (1..=1 MiB). Fixed lengths are checked before allocation;
    /// chunked requests are checked cumulatively before staging each frame.
    pub body_bytes: usize,
    /// Total emitted bytes per connection, including HTTP headers/chunk/SSE framing.
    /// Exhaustion closes the connection and cancels its request.
    pub output_bytes: usize,
}
impl Default for IngressLimits {
    fn default() -> Self {
        Self {
            connections: 16,
            body_bytes: 1 << 20,
            output_bytes: 1 << 20,
        }
    }
}
impl IngressLimits {
    pub fn validate(self) -> Result<()> {
        if self.connections == 0
            || self.body_bytes == 0
            || self.output_bytes == 0
            || self.connections.checked_mul(self.body_bytes).is_none()
            || self.connections.checked_mul(self.output_bytes).is_none()
        {
            Err(Error::Limit("Fabric ingress/output bounds".into()))
        } else if self.connections > crate::http::MAX_CONNECTIONS
            || self.body_bytes > crate::http::MAX_BODY
        {
            Err(Error::Unsupported(
                "HTTP supports at most 64 connections and a 1 MiB body".into(),
            ))
        } else {
            Ok(())
        }
    }
}
