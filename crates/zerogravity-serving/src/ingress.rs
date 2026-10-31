//! Transport limits are distinct from model/device admission.
use crate::{Error, Result};
#[derive(Clone, Copy, Debug)]
pub struct IngressLimits {
    pub connections: usize,
    pub body_bytes: usize,
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
        } else {
            Ok(())
        }
    }
}
