//! Loopback-only plaintext transport with an explicit optional bearer token.
use crate::{Error, Result};
use std::net::SocketAddr;
pub fn require_loopback(address: SocketAddr) -> Result<()> {
    if address.ip().is_loopback() {
        Ok(())
    } else {
        Err(Error::Unsupported(
            "HTTP is loopback-only; use a trusted TLS/auth reverse proxy".into(),
        ))
    }
}

/// Validate optional inference bearer syntax before listener or source effects.
pub fn validate_key(key: Option<&str>) -> Result<()> {
    if key.is_some_and(|key| {
        key.is_empty() || key.len() > 1024 || !key.bytes().all(|b| b.is_ascii_graphic())
    }) {
        return Err(Error::InvalidInput(
            "bearer token must be 1..=1024 visible ASCII bytes".into(),
        ));
    }
    Ok(())
}
