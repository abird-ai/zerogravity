//! External listeners require a later explicit authenticated policy.
use crate::{Error, Result};
use std::net::SocketAddr;
pub fn require_loopback(address: SocketAddr) -> Result<()> {
    if address.ip().is_loopback() {
        Ok(())
    } else {
        Err(Error::Unsupported(
            "Fabric external bind/auth policy is not implemented".into(),
        ))
    }
}
