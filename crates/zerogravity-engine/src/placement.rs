use crate::Result;
pub use zerogravity_core::placement::{PlacementRequest, PlacementTarget, TransferRequest};
/// No inter-domain transfer for colocated roles. No hardware probing,
/// allocation, buffer/stream compatibility assertion or implicit fallback.
pub fn plan(request: &PlacementRequest) -> Result<Vec<TransferRequest>> {
    request.colocated_device()?;
    Ok(Vec::new())
}
