use crate::{Error, Result};
pub use zerogravity_core::placement::{PlacementRequest, PlacementTarget, TransferRequest};
/// No hardware probing, allocation or implicit fallback.
pub fn plan(_request: &PlacementRequest) -> Result<Vec<TransferRequest>> {
    Err(Error::Unsupported(
        "Fabric placement/transfer planning awaits provider adaptation".into(),
    ))
}
