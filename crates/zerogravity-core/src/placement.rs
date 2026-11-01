//! Four independent placement requests. None is an admission or transfer receipt.
use crate::physical::PhysicalDeviceKey;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementTarget {
    Physical(PhysicalDeviceKey),
    /// Future authenticated IO/peer placement, never implicit remote enrollment.
    Storage,
    Remote,
}
#[derive(Clone, Debug)]
pub struct PlacementRequest {
    pub compute: PlacementTarget,
    pub parameters: PlacementTarget,
    pub mutable_state: PlacementTarget,
    pub reusable_cache: PlacementTarget,
}
#[derive(Clone, Debug)]
pub struct TransferRequest {
    pub from: PlacementTarget,
    pub to: PlacementTarget,
    pub bytes: u64,
}
mod tensor_layout;
pub use tensor_layout::PreparedTensorLayout;
