//! Policy inputs, not a second allocator or source owner.
use zerogravity_core::physical::{PhysicalDeviceKey, ResourceOwner};
pub struct ResidencyRequest {
    pub owner: ResourceOwner,
    pub device: PhysicalDeviceKey,
    pub bytes: u64,
}
