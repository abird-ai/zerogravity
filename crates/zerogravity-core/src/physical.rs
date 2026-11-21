//! Original low-level physical owners. Engine policy does not mint another ledger.
pub use crate::device_events::{BufferLeaseId, DeviceBufferIdentity, DeviceEventId, EventPhase};
pub use crate::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
