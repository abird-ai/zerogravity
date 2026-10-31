//! Engine policy only; original physical queues and admission remain below it.
use crate::{Error, Result};
pub use zerogravity_core::device_queue::{DeviceTask, DispatchChoice, FairDeviceQueue};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerLimits {
    pub max_active: usize,
    pub max_queued: usize,
    pub max_input_bytes: usize,
}
impl Default for SchedulerLimits {
    fn default() -> Self {
        Self {
            max_active: 1,
            max_queued: 16,
            max_input_bytes: 1 << 20,
        }
    }
}
impl SchedulerLimits {
    pub fn validate(self) -> Result<()> {
        if self.max_active == 0
            || self.max_input_bytes == 0
            || self.max_active.checked_add(self.max_queued).is_none()
        {
            return Err(Error::Limit("Fabric scheduler bounds".into()));
        }
        Ok(())
    }
}
