//! Engine policy only; original physical queues and admission remain below it.
use crate::{Error, Result};
pub use zerogravity_core::device_queue::{DeviceTask, DispatchChoice, FairDeviceQueue};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerLimits {
    pub max_active: usize,
    /// None removes only the count ceiling; physical host admission still bounds every entry.
    pub max_queued: Option<usize>,
    pub max_input_bytes: usize,
}
impl Default for SchedulerLimits {
    fn default() -> Self {
        Self {
            max_active: 1,
            max_queued: Some(16),
            max_input_bytes: 1 << 20,
        }
    }
}
impl SchedulerLimits {
    pub fn validate(self) -> Result<()> {
        if self.max_active == 0
            || self.max_input_bytes == 0
            || self
                .max_queued
                .is_some_and(|queued| self.max_active.checked_add(queued).is_none())
        {
            return Err(Error::Limit("Fabric scheduler bounds".into()));
        }
        Ok(())
    }
    pub fn allows_pending(self, configured: Option<usize>) -> bool {
        match (self.max_queued, configured) {
            (None, _) => true,
            (Some(limit), Some(configured)) => configured <= limit,
            (Some(_), None) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_count_ceiling_preserves_bounded_and_host_only_admission() {
        let bounded = SchedulerLimits::default();
        assert!(bounded.allows_pending(Some(16)));
        assert!(!bounded.allows_pending(Some(17)));
        assert!(!bounded.allows_pending(None));
        let host_only = SchedulerLimits {
            max_queued: None,
            ..bounded
        };
        assert!(host_only.validate().is_ok());
        assert!(host_only.allows_pending(None));
        assert!(host_only.allows_pending(Some(usize::MAX)));
    }
}
