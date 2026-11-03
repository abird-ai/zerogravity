//! Weak observation of the original native/service owners; no execution authority.
use super::*;
pub struct SourceTextNativeProbe {
    pub(super) native: crate::hip::native::FinalNativeProbe,
    pub(super) service: std::sync::Weak<ResidentService>,
    pub(super) physical: zerogravity_core::resource_arbiter::PhysicalDeviceKey,
}
#[derive(Debug)]
pub struct SourceTextNativeState {
    pub retained: bool,
    pub modules: usize,
    pub code_bytes: usize,
    pub host_bytes: usize,
    pub device_bytes: usize,
    pub service_retained: bool,
    pub service_usable: bool,
    pub pending_events: usize,
    pub quarantined_handles: usize,
    pub completion_owners: usize,
    pub resident_bytes: usize,
    pub physical: zerogravity_core::resource_arbiter::PhysicalDeviceKey,
}
impl SourceTextNativeProbe {
    pub(super) fn from_original(
        native: crate::hip::native::FinalNativeProbe,
        service: &Arc<ResidentService>,
    ) -> Self {
        Self {
            native,
            service: Arc::downgrade(service),
            physical: service.physical_key(),
        }
    }
    pub fn observe(&self) -> Result<SourceTextNativeState> {
        let o = self.native.observe()?;
        let service = self.service.upgrade();
        let (usable, events, handles, owners, resident) = if let Some(service) = &service {
            (
                service.ensure_usable().is_ok(),
                service.pending_events()?,
                crate::physical_quarantine::quarantined_handles_for(service.client())?,
                crate::physical_quarantine::quarantined_completion_owners_for(service.client())?,
                service.resident_bytes()?,
            )
        } else {
            (false, 0, 0, 0, 0)
        };
        Ok(SourceTextNativeState {
            retained: o.retained,
            modules: o.modules,
            code_bytes: o.code_bytes,
            host_bytes: o.host_bytes,
            device_bytes: o.device_bytes,
            service_retained: service.is_some(),
            service_usable: usable,
            pending_events: events,
            quarantined_handles: handles,
            completion_owners: owners,
            resident_bytes: resident,
            physical: self.physical.clone(),
        })
    }
}
