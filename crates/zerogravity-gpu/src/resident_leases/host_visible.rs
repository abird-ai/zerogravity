//! Explicit coherent host producer on the original Vulkan allocation.
//! No pointer export, second backing, implicit staging or device-write read.
use super::*;
use cubecl::{
    device::WgpuBackend,
    wgpu::{AutoCompiler, WgpuServer},
};
use zerogravity_core::{
    device_events::DeviceBufferIdentity,
    resource_arbiter::{AllocationClass, AllocationRequest},
};

impl ResidentService {
    /// Initialize one private scratch/activation view through its actual
    /// coherent host mapping. Borrow the caller's already-owned bytes; do
    /// not allocate a host staging copy or count the mapped storage twice.
    /// The original queue, range ledger, backend barrier and Ready own
    /// lifetime/publication. Mapping misses refuse before a submitted event.
    #[allow(
        dead_code,
        reason = "qualified mapped producer; ordinary uploads retain their nonblocking policy"
    )]
    pub(crate) async fn initialize_host_visible(
        self: &Arc<Self>,
        lease: &ResidentLease,
        data: &[u8],
        expected_root: u64,
    ) -> Result<(ResidentReady, DeviceBufferIdentity)> {
        self.ensure_usable()?;
        self.check_lease(lease)?;
        if data.len() != lease.bytes()
            || data.is_empty()
            || lease.is_initialized()
            || lease.0.workspace.is_some()
            || !matches!(lease.role(), BufferRole::Scratch | BufferRole::Activation)
        {
            return Err(Error::State(
                "host producer requires a fresh exact private scratch/activation view".into(),
            ));
        }
        if !matches!(self.physical_owner().device(), Device::Wgpu(device) if device.backend == WgpuBackend::Vulkan)
        {
            return Err(Error::Unsupported(
                "coherent host producer requires original Vulkan mapping".into(),
            ));
        }

        let ticket = self.queue().submit(
            self.budget(),
            self.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: lease.bytes() as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        self.ensure_usable()?;
        if lease.is_initialized() {
            return Err(Error::State(
                "host producer view initialized before admission".into(),
            ));
        }
        let handle = lease.handle();
        // The actual managed resource pins this exact allocation/range
        // against relocation. No address or mapped reference escapes it.
        let mapped = self
            .client()
            .get_resource::<WgpuServer<AutoCompiler>>(handle.clone())
            .map_err(|error| Error::State(format!("host-visible resource: {error}")))?;
        if mapped.resource().size != lease.bytes() as u64 || !mapped.resource().host_visible() {
            return Err(Error::Unsupported(
                "original allocation has no exact coherent host mapping".into(),
            ));
        }
        let identity = self
            .ledger
            .lock()
            .map_err(|_| Error::State("host-visible backing ledger poisoned".into()))?
            .buffer_identity(lease.id())?;
        if identity.bytes() != lease.bytes() {
            return Err(Error::State("host-visible backing extent differs".into()));
        }

        // Only this actual mapping's original Write event permits
        // the host copy. Same PCI/UMA addressability is never sufficient.
        slot.begin_submission()?;
        let flight = self.submit(expected_root, &[(lease.clone(), DeviceAccess::Write)])?;
        self.client()
            .sync_buffers([&handle])
            .await
            .map_err(|error| Error::State(format!("host-visible prior completion: {error}")))?;
        flight.require_accesses([(lease, DeviceAccess::Write)])?;
        // SAFETY: the exact managed resource is retained; original queue and
        // exclusive range Write prevent any conflicting fabric work. The
        // actual backend barrier completed. No foreign/untracked access is
        // admitted, no mapped pointer escapes, and the host copy is synchronous.
        if !unsafe { mapped.resource().copy_to_host_mapping(data) } {
            return Err(Error::State("retained host-visible mapping changed".into()));
        }
        // Same logical payload accounting as ordinary mapped/queued uploads;
        // not a measurement of DMA or a second memory charge.
        self.uploaded_value_bytes
            .fetch_add(data.len() as u64, Ordering::AcqRel);
        // Host writes completed before the next queue submission. No pending
        // provider work and no CPU read of GPU-written bytes. Same Ready as
        // other producers; this cannot publish a session root.
        let ready = flight.finish_ready(std::slice::from_ref(lease))?;
        slot.finish()?;
        Ok((ready, identity))
    }
}
