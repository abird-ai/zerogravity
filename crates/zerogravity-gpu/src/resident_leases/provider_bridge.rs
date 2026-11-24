//! Explicit managed private-buffer/secondary-stream provider boundary.
//! No extra allocator, physical event ledger, or publication owner.
use super::{ForeignProviderCharge, ResidentLease, ResidentSubmission};
use crate::physical_quarantine::CompletionOwner;
use cubecl::{client::Client, cuda::OwnedCudaQueue};
use std::sync::Arc;
use zerogravity_core::{
    Error, Result,
    device_events::DeviceAccess,
    placement::PreparedTensorLayout,
    provider::{ProviderBufferMode, ProviderCompletionMode, ProviderRequirements},
};

/// Borrowed command scope; no independent completion or allocation authority.
pub(crate) struct CudaProviderCommands<'a, 'flight> {
    flight: &'a ResidentSubmission<'flight>,
    queue: &'a OwnedCudaQueue,
    client: Client,
}

impl CudaProviderCommands<'_, '_> {
    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    pub(crate) fn require_accesses<'a>(
        &self,
        accesses: impl IntoIterator<Item = (&'a ResidentLease, DeviceAccess)>,
    ) -> Result<()> {
        self.flight.require_accesses(accesses)
    }

    /// Same layout/encoding only. All copied bytes (including destination
    /// initialization) belong to the caller's original submission.
    pub(crate) fn copy(
        &self,
        source: &ResidentLease,
        source_offset: usize,
        destination: &ResidentLease,
        destination_offset: usize,
        layout: PreparedTensorLayout,
    ) -> Result<()> {
        self.require_accesses([
            (source, DeviceAccess::Read),
            (destination, DeviceAccess::Write),
        ])?;
        let bytes = layout.byte_len()?;
        let from = source.view(source_offset, bytes)?;
        let to = destination.view(destination_offset, bytes)?;
        layout.validate_region(from.offset_start.unwrap_or(0), bytes as u64)?;
        layout.validate_region(to.offset_start.unwrap_or(0), bytes as u64)?;
        if crate::resident_program::BufferRange::new(&from)?
            .overlaps(&crate::resident_program::BufferRange::new(&to)?)
        {
            return Err(Error::State("provider copy original views overlap".into()));
        }
        // SAFETY: the scope's original flight retains exact accesses; the
        // ordered queue and the trusted caller's layout contract cover copying.
        unsafe { self.queue.enqueue_copy(from, to, bytes, layout.alignment()) }
            .map_err(|error| Error::State(format!("provider private buffer copy: {error}")))
    }
}

impl<'flight> ResidentSubmission<'flight> {
    /// Run one bounded provider region on the original queue, or explicitly
    /// hand off to a same-context secondary queue and return to the original.
    /// Failure consumes the armed flight and follows its existing quarantine.
    /// No new event ledger or Ready is minted by the bridge.
    ///
    /// # Safety
    /// Every command must use the supplied queue/client, the original admitted
    /// leases and retained provider owner. The closure must finish enqueueing
    /// synchronously, initialize all required outputs, and never escape work
    /// beyond the closing dependency. The original validator still decides
    /// readiness. No capture or other context/device is admitted here.
    pub(crate) unsafe fn with_cuda_provider(
        mut self,
        queue: &Arc<OwnedCudaQueue>,
        requirements: ProviderRequirements,
        workspace: &[&ResidentLease],
        charges: &[Arc<ForeignProviderCharge>],
        owner: CompletionOwner,
        enqueue: impl FnOnce(CudaProviderCommands<'_, 'flight>) -> Result<()>,
    ) -> Result<Self> {
        self.retain_completion_owner(owner)?;
        self.retain_completion_owner(queue.clone())?;
        for charge in charges {
            self.retain_completion_owner(charge.clone())?;
        }
        self.require_provider_resources(self.service, requirements, workspace, charges)?;
        let original = self.service.pinned_cuda_queue()?;
        if queue.service_id() != original.service_id()
            || (requirements.completion == ProviderCompletionMode::OriginalQueue
                && queue.logical_stream_id() != original.logical_stream_id())
            || (requirements.completion == ProviderCompletionMode::ExternalFence
                && queue.logical_stream_id() == original.logical_stream_id())
            || (requirements.buffers == ProviderBufferMode::Private && workspace.is_empty())
        {
            return Err(Error::Unsupported(
                "CUDA provider bridge stream/private storage differs".into(),
            ));
        }
        // SAFETY: this clone is exclusively scoped to the exact admitted queue.
        let client = unsafe { queue.bind_client(self.service.client().clone()) }
            .map_err(|error| Error::State(format!("CUDA provider bridge client: {error}")))?;
        if self.service.client().is_capturing() || client.is_capturing() {
            return Err(Error::Unsupported(
                "CUDA provider bridge cannot enter capture".into(),
            ));
        }
        let event_error = |error| Error::State(format!("CUDA provider bridge event: {error}"));
        let external = requirements.completion == ProviderCompletionMode::ExternalFence;
        if external {
            let before = original.record_event().map_err(event_error)?;
            queue.wait_for(&before).map_err(event_error)?;
        }
        enqueue(CudaProviderCommands {
            flight: &self,
            queue,
            client,
        })?;
        if external {
            let after = queue.record_event().map_err(event_error)?;
            original.wait_for(&after).map_err(event_error)?;
        }
        Ok(self)
    }
}
