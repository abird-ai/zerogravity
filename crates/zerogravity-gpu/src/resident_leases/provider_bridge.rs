//! Explicit managed private-buffer/secondary-stream provider boundary.
//! No extra allocator, physical event ledger, or publication owner.
use super::{
    ForeignProviderCharge, ResidentLease, ResidentService, ResidentSubmission,
    require_provider_extents,
};
use crate::physical_quarantine::CompletionOwner;
use cubecl::{client::Client, cuda::OwnedCudaQueue};
use std::sync::Arc;
use zerogravity_core::{
    Error, Result,
    accelerator::program::ValueStorageFormat,
    device_events::DeviceAccess,
    placement::PreparedTensorLayout,
    provider::{ProviderBufferMode, ProviderCompletionMode, ProviderRequirements},
};

/// Checked cold compatibility for one original service and explicitly
/// selected queue/workspace. Borrows actual owners; no submission, transfer,
/// new event or Ready. Recheck live flight membership at enqueue.
#[derive(Clone, Copy)]
pub(crate) struct CudaProviderBinding<'a> {
    service: &'a Arc<ResidentService>,
    queue: &'a Arc<OwnedCudaQueue>,
    requirements: ProviderRequirements,
    workspace: &'a [&'a ResidentLease],
    charges: &'a [Arc<ForeignProviderCharge>],
}
impl<'a> CudaProviderBinding<'a> {
    fn require_queue(&self) -> Result<()> {
        self.service.ensure_usable()?;
        let original = self.service.pinned_cuda_queue()?;
        let same_stream = self.queue.logical_stream_id() == original.logical_stream_id();
        if self.queue.service_id() != original.service_id()
            || same_stream
                != (self.requirements.completion == ProviderCompletionMode::OriginalQueue)
            || (self.requirements.buffers == ProviderBufferMode::Private
                && self.workspace.is_empty())
            || self.service.client().is_capturing()
        {
            return Err(Error::Unsupported(
                "CUDA provider bridge stream/private storage differs or captures".into(),
            ));
        }
        Ok(())
    }
}

impl ResidentService {
    /// Preflight explicit queue and original admitted resources before
    /// creating a flight. Known incompatibility cannot fence a healthy owner.
    pub(crate) fn bind_cuda_provider<'a>(
        self: &'a Arc<Self>,
        queue: &'a Arc<OwnedCudaQueue>,
        requirements: ProviderRequirements,
        workspace: &'a [&'a ResidentLease],
        charges: &'a [Arc<ForeignProviderCharge>],
    ) -> Result<CudaProviderBinding<'a>> {
        let binding = CudaProviderBinding {
            service: self,
            queue,
            requirements,
            workspace,
            charges,
        };
        binding.require_queue()?;
        for lease in workspace {
            self.check_lease(lease)?;
        }
        require_provider_extents(self, requirements, workspace, charges)?;
        Ok(binding)
    }
}

/// Explicit physical view declaration over the original admitted lease.
/// Layout and encoding remain independent of the opaque allocation.
#[derive(Clone, Copy)]
pub(crate) struct ProviderView<'a> {
    lease: &'a ResidentLease,
    offset: usize,
    layout: PreparedTensorLayout,
    format: ValueStorageFormat,
}
impl<'a> ProviderView<'a> {
    pub(crate) fn new(
        lease: &'a ResidentLease,
        offset: usize,
        layout: PreparedTensorLayout,
        format: ValueStorageFormat,
    ) -> Result<Self> {
        let width = format.byte_width();
        if !layout.row_bytes().is_multiple_of(width)
            || !layout.row_stride_bytes().is_multiple_of(width)
            || !layout.alignment().is_multiple_of(width)
        {
            return Err(Error::State("provider view layout/encoding differs".into()));
        }
        lease.typed_view(offset, layout)?;
        Ok(Self {
            lease,
            offset,
            layout,
            format,
        })
    }
    /// Explicit copy only; an equal byte count is not a layout conversion.
    pub(crate) fn require_copy_to(self, destination: Self) -> Result<()> {
        self.copy_views(destination).map(|_| ())
    }

    fn copy_views(
        self,
        destination: Self,
    ) -> Result<(cubecl::server::Handle, cubecl::server::Handle)> {
        if self.layout != destination.layout
            || self.format != destination.format
            || self.layout.row_stride_bytes() != self.layout.row_bytes()
        {
            // This bridge issues one contiguous copy, not a row-wise scatter.
            // Strided views may contain other logical columns in their gaps.
            return Err(Error::Unsupported(
                "provider copy requires identical dense physical layout/encoding".into(),
            ));
        }
        if !Arc::ptr_eq(self.lease.service(), destination.lease.service()) {
            return Err(Error::Unsupported(
                "provider copy requires one original service".into(),
            ));
        }
        let from = self.lease.typed_view(self.offset, self.layout)?;
        let to = destination
            .lease
            .typed_view(destination.offset, destination.layout)?;
        if crate::resident_program::BufferRange::new(&from)?
            .overlaps(&crate::resident_program::BufferRange::new(&to)?)
        {
            return Err(Error::State("provider copy original views overlap".into()));
        }
        Ok((from, to))
    }
}

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
        source: ProviderView<'_>,
        destination: ProviderView<'_>,
    ) -> Result<()> {
        self.require_accesses([
            (source.lease, DeviceAccess::Read),
            (destination.lease, DeviceAccess::Write),
        ])?;
        let (from, to) = source.copy_views(destination)?;
        // SAFETY: exact original flight accesses, identical declared physical
        // layout/encoding and nonoverlap. No cast or implicit format conversion.
        unsafe {
            self.queue.enqueue_copy(
                from,
                to,
                source.layout.byte_len()?,
                source.layout.alignment(),
            )
        }
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
        binding: CudaProviderBinding<'_>,
        owner: CompletionOwner,
        enqueue: impl FnOnce(CudaProviderCommands<'_, 'flight>) -> Result<()>,
    ) -> Result<Self> {
        let CudaProviderBinding {
            service,
            queue,
            requirements,
            workspace,
            charges,
        } = binding;
        self.retain_completion_owner(owner)?;
        self.retain_completion_owner(queue.clone())?;
        for charge in charges {
            self.retain_completion_owner(charge.clone())?;
        }
        self.require_provider_resources(service, requirements, workspace, charges)?;
        // Live capture/usable state can change after cold preflight.
        binding.require_queue()?;
        let original = self.service.pinned_cuda_queue()?;
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
