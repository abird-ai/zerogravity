//! Private operator ownership through the one original resident GPU service.
use super::*;
use crate::resident_leases::{ResidentLease, ResidentReady, ResidentService, ResidentSubmission};
use cubecl::bytes::Bytes;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

struct SourceOwner {
    _window: Arc<TensorWindow>,
    _input: Arc<[u8]>,
    _input_admission: Arc<WindowWorkspace>,
    _weight_transport: WindowWorkspace,
    _input_transport: WindowWorkspace,
    _readback_admission: WindowWorkspace,
}
fn cancel(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        Err(Error::State("private BF16 projection cancelled".into()))
    } else {
        Ok(())
    }
}

pub(crate) struct PreparedProjection {
    service: Arc<ResidentService>,
    shape: Shape,
    identity: Identity,
    owner: Arc<SourceOwner>,
    weights: ResidentLease,
    input: ResidentLease,
    zero: ResidentLease,
    accumulation: ResidentLease,
    rounded: ResidentLease,
    status: ResidentLease,
}
impl PreparedProjection {
    /// A PURE operator correlation, never an invented SessionAuthority/root.
    /// The caller supplies an existing pinned original service, not a device
    /// from which a second context/queue might be created.
    pub(crate) async fn prepare(
        service: Arc<ResidentService>,
        window: Bf16Window,
        input: Bf16Inputs,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        cancel(cancelled)?;
        let shape = input.shape;
        let bytes = shape.owned_bytes()?;
        let client = service.client();
        let hardware = &client.properties().hardware;
        if !crate::activation_quant_fused::is_cuda_service(client)
            || hardware.max_units_per_cube < 64
            || hardware.max_cube_dim.0 < 64
            || hardware.max_shared_memory_size < 512
            || hardware.max_bindings < 6
            || hardware.max_cube_count.0 < shape.outputs as u32
        {
            return Err(Error::Unsupported(
                "CPU-profile BF16 CUDA workgroup/shape capability".into(),
            ));
        }
        service.pinned_cuda_queue()?;
        if window.identity.descriptor.source != input.source.source_identity()
            || service.source_identity() != input.source.source_identity()
            || !window.source.same_issuer(&input.source)
            || window.identity.rows.len() != shape.rows
            || window.identity.descriptor.columns != shape.columns
            || window.identity.window_bytes != shape.weight_bytes
        {
            return Err(malformed(
                "cold original service/source/full geometry binding",
            ));
        }
        if service.remaining_capacity()? < bytes {
            return Err(Error::Limit(
                "private BF16 operator capacity before launch".into(),
            ));
        }
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        cancel(cancelled)?;
        // Other fair work may have retained outputs while this ticket waited.
        // Refuse before the first initialization command or armed work guard.
        service.ensure_usable()?;
        if service.remaining_capacity()? < bytes {
            return Err(Error::Limit(
                "private BF16 operator capacity before launch".into(),
            ));
        }
        // Retain authentic staged bytes before ANY allocation/H2D. The armed
        // original initialization puts this owner in its original quarantine.
        let owner = Arc::new(SourceOwner {
            _window: window.owner.clone(),
            _input: input.bytes.clone(),
            _input_admission: input._admission.clone(),
            // Original upload_bytes creates one Vec for each queued H2D.
            // These actual same-source RAM charges cover those extra copies.
            _weight_transport: window.source.reserve_workspace(shape.weight_bytes)?,
            _input_transport: window.source.reserve_workspace(shape.input_bytes + 4)?,
            _readback_admission: window.source.reserve_workspace(shape.outputs * 12)?,
        });
        slot.begin_submission()?;
        let mut init = service.begin_initialization(correlation)?;
        init.retain_completion_owner(owner.clone())?;
        let weights = init.upload_bytes(window.owner.as_slice(), BufferRole::ImmutableWeights)?;
        let input = init.upload_bytes(&input.bytes, BufferRole::Activation)?;
        let zero = init.upload_bytes(&0u32.to_le_bytes(), BufferRole::ImmutableWeights)?;
        let accumulation = init.allocate_empty(shape.outputs * 4, BufferRole::Output)?;
        let rounded = init.allocate_empty(shape.outputs * 4, BufferRole::Output)?;
        let status = init.allocate_empty(shape.outputs * 4, BufferRole::Scratch)?;
        let initialized = init.complete_sync().await?;
        slot.finish()?;
        drop(initialized);
        cancel(cancelled)?;
        let prepared = Self {
            service,
            shape,
            identity: window.identity,
            owner,
            weights,
            input,
            zero,
            accumulation,
            rounded,
            status,
        };
        prepared.validate_bindings()?;
        Ok(prepared)
    }
    fn validate_bindings(&self) -> Result<()> {
        self.service.ensure_usable()?;
        self.service.pinned_cuda_queue()?;
        let expected = [
            self.shape.weight_bytes,
            self.shape.input_bytes,
            4,
            self.shape.outputs * 4,
            self.shape.outputs * 4,
            self.shape.outputs * 4,
        ];
        let leases = [
            &self.weights,
            &self.input,
            &self.zero,
            &self.accumulation,
            &self.rounded,
            &self.status,
        ];
        let mut ids = std::collections::BTreeSet::new();
        for (lease, bytes) in leases.into_iter().zip(expected) {
            let handle = lease.handle();
            if !Arc::ptr_eq(lease.service(), &self.service)
                || lease.source_identity() != self.identity.descriptor.source
                || lease.service_generation() != self.service.generation()
                || lease.bytes() != bytes
                || handle.service != self.service.client().service_id()
                || !ids.insert(lease.id())
                || !handle.offset_start.unwrap_or(0).is_multiple_of(4)
                || handle.size_in_used() < bytes as u64
            {
                return Err(Error::State(
                    "BF16 original lease/service/generation/size identity".into(),
                ));
            }
        }
        if [&self.weights, &self.input, &self.zero]
            .iter()
            .any(|lease| !lease.is_initialized())
        {
            return Err(Error::State(
                "BF16 input/source not physically initialized".into(),
            ));
        }
        Ok(())
    }
    fn launch(&self, correlation: u64) -> Result<ResidentSubmission<'_>> {
        self.validate_bindings()?;
        let accesses = [
            (self.weights.clone(), DeviceAccess::Read),
            (self.input.clone(), DeviceAccess::Read),
            (self.zero.clone(), DeviceAccess::Read),
            (self.accumulation.clone(), DeviceAccess::Write),
            (self.rounded.clone(), DeviceAccess::Write),
            (self.status.clone(), DeviceAccess::Write),
        ];
        let mut flight = self.service.submit(correlation, &accesses)?;
        flight.retain_completion_owner(self.owner.clone())?;
        let arg = |lease: &ResidentLease, n: usize| -> BufferArg {
            // SAFETY: exact owning leases are already pinned by the armed
            // original submission; cached whole allocations passed preflight.
            unsafe { BufferArg::from_raw_parts(lease.handle(), n) }
        };
        // SAFETY: one64-lane cube per full [request,output row], all shared
        // memory barriers reached by every lane; no padded request or writer.
        unsafe {
            project_cpu_bf16::launch_unchecked(
                self.service.client(),
                CubeCount::Static(self.shape.outputs as u32, 1, 1),
                CubeDim::new_1d(64),
                arg(&self.weights, self.shape.weight_bytes / 4),
                arg(&self.input, self.shape.input_bytes / 4),
                arg(&self.zero, 1),
                arg(&self.accumulation, self.shape.outputs),
                arg(&self.rounded, self.shape.outputs),
                arg(&self.status, self.shape.outputs),
                self.shape.rows,
                self.shape.columns,
            );
        }
        Ok(flight)
    }
    /// Consume the workspace: prior Ready outputs cannot be overwritten by a
    /// later request. No model/session/state publication is performed here.
    pub(crate) async fn execute(
        self,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ProjectionReadback> {
        cancel(cancelled)?;
        self.validate_bindings()?;
        let ticket = self.service.queue().submit(
            self.service.budget(),
            self.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: self.shape.owned_bytes()? as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        cancel(cancelled)?;
        slot.begin_submission()?;
        let flight = self.launch(correlation)?; // ONE actual native B1/2/4/8 launch.
        let outputs = [
            self.accumulation.clone(),
            self.rounded.clone(),
            self.status.clone(),
        ];
        let reads = [
            self.accumulation.readback(0, self.shape.outputs * 4)?,
            self.rounded.readback(0, self.shape.outputs * 4)?,
            self.status.readback(0, self.shape.outputs * 4)?,
        ];
        let (ready, raw) = flight
            .complete_with_readback(&outputs, &reads, |raw| {
                if raw.len() != 3
                    || raw
                        .iter()
                        .any(|value| value.len() != self.shape.outputs * 4)
                    || raw[0]
                        .chunks_exact(4)
                        .any(|b| !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
                    || raw[1].chunks_exact(4).any(|b| {
                        !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite()
                            || u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0xffff != 0
                    })
                    || raw[2]
                        .chunks_exact(4)
                        .any(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) != 0)
                {
                    return Err(Error::State(
                        "BF16 physical output/status certification failed".into(),
                    ));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        cancel(cancelled)?;
        Ok(ProjectionReadback {
            identity: self.identity.clone(),
            batch: self.shape.batch,
            raw,
            ready,
            _owner: self.owner.clone(),
            kernel_dispatches: 1,
        })
    }
    /// Separate process-only physical abort probe. No Ready is fabricated;
    /// real submitted original work and every source/lease are quarantined.
    pub(crate) async fn abandon_after_launch(self, correlation: u64) -> Result<()> {
        self.validate_bindings()?;
        let ticket = self.service.queue().submit(
            self.service.budget(),
            self.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: self.shape.owned_bytes()? as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let flight = self.launch(correlation)?;
        drop(flight);
        drop(slot);
        if self.service.ensure_usable().is_ok() {
            return Err(Error::State(
                "abandoned BF16 submission did not fence original card".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) struct ProjectionReadback {
    pub(crate) identity: Identity,
    pub(crate) batch: usize,
    pub(crate) kernel_dispatches: usize,
    raw: Vec<Bytes>,
    ready: ResidentReady,
    _owner: Arc<SourceOwner>,
}
impl ProjectionReadback {
    pub(crate) fn accumulation(&self) -> &[u8] {
        &self.raw[0]
    }
    pub(crate) fn rounded(&self) -> &[u8] {
        &self.raw[1]
    }
    pub(crate) fn ready_outputs(&self) -> usize {
        self.ready.outputs().len()
    }
    pub(crate) fn completion_events(&self) -> usize {
        self.ready.event_ids().len()
    }
}
