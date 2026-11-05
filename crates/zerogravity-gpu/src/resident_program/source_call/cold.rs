//! Authentic source preparation and physical accounting, before runtime calls.
use super::*;
use crate::resident_leases::{
    ResidentInitialization, ResidentLease, ResidentService, resident_matrix::PendingResidentMatrix,
    source_page,
};
use crate::resident_program::{
    DeviceBuffer,
    bf16_projection::{Bf16ProjectionStorage, PreparedBf16Projection},
};
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

/// Cold/configured-call working set, not a live reservation. Retained results
/// and extra snapshot readers require their own original host admission.
#[derive(Clone, Debug)]
pub struct SourceCallInventory {
    pub resident_weight_bytes: usize,
    pub workspace_bytes: usize,
    pub root_bytes: usize,
    pub root_count: usize,
    pub model_device_upper_bytes: usize,
    pub physical_device_upper_bytes: usize,
    pub context_device_bytes: usize,
    pub host_upper_bytes: usize,
}

pub(crate) struct Admission {
    pub(super) source: VerifiedTensorSource,
    pub(super) layout: Arc<layout::Layout>,
    pub(super) operands: BTreeMap<String, Arc<PreparedBf16Projection>>,
    pub(super) inventory: SourceCallInventory,
    metadata: WindowWorkspace,
}
impl Admission {
    pub(crate) fn new(
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
    ) -> Result<Self> {
        if !(1..=64).contains(&max_sessions) || source.source_identity() != plan.source() {
            return Err(Error::State(
                "source-call original source/session population".into(),
            ));
        }
        let metadata = source.reserve_workspace(8 * 1024 * 1024)?;
        let layout = Arc::new(layout::Layout::derive(plan)?);
        let mut operands = BTreeMap::new();
        for d in layout.matrices.values() {
            let m = source.matrix(&d.name, d.axes)?;
            if m.source_identity() != d.source
                || m.rows() != d.rows
                || m.columns() != d.columns
                || m.tensor().kind != d.kind
            {
                return Err(Error::State(
                    "source-call authenticated matrix differs".into(),
                ));
            }
            operands.insert(
                d.name.clone(),
                PreparedBf16Projection::new(source.clone(), m, Bf16ProjectionStorage::Resident)?,
            );
        }
        for p in layout.parameters.values() {
            source.bind_declared_parameter(p)?;
        }
        let root_count = 2 * max_sessions + 2;
        let root_bytes = layout.preflight.prior_state_bytes;
        let (native_host, native_device) = crate::hip::native::source_preparation_upper_bytes();
        let root_planes = layout
            .preflight
            .states
            .iter()
            .map(|s| s.layout.planes().map(|p| p.0))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum();
        let root_metadata = crate::resident_leases::backing_pool::ResidentPool::metadata_bytes(
            root_planes,
            root_count,
        )?;
        let host_upper_bytes = layout::sum([
            usize::try_from(source.metadata_reserved_bytes())
                .map_err(|_| Error::Limit("source-call source metadata extent".into()))?,
            source.staging_chunk_bytes() * 3,
            32 * 1024 * 1024,
            (max_sessions + 1) * 32 * 1024 * 1024,
            (max_sessions + 2) * 64 * 1024,
            // One initializing root (3x) plus one admitted snapshot read (4x).
            7 * root_bytes,
            root_metadata,
            source_page::transport_host_bytes(&source)?,
            3 * layout.preflight.parameter_bytes,
            native_host,
            2 * 1024 * 1024 * 1024,
        ])?;
        if host_upper_bytes as u64 > source.workspace_limit_bytes()? {
            return Err(Error::Limit("source-call complete host preflight".into()));
        }
        let model_device_upper_bytes = layout::sum([
            layout.preflight.owned_bytes()?,
            root_bytes * root_count,
            native_device,
        ])?;
        let (context_device_bytes, physical_device_upper_bytes) =
            crate::hip::native::resident_source_device_inventory(model_device_upper_bytes)?;
        let inventory = SourceCallInventory {
            resident_weight_bytes: layout.preflight.source_bytes,
            workspace_bytes: layout.preflight.owned_bytes()? - layout.preflight.source_bytes,
            root_bytes,
            root_count,
            model_device_upper_bytes,
            physical_device_upper_bytes,
            context_device_bytes,
            host_upper_bytes,
        };
        Ok(Self {
            source,
            layout,
            operands,
            inventory,
            metadata,
        })
    }
    pub(crate) fn inventory(&self) -> &SourceCallInventory {
        &self.inventory
    }
    pub(crate) fn catalogue(
        &self,
        client: &cubecl::client::Client,
    ) -> Result<(
        Vec<Box<dyn cubecl_server::kernel::CubeKernel>>,
        WindowWorkspace,
    )> {
        Ok((
            prepared::kernels(client, &self.layout, &self.operands)?,
            self.source.reserve_workspace(1024 * 1024)?,
        ))
    }
    pub(crate) async fn prepare(
        self,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        native: Arc<crate::hip::native::PreparedSourceNative>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Pending> {
        init.require_service(&service)?;
        native.require_live(service.client(), self.source.source_identity())?;
        let (native_host, native_device) = crate::hip::native::source_preparation_upper_bytes();
        if native.host_bytes() > native_host || native.device_bytes() > native_device {
            return Err(Error::Limit(
                "source-call actual native inventory exceeds admission".into(),
            ));
        }
        if service.source_identity() != self.source.source_identity()
            || service.client().is_capturing()
            || self.inventory.model_device_upper_bytes > service.capacity()
        {
            return Err(Error::Limit(
                "source-call original physical admission".into(),
            ));
        }
        init.retain_completion_owner(native.clone())?;
        let mut pending = Vec::new();
        for operand in self.operands.values() {
            operand.require_capability(&service)?;
            pending.push((operand.clone(), operand.upload(init, cancelled).await?));
        }
        let page = source_page::ResidentSourcePage::allocate_observed(
            self.source.clone(),
            service.clone(),
            init,
            native.source_page_actor(&service),
        )?;
        let parameters = crate::resident_leases::source_parameters::upload_parameters(
            &self.source,
            &self.layout.parameters,
            init,
        )?;
        let slots = self
            .layout
            .slots
            .iter()
            .map(|n| init.allocate_empty(*n, BufferRole::Activation))
            .collect::<Result<Vec<_>>>()?;
        // Projection scratch is an activation producer: its original lease is
        // retained and reused in queue order, never exposed as a source tensor.
        let scratch = self
            .layout
            .scratch
            .iter()
            .enumerate()
            .map(|(i, n)| {
                init.allocate_empty(
                    *n,
                    if i == layout::PROJECT {
                        BufferRole::Activation
                    } else {
                        BufferRole::Scratch
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let tables = super::tables()
            .into_iter()
            .map(|mut b| {
                b.resize(layout::aligned(b.len())?, 0);
                init.upload_bytes(&b, BufferRole::ImmutableWeights)
            })
            .collect::<Result<Vec<_>>>()?;
        let packed = init.allocate_empty(
            layout::aligned(layout::MAX_ROWS * self.layout.max_columns * 2)?,
            BufferRole::Scratch,
        )?;
        let accumulation = init.allocate_empty(
            layout::aligned(self.layout.projection_scratch_words * 4)?,
            BufferRole::Scratch,
        )?;
        let flags = init.allocate_empty(
            layout::aligned(self.layout.projection_scratch_words * 4)?,
            BufferRole::Scratch,
        )?;
        let status = init.allocate_empty(layout::ALIGNMENT, BufferRole::Scratch)?;
        let zero = init.upload_zeroed(layout::ALIGNMENT, BufferRole::ImmutableWeights)?;
        let mut accesses = Vec::new();
        accesses.extend(
            slots
                .iter()
                .chain(&scratch)
                .chain([&packed, &accumulation, &flags, &status, page.lease()])
                .cloned()
                .map(|l| (l, DeviceAccess::Write)),
        );
        accesses.extend(
            parameters
                .values()
                .chain(&tables)
                .chain([&zero])
                .cloned()
                .map(|l| (l, DeviceAccess::Read)),
        );
        for (_, p) in &pending {
            accesses.push((
                p.as_ref()
                    .ok_or_else(|| {
                        Error::State("source-call resident matrix upload absent".into())
                    })?
                    .lease()
                    .clone(),
                DeviceAccess::Read,
            ));
        }
        let owned = layout::sum(accesses.iter().map(|(l, _)| l.bytes()))?;
        if owned != self.layout.preflight.owned_bytes()? {
            return Err(Error::State(
                "source-call managed bytes differ from cold admission".into(),
            ));
        }
        let values = self
            .layout
            .plan
            .values()
            .iter()
            .zip(&self.layout.bindings)
            .map(|(v, b)| {
                Ok(DeviceBuffer::new(
                    view(&slots[b.slot], 0, b.bytes)?,
                    v.capacity_elements()?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let frame = crate::resident_leases::frame_checkpoint::ResidentFrameCheckpoint::new(
            self.source.clone(),
        )?;
        let cold = Arc::new(Cold {
            frame,
            source: self.source,
            layout: self.layout,
            service,
            native,
            slots,
            scratch,
            parameters,
            tables,
            operands: self.operands,
            page,
            packed,
            accumulation,
            flags,
            status,
            zero,
            values,
            accesses,
            _metadata: self.metadata,
        });
        init.retain_completion_owner(cold.clone())?;
        Ok(Pending {
            cold,
            matrices: pending,
        })
    }
}

pub(in crate::resident_program) struct Cold {
    pub(super) frame: Arc<crate::resident_leases::frame_checkpoint::ResidentFrameCheckpoint>,
    pub(super) source: VerifiedTensorSource,
    pub(super) layout: Arc<layout::Layout>,
    pub(super) service: Arc<ResidentService>,
    pub(super) native: Arc<crate::hip::native::PreparedSourceNative>,
    pub(super) slots: Vec<ResidentLease>,
    pub(super) scratch: Vec<ResidentLease>,
    pub(super) parameters: BTreeMap<String, ResidentLease>,
    pub(super) tables: Vec<ResidentLease>,
    pub(super) operands: BTreeMap<String, Arc<PreparedBf16Projection>>,
    pub(super) page: Arc<source_page::ResidentSourcePage>,
    pub(super) packed: ResidentLease,
    pub(super) accumulation: ResidentLease,
    pub(super) flags: ResidentLease,
    pub(super) status: ResidentLease,
    pub(super) zero: ResidentLease,
    pub(super) values: Vec<DeviceBuffer>,
    pub(super) accesses: Vec<(ResidentLease, DeviceAccess)>,
    _metadata: WindowWorkspace,
}
pub(crate) struct Pending {
    cold: Arc<Cold>,
    matrices: Vec<(Arc<PreparedBf16Projection>, Option<PendingResidentMatrix>)>,
}
impl Pending {
    pub(super) fn bind(self) -> Result<Arc<Cold>> {
        for (p, m) in self.matrices {
            p.bind_initialized(m)?;
        }
        self.cold.native.require_live(
            self.cold.service.client(),
            self.cold.source.source_identity(),
        )?;
        Ok(self.cold)
    }
}
pub(super) fn view(
    lease: &ResidentLease,
    offset: usize,
    bytes: usize,
) -> Result<cubecl::server::Handle> {
    if bytes == 0
        || offset
            .checked_add(bytes)
            .is_none_or(|end| end > lease.bytes())
    {
        return Err(Error::State("source-call exact original lease view".into()));
    }
    Ok(lease
        .handle()
        .offset_start(offset as u64)
        .offset_end((lease.bytes() - offset - bytes) as u64))
}

#[cfg(feature = "hardware-abort-probe")]
impl Cold {
    pub(in crate::resident_program) fn native_probe(&self) -> crate::hip::native::FinalNativeProbe {
        self.native.final_owner_probe()
    }
}
