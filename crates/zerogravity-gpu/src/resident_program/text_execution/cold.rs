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
use zerogravity_core::{
    accelerator::program::{PlanOperation, SourceOperatorKind, ValueStorageFormat},
    device_events::{BufferRole, DeviceAccess},
    model::ModelProgram,
    provider::{ExecutionMode, PreparedRegion, ProviderKind, ProviderRequirements, SelectedMode},
};

/// Cold/configured-call working set, not a live reservation. Retained results
/// and extra snapshot readers require their own original host admission.
#[derive(Clone, Debug)]
pub struct ResidentTextInventory {
    pub total_device_budget_bytes: usize,
    pub model_device_budget_bytes: usize,
    pub physical_admission_bytes: u64,
    pub source_weight_bytes: usize,
    pub resident_weight_bytes: usize,
    pub workspace_bytes: usize,
    pub root_bytes: usize,
    pub root_count: usize,
    pub state_reserve_bytes: usize,
    pub model_device_upper_bytes: usize,
    pub physical_device_upper_bytes: usize,
    pub context_device_bytes: usize,
    pub host_upper_bytes: usize,
}

pub(crate) struct Admission {
    pub(super) budget: crate::hip::native::HipSourceBudget,
    pub(super) native_policy: crate::hip::native::HipSourceNativePolicy,
    guarded_triangular: bool,
    pub(super) source: VerifiedTensorSource,
    pub(super) layout: Arc<layout::Layout>,
    pub(super) operands: BTreeMap<String, Arc<PreparedBf16Projection>>,
    pub(super) inventory: ResidentTextInventory,
    metadata: WindowWorkspace,
}
impl Admission {
    pub(crate) fn new(
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
    ) -> Result<Self> {
        Self::with_budget(
            source,
            plan,
            max_sessions,
            crate::hip::native::HipSourceBudget::default(),
        )
    }
    pub(crate) fn with_budget(
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        budget: crate::hip::native::HipSourceBudget,
    ) -> Result<Self> {
        Self::with_budget_and_native_policy(
            source,
            plan,
            max_sessions,
            budget,
            crate::hip::native::HipSourceNativePolicy::Fallback,
        )
    }
    pub(crate) fn with_budget_and_native_policy(
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        budget: crate::hip::native::HipSourceBudget,
        native_policy: crate::hip::native::HipSourceNativePolicy,
    ) -> Result<Self> {
        if !(1..=64).contains(&max_sessions) || source.source_identity() != plan.source() {
            return Err(Error::State(
                "text-execution original source/session population".into(),
            ));
        }
        let metadata = source.reserve_workspace(8 * 1024 * 1024)?;
        let mut layout = layout::Layout::derive(plan)?;
        let native_gemm = layout.plan.numerical_profile().uses_hip_native_dense();
        let requirements = if native_gemm {
            crate::hip::dense::Dense::REQUIREMENTS
        } else {
            ProviderRequirements::original_queue(0, 0)
        };
        // HIP uses widened F32 activation storage; the same logical region
        // contract must agree with its original SSA storage admission.
        let model = ModelProgram::from_recorded(source.clone(), layout.plan.clone())?;
        for (index, node) in layout.plan.nodes().iter().enumerate() {
            if matches!(&node.operation, PlanOperation::SourceOperatorTemplate { descriptor, .. }
                if matches!(descriptor.kind(), SourceOperatorKind::Projection { .. }))
            {
                let region = model.closed_region(index..index + 1)?;
                let max_rows = region.region().max_rows();
                let prepared = PreparedRegion::new(
                    region,
                    SelectedMode {
                        provider: ProviderKind::Hip,
                        mode: ExecutionMode::Native,
                    },
                    requirements,
                    1..=max_rows,
                    &[ValueStorageFormat::F32],
                    &[ValueStorageFormat::F32],
                )?;
                let capacity = prepared.capacity();
                for binding in capacity.inputs().chain(capacity.outputs()) {
                    let (port, physical) = binding?;
                    if layout.bindings[port.value().0 as usize].bytes != physical.byte_len()? {
                        return Err(Error::State(
                            "HIP projection region storage extent differs".into(),
                        ));
                    }
                }
            }
        }
        if native_gemm {
            if native_policy != crate::hip::native::HipSourceNativePolicy::Gfx1151 {
                return Err(Error::Unsupported(
                    "native rocBLAS profile requires qualified gfx1151".into(),
                ));
            }
            layout.preflight.workspace_bytes = layout::sum([
                layout.preflight.workspace_bytes,
                requirements.workspace_bytes,
            ])?;
        }
        let source_weight_bytes = layout.preflight.source_bytes;
        let root_count = 2 * max_sessions + 2;
        let root_bytes = layout.preflight.prior_state_bytes;
        let state_reserve_bytes = root_bytes
            .checked_mul(root_count)
            .ok_or_else(|| Error::Limit("text-execution root reserve extent".into()))?;
        let (_, native_device) =
            crate::hip::native::source_preparation_upper_bytes_for(native_policy);
        let fixed_device_bytes = layout::sum([
            layout.preflight.owned_bytes()? - source_weight_bytes,
            state_reserve_bytes,
            native_device,
        ])?;
        let storage = weight_storage(
            source_weight_bytes,
            fixed_device_bytes,
            budget.model_device_bytes(),
        )?;
        if storage == Bf16ProjectionStorage::Streamed {
            if native_gemm {
                return Err(Error::resource(
                    "model device memory / VRAM budget",
                    "native rocBLAS resident weights",
                    layout::sum([source_weight_bytes, fixed_device_bytes])? as u64,
                    budget.model_device_bytes() as u64,
                ));
            }
            // Immutable source extent is unchanged. Only the original bounded
            // source-page transport is resident; its bytes are already priced.
            layout.preflight.source_bytes = 0;
        }
        let layout = Arc::new(layout);
        let mut operands = BTreeMap::new();
        for d in layout.matrices.values() {
            let m = source.matrix(&d.name, d.axes)?;
            if m.source_identity() != d.source
                || m.rows() != d.rows
                || m.columns() != d.columns
                || m.tensor().kind != d.kind
            {
                return Err(Error::State(
                    "text-execution authenticated matrix differs".into(),
                ));
            }
            let operand = if storage == Bf16ProjectionStorage::Resident
                && native_policy.prefers_hoisted_projection()
            {
                PreparedBf16Projection::new_resident_hoisted(source.clone(), m)?
            } else {
                PreparedBf16Projection::new(source.clone(), m, storage)?
            };
            operands.insert(d.name.clone(), operand);
        }
        for p in layout.parameters.values() {
            source.bind_declared_parameter(p)?;
        }
        let (native_host, native_device) =
            crate::hip::native::source_preparation_upper_bytes_for(native_policy);
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
                .map_err(|_| Error::Limit("text-execution source metadata extent".into()))?,
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
        let host_limit = source.workspace_limit_bytes()?;
        if host_upper_bytes as u64 > host_limit {
            return Err(Error::resource(
                "host memory",
                "text-execution complete host preflight",
                host_upper_bytes as u64,
                host_limit,
            ));
        }
        let model_device_upper_bytes = layout::sum([
            layout.preflight.owned_bytes()?,
            state_reserve_bytes,
            native_device,
        ])?;
        let (context_device_bytes, physical_device_upper_bytes) =
            budget.inventory(model_device_upper_bytes)?;
        let inventory = ResidentTextInventory {
            total_device_budget_bytes: budget.total_device_bytes(),
            model_device_budget_bytes: budget.model_device_bytes(),
            physical_admission_bytes: budget.physical_admission_bytes(),
            source_weight_bytes,
            resident_weight_bytes: layout.preflight.source_bytes,
            workspace_bytes: layout.preflight.owned_bytes()? - layout.preflight.source_bytes,
            root_bytes,
            root_count,
            state_reserve_bytes,
            model_device_upper_bytes,
            physical_device_upper_bytes,
            context_device_bytes,
            host_upper_bytes,
        };
        Ok(Self {
            budget,
            native_policy,
            guarded_triangular: native_policy.prefers_guarded_triangular(),
            source,
            layout,
            operands,
            inventory,
            metadata,
        })
    }
    pub(crate) fn inventory(&self) -> &ResidentTextInventory {
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
            prepared::kernels(
                client,
                &self.layout,
                &self.operands,
                self.guarded_triangular,
            )?,
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
        let (native_host, native_device) =
            crate::hip::native::source_preparation_upper_bytes_for(self.native_policy);
        if native.host_bytes() > native_host || native.device_bytes() > native_device {
            return Err(Error::State(
                "text-execution actual native inventory exceeds its admitted upper bound".into(),
            ));
        }
        if service.source_identity() != self.source.source_identity()
            || service.client().is_capturing()
        {
            return Err(Error::State(
                "text-execution service/source/capture state differs".into(),
            ));
        }
        if self.inventory.model_device_upper_bytes > service.capacity() {
            return Err(Error::resource(
                "GPU provider memory / VRAM budget",
                "text-execution model/native/root admission",
                self.inventory.model_device_upper_bytes as u64,
                service.capacity() as u64,
            ));
        }
        init.retain_completion_owner(native.clone())?;
        let native_gemm = self.layout.plan.numerical_profile().uses_hip_native_dense();
        if native_gemm {
            native.prepare_rocblas(&service)?;
        }
        let mut pending = Vec::new();
        for operand in self.operands.values() {
            operand.require_capability(&service)?;
            pending.push((
                operand.clone(),
                operand
                    .upload_with_finiteness(init, cancelled, native_gemm)
                    .await?,
            ));
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
        let rocblas_workspace = native_gemm
            .then(|| {
                init.allocate_empty(
                    crate::hip::dense::Dense::REQUIREMENTS.workspace_bytes,
                    BufferRole::Scratch,
                )
            })
            .transpose()?;
        let mut accesses = Vec::new();
        accesses.extend(
            rocblas_workspace
                .iter()
                .cloned()
                .map(|l| (l, DeviceAccess::Write)),
        );
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
        for (operand, pending) in &pending {
            match (operand.storage(), pending) {
                (Bf16ProjectionStorage::Resident, Some(matrix)) => {
                    accesses.push((matrix.lease().clone(), DeviceAccess::Read));
                }
                (Bf16ProjectionStorage::Streamed, None) => {}
                _ => {
                    return Err(Error::State(
                        "text-execution selected weight owner differs".into(),
                    ));
                }
            }
        }
        let owned = layout::sum(accesses.iter().map(|(l, _)| l.bytes()))?;
        if owned != self.layout.preflight.owned_bytes()? {
            return Err(Error::State(
                "text-execution managed bytes differ from cold admission".into(),
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
            guarded_triangular: self.guarded_triangular,
            source: self.source,
            layout: self.layout,
            service,
            native,
            rocblas_workspace,
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
    pub(super) guarded_triangular: bool,
    pub(super) frame: Arc<crate::resident_leases::frame_checkpoint::ResidentFrameCheckpoint>,
    pub(super) source: VerifiedTensorSource,
    pub(super) layout: Arc<layout::Layout>,
    pub(super) service: Arc<ResidentService>,
    pub(super) native: Arc<crate::hip::native::PreparedSourceNative>,
    pub(super) rocblas_workspace: Option<ResidentLease>,
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
    lease.view(offset, bytes)
}

#[cfg(feature = "hardware-abort-probe")]
impl Cold {
    pub(in crate::resident_program) fn native_probe(&self) -> crate::hip::native::FinalNativeProbe {
        self.native.final_owner_probe()
    }
}

fn weight_storage(weights: usize, fixed: usize, limit: usize) -> Result<Bf16ProjectionStorage> {
    let available = limit.checked_sub(fixed).ok_or_else(|| {
        Error::resource(
            "model device memory / VRAM budget",
            "text-execution fixed workspace/native/root admission",
            fixed as u64,
            limit as u64,
        )
    })?;
    if weights == 0 {
        return Err(Error::State(
            "text-execution original weight extent absent".into(),
        ));
    }
    Ok(if weights <= available {
        Bf16ProjectionStorage::Resident
    } else {
        Bf16ProjectionStorage::Streamed
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_total_prices_context_native_and_roots_before_weight_placement() {
        use crate::hip::native::HipSourceBudget;
        const GIB: usize = 1024 * 1024 * 1024;
        let (weights, fixed) = (50 * GIB, 12 * GIB);
        assert_eq!(
            weight_storage(
                weights,
                fixed,
                HipSourceBudget::default().model_device_bytes()
            )
            .unwrap(),
            Bf16ProjectionStorage::Streamed
        );
        let exact = (weights + fixed + 4 * GIB) as u64;
        for total in [exact, 75 * GIB as u64] {
            let budget = HipSourceBudget::new(total).unwrap();
            assert_eq!(
                weight_storage(weights, fixed, budget.model_device_bytes()).unwrap(),
                Bf16ProjectionStorage::Resident
            );
            assert_eq!(budget.inventory(weights + fixed).unwrap().1 as u64, exact);
        }
        let short = HipSourceBudget::new(exact - 1).unwrap();
        assert_eq!(
            weight_storage(weights, fixed, short.model_device_bytes()).unwrap(),
            Bf16ProjectionStorage::Streamed
        );
        let floor = HipSourceBudget::new((fixed + 4 * GIB - 1) as u64).unwrap();
        assert!(weight_storage(weights, fixed, floor.model_device_bytes()).is_err());
    }
    #[test]
    fn storage_selection_preserves_actual_caps_and_never_omits_fixed_owners() {
        assert_eq!(
            weight_storage(8, 16, 24).unwrap(),
            Bf16ProjectionStorage::Resident
        );
        assert_eq!(
            weight_storage(9, 16, 24).unwrap(),
            Bf16ProjectionStorage::Streamed
        );
        assert_eq!(
            weight_storage(usize::MAX, 24, 24).unwrap(),
            Bf16ProjectionStorage::Streamed
        );
        assert!(weight_storage(1, 25, 24).is_err());
        assert!(weight_storage(0, 0, 24).is_err());
    }
}
