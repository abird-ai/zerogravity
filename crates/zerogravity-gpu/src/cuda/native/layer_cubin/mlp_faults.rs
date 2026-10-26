//! Closed connected faults; SAME prepared graph and producer completion owners.
use super::super::activation_cubin::resident::ActivationRetentionWitness;
use super::super::mlp_cubin::resident::MlpRetentionWitness;
use super::super::projection_cubin::resident::NativeRetentionWitness;
use super::*;
use crate::cuda::rms::{FusedRmsRetentionState, FusedRmsRetentionWitness};
use crate::physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for};
use std::sync::Weak;
use zerogravity_core::resource_arbiter::{DeviceBudgetSnapshot, PhysicalDeviceKey};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerMlpCubinFault {
    AdmissionPressure,
    CancelBeforeDown,
    UnknownAfterMlp,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LayerMlpCubinFaultOutcome {
    pub status: &'static str,
    pub submitted_calls: usize,
    pub rms_stages: usize,
    pub gdn_launches: usize,
    pub postnorm_stages: usize,
    pub native_launches: usize,
    pub mlp_projection_calls: usize,
    pub activation_launches: usize,
    pub workspace_clears: usize,
    pub compared_words: usize,
    pub ready_completions: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub source_windows: usize,
    pub prepared_calls: usize,
    pub prepared_mlp_buffers: usize,
    pub retained_call_owners: usize,
    pub active_mlp_buffers_retained: usize,
    pub source_owner_retained: bool,
    pub original_accesses_retained: bool,
    pub original_readbacks_retained: bool,
    pub original_fused_source_retained: bool,
    pub image_owners_retained: usize,
    pub kernel_owners_retained: usize,
    pub native_charge_bytes: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub host_final_zero_established: bool,
}

/// Same exact five fixtures/images/preparation and first actual producer flight.
/// Uncertain completion never synchronizes, compares, retires or creates Ready.
pub async fn qualify_layer_mlp_cubin_fault(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [LayerFixture; 5],
    inputs: LayerMlpCubinInputs,
    fault: LayerMlpCubinFault,
    cancelled: Option<&AtomicBool>,
) -> Result<LayerMlpCubinFaultOutcome> {
    match qualify_layer_inner_with_activation(
        device,
        source,
        LayerFixtures::FivePhasesMlp(fixtures),
        inputs.projections,
        Some(inputs.activation),
        Mode::MlpFault(fault),
        cancelled,
    )
    .await?
    {
        Outcome::MlpFault(result) => Ok(*result),
        _ => Err(Error::State(
            "connected MLP fault returned another outcome".into(),
        )),
    }
}

pub(super) fn validate_mode(scope: LayerScope, mode: Mode) -> Result<()> {
    if matches!(mode, Mode::MlpFault(_)) && !scope.mlp()
        || matches!(mode, Mode::Fault(_)) && scope.mlp()
    {
        return Err(Error::Unsupported(
            "distinct connected MLP fault scope".into(),
        ));
    }
    Ok(())
}

fn envelope(snapshot: &DeviceBudgetSnapshot) -> bool {
    snapshot.reserved == MLP_PROVIDER_CAP as u64
        && snapshot.reservations == 1
        && snapshot.per_class.len() == 1
        && snapshot.per_class.get(&AllocationClass::ProviderEnvelope)
            == Some(&(MLP_PROVIDER_CAP as u64))
}

pub(super) struct PressureWitness {
    key: PhysicalDeviceKey,
    owner: Weak<LayerOwner>,
    service: Weak<ResidentService>,
}
pub(super) fn admission_pressure(
    owner: &Arc<LayerOwner>,
    service: &Arc<ResidentService>,
) -> Result<PressureWitness> {
    let key = service.physical_key();
    let before = ResourceArbiter::process_shared().snapshot(&key)?;
    if !owner.fixtures.scope().mlp()
        || service.capacity() != MLP_PROVIDER_CAP
        || !envelope(&before)
        || before.fenced
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "connected MLP pressure requires fresh 144MiB envelope".into(),
        ));
    }
    if !matches!(
        service.allocate_uninitialized(MLP_PROVIDER_CAP + 1, BufferRole::Scratch),
        Err(Error::Limit(_))
    ) || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "connected MLP pressure did not refuse before H2D".into(),
        ));
    }
    service.ensure_usable()?;
    Ok(PressureWitness {
        key,
        owner: Arc::downgrade(owner),
        service: Arc::downgrade(service),
    })
}
impl PressureWitness {
    pub(super) fn inspect(self) -> Result<LayerMlpCubinFaultOutcome> {
        let snapshot = ResourceArbiter::process_shared().snapshot(&self.key)?;
        if self.owner.upgrade().is_some()
            || self.service.upgrade().is_some()
            || snapshot.reserved != 0
            || snapshot.reservations != 0
            || !snapshot.per_class.is_empty()
            || snapshot.fenced
        {
            return Err(Error::State(
                "connected MLP pressure envelope did not retire".into(),
            ));
        }
        Ok(LayerMlpCubinFaultOutcome {
            status: "ADMISSION_REFUSED_BEFORE_H2D",
            ..Default::default()
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SubmittedStages {
    pub(super) rms: usize,
    pub(super) native: usize,
    pub(super) gdn: usize,
    pub(super) postnorm: usize,
    pub(super) mlp_calls: usize,
    pub(super) activation: usize,
    pub(super) clears: usize,
}
fn validate_stages(fault: LayerMlpCubinFault, stages: SubmittedStages) -> Result<()> {
    let (native, mlp_calls) = match fault {
        LayerMlpCubinFault::CancelBeforeDown => (6, 1),
        LayerMlpCubinFault::UnknownAfterMlp => (8, 2),
        LayerMlpCubinFault::AdmissionPressure => {
            return Err(Error::State("pressure cannot submit MLP stages".into()));
        }
    };
    if stages
        != (SubmittedStages {
            rms: 10,
            native,
            gdn: 12,
            postnorm: 10,
            mlp_calls,
            activation: 1,
            clears: 0,
        })
    {
        return Err(Error::State(
            "connected MLP exact first-call stage boundary".into(),
        ));
    }
    Ok(())
}
fn validate_native(images: [usize; 3], records: [usize; 3], charges: [usize; 3]) -> Result<()> {
    if images != [6, 7, 7] || records != [11, 7, 3] || charges != [FOREIGN_BYTES; 3] {
        return Err(Error::State(
            "connected MLP original seven images/21 records/one admission".into(),
        ));
    }
    Ok(())
}
fn validate_readbacks(layout: &[(usize, usize)]) -> Result<()> {
    if layout.len() != 33
        || layout[..7]
            != [
                (0, 2048),
                (0, 2048),
                (0, 2048),
                (0, 2048),
                (0, 4096),
                (0, 2048),
                (0, 2048),
            ]
        || layout.iter().any(|(_, bytes)| *bytes == 0)
    {
        return Err(Error::State(
            "connected MLP original first-call readback cohort".into(),
        ));
    }
    Ok(())
}
type ReadbackIdentity = (BufferLeaseId, usize, usize);
fn readback_identity(read: &ResidentReadback) -> ReadbackIdentity {
    (read.lease_id(), read.offset(), read.bytes())
}

pub(super) struct UnknownWitness {
    fault: LayerMlpCubinFault,
    key: PhysicalDeviceKey,
    owner: Weak<LayerOwner>,
    call: Weak<Call>,
    calls: Vec<Weak<Call>>,
    rms: Weak<RmsPrepared>,
    gdn: Weak<GdnPrepared>,
    fused: Weak<FusedRmsPrepared>,
    fused_source: FusedRmsRetentionWitness,
    fused_state: FusedRmsRetentionState,
    programs: Weak<ProjectionPrograms>,
    mlp_programs: Weak<MlpPrograms>,
    activation_programs: Weak<ActivationPrograms>,
    native: NativeRetentionWitness,
    mlp_native: MlpRetentionWitness,
    activation_native: ActivationRetentionWitness,
    service: Weak<ResidentService>,
    accesses: Vec<(BufferLeaseId, DeviceAccess)>,
    readbacks: Vec<ReadbackIdentity>,
    source_windows: usize,
    resident_bytes: usize,
    uploaded_value_bytes: u64,
    active_mlp: [BufferLeaseId; 3],
}
impl UnknownWitness {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        fault: LayerMlpCubinFault,
        owner: &Arc<LayerOwner>,
        call: &Arc<Call>,
        calls: &[Arc<Call>],
        programs: &Arc<ProjectionPrograms>,
        mlp_programs: &Arc<MlpPrograms>,
        activation_programs: &Arc<ActivationPrograms>,
        weights: &[ResidentLease],
        parameters: &[ResidentLease],
        workspace: &ResidentLease,
        service: &Arc<ResidentService>,
    ) -> Result<Self> {
        if fault == LayerMlpCubinFault::AdmissionPressure
            || !owner.fixtures.scope().mlp()
            || calls.len() != 5
            || !Arc::ptr_eq(call, &calls[0])
            || call.phase != GdnPhase::InitialOne
            || call.snapshot.is_some()
            || calls.iter().any(|value| value.mlp.is_none())
        {
            return Err(Error::State(
                "connected MLP fault requires original first prepared call".into(),
            ));
        }
        let mlp = call
            .mlp
            .as_ref()
            .ok_or_else(|| Error::State("actual MLP buffers absent".into()))?;
        let fused = call
            .postnorm
            .as_ref()
            .ok_or_else(|| Error::State("actual fused source owner absent".into()))?;
        let fused_source = fused.retention_witness();
        let fused_state = fused_source.inspect()?;
        let projected = call
            .projected
            .as_ref()
            .ok_or_else(|| Error::State("actual output projection absent".into()))?;
        if weights.len() != 5
            || parameters.len() != 4
            || owner._windows.is_empty()
            || weights
                .iter()
                .zip(&owner.weights)
                .any(|(lease, raw)| lease.bytes() != raw.len())
            || parameters
                .iter()
                .zip(&owner.parameters)
                .any(|(lease, raw)| lease.bytes() != raw.len())
            || workspace.bytes() != WORKSPACE_BYTES
            || [
                mlp.gate_up.bytes(),
                mlp.activation.bytes(),
                mlp.output.bytes(),
            ] != [14336, 7168, 2048]
            || [
                mlp.gate_up.is_initialized(),
                mlp.activation.is_initialized(),
                mlp.output.is_initialized(),
            ] != [false; 3]
            || fused_state.source != owner._source.source_identity()
            || fused_state.source != service.source_identity()
            || fused_state.rows != 1
            || fused_state.input_ids != [projected.id(), call.rms.embedding().id()]
            || fused_state.output_initialized != [false; 3]
            || fused_state.host_workspace_bytes == 0
        {
            return Err(Error::State(
                "connected MLP original source/buffer/gamma bindings".into(),
            ));
        }
        let mut expected = weights
            .iter()
            .chain(parameters)
            .map(|value| (value, DeviceAccess::Read))
            .collect::<Vec<_>>();
        expected.extend([
            (&mlp.gate_up, DeviceAccess::Write),
            (&mlp.activation, DeviceAccess::Write),
            (&mlp.output, DeviceAccess::Write),
            (workspace, DeviceAccess::Write),
            (&call.cache, DeviceAccess::Write),
            (projected, DeviceAccess::Write),
            (fused.normalized(), DeviceAccess::Write),
        ]);
        for (value, mode) in expected {
            if !call.accesses.iter().any(|(lease, access)| {
                lease.id() == value.id()
                    && *access == mode
                    && Arc::ptr_eq(lease.service(), service)
                    && lease.source_identity() == fused_state.source
            }) {
                return Err(Error::State(
                    "connected MLP original full-flight access union".into(),
                ));
            }
        }
        let readbacks = call.reads.iter().map(readback_identity).collect::<Vec<_>>();
        validate_readbacks(
            &readbacks
                .iter()
                .map(|&(_, offset, bytes)| (offset, bytes))
                .collect::<Vec<_>>(),
        )?;
        let mut declared = Vec::from(call.rms.readbacks()?);
        declared.extend(fused.readbacks()?);
        for record in &owner.fixtures.as_slice()[0].expected {
            declared.push(call.readback(record.output)?);
        }
        if readbacks != declared.iter().map(readback_identity).collect::<Vec<_>>()
            || readbacks[3..7]
                .iter()
                .map(|(id, _, _)| *id)
                .collect::<Vec<_>>()
                != fused_state.readback_ids
        {
            return Err(Error::State(
                "connected MLP original readback lease identities".into(),
            ));
        }
        let native = programs.retention_witness()?;
        let mlp_native = mlp_programs.retention_witness()?;
        let activation_native = activation_programs.retention_witness()?;
        let ordinary = native.inspect()?;
        let mlp_state = mlp_native.inspect()?;
        let act_state = activation_native.inspect()?;
        validate_native(
            [
                ordinary.image_owners,
                mlp_state.image_owners,
                act_state.image_owners,
            ],
            [
                ordinary.kernel_owners,
                mlp_state.kernel_owners,
                act_state.kernel_owners,
            ],
            [
                ordinary.native_charge_bytes,
                mlp_state.native_charge_bytes,
                act_state.native_charge_bytes,
            ],
        )?;
        let key = service.physical_key();
        let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
        let resident_bytes = service.resident_bytes()?;
        if service.capacity() != MLP_PROVIDER_CAP
            || !envelope(&snapshot)
            || snapshot.fenced
            || resident_bytes <= FOREIGN_BYTES
            || resident_bytes > MLP_PROVIDER_CAP
            || service.pending_events()? != 0
            || service.uploaded_value_bytes() == 0
            || quarantined_handles_for(service.client())? != 0
            || quarantined_completion_owners_for(service.client())? != 0
        {
            return Err(Error::State(
                "connected MLP requires fresh actual initialized preparation".into(),
            ));
        }
        service.ensure_usable()?;
        Ok(Self {
            fault,
            key,
            owner: Arc::downgrade(owner),
            call: Arc::downgrade(call),
            calls: calls.iter().map(Arc::downgrade).collect(),
            rms: Arc::downgrade(&call.rms),
            gdn: Arc::downgrade(&call.gdn),
            fused: Arc::downgrade(fused),
            fused_source,
            fused_state,
            programs: Arc::downgrade(programs),
            mlp_programs: Arc::downgrade(mlp_programs),
            activation_programs: Arc::downgrade(activation_programs),
            native,
            mlp_native,
            activation_native,
            service: Arc::downgrade(service),
            accesses: call
                .accesses
                .iter()
                .map(|(lease, access)| (lease.id(), *access))
                .collect(),
            readbacks,
            source_windows: owner._windows.len(),
            resident_bytes,
            uploaded_value_bytes: service.uploaded_value_bytes(),
            active_mlp: [mlp.gate_up.id(), mlp.activation.id(), mlp.output.id()],
        })
    }

    /// Host-only observation after ALL caller-local strong references drop.
    /// Future Call owners need not survive: they are not first-flight operands.
    pub(super) fn inspect(self, stages: SubmittedStages) -> Result<LayerMlpCubinFaultOutcome> {
        validate_stages(self.fault, stages)?;
        let owner = self
            .owner
            .upgrade()
            .ok_or_else(|| Error::State("connected MLP source owner lost".into()))?;
        let call = self
            .call
            .upgrade()
            .ok_or_else(|| Error::State("connected MLP actual Call lost".into()))?;
        let service = self
            .service
            .upgrade()
            .ok_or_else(|| Error::State("connected MLP original service lost".into()))?;
        if self.rms.upgrade().is_none()
            || self.gdn.upgrade().is_none()
            || self.fused.upgrade().is_none()
            || self.programs.upgrade().is_none()
            || self.mlp_programs.upgrade().is_none()
            || self.activation_programs.upgrade().is_none()
        {
            return Err(Error::State(
                "connected MLP actual submitted completion owners lost".into(),
            ));
        }
        let ordinary = self.native.inspect()?;
        let mlp = self.mlp_native.inspect()?;
        let activation = self.activation_native.inspect()?;
        validate_native(
            [
                ordinary.image_owners,
                mlp.image_owners,
                activation.image_owners,
            ],
            [
                ordinary.kernel_owners,
                mlp.kernel_owners,
                activation.kernel_owners,
            ],
            [
                ordinary.native_charge_bytes,
                mlp.native_charge_bytes,
                activation.native_charge_bytes,
            ],
        )?;
        let active = call
            .mlp
            .as_ref()
            .ok_or_else(|| Error::State("connected MLP active payload lost".into()))?;
        let fused = self.fused_source.inspect()?;
        let snapshot = ResourceArbiter::process_shared().snapshot(&self.key)?;
        let handles = quarantined_handles_for(service.client())?;
        let owners = quarantined_completion_owners_for(service.client())?;
        if owner._source.source_identity() != self.fused_state.source
            || owner._windows.len() != self.source_windows
            || fused != self.fused_state
            || [
                active.gate_up.id(),
                active.activation.id(),
                active.output.id(),
            ] != self.active_mlp
            || [
                active.gate_up.is_initialized(),
                active.activation.is_initialized(),
                active.output.is_initialized(),
            ] != [false; 3]
            || call
                .accesses
                .iter()
                .map(|(lease, access)| (lease.id(), *access))
                .collect::<Vec<_>>()
                != self.accesses
            || call.reads.iter().map(readback_identity).collect::<Vec<_>>() != self.readbacks
            || ordinary.image_admission_bytes == 0
            || mlp.image_admission_bytes != ordinary.image_admission_bytes
            || activation.image_admission_bytes != ordinary.image_admission_bytes
            || service.ensure_usable().is_ok()
            || service.resident_bytes()? != self.resident_bytes
            || service.pending_events()? != 1
            || service.uploaded_value_bytes() != self.uploaded_value_bytes
            || handles == 0
            || owners < 8
            || !snapshot.fenced
            || !envelope(&snapshot)
        {
            return Err(Error::State(
                "connected MLP unknown original ownership/fence differs".into(),
            ));
        }
        Ok(LayerMlpCubinFaultOutcome {
            status: if self.fault == LayerMlpCubinFault::CancelBeforeDown {
                "CANCELLED_UNKNOWN_RETAINED"
            } else {
                "UNKNOWN_RETAINED"
            },
            submitted_calls: 1,
            rms_stages: stages.rms,
            gdn_launches: stages.gdn,
            postnorm_stages: stages.postnorm,
            native_launches: stages.native,
            mlp_projection_calls: stages.mlp_calls,
            activation_launches: stages.activation,
            workspace_clears: stages.clears,
            resident_bytes: self.resident_bytes,
            pending_events: 1,
            uploaded_value_bytes: self.uploaded_value_bytes,
            quarantined_handles: handles,
            quarantined_completion_owners: owners,
            source_windows: self.source_windows,
            prepared_calls: self.calls.len(),
            prepared_mlp_buffers: 3 * self.calls.len(),
            retained_call_owners: self
                .calls
                .iter()
                .filter(|call| call.upgrade().is_some())
                .count(),
            active_mlp_buffers_retained: 3,
            source_owner_retained: true,
            original_accesses_retained: true,
            original_readbacks_retained: true,
            original_fused_source_retained: true,
            image_owners_retained: 7,
            kernel_owners_retained: ordinary.kernel_owners
                + mlp.kernel_owners
                + activation.kernel_owners,
            native_charge_bytes: FOREIGN_BYTES,
            physical_reserved_bytes: snapshot.reserved,
            physical_fenced: snapshot.fenced,
            ..Default::default()
        })
    }
}

#[cfg(test)]
#[path = "mlp_faults/tests.rs"]
mod tests;
