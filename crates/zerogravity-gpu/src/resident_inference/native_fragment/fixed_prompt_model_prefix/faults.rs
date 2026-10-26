//! Cuts in the SAME first3 prepared graph and original full-model traversal.
use super::*;
use crate::cuda::native::fixed_prompt_model_native::FixedPromptNativeRetention;
use std::sync::{Mutex as StdMutex, Weak};
use zerogravity_artifacts::source_weights::SourceOwnerWitness;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::device_events::BufferLeaseId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedPromptModelPrefixFault {
    AdmissionPressure,
    UnknownAfterConvolution,
    CancelBeforeRecurrent,
    UnknownAfterQkvMain,
    CancelBeforeQkvReducer,
    UnknownAfterQk,
    CancelBeforeKvAppend,
    UnknownAfterKvAppend,
    CancelBeforeKvTransport,
    UnknownAfterFa,
    CancelBeforeFa,
    UnknownAfterSigmoid,
    CancelBeforeMultiply,
    UnknownAfterAttentionOut,
    CancelBeforeActivation,
    UnknownAfterLayer3DownMain,
    CancelBeforeLayer3DownReducer,
}

#[derive(Clone, Debug)]
pub struct FixedPromptModelPrefixFaultOutcome {
    pub fault: FixedPromptModelPrefixFault,
    pub status: &'static str,
    pub ready_completions: usize,
    pub readback_commands: usize,
    pub expected_callbacks: usize,
    pub completed_prefixes: usize,
    pub native_launches: usize,
    pub convolution_dispatches: usize,
    pub qkv_main_reached: bool,
    pub qkv_native_enqueues: usize,
    pub qkv_output_unpromoted: bool,
    pub qk_native_enqueues: usize,
    pub kv_native_enqueues: usize,
    pub kv_transport_enqueues: usize,
    pub kv_candidate_unpromoted: bool,
    pub fa_native_enqueues: usize,
    pub fa_output_unpromoted: bool,
    pub tail_native_enqueues: [usize; 6],
    pub tail_outputs_unpromoted: bool,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub original_root: Option<RootBinding>,
    pub prepared_state_buffers: usize,
    pub retained_plan_values: usize,
    pub retained_source_weight_groups: usize,
    pub prepared_graph_leases: usize,
    pub submitted_current_leases: usize,
    pub readback_descriptors: usize,
    pub uninitialized_prepared_leases: usize,
    pub source_retained: bool,
    pub all_prepared_owners_retained: bool,
    pub all_readback_descriptors_retained: bool,
    pub all_future_uninitialized: bool,
    pub native_image_owners: usize,
    pub native_kernel_owners: usize,
    pub native_charge_bytes: usize,
    pub image_admission_bytes: usize,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub model_root_published: bool,
    pub parent_model_accepted: bool,
}

/// Flight-only ownership; images/native/body never point back here.
pub(super) struct PreparedOwners {
    pub(super) cold: Arc<ColdOwners>,
    pub(super) cut: Arc<FixedPromptPrefixCut>,
    pub(super) native: Arc<FixedPromptModelNativePrograms>,
    pub(super) cohort: Arc<Vec<ResidentReadback>>,
    pub(super) _metadata: WindowWorkspace,
}

type Descriptor = (BufferLeaseId, usize, usize);

#[derive(Clone, Debug, PartialEq, Eq)]
struct Snapshot {
    leases: BTreeMap<BufferLeaseId, usize>,
    descriptors: Vec<Descriptor>,
    uninitialized: BTreeSet<BufferLeaseId>,
    future: BTreeSet<BufferLeaseId>,
    state_buffers: usize,
    values: usize,
    weights: usize,
    native: FixedPromptNativeRetention,
    resident: usize,
    uploaded: u64,
}

impl Snapshot {
    fn require_same(&self, actual: &Self) -> Result<()> {
        if self != actual {
            return Err(Error::State("canonical47 first3 retained graph/descriptor/initialization/native metadata changed".into()));
        }
        Ok(())
    }
}

fn require_future_uninitialized(
    future: &BTreeSet<BufferLeaseId>,
    uninitialized: &BTreeSet<BufferLeaseId>,
) -> Result<()> {
    if future.is_empty() || !future.is_subset(uninitialized) {
        return Err(Error::State(
            "canonical47 unknown future value/candidate was initialized".into(),
        ));
    }
    Ok(())
}

fn require_partial_qkv_witness(
    fault: FixedPromptModelPrefixFault,
    qkv_cut: bool,
    enqueues: usize,
    output_unpromoted: bool,
) -> Result<()> {
    if matches!(
        fault,
        FixedPromptModelPrefixFault::UnknownAfterQkvMain
            | FixedPromptModelPrefixFault::CancelBeforeQkvReducer
    ) && (!qkv_cut || enqueues != 1 || !output_unpromoted)
    {
        return Err(Error::State(
            "canonical47 actual one-main/no-reducer/unpromoted-Y witness absent".into(),
        ));
    }
    Ok(())
}

fn require_partial_kv_witness(
    fault: FixedPromptModelPrefixFault,
    cut: bool,
    counts: (usize, usize, usize),
    qk_unpromoted: bool,
    kv_unpromoted: bool,
) -> Result<()> {
    use FixedPromptModelPrefixFault as F;
    let expected_kv = match fault {
        F::UnknownAfterQk | F::CancelBeforeKvAppend => 0,
        F::UnknownAfterKvAppend | F::CancelBeforeKvTransport => 1,
        _ => return Ok(()),
    };
    if !cut || counts != (1, expected_kv, 0) || !qk_unpromoted || !kv_unpromoted {
        return Err(Error::State(
            "canonical47 actual QK/KV partial cut/no-transport/unpromoted-output witness".into(),
        ));
    }
    Ok(())
}

fn require_partial_fa_witness(
    fault: FixedPromptModelPrefixFault,
    cut: bool,
    counts: (usize, usize, usize),
    fa: usize,
    output_unpromoted: bool,
) -> Result<()> {
    use FixedPromptModelPrefixFault as F;
    let expected = match fault {
        F::UnknownAfterFa => 1,
        F::CancelBeforeFa => 0,
        _ => return Ok(()),
    };
    if !cut || counts != (1, 1, 1) || fa != expected || !output_unpromoted {
        return Err(Error::State(
            "canonical47 actual FA cut/unpromoted O/LSE witness absent".into(),
        ));
    }
    Ok(())
}

fn expected_tail_counts(fault: FixedPromptModelPrefixFault) -> Option<[usize; 6]> {
    use FixedPromptModelPrefixFault as F;
    Some(match fault {
        F::UnknownAfterSigmoid | F::CancelBeforeMultiply => [1, 0, 0, 0, 0, 0],
        F::UnknownAfterAttentionOut => [1, 1, 1, 0, 0, 0],
        F::CancelBeforeActivation => [1, 1, 1, 1, 0, 0],
        F::UnknownAfterLayer3DownMain | F::CancelBeforeLayer3DownReducer => [1, 1, 1, 1, 1, 1],
        _ => return None,
    })
}
fn require_tail_witness(
    fault: FixedPromptModelPrefixFault,
    cut: bool,
    prior: (usize, usize, usize),
    fa: usize,
    counts: [usize; 6],
    unpromoted: bool,
) -> Result<()> {
    if let Some(expected) = expected_tail_counts(fault)
        && (!cut || prior != (1, 1, 1) || fa != 1 || counts != expected || !unpromoted)
    {
        return Err(Error::State(
            "canonical47 actual layer3 tail partial/native/unpromoted witness absent".into(),
        ));
    }
    Ok(())
}
fn snapshot(owners: &PreparedOwners, service: &ResidentService) -> Result<Snapshot> {
    let cold = &owners.cold;
    let mut actual = BTreeMap::<BufferLeaseId, ResidentLease>::new();
    for (lease, _) in cold._body.accesses()? {
        actual.insert(lease.id(), lease.clone());
    }
    for domain in cold._initial.iter().chain(&cold._candidate) {
        for lease in &domain.leases {
            actual.insert(lease.id(), lease.clone());
        }
    }
    let descriptors = owners
        .cohort
        .iter()
        .map(|r| (r.lease_id(), r.offset(), r.bytes()))
        .collect::<Vec<_>>();
    if descriptors.iter().any(|(id, offset, bytes)| {
        actual.get(id).is_none_or(|lease| {
            *bytes == 0
                || offset
                    .checked_add(*bytes)
                    .is_none_or(|end| end > lease.bytes())
        })
    }) {
        return Err(Error::State(
            "canonical47 actual readback descriptor outside prepared graph".into(),
        ));
    }
    let uninitialized = actual
        .values()
        .filter(|l| !l.is_initialized())
        .map(ResidentLease::id)
        .collect();
    let convolution = owners.cut.scope().gdn_site(0)?.nodes()[0];
    let mut future = owners.cut.nodes()[convolution + 1..]
        .iter()
        .flat_map(|node| &node.outputs)
        .map(|id| cold._body.values().lease(*id).map(ResidentLease::id))
        .collect::<Result<BTreeSet<_>>>()?;
    future.extend(
        cold._candidate
            .iter()
            .flat_map(|d| d.leases.iter().map(ResidentLease::id)),
    );
    require_future_uninitialized(&future, &uninitialized)?;
    Ok(Snapshot {
        leases: actual.into_iter().map(|(id, l)| (id, l.bytes())).collect(),
        descriptors,
        uninitialized,
        future,
        state_buffers: cold
            ._initial
            .iter()
            .chain(&cold._candidate)
            .map(|d| d.leases.len())
            .sum(),
        values: owners.cut.values().len(),
        weights: cold._weights.accesses().len(),
        native: owners.native.retention_snapshot()?,
        resident: service.resident_bytes()?,
        uploaded: service.uploaded_value_bytes(),
    })
}

struct Witness {
    owners: Weak<PreparedOwners>,
    service: Weak<ResidentService>,
    source: SourceOwnerWitness,
    current: BTreeSet<BufferLeaseId>,
    snapshot: Snapshot,
    pending_before: usize,
    base: RootBinding,
}

#[derive(Default)]
pub(super) struct Probe(StdMutex<Option<Witness>>);

impl Probe {
    fn arm(
        &self,
        owners: &Arc<PreparedOwners>,
        service: &Arc<ResidentService>,
        accesses: &[(ResidentLease, DeviceAccess)],
        base: RootBinding,
    ) -> Result<()> {
        let mut guard = self
            .0
            .lock()
            .map_err(|_| Error::State("canonical47 first3 probe poisoned".into()))?;
        if guard.is_some() {
            return Err(Error::State(
                "canonical47 first3 probe already armed".into(),
            ));
        }
        let prepared = snapshot(owners, service)?;
        let current = accesses
            .iter()
            .map(|(l, _)| l.id())
            .collect::<BTreeSet<_>>();
        let pending_before = service.pending_events()?;
        if current.len() != accesses.len()
            || current.is_empty()
            || current.iter().any(|id| !prepared.leases.contains_key(id))
            || prepared.uninitialized.is_empty()
            || pending_before != 0
        {
            return Err(Error::State(
                "canonical47 first3 exact prepared/submitted cohort required".into(),
            ));
        }
        *guard = Some(Witness {
            owners: Arc::downgrade(owners),
            service: Arc::downgrade(service),
            source: owners.cut.scope().source().weak_owner(),
            current,
            snapshot: prepared,
            pending_before,
            base,
        });
        Ok(())
    }

    fn cut_observed(&self) -> Result<bool> {
        let guard = self
            .0
            .lock()
            .map_err(|_| Error::State("canonical47 first3 probe poisoned".into()))?;
        Ok(guard
            .as_ref()
            .and_then(|w| w.owners.upgrade())
            .is_some_and(|owners| owners.cold._body.fault_cut_observed()))
    }

    fn inspect(
        &self,
        fault: FixedPromptModelPrefixFault,
    ) -> Result<FixedPromptModelPrefixFaultOutcome> {
        let guard = self
            .0
            .lock()
            .map_err(|_| Error::State("canonical47 first3 probe poisoned".into()))?;
        let w = guard
            .as_ref()
            .ok_or_else(|| Error::State("canonical47 first3 refused before intended cut".into()))?;
        // The failed inner future and its caller-owned inputs are gone. Only
        // original quarantine ownership may keep these Weak targets alive.
        if !w.source.retained() {
            return Err(Error::State("canonical47 unknown source released".into()));
        }
        let source = &w.source;
        let owners = w
            .owners
            .upgrade()
            .ok_or_else(|| Error::State("canonical47 unknown prepared graph released".into()))?;
        let service = w
            .service
            .upgrade()
            .ok_or_else(|| Error::State("canonical47 unknown service released".into()))?;

        let after = snapshot(&owners, &service)?;
        w.snapshot.require_same(&after)?;
        let body = &owners.cold._body;
        let qkv_enqueues = owners.native.qkv_enqueues();
        let qkv_output_unpromoted = owners.native.qkv_output_unpromoted()?;
        require_partial_qkv_witness(
            fault,
            owners.cut.includes_qkv(),
            qkv_enqueues,
            qkv_output_unpromoted,
        )?;
        let (qk, kv, transport) = owners.native.attention_counts();
        let (qk_unpromoted, kv_candidate_unpromoted) = owners
            .native
            .attention_binding()
            .map(|binding| binding.outputs_unpromoted())
            .transpose()?
            .unwrap_or((false, false));
        require_partial_kv_witness(
            fault,
            owners.cut.includes_kv(),
            (qk, kv, transport),
            qk_unpromoted,
            kv_candidate_unpromoted,
        )?;
        let fa_native_enqueues = owners.native.fa_enqueues();
        let fa_output_unpromoted = owners.native.fa_output_unpromoted()?;
        require_partial_fa_witness(
            fault,
            owners.cut.includes_fa(),
            (qk, kv, transport),
            fa_native_enqueues,
            fa_output_unpromoted,
        )?;
        let tail_native_enqueues = owners.native.tail_native_enqueues();
        let tail_outputs_unpromoted = owners.native.tail_outputs_unpromoted()?;
        require_tail_witness(
            fault,
            owners.cut.includes_layer3(),
            (qk, kv, transport),
            fa_native_enqueues,
            tail_native_enqueues,
            tail_outputs_unpromoted,
        )?;
        let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?;
        let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let completion_owners =
            crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        if !source.same_issuer(owners.cut.scope().source())
            || !body.fault_cut_observed()
            || body.convolution_dispatches() == 0
            || w.current.iter().any(|id| !after.leases.contains_key(id))
            || service.pending_events()? != w.pending_before + 1
            || service.ensure_usable().is_ok()
            || !physical.fenced
            || physical.reserved != service.capacity() as u64
            || handles < w.current.len()
            || completion_owners == 0
        {
            return Err(Error::State("canonical47 first3 actual submitted unknown did not preserve original fence/owners".into()));
        }
        Ok(FixedPromptModelPrefixFaultOutcome {
            fault,
            status: "UNKNOWN_RETAINED",
            ready_completions: 0,
            readback_commands: 0,
            expected_callbacks: 0,
            completed_prefixes: 0,
            native_launches: body.native_launches(),
            convolution_dispatches: body.convolution_dispatches(),
            qkv_main_reached: qkv_enqueues > 0,
            qkv_native_enqueues: qkv_enqueues,
            qkv_output_unpromoted,
            qk_native_enqueues: qk,
            kv_native_enqueues: kv,
            kv_transport_enqueues: transport,
            kv_candidate_unpromoted,
            fa_native_enqueues,
            fa_output_unpromoted,
            tail_native_enqueues,
            tail_outputs_unpromoted,
            resident_bytes: after.resident,
            pending_events: service.pending_events()?,
            uploaded_value_bytes: after.uploaded,
            original_root: Some(w.base),
            prepared_state_buffers: after.state_buffers,
            retained_plan_values: after.values,
            retained_source_weight_groups: after.weights,
            prepared_graph_leases: after.leases.len(),
            submitted_current_leases: w.current.len(),
            readback_descriptors: after.descriptors.len(),
            uninitialized_prepared_leases: after.uninitialized.len(),
            source_retained: true,
            all_prepared_owners_retained: true,
            all_readback_descriptors_retained: true,
            all_future_uninitialized: true,
            native_image_owners: after.native.images,
            native_kernel_owners: after.native.kernels,
            native_charge_bytes: after.native.native_charge_bytes,
            image_admission_bytes: after.native.image_admission_bytes,
            quarantined_handles: handles,
            quarantined_completion_owners: completion_owners,
            physical_reserved_bytes: physical.reserved,
            physical_fenced: physical.fenced,
            model_root_published: false,
            parent_model_accepted: false,
        })
    }
}

pub(super) enum Mode {
    Compare,
    Fault {
        fault: FixedPromptModelPrefixFault,
        probe: Arc<Probe>,
    },
}

impl Mode {
    pub(super) fn is_pressure(&self) -> bool {
        matches!(
            self,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::AdmissionPressure,
                ..
            }
        )
    }

    pub(super) fn arm(
        &self,
        owners: &Arc<PreparedOwners>,
        service: &Arc<ResidentService>,
        accesses: &[(ResidentLease, DeviceAccess)],
        base: RootBinding,
    ) -> Result<()> {
        match self {
            Self::Compare => Ok(()),
            Self::Fault { probe, .. } => probe.arm(owners, service, accesses, base),
        }
    }

    pub(super) fn fault_mode(&self) -> usize {
        match self {
            Self::Fault {
                fault: FixedPromptModelPrefixFault::UnknownAfterConvolution,
                ..
            } => 1,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::CancelBeforeRecurrent,
                ..
            } => 2,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::UnknownAfterQkvMain,
                ..
            } => 3,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::CancelBeforeQkvReducer,
                ..
            } => 4,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::UnknownAfterQk,
                ..
            } => 5,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::CancelBeforeKvAppend,
                ..
            } => 6,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::UnknownAfterKvAppend,
                ..
            } => 7,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::CancelBeforeKvTransport,
                ..
            } => 8,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::UnknownAfterFa,
                ..
            } => 9,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::CancelBeforeFa,
                ..
            } => 10,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::UnknownAfterSigmoid,
                ..
            } => 11,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::CancelBeforeMultiply,
                ..
            } => 12,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::UnknownAfterAttentionOut,
                ..
            } => 13,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::CancelBeforeActivation,
                ..
            } => 14,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::UnknownAfterLayer3DownMain,
                ..
            } => 15,
            Self::Fault {
                fault: FixedPromptModelPrefixFault::CancelBeforeLayer3DownReducer,
                ..
            } => 16,
            _ => 0,
        }
    }
}

pub(super) enum Outcome {
    Compared(FixedPromptModelPrefixReceipt),
    Fault(Box<FixedPromptModelPrefixFaultOutcome>),
}

fn intended_error(error: &Error, fault: FixedPromptModelPrefixFault) -> bool {
    matches!((error, fault),
        (Error::State(message), FixedPromptModelPrefixFault::UnknownAfterConvolution)
            if message == "deliberate full24 unknown after actual source Conv and canonical copy")
        || matches!((error, fault),
        (Error::State(message), FixedPromptModelPrefixFault::CancelBeforeRecurrent)
            if message == "closed original cubin qualification cancelled")
        || matches!((error, fault),
        (Error::State(message), FixedPromptModelPrefixFault::UnknownAfterQkvMain)
            if message == "deliberate canonical47 unknown after actual QKV main")
        || matches!((error, fault),
        (Error::State(message), FixedPromptModelPrefixFault::CancelBeforeQkvReducer)
            if message == "closed original cubin qualification cancelled")
        || matches!((error,fault),(Error::State(message),FixedPromptModelPrefixFault::UnknownAfterQk)
            if message=="deliberate canonical47 unknown after actual QK")
        || matches!((error,fault),(Error::State(message),FixedPromptModelPrefixFault::UnknownAfterKvAppend)
            if message=="deliberate canonical47 unknown after actual KV append")
        || matches!((error,fault),(Error::State(message),FixedPromptModelPrefixFault::CancelBeforeKvAppend|FixedPromptModelPrefixFault::CancelBeforeKvTransport)
            if message=="full24 original KV transport cancelled")
        || matches!((error,fault),(Error::State(message),FixedPromptModelPrefixFault::UnknownAfterFa)
            if message=="deliberate canonical47 unknown after actual FA")
        || matches!((error,fault),(Error::State(message),FixedPromptModelPrefixFault::CancelBeforeFa)
            if message=="closed original cubin qualification cancelled")
        || matches!((error,fault),(Error::State(message),FixedPromptModelPrefixFault::UnknownAfterSigmoid) if message=="deliberate canonical47 unknown after actual Sigmoid")
        || matches!((error,fault),(Error::State(message),FixedPromptModelPrefixFault::UnknownAfterAttentionOut) if message=="deliberate canonical47 unknown after actual AttentionOut")
        || matches!((error,fault),(Error::State(message),FixedPromptModelPrefixFault::UnknownAfterLayer3DownMain) if message=="deliberate canonical47 unknown after actual layer3 Down main")
        || matches!((error,fault),(Error::State(message),FixedPromptModelPrefixFault::CancelBeforeMultiply|FixedPromptModelPrefixFault::CancelBeforeActivation|FixedPromptModelPrefixFault::CancelBeforeLayer3DownReducer) if message=="closed original cubin qualification cancelled")
}

fn require_fault_cut(
    kind: FixedPromptPrefixKind,
    fault: FixedPromptModelPrefixFault,
) -> Result<()> {
    if kind == FixedPromptPrefixKind::FirstThree
        && matches!(
            fault,
            FixedPromptModelPrefixFault::UnknownAfterQkvMain
                | FixedPromptModelPrefixFault::CancelBeforeQkvReducer
        )
    {
        return Err(Error::Unsupported(
            "canonical47 partial QKV faults require original33-node cut".into(),
        ));
    }
    if !matches!(
        kind,
        FixedPromptPrefixKind::ThroughAttentionKv
            | FixedPromptPrefixKind::ThroughAttentionFa
            | FixedPromptPrefixKind::ThroughLayer3
    ) && matches!(
        fault,
        FixedPromptModelPrefixFault::UnknownAfterQk
            | FixedPromptModelPrefixFault::CancelBeforeKvAppend
            | FixedPromptModelPrefixFault::UnknownAfterKvAppend
            | FixedPromptModelPrefixFault::CancelBeforeKvTransport
    ) {
        return Err(Error::Unsupported(
            "canonical47 QK/KV partial faults require original35-node cut".into(),
        ));
    }
    if !matches!(
        kind,
        FixedPromptPrefixKind::ThroughAttentionFa | FixedPromptPrefixKind::ThroughLayer3
    ) && matches!(
        fault,
        FixedPromptModelPrefixFault::UnknownAfterFa | FixedPromptModelPrefixFault::CancelBeforeFa
    ) {
        return Err(Error::Unsupported(
            "canonical47 FA faults require original36-node cut".into(),
        ));
    }
    if kind != FixedPromptPrefixKind::ThroughLayer3 && expected_tail_counts(fault).is_some() {
        return Err(Error::Unsupported(
            "canonical47 tail faults require original43-node cut".into(),
        ));
    }
    Ok(())
}

/// No expected reader. Pre-set cancellation precedes all NEW preparation and
/// service/provider work; source/image/control caller owners may already exist.
pub async fn qualify_fixed_prompt_model_prefix_fault(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptMlpQualificationInputs,
    fault: FixedPromptModelPrefixFault,
    cancelled: Option<&AtomicBool>,
) -> Result<FixedPromptModelPrefixFaultOutcome> {
    qualify_fault(
        device,
        source,
        record,
        arithmetic,
        inputs,
        fault,
        cancelled,
        FixedPromptPrefixKind::FirstThree,
        None,
    )
    .await
}

/// SAME original graph/flight; partial main cuts retain all original owners.
pub async fn qualify_fixed_prompt_model_qkv_prefix_fault(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptMlpQualificationInputs,
    fault: FixedPromptModelPrefixFault,
    cancelled: Option<&AtomicBool>,
) -> Result<FixedPromptModelPrefixFaultOutcome> {
    qualify_fault(
        device,
        source,
        record,
        arithmetic,
        inputs,
        fault,
        cancelled,
        FixedPromptPrefixKind::ThroughAttentionQkv,
        None,
    )
    .await
}

/// Same original body/flight; no expected ranges in any uncertain path.
pub async fn qualify_fixed_prompt_model_kv_prefix_fault(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptQkKvQualificationInputs,
    fault: FixedPromptModelPrefixFault,
    cancelled: Option<&AtomicBool>,
) -> Result<FixedPromptModelPrefixFaultOutcome> {
    let FixedPromptQkKvQualificationInputs { native, rotary } = inputs;
    qualify_fault(
        device,
        source,
        record,
        arithmetic,
        native,
        fault,
        cancelled,
        FixedPromptPrefixKind::ThroughAttentionKv,
        rotary,
    )
    .await
}

pub async fn qualify_fixed_prompt_model_fa_prefix_fault(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptQkKvQualificationInputs,
    fault: FixedPromptModelPrefixFault,
    cancelled: Option<&AtomicBool>,
) -> Result<FixedPromptModelPrefixFaultOutcome> {
    let FixedPromptQkKvQualificationInputs { native, rotary } = inputs;
    qualify_fault(
        device,
        source,
        record,
        arithmetic,
        native,
        fault,
        cancelled,
        FixedPromptPrefixKind::ThroughAttentionFa,
        rotary,
    )
    .await
}

pub async fn qualify_fixed_prompt_model_layer3_prefix_fault(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptQkKvQualificationInputs,
    fault: FixedPromptModelPrefixFault,
    cancelled: Option<&AtomicBool>,
) -> Result<FixedPromptModelPrefixFaultOutcome> {
    let FixedPromptQkKvQualificationInputs { native, rotary } = inputs;
    qualify_fault(
        device,
        source,
        record,
        arithmetic,
        native,
        fault,
        cancelled,
        FixedPromptPrefixKind::ThroughLayer3,
        rotary,
    )
    .await
}

#[expect(
    clippy::too_many_arguments,
    reason = "private shared typed-cut adapter; public APIs remain seven arguments"
)]
async fn qualify_fault(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptMlpQualificationInputs,
    fault: FixedPromptModelPrefixFault,
    cancelled: Option<&AtomicBool>,
    kind: FixedPromptPrefixKind,
    rotary: Option<crate::cuda::native::source_rotary::RotaryOutcome>,
) -> Result<FixedPromptModelPrefixFaultOutcome> {
    not_cancelled(cancelled)?;
    require_fault_cut(kind, fault)?;
    let probe = Arc::new(Probe::default());
    let result = qualify_inner(
        device,
        source,
        record,
        arithmetic,
        inputs,
        Execution {
            cancelled,
            kind,
            rotary,
            mode: Mode::Fault {
                fault,
                probe: probe.clone(),
            },
        },
        |_: FixedPromptModelPrefixReadback<'_>| {
            Err(Error::State(
                "canonical47 first3 fault cannot compare/readback".into(),
            ))
        },
    )
    .await;
    match result {
        Ok(Outcome::Fault(value)) => Ok(*value),
        Ok(Outcome::Compared(_)) => Err(Error::State(
            "canonical47 first3 fault completed positive prefix".into(),
        )),
        Err(error) => {
            if intended_error(&error, fault) && probe.cut_observed()? {
                probe.inspect(fault)
            } else {
                Err(error)
            }
        }
    }
}

pub(super) fn pressure(
    service: Arc<ResidentService>,
) -> Result<FixedPromptModelPrefixFaultOutcome> {
    // SAME original ledger cap+1 attempt; it must refuse before allocation/H2D.
    let active = super::super::faults::pressure(service, GdnFragmentFault::AdmissionPressure)?;
    Ok(FixedPromptModelPrefixFaultOutcome {
        fault: FixedPromptModelPrefixFault::AdmissionPressure,
        status: active.status,
        ready_completions: 0,
        readback_commands: 0,
        expected_callbacks: 0,
        completed_prefixes: 0,
        native_launches: 0,
        convolution_dispatches: 0,
        qkv_main_reached: false,
        qkv_native_enqueues: 0,
        qkv_output_unpromoted: false,
        qk_native_enqueues: 0,
        kv_native_enqueues: 0,
        kv_transport_enqueues: 0,
        kv_candidate_unpromoted: false,
        fa_native_enqueues: 0,
        fa_output_unpromoted: false,
        tail_native_enqueues: [0; 6],
        tail_outputs_unpromoted: false,
        resident_bytes: active.resident_bytes,
        pending_events: active.pending_events,
        uploaded_value_bytes: active.uploaded_value_bytes,
        original_root: None,
        prepared_state_buffers: 0,
        retained_plan_values: 0,
        retained_source_weight_groups: 0,
        prepared_graph_leases: 0,
        submitted_current_leases: 0,
        readback_descriptors: 0,
        uninitialized_prepared_leases: 0,
        source_retained: false,
        all_prepared_owners_retained: false,
        all_readback_descriptors_retained: false,
        all_future_uninitialized: true,
        native_image_owners: 0,
        native_kernel_owners: 0,
        native_charge_bytes: 0,
        image_admission_bytes: 0,
        quarantined_handles: active.quarantined_handles,
        quarantined_completion_owners: active.quarantined_completion_owners,
        physical_reserved_bytes: active.physical_reserved_bytes,
        physical_fenced: active.physical_fenced,
        model_root_published: false,
        parent_model_accepted: false,
    })
}

#[cfg(test)]
mod tests;
