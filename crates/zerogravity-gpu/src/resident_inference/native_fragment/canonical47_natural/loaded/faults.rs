//! Real cached submitted cuts; inspect only Weak owners after all caller drops.
use super::*;
use crate::resident_leases::ResidentReadback;
use crate::resident_program::native_full::FullModelProgram;
use zerogravity_core::device_events::BufferLeaseId;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Canonical47NaturalFault {
    UnknownAfterFirstConvolution,
    CancelBeforeFirstRecurrent,
    UnknownAfterLastFa,
}
impl Canonical47NaturalFault {
    fn code(self) -> usize {
        match self {
            Self::UnknownAfterFirstConvolution => 1,
            Self::CancelBeforeFirstRecurrent => 2,
            Self::UnknownAfterLastFa => 19,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Canonical47NaturalFaultOutcome {
    pub fault: Canonical47NaturalFault,
    pub status: &'static str,
    pub root_before: RootBinding,
    pub root_after: RootBinding,
    pub source_retained: bool,
    pub body_retained: bool,
    pub cold_graph_retained: bool,
    pub candidate_buffers_unpromoted: bool,
    pub prepared_graph_leases: usize,
    pub submitted_current_leases: usize,
    pub readback_descriptors: usize,
    pub native_image_owners: usize,
    pub native_kernel_owners: usize,
    pub native_charge_bytes: usize,
    pub image_admission_bytes: usize,
    pub native_launches: usize,
    pub convolution_dispatches: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub physical_fenced: bool,
    pub physical_reserved_bytes: u64,
    pub ready_completions: usize,
    pub readback_commands: usize,
    pub expected_callbacks: usize,
    pub published_roots: usize,
}
#[derive(PartialEq, Eq)]
struct Snapshot {
    leases: LeaseSnapshot,
    submitted: BTreeSet<BufferLeaseId>,
    descriptors: Vec<(BufferLeaseId, usize, usize)>,
    native: (usize, usize, usize, usize),
}
type LeaseSnapshot = BTreeMap<BufferLeaseId, (usize, u64, bool)>;
type Descriptor = (BufferLeaseId, usize, usize);
fn require_descriptors(leases: &LeaseSnapshot, descriptors: &[Descriptor]) -> Result<()> {
    // The sampler repeats the shared source status as the final descriptor.
    let ids = descriptors.iter().map(|d| d.0).collect::<BTreeSet<_>>();
    if descriptors.len() != 27
        || ids.len() != 26
        || descriptors.first() != descriptors.last()
        || descriptors.iter().enumerate().any(|(i, (id, offset, n))| {
            *offset != 0
                || *n != (if i == 25 { 12 } else { 4 })
                || leases.get(id).is_none_or(|(bytes, _, _)| {
                    offset.checked_add(*n).is_none_or(|end| end > *bytes)
                })
        })
    {
        return Err(Error::State(
            "canonical actual production descriptor cohort/extent differs".into(),
        ));
    }
    Ok(())
}
fn snapshot(
    cold: &ColdModel,
    body: &FullModelProgram,
    descriptors: &[Descriptor],
) -> Result<Snapshot> {
    let mut leases = BTreeMap::new();
    let mut add = |l: ResidentLease| -> Result<()> {
        if l.source_identity() != cold.source.source().source_identity()
            || l.service_generation() != l.service().generation()
        {
            return Err(Error::State(
                "canonical quarantine same source/service/generation".into(),
            ));
        }
        leases.insert(
            l.id(),
            (l.bytes(), l.allocation_generation(), l.is_initialized()),
        );
        Ok(())
    };
    for v in &cold.values {
        for (l, _) in v.storage_accesses() {
            add(l)?;
        }
    }
    for g in &cold.gdn {
        for p in g.values() {
            for (l, _) in p.storage_accesses() {
                add(l)?;
            }
        }
    }
    for (l, _) in cold.native.storage_accesses() {
        add(l)?;
    }
    add(cold.tied.lease().clone())?;
    add(cold.rotary.lease().clone())?;
    add(cold.positions.lease().clone())?;
    for p in cold.parameters.values() {
        for (l, _) in p.immutable_weights() {
            add(l)?;
        }
    }
    for p in cold.attention_parameters.values() {
        for (l, _) in p.immutable_weights() {
            add(l)?;
        }
    }
    let mut submitted = BTreeSet::new();
    for (l, _) in body.accesses()? {
        submitted.insert(l.id());
        add(l.clone())?;
    }
    require_descriptors(&leases, descriptors)?;
    Ok(Snapshot {
        leases,
        submitted,
        descriptors: descriptors.to_vec(),
        native: cold.native.retention()?,
    })
}
pub(in crate::resident_inference::native_fragment::canonical47_natural) struct ProbeRecord {
    cold: Weak<ColdModel>,
    body: Weak<FullModelProgram>,
    input: Weak<CanonicalCallInput>,
    state: Weak<CanonicalCallState>,
    service: Weak<ResidentService>,
    source: Weak<OriginalWindowModelSource>,
    before: Snapshot,
    _metadata: WindowWorkspace,
}
impl ProbeRecord {
    pub(in crate::resident_inference::native_fragment::canonical47_natural) fn capture(
        cold: &Arc<ColdModel>,
        body: &Arc<FullModelProgram>,
        input: &Arc<CanonicalCallInput>,
        state: &Arc<CanonicalCallState>,
        reads: &[ResidentReadback],
    ) -> Result<Self> {
        let metadata = cold
            .source
            .source()
            .source()
            .reserve_workspace(256 * 1024)?;
        if input.base() != 47
            || input.rows() != 1
            || !Arc::ptr_eq(state.input(), input)
            || !body.canonical47_call_retained()?
            || state.outputs().iter().any(ResidentLease::is_initialized)
        {
            return Err(Error::State(
                "canonical exact fresh cached witness graph".into(),
            ));
        }
        let descriptors = reads
            .iter()
            .map(|r| (r.lease_id(), r.offset(), r.bytes()))
            .collect::<Vec<_>>();
        let before = snapshot(cold, body, &descriptors)?;
        Ok(Self {
            cold: Arc::downgrade(cold),
            body: Arc::downgrade(body),
            input: Arc::downgrade(input),
            state: Arc::downgrade(state),
            service: Arc::downgrade(body.values().service()),
            source: Arc::downgrade(cold.source.source()),
            before,
            _metadata: metadata,
        })
    }
    fn inspect(
        self,
        fault: Canonical47NaturalFault,
        before: RootBinding,
        after: RootBinding,
    ) -> Result<Canonical47NaturalFaultOutcome> {
        let cold = self
            .cold
            .upgrade()
            .ok_or_else(|| Error::State("canonical quarantine lost entire cold graph".into()))?;
        let body = self
            .body
            .upgrade()
            .ok_or_else(|| Error::State("canonical quarantine lost actual current body".into()))?;
        let input = self
            .input
            .upgrade()
            .ok_or_else(|| Error::State("canonical quarantine lost current call seal".into()))?;
        let state = self.state.upgrade().ok_or_else(|| {
            Error::State("canonical quarantine lost canonical candidate owners".into())
        })?;
        let service = self
            .service
            .upgrade()
            .ok_or_else(|| Error::State("canonical quarantine lost original service".into()))?;
        let source = self
            .source
            .upgrade()
            .ok_or_else(|| Error::State("canonical quarantine lost immutable source".into()))?;
        if before != after
            || input.attempt().base() != before
            || !Arc::ptr_eq(&source, cold.source.source())
            || !Arc::ptr_eq(input.record(), body.record())
            || !Arc::ptr_eq(state.input(), &input)
            || self.before != snapshot(&cold, &body, &self.before.descriptors)?
            || !body.fault_cut_observed()
            || !body.canonical47_call_retained()?
            || state.outputs().iter().any(ResidentLease::is_initialized)
            || service.pending_events()? != 1
            || service.ensure_usable().is_ok()
        {
            return Err(Error::State(
                "canonical unknown cached publication/owner/state/fence contract differs".into(),
            ));
        }
        let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?;
        let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let owners =
            crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        if !physical.fenced
            || physical.reserved == 0
            || handles != self.before.submitted.len()
            || owners == 0
        {
            return Err(Error::State(
                "canonical actual current flight quarantine absent".into(),
            ));
        }
        let n = self.before.native;
        Ok(Canonical47NaturalFaultOutcome {
            fault,
            status: "UNKNOWN_CACHED_CALL_RETAINED_NO_PUBLICATION",
            root_before: before,
            root_after: after,
            source_retained: true,
            body_retained: true,
            cold_graph_retained: true,
            candidate_buffers_unpromoted: true,
            prepared_graph_leases: self.before.leases.len(),
            submitted_current_leases: self.before.submitted.len(),
            readback_descriptors: self.before.descriptors.len(),
            native_image_owners: n.0,
            native_kernel_owners: n.1,
            native_charge_bytes: n.2,
            image_admission_bytes: n.3,
            native_launches: body.native_launches(),
            convolution_dispatches: body.convolution_dispatches(),
            resident_bytes: service.resident_bytes()?,
            pending_events: service.pending_events()?,
            quarantined_handles: handles,
            quarantined_completion_owners: owners,
            physical_fenced: physical.fenced,
            physical_reserved_bytes: physical.reserved,
            ready_completions: 0,
            readback_commands: 0,
            expected_callbacks: 0,
            published_roots: 0,
        })
    }
}
impl Canonical47Session {
    pub async fn probe_cached(
        self,
        fault: Canonical47NaturalFault,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Canonical47NaturalFaultOutcome> {
        not_cancelled(cancelled)?;
        if Arc::strong_count(&self.loaded) != 1
            || self.loaded.active.load(Ordering::Acquire) != 1
            || self.loaded.pending.load(Ordering::Acquire) != 0
            || self.loaded.snapshots.load(Ordering::Acquire) != 0
            || self.binding().frontier() != 47
        {
            return Err(Error::State(
                "canonical cached fault consumes sole loaded session after genuine initial commit"
                    .into(),
            ));
        }
        let mut witness = None;
        let before = self.binding();
        let feedback = self.feedback.clone().ok_or_else(|| {
            Error::State("canonical cached fault lacks actual committed feedback".into())
        })?;
        let input = self.original.bind_canonical47_feedback(
            self.loaded.cold.source.clone(),
            feedback,
            cancelled,
        )?;
        let state = self
            .original
            .bind_fresh_canonical47_state(input.clone(), cancelled)
            .await?;
        let result = self
            .original
            .prepare_canonical_call(
                self.loaded.cold.clone(),
                input.clone(),
                state.clone(),
                Some((fault.code(), &mut witness)),
                None,
                cancelled,
            )
            .await;
        let after = self.binding();
        if result.is_ok() {
            return Err(Error::State(
                "canonical deliberately cut cached call unexpectedly completed".into(),
            ));
        }
        let witness = witness.ok_or_else(|| {
            Error::State(format!(
                "canonical cached cut failed before actual graph binding: {:?}",
                result.as_ref().err()
            ))
        })?;
        // No external strong body/source/model/session owner supplies this proof.
        drop(result);
        drop(input);
        drop(state);
        drop(self);
        witness.inspect(fault, before, after)
    }
}

#[cfg(test)]
mod descriptor_tests {
    use super::*;
    use zerogravity_core::device_events::{BufferRole, DeviceEventLedger};
    fn cohort() -> (DeviceEventLedger, LeaseSnapshot, Vec<Descriptor>) {
        let mut ledger = DeviceEventLedger::new(0, 116).unwrap();
        let mut leases = BTreeMap::new();
        let mut descriptors = Vec::new();
        for i in 0..26 {
            let bytes = if i == 25 { 12 } else { 4 };
            let id = ledger.admit(bytes, BufferRole::Output).unwrap();
            leases.insert(id, (bytes, 1, false));
            descriptors.push((id, 0, bytes));
        }
        descriptors.push(descriptors[0]);
        (ledger, leases, descriptors)
    }
    #[test]
    fn witness_uses_only_the_actual_production_cohort() {
        let (mut ledger, leases, descriptors) = cohort();
        require_descriptors(&leases, &descriptors).unwrap();
        assert_eq!(descriptors.iter().map(|d| d.2).sum::<usize>(), 116);
        assert_eq!(descriptors[0], descriptors[26]);
        assert_eq!(leases.len(), 26);
        let mut wrong_status = descriptors.clone();
        wrong_status[26] = descriptors[1];
        assert!(require_descriptors(&leases, &wrong_status).is_err());
        let mut unrelated_alias = descriptors.clone();
        unrelated_alias[2] = descriptors[1];
        assert!(require_descriptors(&leases, &unrelated_alias).is_err());
        let mut distinct = descriptors.clone();
        let id = ledger.admit(4, BufferRole::Output).unwrap();
        let mut distinct_leases = leases.clone();
        distinct_leases.insert(id, (4, 1, false));
        distinct[26] = (id, 0, 4);
        assert!(require_descriptors(&distinct_leases, &distinct).is_err());
        for i in 0..descriptors.len() {
            let mut missing = descriptors.clone();
            missing.remove(i);
            assert!(require_descriptors(&leases, &missing).is_err());
        }
        let mut diagnostic = descriptors.clone();
        diagnostic.extend(descriptors.clone());
        assert!(require_descriptors(&leases, &diagnostic).is_err());
        let mut duplicate = descriptors.clone();
        duplicate[0] = duplicate[1];
        assert!(require_descriptors(&leases, &duplicate).is_err());
    }
    #[test]
    fn witness_rejects_changed_or_outside_retained_ranges() {
        let (_ledger, leases, descriptors) = cohort();
        for i in 0..descriptors.len() {
            let mut changed = descriptors.clone();
            changed[i].1 = 1;
            assert!(require_descriptors(&leases, &changed).is_err());
            changed = descriptors.clone();
            changed[i].2 += 4;
            assert!(require_descriptors(&leases, &changed).is_err());
        }
        let mut absent = leases.clone();
        absent.remove(&descriptors[0].0);
        assert!(require_descriptors(&absent, &descriptors).is_err());
        let mut short = leases;
        short.get_mut(&descriptors[25].0).unwrap().0 = 8;
        assert!(require_descriptors(&short, &descriptors).is_err());
    }
}
