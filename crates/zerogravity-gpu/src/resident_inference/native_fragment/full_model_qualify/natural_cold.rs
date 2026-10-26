//! SAME cold setup; fresh authenticated calls reuse original storage and kernels.
use super::*;
use crate::resident_inference::native_fragment::natural_publication::{
    NaturalCallInput, NaturalCallState, NaturalStatePool,
};
use crate::resident_program::natural_binding::ColdProgramSlot;
use std::sync::{Weak, atomic::Ordering};
mod budget;
mod reusable;
use budget::require_host_bound;
pub use reusable::qualify_natural_reuse_pressure;
pub use reusable::{
    PreparedReusableNatural, ReusableNaturalStorage, prepare_reusable_natural_cold,
};

pub struct PreparedNaturalCold {
    pub model: Arc<ResidentModel>,
    pub sessions: [ResidentSession; 2],
    pub slots: Arc<ColdModelSlots>,
    pub state_pool: Arc<NaturalStatePool>,
}
/// No strong Model/Session edge: Model.workspace may own a bound call body.
/// This owner is retained by original init/flight, including future-uninitialized
/// slots, without marking future values initialized/current Ready.
pub struct ColdModelSlots {
    templates: Vec<ColdProgramSlot>,
    model: Weak<ResidentModel>,
    bases: [RootBinding; 2],
    owners: Arc<FullOwners>,
    retirement_started: AtomicBool,
    _metadata: zerogravity_artifacts::source_weights::WindowWorkspace,
}
impl ColdModelSlots {
    pub(super) fn new(
        model: &Arc<ResidentModel>,
        sessions: &[ResidentSession; 2],
        templates: Vec<ColdProgramSlot>,
        owners: Arc<FullOwners>,
        metadata: zerogravity_artifacts::source_weights::WindowWorkspace,
    ) -> Result<Arc<Self>> {
        if templates.len() != 5 || templates.capacity() > 8 {
            return Err(Error::State(
                "natural exact five original cold storage slots".into(),
            ));
        }
        for (index, slot) in templates.iter().enumerate() {
            if crate::resident_program::natural_binding::slot(slot.record().call())? != index {
                return Err(Error::State("natural original cold slot order".into()));
            }
        }
        Ok(Arc::new(Self {
            templates,
            model: Arc::downgrade(model),
            bases: [sessions[0].binding(), sessions[1].binding()],
            owners,
            retirement_started: AtomicBool::new(false),
            _metadata: metadata,
        }))
    }
    pub(crate) fn completion_owner(
        self: &Arc<Self>,
    ) -> crate::physical_quarantine::CompletionOwner {
        self.clone()
    }
    pub(crate) fn bind_call(
        self: &Arc<Self>,
        session: &ResidentSession,
        input: Arc<NaturalCallInput>,
        state: Arc<NaturalCallState>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ResidentProgram> {
        not_cancelled(cancelled)?;
        input.require_session(session)?;
        state.require_current(session)?;
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("natural original model retired".into()))?;
        let index = input.slot().index();
        let h = history(index);
        let current = session.binding();
        let base = self.bases[h];
        if self.retirement_started.load(Ordering::Acquire)
            || index >= 5
            || !Arc::ptr_eq(&model, &session.model)
            || !Arc::ptr_eq(&input, state.input())
            || current.session() != base.session()
            || current.session_epoch() != base.session_epoch()
            || current.root() != local_phase(index) as u64
            || input.slot().base() as u64 != current.frontier()
        {
            return Err(Error::State(
                "natural original session/history/epoch/call slot mismatch".into(),
            ));
        }
        let binding = session.bind_text_positions(
            input.record().clone(),
            input.attempt(),
            Some(input.phase_binding().clone()),
            cancelled,
        )?;
        let positions = self.owners._positions[index].bind_natural(binding)?;
        self.templates[index].bind(input, state, positions, self.owners.tied.clone(), cancelled)
    }
    /// All five original commits must be real and current. Pending/unknown work
    /// has no cleanup path here; it remains fenced with the whole cold graph.
    pub async fn verify_source_and_retire(
        self: &Arc<Self>,
        sessions: &[ResidentSession; 2],
        cancelled: Option<&AtomicBool>,
    ) -> Result<(usize, usize)> {
        not_cancelled(cancelled)?;
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("natural model absent at retirement".into()))?;
        for (h, session) in sessions.iter().enumerate() {
            let now = session.binding();
            let base = self.bases[h];
            if !Arc::ptr_eq(&session.model, &model)
                || now.session() != base.session()
                || now.session_epoch() != base.session_epoch()
                || now.frontier() != [2, 67][h]
                || now.root() != [2, 3][h]
            {
                return Err(Error::State(
                    "natural retirement requires exact original five published calls".into(),
                ));
            }
            model.validate_domains(&session.domains, true)?;
        }
        if self.templates.iter().any(|slot| !slot.is_claimed())
            || self
                .owners
                .candidates
                .iter()
                .flatten()
                .flat_map(|domain| &domain.leases)
                .any(|lease| !lease.is_initialized())
            || model.service.pending_events()? != 0
        {
            return Err(Error::State(
                "natural retirement before original completed candidate graph".into(),
            ));
        }
        self.retire_quiescent(&model, cancelled).await
    }
    pub(crate) async fn verify_source_and_retire_probe(
        self: &Arc<Self>,
        sessions: &[ResidentSession; 2],
        kind: NaturalRetirementProbe,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(usize, usize)> {
        not_cancelled(cancelled)?;
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("natural probe model owner retired".into()))?;
        let (root, epoch) = match kind {
            NaturalRetirementProbe::CancelBeforeCommit => (0, 0),
            NaturalRetirementProbe::RestoredFirst => (2, 1),
        };
        for (h, session) in sessions.iter().enumerate() {
            let now = session.binding();
            let base = self.bases[h];
            if !Arc::ptr_eq(&session.model, &model)
                || now.session() != base.session()
                || now.frontier() != 0
                || now.root() != if h == 0 { root } else { 0 }
                || now.session_epoch() != if h == 0 { epoch } else { base.session_epoch() }
            {
                return Err(Error::State(
                    "natural safe-retirement probe original published roots differ".into(),
                ));
            }
            model.validate_domains(&session.domains, true)?;
        }
        if self
            .templates
            .iter()
            .enumerate()
            .any(|(index, slot)| slot.is_claimed() != (index == 0))
            || self
                .owners
                .candidates
                .iter()
                .enumerate()
                .any(|(index, domains)| {
                    domains
                        .iter()
                        .flat_map(|domain| &domain.leases)
                        .any(|lease| lease.is_initialized() != (index == 0))
                })
        {
            return Err(Error::State(
                "natural probe requires only actually completed first candidate".into(),
            ));
        }
        self.retire_quiescent(&model, cancelled).await
    }
    async fn retire_quiescent(
        self: &Arc<Self>,
        model: &ResidentModel,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(usize, usize)> {
        model.service.ensure_usable()?;
        if model.service.pending_events()? != 0
            || self
                .retirement_started
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(Error::State(
                "natural source retirement requires quiescent original service once".into(),
            ));
        }
        let source = self
            .owners
            .weights
            .verify_source_windows(self.completion_owner(), cancelled)
            .await?
            .0;
        let tied = super::verify_tied_and_retire(&model.service, &self.owners, cancelled).await?;
        Ok((source, tied))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NaturalRetirementProbe {
    CancelBeforeCommit,
    RestoredFirst,
}
/// Root reference/expected archives are never inputs. This performs only the
/// SAME full cold preparation; no node loop or model publication runs here.
pub async fn prepare_natural_cold(
    device: Device,
    scopes: SourceModelScopes,
    inputs: SourceModelInputs,
    rotary: Option<RotaryOutcome>,
    cancelled: Option<&AtomicBool>,
) -> Result<PreparedNaturalCold> {
    let (source, records) = scopes.into_parts();
    not_cancelled(cancelled)?;
    require_host_bound(&source, &records)?;
    match super::qualify_inner(
        device,
        source,
        records,
        inputs,
        Execution {
            rotary,
            cancelled,
            mode: Mode::Compare,
            natural: true,
            reusable: false,
        },
        |_| {
            Err(Error::State(
                "natural cold preparation unexpectedly entered a mathematical traversal".into(),
            ))
        },
    )
    .await?
    {
        Outcome::Natural(prepared) => Ok(prepared),
        _ => Err(Error::State(
            "natural cold preparation returned non-cold result".into(),
        )),
    }
}

/// Immutable metadata only: no extra source/device/Ready owner is retained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NaturalColdSnapshot {
    pub(crate) prepared_ids: BTreeSet<zerogravity_core::device_events::BufferLeaseId>,
    pub(crate) readback_cohorts:
        Vec<Vec<(zerogravity_core::device_events::BufferLeaseId, usize, usize)>>,
    pub(crate) prepared_state_buffers: usize,
    pub(crate) retained_plan_values: usize,
    pub(crate) retained_source_weight_groups: usize,
    pub(crate) native_images: usize,
    pub(crate) native_head_kernels: usize,
    pub(crate) native_charge_bytes: usize,
    pub(crate) image_admission_bytes: usize,
    pub(crate) resident_bytes: usize,
    pub(crate) future_uninitialized: bool,
}
impl ColdModelSlots {
    pub(crate) fn quarantine_snapshot(&self) -> Result<NaturalColdSnapshot> {
        use zerogravity_core::accelerator::program::ValueId;
        let mut prepared_ids = BTreeSet::new();
        for body in &self.owners.bodies {
            prepared_ids.extend(body.accesses()?.iter().map(|(lease, _)| lease.id()));
        }
        prepared_ids.extend(
            self.owners
                .initial
                .iter()
                .flatten()
                .chain(self.owners.candidates.iter().flatten())
                .flat_map(|domain| domain.leases.iter().map(ResidentLease::id)),
        );
        let native = self.owners.library.inspect_native_retention()?;
        let future_uninitialized = self.owners.bodies.iter().skip(1).all(|body| {
            (1..327).all(|index| {
                body.values()
                    .lease(ValueId(index))
                    .is_ok_and(|lease| !lease.is_initialized())
            })
        }) && self
            .owners
            .candidates
            .iter()
            .skip(1)
            .flatten()
            .flat_map(|domain| &domain.leases)
            .all(|lease| !lease.is_initialized());
        Ok(NaturalColdSnapshot {
            prepared_ids,
            readback_cohorts: self
                .owners
                .frames
                .iter()
                .map(|frame| {
                    frame
                        .reads
                        .iter()
                        .map(|read| (read.lease_id(), read.offset(), read.bytes()))
                        .collect()
                })
                .collect(),
            prepared_state_buffers: self
                .owners
                .initial
                .iter()
                .flatten()
                .chain(self.owners.candidates.iter().flatten())
                .map(|domain| domain.leases.len())
                .sum(),
            retained_plan_values: self.owners.bodies.len() * 327,
            retained_source_weight_groups: self.owners.weights.accesses().len() + 1,
            native_images: native.images,
            native_head_kernels: native.head_kernels,
            native_charge_bytes: native.native_charge_bytes,
            image_admission_bytes: native.image_admission_bytes,
            resident_bytes: self.owners.bodies[0].values().service().resident_bytes()?,
            future_uninitialized,
        })
    }
}

impl PreparedNaturalCold {
    pub fn observer(&self) -> super::super::natural_publication::NaturalObserver {
        super::super::natural_publication::NaturalObserver::new(&self.model)
    }
}
