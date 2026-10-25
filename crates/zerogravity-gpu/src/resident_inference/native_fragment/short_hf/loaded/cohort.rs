//! Actual per-request issuers for one physical cohort operation.
use super::*;
use crate::resident_batch::ResidentBatchRows;
mod publication;
mod states;
mod workspace;
pub use publication::{ShortHfCohortReceipt, ShortHfCohortRowReceipt, ShortHfPendingCohort};
pub(crate) use states::DecodeCohortStates;
pub(crate) use workspace::DecodeCohortWorkspace;

/// No public mint from IDs/positions. Every row retains its original computed
/// feedback, source-issued record, attempt and independent prior root.
pub(crate) struct DecodeCohortInputs {
    loaded: Weak<ShortHfLoaded>,
    roots: ResidentBatchRows,
    rows: Box<[Arc<ShortHfCallInput>]>,
    _metadata: WindowWorkspace,
}
impl DecodeCohortInputs {
    pub(crate) fn issue(
        sessions: &[&ShortHfSession],
        cancelled: &[&AtomicBool],
    ) -> Result<Arc<Self>> {
        Self::issue_inner(sessions, cancelled, None)
    }
    /// Initial roots come from each actual source-issued prompt call. The
    /// original N1 record is a logical row, never permission for B-native math.
    pub(crate) fn issue_canonical47(
        sessions: &[&ShortHfSession],
        ids: &[u32],
        cancelled: &[&AtomicBool],
    ) -> Result<Arc<Self>> {
        if ids.len() != 47 {
            return Err(Error::Unsupported("HF canonical cohort prompt47".into()));
        }
        Self::issue_inner(sessions, cancelled, Some(ids))
    }
    fn issue_inner(
        sessions: &[&ShortHfSession],
        cancelled: &[&AtomicBool],
        prompt: Option<&[u32]>,
    ) -> Result<Arc<Self>> {
        if !matches!(sessions.len(), 4 | 8) || cancelled.len() != sessions.len() {
            return Err(Error::Malformed(
                "HF decode cohort actual4/8 requests".into(),
            ));
        }
        let first = sessions[0];
        first.loaded.service().ensure_usable()?;
        if first.loaded.closed.load(Ordering::Acquire)
            || first.loaded.closing.load(Ordering::Acquire)
        {
            return Err(Error::State("HF decode cohort source closing".into()));
        }
        let mut roots = Vec::with_capacity(sessions.len());
        for (session, flag) in sessions.iter().zip(cancelled) {
            request_not_cancelled(Some(flag))?;
            if !Arc::ptr_eq(&session.loaded, &first.loaded)
                || !Arc::ptr_eq(&session.original.model, &first.original.model)
                || match prompt {
                    Some(_) => session.binding().frontier() != 0 || session.feedback.is_some(),
                    None => {
                        session.binding().frontier() == 0
                            || session.feedback.is_none()
                            || !session.cached_cohort_eligible()
                    }
                }
            {
                return Err(Error::State(
                    "HF cohort foreign model/uncommitted feedback".into(),
                ));
            }
            roots.push(session.binding());
        }
        let row_elements = first
            .loaded
            .source()
            .model_profile()
            .hidden_size()
            .checked_mul(prompt.map_or(1, <[u32]>::len))
            .ok_or_else(|| Error::Limit("HF cohort actual per-request BF16 row extent".into()))?;
        let roots = ResidentBatchRows::bf16(&roots, row_elements)?;
        let metadata = first
            .loaded
            .source()
            .source()
            .source()
            .reserve_workspace(64 * 1024)?;
        let mut rows: Vec<Arc<ShortHfCallInput>> = Vec::with_capacity(sessions.len());
        for (session, flag) in sessions.iter().zip(cancelled) {
            let input = if let Some(ids) = prompt {
                session.original.bind_short_hf_prompt(
                    session.loaded.cold.bindings.clone(),
                    ids,
                    Some(flag),
                )?
            } else {
                session.original.bind_short_hf_feedback(
                    session.loaded.cold.bindings.clone(),
                    session.feedback.clone().ok_or_else(|| {
                        Error::State("HF cohort original feedback disappeared".into())
                    })?,
                    Some(flag),
                )?
            };
            // Logical time stays per request. B is a separate physical axis;
            // neither the span nor its original mathematical plan is retagged.
            if input.rows() != prompt.map_or(1, <[u32]>::len)
                || input.base() as u64 != session.binding().frontier()
                || prompt.is_some_and(|ids| input.ids() != ids)
                || input.record().plan().nodes().len() != 255
                || input.record().plan().values().len() != 327
                || !Arc::ptr_eq(input.source().source(), first.loaded.source().source())
            {
                return Err(Error::State(
                    "HF cohort original source/record/time axis".into(),
                ));
            }
            input.require_session(&session.original)?;
            if let Some(first) = rows.first() {
                require_same_cohort_recipe(first, &input)?;
            }
            rows.push(input);
        }
        let result = Arc::new(Self {
            loaded: Arc::downgrade(&first.loaded),
            roots,
            rows: rows.into_boxed_slice(),
            _metadata: metadata,
        });
        result.require_current(sessions)?;
        Ok(result)
    }
    pub(crate) fn require_current(&self, sessions: &[&ShortHfSession]) -> Result<()> {
        let loaded = self
            .loaded
            .upgrade()
            .ok_or_else(|| Error::State("HF cohort loaded source retired".into()))?;
        if self.roots.word_bytes() != 2
            || sessions.len() != self.rows.len()
            || sessions.iter().any(|s| !Arc::ptr_eq(&s.loaded, &loaded))
        {
            return Err(Error::State(
                "HF cohort source/current row owner changed".into(),
            ));
        }
        self.roots
            .require_current(&sessions.iter().map(|s| s.binding()).collect::<Vec<_>>())?;
        for (input, session) in self.rows.iter().zip(sessions) {
            input.require_session(&session.original)?;
        }
        loaded.service().ensure_usable()
    }
    pub(crate) fn roots(&self) -> &ResidentBatchRows {
        &self.roots
    }
    pub(crate) fn rows(&self) -> &[Arc<ShortHfCallInput>] {
        &self.rows
    }
}

// The source constructor still records each real N1 request independently.
// Ignore only its current issuer/IDs/base; never infer a temporal B-row span.
fn require_same_cohort_recipe(a: &ShortHfCallInput, b: &ShortHfCallInput) -> Result<()> {
    use zerogravity_core::accelerator::program::PlanOperation;
    let (ar, br) = (a.record(), b.record());
    let (ap, bp) = (ar.plan(), br.plan());
    let mode = if a.rows() == 47 && a.base() == 0 {
        VllmBf16Mode::Prefill
    } else if a.rows() == 1 {
        VllmBf16Mode::Decode
    } else {
        return Err(Error::Unsupported(
            "HF cohort canonical47 or cached row".into(),
        ));
    };
    if a.rows() != b.rows()
        || a.scope().call().span().mode() != mode
        || b.scope().call().span().mode() != mode
        || (mode == VllmBf16Mode::Prefill && (b.base() != 0 || a.ids() != b.ids()))
        || !Arc::ptr_eq(a.source().source(), b.source().source())
        || a.call().source() != b.call().source()
        || a.call().model_profile() != b.call().model_profile()
        || ap.numerical_profile() != bp.numerical_profile()
        || ap.values() != bp.values()
        || ap.precisions() != bp.precisions()
        || ap.required_domains() != bp.required_domains()
        || ap.proposed_domains() != bp.proposed_domains()
        || ar.normalized() != br.normalized()
        || ar.logits() != br.logits()
        || ar.boundaries() != br.boundaries()
        || ap.nodes().len() != 255
        || bp.nodes().len() != 255
        || ap.values().len() != 327
        || bp.values().len() != 327
    {
        return Err(Error::State(
            "HF cohort original source/N1/SSA geometry differs".into(),
        ));
    }
    if ar.shares_prepared_semantics(br) {
        return Ok(());
    }
    for (index, (a, b)) in ap.nodes().iter().zip(bp.nodes()).enumerate() {
        if a.inputs != b.inputs || a.outputs != b.outputs || a.state != b.state {
            return Err(Error::State(
                "HF cohort changes original SSA/state edges".into(),
            ));
        }
        match (&a.operation, &b.operation) {
            (
                PlanOperation::TypedVllmBf16Embedding { descriptor: a },
                PlanOperation::TypedVllmBf16Embedding { descriptor: b },
            ) if index == 0 => {
                a.validate()?;
                b.validate()?;
                if a.tensor() != b.tensor() {
                    return Err(Error::State("HF cohort embedding source differs".into()));
                }
            }
            (
                PlanOperation::TypedVllm08 {
                    descriptor: a,
                    state: sa,
                },
                PlanOperation::TypedVllm08 {
                    descriptor: b,
                    state: sb,
                },
            ) => {
                a.validate()?;
                b.validate()?;
                if a.kind() != b.kind()
                    || a.layer() != b.layer()
                    || a.matrices() != b.matrices()
                    || a.parameters() != b.parameters()
                    || a.input_elements() != b.input_elements()
                    || a.output_elements() != b.output_elements()
                    || a.selected_row() != b.selected_row()
                    || sa != sb
                {
                    return Err(Error::State(
                        "HF cohort source operator/role differs".into(),
                    ));
                }
            }
            _ => {
                return Err(Error::State(
                    "HF cohort introduces another mathematical operation".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Real root owners retained at the prepare/complete/publication boundary.
/// This is an admission equation; actual source layouts still determine bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodeCohortRootBound {
    pub(crate) current: usize,
    pub(crate) candidates: usize,
    pub(crate) external_pending: usize,
    pub(crate) snapshots: usize,
    pub(crate) displaced: usize,
    pub(crate) total: usize,
}

/// One preparing/completed native cohort may coexist with the two unchanged
/// standalone proposals. Retain this reservation with Pending after completion.
/// The physical flight/root pool remain the allocation/completion authorities.
pub(crate) struct DecodeCohortCapacity(Arc<AtomicUsize>);
pub(crate) struct DecodeCohortReservation {
    _reservation: CountReservation,
}
impl DecodeCohortCapacity {
    pub(crate) fn new() -> Self {
        Self(Arc::new(AtomicUsize::new(0)))
    }
    pub(crate) fn try_reserve(
        &self,
        service: &ResidentService,
    ) -> Result<Option<DecodeCohortReservation>> {
        // Fenced work must refuse before claiming a slot retained by quarantine.
        service.ensure_usable()?;
        match self
            .0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count == 0).then_some(1)
            }) {
            Ok(_) => Ok(Some(DecodeCohortReservation {
                _reservation: CountReservation(self.0.clone()),
            })),
            Err(_) => Ok(None),
        }
    }
    pub(crate) fn pending(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }
}
impl DecodeCohortRootBound {
    pub(crate) fn derive(active: usize, physical_width: usize) -> Result<Self> {
        if !(1..=8).contains(&active)
            || !matches!(physical_width, 1 | 4 | 8)
            || physical_width > active
        {
            return Err(Error::Limit(
                "HF actual active/batch width admission".into(),
            ));
        }
        // ONE actual DecodeCohortCapacity reservation, retained until proposal
        // drop/commit. Existing external proposals/snapshots/displaced roots
        // remain priced; a second completed8-row Pending is not admitted.
        let (external_pending, snapshots, displaced) = (2, 2, 2);
        let candidates = if physical_width == 1 {
            0
        } else {
            physical_width
        };
        let total = active
            .checked_add(candidates)
            .and_then(|n| n.checked_add(external_pending + snapshots + displaced))
            .ok_or_else(|| Error::Limit("HF cohort root bound overflow".into()))?;
        Ok(Self {
            current: active,
            candidates,
            external_pending,
            snapshots,
            displaced,
            total,
        })
    }
    pub(crate) fn bytes(&self, source_root_layout: &[usize]) -> Result<usize> {
        if self.current + self.candidates + self.external_pending + self.snapshots + self.displaced
            != self.total
            || source_root_layout.len() != 48
            || source_root_layout.contains(&0)
        {
            return Err(Error::State(
                "HF cohort exact48 original root planes".into(),
            ));
        }
        source_root_layout
            .iter()
            .try_fold(0usize, |n, bytes| {
                n.checked_add(*bytes)
                    .ok_or_else(|| Error::Limit("HF root extent sum".into()))
            })?
            .checked_mul(self.total)
            .ok_or_else(|| Error::Limit("HF cohort actual root bytes overflow".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simultaneous_candidates_never_borrow_existing_pending_or_snapshot_credit() -> Result<()> {
        assert_eq!(DecodeCohortRootBound::derive(8, 1)?.total, 14);
        let four = DecodeCohortRootBound::derive(8, 4)?;
        let eight = DecodeCohortRootBound::derive(8, 8)?;
        assert_eq!(
            (
                four.current,
                four.candidates,
                four.external_pending,
                four.snapshots,
                four.displaced
            ),
            (8, 4, 2, 2, 2)
        );
        assert_eq!((four.total, eight.total), (18, 22));
        let layout = [512usize; 48];
        assert_eq!(eight.bytes(&layout)?, 22 * 48 * 512);
        assert!(eight.bytes(&[512; 47]).is_err());
        assert!(eight.bytes(&[usize::MAX; 48]).is_err());
        for (active, width) in [(0, 4), (8, 2), (4, 8), (9, 8), (8, 16)] {
            assert!(DecodeCohortRootBound::derive(active, width).is_err());
        }
        Ok(())
    }
    #[test]
    fn completed_cohort_pending_keeps_its_sole_reservation_until_drop() -> Result<()> {
        let count = Arc::new(AtomicUsize::new(0));
        let reservation = CountReservation::new(count.clone(), 1)?;
        assert!(CountReservation::new(count.clone(), 1).is_err());
        // Completing device work does not drop the caller's Pending proposal.
        assert_eq!(count.load(Ordering::Acquire), 1);
        drop(reservation);
        assert_eq!(count.load(Ordering::Acquire), 0);
        drop(CountReservation::new(count.clone(), 1)?);
        assert_eq!(count.load(Ordering::Acquire), 0);
        Ok(())
    }
}

mod execution;
mod faults;
use crate::resident_program::native_cohort::CohortLibrary;
pub use execution::{ShortHfCohortPreparation, ShortHfCohortReadback};
pub use faults::{ShortHfCohortFault, ShortHfCohortFaultOutcome, ShortHfCohortFaultRetention};

pub(super) const READBACK_BYTES: usize = 640 * 1024 * 1024;

/// Root-free B4 owners on the same original source/client/native context.
pub(super) struct ColdCohort {
    pub(super) capacity: DecodeCohortCapacity,
    phases: crate::vllm_cuda_qualification::cohort_native::CohortPhases,
    workspace: Arc<DecodeCohortWorkspace>,
    library: Arc<CohortLibrary>,
    cache: ResidentLease,
    status: ResidentLease,
    // Both execution paths hold model.workspace through completed readback.
    _shared_readback: Arc<Vec<WindowWorkspace>>,
    _readback: Vec<WindowWorkspace>,
}
impl ColdCohort {
    pub(super) fn prepare_code(
        &self,
        source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
        queue: &cubecl::cuda::OwnedCudaQueue,
    ) -> Result<usize> {
        self.library.prepare_code(source, queue)
    }
    /// # Safety
    /// Caller retains this complete cohort on the original armed retirement
    /// flight after quiescence; no prepare/enqueue can race or retry retirement.
    pub(super) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        // SAFETY: original close barrier and flight remain the authority.
        unsafe { self.library.retire_after_physical_completion() }
    }
    pub(super) fn initialization_host_bytes() -> Result<usize> {
        // Owned zero payload plus the original backend pinned/fallback staging.
        crate::resident_program::native_cohort::CACHE_BYTES
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("B4 cold cache staging extent".into()))
    }
    pub(super) async fn prepare(
        cold: &Arc<ColdModel>,
        scopes: &[Arc<ShortHfModelBinding>],
        service: Arc<ResidentService>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        request_not_cancelled(cancel)?;
        let workspace =
            DecodeCohortWorkspace::prepare(cold.source.clone(), service.clone(), scopes)?;
        let shared_readback = cold.values[0].body.cohort_readback_owner();
        let shared_bytes = shared_readback.iter().try_fold(0usize, |sum, ticket| {
            sum.checked_add(ticket.bytes())
                .ok_or_else(|| Error::Limit("B4 existing readback owner extent".into()))
        })?;
        if shared_bytes != cold.resources.readback_reservation_bytes {
            return Err(Error::State(
                "B4 actual shared readback admission changed".into(),
            ));
        }
        let additional = READBACK_BYTES.checked_sub(shared_bytes).ok_or_else(|| {
            Error::State("B4 shared readback exceeds full diagnostic extent".into())
        })?;
        let readback = inventory::root_host_admission(&cold.source, additional)?;
        let staging = Arc::new(inventory::root_host_admission(
            &cold.source,
            Self::initialization_host_bytes()?,
        )?);
        let mut init = service.begin_initialization(0)?;
        // Retain admission before any H2D; unknown completion retains it too.
        init.retain_completion_owner(staging)?;
        let cache = init.upload_zeroed(
            crate::resident_program::native_cohort::CACHE_BYTES,
            BufferRole::RecurrentState,
        )?;
        // Only complete_sync's real upload event may initialize this cache.
        // Per-layer state transport still supplies each actual current root.
        let status = init.upload_bytes(&[0u8; 4], BufferRole::Scratch)?;
        let library = CohortLibrary::prepare(
            cold.native.clone(),
            scopes,
            service.clone(),
            &mut init,
            cancel,
        )?;
        let result = Arc::new(Self {
            capacity: DecodeCohortCapacity::new(),
            phases: crate::vllm_cuda_qualification::cohort_native::CohortPhases::for_profile(
                cold.source.model_profile(),
            ),
            workspace,
            library,
            cache,
            status,
            _shared_readback: shared_readback,
            _readback: readback,
        });
        init.retain_completion_owner(result.clone())?;
        drop(init.complete_sync().await?);
        Ok(result)
    }
}

impl ShortHfSession {
    /// Genuine N1 or B4 feedback, current independent root and observed cached
    /// span. No lineage stamp is manufactured by this admission predicate.
    pub fn cached_cohort_eligible(&self) -> bool {
        (47..=53).contains(&self.binding().frontier())
            && self.feedback.as_ref().is_some_and(|packet| {
                packet.workload.prompt_rows() == 47 && packet.root == self.binding()
            })
    }
    /// Stamped only by successful original all-row cohort publication. Equal
    /// IDs/frontiers or provisional/native execution never create this marker.
    pub fn cohort_feedback_eligible(&self) -> bool {
        self.cohort_root == Some(self.binding())
            && self.feedback.as_ref().is_some_and(|packet| {
                packet.workload.prompt_rows() == 47 && packet.root == self.binding()
            })
    }
}
