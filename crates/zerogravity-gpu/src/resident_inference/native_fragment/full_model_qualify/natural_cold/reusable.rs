//! Rebind original unused templates; original workspace serializes actual work.
use super::*;

pub(crate) struct PreparedReusableNatural {
    pub(crate) model: Arc<ResidentModel>,
    pub(crate) storage: Arc<ReusableNaturalStorage>,
}

// No strong Model/Session edge: a workspace body may retain this source graph.
pub(crate) struct ReusableNaturalStorage {
    cold: Arc<ColdModelSlots>,
    closing: AtomicBool,
}
struct SessionStaging {
    _source: Arc<OriginalWindowModelSource>,
    _host: [zerogravity_artifacts::source_weights::WindowWorkspace; 2],
}
impl ReusableNaturalStorage {
    pub(crate) fn model(&self) -> Result<Arc<ResidentModel>> {
        self.cold
            .model
            .upgrade()
            .ok_or_else(|| Error::State("reusable original model retired".into()))
    }
    pub(crate) fn source(&self) -> &Arc<OriginalWindowModelSource> {
        &self.cold.owners._source
    }
    /// Fresh ORIGINAL Session root; staging remains admitted through real H2D.
    pub(crate) async fn session(&self, cancelled: Option<&AtomicBool>) -> Result<ResidentSession> {
        not_cancelled(cancelled)?;
        if self.closing.load(Ordering::Acquire) {
            return Err(Error::State("reusable storage is closing".into()));
        }
        let model = self.model()?;
        model.service.ensure_usable()?;
        let host = [
            self.source().source().reserve_workspace(32 * 1024 * 1024)?,
            self.source().source().reserve_workspace(32 * 1024 * 1024)?,
        ];
        let owner = Arc::new(SessionStaging {
            _source: self.source().clone(),
            _host: host,
        });
        let session = model.session_retaining(owner).await?;
        not_cancelled(cancelled)?;
        self.require_open(&session)?;
        Ok(session)
    }
    pub(crate) fn require_open(&self, session: &ResidentSession) -> Result<()> {
        let model = self.model()?;
        if self.closing.load(Ordering::Acquire)
            || self.cold.retirement_started.load(Ordering::Acquire)
            || !Arc::ptr_eq(&model, &session.model)
        {
            return Err(Error::State(
                "reusable storage closed or original model differs".into(),
            ));
        }
        model.service.ensure_usable()
    }
    pub(crate) fn completion_owner(
        self: &Arc<Self>,
    ) -> crate::physical_quarantine::CompletionOwner {
        self.clone()
    }
    /// Caller holds the ORIGINAL fair slot and model.workspace through actual
    /// completion. A fresh state's submission claim is consumed before binding.
    pub(crate) fn bind_call(
        self: &Arc<Self>,
        session: &ResidentSession,
        input: Arc<NaturalCallInput>,
        state: Arc<NaturalCallState>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ResidentProgram> {
        not_cancelled(cancelled)?;
        self.require_open(session)?;
        input.require_session(session)?;
        state.require_current(session)?;
        state.require_reusable_submission_claimed()?;
        if !Arc::ptr_eq(&input, state.input()) || !Arc::ptr_eq(input.source(), self.source()) {
            return Err(Error::State(
                "reusable exact immutable input/state/source differs".into(),
            ));
        }
        let index = input.slot().index();
        let binding = session.bind_text_positions(
            input.source().clone(),
            input.attempt(),
            input.call().request(),
            Some(input.phase_binding().clone()),
            cancelled,
        )?;
        let positions = self.cold.owners._positions[index].bind_natural(binding)?;
        self.cold.templates[index].bind_reusable(
            input,
            state,
            positions,
            self.cold.owners.tied.clone(),
            cancelled,
        )
    }
    pub(crate) fn quarantine_snapshot(&self) -> Result<NaturalColdSnapshot> {
        self.cold.quarantine_snapshot()
    }
    /// Closing stops new work before the original fair/workspace barrier. Source
    /// verifiers acquire their OWN fair slots, so release both barrier locks first.
    pub(crate) async fn verify_source_and_retire(
        self: &Arc<Self>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(usize, usize)> {
        not_cancelled(cancelled)?;
        if self
            .closing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(Error::State(
                "reusable storage retirement is once-only".into(),
            ));
        }
        let model = self.model()?;
        model.service.ensure_usable()?;
        let ticket = model.service.queue().submit(
            model.service.budget(),
            model.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: model.state_bytes as u64,
            }],
            None,
        )?;
        let barrier = ticket.wait().await?;
        let workspace = model.workspace.lock().await;
        model.service.ensure_usable()?;
        if model.service.pending_events()? != 0 {
            return Err(Error::State(
                "reusable source retirement before actual completion".into(),
            ));
        }
        drop(workspace);
        // Unsubmitted slot Drop is normal cancellation, not a manufactured Ready.
        drop(barrier);
        self.cold.retire_quiescent(&model, cancelled).await
    }
}

/// Same genuine cold initialization/native set; no model traversal executes.
pub(crate) async fn prepare_reusable_natural_cold(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    records: [RecordedVllm08Model; 5],
    inputs: Vllm08ModelInputs,
    rotary: Option<RotaryOutcome>,
    cancelled: Option<&AtomicBool>,
) -> Result<PreparedReusableNatural> {
    not_cancelled(cancelled)?;
    require_host_bound(&source, &records)?;
    let prepared = match super::super::qualify_inner(
        device,
        source,
        records,
        inputs,
        Execution {
            rotary,
            cancelled,
            mode: Mode::Compare,
            natural: true,
            reusable: true,
        },
        |_| {
            Err(Error::State(
                "reusable cold setup unexpectedly executed a model".into(),
            ))
        },
    )
    .await?
    {
        Outcome::Natural(prepared) => prepared,
        _ => {
            return Err(Error::State(
                "reusable preparation returned non-cold result".into(),
            ));
        }
    };
    let PreparedNaturalCold {
        model,
        sessions,
        slots,
        state_pool,
    } = prepared;
    // Original cold owners keep source/native/template storage, not authorities.
    // New requests use fresh ORIGINAL SessionAuthority and canonical root owners.
    drop(sessions);
    drop(state_pool);
    Ok(PreparedReusableNatural {
        model,
        storage: Arc::new(ReusableNaturalStorage {
            cold: slots,
            closing: AtomicBool::new(false),
        }),
    })
}

/// Actual cap+1 admission on the NEW derived envelope, before any managed
/// allocation/H2D/native work. Same existing physical pressure implementation.
pub async fn qualify_vllm08_natural_reuse_pressure(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    records: [RecordedVllm08Model; 5],
    inputs: Vllm08ModelInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<super::super::Vllm08ModelFaultOutcome> {
    not_cancelled(cancelled)?;
    require_host_bound(&source, &records)?;
    let mode = Mode::Fault {
        fault: super::super::Vllm08ModelFault::AdmissionPressure,
        probe: Arc::new(super::super::faults::Probe::default()),
    };
    match super::super::qualify_inner(
        device,
        source,
        records,
        inputs,
        Execution {
            rotary: None,
            cancelled,
            mode,
            natural: true,
            reusable: true,
        },
        |_| {
            Err(Error::State(
                "reusable pressure must not enqueue/read back".into(),
            ))
        },
    )
    .await?
    {
        Outcome::Fault(outcome) => Ok(outcome),
        _ => Err(Error::State(
            "reusable pressure returned a mathematical outcome".into(),
        )),
    }
}
