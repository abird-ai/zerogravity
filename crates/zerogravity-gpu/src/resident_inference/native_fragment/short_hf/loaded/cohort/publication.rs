//! Per-row real Ready, one unchanged physical publication barrier.
use super::*;
use crate::resident_inference::native_fragment::short_hf::completion::committed_feedback;

#[derive(Clone, Debug)]
pub struct ShortHfCohortRowReceipt {
    pub root: RootBinding,
    pub selected_id: u32,
    pub max_bits: u32,
    pub consumed_tokens: usize,
}

/// Counts shared physical work once. Individual rows carry no copied launch
/// count: B request results do not mean B independent physical traversals.
#[derive(Clone, Debug)]
pub struct ShortHfCohortReceipt {
    pub physical_rows: usize,
    pub executed_nodes: usize,
    pub native_launches: usize,
    pub native_projection_launches: usize,
    pub sampler_dispatches: usize,
    pub dispatches: usize,
    pub readback_bytes: usize,
    /// Shared finite-certification metrics, counted once: deferred members,
    /// enabled cohorts, immediate/cohort enqueues and cold metadata upper.
    /// Individual rows keep [0; 5] (no copied launch count per row).
    pub finite_certification: [usize; 5],
    pub rows: Vec<ShortHfCohortRowReceipt>,
}

#[must_use = "dropping a completed cohort discards every unpublished candidate"]
pub struct ShortHfPendingCohort {
    inputs: Arc<DecodeCohortInputs>,
    prepared: Vec<PreparedCall>,
    receipt: ShortHfCohortReceipt,
    _reservation: DecodeCohortReservation,
}
impl ShortHfPendingCohort {
    // Private to the actual native execution child. No public serial/bare Ready
    // constructor can mint this receipt from B1 results or predicted launches.
    pub(super) fn from_completed(
        inputs: Arc<DecodeCohortInputs>,
        prepared: Vec<PreparedCall>,
        receipt: ShortHfCohortReceipt,
        reservation: DecodeCohortReservation,
    ) -> Result<Self> {
        let width = inputs.rows.len();
        if !matches!(width, 4 | 8)
            || prepared.len() != width
            || prepared.capacity() > width
            || receipt.physical_rows != width
            || receipt.executed_nodes != 255
            || receipt.native_launches == 0
            || receipt.dispatches < receipt.native_launches
            || receipt.readback_bytes == 0
            || !receipt.rows.is_empty()
        {
            return Err(Error::State(
                "HF actual complete native cohort receipt differs".into(),
            ));
        }
        let events = prepared[0].inference.ready.event_ids();
        if events.len() != 1 {
            return Err(Error::State(
                "HF cohort original single physical completion".into(),
            ));
        }
        for (row, input) in prepared.iter().zip(&inputs.rows) {
            if !Arc::ptr_eq(&row.input, input)
                || row.inference.attempt.base() != input.attempt().base()
                || row.inference.ready.root_binding() != Some(input.attempt().base())
                || row.inference.ready.event_ids() != events
                || row.receipt.executed_nodes != 255
                || row.receipt.consumed_tokens != input.rows()
            {
                return Err(Error::State(
                    "HF cohort row is not from original submit_cohort/Ready".into(),
                ));
            }
        }
        Ok(Self {
            inputs,
            prepared,
            receipt,
            _reservation: reservation,
        })
    }
    /// Actual completed work, including a cohort later discarded by cancellation.
    pub fn dispatches(&self) -> usize {
        self.receipt.dispatches
    }
    pub fn native_launches(&self) -> usize {
        self.receipt.native_launches
    }
    pub fn native_projection_launches(&self) -> usize {
        self.receipt.native_projection_launches
    }
    pub fn sampler_dispatches(&self) -> usize {
        self.receipt.sampler_dispatches
    }
    pub fn len(&self) -> usize {
        self.prepared.len()
    }
    pub fn is_empty(&self) -> bool {
        self.prepared.is_empty()
    }

    /// None means known completion followed by cancellation: discard this whole
    /// private proposal and regroup healthy rows. It never permits unknown-work,
    /// source, math or stale-root failures to retry as a smaller batch.
    pub fn commit(
        self,
        sessions: &mut [&mut ShortHfSession],
        cancelled: &[&AtomicBool],
    ) -> Result<Option<ShortHfCohortReceipt>> {
        if sessions.len() != self.prepared.len() || cancelled.len() != sessions.len() {
            return Err(Error::Malformed(
                "HF cohort exact publication row count".into(),
            ));
        }
        let mut wall = super::super::performance::Phases::new(
            "b4_publication",
            sessions.first().map_or(0, |s| s.binding().frontier()),
            sessions.len(),
        );
        self.inputs
            .require_current(&sessions.iter().map(|s| &**s).collect::<Vec<_>>())?;
        // Recheck ALL actual roots and packet identities before cancellation can
        // choose a safe known-completion discard. Do not hide a stale sibling.
        let mut tokens = Vec::with_capacity(self.prepared.len());
        for (session, row) in sessions.iter().zip(&self.prepared) {
            tokens.push(row.require_completed_for(&session.original)?);
        }
        wall.mark("root_revalidation");
        if cancelled.iter().any(|c| c.load(Ordering::Acquire)) {
            return Ok(None);
        }
        let loaded = self
            .inputs
            .loaded
            .upgrade()
            .ok_or_else(|| Error::State("HF cohort loaded owner retired".into()))?;
        let mut originals = sessions
            .iter_mut()
            .map(|s| &mut s.original)
            .collect::<Vec<_>>();
        let Self {
            inputs,
            prepared,
            mut receipt,
            _reservation,
        } = self;
        let mut inferences = Vec::with_capacity(prepared.len());
        let mut packets = Vec::with_capacity(prepared.len());
        for row in prepared {
            packets.push((
                row.input,
                row.packet,
                row.receipt.selected_id,
                row.receipt.max_bits,
            ));
            inferences.push(row.inference);
        }
        crate::resident_inference::publish_prepared_cohort(
            &loaded.model,
            inputs.roots(),
            &mut originals,
            &mut inferences,
        )?;
        wall.mark("publish");
        drop(originals);
        // The only publisher above reserved every transition before any swap.
        // No fallible work or device command follows publication.
        receipt.rows = Vec::with_capacity(sessions.len());
        for (((session, (input, packet, id, bits)), token), inference) in
            sessions.iter_mut().zip(packets).zip(tokens).zip(inferences)
        {
            let root = session.binding();
            session.cohort_root = Some(root);
            let consumed_tokens = input.rows();
            session.feedback = Some(committed_feedback(&session.original, input, packet, token));
            receipt.rows.push(ShortHfCohortRowReceipt {
                root,
                selected_id: id,
                max_bits: bits,
                consumed_tokens,
            });
            // Original displaced roots are released here, while the single
            // cohort reservation still prices the whole transient owner set.
            drop(inference);
        }
        drop(_reservation);
        wall.succeed();
        Ok(Some(receipt))
    }
}
