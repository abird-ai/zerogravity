//! One original fair slot, workspace, physical event and grouped publication.
use super::*;
use crate::resident_leases::ResidentReadback;
use crate::resident_program::{ResidentProgram, native_cohort::FullModelCohort};
use std::collections::BTreeMap;
use std::ops::Range;
use zerogravity_core::accelerator::program::{RecordedVllm08Model, StateSlot, ValueId};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

pub struct ShortHfCohortReadback<'a> {
    pub records: &'a [Arc<RecordedVllm08Model>],
    pub raw: &'a [Bytes],
    pub values: &'a BTreeMap<u32, usize>,
    pub sums: &'a BTreeMap<usize, usize>,
    pub head_input: Option<usize>,
    pub prior: &'a [Vec<(StateSlot, Range<usize>)>],
    pub candidate: &'a [Vec<(StateSlot, Range<usize>)>],
    pub selected: &'a [(u32, u32)],
}
impl ShortHfCohortReadback<'_> {
    pub fn rms_sum(&self, row: usize, node: usize) -> Result<&[u8]> {
        let record = self
            .records
            .get(row)
            .ok_or_else(|| Error::State("B4 actual sum row".into()))?;
        let bytes = record.call().rows() * record.call().model_profile().hidden_size() * 4;
        let raw = self
            .sums
            .get(&node)
            .and_then(|index| self.raw.get(*index))
            .ok_or_else(|| Error::State("B4 actual unrounded sum absent".into()))?;
        if raw.len() != 4 * bytes {
            return Err(Error::State("B4 full original F32 sum extent".into()));
        }
        Ok(&raw[row * bytes..(row + 1) * bytes])
    }
    /// Actual dense BF16 bytes from one row of a full physical producer.
    pub fn value(&self, row: usize, id: ValueId) -> Result<&[u8]> {
        let record = self
            .records
            .get(row)
            .ok_or_else(|| Error::State("cohort readback actual row".into()))?;
        let bytes = record
            .plan()
            .values()
            .get(id.0 as usize)
            .ok_or_else(|| Error::State("cohort readback actual ValueId".into()))?
            .capacity_elements()?
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("cohort value readback size".into()))?;
        let raw = self
            .values
            .get(&id.0)
            .and_then(|index| self.raw.get(*index))
            .ok_or_else(|| Error::State("cohort value readback absent".into()))?;
        if raw.len() != bytes * self.records.len() {
            return Err(Error::State(
                "cohort complete physical B producer readback".into(),
            ));
        }
        Ok(&raw[row * bytes..(row + 1) * bytes])
    }
}
/// Cancellation is returned only after all current/source checks and known
/// candidate initialization, before any body submission. Other errors stay fatal.
pub enum ShortHfCohortPreparation {
    Ready(ShortHfPendingCohort),
    Capacity,
    CancelledBeforeLaunch,
}
impl ShortHfCohortPreparation {
    fn legacy(self) -> Result<Option<ShortHfPendingCohort>> {
        match self {
            Self::Ready(pending) => Ok(Some(pending)),
            Self::Capacity => Ok(None),
            Self::CancelledBeforeLaunch => Err(Error::State("short HF request cancelled".into())),
        }
    }
}
type Validation<'a> = &'a mut dyn for<'b> FnMut(ShortHfCohortReadback<'b>) -> Result<()>;
fn domains(
    reads: &mut Vec<ResidentReadback>,
    domains: &[Domain],
) -> Result<Vec<(StateSlot, Range<usize>)>> {
    domains
        .iter()
        .map(|domain| {
            let start = reads.len();
            for lease in &domain.leases {
                reads.push(lease.readback(0, lease.bytes())?);
            }
            Ok((domain.slot, start..reads.len()))
        })
        .collect()
}
impl ShortHfLoaded {
    pub fn native_cohort_width(&self) -> usize {
        if self.cohort.is_some() { 4 } else { 0 }
    }
    /// Prepared INITIAL width; cached-only models retain the genuine N1 path.
    pub fn initial_cohort_width(&self) -> usize {
        if self
            .cohort
            .as_ref()
            .is_some_and(|cold| cold.phases.contains(true))
        {
            4
        } else {
            0
        }
    }
    pub async fn prepare_canonical47_cohort(
        self: &Arc<Self>,
        sessions: &[&ShortHfSession],
        ids: &[u32],
        cancelled: &[&AtomicBool],
        validate: Option<Validation<'_>>,
    ) -> Result<Option<ShortHfPendingCohort>> {
        self.prepare_canonical47_cohort_status(sessions, ids, cancelled, validate)
            .await?
            .legacy()
    }
    pub async fn prepare_canonical47_cohort_status(
        self: &Arc<Self>,
        sessions: &[&ShortHfSession],
        ids: &[u32],
        cancelled: &[&AtomicBool],
        validate: Option<Validation<'_>>,
    ) -> Result<ShortHfCohortPreparation> {
        if self.initial_cohort_width() != 4 {
            return Err(Error::Unsupported(
                "source has no prepared INITIAL B4 phase".into(),
            ));
        }
        let flags = [const { AtomicBool::new(false) }; 4];
        let checks = flags.iter().collect::<Vec<_>>();
        let inputs = DecodeCohortInputs::issue_canonical47(sessions, ids, &checks)?;
        self.prepare_cohort_inner(inputs, sessions, cancelled, validate, None)
            .await
    }
    pub async fn prepare_feedback_cohort(
        self: &Arc<Self>,
        sessions: &[&ShortHfSession],
        cancelled: &[&AtomicBool],
        validate: Option<Validation<'_>>,
    ) -> Result<Option<ShortHfPendingCohort>> {
        self.prepare_feedback_cohort_status(sessions, cancelled, validate)
            .await?
            .legacy()
    }
    pub async fn prepare_feedback_cohort_status(
        self: &Arc<Self>,
        sessions: &[&ShortHfSession],
        cancelled: &[&AtomicBool],
        validate: Option<Validation<'_>>,
    ) -> Result<ShortHfCohortPreparation> {
        let flags = [const { AtomicBool::new(false) }; 4];
        let checks = flags.iter().collect::<Vec<_>>();
        let inputs = DecodeCohortInputs::issue(sessions, &checks)?;
        self.prepare_cohort_inner(inputs, sessions, cancelled, validate, None)
            .await
    }
    pub(super) async fn prepare_cohort_inner(
        self: &Arc<Self>,
        inputs: Arc<DecodeCohortInputs>,
        sessions: &[&ShortHfSession],
        cancelled: &[&AtomicBool],
        mut validate: Option<Validation<'_>>,
        probe: Option<Arc<super::faults::Probe>>,
    ) -> Result<ShortHfCohortPreparation> {
        let cold = self.cohort.as_ref().ok_or_else(|| {
            Error::Unsupported("explicit B4 source/shape admission required".into())
        })?;
        if sessions.len() != 4
            || inputs.rows().len() != 4
            || cancelled.len() != 4
            || sessions
                .iter()
                .any(|session| !Arc::ptr_eq(&session.loaded, self))
        {
            return Err(Error::Unsupported(
                "canonical47 actual B4 source cohort only".into(),
            ));
        }
        inputs.require_current(sessions)?;
        let shape = crate::vllm_cuda_qualification::cohort_native::input_geometry(inputs.rows())?;
        if shape.profile != self.cold.source.model_profile() || !cold.phases.contains(shape.initial)
        {
            return Err(Error::Unsupported(
                "B4 phase is outside this prepared source".into(),
            ));
        }
        if cancelled.iter().any(|flag| flag.load(Ordering::Acquire)) {
            return Ok(ShortHfCohortPreparation::CancelledBeforeLaunch);
        }
        // Real cancellation is checked at the next proven prelaunch boundary;
        // partial initialization must drain to known completion first.
        let flags = [const { AtomicBool::new(false) }; 4];
        let checks = flags.iter().collect::<Vec<_>>();
        let Some(states) =
            DecodeCohortStates::prepare(inputs.clone(), sessions, &checks, &cold.capacity).await?
        else {
            return Ok(ShortHfCohortPreparation::Capacity);
        };
        let ticket = self.service().queue().submit(
            self.service().budget(),
            self.service().owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: (self.model.state_bytes * 4) as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        let workspace = self.model.workspace.lock().await;
        states.require_current(sessions)?;
        // Fresh candidate writes are known-complete, roots/source remain exact,
        // and no body/workspace command has launched. Do not catch generic errors.
        if cancelled.iter().any(|flag| flag.load(Ordering::Acquire)) {
            return Ok(ShortHfCohortPreparation::CancelledBeforeLaunch);
        }
        let initial = (inputs.rows()[0].rows(), inputs.rows()[0].base()) == (47, 0);
        let private_bytes = FullModelCohort::allocation_layout(initial, shape.profile)
            .into_iter()
            .try_fold(0usize, |sum, bytes| {
                sum.checked_add((bytes + 511) & !511)
                    .ok_or_else(|| Error::Limit("B4 complete private allocation preflight".into()))
            })?;
        if self.service().remaining_capacity()? < private_bytes {
            return Ok(ShortHfCohortPreparation::Capacity);
        };
        states.claim_submission(sessions)?;
        let permits = inputs
            .rows()
            .iter()
            .zip(sessions)
            .map(|(input, session)| input.permit(&session.original))
            .collect::<Result<Vec<_>>>()?;
        // All current roots and private allocations precede source launch. This
        // body remains call-local; workspace.program keeps its original N1 body.
        let geometry = cold.workspace.geometry(&inputs)?;
        let gammas = self.cold.values[if initial { 0 } else { 5 }]
            .body
            .cohort_gammas()?;
        let tied = self
            .cold
            .native
            .head()
            .resident_weight()
            .cloned()
            .ok_or_else(|| Error::State("B4 source-held full tied table absent".into()))?;
        slot.begin_submission()?;
        let mut init = self
            .service()
            .begin_initialization(inputs.rows()[0].attempt().base().root())?;
        let body = FullModelCohort::prepare(
            self.service().clone(),
            states.rows(),
            &permits,
            geometry,
            cold.status.clone(),
            cold.cache.clone(),
            self.cold.native.clone(),
            cold.library.clone(),
            self.cold.rotary.clone(),
            &self.cold.parameters,
            gammas,
            validate.is_some(),
            probe.as_ref().map(|p| p.fault),
            tied,
            &mut init,
            None,
        )?;
        drop(init.complete_sync().await?);
        let mut program = ResidentProgram::from_cohort_body(body.clone())?;
        let records = inputs
            .rows()
            .iter()
            .map(|row| row.record().clone())
            .collect::<Vec<_>>();
        let mut reads = body.output_readbacks()?;
        let output_reads = reads.len();
        let mut values = BTreeMap::new();
        let mut sums = BTreeMap::new();
        let mut head_input = None;
        let mut prior = Vec::new();
        let mut candidate = Vec::new();
        if validate.is_some() {
            head_input = Some(reads.len());
            reads.push(body.head_input().readback(0, body.head_input().bytes())?);
            for (index, value) in body.values().iter().enumerate() {
                values.insert(index as u32, reads.len());
                let lease = value.original_lease()?;
                reads.push(lease.readback(0, lease.bytes())?);
            }
            for (&node, lease) in body.sums() {
                sums.insert(node, reads.len());
                reads.push(lease.readback(0, lease.bytes())?);
            }
            for (session, state) in sessions.iter().zip(states.rows()) {
                prior.push(domains(&mut reads, &session.original.domains)?);
                candidate.push(domains(&mut reads, state.candidate())?);
            }
        }
        let readback_bytes = reads.iter().try_fold(0usize, |sum, read| {
            sum.checked_add(read.bytes())
                .ok_or_else(|| Error::Limit("B4 whole readback extent".into()))
        })?;
        if readback_bytes > READBACK_BYTES {
            return Err(Error::Limit(
                "B4 whole readback exceeds actual source admission".into(),
            ));
        }
        let packets = body.packets().to_vec();
        let outputs = sessions
            .iter()
            .zip(states.rows())
            .zip(&packets)
            .map(|((session, state), packet)| {
                let mut outputs = state.outputs();
                outputs.push(packet.clone());
                (session.binding(), outputs)
            })
            .collect::<Vec<_>>();
        if outputs.iter().any(|(_, row)| {
            row.len() != 49
                || row.iter().any(|lease| {
                    !body.accesses().iter().any(|(access, kind)| {
                        access.id() == lease.id()
                            && access.allocation_generation() == lease.allocation_generation()
                            && *kind == zerogravity_core::device_events::DeviceAccess::Write
                    })
                })
        }) {
            return Err(Error::State(
                "B4 every actual48 candidates and fresh packet must be same-flight writes".into(),
            ));
        }
        body.require_current(&permits)?;
        let roots = sessions
            .iter()
            .map(|session| session.binding())
            .collect::<Vec<_>>();
        if let Some(probe) = &probe {
            probe.bind(&body)?;
        }
        let mut flight = self.service().submit_cohort(&roots, body.accesses())?;
        flight.retain_completion_owner(body.clone())?;
        flight.retain_completion_owner(self.cold.clone())?;
        flight.retain_completion_owner(cold.clone())?;
        flight.retain_completion_owner(inputs.clone())?;
        flight.retain_completion_owner(states.clone())?;
        // Submitted serving quanta drain to known completion. Cancellation is
        // checked atomically for all rows by the unchanged publisher afterward.
        let submitted = program.launch_cohort(self.service().client(), &permits, None)?;
        if submitted.logits.is_some()
            || !submitted.proposals.is_empty()
            || submitted
                .cohort_proposals
                .as_ref()
                .is_none_or(|rows| rows.len() != 4 || rows.iter().any(|row| row.len() != 42))
        {
            return Err(Error::State(
                "B4 real grouped terminal42 proposals per row".into(),
            ));
        }
        body.require_finished()?;
        let mut selected = None;
        let (ready, raw) = flight
            .complete_cohort_with_readback(&outputs, &reads, |raw| {
                let samples = body.validate_output_readbacks(&raw[..output_reads])?;
                if validate.is_some() {
                    for ((prior, candidate), input) in
                        prior.iter().zip(&candidate).zip(inputs.rows())
                    {
                        let end = input.base() + input.rows();
                        for ((ps, pr), (cs, cr)) in prior.iter().zip(candidate) {
                            if ps != cs {
                                return Err(Error::State(
                                    "B4 all-row state domain ordering".into(),
                                ));
                            }
                            if matches!(ps, StateSlot::Kv(_)) {
                                for (before, after) in raw[pr.clone()].iter().zip(&raw[cr.clone()])
                                {
                                    if before.len() != 1_048_576
                                        || before.len() != after.len()
                                        || before[end * 512 * 4..] != after[end * 512 * 4..]
                                    {
                                        return Err(Error::State(
                                            "B4 canonical future KV bytes must carry unchanged"
                                                .into(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    if let Some(callback) = validate.as_mut() {
                        callback(ShortHfCohortReadback {
                            records: &records,
                            raw,
                            values: &values,
                            sums: &sums,
                            head_input,
                            prior: &prior,
                            candidate: &candidate,
                            selected: &samples,
                        })?;
                    }
                }
                selected = Some(samples);
                Ok(())
            })
            .await?;
        body.require_current(&permits)?;
        drop(permits);
        slot.finish()?;
        let selected =
            selected.ok_or_else(|| Error::State("B4 complete output sample population".into()))?;
        if ready.len() != 4 || selected.len() != 4 {
            return Err(Error::State(
                "B4 exact ordered real row Ready proofs".into(),
            ));
        }
        let mut prepared = Vec::with_capacity(4);
        for (index, ((ready, packet_lease), (id, bits))) in
            ready.into_iter().zip(packets).zip(selected).enumerate()
        {
            let session = sessions[index];
            let input = inputs.rows()[index].clone();
            let state = &states.rows()[index];
            let packet = raw[index][..]
                .try_into()
                .map_err(|_| Error::State("B4 actual12-byte immutable packet".into()))?;
            let inference = PreparedInference {
                attempt: input.attempt().clone(),
                domains: state.candidate().to_vec(),
                ready,
                readback: InferenceReadback {
                    device_token: Some(ResidentToken {
                        lease: packet_lease,
                        producer: None,
                        id,
                        max_bits: bits,
                    }),
                    logit_steps: 1,
                    ..Default::default()
                },
            };
            session.original.validate_prepared(&inference)?;
            let receipt = ShortHfCallReceipt {
                root: session.binding(),
                selected_id: id,
                max_bits: bits,
                executed_nodes: 255,
                native_launches: 0,
                dispatches: 0,
                finite_certification: [0; 5],
                transfer_retirement: [0; 5],
                readback_bytes: 0,
                consumed_tokens: input.rows(),
            };
            prepared.push(PreparedCall {
                inference,
                input,
                packet,
                receipt,
            });
        }
        let receipt = ShortHfCohortReceipt {
            physical_rows: 4,
            executed_nodes: 255,
            native_launches: body.native_launches(),
            native_projection_launches: body.projection_launches(),
            sampler_dispatches: body.sampler_dispatches(),
            dispatches: submitted.dispatches,
            readback_bytes,
            rows: Vec::new(),
        };
        drop(raw);
        drop(submitted);
        drop(program);
        drop(body);
        drop(workspace);
        // All call-local current priors have dropped before this reservation can
        // escape with only the actual candidate owners and fresh row packets.
        let reservation = Arc::try_unwrap(states)
            .map_err(|_| {
                Error::State("B4 current body retained priors after known completion".into())
            })?
            .into_reservation();
        Ok(ShortHfCohortPreparation::Ready(
            ShortHfPendingCohort::from_completed(inputs, prepared, receipt, reservation)?,
        ))
    }
}
