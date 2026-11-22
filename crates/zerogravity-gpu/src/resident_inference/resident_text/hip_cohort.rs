//! Independent cached-decode cohorts for HIP text execution.
//! Only projections share multi-row physical work in the first cut; every stateful
//! operator still addresses each session's original private state.
use super::*;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};

impl ResidentTextModel {
    pub async fn prepare_decode_cohort_admitted(
        self: &Arc<Self>,
        rows: &[(&ResidentTextSession, &BoundInvocation)],
    ) -> Result<ResidentTextCohortPreparation> {
        if !matches!(self.cohort_width, 4 | 8)
            || !matches!(rows.len(), 2 | 4 | 8)
            || rows.len() > self.cohort_width
            || self.contract != NumericalContract::TorchCpuAvx2Bf16MixedV1
        {
            return Err(Error::Unsupported(
                "source HIP decode cohort requires an admitted width".into(),
            ));
        }
        let mut requests = Vec::with_capacity(rows.len());
        let mut scopes = Vec::with_capacity(rows.len());
        let mut sessions = Vec::with_capacity(rows.len());
        let frontier = rows[0].0.binding().frontier();
        for (session, invocation) in rows {
            if !Arc::ptr_eq(&session.model, self)
                || !invocation.input().emit_logits()
                || invocation.input().token_count() != 1
                || invocation.input().base() as u64 != frontier
                || session.binding().frontier() != frontier
            {
                return Err(Error::State(
                    "source HIP decode cohort row/root differs".into(),
                ));
            }
            scopes.push(self.reader_scope(CALL_HOST_BYTES)?);
            requests.push(invocation.materialize(InvocationPhase::Decode)?);
            sessions.push(&session.inner);
        }
        let Some(inner) = self
            .inner
            .prepare_text_execution_decode_cohort(&sessions, &requests, &scopes)
            .await?
        else {
            return Ok(ResidentTextCohortPreparation::Capacity);
        };
        Ok(ResidentTextCohortPreparation::Ready(
            ResidentTextCohortPending { inner, scopes },
        ))
    }
}

impl ResidentModel {
    async fn prepare_text_execution_decode_cohort(
        self: &Arc<Self>,
        sessions: &[&ResidentSession],
        requests: &[ForwardPlan],
        scopes: &[Arc<ReaderScope>],
    ) -> Result<Option<PreparedResidentCohort>> {
        let width = sessions.len();
        if !matches!(width, 2 | 4 | 8) || requests.len() != width || scopes.len() != width {
            return Err(Error::Unsupported(
                "HIP text-execution cohort requires two, four or eight rows".into(),
            ));
        }
        self.service.ensure_usable()?;
        let bases = sessions
            .iter()
            .map(|session| session.binding())
            .collect::<Vec<_>>();
        let roots = ResidentBatchRows::new(&bases, 1)?;
        let frontier = roots.frontier().ok_or_else(|| {
            Error::Unsupported("HIP decode cohort requires an aligned frontier".into())
        })?;
        if roots.source() != self.service.source_identity()
            || sessions
                .iter()
                .any(|session| !Arc::ptr_eq(&session.model, self))
        {
            return Err(Error::State(
                "HIP decode cohort model/source differs".into(),
            ));
        }
        let calls = requests
            .iter()
            .map(|request| {
                request.text_invocation().ok_or_else(|| {
                    Error::State("HIP decode cohort materialized call absent".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if calls.iter().zip(&bases).any(|(call, base)| {
            call.phase() != InvocationPhase::Decode
                || call.tokens().len() != 1
                || call.tokens().base() as u64 != frontier
                || base.frontier() != frontier
                || call.source() != base.source()
                || call.tokens().capacity() != self.capacity
                || call.tokens().vocabulary() as u64 != self.vocab
        }) {
            return Err(Error::Unsupported(
                "HIP decode cohort requires aligned cached rows".into(),
            ));
        }

        let mut attempts = Vec::with_capacity(width);
        for session in sessions {
            self.validate_domains(&session.domains, true)?;
            attempts.push(session.authority.begin_advance(1)?);
        }
        let staging_bytes = self
            .state_bytes
            .checked_mul(width)
            .ok_or_else(|| Error::Limit("HIP decode cohort staging bytes".into()))?;
        let logits_bytes = (self.vocab as usize)
            .checked_mul(width)
            .and_then(|words| words.checked_mul(4))
            .ok_or_else(|| Error::Limit("HIP decode cohort logit bytes".into()))?;

        let allocation = self.allocation.lock().await;
        let mut candidates = Vec::with_capacity(width);
        for session in sessions {
            let position = usize::try_from(session.binding().frontier())
                .map_err(|_| Error::Limit("HIP decode cohort frontier".into()))?;
            match self
                .forward_candidate_state(&session.domains, 1, position)
                .await
            {
                Ok(candidate) => candidates.push(candidate),
                Err(Error::Limit(reason))
                    if matches!(
                        reason.as_str(),
                        "resident pool busy" | "resident pool exhausted"
                    ) =>
                {
                    return Ok(None);
                }
                Err(error) => return Err(error),
            }
        }
        let bindings = candidates
            .iter()
            .map(|candidate| candidate.bindings().to_vec())
            .collect::<Vec<_>>();
        let projection_bytes = {
            let workspace = self.workspace.lock().await;
            if !workspace.program.is_forward_source() {
                return Err(Error::State(
                    "HIP decode cohort reached another resident model".into(),
                ));
            }
            workspace
                .program
                .text_execution_decode_cohort_projection_bytes(width)?
        };
        let transient_bytes = projection_bytes
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(logits_bytes))
            .ok_or_else(|| Error::Limit("HIP decode cohort transient bytes".into()))?;
        if transient_bytes > self.service.remaining_capacity()? {
            return Ok(None);
        }
        let logits = self
            .service
            .allocate_uninitialized(logits_bytes, BufferRole::Output)?;
        // Multi-row projection scratch is transient. The ordinary model retains
        // its compact singleton arena when no cohort is live.
        let projection = [
            self.service
                .allocate_uninitialized(projection_bytes, BufferRole::Activation)?,
            self.service
                .allocate_uninitialized(projection_bytes, BufferRole::Scratch)?,
            self.service
                .allocate_uninitialized(projection_bytes, BufferRole::Scratch)?,
        ];

        let ticket = self.service.queue().submit(
            self.service.budget(),
            ResourceOwner {
                model: self.service.owner().model.clone(),
                session: format!("text-decode-cohort-{}", bases[0].session().get()),
            },
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: staging_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        let workspace = self.workspace.lock().await;
        if !workspace.program.is_forward_source() {
            return Err(Error::State(
                "HIP decode cohort reached another resident model".into(),
            ));
        }
        for (request, binding) in requests.iter().zip(&bindings) {
            workspace.program.validate_forward_source_request(request)?;
            workspace
                .program
                .validate_text_execution_states(request, binding)?;
        }
        let mut accesses = workspace.logit_accesses.clone();
        accesses.push((logits.clone(), DeviceAccess::Write));
        accesses.extend(
            projection
                .iter()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Write)),
        );
        for (session, candidate) in sessions.iter().zip(&candidates) {
            accesses.extend(candidate.state_accesses(&session.domains)?);
        }
        let accesses = dedupe_resident_accesses(accesses)?;
        drop(allocation);

        slot.begin_submission()?;
        let mut flight = self.service.submit_cohort(&bases, &accesses)?;
        for scope in scopes {
            flight.retain_completion_owner(scope.clone())?;
        }
        let submitted = workspace
            .program
            .launch_text_execution_decode_cohort(
                &mut flight,
                requests,
                &bindings,
                [&projection[0], &projection[1], &projection[2]],
                &logits,
            )
            .await?;
        if submitted.fused_dispatches == 0 {
            return Err(Error::State(
                "HIP decode cohort produced no shared projection work".into(),
            ));
        }

        let reads = vec![
            lease_for(&accesses, &submitted.status)?.readback(0, 4)?,
            logits.readback(0, logits_bytes)?,
        ];
        let outputs = bases
            .iter()
            .copied()
            .zip(&candidates)
            .map(|(base, candidate)| Ok((base, candidate.write_outputs()?)))
            .collect::<Result<Vec<_>>>()?;
        let (ready, read) = flight
            .complete_cohort_with_readback(&outputs, &reads, |data| {
                if !all_zero(&data[0]) {
                    return Err(Error::State("HIP decode cohort status refused".into()));
                }
                if words(&data[1])?.iter().any(|value| !value.is_finite()) {
                    return Err(Error::State("HIP decode cohort nonfinite logits".into()));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;

        let logits = words(&read[1])?;
        let row_words = self.vocab as usize;
        let mut prepared = Vec::with_capacity(width);
        for (row, ((attempt, candidate), ready)) in
            attempts.into_iter().zip(candidates).zip(ready).enumerate()
        {
            let domains = candidate.into_domains()?;
            self.validate_domains(&domains, true)?;
            prepared.push(PreparedInference {
                attempt,
                domains,
                derived: Vec::new(),
                ready,
                readback: InferenceReadback {
                    logits: logits[row * row_words..(row + 1) * row_words].to_vec(),
                    dispatches: usize::from(row == 0) * submitted.dispatches,
                    bytes: row_words * 4 + usize::from(row == 0) * 4,
                    chunk_bytes: row_words * 4 + usize::from(row == 0) * 4,
                    state_only_steps: 0,
                    logit_steps: 1,
                    ..Default::default()
                },
            });
        }
        Ok(Some(PreparedResidentCohort {
            model: Arc::clone(self),
            roots,
            prepared,
            dispatches: submitted.dispatches,
            fused_dispatches: submitted.fused_dispatches,
            native_greedy_dispatches: 0,
        }))
    }
}
