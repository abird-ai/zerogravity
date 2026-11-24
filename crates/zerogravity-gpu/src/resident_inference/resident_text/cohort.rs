//! Independent-session native cohorts for the ordinary CUDA source path.
//! Prefill chunks may publish state only or terminal readout; decode retains
//! each session's private committed root while sharing qualified physical work.
use super::*;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};

impl ResidentTextModel {
    /// Compatible native cached-decode calls. Stateful CUDA kernels may internally
    /// split a wider cohort while preserving each session's private frontier/root.
    pub async fn prepare_greedy_decode_cohort_admitted(
        self: &Arc<Self>,
        rows: &[(&ResidentTextSession, &BoundInvocation, &ResidentToken)],
    ) -> Result<ResidentTextCohortPreparation> {
        if !matches!(self.admission.cohort_width, 4 | 8)
            || !matches!(rows.len(), 2 | 4 | 8)
            || rows.len() > self.admission.cohort_width
            || self.contract != NumericalContract::CudaSm86Bf16ReferenceV1
        {
            return Err(Error::Unsupported(
                "source CUDA decode cohort requires a qualified width under the admitted model"
                    .into(),
            ));
        }
        let mut requests = Vec::with_capacity(rows.len());
        let mut scopes = Vec::with_capacity(rows.len());
        let mut sessions = Vec::with_capacity(rows.len());
        for (session, invocation, token) in rows {
            if !Arc::ptr_eq(&session.model, self)
                || !invocation.input().emit_logits()
                || invocation.input().token_count() != 1
                || invocation.input().base() as u64 != session.binding().frontier()
                || invocation.input().ids() != [token.id]
                || token.producer != Some(session.binding())
                || token.lease.bytes() != 12
                || !Arc::ptr_eq(token.lease.service(), &self.inner.service)
            {
                return Err(Error::State(
                    "source CUDA decode cohort row/token/root differs".into(),
                ));
            }
            let call = self.bind_invocation(invocation, InvocationPhase::Decode, false)?;
            scopes.push(self.reader_scope(CALL_HOST_BYTES)?);
            requests.push(call.materialize()?);
            sessions.push(&session.inner);
        }
        let Some(inner) = self
            .inner
            .prepare_forward_source_decode_cohort(&sessions, &requests, &scopes)
            .await?
        else {
            return Ok(ResidentTextCohortPreparation::Capacity);
        };
        Ok(ResidentTextCohortPreparation::Ready(
            ResidentTextCohortPending { inner, scopes },
        ))
    }

    /// Reduction-compatible native prompt chunks at one committed frontier.
    /// Intermediate chunks are state-only; terminal chunks may produce greedy
    /// readout. Compatibility is derived from the source plan, not model names.
    pub(super) async fn prepare_cuda_prefill_cohort_admitted(
        self: &Arc<Self>,
        rows: &[(&ResidentTextSession, &BoundInvocation)],
    ) -> Result<ResidentTextCohortPreparation> {
        if !matches!(self.admission.cohort_width, 4 | 8)
            || !matches!(rows.len(), 2 | 4 | 8)
            || rows.len() > self.admission.cohort_width
            || self.contract != NumericalContract::CudaSm86Bf16ReferenceV1
        {
            return Err(Error::Unsupported(
                "source CUDA prefill cohort requires a qualified width under the admitted model"
                    .into(),
            ));
        }
        let prompt_rows = rows[0].1.input().token_count();
        let emit_logits = rows[0].1.input().emit_logits();
        if !(1..=64).contains(&prompt_rows)
            || rows.iter().any(|(_, invocation)| {
                invocation.input().emit_logits() != emit_logits
                    || !(1..=64).contains(&invocation.input().token_count())
                    || !self.prefill_rows_compatible(prompt_rows, invocation.input().token_count())
            })
        {
            return Err(Error::Unsupported(
                "source CUDA prefill cohort requires reduction-compatible prompt chunks".into(),
            ));
        }
        let mut requests = Vec::with_capacity(rows.len());
        let mut scopes = Vec::with_capacity(rows.len());
        let mut sessions = Vec::with_capacity(rows.len());
        for (session, invocation) in rows {
            if !Arc::ptr_eq(&session.model, self) {
                return Err(Error::State(
                    "source CUDA prefill cohort session/model differs".into(),
                ));
            }
            let call = self.bind_invocation(invocation, InvocationPhase::Prefill, false)?;
            scopes.push(self.reader_scope(CALL_HOST_BYTES)?);
            requests.push(call.materialize()?);
            sessions.push(&session.inner);
        }
        let Some(inner) = self
            .inner
            .prepare_forward_source_prefill_cohort(&sessions, &requests, &scopes, emit_logits)
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
    async fn forward_source_cohort_candidates(
        &self,
        sessions: &[&ResidentSession],
        tokens: &[usize],
    ) -> Result<Option<(Vec<ForwardCandidateState>, Vec<Vec<StateBinding>>)>> {
        if sessions.len() != tokens.len() {
            return Err(Error::State(
                "ordinary source cohort candidate coverage differs".into(),
            ));
        }
        let mut candidates = Vec::with_capacity(sessions.len());
        for (session, &tokens) in sessions.iter().zip(tokens) {
            let position = usize::try_from(session.binding().frontier())
                .map_err(|_| Error::Limit("ordinary source cohort frontier".into()))?;
            match self
                .forward_candidate_state(&session.domains, tokens, position)
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
            .collect();
        Ok(Some((candidates, bindings)))
    }

    async fn prepare_forward_source_prefill_cohort(
        self: &Arc<Self>,
        sessions: &[&ResidentSession],
        requests: &[ForwardPlan],
        scopes: &[Arc<ReaderScope>],
        emit_logits: bool,
    ) -> Result<Option<PreparedResidentCohort>> {
        let width = sessions.len();
        if !matches!(width, 2 | 4 | 8) || requests.len() != width || scopes.len() != width {
            return Err(Error::Unsupported(
                "ordinary source prefill cohort requires two, four or eight rows".into(),
            ));
        }
        self.service.ensure_usable()?;
        let bases = sessions
            .iter()
            .map(|session| session.binding())
            .collect::<Vec<_>>();
        let roots = ResidentBatchRows::new(&bases, 1)?;
        if roots.source() != self.service.source_identity()
            || sessions
                .iter()
                .any(|session| !Arc::ptr_eq(&session.model, self))
        {
            return Err(Error::State(
                "ordinary source prefill cohort model/source differs".into(),
            ));
        }
        let calls = requests
            .iter()
            .map(|request| {
                request.text_invocation().ok_or_else(|| {
                    Error::State("source prefill cohort materialized call absent".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let prompt_rows = calls[0].tokens().len();
        let frontier = roots.frontier().ok_or_else(|| {
            Error::Unsupported(
                "ordinary source prefill cohort requires aligned compatible chunk roots".into(),
            )
        })?;
        if !(1..=64).contains(&prompt_rows)
            || calls.iter().zip(&bases).any(|(call, base)| {
                call.phase() != InvocationPhase::Prefill
                    || call.tokens().base() as u64 != frontier
                    || !(1..=64).contains(&call.tokens().len())
                    || base.frontier() != frontier
                    || call.source() != base.source()
                    || call.tokens().capacity() != self.capacity
                    || call.tokens().vocabulary() as u64 != self.vocab
            })
        {
            return Err(Error::Unsupported(
                "ordinary source prefill cohort requires aligned compatible chunk roots".into(),
            ));
        }
        let mut attempts = Vec::with_capacity(width);
        for (session, call) in sessions.iter().zip(&calls) {
            self.validate_domains(&session.domains, true)?;
            let prompt_advance = u64::try_from(call.tokens().len())
                .map_err(|_| Error::Limit("ordinary source prefill cohort prompt rows".into()))?;
            let attempt = session.authority.begin_advance(prompt_advance)?;
            call.require_attempt(session.binding(), &attempt)?;
            attempts.push(attempt);
        }
        let staging_bytes = self
            .state_bytes
            .checked_mul(width)
            .ok_or_else(|| Error::Limit("ordinary source prefill cohort staging bytes".into()))?;
        let logits_bytes = if emit_logits {
            (self.vocab as usize)
                .checked_mul(width)
                .and_then(|words| words.checked_mul(2))
                .ok_or_else(|| Error::Limit("ordinary source prefill cohort logit bytes".into()))?
        } else {
            0
        };
        let packet_bytes = if emit_logits {
            width
                .checked_mul(12)
                .ok_or_else(|| Error::Limit("ordinary source prefill cohort packet bytes".into()))?
        } else {
            0
        };
        let allocation = self.allocation.lock().await;
        let remaining = self.service.remaining_capacity()?;
        if logits_bytes
            .checked_add(packet_bytes)
            .is_none_or(|bytes| bytes > remaining)
        {
            return Ok(None);
        }

        let row_counts = calls
            .iter()
            .map(|call| call.tokens().len())
            .collect::<Vec<_>>();
        let Some((candidates, bindings)) = self
            .forward_source_cohort_candidates(sessions, &row_counts)
            .await?
        else {
            return Ok(None);
        };
        let logits = emit_logits
            .then(|| {
                self.service
                    .allocate_uninitialized(logits_bytes, BufferRole::Output)
            })
            .transpose()?;
        let packets = if emit_logits {
            (0..width)
                .map(|_| self.service.allocate_uninitialized(12, BufferRole::Output))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let ticket = self.service.queue().submit(
            self.service.budget(),
            ResourceOwner {
                model: self.service.owner().model.clone(),
                session: format!("ordinary-prefill-cohort-{}", bases[0].session().get()),
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
                "ordinary source prefill cohort reached another resident model".into(),
            ));
        }
        for (request, binding) in requests.iter().zip(&bindings) {
            workspace.program.validate_forward_source_request(request)?;
            workspace
                .program
                .validate_text_execution_states(request, binding)?;
        }
        let values = workspace
            .forward_cohort_values
            .as_ref()
            .filter(|values| values.width() >= width)
            .ok_or_else(|| {
                Error::State("ordinary source prefill cohort activation arena absent".into())
            })?;
        let greedy = if emit_logits {
            Some(workspace.greedy.as_ref().ok_or_else(|| {
                Error::Unsupported("ordinary CUDA model has no greedy workspace".into())
            })?)
        } else {
            None
        };
        let mut accesses = if emit_logits {
            workspace.logit_accesses.clone()
        } else {
            workspace.state_accesses.clone()
        };
        accesses.extend(
            values
                .leases()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Write)),
        );
        if let Some(greedy) = greedy {
            for handle in greedy {
                accesses.push((
                    lease_for(&workspace.leases, handle)?.clone(),
                    DeviceAccess::Write,
                ));
            }
        }
        if let Some(logits) = &logits {
            accesses.push((logits.clone(), DeviceAccess::Write));
        }
        accesses.extend(
            packets
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
        let mut state_dispatches = 0usize;
        for candidate in &candidates {
            state_dispatches = state_dispatches
                .checked_add(candidate.enqueue_gather_cuda(&flight)?)
                .ok_or_else(|| Error::Limit("ordinary source cohort gather dispatches".into()))?;
        }
        let mut submitted = workspace
            .program
            .launch_forward_source_prefill_cohort(
                &mut flight,
                requests,
                &bindings,
                logits.clone(),
                values,
            )
            .await?;
        for candidate in &candidates {
            state_dispatches = state_dispatches
                .checked_add(candidate.enqueue_scatter_cuda(&flight)?)
                .ok_or_else(|| Error::Limit("ordinary source cohort scatter dispatches".into()))?;
        }
        submitted.dispatches = submitted
            .dispatches
            .checked_add(state_dispatches)
            .ok_or_else(|| Error::Limit("ordinary source cohort state dispatches".into()))?;
        if submitted.proposals.len() != width {
            return Err(Error::State(
                "ordinary source prefill cohort state proposal coverage".into(),
            ));
        }
        for (row, proposals) in submitted.proposals.iter().enumerate() {
            let expected = workspace.program.plan().proposed_domains();
            if proposals.len() != expected.len() {
                return Err(Error::State(
                    "ordinary source prefill cohort proposed-domain count".into(),
                ));
            }
            for ((version, buffers), expected) in proposals.iter().zip(expected) {
                let binding = bindings[row]
                    .iter()
                    .find(|binding| binding.slot == version.slot)
                    .ok_or_else(|| {
                        Error::State("ordinary source prefill cohort foreign state".into())
                    })?;
                let actual = buffers.handles();
                let target = binding.candidate.handles();
                if version != expected
                    || actual.len() != target.len()
                    || actual.iter().zip(target).any(|(a, b)| !same_handle(a, &b))
                {
                    return Err(Error::State(
                        "ordinary source prefill cohort state effect allocation differs".into(),
                    ));
                }
            }
        }

        if submitted.logits.is_some() != emit_logits {
            return Err(Error::State(
                "ordinary source prefill cohort output policy differs".into(),
            ));
        }
        let row_bytes = self.vocab as usize * 2;
        if emit_logits {
            let logits = submitted.logits.as_ref().ok_or_else(|| {
                Error::State("ordinary source prefill cohort logits absent".into())
            })?;
            let greedy = greedy.ok_or_else(|| {
                Error::State("ordinary source prefill cohort greedy workspace absent".into())
            })?;
            for (row, packet) in packets.iter().enumerate() {
                let start = row * row_bytes;
                let view = logits
                    .handle()
                    .offset_start(start as u64)
                    .offset_end((logits_bytes - start - row_bytes) as u64);
                submitted.dispatches += crate::resident_greedy::launch_bf16_numeric_zero(
                    self.service.client(),
                    &view,
                    self.vocab as usize,
                    &greedy[0],
                    &greedy[1],
                    &packet.handle(),
                    &submitted.status,
                )?;
            }
        }
        let reads = if emit_logits {
            packets
                .iter()
                .map(|packet| packet.readback(0, 12))
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![lease_for(&workspace.leases, &submitted.status)?.readback(0, 4)?]
        };
        let outputs = bases
            .iter()
            .copied()
            .zip(&candidates)
            .enumerate()
            .map(|(row, (base, candidate))| {
                let mut leases = candidate.write_outputs()?;
                if let Some(packet) = packets.get(row) {
                    leases.push(packet.clone());
                }
                Ok((base, leases))
            })
            .collect::<Result<Vec<_>>>()?;
        let (ready, read) = flight
            .complete_cohort_with_readback(&outputs, &reads, |values| {
                if emit_logits {
                    if values.len() != width {
                        return Err(Error::State(
                            "ordinary source prefill cohort packet coverage".into(),
                        ));
                    }
                    for packet in values {
                        if packet.len() != 12 {
                            return Err(Error::State(
                                "ordinary source prefill cohort packet extent".into(),
                            ));
                        }
                        let id = u32::from_ne_bytes(packet[0..4].try_into().unwrap());
                        let bits = u32::from_ne_bytes(packet[4..8].try_into().unwrap());
                        let flags = u32::from_ne_bytes(packet[8..12].try_into().unwrap());
                        if u64::from(id) >= self.vocab
                            || !f32::from_bits(bits).is_finite()
                            || flags != 0
                        {
                            return Err(Error::State(
                                "ordinary source prefill cohort invalid greedy packet".into(),
                            ));
                        }
                    }
                } else if values.len() != 1 || !all_zero(&values[0]) {
                    return Err(Error::State(
                        "ordinary source prefill cohort state-only status refused".into(),
                    ));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;

        let dispatches = submitted.dispatches;
        let fused_dispatches = submitted.fused_dispatches;
        let mut packets = packets.into_iter();
        let mut prepared = Vec::with_capacity(width);
        for (row, ((attempt, candidate), ready)) in
            attempts.into_iter().zip(candidates).zip(ready).enumerate()
        {
            let domains = candidate.into_domains()?;
            let device_token = if emit_logits {
                let packet = packets.next().ok_or_else(|| {
                    Error::State("ordinary source prefill cohort packet missing".into())
                })?;
                Some(ResidentToken {
                    lease: packet,
                    producer: None,
                    id: u32::from_ne_bytes(read[row][0..4].try_into().unwrap()),
                    max_bits: u32::from_ne_bytes(read[row][4..8].try_into().unwrap()),
                })
            } else {
                None
            };
            prepared.push(PreparedInference {
                attempt,
                domains,
                derived: Vec::new(),
                ready,
                readback: InferenceReadback {
                    dispatches: if row == 0 { dispatches } else { 0 },
                    bytes: if emit_logits {
                        12
                    } else {
                        usize::from(row == 0) * 4
                    },
                    chunk_bytes: if emit_logits {
                        12
                    } else {
                        usize::from(row == 0) * 4
                    },
                    state_only_steps: usize::from(!emit_logits),
                    logit_steps: usize::from(emit_logits),
                    device_token,
                    ..Default::default()
                },
            });
        }
        if packets.next().is_some() {
            return Err(Error::State(
                "ordinary source prefill cohort packet surplus".into(),
            ));
        }
        Ok(Some(PreparedResidentCohort {
            model: Arc::clone(self),
            roots,
            prepared,
            dispatches,
            fused_dispatches,
            native_greedy_dispatches: 0,
        }))
    }

    async fn prepare_forward_source_decode_cohort(
        self: &Arc<Self>,
        sessions: &[&ResidentSession],
        requests: &[ForwardPlan],
        scopes: &[Arc<ReaderScope>],
    ) -> Result<Option<PreparedResidentCohort>> {
        let width = sessions.len();
        if !matches!(width, 2 | 4 | 8) || requests.len() != width || scopes.len() != width {
            return Err(Error::Unsupported(
                "ordinary source decode cohort requires two, four or eight rows".into(),
            ));
        }
        self.service.ensure_usable()?;
        let bases = sessions
            .iter()
            .map(|session| session.binding())
            .collect::<Vec<_>>();
        let roots = ResidentBatchRows::new(&bases, 1)?;
        if roots.source() != self.service.source_identity()
            || sessions
                .iter()
                .any(|session| !Arc::ptr_eq(&session.model, self))
        {
            return Err(Error::State(
                "ordinary source cohort model/source differs".into(),
            ));
        }
        let calls = requests
            .iter()
            .map(|request| {
                request
                    .text_invocation()
                    .ok_or_else(|| Error::State("source cohort materialized call absent".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        if calls.iter().zip(&bases).any(|(call, base)| {
            call.phase() != InvocationPhase::Decode
                || call.tokens().len() != 1
                || call.tokens().base() as u64 != base.frontier()
                || call.source() != base.source()
                || call.tokens().capacity() != self.capacity
                || call.tokens().vocabulary() as u64 != self.vocab
        }) {
            return Err(Error::Unsupported(
                "ordinary source cohort requires cached-decode rows bound to private roots".into(),
            ));
        }
        let mut attempts = Vec::with_capacity(width);
        for (session, call) in sessions.iter().zip(&calls) {
            self.validate_domains(&session.domains, true)?;
            let attempt = session.authority.begin_advance(1)?;
            call.require_attempt(session.binding(), &attempt)?;
            attempts.push(attempt);
        }
        let staging_bytes = self
            .state_bytes
            .checked_mul(width)
            .ok_or_else(|| Error::Limit("ordinary source cohort staging bytes".into()))?;
        let logits_bytes = (self.vocab as usize)
            .checked_mul(width)
            .and_then(|words| words.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary source cohort logit bytes".into()))?;
        let packet_bytes = width
            .checked_mul(12)
            .ok_or_else(|| Error::Limit("ordinary source cohort packet bytes".into()))?;
        let allocation = self.allocation.lock().await;
        let remaining = self.service.remaining_capacity()?;
        if logits_bytes
            .checked_add(packet_bytes)
            .is_none_or(|bytes| bytes > remaining)
        {
            return Ok(None);
        }
        let tokens = vec![1usize; width];
        let Some((candidates, bindings)) = self
            .forward_source_cohort_candidates(sessions, &tokens)
            .await?
        else {
            return Ok(None);
        };
        let logits = self
            .service
            .allocate_uninitialized(logits_bytes, BufferRole::Output)?;
        let packets = (0..width)
            .map(|_| self.service.allocate_uninitialized(12, BufferRole::Output))
            .collect::<Result<Vec<_>>>()?;
        let ticket = self.service.queue().submit(
            self.service.budget(),
            ResourceOwner {
                model: self.service.owner().model.clone(),
                session: format!("ordinary-cohort-{}", bases[0].session().get()),
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
                "ordinary source cohort reached another resident model".into(),
            ));
        }
        for (request, binding) in requests.iter().zip(&bindings) {
            workspace.program.validate_forward_source_request(request)?;
            workspace
                .program
                .validate_text_execution_states(request, binding)?;
        }
        let values = workspace
            .forward_cohort_values
            .as_ref()
            .filter(|values| values.width() >= width)
            .ok_or_else(|| {
                Error::State("ordinary source decode cohort activation arena absent".into())
            })?;
        let greedy = workspace.greedy.as_ref().ok_or_else(|| {
            Error::Unsupported("ordinary CUDA model has no greedy workspace".into())
        })?;
        let mut accesses = workspace.logit_accesses.clone();
        if matches!(width, 2 | 4 | 8) {
            accesses.push((values.row_scratch().clone(), DeviceAccess::Write));
        }
        for handle in greedy {
            accesses.push((
                lease_for(&workspace.leases, handle)?.clone(),
                DeviceAccess::Write,
            ));
        }
        accesses.push((logits.clone(), DeviceAccess::Write));
        accesses.extend(
            packets
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
        let mut state_dispatches = 0usize;
        for candidate in &candidates {
            state_dispatches = state_dispatches
                .checked_add(candidate.enqueue_gather_cuda(&flight)?)
                .ok_or_else(|| Error::Limit("ordinary source cohort gather dispatches".into()))?;
        }
        let mut submitted = workspace
            .program
            .launch_forward_source_decode_cohort(
                &mut flight,
                requests,
                &bindings,
                logits.clone(),
                values,
            )
            .await?;
        for candidate in &candidates {
            state_dispatches = state_dispatches
                .checked_add(candidate.enqueue_scatter_cuda(&flight)?)
                .ok_or_else(|| Error::Limit("ordinary source cohort scatter dispatches".into()))?;
        }
        submitted.dispatches = submitted
            .dispatches
            .checked_add(state_dispatches)
            .ok_or_else(|| Error::Limit("ordinary source cohort state dispatches".into()))?;
        if submitted.proposals.len() != width {
            return Err(Error::State(
                "ordinary source cohort state proposal coverage".into(),
            ));
        }
        for (row, proposals) in submitted.proposals.iter().enumerate() {
            let expected = workspace.program.plan().proposed_domains();
            if proposals.len() != expected.len() {
                return Err(Error::State(
                    "ordinary source cohort proposed-domain count".into(),
                ));
            }
            for ((version, buffers), expected) in proposals.iter().zip(expected) {
                let binding = bindings[row]
                    .iter()
                    .find(|binding| binding.slot == version.slot)
                    .ok_or_else(|| Error::State("ordinary source cohort foreign state".into()))?;
                let actual = buffers.handles();
                let target = binding.candidate.handles();
                if version != expected
                    || actual.len() != target.len()
                    || actual.iter().zip(target).any(|(a, b)| !same_handle(a, &b))
                {
                    return Err(Error::State(
                        "ordinary source cohort state effect allocation differs".into(),
                    ));
                }
            }
        }
        let row_bytes = self.vocab as usize * 2;
        let submitted_logits = submitted
            .logits
            .as_ref()
            .ok_or_else(|| Error::State("ordinary source decode cohort logits absent".into()))?;
        for (row, packet) in packets.iter().enumerate() {
            let start = row * row_bytes;
            let view = submitted_logits
                .handle()
                .offset_start(start as u64)
                .offset_end((logits_bytes - start - row_bytes) as u64);
            submitted.dispatches += crate::resident_greedy::launch_bf16_numeric_zero(
                self.service.client(),
                &view,
                self.vocab as usize,
                &greedy[0],
                &greedy[1],
                &packet.handle(),
                &submitted.status,
            )?;
        }
        let reads = packets
            .iter()
            .map(|packet| packet.readback(0, 12))
            .collect::<Result<Vec<_>>>()?;
        let outputs = bases
            .iter()
            .copied()
            .zip(&candidates)
            .zip(&packets)
            .map(|((base, candidate), packet)| {
                let mut leases = candidate.write_outputs()?;
                leases.push(packet.clone());
                Ok((base, leases))
            })
            .collect::<Result<Vec<_>>>()?;
        let (ready, read) = flight
            .complete_cohort_with_readback(&outputs, &reads, |values| {
                if values.len() != width {
                    return Err(Error::State(
                        "ordinary source cohort packet coverage".into(),
                    ));
                }
                for packet in values {
                    if packet.len() != 12 {
                        return Err(Error::State("ordinary source cohort packet extent".into()));
                    }
                    let id = u32::from_ne_bytes(packet[0..4].try_into().unwrap());
                    let bits = u32::from_ne_bytes(packet[4..8].try_into().unwrap());
                    let flags = u32::from_ne_bytes(packet[8..12].try_into().unwrap());
                    if u64::from(id) >= self.vocab
                        || !f32::from_bits(bits).is_finite()
                        || flags != 0
                    {
                        return Err(Error::State(
                            "ordinary source cohort invalid greedy packet".into(),
                        ));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        let dispatches = submitted.dispatches;
        let fused_dispatches = submitted.fused_dispatches;
        let mut prepared = Vec::with_capacity(width);
        for (row, (((attempt, candidate), ready), packet)) in attempts
            .into_iter()
            .zip(candidates)
            .zip(ready)
            .zip(packets)
            .enumerate()
        {
            let domains = candidate.into_domains()?;
            prepared.push(PreparedInference {
                attempt,
                domains,
                derived: Vec::new(),
                ready,
                readback: InferenceReadback {
                    dispatches: if row == 0 { dispatches } else { 0 },
                    bytes: 12,
                    chunk_bytes: 12,
                    logit_steps: 1,
                    device_token: Some(ResidentToken {
                        lease: packet,
                        producer: None,
                        id: u32::from_ne_bytes(read[row][0..4].try_into().unwrap()),
                        max_bits: u32::from_ne_bytes(read[row][4..8].try_into().unwrap()),
                    }),
                    ..Default::default()
                },
            });
        }
        Ok(Some(PreparedResidentCohort {
            model: Arc::clone(self),
            roots,
            prepared,
            dispatches,
            fused_dispatches,
            native_greedy_dispatches: 0,
        }))
    }
}
