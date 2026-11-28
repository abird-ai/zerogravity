//! Independent prefill/decode cohorts for HIP text execution.
//! Share projections and compatible row-wise work while retaining each session's
//! private state roots, admitted flight and completion authority.
use super::*;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};

impl ResidentTextModel {
    pub async fn prepare_decode_cohort_admitted(
        self: &Arc<Self>,
        rows: &[(&ResidentTextSession, &BoundInvocation)],
    ) -> Result<ResidentTextCohortPreparation> {
        self.prepare_hip_cohort_admitted(rows, InvocationPhase::Decode, false, None)
            .await
    }

    pub(super) async fn prepare_hip_cohort_admitted(
        self: &Arc<Self>,
        rows: &[(&ResidentTextSession, &BoundInvocation)],
        phase: InvocationPhase,
        compact: bool,
        device_tokens: Option<&[&ResidentToken]>,
    ) -> Result<ResidentTextCohortPreparation> {
        if !matches!(self.admission.cohort_width, 4 | 8)
            || !matches!(rows.len(), 2 | 4 | 8)
            || rows.len() > self.admission.cohort_width
            || !self.contract.uses_torch_mixed_operators()
        {
            return Err(Error::Unsupported(
                "source HIP cohort requires an admitted width".into(),
            ));
        }
        let mut requests = Vec::with_capacity(rows.len());
        let mut scopes = Vec::with_capacity(rows.len());
        let mut sessions = Vec::with_capacity(rows.len());
        let frontier = rows[0].0.binding().frontier();
        let tokens = rows[0].1.input().token_count();
        let emit_logits = rows[0].1.input().emit_logits();
        if !(1..=64).contains(&tokens)
            || phase == InvocationPhase::Prefill && tokens == 1
            || phase == InvocationPhase::Decode && (tokens != 1 || !emit_logits)
        {
            return Err(Error::Unsupported("HIP cohort invocation shape".into()));
        }
        if compact && !self.supports_device_greedy()
            || device_tokens.is_some_and(|tokens| {
                !compact || phase != InvocationPhase::Decode || tokens.len() != rows.len()
            })
        {
            return Err(Error::Unsupported(
                "HIP cohort feedback/readout policy".into(),
            ));
        }
        let readout = match (emit_logits, compact) {
            (false, _) => Readout::None,
            (true, false) => Readout::Logits,
            (true, true) => Readout::Greedy,
        };
        for (index, (session, invocation)) in rows.iter().enumerate() {
            if !Arc::ptr_eq(&session.model, self)
                || invocation.input().emit_logits() != emit_logits
                || invocation.input().token_count() != tokens
                || invocation.input().base() as u64 != frontier
                || session.binding().frontier() != frontier
            {
                return Err(Error::State("source HIP cohort row/root differs".into()));
            }
            if let Some(tokens) = device_tokens {
                session.require_device_feedback(invocation, tokens[index])?;
            }
            let call = self.bind_invocation(invocation, phase, false)?;
            scopes.push(self.reader_scope(CALL_HOST_BYTES)?);
            requests.push(call.materialize()?);
            sessions.push(&session.inner);
        }
        let Some(inner) = self
            .inner
            .prepare_text_execution_cohort(&sessions, &requests, &scopes, readout, device_tokens)
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
    async fn prepare_text_execution_cohort(
        self: &Arc<Self>,
        sessions: &[&ResidentSession],
        requests: &[ForwardPlan],
        scopes: &[Arc<ReaderScope>],
        readout: Readout,
        device_tokens: Option<&[&ResidentToken]>,
    ) -> Result<Option<PreparedResidentCohort>> {
        let width = sessions.len();
        let emit_logits = !matches!(readout, Readout::None);
        let compact = matches!(readout, Readout::Greedy);
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
        let frontier = roots
            .frontier()
            .ok_or_else(|| Error::Unsupported("HIP cohort requires an aligned frontier".into()))?;
        if roots.source() != self.service.source_identity()
            || sessions
                .iter()
                .any(|session| !Arc::ptr_eq(&session.model, self))
        {
            return Err(Error::State("HIP cohort model/source differs".into()));
        }
        let calls = requests
            .iter()
            .map(|request| {
                request
                    .text_invocation()
                    .ok_or_else(|| Error::State("HIP cohort materialized call absent".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let tokens = calls[0].tokens().len();
        let phase = calls[0].phase();
        if !(1..=64).contains(&tokens)
            || phase == InvocationPhase::Prefill && tokens == 1
            || phase == InvocationPhase::Decode && (tokens != 1 || !emit_logits)
            || calls.iter().zip(&bases).any(|(call, base)| {
                call.phase() != phase
                    || call.tokens().len() != tokens
                    || call.tokens().base() as u64 != frontier
                    || base.frontier() != frontier
                    || call.source() != base.source()
                    || call.tokens().capacity() != self.capacity
                    || call.tokens().vocabulary() as u64 != self.vocab
            })
        {
            return Err(Error::Unsupported(
                "HIP cohort requires aligned invocation rows".into(),
            ));
        }

        if let Some(tokens) = device_tokens
            && (tokens.len() != width
                || phase != InvocationPhase::Decode
                || !compact
                || tokens
                    .iter()
                    .zip(&bases)
                    .zip(&calls)
                    .any(|((token, base), call)| {
                        token.producer != Some(*base)
                            || token.lease.bytes() != 12
                            || !Arc::ptr_eq(token.lease.service(), &self.service)
                            || call.tokens().ids() != [token.id]
                    }))
        {
            return Err(Error::State(
                "HIP cohort committed device feedback differs".into(),
            ));
        }

        let mut attempts = Vec::with_capacity(width);
        for (session, call) in sessions.iter().zip(&calls) {
            self.validate_domains(&session.domains, true)?;
            let attempt = session.authority.begin_advance(tokens as u64)?;
            call.require_attempt(session.binding(), &attempt)?;
            attempts.push(attempt);
        }
        let staging_bytes = self
            .state_bytes
            .checked_mul(width)
            .ok_or_else(|| Error::Limit("HIP cohort staging bytes".into()))?;
        let allocation = self.allocation.lock().await;
        let mut candidates = Vec::with_capacity(width);
        for session in sessions {
            let position = usize::try_from(session.binding().frontier())
                .map_err(|_| Error::Limit("HIP cohort frontier".into()))?;
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
            .collect::<Vec<_>>();
        let buffers = {
            let workspace = self.workspace.lock().await;
            if !workspace.program.is_forward_source() {
                return Err(Error::State(
                    "HIP cohort reached another resident model".into(),
                ));
            }
            let Some(buffers) = workspace
                .program
                .text_execution_cohort_buffers(requests, emit_logits)?
            else {
                return Ok(None);
            };
            buffers
        };
        let packets = if compact {
            if width * 12 > self.service.remaining_capacity()? {
                return Ok(None);
            }
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
                session: format!("text-cohort-{}", bases[0].session().get()),
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
                "HIP cohort reached another resident model".into(),
            ));
        }
        for (request, binding) in requests.iter().zip(&bindings) {
            workspace.program.validate_forward_source_request(request)?;
            workspace
                .program
                .validate_text_execution_states(request, binding)?;
        }
        let mut accesses = if emit_logits {
            workspace.logit_accesses.clone()
        } else {
            workspace.state_accesses.clone()
        };
        accesses.extend(buffers.accesses());
        accesses.extend(
            packets
                .iter()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Write)),
        );
        let token_handles = device_tokens.map(|tokens| {
            accesses.extend(
                tokens
                    .iter()
                    .map(|token| (token.lease.clone(), DeviceAccess::Read)),
            );
            tokens
                .iter()
                .map(|token| token.lease.handle().offset_end(8))
                .collect::<Vec<_>>()
        });
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
            .launch_text_execution_cohort(
                &mut flight,
                requests,
                &bindings,
                &buffers,
                token_handles.as_deref(),
            )
            .await?;
        if submitted.fused_dispatches == 0 {
            return Err(Error::State(
                "HIP cohort produced no shared projection work".into(),
            ));
        }

        let (reads, greedy_dispatches, native_greedy_dispatches) = if compact {
            let logits = buffers
                .logits
                .as_ref()
                .ok_or_else(|| Error::State("HIP compact cohort logits absent".into()))?;
            let greedy =
                workspace
                    .program
                    .launch_hip_source_greedy(&flight, &logits.handle(), &packets)?;
            // Shared status is folded into every packet after the complete
            // cohort reduction. The original flight retains the private leases.
            (
                packets
                    .iter()
                    .map(|packet| packet.readback(0, 12))
                    .collect::<Result<Vec<_>>>()?,
                greedy.dispatches,
                greedy.native_dispatches,
            )
        } else {
            let mut reads = vec![lease_for(&accesses, &submitted.status)?.readback(0, 4)?];
            if let Some(logits) = &buffers.logits {
                reads.push(logits.readback(0, logits.bytes())?);
            }
            (reads, 0, 0)
        };
        let dispatches = submitted
            .dispatches
            .checked_add(greedy_dispatches)
            .ok_or_else(|| Error::Limit("HIP cohort dispatch count".into()))?;
        let outputs = bases
            .iter()
            .copied()
            .zip(&candidates)
            .enumerate()
            .map(|(row, (base, candidate))| {
                let mut leases = candidate.write_outputs()?;
                leases.extend(packets.get(row).cloned());
                Ok((base, leases))
            })
            .collect::<Result<Vec<_>>>()?;
        let (ready, read) = flight
            .complete_cohort_with_readback(&outputs, &reads, |data| {
                if compact {
                    if data.len() != width {
                        return Err(Error::State("HIP cohort greedy packet coverage".into()));
                    }
                    for packet in data {
                        validate_greedy_packet(packet, self.vocab)?;
                    }
                } else {
                    if !all_zero(&data[0]) {
                        return Err(Error::State("HIP cohort status refused".into()));
                    }
                    if emit_logits && words(&data[1])?.iter().any(|value| !value.is_finite()) {
                        return Err(Error::State("HIP cohort nonfinite logits".into()));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;

        let logits = if matches!(readout, Readout::Logits) {
            words(&read[1])?
        } else {
            Vec::new()
        };
        let row_words = self.vocab as usize;
        let mut packets = packets.into_iter();
        let mut prepared = Vec::with_capacity(width);
        for (row, ((attempt, candidate), ready)) in
            attempts.into_iter().zip(candidates).zip(ready).enumerate()
        {
            let domains = candidate.into_domains()?;
            self.validate_domains(&domains, true)?;
            let device_token = packets.next().map(|lease| ResidentToken {
                lease,
                producer: None,
                id: u32::from_ne_bytes(read[row][0..4].try_into().unwrap()),
                max_bits: u32::from_ne_bytes(read[row][4..8].try_into().unwrap()),
            });
            let bytes = match readout {
                Readout::Greedy => 12,
                Readout::Logits => row_words * 4 + usize::from(row == 0) * 4,
                Readout::None => usize::from(row == 0) * 4,
            };
            prepared.push(PreparedInference {
                attempt,
                domains,
                derived: Vec::new(),
                ready,
                readback: InferenceReadback {
                    logits: if matches!(readout, Readout::Logits) {
                        logits[row * row_words..(row + 1) * row_words].to_vec()
                    } else {
                        Vec::new()
                    },
                    device_token,
                    dispatches: usize::from(row == 0) * dispatches,
                    bytes,
                    chunk_bytes: bytes,
                    state_only_steps: usize::from(!emit_logits),
                    logit_steps: usize::from(emit_logits),
                    ..Default::default()
                },
            });
        }
        Ok(Some(PreparedResidentCohort {
            model: Arc::clone(self),
            roots,
            prepared,
            dispatches,
            fused_dispatches: submitted.fused_dispatches,
            native_greedy_dispatches,
            execution_mode: zerogravity_core::provider::ExecutionMode::Native,
        }))
    }
}
