//! Native multi-row lowering, with original per-session logical/physical owners.
use super::*;

impl ResidentModel {
    /// Diagnostic native CUDA batch. Every original ForwardPlan node executes
    /// over all rows; only the existing cohort commit may publish their roots.
    pub async fn prepare_logits_fused_cohort(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
    ) -> Result<PreparedResidentCohort> {
        self.prepare_fused_mode(rows, None, Readout::Logits).await
    }

    /// One native full-model batch and source-exact device greedy selection.
    /// Each independent output packet gains its producer only at cohort commit.
    pub async fn prepare_greedy_fused_cohort(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
    ) -> Result<PreparedResidentCohort> {
        self.prepare_fused_mode(rows, None, Readout::Greedy).await
    }

    /// None means only a checked capacity refusal before any GPU submission.
    /// Errors, including unknown completion, must never select a smaller retry.
    pub async fn prepare_greedy_fused_cohort_admitted(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
    ) -> Result<Option<PreparedResidentCohort>> {
        self.prepare_fused_mode_admitted(rows, None, Readout::Greedy)
            .await
    }

    /// Read original packet words on the GPU under their exact producing roots.
    /// Public host ID/max metadata never selects the next source embedding row.
    pub async fn prepare_device_tokens_fused_cohort(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, &ResidentToken)],
    ) -> Result<PreparedResidentCohort> {
        let sessions = rows
            .iter()
            .map(|(session, _)| (*session, 0))
            .collect::<Vec<_>>();
        let tokens = rows.iter().map(|(_, token)| *token).collect::<Vec<_>>();
        self.prepare_fused_mode(&sessions, Some(&tokens), Readout::Greedy)
            .await
    }

    /// Device packets retain their original producer and physical lease checks.
    pub async fn prepare_device_tokens_fused_cohort_admitted(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, &ResidentToken)],
    ) -> Result<Option<PreparedResidentCohort>> {
        let sessions = rows
            .iter()
            .map(|(session, _)| (*session, 0))
            .collect::<Vec<_>>();
        let tokens = rows.iter().map(|(_, token)| *token).collect::<Vec<_>>();
        self.prepare_fused_mode_admitted(&sessions, Some(&tokens), Readout::Greedy)
            .await
    }

    async fn prepare_fused_mode(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
        device_tokens: Option<&[&ResidentToken]>,
        readout: Readout,
    ) -> Result<PreparedResidentCohort> {
        self.prepare_fused_mode_admitted(rows, device_tokens, readout)
            .await?
            .ok_or_else(|| Error::Limit("native resident batch admission exhausted".into()))
    }

    async fn prepare_fused_mode_admitted(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
        device_tokens: Option<&[&ResidentToken]>,
        readout: Readout,
    ) -> Result<Option<PreparedResidentCohort>> {
        if !self.native_batch {
            return Err(Error::Unsupported(
                "native resident batch requires CUDA".into(),
            ));
        }
        let bases = rows.iter().map(|(s, _)| s.binding()).collect::<Vec<_>>();
        let roots = ResidentBatchRows::new(&bases, 1)?;
        if !matches!(rows.len(), 2 | 4 | 8) {
            return Err(Error::Unsupported(
                "resident native batch rows must be 2/4/8".into(),
            ));
        }
        if roots.source() != self.service.source_identity() {
            return Err(Error::State("native resident batch source mismatch".into()));
        }
        self.service.ensure_usable()?;
        if let Some(tokens) = device_tokens {
            if tokens.len() != rows.len() {
                return Err(Error::State("native batch packet coverage".into()));
            }
            for ((session, _), token) in rows.iter().zip(tokens) {
                if token.producer != Some(session.binding())
                    || !Arc::ptr_eq(token.lease.service(), &self.service)
                    || token.lease.source_identity() != self.service.source_identity()
                    || token.lease.service_generation() != self.service.generation()
                    || !token.lease.is_initialized()
                    || token.lease.bytes() != 12
                {
                    return Err(Error::State(
                        "stale/foreign native batch device token".into(),
                    ));
                }
            }
        }
        let positions = roots
            .frontiers()
            .map(|frontier| {
                usize::try_from(frontier)
                    .map_err(|_| Error::Limit("native resident batch frontier overflow".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        if positions.iter().any(|&position| position >= self.capacity) {
            return Err(Error::Limit(
                "native resident batch capacity exhausted".into(),
            ));
        }
        let mut attempts = Vec::with_capacity(rows.len());
        for (session, token) in rows {
            if !Arc::ptr_eq(&session.model, self) || u64::from(*token) >= self.vocab {
                return Err(Error::State(
                    "foreign/invalid native resident batch row".into(),
                ));
            }
            self.validate_domains(&session.domains, true)?;
            attempts.push(session.authority.begin_advance(1)?);
        }
        let staging_bytes = self
            .state_bytes
            .checked_mul(rows.len())
            .ok_or_else(|| Error::Limit("native resident batch admission overflow".into()))?;
        let packet_bytes = if matches!(readout, Readout::Greedy) {
            rows.len() * 12
        } else {
            0
        };
        let allocation = self.allocation.lock().await;
        // This lock follows any prior actual terminal completion. Account the
        // complete candidate/packet/workspace footprint before retiring an idle
        // band or allocating a single candidate. No scheduler byte credits.
        {
            let mut workspace = self.workspace.lock().await;
            self.service.ensure_usable()?;
            let same_band = workspace
                .batch
                .as_ref()
                .is_some_and(|b| b.rows == rows.len());
            let reclaimable = if same_band {
                0
            } else {
                workspace
                    .batch
                    .as_ref()
                    .map(|b| b.program.budget().owned_bytes())
                    .transpose()?
                    .unwrap_or(0)
            };
            let workspace_bytes = if same_band {
                0
            } else {
                BatchProgram::preflight(&workspace.program, rows.len())?.owned_bytes()?
            };
            let required = staging_bytes
                .checked_add(packet_bytes)
                .and_then(|n| n.checked_add(workspace_bytes))
                .ok_or_else(|| Error::Limit("native resident batch admission overflow".into()))?;
            let available = self
                .service
                .remaining_capacity()?
                .checked_add(reclaimable)
                .ok_or_else(|| Error::Limit("native resident batch capacity overflow".into()))?;
            if required > available || staging_bytes as u64 > self.service.budget().remaining()? {
                return Ok(None);
            }
            if !same_band {
                workspace.batch = None;
            }
        }
        let mut candidates = Vec::with_capacity(rows.len());
        for _ in rows {
            candidates.push(self.allocate_uninitialized_state()?);
        }
        let bindings = rows
            .iter()
            .zip(&candidates)
            .zip(&positions)
            .map(|(((session, _), candidate), &position)| {
                bind_domains(&session.domains, candidate, position)
            })
            .collect::<Result<Vec<_>>>()?;
        let parameters = rows
            .iter()
            .zip(&positions)
            .map(|((_, token), &position)| RuntimeParameters {
                token: *token,
                position,
                valid_tokens: position + 1,
            })
            .collect::<Vec<_>>();
        // State initialization uses allocation -> queue. Preserve that order
        // while admitting a cold batch workspace, before selecting a slot.
        let packets = if matches!(readout, Readout::Greedy) {
            (0..rows.len())
                .map(|_| self.service.allocate_uninitialized(12, BufferRole::Output))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let ticket = self.service.queue().submit(
            self.service.budget(),
            self.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: staging_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        let mut workspace = self.workspace.lock().await;
        if workspace
            .batch
            .as_ref()
            .is_some_and(|b| b.rows != rows.len())
        {
            // The shared workspace lock follows the actual terminal event.
            // A different row band may retire the previous idle workspace.
            workspace.batch = None;
        }
        let batch_budget = BatchProgram::preflight(&workspace.program, rows.len())?;
        if workspace.batch.is_none()
            && batch_budget.owned_bytes()? > self.service.remaining_capacity()?
        {
            return Err(Error::Limit(
                "native resident batch workspace admission exhausted".into(),
            ));
        }
        if workspace.batch.is_none() {
            slot.begin_submission()?;
            let mut initialization = self.service.begin_initialization(bases[0].root())?;
            let mut leases = Vec::new();
            let program = BatchProgram::prepare(
                self.service.client(),
                &workspace.program,
                rows.len(),
                |role, count, data| {
                    let role = match role {
                        ResidentBufferRole::Source => BufferRole::ImmutableWeights,
                        ResidentBufferRole::Value | ResidentBufferRole::Parameter => {
                            BufferRole::Activation
                        }
                        ResidentBufferRole::Workspace | ResidentBufferRole::Status => {
                            BufferRole::Scratch
                        }
                    };
                    let lease = match data {
                        AllocationInit::Empty => initialization.allocate_empty(count, role)?,
                        AllocationInit::Upload(data) => {
                            if data.len() != count {
                                return Err(Error::State("native batch initializer length".into()));
                            }
                            initialization.upload_bytes(data, role)?
                        }
                    };
                    let handle = lease.handle();
                    leases.push((lease, DeviceAccess::Write));
                    Ok(handle)
                },
            )?;
            if program.rows() != rows.len()
                || program.budget().owned_bytes()? != batch_budget.owned_bytes()?
            {
                return Err(Error::State(
                    "native batch prepared geometry/budget changed".into(),
                ));
            }
            for handle in program.owned_handles() {
                lease_for(&leases, &handle)?;
            }
            if leases.iter().map(|(l, _)| l.bytes()).sum::<usize>() != batch_budget.owned_bytes()? {
                return Err(Error::State("native batch allocation coverage".into()));
            }
            let ready = initialization.complete_sync().await?;
            if ready.outputs().len() != leases.len() {
                return Err(Error::State(
                    "native batch initialization incomplete".into(),
                ));
            }
            workspace.batch = Some(BatchWorkspace {
                program,
                leases,
                rows: rows.len(),
            });
            // Initialization reached a real terminal sync. Retire its slot,
            // then release the workspace before waiting for execution: a B1
            // worker selected meanwhile may need that same workspace lock.
            slot.finish()?;
            drop(workspace);
            let ticket = self.service.queue().submit(
                self.service.budget(),
                self.service.owner().clone(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: staging_bytes as u64,
                }],
                None,
            )?;
            slot = ticket.wait().await?;
            workspace = self.workspace.lock().await;
        }
        let batch = workspace.batch.as_ref().unwrap();
        batch
            .program
            .validate_runtime(self.service.client(), &parameters, &bindings)?;
        let device_buffers = device_tokens.map(|tokens| {
            tokens
                .iter()
                .map(|token| DeviceBuffer::new(token.lease.handle(), 1))
                .collect::<Vec<_>>()
        });
        if let Some(tokens) = &device_buffers {
            batch.program.validate_device_tokens(
                self.service.client(),
                &parameters,
                &bindings,
                tokens,
            )?;
        }
        let packet_handles = packets
            .iter()
            .map(ResidentLease::handle)
            .collect::<Vec<_>>();
        if matches!(readout, Readout::Greedy) {
            let logits = batch
                .program
                .logits()
                .ok_or_else(|| Error::State("native batch missing logit geometry".into()))?;
            crate::resident_batch_greedy::validate(
                self.service.client(),
                &logits.handle,
                self.vocab as usize,
                rows.len(),
                batch.program.greedy_workspace(),
                &packet_handles,
                &batch.program.status(),
            )?;
        }
        let handles = batch.program.execution_handles(ResidentOutput::Logits);
        let mut accesses = workspace
            .leases
            .iter()
            .chain(&batch.leases)
            .filter(|(l, _)| handles.iter().any(|h| same_handle(h, &l.handle())))
            .cloned()
            .collect::<Vec<_>>();
        for handle in handles {
            lease_for(&accesses, &handle)?;
        }
        if let Some(tokens) = device_tokens {
            accesses.extend(
                tokens
                    .iter()
                    .map(|token| (token.lease.clone(), DeviceAccess::Read)),
            );
        }
        accesses.extend(
            packets
                .iter()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Write)),
        );
        for ((session, _), candidate) in rows.iter().zip(&candidates) {
            self.validate_domains(candidate, false)?;
            accesses.extend(
                session
                    .domains
                    .iter()
                    .flat_map(|d| d.leases.iter())
                    .cloned()
                    .map(|l| (l, DeviceAccess::Read)),
            );
            accesses.extend(
                candidate
                    .iter()
                    .flat_map(|d| d.leases.iter())
                    .cloned()
                    .map(|l| (l, DeviceAccess::Write)),
            );
        }
        drop(allocation);
        slot.begin_submission()?;
        let flight = self.service.submit_cohort(&bases, &accesses)?;
        let program = &mut workspace.batch.as_mut().unwrap().program;
        let mut submitted = if let Some(tokens) = &device_buffers {
            program.launch_device_tokens(
                self.service.client(),
                &parameters,
                &bindings,
                tokens,
                ResidentOutput::Logits,
            )?
        } else {
            program.launch(
                self.service.client(),
                &parameters,
                &bindings,
                ResidentOutput::Logits,
            )?
        };
        let expected = workspace.program.plan().proposed_domains();
        if submitted.proposals.len() != expected.len() * rows.len()
            || submitted.fused_dispatches == 0
        {
            return Err(Error::State(
                "incomplete native batch effect/dispatch receipt".into(),
            ));
        }
        let mut native_greedy_dispatches = 0;
        if matches!(readout, Readout::Greedy) {
            let logits = submitted
                .logits
                .as_ref()
                .ok_or_else(|| Error::State("native greedy missing logits".into()))?;
            let receipt = crate::resident_batch_greedy::launch(
                self.service.client(),
                &logits.handle,
                self.vocab as usize,
                rows.len(),
                workspace.batch.as_ref().unwrap().program.greedy_workspace(),
                &packet_handles,
                &submitted.status,
            )?;
            submitted.dispatches += receipt.dispatches;
            native_greedy_dispatches = receipt.native_dispatches;
        }
        #[cfg(feature = "hardware-abort-probe")]
        if self
            .abort_after_launch
            .fetch_update(
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
                |remaining| remaining.checked_sub(1),
            )
            .is_ok_and(|remaining| remaining == 1)
        {
            self.launch_seen
                .store(flight.handles().len(), std::sync::atomic::Ordering::Release);
            std::future::pending::<()>().await;
        }
        for ((actual, wanted), domains) in submitted
            .proposals
            .chunks_exact(expected.len())
            .map(|chunk| (chunk, expected))
            .zip(&candidates)
        {
            for ((version, buffers), expected) in actual.iter().zip(wanted) {
                let domain = domains
                    .iter()
                    .find(|d| d.slot == version.slot)
                    .ok_or_else(|| Error::State("native batch foreign domain receipt".into()))?;
                let a = buffers.handles();
                let b = domain.buffers.handles();
                if version != expected
                    || a.len() != b.len()
                    || a.iter().zip(&b).any(|(a, b)| !same_handle(a, b))
                {
                    return Err(Error::State("native batch state receipt mismatch".into()));
                }
            }
        }
        let logits = submitted
            .logits
            .as_ref()
            .ok_or_else(|| Error::State("native batch missing logits".into()))?;
        let expected_logits = (self.vocab as usize)
            .checked_mul(rows.len())
            .ok_or_else(|| Error::Limit("native batch logit size".into()))?;
        if logits.elements != expected_logits {
            return Err(Error::State("native batch logit geometry".into()));
        }
        let mut reads = vec![lease_for(&accesses, &submitted.status)?.readback(0, 4)?];
        if matches!(readout, Readout::Greedy) {
            reads.extend(
                packets
                    .iter()
                    .map(|packet| packet.readback(0, 12))
                    .collect::<Result<Vec<_>>>()?,
            );
        } else {
            reads.push(lease_for(&accesses, &logits.handle)?.readback(0, expected_logits * 4)?);
        }
        let outputs = bases
            .iter()
            .copied()
            .zip(&candidates)
            .enumerate()
            .map(|(row, (base, domains))| {
                let mut outputs = domains
                    .iter()
                    .flat_map(|d| d.leases.iter())
                    .cloned()
                    .collect::<Vec<_>>();
                outputs.extend(packets.get(row).cloned());
                (base, outputs)
            })
            .collect::<Vec<_>>();
        let (ready, read) = flight
            .complete_cohort_with_readback(&outputs, &reads, |data| {
                if !all_zero(&data[0]) {
                    return Err(Error::State("native GPU batch status refused".into()));
                }
                if matches!(readout, Readout::Greedy) {
                    for packet in &data[1..] {
                        let id = u32::from_ne_bytes(packet[0..4].try_into().unwrap());
                        let bits = u32::from_ne_bytes(packet[4..8].try_into().unwrap());
                        let flags = u32::from_ne_bytes(packet[8..12].try_into().unwrap());
                        if u64::from(id) >= self.vocab
                            || flags != 0
                            || !f32::from_bits(bits).is_finite()
                        {
                            return Err(Error::State("invalid native greedy packet".into()));
                        }
                    }
                } else if words(&data[1])?.iter().any(|value| !value.is_finite()) {
                    return Err(Error::State("nonfinite native GPU batch logits".into()));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        let greedy = matches!(readout, Readout::Greedy);
        let logits = if greedy { Vec::new() } else { words(&read[1])? };
        let mut prepared = Vec::with_capacity(rows.len());
        let mut packets = packets.into_iter();
        for (row, ((attempt, domains), ready)) in
            attempts.into_iter().zip(candidates).zip(ready).enumerate()
        {
            prepared.push(PreparedInference {
                attempt,
                domains,
                ready,
                readback: InferenceReadback {
                    // Shared physical work is counted once across the cohort.
                    dispatches: if row == 0 { submitted.dispatches } else { 0 },
                    logits: if greedy {
                        Vec::new()
                    } else {
                        logits[row * self.vocab as usize..(row + 1) * self.vocab as usize].to_vec()
                    },
                    logit_steps: 1,
                    bytes: if greedy { 12 } else { self.vocab as usize * 4 }
                        + usize::from(row == 0) * 4,
                    chunk_bytes: if greedy { 12 } else { self.vocab as usize * 4 }
                        + usize::from(row == 0) * 4,
                    device_token: packets.next().map(|lease| ResidentToken {
                        lease,
                        producer: None,
                        id: u32::from_ne_bytes(read[row + 1][0..4].try_into().unwrap()),
                        max_bits: u32::from_ne_bytes(read[row + 1][4..8].try_into().unwrap()),
                    }),
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
            native_greedy_dispatches,
        }))
    }
}
