//! Native multi-row lowering, with original per-session logical/physical owners.
use super::*;

impl ResidentModel {
    /// Diagnostic native CUDA batch. Every original ForwardPlan node executes
    /// over all rows; only the existing cohort commit may publish their roots.
    pub async fn prepare_logits_fused_cohort(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
    ) -> Result<PreparedResidentCohort> {
        if !self.native_batch {
            return Err(Error::Unsupported(
                "native resident batch requires CUDA".into(),
            ));
        }
        let bases = rows.iter().map(|(s, _)| s.binding()).collect::<Vec<_>>();
        let roots = ResidentBatchRows::new(&bases, 1)?;
        if roots.source() != self.service.source_identity() {
            return Err(Error::State("native resident batch source mismatch".into()));
        }
        let position = usize::try_from(roots.frontier())
            .map_err(|_| Error::Limit("native resident batch frontier overflow".into()))?;
        if position >= self.capacity {
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
        let mut candidates = Vec::with_capacity(rows.len());
        for _ in rows {
            candidates.push(self.allocate_state(false).await?);
        }
        let bindings = rows
            .iter()
            .zip(&candidates)
            .map(|((session, _), candidate)| bind_domains(&session.domains, candidate, position))
            .collect::<Result<Vec<_>>>()?;
        let parameters = rows
            .iter()
            .map(|(_, token)| RuntimeParameters {
                token: *token,
                position,
                valid_tokens: position + 1,
            })
            .collect::<Vec<_>>();
        let staging_bytes = self
            .state_bytes
            .checked_mul(rows.len())
            .ok_or_else(|| Error::Limit("native resident batch admission overflow".into()))?;
        // State initialization uses allocation -> queue. Preserve that order
        // while admitting a cold batch workspace, before selecting a slot.
        let allocation = self.allocation.lock().await;
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
        let submitted = workspace.batch.as_mut().unwrap().program.launch(
            self.service.client(),
            &parameters,
            &bindings,
            ResidentOutput::Logits,
        )?;
        let expected = workspace.program.plan().proposed_domains();
        if submitted.proposals.len() != expected.len() * rows.len()
            || submitted.fused_dispatches == 0
        {
            return Err(Error::State(
                "incomplete native batch effect/dispatch receipt".into(),
            ));
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
        let reads = vec![
            lease_for(&accesses, &submitted.status)?.readback(0, 4)?,
            lease_for(&accesses, &logits.handle)?.readback(0, expected_logits * 4)?,
        ];
        let outputs = bases
            .iter()
            .copied()
            .zip(&candidates)
            .map(|(base, domains)| {
                (
                    base,
                    domains
                        .iter()
                        .flat_map(|d| d.leases.iter())
                        .cloned()
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let (ready, read) = flight
            .complete_cohort_with_readback(&outputs, &reads, |data| {
                if !all_zero(&data[0]) || words(&data[1])?.iter().any(|x| !x.is_finite()) {
                    return Err(Error::State(
                        "native GPU batch status/logits refused".into(),
                    ));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        let logits = words(&read[1])?;
        let mut prepared = Vec::with_capacity(rows.len());
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
                    logits: logits[row * self.vocab as usize..(row + 1) * self.vocab as usize]
                        .to_vec(),
                    logit_steps: 1,
                    bytes: self.vocab as usize * 4 + usize::from(row == 0) * 4,
                    chunk_bytes: self.vocab as usize * 4 + usize::from(row == 0) * 4,
                    ..Default::default()
                },
            });
        }
        Ok(PreparedResidentCohort {
            model: Arc::clone(self),
            roots,
            prepared,
            dispatches: submitted.dispatches,
            fused_dispatches: submitted.fused_dispatches,
        })
    }
}
