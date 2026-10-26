//! Original fair queue/flight/completion and original Session.commit.
use super::loaded::{ColdModel, TextCallReceipt};
use super::*;
use crate::resident_inference::{RuntimeParameters, bind_domains};
use crate::resident_leases::ResidentReadback;

use zerogravity_core::accelerator::program::StateSlot;
use zerogravity_core::device_events::DeviceAccess;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

pub(super) struct PreparedCall {
    pub(super) inference: PreparedInference,
    pub(super) input: Arc<TextCallInput>,
    pub(super) packet: [u8; 12],
    pub(super) receipt: TextCallReceipt,
}
impl PreparedCall {
    pub(super) fn require_completed_for(&self, session: &ResidentSession) -> Result<ResidentToken> {
        self.input.require_session(session)?;
        session.validate_prepared(&self.inference)?;
        let token = self
            .inference
            .readback
            .device_token
            .as_ref()
            .ok_or_else(|| Error::State("short-HF completed packet absent".into()))?;
        if token.producer.is_some()
            || token.id != self.receipt.selected_id
            || token.max_bits != self.receipt.max_bits
            || self.packet[..4] != token.id.to_le_bytes()
            || self.packet[4..8] != token.max_bits.to_le_bytes()
            || self.packet[8..] != [0; 4]
            || !token.lease.is_initialized()
        {
            return Err(Error::State(
                "short-HF private completed proposal packet differs".into(),
            ));
        }
        Ok(token.clone())
    }
}

// Called only after the original publisher has actually committed this root.
// The cloned token keeps the same completed packet lease; no tensor is copied.
pub(super) fn committed_feedback(
    session: &ResidentSession,
    input: Arc<TextCallInput>,
    packet: [u8; 12],
    mut token: ResidentToken,
) -> Arc<CommittedTextPacket> {
    let root = session.binding();
    token.producer = Some(root);
    Arc::new(CommittedTextPacket {
        prompt_tokens: input.prompt_tokens(),
        source: input.source().clone(),
        model: Arc::downgrade(&session.model),
        root,
        packet_bytes: packet,
        allocation_generation: token.lease.allocation_generation(),
        service_generation: token.lease.service_generation(),
        token,
    })
}

fn domains(
    reads: &mut Vec<ResidentReadback>,
    all: &[Domain],
) -> Result<Vec<(StateSlot, std::ops::Range<usize>)>> {
    let mut map = Vec::with_capacity(42);
    for d in all {
        let start = reads.len();
        for l in &d.leases {
            reads.push(l.readback(0, l.bytes())?);
        }
        map.push((d.slot, start..reads.len()));
    }
    Ok(map)
}
impl ResidentSession {
    pub(super) async fn prepare_text_call(
        &self,
        cold: Arc<ColdModel>,
        input: Arc<TextCallInput>,
        state: Arc<TextCallState>,
        mut probe: Option<(usize, &mut Option<super::loaded::ProbeRecord>)>,
        mut validate: Option<TextValidation<'_>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<PreparedCall> {
        let mut wall =
            super::loaded::performance::Phases::new("forward", input.base() as u64, input.rows());
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        state.require_current(self)?;
        let service = &self.model.service;
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: self.model.state_bytes as u64,
            }],
            None,
        )?;
        wall.mark("admission");
        let mut slot = ticket.wait().await?;
        wall.mark("fair_wait");
        not_cancelled(cancelled)?;
        let mut workspace = self.model.workspace.lock().await;
        wall.mark("workspace_wait");
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        state.require_current(self)?;
        state.claim_submission()?;
        // Only the original workspace guard activates a genuine exact geometry.
        // Before submit, failure drops it without queued work; original submit
        // then retains the same run through reads/validation or quarantine.
        let workspace_run = cold.activate_workspace(input.rows())?;
        let mut program = cold
            .bind_program(input.clone(), state.clone(), cancelled)
            .await?;
        wall.mark("bind_program");
        let body = program.full_model_body()?.clone();

        if !Arc::ptr_eq(body.record(), input.record())
            || !Arc::ptr_eq(body.values().record(), input.record())
        {
            return Err(Error::State(
                "short-HF current program/record/value owner differs".into(),
            ));
        }
        let state_ids = self
            .domains
            .iter()
            .chain(state.candidate())
            .flat_map(|d| &d.leases)
            .map(ResidentLease::id)
            .collect::<BTreeSet<_>>();
        program.finalize_full_model_preflight(&state_ids)?;
        let accesses = body.accesses()?.to_vec();
        let packet_lease = body.publication_packet()?.clone();
        let mut outputs = state.outputs();
        outputs.push(packet_lease.clone());
        if outputs.len() != 49
            || outputs.iter().any(|l| {
                !accesses.iter().any(|(a, access)| {
                    a.id() == l.id()
                        && a.allocation_generation() == l.allocation_generation()
                        && *access == DeviceAccess::Write
                })
            })
        {
            return Err(Error::State(
                "short-HF actual48 candidates+immutable packet must be current flight writes"
                    .into(),
            ));
        }
        let diagnostic = validate.is_some();
        let mut layout = body.text_readbacks(diagnostic)?;
        let mut reads = std::mem::take(&mut layout.reads);
        let body_len = reads.len();
        let (prior, candidate) = if diagnostic {
            reads.extend(body.text_extra_reads()?);
            (
                domains(&mut reads, &self.domains)?,
                domains(&mut reads, state.candidate())?,
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let bytes = reads.iter().try_fold(0usize, |a, b| {
            a.checked_add(b.bytes())
                .ok_or_else(|| Error::Limit("short-HF D2H overflow".into()))
        })?;
        body.require_readback_capacity(bytes)?;
        if let Some((mode, witness)) = &mut probe {
            body.arm_text_fault(*mode)?;
            **witness = Some(super::loaded::ProbeRecord::capture(
                &cold, &body, &input, &state, &reads,
            )?);
        }
        let bindings = bind_domains(&self.domains, state.candidate(), input.base())?;
        let runtime = RuntimeParameters {
            token: input.ids()[0],
            position: input.base(),
            valid_tokens: input.rows(),
        };
        let permit = input.permit(self)?;
        not_cancelled(cancelled)?;
        // Drain a submitted serving quantum to actual completion. Disconnect
        // cancellation is checked again by commit; diagnostic cuts stay strict.
        let launch_cancelled = if diagnostic || probe.is_some() {
            cancelled
        } else {
            None
        };
        slot.begin_submission()?;
        let mut flight = service.submit(input.attempt().base().root(), &accesses)?;
        if let Some(run) = workspace_run {
            flight.retain_completion_owner(run)?;
        }
        // Complete cold graph and source owners precede node0 D2D/positions,
        // every transport and every actual native/math launch.
        flight.retain_completion_owner(cold.clone())?;
        flight.retain_completion_owner(body.completion_owner())?;
        flight.retain_completion_owner(input.clone())?;
        flight.retain_completion_owner(state.clone())?;
        let displaced = std::mem::replace(&mut workspace.program, program);
        flight.retain_completion_owner(displaced.full_model_body()?.completion_owner())?;
        wall.mark("preflight");
        let placement = if crate::resident_leases::enqueue_profile_enabled() {
            body.text_enqueue_placement()?
        } else {
            crate::resident_leases::EnqueuePlacement::Caller
        };
        let observer_admission =
            if placement == crate::resident_leases::EnqueuePlacement::DeviceActor {
                Some(
                    input.source().reserve_workspace(
                        crate::resident_leases::ENQUEUE_PROFILE_ADMISSION_BYTES,
                    )?,
                )
            } else {
                None
            };
        let (submitted, device_profile) =
            flight.observe_ordered_enqueue(placement, observer_admission, |flight| {
                crate::resident_inference::rms_scratch::enqueue_with_rms_scratch(
                    &mut workspace,
                    flight,
                    body.text_rms_scope()?,
                    |program| {
                        program.launch_attempt_fragment(
                            service.client(),
                            runtime,
                            &bindings,
                            &permit,
                            launch_cancelled,
                        )
                    },
                )
            })?;
        wall.mark("enqueue");
        if submitted.logits.is_some() || submitted.proposals.len() != 42 {
            return Err(Error::State(
                "short-HF original all42 provisional domains required".into(),
            ));
        }
        body.require_publication_finished()?;
        let mut selected = None;
        let mut completed_packet = None;
        let (ready, raw) = flight
            .complete_with_readback(&outputs, &reads, |raw| {
                let sample = body.validate_text_readbacks(&layout, &raw[..body_len])?;
                let p = layout.packet;
                let packet: [u8; 12] = raw[p][..]
                    .try_into()
                    .map_err(|_| Error::State("short-HF actual12-byte packet".into()))?;
                if diagnostic {
                    body.validate_text_extra(&raw[body_len..body_len + 7])?;
                    let end = input.base() + input.rows();
                    for ((ps, pr), (cs, cr)) in prior.iter().zip(&candidate) {
                        if ps != cs {
                            return Err(Error::State(
                                "short-HF all42 prior/candidate order".into(),
                            ));
                        }
                        if matches!(ps, StateSlot::Kv(_)) {
                            for (a, b) in raw[pr.clone()].iter().zip(&raw[cr.clone()]) {
                                if a.len() != 1_048_576
                                    || a.len() != b.len()
                                    || a[end * 512 * 4..] != b[end * 512 * 4..]
                                {
                                    return Err(Error::State(
                                        "short-HF future KV bytes must carry unchanged".into(),
                                    ));
                                }
                            }
                        }
                    }
                    if let Some(callback) = validate.as_mut() {
                        callback(TextReadback {
                            record: input.record(),
                            raw,
                            prior: &prior,
                            candidate: &candidate,
                            values: &layout.values,
                            sums: &layout.sums,
                            positions: body_len,
                            kv_status: &[3, 7, 11, 15, 19, 23]
                                .into_iter()
                                .enumerate()
                                .map(|(i, l)| (l, body_len + 1 + i))
                                .collect(),
                            packet: layout.packet,
                            selected: sample,
                        })?;
                    }
                }
                selected = Some(sample);
                completed_packet = Some(packet);
                Ok(())
            })
            .await?;
        wall.mark("complete_readback");
        if let Some(profile) = device_profile {
            profile
                .resolve_after_ready(&ready, input.base(), input.rows())
                .await?;
            wall.mark("device_profile_resolve");
        }
        permit.require_invocation(service, input.call(), 0, 24)?;
        input.require_session(self)?;
        slot.finish()?;
        let (id, max_bits) = selected.ok_or_else(|| {
            Error::State("short-HF completion omitted actual sampler validation".into())
        })?;
        let packet = completed_packet.ok_or_else(|| {
            Error::State("short-HF completion omitted immutable packet bytes".into())
        })?;
        let readback = InferenceReadback {
            dispatches: submitted.dispatches,
            bytes,
            chunk_bytes: bytes,
            logit_steps: 1,
            device_token: Some(ResidentToken {
                lease: packet_lease,
                producer: None,
                id,
                max_bits,
            }),
            ..Default::default()
        };
        let inference = PreparedInference {
            attempt: input.attempt().clone(),
            domains: state.candidate().to_vec(),
            ready,
            readback,
        };
        self.validate_prepared(&inference)?;
        let receipt = TextCallReceipt {
            root: self.binding(),
            selected_id: id,
            max_bits,
            executed_nodes: 255,
            native_launches: body.native_launches(),
            dispatches: submitted.dispatches,
            finite_certification: body.finite_certification_metrics(),
            transfer_retirement: body.text_transfer_retirement()?,
            readback_bytes: bytes,
            consumed_tokens: input.rows(),
        };
        drop(raw);
        drop(submitted);
        drop(displaced);
        wall.mark("post_completion");
        wall.succeed();
        Ok(PreparedCall {
            inference,
            input,
            packet,
            receipt,
        })
    }
    pub(super) fn commit_text_call(
        &mut self,
        prepared: PreparedCall,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(TextCallReceipt, Arc<CommittedTextPacket>)> {
        not_cancelled(cancelled)?;
        let token = prepared.require_completed_for(self)?;
        let PreparedCall {
            inference,
            input,
            packet,
            mut receipt,
        } = prepared;
        self.commit(inference)?;
        receipt.root = self.binding();
        let feedback = committed_feedback(self, input, packet, token);
        Ok((receipt, feedback))
    }
}
