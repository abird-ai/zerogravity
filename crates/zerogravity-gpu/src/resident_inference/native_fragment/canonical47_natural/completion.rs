//! Original fair queue/flight/completion and original Session.commit.
use super::loaded::{Canonical47CallReceipt, ColdModel};
use super::*;
use crate::resident_inference::{RuntimeParameters, bind_domains};
use crate::resident_leases::ResidentReadback;
use readback::Layout;
use zerogravity_core::accelerator::program::StateSlot;
use zerogravity_core::device_events::DeviceAccess;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

pub(super) struct PreparedCall {
    pub(super) inference: PreparedInference,
    pub(super) input: Arc<CanonicalCallInput>,
    pub(super) packet: [u8; 12],
    pub(super) receipt: Canonical47CallReceipt,
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
    pub(super) async fn prepare_canonical_call(
        &self,
        cold: Arc<ColdModel>,
        input: Arc<CanonicalCallInput>,
        state: Arc<CanonicalCallState>,
        mut probe: Option<(usize, &mut Option<super::loaded::ProbeRecord>)>,
        mut validate: Option<NaturalValidation<'_>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<PreparedCall> {
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
        let mut slot = ticket.wait().await?;
        not_cancelled(cancelled)?;
        let mut workspace = self.model.workspace.lock().await;
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        state.require_current(self)?;
        state.claim_submission()?;
        let mut program = cold.bind_program(input.clone(), state.clone())?;
        let body = program.full_model_body()?.clone();

        if !Arc::ptr_eq(body.record(), input.record())
            || !Arc::ptr_eq(body.values().record(), input.record())
        {
            return Err(Error::State(
                "canonical current program/record/value owner differs".into(),
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
                "canonical actual48 candidates+immutable packet must be current flight writes"
                    .into(),
            ));
        }
        let diagnostic = validate.is_some();
        let mut reads = if diagnostic {
            body.readbacks()?
        } else {
            body.canonical47_call_status_reads()?
        };
        let body_len = reads.len();
        let duals = body.canonical47_readback_nodes();
        let layout = if diagnostic {
            Some(Layout::new(body_len, 18, 327, &duals)?)
        } else {
            None
        };
        let (prior, candidate) = if diagnostic {
            reads.extend(body.canonical47_call_extra_reads()?);
            let prior = domains(&mut reads, &self.domains)?;
            let candidate = domains(&mut reads, state.candidate())?;
            (prior, candidate)
        } else {
            (Vec::new(), Vec::new())
        };
        let bytes = reads.iter().try_fold(0usize, |a, b| {
            a.checked_add(b.bytes())
                .ok_or_else(|| Error::Limit("canonical D2H overflow".into()))
        })?;
        body.require_readback_capacity(bytes)?;
        if let Some((mode, witness)) = &mut probe {
            body.arm_canonical47_call_fault(*mode)?;
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
        // Complete cold graph and source owners precede node0 D2D/positions,
        // every transport and every actual native/math launch.
        flight.retain_completion_owner(cold.clone())?;
        flight.retain_completion_owner(body.completion_owner())?;
        flight.retain_completion_owner(input.clone())?;
        flight.retain_completion_owner(state.clone())?;
        let displaced = std::mem::replace(&mut workspace.program, program);
        flight.retain_completion_owner(displaced.full_model_body()?.completion_owner())?;
        let submitted = workspace.program.launch_attempt_fragment(
            service.client(),
            runtime,
            &bindings,
            &permit,
            launch_cancelled,
        )?;
        if submitted.logits.is_some() || submitted.proposals.len() != 42 {
            return Err(Error::State(
                "canonical original all42 provisional domains required".into(),
            ));
        }
        body.require_publication_finished()?;
        let mut selected = None;
        let mut completed_packet = None;
        let (ready, raw) = flight
            .complete_with_readback(&outputs, &reads, |raw| {
                let sample = if diagnostic {
                    body.publication_sample(&raw[..body_len])?
                } else {
                    body.canonical47_call_sample_minimal(raw)?
                };
                let p = if diagnostic { 541 } else { 25 };
                let packet: [u8; 12] = raw[p][..]
                    .try_into()
                    .map_err(|_| Error::State("canonical actual12-byte packet".into()))?;
                if diagnostic {
                    body.validate_canonical47_call_extra(&raw[body_len..body_len + 7])?;
                    let end = input.base() + input.rows();
                    for ((ps, pr), (cs, cr)) in prior.iter().zip(&candidate) {
                        if ps != cs {
                            return Err(Error::State(
                                "canonical all42 prior/candidate order".into(),
                            ));
                        }
                        if matches!(ps, StateSlot::Kv(_)) {
                            for (a, b) in raw[pr.clone()].iter().zip(&raw[cr.clone()]) {
                                if a.len() != 1_048_576
                                    || a.len() != b.len()
                                    || a[end * 512 * 4..] != b[end * 512 * 4..]
                                {
                                    return Err(Error::State(
                                        "canonical future KV bytes must carry unchanged".into(),
                                    ));
                                }
                            }
                        }
                    }
                    if let Some(callback) = validate.as_mut() {
                        callback(Canonical47NaturalReadback {
                            record: input.record(),
                            raw,
                            prior: &prior,
                            candidate: &candidate,
                            layout: layout.as_ref().unwrap(),
                            dual_nodes: &duals,
                            selected: sample,
                        })?;
                    }
                }
                selected = Some(sample);
                completed_packet = Some(packet);
                Ok(())
            })
            .await?;
        permit.require_invocation(service, input.call().request(), 24)?;
        input.require_session(self)?;
        slot.finish()?;
        let (id, max_bits) = selected.ok_or_else(|| {
            Error::State("canonical completion omitted actual sampler validation".into())
        })?;
        let packet = completed_packet.ok_or_else(|| {
            Error::State("canonical completion omitted immutable packet bytes".into())
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
        let receipt = Canonical47CallReceipt {
            root: self.binding(),
            selected_id: id,
            max_bits,
            executed_nodes: 255,
            native_launches: body.native_launches(),
            dispatches: submitted.dispatches,
            readback_bytes: bytes,
            consumed_tokens: input.rows(),
        };
        drop(raw);
        drop(submitted);
        drop(displaced);
        Ok(PreparedCall {
            inference,
            input,
            packet,
            receipt,
        })
    }
    pub(super) fn commit_canonical_call(
        &mut self,
        prepared: PreparedCall,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Canonical47CallReceipt, Arc<CommittedPacket>)> {
        not_cancelled(cancelled)?;
        let PreparedCall {
            inference,
            input,
            packet,
            mut receipt,
        } = prepared;
        input.require_session(self)?;
        self.validate_prepared(&inference)?;
        let t = inference
            .readback
            .device_token
            .as_ref()
            .ok_or_else(|| Error::State("canonical completed packet absent".into()))?;
        if t.producer.is_some()
            || t.id != receipt.selected_id
            || t.max_bits != receipt.max_bits
            || packet[..4] != t.id.to_le_bytes()
            || packet[4..8] != t.max_bits.to_le_bytes()
            || packet[8..] != [0; 4]
            || !t.lease.is_initialized()
        {
            return Err(Error::State(
                "canonical private completed proposal packet differs".into(),
            ));
        }
        let committed = self.commit(inference)?;
        let token = committed
            .device_token
            .ok_or_else(|| Error::State("original commit omitted packet".into()))?;
        receipt.root = self.binding();
        let feedback = Arc::new(CommittedPacket {
            source: input.source().source().clone(),
            model: Arc::downgrade(&self.model),
            root: receipt.root,
            packet_bytes: packet,
            allocation_generation: token.lease.allocation_generation(),
            service_generation: token.lease.service_generation(),
            token,
        });
        Ok((receipt, feedback))
    }
}
