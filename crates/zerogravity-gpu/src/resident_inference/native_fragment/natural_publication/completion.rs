//! Original submit/complete/PreparedInference/Session.commit; no shadow publisher.
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_inference::{RuntimeParameters, bind_domains};
use crate::resident_program::ResidentProgram;
use crate::vllm_cuda_qualification::{Vllm08DomainReadback, Vllm08ModelReadback};
use zerogravity_core::accelerator::program::StateSlot;
use zerogravity_core::device_events::DeviceAccess;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

const MAIN_READS: usize = 25 + 3 + 327 + 48 * 4 + 2;
/// Actual common cold/native owner, never an arbitrary replacement for body,
/// input or state owners, which are separately retained before first enqueue.
pub(crate) struct NaturalSubmission {
    pub(crate) input: Arc<NaturalCallInput>,
    pub(crate) state: Arc<NaturalCallState>,
    pub(crate) program: ResidentProgram,
    pub(crate) cold_owner: CompletionOwner,
}
/// Only an original actual completion can construct this private proposal.
pub(crate) struct NaturalPrepared {
    inference: PreparedInference,
    input: Arc<NaturalCallInput>,
    sample: Option<(u32, u32)>,
    packet_bytes: Option<[u8; 12]>,
}
pub(crate) struct NaturalCommitted {
    readback: InferenceReadback,
    feedback: Option<Arc<CommittedBf16Feedback>>,
    root: RootBinding,
}
impl NaturalCommitted {
    pub(crate) fn root(&self) -> RootBinding {
        self.root
    }
    pub(crate) fn feedback(&self) -> Option<&Arc<CommittedBf16Feedback>> {
        self.feedback.as_ref()
    }
    pub(crate) fn readback(&self) -> &InferenceReadback {
        &self.readback
    }
}
fn append_domains(
    reads: &mut Vec<crate::resident_leases::ResidentReadback>,
    domains: &[Domain],
) -> Result<Vec<(StateSlot, std::ops::Range<usize>)>> {
    let mut map = Vec::with_capacity(42);
    for domain in domains {
        let first = reads.len();
        for lease in &domain.leases {
            reads.push(lease.readback(0, lease.bytes())?);
        }
        map.push((domain.slot, first..reads.len()));
    }
    Ok(map)
}
impl ResidentSession {
    /// Returns the same uncommitted PreparedInference used by existing commit.
    /// Public diagnostic callbacks observe completed bytes only, never operands.
    pub(crate) async fn prepare_natural_call<F>(
        &self,
        submission: NaturalSubmission,
        validate: F,
        cancelled: Option<&AtomicBool>,
    ) -> Result<NaturalPrepared>
    where
        F: FnMut(Vllm08ModelReadback<'_>) -> Result<()>,
    {
        let NaturalSubmission {
            input,
            state,
            program,
            cold_owner,
        } = submission;
        self.prepare_natural_call_bound(
            input,
            state,
            cold_owner,
            || Ok(program),
            validate,
            cancelled,
        )
        .await
    }

    /// Bind shared temporary storage only while holding the SAME original
    /// fair ticket and workspace lock. No new traversal or publisher.
    pub(crate) async fn prepare_natural_call_bound<F, B>(
        &self,
        input: Arc<NaturalCallInput>,
        state: Arc<NaturalCallState>,
        cold_owner: CompletionOwner,
        bind_program: B,
        mut validate: F,
        cancelled: Option<&AtomicBool>,
    ) -> Result<NaturalPrepared>
    where
        F: FnMut(Vllm08ModelReadback<'_>) -> Result<()>,
        B: FnOnce() -> Result<ResidentProgram>,
    {
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        state.require_current(self)?;
        if !Arc::ptr_eq(state.input(), &input) {
            return Err(Error::State(
                "natural state and producer input seals differ".into(),
            ));
        }
        let service = &self.model.service;
        let ticket = service.queue().submit(
            service.budget(),
            zerogravity_core::resource_arbiter::ResourceOwner {
                model: service.owner().model.clone(),
                session: format!("session-{}", input.attempt().base().session().get()),
            },
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: self.model.state_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        // Match the original fair-queue → workspace order; no lock across wait.
        let mut workspace = self.model.workspace.lock().await;
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        state.require_current(self)?;
        state.claim_reusable_submission()?;
        let program = bind_program()?;
        let body = program.full_model_body()?.clone();
        if !Arc::ptr_eq(body.record(), input.record())
            || !Arc::ptr_eq(body.values().record(), input.record())
        {
            return Err(Error::State(
                "natural original program/value record differs from trusted input".into(),
            ));
        }
        self.model.validate_domains(&self.domains, true)?;
        self.model.validate_domains(state.candidate(), false)?;
        let candidate_outputs = state.outputs();
        if candidate_outputs.len() != 48 {
            return Err(Error::State(
                "natural exact48 candidate Ready outputs".into(),
            ));
        }
        let accesses = body.accesses()?.to_vec();
        let packet = body.publication_packet()?.clone();
        for lease in candidate_outputs.iter().chain(std::iter::once(&packet)) {
            if !accesses.iter().any(|(actual, access)| {
                actual.id() == lease.id()
                    && actual.allocation_generation() == lease.allocation_generation()
                    && *access == DeviceAccess::Write
            }) {
                return Err(Error::State(
                    "natural actual submitted write cohort lacks candidate/packet".into(),
                ));
            }
        }
        let mut promoted = candidate_outputs;
        if input.slot().emits() {
            promoted.push(packet.clone());
        }
        let mut reads = body.readbacks()?;
        if reads.len() != MAIN_READS {
            return Err(Error::State(
                "natural original full24 numerical readback coverage".into(),
            ));
        }
        let extra_start = reads.len();
        reads.extend(body.publication_extra_reads()?);
        let extra = extra_start..reads.len();
        let prior = append_domains(&mut reads, &self.domains)?;
        let candidate = append_domains(&mut reads, state.candidate())?;
        let bytes = reads.iter().try_fold(0usize, |sum, r| {
            sum.checked_add(r.bytes())
                .ok_or_else(|| Error::Limit("natural complete D2H size overflow".into()))
        })?;
        body.require_readback_capacity(bytes)?;
        let rms_nodes = input
            .record()
            .plan()
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(index, node)| match &node.operation {
                zerogravity_core::accelerator::program::PlanOperation::TypedVllm08 {
                    descriptor,
                    ..
                } if matches!(
                    descriptor.kind(),
                    zerogravity_core::accelerator::program::Vllm08Kind::ResidualRms(_)
                ) =>
                {
                    Some(index)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if rms_nodes.len() != 48 {
            return Err(Error::State("natural original48 dual RMS sites".into()));
        }
        let bindings = bind_domains(&self.domains, state.candidate(), input.slot().base())?;
        let runtime = RuntimeParameters {
            token: input.ids()[0],
            position: input.slot().base(),
            valid_tokens: input.slot().rows(),
        };
        let permit = input.permit(self)?;
        slot.begin_submission()?;
        let mut flight = service.submit(input.attempt().base().root(), &accesses)?;
        flight.retain_completion_owner(cold_owner)?;
        flight.retain_completion_owner(body.completion_owner())?;
        flight.retain_completion_owner(input.clone())?;
        flight.retain_completion_owner(state.clone())?;
        // The original workspace and original node loop remain authoritative.
        let displaced = std::mem::replace(&mut workspace.program, program);
        let submitted = workspace.program.launch_attempt_fragment(
            service.client(),
            runtime,
            &bindings,
            &permit,
            cancelled,
        )?;
        if submitted.logits.is_some() || submitted.proposals.len() != 42 {
            return Err(Error::State(
                "natural full24 must propose exactly all42 original domains".into(),
            ));
        }
        // launch_attempt_fragment already enqueues/counts the exact same-map
        // sampler in its ORIGINAL postlude after Head/canonical certificates.
        body.require_publication_finished()?;
        let mut selected = None;
        let mut completed_packet = None;
        let (ready, raw) = flight
            .complete_with_readback(&promoted, &reads, |raw| {
                selected = Some(body.publication_sample(&raw[..MAIN_READS])?);
                completed_packet =
                    Some(raw[547][..].try_into().map_err(|_| {
                        Error::State("natural original completed packet bytes".into())
                    })?);
                body.validate_publication_extra(&raw[extra.clone()])?;
                let prior_domains = prior
                    .iter()
                    .map(|(slot, range)| Vllm08DomainReadback {
                        slot: *slot,
                        buffers: &raw[range.clone()],
                    })
                    .collect::<Vec<_>>();
                let domains = candidate
                    .iter()
                    .map(|(slot, range)| Vllm08DomainReadback {
                        slot: *slot,
                        buffers: &raw[range.clone()],
                    })
                    .collect::<Vec<_>>();
                let end = input.slot().base() + input.slot().rows();
                for (p, c) in prior_domains.iter().zip(&domains) {
                    if p.slot != c.slot {
                        return Err(Error::State(
                            "natural original all42 canonical domain order differs".into(),
                        ));
                    }
                    if matches!(p.slot, StateSlot::Kv(_)) {
                        for (prior, candidate) in p.buffers.iter().zip(c.buffers) {
                            if prior.len() != 512 * 512 * 4
                                || candidate.len() != prior.len()
                                || prior[end * 512 * 4..] != candidate[end * 512 * 4..]
                            {
                                return Err(Error::State(
                                    "natural original future canonical KV padding was not carried"
                                        .into(),
                                ));
                            }
                        }
                    }
                }
                let rms = rms_nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| (*node, &raw[355 + i * 4..355 + (i + 1) * 4]))
                    .collect::<Vec<_>>();
                validate(Vllm08ModelReadback {
                    phase: input.slot().index(),
                    record: input.record(),
                    values: &raw[28..355],
                    input_rms: &raw[25..28],
                    rms: &rms,
                    prior_domains: &prior_domains,
                    domains: &domains,
                    greedy: &raw[547..549],
                })
            })
            .await?;
        // The immutable session borrow covered every enqueue and completion.
        permit.require_invocation(service, input.call().request(), 24)?;
        input.require_session(self)?;
        slot.finish()?;
        let selected = selected.ok_or_else(|| {
            Error::State("natural original trusted completion omitted packet validation".into())
        })?;
        let mut readback = InferenceReadback {
            dispatches: submitted.dispatches,
            bytes,
            chunk_bytes: bytes,
            state_only_steps: usize::from(!input.slot().emits()),
            logit_steps: usize::from(input.slot().emits()),
            ..Default::default()
        };
        if input.slot().emits() {
            readback.device_token = Some(ResidentToken {
                lease: packet,
                producer: None,
                id: selected.0,
                max_bits: selected.1,
            });
        }
        let inference = PreparedInference {
            attempt: input.attempt().clone(),
            domains: state.candidate().to_vec(),
            ready,
            readback,
        };
        self.validate_prepared(&inference)?;
        drop(raw);
        drop(displaced);
        drop(submitted);
        let sample = input_slot_sample(selected, input.slot());
        let packet_bytes = if sample.is_some() {
            Some(
                completed_packet
                    .ok_or_else(|| Error::State("natural actual packet readback absent".into()))?,
            )
        } else {
            None
        };
        Ok(NaturalPrepared {
            inference,
            input,
            sample,
            packet_bytes,
        })
    }
    /// All fallible source/packet/current-root checks precede original commit.
    /// Prefix64 intentionally publishes state only and emits no sampled token.
    pub(crate) fn commit_natural_call(
        &mut self,
        prepared: NaturalPrepared,
        cancelled: Option<&AtomicBool>,
    ) -> Result<NaturalCommitted> {
        not_cancelled(cancelled)?;
        let NaturalPrepared {
            inference,
            input,
            sample,
            packet_bytes,
        } = prepared;
        input.require_session(self)?;
        self.validate_prepared(&inference)?;
        match (
            sample,
            packet_bytes.as_ref(),
            inference.readback.device_token.as_ref(),
        ) {
            (Some((id, bits)), Some(raw), Some(token))
                if input.slot().emits()
                    && token.producer.is_none()
                    && token.id == id
                    && token.max_bits == bits
                    && token.lease.bytes() == 12
                    && token.lease.role() == BufferRole::Output
                    && token.lease.is_initialized()
                    && raw[..4] == id.to_le_bytes()
                    && raw[4..8] == bits.to_le_bytes()
                    && raw[8..] == [0; 4] => {}
            (None, None, None) if !input.slot().emits() => {}
            _ => {
                return Err(Error::State(
                    "natural original packet/emission proposal differs".into(),
                ));
            }
        }
        let readback = self.commit(inference)?;
        let root = self.binding();
        // This is the original commit-returned token; no public token/id input.
        let feedback = readback
            .device_token
            .as_ref()
            .zip(sample.zip(packet_bytes))
            .map(|(token, ((id, bits), packet_bytes))| {
                Arc::new(CommittedBf16Feedback {
                    source: input.source().clone(),
                    model: Arc::downgrade(&self.model),
                    token: token.clone(),
                    root,
                    selected_id: id,
                    selected_bits: bits,
                    packet_bytes,
                    allocation_generation: token.lease.allocation_generation(),
                    service_generation: token.lease.service_generation(),
                })
            });
        Ok(NaturalCommitted {
            readback,
            feedback,
            root,
        })
    }
}
fn input_slot_sample(sample: (u32, u32), slot: NaturalSlot) -> Option<(u32, u32)> {
    slot.emits().then_some(sample)
}
