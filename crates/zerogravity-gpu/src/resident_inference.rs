//! Physical execution of the engine's one recorded schedule and immutable roots.
//! Arithmetic and layer ordering belong to ForwardPlan; this owner only binds
//! leases, private state, actual completion and the common logical authority.
#[cfg(feature = "qualification")]
#[path = "resident_alias_qualification.rs"]
mod alias_qualification;
#[cfg(feature = "portable-checkpoint")]
#[path = "resident_checkpoint.rs"]
mod checkpoint;
#[cfg(all(feature = "portable-checkpoint", feature = "qualification"))]
pub(crate) use checkpoint::qualify as qualify_checkpoint;
#[cfg(feature = "qualification")]
#[path = "resident_context_qualification.rs"]
mod context_qualification;
#[cfg(feature = "qualification")]
#[path = "resident_prefill_qualification.rs"]
mod prefill_qualification;
#[cfg(feature = "qualification")]
#[path = "resident_range_qualification.rs"]
mod range_qualification;
#[cfg(feature = "qualification")]
#[path = "resident_view_qualification.rs"]
mod view_qualification;
use crate::backend::validate_provider;
use crate::resident_leases::{ResidentLease, ResidentReady, ResidentService};
use crate::resident_program::{
    AllocationInit, DeviceBuffer, ResidentBufferRole, ResidentOutput, ResidentPreflight,
    ResidentProgram, RuntimeParameters, StateBinding, StateBuffers, StateLayout, StateRequirement,
};
use async_lock::Mutex;
use cubecl::{Device, bytes::Bytes, server::Handle};
#[cfg(feature = "qualification")]
use std::path::Path;
use std::sync::Arc;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::program::{
    ForwardPlan, ParameterSlot, PlanObservation as LayerObservation, ValueShape,
};
use zerogravity_core::accelerator::program::{PlanOperation, StateSlot};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};
use zerogravity_core::session_root::{RootAttempt, RootBinding, SessionAuthority};
use zerogravity_core::source::TensorSource;
#[cfg(any(feature = "qualification", feature = "portable-checkpoint"))]
use zerogravity_model_qwen35::LoadedQwenModel;

struct Workspace {
    program: ResidentProgram,
    greedy: [Handle; 2],
    leases: Vec<(ResidentLease, DeviceAccess)>,
    state_accesses: Vec<(ResidentLease, DeviceAccess)>,
    logit_accesses: Vec<(ResidentLease, DeviceAccess)>,
}
/// Shared immutable source and serialized physical workspace, independent roots.
pub struct ResidentModel {
    service: Arc<ResidentService>,
    workspace: Mutex<Workspace>,
    allocation: Mutex<()>,
    states: Vec<StateRequirement>,
    state_bytes: usize,
    capacity: usize,
    vocab: u64,
    #[cfg(feature = "hardware-abort-probe")]
    abort_after_launch: std::sync::atomic::AtomicUsize,
    #[cfg(feature = "hardware-abort-probe")]
    launch_seen: std::sync::atomic::AtomicUsize,
}
#[derive(Clone)]
struct Domain {
    slot: StateSlot,
    buffers: StateBuffers,
    leases: Vec<ResidentLease>,
    generations: Vec<u64>,
}
#[derive(Clone)]
pub struct ResidentSnapshot {
    binding: RootBinding,
    service_generation: u64,
    capacity: usize,
    domains: Vec<Domain>,
}
pub struct ResidentSession {
    model: Arc<ResidentModel>,
    authority: SessionAuthority,
    domains: Vec<Domain>,
}
#[derive(Default)]
pub struct InferenceReadback {
    pub logits: Vec<f32>,
    pub domains: Vec<(StateSlot, Vec<f32>)>,
    pub observations: Vec<LayerObservation>,
    pub dispatches: usize,
    pub bytes: usize,
    pub chunk_bytes: usize,
    pub state_only_steps: usize,
    pub logit_steps: usize,
    pub device_token: Option<ResidentToken>,
}
#[derive(Clone)]
pub struct ResidentToken {
    lease: ResidentLease,
    producer: Option<RootBinding>,
    pub id: u32,
    pub max_bits: u32,
}
#[derive(Clone, Copy)]
enum Readout {
    None,
    Logits,
    Greedy,
}
struct StepInput<'a> {
    token: u32,
    device: Option<&'a ResidentToken>,
    readout: Readout,
    diagnostic: bool,
}
pub struct PreparedInference {
    attempt: RootAttempt,
    domains: Vec<Domain>,
    ready: ResidentReady,
    readback: InferenceReadback,
}

fn words(bytes: &Bytes) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::State("resident f32 readback alignment".into()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_ne_bytes(b.try_into().unwrap())))
        .collect())
}
fn all_zero(bytes: &Bytes) -> bool {
    bytes.iter().all(|b| *b == 0)
}
fn same_handle(a: &Handle, b: &Handle) -> bool {
    let a = a.clone().binding();
    let b = b.clone().binding();
    a.service == b.service && a.memory.id() == b.memory.id() && a.range() == b.range()
}
fn lease_for<'a>(
    leases: &'a [(ResidentLease, DeviceAccess)],
    handle: &Handle,
) -> Result<&'a ResidentLease> {
    leases
        .iter()
        .find(|(l, _)| same_handle(&l.handle(), handle))
        .map(|(l, _)| l)
        .ok_or_else(|| Error::State("unadmitted resident graph handle".into()))
}

impl ResidentModel {
    pub async fn load(
        device: Device,
        model: Arc<dyn TensorSource>,
        plan: ForwardPlan,
        provider_cap: usize,
        device_cap: u64,
    ) -> Result<Arc<Self>> {
        validate_provider(&device)?;
        let (capacity, vocab, preflight) = preflight_profile(model.as_ref(), &plan)?;
        if preflight.source != model.source_identity() {
            return Err(Error::State("resident preflight source mismatch".into()));
        }
        let greedy_sizes = crate::resident_greedy::workspace_bytes(vocab as usize)?;
        let greedy_bytes = greedy_sizes.iter().sum::<usize>();
        // Require source/workspace and two private states before any GPU allocation.
        if preflight
            .total_with_state()?
            .checked_add(greedy_bytes + 12)
            .is_none_or(|n| n > provider_cap)
        {
            return Err(Error::Limit(
                "complete resident graph does not fit provider cap".into(),
            ));
        }
        ResidentProgram::ensure_capability(&device.client())?;
        let service = ResidentService::new(
            device,
            model.source_identity(),
            ResourceOwner {
                model: format!("qwen-source-{:?}", model.source_identity()),
                session: "resident-program".into(),
            },
            provider_cap,
            device_cap,
        )?;
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: preflight.owned_bytes()? as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut initialization = service.begin_initialization(0)?;
        let mut leases = Vec::new();
        let program = ResidentProgram::prepare_with_allocator(
            service.client(),
            model,
            plan,
            |role, bytes, data| {
                let (role, access) = match role {
                    ResidentBufferRole::Source => {
                        (BufferRole::ImmutableWeights, DeviceAccess::Read)
                    }
                    ResidentBufferRole::Value => (BufferRole::Activation, DeviceAccess::Write),
                    ResidentBufferRole::Workspace | ResidentBufferRole::Status => {
                        (BufferRole::Scratch, DeviceAccess::Write)
                    }
                    ResidentBufferRole::Parameter => (BufferRole::Activation, DeviceAccess::Write),
                };
                let lease = match data {
                    AllocationInit::Empty => initialization.allocate_empty(bytes, role)?,
                    AllocationInit::Upload(data) => {
                        if data.len() != bytes {
                            return Err(Error::State("resident initializer byte mismatch".into()));
                        }
                        initialization.upload_bytes(data, role)?
                    }
                };
                let handle = lease.handle();
                leases.push((lease, access));
                Ok(handle)
            },
        )?;
        for handle in program.owned_handles() {
            lease_for(&leases, &handle)?;
        }
        let mut greedy_handles = Vec::new();
        for bytes in greedy_sizes {
            let lease = initialization.allocate_empty(bytes, BufferRole::Scratch)?;
            greedy_handles.push(lease.handle());
            leases.push((lease, DeviceAccess::Write));
        }
        let greedy: [Handle; 2] = greedy_handles
            .try_into()
            .map_err(|_| Error::State("greedy workspace coverage".into()))?;
        if program.source_identity() != service.source_identity()
            || leases.iter().map(|(l, _)| l.bytes()).sum::<usize>()
                != program.budget().owned_bytes()? + greedy_bytes
        {
            return Err(Error::State(
                "resident exact allocation preflight mismatch".into(),
            ));
        }
        let select = |output| {
            let handles = program.execution_handles(output);
            leases
                .iter()
                .filter(|(lease, _)| handles.iter().any(|h| same_handle(h, &lease.handle())))
                .cloned()
                .collect()
        };
        let state_accesses = select(ResidentOutput::StateOnly);
        let logit_accesses = select(ResidentOutput::Logits);
        let ready = initialization.complete_sync().await?;
        if ready.outputs().len() != leases.len() {
            return Err(Error::State("incomplete source preparation".into()));
        }
        slot.finish()?;
        Ok(Arc::new(Self {
            service,
            states: preflight.states,
            state_bytes: preflight.prior_state_bytes,
            capacity,
            vocab,
            #[cfg(feature = "hardware-abort-probe")]
            abort_after_launch: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "hardware-abort-probe")]
            launch_seen: std::sync::atomic::AtomicUsize::new(0),
            workspace: Mutex::new(Workspace {
                program,
                greedy,
                leases,
                state_accesses,
                logit_accesses,
            }),
            allocation: Mutex::new(()),
        }))
    }
    #[cfg(feature = "qualification")]
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        &self.service
    }
    pub async fn session(self: &Arc<Self>) -> Result<ResidentSession> {
        let authority = SessionAuthority::new(self.service.source_identity())?;
        let domains = self.allocate_state(true).await?;
        Ok(ResidentSession {
            model: self.clone(),
            authority,
            domains,
        })
    }
    async fn allocate_state(&self, zero: bool) -> Result<Vec<Domain>> {
        let _admission = self.allocation.lock().await;
        self.service.ensure_usable()?;
        if self.state_bytes > self.service.remaining_capacity()? {
            return Err(Error::Limit("resident state admission exhausted".into()));
        }
        let mut output = Vec::new();
        if zero {
            let ticket = self.service.queue().submit(
                self.service.budget(),
                self.service.owner().clone(),
                &[AllocationRequest {
                    kind: AllocationClass::RecurrentState,
                    bytes: self.state_bytes as u64,
                }],
                None,
            )?;
            let mut slot = ticket.wait().await?;
            slot.begin_submission()?;
            let mut initialization = self.service.begin_initialization(0)?;
            for r in &self.states {
                let mut allocate = |n: usize| {
                    initialization.upload_bytes(&vec![0u8; n * 4], BufferRole::RecurrentState)
                };
                output.push(make_domain(r, &mut allocate)?);
            }
            drop(initialization.complete_sync().await?);
            slot.finish()?;
        } else {
            for r in &self.states {
                let mut allocate = |n: usize| {
                    self.service
                        .allocate_uninitialized(n * 4, BufferRole::RecurrentState)
                };
                output.push(make_domain(r, &mut allocate)?);
            }
        }
        Ok(output)
    }
    fn validate_domains(&self, domains: &[Domain], initialized: bool) -> Result<()> {
        if domains.len() != self.states.len() {
            return Err(Error::State("resident domain coverage".into()));
        }
        for (domain, requirement) in domains.iter().zip(&self.states) {
            if domain.slot != requirement.slot || domain.generations.len() != domain.leases.len() {
                return Err(Error::State(
                    "resident domain slot/generation coverage".into(),
                ));
            }
            let handles = domain.buffers.handles();
            if handles.len() != domain.leases.len() {
                return Err(Error::State("resident physical domain coverage".into()));
            }
            for ((lease, expected), handle) in
                domain.leases.iter().zip(&domain.generations).zip(handles)
            {
                if !Arc::ptr_eq(lease.service(), &self.service)
                    || lease.service_generation() != self.service.generation()
                    || lease.source_identity() != self.service.source_identity()
                    || lease.allocation_generation() != *expected
                    || !same_handle(&lease.handle(), &handle)
                    || (initialized && !lease.is_initialized())
                {
                    return Err(Error::State(
                        "resident domain allocation/service generation mismatch".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    async fn execute(
        &self,
        base: RootBinding,
        position: usize,
        prior: &[Domain],
        candidate: Vec<Domain>,
        input: StepInput<'_>,
    ) -> Result<(Vec<Domain>, ResidentReady, InferenceReadback)> {
        self.service.ensure_usable()?;
        if base.source() != self.service.source_identity() || position >= self.capacity {
            return Err(Error::State("resident source/frontier mismatch".into()));
        }
        let StepInput {
            token,
            device,
            readout,
            diagnostic,
        } = input;
        if u64::from(token) >= self.vocab {
            return Err(Error::Malformed(
                "resident token exceeds original vocabulary".into(),
            ));
        }
        self.validate_domains(prior, true)?;
        self.validate_domains(&candidate, false)?;
        let bindings = bind_domains(prior, &candidate, position)?;
        let ticket = self.service.queue().submit(
            self.service.budget(),
            ResourceOwner {
                model: self.service.owner().model.clone(),
                session: format!("session-{}", base.session().get()),
            },
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: self.state_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        let mut workspace = self.workspace.lock().await;
        let parameters = RuntimeParameters {
            token,
            position,
            valid_tokens: position + 1,
        };
        workspace
            .program
            .validate_runtime(self.service.client(), parameters, &bindings)?;
        let output = if matches!(readout, Readout::None) {
            ResidentOutput::StateOnly
        } else {
            ResidentOutput::Logits
        };
        let mut accesses = match output {
            ResidentOutput::StateOnly => workspace.state_accesses.clone(),
            ResidentOutput::Logits => workspace.logit_accesses.clone(),
        };
        if let Some(token) = device {
            if token.producer != Some(base) || !Arc::ptr_eq(token.lease.service(), &self.service) {
                return Err(Error::State("stale/foreign device token".into()));
            }
            accesses.push((token.lease.clone(), DeviceAccess::Read));
        }
        let packet = if matches!(readout, Readout::Greedy) {
            for handle in &workspace.greedy {
                accesses.push((
                    lease_for(&workspace.leases, handle)?.clone(),
                    DeviceAccess::Write,
                ));
            }
            let lease = self
                .service
                .allocate_uninitialized(12, BufferRole::Output)?;
            accesses.push((lease.clone(), DeviceAccess::Write));
            Some(lease)
        } else {
            None
        };
        for domain in prior {
            accesses.extend(
                domain
                    .leases
                    .iter()
                    .cloned()
                    .map(|l| (l, DeviceAccess::Read)),
            );
        }
        for domain in &candidate {
            accesses.extend(
                domain
                    .leases
                    .iter()
                    .cloned()
                    .map(|l| (l, DeviceAccess::Write)),
            );
        }
        slot.begin_submission()?;
        let flight = self.service.submit(base.root(), &accesses)?;
        let mut submitted = if let Some(token) = device {
            workspace.program.launch_device_token(
                self.service.client(),
                parameters,
                &bindings,
                &DeviceBuffer::new(token.lease.handle(), 1),
                output,
            )?
        } else {
            workspace
                .program
                .launch(self.service.client(), parameters, &bindings, output)?
        };
        let expected = workspace.program.plan().proposed_domains();
        if submitted.proposals.len() != expected.len() {
            return Err(Error::State("incomplete GPU state effect receipt".into()));
        }
        for ((version, buffers), expected) in submitted.proposals.iter().zip(expected) {
            let domain = candidate
                .iter()
                .find(|d| d.slot == version.slot)
                .ok_or_else(|| Error::State("foreign GPU state effect receipt".into()))?;
            let actual = buffers.handles();
            let target = domain.buffers.handles();
            if version.slot != expected.slot
                || version.ordinal != expected.ordinal
                || actual.len() != target.len()
                || actual.iter().zip(target).any(|(a, b)| !same_handle(a, &b))
            {
                return Err(Error::State(
                    "GPU state effect version/allocation mismatch".into(),
                ));
            }
        }
        if let Some(packet) = &packet {
            let logits = submitted
                .logits
                .as_ref()
                .ok_or_else(|| Error::State("missing greedy logits".into()))?;
            submitted.dispatches += crate::resident_greedy::launch(
                self.service.client(),
                &logits.handle,
                self.vocab as usize,
                &workspace.greedy[0],
                &workspace.greedy[1],
                &packet.handle(),
                &submitted.status,
            )?;
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
        let mut cohort = Vec::new();
        let add = |cohort: &mut Vec<_>, handle: &Handle, count: usize| -> Result<()> {
            let lease = lease_for(&accesses, handle)?;
            cohort.push(lease.readback(0, count * 4)?);
            Ok(())
        };
        add(&mut cohort, &submitted.status, 1)?;
        if diagnostic {
            for (block, blocks, row, rows) in &submitted.quant_status {
                add(&mut cohort, block, *blocks)?;
                add(&mut cohort, row, *rows)?;
            }
            for (handle, n) in &submitted.embedding_status {
                add(&mut cohort, handle, *n)?;
            }
        }
        let statuses = cohort.len();
        match readout {
            Readout::Logits => {
                let logits = submitted
                    .logits
                    .as_ref()
                    .ok_or_else(|| Error::State("missing resident logits".into()))?;
                add(&mut cohort, &logits.handle, logits.elements)?;
            }
            Readout::Greedy => {
                let p = packet.as_ref().unwrap();
                cohort.push(p.readback(0, 12)?);
            }
            Readout::None => {
                if submitted.logits.is_some() {
                    return Err(Error::State("state-only graph returned logits".into()));
                }
            }
        }
        let domain_start = cohort.len();
        if diagnostic {
            for domain in &candidate {
                for l in &domain.leases {
                    cohort.push(l.readback(0, l.bytes())?);
                }
            }
        }
        let observation_start = cohort.len();
        let mut observation_keys = Vec::new();
        if diagnostic {
            let nodes = match output {
                ResidentOutput::StateOnly => workspace.program.plan().state_nodes(),
                ResidentOutput::Logits => workspace.program.plan().nodes(),
            };
            for node in nodes {
                match &node.operation {
                    PlanOperation::Observe { layer, stage } => {
                        let value = &workspace.program.values()[node.inputs[0].0 as usize];
                        add(&mut cohort, &value.handle, value.elements)?;
                        observation_keys.push((*layer, *stage, value.elements));
                    }
                    PlanOperation::ObserveState {
                        layer,
                        stage,
                        state,
                    } => {
                        let domain = candidate
                            .iter()
                            .find(|d| d.slot == state.slot)
                            .ok_or_else(|| Error::State("missing observed domain".into()))?;
                        let l = &domain.leases[0];
                        cohort.push(l.readback(0, l.bytes())?);
                        observation_keys.push((*layer, *stage, l.bytes() / 4));
                    }
                    _ => {}
                }
            }
        }
        let mut promoted = candidate
            .iter()
            .flat_map(|d| d.leases.iter().cloned())
            .collect::<Vec<_>>();
        promoted.extend(packet.iter().cloned());
        let expected_readback_bytes = cohort.iter().map(|item| item.bytes()).sum();
        let (ready, read) = flight
            .complete_with_readback(&promoted, &cohort, |values| {
                if values[..statuses].iter().any(|b| !all_zero(b)) {
                    return Err(Error::State(
                        "resident GPU status rejected proposed graph".into(),
                    ));
                }
                if matches!(readout, Readout::Logits)
                    && !words(&values[statuses])?.iter().all(|x| x.is_finite())
                {
                    return Err(Error::State("nonfinite resident logits".into()));
                }
                if matches!(readout, Readout::Greedy) {
                    let packet = &values[statuses];
                    let id = u32::from_ne_bytes(packet[0..4].try_into().unwrap());
                    let flags = u32::from_ne_bytes(packet[8..12].try_into().unwrap());
                    if u64::from(id) >= self.vocab || flags != 0 {
                        return Err(Error::State("invalid device greedy packet".into()));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        let mut result = InferenceReadback {
            logits: if matches!(readout, Readout::Logits) {
                words(&read[statuses])?
            } else {
                Vec::new()
            },
            dispatches: submitted.dispatches,
            bytes: expected_readback_bytes,
            chunk_bytes: expected_readback_bytes,
            state_only_steps: usize::from(output == ResidentOutput::StateOnly),
            logit_steps: usize::from(output == ResidentOutput::Logits),
            ..Default::default()
        };
        if let Some(lease) = packet {
            result.device_token = Some(ResidentToken {
                lease,
                producer: None,
                id: u32::from_ne_bytes(read[statuses][0..4].try_into().unwrap()),
                max_bits: u32::from_ne_bytes(read[statuses][4..8].try_into().unwrap()),
            });
        }
        if diagnostic {
            let mut cursor = domain_start;
            for domain in &candidate {
                let mut flat = Vec::new();
                for l in &domain.leases {
                    let data = words(&read[cursor])?;
                    match domain.buffers {
                        StateBuffers::Vector(_) => flat.extend(data),
                        StateBuffers::Kv { .. } => {
                            let active = l.bytes() / 4 / self.capacity * (position + 1);
                            if data[active..].iter().any(|x| x.to_bits() != 0) {
                                return Err(Error::State(
                                    "resident canonical KV tail is not +0".into(),
                                ));
                            }
                            flat.extend_from_slice(&data[..active]);
                        }
                    }
                    cursor += 1;
                }
                result.domains.push((domain.slot, flat));
            }
            for ((layer, stage, _), data) in
                observation_keys.into_iter().zip(&read[observation_start..])
            {
                let mut values = words(data)?;
                if matches!(stage, "full_qk_dots" | "full_scores" | "full_probs") {
                    let heads = values.len() / self.capacity;
                    values = (0..heads)
                        .flat_map(|h| {
                            values[h * self.capacity..h * self.capacity + position + 1]
                                .iter()
                                .copied()
                        })
                        .collect();
                }
                result.observations.push(LayerObservation {
                    position,
                    layer,
                    stage,
                    values,
                });
            }
        }
        Ok((candidate, ready, result))
    }
}

fn make_domain(
    r: &StateRequirement,
    allocate: &mut impl FnMut(usize) -> Result<ResidentLease>,
) -> Result<Domain> {
    let (leases, buffers) = match r.layout {
        StateLayout::Vector { elements } => {
            let l = allocate(elements)?;
            let b = DeviceBuffer::new(l.handle(), elements);
            (vec![l], StateBuffers::Vector(b))
        }
        StateLayout::Kv { slab, capacity } => {
            let k = allocate(slab * capacity)?;
            let v = allocate(slab * capacity)?;
            let b = StateBuffers::Kv {
                keys: DeviceBuffer::new(k.handle(), slab * capacity),
                values: DeviceBuffer::new(v.handle(), slab * capacity),
            };
            (vec![k, v], b)
        }
    };
    let generations = leases
        .iter()
        .map(ResidentLease::allocation_generation)
        .collect();
    Ok(Domain {
        slot: r.slot,
        buffers,
        leases,
        generations,
    })
}
fn bind_domains(
    prior: &[Domain],
    candidate: &[Domain],
    position: usize,
) -> Result<Vec<StateBinding>> {
    if prior.len() != candidate.len() {
        return Err(Error::State("resident state domain count".into()));
    }
    prior
        .iter()
        .zip(candidate)
        .map(|(p, c)| {
            if p.slot != c.slot {
                return Err(Error::State("resident state domain order".into()));
            }
            Ok(StateBinding {
                slot: p.slot,
                prior: p.buffers.clone(),
                candidate: c.buffers.clone(),
                prior_tokens: position,
            })
        })
        .collect()
}
impl ResidentSnapshot {
    pub async fn read_domains(&self) -> Result<Vec<(StateSlot, Vec<f32>)>> {
        let service = self
            .domains
            .first()
            .and_then(|d| d.leases.first())
            .ok_or_else(|| Error::State("empty resident snapshot".into()))?
            .service()
            .clone();
        service.ensure_usable()?;
        if service.generation() != self.service_generation
            || service.source_identity() != self.binding.source()
        {
            return Err(Error::State("snapshot service generation mismatch".into()));
        }
        let leases = self
            .domains
            .iter()
            .flat_map(|d| d.leases.iter().cloned())
            .collect::<Vec<_>>();
        let accesses = leases
            .iter()
            .cloned()
            .map(|l| (l, DeviceAccess::Read))
            .collect::<Vec<_>>();
        let cohort = leases
            .iter()
            .map(|l| l.readback(0, l.bytes()))
            .collect::<Result<Vec<_>>>()?;
        let ticket = service.queue().submit(
            service.budget(),
            ResourceOwner {
                model: service.owner().model.clone(),
                session: format!("snapshot-{}", self.binding.session().get()),
            },
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: leases.iter().map(|l| l.bytes() as u64).sum(),
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let flight = service.submit(self.binding.root(), &accesses)?;
        let read = flight
            .complete_read_only_readback(&cohort, |data| {
                for b in data {
                    if !words(b)?.iter().all(|v| v.is_finite()) {
                        return Err(Error::State("nonfinite committed resident snapshot".into()));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        let mut cursor = 0;
        let mut output = Vec::new();
        for d in &self.domains {
            let mut flat = Vec::new();
            for l in &d.leases {
                let values = words(&read[cursor])?;
                if matches!(&d.buffers, StateBuffers::Kv { .. }) {
                    let active = l.bytes() / 4 / self.capacity * self.binding.frontier() as usize;
                    if values[active..].iter().any(|x| x.to_bits() != 0) {
                        return Err(Error::State("snapshot noncanonical KV tail".into()));
                    }
                    flat.extend_from_slice(&values[..active]);
                } else {
                    flat.extend(values);
                }
                cursor += 1;
            }
            output.push((d.slot, flat));
        }
        Ok(output)
    }
}

impl ResidentSession {
    pub fn binding(&self) -> RootBinding {
        self.authority.binding()
    }
    pub fn snapshot(&self) -> ResidentSnapshot {
        ResidentSnapshot {
            binding: self.binding(),
            service_generation: self.model.service.generation(),
            capacity: self.model.capacity,
            domains: self.domains.clone(),
        }
    }
    pub async fn prepare(&self, tokens: &[u32], diagnostic: bool) -> Result<PreparedInference> {
        self.prepare_mode(tokens, diagnostic, Readout::Logits).await
    }
    async fn prepare_mode(
        &self,
        tokens: &[u32],
        diagnostic: bool,
        readout: Readout,
    ) -> Result<PreparedInference> {
        let count =
            u64::try_from(tokens.len()).map_err(|_| Error::Limit("resident token count".into()))?;
        let attempt = self.authority.begin_advance(count)?;
        let base = self.binding();
        if base
            .frontier()
            .checked_add(count)
            .is_none_or(|n| n > self.model.capacity as u64)
        {
            return Err(Error::Limit("resident context capacity exhausted".into()));
        }
        if tokens
            .iter()
            .any(|token| u64::from(*token) >= self.model.vocab)
        {
            return Err(Error::Malformed(
                "resident chunk contains invalid source token".into(),
            ));
        }
        // Reserve the complete private ping-pong peak before the first launch.
        let mut pool = vec![self.model.allocate_state(false).await?];
        if tokens.len() > 1 {
            pool.push(self.model.allocate_state(false).await?);
        }
        let mut domains = self.domains.clone();
        let mut last = None;
        let mut dispatches = 0;
        let mut chunk_bytes = 0;
        let mut state_only_steps = 0;
        let mut logit_steps = 0;
        for (i, token) in tokens.iter().enumerate() {
            let (next, ready, readback) = self
                .model
                .execute(
                    base,
                    base.frontier() as usize + i,
                    &domains,
                    pool[i % pool.len()].clone(),
                    StepInput {
                        token: *token,
                        device: None,
                        readout: if i + 1 != tokens.len() {
                            Readout::None
                        } else {
                            readout
                        },
                        diagnostic: diagnostic && i + 1 == tokens.len(),
                    },
                )
                .await?;
            dispatches += readback.dispatches;
            chunk_bytes += readback.chunk_bytes;
            state_only_steps += readback.state_only_steps;
            logit_steps += readback.logit_steps;
            domains = next;
            last = Some((ready, readback));
        }
        let (ready, mut readback) =
            last.ok_or_else(|| Error::State("empty resident transaction".into()))?;
        readback.dispatches = dispatches;
        readback.chunk_bytes = chunk_bytes;
        readback.state_only_steps = state_only_steps;
        readback.logit_steps = logit_steps;
        Ok(PreparedInference {
            attempt,
            domains,
            ready,
            readback,
        })
    }
    pub fn commit(&mut self, mut prepared: PreparedInference) -> Result<InferenceReadback> {
        self.model.service.ensure_usable()?;
        if prepared.ready.expected_root() != prepared.attempt.base().root()
            || !Arc::ptr_eq(prepared.ready.service(), &self.model.service)
            || prepared.domains.len() != self.model.states.len()
            || prepared.ready.event_ids().len() != 1
        {
            return Err(Error::State(
                "foreign/incomplete resident Ready proposal".into(),
            ));
        }
        self.model.validate_domains(&prepared.domains, true)?;
        let mut outputs = prepared
            .domains
            .iter()
            .flat_map(|d| d.leases.iter())
            .collect::<Vec<_>>();
        outputs.extend(prepared.readback.device_token.iter().map(|t| &t.lease));
        if outputs.len() != prepared.ready.outputs().len()
            || outputs
                .iter()
                .zip(prepared.ready.outputs())
                .any(|(a, b)| a.id() != b.id())
        {
            return Err(Error::State(
                "resident promoted domain cohort mismatch".into(),
            ));
        }
        let mut displaced = prepared.domains;
        let published = self.model.service.publish_if_usable(|| {
            let transition = self.authority.prepare_publish(prepared.attempt)?;
            std::mem::swap(&mut self.domains, &mut displaced);
            Ok(transition.commit())
        })?;
        drop(displaced);
        if let Some(token) = prepared.readback.device_token.as_mut() {
            token.producer = Some(published);
        }
        Ok(prepared.readback)
    }
    pub async fn forward(&mut self, tokens: &[u32], diagnostic: bool) -> Result<InferenceReadback> {
        let prepared = self.prepare(tokens, diagnostic).await?;
        self.commit(prepared)
    }
    pub async fn forward_greedy(&mut self, tokens: &[u32]) -> Result<InferenceReadback> {
        let prepared = self.prepare_mode(tokens, false, Readout::Greedy).await?;
        self.commit(prepared)
    }
    /// Publish a checked prompt prefix without running an output-only tail.
    pub async fn prefill_without_logits(
        &mut self,
        tokens: &[u32],
        diagnostic: bool,
    ) -> Result<InferenceReadback> {
        let prepared = self.prepare_mode(tokens, diagnostic, Readout::None).await?;
        self.commit(prepared)
    }
    pub async fn forward_device_token(
        &mut self,
        token: &ResidentToken,
    ) -> Result<InferenceReadback> {
        let base = self.binding();
        if token.producer != Some(base) {
            return Err(Error::State("device token belongs to another root".into()));
        }
        if base.frontier() >= self.model.capacity as u64 {
            return Err(Error::Limit("resident context capacity exhausted".into()));
        }
        let attempt = self.authority.begin_advance(1)?;
        let candidate = self.model.allocate_state(false).await?;
        let (domains, ready, readback) = self
            .model
            .execute(
                base,
                base.frontier() as usize,
                &self.domains,
                candidate,
                StepInput {
                    token: 0,
                    device: Some(token),
                    readout: Readout::Greedy,
                    diagnostic: false,
                },
            )
            .await?;
        self.commit(PreparedInference {
            attempt,
            domains,
            ready,
            readback,
        })
    }
    pub fn restore(&mut self, snapshot: &ResidentSnapshot) -> Result<()> {
        if snapshot.service_generation != self.model.service.generation()
            || snapshot.domains.len() != self.model.states.len()
        {
            return Err(Error::State(
                "foreign resident snapshot service/domains".into(),
            ));
        }
        self.model.service.ensure_usable()?;
        self.model.validate_domains(&snapshot.domains, true)?;
        let attempt = self.authority.begin_restore(snapshot.binding)?;
        let mut displaced = snapshot.domains.clone();
        self.model.service.publish_if_usable(|| {
            let transition = self.authority.prepare_publish(attempt)?;
            std::mem::swap(&mut self.domains, &mut displaced);
            transition.commit();
            Ok(())
        })?;
        drop(displaced);
        Ok(())
    }
    pub async fn reset(&mut self) -> Result<()> {
        let attempt = self.authority.begin_reset()?;
        let mut displaced = self.model.allocate_state(true).await?;
        self.model.service.publish_if_usable(|| {
            let transition = self.authority.prepare_publish(attempt)?;
            std::mem::swap(&mut self.domains, &mut displaced);
            transition.commit();
            Ok(())
        })?;
        drop(displaced);
        Ok(())
    }
}

#[cfg(feature = "qualification")]
#[path = "resident_inference/qualification.rs"]
mod qualification;
#[cfg(feature = "qualification")]
use qualification::{compare_domains, exact};
#[cfg(feature = "qualification")]
pub(crate) use qualification::{qualify, qualify_context};
#[cfg(all(feature = "qualification", feature = "hardware-abort-probe"))]
pub(crate) use qualification::{qualify_abort, qualify_context_abort};

#[cfg(feature = "qualification")]
async fn load_qualified(
    device: Device,
    model: Arc<LoadedQwenModel>,
    capacity: usize,
    provider_cap: usize,
    device_cap: u64,
) -> Result<Arc<ResidentModel>> {
    let plan = model.forward_plan(capacity, true, true)?;
    ResidentModel::load(device, model, plan, provider_cap, device_cap).await
}

/// Cold token-consuming profile geometry. No device/client/admission is touched.
fn validate_profile_plan(plan: &ForwardPlan) -> Result<(usize, u64)> {
    let id = plan
        .logits()
        .ok_or_else(|| Error::Unsupported("resident profile requires logits".into()))?;
    let vocab = profile_vocabulary(plan.values().get(id.0 as usize).copied())?;
    let mut token_embed = false;
    for node in plan.nodes() {
        if let PlanOperation::Embed { tensor, token } = &node.operation
            && token.0 == ParameterSlot::Token
        {
            token_embed = true;
            if tensor.rows != vocab {
                return Err(Error::State(
                    "resident token embedding and logits vocabulary differ".into(),
                ));
            }
        }
    }
    if !token_embed {
        return Err(Error::Unsupported(
            "resident token profile requires a token embedding".into(),
        ));
    }
    Ok((plan.prefix_capacity(), vocab as u64))
}

fn profile_vocabulary(shape: Option<ValueShape>) -> Result<usize> {
    match shape {
        Some(ValueShape::Vector(vocab)) if vocab > 0 && u32::try_from(vocab).is_ok() => Ok(vocab),
        _ => Err(Error::Unsupported(
            "resident logits require a nonempty u32 token vector".into(),
        )),
    }
}

fn preflight_profile(
    source: &dyn TensorSource,
    plan: &ForwardPlan,
) -> Result<(usize, u64, ResidentPreflight)> {
    let (capacity, vocab) = validate_profile_plan(plan)?;
    let budget = ResidentProgram::preflight(source, plan)?;
    Ok((capacity, vocab, budget))
}

#[cfg(test)]
#[path = "resident_inference/cold_tests.rs"]
mod cold_tests;
