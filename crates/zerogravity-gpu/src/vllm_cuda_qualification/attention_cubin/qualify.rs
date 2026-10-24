//! Original qualification lifecycle, genuine computed QKV→QK→KV→FA only.
use super::super::{
    PROVIDER_CAP, finite_bf16,
    projection_cubin::{CubinQualificationInputs, native_error, not_cancelled},
};
use super::faults::{self, CubinAttentionFault, CubinAttentionFaultOutcome};
use super::{descriptor::*, manifests::*, resident::AttentionPrograms, weights::AttentionWeights};
use crate::resident_leases::{ResidentLease, ResidentReadback, ResidentService};
use cubecl::Device;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::{
    Error, Result,
    device_events::{BufferRole, DeviceAccess},
    resource_arbiter::{AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner},
};

const HOST_BYTES: usize = 64 * 1024 * 1024;
const NATIVE_BYTES: usize = 64 * 1024 * 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CubinAttentionOutput {
    PackedQkv,
    Q,
    K,
    Gate,
    AttentionOutput,
    Projected,
    KvActive,
    SigmoidGate,
    Gated,
}
impl CubinAttentionOutput {
    const ALL: [Self; 7] = [
        Self::PackedQkv,
        Self::Q,
        Self::K,
        Self::Gate,
        Self::AttentionOutput,
        Self::Projected,
        Self::KvActive,
    ];
    const GATED_ALL: [Self; 9] = [
        Self::PackedQkv,
        Self::Q,
        Self::K,
        Self::Gate,
        Self::AttentionOutput,
        Self::SigmoidGate,
        Self::Gated,
        Self::Projected,
        Self::KvActive,
    ];
    fn for_scope(scope: AttentionScope) -> &'static [Self] {
        match scope {
            AttentionScope::CapturedOutInput => &Self::ALL,
            AttentionScope::ComputedGate => &Self::GATED_ALL,
        }
    }
    fn role(self) -> Option<AttentionRole> {
        Some(match self {
            Self::PackedQkv => AttentionRole::PackedQkv,
            Self::Q => AttentionRole::Q,
            Self::K => AttentionRole::K,
            Self::Gate => AttentionRole::Gate,
            Self::AttentionOutput => AttentionRole::AttentionOutput,
            Self::Projected => AttentionRole::Projected,
            Self::SigmoidGate => AttentionRole::SigmoidGate,
            Self::Gated => AttentionRole::OutInput,
            Self::KvActive => return None,
        })
    }
    pub fn bytes(self, case: AttentionCase) -> usize {
        let n = case.rows();
        match self {
            Self::PackedQkv => n * 10240,
            Self::Q | Self::Gate | Self::AttentionOutput | Self::SigmoidGate | Self::Gated => {
                n * 4096
            }
            Self::K => n * 1024,
            Self::Projected => n * 2048,
            Self::KvActive => (case.base() + n) * 2048,
        }
    }
}
pub struct CubinAttentionExpected {
    pub output: CubinAttentionOutput,
    pub bytes: Vec<u8>,
}
/// Normalized/gated inputs and cosine prefix are explicit authentic leaves.
/// No captured Q/K/KV/FA output is ever admitted as a device operand.
pub struct CubinAttentionFixture {
    pub case: AttentionCase,
    pub normalized: Vec<u8>,
    pub gated: Vec<u8>,
    pub positions: Vec<i64>,
    pub expected: Vec<CubinAttentionExpected>,
}
/// Connected FA→sigmoid→BF16→multiply→BF16→Out fixture. Expected records are
/// callback-only oracles; no FA/sigmoid/gated mathematical operand is accepted.
pub struct CubinAttentionGatedFixture {
    pub case: AttentionCase,
    pub normalized: Vec<u8>,
    pub positions: Vec<i64>,
    pub expected: Vec<CubinAttentionExpected>,
}
impl CubinAttentionGatedFixture {
    pub(super) fn into_common(self) -> CubinAttentionFixture {
        CubinAttentionFixture {
            case: self.case,
            normalized: self.normalized,
            gated: Vec::new(),
            positions: self.positions,
            expected: self.expected,
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct CubinAttentionComparison {
    pub calls: usize,
    pub producer_flights: usize,
    pub native_launches: usize,
    pub device_copies: usize,
    pub bf16_words: usize,
    pub source_unchanged: bool,
}
pub(super) struct Owner {
    pub(super) source: Arc<VerifiedTensorSource>,
    pub(super) fixtures: Vec<CubinAttentionFixture>,
    cos_sin: Vec<u8>,
    _host: [WindowWorkspace; 4],
}
impl Owner {
    pub(super) fn host_bytes(&self) -> usize {
        self._host.iter().map(WindowWorkspace::bytes).sum()
    }
}
pub(super) struct Job {
    pub(super) bindings: Arc<AttentionBindings>,
    pub(super) accesses: Vec<(ResidentLease, DeviceAccess)>,
    outputs: Vec<ResidentLease>,
    pub(super) reads: Vec<ResidentReadback>,
    checks: Vec<Check>,
    _owner: Arc<Owner>,
}
enum Check {
    Output(CubinAttentionOutput, usize, usize),
    Unchanged(Vec<u8>),
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    Compare,
    Fault(CubinAttentionFault),
    ComputedGate,
    GateFault(CubinAttentionFault),
}
impl Mode {
    fn scope(self) -> AttentionScope {
        match self {
            Self::Compare | Self::Fault(_) => AttentionScope::CapturedOutInput,
            Self::ComputedGate | Self::GateFault(_) => AttentionScope::ComputedGate,
        }
    }
    fn fault(self) -> Option<CubinAttentionFault> {
        match self {
            Self::Fault(f) | Self::GateFault(f) => Some(f),
            _ => None,
        }
    }
}

pub(super) enum Outcome {
    Compared(CubinAttentionComparison),
    Fault(CubinAttentionFaultOutcome),
}

pub async fn qualify_attention_cubins(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinAttentionFixture>,
    cos_sin: Vec<u8>,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<CubinAttentionComparison> {
    match inner(
        device,
        source,
        fixtures,
        cos_sin,
        inputs,
        Mode::Compare,
        cancelled,
    )
    .await?
    {
        Outcome::Compared(r) => Ok(r),
        Outcome::Fault(_) => Err(Error::State("attention comparison returned fault".into())),
    }
}
pub async fn qualify_attention_gated_cubins(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinAttentionGatedFixture>,
    cos_sin: Vec<u8>,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<CubinAttentionComparison> {
    not_cancelled(cancelled)?;
    if fixtures.capacity() > 8 {
        return Err(Error::Limit(
            "attention gated fixture metadata capacity".into(),
        ));
    }
    let fixtures = fixtures
        .into_iter()
        .map(CubinAttentionGatedFixture::into_common)
        .collect();
    match inner(
        device,
        source,
        fixtures,
        cos_sin,
        inputs,
        Mode::ComputedGate,
        cancelled,
    )
    .await?
    {
        Outcome::Compared(result) => Ok(result),
        Outcome::Fault(_) => Err(Error::State("gated comparison returned fault".into())),
    }
}
pub(super) fn validate_fixtures(fixtures: &[CubinAttentionFixture], cos_sin: &[u8]) -> Result<()> {
    validate_fixtures_scoped(fixtures, cos_sin, AttentionScope::CapturedOutInput)
}
pub(super) fn validate_fixtures_scoped(
    fixtures: &[CubinAttentionFixture],
    cos_sin: &[u8],
    scope: AttentionScope,
) -> Result<()> {
    if fixtures.len() != 5 || cos_sin.len() != 8576 || !finite_bf16(cos_sin) {
        return Err(Error::Malformed(
            "attention exact five cases/cosine prefix".into(),
        ));
    }
    let mut capacity = cos_sin.len();
    for (f, &case) in fixtures.iter().zip(&AttentionCase::ALL) {
        if f.case != case
            || f.normalized.len() != case.rows() * 2048
            || if scope == AttentionScope::CapturedOutInput {
                f.gated.len() != case.rows() * 4096
            } else {
                !f.gated.is_empty()
            }
            || !finite_bf16(&f.normalized)
            || !finite_bf16(&f.gated)
            || f.positions.len() != 3 * case.rows()
            || f.expected.len() != CubinAttentionOutput::for_scope(scope).len()
        {
            return Err(Error::Malformed(
                "attention complete original leaf operands".into(),
            ));
        }
        for axis in 0..3 {
            for i in 0..case.rows() {
                if f.positions[axis * case.rows() + i] != (case.base() + i) as i64 {
                    return Err(Error::Malformed(
                        "attention exact original teacher positions".into(),
                    ));
                }
            }
        }
        let mut outputs = BTreeSet::new();
        for e in &f.expected {
            if !outputs.insert(e.output)
                || e.bytes.len() != e.output.bytes(case)
                || !finite_bf16(&e.bytes)
            {
                return Err(Error::Malformed(
                    "attention complete independent expected records".into(),
                ));
            }
            capacity = capacity
                .checked_add(e.bytes.capacity())
                .ok_or_else(|| Error::Limit("attention fixture capacity overflow".into()))?;
        }
        if outputs
            != CubinAttentionOutput::for_scope(scope)
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
        {
            return Err(Error::Malformed(
                "attention exact expected output roles".into(),
            ));
        }
        capacity = capacity
            .checked_add(f.normalized.capacity())
            .and_then(|n| n.checked_add(f.gated.capacity()))
            .and_then(|n| n.checked_add(f.positions.capacity() * 8))
            .ok_or_else(|| Error::Limit("attention fixture capacity overflow".into()))?;
    }
    if capacity > HOST_BYTES {
        return Err(Error::Limit(
            "attention bounded original fixture host capacity".into(),
        ));
    }
    Ok(())
}
fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}
pub(super) fn control_bytes(d: &AttentionDescriptor, role: AttentionRole) -> Option<Vec<u8>> {
    let c = d.controls();
    Some(match role {
        AttentionRole::PositionsI64Pitch65 => i64_bytes(&c.positions),
        AttentionRole::SlotsI64 => i64_bytes(&c.slots[..c.slot_count]),
        AttentionRole::CuQI32 => i32_bytes(&c.cu_q),
        AttentionRole::SeqUsedI32 => i32_bytes(&[c.seq_used]),
        AttentionRole::BlockTableI32 => i32_bytes(&c.block_table),
        AttentionRole::DummyCuKI32 => i32_bytes(&c.dummy_cu_k),
        AttentionRole::KScale | AttentionRole::VScale => 1f32.to_le_bytes().to_vec(),
        _ => return None,
    })
}
fn check_job(
    job: &Job,
    owner: &Owner,
    index: usize,
    raw: &[cubecl::bytes::Bytes],
) -> Result<usize> {
    if raw.len() != job.checks.len() {
        return Err(Error::State("attention exact readback descriptors".into()));
    }
    let mut words = 0;
    for (actual, check) in raw.iter().zip(&job.checks) {
        let expected = match check {
            Check::Unchanged(bytes) => bytes.as_slice(),
            Check::Output(output, offset, n) => {
                let e = owner.fixtures[index]
                    .expected
                    .iter()
                    .find(|e| e.output == *output)
                    .ok_or_else(|| Error::State("attention expected output absent".into()))?;
                words += *n / 2;
                &e.bytes[*offset..*offset + *n]
            }
        };
        if &actual[..] != expected {
            let at = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            let detail = match check {
                Check::Output(output, offset, _) => {
                    let word = at / 2;
                    let start = word * 2;
                    let a = u16::from_le_bytes(actual[start..start + 2].try_into().unwrap());
                    let e = u16::from_le_bytes(expected[start..start + 2].try_into().unwrap());
                    format!(
                        "{output:?} word{} actual0x{a:04x} expected0x{e:04x}",
                        (offset + start) / 2
                    )
                }
                Check::Unchanged(_) => format!(
                    "unchanged byte{at} actual{} expected{}",
                    actual[at], expected[at]
                ),
            };
            return Err(Error::State(format!(
                "original attention leaf{index} {detail}"
            )));
        }
    }
    Ok(words)
}

pub(super) async fn inner(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinAttentionFixture>,
    cos_sin: Vec<u8>,
    inputs: CubinQualificationInputs,
    mode: Mode,
    cancelled: Option<&AtomicBool>,
) -> Result<Outcome> {
    // Before malformed fixtures, workspaces, CUDA service or driver mapping.
    not_cancelled(cancelled)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA attention leaf required".into(),
        ));
    }
    let scope = mode.scope();
    if scope == AttentionScope::CapturedOutInput {
        validate_fixtures(&fixtures, &cos_sin)?
    } else {
        validate_fixtures_scoped(&fixtures, &cos_sin, scope)?
    };
    if fixtures.capacity() > 8 || cos_sin.capacity() > HOST_BYTES || inputs.images.capacity() > 16 {
        return Err(Error::Limit("attention metadata/prefix capacity".into()));
    }
    let source = Arc::new(source);
    let host = [
        source.reserve_workspace(32 * 1024 * 1024)?,
        source.reserve_workspace(32 * 1024 * 1024)?,
        source.reserve_workspace(32 * 1024 * 1024)?,
        source.reserve_workspace(32 * 1024 * 1024)?,
    ];
    let owner = Arc::new(Owner {
        source: source.clone(),
        fixtures,
        cos_sin,
        _host: host,
    });
    not_cancelled(cancelled)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("vllm08-cuda-cubin", "original-attention-leaf")?,
        PROVIDER_CAP,
        inputs.device_cap,
    )?;
    let key = service.physical_key();
    if mode.fault() == Some(CubinAttentionFault::Pressure) {
        let witness = faults::pressure(&source, &owner, &service)?;
        drop((inputs, owner, source, service));
        return witness.retired().map(Outcome::Fault);
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: NATIVE_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    not_cancelled(cancelled)?;
    let mut init = service.begin_initialization(100)?;
    init.retain_completion_owner(owner.clone())?;
    // Two independent genuine sequences. Active pages remain4 and1 exactly;
    // both original cache allocations retain computed state across their calls.
    let zeros = vec![0u8; KV_BYTES];
    let caches = [
        init.upload_bytes(&zeros, BufferRole::Scratch)?,
        init.upload_bytes(&zeros, BufferRole::Scratch)?,
    ];
    drop(zeros);
    let workspace = init.allocate_empty(
        super::super::cubin_manifests::WORKSPACE_BYTES,
        BufferRole::Scratch,
    )?;
    let cos = init.upload_bytes(&owner.cos_sin, BufferRole::ImmutableWeights)?;
    let mut maps = Vec::new();
    for (index, fixture) in owner.fixtures.iter().enumerate() {
        not_cancelled(cancelled)?;
        let d = AttentionDescriptor::with_scope(source.source_identity(), fixture.case, scope)?;
        let mut leases = BTreeMap::new();
        for (role, (bytes, _)) in d.required_roles()? {
            if matches!(
                role,
                AttentionRole::QkvW
                    | AttentionRole::OutW
                    | AttentionRole::QGamma
                    | AttentionRole::KGamma
            ) {
                continue;
            }
            let lease = match role {
                AttentionRole::OriginalKvState => caches[if index < 2 { 0 } else { 1 }].clone(),
                AttentionRole::Workspace => workspace.clone(),
                AttentionRole::CosSinSourcePrefix67 => cos.clone(),
                AttentionRole::QkvInput => {
                    init.upload_bytes(&fixture.normalized, BufferRole::Activation)?
                }
                AttentionRole::OutInput if scope == AttentionScope::CapturedOutInput => {
                    init.upload_bytes(&fixture.gated, BufferRole::Activation)?
                }
                _ => {
                    if let Some(raw) = control_bytes(&d, role) {
                        if raw.len() != bytes {
                            return Err(Error::State("attention exact control backing".into()));
                        }
                        init.upload_bytes(&raw, BufferRole::Activation)?
                    } else {
                        init.allocate_empty(bytes, BufferRole::Output)?
                    }
                }
            };
            leases.insert(role, lease);
        }
        maps.push((d, leases));
    }
    // Last managed allocations. All actual source W/gamma windows/packed H2D
    // owners attach to this SAME real initialization before native load.
    let weights = AttentionWeights::prepare(source.clone(), &service, &mut init, cancelled)?;
    let mut bindings = Vec::new();
    let mut jobs = Vec::new();
    for (d, mut leases) in maps {
        for (&role, lease) in weights.leases() {
            leases.insert(role, lease.clone());
        }
        // SAFETY: all exact controls and source/leaf operands were uploaded by
        // this original init; no captured intermediate or cache was admitted.
        let bound =
            unsafe { AttentionBindings::new(d.clone(), service.clone(), owner.clone(), leases) }?;
        let accesses = bound.accesses()?;
        let outputs = accesses
            .iter()
            .filter(|(_, a)| *a == DeviceAccess::Write)
            .map(|(l, _)| l.clone())
            .collect();
        let mut reads = Vec::new();
        let mut checks = Vec::new();
        for &output in CubinAttentionOutput::for_scope(scope) {
            if let Some(role) = output.role() {
                let l = bound.lease(role)?;
                reads.push(l.readback(0, l.bytes())?);
                checks.push(Check::Output(output, 0, l.bytes()));
            } else {
                let l = bound.lease(AttentionRole::OriginalKvState)?;
                for p in 0..d.case().base() + d.case().rows() {
                    reads.push(l.readback((d.case().block() * 544 + p) * 2048, 2048)?);
                    checks.push(Check::Output(output, p * 2048, 2048));
                }
            }
        }
        for (role, bytes) in [
            (
                AttentionRole::QkvInput,
                owner.fixtures[bindings.len()].normalized.clone(),
            ),
            (AttentionRole::CosSinSourcePrefix67, owner.cos_sin.clone()),
        ] {
            let l = bound.lease(role)?;
            reads.push(l.readback(0, l.bytes())?);
            checks.push(Check::Unchanged(bytes));
        }
        if scope == AttentionScope::CapturedOutInput {
            let l = bound.lease(AttentionRole::OutInput)?;
            reads.push(l.readback(0, l.bytes())?);
            checks.push(Check::Unchanged(
                owner.fixtures[bindings.len()].gated.clone(),
            ));
        }
        for &role in &bound
            .descriptor()
            .required_roles()?
            .keys()
            .copied()
            .collect::<Vec<_>>()
        {
            if let Some(raw) = control_bytes(&d, role) {
                let l = bound.lease(role)?;
                reads.push(l.readback(0, l.bytes())?);
                checks.push(Check::Unchanged(raw));
            }
        }
        let job = Arc::new(Job {
            bindings: bound.clone(),
            accesses,
            outputs,
            reads,
            checks,
            _owner: owner.clone(),
        });
        init.retain_completion_owner(job.clone())?;
        jobs.push(job);
        bindings.push(bound);
    }
    let prepare_images = if scope == AttentionScope::ComputedGate {
        super::prepare_gate_images
    } else {
        super::prepare_images
    };
    let images = prepare_images(
        &source,
        &service,
        owner.clone(),
        inputs,
        &mut init,
        cancelled,
    )?;
    let programs = AttentionPrograms::prepare(images.clone(), bindings, &mut init, cancelled)?;
    drop(init.complete_sync().await?);
    if scope == AttentionScope::ComputedGate {
        for case in AttentionCase::ALL {
            let bound = programs.bindings(case)?;
            if bound.lease(AttentionRole::SigmoidGate)?.is_initialized()
                || bound.lease(AttentionRole::OutInput)?.is_initialized()
            {
                return Err(Error::State(
                    "connected gate outputs were preinitialized instead of produced".into(),
                ));
            }
        }
    }
    slot.finish()?;
    let mut compared = CubinAttentionComparison::default();
    let mut index = 0;
    while let Some(job) = jobs.get(index).cloned() {
        not_cancelled(cancelled)?;
        let witness = if mode.fault().is_some() {
            Some(faults::UnknownWitness::new(
                &source, &owner, &weights, &jobs, &programs, &images, &service,
            )?)
        } else {
            None
        };
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: NATIVE_BYTES as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(200 + index as u64, &job.accesses)?;
        flight.retain_completion_owner(owner.clone())?;
        flight.retain_completion_owner(weights.clone())?;
        flight.retain_completion_owner(programs.completion_owner())?;
        flight.retain_completion_owner(job.clone())?;
        let case = job.bindings.descriptor().case();
        let mut launches = 0;
        let mut copies = 0;
        for &stage in scope.stages() {
            if stage == AttentionStage::Multiply
                && mode == Mode::GateFault(CubinAttentionFault::CancelBeforeMultiply)
            {
                let flag = AtomicBool::new(true);
                let result = unsafe { programs.enqueue(case, stage, Some(&flag)) };
                if !matches!(result, Err(Error::State(_))) {
                    return Err(Error::State(
                        "gate cancellation before multiply was not reached".into(),
                    ));
                }
                let witness = witness
                    .ok_or_else(|| Error::State("gate cancellation witness absent".into()))?;
                drop(flight);
                drop(slot);
                drop((
                    job, jobs, programs, images, weights, workspace, cos, caches, owner, source,
                    service,
                ));
                return witness
                    .inspect(CubinAttentionFault::CancelBeforeMultiply, launches, copies)
                    .map(Outcome::Fault);
            }
            let submitted = if stage == AttentionStage::Fa
                && mode == Mode::Fault(CubinAttentionFault::CancelBeforeCopy)
            {
                let flag = AtomicBool::new(false);
                let result = unsafe { programs.enqueue_cancel_before_copy(case, &flag) };
                if !flag.load(std::sync::atomic::Ordering::Acquire)
                    || !matches!(result, Err(Error::State(_)))
                {
                    return Err(Error::State(
                        "attention actual before-copy cancellation not reached".into(),
                    ));
                }
                launches += 1; // Real original unsplit FA kernel enqueued first.
                let witness = witness
                    .ok_or_else(|| Error::State("attention cancellation witness absent".into()))?;
                drop(flight);
                drop(slot);
                drop((
                    job, jobs, programs, images, weights, workspace, cos, caches, owner, source,
                    service,
                ));
                return witness
                    .inspect(CubinAttentionFault::CancelBeforeCopy, launches, copies)
                    .map(Outcome::Fault);
            } else {
                unsafe { programs.enqueue(case, stage, cancelled) }?
            };
            launches += submitted.native_launches;
            copies += submitted.device_copies;
            if (stage == AttentionStage::Fa
                && mode == Mode::Fault(CubinAttentionFault::UnknownAfterCopy))
                || (stage == AttentionStage::Multiply
                    && mode == Mode::GateFault(CubinAttentionFault::UnknownAfterMultiply))
            {
                let witness = witness
                    .ok_or_else(|| Error::State("attention unknown witness absent".into()))?;
                drop(flight);
                drop(slot);
                drop((
                    job, jobs, programs, images, weights, workspace, cos, caches, owner, source,
                    service,
                ));
                return witness
                    .inspect(mode.fault().unwrap(), launches, copies)
                    .map(Outcome::Fault);
            }
        }
        let mut words = 0;
        let (ready, raw) = flight
            .complete_with_readback(&job.outputs, &job.reads, |raw| {
                words = check_job(&job, &owner, index, raw)?;
                Ok(())
            })
            .await?;
        slot.finish()?;
        compared.calls += 1;
        compared.producer_flights += 1;
        compared.native_launches += launches;
        compared.device_copies += copies;
        compared.bf16_words += words;
        drop((ready, raw, job));
        index += 1;
    }
    let immutable = weights.immutable_weights();
    let accesses = immutable
        .iter()
        .map(|(l, _)| (l.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    let reads = immutable
        .iter()
        .map(|(l, _)| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: NATIVE_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut flight = service.submit(999, &accesses)?;
    flight.retain_completion_owner(programs.clone())?;
    flight.retain_completion_owner(weights.clone())?;
    flight.retain_completion_owner(owner.clone())?;
    not_cancelled(cancelled)?;
    // SAFETY: all actual producer flights physically completed. Original armed
    // initialized-W read-only flight retains every source/native owner through
    // actual unload. Any error takes original unknown quarantine, never Ready.
    unsafe { programs.retire_after_physical_completion() }.map_err(native_error)?;
    let raw = flight
        .complete_read_only_readback(&reads, |raw| {
            if raw.len() != immutable.len()
                || raw
                    .iter()
                    .zip(&immutable)
                    .any(|(actual, (_, expected))| &actual[..] != *expected)
            {
                return Err(Error::State("attention original W/gamma changed".into()));
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    drop((raw, reads, accesses, immutable));
    drop((
        jobs, programs, images, weights, workspace, cos, caches, owner, source,
    ));
    service.ensure_usable()?;
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "attention original buffers/events/native charge not retired".into(),
        ));
    }
    drop(service);
    let retired = ResourceArbiter::process_shared().snapshot(&key)?;
    if retired.reserved != 0 || retired.fenced {
        return Err(Error::State(
            "attention original physical envelope not retired".into(),
        ));
    }
    compared.source_unchanged = true;
    Ok(Outcome::Compared(compared))
}
