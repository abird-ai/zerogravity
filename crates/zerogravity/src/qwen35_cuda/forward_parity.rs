//! Two-process current-source reference -> forward parity harness.
//!
//! Historical reference and forward candidate run in separate processes to
//! close process-local native/context ownership between phases. Phase 1 writes
//! bounded checksummed host evidence after reference retirement/process-local
//! ownership closes. Phase 2 independently authenticates the source and executes
//! the fixed qualification trajectory; evidence only compares completed results.

use super::{Model as ReferenceModel, forward_candidate};
mod evidence;
use crate::gpu::cuda::native::{TextCallReceipt, TextReadback};
use crate::{
    Error, Result,
    accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, SourceCallMode, StateSlot},
    gpu::{ResidentPhysicalType, StateLayout},
    qwen35::{plan::QwenModelVariant, source::binding::OriginalWindowModelSource},
    resource_arbiter::ResourceOwner,
};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

const VOCABULARY: usize = 248_320;
const FRONTIERS: [usize; 4] = [47, 48, 49, 50];
const CURRENT_PARITY_CAPACITY: usize = 512;

pub fn parse_evidence_sha256(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Error::Malformed(
            "forward parity expected evidence SHA256 must be 64 hex digits".into(),
        ));
    }
    let mut digest = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let nibble = |byte: u8| -> Result<u8> {
            match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                b'A'..=b'F' => Ok(byte - b'A' + 10),
                _ => Err(Error::Malformed(
                    "forward parity expected evidence SHA256 contains non-hex".into(),
                )),
            }
        };
        digest[index] = nibble(pair[0])? * 16 + nibble(pair[1])?;
    }
    Ok(digest)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ParitySourceIdentity {
    pub(super) family: String,
    pub(super) repository: String,
    pub(super) revision: String,
    pub(super) hidden_width: usize,
    pub(super) capacity: usize,
}

#[derive(Clone, Debug)]
pub struct ReferenceParityStep {
    pub frontier: usize,
    pub selected_id: u32,
    pub max_bits: u32,
    pub logits: Vec<f32>,
    pub domains: Vec<(StateSlot, Vec<f32>)>,
}

#[derive(Clone, Debug)]
pub struct Prefill47Decode48To50Reference {
    pub capacity: usize,
    pub prompt: Arc<[u32]>,
    pub steps: Vec<ReferenceParityStep>,
}

#[derive(Clone, Debug, Default)]
pub struct ParityStepReport {
    pub frontier: usize,
    pub forward_selected_id: u32,
    pub reference_selected_id: u32,
    pub selected_matches: bool,
    pub max_bits_match: bool,
    pub logit_mismatches: usize,
    pub max_abs_logit_error: f32,
    pub domain_mismatch_values: usize,
    pub domain_mismatch_slots: Vec<StateSlot>,
}

#[derive(Clone, Debug, Default)]
pub struct Prefill47Decode48To50ParityReport {
    pub steps: Vec<ParityStepReport>,
}

#[derive(Clone, Debug)]
pub struct ReferenceEvidenceReceipt {
    pub sha256: [u8; 32],
    pub close: crate::gpu::cuda::native::TextCloseReceipt,
}

#[derive(Clone, Debug)]
pub struct ForwardEvidenceReceipt {
    pub sha256: [u8; 32],
    pub parity: Prefill47Decode48To50ParityReport,
}
impl ForwardEvidenceReceipt {
    pub fn require_exact(&self) -> Result<()> {
        self.parity.require_exact_source_parity()
    }
}
impl Prefill47Decode48To50ParityReport {
    pub fn require_exact_source_parity(&self) -> Result<()> {
        if self.steps.len() != FRONTIERS.len() || self.steps.iter().any(|step| !step_is_exact(step))
        {
            return Err(Error::State(format!(
                "CUDA forward prefill47 + decode48–50 exact source parity failed: steps={} first_failure={:?}",
                self.steps.len(),
                self.steps.iter().find(|step| !step_is_exact(step)),
            )));
        }
        Ok(())
    }
}

pub async fn capture_current_source_prefill47_decode48_to50_evidence(
    device: crate::gpu::Device,
    hf_directory: &Path,
    artifact_directory: &Path,
    evidence_path: &Path,
    cancelled: Option<&AtomicBool>,
) -> Result<ReferenceEvidenceReceipt> {
    const MIB: usize = 1024 * 1024;
    require_not_cancelled(cancelled)?;
    // The historical reference retains all finite scopes, like the model-backed
    // fixtures. Forward execution keeps its separate 4 GiB host profile.
    let host = super::host_admission_mib(
        super::HOST_MIB_MAX,
        ResourceOwner::new(
            "qwen35-cuda-parity-reference",
            "prefill47-decode48-50-evidence",
        )?,
    )?;
    let model = super::qualification::load(
        device,
        hf_directory,
        artifact_directory,
        &host,
        super::PROVIDER_MIB * MIB,
        cancelled,
    )
    .await?;
    let identity = source_identity(&model);
    let run = capture_reference_prefill47_decode48_to50(&model, cancelled).await;
    // Once loaded, cancellation never skips physical/source teardown.
    let close = model.shutdown(None).await;
    let (reference, close) =
        combine_run_and_close("reference prefill47 + decode48–50", run, close)?;
    drop(host);
    require_reference_close(&close)?;
    require_reference(&reference)?;
    if identity.capacity != CURRENT_PARITY_CAPACITY {
        return Err(Error::State(
            "reference prefill47 + decode48–50 qualification context must be 512".into(),
        ));
    }
    let evidence = evidence::ReferenceEvidence {
        identity,
        reference,
        close: close.clone(),
    };
    let sha256 = evidence::write(evidence_path, &evidence)?;
    Ok(ReferenceEvidenceReceipt { sha256, close })
}

pub async fn compare_current_source_prefill47_decode48_to50_evidence(
    device: crate::gpu::Device,
    hf_directory: &Path,
    evidence_path: &Path,
    expected_evidence_sha256: [u8; 32],
    cancelled: Option<&AtomicBool>,
) -> Result<ForwardEvidenceReceipt> {
    const MIB: usize = 1024 * 1024;
    require_not_cancelled(cancelled)?;
    let (evidence, sha256) = evidence::read(evidence_path, expected_evidence_sha256)?;
    if evidence.reference.capacity != CURRENT_PARITY_CAPACITY
        || evidence.identity.capacity != CURRENT_PARITY_CAPACITY
    {
        return Err(Error::State(
            "forward parity evidence context differs from fixed 512 qualification".into(),
        ));
    }

    // Evidence does not choose plan/context/provider. Source and 512-token
    // qualification geometry are independently authenticated here.
    let model = forward_candidate::Model::load(
        device,
        hf_directory,
        CURRENT_PARITY_CAPACITY,
        super::HOST_MIB,
        super::PROVIDER_MIB * MIB,
        (super::DEVICE_MIB * MIB) as u64,
        cancelled,
    )
    .await?;
    let run = async {
        if forward_identity(&model) != evidence.identity {
            return Err(Error::State(
                "forward/reference evidence current-source identity differs".into(),
            ));
        }
        let report =
            compare_forward_prefill47_decode48_to50(&model, &evidence.reference, cancelled).await?;
        report.require_exact_source_parity()?;
        Ok(report)
    }
    .await;
    let close = model.shutdown().await;
    let (parity, ()) = combine_run_and_close("forward prefill47 + decode48–50", run, close)?;
    let receipt = ForwardEvidenceReceipt { sha256, parity };
    receipt.require_exact()?;
    Ok(receipt)
}

fn source_identity(model: &ReferenceModel) -> ParitySourceIdentity {
    let metadata = model.source_metadata();
    ParitySourceIdentity {
        family: metadata.family.to_string(),
        repository: metadata.repository.to_string(),
        revision: metadata.revision.to_string(),
        hidden_width: metadata.hidden_width,
        capacity: metadata.context_capacity_tokens,
    }
}

fn forward_identity(model: &forward_candidate::Model) -> ParitySourceIdentity {
    let metadata = model.source_metadata();
    ParitySourceIdentity {
        family: metadata.family.to_string(),
        repository: metadata.repository.to_string(),
        revision: metadata.revision.to_string(),
        hidden_width: metadata.hidden_width,
        capacity: model.context_capacity_tokens(),
    }
}

fn combine_run_and_close<T, C>(label: &str, run: Result<T>, close: Result<C>) -> Result<(T, C)> {
    match (run, close) {
        (Ok(run), Ok(close)) => Ok((run, close)),
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(close)) => Err(close),
        (Err(error), Err(close)) => Err(Error::State(format!(
            "{label} failed and cleanup also failed: run={error}; cleanup={close}"
        ))),
    }
}

fn require_reference_close(close: &crate::gpu::cuda::native::TextCloseReceipt) -> Result<()> {
    if close.source_weight_windows == 0
        || close.source_weight_windows != close.retirement_source_weight_windows
        || close.streamed_head_windows != close.retirement_streamed_head_windows
        || close.resident_head_windows != close.retirement_resident_head_windows
    {
        return Err(Error::State(
            "reference prefill47 + decode48–50 source retirement receipt differs".into(),
        ));
    }
    Ok(())
}

pub async fn capture_reference_prefill47_decode48_to50(
    model: &Arc<ReferenceModel>,
    cancelled: Option<&AtomicBool>,
) -> Result<Prefill47Decode48To50Reference> {
    if model.source_profile() != QwenModelVariant::Qwen800M
        || model.source_metadata().context_capacity_tokens != CURRENT_PARITY_CAPACITY
    {
        return Err(Error::Unsupported(
            "prefill47 + decode48–50 reference requires approved .8B 512-token context".into(),
        ));
    }
    let prompt: Arc<[u32]> = Arc::from(OriginalWindowModelSource::fixed_prompt_ids().as_slice());
    let mut session = model.session(cancelled).await?;
    if session.binding().frontier() != 0 {
        return Err(Error::State(
            "prefill47 + decode48–50 reference session must start empty".into(),
        ));
    }

    let mut steps = Vec::with_capacity(4);
    let mut captured = None;
    let receipt = session
        .infer_prompt_with(&prompt, cancelled, |readback| {
            captured = Some(capture_reference_step(readback)?);
            Ok(())
        })
        .await?;
    steps.push(finalize_reference_step(
        captured
            .take()
            .ok_or_else(|| Error::State("reference prompt readback absent".into()))?,
        &receipt,
        FRONTIERS[0],
    )?);

    for &frontier in &FRONTIERS[1..] {
        let mut captured = None;
        let receipt = session
            .infer_next_with(cancelled, |readback| {
                captured = Some(capture_reference_step(readback)?);
                Ok(())
            })
            .await?;
        steps.push(finalize_reference_step(
            captured
                .take()
                .ok_or_else(|| Error::State("reference cached readback absent".into()))?,
            &receipt,
            frontier,
        )?);
    }
    if session.binding().frontier() != 50 {
        return Err(Error::State(
            "prefill47 + decode48–50 reference final frontier differs".into(),
        ));
    }
    Ok(Prefill47Decode48To50Reference {
        capacity: model.source_metadata().context_capacity_tokens,
        prompt,
        steps,
    })
}

pub async fn compare_forward_prefill47_decode48_to50(
    model: &Arc<forward_candidate::Model>,
    reference: &Prefill47Decode48To50Reference,
    cancelled: Option<&AtomicBool>,
) -> Result<Prefill47Decode48To50ParityReport> {
    if model.model_profile() != QwenModelVariant::Qwen800M.into()
        || model.context_capacity_tokens() != reference.capacity
        || reference.prompt.as_ref() != OriginalWindowModelSource::fixed_prompt_ids().as_slice()
        || reference.steps.len() != FRONTIERS.len()
    {
        return Err(Error::State(
            "forward/reference prefill47 + decode48–50 profile or prompt differs".into(),
        ));
    }
    let mut session = model.session().await?;
    if session.binding().frontier() != 0 {
        return Err(Error::State(
            "forward prefill47 + decode48–50 session must start empty".into(),
        ));
    }

    let qualification_prompt = OriginalWindowModelSource::fixed_prompt_ids();
    let mut report = Prefill47Decode48To50ParityReport {
        steps: Vec::with_capacity(4),
    };
    let mut natural_feedback = None;
    for (index, &expected_frontier) in FRONTIERS.iter().enumerate() {
        require_not_cancelled(cancelled)?;
        let (mode, ids): (SourceCallMode, &[u32]) = if index == 0 {
            (SourceCallMode::Prefill, qualification_prompt.as_slice())
        } else {
            let id = natural_feedback.as_ref().ok_or_else(|| {
                Error::State("forward prefill47 + decode48–50 natural feedback absent".into())
            })?;
            (SourceCallMode::Decode, std::slice::from_ref(id))
        };
        let readback = if let Some(cancelled) = cancelled {
            session
                .forward_cancellable(mode, ids, true, cancelled)
                .await?
        } else {
            session.forward(mode, ids, true).await?
        };
        require_not_cancelled(cancelled)?;
        let binding = session.binding();
        if binding.frontier() as usize != expected_frontier {
            return Err(Error::State(
                "forward prefill47 + decode48–50 committed frontier differs".into(),
            ));
        }
        let domains = session.snapshot().read_domains().await?;
        let step = compare_step(
            &reference.steps[index],
            binding.frontier() as usize,
            &readback.logits,
            &domains,
        )?;
        natural_feedback = Some(step.forward_selected_id);
        let exact = step_is_exact(&step);
        report.steps.push(step);
        if !exact {
            break;
        }
    }
    Ok(report)
}

fn require_reference(reference: &Prefill47Decode48To50Reference) -> Result<()> {
    let fixed = OriginalWindowModelSource::fixed_prompt_ids();
    if reference.capacity != CURRENT_PARITY_CAPACITY
        || reference.prompt.as_ref() != fixed.as_slice()
        || reference.steps.len() != FRONTIERS.len()
    {
        return Err(Error::State(
            "prefill47 + decode48–50 reference fixed context/prompt/step count differs".into(),
        ));
    }
    let slots = expected_domain_slots();
    for (step, &frontier) in reference.steps.iter().zip(&FRONTIERS) {
        if step.frontier != frontier
            || step.logits.len() != VOCABULARY
            || step.selected_id as usize >= VOCABULARY
            || greedy(&step.logits)? != (step.selected_id, step.max_bits)
        {
            return Err(Error::State(
                "prefill47 + decode48–50 reference step/logit/greedy contract differs".into(),
            ));
        }
        let domains = step
            .domains
            .iter()
            .map(|(slot, values)| (*slot, values))
            .collect::<BTreeMap<_, _>>();
        if domains.len() != slots.len() || domains.keys().ne(slots.iter()) {
            return Err(Error::State(
                "prefill47 + decode48–50 reference exact domain set differs".into(),
            ));
        }
        for &slot in &slots {
            let values = domains[&slot];
            if values.len() != expected_domain_values(slot, frontier)?
                || values.iter().any(|value| !value.is_finite())
                || (matches!(slot, StateSlot::Kv(_))
                    && values.iter().any(|value| value.to_bits() & 0xffff != 0))
            {
                return Err(Error::State(
                    "prefill47 + decode48–50 reference domain extent/numerical contract differs"
                        .into(),
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn expected_domain_slots() -> Vec<StateSlot> {
    let mut slots = Vec::with_capacity(42);
    for layer in 0..24 {
        if layer % 4 == 3 {
            slots.push(StateSlot::Kv(KvDomain(layer)));
        } else {
            slots.push(StateSlot::Conv(ConvDomain(layer)));
            slots.push(StateSlot::Recurrent(RecurrentDomain(layer)));
        }
    }
    slots.sort_unstable();
    slots
}

pub(super) fn expected_domain_values(slot: StateSlot, frontier: usize) -> Result<usize> {
    match slot {
        StateSlot::Conv(ConvDomain(layer)) if layer < 24 && layer % 4 != 3 => Ok(6144 * 3),
        StateSlot::Recurrent(RecurrentDomain(layer)) if layer < 24 && layer % 4 != 3 => {
            Ok(16 * 128 * 128)
        }
        StateSlot::Kv(KvDomain(layer)) if layer < 24 && layer % 4 == 3 => 2usize
            .checked_mul(512)
            .and_then(|n| n.checked_mul(frontier))
            .ok_or_else(|| {
                Error::Limit("prefill47 + decode48–50 expected KV domain extent".into())
            }),
        _ => Err(Error::State(
            "prefill47 + decode48–50 unexpected state-domain geometry".into(),
        )),
    }
}

fn step_is_exact(step: &ParityStepReport) -> bool {
    step.selected_matches
        && step.max_bits_match
        && step.logit_mismatches == 0
        && step.domain_mismatch_values == 0
        && step.domain_mismatch_slots.is_empty()
}

fn require_not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
        return Err(Error::State(
            "prefill47 + decode48–50 parity harness cancelled".into(),
        ));
    }
    Ok(())
}

fn capture_reference_step(readback: TextReadback<'_>) -> Result<ReferenceParityStep> {
    let record = readback.record();
    let frontier = record.call().rows().frontier();
    let output = readback.value(record.selection().output())?;
    let logits = decode_bf16(output, VOCABULARY, "reference logits")?;
    let (selected_id, max_bits) = readback.greedy();

    let mut domains = Vec::with_capacity(record.state_requirements().len());
    for requirement in record.state_requirements() {
        if requirement.physical != ResidentPhysicalType::F32 {
            return Err(Error::State(
                "historical reference canonical state is not F32".into(),
            ));
        }
        let (first, second) = readback.state(requirement.slot, true)?;
        let values = match requirement.layout {
            StateLayout::Vector { elements } => {
                if second.is_some() {
                    return Err(Error::State(
                        "reference vector state has extra plane".into(),
                    ));
                }
                decode_f32(first, elements, false, "reference vector state")?
            }
            StateLayout::Kv { slab, capacity } => {
                let second = second
                    .ok_or_else(|| Error::State("reference KV state second plane absent".into()))?;
                if capacity != record.call().rows().capacity() || frontier > capacity {
                    return Err(Error::State(
                        "reference KV state capacity/frontier differs".into(),
                    ));
                }
                let active = slab
                    .checked_mul(frontier)
                    .ok_or_else(|| Error::Limit("reference KV active extent".into()))?;
                let plane = slab
                    .checked_mul(capacity)
                    .ok_or_else(|| Error::Limit("reference KV plane extent".into()))?;
                let mut values =
                    decode_f32_prefix_with_zero_tail(first, active, plane, true, "reference keys")?;
                values.extend(decode_f32_prefix_with_zero_tail(
                    second,
                    active,
                    plane,
                    true,
                    "reference values",
                )?);
                values
            }
        };
        domains.push((requirement.slot, values));
    }
    if domains.len() != 42 {
        return Err(Error::State(
            "reference prefill47 + decode48–50 state-domain coverage differs".into(),
        ));
    }
    Ok(ReferenceParityStep {
        frontier,
        selected_id,
        max_bits,
        logits,
        domains,
    })
}

fn finalize_reference_step(
    step: ReferenceParityStep,
    receipt: &TextCallReceipt,
    expected_frontier: usize,
) -> Result<ReferenceParityStep> {
    if step.frontier != expected_frontier
        || receipt.root.frontier() as usize != expected_frontier
        || receipt.selected_id != step.selected_id
        || receipt.max_bits != step.max_bits
    {
        return Err(Error::State(
            "reference prefill47 + decode48–50 readback/commit receipt differs".into(),
        ));
    }
    Ok(step)
}

fn compare_step(
    reference: &ReferenceParityStep,
    frontier: usize,
    logits: &[f32],
    domains: &[(StateSlot, Vec<f32>)],
) -> Result<ParityStepReport> {
    if reference.frontier != frontier || logits.len() != reference.logits.len() {
        return Err(Error::State(
            "forward/reference parity step frontier/logit extent differs".into(),
        ));
    }
    let (forward_selected_id, forward_max_bits) = greedy(logits)?;
    let (logit_mismatches, max_abs_logit_error) = compare_values(&reference.logits, logits)?;

    let reference_domains = reference
        .domains
        .iter()
        .map(|(slot, values)| (*slot, values.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let actual_domains = domains
        .iter()
        .map(|(slot, values)| (*slot, values.as_slice()))
        .collect::<BTreeMap<_, _>>();
    if reference_domains.len() != 42
        || actual_domains.len() != 42
        || reference_domains.keys().ne(actual_domains.keys())
    {
        return Err(Error::State(
            "forward/reference parity state-domain identity differs".into(),
        ));
    }
    let mut domain_mismatch_values = 0usize;
    let mut domain_mismatch_slots = Vec::new();
    for (&slot, expected) in &reference_domains {
        let actual = actual_domains[&slot];
        let (mismatches, _) = compare_values(expected, actual)?;
        if mismatches != 0 {
            domain_mismatch_slots.push(slot);
            domain_mismatch_values = domain_mismatch_values
                .checked_add(mismatches)
                .ok_or_else(|| Error::Limit("parity state mismatch count".into()))?;
        }
    }

    Ok(ParityStepReport {
        frontier,
        forward_selected_id,
        reference_selected_id: reference.selected_id,
        selected_matches: forward_selected_id == reference.selected_id,
        max_bits_match: forward_max_bits == reference.max_bits,
        logit_mismatches,
        max_abs_logit_error,
        domain_mismatch_values,
        domain_mismatch_slots,
    })
}

fn compare_values(expected: &[f32], actual: &[f32]) -> Result<(usize, f32)> {
    if expected.len() != actual.len() {
        return Err(Error::State("parity vector element count differs".into()));
    }
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    for (&expected, &actual) in expected.iter().zip(actual) {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(Error::State(
                "parity vector contains nonfinite value".into(),
            ));
        }
        if expected.to_bits() != actual.to_bits() {
            mismatches += 1;
        }
        max_abs = max_abs.max((expected - actual).abs());
    }
    Ok((mismatches, max_abs))
}

fn greedy(logits: &[f32]) -> Result<(u32, u32)> {
    let (&first, rest) = logits
        .split_first()
        .ok_or_else(|| Error::State("parity logits empty".into()))?;
    if !first.is_finite() {
        return Err(Error::State("parity logits nonfinite".into()));
    }
    let mut best_index = 0usize;
    let mut best = first;
    for (offset, &value) in rest.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::State("parity logits nonfinite".into()));
        }
        if value > best {
            best = value;
            best_index = offset + 1;
        }
    }
    Ok((
        u32::try_from(best_index)
            .map_err(|_| Error::Limit("parity selected vocabulary index".into()))?,
        best.to_bits(),
    ))
}

fn decode_bf16(raw: &[u8], elements: usize, label: &str) -> Result<Vec<f32>> {
    if raw.len()
        != elements
            .checked_mul(2)
            .ok_or_else(|| Error::Limit(format!("{label} BF16 byte extent")))?
    {
        return Err(Error::State(format!("{label} BF16 byte extent differs")));
    }
    raw.chunks_exact(2)
        .map(|word| {
            let bits = u16::from_ne_bytes([word[0], word[1]]);
            if bits & 0x7f80 == 0x7f80 {
                return Err(Error::State(format!("{label} contains nonfinite BF16")));
            }
            Ok(f32::from_bits(u32::from(bits) << 16))
        })
        .collect()
}

fn decode_f32(
    raw: &[u8],
    elements: usize,
    require_bf16_widened: bool,
    label: &str,
) -> Result<Vec<f32>> {
    if raw.len()
        != elements
            .checked_mul(4)
            .ok_or_else(|| Error::Limit(format!("{label} F32 byte extent")))?
    {
        return Err(Error::State(format!("{label} F32 byte extent differs")));
    }
    raw.chunks_exact(4)
        .map(|word| {
            let bits = u32::from_ne_bytes(word.try_into().unwrap());
            let value = f32::from_bits(bits);
            if !value.is_finite() || (require_bf16_widened && bits & 0xffff != 0) {
                return Err(Error::State(format!(
                    "{label} is nonfinite or not exact BF16-widened F32"
                )));
            }
            Ok(value)
        })
        .collect()
}

fn decode_f32_prefix_with_zero_tail(
    raw: &[u8],
    active: usize,
    total: usize,
    require_bf16_widened: bool,
    label: &str,
) -> Result<Vec<f32>> {
    if active > total
        || raw.len()
            != total
                .checked_mul(4)
                .ok_or_else(|| Error::Limit(format!("{label} total bytes")))?
    {
        return Err(Error::State(format!("{label} active/total extent differs")));
    }
    let active_bytes = active * 4;
    if raw[active_bytes..].iter().any(|byte| *byte != 0) {
        return Err(Error::State(format!("{label} future tail is not exact +0")));
    }
    decode_f32(&raw[..active_bytes], active, require_bf16_widened, label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_evidence_digest_parser_is_exact_and_case_insensitive() {
        let lower = "0123456789abcdef".repeat(4);
        let upper = lower.to_ascii_uppercase();
        assert_eq!(
            parse_evidence_sha256(&lower).unwrap(),
            parse_evidence_sha256(&upper).unwrap()
        );
        assert!(parse_evidence_sha256(&lower[..63]).is_err());
        assert!(parse_evidence_sha256(&(lower[..63].to_owned() + "z")).is_err());
    }

    #[test]
    fn reference_close_requires_stable_retirement_windows() {
        let close = crate::gpu::cuda::native::TextCloseReceipt {
            source_weight_windows: 3,
            streamed_head_windows: 1,
            retirement_source_weight_windows: 3,
            retirement_streamed_head_windows: 1,
            resident_head_windows: 0,
            retirement_resident_head_windows: 0,
        };
        assert!(require_reference_close(&close).is_ok());
        let mut bad = close.clone();
        bad.retirement_source_weight_windows = 2;
        assert!(require_reference_close(&bad).is_err());
    }

    #[test]
    fn expected_domain_geometry_is_exact_18_18_6() {
        let slots = expected_domain_slots();
        assert_eq!(slots.len(), 42);
        assert_eq!(
            slots
                .iter()
                .filter(|slot| matches!(slot, StateSlot::Conv(_)))
                .count(),
            18
        );
        assert_eq!(
            slots
                .iter()
                .filter(|slot| matches!(slot, StateSlot::Recurrent(_)))
                .count(),
            18
        );
        assert_eq!(
            slots
                .iter()
                .filter(|slot| matches!(slot, StateSlot::Kv(_)))
                .count(),
            6
        );
        assert_eq!(
            expected_domain_values(StateSlot::Kv(KvDomain(3)), 50).unwrap(),
            51_200
        );
    }

    #[test]
    fn parity_report_requires_every_exact_dimension() {
        let mut report = Prefill47Decode48To50ParityReport {
            steps: FRONTIERS
                .into_iter()
                .map(|frontier| ParityStepReport {
                    frontier,
                    forward_selected_id: 1,
                    reference_selected_id: 1,
                    selected_matches: true,
                    max_bits_match: true,
                    ..Default::default()
                })
                .collect(),
        };
        assert!(report.require_exact_source_parity().is_ok());
        report.steps[2].logit_mismatches = 1;
        assert!(report.require_exact_source_parity().is_err());
    }

    #[test]
    fn reference_decoders_require_bf16_bits_and_zero_future_tail() {
        let bf16 = [0x3f80u16, 0xbf80]
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        assert_eq!(decode_bf16(&bf16, 2, "test").unwrap(), [1.0, -1.0]);

        let mut f32 = [1.0f32, -1.0, 0.0, 0.0]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            decode_f32_prefix_with_zero_tail(&f32, 2, 4, true, "test").unwrap(),
            [1.0, -1.0]
        );
        f32[12] = 1;
        assert!(decode_f32_prefix_with_zero_tail(&f32, 2, 4, true, "test").is_err());
    }

    #[test]
    fn exact_comparison_reports_mismatch_without_accepting_tolerance() {
        let expected = [1.0f32, 2.0, 3.0];
        let actual = [1.0f32, f32::from_bits(2.0f32.to_bits() + 1), 3.0];
        let (mismatches, max_abs) = compare_values(&expected, &actual).unwrap();
        assert_eq!(mismatches, 1);
        assert!(max_abs > 0.0);
    }
}
