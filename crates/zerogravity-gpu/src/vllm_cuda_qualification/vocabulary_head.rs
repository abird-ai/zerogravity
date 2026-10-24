//! Original tied-W native head leaf; both scopes use one original lifecycle.
use super::cubin_manifests::ProjectionImage;
use super::projection_cubin::{CubinQualificationInputs, native_error, not_cancelled};
use crate::resident_leases::{
    ResidentLease, ResidentReadback, ResidentService, TIED_BYTES, TIED_WINDOW_BYTES, TIED_WINDOWS,
    TiedWeight,
};
use crate::vllm_cuda_input_rms::{BoundFinalRmsSource, FusedRmsPrepared};
use cubecl::{Device, bytes::Bytes};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{
    Error, Result,
    device_events::{BufferLeaseId, BufferRole, DeviceAccess},
    resource_arbiter::{AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner},
};
use zerogravity_model_qwen35::original_binding::{OriginalWindowModelSource, QwenMatrixRole};
mod faults;
mod qualify;
mod resident;
#[cfg(test)]
mod tests;
pub use faults::{VocabularyHeadFault, VocabularyHeadFaultOutcome, qualify_vocabulary_head_fault};
pub use qualify::{qualify_final_norm_head_cubins, qualify_vocabulary_head_cubins};
use resident::{HeadInputView, HeadPrograms};

const HEAD_CAP: usize = 576 * 1024 * 1024;
const NATIVE_BYTES: usize = 64 * 1024 * 1024;
const HOST_CAP: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabularyHeadCase {
    Initial11,
    Decode353After11,
    InitialFirst64,
    Continued65And66,
    Decode353After66,
}
impl VocabularyHeadCase {
    const ALL: [Self; 5] = [
        Self::Initial11,
        Self::Decode353After11,
        Self::InitialFirst64,
        Self::Continued65And66,
        Self::Decode353After66,
    ];
    pub fn phase_rows(self) -> usize {
        match self {
            Self::InitialFirst64 => 64,
            Self::Continued65And66 => 2,
            _ => 1,
        }
    }
    pub fn selected_row(self) -> usize {
        self.phase_rows() - 1
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabularyHeadOutput {
    SumF32,
    Normalized,
    Residual,
    Selected,
    Vocabulary,
}
impl VocabularyHeadOutput {
    fn bytes(self, case: VocabularyHeadCase) -> usize {
        match self {
            Self::SumF32 => case.phase_rows() * 4096,
            Self::Normalized | Self::Residual => case.phase_rows() * 2048,
            Self::Selected => 2048,
            Self::Vocabulary => 496_640,
        }
    }
}
pub struct VocabularyHeadExpected {
    pub output: VocabularyHeadOutput,
    pub bytes: Vec<u8>,
}
pub struct FinalNormInputs {
    pub branch: Vec<u8>,
    pub residual: Vec<u8>,
}
pub struct VocabularyHeadFixture {
    pub case: VocabularyHeadCase,
    /// Explicit captured mathematical INPUT, only in the captured-selected scope.
    pub selected: Vec<u8>,
    pub final_norm: Option<FinalNormInputs>,
    pub expected: Vec<VocabularyHeadExpected>,
}
/// Reserve same-source admission BEFORE allocating fixture payloads. None is
/// malformed, retained only so earliest cancellation can preempt validation.
pub struct VocabularyHeadFixtures {
    pub fixtures: Vec<VocabularyHeadFixture>,
    pub admission: Option<WindowWorkspace>,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct VocabularyHeadComparison {
    pub calls: usize,
    pub native_launches: usize,
    pub producer_flights: usize,
    pub rms_stages: usize,
    pub sum_f32_words: usize,
    pub normalized_bf16_words: usize,
    pub residual_bf16_words: usize,
    pub selected_bf16_words: usize,
    pub vocabulary_bf16_words: usize,
    pub weight_copies: usize,
    pub weight_checkpoints: usize,
    pub weight_bytes_compared: usize,
    pub weight_readback_flights: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabularyHeadScope {
    CapturedSelected,
    ComputedFinalNorm,
}
type Scope = VocabularyHeadScope;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    Compare,
    Fault(VocabularyHeadFault),
}
pub(super) enum Outcome {
    Compared(VocabularyHeadComparison),
    Fault(VocabularyHeadFaultOutcome),
}
struct HeadOwner {
    cohort: VocabularyHeadFixtures,
    source: Arc<OriginalWindowModelSource>,
    _metadata: WindowWorkspace,
}
struct PhaseJob {
    case: VocabularyHeadCase,
    input: HeadInputView,
    output: ResidentLease,
    fused: Option<Arc<FusedRmsPrepared>>,
    reads: Vec<ResidentReadback>,
}
struct Job {
    phase: Arc<PhaseJob>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    outputs: Vec<ResidentLease>,
}
impl std::ops::Deref for Job {
    type Target = PhaseJob;
    fn deref(&self) -> &PhaseJob {
        &self.phase
    }
}
enum BaseInput {
    Captured(ResidentLease),
    Final {
        branch: ResidentLease,
        prior: ResidentLease,
    },
}
struct BasePhase {
    case: VocabularyHeadCase,
    input: BaseInput,
    output: ResidentLease,
}

fn finite_bf16(bytes: &[u8]) -> bool {
    bytes.len().is_multiple_of(2)
        && bytes
            .chunks_exact(2)
            .all(|b| (u16::from_le_bytes([b[0], b[1]]) & 0x7f80) != 0x7f80)
}
fn finite_f32(bytes: &[u8]) -> bool {
    bytes.len().is_multiple_of(4)
        && bytes
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
}
fn validate_record(
    fixture: &VocabularyHeadFixture,
    case: VocabularyHeadCase,
    scope: Scope,
) -> Result<()> {
    if fixture.case != case || fixture.expected.capacity() > 5 {
        return Err(Error::Malformed("head original call/phase order".into()));
    }
    let roles: &[VocabularyHeadOutput] = match scope {
        Scope::CapturedSelected => &[VocabularyHeadOutput::Vocabulary],
        Scope::ComputedFinalNorm => &[
            VocabularyHeadOutput::SumF32,
            VocabularyHeadOutput::Normalized,
            VocabularyHeadOutput::Residual,
            VocabularyHeadOutput::Selected,
            VocabularyHeadOutput::Vocabulary,
        ],
    };
    if fixture.expected.len() != roles.len()
        || fixture.expected.iter().zip(roles).any(|(e, r)| {
            e.output != *r
                || e.bytes.len() != r.bytes(case)
                || if *r == VocabularyHeadOutput::SumF32 {
                    !finite_f32(&e.bytes)
                } else {
                    !finite_bf16(&e.bytes)
                }
        })
    {
        return Err(Error::Malformed(
            "head independent complete expected role cohort".into(),
        ));
    }
    match scope {
        Scope::CapturedSelected
            if fixture.selected.len() == 2048
                && finite_bf16(&fixture.selected)
                && fixture.final_norm.is_none() => {}
        Scope::ComputedFinalNorm if fixture.selected.is_empty() => {
            let p = fixture
                .final_norm
                .as_ref()
                .ok_or_else(|| Error::Malformed("final norm original dual inputs absent".into()))?;
            if p.branch.len() != case.phase_rows() * 2048
                || p.residual.len() != p.branch.len()
                || !finite_bf16(&p.branch)
                || !finite_bf16(&p.residual)
            {
                return Err(Error::Malformed(
                    "final norm true phase BF16 dual input geometry".into(),
                ));
            }
        }
        _ => {
            return Err(Error::Malformed(
                "captured vs computed head inputs cannot mix".into(),
            ));
        }
    }
    Ok(())
}
fn validate_fixtures(
    source: &OriginalWindowModelSource,
    cohort: &VocabularyHeadFixtures,
    scope: Scope,
    mode: Mode,
) -> Result<()> {
    let count = if mode == Mode::Compare { 5 } else { 1 };
    if cohort.fixtures.len() != count || cohort.fixtures.capacity() > 5 {
        return Err(Error::Malformed(
            "closed five-phase head cohort or one initial fault".into(),
        ));
    }
    let admission = cohort
        .admission
        .as_ref()
        .ok_or_else(|| Error::State("head fixture host admission absent".into()))?;
    if !admission.source().same_issuer(source.source()) || admission.bytes() > HOST_CAP {
        return Err(Error::State(
            "head fixture original issuer/admission".into(),
        ));
    }
    let mut allocated = cohort.fixtures.capacity() * std::mem::size_of::<VocabularyHeadFixture>();
    for (fixture, case) in cohort.fixtures.iter().zip(VocabularyHeadCase::ALL) {
        validate_record(fixture, case, scope)?;
        allocated = allocated
            .checked_add(fixture.selected.capacity())
            .and_then(|n| {
                n.checked_add(
                    fixture.expected.capacity() * std::mem::size_of::<VocabularyHeadExpected>(),
                )
            })
            .ok_or_else(|| Error::Limit("head host capacity overflow".into()))?;
        for e in &fixture.expected {
            allocated = allocated
                .checked_add(e.bytes.capacity())
                .ok_or_else(|| Error::Limit("head expected capacity overflow".into()))?;
        }
        if let Some(p) = &fixture.final_norm {
            allocated = allocated
                .checked_add(p.branch.capacity())
                .and_then(|n| n.checked_add(p.residual.capacity()))
                .ok_or_else(|| Error::Limit("head norm capacity overflow".into()))?;
        }
    }
    if allocated > admission.bytes() || allocated > HOST_CAP {
        return Err(Error::Limit(
            "head fixture actual owned capacities exceed original admission".into(),
        ));
    }
    Ok(())
}
fn insert_access(
    all: &mut BTreeMap<BufferLeaseId, (ResidentLease, DeviceAccess)>,
    lease: ResidentLease,
    access: DeviceAccess,
) -> Result<()> {
    if let Some((old, mode)) = all.get_mut(&lease.id()) {
        if !Arc::ptr_eq(old.service(), lease.service())
            || old.bytes() != lease.bytes()
            || old.allocation_generation() != lease.allocation_generation()
        {
            return Err(Error::State(
                "head aliased value original lease identity differs".into(),
            ));
        }
        if access == DeviceAccess::Write {
            *mode = access;
        }
    } else {
        all.insert(lease.id(), (lease, access));
    }
    Ok(())
}

mod model;
pub(crate) use model::ModelHeadPrograms;
