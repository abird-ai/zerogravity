//! Fixed original .8B47-row RMS leaves; no model/Root/profile permit.
use super::{BoundFinalRmsSource, FusedRmsPrepared, RmsPrepared, resident::finite_bf16};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::{
    OriginalWindowModelSource, QwenMatrixRole, QwenParameterRole,
};

const COUNT: usize = 47;
const COLUMNS: usize = 1024;
const WORDS: usize = COUNT * COLUMNS;
const SOURCE_BYTES: u64 = 1_746_942_600;
const SOURCE_CHUNK: usize = 1024 * 1024;
const PROMPT: [u32; COUNT] = [
    248045, 846, 198, 22791, 411, 1103, 314, 24959, 11, 18101, 539, 73982, 11, 364, 1599, 777,
    3665, 25, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220,
    22, 11, 220, 23, 11, 248046, 198, 248045, 74455, 198,
];

struct SourceOwner {
    model: Arc<OriginalWindowModelSource>,
    final_binding: BoundFinalRmsSource,
    _metadata: WindowWorkspace,
}
#[derive(Clone)]
pub(super) struct Canonical47RmsSource(Arc<SourceOwner>);
impl Canonical47RmsSource {
    fn bind(model: Arc<OriginalWindowModelSource>) -> Result<Self> {
        let metadata = model.source().reserve_workspace(64 * 1024)?;
        model.require_vllm08_full_source()?;
        require_source_extent(
            model.source().source_bytes(),
            model.source().staging_chunk_bytes(),
        )?;
        let embedding = model.matrix_descriptor(QwenMatrixRole::Embedding)?;
        let gamma = model.parameter_descriptor(QwenParameterRole::InputNorm(0))?;
        if embedding.kind != QuantKind::Bf16
            || embedding.rows != 248320
            || embedding.columns != COLUMNS
            || gamma.kind != QuantKind::Bf16
            || gamma.name != "model.language_model.layers.0.input_layernorm.weight"
            || gamma.shape != [COLUMNS as u64]
            || gamma.bytes != COLUMNS * 2
        {
            return Err(Error::Unsupported(
                "canonical47 original BF16 embedding/gamma".into(),
            ));
        }
        model.validate_matrix(QwenMatrixRole::Embedding, &embedding)?;
        model.validate_parameter(QwenParameterRole::InputNorm(0), &gamma)?;
        let final_binding = BoundFinalRmsSource::bind(model.clone())?;
        Ok(Self(Arc::new(SourceOwner {
            model,
            final_binding,
            _metadata: metadata,
        })))
    }
    pub(super) fn source(&self) -> &VerifiedTensorSource {
        self.0.model.source()
    }
    pub(super) fn final_binding(&self) -> &BoundFinalRmsSource {
        &self.0.final_binding
    }
    pub(super) fn require_source(&self, source: &VerifiedTensorSource) -> Result<()> {
        if !source.same_issuer(self.source()) {
            return Err(Error::State(
                "canonical47 original source issuer differs".into(),
            ));
        }
        self.0.model.require_vllm08_full_source()?;
        require_source_extent(source.source_bytes(), source.staging_chunk_bytes())?;
        self.0.final_binding.require_source(source)
    }
}
fn require_source_extent(bytes: u64, chunk: usize) -> Result<()> {
    if bytes != SOURCE_BYTES || chunk != SOURCE_CHUNK {
        return Err(Error::Unsupported(
            "canonical47 original source extent/1MiB staging".into(),
        ));
    }
    Ok(())
}
pub(super) fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("canonical47 RMS cancelled".into()));
    }
    Ok(())
}

/// Original final branch/residual are explicit observed LEAF operands.
pub struct Canonical47DualRmsLeaf {
    pub branch: Vec<u8>,
    pub residual: Vec<u8>,
}
pub enum Canonical47RmsReadback<'a> {
    SourceInput {
        teacher_ids: &'a [u32; COUNT],
        embedding: &'a [u8],
        normalized: &'a [u8],
    },
    FinalDualLeaf {
        sum_f32: &'a [u8],
        normalized: &'a [u8],
        residual: &'a [u8],
    },
}
#[derive(Clone, Debug)]
pub struct Canonical47RmsReceipt {
    pub source_embedding_bf16_words: usize,
    pub ordinary_bf16_words: usize,
    pub dual_sum_f32_words: usize,
    pub dual_normalized_bf16_words: usize,
    pub dual_residual_bf16_words: usize,
    pub actual_stage_dispatches: usize,
    pub readback_calls: usize,
}

fn validate_fixture(ids: &[u32; COUNT], leaf: &Canonical47DualRmsLeaf) -> Result<()> {
    if ids != &PROMPT
        || leaf.branch.len() != WORDS * 2
        || leaf.residual.len() != WORDS * 2
        || !finite_bf16(&leaf.branch)
        || !finite_bf16(&leaf.residual)
    {
        return Err(Error::Malformed(
            "canonical47 exact prompt/finite dual-leaf extents".into(),
        ));
    }
    Ok(())
}

#[path = "canonical47/qualification.rs"]
mod qualification;
pub use qualification::qualify_canonical47_rms;
#[path = "canonical47/faults.rs"]
mod faults;
pub use faults::{Canonical47RmsFault, Canonical47RmsFaultOutcome, qualify_canonical47_rms_fault};

#[cfg(test)]
mod tests {
    use super::super::Rows;
    use super::*;
    #[test]
    fn private47_is_not_a_legacy_or_public_shape_constructor() {
        assert_eq!(PROMPT.len(), COUNT);
        assert_eq!(Rows::FortySeven.count(), 47);
        assert!(Rows::new(47).is_err());
        let leaf = Canonical47DualRmsLeaf {
            branch: vec![0; WORDS * 2],
            residual: vec![0; WORDS * 2],
        };
        assert!(validate_fixture(&PROMPT, &leaf).is_ok());
        let mut wrong = PROMPT;
        wrong[0] = 353;
        assert!(validate_fixture(&wrong, &leaf).is_err());
        assert!(
            validate_fixture(
                &PROMPT,
                &Canonical47DualRmsLeaf {
                    branch: vec![0; 64 * COLUMNS * 2],
                    residual: vec![0; WORDS * 2]
                }
            )
            .is_err()
        );
        let mut nan = leaf;
        nan.branch[..2].copy_from_slice(&0x7fc0u16.to_le_bytes());
        assert!(validate_fixture(&PROMPT, &nan).is_err());
    }

    #[test]
    fn source_staging_scope_never_inherits2b_or_unknown_chunk_budget() {
        assert!(require_source_extent(SOURCE_BYTES, SOURCE_CHUNK).is_ok());
        for (bytes, chunk) in [
            (4_548_221_488, SOURCE_CHUNK),
            (SOURCE_BYTES, 4 * SOURCE_CHUNK),
            (0, SOURCE_CHUNK),
            (SOURCE_BYTES, 0),
        ] {
            assert!(require_source_extent(bytes, chunk).is_err());
        }
    }
}

#[cfg(feature = "vllm-cuda-qualification")]
#[path = "canonical47/component.rs"]
mod component;
