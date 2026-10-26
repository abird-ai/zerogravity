//! Fixed original .8B47-row RMS leaves; no model/Root/profile permit.
use super::{FusedRmsPrepared, RmsPrepared, resident::finite_bf16};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{Error, Result};

const COUNT: usize = 47;
const COLUMNS: usize = 1024;
const WORDS: usize = COUNT * COLUMNS;
const PROMPT: [u32; COUNT] = [
    248045, 846, 198, 22791, 411, 1103, 314, 24959, 11, 18101, 539, 73982, 11, 364, 1599, 777,
    3665, 25, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220,
    22, 11, 220, 23, 11, 248046, 198, 248045, 74455, 198,
];

pub(super) fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("fixed_prompt RMS cancelled".into()));
    }
    Ok(())
}

/// Original final branch/residual are explicit observed LEAF operands.
pub struct FixedPromptDualRmsLeaf {
    pub branch: Vec<u8>,
    pub residual: Vec<u8>,
}
pub enum FixedPromptRmsReadback<'a> {
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
pub struct FixedPromptRmsReceipt {
    pub source_embedding_bf16_words: usize,
    pub ordinary_bf16_words: usize,
    pub dual_sum_f32_words: usize,
    pub dual_normalized_bf16_words: usize,
    pub dual_residual_bf16_words: usize,
    pub actual_stage_dispatches: usize,
    pub readback_calls: usize,
}

fn validate_fixture(ids: &[u32; COUNT], leaf: &FixedPromptDualRmsLeaf) -> Result<()> {
    if ids != &PROMPT
        || leaf.branch.len() != WORDS * 2
        || leaf.residual.len() != WORDS * 2
        || !finite_bf16(&leaf.branch)
        || !finite_bf16(&leaf.residual)
    {
        return Err(Error::Malformed(
            "fixed_prompt exact prompt/finite dual-leaf extents".into(),
        ));
    }
    Ok(())
}

#[path = "fixed_prompt/qualification.rs"]
mod qualification;
pub use qualification::qualify_fixed_prompt_rms;
#[path = "fixed_prompt/faults.rs"]
mod faults;
pub use faults::{FixedPromptRmsFault, FixedPromptRmsFaultOutcome, qualify_fixed_prompt_rms_fault};

#[cfg(test)]
mod tests {
    use super::super::Rows;
    use super::*;
    #[test]
    fn private47_is_not_a_legacy_or_public_shape_constructor() {
        assert_eq!(PROMPT.len(), COUNT);
        assert_eq!(Rows::FixedPrompt.count(), 47);
        assert!(Rows::new(47).is_err());
        let leaf = FixedPromptDualRmsLeaf {
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
                &FixedPromptDualRmsLeaf {
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
}
