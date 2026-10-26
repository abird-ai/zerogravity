//! Closed original2B RMS diagnostics. No model/profile/publication permit.
use super::{FusedRmsPrepared, RmsPrepared, Rows, resident::finite_bf16};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{Error, Result};

const COUNTS: [usize; 5] = [1, 1, 64, 2, 1];
pub(super) fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("original2B RMS cancelled".into()));
    }
    Ok(())
}

/// Observed final branch/residual inputs are explicitly LEAF operands. They
/// establish neither source-produced upstream layers nor a wholemodel route.
pub struct Qwen2bDualRmsLeaf {
    pub branch: Vec<u8>,
    pub residual: Vec<u8>,
}

/// Expected archives may be opened only by this post-completion callback.
pub enum Qwen2bRmsReadback<'a> {
    SourceInput {
        case: usize,
        teacher_ids: &'a [u32],
        embedding: &'a [u8],
        normalized: &'a [u8],
    },
    FinalDualLeaf {
        case: usize,
        sum_f32: &'a [u8],
        normalized: &'a [u8],
        residual: &'a [u8],
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen2bRmsReceipt {
    pub source_embedding_bf16_words: usize,
    pub ordinary_bf16_words: usize,
    pub dual_sum_f32_words: usize,
    pub dual_normalized_bf16_words: usize,
    pub dual_residual_bf16_words: usize,
    pub actual_stage_dispatches: usize,
}

#[path = "qwen2b/qualification.rs"]
mod qualification;
pub use qualification::qualify_qwen2b_rms;

#[cfg(test)]
#[path = "qwen2b/tests.rs"]
mod tests;

#[path = "qwen2b/faults.rs"]
mod faults;
pub use faults::{Qwen2bRmsFault, Qwen2bRmsFaultOutcome, qualify_qwen2b_rms_fault};
