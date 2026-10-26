//! Source-issued current-call operands and one original admitted native set.
mod capture;
mod catalogue;
pub(in crate::cuda::native) use capture::NativeCatalogue;
pub use capture::TextKernelCapture;
pub use catalogue::{
    text_native_provider_bytes, text_native_provider_bytes_for_cohort, text_required_images,
    text_required_images_for_cohort,
};
mod attention;
mod generated;
mod images;
mod inventory;
mod packets;
mod pool;
mod transport;
pub(crate) use inventory::TextNativeInventory;
mod programs;
mod storage;
pub(crate) use programs::TextPrograms;
pub(crate) use storage::TextNativeStorage;

use super::{
    fixed_prompt_mlp::FixedPromptMlpQualificationInputs, model_rotary::ModelRotary,
    model_values::ModelValues, model_weights::ModelWeights, text_model::TextModelBinding,
};
use crate::{
    cuda::rms::{FullModelRmsScratch, FusedRmsPrepared},
    resident_leases::{ResidentLease, TiedWeight},
    resident_program::text_positions::text::TextPositionsStorage,
};
use std::sync::Arc;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::{Error, Result};

pub(crate) struct TextNativeGeometry {
    pub(crate) scope: Arc<TextModelBinding>,
    pub(crate) values: Arc<ModelValues>,
    pub(crate) rms_scope: Arc<FullModelRmsScratch>,
    pub(crate) final_rms: Arc<FusedRmsPrepared>,
}
pub(crate) struct TextNativePreparation {
    pub(crate) source: VerifiedTensorSource,
    pub(crate) capture: TextKernelCapture,
    pub(crate) geometries: Vec<TextNativeGeometry>,
    pub(crate) weights: Arc<ModelWeights>,
    pub(crate) workspace: ResidentLease,
    pub(crate) rotary: Arc<ModelRotary>,
    pub(crate) positions: Arc<TextPositionsStorage>,
    pub(crate) inputs: FixedPromptMlpQualificationInputs,
    pub(crate) audit_head: bool,
    pub(crate) cohort_b4: bool,
    pub(crate) head_mode: super::text_head::HeadMode,
    pub(crate) tied_head: Option<Arc<TiedWeight>>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TextSubmission {
    pub(crate) native: usize,
    pub(crate) clears: usize,
    pub(crate) copies: usize,
}
impl TextSubmission {
    fn difference(self, prior: Self) -> Result<Self> {
        let delta = |a: usize, b: usize| {
            a.checked_sub(b)
                .ok_or_else(|| Error::State("text actual command counter regressed".into()))
        };
        Ok(Self {
            native: delta(self.native, prior.native)?,
            clears: delta(self.clears, prior.clears)?,
            copies: delta(self.copies, prior.copies)?,
        })
    }
}
