//! Source-issued current-call operands and one original admitted native set.
mod catalogue;
pub use catalogue::{
    short_hf_native_provider_bytes, short_hf_native_provider_bytes_for_cohort,
    short_hf_required_images, short_hf_required_images_for_cohort,
};
mod attention;
mod generated;
mod images;
mod inventory;
mod packets;
mod pool;
mod transport;
pub(crate) use inventory::ShortHfNativeInventory;
mod programs;
mod storage;
pub(crate) use programs::ShortHfPrograms;
pub(crate) use storage::ShortHfNativeStorage;

use super::{
    canonical47_mlp::Mlp47QualificationInputs, model_rotary::ModelRotary,
    model_values::ModelValues, model_weights::ModelWeights, short_hf_model::ShortHfModelBinding,
};
use crate::{
    resident_leases::{ResidentLease, TiedWeight},
    resident_program::text_positions::short_hf::ShortHfPositionsStorage,
    vllm_cuda_input_rms::{FullModelRmsScratch, FusedRmsPrepared},
};
use std::sync::Arc;
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::BoundFullModelSource;

pub(crate) struct ShortHfNativeGeometry {
    pub(crate) scope: Arc<ShortHfModelBinding>,
    pub(crate) values: Arc<ModelValues>,
    pub(crate) rms_scope: Arc<FullModelRmsScratch>,
    pub(crate) final_rms: Arc<FusedRmsPrepared>,
}
pub(crate) struct ShortHfNativePreparation {
    pub(crate) source: BoundFullModelSource,
    pub(crate) geometries: Vec<ShortHfNativeGeometry>,
    pub(crate) weights: Arc<ModelWeights>,
    pub(crate) workspace: ResidentLease,
    pub(crate) rotary: Arc<ModelRotary>,
    pub(crate) positions: Arc<ShortHfPositionsStorage>,
    pub(crate) inputs: Mlp47QualificationInputs,
    pub(crate) audit_head: bool,
    pub(crate) cohort_b4: bool,
    pub(crate) head_mode: super::short_hf_head::HeadMode,
    pub(crate) tied_head: Option<Arc<TiedWeight>>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ShortHfSubmission {
    pub(crate) native: usize,
    pub(crate) clears: usize,
    pub(crate) copies: usize,
}
impl ShortHfSubmission {
    fn difference(self, prior: Self) -> Result<Self> {
        let delta = |a: usize, b: usize| {
            a.checked_sub(b)
                .ok_or_else(|| Error::State("short-HF actual command counter regressed".into()))
        };
        Ok(Self {
            native: delta(self.native, prior.native)?,
            clears: delta(self.clears, prior.clears)?,
            copies: delta(self.copies, prior.copies)?,
        })
    }
}
