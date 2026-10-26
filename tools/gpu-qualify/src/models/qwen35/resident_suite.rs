//! Model-specific resident qualification; physical probes remain in GPU.
use std::{path::Path, sync::Arc};
use zerogravity_core::accelerator::program::{PlanObservation as LayerObservation, StateSlot};
use zerogravity_core::device_events::BufferRole;
use zerogravity_core::resource_arbiter::ResourceOwner;
use zerogravity_core::session_root::RootBinding;
use zerogravity_core::{Error, Result};
#[cfg(feature = "cuda")]
use zerogravity_gpu::qualification::resident::quarantined_completion_owners_for;
#[cfg(feature = "cuda")]
use zerogravity_gpu::qualification::resident::{CapturedResidentGraph, Readout, ResidentOutput};
use zerogravity_gpu::qualification::resident::{
    QualificationService, StateLayout, compare_domains, exact, quarantined_handles_for,
};
#[cfg(feature = "hardware-abort-probe")]
use zerogravity_gpu::qualification::resident::{
    qualify_preserved_packets, qualify_preserved_prior,
};
use zerogravity_gpu::{Device, InferenceReadback, ResidentModel, ResidentSession};
use zerogravity_model_qwen35::{LoadedQwenModel, QwenDenseCpu};
#[cfg(feature = "portable-checkpoint")]
mod checkpoint;
mod cohort;
mod context;
mod fused;
mod fused_greedy;
#[cfg(feature = "cuda")]
mod graph;
#[cfg(feature = "cuda")]
mod graph_wide;
mod prefill;
mod qualification;
#[cfg(feature = "portable-checkpoint")]
pub use checkpoint::qualify as qualify_checkpoint;
pub use cohort::qualify as qualify_cohort;
pub use fused::qualify as qualify_fused;
#[cfg(feature = "hardware-abort-probe")]
pub use fused::qualify_abort as qualify_fused_abort;
#[cfg(feature = "portable-checkpoint")]
pub use fused::qualify_boundary as qualify_fused_boundary;
pub use fused_greedy::qualify as qualify_fused_greedy;
#[cfg(feature = "hardware-abort-probe")]
pub use fused_greedy::qualify_abort as qualify_fused_greedy_abort;
#[cfg(feature = "cuda")]
pub use graph::qualify as qualify_graph;
#[cfg(all(feature = "cuda", feature = "hardware-abort-probe"))]
pub use graph::qualify_abort as qualify_graph_abort;
#[cfg(feature = "cuda")]
pub use graph_wide::qualify as qualify_graph_wide;
#[cfg(all(feature = "cuda", feature = "hardware-abort-probe"))]
pub use graph_wide::qualify_abort as qualify_graph_wide_abort;
pub use qualification::{
    qualify as qualify_resident, qualify_bounded_context as qualify_bounded_resident_context,
    qualify_context,
};
#[cfg(feature = "hardware-abort-probe")]
pub use qualification::{qualify_abort, qualify_context_abort};

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

#[cfg(feature = "portable-checkpoint")]
trait QwenCheckpointBridge {
    async fn checkpoint(
        &self,
        source: &LoadedQwenModel,
    ) -> Result<zerogravity_model_qwen35::QwenCheckpoint>;
    async fn import_checkpoint(
        &mut self,
        checkpoint: &zerogravity_model_qwen35::QwenCheckpoint,
    ) -> Result<()>;
}
#[cfg(feature = "portable-checkpoint")]
impl QwenCheckpointBridge for ResidentSession {
    async fn checkpoint(
        &self,
        source: &LoadedQwenModel,
    ) -> Result<zerogravity_model_qwen35::QwenCheckpoint> {
        if source.source_identity() != self.binding().source() {
            return Err(Error::State(
                "checkpoint codec belongs to another loaded source".into(),
            ));
        }
        let snapshot = self.snapshot();
        let domains = snapshot.read_domains().await?;
        source.checkpoint_from_domains(snapshot.binding().frontier() as usize, &domains)
    }
    async fn import_checkpoint(
        &mut self,
        checkpoint: &zerogravity_model_qwen35::QwenCheckpoint,
    ) -> Result<()> {
        self.import_state_domains(
            checkpoint.source_identity(),
            checkpoint.position(),
            &checkpoint.domain_values(),
        )
        .await
    }
}
