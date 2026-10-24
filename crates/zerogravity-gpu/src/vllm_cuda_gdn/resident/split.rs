//! Consumed enqueue token only, not completion/Ready/root authority.
use super::state_adapter::StateCopy;
use super::*;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use zerogravity_core::accelerator::program::StateEffect;

#[must_use = "retain original flight; convolution alone is not a complete state proposal"]
pub(crate) struct GdnConvEnqueued {
    prepared: Arc<GdnPrepared>,
    adapter: Arc<GdnStateAdapter>,
    convolution_dispatches: usize,
}
impl GdnPrepared {
    /// SAME prepared kernels and original armed flight as legacy launch.
    /// Caller retains both Arcs before enqueue. No expected/captured state input.
    pub(crate) fn launch_root_convolution(
        self: &Arc<Self>,
        adapter: Arc<GdnStateAdapter>,
        effect: StateEffect,
        cancelled: Option<&AtomicBool>,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
    ) -> Result<GdnConvEnqueued> {
        not_cancelled(cancelled)?;
        adapter.require_enqueue(permit)?;
        adapter.require_prepared(self)?;
        adapter.require_effect(effect, true)?;
        let count = adapter.copy(StateCopy::ConvIn, self, cancelled)?
            + self.launch_partition(StagePartition::Convolution, cancelled)?
            + adapter.copy(StateCopy::ConvOut, self, cancelled)?;
        Ok(GdnConvEnqueued {
            prepared: self.clone(),
            adapter,
            convolution_dispatches: count,
        })
    }
}
impl GdnConvEnqueued {
    pub(crate) fn convolution_dispatches(&self) -> usize {
        self.convolution_dispatches
    }
    /// Only the recurrence PlanNode consumes this token. A cancellation/error
    /// drops the SAME armed flight into original physical quarantine.
    pub(crate) fn launch_recurrent(
        self,
        effect: StateEffect,
        cancelled: Option<&AtomicBool>,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        let prepared = &self.prepared;
        let adapter = &self.adapter;
        adapter.require_enqueue(permit)?;
        adapter.require_prepared(prepared)?;
        adapter.require_effect(effect, false)?;
        let count = adapter.copy(StateCopy::RecurrentIn, prepared, cancelled)?
            + prepared.launch_partition(StagePartition::Recurrent, cancelled)?
            + adapter.copy(StateCopy::RecurrentOut, prepared, cancelled)?;
        not_cancelled(cancelled)?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_preserves_every_original_stage_in_order_and_all_dispatches() {
        for phase in [
            VllmGdnPhase::InitialOne,
            VllmGdnPhase::Initial64,
            VllmGdnPhase::ContinuedTwo,
            VllmGdnPhase::CachedDecode,
        ] {
            let staged = [StagePartition::Convolution, StagePartition::Recurrent]
                .into_iter()
                .flat_map(|partition| {
                    phase
                        .stages()
                        .iter()
                        .copied()
                        .filter(move |stage| partition.contains(*stage))
                })
                .collect::<Vec<_>>();
            assert_eq!(staged.as_slice(), phase.stages());
            assert_eq!(
                staged
                    .iter()
                    .filter(|&&stage| stage == GdnStage::Conv)
                    .count(),
                1
            );
            let conv = StagePartition::Convolution.dispatches(phase);
            let recurrent = StagePartition::Recurrent.dispatches(phase);
            assert_eq!(
                conv + recurrent,
                match phase {
                    VllmGdnPhase::InitialOne => 12,
                    VllmGdnPhase::CachedDecode => 3,
                    _ => 15,
                }
            );
            assert_eq!(
                conv + recurrent + 4,
                match phase {
                    VllmGdnPhase::InitialOne => 16,
                    VllmGdnPhase::CachedDecode => 7,
                    _ => 19,
                }
            );
        }
    }
}
