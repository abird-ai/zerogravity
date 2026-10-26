//! Same source0..47/page4 copy code, actual per-layer canonical domains.
use super::*;
use crate::cuda::native::fixed_prompt_model_prefix::FixedPromptPrefixCut;
impl ModelKvTransport {
    pub(crate) fn prepare_whole(
        cut: &Arc<FixedPromptPrefixCut>,
        binding: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        cut.require_whole()?;
        binding.require_fixed_prompt_whole(cut)?;
        cut.attention_site(binding.layer())?;
        arena.require_binding(&binding)?;
        Self::prepare_checked(
            cut.scope().source().clone(),
            binding,
            arena,
            init,
            cancelled,
            (0, 47, 4),
        )
    }
}
