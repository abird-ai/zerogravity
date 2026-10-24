//! Same source0..47/page4 copy code, actual per-layer canonical domains.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut;
impl ModelKvTransport {
    pub(crate) fn prepare_whole(
        cut: &Arc<Canonical47PrefixCut>,
        binding: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        cut.require_whole()?;
        binding.require_canonical47_whole(cut)?;
        cut.attention_site(binding.layer().index())?;
        arena.require_binding(&binding)?;
        Self::prepare_checked(
            cut.scope().source().source().clone(),
            binding,
            arena,
            init,
            cancelled,
            (0, 47, 4),
        )
    }
}
