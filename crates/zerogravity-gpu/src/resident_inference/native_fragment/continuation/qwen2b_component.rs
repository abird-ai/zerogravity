//! ONE genuine initial attempt for the two-domain2B diagnostic; no publisher.
use super::*;
use zerogravity_core::accelerator::program::FullModelProfile;
use zerogravity_model_qwen35::original_binding::BoundFullModelSource;
impl GdnAttemptSchedule {
    pub(in crate::resident_inference) fn new_initial_qwen2b_component(
        source: &BoundFullModelSource,
        attempt: RootAttempt,
        invocation: VllmBf16Invocation,
    ) -> Result<Arc<Self>> {
        invocation.validate()?;
        let base = attempt.base();
        if source.model_profile() != FullModelProfile::Qwen2B
            || base.source() != source.source().source_identity()
            || invocation.layer().source() != base.source()
            || invocation.layer().index() != 0
            || base.frontier() != 0
            || invocation.teachers().base() != 0
            || invocation.teachers().capacity() != 512
            || !matches!(
                invocation.legacy_phase()?,
                VllmBf16Phase::InitialOne | VllmBf16Phase::Initial64
            )
            || attempt.frontier() != invocation.rows() as u64
        {
            return Err(Error::Unsupported(
                "genuine current initial2B component attempt only".into(),
            ));
        }
        // Existing1-layer permit means one original Conv/H pair here. No
        // CompletedPrefixPhase/full-layer/RootReady publication is constructed.
        Ok(Arc::new(Self {
            base,
            attempt,
            invocations: vec![invocation],
            completed_layers: 1,
        }))
    }
}
