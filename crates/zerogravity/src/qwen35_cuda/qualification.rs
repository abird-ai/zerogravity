//! Model-owned finite qualification loops; the GPU owns preparation and commit.
use crate::gpu::cuda::native::RotaryOutcome;
use crate::gpu::cuda::native::{
    CommittedBf16Feedback, NaturalFeedbackRequest, NaturalReceipt, NaturalReuseReceipt,
    NaturalReuseRound, NaturalSlot, SourceModelInputs, SourceModelReadback, SourceModelScopes,
    prepare_natural_cold, prepare_reusable_natural_cold,
};
use crate::gpu::{Device, SourceBindings};
use crate::qwen35::plan::{
    QwenFragmentPlan, QwenGdnCall, QwenInputRows, QwenPlan, QwenPrefixLayer, QwenSourceCall,
    ReferencePhase,
};
use crate::qwen35::source::binding::OriginalWindowModelSource;
use crate::{Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
mod leaves;
mod natural;
pub use leaves::*;
mod reuse;
mod source;
pub use natural::qualify_natural_model;
pub use reuse::qualify_natural_reuse;
pub use source::{
    prepare_leaf_source, prepare_rms_leaf_source, prepare_rotary_source,
    prepare_tied_weight_source, qualify_qwen2b_mlp_weights,
};

/// Consume the original model plans once for forced/fault GPU diagnostics.
pub fn source_scopes(
    source: &Arc<OriginalWindowModelSource>,
    records: [QwenPlan; 5],
) -> Result<SourceModelScopes> {
    let prepared = PreparedScopes::new(source.clone(), records)?;
    SourceModelScopes::new(prepared.source.source().clone(), prepared.templates)
}

/// Consume a model-authenticated fragment into the same common graph owner.
pub fn fragment_scope(
    source: &Arc<OriginalWindowModelSource>,
    record: QwenFragmentPlan,
) -> Result<Arc<SourceBindings>> {
    super::bindings::prepare(source, source.prepare_fragment(record)?, None)
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(
            "natural full24 cancelled before publication".into(),
        ));
    }
    Ok(())
}
/// The original five model-authenticated plans are consumed once. Hot feedback
/// changes only its authentic call issuer; immutable mathematical semantics share.
struct PreparedScopes {
    source: Arc<OriginalWindowModelSource>,
    templates: [Arc<SourceBindings>; 5],
}
impl PreparedScopes {
    fn new(source: Arc<OriginalWindowModelSource>, records: [QwenPlan; 5]) -> Result<Self> {
        source.require_small_model_source()?;
        let records = records
            .into_iter()
            .map(|record| {
                let prepared = source.prepare_reference_program(record)?;
                super::bindings::prepare(&source, prepared, None)
            })
            .collect::<Result<Vec<_>>>()?;
        let templates = records
            .try_into()
            .map_err(|_| Error::State("exact five source plans".into()))?;
        let value = Self { source, templates };
        SourceModelScopes::new(value.source.source().clone(), value.templates.clone())?;
        Ok(value)
    }
    fn cold_scopes(&self) -> Result<SourceModelScopes> {
        SourceModelScopes::new(self.source.source().clone(), self.templates.clone())
    }
    fn bind(
        &self,
        slot: NaturalSlot,
        feedback: Option<&NaturalFeedbackRequest<'_>>,
    ) -> Result<Arc<SourceBindings>> {
        let template = &self.templates[slot.index()];
        let ids = match feedback {
            Some(request) if slot.feedback() && request.base() == slot.base() => {
                vec![request.selected_id()]
            }
            None if !slot.feedback() => template.call().rows().ids().to_vec(),
            _ => {
                return Err(Error::State(
                    "model natural prompt/actual feedback scope differs".into(),
                ));
            }
        };
        let phase = match slot {
            NaturalSlot::InitialOne => ReferencePhase::InitialOne,
            NaturalSlot::FeedbackAtOne | NaturalSlot::FeedbackAt66 => ReferencePhase::CachedOne,
            NaturalSlot::Prefix64 => ReferencePhase::Initial64,
            NaturalSlot::FinalTwo => ReferencePhase::ContinuedTwo,
        };
        let source = self.source.source().source_identity();
        let call = QwenSourceCall::new(QwenGdnCall::new(
            QwenPrefixLayer::new(source, 0)?,
            phase,
            QwenInputRows::new(source, slot.base(), 512, &ids)?,
        )?)?;
        let actual = self.source.bind_reference_call(&call)?;
        let metadata = self
            .source
            .source()
            .reserve_workspace(template.rebind_metadata_bytes()?)?;
        Ok(Arc::new(template.rebind(actual, metadata)?))
    }
}
