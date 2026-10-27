//! Model-owned finite qualification loops; the GPU owns preparation and commit.
use crate::gpu::cuda::gdn::{
    GdnKernelConfig, GdnKktKernel, GdnOutputKernel, GdnRecomputeKernel, GdnStateKernel,
    TextGdnImplementation,
};
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
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_artifacts::cache::TransferAdmission;
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

/// Explicit historical finite N1 loader. This API exists only with the
/// `qwen35-cuda-qualification` diagnostic feature.
pub async fn load(
    device: Device,
    hf_directory: &Path,
    artifact_directory: &Path,
    host: &TransferAdmission,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<super::Model>> {
    super::load_profile(
        super::QualificationMode::HistoricalN1,
        device,
        hf_directory,
        artifact_directory,
        host,
        provider_capacity_bytes,
        cancelled,
    )
    .await
}

/// Explicit historical finite B4 loader.
pub async fn load_b4(
    device: Device,
    hf_directory: &Path,
    artifact_directory: &Path,
    host: &TransferAdmission,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<super::Model>> {
    super::load_profile(
        super::QualificationMode::HistoricalB4,
        device,
        hf_directory,
        artifact_directory,
        host,
        provider_capacity_bytes,
        cancelled,
    )
    .await
}

/// Qualification-only four-scope, one-call loader. It does not admit decode.
pub async fn load_fresh_prefill(
    device: Device,
    hf_directory: &Path,
    artifact_directory: &Path,
    host: &TransferAdmission,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<super::Model>> {
    super::load_profile(
        super::QualificationMode::FreshPrefill,
        device,
        hf_directory,
        artifact_directory,
        host,
        provider_capacity_bytes,
        cancelled,
    )
    .await
}

/// Historical finite schedule retained only by explicit qualification paths.
pub(super) const PROMPT_TOKEN_COUNTS: &[usize] = &[47, 57, 62, 67, 72];
pub(super) const FRESH_PREFILL_TOKEN_COUNTS: &[usize] = &[47, 48, 49, 50];

/// Qualification-only historical identity, consumed before the GPU descriptor.
struct HistoricalGdnImplementation {
    recipe_sha256: &'static str,
    input_tokens: usize,
    is_decode: bool,
    config: Option<GdnKernelConfig>,
}
impl HistoricalGdnImplementation {
    fn authenticated(
        recipe_sha256: &'static str,
        input_tokens: usize,
        is_decode: bool,
        config: Option<GdnKernelConfig>,
    ) -> Result<Self> {
        if recipe_sha256.len() != 64
            || !recipe_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || is_decode != config.is_none()
            || (is_decode && input_tokens != 1)
        {
            return Err(Error::State(
                "historical qualification GDN recipe/config identity".into(),
            ));
        }
        Ok(Self {
            recipe_sha256,
            input_tokens,
            is_decode,
            config,
        })
    }
    fn descriptor(self) -> Result<TextGdnImplementation> {
        let _authenticated_recipe = self.recipe_sha256;
        if self.is_decode {
            Ok(TextGdnImplementation::cached())
        } else {
            TextGdnImplementation::prefill(
                self.input_tokens,
                self.config.ok_or_else(|| {
                    Error::State("qualification prefill GDN config absent".into())
                })?,
            )
        }
    }
}

/// Exact independently captured recipes. Qualification authenticates this
/// finite envelope before constructing a recipe-free GPU descriptor.
fn historical_gdn_implementation(
    profile: crate::qwen35::plan::QwenSourceProfile,
    input_tokens: usize,
    is_decode: bool,
) -> Result<Option<HistoricalGdnImplementation>> {
    use crate::qwen35::plan::QwenSourceProfile::{Qwen2B, Qwen08};
    use GdnKktKernel::{Bk32W4S3, Bk32W8S3, Bk64W8S4};
    use GdnOutputKernel::{Bk32Bv32W4S2, Bk32Bv32W4S3};
    use GdnRecomputeKernel::{Bk64Bv64W8S3, Bk64Bv64W8S4};
    use GdnStateKernel::{Bv32W4S2, Bv64W4S2};

    let (recipe_sha256, stages) = match (profile, input_tokens, is_decode) {
        (Qwen08, 47, false) => (
            "b24e5fb2ac4b0e591cb907e8226cd1d80bc24507c0586ecfb5238ea4f3c91f8b",
            Some((Bk32W8S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S2)),
        ),
        (Qwen08, 57, false) => (
            "9ec9c52e82e99b9654d3136f22e10523e1fa6f786a6136f7adefaaedec2aed43",
            Some((Bk32W8S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S3)),
        ),
        (Qwen08, 62, false) => (
            "f487c0f5bcb5580da48fd33f047c4cf67d4098e496e1040e07a4eeafca0c5a65",
            Some((Bk32W8S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S3)),
        ),
        (Qwen08, 67, false) => (
            "213a13cfb815791936bca556a1ab66730b03ac71fe9c72f6586cc5fcab0be3e5",
            Some((Bk64W8S4, Bk64Bv64W8S3, Bv32W4S2, Bk32Bv32W4S3)),
        ),
        (Qwen08, 72, false) => (
            "d2226726eda06c919709e3b7767d019e0dd33f633db925ce266a9f67b48f1d9e",
            Some((Bk32W8S3, Bk64Bv64W8S4, Bv32W4S2, Bk32Bv32W4S2)),
        ),
        (Qwen08, 1, true) => (
            "cf3ef9d3fa84410540c6d969212a3e29c912255d9b404d2430905155c1cf201e",
            None,
        ),
        (Qwen2B, 47, false) => (
            "aaae809312359b22c8404b6ea5ba18073738db731e3df64192372c7f40d82cb6",
            Some((Bk32W4S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S3)),
        ),
        (Qwen2B, 57, false) => (
            "ab5aef404095ded631a2267be095f51b8a4d239a03289d354f3c7b26a1991741",
            Some((Bk32W8S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S2)),
        ),
        (Qwen2B, 62, false) => (
            "b95983abb04db60da606ffe7ecccc547062cf714dc8b11361cd829be41fd452b",
            Some((Bk32W8S3, Bk64Bv64W8S4, Bv32W4S2, Bk32Bv32W4S2)),
        ),
        (Qwen2B, 67, false) => (
            "484de0c7dd651e8646758103db6634e8c34750a44c0d03f4d291deb7c93191df",
            Some((Bk32W8S3, Bk64Bv64W8S4, Bv32W4S2, Bk32Bv32W4S2)),
        ),
        (Qwen2B, 72, false) => (
            "f79c878120e21e36ed8413c5e24f73f159eb4e8a45aaba90e2c4a120cdb30df1",
            Some((Bk32W4S3, Bk64Bv64W8S4, Bv32W4S2, Bk32Bv32W4S3)),
        ),
        (Qwen2B, 1, true) => (
            "18faad78538d9d190c5a31668710f7b3aced6e2dd5d922618108b4044b3355d8",
            None,
        ),
        _ => return Ok(None),
    };
    let config = stages
        .map(|(kkt, recompute, state, output)| {
            GdnKernelConfig::from_stages(kkt, recompute, state, output)
        })
        .transpose()?;
    Ok(Some(HistoricalGdnImplementation::authenticated(
        recipe_sha256,
        input_tokens,
        is_decode,
        config,
    )?))
}

pub(super) fn gdn_implementation(
    profile: crate::qwen35::plan::QwenSourceProfile,
    input_tokens: usize,
    is_decode: bool,
) -> Result<Option<TextGdnImplementation>> {
    historical_gdn_implementation(profile, input_tokens, is_decode)?
        .map(HistoricalGdnImplementation::descriptor)
        .transpose()
}

pub(super) fn fresh_prefill_gdn_implementation(
    profile: crate::qwen35::plan::QwenSourceProfile,
    input_tokens: usize,
    is_decode: bool,
) -> Result<Option<TextGdnImplementation>> {
    use crate::qwen35::plan::QwenSourceProfile::Qwen08;
    use GdnKktKernel::{Bk32W4S3, Bk32W8S3};
    use GdnOutputKernel::{Bk32Bv32W4S2, Bk32Bv32W4S3, Bk32Bv64W4S3};
    use GdnRecomputeKernel::{Bk64Bv64W8S3, Bk64Bv64W8S4};
    use GdnStateKernel::Bv64W4S2;

    if profile != Qwen08 || is_decode {
        return Ok(None);
    }
    let stages = match input_tokens {
        // Preserve the historical 47 configuration exactly.
        47 => (Bk32W8S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S2),
        48 => (Bk32W8S3, Bk64Bv64W8S3, Bv64W4S2, Bk32Bv64W4S3),
        49 => (Bk32W4S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S3),
        50 => (Bk32W4S3, Bk64Bv64W8S3, Bv64W4S2, Bk32Bv32W4S3),
        _ => return Ok(None),
    };
    let config = GdnKernelConfig::from_stages(stages.0, stages.1, stages.2, stages.3)?;
    Ok(Some(TextGdnImplementation::prefill(input_tokens, config)?))
}

#[cfg(test)]
pub(super) fn gdn_recipe_sha256(
    profile: crate::qwen35::plan::QwenSourceProfile,
    input_tokens: usize,
    is_decode: bool,
) -> Result<Option<&'static str>> {
    Ok(
        historical_gdn_implementation(profile, input_tokens, is_decode)?
            .map(|implementation| implementation.recipe_sha256),
    )
}

/// Seal exactly the descriptor selected by the finite qualification table.
/// This comparison runs before source-record or workspace preparation.
pub(super) fn seal_gdn_implementation(
    profile: crate::qwen35::plan::QwenSourceProfile,
    input_tokens: usize,
    is_decode: bool,
    candidate: Option<TextGdnImplementation>,
) -> Result<Option<TextGdnImplementation>> {
    let expected = gdn_implementation(profile, input_tokens, is_decode)?;
    if candidate != expected {
        return Err(Error::Unsupported(
            "qualification GDN descriptor mismatch".into(),
        ));
    }
    Ok(candidate)
}

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
