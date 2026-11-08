//! Source-owned geometry seal. No native/model/session/publication permit.
use super::*;
use crate::plan::{QwenGdnCall, QwenPlan, QwenSourceCall, QwenSourceProfile};

mod prompt;
pub use prompt::BoundPromptCall;

pub(super) const CONFIG2B: &str =
    "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4";
const INDEX2B: &str = "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9";
const BYTES2B: u64 = 4_548_221_488;
const MIB: usize = 1024 * 1024;

// Only bind_full_model_source constructs this after the complete role/profile
// join. Owned config/header/role metadata is immutable; actual window reads
// retain their independent source checks and admission.
struct SourceOwner {
    model: Arc<OriginalWindowModelSource>,
    profile: QwenSourceProfile,
    _metadata: WindowWorkspace,
    prompt: std::sync::Mutex<std::collections::VecDeque<prompt::PreparedRecord>>,
}
#[derive(Clone)]
pub struct BoundFullModelSource(Arc<SourceOwner>);
impl BoundFullModelSource {
    pub fn model_profile(&self) -> QwenSourceProfile {
        self.0.profile
    }
    pub fn source(&self) -> &Arc<OriginalWindowModelSource> {
        &self.0.model
    }
    /// Metadata recording only. Actual lowerers must bind this original owner.
    pub fn record(&self, request: QwenGdnCall) -> Result<QwenPlan> {
        if request.layer().source() != self.0.model.source_identity() {
            return Err(Error::State(
                "full-model sealed source/request issuer".into(),
            ));
        }
        self.0
            .model
            .record_full_model_inner(QwenSourceCall::for_profile(self.0.profile, request)?)
    }
}

impl OriginalWindowModelSource {
    pub fn bind_full_model_source(self: &Arc<Self>) -> Result<BoundFullModelSource> {
        let profile = if self.owner().config_sha256() == CONFIG2B {
            QwenSourceProfile::Qwen2B
        } else {
            QwenSourceProfile::Qwen08
        };
        require_source(self, profile)?;
        // Source descriptors/roles were validated before this admission.
        let metadata = self.source().reserve_workspace(64 * 1024)?;
        Ok(BoundFullModelSource(Arc::new(SourceOwner {
            model: self.clone(),
            profile,
            _metadata: metadata,
            prompt: std::sync::Mutex::new(std::collections::VecDeque::new()),
        })))
    }
}

fn require_2b_pins(
    config: &str,
    index: &str,
    bytes: u64,
    chunk: usize,
    text: usize,
    other: usize,
) -> Result<()> {
    if config != CONFIG2B
        || index != INDEX2B
        || bytes != BYTES2B
        || chunk != MIB
        || text != 320
        || other != 312
    {
        return Err(Error::Unsupported(
            "closed original2B full-model source/config/index/extent/1MiB".into(),
        ));
    }
    Ok(())
}

pub(super) fn require_source(
    model: &OriginalWindowModelSource,
    profile: QwenSourceProfile,
) -> Result<()> {
    if profile == QwenSourceProfile::Qwen08 {
        model.require_small_model_source()?;
    } else {
        require_2b_pins(
            model.owner().config_sha256(),
            model.owner().verified_index_sha256(),
            model.source().source_bytes(),
            model.source().staging_chunk_bytes(),
            model.owner().text_tensor_count(),
            model.owner().other_tensor_count(),
        )?;
    }
    super::source_geometry::require_roles(
        model,
        profile.into(),
        zerogravity_core::encoding::ScalarDType::F32,
    )
}

#[cfg(test)]
#[path = "full_model_source/tests.rs"]
mod tests;
