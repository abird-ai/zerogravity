//! Family source eligibility for explicit backend qualification inputs.
use super::*;
use crate::config::{Family, LayerKind};
const CONFIG_SHA: &str = "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204";

/// Retains the original family owner and its actual authenticated metadata.
/// Descriptor transfer adds no device, model or publication authority.
pub struct PreparedQualificationSource {
    model: Arc<OriginalWindowModelSource>,
    _bound: Option<BoundFullModelSource>,
}
impl PreparedQualificationSource {
    pub fn source(&self) -> &VerifiedTensorSource {
        self.model.source()
    }
    pub fn owner(&self) -> &Arc<OriginalWindowModelSource> {
        &self.model
    }
}
impl BoundFullModelSource {
    /// Consume the existing authenticated metadata owner; no duplicate admission.
    pub fn prepare_qualification_source(
        self,
        profile: QwenSourceProfile,
    ) -> Result<PreparedQualificationSource> {
        if self.model_profile() != profile {
            return Err(Error::Unsupported(
                "closed original qualification source/profile".into(),
            ));
        }
        Ok(PreparedQualificationSource {
            model: self.source().clone(),
            _bound: Some(self),
        })
    }
}
impl OriginalWindowModelSource {
    pub fn prepare_qualification_source(
        self: &Arc<Self>,
        profile: QwenSourceProfile,
    ) -> Result<PreparedQualificationSource> {
        let bound = self.bind_full_model_source()?;
        if bound.model_profile() != profile || self.source().staging_chunk_bytes() != 1024 * 1024 {
            return Err(Error::Unsupported(
                "closed original source/profile/1MiB qualification".into(),
            ));
        }
        Ok(PreparedQualificationSource {
            model: self.clone(),
            _bound: Some(bound),
        })
    }
    pub fn prepare_rotary_source(
        self: &Arc<Self>,
        text: bool,
    ) -> Result<PreparedQualificationSource> {
        require_rotary_source(self, text)?;
        // The original text position path authenticates the complete source;
        // fixed prompt positions retain their original exact config-only scope.
        Ok(PreparedQualificationSource {
            model: self.clone(),
            _bound: None,
        })
    }
}
fn require_fixed_rotary_source(source: &OriginalWindowModelSource) -> Result<()> {
    let c = source.owner().config();
    let g = c.geometry();
    if source.owner().config_sha256() != CONFIG_SHA
        || g.family != Family::QwenHybrid
        || g.hidden_size != 1024
        || g.head_dim != 256
        || g.vocab_size != 248320
        || g.max_position_embeddings != 262144
        || g.num_attention_heads != 8
        || g.num_key_value_heads != 2
        || g.layers.len() != 24
        || c.rotary_dimensions() != 64
        || c.rope_theta() != 10_000_000.0
        || g.layers.iter().enumerate().any(|(i, k)| {
            *k != if i % 4 == 3 {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            }
        })
    {
        return Err(Error::Unsupported(
            "closed original .8B unscaled MRotary source/config".into(),
        ));
    }
    Ok(())
}
fn require_rotary_source(source: &Arc<OriginalWindowModelSource>, text: bool) -> Result<()> {
    if !text {
        return require_fixed_rotary_source(source);
    }
    // The existing original-source binder authenticates each approved model's
    // config/index/shards/roles; geometry alone never issues a source identity.
    let bound = source.bind_full_model_source()?;
    let c = source.owner().config();
    let g = c.geometry();
    if !matches!(
        bound.model_profile(),
        crate::plan::QwenSourceProfile::Qwen08 | crate::plan::QwenSourceProfile::Qwen2B
    ) || g.family != Family::QwenHybrid
        || g.head_dim != 256
        || g.vocab_size != 248320
        || g.max_position_embeddings != 262144
        || g.num_attention_heads != 8
        || g.num_key_value_heads != 2
        || g.layers.len() != 24
        || c.rotary_dimensions() != 64
        || c.rope_theta() != 10_000_000.0
        || g.layers.iter().enumerate().any(|(i, k)| {
            *k != if i % 4 == 3 {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            }
        })
    {
        return Err(Error::Unsupported(
            "text original unscaled MRotary source/config".into(),
        ));
    }
    Ok(())
}
