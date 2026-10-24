//! Sealed source/config owner for the bounded original .8B GDN prefix.
use std::sync::Arc;
use zerogravity_artifacts::model::qwen::GdnOutputGate;
use zerogravity_artifacts::model::qwen_owned::QwenOriginalWindowText;
use zerogravity_artifacts::model::{Family, LayerKind};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::accelerator::program::{
    GdnPrefixLayer, GdnPrefixRmsBinding, GdnPrefixRmsSite,
};
use zerogravity_core::{Error, Result};

pub(crate) struct BoundRmsSource {
    owner: Arc<QwenOriginalWindowText>,
    key: GdnPrefixRmsBinding,
}
impl BoundRmsSource {
    pub(crate) fn bind(
        owner: Arc<QwenOriginalWindowText>,
        layer: usize,
        site: GdnPrefixRmsSite,
    ) -> Result<Self> {
        let config = owner.config();
        let g = config.geometry();
        if g.family != Family::QwenHybrid
            || g.hidden_size != 1024
            || g.vocab_size != 248320
            || g.layers.len() != 24
            || g.head_dim != 256
            || g.num_attention_heads != 8
            || g.num_key_value_heads != 2
            || g.recurrent_key_heads != Some(16)
            || g.recurrent_value_heads != Some(16)
            || config.linear_key_head_dim() != 128
            || config.linear_value_head_dim() != 128
            || config.linear_conv_kernel_dim() != 4
            || config.intermediate_size() != 3584
            || config.gdn_output_gate() != GdnOutputGate::Silu
            || config.rms_norm_eps() != 1e-6
            || !config.attention_output_gate()
            || g.layers.iter().enumerate().any(|(i, kind)| {
                *kind
                    != if i % 4 == 3 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
            })
            || layer > 2
            || g.layers.get(layer) != Some(&LayerKind::LinearAttention)
        {
            return Err(Error::Unsupported(
                "original .8B24/GDN16x128 prefix config".into(),
            ));
        }
        let key = GdnPrefixLayer::new(owner.source_identity(), layer)?.rms(site)?;
        owner.text_tensor(&key.gamma_name())?;
        let actual = owner.source().parameter(&key.gamma_name())?;
        key.validate(&actual.source_parameter(), 1024)?;
        Ok(Self { owner, key })
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        self.owner.source()
    }
    pub(crate) fn key(&self) -> GdnPrefixRmsBinding {
        self.key
    }
    pub(crate) fn require_source(&self, source: &VerifiedTensorSource) -> Result<()> {
        if !self.source().same_issuer(source)
            || self.key.layer().source() != source.source_identity()
        {
            return Err(Error::State("GDN RMS original config/source issuer".into()));
        }
        let actual = source.parameter(&self.key.gamma_name())?;
        self.key.validate(&actual.source_parameter(), 1024)?;
        Ok(())
    }
}
