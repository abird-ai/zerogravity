//! Source-derived ordinary geometry, independent of finite captured recipes.
use super::QwenModelVariant;
use crate::config::{Family, LayerKind};
use crate::source::safetensors::{GdnOutputGate, QwenTextConfig};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrdinaryQwenGeometry {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    query_heads: usize,
    kv_heads: usize,
    key_heads: usize,
    value_heads: usize,
    tied: bool,
}

impl OrdinaryQwenGeometry {
    /// Geometry only. The original source seal separately authenticates all
    /// config/index bytes, roles and storage before retaining this value.
    pub(crate) fn from_config(config: &QwenTextConfig) -> Result<Self> {
        let g = config.geometry();
        if g.family != Family::QwenHybrid
            || g.vocab_size != 248_320
            || g.max_position_embeddings != 262_144
            || g.head_dim != 256
            || g.layers.is_empty()
            || !g.layers.len().is_multiple_of(4)
            || g.layers.iter().enumerate().any(|(i, kind)| {
                *kind
                    != if i % 4 == 3 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
            })
            || config.linear_key_head_dim() != 128
            || config.linear_value_head_dim() != 128
            || config.linear_conv_kernel_dim() != 4
            || config.rms_norm_eps() != 1e-6
            || config.gdn_output_gate() != GdnOutputGate::Silu
            || !config.attention_output_gate()
            || config.rope_theta() != 10_000_000.0
            || config.rotary_dimensions() != 64
        {
            return Err(Error::Unsupported(
                "ordinary Qwen source numerical geometry".into(),
            ));
        }
        let size = |value| {
            usize::try_from(value)
                .map_err(|_| Error::Limit("ordinary Qwen source dimension".into()))
        };
        Ok(Self {
            hidden: size(g.hidden_size)?,
            intermediate: size(config.intermediate_size())?,
            layers: g.layers.len(),
            query_heads: size(g.num_attention_heads)?,
            kv_heads: size(g.num_key_value_heads)?,
            key_heads: size(
                g.recurrent_key_heads
                    .ok_or_else(|| Error::Malformed("ordinary Qwen key heads".into()))?,
            )?,
            value_heads: size(
                g.recurrent_value_heads
                    .ok_or_else(|| Error::Malformed("ordinary Qwen value heads".into()))?,
            )?,
            tied: config.tie_word_embeddings(),
        })
    }
    pub fn hidden_size(self) -> usize {
        self.hidden
    }
    pub fn intermediate_size(self) -> usize {
        self.intermediate
    }
    pub fn layer_count(self) -> usize {
        self.layers
    }
    pub fn vocabulary_size(self) -> usize {
        248_320
    }
    pub fn max_position_embeddings(self) -> usize {
        262_144
    }
    pub fn query_heads(self) -> usize {
        self.query_heads
    }
    pub fn kv_heads(self) -> usize {
        self.kv_heads
    }
    pub fn key_heads(self) -> usize {
        self.key_heads
    }
    pub fn value_heads(self) -> usize {
        self.value_heads
    }
    pub fn query_width(self) -> usize {
        self.query_heads * 256
    }
    pub fn kv_width(self) -> usize {
        self.kv_heads * 256
    }
    pub fn recurrent_width(self) -> usize {
        self.value_heads * 128
    }
    pub fn conv_channels(self) -> usize {
        (2 * self.key_heads + self.value_heads) * 128
    }
    pub fn tied_embeddings(self) -> bool {
        self.tied
    }
    pub fn state_domains(self) -> usize {
        self.layers / 4 * 7
    }
}

impl From<QwenModelVariant> for OrdinaryQwenGeometry {
    fn from(profile: QwenModelVariant) -> Self {
        Self {
            hidden: profile.hidden_size(),
            intermediate: profile.intermediate_size(),
            layers: profile.layer_count(),
            query_heads: 8,
            kv_heads: 2,
            key_heads: 16,
            value_heads: 16,
            tied: true,
        }
    }
}

impl TryFrom<OrdinaryQwenGeometry> for QwenModelVariant {
    type Error = Error;
    fn try_from(geometry: OrdinaryQwenGeometry) -> Result<Self> {
        [Self::Qwen800M, Self::Qwen2B]
            .into_iter()
            .find(|profile| OrdinaryQwenGeometry::from(*profile) == geometry)
            .ok_or_else(|| {
                Error::Unsupported("ordinary geometry has no finite CUDA source profile".into())
            })
    }
}

#[cfg(test)]
pub(super) fn large_fixture() -> OrdinaryQwenGeometry {
    OrdinaryQwenGeometry {
        hidden: 5120,
        intermediate: 17408,
        layers: 64,
        query_heads: 24,
        kv_heads: 4,
        key_heads: 16,
        value_heads: 48,
        tied: false,
    }
}
