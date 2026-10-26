//! Original position-source policy, moved from GPU require_position_source.
use super::*;
use crate::config::{Family, LayerKind};
const CONFIG_SHA: &str = "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204";
pub(super) fn require_reference_positions(source: &OriginalWindowModelSource) -> Result<()> {
    let config = source.owner().config();
    let g = config.geometry();
    // Config hash checks the exact original mrope sections/interleaving absent
    // from the generic geometry object; it never substitutes for source issuer.
    if source.owner().config_sha256() != CONFIG_SHA
        || g.family != Family::QwenHybrid
        || g.hidden_size != 1024
        || g.head_dim != 256
        || g.vocab_size != 248320
        || g.max_position_embeddings != 262144
        || g.num_attention_heads != 8
        || g.num_key_value_heads != 2
        || g.layers.len() != 24
        || config.rotary_dimensions() != 64
        || config.rope_theta() != 10_000_000.0
        || g.layers.iter().enumerate().any(|(index, kind)| {
            *kind
                != if index % 4 == 3 {
                    LayerKind::FullAttention
                } else {
                    LayerKind::LinearAttention
                }
        })
    {
        return Err(Error::Unsupported(
            "closed original .8B text position source geometry".into(),
        ));
    }
    Ok(())
}
