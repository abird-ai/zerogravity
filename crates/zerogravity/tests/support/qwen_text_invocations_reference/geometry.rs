//! Reference extents checked against authenticated source config, not record counts.
use super::*;
use zerogravity::qwen35::config::LayerKind;
use zerogravity::qwen35::source::safetensors::QwenTextConfig;

impl Expected {
    pub(super) fn require_geometry(&self, config: &QwenTextConfig) -> Result<()> {
        let g = config.geometry();
        let linear = g
            .layers
            .iter()
            .filter(|k| **k == LayerKind::LinearAttention)
            .count();
        let attention = g
            .layers
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count();
        need(
            g.layers.len() == self.layers
                && linear + attention == self.layers
                && 2 * linear + attention == self.domains
                && 2 * (linear + attention) == self.planes,
            "reference domains/planes match actual source layers",
        )?;
        for record in &self.doc.records {
            let (shape, dtype) = record_geometry(config, record)?;
            need(
                record.shape == shape && record.dtype == dtype,
                "reference shape/dtype matches authenticated source geometry",
            )?;
        }
        Ok(())
    }
}

fn record_geometry(config: &QwenTextConfig, r: &Record) -> Result<(Vec<usize>, &'static str)> {
    let g = config.geometry();
    let size =
        |n: u64| usize::try_from(n).map_err(|_| Error::Limit("reference source dimension".into()));
    let layer = || {
        r.layer
            .and_then(|i| g.layers.get(i))
            .ok_or_else(|| Error::Malformed("reference source layer".into()))
    };
    let no_layer = || need(r.layer.is_none(), "reference non-layer producer");
    let shape = match r.label.as_str() {
        "positions" => {
            no_layer()?;
            return Ok((vec![3, 1, r.rows], "I64"));
        }
        "embedding" | "normalized" => {
            no_layer()?;
            vec![1, r.rows, size(g.hidden_size)?]
        }
        "layer_output" => {
            layer()?;
            vec![1, r.rows, size(g.hidden_size)?]
        }
        "logits" => {
            no_layer()?;
            vec![1, 1, size(g.vocab_size)?]
        }
        label => {
            let (moment, plane) = label
                .split_once('.')
                .ok_or_else(|| Error::Malformed("reference producer label".into()))?;
            let active = state_frontier(moment, r.position, r.rows)?;
            match (layer()?, plane) {
                (LayerKind::LinearAttention, "conv4") => {
                    let keys = g
                        .recurrent_key_heads
                        .ok_or_else(|| Error::Malformed("source key heads".into()))?;
                    let values = g
                        .recurrent_value_heads
                        .ok_or_else(|| Error::Malformed("source value heads".into()))?;
                    let channels = keys
                        .checked_mul(config.linear_key_head_dim())
                        .and_then(|n| n.checked_mul(2))
                        .and_then(|n| {
                            values
                                .checked_mul(config.linear_value_head_dim())
                                .and_then(|v| n.checked_add(v))
                        })
                        .ok_or_else(|| Error::Limit("reference convolution channels".into()))?;
                    vec![1, size(channels)?, size(config.linear_conv_kernel_dim())?]
                }
                (LayerKind::LinearAttention, "recurrent") => {
                    let heads = g
                        .recurrent_value_heads
                        .ok_or_else(|| Error::Malformed("source value heads".into()))?;
                    return Ok((
                        vec![
                            1,
                            size(heads)?,
                            size(config.linear_key_head_dim())?,
                            size(config.linear_value_head_dim())?,
                        ],
                        "F32",
                    ));
                }
                (LayerKind::FullAttention, "key" | "value") => {
                    vec![1, size(g.num_key_value_heads)?, active, size(g.head_dim)?]
                }
                _ => {
                    return Err(Error::Malformed(
                        "reference plane differs from source layer kind".into(),
                    ));
                }
            }
        }
    };
    Ok((shape, "BF16"))
}

fn state_frontier(moment: &str, position: usize, rows: usize) -> Result<usize> {
    match moment {
        "before" if position > 0 => Ok(position),
        "after" if rows > 0 => position
            .checked_add(rows)
            .ok_or_else(|| Error::Limit("reference state frontier".into())),
        _ => Err(Error::Malformed(
            "reference state moment/initialization".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn state_extents_follow_call_frontiers_not_case_ordinals() {
        assert_eq!(state_frontier("after", 0, 47).unwrap(), 47);
        assert_eq!(state_frontier("before", 495, 17).unwrap(), 495);
        assert_eq!(state_frontier("after", 495, 17).unwrap(), 512);
        assert_eq!(state_frontier("after", 512, 1).unwrap(), 513);
    }
    #[test]
    fn initial_prior_missing_rows_unknown_moments_and_overflow_refuse() {
        for (moment, position, rows) in [
            ("before", 0, 47),
            ("after", 0, 0),
            ("unknown", 1, 1),
            ("after", usize::MAX, 1),
        ] {
            assert!(state_frontier(moment, position, rows).is_err());
        }
    }
}
