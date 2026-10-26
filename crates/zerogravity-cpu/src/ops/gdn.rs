//! Source-order FP32 GDN convolution; caller owns tentative state.
use crate::{Error, Result};

/// Four-tap causal convolution and three-frame history update.
/// Multiplication and left-to-right addition order are part of the contract.
// Previous model helper: source_gdn_conv_step.
pub fn gdn_conv_step(
    activation: &[f32],
    weights: &[f32],
    history: &mut [f32],
    raw_output: &mut [f32],
) -> Result<()> {
    let channels = activation.len();
    if channels == 0
        || channels.checked_mul(3) != Some(history.len())
        || channels.checked_mul(4) != Some(weights.len())
        || raw_output.len() != channels
        || !activation
            .iter()
            .chain(weights)
            .chain(history.iter())
            .all(|v| v.is_finite())
    {
        return Err(Error::Malformed(
            "GDN causal convolution source geometry/nonfinite input".into(),
        ));
    }
    for channel in 0..channels {
        let c = &mut history[channel * 3..channel * 3 + 3];
        let w = &weights[channel * 4..channel * 4 + 4];
        // Preserve original scalar FP32 multiplication and sum order.
        let raw = c[0] * w[0] + c[1] * w[1] + c[2] * w[2] + activation[channel] * w[3];
        raw_output[channel] = raw;
        c[0] = c[1];
        c[1] = c[2];
        c[2] = activation[channel];
    }
    if !raw_output.iter().all(|v| v.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite GDN convolution source output".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn_conv_source_updates_three_frame_history_only_after_valid_inputs() {
        let channels = 32usize;
        let qkv = (0..channels)
            .map(|i| ((i * 13) as f32 - 199.0) / 117.0)
            .collect::<Vec<_>>();
        let coefficients = (0..channels * 4)
            .map(|i| ((i * 37 % 71) as f32 - 35.0) / 509.0)
            .collect::<Vec<_>>();
        let mut history = (0..channels * 3)
            .map(|i| ((i * 19 % 83) as f32 - 41.0) / 239.0)
            .collect::<Vec<_>>();
        let original = history.clone();
        let mut raw = vec![0f32; channels];
        gdn_conv_step(&qkv, &coefficients, &mut history, &mut raw).unwrap();
        for channel in 0..channels {
            let old = &original[channel * 3..channel * 3 + 3];
            let w = &coefficients[channel * 4..channel * 4 + 4];
            let expected = old[0] * w[0] + old[1] * w[1] + old[2] * w[2] + qkv[channel] * w[3];
            assert_eq!(raw[channel].to_bits(), expected.to_bits());
            assert_eq!(
                &history[channel * 3..channel * 3 + 3],
                &[old[1], old[2], qkv[channel]]
            );
        }
        let good = history.clone();
        let mut broken = vec![1.0f32; channels];
        let mut nan = qkv.clone();
        nan[0] = f32::NAN;
        assert!(gdn_conv_step(&nan, &coefficients, &mut history, &mut broken).is_err());
        assert_eq!(good, history, "invalid input mutated private GDN state");
        assert_eq!(broken, vec![1.0f32; channels]);
        assert!(gdn_conv_step(&qkv, &coefficients[..4], &mut history, &mut broken).is_err());
        assert!(gdn_conv_step(&qkv, &coefficients, &mut history[..7], &mut broken).is_err());
    }
}
