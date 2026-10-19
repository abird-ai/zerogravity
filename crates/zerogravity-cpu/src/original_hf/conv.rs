//! Closed pinned Torch CPU BF16 depthwise four-tap convolution window.
//!
//! PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb slow_conv2d NN BF16
//! GEMM, AVX2 build: four independent FMA partials, then three direct adds.
//! This qualifies neither general GEMM nor CUDA/FLA/full-model execution.

use crate::accelerator::SourceIdentity;
use crate::quant::QuantKind;
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

const TAPS: usize = 4;
const HISTORY: usize = TAPS - 1;
const MAX_CHANNELS: usize = 1_048_576 / TAPS;

/// Original immutable rank3 `[channel,1,4]` BF16 weights and exact widened
/// BF16 input/history. History is canonical `[channel,3]`, oldest first.
#[derive(Clone, Copy, Debug)]
pub struct CausalConv4Inputs<'a> {
    pub source: SourceIdentity,
    pub tensor_name: &'a str,
    pub weight: SourceTensorRef<'a>,
    pub activation: &'a [f32],
    pub history: &'a [f32],
}

/// Private results only; the caller retains source/state/publication authority.
#[derive(Debug)]
pub struct CausalConv4Completed {
    pub next_history: Vec<f32>,
    pub raw_fp32: Vec<f32>,
    /// Final convolution BF16 values carried as exactly widened F32 words.
    pub raw_bf16: Vec<f32>,
}

fn validate(input: &CausalConv4Inputs<'_>) -> Result<usize> {
    if !input.source.is_bound() || input.tensor_name.is_empty() {
        return Err(Error::Malformed(
            "pinned BF16 convolution source binding".into(),
        ));
    }
    if input.weight.kind != QuantKind::Bf16 {
        return Err(Error::Unsupported(
            "pinned BF16 convolution requires BF16 source weights".into(),
        ));
    }
    if input.weight.shape.len() != 3 || input.weight.shape[1..] != [1, TAPS as u64] {
        return Err(Error::Malformed(
            "pinned BF16 convolution requires original rank3 four-tap source".into(),
        ));
    }
    let channels = usize::try_from(input.weight.shape[0])
        .ok()
        .filter(|&n| n <= MAX_CHANNELS)
        .ok_or_else(|| Error::Limit("pinned BF16 convolution channel bound".into()))?;
    if channels == 0
        || input.activation.len() != channels
        || channels.checked_mul(HISTORY) != Some(input.history.len())
    {
        return Err(Error::Malformed(
            "pinned BF16 convolution input/history geometry".into(),
        ));
    }
    input.weight.validate()?;
    if input
        .activation
        .iter()
        .chain(input.history)
        .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
        || input.weight.packed_bytes.chunks_exact(2).any(|word| {
            let word = u16::from_le_bytes([word[0], word[1]]);
            word & 0x7f80 == 0x7f80
        })
    {
        return Err(Error::Malformed(
            "pinned convolution requires finite exact BF16 values".into(),
        ));
    }
    Ok(channels)
}

fn round_bf16(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(Error::Malformed(
            "nonfinite pinned BF16 convolution output".into(),
        ));
    }
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    let rounded = f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000);
    if !rounded.is_finite() {
        return Err(Error::Limit(
            "pinned convolution BF16 output overflow".into(),
        ));
    }
    Ok(rounded)
}

/// One finite BF16 four-tap window, including the newest cached-token window.
/// The pinned compiled generic BF16 kernel contracts each partial's first
/// product/add into FMA. Its direct partial reduction has no new +0 seed;
/// replacing it with the projection scalar tail changes underflow zero signs.
/// SiLU, source-cache transport and StateRoot publication remain caller-owned.
pub fn causal_step_bf16_4(input: CausalConv4Inputs<'_>) -> Result<CausalConv4Completed> {
    let channels = validate(&input)?;
    let mut next_history = Vec::with_capacity(channels * HISTORY);
    let mut raw_fp32 = Vec::with_capacity(channels);
    let mut raw_bf16 = Vec::with_capacity(channels);
    for (channel, packed) in input.weight.packed_bytes.chunks_exact(TAPS * 2).enumerate() {
        let history = &input.history[channel * HISTORY..(channel + 1) * HISTORY];
        let window = [
            history[0],
            history[1],
            history[2],
            input.activation[channel],
        ];
        let mut partials = [0.0f32; TAPS];
        for ((sum, &activation), weight) in
            partials.iter_mut().zip(&window).zip(packed.chunks_exact(2))
        {
            let word = u16::from_le_bytes([weight[0], weight[1]]);
            let weight = f32::from_bits(u32::from(word) << 16);
            *sum = weight.mul_add(activation, *sum);
        }
        let raw = ((partials[0] + partials[1]) + partials[2]) + partials[3];
        raw_bf16.push(round_bf16(raw)?);
        raw_fp32.push(raw);
        next_history.extend([history[1], history[2], input.activation[channel]]);
    }
    Ok(CausalConv4Completed {
        next_history,
        raw_fp32,
        raw_bf16,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAME: &str = "model.language_model.layers.0.linear_attn.conv1d.weight";
    type Golden = ([u16; TAPS], [u16; TAPS], u16);

    fn source() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    fn widen(word: u16) -> f32 {
        f32::from_bits(u32::from(word) << 16)
    }
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }
    fn run(cases: &[Golden]) -> CausalConv4Completed {
        let weights = cases
            .iter()
            .flat_map(|(_, weights, _)| weights.iter().flat_map(|v| v.to_le_bytes()))
            .collect::<Vec<_>>();
        let history = cases
            .iter()
            .flat_map(|(input, _, _)| input[..HISTORY].iter().map(|&v| widen(v)))
            .collect::<Vec<_>>();
        let activation = cases
            .iter()
            .map(|(input, _, _)| widen(input[3]))
            .collect::<Vec<_>>();
        let output = causal_step_bf16_4(CausalConv4Inputs {
            source: source(),
            tensor_name: NAME,
            weight: SourceTensorRef {
                kind: QuantKind::Bf16,
                shape: &[u64::try_from(cases.len()).unwrap(), 1, 4],
                packed_bytes: &weights,
            },
            activation: &activation,
            history: &history,
        })
        .unwrap();
        assert_eq!(
            bits(&output.raw_bf16),
            cases
                .iter()
                .map(|(_, _, v)| u32::from(*v) << 16)
                .collect::<Vec<_>>()
        );
        output
    }

    #[test]
    fn compact_original_preactivations_preserve_the_observed_underflow_case() {
        // Original .8B revision2fc0636..., token11, channels
        // 0/127/512/1332/2222/2527/4096/6143. Untouched pinned Torch conv output.
        // Archive SHA256 2c13b27d9477a9e05e9e5860d75a9e1b1697a6015dbba542
        // b92d0ef614121b83; small selected BF16 words, no model fixture needed.
        let output = run(&[
            ([0, 0, 0, 0xbe85], [0xb929, 0x39d4, 0xbb5e, 0xbd98], 0x3c9e),
            ([0, 0, 0, 0xbceb], [0xba3a, 0xba77, 0x3ad5, 0x3da7], 0xbb19),
            ([0, 0, 0, 0xbe9d], [0xbb98, 0xbc22, 0xbc7b, 0x3dd1], 0xbd00),
            ([0, 0, 0, 0xbed1], [0xbc11, 0xbc5e, 0xbd2d, 0x3d4b], 0xbca6),
            ([0, 0, 0, 0x3cd3], [0xbc28, 0xbd5f, 0xbe8a, 0xbc3c], 0xb99b),
            ([0, 0, 0, 0x8315], [0x8220, 0x8220, 0x8220, 0x0220], 0),
            ([0, 0, 0, 0x3d7b], [0x388e, 0xba0c, 0x3db7, 0xbb8c], 0xb989),
            ([0, 0, 0, 0x3f3d], [0xb988, 0xbace, 0xbb58, 0xbd0e], 0xbcd2),
        ]);
        assert_eq!(output.raw_fp32[5].to_bits(), 0);
    }

    #[test]
    fn actual_torch_synthetic_outputs_cover_fma_zero_sign_cancellation_and_bf16_ties() {
        // Actual original causal_conv1d_fn on source-free BF16 inputs, same
        // pinned CPU build. These include the counterexample to scalar-tail reuse.
        let output = run(&[
            ([0x8000; 4], [0x3f80; 4], 0),
            ([0x4b80, 0x3f80, 0xcb80, 0x3f80], [0x3f80; 4], 0x3f80),
            ([0x3f80, 0x3b80, 0, 0], [0x3f80; 4], 0x3f80),
            ([0x3f80, 0x3c40, 0, 0], [0x3f80; 4], 0x3f82),
            (
                [0x3fc0, 0xc000, 0x4080, 0x3e80],
                [0x4000, 0xbf00, 0x3e00, 0xc080],
                0x4060,
            ),
            ([0x0080; 4], [0x0080; 4], 0),
            ([0x8080; 4], [0x0080; 4], 0x8000),
        ]);
        assert_eq!(
            bits(&output.raw_fp32),
            [
                0,
                0x3f80_0000,
                0x3f80_8000,
                0x3f81_8000,
                0x4060_0000,
                0,
                0x8000_0000
            ]
        );
        let tiny = widen(0x8080);
        let weight = widen(0x0080);
        let mut separate = 0.0f32;
        for activation in [tiny; 4] {
            let product = activation * weight;
            separate += product;
        }
        assert_eq!(separate.to_bits(), 0);
        assert_eq!(output.raw_fp32[6].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn canonical_next_three_are_bit_transport_and_inputs_stay_private() {
        let weight = [0x3f80u16.to_le_bytes(); 8].concat();
        let history = [0.0, -0.0, 2.0, 4.0, 0.0, -0.0];
        let activation = [1.0, -0.0];
        let original = bits(&history);
        let output = causal_step_bf16_4(CausalConv4Inputs {
            source: source(),
            tensor_name: NAME,
            weight: SourceTensorRef {
                kind: QuantKind::Bf16,
                shape: &[2, 1, 4],
                packed_bytes: &weight,
            },
            activation: &activation,
            history: &history,
        })
        .unwrap();
        assert_eq!(bits(&history), original);
        assert_eq!(
            bits(&output.next_history),
            bits(&[-0.0, 2.0, 1.0, 0.0, -0.0, -0.0])
        );
        assert_eq!(output.raw_fp32, [3.0, 4.0]);
    }

    #[test]
    fn invalid_source_rank_bytes_dtype_and_bf16_bindings_refuse_before_math() {
        let packed = [0x3f80u16.to_le_bytes(); 4].concat();
        let input = CausalConv4Inputs {
            source: source(),
            tensor_name: NAME,
            weight: SourceTensorRef {
                kind: QuantKind::Bf16,
                shape: &[1, 1, 4],
                packed_bytes: &packed,
            },
            activation: &[1.0],
            history: &[0.0; 3],
        };
        assert!(
            causal_step_bf16_4(CausalConv4Inputs {
                source: SourceIdentity::unbound(),
                ..input
            })
            .is_err()
        );
        assert!(
            causal_step_bf16_4(CausalConv4Inputs {
                tensor_name: "",
                ..input
            })
            .is_err()
        );
        for shape in [&[1, 4][..], &[1, 2, 4], &[1, 1, 3], &[0, 1, 4]] {
            assert!(
                causal_step_bf16_4(CausalConv4Inputs {
                    weight: SourceTensorRef {
                        shape,
                        ..input.weight
                    },
                    ..input
                })
                .is_err()
            );
        }
        assert!(matches!(
            causal_step_bf16_4(CausalConv4Inputs {
                weight: SourceTensorRef {
                    kind: QuantKind::F16,
                    ..input.weight
                },
                ..input
            }),
            Err(Error::Unsupported(_))
        ));
        assert!(
            causal_step_bf16_4(CausalConv4Inputs {
                weight: SourceTensorRef {
                    packed_bytes: &packed[..7],
                    ..input.weight
                },
                ..input
            })
            .is_err()
        );
        for activation in [&[1.0, 2.0][..], &[f32::NAN], &[f32::from_bits(0x3f80_0001)]] {
            assert!(
                causal_step_bf16_4(CausalConv4Inputs {
                    activation,
                    ..input
                })
                .is_err()
            );
        }
        for history in [
            &[0.0; 2][..],
            &[f32::INFINITY, 0.0, 0.0],
            &[f32::from_bits(1), 0.0, 0.0],
        ] {
            assert!(causal_step_bf16_4(CausalConv4Inputs { history, ..input }).is_err());
        }
        for word in [0x7f80u16, 0x7fc0] {
            let packed = [word.to_le_bytes(); 4].concat();
            assert!(
                causal_step_bf16_4(CausalConv4Inputs {
                    weight: SourceTensorRef {
                        packed_bytes: &packed,
                        ..input.weight
                    },
                    ..input
                })
                .is_err()
            );
        }
        assert!(matches!(
            causal_step_bf16_4(CausalConv4Inputs {
                weight: SourceTensorRef {
                    shape: &[(MAX_CHANNELS + 1) as u64, 1, 4],
                    packed_bytes: &[],
                    ..input.weight
                },
                ..input
            }),
            Err(Error::Limit(_))
        ));
    }

    #[test]
    fn fp32_or_bf16_overflow_returns_no_proposed_state() {
        for (activation, weights) in [
            ([widen(0x7f7f); 4], [0x7f7fu16; 4]),
            ([1.0; 4], [0x7f7f, 0x7b00, 0, 0]),
        ] {
            let packed = weights
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let history = &activation[..3];
            let before = bits(history);
            assert!(
                causal_step_bf16_4(CausalConv4Inputs {
                    source: source(),
                    tensor_name: NAME,
                    weight: SourceTensorRef {
                        kind: QuantKind::Bf16,
                        shape: &[1, 1, 4],
                        packed_bytes: &packed
                    },
                    activation: &activation[3..],
                    history,
                })
                .is_err()
            );
            assert_eq!(bits(history), before);
        }
    }
}
