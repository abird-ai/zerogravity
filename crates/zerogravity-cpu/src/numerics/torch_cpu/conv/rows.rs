//! Source-call Conv4 with complete channel-major four-word cache.
//! Arithmetic reuses the original window primitive; private outputs only.
use super::*;

pub struct Inputs<'a> {
    pub source: SourceIdentity,
    pub tensor_name: &'a str,
    pub weight: SourceTensorRef<'a>,
    /// Token-major `[row,channel]` exact BF16 activations.
    pub activation: &'a [f32],
    pub rows: usize,
    /// Original complete `[channel,4]` cache, or genuinely absent initial state.
    pub previous: Option<&'a [f32]>,
}
pub struct Completed {
    pub raw_fp32: Vec<f32>,
    pub raw_bf16: Vec<f32>,
    pub state: Vec<f32>,
}

pub fn causal_bf16(input: Inputs<'_>) -> Result<Completed> {
    let Inputs {
        source,
        tensor_name,
        weight,
        activation,
        rows,
        previous,
    } = input;
    if !source.is_bound() || tensor_name.is_empty() {
        return Err(Error::Malformed("source Conv4 call source binding".into()));
    }
    weight.validate()?;
    if !(1..=64).contains(&rows) || weight.shape.len() != 3 || weight.shape[1..] != [1, 4] {
        return Err(Error::Malformed(
            "source Conv4 call row/weight geometry".into(),
        ));
    }
    let channels = usize::try_from(weight.shape[0])
        .ok()
        .filter(|n| *n > 0 && *n <= MAX_CHANNELS)
        .ok_or_else(|| Error::Limit("source Conv4 call channels".into()))?;
    if activation.len() != rows * channels
        || previous.is_some_and(|v| v.len() != channels * 4)
        || activation
            .iter()
            .chain(previous.into_iter().flatten())
            .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
    {
        return Err(Error::Malformed(
            "source Conv4 call complete BF16 input/cache".into(),
        ));
    }
    let mut history = vec![0.0; channels * 3];
    if let Some(previous) = previous {
        for (target, old) in history.chunks_exact_mut(3).zip(previous.chunks_exact(4)) {
            target.copy_from_slice(&old[1..]);
        }
    }
    // Authenticate source/weight/profile before collecting any call result.
    validate(&CausalConv4Inputs {
        source,
        tensor_name,
        weight,
        activation: &activation[..channels],
        history: &history,
    })?;
    let mut raw_fp32 = Vec::with_capacity(activation.len());
    let mut raw_bf16 = Vec::with_capacity(activation.len());
    for row in activation.chunks_exact(channels) {
        let output = causal_step_bf16_4(CausalConv4Inputs {
            source,
            tensor_name,
            weight,
            activation: row,
            history: &history,
        })?;
        history = output.next_history;
        raw_fp32.extend(output.raw_fp32);
        raw_bf16.extend(output.raw_bf16);
    }
    let mut state = vec![0.0; channels * 4];
    for (channel, target) in state.chunks_exact_mut(4).enumerate() {
        for (slot, value) in target.iter_mut().enumerate() {
            let time = rows + slot;
            *value = if time >= 4 {
                activation[(time - 4) * channels + channel]
            } else {
                previous.map_or(0.0, |old| old[channel * 4 + time])
            };
        }
    }
    Ok(Completed {
        raw_fp32,
        raw_bf16,
        state,
    })
}
