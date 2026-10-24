//! Analytic F32 sparse primitive over the existing projection/GELU kernels.
use super::*;
pub(super) fn evaluate(
    math: &ReferenceTextMath<'_>,
    c: &TextMoeContract,
    input: TextMoeInputs<'_, ReferenceTextValue>,
) -> Result<ReferenceTextValue> {
    let normalized = math.input(input.normalized)?;
    let router = math.input(input.router_input)?;
    let h = normalized.len();
    let experts = c.gate_up.experts;
    let width = c.down.columns;
    c.validate(
        math.source.source_identity(),
        h,
        input.router_scale,
        input.expert_scale,
    )?;
    if router.len() != h {
        return Err(Error::Malformed("analytic router full hidden input".into()));
    }
    math.produce(h, false, || {
        let scratch = experts
            .checked_mul(64)
            .and_then(|n| n.checked_add(h * 24))
            .and_then(|n| n.checked_add(width * 20 + 1024))
            .ok_or_else(|| Error::Limit("analytic sparse scratch".into()))?;
        let _scratch = math.source.reserve_output(scratch)?;
        let unit = vec![1.0; h];
        let mut routing = crate::math::rms_norm(router, &unit, c.epsilon)?;
        let root = (h as f32).sqrt().recip();
        for (x, scale) in routing.iter_mut().zip(input.router_scale.values) {
            *x = (*x * scale) * root;
        }
        let logits = math.source.project(&c.router, &routing, math.cancelled)?;
        if logits.len() != experts {
            return Err(Error::State("analytic router whole expert logits".into()));
        }
        let probabilities = crate::math::softmax(&logits)?;
        let selected = crate::math::top_k(&probabilities, c.selected)?;
        // Ranked selected weights, then ascending expert dispatch as the HF source.
        let sum = selected.iter().map(|i| probabilities[*i]).sum::<f32>();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(Error::Malformed(
                "analytic sparse selected probability sum".into(),
            ));
        }
        let mut output = vec![0.0; h];
        let mut dispatch = selected;
        dispatch.sort_unstable();
        for expert in dispatch {
            math.check()?;
            let gate_up =
                math.source
                    .project_expert(&c.gate_up, expert, normalized, math.cancelled)?;
            if gate_up.len() != 2 * width || gate_up.iter().any(|v| !v.is_finite()) {
                return Err(Error::Malformed("analytic expert gate/up output".into()));
            }
            let activation = (0..width)
                .map(|i| crate::math::gelu_tanh(gate_up[i]) * gate_up[width + i])
                .collect::<Vec<_>>();
            let down = math
                .source
                .project_expert(&c.down, expert, &activation, math.cancelled)?;
            if down.len() != h || down.iter().any(|v| !v.is_finite()) {
                return Err(Error::Malformed("analytic expert down output".into()));
            }
            let probability = (probabilities[expert] / sum) * input.expert_scale.values[expert];
            for (dst, value) in output.iter_mut().zip(down) {
                *dst += value * probability;
            }
        }
        math.check()?;
        Ok(output)
    })
}
