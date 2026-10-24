//! Explicit bounded eager CPU attention; no model/native/publication permission.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorchCpuAttentionGeometry {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub width: usize,
    pub rotary: usize,
    pub theta: f32,
}
impl TorchCpuAttentionGeometry {
    pub fn validate(self, position: usize) -> Result<(usize, usize, usize)> {
        if !(1..=24).contains(&self.query_heads)
            || !(1..=4).contains(&self.kv_heads)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.width != 256
            || self.rotary != 64
            || self.theta.to_bits() != 10_000_000f32.to_bits()
            || position >= 64
        {
            return Err(Error::Unsupported(
                "bounded CPU eager attention geometry/position".into(),
            ));
        }
        Ok((
            self.query_heads * self.width,
            self.kv_heads * self.width,
            position + 1,
        ))
    }
}

/// One current row and immutable, time-major prior KV. No padded cache lanes.
pub struct TorchCpuAttentionInputs<'a, V: ?Sized> {
    pub query: &'a V,
    pub key: &'a V,
    pub value: &'a V,
    pub gate: &'a V,
    pub prior: Option<(&'a V, &'a V)>,
    pub geometry: TorchCpuAttentionGeometry,
    pub position: usize,
}

pub struct TorchCpuAttentionValues<V> {
    pub rotated_query: V,
    pub rotated_key: V,
    pub keys: V,
    pub values: V,
    pub dots: V,
    pub scores: V,
    pub probabilities: V,
    pub context: V,
    pub gated: V,
}
impl<V> TorchCpuAttentionValues<V> {
    pub fn try_map<W>(
        self,
        mut f: impl FnMut(V) -> Result<W>,
    ) -> Result<TorchCpuAttentionValues<W>> {
        Ok(TorchCpuAttentionValues {
            rotated_query: f(self.rotated_query)?,
            rotated_key: f(self.rotated_key)?,
            keys: f(self.keys)?,
            values: f(self.values)?,
            dots: f(self.dots)?,
            scores: f(self.scores)?,
            probabilities: f(self.probabilities)?,
            context: f(self.context)?,
            gated: f(self.gated)?,
        })
    }
}

pub(super) fn result_precision(
    builder: &PlanBuilder,
    geometry: TorchCpuAttentionGeometry,
    position: usize,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    let (q, kv, _) = geometry.validate(position)?;
    let expected = if position == 0 {
        vec![q, kv, kv, q]
    } else {
        vec![q, kv, kv, q, position * kv, position * kv]
    };
    if builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || inputs.len() != expected.len()
        || inputs
            .iter()
            .zip(expected)
            .any(|(v, n)| v.elements != n || v.precision != ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::State(
            "source attention profile/prior/precision/extent".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn record(
    builder: &mut PlanBuilder,
    name: &str,
    input: TorchCpuAttentionInputs<'_, PlanValue>,
) -> Result<TorchCpuAttentionValues<PlanValue>> {
    let (q, kv, tokens) = input.geometry.validate(input.position)?;
    let mut inputs = vec![*input.query, *input.key, *input.value, *input.gate];
    if let Some((k, v)) = input.prior {
        inputs.extend([*k, *v]);
    }
    let scores = input.geometry.query_heads * tokens;
    let output = builder.record(
        name,
        PlanOperation::TypedTorchAttention {
            geometry: input.geometry,
            position: input.position,
        },
        &inputs,
        &[
            q,
            kv,
            tokens * kv,
            tokens * kv,
            scores,
            scores,
            scores,
            q,
            q,
        ]
        .map(ValueShape::Vector),
        None,
    )?;
    let [
        rotated_query,
        rotated_key,
        keys,
        values,
        dots,
        scores,
        probabilities,
        context,
        gated,
    ] = output
        .try_into()
        .map_err(|_| Error::State("source attention output arity".into()))?;
    Ok(TorchCpuAttentionValues {
        rotated_query,
        rotated_key,
        keys,
        values,
        dots,
        scores,
        probabilities,
        context,
        gated,
    })
}
