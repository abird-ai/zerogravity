//! CPU/optional-region adapter for the single generic forward schedule.
//! The existing QwenState remains private and unchanged in representation.

use super::{
    LayerObservation, LayerState, QwenDenseCpu, QwenState, observe, source_gdn_recurrent_step,
};
use crate::accelerator::program::{
    AttentionGeometry, CoefficientValues, ConvDomain, KvDomain, MathematicalBackend, QkValues,
    RecurrentDomain, RopeGeometry, RopeValues, SourceTensor, SourceVector, StateSlot, TensorView,
};
use crate::accelerator::{
    AttentionQkRegion, AttentionSoftmaxRegion, AttentionValueRegion, BinaryOperation, BinaryRegion,
    GdnCoefficientRegion, GdnCoefficients, GdnRecurrentExecution, GdnRecurrentRegion, RopeKvRegion,
};
use crate::{Error, Result};
use std::sync::Arc;

type HostKvView<'a> = (&'a [Arc<[f32]>], &'a [Arc<[f32]>]);

/// Host views share a Vec allocation without cloning numerical payload. A
/// strided view is packed explicitly by contiguous(), never by hidden math.
#[derive(Clone)]
pub(super) struct HostValue {
    data: Arc<Vec<f32>>,
    view: TensorView,
    elements: usize,
}
impl HostValue {
    fn new(values: Vec<f32>) -> Self {
        let elements = values.len();
        Self {
            data: Arc::new(values),
            view: TensorView::vector(0, elements),
            elements,
        }
    }
    fn as_slice(&self) -> Result<&[f32]> {
        if !self.view.is_contiguous() {
            return Err(Error::State(
                "host operator requires explicit contiguous input".into(),
            ));
        }
        self.data
            .get(self.view.offset..self.view.offset + self.elements)
            .ok_or_else(|| Error::State("invalid private host tensor view".into()))
    }
    pub fn into_vec(self) -> Result<Vec<f32>> {
        if self.view.offset == 0 && self.view.is_contiguous() && self.elements == self.data.len() {
            return Ok(Arc::try_unwrap(self.data).unwrap_or_else(|data| data.as_ref().clone()));
        }
        Ok(self.as_slice()?.to_vec())
    }
}

pub(super) struct HostMath<'engine, 'state, 'trace> {
    engine: &'engine QwenDenseCpu,
    state: &'state mut QwenState,
    trace: Option<&'trace mut Vec<LayerObservation>>,
    position: usize,
}
impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
    pub fn new(
        engine: &'engine QwenDenseCpu,
        state: &'state mut QwenState,
        trace: Option<&'trace mut Vec<LayerObservation>>,
    ) -> Self {
        let position = state.position;
        Self {
            engine,
            state,
            trace,
            position,
        }
    }
    fn tensor(&self, tensor: &SourceTensor) -> Result<()> {
        let actual = self.engine.weights.tensor(&tensor.name)?;
        if tensor.source != self.engine.weights.source_identity()
            || tensor.kind != actual.kind
            || actual.shape != [tensor.columns as u64, tensor.rows as u64]
        {
            return Err(Error::State(
                "host forward source descriptor changed".into(),
            ));
        }
        Ok(())
    }
    fn vector(&self, vector: SourceVector<'_>) -> Result<()> {
        if vector.source != self.engine.weights.source_identity() {
            return Err(Error::State(
                "host forward static vector changed source".into(),
            ));
        }
        Ok(())
    }
    fn kv(&self, domain: KvDomain, tokens: usize) -> Result<HostKvView<'_>> {
        match self.state.layers.get(domain.0) {
            Some(LayerState::Full { keys, values })
                if keys.len() == tokens && values.len() == tokens =>
            {
                Ok((keys, values))
            }
            _ => Err(Error::State(
                "attention KV prefix does not match source position".into(),
            )),
        }
    }
}

#[allow(clippy::too_many_arguments)] // Mirrors already qualified closed-op inputs.
impl MathematicalBackend for HostMath<'_, '_, '_> {
    type Value = HostValue;
    type Index = usize;
    fn elements(&self, value: &HostValue) -> usize {
        value.elements
    }
    fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<HostValue> {
        self.tensor(&tensor)?;
        if tensor.name != "token_embd.weight" {
            return Err(Error::State("unexpected Qwen embedding role".into()));
        }
        let token = u32::try_from(token)
            .map_err(|_| Error::Malformed("embedding ID exceeds source token domain".into()))?;
        Ok(HostValue::new(self.engine.embedding_region(token)?))
    }
    fn project(&mut self, tensor: SourceTensor, input: &HostValue) -> Result<HostValue> {
        self.tensor(&tensor)?;
        Ok(HostValue::new(
            self.engine.project(&tensor.name, input.as_slice()?)?,
        ))
    }
    fn view(&mut self, input: &HostValue, view: TensorView) -> Result<HostValue> {
        // A view of an already strided view must first use explicit packing.
        if !input.view.is_contiguous() {
            return Err(Error::State(
                "nested strided host view requires packing".into(),
            ));
        }
        view.validate(input.elements)?;
        let elements = view.elements()?;
        let offset = input
            .view
            .offset
            .checked_add(view.offset)
            .ok_or_else(|| Error::Limit("host tensor offset overflow".into()))?;
        Ok(HostValue {
            data: input.data.clone(),
            view: TensorView { offset, ..view },
            elements,
        })
    }
    fn contiguous(&mut self, input: &HostValue) -> Result<HostValue> {
        if input.view.is_contiguous() {
            return Ok(input.clone());
        }
        let mut values = Vec::with_capacity(input.elements);
        for row in 0..input.view.rows {
            let offset = input.view.offset + row * input.view.row_stride;
            values.extend_from_slice(&input.data[offset..offset + input.view.width]);
        }
        Ok(HostValue::new(values))
    }
    fn rms(
        &mut self,
        name: &str,
        input: &HostValue,
        gamma: SourceVector<'_>,
        eps: f32,
    ) -> Result<HostValue> {
        self.vector(gamma)?;
        if eps.to_bits() != self.engine.eps.to_bits() {
            return Err(Error::State(
                "host residual RMS changed source epsilon".into(),
            ));
        }
        Ok(HostValue::new(self.engine.normalize_region(
            name,
            input.as_slice()?,
            gamma.values,
        )?))
    }
    fn grouped_rms(
        &mut self,
        name: &str,
        input: &HostValue,
        gamma: SourceVector<'_>,
        groups: usize,
        eps: f32,
    ) -> Result<HostValue> {
        self.vector(gamma)?;
        Ok(HostValue::new(self.engine.normalize_grouped_region(
            name,
            input.as_slice()?,
            gamma.values,
            groups,
            eps,
        )?))
    }
    fn silu(
        &mut self,
        name: &str,
        input: &HostValue,
        multiplier: Option<&HostValue>,
    ) -> Result<HostValue> {
        let multiplier = multiplier.map(HostValue::as_slice).transpose()?;
        Ok(HostValue::new(self.engine.silu_region(
            name,
            input.as_slice()?,
            multiplier,
        )?))
    }
    fn add(&mut self, name: &str, left: &HostValue, right: &HostValue) -> Result<HostValue> {
        Ok(HostValue::new(self.engine.binary_region(BinaryRegion {
            source_identity: self.engine.weights.source_identity(),
            operation: name,
            kind: BinaryOperation::Add,
            left: left.as_slice()?,
            right: right.as_slice()?,
        })?))
    }
    fn sigmoid_mul(
        &mut self,
        name: &str,
        value: &HostValue,
        gate: &HostValue,
    ) -> Result<HostValue> {
        Ok(HostValue::new(self.engine.binary_region(BinaryRegion {
            source_identity: self.engine.weights.source_identity(),
            operation: name,
            kind: BinaryOperation::SigmoidGate,
            left: value.as_slice()?,
            right: gate.as_slice()?,
        })?))
    }
    fn conv_step(
        &mut self,
        name: &str,
        domain: ConvDomain,
        input: &HostValue,
        weights: SourceVector<'_>,
    ) -> Result<HostValue> {
        self.vector(weights)?;
        let Some(LayerState::Delta { conv, .. }) = self.state.layers.get_mut(domain.0) else {
            return Err(Error::State("invalid private convolution domain".into()));
        };
        let input = input.as_slice()?;
        let mut output = vec![0.0; input.len()];
        self.engine.gdn_conv_region(
            name,
            input,
            weights.values,
            Arc::make_mut(conv).as_mut_slice(),
            &mut output,
        )?;
        Ok(HostValue::new(output))
    }
    fn coefficients(
        &mut self,
        name: &str,
        alpha: &HostValue,
        beta: &HostValue,
        dt: SourceVector<'_>,
        a: SourceVector<'_>,
        scale: f32,
    ) -> Result<CoefficientValues<HostValue>> {
        self.vector(dt)?;
        self.vector(a)?;
        let ready = self.engine.gdn_coefficient_region(GdnCoefficientRegion {
            source_identity: self.engine.weights.source_identity(),
            operation: name,
            alpha: alpha.as_slice()?,
            beta: beta.as_slice()?,
            dt: dt.values,
            a: a.values,
            scale,
        })?;
        let packed = ready
            .coefficients
            .iter()
            .flat_map(|c| [c.gain, c.beta, c.scale])
            .collect();
        Ok(CoefficientValues {
            packed: HostValue::new(packed),
            softplus: HostValue::new(ready.softplus),
            gate_raw: HostValue::new(ready.gate_raw),
        })
    }
    fn recurrent_step(
        &mut self,
        name: &str,
        domain: RecurrentDomain,
        q: &HostValue,
        k: &HostValue,
        v: &HostValue,
        coefficients: &HostValue,
        width: usize,
    ) -> Result<HostValue> {
        let q = q.as_slice()?;
        let k = k.as_slice()?;
        let v = v.as_slice()?;
        let packed = coefficients.as_slice()?;
        if width == 0
            || !q.len().is_multiple_of(width)
            || (q.len() / width).checked_mul(3) != Some(packed.len())
        {
            return Err(Error::Malformed("host recurrent coefficient layout".into()));
        }
        let coefficients: Vec<_> = packed
            .chunks_exact(3)
            .map(|c| GdnCoefficients {
                gain: c[0],
                beta: c[1],
                scale: c[2],
            })
            .collect();
        let Some(LayerState::Delta { state, .. }) = self.state.layers.get_mut(domain.0) else {
            return Err(Error::State("invalid private recurrent domain".into()));
        };
        let recurrent = Arc::make_mut(state).as_mut_slice();
        let result = self.engine.gdn_recurrent_region(GdnRecurrentRegion {
            source_identity: self.engine.weights.source_identity(),
            operation: name,
            width,
            q,
            k,
            v,
            coefficients: &coefficients,
            state: recurrent,
        })?;
        let mut output = vec![0.0; q.len()];
        match result {
            GdnRecurrentExecution::Completed(ready) => {
                output.copy_from_slice(&ready.output);
                recurrent.copy_from_slice(&ready.state);
            }
            GdnRecurrentExecution::Unsupported => {
                for (h, &coeff) in coefficients.iter().enumerate() {
                    source_gdn_recurrent_step(
                        &mut recurrent[h * width * width..(h + 1) * width * width],
                        &q[h * width..(h + 1) * width],
                        &k[h * width..(h + 1) * width],
                        &v[h * width..(h + 1) * width],
                        coeff,
                        &mut output[h * width..(h + 1) * width],
                    )?;
                }
            }
        }
        Ok(HostValue::new(output))
    }
    fn validate_kv_prefix(&mut self, domain: KvDomain, position: usize) -> Result<()> {
        self.kv(domain, position)?;
        Ok(())
    }
    fn rope(
        &mut self,
        name: &str,
        q: &HostValue,
        k: &HostValue,
        v: &HostValue,
        position: usize,
        geometry: RopeGeometry,
    ) -> Result<RopeValues<HostValue>> {
        let ready = self.engine.rope_kv_region(RopeKvRegion {
            source_identity: self.engine.weights.source_identity(),
            operation: name,
            query: q.as_slice()?,
            key: k.as_slice()?,
            value: v.as_slice()?,
            head_width: geometry.attention.width,
            query_heads: geometry.attention.query_heads,
            kv_heads: geometry.attention.kv_heads,
            rotate: geometry.rotate,
            position,
            theta_scale: geometry.theta_scale,
        })?;
        Ok(RopeValues {
            query: HostValue::new(ready.query),
            key: HostValue::new(ready.key),
            value: HostValue::new(ready.value),
        })
    }
    fn append_kv(&mut self, domain: KvDomain, k: &HostValue, v: &HostValue) -> Result<()> {
        let Some(LayerState::Full { keys, values }) = self.state.layers.get_mut(domain.0) else {
            return Err(Error::State("invalid private KV domain".into()));
        };
        keys.push(Arc::from(k.as_slice()?));
        values.push(Arc::from(v.as_slice()?));
        Ok(())
    }
    fn attention_qk(
        &mut self,
        name: &str,
        domain: KvDomain,
        query: &HostValue,
        valid_tokens: usize,
        geometry: AttentionGeometry,
        scale: f32,
    ) -> Result<QkValues<HostValue>> {
        let (keys, _) = self.kv(domain, valid_tokens)?;
        let ready = self.engine.attention_qk_region(AttentionQkRegion {
            source_identity: self.engine.weights.source_identity(),
            operation: name,
            query: query.as_slice()?,
            keys,
            head_width: geometry.width,
            query_heads: geometry.query_heads,
            kv_heads: geometry.kv_heads,
            scale,
        })?;
        Ok(QkValues {
            dots: HostValue::new(ready.dots),
            scores: HostValue::new(ready.scores),
        })
    }
    fn attention_softmax(
        &mut self,
        name: &str,
        scores: &HostValue,
        heads: usize,
        valid_tokens: usize,
    ) -> Result<HostValue> {
        let ready = self
            .engine
            .attention_softmax_region(AttentionSoftmaxRegion {
                source_identity: self.engine.weights.source_identity(),
                operation: name,
                scores: scores.as_slice()?,
                heads,
                width: valid_tokens,
            })?;
        Ok(HostValue::new(ready.probabilities))
    }
    fn attention_value(
        &mut self,
        name: &str,
        domain: KvDomain,
        probabilities: &HostValue,
        valid_tokens: usize,
        geometry: AttentionGeometry,
    ) -> Result<HostValue> {
        let (_, values) = self.kv(domain, valid_tokens)?;
        Ok(HostValue::new(self.engine.attention_value_region(
            AttentionValueRegion {
                source_identity: self.engine.weights.source_identity(),
                operation: name,
                probabilities: probabilities.as_slice()?,
                values,
                head_width: geometry.width,
                query_heads: geometry.query_heads,
                kv_heads: geometry.kv_heads,
                padded_tokens: valid_tokens.div_ceil(256) * 256,
            },
        )?))
    }
    fn assert_finite(&mut self, layer: usize, value: &HostValue) -> Result<()> {
        if !value.as_slice()?.iter().all(|v| v.is_finite()) {
            return Err(Error::Malformed(format!(
                "nonfinite layer {layer} computation"
            )));
        }
        Ok(())
    }
    fn observe(&mut self, layer: usize, stage: &'static str, value: &HostValue) -> Result<()> {
        if self.trace.is_some() {
            observe(
                &mut self.trace,
                self.position,
                layer,
                stage,
                value.as_slice()?,
            );
        }
        Ok(())
    }
    fn observe_state(
        &mut self,
        layer: usize,
        stage: &'static str,
        domain: StateSlot,
    ) -> Result<()> {
        if self.trace.is_some() {
            let values = match domain {
                StateSlot::Conv(ConvDomain(i)) => match self.state.layers.get(i) {
                    Some(LayerState::Delta { conv, .. }) => conv.as_slice(),
                    _ => return Err(Error::State("invalid observed convolution domain".into())),
                },
                StateSlot::Recurrent(RecurrentDomain(i)) => match self.state.layers.get(i) {
                    Some(LayerState::Delta { state, .. }) => state.as_slice(),
                    _ => return Err(Error::State("invalid observed recurrent domain".into())),
                },
                StateSlot::Kv(_) => {
                    return Err(Error::Unsupported(
                        "KV state observation needs explicit slab export".into(),
                    ));
                }
            };
            observe(&mut self.trace, self.position, layer, stage, values);
        }
        Ok(())
    }
    fn advance_frontier(&mut self) -> Result<()> {
        if self.state.position != self.position {
            return Err(Error::State(
                "duplicate host proposed frontier advance".into(),
            ));
        }
        self.state.position = self
            .state
            .position
            .checked_add(1)
            .ok_or_else(|| Error::Limit("host proposed position overflow".into()))?;
        Ok(())
    }
}
