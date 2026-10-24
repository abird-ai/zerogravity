//! Source-neutral adapter for the existing mathematical schedule.
//! Default F32 analytic plus an explicit unqualified BF16 cast candidate; no native permit.
use crate::{Error, Result};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_core::accelerator::{SourceIdentity, program::*};
#[cfg(all(feature = "cache", target_os = "linux"))]
mod windowed;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub use windowed::{ReferenceTextParameter, WindowReferenceTextSource};
#[cfg(all(test, feature = "cache", target_os = "linux"))]
mod windowed_tests;

mod bf16;
mod moe;
mod observer;
pub use observer::{
    ReferenceTextArithmetic, ReferenceTextObservation, ReferenceTextObserver,
    ReferenceTextOperation,
};
pub type TextAdmission = Arc<dyn Any + Send + Sync>;
/// Immutable source rows/projections. A real window source charges outputs
/// before allocation; synthetic fixtures may use an explicitly bounded owner.
pub trait ReferenceTextSource {
    fn project_expert(
        &self,
        _tensor: &TextExpertTensor,
        _expert: usize,
        _input: &[f32],
        _cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        Err(Error::Unsupported(
            "reference source has no immutable expert planes".into(),
        ))
    }
    fn project_with_arithmetic(
        &self,
        tensor: &SourceTensor,
        input: &[f32],
        arithmetic: ReferenceTextArithmetic,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        match arithmetic {
            ReferenceTextArithmetic::AnalyticF32 => self.project(tensor, input, cancelled),
            ReferenceTextArithmetic::Bf16EagerCandidate => Err(Error::Unsupported(
                "immutable text source did not declare candidate BF16 projection".into(),
            )),
        }
    }
    fn source_identity(&self) -> SourceIdentity;
    fn reserve_output(&self, bytes: usize) -> Result<Option<TextAdmission>>;
    fn embed(
        &self,
        tensor: &SourceTensor,
        token: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>>;
    fn project(
        &self,
        tensor: &SourceTensor,
        input: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>>;
}
struct Storage {
    values: Vec<f32>,
    _admission: Option<TextAdmission>,
}
#[derive(Clone)]
pub struct ReferenceTextValue {
    storage: Arc<Storage>,
    view: TensorView,
    source: SourceIdentity,
    arithmetic: ReferenceTextArithmetic,
}
impl ReferenceTextValue {
    pub fn as_slice(&self) -> Result<&[f32]> {
        if !self.view.is_contiguous() {
            return Err(Error::State(
                "text reference requires explicit contiguous view".into(),
            ));
        }
        self.storage
            .values
            .get(self.view.offset..self.view.offset + self.view.elements()?)
            .ok_or_else(|| Error::State("text reference value extent".into()))
    }
}
pub struct ReferenceTextState {
    source: SourceIdentity,
    arithmetic: ReferenceTextArithmetic,
    capacity: usize,
    frontier: usize,
    kv: BTreeMap<KvDomain, Vec<(ReferenceTextValue, ReferenceTextValue)>>,
    // Each current/proposed state owns its own complete descriptor admission.
    _metadata: Option<TextAdmission>,
}
impl ReferenceTextState {
    pub fn new(
        source: &dyn ReferenceTextSource,
        domains: &[KvDomain],
        capacity: usize,
    ) -> Result<Self> {
        Self::new_with_arithmetic(
            source,
            domains,
            capacity,
            ReferenceTextArithmetic::AnalyticF32,
        )
    }
    pub fn new_with_arithmetic(
        source: &dyn ReferenceTextSource,
        domains: &[KvDomain],
        capacity: usize,
        arithmetic: ReferenceTextArithmetic,
    ) -> Result<Self> {
        if arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate && capacity > 64 {
            return Err(Error::Unsupported(
                "candidate BF16 text cache is bounded to64 rows".into(),
            ));
        }
        if !source.source_identity().is_bound()
            || domains.is_empty()
            || domains.len() > 64
            || capacity == 0
            || capacity > 8192
        {
            return Err(Error::Limit(
                "bounded reference text source/KV capacity".into(),
            ));
        }
        let bytes = domains
            .len()
            .checked_mul(capacity)
            .and_then(|x| {
                x.checked_mul(std::mem::size_of::<(ReferenceTextValue, ReferenceTextValue)>())
            })
            .and_then(|x| x.checked_add(65536))
            .ok_or_else(|| Error::Limit("text state metadata overflow".into()))?;
        let metadata = source.reserve_output(bytes)?;
        let mut kv = BTreeMap::new();
        for domain in domains {
            let mut rows = Vec::new();
            rows.try_reserve_exact(capacity)
                .map_err(|_| Error::Limit("text KV descriptors".into()))?;
            if kv.insert(*domain, rows).is_some() {
                return Err(Error::Malformed("duplicate text KV owner".into()));
            }
        }
        Ok(Self {
            source: source.source_identity(),
            arithmetic,
            capacity,
            frontier: 0,
            kv,
            _metadata: metadata,
        })
    }
    pub fn frontier(&self) -> usize {
        self.frontier
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn fork(&self, source: &dyn ReferenceTextSource) -> Result<Self> {
        if source.source_identity() != self.source {
            return Err(Error::State("foreign reference state fork source".into()));
        }
        self.validate_complete()?;
        let mut next = Self::new_with_arithmetic(
            source,
            &self.kv.keys().copied().collect::<Vec<_>>(),
            self.capacity,
            self.arithmetic,
        )?;
        for (domain, rows) in &self.kv {
            next.kv
                .get_mut(domain)
                .ok_or_else(|| Error::State("reference fork exact domain".into()))?
                .extend(rows.iter().cloned());
        }
        next.frontier = self.frontier;
        Ok(next)
    }
    pub fn domain_tokens(&self, domain: KvDomain) -> Result<usize> {
        Ok(self
            .kv
            .get(&domain)
            .ok_or_else(|| Error::State("foreign text KV domain".into()))?
            .len())
    }
    /// Borrow actual immutable owner rows; no copy, source retag or state write.
    pub fn domain_row(&self, domain: KvDomain, token: usize) -> Result<(&[f32], &[f32])> {
        self.validate_complete()?;
        let rows = self
            .kv
            .get(&domain)
            .ok_or_else(|| Error::State("foreign text KV domain".into()))?;
        let (key, value) = rows
            .get(token)
            .ok_or_else(|| Error::State("text KV row outside completed frontier".into()))?;
        if token >= self.frontier
            || key.source != self.source
            || value.source != self.source
            || key.arithmetic != self.arithmetic
            || value.arithmetic != self.arithmetic
        {
            return Err(Error::State("text KV completed row/source".into()));
        }
        Ok((key.as_slice()?, value.as_slice()?))
    }
    pub fn validate_complete(&self) -> Result<()> {
        if self.frontier > self.capacity || self.kv.values().any(|rows| rows.len() != self.frontier)
        {
            return Err(Error::State(
                "incomplete text transaction KV frontier".into(),
            ));
        }
        Ok(())
    }
}

pub struct ReferenceTextMath<'a> {
    source: &'a dyn ReferenceTextSource,
    state: &'a mut ReferenceTextState,
    cancelled: Option<&'a AtomicBool>,
    advanced: bool,
    position: usize,
    arithmetic: ReferenceTextArithmetic,
    observer: Option<&'a dyn ReferenceTextObserver>,
    bf16_attention_call: Option<(usize, usize)>,
}
impl<'a> ReferenceTextMath<'a> {
    pub fn new(
        source: &'a dyn ReferenceTextSource,
        state: &'a mut ReferenceTextState,
        cancelled: Option<&'a AtomicBool>,
    ) -> Result<Self> {
        Self::new_observed(
            source,
            state,
            cancelled,
            ReferenceTextArithmetic::AnalyticF32,
            None,
        )
    }
    pub fn new_observed(
        source: &'a dyn ReferenceTextSource,
        state: &'a mut ReferenceTextState,
        cancelled: Option<&'a AtomicBool>,
        arithmetic: ReferenceTextArithmetic,
        observer: Option<&'a dyn ReferenceTextObserver>,
    ) -> Result<Self> {
        state.validate_complete()?;
        if arithmetic != state.arithmetic
            || source.source_identity() != state.source
            || state.frontier >= state.capacity
        {
            return Err(Error::State("text backend actual source/capacity".into()));
        }
        let position = state.frontier;
        let result = Self {
            source,
            state,
            cancelled,
            advanced: false,
            position,
            arithmetic,
            observer,
            bf16_attention_call: None,
        };
        result.check()?;
        Ok(result)
    }
    /// Bind actual source call rows/end before candidate attention; no root or state write.
    pub fn bind_bf16_attention_call(&mut self, query_rows: usize, end: usize) -> Result<()> {
        self.check()?;
        if self.arithmetic != ReferenceTextArithmetic::Bf16EagerCandidate
            || self.bf16_attention_call.is_some()
            || self.advanced
            || !(1..=64).contains(&query_rows)
            || end > self.state.capacity
            || end <= self.position
            || query_rows > end
            || self.position < end - query_rows
        {
            return Err(Error::State(
                "candidate BF16 attention actual call geometry".into(),
            ));
        }
        self.bf16_attention_call = Some((query_rows, end));
        Ok(())
    }
    fn check(&self) -> Result<()> {
        if self.cancelled.is_some_and(|x| x.load(Ordering::Acquire)) {
            Err(Error::State("reference text cancelled".into()))
        } else {
            Ok(())
        }
    }
    fn input<'v>(&self, value: &'v ReferenceTextValue) -> Result<&'v [f32]> {
        self.check()?;
        if value.source != self.source.source_identity() || value.arithmetic != self.arithmetic {
            return Err(Error::State("foreign reference text value".into()));
        }
        value.as_slice()
    }
    fn produce(
        &self,
        elements: usize,
        allow_mask: bool,
        calculate: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<ReferenceTextValue> {
        self.check()?;
        if elements == 0 || elements > 1_048_576 {
            return Err(Error::Limit("reference text bounded activation".into()));
        }
        let admission = self.source.reserve_output(
            elements
                .checked_mul(4)
                .and_then(|bytes| bytes.checked_add(256))
                .ok_or_else(|| Error::Limit("reference text bytes/owner metadata".into()))?,
        )?;
        let mut values = calculate()?;
        if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate {
            for value in &mut values {
                if !(allow_mask && *value == f32::NEG_INFINITY) {
                    *value = bf16::round(*value)?;
                }
            }
        }
        if values.len() != elements
            || values.capacity() > elements
            || values
                .iter()
                .any(|x| !(x.is_finite() || allow_mask && *x == f32::NEG_INFINITY))
        {
            return Err(Error::Malformed(
                "reference text complete finite output".into(),
            ));
        }
        self.check()?;
        Ok(ReferenceTextValue {
            storage: Arc::new(Storage {
                values,
                _admission: admission,
            }),
            view: TensorView::vector(0, elements),
            source: self.source.source_identity(),
            arithmetic: self.arithmetic,
        })
    }
    fn produce_observed(
        &self,
        label: &str,
        operation: ReferenceTextOperation,
        elements: usize,
        allow_mask: bool,
        calculate: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<ReferenceTextValue> {
        let value = self.produce(elements, allow_mask, calculate)?;
        self.notify(label, operation, &value)?;
        Ok(value)
    }
    fn notify(
        &self,
        label: &str,
        operation: ReferenceTextOperation,
        value: &ReferenceTextValue,
    ) -> Result<()> {
        if let Some(observer) = self.observer {
            observer.output(ReferenceTextObservation {
                source: self.source.source_identity(),
                position: self.position,
                layer: None,
                arithmetic: self.arithmetic,
                operation,
                label,
                values: value.as_slice()?,
            })?;
        }
        self.check()
    }
    fn kv(
        &self,
        domain: KvDomain,
        valid: usize,
    ) -> Result<&[(ReferenceTextValue, ReferenceTextValue)]> {
        self.check()?;
        let rows = self
            .state
            .kv
            .get(&domain)
            .ok_or_else(|| Error::State("reference text KV source owner".into()))?;
        if valid == 0 || valid > self.state.capacity || rows.len() != valid {
            return Err(Error::State(
                "reference text exact appended KV prefix".into(),
            ));
        }
        Ok(rows)
    }
}
impl MathematicalBackend for ReferenceTextMath<'_> {
    type Value = ReferenceTextValue;
    type Index = usize;
    fn precision(&self, value: &Self::Value) -> ActivationPrecision {
        value.arithmetic.precision()
    }
    fn elements(&self, value: &Self::Value) -> usize {
        value.view.elements().unwrap_or(0)
    }
    fn text_moe(
        &mut self,
        _name: &str,
        contract: &TextMoeContract,
        inputs: TextMoeInputs<'_, Self::Value>,
    ) -> Result<Self::Value> {
        if self.arithmetic != ReferenceTextArithmetic::AnalyticF32 {
            return Err(Error::Unsupported(
                "candidate BF16 text does not cover sparse experts".into(),
            ));
        }
        moe::evaluate(self, contract, inputs)
    }
    fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<Self::Value> {
        if tensor.source != self.source.source_identity()
            || token >= tensor.rows
            || self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate
                && tensor.kind != zerogravity_core::quant::QuantKind::Bf16
        {
            return Err(Error::State("text embedding original source/token".into()));
        }
        self.produce_observed(
            &tensor.name,
            ReferenceTextOperation::Embedding,
            tensor.columns,
            false,
            || self.source.embed(&tensor, token, self.cancelled),
        )
    }
    fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value> {
        let values = self.input(input)?;
        if tensor.source != self.source.source_identity() || tensor.columns != values.len() {
            return Err(Error::State(
                "text projection original source/full input".into(),
            ));
        }
        self.produce_observed(
            &tensor.name,
            ReferenceTextOperation::Projection,
            tensor.rows,
            false,
            || {
                self.source.project_with_arithmetic(
                    &tensor,
                    values,
                    self.arithmetic,
                    self.cancelled,
                )
            },
        )
    }
    fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value> {
        self.input(input)?;
        view.validate(self.elements(input))?;
        Ok(ReferenceTextValue {
            storage: input.storage.clone(),
            view: TensorView {
                offset: view.offset + input.view.offset,
                ..view
            },
            source: input.source,
            arithmetic: input.arithmetic,
        })
    }
    fn contiguous(&mut self, input: &Self::Value) -> Result<Self::Value> {
        self.check()?;
        input.view.validate(input.storage.values.len())?;
        if input.source != self.source.source_identity() || input.arithmetic != self.arithmetic {
            return Err(Error::State("foreign text pack input".into()));
        }
        self.produce_observed(
            "contiguous",
            ReferenceTextOperation::Pack,
            self.elements(input),
            false,
            || {
                let mut out = Vec::new();
                out.try_reserve_exact(self.elements(input))
                    .map_err(|_| Error::Limit("text contiguous output".into()))?;
                for row in 0..input.view.rows {
                    self.check()?;
                    let start = input.view.offset + row * input.view.row_stride;
                    out.extend_from_slice(&input.storage.values[start..start + input.view.width]);
                }
                Ok(out)
            },
        )
    }
    fn rms(
        &mut self,
        name: &str,
        input: &Self::Value,
        gamma: SourceVector<'_>,
        eps: f32,
    ) -> Result<Self::Value> {
        self.grouped_rms(name, input, gamma, 1, eps)
    }
    fn grouped_rms(
        &mut self,
        name: &str,
        input: &Self::Value,
        gamma: SourceVector<'_>,
        groups: usize,
        eps: f32,
    ) -> Result<Self::Value> {
        let x = self.input(input)?;
        if gamma.source != self.source.source_identity()
            || gamma.name.is_empty()
            || groups == 0
            || gamma.values.len().checked_mul(groups) != Some(x.len())
        {
            return Err(Error::State("text RMS source/gamma/group geometry".into()));
        }
        // Existing RMS returns one temporary group beside the complete output.
        let _scratch = self.source.reserve_output(gamma.values.len() * 4)?;
        self.produce_observed(name, ReferenceTextOperation::Rms, x.len(), false, || {
            if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate {
                return bf16::rms(x, gamma.values, eps);
            }
            let mut out = Vec::new();
            out.try_reserve_exact(x.len())
                .map_err(|_| Error::Limit("text RMS output".into()))?;
            for row in x.chunks_exact(gamma.values.len()) {
                self.check()?;
                out.extend(crate::math::rms_norm(row, gamma.values, eps)?);
            }
            Ok(out)
        })
    }
    fn add(&mut self, name: &str, left: &Self::Value, right: &Self::Value) -> Result<Self::Value> {
        let a = self.input(left)?;
        let b = self.input(right)?;
        if a.len() != b.len() {
            return Err(Error::Malformed("text add shapes".into()));
        }
        self.produce_observed(
            name,
            ReferenceTextOperation::Elementwise,
            a.len(),
            false,
            || Ok(a.iter().zip(b).map(|(a, b)| a + b).collect()),
        )
    }
    fn text_elementwise(
        &mut self,
        name: &str,
        operation: TextElementwise,
        inputs: &[&Self::Value],
    ) -> Result<Self::Value> {
        operation.validate(inputs.len())?;
        if matches!(operation, TextElementwise::ScaleBf16Scalar(_))
            && self.arithmetic != ReferenceTextArithmetic::Bf16EagerCandidate
        {
            return Err(Error::Unsupported(
                "BF16 scalar cast requires explicit BF16 candidate".into(),
            ));
        }
        let bf16_scalar = if let TextElementwise::ScaleBf16Scalar(scale) = operation {
            bf16::round(scale)?
        } else {
            0.0
        };
        let values = inputs
            .iter()
            .map(|x| self.input(x))
            .collect::<Result<Vec<_>>>()?;
        let count = values[0].len();
        if values.iter().any(|v| v.len() != count) {
            return Err(Error::Malformed("text elementwise complete shapes".into()));
        }
        if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate
            && matches!(operation, TextElementwise::GeluTanh)
        {
            return self.produce_observed(
                name,
                ReferenceTextOperation::Elementwise,
                count,
                false,
                || {
                    crate::original_hf::transcendentals::gelu_tanh_bf16_widened_f32_contiguous16(
                        self.source.source_identity(),
                        values[0],
                    )
                },
            );
        }
        if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate
            && matches!(operation, TextElementwise::TanhSoftcap(_))
            && !count.is_multiple_of(16)
        {
            return Err(Error::Unsupported(
                "candidate BF16 tanh requires complete original vector spans".into(),
            ));
        }
        self.produce_observed(
            name,
            ReferenceTextOperation::Elementwise,
            count,
            false,
            || {
                Ok((0..count)
                    .map(|i| match operation {
                        TextElementwise::Scale(scale) => values[0][i] * scale,
                        TextElementwise::ScaleBf16Scalar(_) => values[0][i] * bf16_scalar,
                        TextElementwise::GeluTanh => crate::math::gelu_tanh(values[0][i]),
                        TextElementwise::Multiply => values[0][i] * values[1][i],
                        TextElementwise::TanhSoftcap(cap)
                            if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate =>
                        {
                            bf16::softcap(values[0][i], cap)
                        }
                        TextElementwise::TanhSoftcap(cap) => (values[0][i] / cap).tanh() * cap,
                        TextElementwise::Add => values[0][i] + values[1][i],
                    })
                    .collect())
            },
        )
    }
    fn text_rotary(
        &mut self,
        name: &str,
        input: &Self::Value,
        position: usize,
        geometry: TextRotary,
    ) -> Result<Self::Value> {
        let x = self.input(input)?;
        geometry.validate(x.len())?;
        if position != self.state.frontier {
            return Err(Error::State("text rotary exact current position".into()));
        }
        let _candidate_row = if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate {
            Some(self.source.reserve_output(geometry.head_width * 4)?)
        } else {
            None
        };
        self.produce_observed(name, ReferenceTextOperation::Rotary, x.len(), false, || {
            if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate {
                return bf16::rotary(x, position, geometry);
            }
            let mut out = x.to_vec();
            for (head, row) in x.chunks_exact(geometry.head_width).enumerate() {
                self.check()?;
                for pair in 0..geometry.rotated_pairs {
                    let angle = position as f32
                        / geometry
                            .theta
                            .powf((2 * pair) as f32 / geometry.head_width as f32);
                    let (sin, cos) = angle.sin_cos();
                    let a = row[pair];
                    let b = row[geometry.head_width / 2 + pair];
                    let start = head * geometry.head_width;
                    out[start + pair] = a * cos - b * sin;
                    out[start + geometry.head_width / 2 + pair] = b * cos + a * sin;
                }
            }
            Ok(out)
        })
    }
    fn validate_kv_prefix(&mut self, domain: KvDomain, position: usize) -> Result<()> {
        self.check()?;
        if position != self.state.frontier || self.state.domain_tokens(domain)? != position {
            return Err(Error::State("text original KV prefix/frontier".into()));
        }
        Ok(())
    }
    fn append_kv(
        &mut self,
        domain: KvDomain,
        key: &Self::Value,
        value: &Self::Value,
    ) -> Result<()> {
        let k = self.input(key)?;
        let v = self.input(value)?;
        if k.len() != v.len() {
            return Err(Error::Malformed("text KV exact equal widths".into()));
        }
        let rows = self
            .state
            .kv
            .get_mut(&domain)
            .ok_or_else(|| Error::State("text KV append owner absent".into()))?;
        if rows.len() != self.state.frontier || rows.len() >= self.state.capacity {
            return Err(Error::State("text KV append ordinal/capacity".into()));
        }
        rows.push((key.clone(), value.clone()));
        Ok(())
    }
    fn text_window_qk(
        &mut self,
        name: &str,
        domain: KvDomain,
        query: &Self::Value,
        valid: usize,
        g: AttentionGeometry,
        window: Option<usize>,
    ) -> Result<QkValues<Self::Value>> {
        let q = self.input(query)?;
        let rows = self.kv(domain, valid)?;
        let kv_elements = g
            .kv_heads
            .checked_mul(g.width)
            .ok_or_else(|| Error::Limit("reference KV head extent overflow".into()))?;
        if g.width == 0
            || g.kv_heads == 0
            || !g.query_heads.is_multiple_of(g.kv_heads)
            || g.query_heads.checked_mul(g.width) != Some(q.len())
            || window == Some(0)
            || rows
                .iter()
                .any(|(k, v)| self.elements(k) != kv_elements || self.elements(v) != kv_elements)
        {
            return Err(Error::Malformed(
                "text QK actual grouped geometry/window".into(),
            ));
        }
        let n = g
            .query_heads
            .checked_mul(valid)
            .ok_or_else(|| Error::Limit("text score extent".into()))?;
        let _candidate_scratch = if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate {
            self.source.reserve_output(g.width * 2)?
        } else {
            None
        };
        let start = window.map_or(0, |w| valid.saturating_sub(w));
        let scores = self.produce_observed(name, ReferenceTextOperation::Qk, n, true, || {
            let mut out = vec![f32::NEG_INFINITY; n];
            for h in 0..g.query_heads {
                for t in start..valid {
                    self.check()?;
                    let k = self.input(&rows[t].0)?;
                    let kh = h / (g.query_heads / g.kv_heads);
                    out[h * valid + t] = if self.arithmetic
                        == ReferenceTextArithmetic::Bf16EagerCandidate
                    {
                        bf16::dot(
                            &q[h * g.width..(h + 1) * g.width],
                            &k[kh * g.width..(kh + 1) * g.width],
                        )?
                    } else {
                        (0..g.width)
                            .map(|i| f64::from(q[h * g.width + i]) * f64::from(k[kh * g.width + i]))
                            .sum::<f64>() as f32
                    };
                }
            }
            Ok(out)
        })?;
        Ok(QkValues {
            dots: scores.clone(),
            scores,
        })
    }
    fn attention_softmax(
        &mut self,
        name: &str,
        scores: &Self::Value,
        heads: usize,
        valid: usize,
    ) -> Result<Self::Value> {
        let x = self.input(scores)?;
        if heads == 0 || valid == 0 || heads.checked_mul(valid) != Some(x.len()) {
            return Err(Error::Malformed("text softmax shapes".into()));
        }
        // Existing softmax uses an F64 weight row and a returned F32 row.
        let _scratch = self.source.reserve_output(
            valid
                .checked_mul(12)
                .ok_or_else(|| Error::Limit("reference softmax scratch".into()))?,
        )?;
        self.produce_observed(
            name,
            ReferenceTextOperation::Softmax,
            x.len(),
            false,
            || {
                if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate {
                    return crate::original_hf::transcendentals::softmax_f32_lastdim_avx2(
                        self.source.source_identity(),
                        x,
                        valid,
                    );
                }
                let mut out = Vec::new();
                out.try_reserve_exact(x.len())
                    .map_err(|_| Error::Limit("text probabilities".into()))?;
                for row in x.chunks_exact(valid) {
                    self.check()?;
                    let first = row
                        .iter()
                        .position(|x| x.is_finite())
                        .ok_or_else(|| Error::Malformed("text attention all masked".into()))?;
                    if row[..first].iter().any(|x| *x != f32::NEG_INFINITY) {
                        return Err(Error::Malformed("text leading causal/window mask".into()));
                    }
                    out.resize(out.len() + first, 0.0);
                    out.extend(crate::math::softmax(&row[first..])?);
                }
                Ok(out)
            },
        )
    }
    fn attention_value(
        &mut self,
        name: &str,
        domain: KvDomain,
        probabilities: &Self::Value,
        valid: usize,
        g: AttentionGeometry,
    ) -> Result<Self::Value> {
        let p = self.input(probabilities)?;
        let rows = self.kv(domain, valid)?;
        let kv_elements = g
            .kv_heads
            .checked_mul(g.width)
            .ok_or_else(|| Error::Limit("reference KV head extent overflow".into()))?;
        if g.width == 0
            || g.kv_heads == 0
            || !g.query_heads.is_multiple_of(g.kv_heads)
            || g.query_heads.checked_mul(valid) != Some(p.len())
        {
            return Err(Error::Malformed("text value attention shapes".into()));
        }
        let _candidate_scratch = if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate {
            self.source.reserve_output(valid * 6)?
        } else {
            None
        };
        let elements = g
            .query_heads
            .checked_mul(g.width)
            .ok_or_else(|| Error::Limit("text value attention extent overflow".into()))?;
        let pv_geometry = if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate {
            let (query_rows, contraction) = self.bf16_attention_call.ok_or_else(|| {
                Error::State("candidate PV requires actual current-call contraction".into())
            })?;
            Some(crate::original_hf::pv::Bf16PvGeometry {
                query_rows,
                contraction,
                output_width: g.width,
            })
        } else {
            None
        };
        self.produce_observed(name, ReferenceTextOperation::Pv, elements, false, || {
            let mut out = vec![0.0; elements];
            for h in 0..g.query_heads {
                self.check()?;
                let vh = h / (g.query_heads / g.kv_heads);
                for lane in 0..g.width {
                    if self.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate {
                        let mut column = Vec::with_capacity(valid);
                        for pair in rows {
                            column.push(self.input(&pair.1)?[vh * g.width + lane]);
                        }
                        out[h * g.width + lane] = crate::original_hf::pv::bf16_pv_f32_accumulator(
                            self.source.source_identity(),
                            &p[h * valid..(h + 1) * valid],
                            &column,
                            pv_geometry.expect("validated candidate PV geometry"),
                        )?;
                        continue;
                    }
                    let mut sum = 0.0f64;
                    for t in 0..valid {
                        let v = self.input(&rows[t].1)?;
                        if v.len() != kv_elements {
                            return Err(Error::State("text actual V row width".into()));
                        }
                        sum += f64::from(p[h * valid + t]) * f64::from(v[vh * g.width + lane]);
                    }
                    out[h * g.width + lane] = sum as f32;
                }
            }
            Ok(out)
        })
    }
    fn assert_finite(&mut self, _layer: usize, value: &Self::Value) -> Result<()> {
        if self.input(value)?.iter().any(|x| !x.is_finite()) {
            return Err(Error::Malformed("nonfinite text layer output".into()));
        }
        Ok(())
    }
    fn observe(&mut self, layer: usize, stage: &'static str, value: &Self::Value) -> Result<()> {
        self.input(value)?;
        if let Some(observer) = self.observer {
            observer.output(ReferenceTextObservation {
                source: self.source.source_identity(),
                position: self.position,
                layer: (layer != usize::MAX).then_some(layer),
                arithmetic: self.arithmetic,
                operation: ReferenceTextOperation::Boundary,
                label: stage,
                values: value.as_slice()?,
            })?;
        }
        self.check()
    }
    fn observe_state(
        &mut self,
        _layer: usize,
        _stage: &'static str,
        domain: StateSlot,
    ) -> Result<()> {
        self.check()?;
        match domain {
            StateSlot::Kv(d) => {
                self.state.domain_tokens(d)?;
                Ok(())
            }
            _ => Err(Error::Unsupported(
                "reference text has only declared KV domains".into(),
            )),
        }
    }
    fn advance_frontier(&mut self) -> Result<()> {
        self.check()?;
        if self.advanced
            || self
                .state
                .kv
                .values()
                .any(|x| x.len() != self.state.frontier + 1)
        {
            return Err(Error::State(
                "text frontier requires every source KV owner exactly once".into(),
            ));
        }
        self.state.frontier += 1;
        self.advanced = true;
        Ok(())
    }
    fn silu(&mut self, _: &str, _: &Self::Value, _: Option<&Self::Value>) -> Result<Self::Value> {
        Err(Error::Unsupported(
            "undeclared SiLU in analytic text adapter".into(),
        ))
    }
    fn sigmoid_mul(&mut self, _: &str, _: &Self::Value, _: &Self::Value) -> Result<Self::Value> {
        Err(Error::Unsupported(
            "undeclared sigmoid gate in analytic text adapter".into(),
        ))
    }
    fn conv_step(
        &mut self,
        _: &str,
        _: ConvDomain,
        _: &Self::Value,
        _: SourceVector<'_>,
    ) -> Result<Self::Value> {
        Err(Error::Unsupported(
            "no convolution in analytic text adapter".into(),
        ))
    }
    fn coefficients(
        &mut self,
        _: &str,
        _: &Self::Value,
        _: &Self::Value,
        _: SourceVector<'_>,
        _: SourceVector<'_>,
        _: f32,
    ) -> Result<CoefficientValues<Self::Value>> {
        Err(Error::Unsupported(
            "no recurrent coefficients in analytic text adapter".into(),
        ))
    }
    fn recurrent_step(
        &mut self,
        _: &str,
        _: RecurrentDomain,
        _: &Self::Value,
        _: &Self::Value,
        _: &Self::Value,
        _: &Self::Value,
        _: usize,
    ) -> Result<Self::Value> {
        Err(Error::Unsupported(
            "no recurrent state in analytic text adapter".into(),
        ))
    }
    fn rope(
        &mut self,
        _: &str,
        _: &Self::Value,
        _: &Self::Value,
        _: &Self::Value,
        _: usize,
        _: RopeGeometry,
    ) -> Result<RopeValues<Self::Value>> {
        Err(Error::Unsupported(
            "analytic text requires explicit single-input rotary".into(),
        ))
    }
    fn attention_qk(
        &mut self,
        _: &str,
        _: KvDomain,
        _: &Self::Value,
        _: usize,
        _: AttentionGeometry,
        _: f32,
    ) -> Result<QkValues<Self::Value>> {
        Err(Error::Unsupported(
            "analytic text requires explicit windowed unit-scale QK".into(),
        ))
    }
}
