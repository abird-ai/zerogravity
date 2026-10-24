# Draft Patches — volume 14

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/frozen/NEW_DELTA.patch

Original UTF-8 bytes: 51516; SHA256 `407c41d851464c2bacf424461b0eed14bf986b2ec6cf3c336b674626f37b7929`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -14,6 +14,10 @@
 
 mod gdn_conv;
 pub use gdn_conv::{CausalConvSiluContract, CausalConvValues};
+mod coefficient_preparation;
+pub use coefficient_preparation::{
+    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues,
+};
 mod typed;
 pub use typed::{
     ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
@@ -294,6 +298,17 @@
             "backend does not implement explicit causal convolution SiLU contract".into(),
         ))
     }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        _dt: SourceParameter,
+        _alpha: &Self::Value,
+        _beta: &Self::Value,
+        _contract: GdnCoefficientPreparationContract,
+    ) -> Result<GdnCoefficientPreparationValues<Self::Value>> {
+        Err(Error::Unsupported(
+            "backend does not implement original GDN coefficient preparation".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -474,6 +489,14 @@
     TypedOrdinaryRms {
         parameter: SourceParameter,
         contract: OrdinaryRmsContract,
+    },
+    TypedGdnAlphaPreparation {
+        dt: SourceParameter,
+        contract: GdnCoefficientPreparationContract,
+    },
+    TypedGdnBetaPreparation {
+        heads: usize,
+        contract: GdnCoefficientPreparationContract,
     },
     Embed {
         tensor: SourceTensor,
@@ -822,6 +845,32 @@
                 parameter,
                 contract,
             } => gdn_conv::result_precision(self, parameter, *contract, inputs),
+            PlanOperation::TypedGdnAlphaPreparation { dt, contract } => {
+                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+                    || dt.source != self.source
+                    || inputs.len() != 1
+                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
+                {
+                    return Err(Error::Unsupported(
+                        "GDN preparation alpha profile/source/precision".into(),
+                    ));
+                }
+                contract.validate(dt, inputs[0].elements, inputs[0].elements)?;
+                Ok(ActivationPrecision::F32)
+            }
+            PlanOperation::TypedGdnBetaPreparation { heads, .. } => {
+                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+                    || !matches!(*heads, 16 | 48)
+                    || inputs.len() != 1
+                    || inputs[0].elements != *heads
+                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
+                {
+                    return Err(Error::Unsupported(
+                        "GDN preparation beta profile/span/precision".into(),
+                    ));
+                }
+                Ok(ActivationPrecision::Bf16WidenedF32)
+            }
             PlanOperation::View(_) | PlanOperation::Contiguous => inputs
                 .first()
                 .map(|value| value.precision)
@@ -839,6 +888,8 @@
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
+                        | PlanOperation::TypedGdnAlphaPreparation { .. }
+                        | PlanOperation::TypedGdnBetaPreparation { .. }
                 ) =>
             {
                 Ok(ActivationPrecision::F32)
@@ -848,14 +899,11 @@
             )),
         }
     }
-    fn record(
-        &mut self,
-        name: &str,
-        operation: PlanOperation,
+    fn validate_record_inputs(
+        &self,
+        operation: &PlanOperation,
         inputs: &[PlanValue],
-        shapes: &[ValueShape],
-        state: Option<StateEffect>,
-    ) -> Result<Vec<PlanValue>> {
+    ) -> Result<()> {
         for input in inputs {
             if input.owner != self.owner
                 || self.values.get(input.id.0 as usize) != Some(&input.shape)
@@ -866,12 +914,23 @@
                     "foreign or malformed forward-plan value".into(),
                 ));
             }
-            if !input.contiguous && !matches!(&operation, PlanOperation::Contiguous) {
+            if !input.contiguous && !matches!(operation, PlanOperation::Contiguous) {
                 return Err(Error::State(
                     "plan operator requires explicit contiguous input".into(),
                 ));
             }
         }
+        Ok(())
+    }
+    fn record(
+        &mut self,
+        name: &str,
+        operation: PlanOperation,
+        inputs: &[PlanValue],
+        shapes: &[ValueShape],
+        state: Option<StateEffect>,
+    ) -> Result<Vec<PlanValue>> {
+        self.validate_record_inputs(&operation, inputs)?;
         let precision = self.result_precision(&operation, inputs)?;
         let elements: Vec<_> = shapes
             .iter()
@@ -1114,6 +1173,52 @@
         contract: CausalConvSiluContract,
     ) -> Result<CausalConvValues<PlanValue>> {
         gdn_conv::record(self, parameter, input, history, contract)
+    }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        dt: SourceParameter,
+        alpha: &PlanValue,
+        beta: &PlanValue,
+        contract: GdnCoefficientPreparationContract,
+    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
+        let heads = contract.validate(&dt, alpha.elements, beta.elements)?;
+        if self
+            .values
+            .len()
+            .checked_add(3)
+            .is_none_or(|count| count > u32::MAX as usize)
+        {
+            return Err(Error::Limit("GDN preparation value count overflow".into()));
+        }
+        let alpha_op = PlanOperation::TypedGdnAlphaPreparation {
+            dt: dt.clone(),
+            contract,
+        };
+        let beta_op = PlanOperation::TypedGdnBetaPreparation { heads, contract };
+        // Validate BOTH original values and every cold guard before recording.
+        self.validate_record_inputs(&alpha_op, &[*alpha])?;
+        self.validate_record_inputs(&beta_op, &[*beta])?;
+        self.result_precision(&alpha_op, &[*alpha])?;
+        self.result_precision(&beta_op, &[*beta])?;
+        let beta = self.one(
+            &format!("{}.beta-preparation", dt.name),
+            beta_op,
+            &[*beta],
+            ValueShape::Vector(heads),
+            None,
+        )?;
+        let values = self.record(
+            &format!("{}.alpha-preparation", dt.name),
+            alpha_op,
+            &[*alpha],
+            &[ValueShape::Vector(heads); 2],
+            None,
+        )?;
+        Ok(GdnCoefficientPreparationValues {
+            biased_alpha: values[0],
+            softplus: values[1],
+            beta,
+        })
     }
     fn embed(&mut self, tensor: SourceTensor, token: ParameterRef) -> Result<PlanValue> {
         self.source_tensor(&tensor)?;
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/coefficient_preparation.rs
@@ -0,0 +1,160 @@
+//! State-free original coefficient preparation; no exp/g/decay capability.
+use super::{QuantKind, SourceParameter};
+use crate::{Error, Result};
+
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum GdnCoefficientPreparationContract {
+    TorchCpuBf16Avx2,
+}
+impl GdnCoefficientPreparationContract {
+    pub fn validate(self, dt: &SourceParameter, alpha: usize, beta: usize) -> Result<usize> {
+        if !dt.source.is_bound() || dt.name.is_empty() {
+            return Err(Error::State(
+                "GDN preparation raw parameter source/name".into(),
+            ));
+        }
+        if !matches!(alpha, 16 | 48) || beta != alpha || dt.shape != [alpha as u64] {
+            return Err(Error::Malformed(
+                "GDN preparation one-call head/span".into(),
+            ));
+        }
+        let bytes = match dt.kind {
+            QuantKind::Bf16 => 2,
+            QuantKind::F32 => 4,
+            _ => {
+                return Err(Error::Unsupported(
+                    "GDN preparation raw dt_bias dtype".into(),
+                ));
+            }
+        };
+        if dt.bytes != alpha * bytes {
+            return Err(Error::Malformed(
+                "GDN preparation raw dt_bias byte span".into(),
+            ));
+        }
+        Ok(alpha)
+    }
+}
+
+pub struct GdnCoefficientPreparationValues<V> {
+    pub biased_alpha: V,
+    pub softplus: V,
+    pub beta: V,
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::accelerator::SourceIdentity;
+    use crate::accelerator::program::{
+        ActivationPrecision, EmbeddingContract, MathematicalBackend, MatrixAxes, PlanBuilder,
+        ProgramNumericalProfile, SourceTensor, ValueShape,
+    };
+    fn dt(source: SourceIdentity) -> SourceParameter {
+        SourceParameter {
+            source,
+            name: "synthetic.transport.dt_bias".into(),
+            kind: QuantKind::Bf16,
+            shape: vec![16],
+            bytes: 32,
+        }
+    }
+    fn embed(builder: &mut PlanBuilder, source: SourceIdentity) -> super::super::PlanValue {
+        builder
+            .embed_profiled(
+                SourceTensor {
+                    source,
+                    name: "synthetic.transport.embed".into(),
+                    kind: QuantKind::Bf16,
+                    rows: 1,
+                    columns: 16,
+                    axes: MatrixAxes::OutputInput,
+                },
+                PlanBuilder::inputs().token,
+                EmbeddingContract::Bf16OutputInput,
+            )
+            .unwrap()
+    }
+    #[test]
+    fn synthetic_builder_transport_keeps_distinct_precisions_and_no_state() {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        let mut builder = PlanBuilder::new_operator_slice(
+            source,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap();
+        let a = embed(&mut builder, source);
+        let b = embed(&mut builder, source);
+        let values = builder
+            .coefficient_preparation_profiled(
+                dt(source),
+                &a,
+                &b,
+                GdnCoefficientPreparationContract::TorchCpuBf16Avx2,
+            )
+            .unwrap();
+        assert_eq!(values.beta.precision(), ActivationPrecision::Bf16WidenedF32);
+        assert_eq!(values.softplus.precision(), ActivationPrecision::F32);
+        assert_eq!(values.biased_alpha.precision(), ActivationPrecision::F32);
+        let plan = builder.finish_operator_slice(values.softplus).unwrap();
+        assert_eq!(plan.plan().nodes().len(), 4);
+        assert!(plan.plan().required_domains().is_empty());
+        assert!(plan.plan().proposed_domains().is_empty());
+        assert_eq!(plan.plan().frontier_delta(), 0);
+        assert!(plan.plan().logits().is_none());
+        assert!(
+            plan.plan()
+                .values()
+                .iter()
+                .all(|v| matches!(v, ValueShape::Vector(16)))
+        );
+    }
+    #[test]
+    fn foreign_builder_input_refuses_before_recording_either_preparation_node() {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        let mut first = PlanBuilder::new_operator_slice(
+            source,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap();
+        let mut other = PlanBuilder::new_operator_slice(
+            source,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap();
+        let a = embed(&mut first, source);
+        let b = embed(&mut other, source);
+        assert!(
+            first
+                .coefficient_preparation_profiled(
+                    dt(source),
+                    &a,
+                    &b,
+                    GdnCoefficientPreparationContract::TorchCpuBf16Avx2
+                )
+                .is_err()
+        );
+        assert_eq!(
+            first.finish_operator_slice(a).unwrap().plan().nodes().len(),
+            1
+        );
+    }
+    #[test]
+    fn raw_parameter_dtype_bytes_and_span_are_checked() {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        let contract = GdnCoefficientPreparationContract::TorchCpuBf16Avx2;
+        let valid = dt(source);
+        assert_eq!(contract.validate(&valid, 16, 16).unwrap(), 16);
+        let mut wrong = valid.clone();
+        wrong.shape = vec![1, 16];
+        assert!(contract.validate(&wrong, 16, 16).is_err());
+        wrong = valid.clone();
+        wrong.kind = QuantKind::F16;
+        assert!(contract.validate(&wrong, 16, 16).is_err());
+        wrong = valid.clone();
+        wrong.bytes -= 1;
+        assert!(contract.validate(&wrong, 16, 16).is_err());
+        assert!(contract.validate(&valid, 32, 32).is_err());
+        assert!(contract.validate(&valid, 16, 48).is_err());
+    }
+}
--- a/crates/zerogravity-cpu/src/original_hf.rs
+++ b/crates/zerogravity-cpu/src/original_hf.rs
@@ -3,6 +3,7 @@
 //! learned matrix transpose or duplicate mathematical runtime is required.
 //! This does not enable complete trained Qwen3.8 inference.
 
+pub mod coefficient_preparation;
 pub mod conv;
 pub mod norm;
 pub mod recurrent;
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs
@@ -0,0 +1,222 @@
+//! Pinned original GDN coefficient PREPARATION only; no A_log exp/g/decay.
+//! Source modeling762feb6c lines617/619, using accepted sigmoid and softplus.
+use super::transcendentals::{
+    sigmoid_bf16_widened_f32_contiguous_span, softplus_beta1_threshold20_f32_contiguous16,
+};
+use crate::accelerator::SourceIdentity;
+use crate::quant::QuantKind;
+use crate::{Error, Result};
+use zerogravity_core::source::SourceTensorRef;
+
+/// Private stages only. The caller retains source/admission through this call.
+pub struct GdnCoefficientPreparation {
+    source: SourceIdentity,
+    biased_alpha: Vec<f32>,
+    softplus: Vec<f32>,
+    beta: Vec<f32>,
+}
+impl GdnCoefficientPreparation {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn biased_alpha(&self) -> &[f32] {
+        &self.biased_alpha
+    }
+    pub fn softplus(&self) -> &[f32] {
+        &self.softplus
+    }
+    pub fn beta(&self) -> &[f32] {
+        &self.beta
+    }
+    /// Private buffers, with no publication or admission claim.
+    pub fn into_parts(self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
+        (self.biased_alpha, self.softplus, self.beta)
+    }
+}
+
+/// One original [1,1,heads] contiguous a/b call, heads16 or48. Do not flatten
+/// independent calls: BF16 sigmoid's vector32 prefix/scalar tail is observable.
+/// Raw rank1 dt_bias is widened privately; no learned-source mutation/cache.
+/// Physical callers admit the complete private storage before entering.
+pub fn prepare_gdn_coefficients(
+    source: SourceIdentity,
+    dt_bias: SourceTensorRef<'_>,
+    alpha_bf16: &[f32],
+    beta_input_bf16: &[f32],
+) -> Result<GdnCoefficientPreparation> {
+    let heads = alpha_bf16.len();
+    if !source.is_bound()
+        || !matches!(heads, 16 | 48)
+        || beta_input_bf16.len() != heads
+        || dt_bias.shape != [heads as u64]
+    {
+        return Err(Error::Malformed(
+            "original GDN preparation source/head/span".into(),
+        ));
+    }
+    if !matches!(dt_bias.kind, QuantKind::Bf16 | QuantKind::F32) {
+        return Err(Error::Unsupported(
+            "original GDN preparation raw dt_bias dtype".into(),
+        ));
+    }
+    dt_bias.validate()?;
+    if alpha_bf16
+        .iter()
+        .chain(beta_input_bf16)
+        .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
+    {
+        return Err(Error::Malformed(
+            "original GDN preparation finite BF16 a/b".into(),
+        ));
+    }
+    let dt_f32 = match dt_bias.kind {
+        QuantKind::Bf16 => dt_bias
+            .packed_bytes
+            .chunks_exact(2)
+            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
+            .collect::<Vec<_>>(),
+        QuantKind::F32 => dt_bias
+            .packed_bytes
+            .chunks_exact(4)
+            .map(|b| f32::from_le_bytes(b.try_into().expect("validated raw F32 dt_bias")))
+            .collect(),
+        _ => unreachable!("validated dt_bias dtype"),
+    };
+    if dt_f32.iter().any(|v| !v.is_finite()) {
+        return Err(Error::Malformed(
+            "nonfinite original GDN raw dt_bias".into(),
+        ));
+    }
+    // Original line617 executes beta first; source line619 widens a before
+    // addition to raw dt_bias, then materializes FP32 softplus separately.
+    let beta = sigmoid_bf16_widened_f32_contiguous_span(source, beta_input_bf16)?;
+    let biased_alpha = alpha_bf16
+        .iter()
+        .zip(dt_f32)
+        .map(|(a, dt)| a + dt)
+        .collect::<Vec<_>>();
+    if biased_alpha.iter().any(|v| !v.is_finite()) {
+        return Err(Error::Malformed(
+            "nonfinite original GDN biased alpha".into(),
+        ));
+    }
+    let softplus = softplus_beta1_threshold20_f32_contiguous16(source, &biased_alpha)?;
+    Ok(GdnCoefficientPreparation {
+        source,
+        biased_alpha,
+        softplus,
+        beta,
+    })
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    fn source() -> SourceIdentity {
+        SourceIdentity::for_new_allocation().unwrap()
+    }
+    #[test]
+    fn linear_softplus_and_zero_beta_are_independent_exact_stages() {
+        for heads in [16, 48] {
+            let raw = vec![0u8; heads * 2];
+            let a = vec![21.0; heads];
+            let b = vec![0.0; heads];
+            let saved = raw.clone();
+            let out = prepare_gdn_coefficients(
+                source(),
+                SourceTensorRef {
+                    kind: QuantKind::Bf16,
+                    shape: &[heads as u64],
+                    packed_bytes: &raw,
+                },
+                &a,
+                &b,
+            )
+            .unwrap();
+            assert!(
+                out.biased_alpha()
+                    .iter()
+                    .all(|v| v.to_bits() == 21.0f32.to_bits())
+            );
+            assert!(
+                out.softplus()
+                    .iter()
+                    .all(|v| v.to_bits() == 21.0f32.to_bits())
+            );
+            assert!(out.beta().iter().all(|v| v.to_bits() == 0.5f32.to_bits()));
+            assert_eq!(raw, saved);
+        }
+    }
+    #[test]
+    fn raw_f32_dt_is_not_bf16_rounded_or_adjusted() {
+        let word = 0x3380_0001u32;
+        let raw = vec![word.to_le_bytes(); 16].concat();
+        let zero = [0.0; 16];
+        let out = prepare_gdn_coefficients(
+            source(),
+            SourceTensorRef {
+                kind: QuantKind::F32,
+                shape: &[16],
+                packed_bytes: &raw,
+            },
+            &zero,
+            &zero,
+        )
+        .unwrap();
+        assert!(out.biased_alpha().iter().all(|v| v.to_bits() == word));
+        assert_eq!(raw, vec![word.to_le_bytes(); 16].concat());
+    }
+    #[test]
+    fn source_raw_dtype_span_and_nonfinite_refuse() {
+        let raw = vec![0u8; 32];
+        let zero = [0.0; 16];
+        let dt = SourceTensorRef {
+            kind: QuantKind::Bf16,
+            shape: &[16],
+            packed_bytes: &raw,
+        };
+        assert!(prepare_gdn_coefficients(SourceIdentity::unbound(), dt, &zero, &zero).is_err());
+        assert!(prepare_gdn_coefficients(source(), dt, &zero[..15], &zero[..15]).is_err());
+        assert!(
+            prepare_gdn_coefficients(
+                source(),
+                SourceTensorRef {
+                    shape: &[1, 16],
+                    ..dt
+                },
+                &zero,
+                &zero
+            )
+            .is_err()
+        );
+        assert!(
+            prepare_gdn_coefficients(
+                source(),
+                SourceTensorRef {
+                    kind: QuantKind::F16,
+                    ..dt
+                },
+                &zero,
+                &zero
+            )
+            .is_err()
+        );
+        let mut bad = zero;
+        bad[1] = f32::from_bits(0x3f80_0001);
+        assert!(prepare_gdn_coefficients(source(), dt, &bad, &zero).is_err());
+        let mut bad_raw = raw.clone();
+        bad_raw[..2].copy_from_slice(&0x7fc1u16.to_le_bytes());
+        assert!(
+            prepare_gdn_coefficients(
+                source(),
+                SourceTensorRef {
+                    packed_bytes: &bad_raw,
+                    ..dt
+                },
+                &zero,
+                &zero
+            )
+            .is_err()
+        );
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
@@ -0,0 +1,261 @@
+//! Synthetic source/transport/lease guards only; never trained coefficients.
+use super::*;
+use crate::accelerator::program::{
+    GdnCoefficientPreparationContract, PlanBuilder, PlanOperation, ProgramNumericalProfile,
+};
+use crate::operator_slice::coefficient_preparation::GdnCoefficientPreparationInput;
+fn coefficient_source(fixture: &Fixture) -> Loaded {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    cfg["text_config"]["hidden_size"] = json!(256);
+    cfg["text_config"]["linear_num_value_heads"] = json!(16);
+    source_with_config(
+        fixture,
+        serde_json::to_vec(&cfg).unwrap(),
+        true,
+        |name, _, fallback| {
+            if name.ends_with(".dt_bias") {
+                0
+            } else {
+                fallback
+            }
+        },
+        |_| {},
+    )
+}
+fn raw(value: u16) -> Vec<u8> {
+    vec![value.to_le_bytes(); 16].concat()
+}
+fn input<'a>(
+    source: crate::accelerator::SourceIdentity,
+    a: &'a [u8],
+    b: &'a [u8],
+) -> GdnCoefficientPreparationInput<'a> {
+    GdnCoefficientPreparationInput::from_bf16_bytes(
+        source,
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        a,
+        b,
+    )
+    .unwrap()
+}
+#[test]
+fn synthetic_coefficient_stages_retain_two_aggregate_backings_and_source() {
+    let fixture = Fixture::new();
+    let loaded = coefficient_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let a = raw(0x41a8);
+    let b = raw(0); // a21, b0: public linear softplus and sigmoid0.
+    let before = reserved(&admission);
+    let output = operators
+        .execute_gdn_coefficient_preparation_slice(0, input(source.source_identity(), &a, &b), None)
+        .unwrap();
+    assert!(
+        output
+            .biased_alpha_f32()
+            .unwrap()
+            .iter()
+            .all(|v| v.to_bits() == 21.0f32.to_bits())
+    );
+    assert!(
+        output
+            .softplus_f32()
+            .unwrap()
+            .iter()
+            .all(|v| v.to_bits() == 21.0f32.to_bits())
+    );
+    assert!(
+        output
+            .beta_bf16()
+            .unwrap()
+            .iter()
+            .all(|v| v.to_bits() == 0.5f32.to_bits())
+    );
+    assert_eq!(
+        output.raw_dt_bias_parameter().kind,
+        zerogravity_core::quant::QuantKind::Bf16
+    );
+    assert_eq!(
+        output.unevaluated_raw_a_log_parameter().kind,
+        zerogravity_core::quant::QuantKind::F32
+    );
+    assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 34 * 1024 * 1024);
+    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 16]);
+    assert_eq!(output.beta_input_bf16().unwrap(), &[0.0; 16]);
+    drop(output);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn wrong_layer_foreign_input_cancel_and_pressure_refuse_without_leak() {
+    let fixture = Fixture::new();
+    let loaded = coefficient_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let zero = raw(0);
+    let before = reserved(&admission);
+    assert!(
+        operators
+            .execute_gdn_coefficient_preparation_slice(
+                3,
+                input(source.source_identity(), &zero, &zero),
+                None
+            )
+            .is_err()
+    );
+    assert!(
+        operators
+            .execute_gdn_coefficient_preparation_slice(
+                0,
+                input(
+                    crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
+                    &zero,
+                    &zero
+                ),
+                None
+            )
+            .is_err()
+    );
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        operators
+            .execute_gdn_coefficient_preparation_slice(
+                0,
+                input(source.source_identity(), &zero, &zero),
+                Some(&cancelled)
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before);
+    let remaining = admission
+        .arbiter
+        .snapshot(&admission.resource)
+        .unwrap()
+        .remaining;
+    let pressure = source.reserve_workspace(remaining as usize).unwrap();
+    let pressured = reserved(&admission);
+    assert!(
+        operators
+            .execute_gdn_coefficient_preparation_slice(
+                0,
+                input(source.source_identity(), &zero, &zero),
+                None
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), pressured);
+    drop(pressure);
+    assert_eq!(reserved(&admission), before);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn raw_a_log_alias_and_unaware_provider_refuse_before_callback_or_output() {
+    struct Provider(AtomicBool);
+    impl crate::accelerator::MatVecKernel for Provider {
+        fn try_execute(
+            &self,
+            _: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            self.0.store(true, Ordering::Relaxed);
+            panic!("unaware coefficient provider invoked");
+        }
+    }
+    let fixture = Fixture::new();
+    let loaded = coefficient_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let zero = raw(0);
+    let (a, b) = {
+        let math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
+        math.import_coefficient_preparation_input(input(source.source_identity(), &zero, &zero), 16)
+            .unwrap()
+    };
+    let before = reserved(&admission);
+    let provider = Provider(AtomicBool::new(false));
+    {
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(
+            &operators,
+            None,
+            Some(&provider),
+        );
+        assert!(
+            math.coefficient_preparation_profiled(
+                operators
+                    .parameter_descriptor(QwenParameterRole::GdnDt(0))
+                    .unwrap(),
+                &a,
+                &b,
+                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+        );
+    }
+    {
+        let mut math =
+            crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
+        assert!(
+            math.coefficient_preparation_profiled(
+                operators
+                    .parameter_descriptor(QwenParameterRole::GdnA(0))
+                    .unwrap(),
+                &a,
+                &b,
+                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+        );
+    }
+    assert!(!provider.0.load(Ordering::Relaxed));
+    assert_eq!(reserved(&admission), before);
+    drop(a);
+    drop(b);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn plan_appends_preparation_without_exp_state_token_or_producer_invention() {
+    let fixture = Fixture::new();
+    let loaded = coefficient_source(&fixture);
+    let (model, _, _) = bind(loaded);
+    let mut builder =
+        PlanBuilder::new_operator_slice(model.source_identity(), model.program_numerical_profile())
+            .unwrap();
+    // Synthetic same-builder views are transport ONLY, never actual trained a/b.
+    let embed = builder
+        .embed_profiled(
+            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    let a = builder.view(&embed, TensorView::vector(0, 16)).unwrap();
+    let b = builder.view(&embed, TensorView::vector(16, 16)).unwrap();
+    let values = model
+        .append_gdn_coefficient_preparation_slice(&mut builder, 0, &a, &b)
+        .unwrap();
+    let plan = builder.finish_operator_slice(values.softplus).unwrap();
+    assert_eq!(plan.plan().nodes().len(), 5);
+    assert_eq!(
+        plan.plan()
+            .nodes()
+            .iter()
+            .filter(|n| matches!(
+                n.operation,
+                PlanOperation::TypedGdnAlphaPreparation { .. }
+                    | PlanOperation::TypedGdnBetaPreparation { .. }
+            ))
+            .count(),
+        2
+    );
+    assert!(plan.plan().required_domains().is_empty());
+    assert!(plan.plan().proposed_domains().is_empty());
+    assert!(plan.plan().logits().is_none());
+    assert_eq!(plan.plan().frontier_delta(), 0);
+    assert!(model.new_session().is_err());
+    assert!(model.forward_plan(8, true, true).is_err());
+}
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -200,6 +200,55 @@
         normalized_query,
         normalized_key,
     })
+}
+
+/// Pure coefficient preparation only; no exp, g/decay, scaling or state effects.
+pub(crate) fn gdn_coefficient_preparation_slice<B: MathematicalBackend>(
+    source: &(impl crate::operator_slice::OperatorSliceSource + ?Sized),
+    math: &mut B,
+    layer: usize,
+    alpha: &B::Value,
+    beta_input: &B::Value,
+) -> Result<crate::accelerator::program::GdnCoefficientPreparationValues<B::Value>> {
+    use crate::accelerator::program::{ActivationPrecision, GdnCoefficientPreparationContract};
+    let roles = crate::operator_slice::coefficient_preparation::coefficient_roles(source, layer)?;
+    for input in [alpha, beta_input] {
+        expect_elements(
+            math,
+            input,
+            roles.heads,
+            "coefficient preparation actual head count",
+        )?;
+        if math.precision(input) != ActivationPrecision::Bf16WidenedF32 {
+            return Err(Error::Unsupported(
+                "coefficient preparation requires original BF16 a/b".into(),
+            ));
+        }
+    }
+    let values = math.coefficient_preparation_profiled(
+        roles.dt,
+        alpha,
+        beta_input,
+        GdnCoefficientPreparationContract::TorchCpuBf16Avx2,
+    )?;
+    for (value, precision) in [
+        (&values.biased_alpha, ActivationPrecision::F32),
+        (&values.softplus, ActivationPrecision::F32),
+        (&values.beta, ActivationPrecision::Bf16WidenedF32),
+    ] {
+        expect_elements(
+            math,
+            value,
+            roles.heads,
+            "coefficient preparation output head count",
+        )?;
+        if math.precision(value) != precision {
+            return Err(Error::State(
+                "coefficient preparation output precision".into(),
+            ));
+        }
+    }
+    Ok(values)
 }
 
 fn expect_elements<B: MathematicalBackend>(
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -19,6 +19,9 @@
 use std::sync::Arc;
 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
 use zerogravity_artifacts::source_weights::WindowWorkspace;
+
+#[path = "host_math/coefficient_preparation.rs"]
+mod coefficient_preparation;
 
 struct HostStorage {
     values: Vec<f32>,
@@ -526,6 +529,15 @@
                 "causal convolution requires leased original windows".into(),
             ))
         }
+    }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        dt: SourceParameter,
+        alpha: &HostValue,
+        beta: &HostValue,
+        contract: crate::accelerator::program::GdnCoefficientPreparationContract,
+    ) -> Result<crate::accelerator::program::GdnCoefficientPreparationValues<HostValue>> {
+        self.coefficient_preparation_closed(dt, alpha, beta, contract)
     }
     fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<HostValue> {
         self.legacy_precision(&[])?;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs
@@ -0,0 +1,147 @@
+//! One existing HostMath adapter; source/leases retained, no new engine.
+use super::*;
+use crate::accelerator::program::{
+    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues,
+};
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+use crate::operator_slice::coefficient_preparation::GdnCoefficientPreparationInput;
+use crate::operator_slice::coefficient_preparation::coefficient_roles;
+
+impl HostMath<'_, '_, '_> {
+    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+    pub(crate) fn import_coefficient_preparation_input(
+        &self,
+        input: GdnCoefficientPreparationInput<'_>,
+        heads: usize,
+    ) -> Result<(HostValue, HostValue)> {
+        if input.validate()? != heads
+            || input.source_identity() != self.source.source_identity()
+            || input.numerical_profile() != self.source.program_numerical_profile()
+        {
+            return Err(Error::State(
+                "coefficient preparation imported source/profile/heads".into(),
+            ));
+        }
+        if self.region_kernel.is_some() {
+            return Err(Error::Unsupported(
+                "coefficient preparation unaware input provider".into(),
+            ));
+        }
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let workspace = self.prepare_output(None)?; // BEFORE both private copies.
+        let decode = |raw: &[u8]| {
+            raw.chunks_exact(2)
+                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
+                .collect::<Vec<_>>()
+        };
+        let mut values = Vec::with_capacity(2);
+        for raw in [input.alpha_bytes(), input.beta_bytes()] {
+            let value = HostValue::typed(
+                decode(raw),
+                ActivationPrecision::Bf16WidenedF32,
+                input.source_identity(),
+                input.numerical_profile(),
+            )?;
+            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+            let value = value.retain_workspace(workspace.clone())?;
+            values.push(value);
+        }
+        Ok((values.remove(0), values.remove(0)))
+    }
+    pub(super) fn coefficient_preparation_closed(
+        &self,
+        dt: SourceParameter,
+        alpha: &HostValue,
+        beta: &HostValue,
+        contract: GdnCoefficientPreparationContract,
+    ) -> Result<GdnCoefficientPreparationValues<HostValue>> {
+        if self.region_kernel.is_some() {
+            return Err(Error::Unsupported(
+                "physical provider lacks original coefficient preparation contract".into(),
+            ));
+        }
+        let source = self.source.source_identity();
+        let profile = self.source.program_numerical_profile();
+        if profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+            || [alpha, beta].iter().any(|v| {
+                v.precision != ActivationPrecision::Bf16WidenedF32
+                    || v.source != Some(source)
+                    || v.profile != Some(profile)
+            })
+        {
+            return Err(Error::Unsupported(
+                "host coefficient preparation precision/source/profile".into(),
+            ));
+        }
+        let heads = contract.validate(&dt, alpha.elements, beta.elements)?;
+        let mut matching_role = false;
+        for (layer, kind) in self.source.metadata().geometry().layers.iter().enumerate() {
+            if *kind != crate::model::LayerKind::LinearAttention {
+                continue;
+            }
+            let roles = coefficient_roles(self.source.metadata(), layer)?;
+            if roles.dt == dt && roles.heads == heads {
+                matching_role = true;
+                break;
+            }
+        }
+        if !matching_role {
+            return Err(Error::State(
+                "host coefficient preparation original dt_bias role".into(),
+            ));
+        }
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let workspace = {
+            if let HostSource::Windows(owner) = &self.source {
+                let beta_workspace = beta.data.workspace.as_ref().ok_or_else(|| {
+                    Error::State("coefficient beta has no original physical backing".into())
+                })?;
+                if !owner.source().same_issuer(beta_workspace.source()) {
+                    return Err(Error::State(
+                        "coefficient beta source issuer changed".into(),
+                    ));
+                }
+            }
+            self.prepare_output(Some(alpha))? // Shared aggregate for ALL3 outputs.
+        };
+        let prepared = match &self.source {
+            HostSource::Loaded(owner) => {
+                dt.validate_source(*owner)?;
+                let raw = zerogravity_core::source::TensorSource::tensor(*owner, &dt.name)?;
+                zerogravity_cpu::original_hf::coefficient_preparation::prepare_gdn_coefficients(
+                    source,
+                    raw,
+                    alpha.as_slice()?,
+                    beta.as_slice()?,
+                )?
+            }
+            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+            HostSource::Windows(owner) => owner.prepare_coefficient_stages(
+                &dt,
+                alpha.as_slice()?,
+                beta.as_slice()?,
+                self.cancelled,
+            )?,
+        };
+        if prepared.source_identity() != source {
+            return Err(Error::State("coefficient preparation result source".into()));
+        }
+        let (biased, softplus, beta) = prepared.into_parts();
+        let mut outputs = Vec::with_capacity(3);
+        for (values, precision) in [
+            (biased, ActivationPrecision::F32),
+            (softplus, ActivationPrecision::F32),
+            (beta, ActivationPrecision::Bf16WidenedF32),
+        ] {
+            let value = HostValue::typed(values, precision, source, profile)?;
+            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+            let value = value.retain_workspace(workspace.clone())?;
+            outputs.push(value);
+        }
+        Ok(GdnCoefficientPreparationValues {
+            biased_alpha: outputs.remove(0),
+            softplus: outputs.remove(0),
+            beta: outputs.remove(0),
+        })
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -228,6 +228,10 @@
 #[path = "operator_slice_tests.rs"]
 mod tests;
 
+/// Pure source coefficient preparation; no g/decay/exp or state capability.
+#[path = "operator_slice/coefficient_preparation.rs"]
+pub mod coefficient_preparation;
+
 /// Pure attention projections/QK norms; no rotary/cache/session admission.
 #[path = "attention_slice.rs"]
 pub mod attention;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs
@@ -0,0 +1,283 @@
+//! Source-bound coefficient PREPARATION on caller-owned a/b; no g/decay/exp.
+use super::OperatorSliceSource;
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{
+    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues, MatrixAxes, PlanBuilder,
+    PlanValue, ProgramNumericalProfile, SourceParameter,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, LoadedQwenModel, Result};
+use zerogravity_core::quant::QuantKind;
+
+pub(crate) struct CoefficientPreparationRoles {
+    pub(crate) heads: usize,
+    pub(crate) dt: SourceParameter,
+    pub(crate) a_log: SourceParameter,
+}
+/// Immutable complete role descriptors; A_log is not evaluated by this slice.
+pub(crate) fn coefficient_roles(
+    source: &(impl OperatorSliceSource + ?Sized),
+    layer: usize,
+) -> Result<CoefficientPreparationRoles> {
+    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || !source.source_identity().is_bound()
+        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
+    {
+        return Err(Error::Unsupported(
+            "coefficient preparation original LinearAttention source/profile".into(),
+        ));
+    }
+    let heads = usize::try_from(source.geometry().recurrent_value_heads.unwrap_or(0))
+        .map_err(|_| Error::Limit("coefficient heads".into()))?;
+    let hidden = usize::try_from(source.geometry().hidden_size)
+        .map_err(|_| Error::Limit("coefficient hidden".into()))?;
+    if !matches!(heads, 16 | 48) || !matches!(hidden, 256 | 1024 | 2048 | 5120) {
+        return Err(Error::Unsupported(
+            "coefficient preparation closed source geometry".into(),
+        ));
+    }
+    let namespace = format!("model.language_model.layers.{layer}.linear_attn");
+    for (role, suffix) in [
+        (QwenMatrixRole::GdnAlpha(layer), "in_proj_a"),
+        (QwenMatrixRole::GdnBeta(layer), "in_proj_b"),
+    ] {
+        let tensor = source.matrix_descriptor(role)?;
+        if tensor.source != source.source_identity()
+            || tensor.name != format!("{namespace}.{suffix}.weight")
+            || tensor.kind != QuantKind::Bf16
+            || tensor.axes != MatrixAxes::OutputInput
+            || tensor.rows != heads
+            || tensor.columns != hidden
+        {
+            return Err(Error::State(
+                "coefficient preparation original a/b full matrix role".into(),
+            ));
+        }
+    }
+    let dt = source.parameter_descriptor(QwenParameterRole::GdnDt(layer))?;
+    let a_log = source.parameter_descriptor(QwenParameterRole::GdnA(layer))?;
+    if dt.source != source.source_identity()
+        || dt.name != format!("{namespace}.dt_bias")
+        || a_log.source != source.source_identity()
+        || a_log.name != format!("{namespace}.A_log")
+        || a_log.kind != QuantKind::F32
+        || a_log.shape != [heads as u64]
+        || a_log.bytes != heads * 4
+    {
+        return Err(Error::State(
+            "coefficient preparation original raw A_log/dt_bias roles".into(),
+        ));
+    }
+    GdnCoefficientPreparationContract::TorchCpuBf16Avx2.validate(&dt, heads, heads)?;
+    Ok(CoefficientPreparationRoles { heads, dt, a_log })
+}
+
+impl LoadedQwenModel {
+    /// Append only beta/biased-alpha/softplus to caller's original plan values.
+    pub fn append_gdn_coefficient_preparation_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        a: &PlanValue,
+        b: &PlanValue,
+    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
+        self.require_operator_slice()?;
+        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
+    }
+}
+
+/// External actual source/operator a/b bytes; caller authenticates provenance.
+/// No hidden state, matrix producer, token or frontier is asserted by import.
+#[derive(Clone, Copy)]
+pub struct GdnCoefficientPreparationInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    alpha: &'a [u8],
+    beta: &'a [u8],
+}
+impl<'a> GdnCoefficientPreparationInput<'a> {
+    pub fn from_bf16_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        alpha: &'a [u8],
+        beta: &'a [u8],
+    ) -> Result<Self> {
+        let input = Self {
+            source,
+            profile,
+            alpha,
+            beta,
+        };
+        input.validate()?;
+        Ok(input)
+    }
+    pub fn source_identity(self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    pub fn alpha_bytes(self) -> &'a [u8] {
+        self.alpha
+    }
+    pub fn beta_bytes(self) -> &'a [u8] {
+        self.beta
+    }
+    pub(crate) fn validate(self) -> Result<usize> {
+        if !self.source.is_bound()
+            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::State(
+                "coefficient preparation private input source/profile".into(),
+            ));
+        }
+        if !matches!(self.alpha.len(), 32 | 96)
+            || self.beta.len() != self.alpha.len()
+            || self
+                .alpha
+                .chunks_exact(2)
+                .chain(self.beta.chunks_exact(2))
+                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
+        {
+            return Err(Error::Malformed(
+                "coefficient preparation one finite BF16 a/b call".into(),
+            ));
+        }
+        Ok(self.alpha.len() / 2)
+    }
+}
+
+#[cfg(feature = "cpu")]
+pub struct GdnCoefficientPreparationOutput {
+    source: SourceIdentity,
+    layer: usize,
+    dt: SourceParameter,
+    a_log: SourceParameter,
+    alpha: crate::host_math::HostValue,
+    beta_input: crate::host_math::HostValue,
+    values: GdnCoefficientPreparationValues<crate::host_math::HostValue>,
+}
+#[cfg(feature = "cpu")]
+impl GdnCoefficientPreparationOutput {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn layer(&self) -> usize {
+        self.layer
+    }
+    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    }
+    pub fn raw_dt_bias_parameter(&self) -> &SourceParameter {
+        &self.dt
+    }
+    pub fn unevaluated_raw_a_log_parameter(&self) -> &SourceParameter {
+        &self.a_log
+    }
+    pub fn alpha_bf16(&self) -> Result<&[f32]> {
+        self.alpha.as_slice()
+    }
+    pub fn beta_input_bf16(&self) -> Result<&[f32]> {
+        self.beta_input.as_slice()
+    }
+    pub fn biased_alpha_f32(&self) -> Result<&[f32]> {
+        self.values.biased_alpha.as_slice()
+    }
+    pub fn softplus_f32(&self) -> Result<&[f32]> {
+        self.values.softplus.as_slice()
+    }
+    pub fn beta_bf16(&self) -> Result<&[f32]> {
+        self.values.beta.as_slice()
+    }
+}
+
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    pub fn append_gdn_coefficient_preparation_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        a: &PlanValue,
+        b: &PlanValue,
+    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
+        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_gdn_coefficient_preparation_slice(
+        &self,
+        layer: usize,
+        input: GdnCoefficientPreparationInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<GdnCoefficientPreparationOutput> {
+        let roles = coefficient_roles(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let (alpha, beta_input) = math.import_coefficient_preparation_input(input, roles.heads)?;
+        let values = crate::forward::gdn_coefficient_preparation_slice(
+            self,
+            &mut math,
+            layer,
+            &alpha,
+            &beta_input,
+        )?;
+        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State(
+                "window coefficient preparation cancelled".into(),
+            ));
+        }
+        Ok(GdnCoefficientPreparationOutput {
+            source: self.source_identity(),
+            layer,
+            dt: roles.dt,
+            a_log: roles.a_log,
+            alpha,
+            beta_input,
+            values,
+        })
+    }
+    #[cfg(feature = "cpu")]
+    pub(crate) fn prepare_coefficient_stages(
+        &self,
+        dt: &SourceParameter,
+        alpha: &[f32],
+        beta: &[f32],
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<zerogravity_cpu::original_hf::coefficient_preparation::GdnCoefficientPreparation>
+    {
+        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State(
+                "window coefficient raw parameter cancelled".into(),
+            ));
+        }
+        let raw = self.source().parameter(&dt.name)?;
+        raw.validate_source_parameter(dt)?;
+        let window = match cancelled {
+            Some(flag) => {
+                self.source()
+                    .read_parameter_cancellable(&raw, 0..raw.elements(), flag)?
+            }
+            None => self.source().read_parameter(&raw, 0..raw.elements())?,
+        };
+        if !window.is_complete() || window.element_range() != (0..raw.elements()) {
+            return Err(Error::State(
+                "window coefficient complete raw dt_bias".into(),
+            ));
+        }
+        let result =
+            zerogravity_cpu::original_hf::coefficient_preparation::prepare_gdn_coefficients(
+                self.source_identity(),
+                zerogravity_core::source::SourceTensorRef {
+                    kind: raw.tensor().kind,
+                    shape: &raw.tensor().shape,
+                    packed_bytes: window.as_slice(),
+                },
+                alpha,
+                beta,
+            )?;
+        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State(
+                "window coefficient cancelled after synchronous math".into(),
+            ));
+        }
+        Ok(result)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -382,3 +382,6 @@
 
 #[path = "gdn_conv_window_tests.rs"]
 mod gdn_conv;
+
+#[path = "coefficient_preparation_tests.rs"]
+mod coefficient_preparation;

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/frozen/WIRING.patch

Original UTF-8 bytes: 11500; SHA256 `66d4df8be9668f1e3f338454acff12d36b4066750a9737d4e95aa3d98c960d50`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -14,6 +14,10 @@
 
 mod gdn_conv;
 pub use gdn_conv::{CausalConvSiluContract, CausalConvValues};
+mod coefficient_preparation;
+pub use coefficient_preparation::{
+    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues,
+};
 mod typed;
 pub use typed::{
     ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
@@ -294,6 +298,17 @@
             "backend does not implement explicit causal convolution SiLU contract".into(),
         ))
     }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        _dt: SourceParameter,
+        _alpha: &Self::Value,
+        _beta: &Self::Value,
+        _contract: GdnCoefficientPreparationContract,
+    ) -> Result<GdnCoefficientPreparationValues<Self::Value>> {
+        Err(Error::Unsupported(
+            "backend does not implement original GDN coefficient preparation".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -474,6 +489,14 @@
     TypedOrdinaryRms {
         parameter: SourceParameter,
         contract: OrdinaryRmsContract,
+    },
+    TypedGdnAlphaPreparation {
+        dt: SourceParameter,
+        contract: GdnCoefficientPreparationContract,
+    },
+    TypedGdnBetaPreparation {
+        heads: usize,
+        contract: GdnCoefficientPreparationContract,
     },
     Embed {
         tensor: SourceTensor,
@@ -822,6 +845,32 @@
                 parameter,
                 contract,
             } => gdn_conv::result_precision(self, parameter, *contract, inputs),
+            PlanOperation::TypedGdnAlphaPreparation { dt, contract } => {
+                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+                    || dt.source != self.source
+                    || inputs.len() != 1
+                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
+                {
+                    return Err(Error::Unsupported(
+                        "GDN preparation alpha profile/source/precision".into(),
+                    ));
+                }
+                contract.validate(dt, inputs[0].elements, inputs[0].elements)?;
+                Ok(ActivationPrecision::F32)
+            }
+            PlanOperation::TypedGdnBetaPreparation { heads, .. } => {
+                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+                    || !matches!(*heads, 16 | 48)
+                    || inputs.len() != 1
+                    || inputs[0].elements != *heads
+                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
+                {
+                    return Err(Error::Unsupported(
+                        "GDN preparation beta profile/span/precision".into(),
+                    ));
+                }
+                Ok(ActivationPrecision::Bf16WidenedF32)
+            }
             PlanOperation::View(_) | PlanOperation::Contiguous => inputs
                 .first()
                 .map(|value| value.precision)
@@ -839,6 +888,8 @@
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
+                        | PlanOperation::TypedGdnAlphaPreparation { .. }
+                        | PlanOperation::TypedGdnBetaPreparation { .. }
                 ) =>
             {
                 Ok(ActivationPrecision::F32)
@@ -848,14 +899,11 @@
             )),
         }
     }
-    fn record(
-        &mut self,
-        name: &str,
-        operation: PlanOperation,
+    fn validate_record_inputs(
+        &self,
+        operation: &PlanOperation,
         inputs: &[PlanValue],
-        shapes: &[ValueShape],
-        state: Option<StateEffect>,
-    ) -> Result<Vec<PlanValue>> {
+    ) -> Result<()> {
         for input in inputs {
             if input.owner != self.owner
                 || self.values.get(input.id.0 as usize) != Some(&input.shape)
@@ -866,12 +914,23 @@
                     "foreign or malformed forward-plan value".into(),
                 ));
             }
-            if !input.contiguous && !matches!(&operation, PlanOperation::Contiguous) {
+            if !input.contiguous && !matches!(operation, PlanOperation::Contiguous) {
                 return Err(Error::State(
                     "plan operator requires explicit contiguous input".into(),
                 ));
             }
         }
+        Ok(())
+    }
+    fn record(
+        &mut self,
+        name: &str,
+        operation: PlanOperation,
+        inputs: &[PlanValue],
+        shapes: &[ValueShape],
+        state: Option<StateEffect>,
+    ) -> Result<Vec<PlanValue>> {
+        self.validate_record_inputs(&operation, inputs)?;
         let precision = self.result_precision(&operation, inputs)?;
         let elements: Vec<_> = shapes
             .iter()
@@ -1114,6 +1173,52 @@
         contract: CausalConvSiluContract,
     ) -> Result<CausalConvValues<PlanValue>> {
         gdn_conv::record(self, parameter, input, history, contract)
+    }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        dt: SourceParameter,
+        alpha: &PlanValue,
+        beta: &PlanValue,
+        contract: GdnCoefficientPreparationContract,
+    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
+        let heads = contract.validate(&dt, alpha.elements, beta.elements)?;
+        if self
+            .values
+            .len()
+            .checked_add(3)
+            .is_none_or(|count| count > u32::MAX as usize)
+        {
+            return Err(Error::Limit("GDN preparation value count overflow".into()));
+        }
+        let alpha_op = PlanOperation::TypedGdnAlphaPreparation {
+            dt: dt.clone(),
+            contract,
+        };
+        let beta_op = PlanOperation::TypedGdnBetaPreparation { heads, contract };
+        // Validate BOTH original values and every cold guard before recording.
+        self.validate_record_inputs(&alpha_op, &[*alpha])?;
+        self.validate_record_inputs(&beta_op, &[*beta])?;
+        self.result_precision(&alpha_op, &[*alpha])?;
+        self.result_precision(&beta_op, &[*beta])?;
+        let beta = self.one(
+            &format!("{}.beta-preparation", dt.name),
+            beta_op,
+            &[*beta],
+            ValueShape::Vector(heads),
+            None,
+        )?;
+        let values = self.record(
+            &format!("{}.alpha-preparation", dt.name),
+            alpha_op,
+            &[*alpha],
+            &[ValueShape::Vector(heads); 2],
+            None,
+        )?;
+        Ok(GdnCoefficientPreparationValues {
+            biased_alpha: values[0],
+            softplus: values[1],
+            beta,
+        })
     }
     fn embed(&mut self, tensor: SourceTensor, token: ParameterRef) -> Result<PlanValue> {
         self.source_tensor(&tensor)?;
--- a/crates/zerogravity-cpu/src/original_hf.rs
+++ b/crates/zerogravity-cpu/src/original_hf.rs
@@ -3,6 +3,7 @@
 //! learned matrix transpose or duplicate mathematical runtime is required.
 //! This does not enable complete trained Qwen3.8 inference.
 
+pub mod coefficient_preparation;
 pub mod conv;
 pub mod norm;
 pub mod recurrent;
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -200,6 +200,55 @@
         normalized_query,
         normalized_key,
     })
+}
+
+/// Pure coefficient preparation only; no exp, g/decay, scaling or state effects.
+pub(crate) fn gdn_coefficient_preparation_slice<B: MathematicalBackend>(
+    source: &(impl crate::operator_slice::OperatorSliceSource + ?Sized),
+    math: &mut B,
+    layer: usize,
+    alpha: &B::Value,
+    beta_input: &B::Value,
+) -> Result<crate::accelerator::program::GdnCoefficientPreparationValues<B::Value>> {
+    use crate::accelerator::program::{ActivationPrecision, GdnCoefficientPreparationContract};
+    let roles = crate::operator_slice::coefficient_preparation::coefficient_roles(source, layer)?;
+    for input in [alpha, beta_input] {
+        expect_elements(
+            math,
+            input,
+            roles.heads,
+            "coefficient preparation actual head count",
+        )?;
+        if math.precision(input) != ActivationPrecision::Bf16WidenedF32 {
+            return Err(Error::Unsupported(
+                "coefficient preparation requires original BF16 a/b".into(),
+            ));
+        }
+    }
+    let values = math.coefficient_preparation_profiled(
+        roles.dt,
+        alpha,
+        beta_input,
+        GdnCoefficientPreparationContract::TorchCpuBf16Avx2,
+    )?;
+    for (value, precision) in [
+        (&values.biased_alpha, ActivationPrecision::F32),
+        (&values.softplus, ActivationPrecision::F32),
+        (&values.beta, ActivationPrecision::Bf16WidenedF32),
+    ] {
+        expect_elements(
+            math,
+            value,
+            roles.heads,
+            "coefficient preparation output head count",
+        )?;
+        if math.precision(value) != precision {
+            return Err(Error::State(
+                "coefficient preparation output precision".into(),
+            ));
+        }
+    }
+    Ok(values)
 }
 
 fn expect_elements<B: MathematicalBackend>(
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -19,6 +19,9 @@
 use std::sync::Arc;
 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
 use zerogravity_artifacts::source_weights::WindowWorkspace;
+
+#[path = "host_math/coefficient_preparation.rs"]
+mod coefficient_preparation;
 
 struct HostStorage {
     values: Vec<f32>,
@@ -526,6 +529,15 @@
                 "causal convolution requires leased original windows".into(),
             ))
         }
+    }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        dt: SourceParameter,
+        alpha: &HostValue,
+        beta: &HostValue,
+        contract: crate::accelerator::program::GdnCoefficientPreparationContract,
+    ) -> Result<crate::accelerator::program::GdnCoefficientPreparationValues<HostValue>> {
+        self.coefficient_preparation_closed(dt, alpha, beta, contract)
     }
     fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<HostValue> {
         self.legacy_precision(&[])?;
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -228,6 +228,10 @@
 #[path = "operator_slice_tests.rs"]
 mod tests;
 
+/// Pure source coefficient preparation; no g/decay/exp or state capability.
+#[path = "operator_slice/coefficient_preparation.rs"]
+pub mod coefficient_preparation;
+
 /// Pure attention projections/QK norms; no rotary/cache/session admission.
 #[path = "attention_slice.rs"]
 pub mod attention;
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -382,3 +382,6 @@
 
 #[path = "gdn_conv_window_tests.rs"]
 mod gdn_conv;
+
+#[path = "coefficient_preparation_tests.rs"]
+mod coefficient_preparation;

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/frozen/GPU_REFUSAL.patch

Original UTF-8 bytes: 1863; SHA256 `2b688ff898b6c3df5d9a83b21f6a574971f7399ec0d8dac1d50c65d0e6f4fbb6`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -348,6 +348,8 @@
                     PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
+                        | PlanOperation::TypedGdnAlphaPreparation { .. }
+                        | PlanOperation::TypedGdnBetaPreparation { .. }
                 )
             })
         {
@@ -1429,7 +1431,9 @@
             match &node.operation {
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
-                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedOrdinaryRms { .. }
+                | PlanOperation::TypedGdnAlphaPreparation { .. }
+                | PlanOperation::TypedGdnBetaPreparation { .. } => Err(Error::Unsupported(
                     "resident typed operation has no qualified lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -838,7 +838,9 @@
             match &node.operation {
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
-                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedOrdinaryRms { .. }
+                | PlanOperation::TypedGdnAlphaPreparation { .. }
+                | PlanOperation::TypedGdnBetaPreparation { .. } => Err(Error::Unsupported(
                     "resident batch has no qualified typed lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/frozen/NEW_DELTA.patch

Original UTF-8 bytes: 51428; SHA256 `26935bcd7af4cde53cf386a398eb569e501c7d75e660444b995290de85394671`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -12,6 +12,10 @@
 use std::sync::Arc;
 use std::sync::atomic::{AtomicU64, Ordering};
 
+mod coefficient_preparation;
+pub use coefficient_preparation::{
+    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues,
+};
 mod typed;
 pub use typed::{
     ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
@@ -280,6 +284,17 @@
             "backend does not implement explicit ordinary RMS contract".into(),
         ))
     }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        _dt: SourceParameter,
+        _alpha: &Self::Value,
+        _beta: &Self::Value,
+        _contract: GdnCoefficientPreparationContract,
+    ) -> Result<GdnCoefficientPreparationValues<Self::Value>> {
+        Err(Error::Unsupported(
+            "backend does not implement original GDN coefficient preparation".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -456,6 +471,14 @@
     TypedOrdinaryRms {
         parameter: SourceParameter,
         contract: OrdinaryRmsContract,
+    },
+    TypedGdnAlphaPreparation {
+        dt: SourceParameter,
+        contract: GdnCoefficientPreparationContract,
+    },
+    TypedGdnBetaPreparation {
+        heads: usize,
+        contract: GdnCoefficientPreparationContract,
     },
     Embed {
         tensor: SourceTensor,
@@ -800,6 +823,32 @@
                 contract.validate(parameter, inputs[0].elements)?;
                 Ok(ActivationPrecision::Bf16WidenedF32)
             }
+            PlanOperation::TypedGdnAlphaPreparation { dt, contract } => {
+                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+                    || dt.source != self.source
+                    || inputs.len() != 1
+                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
+                {
+                    return Err(Error::Unsupported(
+                        "GDN preparation alpha profile/source/precision".into(),
+                    ));
+                }
+                contract.validate(dt, inputs[0].elements, inputs[0].elements)?;
+                Ok(ActivationPrecision::F32)
+            }
+            PlanOperation::TypedGdnBetaPreparation { heads, .. } => {
+                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+                    || !matches!(*heads, 16 | 48)
+                    || inputs.len() != 1
+                    || inputs[0].elements != *heads
+                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
+                {
+                    return Err(Error::Unsupported(
+                        "GDN preparation beta profile/span/precision".into(),
+                    ));
+                }
+                Ok(ActivationPrecision::Bf16WidenedF32)
+            }
             PlanOperation::View(_) | PlanOperation::Contiguous => inputs
                 .first()
                 .map(|value| value.precision)
@@ -816,6 +865,8 @@
                     PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
+                        | PlanOperation::TypedGdnAlphaPreparation { .. }
+                        | PlanOperation::TypedGdnBetaPreparation { .. }
                 ) =>
             {
                 Ok(ActivationPrecision::F32)
@@ -825,14 +876,11 @@
             )),
         }
     }
-    fn record(
-        &mut self,
-        name: &str,
-        operation: PlanOperation,
+    fn validate_record_inputs(
+        &self,
+        operation: &PlanOperation,
         inputs: &[PlanValue],
-        shapes: &[ValueShape],
-        state: Option<StateEffect>,
-    ) -> Result<Vec<PlanValue>> {
+    ) -> Result<()> {
         for input in inputs {
             if input.owner != self.owner
                 || self.values.get(input.id.0 as usize) != Some(&input.shape)
@@ -843,12 +891,23 @@
                     "foreign or malformed forward-plan value".into(),
                 ));
             }
-            if !input.contiguous && !matches!(&operation, PlanOperation::Contiguous) {
+            if !input.contiguous && !matches!(operation, PlanOperation::Contiguous) {
                 return Err(Error::State(
                     "plan operator requires explicit contiguous input".into(),
                 ));
             }
         }
+        Ok(())
+    }
+    fn record(
+        &mut self,
+        name: &str,
+        operation: PlanOperation,
+        inputs: &[PlanValue],
+        shapes: &[ValueShape],
+        state: Option<StateEffect>,
+    ) -> Result<Vec<PlanValue>> {
+        self.validate_record_inputs(&operation, inputs)?;
         let precision = self.result_precision(&operation, inputs)?;
         let elements: Vec<_> = shapes
             .iter()
@@ -1082,6 +1141,52 @@
             input.shape,
             None,
         )
+    }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        dt: SourceParameter,
+        alpha: &PlanValue,
+        beta: &PlanValue,
+        contract: GdnCoefficientPreparationContract,
+    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
+        let heads = contract.validate(&dt, alpha.elements, beta.elements)?;
+        if self
+            .values
+            .len()
+            .checked_add(3)
+            .is_none_or(|count| count > u32::MAX as usize)
+        {
+            return Err(Error::Limit("GDN preparation value count overflow".into()));
+        }
+        let alpha_op = PlanOperation::TypedGdnAlphaPreparation {
+            dt: dt.clone(),
+            contract,
+        };
+        let beta_op = PlanOperation::TypedGdnBetaPreparation { heads, contract };
+        // Validate BOTH original values and every cold guard before recording.
+        self.validate_record_inputs(&alpha_op, &[*alpha])?;
+        self.validate_record_inputs(&beta_op, &[*beta])?;
+        self.result_precision(&alpha_op, &[*alpha])?;
+        self.result_precision(&beta_op, &[*beta])?;
+        let beta = self.one(
+            &format!("{}.beta-preparation", dt.name),
+            beta_op,
+            &[*beta],
+            ValueShape::Vector(heads),
+            None,
+        )?;
+        let values = self.record(
+            &format!("{}.alpha-preparation", dt.name),
+            alpha_op,
+            &[*alpha],
+            &[ValueShape::Vector(heads); 2],
+            None,
+        )?;
+        Ok(GdnCoefficientPreparationValues {
+            biased_alpha: values[0],
+            softplus: values[1],
+            beta,
+        })
     }
     fn embed(&mut self, tensor: SourceTensor, token: ParameterRef) -> Result<PlanValue> {
         self.source_tensor(&tensor)?;
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/coefficient_preparation.rs
@@ -0,0 +1,160 @@
+//! State-free original coefficient preparation; no exp/g/decay capability.
+use super::{QuantKind, SourceParameter};
+use crate::{Error, Result};
+
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum GdnCoefficientPreparationContract {
+    TorchCpuBf16Avx2,
+}
+impl GdnCoefficientPreparationContract {
+    pub fn validate(self, dt: &SourceParameter, alpha: usize, beta: usize) -> Result<usize> {
+        if !dt.source.is_bound() || dt.name.is_empty() {
+            return Err(Error::State(
+                "GDN preparation raw parameter source/name".into(),
+            ));
+        }
+        if !matches!(alpha, 16 | 48) || beta != alpha || dt.shape != [alpha as u64] {
+            return Err(Error::Malformed(
+                "GDN preparation one-call head/span".into(),
+            ));
+        }
+        let bytes = match dt.kind {
+            QuantKind::Bf16 => 2,
+            QuantKind::F32 => 4,
+            _ => {
+                return Err(Error::Unsupported(
+                    "GDN preparation raw dt_bias dtype".into(),
+                ));
+            }
+        };
+        if dt.bytes != alpha * bytes {
+            return Err(Error::Malformed(
+                "GDN preparation raw dt_bias byte span".into(),
+            ));
+        }
+        Ok(alpha)
+    }
+}
+
+pub struct GdnCoefficientPreparationValues<V> {
+    pub biased_alpha: V,
+    pub softplus: V,
+    pub beta: V,
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::accelerator::SourceIdentity;
+    use crate::accelerator::program::{
+        ActivationPrecision, EmbeddingContract, MathematicalBackend, MatrixAxes, PlanBuilder,
+        ProgramNumericalProfile, SourceTensor, ValueShape,
+    };
+    fn dt(source: SourceIdentity) -> SourceParameter {
+        SourceParameter {
+            source,
+            name: "synthetic.transport.dt_bias".into(),
+            kind: QuantKind::Bf16,
+            shape: vec![16],
+            bytes: 32,
+        }
+    }
+    fn embed(builder: &mut PlanBuilder, source: SourceIdentity) -> super::super::PlanValue {
+        builder
+            .embed_profiled(
+                SourceTensor {
+                    source,
+                    name: "synthetic.transport.embed".into(),
+                    kind: QuantKind::Bf16,
+                    rows: 1,
+                    columns: 16,
+                    axes: MatrixAxes::OutputInput,
+                },
+                PlanBuilder::inputs().token,
+                EmbeddingContract::Bf16OutputInput,
+            )
+            .unwrap()
+    }
+    #[test]
+    fn synthetic_builder_transport_keeps_distinct_precisions_and_no_state() {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        let mut builder = PlanBuilder::new_operator_slice(
+            source,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap();
+        let a = embed(&mut builder, source);
+        let b = embed(&mut builder, source);
+        let values = builder
+            .coefficient_preparation_profiled(
+                dt(source),
+                &a,
+                &b,
+                GdnCoefficientPreparationContract::TorchCpuBf16Avx2,
+            )
+            .unwrap();
+        assert_eq!(values.beta.precision(), ActivationPrecision::Bf16WidenedF32);
+        assert_eq!(values.softplus.precision(), ActivationPrecision::F32);
+        assert_eq!(values.biased_alpha.precision(), ActivationPrecision::F32);
+        let plan = builder.finish_operator_slice(values.softplus).unwrap();
+        assert_eq!(plan.plan().nodes().len(), 4);
+        assert!(plan.plan().required_domains().is_empty());
+        assert!(plan.plan().proposed_domains().is_empty());
+        assert_eq!(plan.plan().frontier_delta(), 0);
+        assert!(plan.plan().logits().is_none());
+        assert!(
+            plan.plan()
+                .values()
+                .iter()
+                .all(|v| matches!(v, ValueShape::Vector(16)))
+        );
+    }
+    #[test]
+    fn foreign_builder_input_refuses_before_recording_either_preparation_node() {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        let mut first = PlanBuilder::new_operator_slice(
+            source,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap();
+        let mut other = PlanBuilder::new_operator_slice(
+            source,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap();
+        let a = embed(&mut first, source);
+        let b = embed(&mut other, source);
+        assert!(
+            first
+                .coefficient_preparation_profiled(
+                    dt(source),
+                    &a,
+                    &b,
+                    GdnCoefficientPreparationContract::TorchCpuBf16Avx2
+                )
+                .is_err()
+        );
+        assert_eq!(
+            first.finish_operator_slice(a).unwrap().plan().nodes().len(),
+            1
+        );
+    }
+    #[test]
+    fn raw_parameter_dtype_bytes_and_span_are_checked() {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        let contract = GdnCoefficientPreparationContract::TorchCpuBf16Avx2;
+        let valid = dt(source);
+        assert_eq!(contract.validate(&valid, 16, 16).unwrap(), 16);
+        let mut wrong = valid.clone();
+        wrong.shape = vec![1, 16];
+        assert!(contract.validate(&wrong, 16, 16).is_err());
+        wrong = valid.clone();
+        wrong.kind = QuantKind::F16;
+        assert!(contract.validate(&wrong, 16, 16).is_err());
+        wrong = valid.clone();
+        wrong.bytes -= 1;
+        assert!(contract.validate(&wrong, 16, 16).is_err());
+        assert!(contract.validate(&valid, 32, 32).is_err());
+        assert!(contract.validate(&valid, 16, 48).is_err());
+    }
+}
--- a/crates/zerogravity-cpu/src/original_hf.rs
+++ b/crates/zerogravity-cpu/src/original_hf.rs
@@ -3,6 +3,7 @@
 //! learned matrix transpose or duplicate mathematical runtime is required.
 //! This does not enable complete trained Qwen3.8 inference.
 
+pub mod coefficient_preparation;
 pub mod conv;
 pub mod norm;
 pub mod recurrent;
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs
@@ -0,0 +1,222 @@
+//! Pinned original GDN coefficient PREPARATION only; no A_log exp/g/decay.
+//! Source modeling762feb6c lines617/619, using accepted sigmoid and softplus.
+use super::transcendentals::{
+    sigmoid_bf16_widened_f32_contiguous_span, softplus_beta1_threshold20_f32_contiguous16,
+};
+use crate::accelerator::SourceIdentity;
+use crate::quant::QuantKind;
+use crate::{Error, Result};
+use zerogravity_core::source::SourceTensorRef;
+
+/// Private stages only. The caller retains source/admission through this call.
+pub struct GdnCoefficientPreparation {
+    source: SourceIdentity,
+    biased_alpha: Vec<f32>,
+    softplus: Vec<f32>,
+    beta: Vec<f32>,
+}
+impl GdnCoefficientPreparation {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn biased_alpha(&self) -> &[f32] {
+        &self.biased_alpha
+    }
+    pub fn softplus(&self) -> &[f32] {
+        &self.softplus
+    }
+    pub fn beta(&self) -> &[f32] {
+        &self.beta
+    }
+    /// Private buffers, with no publication or admission claim.
+    pub fn into_parts(self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
+        (self.biased_alpha, self.softplus, self.beta)
+    }
+}
+
+/// One original [1,1,heads] contiguous a/b call, heads16 or48. Do not flatten
+/// independent calls: BF16 sigmoid's vector32 prefix/scalar tail is observable.
+/// Raw rank1 dt_bias is widened privately; no learned-source mutation/cache.
+/// Physical callers admit the complete private storage before entering.
+pub fn prepare_gdn_coefficients(
+    source: SourceIdentity,
+    dt_bias: SourceTensorRef<'_>,
+    alpha_bf16: &[f32],
+    beta_input_bf16: &[f32],
+) -> Result<GdnCoefficientPreparation> {
+    let heads = alpha_bf16.len();
+    if !source.is_bound()
+        || !matches!(heads, 16 | 48)
+        || beta_input_bf16.len() != heads
+        || dt_bias.shape != [heads as u64]
+    {
+        return Err(Error::Malformed(
+            "original GDN preparation source/head/span".into(),
+        ));
+    }
+    if !matches!(dt_bias.kind, QuantKind::Bf16 | QuantKind::F32) {
+        return Err(Error::Unsupported(
+            "original GDN preparation raw dt_bias dtype".into(),
+        ));
+    }
+    dt_bias.validate()?;
+    if alpha_bf16
+        .iter()
+        .chain(beta_input_bf16)
+        .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
+    {
+        return Err(Error::Malformed(
+            "original GDN preparation finite BF16 a/b".into(),
+        ));
+    }
+    let dt_f32 = match dt_bias.kind {
+        QuantKind::Bf16 => dt_bias
+            .packed_bytes
+            .chunks_exact(2)
+            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
+            .collect::<Vec<_>>(),
+        QuantKind::F32 => dt_bias
+            .packed_bytes
+            .chunks_exact(4)
+            .map(|b| f32::from_le_bytes(b.try_into().expect("validated raw F32 dt_bias")))
+            .collect(),
+        _ => unreachable!("validated dt_bias dtype"),
+    };
+    if dt_f32.iter().any(|v| !v.is_finite()) {
+        return Err(Error::Malformed(
+            "nonfinite original GDN raw dt_bias".into(),
+        ));
+    }
+    // Original line617 executes beta first; source line619 widens a before
+    // addition to raw dt_bias, then materializes FP32 softplus separately.
+    let beta = sigmoid_bf16_widened_f32_contiguous_span(source, beta_input_bf16)?;
+    let biased_alpha = alpha_bf16
+        .iter()
+        .zip(dt_f32)
+        .map(|(a, dt)| a + dt)
+        .collect::<Vec<_>>();
+    if biased_alpha.iter().any(|v| !v.is_finite()) {
+        return Err(Error::Malformed(
+            "nonfinite original GDN biased alpha".into(),
+        ));
+    }
+    let softplus = softplus_beta1_threshold20_f32_contiguous16(source, &biased_alpha)?;
+    Ok(GdnCoefficientPreparation {
+        source,
+        biased_alpha,
+        softplus,
+        beta,
+    })
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    fn source() -> SourceIdentity {
+        SourceIdentity::for_new_allocation().unwrap()
+    }
+    #[test]
+    fn linear_softplus_and_zero_beta_are_independent_exact_stages() {
+        for heads in [16, 48] {
+            let raw = vec![0u8; heads * 2];
+            let a = vec![21.0; heads];
+            let b = vec![0.0; heads];
+            let saved = raw.clone();
+            let out = prepare_gdn_coefficients(
+                source(),
+                SourceTensorRef {
+                    kind: QuantKind::Bf16,
+                    shape: &[heads as u64],
+                    packed_bytes: &raw,
+                },
+                &a,
+                &b,
+            )
+            .unwrap();
+            assert!(
+                out.biased_alpha()
+                    .iter()
+                    .all(|v| v.to_bits() == 21.0f32.to_bits())
+            );
+            assert!(
+                out.softplus()
+                    .iter()
+                    .all(|v| v.to_bits() == 21.0f32.to_bits())
+            );
+            assert!(out.beta().iter().all(|v| v.to_bits() == 0.5f32.to_bits()));
+            assert_eq!(raw, saved);
+        }
+    }
+    #[test]
+    fn raw_f32_dt_is_not_bf16_rounded_or_adjusted() {
+        let word = 0x3380_0001u32;
+        let raw = vec![word.to_le_bytes(); 16].concat();
+        let zero = [0.0; 16];
+        let out = prepare_gdn_coefficients(
+            source(),
+            SourceTensorRef {
+                kind: QuantKind::F32,
+                shape: &[16],
+                packed_bytes: &raw,
+            },
+            &zero,
+            &zero,
+        )
+        .unwrap();
+        assert!(out.biased_alpha().iter().all(|v| v.to_bits() == word));
+        assert_eq!(raw, vec![word.to_le_bytes(); 16].concat());
+    }
+    #[test]
+    fn source_raw_dtype_span_and_nonfinite_refuse() {
+        let raw = vec![0u8; 32];
+        let zero = [0.0; 16];
+        let dt = SourceTensorRef {
+            kind: QuantKind::Bf16,
+            shape: &[16],
+            packed_bytes: &raw,
+        };
+        assert!(prepare_gdn_coefficients(SourceIdentity::unbound(), dt, &zero, &zero).is_err());
+        assert!(prepare_gdn_coefficients(source(), dt, &zero[..15], &zero[..15]).is_err());
+        assert!(
+            prepare_gdn_coefficients(
+                source(),
+                SourceTensorRef {
+                    shape: &[1, 16],
+                    ..dt
+                },
+                &zero,
+                &zero
+            )
+            .is_err()
+        );
+        assert!(
+            prepare_gdn_coefficients(
+                source(),
+                SourceTensorRef {
+                    kind: QuantKind::F16,
+                    ..dt
+                },
+                &zero,
+                &zero
+            )
+            .is_err()
+        );
+        let mut bad = zero;
+        bad[1] = f32::from_bits(0x3f80_0001);
+        assert!(prepare_gdn_coefficients(source(), dt, &bad, &zero).is_err());
+        let mut bad_raw = raw.clone();
+        bad_raw[..2].copy_from_slice(&0x7fc1u16.to_le_bytes());
+        assert!(
+            prepare_gdn_coefficients(
+                source(),
+                SourceTensorRef {
+                    packed_bytes: &bad_raw,
+                    ..dt
+                },
+                &zero,
+                &zero
+            )
+            .is_err()
+        );
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
@@ -0,0 +1,261 @@
+//! Synthetic source/transport/lease guards only; never trained coefficients.
+use super::*;
+use crate::accelerator::program::{
+    GdnCoefficientPreparationContract, PlanBuilder, PlanOperation, ProgramNumericalProfile,
+};
+use crate::operator_slice::coefficient_preparation::GdnCoefficientPreparationInput;
+fn coefficient_source(fixture: &Fixture) -> Loaded {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    cfg["text_config"]["hidden_size"] = json!(256);
+    cfg["text_config"]["linear_num_value_heads"] = json!(16);
+    source_with_config(
+        fixture,
+        serde_json::to_vec(&cfg).unwrap(),
+        true,
+        |name, _, fallback| {
+            if name.ends_with(".dt_bias") {
+                0
+            } else {
+                fallback
+            }
+        },
+        |_| {},
+    )
+}
+fn raw(value: u16) -> Vec<u8> {
+    vec![value.to_le_bytes(); 16].concat()
+}
+fn input<'a>(
+    source: crate::accelerator::SourceIdentity,
+    a: &'a [u8],
+    b: &'a [u8],
+) -> GdnCoefficientPreparationInput<'a> {
+    GdnCoefficientPreparationInput::from_bf16_bytes(
+        source,
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        a,
+        b,
+    )
+    .unwrap()
+}
+#[test]
+fn synthetic_coefficient_stages_retain_two_aggregate_backings_and_source() {
+    let fixture = Fixture::new();
+    let loaded = coefficient_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let a = raw(0x41a8);
+    let b = raw(0); // a21, b0: public linear softplus and sigmoid0.
+    let before = reserved(&admission);
+    let output = operators
+        .execute_gdn_coefficient_preparation_slice(0, input(source.source_identity(), &a, &b), None)
+        .unwrap();
+    assert!(
+        output
+            .biased_alpha_f32()
+            .unwrap()
+            .iter()
+            .all(|v| v.to_bits() == 21.0f32.to_bits())
+    );
+    assert!(
+        output
+            .softplus_f32()
+            .unwrap()
+            .iter()
+            .all(|v| v.to_bits() == 21.0f32.to_bits())
+    );
+    assert!(
+        output
+            .beta_bf16()
+            .unwrap()
+            .iter()
+            .all(|v| v.to_bits() == 0.5f32.to_bits())
+    );
+    assert_eq!(
+        output.raw_dt_bias_parameter().kind,
+        zerogravity_core::quant::QuantKind::Bf16
+    );
+    assert_eq!(
+        output.unevaluated_raw_a_log_parameter().kind,
+        zerogravity_core::quant::QuantKind::F32
+    );
+    assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 34 * 1024 * 1024);
+    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 16]);
+    assert_eq!(output.beta_input_bf16().unwrap(), &[0.0; 16]);
+    drop(output);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn wrong_layer_foreign_input_cancel_and_pressure_refuse_without_leak() {
+    let fixture = Fixture::new();
+    let loaded = coefficient_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let zero = raw(0);
+    let before = reserved(&admission);
+    assert!(
+        operators
+            .execute_gdn_coefficient_preparation_slice(
+                3,
+                input(source.source_identity(), &zero, &zero),
+                None
+            )
+            .is_err()
+    );
+    assert!(
+        operators
+            .execute_gdn_coefficient_preparation_slice(
+                0,
+                input(
+                    crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
+                    &zero,
+                    &zero
+                ),
+                None
+            )
+            .is_err()
+    );
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        operators
+            .execute_gdn_coefficient_preparation_slice(
+                0,
+                input(source.source_identity(), &zero, &zero),
+                Some(&cancelled)
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before);
+    let remaining = admission
+        .arbiter
+        .snapshot(&admission.resource)
+        .unwrap()
+        .remaining;
+    let pressure = source.reserve_workspace(remaining as usize).unwrap();
+    let pressured = reserved(&admission);
+    assert!(
+        operators
+            .execute_gdn_coefficient_preparation_slice(
+                0,
+                input(source.source_identity(), &zero, &zero),
+                None
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), pressured);
+    drop(pressure);
+    assert_eq!(reserved(&admission), before);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn raw_a_log_alias_and_unaware_provider_refuse_before_callback_or_output() {
+    struct Provider(AtomicBool);
+    impl crate::accelerator::MatVecKernel for Provider {
+        fn try_execute(
+            &self,
+            _: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            self.0.store(true, Ordering::Relaxed);
+            panic!("unaware coefficient provider invoked");
+        }
+    }
+    let fixture = Fixture::new();
+    let loaded = coefficient_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let zero = raw(0);
+    let (a, b) = {
+        let math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
+        math.import_coefficient_preparation_input(input(source.source_identity(), &zero, &zero), 16)
+            .unwrap()
+    };
+    let before = reserved(&admission);
+    let provider = Provider(AtomicBool::new(false));
+    {
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(
+            &operators,
+            None,
+            Some(&provider),
+        );
+        assert!(
+            math.coefficient_preparation_profiled(
+                operators
+                    .parameter_descriptor(QwenParameterRole::GdnDt(0))
+                    .unwrap(),
+                &a,
+                &b,
+                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+        );
+    }
+    {
+        let mut math =
+            crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
+        assert!(
+            math.coefficient_preparation_profiled(
+                operators
+                    .parameter_descriptor(QwenParameterRole::GdnA(0))
+                    .unwrap(),
+                &a,
+                &b,
+                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+        );
+    }
+    assert!(!provider.0.load(Ordering::Relaxed));
+    assert_eq!(reserved(&admission), before);
+    drop(a);
+    drop(b);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn plan_appends_preparation_without_exp_state_token_or_producer_invention() {
+    let fixture = Fixture::new();
+    let loaded = coefficient_source(&fixture);
+    let (model, _, _) = bind(loaded);
+    let mut builder =
+        PlanBuilder::new_operator_slice(model.source_identity(), model.program_numerical_profile())
+            .unwrap();
+    // Synthetic same-builder views are transport ONLY, never actual trained a/b.
+    let embed = builder
+        .embed_profiled(
+            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    let a = builder.view(&embed, TensorView::vector(0, 16)).unwrap();
+    let b = builder.view(&embed, TensorView::vector(16, 16)).unwrap();
+    let values = model
+        .append_gdn_coefficient_preparation_slice(&mut builder, 0, &a, &b)
+        .unwrap();
+    let plan = builder.finish_operator_slice(values.softplus).unwrap();
+    assert_eq!(plan.plan().nodes().len(), 5);
+    assert_eq!(
+        plan.plan()
+            .nodes()
+            .iter()
+            .filter(|n| matches!(
+                n.operation,
+                PlanOperation::TypedGdnAlphaPreparation { .. }
+                    | PlanOperation::TypedGdnBetaPreparation { .. }
+            ))
+            .count(),
+        2
+    );
+    assert!(plan.plan().required_domains().is_empty());
+    assert!(plan.plan().proposed_domains().is_empty());
+    assert!(plan.plan().logits().is_none());
+    assert_eq!(plan.plan().frontier_delta(), 0);
+    assert!(model.new_session().is_err());
+    assert!(model.forward_plan(8, true, true).is_err());
+}
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -200,6 +200,55 @@
         normalized_query,
         normalized_key,
     })
+}
+
+/// Pure coefficient preparation only; no exp, g/decay, scaling or state effects.
+pub(crate) fn gdn_coefficient_preparation_slice<B: MathematicalBackend>(
+    source: &(impl crate::operator_slice::OperatorSliceSource + ?Sized),
+    math: &mut B,
+    layer: usize,
+    alpha: &B::Value,
+    beta_input: &B::Value,
+) -> Result<crate::accelerator::program::GdnCoefficientPreparationValues<B::Value>> {
+    use crate::accelerator::program::{ActivationPrecision, GdnCoefficientPreparationContract};
+    let roles = crate::operator_slice::coefficient_preparation::coefficient_roles(source, layer)?;
+    for input in [alpha, beta_input] {
+        expect_elements(
+            math,
+            input,
+            roles.heads,
+            "coefficient preparation actual head count",
+        )?;
+        if math.precision(input) != ActivationPrecision::Bf16WidenedF32 {
+            return Err(Error::Unsupported(
+                "coefficient preparation requires original BF16 a/b".into(),
+            ));
+        }
+    }
+    let values = math.coefficient_preparation_profiled(
+        roles.dt,
+        alpha,
+        beta_input,
+        GdnCoefficientPreparationContract::TorchCpuBf16Avx2,
+    )?;
+    for (value, precision) in [
+        (&values.biased_alpha, ActivationPrecision::F32),
+        (&values.softplus, ActivationPrecision::F32),
+        (&values.beta, ActivationPrecision::Bf16WidenedF32),
+    ] {
+        expect_elements(
+            math,
+            value,
+            roles.heads,
+            "coefficient preparation output head count",
+        )?;
+        if math.precision(value) != precision {
+            return Err(Error::State(
+                "coefficient preparation output precision".into(),
+            ));
+        }
+    }
+    Ok(values)
 }
 
 fn expect_elements<B: MathematicalBackend>(
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -19,6 +19,9 @@
 use std::sync::Arc;
 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
 use zerogravity_artifacts::source_weights::WindowWorkspace;
+
+#[path = "host_math/coefficient_preparation.rs"]
+mod coefficient_preparation;
 
 struct HostStorage {
     values: Vec<f32>,
@@ -507,6 +510,15 @@
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         let output = output.retain_workspace(workspace)?;
         Ok(output)
+    }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        dt: SourceParameter,
+        alpha: &HostValue,
+        beta: &HostValue,
+        contract: crate::accelerator::program::GdnCoefficientPreparationContract,
+    ) -> Result<crate::accelerator::program::GdnCoefficientPreparationValues<HostValue>> {
+        self.coefficient_preparation_closed(dt, alpha, beta, contract)
     }
     fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<HostValue> {
         self.legacy_precision(&[])?;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs
@@ -0,0 +1,147 @@
+//! One existing HostMath adapter; source/leases retained, no new engine.
+use super::*;
+use crate::accelerator::program::{
+    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues,
+};
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+use crate::operator_slice::coefficient_preparation::GdnCoefficientPreparationInput;
+use crate::operator_slice::coefficient_preparation::coefficient_roles;
+
+impl HostMath<'_, '_, '_> {
+    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+    pub(crate) fn import_coefficient_preparation_input(
+        &self,
+        input: GdnCoefficientPreparationInput<'_>,
+        heads: usize,
+    ) -> Result<(HostValue, HostValue)> {
+        if input.validate()? != heads
+            || input.source_identity() != self.source.source_identity()
+            || input.numerical_profile() != self.source.program_numerical_profile()
+        {
+            return Err(Error::State(
+                "coefficient preparation imported source/profile/heads".into(),
+            ));
+        }
+        if self.region_kernel.is_some() {
+            return Err(Error::Unsupported(
+                "coefficient preparation unaware input provider".into(),
+            ));
+        }
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let workspace = self.prepare_output(None)?; // BEFORE both private copies.
+        let decode = |raw: &[u8]| {
+            raw.chunks_exact(2)
+                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
+                .collect::<Vec<_>>()
+        };
+        let mut values = Vec::with_capacity(2);
+        for raw in [input.alpha_bytes(), input.beta_bytes()] {
+            let value = HostValue::typed(
+                decode(raw),
+                ActivationPrecision::Bf16WidenedF32,
+                input.source_identity(),
+                input.numerical_profile(),
+            )?;
+            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+            let value = value.retain_workspace(workspace.clone())?;
+            values.push(value);
+        }
+        Ok((values.remove(0), values.remove(0)))
+    }
+    pub(super) fn coefficient_preparation_closed(
+        &self,
+        dt: SourceParameter,
+        alpha: &HostValue,
+        beta: &HostValue,
+        contract: GdnCoefficientPreparationContract,
+    ) -> Result<GdnCoefficientPreparationValues<HostValue>> {
+        if self.region_kernel.is_some() {
+            return Err(Error::Unsupported(
+                "physical provider lacks original coefficient preparation contract".into(),
+            ));
+        }
+        let source = self.source.source_identity();
+        let profile = self.source.program_numerical_profile();
+        if profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+            || [alpha, beta].iter().any(|v| {
+                v.precision != ActivationPrecision::Bf16WidenedF32
+                    || v.source != Some(source)
+                    || v.profile != Some(profile)
+            })
+        {
+            return Err(Error::Unsupported(
+                "host coefficient preparation precision/source/profile".into(),
+            ));
+        }
+        let heads = contract.validate(&dt, alpha.elements, beta.elements)?;
+        let mut matching_role = false;
+        for (layer, kind) in self.source.metadata().geometry().layers.iter().enumerate() {
+            if *kind != crate::model::LayerKind::LinearAttention {
+                continue;
+            }
+            let roles = coefficient_roles(self.source.metadata(), layer)?;
+            if roles.dt == dt && roles.heads == heads {
+                matching_role = true;
+                break;
+            }
+        }
+        if !matching_role {
+            return Err(Error::State(
+                "host coefficient preparation original dt_bias role".into(),
+            ));
+        }
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let workspace = {
+            if let HostSource::Windows(owner) = &self.source {
+                let beta_workspace = beta.data.workspace.as_ref().ok_or_else(|| {
+                    Error::State("coefficient beta has no original physical backing".into())
+                })?;
+                if !owner.source().same_issuer(beta_workspace.source()) {
+                    return Err(Error::State(
+                        "coefficient beta source issuer changed".into(),
+                    ));
+                }
+            }
+            self.prepare_output(Some(alpha))? // Shared aggregate for ALL3 outputs.
+        };
+        let prepared = match &self.source {
+            HostSource::Loaded(owner) => {
+                dt.validate_source(*owner)?;
+                let raw = zerogravity_core::source::TensorSource::tensor(*owner, &dt.name)?;
+                zerogravity_cpu::original_hf::coefficient_preparation::prepare_gdn_coefficients(
+                    source,
+                    raw,
+                    alpha.as_slice()?,
+                    beta.as_slice()?,
+                )?
+            }
+            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+            HostSource::Windows(owner) => owner.prepare_coefficient_stages(
+                &dt,
+                alpha.as_slice()?,
+                beta.as_slice()?,
+                self.cancelled,
+            )?,
+        };
+        if prepared.source_identity() != source {
+            return Err(Error::State("coefficient preparation result source".into()));
+        }
+        let (biased, softplus, beta) = prepared.into_parts();
+        let mut outputs = Vec::with_capacity(3);
+        for (values, precision) in [
+            (biased, ActivationPrecision::F32),
+            (softplus, ActivationPrecision::F32),
+            (beta, ActivationPrecision::Bf16WidenedF32),
+        ] {
+            let value = HostValue::typed(values, precision, source, profile)?;
+            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+            let value = value.retain_workspace(workspace.clone())?;
+            outputs.push(value);
+        }
+        Ok(GdnCoefficientPreparationValues {
+            biased_alpha: outputs.remove(0),
+            softplus: outputs.remove(0),
+            beta: outputs.remove(0),
+        })
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -228,6 +228,10 @@
 #[path = "operator_slice_tests.rs"]
 mod tests;
 
+/// Pure source coefficient preparation; no g/decay/exp or state capability.
+#[path = "operator_slice/coefficient_preparation.rs"]
+pub mod coefficient_preparation;
+
 /// Pure attention projections/QK norms; no rotary/cache/session admission.
 #[path = "attention_slice.rs"]
 pub mod attention;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs
@@ -0,0 +1,283 @@
+//! Source-bound coefficient PREPARATION on caller-owned a/b; no g/decay/exp.
+use super::OperatorSliceSource;
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{
+    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues, MatrixAxes, PlanBuilder,
+    PlanValue, ProgramNumericalProfile, SourceParameter,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, LoadedQwenModel, Result};
+use zerogravity_core::quant::QuantKind;
+
+pub(crate) struct CoefficientPreparationRoles {
+    pub(crate) heads: usize,
+    pub(crate) dt: SourceParameter,
+    pub(crate) a_log: SourceParameter,
+}
+/// Immutable complete role descriptors; A_log is not evaluated by this slice.
+pub(crate) fn coefficient_roles(
+    source: &(impl OperatorSliceSource + ?Sized),
+    layer: usize,
+) -> Result<CoefficientPreparationRoles> {
+    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || !source.source_identity().is_bound()
+        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
+    {
+        return Err(Error::Unsupported(
+            "coefficient preparation original LinearAttention source/profile".into(),
+        ));
+    }
+    let heads = usize::try_from(source.geometry().recurrent_value_heads.unwrap_or(0))
+        .map_err(|_| Error::Limit("coefficient heads".into()))?;
+    let hidden = usize::try_from(source.geometry().hidden_size)
+        .map_err(|_| Error::Limit("coefficient hidden".into()))?;
+    if !matches!(heads, 16 | 48) || !matches!(hidden, 256 | 1024 | 2048 | 5120) {
+        return Err(Error::Unsupported(
+            "coefficient preparation closed source geometry".into(),
+        ));
+    }
+    let namespace = format!("model.language_model.layers.{layer}.linear_attn");
+    for (role, suffix) in [
+        (QwenMatrixRole::GdnAlpha(layer), "in_proj_a"),
+        (QwenMatrixRole::GdnBeta(layer), "in_proj_b"),
+    ] {
+        let tensor = source.matrix_descriptor(role)?;
+        if tensor.source != source.source_identity()
+            || tensor.name != format!("{namespace}.{suffix}.weight")
+            || tensor.kind != QuantKind::Bf16
+            || tensor.axes != MatrixAxes::OutputInput
+            || tensor.rows != heads
+            || tensor.columns != hidden
+        {
+            return Err(Error::State(
+                "coefficient preparation original a/b full matrix role".into(),
+            ));
+        }
+    }
+    let dt = source.parameter_descriptor(QwenParameterRole::GdnDt(layer))?;
+    let a_log = source.parameter_descriptor(QwenParameterRole::GdnA(layer))?;
+    if dt.source != source.source_identity()
+        || dt.name != format!("{namespace}.dt_bias")
+        || a_log.source != source.source_identity()
+        || a_log.name != format!("{namespace}.A_log")
+        || a_log.kind != QuantKind::F32
+        || a_log.shape != [heads as u64]
+        || a_log.bytes != heads * 4
+    {
+        return Err(Error::State(
+            "coefficient preparation original raw A_log/dt_bias roles".into(),
+        ));
+    }
+    GdnCoefficientPreparationContract::TorchCpuBf16Avx2.validate(&dt, heads, heads)?;
+    Ok(CoefficientPreparationRoles { heads, dt, a_log })
+}
+
+impl LoadedQwenModel {
+    /// Append only beta/biased-alpha/softplus to caller's original plan values.
+    pub fn append_gdn_coefficient_preparation_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        a: &PlanValue,
+        b: &PlanValue,
+    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
+        self.require_operator_slice()?;
+        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
+    }
+}
+
+/// External actual source/operator a/b bytes; caller authenticates provenance.
+/// No hidden state, matrix producer, token or frontier is asserted by import.
+#[derive(Clone, Copy)]
+pub struct GdnCoefficientPreparationInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    alpha: &'a [u8],
+    beta: &'a [u8],
+}
+impl<'a> GdnCoefficientPreparationInput<'a> {
+    pub fn from_bf16_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        alpha: &'a [u8],
+        beta: &'a [u8],
+    ) -> Result<Self> {
+        let input = Self {
+            source,
+            profile,
+            alpha,
+            beta,
+        };
+        input.validate()?;
+        Ok(input)
+    }
+    pub fn source_identity(self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    pub fn alpha_bytes(self) -> &'a [u8] {
+        self.alpha
+    }
+    pub fn beta_bytes(self) -> &'a [u8] {
+        self.beta
+    }
+    pub(crate) fn validate(self) -> Result<usize> {
+        if !self.source.is_bound()
+            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::State(
+                "coefficient preparation private input source/profile".into(),
+            ));
+        }
+        if !matches!(self.alpha.len(), 32 | 96)
+            || self.beta.len() != self.alpha.len()
+            || self
+                .alpha
+                .chunks_exact(2)
+                .chain(self.beta.chunks_exact(2))
+                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
+        {
+            return Err(Error::Malformed(
+                "coefficient preparation one finite BF16 a/b call".into(),
+            ));
+        }
+        Ok(self.alpha.len() / 2)
+    }
+}
+
+#[cfg(feature = "cpu")]
+pub struct GdnCoefficientPreparationOutput {
+    source: SourceIdentity,
+    layer: usize,
+    dt: SourceParameter,
+    a_log: SourceParameter,
+    alpha: crate::host_math::HostValue,
+    beta_input: crate::host_math::HostValue,
+    values: GdnCoefficientPreparationValues<crate::host_math::HostValue>,
+}
+#[cfg(feature = "cpu")]
+impl GdnCoefficientPreparationOutput {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn layer(&self) -> usize {
+        self.layer
+    }
+    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    }
+    pub fn raw_dt_bias_parameter(&self) -> &SourceParameter {
+        &self.dt
+    }
+    pub fn unevaluated_raw_a_log_parameter(&self) -> &SourceParameter {
+        &self.a_log
+    }
+    pub fn alpha_bf16(&self) -> Result<&[f32]> {
+        self.alpha.as_slice()
+    }
+    pub fn beta_input_bf16(&self) -> Result<&[f32]> {
+        self.beta_input.as_slice()
+    }
+    pub fn biased_alpha_f32(&self) -> Result<&[f32]> {
+        self.values.biased_alpha.as_slice()
+    }
+    pub fn softplus_f32(&self) -> Result<&[f32]> {
+        self.values.softplus.as_slice()
+    }
+    pub fn beta_bf16(&self) -> Result<&[f32]> {
+        self.values.beta.as_slice()
+    }
+}
+
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    pub fn append_gdn_coefficient_preparation_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        a: &PlanValue,
+        b: &PlanValue,
+    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
+        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_gdn_coefficient_preparation_slice(
+        &self,
+        layer: usize,
+        input: GdnCoefficientPreparationInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<GdnCoefficientPreparationOutput> {
+        let roles = coefficient_roles(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let (alpha, beta_input) = math.import_coefficient_preparation_input(input, roles.heads)?;
+        let values = crate::forward::gdn_coefficient_preparation_slice(
+            self,
+            &mut math,
+            layer,
+            &alpha,
+            &beta_input,
+        )?;
+        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State(
+                "window coefficient preparation cancelled".into(),
+            ));
+        }
+        Ok(GdnCoefficientPreparationOutput {
+            source: self.source_identity(),
+            layer,
+            dt: roles.dt,
+            a_log: roles.a_log,
+            alpha,
+            beta_input,
+            values,
+        })
+    }
+    #[cfg(feature = "cpu")]
+    pub(crate) fn prepare_coefficient_stages(
+        &self,
+        dt: &SourceParameter,
+        alpha: &[f32],
+        beta: &[f32],
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<zerogravity_cpu::original_hf::coefficient_preparation::GdnCoefficientPreparation>
+    {
+        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State(
+                "window coefficient raw parameter cancelled".into(),
+            ));
+        }
+        let raw = self.source().parameter(&dt.name)?;
+        raw.validate_source_parameter(dt)?;
+        let window = match cancelled {
+            Some(flag) => {
+                self.source()
+                    .read_parameter_cancellable(&raw, 0..raw.elements(), flag)?
+            }
+            None => self.source().read_parameter(&raw, 0..raw.elements())?,
+        };
+        if !window.is_complete() || window.element_range() != (0..raw.elements()) {
+            return Err(Error::State(
+                "window coefficient complete raw dt_bias".into(),
+            ));
+        }
+        let result =
+            zerogravity_cpu::original_hf::coefficient_preparation::prepare_gdn_coefficients(
+                self.source_identity(),
+                zerogravity_core::source::SourceTensorRef {
+                    kind: raw.tensor().kind,
+                    shape: &raw.tensor().shape,
+                    packed_bytes: window.as_slice(),
+                },
+                alpha,
+                beta,
+            )?;
+        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State(
+                "window coefficient cancelled after synchronous math".into(),
+            ));
+        }
+        Ok(result)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -379,3 +379,6 @@
 
 #[path = "attention_slice_tests.rs"]
 mod attention;
+
+#[path = "coefficient_preparation_tests.rs"]
+mod coefficient_preparation;

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/frozen/WIRING.patch

Original UTF-8 bytes: 11412; SHA256 `2b5c5fa231b947c07e0d5130e00657922f163fc19d23c5762699a362fa7b66d7`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -12,6 +12,10 @@
 use std::sync::Arc;
 use std::sync::atomic::{AtomicU64, Ordering};
 
+mod coefficient_preparation;
+pub use coefficient_preparation::{
+    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues,
+};
 mod typed;
 pub use typed::{
     ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
@@ -280,6 +284,17 @@
             "backend does not implement explicit ordinary RMS contract".into(),
         ))
     }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        _dt: SourceParameter,
+        _alpha: &Self::Value,
+        _beta: &Self::Value,
+        _contract: GdnCoefficientPreparationContract,
+    ) -> Result<GdnCoefficientPreparationValues<Self::Value>> {
+        Err(Error::Unsupported(
+            "backend does not implement original GDN coefficient preparation".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -456,6 +471,14 @@
     TypedOrdinaryRms {
         parameter: SourceParameter,
         contract: OrdinaryRmsContract,
+    },
+    TypedGdnAlphaPreparation {
+        dt: SourceParameter,
+        contract: GdnCoefficientPreparationContract,
+    },
+    TypedGdnBetaPreparation {
+        heads: usize,
+        contract: GdnCoefficientPreparationContract,
     },
     Embed {
         tensor: SourceTensor,
@@ -800,6 +823,32 @@
                 contract.validate(parameter, inputs[0].elements)?;
                 Ok(ActivationPrecision::Bf16WidenedF32)
             }
+            PlanOperation::TypedGdnAlphaPreparation { dt, contract } => {
+                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+                    || dt.source != self.source
+                    || inputs.len() != 1
+                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
+                {
+                    return Err(Error::Unsupported(
+                        "GDN preparation alpha profile/source/precision".into(),
+                    ));
+                }
+                contract.validate(dt, inputs[0].elements, inputs[0].elements)?;
+                Ok(ActivationPrecision::F32)
+            }
+            PlanOperation::TypedGdnBetaPreparation { heads, .. } => {
+                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+                    || !matches!(*heads, 16 | 48)
+                    || inputs.len() != 1
+                    || inputs[0].elements != *heads
+                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
+                {
+                    return Err(Error::Unsupported(
+                        "GDN preparation beta profile/span/precision".into(),
+                    ));
+                }
+                Ok(ActivationPrecision::Bf16WidenedF32)
+            }
             PlanOperation::View(_) | PlanOperation::Contiguous => inputs
                 .first()
                 .map(|value| value.precision)
@@ -816,6 +865,8 @@
                     PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
+                        | PlanOperation::TypedGdnAlphaPreparation { .. }
+                        | PlanOperation::TypedGdnBetaPreparation { .. }
                 ) =>
             {
                 Ok(ActivationPrecision::F32)
@@ -825,14 +876,11 @@
             )),
         }
     }
-    fn record(
-        &mut self,
-        name: &str,
-        operation: PlanOperation,
+    fn validate_record_inputs(
+        &self,
+        operation: &PlanOperation,
         inputs: &[PlanValue],
-        shapes: &[ValueShape],
-        state: Option<StateEffect>,
-    ) -> Result<Vec<PlanValue>> {
+    ) -> Result<()> {
         for input in inputs {
             if input.owner != self.owner
                 || self.values.get(input.id.0 as usize) != Some(&input.shape)
@@ -843,12 +891,23 @@
                     "foreign or malformed forward-plan value".into(),
                 ));
             }
-            if !input.contiguous && !matches!(&operation, PlanOperation::Contiguous) {
+            if !input.contiguous && !matches!(operation, PlanOperation::Contiguous) {
                 return Err(Error::State(
                     "plan operator requires explicit contiguous input".into(),
                 ));
             }
         }
+        Ok(())
+    }
+    fn record(
+        &mut self,
+        name: &str,
+        operation: PlanOperation,
+        inputs: &[PlanValue],
+        shapes: &[ValueShape],
+        state: Option<StateEffect>,
+    ) -> Result<Vec<PlanValue>> {
+        self.validate_record_inputs(&operation, inputs)?;
         let precision = self.result_precision(&operation, inputs)?;
         let elements: Vec<_> = shapes
             .iter()
@@ -1082,6 +1141,52 @@
             input.shape,
             None,
         )
+    }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        dt: SourceParameter,
+        alpha: &PlanValue,
+        beta: &PlanValue,
+        contract: GdnCoefficientPreparationContract,
+    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
+        let heads = contract.validate(&dt, alpha.elements, beta.elements)?;
+        if self
+            .values
+            .len()
+            .checked_add(3)
+            .is_none_or(|count| count > u32::MAX as usize)
+        {
+            return Err(Error::Limit("GDN preparation value count overflow".into()));
+        }
+        let alpha_op = PlanOperation::TypedGdnAlphaPreparation {
+            dt: dt.clone(),
+            contract,
+        };
+        let beta_op = PlanOperation::TypedGdnBetaPreparation { heads, contract };
+        // Validate BOTH original values and every cold guard before recording.
+        self.validate_record_inputs(&alpha_op, &[*alpha])?;
+        self.validate_record_inputs(&beta_op, &[*beta])?;
+        self.result_precision(&alpha_op, &[*alpha])?;
+        self.result_precision(&beta_op, &[*beta])?;
+        let beta = self.one(
+            &format!("{}.beta-preparation", dt.name),
+            beta_op,
+            &[*beta],
+            ValueShape::Vector(heads),
+            None,
+        )?;
+        let values = self.record(
+            &format!("{}.alpha-preparation", dt.name),
+            alpha_op,
+            &[*alpha],
+            &[ValueShape::Vector(heads); 2],
+            None,
+        )?;
+        Ok(GdnCoefficientPreparationValues {
+            biased_alpha: values[0],
+            softplus: values[1],
+            beta,
+        })
     }
     fn embed(&mut self, tensor: SourceTensor, token: ParameterRef) -> Result<PlanValue> {
         self.source_tensor(&tensor)?;
--- a/crates/zerogravity-cpu/src/original_hf.rs
+++ b/crates/zerogravity-cpu/src/original_hf.rs
@@ -3,6 +3,7 @@
 //! learned matrix transpose or duplicate mathematical runtime is required.
 //! This does not enable complete trained Qwen3.8 inference.
 
+pub mod coefficient_preparation;
 pub mod conv;
 pub mod norm;
 pub mod recurrent;
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -200,6 +200,55 @@
         normalized_query,
         normalized_key,
     })
+}
+
+/// Pure coefficient preparation only; no exp, g/decay, scaling or state effects.
+pub(crate) fn gdn_coefficient_preparation_slice<B: MathematicalBackend>(
+    source: &(impl crate::operator_slice::OperatorSliceSource + ?Sized),
+    math: &mut B,
+    layer: usize,
+    alpha: &B::Value,
+    beta_input: &B::Value,
+) -> Result<crate::accelerator::program::GdnCoefficientPreparationValues<B::Value>> {
+    use crate::accelerator::program::{ActivationPrecision, GdnCoefficientPreparationContract};
+    let roles = crate::operator_slice::coefficient_preparation::coefficient_roles(source, layer)?;
+    for input in [alpha, beta_input] {
+        expect_elements(
+            math,
+            input,
+            roles.heads,
+            "coefficient preparation actual head count",
+        )?;
+        if math.precision(input) != ActivationPrecision::Bf16WidenedF32 {
+            return Err(Error::Unsupported(
+                "coefficient preparation requires original BF16 a/b".into(),
+            ));
+        }
+    }
+    let values = math.coefficient_preparation_profiled(
+        roles.dt,
+        alpha,
+        beta_input,
+        GdnCoefficientPreparationContract::TorchCpuBf16Avx2,
+    )?;
+    for (value, precision) in [
+        (&values.biased_alpha, ActivationPrecision::F32),
+        (&values.softplus, ActivationPrecision::F32),
+        (&values.beta, ActivationPrecision::Bf16WidenedF32),
+    ] {
+        expect_elements(
+            math,
+            value,
+            roles.heads,
+            "coefficient preparation output head count",
+        )?;
+        if math.precision(value) != precision {
+            return Err(Error::State(
+                "coefficient preparation output precision".into(),
+            ));
+        }
+    }
+    Ok(values)
 }
 
 fn expect_elements<B: MathematicalBackend>(
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -19,6 +19,9 @@
 use std::sync::Arc;
 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
 use zerogravity_artifacts::source_weights::WindowWorkspace;
+
+#[path = "host_math/coefficient_preparation.rs"]
+mod coefficient_preparation;
 
 struct HostStorage {
     values: Vec<f32>,
@@ -507,6 +510,15 @@
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         let output = output.retain_workspace(workspace)?;
         Ok(output)
+    }
+    fn coefficient_preparation_profiled(
+        &mut self,
+        dt: SourceParameter,
+        alpha: &HostValue,
+        beta: &HostValue,
+        contract: crate::accelerator::program::GdnCoefficientPreparationContract,
+    ) -> Result<crate::accelerator::program::GdnCoefficientPreparationValues<HostValue>> {
+        self.coefficient_preparation_closed(dt, alpha, beta, contract)
     }
     fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<HostValue> {
         self.legacy_precision(&[])?;
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -228,6 +228,10 @@
 #[path = "operator_slice_tests.rs"]
 mod tests;
 
+/// Pure source coefficient preparation; no g/decay/exp or state capability.
+#[path = "operator_slice/coefficient_preparation.rs"]
+pub mod coefficient_preparation;
+
 /// Pure attention projections/QK norms; no rotary/cache/session admission.
 #[path = "attention_slice.rs"]
 pub mod attention;
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -379,3 +379,6 @@
 
 #[path = "attention_slice_tests.rs"]
 mod attention;
+
+#[path = "coefficient_preparation_tests.rs"]
+mod coefficient_preparation;

````

## Source: .agents/runs/model-execution-pinned-native-provider-probe-20261020/FOLLOWUP-CLEANUP-LENGTH.patch

Original UTF-8 bytes: 5169; SHA256 `d10d13c4dce99d70892e4f6dee263c4253d53d74871e08d6848b128a6ff61f3d`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/README.md
+++ b/README.md
@@ -75,3 +75,9 @@
 UNQUALIFIED; no automatic rerun or next provider. `prepared-manifest.json`
 retains exact source/input hashes. Licence/redistribution closure is still the
 source-only audit's open boundary; no library is redistributed here.
+
+Root review follow-up preserves initial `frozen-prepared/` bytes: explicit
+actual/expected output-length equality before comparison, and a second
+nonce/name/CID-bound container cleanup/absence check after foreground CLI reap.
+An empty CID file during owned startup uses the independently nonce-bound name;
+nonempty malformed/foreign ownership is refused without container signals.
--- a/TODO.md
+++ b/TODO.md
@@ -7,6 +7,7 @@
 - [x] Prepare one Rust public-operator diagnostic with bounded private aligned buffers, raw-word comparison and actual mapped closure/entry/read-set receipts.
 - [x] Prepare exact GNU-loader/container launcher with static ELF version guard, fixed finite limits and owned cleanup before final evidence I/O.
 - [x] Python AST/syntax only; no Cargo/Rust/native/container/model/runtime execution.
+- [x] Root review repairs: exact output-length equality and post-CLI-reap independent owned-container cleanup/absence; empty startup CID handled by nonce-bound name. First frozen bytes retained.
 - [ ] Independent source/ABI/layout/cleanup review and root build/strict source checks.
 - [ ] Root-only bounded actual native output/closure/dispatch qualification; no inference claim.
 - [ ] Native allocation/admission/production packaging and same-engine typed wiring remain separate unqualified work.
--- a/main.rs
+++ b/main.rs
@@ -185,6 +185,7 @@
     // Expected output is first read AFTER the independent native call completed.
     let original = raw_tensor(root, expected)?;
     let actual = logical(buffer, expected)?;
+    need(actual.len() == original.len(), "complete actual/original output length")?;
     let mut differences = 0usize;
     let mut first = Vec::new();
     for (i, (a, b)) in actual.chunks_exact(4).zip(original.chunks_exact(4)).enumerate() {
--- a/run_probe.py
+++ b/run_probe.py
@@ -109,10 +109,14 @@
 
 def owned_container(name, nonce, cidfile):
     target = name
+    recorded_cid = None
     if cidfile.exists():
         cid = cidfile.read_text().strip()
-        require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
-        target = cid
+        # Podman creates the file before publishing its ID; nonce/name remains
+        # independently inspectable during this owned startup interval.
+        if cid:
+            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
+            target = recorded_cid = cid
     exists = command(PODMAN + ["container", "exists", target])
     if exists.returncode == 1:
         return None
@@ -124,12 +128,12 @@
             "container name/nonce mismatch; no signal")
     cid = value["Id"]
     require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
-    if cidfile.exists():
-        require(cidfile.read_text().strip() == cid, "owned CID changed; no signal")
+    if recorded_cid is not None:
+        require(recorded_cid == cid, "owned CID changed; no signal")
     return cid
 
 
-def cleanup(child, name, nonce, cidfile):
+def cleanup_container(name, nonce, cidfile):
     errors = []
     try:
         cid = owned_container(name, nonce, cidfile)
@@ -144,15 +148,32 @@
                 errors.append("owned stop returned nonzero; owned absence established")
     except BaseException as error:
         errors.append("container cleanup: " + repr(error))
-    if child.poll() is None:
-        try:
-            os.killpg(child.pid, signal.SIGTERM)
-            child.wait(timeout=10)
-        except subprocess.TimeoutExpired:
-            os.killpg(child.pid, signal.SIGKILL)
-            child.wait(timeout=10)
-        except ProcessLookupError:
-            child.wait(timeout=10)
+    return errors
+
+
+def cleanup(child, name, nonce, cidfile):
+    errors = cleanup_container(name, nonce, cidfile)
+    try:
+        if child.poll() is None:
+            try:
+                os.killpg(child.pid, signal.SIGTERM)
+                child.wait(timeout=10)
+            except subprocess.TimeoutExpired:
+                os.killpg(child.pid, signal.SIGKILL)
+                child.wait(timeout=10)
+            except ProcessLookupError:
+                child.wait(timeout=10)
+    except BaseException as error:
+        errors.append("owned foreground cleanup: " + repr(error))
+    finally:
+        # Startup can cross the first absent-name check before the CLI dies.
+        # Rebind owner metadata and establish absence AFTER the CLI reaps.
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    try:
+        require(child.poll() is not None, "owned foreground was not reaped")
+        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
+    except BaseException as error:
+        errors.append("final owned absence: " + repr(error))
     try:
         os.killpg(child.pid, 0)
     except ProcessLookupError:

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/BIT_TRANSPORT_V3.patch

Original UTF-8 bytes: 74854; SHA256 `82ede33a526929c0d70fb2d776ff1227079fde12e321b8465a6f8f5006c70b59`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ capture_v3.py
@@ -0,0 +1,315 @@
+"""ROOT ONLY original27B layer0 GDN module; never a full-model constructor."""
+import gc
+import importlib.metadata
+import inspect
+import json
+import mmap
+import os
+from pathlib import Path
+import resource
+import struct
+import sys
+import time
+from support_v3 import HERE, MODEL_SHA, TORCH_GIT, bounds, digest, fd_sha, fingerprint, metadata, read, require, sha, source_header, unique, write
+
+SOURCE = Path('/source')
+OUTPUT = Path('/output/captured')
+EMBED = 'model.language_model.embed_tokens.weight'
+INPUT_NORM = 'model.language_model.layers.0.input_layernorm.weight'
+PREFIX = 'model.language_model.layers.0.linear_attn.'
+SOURCE_OWNER = None # Original FD/map/storage owner retained until owned process exit.
+
+class SourceOwner:
+    def __init__(self, plan):
+        self.plan, self.fds, self.headers, self.before = plan, {}, {}, {}
+        self.maps, self.tensors, self.readsets, self.versions = {}, {}, {}, {}
+        for name,pin in plan['metadata'].items():
+            read(SOURCE/name,pin=pin)
+        self.index=metadata(SOURCE/'model.safetensors.index.json')
+        for shard,pin in plan['selected_shards'].items():
+            fd=os.open(SOURCE/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            self.fds[shard]=fd
+            self.before[shard]=fingerprint(os.fstat(fd))
+            require(os.fstat(fd).st_size==pin['file']['bytes'] and
+                    fd_sha(fd,0,pin['file']['bytes'])==pin['file']['sha256'],
+                    'complete same opened original selected shard before library initialization')
+            n,raw=source_header(fd,pin)
+            require(fingerprint(os.fstat(fd))==self.before[shard],
+                    'same selected FD unchanged across full hash and bounded header read')
+            self.headers[shard]=(8+n,json.loads(raw,object_pairs_hook=unique))
+
+    def load_views(self,torch):
+        for name,item in self.plan['tensors'].items():
+            shard=item['shard'];begin,header=self.headers[shard];entry=header[name]
+            require(self.index['weight_map'][name]==shard and entry['dtype']=='BF16' and
+                    entry['shape']==item['shape'] and begin+entry['data_offsets'][0]==item['absolute_offset'] and
+                    entry['data_offsets'][1]-entry['data_offsets'][0]==item['bytes'],
+                    'actual exact source role/name/geometry/full-byte extent')
+            # COW protects source files; every window is only its exact parameter
+            # plus the leading alignment bytes, not a whole-shard mapping.
+            mapping=mmap.mmap(self.fds[shard],item['mapping_bytes'],access=mmap.ACCESS_COPY,
+                              offset=item['page_aligned_offset'])
+            self.maps[name]=mapping
+            value=torch.frombuffer(mapping,dtype=torch.bfloat16,count=item['bytes']//2,
+                                   offset=item['view_offset']).reshape(item['shape'])
+            require(value.is_contiguous() and value.device.type=='cpu' and not value.is_meta,
+                    'authentic contiguous BF16 source view')
+            self.tensors[name]=value
+            self.versions[name]=value._version
+            if name==EMBED:
+                self.readsets[name]={str(token):fd_sha(self.fds[shard],item['absolute_offset']+token*10240,10240)
+                                     for token in (11,353)}
+            else:
+                self.readsets[name]=fd_sha(self.fds[shard],item['absolute_offset'],item['bytes'])
+
+    def mapped_sha(self,name,begin,count):
+        value=__import__('hashlib').sha256();view=memoryview(self.maps[name])
+        try:
+            for offset in range(0,count,1024*1024):
+                value.update(view[begin+offset:begin+min(count,offset+1024*1024)])
+        finally:view.release()
+        return value.hexdigest()
+
+    def finish(self):
+        for name,item in self.plan['tensors'].items():
+            require(self.tensors[name]._version==self.versions[name],'original source tensor version unchanged')
+            if name==EMBED:
+                actual={str(t):self.mapped_sha(name,item['view_offset']+t*10240,10240) for t in (11,353)}
+            else:actual=self.mapped_sha(name,item['view_offset'],item['bytes'])
+            require(actual==self.readsets[name],'original mapped learned payload unchanged: '+name)
+        after={}
+        for shard,fd in self.fds.items():
+            pin=self.plan['selected_shards'][shard]['file']
+            after[shard]=fingerprint(os.fstat(fd))
+            require(after[shard]==self.before[shard] and fd_sha(fd,0,pin['bytes'])==pin['sha256'],
+                    'same selected full source FD unchanged after original operators')
+        return after
+
+def main():
+    global SOURCE_OWNER
+    started=time.monotonic()
+    require(not OUTPUT.exists(),'exclusive output directory');OUTPUT.mkdir(mode=0o700)
+    report={'schema':'zerogravity.original27B.layer0-gdn.public-reference.v1','status':'UNQUALIFIED',
+            'steps':[],'tokens':[11,353],'parameter_profile':'original27B-raw-source-bf16-v1',
+            'full_model_StateRoot_native_exp_GPU_eligible':False}
+    try:
+        require(sys.byteorder=='little','original little-endian source')
+        cpus=[int(v) for v in os.environ['ZG_ALLOWED_CPUS'].split(',')]
+        require(len(cpus)==len(set(cpus))==4,'four explicit CPUs');os.sched_setaffinity(0,cpus)
+        require(sorted(os.sched_getaffinity(0))==sorted(cpus),'actual affinity before importing Torch')
+        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void' and
+                os.environ.get('ATEN_CPU_CAPABILITY')=='avx2' and os.environ.get('USE_HUB_KERNELS')=='0' and
+                not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
+                'declared original CPU reference/no external math interposition')
+        require(sha(HERE/'source-plan.json')==os.environ['ZG_SOURCE_PLAN_SHA'],'exact source metadata plan')
+        plan=metadata(HERE/'source-plan.json')
+        derived=bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+        report.update(source_plan_sha256=sha(HERE/'source-plan.json'),source_repo=plan['repository'],
+                      source_revision=plan['revision'],snapshot=plan['snapshot'],pre_library_bounds=derived,
+                      native_ResourceArbiter_admission_qualified=False)
+        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
+        SOURCE_OWNER=SourceOwner(plan)
+        # No library/model initialization or parameter view precedes the bounds.
+        import torch
+        from torch.nn import functional as F
+        from safetensors.torch import save_file
+        from transformers import Qwen3_5Config, DynamicCache
+        from transformers.models.qwen3_5 import modeling_qwen3_5 as original
+        from observer import Observation, ScopedDispatch, info, total_bytes
+        require(torch.__version__=='2.13.0+cu129' and torch.version.git_version==TORCH_GIT and
+                importlib.metadata.version('transformers')=='5.17.0' and
+                importlib.metadata.version('safetensors')=='0.8.0' and not torch.cuda.is_available(),
+                'exact existing pinned CPU runtime')
+        torch.set_num_threads(4);torch.set_num_interop_threads(1)
+        require(torch.backends.cpu.get_cpu_capability()=='AVX2' and torch.backends.mkldnn.enabled,
+                'original CPU AVX2 policy')
+        require(sha(Path(original.__file__))==MODEL_SHA,'exact original declared architecture source')
+        actual_modules={'modeling_qwen3_5.py':original,
+                        'configuration_qwen3_5.py':sys.modules[Qwen3_5Config.__module__],
+                        'cache_utils.py':sys.modules[DynamicCache.__module__],
+                        'hub_kernels.py':sys.modules['transformers.integrations.hub_kernels']}
+        for name,module in actual_modules.items():
+            require(sha(Path(module.__file__))==plan['runtime_source'][name]['sha256'],
+                    'pinned original module '+name)
+        for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule',
+                     'causal_conv1d_fn','causal_conv1d_update'):
+            function=getattr(original,name);found=False
+            while function is not None:
+                closure=inspect.getclosurevars(function).nonlocals
+                if 'is_new_implementation' in closure:
+                    require(closure['is_new_implementation'] is False and
+                            Path(closure['implementation'].__code__.co_filename).resolve()==Path(original.__file__).resolve(),
+                            'refuse optional non-original package implementation: '+name)
+                    found=True
+                function=getattr(function,'__wrapped__',None)
+            require(found,'authentic original public function wrapper: '+name)
+        config=metadata(SOURCE/'config.json')
+        require(config['architectures']==['Qwen3_5ForConditionalGeneration'] and config['model_type']=='qwen3_5' and
+                config['text_config']['model_type']=='qwen3_5_text','actual27B declared original architecture')
+        cfg=Qwen3_5Config.from_dict(config).text_config
+        require((cfg.hidden_size,cfg.linear_num_key_heads,cfg.linear_num_value_heads,cfg.linear_key_head_dim,
+                 cfg.linear_value_head_dim,cfg.linear_conv_kernel_dim)==(5120,16,48,128,128,4),
+                'original27B GDN exact geometry')
+        SOURCE_OWNER.load_views(torch)
+        with torch.device('meta'):
+            gdn=original.Qwen3_5GatedDeltaNet(cfg,layer_idx=0)
+            input_norm=original.Qwen3_5RMSNorm(5120,eps=cfg.rms_norm_eps)
+        parameters={name.removeprefix(PREFIX):value for name,value in SOURCE_OWNER.tensors.items() if name.startswith(PREFIX)}
+        require(set(parameters)==set(dict(gdn.named_parameters())) and len(parameters)==9 and
+                not dict(gdn.named_buffers()),'every original GDN parameter supplied, no random/unfilled fallback')
+        loaded=gdn.load_state_dict(parameters,strict=True,assign=True)
+        require(not loaded.missing_keys and not loaded.unexpected_keys,'strict complete source parameter assignment')
+        input_norm.load_state_dict({'weight':SOURCE_OWNER.tensors[INPUT_NORM]},strict=True,assign=True)
+        gdn.eval();input_norm.eval()
+        parameter_audit={}
+        for name,value in gdn.named_parameters():
+            raw=SOURCE_OWNER.tensors[PREFIX+name]
+            require(value.dtype==torch.bfloat16 and not value.is_meta and value.data_ptr()==raw.data_ptr() and
+                    tuple(value.shape)==tuple(raw.shape),'exact raw BF16 original parameter, no cast/copy: '+name)
+            parameter_audit[name]={**info(value),'pointer':value.data_ptr(),'version':value._version,
+                                   'source_readset_sha256':SOURCE_OWNER.readsets[PREFIX+name]}
+        require(input_norm.weight.data_ptr()==SOURCE_OWNER.tensors[INPUT_NORM].data_ptr() and
+                input_norm.weight.dtype==torch.bfloat16 and gdn.norm.weight.dtype==torch.bfloat16 and
+                gdn.A_log.dtype==gdn.dt_bias.dtype==torch.bfloat16,
+                'original BF16 gamma/A_log/dt_bias/inputnorm stay untouched')
+        baseline={};observations=[];hooks=[];active=[None]
+        from tensor_bits_v3 import tensor_bits
+        def bits(value):return tensor_bits(value,torch)
+        def same(a,b,message):
+            require(a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and bits(a)==bits(b),message)
+        def projection_hook(name):
+            def hook(module,args,result):
+                if active[0] is not None:
+                    active[0].tensor('projection.'+name+'.input',args[0])
+                    active[0].tensor('projection.'+name+'.output',result)
+            return hook
+        def kept(name,value):
+            count=value.numel()*value.element_size()
+            require(total_bytes[0]+count<=128*1024**2,'bounded combined plain/observed tensor payload')
+            baseline[name]=value.detach().contiguous().clone();total_bytes[0]+=count
+        try:
+            kept('source.A_log',gdn.A_log)
+            kept('source.dt_bias',gdn.dt_bias)
+            kept('source.norm.weight',gdn.norm.weight)
+            with torch.inference_mode():
+                for mode in ('plain','observed'):
+                    cache=DynamicCache(config=cfg)
+                    require(len(cache.layers)==64 and not cache.has_previous_state(0,0),'real original cache starts empty')
+                    if mode=='observed':
+                        for name in ('in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'):
+                            hooks.append(gdn.get_submodule(name).register_forward_hook(projection_hook(name)))
+                    for step,token in enumerate((11,353)):
+                        embedding=F.embedding(torch.tensor([[token]],dtype=torch.int64),SOURCE_OWNER.tensors[EMBED])
+                        normalized=input_norm(embedding)
+                        require(embedding.shape==normalized.shape==(1,1,5120),'actual trained original embedding/InputNorm')
+                        require(cache.has_previous_state(0,0) is (step==1),'real initial then cached source dispatch')
+                        if mode=='plain':
+                            output=gdn(normalized,cache_params=cache)
+                            for name,value in [('embedding',embedding),('input_norm',normalized),('output',output),
+                                               ('conv',cache.layers[0].conv_states[0]),('state',cache.layers[0].recurrent_states[0])]:
+                                kept(f'step-{step}.{name}',value)
+                        else:
+                            obs=Observation(step,gdn);observations.append(obs);active[0]=obs
+                            obs.tensor('prefix.embedding',embedding);obs.tensor('prefix.input_norm',normalized)
+                            require(sys.gettrace() is None,'do not replace an existing tracer')
+                            previous=sys.gettrace();mode_observer=ScopedDispatch(obs);obs.mode=mode_observer
+                            try:
+                                sys.settrace(obs.trace)
+                                with mode_observer:
+                                    output=gdn(normalized,cache_params=cache)
+                            finally:
+                                sys.settrace(previous);obs.mode=None;active[0]=None
+                            obs.tensor('cache.after.conv',cache.layers[0].conv_states[0])
+                            obs.tensor('cache.after.state',cache.layers[0].recurrent_states[0])
+                            obs.tensor('module.output',output)
+                            directory=OUTPUT/f'step-{step}';directory.mkdir()
+                            archive=directory/'actual-gdn.safetensors'
+                            save_file(obs.tensors,str(archive),metadata={'scope':'actual original27B layer0 frames; prevalidation only'})
+                            for name,value in obs.tensors.items():obs.meta[name]['raw_sha256']=digest(bits(value))
+                            record={'step':step,'teacher_token':token,'actual_calls':dict(obs.calls),'events':obs.events,
+                                    'dispatch':obs.dispatch,'tensors':obs.meta,'payload_bytes':obs.bytes,
+                                    'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
+                                    'plain_observed_all_outputs_and_state_exact':False}
+                            write(directory/'prevalidation.json',record)
+                            phase='chunk' if step==0 else 'recurrent'
+                            expected={'Qwen3_5GatedDeltaNet.forward':1,'Qwen3_5RMSNormGated.forward':1,
+                                      'l2norm':2,'torch_'+phase+'_gated_delta_rule':1,
+                                      'causal_conv1d_fn' if step==0 else 'causal_conv1d_update':1}
+                            require(dict(obs.calls)==expected,'actual unchanged original initial/cached call schedule')
+                            for name,value in [('embedding',embedding),('input_norm',normalized),('output',output),
+                                               ('conv',cache.layers[0].conv_states[0]),('state',cache.layers[0].recurrent_states[0])]:
+                                same(baseline[f'step-{step}.{name}'],value,'plain/observed complete actual '+name)
+                            same(obs.tensors[phase+'.return.actual_return_1'],cache.layers[0].recurrent_states[0],
+                                 'original public returned state equals real original cache')
+                            if step:
+                                same(observations[0].tensors['chunk.return.actual_return_1'],
+                                     obs.tensors['recurrent.input.initial_state'],'actual initial return supplies real cached prior')
+                            record['plain_observed_all_outputs_and_state_exact']=True
+                            report['steps'].append(record)
+                            total_bytes[0]+=obs.bytes
+                        for layer in cache.layers[1:]:
+                            require(not getattr(layer,'is_conv_states_initialized',[]) or
+                                    not any(layer.is_conv_states_initialized),'no unrelated layer state allocated')
+        finally:
+            active[0]=None
+            for hook in hooks:hook.remove()
+            require(sys.gettrace() is None,'original tracer restored')
+        save_file(baseline,str(OUTPUT/'plain.safetensors'),metadata={'scope':'genuine unobserved two-teacher module outputs/cache'})
+        prior=metadata(Path('/prefix-reference/reference.json'),pin=plan['prior_public_prefix'])
+        require(prior['source_repo']==plan['repository'] and prior['source_revision']==plan['revision'] and
+                prior['snapshot']==plan['snapshot'],'independent original27B prefix reference source binding')
+        prefix_checks=[]
+        for step,token in enumerate((11,353)):
+            old=prior['steps'][step]
+            require(old['token']==token,'independent prefix teacher binding')
+            for stage,actual in [('embedding',baseline[f'step-{step}.embedding']),
+                                 ('normalized',baseline[f'step-{step}.input_norm']),
+                                 ('projected',observations[step].tensors['projection.in_proj_qkv.output'])]:
+                row=old[stage];expected=read(Path('/prefix-reference')/row['file'],pin=row)
+                widened=b''.join((word<<16).to_bytes(4,'little') for word, in struct.iter_unpack('<H',bits(actual)))
+                require(actual.dtype==torch.bfloat16 and len(widened)==row['bytes'] and widened==expected,
+                        'all prior genuine original27B prefix words: '+stage)
+                prefix_checks.append({'step':step,'stage':stage,'words':actual.numel(),'bit_exact':True,'prior':row})
+        for name,value in gdn.named_parameters():
+            require(value.data_ptr()==parameter_audit[name]['pointer'] and value._version==parameter_audit[name]['version'],
+                    'original source parameter storage/version unchanged')
+        after=SOURCE_OWNER.finish()
+        maps=Path('/proc/self/maps').read_text()
+        require(len(maps.encode())<=2*1024**2 and '/nix/store/' not in maps,'bounded actual container mapping closure')
+        objects=sorted({line.split(None,5)[5] for line in maps.splitlines() if len(line.split(None,5))==6 and
+                        line.split(None,5)[5].startswith('/') and '.so' in Path(line.split(None,5)[5]).name})
+        require(len(objects)<=256,'bounded actual loaded-library count')
+        dsos={};total_dso=0
+        for path in objects:
+            size=Path(path).stat().st_size;total_dso+=size
+            require(size<=2*1024**3 and total_dso<=16*1024**3 and ' (deleted)' not in path,'bounded actual mapped provider closure')
+            dsos[path]={'bytes':size,'sha256':sha(path)}
+        report.update(status='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION',
+            source_FDs_before=SOURCE_OWNER.before,source_FDs_after=after,source_readsets=SOURCE_OWNER.readsets,
+            all_selected_FDs_and_mapped_parameters_unchanged=True,all_source_parameters_BF16=True,
+            parameter_audit=parameter_audit,source_owner_retained_until_owned_process_exit=True,
+            norm_profile={'class':'Qwen3_5RMSNormGated','heads':48,'head_width':128,
+                          'raw_gamma_dtype':'torch.bfloat16','weighted_dtype':'torch.bfloat16',
+                          'gate_product_dtype':'torch.float32','return_dtype':'torch.bfloat16',
+                          'epsilon':gdn.norm.variance_epsilon,'activation':gdn.norm.activation},
+            payload_bytes=total_bytes[0],
+            plain_archive={'bytes':(OUTPUT/'plain.safetensors').stat().st_size,'sha256':sha(OUTPUT/'plain.safetensors')},
+            hooks_removed=len(hooks),runtime={'torch':torch.__version__,'torch_git':torch.version.git_version,
+                'transformers':importlib.metadata.version('transformers'),'cpu_capability':torch.backends.cpu.get_cpu_capability(),
+                'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
+                'affinity':sorted(os.sched_getaffinity(0)),'cuda_available':False,'modeling_sha256':MODEL_SHA,
+                'mapped_shared_objects':dsos,'proc_status':Path('/proc/self/status').read_text(),
+                'proc_cgroup':Path('/proc/self/cgroup').read_text(),'proc_mountinfo':Path('/proc/self/mountinfo').read_text()},
+            scope='Original27B private layer0 embedding/InputNorm/GDN two-teacher public reference only; all raw source BF16. No full-model logits/112 StateRoots, native exp/provider allocation eligibility, production memory admission, GPU or performance.')
+        report['prior_prefix_checks']=prefix_checks
+        require(report['payload_bytes']<=128*1024**2,'bounded total actual plain/observed payload')
+    except BaseException as error:
+        report['failure']=repr(error)
+        raise
+    finally:
+        report['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
+        write(OUTPUT/'manifest.json',report)
+    print('ORIGINAL27B_LAYER0_GDN_CAPTURE_ONLY',flush=True)
+
+if __name__=='__main__':main()
--- /dev/null
+++ run_reference_v3.py
@@ -0,0 +1,134 @@
+"""ROOT ONLY owned CPU reference, whole source before/after, no model on import."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import signal
+import stat
+import subprocess
+import time
+import uuid
+from support_v3 import HERE, IMAGE, IMAGE_DIGEST, bounds, fd_sha, fingerprint, frozen, metadata, read, require, sha, write
+
+def source_open(plan):
+    """ROOT-only complete source hashes, bounded chunks; same FDs kept open."""
+    handles, rows = {}, []
+    try:
+        for row in plan['source_files']:
+            require(Path(row['name']).name==row['name'],'contained full source filename')
+            fd=os.open(Path(plan['source_directory'])/row['name'],os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            handles[row['name']]=fd
+            before=os.fstat(fd)
+            require(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'] and
+                    fd_sha(fd,0,row['bytes'])==row['sha256'] and fingerprint(before)==fingerprint(os.fstat(fd)),
+                    'actual full original source bytes/FD immutable before reference')
+            rows.append({'name':row['name'],'bytes':row['bytes'],'sha256':row['sha256'],
+                         'fd_identity':fingerprint(before)})
+        require(sum(row['bytes'] for row in plan['source_files'] if row['name'].endswith('.safetensors'))
+                ==plan['weight_bytes']==55563006776,'all18 authentic original weight shards')
+        return handles,rows
+    except BaseException:
+        for fd in handles.values():os.close(fd)
+        raise
+
+def source_after(handles,rows):
+    after=[]
+    for row in rows:
+        fd=handles[row['name']]
+        require(fingerprint(os.fstat(fd))==row['fd_identity'] and
+                fd_sha(fd,0,row['bytes'])==row['sha256'],'same full original source FD after reference')
+        after.append(dict(row))
+    return after
+
+def tree_size(root):
+    total=0
+    for path in root.rglob('*'):
+        require(not path.is_symlink(),'no output artifact symlink')
+        if path.is_file():
+            require(stat.S_ISREG(path.stat().st_mode),'regular actual artifact')
+            total+=path.stat().st_size
+    require(total<=256*1024**2-128*1024,'bounded whole attempt with reserved completion metadata')
+    return total
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--new-output',type=Path,required=True)
+    args=parser.parse_args()
+    def deadline(_signal,_frame):raise TimeoutError('finite whole source/reference/lifecycle deadline')
+    signal.signal(signal.SIGALRM,deadline);signal.alarm(2400)
+    def cancelled(_signal,_frame):raise InterruptedError('owned reference cancelled; retire only owned container/process')
+    signal.signal(signal.SIGTERM,cancelled)
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+    require(args.new_output.is_absolute() and args.new_output.parent.resolve()==HERE,
+            'exclusive owned attempt directory');args.new_output.mkdir(mode=0o700)
+    cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus)
+    require(sha(HERE/'owned_container_v4.py')=='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
+            'byte-identical reviewed owned container lifecycle')
+    spec=importlib.util.spec_from_file_location('gdn27_owned',HERE/'owned_container_v4.py')
+    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
+    nonce=uuid.uuid4().hex;owned_name='zg-original27-gdn-'+nonce;cid=args.new_output/'owned.cid'
+    receipt={'status':'UNQUALIFIED','freeze_manifest_sha256':freeze,'source_plan_sha256':sha(HERE/'source-plan.json'),
+             'expected_cpus':cpus,'owned_name':owned_name,'owned_nonce':nonce,'image_id':IMAGE,'image_digest':IMAGE_DIGEST,
+             'no_GPU_or_full_model_or_native_provider_admission':True}
+    handles={};child=log=None;failure=None;started=time.monotonic()
+    try:
+        handles,before=source_open(plan);receipt['source_before']=before
+        inspected=helper.command(helper.PODMAN+['image','inspect',IMAGE],15)
+        require(inspected.returncode==0,'existing approved image; no pull')
+        (args.new_output/'image-inspect.json').write_bytes(inspected.stdout)
+        image,=json.loads(inspected.stdout)
+        require(image['Id']==IMAGE.removeprefix('sha256:') and image['Digest']==IMAGE_DIGEST,'pinned actual image')
+        prefix=Path(plan['prior_public_prefix']['path']).parent
+        env={'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'void','ATEN_CPU_CAPABILITY':'avx2',
+             'ZG_ALLOWED_CPUS':','.join(map(str,cpus)),'ZG_SOURCE_PLAN_SHA':sha(HERE/'source-plan.json'),
+             'USE_HUB_KERNELS':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
+             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE',
+             'PYTHONPATH':'/probe','PYTHONDONTWRITEBYTECODE':'1','LD_PRELOAD':'','LD_AUDIT':'','VLLM_PLUGINS':''}
+        argv=helper.PODMAN+['run','--name',owned_name,'--label',helper.LABEL+'='+nonce,'--cidfile',str(cid),
+            '--rm','--pull=never','--network=none','--read-only','--cap-drop=all','--security-opt=no-new-privileges',
+            '--memory=12g','--memory-swap=12g','--cpus=4','--pids-limit=256',
+            '--ulimit','fsize=134217728:134217728','--tmpfs','/tmp:rw,nosuid,nodev,size=1g']
+        for key,value in env.items():argv+=['-e',key+'='+value]
+        argv+=['-v',str(HERE)+':/probe:ro','-v',plan['source_directory']+':/source:ro',
+               '-v',str(prefix)+':/prefix-reference:ro','-v',str(args.new_output)+':/output:rw',
+               '--entrypoint','python3',IMAGE,'/probe/capture_v3.py']
+        receipt['argv']=argv;write(args.new_output/'prepared-invocation.json',receipt)
+        try:
+            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
+            receipt['owned_cli_pid']=child.pid
+            log=helper.Log(child.stdout,args.new_output/'execution.log')
+            receipt['exit_code']=log.observe(child,600)
+        except BaseException as error:failure=error
+        finally:
+            receipt['cleanup_errors']=helper.cleanup(child,owned_name,nonce,cid,receipt) if child is not None else []
+            if log is not None:
+                try:log.finish()
+                except BaseException as error:receipt['cleanup_errors'].append('log cleanup: '+repr(error))
+                finally:receipt['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
+            elif child is not None and child.stdout is not None:
+                try:child.stdout.close()
+                except BaseException as error:receipt['cleanup_errors'].append('uninitialized pipe close: '+repr(error))
+        require(failure is None and receipt.get('exit_code')==0 and receipt['cleanup_errors']==[],
+                'owned original public reference/cleanup failed; preserve this attempt')
+        receipt['source_after']=source_after(handles,before)
+        require(receipt['source_after']==before and frozen()==freeze,'all source/preparation files unchanged')
+        receipt['precompletion_artifact_bytes']=tree_size(args.new_output)
+        receipt['status']='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        if failure is None:failure=error
+    finally:
+        failures=[]
+        for fd in handles.values():
+            try:os.close(fd)
+            except BaseException as error:failures.append(repr(error))
+        receipt['source_FD_cleanup_errors']=failures
+        receipt['source_FDs_closed']=not failures
+        receipt['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        if failure is not None:receipt['failure']=repr(failure)
+        if failure is not None or failures:receipt['status']='FAIL_UNQUALIFIED'
+        write(args.new_output/'invocation.json',receipt)
+    require(receipt['status']=='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION',
+            'reference attempt unqualified; retained original evidence')
+
+if __name__=='__main__':main()
--- /dev/null
+++ support_v3.py
@@ -0,0 +1,146 @@
+"""Bounded metadata/source transport; no Torch/provider/model import."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import struct
+
+HERE = Path(__file__).resolve().parent
+IMAGE = 'sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe'
+IMAGE_DIGEST = 'sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634'
+MODEL_SHA = '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'
+TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+def digest(raw):
+    return hashlib.sha256(raw).hexdigest()
+
+def sha(path):
+    with Path(path).open('rb') as file:
+        return hashlib.file_digest(file, 'sha256').hexdigest()
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON key')
+        result[key] = value
+    return result
+
+def read(path, cap=4*1024*1024, pin=None):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as file:
+        before = os.fstat(file.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
+        raw = file.read(cap+1)
+        after = os.fstat(file.fileno())
+    require(len(raw) <= cap and fingerprint(before) == fingerprint(after), 'immutable metadata FD')
+    if pin is not None:
+        require(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'exact metadata pin')
+    return raw
+
+def metadata(path, cap=4*1024*1024, pin=None):
+    return json.loads(read(path, cap, pin), object_pairs_hook=unique)
+
+def write(path, value):
+    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
+    require(len(raw) <= 4*1024*1024, 'bounded output metadata')
+    with Path(path).open('xb') as file:
+        os.fchmod(file.fileno(),0o600)
+        file.write(raw)
+
+def fingerprint(meta):
+    return [meta.st_dev,meta.st_ino,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns]
+
+def fd_sha(fd, begin, count):
+    value = hashlib.sha256()
+    for offset in range(0,count,1024*1024):
+        raw = os.pread(fd,min(1024*1024,count-offset),begin+offset)
+        require(len(raw) == min(1024*1024,count-offset), 'short source FD read')
+        value.update(raw)
+    return value.hexdigest()
+
+def source_header(fd, pin):
+    """Validate bounded original header length before the body pread."""
+    before = os.fstat(fd)
+    require(stat.S_ISREG(before.st_mode) and before.st_size == pin['file']['bytes'],
+            'exact regular selected source FD')
+    prefix = os.pread(fd, 8, 0)
+    require(len(prefix) == 8, 'complete source header length word')
+    n, = struct.unpack('<Q', prefix)
+    expected = pin['header_bytes']
+    require(type(expected) is int and 1 < expected <= 8*1024*1024 and
+            n == expected and 8+n <= before.st_size,
+            'bounded frozen source header length before body read')
+    raw = os.pread(fd, n, 8)
+    require(len(raw) == n and digest(raw) == pin['header_sha256'] and
+            fingerprint(before) == fingerprint(os.fstat(fd)),
+            'complete exact immutable original source header')
+    return n, raw
+
+def bounds(plan, page):
+    """Integer geometry before importing libraries or allocating parameter maps."""
+    limits = plan['bounds_before_library_or_views']
+    require(page == limits['page_size'] == 4096, 'pinned page geometry')
+    require(len(plan['tensors']) == 11 and len(plan['selected_shards']) == 2, 'exact complete selected roles')
+    mapped, params = 0, 0
+    for name, item in plan['tensors'].items():
+        require(item['dtype'] == 'BF16' and type(item['shape']) is list, 'original BF16 source policy')
+        count = 1
+        for axis in item['shape']:
+            require(type(axis) is int and 0 < axis <= 248320, 'bounded original tensor axis')
+            count *= axis
+        start, length = item['absolute_offset'], item['bytes']
+        require(type(start) is type(length) is int and start >= 8 and length == count*2,
+                'full original typed extent')
+        aligned = start-start%page
+        require(item['page_aligned_offset'] == aligned and item['view_offset'] == start-aligned and
+                item['mapping_bytes'] == length+start-aligned and
+                start+length <= plan['selected_shards'][item['shard']]['file']['bytes'],
+                'exact bounded parameter window; no whole shard mapping')
+        mapped += item['mapping_bytes']
+        if '.linear_attn.' in name:
+            params += length
+    touched = params + 10240 + 20480
+    require(params == limits['selected_parameter_bytes'] == 231752128 and
+            mapped == limits['aggregate_window_mapping_bytes'] == 2774582792 and
+            touched == limits['touched_source_payload_bytes'] == 231782848 and
+            mapped <= limits['mapped_VA_cap_bytes'] == 3*1024**3 and
+            touched <= limits['derived_touched_source_cap_bytes'] == 256*1024**2,
+            'pre-library exact source windows and selected resident payload bound')
+    virtual_envelope=mapped+limits['runtime_VA_allowance_bytes']+limits['conservative_operator_working_envelope_bytes']+limits['capture_payload_cap_bytes']
+    physical_estimate=touched+limits['runtime_resident_allowance_bytes']+limits['conservative_operator_working_envelope_bytes']+limits['capture_payload_cap_bytes']+limits['control_allowance_bytes']
+    require(virtual_envelope < limits['process_AS_cap_bytes'] == 12*1024**3 and
+            physical_estimate < limits['configured_container_RAM_bytes'] and
+            limits['configured_container_RAM_bytes'] == 12*1024**3,
+            'finite explicit reference containment; not native physical admission')
+    return {'mapped_bytes':mapped,'selected_parameter_bytes':params,'touched_source_bytes':touched,
+            'virtual_envelope_bytes':virtual_envelope,'physical_envelope_estimate_bytes':physical_estimate,
+            'native_ResourceArbiter_admission_qualified':False}
+
+def frozen():
+    require(sha(HERE/'prepared-manifest.json') ==
+            '09461af63adc9e53a279a40d6608efcabeaa3d8a405f04d9a55b220bf73ef0f4',
+            'immutable original reference freeze')
+    manifest = metadata(HERE/'prepared-manifest.json')
+    for row in manifest['files']:
+        read(HERE/row['file'],pin=row)
+    for row in manifest['parent_metadata']:
+        read(row['path'],pin=row)
+    require(sha(HERE/'prepared-manifest-v2.json') ==
+            '9bc27ef108e9fdffb440c6a1138e76327e1ce53e7d1793acd9b82dd1f168eade',
+            'immutable header-bound V2 freeze')
+    followup = metadata(HERE/'prepared-manifest-v2.json')
+    require(followup['original_manifest_sha256'] == sha(HERE/'prepared-manifest.json'),
+            'additive header-bound followup parent identity')
+    for row in followup['files']:
+        read(HERE/row['file'], pin=row)
+    transport = metadata(HERE/'prepared-manifest-v3.json')
+    require(transport['original_manifest_sha256'] == sha(HERE/'prepared-manifest.json') and
+            transport['header_bound_manifest_sha256'] == sha(HERE/'prepared-manifest-v2.json'),
+            'additive integer bit-transport parent identity')
+    for row in transport['files']:
+        read(HERE/row['file'], pin=row)
+    return sha(HERE/'prepared-manifest-v3.json')
--- /dev/null
+++ validate_reference_v3.py
@@ -0,0 +1,252 @@
+"""ROOT ONLY stdlib full-byte/source/lifecycle validation; no tensor arithmetic."""
+import argparse
+import json
+import os
+from pathlib import Path
+import struct
+from support_v3 import HERE, MODEL_SHA, TORCH_GIT, bounds, digest, fd_sha, fingerprint, frozen, metadata, read, require, sha, source_header, unique, write
+
+PREFIX='model.language_model.layers.0.linear_attn.'
+
+def source_readsets(plan,report):
+    """ROOT-only actual indexed source bytes, bounded read; no Torch or math."""
+    require(set(report['source_readsets'])==set(plan['tensors']),'all actual source raw readsets')
+    for shard,pin in plan['selected_shards'].items():
+        fd=os.open(Path(plan['source_directory'])/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+        try:
+            before=fingerprint(os.fstat(fd))
+            source_header(fd,pin)
+            require(fingerprint(os.fstat(fd))==before,
+                    'same authenticated selected source header FD')
+            for name,item in plan['tensors'].items():
+                if item['shard']!=shard:continue
+                if name=='model.language_model.embed_tokens.weight':
+                    actual={str(t):fd_sha(fd,item['absolute_offset']+t*10240,10240) for t in (11,353)}
+                else:actual=fd_sha(fd,item['absolute_offset'],item['bytes'])
+                require(actual==report['source_readsets'][name],'independent actual raw source readset: '+name)
+            require(fingerprint(os.fstat(fd))==before,'original FD unchanged during independent source-range validation')
+        finally:os.close(fd)
+
+def archive(path,pin):
+    raw=read(path,128*1024**2,pin);n,=struct.unpack_from('<Q',raw)
+    require(1<n<=8*1024**2 and 8+n<len(raw),'bounded complete raw archive header')
+    head=json.loads(raw[8:8+n],object_pairs_hook=unique);tensors={};spans=[]
+    for name,item in head.items():
+        if name=='__metadata__':continue
+        dtype,shape=item['dtype'],item['shape'];require(dtype in ('F32','BF16'),'original raw archive dtype')
+        count=1
+        require(type(shape) is list and len(shape)<=8,'bounded actual rank')
+        for dim in shape:
+            require(type(dim) is int and 0<dim<=248320,'bounded actual dimension');count*=dim
+        lo,hi=item['data_offsets'];size=2 if dtype=='BF16' else 4
+        require(type(lo) is type(hi) is int and 0<=lo<hi and hi-lo==count*size and 8+n+hi<=len(raw),
+                'complete original dtype/extent geometry')
+        tensors[name]={'dtype':dtype,'shape':shape,'raw':raw[8+n+lo:8+n+hi]};spans.append((lo,hi))
+    cursor=0
+    for lo,hi in sorted(spans):require(lo==cursor,'all disjoint complete archive bytes');cursor=hi
+    require(8+n+cursor==len(raw),'no trailing/unbound archive bytes')
+    return tensors
+
+def tensor(tensors,name,dtype,shape=None):
+    value=tensors[name]
+    require(value['dtype']==dtype and (shape is None or value['shape']==shape),'actual stage dtype/shape: '+name)
+    return value['raw']
+
+def widened(raw):
+    return b''.join((word<<16).to_bytes(4,'little') for word, in struct.iter_unpack('<H',raw))
+
+def operation(rows,scope,line,name):
+    selected=[row for row in rows if row['scope']==scope and row['source_line']==line and row['operation']==name]
+    require(len(selected)==1,'exact actual selected operator: '+str((scope,line,name)))
+    return selected[0]
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');derived=bounds(plan,4096)
+    invocation=metadata(args.output/'invocation.json');directory=args.output/'captured';report=metadata(directory/'manifest.json')
+    require(invocation['status']=='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION' and
+            invocation['freeze_manifest_sha256']==freeze and invocation['source_plan_sha256']==sha(HERE/'source-plan.json') and
+            invocation['exit_code']==0 and invocation['cleanup_errors']==[] and invocation['source_FD_cleanup_errors']==[] and
+            invocation['source_FDs_closed'] is True and invocation.get('owned_container_absent_after_reap') is True,
+            'actual original source/reference/owned container all retired')
+    cleanup=invocation['foreground_cleanup'];logs=invocation['log_cleanup']
+    require(cleanup['leader_reaped'] is True and cleanup['group_absent'] is True and cleanup['exit_code']==0 and
+            cleanup['all_nonzero_signals_before_reap'] is True and cleanup['stage_errors']==[] and
+            logs['pipe_eof'] is logs['all_closed'] is True and logs['total_bytes']==logs['retained_bytes']<=4*1024**2,
+            'original finite owned process/PGID/EOF cleanup and complete evidence')
+    expected={v['name']:(v['bytes'],v['sha256']) for v in plan['source_files']}
+    require(invocation['source_before']==invocation['source_after'] and
+            {v['name']:(v['bytes'],v['sha256']) for v in invocation['source_before']}==expected and
+            len(invocation['source_before'])==len(expected),'all original source files before/after full SHA coverage')
+    require(report['status']=='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION' and
+            report['schema']=='zerogravity.original27B.layer0-gdn.public-reference.v1' and report['tokens']==[11,353] and
+            report['parameter_profile']=='original27B-raw-source-bf16-v1' and
+            report['source_plan_sha256']==sha(HERE/'source-plan.json') and report['source_repo']==plan['repository'] and
+            report['source_revision']==plan['revision'] and report['snapshot']==plan['snapshot'] and
+            report['pre_library_bounds']==derived and report['native_ResourceArbiter_admission_qualified'] is False and
+            report['full_model_StateRoot_native_exp_GPU_eligible'] is False and report['all_source_parameters_BF16'] is True and
+            report['source_owner_retained_until_owned_process_exit'] is True and report['hooks_removed']==5,
+            'authentic private original27B profile and unchanged eligibility guards')
+    require(report['norm_profile']=={'class':'Qwen3_5RMSNormGated','heads':48,'head_width':128,
+        'raw_gamma_dtype':'torch.bfloat16','weighted_dtype':'torch.bfloat16',
+        'gate_product_dtype':'torch.float32','return_dtype':'torch.bfloat16','epsilon':1e-6,'activation':'silu'},
+        'explicit original27B BF16 gamma round profile')
+    require(report['all_selected_FDs_and_mapped_parameters_unchanged'] is True and
+            report['source_FDs_before']==report['source_FDs_after'] and
+            set(report['source_FDs_before'])==set(plan['selected_shards']),'same exact selected original FD owners')
+    source_readsets(plan,report)
+    expected_parameters={name.removeprefix(PREFIX):item for name,item in plan['tensors'].items() if name.startswith(PREFIX)}
+    require(set(report['parameter_audit'])==set(expected_parameters) and len(expected_parameters)==9,
+            'all and only original learned GDN parameters')
+    for name,item in expected_parameters.items():
+        actual=report['parameter_audit'][name]
+        require(actual['dtype']=='torch.bfloat16' and actual['shape']==item['shape'] and
+                actual['device']=='cpu' and actual['contiguous'] is True and actual['bytes']==item['bytes'] and
+                type(actual['pointer']) is int and actual['pointer']>0 and
+                actual['source_readset_sha256']==report['source_readsets'][PREFIX+name],
+                'actual complete original raw parameter binding: '+name)
+    runtime=report['runtime'];require(runtime['torch']=='2.13.0+cu129' and runtime['torch_git']==TORCH_GIT and
+        runtime['transformers']=='5.17.0' and runtime['cpu_capability']=='AVX2' and runtime['threads']==4 and
+        runtime['interop_threads']==1 and runtime['affinity']==invocation['expected_cpus'] and
+        len(set(runtime['affinity']))==4 and runtime['cuda_available'] is False and runtime['modeling_sha256']==MODEL_SHA,
+        'actual pinned unchanged CPU runtime and four CPUs')
+    dsos=runtime['mapped_shared_objects']
+    require(dsos['/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so']['sha256']==
+            '1b4ed646ce51a4c341808552426a0f9e2e2c04961955759701066d335c1631ab' and
+            dsos['/usr/lib/x86_64-linux-gnu/libm.so.6']['sha256']==
+            'fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0',
+            'actual original exp/CPU/scalar-provider mappings, not native admission')
+    plain=archive(directory/'plain.safetensors',report['plain_archive']);steps=[];payload=sum(len(t['raw']) for t in plain.values())
+    for name,shape in [('A_log',[48]),('dt_bias',[48]),('norm.weight',[128])]:
+        require(digest(tensor(plain,'source.'+name,'BF16',shape))==report['source_readsets'][PREFIX+name],
+                'all original BF16 coefficient/gamma source bytes in reference')
+    require(len(report['steps'])==2,'both true source teachers')
+    for step,record in enumerate(report['steps']):
+        phase='chunk' if step==0 else 'recurrent'
+        require(record['step']==step and record['teacher_token']==[11,353][step] and
+                record['plain_observed_all_outputs_and_state_exact'] is True and record['actual_calls']=={
+                    'Qwen3_5GatedDeltaNet.forward':1,'Qwen3_5RMSNormGated.forward':1,'l2norm':2,
+                    'torch_'+phase+'_gated_delta_rule':1,'causal_conv1d_fn' if step==0 else 'causal_conv1d_update':1},
+                'one original chunk64 first, one actual cached call second')
+        calls=[event for event in record['events'] if event['label']==phase+'.input']
+        require(len(calls)==1 and calls[0]['output_final_state'] is True and
+                calls[0]['use_qk_l2norm_in_kernel'] is True and
+                (step==1 or calls[0]['initial_state'] is None and calls[0]['chunk_size']==64),
+                'actual original phase/initial-state/padded-chunk contract')
+        tensors=archive(directory/f'step-{step}/actual-gdn.safetensors',
+                        {'bytes':record['archive_bytes'],'sha256':record['archive_sha256']});steps.append(tensors)
+        before=metadata(directory/f'step-{step}/prevalidation.json')
+        require(before==dict(record,plain_observed_all_outputs_and_state_exact=False),
+                'raw original diagnostics retained before semantic acceptance')
+        require(set(tensors)==set(record['tensors']),'all original raw observed stages bound')
+        for name,item in tensors.items():
+            meta=record['tensors'][name]
+            require(meta['raw_sha256']==digest(item['raw']) and meta['bytes']==len(item['raw']) and
+                    meta['shape']==item['shape'] and meta['dtype']==('torch.bfloat16' if item['dtype']=='BF16' else 'torch.float32') and
+                    meta['device']=='cpu','actual complete observer payload: '+name)
+            payload+=len(item['raw'])
+        require(sum(len(t['raw']) for t in tensors.values())==record['payload_bytes'],'all actual per-step payload counted')
+        for original,observed,dtype,shape in [('embedding','prefix.embedding','BF16',[1,1,5120]),
+                ('input_norm','prefix.input_norm','BF16',[1,1,5120]),('output','module.output','BF16',[1,1,5120]),
+                ('conv','cache.after.conv','BF16',[1,10240,4]),('state','cache.after.state','F32',[1,48,128,128])]:
+            require(tensor(plain,f'step-{step}.'+original,dtype,shape)==tensor(tensors,observed,dtype,shape),
+                    'all plain/observed source outputs and full actual state: '+original)
+        require(tensor(tensors,phase+'.return.actual_return_1','F32',[1,48,128,128])==
+                tensor(tensors,'cache.after.state','F32',[1,48,128,128]),'actual original returned state equals actual cache')
+        require(tensor(tensors,phase+'.return.actual_return_0','BF16',[1,1,48,128])==
+                tensor(tensors,'norm.input.hidden_states','BF16',[48,128]),'original core return feeds actual gated norm')
+        for name in ('query','key'):
+            before=tensor(tensors,'gdn.pre_beta.'+name,'BF16',[1,1,16,128]);after=tensor(tensors,'gdn.repeated.'+name,'BF16',[1,1,48,128])
+            require(after==b''.join(before[h*256:(h+1)*256]*3 for h in range(16)),
+                    'actual original Q/K repeat_interleave(3), no inferred permutation')
+        rows=record['dispatch'];coef=[v for v in rows if v['scope']=='gdn']
+        require([v['operation'] for v in coef]==['aten.sigmoid.default','aten.to.dtype','aten.exp.default','aten.neg.default',
+                'aten.to.dtype','aten.add.Tensor','aten.softplus.default','aten.mul.Tensor'] and
+                [v['source_line'] for v in coef]==[617]+[619]*7,'actual BF16 A_log .float/coefficient source order')
+        def op_raw(row,which='result',index=0):
+            ref=row[which][index] if which=='arguments' else row[which]
+            require(type(ref) is dict and set(ref)=={'tensor'},'actual selected tensor result/argument')
+            return tensors[ref['tensor']]['raw']
+        require(op_raw(coef[0],'arguments')==tensor(tensors,'projection.in_proj_b.output','BF16',[1,1,48]) and
+                op_raw(coef[0])==tensor(tensors,'gdn.coefficients.beta','BF16',[1,1,48]) and
+                op_raw(coef[1],'arguments')==tensor(plain,'source.A_log','BF16',[48]) and
+                op_raw(coef[1])==widened(tensor(plain,'source.A_log','BF16',[48])) and
+                op_raw(coef[4],'arguments')==tensor(tensors,'projection.in_proj_a.output','BF16',[1,1,48]) and
+                op_raw(coef[4])==widened(tensor(tensors,'projection.in_proj_a.output','BF16',[1,1,48])) and
+                op_raw(coef[5],'arguments',1)==tensor(plain,'source.dt_bias','BF16',[48]) and
+                op_raw(coef[-1])==tensor(tensors,phase+'.input.g','F32',[1,1,48]),
+                'genuine source coefficient producer inputs/results, no reference lookup')
+        beta=tensor(tensors,phase+'.input.beta','BF16',[1,1,48])
+        require(all(0.0<=struct.unpack('<f',(word<<16).to_bytes(4,'little'))[0]<=1.0 for word, in struct.iter_unpack('<H',beta)) and
+                all(value<=0.0 for value, in struct.iter_unpack('<f',tensor(tensors,phase+'.input.g','F32',[1,1,48]))),
+                'actual beta and original negative log-decay ranges')
+        norm={}
+        require(len([row for row in rows if row['scope']=='norm'])==12,'complete unchanged original gated norm operation count')
+        for stage,op,line,dtype,shape in [
+            ('core','aten.to.dtype',227,'F32',[48,128]),('squares','aten.pow.Tensor_Scalar',228,'F32',[48,128]),
+            ('mean','aten.mean.dim',228,'F32',[48,1]),('denominator','aten.add.Tensor',230,'F32',[48,1]),
+            ('inverse','aten.rsqrt.default',230,'F32',[48,1]),('normalized','aten.mul.Tensor',230,'F32',[48,128]),
+            ('normalized_bf16','aten.to.dtype',231,'BF16',[48,128]),('weighted_bf16','aten.mul.Tensor',231,'BF16',[48,128]),
+            ('gate','aten.to.dtype',232,'F32',[48,128]),('silu','aten.silu.default',232,'F32',[48,128]),
+            ('product','aten.mul.Tensor',232,'F32',[48,128]),('return','aten.to.dtype',234,'BF16',[48,128])]:
+            actual=operation(rows,'norm',line,op);name=actual['result']['tensor'];norm[stage]=tensor(tensors,name,dtype,shape)
+        require(tensor(tensors,'norm.input.weight','BF16',[128])==tensor(plain,'source.norm.weight','BF16',[128]) and
+                norm['weighted_bf16']==tensor(tensors,'norm.weighted_bf16.hidden_states','BF16',[48,128]) and
+                norm['return']==tensor(tensors,'norm.return.actual_return_0','BF16',[48,128]),
+                'actual source BF16 gamma product rounds BEFORE FP32 SiLU gate')
+        exp=[v for v in rows if v['scope']=='torch_'+phase+'_gated_delta_rule']
+        require([v['source_line'] for v in exp]==([381,386,413,414,415] if step==0 else [483]) and
+                all(v['operation']=='aten.exp.default' for v in exp),'all actual original initial/cached exponentials')
+        masked=exp[0]['arguments'][0]['tensor'] if step==0 else None
+        for name,item in tensors.items():
+            if name==masked:
+                require(item['shape']==[1,48,1,64,64] and item['dtype']=='F32','actual upper-triangle exp input geometry')
+                for index,(word,) in enumerate(struct.iter_unpack('<I',item['raw'])):
+                    i,j=(index//64)%64,index%64
+                    require(word==0xff800000 if i<j else word&0x7f800000!=0x7f800000,
+                            'ONLY source strictly-upper exp mask may contain canonical negative Inf')
+            else:
+                mask=0x7f80 if item['dtype']=='BF16' else 0x7f800000
+                fmt='<H' if item['dtype']=='BF16' else '<I'
+                require(all(word&mask!=mask for word, in struct.iter_unpack(fmt,item['raw'])),'all other actual raw words finite: '+name)
+        if step:
+            require(tensor(steps[0],'chunk.return.actual_return_1','F32',[1,48,128,128])==
+                    tensor(tensors,'recurrent.input.initial_state','F32',[1,48,128,128]) and
+                    op_raw(exp[0])==tensor(tensors,'recurrent.decay_and_prior.decay_t','F32',[1,48,1,1]),
+                    'actual full initial chunk prior and actual cached exp(g), never zero replacement')
+        else:
+            for name in ('query','key','value','beta','decay'):
+                value=tensors['chunk.padded64.'+name];width=128 if name in ('query','key','value') else 1
+                require(value['dtype']=='F32' and value['shape']==([1,48,64,128] if width==128 else [1,48,64]),
+                        'actual first-token source padded64 layout')
+                for head in range(48):
+                    require(value['raw'][(head*64+1)*width*4:(head+1)*64*width*4]==bytes(63*width*4),
+                            'all63 genuine source padding lanes canonical positive zero')
+    require(payload==report['payload_bytes'] and payload<=128*1024**2,'all actual plain/observed payload bytes counted')
+    require(len(report['prior_prefix_checks'])==6 and all(v['bit_exact'] is True for v in report['prior_prefix_checks']),
+            'independent previously accepted original27B embedding/norm/QKV prefix closure')
+    prior=metadata(plan['prior_public_prefix']['path'],pin=plan['prior_public_prefix'])
+    for step in range(2):
+        for stage,name,dataset in [('embedding',f'step-{step}.embedding',plain),
+                                   ('normalized',f'step-{step}.input_norm',plain),
+                                   ('projected','projection.in_proj_qkv.output',steps[step])]:
+            row=prior['steps'][step][stage]
+            require(widened(tensor(dataset,name,'BF16'))==read(Path(plan['prior_public_prefix']['path']).parent/row['file'],pin=row),
+                    'independent every genuine original prefix word')
+    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}=={
+        'manifest.json','plain.safetensors','step-0/actual-gdn.safetensors','step-0/prevalidation.json',
+        'step-1/actual-gdn.safetensors','step-1/prevalidation.json'},'all captured source/byte artifacts accounted')
+    total_artifacts=0
+    for path in args.output.rglob('*'):
+        require(not path.is_symlink(),'no unknown artifact symlink')
+        if path.is_file():total_artifacts+=path.stat().st_size
+    require(total_artifacts<=256*1024**2-128*1024,'independent whole attempt bound including reserved validation record')
+    require(frozen()==freeze,'all immutable source preparation unchanged')
+    result={'status':'PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY','teachers':[11,353],
+            'heads':48,'cached_input_words_available':804960,'cached_output_words_available':798720,
+            'source_plan_sha256':sha(HERE/'source-plan.json'),'manifest_sha256':sha(directory/'manifest.json'),
+            'invocation_sha256':sha(args.output/'invocation.json'),'payload_bytes':payload,
+            'full_model_112_StateRoots_native_exp_allocation_GPU_or_performance_qualified':False}
+    write(args.output/'validation.json',result);print(json.dumps(result,sort_keys=True))
+
+if __name__=='__main__':main()
--- /dev/null
+++ tensor_bits_v3.py
@@ -0,0 +1,19 @@
+"""Bounded logical raw-bit transport only; no tensor/provider import on import."""
+import sys
+from support_v3 import require
+
+
+def tensor_bits(value, torch):
+    require(sys.byteorder == 'little', 'pinned little-endian raw-bit transport')
+    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
+            value.layout == torch.strided, 'raw-bit transport requires a CPU strided tensor')
+    require(value.dtype in (torch.bfloat16, torch.float32),
+            'raw-bit transport only original BF16/F32')
+    require(value.numel()*value.element_size() <= 128*1024**2,
+            'bounded raw-bit transport before view/copy')
+    integer = torch.int16 if value.dtype == torch.bfloat16 else torch.int32
+    # Equal element size keeps singleton/nonunit strides legal. NumPy writes
+    # logical C order; no floating conversion or source/storage mutation.
+    raw = value.detach().view(integer).numpy().tobytes(order='C')
+    require(len(raw) == value.numel()*value.element_size(), 'complete logical raw-bit transport')
+    return raw
--- /dev/null
+++ transport_fixtures_v3.py
@@ -0,0 +1,92 @@
+"""ROOT ONLY pinned Torch byte-transport fixtures; no model or source weights."""
+import os
+from pathlib import Path
+import resource
+import sys
+from support_v3 import TORCH_GIT, digest, require, write
+from tensor_bits_v3 import tensor_bits
+
+
+def main():
+    cpus = [int(v) for v in os.environ['ZG_ALLOWED_CPUS'].split(',')]
+    require(len(cpus) == len(set(cpus)) == 4, 'four explicit transport fixture CPUs')
+    os.sched_setaffinity(0, cpus)
+    require(sorted(os.sched_getaffinity(0)) == sorted(cpus), 'fixture actual CPU affinity')
+    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and
+            os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void' and
+            not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
+            'CPU-only original-runtime fixture policy')
+    resource.setrlimit(resource.RLIMIT_AS, (12*1024**3, 12*1024**3))
+    import torch
+    require(sys.byteorder == 'little' and torch.__version__ == '2.13.0+cu129' and
+            torch.version.git_version == TORCH_GIT and not torch.cuda.is_available(),
+            'exact existing original public Torch runtime, no GPU')
+    torch.set_num_threads(4);torch.set_num_interop_threads(1)
+    require(torch.backends.cpu.get_cpu_capability() == 'AVX2', 'pinned CPU AVX2 fixture runtime')
+    words = {
+        'BF16': [0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080,
+                 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0x7fa1],
+        'F32': [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
+                0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
+                0x7fc01234, 0x7fa12345],
+    }
+    records = []
+    for dtype, values in words.items():
+        size = 2 if dtype == 'BF16' else 4
+        integer = torch.int16 if size == 2 else torch.int32
+        floating = torch.bfloat16 if size == 2 else torch.float32
+        expected = b''.join(v.to_bytes(size, 'little') for v in values)
+        def make(raw):
+            storage = torch.frombuffer(bytearray(raw), dtype=integer)
+            return storage, storage.view(floating)
+        for shape in ('normal', 'singleton-last', 'singleton-middle', 'transpose', 'offset', 'stride2'):
+            storage, base = make(expected)
+            raw_expected = expected
+            if shape == 'normal':
+                value = base.reshape(2, 8)
+            elif shape == 'singleton-last':
+                value = base.as_strided((16, 1), (1, 16))
+                require(value.is_contiguous() and value.stride()[-1] == 16,
+                        'actual singleton-contiguous nonunit final stride witness')
+            elif shape == 'singleton-middle':
+                value = base.as_strided((1, 1, 16, 1), (64, 64, 1, 16))
+            elif shape == 'transpose':
+                value = base.reshape(4, 4).transpose(0, 1)
+                raw_expected = b''.join(values[col*4+row].to_bytes(size, 'little')
+                                        for row in range(4) for col in range(4))
+            elif shape == 'offset':
+                storage, base = make(b'\x00'*(2*size)+expected+b'\x00'*(2*size))
+                value = base.as_strided((16, 1), (1, 16), storage_offset=2)
+            else:
+                storage, base = make(b''.join(v.to_bytes(size,'little')+b'\x5a'*size for v in values))
+                value = base.as_strided((16, 1), (2, 32))
+            before = storage.numpy().tobytes(order='C')
+            actual = tensor_bits(value, torch)
+            require(actual == raw_expected, 'every literal raw word retained: '+dtype+'/'+shape)
+            require(storage.numpy().tobytes(order='C') == before,
+                    'original integer storage/canaries unchanged: '+dtype+'/'+shape)
+            records.append({'dtype': dtype, 'case': shape, 'shape': list(value.shape),
+                            'stride': list(value.stride()), 'storage_offset': value.storage_offset(),
+                            'bytes': len(actual), 'words': len(values), 'raw_sha256': digest(actual),
+                            'literal_expected_sha256': digest(raw_expected),
+                            'every_raw_word_exact': True, 'storage_and_canaries_unchanged': True})
+    refusals = []
+    for dtype in (torch.int16, torch.int32, torch.float64):
+        try:
+            tensor_bits(torch.zeros(1, dtype=dtype), torch)
+        except ValueError as error:
+            require('only original BF16/F32' in str(error), 'explicit dtype refusal')
+            refusals.append(str(dtype))
+        else:
+            raise ValueError('unexpected accepted raw-transport dtype')
+    require(len(records) == 12 and len(refusals) == 3, 'all explicit fixture families')
+    write(Path('/output/transport-fixtures.json'), {
+        'status': 'PASS_RAW_BIT_TRANSPORT_ONLY', 'torch': torch.__version__,
+        'torch_git': torch.version.git_version, 'records': records, 'dtype_refusals': refusals,
+        'affinity': sorted(os.sched_getaffinity(0)), 'words': sum(v['words'] for v in records),
+        'no_model_or_math_profile_qualification': True,
+    })
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ run_transport_fixtures_v3.py
@@ -0,0 +1,108 @@
+"""ROOT ONLY finite owned pinned-runtime raw-transport fixtures, no model/source."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import signal
+import subprocess
+import time
+import uuid
+from support_v3 import HERE, IMAGE, IMAGE_DIGEST, TORCH_GIT, frozen, metadata, require, sha, write
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--new-output', type=Path, required=True)
+    args = parser.parse_args()
+    def cancelled(_signal, _frame):
+        raise InterruptedError('owned transport fixture cancelled; retire owned container')
+    signal.signal(signal.SIGTERM, cancelled)
+    freeze = frozen()
+    require(args.new_output.is_absolute() and args.new_output.parent.resolve() == HERE,
+            'exclusive owned fixture directory')
+    args.new_output.mkdir(mode=0o700)
+    cpus = sorted(os.sched_getaffinity(0))[:4]
+    require(len(cpus) == 4, 'four available fixture CPUs')
+    os.sched_setaffinity(0, cpus)
+    require(sha(HERE/'owned_container_v4.py') ==
+            '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
+            'exact reviewed independent owned cleanup')
+    spec = importlib.util.spec_from_file_location('transport_owned', HERE/'owned_container_v4.py')
+    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
+    image = helper.command(helper.PODMAN+['image', 'inspect', IMAGE], 15)
+    require(image.returncode == 0, 'existing pinned fixture image, no pull')
+    actual, = json.loads(image.stdout)
+    require(actual['Id'] == IMAGE.removeprefix('sha256:') and actual['Digest'] == IMAGE_DIGEST,
+            'same original pinned runtime image')
+    nonce = uuid.uuid4().hex;name = 'zg-27gdn-transport-'+nonce;cid = args.new_output/'owned.cid'
+    argv = helper.PODMAN+['run', '--name', name, '--label', helper.LABEL+'='+nonce,
+        '--cidfile', str(cid), '--rm', '--pull=never', '--network=none', '--read-only',
+        '--cap-drop=all', '--security-opt=no-new-privileges', '--memory=12g', '--memory-swap=12g',
+        '--cpus=4', '--pids-limit=256', '--ulimit', 'fsize=134217728:134217728',
+        '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g']
+    env = {'CUDA_VISIBLE_DEVICES':'-1', 'NVIDIA_VISIBLE_DEVICES':'void', 'ATEN_CPU_CAPABILITY':'avx2',
+           'ZG_ALLOWED_CPUS':','.join(map(str,cpus)), 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4',
+           'OPENBLAS_NUM_THREADS':'4', 'OMP_DYNAMIC':'FALSE', 'MKL_DYNAMIC':'FALSE',
+           'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'VLLM_PLUGINS':'',
+           'PYTHONPATH':'/probe', 'PYTHONDONTWRITEBYTECODE':'1', 'LD_PRELOAD':'', 'LD_AUDIT':''}
+    for key,value in env.items():
+        argv += ['-e', key+'='+value]
+    argv += ['-v',str(HERE)+':/probe:ro', '-v',str(args.new_output)+':/output:rw',
+             '--entrypoint','python3',IMAGE,'/probe/transport_fixtures_v3.py']
+    receipt = {'status':'UNQUALIFIED', 'freeze_manifest_sha256':freeze, 'argv':argv,
+               'owned_name':name, 'owned_nonce':nonce, 'expected_cpus':cpus,
+               'no_model_source_or_GPU':True}
+    write(args.new_output/'prepared-invocation.json',receipt)
+    child = log = failure = None;started = time.monotonic()
+    try:
+        child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
+                                 stderr=subprocess.STDOUT,start_new_session=True)
+        receipt['owned_cli_pid'] = child.pid
+        log = helper.Log(child.stdout,args.new_output/'execution.log')
+        receipt['exit_code'] = log.observe(child,120)
+    except BaseException as error:
+        failure = error
+    finally:
+        receipt['cleanup_errors'] = helper.cleanup(child,name,nonce,cid,receipt) if child is not None else []
+        if log is not None:
+            try:
+                log.finish()
+            except BaseException as error:
+                receipt['cleanup_errors'].append('log cleanup: '+repr(error))
+            finally:
+                receipt['log_cleanup'] = {'pipe_eof':log.eof, 'all_closed':log.closed,
+                                         'total_bytes':log.total, 'retained_bytes':log.retained}
+        elif child is not None and child.stdout is not None:
+            try:
+                child.stdout.close()
+            except BaseException as error:
+                receipt['cleanup_errors'].append('uninitialized pipe close: '+repr(error))
+        receipt['elapsed_seconds_not_benchmark'] = time.monotonic()-started
+    try:
+        require(failure is None and receipt.get('exit_code') == 0 and receipt['cleanup_errors'] == [],
+                'owned actual-runtime raw-transport fixture/cleanup failed')
+        result = metadata(args.new_output/'transport-fixtures.json')
+        require(result['status'] == 'PASS_RAW_BIT_TRANSPORT_ONLY' and
+                result['torch_git'] == TORCH_GIT and result['torch'] == '2.13.0+cu129' and
+                result['affinity'] == cpus and result['words'] == 192 and
+                len(result['records']) == 12 and len(result['dtype_refusals']) == 3 and
+                all(v['every_raw_word_exact'] and v['storage_and_canaries_unchanged'] and
+                    v['raw_sha256'] == v['literal_expected_sha256'] for v in result['records']),
+                'all genuine pinned-Torch raw transport checks')
+        require(frozen() == freeze, 'all original/V2/V3 source unchanged after fixture')
+        receipt['fixture_report_sha256'] = sha(args.new_output/'transport-fixtures.json')
+        receipt['status'] = 'PASS_OWNED_RAW_BIT_TRANSPORT_ONLY'
+    except BaseException as error:
+        if failure is None:
+            failure = error
+    if failure is not None:
+        receipt['failure'] = repr(failure)
+        receipt['status'] = 'FAIL_UNQUALIFIED'
+    write(args.new_output/'invocation.json',receipt)
+    require(receipt['status'] == 'PASS_OWNED_RAW_BIT_TRANSPORT_ONLY',
+            'raw transport fixture unqualified; retain attempt')
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ FOLLOWUP_V3.md
@@ -0,0 +1,50 @@
+# Singleton-stride raw-bit transport
+
+Author: spine_review. Additive source-only V3; original V1/V2 remain immutable.
+
+ROOT V2 attempt `root-reference-v2-1` reached actual original operators but
+failed after saving the first observed archive: BF16 byte reinterpretation
+requires last stride1 when the element size changes. A singleton dimension
+can be logically contiguous while keeping last stride10240. That attempt and
+all original sources/archives stay preserved; no complete reference passed.
+
+`tensor_bits_v3.tensor_bits` accepts only bounded original CPU strided BF16/F32.
+It reinterprets BF16 as int16 and F32 as int32 (same element size), then NumPy
+`tobytes(order='C')` serializes logical row order. There is no numerical cast,
+float arithmetic, storage mutation or replacement operator. Explicit little
+endianness/dtype/device/layout/128MiB bounds precede views and copies. Signed
+zero, subnormal, Inf and NaN payload words must remain unchanged.
+
+`capture_v3.py` replaces only the local bits helper/import. Reader V2 guard,
+original operators/weights/dtypes/hooks/call topology/safetensors writes and all
+expected/source/state checks stay unchanged. No other live original observer
+contains the byte-view hazard. `run_reference_v3.py` selects that entrypoint;
+`validate_reference_v3.py` only changes support import. `support_v3.frozen`
+validates original V1, exact V2, and every additive V3 file before/after use.
+
+ROOT-only `transport_fixtures_v3.py` uses the exact existing public Torch/runtime
+without model/source weights: two dtypes × six layouts (normal, singleton-last,
+singleton-middle, transpose, nonzero offset, stride2),192 literal raw words,
+three unsupported dtype refusals, all underlying integer storage/canaries exact.
+Fixture literals include signed zero, +/-subnormal, finite extrema, +/-Inf and
+quiet/signaling NaN payloads. Expected bytes are literal integer words and
+explicit index permutations, never candidate-generated floating results.
+
+`run_transport_fixtures_v3.py` uses byte-identical owned-v4 lifecycle from the
+original freeze: no network/GPU/model mount, read-only probe/image, fresh output,
+12GiB memory+swap12GiB, quota4/explicit four-CPU affinity,12GiB AS before Torch,
+120s process deadline, complete bounded logs/nonce-CID/PGID/reap/EOF cleanup.
+This is a transport-only control, not model math/profile admission.
+
+ROOT commands from this directory, after source review and quiet-run scheduling:
+
+```bash
+python3 run_transport_fixtures_v3.py --new-output /ABSOLUTE/RUN/root-transport-v3-1
+python3 run_reference_v3.py --new-output /ABSOLUTE/RUN/root-reference-v3-1
+python3 validate_reference_v3.py --output /ABSOLUTE/RUN/root-reference-v3-1
+```
+
+Author checks only seven Python ASTs, narrow diff and original V1/V2 frozen-byte
+checks. No fixture/Torch/operator/model/container/provider/build/GPU execution;
+no source-weight or DSO payload hashes. ROOT executes all new controls and
+full reference; H48/112-StateRoot/full-model/native-exp eligibility stays closed.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/HEADER_BOUND_V2.patch

Original UTF-8 bytes: 65190; SHA256 `a0ff96fe6286d120b1f86ddd81b115d43b3772cac312239e2a0a6e7d4f39bf99`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ capture_v2.py
@@ -0,0 +1,314 @@
+"""ROOT ONLY original27B layer0 GDN module; never a full-model constructor."""
+import gc
+import importlib.metadata
+import inspect
+import json
+import mmap
+import os
+from pathlib import Path
+import resource
+import struct
+import sys
+import time
+from support_v2 import HERE, MODEL_SHA, TORCH_GIT, bounds, digest, fd_sha, fingerprint, metadata, read, require, sha, source_header, unique, write
+
+SOURCE = Path('/source')
+OUTPUT = Path('/output/captured')
+EMBED = 'model.language_model.embed_tokens.weight'
+INPUT_NORM = 'model.language_model.layers.0.input_layernorm.weight'
+PREFIX = 'model.language_model.layers.0.linear_attn.'
+SOURCE_OWNER = None # Original FD/map/storage owner retained until owned process exit.
+
+class SourceOwner:
+    def __init__(self, plan):
+        self.plan, self.fds, self.headers, self.before = plan, {}, {}, {}
+        self.maps, self.tensors, self.readsets, self.versions = {}, {}, {}, {}
+        for name,pin in plan['metadata'].items():
+            read(SOURCE/name,pin=pin)
+        self.index=metadata(SOURCE/'model.safetensors.index.json')
+        for shard,pin in plan['selected_shards'].items():
+            fd=os.open(SOURCE/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            self.fds[shard]=fd
+            self.before[shard]=fingerprint(os.fstat(fd))
+            require(os.fstat(fd).st_size==pin['file']['bytes'] and
+                    fd_sha(fd,0,pin['file']['bytes'])==pin['file']['sha256'],
+                    'complete same opened original selected shard before library initialization')
+            n,raw=source_header(fd,pin)
+            require(fingerprint(os.fstat(fd))==self.before[shard],
+                    'same selected FD unchanged across full hash and bounded header read')
+            self.headers[shard]=(8+n,json.loads(raw,object_pairs_hook=unique))
+
+    def load_views(self,torch):
+        for name,item in self.plan['tensors'].items():
+            shard=item['shard'];begin,header=self.headers[shard];entry=header[name]
+            require(self.index['weight_map'][name]==shard and entry['dtype']=='BF16' and
+                    entry['shape']==item['shape'] and begin+entry['data_offsets'][0]==item['absolute_offset'] and
+                    entry['data_offsets'][1]-entry['data_offsets'][0]==item['bytes'],
+                    'actual exact source role/name/geometry/full-byte extent')
+            # COW protects source files; every window is only its exact parameter
+            # plus the leading alignment bytes, not a whole-shard mapping.
+            mapping=mmap.mmap(self.fds[shard],item['mapping_bytes'],access=mmap.ACCESS_COPY,
+                              offset=item['page_aligned_offset'])
+            self.maps[name]=mapping
+            value=torch.frombuffer(mapping,dtype=torch.bfloat16,count=item['bytes']//2,
+                                   offset=item['view_offset']).reshape(item['shape'])
+            require(value.is_contiguous() and value.device.type=='cpu' and not value.is_meta,
+                    'authentic contiguous BF16 source view')
+            self.tensors[name]=value
+            self.versions[name]=value._version
+            if name==EMBED:
+                self.readsets[name]={str(token):fd_sha(self.fds[shard],item['absolute_offset']+token*10240,10240)
+                                     for token in (11,353)}
+            else:
+                self.readsets[name]=fd_sha(self.fds[shard],item['absolute_offset'],item['bytes'])
+
+    def mapped_sha(self,name,begin,count):
+        value=__import__('hashlib').sha256();view=memoryview(self.maps[name])
+        try:
+            for offset in range(0,count,1024*1024):
+                value.update(view[begin+offset:begin+min(count,offset+1024*1024)])
+        finally:view.release()
+        return value.hexdigest()
+
+    def finish(self):
+        for name,item in self.plan['tensors'].items():
+            require(self.tensors[name]._version==self.versions[name],'original source tensor version unchanged')
+            if name==EMBED:
+                actual={str(t):self.mapped_sha(name,item['view_offset']+t*10240,10240) for t in (11,353)}
+            else:actual=self.mapped_sha(name,item['view_offset'],item['bytes'])
+            require(actual==self.readsets[name],'original mapped learned payload unchanged: '+name)
+        after={}
+        for shard,fd in self.fds.items():
+            pin=self.plan['selected_shards'][shard]['file']
+            after[shard]=fingerprint(os.fstat(fd))
+            require(after[shard]==self.before[shard] and fd_sha(fd,0,pin['bytes'])==pin['sha256'],
+                    'same selected full source FD unchanged after original operators')
+        return after
+
+def main():
+    global SOURCE_OWNER
+    started=time.monotonic()
+    require(not OUTPUT.exists(),'exclusive output directory');OUTPUT.mkdir(mode=0o700)
+    report={'schema':'zerogravity.original27B.layer0-gdn.public-reference.v1','status':'UNQUALIFIED',
+            'steps':[],'tokens':[11,353],'parameter_profile':'original27B-raw-source-bf16-v1',
+            'full_model_StateRoot_native_exp_GPU_eligible':False}
+    try:
+        require(sys.byteorder=='little','original little-endian source')
+        cpus=[int(v) for v in os.environ['ZG_ALLOWED_CPUS'].split(',')]
+        require(len(cpus)==len(set(cpus))==4,'four explicit CPUs');os.sched_setaffinity(0,cpus)
+        require(sorted(os.sched_getaffinity(0))==sorted(cpus),'actual affinity before importing Torch')
+        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void' and
+                os.environ.get('ATEN_CPU_CAPABILITY')=='avx2' and os.environ.get('USE_HUB_KERNELS')=='0' and
+                not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
+                'declared original CPU reference/no external math interposition')
+        require(sha(HERE/'source-plan.json')==os.environ['ZG_SOURCE_PLAN_SHA'],'exact source metadata plan')
+        plan=metadata(HERE/'source-plan.json')
+        derived=bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+        report.update(source_plan_sha256=sha(HERE/'source-plan.json'),source_repo=plan['repository'],
+                      source_revision=plan['revision'],snapshot=plan['snapshot'],pre_library_bounds=derived,
+                      native_ResourceArbiter_admission_qualified=False)
+        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
+        SOURCE_OWNER=SourceOwner(plan)
+        # No library/model initialization or parameter view precedes the bounds.
+        import torch
+        from torch.nn import functional as F
+        from safetensors.torch import save_file
+        from transformers import Qwen3_5Config, DynamicCache
+        from transformers.models.qwen3_5 import modeling_qwen3_5 as original
+        from observer import Observation, ScopedDispatch, info, total_bytes
+        require(torch.__version__=='2.13.0+cu129' and torch.version.git_version==TORCH_GIT and
+                importlib.metadata.version('transformers')=='5.17.0' and
+                importlib.metadata.version('safetensors')=='0.8.0' and not torch.cuda.is_available(),
+                'exact existing pinned CPU runtime')
+        torch.set_num_threads(4);torch.set_num_interop_threads(1)
+        require(torch.backends.cpu.get_cpu_capability()=='AVX2' and torch.backends.mkldnn.enabled,
+                'original CPU AVX2 policy')
+        require(sha(Path(original.__file__))==MODEL_SHA,'exact original declared architecture source')
+        actual_modules={'modeling_qwen3_5.py':original,
+                        'configuration_qwen3_5.py':sys.modules[Qwen3_5Config.__module__],
+                        'cache_utils.py':sys.modules[DynamicCache.__module__],
+                        'hub_kernels.py':sys.modules['transformers.integrations.hub_kernels']}
+        for name,module in actual_modules.items():
+            require(sha(Path(module.__file__))==plan['runtime_source'][name]['sha256'],
+                    'pinned original module '+name)
+        for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule',
+                     'causal_conv1d_fn','causal_conv1d_update'):
+            function=getattr(original,name);found=False
+            while function is not None:
+                closure=inspect.getclosurevars(function).nonlocals
+                if 'is_new_implementation' in closure:
+                    require(closure['is_new_implementation'] is False and
+                            Path(closure['implementation'].__code__.co_filename).resolve()==Path(original.__file__).resolve(),
+                            'refuse optional non-original package implementation: '+name)
+                    found=True
+                function=getattr(function,'__wrapped__',None)
+            require(found,'authentic original public function wrapper: '+name)
+        config=metadata(SOURCE/'config.json')
+        require(config['architectures']==['Qwen3_5ForConditionalGeneration'] and config['model_type']=='qwen3_5' and
+                config['text_config']['model_type']=='qwen3_5_text','actual27B declared original architecture')
+        cfg=Qwen3_5Config.from_dict(config).text_config
+        require((cfg.hidden_size,cfg.linear_num_key_heads,cfg.linear_num_value_heads,cfg.linear_key_head_dim,
+                 cfg.linear_value_head_dim,cfg.linear_conv_kernel_dim)==(5120,16,48,128,128,4),
+                'original27B GDN exact geometry')
+        SOURCE_OWNER.load_views(torch)
+        with torch.device('meta'):
+            gdn=original.Qwen3_5GatedDeltaNet(cfg,layer_idx=0)
+            input_norm=original.Qwen3_5RMSNorm(5120,eps=cfg.rms_norm_eps)
+        parameters={name.removeprefix(PREFIX):value for name,value in SOURCE_OWNER.tensors.items() if name.startswith(PREFIX)}
+        require(set(parameters)==set(dict(gdn.named_parameters())) and len(parameters)==9 and
+                not dict(gdn.named_buffers()),'every original GDN parameter supplied, no random/unfilled fallback')
+        loaded=gdn.load_state_dict(parameters,strict=True,assign=True)
+        require(not loaded.missing_keys and not loaded.unexpected_keys,'strict complete source parameter assignment')
+        input_norm.load_state_dict({'weight':SOURCE_OWNER.tensors[INPUT_NORM]},strict=True,assign=True)
+        gdn.eval();input_norm.eval()
+        parameter_audit={}
+        for name,value in gdn.named_parameters():
+            raw=SOURCE_OWNER.tensors[PREFIX+name]
+            require(value.dtype==torch.bfloat16 and not value.is_meta and value.data_ptr()==raw.data_ptr() and
+                    tuple(value.shape)==tuple(raw.shape),'exact raw BF16 original parameter, no cast/copy: '+name)
+            parameter_audit[name]={**info(value),'pointer':value.data_ptr(),'version':value._version,
+                                   'source_readset_sha256':SOURCE_OWNER.readsets[PREFIX+name]}
+        require(input_norm.weight.data_ptr()==SOURCE_OWNER.tensors[INPUT_NORM].data_ptr() and
+                input_norm.weight.dtype==torch.bfloat16 and gdn.norm.weight.dtype==torch.bfloat16 and
+                gdn.A_log.dtype==gdn.dt_bias.dtype==torch.bfloat16,
+                'original BF16 gamma/A_log/dt_bias/inputnorm stay untouched')
+        baseline={};observations=[];hooks=[];active=[None]
+        def bits(value):return value.detach().contiguous().view(torch.uint8).numpy().tobytes()
+        def same(a,b,message):
+            require(a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and bits(a)==bits(b),message)
+        def projection_hook(name):
+            def hook(module,args,result):
+                if active[0] is not None:
+                    active[0].tensor('projection.'+name+'.input',args[0])
+                    active[0].tensor('projection.'+name+'.output',result)
+            return hook
+        def kept(name,value):
+            count=value.numel()*value.element_size()
+            require(total_bytes[0]+count<=128*1024**2,'bounded combined plain/observed tensor payload')
+            baseline[name]=value.detach().contiguous().clone();total_bytes[0]+=count
+        try:
+            kept('source.A_log',gdn.A_log)
+            kept('source.dt_bias',gdn.dt_bias)
+            kept('source.norm.weight',gdn.norm.weight)
+            with torch.inference_mode():
+                for mode in ('plain','observed'):
+                    cache=DynamicCache(config=cfg)
+                    require(len(cache.layers)==64 and not cache.has_previous_state(0,0),'real original cache starts empty')
+                    if mode=='observed':
+                        for name in ('in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'):
+                            hooks.append(gdn.get_submodule(name).register_forward_hook(projection_hook(name)))
+                    for step,token in enumerate((11,353)):
+                        embedding=F.embedding(torch.tensor([[token]],dtype=torch.int64),SOURCE_OWNER.tensors[EMBED])
+                        normalized=input_norm(embedding)
+                        require(embedding.shape==normalized.shape==(1,1,5120),'actual trained original embedding/InputNorm')
+                        require(cache.has_previous_state(0,0) is (step==1),'real initial then cached source dispatch')
+                        if mode=='plain':
+                            output=gdn(normalized,cache_params=cache)
+                            for name,value in [('embedding',embedding),('input_norm',normalized),('output',output),
+                                               ('conv',cache.layers[0].conv_states[0]),('state',cache.layers[0].recurrent_states[0])]:
+                                kept(f'step-{step}.{name}',value)
+                        else:
+                            obs=Observation(step,gdn);observations.append(obs);active[0]=obs
+                            obs.tensor('prefix.embedding',embedding);obs.tensor('prefix.input_norm',normalized)
+                            require(sys.gettrace() is None,'do not replace an existing tracer')
+                            previous=sys.gettrace();mode_observer=ScopedDispatch(obs);obs.mode=mode_observer
+                            try:
+                                sys.settrace(obs.trace)
+                                with mode_observer:
+                                    output=gdn(normalized,cache_params=cache)
+                            finally:
+                                sys.settrace(previous);obs.mode=None;active[0]=None
+                            obs.tensor('cache.after.conv',cache.layers[0].conv_states[0])
+                            obs.tensor('cache.after.state',cache.layers[0].recurrent_states[0])
+                            obs.tensor('module.output',output)
+                            directory=OUTPUT/f'step-{step}';directory.mkdir()
+                            archive=directory/'actual-gdn.safetensors'
+                            save_file(obs.tensors,str(archive),metadata={'scope':'actual original27B layer0 frames; prevalidation only'})
+                            for name,value in obs.tensors.items():obs.meta[name]['raw_sha256']=digest(bits(value))
+                            record={'step':step,'teacher_token':token,'actual_calls':dict(obs.calls),'events':obs.events,
+                                    'dispatch':obs.dispatch,'tensors':obs.meta,'payload_bytes':obs.bytes,
+                                    'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
+                                    'plain_observed_all_outputs_and_state_exact':False}
+                            write(directory/'prevalidation.json',record)
+                            phase='chunk' if step==0 else 'recurrent'
+                            expected={'Qwen3_5GatedDeltaNet.forward':1,'Qwen3_5RMSNormGated.forward':1,
+                                      'l2norm':2,'torch_'+phase+'_gated_delta_rule':1,
+                                      'causal_conv1d_fn' if step==0 else 'causal_conv1d_update':1}
+                            require(dict(obs.calls)==expected,'actual unchanged original initial/cached call schedule')
+                            for name,value in [('embedding',embedding),('input_norm',normalized),('output',output),
+                                               ('conv',cache.layers[0].conv_states[0]),('state',cache.layers[0].recurrent_states[0])]:
+                                same(baseline[f'step-{step}.{name}'],value,'plain/observed complete actual '+name)
+                            same(obs.tensors[phase+'.return.actual_return_1'],cache.layers[0].recurrent_states[0],
+                                 'original public returned state equals real original cache')
+                            if step:
+                                same(observations[0].tensors['chunk.return.actual_return_1'],
+                                     obs.tensors['recurrent.input.initial_state'],'actual initial return supplies real cached prior')
+                            record['plain_observed_all_outputs_and_state_exact']=True
+                            report['steps'].append(record)
+                            total_bytes[0]+=obs.bytes
+                        for layer in cache.layers[1:]:
+                            require(not getattr(layer,'is_conv_states_initialized',[]) or
+                                    not any(layer.is_conv_states_initialized),'no unrelated layer state allocated')
+        finally:
+            active[0]=None
+            for hook in hooks:hook.remove()
+            require(sys.gettrace() is None,'original tracer restored')
+        save_file(baseline,str(OUTPUT/'plain.safetensors'),metadata={'scope':'genuine unobserved two-teacher module outputs/cache'})
+        prior=metadata(Path('/prefix-reference/reference.json'),pin=plan['prior_public_prefix'])
+        require(prior['source_repo']==plan['repository'] and prior['source_revision']==plan['revision'] and
+                prior['snapshot']==plan['snapshot'],'independent original27B prefix reference source binding')
+        prefix_checks=[]
+        for step,token in enumerate((11,353)):
+            old=prior['steps'][step]
+            require(old['token']==token,'independent prefix teacher binding')
+            for stage,actual in [('embedding',baseline[f'step-{step}.embedding']),
+                                 ('normalized',baseline[f'step-{step}.input_norm']),
+                                 ('projected',observations[step].tensors['projection.in_proj_qkv.output'])]:
+                row=old[stage];expected=read(Path('/prefix-reference')/row['file'],pin=row)
+                widened=b''.join((word<<16).to_bytes(4,'little') for word, in struct.iter_unpack('<H',bits(actual)))
+                require(actual.dtype==torch.bfloat16 and len(widened)==row['bytes'] and widened==expected,
+                        'all prior genuine original27B prefix words: '+stage)
+                prefix_checks.append({'step':step,'stage':stage,'words':actual.numel(),'bit_exact':True,'prior':row})
+        for name,value in gdn.named_parameters():
+            require(value.data_ptr()==parameter_audit[name]['pointer'] and value._version==parameter_audit[name]['version'],
+                    'original source parameter storage/version unchanged')
+        after=SOURCE_OWNER.finish()
+        maps=Path('/proc/self/maps').read_text()
+        require(len(maps.encode())<=2*1024**2 and '/nix/store/' not in maps,'bounded actual container mapping closure')
+        objects=sorted({line.split(None,5)[5] for line in maps.splitlines() if len(line.split(None,5))==6 and
+                        line.split(None,5)[5].startswith('/') and '.so' in Path(line.split(None,5)[5]).name})
+        require(len(objects)<=256,'bounded actual loaded-library count')
+        dsos={};total_dso=0
+        for path in objects:
+            size=Path(path).stat().st_size;total_dso+=size
+            require(size<=2*1024**3 and total_dso<=16*1024**3 and ' (deleted)' not in path,'bounded actual mapped provider closure')
+            dsos[path]={'bytes':size,'sha256':sha(path)}
+        report.update(status='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION',
+            source_FDs_before=SOURCE_OWNER.before,source_FDs_after=after,source_readsets=SOURCE_OWNER.readsets,
+            all_selected_FDs_and_mapped_parameters_unchanged=True,all_source_parameters_BF16=True,
+            parameter_audit=parameter_audit,source_owner_retained_until_owned_process_exit=True,
+            norm_profile={'class':'Qwen3_5RMSNormGated','heads':48,'head_width':128,
+                          'raw_gamma_dtype':'torch.bfloat16','weighted_dtype':'torch.bfloat16',
+                          'gate_product_dtype':'torch.float32','return_dtype':'torch.bfloat16',
+                          'epsilon':gdn.norm.variance_epsilon,'activation':gdn.norm.activation},
+            payload_bytes=total_bytes[0],
+            plain_archive={'bytes':(OUTPUT/'plain.safetensors').stat().st_size,'sha256':sha(OUTPUT/'plain.safetensors')},
+            hooks_removed=len(hooks),runtime={'torch':torch.__version__,'torch_git':torch.version.git_version,
+                'transformers':importlib.metadata.version('transformers'),'cpu_capability':torch.backends.cpu.get_cpu_capability(),
+                'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
+                'affinity':sorted(os.sched_getaffinity(0)),'cuda_available':False,'modeling_sha256':MODEL_SHA,
+                'mapped_shared_objects':dsos,'proc_status':Path('/proc/self/status').read_text(),
+                'proc_cgroup':Path('/proc/self/cgroup').read_text(),'proc_mountinfo':Path('/proc/self/mountinfo').read_text()},
+            scope='Original27B private layer0 embedding/InputNorm/GDN two-teacher public reference only; all raw source BF16. No full-model logits/112 StateRoots, native exp/provider allocation eligibility, production memory admission, GPU or performance.')
+        report['prior_prefix_checks']=prefix_checks
+        require(report['payload_bytes']<=128*1024**2,'bounded total actual plain/observed payload')
+    except BaseException as error:
+        report['failure']=repr(error)
+        raise
+    finally:
+        report['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
+        write(OUTPUT/'manifest.json',report)
+    print('ORIGINAL27B_LAYER0_GDN_CAPTURE_ONLY',flush=True)
+
+if __name__=='__main__':main()
--- /dev/null
+++ run_reference_v2.py
@@ -0,0 +1,134 @@
+"""ROOT ONLY owned CPU reference, whole source before/after, no model on import."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import signal
+import stat
+import subprocess
+import time
+import uuid
+from support_v2 import HERE, IMAGE, IMAGE_DIGEST, bounds, fd_sha, fingerprint, frozen, metadata, read, require, sha, write
+
+def source_open(plan):
+    """ROOT-only complete source hashes, bounded chunks; same FDs kept open."""
+    handles, rows = {}, []
+    try:
+        for row in plan['source_files']:
+            require(Path(row['name']).name==row['name'],'contained full source filename')
+            fd=os.open(Path(plan['source_directory'])/row['name'],os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            handles[row['name']]=fd
+            before=os.fstat(fd)
+            require(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'] and
+                    fd_sha(fd,0,row['bytes'])==row['sha256'] and fingerprint(before)==fingerprint(os.fstat(fd)),
+                    'actual full original source bytes/FD immutable before reference')
+            rows.append({'name':row['name'],'bytes':row['bytes'],'sha256':row['sha256'],
+                         'fd_identity':fingerprint(before)})
+        require(sum(row['bytes'] for row in plan['source_files'] if row['name'].endswith('.safetensors'))
+                ==plan['weight_bytes']==55563006776,'all18 authentic original weight shards')
+        return handles,rows
+    except BaseException:
+        for fd in handles.values():os.close(fd)
+        raise
+
+def source_after(handles,rows):
+    after=[]
+    for row in rows:
+        fd=handles[row['name']]
+        require(fingerprint(os.fstat(fd))==row['fd_identity'] and
+                fd_sha(fd,0,row['bytes'])==row['sha256'],'same full original source FD after reference')
+        after.append(dict(row))
+    return after
+
+def tree_size(root):
+    total=0
+    for path in root.rglob('*'):
+        require(not path.is_symlink(),'no output artifact symlink')
+        if path.is_file():
+            require(stat.S_ISREG(path.stat().st_mode),'regular actual artifact')
+            total+=path.stat().st_size
+    require(total<=256*1024**2-128*1024,'bounded whole attempt with reserved completion metadata')
+    return total
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--new-output',type=Path,required=True)
+    args=parser.parse_args()
+    def deadline(_signal,_frame):raise TimeoutError('finite whole source/reference/lifecycle deadline')
+    signal.signal(signal.SIGALRM,deadline);signal.alarm(2400)
+    def cancelled(_signal,_frame):raise InterruptedError('owned reference cancelled; retire only owned container/process')
+    signal.signal(signal.SIGTERM,cancelled)
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+    require(args.new_output.is_absolute() and args.new_output.parent.resolve()==HERE,
+            'exclusive owned attempt directory');args.new_output.mkdir(mode=0o700)
+    cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus)
+    require(sha(HERE/'owned_container_v4.py')=='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
+            'byte-identical reviewed owned container lifecycle')
+    spec=importlib.util.spec_from_file_location('gdn27_owned',HERE/'owned_container_v4.py')
+    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
+    nonce=uuid.uuid4().hex;owned_name='zg-original27-gdn-'+nonce;cid=args.new_output/'owned.cid'
+    receipt={'status':'UNQUALIFIED','freeze_manifest_sha256':freeze,'source_plan_sha256':sha(HERE/'source-plan.json'),
+             'expected_cpus':cpus,'owned_name':owned_name,'owned_nonce':nonce,'image_id':IMAGE,'image_digest':IMAGE_DIGEST,
+             'no_GPU_or_full_model_or_native_provider_admission':True}
+    handles={};child=log=None;failure=None;started=time.monotonic()
+    try:
+        handles,before=source_open(plan);receipt['source_before']=before
+        inspected=helper.command(helper.PODMAN+['image','inspect',IMAGE],15)
+        require(inspected.returncode==0,'existing approved image; no pull')
+        (args.new_output/'image-inspect.json').write_bytes(inspected.stdout)
+        image,=json.loads(inspected.stdout)
+        require(image['Id']==IMAGE.removeprefix('sha256:') and image['Digest']==IMAGE_DIGEST,'pinned actual image')
+        prefix=Path(plan['prior_public_prefix']['path']).parent
+        env={'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'void','ATEN_CPU_CAPABILITY':'avx2',
+             'ZG_ALLOWED_CPUS':','.join(map(str,cpus)),'ZG_SOURCE_PLAN_SHA':sha(HERE/'source-plan.json'),
+             'USE_HUB_KERNELS':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
+             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE',
+             'PYTHONPATH':'/probe','PYTHONDONTWRITEBYTECODE':'1','LD_PRELOAD':'','LD_AUDIT':'','VLLM_PLUGINS':''}
+        argv=helper.PODMAN+['run','--name',owned_name,'--label',helper.LABEL+'='+nonce,'--cidfile',str(cid),
+            '--rm','--pull=never','--network=none','--read-only','--cap-drop=all','--security-opt=no-new-privileges',
+            '--memory=12g','--memory-swap=12g','--cpus=4','--pids-limit=256',
+            '--ulimit','fsize=134217728:134217728','--tmpfs','/tmp:rw,nosuid,nodev,size=1g']
+        for key,value in env.items():argv+=['-e',key+'='+value]
+        argv+=['-v',str(HERE)+':/probe:ro','-v',plan['source_directory']+':/source:ro',
+               '-v',str(prefix)+':/prefix-reference:ro','-v',str(args.new_output)+':/output:rw',
+               '--entrypoint','python3',IMAGE,'/probe/capture_v2.py']
+        receipt['argv']=argv;write(args.new_output/'prepared-invocation.json',receipt)
+        try:
+            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
+            receipt['owned_cli_pid']=child.pid
+            log=helper.Log(child.stdout,args.new_output/'execution.log')
+            receipt['exit_code']=log.observe(child,600)
+        except BaseException as error:failure=error
+        finally:
+            receipt['cleanup_errors']=helper.cleanup(child,owned_name,nonce,cid,receipt) if child is not None else []
+            if log is not None:
+                try:log.finish()
+                except BaseException as error:receipt['cleanup_errors'].append('log cleanup: '+repr(error))
+                finally:receipt['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
+            elif child is not None and child.stdout is not None:
+                try:child.stdout.close()
+                except BaseException as error:receipt['cleanup_errors'].append('uninitialized pipe close: '+repr(error))
+        require(failure is None and receipt.get('exit_code')==0 and receipt['cleanup_errors']==[],
+                'owned original public reference/cleanup failed; preserve this attempt')
+        receipt['source_after']=source_after(handles,before)
+        require(receipt['source_after']==before and frozen()==freeze,'all source/preparation files unchanged')
+        receipt['precompletion_artifact_bytes']=tree_size(args.new_output)
+        receipt['status']='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        if failure is None:failure=error
+    finally:
+        failures=[]
+        for fd in handles.values():
+            try:os.close(fd)
+            except BaseException as error:failures.append(repr(error))
+        receipt['source_FD_cleanup_errors']=failures
+        receipt['source_FDs_closed']=not failures
+        receipt['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        if failure is not None:receipt['failure']=repr(failure)
+        if failure is not None or failures:receipt['status']='FAIL_UNQUALIFIED'
+        write(args.new_output/'invocation.json',receipt)
+    require(receipt['status']=='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION',
+            'reference attempt unqualified; retained original evidence')
+
+if __name__=='__main__':main()
--- /dev/null
+++ support_v2.py
@@ -0,0 +1,137 @@
+"""Bounded metadata/source transport; no Torch/provider/model import."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import struct
+
+HERE = Path(__file__).resolve().parent
+IMAGE = 'sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe'
+IMAGE_DIGEST = 'sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634'
+MODEL_SHA = '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'
+TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+def digest(raw):
+    return hashlib.sha256(raw).hexdigest()
+
+def sha(path):
+    with Path(path).open('rb') as file:
+        return hashlib.file_digest(file, 'sha256').hexdigest()
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON key')
+        result[key] = value
+    return result
+
+def read(path, cap=4*1024*1024, pin=None):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as file:
+        before = os.fstat(file.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
+        raw = file.read(cap+1)
+        after = os.fstat(file.fileno())
+    require(len(raw) <= cap and fingerprint(before) == fingerprint(after), 'immutable metadata FD')
+    if pin is not None:
+        require(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'exact metadata pin')
+    return raw
+
+def metadata(path, cap=4*1024*1024, pin=None):
+    return json.loads(read(path, cap, pin), object_pairs_hook=unique)
+
+def write(path, value):
+    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
+    require(len(raw) <= 4*1024*1024, 'bounded output metadata')
+    with Path(path).open('xb') as file:
+        os.fchmod(file.fileno(),0o600)
+        file.write(raw)
+
+def fingerprint(meta):
+    return [meta.st_dev,meta.st_ino,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns]
+
+def fd_sha(fd, begin, count):
+    value = hashlib.sha256()
+    for offset in range(0,count,1024*1024):
+        raw = os.pread(fd,min(1024*1024,count-offset),begin+offset)
+        require(len(raw) == min(1024*1024,count-offset), 'short source FD read')
+        value.update(raw)
+    return value.hexdigest()
+
+def source_header(fd, pin):
+    """Validate bounded original header length before the body pread."""
+    before = os.fstat(fd)
+    require(stat.S_ISREG(before.st_mode) and before.st_size == pin['file']['bytes'],
+            'exact regular selected source FD')
+    prefix = os.pread(fd, 8, 0)
+    require(len(prefix) == 8, 'complete source header length word')
+    n, = struct.unpack('<Q', prefix)
+    expected = pin['header_bytes']
+    require(type(expected) is int and 1 < expected <= 8*1024*1024 and
+            n == expected and 8+n <= before.st_size,
+            'bounded frozen source header length before body read')
+    raw = os.pread(fd, n, 8)
+    require(len(raw) == n and digest(raw) == pin['header_sha256'] and
+            fingerprint(before) == fingerprint(os.fstat(fd)),
+            'complete exact immutable original source header')
+    return n, raw
+
+def bounds(plan, page):
+    """Integer geometry before importing libraries or allocating parameter maps."""
+    limits = plan['bounds_before_library_or_views']
+    require(page == limits['page_size'] == 4096, 'pinned page geometry')
+    require(len(plan['tensors']) == 11 and len(plan['selected_shards']) == 2, 'exact complete selected roles')
+    mapped, params = 0, 0
+    for name, item in plan['tensors'].items():
+        require(item['dtype'] == 'BF16' and type(item['shape']) is list, 'original BF16 source policy')
+        count = 1
+        for axis in item['shape']:
+            require(type(axis) is int and 0 < axis <= 248320, 'bounded original tensor axis')
+            count *= axis
+        start, length = item['absolute_offset'], item['bytes']
+        require(type(start) is type(length) is int and start >= 8 and length == count*2,
+                'full original typed extent')
+        aligned = start-start%page
+        require(item['page_aligned_offset'] == aligned and item['view_offset'] == start-aligned and
+                item['mapping_bytes'] == length+start-aligned and
+                start+length <= plan['selected_shards'][item['shard']]['file']['bytes'],
+                'exact bounded parameter window; no whole shard mapping')
+        mapped += item['mapping_bytes']
+        if '.linear_attn.' in name:
+            params += length
+    touched = params + 10240 + 20480
+    require(params == limits['selected_parameter_bytes'] == 231752128 and
+            mapped == limits['aggregate_window_mapping_bytes'] == 2774582792 and
+            touched == limits['touched_source_payload_bytes'] == 231782848 and
+            mapped <= limits['mapped_VA_cap_bytes'] == 3*1024**3 and
+            touched <= limits['derived_touched_source_cap_bytes'] == 256*1024**2,
+            'pre-library exact source windows and selected resident payload bound')
+    virtual_envelope=mapped+limits['runtime_VA_allowance_bytes']+limits['conservative_operator_working_envelope_bytes']+limits['capture_payload_cap_bytes']
+    physical_estimate=touched+limits['runtime_resident_allowance_bytes']+limits['conservative_operator_working_envelope_bytes']+limits['capture_payload_cap_bytes']+limits['control_allowance_bytes']
+    require(virtual_envelope < limits['process_AS_cap_bytes'] == 12*1024**3 and
+            physical_estimate < limits['configured_container_RAM_bytes'] and
+            limits['configured_container_RAM_bytes'] == 12*1024**3,
+            'finite explicit reference containment; not native physical admission')
+    return {'mapped_bytes':mapped,'selected_parameter_bytes':params,'touched_source_bytes':touched,
+            'virtual_envelope_bytes':virtual_envelope,'physical_envelope_estimate_bytes':physical_estimate,
+            'native_ResourceArbiter_admission_qualified':False}
+
+def frozen():
+    require(sha(HERE/'prepared-manifest.json') ==
+            '09461af63adc9e53a279a40d6608efcabeaa3d8a405f04d9a55b220bf73ef0f4',
+            'immutable original reference freeze')
+    manifest = metadata(HERE/'prepared-manifest.json')
+    for row in manifest['files']:
+        read(HERE/row['file'],pin=row)
+    for row in manifest['parent_metadata']:
+        read(row['path'],pin=row)
+    followup = metadata(HERE/'prepared-manifest-v2.json')
+    require(followup['original_manifest_sha256'] == sha(HERE/'prepared-manifest.json'),
+            'additive header-bound followup parent identity')
+    for row in followup['files']:
+        read(HERE/row['file'], pin=row)
+    return sha(HERE/'prepared-manifest-v2.json')
--- /dev/null
+++ validate_reference_v2.py
@@ -0,0 +1,252 @@
+"""ROOT ONLY stdlib full-byte/source/lifecycle validation; no tensor arithmetic."""
+import argparse
+import json
+import os
+from pathlib import Path
+import struct
+from support_v2 import HERE, MODEL_SHA, TORCH_GIT, bounds, digest, fd_sha, fingerprint, frozen, metadata, read, require, sha, source_header, unique, write
+
+PREFIX='model.language_model.layers.0.linear_attn.'
+
+def source_readsets(plan,report):
+    """ROOT-only actual indexed source bytes, bounded read; no Torch or math."""
+    require(set(report['source_readsets'])==set(plan['tensors']),'all actual source raw readsets')
+    for shard,pin in plan['selected_shards'].items():
+        fd=os.open(Path(plan['source_directory'])/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+        try:
+            before=fingerprint(os.fstat(fd))
+            source_header(fd,pin)
+            require(fingerprint(os.fstat(fd))==before,
+                    'same authenticated selected source header FD')
+            for name,item in plan['tensors'].items():
+                if item['shard']!=shard:continue
+                if name=='model.language_model.embed_tokens.weight':
+                    actual={str(t):fd_sha(fd,item['absolute_offset']+t*10240,10240) for t in (11,353)}
+                else:actual=fd_sha(fd,item['absolute_offset'],item['bytes'])
+                require(actual==report['source_readsets'][name],'independent actual raw source readset: '+name)
+            require(fingerprint(os.fstat(fd))==before,'original FD unchanged during independent source-range validation')
+        finally:os.close(fd)
+
+def archive(path,pin):
+    raw=read(path,128*1024**2,pin);n,=struct.unpack_from('<Q',raw)
+    require(1<n<=8*1024**2 and 8+n<len(raw),'bounded complete raw archive header')
+    head=json.loads(raw[8:8+n],object_pairs_hook=unique);tensors={};spans=[]
+    for name,item in head.items():
+        if name=='__metadata__':continue
+        dtype,shape=item['dtype'],item['shape'];require(dtype in ('F32','BF16'),'original raw archive dtype')
+        count=1
+        require(type(shape) is list and len(shape)<=8,'bounded actual rank')
+        for dim in shape:
+            require(type(dim) is int and 0<dim<=248320,'bounded actual dimension');count*=dim
+        lo,hi=item['data_offsets'];size=2 if dtype=='BF16' else 4
+        require(type(lo) is type(hi) is int and 0<=lo<hi and hi-lo==count*size and 8+n+hi<=len(raw),
+                'complete original dtype/extent geometry')
+        tensors[name]={'dtype':dtype,'shape':shape,'raw':raw[8+n+lo:8+n+hi]};spans.append((lo,hi))
+    cursor=0
+    for lo,hi in sorted(spans):require(lo==cursor,'all disjoint complete archive bytes');cursor=hi
+    require(8+n+cursor==len(raw),'no trailing/unbound archive bytes')
+    return tensors
+
+def tensor(tensors,name,dtype,shape=None):
+    value=tensors[name]
+    require(value['dtype']==dtype and (shape is None or value['shape']==shape),'actual stage dtype/shape: '+name)
+    return value['raw']
+
+def widened(raw):
+    return b''.join((word<<16).to_bytes(4,'little') for word, in struct.iter_unpack('<H',raw))
+
+def operation(rows,scope,line,name):
+    selected=[row for row in rows if row['scope']==scope and row['source_line']==line and row['operation']==name]
+    require(len(selected)==1,'exact actual selected operator: '+str((scope,line,name)))
+    return selected[0]
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');derived=bounds(plan,4096)
+    invocation=metadata(args.output/'invocation.json');directory=args.output/'captured';report=metadata(directory/'manifest.json')
+    require(invocation['status']=='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION' and
+            invocation['freeze_manifest_sha256']==freeze and invocation['source_plan_sha256']==sha(HERE/'source-plan.json') and
+            invocation['exit_code']==0 and invocation['cleanup_errors']==[] and invocation['source_FD_cleanup_errors']==[] and
+            invocation['source_FDs_closed'] is True and invocation.get('owned_container_absent_after_reap') is True,
+            'actual original source/reference/owned container all retired')
+    cleanup=invocation['foreground_cleanup'];logs=invocation['log_cleanup']
+    require(cleanup['leader_reaped'] is True and cleanup['group_absent'] is True and cleanup['exit_code']==0 and
+            cleanup['all_nonzero_signals_before_reap'] is True and cleanup['stage_errors']==[] and
+            logs['pipe_eof'] is logs['all_closed'] is True and logs['total_bytes']==logs['retained_bytes']<=4*1024**2,
+            'original finite owned process/PGID/EOF cleanup and complete evidence')
+    expected={v['name']:(v['bytes'],v['sha256']) for v in plan['source_files']}
+    require(invocation['source_before']==invocation['source_after'] and
+            {v['name']:(v['bytes'],v['sha256']) for v in invocation['source_before']}==expected and
+            len(invocation['source_before'])==len(expected),'all original source files before/after full SHA coverage')
+    require(report['status']=='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION' and
+            report['schema']=='zerogravity.original27B.layer0-gdn.public-reference.v1' and report['tokens']==[11,353] and
+            report['parameter_profile']=='original27B-raw-source-bf16-v1' and
+            report['source_plan_sha256']==sha(HERE/'source-plan.json') and report['source_repo']==plan['repository'] and
+            report['source_revision']==plan['revision'] and report['snapshot']==plan['snapshot'] and
+            report['pre_library_bounds']==derived and report['native_ResourceArbiter_admission_qualified'] is False and
+            report['full_model_StateRoot_native_exp_GPU_eligible'] is False and report['all_source_parameters_BF16'] is True and
+            report['source_owner_retained_until_owned_process_exit'] is True and report['hooks_removed']==5,
+            'authentic private original27B profile and unchanged eligibility guards')
+    require(report['norm_profile']=={'class':'Qwen3_5RMSNormGated','heads':48,'head_width':128,
+        'raw_gamma_dtype':'torch.bfloat16','weighted_dtype':'torch.bfloat16',
+        'gate_product_dtype':'torch.float32','return_dtype':'torch.bfloat16','epsilon':1e-6,'activation':'silu'},
+        'explicit original27B BF16 gamma round profile')
+    require(report['all_selected_FDs_and_mapped_parameters_unchanged'] is True and
+            report['source_FDs_before']==report['source_FDs_after'] and
+            set(report['source_FDs_before'])==set(plan['selected_shards']),'same exact selected original FD owners')
+    source_readsets(plan,report)
+    expected_parameters={name.removeprefix(PREFIX):item for name,item in plan['tensors'].items() if name.startswith(PREFIX)}
+    require(set(report['parameter_audit'])==set(expected_parameters) and len(expected_parameters)==9,
+            'all and only original learned GDN parameters')
+    for name,item in expected_parameters.items():
+        actual=report['parameter_audit'][name]
+        require(actual['dtype']=='torch.bfloat16' and actual['shape']==item['shape'] and
+                actual['device']=='cpu' and actual['contiguous'] is True and actual['bytes']==item['bytes'] and
+                type(actual['pointer']) is int and actual['pointer']>0 and
+                actual['source_readset_sha256']==report['source_readsets'][PREFIX+name],
+                'actual complete original raw parameter binding: '+name)
+    runtime=report['runtime'];require(runtime['torch']=='2.13.0+cu129' and runtime['torch_git']==TORCH_GIT and
+        runtime['transformers']=='5.17.0' and runtime['cpu_capability']=='AVX2' and runtime['threads']==4 and
+        runtime['interop_threads']==1 and runtime['affinity']==invocation['expected_cpus'] and
+        len(set(runtime['affinity']))==4 and runtime['cuda_available'] is False and runtime['modeling_sha256']==MODEL_SHA,
+        'actual pinned unchanged CPU runtime and four CPUs')
+    dsos=runtime['mapped_shared_objects']
+    require(dsos['/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so']['sha256']==
+            '1b4ed646ce51a4c341808552426a0f9e2e2c04961955759701066d335c1631ab' and
+            dsos['/usr/lib/x86_64-linux-gnu/libm.so.6']['sha256']==
+            'fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0',
+            'actual original exp/CPU/scalar-provider mappings, not native admission')
+    plain=archive(directory/'plain.safetensors',report['plain_archive']);steps=[];payload=sum(len(t['raw']) for t in plain.values())
+    for name,shape in [('A_log',[48]),('dt_bias',[48]),('norm.weight',[128])]:
+        require(digest(tensor(plain,'source.'+name,'BF16',shape))==report['source_readsets'][PREFIX+name],
+                'all original BF16 coefficient/gamma source bytes in reference')
+    require(len(report['steps'])==2,'both true source teachers')
+    for step,record in enumerate(report['steps']):
+        phase='chunk' if step==0 else 'recurrent'
+        require(record['step']==step and record['teacher_token']==[11,353][step] and
+                record['plain_observed_all_outputs_and_state_exact'] is True and record['actual_calls']=={
+                    'Qwen3_5GatedDeltaNet.forward':1,'Qwen3_5RMSNormGated.forward':1,'l2norm':2,
+                    'torch_'+phase+'_gated_delta_rule':1,'causal_conv1d_fn' if step==0 else 'causal_conv1d_update':1},
+                'one original chunk64 first, one actual cached call second')
+        calls=[event for event in record['events'] if event['label']==phase+'.input']
+        require(len(calls)==1 and calls[0]['output_final_state'] is True and
+                calls[0]['use_qk_l2norm_in_kernel'] is True and
+                (step==1 or calls[0]['initial_state'] is None and calls[0]['chunk_size']==64),
+                'actual original phase/initial-state/padded-chunk contract')
+        tensors=archive(directory/f'step-{step}/actual-gdn.safetensors',
+                        {'bytes':record['archive_bytes'],'sha256':record['archive_sha256']});steps.append(tensors)
+        before=metadata(directory/f'step-{step}/prevalidation.json')
+        require(before==dict(record,plain_observed_all_outputs_and_state_exact=False),
+                'raw original diagnostics retained before semantic acceptance')
+        require(set(tensors)==set(record['tensors']),'all original raw observed stages bound')
+        for name,item in tensors.items():
+            meta=record['tensors'][name]
+            require(meta['raw_sha256']==digest(item['raw']) and meta['bytes']==len(item['raw']) and
+                    meta['shape']==item['shape'] and meta['dtype']==('torch.bfloat16' if item['dtype']=='BF16' else 'torch.float32') and
+                    meta['device']=='cpu','actual complete observer payload: '+name)
+            payload+=len(item['raw'])
+        require(sum(len(t['raw']) for t in tensors.values())==record['payload_bytes'],'all actual per-step payload counted')
+        for original,observed,dtype,shape in [('embedding','prefix.embedding','BF16',[1,1,5120]),
+                ('input_norm','prefix.input_norm','BF16',[1,1,5120]),('output','module.output','BF16',[1,1,5120]),
+                ('conv','cache.after.conv','BF16',[1,10240,4]),('state','cache.after.state','F32',[1,48,128,128])]:
+            require(tensor(plain,f'step-{step}.'+original,dtype,shape)==tensor(tensors,observed,dtype,shape),
+                    'all plain/observed source outputs and full actual state: '+original)
+        require(tensor(tensors,phase+'.return.actual_return_1','F32',[1,48,128,128])==
+                tensor(tensors,'cache.after.state','F32',[1,48,128,128]),'actual original returned state equals actual cache')
+        require(tensor(tensors,phase+'.return.actual_return_0','BF16',[1,1,48,128])==
+                tensor(tensors,'norm.input.hidden_states','BF16',[48,128]),'original core return feeds actual gated norm')
+        for name in ('query','key'):
+            before=tensor(tensors,'gdn.pre_beta.'+name,'BF16',[1,1,16,128]);after=tensor(tensors,'gdn.repeated.'+name,'BF16',[1,1,48,128])
+            require(after==b''.join(before[h*256:(h+1)*256]*3 for h in range(16)),
+                    'actual original Q/K repeat_interleave(3), no inferred permutation')
+        rows=record['dispatch'];coef=[v for v in rows if v['scope']=='gdn']
+        require([v['operation'] for v in coef]==['aten.sigmoid.default','aten.to.dtype','aten.exp.default','aten.neg.default',
+                'aten.to.dtype','aten.add.Tensor','aten.softplus.default','aten.mul.Tensor'] and
+                [v['source_line'] for v in coef]==[617]+[619]*7,'actual BF16 A_log .float/coefficient source order')
+        def op_raw(row,which='result',index=0):
+            ref=row[which][index] if which=='arguments' else row[which]
+            require(type(ref) is dict and set(ref)=={'tensor'},'actual selected tensor result/argument')
+            return tensors[ref['tensor']]['raw']
+        require(op_raw(coef[0],'arguments')==tensor(tensors,'projection.in_proj_b.output','BF16',[1,1,48]) and
+                op_raw(coef[0])==tensor(tensors,'gdn.coefficients.beta','BF16',[1,1,48]) and
+                op_raw(coef[1],'arguments')==tensor(plain,'source.A_log','BF16',[48]) and
+                op_raw(coef[1])==widened(tensor(plain,'source.A_log','BF16',[48])) and
+                op_raw(coef[4],'arguments')==tensor(tensors,'projection.in_proj_a.output','BF16',[1,1,48]) and
+                op_raw(coef[4])==widened(tensor(tensors,'projection.in_proj_a.output','BF16',[1,1,48])) and
+                op_raw(coef[5],'arguments',1)==tensor(plain,'source.dt_bias','BF16',[48]) and
+                op_raw(coef[-1])==tensor(tensors,phase+'.input.g','F32',[1,1,48]),
+                'genuine source coefficient producer inputs/results, no reference lookup')
+        beta=tensor(tensors,phase+'.input.beta','BF16',[1,1,48])
+        require(all(0.0<=struct.unpack('<f',(word<<16).to_bytes(4,'little'))[0]<=1.0 for word, in struct.iter_unpack('<H',beta)) and
+                all(value<=0.0 for value, in struct.iter_unpack('<f',tensor(tensors,phase+'.input.g','F32',[1,1,48]))),
+                'actual beta and original negative log-decay ranges')
+        norm={}
+        require(len([row for row in rows if row['scope']=='norm'])==12,'complete unchanged original gated norm operation count')
+        for stage,op,line,dtype,shape in [
+            ('core','aten.to.dtype',227,'F32',[48,128]),('squares','aten.pow.Tensor_Scalar',228,'F32',[48,128]),
+            ('mean','aten.mean.dim',228,'F32',[48,1]),('denominator','aten.add.Tensor',230,'F32',[48,1]),
+            ('inverse','aten.rsqrt.default',230,'F32',[48,1]),('normalized','aten.mul.Tensor',230,'F32',[48,128]),
+            ('normalized_bf16','aten.to.dtype',231,'BF16',[48,128]),('weighted_bf16','aten.mul.Tensor',231,'BF16',[48,128]),
+            ('gate','aten.to.dtype',232,'F32',[48,128]),('silu','aten.silu.default',232,'F32',[48,128]),
+            ('product','aten.mul.Tensor',232,'F32',[48,128]),('return','aten.to.dtype',234,'BF16',[48,128])]:
+            actual=operation(rows,'norm',line,op);name=actual['result']['tensor'];norm[stage]=tensor(tensors,name,dtype,shape)
+        require(tensor(tensors,'norm.input.weight','BF16',[128])==tensor(plain,'source.norm.weight','BF16',[128]) and
+                norm['weighted_bf16']==tensor(tensors,'norm.weighted_bf16.hidden_states','BF16',[48,128]) and
+                norm['return']==tensor(tensors,'norm.return.actual_return_0','BF16',[48,128]),
+                'actual source BF16 gamma product rounds BEFORE FP32 SiLU gate')
+        exp=[v for v in rows if v['scope']=='torch_'+phase+'_gated_delta_rule']
+        require([v['source_line'] for v in exp]==([381,386,413,414,415] if step==0 else [483]) and
+                all(v['operation']=='aten.exp.default' for v in exp),'all actual original initial/cached exponentials')
+        masked=exp[0]['arguments'][0]['tensor'] if step==0 else None
+        for name,item in tensors.items():
+            if name==masked:
+                require(item['shape']==[1,48,1,64,64] and item['dtype']=='F32','actual upper-triangle exp input geometry')
+                for index,(word,) in enumerate(struct.iter_unpack('<I',item['raw'])):
+                    i,j=(index//64)%64,index%64
+                    require(word==0xff800000 if i<j else word&0x7f800000!=0x7f800000,
+                            'ONLY source strictly-upper exp mask may contain canonical negative Inf')
+            else:
+                mask=0x7f80 if item['dtype']=='BF16' else 0x7f800000
+                fmt='<H' if item['dtype']=='BF16' else '<I'
+                require(all(word&mask!=mask for word, in struct.iter_unpack(fmt,item['raw'])),'all other actual raw words finite: '+name)
+        if step:
+            require(tensor(steps[0],'chunk.return.actual_return_1','F32',[1,48,128,128])==
+                    tensor(tensors,'recurrent.input.initial_state','F32',[1,48,128,128]) and
+                    op_raw(exp[0])==tensor(tensors,'recurrent.decay_and_prior.decay_t','F32',[1,48,1,1]),
+                    'actual full initial chunk prior and actual cached exp(g), never zero replacement')
+        else:
+            for name in ('query','key','value','beta','decay'):
+                value=tensors['chunk.padded64.'+name];width=128 if name in ('query','key','value') else 1
+                require(value['dtype']=='F32' and value['shape']==([1,48,64,128] if width==128 else [1,48,64]),
+                        'actual first-token source padded64 layout')
+                for head in range(48):
+                    require(value['raw'][(head*64+1)*width*4:(head+1)*64*width*4]==bytes(63*width*4),
+                            'all63 genuine source padding lanes canonical positive zero')
+    require(payload==report['payload_bytes'] and payload<=128*1024**2,'all actual plain/observed payload bytes counted')
+    require(len(report['prior_prefix_checks'])==6 and all(v['bit_exact'] is True for v in report['prior_prefix_checks']),
+            'independent previously accepted original27B embedding/norm/QKV prefix closure')
+    prior=metadata(plan['prior_public_prefix']['path'],pin=plan['prior_public_prefix'])
+    for step in range(2):
+        for stage,name,dataset in [('embedding',f'step-{step}.embedding',plain),
+                                   ('normalized',f'step-{step}.input_norm',plain),
+                                   ('projected','projection.in_proj_qkv.output',steps[step])]:
+            row=prior['steps'][step][stage]
+            require(widened(tensor(dataset,name,'BF16'))==read(Path(plan['prior_public_prefix']['path']).parent/row['file'],pin=row),
+                    'independent every genuine original prefix word')
+    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}=={
+        'manifest.json','plain.safetensors','step-0/actual-gdn.safetensors','step-0/prevalidation.json',
+        'step-1/actual-gdn.safetensors','step-1/prevalidation.json'},'all captured source/byte artifacts accounted')
+    total_artifacts=0
+    for path in args.output.rglob('*'):
+        require(not path.is_symlink(),'no unknown artifact symlink')
+        if path.is_file():total_artifacts+=path.stat().st_size
+    require(total_artifacts<=256*1024**2-128*1024,'independent whole attempt bound including reserved validation record')
+    require(frozen()==freeze,'all immutable source preparation unchanged')
+    result={'status':'PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY','teachers':[11,353],
+            'heads':48,'cached_input_words_available':804960,'cached_output_words_available':798720,
+            'source_plan_sha256':sha(HERE/'source-plan.json'),'manifest_sha256':sha(directory/'manifest.json'),
+            'invocation_sha256':sha(args.output/'invocation.json'),'payload_bytes':payload,
+            'full_model_112_StateRoots_native_exp_allocation_GPU_or_performance_qualified':False}
+    write(args.output/'validation.json',result);print(json.dumps(result,sort_keys=True))
+
+if __name__=='__main__':main()
--- /dev/null
+++ test_header_bounds_v2.py
@@ -0,0 +1,96 @@
+"""ROOT-only stdlib reader negatives; no model/source payload/runtime calls."""
+from contextlib import contextmanager
+import os
+import stat
+import struct
+import tempfile
+from types import SimpleNamespace
+import unittest
+from unittest.mock import call, patch
+
+from support_v2 import digest, source_header
+
+
+@contextmanager
+def original_header():
+    raw = b'{"header":true}  '
+    with tempfile.TemporaryFile() as file:
+        file.write(struct.pack('<Q', len(raw)) + raw)
+        file.flush()
+        yield file.fileno(), raw, {'file': {'bytes': len(raw)+8},
+                                  'header_bytes': len(raw), 'header_sha256': digest(raw)}
+
+
+class HeaderBoundsTests(unittest.TestCase):
+    def test_valid_two_bounded_reads(self):
+        with original_header() as (fd, raw, pin):
+            with patch('support_v2.os.pread', wraps=os.pread) as pread:
+                self.assertEqual(source_header(fd, pin), (len(raw), raw))
+            self.assertEqual(pread.call_args_list, [call(fd, 8, 0), call(fd, len(raw), 8)])
+
+    def test_oversized_declared_length_never_reads_body(self):
+        with original_header() as (fd, _, pin):
+            n = 8*1024*1024+1
+            pin['header_bytes'] = n
+            with patch('support_v2.os.pread', return_value=struct.pack('<Q', n)) as pread:
+                with self.assertRaisesRegex(ValueError, 'before body read'):
+                    source_header(fd, pin)
+            self.assertEqual(pread.call_args_list, [call(fd, 8, 0)])
+
+    def test_frozen_length_mismatch_never_reads_body(self):
+        with original_header() as (fd, raw, pin):
+            with patch('support_v2.os.pread', return_value=struct.pack('<Q', len(raw)-1)) as pread:
+                with self.assertRaisesRegex(ValueError, 'before body read'):
+                    source_header(fd, pin)
+            self.assertEqual(pread.call_args_list, [call(fd, 8, 0)])
+
+    def test_length_word_short_read_refuses(self):
+        with original_header() as (fd, _, pin):
+            with patch('support_v2.os.pread', return_value=b'1234') as pread:
+                with self.assertRaisesRegex(ValueError, 'length word'):
+                    source_header(fd, pin)
+            self.assertEqual(pread.call_args_list, [call(fd, 8, 0)])
+
+    def test_body_short_read_refuses(self):
+        with original_header() as (fd, raw, pin):
+            with patch('support_v2.os.pread', side_effect=[struct.pack('<Q', len(raw)), raw[:-1]]):
+                with self.assertRaisesRegex(ValueError, 'immutable original source header'):
+                    source_header(fd, pin)
+
+    def test_wrong_digest_refuses(self):
+        with original_header() as (fd, _, pin):
+            pin['header_sha256'] = '0'*64
+            with self.assertRaisesRegex(ValueError, 'immutable original source header'):
+                source_header(fd, pin)
+
+    def test_fd_mutation_refuses(self):
+        with original_header() as (fd, _, pin):
+            before = os.fstat(fd)
+            after = SimpleNamespace(**{key: getattr(before, key) for key in
+                ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')})
+            after.st_ctime_ns += 1
+            with patch('support_v2.os.fstat', side_effect=[before, after]):
+                with self.assertRaisesRegex(ValueError, 'immutable original source header'):
+                    source_header(fd, pin)
+
+    def test_wrong_file_size_never_reads_length(self):
+        with original_header() as (fd, _, pin):
+            pin['file']['bytes'] += 1
+            with patch('support_v2.os.pread') as pread:
+                with self.assertRaisesRegex(ValueError, 'regular selected source FD'):
+                    source_header(fd, pin)
+            pread.assert_not_called()
+
+    def test_nonregular_fd_never_reads_length(self):
+        with original_header() as (fd, _, pin):
+            before = os.fstat(fd)
+            fake = SimpleNamespace(st_mode=stat.S_IFIFO, st_size=before.st_size)
+            with patch('support_v2.os.fstat', return_value=fake):
+                with patch('support_v2.os.pread') as pread:
+                    with self.assertRaisesRegex(ValueError, 'regular selected source FD'):
+                        source_header(fd, pin)
+            pread.assert_not_called()
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ FOLLOWUP_V2.md
@@ -0,0 +1,37 @@
+# Header-read bound followup
+
+Author: spine_review. Source-only additive V2.
+
+- Original freeze `09461af63adc9e53a279a40d6608efcabeaa3d8a405f04d9a55b220bf73ef0f4`
+  and every original source/observer/plan byte remain unchanged.
+- Peer review found capture read the declared header body before validating its
+  size. Independent validation also needed an explicit pre-read cap.
+- Shared `support_v2.source_header`: retained exact-size regular FD; exact 8-byte
+  length read; pinned length, `1 < n <= 8 MiB`, and in-file extent before body
+  read; exact body length/hash and unchanged FD fingerprint afterward.
+- Capture also checks the FD fingerprint still equals its pre-whole-hash value.
+  Independent validator retains its original per-shard FD through all raw reads.
+- `run_reference_v2.py` selects `capture_v2.py`; all V2 entrypoints bind both
+  original frozen source/metadata and the additive V2 manifest.
+- All public math, original operator calls, hooks, source roles, dtypes, mapping
+  geometry, bounds, archives, expected values, cleanup and eligibility unchanged.
+- Nine prepared stdlib reader fixtures: valid bounds, oversized/mismatched/short
+  length, short body, wrong hash, FD mutation, wrong file size and nonregular FD.
+  Synthetic fixtures establish only predicates when ROOT executes them.
+
+ROOT commands from this directory (not run by author):
+
+```bash
+python3 test_header_bounds_v2.py
+python3 run_reference_v2.py --new-output /ABSOLUTE/RUN/root-reference-v2-1
+python3 validate_reference_v2.py --output /ABSOLUTE/RUN/root-reference-v2-1
+```
+
+Original three schema fixtures passed under ROOT, before this followup. No
+public model/reference execution has occurred. Full H48 capture, finite resource
+containment and independent archive validation remain pending. No native
+provider/exp admission, full-model StateRoot, GPU or profile promotion follows.
+
+Author checks: five Python ASTs; original eleven bytes/frozen hashes; static
+review diff. No fixture, model, Torch, operator, provider, container, build or
+GPU execution; no weight/DSO payload hashes.

````

