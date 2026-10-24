# Draft Patches — volume 3

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 52933; SHA256 `c89eb6c0bf0dabf20f6b2a534f286dd4194769540da4f4e85b4056ba0b708530`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/gated_rms.rs
@@ -0,0 +1,222 @@
+//! Explicit BF16 source gamma product rounded before the FP32 SiLU gate.
+use super::*;
+
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum GatedRmsGammaBf16Contract {
+    TorchCpuBf16Avx2H48Width128,
+}
+impl GatedRmsGammaBf16Contract {
+    pub fn validate(self, parameter: &SourceParameter, core: usize, gate: usize) -> Result<usize> {
+        if !parameter.source.is_bound() || parameter.name.is_empty() {
+            return Err(Error::State("gated RMS original source parameter".into()));
+        }
+        if parameter.kind != QuantKind::Bf16
+            || parameter.shape != [128]
+            || parameter.bytes != 256
+            || core != 48 * 128
+            || gate != core
+        {
+            return Err(Error::Unsupported(
+                "gated RMS BF16 gamma H48 width128 contract".into(),
+            ));
+        }
+        Ok(core)
+    }
+}
+
+pub(super) fn result_precision(
+    builder: &PlanBuilder,
+    parameter: &SourceParameter,
+    contract: GatedRmsGammaBf16Contract,
+    inputs: &[PlanValue],
+) -> Result<ActivationPrecision> {
+    if !builder.operator_slice
+        || builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || parameter.source != builder.source
+        || inputs.len() != 2
+        || inputs
+            .iter()
+            .any(|v| !v.contiguous || v.precision != ActivationPrecision::Bf16WidenedF32)
+    {
+        return Err(Error::Unsupported(
+            "gated RMS BF16 gamma pure profile/precision/source".into(),
+        ));
+    }
+    contract.validate(parameter, inputs[0].elements, inputs[1].elements)?;
+    Ok(ActivationPrecision::Bf16WidenedF32)
+}
+
+pub(super) fn record(
+    builder: &mut PlanBuilder,
+    parameter: SourceParameter,
+    core: &PlanValue,
+    gate: &PlanValue,
+    contract: GatedRmsGammaBf16Contract,
+) -> Result<PlanValue> {
+    let elements = contract.validate(&parameter, core.elements, gate.elements)?;
+    let name = parameter.name.clone();
+    builder.one(
+        &name,
+        PlanOperation::TypedGatedRmsGammaBf16 {
+            parameter,
+            contract,
+        },
+        &[*core, *gate],
+        ValueShape::Vector(elements),
+        None,
+    )
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    fn producer(builder: &mut PlanBuilder, width: usize) -> PlanValue {
+        builder
+            .embed_profiled(
+                SourceTensor {
+                    source: builder.source,
+                    name: format!("fixture.{width}"),
+                    kind: QuantKind::Bf16,
+                    columns: width,
+                    rows: 2,
+                    axes: MatrixAxes::OutputInput,
+                },
+                PlanBuilder::inputs().token,
+                EmbeddingContract::Bf16OutputInput,
+            )
+            .unwrap()
+    }
+    #[test]
+    fn closed_bf16_gamma_node_retains_one_pure_bf16_value() {
+        let id = SourceIdentity::for_new_allocation().unwrap();
+        let mut builder = PlanBuilder::new_operator_slice(
+            id,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap();
+        let core = producer(&mut builder, 6144);
+        let gate = producer(&mut builder, 6144);
+        let parameter = SourceParameter {
+            source: id,
+            name: "original.gamma".into(),
+            kind: QuantKind::Bf16,
+            shape: vec![128],
+            bytes: 256,
+        };
+        let value = builder
+            .gated_rms_gamma_bf16_profiled(
+                parameter,
+                &core,
+                &gate,
+                GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128,
+            )
+            .unwrap();
+        assert_eq!(
+            builder.precision(&value),
+            ActivationPrecision::Bf16WidenedF32
+        );
+        let plan = builder.finish_operator_slice(value).unwrap();
+        assert_eq!(plan.plan().nodes().len(), 3);
+        assert!(plan.plan().required_domains().is_empty());
+        assert!(plan.plan().proposed_domains().is_empty());
+        assert_eq!(plan.plan().frontier_delta(), 0);
+    }
+    #[test]
+    fn f32_gamma_wrong_extent_profile_and_width_are_not_implicitly_admitted() {
+        let id = SourceIdentity::for_new_allocation().unwrap();
+        let mut builder = PlanBuilder::new_operator_slice(
+            id,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap();
+        let core = producer(&mut builder, 6144);
+        let gate = producer(&mut builder, 6144);
+        let contract = GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128;
+        let good = SourceParameter {
+            source: id,
+            name: "original.gamma".into(),
+            kind: QuantKind::Bf16,
+            shape: vec![128],
+            bytes: 256,
+        };
+        for bad in [
+            SourceParameter {
+                kind: QuantKind::F32,
+                bytes: 512,
+                ..good.clone()
+            },
+            SourceParameter {
+                bytes: 512,
+                ..good.clone()
+            },
+            SourceParameter {
+                source: SourceIdentity::for_new_allocation().unwrap(),
+                ..good.clone()
+            },
+        ] {
+            assert!(
+                builder
+                    .gated_rms_gamma_bf16_profiled(bad, &core, &gate, contract)
+                    .is_err()
+            );
+        }
+        builder.profile = ProgramNumericalProfile::GgufSourceV1;
+        assert!(
+            builder
+                .gated_rms_gamma_bf16_profiled(good.clone(), &core, &gate, contract)
+                .is_err()
+        );
+        builder.profile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
+        let short = producer(&mut builder, 128);
+        assert!(
+            builder
+                .gated_rms_gamma_bf16_profiled(good, &core, &short, contract)
+                .is_err()
+        );
+        let plan = builder.finish_operator_slice(core).unwrap();
+        assert_eq!(plan.plan().nodes().len(), 3);
+    }
+    #[test]
+    fn strided_valid_h48_inputs_require_explicit_pack_before_recording() {
+        let id = SourceIdentity::for_new_allocation().unwrap();
+        let mut builder = PlanBuilder::new_operator_slice(
+            id,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap();
+        let backing = producer(&mut builder, 6145);
+        let core = builder
+            .view(
+                &backing,
+                TensorView {
+                    offset: 0,
+                    rows: 2,
+                    width: 3072,
+                    row_stride: 3073,
+                },
+            )
+            .unwrap();
+        let gate = producer(&mut builder, 6144);
+        let gamma = SourceParameter {
+            source: id,
+            name: "original.gamma".into(),
+            kind: QuantKind::Bf16,
+            shape: vec![128],
+            bytes: 256,
+        };
+        let contract = GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128;
+        assert!(
+            builder
+                .gated_rms_gamma_bf16_profiled(gamma.clone(), &core, &gate, contract)
+                .is_err()
+        );
+        assert_eq!(builder.nodes.len(), 3);
+        let packed = builder.contiguous(&core).unwrap();
+        let output = builder
+            .gated_rms_gamma_bf16_profiled(gamma, &packed, &gate, contract)
+            .unwrap();
+        let plan = builder.finish_operator_slice(output).unwrap();
+        assert_eq!(plan.plan().nodes().len(), 5);
+        assert!(plan.plan().proposed_domains().is_empty());
+    }
+}
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -12,7 +12,9 @@
 use std::sync::Arc;
 use std::sync::atomic::{AtomicU64, Ordering};
 
+mod gated_rms;
 mod gdn_conv;
+pub use gated_rms::GatedRmsGammaBf16Contract;
 pub use gdn_conv::{CausalConvSiluContract, CausalConvValues};
 mod typed;
 pub use typed::{
@@ -294,6 +296,17 @@
             "backend does not implement explicit causal convolution SiLU contract".into(),
         ))
     }
+    fn gated_rms_gamma_bf16_profiled(
+        &mut self,
+        _parameter: SourceParameter,
+        _core: &Self::Value,
+        _gate: &Self::Value,
+        _contract: GatedRmsGammaBf16Contract,
+    ) -> Result<Self::Value> {
+        Err(Error::Unsupported(
+            "backend does not implement explicit BF16 gamma gated RMS contract".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -474,6 +487,10 @@
     TypedOrdinaryRms {
         parameter: SourceParameter,
         contract: OrdinaryRmsContract,
+    },
+    TypedGatedRmsGammaBf16 {
+        parameter: SourceParameter,
+        contract: GatedRmsGammaBf16Contract,
     },
     Embed {
         tensor: SourceTensor,
@@ -822,6 +839,10 @@
                 parameter,
                 contract,
             } => gdn_conv::result_precision(self, parameter, *contract, inputs),
+            PlanOperation::TypedGatedRmsGammaBf16 {
+                parameter,
+                contract,
+            } => gated_rms::result_precision(self, parameter, *contract, inputs),
             PlanOperation::View(_) | PlanOperation::Contiguous => inputs
                 .first()
                 .map(|value| value.precision)
@@ -839,6 +860,7 @@
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
+                        | PlanOperation::TypedGatedRmsGammaBf16 { .. }
                 ) =>
             {
                 Ok(ActivationPrecision::F32)
@@ -1114,6 +1136,15 @@
         contract: CausalConvSiluContract,
     ) -> Result<CausalConvValues<PlanValue>> {
         gdn_conv::record(self, parameter, input, history, contract)
+    }
+    fn gated_rms_gamma_bf16_profiled(
+        &mut self,
+        parameter: SourceParameter,
+        core: &PlanValue,
+        gate: &PlanValue,
+        contract: GatedRmsGammaBf16Contract,
+    ) -> Result<PlanValue> {
+        gated_rms::record(self, parameter, core, gate, contract)
     }
     fn embed(&mut self, tensor: SourceTensor, token: ParameterRef) -> Result<PlanValue> {
         self.source_tensor(&tensor)?;
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma/tests.rs
@@ -0,0 +1,72 @@
+use super::*;
+
+fn source() -> SourceIdentity {
+    SourceIdentity::for_new_allocation().unwrap()
+}
+fn tensor(raw: &[u8]) -> SourceTensorRef<'_> {
+    SourceTensorRef {
+        kind: QuantKind::Bf16,
+        shape: &[128],
+        packed_bytes: raw,
+    }
+}
+
+#[test]
+fn bf16_gamma_product_has_its_own_rounding_boundary_before_gate() {
+    // Exact dyadic product253/256 *129/128 =32637/32768 =F32 3f7efa00.
+    // Closest BF16 is255/256 =3f7f; the former F32 product cannot pass through.
+    let n = f32::from_bits(0x3f7d_0000);
+    let g = f32::from_bits(0x3f81_0000);
+    assert_eq!((g * n).to_bits(), 0x3f7e_fa00);
+    assert_eq!(gamma_product(n, g).unwrap().to_bits(), 0x3f7f_0000);
+    assert_ne!(gamma_product(n, g).unwrap().to_bits(), (g * n).to_bits());
+}
+
+#[test]
+fn h48_signed_zero_and_source_identity_survive_private_outputs() {
+    let raw = 0x3f80u16.to_le_bytes().repeat(128);
+    let mut core = vec![0.0; ELEMENTS];
+    core[0] = -0.0;
+    let gate = vec![1.0; ELEMENTS];
+    let id = source();
+    let out = gated_rms_128_bf16_gamma_h48(id, tensor(&raw), &core, &gate).unwrap();
+    assert_eq!(out.source_identity(), id);
+    assert_eq!(out.mean().len(), 48);
+    assert_eq!(out.inverse().len(), 48);
+    assert_eq!(out.weighted_bf16().len(), 6144);
+    assert_eq!(out.output_bf16().len(), 6144);
+    assert_eq!(out.weighted_bf16()[0].to_bits(), 0x8000_0000);
+    assert_eq!(out.output_bf16()[0].to_bits(), 0x8000_0000);
+    assert!(out.product_fp32().iter().all(|v| v.is_finite()));
+}
+
+#[test]
+fn source_codec_extent_headcount_and_precision_refuse_without_outputs() {
+    let raw = 0x3f80u16.to_le_bytes().repeat(128);
+    let core = vec![1.0; ELEMENTS];
+    let gate = core.clone();
+    let id = source();
+    assert!(
+        gated_rms_128_bf16_gamma_h48(SourceIdentity::unbound(), tensor(&raw), &core, &gate)
+            .is_err()
+    );
+    for bad in [
+        SourceTensorRef {
+            kind: QuantKind::F32,
+            ..tensor(&raw)
+        },
+        SourceTensorRef {
+            shape: &[1, 128],
+            ..tensor(&raw)
+        },
+        tensor(&raw[..255]),
+    ] {
+        assert!(gated_rms_128_bf16_gamma_h48(id, bad, &core, &gate).is_err());
+    }
+    assert!(gated_rms_128_bf16_gamma_h48(id, tensor(&raw), &core[..128], &gate[..128]).is_err());
+    for invalid in [f32::NAN, f32::INFINITY, 1.0001] {
+        let mut bad = gate.clone();
+        bad[13] = invalid;
+        assert!(gated_rms_128_bf16_gamma_h48(id, tensor(&raw), &core, &bad).is_err());
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma.rs
@@ -0,0 +1,158 @@
+//! Distinct original27B BF16 gamma rounding, never the F32 gamma prefix.
+use super::{EPSILON, WIDTH, round_bf16, sum_inner_128};
+use crate::accelerator::SourceIdentity;
+use crate::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16;
+use crate::quant::QuantKind;
+use crate::{Error, Result};
+use zerogravity_core::source::SourceTensorRef;
+
+const HEADS: usize = 48;
+const ELEMENTS: usize = HEADS * WIDTH;
+
+/// Private values only. Original source/state/publication authority stays outside.
+#[derive(Debug)]
+pub struct Bf16GammaGatedRms128 {
+    source: SourceIdentity,
+    mean: Vec<f32>,
+    inverse: Vec<f32>,
+    normalized_bf16: Vec<f32>,
+    weighted_bf16: Vec<f32>,
+    silu_fp32: Vec<f32>,
+    product_fp32: Vec<f32>,
+    output_bf16: Vec<f32>,
+}
+impl Bf16GammaGatedRms128 {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn mean(&self) -> &[f32] {
+        &self.mean
+    }
+    pub fn inverse(&self) -> &[f32] {
+        &self.inverse
+    }
+    pub fn normalized_bf16(&self) -> &[f32] {
+        &self.normalized_bf16
+    }
+    pub fn weighted_bf16(&self) -> &[f32] {
+        &self.weighted_bf16
+    }
+    pub fn silu_fp32(&self) -> &[f32] {
+        &self.silu_fp32
+    }
+    pub fn product_fp32(&self) -> &[f32] {
+        &self.product_fp32
+    }
+    pub fn output_bf16(&self) -> &[f32] {
+        &self.output_bf16
+    }
+}
+
+fn exact_bf16(input: &[f32]) -> bool {
+    input
+        .iter()
+        .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
+}
+fn validate(
+    source: SourceIdentity,
+    gamma: SourceTensorRef<'_>,
+    core: &[f32],
+    gate: &[f32],
+) -> Result<[f32; WIDTH]> {
+    if !source.is_bound() {
+        return Err(Error::State("gated RMS BF16 gamma source".into()));
+    }
+    if gamma.kind != QuantKind::Bf16 {
+        return Err(Error::Unsupported(
+            "gated RMS requires original BF16 gamma".into(),
+        ));
+    }
+    if gamma.shape != [128]
+        || gamma.packed_bytes.len() != 256
+        || core.len() != ELEMENTS
+        || gate.len() != ELEMENTS
+    {
+        return Err(Error::Malformed(
+            "gated RMS BF16 gamma H48 width128 extent".into(),
+        ));
+    }
+    gamma.validate()?;
+    if !exact_bf16(core) || !exact_bf16(gate) {
+        return Err(Error::Malformed(
+            "gated RMS finite exact BF16 core/gate".into(),
+        ));
+    }
+    let weights = std::array::from_fn(|i| {
+        f32::from_bits(
+            u32::from(u16::from_le_bytes([
+                gamma.packed_bytes[2 * i],
+                gamma.packed_bytes[2 * i + 1],
+            ])) << 16,
+        )
+    });
+    if !exact_bf16(&weights) {
+        return Err(Error::Malformed("gated RMS finite original gamma".into()));
+    }
+    Ok(weights)
+}
+fn gamma_product(normalized: f32, weight: f32) -> Result<f32> {
+    // Original BF16 Tensor multiply rounds HERE, before FP32 SiLU product.
+    round_bf16(weight * normalized)
+}
+
+/// Genuine raw BF16 gamma128 plus `[48,128]` BF16 core/z, epsilon1e-6.
+/// Same accepted mean/RNE/SLEEF helpers; no exp/A_log/initial/cache/model engine.
+pub fn gated_rms_128_bf16_gamma_h48(
+    source: SourceIdentity,
+    gamma: SourceTensorRef<'_>,
+    core: &[f32],
+    gate: &[f32],
+) -> Result<Bf16GammaGatedRms128> {
+    // Every source/codec/extent/precision guard precedes private allocations/math.
+    let gamma = validate(source, gamma, core, gate)?;
+    let squares = core.iter().map(|&v| v * v).collect::<Vec<_>>();
+    let sums = sum_inner_128(source, &squares)?;
+    let mut mean = Vec::with_capacity(HEADS);
+    let mut inverse = Vec::with_capacity(HEADS);
+    let mut normalized_bf16 = Vec::with_capacity(ELEMENTS);
+    let mut weighted_bf16 = Vec::with_capacity(ELEMENTS);
+    for (row, sum) in core.chunks_exact(WIDTH).zip(sums) {
+        let m = sum / WIDTH as f32;
+        let denominator = m + EPSILON;
+        let inv = 1.0f32 / denominator.sqrt();
+        if !denominator.is_finite() || !inv.is_finite() {
+            return Err(Error::Malformed("nonfinite gated RMS inverse".into()));
+        }
+        mean.push(m);
+        inverse.push(inv);
+        for (&v, &w) in row.iter().zip(&gamma) {
+            let normalized = round_bf16(v * inv)?;
+            normalized_bf16.push(normalized);
+            weighted_bf16.push(gamma_product(normalized, w)?);
+        }
+    }
+    let silu_fp32 = silu_bf16_widened_f32_contiguous16(source, gate)?;
+    let mut product_fp32 = Vec::with_capacity(ELEMENTS);
+    let mut output_bf16 = Vec::with_capacity(ELEMENTS);
+    for (&w, &s) in weighted_bf16.iter().zip(&silu_fp32) {
+        let product = w * s;
+        if !product.is_finite() {
+            return Err(Error::Malformed("nonfinite gated RMS gate product".into()));
+        }
+        product_fp32.push(product);
+        output_bf16.push(round_bf16(product)?);
+    }
+    Ok(Bf16GammaGatedRms128 {
+        source,
+        mean,
+        inverse,
+        normalized_bf16,
+        weighted_bf16,
+        silu_fp32,
+        product_fp32,
+        output_bf16,
+    })
+}
+
+#[cfg(test)]
+mod tests;
--- a/crates/zerogravity-cpu/src/original_hf/norm.rs
+++ b/crates/zerogravity-cpu/src/original_hf/norm.rs
@@ -5,8 +5,10 @@
 //! normalized values to BF16 before multiplying original F32 gamma. This
 //! source-36-F32 profile enables neither a model nor CUDA/FLA execution.
 
+mod bf16_gamma;
 mod ordinary;
 
+pub use bf16_gamma::{Bf16GammaGatedRms128, gated_rms_128_bf16_gamma_h48};
 pub use ordinary::{OrdinaryRms, ordinary_rms};
 
 use super::reduction::sum_inner_128;
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -931,7 +931,8 @@
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
-                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedCausalConvSilu { .. }
+                | PlanOperation::TypedGatedRmsGammaBf16 { .. } => Err(Error::Unsupported(
                     "resident batch has no qualified typed lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -349,6 +349,7 @@
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
+                        | PlanOperation::TypedGatedRmsGammaBf16 { .. }
                 )
             })
         {
@@ -1431,7 +1432,8 @@
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
-                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedCausalConvSilu { .. }
+                | PlanOperation::TypedGatedRmsGammaBf16 { .. } => Err(Error::Unsupported(
                     "resident typed operation has no qualified lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -650,3 +650,36 @@
     }
     Ok(values)
 }
+
+/// Original layer0 H48 source gamma BF16, never a full-GDN/session capability.
+pub(crate) fn gated_rms_bf16_gamma_slice<B: MathematicalBackend>(
+    source: &impl crate::operator_slice::OperatorSliceSource,
+    math: &mut B,
+    layer: usize,
+    core: &B::Value,
+    gate: &B::Value,
+) -> Result<B::Value> {
+    let parameter = crate::operator_slice::gated_rms::gated_rms_role(source, layer)?;
+    if math.elements(core) != 6144
+        || math.elements(gate) != 6144
+        || math.precision(core) != crate::accelerator::program::ActivationPrecision::Bf16WidenedF32
+        || math.precision(gate) != crate::accelerator::program::ActivationPrecision::Bf16WidenedF32
+    {
+        return Err(Error::Unsupported("gated RMS H48 BF16 core/gate".into()));
+    }
+    let result = math.gated_rms_gamma_bf16_profiled(
+        parameter,
+        core,
+        gate,
+        crate::accelerator::program::GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128,
+    )?;
+    if math.elements(&result) != 6144
+        || math.precision(&result)
+            != crate::accelerator::program::ActivationPrecision::Bf16WidenedF32
+    {
+        return Err(Error::State(
+            "gated RMS returned wrong precision/width".into(),
+        ));
+    }
+    Ok(result)
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs
@@ -0,0 +1,389 @@
+//! Synthetic full-source H48/codec/lease tests; no trained arithmetic proof.
+use super::*;
+use crate::accelerator::program::{
+    EmbeddingContract, ProjectionContract, SourceParameter, SourceTensor,
+};
+use crate::operator_slice::gated_rms::{GatedRmsSliceInput, gated_rms_role};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use zerogravity_core::quant::QuantKind;
+use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
+
+fn h48_source(
+    fixture: &Fixture,
+    bf16_gamma: bool,
+) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    let text = &mut cfg["text_config"];
+    text["num_hidden_layers"] = json!(1);
+    text["layer_types"] = json!(["linear_attention"]);
+    text["hidden_size"] = json!(256);
+    text["linear_num_key_heads"] = json!(16);
+    text["linear_num_value_heads"] = json!(48);
+    text["linear_key_head_dim"] = json!(128);
+    text["linear_value_head_dim"] = json!(128);
+    let cfg = serde_json::to_vec(&cfg).unwrap();
+    let specs = QwenTextConfig::from_json(&cfg)
+        .unwrap()
+        .tensor_specs()
+        .unwrap();
+    let mut header = BTreeMap::new();
+    let mut map = BTreeMap::new();
+    let mut payload = Vec::new();
+    for spec in specs {
+        let gamma = spec.name.ends_with("linear_attn.norm.weight");
+        let bf16 = (gamma && bf16_gamma) || spec.shape.len() != 1 || spec.name.ends_with("dt_bias");
+        let begin = payload.len();
+        for _ in 0..spec.shape.iter().product::<u64>() {
+            if bf16 {
+                payload.extend_from_slice(&0x3e80u16.to_le_bytes());
+            } else {
+                payload.extend_from_slice(&0.25f32.to_le_bytes());
+            }
+        }
+        header.insert(
+            spec.name.clone(),
+            json!({"dtype":if bf16 {"BF16"} else {"F32"},
+            "shape":spec.shape,"data_offsets":[begin,payload.len()]}),
+        );
+        map.insert(spec.name, "model.safetensors");
+    }
+    let header = serde_json::to_vec(&header).unwrap();
+    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
+    bytes.extend(header);
+    bytes.extend(&payload);
+    assert!(bytes.len() < 16 * 1024 * 1024);
+    let path = fixture.0.join("model.safetensors");
+    fs::write(&path, &bytes).unwrap();
+    let index =
+        serde_json::to_vec(&json!({"metadata":{"total_size":payload.len()},"weight_map":map}))
+            .unwrap();
+    let index_sha = digest(&index);
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "gated-rms-h48-source-test").unwrap();
+    arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
+    let host = TransferAdmission {
+        arbiter,
+        resource: key,
+        owner: ResourceOwner::new("synthetic", "gated-rms-h48").unwrap(),
+    };
+    let artifact =
+        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).unwrap();
+    let source = SourceWeights::prepare_verified_windows(
+        &index,
+        &index_sha,
+        vec![
+            ExpectedSourceShard::new(
+                "model.safetensors".into(),
+                artifact,
+                bytes.len() as u64,
+                digest(&bytes),
+            )
+            .unwrap(),
+        ],
+        WindowOpenLimits {
+            max_source_bytes: 16 * 1024 * 1024,
+            max_metadata_bytes: 32 * 1024 * 1024,
+            max_shards: 1,
+            max_tensors: 200,
+            scan_chunk_bytes: 64 * 1024,
+        },
+        &host,
+    )
+    .unwrap()
+    .open()
+    .unwrap();
+    (source, host, cfg, index_sha)
+}
+
+fn operators(fixture: &Fixture, bf16_gamma: bool) -> (OriginalWindowOperators, TransferAdmission) {
+    let (source, host, cfg, index) = h48_source(fixture, bf16_gamma);
+    let ops = OriginalWindowOperators::bind(
+        source,
+        &cfg,
+        &digest(&cfg),
+        &index,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )
+    .unwrap();
+    (ops, host)
+}
+fn raw() -> [Vec<u8>; 2] {
+    [
+        0x3f7du16.to_le_bytes().repeat(6144),
+        0xbf00u16.to_le_bytes().repeat(6144),
+    ]
+}
+fn input<'a>(ops: &OriginalWindowOperators, raw: &'a [Vec<u8>; 2]) -> GatedRmsSliceInput<'a> {
+    GatedRmsSliceInput::from_bf16_bytes(
+        ops.source_identity(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        &raw[0],
+        &raw[1],
+    )
+    .unwrap()
+}
+struct ParameterFault<'a> {
+    source: &'a OriginalWindowOperators,
+    norm: SourceParameter,
+    profile: ProgramNumericalProfile,
+}
+impl OperatorSliceSource for ParameterFault<'_> {
+    fn geometry(&self) -> &crate::model::ModelGeometry {
+        self.source.geometry()
+    }
+    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
+        self.source.source_identity()
+    }
+    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    fn rms_epsilon(&self) -> f32 {
+        self.source.rms_epsilon()
+    }
+    fn matrix_descriptor(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
+        self.source.matrix_descriptor(role)
+    }
+    fn parameter_descriptor(&self, role: QwenParameterRole) -> Result<SourceParameter> {
+        assert_eq!(role, QwenParameterRole::GdnNorm(0));
+        Ok(self.norm.clone())
+    }
+    fn embedding_contract(&self) -> EmbeddingContract {
+        self.source.embedding_contract()
+    }
+    fn projection_contract(&self) -> ProjectionContract {
+        self.source.projection_contract()
+    }
+}
+#[test]
+fn exact_original_bf16_gamma_role_codec_extent_source_and_profile_are_required() {
+    let fixture = Fixture::new();
+    let (ops, host) = operators(&fixture, true);
+    let good = gated_rms_role(&ops, 0).unwrap();
+    let before = reserved(&host);
+    assert_eq!(
+        (good.kind, good.shape.clone(), good.bytes),
+        (QuantKind::Bf16, vec![128], 256)
+    );
+    for mut bad in [
+        SourceParameter {
+            kind: QuantKind::F32,
+            bytes: 512,
+            ..good.clone()
+        },
+        SourceParameter {
+            bytes: 512,
+            ..good.clone()
+        },
+        SourceParameter {
+            shape: vec![127],
+            ..good.clone()
+        },
+        SourceParameter {
+            name: "model.language_model.layers.1.linear_attn.norm.weight".into(),
+            ..good.clone()
+        },
+        SourceParameter {
+            source: crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
+            ..good.clone()
+        },
+    ] {
+        let fault = ParameterFault {
+            source: &ops,
+            norm: bad.clone(),
+            profile: ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        };
+        assert!(gated_rms_role(&fault, 0).is_err());
+        assert_eq!(reserved(&host), before);
+        // Also test a different unsupported raw codec without changing geometry.
+        bad.kind = QuantKind::F16;
+        bad.bytes = 256;
+        assert!(
+            gated_rms_role(
+                &ParameterFault {
+                    source: &ops,
+                    norm: bad,
+                    profile: ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+                },
+                0
+            )
+            .is_err()
+        );
+    }
+    assert!(
+        gated_rms_role(
+            &ParameterFault {
+                source: &ops,
+                norm: good,
+                profile: ProgramNumericalProfile::GgufSourceV1
+            },
+            0
+        )
+        .is_err()
+    );
+    assert!(gated_rms_role(&ops, 1).is_err());
+    let f32_fixture = Fixture::new();
+    let (f32_ops, f32_host) = operators(&f32_fixture, false);
+    assert_eq!(
+        f32_ops
+            .parameter_descriptor(QwenParameterRole::GdnNorm(0))
+            .unwrap()
+            .kind,
+        QuantKind::F32
+    );
+    assert!(gated_rms_role(&f32_ops, 0).is_err());
+    drop(f32_ops);
+    assert_eq!(reserved(&f32_host), 0);
+    drop(ops);
+    assert_eq!(reserved(&host), 0);
+}
+#[test]
+fn aggregate_cancel_pressure_retained_inputs_and_result_own_their_actual_source() {
+    let fixture = Fixture::new();
+    let (ops, host) = operators(&fixture, true);
+    let before = reserved(&host);
+    let raw = raw();
+    let input = input(&ops, &raw);
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        ops.execute_gated_rms_bf16_gamma_slice(0, input, Some(&cancelled))
+            .is_err()
+    );
+    assert_eq!(reserved(&host), before);
+    let input_need = (crate::host_math::gated_rms::INPUT_BYTES
+        + crate::host_math::gated_rms::INPUT_SCRATCH) as u64;
+    let output_need = (crate::host_math::gated_rms::OUTPUT_BYTES
+        + crate::host_math::gated_rms::OUTPUT_SCRATCH) as u64;
+    let pressure = host
+        .arbiter
+        .admit(
+            &host.resource,
+            host.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: 128 * 1024 * 1024 - before - input_need - output_need + 1,
+            }],
+        )
+        .unwrap();
+    let pressured = reserved(&host);
+    assert!(
+        ops.execute_gated_rms_bf16_gamma_slice(0, input, None)
+            .is_err()
+    );
+    assert_eq!(reserved(&host), pressured);
+    drop(pressure);
+    assert_eq!(reserved(&host), before);
+    cancelled.store(false, Ordering::Release);
+    let mut math =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, Some(&cancelled), None);
+    let (core, gate) = math.import_gated_rms_input(input).unwrap();
+    assert_eq!(reserved(&host), before + input_need);
+    cancelled.store(true, Ordering::Release);
+    assert!(crate::forward::gated_rms_bf16_gamma_slice(&ops, &mut math, 0, &core, &gate).is_err());
+    assert_eq!(reserved(&host), before + input_need);
+    drop(core);
+    drop(gate);
+    drop(math);
+    assert_eq!(reserved(&host), before);
+    let result = ops
+        .execute_gated_rms_bf16_gamma_slice(0, input, None)
+        .unwrap();
+    assert_eq!(result.core_bf16().unwrap().len(), 6144);
+    assert_eq!(result.gate_bf16().unwrap().len(), 6144);
+    assert_eq!(result.output_bf16().unwrap().len(), 6144);
+    assert_eq!(reserved(&host), before + input_need + output_need);
+    let snapshot = host.arbiter.snapshot(&host.resource).unwrap();
+    assert_eq!(
+        snapshot.per_class.get(&AllocationClass::Output),
+        Some(&(3 * 6144 * 4))
+    );
+    assert_eq!(
+        snapshot
+            .per_class
+            .get(&AllocationClass::TransferStaging)
+            .copied()
+            .unwrap_or(0),
+        0
+    );
+    // This is shared-helper wiring evidence, not an independent math oracle.
+    let parameter = ops
+        .source()
+        .parameter(&gated_rms_role(&ops, 0).unwrap().name)
+        .unwrap();
+    let gamma = ops.source().read_parameter(&parameter, 0..128).unwrap();
+    let expected = zerogravity_cpu::original_hf::norm::gated_rms_128_bf16_gamma_h48(
+        ops.source_identity(),
+        zerogravity_core::source::SourceTensorRef {
+            kind: QuantKind::Bf16,
+            shape: &[128],
+            packed_bytes: gamma.as_slice(),
+        },
+        result.core_bf16().unwrap(),
+        result.gate_bf16().unwrap(),
+    )
+    .unwrap();
+    same(expected.output_bf16(), result.output_bf16().unwrap());
+    drop(expected);
+    drop(gamma);
+    let bits = result
+        .output_bf16()
+        .unwrap()
+        .iter()
+        .map(|v| v.to_bits())
+        .collect::<Vec<_>>();
+    drop(ops);
+    assert_eq!(
+        result
+            .output_bf16()
+            .unwrap()
+            .iter()
+            .map(|v| v.to_bits())
+            .collect::<Vec<_>>(),
+        bits
+    );
+    assert!(reserved(&host) > 0);
+    drop(result);
+    assert_eq!(reserved(&host), 0);
+    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
+}
+#[test]
+fn external_bad_bindings_and_optional_provider_refuse_before_callback_or_new_lease() {
+    struct Provider(std::sync::atomic::AtomicBool);
+    impl crate::accelerator::MatVecKernel for Provider {
+        fn try_execute(
+            &self,
+            _region: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            self.0.store(true, Ordering::Release);
+            panic!("must refuse before provider")
+        }
+    }
+    let fixture = Fixture::new();
+    let (ops, host) = operators(&fixture, true);
+    let before = reserved(&host);
+    let raw = raw();
+    for (id, profile) in [
+        (
+            crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (ops.source_identity(), ProgramNumericalProfile::GgufSourceV1),
+    ] {
+        let bad = GatedRmsSliceInput::from_bf16_bytes(id, profile, &raw[0], &raw[1]);
+        match bad {
+            Ok(input) => assert!(
+                ops.execute_gated_rms_bf16_gamma_slice(0, input, None)
+                    .is_err()
+            ),
+            Err(_) => assert_eq!(profile, ProgramNumericalProfile::GgufSourceV1),
+        }
+        assert_eq!(reserved(&host), before);
+    }
+    let provider = Provider(std::sync::atomic::AtomicBool::new(false));
+    let math = crate::host_math::HostMath::for_window_operator_slice(&ops, None, Some(&provider));
+    assert!(math.import_gated_rms_input(input(&ops, &raw)).is_err());
+    assert!(!provider.0.load(Ordering::Acquire));
+    assert_eq!(reserved(&host), before);
+    drop(math);
+    drop(ops);
+    assert_eq!(reserved(&host), 0);
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/gdn_gated_rms_slice.rs
@@ -0,0 +1,179 @@
+//! One pure original BF16 gamma gated RMS call on explicit observed operands.
+use super::OperatorSliceSource;
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{
+    GatedRmsGammaBf16Contract, MatrixAxes, PlanBuilder, PlanValue, ProgramNumericalProfile,
+    SourceParameter,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, LoadedQwenModel, Result};
+use zerogravity_core::quant::QuantKind;
+
+pub(crate) fn gated_rms_role(
+    source: &impl OperatorSliceSource,
+    layer: usize,
+) -> Result<SourceParameter> {
+    if layer != 0
+        || !source.source_identity().is_bound()
+        || source.program_numerical_profile()
+            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
+        || source.geometry().recurrent_key_heads != Some(16)
+        || source.geometry().recurrent_value_heads != Some(48)
+        || source.rms_epsilon().to_bits() != 1e-6f32.to_bits()
+    {
+        return Err(Error::Unsupported(
+            "original gated RMS BF16 gamma layer0/H48 profile".into(),
+        ));
+    }
+    let parameter = source.parameter_descriptor(QwenParameterRole::GdnNorm(layer))?;
+    let qkv = source.matrix_descriptor(QwenMatrixRole::GdnQkv(layer))?;
+    let gate = source.matrix_descriptor(QwenMatrixRole::GdnGate(layer))?;
+    let output = source.matrix_descriptor(QwenMatrixRole::GdnOutput(layer))?;
+    let hidden = usize::try_from(source.geometry().hidden_size)
+        .map_err(|_| Error::Limit("gated RMS original hidden width".into()))?;
+    if parameter.source != source.source_identity()
+        || parameter.name != "model.language_model.layers.0.linear_attn.norm.weight"
+        || qkv.source != source.source_identity()
+        || qkv.rows != 10240
+        || gate.source != source.source_identity()
+        || gate.rows != 6144
+        || output.source != source.source_identity()
+        || output.columns != 6144
+        || qkv.columns != hidden
+        || gate.columns != hidden
+        || output.rows != hidden
+        || [&qkv, &gate, &output]
+            .iter()
+            .any(|matrix| matrix.kind != QuantKind::Bf16 || matrix.axes != MatrixAxes::OutputInput)
+    {
+        return Err(Error::State(
+            "original gated RMS raw role/width/source".into(),
+        ));
+    }
+    GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128.validate(&parameter, 6144, 6144)?;
+    Ok(parameter)
+}
+
+/// Representation only. The qualifier authenticates actual source call provenance.
+#[derive(Clone, Copy)]
+pub struct GatedRmsSliceInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    core: &'a [u8],
+    gate: &'a [u8],
+}
+impl<'a> GatedRmsSliceInput<'a> {
+    pub fn from_bf16_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        core: &'a [u8],
+        gate: &'a [u8],
+    ) -> Result<Self> {
+        let value = Self {
+            source,
+            profile,
+            core,
+            gate,
+        };
+        value.validate()?;
+        Ok(value)
+    }
+    pub fn source_identity(self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    pub fn raw_inputs(self) -> [&'a [u8]; 2] {
+        [self.core, self.gate]
+    }
+    pub(crate) fn validate(self) -> Result<()> {
+        if !self.source.is_bound()
+            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+            || self.core.len() != 6144 * 2
+            || self.gate.len() != 6144 * 2
+            || [self.core, self.gate].iter().any(|raw| {
+                raw.chunks_exact(2)
+                    .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
+            })
+        {
+            return Err(Error::Malformed(
+                "original gated RMS finite H48 BF16 operands/source/profile".into(),
+            ));
+        }
+        Ok(())
+    }
+}
+impl LoadedQwenModel {
+    pub fn append_gated_rms_bf16_gamma_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        core: &PlanValue,
+        gate: &PlanValue,
+    ) -> Result<PlanValue> {
+        self.require_operator_slice()?;
+        crate::forward::gated_rms_bf16_gamma_slice(self, builder, layer, core, gate)
+    }
+}
+
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+pub struct GatedRmsSliceOutput {
+    source: SourceIdentity,
+    core: crate::host_math::HostValue,
+    gate: crate::host_math::HostValue,
+    output: crate::host_math::HostValue,
+}
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+impl GatedRmsSliceOutput {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    }
+    pub fn core_bf16(&self) -> Result<&[f32]> {
+        self.core.as_slice()
+    }
+    pub fn gate_bf16(&self) -> Result<&[f32]> {
+        self.gate.as_slice()
+    }
+    pub fn output_bf16(&self) -> Result<&[f32]> {
+        self.output.as_slice()
+    }
+}
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    pub fn append_gated_rms_bf16_gamma_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        core: &PlanValue,
+        gate: &PlanValue,
+    ) -> Result<PlanValue> {
+        crate::forward::gated_rms_bf16_gamma_slice(self, builder, layer, core, gate)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_gated_rms_bf16_gamma_slice(
+        &self,
+        layer: usize,
+        input: GatedRmsSliceInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<GatedRmsSliceOutput> {
+        gated_rms_role(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let (core, gate) = math.import_gated_rms_input(input)?;
+        let output =
+            crate::forward::gated_rms_bf16_gamma_slice(self, &mut math, layer, &core, &gate)?;
+        if cancelled.is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State("gated RMS private fragment cancelled".into()));
+        }
+        Ok(GatedRmsSliceOutput {
+            source: self.source_identity(),
+            core,
+            gate,
+            output,
+        })
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/host_math/gated_rms.rs
@@ -0,0 +1,139 @@
+//! Same-source aggregate leases; no provider fallback or state owner.
+use super::*;
+use crate::accelerator::program::GatedRmsGammaBf16Contract;
+use crate::operator_slice::gated_rms::GatedRmsSliceInput;
+
+pub(crate) const INPUT_BYTES: usize = 2 * 6144 * 4;
+pub(crate) const INPUT_SCRATCH: usize = 64 * 1024;
+pub(crate) const OUTPUT_BYTES: usize = 6144 * 4;
+// Seven helper arrays +squares/sums/gamma and final copy fit within this old floor.
+pub(crate) const OUTPUT_SCRATCH: usize = 960 * 1024;
+impl HostMath<'_, '_, '_> {
+    fn gated_rms_windows(&self) -> Result<&crate::original_binding::OriginalWindowOperators> {
+        if self.region_kernel.is_some()
+            || self.engine.is_some()
+            || self.state.is_some()
+            || self.source.program_numerical_profile()
+                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported(
+                "gated RMS private pinned CPU window contract".into(),
+            ));
+        }
+        match &self.source {
+            HostSource::Windows(source) => Ok(source),
+            HostSource::Loaded(_) => Err(Error::Unsupported(
+                "gated RMS requires admitted original windows".into(),
+            )),
+        }
+    }
+    fn gated_rms_cancel(&self) -> Result<()> {
+        if self
+            .cancelled
+            .is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire))
+        {
+            return Err(Error::State("gated RMS private fragment cancelled".into()));
+        }
+        Ok(())
+    }
+    pub(crate) fn import_gated_rms_input(
+        &self,
+        input: GatedRmsSliceInput<'_>,
+    ) -> Result<(HostValue, HostValue)> {
+        let source = self.gated_rms_windows()?;
+        input.validate()?;
+        if input.source_identity() != source.source_identity()
+            || input.numerical_profile() != self.source.program_numerical_profile()
+        {
+            return Err(Error::State("gated RMS input source/profile".into()));
+        }
+        self.gated_rms_cancel()?;
+        let workspace = Arc::new(
+            source
+                .source()
+                .reserve_operator_buffers(INPUT_BYTES, INPUT_SCRATCH)?,
+        );
+        let make = |raw: &[u8]| -> Result<HostValue> {
+            let mut values = Vec::with_capacity(6144);
+            if values.capacity() != 6144 {
+                return Err(Error::Limit("gated RMS input capacity".into()));
+            }
+            values.extend(
+                raw.chunks_exact(2)
+                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)),
+            );
+            HostValue::typed(
+                values,
+                ActivationPrecision::Bf16WidenedF32,
+                source.source_identity(),
+                self.source.program_numerical_profile(),
+            )?
+            .retain_workspace(Some(workspace.clone()))
+        };
+        let [core, gate] = input.raw_inputs();
+        let result = (make(core)?, make(gate)?);
+        self.gated_rms_cancel()?;
+        Ok(result)
+    }
+    pub(super) fn execute_gated_rms_bf16_gamma(
+        &self,
+        parameter: SourceParameter,
+        core: &HostValue,
+        gate: &HostValue,
+        contract: GatedRmsGammaBf16Contract,
+    ) -> Result<HostValue> {
+        let source = self.gated_rms_windows()?;
+        contract.validate(&parameter, core.elements, gate.elements)?;
+        if parameter.source != source.source_identity() {
+            return Err(Error::State("gated RMS parameter source".into()));
+        }
+        for input in [core, gate] {
+            if input.source != Some(source.source_identity())
+                || input.profile != Some(self.source.program_numerical_profile())
+                || input.precision != ActivationPrecision::Bf16WidenedF32
+            {
+                return Err(Error::State(
+                    "gated RMS activation source/profile/precision".into(),
+                ));
+            }
+            let owner = input
+                .data
+                .workspace
+                .as_ref()
+                .ok_or_else(|| Error::State("gated RMS missing physical backing".into()))?;
+            if !source.source().same_issuer(owner.source()) {
+                return Err(Error::State("gated RMS foreign physical issuer".into()));
+            }
+        }
+        if crate::operator_slice::gated_rms::gated_rms_role(source, 0)? != parameter {
+            return Err(Error::State(
+                "gated RMS exact original parameter role".into(),
+            ));
+        }
+        let core = core.as_slice()?;
+        let gate = gate.as_slice()?;
+        ActivationPrecision::Bf16WidenedF32.validate_values(core)?;
+        ActivationPrecision::Bf16WidenedF32.validate_values(gate)?;
+        self.gated_rms_cancel()?;
+        let workspace = Arc::new(
+            source
+                .source()
+                .reserve_operator_buffers(OUTPUT_BYTES, OUTPUT_SCRATCH)?,
+        );
+        let completed = source.gated_rms_bf16_gamma(&parameter, core, gate, self.cancelled)?;
+        let mut values = Vec::with_capacity(6144);
+        if values.capacity() != 6144 || completed.output_bf16().len() != 6144 {
+            return Err(Error::Limit("gated RMS output capacity/length".into()));
+        }
+        values.extend_from_slice(completed.output_bf16());
+        let output = HostValue::typed(
+            values,
+            ActivationPrecision::Bf16WidenedF32,
+            source.source_identity(),
+            self.source.program_numerical_profile(),
+        )?
+        .retain_workspace(Some(workspace))?;
+        self.gated_rms_cancel()?;
+        Ok(output)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -524,6 +524,25 @@
             let _ = (parameter, input, history, contract);
             Err(Error::Unsupported(
                 "causal convolution requires leased original windows".into(),
+            ))
+        }
+    }
+    fn gated_rms_gamma_bf16_profiled(
+        &mut self,
+        parameter: SourceParameter,
+        core: &HostValue,
+        gate: &HostValue,
+        contract: crate::accelerator::program::GatedRmsGammaBf16Contract,
+    ) -> Result<HostValue> {
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        {
+            self.execute_gated_rms_bf16_gamma(parameter, core, gate, contract)
+        }
+        #[cfg(not(all(feature = "original-hf-binding", target_os = "linux")))]
+        {
+            let _ = (parameter, core, gate, contract);
+            Err(Error::Unsupported(
+                "gated RMS requires leased original windows".into(),
             ))
         }
     }
@@ -1053,3 +1072,6 @@
 
 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
 mod gdn_conv;
+
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+mod gated_rms;
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -234,3 +234,6 @@
 
 #[path = "gdn_conv_slice.rs"]
 pub mod gdn_conv;
+
+#[path = "gdn_gated_rms_slice.rs"]
+pub mod gated_rms;
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -382,3 +382,6 @@
 
 #[path = "gdn_conv_window_tests.rs"]
 mod gdn_conv;
+
+#[path = "gated_rms_window_tests.rs"]
+mod gated_rms;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window/gated_rms.rs
@@ -0,0 +1,42 @@
+//! Real raw gamma window on the existing immutable owner, never a learned cache.
+use super::*;
+impl OriginalWindowOperators {
+    pub(crate) fn gated_rms_bf16_gamma(
+        &self,
+        parameter: &SourceParameter,
+        core: &[f32],
+        gate: &[f32],
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<zerogravity_cpu::original_hf::norm::Bf16GammaGatedRms128> {
+        let actual = crate::operator_slice::gated_rms::gated_rms_role(self, 0)?;
+        if actual != *parameter || core.len() != 6144 || gate.len() != 6144 {
+            return Err(Error::State("gated RMS original raw role/geometry".into()));
+        }
+        check_cancel(cancelled)?;
+        let raw = self.source().parameter(&parameter.name)?;
+        raw.validate_source_parameter(parameter)?;
+        let window = match cancelled {
+            Some(flag) => self
+                .source()
+                .read_parameter_cancellable(&raw, 0..128, flag)?,
+            None => self.source().read_parameter(&raw, 0..128)?,
+        };
+        if !window.is_complete() || window.element_range() != (0..128) {
+            return Err(Error::State(
+                "gated RMS incomplete original gamma extent".into(),
+            ));
+        }
+        let completed = zerogravity_cpu::original_hf::norm::gated_rms_128_bf16_gamma_h48(
+            self.source_identity(),
+            zerogravity_core::source::SourceTensorRef {
+                kind: raw.tensor().kind,
+                shape: &raw.tensor().shape,
+                packed_bytes: window.as_slice(),
+            },
+            core,
+            gate,
+        )?;
+        check_cancel(cancelled)?;
+        Ok(completed)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/original_binding/window.rs
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window.rs
@@ -418,3 +418,6 @@
 
 #[cfg(feature = "cpu")]
 mod gdn_conv;
+
+#[cfg(feature = "cpu")]
+mod gated_rms;

````

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/NEW_PREPARATION.patch

Original UTF-8 bytes: 161496; SHA256 `59a8d2fb25aa1016dfbd581d35831e7e8b922d1ec003bb3ca0f5536b1fd30b4b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/Cargo.toml
@@ -0,0 +1,24 @@
+[package]
+name = "original27b-bf16-gamma-gated-rms-replay"
+version = "0.0.0"
+edition = "2024"
+publish = false
+
+[workspace]
+
+[dependencies]
+zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-artifacts", features = ["cache"] }
+zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core" }
+zerogravity-cpu = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu", features = ["cache"] }
+zerogravity-model-qwen35 = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35", features = ["cpu", "original-hf-binding"] }
+serde_json = "=1.0.151"
+sha2 = "=0.10.9"
+libc = "=0.2.190"
+
+[profile.release]
+opt-level = 3
+codegen-units = 1
+
+[[bin]]
+name = "original27b-bf16-gamma-gated-rms-replay"
+path = "main.rs"
--- /dev/null
+++ b/ROOT.md
@@ -0,0 +1,33 @@
+# ROOT-only authentic H48 gated RMS qualification
+
+- Preparation only. Author read small source/JSON metadata, not tensor archives, learned payloads or DSO bytes; no model/provider/operator/tests/build/runtime.
+- Dependency: reviewed16-path43c400a2 /c89eb6c0 candidate +test-only7f266003 /86384ec0 followup. Full HF/session/GDN/initial/exp/GPU guards stay closed.
+- Original teacher11 INITIAL paddedchunk64 and teacher353 TRUECACHED original module calls; both norm inputs are actual BF16 `[48,128]`, original gamma BF16 `[128]`/256B. Exact accepted V3 source/runtime/invocation/manifest/validation/prevalidation pins in source-bindings.json. Metadata codec names are normalized only (`torch.bfloat16`→Safetensors`BF16`); original_torch_dtype retained, no numerical cast.
+
+Scope/counts:
+
+- Same `execute_gated_rms_bf16_gamma_slice`: 2×6144 =12,288 independent final words; 2×(core6144+gate6144)=24,576 retained input words.
+- SAME new CPU helper with SAME original sealed source gamma window: 2×(mean48+inverse48+5×6144)=61,632 words /14 checks. Separate primitive receipts; no alternate fallback/model loop.
+- Both primitive outputs and typed finals are independently compared to untouched actual public archive words. No tolerance or candidate-derived expectation. Original signed zeros/subnormals retained; selected nonfinite stages refuse rather than filter.
+- Source authenticity does not create a live GDN/core/exp producer or certify full27B112StateRoots/session math.
+
+Resources/lifetime:
+
+- Native `ResourceArbiter`128MiB host pool; original factory reserves64MiB source metadata, original binding8MiB. Native8MiB control charge precedes metadata/archive/source reads and selected payload copies. This bounds private admitted payload/offset/control buffers, not OS allocator bookkeeping/page cache/free physical RAM.
+- Before primitive decode/helper arrays: additional1MiB original-source Scratch. Raw gamma staged through existing charged data+offset window and compared exactly to original public raw256B. No fabricated F32gamma ref.
+- Before each typed import/output: original aggregate tickets (input49,152+65,536scratch; result24,576+983,040scratch). Both live outputs retain all6 actual data arrays and original source, long after source/operator callers drop. Temporary diagnostics drop before live ledger checks.
+- Native parses/authenticates each full archive via retained FD and64KiB hash buffer, bounds header before read, checks complete dtypes/shapes/spans/no trailing bytes, and reads at most24,576B per selected stage. Never copies whole47/27MiB archives. Same FD stat/hash rechecked after every selection/fullproof. Python metadata preparation opens no raw archive. Independent raw validator uses bounded same-FD streaming too.
+- Exact steady80MiB; two live typed results80MiB+2,244,608B; retained72MiB+2,244,608B; actual final reserved0/reservations0/no fence. Configured pool is not physical RAM admission.
+- Real pre-cancel/wronglayer and ONE-BYTE-short aggregate pressure checked for each teacher; failures restore exact live ledger. Deterministic after-import cancellation remains the prepared internal model fixture, not claimed as an external native timing test.
+- Exact accepted owned_lifetime.py; owned_native.py differs only in the scope label. Same WNOWAIT/TERM/KILL-before-reap/PGID/pipe ownership,512MiB RLIMIT_AS, four actual CPU affinity IDs,900s child/2400s outer/4MiB log. No Torch/container/GPU/new engine.
+
+ROOT sequence (never author execution):
+
+1. Review implementation +harness +source bindings. Run8source-free fixtures (test_contracts.py) and focused/strict source tests after integration. All original freezes remain immutable.
+2. Copy exact local factory.rs/source-pin.json into this Cargo manifest directory. source-pin.json is accepted7058B4c74fa28; factory is exact existing2302f6a0 copy. It scans all18 original shards through existing admitted window factory; no55GB Vec/mmap.
+3. Compile this standalone Cargo target using coordinator's existing controlled build; optional rustc against freshly built MAIN libraries must preserve the same flags/full linked-source receipt. No downloads. ROOT build-source attestation must list ALL required build paths, exact new children/final repaired test and unchanged numerical sources, plus local Cargo/main/factory/source-pin/source-bindings. Status ROOT_REVIEWED_MERGED_BF16_GAMMA_GATED_RMS_SOURCE; prepared_manifest_sha256; false full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified. Caller attestation is review binding, not independent build proof.
+4. `python3 prepare_reference.py --new-output "$RUN/root-reference-metadata-1"`: metadata only. Actual raw archives are authenticated/extracted in admitted Rust work, not this phase. No independent raw PASS yet.
+5. `python3 run_replay.py --binary "$BINARY" --binary-sha256 "$EXACT_SHA" --reference-directory "$RUN/root-reference-metadata-1" --implementation "$ROOT_BUILD_RECEIPT" --new-output "$RUN/root-native-1"`. Only ROOT runs, after active timing/phase gates.
+6. `python3 validate_replay.py --output "$RUN/root-native-1" --reference-directory "$RUN/root-reference-metadata-1" --implementation "$ROOT_BUILD_RECEIPT"`. Independent stdlib full signed-bit checks/whole archive SHA/source/receipts/lifecycle/ledger/tree. Final native artifact tree<=8MiB, validation receipt reserved64KiB.
+
+Acceptance is only `PASS_PRIVATE_ORIGINAL27B_H48_BF16_GAMMA_GATED_RMS_ONLY`. No full-GDN, A_log/exp, initial chunk/recurrence, canonical/root publication, trained whole model, GPU or performance eligibility.
--- /dev/null
+++ b/contracts.py
@@ -0,0 +1,178 @@
+"""ROOT-only bounded metadata/raw transport; no model/provider imports/calls."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import struct
+
+HERE = Path(__file__).resolve().parent
+INPUTS = ('core_bf16', 'gate_bf16', 'gamma_bf16')
+OUTPUTS = ('mean_f32', 'inverse_f32', 'normalized_bf16', 'weighted_bf16',
+           'silu_f32', 'product_f32', 'output_bf16')
+PROFILE = 'original27B-raw-source-bf16-v1'
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+def digest(raw):
+    return hashlib.sha256(raw).hexdigest()
+
+def sha(path):
+    with Path(path).open('rb') as opened:
+        return hashlib.file_digest(opened, 'sha256').hexdigest()
+
+def unique(pairs):
+    out = {}
+    for key, value in pairs:
+        require(key not in out, 'duplicate JSON key')
+        out[key] = value
+    return out
+
+def identity(value):
+    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
+
+def read(path, cap=2*1024**2, row=None):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as opened:
+        before = os.fstat(opened.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
+        raw = opened.read(cap+1)
+        require(len(raw) == before.st_size and identity(before) == identity(os.fstat(opened.fileno())),
+                'bounded unchanged metadata FD')
+    if row is not None:
+        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'exact metadata bytes/SHA')
+    return raw
+
+def pin(path, row, cap=2*1024**2):
+    read(path, cap, row)
+
+def read_json(path):
+    return json.loads(read(path), object_pairs_hook=unique)
+
+def write(path, value):
+    raw = (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
+    require(len(raw) <= 2*1024**2, 'bounded metadata receipt')
+    with Path(path).open('xb') as opened:
+        opened.write(raw)
+
+def frozen():
+    manifest = read_json(HERE/'prepared-manifest.json')
+    for row in manifest['files']:
+        pin(HERE/row['file'], row)
+        pin(HERE/'frozen-prepared'/row['file'], row)
+    parent = manifest['parent_freeze']
+    pin(parent['path'], parent)
+    root = Path(parent['path']).parent
+    prior = read_json(parent['path'])
+    for row in prior['files']:
+        for directory in ('proposed', 'frozen'):
+            pin(root/directory/row['path'], row)
+    for row in prior['documents']:
+        pin(root/row['file'], row)
+    for row in prior['unchanged_numerical_source']:
+        pin(row['path'], row)
+    followup = manifest['test_style_followup']
+    pin(followup['path'], followup)
+    repaired = read_json(followup['path'])['file']
+    for directory in ('proposed', 'frozen'):
+        pin(Path(followup['path']).parent/directory/repaired['path'], repaired)
+    for row in manifest['unchanged_helper_sources']:
+        pin(row['path'], row)
+    return sha(HERE/'prepared-manifest.json')
+
+def implementation(path):
+    value = read_json(path)
+    require(value['status'] == 'ROOT_REVIEWED_MERGED_BF16_GAMMA_GATED_RMS_SOURCE' and
+            value['prepared_manifest_sha256'] == frozen() and
+            value['full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified'] is False,
+            'mandatory ROOT reviewed merged build-source attestation; not independent build proof')
+    rows = value['files']
+    require(type(rows) is list and 1 <= len(rows) <= 512 and
+            len({r['path'] for r in rows}) == len(rows), 'finite unique actual build source list')
+    by_path = {r['path']: r for r in rows}
+    manifest = read_json(HERE/'prepared-manifest.json')
+    require(set(manifest['required_build_paths']) <= set(by_path), 'complete original/changed build inputs')
+    for row in rows:
+        require(Path(row['path']).is_absolute(), 'actual absolute build input')
+        pin(row['path'], row)
+    for expected in manifest['expected_merged_children']:
+        require(by_path.get(expected['path']) == expected, 'exact reviewed child/numerical implementation')
+    return sha(path)
+
+def finite(raw, dtype):
+    require(dtype in ('BF16', 'F32'), 'closed observed dtype')
+    width, fmt, mask = (2, '<H', 0x7f80) if dtype == 'BF16' else (4, '<I', 0x7f800000)
+    require(raw and len(raw) % width == 0 and
+            all(word & mask != mask for word, in struct.iter_unpack(fmt, raw)),
+            'finite complete observed words; no signed-zero canonicalization')
+
+def widened(raw, dtype):
+    finite(raw, dtype)
+    return b''.join((word << 16).to_bytes(4, 'little') for word, in struct.iter_unpack('<H', raw)) \
+        if dtype == 'BF16' else raw
+
+class RawArchive:
+    """Whole-file streaming authentication; bounded selected reads on SAME FD."""
+    def __init__(self, row):
+        self.row, self.opened = row, None
+    def __enter__(self):
+        self.opened = os.fdopen(os.open(self.row['path'], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb')
+        try:
+            self.before = os.fstat(self.opened.fileno())
+            require(stat.S_ISREG(self.before.st_mode) and self.before.st_size == self.row['bytes'] and
+                    8 < self.before.st_size <= 64*1024**2, 'bounded full raw archive FD')
+            self._authenticate()
+            size, = struct.unpack('<Q', os.pread(self.opened.fileno(), 8, 0))
+            require(1 < size <= 1024**2 and 8+size < self.before.st_size,
+                    'header bound BEFORE reading declared extent')
+            raw = os.pread(self.opened.fileno(), size, 8)
+            require(len(raw) == size, 'complete original header read')
+            self.header = json.loads(raw, object_pairs_hook=unique)
+            require(type(self.header) is dict and 1 <= len(self.header) <= 2048, 'bounded archive fields')
+            self.start, spans = 8+size, []
+            for name, item in self.header.items():
+                if name == '__metadata__':
+                    continue
+                require(item['dtype'] in ('BF16', 'F32') and type(item['shape']) is list and
+                        len(item['shape']) <= 8, 'original rank/dtype')
+                words = 1
+                for axis in item['shape']:
+                    require(type(axis) is int and 0 < axis <= 262144, 'bounded original axis')
+                    words *= axis
+                require(type(item['data_offsets']) is list and len(item['data_offsets']) == 2,
+                        'two original offsets')
+                lo, hi = item['data_offsets']
+                require(type(lo) is type(hi) is int and 0 <= lo < hi <= self.before.st_size-self.start and
+                        hi-lo == words*(2 if item['dtype'] == 'BF16' else 4), 'complete actual archive span')
+                spans.append((lo, hi))
+            end = 0
+            for lo, hi in sorted(spans):
+                require(lo == end, 'no overlapping/gapped actual archive data')
+                end = hi
+            require(self.start+end == self.before.st_size, 'no unaccounted trailing archive bytes')
+            return self
+        except BaseException:
+            self.opened.close()
+            raise
+    def _authenticate(self):
+        self.opened.seek(0)
+        require(hashlib.file_digest(self.opened, 'sha256').hexdigest() == self.row['sha256'] and
+                identity(os.fstat(self.opened.fileno())) == identity(self.before),
+                'complete actual archive digest/FD stable')
+    def raw(self, row):
+        item = self.header[row['name']]
+        require(item['dtype'] == row['dtype'] and item['shape'] == row['shape'], 'actual observed name/dtype/shape')
+        lo, hi = item['data_offsets']
+        require(0 < hi-lo == row['bytes'] <= 24576, 'bounded actual selected stage')
+        raw = os.pread(self.opened.fileno(), hi-lo, self.start+lo)
+        require(len(raw) == hi-lo and digest(raw) == row['raw_sha256'] and
+                identity(os.fstat(self.opened.fileno())) == identity(self.before), 'all raw stage bits and same retained FD')
+        finite(raw, row['dtype'])
+        return raw
+    def __exit__(self, typ, value, trace):
+        try:
+            if typ is None:
+                self._authenticate()
+        finally:
+            self.opened.close()
--- /dev/null
+++ b/factory.rs
@@ -0,0 +1,852 @@
+#![allow(dead_code)]
+//! Root-only actual-source byte/window harness. No model or GPU execution.
+use serde_json::{Value, json};
+use sha2::{Digest, Sha256};
+use std::collections::BTreeMap;
+use std::error::Error;
+use std::fs::{File, OpenOptions};
+use std::io::Read;
+use std::ops::Range;
+use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
+use std::path::{Path, PathBuf};
+use std::sync::atomic::AtomicBool;
+use std::time::Instant;
+use zerogravity_artifacts::Error as ZgError;
+use zerogravity_artifacts::artifact::Artifact;
+use zerogravity_artifacts::cache::TransferAdmission;
+use zerogravity_artifacts::source_weights::{
+    ExpectedSourceShard, SourceWeights, VerifiedMatrix, VerifiedTensorSource, VerifiedWindowPlan,
+    WindowOpenLimits,
+};
+use zerogravity_core::accelerator::program::MatrixAxes;
+use zerogravity_core::quant::QuantKind;
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
+};
+
+type Proof<T> = Result<T, Box<dyn Error>>;
+const SNAPSHOT: &str = "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e";
+const REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
+const REPORT_SHA: &str = "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a";
+const INDEX_SHA: &str = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df";
+const BUNDLE_SHA: &str = "7e6a80bc4fc06b6237914007bde59a48a6e4fb0ea570bd5078b3101eb169faf2";
+const SELECTED_BYTES: u64 = 55_586_113_293;
+const WEIGHT_BYTES: u64 = 55_563_006_776;
+const PAYLOAD_BYTES: u64 = 55_562_855_904;
+const META: u64 = 64 * 1024 * 1024;
+const HOST: u64 = 128 * 1024 * 1024;
+const SCAN: usize = 4 * 1024 * 1024;
+const SOURCE_DIR: &str = "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files";
+
+struct Shard {
+    name: String,
+    expected_bytes: u64,
+    expected_sha: String,
+    // Both Artifact and independent raw reference derive from this one open FD.
+    file: File,
+}
+
+fn ensure(condition: bool, message: impl Into<String>) -> Proof<()> {
+    if !condition {
+        let message: String = message.into();
+        return Err(message.into());
+    }
+    Ok(())
+}
+
+fn sha(bytes: &[u8]) -> String {
+    format!("{:x}", Sha256::digest(bytes))
+}
+
+fn bounded_file(path: &Path, maximum: usize) -> Proof<Vec<u8>> {
+    let file = OpenOptions::new()
+        .read(true)
+        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+        .open(path)?;
+    ensure(file.metadata()?.is_file(), "control source is not regular")?;
+    ensure(
+        file.metadata()?.len() <= maximum as u64,
+        "control source byte cap",
+    )?;
+    let mut bytes = Vec::new();
+    file.take(maximum as u64 + 1).read_to_end(&mut bytes)?;
+    ensure(bytes.len() <= maximum, "control source grew beyond cap")?;
+    Ok(bytes)
+}
+
+fn reserved(host: &TransferAdmission) -> Proof<u64> {
+    Ok(host.arbiter.snapshot(&host.resource)?.reserved)
+}
+
+fn ledger(host: &TransferAdmission, metadata: u64, staging: u64, other: u64) -> Proof<()> {
+    let current = host.arbiter.snapshot(&host.resource)?;
+    ensure(!current.fenced, "host window pool unexpectedly fenced")?;
+    ensure(current.limit == HOST, "configured pool limit changed")?;
+    ensure(
+        current.reserved == metadata + staging + other,
+        "live total ledger mismatch",
+    )?;
+    for (class, expected) in [
+        (AllocationClass::Scratch, metadata),
+        (AllocationClass::TransferStaging, staging),
+        (AllocationClass::Other, other),
+    ] {
+        ensure(
+            current.per_class.get(&class).copied().unwrap_or(0) == expected,
+            format!("live allocation-class mismatch: {class:?}"),
+        )?;
+    }
+    ensure(
+        current.remaining + current.reserved == HOST,
+        "remaining ledger mismatch",
+    )?;
+    Ok(())
+}
+
+fn limits() -> WindowOpenLimits {
+    WindowOpenLimits {
+        max_source_bytes: 64 * 1024 * 1024 * 1024,
+        max_metadata_bytes: META,
+        max_shards: 18,
+        max_tensors: 1199,
+        scan_chunk_bytes: SCAN,
+    }
+}
+
+fn expected_shards(shards: &[Shard]) -> Proof<Vec<ExpectedSourceShard>> {
+    shards
+        .iter()
+        .map(|shard| {
+            let clone = shard.file.try_clone()?;
+            let original = shard.file.metadata()?;
+            let same = clone.metadata()?;
+            ensure(
+                (same.dev(), same.ino(), same.len())
+                    == (original.dev(), original.ino(), original.len()),
+                "descriptor clone changed source",
+            )?;
+            Ok(ExpectedSourceShard::new(
+                shard.name.clone(),
+                Artifact::from_file(clone, Path::new(&shard.name))?,
+                shard.expected_bytes,
+                shard.expected_sha.clone(),
+            )?)
+        })
+        .collect()
+}
+
+fn plan(index: &[u8], shards: &[Shard], host: &TransferAdmission) -> Proof<VerifiedWindowPlan> {
+    Ok(SourceWeights::prepare_verified_windows(
+        index,
+        INDEX_SHA,
+        expected_shards(shards)?,
+        limits(),
+        host,
+    )?)
+}
+
+fn expected_limit<T>(result: zerogravity_artifacts::Result<T>, label: &str) -> Proof<String> {
+    match result {
+        Err(ZgError::Limit(message)) => Ok(message),
+        Err(error) => Err(format!("{label}: wrong refusal: {error}").into()),
+        Ok(value) => {
+            drop(value);
+            Err(format!("{label}: operation unexpectedly succeeded").into())
+        }
+    }
+}
+
+fn expected_cancel<T>(result: zerogravity_artifacts::Result<T>, label: &str) -> Proof<String> {
+    match result {
+        Err(ZgError::State(message)) if message.contains("cancelled") => Ok(message),
+        Err(error) => Err(format!("{label}: wrong refusal: {error}").into()),
+        Ok(value) => {
+            drop(value);
+            Err(format!("{label}: operation unexpectedly succeeded").into())
+        }
+    }
+}
+
+fn guards(index: &[u8], shards: &[Shard], host: &TransferAdmission) -> Proof<Value> {
+    let mut small_metadata = limits();
+    small_metadata.max_metadata_bytes = 1;
+    let metadata = expected_limit(
+        SourceWeights::prepare_verified_windows(
+            index,
+            INDEX_SHA,
+            expected_shards(shards)?,
+            small_metadata,
+            host,
+        ),
+        "metadata envelope",
+    )?;
+    ledger(host, 0, 0, 0)?;
+
+    let mut source_limit = limits();
+    source_limit.max_source_bytes = WEIGHT_BYTES - 1;
+    let source = expected_limit(
+        SourceWeights::prepare_verified_windows(
+            index,
+            INDEX_SHA,
+            expected_shards(shards)?,
+            source_limit,
+            host,
+        ),
+        "whole-source file bound",
+    )?;
+    ledger(host, 0, 0, 0)?;
+
+    let held_bytes = HOST - META + 1;
+    let held = host.arbiter.admit(
+        &host.resource,
+        host.owner.clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Other,
+            bytes: held_bytes,
+        }],
+    )?;
+    let unavailable = expected_limit(
+        SourceWeights::prepare_verified_windows(
+            index,
+            INDEX_SHA,
+            expected_shards(shards)?,
+            limits(),
+            host,
+        ),
+        "metadata live-ledger admission",
+    )?;
+    ledger(host, 0, 0, held_bytes)?;
+    drop(held);
+    ledger(host, 0, 0, 0)?;
+
+    let pending = plan(index, shards, host)?;
+    ledger(host, META, 0, 0)?;
+    let cancelled = expected_cancel(
+        pending.open_cancellable(&AtomicBool::new(true)),
+        "pre-cancelled factory open",
+    )?;
+    ledger(host, 0, 0, 0)?;
+    Ok(json!({
+        "metadata_envelope_refused":metadata,
+        "source_file_bound_refused":source,
+        "metadata_live_ledger_refused":unavailable,
+        "pre_cancelled_factory_refused":cancelled,
+        "scope":"real headers inspected; factory pre-cancel before whole payload hash scan; no source published",
+        "post_guard_reserved_bytes":0,
+    }))
+}
+
+fn expected_staging(shard_bytes: u64, offset: u64, bytes: usize) -> Proof<u64> {
+    let end = offset
+        .checked_add(bytes as u64)
+        .ok_or("reference range overflow")?;
+    ensure(
+        bytes != 0 && end <= shard_bytes,
+        "reference range outside source",
+    )?;
+    let first = offset / SCAN as u64;
+    let last = (end - 1) / SCAN as u64;
+    let staged = ((last + 1) * SCAN as u64).min(shard_bytes) - first * SCAN as u64;
+    let offsets = (last - first + 1) * std::mem::size_of::<(usize, Range<usize>)>() as u64;
+    Ok(staged + offsets)
+}
+
+fn compare_rows(
+    source: &VerifiedTensorSource,
+    matrix: &VerifiedMatrix,
+    raw: &Shard,
+    rows: Range<usize>,
+    label: &str,
+    host: &TransferAdmission,
+) -> Proof<Value> {
+    ledger(host, META, 0, 0)?;
+    let offset = matrix.tensor().byte_offset + rows.start as u64 * matrix.columns() as u64 * 2;
+    let bytes = (rows.end - rows.start) * matrix.columns() * 2;
+    ensure(bytes <= 256 * 1024, "raw reference buffer cap")?;
+    let window = source.read_rows(matrix, rows.clone())?;
+    let charge = expected_staging(raw.expected_bytes, offset, bytes)?;
+    ensure(
+        window.charged_staging_bytes() as u64 == charge,
+        "exact staged payload+offset charge",
+    )?;
+    ledger(host, META, charge, 0)?;
+    ensure(
+        window.row_range() == rows && window.source_offset() == offset,
+        "window row/offset mismatch",
+    )?;
+    ensure(
+        window.as_slice().len() == bytes,
+        "window omitted full-width source row",
+    )?;
+    ensure(
+        window.matrix().source_identity() == source.source_identity(),
+        "window source identity changed",
+    )?;
+    ensure(
+        window.verified_bundle_sha256() == BUNDLE_SHA,
+        "window bundle changed",
+    )?;
+    ensure(
+        window.shard_basename() == raw.name && window.shard_sha256() == raw.expected_sha,
+        "window original shard binding changed",
+    )?;
+    ensure(
+        window.extent_index_sha256().len() == 64,
+        "extent receipt absent",
+    )?;
+
+    // The harness's independent comparison copy has a separate, explicit charge.
+    let reference_lease = host.arbiter.admit(
+        &host.resource,
+        host.owner.clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Other,
+            bytes: bytes as u64,
+        }],
+    )?;
+    let mut reference = vec![0u8; bytes];
+    raw.file.read_exact_at(&mut reference, offset)?;
+    ledger(host, META, charge, bytes as u64)?;
+    ensure(
+        reference.as_slice() == window.as_slice(),
+        "actual source row byte mismatch",
+    )?;
+    let record = json!({
+        "matrix":matrix.tensor().name,
+        "case":label,
+        "rows":[rows.start,rows.end],
+        "shape":[matrix.rows(),matrix.columns()],
+        "dtype":"BF16",
+        "axes":"OutputInput",
+        "source_identity":format!("{:?}",source.source_identity()),
+        "shard":raw.name,
+        "shard_sha256":raw.expected_sha,
+        "source_offset":offset,
+        "requested_bytes":bytes,
+        "charged_staging_bytes":charge,
+        "crosses_scan_extent":offset / SCAN as u64 != (offset + bytes as u64 - 1) / SCAN as u64,
+        "byte_exact":true,
+        "raw_and_window_sha256":sha(&reference),
+        "extent_index_sha256":window.extent_index_sha256(),
+    });
+    drop(reference);
+    drop(reference_lease);
+    ledger(host, META, charge, 0)?;
+    drop(window);
+    ledger(host, META, 0, 0)?;
+    Ok(record)
+}
+
+fn exercise(
+    source: VerifiedTensorSource,
+    shards: &[Shard],
+    host: &TransferAdmission,
+) -> Proof<Value> {
+    ensure(
+        source.source_identity().is_bound(),
+        "factory did not mint bound source identity",
+    )?;
+    let identity = source.source_identity();
+    ensure(
+        source.source_bytes() == WEIGHT_BYTES,
+        "whole weight byte count",
+    )?;
+    ensure(
+        source.metadata_reserved_bytes() == META,
+        "source metadata charge",
+    )?;
+    ensure(
+        source.verified_shard_index_sha256() == INDEX_SHA,
+        "source index pin",
+    )?;
+    ensure(
+        source.verified_bundle_sha256() == BUNDLE_SHA,
+        "source bundle pin",
+    )?;
+    ledger(host, META, 0, 0)?;
+    let clone = source.clone();
+    ensure(
+        clone.source_identity() == identity,
+        "source clone minted another identity",
+    )?;
+    drop(clone);
+
+    let specimens = [
+        (
+            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
+            10_240,
+            5_120,
+            1_120_792,
+            "model-00001-of-00018.safetensors",
+        ),
+        (
+            "model.language_model.embed_tokens.weight",
+            248_320,
+            5_120,
+            152,
+            "model-00003-of-00018.safetensors",
+        ),
+        (
+            "model.language_model.layers.63.mlp.down_proj.weight",
+            5_120,
+            17_408,
+            1_364_259_120,
+            "model-00017-of-00018.safetensors",
+        ),
+        (
+            "lm_head.weight",
+            248_320,
+            5_120,
+            1_760,
+            "model-00018-of-00018.safetensors",
+        ),
+    ];
+    let mut matrices = Vec::new();
+    let mut samples = Vec::new();
+    for (name, rows, columns, absolute_offset, basename) in specimens {
+        let matrix = source.matrix(name, MatrixAxes::OutputInput)?;
+        ensure(
+            matrix.source_identity() == identity,
+            "matrix changed source identity",
+        )?;
+        ensure(
+            matrix.axes() == MatrixAxes::OutputInput,
+            "matrix axes changed",
+        )?;
+        ensure(
+            (matrix.rows(), matrix.columns()) == (rows, columns),
+            "actual matrix shape differs from pinned header",
+        )?;
+        ensure(
+            matrix.tensor().shape == [rows as u64, columns as u64],
+            "raw source axes mismatch",
+        )?;
+        ensure(
+            matrix.tensor().kind == QuantKind::Bf16,
+            "raw source is not BF16",
+        )?;
+        ensure(
+            matrix.tensor().byte_len == (rows * columns * 2) as u64,
+            "raw source byte geometry",
+        )?;
+        ensure(
+            matrix.tensor().byte_offset == absolute_offset,
+            "actual source offset differs from independently inspected original header",
+        )?;
+        let raw = shards
+            .iter()
+            .find(|shard| shard.name == basename)
+            .ok_or("missing raw source FD")?;
+        samples.push(compare_rows(
+            &source,
+            &matrix,
+            raw,
+            0..1,
+            "first_full_width_row",
+            host,
+        )?);
+        samples.push(compare_rows(
+            &source,
+            &matrix,
+            raw,
+            rows - 1..rows,
+            "last_full_width_row",
+            host,
+        )?);
+        let start = matrix.tensor().byte_offset;
+        let boundary = (start / SCAN as u64 + 1) * SCAN as u64;
+        let split_row = ((boundary - start) / (columns * 2) as u64) as usize;
+        let split = split_row.saturating_sub(1)..(split_row + 1).min(rows);
+        let requested_start = start + split.start as u64 * (columns * 2) as u64;
+        let requested_end = start + split.end as u64 * (columns * 2) as u64;
+        ensure(
+            requested_start < boundary && requested_end > boundary,
+            "fixture does not straddle actual scan boundary",
+        )?;
+        samples.push(compare_rows(
+            &source,
+            &matrix,
+            raw,
+            split,
+            "scan_extent_split_full_width_rows",
+            host,
+        )?);
+        matrices.push(matrix);
+    }
+
+    let matrix = &matrices[0];
+    let cancelled = expected_cancel(
+        source.read_rows_cancellable(matrix, 0..1, &AtomicBool::new(true)),
+        "pre-cancelled row",
+    )?;
+    ledger(host, META, 0, 0)?;
+    ensure(
+        source
+            .matrix(&matrix.tensor().name, MatrixAxes::InputOutput)
+            .is_err(),
+        "wrong axes admitted",
+    )?;
+    ensure(
+        source
+            .matrix(
+                "model.language_model.layers.0.input_layernorm.weight",
+                MatrixAxes::OutputInput,
+            )
+            .is_err(),
+        "actual rank-one parameter admitted as matrix",
+    )?;
+    ensure(
+        source
+            .matrix("__not_in_the_original_index__", MatrixAxes::OutputInput)
+            .is_err(),
+        "unknown matrix admitted",
+    )?;
+    for rows in [
+        0..0,
+        matrix.rows()..matrix.rows() + 1,
+        usize::MAX..usize::MAX,
+    ] {
+        ensure(
+            source.read_rows(matrix, rows).is_err(),
+            "invalid row range admitted",
+        )?;
+        ledger(host, META, 0, 0)?;
+    }
+    let oversized = expected_limit(
+        source.read_rows(&matrices[1], 0..matrices[1].rows()),
+        "whole embedding tensor materialization",
+    )?;
+    ledger(host, META, 0, 0)?;
+
+    let pin = matrices[0].clone();
+    let last = &matrices[3];
+    let retained = source.read_rows(last, last.rows() - 1..last.rows())?;
+    let retained_sha = sha(retained.as_slice());
+    let charge = retained.charged_staging_bytes() as u64;
+    ledger(host, META, charge, 0)?;
+    drop(matrices);
+    drop(source);
+    ensure(
+        pin.source_identity() == identity && retained.matrix().source_identity() == identity,
+        "descriptor/window lost original source identity",
+    )?;
+    ledger(host, META, charge, 0)?;
+    drop(pin);
+    // Only the owned immutable window retains source metadata and its live FD set.
+    ledger(host, META, charge, 0)?;
+    ensure(
+        sha(retained.as_slice()) == retained_sha,
+        "window bytes changed after all caller source handles dropped",
+    )?;
+    drop(retained);
+    ledger(host, 0, 0, 0)?;
+    Ok(json!({
+        "source_identity":format!("{identity:?}"),
+        "source_bytes":WEIGHT_BYTES,
+        "source_bundle_sha256":BUNDLE_SHA,
+        "index_sha256":INDEX_SHA,
+        "matrix_count":4,
+        "samples":samples,
+        "pre_cancelled_row_refused":cancelled,
+        "wrong_axes_missing_name_rank_one_bad_rows_refused":true,
+        "whole_tensor_materialization_refused":oversized,
+        "only_window_retains_source_metadata_and_stage_after_all_caller_sources_drop":true,
+        "window_after_source_drop_sha256":retained_sha,
+        "final_reserved_bytes":0,
+    }))
+}
+
+fn run(host: &TransferAdmission) -> Proof<Value> {
+    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
+    let report_bytes = bounded_file(&root.join("source-pin.json"), 64 * 1024)?;
+    ensure(
+        sha(&report_bytes) == REPORT_SHA,
+        "acquired source report pin mismatch",
+    )?;
+    let report: Value = serde_json::from_slice(&report_bytes)?;
+    ensure(
+        report["snapshot"] == SNAPSHOT
+            && report["repository"] == "Qwen/Qwen3.8-27B"
+            && report["revision"] == REVISION
+            && report["reference_directory"] == SOURCE_DIR,
+        "acquired snapshot provenance differs",
+    )?;
+    ensure(
+        report["selected_bytes"] == SELECTED_BYTES
+            && report["weight_bytes"] == WEIGHT_BYTES
+            && report["payload_bytes"] == PAYLOAD_BYTES
+            && report["tensor_count"] == 1199,
+        "acquired byte/count pin differs",
+    )?;
+    let files = report["files"]
+        .as_array()
+        .ok_or("source report files absent")?;
+    ensure(files.len() == 31, "selected source file count")?;
+    let mut pinned = BTreeMap::new();
+    let mut total = 0u64;
+    for entry in files {
+        let name = entry["name"].as_str().ok_or("entry name absent")?;
+        let bytes = entry["bytes"].as_u64().ok_or("entry bytes absent")?;
+        let hash = entry["sha256"].as_str().ok_or("entry hash absent")?;
+        total = total
+            .checked_add(bytes)
+            .ok_or("selected byte sum overflow")?;
+        ensure(
+            pinned
+                .insert(name.to_string(), (bytes, hash.to_string()))
+                .is_none(),
+            "duplicate source report entry",
+        )?;
+    }
+    ensure(
+        total == SELECTED_BYTES,
+        "selected report bytes do not close",
+    )?;
+    let index = bounded_file(
+        &Path::new(SOURCE_DIR).join("model.safetensors.index.json"),
+        1024 * 1024,
+    )?;
+    ensure(
+        sha(&index) == INDEX_SHA,
+        "actual original index SHA256 mismatch",
+    )?;
+    let index_info = zerogravity_artifacts::sharded::SafetensorsIndex::from_json(&index)?;
+    ensure(
+        index_info.declared_total_size() == PAYLOAD_BYTES
+            && index_info.tensors().len() == 1199
+            && index_info.shards().len() == 18,
+        "actual original index closure differs",
+    )?;
+    let mut shards = Vec::with_capacity(18);
+    let mut descriptors = Vec::new();
+    for number in 1..=18 {
+        let name = format!("model-{number:05}-of-00018.safetensors");
+        let (bytes, hash) = pinned.get(&name).ok_or("pinned shard absent")?;
+        ensure(
+            index_info.shards().contains(name.as_str()),
+            "pin has extra shard",
+        )?;
+        let file = OpenOptions::new()
+            .read(true)
+            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+            .open(Path::new(SOURCE_DIR).join(&name))?;
+        let stat = file.metadata()?;
+        ensure(
+            stat.is_file() && stat.len() == *bytes,
+            "opened original shard type/length differs",
+        )?;
+        descriptors.push(json!({"shard":name,"dev":stat.dev(),"ino":stat.ino(),"bytes":stat.len(),"sha256":hash}));
+        shards.push(Shard {
+            name,
+            expected_bytes: *bytes,
+            expected_sha: hash.clone(),
+            file,
+        });
+    }
+    ensure(
+        shards.iter().map(|shard| shard.expected_bytes).sum::<u64>() == WEIGHT_BYTES,
+        "whole original weight-file byte sum differs",
+    )?;
+    eprintln!("phase=real_header_only_guards source_weight_bytes={WEIGHT_BYTES}");
+    let refusals = guards(&index, &shards, host)?;
+    let pending = plan(&index, &shards, host)?;
+    ensure(
+        pending.source_bytes() == WEIGHT_BYTES && pending.metadata_reserved_bytes() == META,
+        "prepared whole-source bounds differ",
+    )?;
+    ledger(host, META, 0, 0)?;
+    eprintln!(
+        "phase=whole_source_sha256_scan_start shard_count=18 scan_buffer_bytes={SCAN} metadata_reserved_bytes={META}"
+    );
+    let started = Instant::now();
+    let source = pending.open()?;
+    let scan_milliseconds = started.elapsed().as_millis();
+    eprintln!("phase=whole_source_sha256_scan_ready elapsed_ms={scan_milliseconds}");
+    let windows = exercise(source, &shards, host)?;
+    ledger(host, 0, 0, 0)?;
+    Ok(json!({
+        "schema":"zerogravity.original-owned-window-qualification.v1",
+        "status":"PASS_SOURCE_WINDOWS_ONLY",
+        "repository":"Qwen/Qwen3.8-27B",
+        "revision":REVISION,
+        "snapshot":SNAPSHOT,
+        "selected_artifact_bytes":SELECTED_BYTES,
+        "weight_file_bytes_scanned":WEIGHT_BYTES,
+        "tensor_payload_bytes":PAYLOAD_BYTES,
+        "whole_source_sha256_verified":true,
+        "retained_raw_reference_fds":descriptors,
+        "scan_chunk_bytes":SCAN,
+        "whole_source_scan_milliseconds":scan_milliseconds,
+        "configured_metadata_reserved_bytes":META,
+        "configured_host_pool_limit_bytes":HOST,
+        "refusals":refusals,
+        "windows":windows,
+        "physical_host_ram_admission_qualified":false,
+        "trained_math_or_model_inference_qualified":false,
+        "ssd_gpu_streaming_inference_qualified":false,
+        "gpu_or_state_root_publication_performed":false,
+        "scope":"one source issuer; whole weight SHA/index/header closure; owned bounded raw BF16 row bytes; configured host ledger only",
+    }))
+}
+
+fn main() {
+    let arbiter = ResourceArbiter::process_shared();
+    let resource = PhysicalDeviceKey::new("host", "explicit-owned-window-qualification-pool")
+        .expect("fixed resource label");
+    arbiter
+        .register_matching(resource.clone(), HOST)
+        .expect("fresh explicit pool registration");
+    let host = TransferAdmission {
+        arbiter,
+        resource,
+        owner: ResourceOwner::new(
+            "Qwen/Qwen3.8-27B",
+            "actual-original-source-window-qualification",
+        )
+        .expect("fixed source owner"),
+    };
+    let result = run(&host);
+    let final_state = host.arbiter.snapshot(&host.resource);
+    let clean = matches!(&final_state, Ok(state) if state.reserved == 0 && state.reservations == 0 && !state.fenced);
+    let success = result.is_ok() && clean;
+    let mut report = match result {
+        Ok(report) => report,
+        Err(error) => json!({"status":"FAIL_UNQUALIFIED","error":error.to_string()}),
+    };
+    report["final_owned_ledger_clean"] = json!(clean);
+    report["final_owned_ledger_reserved_bytes"] =
+        json!(final_state.as_ref().ok().map(|state| state.reserved));
+    if !clean {
+        report["status"] = json!("FAIL_UNQUALIFIED");
+        report["ledger_cleanup_error"] = json!(format!("{final_state:?}"));
+    }
+    println!(
+        "{}",
+        serde_json::to_string_pretty(&report).expect("bounded report serialization")
+    );
+    if !success {
+        std::process::exit(2);
+    }
+}
+
+/// Only this additive wrapper is called by the new replay. Original factory
+/// source below remains retained verbatim; its old main/exercise are not called.
+pub struct SourceReady {
+    pub source: VerifiedTensorSource,
+    pub config: Vec<u8>,
+    pub descriptors: Vec<Value>,
+    pub scan_milliseconds: u128,
+}
+pub fn open_source(host: &TransferAdmission) -> Proof<SourceReady> {
+    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
+    let report_bytes = bounded_file(&root.join("source-pin.json"), 64 * 1024)?;
+    ensure(
+        sha(&report_bytes) == REPORT_SHA,
+        "acquired source report pin mismatch",
+    )?;
+    let report: Value = serde_json::from_slice(&report_bytes)?;
+    ensure(
+        report["snapshot"] == SNAPSHOT
+            && report["repository"] == "Qwen/Qwen3.8-27B"
+            && report["revision"] == REVISION
+            && report["reference_directory"] == SOURCE_DIR,
+        "acquired snapshot provenance differs",
+    )?;
+    ensure(
+        report["selected_bytes"] == SELECTED_BYTES
+            && report["weight_bytes"] == WEIGHT_BYTES
+            && report["payload_bytes"] == PAYLOAD_BYTES
+            && report["tensor_count"] == 1199,
+        "acquired byte/count pin differs",
+    )?;
+    let files = report["files"]
+        .as_array()
+        .ok_or("source report files absent")?;
+    ensure(files.len() == 31, "selected source file count")?;
+    let mut pinned = BTreeMap::new();
+    let mut total = 0u64;
+    for entry in files {
+        let name = entry["name"].as_str().ok_or("entry name absent")?;
+        let bytes = entry["bytes"].as_u64().ok_or("entry bytes absent")?;
+        let hash = entry["sha256"].as_str().ok_or("entry hash absent")?;
+        total = total
+            .checked_add(bytes)
+            .ok_or("selected byte sum overflow")?;
+        ensure(
+            pinned
+                .insert(name.to_string(), (bytes, hash.to_string()))
+                .is_none(),
+            "duplicate source report entry",
+        )?;
+    }
+    ensure(
+        total == SELECTED_BYTES,
+        "selected report bytes do not close",
+    )?;
+    let index = bounded_file(
+        &Path::new(SOURCE_DIR).join("model.safetensors.index.json"),
+        1024 * 1024,
+    )?;
+    ensure(
+        sha(&index) == INDEX_SHA,
+        "actual original index SHA256 mismatch",
+    )?;
+    let index_info = zerogravity_artifacts::sharded::SafetensorsIndex::from_json(&index)?;
+    ensure(
+        index_info.declared_total_size() == PAYLOAD_BYTES
+            && index_info.tensors().len() == 1199
+            && index_info.shards().len() == 18,
+        "actual original index closure differs",
+    )?;
+    let mut shards = Vec::with_capacity(18);
+    let mut descriptors = Vec::new();
+    for number in 1..=18 {
+        let name = format!("model-{number:05}-of-00018.safetensors");
+        let (bytes, hash) = pinned.get(&name).ok_or("pinned shard absent")?;
+        ensure(
+            index_info.shards().contains(name.as_str()),
+            "pin has extra shard",
+        )?;
+        let file = OpenOptions::new()
+            .read(true)
+            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+            .open(Path::new(SOURCE_DIR).join(&name))?;
+        let stat = file.metadata()?;
+        ensure(
+            stat.is_file() && stat.len() == *bytes,
+            "opened original shard type/length differs",
+        )?;
+        descriptors.push(json!({"shard":name,"dev":stat.dev(),"ino":stat.ino(),"bytes":stat.len(),"sha256":hash}));
+        shards.push(Shard {
+            name,
+            expected_bytes: *bytes,
+            expected_sha: hash.clone(),
+            file,
+        });
+    }
+    ensure(
+        shards.iter().map(|shard| shard.expected_bytes).sum::<u64>() == WEIGHT_BYTES,
+        "whole original weight-file byte sum differs",
+    )?;
+    let config = bounded_file(&Path::new(SOURCE_DIR).join("config.json"), 64 * 1024)?;
+    ensure(
+        sha(&config) == "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
+        "original config SHA",
+    )?;
+    let pending = plan(&index, &shards, host)?;
+    let started = Instant::now();
+    let source = pending.open()?;
+    ensure(
+        source.source_bytes() == WEIGHT_BYTES
+            && source.metadata_reserved_bytes() == META
+            && source.verified_shard_index_sha256() == INDEX_SHA
+            && source.verified_bundle_sha256() == BUNDLE_SHA,
+        "source factory closure",
+    )?;
+    Ok(SourceReady {
+        source,
+        config,
+        descriptors,
+        scan_milliseconds: started.elapsed().as_millis(),
+    })
+}
--- /dev/null
+++ b/main.rs
@@ -0,0 +1,690 @@
+//! ROOT ONLY: original H48 raw-gamma staged operator, no model/session producer.
+mod factory;
+use serde_json::{Value, json};
+use sha2::{Digest, Sha256};
+use std::collections::BTreeMap;
+use std::error::Error;
+use std::fs::{File, OpenOptions};
+use std::io::{Read, Write};
+use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
+use std::path::{Component, Path};
+use std::sync::atomic::AtomicBool;
+use zerogravity_artifacts::cache::TransferAdmission;
+use zerogravity_core::accelerator::program::ProgramNumericalProfile;
+use zerogravity_core::quant::QuantKind;
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, DeviceBudgetSnapshot, PhysicalDeviceKey, ResourceArbiter,
+    ResourceOwner,
+};
+use zerogravity_model_qwen35::operator_slice::gated_rms::{
+    GatedRmsSliceInput, GatedRmsSliceOutput,
+};
+use zerogravity_model_qwen35::original_binding::{OriginalWindowOperators, QwenNumericalProfile};
+
+type Proof<T> = Result<T, Box<dyn Error>>;
+const MIB: u64 = 1024 * 1024;
+const HOST: u64 = 128 * MIB;
+const CONTROL: u64 = 8 * MIB;
+const SOURCE_META: u64 = 64 * MIB;
+const DIAGNOSTIC: usize = 1024 * 1024;
+const INPUT: u64 = 2 * 6144 * 4 + 64 * 1024;
+const OUTPUT: u64 = 6144 * 4 + 960 * 1024;
+const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
+const STAGES: [&str; 7] = [
+    "mean_f32",
+    "inverse_f32",
+    "normalized_bf16",
+    "weighted_bf16",
+    "silu_f32",
+    "product_f32",
+    "output_bf16",
+];
+fn ensure(ok: bool, message: impl Into<String>) -> Proof<()> {
+    if ok {
+        Ok(())
+    } else {
+        Err(message.into().into())
+    }
+}
+fn text(v: &Value) -> Proof<&str> {
+    v.as_str().ok_or_else(|| "required string".into())
+}
+fn sha(raw: &[u8]) -> String {
+    format!("{:x}", Sha256::digest(raw))
+}
+fn stable(file: &File) -> Proof<[u64; 7]> {
+    let m = file.metadata()?;
+    ensure(m.is_file(), "regular retained input FD")?;
+    Ok([
+        m.dev(),
+        m.ino(),
+        m.len(),
+        m.mtime() as u64,
+        m.mtime_nsec() as u64,
+        m.ctime() as u64,
+        m.ctime_nsec() as u64,
+    ])
+}
+fn open(path: &Path) -> Proof<File> {
+    Ok(OpenOptions::new()
+        .read(true)
+        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+        .open(path)?)
+}
+fn read(path: &Path, cap: usize) -> Proof<Vec<u8>> {
+    let f = open(path)?;
+    let before = stable(&f)?;
+    ensure(
+        before[2] > 0 && before[2] <= cap as u64,
+        "bounded regular metadata",
+    )?;
+    let mut raw = Vec::with_capacity(before[2] as usize);
+    f.try_clone()?.take(cap as u64 + 1).read_to_end(&mut raw)?;
+    ensure(
+        raw.len() as u64 == before[2] && stable(&f)? == before,
+        "metadata FD unchanged",
+    )?;
+    Ok(raw)
+}
+fn file_sha(file: &File) -> Proof<String> {
+    let before = stable(file)?;
+    let mut sum = Sha256::new();
+    let mut buf = [0u8; 65536];
+    let mut offset = 0u64;
+    while offset < before[2] {
+        let n = ((before[2] - offset) as usize).min(buf.len());
+        file.read_exact_at(&mut buf[..n], offset)?;
+        sum.update(&buf[..n]);
+        offset += n as u64;
+    }
+    ensure(stable(file)? == before, "full retained archive FD stable")?;
+    Ok(format!("{:x}", sum.finalize()))
+}
+fn relative(v: &Value) -> Proof<&Path> {
+    let p = Path::new(text(v)?);
+    ensure(
+        !p.as_os_str().is_empty() && p.components().all(|p| matches!(p, Component::Normal(_))),
+        "contained output name",
+    )?;
+    Ok(p)
+}
+fn write_json(path: &Path, value: &Value) -> Proof<()> {
+    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
+    serde_json::to_writer_pretty(&mut f, value)?;
+    f.write_all(b"\n")?;
+    Ok(())
+}
+fn snapshot(host: &TransferAdmission) -> Proof<DeviceBudgetSnapshot> {
+    Ok(host.arbiter.snapshot(&host.resource)?)
+}
+fn class(row: &DeviceBudgetSnapshot, kind: AllocationClass) -> u64 {
+    row.per_class.get(&kind).copied().unwrap_or(0)
+}
+fn ledger(row: &DeviceBudgetSnapshot) -> Value {
+    json!({"reserved":row.reserved,"reservations":row.reservations,"fenced":row.fenced,"per_class":row.per_class.iter().map(|(k,v)|(format!("{k:?}"),*v)).collect::<BTreeMap<_,_>>()})
+}
+fn same_ledger(host: &TransferAdmission, before: &DeviceBudgetSnapshot) -> Proof<()> {
+    let after = snapshot(host)?;
+    ensure(
+        after.reserved == before.reserved
+            && after.reservations == before.reservations
+            && after.per_class == before.per_class
+            && !after.fenced,
+        "failed call retires all private leases",
+    )
+}
+struct Archive {
+    file: File,
+    pin: Value,
+    identity: [u64; 7],
+    header: Value,
+    start: u64,
+}
+impl Archive {
+    fn new(pin: &Value) -> Proof<Self> {
+        let file = open(Path::new(text(&pin["path"])?))?;
+        let identity = stable(&file)?;
+        ensure(
+            identity[2] == pin["bytes"].as_u64().ok_or("archive length")?
+                && identity[2] <= 64 * MIB,
+            "actual complete bounded archive",
+        )?;
+        ensure(
+            file_sha(&file)? == text(&pin["sha256"])?,
+            "whole actual archive SHA before extraction",
+        )?;
+        let mut size = [0u8; 8];
+        file.read_exact_at(&mut size, 0)?;
+        let n = u64::from_le_bytes(size);
+        ensure(
+            n > 1 && n <= MIB && 8 + n < identity[2],
+            "archive header bound BEFORE read",
+        )?;
+        let mut raw = vec![0u8; n as usize];
+        file.read_exact_at(&mut raw, 8)?;
+        let header: Value = serde_json::from_slice(&raw)?;
+        let fields = header.as_object().ok_or("archive fields")?;
+        ensure(
+            !fields.is_empty() && fields.len() <= 2048,
+            "bounded archive field count",
+        )?;
+        let mut ranges = Vec::with_capacity(fields.len());
+        for (name, item) in fields {
+            if name == "__metadata__" {
+                continue;
+            }
+            let dtype = text(&item["dtype"])?;
+            ensure(matches!(dtype, "BF16" | "F32"), "source archive dtype")?;
+            let shape = item["shape"].as_array().ok_or("archive rank")?;
+            ensure(shape.len() <= 8, "archive rank bound")?;
+            let mut words = 1u64;
+            for v in shape {
+                let axis = v
+                    .as_u64()
+                    .filter(|n| *n > 0 && *n <= 262144)
+                    .ok_or("archive axis")?;
+                words = words.checked_mul(axis).ok_or("archive count overflow")?;
+            }
+            let bounds = item["data_offsets"]
+                .as_array()
+                .filter(|a| a.len() == 2)
+                .ok_or("archive offsets")?;
+            let lo = bounds[0].as_u64().ok_or("archive begin")?;
+            let hi = bounds[1].as_u64().ok_or("archive end")?;
+            ensure(
+                lo < hi
+                    && hi <= identity[2] - 8 - n
+                    && hi - lo
+                        == words
+                            .checked_mul(if dtype == "BF16" { 2 } else { 4 })
+                            .ok_or("archive width overflow")?,
+                "complete source archive extent",
+            )?;
+            ranges.push((lo, hi));
+        }
+        ranges.sort_unstable();
+        let mut end = 0;
+        for (lo, hi) in ranges {
+            ensure(lo == end, "archive no gaps/overlap")?;
+            end = hi;
+        }
+        ensure(
+            8 + n + end == identity[2] && stable(&file)? == identity,
+            "complete stable archive/no trailing",
+        )?;
+        Ok(Self {
+            file,
+            pin: pin.clone(),
+            identity,
+            header,
+            start: 8 + n,
+        })
+    }
+    fn raw(&self, row: &Value) -> Proof<Vec<u8>> {
+        let item = &self.header[text(&row["name"])?];
+        ensure(
+            item["dtype"] == row["dtype"] && item["shape"] == row["shape"],
+            "actual source name/dtype/shape",
+        )?;
+        let lo = item["data_offsets"][0].as_u64().ok_or("stage offset")?;
+        let hi = item["data_offsets"][1].as_u64().ok_or("stage end")?;
+        ensure(
+            hi > lo && hi - lo == row["bytes"].as_u64().ok_or("stage bytes")? && hi - lo <= 24576,
+            "bounded complete norm stage",
+        )?;
+        ensure(
+            stable(&self.file)? == self.identity,
+            "archive FD before stage",
+        )?;
+        let mut raw = vec![0u8; (hi - lo) as usize];
+        self.file.read_exact_at(&mut raw, self.start + lo)?;
+        ensure(
+            stable(&self.file)? == self.identity && sha(&raw) == text(&row["raw_sha256"])?,
+            "exact raw observed stage before interpretation",
+        )?;
+        let width = if row["dtype"] == "BF16" { 2 } else { 4 };
+        for b in raw.chunks_exact(width) {
+            let bits = if width == 2 {
+                u32::from(u16::from_le_bytes([b[0], b[1]])) << 16
+            } else {
+                u32::from_le_bytes(b.try_into()?)
+            };
+            ensure(
+                f32::from_bits(bits).is_finite(),
+                "selected original norm stage finite; preserve all bits",
+            )?;
+        }
+        Ok(raw)
+    }
+    fn close_proof(&self) -> Proof<()> {
+        ensure(
+            stable(&self.file)? == self.identity
+                && file_sha(&self.file)? == text(&self.pin["sha256"])?,
+            "complete retained archive unchanged after all stage reads",
+        )
+    }
+}
+fn compare(
+    actual: &[f32],
+    row: &Value,
+    archive: &Archive,
+    out: &Path,
+    label: &str,
+    write: bool,
+) -> Proof<Value> {
+    let expected = archive.raw(row)?;
+    let width = if row["dtype"] == "BF16" { 2 } else { 4 };
+    ensure(
+        actual.len() * width == expected.len(),
+        "complete word coverage before zip",
+    )?;
+    let mut different = 0;
+    let mut first = Value::Null;
+    let mut max_abs = 0f64;
+    for (i, (a, b)) in actual.iter().zip(expected.chunks_exact(width)).enumerate() {
+        let bits = if width == 2 {
+            u32::from(u16::from_le_bytes([b[0], b[1]])) << 16
+        } else {
+            u32::from_le_bytes(b.try_into()?)
+        };
+        ensure(
+            a.is_finite() && (width != 2 || a.to_bits() & 0xffff == 0),
+            "finite exact output dtype",
+        )?;
+        if a.to_bits() != bits {
+            different += 1;
+            max_abs = max_abs.max((f64::from(*a) - f64::from(f32::from_bits(bits))).abs());
+            if first.is_null() {
+                first = json!({"word":i,"actual":format!("{:08x}",a.to_bits()),"expected":format!("{bits:08x}")});
+            }
+        }
+    }
+    // Control envelope covers the one expected slice and this one emitted Vec.
+    let bytes = actual
+        .iter()
+        .flat_map(|v| v.to_bits().to_le_bytes())
+        .collect::<Vec<_>>();
+    let file = format!("{label}.f32le");
+    if write {
+        OpenOptions::new()
+            .write(true)
+            .create_new(true)
+            .open(out.join(relative(&json!(file))?))?
+            .write_all(&bytes)?;
+    }
+    Ok(
+        json!({"label":label,"words":actual.len(),"different_words":different,"first_difference":first,"max_abs":max_abs,"file":file,"bytes":bytes.len(),"sha256":sha(&bytes),"reference":row}),
+    )
+}
+fn typed(
+    value: &GatedRmsSliceOutput,
+    rows: &Value,
+    archive: &Archive,
+    out: &Path,
+    step: usize,
+    write: bool,
+) -> Proof<Value> {
+    let mut checks = Vec::new();
+    for (name, actual) in [
+        ("core_bf16", value.core_bf16()?),
+        ("gate_bf16", value.gate_bf16()?),
+        ("output_bf16", value.output_bf16()?),
+    ] {
+        checks.push(compare(
+            actual,
+            &rows[name],
+            archive,
+            out,
+            &format!("step{step}.typed.{name}"),
+            write,
+        )?);
+    }
+    let different = checks
+        .iter()
+        .map(|v| v["different_words"].as_u64().unwrap_or(u64::MAX))
+        .sum::<u64>();
+    Ok(
+        json!({"step":step,"output_words":6144,"retained_input_words":12288,"different_words":different,"checks":checks}),
+    )
+}
+fn primitive(
+    ops: &OriginalWindowOperators,
+    rows: &Value,
+    archive: &Archive,
+    out: &Path,
+    step: usize,
+) -> Proof<Value> {
+    let source = ops.source();
+    let gamma = source.parameter("model.language_model.layers.0.linear_attn.norm.weight")?;
+    ensure(
+        gamma.tensor().kind == QuantKind::Bf16
+            && gamma.tensor().shape == [128]
+            && gamma.tensor().byte_len == 256,
+        "actual original gamma codec/shape BEFORE payload copies",
+    )?;
+    let _diagnostic = source.reserve_workspace(DIAGNOSTIC)?;
+    let c = archive.raw(&rows["core_bf16"])?;
+    let z = archive.raw(&rows["gate_bf16"])?;
+    let expected_gamma = archive.raw(&rows["gamma_bf16"])?;
+    let window = source.read_parameter(&gamma, 0..128)?;
+    ensure(
+        gamma.tensor().kind == QuantKind::Bf16
+            && gamma.tensor().shape == [128]
+            && gamma.tensor().byte_len == 256
+            && window.is_complete()
+            && window.source_offset() == 168892952
+            && window.shard_basename() == "model-00001-of-00018.safetensors"
+            && window.as_slice() == expected_gamma,
+        "genuine immutable raw BF16 gamma equals actual public source, no F32 proxy",
+    )?;
+    let decode = |raw: &[u8]| {
+        raw.chunks_exact(2)
+            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
+            .collect::<Vec<_>>()
+    };
+    let core = decode(&c);
+    let gate = decode(&z);
+    let result = zerogravity_cpu::original_hf::norm::gated_rms_128_bf16_gamma_h48(
+        source.source_identity(),
+        zerogravity_core::source::SourceTensorRef {
+            kind: gamma.tensor().kind,
+            shape: &gamma.tensor().shape,
+            packed_bytes: window.as_slice(),
+        },
+        &core,
+        &gate,
+    )?;
+    let gamma_source = json!({"offset":window.source_offset(),"bytes":window.as_slice().len(),"shard":window.shard_basename(),"shard_sha256":window.shard_sha256(),"bundle_sha256":window.verified_bundle_sha256()});
+    let values = [
+        result.mean(),
+        result.inverse(),
+        result.normalized_bf16(),
+        result.weighted_bf16(),
+        result.silu_fp32(),
+        result.product_fp32(),
+        result.output_bf16(),
+    ];
+    let mut checks = Vec::new();
+    for (name, actual) in STAGES.into_iter().zip(values) {
+        checks.push(compare(
+            actual,
+            &rows[name],
+            archive,
+            out,
+            &format!("step{step}.stage.{name}"),
+            true,
+        )?);
+    }
+    let words = checks
+        .iter()
+        .map(|v| v["words"].as_u64().unwrap_or(0))
+        .sum::<u64>();
+    let different = checks
+        .iter()
+        .map(|v| v["different_words"].as_u64().unwrap_or(u64::MAX))
+        .sum::<u64>();
+    ensure(words == 30816, "all7 original stages complete")?;
+    Ok(
+        json!({"step":step,"words":words,"different_words":different,"checks":checks,"diagnostic_scratch_bytes":DIAGNOSTIC,"gamma_source":gamma_source}),
+    )
+}
+fn run() -> Proof<Value> {
+    ensure(!cfg!(debug_assertions), "release-only operator replay")?;
+    let args = std::env::args().skip(1).collect::<Vec<_>>();
+    ensure(
+        args.len() == 6 && args[5] == "27b",
+        "original27B two-teacher replay only",
+    )?;
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "bf16-gamma-gated-rms-archive-replay")?;
+    arbiter.register(key.clone(), HOST)?;
+    let host = TransferAdmission {
+        arbiter,
+        resource: key,
+        owner: ResourceOwner::new("27b", "bf16-gamma-gated-rms-private-replay")?,
+    };
+    // Admission precedes all metadata/archive payload reads/copies or source factory.
+    let control = host.arbiter.admit(
+        &host.resource,
+        host.owner.clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Other,
+            bytes: CONTROL,
+        }],
+    )?;
+    let input_bytes = read(Path::new(&args[0]), MIB as usize)?;
+    let reference_bytes = read(Path::new(&args[2]), MIB as usize)?;
+    ensure(
+        sha(&input_bytes) == args[1] && sha(&reference_bytes) == args[3],
+        "exact raw metadata arguments",
+    )?;
+    let inputs: Value = serde_json::from_slice(&input_bytes)?;
+    let reference: Value = serde_json::from_slice(&reference_bytes)?;
+    let bindings: Value = serde_json::from_str(include_str!("source-bindings.json"))?;
+    for document in [&inputs, &reference] {
+        ensure(
+            document["model"] == "27b"
+                && document["heads"] == 48
+                && document["layer"] == 0
+                && document["profile"] == "original27B-raw-source-bf16-v1"
+                && document["teacher_tokens"] == json!([11, 353])
+                && document["source_bindings_sha256"]
+                    == sha(include_bytes!("source-bindings.json")),
+            "exact original source/profile/two real calls",
+        )?;
+    }
+    ensure(
+        inputs["schema"] == "zerogravity.gated-rms.archived-inputs.h48.v1"
+            && reference["schema"] == "zerogravity.gated-rms.archived-reference.h48.v1"
+            && reference["inputs_sha256"] == args[1]
+            && inputs["steps"].as_array().is_some_and(|v| v.len() == 2)
+            && reference["steps"].as_array().is_some_and(|v| v.len() == 2),
+        "complete two-phase metadata link",
+    )?;
+    let out = Path::new(&args[4]);
+    std::fs::create_dir(out)?;
+    let ready = factory::open_source(&host)?;
+    let source = ready.source;
+    ensure(
+        source.metadata_reserved_bytes() == SOURCE_META,
+        "unchanged admitted original source metadata",
+    )?;
+    let files = bindings["source"]["source_files"]
+        .as_array()
+        .ok_or("source pins")?;
+    let source_hash = |name: &str| -> Proof<&str> {
+        text(
+            &files
+                .iter()
+                .find(|v| v["path"] == name)
+                .ok_or("source pin")?["sha256"],
+        )
+    };
+    let ops = OriginalWindowOperators::bind(
+        source.clone(),
+        &ready.config,
+        source_hash("config.json")?,
+        source_hash("model.safetensors.index.json")?,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )?;
+    ensure(
+        ops.binding().text_tensor_count() == 851
+            && ops.binding().config().geometry().recurrent_value_heads == Some(48),
+        "genuine851-role H48 binding",
+    )?;
+    let source_binding = json!({"identity":format!("{:?}",source.source_identity()),"verified_bundle_sha256":source.verified_bundle_sha256(),"verified_index_sha256":source.verified_shard_index_sha256(),"complete_text_roles":851});
+    let before = snapshot(&host)?;
+    ensure(
+        before.reserved == SOURCE_META + 8 * MIB + CONTROL && before.reservations == 4,
+        "steady original source/binding/control reservations",
+    )?;
+    let archives = [
+        Archive::new(&bindings["step0_archive"])?,
+        Archive::new(&bindings["step1_archive"])?,
+    ];
+    let mut results = Vec::with_capacity(2);
+    let mut primitives = Vec::with_capacity(2);
+    let mut comparisons = Vec::with_capacity(2);
+    let mut negative = Vec::new();
+    for (step, archive) in archives.iter().enumerate() {
+        let rows = &bindings["steps"][step]["tensors"];
+        ensure(
+            inputs["steps"][step]["operands"]["core_bf16"] == rows["core_bf16"]
+                && inputs["steps"][step]["operands"]["gate_bf16"] == rows["gate_bf16"]
+                && inputs["steps"][step]["operands"]["gamma_bf16"] == rows["gamma_bf16"],
+            "metadata operands identical to bound source call",
+        )?;
+        for name in STAGES {
+            ensure(
+                reference["steps"][step]["targets"][name] == rows[name],
+                "metadata targets identical to observed call",
+            )?;
+        }
+        let c = archive.raw(&rows["core_bf16"])?;
+        let z = archive.raw(&rows["gate_bf16"])?;
+        let input = GatedRmsSliceInput::from_bf16_bytes(source.source_identity(), PROFILE, &c, &z)?;
+        let prior = snapshot(&host)?;
+        ensure(
+            ops.execute_gated_rms_bf16_gamma_slice(0, input, Some(&AtomicBool::new(true)))
+                .is_err(),
+            "pre-cancel refuses before import",
+        )?;
+        same_ledger(&host, &prior)?;
+        ensure(
+            ops.execute_gated_rms_bf16_gamma_slice(1, input, None)
+                .is_err(),
+            "wrong layer refuses",
+        )?;
+        same_ledger(&host, &prior)?;
+        let needed = INPUT + OUTPUT;
+        let mut pressure = Vec::new();
+        let held = snapshot(&host)?
+            .remaining
+            .checked_sub(needed - 1)
+            .ok_or("pressure fit")?;
+        let mut remaining = held;
+        while remaining > 0 {
+            let n = remaining.min(32 * MIB);
+            pressure.push(source.reserve_workspace(n as usize)?);
+            remaining -= n;
+        }
+        let pressured = snapshot(&host)?;
+        ensure(
+            pressured.remaining == needed - 1
+                && ops
+                    .execute_gated_rms_bf16_gamma_slice(0, input, None)
+                    .is_err(),
+            "actual pool one byte short of aggregate",
+        )?;
+        same_ledger(&host, &pressured)?;
+        drop(pressure);
+        same_ledger(&host, &prior)?;
+        primitives.push(primitive(&ops, rows, archive, out, step)?);
+        same_ledger(&host, &prior)?;
+        let result = ops.execute_gated_rms_bf16_gamma_slice(0, input, None)?;
+        ensure(
+            result.source_identity() == source.source_identity()
+                && result.numerical_profile() == PROFILE,
+            "typed original source/profile result",
+        )?;
+        comparisons.push(typed(&result, rows, archive, out, step, true)?);
+        results.push(result);
+        ensure(
+            snapshot(&host)?.reserved == before.reserved + (step as u64 + 1) * needed,
+            "all retained typed inputs/output charged",
+        )?;
+        negative.push(json!({"step":step,"pre_cancel":true,"wrong_layer":true,"real_one_byte_short_aggregate_refusal":true}));
+    }
+    let stage_words = primitives
+        .iter()
+        .map(|v| v["words"].as_u64().unwrap_or(0))
+        .sum::<u64>();
+    let differences = primitives
+        .iter()
+        .chain(&comparisons)
+        .map(|v| v["different_words"].as_u64().unwrap_or(u64::MAX))
+        .sum::<u64>();
+    write_json(
+        &out.join("comparisons.json"),
+        &json!({"primitive":primitives,"typed":comparisons,"different_words":differences}),
+    )?;
+    ensure(
+        stage_words == 61632 && differences == 0,
+        "original stages/typed final differ; retain exact mismatch records",
+    )?;
+    let live = snapshot(&host)?;
+    ensure(
+        class(&live, AllocationClass::Output) == 2 * 3 * 6144 * 4
+            && class(&live, AllocationClass::TransferStaging) == 0,
+        "all actual typed payload owners and no gamma staging remain",
+    )?;
+    drop(ops);
+    drop(source);
+    drop(ready.config);
+    let retained = snapshot(&host)?;
+    ensure(
+        retained.reserved == SOURCE_META + CONTROL + 2 * (INPUT + OUTPUT),
+        "source retained by actual2 private results",
+    )?;
+    let mut retained_checks = Vec::new();
+    for (step, (result, archive)) in results.iter().zip(&archives).enumerate() {
+        retained_checks.push(typed(
+            result,
+            &bindings["steps"][step]["tensors"],
+            archive,
+            out,
+            step,
+            false,
+        )?);
+    }
+    ensure(
+        retained_checks == comparisons,
+        "all retained source/input/final bits unchanged after callers drop",
+    )?;
+    for archive in &archives {
+        archive.close_proof()?;
+    }
+    drop(results);
+    ensure(
+        snapshot(&host)?.reserved == CONTROL,
+        "source/result last drop leaves only control reservation",
+    )?;
+    drop(archives);
+    drop(inputs);
+    drop(reference);
+    drop(bindings);
+    drop(input_bytes);
+    drop(reference_bytes);
+    drop(control);
+    let final_row = snapshot(&host)?;
+    ensure(
+        final_row.reserved == 0 && final_row.reservations == 0 && !final_row.fenced,
+        "original host arbiter final0",
+    )?;
+    let affinity = std::fs::read_to_string("/proc/self/status")?
+        .lines()
+        .find(|v| v.starts_with("Cpus_allowed_list:"))
+        .ok_or("actual four-CPU affinity")?
+        .to_owned();
+    Ok(
+        json!({"status":"PASS_PRIVATE_ORIGINAL_H48_BF16_GAMMA_GATED_RMS_ONLY","inputs_sha256":args[1],"reference_sha256":args[3],"actual_affinity":affinity,"primitive_stage_checks":14,"primitive_stage_words":61632,"typed_final_checks":2,"typed_final_words":12288,"retained_input_words":24576,"comparisons":comparisons,"primitive":primitives,"retained_comparisons":retained_checks,"negative":negative,"steady_ledger":ledger(&before),"live_ledger":ledger(&live),"retained_ledger":ledger(&retained),"final_ledger":ledger(&final_row),"metadata_and_payload_admitted_before_native_extraction":true,"gamma_actual_source_codec_extent_raw_identity":true,"source_binding":source_binding,"full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified":false}),
+    )
+}
+fn main() {
+    let result = run();
+    let report = match &result {
+        Ok(v) => v.clone(),
+        Err(e) => json!({"status":"FAIL_UNQUALIFIED","error":e.to_string()}),
+    };
+    let written = std::env::args()
+        .nth(5)
+        .ok_or("output missing")
+        .and_then(|p| {
+            write_json(&Path::new(&p).join("report.json"), &report)
+                .map_err(|_| "report write failed")
+        });
+    println!("{report}");
+    if result.is_err() || written.is_err() {
+        std::process::exit(1);
+    }
+}
--- /dev/null
+++ b/owned_lifetime.py
@@ -0,0 +1,91 @@
+"""Exact accepted owned-native leader cleanup functions; no child on import."""
+import os
+import resource
+import signal
+import time
+from contracts import require
+CAP = 4 * 1024 * 1024
+
+
+def unreaped(child):
+    require(child.returncode is None, "leader already reaped")
+    value = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if value is None:
+        return None
+    require(value.si_pid == child.pid, "owned child identity")
+    return value.si_status if value.si_code == os.CLD_EXITED else -value.si_status
+
+
+def cleanup_owned(child, record, errors, *, observe=unreaped, pause=time.sleep):
+    """Independent bounded stages; never signal after the first reap attempt."""
+    def bound():
+        observe(child)
+        require(os.getpgid(child.pid) == child.pid and os.getsid(child.pid) == child.pid,
+                "owned session/group binding")
+        return child.pid
+
+    try:
+        os.killpg(bound(), signal.SIGTERM)
+        record["group_term_before_reap"] = True
+    except ProcessLookupError:
+        record["group_term_absent"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
+    try:
+        grace = time.monotonic() + 0.5
+        while time.monotonic() < grace:
+            observe(child)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # TERM/grace/one KILL error cannot skip the separately bound KILL stage.
+    # Both attempts occur before ANY wait, while the owned leader is unreaped.
+    for attempt in range(2):
+        try:
+            os.killpg(bound(), signal.SIGKILL)
+            record["group_kill_before_reap"] = True
+            break
+        except ProcessLookupError:
+            record["group_kill_absent_before_reap"] = True
+            break
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    # ALWAYS attempt reap even after signal/ownership failures. Popen.wait only
+    # targets this child. An interrupted first wait cannot skip the final wait.
+    # No signal operation occurs below this point, including after timeout.
+    try:
+        record["reaped_exit_code"] = child.wait(timeout=5)
+        record["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned initial reap: " + repr(error))
+    finally:
+        if child.returncode is None:
+            try:
+                record["reaped_exit_code"] = child.wait(timeout=5)
+                record["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final reap: " + repr(error))
+        else:
+            record["reaped_exit_code"] = child.returncode
+            record["leader_reaped"] = True
+    if record.get("leader_reaped"):
+        try:
+            deadline = time.monotonic() + 5
+            while True:
+                try:
+                    os.killpg(child.pid, 0)  # Read-only absence check, NEVER kill after reap.
+                except ProcessLookupError:
+                    record["group_absent"] = True
+                    break
+                require(time.monotonic() < deadline, "owned process group remains")
+                pause(0.05)
+        except BaseException as error:
+            errors.append("owned group absence: " + repr(error))
+
+
+def launch_limits():
+    # Python is single-threaded at Popen; no foreign runtime is imported.
+    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
+    resource.setrlimit(resource.RLIMIT_CPU, (3601, 3601))
+    resource.setrlimit(resource.RLIMIT_FSIZE, (CAP, CAP))
+    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
--- /dev/null
+++ b/owned_native.py
@@ -0,0 +1,114 @@
+"""ROOT ONLY native CPU helper orchestration. No Torch/model/container imports."""
+import argparse
+import json
+import os
+from pathlib import Path
+import selectors
+import signal
+import stat
+import subprocess
+import time
+
+from contracts import HERE, frozen, pin, read_json, require, sha, write
+from owned_lifetime import cleanup_owned, launch_limits, unreaped
+
+CAP = 4 * 1024 * 1024
+
+
+def source_files(binding):
+    """ROOT performs complete source scans; never provider/opaque-memory math."""
+    root = Path(binding["source_directory"])
+    rows = []
+    for item in binding["source_files"]:
+        path = root / item["path"]
+        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
+        with os.fdopen(os.open(path, flags), "rb") as opened:
+            before = os.fstat(opened.fileno())
+            require(stat.S_ISREG(before.st_mode) and before.st_size == item["bytes"], "original source regular FD/full length")
+            import hashlib
+            digest = hashlib.file_digest(opened, "sha256").hexdigest()
+            after = os.fstat(opened.fileno())
+        identity = lambda value: [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
+        require(identity(before) == identity(after) and digest == item["sha256"], "complete original source FD digest/identity")
+        rows.append({"path": str(path), "bytes": before.st_size, "sha256": digest,
+                     "fd_identity": identity(before)})
+    return rows
+
+
+def execute(binary, binary_fd, inputs, inputs_sha, reference, reference_sha, output, model, cpus, freeze, source_before):
+    argv = [str(binary),str(inputs),inputs_sha,str(reference),reference_sha,str(output),model]
+    environment = {key: value for key, value in os.environ.items()
+                   if key in ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "XDG_CACHE_HOME")}
+    environment.update(RAYON_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
+                       CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void",
+                       ATEN_CPU_CAPABILITY="avx2")
+    record = {"status": "UNQUALIFIED", "argv": argv, "freeze_manifest_sha256": freeze,
+              "binary_sha256": sha(binary), "cpu_affinity": cpus, "scope": "private archived original H48 BF16-gamma gated RMS only",
+              "address_space_cap_bytes": 512 * 1024**2, "wall_cap_seconds": 900,
+              "log_cap_bytes": CAP, "source_before": source_before,
+              "no_model_or_session_or_GPU_execution": True}
+    child = selector = pipe = log = None
+    errors, primary, eof, retained = [], None, False, 0
+    started = time.monotonic()
+    try:
+        log = (output.parent / (model + "-lifecycle") / "execution.log").open("xb")
+        child = subprocess.Popen(argv, executable=f"/proc/self/fd/{binary_fd}", pass_fds=(binary_fd,),
+            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
+            env=environment, start_new_session=True, preexec_fn=launch_limits)
+        # Every post-spawn operation is covered by cleanup below.
+        record["owned_cli_pid"] = child.pid
+        pipe = child.stdout
+        os.set_blocking(pipe.fileno(), False)
+        selector = selectors.DefaultSelector()
+        selector.register(pipe, selectors.EVENT_READ)
+
+        def drain(timeout):
+            nonlocal retained, eof
+            for key, _ in selector.select(timeout):
+                try:
+                    data = os.read(key.fd, 65536)
+                except BlockingIOError:
+                    continue
+                if not data:
+                    selector.unregister(key.fileobj)
+                    eof = True
+                else:
+                    require(retained + len(data) <= CAP, "bounded child combined log")
+                    log.write(data)
+                    retained += len(data)
+        while True:
+            drain(0.05)
+            status = unreaped(child)
+            if status is not None:
+                record["exit_code"] = status
+                break
+            require(time.monotonic() - started < 900, "finite helper deadline")
+    except BaseException as error:
+        primary = repr(error)
+    finally:
+        if child is not None:
+            cleanup_owned(child, record, errors)
+        if selector is not None:
+            try:
+                deadline = time.monotonic() + 2
+                while not eof and time.monotonic() < deadline:
+                    drain(0.05)
+                require(eof, "owned pipe EOF deadline")
+            except BaseException as error:
+                errors.append("final drain: " + repr(error))
+        for name, owned in (("selector", selector), ("pipe", pipe), ("log", log)):
+            if owned is not None:
+                try:
+                    owned.close()
+                    record[name + "_closed"] = True
+                except BaseException as error:
+                    errors.append(name + " close: " + repr(error))
+        record.update(primary_error=primary, cleanup_errors=errors, pipe_eof=eof, log_bytes=retained,
+                      elapsed_seconds_not_benchmark=time.monotonic() - started)
+    success = primary is None and not errors and record.get("exit_code") == record.get("reaped_exit_code") == 0 \
+        and record.get("leader_reaped") and record.get("group_absent") and eof
+    record["status"] = "CHILD_EXECUTED_REQUIRES_VALIDATION" if success else "FAIL_UNQUALIFIED"
+    write(output.parent / (model + "-lifecycle") / "invocation.json", record)
+    require(success, "owned native helper failed; preserve fresh attempt")
+    return record
+
--- /dev/null
+++ b/prepare_reference.py
@@ -0,0 +1,118 @@
+"""ROOT-only metadata transport. Does not open/archive/hash any tensor payload."""
+import argparse
+import json
+from pathlib import Path
+from contracts import HERE, INPUTS, OUTPUTS, PROFILE, frozen, pin, read_json, require, sha, write
+
+def authenticate_metadata():
+    binding = read_json(HERE/'source-bindings.json')
+    for key in ('capture_manifest', 'capture_validation', 'capture_invocation', 'capture_preparation',
+                'source_plan', 'factory_report', 'step0_metadata', 'step1_metadata'):
+        pin(binding[key]['path'], binding[key])
+    capture = read_json(binding['capture_manifest']['path'])
+    validation = read_json(binding['capture_validation']['path'])
+    invocation = read_json(binding['capture_invocation']['path'])
+    require(capture['status'] == 'CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION' and
+            capture['source_repo'] == binding['source']['repo'] == 'Qwen/Qwen3.8-27B' and
+            capture['source_revision'] == binding['source']['revision'] == '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0' and
+            capture['snapshot'] == binding['source']['snapshot'] == '98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e' and
+            capture['parameter_profile'] == PROFILE and capture['tokens'] == [11, 353] and
+            capture['all_source_parameters_BF16'] is True and
+            capture['full_model_StateRoot_native_exp_GPU_eligible'] is False,
+            'actual complete original27B public raw-source profile')
+    require(validation['status'] == 'PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY' and
+            validation['manifest_sha256'] == binding['capture_manifest']['sha256'] and
+            validation['source_plan_sha256'] == capture['source_plan_sha256'] == binding['source_plan']['sha256'] and
+            validation['heads'] == 48 and validation['teachers'] == [11, 353] and
+            validation['full_model_112_StateRoots_native_exp_allocation_GPU_or_performance_qualified'] is False,
+            'independent original reference accepted first')
+    require(invocation['status'] == 'EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION' and
+            invocation['exit_code'] == 0 and invocation['cleanup_errors'] == [] and
+            invocation['source_FD_cleanup_errors'] == [] and invocation['source_FDs_closed'] is True and
+            invocation['owned_container_absent_after_reap'] is True and
+            invocation['source_before'] == invocation['source_after'] and
+            sha(binding['capture_invocation']['path']) == validation['invocation_sha256'],
+            'prior genuine runtime/source owner lifecycle closed')
+    runtime = capture['runtime']
+    require(runtime['torch'] == '2.13.0+cu129' and
+            runtime['torch_git'] == 'cf30153c4c131c8164ee7798e5022d810682e2cb' and
+            runtime['transformers'] == '5.17.0' and runtime['cpu_capability'] == 'AVX2' and
+            runtime['threads'] == 4 and runtime['interop_threads'] == 1 and runtime['cuda_available'] is False and
+            runtime['modeling_sha256'] == '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b',
+            'actual pinned unchanged original public CPU math')
+    norm = capture['norm_profile']
+    require(norm == binding['norm_profile'] and norm['heads'] == 48 and norm['head_width'] == 128 and
+            norm['raw_gamma_dtype'] == norm['weighted_dtype'] == norm['return_dtype'] == 'torch.bfloat16' and
+            norm['gate_product_dtype'] == 'torch.float32' and norm['epsilon'] == 1e-6 and norm['activation'] == 'silu',
+            'distinct gammaBF16/weightedBF16→gateF32→returnBF16 contract')
+    steps = capture['steps']
+    require(len(steps) == 2 and [s['step'] for s in steps] == [0, 1] and
+            [s['teacher_token'] for s in steps] == [11, 353] and
+            all(s['plain_observed_all_outputs_and_state_exact'] is True for s in steps),
+            'both real initial/cached calls plain/observed original outputs/state exact')
+    for index, (step, bound) in enumerate(zip(steps, binding['steps'])):
+        require(read_json(binding[f'step{index}_metadata']['path']) ==
+                dict(step, plain_observed_all_outputs_and_state_exact=False), 'original prevalidation metadata unchanged')
+        require(step['archive_bytes'] == binding[f'step{index}_archive']['bytes'] and
+                step['archive_sha256'] == binding[f'step{index}_archive']['sha256'] and
+                [d for d in step['dispatch'] if d['scope'] == 'norm'] == bound['actual_norm_dispatch'],
+                'full raw archive/actual twelve operations bound; no arithmetic replay here')
+        require(len(bound['actual_norm_dispatch']) == 12, 'complete unchanged norm operator schedule')
+        for name, row in bound['tensors'].items():
+            actual = dict(name=row['name'], **step['tensors'][row['name']])
+            actual['original_torch_dtype'] = actual['dtype']
+            actual['dtype'] = {'torch.bfloat16':'BF16', 'torch.float32':'F32'}[actual['dtype']]
+            require(actual == row and row['device'] == 'cpu' and row['contiguous'] is True,
+                    'exact original descriptor/bit hash/geometry: '+name)
+        require(step['tensors'][bound['semantic_frame_return']]['raw_sha256'] == bound['tensors']['output_bf16']['raw_sha256'] and
+                step['tensors'][bound['semantic_frame_weighted']]['raw_sha256'] == bound['tensors']['weighted_bf16']['raw_sha256'],
+                'dispatch outputs equal actual frame weighted/return, no guessed semantic label')
+    require(steps[0]['tensors']['chunk.return.actual_return_1']['raw_sha256'] ==
+            steps[1]['tensors']['recurrent.input.initial_state']['raw_sha256'],
+            'actual initial state supplies cached prior; earlier independent raw proof remains bound')
+    require(steps[1]['actual_calls'] == {'Qwen3_5GatedDeltaNet.forward': 1, 'Qwen3_5RMSNormGated.forward': 1,
+            'causal_conv1d_update': 1, 'l2norm': 2, 'torch_recurrent_gated_delta_rule': 1},
+            'second call is genuine cached353, not fresh initial substitute')
+    return binding
+
+def documents():
+    binding = authenticate_metadata()
+    common = dict(model='27b', heads=48, layer=0, profile=PROFILE, teacher_tokens=[11,353],
+                  source_bindings_sha256=sha(HERE/'source-bindings.json'),
+                  no_payload_extracted_and_no_new_operator_run=True)
+    inputs = dict(common, schema='zerogravity.gated-rms.archived-inputs.h48.v1', steps=[
+        dict(step=s['step'], teacher_token=s['teacher_token'], operands={n:s['tensors'][n] for n in INPUTS})
+        for s in binding['steps']])
+    reference = dict(common, schema='zerogravity.gated-rms.archived-reference.h48.v1', steps=[
+        dict(step=s['step'], teacher_token=s['teacher_token'], targets={n:s['tensors'][n] for n in OUTPUTS})
+        for s in binding['steps']], expected_primitive_stage_checks=14, expected_primitive_stage_words=61632,
+        expected_typed_final_checks=2, expected_typed_final_words=12288, expected_retained_input_words=24576)
+    return inputs, reference
+
+def validate(directory):
+    inputs, reference = documents()
+    require(read_json(directory/'inputs.json') == inputs, 'complete exact metadata-only operands')
+    reference['inputs_sha256'] = sha(directory/'inputs.json')
+    require(read_json(directory/'reference.json') == reference and
+            {p.name for p in directory.iterdir()} == {'inputs.json','reference.json'},
+            'complete metadata-only targets/link/tree')
+    return dict(status='PASS_METADATA_BINDING_ONLY_RAW_BYTES_NOT_VALIDATED',
+        inputs_sha256=sha(directory/'inputs.json'), reference_sha256=sha(directory/'reference.json'),
+        source_bindings_sha256=sha(HERE/'source-bindings.json'),
+        full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified=False)
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--new-output', type=Path, required=True)
+    args = parser.parse_args()
+    frozen()
+    inputs, reference = documents()
+    require(args.new_output.resolve().parent == HERE, 'fresh own bounded metadata directory')
+    args.new_output.mkdir(mode=0o700)
+    write(args.new_output/'inputs.json', inputs)
+    reference['inputs_sha256'] = sha(args.new_output/'inputs.json')
+    write(args.new_output/'reference.json', reference)
+    print(json.dumps(validate(args.new_output), sort_keys=True))
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ b/run_replay.py
@@ -0,0 +1,68 @@
+"""ROOT ONLY fresh owned native child; no provider/model/operator on import."""
+import argparse
+import hashlib
+import os
+from pathlib import Path
+import signal
+from contracts import HERE, frozen, implementation, read_json, require, sha, write
+from prepare_reference import validate as validate_archived
+from owned_native import execute, source_files
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--binary', type=Path, required=True)
+    parser.add_argument('--binary-sha256', required=True)
+    parser.add_argument('--reference-directory', type=Path, required=True)
+    parser.add_argument('--implementation', type=Path, required=True)
+    parser.add_argument('--new-output', type=Path, required=True)
+    args = parser.parse_args()
+    def deadline(_signal, _frame):
+        raise TimeoutError('finite original source audit and gated RMS proof deadline')
+    signal.signal(signal.SIGALRM, deadline)
+    signal.alarm(2400)
+    freeze = frozen()
+    source_attestation = implementation(args.implementation)
+    archived = validate_archived(args.reference_directory)
+    require(args.reference_directory.resolve().parent == HERE and
+            args.new_output.resolve().parent == HERE, 'own bounded reference/attempt only')
+    require(not args.binary.is_symlink() and args.binary.is_file() and
+            sha(args.binary) == args.binary_sha256, 'ROOT reviewed actual native binary')
+    cpus = sorted(os.sched_getaffinity(0))[:4]
+    require(len(cpus) == 4, 'four available CPUs')
+    os.sched_setaffinity(0, cpus)
+    args.new_output.mkdir(mode=0o700)
+    (args.new_output/'27b-lifecycle').mkdir(mode=0o700)
+    inputs, reference = args.reference_directory/'inputs.json', args.reference_directory/'reference.json'
+    binding = read_json(HERE/'source-bindings.json')['source']
+    fd = os.open(args.binary.resolve(), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
+    try:
+        with os.fdopen(os.dup(fd), 'rb') as opened:
+            require(hashlib.file_digest(opened, 'sha256').hexdigest() == args.binary_sha256, 'actual exec-FD SHA')
+        before = source_files(binding)
+        output = args.new_output/'27b'
+        execute(args.binary.resolve(), fd, inputs.resolve(), sha(inputs), reference.resolve(), sha(reference),
+                output.resolve(), '27b', cpus, freeze, before)
+        after = source_files(binding)
+        require(before == after, 'all full original source files unchanged')
+        require(implementation(args.implementation) == source_attestation and frozen() == freeze and
+                sha(args.binary) == args.binary_sha256 and validate_archived(args.reference_directory) == archived,
+                'all actual build/source/public-reference bindings unchanged')
+        write(output/'source-after.json', {'source_before':before,'source_after':after,'all_equal':True})
+        report = read_json(output/'report.json')
+        require(report['status'] == 'PASS_PRIVATE_ORIGINAL_H48_BF16_GAMMA_GATED_RMS_ONLY' and
+                report['primitive_stage_words'] == 61632 and report['typed_final_words'] == 12288,
+                'actual gated RMS stages and typed finals require independent raw validation')
+        write(args.new_output/'completed.json', {
+            'status':'EXECUTED_REQUIRES_INDEPENDENT_RAW_VALIDATION',
+            'freeze_manifest_sha256':freeze,'binary_sha256':args.binary_sha256,
+            'ROOT_build_source_attestation_sha256':source_attestation,
+            'archived_reference_validation':archived,
+            'report_sha256':sha(output/'report.json'),
+            'source_after_sha256':sha(output/'source-after.json'),
+            'invocation_sha256':sha(args.new_output/'27b-lifecycle/invocation.json'),
+            'full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified':False})
+    finally:
+        os.close(fd)
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ b/source-bindings.json
@@ -0,0 +1,1095 @@
+{
+  "archive_payloads_read_or_hashed_by_preparer": false,
+  "capture_invocation": {
+    "bytes": 22448,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/invocation.json",
+    "sha256": "61b1787ca857a1464fd61c8132475dbe7bac2b4282b3b7ff4db7bc15d427512f"
+  },
+  "capture_manifest": {
+    "bytes": 233492,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/manifest.json",
+    "sha256": "62e82637c7d24a58b733f2358b3061efd767725901314df4ff7378707d946ab8"
+  },
+  "capture_preparation": {
+    "bytes": 2322,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/prepared-manifest-v3.json",
+    "sha256": "d3614b48bb41033a4052d765d3f7633d456812670783049db75ceadf33d8817d"
+  },
+  "capture_validation": {
+    "bytes": 596,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/validation.json",
+    "sha256": "951f74e33de53c4b6b5254bd980df1f6c05f6b9869a46bf617c7fa93184d344d"
+  },
+  "expected_totals": {
+    "all_input_words_including_repeated_gamma": 24832,
+    "input_words_per_step_including_source_gamma": 12416,
+    "primitive_stage_checks": 14,
+    "primitive_stage_words": 61632,
+    "primitive_stage_words_per_step": 30816,
+    "typed_slice_output_checks": 2,
+    "typed_slice_output_words": 12288
+  },
+  "factory_report": {
+    "bytes": 7058,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/source-pin.json",
+    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+  },
+  "full_model_StateRoots_exp_GDN_GPU_or_performance_qualified": false,
+  "gamma_original_tensor": {
+    "absolute_offset": 168892952,
+    "bytes": 256,
+    "dtype": "BF16",
+    "mapping_bytes": 2840,
+    "name": "model.language_model.layers.0.linear_attn.norm.weight",
+    "page_aligned_offset": 168890368,
+    "shape": [
+      128
+    ],
+    "shard": "model-00001-of-00018.safetensors",
+    "view_offset": 2584
+  },
+  "norm_profile": {
+    "activation": "silu",
+    "class": "Qwen3_5RMSNormGated",
+    "epsilon": 1e-06,
+    "gate_product_dtype": "torch.float32",
+    "head_width": 128,
+    "heads": 48,
+    "raw_gamma_dtype": "torch.bfloat16",
+    "return_dtype": "torch.bfloat16",
+    "weighted_dtype": "torch.bfloat16"
+  },
+  "plain_archive": {
+    "bytes": 6518376,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/plain.safetensors",
+    "sha256": "9b20753b733f5f870abf4691eaecf9ce42f5edec0ecec8ae1260684273d5bab8"
+  },
+  "producer_scope": "Original unchanged Qwen3_5RMSNormGated.forward actual H48 call, teacher11 initial paddedchunk64 and truecached353; no live initial/exp/recurrence producer eligibility",
+  "profile": "original27B-raw-source-bf16-v1",
+  "runtime": {
+    "affinity": [
+      0,
+      1,
+      2,
+      3
+    ],
+    "cpu_capability": "AVX2",
+    "cuda_available": false,
+    "interop_threads": 1,
+    "mapped_shared_objects": "UNCHANGED ACCEPTED CAPTURE MANIFEST; no new DSO reads by preparer",
+    "modeling_sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b",
+    "proc_cgroup": "0::/\n",
+    "proc_mountinfo": "1577 1638 0:36 /@home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020 /probe ro,relatime - btrfs /dev/mapper/luks-696fd608-b5dc-4a8d-94ee-4925d011a449 rw,compress=zstd:3,ssd,discard=async,space_cache=v2,subvolid=257,subvol=/@home\n1578 1638 0:36 /@home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files /source ro,relatime - btrfs /dev/mapper/luks-696fd608-b5dc-4a8d-94ee-4925d011a449 rw,compress=zstd:3,ssd,discard=async,space_cache=v2,subvolid=257,subvol=/@home\n1579 1638 0:36 /@home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-typed-oracle-20261020/root-reference-1/captured /prefix-reference ro,relatime - btrfs /dev/mapper/luks-696fd608-b5dc-4a8d-94ee-4925d011a449 rw,compress=zstd:3,ssd,discard=async,space_cache=v2,subvolid=257,subvol=/@home\n1580 1638 0:36 /@home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1 /output rw,relatime - btrfs /dev/mapper/luks-696fd608-b5dc-4a8d-94ee-4925d011a449 rw,compress=zstd:3,ssd,discard=async,space_cache=v2,subvolid=257,subvol=/@home\n1581 1590 0:107 / /dev/shm rw,nosuid,nodev,noexec,relatime - tmpfs shm rw,size=64000k,uid=1000,gid=1000\n1582 1586 0:73 /abird/zerogravity/vllm-0.31.0-cu129/containers/overlay-containers/c868658058aee26f1748603f0f90a11490b1bfbbbd059823ad3bb05a4d324d5f/userdata/.containerenv /run/.containerenv ro,nosuid,nodev,noexec,relatime - tmpfs tmpfs rw,size=2858164k,nr_inodes=714541,mode=700,uid=1000,gid=1000\n1583 1638 0:73 /abird/zerogravity/vllm-0.31.0-cu129/containers/overlay-containers/c868658058aee26f1748603f0f90a11490b1bfbbbd059823ad3bb05a4d324d5f/userdata/hostname /etc/hostname ro,nosuid,nodev,noexec,relatime - tmpfs tmpfs rw,size=2858164k,nr_inodes=714541,mode=700,uid=1000,gid=1000\n1584 1638 0:73 /abird/zerogravity/vllm-0.31.0-cu129/containers/overlay-containers/c868658058aee26f1748603f0f90a11490b1bfbbbd059823ad3bb05a4d324d5f/userdata/hosts /etc/hosts ro,nosuid,nodev,noexec,relatime - tmpfs tmpfs rw,size=2858164k,nr_inodes=714541,mode=700,uid=1000,gid=1000\n1586 1638 0:120 / /run rw,nosuid,nodev,relatime - tmpfs tmpfs rw,uid=1000,gid=1000\n1588 1638 0:122 / /tmp rw,nosuid,nodev,relatime - tmpfs tmpfs rw,size=1048576k,uid=1000,gid=1000\n1589 1638 0:123 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw\n1590 1638 0:124 / /dev rw,nosuid - tmpfs tmpfs rw,size=65536k,mode=755,uid=1000,gid=1000\n1591 1638 0:125 / /sys ro,nosuid,nodev,noexec,relatime - sysfs sysfs rw\n1592 1638 0:126 / /var/tmp rw,nosuid,nodev,relatime - tmpfs tmpfs rw,uid=1000,gid=1000\n1593 1590 0:127 / /dev/pts rw,nosuid,noexec,relatime - devpts devpts rw,gid=100004,mode=620,ptmxmode=666\n1594 1590 0:118 / /dev/mqueue rw,nosuid,nodev,noexec,relatime - mqueue mqueue rw\n1595 1591 0:23 / /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw,nsdelegate,memory_recursiveprot,memory_hugetlb_accounting\n1596 1589 0:73 /crun/.empty-directory /proc/acpi ro,nosuid,nodev,relatime - tmpfs tmpfs rw,size=2858164k,nr_inodes=714541,mode=700,uid=1000,gid=1000\n1638 1597 0:110 / / ro,relatime - overlay overlay rw,lowerdir=3:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20:21:22:23:24:25:26:27:28:29:30:31:32:33:34:35:36:37:38:39:40:41:42:43:44:45:46,upperdir=/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/2d9d99c4a77b56a3f4462462bdbb7b022033c01b1773d788651044fd8b71f9a6/diff,workdir=/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/2d9d99c4a77b56a3f4462462bdbb7b022033c01b1773d788651044fd8b71f9a6/work,redirect_dir=nofollow,uuid=on,fsync=volatile,userxattr\n1639 1590 0:7 /null /dev/null rw,nosuid,noexec - devtmpfs devtmpfs rw,size=1429084k,nr_inodes=3549370,mode=755\n1640 1590 0:7 /zero /dev/zero rw,nosuid,noexec - devtmpfs devtmpfs rw,size=1429084k,nr_inodes=3549370,mode=755\n1641 1590 0:7 /full /dev/full rw,nosuid,noexec - devtmpfs devtmpfs rw,size=1429084k,nr_inodes=3549370,mode=755\n1642 1590 0:7 /tty /dev/tty rw,nosuid,noexec - devtmpfs devtmpfs rw,size=1429084k,nr_inodes=3549370,mode=755\n1643 1590 0:7 /random /dev/random rw,nosuid,noexec - devtmpfs devtmpfs rw,size=1429084k,nr_inodes=3549370,mode=755\n1644 1590 0:7 /urandom /dev/urandom rw,nosuid,noexec - devtmpfs devtmpfs rw,size=1429084k,nr_inodes=3549370,mode=755\n1645 1589 0:73 /crun/.empty-directory /proc/scsi ro,nosuid,nodev,relatime - tmpfs tmpfs rw,size=2858164k,nr_inodes=714541,mode=700,uid=1000,gid=1000\n1646 1589 0:7 /null /proc/kcore ro,nosuid - devtmpfs devtmpfs rw,size=1429084k,nr_inodes=3549370,mode=755\n1647 1589 0:7 /null /proc/keys ro,nosuid - devtmpfs devtmpfs rw,size=1429084k,nr_inodes=3549370,mode=755\n1648 1591 0:73 /crun/.empty-directory /sys/devices/virtual/powercap ro,nosuid,nodev,relatime - tmpfs tmpfs rw,size=2858164k,nr_inodes=714541,mode=700,uid=1000,gid=1000\n1649 1589 0:7 /null /proc/timer_list ro,nosuid - devtmpfs devtmpfs rw,size=1429084k,nr_inodes=3549370,mode=755\n1650 1591 0:73 /crun/.empty-directory /sys/firmware ro,nosuid,nodev,relatime - tmpfs tmpfs rw,size=2858164k,nr_inodes=714541,mode=700,uid=1000,gid=1000\n1651 1648 0:73 /crun/.empty-directory /sys/devices/virtual/powercap ro,nosuid,nodev,relatime - tmpfs tmpfs rw,size=2858164k,nr_inodes=714541,mode=700,uid=1000,gid=1000\n1652 1589 0:7 /null /proc/interrupts ro,nosuid - devtmpfs devtmpfs rw,size=1429084k,nr_inodes=3549370,mode=755\n1654 1589 0:123 /asound /proc/asound ro,nosuid,nodev,noexec,relatime - proc proc rw\n1655 1589 0:123 /bus /proc/bus ro,nosuid,nodev,noexec,relatime - proc proc rw\n1656 1589 0:123 /fs /proc/fs ro,nosuid,nodev,noexec,relatime - proc proc rw\n1657 1589 0:123 /irq /proc/irq ro,nosuid,nodev,noexec,relatime - proc proc rw\n1658 1589 0:123 /sys /proc/sys ro,nosuid,nodev,noexec,relatime - proc proc rw\n1659 1589 0:123 /sysrq-trigger /proc/sysrq-trigger ro,nosuid,nodev,noexec,relatime - proc proc rw\n",
+    "proc_status": "Name:\tpython3\nUmask:\t0022\nState:\tR (running)\nTgid:\t1\nNgid:\t0\nPid:\t1\nPPid:\t0\nTracerPid:\t0\nUid:\t0\t0\t0\t0\nGid:\t0\t0\t0\t0\nFDSize:\t64\nGroups:\t \nNStgid:\t1\nNSpid:\t1\nNSpgid:\t1\nNSsid:\t1\nKthread:\t0\nVmPeak:\t 8485836 kB\nVmSize:\t 8481736 kB\nVmLck:\t       0 kB\nVmPin:\t       0 kB\nVmHWM:\t 1079128 kB\nVmRSS:\t 1076628 kB\nRssAnon:\t  521684 kB\nRssFile:\t  554944 kB\nRssShmem:\t       0 kB\nVmData:\t 3869224 kB\nVmStk:\t     132 kB\nVmExe:\t    2956 kB\nVmLib:\t 2884816 kB\nVmPTE:\t    3292 kB\nVmSwap:\t       0 kB\nHugetlbPages:\t       0 kB\nCoreDumping:\t0\nTHP_enabled:\t1\nuntag_mask:\t0xffffffffffffffff\nThreads:\t10\nSigQ:\t0/111367\nSigPnd:\t0000000000000000\nShdPnd:\t0000000000000000\nSigBlk:\t0000000000000000\nSigIgn:\t0000000001001000\nSigCgt:\t0000000100000002\nCapInh:\t0000000000000000\nCapPrm:\t0000000000000000\nCapEff:\t0000000000000000\nCapBnd:\t0000000000000000\nCapAmb:\t0000000000000000\nNoNewPrivs:\t1\nSeccomp:\t2\nSeccomp_filters:\t1\nSpeculation_Store_Bypass:\tthread vulnerable\nSpeculationIndirectBranch:\tconditional enabled\nCpus_allowed:\t000f\nCpus_allowed_list:\t0-3\nMems_allowed:\t00000000,00000001\nMems_allowed_list:\t0\nvoluntary_ctxt_switches:\t3531\nnonvoluntary_ctxt_switches:\t15644\nx86_Thread_features:\t\nx86_Thread_features_locked:\t\n",
+    "threads": 4,
+    "torch": "2.13.0+cu129",
+    "torch_git": "cf30153c4c131c8164ee7798e5022d810682e2cb",
+    "transformers": "5.17.0"
+  },
+  "schema": "zerogravity.original27b.bf16-gamma-gated-rms.archived-bindings.v1",
+  "source": {
+    "heads": 48,
+    "repo": "Qwen/Qwen3.8-27B",
+    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
+    "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
+    "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
+    "source_files": [
+      {
+        "bytes": 11544,
+        "path": "LICENSE",
+        "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
+      },
+      {
+        "bytes": 65012,
+        "path": "README.md",
+        "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
+      },
+      {
+        "bytes": 8952,
+        "path": "chat_template.jinja",
+        "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
+      },
+      {
+        "bytes": 4312,
+        "path": "config.json",
+        "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+      },
+      {
+        "bytes": 238,
+        "path": "crc32.txt",
+        "sha256": "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83"
+      },
+      {
+        "bytes": 202,
+        "path": "generation_config.json",
+        "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
+      },
+      {
+        "bytes": 3353259,
+        "path": "merges.txt",
+        "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
+      },
+      {
+        "bytes": 3966730552,
+        "path": "model-00001-of-00018.safetensors",
+        "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+      },
+      {
+        "bytes": 3043080328,
+        "path": "model-00002-of-00018.safetensors",
+        "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb"
+      },
+      {
+        "bytes": 2542796952,
+        "path": "model-00003-of-00018.safetensors",
+        "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"
+      },
+      {
+        "bytes": 3988973152,
+        "path": "model-00004-of-00018.safetensors",
+        "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a"
+      },
+      {
+        "bytes": 2099339864,
+        "path": "model-00005-of-00018.safetensors",
+        "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701"
+      },
+      {
+        "bytes": 3979553696,
+        "path": "model-00006-of-00018.safetensors",
+        "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0"
+      },
+      {
+        "bytes": 2108759344,
+        "path": "model-00007-of-00018.safetensors",
+        "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"
+      },
+      {
+        "bytes": 3979553696,
+        "path": "model-00008-of-00018.safetensors",
+        "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79"
+      },
+      {
+        "bytes": 2108759344,
+        "path": "model-00009-of-00018.safetensors",
+        "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8"
+      },
+      {
+        "bytes": 3979553696,
+        "path": "model-00010-of-00018.safetensors",
+        "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9"
+      },
+      {
+        "bytes": 2108759344,
+        "path": "model-00011-of-00018.safetensors",
+        "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8"
+      },
+      {
+        "bytes": 3979553696,
+        "path": "model-00012-of-00018.safetensors",
+        "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55"
+      },
+      {
+        "bytes": 2108759344,
+        "path": "model-00013-of-00018.safetensors",
+        "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a"
+      },
+      {
+        "bytes": 3979553696,
+        "path": "model-00014-of-00018.safetensors",
+        "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b"
+      },
+      {
+        "bytes": 2108759344,
+        "path": "model-00015-of-00018.safetensors",
+        "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22"
+      },
+      {
+        "bytes": 3979564040,
+        "path": "model-00016-of-00018.safetensors",
+        "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"
+      },
+      {
+        "bytes": 2108759344,
+        "path": "model-00017-of-00018.safetensors",
+        "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8"
+      },
+      {
+        "bytes": 3392197344,
+        "path": "model-00018-of-00018.safetensors",
+        "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"
+      },
+      {
+        "bytes": 112216,
+        "path": "model.safetensors.index.json",
+        "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+      },
+      {
+        "bytes": 390,
+        "path": "preprocessor_config.json",
+        "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
+      },
+      {
+        "bytes": 12809320,
+        "path": "tokenizer.json",
+        "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
+      },
+      {
+        "bytes": 17928,
+        "path": "tokenizer_config.json",
+        "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
+      },
+      {
+        "bytes": 385,
+        "path": "video_preprocessor_config.json",
+        "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
+      },
+      {
+        "bytes": 6722759,
+        "path": "vocab.json",
+        "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
+      }
+    ]
+  },
+  "source_plan": {
+    "bytes": 15423,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/source-plan.json",
+    "sha256": "658280444d4e3d1b6a3a8f2ce4b97e4be7cbff2c8a6ea7c66b3c733c78f31638"
+  },
+  "step0_archive": {
+    "bytes": 46720712,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-0/actual-gdn.safetensors",
+    "sha256": "360ec87dbaaa0647ec72f4959d869e7d04fa1e5eed1fe7ab5b91b22bdc33b29a"
+  },
+  "step0_metadata": {
+    "bytes": 84104,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-0/prevalidation.json",
+    "sha256": "17bd8f5430931cb33a22af32ac056f7357b02df32c9224d1c67a6361be20f46b"
+  },
+  "step1_archive": {
+    "bytes": 27153160,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-1/actual-gdn.safetensors",
+    "sha256": "907044c0889af975d573cb0bbacc0e8f3ac1212aa60822c36bc770d9868de038"
+  },
+  "step1_metadata": {
+    "bytes": 74924,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-1/prevalidation.json",
+    "sha256": "7ceceff0e776b7c4f1c97074de7b065264b69e374397f97bc9f1535f4517dc60"
+  },
+  "steps": [
+    {
+      "actual_norm_dispatch": [
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.013.args.0"
+            },
+            {
+              "torch_value": "torch.float32"
+            }
+          ],
+          "index": 13,
+          "keyword_arguments": {},
+          "operation": "aten.to.dtype",
+          "result": {
+            "tensor": "dispatch.013.result"
+          },
+          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
+          "scope": "norm",
+          "source_line": 227
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.014.args.0"
+            },
+            2
+          ],
+          "index": 14,
+          "keyword_arguments": {},
+          "operation": "aten.pow.Tensor_Scalar",
+          "result": {
+            "tensor": "dispatch.014.result"
+          },
+          "schema": "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor",
+          "scope": "norm",
+          "source_line": 228
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.015.args.0"
+            },
+            [
+              -1
+            ],
+            true
+          ],
+          "index": 15,
+          "keyword_arguments": {},
+          "operation": "aten.mean.dim",
+          "result": {
+            "tensor": "dispatch.015.result"
+          },
+          "schema": "aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
+          "scope": "norm",
+          "source_line": 228
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.016.args.0"
+            },
+            1e-06
+          ],
+          "index": 16,
+          "keyword_arguments": {},
+          "operation": "aten.add.Tensor",
+          "result": {
+            "tensor": "dispatch.016.result"
+          },
+          "schema": "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
+          "scope": "norm",
+          "source_line": 230
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.017.args.0"
+            }
+          ],
+          "index": 17,
+          "keyword_arguments": {},
+          "operation": "aten.rsqrt.default",
+          "result": {
+            "tensor": "dispatch.017.result"
+          },
+          "schema": "aten::rsqrt(Tensor self) -> Tensor",
+          "scope": "norm",
+          "source_line": 230
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.018.args.0"
+            },
+            {
+              "tensor": "dispatch.018.args.1"
+            }
+          ],
+          "index": 18,
+          "keyword_arguments": {},
+          "operation": "aten.mul.Tensor",
+          "result": {
+            "tensor": "dispatch.018.result"
+          },
+          "schema": "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
+          "scope": "norm",
+          "source_line": 230
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.019.args.0"
+            },
+            {
+              "torch_value": "torch.bfloat16"
+            }
+          ],
+          "index": 19,
+          "keyword_arguments": {},
+          "operation": "aten.to.dtype",
+          "result": {
+            "tensor": "dispatch.019.result"
+          },
+          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
+          "scope": "norm",
+          "source_line": 231
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.020.args.0"
+            },
+            {
+              "tensor": "dispatch.020.args.1"
+            }
+          ],
+          "index": 20,
+          "keyword_arguments": {},
+          "operation": "aten.mul.Tensor",
+          "result": {
+            "tensor": "dispatch.020.result"
+          },
+          "schema": "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
+          "scope": "norm",
+          "source_line": 231
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.021.args.0"
+            },
+            {
+              "torch_value": "torch.float32"
+            }
+          ],
+          "index": 21,
+          "keyword_arguments": {},
+          "operation": "aten.to.dtype",
+          "result": {
+            "tensor": "dispatch.021.result"
+          },
+          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
+          "scope": "norm",
+          "source_line": 232
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.022.args.0"
+            }
+          ],
+          "index": 22,
+          "keyword_arguments": {},
+          "operation": "aten.silu.default",
+          "result": {
+            "tensor": "dispatch.022.result"
+          },
+          "schema": "aten::silu(Tensor self) -> Tensor",
+          "scope": "norm",
+          "source_line": 232
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.023.args.0"
+            },
+            {
+              "tensor": "dispatch.023.args.1"
+            }
+          ],
+          "index": 23,
+          "keyword_arguments": {},
+          "operation": "aten.mul.Tensor",
+          "result": {
+            "tensor": "dispatch.023.result"
+          },
+          "schema": "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
+          "scope": "norm",
+          "source_line": 232
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.024.args.0"
+            },
+            {
+              "torch_value": "torch.bfloat16"
+            }
+          ],
+          "index": 24,
+          "keyword_arguments": {},
+          "operation": "aten.to.dtype",
+          "result": {
+            "tensor": "dispatch.024.result"
+          },
+          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
+          "scope": "norm",
+          "source_line": 234
+        }
+      ],
+      "semantic_frame_return": "norm.return.actual_return_0",
+      "semantic_frame_weighted": "norm.weighted_bf16.hidden_states",
+      "step": 0,
+      "teacher_token": 11,
+      "tensors": {
+        "core_bf16": {
+          "bytes": 12288,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "norm.input.hidden_states",
+          "numel": 6144,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "7437ae8be137151718482d6121e0a515cce65602c7e90518dc5dd6136c754353",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "gamma_bf16": {
+          "bytes": 256,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "norm.input.weight",
+          "numel": 128,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "70d11f294923a94aac97bedd27e596dbf496746c6b9e3b2b7c38798a0d2522cc",
+          "shape": [
+            128
+          ],
+          "stride": [
+            1
+          ]
+        },
+        "gate_bf16": {
+          "bytes": 12288,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "norm.input.gate",
+          "numel": 6144,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "d1775651efb85f84d09e472b13993f5f69792f0f7d6872e277b2bbb072350d1b",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "inverse_f32": {
+          "bytes": 192,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "F32",
+          "name": "dispatch.017.result",
+          "numel": 48,
+          "original_torch_dtype": "torch.float32",
+          "raw_sha256": "3a38fa72cb42b37858aab0701f7159d14eade048ffa98197c100c4879f31ace4",
+          "shape": [
+            48,
+            1
+          ],
+          "stride": [
+            1,
+            1
+          ]
+        },
+        "mean_f32": {
+          "bytes": 192,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "F32",
+          "name": "dispatch.015.result",
+          "numel": 48,
+          "original_torch_dtype": "torch.float32",
+          "raw_sha256": "8d978e5e7bca018d4a187190004da41f44dc6e31880bb06288742fc732c406c8",
+          "shape": [
+            48,
+            1
+          ],
+          "stride": [
+            1,
+            1
+          ]
+        },
+        "normalized_bf16": {
+          "bytes": 12288,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "dispatch.019.result",
+          "numel": 6144,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "808f25c5e2f016e6b3d0367b099c03fc5af4e13a5c149ce26cced98fdf8571d3",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "output_bf16": {
+          "bytes": 12288,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "dispatch.024.result",
+          "numel": 6144,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "56e86c82d80ff8bb029ff113cfe65571ddb5f78f5940d9f12e244c3684580323",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "product_f32": {
+          "bytes": 24576,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "F32",
+          "name": "dispatch.023.result",
+          "numel": 6144,
+          "original_torch_dtype": "torch.float32",
+          "raw_sha256": "cd3cece2d490f05a249a9886f7c8057b71a0ce3cf1e6e2b18ea88c2c758de009",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "silu_f32": {
+          "bytes": 24576,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "F32",
+          "name": "dispatch.022.result",
+          "numel": 6144,
+          "original_torch_dtype": "torch.float32",
+          "raw_sha256": "22dcbb4b31b1cda450692b4d6872c14491e07ff238f2f8eb8b7d75c6eee7ad28",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "weighted_bf16": {
+          "bytes": 12288,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "dispatch.020.result",
+          "numel": 6144,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "8c93f3ff3ad59a261ca08998feb91aa8502628feada7b7d09b21296205ac5bfe",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        }
+      }
+    },
+    {
+      "actual_norm_dispatch": [
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.009.args.0"
+            },
+            {
+              "torch_value": "torch.float32"
+            }
+          ],
+          "index": 9,
+          "keyword_arguments": {},
+          "operation": "aten.to.dtype",
+          "result": {
+            "tensor": "dispatch.009.result"
+          },
+          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
+          "scope": "norm",
+          "source_line": 227
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.010.args.0"
+            },
+            2
+          ],
+          "index": 10,
+          "keyword_arguments": {},
+          "operation": "aten.pow.Tensor_Scalar",
+          "result": {
+            "tensor": "dispatch.010.result"
+          },
+          "schema": "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor",
+          "scope": "norm",
+          "source_line": 228
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.011.args.0"
+            },
+            [
+              -1
+            ],
+            true
+          ],
+          "index": 11,
+          "keyword_arguments": {},
+          "operation": "aten.mean.dim",
+          "result": {
+            "tensor": "dispatch.011.result"
+          },
+          "schema": "aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
+          "scope": "norm",
+          "source_line": 228
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.012.args.0"
+            },
+            1e-06
+          ],
+          "index": 12,
+          "keyword_arguments": {},
+          "operation": "aten.add.Tensor",
+          "result": {
+            "tensor": "dispatch.012.result"
+          },
+          "schema": "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
+          "scope": "norm",
+          "source_line": 230
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.013.args.0"
+            }
+          ],
+          "index": 13,
+          "keyword_arguments": {},
+          "operation": "aten.rsqrt.default",
+          "result": {
+            "tensor": "dispatch.013.result"
+          },
+          "schema": "aten::rsqrt(Tensor self) -> Tensor",
+          "scope": "norm",
+          "source_line": 230
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.014.args.0"
+            },
+            {
+              "tensor": "dispatch.014.args.1"
+            }
+          ],
+          "index": 14,
+          "keyword_arguments": {},
+          "operation": "aten.mul.Tensor",
+          "result": {
+            "tensor": "dispatch.014.result"
+          },
+          "schema": "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
+          "scope": "norm",
+          "source_line": 230
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.015.args.0"
+            },
+            {
+              "torch_value": "torch.bfloat16"
+            }
+          ],
+          "index": 15,
+          "keyword_arguments": {},
+          "operation": "aten.to.dtype",
+          "result": {
+            "tensor": "dispatch.015.result"
+          },
+          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
+          "scope": "norm",
+          "source_line": 231
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.016.args.0"
+            },
+            {
+              "tensor": "dispatch.016.args.1"
+            }
+          ],
+          "index": 16,
+          "keyword_arguments": {},
+          "operation": "aten.mul.Tensor",
+          "result": {
+            "tensor": "dispatch.016.result"
+          },
+          "schema": "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
+          "scope": "norm",
+          "source_line": 231
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.017.args.0"
+            },
+            {
+              "torch_value": "torch.float32"
+            }
+          ],
+          "index": 17,
+          "keyword_arguments": {},
+          "operation": "aten.to.dtype",
+          "result": {
+            "tensor": "dispatch.017.result"
+          },
+          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
+          "scope": "norm",
+          "source_line": 232
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.018.args.0"
+            }
+          ],
+          "index": 18,
+          "keyword_arguments": {},
+          "operation": "aten.silu.default",
+          "result": {
+            "tensor": "dispatch.018.result"
+          },
+          "schema": "aten::silu(Tensor self) -> Tensor",
+          "scope": "norm",
+          "source_line": 232
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.019.args.0"
+            },
+            {
+              "tensor": "dispatch.019.args.1"
+            }
+          ],
+          "index": 19,
+          "keyword_arguments": {},
+          "operation": "aten.mul.Tensor",
+          "result": {
+            "tensor": "dispatch.019.result"
+          },
+          "schema": "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
+          "scope": "norm",
+          "source_line": 232
+        },
+        {
+          "arguments": [
+            {
+              "tensor": "dispatch.020.args.0"
+            },
+            {
+              "torch_value": "torch.bfloat16"
+            }
+          ],
+          "index": 20,
+          "keyword_arguments": {},
+          "operation": "aten.to.dtype",
+          "result": {
+            "tensor": "dispatch.020.result"
+          },
+          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
+          "scope": "norm",
+          "source_line": 234
+        }
+      ],
+      "semantic_frame_return": "norm.return.actual_return_0",
+      "semantic_frame_weighted": "norm.weighted_bf16.hidden_states",
+      "step": 1,
+      "teacher_token": 353,
+      "tensors": {
+        "core_bf16": {
+          "bytes": 12288,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "norm.input.hidden_states",
+          "numel": 6144,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "2484b58779bee4063507010b4e69061ee135e1d084c460d33c0fbd06590ce97e",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "gamma_bf16": {
+          "bytes": 256,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "norm.input.weight",
+          "numel": 128,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "70d11f294923a94aac97bedd27e596dbf496746c6b9e3b2b7c38798a0d2522cc",
+          "shape": [
+            128
+          ],
+          "stride": [
+            1
+          ]
+        },
+        "gate_bf16": {
+          "bytes": 12288,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "norm.input.gate",
+          "numel": 6144,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "bf9d167dcd86dc25cc65a23b8a6a2fa3f1d800a6932b6046f30f605479622e9d",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "inverse_f32": {
+          "bytes": 192,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "F32",
+          "name": "dispatch.013.result",
+          "numel": 48,
+          "original_torch_dtype": "torch.float32",
+          "raw_sha256": "52c3212e739d8e9744fc2816ab8af88527ce7f0668075c0ab1d89c44882305ad",
+          "shape": [
+            48,
+            1
+          ],
+          "stride": [
+            1,
+            1
+          ]
+        },
+        "mean_f32": {
+          "bytes": 192,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "F32",
+          "name": "dispatch.011.result",
+          "numel": 48,
+          "original_torch_dtype": "torch.float32",
+          "raw_sha256": "b097a160c68b757bf35a3be5469a8f3ca3433397e59a8e8efad3f3e6e64b26eb",
+          "shape": [
+            48,
+            1
+          ],
+          "stride": [
+            1,
+            1
+          ]
+        },
+        "normalized_bf16": {
+          "bytes": 12288,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "dispatch.015.result",
+          "numel": 6144,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "bbeac6bdd7cf11424e51100e0b27fd7fe7b879addbaa6376e2a650268a34be90",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "output_bf16": {
+          "bytes": 12288,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "dispatch.020.result",
+          "numel": 6144,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "74389fd9fec7efb8281d676207ea979af3900beb64bf8d9e41d4c28c5fdc321a",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "product_f32": {
+          "bytes": 24576,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "F32",
+          "name": "dispatch.019.result",
+          "numel": 6144,
+          "original_torch_dtype": "torch.float32",
+          "raw_sha256": "fb023a0c354a89ef8efa062c1789028defc1930c5b7022d18f60ae848c4851f3",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "silu_f32": {
+          "bytes": 24576,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "F32",
+          "name": "dispatch.018.result",
+          "numel": 6144,
+          "original_torch_dtype": "torch.float32",
+          "raw_sha256": "a2bbc24cdb0451f529649ef8cc8e4693c75e92fac73f884eef192b1fc87bd1db",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        },
+        "weighted_bf16": {
+          "bytes": 12288,
+          "contiguous": true,
+          "device": "cpu",
+          "dtype": "BF16",
+          "name": "dispatch.016.result",
+          "numel": 6144,
+          "original_torch_dtype": "torch.bfloat16",
+          "raw_sha256": "e1e1cd73fc4546028e36285501a2410519a81d9ddb9ee16a277da247a0eeb350",
+          "shape": [
+            48,
+            128
+          ],
+          "stride": [
+            128,
+            1
+          ]
+        }
+      }
+    }
+  ]
+}
--- /dev/null
+++ b/source-pin.json
@@ -0,0 +1,209 @@
+{
+  "classification": "authenticated_cache_source_only_not_model_execution",
+  "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
+  "cache": "/home/pvl/.cache/abird/zerogravity/models",
+  "reference_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
+  "repository": "Qwen/Qwen3.8-27B",
+  "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
+  "selected_bytes": 55586113293,
+  "weight_bytes": 55563006776,
+  "payload_bytes": 55562855904,
+  "files": [
+    {
+      "name": "LICENSE",
+      "bytes": 11544,
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
+      "git_blob_verified": true
+    },
+    {
+      "name": "README.md",
+      "bytes": 65012,
+      "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641",
+      "git_blob_verified": true
+    },
+    {
+      "name": "chat_template.jinja",
+      "bytes": 8952,
+      "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
+      "git_blob_verified": true
+    },
+    {
+      "name": "config.json",
+      "bytes": 4312,
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
+      "git_blob_verified": true
+    },
+    {
+      "name": "crc32.txt",
+      "bytes": 238,
+      "sha256": "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83",
+      "git_blob_verified": true
+    },
+    {
+      "name": "generation_config.json",
+      "bytes": 202,
+      "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
+      "git_blob_verified": true
+    },
+    {
+      "name": "merges.txt",
+      "bytes": 3353259,
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
+      "git_blob_verified": true
+    },
+    {
+      "name": "model-00001-of-00018.safetensors",
+      "bytes": 3966730552,
+      "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00002-of-00018.safetensors",
+      "bytes": 3043080328,
+      "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00003-of-00018.safetensors",
+      "bytes": 2542796952,
+      "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00004-of-00018.safetensors",
+      "bytes": 3988973152,
+      "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00005-of-00018.safetensors",
+      "bytes": 2099339864,
+      "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00006-of-00018.safetensors",
+      "bytes": 3979553696,
+      "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00007-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00008-of-00018.safetensors",
+      "bytes": 3979553696,
+      "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00009-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00010-of-00018.safetensors",
+      "bytes": 3979553696,
+      "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00011-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00012-of-00018.safetensors",
+      "bytes": 3979553696,
+      "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00013-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00014-of-00018.safetensors",
+      "bytes": 3979553696,
+      "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00015-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00016-of-00018.safetensors",
+      "bytes": 3979564040,
+      "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00017-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00018-of-00018.safetensors",
+      "bytes": 3392197344,
+      "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model.safetensors.index.json",
+      "bytes": 112216,
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
+      "git_blob_verified": true
+    },
+    {
+      "name": "preprocessor_config.json",
+      "bytes": 390,
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
+      "git_blob_verified": true
+    },
+    {
+      "name": "tokenizer.json",
+      "bytes": 12809320,
+      "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
+      "git_blob_verified": false
+    },
+    {
+      "name": "tokenizer_config.json",
+      "bytes": 17928,
+      "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
+      "git_blob_verified": true
+    },
+    {
+      "name": "video_preprocessor_config.json",
+      "bytes": 385,
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
+      "git_blob_verified": true
+    },
+    {
+      "name": "vocab.json",
+      "bytes": 6722759,
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
+      "git_blob_verified": true
+    }
+  ],
+  "tensor_count": 1199,
+  "ordinary_text_tensors": 851,
+  "other_tensors": 348,
+  "ordinary_text_bytes": 53791996928,
+  "full_file_sha256s_independent": true,
+  "ordinary_source_shapes_match_config": true,
+  "headers_match_preflight": true,
+  "buffer_bytes": 1048576,
+  "statvfs_free_after": 544125792256,
+  "finished_utc": "2026-10-19T15:50:57Z"
+}
--- /dev/null
+++ b/test_contracts.py
@@ -0,0 +1,99 @@
+"""Prepared source-free transport/validator fixtures; ROOT executes, author did not."""
+import hashlib
+import json
+from pathlib import Path
+import struct
+import tempfile
+import unittest
+from unittest.mock import patch
+from contracts import RawArchive, finite, read_json, unique, widened
+from validate_replay import comparison, exact_int
+
+def archive_file(directory, raw=b'\x00\x80\x01\x00', shape=None):
+    shape = shape or [2]
+    header = json.dumps({'value':{'dtype':'BF16','shape':shape,'data_offsets':[0,len(raw)]}},separators=(',',':')).encode()
+    data = len(header).to_bytes(8,'little')+header+raw
+    path = directory/'archive.safetensors'
+    path.write_bytes(data)
+    pin = dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
+    row = dict(name='value',dtype='BF16',shape=shape,bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest())
+    return pin,row
+
+class Contracts(unittest.TestCase):
+    def test_signed_zero_and_subnormal_transport_is_bit_exact(self):
+        self.assertEqual(widened(b'\x00\x80\x01\x00','BF16'),bytes.fromhex('0000008000000100'))
+        raw = bytes.fromhex('0000008001000000')
+        self.assertEqual(widened(raw,'F32'),raw)
+    def test_nonfinite_dtypes_are_not_filtered_or_cast(self):
+        for dtype,raw in [('BF16',bytes.fromhex('807f')),('F32',bytes.fromhex('0100c07f'))]:
+            with self.assertRaises(ValueError):
+                finite(raw,dtype)
+    def test_header_declared_size_is_refused_before_large_read(self):
+        with tempfile.TemporaryDirectory() as temporary:
+            directory = Path(temporary)
+            path = directory/'bad.safetensors'
+            raw = (2*1024**2).to_bytes(8,'little')+b'0'*8
+            path.write_bytes(raw)
+            row = dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
+            import os
+            original,seen = os.pread,[]
+            def observed(fd,count,offset):
+                seen.append(count)
+                return original(fd,count,offset)
+            with patch('contracts.os.pread',observed):
+                with self.assertRaises(ValueError):
+                    with RawArchive(row):
+                        self.fail('invalid declared header accepted')
+            self.assertEqual(seen,[8])
+    def test_same_fd_raw_signed_bits_and_post_read_tamper_refusal(self):
+        with tempfile.TemporaryDirectory() as temporary:
+            directory = Path(temporary)
+            pin,row = archive_file(directory)
+            with RawArchive(pin) as archive:
+                self.assertEqual(archive.raw(row),b'\x00\x80\x01\x00')
+            with self.assertRaises(ValueError):
+                with RawArchive(pin):
+                    path = Path(pin['path'])
+                    raw = bytearray(path.read_bytes())
+                    raw[-1] ^= 1
+                    path.write_bytes(raw)
+    def test_wrong_dtype_shape_hash_and_short_coverage_refuse(self):
+        with tempfile.TemporaryDirectory() as temporary:
+            pin,row = archive_file(Path(temporary))
+            with RawArchive(pin) as archive:
+                for changed in [dict(row,dtype='F32'),dict(row,shape=[1,2]),dict(row,raw_sha256='0'*64),dict(row,bytes=2)]:
+                    with self.assertRaises(ValueError):
+                        archive.raw(changed)
+    def test_duplicate_json_and_boolean_counts_are_rejected(self):
+        with self.assertRaises(ValueError):
+            unique([('key',1),('key',2)])
+        for count in [False,True,0.0,'0']:
+            with self.assertRaises(ValueError):
+                exact_int(count,0)
+    def test_independent_bit_check_rejects_signed_zero_fabricated_success(self):
+        with tempfile.TemporaryDirectory() as temporary:
+            directory = Path(temporary)
+            pin,row = archive_file(directory,raw=b'\x00\x80',shape=[1])
+            raw = bytes.fromhex('00000000')
+            path = directory/'case.output_bf16.f32le'
+            path.write_bytes(raw)
+            record = dict(label='case.output_bf16',reference=row,different_words=0,first_difference=None,
+                          max_abs=0,file=path.name,words=1,bytes=4,sha256=hashlib.sha256(raw).hexdigest())
+            with RawArchive(pin) as archive:
+                with self.assertRaises(ValueError):
+                    comparison([record],['output_bf16'],'case.',{'output_bf16':row},archive,directory)
+    def test_full_word_coverage_precedes_zip(self):
+        with tempfile.TemporaryDirectory() as temporary:
+            directory = Path(temporary)
+            pin,row = archive_file(directory)
+            raw = widened(bytes.fromhex('00800100'),'BF16')
+            path = directory/'case.output_bf16.f32le'
+            path.write_bytes(raw)
+            record = dict(label='case.output_bf16',reference=row,different_words=0,first_difference=None,
+                          max_abs=0,file=path.name,words=1,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
+            with RawArchive(pin) as archive:
+                with self.assertRaises(ValueError):
+                    comparison([record],['output_bf16'],'case.',{'output_bf16':row},archive,directory)
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ b/validate_replay.py
@@ -0,0 +1,178 @@
+"""ROOT independent full signed-bit/raw-archive/source/ledger/lifetime validation."""
+import argparse
+import json
+import hashlib
+import os
+from pathlib import Path
+import stat
+from contracts import (HERE, INPUTS, OUTPUTS, RawArchive, frozen, identity, implementation, read,
+                       read_json, require, sha, widened, write)
+from prepare_reference import authenticate_metadata, validate as validate_metadata
+
+def exact_int(value, expected):
+    require(type(value) is int and value == expected, 'exact nonbool integer count')
+
+def affinity(text):
+    require(type(text) is str and text.startswith('Cpus_allowed_list:'), 'actual proc affinity')
+    values = []
+    for item in text.split(':',1)[1].strip().split(','):
+        span = item.split('-')
+        require(1 <= len(span) <= 2 and all(v.isdecimal() for v in span), 'strict CPU span')
+        lo, hi = int(span[0]), int(span[-1])
+        require(0 <= lo <= hi < 4096, 'bounded CPU span')
+        values.extend(range(lo,hi+1))
+    require(len(values) == 4 and values == sorted(set(values)), 'four actual distinct CPUs')
+    return values
+
+def tree(directory, reserve=0):
+    paths = list(directory.rglob('*'))
+    require(len(paths) <= 64, 'bounded native diagnostic tree entries')
+    total = 0
+    for path in paths:
+        info = path.lstat()
+        require(not path.is_symlink() and (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)),
+                'owned ordinary diagnostic tree')
+        if stat.S_ISREG(info.st_mode):
+            total += info.st_size
+    require(total+reserve <= 8*1024**2, 'bounded complete diagnostic artifact tree')
+    return total
+
+def comparison(records, names, prefix, rows, archive, directory):
+    require(type(records) is list and len(records) == len(names), 'complete comparison records')
+    files, words = set(), 0
+    for record, name in zip(records, names):
+        label = prefix+name
+        expected = widened(archive.raw(rows[name]), rows[name]['dtype'])
+        require(record['label'] == label and record['reference'] == rows[name] and
+                record['different_words'] == 0 and record['first_difference'] is None and
+                record['max_abs'] == 0 and record['file'] == label+'.f32le',
+                'bound actual stage/order/source/counters: '+label)
+        exact_int(record['words'], len(expected)//4)
+        exact_int(record['bytes'], len(expected))
+        require(read(directory/record['file'],24576,record) == expected,
+                'independent every raw signed output bit: '+label)
+        files.add(record['file'])
+        words += record['words']
+    return files, words
+
+def original_gamma(binding, source):
+    name = 'model-00001-of-00018.safetensors'
+    pin = next(r for r in binding['source']['source_files'] if r['path'] == name)
+    path = Path(binding['source']['source_directory'])/name
+    prior = next(r for r in source['source_after'] if r['path'] == str(path))
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as opened:
+        before = os.fstat(opened.fileno())
+        require(stat.S_ISREG(before.st_mode) and before.st_size == pin['bytes'] and
+                identity(before) == tuple(prior['fd_identity']) and
+                hashlib.file_digest(opened,'sha256').hexdigest() == pin['sha256'],
+                'independent original gamma shard full hash/FD identity before selected source read')
+        raw = os.pread(opened.fileno(),256,binding['gamma_original_tensor']['absolute_offset'])
+        require(len(raw) == 256 and identity(os.fstat(opened.fileno())) == identity(before),
+                'complete original gamma extent on same unchanged authenticated shard FD')
+    return raw
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--output', type=Path, required=True)
+    parser.add_argument('--reference-directory', type=Path, required=True)
+    parser.add_argument('--implementation', type=Path, required=True)
+    args = parser.parse_args()
+    require(args.output.resolve().parent == args.reference_directory.resolve().parent == HERE,
+            'owned attempt and reference only')
+    freeze, build = frozen(), implementation(args.implementation)
+    metadata, binding = validate_metadata(args.reference_directory), authenticate_metadata()
+    done = read_json(args.output/'completed.json')
+    require(done['status'] == 'EXECUTED_REQUIRES_INDEPENDENT_RAW_VALIDATION' and
+            done['freeze_manifest_sha256'] == freeze and
+            done['ROOT_build_source_attestation_sha256'] == build and
+            done['archived_reference_validation'] == metadata and
+            done['full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified'] is False,
+            'actual frozen build/source/reference receipt closure')
+    directory, lifecycle = args.output/'27b', args.output/'27b-lifecycle'
+    for file,key in [('report.json','report_sha256'),('source-after.json','source_after_sha256')]:
+        require(sha(directory/file) == done[key], 'actual report/source receipt digest')
+    require(sha(lifecycle/'invocation.json') == done['invocation_sha256'], 'actual owned native invocation digest')
+    report, invocation = read_json(directory/'report.json'), read_json(lifecycle/'invocation.json')
+    require(report['status'] == 'PASS_PRIVATE_ORIGINAL_H48_BF16_GAMMA_GATED_RMS_ONLY' and
+            report['inputs_sha256'] == metadata['inputs_sha256'] and report['reference_sha256'] == metadata['reference_sha256'] and
+            report['full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified'] is False and
+            report['metadata_and_payload_admitted_before_native_extraction'] is True and
+            report['gamma_actual_source_codec_extent_raw_identity'] is True, 'same private original operator only')
+    for key,count in [('primitive_stage_checks',14),('primitive_stage_words',61632),('typed_final_checks',2),
+                      ('typed_final_words',12288),('retained_input_words',24576)]:
+        exact_int(report[key],count)
+    require(invocation['status'] == 'CHILD_EXECUTED_REQUIRES_VALIDATION' and
+            invocation['binary_sha256'] == done['binary_sha256'] and invocation['freeze_manifest_sha256'] == freeze and
+            invocation['primary_error'] is None and invocation['cleanup_errors'] == [] and
+            invocation['exit_code'] == invocation['reaped_exit_code'] == 0 and
+            all(invocation[k] is True for k in ('leader_reaped','group_absent','pipe_eof','selector_closed','pipe_closed','log_closed')) and
+            affinity(report['actual_affinity']) == invocation['cpu_affinity'], 'original actual owned child/lifetime/CPU proof')
+    source = read_json(directory/'source-after.json')
+    require(source['all_equal'] is True and source['source_before'] == source['source_after'] == invocation['source_before'],
+            'whole original trained source before/after actual immutable')
+    sb = report['source_binding']
+    require(type(sb['identity']) is str and sb['identity'].startswith('SourceIdentity(') and sb['identity'] != 'SourceIdentity(0)' and
+            sb['verified_index_sha256'] == '77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df' and
+            sb['verified_bundle_sha256'] == '7e6a80bc4fc06b6237914007bde59a48a6e4fb0ea570bd5078b3101eb169faf2' and
+            sb['complete_text_roles'] == 851, 'actual original complete source issuer/index/bundle')
+    mib = 1024**2
+    needed = 2*6144*4+64*1024+6144*4+960*1024
+    steady, live, retained, final = [report[k+'_ledger'] for k in ('steady','live','retained','final')]
+    require(steady['reserved'] == 80*mib and steady['reservations'] == 4 and
+            steady['per_class'].get('Other',0) == 8*mib and steady['per_class'].get('Scratch',0) == 72*mib,
+            'original steady source/binding/control before archive extraction')
+    require(live['reserved'] == 80*mib+2*needed and retained['reserved'] == 72*mib+2*needed,
+            'both actual typed input/output scratch/source owners retained')
+    for row in (live,retained):
+        require(row['per_class'].get('Output',0) == 2*3*6144*4 and
+                row['per_class'].get('TransferStaging',0) == 0 and row['fenced'] is False, 'actual admitted payload/retirement')
+    require(final['reserved'] == final['reservations'] == 0 and final['fenced'] is False, 'actual lastdrop original ledger0')
+    require(report['negative'] == [dict(step=i,pre_cancel=True,wrong_layer=True,real_one_byte_short_aggregate_refusal=True)
+                                   for i in range(2)], 'actual real-pool/cancel/wrong-layer proof for both teachers')
+    require(len(report['primitive']) == len(report['comparisons']) == 2 and
+            report['retained_comparisons'] == report['comparisons'], 'separate two-phase primitive/typed/retained proof')
+    actual = read_json(directory/'comparisons.json')
+    require(actual == dict(primitive=report['primitive'], typed=report['comparisons'], different_words=0), 'actual comparison receipt')
+    files = {'report.json','source-after.json','comparisons.json'}
+    stage_words, typed_words = 0, 0
+    gamma_raw = original_gamma(binding, source)
+    for step in range(2):
+        with RawArchive(binding[f'step{step}_archive']) as archive:
+            rows = binding['steps'][step]['tensors']
+            require(archive.raw(rows['gamma_bf16']) == gamma_raw, 'independent actual original gamma bytes equal both public calls')
+            primitive, typed = report['primitive'][step], report['comparisons'][step]
+            require(primitive['step'] == typed['step'] == step and primitive['words'] == 30816 and
+                    primitive['different_words'] == typed['different_words'] == 0 and primitive['diagnostic_scratch_bytes'] == mib and
+                    typed['output_words'] == 6144 and typed['retained_input_words'] == 12288,
+                    'per-phase exact stage/type counts and admitted diagnostic envelope')
+            gamma = primitive['gamma_source']
+            require(gamma['offset'] == binding['gamma_original_tensor']['absolute_offset'] == 168892952 and
+                    gamma['bytes'] == 256 and gamma['shard'] == 'model-00001-of-00018.safetensors' and
+                    gamma['shard_sha256'] == next(r['sha256'] for r in binding['source']['source_files'] if r['path'] == gamma['shard']) and
+                    gamma['bundle_sha256'] == sb['verified_bundle_sha256'],
+                    'actual gamma raw extent/shard/bundle remains original, never a proxy tensor')
+            current, words = comparison(primitive['checks'],OUTPUTS,f'step{step}.stage.',rows,archive,directory)
+            files |= current
+            stage_words += words
+            current, words = comparison(typed['checks'],('core_bf16','gate_bf16','output_bf16'),
+                                        f'step{step}.typed.',rows,archive,directory)
+            files |= current
+            typed_words += words
+    require(stage_words == 61632 and typed_words == 36864 and
+            {p.name for p in directory.iterdir()} == files and
+            {p.name for p in lifecycle.iterdir()} == {'execution.log','invocation.json'},
+            'all7 actual primitive arrays and typed final+inputs, complete raw artifact tree')
+    tree(args.output,reserve=65536)
+    value = dict(status='PASS_PRIVATE_ORIGINAL27B_H48_BF16_GAMMA_GATED_RMS_ONLY',
+        primitive_stage_checks=14, primitive_stage_words=61632, independent_typed_final_checks=2,
+        independent_typed_final_words=12288, retained_input_words=24576,
+        completed_sha256=sha(args.output/'completed.json'), freeze_manifest_sha256=freeze,
+        ROOT_build_source_attestation_sha256=build,
+        full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified=False)
+    require(frozen() == freeze and implementation(args.implementation) == build, 'all prepared/build sources unchanged')
+    write(args.output/'validation.json',value)
+    tree(args.output)
+    print(json.dumps(value,sort_keys=True))
+
+if __name__ == '__main__':
+    main()

````

