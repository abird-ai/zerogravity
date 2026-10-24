# Draft Patches — volume 9

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 52773; SHA256 `b53625fa619545bfa1a8604ca000bd08a7d2a2a41a9bff0aac99116e6fbfceea`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-artifacts/src/source_weights/tensor_window/conv_parameter.rs
@@ -0,0 +1,36 @@
+//! Sealed original BF16 depthwise four-tap parameter transport only.
+use super::*;
+impl VerifiedTensorSource {
+    /// Original rank3 metadata and raw elements, without model eligibility.
+    /// `parameter()` remains rank1-only; no synthetic flattened tensor exists.
+    pub fn conv4_parameter(&self, name: &str) -> Result<VerifiedParameter> {
+        let location = self
+            .0
+            .tensors
+            .get(name)
+            .ok_or_else(|| Error::Malformed("original conv4 parameter missing".into()))?;
+        let [channels, 1, 4] = location.info.shape.as_slice() else {
+            return Err(Error::Unsupported("original conv4 raw rank/shape".into()));
+        };
+        let channels = usize::try_from(*channels)
+            .ok()
+            .filter(|n| *n > 0 && *n <= 16384 && n.is_multiple_of(32))
+            .ok_or_else(|| Error::Limit("original conv4 channel bound".into()))?;
+        let elements = channels
+            .checked_mul(4)
+            .ok_or_else(|| Error::Limit("original conv4 element count".into()))?;
+        let bytes = elements
+            .checked_mul(2)
+            .ok_or_else(|| Error::Limit("original conv4 byte count".into()))?;
+        if location.info.kind != QuantKind::Bf16 || location.info.byte_len != bytes as u64 {
+            return Err(Error::Unsupported("original conv4 raw dtype/length".into()));
+        }
+        Ok(VerifiedParameter {
+            location: location.clone(),
+            owner: self.0.clone(),
+            elements,
+            element_bytes: 2,
+            bytes,
+        })
+    }
+}
--- a/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs
+++ b/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs
@@ -103,7 +103,7 @@
     matrix: VerifiedMatrix,
 }
 
-/// Sealed full original rank1 parameter; no matrix axes or adjusted values.
+/// Sealed full original rank1 or closed conv4 parameter; no adjusted values.
 #[derive(Clone)]
 pub struct VerifiedParameter {
     location: Arc<TensorLocation>,
@@ -493,7 +493,7 @@
             bytes,
         })
     }
-    /// Original Safetensors rank1 is contiguous; arbitrary strides have no API.
+    /// Original parameter elements are contiguous; arbitrary strides have no API.
     pub fn read_parameter(
         &self,
         parameter: &VerifiedParameter,
@@ -802,3 +802,5 @@
         Ok(self.position)
     }
 }
+
+mod conv_parameter;
--- /dev/null
+++ b/crates/zerogravity-artifacts/tests/tensor_window/conv4.rs
@@ -0,0 +1,101 @@
+//! Synthetic transport only; no trained convolution or model qualification.
+use super::*;
+fn fixture(dtype: &str, shape: &[u64], bytes: usize) -> Fixture {
+    let mut fixture = Fixture::new();
+    let raw = (0..bytes / 2)
+        .flat_map(|i| (0x3f80u16 + (i % 8) as u16).to_le_bytes())
+        .collect::<Vec<_>>();
+    fixture.second = safetensors("conv", dtype, shape, &raw);
+    fixture.index = serde_json::to_vec(
+        &json!({"metadata":{"total_size":64+raw.len()},"weight_map":{"matrix":A,"conv":B}}),
+    )
+    .unwrap();
+    fs::write(fixture.path.join(B), &fixture.second).unwrap();
+    fixture
+}
+#[test]
+fn full_original_rank3_and_partial_range_keep_truthful_metadata_and_owner() {
+    let f = fixture("BF16", &[32, 1, 4], 256);
+    let host = admission(64 * 1024 * 1024, "host");
+    let source = f.plan(&host).unwrap().open().unwrap();
+    let before = reserved(&host);
+    assert!(source.parameter("conv").is_err());
+    let raw = source.conv4_parameter("conv").unwrap();
+    let descriptor = raw.source_parameter();
+    assert_eq!(descriptor.shape, [32, 1, 4]);
+    assert_eq!(descriptor.bytes, 256);
+    raw.validate_source_parameter(&descriptor).unwrap();
+    let partial = source.read_parameter(&raw, 4..8).unwrap();
+    assert!(!partial.is_complete());
+    assert_eq!(partial.parameter().tensor().shape, [32, 1, 4]);
+    assert_eq!(partial.element_range(), 4..8);
+    drop(partial);
+    assert_eq!(reserved(&host), before);
+    let full = source.read_parameter(&raw, 0..128).unwrap();
+    assert!(full.is_complete());
+    assert_eq!(full.as_slice().len(), 256);
+    assert!(full.charged_staging_bytes() >= 256);
+    assert!(source.read_parameter(&raw, 0..129).is_err());
+    assert!(source.read_parameter(&raw, 4..4).is_err());
+    let foreign = f.plan(&host).unwrap().open().unwrap();
+    assert!(foreign.read_parameter(&raw, 0..128).is_err());
+    drop(foreign);
+    drop(descriptor);
+    drop(raw);
+    drop(source);
+    assert_eq!(full.as_slice().len(), 256);
+    assert!(reserved(&host) > 0);
+    drop(full);
+    assert_eq!(reserved(&host), 0);
+}
+#[test]
+fn geometry_corruption_cancel_and_pressure_refuse_without_leases() {
+    for (dtype, shape, bytes) in [
+        ("F32", vec![32, 1, 4], 512),
+        ("BF16", vec![128], 256),
+        ("BF16", vec![32, 4, 1], 256),
+        ("BF16", vec![16, 1, 4], 128),
+    ] {
+        let f = fixture(dtype, &shape, bytes);
+        let host = admission(64 * 1024 * 1024, "host");
+        let source = f.plan(&host).unwrap().open().unwrap();
+        assert!(source.conv4_parameter("conv").is_err());
+        drop(source);
+        assert_eq!(reserved(&host), 0);
+    }
+    let f = fixture("BF16", &[32, 1, 4], 256);
+    let host = admission(64 * 1024 * 1024, "host");
+    let source = f.plan(&host).unwrap().open().unwrap();
+    let raw = source.conv4_parameter("conv").unwrap();
+    let baseline = reserved(&host);
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        source
+            .read_parameter_cancellable(&raw, 0..128, &cancelled)
+            .is_err()
+    );
+    assert_eq!(reserved(&host), baseline);
+    let remaining = 64 * 1024 * 1024 - baseline;
+    let pressure = host
+        .arbiter
+        .admit(
+            &host.resource,
+            host.owner.clone(),
+            &[zerogravity_core::resource_arbiter::AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: remaining,
+            }],
+        )
+        .unwrap();
+    assert!(source.read_parameter(&raw, 0..128).is_err());
+    drop(pressure);
+    assert_eq!(reserved(&host), baseline);
+    let mut corrupt = f.second.clone();
+    *corrupt.last_mut().unwrap() ^= 1;
+    fs::write(f.path.join(B), corrupt).unwrap();
+    assert!(source.read_parameter(&raw, 0..128).is_err());
+    assert_eq!(reserved(&host), baseline);
+    drop(raw);
+    drop(source);
+    assert_eq!(reserved(&host), 0);
+}
--- a/crates/zerogravity-artifacts/tests/tensor_window.rs
+++ b/crates/zerogravity-artifacts/tests/tensor_window.rs
@@ -944,3 +944,6 @@
     assert_eq!(reserved(&host), 0);
     assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
 }
+
+#[path = "tensor_window/conv4.rs"]
+mod conv4;
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/gdn_conv.rs
@@ -0,0 +1,228 @@
+//! Closed pinned CPU region; three private BF16 outputs, no state effects.
+use super::*;
+
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum CausalConvSiluContract {
+    TorchCpuBf16Avx2,
+}
+impl CausalConvSiluContract {
+    /// Original raw depthwise `[channel,1,4]`, never an adjusted vector.
+    pub fn validate(
+        self,
+        parameter: &SourceParameter,
+        input: usize,
+        history: usize,
+    ) -> Result<usize> {
+        let channels = input;
+        if !parameter.source.is_bound() || parameter.name.is_empty() {
+            return Err(Error::State("causal convolution source binding".into()));
+        }
+        if channels == 0
+            || channels > 16384
+            || !channels.is_multiple_of(32)
+            || parameter.kind != QuantKind::Bf16
+            || parameter.shape.as_slice() != [channels as u64, 1, 4]
+            || channels.checked_mul(8) != Some(parameter.bytes)
+            || channels.checked_mul(3) != Some(history)
+        {
+            return Err(Error::Unsupported(
+                "closed causal convolution raw geometry/dtype".into(),
+            ));
+        }
+        Ok(channels)
+    }
+}
+
+#[derive(Clone)]
+pub struct CausalConvValues<V> {
+    pub raw: V,
+    pub activated: V,
+    /// Canonical `[channel,3]`, oldest to newest. Caller owns publication.
+    pub next_history: V,
+}
+
+pub(super) fn result_precision(
+    builder: &PlanBuilder,
+    parameter: &SourceParameter,
+    contract: CausalConvSiluContract,
+    inputs: &[PlanValue],
+) -> Result<ActivationPrecision> {
+    if !builder.operator_slice
+        || builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || parameter.source != builder.source
+        || inputs.len() != 2
+        || inputs
+            .iter()
+            .any(|v| v.precision != ActivationPrecision::Bf16WidenedF32)
+    {
+        return Err(Error::Unsupported(
+            "causal convolution profile/source/precision/slice".into(),
+        ));
+    }
+    contract.validate(parameter, inputs[0].elements, inputs[1].elements)?;
+    Ok(ActivationPrecision::Bf16WidenedF32)
+}
+pub(super) fn record(
+    builder: &mut PlanBuilder,
+    parameter: SourceParameter,
+    input: &PlanValue,
+    history: &PlanValue,
+    contract: CausalConvSiluContract,
+) -> Result<CausalConvValues<PlanValue>> {
+    let channels = contract.validate(&parameter, input.elements, history.elements)?;
+    let name = parameter.name.clone();
+    let values = builder.record(
+        &name,
+        PlanOperation::TypedCausalConvSilu {
+            parameter,
+            contract,
+        },
+        &[*input, *history],
+        &[
+            ValueShape::Vector(channels),
+            ValueShape::Vector(channels),
+            ValueShape::Vector(channels * 3),
+        ],
+        None,
+    )?;
+    let [raw, activated, next_history]: [PlanValue; 3] = values
+        .try_into()
+        .map_err(|_| Error::State("causal convolution output arity".into()))?;
+    Ok(CausalConvValues {
+        raw,
+        activated,
+        next_history,
+    })
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    fn fixture() -> (PlanBuilder, SourceParameter) {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        (
+            PlanBuilder::new_operator_slice(
+                source,
+                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            )
+            .unwrap(),
+            SourceParameter {
+                source,
+                name: "original.conv.weight".into(),
+                kind: QuantKind::Bf16,
+                shape: vec![32, 1, 4],
+                bytes: 256,
+            },
+        )
+    }
+    // Synthetic embedding metadata only; no original model producer claim.
+    fn input(builder: &mut PlanBuilder, width: usize) -> PlanValue {
+        builder
+            .embed_profiled(
+                SourceTensor {
+                    source: builder.source,
+                    name: format!("synthetic.{width}"),
+                    kind: QuantKind::Bf16,
+                    axes: MatrixAxes::OutputInput,
+                    columns: width,
+                    rows: 2,
+                },
+                PlanBuilder::inputs().token,
+                EmbeddingContract::Bf16OutputInput,
+            )
+            .unwrap()
+    }
+    #[test]
+    fn pure_three_outputs_have_bf16_precision_and_no_state_effect() {
+        let (mut b, p) = fixture();
+        let x = input(&mut b, 32);
+        let h = input(&mut b, 96);
+        let output = b
+            .causal_conv_silu_profiled(p, &x, &h, CausalConvSiluContract::TorchCpuBf16Avx2)
+            .unwrap();
+        for (v, n) in [
+            (&output.raw, 32),
+            (&output.activated, 32),
+            (&output.next_history, 96),
+        ] {
+            assert_eq!(b.elements(v), n);
+            assert_eq!(b.precision(v), ActivationPrecision::Bf16WidenedF32);
+        }
+        let recorded = b.finish_operator_slice(output.activated).unwrap();
+        assert_eq!(recorded.plan().nodes().len(), 3);
+        assert!(recorded.plan().required_domains().is_empty());
+        assert!(recorded.plan().proposed_domains().is_empty());
+        assert_eq!(recorded.plan().frontier_delta(), 0);
+    }
+    #[test]
+    fn raw_shape_dtype_length_source_and_history_are_closed() {
+        let (mut b, p) = fixture();
+        let x = input(&mut b, 32);
+        let h = input(&mut b, 96);
+        let contract = CausalConvSiluContract::TorchCpuBf16Avx2;
+        for wrong in [
+            SourceParameter {
+                source: SourceIdentity::for_new_allocation().unwrap(),
+                ..p.clone()
+            },
+            SourceParameter {
+                shape: vec![128],
+                ..p.clone()
+            },
+            SourceParameter {
+                shape: vec![32, 4, 1],
+                ..p.clone()
+            },
+            SourceParameter {
+                kind: QuantKind::F32,
+                ..p.clone()
+            },
+            SourceParameter {
+                bytes: 255,
+                ..p.clone()
+            },
+        ] {
+            assert!(
+                b.causal_conv_silu_profiled(wrong, &x, &h, contract)
+                    .is_err()
+            );
+        }
+        let short = input(&mut b, 95);
+        assert!(
+            b.causal_conv_silu_profiled(p.clone(), &x, &short, contract)
+                .is_err()
+        );
+        let mut foreign = PlanBuilder::new_operator_slice(b.source, b.profile).unwrap();
+        let other = input(&mut foreign, 32);
+        assert!(
+            b.causal_conv_silu_profiled(p, &other, &h, contract)
+                .is_err()
+        );
+    }
+    #[test]
+    fn valid_geometry_strided_values_require_explicit_pack() {
+        let (mut b, p) = fixture();
+        let base = input(&mut b, 64);
+        let h = input(&mut b, 96);
+        let strided = b
+            .view(
+                &base,
+                TensorView {
+                    offset: 0,
+                    rows: 32,
+                    width: 1,
+                    row_stride: 2,
+                },
+            )
+            .unwrap();
+        let contract = CausalConvSiluContract::TorchCpuBf16Avx2;
+        let error = b
+            .causal_conv_silu_profiled(p.clone(), &strided, &h, contract)
+            .err()
+            .unwrap();
+        assert!(error.to_string().contains("contiguous"));
+        let packed = b.contiguous(&strided).unwrap();
+        b.causal_conv_silu_profiled(p, &packed, &h, contract)
+            .unwrap();
+    }
+}
--- a/crates/zerogravity-core/src/accelerator/program/typed_tests.rs
+++ b/crates/zerogravity-core/src/accelerator/program/typed_tests.rs
@@ -525,3 +525,24 @@
     );
     assert_eq!(slice.plan().frontier_delta(), 0);
 }
+
+#[test]
+fn unaware_backend_causal_conv_refuses_before_any_callback() {
+    let mut backend = UnawareBackend;
+    let parameter = SourceParameter {
+        source: SourceIdentity::for_new_allocation().unwrap(),
+        name: "actual.conv.weight".into(),
+        kind: QuantKind::Bf16,
+        shape: vec![32, 1, 4],
+        bytes: 256,
+    };
+    assert!(matches!(
+        backend.causal_conv_silu_profiled(
+            parameter,
+            &vec![0.0; 32],
+            &vec![0.0; 96],
+            CausalConvSiluContract::TorchCpuBf16Avx2
+        ),
+        Err(Error::Unsupported(_))
+    ));
+}
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -12,6 +12,8 @@
 use std::sync::Arc;
 use std::sync::atomic::{AtomicU64, Ordering};
 
+mod gdn_conv;
+pub use gdn_conv::{CausalConvSiluContract, CausalConvValues};
 mod typed;
 pub use typed::{
     ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
@@ -280,6 +282,18 @@
             "backend does not implement explicit ordinary RMS contract".into(),
         ))
     }
+    /// Pure original convolution window; no state owner or implicit fallback.
+    fn causal_conv_silu_profiled(
+        &mut self,
+        _parameter: SourceParameter,
+        _input: &Self::Value,
+        _history: &Self::Value,
+        _contract: CausalConvSiluContract,
+    ) -> Result<CausalConvValues<Self::Value>> {
+        Err(Error::Unsupported(
+            "backend does not implement explicit causal convolution SiLU contract".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -452,6 +466,10 @@
     TypedProject {
         tensor: SourceTensor,
         contract: ProjectionContract,
+    },
+    TypedCausalConvSilu {
+        parameter: SourceParameter,
+        contract: CausalConvSiluContract,
     },
     TypedOrdinaryRms {
         parameter: SourceParameter,
@@ -800,6 +818,10 @@
                 contract.validate(parameter, inputs[0].elements)?;
                 Ok(ActivationPrecision::Bf16WidenedF32)
             }
+            PlanOperation::TypedCausalConvSilu {
+                parameter,
+                contract,
+            } => gdn_conv::result_precision(self, parameter, *contract, inputs),
             PlanOperation::View(_) | PlanOperation::Contiguous => inputs
                 .first()
                 .map(|value| value.precision)
@@ -816,6 +838,7 @@
                     PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
+                        | PlanOperation::TypedCausalConvSilu { .. }
                 ) =>
             {
                 Ok(ActivationPrecision::F32)
@@ -1082,6 +1105,15 @@
             input.shape,
             None,
         )
+    }
+    fn causal_conv_silu_profiled(
+        &mut self,
+        parameter: SourceParameter,
+        input: &PlanValue,
+        history: &PlanValue,
+        contract: CausalConvSiluContract,
+    ) -> Result<CausalConvValues<PlanValue>> {
+        gdn_conv::record(self, parameter, input, history, contract)
     }
     fn embed(&mut self, tensor: SourceTensor, token: ParameterRef) -> Result<PlanValue> {
         self.source_tensor(&tensor)?;
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -608,3 +608,45 @@
         }
     }
 }
+
+/// Pure original window only; proposed history is not a StateEffect.
+pub(crate) fn gdn_convolution_silu_slice<B: MathematicalBackend>(
+    source: &impl OperatorSliceSource,
+    math: &mut B,
+    layer: usize,
+    input: &B::Value,
+    history: &B::Value,
+) -> Result<crate::accelerator::program::CausalConvValues<B::Value>> {
+    use crate::accelerator::program::{ActivationPrecision, CausalConvSiluContract};
+    let (parameter, channels) = crate::operator_slice::gdn_conv::convolution_role(source, layer)?;
+    if math.elements(input) != channels
+        || math.elements(history) != channels * 3
+        || [input, history]
+            .iter()
+            .any(|v| math.precision(v) != ActivationPrecision::Bf16WidenedF32)
+    {
+        return Err(Error::State(
+            "original convolution slice geometry/precision".into(),
+        ));
+    }
+    let values = math.causal_conv_silu_profiled(
+        parameter,
+        input,
+        history,
+        CausalConvSiluContract::TorchCpuBf16Avx2,
+    )?;
+    for (value, elements) in [
+        (&values.raw, channels),
+        (&values.activated, channels),
+        (&values.next_history, channels * 3),
+    ] {
+        if math.elements(value) != elements
+            || math.precision(value) != ActivationPrecision::Bf16WidenedF32
+        {
+            return Err(Error::State(
+                "original convolution region result geometry/precision".into(),
+            ));
+        }
+    }
+    Ok(values)
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/gdn_conv_slice.rs
@@ -0,0 +1,220 @@
+//! Pure original causal convolution + SiLU on explicit observed operands.
+//! No cache owner, producer/frontier assertion or full-model eligibility.
+use super::OperatorSliceSource;
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{
+    ActivationPrecision, CausalConvSiluContract, CausalConvValues, PlanBuilder, PlanValue,
+    ProgramNumericalProfile, SourceParameter,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, LoadedQwenModel, Result};
+
+pub(crate) fn convolution_role(
+    source: &impl OperatorSliceSource,
+    layer: usize,
+) -> Result<(SourceParameter, usize)> {
+    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || !source.source_identity().is_bound()
+        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
+    {
+        return Err(Error::Unsupported(
+            "closed original convolution profile/layer".into(),
+        ));
+    }
+    let parameter = source.parameter_descriptor(QwenParameterRole::GdnConvolution(layer))?;
+    let channels = source
+        .matrix_descriptor(QwenMatrixRole::GdnQkv(layer))?
+        .rows;
+    if parameter.source != source.source_identity()
+        || parameter.name
+            != format!("model.language_model.layers.{layer}.linear_attn.conv1d.weight")
+    {
+        return Err(Error::State("original convolution raw role/source".into()));
+    }
+    CausalConvSiluContract::TorchCpuBf16Avx2.validate(
+        &parameter,
+        channels,
+        channels
+            .checked_mul(3)
+            .ok_or_else(|| Error::Limit("convolution history width".into()))?,
+    )?;
+    Ok((parameter, channels))
+}
+
+/// Observed operand phase only; never a committed-session binding.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum ConvInputPhase {
+    /// Initial operator's implicit left padding, supplied as authenticated +0.
+    InitialPadding,
+    /// Last three entries of the original incoming `[channel,4]` cache.
+    CachedTail,
+}
+
+/// Qualifier authenticates external source/archive provenance. This type checks
+/// representation only, and never asserts a model producer or StateRoot.
+#[derive(Clone, Copy)]
+pub struct GdnConvSliceInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    phase: ConvInputPhase,
+    activation: &'a [u8],
+    history: &'a [u8],
+}
+impl<'a> GdnConvSliceInput<'a> {
+    pub fn from_bf16_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        phase: ConvInputPhase,
+        activation: &'a [u8],
+        history: &'a [u8],
+    ) -> Result<Self> {
+        let input = Self {
+            source,
+            profile,
+            phase,
+            activation,
+            history,
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
+    pub fn phase(self) -> ConvInputPhase {
+        self.phase
+    }
+    pub fn precision(self) -> ActivationPrecision {
+        ActivationPrecision::Bf16WidenedF32
+    }
+    pub fn activation_bytes(self) -> &'a [u8] {
+        self.activation
+    }
+    pub fn history_bytes(self) -> &'a [u8] {
+        self.history
+    }
+    pub(crate) fn validate(self) -> Result<usize> {
+        let channels = self.activation.len() / 2;
+        if !self.source.is_bound()
+            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+            || channels == 0
+            || channels > 16384
+            || !channels.is_multiple_of(32)
+            || self.activation.len() != channels * 2
+            || self.history.len() != channels * 6
+            || self
+                .activation
+                .chunks_exact(2)
+                .chain(self.history.chunks_exact(2))
+                .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
+            || (self.phase == ConvInputPhase::InitialPadding
+                && self.history.iter().any(|v| *v != 0))
+        {
+            return Err(Error::Malformed(
+                "private convolution phase/source/finite BF16 operands".into(),
+            ));
+        }
+        Ok(channels)
+    }
+}
+
+impl LoadedQwenModel {
+    /// Append only, to existing same-builder BF16 values; no execution admission.
+    pub fn append_gdn_convolution_silu_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &PlanValue,
+        history: &PlanValue,
+    ) -> Result<CausalConvValues<PlanValue>> {
+        self.require_operator_slice()?;
+        crate::forward::gdn_convolution_silu_slice(self, builder, layer, input, history)
+    }
+}
+
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+pub struct GdnConvolutionSiluOutput {
+    source: SourceIdentity,
+    layer: usize,
+    phase: ConvInputPhase,
+    input: crate::host_math::HostValue,
+    prior: crate::host_math::HostValue,
+    values: CausalConvValues<crate::host_math::HostValue>,
+}
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+impl GdnConvolutionSiluOutput {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    }
+    pub fn precision(&self) -> ActivationPrecision {
+        ActivationPrecision::Bf16WidenedF32
+    }
+    pub fn layer(&self) -> usize {
+        self.layer
+    }
+    pub fn phase(&self) -> ConvInputPhase {
+        self.phase
+    }
+    pub fn input(&self) -> Result<&[f32]> {
+        self.input.as_slice()
+    }
+    pub fn prior_history(&self) -> Result<&[f32]> {
+        self.prior.as_slice()
+    }
+    pub fn raw(&self) -> Result<&[f32]> {
+        self.values.raw.as_slice()
+    }
+    pub fn activated(&self) -> Result<&[f32]> {
+        self.values.activated.as_slice()
+    }
+    pub fn next_history(&self) -> Result<&[f32]> {
+        self.values.next_history.as_slice()
+    }
+}
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    pub fn append_gdn_convolution_silu_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &PlanValue,
+        history: &PlanValue,
+    ) -> Result<CausalConvValues<PlanValue>> {
+        crate::forward::gdn_convolution_silu_slice(self, builder, layer, input, history)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_gdn_convolution_silu_slice(
+        &self,
+        layer: usize,
+        input: GdnConvSliceInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<GdnConvolutionSiluOutput> {
+        let (_, channels) = convolution_role(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let (activation, prior) = math.import_gdn_conv_input(input, channels)?;
+        let values = crate::forward::gdn_convolution_silu_slice(
+            self,
+            &mut math,
+            layer,
+            &activation,
+            &prior,
+        )?;
+        if cancelled.is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State("window convolution slice cancelled".into()));
+        }
+        Ok(GdnConvolutionSiluOutput {
+            source: self.source_identity(),
+            layer,
+            phase: input.phase(),
+            input: activation,
+            prior,
+            values,
+        })
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/gdn_conv_window_tests.rs
@@ -0,0 +1,382 @@
+//! Synthetic role/wiring/lease checks; genuine trained qualification is separate.
+use super::*;
+use crate::accelerator::program::{CausalConvSiluContract, ProgramNumericalProfile};
+use crate::operator_slice::gdn_conv::{ConvInputPhase, GdnConvSliceInput};
+use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
+
+fn conv_source(fixture: &Fixture) -> Loaded {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    cfg["text_config"]["hidden_size"] = json!(256);
+    cfg["text_config"]["linear_key_head_dim"] = json!(8);
+    cfg["text_config"]["linear_value_head_dim"] = json!(8);
+    source_with_config(
+        fixture,
+        serde_json::to_vec(&cfg).unwrap(),
+        true,
+        |name, index, fallback| {
+            if name.ends_with(".conv1d.weight") {
+                if index % 4 == 3 { 0x3f80 } else { 0 }
+            } else {
+                fallback
+            }
+        },
+        |_| {},
+    )
+}
+fn bf16(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
+    values
+        .into_iter()
+        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
+        .collect()
+}
+fn private<'a>(
+    source: crate::accelerator::SourceIdentity,
+    phase: ConvInputPhase,
+    x: &'a [u8],
+    h: &'a [u8],
+) -> GdnConvSliceInput<'a> {
+    GdnConvSliceInput::from_bf16_bytes(
+        source,
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        phase,
+        x,
+        h,
+    )
+    .unwrap()
+}
+#[test]
+fn pure_window_region_matches_closed_helpers_and_preserves_canonical_history_bits() {
+    let f = Fixture::new();
+    let loaded = conv_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let x = bf16((0..32).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }));
+    let h = bf16((0..32).flat_map(|_| [1.0, -0.0, 2.0]));
+    let output = ops
+        .execute_gdn_convolution_silu_slice(
+            0,
+            private(source.source_identity(), ConvInputPhase::CachedTail, &x, &h),
+            None,
+        )
+        .unwrap();
+    assert_eq!(output.layer(), 0);
+    assert_eq!(output.phase(), ConvInputPhase::CachedTail);
+    assert_eq!(output.source_identity(), source.source_identity());
+    // Last tap 1 and three zero weights make each positive input exact.
+    same(output.input().unwrap(), output.raw().unwrap());
+    let silu = zerogravity_cpu::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16(
+        source.source_identity(),
+        output.raw().unwrap(),
+    )
+    .unwrap();
+    assert_eq!(silu.len(), output.activated().unwrap().len());
+    for (expected, actual) in silu.iter().zip(output.activated().unwrap()) {
+        let bits = expected.to_bits();
+        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
+        assert_eq!(actual.to_bits(), rounded);
+    }
+    for (channel, next) in output.next_history().unwrap().chunks_exact(3).enumerate() {
+        assert_eq!(next[0].to_bits(), (-0.0f32).to_bits());
+        assert_eq!(next[1], 2.0);
+        assert_eq!(
+            next[2].to_bits(),
+            output.input().unwrap()[channel].to_bits()
+        );
+    }
+    let snapshot = admission.arbiter.snapshot(&admission.resource).unwrap();
+    assert_eq!(reserved(&admission), baseline + 2 * 960 * 1024 + 32 * 36);
+    assert_eq!(
+        snapshot
+            .per_class
+            .get(&AllocationClass::Output)
+            .copied()
+            .unwrap_or(0),
+        32 * 36
+    );
+    assert_eq!(
+        snapshot
+            .per_class
+            .get(&AllocationClass::TransferStaging)
+            .copied()
+            .unwrap_or(0),
+        0
+    );
+    let (model, _, _) = bind(loaded);
+    assert_eq!(model.initialized_cpu_state_bytes(), 0);
+    assert!(model.new_session().is_err());
+    assert!(model.forward_plan(8, true, false).is_err());
+    drop(ops);
+    drop(source);
+    assert_eq!(output.raw().unwrap().len(), 32);
+    assert!(reserved(&admission) > 0);
+    drop(output);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn initial_padding_is_explicit_and_never_canonicalizes_signed_zero() {
+    let id = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
+    let x = bf16([1.0; 32]);
+    let zero = bf16([0.0; 96]);
+    let negative = bf16([-0.0; 96]);
+    private(id, ConvInputPhase::InitialPadding, &x, &zero);
+    assert!(
+        GdnConvSliceInput::from_bf16_bytes(
+            id,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            ConvInputPhase::InitialPadding,
+            &x,
+            &negative
+        )
+        .is_err()
+    );
+    private(id, ConvInputPhase::CachedTail, &x, &negative);
+    for bytes in [vec![0; 63], vec![0xff; 64]] {
+        assert!(
+            GdnConvSliceInput::from_bf16_bytes(
+                id,
+                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+                ConvInputPhase::CachedTail,
+                &bytes,
+                &zero
+            )
+            .is_err()
+        );
+    }
+}
+#[test]
+fn aggregate_pressure_cancel_and_live_output_never_leak_or_reuse_admission() {
+    let f = Fixture::new();
+    let loaded = conv_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let x = bf16([1.0; 32]);
+    let h = bf16([0.0; 96]);
+    let input = private(
+        source.source_identity(),
+        ConvInputPhase::InitialPadding,
+        &x,
+        &h,
+    );
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        ops.execute_gdn_convolution_silu_slice(0, input, Some(&cancelled))
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), baseline);
+    let pressure = admission
+        .arbiter
+        .admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: 128 * 1024 * 1024 - baseline,
+            }],
+        )
+        .unwrap();
+    assert!(
+        ops.execute_gdn_convolution_silu_slice(0, input, None)
+            .is_err()
+    );
+    drop(pressure);
+    assert_eq!(reserved(&admission), baseline);
+    // Leave enough for BOTH input Vecs, but one byte short of the aggregate
+    // three-result envelope. A failed second admission must retire the inputs.
+    let input_reservation = 960 * 1024 + 32 * 16;
+    let result_reservation = 960 * 1024 + 32 * 20;
+    let held = admission
+        .arbiter
+        .admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: 128 * 1024 * 1024 - baseline - input_reservation - result_reservation + 1,
+            }],
+        )
+        .unwrap();
+    let pressured = reserved(&admission);
+    assert!(
+        ops.execute_gdn_convolution_silu_slice(0, input, None)
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), pressured);
+    drop(held);
+    assert_eq!(reserved(&admission), baseline);
+    // Cancellation between actual operand import and the region, without clocks.
+    cancelled.store(false, Ordering::Release);
+    let mut math =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, Some(&cancelled), None);
+    let (activation, history) = math.import_gdn_conv_input(input, 32).unwrap();
+    let after_import = reserved(&admission);
+    assert_eq!(after_import, baseline + 960 * 1024 + 32 * 16);
+    let parameter = ops
+        .parameter_descriptor(QwenParameterRole::GdnConvolution(0))
+        .unwrap();
+    cancelled.store(true, Ordering::Release);
+    assert!(
+        math.causal_conv_silu_profiled(
+            parameter.clone(),
+            &activation,
+            &history,
+            CausalConvSiluContract::TorchCpuBf16Avx2
+        )
+        .is_err()
+    );
+    assert_eq!(reserved(&admission), after_import);
+    cancelled.store(false, Ordering::Release);
+    let values = math
+        .causal_conv_silu_profiled(
+            parameter,
+            &activation,
+            &history,
+            CausalConvSiluContract::TorchCpuBf16Avx2,
+        )
+        .unwrap();
+    let clone = values.activated.clone();
+    assert!(clone.clone().into_vec().is_err());
+    drop(values);
+    assert_eq!(clone.as_slice().unwrap().len(), 32);
+    drop(clone);
+    assert_eq!(reserved(&admission), after_import);
+    drop(math);
+    drop(activation);
+    drop(history);
+    assert_eq!(reserved(&admission), baseline);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn same_source_strided_values_foreign_issuer_and_unaware_provider_refuse() {
+    struct Unaware;
+    impl crate::accelerator::MatVecKernel for Unaware {
+        fn try_execute(
+            &self,
+            _: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            panic!("typed convolution must refuse before any provider callback")
+        }
+    }
+    let f = Fixture::new();
+    let loaded = conv_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let x = bf16([1.0; 32]);
+    let h = bf16([0.0; 96]);
+    let input = private(
+        source.source_identity(),
+        ConvInputPhase::InitialPadding,
+        &x,
+        &h,
+    );
+    let provider = Unaware;
+    let unsupported =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, None, Some(&provider));
+    assert!(unsupported.import_gdn_conv_input(input, 32).is_err());
+    assert_eq!(reserved(&admission), baseline);
+    let mut math = crate::host_math::HostMath::for_window_operator_slice(&ops, None, None);
+    let (activation, history) = math.import_gdn_conv_input(input, 32).unwrap();
+    let strided = math
+        .view(
+            &history,
+            TensorView {
+                offset: 0,
+                rows: 32,
+                width: 1,
+                row_stride: 2,
+            },
+        )
+        .unwrap();
+    let parameter = ops
+        .parameter_descriptor(QwenParameterRole::GdnConvolution(0))
+        .unwrap();
+    let before = reserved(&admission);
+    let error = math
+        .causal_conv_silu_profiled(
+            parameter.clone(),
+            &strided,
+            &history,
+            CausalConvSiluContract::TorchCpuBf16Avx2,
+        )
+        .err()
+        .unwrap();
+    assert!(error.to_string().contains("contiguous"));
+    assert_eq!(reserved(&admission), before);
+    let foreign_source = windows(&f, &loaded).0;
+    let foreign_ops = bind_windows(foreign_source, &loaded);
+    let mut foreign =
+        crate::host_math::HostMath::for_window_operator_slice(&foreign_ops, None, None);
+    assert!(
+        foreign
+            .causal_conv_silu_profiled(
+                parameter,
+                &activation,
+                &history,
+                CausalConvSiluContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+    );
+    drop(foreign);
+    drop(foreign_ops);
+    drop(math);
+    drop(strided);
+    drop(activation);
+    drop(history);
+    assert_eq!(reserved(&admission), baseline);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+
+#[test]
+fn actual_role_plan_append_uses_same_builder_and_never_adds_state() {
+    use crate::accelerator::program::{ActivationPrecision, PlanBuilder, PlanOperation};
+    let fixture = Fixture::new();
+    let loaded = conv_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let mut builder = PlanBuilder::new_operator_slice(
+        source.source_identity(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+    )
+    .unwrap();
+    let embedding = builder
+        .embed_profiled(
+            operators
+                .matrix_descriptor(QwenMatrixRole::Embedding)
+                .unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    // Synthetic embedding-produced operands solely exercise the actual source
+    // roles/same-builder path; they are not claimed as trained QKV/cache inputs.
+    let x = builder.view(&embedding, TensorView::vector(0, 32)).unwrap();
+    let h = builder
+        .view(&embedding, TensorView::vector(32, 96))
+        .unwrap();
+    let result = operators
+        .append_gdn_convolution_silu_slice(&mut builder, 0, &x, &h)
+        .unwrap();
+    assert_eq!(
+        builder.precision(&result.activated),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    let recorded = builder.finish_operator_slice(result.activated).unwrap();
+    assert_eq!(recorded.plan().nodes().len(), 4);
+    assert!(matches!(
+        recorded.plan().nodes()[3].operation,
+        PlanOperation::TypedCausalConvSilu { .. }
+    ));
+    assert!(recorded.plan().required_domains().is_empty());
+    assert!(recorded.plan().proposed_domains().is_empty());
+    assert_eq!(recorded.plan().frontier_delta(), 0);
+    drop(recorded);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/host_math/gdn_conv.rs
@@ -0,0 +1,152 @@
+//! Aggregate admitted private inputs/results for the pure original region.
+use super::*;
+use crate::accelerator::program::{CausalConvSiluContract, CausalConvValues};
+use crate::operator_slice::gdn_conv::GdnConvSliceInput;
+
+const SCRATCH: usize = 960 * 1024;
+impl HostMath<'_, '_, '_> {
+    fn conv_windows(&self) -> Result<&crate::original_binding::OriginalWindowOperators> {
+        if self.region_kernel.is_some()
+            || self.engine.is_some()
+            || self.state.is_some()
+            || self.source.program_numerical_profile()
+                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported(
+                "convolution requires private pinned CPU window contract".into(),
+            ));
+        }
+        match &self.source {
+            HostSource::Windows(source) => Ok(source),
+            HostSource::Loaded(_) => Err(Error::Unsupported(
+                "convolution requires admitted window backing".into(),
+            )),
+        }
+    }
+    fn check_conv_cancel(&self) -> Result<()> {
+        if self
+            .cancelled
+            .is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire))
+        {
+            return Err(Error::State("window convolution cancelled".into()));
+        }
+        Ok(())
+    }
+    pub(crate) fn import_gdn_conv_input(
+        &self,
+        input: GdnConvSliceInput<'_>,
+        channels: usize,
+    ) -> Result<(HostValue, HostValue)> {
+        let source = self.conv_windows()?;
+        if input.validate()? != channels
+            || input.source_identity() != source.source_identity()
+            || input.numerical_profile() != self.source.program_numerical_profile()
+        {
+            return Err(Error::State(
+                "convolution input full width/source/profile".into(),
+            ));
+        }
+        self.check_conv_cancel()?;
+        // Both allocations are charged once, before either Vec allocation.
+        let workspace = Arc::new(
+            source.source().reserve_operator_buffers(
+                channels
+                    .checked_mul(16)
+                    .ok_or_else(|| Error::Limit("convolution input byte count".into()))?,
+                SCRATCH,
+            )?,
+        );
+        let make = |bytes: &[u8]| -> Result<HostValue> {
+            let values = bytes
+                .chunks_exact(2)
+                .map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16))
+                .collect();
+            HostValue::typed(
+                values,
+                ActivationPrecision::Bf16WidenedF32,
+                source.source_identity(),
+                self.source.program_numerical_profile(),
+            )?
+            .retain_workspace(Some(workspace.clone()))
+        };
+        let result = (
+            make(input.activation_bytes())?,
+            make(input.history_bytes())?,
+        );
+        self.check_conv_cancel()?;
+        Ok(result)
+    }
+    pub(super) fn execute_causal_conv_silu(
+        &self,
+        parameter: SourceParameter,
+        input: &HostValue,
+        history: &HostValue,
+        contract: CausalConvSiluContract,
+    ) -> Result<CausalConvValues<HostValue>> {
+        let source = self.conv_windows()?;
+        let channels = contract.validate(&parameter, input.elements, history.elements)?;
+        if parameter.source != source.source_identity() {
+            return Err(Error::State("convolution parameter source".into()));
+        }
+        for value in [input, history] {
+            if value.source != Some(source.source_identity())
+                || value.profile != Some(self.source.program_numerical_profile())
+                || value.precision != ActivationPrecision::Bf16WidenedF32
+            {
+                return Err(Error::State(
+                    "convolution activation source/profile/precision".into(),
+                ));
+            }
+            let workspace = value.data.workspace.as_ref().ok_or_else(|| {
+                Error::State("convolution activation missing physical backing".into())
+            })?;
+            if !source.source().same_issuer(workspace.source()) {
+                return Err(Error::State(
+                    "convolution activation foreign source issuer".into(),
+                ));
+            }
+        }
+        // Contiguity and source membership checked before allocation or math.
+        let activation = input.as_slice()?;
+        let prior = history.as_slice()?;
+        ActivationPrecision::Bf16WidenedF32.validate_values(activation)?;
+        ActivationPrecision::Bf16WidenedF32.validate_values(prior)?;
+        self.check_conv_cancel()?;
+        // Three retained results: C raw + C activated + 3C history, F32 backing.
+        // Scratch conservatively covers raw FP32 + SiLU FP32 + rounding Vec.
+        let workspace = Arc::new(
+            source.source().reserve_operator_buffers(
+                channels
+                    .checked_mul(20)
+                    .ok_or_else(|| Error::Limit("convolution output byte count".into()))?,
+                SCRATCH,
+            )?,
+        );
+        let completed = source.convolve(&parameter, activation, prior, self.cancelled)?;
+        let silu =
+            zerogravity_cpu::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16(
+                source.source_identity(),
+                &completed.raw_bf16,
+            )?;
+        let activated = silu
+            .into_iter()
+            .map(round_bf16)
+            .collect::<Result<Vec<_>>>()?;
+        let make = |values| -> Result<HostValue> {
+            HostValue::typed(
+                values,
+                ActivationPrecision::Bf16WidenedF32,
+                source.source_identity(),
+                self.source.program_numerical_profile(),
+            )?
+            .retain_workspace(Some(workspace.clone()))
+        };
+        let result = CausalConvValues {
+            raw: make(completed.raw_bf16)?,
+            activated: make(activated)?,
+            next_history: make(completed.next_history)?,
+        };
+        self.check_conv_cancel()?;
+        Ok(result)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -507,6 +507,25 @@
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         let output = output.retain_workspace(workspace)?;
         Ok(output)
+    }
+    fn causal_conv_silu_profiled(
+        &mut self,
+        parameter: SourceParameter,
+        input: &HostValue,
+        history: &HostValue,
+        contract: crate::accelerator::program::CausalConvSiluContract,
+    ) -> Result<crate::accelerator::program::CausalConvValues<HostValue>> {
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        {
+            self.execute_causal_conv_silu(parameter, input, history, contract)
+        }
+        #[cfg(not(all(feature = "original-hf-binding", target_os = "linux")))]
+        {
+            let _ = (parameter, input, history, contract);
+            Err(Error::Unsupported(
+                "causal convolution requires leased original windows".into(),
+            ))
+        }
     }
     fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<HostValue> {
         self.legacy_precision(&[])?;
@@ -1031,3 +1050,6 @@
         }
     }
 }
+
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+mod gdn_conv;
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -231,3 +231,6 @@
 /// Pure attention projections/QK norms; no rotary/cache/session admission.
 #[path = "attention_slice.rs"]
 pub mod attention;
+
+#[path = "gdn_conv_slice.rs"]
+pub mod gdn_conv;
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -379,3 +379,6 @@
 
 #[path = "attention_slice_tests.rs"]
 mod attention;
+
+#[path = "gdn_conv_window_tests.rs"]
+mod gdn_conv;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window/gdn_conv.rs
@@ -0,0 +1,59 @@
+//! Original full rank3 stage; arithmetic remains in accepted CPU helpers.
+use super::*;
+impl OriginalWindowOperators {
+    pub(crate) fn convolve(
+        &self,
+        parameter: &SourceParameter,
+        input: &[f32],
+        history: &[f32],
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<zerogravity_cpu::original_hf::conv::CausalConv4Completed> {
+        let mut found = false;
+        for (layer, kind) in self.geometry().layers.iter().enumerate() {
+            if *kind != crate::model::LayerKind::LinearAttention {
+                continue;
+            }
+            let (actual, channels) =
+                crate::operator_slice::gdn_conv::convolution_role(self, layer)?;
+            if actual == *parameter && input.len() == channels && history.len() == channels * 3 {
+                found = true;
+                break;
+            }
+        }
+        if !found {
+            return Err(Error::State(
+                "window convolution original role/geometry".into(),
+            ));
+        }
+        check_cancel(cancelled)?;
+        let raw = self.source().conv4_parameter(&parameter.name)?;
+        raw.validate_source_parameter(parameter)?;
+        let window = match cancelled {
+            Some(flag) => {
+                self.source()
+                    .read_parameter_cancellable(&raw, 0..raw.elements(), flag)?
+            }
+            None => self.source().read_parameter(&raw, 0..raw.elements())?,
+        };
+        if !window.is_complete() || window.element_range() != (0..raw.elements()) {
+            return Err(Error::State(
+                "window convolution incomplete raw parameter".into(),
+            ));
+        }
+        let output = zerogravity_cpu::original_hf::conv::causal_step_bf16_4(
+            zerogravity_cpu::original_hf::conv::CausalConv4Inputs {
+                source: self.source_identity(),
+                tensor_name: &parameter.name,
+                weight: zerogravity_core::source::SourceTensorRef {
+                    kind: raw.tensor().kind,
+                    shape: &raw.tensor().shape,
+                    packed_bytes: window.as_slice(),
+                },
+                activation: input,
+                history,
+            },
+        )?;
+        check_cancel(cancelled)?;
+        Ok(output)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/original_binding/window.rs
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window.rs
@@ -415,3 +415,6 @@
         self.projected.as_slice()
     }
 }
+
+#[cfg(feature = "cpu")]
+mod gdn_conv;

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/STEADY_LEDGER_HARNESS_ONLY_V4.patch

Original UTF-8 bytes: 764; SHA256 `9769f8b6fafb3666995607da23b916d2bb86a018bad8b07359cc9b2684868088`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/main.rs
+++ b/main.rs
@@ -341,8 +341,12 @@
     )?;
     let baseline = snapshot(&host)?;
     ensure(
-        baseline.reserved == SOURCE_META + 4 * MIB + CONTROL
-            && class(&baseline, AllocationClass::Output) == 0,
+        baseline.reserved == SOURCE_META + 8 * MIB + CONTROL
+            && baseline.reservations == 4
+            && class(&baseline, AllocationClass::Scratch) == SOURCE_META + 8 * MIB
+            && class(&baseline, AllocationClass::Other) == CONTROL
+            && class(&baseline, AllocationClass::Output) == 0
+            && class(&baseline, AllocationClass::TransferStaging) == 0,
         "exact steady admitted source/control state",
     )?;
     let cases = inputs["cases"].as_array().ok_or("both input cases")?;

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-prepared/GPU_REFUSAL.patch

Original UTF-8 bytes: 1645; SHA256 `f09d057c3c65d75907f85ade140c45c05465d20b9b831a254b86475b8299e419`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -838,7 +838,8 @@
             match &node.operation {
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
-                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedOrdinaryRms { .. }
+                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                     "resident batch has no qualified typed lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -348,6 +348,7 @@
                     PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
+                        | PlanOperation::TypedCausalConvSilu { .. }
                 )
             })
         {
@@ -1429,7 +1430,8 @@
             match &node.operation {
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
-                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedOrdinaryRms { .. }
+                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                     "resident typed operation has no qualified lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-prepared/NEW_DELTA.patch

Original UTF-8 bytes: 52773; SHA256 `b53625fa619545bfa1a8604ca000bd08a7d2a2a41a9bff0aac99116e6fbfceea`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-artifacts/src/source_weights/tensor_window/conv_parameter.rs
@@ -0,0 +1,36 @@
+//! Sealed original BF16 depthwise four-tap parameter transport only.
+use super::*;
+impl VerifiedTensorSource {
+    /// Original rank3 metadata and raw elements, without model eligibility.
+    /// `parameter()` remains rank1-only; no synthetic flattened tensor exists.
+    pub fn conv4_parameter(&self, name: &str) -> Result<VerifiedParameter> {
+        let location = self
+            .0
+            .tensors
+            .get(name)
+            .ok_or_else(|| Error::Malformed("original conv4 parameter missing".into()))?;
+        let [channels, 1, 4] = location.info.shape.as_slice() else {
+            return Err(Error::Unsupported("original conv4 raw rank/shape".into()));
+        };
+        let channels = usize::try_from(*channels)
+            .ok()
+            .filter(|n| *n > 0 && *n <= 16384 && n.is_multiple_of(32))
+            .ok_or_else(|| Error::Limit("original conv4 channel bound".into()))?;
+        let elements = channels
+            .checked_mul(4)
+            .ok_or_else(|| Error::Limit("original conv4 element count".into()))?;
+        let bytes = elements
+            .checked_mul(2)
+            .ok_or_else(|| Error::Limit("original conv4 byte count".into()))?;
+        if location.info.kind != QuantKind::Bf16 || location.info.byte_len != bytes as u64 {
+            return Err(Error::Unsupported("original conv4 raw dtype/length".into()));
+        }
+        Ok(VerifiedParameter {
+            location: location.clone(),
+            owner: self.0.clone(),
+            elements,
+            element_bytes: 2,
+            bytes,
+        })
+    }
+}
--- a/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs
+++ b/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs
@@ -103,7 +103,7 @@
     matrix: VerifiedMatrix,
 }
 
-/// Sealed full original rank1 parameter; no matrix axes or adjusted values.
+/// Sealed full original rank1 or closed conv4 parameter; no adjusted values.
 #[derive(Clone)]
 pub struct VerifiedParameter {
     location: Arc<TensorLocation>,
@@ -493,7 +493,7 @@
             bytes,
         })
     }
-    /// Original Safetensors rank1 is contiguous; arbitrary strides have no API.
+    /// Original parameter elements are contiguous; arbitrary strides have no API.
     pub fn read_parameter(
         &self,
         parameter: &VerifiedParameter,
@@ -802,3 +802,5 @@
         Ok(self.position)
     }
 }
+
+mod conv_parameter;
--- /dev/null
+++ b/crates/zerogravity-artifacts/tests/tensor_window/conv4.rs
@@ -0,0 +1,101 @@
+//! Synthetic transport only; no trained convolution or model qualification.
+use super::*;
+fn fixture(dtype: &str, shape: &[u64], bytes: usize) -> Fixture {
+    let mut fixture = Fixture::new();
+    let raw = (0..bytes / 2)
+        .flat_map(|i| (0x3f80u16 + (i % 8) as u16).to_le_bytes())
+        .collect::<Vec<_>>();
+    fixture.second = safetensors("conv", dtype, shape, &raw);
+    fixture.index = serde_json::to_vec(
+        &json!({"metadata":{"total_size":64+raw.len()},"weight_map":{"matrix":A,"conv":B}}),
+    )
+    .unwrap();
+    fs::write(fixture.path.join(B), &fixture.second).unwrap();
+    fixture
+}
+#[test]
+fn full_original_rank3_and_partial_range_keep_truthful_metadata_and_owner() {
+    let f = fixture("BF16", &[32, 1, 4], 256);
+    let host = admission(64 * 1024 * 1024, "host");
+    let source = f.plan(&host).unwrap().open().unwrap();
+    let before = reserved(&host);
+    assert!(source.parameter("conv").is_err());
+    let raw = source.conv4_parameter("conv").unwrap();
+    let descriptor = raw.source_parameter();
+    assert_eq!(descriptor.shape, [32, 1, 4]);
+    assert_eq!(descriptor.bytes, 256);
+    raw.validate_source_parameter(&descriptor).unwrap();
+    let partial = source.read_parameter(&raw, 4..8).unwrap();
+    assert!(!partial.is_complete());
+    assert_eq!(partial.parameter().tensor().shape, [32, 1, 4]);
+    assert_eq!(partial.element_range(), 4..8);
+    drop(partial);
+    assert_eq!(reserved(&host), before);
+    let full = source.read_parameter(&raw, 0..128).unwrap();
+    assert!(full.is_complete());
+    assert_eq!(full.as_slice().len(), 256);
+    assert!(full.charged_staging_bytes() >= 256);
+    assert!(source.read_parameter(&raw, 0..129).is_err());
+    assert!(source.read_parameter(&raw, 4..4).is_err());
+    let foreign = f.plan(&host).unwrap().open().unwrap();
+    assert!(foreign.read_parameter(&raw, 0..128).is_err());
+    drop(foreign);
+    drop(descriptor);
+    drop(raw);
+    drop(source);
+    assert_eq!(full.as_slice().len(), 256);
+    assert!(reserved(&host) > 0);
+    drop(full);
+    assert_eq!(reserved(&host), 0);
+}
+#[test]
+fn geometry_corruption_cancel_and_pressure_refuse_without_leases() {
+    for (dtype, shape, bytes) in [
+        ("F32", vec![32, 1, 4], 512),
+        ("BF16", vec![128], 256),
+        ("BF16", vec![32, 4, 1], 256),
+        ("BF16", vec![16, 1, 4], 128),
+    ] {
+        let f = fixture(dtype, &shape, bytes);
+        let host = admission(64 * 1024 * 1024, "host");
+        let source = f.plan(&host).unwrap().open().unwrap();
+        assert!(source.conv4_parameter("conv").is_err());
+        drop(source);
+        assert_eq!(reserved(&host), 0);
+    }
+    let f = fixture("BF16", &[32, 1, 4], 256);
+    let host = admission(64 * 1024 * 1024, "host");
+    let source = f.plan(&host).unwrap().open().unwrap();
+    let raw = source.conv4_parameter("conv").unwrap();
+    let baseline = reserved(&host);
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        source
+            .read_parameter_cancellable(&raw, 0..128, &cancelled)
+            .is_err()
+    );
+    assert_eq!(reserved(&host), baseline);
+    let remaining = 64 * 1024 * 1024 - baseline;
+    let pressure = host
+        .arbiter
+        .admit(
+            &host.resource,
+            host.owner.clone(),
+            &[zerogravity_core::resource_arbiter::AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: remaining,
+            }],
+        )
+        .unwrap();
+    assert!(source.read_parameter(&raw, 0..128).is_err());
+    drop(pressure);
+    assert_eq!(reserved(&host), baseline);
+    let mut corrupt = f.second.clone();
+    *corrupt.last_mut().unwrap() ^= 1;
+    fs::write(f.path.join(B), corrupt).unwrap();
+    assert!(source.read_parameter(&raw, 0..128).is_err());
+    assert_eq!(reserved(&host), baseline);
+    drop(raw);
+    drop(source);
+    assert_eq!(reserved(&host), 0);
+}
--- a/crates/zerogravity-artifacts/tests/tensor_window.rs
+++ b/crates/zerogravity-artifacts/tests/tensor_window.rs
@@ -944,3 +944,6 @@
     assert_eq!(reserved(&host), 0);
     assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
 }
+
+#[path = "tensor_window/conv4.rs"]
+mod conv4;
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/gdn_conv.rs
@@ -0,0 +1,228 @@
+//! Closed pinned CPU region; three private BF16 outputs, no state effects.
+use super::*;
+
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum CausalConvSiluContract {
+    TorchCpuBf16Avx2,
+}
+impl CausalConvSiluContract {
+    /// Original raw depthwise `[channel,1,4]`, never an adjusted vector.
+    pub fn validate(
+        self,
+        parameter: &SourceParameter,
+        input: usize,
+        history: usize,
+    ) -> Result<usize> {
+        let channels = input;
+        if !parameter.source.is_bound() || parameter.name.is_empty() {
+            return Err(Error::State("causal convolution source binding".into()));
+        }
+        if channels == 0
+            || channels > 16384
+            || !channels.is_multiple_of(32)
+            || parameter.kind != QuantKind::Bf16
+            || parameter.shape.as_slice() != [channels as u64, 1, 4]
+            || channels.checked_mul(8) != Some(parameter.bytes)
+            || channels.checked_mul(3) != Some(history)
+        {
+            return Err(Error::Unsupported(
+                "closed causal convolution raw geometry/dtype".into(),
+            ));
+        }
+        Ok(channels)
+    }
+}
+
+#[derive(Clone)]
+pub struct CausalConvValues<V> {
+    pub raw: V,
+    pub activated: V,
+    /// Canonical `[channel,3]`, oldest to newest. Caller owns publication.
+    pub next_history: V,
+}
+
+pub(super) fn result_precision(
+    builder: &PlanBuilder,
+    parameter: &SourceParameter,
+    contract: CausalConvSiluContract,
+    inputs: &[PlanValue],
+) -> Result<ActivationPrecision> {
+    if !builder.operator_slice
+        || builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || parameter.source != builder.source
+        || inputs.len() != 2
+        || inputs
+            .iter()
+            .any(|v| v.precision != ActivationPrecision::Bf16WidenedF32)
+    {
+        return Err(Error::Unsupported(
+            "causal convolution profile/source/precision/slice".into(),
+        ));
+    }
+    contract.validate(parameter, inputs[0].elements, inputs[1].elements)?;
+    Ok(ActivationPrecision::Bf16WidenedF32)
+}
+pub(super) fn record(
+    builder: &mut PlanBuilder,
+    parameter: SourceParameter,
+    input: &PlanValue,
+    history: &PlanValue,
+    contract: CausalConvSiluContract,
+) -> Result<CausalConvValues<PlanValue>> {
+    let channels = contract.validate(&parameter, input.elements, history.elements)?;
+    let name = parameter.name.clone();
+    let values = builder.record(
+        &name,
+        PlanOperation::TypedCausalConvSilu {
+            parameter,
+            contract,
+        },
+        &[*input, *history],
+        &[
+            ValueShape::Vector(channels),
+            ValueShape::Vector(channels),
+            ValueShape::Vector(channels * 3),
+        ],
+        None,
+    )?;
+    let [raw, activated, next_history]: [PlanValue; 3] = values
+        .try_into()
+        .map_err(|_| Error::State("causal convolution output arity".into()))?;
+    Ok(CausalConvValues {
+        raw,
+        activated,
+        next_history,
+    })
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    fn fixture() -> (PlanBuilder, SourceParameter) {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        (
+            PlanBuilder::new_operator_slice(
+                source,
+                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            )
+            .unwrap(),
+            SourceParameter {
+                source,
+                name: "original.conv.weight".into(),
+                kind: QuantKind::Bf16,
+                shape: vec![32, 1, 4],
+                bytes: 256,
+            },
+        )
+    }
+    // Synthetic embedding metadata only; no original model producer claim.
+    fn input(builder: &mut PlanBuilder, width: usize) -> PlanValue {
+        builder
+            .embed_profiled(
+                SourceTensor {
+                    source: builder.source,
+                    name: format!("synthetic.{width}"),
+                    kind: QuantKind::Bf16,
+                    axes: MatrixAxes::OutputInput,
+                    columns: width,
+                    rows: 2,
+                },
+                PlanBuilder::inputs().token,
+                EmbeddingContract::Bf16OutputInput,
+            )
+            .unwrap()
+    }
+    #[test]
+    fn pure_three_outputs_have_bf16_precision_and_no_state_effect() {
+        let (mut b, p) = fixture();
+        let x = input(&mut b, 32);
+        let h = input(&mut b, 96);
+        let output = b
+            .causal_conv_silu_profiled(p, &x, &h, CausalConvSiluContract::TorchCpuBf16Avx2)
+            .unwrap();
+        for (v, n) in [
+            (&output.raw, 32),
+            (&output.activated, 32),
+            (&output.next_history, 96),
+        ] {
+            assert_eq!(b.elements(v), n);
+            assert_eq!(b.precision(v), ActivationPrecision::Bf16WidenedF32);
+        }
+        let recorded = b.finish_operator_slice(output.activated).unwrap();
+        assert_eq!(recorded.plan().nodes().len(), 3);
+        assert!(recorded.plan().required_domains().is_empty());
+        assert!(recorded.plan().proposed_domains().is_empty());
+        assert_eq!(recorded.plan().frontier_delta(), 0);
+    }
+    #[test]
+    fn raw_shape_dtype_length_source_and_history_are_closed() {
+        let (mut b, p) = fixture();
+        let x = input(&mut b, 32);
+        let h = input(&mut b, 96);
+        let contract = CausalConvSiluContract::TorchCpuBf16Avx2;
+        for wrong in [
+            SourceParameter {
+                source: SourceIdentity::for_new_allocation().unwrap(),
+                ..p.clone()
+            },
+            SourceParameter {
+                shape: vec![128],
+                ..p.clone()
+            },
+            SourceParameter {
+                shape: vec![32, 4, 1],
+                ..p.clone()
+            },
+            SourceParameter {
+                kind: QuantKind::F32,
+                ..p.clone()
+            },
+            SourceParameter {
+                bytes: 255,
+                ..p.clone()
+            },
+        ] {
+            assert!(
+                b.causal_conv_silu_profiled(wrong, &x, &h, contract)
+                    .is_err()
+            );
+        }
+        let short = input(&mut b, 95);
+        assert!(
+            b.causal_conv_silu_profiled(p.clone(), &x, &short, contract)
+                .is_err()
+        );
+        let mut foreign = PlanBuilder::new_operator_slice(b.source, b.profile).unwrap();
+        let other = input(&mut foreign, 32);
+        assert!(
+            b.causal_conv_silu_profiled(p, &other, &h, contract)
+                .is_err()
+        );
+    }
+    #[test]
+    fn valid_geometry_strided_values_require_explicit_pack() {
+        let (mut b, p) = fixture();
+        let base = input(&mut b, 64);
+        let h = input(&mut b, 96);
+        let strided = b
+            .view(
+                &base,
+                TensorView {
+                    offset: 0,
+                    rows: 32,
+                    width: 1,
+                    row_stride: 2,
+                },
+            )
+            .unwrap();
+        let contract = CausalConvSiluContract::TorchCpuBf16Avx2;
+        let error = b
+            .causal_conv_silu_profiled(p.clone(), &strided, &h, contract)
+            .err()
+            .unwrap();
+        assert!(error.to_string().contains("contiguous"));
+        let packed = b.contiguous(&strided).unwrap();
+        b.causal_conv_silu_profiled(p, &packed, &h, contract)
+            .unwrap();
+    }
+}
--- a/crates/zerogravity-core/src/accelerator/program/typed_tests.rs
+++ b/crates/zerogravity-core/src/accelerator/program/typed_tests.rs
@@ -525,3 +525,24 @@
     );
     assert_eq!(slice.plan().frontier_delta(), 0);
 }
+
+#[test]
+fn unaware_backend_causal_conv_refuses_before_any_callback() {
+    let mut backend = UnawareBackend;
+    let parameter = SourceParameter {
+        source: SourceIdentity::for_new_allocation().unwrap(),
+        name: "actual.conv.weight".into(),
+        kind: QuantKind::Bf16,
+        shape: vec![32, 1, 4],
+        bytes: 256,
+    };
+    assert!(matches!(
+        backend.causal_conv_silu_profiled(
+            parameter,
+            &vec![0.0; 32],
+            &vec![0.0; 96],
+            CausalConvSiluContract::TorchCpuBf16Avx2
+        ),
+        Err(Error::Unsupported(_))
+    ));
+}
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -12,6 +12,8 @@
 use std::sync::Arc;
 use std::sync::atomic::{AtomicU64, Ordering};
 
+mod gdn_conv;
+pub use gdn_conv::{CausalConvSiluContract, CausalConvValues};
 mod typed;
 pub use typed::{
     ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
@@ -280,6 +282,18 @@
             "backend does not implement explicit ordinary RMS contract".into(),
         ))
     }
+    /// Pure original convolution window; no state owner or implicit fallback.
+    fn causal_conv_silu_profiled(
+        &mut self,
+        _parameter: SourceParameter,
+        _input: &Self::Value,
+        _history: &Self::Value,
+        _contract: CausalConvSiluContract,
+    ) -> Result<CausalConvValues<Self::Value>> {
+        Err(Error::Unsupported(
+            "backend does not implement explicit causal convolution SiLU contract".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -452,6 +466,10 @@
     TypedProject {
         tensor: SourceTensor,
         contract: ProjectionContract,
+    },
+    TypedCausalConvSilu {
+        parameter: SourceParameter,
+        contract: CausalConvSiluContract,
     },
     TypedOrdinaryRms {
         parameter: SourceParameter,
@@ -800,6 +818,10 @@
                 contract.validate(parameter, inputs[0].elements)?;
                 Ok(ActivationPrecision::Bf16WidenedF32)
             }
+            PlanOperation::TypedCausalConvSilu {
+                parameter,
+                contract,
+            } => gdn_conv::result_precision(self, parameter, *contract, inputs),
             PlanOperation::View(_) | PlanOperation::Contiguous => inputs
                 .first()
                 .map(|value| value.precision)
@@ -816,6 +838,7 @@
                     PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
+                        | PlanOperation::TypedCausalConvSilu { .. }
                 ) =>
             {
                 Ok(ActivationPrecision::F32)
@@ -1082,6 +1105,15 @@
             input.shape,
             None,
         )
+    }
+    fn causal_conv_silu_profiled(
+        &mut self,
+        parameter: SourceParameter,
+        input: &PlanValue,
+        history: &PlanValue,
+        contract: CausalConvSiluContract,
+    ) -> Result<CausalConvValues<PlanValue>> {
+        gdn_conv::record(self, parameter, input, history, contract)
     }
     fn embed(&mut self, tensor: SourceTensor, token: ParameterRef) -> Result<PlanValue> {
         self.source_tensor(&tensor)?;
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -608,3 +608,45 @@
         }
     }
 }
+
+/// Pure original window only; proposed history is not a StateEffect.
+pub(crate) fn gdn_convolution_silu_slice<B: MathematicalBackend>(
+    source: &impl OperatorSliceSource,
+    math: &mut B,
+    layer: usize,
+    input: &B::Value,
+    history: &B::Value,
+) -> Result<crate::accelerator::program::CausalConvValues<B::Value>> {
+    use crate::accelerator::program::{ActivationPrecision, CausalConvSiluContract};
+    let (parameter, channels) = crate::operator_slice::gdn_conv::convolution_role(source, layer)?;
+    if math.elements(input) != channels
+        || math.elements(history) != channels * 3
+        || [input, history]
+            .iter()
+            .any(|v| math.precision(v) != ActivationPrecision::Bf16WidenedF32)
+    {
+        return Err(Error::State(
+            "original convolution slice geometry/precision".into(),
+        ));
+    }
+    let values = math.causal_conv_silu_profiled(
+        parameter,
+        input,
+        history,
+        CausalConvSiluContract::TorchCpuBf16Avx2,
+    )?;
+    for (value, elements) in [
+        (&values.raw, channels),
+        (&values.activated, channels),
+        (&values.next_history, channels * 3),
+    ] {
+        if math.elements(value) != elements
+            || math.precision(value) != ActivationPrecision::Bf16WidenedF32
+        {
+            return Err(Error::State(
+                "original convolution region result geometry/precision".into(),
+            ));
+        }
+    }
+    Ok(values)
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/gdn_conv_slice.rs
@@ -0,0 +1,220 @@
+//! Pure original causal convolution + SiLU on explicit observed operands.
+//! No cache owner, producer/frontier assertion or full-model eligibility.
+use super::OperatorSliceSource;
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{
+    ActivationPrecision, CausalConvSiluContract, CausalConvValues, PlanBuilder, PlanValue,
+    ProgramNumericalProfile, SourceParameter,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, LoadedQwenModel, Result};
+
+pub(crate) fn convolution_role(
+    source: &impl OperatorSliceSource,
+    layer: usize,
+) -> Result<(SourceParameter, usize)> {
+    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || !source.source_identity().is_bound()
+        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
+    {
+        return Err(Error::Unsupported(
+            "closed original convolution profile/layer".into(),
+        ));
+    }
+    let parameter = source.parameter_descriptor(QwenParameterRole::GdnConvolution(layer))?;
+    let channels = source
+        .matrix_descriptor(QwenMatrixRole::GdnQkv(layer))?
+        .rows;
+    if parameter.source != source.source_identity()
+        || parameter.name
+            != format!("model.language_model.layers.{layer}.linear_attn.conv1d.weight")
+    {
+        return Err(Error::State("original convolution raw role/source".into()));
+    }
+    CausalConvSiluContract::TorchCpuBf16Avx2.validate(
+        &parameter,
+        channels,
+        channels
+            .checked_mul(3)
+            .ok_or_else(|| Error::Limit("convolution history width".into()))?,
+    )?;
+    Ok((parameter, channels))
+}
+
+/// Observed operand phase only; never a committed-session binding.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum ConvInputPhase {
+    /// Initial operator's implicit left padding, supplied as authenticated +0.
+    InitialPadding,
+    /// Last three entries of the original incoming `[channel,4]` cache.
+    CachedTail,
+}
+
+/// Qualifier authenticates external source/archive provenance. This type checks
+/// representation only, and never asserts a model producer or StateRoot.
+#[derive(Clone, Copy)]
+pub struct GdnConvSliceInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    phase: ConvInputPhase,
+    activation: &'a [u8],
+    history: &'a [u8],
+}
+impl<'a> GdnConvSliceInput<'a> {
+    pub fn from_bf16_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        phase: ConvInputPhase,
+        activation: &'a [u8],
+        history: &'a [u8],
+    ) -> Result<Self> {
+        let input = Self {
+            source,
+            profile,
+            phase,
+            activation,
+            history,
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
+    pub fn phase(self) -> ConvInputPhase {
+        self.phase
+    }
+    pub fn precision(self) -> ActivationPrecision {
+        ActivationPrecision::Bf16WidenedF32
+    }
+    pub fn activation_bytes(self) -> &'a [u8] {
+        self.activation
+    }
+    pub fn history_bytes(self) -> &'a [u8] {
+        self.history
+    }
+    pub(crate) fn validate(self) -> Result<usize> {
+        let channels = self.activation.len() / 2;
+        if !self.source.is_bound()
+            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+            || channels == 0
+            || channels > 16384
+            || !channels.is_multiple_of(32)
+            || self.activation.len() != channels * 2
+            || self.history.len() != channels * 6
+            || self
+                .activation
+                .chunks_exact(2)
+                .chain(self.history.chunks_exact(2))
+                .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
+            || (self.phase == ConvInputPhase::InitialPadding
+                && self.history.iter().any(|v| *v != 0))
+        {
+            return Err(Error::Malformed(
+                "private convolution phase/source/finite BF16 operands".into(),
+            ));
+        }
+        Ok(channels)
+    }
+}
+
+impl LoadedQwenModel {
+    /// Append only, to existing same-builder BF16 values; no execution admission.
+    pub fn append_gdn_convolution_silu_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &PlanValue,
+        history: &PlanValue,
+    ) -> Result<CausalConvValues<PlanValue>> {
+        self.require_operator_slice()?;
+        crate::forward::gdn_convolution_silu_slice(self, builder, layer, input, history)
+    }
+}
+
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+pub struct GdnConvolutionSiluOutput {
+    source: SourceIdentity,
+    layer: usize,
+    phase: ConvInputPhase,
+    input: crate::host_math::HostValue,
+    prior: crate::host_math::HostValue,
+    values: CausalConvValues<crate::host_math::HostValue>,
+}
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+impl GdnConvolutionSiluOutput {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    }
+    pub fn precision(&self) -> ActivationPrecision {
+        ActivationPrecision::Bf16WidenedF32
+    }
+    pub fn layer(&self) -> usize {
+        self.layer
+    }
+    pub fn phase(&self) -> ConvInputPhase {
+        self.phase
+    }
+    pub fn input(&self) -> Result<&[f32]> {
+        self.input.as_slice()
+    }
+    pub fn prior_history(&self) -> Result<&[f32]> {
+        self.prior.as_slice()
+    }
+    pub fn raw(&self) -> Result<&[f32]> {
+        self.values.raw.as_slice()
+    }
+    pub fn activated(&self) -> Result<&[f32]> {
+        self.values.activated.as_slice()
+    }
+    pub fn next_history(&self) -> Result<&[f32]> {
+        self.values.next_history.as_slice()
+    }
+}
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    pub fn append_gdn_convolution_silu_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &PlanValue,
+        history: &PlanValue,
+    ) -> Result<CausalConvValues<PlanValue>> {
+        crate::forward::gdn_convolution_silu_slice(self, builder, layer, input, history)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_gdn_convolution_silu_slice(
+        &self,
+        layer: usize,
+        input: GdnConvSliceInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<GdnConvolutionSiluOutput> {
+        let (_, channels) = convolution_role(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let (activation, prior) = math.import_gdn_conv_input(input, channels)?;
+        let values = crate::forward::gdn_convolution_silu_slice(
+            self,
+            &mut math,
+            layer,
+            &activation,
+            &prior,
+        )?;
+        if cancelled.is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State("window convolution slice cancelled".into()));
+        }
+        Ok(GdnConvolutionSiluOutput {
+            source: self.source_identity(),
+            layer,
+            phase: input.phase(),
+            input: activation,
+            prior,
+            values,
+        })
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/gdn_conv_window_tests.rs
@@ -0,0 +1,382 @@
+//! Synthetic role/wiring/lease checks; genuine trained qualification is separate.
+use super::*;
+use crate::accelerator::program::{CausalConvSiluContract, ProgramNumericalProfile};
+use crate::operator_slice::gdn_conv::{ConvInputPhase, GdnConvSliceInput};
+use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
+
+fn conv_source(fixture: &Fixture) -> Loaded {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    cfg["text_config"]["hidden_size"] = json!(256);
+    cfg["text_config"]["linear_key_head_dim"] = json!(8);
+    cfg["text_config"]["linear_value_head_dim"] = json!(8);
+    source_with_config(
+        fixture,
+        serde_json::to_vec(&cfg).unwrap(),
+        true,
+        |name, index, fallback| {
+            if name.ends_with(".conv1d.weight") {
+                if index % 4 == 3 { 0x3f80 } else { 0 }
+            } else {
+                fallback
+            }
+        },
+        |_| {},
+    )
+}
+fn bf16(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
+    values
+        .into_iter()
+        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
+        .collect()
+}
+fn private<'a>(
+    source: crate::accelerator::SourceIdentity,
+    phase: ConvInputPhase,
+    x: &'a [u8],
+    h: &'a [u8],
+) -> GdnConvSliceInput<'a> {
+    GdnConvSliceInput::from_bf16_bytes(
+        source,
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        phase,
+        x,
+        h,
+    )
+    .unwrap()
+}
+#[test]
+fn pure_window_region_matches_closed_helpers_and_preserves_canonical_history_bits() {
+    let f = Fixture::new();
+    let loaded = conv_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let x = bf16((0..32).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }));
+    let h = bf16((0..32).flat_map(|_| [1.0, -0.0, 2.0]));
+    let output = ops
+        .execute_gdn_convolution_silu_slice(
+            0,
+            private(source.source_identity(), ConvInputPhase::CachedTail, &x, &h),
+            None,
+        )
+        .unwrap();
+    assert_eq!(output.layer(), 0);
+    assert_eq!(output.phase(), ConvInputPhase::CachedTail);
+    assert_eq!(output.source_identity(), source.source_identity());
+    // Last tap 1 and three zero weights make each positive input exact.
+    same(output.input().unwrap(), output.raw().unwrap());
+    let silu = zerogravity_cpu::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16(
+        source.source_identity(),
+        output.raw().unwrap(),
+    )
+    .unwrap();
+    assert_eq!(silu.len(), output.activated().unwrap().len());
+    for (expected, actual) in silu.iter().zip(output.activated().unwrap()) {
+        let bits = expected.to_bits();
+        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
+        assert_eq!(actual.to_bits(), rounded);
+    }
+    for (channel, next) in output.next_history().unwrap().chunks_exact(3).enumerate() {
+        assert_eq!(next[0].to_bits(), (-0.0f32).to_bits());
+        assert_eq!(next[1], 2.0);
+        assert_eq!(
+            next[2].to_bits(),
+            output.input().unwrap()[channel].to_bits()
+        );
+    }
+    let snapshot = admission.arbiter.snapshot(&admission.resource).unwrap();
+    assert_eq!(reserved(&admission), baseline + 2 * 960 * 1024 + 32 * 36);
+    assert_eq!(
+        snapshot
+            .per_class
+            .get(&AllocationClass::Output)
+            .copied()
+            .unwrap_or(0),
+        32 * 36
+    );
+    assert_eq!(
+        snapshot
+            .per_class
+            .get(&AllocationClass::TransferStaging)
+            .copied()
+            .unwrap_or(0),
+        0
+    );
+    let (model, _, _) = bind(loaded);
+    assert_eq!(model.initialized_cpu_state_bytes(), 0);
+    assert!(model.new_session().is_err());
+    assert!(model.forward_plan(8, true, false).is_err());
+    drop(ops);
+    drop(source);
+    assert_eq!(output.raw().unwrap().len(), 32);
+    assert!(reserved(&admission) > 0);
+    drop(output);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn initial_padding_is_explicit_and_never_canonicalizes_signed_zero() {
+    let id = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
+    let x = bf16([1.0; 32]);
+    let zero = bf16([0.0; 96]);
+    let negative = bf16([-0.0; 96]);
+    private(id, ConvInputPhase::InitialPadding, &x, &zero);
+    assert!(
+        GdnConvSliceInput::from_bf16_bytes(
+            id,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            ConvInputPhase::InitialPadding,
+            &x,
+            &negative
+        )
+        .is_err()
+    );
+    private(id, ConvInputPhase::CachedTail, &x, &negative);
+    for bytes in [vec![0; 63], vec![0xff; 64]] {
+        assert!(
+            GdnConvSliceInput::from_bf16_bytes(
+                id,
+                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+                ConvInputPhase::CachedTail,
+                &bytes,
+                &zero
+            )
+            .is_err()
+        );
+    }
+}
+#[test]
+fn aggregate_pressure_cancel_and_live_output_never_leak_or_reuse_admission() {
+    let f = Fixture::new();
+    let loaded = conv_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let x = bf16([1.0; 32]);
+    let h = bf16([0.0; 96]);
+    let input = private(
+        source.source_identity(),
+        ConvInputPhase::InitialPadding,
+        &x,
+        &h,
+    );
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        ops.execute_gdn_convolution_silu_slice(0, input, Some(&cancelled))
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), baseline);
+    let pressure = admission
+        .arbiter
+        .admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: 128 * 1024 * 1024 - baseline,
+            }],
+        )
+        .unwrap();
+    assert!(
+        ops.execute_gdn_convolution_silu_slice(0, input, None)
+            .is_err()
+    );
+    drop(pressure);
+    assert_eq!(reserved(&admission), baseline);
+    // Leave enough for BOTH input Vecs, but one byte short of the aggregate
+    // three-result envelope. A failed second admission must retire the inputs.
+    let input_reservation = 960 * 1024 + 32 * 16;
+    let result_reservation = 960 * 1024 + 32 * 20;
+    let held = admission
+        .arbiter
+        .admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: 128 * 1024 * 1024 - baseline - input_reservation - result_reservation + 1,
+            }],
+        )
+        .unwrap();
+    let pressured = reserved(&admission);
+    assert!(
+        ops.execute_gdn_convolution_silu_slice(0, input, None)
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), pressured);
+    drop(held);
+    assert_eq!(reserved(&admission), baseline);
+    // Cancellation between actual operand import and the region, without clocks.
+    cancelled.store(false, Ordering::Release);
+    let mut math =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, Some(&cancelled), None);
+    let (activation, history) = math.import_gdn_conv_input(input, 32).unwrap();
+    let after_import = reserved(&admission);
+    assert_eq!(after_import, baseline + 960 * 1024 + 32 * 16);
+    let parameter = ops
+        .parameter_descriptor(QwenParameterRole::GdnConvolution(0))
+        .unwrap();
+    cancelled.store(true, Ordering::Release);
+    assert!(
+        math.causal_conv_silu_profiled(
+            parameter.clone(),
+            &activation,
+            &history,
+            CausalConvSiluContract::TorchCpuBf16Avx2
+        )
+        .is_err()
+    );
+    assert_eq!(reserved(&admission), after_import);
+    cancelled.store(false, Ordering::Release);
+    let values = math
+        .causal_conv_silu_profiled(
+            parameter,
+            &activation,
+            &history,
+            CausalConvSiluContract::TorchCpuBf16Avx2,
+        )
+        .unwrap();
+    let clone = values.activated.clone();
+    assert!(clone.clone().into_vec().is_err());
+    drop(values);
+    assert_eq!(clone.as_slice().unwrap().len(), 32);
+    drop(clone);
+    assert_eq!(reserved(&admission), after_import);
+    drop(math);
+    drop(activation);
+    drop(history);
+    assert_eq!(reserved(&admission), baseline);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn same_source_strided_values_foreign_issuer_and_unaware_provider_refuse() {
+    struct Unaware;
+    impl crate::accelerator::MatVecKernel for Unaware {
+        fn try_execute(
+            &self,
+            _: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            panic!("typed convolution must refuse before any provider callback")
+        }
+    }
+    let f = Fixture::new();
+    let loaded = conv_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let x = bf16([1.0; 32]);
+    let h = bf16([0.0; 96]);
+    let input = private(
+        source.source_identity(),
+        ConvInputPhase::InitialPadding,
+        &x,
+        &h,
+    );
+    let provider = Unaware;
+    let unsupported =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, None, Some(&provider));
+    assert!(unsupported.import_gdn_conv_input(input, 32).is_err());
+    assert_eq!(reserved(&admission), baseline);
+    let mut math = crate::host_math::HostMath::for_window_operator_slice(&ops, None, None);
+    let (activation, history) = math.import_gdn_conv_input(input, 32).unwrap();
+    let strided = math
+        .view(
+            &history,
+            TensorView {
+                offset: 0,
+                rows: 32,
+                width: 1,
+                row_stride: 2,
+            },
+        )
+        .unwrap();
+    let parameter = ops
+        .parameter_descriptor(QwenParameterRole::GdnConvolution(0))
+        .unwrap();
+    let before = reserved(&admission);
+    let error = math
+        .causal_conv_silu_profiled(
+            parameter.clone(),
+            &strided,
+            &history,
+            CausalConvSiluContract::TorchCpuBf16Avx2,
+        )
+        .err()
+        .unwrap();
+    assert!(error.to_string().contains("contiguous"));
+    assert_eq!(reserved(&admission), before);
+    let foreign_source = windows(&f, &loaded).0;
+    let foreign_ops = bind_windows(foreign_source, &loaded);
+    let mut foreign =
+        crate::host_math::HostMath::for_window_operator_slice(&foreign_ops, None, None);
+    assert!(
+        foreign
+            .causal_conv_silu_profiled(
+                parameter,
+                &activation,
+                &history,
+                CausalConvSiluContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+    );
+    drop(foreign);
+    drop(foreign_ops);
+    drop(math);
+    drop(strided);
+    drop(activation);
+    drop(history);
+    assert_eq!(reserved(&admission), baseline);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+
+#[test]
+fn actual_role_plan_append_uses_same_builder_and_never_adds_state() {
+    use crate::accelerator::program::{ActivationPrecision, PlanBuilder, PlanOperation};
+    let fixture = Fixture::new();
+    let loaded = conv_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let mut builder = PlanBuilder::new_operator_slice(
+        source.source_identity(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+    )
+    .unwrap();
+    let embedding = builder
+        .embed_profiled(
+            operators
+                .matrix_descriptor(QwenMatrixRole::Embedding)
+                .unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    // Synthetic embedding-produced operands solely exercise the actual source
+    // roles/same-builder path; they are not claimed as trained QKV/cache inputs.
+    let x = builder.view(&embedding, TensorView::vector(0, 32)).unwrap();
+    let h = builder
+        .view(&embedding, TensorView::vector(32, 96))
+        .unwrap();
+    let result = operators
+        .append_gdn_convolution_silu_slice(&mut builder, 0, &x, &h)
+        .unwrap();
+    assert_eq!(
+        builder.precision(&result.activated),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    let recorded = builder.finish_operator_slice(result.activated).unwrap();
+    assert_eq!(recorded.plan().nodes().len(), 4);
+    assert!(matches!(
+        recorded.plan().nodes()[3].operation,
+        PlanOperation::TypedCausalConvSilu { .. }
+    ));
+    assert!(recorded.plan().required_domains().is_empty());
+    assert!(recorded.plan().proposed_domains().is_empty());
+    assert_eq!(recorded.plan().frontier_delta(), 0);
+    drop(recorded);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/host_math/gdn_conv.rs
@@ -0,0 +1,152 @@
+//! Aggregate admitted private inputs/results for the pure original region.
+use super::*;
+use crate::accelerator::program::{CausalConvSiluContract, CausalConvValues};
+use crate::operator_slice::gdn_conv::GdnConvSliceInput;
+
+const SCRATCH: usize = 960 * 1024;
+impl HostMath<'_, '_, '_> {
+    fn conv_windows(&self) -> Result<&crate::original_binding::OriginalWindowOperators> {
+        if self.region_kernel.is_some()
+            || self.engine.is_some()
+            || self.state.is_some()
+            || self.source.program_numerical_profile()
+                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported(
+                "convolution requires private pinned CPU window contract".into(),
+            ));
+        }
+        match &self.source {
+            HostSource::Windows(source) => Ok(source),
+            HostSource::Loaded(_) => Err(Error::Unsupported(
+                "convolution requires admitted window backing".into(),
+            )),
+        }
+    }
+    fn check_conv_cancel(&self) -> Result<()> {
+        if self
+            .cancelled
+            .is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire))
+        {
+            return Err(Error::State("window convolution cancelled".into()));
+        }
+        Ok(())
+    }
+    pub(crate) fn import_gdn_conv_input(
+        &self,
+        input: GdnConvSliceInput<'_>,
+        channels: usize,
+    ) -> Result<(HostValue, HostValue)> {
+        let source = self.conv_windows()?;
+        if input.validate()? != channels
+            || input.source_identity() != source.source_identity()
+            || input.numerical_profile() != self.source.program_numerical_profile()
+        {
+            return Err(Error::State(
+                "convolution input full width/source/profile".into(),
+            ));
+        }
+        self.check_conv_cancel()?;
+        // Both allocations are charged once, before either Vec allocation.
+        let workspace = Arc::new(
+            source.source().reserve_operator_buffers(
+                channels
+                    .checked_mul(16)
+                    .ok_or_else(|| Error::Limit("convolution input byte count".into()))?,
+                SCRATCH,
+            )?,
+        );
+        let make = |bytes: &[u8]| -> Result<HostValue> {
+            let values = bytes
+                .chunks_exact(2)
+                .map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16))
+                .collect();
+            HostValue::typed(
+                values,
+                ActivationPrecision::Bf16WidenedF32,
+                source.source_identity(),
+                self.source.program_numerical_profile(),
+            )?
+            .retain_workspace(Some(workspace.clone()))
+        };
+        let result = (
+            make(input.activation_bytes())?,
+            make(input.history_bytes())?,
+        );
+        self.check_conv_cancel()?;
+        Ok(result)
+    }
+    pub(super) fn execute_causal_conv_silu(
+        &self,
+        parameter: SourceParameter,
+        input: &HostValue,
+        history: &HostValue,
+        contract: CausalConvSiluContract,
+    ) -> Result<CausalConvValues<HostValue>> {
+        let source = self.conv_windows()?;
+        let channels = contract.validate(&parameter, input.elements, history.elements)?;
+        if parameter.source != source.source_identity() {
+            return Err(Error::State("convolution parameter source".into()));
+        }
+        for value in [input, history] {
+            if value.source != Some(source.source_identity())
+                || value.profile != Some(self.source.program_numerical_profile())
+                || value.precision != ActivationPrecision::Bf16WidenedF32
+            {
+                return Err(Error::State(
+                    "convolution activation source/profile/precision".into(),
+                ));
+            }
+            let workspace = value.data.workspace.as_ref().ok_or_else(|| {
+                Error::State("convolution activation missing physical backing".into())
+            })?;
+            if !source.source().same_issuer(workspace.source()) {
+                return Err(Error::State(
+                    "convolution activation foreign source issuer".into(),
+                ));
+            }
+        }
+        // Contiguity and source membership checked before allocation or math.
+        let activation = input.as_slice()?;
+        let prior = history.as_slice()?;
+        ActivationPrecision::Bf16WidenedF32.validate_values(activation)?;
+        ActivationPrecision::Bf16WidenedF32.validate_values(prior)?;
+        self.check_conv_cancel()?;
+        // Three retained results: C raw + C activated + 3C history, F32 backing.
+        // Scratch conservatively covers raw FP32 + SiLU FP32 + rounding Vec.
+        let workspace = Arc::new(
+            source.source().reserve_operator_buffers(
+                channels
+                    .checked_mul(20)
+                    .ok_or_else(|| Error::Limit("convolution output byte count".into()))?,
+                SCRATCH,
+            )?,
+        );
+        let completed = source.convolve(&parameter, activation, prior, self.cancelled)?;
+        let silu =
+            zerogravity_cpu::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16(
+                source.source_identity(),
+                &completed.raw_bf16,
+            )?;
+        let activated = silu
+            .into_iter()
+            .map(round_bf16)
+            .collect::<Result<Vec<_>>>()?;
+        let make = |values| -> Result<HostValue> {
+            HostValue::typed(
+                values,
+                ActivationPrecision::Bf16WidenedF32,
+                source.source_identity(),
+                self.source.program_numerical_profile(),
+            )?
+            .retain_workspace(Some(workspace.clone()))
+        };
+        let result = CausalConvValues {
+            raw: make(completed.raw_bf16)?,
+            activated: make(activated)?,
+            next_history: make(completed.next_history)?,
+        };
+        self.check_conv_cancel()?;
+        Ok(result)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -507,6 +507,25 @@
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         let output = output.retain_workspace(workspace)?;
         Ok(output)
+    }
+    fn causal_conv_silu_profiled(
+        &mut self,
+        parameter: SourceParameter,
+        input: &HostValue,
+        history: &HostValue,
+        contract: crate::accelerator::program::CausalConvSiluContract,
+    ) -> Result<crate::accelerator::program::CausalConvValues<HostValue>> {
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        {
+            self.execute_causal_conv_silu(parameter, input, history, contract)
+        }
+        #[cfg(not(all(feature = "original-hf-binding", target_os = "linux")))]
+        {
+            let _ = (parameter, input, history, contract);
+            Err(Error::Unsupported(
+                "causal convolution requires leased original windows".into(),
+            ))
+        }
     }
     fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<HostValue> {
         self.legacy_precision(&[])?;
@@ -1031,3 +1050,6 @@
         }
     }
 }
+
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+mod gdn_conv;
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -231,3 +231,6 @@
 /// Pure attention projections/QK norms; no rotary/cache/session admission.
 #[path = "attention_slice.rs"]
 pub mod attention;
+
+#[path = "gdn_conv_slice.rs"]
+pub mod gdn_conv;
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -379,3 +379,6 @@
 
 #[path = "attention_slice_tests.rs"]
 mod attention;
+
+#[path = "gdn_conv_window_tests.rs"]
+mod gdn_conv;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window/gdn_conv.rs
@@ -0,0 +1,59 @@
+//! Original full rank3 stage; arithmetic remains in accepted CPU helpers.
+use super::*;
+impl OriginalWindowOperators {
+    pub(crate) fn convolve(
+        &self,
+        parameter: &SourceParameter,
+        input: &[f32],
+        history: &[f32],
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<zerogravity_cpu::original_hf::conv::CausalConv4Completed> {
+        let mut found = false;
+        for (layer, kind) in self.geometry().layers.iter().enumerate() {
+            if *kind != crate::model::LayerKind::LinearAttention {
+                continue;
+            }
+            let (actual, channels) =
+                crate::operator_slice::gdn_conv::convolution_role(self, layer)?;
+            if actual == *parameter && input.len() == channels && history.len() == channels * 3 {
+                found = true;
+                break;
+            }
+        }
+        if !found {
+            return Err(Error::State(
+                "window convolution original role/geometry".into(),
+            ));
+        }
+        check_cancel(cancelled)?;
+        let raw = self.source().conv4_parameter(&parameter.name)?;
+        raw.validate_source_parameter(parameter)?;
+        let window = match cancelled {
+            Some(flag) => {
+                self.source()
+                    .read_parameter_cancellable(&raw, 0..raw.elements(), flag)?
+            }
+            None => self.source().read_parameter(&raw, 0..raw.elements())?,
+        };
+        if !window.is_complete() || window.element_range() != (0..raw.elements()) {
+            return Err(Error::State(
+                "window convolution incomplete raw parameter".into(),
+            ));
+        }
+        let output = zerogravity_cpu::original_hf::conv::causal_step_bf16_4(
+            zerogravity_cpu::original_hf::conv::CausalConv4Inputs {
+                source: self.source_identity(),
+                tensor_name: &parameter.name,
+                weight: zerogravity_core::source::SourceTensorRef {
+                    kind: raw.tensor().kind,
+                    shape: &raw.tensor().shape,
+                    packed_bytes: window.as_slice(),
+                },
+                activation: input,
+                history,
+            },
+        )?;
+        check_cancel(cancelled)?;
+        Ok(output)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/original_binding/window.rs
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window.rs
@@ -415,3 +415,6 @@
         self.projected.as_slice()
     }
 }
+
+#[cfg(feature = "cpu")]
+mod gdn_conv;

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-v2/BOUNDED_ORACLE_IO_V2.patch

Original UTF-8 bytes: 3395; SHA256 `edd6d71d802a44e24aa7590c1f0ac2a4ffb3f54f4e9ed70078563137f6c2ba72`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/reference.py
+++ b/reference.py
@@ -7,6 +7,7 @@
 import os
 from pathlib import Path
 import struct
+import stat
 
 
 def require(ok, message):
@@ -18,12 +19,23 @@
     return hashlib.sha256(raw).hexdigest()
 
 
+def small(path, cap):
+    with os.fdopen(os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW),'rb') as opened:
+        before=os.fstat(opened.fileno())
+        require(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'bounded regular operator metadata/operand')
+        data=opened.read(cap+1)
+        after=os.fstat(opened.fileno())
+    identity=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
+    require(len(data)<=cap and identity(before)==identity(after),'bounded immutable selected input FD')
+    return data
+
+
 def checked(root, record):
     name=record['file']
     require(type(name) is str and Path(name).name == name, 'contained selected operand')
     path=root/name
     require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 128*1024, 'bounded raw operand')
-    raw=path.read_bytes()
+    raw=small(path,128*1024)
     require(len(raw) == record['bytes'] and sha(raw) == record['sha256'], 'exact selected operand pin')
     return raw
 
@@ -59,8 +71,8 @@
     require(torch.backends.cpu.get_cpu_capability() == 'AVX2' and not torch.cuda.is_available()
             and torch.backends.mkldnn.enabled, 'original CPU-only AVX2/mkldnn policy')
     source=Path('/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py')
-    require(sha(source.read_bytes()) == '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b', 'unchanged official convolution caller')
-    document=args.inputs.read_bytes()
+    require(sha(small(source,1024*1024)) == '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b', 'unchanged official convolution caller')
+    document=small(args.inputs,128*1024)
     require(len(document)<=128*1024 and sha(document)==args.inputs_sha256,'exact prepared inputs metadata')
     inputs=json.loads(document)
     require(inputs['schema']=='zerogravity.original-conv-fragment.inputs.v1' and inputs['profile']=='source-36-f32-restored','genuine input protocol')
--- a/run_replay.py
+++ b/run_replay.py
@@ -36,12 +36,12 @@
 
 
 def frozen():
-    manifest = read_json(HERE / "freeze-manifest.json")
+    manifest = read_json(HERE / "freeze-manifest-v2.json")
     for item in manifest["files"]:
         pin(HERE / item["file"], item)
     for item in manifest["main_sources"]:
         pin(item["path"], item)
-    return sha(HERE / "freeze-manifest.json")
+    return sha(HERE / "freeze-manifest-v2.json")
 
 
 def execute(binary, binary_fd, inputs, inputs_sha, reference, reference_sha, output, model, cpus, freeze, source_before):
--- a/ROOT-QUALIFICATION.md
+++ b/ROOT-QUALIFICATION.md
@@ -1,6 +1,7 @@
 # ROOT-only genuine qualification
 
-Status: PREPARED / UNQUALIFIED. Author parsed/formatted source only. No raw
+Status: PREPARED / UNQUALIFIED. Latest additive freeze: `freeze-manifest-v2.json`;
+initial `freeze-manifest.json`/`frozen-prepared/` retained unchanged. Author parsed/formatted source only. No raw
 model/tensor payload was hashed or operator/reference/harness executed.
 `bindings.json` copies expected source/capture hashes from genuine existing
 receipts; author independently read/pinned only small source/metadata.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-v2/GPU_REFUSAL.patch

Original UTF-8 bytes: 1645; SHA256 `f09d057c3c65d75907f85ade140c45c05465d20b9b831a254b86475b8299e419`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -838,7 +838,8 @@
             match &node.operation {
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
-                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedOrdinaryRms { .. }
+                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                     "resident batch has no qualified typed lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -348,6 +348,7 @@
                     PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
+                        | PlanOperation::TypedCausalConvSilu { .. }
                 )
             })
         {
@@ -1429,7 +1430,8 @@
             match &node.operation {
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
-                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedOrdinaryRms { .. }
+                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                     "resident typed operation has no qualified lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-v2/NEW_DELTA.patch

Original UTF-8 bytes: 52773; SHA256 `b53625fa619545bfa1a8604ca000bd08a7d2a2a41a9bff0aac99116e6fbfceea`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-artifacts/src/source_weights/tensor_window/conv_parameter.rs
@@ -0,0 +1,36 @@
+//! Sealed original BF16 depthwise four-tap parameter transport only.
+use super::*;
+impl VerifiedTensorSource {
+    /// Original rank3 metadata and raw elements, without model eligibility.
+    /// `parameter()` remains rank1-only; no synthetic flattened tensor exists.
+    pub fn conv4_parameter(&self, name: &str) -> Result<VerifiedParameter> {
+        let location = self
+            .0
+            .tensors
+            .get(name)
+            .ok_or_else(|| Error::Malformed("original conv4 parameter missing".into()))?;
+        let [channels, 1, 4] = location.info.shape.as_slice() else {
+            return Err(Error::Unsupported("original conv4 raw rank/shape".into()));
+        };
+        let channels = usize::try_from(*channels)
+            .ok()
+            .filter(|n| *n > 0 && *n <= 16384 && n.is_multiple_of(32))
+            .ok_or_else(|| Error::Limit("original conv4 channel bound".into()))?;
+        let elements = channels
+            .checked_mul(4)
+            .ok_or_else(|| Error::Limit("original conv4 element count".into()))?;
+        let bytes = elements
+            .checked_mul(2)
+            .ok_or_else(|| Error::Limit("original conv4 byte count".into()))?;
+        if location.info.kind != QuantKind::Bf16 || location.info.byte_len != bytes as u64 {
+            return Err(Error::Unsupported("original conv4 raw dtype/length".into()));
+        }
+        Ok(VerifiedParameter {
+            location: location.clone(),
+            owner: self.0.clone(),
+            elements,
+            element_bytes: 2,
+            bytes,
+        })
+    }
+}
--- a/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs
+++ b/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs
@@ -103,7 +103,7 @@
     matrix: VerifiedMatrix,
 }
 
-/// Sealed full original rank1 parameter; no matrix axes or adjusted values.
+/// Sealed full original rank1 or closed conv4 parameter; no adjusted values.
 #[derive(Clone)]
 pub struct VerifiedParameter {
     location: Arc<TensorLocation>,
@@ -493,7 +493,7 @@
             bytes,
         })
     }
-    /// Original Safetensors rank1 is contiguous; arbitrary strides have no API.
+    /// Original parameter elements are contiguous; arbitrary strides have no API.
     pub fn read_parameter(
         &self,
         parameter: &VerifiedParameter,
@@ -802,3 +802,5 @@
         Ok(self.position)
     }
 }
+
+mod conv_parameter;
--- /dev/null
+++ b/crates/zerogravity-artifacts/tests/tensor_window/conv4.rs
@@ -0,0 +1,101 @@
+//! Synthetic transport only; no trained convolution or model qualification.
+use super::*;
+fn fixture(dtype: &str, shape: &[u64], bytes: usize) -> Fixture {
+    let mut fixture = Fixture::new();
+    let raw = (0..bytes / 2)
+        .flat_map(|i| (0x3f80u16 + (i % 8) as u16).to_le_bytes())
+        .collect::<Vec<_>>();
+    fixture.second = safetensors("conv", dtype, shape, &raw);
+    fixture.index = serde_json::to_vec(
+        &json!({"metadata":{"total_size":64+raw.len()},"weight_map":{"matrix":A,"conv":B}}),
+    )
+    .unwrap();
+    fs::write(fixture.path.join(B), &fixture.second).unwrap();
+    fixture
+}
+#[test]
+fn full_original_rank3_and_partial_range_keep_truthful_metadata_and_owner() {
+    let f = fixture("BF16", &[32, 1, 4], 256);
+    let host = admission(64 * 1024 * 1024, "host");
+    let source = f.plan(&host).unwrap().open().unwrap();
+    let before = reserved(&host);
+    assert!(source.parameter("conv").is_err());
+    let raw = source.conv4_parameter("conv").unwrap();
+    let descriptor = raw.source_parameter();
+    assert_eq!(descriptor.shape, [32, 1, 4]);
+    assert_eq!(descriptor.bytes, 256);
+    raw.validate_source_parameter(&descriptor).unwrap();
+    let partial = source.read_parameter(&raw, 4..8).unwrap();
+    assert!(!partial.is_complete());
+    assert_eq!(partial.parameter().tensor().shape, [32, 1, 4]);
+    assert_eq!(partial.element_range(), 4..8);
+    drop(partial);
+    assert_eq!(reserved(&host), before);
+    let full = source.read_parameter(&raw, 0..128).unwrap();
+    assert!(full.is_complete());
+    assert_eq!(full.as_slice().len(), 256);
+    assert!(full.charged_staging_bytes() >= 256);
+    assert!(source.read_parameter(&raw, 0..129).is_err());
+    assert!(source.read_parameter(&raw, 4..4).is_err());
+    let foreign = f.plan(&host).unwrap().open().unwrap();
+    assert!(foreign.read_parameter(&raw, 0..128).is_err());
+    drop(foreign);
+    drop(descriptor);
+    drop(raw);
+    drop(source);
+    assert_eq!(full.as_slice().len(), 256);
+    assert!(reserved(&host) > 0);
+    drop(full);
+    assert_eq!(reserved(&host), 0);
+}
+#[test]
+fn geometry_corruption_cancel_and_pressure_refuse_without_leases() {
+    for (dtype, shape, bytes) in [
+        ("F32", vec![32, 1, 4], 512),
+        ("BF16", vec![128], 256),
+        ("BF16", vec![32, 4, 1], 256),
+        ("BF16", vec![16, 1, 4], 128),
+    ] {
+        let f = fixture(dtype, &shape, bytes);
+        let host = admission(64 * 1024 * 1024, "host");
+        let source = f.plan(&host).unwrap().open().unwrap();
+        assert!(source.conv4_parameter("conv").is_err());
+        drop(source);
+        assert_eq!(reserved(&host), 0);
+    }
+    let f = fixture("BF16", &[32, 1, 4], 256);
+    let host = admission(64 * 1024 * 1024, "host");
+    let source = f.plan(&host).unwrap().open().unwrap();
+    let raw = source.conv4_parameter("conv").unwrap();
+    let baseline = reserved(&host);
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        source
+            .read_parameter_cancellable(&raw, 0..128, &cancelled)
+            .is_err()
+    );
+    assert_eq!(reserved(&host), baseline);
+    let remaining = 64 * 1024 * 1024 - baseline;
+    let pressure = host
+        .arbiter
+        .admit(
+            &host.resource,
+            host.owner.clone(),
+            &[zerogravity_core::resource_arbiter::AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: remaining,
+            }],
+        )
+        .unwrap();
+    assert!(source.read_parameter(&raw, 0..128).is_err());
+    drop(pressure);
+    assert_eq!(reserved(&host), baseline);
+    let mut corrupt = f.second.clone();
+    *corrupt.last_mut().unwrap() ^= 1;
+    fs::write(f.path.join(B), corrupt).unwrap();
+    assert!(source.read_parameter(&raw, 0..128).is_err());
+    assert_eq!(reserved(&host), baseline);
+    drop(raw);
+    drop(source);
+    assert_eq!(reserved(&host), 0);
+}
--- a/crates/zerogravity-artifacts/tests/tensor_window.rs
+++ b/crates/zerogravity-artifacts/tests/tensor_window.rs
@@ -944,3 +944,6 @@
     assert_eq!(reserved(&host), 0);
     assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
 }
+
+#[path = "tensor_window/conv4.rs"]
+mod conv4;
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/gdn_conv.rs
@@ -0,0 +1,228 @@
+//! Closed pinned CPU region; three private BF16 outputs, no state effects.
+use super::*;
+
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum CausalConvSiluContract {
+    TorchCpuBf16Avx2,
+}
+impl CausalConvSiluContract {
+    /// Original raw depthwise `[channel,1,4]`, never an adjusted vector.
+    pub fn validate(
+        self,
+        parameter: &SourceParameter,
+        input: usize,
+        history: usize,
+    ) -> Result<usize> {
+        let channels = input;
+        if !parameter.source.is_bound() || parameter.name.is_empty() {
+            return Err(Error::State("causal convolution source binding".into()));
+        }
+        if channels == 0
+            || channels > 16384
+            || !channels.is_multiple_of(32)
+            || parameter.kind != QuantKind::Bf16
+            || parameter.shape.as_slice() != [channels as u64, 1, 4]
+            || channels.checked_mul(8) != Some(parameter.bytes)
+            || channels.checked_mul(3) != Some(history)
+        {
+            return Err(Error::Unsupported(
+                "closed causal convolution raw geometry/dtype".into(),
+            ));
+        }
+        Ok(channels)
+    }
+}
+
+#[derive(Clone)]
+pub struct CausalConvValues<V> {
+    pub raw: V,
+    pub activated: V,
+    /// Canonical `[channel,3]`, oldest to newest. Caller owns publication.
+    pub next_history: V,
+}
+
+pub(super) fn result_precision(
+    builder: &PlanBuilder,
+    parameter: &SourceParameter,
+    contract: CausalConvSiluContract,
+    inputs: &[PlanValue],
+) -> Result<ActivationPrecision> {
+    if !builder.operator_slice
+        || builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || parameter.source != builder.source
+        || inputs.len() != 2
+        || inputs
+            .iter()
+            .any(|v| v.precision != ActivationPrecision::Bf16WidenedF32)
+    {
+        return Err(Error::Unsupported(
+            "causal convolution profile/source/precision/slice".into(),
+        ));
+    }
+    contract.validate(parameter, inputs[0].elements, inputs[1].elements)?;
+    Ok(ActivationPrecision::Bf16WidenedF32)
+}
+pub(super) fn record(
+    builder: &mut PlanBuilder,
+    parameter: SourceParameter,
+    input: &PlanValue,
+    history: &PlanValue,
+    contract: CausalConvSiluContract,
+) -> Result<CausalConvValues<PlanValue>> {
+    let channels = contract.validate(&parameter, input.elements, history.elements)?;
+    let name = parameter.name.clone();
+    let values = builder.record(
+        &name,
+        PlanOperation::TypedCausalConvSilu {
+            parameter,
+            contract,
+        },
+        &[*input, *history],
+        &[
+            ValueShape::Vector(channels),
+            ValueShape::Vector(channels),
+            ValueShape::Vector(channels * 3),
+        ],
+        None,
+    )?;
+    let [raw, activated, next_history]: [PlanValue; 3] = values
+        .try_into()
+        .map_err(|_| Error::State("causal convolution output arity".into()))?;
+    Ok(CausalConvValues {
+        raw,
+        activated,
+        next_history,
+    })
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    fn fixture() -> (PlanBuilder, SourceParameter) {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        (
+            PlanBuilder::new_operator_slice(
+                source,
+                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            )
+            .unwrap(),
+            SourceParameter {
+                source,
+                name: "original.conv.weight".into(),
+                kind: QuantKind::Bf16,
+                shape: vec![32, 1, 4],
+                bytes: 256,
+            },
+        )
+    }
+    // Synthetic embedding metadata only; no original model producer claim.
+    fn input(builder: &mut PlanBuilder, width: usize) -> PlanValue {
+        builder
+            .embed_profiled(
+                SourceTensor {
+                    source: builder.source,
+                    name: format!("synthetic.{width}"),
+                    kind: QuantKind::Bf16,
+                    axes: MatrixAxes::OutputInput,
+                    columns: width,
+                    rows: 2,
+                },
+                PlanBuilder::inputs().token,
+                EmbeddingContract::Bf16OutputInput,
+            )
+            .unwrap()
+    }
+    #[test]
+    fn pure_three_outputs_have_bf16_precision_and_no_state_effect() {
+        let (mut b, p) = fixture();
+        let x = input(&mut b, 32);
+        let h = input(&mut b, 96);
+        let output = b
+            .causal_conv_silu_profiled(p, &x, &h, CausalConvSiluContract::TorchCpuBf16Avx2)
+            .unwrap();
+        for (v, n) in [
+            (&output.raw, 32),
+            (&output.activated, 32),
+            (&output.next_history, 96),
+        ] {
+            assert_eq!(b.elements(v), n);
+            assert_eq!(b.precision(v), ActivationPrecision::Bf16WidenedF32);
+        }
+        let recorded = b.finish_operator_slice(output.activated).unwrap();
+        assert_eq!(recorded.plan().nodes().len(), 3);
+        assert!(recorded.plan().required_domains().is_empty());
+        assert!(recorded.plan().proposed_domains().is_empty());
+        assert_eq!(recorded.plan().frontier_delta(), 0);
+    }
+    #[test]
+    fn raw_shape_dtype_length_source_and_history_are_closed() {
+        let (mut b, p) = fixture();
+        let x = input(&mut b, 32);
+        let h = input(&mut b, 96);
+        let contract = CausalConvSiluContract::TorchCpuBf16Avx2;
+        for wrong in [
+            SourceParameter {
+                source: SourceIdentity::for_new_allocation().unwrap(),
+                ..p.clone()
+            },
+            SourceParameter {
+                shape: vec![128],
+                ..p.clone()
+            },
+            SourceParameter {
+                shape: vec![32, 4, 1],
+                ..p.clone()
+            },
+            SourceParameter {
+                kind: QuantKind::F32,
+                ..p.clone()
+            },
+            SourceParameter {
+                bytes: 255,
+                ..p.clone()
+            },
+        ] {
+            assert!(
+                b.causal_conv_silu_profiled(wrong, &x, &h, contract)
+                    .is_err()
+            );
+        }
+        let short = input(&mut b, 95);
+        assert!(
+            b.causal_conv_silu_profiled(p.clone(), &x, &short, contract)
+                .is_err()
+        );
+        let mut foreign = PlanBuilder::new_operator_slice(b.source, b.profile).unwrap();
+        let other = input(&mut foreign, 32);
+        assert!(
+            b.causal_conv_silu_profiled(p, &other, &h, contract)
+                .is_err()
+        );
+    }
+    #[test]
+    fn valid_geometry_strided_values_require_explicit_pack() {
+        let (mut b, p) = fixture();
+        let base = input(&mut b, 64);
+        let h = input(&mut b, 96);
+        let strided = b
+            .view(
+                &base,
+                TensorView {
+                    offset: 0,
+                    rows: 32,
+                    width: 1,
+                    row_stride: 2,
+                },
+            )
+            .unwrap();
+        let contract = CausalConvSiluContract::TorchCpuBf16Avx2;
+        let error = b
+            .causal_conv_silu_profiled(p.clone(), &strided, &h, contract)
+            .err()
+            .unwrap();
+        assert!(error.to_string().contains("contiguous"));
+        let packed = b.contiguous(&strided).unwrap();
+        b.causal_conv_silu_profiled(p, &packed, &h, contract)
+            .unwrap();
+    }
+}
--- a/crates/zerogravity-core/src/accelerator/program/typed_tests.rs
+++ b/crates/zerogravity-core/src/accelerator/program/typed_tests.rs
@@ -525,3 +525,24 @@
     );
     assert_eq!(slice.plan().frontier_delta(), 0);
 }
+
+#[test]
+fn unaware_backend_causal_conv_refuses_before_any_callback() {
+    let mut backend = UnawareBackend;
+    let parameter = SourceParameter {
+        source: SourceIdentity::for_new_allocation().unwrap(),
+        name: "actual.conv.weight".into(),
+        kind: QuantKind::Bf16,
+        shape: vec![32, 1, 4],
+        bytes: 256,
+    };
+    assert!(matches!(
+        backend.causal_conv_silu_profiled(
+            parameter,
+            &vec![0.0; 32],
+            &vec![0.0; 96],
+            CausalConvSiluContract::TorchCpuBf16Avx2
+        ),
+        Err(Error::Unsupported(_))
+    ));
+}
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -12,6 +12,8 @@
 use std::sync::Arc;
 use std::sync::atomic::{AtomicU64, Ordering};
 
+mod gdn_conv;
+pub use gdn_conv::{CausalConvSiluContract, CausalConvValues};
 mod typed;
 pub use typed::{
     ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
@@ -280,6 +282,18 @@
             "backend does not implement explicit ordinary RMS contract".into(),
         ))
     }
+    /// Pure original convolution window; no state owner or implicit fallback.
+    fn causal_conv_silu_profiled(
+        &mut self,
+        _parameter: SourceParameter,
+        _input: &Self::Value,
+        _history: &Self::Value,
+        _contract: CausalConvSiluContract,
+    ) -> Result<CausalConvValues<Self::Value>> {
+        Err(Error::Unsupported(
+            "backend does not implement explicit causal convolution SiLU contract".into(),
+        ))
+    }
     fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
     fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
     fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
@@ -452,6 +466,10 @@
     TypedProject {
         tensor: SourceTensor,
         contract: ProjectionContract,
+    },
+    TypedCausalConvSilu {
+        parameter: SourceParameter,
+        contract: CausalConvSiluContract,
     },
     TypedOrdinaryRms {
         parameter: SourceParameter,
@@ -800,6 +818,10 @@
                 contract.validate(parameter, inputs[0].elements)?;
                 Ok(ActivationPrecision::Bf16WidenedF32)
             }
+            PlanOperation::TypedCausalConvSilu {
+                parameter,
+                contract,
+            } => gdn_conv::result_precision(self, parameter, *contract, inputs),
             PlanOperation::View(_) | PlanOperation::Contiguous => inputs
                 .first()
                 .map(|value| value.precision)
@@ -816,6 +838,7 @@
                     PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
+                        | PlanOperation::TypedCausalConvSilu { .. }
                 ) =>
             {
                 Ok(ActivationPrecision::F32)
@@ -1082,6 +1105,15 @@
             input.shape,
             None,
         )
+    }
+    fn causal_conv_silu_profiled(
+        &mut self,
+        parameter: SourceParameter,
+        input: &PlanValue,
+        history: &PlanValue,
+        contract: CausalConvSiluContract,
+    ) -> Result<CausalConvValues<PlanValue>> {
+        gdn_conv::record(self, parameter, input, history, contract)
     }
     fn embed(&mut self, tensor: SourceTensor, token: ParameterRef) -> Result<PlanValue> {
         self.source_tensor(&tensor)?;
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -608,3 +608,45 @@
         }
     }
 }
+
+/// Pure original window only; proposed history is not a StateEffect.
+pub(crate) fn gdn_convolution_silu_slice<B: MathematicalBackend>(
+    source: &impl OperatorSliceSource,
+    math: &mut B,
+    layer: usize,
+    input: &B::Value,
+    history: &B::Value,
+) -> Result<crate::accelerator::program::CausalConvValues<B::Value>> {
+    use crate::accelerator::program::{ActivationPrecision, CausalConvSiluContract};
+    let (parameter, channels) = crate::operator_slice::gdn_conv::convolution_role(source, layer)?;
+    if math.elements(input) != channels
+        || math.elements(history) != channels * 3
+        || [input, history]
+            .iter()
+            .any(|v| math.precision(v) != ActivationPrecision::Bf16WidenedF32)
+    {
+        return Err(Error::State(
+            "original convolution slice geometry/precision".into(),
+        ));
+    }
+    let values = math.causal_conv_silu_profiled(
+        parameter,
+        input,
+        history,
+        CausalConvSiluContract::TorchCpuBf16Avx2,
+    )?;
+    for (value, elements) in [
+        (&values.raw, channels),
+        (&values.activated, channels),
+        (&values.next_history, channels * 3),
+    ] {
+        if math.elements(value) != elements
+            || math.precision(value) != ActivationPrecision::Bf16WidenedF32
+        {
+            return Err(Error::State(
+                "original convolution region result geometry/precision".into(),
+            ));
+        }
+    }
+    Ok(values)
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/gdn_conv_slice.rs
@@ -0,0 +1,220 @@
+//! Pure original causal convolution + SiLU on explicit observed operands.
+//! No cache owner, producer/frontier assertion or full-model eligibility.
+use super::OperatorSliceSource;
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{
+    ActivationPrecision, CausalConvSiluContract, CausalConvValues, PlanBuilder, PlanValue,
+    ProgramNumericalProfile, SourceParameter,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, LoadedQwenModel, Result};
+
+pub(crate) fn convolution_role(
+    source: &impl OperatorSliceSource,
+    layer: usize,
+) -> Result<(SourceParameter, usize)> {
+    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || !source.source_identity().is_bound()
+        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
+    {
+        return Err(Error::Unsupported(
+            "closed original convolution profile/layer".into(),
+        ));
+    }
+    let parameter = source.parameter_descriptor(QwenParameterRole::GdnConvolution(layer))?;
+    let channels = source
+        .matrix_descriptor(QwenMatrixRole::GdnQkv(layer))?
+        .rows;
+    if parameter.source != source.source_identity()
+        || parameter.name
+            != format!("model.language_model.layers.{layer}.linear_attn.conv1d.weight")
+    {
+        return Err(Error::State("original convolution raw role/source".into()));
+    }
+    CausalConvSiluContract::TorchCpuBf16Avx2.validate(
+        &parameter,
+        channels,
+        channels
+            .checked_mul(3)
+            .ok_or_else(|| Error::Limit("convolution history width".into()))?,
+    )?;
+    Ok((parameter, channels))
+}
+
+/// Observed operand phase only; never a committed-session binding.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum ConvInputPhase {
+    /// Initial operator's implicit left padding, supplied as authenticated +0.
+    InitialPadding,
+    /// Last three entries of the original incoming `[channel,4]` cache.
+    CachedTail,
+}
+
+/// Qualifier authenticates external source/archive provenance. This type checks
+/// representation only, and never asserts a model producer or StateRoot.
+#[derive(Clone, Copy)]
+pub struct GdnConvSliceInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    phase: ConvInputPhase,
+    activation: &'a [u8],
+    history: &'a [u8],
+}
+impl<'a> GdnConvSliceInput<'a> {
+    pub fn from_bf16_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        phase: ConvInputPhase,
+        activation: &'a [u8],
+        history: &'a [u8],
+    ) -> Result<Self> {
+        let input = Self {
+            source,
+            profile,
+            phase,
+            activation,
+            history,
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
+    pub fn phase(self) -> ConvInputPhase {
+        self.phase
+    }
+    pub fn precision(self) -> ActivationPrecision {
+        ActivationPrecision::Bf16WidenedF32
+    }
+    pub fn activation_bytes(self) -> &'a [u8] {
+        self.activation
+    }
+    pub fn history_bytes(self) -> &'a [u8] {
+        self.history
+    }
+    pub(crate) fn validate(self) -> Result<usize> {
+        let channels = self.activation.len() / 2;
+        if !self.source.is_bound()
+            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+            || channels == 0
+            || channels > 16384
+            || !channels.is_multiple_of(32)
+            || self.activation.len() != channels * 2
+            || self.history.len() != channels * 6
+            || self
+                .activation
+                .chunks_exact(2)
+                .chain(self.history.chunks_exact(2))
+                .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
+            || (self.phase == ConvInputPhase::InitialPadding
+                && self.history.iter().any(|v| *v != 0))
+        {
+            return Err(Error::Malformed(
+                "private convolution phase/source/finite BF16 operands".into(),
+            ));
+        }
+        Ok(channels)
+    }
+}
+
+impl LoadedQwenModel {
+    /// Append only, to existing same-builder BF16 values; no execution admission.
+    pub fn append_gdn_convolution_silu_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &PlanValue,
+        history: &PlanValue,
+    ) -> Result<CausalConvValues<PlanValue>> {
+        self.require_operator_slice()?;
+        crate::forward::gdn_convolution_silu_slice(self, builder, layer, input, history)
+    }
+}
+
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+pub struct GdnConvolutionSiluOutput {
+    source: SourceIdentity,
+    layer: usize,
+    phase: ConvInputPhase,
+    input: crate::host_math::HostValue,
+    prior: crate::host_math::HostValue,
+    values: CausalConvValues<crate::host_math::HostValue>,
+}
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+impl GdnConvolutionSiluOutput {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    }
+    pub fn precision(&self) -> ActivationPrecision {
+        ActivationPrecision::Bf16WidenedF32
+    }
+    pub fn layer(&self) -> usize {
+        self.layer
+    }
+    pub fn phase(&self) -> ConvInputPhase {
+        self.phase
+    }
+    pub fn input(&self) -> Result<&[f32]> {
+        self.input.as_slice()
+    }
+    pub fn prior_history(&self) -> Result<&[f32]> {
+        self.prior.as_slice()
+    }
+    pub fn raw(&self) -> Result<&[f32]> {
+        self.values.raw.as_slice()
+    }
+    pub fn activated(&self) -> Result<&[f32]> {
+        self.values.activated.as_slice()
+    }
+    pub fn next_history(&self) -> Result<&[f32]> {
+        self.values.next_history.as_slice()
+    }
+}
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    pub fn append_gdn_convolution_silu_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &PlanValue,
+        history: &PlanValue,
+    ) -> Result<CausalConvValues<PlanValue>> {
+        crate::forward::gdn_convolution_silu_slice(self, builder, layer, input, history)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_gdn_convolution_silu_slice(
+        &self,
+        layer: usize,
+        input: GdnConvSliceInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<GdnConvolutionSiluOutput> {
+        let (_, channels) = convolution_role(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let (activation, prior) = math.import_gdn_conv_input(input, channels)?;
+        let values = crate::forward::gdn_convolution_silu_slice(
+            self,
+            &mut math,
+            layer,
+            &activation,
+            &prior,
+        )?;
+        if cancelled.is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State("window convolution slice cancelled".into()));
+        }
+        Ok(GdnConvolutionSiluOutput {
+            source: self.source_identity(),
+            layer,
+            phase: input.phase(),
+            input: activation,
+            prior,
+            values,
+        })
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/gdn_conv_window_tests.rs
@@ -0,0 +1,382 @@
+//! Synthetic role/wiring/lease checks; genuine trained qualification is separate.
+use super::*;
+use crate::accelerator::program::{CausalConvSiluContract, ProgramNumericalProfile};
+use crate::operator_slice::gdn_conv::{ConvInputPhase, GdnConvSliceInput};
+use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
+
+fn conv_source(fixture: &Fixture) -> Loaded {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    cfg["text_config"]["hidden_size"] = json!(256);
+    cfg["text_config"]["linear_key_head_dim"] = json!(8);
+    cfg["text_config"]["linear_value_head_dim"] = json!(8);
+    source_with_config(
+        fixture,
+        serde_json::to_vec(&cfg).unwrap(),
+        true,
+        |name, index, fallback| {
+            if name.ends_with(".conv1d.weight") {
+                if index % 4 == 3 { 0x3f80 } else { 0 }
+            } else {
+                fallback
+            }
+        },
+        |_| {},
+    )
+}
+fn bf16(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
+    values
+        .into_iter()
+        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
+        .collect()
+}
+fn private<'a>(
+    source: crate::accelerator::SourceIdentity,
+    phase: ConvInputPhase,
+    x: &'a [u8],
+    h: &'a [u8],
+) -> GdnConvSliceInput<'a> {
+    GdnConvSliceInput::from_bf16_bytes(
+        source,
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        phase,
+        x,
+        h,
+    )
+    .unwrap()
+}
+#[test]
+fn pure_window_region_matches_closed_helpers_and_preserves_canonical_history_bits() {
+    let f = Fixture::new();
+    let loaded = conv_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let x = bf16((0..32).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }));
+    let h = bf16((0..32).flat_map(|_| [1.0, -0.0, 2.0]));
+    let output = ops
+        .execute_gdn_convolution_silu_slice(
+            0,
+            private(source.source_identity(), ConvInputPhase::CachedTail, &x, &h),
+            None,
+        )
+        .unwrap();
+    assert_eq!(output.layer(), 0);
+    assert_eq!(output.phase(), ConvInputPhase::CachedTail);
+    assert_eq!(output.source_identity(), source.source_identity());
+    // Last tap 1 and three zero weights make each positive input exact.
+    same(output.input().unwrap(), output.raw().unwrap());
+    let silu = zerogravity_cpu::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16(
+        source.source_identity(),
+        output.raw().unwrap(),
+    )
+    .unwrap();
+    assert_eq!(silu.len(), output.activated().unwrap().len());
+    for (expected, actual) in silu.iter().zip(output.activated().unwrap()) {
+        let bits = expected.to_bits();
+        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
+        assert_eq!(actual.to_bits(), rounded);
+    }
+    for (channel, next) in output.next_history().unwrap().chunks_exact(3).enumerate() {
+        assert_eq!(next[0].to_bits(), (-0.0f32).to_bits());
+        assert_eq!(next[1], 2.0);
+        assert_eq!(
+            next[2].to_bits(),
+            output.input().unwrap()[channel].to_bits()
+        );
+    }
+    let snapshot = admission.arbiter.snapshot(&admission.resource).unwrap();
+    assert_eq!(reserved(&admission), baseline + 2 * 960 * 1024 + 32 * 36);
+    assert_eq!(
+        snapshot
+            .per_class
+            .get(&AllocationClass::Output)
+            .copied()
+            .unwrap_or(0),
+        32 * 36
+    );
+    assert_eq!(
+        snapshot
+            .per_class
+            .get(&AllocationClass::TransferStaging)
+            .copied()
+            .unwrap_or(0),
+        0
+    );
+    let (model, _, _) = bind(loaded);
+    assert_eq!(model.initialized_cpu_state_bytes(), 0);
+    assert!(model.new_session().is_err());
+    assert!(model.forward_plan(8, true, false).is_err());
+    drop(ops);
+    drop(source);
+    assert_eq!(output.raw().unwrap().len(), 32);
+    assert!(reserved(&admission) > 0);
+    drop(output);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn initial_padding_is_explicit_and_never_canonicalizes_signed_zero() {
+    let id = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
+    let x = bf16([1.0; 32]);
+    let zero = bf16([0.0; 96]);
+    let negative = bf16([-0.0; 96]);
+    private(id, ConvInputPhase::InitialPadding, &x, &zero);
+    assert!(
+        GdnConvSliceInput::from_bf16_bytes(
+            id,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            ConvInputPhase::InitialPadding,
+            &x,
+            &negative
+        )
+        .is_err()
+    );
+    private(id, ConvInputPhase::CachedTail, &x, &negative);
+    for bytes in [vec![0; 63], vec![0xff; 64]] {
+        assert!(
+            GdnConvSliceInput::from_bf16_bytes(
+                id,
+                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+                ConvInputPhase::CachedTail,
+                &bytes,
+                &zero
+            )
+            .is_err()
+        );
+    }
+}
+#[test]
+fn aggregate_pressure_cancel_and_live_output_never_leak_or_reuse_admission() {
+    let f = Fixture::new();
+    let loaded = conv_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let x = bf16([1.0; 32]);
+    let h = bf16([0.0; 96]);
+    let input = private(
+        source.source_identity(),
+        ConvInputPhase::InitialPadding,
+        &x,
+        &h,
+    );
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        ops.execute_gdn_convolution_silu_slice(0, input, Some(&cancelled))
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), baseline);
+    let pressure = admission
+        .arbiter
+        .admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: 128 * 1024 * 1024 - baseline,
+            }],
+        )
+        .unwrap();
+    assert!(
+        ops.execute_gdn_convolution_silu_slice(0, input, None)
+            .is_err()
+    );
+    drop(pressure);
+    assert_eq!(reserved(&admission), baseline);
+    // Leave enough for BOTH input Vecs, but one byte short of the aggregate
+    // three-result envelope. A failed second admission must retire the inputs.
+    let input_reservation = 960 * 1024 + 32 * 16;
+    let result_reservation = 960 * 1024 + 32 * 20;
+    let held = admission
+        .arbiter
+        .admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: 128 * 1024 * 1024 - baseline - input_reservation - result_reservation + 1,
+            }],
+        )
+        .unwrap();
+    let pressured = reserved(&admission);
+    assert!(
+        ops.execute_gdn_convolution_silu_slice(0, input, None)
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), pressured);
+    drop(held);
+    assert_eq!(reserved(&admission), baseline);
+    // Cancellation between actual operand import and the region, without clocks.
+    cancelled.store(false, Ordering::Release);
+    let mut math =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, Some(&cancelled), None);
+    let (activation, history) = math.import_gdn_conv_input(input, 32).unwrap();
+    let after_import = reserved(&admission);
+    assert_eq!(after_import, baseline + 960 * 1024 + 32 * 16);
+    let parameter = ops
+        .parameter_descriptor(QwenParameterRole::GdnConvolution(0))
+        .unwrap();
+    cancelled.store(true, Ordering::Release);
+    assert!(
+        math.causal_conv_silu_profiled(
+            parameter.clone(),
+            &activation,
+            &history,
+            CausalConvSiluContract::TorchCpuBf16Avx2
+        )
+        .is_err()
+    );
+    assert_eq!(reserved(&admission), after_import);
+    cancelled.store(false, Ordering::Release);
+    let values = math
+        .causal_conv_silu_profiled(
+            parameter,
+            &activation,
+            &history,
+            CausalConvSiluContract::TorchCpuBf16Avx2,
+        )
+        .unwrap();
+    let clone = values.activated.clone();
+    assert!(clone.clone().into_vec().is_err());
+    drop(values);
+    assert_eq!(clone.as_slice().unwrap().len(), 32);
+    drop(clone);
+    assert_eq!(reserved(&admission), after_import);
+    drop(math);
+    drop(activation);
+    drop(history);
+    assert_eq!(reserved(&admission), baseline);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn same_source_strided_values_foreign_issuer_and_unaware_provider_refuse() {
+    struct Unaware;
+    impl crate::accelerator::MatVecKernel for Unaware {
+        fn try_execute(
+            &self,
+            _: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            panic!("typed convolution must refuse before any provider callback")
+        }
+    }
+    let f = Fixture::new();
+    let loaded = conv_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let x = bf16([1.0; 32]);
+    let h = bf16([0.0; 96]);
+    let input = private(
+        source.source_identity(),
+        ConvInputPhase::InitialPadding,
+        &x,
+        &h,
+    );
+    let provider = Unaware;
+    let unsupported =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, None, Some(&provider));
+    assert!(unsupported.import_gdn_conv_input(input, 32).is_err());
+    assert_eq!(reserved(&admission), baseline);
+    let mut math = crate::host_math::HostMath::for_window_operator_slice(&ops, None, None);
+    let (activation, history) = math.import_gdn_conv_input(input, 32).unwrap();
+    let strided = math
+        .view(
+            &history,
+            TensorView {
+                offset: 0,
+                rows: 32,
+                width: 1,
+                row_stride: 2,
+            },
+        )
+        .unwrap();
+    let parameter = ops
+        .parameter_descriptor(QwenParameterRole::GdnConvolution(0))
+        .unwrap();
+    let before = reserved(&admission);
+    let error = math
+        .causal_conv_silu_profiled(
+            parameter.clone(),
+            &strided,
+            &history,
+            CausalConvSiluContract::TorchCpuBf16Avx2,
+        )
+        .err()
+        .unwrap();
+    assert!(error.to_string().contains("contiguous"));
+    assert_eq!(reserved(&admission), before);
+    let foreign_source = windows(&f, &loaded).0;
+    let foreign_ops = bind_windows(foreign_source, &loaded);
+    let mut foreign =
+        crate::host_math::HostMath::for_window_operator_slice(&foreign_ops, None, None);
+    assert!(
+        foreign
+            .causal_conv_silu_profiled(
+                parameter,
+                &activation,
+                &history,
+                CausalConvSiluContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+    );
+    drop(foreign);
+    drop(foreign_ops);
+    drop(math);
+    drop(strided);
+    drop(activation);
+    drop(history);
+    assert_eq!(reserved(&admission), baseline);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+
+#[test]
+fn actual_role_plan_append_uses_same_builder_and_never_adds_state() {
+    use crate::accelerator::program::{ActivationPrecision, PlanBuilder, PlanOperation};
+    let fixture = Fixture::new();
+    let loaded = conv_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let mut builder = PlanBuilder::new_operator_slice(
+        source.source_identity(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+    )
+    .unwrap();
+    let embedding = builder
+        .embed_profiled(
+            operators
+                .matrix_descriptor(QwenMatrixRole::Embedding)
+                .unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    // Synthetic embedding-produced operands solely exercise the actual source
+    // roles/same-builder path; they are not claimed as trained QKV/cache inputs.
+    let x = builder.view(&embedding, TensorView::vector(0, 32)).unwrap();
+    let h = builder
+        .view(&embedding, TensorView::vector(32, 96))
+        .unwrap();
+    let result = operators
+        .append_gdn_convolution_silu_slice(&mut builder, 0, &x, &h)
+        .unwrap();
+    assert_eq!(
+        builder.precision(&result.activated),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    let recorded = builder.finish_operator_slice(result.activated).unwrap();
+    assert_eq!(recorded.plan().nodes().len(), 4);
+    assert!(matches!(
+        recorded.plan().nodes()[3].operation,
+        PlanOperation::TypedCausalConvSilu { .. }
+    ));
+    assert!(recorded.plan().required_domains().is_empty());
+    assert!(recorded.plan().proposed_domains().is_empty());
+    assert_eq!(recorded.plan().frontier_delta(), 0);
+    drop(recorded);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/host_math/gdn_conv.rs
@@ -0,0 +1,152 @@
+//! Aggregate admitted private inputs/results for the pure original region.
+use super::*;
+use crate::accelerator::program::{CausalConvSiluContract, CausalConvValues};
+use crate::operator_slice::gdn_conv::GdnConvSliceInput;
+
+const SCRATCH: usize = 960 * 1024;
+impl HostMath<'_, '_, '_> {
+    fn conv_windows(&self) -> Result<&crate::original_binding::OriginalWindowOperators> {
+        if self.region_kernel.is_some()
+            || self.engine.is_some()
+            || self.state.is_some()
+            || self.source.program_numerical_profile()
+                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported(
+                "convolution requires private pinned CPU window contract".into(),
+            ));
+        }
+        match &self.source {
+            HostSource::Windows(source) => Ok(source),
+            HostSource::Loaded(_) => Err(Error::Unsupported(
+                "convolution requires admitted window backing".into(),
+            )),
+        }
+    }
+    fn check_conv_cancel(&self) -> Result<()> {
+        if self
+            .cancelled
+            .is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire))
+        {
+            return Err(Error::State("window convolution cancelled".into()));
+        }
+        Ok(())
+    }
+    pub(crate) fn import_gdn_conv_input(
+        &self,
+        input: GdnConvSliceInput<'_>,
+        channels: usize,
+    ) -> Result<(HostValue, HostValue)> {
+        let source = self.conv_windows()?;
+        if input.validate()? != channels
+            || input.source_identity() != source.source_identity()
+            || input.numerical_profile() != self.source.program_numerical_profile()
+        {
+            return Err(Error::State(
+                "convolution input full width/source/profile".into(),
+            ));
+        }
+        self.check_conv_cancel()?;
+        // Both allocations are charged once, before either Vec allocation.
+        let workspace = Arc::new(
+            source.source().reserve_operator_buffers(
+                channels
+                    .checked_mul(16)
+                    .ok_or_else(|| Error::Limit("convolution input byte count".into()))?,
+                SCRATCH,
+            )?,
+        );
+        let make = |bytes: &[u8]| -> Result<HostValue> {
+            let values = bytes
+                .chunks_exact(2)
+                .map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16))
+                .collect();
+            HostValue::typed(
+                values,
+                ActivationPrecision::Bf16WidenedF32,
+                source.source_identity(),
+                self.source.program_numerical_profile(),
+            )?
+            .retain_workspace(Some(workspace.clone()))
+        };
+        let result = (
+            make(input.activation_bytes())?,
+            make(input.history_bytes())?,
+        );
+        self.check_conv_cancel()?;
+        Ok(result)
+    }
+    pub(super) fn execute_causal_conv_silu(
+        &self,
+        parameter: SourceParameter,
+        input: &HostValue,
+        history: &HostValue,
+        contract: CausalConvSiluContract,
+    ) -> Result<CausalConvValues<HostValue>> {
+        let source = self.conv_windows()?;
+        let channels = contract.validate(&parameter, input.elements, history.elements)?;
+        if parameter.source != source.source_identity() {
+            return Err(Error::State("convolution parameter source".into()));
+        }
+        for value in [input, history] {
+            if value.source != Some(source.source_identity())
+                || value.profile != Some(self.source.program_numerical_profile())
+                || value.precision != ActivationPrecision::Bf16WidenedF32
+            {
+                return Err(Error::State(
+                    "convolution activation source/profile/precision".into(),
+                ));
+            }
+            let workspace = value.data.workspace.as_ref().ok_or_else(|| {
+                Error::State("convolution activation missing physical backing".into())
+            })?;
+            if !source.source().same_issuer(workspace.source()) {
+                return Err(Error::State(
+                    "convolution activation foreign source issuer".into(),
+                ));
+            }
+        }
+        // Contiguity and source membership checked before allocation or math.
+        let activation = input.as_slice()?;
+        let prior = history.as_slice()?;
+        ActivationPrecision::Bf16WidenedF32.validate_values(activation)?;
+        ActivationPrecision::Bf16WidenedF32.validate_values(prior)?;
+        self.check_conv_cancel()?;
+        // Three retained results: C raw + C activated + 3C history, F32 backing.
+        // Scratch conservatively covers raw FP32 + SiLU FP32 + rounding Vec.
+        let workspace = Arc::new(
+            source.source().reserve_operator_buffers(
+                channels
+                    .checked_mul(20)
+                    .ok_or_else(|| Error::Limit("convolution output byte count".into()))?,
+                SCRATCH,
+            )?,
+        );
+        let completed = source.convolve(&parameter, activation, prior, self.cancelled)?;
+        let silu =
+            zerogravity_cpu::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16(
+                source.source_identity(),
+                &completed.raw_bf16,
+            )?;
+        let activated = silu
+            .into_iter()
+            .map(round_bf16)
+            .collect::<Result<Vec<_>>>()?;
+        let make = |values| -> Result<HostValue> {
+            HostValue::typed(
+                values,
+                ActivationPrecision::Bf16WidenedF32,
+                source.source_identity(),
+                self.source.program_numerical_profile(),
+            )?
+            .retain_workspace(Some(workspace.clone()))
+        };
+        let result = CausalConvValues {
+            raw: make(completed.raw_bf16)?,
+            activated: make(activated)?,
+            next_history: make(completed.next_history)?,
+        };
+        self.check_conv_cancel()?;
+        Ok(result)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -507,6 +507,25 @@
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         let output = output.retain_workspace(workspace)?;
         Ok(output)
+    }
+    fn causal_conv_silu_profiled(
+        &mut self,
+        parameter: SourceParameter,
+        input: &HostValue,
+        history: &HostValue,
+        contract: crate::accelerator::program::CausalConvSiluContract,
+    ) -> Result<crate::accelerator::program::CausalConvValues<HostValue>> {
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        {
+            self.execute_causal_conv_silu(parameter, input, history, contract)
+        }
+        #[cfg(not(all(feature = "original-hf-binding", target_os = "linux")))]
+        {
+            let _ = (parameter, input, history, contract);
+            Err(Error::Unsupported(
+                "causal convolution requires leased original windows".into(),
+            ))
+        }
     }
     fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<HostValue> {
         self.legacy_precision(&[])?;
@@ -1031,3 +1050,6 @@
         }
     }
 }
+
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+mod gdn_conv;
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -231,3 +231,6 @@
 /// Pure attention projections/QK norms; no rotary/cache/session admission.
 #[path = "attention_slice.rs"]
 pub mod attention;
+
+#[path = "gdn_conv_slice.rs"]
+pub mod gdn_conv;
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -379,3 +379,6 @@
 
 #[path = "attention_slice_tests.rs"]
 mod attention;
+
+#[path = "gdn_conv_window_tests.rs"]
+mod gdn_conv;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window/gdn_conv.rs
@@ -0,0 +1,59 @@
+//! Original full rank3 stage; arithmetic remains in accepted CPU helpers.
+use super::*;
+impl OriginalWindowOperators {
+    pub(crate) fn convolve(
+        &self,
+        parameter: &SourceParameter,
+        input: &[f32],
+        history: &[f32],
+        cancelled: Option<&AtomicBool>,
+    ) -> Result<zerogravity_cpu::original_hf::conv::CausalConv4Completed> {
+        let mut found = false;
+        for (layer, kind) in self.geometry().layers.iter().enumerate() {
+            if *kind != crate::model::LayerKind::LinearAttention {
+                continue;
+            }
+            let (actual, channels) =
+                crate::operator_slice::gdn_conv::convolution_role(self, layer)?;
+            if actual == *parameter && input.len() == channels && history.len() == channels * 3 {
+                found = true;
+                break;
+            }
+        }
+        if !found {
+            return Err(Error::State(
+                "window convolution original role/geometry".into(),
+            ));
+        }
+        check_cancel(cancelled)?;
+        let raw = self.source().conv4_parameter(&parameter.name)?;
+        raw.validate_source_parameter(parameter)?;
+        let window = match cancelled {
+            Some(flag) => {
+                self.source()
+                    .read_parameter_cancellable(&raw, 0..raw.elements(), flag)?
+            }
+            None => self.source().read_parameter(&raw, 0..raw.elements())?,
+        };
+        if !window.is_complete() || window.element_range() != (0..raw.elements()) {
+            return Err(Error::State(
+                "window convolution incomplete raw parameter".into(),
+            ));
+        }
+        let output = zerogravity_cpu::original_hf::conv::causal_step_bf16_4(
+            zerogravity_cpu::original_hf::conv::CausalConv4Inputs {
+                source: self.source_identity(),
+                tensor_name: &parameter.name,
+                weight: zerogravity_core::source::SourceTensorRef {
+                    kind: raw.tensor().kind,
+                    shape: &raw.tensor().shape,
+                    packed_bytes: window.as_slice(),
+                },
+                activation: input,
+                history,
+            },
+        )?;
+        check_cancel(cancelled)?;
+        Ok(output)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/original_binding/window.rs
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window.rs
@@ -415,3 +415,6 @@
         self.projected.as_slice()
     }
 }
+
+#[cfg(feature = "cpu")]
+mod gdn_conv;

````

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v5-20261020/BOUNDED_ORACLE_IO_V2.patch

Original UTF-8 bytes: 3395; SHA256 `edd6d71d802a44e24aa7590c1f0ac2a4ffb3f54f4e9ed70078563137f6c2ba72`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/reference.py
+++ b/reference.py
@@ -7,6 +7,7 @@
 import os
 from pathlib import Path
 import struct
+import stat
 
 
 def require(ok, message):
@@ -18,12 +19,23 @@
     return hashlib.sha256(raw).hexdigest()
 
 
+def small(path, cap):
+    with os.fdopen(os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW),'rb') as opened:
+        before=os.fstat(opened.fileno())
+        require(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'bounded regular operator metadata/operand')
+        data=opened.read(cap+1)
+        after=os.fstat(opened.fileno())
+    identity=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
+    require(len(data)<=cap and identity(before)==identity(after),'bounded immutable selected input FD')
+    return data
+
+
 def checked(root, record):
     name=record['file']
     require(type(name) is str and Path(name).name == name, 'contained selected operand')
     path=root/name
     require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 128*1024, 'bounded raw operand')
-    raw=path.read_bytes()
+    raw=small(path,128*1024)
     require(len(raw) == record['bytes'] and sha(raw) == record['sha256'], 'exact selected operand pin')
     return raw
 
@@ -59,8 +71,8 @@
     require(torch.backends.cpu.get_cpu_capability() == 'AVX2' and not torch.cuda.is_available()
             and torch.backends.mkldnn.enabled, 'original CPU-only AVX2/mkldnn policy')
     source=Path('/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py')
-    require(sha(source.read_bytes()) == '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b', 'unchanged official convolution caller')
-    document=args.inputs.read_bytes()
+    require(sha(small(source,1024*1024)) == '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b', 'unchanged official convolution caller')
+    document=small(args.inputs,128*1024)
     require(len(document)<=128*1024 and sha(document)==args.inputs_sha256,'exact prepared inputs metadata')
     inputs=json.loads(document)
     require(inputs['schema']=='zerogravity.original-conv-fragment.inputs.v1' and inputs['profile']=='source-36-f32-restored','genuine input protocol')
--- a/run_replay.py
+++ b/run_replay.py
@@ -36,12 +36,12 @@
 
 
 def frozen():
-    manifest = read_json(HERE / "freeze-manifest.json")
+    manifest = read_json(HERE / "freeze-manifest-v2.json")
     for item in manifest["files"]:
         pin(HERE / item["file"], item)
     for item in manifest["main_sources"]:
         pin(item["path"], item)
-    return sha(HERE / "freeze-manifest.json")
+    return sha(HERE / "freeze-manifest-v2.json")
 
 
 def execute(binary, binary_fd, inputs, inputs_sha, reference, reference_sha, output, model, cpus, freeze, source_before):
--- a/ROOT-QUALIFICATION.md
+++ b/ROOT-QUALIFICATION.md
@@ -1,6 +1,7 @@
 # ROOT-only genuine qualification
 
-Status: PREPARED / UNQUALIFIED. Author parsed/formatted source only. No raw
+Status: PREPARED / UNQUALIFIED. Latest additive freeze: `freeze-manifest-v2.json`;
+initial `freeze-manifest.json`/`frozen-prepared/` retained unchanged. Author parsed/formatted source only. No raw
 model/tensor payload was hashed or operator/reference/harness executed.
 `bindings.json` copies expected source/capture hashes from genuine existing
 receipts; author independently read/pinned only small source/metadata.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v5-20261020/GPU_REFUSAL.patch

Original UTF-8 bytes: 1645; SHA256 `f09d057c3c65d75907f85ade140c45c05465d20b9b831a254b86475b8299e419`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -838,7 +838,8 @@
             match &node.operation {
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
-                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedOrdinaryRms { .. }
+                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                     "resident batch has no qualified typed lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -348,6 +348,7 @@
                     PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
+                        | PlanOperation::TypedCausalConvSilu { .. }
                 )
             })
         {
@@ -1429,7 +1430,8 @@
             match &node.operation {
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
-                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedOrdinaryRms { .. }
+                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                     "resident typed operation has no qualified lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {

````

