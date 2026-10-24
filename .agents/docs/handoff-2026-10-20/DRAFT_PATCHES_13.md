# Draft Patches — volume 13

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-original-attention-slice-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 40847; SHA256 `cf5187a2f46d0cfb54707a0cbb6f5ef9940f10378441922aca1977754bfe985d`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/attention_slice.rs
@@ -0,0 +1,327 @@
+//! Closed projections/QK norms over an explicit private input, never a layer.
+use super::OperatorSliceSource;
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{
+    ActivationPrecision, MatrixAxes, OrdinaryRmsContract, PlanBuilder, PlanValue,
+    ProgramNumericalProfile, SourceParameter, SourceTensor,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, LoadedQwenModel, Result};
+use zerogravity_core::quant::QuantKind;
+
+pub(crate) struct AttentionRoles {
+    pub(crate) hidden: usize,
+    pub(crate) query_heads: usize,
+    pub(crate) kv_heads: usize,
+    pub(crate) width: usize,
+    pub(crate) query: SourceTensor,
+    pub(crate) key: SourceTensor,
+    pub(crate) value: SourceTensor,
+    pub(crate) query_norm: SourceParameter,
+    pub(crate) key_norm: SourceParameter,
+}
+
+/// Check every role before recording or importing a private activation.
+pub(crate) fn attention_roles(
+    source: &impl OperatorSliceSource,
+    layer: usize,
+) -> Result<AttentionRoles> {
+    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || source.rms_epsilon().to_bits() != 1e-6f32.to_bits()
+        || !source.source_identity().is_bound()
+    {
+        return Err(Error::Unsupported(
+            "closed attention slice profile/source/epsilon".into(),
+        ));
+    }
+    let geometry = source.geometry();
+    if geometry.layers.get(layer) != Some(&crate::model::LayerKind::FullAttention) {
+        return Err(Error::Unsupported(
+            "attention slice requires actual FullAttention layer".into(),
+        ));
+    }
+    let size = |value| {
+        usize::try_from(value).map_err(|_| Error::Limit("original attention geometry".into()))
+    };
+    let hidden = size(geometry.hidden_size)?;
+    let query_heads = size(geometry.num_attention_heads)?;
+    let kv_heads = size(geometry.num_key_value_heads)?;
+    let width = size(geometry.head_dim)?;
+    if !matches!(hidden, 256 | 1024 | 2048 | 5120)
+        || width != 256
+        || query_heads == 0
+        || kv_heads == 0
+        || !query_heads.is_multiple_of(kv_heads)
+    {
+        return Err(Error::Unsupported(
+            "closed attention slice head/hidden geometry".into(),
+        ));
+    }
+    let query_rows = query_heads
+        .checked_mul(width)
+        .and_then(|n| n.checked_mul(2))
+        .filter(|n| *n <= 16384)
+        .ok_or_else(|| Error::Limit("attention query/gate rows".into()))?;
+    let kv_rows = kv_heads
+        .checked_mul(width)
+        .filter(|n| *n <= 16384)
+        .ok_or_else(|| Error::Limit("attention key/value rows".into()))?;
+    let query = source.matrix_descriptor(QwenMatrixRole::Query(layer))?;
+    let key = source.matrix_descriptor(QwenMatrixRole::Key(layer))?;
+    let value = source.matrix_descriptor(QwenMatrixRole::Value(layer))?;
+    let namespace = format!("model.language_model.layers.{layer}.self_attn");
+    for (matrix, rows, suffix) in [
+        (&query, query_rows, "q_proj"),
+        (&key, kv_rows, "k_proj"),
+        (&value, kv_rows, "v_proj"),
+    ] {
+        if matrix.name != format!("{namespace}.{suffix}.weight")
+            || matrix.source != source.source_identity()
+            || matrix.kind != QuantKind::Bf16
+            || matrix.axes != MatrixAxes::OutputInput
+            || matrix.columns != hidden
+            || matrix.rows != rows
+        {
+            return Err(Error::State(
+                "attention original full matrix role/axes/shape".into(),
+            ));
+        }
+        matrix.packed_bytes()?;
+    }
+    let query_norm = source.parameter_descriptor(QwenParameterRole::QueryNorm(layer))?;
+    let key_norm = source.parameter_descriptor(QwenParameterRole::KeyNorm(layer))?;
+    for (parameter, elements, suffix) in [
+        (&query_norm, query_rows / 2, "q_norm"),
+        (&key_norm, kv_rows, "k_norm"),
+    ] {
+        if parameter.name != format!("{namespace}.{suffix}.weight")
+            || parameter.source != source.source_identity()
+            || OrdinaryRmsContract::TorchCpuBf16Avx2.validate(parameter, elements)? != width
+        {
+            return Err(Error::State(
+                "attention original Q/K norm role/width".into(),
+            ));
+        }
+    }
+    Ok(AttentionRoles {
+        hidden,
+        query_heads,
+        kv_heads,
+        width,
+        query,
+        key,
+        value,
+        query_norm,
+        key_norm,
+    })
+}
+
+/// Existing typed values only. The input's producer remains the caller's.
+/// No state domains, rotary, cache effects, token schedule or logits are added.
+pub struct AttentionProjectionValues<V> {
+    pub(crate) query_gate_projection: V,
+    pub(crate) key_projection: V,
+    pub(crate) value_projection: V,
+    pub(crate) query: V,
+    pub(crate) gate: V,
+    pub(crate) normalized_query: V,
+    pub(crate) normalized_key: V,
+}
+impl<V> AttentionProjectionValues<V> {
+    pub fn query_gate_projection(&self) -> &V {
+        &self.query_gate_projection
+    }
+    pub fn key_projection(&self) -> &V {
+        &self.key_projection
+    }
+    pub fn value_projection(&self) -> &V {
+        &self.value_projection
+    }
+    pub fn query(&self) -> &V {
+        &self.query
+    }
+    pub fn gate(&self) -> &V {
+        &self.gate
+    }
+    pub fn normalized_query(&self) -> &V {
+        &self.normalized_query
+    }
+    pub fn normalized_key(&self) -> &V {
+        &self.normalized_key
+    }
+}
+impl LoadedQwenModel {
+    /// Append to the caller's same-builder, same-source typed input.
+    /// An embedding fixture is transport evidence, not a trained layer input.
+    pub fn append_attention_projection_norm_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &PlanValue,
+    ) -> Result<AttentionProjectionValues<PlanValue>> {
+        self.require_operator_slice()?;
+        crate::forward::attention_projection_norm_slice(self, builder, layer, input)
+    }
+}
+
+/// Qualifier-supplied raw BF16 bytes. The caller authenticates their external
+/// source/archive provenance; this contract checks only binding/representation.
+/// Import never asserts a model producer, StateRoot or token frontier.
+#[derive(Clone, Copy)]
+pub struct AttentionSliceInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    bytes: &'a [u8],
+}
+impl<'a> AttentionSliceInput<'a> {
+    pub fn from_bf16_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        bytes: &'a [u8],
+    ) -> Result<Self> {
+        let value = Self {
+            source,
+            profile,
+            bytes,
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
+    pub fn precision(self) -> ActivationPrecision {
+        ActivationPrecision::Bf16WidenedF32
+    }
+    pub fn bytes(self) -> &'a [u8] {
+        self.bytes
+    }
+    pub(crate) fn validate(self) -> Result<()> {
+        if !self.source.is_bound()
+            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::State(
+                "attention private input source/profile".into(),
+            ));
+        }
+        if self.bytes.is_empty()
+            || self.bytes.len() > 2 * 5120
+            || !self.bytes.len().is_multiple_of(2)
+            || self
+                .bytes
+                .chunks_exact(2)
+                .any(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 == 0x7f80)
+        {
+            return Err(Error::Malformed(
+                "attention private finite BF16 input bytes".into(),
+            ));
+        }
+        Ok(())
+    }
+}
+
+#[cfg(feature = "cpu")]
+pub struct AttentionProjectionNormOutput {
+    source: SourceIdentity,
+    layer: usize,
+    input: crate::host_math::HostValue,
+    values: AttentionProjectionValues<crate::host_math::HostValue>,
+}
+#[cfg(feature = "cpu")]
+impl AttentionProjectionNormOutput {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn layer(&self) -> usize {
+        self.layer
+    }
+    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    }
+    pub fn precision(&self) -> ActivationPrecision {
+        ActivationPrecision::Bf16WidenedF32
+    }
+    pub fn input(&self) -> Result<&[f32]> {
+        self.input.as_slice()
+    }
+    pub fn query_gate_projection(&self) -> Result<&[f32]> {
+        self.values.query_gate_projection.as_slice()
+    }
+    pub fn key_projection(&self) -> Result<&[f32]> {
+        self.values.key_projection.as_slice()
+    }
+    pub fn value_projection(&self) -> Result<&[f32]> {
+        self.values.value_projection.as_slice()
+    }
+    pub fn query(&self) -> Result<&[f32]> {
+        self.values.query.as_slice()
+    }
+    pub fn gate(&self) -> Result<&[f32]> {
+        self.values.gate.as_slice()
+    }
+    pub fn normalized_query(&self) -> Result<&[f32]> {
+        self.values.normalized_query.as_slice()
+    }
+    pub fn normalized_key(&self) -> Result<&[f32]> {
+        self.values.normalized_key.as_slice()
+    }
+}
+#[cfg(feature = "cpu")]
+impl LoadedQwenModel {
+    /// Pure numerical fragment on explicit observed operator input, no session.
+    pub fn execute_attention_projection_norm_slice(
+        &self,
+        layer: usize,
+        input: AttentionSliceInput<'_>,
+    ) -> Result<AttentionProjectionNormOutput> {
+        self.require_operator_slice()?;
+        let roles = attention_roles(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_operator_slice(self, None);
+        let input = math.import_attention_input(input, roles.hidden)?;
+        let values =
+            crate::forward::attention_projection_norm_slice(self, &mut math, layer, &input)?;
+        Ok(AttentionProjectionNormOutput {
+            source: self.source_identity(),
+            layer,
+            input,
+            values,
+        })
+    }
+}
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    /// No external-input node is invented; caller retains the source/metadata.
+    pub fn append_attention_projection_norm_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &PlanValue,
+    ) -> Result<AttentionProjectionValues<PlanValue>> {
+        crate::forward::attention_projection_norm_slice(self, builder, layer, input)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_attention_projection_norm_slice(
+        &self,
+        layer: usize,
+        input: AttentionSliceInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<AttentionProjectionNormOutput> {
+        let roles = attention_roles(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let input = math.import_attention_input(input, roles.hidden)?;
+        let values =
+            crate::forward::attention_projection_norm_slice(self, &mut math, layer, &input)?;
+        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State("window attention slice cancelled".into()));
+        }
+        Ok(AttentionProjectionNormOutput {
+            source: self.source_identity(),
+            layer,
+            input,
+            values,
+        })
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/attention_slice_tests.rs
@@ -0,0 +1,499 @@
+//! Synthetic exact-role transport/ownership only; no trained hidden-state claim.
+use super::*;
+use crate::accelerator::program::{
+    ActivationPrecision, PlanBuilder, ProgramNumericalProfile, ProjectionContract,
+};
+use crate::operator_slice::attention::AttentionSliceInput;
+
+fn attention_source(fixture: &Fixture) -> Loaded {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    cfg["text_config"]["hidden_size"] = json!(256);
+    cfg["text_config"]["head_dim"] = json!(256);
+    // Deliberately same width as Q/K norms: role checks must reject GDN gamma.
+    cfg["text_config"]["linear_value_head_dim"] = json!(256);
+    source_with_config(
+        fixture,
+        serde_json::to_vec(&cfg).unwrap(),
+        true,
+        |name, element, fallback| {
+            let value = if name.ends_with(".self_attn.q_proj.weight") {
+                [1.0f32, -2.0, 3.0, -4.0][element / (256 * 256)]
+            } else if name.ends_with(".self_attn.k_proj.weight") {
+                5.0
+            } else if name.ends_with(".self_attn.v_proj.weight") {
+                6.0
+            } else {
+                return fallback;
+            };
+            if element.is_multiple_of(256) {
+                (value.to_bits() >> 16) as u16
+            } else {
+                0
+            }
+        },
+        |_| {},
+    )
+}
+fn raw_input() -> Vec<u8> {
+    let mut bytes = vec![0; 512];
+    bytes[..2].copy_from_slice(&0x3f80u16.to_le_bytes());
+    bytes
+}
+fn input(source: crate::accelerator::SourceIdentity, bytes: &[u8]) -> AttentionSliceInput<'_> {
+    AttentionSliceInput::from_bf16_bytes(
+        source,
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        bytes,
+    )
+    .unwrap()
+}
+
+#[test]
+fn explicit_attention_input_uses_exact_rows_raw_norms_and_no_session() {
+    let fixture = Fixture::new();
+    let loaded = attention_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let (model, _, _) = bind(loaded);
+    let bytes = raw_input();
+    let expected = model
+        .execute_attention_projection_norm_slice(3, input(model.source_identity(), &bytes))
+        .unwrap();
+    let before = reserved(&admission);
+    let output = operators
+        .execute_attention_projection_norm_slice(3, input(source.source_identity(), &bytes), None)
+        .unwrap();
+    assert_eq!(output.source_identity(), source.source_identity());
+    assert_eq!(output.layer(), 3);
+    assert_eq!(output.precision(), ActivationPrecision::Bf16WidenedF32);
+    assert_eq!(output.numerical_profile(), expected.numerical_profile());
+    same(expected.input().unwrap(), output.input().unwrap());
+    same(
+        expected.query_gate_projection().unwrap(),
+        output.query_gate_projection().unwrap(),
+    );
+    same(
+        expected.key_projection().unwrap(),
+        output.key_projection().unwrap(),
+    );
+    same(
+        expected.value_projection().unwrap(),
+        output.value_projection().unwrap(),
+    );
+    same(expected.query().unwrap(), output.query().unwrap());
+    same(expected.gate().unwrap(), output.gate().unwrap());
+    same(
+        expected.normalized_query().unwrap(),
+        output.normalized_query().unwrap(),
+    );
+    same(
+        expected.normalized_key().unwrap(),
+        output.normalized_key().unwrap(),
+    );
+    // Independent integer row labels distinguish [head,q/gate,width] packing.
+    assert_eq!(
+        output.query_gate_projection().unwrap(),
+        [
+            vec![1.0; 256],
+            vec![-2.0; 256],
+            vec![3.0; 256],
+            vec![-4.0; 256]
+        ]
+        .concat()
+    );
+    assert_eq!(
+        output.query().unwrap(),
+        [vec![1.0; 256], vec![3.0; 256]].concat()
+    );
+    assert_eq!(
+        output.gate().unwrap(),
+        [vec![-2.0; 256], vec![-4.0; 256]].concat()
+    );
+    assert_eq!(output.key_projection().unwrap(), &[5.0; 256]);
+    assert_eq!(output.value_projection().unwrap(), &[6.0; 256]);
+    assert_eq!(reserved(&admission), before + 8 * 1024 * 1024);
+    assert_eq!(model.initialized_cpu_state_bytes(), 0);
+    assert!(model.new_session().is_err());
+    assert!(model.forward_plan(8, true, false).is_err());
+    let query_norm = model
+        .parameter_descriptor(QwenParameterRole::QueryNorm(3))
+        .unwrap();
+    assert_eq!(query_norm.kind, zerogravity_core::quant::QuantKind::F32);
+    assert!(
+        model
+            .tensor_bytes(&query_norm.name)
+            .unwrap()
+            .chunks_exact(4)
+            .all(|word| word == 0.25f32.to_le_bytes())
+    );
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 40 * 1024 * 1024);
+    same(
+        expected.normalized_query().unwrap(),
+        output.normalized_query().unwrap(),
+    );
+    drop(output);
+    assert_eq!(reserved(&admission), 0);
+    assert!(
+        !admission
+            .arbiter
+            .snapshot(&admission.resource)
+            .unwrap()
+            .fenced
+    );
+}
+
+#[test]
+fn plan_append_has_no_input_invention_or_state_and_rejects_foreign_builder() {
+    let fixture = Fixture::new();
+    let (model, _, _) = bind(attention_source(&fixture));
+    let profile = model.program_numerical_profile();
+    let mut builder = PlanBuilder::new_operator_slice(model.source_identity(), profile).unwrap();
+    // Synthetic TypedEmbed is ONLY an existing same-builder transport fixture.
+    let embedding = builder
+        .embed_profiled(
+            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    assert!(
+        model
+            .append_attention_projection_norm_slice(&mut builder, 0, &embedding)
+            .is_err()
+    );
+    assert!(
+        model
+            .append_attention_projection_norm_slice(&mut builder, 4, &embedding)
+            .is_err()
+    );
+    let values = model
+        .append_attention_projection_norm_slice(&mut builder, 3, &embedding)
+        .unwrap();
+    let recorded = builder
+        .finish_operator_slice(*values.normalized_query())
+        .unwrap();
+    assert_eq!(recorded.plan().nodes().len(), 10);
+    assert_eq!(
+        recorded
+            .plan()
+            .nodes()
+            .iter()
+            .filter(|node| matches!(node.operation, PlanOperation::TypedProject { .. }))
+            .count(),
+        3
+    );
+    assert_eq!(
+        recorded
+            .plan()
+            .nodes()
+            .iter()
+            .filter(|node| matches!(node.operation, PlanOperation::TypedOrdinaryRms { .. }))
+            .count(),
+        2
+    );
+    let views = recorded
+        .plan()
+        .nodes()
+        .iter()
+        .filter_map(|node| match node.operation {
+            PlanOperation::View(view) => Some(view),
+            _ => None,
+        })
+        .collect::<Vec<_>>();
+    assert_eq!(views.len(), 2);
+    assert_eq!(
+        (views[0].offset, views[1].offset, views[0].row_stride),
+        (0, 256, 512)
+    );
+    assert_eq!(
+        values.query_gate_projection().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert_eq!(
+        values.key_projection().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert_eq!(
+        values.value_projection().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert_eq!(
+        values.query().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert_eq!(
+        values.gate().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert_eq!(
+        values.normalized_key().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert!(recorded.plan().required_domains().is_empty());
+    assert!(recorded.plan().proposed_domains().is_empty());
+    assert_eq!(recorded.plan().frontier_delta(), 0);
+    let mut first = PlanBuilder::new_operator_slice(model.source_identity(), profile).unwrap();
+    let foreign = first
+        .embed_profiled(
+            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    let mut second = PlanBuilder::new_operator_slice(model.source_identity(), profile).unwrap();
+    assert!(
+        model
+            .append_attention_projection_norm_slice(&mut second, 3, &foreign)
+            .is_err()
+    );
+    assert_eq!(
+        first
+            .finish_operator_slice(foreign)
+            .unwrap()
+            .plan()
+            .nodes()
+            .len(),
+        1
+    );
+}
+
+#[test]
+fn raw_bf16_import_preserves_zero_subnormal_bits_and_checks_before_admission() {
+    let fixture = Fixture::new();
+    let loaded = attention_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
+    let mut bytes = raw_input();
+    for (index, bits) in [0u16, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f]
+        .into_iter()
+        .enumerate()
+    {
+        bytes[2 * index..2 * index + 2].copy_from_slice(&bits.to_le_bytes());
+    }
+    let before = reserved(&admission);
+    let imported = math
+        .import_attention_input(input(source.source_identity(), &bytes), 256)
+        .unwrap();
+    for (word, value) in bytes.chunks_exact(2).zip(imported.as_slice().unwrap()) {
+        assert_eq!(
+            value.to_bits(),
+            u32::from(u16::from_le_bytes([word[0], word[1]])) << 16
+        );
+    }
+    assert_eq!(reserved(&admission), before + 1024 * 1024);
+    assert!(imported.clone().into_vec().is_err());
+    drop(imported);
+    assert_eq!(reserved(&admission), before);
+    for invalid in [
+        Vec::new(),
+        vec![0],
+        vec![0; 2 * 5120 + 2],
+        0x7f80u16.to_le_bytes().to_vec(),
+        0xff80u16.to_le_bytes().to_vec(),
+        0x7fc0u16.to_le_bytes().to_vec(),
+    ] {
+        assert!(
+            AttentionSliceInput::from_bf16_bytes(
+                source.source_identity(),
+                operators.program_numerical_profile(),
+                &invalid
+            )
+            .is_err()
+        );
+    }
+    assert!(
+        AttentionSliceInput::from_bf16_bytes(
+            crate::accelerator::SourceIdentity::unbound(),
+            operators.program_numerical_profile(),
+            &bytes
+        )
+        .is_err()
+    );
+    assert!(
+        AttentionSliceInput::from_bf16_bytes(
+            source.source_identity(),
+            ProgramNumericalProfile::GgufSourceV1,
+            &bytes
+        )
+        .is_err()
+    );
+    let foreign = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
+    assert!(
+        math.import_attention_input(input(foreign, &bytes), 256)
+            .is_err()
+    );
+    assert!(
+        math.import_attention_input(input(source.source_identity(), &bytes[..510]), 256)
+            .is_err()
+    );
+    assert!(
+        math.import_attention_input(input(source.source_identity(), &bytes), 128)
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before);
+}
+
+#[test]
+fn exact_role_provider_and_foreign_issuer_refusals_have_no_partial_output() {
+    struct Unaware(AtomicBool);
+    impl crate::accelerator::MatVecKernel for Unaware {
+        fn try_execute(
+            &self,
+            _: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            self.0.store(true, Ordering::Relaxed);
+            panic!("typed attention never calls an unaware provider")
+        }
+    }
+    let fixture = Fixture::new();
+    let loaded = attention_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let bytes = raw_input();
+    let mut math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
+    let imported = math
+        .import_attention_input(input(source.source_identity(), &bytes), 256)
+        .unwrap();
+    let before = reserved(&admission);
+    let gdn_gamma = operators
+        .parameter_descriptor(QwenParameterRole::GdnNorm(0))
+        .unwrap();
+    assert_eq!(
+        OrdinaryRmsContract::TorchCpuBf16Avx2
+            .validate(&gdn_gamma, 256)
+            .unwrap(),
+        256
+    );
+    assert!(
+        math.ordinary_rms_profiled(gdn_gamma, &imported, OrdinaryRmsContract::TorchCpuBf16Avx2)
+            .is_err()
+    );
+    assert!(
+        math.project_profiled(
+            operators
+                .matrix_descriptor(QwenMatrixRole::FfnGate(3))
+                .unwrap(),
+            &imported,
+            ProjectionContract::TorchCpuBf16Avx2
+        )
+        .is_err()
+    );
+    assert!(
+        math.ordinary_rms_profiled(
+            operators
+                .parameter_descriptor(QwenParameterRole::QueryNorm(3))
+                .unwrap(),
+            &imported,
+            OrdinaryRmsContract::TorchCpuBf16Avx2
+        )
+        .is_err()
+    ); // Two query heads required.
+    let mut forged = operators
+        .matrix_descriptor(QwenMatrixRole::Query(3))
+        .unwrap();
+    forged.rows -= 1;
+    assert!(
+        math.project_profiled(forged, &imported, ProjectionContract::TorchCpuBf16Avx2)
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before);
+    let provider = Unaware(AtomicBool::new(false));
+    let unaware =
+        crate::host_math::HostMath::for_window_operator_slice(&operators, None, Some(&provider));
+    assert!(
+        unaware
+            .import_attention_input(input(source.source_identity(), &bytes), 256)
+            .is_err()
+    );
+    assert!(!provider.0.load(Ordering::Relaxed));
+    assert_eq!(reserved(&admission), before);
+    let (other_source, other_admission) = windows(&fixture, &loaded);
+    let other = bind_windows(other_source.clone(), &loaded);
+    let mut foreign = crate::host_math::HostMath::for_window_operator_slice(&other, None, None);
+    let other_before = reserved(&other_admission);
+    assert!(
+        foreign
+            .project_profiled(
+                other.matrix_descriptor(QwenMatrixRole::Query(3)).unwrap(),
+                &imported,
+                ProjectionContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&other_admission), other_before);
+    drop(foreign);
+    drop(other);
+    drop(other_source);
+    assert_eq!(reserved(&other_admission), 0);
+}
+
+#[test]
+fn cancel_and_real_capacity_refusal_release_partial_private_values() {
+    let fixture = Fixture::new();
+    let loaded = attention_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let bytes = raw_input();
+    let before = reserved(&admission);
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        operators
+            .execute_attention_projection_norm_slice(
+                3,
+                input(source.source_identity(), &bytes),
+                Some(&cancelled)
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before);
+    cancelled.store(false, Ordering::Release);
+    let mut math =
+        crate::host_math::HostMath::for_window_operator_slice(&operators, Some(&cancelled), None);
+    let imported = math
+        .import_attention_input(input(source.source_identity(), &bytes), 256)
+        .unwrap();
+    assert_eq!(reserved(&admission), before + 1024 * 1024);
+    cancelled.store(true, Ordering::Release);
+    assert!(
+        crate::forward::attention_projection_norm_slice(&operators, &mut math, 3, &imported)
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before + 1024 * 1024);
+    drop(imported);
+    drop(math);
+    assert_eq!(reserved(&admission), before);
+    let mut pressure = Vec::new();
+    loop {
+        let remaining = admission
+            .arbiter
+            .snapshot(&admission.resource)
+            .unwrap()
+            .remaining;
+        if remaining == 0 {
+            break;
+        }
+        pressure.push(
+            source
+                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
+                .unwrap(),
+        );
+    }
+    assert!(
+        operators
+            .execute_attention_projection_norm_slice(
+                3,
+                input(source.source_identity(), &bytes),
+                None
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), 128 * 1024 * 1024);
+    drop(pressure);
+    assert_eq!(reserved(&admission), before);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -116,6 +116,83 @@
         ));
     }
     Ok((embedding, normalized, projected))
+}
+
+/// Pure prefix of the existing attention role schedule, before rotary/state.
+pub(crate) fn attention_projection_norm_slice<B: MathematicalBackend>(
+    model: &impl OperatorSliceSource,
+    math: &mut B,
+    layer: usize,
+    input: &B::Value,
+) -> Result<crate::operator_slice::attention::AttentionProjectionValues<B::Value>> {
+    use crate::accelerator::program::ActivationPrecision;
+    use crate::operator_slice::attention::{AttentionProjectionValues, attention_roles};
+    let roles = attention_roles(model, layer)?;
+    if math.elements(input) != roles.hidden
+        || math.precision(input) != ActivationPrecision::Bf16WidenedF32
+    {
+        return Err(Error::Malformed(
+            "attention slice typed full hidden input".into(),
+        ));
+    }
+    let query_gate_projection =
+        math.project_profiled(roles.query, input, model.projection_contract())?;
+    let key_projection = math.project_profiled(roles.key, input, model.projection_contract())?;
+    let value_projection =
+        math.project_profiled(roles.value, input, model.projection_contract())?;
+    let query_view = math.view(
+        &query_gate_projection,
+        TensorView {
+            offset: 0,
+            rows: roles.query_heads,
+            width: roles.width,
+            row_stride: roles.width * 2,
+        },
+    )?;
+    let query = math.contiguous(&query_view)?;
+    let gate_view = math.view(
+        &query_gate_projection,
+        TensorView {
+            offset: roles.width,
+            rows: roles.query_heads,
+            width: roles.width,
+            row_stride: roles.width * 2,
+        },
+    )?;
+    let gate = math.contiguous(&gate_view)?;
+    let normalized_query = math.ordinary_rms_profiled(
+        roles.query_norm,
+        &query,
+        OrdinaryRmsContract::TorchCpuBf16Avx2,
+    )?;
+    let normalized_key = math.ordinary_rms_profiled(
+        roles.key_norm,
+        &key_projection,
+        OrdinaryRmsContract::TorchCpuBf16Avx2,
+    )?;
+    if [
+        &query_gate_projection,
+        &key_projection,
+        &value_projection,
+        &query,
+        &gate,
+        &normalized_query,
+        &normalized_key,
+    ]
+    .iter()
+    .any(|value| math.precision(value) != ActivationPrecision::Bf16WidenedF32)
+    {
+        return Err(Error::State("attention slice output precision".into()));
+    }
+    Ok(AttentionProjectionValues {
+        query_gate_projection,
+        key_projection,
+        value_projection,
+        query,
+        gate,
+        normalized_query,
+        normalized_key,
+    })
 }
 
 fn expect_elements<B: MathematicalBackend>(
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -232,6 +232,60 @@
             }
         }
         Ok(Some(Arc::new(source.workspace()?)))
+    }
+    /// Explicit operator input transport, never a model producer or root.
+    pub(super) fn import_attention_input(
+        &self,
+        input: crate::operator_slice::attention::AttentionSliceInput<'_>,
+        hidden: usize,
+    ) -> Result<HostValue> {
+        input.validate()?;
+        if self.region_kernel.is_some() {
+            return Err(Error::Unsupported(
+                "typed attention CPU input provider".into(),
+            ));
+        }
+        if input.source_identity() != self.source.source_identity()
+            || input.numerical_profile() != self.source.program_numerical_profile()
+            || self.source.metadata().geometry().hidden_size != hidden as u64
+            || input.bytes().len()
+                != hidden
+                    .checked_mul(2)
+                    .ok_or_else(|| Error::Limit("attention private input width".into()))?
+        {
+            return Err(Error::State(
+                "attention input source/profile/full hidden width".into(),
+            ));
+        }
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let workspace = self.prepare_output(None)?;
+        let mut values = Vec::new();
+        values
+            .try_reserve_exact(hidden)
+            .map_err(|_| Error::Limit("private attention input transport".into()))?;
+        values.extend(
+            input.bytes().chunks_exact(2).map(|word| {
+                f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16)
+            }),
+        );
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        if self
+            .cancelled
+            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
+        {
+            return Err(Error::State(
+                "attention input cancelled after transport".into(),
+            ));
+        }
+        let output = HostValue::typed(
+            values,
+            input.precision(),
+            input.source_identity(),
+            input.numerical_profile(),
+        )?;
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let output = output.retain_workspace(workspace)?;
+        Ok(output)
     }
     fn engine(&self) -> Result<&'engine QwenDenseCpu> {
         self.engine
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -8,6 +8,7 @@
 
 /// Pure role metadata only; never a layer/token/session execution capability.
 pub(crate) trait OperatorSliceSource {
+    fn geometry(&self) -> &crate::model::ModelGeometry;
     fn source_identity(&self) -> crate::accelerator::SourceIdentity;
     fn program_numerical_profile(&self) -> ProgramNumericalProfile;
     fn rms_epsilon(&self) -> f32;
@@ -23,6 +24,9 @@
     fn projection_contract(&self) -> ProjectionContract;
 }
 impl OperatorSliceSource for LoadedQwenModel {
+    fn geometry(&self) -> &crate::model::ModelGeometry {
+        LoadedQwenModel::geometry(self)
+    }
     fn source_identity(&self) -> crate::accelerator::SourceIdentity {
         LoadedQwenModel::source_identity(self)
     }
@@ -223,3 +227,7 @@
 ))]
 #[path = "operator_slice_tests.rs"]
 mod tests;
+
+/// Pure attention projections/QK norms; no rotary/cache/session admission.
+#[path = "attention_slice.rs"]
+pub mod attention;
--- a/crates/zerogravity-model-qwen35/src/operator_slice_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_tests.rs
@@ -87,6 +87,15 @@
     cfg["text_config"]["hidden_size"] = json!(hidden);
     cfg["text_config"]["vocab_size"] = json!(vocabulary);
     let cfg = serde_json::to_vec(&cfg).unwrap();
+    source_with_config(fixture, cfg, norm_bf16, |_, _, bits| bits, mutate)
+}
+fn source_with_config(
+    fixture: &Fixture,
+    cfg: Vec<u8>,
+    norm_bf16: bool,
+    bf16_word: impl Fn(&str, usize, u16) -> u16,
+    mutate: impl FnOnce(&mut Value),
+) -> Loaded {
     let specs = QwenTextConfig::from_json(&cfg)
         .unwrap()
         .tensor_specs()
@@ -103,7 +112,7 @@
             "BF16"
         };
         let start = payload.len();
-        for _ in 0..spec.shape.iter().product::<u64>() {
+        for element in 0..spec.shape.iter().product::<u64>() {
             match kind {
                 "F32" => payload.extend_from_slice(&0.25f32.to_le_bytes()),
                 "F16" => payload.extend_from_slice(&0x3400u16.to_le_bytes()),
@@ -113,7 +122,9 @@
                     } else {
                         0x3f80
                     };
-                    payload.extend_from_slice(&bits.to_le_bytes());
+                    payload.extend_from_slice(
+                        &bf16_word(&spec.name, element as usize, bits).to_le_bytes(),
+                    );
                 }
                 _ => unreachable!(),
             }
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -376,3 +376,6 @@
     drop(source);
     assert_eq!(reserved(&admission), 0);
 }
+
+#[path = "attention_slice_tests.rs"]
+mod attention;
--- a/crates/zerogravity-model-qwen35/src/original_binding/window.rs
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window.rs
@@ -194,7 +194,34 @@
         cancelled: Option<&AtomicBool>,
     ) -> Result<Vec<f32>> {
         if *parameter != self.parameter_descriptor(QwenParameterRole::InputNorm(0))? {
-            return Err(Error::State("window ordinary RMS original role".into()));
+            let mut valid = false;
+            for (layer, kind) in self.geometry().layers.iter().enumerate() {
+                if *kind != crate::model::LayerKind::FullAttention {
+                    continue;
+                }
+                if *parameter != self.parameter_descriptor(QwenParameterRole::QueryNorm(layer))?
+                    && *parameter != self.parameter_descriptor(QwenParameterRole::KeyNorm(layer))?
+                {
+                    continue;
+                }
+                let roles = crate::operator_slice::attention::attention_roles(self, layer)?;
+                let elements = if *parameter == roles.query_norm {
+                    Some(roles.query_heads * roles.width)
+                } else if *parameter == roles.key_norm {
+                    Some(roles.kv_heads * roles.width)
+                } else {
+                    None
+                };
+                if elements == Some(input.len()) {
+                    valid = true;
+                    break;
+                }
+            }
+            if !valid {
+                return Err(Error::State(
+                    "window ordinary RMS original role/heads".into(),
+                ));
+            }
         }
         check_cancel(cancelled)?;
         let raw = self.source().parameter(&parameter.name)?;
@@ -234,9 +261,31 @@
     ) -> Result<Vec<f32>> {
         self.validate_matrix(tensor)?;
         if *tensor != self.matrix_descriptor(QwenMatrixRole::GdnQkv(0))? {
-            return Err(Error::Unsupported(
-                "closed original window projection role".into(),
-            ));
+            let mut valid = false;
+            for (layer, kind) in self.geometry().layers.iter().enumerate() {
+                if *kind != crate::model::LayerKind::FullAttention {
+                    continue;
+                }
+                if ![
+                    self.matrix_descriptor(QwenMatrixRole::Query(layer))?,
+                    self.matrix_descriptor(QwenMatrixRole::Key(layer))?,
+                    self.matrix_descriptor(QwenMatrixRole::Value(layer))?,
+                ]
+                .contains(tensor)
+                {
+                    continue;
+                }
+                let roles = crate::operator_slice::attention::attention_roles(self, layer)?;
+                if [&roles.query, &roles.key, &roles.value].contains(&tensor) {
+                    valid = true;
+                    break;
+                }
+            }
+            if !valid {
+                return Err(Error::Unsupported(
+                    "closed original window projection role".into(),
+                ));
+            }
         }
         check_cancel(cancelled)?;
         if input.len() != tensor.columns {
@@ -285,6 +334,9 @@
     }
 }
 impl OperatorSliceSource for OriginalWindowOperators {
+    fn geometry(&self) -> &crate::model::ModelGeometry {
+        self.binding.config.geometry()
+    }
     fn source_identity(&self) -> crate::accelerator::SourceIdentity {
         self.checked.source_identity()
     }

````

## Source: .agents/runs/model-execution-original-attention-slice-20261020/frozen/NEW_DELTA.patch

Original UTF-8 bytes: 40847; SHA256 `cf5187a2f46d0cfb54707a0cbb6f5ef9940f10378441922aca1977754bfe985d`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/attention_slice.rs
@@ -0,0 +1,327 @@
+//! Closed projections/QK norms over an explicit private input, never a layer.
+use super::OperatorSliceSource;
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{
+    ActivationPrecision, MatrixAxes, OrdinaryRmsContract, PlanBuilder, PlanValue,
+    ProgramNumericalProfile, SourceParameter, SourceTensor,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, LoadedQwenModel, Result};
+use zerogravity_core::quant::QuantKind;
+
+pub(crate) struct AttentionRoles {
+    pub(crate) hidden: usize,
+    pub(crate) query_heads: usize,
+    pub(crate) kv_heads: usize,
+    pub(crate) width: usize,
+    pub(crate) query: SourceTensor,
+    pub(crate) key: SourceTensor,
+    pub(crate) value: SourceTensor,
+    pub(crate) query_norm: SourceParameter,
+    pub(crate) key_norm: SourceParameter,
+}
+
+/// Check every role before recording or importing a private activation.
+pub(crate) fn attention_roles(
+    source: &impl OperatorSliceSource,
+    layer: usize,
+) -> Result<AttentionRoles> {
+    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || source.rms_epsilon().to_bits() != 1e-6f32.to_bits()
+        || !source.source_identity().is_bound()
+    {
+        return Err(Error::Unsupported(
+            "closed attention slice profile/source/epsilon".into(),
+        ));
+    }
+    let geometry = source.geometry();
+    if geometry.layers.get(layer) != Some(&crate::model::LayerKind::FullAttention) {
+        return Err(Error::Unsupported(
+            "attention slice requires actual FullAttention layer".into(),
+        ));
+    }
+    let size = |value| {
+        usize::try_from(value).map_err(|_| Error::Limit("original attention geometry".into()))
+    };
+    let hidden = size(geometry.hidden_size)?;
+    let query_heads = size(geometry.num_attention_heads)?;
+    let kv_heads = size(geometry.num_key_value_heads)?;
+    let width = size(geometry.head_dim)?;
+    if !matches!(hidden, 256 | 1024 | 2048 | 5120)
+        || width != 256
+        || query_heads == 0
+        || kv_heads == 0
+        || !query_heads.is_multiple_of(kv_heads)
+    {
+        return Err(Error::Unsupported(
+            "closed attention slice head/hidden geometry".into(),
+        ));
+    }
+    let query_rows = query_heads
+        .checked_mul(width)
+        .and_then(|n| n.checked_mul(2))
+        .filter(|n| *n <= 16384)
+        .ok_or_else(|| Error::Limit("attention query/gate rows".into()))?;
+    let kv_rows = kv_heads
+        .checked_mul(width)
+        .filter(|n| *n <= 16384)
+        .ok_or_else(|| Error::Limit("attention key/value rows".into()))?;
+    let query = source.matrix_descriptor(QwenMatrixRole::Query(layer))?;
+    let key = source.matrix_descriptor(QwenMatrixRole::Key(layer))?;
+    let value = source.matrix_descriptor(QwenMatrixRole::Value(layer))?;
+    let namespace = format!("model.language_model.layers.{layer}.self_attn");
+    for (matrix, rows, suffix) in [
+        (&query, query_rows, "q_proj"),
+        (&key, kv_rows, "k_proj"),
+        (&value, kv_rows, "v_proj"),
+    ] {
+        if matrix.name != format!("{namespace}.{suffix}.weight")
+            || matrix.source != source.source_identity()
+            || matrix.kind != QuantKind::Bf16
+            || matrix.axes != MatrixAxes::OutputInput
+            || matrix.columns != hidden
+            || matrix.rows != rows
+        {
+            return Err(Error::State(
+                "attention original full matrix role/axes/shape".into(),
+            ));
+        }
+        matrix.packed_bytes()?;
+    }
+    let query_norm = source.parameter_descriptor(QwenParameterRole::QueryNorm(layer))?;
+    let key_norm = source.parameter_descriptor(QwenParameterRole::KeyNorm(layer))?;
+    for (parameter, elements, suffix) in [
+        (&query_norm, query_rows / 2, "q_norm"),
+        (&key_norm, kv_rows, "k_norm"),
+    ] {
+        if parameter.name != format!("{namespace}.{suffix}.weight")
+            || parameter.source != source.source_identity()
+            || OrdinaryRmsContract::TorchCpuBf16Avx2.validate(parameter, elements)? != width
+        {
+            return Err(Error::State(
+                "attention original Q/K norm role/width".into(),
+            ));
+        }
+    }
+    Ok(AttentionRoles {
+        hidden,
+        query_heads,
+        kv_heads,
+        width,
+        query,
+        key,
+        value,
+        query_norm,
+        key_norm,
+    })
+}
+
+/// Existing typed values only. The input's producer remains the caller's.
+/// No state domains, rotary, cache effects, token schedule or logits are added.
+pub struct AttentionProjectionValues<V> {
+    pub(crate) query_gate_projection: V,
+    pub(crate) key_projection: V,
+    pub(crate) value_projection: V,
+    pub(crate) query: V,
+    pub(crate) gate: V,
+    pub(crate) normalized_query: V,
+    pub(crate) normalized_key: V,
+}
+impl<V> AttentionProjectionValues<V> {
+    pub fn query_gate_projection(&self) -> &V {
+        &self.query_gate_projection
+    }
+    pub fn key_projection(&self) -> &V {
+        &self.key_projection
+    }
+    pub fn value_projection(&self) -> &V {
+        &self.value_projection
+    }
+    pub fn query(&self) -> &V {
+        &self.query
+    }
+    pub fn gate(&self) -> &V {
+        &self.gate
+    }
+    pub fn normalized_query(&self) -> &V {
+        &self.normalized_query
+    }
+    pub fn normalized_key(&self) -> &V {
+        &self.normalized_key
+    }
+}
+impl LoadedQwenModel {
+    /// Append to the caller's same-builder, same-source typed input.
+    /// An embedding fixture is transport evidence, not a trained layer input.
+    pub fn append_attention_projection_norm_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &PlanValue,
+    ) -> Result<AttentionProjectionValues<PlanValue>> {
+        self.require_operator_slice()?;
+        crate::forward::attention_projection_norm_slice(self, builder, layer, input)
+    }
+}
+
+/// Qualifier-supplied raw BF16 bytes. The caller authenticates their external
+/// source/archive provenance; this contract checks only binding/representation.
+/// Import never asserts a model producer, StateRoot or token frontier.
+#[derive(Clone, Copy)]
+pub struct AttentionSliceInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    bytes: &'a [u8],
+}
+impl<'a> AttentionSliceInput<'a> {
+    pub fn from_bf16_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        bytes: &'a [u8],
+    ) -> Result<Self> {
+        let value = Self {
+            source,
+            profile,
+            bytes,
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
+    pub fn precision(self) -> ActivationPrecision {
+        ActivationPrecision::Bf16WidenedF32
+    }
+    pub fn bytes(self) -> &'a [u8] {
+        self.bytes
+    }
+    pub(crate) fn validate(self) -> Result<()> {
+        if !self.source.is_bound()
+            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::State(
+                "attention private input source/profile".into(),
+            ));
+        }
+        if self.bytes.is_empty()
+            || self.bytes.len() > 2 * 5120
+            || !self.bytes.len().is_multiple_of(2)
+            || self
+                .bytes
+                .chunks_exact(2)
+                .any(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 == 0x7f80)
+        {
+            return Err(Error::Malformed(
+                "attention private finite BF16 input bytes".into(),
+            ));
+        }
+        Ok(())
+    }
+}
+
+#[cfg(feature = "cpu")]
+pub struct AttentionProjectionNormOutput {
+    source: SourceIdentity,
+    layer: usize,
+    input: crate::host_math::HostValue,
+    values: AttentionProjectionValues<crate::host_math::HostValue>,
+}
+#[cfg(feature = "cpu")]
+impl AttentionProjectionNormOutput {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn layer(&self) -> usize {
+        self.layer
+    }
+    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    }
+    pub fn precision(&self) -> ActivationPrecision {
+        ActivationPrecision::Bf16WidenedF32
+    }
+    pub fn input(&self) -> Result<&[f32]> {
+        self.input.as_slice()
+    }
+    pub fn query_gate_projection(&self) -> Result<&[f32]> {
+        self.values.query_gate_projection.as_slice()
+    }
+    pub fn key_projection(&self) -> Result<&[f32]> {
+        self.values.key_projection.as_slice()
+    }
+    pub fn value_projection(&self) -> Result<&[f32]> {
+        self.values.value_projection.as_slice()
+    }
+    pub fn query(&self) -> Result<&[f32]> {
+        self.values.query.as_slice()
+    }
+    pub fn gate(&self) -> Result<&[f32]> {
+        self.values.gate.as_slice()
+    }
+    pub fn normalized_query(&self) -> Result<&[f32]> {
+        self.values.normalized_query.as_slice()
+    }
+    pub fn normalized_key(&self) -> Result<&[f32]> {
+        self.values.normalized_key.as_slice()
+    }
+}
+#[cfg(feature = "cpu")]
+impl LoadedQwenModel {
+    /// Pure numerical fragment on explicit observed operator input, no session.
+    pub fn execute_attention_projection_norm_slice(
+        &self,
+        layer: usize,
+        input: AttentionSliceInput<'_>,
+    ) -> Result<AttentionProjectionNormOutput> {
+        self.require_operator_slice()?;
+        let roles = attention_roles(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_operator_slice(self, None);
+        let input = math.import_attention_input(input, roles.hidden)?;
+        let values =
+            crate::forward::attention_projection_norm_slice(self, &mut math, layer, &input)?;
+        Ok(AttentionProjectionNormOutput {
+            source: self.source_identity(),
+            layer,
+            input,
+            values,
+        })
+    }
+}
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    /// No external-input node is invented; caller retains the source/metadata.
+    pub fn append_attention_projection_norm_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &PlanValue,
+    ) -> Result<AttentionProjectionValues<PlanValue>> {
+        crate::forward::attention_projection_norm_slice(self, builder, layer, input)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_attention_projection_norm_slice(
+        &self,
+        layer: usize,
+        input: AttentionSliceInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<AttentionProjectionNormOutput> {
+        let roles = attention_roles(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let input = math.import_attention_input(input, roles.hidden)?;
+        let values =
+            crate::forward::attention_projection_norm_slice(self, &mut math, layer, &input)?;
+        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State("window attention slice cancelled".into()));
+        }
+        Ok(AttentionProjectionNormOutput {
+            source: self.source_identity(),
+            layer,
+            input,
+            values,
+        })
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/attention_slice_tests.rs
@@ -0,0 +1,499 @@
+//! Synthetic exact-role transport/ownership only; no trained hidden-state claim.
+use super::*;
+use crate::accelerator::program::{
+    ActivationPrecision, PlanBuilder, ProgramNumericalProfile, ProjectionContract,
+};
+use crate::operator_slice::attention::AttentionSliceInput;
+
+fn attention_source(fixture: &Fixture) -> Loaded {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    cfg["text_config"]["hidden_size"] = json!(256);
+    cfg["text_config"]["head_dim"] = json!(256);
+    // Deliberately same width as Q/K norms: role checks must reject GDN gamma.
+    cfg["text_config"]["linear_value_head_dim"] = json!(256);
+    source_with_config(
+        fixture,
+        serde_json::to_vec(&cfg).unwrap(),
+        true,
+        |name, element, fallback| {
+            let value = if name.ends_with(".self_attn.q_proj.weight") {
+                [1.0f32, -2.0, 3.0, -4.0][element / (256 * 256)]
+            } else if name.ends_with(".self_attn.k_proj.weight") {
+                5.0
+            } else if name.ends_with(".self_attn.v_proj.weight") {
+                6.0
+            } else {
+                return fallback;
+            };
+            if element.is_multiple_of(256) {
+                (value.to_bits() >> 16) as u16
+            } else {
+                0
+            }
+        },
+        |_| {},
+    )
+}
+fn raw_input() -> Vec<u8> {
+    let mut bytes = vec![0; 512];
+    bytes[..2].copy_from_slice(&0x3f80u16.to_le_bytes());
+    bytes
+}
+fn input(source: crate::accelerator::SourceIdentity, bytes: &[u8]) -> AttentionSliceInput<'_> {
+    AttentionSliceInput::from_bf16_bytes(
+        source,
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        bytes,
+    )
+    .unwrap()
+}
+
+#[test]
+fn explicit_attention_input_uses_exact_rows_raw_norms_and_no_session() {
+    let fixture = Fixture::new();
+    let loaded = attention_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let (model, _, _) = bind(loaded);
+    let bytes = raw_input();
+    let expected = model
+        .execute_attention_projection_norm_slice(3, input(model.source_identity(), &bytes))
+        .unwrap();
+    let before = reserved(&admission);
+    let output = operators
+        .execute_attention_projection_norm_slice(3, input(source.source_identity(), &bytes), None)
+        .unwrap();
+    assert_eq!(output.source_identity(), source.source_identity());
+    assert_eq!(output.layer(), 3);
+    assert_eq!(output.precision(), ActivationPrecision::Bf16WidenedF32);
+    assert_eq!(output.numerical_profile(), expected.numerical_profile());
+    same(expected.input().unwrap(), output.input().unwrap());
+    same(
+        expected.query_gate_projection().unwrap(),
+        output.query_gate_projection().unwrap(),
+    );
+    same(
+        expected.key_projection().unwrap(),
+        output.key_projection().unwrap(),
+    );
+    same(
+        expected.value_projection().unwrap(),
+        output.value_projection().unwrap(),
+    );
+    same(expected.query().unwrap(), output.query().unwrap());
+    same(expected.gate().unwrap(), output.gate().unwrap());
+    same(
+        expected.normalized_query().unwrap(),
+        output.normalized_query().unwrap(),
+    );
+    same(
+        expected.normalized_key().unwrap(),
+        output.normalized_key().unwrap(),
+    );
+    // Independent integer row labels distinguish [head,q/gate,width] packing.
+    assert_eq!(
+        output.query_gate_projection().unwrap(),
+        [
+            vec![1.0; 256],
+            vec![-2.0; 256],
+            vec![3.0; 256],
+            vec![-4.0; 256]
+        ]
+        .concat()
+    );
+    assert_eq!(
+        output.query().unwrap(),
+        [vec![1.0; 256], vec![3.0; 256]].concat()
+    );
+    assert_eq!(
+        output.gate().unwrap(),
+        [vec![-2.0; 256], vec![-4.0; 256]].concat()
+    );
+    assert_eq!(output.key_projection().unwrap(), &[5.0; 256]);
+    assert_eq!(output.value_projection().unwrap(), &[6.0; 256]);
+    assert_eq!(reserved(&admission), before + 8 * 1024 * 1024);
+    assert_eq!(model.initialized_cpu_state_bytes(), 0);
+    assert!(model.new_session().is_err());
+    assert!(model.forward_plan(8, true, false).is_err());
+    let query_norm = model
+        .parameter_descriptor(QwenParameterRole::QueryNorm(3))
+        .unwrap();
+    assert_eq!(query_norm.kind, zerogravity_core::quant::QuantKind::F32);
+    assert!(
+        model
+            .tensor_bytes(&query_norm.name)
+            .unwrap()
+            .chunks_exact(4)
+            .all(|word| word == 0.25f32.to_le_bytes())
+    );
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 40 * 1024 * 1024);
+    same(
+        expected.normalized_query().unwrap(),
+        output.normalized_query().unwrap(),
+    );
+    drop(output);
+    assert_eq!(reserved(&admission), 0);
+    assert!(
+        !admission
+            .arbiter
+            .snapshot(&admission.resource)
+            .unwrap()
+            .fenced
+    );
+}
+
+#[test]
+fn plan_append_has_no_input_invention_or_state_and_rejects_foreign_builder() {
+    let fixture = Fixture::new();
+    let (model, _, _) = bind(attention_source(&fixture));
+    let profile = model.program_numerical_profile();
+    let mut builder = PlanBuilder::new_operator_slice(model.source_identity(), profile).unwrap();
+    // Synthetic TypedEmbed is ONLY an existing same-builder transport fixture.
+    let embedding = builder
+        .embed_profiled(
+            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    assert!(
+        model
+            .append_attention_projection_norm_slice(&mut builder, 0, &embedding)
+            .is_err()
+    );
+    assert!(
+        model
+            .append_attention_projection_norm_slice(&mut builder, 4, &embedding)
+            .is_err()
+    );
+    let values = model
+        .append_attention_projection_norm_slice(&mut builder, 3, &embedding)
+        .unwrap();
+    let recorded = builder
+        .finish_operator_slice(*values.normalized_query())
+        .unwrap();
+    assert_eq!(recorded.plan().nodes().len(), 10);
+    assert_eq!(
+        recorded
+            .plan()
+            .nodes()
+            .iter()
+            .filter(|node| matches!(node.operation, PlanOperation::TypedProject { .. }))
+            .count(),
+        3
+    );
+    assert_eq!(
+        recorded
+            .plan()
+            .nodes()
+            .iter()
+            .filter(|node| matches!(node.operation, PlanOperation::TypedOrdinaryRms { .. }))
+            .count(),
+        2
+    );
+    let views = recorded
+        .plan()
+        .nodes()
+        .iter()
+        .filter_map(|node| match node.operation {
+            PlanOperation::View(view) => Some(view),
+            _ => None,
+        })
+        .collect::<Vec<_>>();
+    assert_eq!(views.len(), 2);
+    assert_eq!(
+        (views[0].offset, views[1].offset, views[0].row_stride),
+        (0, 256, 512)
+    );
+    assert_eq!(
+        values.query_gate_projection().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert_eq!(
+        values.key_projection().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert_eq!(
+        values.value_projection().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert_eq!(
+        values.query().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert_eq!(
+        values.gate().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert_eq!(
+        values.normalized_key().precision(),
+        ActivationPrecision::Bf16WidenedF32
+    );
+    assert!(recorded.plan().required_domains().is_empty());
+    assert!(recorded.plan().proposed_domains().is_empty());
+    assert_eq!(recorded.plan().frontier_delta(), 0);
+    let mut first = PlanBuilder::new_operator_slice(model.source_identity(), profile).unwrap();
+    let foreign = first
+        .embed_profiled(
+            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    let mut second = PlanBuilder::new_operator_slice(model.source_identity(), profile).unwrap();
+    assert!(
+        model
+            .append_attention_projection_norm_slice(&mut second, 3, &foreign)
+            .is_err()
+    );
+    assert_eq!(
+        first
+            .finish_operator_slice(foreign)
+            .unwrap()
+            .plan()
+            .nodes()
+            .len(),
+        1
+    );
+}
+
+#[test]
+fn raw_bf16_import_preserves_zero_subnormal_bits_and_checks_before_admission() {
+    let fixture = Fixture::new();
+    let loaded = attention_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
+    let mut bytes = raw_input();
+    for (index, bits) in [0u16, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f]
+        .into_iter()
+        .enumerate()
+    {
+        bytes[2 * index..2 * index + 2].copy_from_slice(&bits.to_le_bytes());
+    }
+    let before = reserved(&admission);
+    let imported = math
+        .import_attention_input(input(source.source_identity(), &bytes), 256)
+        .unwrap();
+    for (word, value) in bytes.chunks_exact(2).zip(imported.as_slice().unwrap()) {
+        assert_eq!(
+            value.to_bits(),
+            u32::from(u16::from_le_bytes([word[0], word[1]])) << 16
+        );
+    }
+    assert_eq!(reserved(&admission), before + 1024 * 1024);
+    assert!(imported.clone().into_vec().is_err());
+    drop(imported);
+    assert_eq!(reserved(&admission), before);
+    for invalid in [
+        Vec::new(),
+        vec![0],
+        vec![0; 2 * 5120 + 2],
+        0x7f80u16.to_le_bytes().to_vec(),
+        0xff80u16.to_le_bytes().to_vec(),
+        0x7fc0u16.to_le_bytes().to_vec(),
+    ] {
+        assert!(
+            AttentionSliceInput::from_bf16_bytes(
+                source.source_identity(),
+                operators.program_numerical_profile(),
+                &invalid
+            )
+            .is_err()
+        );
+    }
+    assert!(
+        AttentionSliceInput::from_bf16_bytes(
+            crate::accelerator::SourceIdentity::unbound(),
+            operators.program_numerical_profile(),
+            &bytes
+        )
+        .is_err()
+    );
+    assert!(
+        AttentionSliceInput::from_bf16_bytes(
+            source.source_identity(),
+            ProgramNumericalProfile::GgufSourceV1,
+            &bytes
+        )
+        .is_err()
+    );
+    let foreign = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
+    assert!(
+        math.import_attention_input(input(foreign, &bytes), 256)
+            .is_err()
+    );
+    assert!(
+        math.import_attention_input(input(source.source_identity(), &bytes[..510]), 256)
+            .is_err()
+    );
+    assert!(
+        math.import_attention_input(input(source.source_identity(), &bytes), 128)
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before);
+}
+
+#[test]
+fn exact_role_provider_and_foreign_issuer_refusals_have_no_partial_output() {
+    struct Unaware(AtomicBool);
+    impl crate::accelerator::MatVecKernel for Unaware {
+        fn try_execute(
+            &self,
+            _: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            self.0.store(true, Ordering::Relaxed);
+            panic!("typed attention never calls an unaware provider")
+        }
+    }
+    let fixture = Fixture::new();
+    let loaded = attention_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let bytes = raw_input();
+    let mut math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
+    let imported = math
+        .import_attention_input(input(source.source_identity(), &bytes), 256)
+        .unwrap();
+    let before = reserved(&admission);
+    let gdn_gamma = operators
+        .parameter_descriptor(QwenParameterRole::GdnNorm(0))
+        .unwrap();
+    assert_eq!(
+        OrdinaryRmsContract::TorchCpuBf16Avx2
+            .validate(&gdn_gamma, 256)
+            .unwrap(),
+        256
+    );
+    assert!(
+        math.ordinary_rms_profiled(gdn_gamma, &imported, OrdinaryRmsContract::TorchCpuBf16Avx2)
+            .is_err()
+    );
+    assert!(
+        math.project_profiled(
+            operators
+                .matrix_descriptor(QwenMatrixRole::FfnGate(3))
+                .unwrap(),
+            &imported,
+            ProjectionContract::TorchCpuBf16Avx2
+        )
+        .is_err()
+    );
+    assert!(
+        math.ordinary_rms_profiled(
+            operators
+                .parameter_descriptor(QwenParameterRole::QueryNorm(3))
+                .unwrap(),
+            &imported,
+            OrdinaryRmsContract::TorchCpuBf16Avx2
+        )
+        .is_err()
+    ); // Two query heads required.
+    let mut forged = operators
+        .matrix_descriptor(QwenMatrixRole::Query(3))
+        .unwrap();
+    forged.rows -= 1;
+    assert!(
+        math.project_profiled(forged, &imported, ProjectionContract::TorchCpuBf16Avx2)
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before);
+    let provider = Unaware(AtomicBool::new(false));
+    let unaware =
+        crate::host_math::HostMath::for_window_operator_slice(&operators, None, Some(&provider));
+    assert!(
+        unaware
+            .import_attention_input(input(source.source_identity(), &bytes), 256)
+            .is_err()
+    );
+    assert!(!provider.0.load(Ordering::Relaxed));
+    assert_eq!(reserved(&admission), before);
+    let (other_source, other_admission) = windows(&fixture, &loaded);
+    let other = bind_windows(other_source.clone(), &loaded);
+    let mut foreign = crate::host_math::HostMath::for_window_operator_slice(&other, None, None);
+    let other_before = reserved(&other_admission);
+    assert!(
+        foreign
+            .project_profiled(
+                other.matrix_descriptor(QwenMatrixRole::Query(3)).unwrap(),
+                &imported,
+                ProjectionContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&other_admission), other_before);
+    drop(foreign);
+    drop(other);
+    drop(other_source);
+    assert_eq!(reserved(&other_admission), 0);
+}
+
+#[test]
+fn cancel_and_real_capacity_refusal_release_partial_private_values() {
+    let fixture = Fixture::new();
+    let loaded = attention_source(&fixture);
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let bytes = raw_input();
+    let before = reserved(&admission);
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        operators
+            .execute_attention_projection_norm_slice(
+                3,
+                input(source.source_identity(), &bytes),
+                Some(&cancelled)
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before);
+    cancelled.store(false, Ordering::Release);
+    let mut math =
+        crate::host_math::HostMath::for_window_operator_slice(&operators, Some(&cancelled), None);
+    let imported = math
+        .import_attention_input(input(source.source_identity(), &bytes), 256)
+        .unwrap();
+    assert_eq!(reserved(&admission), before + 1024 * 1024);
+    cancelled.store(true, Ordering::Release);
+    assert!(
+        crate::forward::attention_projection_norm_slice(&operators, &mut math, 3, &imported)
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before + 1024 * 1024);
+    drop(imported);
+    drop(math);
+    assert_eq!(reserved(&admission), before);
+    let mut pressure = Vec::new();
+    loop {
+        let remaining = admission
+            .arbiter
+            .snapshot(&admission.resource)
+            .unwrap()
+            .remaining;
+        if remaining == 0 {
+            break;
+        }
+        pressure.push(
+            source
+                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
+                .unwrap(),
+        );
+    }
+    assert!(
+        operators
+            .execute_attention_projection_norm_slice(
+                3,
+                input(source.source_identity(), &bytes),
+                None
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), 128 * 1024 * 1024);
+    drop(pressure);
+    assert_eq!(reserved(&admission), before);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -116,6 +116,83 @@
         ));
     }
     Ok((embedding, normalized, projected))
+}
+
+/// Pure prefix of the existing attention role schedule, before rotary/state.
+pub(crate) fn attention_projection_norm_slice<B: MathematicalBackend>(
+    model: &impl OperatorSliceSource,
+    math: &mut B,
+    layer: usize,
+    input: &B::Value,
+) -> Result<crate::operator_slice::attention::AttentionProjectionValues<B::Value>> {
+    use crate::accelerator::program::ActivationPrecision;
+    use crate::operator_slice::attention::{AttentionProjectionValues, attention_roles};
+    let roles = attention_roles(model, layer)?;
+    if math.elements(input) != roles.hidden
+        || math.precision(input) != ActivationPrecision::Bf16WidenedF32
+    {
+        return Err(Error::Malformed(
+            "attention slice typed full hidden input".into(),
+        ));
+    }
+    let query_gate_projection =
+        math.project_profiled(roles.query, input, model.projection_contract())?;
+    let key_projection = math.project_profiled(roles.key, input, model.projection_contract())?;
+    let value_projection =
+        math.project_profiled(roles.value, input, model.projection_contract())?;
+    let query_view = math.view(
+        &query_gate_projection,
+        TensorView {
+            offset: 0,
+            rows: roles.query_heads,
+            width: roles.width,
+            row_stride: roles.width * 2,
+        },
+    )?;
+    let query = math.contiguous(&query_view)?;
+    let gate_view = math.view(
+        &query_gate_projection,
+        TensorView {
+            offset: roles.width,
+            rows: roles.query_heads,
+            width: roles.width,
+            row_stride: roles.width * 2,
+        },
+    )?;
+    let gate = math.contiguous(&gate_view)?;
+    let normalized_query = math.ordinary_rms_profiled(
+        roles.query_norm,
+        &query,
+        OrdinaryRmsContract::TorchCpuBf16Avx2,
+    )?;
+    let normalized_key = math.ordinary_rms_profiled(
+        roles.key_norm,
+        &key_projection,
+        OrdinaryRmsContract::TorchCpuBf16Avx2,
+    )?;
+    if [
+        &query_gate_projection,
+        &key_projection,
+        &value_projection,
+        &query,
+        &gate,
+        &normalized_query,
+        &normalized_key,
+    ]
+    .iter()
+    .any(|value| math.precision(value) != ActivationPrecision::Bf16WidenedF32)
+    {
+        return Err(Error::State("attention slice output precision".into()));
+    }
+    Ok(AttentionProjectionValues {
+        query_gate_projection,
+        key_projection,
+        value_projection,
+        query,
+        gate,
+        normalized_query,
+        normalized_key,
+    })
 }
 
 fn expect_elements<B: MathematicalBackend>(
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -232,6 +232,60 @@
             }
         }
         Ok(Some(Arc::new(source.workspace()?)))
+    }
+    /// Explicit operator input transport, never a model producer or root.
+    pub(super) fn import_attention_input(
+        &self,
+        input: crate::operator_slice::attention::AttentionSliceInput<'_>,
+        hidden: usize,
+    ) -> Result<HostValue> {
+        input.validate()?;
+        if self.region_kernel.is_some() {
+            return Err(Error::Unsupported(
+                "typed attention CPU input provider".into(),
+            ));
+        }
+        if input.source_identity() != self.source.source_identity()
+            || input.numerical_profile() != self.source.program_numerical_profile()
+            || self.source.metadata().geometry().hidden_size != hidden as u64
+            || input.bytes().len()
+                != hidden
+                    .checked_mul(2)
+                    .ok_or_else(|| Error::Limit("attention private input width".into()))?
+        {
+            return Err(Error::State(
+                "attention input source/profile/full hidden width".into(),
+            ));
+        }
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let workspace = self.prepare_output(None)?;
+        let mut values = Vec::new();
+        values
+            .try_reserve_exact(hidden)
+            .map_err(|_| Error::Limit("private attention input transport".into()))?;
+        values.extend(
+            input.bytes().chunks_exact(2).map(|word| {
+                f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16)
+            }),
+        );
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        if self
+            .cancelled
+            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
+        {
+            return Err(Error::State(
+                "attention input cancelled after transport".into(),
+            ));
+        }
+        let output = HostValue::typed(
+            values,
+            input.precision(),
+            input.source_identity(),
+            input.numerical_profile(),
+        )?;
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let output = output.retain_workspace(workspace)?;
+        Ok(output)
     }
     fn engine(&self) -> Result<&'engine QwenDenseCpu> {
         self.engine
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -8,6 +8,7 @@
 
 /// Pure role metadata only; never a layer/token/session execution capability.
 pub(crate) trait OperatorSliceSource {
+    fn geometry(&self) -> &crate::model::ModelGeometry;
     fn source_identity(&self) -> crate::accelerator::SourceIdentity;
     fn program_numerical_profile(&self) -> ProgramNumericalProfile;
     fn rms_epsilon(&self) -> f32;
@@ -23,6 +24,9 @@
     fn projection_contract(&self) -> ProjectionContract;
 }
 impl OperatorSliceSource for LoadedQwenModel {
+    fn geometry(&self) -> &crate::model::ModelGeometry {
+        LoadedQwenModel::geometry(self)
+    }
     fn source_identity(&self) -> crate::accelerator::SourceIdentity {
         LoadedQwenModel::source_identity(self)
     }
@@ -223,3 +227,7 @@
 ))]
 #[path = "operator_slice_tests.rs"]
 mod tests;
+
+/// Pure attention projections/QK norms; no rotary/cache/session admission.
+#[path = "attention_slice.rs"]
+pub mod attention;
--- a/crates/zerogravity-model-qwen35/src/operator_slice_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_tests.rs
@@ -87,6 +87,15 @@
     cfg["text_config"]["hidden_size"] = json!(hidden);
     cfg["text_config"]["vocab_size"] = json!(vocabulary);
     let cfg = serde_json::to_vec(&cfg).unwrap();
+    source_with_config(fixture, cfg, norm_bf16, |_, _, bits| bits, mutate)
+}
+fn source_with_config(
+    fixture: &Fixture,
+    cfg: Vec<u8>,
+    norm_bf16: bool,
+    bf16_word: impl Fn(&str, usize, u16) -> u16,
+    mutate: impl FnOnce(&mut Value),
+) -> Loaded {
     let specs = QwenTextConfig::from_json(&cfg)
         .unwrap()
         .tensor_specs()
@@ -103,7 +112,7 @@
             "BF16"
         };
         let start = payload.len();
-        for _ in 0..spec.shape.iter().product::<u64>() {
+        for element in 0..spec.shape.iter().product::<u64>() {
             match kind {
                 "F32" => payload.extend_from_slice(&0.25f32.to_le_bytes()),
                 "F16" => payload.extend_from_slice(&0x3400u16.to_le_bytes()),
@@ -113,7 +122,9 @@
                     } else {
                         0x3f80
                     };
-                    payload.extend_from_slice(&bits.to_le_bytes());
+                    payload.extend_from_slice(
+                        &bf16_word(&spec.name, element as usize, bits).to_le_bytes(),
+                    );
                 }
                 _ => unreachable!(),
             }
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -376,3 +376,6 @@
     drop(source);
     assert_eq!(reserved(&admission), 0);
 }
+
+#[path = "attention_slice_tests.rs"]
+mod attention;
--- a/crates/zerogravity-model-qwen35/src/original_binding/window.rs
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window.rs
@@ -194,7 +194,34 @@
         cancelled: Option<&AtomicBool>,
     ) -> Result<Vec<f32>> {
         if *parameter != self.parameter_descriptor(QwenParameterRole::InputNorm(0))? {
-            return Err(Error::State("window ordinary RMS original role".into()));
+            let mut valid = false;
+            for (layer, kind) in self.geometry().layers.iter().enumerate() {
+                if *kind != crate::model::LayerKind::FullAttention {
+                    continue;
+                }
+                if *parameter != self.parameter_descriptor(QwenParameterRole::QueryNorm(layer))?
+                    && *parameter != self.parameter_descriptor(QwenParameterRole::KeyNorm(layer))?
+                {
+                    continue;
+                }
+                let roles = crate::operator_slice::attention::attention_roles(self, layer)?;
+                let elements = if *parameter == roles.query_norm {
+                    Some(roles.query_heads * roles.width)
+                } else if *parameter == roles.key_norm {
+                    Some(roles.kv_heads * roles.width)
+                } else {
+                    None
+                };
+                if elements == Some(input.len()) {
+                    valid = true;
+                    break;
+                }
+            }
+            if !valid {
+                return Err(Error::State(
+                    "window ordinary RMS original role/heads".into(),
+                ));
+            }
         }
         check_cancel(cancelled)?;
         let raw = self.source().parameter(&parameter.name)?;
@@ -234,9 +261,31 @@
     ) -> Result<Vec<f32>> {
         self.validate_matrix(tensor)?;
         if *tensor != self.matrix_descriptor(QwenMatrixRole::GdnQkv(0))? {
-            return Err(Error::Unsupported(
-                "closed original window projection role".into(),
-            ));
+            let mut valid = false;
+            for (layer, kind) in self.geometry().layers.iter().enumerate() {
+                if *kind != crate::model::LayerKind::FullAttention {
+                    continue;
+                }
+                if ![
+                    self.matrix_descriptor(QwenMatrixRole::Query(layer))?,
+                    self.matrix_descriptor(QwenMatrixRole::Key(layer))?,
+                    self.matrix_descriptor(QwenMatrixRole::Value(layer))?,
+                ]
+                .contains(tensor)
+                {
+                    continue;
+                }
+                let roles = crate::operator_slice::attention::attention_roles(self, layer)?;
+                if [&roles.query, &roles.key, &roles.value].contains(&tensor) {
+                    valid = true;
+                    break;
+                }
+            }
+            if !valid {
+                return Err(Error::Unsupported(
+                    "closed original window projection role".into(),
+                ));
+            }
         }
         check_cancel(cancelled)?;
         if input.len() != tensor.columns {
@@ -285,6 +334,9 @@
     }
 }
 impl OperatorSliceSource for OriginalWindowOperators {
+    fn geometry(&self) -> &crate::model::ModelGeometry {
+        self.binding.config.geometry()
+    }
     fn source_identity(&self) -> crate::accelerator::SourceIdentity {
         self.checked.source_identity()
     }

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/GPU_REFUSAL.patch

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

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/NEW_DELTA.patch

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

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/WIRING.patch

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

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/GPU_REFUSAL.patch

Original UTF-8 bytes: 1932; SHA256 `23c1780ec2dfd0245f882fc8da99e8ef8901a06ef00aa21be171aff8da952ee7`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -349,6 +349,8 @@
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
+                        | PlanOperation::TypedGdnAlphaPreparation { .. }
+                        | PlanOperation::TypedGdnBetaPreparation { .. }
                 )
             })
         {
@@ -1431,7 +1433,9 @@
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
-                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedCausalConvSilu { .. }
+                | PlanOperation::TypedGdnAlphaPreparation { .. }
+                | PlanOperation::TypedGdnBetaPreparation { .. } => Err(Error::Unsupported(
                     "resident typed operation has no qualified lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -839,7 +839,9 @@
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
-                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedCausalConvSilu { .. }
+                | PlanOperation::TypedGdnAlphaPreparation { .. }
+                | PlanOperation::TypedGdnBetaPreparation { .. } => Err(Error::Unsupported(
                     "resident batch has no qualified typed lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/NEW_DELTA.patch

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

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/ROOT_VALIDATOR_V3.review.patch

Original UTF-8 bytes: 1134; SHA256 `4bff7f639d83cb14212dae4c1c9f3f0fccabda0b49fe1010e6b9dbee75513692`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- original/validate_reference.py
+++ ROOT/root_validate_reference_v3.py
@@ -10,6 +10,7 @@
     require(document['schema']=='zerogravity.original-gdn-preparation.inputs.v1' and document['profile']==PROFILE and document['bindings_sha256']==sha(HERE/'bindings.json') and document['no_arithmetic_or_hidden_reconstruction']is True and set(document['models'])=={'0.8b','2b'},'authentic input manifest closure')
     for label,item in binding['models'].items():
         model=document['models'][label];data,roles,source=source_parts(item)
+        source=json.loads(json.dumps(source))  # FD tuples cross the actual JSON transport as lists.
         require(model['repo']==item['repo'] and model['revision']==item['revision'] and model['hidden']==item['hidden'] and model['heads']==16 and model['roles']==roles and model['source_before']==model['source_after']==source,'original full-source/role/FD proof')
         for name,(raw,dtype,shape) in data.items():
             row=model['parameters'][name];require(row['dtype']==dtype and row['shape']==shape and raw_file(path.parent,row)==raw,'source-selected complete parameter raw bytes')

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/WIRING.patch

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

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/frozen/GPU_REFUSAL.patch

Original UTF-8 bytes: 1932; SHA256 `23c1780ec2dfd0245f882fc8da99e8ef8901a06ef00aa21be171aff8da952ee7`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -349,6 +349,8 @@
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
+                        | PlanOperation::TypedGdnAlphaPreparation { .. }
+                        | PlanOperation::TypedGdnBetaPreparation { .. }
                 )
             })
         {
@@ -1431,7 +1433,9 @@
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
-                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedCausalConvSilu { .. }
+                | PlanOperation::TypedGdnAlphaPreparation { .. }
+                | PlanOperation::TypedGdnBetaPreparation { .. } => Err(Error::Unsupported(
                     "resident typed operation has no qualified lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -839,7 +839,9 @@
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
-                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedCausalConvSilu { .. }
+                | PlanOperation::TypedGdnAlphaPreparation { .. }
+                | PlanOperation::TypedGdnBetaPreparation { .. } => Err(Error::Unsupported(
                     "resident batch has no qualified typed lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {

````

