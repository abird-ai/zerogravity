# Draft Patches — volume 17

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 41922; SHA256 `d1e4898e2a39722d1d1e0cb68e51c0d66c48026b1436fad09c0f5b8fad269482`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -50,6 +50,67 @@
         input,
         model.projection_contract(),
     )
+}
+
+/// One typed original projection, with the caller's existing value producer.
+/// No layer/token traversal or state/publication is introduced.
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+pub(crate) fn pure_projection_slice<B: MathematicalBackend>(
+    source: &impl OperatorSliceSource,
+    math: &mut B,
+    role: QwenMatrixRole,
+    input: &B::Value,
+) -> Result<B::Value> {
+    use crate::accelerator::program::{ActivationPrecision, ProgramNumericalProfile};
+    let tensor = source.matrix_descriptor(role)?;
+    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || math.precision(input) != ActivationPrecision::Bf16WidenedF32
+        || math.elements(input) != tensor.columns
+    {
+        return Err(Error::State(
+            "pure projection source profile/input geometry".into(),
+        ));
+    }
+    let rows = tensor.rows;
+    let output = math.project_profiled(tensor, input, source.projection_contract())?;
+    if math.elements(&output) != rows
+        || math.precision(&output) != ActivationPrecision::Bf16WidenedF32
+    {
+        return Err(Error::State("pure projection complete typed output".into()));
+    }
+    Ok(output)
+}
+
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+pub(crate) fn pure_ordinary_rms_slice<B: MathematicalBackend>(
+    source: &impl OperatorSliceSource,
+    math: &mut B,
+    role: QwenParameterRole,
+    input: &B::Value,
+) -> Result<B::Value> {
+    use crate::accelerator::program::{ActivationPrecision, ProgramNumericalProfile};
+    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || source.rms_epsilon().to_bits() != 1e-6f32.to_bits()
+        || math.precision(input) != ActivationPrecision::Bf16WidenedF32
+    {
+        return Err(Error::State(
+            "pure ordinary RMS source profile/precision".into(),
+        ));
+    }
+    let elements = math.elements(input);
+    let output = math.ordinary_rms_profiled(
+        source.parameter_descriptor(role)?,
+        input,
+        OrdinaryRmsContract::TorchCpuBf16Avx2,
+    )?;
+    if math.elements(&output) != elements
+        || math.precision(&output) != ActivationPrecision::Bf16WidenedF32
+    {
+        return Err(Error::State(
+            "pure ordinary RMS complete typed output".into(),
+        ));
+    }
+    Ok(output)
 }
 
 /// Pure fragment of this same source-role schedule. It never touches state,
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -209,7 +209,12 @@
         }
     }
     #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
-    fn prepare_output(&self, input: Option<&HostValue>) -> Result<Option<Arc<WindowWorkspace>>> {
+    fn prepare_output(
+        &self,
+        input: Option<&HostValue>,
+        output_elements: usize,
+        scratch_elements: usize,
+    ) -> Result<Option<Arc<WindowWorkspace>>> {
         let HostSource::Windows(source) = &self.source else {
             return Ok(None);
         };
@@ -231,7 +236,9 @@
                 ));
             }
         }
-        Ok(Some(Arc::new(source.workspace()?)))
+        Ok(Some(Arc::new(
+            source.workspace(output_elements, scratch_elements)?,
+        )))
     }
     /// Explicit operator input transport, never a model producer or root.
     pub(super) fn import_attention_input(
@@ -258,7 +265,7 @@
             ));
         }
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
-        let workspace = self.prepare_output(None)?;
+        let workspace = self.prepare_output(None, hidden, 1)?;
         let mut values = Vec::new();
         values
             .try_reserve_exact(hidden)
@@ -286,6 +293,52 @@
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         let output = output.retain_workspace(workspace)?;
         Ok(output)
+    }
+    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+    pub(crate) fn import_pure_input(
+        &self,
+        input: crate::operator_slice::pure::PureSliceInput<'_>,
+        elements: usize,
+    ) -> Result<HostValue> {
+        input.validate()?;
+        if self.region_kernel.is_some()
+            || input.source_identity() != self.source.source_identity()
+            || input.numerical_profile() != self.source.program_numerical_profile()
+            || input.bytes().len()
+                != elements
+                    .checked_mul(2)
+                    .ok_or_else(|| Error::Limit("pure operator input bytes".into()))?
+        {
+            return Err(Error::State(
+                "pure operator input provider/source/profile/width".into(),
+            ));
+        }
+        let workspace = self.prepare_output(None, elements, 1)?;
+        let mut values = Vec::new();
+        values
+            .try_reserve_exact(elements)
+            .map_err(|_| Error::Limit("private pure input transport".into()))?;
+        if values.capacity() != elements {
+            return Err(Error::Limit("pure input exceeds admitted capacity".into()));
+        }
+        values.extend(
+            input.bytes().chunks_exact(2).map(|word| {
+                f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16)
+            }),
+        );
+        if self
+            .cancelled
+            .is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire))
+        {
+            return Err(Error::State("pure operator input cancelled".into()));
+        }
+        HostValue::typed(
+            values,
+            ActivationPrecision::Bf16WidenedF32,
+            input.source_identity(),
+            input.numerical_profile(),
+        )?
+        .retain_workspace(workspace)
     }
     fn engine(&self) -> Result<&'engine QwenDenseCpu> {
         self.engine
@@ -379,7 +432,7 @@
         let token = u32::try_from(token)
             .map_err(|_| Error::Malformed("typed embedding token domain".into()))?;
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
-        let workspace = self.prepare_output(None)?;
+        let workspace = self.prepare_output(None, tensor.columns, 1)?;
         let values = match &self.source {
             HostSource::Loaded(source) => {
                 crate::weights::source_reference_embedding(source.embedding_source(token)?)?
@@ -424,8 +477,27 @@
                 "host typed projection source/contract".into(),
             ));
         }
-        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
-        let workspace = self.prepare_output(Some(input))?;
+        // Full typed input and role checks precede physical output admission.
+        let activation = input.as_slice()?;
+        ActivationPrecision::Bf16WidenedF32.validate_values(activation)?;
+        if activation.len() != tensor.columns {
+            return Err(Error::Malformed(
+                "typed projection complete input width".into(),
+            ));
+        }
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        if let HostSource::Windows(source) = &self.source {
+            source.validate_projection_matrix(&tensor)?;
+        }
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let workspace = self.prepare_output(
+            Some(input),
+            tensor.rows,
+            tensor
+                .rows
+                .checked_add(128)
+                .ok_or_else(|| Error::Limit("projection scratch element count".into()))?,
+        )?;
         let values = match &self.source {
             HostSource::Loaded(source) => {
                 zerogravity_cpu::original_hf::output_first_projection_with_profile(
@@ -442,10 +514,26 @@
                 source.project(&tensor, input.as_slice()?, self.cancelled)?
             }
         };
-        let values = values
-            .into_iter()
-            .map(round_bf16)
-            .collect::<Result<Vec<_>>>()?;
+        // Do not rely on Result<Vec<_>> collection capacity for wide outputs.
+        // The original row order and RNE helper are unchanged.
+        if values.len() != tensor.rows {
+            return Err(Error::State(
+                "complete typed projection output length".into(),
+            ));
+        }
+        let mut rounded = Vec::new();
+        rounded
+            .try_reserve_exact(tensor.rows)
+            .map_err(|_| Error::Limit("complete typed BF16 output allocation".into()))?;
+        if rounded.capacity() != tensor.rows {
+            return Err(Error::Limit(
+                "typed BF16 output exceeds admitted capacity".into(),
+            ));
+        }
+        for value in values {
+            rounded.push(round_bf16(value)?);
+        }
+        let values = rounded;
         let output = HostValue::typed(
             values,
             ActivationPrecision::Bf16WidenedF32,
@@ -479,8 +567,27 @@
             ));
         }
         let width = contract.validate(&parameter, input.elements)?;
-        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
-        let workspace = self.prepare_output(Some(input))?;
+        let activation = input.as_slice()?;
+        ActivationPrecision::Bf16WidenedF32.validate_values(activation)?;
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        if let HostSource::Windows(source) = &self.source {
+            source.validate_ordinary_parameter(&parameter, input.elements)?;
+        }
+        // Closed diagnostic keeps5 full vectors,2 weight vectors and4 row
+        // vectors while its final output is copied into the retained backing.
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let scratch = input
+            .elements
+            .checked_mul(5)
+            .and_then(|n| width.checked_mul(2).and_then(|w| n.checked_add(w)))
+            .and_then(|n| {
+                (input.elements / width)
+                    .checked_mul(4)
+                    .and_then(|r| n.checked_add(r))
+            })
+            .ok_or_else(|| Error::Limit("ordinary RMS scratch element count".into()))?;
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        let workspace = self.prepare_output(Some(input), input.elements, scratch)?;
         let values = match &self.source {
             HostSource::Loaded(source) => {
                 parameter.validate_source(*source)?;
@@ -572,7 +679,7 @@
             return Ok(input.clone());
         }
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
-        let workspace = self.prepare_output(Some(input))?;
+        let workspace = self.prepare_output(Some(input), input.elements, 1)?;
         let mut values = Vec::with_capacity(input.elements);
         for row in 0..input.view.rows {
             let offset = input.view.offset + row * input.view.row_stride;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/operator_slice/pure.rs
@@ -0,0 +1,163 @@
+//! One pure projection or ordinary norm on explicit private BF16 input.
+//! No producer, session, activation nonlinearities or model execution is asserted.
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{ActivationPrecision, ProgramNumericalProfile};
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+use crate::accelerator::program::{PlanBuilder, PlanValue};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, Result};
+
+/// External raw transport only. Caller independently binds archive/source and
+/// producer provenance; this type validates representation and source/profile.
+#[derive(Clone, Copy)]
+pub struct PureSliceInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    bytes: &'a [u8],
+}
+impl<'a> PureSliceInput<'a> {
+    pub fn from_bf16_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        bytes: &'a [u8],
+    ) -> Result<Self> {
+        let input = Self {
+            source,
+            profile,
+            bytes,
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
+            return Err(Error::State("pure input source/profile".into()));
+        }
+        if self.bytes.is_empty()
+            || self.bytes.len() > 2 * 17408
+            || !self.bytes.len().is_multiple_of(2)
+            || self
+                .bytes
+                .chunks_exact(2)
+                .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
+        {
+            return Err(Error::Malformed(
+                "pure input bounded finite BF16 bytes".into(),
+            ));
+        }
+        Ok(())
+    }
+}
+
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum PureSliceRole {
+    Projection(QwenMatrixRole),
+    OrdinaryRms(QwenParameterRole),
+}
+
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    /// Caller retains its existing same-builder producer and source/metadata.
+    /// No external-input node, root or complete model plan is manufactured.
+    pub fn append_projection_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        role: QwenMatrixRole,
+        input: &PlanValue,
+    ) -> Result<PlanValue> {
+        self.projection_role(role)?;
+        crate::forward::pure_projection_slice(self, builder, role, input)
+    }
+    pub fn append_ordinary_rms_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        role: QwenParameterRole,
+        input: &PlanValue,
+    ) -> Result<PlanValue> {
+        self.ordinary_hidden_role(role)?;
+        crate::forward::pure_ordinary_rms_slice(self, builder, role, input)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_projection_slice(
+        &self,
+        role: QwenMatrixRole,
+        input: PureSliceInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<PureSliceOutput> {
+        let matrix = self.projection_role(role)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let input = math.import_pure_input(input, matrix.columns)?;
+        let output = crate::forward::pure_projection_slice(self, &mut math, role, &input)?;
+        Ok(PureSliceOutput {
+            source: self.source().source_identity(),
+            role: PureSliceRole::Projection(role),
+            input,
+            output,
+        })
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_ordinary_rms_slice(
+        &self,
+        role: QwenParameterRole,
+        input: PureSliceInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<PureSliceOutput> {
+        use crate::operator_slice::OperatorSliceSource;
+        self.ordinary_hidden_role(role)?;
+        let hidden = usize::try_from(self.geometry().hidden_size)
+            .map_err(|_| Error::Limit("ordinary pure input width".into()))?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let input = math.import_pure_input(input, hidden)?;
+        let output = crate::forward::pure_ordinary_rms_slice(self, &mut math, role, &input)?;
+        Ok(PureSliceOutput {
+            source: self.source().source_identity(),
+            role: PureSliceRole::OrdinaryRms(role),
+            input,
+            output,
+        })
+    }
+}
+
+/// Retained original-source input/output backing, borrowed access only.
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+pub struct PureSliceOutput {
+    source: SourceIdentity,
+    role: PureSliceRole,
+    input: crate::host_math::HostValue,
+    output: crate::host_math::HostValue,
+}
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+impl PureSliceOutput {
+    pub fn role(&self) -> PureSliceRole {
+        self.role
+    }
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
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
+    pub fn output(&self) -> Result<&[f32]> {
+        self.output.as_slice()
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -234,3 +234,7 @@
 
 #[path = "gdn_conv_slice.rs"]
 pub mod gdn_conv;
+
+/// Source-role pure projection/ordinary-norm fragments; no complete HF admission.
+#[path = "operator_slice/pure.rs"]
+pub mod pure;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs
@@ -0,0 +1,367 @@
+//! Source-free full-header fixtures; no trained27B/operator independence claim.
+use super::*;
+use crate::accelerator::program::ProgramNumericalProfile;
+use crate::operator_slice::pure::PureSliceInput;
+use zerogravity_core::resource_arbiter::AllocationClass;
+
+fn input<'a>(source: &OriginalWindowOperators, bytes: &'a [u8]) -> PureSliceInput<'a> {
+    PureSliceInput::from_bf16_bytes(
+        source.source().source_identity(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        bytes,
+    )
+    .unwrap()
+}
+
+#[test]
+fn all_bound_projection_roles_use_complete_actual_source_and_retained_results() {
+    for tied in [false, true] {
+        let fixture = Fixture::new();
+        let loaded = source_with_options(&fixture, tied, 256, true, |_| {});
+        let (source, admission) = windows(&fixture, &loaded);
+        let operators = bind_windows(source.clone(), &loaded);
+        for role in [
+            QwenMatrixRole::GdnQkv(0),
+            QwenMatrixRole::GdnGate(0),
+            QwenMatrixRole::GdnAlpha(0),
+            QwenMatrixRole::GdnBeta(0),
+            QwenMatrixRole::GdnOutput(0),
+            QwenMatrixRole::Query(3),
+            QwenMatrixRole::Key(3),
+            QwenMatrixRole::Value(3),
+            QwenMatrixRole::AttentionOutput(3),
+            QwenMatrixRole::FfnGate(0),
+            QwenMatrixRole::FfnUp(0),
+            QwenMatrixRole::FfnDown(0),
+            QwenMatrixRole::Output,
+        ] {
+            let tensor = operators.matrix_descriptor(role).unwrap();
+            let bytes = 0x3f80u16.to_le_bytes().repeat(tensor.columns);
+            let before = reserved(&admission);
+            let result = operators
+                .execute_projection_slice(role, input(&operators, &bytes), None)
+                .unwrap();
+            let sign = if role == QwenMatrixRole::Output && !tied {
+                -1.0
+            } else {
+                1.0
+            };
+            assert_eq!(result.input().unwrap(), vec![1.0; tensor.columns]);
+            assert_eq!(
+                result.output().unwrap(),
+                vec![sign * tensor.columns as f32; tensor.rows]
+            );
+            assert_eq!(result.source_identity(), source.source_identity());
+            assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
+            drop(result);
+            assert_eq!(reserved(&admission), before);
+        }
+        drop(operators);
+        drop(source);
+        assert_eq!(reserved(&admission), 0);
+    }
+}
+
+#[test]
+fn ordinary_hidden_roles_preserve_raw_gamma_and_complete_source_membership() {
+    let fixture = Fixture::new();
+    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
+    for role in [
+        QwenParameterRole::InputNorm(0),
+        QwenParameterRole::InputNorm(2),
+        QwenParameterRole::FfnNorm(0),
+        QwenParameterRole::FinalNorm,
+    ] {
+        let result = operators
+            .execute_ordinary_rms_slice(role, input(&operators, &bytes), None)
+            .unwrap();
+        let expected = if matches!(role, QwenParameterRole::InputNorm(_)) {
+            2.0
+        } else {
+            1.25
+        };
+        assert_eq!(result.output().unwrap(), vec![expected; 256]);
+    }
+    let before = reserved(&admission);
+    for role in [
+        QwenParameterRole::GdnNorm(0),
+        QwenParameterRole::GdnA(0),
+        QwenParameterRole::GdnDt(0),
+        QwenParameterRole::InputNorm(99),
+    ] {
+        assert!(
+            operators
+                .execute_ordinary_rms_slice(role, input(&operators, &bytes), None)
+                .is_err()
+        );
+        assert_eq!(reserved(&admission), before);
+    }
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+
+#[test]
+fn same_plan_projection_and_norm_nodes_do_not_create_state_or_external_producers() {
+    let fixture = Fixture::new();
+    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
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
+    let normalized = operators
+        .append_ordinary_rms_slice(&mut builder, QwenParameterRole::FfnNorm(0), &embedding)
+        .unwrap();
+    let up = operators
+        .append_projection_slice(&mut builder, QwenMatrixRole::FfnUp(0), &normalized)
+        .unwrap();
+    let down = operators
+        .append_projection_slice(&mut builder, QwenMatrixRole::FfnDown(0), &up)
+        .unwrap();
+    let head = operators
+        .append_projection_slice(&mut builder, QwenMatrixRole::Output, &down)
+        .unwrap();
+    let plan = builder.finish_operator_slice(head).unwrap();
+    assert_eq!(plan.plan().nodes().len(), 5);
+    assert!(plan.plan().proposed_domains().is_empty());
+    assert!(matches!(
+        plan.plan().nodes()[4].operation,
+        PlanOperation::TypedProject { .. }
+    ));
+    drop(operators);
+    drop(source);
+    drop(plan);
+    assert_eq!(reserved(&admission), 0);
+}
+
+#[test]
+fn checked_actual_vocabulary_output_and_wide_input_charge_grow_before_allocation() {
+    let fixture = Fixture::new();
+    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let before = admission.arbiter.snapshot(&admission.resource).unwrap();
+    let head = operators.workspace(248320, 248448).unwrap();
+    let current = admission.arbiter.snapshot(&admission.resource).unwrap();
+    assert_eq!(
+        current.per_class[&AllocationClass::Output]
+            - before
+                .per_class
+                .get(&AllocationClass::Output)
+                .copied()
+                .unwrap_or(0),
+        248320 * 4
+    );
+    assert_eq!(head.bytes(), (248320 + 248448) * 4);
+    drop(head);
+    let wide = operators.workspace(17408, 1).unwrap();
+    assert_eq!(wide.bytes(), 17408 * 4 + 960 * 1024);
+    drop(wide);
+    assert!(operators.workspace(usize::MAX, 1).is_err());
+    assert!(operators.workspace(1, usize::MAX).is_err());
+    assert_eq!(reserved(&admission), before.reserved);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+
+#[test]
+fn invalid_input_role_capacity_and_cancel_refuse_without_leaking_or_fencing() {
+    let fixture = Fixture::new();
+    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
+    let before = reserved(&admission);
+    for role in [
+        QwenMatrixRole::Embedding,
+        QwenMatrixRole::GdnGate(3),
+        QwenMatrixRole::Query(0),
+    ] {
+        assert!(
+            operators
+                .execute_projection_slice(role, input(&operators, &bytes), None)
+                .is_err()
+        );
+    }
+    assert!(
+        operators
+            .execute_projection_slice(
+                QwenMatrixRole::FfnUp(0),
+                input(&operators, &bytes[..2]),
+                None
+            )
+            .is_err()
+    );
+    assert!(
+        operators
+            .execute_projection_slice(
+                QwenMatrixRole::FfnUp(0),
+                input(&operators, &bytes),
+                Some(&AtomicBool::new(true))
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), before);
+    for bad in [
+        vec![0],
+        0x7fc1u16.to_le_bytes().to_vec(),
+        vec![0; 2 * 17409],
+    ] {
+        assert!(
+            PureSliceInput::from_bf16_bytes(
+                source.source_identity(),
+                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+                &bad
+            )
+            .is_err()
+        );
+    }
+    let pressure = source.reserve_workspace(32 * 1024 * 1024).unwrap();
+    let pressure2 = source.reserve_workspace(32 * 1024 * 1024).unwrap();
+    let remaining = admission
+        .arbiter
+        .snapshot(&admission.resource)
+        .unwrap()
+        .remaining;
+    let pressure3 = source
+        .reserve_workspace((remaining - (2 * 1024 * 1024 - 1)) as usize)
+        .unwrap();
+    let held = reserved(&admission);
+    assert!(
+        operators
+            .execute_projection_slice(QwenMatrixRole::Output, input(&operators, &bytes), None)
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), held);
+    assert!(
+        !admission
+            .arbiter
+            .snapshot(&admission.resource)
+            .unwrap()
+            .fenced
+    );
+    drop(pressure3);
+    drop(pressure2);
+    drop(pressure);
+    assert_eq!(reserved(&admission), before);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+
+#[test]
+fn returned_borrowed_results_keep_source_and_original_tickets_after_callers_drop() {
+    let fixture = Fixture::new();
+    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
+    let source_metadata = source.metadata_reserved_bytes();
+    let result = operators
+        .execute_projection_slice(QwenMatrixRole::Output, input(&operators, &bytes), None)
+        .unwrap();
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), source_metadata + 2 * 1024 * 1024);
+    assert_eq!(result.output().unwrap(), vec![-256.0; 16]);
+    assert_eq!(result.input().unwrap(), vec![1.0; 256]);
+    drop(result);
+    assert_eq!(reserved(&admission), 0);
+}
+
+#[test]
+fn unaware_provider_refuses_new_roles_before_any_callback_or_admission() {
+    struct Unaware(AtomicBool);
+    impl crate::accelerator::MatVecKernel for Unaware {
+        fn try_execute(
+            &self,
+            _: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
+            panic!("pure original CPU role cannot call an unaware provider");
+        }
+    }
+    let fixture = Fixture::new();
+    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
+    let provider = Unaware(AtomicBool::new(false));
+    let plain = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
+    let activation = plain
+        .import_pure_input(input(&operators, &bytes), 256)
+        .unwrap();
+    let before = reserved(&admission);
+    let mut unaware =
+        crate::host_math::HostMath::for_window_operator_slice(&operators, None, Some(&provider));
+    assert!(
+        unaware
+            .project_profiled(
+                operators.matrix_descriptor(QwenMatrixRole::Output).unwrap(),
+                &activation,
+                crate::accelerator::program::ProjectionContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+    );
+    assert!(!provider.0.load(std::sync::atomic::Ordering::Relaxed));
+    assert_eq!(reserved(&admission), before);
+    drop(activation);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+
+#[test]
+fn same_content_foreign_window_issuer_cannot_rebind_a_retained_input() {
+    let fixture = Fixture::new();
+    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
+    let (source_a, admission_a) = windows(&fixture, &loaded);
+    let (source_b, admission_b) = windows(&fixture, &loaded);
+    assert_eq!(source_a.source_identity(), source_b.source_identity());
+    assert!(!source_a.same_issuer(&source_b));
+    let operators_a = bind_windows(source_a.clone(), &loaded);
+    let operators_b = bind_windows(source_b.clone(), &loaded);
+    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
+    let math_a = crate::host_math::HostMath::for_window_operator_slice(&operators_a, None, None);
+    let activation = math_a
+        .import_pure_input(input(&operators_a, &bytes), 256)
+        .unwrap();
+    let before_b = reserved(&admission_b);
+    let mut math_b =
+        crate::host_math::HostMath::for_window_operator_slice(&operators_b, None, None);
+    assert!(
+        math_b
+            .project_profiled(
+                operators_b
+                    .matrix_descriptor(QwenMatrixRole::Output)
+                    .unwrap(),
+                &activation,
+                crate::accelerator::program::ProjectionContract::TorchCpuBf16Avx2
+            )
+            .is_err()
+    );
+    assert_eq!(reserved(&admission_b), before_b);
+    drop(activation);
+    drop(operators_a);
+    drop(operators_b);
+    drop(source_a);
+    drop(source_b);
+    assert_eq!(reserved(&admission_a), 0);
+    assert_eq!(reserved(&admission_b), 0);
+}
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -382,3 +382,6 @@
 
 #[path = "gdn_conv_window_tests.rs"]
 mod gdn_conv;
+
+#[path = "operator_slice_window_roles_tests.rs"]
+mod pure_roles;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window/pure.rs
@@ -0,0 +1,117 @@
+//! Config/header-bound roles only; source bytes and CPU arithmetic stay unchanged.
+use super::*;
+use crate::accelerator::program::OrdinaryRmsContract;
+use zerogravity_core::quant::QuantKind;
+
+const MAX_INPUT: usize = 17408;
+const MAX_OUTPUT: usize = 248320;
+
+impl OriginalWindowOperators {
+    pub(crate) fn projection_role(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
+        if role == QwenMatrixRole::Embedding {
+            return Err(Error::Unsupported(
+                "embedding is not a projection role".into(),
+            ));
+        }
+        let tensor = self.matrix_descriptor(role)?;
+        self.validate_projection_matrix(&tensor)?;
+        Ok(tensor)
+    }
+    pub(crate) fn validate_projection_matrix(&self, tensor: &SourceTensor) -> Result<()> {
+        self.validate_matrix(tensor)?;
+        // Complete config/index/header closure issued every original role.
+        // Never rename a tensor or present partial geometry as its full source.
+        if !self
+            .binding
+            .matrices
+            .iter()
+            .any(|(role, actual)| *role != QwenMatrixRole::Embedding && actual == tensor)
+            || tensor.source != self.source_identity()
+            || tensor.kind != QuantKind::Bf16
+            || tensor.axes != MatrixAxes::OutputInput
+            || tensor.columns == 0
+            || tensor.columns > MAX_INPUT
+            || tensor.rows == 0
+            || tensor.rows > MAX_OUTPUT
+        {
+            return Err(Error::Unsupported(
+                "closed original window pure matrix role/geometry".into(),
+            ));
+        }
+        tensor
+            .kind
+            .byte_len(tensor.columns as u64)?
+            .checked_mul(tensor.rows as u64)
+            .ok_or_else(|| Error::Limit("pure original full matrix bytes".into()))?;
+        Ok(())
+    }
+    pub(crate) fn ordinary_hidden_role(&self, role: QwenParameterRole) -> Result<SourceParameter> {
+        if !matches!(
+            role,
+            QwenParameterRole::InputNorm(_)
+                | QwenParameterRole::FfnNorm(_)
+                | QwenParameterRole::FinalNorm
+        ) {
+            return Err(Error::Unsupported("pure ordinary hidden norm role".into()));
+        }
+        let parameter = self.parameter_descriptor(role)?;
+        let hidden = usize::try_from(self.geometry().hidden_size)
+            .map_err(|_| Error::Limit("ordinary hidden norm width".into()))?;
+        if parameter.source != self.source_identity()
+            || OrdinaryRmsContract::TorchCpuBf16Avx2.validate(&parameter, hidden)? != hidden
+        {
+            return Err(Error::State(
+                "ordinary hidden norm full raw parameter".into(),
+            ));
+        }
+        Ok(parameter)
+    }
+    #[cfg(feature = "cpu")]
+    pub(crate) fn validate_ordinary_parameter(
+        &self,
+        parameter: &SourceParameter,
+        elements: usize,
+    ) -> Result<()> {
+        let hidden = usize::try_from(self.geometry().hidden_size)
+            .map_err(|_| Error::Limit("ordinary hidden norm width".into()))?;
+        if elements == hidden
+            && self.binding.parameters.iter().any(|(role, actual)| {
+                matches!(
+                    role,
+                    QwenParameterRole::InputNorm(_)
+                        | QwenParameterRole::FfnNorm(_)
+                        | QwenParameterRole::FinalNorm
+                ) && actual == parameter
+            })
+        {
+            if parameter.source != self.source_identity()
+                || OrdinaryRmsContract::TorchCpuBf16Avx2.validate(parameter, elements)? != hidden
+            {
+                return Err(Error::State(
+                    "ordinary hidden norm exact source/width".into(),
+                ));
+            }
+            return Ok(());
+        }
+        // Existing qualified attention Q/K heads retain their original guards.
+        for (layer, kind) in self.geometry().layers.iter().enumerate() {
+            if *kind != crate::model::LayerKind::FullAttention {
+                continue;
+            }
+            let roles = crate::operator_slice::attention::attention_roles(self, layer)?;
+            let expected = if *parameter == roles.query_norm {
+                Some(roles.query_heads * roles.width)
+            } else if *parameter == roles.key_norm {
+                Some(roles.kv_heads * roles.width)
+            } else {
+                None
+            };
+            if expected == Some(elements) {
+                return Ok(());
+            }
+        }
+        Err(Error::State(
+            "window ordinary RMS original role/heads".into(),
+        ))
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/original_binding/window.rs
+++ b/crates/zerogravity-model-qwen35/src/original_binding/window.rs
@@ -1,4 +1,4 @@
-//! Three pure original operators over one sealed source, never a model/session.
+//! Pure original operators over one sealed source, never a model/session.
 use super::*;
 use crate::accelerator::program::{
     EmbeddingContract, PlanBuilder, ProgramNumericalProfile, ProjectionContract,
@@ -121,11 +121,22 @@
         Ok(())
     }
     #[cfg(feature = "cpu")]
-    pub(crate) fn workspace(&self) -> Result<WindowWorkspace> {
-        // Covers private output, existing RMS diagnostic frame, raw/final
-        // projection arrays and one128-row accumulator. No learned cache.
-        self.source()
-            .reserve_operator_buffers(64 * 1024, 960 * 1024)
+    pub(crate) fn workspace(
+        &self,
+        output_elements: usize,
+        scratch_elements: usize,
+    ) -> Result<WindowWorkspace> {
+        // Preserve conservative old floors; grow before allocation for actual
+        // complete outputs/raw projection/RMS frame. Staged source has its own lease.
+        let bytes = |n: usize| {
+            n.checked_mul(std::mem::size_of::<f32>())
+                .filter(|&n| n > 0)
+                .ok_or_else(|| Error::Limit("window operator buffer bytes".into()))
+        };
+        self.source().reserve_operator_buffers(
+            bytes(output_elements)?.max(64 * 1024),
+            bytes(scratch_elements)?.max(960 * 1024),
+        )
     }
     #[cfg(feature = "cpu")]
     pub fn execute(
@@ -193,36 +204,7 @@
         input: &[f32],
         cancelled: Option<&AtomicBool>,
     ) -> Result<Vec<f32>> {
-        if *parameter != self.parameter_descriptor(QwenParameterRole::InputNorm(0))? {
-            let mut valid = false;
-            for (layer, kind) in self.geometry().layers.iter().enumerate() {
-                if *kind != crate::model::LayerKind::FullAttention {
-                    continue;
-                }
-                if *parameter != self.parameter_descriptor(QwenParameterRole::QueryNorm(layer))?
-                    && *parameter != self.parameter_descriptor(QwenParameterRole::KeyNorm(layer))?
-                {
-                    continue;
-                }
-                let roles = crate::operator_slice::attention::attention_roles(self, layer)?;
-                let elements = if *parameter == roles.query_norm {
-                    Some(roles.query_heads * roles.width)
-                } else if *parameter == roles.key_norm {
-                    Some(roles.kv_heads * roles.width)
-                } else {
-                    None
-                };
-                if elements == Some(input.len()) {
-                    valid = true;
-                    break;
-                }
-            }
-            if !valid {
-                return Err(Error::State(
-                    "window ordinary RMS original role/heads".into(),
-                ));
-            }
-        }
+        self.validate_ordinary_parameter(parameter, input.len())?;
         check_cancel(cancelled)?;
         let raw = self.source().parameter(&parameter.name)?;
         raw.validate_source_parameter(parameter)?;
@@ -260,33 +242,7 @@
         cancelled: Option<&AtomicBool>,
     ) -> Result<Vec<f32>> {
         self.validate_matrix(tensor)?;
-        if *tensor != self.matrix_descriptor(QwenMatrixRole::GdnQkv(0))? {
-            let mut valid = false;
-            for (layer, kind) in self.geometry().layers.iter().enumerate() {
-                if *kind != crate::model::LayerKind::FullAttention {
-                    continue;
-                }
-                if ![
-                    self.matrix_descriptor(QwenMatrixRole::Query(layer))?,
-                    self.matrix_descriptor(QwenMatrixRole::Key(layer))?,
-                    self.matrix_descriptor(QwenMatrixRole::Value(layer))?,
-                ]
-                .contains(tensor)
-                {
-                    continue;
-                }
-                let roles = crate::operator_slice::attention::attention_roles(self, layer)?;
-                if [&roles.query, &roles.key, &roles.value].contains(&tensor) {
-                    valid = true;
-                    break;
-                }
-            }
-            if !valid {
-                return Err(Error::Unsupported(
-                    "closed original window projection role".into(),
-                ));
-            }
-        }
+        self.validate_projection_matrix(tensor)?;
         check_cancel(cancelled)?;
         if input.len() != tensor.columns {
             return Err(Error::Malformed(
@@ -300,6 +256,11 @@
         output
             .try_reserve_exact(tensor.rows)
             .map_err(|_| Error::Limit("private window complete output".into()))?;
+        if output.capacity() != tensor.rows {
+            return Err(Error::Limit(
+                "window raw output exceeds admitted capacity".into(),
+            ));
+        }
         for start in (0..tensor.rows).step_by(128) {
             check_cancel(cancelled)?;
             let rows = start..(start + 128).min(tensor.rows);
@@ -418,3 +379,5 @@
 
 #[cfg(feature = "cpu")]
 mod gdn_conv;
+
+mod pure;

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/followup-v2/EXACT_HIDDEN_FOLLOWUP.patch

Original UTF-8 bytes: 2793; SHA256 `0add9cd066b69e3730ce88fbad30d8c5fed64eb108d673b734e96cdb9adb8ce1`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -98,6 +98,13 @@
         ));
     }
     let elements = math.elements(input);
+    let hidden = usize::try_from(source.geometry().hidden_size)
+        .map_err(|_| Error::Limit("pure ordinary RMS hidden width".into()))?;
+    if elements != hidden {
+        return Err(Error::Malformed(
+            "pure ordinary RMS exact hidden input".into(),
+        ));
+    }
     let output = math.ordinary_rms_profiled(
         source.parameter_descriptor(role)?,
         input,
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs
@@ -365,3 +365,53 @@
     assert_eq!(reserved(&admission_a), 0);
     assert_eq!(reserved(&admission_b), 0);
 }
+
+#[test]
+fn valid_two_hidden_qkv_producer_cannot_record_one_hidden_norm_or_partial_node() {
+    let fixture = Fixture::new();
+    let mut cfg: Value = serde_json::from_slice(&config(false)).unwrap();
+    cfg["text_config"]["hidden_size"] = json!(256);
+    cfg["text_config"]["linear_key_head_dim"] = json!(128);
+    cfg["text_config"]["linear_value_head_dim"] = json!(128);
+    let loaded = source_with_config(
+        &fixture,
+        serde_json::to_vec(&cfg).unwrap(),
+        true,
+        |_, _, bits| bits,
+        |_| {},
+    );
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let matrix = operators
+        .matrix_descriptor(QwenMatrixRole::GdnQkv(0))
+        .unwrap();
+    assert_eq!(matrix.rows, 512);
+    assert_eq!(matrix.columns, 256);
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
+    let qkv = operators
+        .append_projection_slice(&mut builder, QwenMatrixRole::GdnQkv(0), &embedding)
+        .unwrap();
+    assert!(matches!(operators.append_ordinary_rms_slice(&mut builder,
+        QwenParameterRole::InputNorm(0), &qkv),
+        Err(Error::Malformed(message)) if message.contains("exact hidden input")));
+    let recorded = builder.finish_operator_slice(qkv).unwrap();
+    assert_eq!(recorded.plan().nodes().len(), 2);
+    assert!(recorded.plan().proposed_domains().is_empty());
+    drop(recorded);
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/followup-v3/COMPLETED_CANCEL_FOLLOWUP.patch

Original UTF-8 bytes: 3770; SHA256 `25d9270bac33008e8cef0f2c46ad6985f2153d8d6d1012f7c04f3fa55988facd`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/operator_slice/pure.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice/pure.rs
@@ -102,12 +102,15 @@
         let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
         let input = math.import_pure_input(input, matrix.columns)?;
         let output = crate::forward::pure_projection_slice(self, &mut math, role, &input)?;
-        Ok(PureSliceOutput {
-            source: self.source().source_identity(),
-            role: PureSliceRole::Projection(role),
-            input,
-            output,
-        })
+        finish_pure_output(
+            PureSliceOutput {
+                source: self.source().source_identity(),
+                role: PureSliceRole::Projection(role),
+                input,
+                output,
+            },
+            cancelled,
+        )
     }
     #[cfg(feature = "cpu")]
     pub fn execute_ordinary_rms_slice(
@@ -123,12 +126,15 @@
         let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
         let input = math.import_pure_input(input, hidden)?;
         let output = crate::forward::pure_ordinary_rms_slice(self, &mut math, role, &input)?;
-        Ok(PureSliceOutput {
-            source: self.source().source_identity(),
-            role: PureSliceRole::OrdinaryRms(role),
-            input,
-            output,
-        })
+        finish_pure_output(
+            PureSliceOutput {
+                source: self.source().source_identity(),
+                role: PureSliceRole::OrdinaryRms(role),
+                input,
+                output,
+            },
+            cancelled,
+        )
     }
 }
 
@@ -161,3 +167,17 @@
         self.output.as_slice()
     }
 }
+
+/// Boundary after all synchronous math/RNE/validation, before private return.
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+pub(crate) fn finish_pure_output(
+    output: PureSliceOutput,
+    cancelled: Option<&std::sync::atomic::AtomicBool>,
+) -> Result<PureSliceOutput> {
+    if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
+        return Err(Error::State(
+            "pure window output cancelled before return".into(),
+        ));
+    }
+    Ok(output)
+}
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs
@@ -415,3 +415,34 @@
     drop(source);
     assert_eq!(reserved(&admission), 0);
 }
+
+#[test]
+fn completed_private_output_cancellation_drops_actual_source_backing_before_return() {
+    let fixture = Fixture::new();
+    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
+    let (source, admission) = windows(&fixture, &loaded);
+    let operators = bind_windows(source.clone(), &loaded);
+    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
+    let before = reserved(&admission);
+    let output = operators
+        .execute_projection_slice(QwenMatrixRole::Output, input(&operators, &bytes), None)
+        .unwrap();
+    assert_eq!(output.output().unwrap(), vec![-256.0; 16]);
+    assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
+    let flag = AtomicBool::new(true);
+    assert!(
+        matches!(crate::operator_slice::pure::finish_pure_output(output, Some(&flag)),
+        Err(Error::State(message)) if message.contains("cancelled before return"))
+    );
+    assert_eq!(reserved(&admission), before);
+    assert!(
+        !admission
+            .arbiter
+            .snapshot(&admission.resource)
+            .unwrap()
+            .fenced
+    );
+    drop(operators);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/qualification/QUALIFICATION_PREPARATION.patch

Original UTF-8 bytes: 129117; SHA256 `ee8feccbaef9f3fd4d38b77cb0c9e779581bcd54164f7b9dcfd29449a0e0e4fa`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ Cargo.toml
@@ -0,0 +1,24 @@
+[package]
+name = "qwen38-window-pure-roles-replay"
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
+name = "qwen38-window-pure-roles-replay"
+path = "main.rs"
--- /dev/null
+++ ROOT.md
@@ -0,0 +1,81 @@
+# ROOT-only genuine original27B pure-role gate
+
+Author: spine_review. Prepared source only; no fixtures/build/model/operator run.
+
+## Scope and provenance
+
+- Original snapshot98cea13c/revision1d4bf0f2, all18 acquired weight shards
+  55,563,006,776B; actual index1199/ordinary-text851 closure. Reused accepted
+  `factory.rs` byte-identical from the original27B three-operator replay.
+- Public PyTorch original BF16 F.linear: GDN QKV/Z/A/B/out; attention full Q+gate,
+  K/V/out; MLP gate/up/down; real untied lm_head. Three original RMSNorm calls:
+  input0, post-attention0 and final. No nonlinearities, residuals, positions,
+  caches, token/layer traversal or state publishers are added.
+- Two external teacher probes11/353. Projection hidden5120 inputs are exact
+  accepted archived original InputNorm0 BF16 words; norm inputs use accepted
+  original embedding words. Only the GDN input prefix has its actual trained
+  producer here. Other hidden-role inputs remain external probes. Nonhidden
+  widths6144/17408 use explicitly literal finite nonzero BF16 patterns, never
+  reconstructed or declared trained producer activations.
+- Sixteen actual source/header role shapes, unchanged raw BF16, three original
+  retained shard FDs. Each parameter mapping is exact page-aligned range plus
+  leading alignment, never a whole shard/full55GB mapping. Original head alone
+  is about2.543GB virtual. All parameter FD/map/tensor owners survive to exit;
+  full source/hash/version/readset checks before/after actual public operators.
+- Public capture calls each genuine operator twice, all input/output bits
+  identical. This is a repeat control, no newly invented instrumentation route.
+  Pinned original image/Torch2.13cf30153/Transformers5.17 source, AVX2/threads4;
+  CPU loader/activation profile is distinct from vLLM CUDA/FLA.
+- Exact parameter VA3,519,025,784B +6GiB runtime VA estimate +2GiB operator
+  estimate +4MiB capture <12GiB RLIMIT_AS. RAM/totalRAM+swap12GiB,CPU quota4,
+  noGPU/network, read-only source/probe/image, finite600s child/2400s parent,
+  32MiB whole artifacts. Estimates/cgroup containment do not prove complete
+  physical host/provider admission. Mapping VA is not resident RSS.
+
+## Execution, after review and timing hold
+
+1. ROOT run `python3 test_contracts.py` here (four prepared stdlib bounds tests).
+   Run the separate original27B GDN V3 actual-runtime raw serializer fixtures
+   first; tensor-bit helper body is the same equal-size integer view/C-order
+   transport with only its local support import changed.
+2. ROOT `python3 run_reference.py --new-output /ABSOLUTE/QUAL/root-reference-1`.
+   Actual full55GB source hash scans happen only under ROOT's quiet scheduling.
+3. ROOT `python3 validate_reference.py --output /ABSOLUTE/QUAL/root-reference-1`.
+   Independent stdlib full raw/header/readset/teacher/count/lifetime validator.
+   All32 cases/all inputs/all output words must pass; no tolerance.
+4. ROOT merges reviewed role candidate0f9 plus exact-hidden65cd and completed-cancel7759 followups, runs
+   strict/focused/default/source-window regressions; builds this Cargo.toml only
+   under its one-coordinator/offline actual-source isolation. No author Cargo.
+5. ROOT writes actual build attestation (not fabricated by these preparations):
+   status `ROOT_COMPILED_ORIGINAL27B_PURE_ROLES_ONLY`, actual binary/path/hash,
+   build_exit_code0, explicit absolute root_worktree, actual_source_files eight
+   path/bytes/SHA entries matching original+followup source, candidate_manifest
+   0f9f2a8f…,exact_hidden_followup_manifest65cd4364…,completed_cancel_followup_manifest7759599f…. Preserve actual compile
+   command/log/effective lock/artifact-source proof. These labels/caller hashes
+   alone are not independent evidence of compilation or correct arithmetic.
+6. ROOT `python3 run_native.py --binary ABS --binary-sha256 SHA \
+   --reference-directory /ABSOLUTE/QUAL/root-reference-1 \
+   --build-attestation ABS --new-output /ABSOLUTE/QUAL/root-native-1`.
+
+## Native proof boundary
+
+- SAME public OriginalWindowOperators pure APIs; SAME accepted source factory,
+  SourceWeights windows, HostMath/CPU64/RNE/ordinary norm. No replacement engine.
+- Full32-case input/output bit comparisons. Full248320-row untied head and
+ 17408-wide down included; every signed-zero bit is compared without tolerance.
+- Real configured original ResourceArbiter128MiB host domain, source metadata
+ 64MiB, qualifier-control8MiB, model metadata4MiB, actual per-op output/scratch
+  and row-window data+offset tickets. No claim128MiB is physical free host RAM.
+- Actual pre-cancel and real pool exhaustion refuse; every temporary owner
+  releases, no device fence. Last actual head result holds source/input/output
+  leases after caller/source drop; final all payload/source owner drop0.
+- Native process512MiB AS,CPU4/noGPU/no provider, bounded1800s/4MiB logs.
+  Exact accepted owned-v4 generic WNOWAIT group cleanup function reused, two
+  independent KILL attempts before any reap, complete group absence/EOF.
+- PASS is only private source-authentic pure-operator numerical/transport scope.
+  No all-role trained activation producer, GDN BF16 gamma/Alog/exp/initial/cached
+  math,112 StateRoots, session/full27B model, GPU eligibility or speed claim.
+
+Source-only author validation: Python ASTs/Rustfmt/small source/metadata pin and
+header reads. No weight/DSO payload hashes, fixture/Torch/operator/model/native/
+container/build/GPU execution. All failed/interrupted ROOT attempts stay intact.
--- /dev/null
+++ capture.py
@@ -0,0 +1,146 @@
+"""ROOT ONLY original-weight public pure operators; never a model constructor."""
+import importlib.metadata
+import json
+import mmap
+import os
+from pathlib import Path
+import resource
+import stat
+import struct
+import sys
+from support import HERE,MODEL_SHA,TORCH_GIT,bounds,digest,fd_sha,fingerprint,metadata,read,require,sha,source_header,unique,write
+from tensor_bits import tensor_bits
+
+OWNER=None
+
+class SourceOwner:
+    def __init__(self,plan):
+        self.plan=plan;self.fds={};self.headers={};self.before={};self.maps={};self.values={};self.versions={};self.readsets={}
+        for name,pin in plan['metadata'].items():read(Path('/source')/name,pin=pin)
+        config=metadata(Path('/source/config.json'));text=config['text_config']
+        require(config['architectures']==['Qwen3_5ForConditionalGeneration']and
+                text['hidden_size']==5120 and text['intermediate_size']==17408 and
+                text['vocab_size']==248320 and text['rms_norm_eps']==1e-6 and
+                text['tie_word_embeddings']is False,'exact original27B pure-role config/epsilon')
+        index=metadata(Path('/source/model.safetensors.index.json'))['weight_map']
+        for shard,pin in plan['selected_shards'].items():
+            fd=os.open(Path('/source')/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW);self.fds[shard]=fd
+            actual=os.fstat(fd)
+            require(stat.S_ISREG(actual.st_mode)and actual.st_size==pin['file']['bytes'],
+                    'exact regular selected shard before any full scan')
+            self.before[shard]=fingerprint(actual)
+            require(fd_sha(fd,0,pin['file']['bytes'])==pin['file']['sha256'],'same original full selected shard')
+            n,raw=source_header(fd,pin)
+            require(fingerprint(os.fstat(fd))==self.before[shard],'selected FD immutable before imports')
+            self.headers[shard]=(8+n,json.loads(raw,object_pairs_hook=unique))
+        for role in plan['roles']:
+            shard=role['shard'];start,header=self.headers[shard];row=header[role['name']]
+            require(index[role['name']]==shard and row['shape']==role['shape'] and row['dtype']=='BF16' and
+                    start+row['data_offsets'][0]==role['absolute_offset'] and
+                    row['data_offsets'][1]-row['data_offsets'][0]==role['bytes'],'exact full original role/header')
+    def map(self,torch):
+        for role in self.plan['roles']:
+            name=role['name'];mapping=mmap.mmap(self.fds[role['shard']],role['mapping_bytes'],
+                access=mmap.ACCESS_COPY,offset=role['page_aligned_offset']);self.maps[name]=mapping
+            value=torch.frombuffer(mapping,dtype=torch.bfloat16,count=role['bytes']//2,
+                                   offset=role['view_offset']).reshape(role['shape'])
+            require(value.is_contiguous() and value.device.type=='cpu','authentic original role tensor')
+            self.values[name]=value;self.versions[name]=value._version
+            self.readsets[name]=fd_sha(self.fds[role['shard']],role['absolute_offset'],role['bytes'])
+    def finish(self):
+        for role in self.plan['roles']:
+            name=role['name'];require(self.values[name]._version==self.versions[name],'original source version unchanged')
+            h=__import__('hashlib').sha256();view=memoryview(self.maps[name])
+            try:
+                start=role['view_offset']
+                for offset in range(0,role['bytes'],1024**2):h.update(view[start+offset:start+min(role['bytes'],offset+1024**2)])
+            finally:view.release()
+            require(h.hexdigest()==self.readsets[name],'every original mapped role byte unchanged')
+        for shard,fd in self.fds.items():
+            pin=self.plan['selected_shards'][shard]['file']
+            require(fingerprint(os.fstat(fd))==self.before[shard] and fd_sha(fd,0,pin['bytes'])==pin['sha256'],
+                    'same retained source FD/full bytes unchanged')
+
+
+def main():
+    global OWNER
+    output=Path('/output/captured');require(not output.exists(),'exclusive actual public reference');output.mkdir(mode=0o700)
+    report={'status':'UNQUALIFIED','cases':[],'whole_model_or_state_or_GPU_qualified':False}
+    try:
+        cpus=[int(v)for v in os.environ['ZG_ALLOWED_CPUS'].split(',')]
+        require(len(cpus)==len(set(cpus))==4,'exact four CPUs');os.sched_setaffinity(0,cpus)
+        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void' and
+                os.environ.get('ATEN_CPU_CAPABILITY')=='avx2' and not os.environ.get('LD_PRELOAD') and
+                not os.environ.get('LD_AUDIT'),'pinned public CPU policy')
+        require(sha(HERE/'source-plan.json')==os.environ['ZG_SOURCE_PLAN_SHA'],'bound independent source plan')
+        plan=metadata(HERE/'source-plan.json');derived=bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
+        OWNER=SourceOwner(plan)
+        import torch
+        from torch.nn import functional as F
+        from transformers.models.qwen3_5 import modeling_qwen3_5 as original
+        require(torch.__version__=='2.13.0+cu129' and torch.version.git_version==TORCH_GIT and
+                importlib.metadata.version('transformers')=='5.17.0' and sha(original.__file__)==MODEL_SHA and
+                not torch.cuda.is_available(),'exact pinned original public CPU implementation')
+        torch.set_num_threads(4);torch.set_num_interop_threads(1)
+        require(torch.backends.cpu.get_cpu_capability()=='AVX2','actual CPU AVX2 target')
+        OWNER.map(torch)
+        prefix=metadata(Path('/prefix-reference/reference.json'),pin=plan['prior_public_prefix'])
+        require(prefix['source_repo']==plan['repository'] and prefix['source_revision']==plan['revision'] and
+                prefix['snapshot']==plan['snapshot'],'authentic accepted original27B input archive')
+        total=0
+        def dump(name,raw):
+            nonlocal total
+            total+=len(raw);require(total<=4*1024**2 and len(raw)<=2*1024**2,'bounded complete operator payload')
+            with (output/name).open('xb')as file:file.write(raw)
+            return {'file':name,'bytes':len(raw),'sha256':digest(raw)}
+        with torch.inference_mode():
+            for step,token in enumerate((11,353)):
+                require(prefix['steps'][step]['token']==token,'actual external teacher binding')
+                for i,role in enumerate(plan['roles']):
+                    count=role['input_elements']
+                    if count==5120:
+                        prior=prefix['steps'][step][role['input_stage']]
+                        full=read(Path('/prefix-reference')/prior['file'],pin=prior)
+                        require(len(full)==5120*4 and all(word&0xffff==0 for word, in struct.iter_unpack('<I',full)),
+                                'exact BF16-widened accepted input words')
+                        raw=b''.join((word>>16).to_bytes(2,'little')for word, in struct.iter_unpack('<I',full))
+                        require(digest(raw)==prior['bf16_raw_sha256'],'all accepted genuine raw BF16 input bits')
+                    else:
+                        prior=None
+                        raw=b''.join(((0xbf00 if (j+token)%2 else 0x3f00)|((j*13+token)%128)).to_bytes(2,'little')for j in range(count))
+                    x=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(1,count)
+                    weight=OWNER.values[role['name']]
+                    if role['kind']=='projection':
+                        evaluate=lambda:F.linear(x,weight)
+                    else:
+                        with torch.device('meta'):norm=original.Qwen3_5RMSNorm(count,eps=1e-6)
+                        norm.load_state_dict({'weight':weight},strict=True,assign=True);norm.eval()
+                        require(norm.weight.data_ptr()==weight.data_ptr()and norm.weight.dtype==torch.bfloat16,'raw original norm, no learned cast/cache')
+                        evaluate=lambda:norm(x)
+                    first=evaluate();second=evaluate()
+                    require(first.dtype==second.dtype==torch.bfloat16 and list(first.shape)==list(second.shape)==[1,role['output_elements']],
+                            'actual complete BF16 public operator output')
+                    firstraw=tensor_bits(first,torch);secondraw=tensor_bits(second,torch)
+                    require(firstraw==secondraw and tensor_bits(x,torch)==raw,'two genuine calls/input unchanged all bits')
+                    require(all(word&0x7f80!=0x7f80 for word, in struct.iter_unpack('<H',firstraw)),'finite public output raw words')
+                    stem=f'step-{step}-role-{i:02}'
+                    report['cases'].append({'step':step,'teacher_token':token,'role':role['role'],'layer':role['layer'],
+                        'kind':role['kind'],'source_tensor':role['name'],'source_shape':role['shape'],'source_dtype':'BF16',
+                        'input_provenance':role['input_provenance'],'input_prior':prior,
+                        'input':dump(stem+'.input.bf16le.bin',raw),'output':dump(stem+'.output.bf16le.bin',firstraw),
+                        'original_public_operator_calls':2,'both_calls_full_bits_exact':True})
+        OWNER.finish()
+        report.update(status='CAPTURED_ORIGINAL27B_PURE_OPERATOR_REFERENCE_REQUIRES_VALIDATION',
+            source_plan_sha256=sha(HERE/'source-plan.json'),source_repo=plan['repository'],source_revision=plan['revision'],
+            snapshot=plan['snapshot'],pre_library_bounds=derived,source_FDs=OWNER.before,source_readsets=OWNER.readsets,
+            all_original_source_and_mapped_bytes_unchanged=True,source_owner_retained_until_process_exit=True,
+            payload_bytes=total,torch=torch.__version__,torch_git=torch.version.git_version,modeling_sha256=MODEL_SHA,
+            affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
+            scope=plan['scope'])
+    except BaseException as error:
+        report['failure']=repr(error);raise
+    finally:write(output/'manifest.json',report)
+
+
+if __name__=='__main__':main()
--- /dev/null
+++ factory.rs
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
+++ main.rs
@@ -0,0 +1,339 @@
+//! ROOT only: source-authentic pure roles; no layer/token/session/state engine.
+mod factory;
+use serde_json::{Value, json};
+use sha2::{Digest, Sha256};
+use std::error::Error;
+use std::fs::OpenOptions;
+use std::io::{Read, Write};
+use std::os::unix::fs::OpenOptionsExt;
+use std::path::Path;
+use std::sync::atomic::AtomicBool;
+use zerogravity_artifacts::cache::TransferAdmission;
+use zerogravity_core::accelerator::program::ProgramNumericalProfile;
+use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
+use zerogravity_model_qwen35::operator_slice::pure::{
+    PureSliceInput, PureSliceOutput, PureSliceRole,
+};
+use zerogravity_model_qwen35::original_binding::{
+    OriginalWindowOperators, QwenMatrixRole as M, QwenNumericalProfile, QwenParameterRole as P,
+};
+
+type Proof<T> = Result<T, Box<dyn Error>>;
+const HOST: u64 = 128 * 1024 * 1024;
+const CONTROL: usize = 8 * 1024 * 1024;
+fn need(ok: bool, why: &str) -> Proof<()> {
+    if ok { Ok(()) } else { Err(why.into()) }
+}
+fn sha(raw: &[u8]) -> String {
+    format!("{:x}", Sha256::digest(raw))
+}
+fn read(path: &Path, cap: usize) -> Proof<Vec<u8>> {
+    let mut file = OpenOptions::new()
+        .read(true)
+        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+        .open(path)?;
+    let before = file.metadata()?;
+    need(
+        before.is_file() && before.len() <= cap as u64,
+        "bounded actual regular reference",
+    )?;
+    let mut raw = Vec::new();
+    Read::by_ref(&mut file)
+        .take(cap as u64 + 1)
+        .read_to_end(&mut raw)?;
+    use std::os::unix::fs::MetadataExt;
+    let after = file.metadata()?;
+    need(
+        raw.len() <= cap
+            && raw.len() as u64 == before.len()
+            && before.dev() == after.dev()
+            && before.ino() == after.ino()
+            && before.len() == after.len()
+            && before.mtime_nsec() == after.mtime_nsec()
+            && before.ctime_nsec() == after.ctime_nsec()
+            && before.mtime() == after.mtime()
+            && before.ctime() == after.ctime(),
+        "same immutable actual reference FD",
+    )?;
+    Ok(raw)
+}
+fn field<'a>(value: &'a Value, key: &str) -> Proof<&'a str> {
+    value[key]
+        .as_str()
+        .ok_or_else(|| format!("missing string:{key}").into())
+}
+fn payload(root: &Path, row: &Value) -> Proof<Vec<u8>> {
+    let name = field(row, "file")?;
+    need(
+        Path::new(name).file_name().and_then(|v| v.to_str()) == Some(name),
+        "contained reference filename",
+    )?;
+    let raw = read(&root.join(name), 2 * 1024 * 1024)?;
+    need(
+        row["bytes"] == raw.len() && row["sha256"] == sha(&raw) && raw.len().is_multiple_of(2),
+        "full original reference raw bytes",
+    )?;
+    Ok(raw)
+}
+fn compare(actual: &[f32], raw: &[u8]) -> Proof<usize> {
+    need(
+        actual.len().checked_mul(2) == Some(raw.len()),
+        "complete raw word coverage",
+    )?;
+    for (i, (value, word)) in actual.iter().zip(raw.chunks_exact(2)).enumerate() {
+        let expected = u32::from(u16::from_le_bytes([word[0], word[1]])) << 16;
+        if value.to_bits() != expected {
+            return Err(format!(
+                "UNQUALIFIED first exact mismatch word{i}: {:08x}!={expected:08x}",
+                value.to_bits()
+            )
+            .into());
+        }
+    }
+    Ok(actual.len())
+}
+fn role(row: &Value) -> Proof<PureSliceRole> {
+    let layer = || -> Proof<usize> {
+        row["layer"]
+            .as_u64()
+            .and_then(|n| usize::try_from(n).ok())
+            .ok_or_else(|| "actual layer".into())
+    };
+    Ok(match (field(row, "kind")?, field(row, "role")?) {
+        ("projection", "GdnQkv") => PureSliceRole::Projection(M::GdnQkv(layer()?)),
+        ("projection", "GdnGate") => PureSliceRole::Projection(M::GdnGate(layer()?)),
+        ("projection", "GdnAlpha") => PureSliceRole::Projection(M::GdnAlpha(layer()?)),
+        ("projection", "GdnBeta") => PureSliceRole::Projection(M::GdnBeta(layer()?)),
+        ("projection", "GdnOutput") => PureSliceRole::Projection(M::GdnOutput(layer()?)),
+        ("projection", "Query") => PureSliceRole::Projection(M::Query(layer()?)),
+        ("projection", "Key") => PureSliceRole::Projection(M::Key(layer()?)),
+        ("projection", "Value") => PureSliceRole::Projection(M::Value(layer()?)),
+        ("projection", "AttentionOutput") => {
+            PureSliceRole::Projection(M::AttentionOutput(layer()?))
+        }
+        ("projection", "FfnGate") => PureSliceRole::Projection(M::FfnGate(layer()?)),
+        ("projection", "FfnUp") => PureSliceRole::Projection(M::FfnUp(layer()?)),
+        ("projection", "FfnDown") => PureSliceRole::Projection(M::FfnDown(layer()?)),
+        ("projection", "Output") => PureSliceRole::Projection(M::Output),
+        ("ordinary_rms", "InputNorm") => PureSliceRole::OrdinaryRms(P::InputNorm(layer()?)),
+        ("ordinary_rms", "FfnNorm") => PureSliceRole::OrdinaryRms(P::FfnNorm(layer()?)),
+        ("ordinary_rms", "FinalNorm") => PureSliceRole::OrdinaryRms(P::FinalNorm),
+        _ => return Err("unknown pure role".into()),
+    })
+}
+fn run(
+    host: &TransferAdmission,
+    directory: &Path,
+    reference_sha: &str,
+    validation_sha: &str,
+) -> Proof<Value> {
+    let ready = factory::open_source(host)?;
+    let source = ready.source;
+    let control = source.reserve_workspace(CONTROL)?;
+    let plan: Value = serde_json::from_slice(include_bytes!("source-plan.json"))?;
+    let refraw = read(&directory.join("captured/manifest.json"), 4 * 1024 * 1024)?;
+    let validraw = read(&directory.join("validation.json"), 4 * 1024 * 1024)?;
+    need(
+        sha(&refraw) == reference_sha && sha(&validraw) == validation_sha,
+        "ROOT bound independently validated public artifacts",
+    )?;
+    let reference: Value = serde_json::from_slice(&refraw)?;
+    let validated: Value = serde_json::from_slice(&validraw)?;
+    need(
+        validated["status"] == "PASS_ORIGINAL27B_PURE_OPERATOR_REFERENCE_ONLY"
+            && validated["manifest_sha256"] == reference_sha
+            && validated["source_plan_sha256"] == sha(include_bytes!("source-plan.json"))
+            && reference["source_plan_sha256"] == validated["source_plan_sha256"],
+        "genuine public reference/profile binding, no caller hash as independent math proof",
+    )?;
+    let operators = OriginalWindowOperators::bind(
+        source.clone(),
+        &ready.config,
+        "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
+        "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )?;
+    need(
+        operators.binding().text_tensor_count() == 851 && source.source_bytes() == 55_563_006_776,
+        "actual complete original27B851 role source closure",
+    )?;
+    let cases = reference["cases"].as_array().ok_or("public cases")?;
+    let roles = plan["roles"].as_array().ok_or("frozen roles")?;
+    need(
+        cases.len() == 32 && roles.len() == 16,
+        "full16 roles/two probes",
+    )?;
+    let steady = host.arbiter.snapshot(&host.resource)?.reserved;
+    let first = payload(&directory.join("captured"), &cases[0]["input"])?;
+    let input = PureSliceInput::from_bf16_bytes(
+        source.source_identity(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        &first,
+    )?;
+    need(
+        matches!(operators.execute_projection_slice(M::GdnQkv(0),input,Some(&AtomicBool::new(true))),
+          Err(zerogravity_core::Error::State(ref message)) if message.contains("cancelled")),
+        "actual pre-cancel refuses",
+    )?;
+    let mut pressure = Vec::new();
+    loop {
+        let remaining = host.arbiter.snapshot(&host.resource)?.remaining;
+        if remaining <= 1 {
+            break;
+        }
+        pressure.push(source.reserve_workspace((remaining - 1).min(32 * 1024 * 1024) as usize)?);
+    }
+    let held_bytes = host.arbiter.snapshot(&host.resource)?.reserved;
+    need(
+        matches!(operators.execute_projection_slice(M::GdnQkv(0),input,None),
+          Err(zerogravity_core::Error::Limit(ref message)) if message=="physical device memory admission refused"),
+        "actual arbiter capacity refusal",
+    )?;
+    need(
+        host.arbiter.snapshot(&host.resource)?.reserved == held_bytes,
+        "refusal leaves prior real charge exact",
+    )?;
+    drop(pressure);
+    need(
+        host.arbiter.snapshot(&host.resource)?.reserved == steady,
+        "pressure owners restore exact ledger",
+    )?;
+    let mut compared = 0usize;
+    let mut reports = Vec::new();
+    let mut retained: Option<PureSliceOutput> = None;
+    for (position, row) in cases.iter().enumerate() {
+        let (step, index) = (position / 16, position % 16);
+        let expected = &roles[index];
+        need(
+            row["step"] == step
+                && row["teacher_token"] == [11, 353][step]
+                && row["role"] == expected["role"]
+                && row["layer"] == expected["layer"]
+                && row["kind"] == expected["kind"]
+                && row["source_tensor"] == expected["name"]
+                && row["source_shape"] == expected["shape"],
+            "ordered source/role/teacher closure",
+        )?;
+        let original = source.tensor_info(field(row, "source_tensor")?)?;
+        let shape = expected["shape"]
+            .as_array()
+            .ok_or("original shape")?
+            .iter()
+            .map(|v| v.as_u64().ok_or("shape axis"))
+            .collect::<Result<Vec<_>, _>>()?;
+        need(
+            original.shape == shape && original.kind == zerogravity_core::quant::QuantKind::Bf16,
+            "actual sealed original source shape/codec",
+        )?;
+        let x = payload(&directory.join("captured"), &row["input"])?;
+        let y = payload(&directory.join("captured"), &row["output"])?;
+        let transport = PureSliceInput::from_bf16_bytes(
+            source.source_identity(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            &x,
+        )?;
+        let target = role(row)?;
+        let output = match target {
+            PureSliceRole::Projection(role) => {
+                operators.execute_projection_slice(role, transport, None)?
+            }
+            PureSliceRole::OrdinaryRms(role) => {
+                operators.execute_ordinary_rms_slice(role, transport, None)?
+            }
+        };
+        need(
+            output.role() == target && output.source_identity() == source.source_identity(),
+            "exact native output identity",
+        )?;
+        compared += compare(output.input()?, &x)?;
+        compared += compare(output.output()?, &y)?;
+        reports.push(json!({"position":position,"role":row["role"],"source_tensor":row["source_tensor"],"output_words":output.output()?.len(),"all_bits_exact":true}));
+        if target == PureSliceRole::Projection(M::Output) && step == 1 {
+            retained = Some(output);
+        }
+    }
+    need(
+        compared
+            == validated["words"]
+                .as_u64()
+                .ok_or("complete independent words")? as usize,
+        "all actual public/native words compared",
+    )?;
+    let result = retained.ok_or("retained actual untied head output")?;
+    let source_meta = source.metadata_reserved_bytes();
+    drop(operators);
+    drop(source);
+    drop(ready.config);
+    let retained_ledger = host.arbiter.snapshot(&host.resource)?;
+    need(
+        !retained_ledger.fenced
+            && retained_ledger.reserved == source_meta + CONTROL as u64 + 3_035_648,
+        "actual head input/output/source/control retained after caller drop",
+    )?;
+    need(
+        result.output()?.len() == 248320 && result.input()?.len() == 5120,
+        "full untied head remains borrowable",
+    )?;
+    drop(result);
+    drop(first);
+    drop(refraw);
+    drop(validraw);
+    drop(reference);
+    drop(validated);
+    drop(plan);
+    drop(control);
+    need(
+        host.arbiter.snapshot(&host.resource)?.reserved == 0,
+        "last private/source owner releases exact budget",
+    )?;
+    Ok(
+        json!({"cases":reports,"compared_words":compared,"all_exact":true,"actual_pre_cancel_and_pressure_refused":true,
+        "retained_head_bytes":3035648,"whole_model_StateRoot_GPU_or_performance_qualified":false}),
+    )
+}
+fn main() {
+    let args = std::env::args().collect::<Vec<_>>();
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "original27-pure-role-replay").expect("key");
+    arbiter
+        .register(key.clone(), HOST)
+        .expect("isolated bounded ledger");
+    let host = TransferAdmission {
+        arbiter,
+        resource: key,
+        owner: ResourceOwner::new("original27", "pure-role-root-replay").expect("owner"),
+    };
+    let result = (|| -> Proof<Value> {
+        need(
+            args.len() == 5,
+            "REFERENCE_DIR MANIFEST_SHA VALIDATION_SHA OUTPUT",
+        )?;
+        run(&host, Path::new(&args[1]), &args[2], &args[3])
+    })();
+    let final_ledger = host.arbiter.snapshot(&host.resource).expect("final ledger");
+    let okay = result.is_ok() && final_ledger.reserved == 0 && !final_ledger.fenced;
+    let report = match result {
+        Ok(result) => {
+            json!({"status":if okay{"PASS_PRIVATE_ORIGINAL27B_PURE_ROLES_ONLY"}else{"UNQUALIFIED"},"result":result})
+        }
+        Err(error) => json!({"status":"UNQUALIFIED","error":error.to_string()}),
+    };
+    let report = json!({"proof":report,"final_reserved":final_ledger.reserved,"final_fenced":final_ledger.fenced,
+                      "configured_host_limit":HOST,"whole_host_physical_availability_qualified":false});
+    if let Some(path) = args.get(4) {
+        let write = (|| -> Proof<()> {
+            let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
+            f.write_all(&serde_json::to_vec_pretty(&report)?)?;
+            f.sync_all()?;
+            Ok(())
+        })();
+        if let Err(error) = write {
+            eprintln!("REPORT_FAILURE:{error}");
+            std::process::exit(1);
+        }
+    }
+    if !okay {
+        eprintln!("{report}");
+        std::process::exit(2);
+    }
+}
--- /dev/null
+++ owned_container_v4.py
@@ -0,0 +1,280 @@
+"""ROOT ONLY finite owned lifetime. No execution occurs by importing this file.
+
+Additive v4: original nonce/CID binding and commands preserved. WNOWAIT and
+selector cleanup follow the reviewed AV runner, with whole-group KILL strictly
+before leader reap: its reserved PID prevents accidental process-group reuse.
+"""
+import hashlib
+import json
+import os
+from pathlib import Path
+import re
+import selectors
+import signal
+import subprocess
+import time
+
+PODMAN = ["podman", "--root", "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers", "--runroot", "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers"]
+LABEL = "abird.zerogravity.vllm-gpu-oracle-owner"
+CAP = 4 * 1024 * 1024
+
+
+def require(value, message):
+    if not value:
+        raise ValueError(message)
+
+
+def sha(path):
+    with Path(path).open("rb") as file:
+        return hashlib.file_digest(file, "sha256").hexdigest()
+
+
+def command(argv, timeout=10):
+    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                            stderr=subprocess.PIPE, timeout=timeout, check=False)
+    require(len(result.stdout) + len(result.stderr) <= 1024 * 1024, "bounded command receipt")
+    return result
+
+
+def leader_status_unreaped(process):
+    require(process.returncode is None, "owned leader already reaped; never signal a recycled PGID")
+    observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if observed is None:
+        return None
+    require(observed.si_pid == process.pid and observed.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
+            "owned leader exit observation")
+    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
+
+
+class Log:
+    """Main-thread bounded drain; no thread/future can survive cleanup."""
+    def __init__(self, pipe, path):
+        self.pipe, self.path = pipe, Path(path)
+        self.selector = self.file = None
+        self.retained = self.total = 0
+        self.eof = self.closed = False
+        self.errors = []
+        try:
+            self.file = self.path.open("xb")
+            os.chmod(self.path, 0o600)
+            os.set_blocking(pipe.fileno(), False)
+            self.selector = selectors.DefaultSelector()
+            self.selector.register(pipe, selectors.EVENT_READ)
+        except BaseException:
+            if self.selector is not None:
+                self.selector.close()
+            if self.file is not None:
+                self.file.close()
+            raise
+
+    def drain(self, timeout):
+        for key, _ in self.selector.select(timeout):
+            # One bounded block per wake; the exit/deadline remains observable
+            # even when an owned writer continuously produces output.
+            try:
+                raw = os.read(key.fd, 65536)
+            except BlockingIOError:
+                continue
+            if not raw:
+                self.selector.unregister(key.fileobj)
+                self.eof = True
+                continue
+            self.total += len(raw)
+            keep = raw[:max(0, CAP - self.retained)]
+            if keep:
+                try:
+                    self.file.write(keep)
+                    self.retained += len(keep)
+                except OSError as error:
+                    self.errors.append(repr(error))
+
+    def observe(self, process, seconds):
+        deadline = time.monotonic() + seconds
+        while time.monotonic() < deadline:
+            self.drain(0.05)
+            require(self.total <= CAP and not self.errors, "owned log failed or exceeded4MiB")
+            status = leader_status_unreaped(process)
+            if status is not None:
+                return status
+        raise TimeoutError("owned foreground exit deadline (leader remains unreaped)")
+
+    def finish(self):
+        failures = []
+        try:
+            deadline = time.monotonic() + 3
+            while not self.eof and time.monotonic() < deadline:
+                self.drain(0.05)
+            require(self.eof, "owned stdout pipe EOF deadline")
+            require(self.total <= CAP and not self.errors, "owned log evidence failed/truncated")
+            self.file.flush()
+            os.fsync(self.file.fileno())
+        except BaseException as error:
+            failures.append("drain/evidence: " + repr(error))
+        finally:
+            # Close all three even when EOF, evidence I/O or one close fails.
+            for name, close in (("selector", self.selector.close), ("log", self.file.close),
+                                ("pipe", self.pipe.close)):
+                try:
+                    close()
+                except BaseException as error:
+                    failures.append(name + ": " + repr(error))
+            self.closed = self.pipe.closed and self.file.closed and not self.selector.get_map()
+        if failures:
+            raise RuntimeError("owned log cleanup: " + "; ".join(failures))
+
+
+def owned_container(name, nonce, cidfile):
+    target, recorded_cid = name, None
+    if cidfile.exists():
+        cid = cidfile.read_text().strip()
+        if cid:
+            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
+            target = recorded_cid = cid
+    exists = command(PODMAN + ["container", "exists", target])
+    if exists.returncode == 1:
+        return None
+    require(exists.returncode == 0, "owned container existence check failed")
+    inspected = command(PODMAN + ["container", "inspect", target])
+    require(inspected.returncode == 0, "owned container inspect failed")
+    value, = json.loads(inspected.stdout)
+    require(value["Name"] == name and value["Config"].get("Labels", {}).get(LABEL) == nonce,
+            "container name/nonce mismatch; no signal")
+    cid = value["Id"]
+    require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
+    if recorded_cid is not None:
+        require(recorded_cid == cid, "owned CID changed; no signal")
+    return cid
+
+
+def cleanup_container(name, nonce, cidfile):
+    errors = []
+    try:
+        cid = owned_container(name, nonce, cidfile)
+    except BaseException as error:
+        return ["container owner binding: " + repr(error)]
+    if cid is None:
+        return errors
+    try:
+        stopped = command(PODMAN + ["stop", "--time", "5", cid], 15)
+        if stopped.returncode != 0:
+            errors.append("owned stop returned nonzero")
+    except BaseException as error:
+        errors.append("container stop: " + repr(error))
+    try:
+        # Fresh exact owner binding is required even after a failed/timedout
+        # stop. A stop failure must not skip the bounded force-remove attempt.
+        if owned_container(name, nonce, cidfile) is not None:
+            removed = command(PODMAN + ["rm", "--force", cid], 15)
+            require(removed.returncode == 0, "owned container force cleanup failed")
+    except BaseException as error:
+        errors.append("container force cleanup: " + repr(error))
+    try:
+        require(owned_container(name, nonce, cidfile) is None, "owned container survived cleanup")
+    except BaseException as error:
+        errors.append("container final absence: " + repr(error))
+    return errors
+
+
+def group_exists(pgid):
+    try:
+        os.killpg(pgid, 0)
+        return True
+    except ProcessLookupError:
+        return False
+
+
+def signal_group_unreaped(process, sig):
+    # waitid authenticates that this is still our unreaped child. start_new_session
+    # made it the session/group leader; no foreign group is inferred from a PID.
+    leader_status_unreaped(process)
+    require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
+            "owned leader session/PGID binding")
+    os.killpg(process.pid, sig)
+
+
+def cleanup_group(process, receipt, *, observe=leader_status_unreaped, pause=time.sleep):
+    """Independent bounded stages; no nonzero signal after any reap attempt."""
+    state = receipt["foreground_cleanup"] = {"pgid": process.pid, "term_sent": False,
+        "kill_sent_before_reap": False, "leader_reaped": False, "group_absent": False,
+        "kill_attempts_before_reap": 0, "leader_reap_attempts": 0}
+    errors = []
+    try:
+        signal_group_unreaped(process, signal.SIGTERM)
+        state["term_sent"] = True
+    except ProcessLookupError:
+        state["term_group_absent_before_reap"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
+    try:
+        deadline = time.monotonic() + 1
+        while time.monotonic() < deadline:
+            observe(process)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # Both independently authenticated KILL attempts precede ANY wait/reap.
+    for attempt in range(2):
+        state["kill_attempts_before_reap"] += 1
+        try:
+            signal_group_unreaped(process, signal.SIGKILL)
+            state["kill_sent_before_reap"] = True
+        except ProcessLookupError:
+            state["kill_group_absent_before_reap"] = True
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    state["all_nonzero_signals_before_reap"] = True
+    # Popen.wait targets only this child. A failed first wait cannot skip final wait.
+    try:
+        state["leader_reap_attempts"] += 1
+        state["exit_code"] = process.wait(timeout=5)
+        state["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned first leader reap: " + repr(error))
+    finally:
+        if process.returncode is None:
+            try:
+                state["leader_reap_attempts"] += 1
+                state["exit_code"] = process.wait(timeout=5)
+                state["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final leader reap: " + repr(error))
+        else:
+            state["exit_code"] = process.returncode
+            state["leader_reaped"] = True
+    try:
+        require(state["leader_reaped"], "owned leader was not reaped")
+        deadline = time.monotonic() + 5
+        while group_exists(process.pid) and time.monotonic() < deadline:
+            pause(0.05)  # group_exists uses signal0 only; never signal after reap.
+        state["group_absent"] = not group_exists(process.pid)
+        require(state["group_absent"], "owned foreground group remains after final KILL/reap")
+    except BaseException as error:
+        errors.append("owned group absence: " + repr(error))
+    state["stage_errors"] = list(errors)
+    if errors:
+        raise RuntimeError("independent owned foreground cleanup: " + "; ".join(errors))
+
+
+def cleanup(process, name, nonce, cidfile, receipt):
+    errors = []
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("initial owned container cleanup: " + repr(error))
+    try:
+        cleanup_group(process, receipt)
+    except BaseException as error:
+        errors.append("owned foreground cleanup: " + repr(error))
+    # Rediscover startup races using only the original nonce/name/CID binding.
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("final owned container cleanup: " + repr(error))
+    try:
+        require(receipt.get("foreground_cleanup", {}).get("leader_reaped"), "owned leader not reaped")
+        require(receipt["foreground_cleanup"]["group_absent"], "owned group absence not established")
+        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
+        receipt["owned_container_absent_after_reap"] = True
+    except BaseException as error:
+        errors.append("final owned absence: " + repr(error))
+    return errors
--- /dev/null
+++ run_native.py
@@ -0,0 +1,127 @@
+"""ROOT ONLY owned same-engine pure-role child; no model/runtime on import."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import resource
+import signal
+import subprocess
+import time
+from support import HERE,frozen,metadata,require,sha,write
+
+
+def main():
+    parser=argparse.ArgumentParser()
+    parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--binary-sha256',required=True)
+    parser.add_argument('--reference-directory',type=Path,required=True)
+    parser.add_argument('--build-attestation',type=Path,required=True);parser.add_argument('--new-output',type=Path,required=True)
+    args=parser.parse_args()
+    freeze=frozen()
+    require(args.binary.is_absolute()and not args.binary.is_symlink()and sha(args.binary)==args.binary_sha256,
+            'actual ROOT-built immutable same-engine executable')
+    build=metadata(args.build_attestation)
+    require(build['status']=='ROOT_COMPILED_ORIGINAL27B_PURE_ROLES_ONLY'and build['build_exit_code']==0 and
+            build['binary_sha256']==args.binary_sha256 and Path(build['binary']).resolve()==args.binary.resolve()and
+            build['candidate_manifest_sha256']=='0f9f2a8ff4ff28f5f3284063aecff35508ca61cdd083dac454400e7aec896d3e'and
+            build['exact_hidden_followup_manifest_sha256']=='65cd436471e26679d4ff596538fda31c9b1c6f28cef6d9a55f7c99e9d28ca82b'and
+            build['completed_cancel_followup_manifest_sha256']=='7759599f51215de539167a3f71b0efb39aace80fcf817b3433fbe5eba6dfdc6c',
+            'ROOT actual build/candidate/followup binding; marker alone is not compilation proof')
+    actual_files=build['actual_source_files'];require(len(actual_files)==8,'all eight actual implementation source paths')
+    worktree=Path(build['root_worktree']);require(worktree.is_absolute()and worktree.is_dir(),'explicit ROOT build worktree')
+    expected=metadata(HERE.parent/'freeze-manifest.json')['files']
+    fixes={row['path']:row for row in metadata(HERE.parent/'followup-v2/freeze-manifest.json')['files']}
+    fixes.update({row['path']:row for row in metadata(HERE.parent/'followup-v3/freeze-manifest.json')['files']})
+    for row in expected:
+        pin=fixes.get(row['path'],row);actual=[v for v in actual_files if v['path']==row['path']]
+        require(len(actual)==1 and actual[0]['sha256']==pin['sha256']and actual[0]['bytes']==pin['bytes'],
+                'actual compiled source attestation matches exact reviewed implementation')
+        require((worktree/row['path']).stat().st_size==pin['bytes']and sha(worktree/row['path'])==pin['sha256'],
+                'actual ROOT implementation source bytes still exact')
+    for pin in metadata(HERE.parent/'freeze-manifest.json')['unchanged_dependency_source_pins']:
+        require((worktree/pin['path']).stat().st_size==pin['bytes']and sha(worktree/pin['path'])==pin['sha256'],
+                'unchanged original CPU math/core/source closure bytes')
+    require(args.reference_directory.is_absolute()and args.reference_directory.parent.resolve()==HERE and
+            args.new_output.is_absolute()and args.new_output.parent.resolve()==HERE,'owned reference/native outputs')
+    validation=metadata(args.reference_directory/'validation.json')
+    require(validation['status']=='PASS_ORIGINAL27B_PURE_OPERATOR_REFERENCE_ONLY'and
+            validation['freeze_sha256']==freeze and validation['manifest_sha256']==sha(args.reference_directory/'captured/manifest.json'),
+            'independently validated genuine public raw reference first')
+    cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus)
+    args.new_output.mkdir(mode=0o700)
+    require(sha(HERE/'owned_container_v4.py')=='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
+            'exact reviewed original WNOWAIT cleanup')
+    spec=importlib.util.spec_from_file_location('pure_native_owned',HERE/'owned_container_v4.py')
+    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
+    fd=os.open(args.binary,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+    child=log=None;failure=None;started=time.monotonic()
+    receipt={'status':'UNQUALIFIED','freeze_sha256':freeze,'binary_sha256':args.binary_sha256,
+             'build_attestation_sha256':sha(args.build_attestation),'expected_cpus':cpus,
+             'scope':'same-engine private original27B pure roles only; no full-model/state/GPU'}
+    try:
+        with os.fdopen(os.dup(fd),'rb')as opened:
+            import hashlib
+            require(hashlib.file_digest(opened,'sha256').hexdigest()==args.binary_sha256,'exact retained exec FD')
+        manifest_sha=sha(args.reference_directory/'captured/manifest.json');validation_sha=sha(args.reference_directory/'validation.json')
+        argv=[str(args.binary),str(args.reference_directory),manifest_sha,validation_sha,str(args.new_output/'report.json')]
+        receipt['argv']=argv
+        environment={k:v for k,v in os.environ.items()if k in ('PATH','HOME','LANG','LC_ALL','TZ','XDG_CACHE_HOME')}
+        environment.update(RAYON_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void')
+        def limits():
+            os.sched_setaffinity(0,cpus)
+            resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
+            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
+            resource.setrlimit(resource.RLIMIT_CPU,(1801,1801))
+            resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,4*1024**2))
+        def cancelled(_signal,_frame):raise InterruptedError('owned native pure replay cancelled')
+        signal.signal(signal.SIGTERM,cancelled)
+        try:
+            child=subprocess.Popen(argv,executable=f'/proc/self/fd/{fd}',pass_fds=(fd,),
+                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
+                env=environment,start_new_session=True,preexec_fn=limits)
+            receipt['owned_cli_pid']=child.pid
+            log=helper.Log(child.stdout,args.new_output/'execution.log')
+            receipt['exit_code']=log.observe(child,1800)
+        except BaseException as error:failure=error
+        finally:
+            receipt['cleanup_errors']=[]
+            if child is not None:
+                try:helper.cleanup_group(child,receipt)
+                except BaseException as error:receipt['cleanup_errors'].append('group:'+repr(error))
+            if log is not None:
+                try:log.finish()
+                except BaseException as error:receipt['cleanup_errors'].append('log:'+repr(error))
+                finally:receipt['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
+            elif child is not None and child.stdout is not None:
+                try:child.stdout.close()
+                except BaseException as error:receipt['cleanup_errors'].append('uninitialized pipe:'+repr(error))
+        require(failure is None and receipt.get('exit_code')==0 and receipt['cleanup_errors']==[],
+                'owned native replay/cleanup failed; no automatic retry')
+        report=metadata(args.new_output/'report.json')
+        require(report['proof']['status']=='PASS_PRIVATE_ORIGINAL27B_PURE_ROLES_ONLY'and
+                report['final_reserved']==0 and report['final_fenced']is False and
+                report['proof']['result']['compared_words']==validation['words'],
+                'all actual native raw outputs/input words exact and final original ledger0')
+        for row in expected:
+            pin=fixes.get(row['path'],row)
+            require(sha(worktree/row['path'])==pin['sha256'],'actual ROOT implementation unchanged after native replay')
+        for pin in metadata(HERE.parent/'freeze-manifest.json')['unchanged_dependency_source_pins']:
+            require(sha(worktree/pin['path'])==pin['sha256'],'original math/core/source dependencies unchanged after native replay')
+        require(frozen()==freeze and sha(args.binary)==args.binary_sha256 and
+                sha(args.reference_directory/'captured/manifest.json')==manifest_sha and
+                sha(args.reference_directory/'validation.json')==validation_sha,
+                'all preparation/binary/independent raw reference unchanged')
+        receipt.update(status='PASS_OWNED_ORIGINAL27B_PURE_ROLE_REPLAY_ONLY',report_sha256=sha(args.new_output/'report.json'))
+    except BaseException as error:
+        if failure is None:failure=error
+    finally:
+        try:os.close(fd)
+        except BaseException as error:
+            if failure is None:failure=error
+        receipt['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        if failure is not None:receipt.update(status='FAIL_UNQUALIFIED',failure=repr(failure))
+        write(args.new_output/'invocation.json',receipt)
+    require(receipt['status']=='PASS_OWNED_ORIGINAL27B_PURE_ROLE_REPLAY_ONLY','native proof unqualified; retained original attempt')
+
+
+if __name__=='__main__':main()
--- /dev/null
+++ run_reference.py
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
+from support import HERE, IMAGE, IMAGE_DIGEST, bounds, fd_sha, fingerprint, frozen, metadata, read, require, sha, write
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
+    require(total<=32*1024**2-128*1024,'bounded whole attempt with reserved completion metadata')
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
+    spec=importlib.util.spec_from_file_location('roles27_owned',HERE/'owned_container_v4.py')
+    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
+    nonce=uuid.uuid4().hex;owned_name='zg-original27-roles-'+nonce;cid=args.new_output/'owned.cid'
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
+               '--entrypoint','python3',IMAGE,'/probe/capture.py']
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
+        receipt['status']='EXECUTED_ORIGINAL27B_PURE_ROLES_REQUIRES_INDEPENDENT_VALIDATION'
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
+    require(receipt['status']=='EXECUTED_ORIGINAL27B_PURE_ROLES_REQUIRES_INDEPENDENT_VALIDATION',
+            'reference attempt unqualified; retained original evidence')
+
+if __name__=='__main__':main()
--- /dev/null
+++ source-plan.json
@@ -0,0 +1,622 @@
+{
+  "bounds": {
+    "CPU_threads": 4,
+    "artifact_bytes": 33554432,
+    "container_RAM_bytes": 12884901888,
+    "mapped_bytes": 3519025784,
+    "operator_working_allowance_bytes": 2147483648,
+    "page_size": 4096,
+    "payload_bytes": 4194304,
+    "process_AS_bytes": 12884901888,
+    "runtime_VA_allowance_bytes": 6442450944,
+    "whole_model_native_admission_qualified": false
+  },
+  "metadata": {
+    "config.json": {
+      "bytes": 4312,
+      "git_blob_verified": true,
+      "name": "config.json",
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+    },
+    "model.safetensors.index.json": {
+      "bytes": 112216,
+      "git_blob_verified": true,
+      "name": "model.safetensors.index.json",
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+    }
+  },
+  "prior_full_source_verification": {
+    "bytes": 7058,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-owned-window-20261020/source-pin.json",
+    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+  },
+  "prior_public_prefix": {
+    "bytes": 65417,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-typed-oracle-20261020/root-reference-1/captured/reference.json",
+    "sha256": "794765650ff40a52b821cebf85889cb46593b337ea79b94b50f4bad4d244e0aa"
+  },
+  "repository": "Qwen/Qwen3.8-27B",
+  "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
+  "roles": [
+    {
+      "absolute_offset": 1120792,
+      "bytes": 104857600,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 104860184,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
+      "output_elements": 10240,
+      "page_aligned_offset": 1118208,
+      "role": "GdnQkv",
+      "shape": [
+        10240,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    {
+      "absolute_offset": 105978392,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 62917144,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_z.weight",
+      "output_elements": 6144,
+      "page_aligned_offset": 105975808,
+      "role": "GdnGate",
+      "shape": [
+        6144,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    {
+      "absolute_offset": 137752,
+      "bytes": 491520,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 494104,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_a.weight",
+      "output_elements": 48,
+      "page_aligned_offset": 135168,
+      "role": "GdnAlpha",
+      "shape": [
+        48,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    {
+      "absolute_offset": 629272,
+      "bytes": 491520,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 494104,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_b.weight",
+      "output_elements": 48,
+      "page_aligned_offset": 626688,
+      "role": "GdnBeta",
+      "shape": [
+        48,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    {
+      "absolute_offset": 168893208,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "input_elements": 6144,
+      "input_provenance": "external_literal_BF16_pattern_v1_no_trained_producer_claim",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 62917400,
+      "name": "model.language_model.layers.0.linear_attn.out_proj.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 168890368,
+      "role": "GdnOutput",
+      "shape": [
+        5120,
+        6144
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    },
+    {
+      "absolute_offset": 2907880088,
+      "bytes": 125829120,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 3,
+      "mapping_bytes": 125831832,
+      "name": "model.language_model.layers.3.self_attn.q_proj.weight",
+      "output_elements": 12288,
+      "page_aligned_offset": 2907877376,
+      "role": "Query",
+      "shape": [
+        12288,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2712
+    },
+    {
+      "absolute_offset": 2834479256,
+      "bytes": 10485760,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 3,
+      "mapping_bytes": 10487960,
+      "name": "model.language_model.layers.3.self_attn.k_proj.weight",
+      "output_elements": 1024,
+      "page_aligned_offset": 2834477056,
+      "role": "Key",
+      "shape": [
+        1024,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2200
+    },
+    {
+      "absolute_offset": 3033709208,
+      "bytes": 10485760,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 3,
+      "mapping_bytes": 10488472,
+      "name": "model.language_model.layers.3.self_attn.v_proj.weight",
+      "output_elements": 1024,
+      "page_aligned_offset": 3033706496,
+      "role": "Value",
+      "shape": [
+        1024,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2712
+    },
+    {
+      "absolute_offset": 2844965016,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "input_elements": 6144,
+      "input_provenance": "external_literal_BF16_pattern_v1_no_trained_producer_claim",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 3,
+      "mapping_bytes": 62916760,
+      "name": "model.language_model.layers.3.self_attn.o_proj.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 2844962816,
+      "role": "AttentionOutput",
+      "shape": [
+        5120,
+        6144
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2200
+    },
+    {
+      "absolute_offset": 410065688,
+      "bytes": 178257920,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 178260760,
+      "name": "model.language_model.layers.0.mlp.gate_proj.weight",
+      "output_elements": 17408,
+      "page_aligned_offset": 410062848,
+      "role": "FfnGate",
+      "shape": [
+        17408,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    },
+    {
+      "absolute_offset": 588323608,
+      "bytes": 178257920,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 178260760,
+      "name": "model.language_model.layers.0.mlp.up_proj.weight",
+      "output_elements": 17408,
+      "page_aligned_offset": 588320768,
+      "role": "FfnUp",
+      "shape": [
+        17408,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    },
+    {
+      "absolute_offset": 231807768,
+      "bytes": 178257920,
+      "dtype": "BF16",
+      "input_elements": 17408,
+      "input_provenance": "external_literal_BF16_pattern_v1_no_trained_producer_claim",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 178260760,
+      "name": "model.language_model.layers.0.mlp.down_proj.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 231804928,
+      "role": "FfnDown",
+      "shape": [
+        5120,
+        17408
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    },
+    {
+      "absolute_offset": 1760,
+      "bytes": 2542796800,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": null,
+      "mapping_bytes": 2542798560,
+      "name": "lm_head.weight",
+      "output_elements": 248320,
+      "page_aligned_offset": 0,
+      "role": "Output",
+      "shape": [
+        248320,
+        5120
+      ],
+      "shard": "model-00018-of-00018.safetensors",
+      "view_offset": 1760
+    },
+    {
+      "absolute_offset": 45400,
+      "bytes": 10240,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_embedding_teacher_external_probe",
+      "input_stage": "embedding",
+      "kind": "ordinary_rms",
+      "layer": 0,
+      "mapping_bytes": 10584,
+      "name": "model.language_model.layers.0.input_layernorm.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 45056,
+      "role": "InputNorm",
+      "shape": [
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 344
+    },
+    {
+      "absolute_offset": 766581528,
+      "bytes": 10240,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_embedding_teacher_external_probe",
+      "input_stage": "embedding",
+      "kind": "ordinary_rms",
+      "layer": 0,
+      "mapping_bytes": 13080,
+      "name": "model.language_model.layers.0.post_attention_layernorm.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 766578688,
+      "role": "FfnNorm",
+      "shape": [
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    },
+    {
+      "absolute_offset": 3979553800,
+      "bytes": 10240,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_embedding_teacher_external_probe",
+      "input_stage": "embedding",
+      "kind": "ordinary_rms",
+      "layer": null,
+      "mapping_bytes": 13320,
+      "name": "model.language_model.norm.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 3979550720,
+      "role": "FinalNorm",
+      "shape": [
+        5120
+      ],
+      "shard": "model-00016-of-00018.safetensors",
+      "view_offset": 3080
+    }
+  ],
+  "runtime_source": {
+    "cache_utils.py": {
+      "bytes": 104677,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/cache_utils.py",
+      "sha256": "702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c"
+    },
+    "configuration_qwen3_5.py": {
+      "bytes": 8395,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/configuration_qwen3_5.py",
+      "sha256": "19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da"
+    },
+    "hub_kernels.py": {
+      "bytes": 44974,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/hub_kernels.py",
+      "sha256": "c1eafefe0cbdf7f0deaa21b1ef6cb6d7c6082b3f486651d069ff38fcf3e53c95"
+    },
+    "modeling_qwen3_5.py": {
+      "bytes": 95685,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/modeling_qwen3_5.py",
+      "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+    }
+  },
+  "schema": "zerogravity.original27B.pure-window-roles.public-oracle-plan.v1",
+  "scope": "Thirteen source-authentic BF16 pure projections and three original ordinary norms, two teacher-bound external probes. Hidden input from accepted actual original27B InputNorm0; nonhidden input literal nonzero finite BF16 probe, never a trained activation producer. No full model/session/state/exp/GPU/throughput claim.",
+  "selected_shards": {
+    "model-00001-of-00018.safetensors": {
+      "file": {
+        "bytes": 3966730552,
+        "git_blob_verified": false,
+        "name": "model-00001-of-00018.safetensors",
+        "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+      },
+      "header_bytes": 45392,
+      "header_sha256": "7bdeb03df566aa804dca2cb48e420908cb1864e5222858ad04918299883287e0"
+    },
+    "model-00016-of-00018.safetensors": {
+      "file": {
+        "bytes": 3979564040,
+        "git_blob_verified": false,
+        "name": "model-00016-of-00018.safetensors",
+        "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"
+      },
+      "header_bytes": 10048,
+      "header_sha256": "19bf5bf58f5b46c36817fb179af3d226d7394466a7905c8464a9c119581f18e0"
+    },
+    "model-00018-of-00018.safetensors": {
+      "file": {
+        "bytes": 3392197344,
+        "git_blob_verified": false,
+        "name": "model-00018-of-00018.safetensors",
+        "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"
+      },
+      "header_bytes": 1752,
+      "header_sha256": "2e6b71178b57a98de637fa669dedfaf35cbed1388914ffc93238bddd169eab54"
+    }
+  },
+  "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
+  "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
+  "source_files": [
+    {
+      "bytes": 11544,
+      "git_blob_verified": true,
+      "name": "LICENSE",
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
+    },
+    {
+      "bytes": 65012,
+      "git_blob_verified": true,
+      "name": "README.md",
+      "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
+    },
+    {
+      "bytes": 8952,
+      "git_blob_verified": true,
+      "name": "chat_template.jinja",
+      "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
+    },
+    {
+      "bytes": 4312,
+      "git_blob_verified": true,
+      "name": "config.json",
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+    },
+    {
+      "bytes": 238,
+      "git_blob_verified": true,
+      "name": "crc32.txt",
+      "sha256": "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83"
+    },
+    {
+      "bytes": 202,
+      "git_blob_verified": true,
+      "name": "generation_config.json",
+      "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
+    },
+    {
+      "bytes": 3353259,
+      "git_blob_verified": true,
+      "name": "merges.txt",
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
+    },
+    {
+      "bytes": 3966730552,
+      "git_blob_verified": false,
+      "name": "model-00001-of-00018.safetensors",
+      "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+    },
+    {
+      "bytes": 3043080328,
+      "git_blob_verified": false,
+      "name": "model-00002-of-00018.safetensors",
+      "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb"
+    },
+    {
+      "bytes": 2542796952,
+      "git_blob_verified": false,
+      "name": "model-00003-of-00018.safetensors",
+      "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"
+    },
+    {
+      "bytes": 3988973152,
+      "git_blob_verified": false,
+      "name": "model-00004-of-00018.safetensors",
+      "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a"
+    },
+    {
+      "bytes": 2099339864,
+      "git_blob_verified": false,
+      "name": "model-00005-of-00018.safetensors",
+      "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00006-of-00018.safetensors",
+      "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00007-of-00018.safetensors",
+      "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00008-of-00018.safetensors",
+      "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00009-of-00018.safetensors",
+      "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00010-of-00018.safetensors",
+      "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00011-of-00018.safetensors",
+      "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00012-of-00018.safetensors",
+      "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00013-of-00018.safetensors",
+      "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00014-of-00018.safetensors",
+      "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00015-of-00018.safetensors",
+      "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22"
+    },
+    {
+      "bytes": 3979564040,
+      "git_blob_verified": false,
+      "name": "model-00016-of-00018.safetensors",
+      "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00017-of-00018.safetensors",
+      "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8"
+    },
+    {
+      "bytes": 3392197344,
+      "git_blob_verified": false,
+      "name": "model-00018-of-00018.safetensors",
+      "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"
+    },
+    {
+      "bytes": 112216,
+      "git_blob_verified": true,
+      "name": "model.safetensors.index.json",
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+    },
+    {
+      "bytes": 390,
+      "git_blob_verified": true,
+      "name": "preprocessor_config.json",
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
+    },
+    {
+      "bytes": 12809320,
+      "git_blob_verified": false,
+      "name": "tokenizer.json",
+      "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
+    },
+    {
+      "bytes": 17928,
+      "git_blob_verified": true,
+      "name": "tokenizer_config.json",
+      "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
+    },
+    {
+      "bytes": 385,
+      "git_blob_verified": true,
+      "name": "video_preprocessor_config.json",
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
+    },
+    {
+      "bytes": 6722759,
+      "git_blob_verified": true,
+      "name": "vocab.json",
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
+    }
+  ],
+  "weight_bytes": 55563006776
+}
--- /dev/null
+++ support.py
@@ -0,0 +1,117 @@
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
+    limits=plan['bounds']
+    require(page==limits['page_size']==4096 and len(plan['roles'])==16, 'fixed actual role/page closure')
+    require(len({v['name']for v in plan['roles']})==16,'unique complete original role names')
+    mapped=0
+    for item in plan['roles']:
+        require(item['dtype']=='BF16' and item['kind'] in ('projection','ordinary_rms'), 'original role codec/kind')
+        size=1
+        for dim in item['shape']:
+            require(type(dim)is int and 0<dim<=248320,'bounded original role dimension');size*=dim
+        shape=item['shape']
+        require((item['kind']=='projection' and len(shape)==2 and item['input_elements']==shape[1] and
+                 item['output_elements']==shape[0] and 0<shape[1]<=17408)or
+                (item['kind']=='ordinary_rms' and shape==[5120] and
+                 item['input_elements']==item['output_elements']==5120),
+                'exact operation input/output geometry before library initialization')
+        require(item['bytes']==2*size and item['absolute_offset']>=8 and
+                item['absolute_offset']+item['bytes']<=plan['selected_shards'][item['shard']]['file']['bytes'],
+                'full exact original role extent')
+        begin=item['absolute_offset'];aligned=begin-begin%page
+        require(item['page_aligned_offset']==aligned and item['view_offset']==begin-aligned and
+                item['mapping_bytes']==item['bytes']+begin-aligned,'bounded role-only mapping')
+        mapped+=item['mapping_bytes']
+    require(mapped==limits['mapped_bytes'] and mapped+limits['runtime_VA_allowance_bytes']+
+            limits['operator_working_allowance_bytes']+limits['payload_bytes']<limits['process_AS_bytes'] and
+            limits['process_AS_bytes']==limits['container_RAM_bytes']==12*1024**3 and
+            limits['payload_bytes']==4*1024**2,'finite explicit containment, not native physical admission')
+    return {'mapped_bytes':mapped,'process_AS_bytes':limits['process_AS_bytes'],
+            'native_ResourceArbiter_admission_qualified':False}
+
+def frozen():
+    manifest=metadata(HERE/'prepared-manifest.json')
+    for row in manifest['files']:read(HERE/row['file'],pin=row)
+    for row in manifest['parent_metadata']:read(row['path'],pin=row)
+    return sha(HERE/'prepared-manifest.json')
--- /dev/null
+++ tensor_bits.py
@@ -0,0 +1,19 @@
+"""Bounded logical raw-bit transport only; no tensor/provider import on import."""
+import sys
+from support import require
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
+++ test_contracts.py
@@ -0,0 +1,34 @@
+"""ROOT-only source-free bounds fixtures; no tensor/model imports."""
+import copy
+import unittest
+from support import HERE,bounds,metadata
+
+
+class BoundsTests(unittest.TestCase):
+    def setUp(self):
+        self.plan=metadata(HERE/'source-plan.json')
+
+    def test_actual_16_header_role_plan_is_containment_only(self):
+        result=bounds(self.plan,4096)
+        self.assertEqual(result['mapped_bytes'],3519025784)
+        self.assertIs(result['native_ResourceArbiter_admission_qualified'],False)
+
+    def test_mapping_and_role_extent_corruption_refuses_before_libraries(self):
+        for key in ('mapping_bytes','view_offset','bytes'):
+            changed=copy.deepcopy(self.plan);changed['roles'][0][key]+=1
+            with self.subTest(key=key),self.assertRaises(ValueError):bounds(changed,4096)
+
+    def test_input_output_geometry_and_duplicate_role_refuse(self):
+        for key in ('input_elements','output_elements'):
+            changed=copy.deepcopy(self.plan);changed['roles'][0][key]+=1
+            with self.subTest(key=key),self.assertRaises(ValueError):bounds(changed,4096)
+        changed=copy.deepcopy(self.plan);changed['roles'][1]['name']=changed['roles'][0]['name']
+        with self.assertRaises(ValueError):bounds(changed,4096)
+
+    def test_unsupported_AS_and_page_geometry_refuse(self):
+        with self.assertRaises(ValueError):bounds(self.plan,8192)
+        changed=copy.deepcopy(self.plan);changed['bounds']['process_AS_bytes']-=1
+        with self.assertRaises(ValueError):bounds(changed,4096)
+
+
+if __name__=='__main__':unittest.main()
--- /dev/null
+++ validate_reference.py
@@ -0,0 +1,98 @@
+"""ROOT ONLY exact public raw-output/source/lifecycle validation, stdlib only."""
+import argparse
+import json
+import os
+from pathlib import Path
+import struct
+from support import HERE,MODEL_SHA,TORCH_GIT,bounds,digest,fd_sha,fingerprint,frozen,metadata,read,require,sha,source_header,unique,write
+
+
+def main():
+    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
+    require(args.output.is_absolute()and args.output.parent.resolve()==HERE,'owned original-reference directory')
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');inv=metadata(args.output/'invocation.json')
+    r=metadata(args.output/'captured/manifest.json')
+    require(inv['status']=='EXECUTED_ORIGINAL27B_PURE_ROLES_REQUIRES_INDEPENDENT_VALIDATION' and
+            inv['freeze_manifest_sha256']==freeze and inv['source_plan_sha256']==sha(HERE/'source-plan.json') and
+            inv['source_before']==inv['source_after'] and inv['source_FDs_closed'] is True and
+            inv['source_FD_cleanup_errors']==[] and inv['cleanup_errors']==[] and inv['exit_code']==0 and
+            inv['owned_container_absent_after_reap'] is True,'all original source/process lifetime complete')
+    g=inv['foreground_cleanup'];log=inv['log_cleanup']
+    require(g['leader_reaped'] is True and g['group_absent'] is True and g['all_nonzero_signals_before_reap'] is True and
+            g['stage_errors']==[] and g['kill_attempts_before_reap']==2 and
+            log['pipe_eof'] is log['all_closed'] is True and log['total_bytes']==log['retained_bytes']<=4*1024**2,
+            'complete owned leader/group/EOF cleanup, no truncated evidence')
+    expected={row['name']:(row['bytes'],row['sha256'])for row in plan['source_files']}
+    require(len(inv['source_before'])==len(expected) and
+            {row['name']:(row['bytes'],row['sha256'])for row in inv['source_before']}==expected,
+            'all original files/full18-shard provenance')
+    require(r['status']=='CAPTURED_ORIGINAL27B_PURE_OPERATOR_REFERENCE_REQUIRES_VALIDATION' and
+            r['source_plan_sha256']==sha(HERE/'source-plan.json') and r['source_repo']==plan['repository'] and
+            r['source_revision']==plan['revision'] and r['snapshot']==plan['snapshot'] and
+            r['torch']=='2.13.0+cu129' and r['torch_git']==TORCH_GIT and r['modeling_sha256']==MODEL_SHA and
+            r['affinity']==inv['expected_cpus'] and len(set(r['affinity']))==4 and
+            r['threads']==4 and r['interop_threads']==1 and r['pre_library_bounds']==bounds(plan,4096) and
+            r['all_original_source_and_mapped_bytes_unchanged'] is True and
+            r['source_owner_retained_until_process_exit'] is True and
+            r['whole_model_or_state_or_GPU_qualified'] is False,'actual original CPU/profile/role-only scope')
+    prefix=metadata(plan['prior_public_prefix']['path'],pin=plan['prior_public_prefix'])
+    require(len(r['cases'])==32 and len(plan['roles'])==16,'all16 roles/two teacher probes')
+    total=0;words=0;files={'manifest.json'}
+    for position,row in enumerate(r['cases']):
+        step,i=divmod(position,16);role=plan['roles'][i];token=(11,353)[step]
+        require(row['step']==step and row['teacher_token']==token and
+                all(row[key]==role[key]for key in ('role','layer','kind','input_provenance')) and
+                row['source_tensor']==role['name'] and row['source_shape']==role['shape'] and row['source_dtype']=='BF16' and
+                row['original_public_operator_calls']==2 and row['both_calls_full_bits_exact'] is True,
+                'exact original role/source/teacher/full-call closure')
+        payload={}
+        for key,count in [('input',role['input_elements']),('output',role['output_elements'])]:
+            pin=row[key];name=pin['file']
+            require(Path(name).name==name and name not in files and pin['bytes']==count*2,
+                    'unique complete BF16 payload extent')
+            raw=read(args.output/'captured'/name,2*1024**2,pin)
+            require(all(word&0x7f80!=0x7f80 for word, in struct.iter_unpack('<H',raw)),'all finite BF16 raw words')
+            payload[key]=raw;files.add(name);total+=len(raw);words+=count
+        if role['input_elements']==5120:
+            prior=prefix['steps'][step][role['input_stage']]
+            require(prefix['steps'][step]['token']==token and row['input_prior']==prior,'accepted external input provenance')
+            full=read(Path(plan['prior_public_prefix']['path']).parent/prior['file'],pin=prior)
+            require(len(full)==5120*4 and all(word&0xffff==0 for word, in struct.iter_unpack('<I',full)),
+                    'all accepted source input BF16-exact words')
+            actual=b''.join((word>>16).to_bytes(2,'little')for word, in struct.iter_unpack('<I',full))
+            require(actual==payload['input'] and digest(actual)==prior['bf16_raw_sha256'],
+                    'exact independent archived actual original input bytes')
+        else:
+            raw=b''.join(((0xbf00 if (j+token)%2 else 0x3f00)|((j*13+token)%128)).to_bytes(2,'little')
+                         for j in range(role['input_elements']))
+            require(row['input_prior'] is None and raw==payload['input'],'literal external BF16 probe, no trained producer')
+    require(total==r['payload_bytes']<=4*1024**2 and
+            {p.name for p in (args.output/'captured').iterdir()}==files,'all payload bytes/files accountable')
+    require(set(r['source_readsets'])=={v['name']for v in plan['roles']},'all16 original learned readsets')
+    for shard,pin in plan['selected_shards'].items():
+        fd=os.open(Path(plan['source_directory'])/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+        try:
+            before=fingerprint(os.fstat(fd));n,raw=source_header(fd,pin);head=json.loads(raw,object_pairs_hook=unique)
+            for role in plan['roles']:
+                if role['shard']!=shard:continue
+                header=head[role['name']]
+                require(header['shape']==role['shape'] and header['dtype']=='BF16' and
+                        8+n+header['data_offsets'][0]==role['absolute_offset'] and
+                        header['data_offsets'][1]-header['data_offsets'][0]==role['bytes'] and
+                        fd_sha(fd,role['absolute_offset'],role['bytes'])==r['source_readsets'][role['name']],
+                        'independent full original role/header/raw source-byte identity')
+            require(fingerprint(os.fstat(fd))==before,'original validation FD immutable')
+        finally:os.close(fd)
+    tree=0
+    for path in args.output.rglob('*'):
+        require(not path.is_symlink(),'no owned output symlink')
+        if path.is_file():tree+=path.stat().st_size
+    require(tree<=32*1024**2 and frozen()==freeze,'whole artifact and source/preparation closure')
+    write(args.output/'validation.json',{'status':'PASS_ORIGINAL27B_PURE_OPERATOR_REFERENCE_ONLY',
+        'manifest_sha256':sha(args.output/'captured/manifest.json'),'invocation_sha256':sha(args.output/'invocation.json'),
+        'freeze_sha256':freeze,'source_plan_sha256':sha(HERE/'source-plan.json'),'cases':32,'roles':16,
+        'words':words,'payload_bytes':total,'trained_activation_producer_for_all_roles_qualified':False,
+        'native_CPU_parity_full_model_StateRoot_GPU_or_performance_qualified':False})
+
+
+if __name__=='__main__':main()

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/qualification/metadata-v2/SOURCE_METADATA_FOLLOWUP.patch

Original UTF-8 bytes: 13722; SHA256 `5732c45adfe4a38312ae48b7b73c27ebff9ec8cdff5c65ee16e98896fafa4d8f`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ source-pin.json
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
+++ run_native_v2.py
@@ -0,0 +1,58 @@
+"""ROOT ONLY additive source-metadata binding; original native runner unchanged."""
+import argparse
+from pathlib import Path
+
+from support import HERE, frozen, metadata, read, require, sha
+
+LAYER = HERE / 'metadata-v2'
+ORIGINAL = '16809776089e1c3063629a0e0df556512a66b9ccd04d6812d56e47b744f4cb40'
+REPORT = '4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a'
+
+
+def closure():
+    require(frozen() == ORIGINAL, 'unchanged original thirteen-file preparation')
+    manifest = metadata(LAYER / 'prepared-manifest.json')
+    for item in manifest['files']:
+        read(HERE / item['file'], pin=item)
+    for item in manifest['parent_metadata']:
+        read(item['path'], pin=item)
+    require(sha(HERE / 'source-pin.json') == REPORT,
+            'exact accepted original source report; factory REPORT_SHA unchanged')
+    return sha(LAYER / 'prepared-manifest.json'), manifest
+
+
+def verify_build(build, layer_sha, manifest):
+    require(build['qualification_manifest_sha256'] == ORIGINAL and
+            build['qualification_metadata_followup_manifest_sha256'] == layer_sha and
+            Path(build['qualification_directory']).resolve() == HERE,
+            'actual ROOT compile uses this exact original qualification directory')
+    expected = {row['file']: row for row in manifest['native_compile_sources']}
+    actual = build['qualification_source_files']
+    require(type(actual) is list and len(actual) == len(expected),
+            'all actual native harness source/metadata bindings')
+    for row in actual:
+        require(type(row) is dict and row.get('file') in expected,
+                'no foreign or duplicate native compile source')
+        pin = expected.pop(row['file'])
+        require(row['bytes'] == pin['bytes'] and row['sha256'] == pin['sha256'],
+                'exact actual native harness source/metadata attestation')
+    require(not expected, 'complete native compile source/metadata set')
+
+
+def main():
+    parser = argparse.ArgumentParser(add_help=False)
+    parser.add_argument('--build-attestation', type=Path, required=True)
+    args, _ = parser.parse_known_args()
+    layer_sha, manifest = closure()
+    build_sha = sha(args.build_attestation)
+    verify_build(metadata(args.build_attestation), layer_sha, manifest)
+    import run_native as original
+    try:
+        original.main()
+    finally:
+        require(closure()[0] == layer_sha and sha(args.build_attestation) == build_sha,
+                'unchanged additive metadata and original native build binding')
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ test_native_metadata_v2.py
@@ -0,0 +1,44 @@
+"""Prepared source-free metadata negatives; ROOT alone executes."""
+import copy
+import unittest
+
+from run_native_v2 import HERE, ORIGINAL, verify_build
+
+
+class MetadataBinding(unittest.TestCase):
+    def fixture(self):
+        rows = [dict(file=name, bytes=index+1, sha256=str(index)*64)
+                for index, name in enumerate(('Cargo.toml', 'main.rs', 'factory.rs', 'source-pin.json'))]
+        manifest = dict(native_compile_sources=rows)
+        build = dict(qualification_manifest_sha256=ORIGINAL,
+                     qualification_metadata_followup_manifest_sha256='a'*64,
+                     qualification_directory=str(HERE), qualification_source_files=copy.deepcopy(rows))
+        return build, manifest
+
+    def test_complete_exact_four_sources(self):
+        build, manifest = self.fixture()
+        verify_build(build, 'a'*64, manifest)
+
+    def test_missing_or_duplicate_source_report_refuses(self):
+        for duplicate in (False, True):
+            build, manifest = self.fixture()
+            if duplicate:
+                build['qualification_source_files'][-1] = build['qualification_source_files'][0]
+            else:
+                build['qualification_source_files'].pop()
+            with self.assertRaises(ValueError):
+                verify_build(build, 'a'*64, manifest)
+
+    def test_changed_report_or_foreign_compile_directory_refuses(self):
+        for directory in (False, True):
+            build, manifest = self.fixture()
+            if directory:
+                build['qualification_directory'] = '/foreign-qualification'
+            else:
+                build['qualification_source_files'][-1]['sha256'] = 'f'*64
+            with self.assertRaises(ValueError):
+                verify_build(build, 'a'*64, manifest)
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ metadata-v2/ROOT.md
@@ -0,0 +1,26 @@
+# Additive native source metadata binding
+
+Author: spine_review. Source only; original thirteen-file16809776 unchanged.
+
+- Added qualification/source-pin.json: exact accepted7058B source-report4c74fa28.
+  Its path now matches unchanged factory.rs `CARGO_MANIFEST_DIR/source-pin.json`;
+  no source/model/weights/role or math mutation. Original factory REPORT_SHA stays.
+- Build the existing qualification/Cargo.toml from that exact directory. Include
+  four entries Cargo.toml/main.rs/factory.rs/source-pin.json in actual build
+  attestation qualification_source_files, plus qualification_directory,
+  qualification_manifest_sha25616809776 and
+  qualification_metadata_followup_manifest_sha256 from this layer. Preserve all
+  original implementation/compile/log/lock/binary evidence; labels are not proof.
+- Execute run_native_v2.py with the original run_native.py arguments. It validates
+  this additive layer and all four compile inputs before/after delegating the
+  unchanged original owner/ledger/full-word replay. ROOT runs three prepared
+  test_native_metadata_v2.py tests first; author AST only, no fixtures/runtime.
+
+The original12GiB AS/RAM cap stays. Mapped3,519,025,784B is exact;6GiB runtime and
+2GiB operator allowances are estimates, not a source-backed peak bound. Pinned
+Torch cf30153 Linear.cpp→LinearAlgebra.cpp maps this contiguous BF16[1,5120]
+head to CPUBlas(T,N,248320,1,5120). The noAVX512BF16 CPU64 route uses per-row
+BF16 inputs/FP32 register reduction, not a full5,085,593,600B F32 weight clone.
+Actual compiled route/provider/library/allocator/thread overhead and head peak
+are unqualified until ROOT executes under unchanged containment. No cap widening
+or public-operator replacement is authorized by these estimates.

````

## Source: .agents/runs/model-execution-root-bf16-composition-20261020/root-coefficient-output-size-adapter-2.patch

Original UTF-8 bytes: 988; SHA256 `035c17b5db5559b473249561dd5cf99428ef2f7239778f77ee4a59fd3f1ece8f`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs
@@ -28,7 +28,7 @@
             ));
         }
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
-        let workspace = self.prepare_output(None)?; // BEFORE both private copies.
+        let workspace = self.prepare_output(None, 2 * heads, 1)?; // BEFORE both private copies.
         let decode = |raw: &[u8]| {
             raw.chunks_exact(2)
                 .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
@@ -102,7 +102,7 @@
                     ));
                 }
             }
-            self.prepare_output(Some(alpha))? // Shared aggregate for ALL3 outputs.
+            self.prepare_output(Some(alpha), 3 * heads, 1)? // Shared aggregate for ALL3 outputs.
         };
         let prepared = match &self.source {
             HostSource::Loaded(owner) => {

````

## Source: .agents/runs/model-execution-root-bf16-composition-20261020/root-coefficient-test-array-style-3.patch

Original UTF-8 bytes: 805; SHA256 `30130924e17c8d8c08e0582c77dc2e8f46ec3e99342aa3a938036a0264588e3a`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs
+++ b/crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs
@@ -150,7 +150,7 @@
     #[test]
     fn raw_f32_dt_is_not_bf16_rounded_or_adjusted() {
         let word = 0x3380_0001u32;
-        let raw = vec![word.to_le_bytes(); 16].concat();
+        let raw = [word.to_le_bytes(); 16].concat();
         let zero = [0.0; 16];
         let out = prepare_gdn_coefficients(
             source(),
@@ -164,7 +164,7 @@
         )
         .unwrap();
         assert!(out.biased_alpha().iter().all(|v| v.to_bits() == word));
-        assert_eq!(raw, vec![word.to_le_bytes(); 16].concat());
+        assert_eq!(raw, [word.to_le_bytes(); 16].concat());
     }
     #[test]
     fn source_raw_dtype_span_and_nonfinite_refuse() {

````

## Source: .agents/runs/model-execution-root-bf16-composition-20261020/root-model-coefficient-test-array-style-5.patch

Original UTF-8 bytes: 362; SHA256 `10c3b4eec7d42cca192f9972c05d20e2ef88988c1f22a84d7d456395da015c6a`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
@@ -23,7 +23,7 @@
     )
 }
 fn raw(value: u16) -> Vec<u8> {
-    vec![value.to_le_bytes(); 16].concat()
+    [value.to_le_bytes(); 16].concat()
 }
 fn input<'a>(
     source: crate::accelerator::SourceIdentity,

````

## Source: .agents/runs/model-execution-root-bf16-composition-20261020/root-model-fixture-contract-reconciliation-6.patch

Original UTF-8 bytes: 3747; SHA256 `8ad1b36e65f521621630de8018edea13857e74a75e7a2ee97e351e8f870788f7`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs
@@ -9,8 +9,9 @@
 fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {
     let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
     let text = &mut cfg["text_config"];
-    text["num_hidden_layers"] = json!(1);
-    text["layer_types"] = json!(["linear_attention"]);
+    text["num_hidden_layers"] = json!(2);
+    text["full_attention_interval"] = json!(2);
+    text["layer_types"] = json!(["linear_attention", "full_attention"]);
     text["hidden_size"] = json!(256);
     text["linear_num_key_heads"] = json!(16);
     text["linear_num_value_heads"] = json!(48);
--- a/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs
@@ -14,8 +14,9 @@
 ) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {
     let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
     let text = &mut cfg["text_config"];
-    text["num_hidden_layers"] = json!(1);
-    text["layer_types"] = json!(["linear_attention"]);
+    text["num_hidden_layers"] = json!(2);
+    text["full_attention_interval"] = json!(2);
+    text["layer_types"] = json!(["linear_attention", "full_attention"]);
     text["hidden_size"] = json!(256);
     text["linear_num_key_heads"] = json!(16);
     text["linear_num_value_heads"] = json!(48);
--- a/crates/zerogravity-model-qwen35/src/attention_slice_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/attention_slice_tests.rs
@@ -374,7 +374,7 @@
     assert!(
         math.project_profiled(
             operators
-                .matrix_descriptor(QwenMatrixRole::FfnGate(3))
+                .matrix_descriptor(QwenMatrixRole::Embedding)
                 .unwrap(),
             &imported,
             ProjectionContract::TorchCpuBf16Avx2
--- a/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
@@ -129,12 +129,18 @@
             .is_err()
     );
     assert_eq!(reserved(&admission), before);
-    let remaining = admission
-        .arbiter
-        .snapshot(&admission.resource)
-        .unwrap()
-        .remaining;
-    let pressure = source.reserve_workspace(remaining as usize).unwrap();
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
+        pressure.push(source.reserve_workspace(remaining.min(32 * 1024 * 1024) as usize).unwrap());
+    }
     let pressured = reserved(&admission);
     assert!(
         operators
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs
@@ -328,12 +328,12 @@
 }
 
 #[test]
-fn same_content_foreign_window_issuer_cannot_rebind_a_retained_input() {
+fn separately_opened_same_content_source_cannot_rebind_a_retained_input() {
     let fixture = Fixture::new();
     let loaded = source_with_options(&fixture, false, 256, true, |_| {});
     let (source_a, admission_a) = windows(&fixture, &loaded);
     let (source_b, admission_b) = windows(&fixture, &loaded);
-    assert_eq!(source_a.source_identity(), source_b.source_identity());
+    assert_ne!(source_a.source_identity(), source_b.source_identity());
     assert!(!source_a.same_issuer(&source_b));
     let operators_a = bind_windows(source_a.clone(), &loaded);
     let operators_b = bind_windows(source_b.clone(), &loaded);

````

## Source: .agents/runs/model-execution-root-bf16-composition-20261020/root-model-fixture-format-6.patch

Original UTF-8 bytes: 558; SHA256 `252c4e4d6bfffdc2267086341fdeeaf877be8489003fd68449599ce9dc987c19`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
@@ -139,7 +139,11 @@
         if remaining == 0 {
             break;
         }
-        pressure.push(source.reserve_workspace(remaining.min(32 * 1024 * 1024) as usize).unwrap());
+        pressure.push(
+            source
+                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
+                .unwrap(),
+        );
     }
     let pressured = reserved(&admission);
     assert!(

````

## Source: .agents/runs/model-execution-root-bf16-composition-20261020/root-model-tied-role-parameter-lifetime-fixture-7.patch

Original UTF-8 bytes: 1527; SHA256 `d83a5e9564e7420c400422f69ae86b07fd9e7ad015046c91d6811be3c50d1632`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/attention_slice_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/attention_slice_tests.rs
@@ -371,15 +371,21 @@
         math.ordinary_rms_profiled(gdn_gamma, &imported, OrdinaryRmsContract::TorchCpuBf16Avx2)
             .is_err()
     );
-    assert!(
-        math.project_profiled(
-            operators
-                .matrix_descriptor(QwenMatrixRole::Embedding)
+    // Tied output and embedding share a descriptor; the explicit role API
+    // rejects Embedding before importing another private value.
+    assert!(
+        operators
+            .execute_projection_slice(
+                QwenMatrixRole::Embedding,
+                crate::operator_slice::pure::PureSliceInput::from_bf16_bytes(
+                    source.source_identity(),
+                    ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+                    &bytes,
+                )
                 .unwrap(),
-            &imported,
-            ProjectionContract::TorchCpuBf16Avx2
-        )
-        .is_err()
+                None,
+            )
+            .is_err()
     );
     assert!(
         math.ordinary_rms_profiled(
--- a/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs
@@ -323,6 +323,7 @@
     same(expected.output_bf16(), result.output_bf16().unwrap());
     drop(expected);
     drop(gamma);
+    drop(parameter);
     let bits = result
         .output_bf16()
         .unwrap()

````

## Source: .agents/runs/model-execution-root-bf16-composition-20261020/root-window-test-visibility-import-4.patch

Original UTF-8 bytes: 1201; SHA256 `11e8fe4b19d5f18873a289db0dabd5a69a02bd2fa66ae8c3a10faf73e65e236f`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs
@@ -249,10 +249,9 @@
             .is_err()
     );
     assert_eq!(reserved(&host), before);
-    let input_need = (crate::host_math::gated_rms::INPUT_BYTES
-        + crate::host_math::gated_rms::INPUT_SCRATCH) as u64;
-    let output_need = (crate::host_math::gated_rms::OUTPUT_BYTES
-        + crate::host_math::gated_rms::OUTPUT_SCRATCH) as u64;
+    // Independent declared H48 payloads and conservative scratch floors.
+    let input_need = (2 * 48 * 128 * 4 + 64 * 1024) as u64;
+    let output_need = (48 * 128 * 4 + 960 * 1024) as u64;
     let pressure = host
         .arbiter
         .admit(
--- a/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs
@@ -3,7 +3,7 @@
 mod source_gamma;
 
 use super::*;
-use crate::accelerator::program::{CachedGdnContract, CachedGdnOperands};
+use crate::accelerator::program::CachedGdnContract;
 use crate::operator_slice::cached_gdn::CachedGdnSliceInput;
 use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
 

````

## Source: .agents/runs/model-execution-vllm-gemm-operands-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 59596; SHA256 `38d1c38a23c9215fcee92fee49cfa8844f3e85feb4701260316b688dc51ea2a5`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git .agents/runs/model-execution-vllm-gemm-operands-20261020/README.md .agents/runs/model-execution-vllm-gemm-operands-20261020/README.md
new file mode 100644
index 0000000..b8ca672
--- /dev/null
+++ .agents/runs/model-execution-vllm-gemm-operands-20261020/README.md
@@ -0,0 +1,109 @@
+# Authentic original-HF CUDA GEMM operand extract
+
+Parent: `.agents/plans/model-execution/TODO.md`. Owner: `/root/batch_design_review`.
+SOURCE PREPARATION ONLY. ROOT executes tests/export/validation. No tracked source,
+old oracle, native provider or shared plan change. No author Torch/model/GPU/
+operator/container/runtime execution or actual raw/DSO payload read/hash.
+
+The accepted genuine vLLM0.31.0 eager-B1 capture already retains all required
+original public-hook BF16 operands/results for teachers11→353. No new model run,
+observer rebuild, transfer or tensor kernel is needed. New extraction is stdlib
+file-byte transport over that immutable artifact, not a numerical oracle.
+
+## Authority and preserved gates
+
+Exact accepted parent: `root-probe-v4-1` under the original GEMM observation run.
+Its actual `gemm-validation.json` SHA is
+`55d68c7a90ac766b97743d2fa8c936d9044c04df8f7611207736d3703c5588be`.
+`bindings.json` pins49 existing source files (including all four source freezes),
+20 actual metadata/trace files, and expected raw/observer identities copied from
+the genuine reports. ROOT hashes the raw/build payloads; author did not read them.
+
+- Re-execute **unchanged source** of both accepted stdlib validators. Definitions
+  load from pinned source bytes, not bytecode caches. Only their two metadata
+  `write` bindings are replaced with a bounded collector: requested paths must be
+  exactly the old two report paths; each result must equal the authenticated
+  accepted report. No old path is opened for writing. All original guards,
+  comparisons, constants and returned results remain unchanged.
+- Original full496640 raw pre-sampler logit words stay plain/observed byte-exact;
+  all320 actual source/cast audits before/after, actual runtime/source/image,
+  four genuine GEMM metadata/pointer/stream calls, full mapped provider hashes,
+  trace health, original public settings, and repaired owned-container cleanup
+  must still pass. Existing raw and trace archives are verified before/after.
+- All four call keys/phase/order bind `qkvz0,ba0,qkvz1,ba1`, with actual input,
+  learned matrix and returned output pointers/storage plus original PID/TID/
+  stream. This closed extract additionally requires the authentic observed Ex
+  settings: transA1/B0, BF16 A/B/C14, compute68, requested algo99, F32 alpha1/
+  beta0, math/atomics/pointer0, stream0, explicit workspace8519680B. No hidden
+  selected kernel or algorithm is inferred from that public requested enum.
+- Original raw archive record **index + label + call + source metadata + byte
+  offset/count/hash** closes each extracted row to these actual hook instances.
+  Metadata labels or source-free fixtures alone never authenticate actual values.
+
+## Actual hook ordering
+
+Pinned `gemm_observer.py::ZgGemmExtension.zg_oracle_install` calls the original
+base install first. Base registers its raw forward hook; GEMM extension then
+registers the public metadata pre-hook and a later forward hook. For each genuine
+call: metadata pre-hook records original args[0]/weight pointers and begins the
+public phase; stock projection executes unchanged; the earlier base hook copies
+that exact args[0]/returned output[0]; the later GEMM hook ends the phase and
+records that same output tuple pointer. Every hook returns None. No arithmetic,
+tensor/module forward, sampler, learned weight or route was replaced.
+
+The original raw transfer uses detach→blocking `.to(cpu)`→contiguous→uint8 bytes.
+Its synchronization/transfer overhead is diagnostic and already covered by the
+plain/observed raw-logit guard; it is not a timing/default-graph qualification.
+The new extract performs **zero new GPU/model/operator calls or D2H transfers**.
+The original C interposer only observes public metadata/pointer extents; it never
+reads device-pointer contents. No interposer/worker/launcher bytes are modified.
+
+## Compact values and learned row layout
+
+Six files, exactly36992B:
+
+- Each teacher: one2048B authentic BF16 input shared by QKVZ/BA, one16384B QKVZ
+  result, one64B BA result. Signed zero/subnormal and all other bits preserved;
+  require every value finite, without dtype conversions or reconstruction.
+- QKVZ full raw learned matrix8192×1024: source `in_proj_qkv.weight` occupies
+  rows0..6144; source `in_proj_z.weight` rows6144..8192.
+- BA full raw learned matrix32×1024: source `in_proj_b.weight` rows0..16;
+  source `in_proj_a.weight` rows16..32. It is **b,a**, not lexical source order.
+- Full original source descriptors/names/headers, actual packed rows/strides,
+  public mapper chain and source/loaded audit hashes remain attached. Stream-hash
+  each existing packed-weight archive subrange against its corresponding actual
+  source audit. Do not copy/export the existing16MiB QKVZ weight again.
+
+Load profile is the actual CUDA vLLM profile:284 sourceBF16→loadedBF16;
+18 A_log sourceF32→loadedF32;18 GDN gamma sourceF32→loadedBF16 explicitly. This
+is distinct from the accepted restored-F32 TorchCPU profile. No silent cast36,
+CPU reducer inference or wider source/profile eligibility follows.
+
+## ROOT-only commands
+
+Run only after independent source review and the CPU/GPU quiet phase. Six Python
+files are AST-parsed only by author; ten source-free test methods are prepared,
+not executed. Tests exercise file range bits, signed zero/subnormals/nonfinite,
+source/load profile, packed order, raw keys, call/stream/pointer/settings changes,
+and actual read-only writer interception using explicitly synthetic definitions.
+They are not model/operator/raw-value oracle evidence.
+
+```sh
+run=.agents/runs/model-execution-vllm-gemm-operands-20261020
+python3 "$run/test_contracts.py"
+python3 "$run/export_operands.py" --new-output "$PWD/$run/root-extract-1"
+python3 "$run/validate_extract.py" --extract "$PWD/$run/root-extract-1" \
+  --new-report "$PWD/$run/root-extract-validation-1.json"
+```
+
+Export and validator restrict their own process to four available CPUs,
+512MiB address space,120CPU-seconds and180s alarm. They spawn no children, load
+no provider/runtime, and only read originals. New outputs are exclusive0600 in
+fresh0700 directories; failures never authorize overwrite/retry or old receipt
+rewrites. A validator report belongs outside the closed extract directory.
+
+Remaining gates: ROOT source review/test/export + independent raw binding
+validation; native original-actor provider/internal allocation/workspace admission;
+real native BA first, then QKVZ operator replay. Source/queue/leases/completion/
+quarantine must remain original. Full all42/logits native-HF, cached/initial FLA,
+wider/mixed batches/default graphs and matched performance remain unqualified.
diff --git .agents/runs/model-execution-vllm-gemm-operands-20261020/TODO.md .agents/runs/model-execution-vllm-gemm-operands-20261020/TODO.md
new file mode 100644
index 0000000..45360ec
--- /dev/null
+++ .agents/runs/model-execution-vllm-gemm-operands-20261020/TODO.md
@@ -0,0 +1,25 @@
+# Authentic CUDA GEMM operand capsule
+
+Parent: `.agents/plans/model-execution/TODO.md`. Owner: `/root/batch_design_review`.
+Owned scope: this new ignored run only; no original source/captures or MAIN edits.
+
+- [x] Inspect pinned original hook order and actual accepted four-call metadata.
+  Existing archive already contains genuine BF16 input/output rows; no rerun.
+- [x] Prepare strict49-source/20-metadata/3-payload identity closure; payload
+  identities copied from original genuine reports, not read/hashed by author.
+- [x] Prepare unchanged accepted full validator delegation with only metadata
+  write interception; require both results equal authenticated originals.
+- [x] Prepare36992B compact row export, actual call/pointer/stream/settings/raw
+  key/index binding; source-audit packed-weight subrange hashes, no weight copy.
+- [x] Preserve actual284/18/18 CUDA load-cast profile, BA b,a/QKVZ qkv,z layout;
+  all old source/runtime/fullV/320audit/provider/cleanup gates stay mandatory.
+- [x] Prepare10 source-free tests; Python AST only, no tests executed by author.
+- [x] Freeze source/patch/docs for independent review. Originals stay unchanged.
+- [ ] ROOT independent review and source-free tests.
+- [ ] ROOT fresh bounded read-only export and independent validation.
+- [ ] Original-actor native provider/internal workspace/allocation admission.
+- [ ] ROOT physical native BA then QKVZ replay against authentic capsule words.
+
+No native provider/math implementation, full-HF/session/profile/graph/wider-batch
+or performance eligibility. No author Torch/model/GPU/operator/runtime/container
+execution or actual source/model/DSO/raw payload hash/read.
diff --git .agents/runs/model-execution-vllm-gemm-operands-20261020/accepted.py .agents/runs/model-execution-vllm-gemm-operands-20261020/accepted.py
new file mode 100644
index 0000000..6525ed6
--- /dev/null
+++ .agents/runs/model-execution-vllm-gemm-operands-20261020/accepted.py
@@ -0,0 +1,56 @@
+"""Read-only delegation to unchanged pinned validators, metadata writes retained locally."""
+import sys
+import types
+from helpers import bounded, digest, jread, require
+
+
+def authenticate(bindings):
+    from pathlib import Path
+    base = Path(bindings['accepted_source_directory'])
+    capture = Path(bindings['accepted_capture_directory'])
+    for name, pin in bindings['source_files'].items():
+        require(digest(base / name) == pin, 'unchanged accepted source: ' + name)
+    for name, pin in bindings['metadata_files'].items():
+        require(digest(capture / name) == pin, 'unchanged accepted metadata: ' + name)
+    # Raw data/build DSO are ROOT-validated from prior authenticated reports;
+    # the preparer never hashes or opens those actual payloads.
+    for name, pin in bindings['payload_files'].items():
+        require(digest(capture / name) == pin, 'actual authenticated accepted payload: ' + name)
+    return base, capture
+
+
+def unchanged_revalidation(base, capture):
+    expected = {capture / 'validation.json': jread(capture / 'validation.json'),
+                capture / 'gemm-validation.json': jread(capture / 'gemm-validation.json')}
+    retained = {}
+
+    def collect(path, value):
+        # The only intercepted function writes metadata. No comparison, guard,
+        # tensor/operator/library path or acceptance output is altered.
+        require(path in expected and path not in retained, 'only two exact accepted metadata write sites')
+        require(value == expected[path], 'unchanged full validator result must equal authenticated actual acceptance')
+        retained[path] = value
+
+    previous = {name: sys.modules.get(name) for name in ['support', 'validate', '_zg_accepted_gemm_v4']}
+    modules = {}
+    try:
+        for name, file in [('support', 'support.py'), ('validate', 'validate.py'),
+                           ('_zg_accepted_gemm_v4', 'validate_gemm_v4.py')]:
+            module = types.ModuleType(name)
+            module.__file__ = str(base / file)
+            sys.modules[name] = module
+            exec(compile(bounded(base / file), str(base / file), 'exec'), module.__dict__)
+            modules[name] = module
+            if name != 'support':
+                module.write = collect
+        result = modules['_zg_accepted_gemm_v4'].validate(capture)
+        require(set(retained) == set(expected) and result == expected[capture / 'gemm-validation.json'],
+                'both unchanged accepted validators ran nonvacuously; originals never written')
+        return result, modules['_zg_accepted_gemm_v4'].trace_validate(
+            capture / 'observed', jread(capture / 'observed/gemm-report.json'))
+    finally:
+        for name, old in previous.items():
+            if old is None:
+                sys.modules.pop(name, None)
+            else:
+                sys.modules[name] = old
diff --git .agents/runs/model-execution-vllm-gemm-operands-20261020/bindings.json .agents/runs/model-execution-vllm-gemm-operands-20261020/bindings.json
new file mode 100644
index 0000000..984391b
--- /dev/null
+++ .agents/runs/model-execution-vllm-gemm-operands-20261020/bindings.json
@@ -0,0 +1,304 @@
+{
+  "status": "SOURCE_ONLY_BINDINGS_PREPARED_PAYLOAD_UNREAD_BY_AUTHOR",
+  "accepted_source_directory": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-vllm-gemm-observation-20261020",
+  "accepted_capture_directory": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-vllm-gemm-observation-20261020/root-probe-v4-1",
+  "source_files": {
+    "freeze-manifest.json": {
+      "bytes": 2928,
+      "sha256": "b08a3e46bb2f98d02d0b58f5cd412f2bac974c9ee4d7916e3ad3459f43a1350d"
+    },
+    "README.md": {
+      "bytes": 6499,
+      "sha256": "d5522020a2e3b90b0c50a65a3935b3b6435992c70571e3f94d74c59374920582"
+    },
+    "TODO.md": {
+      "bytes": 2411,
+      "sha256": "b8c07ca8f975f68c735aa7822f5c2ea99027d86a70c7e241721d04a676e30695"
+    },
+    "build_inside.py": {
+      "bytes": 2774,
+      "sha256": "735c4f1bb1e116736d93afabc07bcebcc9ba0c1bd6e836708c92aba43d770f18"
+    },
+    "elf-public-abi.json": {
+      "bytes": 9532,
+      "sha256": "d0519373b782bebec5c1b46302b47c0afeb6c88f5929472ca02c9b3e3a72482f"
+    },
+    "elf_public_abi.py": {
+      "bytes": 5938,
+      "sha256": "85f50f1368a310c8a38aa09651342696e7b7691b6cedae4b2d08115d61ca19b9"
+    },
+    "gemm-pins.json": {
+      "bytes": 3251,
+      "sha256": "11d036e890464b6d20796cc9936bc60150233b87530b8aa1e5045a1c4d1de46f"
+    },
+    "gemm_observer.py": {
+      "bytes": 9568,
+      "sha256": "21f97ec230059db0e2605862424c7d2193d6bf6d174dd51e434afe2986c03b22"
+    },
+    "interposer.cpp": {
+      "bytes": 18015,
+      "sha256": "b81caf1aa3f4f62fcc59007e7b9119bcc6b7ce0e5b2f8b7bd89ca4940f94e423"
+    },
+    "interposer.map": {
+      "bytes": 289,
+      "sha256": "35fd4363f49d3391ccf1a6a31f4de683f02d13ec4011a6c9ab9862e0fd148a8c"
+    },
+    "owned_container_v3.py": {
+      "bytes": 9859,
+      "sha256": "64bd4f6e622285d8fb8524f15f0c19107940f4cec70530403a70a32d454ebd01"
+    },
+    "public_forward_probe.py": {
+      "bytes": 6306,
+      "sha256": "4c6688dcff5561eedef9bd12665f92b9651a1cebf3b0b34c5639108487312595"
+    },
+    "run_probe.py": {
+      "bytes": 7547,
+      "sha256": "4a16e359ea85b6fbfd7030cb8d24c8f5cccb7dd3f210ae149ac29d6858b56e68"
+    },
+    "source-pins.json": {
+      "bytes": 16361,
+      "sha256": "2f0ceb1831ff8fae7d165c82c38cf04f2917170b5da7a90586487cba08b6acd0"
+    },
+    "support.py": {
+      "bytes": 4736,
+      "sha256": "62c1a358c004db7bcbfe8b5e10878404dca12f961de6dcd597aa3f836143b39c"
+    },
+    "test_validate_gemm.py": {
+      "bytes": 3025,
+      "sha256": "30a5daeb984a5a40e0c6c2cc9d42e9d8e6b81aaa8575d75c3fa25c272ca7e8f3"
+    },
+    "validate.py": {
+      "bytes": 10889,
+      "sha256": "eb7f3dab178bb3e0d13bc6036ff5eb43362225b878fd8137e2de92ada2f6ba34"
+    },
+    "validate_gemm.py": {
+      "bytes": 16719,
+      "sha256": "90fe88a9d927e9c22b974c8c183f86b3591350e9642a9eab5a419c10637d6876"
+    },
+    "worker_observer.py": {
+      "bytes": 17241,
+      "sha256": "9b09b602c948759d80595389c97e00c35bb315002fdca6d9241e08db2ba6ce7e"
+    },
+    "freeze-manifest-v2.json": {
+      "bytes": 1951,
+      "sha256": "248cde267dbe9ecbdc552dd18a91af507f5faa2f7711ebe238cdf1e61b64a3dc"
+    },
+    "interposer_v2.cpp": {
+      "bytes": 18243,
+      "sha256": "9c83eb61799f85d28b9a40ec07291671d28769c6e48f0b39f7259e2029b6e856"
+    },
+    "build_inside_v2.py": {
+      "bytes": 2783,
+      "sha256": "1e1065b39b398f8ed8f1c518563f1ff1a5f376f5a096a10e02851732baa201a9"
+    },
+    "owned_container_v4.py": {
+      "bytes": 11840,
+      "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+    },
+    "run_probe_v2.py": {
+      "bytes": 8400,
+      "sha256": "1c8b074b768b78eacb1820258728c5bc14a39537550a2a37bbb8fa80680aaec5"
+    },
+    "validate_gemm_v2.py": {
+      "bytes": 18128,
+      "sha256": "4833df7521a3676c4c9877b62370c9ef90891100bb94ae1761c3602b47ef3989"
+    },
+    "test_cleanup_v4.py": {
+      "bytes": 5493,
+      "sha256": "015bab0edef82469eb6a539cfbddd67e915a4674e7923b54627fac54c75d9d41"
+    },
+    "test_validate_gemm_v2.py": {
+      "bytes": 3657,
+      "sha256": "432256ecd08128cfb8ea76bc3732f4e293827dd23f6cc7882eb4a3e9957e7032"
+    },
+    "README_V2.md": {
+      "bytes": 4956,
+      "sha256": "55b82efe660e87f81cba3b4a14f600f7563fa5cf9e41f0c2d8e506bbe297d669"
+    },
+    "TODO_V2.md": {
+      "bytes": 1519,
+      "sha256": "9dd8d683f45e1f3576181af5cd8bbd2914df9400fa1e9cd51590583dc5087f8b"
+    },
+    "REPAIR_DIFFS_V2.patch": {
+      "bytes": 17791,
+      "sha256": "b5fe17a0ae849603027bdbbb4133da4f9a8ef9c7be05e705606759fc2c619ec6"
+    },
+    "ADDITIVE_V2.patch": {
+      "bytes": 76915,
+      "sha256": "fcc093da0a4933e175f43a62a81e0fa66ea290fd4775b0359f735b318c9fdf9f"
+    },
+    "freeze-manifest-v3.json": {
+      "bytes": 1342,
+      "sha256": "256be8610a012700c10353e1f382fcf97b7278b5d3d5cd2bb241f96bde98f37c"
+    },
+    "interposer_v3.cpp": {
+      "bytes": 18297,
+      "sha256": "cde48d1401b96bd6ca75cd860e30ef702abf3e44a54f1a43e55a042e99339787"
+    },
+    "build_inside_v3.py": {
+      "bytes": 2783,
+      "sha256": "12b8781b47ee83b92f40f23a9ac5a696923e73cbbe9cb66a851a190bb582aa97"
+    },
+    "run_probe_v3.py": {
+      "bytes": 8829,
+      "sha256": "8b5833b765fbc9eebfd2ef3fc12166bcf0f67e4eed2d8f48786e90206bd98472"
+    },
+    "validate_gemm_v3.py": {
+      "bytes": 18128,
+      "sha256": "dc96d4a8474535d08ae1edc401dd717ee0da2ddeb266d0b81fb841be484d9d05"
+    },
+    "README_V3.md": {
+      "bytes": 2719,
+      "sha256": "49b75766f8a045368daa82840886a27051cf2b3ec3619fb94c34c1cb8fb2b0ab"
+    },
+    "REPAIR_DIFFS_V3.patch": {
+      "bytes": 6205,
+      "sha256": "cba6cc619e25b8ab1d539a068a5455848fc9422262c7187ae9c1e4f63e2c0132"
+    },
+    "ADDITIVE_V3.patch": {
+      "bytes": 51899,
+      "sha256": "ac5fd30c6d91155d0170dba40c1fbf83f2203428e7fd0105b51a502c672d8c21"
+    },
+    "freeze-manifest-v4.json": {
+      "bytes": 1707,
+      "sha256": "d7a240eef382ab5046ddeba529eb0275950fc888b944371ca05911e2b4536565"
+    },
+    "interposer_v4.cpp": {
+      "bytes": 21240,
+      "sha256": "07a10e650bd6efbb982b7f2509a2b6b610b90cf1f3092722e3ac1e4076aaa073"
+    },
+    "build_inside_v4.py": {
+      "bytes": 3585,
+      "sha256": "0c21e01b13104dace47a60d9df9f0c56877156a418232de09d0246de4dc599f6"
+    },
+    "run_probe_v4.py": {
+      "bytes": 9345,
+      "sha256": "d60bc29c09cdb6f81a41535cf04ee8afa1e47e768cb25d08d46d7ee2c9d2290d"
+    },
+    "validate_gemm_v4.py": {
+      "bytes": 19193,
+      "sha256": "9ed01e9d35651cbc45e7c5642a3383bbe385269b638f02c1bdccf7d10ce9cdab"
+    },
+    "loader-scope-v4.json": {
+      "bytes": 7358,
+      "sha256": "aba1cfee62457aaa5bd33bb091505484b1b3b1041086ef68fe9582b9138f26d8"
+    },
+    "resolver-pins-v4.json": {
+      "bytes": 2038,
+      "sha256": "908bf996d291f24bf1cf31a3062c9573a8e5c30539b734774d039c0c3ecf5116"
+    },
+    "README_V4.md": {
+      "bytes": 3882,
+      "sha256": "3b83b6db6c4e29daf2a0fa501b06af84306a7459236417302d8fcc5fa3ebb36a"
+    },
+    "REPAIR_DIFFS_V4.patch": {
+      "bytes": 14637,
+      "sha256": "fa181bfedd3d6ab3286534b2db3e86be83f738fd69b30482ccdfe9030c975779"
+    },
+    "ADDITIVE_V4.patch": {
+      "bytes": 83121,
+      "sha256": "ce3e15b6e36347c13179e214ea02215643057b3c1dd589e9a902bc6143948ed9"
+    }
+  },
+  "metadata_files": {
+    "validation.json": {
+      "bytes": 711,
+      "sha256": "2fa80108c827471e078c6f53adc2e2f516bd8c73049305190b6d66cdf9d9e7eb"
+    },
+    "gemm-validation.json": {
+      "bytes": 1507,
+      "sha256": "55d68c7a90ac766b97743d2fa8c936d9044c04df8f7611207736d3703c5588be"
+    },
+    "image-inspect.json": {
+      "bytes": 59681,
+      "sha256": "487f85ed21d73c87c4742f1e5efdcee684cead82222c66aea64f56194d93f183"
+    },
+    "build/invocation.json": {
+      "bytes": 5409,
+      "sha256": "6705f7f598cc0acb340c91a9c43015f6d959178c55f409db0dcb066ffbfab349"
+    },
+    "build/build-result.json": {
+      "bytes": 1207,
+      "sha256": "c3bed9520c6f2ab718df83464ca7874664e9de88d5421734457e469cfbe1a80b"
+    },
+    "plain/invocation.json": {
+      "bytes": 5476,
+      "sha256": "6d9fb1143e4d74adff7c74aa25b5421686b6bab2bcc7bc52bccc9f731f73f637"
+    },
+    "plain/runtime.json": {
+      "bytes": 18139,
+      "sha256": "b5dc605b5a34ef65f1788d1a768c45e3d50ceb222ee94a350880eb11fbd521b3"
+    },
+    "plain/runtime-identities.json": {
+      "bytes": 236,
+      "sha256": "19e7e7f5e341bd380e442a26a03ef14cc431268b428a54de5ea7a809bcaaee85"
+    },
+    "plain/requested-config.json": {
+      "bytes": 602,
+      "sha256": "8786527d9080dbae0f2be13bb65a1671217a060c04e1ba39f00449178c163892"
+    },
+    "plain/result.json": {
+      "bytes": 1087,
+      "sha256": "65393dcdfadc9040cd63316836781f89a9e51b722a209b4814f9414f93124de2"
+    },
+    "plain/worker-report.json": {
+      "bytes": 855732,
+      "sha256": "1655a36b42004a16f7b75b6690a90a0948300300875304f8d845e1886c9d87cb"
+    },
+    "plain/gemm-report.json": {
+      "bytes": 2604,
+      "sha256": "89e4c9e63fce0545bed22c1997443cfe432702dc2b64cbfac810765aeb6854d9"
+    },
+    "observed/invocation.json": {
+      "bytes": 5729,
+      "sha256": "6d08c5f5e37c07b3d6d5b19030243ec5af17ef15d28dbd9be3ecc1c047f63d03"
+    },
+    "observed/runtime.json": {
+      "bytes": 18139,
+      "sha256": "b5dc605b5a34ef65f1788d1a768c45e3d50ceb222ee94a350880eb11fbd521b3"
+    },
+    "observed/runtime-identities.json": {
+      "bytes": 236,
+      "sha256": "19e7e7f5e341bd380e442a26a03ef14cc431268b428a54de5ea7a809bcaaee85"
+    },
+    "observed/requested-config.json": {
+      "bytes": 602,
+      "sha256": "8786527d9080dbae0f2be13bb65a1671217a060c04e1ba39f00449178c163892"
+    },
+    "observed/result.json": {
+      "bytes": 1090,
+      "sha256": "48a83dd196e278de5542ab111f2472a78cc2aab2b7a942bcdf5cfc02ec529573"
+    },
+    "observed/worker-report.json": {
+      "bytes": 865043,
+      "sha256": "7c2618bf84c1c54824656b4e104dd5fe6d6a9a412afa7e79a5b558e4bf101b2c"
+    },
+    "observed/gemm-report.json": {
+      "bytes": 8596,
+      "sha256": "e060e7530817e3ab64389157508437b58b5b6aec4e84f6d77924101d7bef8cc2"
+    },
+    "observed/gemm-api.jsonl": {
+      "bytes": 6025,
+      "sha256": "d56e802a4d5a451948a48f8daf121e4f4d565532b1982de45e177fa1b91b1b35"
+    }
+  },
+  "payload_files": {
+    "plain/raw.bin": {
+      "bytes": 993344,
+      "sha256": "cecf205890e4794dcf2df3b61e559330278d483a6c28d77b11e568e2a5665304"
+    },
+    "observed/raw.bin": {
+      "bytes": 17945120,
+      "sha256": "d243efa6bec36b67846fe8b0aa46335955c7fd333cc8bac21f6a5a9c3cf7dfb5"
+    },
+    "build/libzg_gemm_observer.so": {
+      "bytes": 48192,
+      "sha256": "f648eff0ecb09fde0223d570979e9fd505786b90fdd22c2df08a51ff9d3dc93b"
+    }
+  },
+  "scope": "Source/metadata bytes pinned; payload identities copied from actual accepted reports for ROOT verification. No model/runtime re-execution.",
+  "actual_parent_acceptance": {
+    "bytes": 1507,
+    "sha256": "55d68c7a90ac766b97743d2fa8c936d9044c04df8f7611207736d3703c5588be"
+  }
+}
diff --git .agents/runs/model-execution-vllm-gemm-operands-20261020/contracts.py .agents/runs/model-execution-vllm-gemm-operands-20261020/contracts.py
new file mode 100644
index 0000000..69969ac
--- /dev/null
+++ .agents/runs/model-execution-vllm-gemm-operands-20261020/contracts.py
@@ -0,0 +1,104 @@
+"""Closed source/call contracts over authenticated accepted observations only."""
+from collections import Counter
+from helpers import require
+
+PREFIX = 'model.language_model.layers.'
+PARAMETER = 'language_model.model.layers.0.linear_attn.'
+SOURCES = {
+    'ba': [('in_proj_b.weight', [0, 16]), ('in_proj_a.weight', [16, 32])],
+    'qkvz': [('in_proj_qkv.weight', [0, 6144]), ('in_proj_z.weight', [6144, 8192])],
+}
+
+
+def loaded_profile(worker):
+    roles = worker['source_roles']
+    audits = worker['parameters_before']
+    require(len(roles) == len(audits) == 320 and audits == worker['parameters_after'], 'all320 actual source audits unchanged')
+    by_source = {row['source']: row for row in roles}
+    by_audit = {row['source']: row for row in audits}
+    require(len(by_source) == len(by_audit) == 320 and by_source.keys() == by_audit.keys(), 'exact complete unique source-role coverage')
+    classes = Counter()
+    for name, role in by_source.items():
+        actual = by_audit[name]
+        source_dtype = role['source_header']['dtype']
+        loaded = role['loaded_dtype']
+        require(actual['source_dtype'] == source_dtype and actual['loaded_dtype'] == loaded
+                and actual['parameter'] == role['parameter'] and actual['rows'] == role['rows'],
+                'each real source-to-loaded audit binds its actual packed role')
+        if source_dtype == 'BF16':
+            require(loaded == 'torch.bfloat16' and actual['cast_is_explicit_observation'] is False
+                    and actual['source_words_sha256'] == actual['loaded_words_sha256'], 'original BF16 stays BF16')
+            classes['BF16_to_BF16'] += 1
+        elif name.endswith('.linear_attn.A_log'):
+            require(source_dtype == 'F32' and loaded == 'torch.float32'
+                    and actual['cast_is_explicit_observation'] is False
+                    and actual['source_words_sha256'] == actual['loaded_words_sha256'], 'A_log remains actual F32')
+            classes['A_log_F32_to_F32'] += 1
+        else:
+            require(name.endswith('.linear_attn.norm.weight') and source_dtype == 'F32'
+                    and loaded == 'torch.bfloat16' and actual['cast_is_explicit_observation'] is True,
+                    'actual trained GDN gamma sourceF32 loadedBF16 is explicit, never CPU-restored F32')
+            classes['GDN_gamma_F32_to_BF16'] += 1
+    require(dict(classes) == {'BF16_to_BF16': 284, 'A_log_F32_to_F32': 18, 'GDN_gamma_F32_to_BF16': 18},
+            'exact genuine VLLM loaded dtype profile')
+    return by_source, by_audit, dict(classes)
+
+
+def packed_roles(label, roles, audits, route):
+    require(label in SOURCES, 'closed packed projection role')
+    result = []
+    for suffix, rows in SOURCES[label]:
+        source = PREFIX + '0.linear_attn.' + suffix
+        require(source in roles and source in audits, 'complete actual exact source namespace')
+        role, audit = roles[source], audits[source]
+        parameter = PARAMETER + 'in_proj_' + label + '.weight'
+        require(role['parameter'] == audit['parameter'] == parameter and role['rows'] == audit['rows'] == rows,
+                'genuine exact layer0 packed row order')
+        require(role['source_header']['dtype'] == 'BF16' and role['loaded_dtype'] == 'torch.bfloat16'
+                and role['source_header']['shape'] == [rows[1] - rows[0], 1024]
+                and role['loaded_shape'] == route['weight']['shape']
+                and audit['loaded_bytes'] == (rows[1] - rows[0]) * 1024 * 2,
+                'full original descriptor/source row extent, no reduced synthetic tensor')
+        result.append(dict(source=source, full_source_header=role['source_header'],
+                           actual_parameter=parameter, packed_rows=rows, mapper_chain=role['mapper_chain'],
+                           source_words_sha256=audit['source_words_sha256'],
+                           loaded_words_sha256=audit['loaded_words_sha256'], loaded_bytes=audit['loaded_bytes']))
+    return result
+
+
+def bind_calls(gemm, events):
+    calls = gemm['calls']
+    require(len(calls) == 4, 'four actual calls, never synthetic placeholders')
+    by_key = {(row['label'], row['call']): row for row in calls}
+    require(by_key.keys() == {(label, call) for label in ['qkvz', 'ba'] for call in [0, 1]}, 'all four complete distinct call keys')
+    require([(row['label'], row['call']) for row in calls] == [('qkvz', 0), ('ba', 0), ('qkvz', 1), ('ba', 1)],
+            'actual source module ordering for teachers11 then353')
+    selected = []
+    for label, call in [('qkvz', 0), ('ba', 0), ('qkvz', 1), ('ba', 1)]:
+        row = by_key[label, call]
+        phase, width = (1, 8192) if label == 'qkvz' else (2, 32)
+        require(row['phase'] == phase and row['stream'] == 0, 'source-defined exact public role/stock stream')
+        actual = [event for event in events if event.get('event') == 'cublasGemmEx'
+                  and event.get('depth') == 0 and (event.get('phase'), event.get('call')) == (phase, call)]
+        require(len(actual) == 1, 'one genuine public Ex operation bound to this original module call')
+        event = actual[0]
+        require((event['m'], event['n'], event['k'], event['lda'], event['ldb'], event['ldc'])
+                == (width, 1, 1024, 1024, 1024, width), 'actual complete source geometry and strides')
+        require(event['transa'] == 1 and event['transb'] == 0 and event['Atype'] == event['Btype'] == event['Ctype'] == 14
+                and event['compute_type'] == 68 and event['requested_algorithm'] == 99
+                and event['alpha_f32_bits'] == 0x3f800000 and event['beta_f32_bits'] == 0
+                and event['math_mode'] == event['atomics_mode'] == event['pointer_mode'] == event['stream'] == 0
+                and event['workspace_bytes'] == 8519680 and event['return_status'] == 0,
+                'actual qualified public settings only; no algorithm/hidden-kernel inference')
+        require(event['A'] == row['weight']['pointer'] and event['B'] == row['input']['pointer']
+                and event['C'] == row['output']['pointer'] and event['pid'] == row['pid'] and event['tid'] == row['tid']
+                and row['weight'] == gemm['routes'][label]['weight'], 'actual input/weight/returned-output pointer owner and thread')
+        require(len({event['A'], event['B'], event['C']}) == 3, 'original separate learned/input/output storage pointers')
+        selected.append(dict(label=label, call=call, teacher=(11, 353)[call], phase=phase,
+                             actual_module=gemm['routes'][label]['module'], original_tensor_metadata=row,
+                             actual_public_api_event=event))
+    for call in [0, 1]:
+        require(by_key['qkvz', call]['input'] == by_key['ba', call]['input'], 'same genuine normalized tensor object storage per teacher')
+    require(sum(event.get('event', '').startswith('cublas') for event in events) == 4,
+            'this closed replay profile has the four actual Ex calls and no unseen nested/Lt route')
+    return selected
diff --git .agents/runs/model-execution-vllm-gemm-operands-20261020/export_operands.py .agents/runs/model-execution-vllm-gemm-operands-20261020/export_operands.py
new file mode 100644
index 0000000..e29d030
--- /dev/null
+++ .agents/runs/model-execution-vllm-gemm-operands-20261020/export_operands.py
@@ -0,0 +1,133 @@
+#!/usr/bin/env python3
+"""ROOT ONLY read-only existing genuine v4 archive extraction. No Torch/CUDA/model."""
+import argparse
+import os
+from pathlib import Path
+import time
+from accepted import authenticate, unchanged_revalidation
+from contracts import PARAMETER, bind_calls, loaded_profile, packed_roles
+from helpers import HERE, digest, digest_range, finite_bf16, jread, raw_row, read_range
+from helpers import process_limits, records, require, save, write
+
+
+def own_source():
+    freeze = jread(HERE / 'freeze-manifest.json')
+    for name, pin in freeze['files'].items():
+        require(digest(HERE / name) == pin, 'reviewed extractor source differs: ' + name)
+    return digest(HERE / 'freeze-manifest.json')
+
+
+def inspect_existing():
+    frozen = own_source()
+    bindings = jread(HERE / 'bindings.json')
+    base, capture = authenticate(bindings)
+    accepted, events = unchanged_revalidation(base, capture)
+    require(accepted['status'] == 'PASS_EAGER_B1_PUBLIC_GEMM_API_ONLY'
+            and accepted['plain_observed_raw_logits_byte_exact'] is True
+            and accepted['full_vocabulary_words_compared'] == 496640
+            and accepted['original_source_roles_bound'] == 320
+            and accepted['loaded_parameter_bytes_verified_before_and_after'] is True,
+            'genuine source/full vocabulary/loaded parameter gates remain mandatory')
+    worker = jread(capture / 'observed/worker-report.json')
+    gemm = jread(capture / 'observed/gemm-report.json')
+    runtime = jread(capture / 'observed/runtime.json')
+    require(runtime['source_repo'] == 'Qwen/Qwen3.5-0.8B'
+            and runtime['source_revision'] == '2fc06364715b967f1860aea9cf38778875588b17'
+            and runtime['source_receipt_sha256'] == 'c4c1d349743c0e0ba51ab07f769348e6b13f9896c354303c7af5e839c11dbf09',
+            'authentic original source identity from the actual pinned run')
+    rows = records(worker)
+    roles, audits, profile = loaded_profile(worker)
+    calls = bind_calls(gemm, events)
+    archive = capture / 'observed/raw.bin'
+    payloads, exported, packing = {}, [], {}
+    for label, width in [('qkvz', 8192), ('ba', 32)]:
+        route = gemm['routes'][label]
+        weight_key = 'parameter.' + PARAMETER + 'in_proj_' + label + '.weight'
+        require((weight_key, None) in rows, 'actual loaded packed weight archive record')
+        index, weight = rows[weight_key, None]
+        require(weight['dtype'] == 'torch.bfloat16' and weight['device'] == 'cuda:0'
+                and weight['shape'] == [width, 1024] and weight['stride'] == [1024, 1]
+                and weight['bytes'] == width * 1024 * 2, 'full original actual packed weight snapshot')
+        sources = packed_roles(label, roles, audits, route)
+        for source in sources:
+            start, end = source['packed_rows']
+            source['original_archive_slice'] = digest_range(archive, weight['offset'] + start * 1024 * 2,
+                (end - start) * 1024 * 2, source['loaded_words_sha256'])
+        packing[label] = dict(actual_parameter=PARAMETER + 'in_proj_' + label + '.weight',
+                              actual_full_weight_metadata=route['weight'], actual_archive_record_index=index,
+                              actual_archive_record=weight, ordered_original_sources=sources,
+                              new_weight_payload_copied=False)
+    for call, teacher in enumerate([11, 353]):
+        actual_rows = {}
+        for label, width in [('qkvz.input', 1024), ('ba.input', 1024), ('qkvz.output', 8192), ('ba.output', 32)]:
+            index, row = raw_row(rows, label, call, width)
+            data = read_range(archive, row['offset'], row['bytes'], row['sha256'])
+            finite_bf16(data)
+            actual_rows[label] = (index, row, data)
+        require(actual_rows['qkvz.input'][2] == actual_rows['ba.input'][2], 'both actual projection operands agree byte for byte')
+        for label, name in [('qkvz.input', 'input'), ('qkvz.output', 'qkvz-output'), ('ba.output', 'ba-output')]:
+            index, row, data = actual_rows[label]
+            file = f'teacher-{call}-{name}.bf16le'
+            require(file not in payloads, 'unique compact raw payload')
+            payloads[file] = data
+            source_rows = [dict(index=index, record=row)]
+            if name == 'input':
+                source_rows.append(dict(index=actual_rows['ba.input'][0], record=actual_rows['ba.input'][1]))
+            exported.append(dict(file=file, bytes=len(data), sha256=row['sha256'], dtype='BF16LE',
+                                 logical_shape=row['shape'], call=call, teacher=teacher,
+                                 exact_original_archive_rows=source_rows))
+    for item in calls:
+        call, label = item['call'], item['label']
+        item['input_file'] = f'teacher-{call}-input.bf16le'
+        item['output_file'] = f'teacher-{call}-{label}-output.bf16le'
+        # Explicit original keys/indexes close a raw archive to this actual
+        # source hook ordering; labels alone are never the authority.
+        item['archive_input_record_index'], item['archive_input_record'] = raw_row(rows, label + '.input', call, 1024)
+        item['archive_output_record_index'], item['archive_output_record'] = raw_row(rows, label + '.output', call,
+                                                                                   8192 if label == 'qkvz' else 32)
+    require(len(payloads) == 6 and sum(map(len, payloads.values())) == 36992, 'exact bounded compact real payload')
+    authenticate(bindings)  # Original captures, source, metadata, observer/raw bytes remain unchanged.
+    require(own_source() == frozen, 'extractor source changed during validation')
+    manifest = dict(status='AUTHENTIC_VLLM_EAGER_B1_GEMM_OPERANDS_ONLY_NATIVE_PROFILE_UNQUALIFIED',
+        schema='zerogravity.vllm.public-gemm-operands.v1', extractor_source_freeze=frozen,
+        accepted_capture=str(capture), accepted_source=str(base), parent_bindings=digest(HERE / 'bindings.json'),
+        original_source_identity={key: runtime[key] for key in ['source_repo', 'source_revision', 'source_receipt_sha256']},
+        accepted_actual_validation=bindings['metadata_files']['gemm-validation.json'],
+        accepted_revalidation=accepted, original_observed_archive=worker['raw'] | {'records': 'retained in pinned original worker-report'},
+        teacher_ids=[11, 353], calls=calls, learned_packing=packing, loaded_dtype_profile=profile,
+        files=exported, total_payload_bytes=36992,
+        transport=dict(new_gpu_transfers=0, new_model_or_operator_calls=0,
+                       original_capture='unchanged public forward hook detach().to(cpu).contiguous raw-bit transport',
+                       original_transfer_scope='blocking D2H/synchronization observational overhead; no timing claim',
+                       extraction='stdlib file-byte transport only; no dtype cast, reconstruction or arithmetic'),
+        unsupported=dict(native_provider_internal_allocations_admitted=False, native_cuda_math_qualified=False,
+                         all42_captured=False, default_graphs=False, wider_or_mixed_rows=False,
+                         full_original_HF_engine_qualified=False, performance_qualified=False))
+    return manifest, payloads, accepted
+
+
+def main():
+    p = argparse.ArgumentParser(description=__doc__)
+    p.add_argument('--new-output', required=True, type=Path)
+    a = p.parse_args()
+    output = a.new_output.resolve()
+    require(output.is_relative_to(HERE) and output != HERE and not output.exists(), 'fresh private extract directory')
+    # ROOT runs a finite original process; these process-local limits precede
+    # reading source payloads and never affect a provider, actor or parent.
+    process_limits()
+    started = time.monotonic()
+    manifest, payloads, accepted = inspect_existing()
+    output.mkdir(mode=0o700)
+    for file, data in payloads.items():
+        save(output / file, data)
+    write(output / 'accepted-revalidation.json', accepted)
+    write(output / 'manifest.json', manifest)
+    write(output / 'export-invocation.json', dict(status='EXTRACTED_AUTHENTIC_ARCHIVE_ONLY_VALIDATOR_REQUIRED',
+          original_capture=manifest['accepted_capture'], new_output=str(output), source_freeze=manifest['extractor_source_freeze'],
+          cpu_affinity=sorted(os.sched_getaffinity(0)), elapsed_seconds_not_benchmark=time.monotonic() - started,
+          tensors_or_models_executed=False, parent_originals_written=False))
+    print(manifest['status'], 'bytes=36992 files=6')
+
+
+if __name__ == '__main__':
+    main()
diff --git .agents/runs/model-execution-vllm-gemm-operands-20261020/helpers.py .agents/runs/model-execution-vllm-gemm-operands-20261020/helpers.py
new file mode 100644
index 0000000..0cfa07c
--- /dev/null
+++ .agents/runs/model-execution-vllm-gemm-operands-20261020/helpers.py
@@ -0,0 +1,141 @@
+"""Inert stdlib-only compact transport helpers; no numerical/provider runtime."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import struct
+
+HERE = Path(__file__).resolve().parent
+CAP = 4 * 1024 * 1024
+
+
+def process_limits():
+    # ROOT-only explicit main call; importing never changes process state.
+    import resource
+    import signal
+    cpus = sorted(os.sched_getaffinity(0))[:4]
+    require(len(cpus) == 4, 'four allowed file-validator CPUs')
+    os.sched_setaffinity(0, cpus)
+    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
+    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
+    signal.alarm(180)
+    return cpus
+
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+
+def digest(path):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
+        a = os.fstat(stream.fileno())
+        require(stat.S_ISREG(a.st_mode), 'regular pinned artifact: ' + str(path))
+        value = hashlib.sha256()
+        for block in iter(lambda: stream.read(1024 * 1024), b''):
+            value.update(block)
+        b = os.fstat(stream.fileno())
+        require(all(getattr(a, field) == getattr(b, field) for field in
+                    ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns']), 'artifact changed while reading')
+    return dict(bytes=a.st_size, sha256=value.hexdigest())
+
+
+def bounded(path, limit=CAP):
+    path = Path(path)
+    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
+            'bounded regular metadata: ' + str(path))
+    with path.open('rb') as stream:
+        raw = stream.read(limit + 1)
+    require(len(raw) <= limit, 'bounded read')
+    return raw
+
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON key')
+        result[key] = value
+    return result
+
+
+def invalid_constant(value):
+    raise ValueError('nonfinite JSON constant: ' + value)
+
+
+def jread(path):
+    return json.loads(bounded(path), object_pairs_hook=unique, parse_constant=invalid_constant)
+
+
+def write(path, value):
+    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
+    require(len(data) <= CAP, 'bounded compact manifest')
+    save(path, data)
+
+
+def save(path, data):
+    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
+    with os.fdopen(fd, 'wb') as stream:
+        stream.write(data)
+        stream.flush()
+        os.fsync(stream.fileno())
+
+
+def read_range(path, offset, count, expected_sha):
+    require(type(offset) is int and type(count) is int and offset >= 0 and 0 < count <= 32 * 1024**2,
+            'bounded exact archived range')
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
+        a = os.fstat(stream.fileno())
+        require(stat.S_ISREG(a.st_mode) and offset + count <= a.st_size <= 32 * 1024**2,
+                'original archive regular range')
+        stream.seek(offset)
+        data = stream.read(count)
+        b = os.fstat(stream.fileno())
+        require(all(getattr(a, field) == getattr(b, field) for field in
+                    ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns']), 'archive changed during range read')
+    require(len(data) == count and hashlib.sha256(data).hexdigest() == expected_sha, 'authentic archive range bits')
+    return data
+
+
+def digest_range(path, offset, count, expected_sha):
+    require(type(offset) is int and type(count) is int and offset >= 0 and 0 < count <= 32 * 1024**2,
+            'bounded packed source subrange')
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
+        a = os.fstat(stream.fileno())
+        require(stat.S_ISREG(a.st_mode) and offset + count <= a.st_size <= 32 * 1024**2, 'original packed range')
+        stream.seek(offset)
+        left, value = count, hashlib.sha256()
+        while left:
+            block = stream.read(min(left, 1024 * 1024))
+            require(block, 'complete packed source range')
+            value.update(block)
+            left -= len(block)
+        b = os.fstat(stream.fileno())
+        require(all(getattr(a, field) == getattr(b, field) for field in
+                    ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns']), 'packed archive changed while hashing')
+    require(value.hexdigest() == expected_sha, 'actual packed row bytes equal the corresponding real source audit')
+    return dict(offset=offset, bytes=count, sha256=value.hexdigest())
+
+
+def finite_bf16(data):
+    require(data and len(data) % 2 == 0, 'positive whole BF16 words')
+    require(all(word & 0x7f80 != 0x7f80 for word, in struct.iter_unpack('<H', data)),
+            'all authentic BF16 values finite; preserve signed zero/subnormals')
+
+
+def records(report):
+    result = {}
+    for index, row in enumerate(report['raw']['records']):
+        key = (row['label'], row.get('call'))
+        require(key not in result, 'unique original archive label/call key')
+        result[key] = (index, row)
+    return result
+
+
+def raw_row(rows, label, call, width):
+    require((label, call) in rows, 'nonvacuous exact original row key')
+    index, row = rows[label, call]
+    require(row['shape'] == [1, width] and row['stride'] == [width, 1]
+            and row['dtype'] == 'torch.bfloat16' and row['device'] == 'cuda:0'
+            and row['bytes'] == width * 2, 'genuine contiguous BF16 archive row')
+    return index, row
diff --git .agents/runs/model-execution-vllm-gemm-operands-20261020/test_contracts.py .agents/runs/model-execution-vllm-gemm-operands-20261020/test_contracts.py
new file mode 100644
index 0000000..f0902db
--- /dev/null
+++ .agents/runs/model-execution-vllm-gemm-operands-20261020/test_contracts.py
@@ -0,0 +1,174 @@
+"""ROOT ONLY source-free transport/metadata negatives, never a genuine oracle."""
+import copy
+import hashlib
+from pathlib import Path
+import struct
+import tempfile
+import unittest
+from accepted import unchanged_revalidation
+from contracts import PARAMETER, PREFIX, bind_calls, loaded_profile, packed_roles
+from helpers import digest_range, finite_bf16, raw_row, read_range, records
+
+
+def call_fixture():
+    routes = {label: dict(module=PARAMETER + 'in_proj_' + label,
+                         weight=dict(pointer=1000000 if label == 'qkvz' else 18000000))
+              for label in ['qkvz', 'ba']}
+    calls, events = [], []
+    for call in [0, 1]:
+        for label, phase, rows in [('qkvz', 1, 8192), ('ba', 2, 32)]:
+            item = dict(label=label, phase=phase, call=call, stream=0, pid=101, tid=102,
+                        input=dict(pointer=20000000 + call * 10000), weight=routes[label]['weight'],
+                        output=dict(pointer=22000000 + call * 100000 + phase * 20000))
+            calls.append(item)
+            events.append(dict(event='cublasGemmEx', depth=0, phase=phase, call=call,
+                m=rows, n=1, k=1024, lda=1024, ldb=1024, ldc=rows, transa=1, transb=0,
+                Atype=14, Btype=14, Ctype=14, compute_type=68, requested_algorithm=99,
+                alpha_f32_bits=0x3f800000, beta_f32_bits=0, math_mode=0, atomics_mode=0,
+                pointer_mode=0, stream=0, workspace_bytes=8519680, return_status=0,
+                A=item['weight']['pointer'], B=item['input']['pointer'], C=item['output']['pointer'], pid=101, tid=102))
+    return dict(routes=routes, calls=calls), events
+
+
+def profile_fixture():
+    roles, audits = [], []
+    names = [(f'model.language_model.fixture.parameter{i}', 'BF16', 'torch.bfloat16', False) for i in range(284)]
+    for layer in range(24):
+        if layer % 4 != 3:
+            names.extend([(PREFIX + str(layer) + '.linear_attn.A_log', 'F32', 'torch.float32', False),
+                          (PREFIX + str(layer) + '.linear_attn.norm.weight', 'F32', 'torch.bfloat16', True)])
+    for name, source, loaded, cast in names:
+        roles.append(dict(source=name, parameter='fixture.' + name, rows=[0, 1],
+                          source_header=dict(dtype=source), loaded_dtype=loaded))
+        audits.append(dict(source=name, parameter='fixture.' + name, rows=[0, 1], source_dtype=source,
+                           loaded_dtype=loaded, cast_is_explicit_observation=cast,
+                           source_words_sha256='a' * 64, loaded_words_sha256=('b' if cast else 'a') * 64))
+    return dict(source_roles=roles, parameters_before=audits, parameters_after=copy.deepcopy(audits))
+
+
+class Contracts(unittest.TestCase):
+    def test_four_ordered_distinct_metadata_calls_only(self):
+        gemm, events = call_fixture()
+        self.assertEqual([(r['label'], r['teacher']) for r in bind_calls(gemm, events)],
+                         [('qkvz', 11), ('ba', 11), ('qkvz', 353), ('ba', 353)])
+
+    def test_missing_reordered_or_duplicate_calls_refused(self):
+        for mutation in ['missing', 'order', 'duplicate']:
+            gemm, events = call_fixture()
+            if mutation == 'missing':
+                gemm['calls'].pop()
+            elif mutation == 'order':
+                gemm['calls'][0], gemm['calls'][1] = gemm['calls'][1], gemm['calls'][0]
+            else:
+                gemm['calls'][3] = gemm['calls'][2]
+            with self.assertRaises(ValueError):
+                bind_calls(gemm, events)
+
+    def test_profile_changes_and_extra_api_route_refused(self):
+        for key, value in [('event', 'cublasLtMatmul'), ('compute_type', 69), ('requested_algorithm', 0),
+                           ('stream', 1), ('workspace_bytes', 0), ('beta_f32_bits', 0x3f800000)]:
+            gemm, events = call_fixture()
+            events[0][key] = value
+            with self.assertRaises(ValueError):
+                bind_calls(gemm, events)
+        gemm, events = call_fixture()
+        events.append(dict(events[0], depth=1))
+        with self.assertRaises(ValueError):
+            bind_calls(gemm, events)
+
+    def test_original_pointer_and_teacher_object_coupling_refused(self):
+        for target in ['input', 'output']:
+            gemm, events = call_fixture()
+            gemm['calls'][0][target] = dict(pointer=777)
+            with self.assertRaises(ValueError):
+                bind_calls(gemm, events)
+        gemm, events = call_fixture()
+        gemm['calls'][1]['input'] = gemm['calls'][3]['input']
+        events[1]['B'] = gemm['calls'][1]['input']['pointer']
+        with self.assertRaises(ValueError):
+            bind_calls(gemm, events)
+
+    def test_source36_cast_profile_is_separate(self):
+        worker = profile_fixture()
+        self.assertEqual(loaded_profile(worker)[2], dict(BF16_to_BF16=284, A_log_F32_to_F32=18, GDN_gamma_F32_to_BF16=18))
+        for field, value in [('loaded_dtype', 'torch.float32'), ('cast_is_explicit_observation', False)]:
+            invalid = copy.deepcopy(worker)
+            index = next(i for i, r in enumerate(invalid['parameters_before']) if r['cast_is_explicit_observation'])
+            invalid['parameters_before'][index][field] = value
+            invalid['parameters_after'] = copy.deepcopy(invalid['parameters_before'])
+            if field == 'loaded_dtype':
+                invalid['source_roles'][index]['loaded_dtype'] = value
+            with self.assertRaises(ValueError):
+                loaded_profile(invalid)
+
+    def test_actual_packed_ba_order_and_namespace(self):
+        roles, audits = {}, {}
+        for suffix, rows in [('in_proj_b.weight', [0, 16]), ('in_proj_a.weight', [16, 32])]:
+            name = PREFIX + '0.linear_attn.' + suffix
+            roles[name] = dict(source_header=dict(dtype='BF16', shape=[16, 1024]), loaded_dtype='torch.bfloat16',
+                loaded_shape=[32, 1024], parameter=PARAMETER + 'in_proj_ba.weight', rows=rows, mapper_chain=[])
+            audits[name] = dict(parameter=PARAMETER + 'in_proj_ba.weight', rows=rows, loaded_bytes=32768,
+                                source_words_sha256='a' * 64, loaded_words_sha256='a' * 64)
+        route = dict(weight=dict(shape=[32, 1024]))
+        self.assertEqual([x['packed_rows'] for x in packed_roles('ba', roles, audits, route)], [[0, 16], [16, 32]])
+        invalid = copy.deepcopy(roles)
+        invalid[PREFIX + '0.linear_attn.in_proj_a.weight']['rows'] = [0, 16]
+        with self.assertRaises(ValueError):
+            packed_roles('ba', invalid, audits, route)
+        del invalid[PREFIX + '0.linear_attn.in_proj_a.weight']
+        with self.assertRaises(ValueError):
+            packed_roles('ba', invalid, audits, route)
+
+    def test_original_raw_index_call_shape_not_labels_alone(self):
+        row = dict(label='ba.output', call=1, shape=[1, 32], stride=[32, 1], dtype='torch.bfloat16',
+                   device='cuda:0', bytes=64, offset=4, sha256='a' * 64)
+        report = dict(raw=dict(records=[row]))
+        self.assertEqual(raw_row(records(report), 'ba.output', 1, 32), (0, row))
+        with self.assertRaises(ValueError):
+            raw_row(records(report), 'ba.output', 0, 32)
+        with self.assertRaises(ValueError):
+            records(dict(raw=dict(records=[row, row])))
+        row['stride'] = [1, 32]
+        with self.assertRaises(ValueError):
+            raw_row(records(report), 'ba.output', 1, 32)
+
+    def test_bit_transport_preserves_signed_zero_and_subnormal(self):
+        data = struct.pack('<5H', 0, 0x8000, 1, 0x8001, 0x3f80)
+        finite_bf16(data)
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder) / 'raw.bin'
+            path.write_bytes(b'head' + data + b'tail')
+            value = hashlib.sha256(data).hexdigest()
+            self.assertEqual(read_range(path, 4, len(data), value), data)
+            self.assertEqual(digest_range(path, 4, len(data), value)['sha256'], value)
+            for offset in [0, 5, 100]:
+                with self.assertRaises(ValueError):
+                    read_range(path, offset, len(data), value)
+            with self.assertRaises(ValueError):
+                digest_range(path, 0, len(data), value)
+
+    def test_nonfinite_raw_values_refused(self):
+        for word in [0x7f80, 0xff80, 0x7fc1]:
+            with self.assertRaises(ValueError):
+                finite_bf16(struct.pack('<H', word))
+
+    def test_unchanged_validator_write_sink_preserves_original_files(self):
+        # Synthetic definitions exercise IO redirection only; not oracle evidence.
+        with tempfile.TemporaryDirectory() as folder:
+            root = Path(folder)
+            base, capture = root / 'source', root / 'capture'
+            base.mkdir(); capture.mkdir()
+            (capture / 'observed').mkdir()
+            (capture / 'validation.json').write_text('{"subset":1}')
+            (capture / 'gemm-validation.json').write_text('{"gemm":2}')
+            (capture / 'observed/gemm-report.json').write_text('{}')
+            (base / 'support.py').write_text('def write(*args):\n    raise AssertionError("original writer must never run")\n')
+            (base / 'validate.py').write_text('from support import write\ndef validate(p):\n    write(p/"validation.json",{"subset":1})\n')
+            (base / 'validate_gemm_v4.py').write_text('from support import write\nfrom validate import validate as subset\ndef validate(p):\n    subset(p)\n    write(p/"gemm-validation.json",{"gemm":2})\n    return {"gemm":2}\ndef trace_validate(*args):\n    return []\n')
+            before = {p: p.read_bytes() for p in capture.rglob('*.json')}
+            self.assertEqual(unchanged_revalidation(base, capture), ({'gemm': 2}, []))
+            self.assertEqual(before, {p: p.read_bytes() for p in capture.rglob('*.json')})
+
+
+if __name__ == '__main__':
+    unittest.main()
diff --git .agents/runs/model-execution-vllm-gemm-operands-20261020/validate_extract.py .agents/runs/model-execution-vllm-gemm-operands-20261020/validate_extract.py
new file mode 100644
index 0000000..2e2507d
--- /dev/null
+++ .agents/runs/model-execution-vllm-gemm-operands-20261020/validate_extract.py
@@ -0,0 +1,50 @@
+#!/usr/bin/env python3
+"""ROOT ONLY independently rebind compact bytes to unchanged actual accepted source archive."""
+import argparse
+from pathlib import Path
+from export_operands import inspect_existing
+from helpers import HERE, bounded, digest, jread, process_limits, require, write
+
+
+def validate(directory):
+    actual = jread(directory / 'manifest.json')
+    expected, payloads, accepted = inspect_existing()
+    require(actual == expected and jread(directory / 'accepted-revalidation.json') == accepted,
+            'complete manifest must equal freshly authenticated original metadata/API/raw/role derivation')
+    require({p.name for p in directory.iterdir()} == set(payloads) | {'manifest.json', 'accepted-revalidation.json', 'export-invocation.json'},
+            'no missing/unlisted compact payload or metadata')
+    for file, data in payloads.items():
+        require(bounded(directory / file, 16384) == data, 'every extracted original bit exact: ' + file)
+    invocation = jread(directory / 'export-invocation.json')
+    require(invocation['status'] == 'EXTRACTED_AUTHENTIC_ARCHIVE_ONLY_VALIDATOR_REQUIRED'
+            and invocation['source_freeze'] == expected['extractor_source_freeze']
+            and invocation['original_capture'] == expected['accepted_capture']
+            and invocation['new_output'] == str(directory)
+            and invocation['tensors_or_models_executed'] is False and invocation['parent_originals_written'] is False,
+            'explicit read-only file transport scope')
+    return dict(status='PASS_AUTHENTIC_EAGER_B1_ARCHIVE_OPERANDS_ONLY_NATIVE_PROFILE_UNQUALIFIED',
+                compact_manifest=digest(directory / 'manifest.json'), payload_bytes=36992,
+                actual_gemm_calls=4, teacher_ids=[11, 353], original_full_logits_words=496640,
+                original_source_audits=320, BA_order=['b', 'a'], QKVZ_order=['qkv', 'z'],
+                new_GPU_or_model_execution=False, native_provider_internal_allocation_admission=False,
+                all42_or_full_model_qualified=False, performance_qualified=False)
+
+
+def main():
+    p = argparse.ArgumentParser(description=__doc__)
+    p.add_argument('--extract', required=True, type=Path)
+    p.add_argument('--new-report', required=True, type=Path)
+    a = p.parse_args()
+    directory = a.extract.resolve(strict=True)
+    require(directory.is_relative_to(HERE) and directory != HERE, 'owned compact extract only')
+    new_report = a.new_report.resolve()
+    require(new_report.is_relative_to(HERE) and not new_report.is_relative_to(directory)
+            and new_report.parent.is_dir() and not new_report.exists(), 'fresh own report outside the closed extract and originals')
+    process_limits()
+    report = validate(directory)
+    write(new_report, report)
+    print(report['status'])
+
+
+if __name__ == '__main__':
+    main()

````

