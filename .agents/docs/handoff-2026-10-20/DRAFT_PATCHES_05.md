# Draft Patches — volume 5

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 45177; SHA256 `061cfc0d9fe7842713f6b549ec9f468a87936ba88fc32e74a4b8a52b01a91e2e`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -12,6 +12,8 @@
 use std::sync::Arc;
 use std::sync::atomic::{AtomicU64, Ordering};
 
+mod cached_gdn;
+pub use cached_gdn::{CachedGdnContract, CachedGdnOperands, CachedGdnValues};
 mod gdn_conv;
 pub use gdn_conv::{CausalConvSiluContract, CausalConvValues};
 mod typed;
@@ -282,6 +284,18 @@
             "backend does not implement explicit ordinary RMS contract".into(),
         ))
     }
+    /// Cached single-token arithmetic only; decay is an explicit external input.
+    fn cached_gdn_profiled(
+        &mut self,
+        _source: SourceIdentity,
+        _heads: usize,
+        _input: &CachedGdnOperands<Self::Value>,
+        _contract: CachedGdnContract,
+    ) -> Result<CachedGdnValues<Self::Value>> {
+        Err(Error::Unsupported(
+            "backend does not implement explicit cached GDN contract".into(),
+        ))
+    }
     /// Pure original convolution window; no state owner or implicit fallback.
     fn causal_conv_silu_profiled(
         &mut self,
@@ -466,6 +480,16 @@
     TypedProject {
         tensor: SourceTensor,
         contract: ProjectionContract,
+    },
+    TypedCachedGdnCore {
+        source: SourceIdentity,
+        heads: usize,
+        contract: CachedGdnContract,
+    },
+    TypedCachedGdnBf16Return {
+        source: SourceIdentity,
+        heads: usize,
+        contract: CachedGdnContract,
     },
     TypedCausalConvSilu {
         parameter: SourceParameter,
@@ -818,6 +842,16 @@
                 contract.validate(parameter, inputs[0].elements)?;
                 Ok(ActivationPrecision::Bf16WidenedF32)
             }
+            PlanOperation::TypedCachedGdnCore {
+                source,
+                heads,
+                contract,
+            } => cached_gdn::result_precision(self, *source, *heads, *contract, inputs, false),
+            PlanOperation::TypedCachedGdnBf16Return {
+                source,
+                heads,
+                contract,
+            } => cached_gdn::result_precision(self, *source, *heads, *contract, inputs, true),
             PlanOperation::TypedCausalConvSilu {
                 parameter,
                 contract,
@@ -839,6 +873,8 @@
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
+                        | PlanOperation::TypedCachedGdnCore { .. }
+                        | PlanOperation::TypedCachedGdnBf16Return { .. }
                 ) =>
             {
                 Ok(ActivationPrecision::F32)
@@ -1105,6 +1141,15 @@
             input.shape,
             None,
         )
+    }
+    fn cached_gdn_profiled(
+        &mut self,
+        source: SourceIdentity,
+        heads: usize,
+        input: &CachedGdnOperands<PlanValue>,
+        contract: CachedGdnContract,
+    ) -> Result<CachedGdnValues<PlanValue>> {
+        cached_gdn::record(self, source, heads, input, contract)
     }
     fn causal_conv_silu_profiled(
         &mut self,
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/cached_gdn.rs
@@ -0,0 +1,301 @@
+//! Pure cached-token core; explicit decay boundary, never initial chunk64.
+use super::*;
+
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum CachedGdnContract {
+    TorchCpuBf16Avx2Width128,
+}
+impl CachedGdnContract {
+    pub fn validate(self, heads: usize, lengths: &[usize; 6]) -> Result<(usize, usize)> {
+        let vector = heads.checked_mul(128);
+        let state = vector.and_then(|n| n.checked_mul(128));
+        let (Some(vector), Some(state)) = (vector, state) else {
+            return Err(Error::Limit("cached GDN geometry overflow".into()));
+        };
+        if heads == 0 || state > 1_048_576 {
+            return Err(Error::Limit("cached GDN head/element bound".into()));
+        }
+        if *lengths != [vector, vector, vector, heads, heads, state] {
+            return Err(Error::Malformed(
+                "cached GDN explicit operand geometry".into(),
+            ));
+        }
+        Ok((vector, state))
+    }
+}
+
+/// Q/K are already repeated to value heads. No implicit head/layout conversion.
+#[derive(Clone)]
+pub struct CachedGdnOperands<V> {
+    pub query: V,
+    pub key: V,
+    pub value: V,
+    pub beta: V,
+    /// Explicit finite FP32 exp(g); this type grants no coefficient eligibility.
+    pub decay: V,
+    /// FP32 [head,key128,value128], distinct from canonical value-major state.
+    pub previous_state: V,
+}
+#[derive(Clone)]
+pub struct CachedGdnValues<V> {
+    pub core_f32: V,
+    pub output_bf16: V,
+    pub next_state: V,
+}
+
+pub(super) fn result_precision(
+    builder: &PlanBuilder,
+    source: SourceIdentity,
+    heads: usize,
+    contract: CachedGdnContract,
+    inputs: &[PlanValue],
+    bf16_return: bool,
+) -> Result<ActivationPrecision> {
+    if !builder.operator_slice
+        || builder.source != source
+        || builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    {
+        return Err(Error::Unsupported(
+            "cached GDN source/profile/private slice".into(),
+        ));
+    }
+    if bf16_return {
+        let vector = heads
+            .checked_mul(128)
+            .filter(|_| heads > 0 && heads <= 64)
+            .ok_or_else(|| Error::Limit("cached GDN return head bound".into()))?;
+        if inputs.len() != 1
+            || inputs[0].elements != vector
+            || inputs[0].precision != ActivationPrecision::F32
+        {
+            return Err(Error::Malformed("cached GDN return F32 core".into()));
+        }
+        Ok(ActivationPrecision::Bf16WidenedF32)
+    } else {
+        let lengths: [usize; 6] = inputs
+            .iter()
+            .map(|v| v.elements)
+            .collect::<Vec<_>>()
+            .try_into()
+            .map_err(|_| Error::Malformed("cached GDN operand arity".into()))?;
+        contract.validate(heads, &lengths)?;
+        if inputs[..4]
+            .iter()
+            .any(|v| v.precision != ActivationPrecision::Bf16WidenedF32)
+            || inputs[4..]
+                .iter()
+                .any(|v| v.precision != ActivationPrecision::F32)
+        {
+            return Err(Error::Unsupported(
+                "cached GDN BF16/F32 operand precision".into(),
+            ));
+        }
+        Ok(ActivationPrecision::F32)
+    }
+}
+
+pub(super) fn record(
+    builder: &mut PlanBuilder,
+    source: SourceIdentity,
+    heads: usize,
+    input: &CachedGdnOperands<PlanValue>,
+    contract: CachedGdnContract,
+) -> Result<CachedGdnValues<PlanValue>> {
+    let inputs = [
+        input.query,
+        input.key,
+        input.value,
+        input.beta,
+        input.decay,
+        input.previous_state,
+    ];
+    let (vector, state) = contract.validate(heads, &inputs.map(|v| v.elements))?;
+    // Existing record validates ALL original builder/value/stride memberships
+    // before recording either pure node. No record precision policy changes.
+    let values = builder.record(
+        "original.cached_gdn.core",
+        PlanOperation::TypedCachedGdnCore {
+            source,
+            heads,
+            contract,
+        },
+        &inputs,
+        &[ValueShape::Vector(vector), ValueShape::Vector(state)],
+        None,
+    )?;
+    let [core_f32, next_state]: [PlanValue; 2] = values
+        .try_into()
+        .map_err(|_| Error::State("cached GDN core output arity".into()))?;
+    let output_bf16 = builder.one(
+        "original.cached_gdn.return_bf16",
+        PlanOperation::TypedCachedGdnBf16Return {
+            source,
+            heads,
+            contract,
+        },
+        &[core_f32],
+        ValueShape::Vector(vector),
+        None,
+    )?;
+    Ok(CachedGdnValues {
+        core_f32,
+        output_bf16,
+        next_state,
+    })
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    fn input(builder: &mut PlanBuilder, n: usize, bf16: bool) -> PlanValue {
+        if bf16 {
+            builder
+                .embed_profiled(
+                    SourceTensor {
+                        source: builder.source,
+                        name: format!("synthetic.operand.{n}"),
+                        kind: QuantKind::Bf16,
+                        axes: MatrixAxes::OutputInput,
+                        columns: n,
+                        rows: 1,
+                    },
+                    PlanBuilder::inputs().token,
+                    EmbeddingContract::Bf16OutputInput,
+                )
+                .unwrap()
+        } else {
+            // Test-only F32 operand metadata; no executable source/model claim.
+            builder
+                .one(
+                    "synthetic.fp32_operand",
+                    PlanOperation::Observe {
+                        layer: 0,
+                        stage: "synthetic_fp32_operand",
+                    },
+                    &[],
+                    ValueShape::Vector(n),
+                    None,
+                )
+                .unwrap()
+        }
+    }
+    fn fixture(b: &mut PlanBuilder) -> CachedGdnOperands<PlanValue> {
+        CachedGdnOperands {
+            query: input(b, 128, true),
+            key: input(b, 128, true),
+            value: input(b, 128, true),
+            beta: input(b, 1, true),
+            decay: input(b, 1, false),
+            previous_state: input(b, 16384, false),
+        }
+    }
+    fn builder() -> PlanBuilder {
+        PlanBuilder::new_operator_slice(
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap()
+    }
+    #[test]
+    fn cached_core_and_return_have_separate_precisions_without_state_effects() {
+        let mut b = builder();
+        let operands = fixture(&mut b);
+        let source = b.source;
+        let result = b
+            .cached_gdn_profiled(
+                source,
+                1,
+                &operands,
+                CachedGdnContract::TorchCpuBf16Avx2Width128,
+            )
+            .unwrap();
+        assert_eq!(result.core_f32.precision(), ActivationPrecision::F32);
+        assert_eq!(result.next_state.precision(), ActivationPrecision::F32);
+        assert_eq!(
+            result.output_bf16.precision(),
+            ActivationPrecision::Bf16WidenedF32
+        );
+        let recorded = b.finish_operator_slice(result.output_bf16).unwrap();
+        assert_eq!(recorded.plan().nodes().len(), 8);
+        assert!(recorded.plan().required_domains().is_empty());
+        assert!(recorded.plan().proposed_domains().is_empty());
+        assert_eq!(recorded.plan().frontier_delta(), 0);
+    }
+    #[test]
+    fn foreign_inputs_and_valid_geometry_strides_refuse_before_first_core_node() {
+        let mut b = builder();
+        let mut operands = fixture(&mut b);
+        let source = b.source;
+        let mut other = builder();
+        operands.decay = input(&mut other, 1, false);
+        assert!(
+            b.cached_gdn_profiled(
+                source,
+                1,
+                &operands,
+                CachedGdnContract::TorchCpuBf16Avx2Width128
+            )
+            .is_err()
+        );
+        assert_eq!(b.nodes.len(), 6);
+        operands.decay = input(&mut b, 1, false);
+        let base = input(&mut b, 256, true);
+        operands.query = b
+            .view(
+                &base,
+                TensorView {
+                    offset: 0,
+                    rows: 128,
+                    width: 1,
+                    row_stride: 2,
+                },
+            )
+            .unwrap();
+        let error = b
+            .cached_gdn_profiled(
+                source,
+                1,
+                &operands,
+                CachedGdnContract::TorchCpuBf16Avx2Width128,
+            )
+            .err()
+            .unwrap();
+        assert!(error.to_string().contains("contiguous"));
+        let nodes = b.nodes.len();
+        operands.query = b.contiguous(&operands.query).unwrap();
+        let decay = operands.decay;
+        operands.decay = operands.beta;
+        assert!(
+            b.cached_gdn_profiled(
+                source,
+                1,
+                &operands,
+                CachedGdnContract::TorchCpuBf16Avx2Width128
+            )
+            .is_err()
+        );
+        assert_eq!(b.nodes.len(), nodes + 1);
+        operands.decay = decay;
+        b.cached_gdn_profiled(
+            source,
+            1,
+            &operands,
+            CachedGdnContract::TorchCpuBf16Avx2Width128,
+        )
+        .unwrap();
+    }
+    #[test]
+    fn closed_bounds_cover48heads_without_head_padding() {
+        let c = CachedGdnContract::TorchCpuBf16Avx2Width128;
+        assert_eq!(
+            c.validate(48, &[6144, 6144, 6144, 48, 48, 786432]).unwrap(),
+            (6144, 786432)
+        );
+        assert!(c.validate(0, &[0; 6]).is_err());
+        assert!(
+            c.validate(65, &[8320, 8320, 8320, 65, 65, 1064960])
+                .is_err()
+        );
+        assert!(c.validate(usize::MAX, &[0; 6]).is_err());
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs
@@ -0,0 +1,327 @@
+//! Synthetic ownership/schedule checks; genuine archived replay is separate.
+use super::*;
+use crate::accelerator::program::{CachedGdnContract, CachedGdnOperands};
+use crate::operator_slice::cached_gdn::CachedGdnSliceInput;
+use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
+
+fn cached_source(fixture: &Fixture) -> Loaded {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    cfg["text_config"]["hidden_size"] = json!(256);
+    cfg["text_config"]["linear_key_head_dim"] = json!(128);
+    cfg["text_config"]["linear_value_head_dim"] = json!(128);
+    source_with_config(
+        fixture,
+        serde_json::to_vec(&cfg).unwrap(),
+        true,
+        |_, _, v| v,
+        |_| {},
+    )
+}
+fn raw() -> [Vec<u8>; 6] {
+    let bf16 = |values: Vec<f32>| {
+        values
+            .into_iter()
+            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
+            .collect()
+    };
+    let fp32 = |values: Vec<f32>| values.into_iter().flat_map(f32::to_le_bytes).collect();
+    [
+        bf16(
+            (0..256)
+                .map(|i| if i % 3 == 0 { -0.0 } else { 0.5 })
+                .collect(),
+        ),
+        bf16(
+            (0..256)
+                .map(|i| if i % 2 == 0 { 0.25 } else { -0.5 })
+                .collect(),
+        ),
+        bf16((0..256).map(|i| if i < 128 { 1.0 } else { -2.0 }).collect()),
+        bf16(vec![0.5, 0.25]),
+        fp32(vec![0.75, 0.5]),
+        fp32(
+            (0..32768)
+                .map(|i| {
+                    if i % 11 == 0 {
+                        -0.0
+                    } else {
+                        (i % 9) as f32 * 0.125
+                    }
+                })
+                .collect(),
+        ),
+    ]
+}
+fn input<'a>(
+    source: crate::accelerator::SourceIdentity,
+    raw: &'a [Vec<u8>; 6],
+) -> CachedGdnSliceInput<'a> {
+    CachedGdnSliceInput::from_bytes(
+        source,
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        2,
+        raw.each_ref().map(|v| v.as_slice()),
+    )
+    .unwrap()
+}
+fn imported_bytes() -> u64 {
+    2 * (16384 + 3 * 128 + 2) * 4
+}
+fn result_bytes() -> u64 {
+    2 * (16384 + 2 * 128) * 4
+}
+
+#[test]
+fn cached_fragment_reuses_closed_core_keeps_signed_bits_and_retains_all_outputs() {
+    let f = Fixture::new();
+    let loaded = cached_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let raw = raw();
+    let actual = ops
+        .execute_cached_gdn_slice(0, input(source.source_identity(), &raw), None)
+        .unwrap();
+    let originals = actual.original_inputs().unwrap();
+    for (index, (raw, values)) in raw.iter().zip(originals).enumerate() {
+        if index < 4 {
+            assert_eq!(raw.len(), values.len() * 2);
+            for (bytes, value) in raw.chunks_exact(2).zip(values) {
+                assert_eq!(
+                    value.to_bits(),
+                    u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16
+                );
+            }
+        } else {
+            assert_eq!(raw.len(), values.len() * 4);
+            for (bytes, value) in raw.chunks_exact(4).zip(values) {
+                assert_eq!(
+                    value.to_bits(),
+                    u32::from_le_bytes(bytes.try_into().unwrap())
+                );
+            }
+        }
+    }
+
+    let expected = zerogravity_cpu::original_hf::recurrent::cached_step_128(
+        zerogravity_cpu::original_hf::recurrent::CachedGdn128Inputs {
+            source: source.source_identity(),
+            heads: 2,
+            query: originals[0],
+            key: originals[1],
+            value: originals[2],
+            beta: originals[3],
+            decay: originals[4],
+            previous_state: originals[5],
+        },
+    )
+    .unwrap();
+    same(&expected.core_f32, actual.core_f32().unwrap());
+    same(&expected.output_bf16, actual.output_bf16().unwrap());
+    same(&expected.state, actual.next_state_key_major().unwrap());
+    assert_eq!(actual.layer(), 0);
+    assert_eq!(actual.heads(), 2);
+    let scratch = crate::host_math::cached_gdn::CACHED_INPUT_SCRATCH as u64
+        + crate::host_math::cached_gdn::cached_core_scratch(2).unwrap() as u64;
+    assert_eq!(
+        reserved(&admission),
+        baseline + imported_bytes() + result_bytes() + scratch
+    );
+    let snapshot = admission.arbiter.snapshot(&admission.resource).unwrap();
+    assert_eq!(
+        snapshot.per_class[&AllocationClass::Output],
+        imported_bytes() + result_bytes()
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
+    assert!(model.new_session().is_err());
+    assert!(model.forward_plan(8, true, false).is_err());
+    drop(ops);
+    drop(source);
+    same(&expected.state, actual.next_state_key_major().unwrap());
+    same(&expected.output_bf16, actual.output_bf16().unwrap());
+    assert!(reserved(&admission) > 0);
+    drop(actual);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn bad_raw_phase_representation_and_finite_boundaries_refuse() {
+    let id = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
+    let original = raw();
+    input(id, &original);
+    for (index, bad) in [
+        (0, vec![0; 511]),
+        (3, vec![0, 0x40, 0, 0x3f]),
+        (4, f32::NAN.to_le_bytes().repeat(2)),
+        (4, 2f32.to_le_bytes().repeat(2)),
+        (5, f32::INFINITY.to_le_bytes().repeat(32768)),
+    ] {
+        let mut changed = original.clone();
+        changed[index] = bad;
+        assert!(
+            CachedGdnSliceInput::from_bytes(
+                id,
+                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+                2,
+                changed.each_ref().map(|v| v.as_slice())
+            )
+            .is_err()
+        );
+    }
+    // Zero state is legal cached input; it never becomes initial chunk64.
+    let mut zero = original;
+    zero[5] = vec![0; 32768 * 4];
+    input(id, &zero);
+}
+#[test]
+fn real_pressure_pre_cancel_and_after_import_cancel_release_only_owned_leases() {
+    let f = Fixture::new();
+    let loaded = cached_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let raw = raw();
+    let value = input(source.source_identity(), &raw);
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        ops.execute_cached_gdn_slice(0, value, Some(&cancelled))
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), baseline);
+    let core_scratch = crate::host_math::cached_gdn::cached_core_scratch(2).unwrap() as u64;
+    let needed = imported_bytes()
+        + result_bytes()
+        + core_scratch
+        + crate::host_math::cached_gdn::CACHED_INPUT_SCRATCH as u64;
+    let pressure = admission
+        .arbiter
+        .admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: 128 * 1024 * 1024 - baseline - needed + 1,
+            }],
+        )
+        .unwrap();
+    let before = reserved(&admission);
+    assert!(ops.execute_cached_gdn_slice(0, value, None).is_err());
+    assert_eq!(reserved(&admission), before);
+    drop(pressure);
+    cancelled.store(false, Ordering::Release);
+    let mut math =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, Some(&cancelled), None);
+    let operands = math.import_cached_gdn_input(value, 2).unwrap();
+    let imported = reserved(&admission);
+    cancelled.store(true, Ordering::Release);
+    assert!(
+        math.cached_gdn_profiled(
+            source.source_identity(),
+            2,
+            &operands,
+            CachedGdnContract::TorchCpuBf16Avx2Width128
+        )
+        .is_err()
+    );
+    assert_eq!(reserved(&admission), imported);
+    cancelled.store(false, Ordering::Release);
+    let results = math
+        .cached_gdn_profiled(
+            source.source_identity(),
+            2,
+            &operands,
+            CachedGdnContract::TorchCpuBf16Avx2Width128,
+        )
+        .unwrap();
+    let clone = results.next_state.clone();
+    assert!(clone.clone().into_vec().is_err());
+    drop(results);
+    assert_eq!(clone.as_slice().unwrap().len(), 32768);
+    drop(clone);
+    assert_eq!(reserved(&admission), imported);
+    drop(operands);
+    assert_eq!(reserved(&admission), baseline);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn foreign_issuer_strided_inputs_unaware_provider_and_wrong_layer_refuse() {
+    struct Unaware;
+    impl crate::accelerator::MatVecKernel for Unaware {
+        fn try_execute(
+            &self,
+            _: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            panic!("cached GDN must refuse before unaware provider callback")
+        }
+    }
+    let f = Fixture::new();
+    let loaded = cached_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let raw = raw();
+    let value = input(source.source_identity(), &raw);
+    assert!(ops.execute_cached_gdn_slice(3, value, None).is_err());
+    let provider = Unaware;
+    let unaware =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, None, Some(&provider));
+    assert!(unaware.import_cached_gdn_input(value, 2).is_err());
+    assert_eq!(reserved(&admission), baseline);
+    let mut math = crate::host_math::HostMath::for_window_operator_slice(&ops, None, None);
+    let operands = math.import_cached_gdn_input(value, 2).unwrap();
+    let before = reserved(&admission);
+    let strided = math
+        .view(
+            &operands.previous_state,
+            TensorView {
+                offset: 0,
+                rows: 256,
+                width: 1,
+                row_stride: 2,
+            },
+        )
+        .unwrap();
+    let mut bad = operands.clone();
+    bad.query = strided;
+    let error = math
+        .cached_gdn_profiled(
+            source.source_identity(),
+            2,
+            &bad,
+            CachedGdnContract::TorchCpuBf16Avx2Width128,
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
+            .cached_gdn_profiled(
+                source.source_identity(),
+                2,
+                &operands,
+                CachedGdnContract::TorchCpuBf16Avx2Width128
+            )
+            .is_err()
+    );
+    drop(foreign_ops);
+    drop(bad);
+    drop(operands);
+    assert_eq!(reserved(&admission), baseline);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -650,3 +650,55 @@
     }
     Ok(values)
 }
+
+/// Pure cached-token arithmetic over explicit inputs; no StateEffect or exp.
+pub(crate) fn cached_gdn_slice<B: MathematicalBackend>(
+    source: &impl OperatorSliceSource,
+    math: &mut B,
+    layer: usize,
+    input: &crate::accelerator::program::CachedGdnOperands<B::Value>,
+) -> Result<crate::accelerator::program::CachedGdnValues<B::Value>> {
+    use crate::accelerator::program::{ActivationPrecision, CachedGdnContract};
+    let heads = crate::operator_slice::cached_gdn::cached_heads(source, layer)?;
+    let operands = [
+        &input.query,
+        &input.key,
+        &input.value,
+        &input.beta,
+        &input.decay,
+        &input.previous_state,
+    ];
+    let contract = CachedGdnContract::TorchCpuBf16Avx2Width128;
+    contract.validate(heads, &operands.map(|v| math.elements(v)))?;
+    for (index, value) in operands.iter().enumerate() {
+        let expected = if index < 4 {
+            ActivationPrecision::Bf16WidenedF32
+        } else {
+            ActivationPrecision::F32
+        };
+        if math.precision(value) != expected {
+            return Err(Error::State("cached GDN source operand precision".into()));
+        }
+    }
+    let values = math.cached_gdn_profiled(source.source_identity(), heads, input, contract)?;
+    for (value, elements, precision) in [
+        (&values.core_f32, heads * 128, ActivationPrecision::F32),
+        (
+            &values.output_bf16,
+            heads * 128,
+            ActivationPrecision::Bf16WidenedF32,
+        ),
+        (
+            &values.next_state,
+            heads * 128 * 128,
+            ActivationPrecision::F32,
+        ),
+    ] {
+        if math.elements(value) != elements || math.precision(value) != precision {
+            return Err(Error::State(
+                "cached GDN explicit result geometry/precision".into(),
+            ));
+        }
+    }
+    Ok(values)
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/gdn_cached_slice.rs
@@ -0,0 +1,211 @@
+//! Observed cached-token operands only; no coefficient producer/cache owner.
+use super::OperatorSliceSource;
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{
+    CachedGdnContract, CachedGdnOperands, CachedGdnValues, PlanBuilder, PlanValue,
+    ProgramNumericalProfile,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, LoadedQwenModel, Result};
+
+pub(crate) fn cached_heads(source: &impl OperatorSliceSource, layer: usize) -> Result<usize> {
+    if !source.source_identity().is_bound()
+        || source.program_numerical_profile()
+            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
+    {
+        return Err(Error::Unsupported(
+            "cached GDN original profile/layer".into(),
+        ));
+    }
+    let heads = usize::try_from(source.geometry().recurrent_value_heads.unwrap_or(0))
+        .map_err(|_| Error::Limit("cached GDN value heads".into()))?;
+    let keys = usize::try_from(source.geometry().recurrent_key_heads.unwrap_or(0))
+        .map_err(|_| Error::Limit("cached GDN key heads".into()))?;
+    if keys == 0 || !heads.is_multiple_of(keys) || heads == 0 || heads > 64 {
+        return Err(Error::Unsupported(
+            "cached GDN repeated-head geometry".into(),
+        ));
+    }
+    let qkv = source.matrix_descriptor(QwenMatrixRole::GdnQkv(layer))?;
+    let output = source.matrix_descriptor(QwenMatrixRole::GdnOutput(layer))?;
+    let norm = source.parameter_descriptor(QwenParameterRole::GdnNorm(layer))?;
+    if qkv.source != source.source_identity()
+        || output.source != source.source_identity()
+        || qkv.rows != (keys * 2 + heads) * 128
+        || output.columns != heads * 128
+        || norm.source != source.source_identity()
+        || norm.shape != [128]
+        || norm.kind != zerogravity_core::quant::QuantKind::F32
+        || norm.bytes != 512
+    {
+        return Err(Error::Unsupported(
+            "cached GDN original width128/source geometry".into(),
+        ));
+    }
+    Ok(heads)
+}
+
+/// Raw source-function inputs AFTER Q/K repeat; qualifier authenticates archive
+/// provenance. Finite FP32 decay is explicitly supplied, never computed here.
+#[derive(Clone, Copy)]
+pub struct CachedGdnSliceInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    heads: usize,
+    words: [&'a [u8]; 6],
+}
+impl<'a> CachedGdnSliceInput<'a> {
+    /// q/k/v/beta are BF16 LE; decay and key-major prior state are FP32 LE.
+    /// A zero prior still selects CACHED arithmetic, never initial chunk64.
+    pub fn from_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        heads: usize,
+        words: [&'a [u8]; 6],
+    ) -> Result<Self> {
+        let result = Self {
+            source,
+            profile,
+            heads,
+            words,
+        };
+        result.validate()?;
+        Ok(result)
+    }
+    pub fn source_identity(self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    pub fn heads(self) -> usize {
+        self.heads
+    }
+    pub fn raw_operands(self) -> [&'a [u8]; 6] {
+        self.words
+    }
+    pub(crate) fn validate(self) -> Result<()> {
+        if !self.source.is_bound()
+            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported("cached GDN input source/profile".into()));
+        }
+        let mut lengths = [0; 6];
+        for (index, bytes) in self.words.iter().enumerate() {
+            let size = if index < 4 { 2 } else { 4 };
+            if bytes.len() % size != 0 {
+                return Err(Error::Malformed("cached GDN partial raw word".into()));
+            }
+            lengths[index] = bytes.len() / size;
+        }
+        CachedGdnContract::TorchCpuBf16Avx2Width128.validate(self.heads, &lengths)?;
+        for (index, bytes) in self.words.iter().enumerate() {
+            if index < 4 {
+                for raw in bytes.chunks_exact(2) {
+                    let bits = u16::from_le_bytes([raw[0], raw[1]]);
+                    let v = f32::from_bits(u32::from(bits) << 16);
+                    if !v.is_finite() || index == 3 && !(0.0..=1.0).contains(&v) {
+                        return Err(Error::Malformed("cached GDN finite BF16/beta".into()));
+                    }
+                }
+            } else {
+                for raw in bytes.chunks_exact(4) {
+                    let v = f32::from_bits(u32::from_le_bytes(raw.try_into().unwrap()));
+                    if !v.is_finite() || index == 4 && !(0.0..=1.0).contains(&v) {
+                        return Err(Error::Malformed("cached GDN finite FP32/decay".into()));
+                    }
+                }
+            }
+        }
+        Ok(())
+    }
+}
+
+impl LoadedQwenModel {
+    pub fn append_cached_gdn_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &CachedGdnOperands<PlanValue>,
+    ) -> Result<CachedGdnValues<PlanValue>> {
+        self.require_operator_slice()?;
+        crate::forward::cached_gdn_slice(self, builder, layer, input)
+    }
+}
+
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+pub struct CachedGdnSliceOutput {
+    source: SourceIdentity,
+    layer: usize,
+    heads: usize,
+    input: CachedGdnOperands<crate::host_math::HostValue>,
+    values: CachedGdnValues<crate::host_math::HostValue>,
+}
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+impl CachedGdnSliceOutput {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    }
+    pub fn layer(&self) -> usize {
+        self.layer
+    }
+    pub fn heads(&self) -> usize {
+        self.heads
+    }
+    pub fn original_inputs(&self) -> Result<[&[f32]; 6]> {
+        Ok([
+            self.input.query.as_slice()?,
+            self.input.key.as_slice()?,
+            self.input.value.as_slice()?,
+            self.input.beta.as_slice()?,
+            self.input.decay.as_slice()?,
+            self.input.previous_state.as_slice()?,
+        ])
+    }
+    pub fn core_f32(&self) -> Result<&[f32]> {
+        self.values.core_f32.as_slice()
+    }
+    pub fn output_bf16(&self) -> Result<&[f32]> {
+        self.values.output_bf16.as_slice()
+    }
+    pub fn next_state_key_major(&self) -> Result<&[f32]> {
+        self.values.next_state.as_slice()
+    }
+}
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    pub fn append_cached_gdn_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &CachedGdnOperands<PlanValue>,
+    ) -> Result<CachedGdnValues<PlanValue>> {
+        crate::forward::cached_gdn_slice(self, builder, layer, input)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_cached_gdn_slice(
+        &self,
+        layer: usize,
+        input: CachedGdnSliceInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<CachedGdnSliceOutput> {
+        let heads = cached_heads(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let operands = math.import_cached_gdn_input(input, heads)?;
+        let values = crate::forward::cached_gdn_slice(self, &mut math, layer, &operands)?;
+        if cancelled.is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State("cached GDN fragment cancelled".into()));
+        }
+        Ok(CachedGdnSliceOutput {
+            source: self.source_identity(),
+            layer,
+            heads,
+            input: operands,
+            values,
+        })
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -507,6 +507,25 @@
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         let output = output.retain_workspace(workspace)?;
         Ok(output)
+    }
+    fn cached_gdn_profiled(
+        &mut self,
+        source: crate::accelerator::SourceIdentity,
+        heads: usize,
+        input: &crate::accelerator::program::CachedGdnOperands<HostValue>,
+        contract: crate::accelerator::program::CachedGdnContract,
+    ) -> Result<crate::accelerator::program::CachedGdnValues<HostValue>> {
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        {
+            self.execute_cached_gdn(source, heads, input, contract)
+        }
+        #[cfg(not(all(feature = "original-hf-binding", target_os = "linux")))]
+        {
+            let _ = (source, heads, input, contract);
+            Err(Error::Unsupported(
+                "cached GDN requires leased original windows".into(),
+            ))
+        }
     }
     fn causal_conv_silu_profiled(
         &mut self,
@@ -1053,3 +1072,6 @@
 
 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
 mod gdn_conv;
+
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+pub(crate) mod cached_gdn;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/host_math/cached_gdn.rs
@@ -0,0 +1,194 @@
+//! Admitted private transport/core; no provider exp, state owner or publication.
+use super::*;
+use crate::accelerator::program::{CachedGdnContract, CachedGdnOperands, CachedGdnValues};
+use crate::operator_slice::cached_gdn::CachedGdnSliceInput;
+
+pub(crate) const CACHED_INPUT_SCRATCH: usize = 64 * 1024;
+pub(crate) fn cached_core_scratch(heads: usize) -> Result<usize> {
+    heads
+        .checked_mul(128 * 128 * 4 * 6 + 128 * 4 * 16)
+        .and_then(|n| n.checked_add(256 * 1024))
+        .ok_or_else(|| Error::Limit("cached GDN conservative scratch bound".into()))
+}
+impl HostMath<'_, '_, '_> {
+    fn cached_windows(&self) -> Result<&crate::original_binding::OriginalWindowOperators> {
+        if self.region_kernel.is_some()
+            || self.engine.is_some()
+            || self.state.is_some()
+            || self.source.program_numerical_profile()
+                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported(
+                "cached GDN private pinned CPU window contract".into(),
+            ));
+        }
+        match &self.source {
+            HostSource::Windows(source) => Ok(source),
+            HostSource::Loaded(_) => Err(Error::Unsupported(
+                "cached GDN requires admitted window backing".into(),
+            )),
+        }
+    }
+    fn check_cached_cancel(&self) -> Result<()> {
+        if self
+            .cancelled
+            .is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire))
+        {
+            return Err(Error::State("cached GDN private fragment cancelled".into()));
+        }
+        Ok(())
+    }
+    pub(crate) fn import_cached_gdn_input(
+        &self,
+        input: CachedGdnSliceInput<'_>,
+        heads: usize,
+    ) -> Result<CachedGdnOperands<HostValue>> {
+        let source = self.cached_windows()?;
+        input.validate()?;
+        if input.source_identity() != source.source_identity()
+            || input.heads() != heads
+            || input.numerical_profile() != self.source.program_numerical_profile()
+        {
+            return Err(Error::State(
+                "cached GDN input source/profile/head count".into(),
+            ));
+        }
+        self.check_cached_cancel()?;
+        let words = heads
+            .checked_mul(128 * 128 + 3 * 128 + 2)
+            .and_then(|n| n.checked_mul(4))
+            .ok_or_else(|| Error::Limit("cached GDN aggregate input bytes".into()))?;
+        // ALL six retained payloads admitted before the first private Vec.
+        let workspace = Arc::new(
+            source
+                .source()
+                .reserve_operator_buffers(words, CACHED_INPUT_SCRATCH)?,
+        );
+        let mut imported = Vec::with_capacity(6);
+        for (index, raw) in input.raw_operands().iter().enumerate() {
+            let values = if index < 4 {
+                raw.chunks_exact(2)
+                    .map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16))
+                    .collect()
+            } else {
+                raw.chunks_exact(4)
+                    .map(|v| f32::from_bits(u32::from_le_bytes(v.try_into().unwrap())))
+                    .collect()
+            };
+            let precision = if index < 4 {
+                ActivationPrecision::Bf16WidenedF32
+            } else {
+                ActivationPrecision::F32
+            };
+            imported.push(
+                HostValue::typed(
+                    values,
+                    precision,
+                    source.source_identity(),
+                    self.source.program_numerical_profile(),
+                )?
+                .retain_workspace(Some(workspace.clone()))?,
+            );
+        }
+        let [query, key, value, beta, decay, previous_state]: [HostValue; 6] = imported
+            .try_into()
+            .map_err(|_| Error::State("cached GDN imported operand arity".into()))?;
+        self.check_cached_cancel()?;
+        Ok(CachedGdnOperands {
+            query,
+            key,
+            value,
+            beta,
+            decay,
+            previous_state,
+        })
+    }
+    pub(super) fn execute_cached_gdn(
+        &self,
+        identity: crate::accelerator::SourceIdentity,
+        heads: usize,
+        input: &CachedGdnOperands<HostValue>,
+        contract: CachedGdnContract,
+    ) -> Result<CachedGdnValues<HostValue>> {
+        let source = self.cached_windows()?;
+        let operands = [
+            &input.query,
+            &input.key,
+            &input.value,
+            &input.beta,
+            &input.decay,
+            &input.previous_state,
+        ];
+        let lengths = operands.map(|v| v.elements);
+        let (vector, state) = contract.validate(heads, &lengths)?;
+        if identity != source.source_identity()
+            || self.source.metadata().geometry().recurrent_value_heads != Some(heads as u64)
+        {
+            return Err(Error::State(
+                "cached GDN source/configured head count".into(),
+            ));
+        }
+        for (index, value) in operands.iter().enumerate() {
+            let precision = if index < 4 {
+                ActivationPrecision::Bf16WidenedF32
+            } else {
+                ActivationPrecision::F32
+            };
+            if value.source != Some(identity)
+                || value.profile != Some(self.source.program_numerical_profile())
+                || value.precision != precision
+            {
+                return Err(Error::State(
+                    "cached GDN operand source/profile/precision".into(),
+                ));
+            }
+            let workspace = value.data.workspace.as_ref().ok_or_else(|| {
+                Error::State("cached GDN operand missing physical backing".into())
+            })?;
+            if !source.source().same_issuer(workspace.source()) {
+                return Err(Error::State("cached GDN operand foreign issuer".into()));
+            }
+            precision.validate_values(value.as_slice()?)?;
+        }
+        self.check_cached_cancel()?;
+        let bytes = state
+            .checked_add(vector * 2)
+            .and_then(|n| n.checked_mul(4))
+            .ok_or_else(|| Error::Limit("cached GDN aggregate output bytes".into()))?;
+        // Covers every temporary materialized state/product and small reducer
+        // arrays of the existing helper. No native opaque allocator is called.
+        let workspace = Arc::new(
+            source
+                .source()
+                .reserve_operator_buffers(bytes, cached_core_scratch(heads)?)?,
+        );
+        let completed = zerogravity_cpu::original_hf::recurrent::cached_step_128(
+            zerogravity_cpu::original_hf::recurrent::CachedGdn128Inputs {
+                source: identity,
+                heads,
+                query: input.query.as_slice()?,
+                key: input.key.as_slice()?,
+                value: input.value.as_slice()?,
+                beta: input.beta.as_slice()?,
+                decay: input.decay.as_slice()?,
+                previous_state: input.previous_state.as_slice()?,
+            },
+        )?;
+        let make = |values, precision| -> Result<HostValue> {
+            HostValue::typed(
+                values,
+                precision,
+                identity,
+                self.source.program_numerical_profile(),
+            )?
+            .retain_workspace(Some(workspace.clone()))
+        };
+        let result = CachedGdnValues {
+            core_f32: make(completed.core_f32, ActivationPrecision::F32)?,
+            output_bf16: make(completed.output_bf16, ActivationPrecision::Bf16WidenedF32)?,
+            next_state: make(completed.state, ActivationPrecision::F32)?,
+        };
+        self.check_cached_cancel()?;
+        Ok(result)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -234,3 +234,6 @@
 
 #[path = "gdn_conv_slice.rs"]
 pub mod gdn_conv;
+
+#[path = "gdn_cached_slice.rs"]
+pub mod cached_gdn;
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -382,3 +382,6 @@
 
 #[path = "gdn_conv_window_tests.rs"]
 mod gdn_conv;
+
+#[path = "cached_gdn_window_tests.rs"]
+mod cached_gdn;

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/UNUSED_GAMMA_CODEC_FOLLOWUP.patch

Original UTF-8 bytes: 11811; SHA256 `6e41d1e4dff19e737971328c8bda5e50b5acbc4462aaf8acdafaec3d7458b82b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- crates/zerogravity-model-qwen35/src/gdn_cached_slice.rs
+++ crates/zerogravity-model-qwen35/src/gdn_cached_slice.rs
@@ -36,8 +36,12 @@
         || output.columns != heads * 128
         || norm.source != source.source_identity()
         || norm.shape != [128]
-        || norm.kind != zerogravity_core::quant::QuantKind::F32
-        || norm.bytes != 512
+        // Gamma is source metadata only here; cached recurrence never reads it.
+        || !matches!(
+            (norm.kind, norm.bytes),
+            (zerogravity_core::quant::QuantKind::F32, 512)
+                | (zerogravity_core::quant::QuantKind::Bf16, 256)
+        )
     {
         return Err(Error::Unsupported(
             "cached GDN original width128/source geometry".into(),
--- crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs
+++ crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs
@@ -1,4 +1,7 @@
 //! Synthetic ownership/schedule checks; genuine archived replay is separate.
+#[path = "cached_gdn_h48_tests.rs"]
+mod source_gamma;
+
 use super::*;
 use crate::accelerator::program::{CachedGdnContract, CachedGdnOperands};
 use crate::operator_slice::cached_gdn::CachedGdnSliceInput;
--- /dev/null
+++ crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs
@@ -0,0 +1,314 @@
+//! Full-header synthetic H48 codec/owner checks; no trained arithmetic evidence.
+use super::*;
+use crate::accelerator::program::{
+    EmbeddingContract, ProjectionContract, SourceParameter, SourceTensor,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use zerogravity_core::quant::QuantKind;
+
+fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {
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
+        let bf16 = gamma || spec.shape.len() != 1 || spec.name.ends_with("dt_bias");
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
+    let key = PhysicalDeviceKey::new("host", "cached-h48-source-test").unwrap();
+    arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
+    let host = TransferAdmission {
+        arbiter,
+        resource: key,
+        owner: ResourceOwner::new("synthetic", "cached-h48").unwrap(),
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
+fn h48_raw() -> [Vec<u8>; 6] {
+    [
+        0x3f00u16.to_le_bytes().repeat(48 * 128),
+        0xbe80u16.to_le_bytes().repeat(48 * 128),
+        0x3f80u16.to_le_bytes().repeat(48 * 128),
+        0x3f00u16.to_le_bytes().repeat(48),
+        0.75f32.to_le_bytes().repeat(48),
+        (0..48 * 128 * 128)
+            .flat_map(|i| if i % 11 == 0 { -0.0f32 } else { 0.125 }.to_le_bytes())
+            .collect(),
+    ]
+}
+
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
+
+#[test]
+fn actual_f32_and_bf16_source_metadata_allow_only_the_unused_gamma_extent() {
+    let old = Fixture::new();
+    let loaded = cached_source(&old);
+    let (source, _) = windows(&old, &loaded);
+    let old_ops = bind_windows(source, &loaded);
+    assert_eq!(
+        old_ops
+            .parameter_descriptor(QwenParameterRole::GdnNorm(0))
+            .unwrap()
+            .kind,
+        QuantKind::F32
+    );
+    assert_eq!(
+        crate::operator_slice::cached_gdn::cached_heads(&old_ops, 0).unwrap(),
+        2
+    );
+
+    let fixture = Fixture::new();
+    let (source, host, cfg, index) = h48_source(&fixture);
+    let ops = OriginalWindowOperators::bind(
+        source.clone(),
+        &cfg,
+        &digest(&cfg),
+        &index,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )
+    .unwrap();
+    let norm = ops
+        .parameter_descriptor(QwenParameterRole::GdnNorm(0))
+        .unwrap();
+    assert_eq!(
+        (norm.kind, norm.shape.clone(), norm.bytes),
+        (QuantKind::Bf16, vec![128], 256)
+    );
+    assert_eq!(
+        crate::operator_slice::cached_gdn::cached_heads(&ops, 0).unwrap(),
+        48
+    );
+    let baseline = reserved(&host);
+    for (kind, bytes, shape, foreign, profile) in [
+        (
+            QuantKind::F16,
+            256,
+            vec![128],
+            false,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            512,
+            vec![128],
+            false,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::F32,
+            256,
+            vec![128],
+            false,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            256,
+            vec![127],
+            false,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            256,
+            vec![128],
+            true,
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            256,
+            vec![128],
+            false,
+            ProgramNumericalProfile::GgufSourceV1,
+        ),
+    ] {
+        let mut bad = norm.clone();
+        bad.kind = kind;
+        bad.bytes = bytes;
+        bad.shape = shape;
+        if foreign {
+            bad.source = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
+        }
+        let fault = ParameterFault {
+            source: &ops,
+            norm: bad,
+            profile,
+        };
+        assert!(crate::operator_slice::cached_gdn::cached_heads(&fault, 0).is_err());
+        assert_eq!(reserved(&host), baseline);
+    }
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&host), 0);
+}
+
+#[test]
+fn bf16_h48_cached_fragment_retains_real_owners_and_obeys_cancel_and_pool_pressure() {
+    let fixture = Fixture::new();
+    let (source, host, cfg, index) = h48_source(&fixture);
+    let ops = OriginalWindowOperators::bind(
+        source.clone(),
+        &cfg,
+        &digest(&cfg),
+        &index,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )
+    .unwrap();
+    let raw = h48_raw();
+    let input = CachedGdnSliceInput::from_bytes(
+        source.source_identity(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        48,
+        raw.each_ref().map(Vec::as_slice),
+    )
+    .unwrap();
+    let baseline = reserved(&host);
+    assert!(
+        ops.execute_cached_gdn_slice(0, input, Some(&AtomicBool::new(true)))
+            .is_err()
+    );
+    assert_eq!(reserved(&host), baseline);
+    let needed = (48 * (16384 + 3 * 128 + 2) * 4
+        + 48 * (16384 + 2 * 128) * 4
+        + crate::host_math::cached_gdn::CACHED_INPUT_SCRATCH
+        + crate::host_math::cached_gdn::cached_core_scratch(48).unwrap()) as u64;
+    let pressure = host
+        .arbiter
+        .admit(
+            &host.resource,
+            host.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: 128 * 1024 * 1024 - baseline - needed + 1,
+            }],
+        )
+        .unwrap();
+    let pressured = reserved(&host);
+    assert!(ops.execute_cached_gdn_slice(0, input, None).is_err());
+    assert_eq!(reserved(&host), pressured);
+    drop(pressure);
+    let result = ops.execute_cached_gdn_slice(0, input, None).unwrap();
+    assert_eq!(result.heads(), 48);
+    assert_eq!(result.core_f32().unwrap().len(), 6144);
+    assert_eq!(result.output_bf16().unwrap().len(), 6144);
+    assert_eq!(result.next_state_key_major().unwrap().len(), 786432);
+    assert_eq!(reserved(&host), baseline + needed);
+    let expected = result
+        .next_state_key_major()
+        .unwrap()
+        .iter()
+        .map(|v| v.to_bits())
+        .collect::<Vec<_>>();
+    drop(ops);
+    drop(source);
+    assert_eq!(
+        result
+            .next_state_key_major()
+            .unwrap()
+            .iter()
+            .map(|v| v.to_bits())
+            .collect::<Vec<_>>(),
+        expected
+    );
+    drop(result);
+    assert_eq!(reserved(&host), 0);
+    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
+}

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/H48_REPLAY_PREPARATION.patch

Original UTF-8 bytes: 103434; SHA256 `5b54f199c0b76430709e6058109d052b8dba7896e7ce29571bc9eec6b410a439`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- old-H16/Cargo.toml
+++ H48/Cargo.toml
@@ -1,5 +1,5 @@
 [package]
-name = "original-cached-gdn-archive-replay"
+name = "original27b-cached-gdn-archive-replay"
 version = "0.0.0"
 edition = "2024"
 publish = false
@@ -19,5 +19,5 @@
 codegen-units = 1
 
 [[bin]]
-name = "original-cached-gdn-archive-replay"
+name = "original27b-cached-gdn-archive-replay"
 path = "main.rs"
--- old-H16/ROOT.md
+++ H48/ROOT.md
@@ -1,80 +1,55 @@
-# ROOT-only archived cached GDN gate
+# ROOT-only authentic H48 cached typed replay
 
-Author: spine_review. Parent: model-execution.
+Author: spine_review. Original .8B931d/57a/stride proposal and captures unchanged.
+No author build/tests/Torch/provider/operator/model/GPU execution or payload hash.
 
-- Source-only preparation. No author build, tests, operator/provider/model,
-  container, GPU, MAIN or shared-plan changes. Preserve original57a4e64 freeze.
-- Parent `STRIDE_FOLLOWUP.patch` changes ONLY rows256→2 and bad.query→bad.decay.
-  Original F32 backing, same issuer and valid two-element decay shape now reach
-  the intended contiguous guard. No production change in that followup.
-- `.8B` actual public step1 archive provides all six cached inputs and all three
-  results. Whole archive SHA, selected byte SHA, dtype/shape/stride, previous
-  real chunk return and98 historical fullV/cache checks are mandatory.
-- This is a replay against an already captured authentic public call. Exporter
-  reads actual original archive bytes; it calls no Torch/operator/model and
-  derives no expected value. It does not claim a new public runtime execution.
-- Imported exp(g) is the actual archived decay. No native exp, live coefficient
-  producer, initial-token substitute, state publication, canonical transpose,
-  source mutation, model/session admission or new engine is introduced.
-- Native helper opens and verifies the genuine complete `.8B` shard through
-  existing SourceWeights/OriginalWindowOperators and calls ONLY the public
-  execute_cached_gdn_slice. All266240 output words are compared to independent
-  archive bits, plus268320 retained input words. Signed zero/subnormals survive.
-- Existing owned_lifetime.py copied byte-exact; owned_native.py retains original
-  source_files/execute bodies, changing only scope wording and removing unused
-  coefficient main. Exec-FD SHA, exact four CPUs, noGPU environment,512MiB AS,
-  finite900s child/2400s outer deadline,4MiB combined logs, WNOWAIT/group-bound
-  TERM/grace/two-KILL-before-any-reap/independent reap/EOF remain.
-- Host ledger128MiB:64MiB sealed source +two4MiB bindings +8MiB qualifier control.
-  Aggregate six-input and three-result payloads and conservative helper scratch
-  use original same-source host reservations. Real pool pressure/pre-cancel/
-  wrong-layer refusal, exact live/retained accounting and last-drop0 required.
-  Configured ledger and AS limits are not a whole-host physical-RAM claim.
-- ROOT must independently review/merge both prepared source patches, record the
-  complete actual build/source list and attest the resulting native binary.
-  `--implementation` is an explicit ROOT review/build attestation, not proof
-  obtained merely by providing hashes. All changed paths are required; exact
-  new child and unchanged original arithmetic pins are checked independently.
-  Parent Program/HostMath merges must preserve concurrent coefficient nodes.
+- Source: genuine original27B public V3 root-reference-v3-1, actual11 initial
+  paddedchunk64→353 cached, original BF16 source gamma/Alog/dt. Existing independent
+  validation pins all genuine plain/observed outputs and actual source/cache state.
+- Export ALL804960 original inputs and798720 outputs, exact signed bits/layout:
+  q/k/v/beta BF16; observed exp(g)F32; previous786432-word key-major state;
+  core6144F32, return6144BF16, next786432F32. Entire actual archives are authenticated;
+  unselected original chunk exp masks may contain-Inf, selectednine stages must be
+  finite. No native exp, head repetition, reconstructed expected values or cache.
+- Source factory.rs +source-pin.json are byte-identical to accepted27B factory
+  and7058B4c74fa28 acquisition metadata. SAME VerifiedTensorSource full18-shard
+  55.563GB bounded scan/header851text closure, no whole-source Vec/mmap or4GiB
+  resident-guard change. All source scans happen ONLY when ROOT schedules them.
+- Native uses SAME execute_cached_gdn_slice/PlanBuilder/HostMath/CPU128 reducer.
+  Only separately reviewed typed followup26d9 permits actual unused metadata
+  gammaBF16[128]/256B alongside existingF32[128]/512B. Gamma is never consumed by
+  cached recurrence. Gatednorm/Alog/exp/initial/fullGDN/112StateRoots/session/GPU
+  and complete27B inference remain closed. No separate primitive fallback.
+- SAME configured128MiB host ledger, source64MiB+bindings8MiB. Qualifier control
+ 16MiB covers actual rawinputs3,182,880B +largest expected3,145,728B +largest
+  emitted-F32Vec3,145,728B +two<=1MiB metadata buffers and bounded JSON overhead.
+  This is a larger actual reservation, not a widened pool or physical-RAM claim.
+  Import3,219,840B +results3,194,880B +scratch19,595,264B; live118,284,672B<128MiB.
+  Real pool exhaustion/pre-cancel/wrongFA-layer refusal, retained source+allnine
+  stages after caller drops, and final reservations0/nonfenced are mandatory.
+- Exact old owned_lifetime.py and owned_native.py retained. Exec-FD-bound child,
+  four CPUs/noGPU,512MiBAS/900s child/2400s outer/4MiBperfile logs. Original
+  WNOWAIT owned TERM/grace/KILL-before-reap/independent reap/groupabsence/EOF.
+  Closed process observation is not a GPUReady or recovery mechanism.
 
-## Availability limits
+ROOT sequence after review/merge and timing holds:
 
-- 2B actual paired generic archive has final recurrent state and QKV/conv, but
-  lacks cached q/k/v/beta/decay/core-call inputs/targets. Do not reconstruct them.
-- 27B actual public reference contains embedding/InputNorm/QKV only. No authentic
-  H48 cached operands/results exist in the inspected archives; historical27B
-  runs do not supply them. Current source-level cached_heads also requires F32
-  GdnNorm while original27B has BF16 gamma. Thus H48 dimension compatibility
-  does not make the current original27B slice executable. No dtype gate changed.
-- `availability.json` records actual inspected archive headers/reference scope.
-  No synthetic, repeated `.8B`, pre-convolution QKV or guessed state substitutes.
-  Future2B/H48 need actual original public initial+cached capture with immutable
-  source/runtime pins and independently transparent full-model evidence first.
+1. Run five prepared stdlib test_contracts.py methods, then export_reference.py
+   --new-output ABS_THIS_REPLAY/root-reference-1. Exporter calls no model/operator.
+2. Review/apply original57a cacheddelta+stride03dc+H48metadata26d9 by hunk. Keep
+   concurrent coefficient/pure-role nodes. Run focused/strict/full isolated gates.
+3. Build this Cargo.toml with ROOT's sole offline Cargo coordinator. Preserve actual
+   lock/command/log/source/binary proof. Write original required source attestation:
+   ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE, prepared_manifest_sha256=this freeze,
+   full_model_or_StateRoot_or_exp_or_GPU_qualified=false and complete files pins.
+   It must include all required_build_paths, expected updated children and exact
+   local Cargo/main/factory/source-pin/source-bindings; markers alone are not proof.
+4. Run run_replay.py with original arguments --binary/--binary-sha256,
+   --reference-directory ABS_THIS_REPLAY/root-reference-1, --implementation,
+   --new-output ABS_THIS_REPLAY/root-native-1; then validate_replay.py with output,
+   reference-directory and implementation. Compare ALL1,603,680 words before and
+   after caller drops, including full786432 states, never tolerance or prefix zip.
 
-## ROOT sequence
-
-Run only after current benchmark/GPU holds and source integration permit it.
-Set `RUN` to this `replay/` directory. All output directories are fresh.
-
-```bash
-python3 "$RUN/test_contracts.py"
-python3 "$RUN/export_reference.py" --new-output "$RUN/root-reference-1"
-CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
-  cargo build --release --offline --manifest-path "$RUN/Cargo.toml"
-python3 "$RUN/run_replay.py" --binary "$ROOT_ATTESTED_BINARY" \
-  --binary-sha256 "$ROOT_BINARY_SHA" \
-  --reference-directory "$RUN/root-reference-1" \
-  --implementation "$ROOT_SOURCE_ATTESTATION" --new-output "$RUN/root-native-1"
-python3 "$RUN/validate_replay.py" --output "$RUN/root-native-1" \
-  --reference-directory "$RUN/root-reference-1" --implementation "$ROOT_SOURCE_ATTESTATION"
-```
-
-ROOT source attestation JSON: `status=ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE`,
-`prepared_manifest_sha256=<this prepared-manifest.json SHA>`,
-`full_model_or_StateRoot_or_exp_or_GPU_qualified=false`, and `files` containing
-the complete reviewed actual build-source rows `{path,bytes,sha256}`. Required
-paths and exact child/dependency pins live in prepared-manifest.json. Preserve
-actual Cargo lock/artifact/build/source receipts independently.
-
-Actual acceptance still requires focused/strict feature tests, original source
-routes and full isolated repository gate. No new acceptance is claimed here.
+Both original .8B/H16 and new genuine27B/H48 gates remain required independently.
+No new receipt/public-hash declaration establishes native support before ROOT's
+actual compile, independent trained comparison and lifecycle/ledger qualification.
--- old-H16/contracts.py
+++ H48/contracts.py
@@ -10,7 +10,7 @@
 INPUTS = ('query', 'key', 'value', 'beta', 'decay', 'previous_state')
 OUTPUTS = ('core_f32', 'output_bf16', 'next_state')
 STAGES = INPUTS + OUTPUTS
-PROFILE = 'source-36-f32-restored'
+PROFILE = 'original27B-raw-source-bf16-v1'
 
 def require(ok, message):
     if not ok:
@@ -69,6 +69,8 @@
         pin(row['path'], row)
     for row in document['stride_followup']:
         pin(row['path'], row)
+    for row in document['H48_metadata']:
+        pin(row['path'], row)
     return sha(HERE/'prepared-manifest.json')
 
 def implementation(path):
@@ -99,7 +101,7 @@
     require(all(bits & mask != mask for bits, in struct.iter_unpack(fmt, raw)), 'finite raw input/output')
 
 def archive(row):
-    raw = read(row['path'], 32*1024*1024, row)
+    raw = read(row['path'], 64*1024*1024, row)
     n, = struct.unpack_from('<Q', raw)
     require(1 < n <= 1024*1024 and 8+n < len(raw), 'bounded original archive header')
     header = json.loads(raw[8:8+n], object_pairs_hook=unique)
@@ -119,7 +121,7 @@
                 hi-lo == count*(2 if dtype == 'BF16' else 4) and 8+n+hi <= len(raw),
                 'complete typed original archive extent')
         payload = raw[8+n+lo:8+n+hi]
-        finite(payload, dtype)
+        # Unselected authentic chunk masks may contain -Inf; selected stages are finite-checked.
         tensors[name] = {'dtype': dtype, 'shape': shape, 'raw': payload}
         spans.append((lo, hi))
     end = 0
--- old-H16/export_reference.py
+++ H48/export_reference.py
@@ -1,110 +1,157 @@
-"""ROOT ONLY archived genuine public reference extraction; no arithmetic."""
+"""ROOT ONLY genuine H48 archive transport; no operator or expected-value math."""
 import argparse
 import json
 from pathlib import Path
-from contracts import HERE, INPUTS, OUTPUTS, PROFILE, STAGES, archive, digest, frozen, pin, read_json, require, sha, write
+
+from contracts import (HERE, INPUTS, OUTPUTS, PROFILE, STAGES, archive, digest,
+                       finite, frozen, pin, raw_file, read_json, require, sha, write)
 
 NAMES = {
-    'query': ('recurrent.input.query', 'BF16', [1,1,16,128]),
-    'key': ('recurrent.input.key', 'BF16', [1,1,16,128]),
-    'value': ('recurrent.input.value', 'BF16', [1,1,16,128]),
-    'beta': ('recurrent.input.beta', 'BF16', [1,1,16]),
-    'decay': ('recurrent.decay_and_prior.decay_t', 'F32', [1,16,1,1]),
-    'previous_state': ('recurrent.input.initial_state', 'F32', [1,16,128,128]),
-    'core_f32': ('recurrent.fp32_core.core_attn_out', 'F32', [1,16,1,128]),
-    'output_bf16': ('recurrent.return.actual_return_0', 'BF16', [1,1,16,128]),
-    'next_state': ('recurrent.return.actual_return_1', 'F32', [1,16,128,128]),
+    'query': ('recurrent.input.query', 'BF16', [1, 1, 48, 128]),
+    'key': ('recurrent.input.key', 'BF16', [1, 1, 48, 128]),
+    'value': ('recurrent.input.value', 'BF16', [1, 1, 48, 128]),
+    'beta': ('recurrent.input.beta', 'BF16', [1, 1, 48]),
+    'decay': ('recurrent.decay_and_prior.decay_t', 'F32', [1, 48, 1, 1]),
+    'previous_state': ('recurrent.input.initial_state', 'F32', [1, 48, 128, 128]),
+    'core_f32': ('recurrent.fp32_core.core_attn_out', 'F32', [1, 48, 1, 128]),
+    'output_bf16': ('recurrent.return.actual_return_0', 'BF16', [1, 1, 48, 128]),
+    'next_state': ('recurrent.return.actual_return_1', 'F32', [1, 48, 128, 128]),
 }
+
+
+def public_scope(capture, validation, binding):
+    require(capture['status'] == 'CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION' and
+            capture['source_repo'] == 'Qwen/Qwen3.8-27B' and
+            capture['source_revision'] == '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0' and
+            capture['snapshot'] == '98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e' and
+            capture['parameter_profile'] == PROFILE and capture['tokens'] == [11, 353] and
+            capture['all_source_parameters_BF16'] is True and
+            capture['full_model_StateRoot_native_exp_GPU_eligible'] is False,
+            'authentic original27B raw-source profile, never small-model restored gamma')
+    require(validation['status'] == 'PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY' and
+            validation['manifest_sha256'] == binding['capture_manifest']['sha256'] and
+            validation['source_plan_sha256'] == capture['source_plan_sha256'] == binding['source_plan']['sha256'] and
+            validation['heads'] == 48 and validation['teachers'] == [11, 353] and
+            validation['cached_input_words_available'] == 804960 and
+            validation['cached_output_words_available'] == 798720 and
+            validation['full_model_112_StateRoots_native_exp_allocation_GPU_or_performance_qualified'] is False,
+            'independent complete original public H48 validation first')
+    runtime = capture['runtime']
+    require(runtime['torch'] == '2.13.0+cu129' and
+            runtime['torch_git'] == 'cf30153c4c131c8164ee7798e5022d810682e2cb' and
+            runtime['transformers'] == '5.17.0' and runtime['cpu_capability'] == 'AVX2' and
+            runtime['threads'] == 4 and runtime['interop_threads'] == 1 and runtime['cuda_available'] is False and
+            runtime['modeling_sha256'] == '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b',
+            'actual pinned original public CPU runtime, no GPU/FLA profile inference')
+    norm = capture['norm_profile']
+    require(norm['heads'] == 48 and norm['head_width'] == 128 and
+            norm['raw_gamma_dtype'] == norm['weighted_dtype'] == 'torch.bfloat16',
+            'real original gamma BF16; no norm math eligibility follows')
+
 
 def authenticated():
     binding = read_json(HERE/'source-bindings.json')
-    for key in ('capture_manifest','capture_validation','step0_metadata','step1_metadata'):
+    for key in ('capture_manifest', 'capture_validation', 'capture_invocation', 'source_plan',
+                'capture_preparation', 'factory_report', 'step0_metadata', 'step1_metadata'):
         pin(binding[key]['path'], binding[key])
     capture = read_json(binding['capture_manifest']['path'])
     validation = read_json(binding['capture_validation']['path'])
-    require(capture['profile'] == PROFILE and capture['tokens'] == [11,353] and
-            capture['source_reverified_after_capture'] is True and
-            capture['modeling_sha256'] == '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b',
-            'original pinned source/function/profile capture')
-    require(validation['validated'] is True and validation['raw_prior_capture_checks'] == 98 and
-            validation['second_token_initial_state_is_first_token_return_bits'] is True,
-            'accepted actual fullV/cache transparency and first→cached link')
-    before, actual = archive(binding['step0_archive']), archive(binding['step1_archive'])
-    metadata = read_json(binding['step1_metadata']['path'])
-    require(metadata['step'] == 1 and metadata['input_token'] == 353 and
-            metadata['actual_return_matches_cache'] is True and
-            metadata['actual_operator_calls'] == {'l2norm':2,'torch_recurrent_gated_delta_rule':1},
-            'one actual cached call, not initial chunk or synthetic state')
+    public_scope(capture, validation, binding)
+    invocation = read_json(binding['capture_invocation']['path'])
+    require(invocation['status'] == 'EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION' and
+            invocation['exit_code'] == 0 and invocation['cleanup_errors'] == [] and
+            invocation['source_FD_cleanup_errors'] == [] and invocation['source_FDs_closed'] is True and
+            invocation['owned_container_absent_after_reap'] is True and
+            invocation['source_before'] == invocation['source_after'] and
+            sha(binding['capture_invocation']['path']) == validation['invocation_sha256'],
+            'prior public reference source/owner lifecycle closed, no new runtime invocation')
+    before, actual, plain = (archive(binding[key]) for key in ('step0_archive', 'step1_archive', 'plain_archive'))
+    steps = capture['steps']
+    require(len(steps) == 2 and [s['teacher_token'] for s in steps] == [11, 353] and
+            [s['step'] for s in steps] == [0, 1] and
+            all(s['plain_observed_all_outputs_and_state_exact'] is True for s in steps),
+            'both genuine phases, plain/observed complete output/cache transparency')
+    for i, record in enumerate(steps):
+        require(read_json(binding[f'step{i}_metadata']['path']) ==
+                dict(record, plain_observed_all_outputs_and_state_exact=False),
+                'original prevalidation metadata retained unchanged')
+        require(record['archive_bytes'] == binding[f'step{i}_archive']['bytes'] and
+                record['archive_sha256'] == binding[f'step{i}_archive']['sha256'],
+                'all actual complete archive identity, not selected expected fragments')
+    require(steps[1]['actual_calls'] == {'Qwen3_5GatedDeltaNet.forward': 1,
+            'Qwen3_5RMSNormGated.forward': 1, 'causal_conv1d_update': 1,
+            'l2norm': 2, 'torch_recurrent_gated_delta_rule': 1},
+            'one actual cached source call; no zero-state initial substitute')
     prior = before['chunk.return.actual_return_1']
-    require(prior['dtype'] == 'F32' and prior['shape'] == [1,16,128,128] and
+    require(prior['dtype'] == 'F32' and prior['shape'] == [1, 48, 128, 128] and
             prior['raw'] == actual['recurrent.input.initial_state']['raw'] ==
-            actual['recurrent.decay_and_prior.last_recurrent_state']['raw'],
-            'full actual initial chunk returned state equals original cached prior')
+            actual['recurrent.decay_and_prior.last_recurrent_state']['raw'] == plain['step-0.state']['raw'] and
+            actual['recurrent.return.actual_return_1']['raw'] == plain['step-1.state']['raw'],
+            'all786432 actual source state words link initial→cached and unchanged plain cache')
     selected = {}
     for name, (key, dtype, shape) in NAMES.items():
-        tensor, description = actual[key], metadata['tensors'][key]
-        require(tensor['dtype'] == dtype and tensor['shape'] == shape and
-                description['shape'] == shape and description['contiguous'] is True and
+        tensor, description = actual[key], steps[1]['tensors'][key]
+        require(tensor['dtype'] == dtype and tensor['shape'] == description['shape'] == shape and
+                description['contiguous'] is True and description['device'] == 'cpu' and
+                description['dtype'] == ('torch.bfloat16' if dtype == 'BF16' else 'torch.float32') and
                 description['raw_sha256'] == digest(tensor['raw']) and
-                description['bytes'] == len(tensor['raw']), 'actual full typed cached stage: '+name)
+                description['bytes'] == len(tensor['raw']), 'actual full typed H48 stage: '+name)
+        finite(tensor['raw'], dtype)
         selected[name] = (tensor, description)
     return binding, selected
+
 
 def validate(directory):
     binding, selected = authenticated()
     inputs, reference = read_json(directory/'inputs.json'), read_json(directory/'reference.json')
-    require(inputs['schema'] == 'zerogravity.cached-gdn.archived-inputs.v1' and
-            reference['schema'] == 'zerogravity.cached-gdn.archived-reference.v1' and
+    require(inputs['schema'] == 'zerogravity.cached-gdn.archived-inputs.h48.v1' and
+            reference['schema'] == 'zerogravity.cached-gdn.archived-reference.h48.v1' and
             inputs['source_bindings_sha256'] == reference['source_bindings_sha256'] == sha(HERE/'source-bindings.json') and
             reference['inputs_sha256'] == sha(directory/'inputs.json') and
-            inputs['model'] == reference['model'] == '0.8b' and
-            inputs['heads'] == reference['heads'] == 16 and inputs['layer'] == reference['layer'] == 0 and
-            inputs['profile'] == reference['profile'] == PROFILE and
-            reference['expected_output_words'] == 266240 and
-            reference['external_decay_no_native_exp'] is True,
-            'exact genuine cached artifact namespace/profile')
-    from contracts import raw_file
+            inputs['model'] == reference['model'] == '27b' and
+            inputs['heads'] == reference['heads'] == 48 and inputs['layer'] == reference['layer'] == 0 and
+            inputs['profile'] == reference['profile'] == PROFILE and reference['expected_output_words'] == 798720 and
+            reference['external_decay_no_native_exp'] is True, 'exact genuine H48 artifact namespace/profile')
     require(set(inputs['operands']) == set(INPUTS) and set(reference['targets']) == set(OUTPUTS),
-            'all six actual inputs and all three actual outputs')
+            'all six inputs and three results')
     for name in STAGES:
         row = inputs['operands'][name] if name in INPUTS else reference['targets'][name]
         tensor, description = selected[name]
         require(row['original_tensor'] == NAMES[name][0] and row['dtype'] == tensor['dtype'] and
                 row['shape'] == tensor['shape'] and row['original_stride'] == description['stride'] and
-                raw_file(directory, row) == tensor['raw'], 'all signed original cached words: '+name)
-    require({p.name for p in directory.iterdir()} == {'inputs.json','reference.json',*(name+'.raw' for name in STAGES)},
-            'complete immutable extracted artifact tree')
-    return {'status':'PASS_ARCHIVED_PUBLIC_CACHED_OPERANDS_AND_TARGETS_ONLY',
-            'inputs_sha256':sha(directory/'inputs.json'), 'reference_sha256':sha(directory/'reference.json'),
-            'source_bindings_sha256':sha(HERE/'source-bindings.json'), 'expected_output_words':266240,
-            'full_model_or_live_decay_or_StateRoot_or_H48_qualified':False}
+                raw_file(directory, row) == tensor['raw'], 'all signed source words: '+name)
+    require({p.name for p in directory.iterdir()} == {'inputs.json', 'reference.json', *(name+'.raw' for name in STAGES)},
+            'complete extracted artifact tree')
+    return dict(status='PASS_ARCHIVED_PUBLIC_H48_OPERANDS_AND_TARGETS_ONLY',
+        inputs_sha256=sha(directory/'inputs.json'), reference_sha256=sha(directory/'reference.json'),
+        source_bindings_sha256=sha(HERE/'source-bindings.json'), expected_output_words=798720,
+        full_model_or_live_decay_or_112StateRoots_or_GatedNorm_or_GPU_qualified=False)
+
 
 def main():
     parser = argparse.ArgumentParser(description=__doc__)
     parser.add_argument('--new-output', type=Path, required=True)
     args = parser.parse_args()
     frozen()
-    binding, selected = authenticated()
-    require(args.new_output.resolve().parent == HERE, 'fresh own-run export only')
+    _, selected = authenticated()
+    require(args.new_output.resolve().parent == HERE, 'fresh own export only')
     args.new_output.mkdir(mode=0o700)
     rows = {}
     for name, (tensor, description) in selected.items():
         raw = tensor['raw']
-        file = name+'.raw'
-        with (args.new_output/file).open('xb') as opened:
+        with (args.new_output/(name+'.raw')).open('xb') as opened:
             opened.write(raw)
-        rows[name] = {'file':file,'bytes':len(raw),'sha256':digest(raw),'dtype':tensor['dtype'],
-                      'shape':tensor['shape'],'original_stride':description['stride'],
-                      'original_tensor':NAMES[name][0]}
-    common = {'model':'0.8b','heads':16,'layer':0,'teacher_tokens':[11,353],'profile':PROFILE,
-              'source_bindings_sha256':sha(HERE/'source-bindings.json'),
-              'external_decay_no_native_exp':True,'no_new_public_operator_or_model_execution':True}
-    write(args.new_output/'inputs.json',dict(common,schema='zerogravity.cached-gdn.archived-inputs.v1',
-          operands={name:rows[name] for name in INPUTS}))
-    write(args.new_output/'reference.json',dict(common,schema='zerogravity.cached-gdn.archived-reference.v1',
-          inputs_sha256=sha(args.new_output/'inputs.json'),targets={name:rows[name] for name in OUTPUTS},
-          expected_output_words=266240))
-    print(json.dumps(validate(args.new_output),sort_keys=True))
+        rows[name] = dict(file=name+'.raw', bytes=len(raw), sha256=digest(raw), dtype=tensor['dtype'],
+                          shape=tensor['shape'], original_stride=description['stride'], original_tensor=NAMES[name][0])
+    common = dict(model='27b', heads=48, layer=0, teacher_tokens=[11, 353], profile=PROFILE,
+        source_bindings_sha256=sha(HERE/'source-bindings.json'), external_decay_no_native_exp=True,
+        no_new_public_operator_or_model_execution=True)
+    write(args.new_output/'inputs.json', dict(common, schema='zerogravity.cached-gdn.archived-inputs.h48.v1',
+                                            operands={name: rows[name] for name in INPUTS}))
+    write(args.new_output/'reference.json', dict(common, schema='zerogravity.cached-gdn.archived-reference.h48.v1',
+        inputs_sha256=sha(args.new_output/'inputs.json'), targets={name: rows[name] for name in OUTPUTS}, expected_output_words=798720))
+    print(json.dumps(validate(args.new_output), sort_keys=True))
+
 
 if __name__ == '__main__':
     main()
--- old-H16/main.rs
+++ H48/main.rs
@@ -1,4 +1,5 @@
 //! ROOT ONLY: actual archived cached operands/outputs through shared HostMath.
+mod factory;
 use serde_json::{Value, json};
 use sha2::{Digest, Sha256};
 use std::collections::BTreeMap;
@@ -8,11 +9,8 @@
 use std::os::unix::fs::OpenOptionsExt;
 use std::path::{Component, Path};
 use std::sync::atomic::AtomicBool;
-use zerogravity_artifacts::artifact::Artifact;
 use zerogravity_artifacts::cache::TransferAdmission;
-use zerogravity_artifacts::source_weights::{
-    ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits,
-};
+use zerogravity_artifacts::source_weights::VerifiedTensorSource;
 use zerogravity_core::accelerator::program::ProgramNumericalProfile;
 use zerogravity_core::resource_arbiter::{
     AllocationClass, AllocationRequest, DeviceBudgetSnapshot, PhysicalDeviceKey, ResourceArbiter,
@@ -26,7 +24,7 @@
 const MIB: u64 = 1024 * 1024;
 const HOST: u64 = 128 * MIB;
 const SOURCE_META: u64 = 64 * MIB;
-const CONTROL: u64 = 8 * MIB;
+const CONTROL: u64 = 16 * MIB;
 const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
 const STAGES: [&str; 9] = [
     "query",
@@ -39,10 +37,10 @@
     "output_bf16",
     "next_state",
 ];
-const INPUT_BYTES: u64 = 16 * (16384 + 3 * 128 + 2) * 4;
-const OUTPUT_BYTES: u64 = 16 * (16384 + 2 * 128) * 4;
+const INPUT_BYTES: u64 = 48 * (16384 + 3 * 128 + 2) * 4;
+const OUTPUT_BYTES: u64 = 48 * (16384 + 2 * 128) * 4;
 const INPUT_SCRATCH: u64 = 64 * 1024;
-const CORE_SCRATCH: u64 = 16 * (6 * 16384 * 4 + 16 * 128 * 4) + 256 * 1024;
+const CORE_SCRATCH: u64 = 48 * (6 * 16384 * 4 + 16 * 128 * 4) + 256 * 1024;
 fn ensure(ok: bool, message: impl Into<String>) -> Proof<()> {
     if ok {
         Ok(())
@@ -118,71 +116,15 @@
         "all private failed-call buffers released",
     )
 }
-fn source(binding: &Value, host: &TransferAdmission) -> Proof<(VerifiedTensorSource, Vec<u8>)> {
-    let root = Path::new(text(&binding["source_directory"])?);
-    let files = binding["source_files"]
-        .as_array()
-        .ok_or("original source file pins")?;
-    let get = |name: &str| -> Proof<&Value> {
-        files
-            .iter()
-            .find(|row| row["path"] == name)
-            .ok_or_else(|| "required source file pin".into())
-    };
-    let config = pinned(&root.join("config.json"), get("config.json")?, MIB as usize)?;
-    let index = pinned(
-        &root.join("model.safetensors.index.json"),
-        get("model.safetensors.index.json")?,
-        MIB as usize,
-    )?;
-    let parsed: Value = serde_json::from_slice(&index)?;
-    let shard_names = parsed["weight_map"]
-        .as_object()
-        .ok_or("complete original index")?
-        .values()
-        .map(text)
-        .collect::<Proof<std::collections::BTreeSet<_>>>()?;
-    ensure(shard_names.len() == 1, "approved single source shard")?;
-    let mut shards = Vec::new();
-    for name in shard_names {
-        relative(&json!(name))?;
-        let expected = get(name)?;
-        let file = OpenOptions::new()
-            .read(true)
-            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
-            .open(root.join(name))?;
-        ensure(
-            file.metadata()?.is_file()
-                && file.metadata()?.len() == expected["bytes"].as_u64().ok_or("shard length")?,
-            "same original full regular shard",
-        )?;
-        shards.push(ExpectedSourceShard::new(
-            name.into(),
-            Artifact::from_file(file, Path::new(name))?,
-            expected["bytes"].as_u64().ok_or("shard length")?,
-            text(&expected["sha256"])?.into(),
-        )?);
-    }
-    let source = SourceWeights::prepare_verified_windows(
-        &index,
-        text(&get("model.safetensors.index.json")?["sha256"])?,
-        shards,
-        WindowOpenLimits {
-            max_source_bytes: 5 * 1024 * MIB,
-            max_metadata_bytes: SOURCE_META,
-            max_shards: 1,
-            max_tensors: 1024,
-            scan_chunk_bytes: MIB as usize,
-        },
-        host,
-    )?
-    .open()?;
-    ensure(
-        source.metadata_reserved_bytes() == SOURCE_META && source.source_identity().is_bound(),
-        "sealed original issuer and admitted metadata",
-    )?;
-    Ok((source, config))
-}
+fn source(_binding: &Value, host: &TransferAdmission) -> Proof<(VerifiedTensorSource, Vec<u8>)> {
+    let ready = factory::open_source(host)?;
+    ensure(
+        ready.source.metadata_reserved_bytes() == SOURCE_META,
+        "same accepted original27B source metadata reservation",
+    )?;
+    Ok((ready.source, ready.config))
+}
+
 fn row<'a>(inputs: &'a Value, reference: &'a Value, name: &str) -> &'a Value {
     if STAGES[..6].contains(&name) {
         &inputs["operands"][name]
@@ -193,9 +135,9 @@
 fn raw(root: &Path, descriptor: &Value, name: &str) -> Proof<Vec<u8>> {
     let bf16 = matches!(name, "query" | "key" | "value" | "beta" | "output_bf16");
     let words = match name {
-        "beta" | "decay" => 16,
-        "previous_state" | "next_state" => 262144,
-        _ => 2048,
+        "beta" | "decay" => 48,
+        "previous_state" | "next_state" => 786432,
+        _ => 6144,
     };
     ensure(
         descriptor["dtype"] == if bf16 { "BF16" } else { "F32" },
@@ -204,7 +146,7 @@
     let bytes = pinned(
         &root.join(relative(&descriptor["file"])?),
         descriptor,
-        2 * MIB as usize,
+        4 * MIB as usize,
     )?;
     ensure(
         bytes.len() == words * if bf16 { 2 } else { 4 },
@@ -235,7 +177,7 @@
     after_drop: bool,
 ) -> Proof<Value> {
     ensure(
-        value.layer() == 0 && value.heads() == 16 && value.numerical_profile() == PROFILE,
+        value.layer() == 0 && value.heads() == 48 && value.numerical_profile() == PROFILE,
         "actual same-engine source profile/layer/head binding",
     )?;
     let mut stages = Vec::new();
@@ -290,7 +232,7 @@
     let record = json!({"after_callers_drop":after_drop,"words":words,"different_words":different,"stages":stages});
     write_json(&out.join(format!("comparison-{after_drop}.json")), &record)?;
     ensure(
-        words == 534560 && different == 0,
+        words == 1603680 && different == 0,
         "actual cached signed bits differ; preserve first mismatch",
     )?;
     Ok(record)
@@ -308,7 +250,7 @@
     let host = TransferAdmission {
         arbiter,
         resource: key,
-        owner: ResourceOwner::new("0.8b", "cached-gdn-private-replay")?,
+        owner: ResourceOwner::new("27b", "cached-gdn-private-replay")?,
     };
     let control = host.arbiter.admit(
         &host.resource,
@@ -338,15 +280,23 @@
         QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
     )?;
     ensure(
-        operators.binding().text_tensor_count() == 320
+        operators.binding().text_tensor_count() == 851
             && operators
                 .binding()
                 .config()
                 .geometry()
                 .recurrent_value_heads
-                == Some(16),
-        "genuine complete source 320/H16",
-    )?;
+                == Some(48),
+        "genuine complete source 851/H48",
+    )?;
+    let gamma = source.parameter("model.language_model.layers.0.linear_attn.norm.weight")?;
+    ensure(
+        gamma.tensor().kind == zerogravity_core::quant::QuantKind::Bf16
+            && gamma.tensor().shape == [128]
+            && gamma.tensor().byte_len == 256,
+        "actual original27B BF16 gamma extent, unused by cached recurrence",
+    )?;
+    drop(gamma);
     let before = snapshot(&host)?;
     ensure(
         before.reserved == SOURCE_META + 8 * MIB + CONTROL
@@ -366,7 +316,7 @@
         .try_into()
         .map_err(|_| "six original raw operands")?;
     let make_input =
-        || CachedGdnSliceInput::from_bytes(source.source_identity(), PROFILE, 16, parts);
+        || CachedGdnSliceInput::from_bytes(source.source_identity(), PROFILE, 48, parts);
     ensure(
         operators
             .execute_cached_gdn_slice(0, make_input()?, Some(&AtomicBool::new(true)))
@@ -433,11 +383,11 @@
         "original arbiter final zero",
     )?;
     Ok(
-        json!({"model":"0.8b","heads":16,"imported_words":268320,"output_words":266240,
-        "compared_words":534560,"comparison":comparison,"steady_ledger":ledger(&before),
+        json!({"model":"27b","heads":48,"imported_words":804960,"output_words":798720,
+        "compared_words":1603680,"comparison":comparison,"steady_ledger":ledger(&before),
         "live_ledger":ledger(&live),"retained_ledger":ledger(&retained),"final_ledger":ledger(&final_row),
         "pre_cancel":true,"wrong_layer":true,"real_pressure_refusal":true,
-        "external_decay_no_exp_or_initial_or_StateRoot":true}),
+        "external_decay_no_exp_or_initial_or_StateRoot":true,"original_gamma_bf16_metadata_only":true}),
     )
 }
 fn run() -> Proof<Value> {
@@ -446,8 +396,8 @@
     }
     let args = std::env::args().skip(1).collect::<Vec<_>>();
     ensure(
-        args.len() == 6 && args[5] == "0.8b",
-        "only source-authentic available .8B cached archive",
+        args.len() == 6 && args[5] == "27b",
+        "only source-authentic original27B cached archive",
     )?;
     let input_bytes = read(Path::new(&args[0]), MIB as usize)?;
     let reference_bytes = read(Path::new(&args[2]), MIB as usize)?;
@@ -459,10 +409,10 @@
     let reference: Value = serde_json::from_slice(&reference_bytes)?;
     for document in [&inputs, &reference] {
         ensure(
-            document["model"] == "0.8b"
-                && document["heads"] == 16
+            document["model"] == "27b"
+                && document["heads"] == 48
                 && document["layer"] == 0
-                && document["profile"] == "source-36-f32-restored"
+                && document["profile"] == "original27B-raw-source-bf16-v1"
                 && document["teacher_tokens"] == json!([11, 353])
                 && document["external_decay_no_native_exp"] == true
                 && document["source_bindings_sha256"]
@@ -471,8 +421,8 @@
         )?;
     }
     ensure(
-        inputs["schema"] == "zerogravity.cached-gdn.archived-inputs.v1"
-            && reference["schema"] == "zerogravity.cached-gdn.archived-reference.v1"
+        inputs["schema"] == "zerogravity.cached-gdn.archived-inputs.h48.v1"
+            && reference["schema"] == "zerogravity.cached-gdn.archived-reference.h48.v1"
             && reference["inputs_sha256"] == args[1],
         "original public reference links actual operands",
     )?;
@@ -498,7 +448,7 @@
     Ok(
         json!({"status":"PASS_PRIVATE_ARCHIVED_CACHED_GDN_ONLY","result":result,
         "inputs_sha256":args[1],"reference_sha256":args[3],"actual_affinity":affinity,
-        "full_model_or_live_coefficients_or_StateRoot_or_H48_or_GPU_qualified":false}),
+        "full_model_or_live_coefficients_or_112StateRoots_or_GatedNorm_or_GPU_qualified":false}),
     )
 }
 fn main() {
--- /dev/null
+++ H48/factory.rs
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
+++ H48/source-pin.json
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
--- old-H16/run_replay.py
+++ H48/run_replay.py
@@ -31,7 +31,7 @@
     require(len(cpus) == 4, 'four available CPUs')
     os.sched_setaffinity(0, cpus)
     args.new_output.mkdir(mode=0o700)
-    (args.new_output/'0.8b-lifecycle').mkdir(mode=0o700)
+    (args.new_output/'27b-lifecycle').mkdir(mode=0o700)
     inputs, reference = args.reference_directory/'inputs.json', args.reference_directory/'reference.json'
     binding = read_json(HERE/'source-bindings.json')['source']
     fd = os.open(args.binary.resolve(), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
@@ -39,9 +39,9 @@
         with os.fdopen(os.dup(fd), 'rb') as opened:
             require(hashlib.file_digest(opened, 'sha256').hexdigest() == args.binary_sha256, 'actual exec-FD SHA')
         before = source_files(binding)
-        output = args.new_output/'0.8b'
+        output = args.new_output/'27b'
         execute(args.binary.resolve(), fd, inputs.resolve(), sha(inputs), reference.resolve(), sha(reference),
-                output.resolve(), '0.8b', cpus, freeze, before)
+                output.resolve(), '27b', cpus, freeze, before)
         after = source_files(binding)
         require(before == after, 'all full original source files unchanged')
         require(implementation(args.implementation) == source_attestation and frozen() == freeze and
@@ -50,7 +50,7 @@
         write(output/'source-after.json', {'source_before':before,'source_after':after,'all_equal':True})
         report = read_json(output/'report.json')
         require(report['status'] == 'PASS_PRIVATE_ARCHIVED_CACHED_GDN_ONLY' and
-                report['result']['output_words'] == 266240 and report['result']['compared_words'] == 534560,
+                report['result']['output_words'] == 798720 and report['result']['compared_words'] == 1603680,
                 'complete actual cached results require independent raw validation')
         write(args.new_output/'completed.json', {
             'status':'EXECUTED_REQUIRES_INDEPENDENT_RAW_VALIDATION',
@@ -59,8 +59,8 @@
             'archived_reference_validation':archived,
             'report_sha256':sha(output/'report.json'),
             'source_after_sha256':sha(output/'source-after.json'),
-            'invocation_sha256':sha(args.new_output/'0.8b-lifecycle/invocation.json'),
-            'full_model_or_live_decay_or_StateRoot_or_H48_or_GPU_qualified':False})
+            'invocation_sha256':sha(args.new_output/'27b-lifecycle/invocation.json'),
+            'full_model_or_live_decay_or_112StateRoots_or_GatedNorm_or_GPU_qualified':False})
     finally:
         os.close(fd)
 
--- old-H16/source-bindings.json
+++ H48/source-bindings.json
@@ -1,50 +1,46 @@
 {
+  "capture_invocation": {
+    "bytes": 22448,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/invocation.json",
+    "sha256": "61b1787ca857a1464fd61c8132475dbe7bac2b4282b3b7ff4db7bc15d427512f"
+  },
   "capture_manifest": {
-    "bytes": 110084,
-    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/manifest.json",
-    "sha256": "6ad1051588c2f2239577f512e27597397c0fcd8929f4082ead77ccedfc81743b"
+    "bytes": 233492,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/manifest.json",
+    "sha256": "62e82637c7d24a58b733f2358b3061efd767725901314df4ff7378707d946ab8"
+  },
+  "capture_preparation": {
+    "bytes": 2322,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/prepared-manifest-v3.json",
+    "sha256": "d3614b48bb41033a4052d765d3f7633d456812670783049db75ceadf33d8817d"
   },
   "capture_validation": {
-    "bytes": 1011,
-    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/validation.json",
-    "sha256": "95ee2462a13ac6d92c856a94af1f090f45ffbefbf6b326fe122d8b39ed77dbfe"
-  },
-  "heads": 16,
+    "bytes": 596,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/validation.json",
+    "sha256": "951f74e33de53c4b6b5254bd980df1f6c05f6b9869a46bf617c7fa93184d344d"
+  },
+  "factory_report": {
+    "bytes": 7058,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/source-pin.json",
+    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+  },
+  "heads": 48,
   "layer": 0,
-  "model": "0.8b",
-  "producer_scope": "Actual archived original public cached call; native decay remains external observed exp(g), never recomputed",
-  "profile": "source-36-f32-restored",
-  "schema": "zerogravity.cached-gdn.archived-source-bindings.v1",
+  "model": "27b",
+  "plain_archive": {
+    "bytes": 6518376,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/plain.safetensors",
+    "sha256": "9b20753b733f5f870abf4691eaecf9ce42f5edec0ecec8ae1260684273d5bab8"
+  },
+  "producer_scope": "Authentic original27B layer0 cached public call, teacher353 after true teacher11 padded initialchunk64; decay is external observed exp(g)",
+  "profile": "original27B-raw-source-bf16-v1",
+  "schema": "zerogravity.cached-gdn.archived-source-bindings.h48.v1",
   "source": {
-    "capture_manifest": {
-      "bytes": 310033,
-      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/manifest.json",
-      "sha256": "e007d5e8058a218614a6135d829d35ad5a3b72d4dbdc4916a39498426057cd76"
-    },
-    "heads": 16,
-    "hidden": 1024,
-    "modeling_source": {
-      "path": "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py",
-      "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
-    },
-    "repo": "Qwen/Qwen3.5-0.8B",
-    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
-    "runtime": {
-      "cpu_capability": "AVX2",
-      "cuda_available": false,
-      "default_dtype": "torch.float32",
-      "machine": "x86_64",
-      "mkldnn_enabled": true,
-      "processor": "x86_64",
-      "python": "3.12.3",
-      "safetensors": "0.8.0",
-      "torch": "2.13.0+cu129",
-      "torch_interop_threads": 1,
-      "torch_num_threads": 4,
-      "transformers": "5.17.0",
-      "vllm": "0.31.0+cu129"
-    },
-    "source_directory": "/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17",
+    "heads": 48,
+    "repo": "Qwen/Qwen3.8-27B",
+    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
+    "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
+    "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
     "source_files": [
       {
         "bytes": 11544,
@@ -52,14 +48,29 @@
         "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
       },
       {
-        "bytes": 7755,
+        "bytes": 65012,
+        "path": "README.md",
+        "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
+      },
+      {
+        "bytes": 8952,
         "path": "chat_template.jinja",
-        "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
-      },
-      {
-        "bytes": 2907,
+        "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
+      },
+      {
+        "bytes": 4312,
         "path": "config.json",
-        "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
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
       },
       {
         "bytes": 3353259,
@@ -67,14 +78,99 @@
         "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
       },
       {
-        "bytes": 1746942600,
-        "path": "model.safetensors-00001-of-00001.safetensors",
-        "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
-      },
-      {
-        "bytes": 50900,
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
         "path": "model.safetensors.index.json",
-        "sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e"
+        "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
       },
       {
         "bytes": 390,
@@ -82,14 +178,14 @@
         "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
       },
       {
-        "bytes": 12807982,
+        "bytes": 12809320,
         "path": "tokenizer.json",
-        "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
-      },
-      {
-        "bytes": 16709,
+        "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
+      },
+      {
+        "bytes": 17928,
         "path": "tokenizer_config.json",
-        "sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
+        "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
       },
       {
         "bytes": 385,
@@ -101,72 +197,31 @@
         "path": "vocab.json",
         "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
       }
-    ],
-    "steps": [
-      {
-        "input_token": 11,
-        "metadata": {
-          "bytes": 53537,
-          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-0/manifest.json",
-          "sha256": "d35aaea5180947a4efe02704563cc90a4ae5a0708af60e982740dbdb2e2e2958"
-        },
-        "observed": {
-          "bytes": 1646080,
-          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-0/observed-intermediates.safetensors",
-          "pin_origin": "existing actual capture manifest; author did not hash payload",
-          "sha256": "87d6434fae1a29bac9b308fda8b0054a12bcfa38b01a5a44041fce9443006201"
-        },
-        "raw": {
-          "bytes": 20272592,
-          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-0/raw-official.safetensors",
-          "pin_origin": "existing actual capture manifest; author did not hash payload",
-          "sha256": "d46a4d2115aafc5ff74bc5807f3174900f9f841409418aef6852825c4dabb478"
-        }
-      },
-      {
-        "input_token": 353,
-        "metadata": {
-          "bytes": 54310,
-          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-1/manifest.json",
-          "sha256": "020ac70910386cb3c8e91d1b99d43ad36f4f9b89852c1b5f8a104f74cc861577"
-        },
-        "observed": {
-          "bytes": 1609216,
-          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-1/observed-intermediates.safetensors",
-          "pin_origin": "existing actual capture manifest; author did not hash payload",
-          "sha256": "c730449ea0cc432696f5adabc26efcf9461479a19bd533301f730b7ce060ef27"
-        },
-        "raw": {
-          "bytes": 20284880,
-          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-1/raw-official.safetensors",
-          "pin_origin": "existing actual capture manifest; author did not hash payload",
-          "sha256": "8d4f50dc3c4ebcefed24b178a66aa86543a7dbf652158382accc8fc5b93f781b"
-        }
-      }
     ]
   },
+  "source_plan": {
+    "bytes": 15423,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/source-plan.json",
+    "sha256": "658280444d4e3d1b6a3a8f2ce4b97e4be7cbff2c8a6ea7c66b3c733c78f31638"
+  },
   "step0_archive": {
-    "bytes": 12453400,
-    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-0/actual-gdn-frames.safetensors",
-    "sha256": "a493dd31bf1c511edee9198ac4e8eb30f7141f79207d2ce836dfb8ef02ad0450"
+    "bytes": 46720712,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-0/actual-gdn.safetensors",
+    "sha256": "360ec87dbaaa0647ec72f4959d869e7d04fa1e5eed1fe7ab5b91b22bdc33b29a"
   },
   "step0_metadata": {
-    "bytes": 46710,
-    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-0/manifest.json",
-    "sha256": "ea855e806fa7a0001b943d3dc56ca5f73a35814bda5f15a79c87418572731947"
+    "bytes": 84104,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-0/prevalidation.json",
+    "sha256": "17bd8f5430931cb33a22af32ac056f7357b02df32c9224d1c67a6361be20f46b"
   },
   "step1_archive": {
-    "bytes": 6472400,
-    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-1/actual-gdn-frames.safetensors",
-    "sha256": "c8b7c72d0644c418dfa5c74e974678f7ee76e3d7cc9e83a7c157c3a9f836c0fa"
+    "bytes": 27153160,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-1/actual-gdn.safetensors",
+    "sha256": "907044c0889af975d573cb0bbacc0e8f3ac1212aa60822c36bc770d9868de038"
   },
   "step1_metadata": {
-    "bytes": 41192,
-    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-1/manifest.json",
-    "sha256": "f146eba12662fc9acf7d1cb647244024c2c3747fb9e90d241afeb8a6a9beb8a7"
-  },
-  "tokens": [
-    11,
-    353
-  ]
+    "bytes": 74924,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-1/prevalidation.json",
+    "sha256": "7ceceff0e776b7c4f1c97074de7b065264b69e374397f97bc9f1535f4517dc60"
+  }
 }
--- old-H16/test_contracts.py
+++ H48/test_contracts.py
@@ -6,8 +6,38 @@
 import unittest
 from contracts import archive, digest, finite, read_json, widened
 from validate_replay import affinity
+from export_reference import public_scope
 
 class ByteContracts(unittest.TestCase):
+    def scope(self):
+        capture = dict(status='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION',
+            source_repo='Qwen/Qwen3.8-27B', source_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
+            snapshot='98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e',
+            parameter_profile='original27B-raw-source-bf16-v1', tokens=[11,353], all_source_parameters_BF16=True,
+            full_model_StateRoot_native_exp_GPU_eligible=False, source_plan_sha256='a'*64,
+            runtime=dict(torch='2.13.0+cu129',torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb',
+                transformers='5.17.0',cpu_capability='AVX2',threads=4,interop_threads=1,cuda_available=False,
+                modeling_sha256='762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'),
+            norm_profile=dict(heads=48,head_width=128,raw_gamma_dtype='torch.bfloat16',weighted_dtype='torch.bfloat16'))
+        validation=dict(status='PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY',manifest_sha256='b'*64,
+            source_plan_sha256='a'*64,heads=48,teachers=[11,353],cached_input_words_available=804960,
+            cached_output_words_available=798720,
+            full_model_112_StateRoots_native_exp_allocation_GPU_or_performance_qualified=False)
+        binding=dict(capture_manifest=dict(sha256='b'*64),source_plan=dict(sha256='a'*64))
+        return capture, validation, binding
+
+    def test_genuine_h48_source_scope_and_exact_counts(self):
+        public_scope(*self.scope())
+        for field,value in [('heads',16),('cached_input_words_available',804959),('cached_output_words_available',798719)]:
+            capture, validation, binding=self.scope();validation[field]=value
+            with self.assertRaises(ValueError):public_scope(capture,validation,binding)
+
+    def test_restored_small_model_profile_or_gpu_claim_refuses(self):
+        for field,value in [('parameter_profile','source-36-f32-restored'),('source_repo','Qwen/Qwen3.5-0.8B'),
+                            ('full_model_StateRoot_native_exp_GPU_eligible',True)]:
+            capture,validation,binding=self.scope();capture[field]=value
+            with self.assertRaises(ValueError):public_scope(capture,validation,binding)
+
     def test_signed_zero_subnormal_and_bf16_transport(self):
         raw = bytes.fromhex('008001000000')
         finite(raw, 'BF16')
--- old-H16/validate_replay.py
+++ H48/validate_replay.py
@@ -31,9 +31,9 @@
             done['freeze_manifest_sha256'] == freeze and
             done['ROOT_build_source_attestation_sha256'] == build_source and
             done['archived_reference_validation'] == archived and
-            done['full_model_or_live_decay_or_StateRoot_or_H48_or_GPU_qualified'] is False,
+            done['full_model_or_live_decay_or_112StateRoots_or_GatedNorm_or_GPU_qualified'] is False,
             'exact original public/source/build receipt closure')
-    directory, lifecycle = args.output/'0.8b', args.output/'0.8b-lifecycle'
+    directory, lifecycle = args.output/'27b', args.output/'27b-lifecycle'
     for file, key in [('report.json','report_sha256'),('source-after.json','source_after_sha256')]:
         require(sha(directory/file) == done[key], 'exact native result/source receipt')
     require(sha(lifecycle/'invocation.json') == done['invocation_sha256'], 'exact original owned-child receipt')
@@ -43,7 +43,7 @@
     require(report['status'] == 'PASS_PRIVATE_ARCHIVED_CACHED_GDN_ONLY' and
             report['inputs_sha256'] == archived['inputs_sha256'] and
             report['reference_sha256'] == archived['reference_sha256'] and
-            report['full_model_or_live_coefficients_or_StateRoot_or_H48_or_GPU_qualified'] is False,
+            report['full_model_or_live_coefficients_or_112StateRoots_or_GatedNorm_or_GPU_qualified'] is False,
             'pure archived cached operator only')
     require(invocation['status'] == 'CHILD_EXECUTED_REQUIRES_VALIDATION' and
             invocation['binary_sha256'] == done['binary_sha256'] and invocation['freeze_manifest_sha256'] == freeze and
@@ -55,19 +55,19 @@
     require(source['all_equal'] is True and source['source_before'] == source['source_after'] == invocation['source_before'],
             'actual whole original source unchanged')
     result = report['result']
-    require(result['model'] == '0.8b' and result['heads'] == 16 and result['imported_words'] == 268320 and
-            result['output_words'] == 266240 and result['compared_words'] == 534560 and
-            all(result[k] is True for k in ('pre_cancel','wrong_layer','real_pressure_refusal','external_decay_no_exp_or_initial_or_StateRoot')),
-            'all H16 stages and actual negative receipts')
+    require(result['model'] == '27b' and result['heads'] == 48 and result['imported_words'] == 804960 and
+            result['output_words'] == 798720 and result['compared_words'] == 1603680 and
+            all(result[k] is True for k in ('pre_cancel','wrong_layer','real_pressure_refusal','external_decay_no_exp_or_initial_or_StateRoot','original_gamma_bf16_metadata_only')),
+            'all H48 stages and actual negative receipts')
     mib = 1024*1024
-    input_bytes, output_bytes = 268320*4, 266240*4
-    scratch = 65536 + 16*(6*16384*4+16*128*4)+256*1024
+    input_bytes, output_bytes = 804960*4, 798720*4
+    scratch = 65536 + 48*(6*16384*4+16*128*4)+256*1024
     steady, live, retained, final = [result[k+'_ledger'] for k in ('steady','live','retained','final')]
-    require(steady['reserved'] == 80*mib and steady['reservations'] == 4 and
-            steady['per_class'].get('Scratch',0) == 72*mib and steady['per_class'].get('Other',0) == 8*mib,
+    require(steady['reserved'] == 88*mib and steady['reservations'] == 4 and
+            steady['per_class'].get('Scratch',0) == 72*mib and steady['per_class'].get('Other',0) == 16*mib,
             'source/two original binding/control reservations')
-    require(live['reserved'] == 80*mib+input_bytes+output_bytes+scratch and
-            retained['reserved'] == 72*mib+input_bytes+output_bytes+scratch,
+    require(live['reserved'] == 88*mib+input_bytes+output_bytes+scratch and
+            retained['reserved'] == 80*mib+input_bytes+output_bytes+scratch,
             'aggregate payload/scratch/source owners remain through caller drop')
     for ledger in (live,retained):
         require(ledger['per_class'].get('Output',0) == input_bytes+output_bytes and
@@ -77,7 +77,7 @@
     inputs = read_json(args.reference_directory/'inputs.json')
     reference = read_json(args.reference_directory/'reference.json')
     comparison = result['comparison']
-    require(comparison['after_callers_drop'] is False and comparison['words'] == 534560 and
+    require(comparison['after_callers_drop'] is False and comparison['words'] == 1603680 and
             comparison['different_words'] == 0 and [v['stage'] for v in comparison['stages']] == list(STAGES),
             'complete actual stage ordering/counts')
     total = 0
@@ -94,16 +94,16 @@
         total += record['words']
     for dropped in (False,True):
         record = read_json(directory/('comparison-'+str(dropped).lower()+'.json'))
-        require(record['after_callers_drop'] is dropped and record['words'] == 534560 and
+        require(record['after_callers_drop'] is dropped and record['words'] == 1603680 and
                 record['different_words'] == 0 and record['stages'] == comparison['stages'],
                 'same full archived words remain accessible after source caller drop')
-    require(total == 534560 and {p.name for p in directory.iterdir()} == files and frozen() == freeze,
+    require(total == 1603680 and {p.name for p in directory.iterdir()} == files and frozen() == freeze,
             'all import/output words and native artifact tree accounted')
-    value = {'status':'PASS_ARCHIVED_H16_CACHED_GDN_PRIVATE_SLICE_ONLY',
-             'imported_words':268320,'independent_output_words':266240,'all_compared_words':total,
+    value = {'status':'PASS_ARCHIVED_H48_CACHED_GDN_PRIVATE_SLICE_ONLY',
+             'imported_words':804960,'independent_output_words':798720,'all_compared_words':total,
              'completed_sha256':sha(args.output/'completed.json'),'freeze_manifest_sha256':freeze,
              'ROOT_build_source_attestation_sha256':build_source,
-             'full_model_or_live_decay_or_StateRoot_or_H48_or_GPU_qualified':False}
+             'full_model_or_live_decay_or_112StateRoots_or_GatedNorm_or_GPU_qualified':False}
     write(args.output/'validation.json',value)
     print(json.dumps(value,sort_keys=True))
 

````

