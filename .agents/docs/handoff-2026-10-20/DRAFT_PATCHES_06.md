# Draft Patches — volume 6

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/NEW_DELTA.patch

Original UTF-8 bytes: 217346; SHA256 `4f618f74d676f2c77d98dcdd5ef188a85f301365830972fdd9ebda26246ab2c9`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/Cargo.toml
@@ -0,0 +1,23 @@
+[package]
+name = "original27b-cached-gdn-archive-replay"
+version = "0.0.0"
+edition = "2024"
+publish = false
+
+[workspace]
+
+[dependencies]
+zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts", features = ["cache"] }
+zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core" }
+zerogravity-model-qwen35 = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35", features = ["cpu", "original-hf-binding"] }
+serde_json = "=1.0.151"
+sha2 = "=0.10.9"
+libc = "=0.2.190"
+
+[profile.release]
+opt-level = 3
+codegen-units = 1
+
+[[bin]]
+name = "original27b-cached-gdn-archive-replay"
+path = "main.rs"
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/ISOLATED_ROOT.md
@@ -0,0 +1,36 @@
+# Isolated ROOT replay source relocation — coordinator only
+
+- Additive preparation; original13 files/freeze/frozen copies remain immutable. `ROOT.md` is retained byte-identical provenance. This file supersedes only its MAIN merge/build-path instructions.
+- Target `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`, final ROOT source freeze7 `036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615` (521 Rust/manifests files). MAIN remains unchanged by this preparation.
+- All main.rs/factory/source-pin/source-bindings/reference exporter/raw validator/runner/owned lifetime/old fixtures are exact original bytes. Cargo changes only3 or4 absolute crate dependency roots. contracts.py changes only frozen()/implementation() source attestation dispatch; every finite/geometry/raw-word/count/source/lifetime/numerical check remains exact original.
+- Exact original parent/stride/H48/source binding checks still execute through the authenticated original pure metadata contract. Original13 live/frozen files are revalidated. No old MERGED receipt is accepted.
+- relocation.json conserves every original required path and expected child. Whole521 current source inventory must match the exact final ROOT freeze; each recorded adapter/test exception is an exact unique text delta against immutable parent source. Eight exceptions cover only ROOT corrections2–7. The sole production adapter adds checked coefficient output-size arguments with the same old floors; all other exceptions are tests. A retained attention test baseline is byte-bound to the original ROOT freeze1, without a mutable MAIN dependency.
+- ROOT already executed strict selected CPU/core/model and237 source-free focused tests (0fail/2ignored); exact root-cpu-focused-7.json is pinned. These do not qualify trained native replay, full HF/session/StateRoots/exp/initial GDN/GPU/performance.
+
+## ROOT sequence
+
+1. Review these18 files, original preparation and recorded ROOT exceptions. Run `python3 -m unittest -v test_relocation.py test_contracts.py` **from this directory**, coordinator only. Fourteen new relocation fixture methods are prepared, not executed by the author. They use tiny synthetic source files/metadata, no model/provider/operator/runtime. Whole521 fixture cardinality is a metadata inventory test.
+2. Build this local Cargo manifest with the coordinator's existing controlled/offline source closure. `Cargo.toml` points only at the isolated ROOT crates, with original dependencies/features unchanged. Do not compile against MAIN or label the source MAIN-merged. A fresh local Cargo.lock, if created, is an additional mandatory actual build input. Actual compiler/flags/external-package/binary binding remains the coordinator's build evidence.
+3. After actual ROOT source/build review, create an **isolated source attestation** with the fields below. Hash every final file after build, before execution. This review receipt is not an independent proof of the binary's compiler inputs.
+4. Run the unchanged reference metadata preparation, native replay and independent validator shown in retained ROOT.md, **using the scripts in this directory** and the new isolated implementation receipt. Use fresh exclusive output directories. The original reference source/runtime/archive pins and full raw word/lifetime/negative limits are unchanged; raw payload authentication occurs only in the admitted native worker and independent validator, never source preparation.
+5. Revalidate source freeze/receipt after execution via the unchanged runner/validator. Any source mismatch, missing/excess build path, old MAIN status, unexpected arithmetic delta or lifetime failure refuses acceptance.
+
+Required receipt fields:
+
+```json
+{
+  "status": "ROOT_REVIEWED_ISOLATED_CACHED_GDN_SOURCE",
+  "prepared_manifest_sha256": "SHA256 of this directory's prepared-manifest.json",
+  "original_prepared_manifest_sha256": "4ecc28732a41c800dccb0e3d6a3bafe88049f0c1d09f6c39f3110940453b8118",
+  "root_source_freeze_sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615",
+  "workspace_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
+  "MAIN_merged": false,
+  "full_model_or_StateRoot_or_exp_or_GPU_qualified": false,
+  "files": ["exact path/bytes/sha256 rows described below"]
+}
+```
+
+- `files` = all521 final ROOT freeze entries with workspace-root absolute paths, plus this directory's exact Cargo.toml/main.rs/factory.rs/source-pin.json/source-bindings.json; plus this directory's Cargo.lock if present. Exactly526 or527 unique rows, no MAIN aliases/missing/unlisted source. `relocated_source.workspace` returns the authenticated521 rows; `relocation.json.copy_map[*].relocated` supplies the local5. All expected-child assertions (including any repaired test child) must match relocation.json. The actual optional local lock must be pinned with the same2MiB BEFORE-read/fstat/regular-file policy.
+- Source-only relocation preserves independent input/result owner charges, cancellation/pressure/refusal probes, retained result/source lifetime, final reserved0 and all raw exact-bit limits. No allowance to treat caller attestation as independent runtime or numerical proof.
+
+Acceptance scope: SAME execute_cached_gdn_slice, authentic H48 cached teacher353 following original initial11 padded64 state;804,960 input +798,720 output =1,603,680 exact raw words. Observed external decay remains external. No initial/exp/fullGDN/gatednorm/canonical StateRoot/session/GPU eligibility. Original33 required paths and12 expected children conserved.
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/ROOT.md
@@ -0,0 +1,55 @@
+# ROOT-only authentic H48 cached typed replay
+
+Author: spine_review. Original .8B931d/57a/stride proposal and captures unchanged.
+No author build/tests/Torch/provider/operator/model/GPU execution or payload hash.
+
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
+
+ROOT sequence after review/merge and timing holds:
+
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
+
+Both original .8B/H16 and new genuine27B/H48 gates remain required independently.
+No new receipt/public-hash declaration establishes native support before ROOT's
+actual compile, independent trained comparison and lifecycle/ledger qualification.
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/baseline-attention-slice-tests.rs
@@ -0,0 +1,497 @@
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
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/contracts.py
@@ -0,0 +1,114 @@
+"""ROOT-only byte transport; no operator/model/provider calls on import."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import struct
+
+HERE = Path(__file__).resolve().parent
+INPUTS = ('query', 'key', 'value', 'beta', 'decay', 'previous_state')
+OUTPUTS = ('core_f32', 'output_bf16', 'next_state')
+STAGES = INPUTS + OUTPUTS
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
+    value = {}
+    for key, item in pairs:
+        require(key not in value, 'duplicate JSON field')
+        value[key] = item
+    return value
+
+def identity(meta):
+    return (meta.st_dev, meta.st_ino, meta.st_size, meta.st_mtime_ns, meta.st_ctime_ns)
+
+def read(path, cap=2*1024*1024, row=None):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as opened:
+        before = os.fstat(opened.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular input')
+        raw = opened.read(cap+1)
+        after = os.fstat(opened.fileno())
+    require(len(raw) <= cap and identity(before) == identity(after), 'unchanged bounded input')
+    if row is not None:
+        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'exact input bytes/SHA')
+    return raw
+
+def pin(path, row, cap=2*1024*1024):
+    read(path, cap, row)
+
+def read_json(path):
+    return json.loads(read(path), object_pairs_hook=unique)
+
+def write(path, value):
+    with Path(path).open('x') as opened:
+        json.dump(value, opened, indent=2, sort_keys=True)
+        opened.write('\n')
+
+def frozen():
+    from relocated_source import frozen as checked_frozen
+    return checked_frozen(HERE, read_json, pin, require, sha)
+
+def implementation(path):
+    from relocated_source import implementation as checked_implementation
+    return checked_implementation(HERE, path, read_json, pin, require, sha)
+
+
+def finite(raw, dtype):
+    require(dtype in ('BF16', 'F32'), 'closed source dtype')
+    size, fmt, mask = (2, '<H', 0x7f80) if dtype == 'BF16' else (4, '<I', 0x7f800000)
+    require(raw and len(raw) % size == 0, 'complete raw finite words')
+    require(all(bits & mask != mask for bits, in struct.iter_unpack(fmt, raw)), 'finite raw input/output')
+
+def archive(row):
+    raw = read(row['path'], 64*1024*1024, row)
+    n, = struct.unpack_from('<Q', raw)
+    require(1 < n <= 1024*1024 and 8+n < len(raw), 'bounded original archive header')
+    header = json.loads(raw[8:8+n], object_pairs_hook=unique)
+    tensors, spans = {}, []
+    for name, item in header.items():
+        if name == '__metadata__':
+            continue
+        dtype, shape = item['dtype'], item['shape']
+        require(dtype in ('BF16', 'F32'), 'original cached archive dtypes')
+        count = 1
+        require(type(shape) is list and len(shape) <= 8, 'bounded source rank')
+        for axis in shape:
+            require(type(axis) is int and 0 < axis <= 262144, 'bounded source axis')
+            count *= axis
+        lo, hi = item['data_offsets']
+        require(type(lo) is type(hi) is int and 0 <= lo < hi and
+                hi-lo == count*(2 if dtype == 'BF16' else 4) and 8+n+hi <= len(raw),
+                'complete typed original archive extent')
+        payload = raw[8+n+lo:8+n+hi]
+        # Unselected authentic chunk masks may contain -Inf; selected stages are finite-checked.
+        tensors[name] = {'dtype': dtype, 'shape': shape, 'raw': payload}
+        spans.append((lo, hi))
+    end = 0
+    for lo, hi in sorted(spans):
+        require(lo == end, 'complete nonoverlapping original archive')
+        end = hi
+    require(8+n+end == len(raw), 'no trailing original archive bytes')
+    return tensors
+
+def raw_file(root, row):
+    require(Path(row['file']).name == row['file'], 'contained operand filename')
+    raw = read(root/row['file'], 4*1024*1024, row)
+    finite(raw, row['dtype'])
+    return raw
+
+def widened(raw, dtype):
+    if dtype == 'BF16':
+        return b''.join((word << 16).to_bytes(4, 'little') for word, in struct.iter_unpack('<H', raw))
+    require(dtype == 'F32', 'source F32 transport')
+    return raw
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/export_reference.py
@@ -0,0 +1,157 @@
+"""ROOT ONLY genuine H48 archive transport; no operator or expected-value math."""
+import argparse
+import json
+from pathlib import Path
+
+from contracts import (HERE, INPUTS, OUTPUTS, PROFILE, STAGES, archive, digest,
+                       finite, frozen, pin, raw_file, read_json, require, sha, write)
+
+NAMES = {
+    'query': ('recurrent.input.query', 'BF16', [1, 1, 48, 128]),
+    'key': ('recurrent.input.key', 'BF16', [1, 1, 48, 128]),
+    'value': ('recurrent.input.value', 'BF16', [1, 1, 48, 128]),
+    'beta': ('recurrent.input.beta', 'BF16', [1, 1, 48]),
+    'decay': ('recurrent.decay_and_prior.decay_t', 'F32', [1, 48, 1, 1]),
+    'previous_state': ('recurrent.input.initial_state', 'F32', [1, 48, 128, 128]),
+    'core_f32': ('recurrent.fp32_core.core_attn_out', 'F32', [1, 48, 1, 128]),
+    'output_bf16': ('recurrent.return.actual_return_0', 'BF16', [1, 1, 48, 128]),
+    'next_state': ('recurrent.return.actual_return_1', 'F32', [1, 48, 128, 128]),
+}
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
+
+def authenticated():
+    binding = read_json(HERE/'source-bindings.json')
+    for key in ('capture_manifest', 'capture_validation', 'capture_invocation', 'source_plan',
+                'capture_preparation', 'factory_report', 'step0_metadata', 'step1_metadata'):
+        pin(binding[key]['path'], binding[key])
+    capture = read_json(binding['capture_manifest']['path'])
+    validation = read_json(binding['capture_validation']['path'])
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
+    prior = before['chunk.return.actual_return_1']
+    require(prior['dtype'] == 'F32' and prior['shape'] == [1, 48, 128, 128] and
+            prior['raw'] == actual['recurrent.input.initial_state']['raw'] ==
+            actual['recurrent.decay_and_prior.last_recurrent_state']['raw'] == plain['step-0.state']['raw'] and
+            actual['recurrent.return.actual_return_1']['raw'] == plain['step-1.state']['raw'],
+            'all786432 actual source state words link initial→cached and unchanged plain cache')
+    selected = {}
+    for name, (key, dtype, shape) in NAMES.items():
+        tensor, description = actual[key], steps[1]['tensors'][key]
+        require(tensor['dtype'] == dtype and tensor['shape'] == description['shape'] == shape and
+                description['contiguous'] is True and description['device'] == 'cpu' and
+                description['dtype'] == ('torch.bfloat16' if dtype == 'BF16' else 'torch.float32') and
+                description['raw_sha256'] == digest(tensor['raw']) and
+                description['bytes'] == len(tensor['raw']), 'actual full typed H48 stage: '+name)
+        finite(tensor['raw'], dtype)
+        selected[name] = (tensor, description)
+    return binding, selected
+
+
+def validate(directory):
+    binding, selected = authenticated()
+    inputs, reference = read_json(directory/'inputs.json'), read_json(directory/'reference.json')
+    require(inputs['schema'] == 'zerogravity.cached-gdn.archived-inputs.h48.v1' and
+            reference['schema'] == 'zerogravity.cached-gdn.archived-reference.h48.v1' and
+            inputs['source_bindings_sha256'] == reference['source_bindings_sha256'] == sha(HERE/'source-bindings.json') and
+            reference['inputs_sha256'] == sha(directory/'inputs.json') and
+            inputs['model'] == reference['model'] == '27b' and
+            inputs['heads'] == reference['heads'] == 48 and inputs['layer'] == reference['layer'] == 0 and
+            inputs['profile'] == reference['profile'] == PROFILE and reference['expected_output_words'] == 798720 and
+            reference['external_decay_no_native_exp'] is True, 'exact genuine H48 artifact namespace/profile')
+    require(set(inputs['operands']) == set(INPUTS) and set(reference['targets']) == set(OUTPUTS),
+            'all six inputs and three results')
+    for name in STAGES:
+        row = inputs['operands'][name] if name in INPUTS else reference['targets'][name]
+        tensor, description = selected[name]
+        require(row['original_tensor'] == NAMES[name][0] and row['dtype'] == tensor['dtype'] and
+                row['shape'] == tensor['shape'] and row['original_stride'] == description['stride'] and
+                raw_file(directory, row) == tensor['raw'], 'all signed source words: '+name)
+    require({p.name for p in directory.iterdir()} == {'inputs.json', 'reference.json', *(name+'.raw' for name in STAGES)},
+            'complete extracted artifact tree')
+    return dict(status='PASS_ARCHIVED_PUBLIC_H48_OPERANDS_AND_TARGETS_ONLY',
+        inputs_sha256=sha(directory/'inputs.json'), reference_sha256=sha(directory/'reference.json'),
+        source_bindings_sha256=sha(HERE/'source-bindings.json'), expected_output_words=798720,
+        full_model_or_live_decay_or_112StateRoots_or_GatedNorm_or_GPU_qualified=False)
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--new-output', type=Path, required=True)
+    args = parser.parse_args()
+    frozen()
+    _, selected = authenticated()
+    require(args.new_output.resolve().parent == HERE, 'fresh own export only')
+    args.new_output.mkdir(mode=0o700)
+    rows = {}
+    for name, (tensor, description) in selected.items():
+        raw = tensor['raw']
+        with (args.new_output/(name+'.raw')).open('xb') as opened:
+            opened.write(raw)
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
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/factory.rs
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
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/main.rs
@@ -0,0 +1,471 @@
+//! ROOT ONLY: actual archived cached operands/outputs through shared HostMath.
+mod factory;
+use serde_json::{Value, json};
+use sha2::{Digest, Sha256};
+use std::collections::BTreeMap;
+use std::error::Error;
+use std::fs::OpenOptions;
+use std::io::{Read, Write};
+use std::os::unix::fs::OpenOptionsExt;
+use std::path::{Component, Path};
+use std::sync::atomic::AtomicBool;
+use zerogravity_artifacts::cache::TransferAdmission;
+use zerogravity_artifacts::source_weights::VerifiedTensorSource;
+use zerogravity_core::accelerator::program::ProgramNumericalProfile;
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, DeviceBudgetSnapshot, PhysicalDeviceKey, ResourceArbiter,
+    ResourceOwner,
+};
+use zerogravity_model_qwen35::operator_slice::cached_gdn::{
+    CachedGdnSliceInput, CachedGdnSliceOutput,
+};
+use zerogravity_model_qwen35::original_binding::{OriginalWindowOperators, QwenNumericalProfile};
+type Proof<T> = Result<T, Box<dyn Error>>;
+const MIB: u64 = 1024 * 1024;
+const HOST: u64 = 128 * MIB;
+const SOURCE_META: u64 = 64 * MIB;
+const CONTROL: u64 = 16 * MIB;
+const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
+const STAGES: [&str; 9] = [
+    "query",
+    "key",
+    "value",
+    "beta",
+    "decay",
+    "previous_state",
+    "core_f32",
+    "output_bf16",
+    "next_state",
+];
+const INPUT_BYTES: u64 = 48 * (16384 + 3 * 128 + 2) * 4;
+const OUTPUT_BYTES: u64 = 48 * (16384 + 2 * 128) * 4;
+const INPUT_SCRATCH: u64 = 64 * 1024;
+const CORE_SCRATCH: u64 = 48 * (6 * 16384 * 4 + 16 * 128 * 4) + 256 * 1024;
+fn ensure(ok: bool, message: impl Into<String>) -> Proof<()> {
+    if ok {
+        Ok(())
+    } else {
+        Err(message.into().into())
+    }
+}
+fn text(value: &Value) -> Proof<&str> {
+    value
+        .as_str()
+        .ok_or_else(|| "required string missing".into())
+}
+fn sha(bytes: &[u8]) -> String {
+    format!("{:x}", Sha256::digest(bytes))
+}
+fn read(path: &Path, cap: usize) -> Proof<Vec<u8>> {
+    let file = OpenOptions::new()
+        .read(true)
+        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+        .open(path)?;
+    ensure(
+        file.metadata()?.is_file() && file.metadata()?.len() <= cap as u64,
+        "bounded regular input",
+    )?;
+    let mut raw = Vec::new();
+    file.take(cap as u64 + 1).read_to_end(&mut raw)?;
+    ensure(raw.len() <= cap, "bounded input grew")?;
+    Ok(raw)
+}
+fn pinned(path: &Path, pin: &Value, cap: usize) -> Proof<Vec<u8>> {
+    let raw = read(path, cap)?;
+    ensure(
+        pin["bytes"].as_u64() == Some(raw.len() as u64) && pin["sha256"] == sha(&raw),
+        "exact input file digest/length",
+    )?;
+    Ok(raw)
+}
+fn relative(value: &Value) -> Proof<&Path> {
+    let path = Path::new(text(value)?);
+    ensure(
+        !path.as_os_str().is_empty()
+            && path
+                .components()
+                .all(|part| matches!(part, Component::Normal(_))),
+        "contained input name",
+    )?;
+    Ok(path)
+}
+fn write_json(path: &Path, value: &Value) -> Proof<()> {
+    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
+    serde_json::to_writer_pretty(&mut output, value)?;
+    output.write_all(b"\n")?;
+    Ok(())
+}
+fn snapshot(host: &TransferAdmission) -> Proof<DeviceBudgetSnapshot> {
+    Ok(host.arbiter.snapshot(&host.resource)?)
+}
+fn class(row: &DeviceBudgetSnapshot, kind: AllocationClass) -> u64 {
+    row.per_class.get(&kind).copied().unwrap_or(0)
+}
+fn ledger(row: &DeviceBudgetSnapshot) -> Value {
+    json!({"reserved":row.reserved,"remaining":row.remaining,"reservations":row.reservations,
+        "fenced":row.fenced,"per_class":row.per_class.iter()
+            .map(|(k,v)|(format!("{k:?}"),*v)).collect::<BTreeMap<_,_>>()})
+}
+fn same_ledger(host: &TransferAdmission, before: &DeviceBudgetSnapshot) -> Proof<()> {
+    let after = snapshot(host)?;
+    ensure(
+        after.reserved == before.reserved
+            && after.per_class == before.per_class
+            && after.reservations == before.reservations
+            && !after.fenced,
+        "all private failed-call buffers released",
+    )
+}
+fn source(_binding: &Value, host: &TransferAdmission) -> Proof<(VerifiedTensorSource, Vec<u8>)> {
+    let ready = factory::open_source(host)?;
+    ensure(
+        ready.source.metadata_reserved_bytes() == SOURCE_META,
+        "same accepted original27B source metadata reservation",
+    )?;
+    Ok((ready.source, ready.config))
+}
+
+fn row<'a>(inputs: &'a Value, reference: &'a Value, name: &str) -> &'a Value {
+    if STAGES[..6].contains(&name) {
+        &inputs["operands"][name]
+    } else {
+        &reference["targets"][name]
+    }
+}
+fn raw(root: &Path, descriptor: &Value, name: &str) -> Proof<Vec<u8>> {
+    let bf16 = matches!(name, "query" | "key" | "value" | "beta" | "output_bf16");
+    let words = match name {
+        "beta" | "decay" => 48,
+        "previous_state" | "next_state" => 786432,
+        _ => 6144,
+    };
+    ensure(
+        descriptor["dtype"] == if bf16 { "BF16" } else { "F32" },
+        "exact source stage dtype",
+    )?;
+    let bytes = pinned(
+        &root.join(relative(&descriptor["file"])?),
+        descriptor,
+        4 * MIB as usize,
+    )?;
+    ensure(
+        bytes.len() == words * if bf16 { 2 } else { 4 },
+        "complete all-head raw stage",
+    )?;
+    Ok(bytes)
+}
+fn views(value: &CachedGdnSliceOutput) -> Proof<[&[f32]; 9]> {
+    let inputs = value.original_inputs()?;
+    Ok([
+        inputs[0],
+        inputs[1],
+        inputs[2],
+        inputs[3],
+        inputs[4],
+        inputs[5],
+        value.core_f32()?,
+        value.output_bf16()?,
+        value.next_state_key_major()?,
+    ])
+}
+fn check(
+    value: &CachedGdnSliceOutput,
+    inputs: &Value,
+    reference: &Value,
+    reference_root: &Path,
+    out: &Path,
+    after_drop: bool,
+) -> Proof<Value> {
+    ensure(
+        value.layer() == 0 && value.heads() == 48 && value.numerical_profile() == PROFILE,
+        "actual same-engine source profile/layer/head binding",
+    )?;
+    let mut stages = Vec::new();
+    let mut different = 0;
+    let mut words = 0;
+    for (name, actual) in STAGES.into_iter().zip(views(value)?) {
+        let descriptor = row(inputs, reference, name);
+        let expected = raw(reference_root, descriptor, name)?;
+        let bf16 = descriptor["dtype"] == "BF16";
+        let width = if bf16 { 2 } else { 4 };
+        ensure(
+            actual.len() * width == expected.len(),
+            "all actual words before zip",
+        )?;
+        let mut delta = 0;
+        let mut first = Value::Null;
+        for (index, (actual, bytes)) in actual.iter().zip(expected.chunks_exact(width)).enumerate()
+        {
+            let bits = if bf16 {
+                u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16
+            } else {
+                u32::from_le_bytes(bytes.try_into()?)
+            };
+            if actual.to_bits() != bits {
+                delta += 1;
+                if first.is_null() {
+                    first = json!({"word":index,"actual":format!("{:08x}",actual.to_bits()),"expected":format!("{bits:08x}")});
+                }
+            }
+            ensure(
+                actual.is_finite() && (!bf16 || actual.to_bits() & 0xffff == 0),
+                "finite exact original output precision",
+            )?;
+        }
+        let bytes = actual
+            .iter()
+            .flat_map(|v| v.to_bits().to_le_bytes())
+            .collect::<Vec<_>>();
+        let file = format!("{name}.f32le");
+        if !after_drop {
+            OpenOptions::new()
+                .write(true)
+                .create_new(true)
+                .open(out.join(&file))?
+                .write_all(&bytes)?;
+        }
+        stages.push(json!({"stage":name,"words":actual.len(),"different_words":delta,
+            "first_difference":first,"file":file,"bytes":bytes.len(),"sha256":sha(&bytes),"reference":descriptor}));
+        words += actual.len();
+        different += delta;
+    }
+    let record = json!({"after_callers_drop":after_drop,"words":words,"different_words":different,"stages":stages});
+    write_json(&out.join(format!("comparison-{after_drop}.json")), &record)?;
+    ensure(
+        words == 1603680 && different == 0,
+        "actual cached signed bits differ; preserve first mismatch",
+    )?;
+    Ok(record)
+}
+fn exercise(
+    binding: &Value,
+    inputs: &Value,
+    reference: &Value,
+    reference_root: &Path,
+    out: &Path,
+) -> Proof<Value> {
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "cached-gdn-archive-replay")?;
+    arbiter.register(key.clone(), HOST)?;
+    let host = TransferAdmission {
+        arbiter,
+        resource: key,
+        owner: ResourceOwner::new("27b", "cached-gdn-private-replay")?,
+    };
+    let control = host.arbiter.admit(
+        &host.resource,
+        host.owner.clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Other,
+            bytes: CONTROL,
+        }],
+    )?;
+    let (source, config) = source(binding, &host)?;
+    let files = binding["source_files"]
+        .as_array()
+        .ok_or("source file pins")?;
+    let source_hash = |name: &str| -> Proof<&str> {
+        text(
+            &files
+                .iter()
+                .find(|v| v["path"] == name)
+                .ok_or("source pin")?["sha256"],
+        )
+    };
+    let operators = OriginalWindowOperators::bind(
+        source.clone(),
+        &config,
+        source_hash("config.json")?,
+        source_hash("model.safetensors.index.json")?,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )?;
+    ensure(
+        operators.binding().text_tensor_count() == 851
+            && operators
+                .binding()
+                .config()
+                .geometry()
+                .recurrent_value_heads
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
+    let before = snapshot(&host)?;
+    ensure(
+        before.reserved == SOURCE_META + 8 * MIB + CONTROL
+            && before.reservations == 4
+            && class(&before, AllocationClass::Scratch) == SOURCE_META + 8 * MIB
+            && class(&before, AllocationClass::Other) == CONTROL,
+        "source/two-binding/control steady admission",
+    )?;
+    let originals = STAGES[..6]
+        .iter()
+        .map(|name| raw(reference_root, row(inputs, reference, name), name))
+        .collect::<Proof<Vec<_>>>()?;
+    let parts: [&[u8]; 6] = originals
+        .iter()
+        .map(Vec::as_slice)
+        .collect::<Vec<_>>()
+        .try_into()
+        .map_err(|_| "six original raw operands")?;
+    let make_input =
+        || CachedGdnSliceInput::from_bytes(source.source_identity(), PROFILE, 48, parts);
+    ensure(
+        operators
+            .execute_cached_gdn_slice(0, make_input()?, Some(&AtomicBool::new(true)))
+            .is_err(),
+        "pre-cancel refuses",
+    )?;
+    same_ledger(&host, &before)?;
+    ensure(
+        operators
+            .execute_cached_gdn_slice(3, make_input()?, None)
+            .is_err(),
+        "wrong full-attention layer refuses",
+    )?;
+    same_ledger(&host, &before)?;
+    let mut pressure = Vec::new();
+    while snapshot(&host)?.remaining > 0 {
+        pressure.push(source.reserve_workspace(snapshot(&host)?.remaining.min(32 * MIB) as usize)?);
+    }
+    let pressured = snapshot(&host)?;
+    ensure(
+        pressured.reserved == HOST
+            && operators
+                .execute_cached_gdn_slice(0, make_input()?, None)
+                .is_err(),
+        "real original pool refuses before private operand allocation",
+    )?;
+    same_ledger(&host, &pressured)?;
+    drop(pressure);
+    same_ledger(&host, &before)?;
+    let result = operators.execute_cached_gdn_slice(0, make_input()?, None)?;
+    ensure(
+        result.source_identity() == source.source_identity(),
+        "original issuer unchanged",
+    )?;
+    let comparison = check(&result, inputs, reference, reference_root, out, false)?;
+    let live = snapshot(&host)?;
+    ensure(
+        live.reserved
+            == before.reserved + INPUT_BYTES + OUTPUT_BYTES + INPUT_SCRATCH + CORE_SCRATCH
+            && class(&live, AllocationClass::Output) == INPUT_BYTES + OUTPUT_BYTES
+            && class(&live, AllocationClass::TransferStaging) == 0,
+        "all six retained inputs/three results and conservative scratch charged",
+    )?;
+    drop(operators);
+    drop(source);
+    drop(config);
+    let retained = snapshot(&host)?;
+    ensure(
+        retained.reserved
+            == SOURCE_META + CONTROL + INPUT_BYTES + OUTPUT_BYTES + INPUT_SCRATCH + CORE_SCRATCH
+            && class(&retained, AllocationClass::Output) == INPUT_BYTES + OUTPUT_BYTES,
+        "same sealed source/all inputs/results survive caller drop",
+    )?;
+    check(&result, inputs, reference, reference_root, out, true)?;
+    drop(result);
+    ensure(
+        snapshot(&host)?.reserved == CONTROL,
+        "all source/data reservations retire at output last drop",
+    )?;
+    drop(control);
+    let final_row = snapshot(&host)?;
+    ensure(
+        final_row.reserved == 0 && final_row.reservations == 0 && !final_row.fenced,
+        "original arbiter final zero",
+    )?;
+    Ok(
+        json!({"model":"27b","heads":48,"imported_words":804960,"output_words":798720,
+        "compared_words":1603680,"comparison":comparison,"steady_ledger":ledger(&before),
+        "live_ledger":ledger(&live),"retained_ledger":ledger(&retained),"final_ledger":ledger(&final_row),
+        "pre_cancel":true,"wrong_layer":true,"real_pressure_refusal":true,
+        "external_decay_no_exp_or_initial_or_StateRoot":true,"original_gamma_bf16_metadata_only":true}),
+    )
+}
+fn run() -> Proof<Value> {
+    if cfg!(debug_assertions) {
+        return Err("release-only cached operator proof".into());
+    }
+    let args = std::env::args().skip(1).collect::<Vec<_>>();
+    ensure(
+        args.len() == 6 && args[5] == "27b",
+        "only source-authentic original27B cached archive",
+    )?;
+    let input_bytes = read(Path::new(&args[0]), MIB as usize)?;
+    let reference_bytes = read(Path::new(&args[2]), MIB as usize)?;
+    ensure(
+        sha(&input_bytes) == args[1] && sha(&reference_bytes) == args[3],
+        "actual input/reference metadata SHA",
+    )?;
+    let inputs: Value = serde_json::from_slice(&input_bytes)?;
+    let reference: Value = serde_json::from_slice(&reference_bytes)?;
+    for document in [&inputs, &reference] {
+        ensure(
+            document["model"] == "27b"
+                && document["heads"] == 48
+                && document["layer"] == 0
+                && document["profile"] == "original27B-raw-source-bf16-v1"
+                && document["teacher_tokens"] == json!([11, 353])
+                && document["external_decay_no_native_exp"] == true
+                && document["source_bindings_sha256"]
+                    == sha(include_bytes!("source-bindings.json")),
+            "authentic cached-only source/profile/phase",
+        )?;
+    }
+    ensure(
+        inputs["schema"] == "zerogravity.cached-gdn.archived-inputs.h48.v1"
+            && reference["schema"] == "zerogravity.cached-gdn.archived-reference.h48.v1"
+            && reference["inputs_sha256"] == args[1],
+        "original public reference links actual operands",
+    )?;
+    ensure(
+        Path::new(&args[0]).parent() == Path::new(&args[2]).parent(),
+        "same immutable raw reference directory",
+    )?;
+    let out = Path::new(&args[4]);
+    std::fs::create_dir(out)?;
+    let bindings: Value = serde_json::from_str(include_str!("source-bindings.json"))?;
+    let result = exercise(
+        &bindings["source"],
+        &inputs,
+        &reference,
+        Path::new(&args[2]).parent().ok_or("reference root")?,
+        out,
+    )?;
+    let affinity = std::fs::read_to_string("/proc/self/status")?
+        .lines()
+        .find(|v| v.starts_with("Cpus_allowed_list:"))
+        .ok_or("actual CPU affinity")?
+        .to_owned();
+    Ok(
+        json!({"status":"PASS_PRIVATE_ARCHIVED_CACHED_GDN_ONLY","result":result,
+        "inputs_sha256":args[1],"reference_sha256":args[3],"actual_affinity":affinity,
+        "full_model_or_live_coefficients_or_112StateRoots_or_GatedNorm_or_GPU_qualified":false}),
+    )
+}
+fn main() {
+    let result = run();
+    let report = match &result {
+        Ok(value) => value.clone(),
+        Err(error) => json!({"status":"FAIL_UNQUALIFIED","error":error.to_string()}),
+    };
+    let written = std::env::args()
+        .nth(5)
+        .ok_or("output path missing")
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
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/owned_lifetime.py
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
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/owned_native.py
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
+              "binary_sha256": sha(binary), "cpu_affinity": cpus, "scope": "private archived original cached GDN only",
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
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/relocated_source.py
@@ -0,0 +1,242 @@
+"""Source-only relocation contract. No model/provider/math/process operations."""
+import hashlib
+import importlib.util
+import json
+import os
+from pathlib import Path
+import stat
+
+CONTRACT_REPLACEMENT = '''def frozen():
+    from relocated_source import frozen as checked_frozen
+    return checked_frozen(HERE, read_json, pin, require, sha)
+
+def implementation(path):
+    from relocated_source import implementation as checked_implementation
+    return checked_implementation(HERE, path, read_json, pin, require, sha)
+
+'''
+
+def _raw(path, row, require):
+    path = Path(path)
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as opened:
+        before = os.fstat(opened.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 2*1024**2,
+                'bounded regular source/metadata only')
+        raw = opened.read(2*1024**2+1)
+        after = os.fstat(opened.fileno())
+    identity = lambda v: (v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
+    require(identity(before) == identity(after) and len(raw) == before.st_size == row['bytes'] and
+            hashlib.sha256(raw).hexdigest() == row['sha256'], 'exact same-FD immutable source pin')
+    return raw
+
+def remap(path, config, require):
+    path = Path(path)
+    old = Path(config['original_harness'])
+    root = Path(config['main_root'])
+    require(path.is_absolute() and path == Path(os.path.normpath(path)), 'canonical absolute build input')
+    if path.is_relative_to(old):
+        relative = path.relative_to(old)
+        require(len(relative.parts) == 1 and relative.name in
+                ('Cargo.toml','main.rs','factory.rs','source-pin.json','source-bindings.json'),
+                'only exact original local build inputs may move')
+        return Path(config['relocated_harness'])/relative
+    require(path.is_relative_to(root), 'original build input belongs to original source tree')
+    relative = path.relative_to(root)
+    require(relative.parts[0] == 'crates' or str(relative) in ('Cargo.toml','Cargo.lock'),
+            'only MAIN crate/manifests remap; source artifacts remain original')
+    return Path(config['workspace_root'])/relative
+
+def normalized(before, replacements, require):
+    value = before.decode()
+    for edit in replacements:
+        require(edit['before'] and type(edit['expected_occurrences']) is int and
+                edit['expected_occurrences'] == 1 and value.count(edit['before']) == 1,
+                'exact unique ROOT-recorded source-only adapter/test replacement')
+        value = value.replace(edit['before'],edit['after'],1)
+    return value.encode()
+
+def _copies(here, config, require):
+    old = Path(config['original_harness'])
+    require(Path(config['relocated_harness']) == here and here.resolve() == here and
+            old.resolve() == old and old != here, 'explicit original/isolated paths, no runtime discovery')
+    for item in config['copy_map']:
+        require(item['file'] == Path(item['file']).name and
+                item['original']['path'] == str(old/item['file']) and
+                item['relocated']['path'] == str(here/item['file']), 'exact copied filename/path binding')
+        before = _raw(item['original']['path'],item['original'],require)
+        after = _raw(item['relocated']['path'],item['relocated'],require)
+        policy = item['policy']
+        if policy == 'BYTE_IDENTICAL':
+            require(after == before, 'original harness/runner/limits/expected math/lifetime unchanged')
+        elif policy == 'ONLY_LOCAL_DEPENDENCY_ROOT_REMAP':
+            require(item['file'] == 'Cargo.toml', 'only Cargo dependency roots differ')
+            a = (config['main_root']+'/crates/').encode()
+            b = (config['workspace_root']+'/crates/').encode()
+            require(before.count(a) in (3,4) and after == before.replace(a,b),
+                    'exact Cargo dependency root substitution only')
+        elif policy == 'ONLY_SOURCE_ATTESTATION_CONTRACT_RELOCATION':
+            require(item['file'] == 'contracts.py', 'only source contract functions differ')
+            source = before.decode()
+            start, end = source.index('def frozen():'), source.index('\ndef finite(')
+            require(after == (source[:start]+CONTRACT_REPLACEMENT+source[end:]).encode(),
+                    'all raw math/word transport/limits unchanged around source-only contract')
+        else:
+            require(False, 'unknown copy policy refused')
+
+def _original(config, read_json, pin, require):
+    original = config['original_preparation']
+    pin(original['path'],original)
+    prior = read_json(original['path'])
+    old = Path(config['original_harness'])
+    require(Path(original['path']) == old/'prepared-manifest.json', 'exact original preparation')
+    for row in prior['files']:
+        require(row['file'] == Path(row['file']).name, 'original finite preparation filenames')
+        pin(old/row['file'],row)
+        pin(old/'frozen-prepared'/row['file'],row)
+    contract = config['original_contract']
+    require(contract['path'] == str(old/'contracts.py'), 'only original frozen pure metadata contract')
+    pin(contract['path'],contract)
+    # Exact old source is authenticated BEFORE loading. Its frozen() only checks
+    # original source/metadata, never implementation(), operators or archives.
+    spec = importlib.util.spec_from_file_location('_zg_authenticated_original_contract',contract['path'])
+    require(spec is not None and spec.loader is not None, 'original pure contract loader')
+    module = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(module)
+    require(module.frozen() == original['sha256'], 'all original parent/stride/H48/math source proof preserved')
+    return prior
+
+def workspace(config, read_json, require):
+    root = Path(config['workspace_root'])
+    require(root.is_absolute() and root.resolve() == root and root != Path(config['main_root']),
+            'explicit distinct canonical isolated ROOT source tree')
+    pin = config['source_freeze']
+    raw = _raw(pin['path'],pin,require)
+    document = json.loads(raw)
+    require(document['full_model_or_GPU_qualified'] is False and
+            type(config['source_count']) is int and len(document['files']) == config['source_count'] == 521,
+            'exact ROOT-reviewed521 Rust/manifests closure; not runtime/full-model proof')
+    by_path = {}
+    for row in document['files']:
+        relative = Path(row['file'])
+        require(not relative.is_absolute() and relative.parts and
+                all(v not in ('.','..') for v in relative.parts) and
+                (relative.suffix == '.rs' or relative.name in ('Cargo.toml','Cargo.lock')),
+                'finite source-only relative manifest entry')
+        path = root/relative
+        require(path.resolve() == path and str(path) not in by_path, 'no source symlink/duplicate')
+        _raw(path,row,require)
+        by_path[str(path)] = dict(path=str(path),bytes=row['bytes'],sha256=row['sha256'])
+    actual = {str(root/'Cargo.toml'),str(root/'Cargo.lock')}
+    actual.update(str(p) for p in (root/'crates').rglob('*') if p.is_file() and
+                  (p.suffix == '.rs' or p.name == 'Cargo.toml'))
+    require(actual == set(by_path), 'complete current ROOT Rust/manifests inventory, no unlisted code')
+    for row in (config['ROOT_source_patch_pins']+config['ROOT_composition_journal_pins']+
+                [config['ROOT_focused_check_pin']]):
+        _raw(row['path'],row,require)
+    initial_pin = config['ROOT_original_source_freeze']
+    initial = json.loads(_raw(initial_pin['path'],initial_pin,require))
+    require(len(initial['files']) == 521 and initial['full_model_or_GPU_qualified'] is False,
+            'original ROOT-reviewed source closure before recorded corrections')
+    for exception in config['normalization_exceptions']:
+        require(exception['classification'] in
+                ('adapter_only_two_head_count_arguments','test_only_two_vec_array_substitutions',
+                 'test_only_same_H48_budget_valid_schedule_parameter_drop',
+                 'test_only_unused_import_removal','test_only_array_and_real_pressure_chunks',
+                 'test_only_valid_layer_schedule','test_only_explicit_closed_embedding_role_negative',
+                 'test_only_distinct_fresh_source_identity'), 'only named ROOT adapter/test corrections')
+        require(exception['isolated']['path'] == str(root/exception['file']) and
+                by_path.get(exception['isolated']['path']) == exception['isolated'],
+                'exception must match exact ROOT source-freeze row')
+        before = _raw(exception['original']['path'],exception['original'],require)
+        if exception['classification'] == 'test_only_explicit_closed_embedding_role_negative':
+            original_rows = [row for row in initial['files'] if row['file'] == exception['file']]
+            require(len(original_rows) == 1 and
+                    (original_rows[0]['bytes'],original_rows[0]['sha256']) ==
+                    (exception['original']['bytes'],exception['original']['sha256']),
+                    'retained exact original ROOT attention fixture; no MAIN dependency')
+        after = _raw(exception['isolated']['path'],exception['isolated'],require)
+        require(normalized(before,exception['replacements'],require) == after,
+                'only declared ROOT adapter/test changes; no arithmetic or unreviewed body change')
+    return by_path
+
+def frozen(here, read_json, pin, require, sha):
+    manifest = read_json(here/'prepared-manifest.json')
+    rows = manifest['files']
+    require(type(rows) is list and len(rows) == 18 and
+            len({row['file'] for row in rows}) == 18 and
+            all(row['file'] == Path(row['file']).name for row in rows),
+            'exact18 finite relocated source/metadata filenames')
+    for row in rows:
+        pin(here/row['file'],row)
+        pin(here/'frozen-prepared'/row['file'],row)
+    config = read_json(here/'relocation.json')
+    require(config['full_HF_model_StateRoot_exp_initial_GDN_GPU_or_performance_qualified'] is False and
+            config['path_remap_is_not_MAIN_merge_or_independent_build_proof'] is True,
+            'relocation scope remains source-only isolated')
+    _copies(here,config,require)
+    prior = _original(config,read_json,pin,require)
+    require(len(config['copy_map']) == len(prior['files']) == 13 and
+            {row['file'] for row in config['copy_map']} == {row['file'] for row in prior['files']},
+            'all13 original harness/fixture/lifetime files conserved')
+    require([str(remap(p,config,require)) for p in prior['required_build_paths']] == config['required_build_paths'],
+            'all original required paths conserved by exact root/local remap')
+    require([m['original'] for m in config['expected_child_map']] == prior['expected_merged_children'],
+            'all original expected-child assertions conserved')
+    workspace(config,read_json,require)
+    for item in config['expected_child_map']:
+        require(item['isolated']['path'] == str(remap(item['original']['path'],config,require)),
+                'each expected child maps to exact intended source')
+        actual = item['isolated']
+        _raw(actual['path'],actual,require)
+        if item['classification'] == 'BYTE_IDENTICAL_ROOT_REMAP':
+            require((actual['bytes'],actual['sha256']) == (item['original']['bytes'],item['original']['sha256']),
+                    'unchanged original arithmetic/fixture bytes under new root')
+        elif item['classification'] == 'ONLY_LOCAL_DEPENDENCY_ROOT_REMAP':
+            require(Path(actual['path']) == here/'Cargo.toml', 'only exact local Cargo root substitution')
+        else:
+            match = [e for e in config['normalization_exceptions'] if e['isolated'] == actual and
+                     e['classification'] == item['classification'] and
+                     e['original']['bytes'] == item['original']['bytes'] and
+                     e['original']['sha256'] == item['original']['sha256']]
+            require(len(match) == 1, 'only explicitly reviewed source-identity preserving ROOT test corrections')
+    return sha(here/'prepared-manifest.json')
+
+def implementation(here, path, read_json, pin, require, sha):
+    freeze = frozen(here,read_json,pin,require,sha)
+    config = read_json(here/'relocation.json')
+    value = read_json(path)
+    require(value['status'] == config['required_isolated_status'] and
+            value['prepared_manifest_sha256'] == freeze and
+            value['original_prepared_manifest_sha256'] == config['original_preparation']['sha256'] and
+            value['root_source_freeze_sha256'] == config['source_freeze']['sha256'] and
+            value['workspace_root'] == config['workspace_root'] and
+            value[config['false_scope_flag']] is False and
+            value['MAIN_merged'] is False,
+            'ROOT reviewed ISOLATED build-source receipt; old MERGED/MAIN status refused')
+    expected = workspace(config,read_json,require)
+    for name in ('Cargo.toml','main.rs','factory.rs','source-pin.json','source-bindings.json'):
+        entry = next(row for row in config['copy_map'] if row['file'] == name)['relocated']
+        expected[entry['path']] = entry
+    # A Cargo-generated local lock, if present, is an explicit extra actual build
+    # input. It does not replace ROOT's pinned whole-workspace lock/cohort proof.
+    lock = here/'Cargo.lock'
+    if lock.exists():
+        with os.fdopen(os.open(lock,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW),'rb') as opened:
+            before = os.fstat(opened.fileno())
+            require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 2*1024**2,
+                    'local build lock bound BEFORE reading')
+            raw = opened.read(2*1024**2+1)
+            after = os.fstat(opened.fileno())
+        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
+                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and
+                len(raw) == before.st_size, 'actual local build lock unchanged')
+        expected[str(lock)] = dict(path=str(lock),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
+    rows = value['files']
+    require(type(rows) is list and len(rows) == len(expected) <= 527 and
+            len({row['path'] for row in rows}) == len(rows), 'complete bounded521+5(+lock) build-source list')
+    by_path = {row['path']:row for row in rows}
+    require(by_path == expected and set(config['required_build_paths']) <= set(by_path),
+            'exact full isolated ROOT +unchanged local build inputs; no MAIN aliases or missing code')
+    for row in rows:
+        _raw(row['path'],row,require)
+    return sha(path)
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/relocation.json
@@ -0,0 +1,690 @@
+{
+  "ROOT_composition_journal_pins": [
+    {
+      "bytes": 894,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-parent-source-verification-1.json",
+      "sha256": "a2b255f50a037c42f32a3d831afc7f9c65157cd76fc0c25d70a4f2e319c59616"
+    },
+    {
+      "bytes": 6676,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/coefficient-core-composition-1.json",
+      "sha256": "251b74b71ec6ec150292784af7b90b2595fdb9fe38f65e96992c50bd438d4a8e"
+    },
+    {
+      "bytes": 229,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/coefficient-cold-refusal-composition-1.json",
+      "sha256": "426ef7301495c43a836bf7a3683101c61c5a11746245a39da1663f427a8f8901"
+    },
+    {
+      "bytes": 2732,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/gamma-composition-1.json",
+      "sha256": "4ce2b9c34f6a370a05e1592cc3bb772707ea43e8067685e1cd765227282c871f"
+    },
+    {
+      "bytes": 1344,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/gamma-composition-2.json",
+      "sha256": "5a613cfb590d2f1b4557b050c0284fcc1ae844d370176e4610d3beaaffc37def"
+    },
+    {
+      "bytes": 367,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/gamma-composition-3.json",
+      "sha256": "3d137e5fc87d304cc6a53f06ca1c1544dd1a36bdf40d74a8b94ab476ff84f24d"
+    },
+    {
+      "bytes": 379,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/gamma-composition-4.json",
+      "sha256": "17cd79987b3ef158eb494606a7f8ba80753f2c4523d9abd0d288bb1225863bdb"
+    },
+    {
+      "bytes": 361,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/gamma-composition-5.json",
+      "sha256": "15f4b9ddc5fb8c2dbadd8d09bdfbd042ff27ae493463cac09f0195b158ce9b6c"
+    }
+  ],
+  "ROOT_focused_check_pin": {
+    "bytes": 1738,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-cpu-focused-7.json",
+    "sha256": "7d72b5c08e45316e5ab8ec29a8e605392abca1cc1e5fd835aeb97b69ef82a8aa"
+  },
+  "ROOT_focused_check_scope": "Actual ROOT strict +237 source-free focused tests only; original trained reference replays remain required",
+  "ROOT_original_source_freeze": {
+    "bytes": 98702,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-1.json",
+    "sha256": "a227fa210ed042be140b3c3a88a5672dbb3ed0a7fced8357e10f9f780506bac1"
+  },
+  "ROOT_source_patch_pins": [
+    {
+      "bytes": 988,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-coefficient-output-size-adapter-2.patch",
+      "sha256": "035c17b5db5559b473249561dd5cf99428ef2f7239778f77ee4a59fd3f1ece8f"
+    },
+    {
+      "bytes": 805,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-coefficient-test-array-style-3.patch",
+      "sha256": "30130924e17c8d8c08e0582c77dc2e8f46ec3e99342aa3a938036a0264588e3a"
+    },
+    {
+      "bytes": 1201,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-window-test-visibility-import-4.patch",
+      "sha256": "11e8fe4b19d5f18873a289db0dabd5a69a02bd2fa66ae8c3a10faf73e65e236f"
+    },
+    {
+      "bytes": 362,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-coefficient-test-array-style-5.patch",
+      "sha256": "10c3b4eec7d42cca192f9972c05d20e2ef88988c1f22a84d7d456395da015c6a"
+    },
+    {
+      "bytes": 3747,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-fixture-contract-reconciliation-6.patch",
+      "sha256": "8ad1b36e65f521621630de8018edea13857e74a75e7a2ee97e351e8f870788f7"
+    },
+    {
+      "bytes": 558,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-fixture-format-6.patch",
+      "sha256": "252c4e4d6bfffdc2267086341fdeeaf877be8489003fd68449599ce9dc987c19"
+    },
+    {
+      "bytes": 1527,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-tied-role-parameter-lifetime-fixture-7.patch",
+      "sha256": "d83a5e9564e7420c400422f69ae86b07fd9e7ad015046c91d6811be3c50d1632"
+    }
+  ],
+  "case": "cached",
+  "copy_map": [
+    {
+      "file": "Cargo.toml",
+      "original": {
+        "bytes": 696,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/Cargo.toml",
+        "sha256": "b7923bb59732630e3dd442e506e84fd5bdbeac7f820159164b6d65ddb5907289"
+      },
+      "policy": "ONLY_LOCAL_DEPENDENCY_ROOT_REMAP",
+      "relocated": {
+        "bytes": 819,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/Cargo.toml",
+        "sha256": "fd33af8779c0e4eecd5703e07d43c1a01770a4ac509c0fae516d5af758e34ad0"
+      }
+    },
+    {
+      "file": "ROOT.md",
+      "original": {
+        "bytes": 3981,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/ROOT.md",
+        "sha256": "4be3506715b46adf49998cd6b3e7dfe2fd18707dd5e811077a57984701210470"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 3981,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/ROOT.md",
+        "sha256": "4be3506715b46adf49998cd6b3e7dfe2fd18707dd5e811077a57984701210470"
+      }
+    },
+    {
+      "file": "contracts.py",
+      "original": {
+        "bytes": 6025,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/contracts.py",
+        "sha256": "c99eb9260e6af834ef7d1f69edcb99eb6ae82ca425192c9b5aef79b52caff794"
+      },
+      "policy": "ONLY_SOURCE_ATTESTATION_CONTRACT_RELOCATION",
+      "relocated": {
+        "bytes": 4350,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/contracts.py",
+        "sha256": "92d773ae4a78eee636f3ffe456f1dfc454597aa9c02755ac0db4b30d979c270a"
+      }
+    },
+    {
+      "file": "export_reference.py",
+      "original": {
+        "bytes": 10530,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/export_reference.py",
+        "sha256": "4cdee4e696213ce8c0f93cdbcd3d567ad629f035a16e7d931a6ac626edb0bb6e"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 10530,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/export_reference.py",
+        "sha256": "4cdee4e696213ce8c0f93cdbcd3d567ad629f035a16e7d931a6ac626edb0bb6e"
+      }
+    },
+    {
+      "file": "main.rs",
+      "original": {
+        "bytes": 17126,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/main.rs",
+        "sha256": "9399b8e3f849b7ea9179f3458b36781a5f91a5028e68724513177e717d81b299"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 17126,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/main.rs",
+        "sha256": "9399b8e3f849b7ea9179f3458b36781a5f91a5028e68724513177e717d81b299"
+      }
+    },
+    {
+      "file": "factory.rs",
+      "original": {
+        "bytes": 29779,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/factory.rs",
+        "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 29779,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/factory.rs",
+        "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
+      }
+    },
+    {
+      "file": "source-pin.json",
+      "original": {
+        "bytes": 7058,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/source-pin.json",
+        "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 7058,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/source-pin.json",
+        "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+      }
+    },
+    {
+      "file": "owned_lifetime.py",
+      "original": {
+        "bytes": 3702,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/owned_lifetime.py",
+        "sha256": "c10d034bd8ae0ca1dc065b68bd27f2bbe6f092a0cda0cbf8af738a363f0229b8"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 3702,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/owned_lifetime.py",
+        "sha256": "c10d034bd8ae0ca1dc065b68bd27f2bbe6f092a0cda0cbf8af738a363f0229b8"
+      }
+    },
+    {
+      "file": "owned_native.py",
+      "original": {
+        "bytes": 5561,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/owned_native.py",
+        "sha256": "e339e23ce5d1c5e1353fed1a8ec6b67735050e565d94b2af32eb81a9082980ab"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 5561,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/owned_native.py",
+        "sha256": "e339e23ce5d1c5e1353fed1a8ec6b67735050e565d94b2af32eb81a9082980ab"
+      }
+    },
+    {
+      "file": "run_replay.py",
+      "original": {
+        "bytes": 3886,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/run_replay.py",
+        "sha256": "0f535245d22ff96a59659d55f015caccd53026d496b83464c8f676349f9dcf11"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 3886,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/run_replay.py",
+        "sha256": "0f535245d22ff96a59659d55f015caccd53026d496b83464c8f676349f9dcf11"
+      }
+    },
+    {
+      "file": "source-bindings.json",
+      "original": {
+        "bytes": 9260,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/source-bindings.json",
+        "sha256": "55dfb8e385a53d01abbb8fab762f492022f7217aa8a2fc95e3dd42c0a5146a13"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 9260,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/source-bindings.json",
+        "sha256": "55dfb8e385a53d01abbb8fab762f492022f7217aa8a2fc95e3dd42c0a5146a13"
+      }
+    },
+    {
+      "file": "test_contracts.py",
+      "original": {
+        "bytes": 4430,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/test_contracts.py",
+        "sha256": "4db1acac757844fd9b7eaf41b9c3bfd24f0e603c3b87493c6835d11345fd9531"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 4430,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/test_contracts.py",
+        "sha256": "4db1acac757844fd9b7eaf41b9c3bfd24f0e603c3b87493c6835d11345fd9531"
+      }
+    },
+    {
+      "file": "validate_replay.py",
+      "original": {
+        "bytes": 7558,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/validate_replay.py",
+        "sha256": "6f074e414d0d7e90804854b3d86dca39827179dd63acc4075da453ec43bbd141"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 7558,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/validate_replay.py",
+        "sha256": "6f074e414d0d7e90804854b3d86dca39827179dd63acc4075da453ec43bbd141"
+      }
+    }
+  ],
+  "expected_child_map": [
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 9795,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program/cached_gdn.rs",
+        "sha256": "83eea3ec759971840ab4cc0582ed13bc49bb7d9c033bad7c49ecd9041c49f206"
+      },
+      "original": {
+        "bytes": 9795,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/accelerator/program/cached_gdn.rs",
+        "sha256": "83eea3ec759971840ab4cc0582ed13bc49bb7d9c033bad7c49ecd9041c49f206"
+      }
+    },
+    {
+      "classification": "test_only_unused_import_removal",
+      "isolated": {
+        "bytes": 11198,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs",
+        "sha256": "c2a0115da9308799aacf450c9c1c12f7b05141b9f57d32479aef2e581866a0d4"
+      },
+      "original": {
+        "bytes": 11219,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs",
+        "sha256": "d255862bbc9e97b2867c12418df0fac628caaedb0099b5d5e23a55d08506fa84"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 8136,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gdn_cached_slice.rs",
+        "sha256": "e80ebb297e18e597a6177c38780b282b29e89fe294e6858d8c97bfa6b686eb7b"
+      },
+      "original": {
+        "bytes": 8136,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/gdn_cached_slice.rs",
+        "sha256": "e80ebb297e18e597a6177c38780b282b29e89fe294e6858d8c97bfa6b686eb7b"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 7735,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math/cached_gdn.rs",
+        "sha256": "cfb8884d0e28e48639400ef1c5108f6927deeda4d2d5f216f547cebe09583832"
+      },
+      "original": {
+        "bytes": 7735,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/host_math/cached_gdn.rs",
+        "sha256": "cfb8884d0e28e48639400ef1c5108f6927deeda4d2d5f216f547cebe09583832"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 7307,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/recurrent.rs",
+        "sha256": "b7489e52d220e07ba0c2e324b455f0240edf2ba5c21fcd6fe13befc39b8989a6"
+      },
+      "original": {
+        "bytes": 7307,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/original_hf/recurrent.rs",
+        "sha256": "b7489e52d220e07ba0c2e324b455f0240edf2ba5c21fcd6fe13befc39b8989a6"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 12129,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/reduction.rs",
+        "sha256": "5c65b181b04d1cedd3debf54b6b862cf0aa30fb38b7decfd0eedbe050ad234c4"
+      },
+      "original": {
+        "bytes": 12129,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/original_hf/reduction.rs",
+        "sha256": "5c65b181b04d1cedd3debf54b6b862cf0aa30fb38b7decfd0eedbe050ad234c4"
+      }
+    },
+    {
+      "classification": "test_only_valid_layer_schedule",
+      "isolated": {
+        "bytes": 10274,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs",
+        "sha256": "8a70bd9f52f299ac1cbac9df3e420f1096edd0a5acd77779bcae4d5aac99dcbc"
+      },
+      "original": {
+        "bytes": 10208,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs",
+        "sha256": "7e47e38f7b50277261de04cad6567cfbae4120015aa6b68d95db0433003e2db1"
+      }
+    },
+    {
+      "classification": "ONLY_LOCAL_DEPENDENCY_ROOT_REMAP",
+      "isolated": {
+        "bytes": 819,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/Cargo.toml",
+        "sha256": "fd33af8779c0e4eecd5703e07d43c1a01770a4ac509c0fae516d5af758e34ad0"
+      },
+      "original": {
+        "bytes": 696,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/Cargo.toml",
+        "sha256": "b7923bb59732630e3dd442e506e84fd5bdbeac7f820159164b6d65ddb5907289"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 17126,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/main.rs",
+        "sha256": "9399b8e3f849b7ea9179f3458b36781a5f91a5028e68724513177e717d81b299"
+      },
+      "original": {
+        "bytes": 17126,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/main.rs",
+        "sha256": "9399b8e3f849b7ea9179f3458b36781a5f91a5028e68724513177e717d81b299"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 29779,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/factory.rs",
+        "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
+      },
+      "original": {
+        "bytes": 29779,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/factory.rs",
+        "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 7058,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/source-pin.json",
+        "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+      },
+      "original": {
+        "bytes": 7058,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/source-pin.json",
+        "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 9260,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/source-bindings.json",
+        "sha256": "55dfb8e385a53d01abbb8fab762f492022f7217aa8a2fc95e3dd42c0a5146a13"
+      },
+      "original": {
+        "bytes": 9260,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/source-bindings.json",
+        "sha256": "55dfb8e385a53d01abbb8fab762f492022f7217aa8a2fc95e3dd42c0a5146a13"
+      }
+    }
+  ],
+  "false_scope_flag": "full_model_or_StateRoot_or_exp_or_GPU_qualified",
+  "full_HF_model_StateRoot_exp_initial_GDN_GPU_or_performance_qualified": false,
+  "main_root": "/home/pvl/spaces/abird/src/zerogravity",
+  "normalization_exceptions": [
+    {
+      "classification": "adapter_only_two_head_count_arguments",
+      "file": "crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs",
+      "isolated": {
+        "bytes": 6351,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs",
+        "sha256": "c4cbb9c57cb1b39db15aae4533fa3baf3724974f41b74a9b842abb572aca8d60"
+      },
+      "original": {
+        "bytes": 6323,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/proposed/crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs",
+        "sha256": "23929aec9bfc531b8cb0d1259017358f706454c1ae0636c09ff442950c51efcf"
+      },
+      "replacements": [
+        {
+          "after": "            ));\n        }\n        #[cfg(all(feature = \"original-hf-binding\", target_os = \"linux\"))]\n        let workspace = self.prepare_output(None, 2 * heads, 1)?; // BEFORE both private copies.\n        let decode = |raw: &[u8]| {\n            raw.chunks_exact(2)\n                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))\n",
+          "before": "            ));\n        }\n        #[cfg(all(feature = \"original-hf-binding\", target_os = \"linux\"))]\n        let workspace = self.prepare_output(None)?; // BEFORE both private copies.\n        let decode = |raw: &[u8]| {\n            raw.chunks_exact(2)\n                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))\n",
+          "expected_occurrences": 1
+        },
+        {
+          "after": "                    ));\n                }\n            }\n            self.prepare_output(Some(alpha), 3 * heads, 1)? // Shared aggregate for ALL3 outputs.\n        };\n        let prepared = match &self.source {\n            HostSource::Loaded(owner) => {\n",
+          "before": "                    ));\n                }\n            }\n            self.prepare_output(Some(alpha))? // Shared aggregate for ALL3 outputs.\n        };\n        let prepared = match &self.source {\n            HostSource::Loaded(owner) => {\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_two_vec_array_substitutions",
+      "file": "crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs",
+      "isolated": {
+        "bytes": 7234,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs",
+        "sha256": "a3032c27074f9a49545052f3d96062a02539616f45c1d0d8e3c9d578ed75e572"
+      },
+      "original": {
+        "bytes": 7242,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/proposed/crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs",
+        "sha256": "031299ebca94fd5a4d1ea81ba2e8447e2d94bebe83de9076f877e6d7ded3dcd5"
+      },
+      "replacements": [
+        {
+          "after": "    #[test]\n    fn raw_f32_dt_is_not_bf16_rounded_or_adjusted() {\n        let word = 0x3380_0001u32;\n        let raw = [word.to_le_bytes(); 16].concat();\n        let zero = [0.0; 16];\n        let out = prepare_gdn_coefficients(\n            source(),\n",
+          "before": "    #[test]\n    fn raw_f32_dt_is_not_bf16_rounded_or_adjusted() {\n        let word = 0x3380_0001u32;\n        let raw = vec![word.to_le_bytes(); 16].concat();\n        let zero = [0.0; 16];\n        let out = prepare_gdn_coefficients(\n            source(),\n",
+          "expected_occurrences": 1
+        },
+        {
+          "after": "        )\n        .unwrap();\n        assert!(out.biased_alpha().iter().all(|v| v.to_bits() == word));\n        assert_eq!(raw, [word.to_le_bytes(); 16].concat());\n    }\n    #[test]\n    fn source_raw_dtype_span_and_nonfinite_refuse() {\n",
+          "before": "        )\n        .unwrap();\n        assert!(out.biased_alpha().iter().all(|v| v.to_bits() == word));\n        assert_eq!(raw, vec![word.to_le_bytes(); 16].concat());\n    }\n    #[test]\n    fn source_raw_dtype_span_and_nonfinite_refuse() {\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_same_H48_budget_valid_schedule_parameter_drop",
+      "file": "crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs",
+      "isolated": {
+        "bytes": 13508,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs",
+        "sha256": "b6f7fcb6d8828051b6004b90f10c7b549303d94c9c1bf358ca701d3fda524b40"
+      },
+      "original": {
+        "bytes": 13482,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/test-style-v2/frozen/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs",
+        "sha256": "1ee2170fceaebb0bbc0a6f48daf4d73687f6429eb48ca692a65c15540d798763"
+      },
+      "replacements": [
+        {
+          "after": ") -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {\n    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();\n    let text = &mut cfg[\"text_config\"];\n    text[\"num_hidden_layers\"] = json!(2);\n    text[\"full_attention_interval\"] = json!(2);\n    text[\"layer_types\"] = json!([\"linear_attention\", \"full_attention\"]);\n    text[\"hidden_size\"] = json!(256);\n    text[\"linear_num_key_heads\"] = json!(16);\n    text[\"linear_num_value_heads\"] = json!(48);\n",
+          "before": ") -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {\n    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();\n    let text = &mut cfg[\"text_config\"];\n    text[\"num_hidden_layers\"] = json!(1);\n    text[\"layer_types\"] = json!([\"linear_attention\"]);\n    text[\"hidden_size\"] = json!(256);\n    text[\"linear_num_key_heads\"] = json!(16);\n    text[\"linear_num_value_heads\"] = json!(48);\n",
+          "expected_occurrences": 1
+        },
+        {
+          "after": "            .is_err()\n    );\n    assert_eq!(reserved(&host), before);\n    // Independent declared H48 payloads and conservative scratch floors.\n    let input_need = (2 * 48 * 128 * 4 + 64 * 1024) as u64;\n    let output_need = (48 * 128 * 4 + 960 * 1024) as u64;\n    let pressure = host\n        .arbiter\n        .admit(\n",
+          "before": "            .is_err()\n    );\n    assert_eq!(reserved(&host), before);\n    let input_need = (crate::host_math::gated_rms::INPUT_BYTES\n        + crate::host_math::gated_rms::INPUT_SCRATCH) as u64;\n    let output_need = (crate::host_math::gated_rms::OUTPUT_BYTES\n        + crate::host_math::gated_rms::OUTPUT_SCRATCH) as u64;\n    let pressure = host\n        .arbiter\n        .admit(\n",
+          "expected_occurrences": 1
+        },
+        {
+          "after": "    same(expected.output_bf16(), result.output_bf16().unwrap());\n    drop(expected);\n    drop(gamma);\n    drop(parameter);\n    let bits = result\n        .output_bf16()\n        .unwrap()\n",
+          "before": "    same(expected.output_bf16(), result.output_bf16().unwrap());\n    drop(expected);\n    drop(gamma);\n    let bits = result\n        .output_bf16()\n        .unwrap()\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_unused_import_removal",
+      "file": "crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs",
+      "isolated": {
+        "bytes": 11198,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs",
+        "sha256": "c2a0115da9308799aacf450c9c1c12f7b05141b9f57d32479aef2e581866a0d4"
+      },
+      "original": {
+        "bytes": 11219,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/proposed/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs",
+        "sha256": "d255862bbc9e97b2867c12418df0fac628caaedb0099b5d5e23a55d08506fa84"
+      },
+      "replacements": [
+        {
+          "after": "mod source_gamma;\n\nuse super::*;\nuse crate::accelerator::program::CachedGdnContract;\nuse crate::operator_slice::cached_gdn::CachedGdnSliceInput;\nuse zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};\n\n",
+          "before": "mod source_gamma;\n\nuse super::*;\nuse crate::accelerator::program::{CachedGdnContract, CachedGdnOperands};\nuse crate::operator_slice::cached_gdn::CachedGdnSliceInput;\nuse zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};\n\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_array_and_real_pressure_chunks",
+      "file": "crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
+      "isolated": {
+        "bytes": 8890,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
+        "sha256": "74b8a311402e42c04cd3060dd34a8c4dfa05dd2d061f4d2ae89f896eecb7fec8"
+      },
+      "original": {
+        "bytes": 8682,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/proposed/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
+        "sha256": "36e443b77bd8641c4005d88ee51316196a0507905a7b5cf47bdb30b3356fcfcf"
+      },
+      "replacements": [
+        {
+          "after": "    )\n}\nfn raw(value: u16) -> Vec<u8> {\n    [value.to_le_bytes(); 16].concat()\n}\nfn input<'a>(\n    source: crate::accelerator::SourceIdentity,\n",
+          "before": "    )\n}\nfn raw(value: u16) -> Vec<u8> {\n    vec![value.to_le_bytes(); 16].concat()\n}\nfn input<'a>(\n    source: crate::accelerator::SourceIdentity,\n",
+          "expected_occurrences": 1
+        },
+        {
+          "after": "            .is_err()\n    );\n    assert_eq!(reserved(&admission), before);\n    let mut pressure = Vec::new();\n    loop {\n        let remaining = admission\n            .arbiter\n            .snapshot(&admission.resource)\n            .unwrap()\n            .remaining;\n        if remaining == 0 {\n            break;\n        }\n        pressure.push(\n            source\n                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)\n                .unwrap(),\n        );\n    }\n    let pressured = reserved(&admission);\n    assert!(\n        operators\n",
+          "before": "            .is_err()\n    );\n    assert_eq!(reserved(&admission), before);\n    let remaining = admission\n        .arbiter\n        .snapshot(&admission.resource)\n        .unwrap()\n        .remaining;\n    let pressure = source.reserve_workspace(remaining as usize).unwrap();\n    let pressured = reserved(&admission);\n    assert!(\n        operators\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_valid_layer_schedule",
+      "file": "crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs",
+      "isolated": {
+        "bytes": 10274,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs",
+        "sha256": "8a70bd9f52f299ac1cbac9df3e420f1096edd0a5acd77779bcae4d5aac99dcbc"
+      },
+      "original": {
+        "bytes": 10208,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/proposed/crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs",
+        "sha256": "7e47e38f7b50277261de04cad6567cfbae4120015aa6b68d95db0433003e2db1"
+      },
+      "replacements": [
+        {
+          "after": "fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {\n    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();\n    let text = &mut cfg[\"text_config\"];\n    text[\"num_hidden_layers\"] = json!(2);\n    text[\"full_attention_interval\"] = json!(2);\n    text[\"layer_types\"] = json!([\"linear_attention\", \"full_attention\"]);\n    text[\"hidden_size\"] = json!(256);\n    text[\"linear_num_key_heads\"] = json!(16);\n    text[\"linear_num_value_heads\"] = json!(48);\n",
+          "before": "fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {\n    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();\n    let text = &mut cfg[\"text_config\"];\n    text[\"num_hidden_layers\"] = json!(1);\n    text[\"layer_types\"] = json!([\"linear_attention\"]);\n    text[\"hidden_size\"] = json!(256);\n    text[\"linear_num_key_heads\"] = json!(16);\n    text[\"linear_num_value_heads\"] = json!(48);\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_explicit_closed_embedding_role_negative",
+      "file": "crates/zerogravity-model-qwen35/src/attention_slice_tests.rs",
+      "isolated": {
+        "bytes": 16955,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/attention_slice_tests.rs",
+        "sha256": "f1cc417f41b62acb368161396ff703154c1fcce9ad480ff5c14c9c09518fdf9b"
+      },
+      "original": {
+        "bytes": 16627,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/baseline-attention-slice-tests.rs",
+        "sha256": "57e3abc73a91cf7f661b64db2865c47c21e9eb9c2ca4f6483546d60643b2730e"
+      },
+      "replacements": [
+        {
+          "after": "        math.ordinary_rms_profiled(gdn_gamma, &imported, OrdinaryRmsContract::TorchCpuBf16Avx2)\n            .is_err()\n    );\n    // Tied output and embedding share a descriptor; the explicit role API\n    // rejects Embedding before importing another private value.\n    assert!(\n        operators\n            .execute_projection_slice(\n                QwenMatrixRole::Embedding,\n                crate::operator_slice::pure::PureSliceInput::from_bf16_bytes(\n                    source.source_identity(),\n                    ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,\n                    &bytes,\n                )\n                .unwrap(),\n                None,\n            )\n            .is_err()\n    );\n    assert!(\n        math.ordinary_rms_profiled(\n",
+          "before": "        math.ordinary_rms_profiled(gdn_gamma, &imported, OrdinaryRmsContract::TorchCpuBf16Avx2)\n            .is_err()\n    );\n    assert!(\n        math.project_profiled(\n            operators\n                .matrix_descriptor(QwenMatrixRole::FfnGate(3))\n                .unwrap(),\n            &imported,\n            ProjectionContract::TorchCpuBf16Avx2\n        )\n        .is_err()\n    );\n    assert!(\n        math.ordinary_rms_profiled(\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_distinct_fresh_source_identity",
+      "file": "crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs",
+      "isolated": {
+        "bytes": 16480,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs",
+        "sha256": "9769cf2902bf6fda572a4a52010cfbedb5bbc06e9869f3f6306043290389df8f"
+      },
+      "original": {
+        "bytes": 16477,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/followup-v3/proposed/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs",
+        "sha256": "7108060b361abe6d52680e161b75e3ec16e2057a019ace4c6455400b5067617d"
+      },
+      "replacements": [
+        {
+          "after": "}\n\n#[test]\nfn separately_opened_same_content_source_cannot_rebind_a_retained_input() {\n    let fixture = Fixture::new();\n    let loaded = source_with_options(&fixture, false, 256, true, |_| {});\n    let (source_a, admission_a) = windows(&fixture, &loaded);\n    let (source_b, admission_b) = windows(&fixture, &loaded);\n    assert_ne!(source_a.source_identity(), source_b.source_identity());\n    assert!(!source_a.same_issuer(&source_b));\n    let operators_a = bind_windows(source_a.clone(), &loaded);\n    let operators_b = bind_windows(source_b.clone(), &loaded);\n",
+          "before": "}\n\n#[test]\nfn same_content_foreign_window_issuer_cannot_rebind_a_retained_input() {\n    let fixture = Fixture::new();\n    let loaded = source_with_options(&fixture, false, 256, true, |_| {});\n    let (source_a, admission_a) = windows(&fixture, &loaded);\n    let (source_b, admission_b) = windows(&fixture, &loaded);\n    assert_eq!(source_a.source_identity(), source_b.source_identity());\n    assert!(!source_a.same_issuer(&source_b));\n    let operators_a = bind_windows(source_a.clone(), &loaded);\n    let operators_b = bind_windows(source_b.clone(), &loaded);\n",
+          "expected_occurrences": 1
+        }
+      ]
+    }
+  ],
+  "original_contract": {
+    "bytes": 6025,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/contracts.py",
+    "sha256": "c99eb9260e6af834ef7d1f69edcb99eb6ae82ca425192c9b5aef79b52caff794"
+  },
+  "original_harness": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay",
+  "original_preparation": {
+    "bytes": 13272,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/prepared-manifest.json",
+    "sha256": "4ecc28732a41c800dccb0e3d6a3bafe88049f0c1d09f6c39f3110940453b8118"
+  },
+  "path_remap_is_not_MAIN_merge_or_independent_build_proof": true,
+  "relocated_harness": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2",
+  "required_build_paths": [
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/Cargo.lock",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/Cargo.toml",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts/Cargo.toml",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts/src/source_weights.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts/src/ssd_extent.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts/src/verified_shards.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/Cargo.toml",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program/cached_gdn.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program/typed.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/resource_arbiter.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/Cargo.toml",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/recurrent.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/reduction.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-gpu/src/resident_program.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-gpu/src/resident_program/batch.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/Cargo.toml",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/forward.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gdn_cached_slice.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math/cached_gdn.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/original_binding/raw_state.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/original_binding/window.rs",
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/Cargo.toml",
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/main.rs",
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/factory.rs",
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/source-pin.json",
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/source-bindings.json",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs"
+  ],
+  "required_isolated_status": "ROOT_REVIEWED_ISOLATED_CACHED_GDN_SOURCE",
+  "schema": "zerogravity.original27b.replay.isolated-root-relocation.v2",
+  "source_count": 521,
+  "source_freeze": {
+    "bytes": 99248,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-7.json",
+    "sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615"
+  },
+  "workspace_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020"
+}
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/run_replay.py
@@ -0,0 +1,68 @@
+"""ROOT ONLY fresh owned native child; no provider/model/operator on import."""
+import argparse
+import hashlib
+import os
+from pathlib import Path
+import signal
+from contracts import HERE, frozen, implementation, read_json, require, sha, write
+from export_reference import validate as validate_archived
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
+        raise TimeoutError('finite original source audit and cached operator proof deadline')
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
+        require(report['status'] == 'PASS_PRIVATE_ARCHIVED_CACHED_GDN_ONLY' and
+                report['result']['output_words'] == 798720 and report['result']['compared_words'] == 1603680,
+                'complete actual cached results require independent raw validation')
+        write(args.new_output/'completed.json', {
+            'status':'EXECUTED_REQUIRES_INDEPENDENT_RAW_VALIDATION',
+            'freeze_manifest_sha256':freeze,'binary_sha256':args.binary_sha256,
+            'ROOT_build_source_attestation_sha256':source_attestation,
+            'archived_reference_validation':archived,
+            'report_sha256':sha(output/'report.json'),
+            'source_after_sha256':sha(output/'source-after.json'),
+            'invocation_sha256':sha(args.new_output/'27b-lifecycle/invocation.json'),
+            'full_model_or_live_decay_or_112StateRoots_or_GatedNorm_or_GPU_qualified':False})
+    finally:
+        os.close(fd)
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/source-bindings.json
@@ -0,0 +1,227 @@
+{
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
+  "factory_report": {
+    "bytes": 7058,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/source-pin.json",
+    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+  },
+  "heads": 48,
+  "layer": 0,
+  "model": "27b",
+  "plain_archive": {
+    "bytes": 6518376,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/plain.safetensors",
+    "sha256": "9b20753b733f5f870abf4691eaecf9ce42f5edec0ecec8ae1260684273d5bab8"
+  },
+  "producer_scope": "Authentic original27B layer0 cached public call, teacher353 after true teacher11 padded initialchunk64; decay is external observed exp(g)",
+  "profile": "original27B-raw-source-bf16-v1",
+  "schema": "zerogravity.cached-gdn.archived-source-bindings.h48.v1",
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
+  }
+}
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/source-pin.json
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
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/test_contracts.py
@@ -0,0 +1,77 @@
+"""Prepared source-free byte-validation fixtures; no provider/model calls."""
+import json
+from pathlib import Path
+import struct
+import tempfile
+import unittest
+from contracts import archive, digest, finite, read_json, widened
+from validate_replay import affinity
+from export_reference import public_scope
+
+class ByteContracts(unittest.TestCase):
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
+    def test_signed_zero_subnormal_and_bf16_transport(self):
+        raw = bytes.fromhex('008001000000')
+        finite(raw, 'BF16')
+        self.assertEqual(widened(raw, 'BF16'), bytes.fromhex('000000800000010000000000'))
+        fp = bytes.fromhex('0000008001000000')
+        finite(fp, 'F32')
+        self.assertEqual(widened(fp, 'F32'), fp)
+        for dtype, bad in [('BF16',bytes.fromhex('807f')),('F32',bytes.fromhex('000080ff'))]:
+            with self.assertRaises(ValueError):
+                finite(bad, dtype)
+
+    def test_actual_archive_extent_and_trailing_byte_refusals(self):
+        payload = bytes.fromhex('0000008001000000')
+        header = json.dumps({'x':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}).encode()
+        raw = struct.pack('<Q',len(header))+header+payload
+        with tempfile.TemporaryDirectory() as root:
+            path = Path(root)/'small.safetensors'
+            path.write_bytes(raw)
+            pin = {'path':str(path),'bytes':len(raw),'sha256':digest(raw)}
+            self.assertEqual(archive(pin)['x']['raw'], payload)
+            path.write_bytes(raw+b'\0')
+            with self.assertRaises(ValueError):
+                archive(dict(pin,bytes=len(raw)+1,sha256=digest(raw+b'\0')))
+
+    def test_duplicate_metadata_and_actual_cpu_binding(self):
+        with tempfile.TemporaryDirectory() as root:
+            path = Path(root)/'metadata.json'
+            path.write_text('{"x":1,"x":2}')
+            with self.assertRaises(ValueError):
+                read_json(path)
+        self.assertEqual(affinity('Cpus_allowed_list:\t0-3'), [0,1,2,3])
+        for bad in ('Cpus_allowed_list:\t0-2','Cpus_allowed_list:\t0-4','Cpus_allowed_list:\t0,1,1,2'):
+            with self.assertRaises(ValueError):
+                affinity(bad)
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/test_relocation.py
@@ -0,0 +1,238 @@
+"""ROOT-only source-free relocation tests; no tensors, operators or providers."""
+import copy
+import hashlib
+import json
+from pathlib import Path
+import tempfile
+import unittest
+from unittest import mock
+
+import relocated_source as contract
+
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+
+def row(path):
+    raw = Path(path).read_bytes()
+    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
+
+
+def write_json(path, value):
+    Path(path).write_text(json.dumps(value, sort_keys=True) + '\n')
+
+
+def read_json(path):
+    return json.loads(Path(path).read_text())
+
+
+class RelocationTests(unittest.TestCase):
+    def setUp(self):
+        self.temporary = tempfile.TemporaryDirectory(prefix='zg-source-relocation-')
+        self.addCleanup(self.temporary.cleanup)
+        self.base = Path(self.temporary.name).resolve()
+        self.main = self.base / 'main'
+        self.original = self.main / 'original-harness'
+        self.here = self.main / 'isolated-harness'
+        self.workspace = self.base / 'isolated-root'
+        for path in (self.original, self.here, self.workspace):
+            path.mkdir(parents=True)
+        self.config = dict(main_root=str(self.main), original_harness=str(self.original),
+                           relocated_harness=str(self.here), workspace_root=str(self.workspace))
+
+    def copy_fixture(self, name, before, after, policy):
+        a, b = self.original / name, self.here / name
+        a.write_bytes(before)
+        b.write_bytes(after)
+        config = dict(self.config, copy_map=[dict(file=name, original=row(a), relocated=row(b),
+                                                 policy=policy)])
+        return config
+
+    def source_fixture(self):
+        # Exactly the production inventory cardinality, with tiny synthetic
+        # source comments. This does not execute or authenticate model code.
+        files = []
+        for name in ('Cargo.toml', 'Cargo.lock'):
+            path = self.workspace / name
+            path.write_text('# synthetic source fixture\n')
+            files.append(dict(file=name, bytes=path.stat().st_size, sha256=row(path)['sha256']))
+        for index in range(519):
+            name = f'crates/fixture/src/s{index:03}.rs'
+            path = self.workspace / name
+            path.parent.mkdir(parents=True, exist_ok=True)
+            path.write_text(f'// synthetic source {index}\n')
+            files.append(dict(file=name, bytes=path.stat().st_size, sha256=row(path)['sha256']))
+        source = self.base / 'freeze.json'
+        write_json(source, dict(files=files, full_model_or_GPU_qualified=False))
+        config = dict(self.config, source_count=521, source_freeze=row(source),
+                      ROOT_original_source_freeze=row(source), ROOT_source_patch_pins=[],
+                      ROOT_composition_journal_pins=[], ROOT_focused_check_pin=row(source),
+                      normalization_exceptions=[])
+        return config
+
+    def implementation_fixture(self):
+        source = self.workspace / 'crates' / 'fixture.rs'
+        source.parent.mkdir()
+        source.write_text('// synthetic source receipt fixture\n')
+        expected = {str(source): row(source)}
+        copies = []
+        for name in ('Cargo.toml', 'main.rs', 'factory.rs', 'source-pin.json', 'source-bindings.json'):
+            path = self.here / name
+            path.write_text('synthetic source metadata only\n')
+            copies.append(dict(file=name, relocated=row(path)))
+            expected[str(path)] = row(path)
+        config = dict(self.config, copy_map=copies, required_isolated_status='ROOT_REVIEWED_ISOLATED_FIXTURE',
+                      original_preparation=dict(sha256='1'*64), source_freeze=dict(sha256='2'*64),
+                      false_scope_flag='full_model_qualified', required_build_paths=list(expected))
+        write_json(self.here/'relocation.json', config)
+        receipt = dict(status=config['required_isolated_status'], prepared_manifest_sha256='3'*64,
+                       original_prepared_manifest_sha256='1'*64, root_source_freeze_sha256='2'*64,
+                       workspace_root=str(self.workspace), full_model_qualified=False,
+                       MAIN_merged=False, files=list(expected.values()))
+        path = self.base / 'implementation.json'
+        return path, receipt, expected
+
+    def validate_implementation(self, path, receipt, expected):
+        write_json(path, receipt)
+        with mock.patch.object(contract, 'frozen', return_value='3'*64), \
+                mock.patch.object(contract, 'workspace', return_value=copy.deepcopy(expected)):
+            return contract.implementation(self.here, path, read_json, None, require,
+                                           lambda p: row(p)['sha256'])
+
+    def test_remap_only_exact_crate_manifest_and_original_local_inputs(self):
+        self.assertEqual(contract.remap(self.main/'crates/c/src/lib.rs', self.config, require),
+                         self.workspace/'crates/c/src/lib.rs')
+        self.assertEqual(contract.remap(self.main/'Cargo.lock', self.config, require),
+                         self.workspace/'Cargo.lock')
+        self.assertEqual(contract.remap(self.original/'main.rs', self.config, require), self.here/'main.rs')
+
+    def test_reject_artifact_sibling_and_noncanonical_build_paths(self):
+        for path in (self.main/'.agents/source.bin', self.original/'source.bin',
+                     self.base/'main-sibling/crates/lib.rs', self.main/'crates/../secret.rs'):
+            with self.subTest(path=str(path)), self.assertRaises(ValueError):
+                contract.remap(path, self.config, require)
+
+    def test_unique_normalization_rejects_missing_and_ambiguous_text(self):
+        edits = [dict(before='// before\n', after='// after\n', expected_occurrences=1)]
+        self.assertEqual(contract.normalized(b'// before\n', edits, require), b'// after\n')
+        for before in (b'// absent\n', b'// before\n// before\n'):
+            with self.assertRaises(ValueError):
+                contract.normalized(before, edits, require)
+
+    def test_cargo_only_root_substitution_not_new_options(self):
+        before = ''.join(f'dep{i} = "{self.main}/crates/c{i}"\n' for i in range(3)).encode()
+        after = before.replace((str(self.main)+'/crates/').encode(),
+                               (str(self.workspace)+'/crates/').encode())
+        config = self.copy_fixture('Cargo.toml', before, after, 'ONLY_LOCAL_DEPENDENCY_ROOT_REMAP')
+        contract._copies(self.here, config, require)
+        changed = self.here/'Cargo.toml'
+        changed.write_bytes(after+b'new_feature = true\n')
+        config['copy_map'][0]['relocated'] = row(changed)
+        with self.assertRaises(ValueError):
+            contract._copies(self.here, config, require)
+
+    def test_contract_replacement_preserves_all_non_attestation_code(self):
+        before = b'PREFIX = "raw-byte policy"\ndef frozen():\n    return 1\ndef implementation(path):\n    return path\n\ndef finite(raw, dtype):\n    return raw\n'
+        source = before.decode()
+        start, end = source.index('def frozen():'), source.index('\ndef finite(')
+        after = (source[:start]+contract.CONTRACT_REPLACEMENT+source[end:]).encode()
+        config = self.copy_fixture('contracts.py', before, after,
+                                   'ONLY_SOURCE_ATTESTATION_CONTRACT_RELOCATION')
+        contract._copies(self.here, config, require)
+        changed = self.here/'contracts.py'
+        changed.write_bytes(after.replace(b'return raw', b'return None'))
+        config['copy_map'][0]['relocated'] = row(changed)
+        with self.assertRaises(ValueError):
+            contract._copies(self.here, config, require)
+
+    def test_byte_identical_runner_rejects_rehashed_changed_limits(self):
+        config = self.copy_fixture('run_replay.py', b'WALL = 900\n', b'WALL = 901\n', 'BYTE_IDENTICAL')
+        with self.assertRaises(ValueError):
+            contract._copies(self.here, config, require)
+
+    def test_full_521_source_inventory_accepts_exact_and_rejects_new_code(self):
+        config = self.source_fixture()
+        self.assertEqual(len(contract.workspace(config, read_json, require)), 521)
+        (self.workspace/'crates/fixture/src/unlisted.rs').write_text('// unexpected source\n')
+        with self.assertRaises(ValueError):
+            contract.workspace(config, read_json, require)
+
+    def test_full_source_pin_rejects_changed_original_math_bytes(self):
+        config = self.source_fixture()
+        (self.workspace/'crates/fixture/src/s000.rs').write_text('// modified source\n')
+        with self.assertRaises(ValueError):
+            contract.workspace(config, read_json, require)
+
+    def test_rehashed_extra_change_still_fails_declared_normalization(self):
+        config = self.source_fixture()
+        target = self.workspace/'crates/fixture/src/s000.rs'
+        old = self.base/'original-source.rs'
+        old.write_bytes(target.read_bytes())
+        target.write_text('// reviewed change\n// undeclared extra change\n')
+        freeze = read_json(config['source_freeze']['path'])
+        for entry in freeze['files']:
+            if entry['file'] == str(target.relative_to(self.workspace)):
+                entry.update(bytes=target.stat().st_size, sha256=row(target)['sha256'])
+        write_json(config['source_freeze']['path'], freeze)
+        config['source_freeze'] = row(config['source_freeze']['path'])
+        config['ROOT_original_source_freeze'] = config['source_freeze']
+        config['ROOT_focused_check_pin'] = config['source_freeze']
+        config['normalization_exceptions'] = [dict(
+            file=str(target.relative_to(self.workspace)), classification='adapter_only_two_head_count_arguments',
+            original=row(old), isolated=row(target), replacements=[dict(
+                before=old.read_text(), after='// reviewed change\n', expected_occurrences=1)])]
+        with self.assertRaises(ValueError):
+            contract.workspace(config, read_json, require)
+
+    def test_source_reader_refuses_symlinks_and_oversized_metadata(self):
+        original = self.base/'small.rs'
+        original.write_text('// bounded source\n')
+        link = self.base/'linked.rs'
+        link.symlink_to(original)
+        with self.assertRaises(OSError):
+            contract._raw(link, row(original), require)
+        large = self.base/'oversized.rs'
+        with large.open('wb') as opened:
+            opened.truncate(2*1024**2+1)
+        with self.assertRaises(ValueError):
+            contract._raw(large, dict(bytes=large.stat().st_size, sha256='0'*64), require)
+
+    def test_isolated_receipt_accepts_exact_sources(self):
+        path, receipt, expected = self.implementation_fixture()
+        result = self.validate_implementation(path, receipt, expected)
+        self.assertEqual(result, row(path)['sha256'])
+
+    def test_old_main_merged_or_wrong_isolated_root_receipts_refuse(self):
+        path, original, expected = self.implementation_fixture()
+        for updates in (dict(status='ROOT_REVIEWED_MERGED_FIXTURE'), dict(MAIN_merged=True),
+                        dict(workspace_root=str(self.main)), dict(root_source_freeze_sha256='0'*64),
+                        dict(full_model_qualified=True)):
+            receipt = dict(original, **updates)
+            with self.subTest(updates=updates), self.assertRaises(ValueError):
+                self.validate_implementation(path, receipt, expected)
+
+    def test_receipt_missing_source_and_extra_main_alias_refuse(self):
+        path, original, expected = self.implementation_fixture()
+        missing = copy.deepcopy(original)
+        missing['files'].pop()
+        with self.assertRaises(ValueError):
+            self.validate_implementation(path, missing, expected)
+        extra = copy.deepcopy(original)
+        extra['files'].append(dict(path=str(self.main/'crates/alias.rs'), bytes=1, sha256='0'*64))
+        with self.assertRaises(ValueError):
+            self.validate_implementation(path, extra, expected)
+
+    def test_local_cargo_generated_lock_is_mandatory_actual_build_input(self):
+        path, receipt, expected = self.implementation_fixture()
+        lock = self.here/'Cargo.lock'
+        lock.write_text('# actual local generated build lock fixture\n')
+        with self.assertRaises(ValueError):
+            self.validate_implementation(path, receipt, expected)
+        receipt['files'].append(row(lock))
+        self.assertEqual(self.validate_implementation(path, receipt, expected), row(path)['sha256'])
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/validate_replay.py
@@ -0,0 +1,111 @@
+"""ROOT ONLY independent full signed-bit/source/ledger/native-lifetime check."""
+import argparse
+import json
+from pathlib import Path
+from contracts import INPUTS, OUTPUTS, STAGES, frozen, implementation, raw_file, read, read_json, require, sha, widened, write
+from export_reference import validate as validate_archived
+
+def affinity(text):
+    require(type(text) is str and text.startswith('Cpus_allowed_list:'), 'actual proc CPU affinity')
+    values = []
+    for item in text.split(':',1)[1].strip().split(','):
+        span = item.split('-')
+        require(1 <= len(span) <= 2 and all(v.isdecimal() for v in span), 'strict CPU span')
+        lo, hi = int(span[0]), int(span[-1])
+        require(0 <= lo <= hi < 4096, 'bounded actual CPU span')
+        values.extend(range(lo,hi+1))
+    require(len(values) == 4 and values == sorted(set(values)), 'four actual distinct CPUs')
+    return values
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--output', type=Path, required=True)
+    parser.add_argument('--reference-directory', type=Path, required=True)
+    parser.add_argument('--implementation', type=Path, required=True)
+    args = parser.parse_args()
+    freeze = frozen()
+    archived = validate_archived(args.reference_directory)
+    build_source = implementation(args.implementation)
+    done = read_json(args.output/'completed.json')
+    require(done['status'] == 'EXECUTED_REQUIRES_INDEPENDENT_RAW_VALIDATION' and
+            done['freeze_manifest_sha256'] == freeze and
+            done['ROOT_build_source_attestation_sha256'] == build_source and
+            done['archived_reference_validation'] == archived and
+            done['full_model_or_live_decay_or_112StateRoots_or_GatedNorm_or_GPU_qualified'] is False,
+            'exact original public/source/build receipt closure')
+    directory, lifecycle = args.output/'27b', args.output/'27b-lifecycle'
+    for file, key in [('report.json','report_sha256'),('source-after.json','source_after_sha256')]:
+        require(sha(directory/file) == done[key], 'exact native result/source receipt')
+    require(sha(lifecycle/'invocation.json') == done['invocation_sha256'], 'exact original owned-child receipt')
+    report = read_json(directory/'report.json')
+    invocation = read_json(lifecycle/'invocation.json')
+    source = read_json(directory/'source-after.json')
+    require(report['status'] == 'PASS_PRIVATE_ARCHIVED_CACHED_GDN_ONLY' and
+            report['inputs_sha256'] == archived['inputs_sha256'] and
+            report['reference_sha256'] == archived['reference_sha256'] and
+            report['full_model_or_live_coefficients_or_112StateRoots_or_GatedNorm_or_GPU_qualified'] is False,
+            'pure archived cached operator only')
+    require(invocation['status'] == 'CHILD_EXECUTED_REQUIRES_VALIDATION' and
+            invocation['binary_sha256'] == done['binary_sha256'] and invocation['freeze_manifest_sha256'] == freeze and
+            invocation['primary_error'] is None and invocation['cleanup_errors'] == [] and
+            invocation['exit_code'] == invocation['reaped_exit_code'] == 0 and
+            all(invocation[k] is True for k in ('leader_reaped','group_absent','pipe_eof','selector_closed','pipe_closed','log_closed')),
+            'actual independently retired original owned child/PGID/pipe')
+    require(affinity(report['actual_affinity']) == invocation['cpu_affinity'], 'actual four-CPU native execution')
+    require(source['all_equal'] is True and source['source_before'] == source['source_after'] == invocation['source_before'],
+            'actual whole original source unchanged')
+    result = report['result']
+    require(result['model'] == '27b' and result['heads'] == 48 and result['imported_words'] == 804960 and
+            result['output_words'] == 798720 and result['compared_words'] == 1603680 and
+            all(result[k] is True for k in ('pre_cancel','wrong_layer','real_pressure_refusal','external_decay_no_exp_or_initial_or_StateRoot','original_gamma_bf16_metadata_only')),
+            'all H48 stages and actual negative receipts')
+    mib = 1024*1024
+    input_bytes, output_bytes = 804960*4, 798720*4
+    scratch = 65536 + 48*(6*16384*4+16*128*4)+256*1024
+    steady, live, retained, final = [result[k+'_ledger'] for k in ('steady','live','retained','final')]
+    require(steady['reserved'] == 88*mib and steady['reservations'] == 4 and
+            steady['per_class'].get('Scratch',0) == 72*mib and steady['per_class'].get('Other',0) == 16*mib,
+            'source/two original binding/control reservations')
+    require(live['reserved'] == 88*mib+input_bytes+output_bytes+scratch and
+            retained['reserved'] == 80*mib+input_bytes+output_bytes+scratch,
+            'aggregate payload/scratch/source owners remain through caller drop')
+    for ledger in (live,retained):
+        require(ledger['per_class'].get('Output',0) == input_bytes+output_bytes and
+                ledger['per_class'].get('TransferStaging',0) == 0 and ledger['fenced'] is False,
+                'all actual cached payload reservations, no transfer/root work')
+    require(final['reserved'] == final['reservations'] == 0 and final['fenced'] is False, 'actual last-drop zero')
+    inputs = read_json(args.reference_directory/'inputs.json')
+    reference = read_json(args.reference_directory/'reference.json')
+    comparison = result['comparison']
+    require(comparison['after_callers_drop'] is False and comparison['words'] == 1603680 and
+            comparison['different_words'] == 0 and [v['stage'] for v in comparison['stages']] == list(STAGES),
+            'complete actual stage ordering/counts')
+    total = 0
+    files = {'report.json','source-after.json','comparison-false.json','comparison-true.json'}
+    for record in comparison['stages']:
+        name = record['stage']
+        row = inputs['operands'][name] if name in INPUTS else reference['targets'][name]
+        expected = widened(raw_file(args.reference_directory,row),row['dtype'])
+        require(record['reference'] == row and record['different_words'] == 0 and
+                record['first_difference'] is None and record['words']*4 == len(expected) and
+                record['file'] == name+'.f32le' and read(directory/record['file'],4*mib,record) == expected,
+                'independent every signed actual/expected cached word: '+name)
+        files.add(record['file'])
+        total += record['words']
+    for dropped in (False,True):
+        record = read_json(directory/('comparison-'+str(dropped).lower()+'.json'))
+        require(record['after_callers_drop'] is dropped and record['words'] == 1603680 and
+                record['different_words'] == 0 and record['stages'] == comparison['stages'],
+                'same full archived words remain accessible after source caller drop')
+    require(total == 1603680 and {p.name for p in directory.iterdir()} == files and frozen() == freeze,
+            'all import/output words and native artifact tree accounted')
+    value = {'status':'PASS_ARCHIVED_H48_CACHED_GDN_PRIVATE_SLICE_ONLY',
+             'imported_words':804960,'independent_output_words':798720,'all_compared_words':total,
+             'completed_sha256':sha(args.output/'completed.json'),'freeze_manifest_sha256':freeze,
+             'ROOT_build_source_attestation_sha256':build_source,
+             'full_model_or_live_decay_or_112StateRoots_or_GatedNorm_or_GPU_qualified':False}
+    write(args.output/'validation.json',value)
+    print(json.dumps(value,sort_keys=True))
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/prepared-manifest.json
@@ -0,0 +1,172 @@
+{
+  "ROOT_focused_check_pin": {
+    "bytes": 1738,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-cpu-focused-7.json",
+    "sha256": "7d72b5c08e45316e5ab8ec29a8e605392abca1cc1e5fd835aeb97b69ef82a8aa"
+  },
+  "ROOT_recorded_adapter_and_test_patch_pins": [
+    {
+      "bytes": 988,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-coefficient-output-size-adapter-2.patch",
+      "sha256": "035c17b5db5559b473249561dd5cf99428ef2f7239778f77ee4a59fd3f1ece8f"
+    },
+    {
+      "bytes": 805,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-coefficient-test-array-style-3.patch",
+      "sha256": "30130924e17c8d8c08e0582c77dc2e8f46ec3e99342aa3a938036a0264588e3a"
+    },
+    {
+      "bytes": 1201,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-window-test-visibility-import-4.patch",
+      "sha256": "11e8fe4b19d5f18873a289db0dabd5a69a02bd2fa66ae8c3a10faf73e65e236f"
+    },
+    {
+      "bytes": 362,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-coefficient-test-array-style-5.patch",
+      "sha256": "10c3b4eec7d42cca192f9972c05d20e2ef88988c1f22a84d7d456395da015c6a"
+    },
+    {
+      "bytes": 3747,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-fixture-contract-reconciliation-6.patch",
+      "sha256": "8ad1b36e65f521621630de8018edea13857e74a75e7a2ee97e351e8f870788f7"
+    },
+    {
+      "bytes": 558,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-fixture-format-6.patch",
+      "sha256": "252c4e4d6bfffdc2267086341fdeeaf877be8489003fd68449599ce9dc987c19"
+    },
+    {
+      "bytes": 1527,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-tied-role-parameter-lifetime-fixture-7.patch",
+      "sha256": "d83a5e9564e7420c400422f69ae86b07fd9e7ad015046c91d6811be3c50d1632"
+    }
+  ],
+  "ROOT_recorded_source_exception_files": 8,
+  "ROOT_review_receipt_not_independent_compiler_input_proof": true,
+  "actual_trained_native_replay_or_new_source_free_tests_executed_by_author": false,
+  "allowed_harness_changes": [
+    "Cargo.toml exact absolute dependency root only",
+    "contracts.py only frozen/implementation source attestation dispatch"
+  ],
+  "byte_identical_original_harness_files": 11,
+  "case": "cached",
+  "conserved_original_harness_files": 13,
+  "expected_source_files": 521,
+  "files": [
+    {
+      "bytes": 819,
+      "file": "Cargo.toml",
+      "sha256": "fd33af8779c0e4eecd5703e07d43c1a01770a4ac509c0fae516d5af758e34ad0"
+    },
+    {
+      "bytes": 5603,
+      "file": "ISOLATED_ROOT.md",
+      "sha256": "fd85b27d997f32712fca96566537e644e2d13f975518064f38c6479da0f0c70b"
+    },
+    {
+      "bytes": 3981,
+      "file": "ROOT.md",
+      "sha256": "4be3506715b46adf49998cd6b3e7dfe2fd18707dd5e811077a57984701210470"
+    },
+    {
+      "bytes": 16627,
+      "file": "baseline-attention-slice-tests.rs",
+      "sha256": "57e3abc73a91cf7f661b64db2865c47c21e9eb9c2ca4f6483546d60643b2730e"
+    },
+    {
+      "bytes": 4350,
+      "file": "contracts.py",
+      "sha256": "92d773ae4a78eee636f3ffe456f1dfc454597aa9c02755ac0db4b30d979c270a"
+    },
+    {
+      "bytes": 10530,
+      "file": "export_reference.py",
+      "sha256": "4cdee4e696213ce8c0f93cdbcd3d567ad629f035a16e7d931a6ac626edb0bb6e"
+    },
+    {
+      "bytes": 29779,
+      "file": "factory.rs",
+      "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
+    },
+    {
+      "bytes": 17126,
+      "file": "main.rs",
+      "sha256": "9399b8e3f849b7ea9179f3458b36781a5f91a5028e68724513177e717d81b299"
+    },
+    {
+      "bytes": 3702,
+      "file": "owned_lifetime.py",
+      "sha256": "c10d034bd8ae0ca1dc065b68bd27f2bbe6f092a0cda0cbf8af738a363f0229b8"
+    },
+    {
+      "bytes": 5561,
+      "file": "owned_native.py",
+      "sha256": "e339e23ce5d1c5e1353fed1a8ec6b67735050e565d94b2af32eb81a9082980ab"
+    },
+    {
+      "bytes": 14762,
+      "file": "relocated_source.py",
+      "sha256": "88cd0514f9889ff185a7e839588818ea3c2f9da32d956d7eff128db2de6f63d0"
+    },
+    {
+      "bytes": 45365,
+      "file": "relocation.json",
+      "sha256": "578d3f1dd457a8adfd80e8ef9aaf7aa347dbdbc55e2febe627e31ea00711e98e"
+    },
+    {
+      "bytes": 3886,
+      "file": "run_replay.py",
+      "sha256": "0f535245d22ff96a59659d55f015caccd53026d496b83464c8f676349f9dcf11"
+    },
+    {
+      "bytes": 9260,
+      "file": "source-bindings.json",
+      "sha256": "55dfb8e385a53d01abbb8fab762f492022f7217aa8a2fc95e3dd42c0a5146a13"
+    },
+    {
+      "bytes": 7058,
+      "file": "source-pin.json",
+      "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+    },
+    {
+      "bytes": 4430,
+      "file": "test_contracts.py",
+      "sha256": "4db1acac757844fd9b7eaf41b9c3bfd24f0e603c3b87493c6835d11345fd9531"
+    },
+    {
+      "bytes": 12587,
+      "file": "test_relocation.py",
+      "sha256": "e11481fc76097bc212300c40103685c50facbaf88b99dd30a0b8f8acde424b7f"
+    },
+    {
+      "bytes": 7558,
+      "file": "validate_replay.py",
+      "sha256": "6f074e414d0d7e90804854b3d86dca39827179dd63acc4075da453ec43bbd141"
+    }
+  ],
+  "full_HF_model_StateRoot_exp_initial_GDN_GPU_or_performance_qualified": false,
+  "original_expected_children": 12,
+  "original_factory_source_pin_rawword_limits_and_lifetime_checks_unchanged": true,
+  "original_preparation": {
+    "bytes": 13272,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/prepared-manifest.json",
+    "sha256": "4ecc28732a41c800dccb0e3d6a3bafe88049f0c1d09f6c39f3110940453b8118"
+  },
+  "original_required_build_paths": 33,
+  "prepared_new_source_free_fixture_methods": 14,
+  "root_source_freeze": {
+    "bytes": 99248,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-7.json",
+    "sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615"
+  },
+  "schema": "zerogravity.original27b.isolated-root.replay.preparation.v2",
+  "source_only_author_checks": [
+    "AST9 Python per derivative",
+    "original13 live/frozen bytes authenticated",
+    "11 harness bytes identical; exact Cargo remap and all nonattestation contract AST nodes unchanged",
+    "all521 current ROOT small source bytes match final7",
+    "all8 exceptions reconstruct exact final ROOT bytes",
+    "original33/12 cached or25/9 gamma assertion cardinalities conserved"
+  ],
+  "status": "SOURCE_ONLY_ADDITIVE_RELOCATION_UNCOMPILED_UNEXECUTED_REQUIRES_ROOT_REVIEW"
+}

````

