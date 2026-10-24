# Draft Patches — volume 4

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/NEW_DELTA.patch

Original UTF-8 bytes: 262200; SHA256 `35b62aced4d64dd462ffe597ccc5a3f97fcadacad0ecf73272e3be6df46f452c`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/Cargo.toml
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
+zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts", features = ["cache"] }
+zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core" }
+zerogravity-cpu = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu", features = ["cache"] }
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
+name = "original27b-bf16-gamma-gated-rms-replay"
+path = "main.rs"
--- /dev/null
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/ISOLATED_ROOT.md
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
+  "status": "ROOT_REVIEWED_ISOLATED_BF16_GAMMA_GATED_RMS_SOURCE",
+  "prepared_manifest_sha256": "SHA256 of this directory's prepared-manifest.json",
+  "original_prepared_manifest_sha256": "f1c1f28c21616fb1057dad082a5b27a42a587bdac95ec9603d7f0c316c4eb8d4",
+  "root_source_freeze_sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615",
+  "workspace_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
+  "MAIN_merged": false,
+  "full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified": false,
+  "files": ["exact path/bytes/sha256 rows described below"]
+}
+```
+
+- `files` = all521 final ROOT freeze entries with workspace-root absolute paths, plus this directory's exact Cargo.toml/main.rs/factory.rs/source-pin.json/source-bindings.json; plus this directory's Cargo.lock if present. Exactly526 or527 unique rows, no MAIN aliases/missing/unlisted source. `relocated_source.workspace` returns the authenticated521 rows; `relocation.json.copy_map[*].relocated` supplies the local5. All expected-child assertions (including any repaired test child) must match relocation.json. The actual optional local lock must be pinned with the same2MiB BEFORE-read/fstat/regular-file policy.
+- Source-only relocation preserves independent input/result owner charges, cancellation/pressure/refusal probes, retained result/source lifetime, final reserved0 and all raw exact-bit limits. No allowance to treat caller attestation as independent runtime or numerical proof.
+
+Acceptance scope: SAME execute_gated_rms_bf16_gamma_slice; original initial11/cached353 typed finals12,288 words +retained inputs24,576 words; SAME helper fourteen stage checks61,632 words =2×(48+48+5×6144). BF16 gamma product rounding before gate is unchanged. Original25 required paths and9 expected children conserved. Diagnostic primitive parameter/window/stage owners retire on primitive() return before caller ledger/final0 checks. No initial GDN/exp/recurrence/full27B112StateRoots/session/GPU/performance eligibility.
--- /dev/null
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/ROOT.md
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/baseline-attention-slice-tests.rs
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/contracts.py
@@ -0,0 +1,143 @@
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
+    from relocated_source import frozen as checked_frozen
+    return checked_frozen(HERE, read_json, pin, require, sha)
+
+def implementation(path):
+    from relocated_source import implementation as checked_implementation
+    return checked_implementation(HERE, path, read_json, pin, require, sha)
+
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/factory.rs
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/main.rs
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/owned_lifetime.py
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/owned_native.py
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/prepare_reference.py
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/relocated_source.py
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/relocation.json
@@ -0,0 +1,643 @@
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
+  "case": "gamma",
+  "copy_map": [
+    {
+      "file": "Cargo.toml",
+      "original": {
+        "bytes": 815,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/Cargo.toml",
+        "sha256": "fbd9b555136916dcab678040a8ab8bf01abce6114f61172a9a3dcf51c46cffb5"
+      },
+      "policy": "ONLY_LOCAL_DEPENDENCY_ROOT_REMAP",
+      "relocated": {
+        "bytes": 979,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/Cargo.toml",
+        "sha256": "c184a8ab90f4a44fc0bb5d54e3b555109f43ce0173824c6c79fe110b0c5d74c9"
+      }
+    },
+    {
+      "file": "ROOT.md",
+      "original": {
+        "bytes": 5568,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/ROOT.md",
+        "sha256": "b2ff32e5a5eaaf8c82a3d2e04b368289afaa50fd9b740615a3377c83bff97eae"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 5568,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/ROOT.md",
+        "sha256": "b2ff32e5a5eaaf8c82a3d2e04b368289afaa50fd9b740615a3377c83bff97eae"
+      }
+    },
+    {
+      "file": "contracts.py",
+      "original": {
+        "bytes": 8191,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/contracts.py",
+        "sha256": "25a9b3ced16926746567a9d41998a9f35bf1f40598efb87d0d0a80b84bab032b"
+      },
+      "policy": "ONLY_SOURCE_ATTESTATION_CONTRACT_RELOCATION",
+      "relocated": {
+        "bytes": 6381,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/contracts.py",
+        "sha256": "89855e549998916c0a64de38ad0cc25940897e2abeaa0096d7e98466540ab720"
+      }
+    },
+    {
+      "file": "factory.rs",
+      "original": {
+        "bytes": 29779,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/factory.rs",
+        "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 29779,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/factory.rs",
+        "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
+      }
+    },
+    {
+      "file": "main.rs",
+      "original": {
+        "bytes": 25749,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/main.rs",
+        "sha256": "16dc98c8dd2234e2bb5afc94d511650515b938408e69e78fd70182289f01e92e"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 25749,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/main.rs",
+        "sha256": "16dc98c8dd2234e2bb5afc94d511650515b938408e69e78fd70182289f01e92e"
+      }
+    },
+    {
+      "file": "owned_lifetime.py",
+      "original": {
+        "bytes": 3702,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/owned_lifetime.py",
+        "sha256": "c10d034bd8ae0ca1dc065b68bd27f2bbe6f092a0cda0cbf8af738a363f0229b8"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 3702,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/owned_lifetime.py",
+        "sha256": "c10d034bd8ae0ca1dc065b68bd27f2bbe6f092a0cda0cbf8af738a363f0229b8"
+      }
+    },
+    {
+      "file": "owned_native.py",
+      "original": {
+        "bytes": 5575,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/owned_native.py",
+        "sha256": "c413b2a0e2db388275837e15edf6f46dcb48c29c4c7fb3466c002cbccba13fbc"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 5575,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/owned_native.py",
+        "sha256": "c413b2a0e2db388275837e15edf6f46dcb48c29c4c7fb3466c002cbccba13fbc"
+      }
+    },
+    {
+      "file": "prepare_reference.py",
+      "original": {
+        "bytes": 8478,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/prepare_reference.py",
+        "sha256": "e81f2448f2845cee9ffacc99118b673e9f0af922ee7f73c255c90cf5bfdbfbd2"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 8478,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/prepare_reference.py",
+        "sha256": "e81f2448f2845cee9ffacc99118b673e9f0af922ee7f73c255c90cf5bfdbfbd2"
+      }
+    },
+    {
+      "file": "run_replay.py",
+      "original": {
+        "bytes": 3889,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/run_replay.py",
+        "sha256": "9589b81d4a2c367db27d67ae0e0d50edfdd79a920efb95da6e11ca05c41993db"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 3889,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/run_replay.py",
+        "sha256": "9589b81d4a2c367db27d67ae0e0d50edfdd79a920efb95da6e11ca05c41993db"
+      }
+    },
+    {
+      "file": "source-bindings.json",
+      "original": {
+        "bytes": 41010,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/source-bindings.json",
+        "sha256": "9bf6a61664a9648e98352ffca16ac6fd5016d4b7e52207f9e9a13a8c654456af"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 41010,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/source-bindings.json",
+        "sha256": "9bf6a61664a9648e98352ffca16ac6fd5016d4b7e52207f9e9a13a8c654456af"
+      }
+    },
+    {
+      "file": "source-pin.json",
+      "original": {
+        "bytes": 7058,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/source-pin.json",
+        "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 7058,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/source-pin.json",
+        "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+      }
+    },
+    {
+      "file": "test_contracts.py",
+      "original": {
+        "bytes": 5204,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/test_contracts.py",
+        "sha256": "7acd536b049636fc26dca96c33fb799469fe567887e713e95379650f104934f6"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 5204,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/test_contracts.py",
+        "sha256": "7acd536b049636fc26dca96c33fb799469fe567887e713e95379650f104934f6"
+      }
+    },
+    {
+      "file": "validate_replay.py",
+      "original": {
+        "bytes": 12045,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/validate_replay.py",
+        "sha256": "8155faf71b877794b5a15aec40a9cab81cb5c37fdb588a47ab807ca2d7fd5def"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 12045,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/validate_replay.py",
+        "sha256": "8155faf71b877794b5a15aec40a9cab81cb5c37fdb588a47ab807ca2d7fd5def"
+      }
+    }
+  ],
+  "expected_child_map": [
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 7572,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program/gated_rms.rs",
+        "sha256": "7b8f589f7d2854de07d304d7f2153350851a3c33d9bae820309a761c7ae7c0f7"
+      },
+      "original": {
+        "bytes": 7572,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/accelerator/program/gated_rms.rs",
+        "sha256": "7b8f589f7d2854de07d304d7f2153350851a3c33d9bae820309a761c7ae7c0f7"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 2499,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma/tests.rs",
+        "sha256": "acddd75ad40bc7e0bce36e88cbaeb5aa5168570185e66e4a9e82140f6d162383"
+      },
+      "original": {
+        "bytes": 2499,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma/tests.rs",
+        "sha256": "acddd75ad40bc7e0bce36e88cbaeb5aa5168570185e66e4a9e82140f6d162383"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 5021,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma.rs",
+        "sha256": "ee45b5047bab51388b9fd9f3e60ed9446798fa919cfbd9c379d1c19772700611"
+      },
+      "original": {
+        "bytes": 5021,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma.rs",
+        "sha256": "ee45b5047bab51388b9fd9f3e60ed9446798fa919cfbd9c379d1c19772700611"
+      }
+    },
+    {
+      "classification": "test_only_same_H48_budget_valid_schedule_parameter_drop",
+      "isolated": {
+        "bytes": 13508,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs",
+        "sha256": "b6f7fcb6d8828051b6004b90f10c7b549303d94c9c1bf358ca701d3fda524b40"
+      },
+      "original": {
+        "bytes": 13482,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs",
+        "sha256": "1ee2170fceaebb0bbc0a6f48daf4d73687f6429eb48ca692a65c15540d798763"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 6519,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gdn_gated_rms_slice.rs",
+        "sha256": "c0ee66540859eda49f5f2ff17f1ba26adf28864445eca01ec5380fc10d7a7ad2"
+      },
+      "original": {
+        "bytes": 6519,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/gdn_gated_rms_slice.rs",
+        "sha256": "c0ee66540859eda49f5f2ff17f1ba26adf28864445eca01ec5380fc10d7a7ad2"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 5649,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math/gated_rms.rs",
+        "sha256": "0420e4e1c94fe94875f2dce2f10d0d7f4b6ea6627ca1ded9a53e492567509df7"
+      },
+      "original": {
+        "bytes": 5649,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/host_math/gated_rms.rs",
+        "sha256": "0420e4e1c94fe94875f2dce2f10d0d7f4b6ea6627ca1ded9a53e492567509df7"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 1665,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/original_binding/window/gated_rms.rs",
+        "sha256": "0d9f1b1e599cf7104e73817fa89067a2837b564a7af71bcd304acec94d2a6bf1"
+      },
+      "original": {
+        "bytes": 1665,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/original_binding/window/gated_rms.rs",
+        "sha256": "0d9f1b1e599cf7104e73817fa89067a2837b564a7af71bcd304acec94d2a6bf1"
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
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "isolated": {
+        "bytes": 8700,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/transcendentals.rs",
+        "sha256": "d73cd12c4f305b7b62a5a685d3fd975369f060f07570d38c0625b2584d974217"
+      },
+      "original": {
+        "bytes": 8700,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/original_hf/transcendentals.rs",
+        "sha256": "d73cd12c4f305b7b62a5a685d3fd975369f060f07570d38c0625b2584d974217"
+      }
+    }
+  ],
+  "false_scope_flag": "full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified",
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
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/baseline-attention-slice-tests.rs",
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
+    "bytes": 8191,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/contracts.py",
+    "sha256": "25a9b3ced16926746567a9d41998a9f35bf1f40598efb87d0d0a80b84bab032b"
+  },
+  "original_harness": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay",
+  "original_preparation": {
+    "bytes": 9671,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/prepared-manifest.json",
+    "sha256": "f1c1f28c21616fb1057dad082a5b27a42a587bdac95ec9603d7f0c316c4eb8d4"
+  },
+  "path_remap_is_not_MAIN_merge_or_independent_build_proof": true,
+  "relocated_harness": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2",
+  "required_build_paths": [
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/Cargo.toml",
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/factory.rs",
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/main.rs",
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/source-bindings.json",
+    "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/source-pin.json",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program/gated_rms.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/resource_arbiter.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/norm.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/norm/bf16_gamma/tests.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/reduction.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/transcendentals.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-gpu/src/resident_program.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-gpu/src/resident_program/batch.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/forward.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gdn_gated_rms_slice.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math/gated_rms.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/original_binding/window.rs",
+    "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/original_binding/window/gated_rms.rs"
+  ],
+  "required_isolated_status": "ROOT_REVIEWED_ISOLATED_BF16_GAMMA_GATED_RMS_SOURCE",
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/run_replay.py
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/source-bindings.json
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/source-pin.json
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/test_contracts.py
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/test_relocation.py
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
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/validate_replay.py
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
--- /dev/null
+++ b/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/prepared-manifest.json
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
+  "case": "gamma",
+  "conserved_original_harness_files": 13,
+  "expected_source_files": 521,
+  "files": [
+    {
+      "bytes": 979,
+      "file": "Cargo.toml",
+      "sha256": "c184a8ab90f4a44fc0bb5d54e3b555109f43ce0173824c6c79fe110b0c5d74c9"
+    },
+    {
+      "bytes": 5796,
+      "file": "ISOLATED_ROOT.md",
+      "sha256": "6ca8b67cd33a1d95d5985321beff1e32f9ebef9fbb249cb0b102a529a583816a"
+    },
+    {
+      "bytes": 5568,
+      "file": "ROOT.md",
+      "sha256": "b2ff32e5a5eaaf8c82a3d2e04b368289afaa50fd9b740615a3377c83bff97eae"
+    },
+    {
+      "bytes": 16627,
+      "file": "baseline-attention-slice-tests.rs",
+      "sha256": "57e3abc73a91cf7f661b64db2865c47c21e9eb9c2ca4f6483546d60643b2730e"
+    },
+    {
+      "bytes": 6381,
+      "file": "contracts.py",
+      "sha256": "89855e549998916c0a64de38ad0cc25940897e2abeaa0096d7e98466540ab720"
+    },
+    {
+      "bytes": 29779,
+      "file": "factory.rs",
+      "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
+    },
+    {
+      "bytes": 25749,
+      "file": "main.rs",
+      "sha256": "16dc98c8dd2234e2bb5afc94d511650515b938408e69e78fd70182289f01e92e"
+    },
+    {
+      "bytes": 3702,
+      "file": "owned_lifetime.py",
+      "sha256": "c10d034bd8ae0ca1dc065b68bd27f2bbe6f092a0cda0cbf8af738a363f0229b8"
+    },
+    {
+      "bytes": 5575,
+      "file": "owned_native.py",
+      "sha256": "c413b2a0e2db388275837e15edf6f46dcb48c29c4c7fb3466c002cbccba13fbc"
+    },
+    {
+      "bytes": 8478,
+      "file": "prepare_reference.py",
+      "sha256": "e81f2448f2845cee9ffacc99118b673e9f0af922ee7f73c255c90cf5bfdbfbd2"
+    },
+    {
+      "bytes": 14762,
+      "file": "relocated_source.py",
+      "sha256": "88cd0514f9889ff185a7e839588818ea3c2f9da32d956d7eff128db2de6f63d0"
+    },
+    {
+      "bytes": 41850,
+      "file": "relocation.json",
+      "sha256": "19f2b6a756b1f464ccf2f5e84b32a7976d7d647549e7f2015ced2423dfe76782"
+    },
+    {
+      "bytes": 3889,
+      "file": "run_replay.py",
+      "sha256": "9589b81d4a2c367db27d67ae0e0d50edfdd79a920efb95da6e11ca05c41993db"
+    },
+    {
+      "bytes": 41010,
+      "file": "source-bindings.json",
+      "sha256": "9bf6a61664a9648e98352ffca16ac6fd5016d4b7e52207f9e9a13a8c654456af"
+    },
+    {
+      "bytes": 7058,
+      "file": "source-pin.json",
+      "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+    },
+    {
+      "bytes": 5204,
+      "file": "test_contracts.py",
+      "sha256": "7acd536b049636fc26dca96c33fb799469fe567887e713e95379650f104934f6"
+    },
+    {
+      "bytes": 12587,
+      "file": "test_relocation.py",
+      "sha256": "e11481fc76097bc212300c40103685c50facbaf88b99dd30a0b8f8acde424b7f"
+    },
+    {
+      "bytes": 12045,
+      "file": "validate_replay.py",
+      "sha256": "8155faf71b877794b5a15aec40a9cab81cb5c37fdb588a47ab807ca2d7fd5def"
+    }
+  ],
+  "full_HF_model_StateRoot_exp_initial_GDN_GPU_or_performance_qualified": false,
+  "original_expected_children": 9,
+  "original_factory_source_pin_rawword_limits_and_lifetime_checks_unchanged": true,
+  "original_preparation": {
+    "bytes": 9671,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/prepared-manifest.json",
+    "sha256": "f1c1f28c21616fb1057dad082a5b27a42a587bdac95ec9603d7f0c316c4eb8d4"
+  },
+  "original_required_build_paths": 25,
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

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/test-style-v2/TEST_STYLE_FOLLOWUP.patch

Original UTF-8 bytes: 644; SHA256 `86384ec02d7b8bfd3b36b328a063779b47a746eca3c3cfefddfc9f4acb7dce51`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs
@@ -282,7 +282,6 @@
     assert_eq!(reserved(&host), before + input_need);
     drop(core);
     drop(gate);
-    drop(math);
     assert_eq!(reserved(&host), before);
     let result = ops
         .execute_gated_rms_bf16_gamma_slice(0, input, None)
@@ -383,7 +382,6 @@
     assert!(math.import_gated_rms_input(input(&ops, &raw)).is_err());
     assert!(!provider.0.load(Ordering::Acquire));
     assert_eq!(reserved(&host), before);
-    drop(math);
     drop(ops);
     assert_eq!(reserved(&host), 0);
 }

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/GPU_REFUSAL.patch

Original UTF-8 bytes: 1917; SHA256 `0a49ceafd9426512204d13bf4686c7a2f50ea01a22c80e3e7f7a56c8a26a6897`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -349,6 +349,8 @@
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
+                        | PlanOperation::TypedCachedGdnCore { .. }
+                        | PlanOperation::TypedCachedGdnBf16Return { .. }
                 )
             })
         {
@@ -1431,7 +1433,9 @@
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
-                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedCausalConvSilu { .. }
+                | PlanOperation::TypedCachedGdnCore { .. }
+                | PlanOperation::TypedCachedGdnBf16Return { .. } => Err(Error::Unsupported(
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
+                | PlanOperation::TypedCachedGdnCore { .. }
+                | PlanOperation::TypedCachedGdnBf16Return { .. } => Err(Error::Unsupported(
                     "resident batch has no qualified typed lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {

````

