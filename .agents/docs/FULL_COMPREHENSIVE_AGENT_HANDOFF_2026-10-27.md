# ZeroGravity full comprehensive handoff — 27 Oct 2026

Author: @pvl

This is the current non-Fabric handoff and the entry point for the next agent.
It does not replace the detailed plan ledgers, specifications, historical
failures, or evidence named below. It reconciles them and records the current
implementation boundary.

## 0. Mission

Continue correctness-first ZeroGravity as one source-authentic Rust inference
system:

- one mathematical traversal and shared `ForwardPlan`;
- one physical ResourceArbiter;
- one queue/flight/completion/quarantine path;
- independently owned `StateRoot`s;
- immutable authenticated model/source ownership;
- source-derived runtime planning, never captured testcase dispatch;
- independently qualified CUDA, CPU, HIP/ROCm, and Vulkan implementations;
- model correctness before serving-performance claims.

Immediate product objective:

1. Finish ordinary source-derived Qwen3.5 `.8B` and `2B` CUDA execution.
2. Requalify initial/cached correctness, physical ownership, faults and shutdown.
3. Run one serial matched vLLM campaign: canonical, ragged, queue scaling and
   long contexts.
4. Continue genuine long context, Qwen3.8-27B, Gemma4 and SSD execution under
   their own source/numerical/physical contracts.

Fabric vNext is explicitly outside this mission. Do not read, stage, edit, merge,
or report the unrelated untracked `FABRIC_VNEXT*` files or `fabric-vnext*`
plans.

## 1. Mandatory startup procedure

Run from `/home/pvl/spaces/abird/src/zerogravity`:

```bash
export PYTHONDONTWRITEBYTECODE=1
git status --short --branch
git log --show-signature --oneline -20
git worktree list --porcelain
nvidia-smi
```

Then read, in order:

1. `AGENTS.md` completely.
2. `.agents/plans/README.md` completely.
3. `.agents/plans/TODO.md`.
4. This handoff completely.
5. `.agents/docs/handoff-2026-10-26/USER_DECISIONS.md`.
6. `.agents/docs/handoff-2026-10-26/DESIGN_AND_DECISIONS.md`.
7. `.agents/docs/handoff-2026-10-26/EVIDENCE_AND_CONTINUATION.md`.
8. `.agents/docs/handoff-2026-10-26/OPEN_WORK.md`.
9. `.agents/docs/handoff-2026-10-26/SOURCE_CATALOG.md`.
10. `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`.
11. `.agents/plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md`.
12. `.agents/docs/REVIEW_ORDINARY_CANDIDATE_29B491D_2026-10-27.md` — the interrupted
    review of the pending ordinary CUDA branch, with the exact resume checklist.
13. Current TODOs:
    - `.agents/plans/numerics-and-context/TODO.md`
    - `.agents/plans/model-execution/TODO.md`
    - `.agents/plans/serving-and-speculation/TODO.md`
    - `.agents/plans/cpu-gpu-optimization/TODO.md`
    - `.agents/plans/memory-and-streaming/TODO.md`

For the complete history before 26 Oct, use these indexes rather than guessing
from dated prose:

- `.agents/docs/HANDOFF_ALL_COMMIT_LINEAGE_2026-10-25.md`
- `.agents/docs/HANDOFF_ALL_DESIGN_AND_SPEC_SOURCES_2026-10-25.md`
- `.agents/docs/HANDOFF_ALL_OPEN_PLAN_ITEMS_2026-10-25.md`
- `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md`
- `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md`
- `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-20.md`
- `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25.md`
- `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md`
- `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-NIGHT.md`
- `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-26.md`
- `.agents/docs/handoff-2026-10-20/HISTORY_DESIGN_AND_COVERAGE_AUDIT.md`
- `.agents/docs/handoff-2026-10-20/CONTEXT_CATALOG.md`
- `.agents/docs/handoff-2026-10-20/CURRENT_TAKEOVER.md`

Those files and the plan TODOs retain individual historical stages, failures,
physical runs and source identities. Do not flatten failed or superseded scopes
into a later pass.

## 2. Non-negotiable engineering rules

### 2.1 Execution authority

- Preserve one engine. Do not add another token loop, allocator, state issuer,
  root publisher, CUDA queue, completion service, or runtime.
- `ForwardPlan` describes shared mathematics. Runtime invocation carries rows,
  base/frontier and output selection. `StateRoot` and completion authority remain
  outside the immutable plan.
- All physical work enters the existing admitted queue/flight owner and becomes
  publishable only after actual completion and status validation.
- Unknown work stays fenced/quarantined. Process exit is not device completion.

### 2.2 Qualification boundaries

Keep these claims separate:

1. authenticated source identity;
2. mathematical/numerical contract;
3. physical implementation and ownership;
4. whole-model logits/state correctness;
5. serving behavior;
6. determinism;
7. performance.

A source hash, reverse specialization, CUDA compilation, reflected ABI, fixture
comparison, cleanup pass, or final token does not establish the other claims.

### 2.3 Captures are never production authority

The following may remain test/qualification fixtures but may not select ordinary
production execution:

- prompt IDs or prompt-length whitelists;
- testcase/call/node ordinals;
- packet-content variants or compatibility maps;
- captured recipe/catalogue hashes;
- nearest captured case;
- finite vendor image inventories renamed as capabilities.

Do not add 48–50 production capture arms or qualification fallback. Ordinary
unsupported Qwen CUDA must refuse before device/source/private allocation.

### 2.4 Ownership and resources

- Preserve immutable source owners through every plan, invocation, buffer,
  readback and completion reader that uses them.
- Charge real host/device backing once; distinguish metadata, aliases and copies.
- Preserve queue, flight, handle, source, state, readback and quarantine owners
  until actual retirement.
- Maintain independent roots for independent requests even when immutable source
  or prepared code is shared.
- Keep exact source/model/compiler/image/ABI identities and resource asymmetries
  in benchmark reports.

### 2.5 Repository safety

- Root `README.md` is user-authored: do not change its bytes.
- Preserve unrelated user work, especially the untracked Fabric vNext files.
- Preserve the 11-file ignored extraction overlay at
  `/home/pvl/spaces/abird/src/.zerogravity-worktrees/qwen35-prompt-reference-20261026/tools/cuda-gdn-recipes/`.
- Do not download models/SDKs, change device permissions/recovery, launch the held
  Renoir/gfx90c AMD path, push, tag, publish, deploy, rewrite history, or remove
  another owner's work without explicit authorization.
- Commits are signed. Never disable signing. This shell may need:
  `SSH_AUTH_SOCK=/run/user/1000/gcr/.ssh`.
- Use scoped commits and inspect staged paths before committing.

## 3. Initial design and source of truth

The architecture began as a typed, source-authenticated execution fabric rather
than a model-specific script. Current implementation must continue to follow:

- engine/protocol: `.agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`
- minimal spine: `.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`
- invariants/failures: `.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md`
- async command IR: `.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md`
- model semantics: `.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md`
- source/loading formats: `.agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md`
- state/cache/transactions: `.agents/docs/STATE_CACHE_AND_TRANSACTIONS.md`
- memory/residency/IO: `.agents/docs/MEMORY_RESIDENCY_AND_IO.md`
- scheduler/batching/speculation:
  `.agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md`
- serving/runtime UX: `.agents/docs/RUNTIME_SERVING_AND_UX.md`
- compiler/IR: `.agents/docs/SEMANTIC_IR_AND_COMPILER.md`
- device kernels: `.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md`
- workspace rules: `.agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md`
- security/reliability: `.agents/docs/SECURITY_AND_RELIABILITY.md`
- source geometry/naming:
  `.agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md`
- shared execution architecture:
  `.agents/plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md`
- immutable sharded sources:
  `.agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md`

The active acceptance policy is newer and stricter than some early roadmap prose:
actual live source, tests and measured results outrank dated plans.

## 4. Chronology and completed foundations

### 4.1 Baseline and acquisition

- Accepted baseline, plan split and historical cleanup landed by 18 Oct.
- XDG-owned config/data/cache/state use `$XDG_*_HOME/abird/zerogravity`, falling
  back to `HOME`; no alternate legacy root.
- Typed model acquisition, immutable source manifests, source identities,
  sharded index/header validation and bounded original-source windows exist.
- Qwen3.8-27B's 55.6 GB source and indexes are authenticated. Full trained
  inference is not qualified.
- Verified source-only SSD extents and original-index tensor windows exist;
  streamed trained execution, eviction and resident-vs-SSD parity remain open.

### 4.2 Shared engine and ownership

- One Rust mathematical engine, typed plans/operators, physical arbiter,
  queue/flight/completion/quarantine, independent roots, cancellation and stale
  publication controls exist.
- Bounded CPU/OpenAI serving, source ownership, shutdown and process/container
  lifecycle evidence exist.
- Finite CUDA Qwen execution, B4 cohorts, independent roots, faults and
  retirement exist only in their recorded qualification scope.
- HIP/Vulkan transport foundations and model-specific candidate routes exist;
  they do not grant whole-model backend parity.

### 4.3 Qwen short correctness and serving history

- Historical finite `.8B`/`2B` Qwen qualification covers prompts
  47/57/62/67/72, natural feedback, all 24 boundaries, all 42 state domains,
  full 248,320 logits, roots, cancellation, faults and cleanup in the recorded
  finite profile.
- Existing finite OpenAI/API serving reached active 8 with B4 native cohorts.
- C16 means 16 clients queued over active at most 8, not a B16 kernel.
- Preserve historical `.8B`/`2B` API, lifecycle and performance evidence in
  the owning TODOs and `cpu-gpu-optimization/SHORT_HF_RESULTS.md`.
- Preserve `FAIL101` case 5 and the six N1/B4 QKVZ BF16 differences at frontier
  47. They are not waived by later width-specific successes.

### 4.4 Production/qualification separation

The 26 Oct audit found that ordinary CUDA still depended on five prompt lengths,
recipe hashes, length-specific GDN source and duplicate native packet inventories.
The following corrections landed before this handoff:

- semantic prompt validity separated from backend qualification;
- finite Qwen CUDA isolated behind `qwen35-cuda-qualification`;
- `cuda-native` production transport separated from
  `cuda-native-qualification`;
- ordinary Qwen refusal moved before source/device/private allocation;
- route targets restricted to exact `qwen3.5-0.8b` and `qwen3.5-2b`;
- rotary preparation split into fault-free production and qualification routes;
- ordinary serving no longer receives finite capture authority transitively.

Rejected native-plan candidates `15d2da4`, `fa99e760`, and `e86b8d3f` remain
outside main because they preserved prompt/call/node dependence, decode
collisions, multiplicity loss, or permissive ABIs.

### 4.5 Parameterized GDN

- All 17 unique GDN instruction bodies are parameterized, not only convolution.
- Packed scalar ABIs, exact `F32Bits`, checked grids/extents, five-slot cache
  validation, aliases and overflow checks exist.
- Six backend-owned configurations are explicit.
- All 96 historical tuples remain reverse-specialization fixtures.
- Executable selection consumes sealed `GdnKernelConfig`; recipe/catalogue hashes
  no longer authorize execution.
- Physical SM86 NVRTC compile/load/reflection passed for all 18 entries, including
  pointer and packed-scalar layouts. This was ABI/source evidence, not numerical
  qualification.
- Learned parameter ownership uses text/non-text `ParameterRoute`, not a
  per-call GDN kernel identity.

### 4.6 Fresh `.8B` prefill 47–50

Qualification-only fresh support was added without broadening ordinary serving:

- `QualificationMode::FreshPrefill` and `Text08FreshPrefill` remain diagnostic.
- Fresh references resolve through
  `prefill-{47,48,49,50}-discovery/observed`.
- Each cell owns its prompt in its own `observed/result.json`.
- Rows 48–50 are admitted only for `.8B`/1024-column selected-head and embedding
  paths; `2B`/H2048 remains closed.
- Fixture host baseline is accounted separately from exact 512 MiB backend
  device and 384 MiB backend host ownership.

The RTX 3060 physical test passed genuine `.8B` prefill rows 47, 48, 49 and 50:
all 24 boundaries, all 42 post-state checks, consumed priors, complete logits,
natural greedy ID `248068`, root advancement, four independent sessions,
retirement and resource conservation.

Exact command:

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
python3 scripts/native-gpu.py -- \
cargo test --release --locked --offline -p zerogravity \
  --no-default-features --features qwen35-cuda-qualification \
  --test qwen35_cuda \
  fresh_prefill_state_logits_root_and_cleanup_agreement \
  -- --ignored --exact --nocapture --test-threads=1
```

Result marker:
`QWEN35_CUDA_FRESH_PREFILL_PASS prompts=47,48,49,50 calls=4 decode=false`.
The GPU returned to 41 MiB idle with no compute process.

This does not qualify cached 47→48→49→50 on current source, ordinary serving,
long contexts, or performance.

## 5. Current main implementation checkpoint

At handoff preparation, main was based on `e99c76e` plus these signed units:

- `ad82feb` — call-free source operator plans.
- `f7ff4f9` — retain authenticated ordinary source owner; legacy captured and
  resident/batch consumers explicitly refuse the new template.
- `1921db1` — expand bounded serving comparison matrices.
- `1d278ed` — retain complete matrix-cell reports and refuse unsafe output paths.
- `fe228ab` — ordinary chunked GDN runtime API.
- `52bfb51` — complete GDN transport/state schedule and close launch authority.

The seven preceding fresh-prefill commits are:

- `eb38dd5`
- `ba7fb03`
- `fadf2b1`
- `ae9a843`
- `532a35c`
- `54d7ea7`
- `e99c76e`

Run `git log --show-signature origin/main..HEAD` for the final documentation and
any subsequently integrated ordinary-primitive commits. All accepted commits
must have good signatures.

### 5.0 Interrupted work at handoff time

The final independent review of the ordinary non-GDN CUDA candidate **did not
finish**. The reviewer agent was cut off by a model usage limit
(`Codex error: The usage limit has been reached`) after the layout/stride sweep,
the sigmoid/multiply split and part of the `kernel.rs` ABI read. It never opened
`crates/zerogravity/src/qwen35_cuda/ordinary.rs`, never audited the `kernels.cu`
status sites, and issued no verdict. Nothing was integrated.

Signed branch awaiting revalidation:
`pi-web-agent-8026e968-eb45-46fd-a2a1-d669a232fe6b` → `29b491d`

- `ec91ad1` Add dynamic CUDA ordinary text primitives
- `3a91a97` fix(cuda): constrain ordinary text candidate
- `29b491d` fix(cuda): close ordinary Qwen review blockers
- base `52bfb51`; 9 files, +2852/−5

`main` does not contain `ordinary_text` or `qwen35_cuda/ordinary.rs`. That worktree
was removed; re-materialize read-only with
`git worktree add /tmp/rev-29b491d 29b491d` or `git show 29b491d:<path>`.

Full checklist, verified items and open items:
`.agents/docs/REVIEW_ORDINARY_CANDIDATE_29B491D_2026-10-27.md`.

Conditional position: the candidate is acceptable **as unselected primitives with
no backend or launch authority** once these close — the RMS reduction-overflow
status gap, the `SiluUp` scalar-word-order check, the `byte_size()` `#[cfg]` and
alias-rule confirmations, the untested `qwen35-cuda`/default/`--all-features`
builds, and a truthful refresh of the ordinary refusal text. No numerical,
backend or performance claim follows from integration.

Three coordinator-verified findings to carry forward:

1. `zg_ot_rms_v1`, `zg_ot_residual_rms_v1` and the head-RMS macro validate inputs,
   `gamma`, `inverse` and `y` but not the reduction result. Finite inputs whose
   squares or accumulated sum overflow to `+inf` give `inverse = 0`, `y = 0`,
   status 0 — a silent accepted zero row.
2. The BF16-round-then-multiply-in-FP32 pattern is exact, not a deviation: the
   product of two BF16 values is exactly representable in FP32, so a single BF16
   store equals a BF16 multiply. Do not record it as a tolerance.
3. `main`'s refusal text
   (`"Qwen CUDA has no ordinary dynamic GDN implementation…"`) is stale after the
   parameterized dynamic GDN runtime landed, and the branch's replacement text
   wrongly calls dynamic GDN incomplete. The refusal must name the actual gap:
   source-bound non-GDN projection/attention lowering plus physical
   queue/flight/completion/`StateRoot` ownership.

### 5.1 Call-free ordinary plan

`SourceOperatorTemplate` and runtime invocation binding now separate shared plan
structure from call rows/base/frontier/output selection. Qwen plans provide:

- exact `.8B` and `2B` source profiles;
- 24 fixed layers in the genuine 3 linear-attention + 1 full-attention pattern;
- exactly 42 state domains: 18 convolution, 18 recurrent, 6 KV;
- contexts 512 through 32,768;
- chunks up to 64 rows;
- last-active-row logits and explicit emit suppression;
- no stored `SourceCall`, prompt/call/node/root/completion authority;
- retained authenticated `OrdinaryCudaSourceSeal`, source owner and workspace.

Legacy consumers refuse `SourceOperatorTemplate` rather than silently lowering
it through captured paths. A new ordinary physical owner/lowerer is still needed.

### 5.2 Call-free GDN runtime

The crate-private GDN runtime now provides explicit configuration, arbitrary
prefill decomposition and cached decode. It preserves the existing
parameterized implementations and has no public second launcher.

- Non-null cache slots are 1–4; slot 0 remains a CUDA sentinel and is refused.
- Cached schedule:
  `CopyQkvPrefix → PackZ → Conv → Decode → Gated`.
- Multi-row prefill schedule:
  `PackA → PackB → PackZ → Conv → Prep → Cumsum → Kkt → ZeroInverse → Solve → Recompute → Zero/GatherInitial → State → ScatterFinal → Output → Gated`.
- Initial, final and cache recurrent views have explicit ownership and alias
  guards; cached decode remains explicitly in-place.
- Launch binding/preparation remains crate-private and caller-flight-owned; it
  cannot publish a root.

No current-source numerical or GPU launch qualification has been run for this
new call-free schedule.

### 5.3 Benchmark tooling

`tools/serve-matrix.py` serializes canonical/ragged C1/C4/C8/C16/C32/C64 and
explicit context cells through `serve-compare.py`. It keeps each full report as
an atomic sibling JSON, indexes it from the aggregate, stops on failure, and
refuses output/input aliases or existing outputs unless `--overwrite`.

`serve-compare.py` accepts concurrency through 64 and up to 1,024 requests,
preserving balanced ABBA/BAAB execution, lifecycle ownership and source/tokenizer
attestation. `serve-bench.py` supports bounded workload envelopes.

The tool does not create execution support. Long-context prompt usage and
provider-side prefill timing still need explicit retention/binding before the
long campaign.

### 5.4 Production queue scaling gate

The pre-existing production CPU Router test was run on current source with the
approved tokenizer:

```bash
ZG_APPROVED_TOKENIZER=/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17/tokenizer.json \
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
cargo test --release --locked --offline -p zerogravity-cli \
  --no-default-features --features serve \
  'serve::engine::tests::serving_population::real_serving_populations_match_actual_router_state' \
  -- --ignored --exact --nocapture --test-threads=1
```

Result: 1 passed in 26.34 s after the release build. It exercised the real
Router, production workers, two workers × four active sessions, actual Qwen
source/tokenizer/model execution, offered populations 16/32/64, unique receipts,
real decoded output, active/peak/queue health, request/source/queue/worker host
charges, cleanup, and real queue-cap refusal/recovery.

This closes the source-free production queue/admission prerequisite for C32/C64.
It does not qualify the missing ordinary CUDA model route or authorize those
benchmark cells by itself. The synthetic candidate commit `68cbef5` was not
integrated because it reimplemented active scheduling in a test.

## 6. Current blockers and honest capability boundary

### 6.1 Ordinary CUDA is still unavailable

Ordinary Qwen CUDA must continue to refuse early. The call-free plan and the
complete parameterized GDN schedule exist, but source-derived non-GDN lowering,
physical ownership and whole-model integration are not accepted.

History of the non-GDN candidate, so it is not repeated:

1. `2f00b57` (first attempt) and corrections `0b2b8e8`/`8636a55` were rejected.
   Review found dense ABIs that could not bind packed/strided multi-row Q/K, KV or
   gate/up layouts; a fused sigmoid×multiply with the wrong rounding contract;
   incomplete nonfinite status; no adapter into `OrdinaryQwenPlan` and the
   flight/completion owner; and a 262,144 context claim contradicting the plan's
   32,768 limit.
2. The successor series `ec91ad1`/`3a91a97`/`29b491d` on branch
   `pi-web-agent-8026e968…` addresses those points and adds an explicit
   "unintegrable" refusal, but its independent review was cut short by a usage
   limit and it is **not integrated**. See §5.0 and
   `.agents/docs/REVIEW_ORDINARY_CANDIDATE_29B491D_2026-10-27.md`.

Rules for whoever resumes:

- Do not integrate either candidate unchanged, and do not revive `2f00b57`.
- Integration is allowed only as unselected primitives with no backend or launch
  authority, after the §5.0 conditions close.
- Private unselected online-FP32 attention is not a Qwen qualification and must not
  be selectable as ordinary authority.
- The final accepted tree must either contain a genuinely integrated
  plan → invocation → physical owner → completion/`StateRoot` path, or contain no
  ordinary backend and keep the honest early refusal. A half-wired backend is not
  acceptable.

### 6.2 Current-source requalification remains open

Before any timing:

- cached 47→48→49→50 continuation on the assembled source;
- affected cancellation, fault, retirement and shutdown scopes;
- ordinary source-derived full `.8B` and `2B` initial/cached paths;
- complete physical/resource conservation;
- final assembled isolated `scripts/check.sh`.

### 6.3 Long contexts

No ordinary long-context CUDA route is qualified. Required order:

1. `.8B`: 512/1K/2K/4K/8K correctness and physical fit.
2. `.8B`: 16K/32K with chunking, continuation, replay, faults and state history.
3. `2B`: admit only exact fitted cells. Current F32/two-root 32K does not fit the
   6 GiB target.
4. Independently qualify original-HF CPU, HIP/ROCm and Vulkan cells.

### 6.4 Performance

No benchmark from the 27 Oct campaign has run. Existing values are historical,
single-wave and preliminary. Do not claim broad parity, confidence, p99,
equal-hard-cap, latest-vLLM, long-context performance, or C32/C64 scaling.

The ignored fresh-prefill harness is qualification-only. It is not a serving
benchmark.

## 7. Exact benchmark campaign

Read
`.agents/plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md`.
Summary:

1. `.8B` and `2B`, canonical 47 tokens, eight natural output tokens,
   C1/C4/C8/C16.
2. Ragged 47/57/62/67/72 at the same bands.
3. C32/C64 only after the model route joins the already-passed production queue
   gate; report actual active at most 8/B4 behavior.
4. Contexts 512/1K/2K/4K/8K then 16K/32K, beginning C1 and reporting TTFT/prefill
   separately from decode ITL/rate.
5. Two balanced repetitions, ABBA/BAAB; separate cold-process and warmed-process
   campaigns; one GPU owner and one cell at a time.

Pinned historical rates are in the campaign plan and
`cpu-gpu-optimization/SHORT_HF_RESULTS.md`. Preserve the `2B` C16 loss.

## 8. Models, sources and retained artifacts

### 8.1 Qwen `.8B`

- Revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Root:
  `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`.
- Approved tokenizer SHA256:
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.
- Fresh native bundle:
  `.agents/runs/numerics-and-context-sequence-root-20261026/root/current-native-bundles/0.8b`.
- Fresh references:
  `.agents/runs/numerics-and-context-sequence-root-20261026/root/prefill-{47,48,49,50}-discovery/observed`.

### 8.2 Qwen `2B`

- Revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Exact source/bundle pins:
  `.agents/runs/numerics-and-context-sequence-root-20261026/root/bundle-2b-pins.json`.
- Keep `.8B` and `2B` shapes, cache keys, source ownership and fit claims distinct.

### 8.3 vLLM

- Version: `0.31.0`.
- Commit: `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Retained package:
  `.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/`.
- Retained results:
  - `.agents/plans/cpu-gpu-optimization/SHORT_HF_RESULTS.md`
  - `.agents/runs/cpu-gpu-optimization-cohort-b4-root-20261023/`
  - `.agents/runs/b4-takeover-codex-20261025/matched-c16-v6/`

### 8.4 Parameter/capture evidence

- Parameter proof:
  `.agents/runs/numerics-and-context-sequence-root-20261026/profile-implementation/gdn-extension/parameter-proof/`.
- Native 48–50 candidate evidence:
  `.agents/runs/numerics-and-context-sequence-root-20261026/root/native-prefill48-50-candidate-v2/`.
- These are qualification evidence, not ordinary execution authority.

## 9. Backend and broader project status

### CUDA

- Finite qualification paths are mature and physically exercised.
- Parameterized GDN source/ABI is established.
- Ordinary call-free plan and GDN runtime are integrated.
- Ordinary complete non-GDN lowering/owner and current-source model qualification
  remain open.

### CPU

- Genuine CPU model/router foundations, source-derived reference math and real
  production queue tests exist.
- Original-HF whole-model `.8B`/`2B` long-context qualification remains open.

### HIP/ROCm and Vulkan

- Shared transport/ownership and finite candidate checks exist.
- They must independently validate original source, complete logits/state,
  cancellation, faults and physical retirement. CUDA evidence does not transfer.
- Local Renoir/gfx90c remains on launch/recovery hold.

### Qwen3.8-27B

- 55.6 GB original source/index/hashes and bounded source windows are authenticated.
- Some source-real operator/state slices and full64/112-domain historical work are
  recorded in the model plan.
- Full trained inference, serving, performance and SSD-streamed parity remain
  unqualified.

### Gemma4

- Separate source/model/portable candidate work exists and has model-specific
  physical records.
- Complete target variants, original learned full-model parity, serving and
  performance remain separate open gates.

### Pipelines and routing

- Typed DAG/configuration, CLI/device routing and shared execution scaffolding
  exist.
- No executable qualified multi-model/media pipeline follows from that
  scaffolding. Keep this work independent of the Qwen benchmark critical path.

## 10. Preserved failures and rejected interpretations

Never relabel or silently omit:

- case 5 `FAIL101`;
- six N1/B4 QKVZ BF16 differences at frontier 47;
- incomplete/failed capture parents;
- plain/observed original configuration divergences documented in the numerical
  plan;
- historical 2B C16 performance loss;
- earlier container status 137 with unknown fatal/OOM cause;
- failed or interrupted controller scopes;
- source proof without numerical proof;
- process cleanup without physical completion;
- finite capture support as ordinary capability.

## 11. Validation cadence

- During implementation, use narrow compile/tests only to resolve concrete
  blockers.
- Batch related implementation first.
- Run physical model/GPU tests serially under `scripts/native-gpu.py`.
- Inspect each semantic and cleanup result; later cleanup does not erase earlier
  failure.
- Run the complete gate last:

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
python3 scripts/native-gpu.py -- bash scripts/check.sh
```

Do not treat compilation, host tests, an ignored compile/reflection diagnostic,
or a subagent report as independent acceptance. The coordinator must inspect
source and results.

When copying worktree source, give it fresh mtimes so Cargo cannot reuse a stale
library.

## 12. Next implementation sequence

Resume at the interruption, not before it.

0. **Revalidate from the interruption point.** Re-read
   `.agents/docs/REVIEW_ORDINARY_CANDIDATE_29B491D_2026-10-27.md`, re-materialize
   branch tip `29b491d` read-only, and independently re-derive the reviewer's
   verified items instead of trusting them. Then close, in the candidate tree:
   the RMS/head-RMS reduction-overflow status gap; the `SiluUp` scalar-word order;
   the `StaticScalarSchema::byte_size()` `#[cfg]` and alias-rule confirmations;
   and run plain `qwen35-cuda`, default and `--all-features` builds. Refresh the
   ordinary refusal wording. Integrate only as unselected primitives, or explicitly
   reject the series and record why.
1. Finish the ordinary non-GDN CUDA path: exact packed/strided layouts, separate
   rounded Sigmoid/Multiply, complete status, and a real lowering into the existing
   physical owner. No half-wired backend.
2. Connect `OrdinaryQwenPlan` + runtime invocation + source seal to the ordinary
   owner and the complete call-free GDN schedule.
3. Add source-derived tensor bindings, workspace/inventory/image/vendor permits
   and exact resource admission for `.8B` and `2B`; no generated packet tables.
4. Run host/source/ABI checks and physical SM86 compile/reflection.
5. Physically qualify `.8B` initial and cached 47→50, then the historical ragged
   set; faults, cancellation, retirement and shutdown included.
6. Qualify `2B` under its own geometry and fit.
7. Qualify `.8B` context ladder, then only fitted `2B` cells.
8. Run final assembled gate.
9. Prepare exact workload/lifecycle inputs and execute the serial benchmark
   campaign.
10. Diagnose `2B` C16, then continue batching/paged state/prefix COW/speculation,
    Qwen27B SSD execution and Gemma.

Do not start timings at step 4 or after only `.8B` qualification. The requested
campaign is one combined serial run after all admitted cells are ready.

## 13. Worktree and branch hygiene at handoff

Expected persistent worktrees:

1. Main:
   `/home/pvl/spaces/abird/src/zerogravity`
2. Preserved extraction overlay:
   `/home/pvl/spaces/abird/src/.zerogravity-worktrees/qwen35-prompt-reference-20261026`

Any `pi-web-agent-*` worktree should exist only while its agent is active. Before
handoff, integrate accepted signed commits, remove clean obsolete worktrees and
prune stale metadata. Never remove the extraction overlay.

At this handoff the registered worktrees are exactly those two, and
`git worktree prune` is clean. However
`/home/pvl/spaces/abird/src/zerogravity-worktrees/` still holds five **unregistered
full source copies with no `.git`** from earlier owners:

| directory | approx size | recorded subject |
|---|---:|---|
| `qwen27b-attention-tail-20261022` | 208G | Qwen27B attention tail |
| `root-lowest-f32-20261020` | 137G | lowest F32 repair |
| `gemma4-ple-pi-20261024` | 16G | Gemma4 PLE |
| `qwen27b-natural-pi-20261024` | 9.5G | Qwen27B natural comparison |
| `vulkan-gemma-pi-20261024` | 34M | Vulkan Gemma candidate |

They are no longer git worktrees, so their commits either already exist in the
repository or are only recoverable from the copies themselves. Treat them as
potentially unique unintegrated work: do **not** delete them without explicit user
confirmation, and do not modify files inside them.

Expected main status after the handoff commit:

- tracked tree clean;
- only unrelated untracked Fabric vNext files/plans visible;
- no push performed.

## 14. Completion definitions

### Ordinary Qwen CUDA complete

- source-derived plan/lowering only;
- exact authenticated source lifetime;
- exact full-model numerical/state comparison;
- genuine cached continuation;
- cancellation/fault/quarantine/publication safety;
- exact resource fit and cleanup;
- serving route enabled only after all above.

### Benchmark campaign complete

- every authorized cell has complete per-run and aggregate output;
- source/tokenizer/output/usage agreement recorded;
- cold and warm separated;
- TTFT/prefill and decode ITL/rate separated;
- actual queue/active/cohort behavior recorded;
- all provider/model/device owners close;
- failures remain failures;
- no unsupported `2B` context is silently omitted or altered.

### Project-wide model/backend claim

Requires independent source, numerical, model-quality, physical, serving and
performance qualification for that exact model/backend/profile. No cross-backend
inheritance.

## 15. Starting prompt for the next agent

The copyable prompt is also stored at
`.agents/docs/NEXT_AGENT_PROMPT_2026-10-27.md`.

> Work in `/home/pvl/spaces/abird/src/zerogravity`. Use GPT-5.6 Sol high/xhigh for
> coordination, implementation and review.
>
> Read `AGENTS.md`, `.agents/plans/README.md`, `.agents/plans/TODO.md`,
> `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-27.md` and
> `.agents/docs/REVIEW_ORDINARY_CANDIDATE_29B491D_2026-10-27.md` completely, then
> follow the handoff's mandatory reading order.
>
> The previous review was interrupted by a usage limit. Resume at handoff §12
> step 0: revalidate the pending ordinary CUDA branch
> `pi-web-agent-8026e968-eb45-46fd-a2a1-d669a232fe6b` at `29b491d` (not integrated,
> `main` is `52bfb51`) from the recorded interruption point, independently
> re-deriving the reviewer's verified claims rather than trusting them. Close the
> RMS reduction-overflow status gap, the `SiluUp` scalar-word order, the
> `byte_size()` `#[cfg]` and alias-rule confirmations, and the untested
> `qwen35-cuda`/default/`--all-features` builds; refresh the stale ordinary refusal
> wording. Then either integrate it strictly as unselected primitives with no
> backend or launch authority, or reject the series and record why.
>
> After that, continue source-derived non-GDN ordinary Qwen CUDA integration through
> the existing physical owner, then current-source cached/fault/shutdown and
> long-context qualification, the final `scripts/check.sh`, and only then the one
> serial matched vLLM campaign in
> `.agents/plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md`.
>
> Preserve the root README, unrelated untracked Fabric vNext files, the
> `qwen35-prompt-reference-20261026/tools/cuda-gdn-recipes/` overlay, all
> source/model identities, independent `StateRoot`s, one `ForwardPlan`, one
> arbiter and one queue/flight/completion/quarantine path. Prompt IDs, lengths,
> call/node ordinals, packet contents, capture tables and recipe hashes must never
> authorize ordinary execution. Ordinary unsupported requests must refuse before
> allocation. Do not run Fabric vNext, push, download, change
> permissions/recovery, or launch held AMD hardware. Inspect and commit only owned
> paths with signed scoped commits; pause and report at substantial verified
> checkpoints.
