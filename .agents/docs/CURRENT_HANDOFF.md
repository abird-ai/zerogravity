# ZeroGravity current comprehensive handoff

This is the canonical cross-window handoff for the repository.

It is deliberately not a generated concatenation of every historical plan, run log, source snapshot or per-file SHA catalogue. Those mechanisms were retired. Current source and tests are the implementation truth, the four maintained planning/status files are the navigation spine, the maintained subsystem and Fabric plans hold detailed design/work items, and Git history retains superseded material.

A successor should be able to start here without prior chat history, understand how the project evolved, identify what is implemented versus merely designed, and continue either the current Qwen correctness critical path or the separately authorized Fabric vNext transition.

## 1. Authority and read order

Authority, highest first:

1. Current source and current tests.
2. AGENTS.md repository rules.
3. The four-file maintained planning/status spine:
   - ../plans/README.md
   - ../plans/TODO.md
   - ../plans/CURRENT_STATUS_AND_TRACKING.md
   - ../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md
4. This handoff, for cross-window context and ordered navigation.
5. The owning subsystem plan/specification for the work being changed.
6. Fabric vNext / transition plans when the user explicitly asks for that architecture work.
7. Git history for superseded plans, old handoffs, removed run artifacts and old branches.

Do not resurrect deleted handoff archives, transient .agents/runs trees, old side branches or old worktrees as live provenance.

Current owned CUDA implementation checkpoint: `2983bb7` (signed). Exact `.8B`
cached47→50, lifecycle/capacity and source gates pass; broader reference scopes
remain open. Earlier checkpoints below are historical. Re-read git status/log.

## 2. Immediate startup for a successor

**Owner resumed: continue and finish.** Owned CUDA `.8B` cached47→50 exact parity
passes; broader lifecycle/context and the final gate precede ROCm. No benchmark detour. Current
results are in section8 and the maintained status/numerics TODO.

Repository:

    /home/pvl/spaces/abird/src/zerogravity

Persistent topology:

- main is the only persistent local branch.
- main is the only persistent repository checkout.
- origin/main is the only remote branch currently relevant.
- no stashes.
- do not create durable side branches/checkouts as coordination state.
- do not push/tag/release unless the user explicitly authorizes it.

Recommended startup:

    export CARGO_BUILD_JOBS=8
    export RAYON_NUM_THREADS=8
    export PYTHONDONTWRITEBYTECODE=1
    git status --short --branch
    git log --show-signature --oneline -20
    git worktree list --porcelain
    nvidia-smi

Then read, in order:

1. AGENTS.md
2. this file
3. ../plans/README.md
4. ../plans/TODO.md
5. ../plans/CURRENT_STATUS_AND_TRACKING.md
6. ../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md

For the current Qwen correctness critical path also read:

7. ../plans/serving-and-speculation/TODO.md
8. ../plans/numerics-and-context/CORRECTNESS_POLICY.md
9. ../plans/numerics-and-context/TODO.md
10. ../plans/model-execution/TODO.md
11. ../plans/cpu-gpu-optimization/TODO.md
12. ../plans/memory-and-streaming/TODO.md
13. ../plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md

For Fabric vNext work also read:

14. ../plans/fabric-vnext/HANDOFF.md
15. ../plans/fabric-vnext/README.md
16. ../plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md
17. ../plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md
18. ../plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md
19. ../plans/fabric-vnext-transition/HANDOFF.md
20. ../plans/fabric-vnext-transition/README.md
21. ../plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md
22. ../plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md
23. ../plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md
24. ../plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md
25. ../plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md
26. ../plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md
27. ../plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md
28. ../plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md
29. ../plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md
30. ../plans/fabric-vnext-transition/TODO.md

That is the intended next-window set. Do not load every historical document up front.

## 3. Original mission and enduring product design

ZeroGravity began as a Rust-native inference compiler/runtime and evolved into a broader execution fabric.

The enduring product goal is one Rust-owned execution fabric that can support:

- text generation;
- embeddings/reranking and encoder tasks;
- image generation/editing;
- video/temporal generation;
- ASR;
- TTS and audio codecs;
- realtime/full-duplex media;
- combined multimodal programs;
- future optional training/update tasks where explicitly designed.

The user-facing aspiration is simple:

- one binary;
- one Rust SDK;
- local-first operation;
- automatic hardware selection when safe;
- explicit fit/capacity explanations;
- OpenAI-compatible serving where implemented;
- no Python inference server dependency for qualified local paths;
- the same runtime for portable and highly specialized execution rather than a separate server per model or modality.

The performance goal is measured Pareto leadership where specialization is justified, never a blanket “always fastest” claim.

The architectural idea from the beginning has been:

- model mathematics and source semantics are defined once;
- state/effects are explicit;
- portable/reference execution exists first;
- optimized hardware paths replace closed semantic regions only when legal and qualified;
- one logical session/root authority publishes committed state;
- one shared physical resource authority owns real memory/device budgets;
- device workers own actual asynchronous compute/events but cannot invent logical commits;
- serving, model loading, scheduling and resource ownership are not cloned for each specialization.

The initial vision and full early architecture remain in:

- VISION_SCOPE_AND_PRINCIPLES.md
- ../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md
- the current public Engineering Specification and Whitepaper under ../../docs/
- current Fabric vNext plans for the next structural transition.

## 4. Engineering philosophy and non-negotiable invariants

Priority order:

1. semantic/state correctness;
2. source and quality fidelity;
3. safety, ownership and crash behavior;
4. compatibility and user clarity;
5. stable latency/throughput;
6. peak specialized speed;
7. experimental complexity only when evidence earns it.

Rust design preference:

- clean, robust, intuitive ownership;
- shared lowest layer when it genuinely benefits multiple callers;
- zero-cost abstractions where natural;
- hot-path optimization where simple and maintainable;
- prefer a clean design over convoluted optimization when the two conflict;
- no second runtime/allocator/state publisher hidden behind “optimization.”

Hard invariants:

- exact source/model identity remains explicit;
- model metadata or a registry slug never grants execution support;
- a finite captured testcase is qualification evidence, not general runtime authority;
- prompt IDs, prompt lengths, ordinals, packet contents, capture hashes or nearest-case tables must not select ordinary production math;
- one logical StateRoot publication authority;
- one process-shared physical ResourceArbiter;
- actual completion/event/fence governs release;
- completion-unknown work is quarantined, not silently recycled;
- source readers/owners survive until actual final consumers retire;
- cancellation cannot publish partial state;
- unsupported shape/source/backend is explicit Unsupported/refusal, never silent fallback to another implementation;
- a passing cleanup or final case cannot relabel an earlier failure;
- compile/source-only success does not imply physical inference support;
- performance follows correctness;
- no universal numeric epsilon; each operator/model/profile owns its declared contract.

The shared correctness policy is:

    ../plans/numerics-and-context/CORRECTNESS_POLICY.md

It deliberately combines the useful ideas from vLLM and llama.cpp without blindly copying either:

- exact semantics/source/state checks;
- independent operator numeric checks;
- model-level output/distribution/quality checks;
- determinism/batch-invariance as separate claims.

## 5. Important rejected patterns

Do not reintroduce:

- one model-specific server per family;
- duplicate resource wallets;
- duplicate StateRoot publishers;
- generic production dispatch based on testcase/capture identity;
- arbitrary Hugging Face remote Python execution;
- metadata-only “support” claims;
- universal tolerance constants;
- hidden precision changes to hit benchmark numbers;
- KV-only prefix reuse for models with additional recurrent/SSM/GDN state;
- benchmark claims before exact source/output/ownership gates;
- a giant permanent graph interpreter in the token hot path;
- per-kernel trait/object hierarchies with no demonstrated need;
- durable agent side branches/worktrees as coordination state;
- monolithic generated handoff archives or documentation SHA catalogues.

## 6. Architecture evolution from the beginning

### 10–12 Oct: initial architecture and v4 consolidation

The early design converged on:

- semantic/effect-aware model programs;
- portable CPU/reference execution;
- optimized device regions;
- extreme model/hardware specialization as optional qualified providers;
- TaskProgram/TaskDriver for non-autoregressive tasks;
- explicit model/session/resource/device boundaries;
- multi-model routing and OpenAI-compatible serving;
- local-first model store/cache/acquisition;
- future SSD and distributed placement.

The v4 minimality principle reduced durable representations to the semantic program and prepared executable/bundle; lower-level compiler/device views are transient.

### 13–16 Oct: source formats, quantization, model families and GPU ownership

Work established:

- GGUF/safetensors/sharded source handling;
- Qwen3.5 and Gemma source/model contracts;
- CPU source-exact math and quant paths;
- GPU backend separation for CUDA/HIP/Vulkan;
- physical ResourceArbiter and source/device ownership;
- exact source/state/fixture policies;
- XDG cache/source behavior;
- early model acquisition/cache foundations;
- multimodal/task architecture research.

Qwen .8B and 2B became the immediate correctness vehicles.

### 17–22 Oct: full-model numerics, GPU native paths and serving foundations

Work expanded through:

- full 24-layer Qwen execution/reference comparisons;
- all 42 Qwen state domains;
- full 248,320 logits;
- RMSNorm/GDN/attention/FFN numerical investigation;
- retained source/window ownership;
- GPU native image/launch qualification;
- cancellation/fault/quarantine;
- serving queues, streaming and shutdown;
- OpenAI-style API subset;
- preliminary matched vLLM comparisons.

Historical numeric differences remain evidence, not current acceptance.

### 23–26 Oct: integration, routing, acquisition and broader backend work

Work covered:

- device routing, including exact /dev selectors where appropriate;
- CLI/config/source precedence;
- acquisition providers and offline policies;
- cache transactions/reclamation;
- pipeline/config validation;
- Qwen27 and Gemma finite qualification lanes;
- HIP/Vulkan qualification work;
- serving concurrency/queue scaling;
- shared state/resource ownership audits;
- repeated fresh bug/regression reviews.

### 27 Oct: Fabric vNext design

The user requested a clean future architecture where the whole target structure becomes visible early in code, then current implementation migrates into it.

The first structural milestone is intended to introduce model-neutral engine and reusable serving crates, visible typed contracts and explicit unsupported stubs before deep migration.

This remains a design/transition plan. It has not yet been implemented as Fabric S0.

### 28 Oct: ordinary source-derived Qwen CUDA integration

Current source now has ordinary Qwen3.5 .8B/2B CUDA execution assembled under the existing resident owner:

- one call-free ForwardPlan;
- authenticated source owner;
- request-local materialization;
- cuBLAS projection path;
- source-exact attention;
- arbitrary-row parameterized GDN;
- F32/F32/BF16 two-root state;
- one queue/flight/completion/quarantine path;
- existing StateRoot publication authority.

It remains qualification-only. No production ordinary CUDA route is enabled.

### 28–29 Oct: evidence hardening, cache human aliases and repository normalization

Recent work also:

- made cache by-name aliases crash-consistent and explicitly non-authoritative;
- simplified ordinary parity evidence so the checked-in capture procedure defines the reference and only the evidence content digest crosses processes;
- removed code/doc SHA catalogues from parity evidence;
- made compare-existing independent of the historical native bundle;
- retired machine-local capture-only tests and dead run-artifact references;
- reduced topology to main-only;
- pruned superseded handoff archives;
- normalized current planning provenance;
- reviewed and fixed environment-sensitive PID1 serving signal tests;
- added Qwen27 layer9 source-contract coverage to the shared check gate;
- refreshed vNext handoff method to use this canonical pointer + maintained files + Git history.

## 7. Current repository/runtime structure

Current workspace owners remain:

- zerogravity-core
- zerogravity-artifacts
- zerogravity-cpu
- zerogravity-gpu
- zerogravity-model-qwen35
- zerogravity-model-gemma4
- zerogravity
- zerogravity-cli
- zerogravity-gpu-qualify

The future Fabric crates zerogravity-engine and zerogravity-serving do not exist yet.

Current logical ownership:

- core: semantic/state/resource/device/pipeline contracts;
- artifacts: source formats, immutable sources, cache, acquisition;
- model crates: family math/source/state/tokenizer semantics;
- CPU/GPU: physical execution providers;
- zerogravity facade: current public composition surface;
- CLI: still owns more reusable application/serving policy than final vNext design wants;
- serving currently lives under CLI and is intended to move to a reusable library during vNext.

## 8. Current Qwen ordinary CUDA status

### What is implemented

Ordinary .8B/2B source-derived CUDA is integrated under the resident execution owner.

Source-free tests establish:

- ordinary cancellation checks before allocation, before flight, during source execution, after launch/completion and before commit;
- armed-flight cancellation enters existing unknown-completion/quarantine behavior;
- explicit provider/pool shutdown;
- unique model ownership before shutdown;
- zero resident bytes/events before namespace close;
- source/provider ownership survives until retirement;
- no alternate root publication path.

The ordinary production route remains intentionally refused until real-current-source qualification closes.

### Exact approved .8B source

Repository/model:

    Qwen/Qwen3.5-0.8B

Revision:

    2fc06364715b967f1860aea9cf38778875588b17

Visible approved source path:

    /home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17

Weight shard:

    model.safetensors-00001-of-00001.safetensors
    bytes: 1,746,942,600
    SHA256: 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696

The tracked source manifest is:

    crates/zerogravity/src/qwen35_cuda/source08.json

### Recovered reference and original blocker — 29 Oct

The old assembled bundle was purged, but its exact `.8B` N1 inputs are recovered
offline under XDG state. Retained canonical images plus two whole SM80
FlashAttention modules extracted from the pinned local reference image and the
original cuBLAS providers match current Rust identities. All 11 model members
verify; maintained bundle validation and isolated structural preflight pass.

Fresh phase-1 cached47→50 reference capture passes after selecting the existing
8 GiB reference-fixture host profile, with all 1,008 source windows retired.
The original 4 GiB refusal stays failed. The independently emitted evidence
content digest is retained in the owning numerics plan and external evidence.

Fresh ordinary comparison fails before model math: pinned `cublasCreate_v2`
returns status 1 after the owned allocator denies a private `cuGetExportTable`
call. Zero allocation attempts/charges and bounded resolver observations identify
this as a provider ownership blocker, not missing model/bundle bytes. Do not
forward opaque tables or preload cuBLAS around admission to make it run.

The current evidence and remaining work live in
`../plans/numerics-and-context/TODO.md`. `.8B` cached47→50 is exact; broader
lifecycle/context and production routing remain unqualified. Owner order is CUDA correctness first,
then ROCm on `pvl-x2`, with no subagents.

Maintained tooling:

- tools/qwen35-cuda-reference/
- tools/qwen35-cuda-bundle/
- scripts/qualify-qwen35-ordinary-cached47-50.sh

The Python discovery/packed-capture protocol still requires explicit retained
external capture sources. Exact offline bundle recovery is documented in the
bundle helper README; the Rust two-process parity wrapper uses that recovered
bundle. These are qualification inputs, not production authority.

### Owned execution checkpoint — `2983bb7`

Owner approved replacing ordinary opaque BLAS with shared driver-only GEMM.
The allocation-free tensor-core/SIMT primitive passes 180 synthetic cases and
eight authenticated prefill/cached projections exactly, including guards and
retirement. Typed SSA roles choose explicit reductions without captured node or
prompt lookup. Original compiler-context reservations and physical flights remain
the owners; pinned reference/source/resource limits are unchanged. Ordinary
comparison no longer requires cuBLAS paths.

Corrected the rotary admission omission (+280 bytes), GDN partition binding and
premature BF16 rounding of residual RMS sums. Row RMS retains Torch vector-four
FP32 boundaries; Q/K RMS instead follows the fused Triton adjacent-pair/warp
reduction. CUDA rotary now uses FP32 FMA and final BF16 rounding; the CPU-target
BF16 product contract is unchanged. All211 GPU host tests (43 ignored) and an
independent host-FMA rotary comparison at14 positions through32767 pass.

Owned tile-64 CUDA attention now preserves FlashAttention's BF16 unnormalized
probabilities, FTZ FP32 online softmax and tensor-core QK/PV. Runtime geometry,
not captured-case routing; no new device scratch or allocation authority. Both
independently authenticated prefill/cached attention captures match exactly.
Nine analytic causal/multitile cases through a32K active prefix and three
nonfinite refusals pass with guards and retirement. They are not long-context
model parity.

Fresh `.8B` cached47→48→49→50 passes exactly: all248,320 logits, all42 state
domains and natural feedback at every step, with normal model shutdown. The
reference bytes/digest are unchanged. The earlier229,459-logit /4,359,077-state
failure (max0.28125) remains failed historical evidence.

Replaced obsolete row/residual CPU-profile CUDA oracles with independent staged
CUDA/Torch comparisons:16 cases across widths1024/2048, rows1/2/17/64 and residual
modes pass exactly, including non-BF16 residual sums, guards and retirement.
Real post-launch cancellation and abandoned-future probes pass:396 original
handles quarantined, prior bits/tails intact, unchanged root, peer refusal and
retained charges after external drops. Normal lifecycle passes all13 admitted
capacities (`.8B` through32K,2B through16K), each at48 active tokens;2B32K refuses
at pure preflight. Long-active-context/2B numerical parity remains open. Final
isolated repository gate passes2,129 Rust executions/0fail/83ignored;14 Python
suites and ancillary gates pass. CUDA/native host675/0fail/54ignored and strict
CUDA/HIP/Vulkan feature checks pass.
The2B approved snapshot exists in the canonical cache; its extra reference image
must be recovered independently, not replaced by the `.8B` inventory.

Owned execution landed in signed main commit `2983bb7`; nothing is pushed and
production remains disabled. The extra exact2B reference image was not found
among1,252 extracted SM80 modules; retained original capture inputs/long-context
comparisons are also missing from the current qualification root. Do not replace
or repin them. ROCm is unqualified; read-only SSH reached pvl-x2 and observed AMD
PCI1586, cached ROCm/Rust/Python and prior qualification inputs, but no current
repository checkout or global Python/Cargo. Existing device nodes are unchanged;
no physical AMD test, SDK/model download or privilege change. Remote cached
inputs also contain no exact2B image. See the owning numerics TODO.

Do not recreate deleted `.agents/runs` paths or treat old run names as provenance.

## 9. Ordinary cached47→50 parity protocol

Current parity design was deliberately simplified.

The checked-in reference capture procedure defines how phase 1 is produced.

Evidence schema v3 contains:

- source coordinates;
- prompt;
- 47→48→49→50 observations;
- full logits;
- all state domains;
- selected IDs/max-bit receipts;
- teardown/close receipt;
- an embedded corruption digest.

Cross-process trust:

- phase 1 emits the evidence content SHA256 in stdout;
- normal wrapper execution carries that digest in memory to phase 2;
- compare-existing requires the independently supplied trusted digest;
- the embedded digest is corruption detection, not a self-authenticating implementation identity.

Removed intentionally:

- reference implementation SHA graph;
- source-code SHA catalogues;
- migration pin catalogues;
- stage-receipt authority;
- code-hash-based evidence identity.

Phase 2 does not need the historical native bundle; it needs:

- captured evidence;
- trusted phase-1 evidence digest;
- current approved model source;
- ordinary CUDA providers.

Normal phase 1 still needs the qualified native/reference bundle.

The wrapper is:

    scripts/qualify-qwen35-ordinary-cached47-50.sh

Modes:

- normal: structural checks, ephemeral legacy staging if necessary, reference capture, digest handoff, ordinary compare;
- preflight-only: structural/environment checks only, no large copy;
- compare-existing: no historical bundle required.

## 10. Exact ordinary CUDA context admission

Under the unchanged 4608 MiB provider / 5120 MiB physical limits, totals include
cold model, two roots and retained conservative admission headroom. The owned
GEMM allocates no scratch; these are not claims of physical scratch usage.
The 30-Oct correction includes the previously omitted 280-byte rotary table.

### .8B

- 512: 1,652,916,585 B
- 1K: 1,667,662,185 B
- 2K: 1,697,153,385 B
- 4K: 1,756,135,785 B
- 8K: 1,874,100,585 B
- 16K: 2,110,030,185 B
- 32K: 2,581,889,385 B

All fit admission.

### 2B

- 512: 3,911,846,249 B
- 1K: 3,926,591,849 B
- 2K: 3,956,083,049 B
- 4K: 4,015,065,449 B
- 8K: 4,133,030,249 B
- 16K: 4,368,959,849 B
- 32K: 4,840,819,049 B

512 through 16K fit.

32K exceeds 4,831,838,208 B by 8,980,841 B and must refuse before physical allocation.

These are admission/fit results only. Every admitted context still needs real-model correctness, continuation, fault and ownership qualification.

## 11. Current Qwen critical path

Unless the user explicitly switches to Fabric vNext implementation, this is the main execution sequence:

1. Preserve the recovered exact `.8B` native/reference inputs and approved source.
2. Preserve passing structural preflight and fresh phase-1 capture/retirement evidence; retain its independently emitted digest.
3. Finish qualifying owned driver-only GEMM integration without weakening admission or quarantine.
4. Start a fresh ordinary process with `--compare-existing` and compare exact cached 47→48→49→50:
   - 248,320 logits;
   - all 42 state domains;
   - natural feedback;
   - source/root identity.
5. Run real after-launch cancellation/fault/unknown-completion/quarantine.
6. Run retirement/provider shutdown/real-model shutdown.
7. Qualify .8B context ladder 512/1K/2K/4K/8K/16K/32K.
8. Qualify 2B only for admitted cells; 32K remains a required pre-allocation refusal under current plan.
9. Run the assembled repository gate.
10. Qualify ROCm/HIP on `pvl-x2` next; no downloads or privilege changes without authorization.
11. Deferred, not the current task: the serial matched ZeroGravity/vLLM campaign:
    - canonical;
    - ragged;
    - queue scaling;
    - exact context workloads;
    - ABBA/BAAB repetitions;
    - cold and warm separated;
    - TTFT/prefill/decode/throughput/goodput separately reported.

Benchmark plan:

    ../plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md

## 12. Serving and router state

The project already has:

- bounded HTTP handling;
- health/models;
- OpenAI-style text/chat completions subset;
- JSON/SSE streaming;
- request/output usage accounting;
- bounded pending queue;
- cancellation/disconnect propagation;
- CPU route;
- finite/native CUDA routes where qualified;
- source/model route validation before side effects;
- graceful worker join/shutdown;
- source-free C16/C32/C64 queue/population coverage.

The source-free production router queue is not the blocker for ordinary Qwen CUDA; the model route is.

Recent fresh review found the process-signal fixture incorrectly assumed every environment could nest a bubblewrap PID1 namespace. The shared fixture now:

- always exercises ordinary SIGTERM/SIGINT process shutdown;
- preflights PID1 namespace capability;
- skips PID1 rows only for explicit namespace/permission unavailability;
- hard-fails unexpected bubblewrap/preflight errors;
- reports skip reason explicitly;
- Rust asserts ordinary rows cannot skip and PID1 rows pass when capability exists.

Current serve all-targets result after the fix:

- 89 passed;
- 0 failed;
- 4 intentionally ignored source/hardware fixtures.

## 13. Cache/acquisition current state

Canonical cache data is authoritative; human by-name naming is not.

Human projection design:

    models/
      snapshots/<digest>/...
      refs/...
      .by-name-initialized
      .by-name-state/
      by-name/

Key properties:

- runtime/model selection never reads by-name state;
- by-name projection is disposable;
- explicit aliases and automatic names live in durable human state;
- dirty modes recover projection/reverse/reconciliation;
- publication can leave a pending human update and later self-heal;
- missing state after prior initialization refuses rather than silently losing aliases;
- reset requires explicit alias discard;
- mount-boundary checks use mount identity, not just st_dev;
- process-exit/fault matrices exercise human metadata and reclamation recovery;
- canonical snapshot/ref availability is not downgraded by human metadata failure.

Acquisition/cache supports bounded, source-aware providers and offline exact-cache behavior for the implemented providers, including Hugging Face/GitHub/Xet/OCI/S3/GCS/Azure/ModelScope/object flows where the relevant feature is enabled.

No provider metadata or alias may upgrade source rights.

## 14. Device/CLI/config state

Current CLI/device work includes:

- explicit CPU/device routing;
- exact PCI/device identity;
- validated direct /dev selectors for supported DRM/NVIDIA nodes;
- NVIDIA minor is not treated as CUDA ordinal;
- aliases converge on one observed device identity;
- source/alias/device policy precedes execution side effects;
- bounded YAML pipeline/config checking;
- strict config/env/argv precedence;
- dry-run and refusal behavior.

Pipeline configuration is more complete than pipeline execution. A valid YAML graph does not imply the task pipeline is implemented.

## 15. Memory/SSD/state direction

Implemented foundations include:

- shared physical resource accounting;
- immutable source extents;
- bounded source windows;
- source descriptor retention;
- transfer staging admission;
- retained GPU readers through actual completion;
- state/root/cancellation/quarantine semantics.

Open work remains for:

- production paged complete-domain state;
- complete-domain prefix reuse/COW;
- product snapshot persistence/restore;
- real SSD model streaming;
- expert/source IO QoS;
- eviction under actual GPU completion;
- broader multi-model residency.

Do not call KV-only reuse a hybrid-state prefix hit when recurrent/GDN/SSM/other domains are missing.

## 16. Model-family status discipline

Qwen .8B/2B are the primary current ordinary CUDA correctness path.

Qwen27, Gemma4 and other families have substantial source/reference/finite backend work, but do not inherit .8B support automatically.

Keep these distinctions:

- source parser support;
- family mathematical support;
- CPU reference support;
- a finite backend qualification;
- a production route;
- long-context support;
- serving support;
- performance qualification.

They are separate claims.

Future/target families and tasks discussed in the design include broader Qwen, Gemma4, Llama/Mistral, GLM, DeepSeek, image, ASR, TTS, Whisper/Canary/Parakeet-style audio systems and multimodal pipelines. Treat these as roadmap targets unless current source/tests explicitly implement and qualify them.

## 17. Fabric vNext — current canonical design

Fabric vNext is the next architectural transition, not current implementation.

User intent:

- make the final architecture visible in code immediately;
- break pre-v1 internal/public APIs when that produces a cleaner design;
- move current working code into the new ownership model instead of carrying compatibility layers forever;
- allow private compile-safe placeholders/Unsupported where implementation is not yet migrated;
- no reachable todo panic on an advertised supported path;
- preserve all current exact source/numeric/state/resource safety;
- qualify the assembled transition once, then fix regressions in small focused commits.

Target dependency/ownership idea:

- zerogravity-core: semantic/numeric/state/provider/physical contracts;
- zerogravity-artifacts: containers/codecs/source/acquisition/store;
- model crates: family source/math/state/tokenization/specialization;
- CPU/GPU: physical providers;
- new zerogravity-engine: model/backend-neutral runtime, scheduling, sessions, residency;
- new zerogravity-serving: reusable server/protocol layer depending on engine/core, not facade/CLI;
- zerogravity facade: composition root/public SDK;
- CLI: arguments/presentation/signals only.

The facade is the composition root. Engine must not depend on Qwen/Gemma or physical CUDA/CPU implementations. Serving must not depend on facade/CLI. CLI should call the public SDK.

### Planned transition sequence

S0 — full structural skeleton first

- create engine and serving crates;
- expose top-level target ownership/contracts;
- compile;
- explicit Unsupported for pending operations;
- architecture checker updated;
- no heavyweight model qualification yet.

S1 — exact source/quant/encoding migration.

S2 — family math + GPU provider/qualification separation.

S3 — canonical state/root/resource publication and ownership.

F0 — model-neutral engine.

F1 — complete public Rust SDK.

F2 — reusable serving library.

F3 — thin CLI and deletion of shadow runtime copies.

F4 — feature parity matrix across all currently real behavior/refusals.

Q0 — one assembled broad qualification.

Q1 — focused bugfix queue at lowest shared layers.

Q2 — final full gate, physical/source qualification and capability matrix.

Only after transition:

- broader models;
- vLLM-style continuous batching/paged state;
- larger model residency;
- SSD execution;
- Gemma broad native routes;
- additional quant formats;
- sparse/MoE/MLA/Engram/MTP;
- multimodal ASR/TTS/image/video;
- NPU/remote distributed placement;
- calibrated autotuning/benchmarking.

The vNext documents no longer require or promise a monolithic archive or documentation SHA index. This handoff is the cross-window pointer. Git history is the archive.

## 18. Current plan queue

Authoritative queue:

    ../plans/TODO.md

Current priorities at this handoff:

P0:
- serving-and-speculation — ordinary .8B cached47→50/lifecycle/context qualification, then matched vLLM campaign.

P1:
- numerics-and-context;
- memory-and-streaming;
- cpu-gpu-optimization;
- pipeline-composition;
- cli-device-routing.

P2:
- model-execution;
- model-acquisition.

Icebox:
- multimodal;
- distributed-execution;
- platform-and-release.

Fabric vNext remains a maintained architecture/transition packet; actual S0 starts only when the user explicitly asks to begin that implementation.

## 19. Fresh review findings and fixes immediately before this handoff

Fresh review found and resolved:

1. Serving shutdown test portability:
   - nested PID1 bubblewrap was assumed universally available;
   - capability is now preflighted;
   - ordinary TERM/INT always run;
   - PID1 skip is explicit and limited to permission/namespace unavailability;
   - unexpected bubblewrap errors still fail.

2. Shared check coverage:
   - qwen27b-layer9-prefix-reference source-contract tests were maintained but absent from scripts/check.sh;
   - they are now part of the shared source-only gate.

3. Repository policy contradiction:
   - repo-specific instructions said main-only topology but later recommended persistent side worktrees;
   - repo-specific policy now consistently uses main-only persistence, explicit file ownership and ephemeral external scratch.

4. Fabric vNext handoff drift:
   - live docs still promised combined single-file archives / documentation SHA catalogues;
   - those instructions are retired;
   - canonical handoff method is this file + four-file spine + maintained vNext plans + Git history.

5. Active Qwen plan drift:
   - three plans still said the approved .8B safetensor was inaccessible;
   - it is now visible and hash-verified;
   - plans now identify the true missing input: qualified native/reference bundle.

6. Active plan completion drift:
   - older top sections still listed non-GDN ordinary lowering and context-fit derivation as open;
   - later accepted implementation had already closed those;
   - the top open-work statements are reconciled.

7. Source hygiene:
   - ignored Python bytecode left by earlier tooling was removed.

No additional source/Cargo correctness regression was found in the fresh pass after those fixes.

## 20. Validation state

The repository was comprehensively exercised during the cleanup/review sequence.

Important passing scopes include:

- formatting;
- architecture dependency/DAG checks;
- architecture checker self-tests;
- HIP/Vulkan/CUDA feature-boundary compiles;
- ordinary Qwen route/refusal feature-boundary tests;
- Qwen qualification route tests;
- CPU dependency isolation;
- strict CPU clippy;
- full CPU workspace tests;
- minimal no-default feature checks;
- cache transaction tests;
- diagnostic cache process-exit/reclamation/by-name recovery tests;
- source-only model/reference contracts;
- CLI cache-only tests;
- shared CPU preparation/cache tests including real local CPU inference;
- acquisition base and diagnostic artifact/CLI tests;
- release CPU CLI build;
- installer integrity/idempotency/version fixture;
- shellcheck;
- Nix formatting;
- serving benchmark/comparator/matrix self-tests;
- numerical layer/attention/cpu-qualification/native-GPU script controls.

After the PID1 harness fix:

- full serve all-targets: 89 passed, 0 failed, 4 intentionally ignored;
- qwen27 initial-core source contract: 7 passed;
- qwen27 layer9-prefix source contract: 6 passed.

A previous monolithic scripts/check.sh invocation exceeded the connector's single-call wall while compiling, so the same sections were executed in bounded groups. Do not interpret the connector wall as a source failure.

The 29-Oct continuation recovered `.8B` native inputs and passed fresh reference
capture. The old opaque-provider startup still refuses a private driver
export-table call. The30-Oct driver-only path supersedes that blocker and passes
exact `.8B` cached47→50; broader physical scopes remain open (section8).

## 21. Git/topology cleanup history

Recent normalization commits include:

- 768b724 — serving Qwen CUDA qualification-boundary test alignment
- 932f564 — GDN recipe capture-tool migration
- 734d29f — retire obsolete Qwen27 replay examples
- 2ff3d49 — add Fabric vNext transition design
- 15f5bd3 — document retired worktree/run cleanup
- 9abd936 — retire machine-local capture dependencies
- ace934b — prune superseded handoff archives
- 7b5e50e — normalize current planning provenance
- 379d91d — capability-aware PID1 serving shutdown control
- ae8cd89 — Qwen27 layer9 source contract added to shared gate

The large archive prune removed hundreds of superseded handoff/sourcebook/run-history files. That material remains in Git history.

Final topology after cleanup:

- one persistent checkout;
- one local branch: main;
- origin/main remote;
- no stashes;
- no old pi-web-agent branches;
- no old worktree-pool references;
- no tracked .agents/runs provenance;
- root README remained byte-identical during cleanup.

## 22. How to continue safely

If the user says “continue the main task”:

- stay on the Qwen ordinary CUDA critical path in section 11;
- do not start Fabric S0;
- preserve recovered native inputs and passing phase-1 reference evidence;
- preserve the passing owned `.8B` cached47→50/lifecycle/source-gate checkpoint;
- recover missing exact2B and long-active-context reference inputs;
- finish those broader numerical scopes without altering the reference;
- latest owner order: ROCm on `pvl-x2` after CUDA correctness, no subagents;
- benchmark only after correctness and without displacing that ROCm task.

If the user says “begin Fabric vNext” or equivalent:

- re-read current HEAD/source;
- read the vNext/transition window set;
- confirm explicit implementation authorization;
- start with S0 full structural skeleton;
- do not treat historical fadf2b1/195099ca source maps as current;
- keep current source behavior/refusals intact while moving ownership;
- no duplicate engine/resource/state authority.

If the user asks for another review:

- review current source afresh;
- do not merely replay this handoff's conclusions;
- prioritize correctness/regressions and shared low-level fixes;
- update current plans/status when factual state changes.

## 23. Do not do

- Do not push automatically.
- Do not tag/release automatically.
- Do not recreate old worktrees/branches.
- Do not recreate .agents/runs as tracked provenance.
- Do not rebuild the retired monolithic handoff archive.
- Do not make per-document SHA catalogues.
- Do not download/substitute model sources without authorization.
- Do not infer physical GPU support from compilation.
- Do not enable ordinary Qwen CUDA production routing before real source+native qualification.
- Do not benchmark before parity/lifecycle/context gates.
- Do not weaken source/state/resource semantics to make a context or benchmark fit.
- Do not modify the root README unless the user explicitly asks.

## 24. The successor's first decision

After reading the required window set, determine which of these two paths the user's latest message requests:

A. Current implementation critical path:
   retain passing owned CUDA `.8B` cached47→50 exact parity -> finish broader
   lifecycle/context and2B numerical coverage -> final gate ->
   ROCm on `pvl-x2` (latest owner order). Matched benchmarking remains deferred.

B. Fabric vNext transition:
   current-source resync -> S0 full visible architecture skeleton -> S1/S2/S3 -> F0/F1/F2/F3/F4 -> Q0/Q1/Q2.

Do not mix the two implicitly. The serving benchmark plan explicitly excludes Fabric vNext; the Fabric transition must preserve current verified behavior and qualification boundaries.

This document should remain concise enough to load in one window while still carrying the complete continuity story. Detailed facts belong in the maintained linked plans, not duplicated here. When a fact changes, update the owning plan/status file first and this handoff only when the cross-window continuation materially changes.
