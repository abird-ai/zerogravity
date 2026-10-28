# ZeroGravity full comprehensive handoff — 28 Oct 2026

Author: @pvl

This is the canonical non-Fabric takeover document for the next agent.

It supersedes the current-state conclusions in FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-27.md, but it deliberately preserves that entire handoff later in this file as a historical appendix so no design rationale, failure record, source identity, or earlier work item is lost.

Precedence rule:

1. Sections 0–20 of this 28 Oct handoff are current truth.
2. Current source code and measured physical results outrank older prose.
3. The appended 27 Oct handoff is historical context. Any statement there that conflicts with sections 0–20 is superseded, not deleted.
4. Detailed plan ledgers and sourcebooks remain authoritative for subsystem-specific history that is only summarized here.
5. Fabric vNext remains outside this handoff and must not be mixed into this workstream.

---

## 0. Mission and immediate objective

Continue ZeroGravity as one source-authentic Rust execution fabric, not as a collection of model-specific scripts.

The core architectural promise remains:

- one shared mathematical traversal and one ForwardPlan model;
- one physical ResourceArbiter;
- one queue / flight / completion / quarantine path;
- independently owned StateRoots;
- immutable authenticated source ownership;
- source-derived planning rather than captured testcase dispatch;
- backend-specific implementations that independently qualify the same mathematical/source contract;
- correctness and physical ownership before serving or performance claims;
- one engine across CLI, SDK, serving, routing, pipelines, models and modalities.

Immediate objective at this handoff:

1. Run the artifact-dependent current-source Qwen3.5 .8B cached 47→50 exact parity campaign.
2. Compare all 248,320 logits and all 42 canonical state domains at frontiers 47, 48, 49 and 50.
3. Then qualify after-launch cancellation/fault, retirement and real-model shutdown on the assembled ordinary path.
4. Then run .8B context correctness 512→32K.
5. Admit 2B only where the exact two-root admission plan fits.
6. Only after correctness/ownership/context qualification, run the matched vLLM campaign.
7. Continue broader Qwen3.8-27B, Gemma4, HIP/ROCm, Vulkan, SSD streaming, pipelines and serving work under their own contracts.

The meaningful source-independent ordinary CUDA correctness surface has been exhausted. The current hard boundary is missing exact held artifacts, not missing implementation.

---

## 1. Repository snapshot at handoff

Repository:

    /home/pvl/spaces/abird/src/zerogravity

Current main:

    d03bb3a94c431b9a50a7767a9da3cfe177bebc7c
    docs: record ordinary parity restart semantics

Branch state at handoff creation:

- main is 24 commits ahead of origin/main.
- no push has been performed.
- only the main worktree is registered.
- all finished ordinary/review/prompt-reference worktrees were merged or otherwise reconciled and removed.
- work now proceeds purely in the main worktree.
- tracked tree is clean before this handoff update.
- unrelated untracked Fabric vNext documents/plans remain intentionally preserved.
- tools/cuda-gdn-recipes remains intentionally preserved as untracked source/reference material.

Expected untracked material that must not be deleted or staged accidentally:

- .agents/docs/FABRIC_VNEXT_COMPLETE_REFERENCE_2026-10-27.md
- .agents/docs/FABRIC_VNEXT_INDEX_2026-10-27.md
- .agents/docs/FABRIC_VNEXT_TRANSITION_COMPLETE_REFERENCE_2026-10-27.md
- .agents/docs/FABRIC_VNEXT_TRANSITION_INDEX_2026-10-27.md
- .agents/plans/fabric-vnext/
- .agents/plans/fabric-vnext-transition/
- tools/cuda-gdn-recipes/

Root README is user-authored. Do not change its bytes unless explicitly asked.

Do not push, tag, publish, deploy, rewrite history, download model/provider replacements, change device permissions/recovery, or run held AMD hardware without explicit authorization.

Commits must remain signed.

---

## 2. Mandatory startup procedure for the next agent

Start in:

    /home/pvl/spaces/abird/src/zerogravity

Use at least 8 Cargo build jobs unless a concrete memory/resource reason requires otherwise:

    export CARGO_BUILD_JOBS=8
    export RAYON_NUM_THREADS=8
    export PYTHONDONTWRITEBYTECODE=1

Then run:

    git status --short --branch
    git log --show-signature --oneline -30
    git worktree list --porcelain
    nvidia-smi

Read completely, in this order:

1. AGENTS.md
2. .agents/plans/README.md
3. .agents/plans/TODO.md
4. .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-28.md
5. .agents/docs/NEXT_AGENT_PROMPT_2026-10-28.md
6. .agents/docs/handoff-2026-10-26/USER_DECISIONS.md
7. .agents/docs/handoff-2026-10-26/DESIGN_AND_DECISIONS.md
8. .agents/docs/handoff-2026-10-26/EVIDENCE_AND_CONTINUATION.md
9. .agents/docs/handoff-2026-10-26/OPEN_WORK.md
10. .agents/docs/handoff-2026-10-26/SOURCE_CATALOG.md
11. .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md
12. .agents/plans/model-execution/TODO.md
13. .agents/plans/numerics-and-context/TODO.md
14. .agents/plans/cpu-gpu-optimization/TODO.md
15. .agents/plans/serving-and-speculation/TODO.md
16. .agents/plans/memory-and-streaming/TODO.md
17. .agents/plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md
18. .agents/docs/REVIEW_ORDINARY_CANDIDATE_29B491D_2026-10-27.md for historical review context only.

For complete earlier history, also use:

- .agents/docs/HANDOFF_ALL_COMMIT_LINEAGE_2026-10-25.md
- .agents/docs/HANDOFF_ALL_DESIGN_AND_SPEC_SOURCES_2026-10-25.md
- .agents/docs/HANDOFF_ALL_OPEN_PLAN_ITEMS_2026-10-25.md
- .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md
- .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md
- .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-20.md
- .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25.md
- .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md
- .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-NIGHT.md
- .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-26.md
- .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-27.md
- .agents/docs/handoff-2026-10-20/HISTORY_DESIGN_AND_COVERAGE_AUDIT.md
- .agents/docs/handoff-2026-10-20/CONTEXT_CATALOG.md
- .agents/docs/handoff-2026-10-20/CURRENT_TAKEOVER.md

Do not assume a dated handoff is current just because it is detailed. This Oct 28 file is the takeover authority for present state.

---

## 3. Original product vision and architecture

ZeroGravity was designed as a single Rust-native AI execution fabric, not a dedicated Qwen runtime.

The intended product is:

- one binary/runtime for text, image, video, audio, real-time and combined multimodal inference;
- model-specific math, kernels, memory placement, scheduling and serving composed under one engine;
- exact model routing and multi-model routing as distinct concerns;
- OpenAI-compatible APIs built into the engine;
- load balancing and multiple resident models;
- single-process or per-model-process deployment where appropriate;
- YAML/config-driven multi-model pipelines;
- pipelines such as ASR → LLM → TTS in one execution fabric;
- future image/audio/video streaming under the same scheduler/ownership model;
- heterogeneous CPU/GPU/NPU/remote placement;
- explicit device selection including Linux device-path semantics where appropriate;
- SSD-streamed model execution and virtual tensor residency;
- batching, paged state, prefix COW and speculation as shared scheduler capabilities, not model forks.

Current and planned model families include Qwen, Gemma, Llama/Mistral, DeepSeek/GLM-style models and multimodal/audio models. Those broader targets do not reduce the exact qualification burden of any current model/backend cell.

Canonical architecture references:

- .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md
- .agents/docs/VISION_SCOPE_AND_PRINCIPLES.md
- .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md
- .agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md
- .agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md
- .agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md
- .agents/docs/ASYNC_RESOURCE_COMMAND_IR.md
- .agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md
- .agents/docs/STATE_CACHE_AND_TRANSACTIONS.md
- .agents/docs/MEMORY_RESIDENCY_AND_IO.md
- .agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md
- .agents/docs/RUNTIME_SERVING_AND_UX.md
- .agents/docs/SEMANTIC_IR_AND_COMPILER.md
- .agents/docs/DEVICE_KERNEL_AND_CODEGEN.md
- .agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md
- .agents/docs/SECURITY_AND_RELIABILITY.md
- .agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md
- .agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md
- .agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md
- .agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md

The active acceptance policy is stricter than some early vision documents: live source, measured physical behavior and current invariants outrank roadmap prose.

---

## 4. Non-negotiable engineering rules

### 4.1 One execution authority

Do not add:

- another token loop;
- another root issuer;
- another CUDA queue;
- another completion service;
- another allocator authority;
- another state publication path;
- another shadow model runtime.

ForwardPlan is the shared mathematical program. Runtime invocation carries call-local rows/base/frontier/output selection. StateRoot/completion/publication remain outside the immutable plan.

Every physical GPU request must use the existing admitted queue/flight/completion/quarantine ownership.

Unknown work is fenced/quarantined. Process exit is not physical completion.

### 4.2 Captures and historical fixtures are never production authority

The following may be qualification evidence only:

- prompt IDs;
- prompt lengths;
- testcase/call/node ordinals;
- packet contents;
- finite native image inventories;
- recipe/catalogue hashes;
- captured launch tables;
- nearest historical testcase;
- prompt-specific kernel selections.

Ordinary execution must be selected from authenticated source/model semantics and current runtime geometry, not a capture coordinate.

### 4.3 Source identity, math, physical implementation and performance are separate claims

Never conflate:

1. source identity;
2. mathematical/numerical correctness;
3. physical ABI/kernel execution;
4. whole-model logits/state parity;
5. ownership/lifecycle correctness;
6. serving behavior;
7. determinism;
8. performance.

A pass in one category does not grant the others.

### 4.4 State and publication

Independent requests have independent roots even if they share immutable weights or prepared kernels.

Ordinary Qwen canonical state is:

- convolution roots: F32 canonical;
- recurrent roots: F32 canonical;
- KV roots: BF16 canonical physical storage;
- snapshots widen active KV BF16 to F32 only for comparison;
- future KV tail must remain exact +0 in committed-root snapshots.

State mutation publishes only after actual completion and shared-status validation.

### 4.5 Resource ownership

Every resident/resource cost must be preflighted before allocation.

Keep source owners, queue/flight owners, state owners, readback owners, provider owners and quarantine owners alive until their true retirement point.

Do not hide provider memory, source windows, host evidence, native modules or state roots outside admission accounting.

### 4.6 Repository safety

- Work only on main unless a new isolated scope is truly required.
- Preserve unrelated untracked Fabric vNext material.
- Preserve tools/cuda-gdn-recipes.
- Do not delete other owners’ historical copies.
- Use signed scoped commits.
- Inspect staged paths before commit.
- No push unless explicitly requested.

---

## 5. Original Qwen CUDA problem and why the ordinary path was required

Historical finite Qwen CUDA qualification was strong but finite.

It had valid evidence for specific source/model/prompt scopes, but ordinary production execution could not remain coupled to:

- five prompt lengths;
- prompt-specific native packet inventories;
- length-specific GDN recipes;
- capture hashes;
- call/node ordinals;
- finite compatibility maps.

The Oct 26–27 architecture work therefore separated:

- semantic prompt/model validity;
- shared call-free source plan;
- call-local invocation binding;
- parameterized GDN geometry;
- qualification-only captured paths;
- production CUDA transport;
- ordinary source-derived runtime.

The design target became:

- one call-free ordinary plan per source profile/context;
- arbitrary 1–64 row calls;
- arbitrary valid base/frontier;
- initial prefill and cached decode;
- context capacity 512..32768;
- exact authenticated source tensors/parameters;
- same resident service/flight/completion/StateRoot authority as the rest of the engine.

---

## 6. Ordinary Qwen source geometry and semantics

Supported ordinary source profiles:

Qwen3.5 .8B:

- repository: Qwen/Qwen3.5-0.8B
- revision: 2fc06364715b967f1860aea9cf38778875588b17
- hidden width: 1024
- FFN intermediate: 3584
- vocabulary: 248,320

Qwen3.5 2B:

- repository: Qwen/Qwen3.5-2B
- revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
- hidden width: 2048
- FFN intermediate: 6144
- vocabulary: 248,320

Shared model structure:

- 24 layers;
- full attention every fourth layer: 3, 7, 11, 15, 19, 23;
- remaining 18 layers are GDN/linear-attention layers;
- state domains: 18 Conv + 18 Recurrent + 6 KV = 42;
- Q heads: 8;
- KV heads: 2;
- head width: 256;
- rotary width: 64;
- RoPE theta: 10,000,000;
- RMS epsilon: 1e-6;
- max ordinary chunk rows: 64;
- ordinary context range: 512..32768.

GDN geometry:

- packed QKV: 6144;
- Z: 2048;
- coefficient channels: 32;
- convolution kernel: 4;
- recurrent heads: 16;
- recurrent head width: 128;
- cache slots: 5.

Current neutral ordinary GDN stage configuration:

- KKT: Bk32W8S3
- recompute: Bk64Bv64W8S3
- state: Bv64W4S2
- output: Bk32Bv64W4S3

---

## 7. Ordinary plan and source authority

The call-free ordinary plan is source-derived and contains no request-specific execution authority.

The plan separates:

- immutable semantics/topology;
- source tensor/parameter descriptors;
- state requirements;
- call-local SourceCall / SourceRows;
- runtime mode (prefill/decode);
- base/frontier;
- token IDs;
- output selection.

Materialized request plans must share the cold plan semantics/topology. A request cannot invent a new operator graph.

Source numerical contract for ordinary Qwen is the source-reference CUDA BF16 contract, not a capture label.

Prompt IDs and prompt length are data, not dispatch authority.

---

## 8. Ordinary CUDA implementation architecture

### 8.1 Primitive layer

Dynamic ordinary text primitives live under the CUDA ordinary-text implementation.

Reviewed operations include:

- embedding;
- RMS;
- residual RMS;
- query/key head RMS;
- RoPE support;
- SiLU + up;
- sigmoid;
- multiply;
- KV append;
- BF16/F32 transport;
- final-logit transport.

The legacy ordinary RoPE implementation is no longer production authority because physical qualification found a real source mismatch. See §13.

### 8.2 Owned cuBLAS projection path

The ordinary projection path uses an explicitly owned cuBLAS/cuBLASLt provider, not implicit process-global activation.

Important rules:

- provider identity is pinned by SONAME, canonical DSO extent and SHA256;
- activation failures are fail-closed;
- retirement is explicit;
- provider/pool ownership remains attached to physical work;
- model shutdown retires provider/pool before resident leases and namespace close.

Projection packs are reused/deduplicated by exact matrix descriptor identity, including tied embedding/output where applicable.

### 8.3 Parameter transport

Verified source tensors/parameters are uploaded by descriptor.

Storage classes include:

- Original;
- Bf16Rne.

F32→BF16 conversion uses round-to-nearest-even.

Nonfinite F32 and finite overflow to BF16 infinity are refused.

Signed zero and subnormal behavior is preserved.

### 8.4 Resource layout and preflight

OrdinaryCudaAdmission derives cold + two-root costs before allocation.

The layout accounts for:

- packed source matrices;
- uploaded parameters;
- inverse-frequency data;
- reusable BF16 SSA value slots;
- residual aliases;
- GDN internal workspace;
- attention managed scratch;
- source-exact rotary scratch/tables;
- token controls;
- final F32 logits;
- shared status;
- owned BLAS workspace;
- foreign provider charge;
- two independent root sets.

Residual RMS carried values intentionally alias the same physical residual storage.

GDN Rstd, InitialState and FinalState are internal local scratch, not canonical external roots.

### 8.5 Cold owner

OrdinaryCudaCold owns the immutable/reusable ordinary CUDA model resources:

- source/model seal;
- call-free plan;
- service;
- ordinary primitive program;
- reusable value slots;
- token IDs;
- final F32 logits;
- inverse frequency;
- source-exact rotary owner;
- source-exact attention owner;
- GDN internal buffers;
- matrix packs;
- parameter owners;
- projection node map;
- embedding pack;
- status lease.

### 8.6 Resident body

ResidentProgramBody::OrdinaryCuda binds the cold owner and BLAS owner into the existing ResidentProgram.

It does not create a parallel runtime.

ResidentProgram execution handles include the exact ordinary accesses.

Legacy body accessors refuse ordinary bodies.

### 8.7 Request executor

The ordinary executor:

- authenticates request semantics/topology;
- validates typed prior/candidate state roots;
- requires original cold/root accesses on the existing physical flight;
- retains cold/BLAS owners;
- writes call-local token IDs/control data;
- traverses source nodes in order;
- uses owned projection enqueue;
- uses source-exact rotary;
- uses exact KV prefix carry + append;
- uses source-exact attention wrapper;
- uses generic GDN convolution/recurrent partitions;
- widens final BF16 vocabulary output into the existing F32 logits readback;
- returns the normal ResidentSubmission;
- leaves completion/readback/root promotion to the existing ResidentInference path.

There is no shadow token loop or alternate publication authority.

### 8.8 1–64 row ordinary execution

One materialized ordinary request may contain 1–64 rows.

Longer prefill is handled by repeated ordinary chunks at the higher layer.

Cached decode is an explicit SourceCallMode::Decode with one row and nonzero base/frontier.

### 8.9 Cancellation

Ordinary cancellation is checked:

- before candidate-root allocation;
- before queue/flight arming;
- between source nodes;
- after launch;
- after completion;
- immediately before commit.

Cancellation after the physical flight is armed does not attempt unsafe in-place GPU cancellation. Returning through the armed ResidentSubmission Drop path marks completion unknown, fences/quarantines the service and retains original owners until safe retirement.

### 8.10 Explicit shutdown

Ordinary model shutdown requires unique model ownership.

Order:

1. ensure no live sessions/model siblings;
2. retire owned cuBLAS provider/pool using the original service;
3. drop resident workspace/state-pool/cold owners;
4. run allocator cleanup;
5. verify resident bytes == 0;
6. verify model events == 0;
7. close model namespace.

Intentional namespace close is distinguishable from unknown physical failure so ResidentService Drop does not quarantine a cleanly closed device.

---

## 9. Canonical state transport

Canonical ordinary Qwen roots:

Conv:

- external canonical: F32;
- native GDN cache: BF16 layout;
- ConvIn validates exact BF16-widened finite F32 and converts/transposes;
- ConvOut restores canonical F32.

Recurrent:

- external canonical: F32 H/K/V;
- native runtime: F32 H/V/K transpose;
- RecurrentIn / RecurrentOut are exact transposition paths.

KV:

- canonical physical storage: BF16;
- prior active prefix is copied bitwise into the candidate root;
- current rows append into [base, base+rows);
- future tail remains untouched/zero by root lifecycle;
- attention widens only the active prefix;
- committed snapshot widens active BF16 to F32 and verifies exact +0 future tail.

---

## 10. Complete ordinary CUDA integration lineage after the 27 Oct baseline

The following sequence is important. Do not collapse it into “ordinary CUDA was added.”

From base 52bfb51 through current main:

- 901d23e — 27 Oct handoff checkpoint.
- ea66b3a — dynamic CUDA ordinary text primitives.
- 68393e5 — constrain ordinary text candidate.
- 6b4a2ec — close ordinary Qwen review blockers.
- 441fa64 — close primitive review gaps.
- ee67aa8 — close ordinary primitive review documentation.
- c2494ac — bind ordinary source invocations.
- 301e912 — generalize owned BF16 GEMM geometry.
- dcfbd74 — admit verified BF16 matrix upload.
- b116ccd — expose owned cuBLAS to ordinary CUDA.
- b4a9263 — ordinary projection owner.
- 40c8721 — bind primitives to shared owner status.
- 71cd737 — source-exact ordinary attention.
- e5f148e — projection reuse.
- 18579ff — explicit resident state precision.
- 9abd1d9 — typed resident state layout validation.
- 4fde757 — canonical GDN state transport.
- cba4f5e — partition ordinary GDN runtime schedule.
- 8e964ce — neutral ordinary GDN primitives.
- 8d5035d — ordinary parameter transport.
- 57379b3 — ordinary resource layout.
- 685e9be — matrix-pack reuse.
- 6f5d43d — cold source owner.
- 11aaf2b — ordinary resident body.
- 81318ad — expose cold values to body.
- 00ee779 — residual alias and GDN scratch ownership.
- 3865ffd — attention active-KV bound.
- c7b1962 — retain GDN local states.
- a8eef41 — active-view transport.
- d0f557a — execute ordinary through resident owner.
- 9edece3 — separate owned vs execution accesses.
- be5362e — load ordinary resident model.
- 1f69da6 — execute one resident chunk.
- b379a76 — Qwen qualification adapter.
- 8c24599 — close qualification build edges.
- c680b6c — explicit ordinary shutdown.
- eeb38bc — mark ordinary qualification-only.
- 5e29118 — context admission matrix.
- ec44357 — request cancellation.
- f2603b3 — canonical KV root snapshots.
- e93e7b4 — serial parity harness.
- 014b773 — runnable parity qualification.
- ceae923 — harden BLAS retirement.
- 9e56643 — harden BLAS activation failures.
- 487a796 — two-process parity isolation.
- ee70ec6 — source-exact ordinary rotary.
- 7842667 — rotary qualification record.
- ea4b67b — refuse legacy ordinary RoPE contract.
- 62ec2cb — isolate feature boundaries.
- 13703a4 — owned BLAS physical qualification harness.
- 9ae5e7e — provider blocker record.
- ccebfb0 — primitive numerical qualification.
- f6edcbb — primitive parity record.
- 642e245 — residual RMS qualification.
- ff512af — residual RMS record.
- 77dcafa — multiply and KV append qualification.
- 107e301 — multiply/KV record.
- 04fad2a — embedding qualification.
- 4274b19 — embedding record.
- f17540b — attention wrapper qualification.
- 7c67dd8 — attention wrapper record.
- 3eb3f74 — canonical state transport qualification.
- a6743c9 — state transport record.
- 8f4450c — fix cached GDN Gated launch grid.
- 8dc7cdf — generic GDN qualification record.
- ee5dfe1 — serial cached47 parity runner.
- c4f54eb — parity runbook record.
- b5c353c — parity restart/evidence hardening.
- d03bb3a — parity restart documentation.

All accepted implementation commits are signed.

The earlier 29b491d branch/review is historical only. Its useful work was independently reviewed, corrected and integrated through this lineage. Do not resurrect that worktree as current implementation.

---

## 11. Exact context admission matrix

Provider allowance:

    4608 MiB = 4,831,838,208 bytes

Physical envelope:

    5120 MiB

Exact cold + two-root ordinary totals after source-exact rotary:

Qwen .8B:

- 512: 1,652,916,305 B
- 1K: 1,667,661,905 B
- 2K: 1,697,153,105 B
- 4K: 1,756,135,505 B
- 8K: 1,874,100,305 B
- 16K: 2,110,029,905 B
- 32K: 2,581,889,105 B

All .8B cells fit admission.

Qwen 2B:

- 512: 3,911,845,969 B
- 1K: 3,926,591,569 B
- 2K: 3,956,082,769 B
- 4K: 4,015,065,169 B
- 8K: 4,133,029,969 B
- 16K: 4,368,959,569 B
- 32K: 4,840,818,769 B

2B 512→16K fit.

2B 32K exceeds the 4608 MiB provider envelope by:

    8,980,561 bytes

It must refuse before allocation.

These are admission/fit results only. They do not prove physical long-context correctness.

---

## 12. Source-independent physical qualification completed on RTX 3060

The following ordinary path surfaces have actual NVIDIA physical qualification.

### 12.1 RoPE

A clean physical test found a real defect in the old ordinary RoPE path:

- at position 1, query element 3:
  - ordinary legacy result: 0xc4dd
  - source-derived result: 0xc4dc
- difference: 1 BF16 ULP.

Cause: production ordinary used CUDA sincosf, which is not the source numerical schedule.

Fix:

- production ordinary RoPE no longer uses the legacy libdevice sincosf primitive;
- it uses the source-derived angle reduction, source trig/SLEEF schedule, source rotary apply and exact BF16 widen/pack transport.

Physical exact BF16 parity passes positions:

- 0
- 1
- 47
- 48
- 49
- 50
- 511
- 512
- 1023
- 2047
- 4095
- 8191
- 16383
- 32767

### 12.2 Full RMS

Exact BF16 source parity, zero shared status:

- rows 1
- rows 2
- rows 17
- rows 64

### 12.3 Residual RMS

Exact parity for:

- in-place BF16 residual update;
- normalized BF16 output;
- zero shared status;
- rows 1/2/17/64.

### 12.4 Query/key head RMS

Real packed attention layout:

- query: 8 × 256 groups;
- key: 2 × 256 groups.

Exact BF16 source parity rows 1/2/17/64.

### 12.5 SiLU and sigmoid

Ordinary FFN SiLU×up and attention sigmoid were compared to the pinned source SLEEF/transcendental schedule.

Exact BF16 parity rows 1/2/17/64.

### 12.6 Post-sigmoid multiply

Exact source-rounded BF16 multiply parity rows 1/2/17/64.

### 12.7 KV append

At base 31/capacity 512, rows 1/2/17/64:

- key/value payloads land exactly in [base, base+rows);
- all cache words outside the interval remain untouched sentinel values;
- status remains zero.

### 12.8 Embedding

Real .8B table geometry:

- vocabulary 248,320;
- hidden width 1024.

Exact BF16 gather rows 1/2/17/64.

Out-of-range token == vocabulary:

- output row is all zero;
- status is exactly STATUS_OUT_OF_RANGE.

### 12.9 Attention wrapper

The private production OrdinaryCudaAttention wrapper was tested with:

- authentic materialized .8B FlashAttention descriptor;
- base 31/capacity 512;
- rows 1/2/17/64;
- future KV tail deliberately filled with BF16 NaNs.

Results:

- wrapper widens only active prefix;
- future NaN tail does not leak into computation/status;
- per-row active causal frontier is base + row + 1;
- output matches an independently scheduled source-attention oracle bit-for-bit;
- status remains zero.

### 12.10 Canonical GDN state transport

Physical NVIDIA round-trip:

- ConvIn→ConvOut exact canonical bytes;
- RecurrentIn→RecurrentOut exact canonical bytes;
- zero shared status;
- zero resident bytes/events before namespace close.

Malformed test:

- one canonical conv F32 value with non-BF16 low mantissa bits;
- transport sets fail-closed status instead of silently truncating.

### 12.11 Generic GDN runtime

Physical zero-fixture execution:

- initial prefill rows 1;
- initial prefill rows 2;
- initial prefill rows 17;
- initial prefill rows 64;
- continued prefill frontier 64, rows 17;
- cached decode frontier 71.

Both convolution and recurrent partitions execute.

The final Gated buffer is deliberately poisoned before each case. Active output is overwritten to numerical zero; inactive tail remains poisoned.

This test caught another real bug described next.

---

## 13. Real defects caught and fixed during Oct 28 qualification

### 13.1 Source-inexact ordinary RoPE

See §12.1.

Old CUDA sincosf path was not bit-exact to source.

Production now uses source-exact rotary/trig.

### 13.2 Cached GDN Gated launch grid

Before fix:

- parameterized cached Gated launched grid [4,1,1];
- physical test wrote only 512 of declared 2,048 BF16 outputs.

Independent source-derived evidence:

- pinned cached PTX SHA:
  a97d127b124f14d226c2b24cff907e8c7aa260545e41743f84581acf6d510bbb
- PTX uses ctaid.x to select 128-BF16 output chunks;
- immutable historical asset metadata for the same PTX records grid [16,1,1].

Correct grid:

    [16,1,1]

Prefill grid remains source-derived rows/4.

A regression test pins:

- cached: [16,1,1]
- 47-token prefill: [188,1,1]

After fix, cached-decode zero-fixture qualification passes.

Do not revert this to a shared rows/4 formula.

---

## 14. Lifecycle, cancellation and fault ownership

Source-free/lower-layer lifecycle coverage now includes:

- ordinary explicit cancellation checks;
- armed-flight abandonment through ResidentSubmission Drop;
- delayed event reader source/quota retention;
- unknown module release fences siblings;
- independent close/reload;
- explicit provider retirement;
- model namespace close only after zero bytes/events;
- unique-model-owner shutdown requirement.

What remains artifact-dependent:

- cancellation after actual full-model ordinary launches with approved source/provider;
- real full-model provider/fault interaction;
- real model shutdown/retirement with approved source/provider loaded.

Those must run after cached47→50 exact parity and before production route activation.

---

## 15. Owned cuBLAS qualification and exact provider identity

An isolated physical qualification harness exists for the ordinary owned-cuBLAS path at dynamic rows:

- 1
- 2
- 17
- 64

It uses BF16 identity weights, exact output verification, resident flight/provider/pool ownership, explicit retirement and zero-ledger close.

The harness compiles and strict qualification,cuda-cublas Clippy passes.

Physical execution is currently blocked because the exact pinned provider DSOs are not connector-visible.

Pinned identities:

libcublas.so.12

- bytes: 105,140,976
- SHA256:
  d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048

libcublasLt.so.12

- bytes: 749,205,904
- SHA256:
  96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541

Historical validated path:

    /usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/

Those files are not visible through the current connector.

The available Nix CUDA 12.9.2.10 cuBLAS DSOs have different sizes/hashes.

Do not repin or substitute them casually.

---

## 16. Approved source identities and current artifact blockers

### 16.1 .8B approved source

Repository:

    Qwen/Qwen3.5-0.8B

Revision:

    2fc06364715b967f1860aea9cf38778875588b17

Weight shard:

    model.safetensors-00001-of-00001.safetensors
    bytes = 1,746,942,600
    SHA256 = 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696

Index:

    model.safetensors.index.json
    bytes = 50,900
    SHA256 = d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e

Tokenizer:

    SHA256 = 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42

Complete source census is checked in at:

    crates/zerogravity/src/qwen35_cuda/source08.json

### 16.2 2B approved source

Repository:

    Qwen/Qwen3.5-2B

Revision:

    15852e8c16360a2fea060d615a32b45270f8a8fc

Complete source census:

    crates/zerogravity/src/qwen35_cuda/source2b.json

Do not reuse .8B shape/fit/source assumptions for 2B.

### 16.3 Native bundle pins

Complete CUDA bundle migration pins:

    tools/qwen35-cuda-bundle/pins.json

The deterministic parity wrapper checks every non-provider .8B resource in that manifest by size/SHA before phase 1.

### 16.4 Current blocker

The current connector-visible target/models/qwen35-0.8b directory is not the approved full HF tree.

A real wrapper preflight with valid CUDA/NVRTC stops at:

    approved source member absent: .../target/models/qwen35-0.8b/LICENSE

This happens before GPU work.

The exact approved weight shard/provider/native bundle payloads must reappear. Do not download substitutes unless explicitly authorized.

---

## 17. Two-process cached47→50 exact parity campaign

Why two processes:

The historical reference path maps cuBLAS/native state differently from the ordinary owned-BLAS activation contract. Same-process reference→ordinary risks contaminating provider activation/ownership.

Therefore parity is intentionally split:

Phase 1:

- load authenticated historical current-source .8B;
- fixed 512 context;
- execute natural 47-token prefill then 48/49/50 continuation;
- capture full 248,320 logits;
- capture greedy selected ID/max bits;
- capture all 42 canonical state domains;
- verify reference KV future +0 and BF16-widened F32 semantics;
- explicitly shut down historical model;
- write bounded SHA-protected host evidence.

Phase 2 in a fresh process:

- independently authenticate source;
- independently authenticate provider paths;
- fixed 512 context;
- same fixed qualification prompt;
- ordinary path uses its own greedy token feedback after frontier 47;
- compare all logits bitwise;
- compare selected ID/max bits;
- compare all 42 domains bitwise;
- stop at first exact mismatch;
- explicitly shut down ordinary model.

Evidence never authorizes plan, prompt, context, provider or next token.

---

## 18. Parity evidence format and restart semantics

Evidence implementation:

    crates/zerogravity/src/qwen35_cuda/ordinary_parity/evidence.rs

Properties:

- explicit magic/schema;
- bounded file size <= 160 MiB;
- source identity/profile metadata;
- close/retirement receipt;
- exact 47-token prompt;
- exact frontiers 47/48/49/50;
- full 248,320 logits each step;
- exact sorted 42-domain state set;
- finite-only values;
- KV values must be BF16-widened F32;
- SHA256 over payload;
- trailing/unknown bytes rejected;
- symlink evidence rejected.

Publication was hardened on 28 Oct:

- create a sibling create-new temporary file;
- write/hash/flush;
- append digest;
- fsync file;
- atomically hard-link temp into create-new final evidence name;
- fsync directory;
- remove temp;
- fsync directory again.

A crash before publication cannot leave a partial final evidence file.

A second publisher cannot overwrite an existing final evidence file.

Restart mode:

    --compare-existing

This mode:

- requires an existing regular non-symlink evidence file;
- reruns full source/kernel/provider/CUDA/NVRTC/GPU-idle preflight;
- skips phase 1;
- runs only phase 2;
- evidence SHA/schema is revalidated before ordinary load.

---

## 19. Deterministic serial parity runner

Script:

    scripts/qualify-qwen35-ordinary-cached47-50.sh

Default mode:

- requires a fresh absolute evidence path;
- authenticates every artifact;
- requires idle NVIDIA compute queue;
- runs phase 1;
- requires phase-1 evidence publication;
- rechecks GPU idleness;
- runs phase 2 in a fresh process;
- rechecks GPU idleness.

Preflight-only mode:

    --preflight-only

Restart mode:

    --compare-existing

Required explicit environment:

    ZG_QWEN35_MODEL_DIR
    ZG_QWEN35_CUDA_KERNEL_DIR
    ZG_QWEN35_ORDINARY_PARITY_EVIDENCE
    ZG_QWEN35_CUBLAS
    ZG_QWEN35_CUBLAS_LT
    CUDA_PATH
    LD_LIBRARY_PATH

The script performs no acquisition, provider substitution, repinning, permission changes or GPU recovery.

It verifies:

1. complete source08.json file census;
2. complete .8B non-provider native bundle census from tools/qwen35-cuda-bundle/pins.json;
3. exact cuBLAS/cuBLASLt SONAME, size and SHA;
4. CUDA include view;
5. NVRTC visibility;
6. fresh or existing evidence semantics as appropriate;
7. idle GPU.

It then runs the existing integration tests:

    ordinary_cached47_to50_capture_reference_evidence

and

    ordinary_cached47_to50_compare_reference_evidence

from:

    crates/zerogravity/tests/qwen35_cuda.rs

Cargo commands use:

- --release
- --no-default-features
- qwen35-cuda-qualification
- separate test processes
- default CARGO_BUILD_JOBS=8
- default RAYON_NUM_THREADS=8

Release-mode cargo check of the exact integration-test surface passed.

---

## 20. Current validation state and exact next order

Current green source-free/host gates:

- ordinary qualification unit tests;
- parity evidence atomic publish test;
- parity comparison tests;
- strict qwen35-cuda-qualification Clippy;
- strict qualification,cuda-cublas Clippy where applicable;
- default/plain qwen35-cuda checks;
- all-features check;
- formatting;
- diff checks;
- release-mode qwen35_cuda integration cargo check;
- wrapper bash syntax/shell checks.

Current ordinary qualification library test result at handoff:

- 10 ordinary tests passed;
- 0 failed.

All-features currently emits three pre-existing probe-only dead-code warnings in source-page/diagnostic APIs. They are not ordinary-path failures.

The exact next sequence is:

1. Make exact approved .8B source tree visible.
2. Make exact pinned native bundle payloads visible.
3. Make exact pinned cuBLAS/cuBLASLt DSOs visible.
4. Build/use explicit CUDA_PATH + NVRTC environment.
5. Run parity wrapper --preflight-only.
6. Run full serial parity wrapper.
7. Require exact 47→50 logit + 42-domain parity.
8. If phase 1 succeeded but phase 2 needs retry, use --compare-existing.
9. Run after-launch cancellation/fault qualification with approved artifacts.
10. Run real-model explicit shutdown/retirement qualification.
11. Run .8B context correctness:
    - 512
    - 1K
    - 2K
    - 4K
    - 8K
    - 16K
    - 32K
12. Admit 2B only through exact fitted capacities; 32K remains refused under current envelope.
13. Run final assembled repository gate.
14. Only then run the serial matched vLLM campaign.
15. After benchmark closure, resume higher-order batching/paged-state/prefix-COW/speculation and larger-model work.

Do not benchmark before steps 7–13.

Do not activate the ordinary production route before whole-model parity + lifecycle + context qualification.

---

## 21. Broader subsystem status after the ordinary critical path

### CUDA

- finite historical qualification remains valuable evidence;
- ordinary source-derived path is integrated;
- source-independent physical hot-path qualification is broad and current;
- whole-model approved-source parity remains artifact-blocked;
- route remains qualification-only.

### CPU

- source-derived model/reference foundations and real router/queue tests exist;
- CPU evidence does not grant CUDA/HIP/Vulkan claims;
- original-source long-context/full-model campaigns remain separate.

### HIP/ROCm and Vulkan

- shared transport/ownership foundations exist;
- backend-specific qualification remains independent;
- do not inherit CUDA correctness;
- held local Renoir/gfx90c launch/recovery remains out of scope unless explicitly authorized.

### Serving and speculation

Existing serving/router/batching work is substantial.

Correctness-first order remains:

1. current ordinary model parity;
2. lifecycle/context qualification;
3. matched campaign;
4. then batching/paged state/prefix COW/speculation improvements.

Historical 2B C16 loss must remain visible.

### Memory and SSD

Virtual tensor/source-window/SSD foundations exist.

Do not claim trained streamed parity until source-real execution, eviction/reload and state correctness are independently qualified.

### Qwen3.8-27B

Original source/index/weights are authenticated.

Some operator/state work exists.

Full trained inference, serving, SSD execution and performance remain open.

### Gemma4

Separate source/model/backend candidate work exists.

Do not mix Gemma qualification with the Qwen ordinary CUDA critical path.

### CLI/device routing

The architecture supports separating model/source intent from physical device selection.

The user has explicitly wanted intuitive device selection, including exact Linux device-path support where appropriate, while still admitting required driver/control nodes such as ROCm/CUDA support nodes through the lower device logic.

This is a separate plan stream. Do not destabilize the current parity campaign to redesign CLI flags.

### Pipelines/multimodal

The long-term engine must support model pipelines and multimodal execution under the same ownership/scheduling fabric.

Typed DAG/config scaffolding exists, but no broad multimodal production claim follows from that scaffolding alone.

---

## 22. Historical failures and interpretations that must remain preserved

Never erase or relabel:

- case 5 FAIL101;
- six N1/B4 QKVZ BF16 differences at frontier 47;
- incomplete/failed capture parents;
- source proof without numerical proof;
- compile/reflection proof without execution proof;
- cleanup/process exit without physical completion;
- finite capture qualification as ordinary production authority;
- historical 2B C16 performance loss;
- container status 137 with unknown fatal/OOM cause;
- failed/interrupted controller scopes;
- old non-GDN candidates that were rejected for packed/strided/rounding/status/authority gaps;
- RoPE 1-BF16-ULP mismatch found on 28 Oct;
- cached GDN Gated 512/2048 overwrite bug found on 28 Oct.

These failures are useful design evidence.

---

## 23. Current benchmark policy

Do not run the requested benchmark campaign yet.

The existing campaign is specified in:

    .agents/plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md

When correctness gates close, the benchmark order remains:

1. .8B and 2B canonical 47-token workload, 8 natural output tokens.
2. C1/C4/C8/C16.
3. Ragged 47/57/62/67/72.
4. C32/C64 only after queue/admission/model-route gates.
5. Context ladder 512→8K, then 16K/32K.
6. TTFT/prefill and decode ITL/rate reported separately.
7. Balanced ABBA/BAAB repetitions.
8. Separate cold and warm campaigns.
9. One GPU owner/cell at a time.
10. Exact source/provider/device identities in every report.

vLLM historical baseline:

- version 0.31.0
- commit db9527a46873454610df6dbedf79a36d6bf1a7f6

Do not claim “matched vLLM” from historical single-wave numbers.

---

## 24. Validation commands and working cadence

Default engineering parallelism:

    export CARGO_BUILD_JOBS=8
    export RAYON_NUM_THREADS=8

Use narrow checks during implementation.

At substantial checkpoints run the owning tests plus strict Clippy.

Examples:

    cargo test --locked --offline --no-default-features \
      -p zerogravity --features qwen35-cuda-qualification \
      ordinary_ --lib

    cargo clippy --locked --offline --no-default-features \
      -p zerogravity --features qwen35-cuda-qualification \
      --all-targets -- -D warnings

    cargo check --locked --offline -p zerogravity --all-features

    cargo fmt --all -- --check
    git diff --check

Release integration surface:

    cargo check --locked --offline --release --no-default-features \
      -p zerogravity --features qwen35-cuda-qualification \
      --test qwen35_cuda

For actual physical model/GPU campaigns:

- run serially;
- require idle GPU;
- use explicit CUDA/NVRTC/provider environment;
- inspect result and cleanup markers;
- never infer success from process exit alone.

---

## 25. Commit and worktree policy

Current expectation:

- one main worktree only;
- no temporary worktree unless a new isolated scope truly needs it;
- merge accepted work back promptly;
- remove finished clean worktrees;
- preserve unrelated untracked user material;
- signed scoped commits only;
- no push unless requested.

The previous instruction to preserve a registered prompt-reference worktree is obsolete as a current worktree rule: its relevant recipe overlay was copied into main as tools/cuda-gdn-recipes and the worktree was removed. Preserve the overlay itself.

---

## 26. Completion definitions

Ordinary Qwen CUDA is not production-complete until all of these hold for the exact admitted source/profile:

- source-derived plan/lowering only;
- exact authenticated source lifetime;
- exact whole-model logits parity;
- exact whole-model 42-domain state parity;
- genuine cached continuation;
- cancellation/fault/quarantine safety;
- exact provider retirement/shutdown;
- exact resource fit;
- context correctness for the claimed capacities;
- serving route activation only after those gates.

The benchmark campaign is not complete until:

- every authorized cell has complete per-run and aggregate output;
- cold/warm are separated;
- TTFT/prefill and decode are separated;
- actual active/queue/cohort behavior is reported;
- failures remain failures;
- all model/provider/device owners close cleanly.

A project-wide model/backend claim always requires independent source, numerical, physical, whole-model, lifecycle, serving and performance qualification for that exact cell.

---

## 27. Exact handoff point

The next agent should NOT resume primitive implementation.

The ordinary CUDA implementation is integrated.

The next agent should NOT start vLLM timing.

The next agent should NOT begin long-context performance work.

The next agent should first revalidate current main and artifact availability.

If the approved source/provider/native bundle are still missing, report the exact missing artifact and stop at that boundary.

If they are available:

1. run scripts/qualify-qwen35-ordinary-cached47-50.sh --preflight-only;
2. run the full script;
3. inspect exact parity and shutdown;
4. continue the ordered lifecycle/context gates.

If phase 1 evidence already exists from a successful capture:

    scripts/qualify-qwen35-ordinary-cached47-50.sh --compare-existing

Do not delete or replace evidence casually.

---

## 28. Starting prompt

The copyable prompt is stored separately at:

    .agents/docs/NEXT_AGENT_PROMPT_2026-10-28.md

It is also repeated at the very end of this file after the historical appendix.

---

# Historical appendix — complete 27 Oct handoff

Everything below this point is the 27 Oct comprehensive handoff retained as historical context.

Important: where it says ordinary CUDA is unavailable, the pending 29b491d branch is unintegrated, two worktrees exist, or non-GDN ownership is missing, that is historical and superseded by sections 0–28 above.

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


---

# End historical appendix


---

# Current copy-paste next-agent prompt — 28 Oct 2026

# Next agent prompt — 28 Oct 2026

Work in /home/pvl/spaces/abird/src/zerogravity and continue only the non-Fabric ZeroGravity execution workstream.

Read, completely and in order:

1. AGENTS.md
2. .agents/plans/README.md
3. .agents/plans/TODO.md
4. .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-28.md
5. .agents/docs/handoff-2026-10-26/USER_DECISIONS.md
6. .agents/docs/handoff-2026-10-26/DESIGN_AND_DECISIONS.md
7. .agents/docs/handoff-2026-10-26/EVIDENCE_AND_CONTINUATION.md
8. .agents/docs/handoff-2026-10-26/OPEN_WORK.md
9. .agents/docs/handoff-2026-10-26/SOURCE_CATALOG.md
10. .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md
11. .agents/plans/model-execution/TODO.md
12. .agents/plans/numerics-and-context/TODO.md
13. .agents/plans/cpu-gpu-optimization/TODO.md
14. .agents/plans/serving-and-speculation/TODO.md
15. .agents/plans/memory-and-streaming/TODO.md
16. .agents/plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md

Before changing anything run:

    export CARGO_BUILD_JOBS=8
    export RAYON_NUM_THREADS=8
    export PYTHONDONTWRITEBYTECODE=1
    git status --short --branch
    git log --show-signature --oneline -30
    git worktree list --porcelain
    nvidia-smi

Current truth:

- work only on main unless a genuinely new isolated scope requires otherwise;
- only the main worktree should be registered;
- ordinary source-derived Qwen3.5 .8B/2B CUDA is integrated but qualification-only;
- one ForwardPlan, one ResourceArbiter, one queue/flight/completion/quarantine path and the existing StateRoot authority must remain the only execution/publication authority;
- source-derived execution must never be selected by prompt IDs/lengths, call/node ordinals, packet contents, capture tables or recipe/catalogue hashes;
- the source-independent hot path has already been physically qualified on the RTX 3060: source-exact RoPE, all RMS variants, SiLU/sigmoid/multiply, KV append, .8B embedding, attention wrapper, canonical GDN state transport and generic GDN initial/continued prefill + cached decode;
- two real defects were already found and fixed: source-inexact legacy RoPE and cached GDN Gated grid [4,1,1] writing only 512/2048 outputs; cached Gated is now [16,1,1];
- exact admission currently fits .8B through 32K and 2B through 16K; 2B 32K must refuse before allocation;
- do not redo primitive work unless a new failing qualification proves it necessary.

Immediate next goal:

1. Revalidate main and exact artifact availability.
2. If the exact approved .8B HF source tree, pinned native bundle payloads and pinned cuBLAS/cuBLASLt DSOs are available, run:

       scripts/qualify-qwen35-ordinary-cached47-50.sh --preflight-only

3. If preflight passes, run the same script without a mode. It must execute historical reference capture first and ordinary comparison in a fresh second process.
4. Require exact cached 47→48→49→50 parity for all 248,320 logits, selected token/max bits and all 42 canonical state domains.
5. If phase 1 succeeded but phase 2 must be retried, use:

       scripts/qualify-qwen35-ordinary-cached47-50.sh --compare-existing

6. After exact parity, run after-launch cancellation/fault, retirement and real-model shutdown qualification.
7. Then run .8B physical context correctness 512/1K/2K/4K/8K/16K/32K; admit 2B only where the exact two-root admission fits.
8. Run the final assembled repository gate.
9. Only then run the one serial matched vLLM campaign in the benchmark plan.

Current artifact blockers recorded at handoff:

- approved Qwen/Qwen3.5-0.8B revision 2fc06364715b967f1860aea9cf38778875588b17;
- weight shard model.safetensors-00001-of-00001.safetensors, 1,746,942,600 bytes, SHA256 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696;
- pinned libcublas.so.12: 105,140,976 bytes, SHA256 d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048;
- pinned libcublasLt.so.12: 749,205,904 bytes, SHA256 96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541;
- the connector-visible target/models/qwen35-0.8b tree and current Nix cuBLAS build are not substitutes.

If those exact held artifacts are still unavailable, report the exact missing identity and stop at that boundary. Do not download, repin or substitute them. Do not skip ahead to context timing or vLLM benchmarks.

Preserve:

- root README bytes;
- unrelated untracked Fabric vNext files/plans;
- tools/cuda-gdn-recipes;
- all historical failure records;
- independent StateRoots;
- exact source/provider identities;
- signed scoped commits.

Do not run Fabric vNext, push, tag, publish, change device permissions/recovery, or launch held AMD hardware. Use at least 8 Cargo jobs by default. Keep physical GPU work serial. Independently inspect any delegated work before integrating it. Report substantial verified checkpoints with current goal, what is complete, blockers and the exact next step.
