# DwarfStar optimization research for one Rust execution fabric

Author: `@dwarfstar_optimization_research`
Date: 2026-10-21
Parent: execution-fabric research; coordinator owns shared plans.
Scope: primary text/source inspection only; no builds, foreign engine execution,
GPU/model runs, model/SDK/executable downloads or device changes.

## 1. Decision

- **Keep one Rust mathematical engine.** DwarfStar is evidence for narrow model
  graphs, shape-specific kernels, bounded residency and explicit session state;
  it is not a proposed second backend process, allocator or token loop.
- **Immediately after vLLM numerical/serving parity:** profile the Qwen .8B/2B
  critical path, then qualify same-source fusion, shape/row-count dispatch,
  amortized packing, shared scratch, fewer state/readout transfers and bounded
  prefill/decode interleaving. Measure short requests and long continuations.
- Follow with **original dense Qwen3.8 27B** source-window execution and admitted
  SSD/layer overlap, then Gemma4 E2B/26B/31B. The 27B checkpoint is dense:
  MoE hot-expert caching cannot skip its required matrices. Expert locality and
  learned n-gram/Engram tables chiefly apply to later sparse models.
- DeepSeek CSA/HCA/mHC, GLM KDA, Qwen Flash-Next QSA/MoE/PLE and hardware-specific
  M5/gfx1151/GB10 techniques remain deferred until their source-real model and
  device gates. These are model semantics or capability specializations, not
  interchangeable shortcuts for current Qwen.
- Optimize **correct emitted SLO-goodput**, TTFT/ITL tails, fairness, physical
  memory and energy. No universal outperformance claim follows from this study.

## 2. Authority, pin and limits

- Canonical repository: [antirez/ds4](https://github.com/antirez/ds4), GitHub API
  `fork=false`, default branch `main`; read on 2026-10-21.
- Exact source: **`5379c1c9ac315f1540371b06a3db88491d4059ff`**,
  committer date **2026-10-19 10:55:24 UTC**. Subject: download script's default
  changed to the new DS4.1 Flash checkpoint. [Commit][pin]
- API tags and releases were both empty. This is a pinned source revision,
  **not a numbered release**. [Small machine-readable pin](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/execution-fabric-dwarfstar-research-20261021/UPSTREAM_PIN.json).
- The historical local atlas inspected `fc80bd695da76ee14bcb86820bf900a5cdfda806`
  (2026-10-07). It and the Strix dossier are historical leads only. Current
  code/QA includes subsequent V4.1 DSpark/TP, router and causality fixes.
- Fetched 85 selected text/doc/source files, 13,428,204 bytes, from this exact
  revision; inspected the referenced docs, key source branches and tests; excluded model weights, images, large prompt datasets and binaries.
  The ephemeral source cache is `/tmp/zg-ds4-source-research-20261021`.
  GitHub metadata/history/issues were read separately. A symbol/branch inventory
  is not whole-repository formal verification; uninspected shapes/backends remain
  unknown. Every physical/quality/timing result below belongs to upstream.
- Upstream contains C, Objective-C, CUDA/HIP and Metal implementations with
  selected GGML-derived pieces; it does not link the complete GGML runtime.
  Flat/global shape, worker and backend state make the API an instructive seam,
  **not evidence for simultaneous arbitrary model instances**. [API][api],
  [core][core], [MMQ provenance][vendor].

## 3. Zerogravity evidence boundary

Read current ISO plan rulebook, queue, CPU/GPU, serving, memory, model and
distributed TODOs plus the live source cited below. ISO HEAD is
`fb68d8fa165de16e21d1bd37afef3790b80da971`; extensive pre-existing dirty source
is preserved. Root README blob remains
`a7be69c35cf61d14e8810f9cb5a6a43d1397bf27`.

Evidence labels:

- **B — bounded tested:** existing source plus explicitly recorded bounded
  qualification; tests were inspected, **not rerun by this research**.
- **I — isolated:** active ISO source/parent-supplied fresh result; MAIN or public
  serving promotion is not implied.
- **C — current code:** inspected implementation, no broader qualification claim.
- **P — plan:** live TODO/contract; no implementation inferred.
- **A — absent:** no applicable implementation found in inspected runtime paths.
- **N — inapplicable:** model/hardware/source semantics do not apply to this slice.
- **U — unknown:** insufficient authoritative evidence; cannot promote.

Current parent-supplied result supersedes dated prose: .8B original-source forced
five phases `1/1/64/2/1`, full24 layers/all42 domains/full vocabulary/five greedy
outputs, fault/regression and complete `scripts/check.sh` gate17 PASS **ISOLATED**. Its
[qualifier][z-native] explicitly owns no natural Ready/root publisher or serving
capability. Original stock2B vLLM natural five-phase/all42/24-boundary/full-V paired
reference passes; native original2B remains open. Native GGUF .8B/2B B2/4/8,
row-state, teacher-prefix and private graph scopes are bounded tested; temporal
prefill, paged full-prefix, production graph admission and broad vLLM performance
remain open. Original 27B cached bytes and source windows are not trained runtime
qualification. Gemma4 modules explicitly expose contracts, not numerical models.

Local implementation register (also used in every inventory row):

| Key | Actual seam/source/test | Evidence and scope |
|---|---|---|
| Z1 | [single model traversal][z-forward], [typed program][z-program], [typed tests][z-typed-tests] | B/C; one `MathematicalBackend`/recorded `ForwardPlan`; profiles/casts/source roles are explicit. |
| Z2 | [immutable tensor source][z-source], [original owned windows][z-window], [window tests][z-window-tests], [SSD extents][z-ssd], [extent tests][z-ssd-tests] | B/I source-only and bounded operator binding; windows retain descriptors/identity/host charges. No production async streamed model/eviction. |
| Z3 | [ResourceArbiter][z-arbiter], [fair queue][z-queue], [physical service][z-service], [leases][z-leases], [quarantine][z-quarantine], [worker tests][z-workers] | B/C; one process-wide physical authority, actual completion ownership and unknown-work quarantine. Global host/UMA/OS accounting remains P. |
| Z4 | [SessionAuthority][z-session], [all-domain StateRoot][z-state], [Qwen checkpoints][z-checkpoint], [resident checkpoints][z-resident-checkpoint] | B/C bounded committed snapshots/rollback; portable diagnostic checkpoints are not product full-prefix/disk persistence. |
| Z5 | [source quantization][z-quant], [CPU math][z-cpu], [batched source projection][z-batch-proj], [activation fusion][z-activation], [resident program][z-resident-program] | B/C original codec/source dots and bounded fusion; no blanket tensor-core/native-F32/fast-libm permission. |
| Z6 | [serving wave formation][z-wave], [original cohort publisher][z-cohort], [teacher-prefix qualifier][z-prefix], [batched program][z-batch-program] | B .8B/2B GGUF B2/4/8, independent frontiers/refill/readout barriers. Broad temporal prefill/paging/fair mixed scheduling P. |
| Z7 | [private graph owner][z-graph], [batch replay body][z-graph-body] | B private qualifier only; production constructors, metadata admission, events/storage/drop participation P. |
| Z8 | [full .8B original-source qualifier][z-native], [native faults][z-native-faults], [continuation][z-continuation] | I; source-produced five phases use original workspace, queues, leases; natural publication/serving/open2B not established. |
| Z9 | [Gemma family][z-gemma], [Gemma MoE contract][z-gemma-moe], [model TODO][z-model-plan] | C contract, P trained CPU/GPU runtime. DeepSeek/GLM/Flash-Next executable support A. |
| Z10 | [serving/speculation TODO][z-serving-plan], [memory TODO][z-memory-plan], [optimization TODO][z-opt-plan], [distributed TODO][z-dist-plan], [pipeline structure][z-pipeline] | P for full-state prefixes, speculative target verification, SSD model execution, distributed inference, multi-model pipelines. Pipeline checking is pure scaffolding. |

## 4. Per-optimization inventory

Classification: **E** inference-preserving candidate (must still pass the chosen
source numerical profile); **M** learned model/checkpoint semantics; **Q** quality,
distribution or arithmetic-profile change requiring separate explicit identity
and acceptance. A source-derived pack may be E; requantization is Q.

### Model graph, numerical identity and CPU

| ID | Mechanism in pinned DwarfStar | Class; fabric stage | Actual Zerogravity mapping and transfer condition |
|---|---|---|---|
| D01 | Narrow `engine` for loaded weights and mutable `session` for full-prefix sync/eval/rewind; CLI/server avoid tensor internals. [API][api] | E; source/IR/control | Z1/Z4 B/C; preserve same TaskCoordinator/SessionAuthority. General multi-model shared process is Z10 P; avoid copying DwarfStar globals. |
| D02 | Dedicated model graphs and explicit unsupported-feature refusal; Qwen restricted to Metal/single CUDA (CPU first-token diagnostics), no TP/PP/expert streaming. [Qwen][qwen], [open guards][guards] | M/E; source/IR/placement | Z1 B and Z9 C/P; reuse typed schedule and capability refusal. Flash-Next is A/N for current .8B/2B. |
| D03 | CPU row worker pool persists across calls; small/nested work stays serial; ARM NEON/DOTPROD quantized dots. [pool][pool], [core][core] | E; scheduler/kernels | Z5 C/B scalar/profiled math; x86 capability kernels and topology tuning Z10 P. No inspected x86 AVX/VNNI path in DwarfStar core to port. Pool construction must not duplicate global fair workers. |
| D04 | Source GGUF codecs, custom mixed expert packs, imatrix IQ2/Q4, native MXFP4 and padded down rows; verify actual recipe rather than bpw label. [conversion][conversion], [Qwen][qwen], [imatrix][imatrix] | E for layout; Q for quantization; source/numerics | Z2/Z5 B/C source codec identity; derived quality contract Z10 P. Preserve source hashes, calibration and original tensor axes. Q2/Q4 gains are not engine speed comparisons. |
| D05 | `--quality`/strict controls restore selected reference paths; official continuation/NLL/top-token gates complement kernel oracles. [QA][qa], [quality][quality] | E/Q; numerics/measurement | Z1/Z8 explicit numerical contracts, independent all-V/all-domain comparison B/I. Upstream tolerance or coherent output cannot replace local raw-word equivalence requirements. |

### Kernels, fusion, layouts and dispatch

| ID | Mechanism in pinned DwarfStar | Class; fabric stage | Actual Zerogravity mapping and transfer condition |
|---|---|---|---|
| D06 | Regime-specific GEMV for decode/few verifier rows versus tiled GEMM/MMQ for large prefill; tails and shape guards select fallback. [CUDA dispatch][cuda-dense], [Qwen CUDA][qcuda], [Metal dense][metal-dense] | E candidate; kernels/scheduler | Z5/Z6 B/C source dots and finite bands; broad per-shape tuner P. Benchmark B1/few-row/large-N thresholds together; retain identical casts/reduction or name a separate profile. |
| D07 | Fused dequant-MMQ lifts GGML CUDA kernels through a small ABI; routed IQ2 gate/up + Q2 down, Q8 dense, weighted/clamped SwiGLU/slot reduction. [MMQ integration][mmq], [vendor][vendor] | Q arithmetic candidate; kernels | Z5 B original dot/fusion, generic MMQ family A/P. Reimplement within Rust-owned IR/leases; no GGML allocator/graph. Input Q8_1 quantization and changed reductions need source-aware acceptance. |
| D08 | Aligned SoA/D2R and TP expert repacks reduce reload/address overhead; source bytes/codecs unchanged. Actual later call sites extend the older raw-tier comment. [dense call][cuda-dense], [repack][repack], [D2R][d2r] | E layout candidate; residency/kernels | Z2/Z5 C/B raw source ownership; amortized derived pack P. Key by exact source/codec/layout/device/version; charge pack plus retained original, tails and startup. Producer-fold hook currently always misses in main. |
| D09 | Qwen dense tensor-core arithmetic splits operands into leading/fine components and accumulates correction separately; F16 paths use scaling/residuals to protect range. [implementation][qcuda-matrix] | Q numerical candidate; numerics/kernels | Z1/Z5 profile-specific operations; this algorithm A. SM86 instruction availability is insufficient: signed-zero/subnormal/RNE/cast/reduction/all-V/state gates first. No exactness inferred from residual correction. |
| D10 | Qwen fused Q/K norms+RoPE/cache stores, HC gate/mix/injection and residual operations reduce passes; normalization reused across injection chunks. [Qwen CUDA][qcuda], [history][history] | E candidate/M math; IR/kernels | Z1/Z5 B/C typed residual/RMS, GDN/rope source operations; Z8 I BF16 profile. Fuse only semantically adjacent nodes with exact rounded intermediates and shared ownership. HC is N for current dense Qwen variant. |
| D11 | Qwen GDN register scan holds state rows through a chunk, emits per-token output and snapshots requested intermediate tokens; convolution history snapshots align with token boundary. [scan][qcuda-gdn] | M/E; kernels/state | Z1/Z8 B/I GDN fragments and five-phase path; full temporal-prefill support P. Do not substitute this FP32 scan for pinned BF16/FLA chunk64 arithmetic; independently gate every retained root. |
| D12 | Grouped Qwen attention shares K/V reads across query heads, splits long decode over keys, merges stable partial softmax; larger prefill uses residual-corrected tensor-core products. [attention][qcuda-attn] | E/Q candidate; kernels | Z5/Z6 bounded QK/AV batching B; long full-prefix/paged attention P. Gate causal/tail/masked selections, head-group mapping, extreme scales and same-profile full logits. |
| D13 | Sparse selector uses radix threshold + stable tie gather, then exact causal block expansion; skips sorting when all candidates retained. [Qwen selector][qcuda-select], [skip-sort commit][skip-sort] | E for fixed semantics; kernels/readout | Z5 deterministic CPU top-k C, executable Flash-Next selector A. Reuse as typed primitive only after learned selection rules are sourced; ties/signed zero and boundary coverage mandatory. |
| D14 | CUDA small-row Q8/shared/routed expert kernels, direct-down sums, compact row layouts and batched compressor state updates reduce memory/launch cost. [CUDA][cuda], [compressor][cuda-compressor], [QA][qa] | E/Q candidate; kernels/state | Z5/Z6 B source batching/fusion; DeepSeek-specific compressor/MoE A/N now. Transfer row-count specialization and prefix snapshots; do not conflate dispatch counts with physical launch enumeration. |
| D15 | Side-stream shared-expert/Q8 work overlaps independent main-stream work with ready/done events and separate handles/scratch; disabled during capture or incompatible shared scratch. [side regions][cuda-side] | E; scheduler/completion | Z3 B actual single-queue lifetime; concurrency/multi-stream P. Charge every buffer/event, avoid cross-stream alias, test partial launch errors and actual completion before reuse. |
| D16 | M5 Metal TensorOps/NAX dense and MoE paths, packed routed tiles, tile width/K-split choices and double-buffered staging; conventional paths remain guarded. [Metal dense][metal-dense], [MoE][metal-moe], [QA][qa-m5] | E/Q candidate; kernels/residency | Z5 CubeCL/native qualified profiles B; Metal backend/M5 path A/N for local SM86. Portable idea: shape-selected packing/staging; no hardware performance extrapolation. |
| D17 | gfx1151 IQ2 MMQ x64/y64, wave32 rocWMMA Q8/F16 tiles, 32-head/80-key attention, vectorized staging/Q2 slab reuse. [recorded campaign][strix-bench], [ROCm matmul][rocm-mm], [ROCm MoE][rocm-moe] | E/Q candidate; kernels | Z5 accepted bounded backend profiles; optimized gfx1151-specific family P/A. gfx90c is N and remains launch-held. Upstream validation allows numerical envelopes against an earlier stack, not universal exact logits. |
| D18 | hipBLASLt shape/solution cache requires exact library version/revision; unsupported versions use fallback. [guard][hipblaslt] | E/Q; capability/dispatch | Z1/Z5 numerical profiles B/C; robust per-library selector P. Keep compiler/driver/libm fingerprint and exact shape/call schedule in qualification key. |
| D19 | CUDA decode-island graphs: eager warmup, capture/instantiate next encounter, then pointer-keyed replay; failure marks dead and re-encodes eager. Position-dependent work/network gates stay outside. [graphs][cuda-graphs] | E; IR/scheduler/completion | Z7 B private graph qualifier, production admission P. Transfer static islands/guards without bypassing original queue, metadata accounting, all reader leases or quarantine. A persistent all-model GPU loop is not established here. |

### Model-specific long-context semantics

| ID | Mechanism in pinned DwarfStar | Class; fabric stage | Actual Zerogravity mapping and transfer condition |
|---|---|---|---|
| D20 | DeepSeek V4 CSA/HCA mixes raw local window, learned compressed history and sparse indexer; mHC and source-defined FP8 cache/index round trips are part of math. V4.1 changes graph/ratio/state. [core][core], [models][models], [graph API][gpu-api] | M; source/IR/numerics/state | Z9 A/P, N for Qwen/Gemma. No generic KV compression substitute. Require official tensors, compressor boundary/tie/masking and all-state rollback references per checkpoint. |
| D21 | GLM MLA/DSA and GLM5.3 KDA have compact KV/recurrent layouts and model-specific dense→sparse boundary at 2051 visible tokens. [KDA kernel][kda], [ROCm GLM][rocm-glm], [QA][qa] | M; source/IR/state | Z9 A/P. Existing Qwen GDN math is not GLM KDA. Do not transplant old dense-to4096 benchmark dispatch into correctness claims. |
| D22 | Qwen Flash-Next gated GQA/QSA, GDN, hyperconnections, MoE, learned PLE/n-grams and MTP are one checkpoint's graph. Optional static YaRN changes context behavior/quality. [Qwen][qwen] | M; optional YaRN Q; source/state | Z9 A/P; current .8B/2B GDN source work B/I does not qualify Flash-Next. Separate target model support, extension policy and optional speculation. |

### Memory, SSD, prefetch and immutable readers

| ID | Mechanism in pinned DwarfStar | Class; fabric stage | Actual Zerogravity mapping and transfer condition |
|---|---|---|---|
| D23 | Fit non-routed weights/context/scratch/prefill reserve first, then size expert cache; resident is preferred when it fits. Sessions/context reduce cache headroom. [SSD guide][ssd], [planner][ssd-code] | E; placement/residency | Z3 physical ledger B, host/UMA end-to-end P. Admission must include original + derived weights, duplicated staging, graphs and all concurrent roots; never equate GPU addressability with RAM. |
| D24 | Fixed expert slabs avoid per-expert VM/buffer churn; dominant single-size class excludes larger boosted mixed-quant layers from that cache. [geometry][slab-class], [Metal cache][metal-cache] | E; residency | Z2 source windows B/I; actual expert slab allocator A/P. Use planned mixed-codec size classes under one byte cap, not one giant uniform slot or a second hidden arena. |
| D25 | Route hotness/frequency decay + age, layer/global budgets and model hotlists seed useful experts; selected/current readers protected from eviction. [Metal eviction][metal-evict], [core profiles][core] | E; residency/scheduler | Z10 P; only source-only extent/window staging is B. Predictions may change residency, never router results/expert count. Test hot/cold transitions, adversarial requests and fairness. |
| D26 | Cache generation tracks encoded, owned and committed command buffers, protects in-flight slots and waits before safe reuse. [sequence tracking][metal-seq], [victim protection][metal-evict] | E; completion/residency | Z3 B leases/events/quarantine; source-neutral cache/ABA Z10 P. Integrate encoded-but-unsubmitted/captured readers, not merely submitted event refs; unknown completion must fence storage. |
| D27 | Metal selected-load worker/pread pool overlaps missing-expert reads with resident/shared work; address tables split cached/missing experts without changing routing. [load admission][metal-load], [cache][metal-cache] | E; residency/kernels | Z2 B immutable source materialization; async readiness/consumer binding A/P. Private pending bytes become visible only after complete validated read and genuine copy completion. |
| D28 | CUDA next-layer prefetch uses one reader, two bounded host staging buffers, 8MiB chunks and H2D events; read/file order coalescing; foreground alone publishes cache index. [prefetch][cuda-prefetch] | E; residency/completion | Z2 source extents/windows B/I, actual async device source A/P. Preserve descriptor/source hashes, reader leases and cancellation join. Dense27B needs layer/window planning, not sparse-expert routing. |
| D29 | O_DIRECT with alignment/EOF handling and buffered fallback; buffered read advisory eviction limits page-cache pollution. [CUDA read helper][cuda-read], [prefetch][cuda-prefetch] | E; source/IO | Z2 positional verified reads B, direct/async/platform IO P. Optional mode cannot change bytes or weaken short-read/hash/error semantics. Measure read amplification and actual page-cache/host effects. |
| D30 | Read-ahead slots enter as low-priority predictions; actual router reuse promotes them, so prefill/prefetch cannot masquerade as demand hits and evict decode-hot working set. [publish policy][cuda-prefetch] | E; residency/scheduler | Z10 P. Distinguish demand hits, useful prefetch, unused bytes and hot-set displacement in counters; benchmark subsequent decode alongside prefill gains. |
| D31 | Qwen 95.37GiB original BF16 n-gram table stays outside maps/residency; bounded sorted/deduplicated row reads preserve caller order. [row reader][qwen-rows], [Qwen][qwen] | M/E; source/residency | Z9 A/P learned table; Z2 reusable bounded read foundation B/I. No source requantization or expansion of old quantized tables presented as restoration. Not runtime n-gram speculation. |
| D32 | V4.1 Engram FP8 code/scale→BF16 lookup table, bounded batch reads, duplicate reuse and a persistent four-worker single-token reader; per-table completion permits overlap. [Engram][engram] | M/E; source/kernels/IO | Z9 A/P, Z2 B foundation. Preserve FP32 scale/BF16 rounding, model hash/history state and error invalidation; no mandatory IO for models with no learned table. |
| D33 | GPU work submitted before Qwen decode/batch n-gram reads hides some disk latency; V4.1 CUDA reader overlaps disk n-grams with target decode. [batch gather][qwen-batch], [Engram][engram], [history][history] | E; scheduler/residency | Z3 fair workers B; real async IO/compute overlap P. Separate CPU staging/SSD credits, no lock held across wait and no allocation per row without admission. |
| D34 | Non-routed weights stay resident; skip warming disk-only tables/unowned TP shards; exact mapped range views avoid whole-file pinning. Linux nonmovable memory excludes CMA. [CUDA][cuda], [Metal][metal], [Linux memory][linux-memory] | E; placement/residency | Z2/Z3 B boundaries; comprehensive physical host/UMA planning P. OS-visible source mapping, pinned memory and device views may overlap—count physical backing once. |
| D35 | Shared per-engine prefill workspace lets additional sessions retain caches instead of several GiB scratch; idle arena can grow for wider sessions. [server][server], [Qwen batch][qwen-batch] | E; residency/scheduler | Z6 B shared model/band workspace, Z3 actual charge; multi-model reusable scratch policy P/C. Reuse only when source/layout/stream/generation and active-use exclusivity are proven. |

### Serving, prefix identity and state persistence

| ID | Mechanism in pinned DwarfStar | Class; fabric stage | Actual Zerogravity mapping and transfer condition |
|---|---|---|---|
| D36 | Native session batching shares weight work while recurrent/KV/PLE histories stay independent; batch size one follows ordinary eval. Backend/model/fixed-frontier limits select ordered fallback. [API][api], [server][server], [batch code][qwen-batch] | E/Q arithmetic; scheduler/state | Z6 B .8B/2B B2/4/8 independent roots/frontiers. Broad model/backends P. Verify companion isolation/reordering/full-state equality; concurrency alone is not native batching. |
| D37 | Bounded prefill quantum yields to decode, slot round robin and decode coalescing; some Flash CUDA layouts mix prefill+decode natively, others serialize. [scheduler][server-scheduler], [mixed code][cuda-mixed] | E; scheduler | Z3/Z6 B finite source-aware waves/fair workers, full aged temporal prefill/overlap P. Measure starvation, long-prompt admission, cancel/output-backpressure and true ITL, not mean tokens/s alone. |
| D38 | Full-prefix sync uses live checkpoint when exact tokens match; edited history rebuilds; images require positions/fingerprints. [API][api], [server][server] | E; state/control | Z4 B immutable per-session roots/snapshots; Z10 full-domain prefix/COW P. Key source/numerics/template/tokenizer/adapter/tenant and all state domains, not token count or KV pointer alone. |
| D39 | Disk KV store uses rendered-text prefix keys plus exact token/graph payload, size/ABI/model/context checks, budgets, continued/cold/anchor save policy and hit decay. [store header][kv-header], [store][kv] | E; state/persistence | Z4 B diagnostic checkpoint, Z10 product disk cache P. DwarfStar's optional cross-quant reuse is incompatible with ZG exact immutable source default; require exact source/profile identity. |
| D40 | Tool DSML spans/IDs, visible versus reasoning history and text/image prefixes preserve replay identity across API round trips; bad boundaries can cause full prefill or stale-history reuse. [server][server], [open tracker][issue1201] | E; control/state | Z10 P broad OpenAI/Anthropic/tools/router behavior. Bind authenticated input frontier; canonical JSON equivalence is not token/source equivalence. Apply updated history before accepting tool-ID reuse. |
| D41 | Committed session payload includes model recurrent/history/index state and valid frontier logits; interrupted partial layer-major work invalidates; TP restores often replay exact tokens on both ranks. [API][api], [Qwen][qwen], [distributed][distributed] | E; state/completion | Z4 B committed checkpoint/rollback; Z8 I multi-phase state; full production persistence P. Restore cannot synthesize local SourceIdentity/Ready or omit recurrent/draft/RNG/grammar/media state. |
| D42 | Cancellation/disconnect removes queued/pending work but preserves worker ownership of running steps; cooperative sync drains before returning interruption. [API][api], [server cancellation tests][server-cancel] | E; completion/scheduler | Z3/Z6 B original actual leases/fair worker and HTTP failures. Preserve unknown-completion quarantine; connection close or timeout alone cannot authorize free/reuse. |

### Speculation and target-authoritative verification

| ID | Mechanism in pinned DwarfStar | Class; fabric stage | Actual Zerogravity mapping and transfer condition |
|---|---|---|---|
| D43 | Checkpoint-matched learned DSpark support GGUF drafts up to five tokens; legacy MTP excluded when DSpark active. V4 versus V4.1 support graphs differ; V4.1 draft stays resident even target streams. [speculation][spec], [support graph][dspark] | M/E; source/IR/scheduler | Z10 P, executable learned drafter A. Same source/graph/physical authorities; admit both target+drafters+branches. Current model support takes priority; no inferred DSpark2 version. |
| D44 | GLM built-in MTP up to two committed tokens; Qwen adaptive one/two-ahead chaining, disabling deeper draft after poor second acceptance; native Metal speculative session batches. [spec][spec], [Qwen][qwen] | M/E; scheduler/state | Z10 P. MTP source weights must exist in exact checkpoint; native .8B/2B batching does not qualify any learned MTP. Independent all-accepted-prefix rollback tests first. |
| D45 | **Default nonzero-temperature policy accepts matching greedy drafts directly.** Exact mode uses target probability for deterministic proposal or p/q and residual mass for sampled proposal. [sampling docs][spec], [sampling tests][sampling-tests] | Default Q; exact E distribution; readout/state | Z10 P. Exact target-distribution correction is mandatory for a source-preserving claim. Equal seeds across schedules are stronger than equal distributions and require explicit RNG ownership. |
| D46 | Batched target verifier, seed+drafters in one pass, small-row expert/output kernels; direct accepted-prefix state commits avoid replay. Verifier FP order can differ from serial. [spec][spec], [CUDA][cuda], [QA][qa] | E/Q numerics; kernels/state | Z6 B batched target math is a foundation, not speculative execution. Z4 rollback B; entire mechanism P. Require independent target authority and chosen profile full-logit/state gates, not upstream tolerance adoption. |
| D47 | Confidence pruning + acceptance/cost scheduler/backoff and tail caps avoid unproductive proposals; gfx1151 four-cycle probe/four-token acceptance floor. [Strix campaign][strix-bench], [spec][spec] | E if target-correct; scheduler | Z10 P. Price proposer+verifier+replay+IO and global fair-worker opportunity cost; preserve cold/short/low-acceptance negative cases. Disable per context/request if predicted committed goodput falls. |
| D48 | Per-prefix recurrent/compressor/conv snapshots and recent verified rewind avoid replay at tool sampling-mode boundaries; partial rejection/exact TP paths may still replay. [spec][spec], [rewind tests][rewind-tests], [EOS tests][eos-tests] | E; state/readout/completion | Z4 B rollback, Z8 I continuation. Runtime speculative full-domain snapshot manager P. Restore target/draft RNG/grammar and native control-mode before resampling boundary; rejected future state must be invisible. |
| D49 | Learned PLE/Engram hash rows are **not** runtime suffix/n-gram token proposals. No independent prompt-lookup/suffix speculative strategy found in inspected DwarfStar API/engine paths. [Qwen][qwen], [API][api] | A upstream lookup; source/scheduler | Z10 explicit P for runtime n-gram/suffix strategy. It may need no extra weights but still needs target verification/full rollback and memory/fairness admission; do not count learned n-grams as implementation. |

### Distributed placement, transport and physical cost

| ID | Mechanism in pinned DwarfStar | Class; fabric stage | Actual Zerogravity mapping and transfer condition |
|---|---|---|---|
| D50 | Monotonic whole-layer placement over per-GPU budgets with embedding/head accounting; unsupported CPU-spill layouts refuse. [packer][layer-pack], [CUDA guide][multi] | E; placement/residency | Z3 B per-host admission, Z10 distributed/heterogeneous execution P. Source axes/state partitions unchanged; free-memory estimate cannot bypass common ledger or actual OS headroom. |
| D51 | In-process CUDA pairs shard routed experts, replicate attention/router/shared work, combine pair partials; pairs own layer ranges and vocabulary head is sharded. [CUDA guide][multi], [GPU placement API][mgpu] | E/Q reduction; placement/transport | Z10 P/A numerical distributed runtime. This is a specific TP/PP/expert-ownership combination, not arbitrary EP/DP/CP/sequence parallel support. Exact partition algebra/gates first. |
| D52 | Two-rank Metal/CUDA network TP partitions experts and model-specific attention/head work; shared-event/flag/poll GPU↔service-thread gates overlap peer exchange and prefetch. [transport][tp], [Metal gates][metal-gates], [distributed][distributed] | E/Q; transport/completion | Z10 P; existing fleet discovery does not qualify inference. Keep transport behind original math schedule; charge host slabs/events, retain peer epochs and completion owners. |
| D53 | RDMA/TCP negotiated windows, receive/completion accounting, direct Spark host-visible slabs with staged fallback; split DSpark drafter optional when link/memory permits. [TP][tp], [QA][qa-tp] | E; transport/residency | Z10 P. UMA host-visible RDMA is not discrete GPUDirect. Version negotiation, exact payloads, stale/tiny/partial transfer, cancellation and both-rank fault tests required. |
| D54 | Pipeline TCP moves activations through complete layer slices; multiple prefill chunks in flight overlap stages; one decode stream still serially traverses the route. [pipeline][distributed], [implementation][dist-code] | E; placement/scheduler | Z10 P. Capacity/long-prefill candidate only; include transfer/crypto/copy latency and immutable whole-state hydration. No promise of faster one-stream decode. |
| D55 | Pipeline 16/8-bit activation wire options reduce traffic but change numerical precision; default is 32-bit. [distributed][distributed] | Q; numerics/transport | Z10 P. Explicit wire numerical profile and source/state quality gates; not a lossless transport optimization. |
| D56 | Peer failure latches, releases gate waits to avoid deadlock, aborts eval; route invalidation and exact-prefix replay restore future work. No authentication/encryption in upstream network protocol. [Metal gates][metal-gates], [TP][tp], [distributed][distributed] | E; completion/state/control | Z3 actual quarantine B locally, Z10 authenticated remote epochs/whole-state transfers P. Do not copy unauthenticated trust assumptions or release unknown remote work on timeout. |

### Frontends, power and measurement

| ID | Mechanism in pinned DwarfStar | Class; fabric stage | Actual Zerogravity mapping and transfer condition |
|---|---|---|---|
| D57 | Native coding agent shares engine/session state directly; API adapters expose chat/completions/responses/messages/tools/reasoning/SSE. Saved sessions can omit large payload and replay. [server][server], [README][readme], [agent][agent] | E control; API/state | Z6 bounded serving B; Z10 general protocols/agent/multi-pipeline P. Useful adapter pattern; keep product-neutral typed task API, no native agent process as inference fabric. |
| D58 | Duty-cycle power control sleeps after measured layer/token work using running timing averages; not a GPU watt-cap setter. Several model paths require100%. [power code][power], [README][readme] | E latency tradeoff; scheduler/power | Z10 P broad energy policy. Measure joules/correct token and tails with fair workers; avoid sleeping while holding globally useful device ownership. No claimed power measurements here. |
| D59 | Memory/startup/context-frontier/acceptance profiling, alternating repeats, warmup, quality checks and release slowdown thresholds; engine and HTTP concurrency benches reject incomplete output. [performance][perf], [QA][qa], [bench guide][speed-guide] | E methodology; measurement | Z6 bounded native ABBA/CUPTI counting B, parent numerical qualification I; complete same-source stockvLLM C1/C8/long/cold/warm/tails/power P. No our physical execution in this report. |
| D60 | CPU affinity matters on heterogeneous Spark cores; persistent workers and IO policy are host-specific; no general NUMA placement optimizer or whole-model persistent device scheduler established by inspected code. [QA][qa-tp], [pool][pool] | E candidate; placement/scheduler | Z10 topology/persistent scheduling P/U. Bind to measured cache/thread/NUMA/link topology, permitted cgroups and device-specific profile; retain Renoir hold. |

## 5. Unmerged proposals are separate evidence

GitHub issue/PR text is primary maintainer/contributor evidence of a proposal,
not merged-main implementation or our reproduced result. The six linked PRs
were independently observed open on their GitHub pages; some comments report
cherry-picks into other forks, which do not change canonical main. As of this read:

| Proposal | Visible mechanism/evidence | Zerogravity use |
|---|---|---|
| [Tracker #1202][issue1202], [PR #1154](https://github.com/antirez/ds4/pull/1154) | Read next Qwen prefill chunk's original BF16 PLE rows while current Metal chunk computes; one extra buffer capped128MiB (80MiB at measured8192 chunk), joined before frontier visibility, synchronous fallback. Contributor records +3.94% fresh16K in eight-run comparison; separate four-run +6.27% smoke is not another gain. | Z2/Z3/Z10 P; transferable bounded IO pipeline after required source/state/runtime gates. Not currently needed by dense .8B/2B learned graph. |
| [Tracker #1202][issue1202], [PR #1198](https://github.com/antirez/ds4/pull/1198) | Reuse Q4_K packed header/quant loads in Metal NAX loader with arithmetic/geometry unchanged; +2.376% text/+2.274% code fresh16K on M5Max128GiB resident QwenQ4. Baseline already includes #1154. | Z5 P; portable quant block/header reuse. No final combined-versus-stock timing; percentages cannot be added. |
| [Tracker #1201][issue1201], [PR #1175](https://github.com/antirez/ds4/pull/1175) | Verify parent conversation before tool-ID prefix shortcut, include new user messages after results, reject edited/image-changed history. | Z4/Z10 P; correctness prerequisite. Avoided replay or reduced prompt work is not faster matrix arithmetic. |
| [Tracker #1201][issue1201], [PR #1169](https://github.com/antirez/ds4/pull/1169)/[PR #1206](https://github.com/antirez/ds4/pull/1206) | Keep exact sampled tool spans, IDs, EOS/closed-turn boundaries and reasoning-summary/omission-compatible visible keys. | Z4/Z10 P; source/template/history identity and all-state-root restore. Different client echo policies require explicit validation. |
| [Tracker #1201][issue1201], [PR #1200](https://github.com/antirez/ds4/pull/1200) | Save eligible completed text prompts before generation; at most two supplemental files, min(cache-budget/8,16GiB), no ordinary-cache eviction on admission and supplements evicted first. One four-request task-switch workflow99.664→62.872s, return38.404→.976s, about57K replay tokens avoided and ~1GiB extra logical disk state. | Z10 P persistence policy; include write latency/IO/endurance/memory cost. Small workflow timing with its integration baseline; not universal response or kernel speed. |
| [Issue #1240][issue1240] | Contributor posts Engram verification-row batch-read patch and28 tok/s on M3Ultra512GiB screenshot. No sufficiently controlled complete source/quant/context/acceptance/timing matrix in issue body. | Z2/Z9 P/deferred; batching tiny random reads is plausible, performance claim U. Do not import patch or rank it against current engine results. |

## 6. Benchmarks: exact scope and missing fields

All numbers are upstream statements; **none physically verified by this run**.
No row is a cross-engine ranking. Where source commit/checksum, context,
acceptance, cold/warm or metric definition is missing, it stays missing.

| Record | Model/quant/hardware/workload | Reported result | Qualification boundary |
|---|---|---|---|
| [Flash baseline][perf] | M5Max128GB, FlashQ2; continued2048-token intervals;128 greedy non-EOS decode tokens per frontier. | Context2048:790.18 prefill/39.35gen t/s;65536:398.50/27.64. | Stored baseline, not current-head sweep; load excluded from frontier rates. No request-SLO/tail/power matrix. |
| [Spark baseline][perf] | DGXSpark128GB, FlashQ2; same stated interval/probe policy. | 2048:825.76/18.05;65536:822.98/13.84 t/s. | Cross-hardware/source recipe control not sufficient to call an engine win. Snapshot versus replay must be distinguished. |
| [gfx1151 tuning][strix-bench] | FrameworkDesktop128GB, ROCm7.14, exact named0731 IQ2XXS/w2Q2K/AProjQ8/SExpQ8/OutQ8 imatrix; PromessiSposi pure prefill2048 increments→4096. | Warm4K187.26→292.86 t/s(+56.4%). | No full source SHA pair in record. Frontier bytes exact to80-key attention reference; earlier-stack full logits permit maxabs4.446/RMSE.770 within declared5.14/.871 envelope, not exact original model parity. |
| [gfx1151 DSpark followup][strix-bench] | ROCm10.0;2K low-acceptance/16K full-acceptance,128 outputs. | Low15.40 aggregate vs ordinary15.57 t/s; high25.71; scheduler bypass after4 cycles below4 accepted/cycle. | Controls acceptance; exact-sampling throughput not established. Aggregate includes first token/block; steady number separate. |
| [M5 streaming][ssd] |128GB M5Max; GLM5.3 Flash Q4_K177.77GiB;2K prompt+1K append;128 generated at each frontier; automatic cache/no speculation;2026-09-06. | Median3 runs121 initial/104 continued prefill,11.9/14.9gen t/s. | Capacity tradeoff, cache prompt-dependent; first-token wait included generation. Not equal to resident or another quant. |
| [Short append cache][ssd] | FullGLM5.3 IQ2_XXS196.58GiB;8Kcontext,61.35GiB cache;16-token append/64gen;3-run medians. | Append30.8→2.9s; following generation4.09→5.11t/s; compared logits/text identical. | Specific short continuation. No same gain for initial prefill; publication source pair not given in guide. |
| [Qwen Spark][qwen] | SingleSpark, resident Q2(41.73GiB) orQ4(69.74GiB), BF16 n-grams95.37GiB disk; PromessiSposi;8192chunk;128 teacher tokens;mean2runs. | Q2first1024=516, next7168=745, decode22.6t/s;Q4=513/755/21.2. Fresh32KQ2~771t/s at32768chunk. | Loading excluded, no MTP. Q2/Q4 have different quant recipes; rates cannot rank numerical engines.32-token Q4 append~273ms shows small-N sensitivity. |
| [Qwen checkpoint fix][qwen-checkpoint-bench] | M3Ultra512GiB;2026-09-13;baseline91095b7/fixd8ba365;old IQ2/Q2Pad768 MTP main + **quantized Q4_1 PLE sidecar**;warmed ABBA,2/build. | All throughput differences<.7%; text byte-identical for6generation cases. | Historical source differs from current BF16 embedded n-grams. No statistical equivalence or current-head result. |
| [Single Spark DSpark][spark] | Flash0731residentQ2;4Kallocated,512chunk,256outputs;median3,2026-09-06. | Greedy code19.72→31.41t/s;temp1code19.53→29.98;unpredictable prose19.53→18.81. | Positive-temp uses opportunistic distribution. Negative result retained; prefill unaffected. |
| [Dual Spark DSpark][spark] | Flash0731Q2/MXFP4;2Sparks direct200Gb/sRoCE;64Kallocated;temp0;512outputs;CPUtaskset5-9,15-19;median2(4MXFP4prose),2026-10-04. | Q2prose44.0/code83.3t/s;MXFP4prose41.7/code75.6. | Prompts105/31tokens;first block included. More hardware/capacity and different formats are not an engine-only gain. |
| [CUDA native sessions][multi] | Supported8L40S Flash layout;16decode rows;Q4 recommended native grouped expert format. | ~126aggregate gen t/s. | Per-client speed, full workload/checksum/commit/tail matrix not all supplied by short guide; no our reproduction. |
| [V4.1 SSD batching][qa] | SingleSparkQ2 SSD automaticcache;1Kcontext,24steps/session,8sessions. | ~11aggregate t/s vs~8.4 ordered in Spark guide. | Narrow supported2..8 row source path; `SERVER.md` table's all-single-GPU fallback wording is stale for this case. |

## 7. Negative experiments, bugs and drift

- [Pinned main causality fix][causal-fix] `902baebb70e66b25dc40e05d2672d194fd85c9c8`
  merges Metal compressed-attention causality repair (2026-10-19). Old successful
  speed/quality snapshots do not automatically include it. Causal future-token
  poisoning and compressed-boundary tests belong in any local attention port.
- [gfx1151 campaign][strix-bench] removed experimental compact IQ2 worklist after
  a small-context GPU memory fault. Retained stock launch+x64/y64 measured294.38
  at4K/268.51at16K. Faster worklist is not a candidate to enable blindly.
- [CUDA core][mmq] raw MMQ fallback for certain GLM shapes onGB10 is
  allocator-dependent; deterministic paths retained. Replicated TP work may not
  independently change arithmetic after rank-local allocation/kernel failure.
- [Side-stream guard][cuda-side] rejects concurrent use of global alignedQ8_1
  scratch. [MXFP4 guard][cuda] rejects persistent global MMQ scratch on multiGPU.
  A shared scratch optimization can invalidate overlap or topology support.
- [QA][qa] corrects GLM sparse boundary to2051 and padded-selection decode
  masking; benchmark against old dense-to4096 path is not valid evidence.
  ROCm tie/signed-zero/router-probability and exact-row decode arithmetic were
  repaired in [5c3b0a49][router-fix]/[f3ee00a1][row-fix].
- [Open#1235][issue1235] reports current1017Q2 filename paired with stale expected
  SHA256, rejecting correct341GiB file. Pinned script line12 selects1017 while line462 retains the reported old
  hash; no download/hash reproduction. Keep integrity refusal; never solve by skipping
  verification. New checkpoint default does not requalify old DSpark support.
- [Open#1223][issue1223] reports full DS4_TEST_DSPARK suite exits on its own
  instance lock after earlier shared engine stays open; isolated test passes.
  Open issue describes local fix, not merged-main complete suite success.
- [Open#1214][issue1214]/[1211][issue1211] report vision token mismatch and
  reasoning-echoing disk-key misses atfc80bd6; [tracker#1201][issue1201] also
  identifies stale-parent history shortcuts. Treat as issue-reported current
  risks, not our proven reproduction or permission to loosen identity checks.
- [Open#1209][issue1209] reports GLM5.3Q2 heterogeneous Strix/Spark PP heap abort.
  Generic PP docs do not qualify every model/backend pairing.
- [Spec guide][spec] says V4.1 DSpark supports Metal/CUDA networkTP;
  [distributed][distributed]/[Spark][spark] still say unsupported. Actual
  [97079d7d][tp-dspark] adds support with reference/tests and later RDMA-window
  reuse; record scoped current implementation, retain physical validation gap.
- [Server table][server] says singleCUDA fallback, but newer V4.1 SSD2..8 and
  networkTP5..8 paths and QA exist. Qwen CUDA remains ordered. Native grouping
  is selected by actual guards, not a general backend label.
- [MMQ provenance][vendor] says no CUDA graphs and older tier inventories;
  [decode-island source][cuda-graphs] implements graphs outside the adapter.
  [MMQ integration comment][mmq] says aligned tiers left behind, but current
  [alignedD2R call sites][cuda-dense] and repackers exist for selected shapes.
  The producer-fold registration hook still always misses. Resolve per symbol.
- Qwen byte-identical same-schedule replay is separate from cross-schedule
  equality/quality: nearly tied routed experts amplify legitimate roundoff.
  Upstream explicitly tests companion independence, paired NLL and task quality.
  Zerogravity retains its own stronger source-profile acceptance contract.

## 8. Composition hypotheses inside the existing fabric

These are hypotheses with no benchmark promise; choose after profiling and
parity. They introduce strategies at existing stages, not another engine.

1. **Qwen .8B/2B resident:** original source→typed numerical schedule→one admitted
   weight pack→few-row/dense dispatch→source-exact fusion→one fair worker→actual
   completion→original all-domain root publisher→minimal requested readout.
   Combine Z6 teacher-prefix StateOnly, existing batched projections, row-state
   transport and qualified greedy readout. Avoid full-vocabulary head on teacher
   tokens when semantically unnecessary; retain final full-head/readout barrier.
   Compare .8B/2B ordinary/natural serving first; source-only qualifier is not it.
2. **Long prompt + interactive decode:** shared source weights/prepaid scratch,
   aged bounded temporal chunks and independent decode cohorts; choose row
   geometry jointly with prefill quantum. Later static islands replay only
   after metadata/events/storage/drop admission closes. Charge inactive root
   snapshots/output credit. Physical overlap is optional and separately gated;
   deterministic serialized policy is the correctness baseline.
3. **Dense27B capacity path:** sealed original tensor windows→bounded coalesced
   next-layer reads→host staging lease→real copy readiness→current layer GPU
   compute→completion-retained windows. Double buffering may hide part of IO;
   every needed dense matrix still transfers/executes. First establish native
   full model/state; measure SSD read amplification, raw bandwidth floor,
   cold page cache and OS headroom. No MoE hit-rate assumption.
4. **Gemma after trained family gates:** reuse dispatch, source packs, scratch,
   scheduler and completion infrastructure. E2B PLE,26B MoE/K=V/sharedKV and31B
   dense semantics are independent. Shared immutable source storage is possible
   only for exact identical tensors; similar roles do not authorize aliasing.
5. **Later sparse/learned-table path:** common byte-budgeted size classes and
   immutable reader leases host expert hotsets plus bounded PLE/Engram row IO;
   demand-aware prefetch competes through the same RAM/SSD/device credits.
   One graph can combine learned tables, native batches and target-correct MTP;
   independently bind all recurrent/indexer/conv/draft/grammar roots.
6. **Joint batching/speculation:** select (cohort shape, draft depth, IO forecast,
   kernel profile) per fair opportunity from measured complete-cycle costs.
   More speculative rows can improve GEMM reuse while consuming scratch and
   delaying other sessions. Back off on low acceptance, short tails or SSD
   pressure. Price rejected/canceled/wasted work, not only accepted-token ratio.
7. **Later multi-device:** same typed graph chooses layer/expert/head partitions
   and a transport implementation; exact source/state algebra precedes RDMA
   tuning. Host-visible UMA exchange and discrete copies are separate placements.
   Admission/fairness/quarantine remain with their original owners across peers.

## 9. Required promotion gates

- **G0 prerequisite:** finish intended pinned vLLM source/numerical/API/serving
  parity for .8B/2B, including natural original-source2B and declared profiles;
  inspect failed/interrupted runs. Dated prose and Git SHAs alone are no result.
- **G1 source:** immutable full source/index/header/tensor/config/tokenizer/chat/
  processor/calibration identity; copied/derived pack provenance; retained open
  descriptors; wrong/truncated/replaced sources refuse. Match axes, padding,
  signed codes, zero points, scales and every selected tensor.
- **G2 numerics:** scalar edge fingerprints (NaN/inf refusal, signed zero,
  subnormal, RNE, overflow, FMA, reduction/cast order); independent actual trained
  intermediates/full vocabulary/all state at every phase and dispatch boundary.
  Short and long causal masks, ties, head mappings, empty/tail experts, teacher
  nonzero histories and 64-token chunk transitions. New approximate profile
  requires independent explicit acceptance; do not quietly relax existing gates.
- **G3 ownership/state:** independent session/tenant/source roots, reordered and
  changed companion invariance, exact snapshots/rollback for every accepted
  prefix, duplicate/stale generations, source/template/sampling/grammar/media
  identity and complete cancellation invalidation. Never publish a partial root.
- **G4 physical:** original ResourceArbiter before every actual source/derived/
  state/scratch/staging/event/graph allocation; no double-counted UMA backing or
  hidden cache. Actual ready/failure/unknown-completion, encoded/captured reader
  leases, ABA/victim refusal, pressure/shortIO/failedcopy/disconnect and final
  drop/recovery proofs. Keep gfx90c launch hold.
- **G5 speculation:** independent target-authoritative accept/reject/residual
  distribution, target/draft RNG and control-mode boundaries; test wrong drafts,
  EOS/stop/tool boundary, every prefix0..depth, full rejection/partial/full and
  failed verifier. Distinguish exact-distribution from same-seed serial identity.
- **G6 measurement:** separate profiled counting from unprofiled timing; matched
  model source/quant/activation/state precision/tokenizer/hardware/compiler/
  driver/context/offer trace/output settings/caps. Cold process, storage/page
  cache cold, warm weights and warm prefix are different conditions. Alternate
  order/repeat, retain failures and all offered/rejected/canceled work; report
  correct emitted goodput, TTFT/ITLp50/p95/p99, fairness/starvation, memory peaks,
  IO, energy/token and startup/pack cost. No cross-quant/model engine ranking.
- **G7 full gate:** relevant source/state/cancel/physical tests plus complete
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  in authorized approved-input isolation before accepting implementation. This
  research ran no implementation gate or physical experiment.

## 10. License/provenance notes

- [Root LICENSE][license] is MIT, retaining2026ds4authors,2023–2026GGMLauthors and
  2023DeepSeek copyright. Reusing/adapting substantial source requires the
  corresponding notices; independent algorithm reimplementation should still
  record exact source and provenance. This is engineering provenance, not a
  legal opinion or relicensing action.
- [MMQ VENDOR][vendor] pins GGML/llama.cpp
  `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (2026-05-14,MIT), distinguishes
  copied/modified files and the adapter; graph/runtime and allocator are not
  needed to study the kernel. Retain attribution per actual reused file.
- Some CUDA graph/MMQ design is explicitly attributed to Entrpi's fork and
  commits in current upstream. Canonical `antirez/ds4` is the inspected target;
  fork-only features/patches are separate research, not upstream capabilities.
- GGUF model/checkpoint/tokenizer/drafter/vision weights and hosted reference
  data retain their own source licenses/terms; engineMIT does not license all
  model payloads. [Model card][model-card], [evaluation provenance][eval-data],
  [quality fixtures][quality-data] identify additional provenance. No weight or
  dataset download, license acceptance or source import occurred here.
- Per-file scan is not a comprehensive dependency/legal audit. Optional GPU
  SDK/vendor libraries and any future copied kernel need their own review.

## 11. Completion and limitations

- Report covers60 main-source mechanisms plus six separately identified
  proposal groups. Each main row maps to an actual local seam/status; missing
  model/runtime/physical behavior remains explicit.
- Historical atlas/dossier, current plans/source, pinned docs/code, recent
  authoritative commit metadata and selected open issue text inspected.
- No implementation or shared plan edited. Root README preserved. Only this
  run's REPORT/TODO/pin metadata written.
- Validation PASS:60 sequential unique mechanism IDs,133 defined references,
  all local targets and cached upstream file/line anchors exist, README blob
  unchanged; whitespace and owned-path checks pass. No physical claims.

[pin]: https://github.com/antirez/ds4/commit/5379c1c9ac315f1540371b06a3db88491d4059ff
[readme]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/README.md
[api]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.h#L11
[core]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.c
[pool]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.c#L2081
[guards]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.c#L71862
[slab-class]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.c#L5098
[qwen-rows]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.c#L58294
[qwen-batch]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.c#L79819
[cuda-mixed]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.c#L85933
[power]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.c#L17863
[cuda]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_cuda.cu
[cuda-graphs]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_cuda.cu#L970
[mmq]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_cuda.cu#L1234
[cuda-dense]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_cuda.cu#L16380
[cuda-side]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_cuda.cu#L35942
[cuda-compressor]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_cuda.cu#L36002
[cuda-prefetch]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_cuda.cu#L29182
[cuda-read]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_cuda.cu#L2248
[vendor]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/cuda/mmq/VENDOR.md
[repack]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/cuda/mmq/ds4_repack.cu
[d2r]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/cuda/mmq/ds4_mmq_d2r.cu
[qcuda]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_qwen4_cuda.cuh
[qcuda-matrix]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_qwen4_cuda.cuh#L763
[qcuda-gdn]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_qwen4_cuda.cuh#L1653
[qcuda-attn]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_qwen4_cuda.cuh#L173
[qcuda-select]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_qwen4_cuda.cuh#L112
[metal]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_metal.m
[metal-cache]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_metal.m#L13955
[metal-seq]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_metal.m#L1471
[metal-evict]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_metal.m#L15340
[metal-load]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_metal.m#L16477
[metal-gates]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_metal.m#L10191
[metal-dense]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/metal/dense.metal
[metal-moe]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/metal/moe.metal
[rocm-mm]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/rocm/ds4_rocm_matmul.cuh
[rocm-moe]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/rocm/ds4_rocm_moe.cuh
[rocm-glm]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/rocm/ds4_rocm_glm.cuh
[hipblaslt]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/rocm/ds4_rocm_hipblaslt.cuh#L168
[kda]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/metal/glm53_kda.metal
[gpu-api]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_gpu.h
[ssd]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/docs/SSD_STREAMING.md
[ssd-code]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_ssd.c
[engram]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_engram.c
[linux-memory]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_linux_memory.h
[server]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/docs/SERVER.md
[server-scheduler]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_server.c#L12410
[server-cancel]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_server.c#L21339
[kv-header]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_kvstore.h
[kv]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_kvstore.c
[spec]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/docs/SPECULATIVE_DECODING.md
[dspark]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_deepseek41_dspark.h
[sampling-tests]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/tests/test_sampling.c
[rewind-tests]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/tests/test_dspark_rewind.py
[eos-tests]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/tests/test_dspark_eos_contract.py
[multi]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/docs/CUDA_MULTI_GPU.md
[mgpu]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_gpu_mgpu.h
[layer-pack]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_layer_pack.c
[distributed]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/docs/DISTRIBUTED.md
[dist-code]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_distributed.c
[tp]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_tp.c
[agent]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_agent.c
[perf]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/docs/PERFORMANCE.md
[qa]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/QA_BEFORE_RELEASES.md
[qa-m5]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/QA_BEFORE_RELEASES.md#L1353
[qa-tp]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/QA_BEFORE_RELEASES.md#L1927
[strix-bench]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/speed-bench/gfx1151-prefill-results.md
[speed-guide]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/speed-bench/README.md
[qwen-checkpoint-bench]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/speed-bench/qwen38-checkpoints/README.md
[spark]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/docs/DGX_SPARK.md
[qwen]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/docs/QWEN38_FLASH_NEXT.md
[models]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/docs/MODELS.md
[conversion]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/gguf-tools/README.md
[imatrix]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/gguf-tools/imatrix/README.md
[quality]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/gguf-tools/quality-testing/README.md
[license]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/LICENSE
[model-card]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/MODEL_CARD.md
[eval-data]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/EVAL_DATA.md
[quality-data]: https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/gguf-tools/quality-testing/data/README.md
[history]: https://github.com/antirez/ds4/commits/5379c1c9ac315f1540371b06a3db88491d4059ff
[causal-fix]: https://github.com/antirez/ds4/commit/902baebb70e66b25dc40e05d2672d194fd85c9c8
[tp-dspark]: https://github.com/antirez/ds4/commit/97079d7dcf9485ec2eb6da95cdc4293af69bb2fd
[skip-sort]: https://github.com/antirez/ds4/commit/b290d83bc5195ab3b9a981224240e2bfc4266b5a
[router-fix]: https://github.com/antirez/ds4/commit/5c3b0a49ea1ab9e0dffe4e055bf245345018286a
[row-fix]: https://github.com/antirez/ds4/commit/f3ee00a1d2931eec43a2ed9306c0ada2330e55df
[issue1201]: https://github.com/antirez/ds4/issues/1201
[issue1202]: https://github.com/antirez/ds4/issues/1202
[issue1209]: https://github.com/antirez/ds4/issues/1209
[issue1211]: https://github.com/antirez/ds4/issues/1211
[issue1214]: https://github.com/antirez/ds4/issues/1214
[issue1223]: https://github.com/antirez/ds4/issues/1223
[issue1235]: https://github.com/antirez/ds4/issues/1235
[issue1240]: https://github.com/antirez/ds4/issues/1240
[z-forward]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-qwen35/src/forward.rs
[z-program]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/accelerator/program.rs
[z-typed-tests]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/accelerator/program/typed_tests.rs
[z-source]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/source.rs
[z-window]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs
[z-window-tests]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/tests/tensor_window.rs
[z-ssd]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/src/ssd_extent.rs
[z-ssd-tests]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/tests/ssd_extent.rs
[z-arbiter]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/resource_arbiter.rs
[z-queue]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/device_queue.rs
[z-service]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/physical_service.rs
[z-leases]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_leases.rs
[z-quarantine]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/physical_quarantine.rs
[z-workers]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/model_workers.rs
[z-session]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/session_root.rs
[z-state]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/state.rs
[z-checkpoint]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-qwen35/src/checkpoint.rs
[z-resident-checkpoint]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_checkpoint.rs
[z-quant]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/quant.rs
[z-cpu]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-cpu/src/math.rs
[z-batch-proj]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/batch_quant_projection.rs
[z-activation]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/activation_quant_fused.rs
[z-resident-program]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_program.rs
[z-wave]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-cli/src/serve/gpu_batch.rs
[z-cohort]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
[z-prefix]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_inference/prefix_qualification.rs
[z-batch-program]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_program/batch.rs
[z-graph]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_inference/graph.rs
[z-graph-body]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_program/batch/graph.rs
[z-native]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_inference/native_fragment/full_model_qualify.rs
[z-native-faults]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_inference/native_fragment/full_model_qualify/faults.rs
[z-continuation]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_inference/native_fragment/continuation.rs
[z-gemma]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-gemma4/src/lib.rs
[z-gemma-moe]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-gemma4/src/moe.rs
[z-model-plan]: /home/pvl/spaces/abird/src/zerogravity/.agents/plans/model-execution/TODO.md
[z-serving-plan]: /home/pvl/spaces/abird/src/zerogravity/.agents/plans/serving-and-speculation/TODO.md
[z-memory-plan]: /home/pvl/spaces/abird/src/zerogravity/.agents/plans/memory-and-streaming/TODO.md
[z-opt-plan]: /home/pvl/spaces/abird/src/zerogravity/.agents/plans/cpu-gpu-optimization/TODO.md
[z-dist-plan]: /home/pvl/spaces/abird/src/zerogravity/.agents/plans/distributed-execution/TODO.md
[z-pipeline]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/pipeline.rs
