# Maya: source-backed optimization research

Author: `/root/maya_optimization_research`
Coordinator: `/root`
Date: 2026-10-21, Asia/Singapore
Scope: read-only upstream/local research; own run documents only.

## Result

- Correct identity: **[mw00/project-maya][M01]**, the C++/CUDA/HIP inference engine derived from Strata. It is unrelated to Maya/Aya model names.
- Current inspected main: **`3384b5e9366e696e5f23f51772fb94a8a7948e41`**, commit date **2026-10-20T19:13:01Z**, subject `CHANGELOG: v1.1.19's headline - DeepSeek long prompts up to ~1.5x faster, same answers (#319)`.
- `VERSION`: **1.1.19**. Latest inspected tag `v1.1.19` points to **`3bf6867d344212166c09d2a5fe086b9ab90ac2fb`**; current main has subsequent changelog text. Every external source link below pins the inspected main commit.
- License: root [MIT][M02], copyright 2026 Peasant Smith/Project Maya and Niko1221/Strata contributors; [DeepSeek MIT notice][M03] also present. EXL3 header identifies exllamav3/MIT; imported GGML/vision/component attribution still requires per-file/dependency checks before copying. Model licenses and artifact licenses are separate. This report imports no code.
- **Best immediate transfer after Qwen reference parity:** same-arithmetic dense weight reuse across request rows, prepared source/shape specialization, reduction-preserving kernel work, fewer redundant dispatches/state copies, measured batch/graph selection. Existing Rust implementations must be extended, not duplicated.
- **Best later memory transfer:** phase-based memory lending, layer-major weight reuse, bounded staging/read overlap, persistent CPU pools, measured topology-aware placement and separate expert versus lookup/state cache policies.
- **Major exactness boundary:** Maya GLM CPU/GPU expert placement changes arithmetic; batched prefill changes projection arithmetic; cache layout can change sum order. These are not exact-parity optimizations merely because their output quality is acceptable. Keep explicit numerical profiles and original StateRoot/source/completion gates.
- Maya now includes up to eight GLM requests, DFlash2, current HIP/UMA paths and a **separate DeepSeek engine**. The historical local atlas's narrower description is useful provenance, not current feature status. Separate per-model Maya runners are **not** a design recommendation for Zerogravity.
- The source-backed mechanism inventory is in [OPTIMIZATIONS.md](maya-inventory.md): **MAYA-01..42**. It is a bounded inspection of substantive mechanisms, not a claim to enumerate every branch, kernel, PR or hidden optimization.

## Research method and qualification limits

- Read current applicable AGENTS, plan README/index, serving, numerics, memory, optimization and distributed TODOs; historical [Maya atlas pointer][L01]. Live source/plans supersede dated handoffs.
- Read upstream GitHub API current commit/tag/tree, README, relevant headers/implementation portions, tests/CI metadata, architecture/performance/settings docs and published negative results. Source text fetched with bounded read-only HTTP; no upstream program executed.
- Inspected local MAIN `/home/pvl/spaces/abird/src/zerogravity` and ISO `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020`. Both report HEAD `fb68d8fa165de16e21d1bd37afef3790b80da971`; their working trees differ. For example `accelerator/program/vllm_model.rs` and `vllm_cuda_gdn/resident/model.rs` exist only in ISO; resident graph/batch/state-transport, ResourceArbiter, ContextOwner and Gemma config files compared byte-identical between MAIN and ISO. Source-window/program files differ.
- **No builds, tests, GPUs, reference engines, models, SDKs, executables, device changes, commits or publication.** Upstream tests are evidence of test implementation; published results are author/community reports. Local qualifications below are current plan/coordinator results, not new executions by this researcher.
- Readiness, resource reservation, source authority, cancellation, root publication and unknown-completion quarantine remain original Rust authorities. A descriptive plan/hash or saved slot cannot issue a live permit.

## Current implementation, rather than the old port study

The [GLM port study][M04] explicitly dates its initial assumptions to 2026-09-29/engine0.1.27. Several header introductions likewise retain older milestone prose. For example `expert_cache.hpp` still describes slot storage before hit computation, while current `glm_fast_path.cu`, `glm_rows.cu` and `glm_prefill.cu` implement the production consumers. Resolve conflicts using actual call sites and current settings/changelog.

- **GLM-5.3-Flash:** 45-layer hybrid, 34 KDA recurrent and 11 sparse MLA/DSA layers, 42 MoE layers with 288 routed experts/top8, three initial dense layers, mHC and model-owned NextN. These architecture facts explain the benefits; they are not free optimizations applicable to dense Qwen or Gemma.
- **GLM fast path:** resolves weight pointers at load; device router looks up resident experts; host-mapped request ring and service thread handle misses; per-layer partitions prevent ordinary miss replacement from overwriting other layers' readers. Private nonblocking streams and copy stream overlap work. [Actual implementation][M05].
- **Current multi-request:** one GPU uses one token per conversation with shared dense reads; two or more GPUs can pipeline different sequences, optionally batch groups per part. Per-conversation recurrence, latent/indexer caches and snapshots remain independent. `--parallel`1..8 is implemented; it should not be described as absent. [Rows][M06], [sequence state][M07], [settings][M08].
- **DeepSeek-V4.1-Flash:** separate `maya-ds4` code path, its own GGUF/EXL3 expert execution, Engram, DSpark, tensor/layer grouping and prompt schedule. Review transferable mechanisms without importing its separate model loop. [Current changelog][M09], [model][M10].
- **Platforms:** CUDA, HIP and Windows code exist, with materially different qualification ranges. Current AMD notes support gfx1100/gfx1201/gfx1151 and experimental gfx1030; older RDNA4 notes saying gfx1151 only builds are historical. CUDA-name mapping or bundled code objects do not establish physical correctness on an untested architecture. Nothing authorizes a Renoir/gfx90c launch.

## Exactness classification

These classes must remain separate when integrating any row of the inventory:

| Class | Meaning | Maya examples | Required Rust treatment |
|---|---|---|---|
| **E: exact representation/schedule** | Same source operands, operation order, casts, masking, state and output rules; only reads/dispatch/placement change | dense `mv_nt` weight reuse preserving each column; radix DSA top-k preserving stable ties; exact sign expansion | Same-profile full logits/all-state plus physical lifetime/cancel tests; no assumed speedup |
| **F: floating schedule/profile** | Same real-valued equations, different finite-precision path, reduction or operand casts | GLM prefill FP16 GEMM versus q8_1 GEMV; CPU Q8_K versus GPU q8_1; grouping resident/staged expert sums; latent F16/INT8 | Explicit source/activation/state/reduction profile and independent quality gate; never silently call it exact parity |
| **D: target distribution preserved** | Corrected speculative sampling targets the same distribution but consumes RNG differently | NextN `min(1,p/q)` and residual `max(0,p-q)` path | Distribution and RNG contract; seed equality is separate from distribution correctness |
| **A: altered model/context/distribution** | Weights, cache precision, selected experts, prompt graph or sampling policy changes semantics | new mixed-bit quant; Q4 KV; light-expert skipping; aggressive128-row DeepSeek prompt skip; derisked weights | Explicit user-selected profile/source and measured quality/context acceptance; excluded from exact-parity claims |

**Concrete traps from current source:**

- [GLM repeatability settings][M08]: GPU activations quantized in blocks32 versus ggml CPU blocks256; measured expert relative-L2 discrepancy about2%. Router top8 amplifies it into different later expert sets and tokens. Dynamic pool sizing, CPU split retiming, background moves and RAM tier sizing all change which arithmetic path executes.
- Same source documents six256-token prompts across five starts: fixed pool/RAM/CPU plan/moves produced70/70 matching token-stream pairs; under a300MiB VRAM perturbation fixed settings matched105/105, automatic pool sizing matched0/105. These are bounded one-PC results, not universal determinism proof.
- CPU lane off removed that placement-dependent decode difference in the tested setup, with4–10% lower decode speed. Different prefill group layouts, builds or drivers still need separate qualification.
- [Prefill implementation][M11] explicitly uses FP16 tensor-core projections instead of the one-token path's q8_1 products. It is faster by different numerical arithmetic.
- Loading a drafter consumes memory and can change resident versus staged prefill group ordering, even if toggling verification **with the same drafter already loaded** is bit-exact. Current settings report tokens diverging after17/48 when comparing loaded/unloaded DFlash2; token-by-token prefill removed that observed difference. Account for resident-memory perturbation when evaluating speculation.
- [Speculative sampler][M12] explicitly distinguishes same distribution from same plain seeded token stream; greedy/exact-match mode has different requirements from corrected sampled mode. Its inspected gate excludes penalty requests from that opt-in path.
- Sampling support is itself bounded: the inspected [sampler][M28] caps sampled top-k lists at64 (`0`/larger still use64), uses penalties→top-k→top-p→min-p→temperature, and current request settings cap penalty history at4096 tokens. Match that declared distribution when comparing; do not assume arbitrary top-k/full-vocabulary sampled parity.
- [DeepSeek exact prompt skip][M13] preserves dependency support but qualifies its claim with expert summation order. It is a mathematical graph-liveness result, not a universal bitwise result.

## Resource, cache and state contracts worth transferring

### Distinct memory objects and tier policies

- Expert weights are immutable model data. KV/latent/indexer/recurrent/conv state is conversation data. Engram/PLE is learned lookup data whose rows depend on token IDs. Prefix checkpoints combine multiple state domains. Each requires different keys, replacement policy, readiness and ownership.
- Current [GLM tier structures][M14] use VRAM states FREE/RESIDENT/SPARE/DRAINING/LENT/LANDING, RAM size classes with HOLD/RELEASE/DEMOTE/NEW/LOAD/PIN/HELPER states, original live copies retained until copy events complete, per-layer slot partitions, aged LFU with recency tie-break and persistent usage counts. Lending holds background moves; fully resident checks can omit miss scaffolding.
- Duplicate writes to the same device residency-table key in one parallel update batch formerly raced. Current `upd_gen/upd_at` deduplicates to the final value. This generation is an update-batch identity, **not proof of universal source/cache/graph ABA safety**.
- [Direct IO][M15] implements Windows unbuffered overlapped/IOCP and Linux a bounded pool of blocking `O_DIRECT` pread threads. Linux thread count is queue depth; source does not magically implement io_uring. Aligned buffers/lengths/offsets are required. Plain `FileExpertSource` mmap/advisory-prefetch behavior is a separate source path.
- [KV streaming][M16] keeps authoritative FP16 or INT8 latents in pinned mapped host memory; device CLOCK-resolves selected pages, excludes this call's readers and copies missed blocks. It is not cross-request prefix sharing, and it is not a substitute for all-domain snapshot ownership.
- [Engram IO][M17] knows rows as soon as token IDs are known, issues bounded FIFO reads ahead of their consuming layer and keeps a CLOCK row cache. This is deterministic future knowledge; predicted MoE experts are uncertain and may waste IO. They deserve distinct lanes under one physical IO budget.

### Why Rust must preserve stronger authorities

Maya's per-layer single-stream cache safety is useful evidence for that execution shape. Zerogravity additionally has independently live sessions, B2/4/8 rows, fair queues, encoded-but-unsubmitted work, private graph replays, cancellation and unknown-work quarantine. A slot can be read by work not visible to a host cache's current loop. Transfers therefore require:

1. Source-qualified logical tensor/expert identity separate from physical residency and source-derived layout.
2. Admission before allocations/IO/copies, through original ResourceArbiter, including graph metadata, staging, registered host pages, pools, in-flight old/new copies and scratch.
3. Original source/IO/device reader leases from preparation/capture through actual completion or quarantine; generations checked before reuse, including encoded and captured readers.
4. Copy readiness that reflects the real event, with original live copy retained until every reader and transfer has completed; no host flag alone can free/reassign storage.
5. Independent per-session root/epoch/frontier, complete domain dependencies and atomic publication. Sharing weight reads cannot share mutable state.

These are integration requirements/inferences, not claims that Maya has Zerogravity's physical arbiter or failure model. No comprehensive Maya failure/quarantine audit was performed.

### Complete snapshots, not KV-only hits

- [GLM snapshots/slots][M18] capture KDA state and conv history, DSA latent/indexer pooled/open-tail domains and NextN where applicable. Restoring a position is safe for append-only caches only under that exact source's rewriting/masking assumptions. Speculative reject restores recurrence and replays accepted prefixes.
- DFlash also owns taps and its own feature/KV ring; [its reset/inject APIs][M19] invalidate positions after rollback or imported state. A target-root restore alone is insufficient if a drafter's cached feature positions survive.
- Inspected native slot header checks magic, number of halves, positions, layer layout/run sizes. It does not itself show a cryptographically authenticated model/profile/tenant identity. Do not treat the string `not a slot of this model` as such an identity proof. Server-side keying or trusted same-process use may add constraints; this was not exhaustively audited.
- Rust full-domain prefix reuse still needs source/loader/quant/numeric, tokenizer/template/processor/adapter, tenant/grants, positions/schema plus the **greatest jointly restorable frontier** across all KV/SWA/recurrent/indexer/draft/grammar/media domains. Geometry equality or token text alone is insufficient.

## Current local mapping: established, isolated and open

Detailed row-by-row dispositions are in the inventory. These are the recurring anchors:

| Local status | Current evidence and precise scope |
|---|---|
| **MAIN implemented + bounded qualified** | Shared one-engine `ForwardPlan`/`PlanOperation`; independent root/session binding; original physical arbiter/fair worker; original GPU leases and unknown-work quarantine. [Core program][L02], [root primitive][L03], [arbiter][L04], [batch rows][L05]. Preserve existing authority rather than introducing an optimizer-owned runtime. |
| **MAIN implemented + bounded qualified** | Source-exact GGUF projections, full-model CUDA B2/4/8 serving/unequal frontiers/teacher-prefix behavior, private full-model graph qualification and row-native state transport exist. The serving plan records46 physical model/owner/private-graph/failure cases,10 operator/failure cases and complete1251-test gate. [Serving queue][L06], [private graph][L07], [batch graph][L08], [state transport][L09]. These are real accomplishments; production graph admission, full temporal GPU prefill, broad context/parity/performance remain open. |
| **MAIN implemented + bounded qualified** | Verified source extents/windows and actual original Qwen3.8-27B55,563,006,776-byte source scan/row-window qualification; no whole55GB resident allocation. [Memory queue][L10], [source windows][L11]. Integration into trained streamed inference/async readiness remains open. ISO adds further typed/window support; do not erase MAIN foundation. |
| **ISO only, bounded physical result reported by coordinator** | Original .8B forced full24 BF16 model boundaries, dual residual outputs, complete GDN/KV/head and cancellation/physical tests passed; source-linked `vllm_model`/resident model additions remain isolated. [ISO model][L12], [ISO resident body][L13]. This is **not natural serving/vLLM parity**, nor2B native whole-model qualification. |
| **Reference acceptance, not native execution** | Coordinator reports genuine original2B natural/all42 vLLM reference now passes. This replaces stale pending run-file prose, but native2B lowering, natural feedback, serving and performance remain open. No new independent execution by this agent. |
| **Contracts/design only for whole model** | Gemma4 variant/config/tensor/shared-KV/MoE contracts exist, with source tests. [Gemma library][L14], [Gemma config][L15]. Whole-model trained Gemma4 inference is unqualified. |
| **Design/open integration** | Phase-lending/global UMA accounting, trained SSD model execution, cache ABA across prepared/captured work, source-exact mixed CPU/GPU expert scheduling, complete hybrid prefix/COW, trained speculation and distributed model execution. [Memory][L10], [optimization][L16], [distributed][L17], [batch/prefix design][L18]. |

**Dense27B constraint:** original source is dense, approximately55.56GB; target working budget5GiB. MoE's top-k experts and hot sparse cache do not eliminate dense projection reads. Use bounded source windows, layer/row tiling, reuse across batch/prompt rows and explicit IO cost. Any claim that the model becomes sparse, resident or fast requires separate source/physical evidence. Acquisition/source-window qualification is not inference support.

## Composition through one mathematical engine

The following is a design inference from Maya mechanisms and current Rust source, not implemented work from this run.

### Keep semantic graph separate from executable schedule

- Preserve existing model description/traversal, source operands, state effects and `ForwardPlan`; add typed source semantics only where needed. Model differences belong to operators/attributes and source-aware lowering. Do not create Qwen, Gemma or DeepSeek serving loops to gain optimized kernels.
- Select variants along independent axes: **operator math/profile**, **shape/layout/kernel capability**, **residency/IO**, **prefill/decode/batch scheduling**, **prepared/captured command mechanism**, **sampling/draft policy**. Each variant declares required capabilities, owned resources, read/write effects and fallbacks.
- A candidate fusion must preserve typed producer/consumer edges, source descriptors, casts, recurrence/attention masks, selected IDs/tie handling, physical leases and publisher boundaries. The .8B dual BF16 residual/F32 sum work is a concrete reason not to fuse from abstract real-number equations alone.
- Validate legality first; benchmark only legal variants. Measured faster shapes can populate a versioned tuning table keyed by source/numerics/hardware/compiler/driver/shape. A table proposes a choice; the current session, arbiter and queue still authorize it.
- Reuse existing `BatchProgram`, scalar/resident source math and original worker scheduling. Different backend lowering bodies are acceptable; different model math engines or duplicate traversal authorities are not.

### Interactions that prevent multiplying advertised speedups

| Composition | Opportunity | Conflict and gate |
|---|---|---|
| Dense batch × graph × device sampling | Amortize weight reads and enqueue/readback overhead | Stable shape/workspace/stream keys; packed per-row roots and separate RNG/EOS; production graph resources and retirement still need admission |
| Larger prefill × expert cache | Fewer weight/expert rereads | More scratch borrows expert slots, evicts hot decode working set and raises TTFT/ITL for other sessions; compare whole request and fair interference |
| CPU lane × async IO | CPU compute overlaps copy/SSD | Cores, NUMA bandwidth and SSD readers contend; SMT was harmful upstream; exact arithmetic requires a matching profile across placements |
| Drafting × batch concurrency | Verify several tokens with one weight read | Drafter memory reduces resident target capacity; request batches already amortize reads; Maya disables DFlash2 when parallel>1; measure joint policy |
| Prefix reuse × draft taps × shared context | Avoid replay of common authorized context | Snapshot must include or invalidate drafter state and all recurrent/indexer/open-tail state; source/tenant/grant epochs remain live |
| Cache promotion × captured graph | More expert hits with fewer launches | Captured address/plan changes, encoded reads and background copies extend lifetimes; invalidate key and hold reader leases before any reuse |
| Layer-major multi-chunk × low-memory target | Read a layer once across prompt chunks | Stores inter-layer activations or states, increases host traffic and possibly prefill monopolization; compare saved weight IO against extra activation IO |
| Tensor parallel × quant × heterogeneous devices | Increase bandwidth/capacity | Some quant blocks/rotations cannot split arbitrarily; new reduction topology changes rounding; slow links can dominate |
| Native sparse selection × state streaming | Fetch only selected model-defined pages | Dense attention models cannot adopt sparse DSA selection as an exact optimization; page copies remain governed by original physical ownership |

### Shared-context opportunities beyond a direct Maya port

All are hypotheses requiring measurement and correctness gates; none establishes a new source identity or live authorization.

1. **Authorized complete-context base plus private continuations.** Use existing [ContextOwner/ContextReader][L19] to freeze selected context inputs, render/tokenize once for identical source/template/profile, reuse complete immutable prefix state, then fork private roots. Revocation/`require_fresh` must be rechecked at delivery. Different model weights can share authorized text/tokenization only where tokenizers agree; never share KV/recurrent values merely because their text matches.
2. **Co-schedule requests sharing immutable weight/layout and shape.** Batch identical source/profile rows even when causal frontiers differ; existing B2/4/8 infrastructure is the base. Cache/prefix hit can influence grouping, but age/deadline fairness must bound waiting. State lineage remains per row.
3. **A common deterministic lookup dependency lane.** PLE/Engram/media table indices known from input IDs can issue bounded source reads before their consuming node. Keep them separate from speculative expert prediction and arbitrate total SSD/core/staging bandwidth. This generalizes to source-known lookup dependencies without adding model loops.
4. **Graph liveness from actual KV ownership.** DeepSeek's prompt skip suggests proving dependency cones in the existing typed graph. Gemma shared-KV/window relationships may expose dead work, but are **not sufficient** to import DeepSeek's128/~2700 cut: every residual, recurrent, embedding/PLE and future decode state dependency must be closed. Preserve finite-precision schedule or label a new profile.
5. **Context-aware weight-window reuse for dense27B.** Process compatible prompt rows or bounded request cohorts through one authenticated dense window before releasing it, subject to existing fairness and maximum in-flight state budget. This can reduce repeated SSD scans; it does not sparsify the matrix or erase activation-transfer cost.
6. **Transactional speculative branches on the same StateRoot machinery.** Draft, verify and commit are graph/task variants with complete private state. Accepted-prefix replay can reuse source math, but target RNG/grammar/output delivery and actual GPU completion govern publication. Separate drafter state and memory must be admitted before selection.
7. **Global pressure-aware optional work.** When fully resident or latency-critical, omit cache/IO/prediction scaffolding and speculative work whose modeled benefit is negative. Compare end-to-end goodput; policy may reduce aggressiveness under pressure without changing source/math/profile.

## Performance evidence and negative results

All following numbers are **upstream reports**, not local reproduced gains or matched vLLM comparisons.

| Evidence | Result | Practical limit |
|---|---|---|
| [2026-10-06 GLM experiment][M20] | Mercury2xV100: batched prefill2.6K prompt91s→12s; pipelined MTP plain26.2tok/s versus40.0 at split24 in a stated300-output run | Older Unsloth quant,2GPU topology,30GB RAM, greedy warmed context; prefill arithmetic differs; not Qwen/Gemma/singleGPU speedup |
| [RDNA exact expert work][M21] | gfx1100 synthetic kernels9.8–35.0% lower latency;620 synthetic comparisons bit-exact; full-model report15.1→16.4tok/s(+8.6%) | Shape4096×2048 and listed IQ types; gfx1151 downIQ3 regression retained legacy; gfx1201 untuned. No Renoir applicability |
| [KDA graph note][M22] |30.7259→31.4472tok/s(+2.35%), two alternating-order repeats of one32,755-token seeded prompt | Opt-in; actual-pack/current-main quality matrix explicitly interrupted/pending; no AMD physical validation. Shipping code is not broad qualification |
| [Multi-request settings][M08] | V100 aggregate×1.41C2/×1.35C4; all-resident Gorgon Halo×1.67C2/×2.02C4 | Exact tokens only claimed in tested greedy CPU-lane-off2/4 setup; private context memory reduces expert residency; multiGPUk2/k4 regressed |
| [Maya-S quant report][M23] |8 held-out sequences/7,672positions, top64+rest KL0.428;83.3% same top token;97.9% average five-task zero-shot accuracy recovery | This is changed source/quant, not97.9% exact inference;400 questions/task, limited long-answer/context screens, no statistical equivalence to FP8 or Qwen/Gemma |
| [DeepSeek current release][M09] | Stated32K long-prompt+30%oneGPU,8–16K+43–54%Strix; layer-major16K reads264GB versus736GB; chunk fix7x3090 32K99→1010tok/s | Different architecture/source/topology; some are bug/shape fallback repairs; exact skip caveats; no hardware reproduction here |

The [progress log][M24], [spec experiment][M20] and [quant report][M25] preserve useful counterexamples:

- Predicted disk/next-layer expert prefetch increased IO and slowed inference despite higher hit rate; only~40–45% of misses predictable on cited traces. An exact prediction mechanism can lose overall.
- On an early limited-memory trace LFU≈LRU and even Belady left capacity misses; on a later singleGPU trace LFU77.5%, LRU67%, learned77%, theoretical optimum87%. These are different traces/regimes, not a contradiction or universal ranking.
- CPU expert offload initially slower due thread startup; persistent pools improved it. Offloading **all** misses then reduced promotion/hit rate and lost. Twice as many logical CPU threads reduced21.8→13.3tok/s by starving readers.
- OneGPU MTP can fail to pay because consecutive routed tokens share few experts and PCIe traffic roughly doubles. DFlash2 slower on V100 than MTP; confidence/window/topology need measured selection.
- Prompt caching all read experts retained too much cold data and increased later IO. Large prefill chunks faster oneGPU but worse on a split with insufficient pipeline overlap.
- Wider quant loads, separate shared-down launch and occupancy restriction did not beat selected RDNA paths. gfx1151 optimized IQ3 down was0.972× and reverted selectively.
- Scale refitting improved reconstruction5–38% and slightly improved KL, but held-out text was worse in aggregate; Maya-M ships without it. Kernel error or reconstruction metric alone cannot choose a model profile.
- Graph launch reduction cannot be presented as measured hardware throughput; thermal/power/RAM/SSD upgrades in historical logs are not isolated software gains.

**Suggested local methodology after prerequisite parity:** freeze model/source/quant/tokenizer/template/sampler/profile and complete starting state; same tokens for compute AB comparisons where required; compare prefill/decode, solo/concurrent, cold process/warm process and storage cache state separately. Report TTFT, individually observed ITL, output agreement, SLO goodput, memory/IO/power/fairness and actual physical dispatch counts. Repeated ABBA/BAAB samples and negative candidates are required; preserve failed/interrupted gates. A short unprofiled C1 gain cannot substitute C8 behavior or source correctness.

## Prioritization under the user's model order

1. **Finish same-source/numerics/state parity first:** .8B natural/native delivery and2B native/serving; broader correctness as current queue requires. Forced full24 and reference acceptance remain bounded prerequisites.
2. **Qwen resident exact work:** extend current dense multi-row projection reuse, source-selected prepared dispatch, source-exact fusion and state transport, then qualified production graph and device readout. The local engine already has the right basis; optimize measured hot paths in it.
3. **Dense Qwen3.8-27B5GiB:** authenticated bounded windows, row/layer schedule, read coalescing/reuse, genuine copy readiness and trained resident-versus-streamed full-state parity. Model working set feasibility and IO floors precede performance claims.
4. **Gemma4:** qualify whole-model semantics in the same engine first, including shared-KV ownership, windows, normalization/logit caps and MoE only on variants that actually contain it. Reuse dense/scheduling/memory mechanisms with fresh shape/profile evidence.
5. **Later sparse Qwen/GLM/DeepSeek:** expert tiering, model-native sparse/indexed attention, MTP/DFlash/lookup, Engram/PLE and distributed placement. Keep licenses, model acquisition and physical launches within separately authorized scope.
6. **Explicit changed profiles only after exact reference baseline:** mixed-bit derived quant, low-precision state, tensor-core arithmetic changes and approximate routing/context. Source identity and numerical profile must expose the change.

## Blind spots and remaining research

- This is a bounded substantive source review, not every file/branch/PR or a formal Maya correctness/security/lifetime audit. Large `ds4_model.cu`, `generate.cpp` and GLM units were inspected at mechanism/contract sites rather than line by line.
- No CUDA/HIP/Windows execution, source quality reproduction, cache-source mutation campaign, fault injection, multiGPU trace or independent benchmark was run. Public CI itself runs Python and compatibility mapping without GPU; maintainers report separate V100 premerge execution. [CI][M26], [contribution instructions][M27].
- Exact compiled ISA/reduction behavior on an arbitrary driver/backend is unknown; declarations and source tests do not establish hardware parity. Full CPU/GPU DeepSeek low-bit sum equivalence is a release claim with limited source inspection, not independently proven here.
- No all-domain tenant-safe cross-process prefix/COW or production distributed network protocol was established by this inspection. Local multiGPU links and helper ownership are not authenticated multi-node state transfer.
- The current memory-plan prose links `BATCHING_AND_PAGED_STATE.md`, which is absent in inspected ISO. This report links the extant `FUSED_GPU_BATCHING.md`; the missing upstream local document was not reconstructed or edited by this run.
- Performance figures are heterogeneous sources, versions, quants, prompts, context sizes and hardware. Neither latest README maxima nor older logs support a direct vLLM/Qwen/Gemma performance ratio.
- Publication/runtime promotion remains coordinator work. This run leaves implementation and all shared plans unchanged.

## Pinned primary sources

[M01]: https://github.com/mw00/project-maya/tree/3384b5e9366e696e5f23f51772fb94a8a7948e41
[M02]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/LICENSE
[M03]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/LICENSE.deepseek-v41
[M04]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/docs/GLM5-FLASH.md#L1-L13
[M05]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/core/glm_fast_path.cu#L1-L17
[M06]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/core/glm_rows.cu#L1-L15
[M07]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/core/glm_multi.cu#L1-L35
[M08]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/docs/SETTINGS.md#L168-L282
[M09]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/CHANGELOG.md#L6-L95
[M10]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/ds4/ds4_model.cu
[M11]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/core/glm_prefill.cu#L1-L18
[M12]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/core/glm_fast_path.cu#L6324-L6347
[M13]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/ds4/ds4_model.cu#L8491-L8536
[M14]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/core/glm_fast_state.hpp#L496-L617
[M15]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/platform/direct_file.cpp
[M16]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/include/strata/kernels/glm_kv_stream.hpp#L1-L40
[M17]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/ds4/ds4_engram_io.hpp#L1-L8
[M18]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/src/core/glm_fast_path.cu#L6513-L7080
[M19]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/include/strata/core/glm_dflash.hpp#L1-L88
[M20]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/bench/results/2026-10-06-glm-spec/README.md
[M21]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/docs/AMD_EXPERT_KERNELS.md
[M22]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/docs/GLM-KDA-GRAPHS.md
[M23]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/bench/results/MAYA-S.md
[M24]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/bench/results/PROGRESS.md
[M25]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/bench/results/MAYA-M.md#L46-L51
[M26]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/.github/workflows/tests.yml#L1-L39
[M27]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/CONTRIBUTING.md#L9-L53
[M28]: https://github.com/mw00/project-maya/blob/3384b5e9366e696e5f23f51772fb94a8a7948e41/include/strata/kernels/sampler.hpp#L1-L35

[L01]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/.agents/prior-art/RESEARCH_SOURCE_ATLAS.md
[L02]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/accelerator/program.rs
[L03]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/state.rs
[L04]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/resource_arbiter.rs
[L05]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_batch.rs
[L06]: /home/pvl/spaces/abird/src/zerogravity/.agents/plans/serving-and-speculation/TODO.md
[L07]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_inference/graph.rs
[L08]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_program/batch/graph.rs
[L09]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_state_transport.rs
[L10]: /home/pvl/spaces/abird/src/zerogravity/.agents/plans/memory-and-streaming/TODO.md
[L11]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs
[L12]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/accelerator/program/vllm_model.rs
[L13]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/vllm_cuda_gdn/resident/model.rs
[L14]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-gemma4/src/lib.rs
[L15]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-gemma4/src/config.rs
[L16]: /home/pvl/spaces/abird/src/zerogravity/.agents/plans/cpu-gpu-optimization/TODO.md
[L17]: /home/pvl/spaces/abird/src/zerogravity/.agents/plans/distributed-execution/TODO.md
[L18]: /home/pvl/spaces/abird/src/zerogravity/.agents/plans/serving-and-speculation/FUSED_GPU_BATCHING.md
[L19]: /home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/context_snapshot.rs
