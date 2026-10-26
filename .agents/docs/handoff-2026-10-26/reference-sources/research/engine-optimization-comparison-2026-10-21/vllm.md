# vLLM optimization inventory and Zerogravity composition

Author: `/root/vllm_optimization_research` (GPT-6 research agent).
Date: 2026-10-21. Parent: `cpu-gpu-optimization`; coordinator owns reconciliation.
Scope: read-only primary-source research; no engine execution, build, benchmark,
model/SDK download or GPU use. Only this run directory was written.

## 1. Result and provenance

**Adopt mechanisms inside the existing Rust mathematical engine.** The highest
immediate candidates after complete original-vLLM parity are resident metadata,
qualified graph/launch reduction, actual temporal ragged prefill, then source-safe
hybrid prefix/COW. Combine them through existing SourceIdentity, private StateRoot,
ResourceArbiter and real completion ownership. Low-bit state, new arithmetic,
drafting, weight streaming and distributed execution remain independent policies
with their own correctness, quality and physical gates. No speedup is established
by this research.

| Boundary | Exact version | Meaning |
| --- | --- | --- |
| Frozen upstream main | [`8e2beaa3bff595577a069d49d96bde9074294ec2`](https://github.com/vllm-project/vllm/commit/8e2beaa3bff595577a069d49d96bde9074294ec2), committer `2026-10-21T04:20:45Z` | All pinned code/doc links below use this tree unless stated. Main is not the stable release. |
| Latest stable | [`v0.32.0`](https://github.com/vllm-project/vllm/releases/tag/v0.32.0), published `2026-10-18T06:50:48Z`; tag resolves to `5df63a7203cdb9328d824a27b682e77c7b9928be` | Release metadata inspected separately; a path existing in release proves neither all current behavior nor runtime support. |
| Actual local comparison | [`db9527a46873454610df6dbedf79a36d6bf1a7f6`](https://github.com/vllm-project/vllm/commit/db9527a46873454610df6dbedf79a36d6bf1a7f6), `2026-10-02T22:18:12Z`; vLLM `0.31.0` | Coordinator's pinned Torch2.13/cu129, SM86 RTX3060Laptop6GB, fixed relative5GiB operator target. Keep this baseline fixed. |

Live GitHub API resolved those SHAs. The web reader returned an older cached
`main` response initially; direct live API metadata and immutable raw URLs were
used to freeze the research. `upstream-metadata.json`, tree manifests and
`version-presence.json` retain the read provenance; fetched files are text only.

Main/release/reference all contain the MRV2, DFlash2, Gemma4 MTP, HiSparse and
hybrid-cache paths inspected here, but many blobs differ. This report does not
claim those features were introduced after0.31. The inspected Gemma4 speculator
is byte-identical main↔0.32 but differs from0.31; Qwen3.5/Gemma4 model files also
match0.32 and differ from0.31. The adaptive-verification implementation happens
to match all three frozen trees. Feature execution still depends on config,
model, backend, dtype, topology and build.

**Removed versus current:**0.32 release notes remove cascade attention, the old
`torch.compile` sequence-parallel/AsyncTP passes and `mamba_cache_mode="all"`.
The pinned0.31 tree still contains related source/tests. Do not promise current
cascade attention, conflate old AsyncTP with DBO, or revive an obsolete cache
mode. New model-specific sequence partitioning is a different mechanism.
[Release notes][release]; [current cache types][cacheconfig].

## 2. Architecture checked in code

- Frontend processes own HTTP, tokenization, multimodal preparation and output
  delivery; EngineCore owns scheduling/cache coordination; workers execute model
  math. DP adds replicas/coordinator; TP/PP adds workers. Multiprocess structure
  addresses Python overhead and distribution; it is not a required Rust design.
  [Architecture][arch].
- V1 scheduler expresses work as the deficit between computed tokens and known
  prompt/output/speculative tokens, under token, request, encoder and KV budgets.
  Running work is handled before waiting admission; priority/FCFS and preemption
  are explicit. There is no separate mathematical prefill/decode engine.
  [Scheduler, lines622–970][scheduler].
- `EngineCore.step_with_batch_queue()` dispatches nonblocking work, keeps a bounded
  queue of futures, processes aborts before integrating results, and defers grammar
  sampling when it needs prior outputs. This is CPU/GPU pipeline overlap, not proof
  of simultaneous kernels. [EngineCore][engine].
- MRV1 is `vllm/v1/worker/gpu_model_runner.py`; MRV2 is
  `vllm/v1/worker/gpu/model_runner.py`. **MRV2 is not a vLLM V2 scheduler/second
  mathematical engine.** It uses stable active-request rows, GPU gather/input
  preparation, diff uploads and modular sampling/speculation. Its design doc
  explicitly retains feature/test gaps; default selection is version/platform
  dependent. [MRV2 design][mrv2doc]; [runner][runner]; [request rows][requeststate].
- Cache API → coordinator → per-spec managers → block pools. Actual current
  hybrid lookup performs a monotone fixed-point intersection across participating
  groups. The older design doc's "exactly full attention plus one other" limit is
  stale relative to this source. Cache identity and physical storage are separate;
  non-cacheable scratch groups do not participate. [Coordinator][coordinator].
- Compile/fusion, attention, linear and MoE backends are capability-selected
  implementation strategies. CUDA graph dispatch is separate from compilation.
  Target-authoritative rejection sampling is separate from proposal generation.
  Those are useful policy boundaries for the one Rust engine, without importing
  vLLM's allocator or Python serving stack. [Graph manager][graphcode];
  [kernel config][kernelconfig]; [speculator][speculator].

## 3. Local coverage labels and actual anchors

`A` = existing accepted **bounded** route, backed by inspected current code plus
current plan records; not a fresh execution by this agent. `I` = bounded isolated
implementation/qualification supplied by coordinator and visible in this dirty
worktree, not production. `P` = contract/TODO only. `N` = no equivalent route found
in inspected production scopes. `H` = hardware/model inapplicable to current SM86
Qwen scope. Multiple labels separate foundations from the missing optimization.

| Anchor | Inspected local source / current record | What it establishes; limitation |
| --- | --- | --- |
| L1 | `crates/zerogravity-core/src/{source,session_root,state,resource_arbiter}.rs` | One source authority, independent session/root publication, process-shared physical budget and unknown-work fence. SourceIdentity is process-local allocation identity, **not** a portable content hash. |
| L2 | `crates/zerogravity-cli/src/serve/{gpu_batch,gpu_engine/cohort}.rs` | Finite source-aware waves; B2/4/8 unpadded survivor regrouping; private all-row publisher; teacher-prefix/tail partition. Native row means one token at its own frontier. It does not establish time-parallel multi-token prefill. |
| L3 | `crates/zerogravity-gpu/src/resident_inference.rs`, `resident_inference/fused.rs`, `resident_program/batch.rs` | Shared immutable ResidentModel; serialized physical workspace; private per-session domains; actual completion before publication. Multiple native request rows exist within qualified GGUF scope. |
| L4 | `crates/zerogravity-gpu/src/resident_inference/graph{.rs,/qualification.rs}`, serving TODO production-graph item | Private full-model graph qualification exists; complete same-card constructor/transport/external-storage/final-drop ownership and graph metadata admission still block production selection. |
| L5 | `crates/zerogravity-gpu/src/resident_{greedy,batch_greedy}.rs`, `resident_program/bf16_greedy.rs` | Bounded device greedy packets and output feedback foundations. General stochastic sampler and speculative branch runtime remain open. |
| L6 | `crates/zerogravity-gpu/src/resident_inference/native_fragment/full_model_qualify.rs`, `resident_program/native_full.rs`, core `accelerator/program/vllm_model.rs` | Source-produced five full24 phases and full-model typed operations inside original owner/node traversal. The source explicitly says diagnostic/unpublished; receipt has `model_root_published:false`. |
| L7 | `crates/zerogravity-gpu/src/vllm_cuda_gdn/`, `vllm_cuda_qualification/`, core `accelerator/program/vllm_{gdn,native,model}/` | Native-source arithmetic, BF16 physical values, imported pinned kernel qualification and actual BLAS allocator ownership are isolated bounded strategies. They do not close natural generation or HTTP parity. |
| L8 | `crates/zerogravity-artifacts/src/{ssd_extent.rs,source_weights/tensor_window.rs}` and `memory-and-streaming/TODO.md` | Verified immutable leased source windows/coalesced positional IO; no production SSD-streamed model, async H2D readiness, eviction or expert cache. |
| L9 | `crates/zerogravity-core/src/{context_snapshot,context_filter}.rs`; serving TODO context item | ContextOwner/Reader grants, immutable records and freshness foundations. Actual router/session binding, delivery recheck, multi-tenant shared context and trained model consumption remain open. |
| L10 | `tools/serve-{bench,compare,lifecycle}.py`, `crates/zerogravity-cli/src/serve/token_metrics.rs`, serving TODO | Owned serialized provider lifecycle, bounded timing/ID observations and some hardware-count diagnostics. Broad matched source-correct goodput/tails parity remains open. |
| L11 | `cpu-gpu-optimization/TODO.md`, model-execution/memory/serving plans | Explicit open SIMD, tensor arithmetic, graphs, tiering, speculative state and other-model contracts. A checklist is not implementation. |

Coordinator's current isolated checkpoint: original0.8B whole full24/fivephase
`1/1/64/2/1`, all42 domains, complete vocabulary and five greedy outputs EXACT,
plus fault/regression/required gate PASS; roots remain forced353/unpublished.
**This statement is supplied checkpoint evidence, not my physical verification.**
I inspected L6/L7 and confirmed the diagnostic boundary. Natural feedback, served
root publication and complete0.8B/2B parity stay open; original2B reference capture
is underway. Original27B is cached/source-window qualified but runtime open;
Gemma4 contracts are present but trained runtime remains unqualified.

Current accepted production/plan observations can predate that isolated route.
For example serving TODO records earlier eager GGUF-vs-stock numbers with
non-equivalent activation/cast profiles, and source-aware GGUF worker/HTTP
qualification; none qualifies this original-HF route or a current performance win.

## 4. Optimization inventory

Each row names an actual mechanism, its prerequisites/semantic risk and the local
coverage. Linked code was inspected at the relevant definitions/dispatch/ownership
sections; broad support lists are documentation evidence only. No upstream tests
or kernels were executed here.

### 4.1 Scheduler, batches and host/device orchestration

| ID / mechanism | Source-visible vLLM behavior, prerequisite and interaction | Zerogravity mapping / action |
| --- | --- | --- |
| V01 Continuous native batching | Fill token budget across active/new requests and reuse weight residency; cap sequences, scheduled tokens and actual KV allocation. Batch shape changes can change floating arithmetic. [Scheduler][scheduler] | **A+I:** L2/L3 accepted GGUF B2/4/8; L6 original-HF isolated. Close full source/feedback parity before broadening bands/traffic. |
| V02 Chunked/ragged prefill | Budget-limited long-prompt chunks combine with ongoing decode. Long-prefill cap/adaptive fair share and cadence reduce interference; larger chunks trade ITL for TTFT/utilization. Scheduler granularity is distinct from temporal kernel parallelism. [Scheduler][scheduler]; [scheduler config][schedconfig] | **A+P:** finite ≤32-token/B1 quanta and native one-token teacher rows exist. Actual multi-token/time-parallel dense/GDN prefill and mixed ragged model programs remain open. |
| V03 Admission, watermarks and recompute | KV refusal can preempt a low-priority/running-tail request and replay later; full-input reservation and watermarks reduce thrashing. Queued request/token limits and active limits differ from runner buffer capacity. Recompute adds tails and wasted work. [Config][schedconfig]; [scheduler][scheduler] | **A+P:** L1 physical refusal/quarantine and bounded queues accepted; no paged production victim/recompute policy. Keep fairness/finite headroom, never turn logical eviction into physical credit. |
| V04 Async scheduler and bounded in-flight batches | Placeholder outputs permit scheduling the next step before CPU receives sampled IDs. Stale results/preemption require rollback-safe bookkeeping. Engine futures process aborts before output integration. [Async scheduler][asyncsched]; [queue][engine] | **P:** fair workers and device token feedback exist, but general asynchronous multi-flight production pipeline is not qualified. Retain per-attempt epochs and Ready before root publication. |
| V05 Stable request rows + GPU gather | MRV2 permanently assigns an active row, gathers per-step order, keeps computed counts/draft IDs on GPU and stages only changes; preemption ends that row lifetime. [Rows][requeststate]; [input batch][inputbatch] | **P:** L2 rebuilds cohorts; L3 retains source/state. Add finite leased row slabs and epoch-tagged handles inside existing worker, measuring metadata cost before porting wholesale. |
| V06 Diff upload, UVA and async-safe staging | StagedWriteTensor packs ragged writes; UVA avoids duplicating large prompt histories; round-robin pinned buffers must outlive all GPU readers. Explicit-copy fallback exists. [Buffer utilities][buffers]; [block tables][blocktable] | **P:** original leases/transport are usable foundations; no universal metadata/UVA route. Admitted pinned host bytes, generation/capture readers and fresh buffers remain required. |
| V07 Async compact outputs | Sampled IDs/counts/logprobs copied on a separate stream after waiting for model work, with completion event before CPU access. Payload is compact; output synchronization still exists. [Output utilities][asyncutils] | **A+P:** L5 compact greedy packet accepted; L7 parity still forced. Reduce scalar/readback sync while retaining true token, error and publication boundaries. |
| V08 Grammar/output pipeline | Grammar-dependent sample stage may await prior accepted tokens even if forward work is ahead. Logit processor ordering is explicit; tokenization, parser, delivery and backpressure remain outside model kernels. [Engine][engine]; [sampler][sampler]; [grammar][grammar] | **A+P:** bounded greedy JSON/SSE/backpressure exists. Grammar/tools/stochastic output and async grammar state are unimplemented; never claim arbitrary API parity from fast forward. |

### 4.2 State layout, prefix reuse and tiers

| ID / mechanism | Source-visible vLLM behavior, prerequisite and interaction | Zerogravity mapping / action |
| --- | --- | --- |
| V09 Paged KV | Fixed-size physical blocks + logical block tables avoid contiguous per-sequence reservations; attention consumes pages. Tail waste, table bytes and block-size compatibility remain. [Cache interface][cacheiface]; [manager][cachemanager]; [pool][blockpool] | **P:** L3 currently allocates capacity-bound private domains. Implement physical page leases as views of original storage/arbiter; one state owner, no new allocator. |
| V10 Hash prefix caching | Hash chain binds parent prefix, tokens and extras such as adapter/media/salt; cache completed blocks and reference-count hits; full-hit last-token/logit work may still be needed. [Prefix design][prefixdoc]; [pool][blockpool] | **P:** aliases share weights, not an accepted full-state prefix cache. Bind portable source digest + process SourceIdentity + semantic profile + authorized context; cache only complete roots. |
| V11 Hybrid fixed-point cache intersection | Per-type managers may retain different dense/window/recurrent spans. Current coordinator repeatedly reduces candidate hit length until all groups agree; scratch groups opt out; arbitrary spec-group count is supported by this implementation. [Coordinator][coordinator] | **P:** Qwen's full KV + Conv/GDN all42 must agree at one frontier. Generalize to Gemma SWA/KV sharing later; do not equate matching token prefix with complete state coverage. |
| V12 Partial-tail COW | Fine hash granularity can reuse part of a larger block; before writing, allocate private destination and retain source+destination through the copy step. Deferred frees are fenced by processed step sequence. [COW][singlemanager]; [deferred free][scheduler] | **P:** private roots/checkpoints exist. Reuse them for immutable shared prefix pages and private mutation; same-step abort, source eviction, capture replay and ABA must not recycle either endpoint. |
| V13 Sliding/chunk-local/ring retention | Reclaim state outside a model-declared window; local/chunked and scratch-ring managers have distinct reuse/skip rules. This preserves source semantics only for actual source-local attention. [Managers][singlemanager]; [specs][cacheiface] | **P/H:** important for Gemma4 and other local models; not permission to truncate full-attention Qwen contexts. All causal frontier/window/replay boundaries need model qualification. |
| V14 GDN/Mamba prefix checkpoints | `align` mode caches recurrent+conv checkpoints at valid boundaries; checkpoints inside supported prefill kernels reduce extra splitting. Shared-prefix junction stops allow siblings to reuse recurrent state. Extra state blocks and draft lookahead affect admission. [MambaManager][singlemanager]; [GDN checkpoint path][gdn] | **P+I:** L7 actual GDN state exists in bounded scope, full-domain prefix reuse absent. Pin every recurrent state schema/dtype and source arithmetic schedule. |
| V15 GPU↔CPU completed-prefix offload | OffloadingConnector copies completed chunks through pinned CPU primary storage; LRU/ARC policy, prompt-only selection and load/offload token caps control work. CUDA/ROCm/XPU implementation support is conditional. [Guide][kvguide]; [CPU manager][cpukv] | **P:** source IO L8 is unrelated to produced-state offload. Add root-bound leased state serialization/copy completion; no hit until bytes and schema are valid. |
| V16 CPU↔filesystem/object/P2P tiering | Secondary tiers stage through CPU; configuration-digest/hash filenames; bounded policies and asynchronous lookup/promotion; peer polling can run between engine steps. [Tier manager][tiermanager]; [KV guide][kvguide] | **P:** no product state tier. SourceWeight windows must remain a separate identity domain from mutable causal state, while sharing budget/IO scheduling machinery. |
| V17 Simple CPU/disk KV connector | Simpler separate backend offers explicit pool sizing and disk-copy workers. Connector names do not imply interchangeable full hybrid support or identical lifetime semantics. [Disk backend][simpledisk]; [worker][simpleworker] | **N/P:** evaluate one coherent cache/tier design; do not independently port every alternative connector or create duplicate budget authorities. |
| V18 HiSparse host/hot/resident resolution | **Experimental sparse-MLA-specific:** coordinator owns host block identity; normal cache manager owns GPU pages; fused resolver checks resident→hot→pinned host with GPU LRU. Spec verification unions rows so they cannot evict each other's readers. [Design][hisparsedoc]; [runtime paths][hisparsetree] | **H/P:** not current Qwen dense/GDN or Gemma4 first slice. Transfer the no-CPU-decision/hot-miss and split enqueue-vs-complete ideas to future expert/state caches, preserving stricter ZG actual completion. |
| V19 Cross-rank/shared physical host cache | Some HiSparse MP-TP layouts map one host pool with writer IPC events; logical per-replica capacity is not physical per-node bytes. [HiSparse design][hisparsedoc] | **P/H:** relevant later UMA/two-device composition; no automatic GB10 shared-memory or coherence claim. Charge real host pools once and every private view/metadata separately. |

### 4.3 Loading, physical residency and CPU

| ID / mechanism | Source-visible vLLM behavior, prerequisite and interaction | Zerogravity mapping / action |
| --- | --- | --- |
| V20 Loader format and IO strategy | Lazy/eager/prefetched safetensors, optional multithread load, fastsafetensors/instanttensor, PT/npcache and alternate loaders. Multithread default-loader path rejects non-lazy strategy combinations; no silent strategy override. [Default loader][loader]; [iterators][weightutils] | **A+P:** verified windows/coalesced IO L8. Stream only indexed/hash-verified original bytes into admitted staging; cache warmness and setup cost belong in measurement. |
| V21 Sharded/local-expert loading | Load pre-sharded checkpoints rather than reading complete tensors per TP worker; EP filtering skips non-local experts when eligible, with EPLB constraints. [Sharded loader][shardedloader]; [loader][loader] | **P/H:** future multi-device/27B streaming. Bind original logical roles, geometry and source identity before filtered physical placement. |
| V22 Cold packing and merged projections | Load Q/K/V and gate/up into merged projection layouts, reorder scales/packs, use capability/shape-specific quantized GEMMs; prepare once. Same source codec can be repacked without adding quantization, but output arithmetic may change. [Linear][linear]; [Marlin][marlin]; [Qwen][qwen] | **A+I+P:** existing native batch projections and isolated source-native operators. Keep pack fingerprint separate from source; compare all outputs/state under each arithmetic profile. |
| V23 IPC prepared-weight cache | Daemon shares postprocessed weights by CUDA IPC or clones them; fingerprints/platform/quant constraints apply. Zero-copy mode is incompatible with sleep offload of daemon-owned storage. [IPC loader][ipcloader] | **N/P:** sharing within one ResidentModel already exists. Prefer extending immutable source owner; do not add foreign daemon allocation authority without a physical ownership design. |
| V24 UVA weight offload | GPU directly reads pinned host parameters; fallback uses on-demand functional copies. Parameter-name segment selection and byte caps control what leaves VRAM. PCIe bandwidth can dominate every forward. [UVA offloader][uvaoffload] | **P:** useful capacity strategy for27B only after original windows→copy readiness; never call host addressability extra VRAM or promise faster decode. |
| V25 Asynchronous layer prefetch | Static typed/strided buffer slots, copy stream events and wait/start ops permit compile/graph capture; more lookahead hides latency but reserves more GPU slots. Derived from SGLang code. [Prefetch offloader][prefetchoffload]; [offload config][offloadconfig] | **P:** strong future streaming mechanism under one arbiter. Protect encoded, captured, queued and active readers from circular-slot reuse; resident fast path must do no offload bookkeeping/IO. |
| V26 CPU topology/thread controls | NUMA memory/core binding, reserved frontend cores, per-worker OpenMP placement and CPU KV budgets; eager Zen blocked-layout prepack reduces repeated conversion. [CPU guidance][cpudoc] | **P:** existing CPU math is not SIMD/topology maximality. Qualify AVX2/AVX512/VNNI/AMX/ARM policies only on actual available hardware with source-rounding and oversubscription checks. |
| V27 Sleep, discard and wake | Optional managed pools offload/discard weights/KV; selective wake, graph-pool backup and experimental NCCL suspend have distinct prerequisites and costs. [Sleep][sleepdoc] | **P:** model lifecycle policy can reuse immutable source/residency owner, not silently discard private sessions. Unknown-work quarantine cannot be cleared by sleep/deadline. |
| V28 Startup compile/kernel caches | Persisted compile cache, preloaded autotune/kernel artifacts and explicit cache-size configuration trade cold start against steady-state memory. Source/compiler/GPU/config changes invalidate entries. [Compile][compiledoc]; [release][release] | **P:** original prepared programs exist; versioned compiled artifacts must be admitted, source-bound and invalidated safely. No kernel package downloads authorized by this research. |

### 4.4 Attention, recurrence and precision

| ID / mechanism | Source-visible vLLM behavior, prerequisite and interaction | Zerogravity mapping / action |
| --- | --- | --- |
| V29 Fused exact-pattern attention | FlashAttention/FlashInfer/Triton combine softmax/attention memory movement and choose decode/varlen paths. Algorithm avoids materializing full score matrices; implementation reduction/cast ordering still differs. Backend validates dtype/head/page/capability. [Registry][attnregistry]; [FA][flashattn]; [FI][flashinfer] | **I+P:** pinned attention images/arithmetic in L7; production exact-source route incomplete. Temporal prefill must close full logits/all42 plus edge/long contexts before promotion. |
| V30 Source window/sink and mixed masks | Attention capabilities include source-declared SWA, sinks, bidirectional multimodal prefixes, logit softcaps and causal decode. Composite backends select per-current-query mask, sharing compatible KV/workspace. [Attention design][attndoc]; [Gemma4 source][gemma] | **P/H:** required Gemma4 semantics, not generic sparse approximation. Text-only correctness first; media/mask contracts and changed cache identity separately. |
| V31 MLA and sparse indexed attention | Compressed latent KV, backend-specific expanded/ragged prefill versus latent decode, sparse model indexer/top-k and index sharing. Requires model-trained MLA/sparsity semantics and hardware/backend support. [MLA layer][mla]; [FlashMLA][flashmla]; [FlashInfer MLA][fimla] | **H/P:** defer DeepSeek/other MLA and sparse Qwen variants; do not insert MLA into Qwen3.5 or Gemma4 dense attention. Shared indexing is model/layer semantic sharing, not arbitrary cache aliasing. |
| V32 FP8 KV / selective layers | Per-tensor or calibrated per-head Q/K/V scales; some FA3 modes quantize Q too; skip sensitive layers. Scale storage, calibration and activation math are part of identity. [KV quant guide][kvquant]; [cache config][cacheconfig] | **P:** derived quality policy, not original StateRoot precision. SM86 format storage does not imply native FP8 tensor cores. Compare quality and genuine memory/SLO benefit at declared profile. |
| V33 NVFP4 KV | Native pack/unpack kernels plus backend-specific attention support;0.32 expands SM8x/SM90/SM12x coverage, but head/page and kernel paths remain conditional. [KV kernels][nvfp4kv]; [FI backend][flashinfer] | **P/H:** possible capacity experiment later; current Gemma4 head512 support extension is open PR62117. Never infer full model support from a standalone pack kernel. |
| V34 TurboQuant KV | FP8-K/4-bit-V or rotated Lloyd-Max3/4-bit-K and3/4-bit-V presets; norm correction and boundary-layer policy; compressed decode, continuation paths and workspace reservations. This is lossy state. [Config][tqconfig]; [backend][tqbackend] | **P:** later separately governed calibration/quality experiment. Source comments include empirical PPL claims, not our evaluation. Long continuation must avoid repeated whole-cache dequant amplification. |
| V35 UltraQuant KV | Separate4-bit E2M1/UE8M0 group32 format with rotated K; BF16 frontend,64-token preferred pages; gfx950 FlyDSL scaled-MFMA fast path plus fallbacks. Sink models rejected by backend. [Format][uqformat]; [backend][uqbackend] | **H/P:** no Renoir/gfx90c launch, no current SM86 adoption claim. Retain format/rotation/scales/version as derived-state identity. |
| V36 GDN/Mamba fused decode | Qwen GDN merges projections/Conv/gating/state/output work using platform-specific kernels. Current CUDA fused decode requires BF16 model/Conv, K=V128, supported state dtype/layout and CC≥8.0; packed recurrent/spec paths are separate dispatches. [GDN source][gdn]; [CUDA kernel][gdncu] | **I+P:** L7 real bounded pinned GDN math; broad batch/chunk/natural-feedback source parity still required. State acceptance must include Conv, recurrent and every attention domain. |
| V37 GDN temporal chunk prefill | Chunk recurrence/scan kernels exploit temporal work and checkpointing. Main selects FlashInfer on Hopper, or CUDA13 Blackwell with K128; in-tree CuteDSL opt-in SM10x; ROCm AITER FlyDSL is CDNA3+ with shape/dtype guards; other paths use Triton/FLA/SYCL. [Dispatch][gdn]; [metadata][gdnbackend] | **I+P/H:** SM86/cu129 uses the compatible Triton/FLA lineage, not Hopper/Blackwell fast path. Preserve declared chunk64/cast/reduction schedule and compare trained all-state outputs. |
| V38 BF16/FP16/FP32 arithmetic policy | Precision, reduction, tensor-core kernels and activation/sampling casts change numerical behavior even for unchanged source weights. Beta batch-invariance mode pins selected kernel/reduction behavior and can sacrifice performance; it is not universal model parity. [Batch invariance][invariancedoc]; [kernel dispatch][kernelconfig] | **A+I+P:** original WideF64 and isolated pinned-vLLM BF16/F32 profiles remain distinct. Never silently substitute “faster equivalent” GEMM or infer Qwen3.5/Gemma4 invariance from a tested-model list. |

### 4.5 Dense/MoE kernels, fusion, graphs and tuning

| ID / mechanism | Source-visible vLLM behavior, prerequisite and interaction | Zerogravity mapping / action |
| --- | --- | --- |
| V39 Dense GEMM/backend selection | Merged linears choose cuBLAS/CUTLASS/Marlin/Machete/FlashInfer/Humming/b12x etc by format, shape and capability; small-M decode and large prefill need different plans. [Linear][linear]; [kernel config][kernelconfig]; [Marlin][marlin] | **A+I+P:** reuse typed operator strategy/physical layout in existing engine. Tune admitted workload bands, not an unbounded autotuner inside requests; fingerprint compiler/driver and original rounding. |
| V40 Weight/activation quantization | AWQ/GPTQ, INT8/W8A8, FP8, block FP8, MXFP4/NVFP4 and online methods reduce bytes/compute with codec/backend constraints. Reading a quantized original source versus deriving a new quantized source are different claims. [Quant overview][quantdoc]; [FP8 implementation][fp8] | **A+P:** original codecs accepted only within their own scopes; extra low-bit/calibrated packs need separate identity, held-out quality and physical tests. SM86 lacks native Hopper/Blackwell FP8/FP4 math. |
| V41 b12x packed SM120/121 kernels | Optional dense/embedding/MoE support includes FP8/FP4 and select ggml block codecs; BF16 activations or format-specific variants. MoE does not support EP/expert maps/EXL3/NF3. [b12x guide][b12xdoc]; [MoE wrapper][b12xmoe] | **H/P:** relevant to future GB10 review, not SM86. Preserve original block boundaries; do not assume all Blackwell subfamilies share kernels. |
| V42 Norm/residual/activation fusion | Add+RMS, norm+quant, activation+quant, RoPE+cache, QK-norm+RoPE+cache and MLA variants eliminate launches/memory traffic. Specific dtype/shape/backend/graph visibility gates apply; FP8/NVFP4 fusions also change precision policy. [Fusion passes][fusiontree]; [QK pass][qkfusion] | **A+I+P:** some original/native fusions and L7 dualRMS paths exist. Audit exact intermediate casts, signed zero, epsilon, RoPE data and aliasing; do not merge merely similar formulas. |
| V43 MoE routing + grouped experts | Route top-k, permute/quantize tokens, grouped GEMMs, activation and weighted reduction; modular prepare/finalize separates communication from experts. Layout/activation/weight-on-input semantics must match. [Modular kernel][moemodular]; [kernel feature map][moedoc] | **P/H:** Qwen27B runtime first; later Gemma26B/sparse Qwen. Coalesce only actual source-selected experts and retain per-request routing/state. |
| V44 EPLB and expert replicas | Balance measured logical-expert traffic across physical slots/replicas, preserve logical IDs and minimize migration; redundant copies consume real memory. Elastic changes need drained/committed mappings. [EPLB policy][eplb]; [EP guide][epdoc] | **P/H:** future distributed MoE/expert cache. Router probabilities and model identity cannot follow physical placement heuristics. |
| V45 Expert substitution/pruning | MoNE checkpoints replace designated experts with constants, keep logical routing and compact retained weights; restricted configurations. It is a different derived model, not source-preserving scheduling. [Expert substitution][monedoc] | **P/H:** defer until explicitly approved derived quality work; never hide it in expert residency/eviction policy. |
| V46 Piecewise compilation and IR | Dynamic graphs split around incompatible ops; compiler fusions/custom ops and cache keys include policy/version dependencies; static/range specialization amortizes compile. Optimization levels trade startup vs execution; O3 currently equals O2. [Compile][compiledoc]; [levels][levelsdoc]; [config][compileconfig] | **A+P:** typed ForwardPlan/prepared kernels exist; not `torch.compile` adoption. Add guarded strategy specializations inside the same math graph, with cold costs and source/driver identity. |
| V47 Full/piecewise CUDA graphs | Modes NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE choose compatible captures by shape, query class, adapters and microbatches. Stable addresses/padding and pools cost memory; unsupported batches fall back. [Graph source][graphcode]; [design][graphdoc] | **A-private+P-production:** L4. Close all physical admission/transport/lifetime holes first; capture/replay is a launch strategy, never completion or authority to publish. |
| V48 Fused multi-step draft graph | Target-dependent AR drafters can capture multiple dependent steps once. Backends must update all derived metadata through capture-safe GPU ops and persistent storage; fixed-position Gemma assistants differ. [MRV2 design][mrv2doc]; [speculator][speculator] | **P:** after basic target-authoritative speculation and production graphs. Source-specific metadata generation must be graph-bound and retained until true completion. |
| V49 Autotune and profiling | Backend GEMM/attention tuning, warmup before filling memory, cached kernel/config tables and policy-specific cost curves. GDN source explicitly warms autotuned prefill while memory is available. [GDN warmup][gdn]; [kernel config][kernelconfig]; [adaptive code][adaptivecode] | **P:** bounded offline/pre-admission tuning; include transient candidate/workspace and cold overhead. No unconstrained tuning on the critical path or under reserved-memory fiction. |
| V50 PDL / launch dependency overlap | CUDA programmatic dependency launch enables eligible SM≥9 kernels to launch ahead with proper completion signals; fusion code guards output-scale early-read hazards. [CUDA capability][cudaplatform]; [allreduce fusion][arfusion] | **H/P:** not SM86. Future capable CUDA strategy requires per-producer dependency/physical stress tests; PDL is not permission for early storage release. |

### 4.6 Sampling and speculation

| ID / mechanism | Source-visible vLLM behavior, prerequisite and interaction | Zerogravity mapping / action |
| --- | --- | --- |
| V51 GPU sampler and selected logprobs | MRV2 processes per-request parameters through indirection, Gumbel RNG and optional FlashInfer/XPU sampling, plus top-k-first logprob work. Bias/penalty order matters; prompt logprobs and large verify logits can be chunked. [Sampler][sampler]; [Gumbel][gumbel]; [rejection adapter][reject] | **A-greedy+P-general:** L5. Seed/counter and grammar belong to request branches; preserve target dtype/casts and optional full-logit diagnostic route. |
| V52 Standard/block rejection verifier | Greedy uses authoritative target argmax; probabilistic path accepts/corrects with target/draft distributions and residual resampling. Block verifier is another explicit method; synthetic acceptance is diagnostic. [Rejection math][rejectutils]; [config][specconfig] | **P:** target-only baseline then exact accepted-prefix tests. Restore every KV/recurrent/grammar/RNG/index/media domain on rejection, not just token IDs. Distribution correctness is not bit-identical batch numerics. |
| V53 N-gram CPU/GPU proposal | Prompt/history lookup proposes known continuations without learned draft weights; GPU variant keeps/scans dense token history and supports variable draft lengths. Verification still costs compute/state. [GPU ngram][ngram]; [ngram guide][ngramdoc] | **P:** first low-risk speculative candidate after parity; cap history and depth by actual bytes/load, and skip when no useful match or verification harms goodput. |
| V54 Suffix/global-tree proposal | Suffix trees reuse prompt/output statistics and vary speculative depth; global cached requests introduce memory and tenant-isolation considerations. MRV1 implementation exists; MRV2 suffix support is an open PR in this snapshot. [Suffix][suffixdoc]; [open PR60339](https://github.com/vllm-project/vllm/pull/60339) | **P:** no cross-tenant proposal cache without grants; source/tokenizer/history identity and fair bounded storage apply even though target verifies. |
| V55 Separate draft, EAGLE/EAGLE3 | Independent small AR draft or target-hidden-state-conditioned head; separate or shared embeddings/head, model-specific taps and draft state. Parallel PARD-style proposal is optional. [Speculator][speculator]; [EAGLE wrapper][eagle]; [draft guide][draftdoc] | **P:** share immutable source where truly compatible; private draft roots and target-authoritative commit. Extra model weights/compute/activation taps must earn measured useful work. |
| V56 Native/multi-module MTP | Use trained MTP heads; multi-module prefill lookahead and draft state require special boundaries; quantized draft-only head can affect proposals while target stays authoritative. [MTP code][mtp]; [MTP guide][mtpdoc]; [scheduler][scheduler] | **P:** original Qwen backbone's excluded MTP tensors are not qualified speculative execution. Source inventory, actual head contracts, all prefix lengths and rollback first. |
| V57 DFlash/DFlash2/DSpark | Block/parallel drafters exploit target hidden taps; DFlash2 has separate source path. Do not equate a class being present with all target combinations, masking and cache support. [DFlash][dflash]; [DFlash2][dflash2]; [config][specconfig] | **P/H:** later Qwen/Gemma variants only when their actual trained drafter exists/is authorized. Adaptive Qwen GDN DSpark integration remains open PR53929. |
| V58 Gemma4 assistant MTP | Q-only assistant reads target KV; source maps matching sliding/full layer types and copies target KV scales. Positions do not advance during draft loops; embeddings may share. [Gemma speculator][gemmaspec]; [model][gemmamtp] | **P:** after Gemma trained target runtime. Cross-model cache sharing is an explicit architecture contract; it does not permit arbitrary two-model StateRoot sharing. |
| V59 Dynamic depth and adaptive verification | Batch-size K schedule or confidence/cost-budgeted per-request verification; target/draft costs have graph-padding discontinuities. Main adaptive path requires compatible device-length attention/full graphs and rejects LoRA/PP; dynamic-K DP independently selecting K can deadlock collectives. [Adaptive][adaptivecode]; [requirements][adaptivedoc]; [dynamic][dynamicdoc] | **P:** useful eventual shared-worker policy. Optimize marginal accepted target tokens per occupied resource/time, choose K=0 under pressure; qualification includes negative/no-speedup cases. |
| V60 Lossy verification/cross-vocab drafts | FLy deliberately relaxes target verification; heterogeneous-vocabulary mapping is limited to supported draft sampling modes. Exact stochastic distribution and exact target trace are different guarantees. [FLy][flydoc]; [vocab/config][specconfig] | **P/H:** keep FLy disabled for original-source parity. Cross-vocab draft requires independently verified token mapping; never make it a hidden fallback. |

### 4.7 Distribution, placement, multimodel and observability

| ID / mechanism | Source-visible vLLM behavior, prerequisite and interaction | Zerogravity mapping / action |
| --- | --- | --- |
| V61 TP/PP/DP composition | TP shards per-layer weights, PP layers, DP requests; memory relief versus collectives/bubbles/replica costs. Dense DP can progress independently; MoE ranks must coordinate compatible forward counts. [Parallel guide][paralleldoc]; [DP][dpdoc] | **P/H:** no distributed runtime qualification. For2×GB10 use measured links/coherence/capacity; parent hardware agent owns recipe. Do not import multi-engine ownership as Rust architecture. |
| V62 Decode/prefill context parallel | Context-sharded attention/KV reduces duplication or long-prefill cost, requiring backend/spec/topology support and collective reduction of partial results. Current code also validates hybrid replicated groups. [CP guide][cpdoc]; [coordinator checks][coordinator] | **P/H:** later long contexts; no Qwen/Gemma inference claim from general DCP/PCP docs. Quantify communication and replicated non-attention state. |
| V63 Collective specialization | NCCL, custom P2P allreduce, FlashInfer and platform alternatives; actual topology/world-size/size/alignment gates and fixed-order deterministic modes. >2 PCIe-only devices can disable custom path. [Custom allreduce][allreduce]; [batch invariance][invariancedoc] | **P/H:** new physical transport strategy under original completion/arbiter. Collectives consume admitted scratch/IPC and must quarantine partial failures. |
| V64 Dual batch overlap (DBO) | Split microbatches and overlap MoE all-to-all with computation; coordinated ranks, eligible prepare/finalize and graph behavior. Doc describes DP+EP/DeepEP scope; inspect backend-specific current code before expanding. [DBO design][dbodoc]; [modular kernel][moemodular] | **H/P:** not generic single-SM86 prefill/decode overlap. Future multi-device MoE only; fair local chunks/stream overlap are a different task. |
| V65 P/D and encoder disaggregation | Independent placements/SLO tuning; NIXL/Mooncake/offload connectors transfer produced state/embeddings. P/D is experimental and documentation explicitly denies a universal throughput gain. [P/D guide][pddoc]; [encoder guide][encoderdoc] | **P/H:** defer remote execution; one engine can first split local scheduling classes. Transfer entire required state identity/frontier, not only KV for GDN. |
| V66 Transfer leases and push/pull | Producer lease+heartbeats protects queued/in-flight consumers; notifications return completion; push planning and heterogeneous layouts have separate protocols. Forward-loop heartbeats can be delayed. [NIXL leases][nixldoc]; [push][pushdoc]; [worker][nixlworker] | **P/H:** borrow transaction/epoch/reader-retention ideas. A timeout or peer silence never proves GPU/remote DMA stopped; ZG completion/quarantine remains authoritative. |
| V67 LoRA/multi-model tenancy | Adapters batch atop one backbone; adapter IDs affect prefix identity and graph shape; adapter limits consume memory. Sleep and weight residency enable serial multiplexing, not arbitrary concurrent independent-model fusion. [LoRA][loradoc]; [IPC][ipcloader] | **A-alias+P:** aliases share immutable source now. General adapters/hot reload/model multiplexing need separate source epochs, all physical charges and fair work. Never share different models' KV from similar prompts. |
| V68 Frontend/media reuse | Rust fast tokenizer option, API scale-out, processor/shared-memory caches and encoder-output reuse can reduce host/IPC overhead. Local snapshot/tokenizer version and trusted media identity matter. [Tuning][optimizationdoc]; [encoder][encoderdoc] | **A-text+P:** Rust tokenizer and bounded HTTP exist; actual media runtime absent. Reuse authorized immutable context/tokenization before considering GPU tokenization. |
| V69 Metrics and tuning objective | Request/queue/prefill/decode timing, prefix/tier/preemption/acceptance metrics and optional per-request IDs aid analysis; metrics have overhead and clock boundaries. API TTFT described here starts at scheduling, not client offer. [Per-request][metricsdoc]; [metrics design][metricdesign] | **A-bounded+P:** L10. Preserve offer/arrival/admission/first-commit/ID-delivery clocks, true token ITL and correct SLO-goodput; sampled counters are not actual hardware work. |

## 5. Shipped code, open work and numerical limits

All linked inventory paths are present in frozen main. Some mechanisms are
explicitly experimental (HiSparse/P-D, custom proposer, selected controls), and
runtime compatibility is narrower than file presence. Read-only code inspection
cannot validate vendor binary kernels or platform capability claims. It also does
not establish all source formats/model families/backends in a generic table.

Open GitHub PR search snapshot was captured separately in `open-pr-snapshot.json`.
These were **open**, not shipped mechanisms, at lookup; search listings are not
proof the proposal is correct or exhaustive:

| Open PR | Boundary that must stay open |
| --- | --- |
| [61779](https://github.com/vllm-project/vllm/pull/61779) | GDN mixed-batch static metadata/grids, graph preparation1/2. |
| [52244](https://github.com/vllm-project/vllm/pull/52244) | Hybrid GDN prefix hits with MTP. |
| [54163](https://github.com/vllm-project/vllm/pull/54163) | DFlash/DSpark trailing prefix block handling and Mamba recompute. |
| [53929](https://github.com/vllm-project/vllm/pull/53929) | Adaptive DSpark verification on Qwen GDN. |
| [60339](https://github.com/vllm-project/vllm/pull/60339) | MRV2 suffix speculator; do not infer it from MRV1 suffix. |
| [55947](https://github.com/vllm-project/vllm/pull/55947) | Uno draft-only-LoRA shared-model speculation. |
| [62117](https://github.com/vllm-project/vllm/pull/62117) | NVFP4 KV head512/Gemma4 SM100 extension. |
| [47507](https://github.com/vllm-project/vllm/pull/47507) | Gemma4 `k_eq_v` compressed-tensors shard aliases. |
| [62525](https://github.com/vllm-project/vllm/pull/62525) | Gemma4 MoE ignoring CUDA-graph padding rows. |
| [62790](https://github.com/vllm-project/vllm/pull/62790) | MRV2 grammar bitmask int64 logit row offsets. |

Code-over-prose qualifications:

- Hybrid fixed-point algorithm supports more than the old doc's two-group case.
- Generic speculation README's glance table says adaptive verification is
  DSpark-only; dedicated guide/config/current manager describe broader model-based
  drafters. Backend and Qwen GDN integration gaps still apply. Use code+specific
  guide, not that table alone.
- Gemma4 assistant speculator comments mention a forced Triton backend; current
  attention docs contain Hopper/Blackwell composite paths. Preserve actual
  selected backend and source config; comments do not prove a universal forced
  backend across all Gemma4 deployments.
- TurboQuant backend's introductory comment describes an older FP16-V layout;
  current config/store call sites carry3/4-bit values and packed sizes. Format
  identity must come from actual codec/version, not an introductory comment.
- Upstream's standard speculative losslessness is distributional under numerical
  assumptions; docs explicitly do not guarantee stable logprobs across runs/
  batch shapes. FLy is lossy. This never closes ZG's stricter full-logit/state
  source compatibility gate. [Speculation guide][specdoc].

**Claims versus verification:** vLLM docs/release notes contain kernel, startup,
API, HiSparse and throughput percentage claims on author workloads. I have not
reproduced any. No figure in those sources estimates a Zerogravity win. Reports
of format compression/PPL and multimodal batch-DP improvements likewise remain
author claims. Source inspection here verifies named dispatch/algorithm/ownership
structure, not those benchmark outcomes.

## 6. Composable Rust design hypotheses

These are implementation hypotheses for the coordinator's cross-engine design;
not an architecture replacement or accepted performance conclusion.

1. **One worker, several execution strategies.** Keep ForwardPlan mathematical
   operations and source precision fixed. A compatible batch descriptor includes
   source/model epoch, arithmetic profile, per-row root/frontier, request token
   spans, state schema, masks, pack/backend capability and output mode. Choose
   scalar/native batch/temporal prefill/graph from these facts. Scheduling can be
   general while Qwen/Gemma operators remain source-specific.
2. **One physical ledger for every strategy.** Charge weights, private/shared
   state pages, candidate roots, row metadata, graph objects/pools, compiler
   staging, pinned host buffers, IO slots, draft models and transfer readers
   before work. Credit returns only at original owners' real completion/retirement.
   Logical cache eviction removes an index entry; outstanding readers retain
   bytes and charge. No new engine, allocator, shadow StateRoot or host-VRAM sum.
3. **Metadata persistence with explicit epochs.** Stable rows and staged diffs can
   eliminate repeated cohort/position/table construction. Rows are handles with
   generation, source and session binding; a freed row cannot be reused while
   queued, captured or physical readers exist. Separate immutable CPU write
   payloads from in-flight DMA buffers. Early scheduling may prepare candidates,
   never publish them.
4. **Shared context has two different reuse levels.** Authorized ContextOwner
   snapshots can share immutable text/media/source records across applications.
   Same-source/tokenizer/template/profile requests can additionally reuse
   tokenization and a complete full-domain prefix checkpoint. Different models
   share authorized input content/IO, **not** arbitrary embeddings/KV/GDN state.
   `require_fresh` must be checked at actual delivery and cancel/revoke must
   invalidate future use without freeing existing physical readers early.
5. **One full-state prefix index, typed domain policies.** Key portable original
   source/content identity, process source/model epoch, tokenizer/template,
   adapter, precision/kernel semantics, tenant grant/salt and consumed IDs. Each
   domain exposes its valid restore/checkpoint frontiers and storage readers;
   coordinator finds a common frontier by monotone intersection. This handles
   Qwen full KV+Conv/GDN and later Gemma SWA/shared-KV without pretending every
   domain is a token-indexed KV array. COW only mutated pages/state.
6. **Temporal work and cross-request rows compose.** Pack ragged token spans
   within one source-compatible work group; dense projections process many tokens
   once while recurrence preserves causal source-chunk semantics. Decode urgency
   and checkpoint boundaries cap prefill chunks. Shared-prefix hits reduce actual
   work before batch shape is chosen, potentially freeing room for more requests;
   no expected multiplicative speedup is asserted.
7. **Shared residency/IO policy without mandatory IO.** The existing verified
   tensor-window owner can feed future resident→prefetched GPU strategies.
   Produced-state cache and original weight extents retain separate identities,
   but share fair IO queues, transfer staging and physical headroom. Expert
   streaming later reuses these mechanics with source-selected logical expert IDs
   and protected generations. A completely resident model incurs no per-token IO.
8. **Speculation spends spare capacity, not unbounded optimism.** Start with
   n-gram then native MTP/approved drafters. The same worker schedules target and
   draft operations; private branch roots hold every domain. Choose draft depth
   from measured marginal accepted useful tokens and load, with K=0 valid. Full
   rollback/target correction is mandatory; no synthetic acceptance in benchmarks.
9. **Placement is a strategy, not model semantics.** Local overlap, two-device
   tensor/layer partition, offload and remote transfers use explicit topology and
   completion contracts. Cross-model cache/context affinity is one scheduling
   signal subject to fairness, source isolation and memory; it cannot starve a
   request merely to preserve a hit rate.

## 7. Negative interactions and guardrails

| Combination | Risk / required decision |
| --- | --- |
| Graphs + metadata diffs/COW/offload | Stable captured addresses can outlive cache index membership. Retain every generation and all copy source/destination/graph readers; account graph pools. |
| Wider batches + original arithmetic | GEMM/reduction choices can change logits before GDN amplifies them. Requalify full vocabulary/all42, not just greedy agreement. |
| Temporal prefill + GDN checkpointing/MTP | Chunk/cast schedule and mid-chunk checkpoint grid change state; MTP known-token lookahead must not be replaced with sampled drafts near prefill tail. |
| Prefix reuse + variable masks/precision/adapters/context grants | Same text does not imply same state. Intersect all domains and include complete semantic/authorization identity. |
| Larger prefill + decode tails | Better GPU utilization can worsen ITL/starvation. Tune chunk cap/urgency against offered-load tails, with aging/finite fairness. |
| Speculation + saturated batching | Draft/verify slots, full-vocab buffers and private branch state displace useful target requests. Reduce K or disable; rejection count alone is not saved time. |
| Adaptive verifier + graphs/LoRA/PP/DP | Compatibility constraints and independent K choices can invalidate shape/collective assumptions. Capability matrix must fail closed. |
| KV compression + short continuation | Decompressing full cache per chunk can erase memory wins. Price continuation workspace and repeated read amplification. |
| Weight prefetch + graph captures | Circular slots cannot overwrite captured/queued readers; future-copy bytes reduce model/state capacity. Tune horizon under actual capacity. |
| Offload + host/UMA pressure | Pinned copies, process views, page cache and allocator pools consume physical memory beyond logical cache bytes. Neither VRAM/GTT addressability nor logical free equals OS headroom. |
| TP/EP/DBO + tiny models/PCIe | Collective/sync cost can exceed compute saved; microbatching/replication also consume memory. Use actual interconnect/capability measurements. |
| Sleep/reload + source immutability | Reload needs a new model/source epoch and cache invalidation; existing roots/graphs cannot silently read replaced weights. IPC zero-copy/sleep incompatibility is explicit upstream. |
| Cache-affinity routing + fair workers | Hot-prefix preference can strand cold tenants or overload one worker. Cap affinity benefit; measure starvation, queue age and delivered SLO. |
| Profiling/per-request metrics + latency | Instrumentation adds work/sync and can coalesce output. Separate unprofiled primary timing from transparent source/physical diagnostics. |

## 8. Prioritized work after current parity

User order retained: **Qwen0.8B/2B → matched benchmarks/long context → Qwen27B →
Gemma4 E2B/26B/31B → sparse Qwen/other models.** A large generic engine win can be
implemented immediately after parity without changing this model acceptance order.
Root coordinator must place these candidates in owned plans; this run edits none.

| Priority | Concrete work | Exit / reason |
| --- | --- | --- |
| Gate0, current | Close original0.8B natural greedy feedback/real SessionAuthority publication and2B full source parity; complete default serving route and negative completion ownership. | Five forced source-exact phases are a prerequisite, not deployed runtime qualification. Keep pinned0.31 baseline. |
| Next1 | Matched original-source unprofiled ABBA/BAAB C1/C2/C4/C8 and offered-load baseline; attribute CPU prepare, upload, launches, arithmetic, sampling, readback and delivery separately. | Bottleneck-ranked complete requests, explicit cap/profile/source identity and actual kernel/allocator data. No optimization selection by tok/s alone. |
| Next2 | Close production graph physical admission; persist row metadata and pack changed position/table data; remove proven redundant transfers/normalization passes through original lease owner. | Full logits/all-state + unknown-completion tests; repeated warm/cold complete-request gains. These benefit every model and can precede27B. |
| Next3 | Actual temporal ragged prefill/dense projection and source-correct GDN chunk kernels; source-aware mixed decode/prefill budgets. | Retain original chunk/cast semantics; all states/long-boundaries, cancellation and fair tails. Existing native per-request teacher rows are a foundation. |
| Next4 | Full-domain paged prefix manager with checkpoint intersection and COW; authorized ContextOwner/router binding and tokenizer/template reuse. | Cross-session isolation, non-block-aligned recurrent tails, grant revoke, copy retention, safe pressure/eviction and long trained parity. |
| Next5 | Complete independent long-context gates0.8B/2B under declared native/default arithmetic, then16K/32K only after prerequisites. | Preserve exact-source limits; do not replace default-bulk discrepancy with the accepted explicit reference-control result. |
| Next6 | Original27B first actual runtime using verified windows; bounded source-safe host/GPU prefetch if full residency cannot fit, then complete logits/StateRoot and long/cold/warm IO tests. | L8 is source IO only. No streamed-model claim from55GB shard scans. Profile bandwidth before expert/SSD optimization. |
| Next7 | Gemma4 E2B target math/tokenizer/masks/PLE/KV-sharing first,26B expert routing then31B as requested; native attention/backend guards at head256/512; assistant MTP later. | Real approved trained sources, own all-state schema/corpus/reference, held-out quality for derived codecs. Do not inherit Qwen results. |
| Next8 | N-gram verification, then native Qwen MTP/approved EAGLE/DFlash/Gemma assistants and adaptive depth if baseline is memory-bound and low-QPS. | Exact correction/branch rollback, distribution tests, actual accepted useful work, and stress K=0/load transitions. |
| Later | Original-source repacks, SIMD/topology, hardware-specific GEMM/PDL/FP8/FP4, derived KV/codebooks and sparse Qwen variants. | Explicit physical capability + numerical/quality profile + complete-request win. Preserve negative candidates. |
| Deferred | DeepSeek MLA/HiSparse/Engram, distributed EP/EPLB/DBO/remote P-D/encoder service, other model families and unsupported GPU backends. | Actual models/topology/transport qualification and user authorization. Renoir/gfx90c launch/recovery hold unchanged. |

## 9. Acceptance matrix for any selected mechanism

These are proposed gates, **not tests run in this research**.

- **Source/arithmetic:** freeze original model/index/shards/tokenizer/template,
  runtime/driver/compiler/device, source codec, activation/state/KV/sampler dtype,
  chunk/microbatch/backend/graph settings. Compare independently produced complete
  vocabulary logits, all required causal state domains, selected stage operands,
  generated ID/text/EOS/usage and replay. Internal CPU/B1 parity does not replace
  the pinned original oracle. Derived-quality profiles get separate calibration
  identity, held-out tasks/perplexity and quality limits.
- **Shape/causality:**0-length rejection; prompt1; page/hash/chunk/window±1;
  partial tails and exact cache hit; unequal frontiers; ragged/mixed batches;
  stop/EOS/max-output survivors; long Unicode/raw-ID histories; large vocabulary
  and offset overflow. Graph padding must neither route experts nor mutate roots.
- **Prefix/security:** source/model/template/adapter/dtype/mask mismatch; deliberate
  hash collision handling; authorized same-prefix sharing; tenant/salt/grant
  mismatch; revoke/cancel between lookup/preparation/Ready/delivery; full-domain
  checkpoint intersection and COW independence after diverging continuation.
- **Physical ownership:** refusal before allocation/upload/launch; host/GPU/IO/
  graph staging pressure; partial copy failure; encoded-but-unsubmitted/captured
  reader; event/DMA unknown completion; cancel before/after each stage; source or
  destination eviction; generation ABA; service-wide fence; trusted retirement.
  Unknown work retains every original handle/reservation; no reset/recovery is
  authorized by a timeout.
- **Scheduling:** fairness under mixed long prefill/short decode, stopped and
  backpressured rows, later arrivals, heavy cache-hit tenant, low-headroom
  recompute/offload, IO stalls and load transitions. Bounded workers/queues,
  cancellation latency and starvation limits are measured, not inferred.
- **Speculation:** all accepted prefix lengths0..K, wrong draft rejection and
  target correction, EOS inside proposal, grammar/RNG ownership, sampled
  distribution convergence, forced low acceptance, confidence errors and cases
  where draft slows service. Every target/draft state domain follows commit.
- **Performance:** equal approved source/profile/caps/tokenizer/EOS/limits and
  offered trace; isolated serialized provider ownership/cleanup; unprofiled
  ABBA/BAAB cold-process and warm runs; separate storage/cache coldness if tested.
  Include startup/compile/pack/tune, p50/p95/p99 queue/TTFT/true token ITL/E2E,
  fairness, memory peaks/allocator retention, IO bytes/amplification, power and
  actual physical dispatch/copy counts with transparent instrumentation.
- **Goodput:** count requests meeting declared correctness/quality and SLOs over
  **offered** elapsed time, and useful committed/delivered tokens separately.
  Failed, late, canceled or invalid outputs remain in offered-load accounting;
  rejected drafts and discarded Ready results are not useful output. Client
  coalescing does not provide individual-token ITL. No cache-hit rate, accepted
  draft length or raw kernel tok/s substitutes for this result.
- **Repository gate:** relevant new unit/integration/actual source/state/
  cancellation/physical tests followed by the complete approved isolation command:
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`.
  Source merge copies use fresh modification times to avoid stale Cargo reuse.
  Running it requires the implementation phase's approved inputs/hardware; none
  ran here.

## 10. License, research coverage and handoff

vLLM is Apache-2.0; inspected files carry SPDX and contributor notices. Preserve
license/NOTICE/provenance on actual code reuse, inspect each third-party subtree
and dependency license separately, and distinguish algorithm reimplementation
from copying vendor code/binaries. The prefetch file explicitly attributes
SGLang; FlashAttention, FlashInfer, CUTLASS, Triton/FLA/AITER/b12x and CUDA kernel
images have separate source/version/license dependencies. A repository-level
Apache license does not license model weights, SDK redistribution or uninspected
third-party code. [License][license]; [prefetch attribution][prefetchoffload].

Read coverage: exact repository trees for main/0.32/reference; selected current
architecture/cache/scheduler/attention/quantization/speculation/parallelism/docs;
actual scheduler and EngineCore queue sections; stable-row/staged-buffer/input/
output preparation; hybrid fixed-point/COW/Mamba manager sections; source loading,
offloader buffers, GEMM/precision selectors, fusion/graphs, GDN dispatch and
kernels, sampler/residual rejection, speculator dispatch, Gemma target/assistant,
collective/EPLB ownership and local ZG anchors above. Large files were read in
relevant sections rather than treated as fully audited. Alternative connector,
quantization and model-family implementations not specifically read are
identified from docs/registry/tree only. This is a substantive-family inventory,
not an exhaustive audit of every model/kernel/PR or a runtime compatibility
matrix. No executable upstream was downloaded or run; bounded source text was
fetched. Root README/source/shared-plan bytes were not edited.

Next coordinator action: reconcile V01–V69 with the other engine/hardware reports,
choose shared policy/mechanism primitives, then add prioritized concrete tasks to
the owned plans. Do not mark any implementation or test complete from this report.

## Pinned primary source links

[release]: https://github.com/vllm-project/vllm/releases/tag/v0.32.0
[cacheconfig]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/config/cache.py
[arch]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/arch_overview.md
[scheduler]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/core/sched/scheduler.py
[engine]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/engine/core.py
[mrv2doc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/model_runner_v2.md
[runner]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/model_runner.py
[requeststate]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/states.py
[coordinator]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/core/kv_cache_coordinator.py
[graphcode]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/cudagraph_utils.py
[kernelconfig]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/config/kernel.py
[speculator]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/spec_decode/speculator.py
[schedconfig]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/config/scheduler.py
[asyncsched]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/core/sched/async_scheduler.py
[inputbatch]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/input_batch.py
[buffers]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/buffer_utils.py
[blocktable]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/block_table.py
[asyncutils]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/async_utils.py
[sampler]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/sample/sampler.py
[grammar]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/structured_outputs.py
[cacheiface]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/kv_cache_interface.py
[cachemanager]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/core/kv_cache_manager.py
[blockpool]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/core/block_pool.py
[prefixdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/prefix_caching.md
[singlemanager]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/core/single_type_kv_cache_manager.py
[gdn]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
[kvguide]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/kv_offloading_usage.md
[cpukv]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/kv_offload/cpu/manager.py
[tiermanager]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/kv_offload/tiering/manager.py
[simpledisk]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/simple_kv_offload/disk_backend.py
[simpleworker]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/simple_kv_offload/worker.py
[hisparsedoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/hisparse.md
[hisparsetree]: https://github.com/vllm-project/vllm/tree/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/hisparse
[loader]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/model_loader/default_loader.py
[weightutils]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/model_loader/weight_utils.py
[shardedloader]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/model_loader/sharded_state_loader.py
[linear]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/layers/linear.py
[marlin]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/layers/quantization/utils/marlin_utils.py
[qwen]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/models/qwen3_5.py
[ipcloader]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/model_loader/weight_cache/ipc_loader.py
[uvaoffload]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/offloader/uva.py
[prefetchoffload]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/offloader/prefetch.py
[offloadconfig]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/config/offload.py
[cpudoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/getting_started/installation/cpu.md
[sleepdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/sleep_mode.md
[compiledoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/torch_compile.md
[attnregistry]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/attention/backends/registry.py
[flashattn]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/attention/backends/flash_attn.py
[flashinfer]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/attention/backends/flashinfer.py
[attndoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/attention_backends.md
[gemma]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/models/gemma4.py
[mla]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/layers/attention/mla_attention.py
[flashmla]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/attention/backends/mla/flashmla.py
[fimla]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/attention/backends/mla/flashinfer_mla.py
[kvquant]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/quantization/quantized_kvcache.md
[nvfp4kv]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/csrc/nvfp4_kv_cache_kernels.cu
[tqconfig]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/layers/quantization/turboquant/config.py
[tqbackend]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/attention/backends/turboquant_attn.py
[uqformat]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/attention/ops/ultraquant/format.py
[uqbackend]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/attention/backends/ultraquant_attn.py
[gdncu]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/csrc/gdn/fused_gdn_decode_kernel.cu
[gdnbackend]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/attention/backends/gdn_attn.py
[invariancedoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/batch_invariance.md
[quantdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/quantization/README.md
[fp8]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/layers/quantization/fp8.py
[b12xdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/quantization/b12x.md
[b12xmoe]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/layers/quantization/utils/b12x_moe.py
[fusiontree]: https://github.com/vllm-project/vllm/tree/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/compilation/passes/fusion
[qkfusion]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py
[moemodular]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/layers/fused_moe/modular_kernel.py
[moedoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/moe_kernel_features.md
[eplb]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/distributed/eplb/policy/default.py
[epdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/serving/expert_parallel_deployment.md
[monedoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/expert_substitution.md
[levelsdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/optimization_levels.md
[compileconfig]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/config/compilation.py
[graphdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/cuda_graphs.md
[adaptivecode]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py
[cudaplatform]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/platforms/cuda.py
[arfusion]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/compilation/passes/fusion/allreduce_rms_fusion.py
[gumbel]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/sample/gumbel.py
[reject]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
[rejectutils]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
[specconfig]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/config/speculative.py
[ngram]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/spec_decode/ngram/speculator.py
[ngramdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/speculative_decoding/n_gram.md
[suffixdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/speculative_decoding/suffix.md
[eagle]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py
[draftdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/speculative_decoding/draft_model.md
[mtp]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py
[mtpdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/speculative_decoding/mtp.md
[dflash]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
[dflash2]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py
[gemmaspec]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py
[gemmamtp]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/model_executor/models/gemma4_mtp.py
[adaptivedoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/speculative_decoding/adaptive_verification.md
[dynamicdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/speculative_decoding/dynamic_speculative_decoding.md
[flydoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/speculative_decoding/fly.md
[paralleldoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/serving/parallelism_scaling.md
[dpdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/serving/data_parallel_deployment.md
[cpdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/serving/context_parallel_deployment.md
[allreduce]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/distributed/device_communicators/custom_all_reduce.py
[dbodoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/dbo.md
[pddoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/disagg_prefill.md
[encoderdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/disagg_encoder.md
[nixldoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/nixl_kv_cache_lease.md
[pushdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/nixl_kv_push_connector.md
[nixlworker]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py
[loradoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/lora.md
[optimizationdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/configuration/optimization.md
[metricsdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/per_request_metrics.md
[metricdesign]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/design/metrics.md
[specdoc]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/docs/features/speculative_decoding/README.md
[license]: https://github.com/vllm-project/vllm/blob/8e2beaa3bff595577a069d49d96bde9074294ec2/LICENSE
