# Qwen basics, complete reference parity and advanced decoding

**16 October 2026 — implementation roadmap; new gates below are NOT RUN.**
The user explicitly orders: get the Qwen basics working, match every capability
and optimization of llama.cpp and vLLM, then enable further appropriate methods,
including MTP, speculative decoding, n-gram, DSpark2 and DFlash2. This extends
the existing serving and specialization plans; it does not replace the current
source-correct engine or restart its architecture. Parallel model source work
continues under [the model addition plan](../plans/model-execution/TODO.md).

## Ordered implementation and honest completion

1. **Qwen basics:** first finish source/configuration/tensor admission and real
   CPU execution for the exact new Qwen3.8 27B checkpoint. Keep the original
   Qwen3.5 0.8B/2B paths exact. Then qualify the new model's full resident GPU
   execution through the same mathematical schedule and backend interfaces.
   Baseline decoding, tokenizer/template/stop behavior, independent sessions,
   rollback and physical ownership must work before speculative optimizations.
2. **Reference capability and optimization parity:** enumerate and implement
   the complete union of both pinned upstream capabilities applicable to Qwen,
   including public interfaces and internal execution optimizations. Follow the
   shared layout → measured left-right → XDG acquisition sequence. An interface
   shim, parser or renamed option is not behavior or performance parity.
3. **Advanced decoding and additional optimizations:** implement and physically
   qualify applicable MTP, draft models, n-gram/suffix lookup, EAGLE variants,
   DFlash/DFlash2, DSpark and any separately source-verified DSpark2, parallel
   drafting, dynamic depth and confidence-based verification. Evaluate their
   combinations with batching, caching, quantization and placement.

Several named methods already exist in the reference tools. Their rows remain
open in the parity ledger until their implementation and advanced-decoding
gates pass; moving work to step3 never counts as completing those rows early.
"Full parity" is reserved for the complete ledger, including these overlaps.
Additional optimization ideas remain separate from verified upstream behavior.

## Pinned source baseline and coverage denominator

| Project | Source revision | Purpose |
| --- | --- | --- |
| llama.cpp | `2851b3bcbf8d993a2243ae751c0a7dfa8e944a40` | Capability/optimization baseline; same pin as acquisition research |
| vLLM | `a8be1500bad93d3c459e435bf71194aa8062e49d` | Capability/optimization baseline; same pin as acquisition research |
| DeepSpec | `005e03b81cec38b7da6399833d609ee89a2587f2` | Primary DSpark source and exact drafter compatibility research |
| DFlash | `07ebd93db9f472af339b644bb70221ad8428328a` | Primary DFlash/DFlash2 source and exact drafter compatibility research |

These are parity research pins, not replacements for the independent numerical
oracle or each model's own immutable checkpoint/reference revisions. Refresh
upstream deliberately: review a pinned delta and add new rows, keeping the old
comparison reproducible. A mutable "latest" label cannot define completion.

Historical research captures and repository-tree receipts are retired under the
[accepted checkpoint](../plans/accepted-baseline/TODO.md). The preserved documentation frontier contains 145
**unreviewed** document paths. It is a seed, not an exhaustive capability
inventory or a passing result. No checkpoint, tokenizer body or SDK was acquired.

The 105 provisional capability groups, the retained
[CLI candidates](../plans/qwen-parity/CLI_OPTION_FRONTIER.csv) and
[API candidates](../plans/qwen-parity/API_ROUTE_FRONTIER.csv) frontier lists remain open
planning inputs.
The historical audit counted16 classes,12,989 pinned paths and11,615 source
blobs awaiting semantic disposition. Its raw tree captures are retired; regenerate
the complete frontier from the explicit source revisions before closing parity.
The twelve QPA-C closure gates remain proposed, not passed. These inventories
are a broader starting point, never an exhaustive-parity claim.

The exhaustive ledger must enumerate **all** public CLI/API/configuration
options, model/task and codec registries, sampler/logits processors, attention
and state kernels, backend/build switches, loaders, cache and scheduler modes,
parallel/connector interfaces, documentation and their reference tests. Record
each source item at its exact path/symbol/revision and map aliases or equivalent
implementations to canonical capability IDs. Generated arguments and delegated
components require their own inspected source boundary. No unexplained source
frontier item may disappear from the denominator.

Each capability needs: reference project/pin, source evidence, exact behavior,
Qwen variant/task applicability, shape/codec/device restrictions, dependencies,
owner/phase, local implementation location, acceptance IDs, actual test receipts,
performance measurement and outstanding gaps. Statuses distinguish UNREVIEWED,
PLANNED, IMPLEMENTED, VERIFIED and NOT-APPLICABLE-WITH-REASON. The latter needs
an independently reviewed semantic or physical incompatibility; it cannot hide
unfinished implementation or erase the broader supported-hardware roadmap.

## Required capability inventory

The categories below are coverage obligations, not claims that both reference
tools support every item or that Zerogravity already implements them. Split
every concrete upstream mechanism into its own sourced ledger entry.

| Area | Complete coverage obligation |
| --- | --- |
| Model semantics and tasks | Dense/MoE/hybrid Qwen variants, exact layer gates/norms/RoPE, GDN/QSA/PLE where source-defined, logits, tokenization/chat templates, reasoning control, tool parsing, embeddings/scoring/pooling and released image/video/audio pathways |
| Artifacts and adapters | GGUF/split GGUF, safetensors/shards/sidecars, all registered loaders and Qwen-applicable weight codecs, source/derived identity, conversion/quantization, LoRA/adapters, draft/vision components, pinned acquisition/cache/offline behavior |
| Sampling and output | Every sampler/logits processor and composition order, RNG/seed/ties, logprobs, stop/EOS, penalties/bias, grammar/JSON/schema/regex, tools/reasoning parsing, streaming and accepted-prefix output |
| CPU execution | Threading/affinity/NUMA, ISA-selected and BLAS paths, tiling/packing, weight reuse, quantized GEMV/GEMM, prefill/decode specialization, batch shapes and memory mapping/prefetch/locking policies |
| GPU and mixed execution | CUDA/HIP/Vulkan and remaining backend goals, all attention/projection/activation/quantization kernels, source-exact fusions, flash/paged/sliding/hybrid attention, matrix instructions, graph capture/replay, device sampling, streams/events, overlap and placement/offload/splitting |
| Memory and caches | KV/state formats, paging/COW/prefix sharing, hybrid recurrent snapshots, quantized caches, context extension, sparse/index/encoder caches, cache keys/eviction, CPU/SSD offload, sleep/preload/snapshots and pinned/UMA/zero-copy transfers |
| Scheduling and concurrency | Continuous/chunked/async prefill and decode, batching/bucketing, preemption/priorities/fairness, admission, context budgets, cancellation, multi-model residency, encoder/decoder scheduling and cold/warm load behavior |
| Distributed execution | Tensor/pipeline/data/context/expert parallelism, collective/communication overlap, expert routing/load balancing, multi-device split modes, RPC/disaggregation/connectors and transfer/cache ownership |
| Serving and integration | Offline/online/batch operation, relevant OpenAI/Anthropic interfaces, model listing/routing, tools/multimodal/reasoning, metrics/tracing/profiling/health, request validation, lifecycle and documented integration behavior |
| Speculation and joint tuning | Every upstream proposer/verifier/scheduler mode, dynamic depth/trees/confidence, heterogeneous vocab rules where supported, cache/sampler/grammar interactions, draft placement and useful goodput under mixed concurrent traffic |

In particular, preserve hybrid SSM/recurrent prefix checkpoints and replay,
bounded sliding-window replay, persistent decoder/monokernel options, kernel
autotuning and compilation/AOT warm-start/cache identity as distinct sourced
optimization rows. A prepared Rust plan is not driver graph capture. The
speculative inventory also keeps LiLiCoRR and all registered rejection policies;
each requires its own target/version/quality contract before promotion.

Use the existing [joint serving acceptance plan](VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md),
[artifact pull contract](MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md),
[hybrid/drafter semantics](../docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md)
and [specialization gates](EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md).
Reuse their owners and tests; do not build a second serving engine or duplicate
completion/session/resource authority for parity.

## Basics gates

| ID | Required acceptance before baseline promotion |
| --- | --- |
| QBASE-01 | Exact checkpoint/config/tokenizer/template/reference identities; source-defined GDN and full-attention gates distinguished; no semantics inferred from family name |
| QBASE-02 | Exact tensor headers, shape/dtype/codec/shard closure and untied output binding; bounded malformed/overflow/duplicate rejection before allocation |
| QBASE-03 | Explicit source storage and physical admission for artifacts larger than the current4GiB source materialization cap; no silent cap increase or duplicate whole-model copies |
| QBASE-04 | All mathematical operators and reductions match the chosen source; grouped Q/K→V GDN geometry and intermediate rounding tested independently |
| QBASE-05 | Real trained CPU full-vocabulary/state, multi-token and nonzero replay tests against an independent executable reference; fixtures alone do not qualify inference |
| QBASE-06 | Tokenizer/chat/reasoning/stop boundaries, greedy behavior and supported sampling agree with the declared contract |
| QBASE-07 | Independent sessions/shared immutable source, reset/snapshot/stale attempt, resource refusal and rollback preserve the sole committed root |
| QBASE-08 | Full GPU model graph on each applicable initial hardware profile: actual physical source/state/readback, admission, cancellation and original-handle quarantine |
| QBASE-09 | Genuine context/corpus and shape boundaries qualified explicitly; original512 GPU capacity and existing8K divergence/16K/32K gaps remain visible |
| QBASE-10 | Default/minimal/composed features, build/lint/tests, source qualification and documentation integrity pass; unsupported combinations fail explicitly |

The [large-source basic design](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md)
proposes verified, explicitly admitted owned shards inside the existing source
owner. Its six acceptance gates are unrun. Keep the current4GiB default;
read-only/private mappings alone do not freeze externally writable files.
Route existing and new host owners through a coherent configured physical pool
before claiming global admission, and account for UMA sharing with GPU memory.
Original large BF16 payloads exceed local RAM; select a capable authorized host
or an explicitly identified and independently qualified encoded source profile.

## Speculative and advanced execution contracts

MTP is a trained source head; prompt n-grams are history-based proposals;
hashed PLE/source lookup tables are model parameters. Keep all three distinct.
EAGLE, DFlash2 and DSpark are checkpoint-specific proposal programs, not a
generic switch compatible with any Qwen checkpoint. Pin target, drafter,
tokenizer/vocabulary mapping, hidden-state taps, embeddings/head sharing,
position/mask/block limits, precision and any confidence/Markov components.

Primary DFlash source lists a Qwen3.8-27B DFlash2 checkpoint. DeepSpec documents
DSpark and DFlash; a separate **DSpark2** version has not yet been established
from primary source in this research. Keep the requested DSpark2 track open,
resolve its exact source/version and evaluate it when available; do not rename
DSpark or assume it is DFlash2. No auxiliary weights are downloaded here.

The target verifier remains authoritative. Every proposed branch owns its
private target/draft KV, GDN/QSA/index state, positions, token history,
sampler/RNG/grammar/stop state and output frontier. Publish only the accepted
legal prefix through the existing session authority. Zero acceptance, rejection,
EOS/grammar termination, cancellation and stale completion must discard the
remaining branch without corrupting the committed root. GPU cancellation keeps
all original handles/charges until actual completion or quarantine/fencing.

For greedy lossless mode require exact target-only tokens and accepted state
under the declared tie policy. For stochastic mode implement the correct
accept/reject/correction distribution, including zero-probability and numerical
boundaries, with explicit target/draft RNG ownership. Distributional correctness
does not imply identical fixed-seed draws for algorithms consuming RNG in a
different order; document/test any stronger reproducibility promise separately.
Never widen source arithmetic tolerances or use predicted confidence as token
acceptance authority. Synthetic acceptance is a labeled benchmark diagnostic,
never valid model output. Any target-only policy is explicit before execution;
a failed GPU region never silently substitutes CPU math.

All gates below are **PLANNED**, with real model/device fixtures required:

| ID | Required acceptance |
| --- | --- |
| QSPEC-01 | Native MTP tensor/head/tap/position/state and target compatibility |
| QSPEC-02 | Separate draft model plus vocabulary/embedding mapping and placement |
| QSPEC-03 | Every source n-gram/cache/map/suffix variant, repeat boundaries, bounded memory and cross-request isolation |
| QSPEC-04 | Exact EAGLE variant and hidden-state tap/vocabulary contract |
| QSPEC-05 | Exact DFlash/DFlash2 mask, block diffusion, positions/taps and target pairing |
| QSPEC-06 | DSpark Markov/confidence scheduling and separately verified DSpark2 semantics if available |
| QSPEC-07 | Accepted lengths0..K, empty proposals, all-reject/all-accept and bonus token boundaries |
| QSPEC-08 | Greedy target-only token/state equality and stochastic correction/distribution properties |
| QSPEC-09 | Hybrid recurrent/KV/index rollback across chunk/window/padding/checkpoint boundaries |
| QSPEC-10 | Grammar/tools/reasoning/stop/EOS/logprobs and streaming emit accepted output only |
| QSPEC-11 | Wrong/missing/stale drafter, source/codec/tap mismatch and unsupported vocabulary refuse safely |
| QSPEC-12 | Physical target+draft+branch memory admission and concurrent shared-source accounting |
| QSPEC-13 | Actual post-launch cancellation, stale completion, original handle quarantine and device fence |
| QSPEC-14 | Bounded adaptive depth/trees/confidence with load-aware scheduling, fairness and single-stream/saturated goodput |
| QSPEC-15 | Pairwise then material higher-order combinations with quantization, caches, graph capture, batching and placement |
| QSPEC-16 | Controlled target-only/reference/new-path benchmarks, quality and complete-request counters on all applicable initial devices/CPUs |

## Parity and performance closure

Complete parity requires zero untriaged frontier items, a disposition for every
enumerated reference capability, passing functional/source/ownership tests for
each applicable row, and a separately documented performance result for every
optimization row. A documented incompatibility remains visible; it does not
authorize claiming that missing functionality is implemented. Initial hardware
qualification covers the existing five-GPU/three-CPU fleet; unsupported or held
devices remain explicit and the wider backend roadmap remains open.

Measure identical checkpoint/derived precision, context/corpus, sampling,
concurrency and power/thermal conditions. Report TTFT, TPOT/tail latency,
prefill/decode throughput, accepted useful output, saturated goodput, peak
physical memory, source transfers, draft acceptance/reject work and fairness.
Include cold/warm cases and both baseline and combined optimizations. Record
regressions and select paths by measured workload/capability rather than enabling
every optimization unconditionally. No speedup or parity claim follows from a
lower launch count, a tiny operator benchmark or a reference project's claim.
