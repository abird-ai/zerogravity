# Design walkthroughs — concrete execution and fallback traces

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


These are **target behavioral examples**, not implemented traces. They force the system design to account for real effects across model, compiler, device, memory and serving subsystems.

## Example 1 — Dense Llama-like model on unsupported GPU specialization

1. User chooses known GGUF with valid tokenizer/config and model SHA.
2. Loader creates artifact/tensor map; parser checks dimensions, quant block, RoPE/heads.
3. Model recipe yields Semantic IR: embed, norm, QKV, causal GQA, residual, FFN, LM head.
4. Capability probe sees generic Vulkan backend but no architecture-specific quant MMQ.
5. Compiler selects portable attention and tested generic quant GEMV/GEMM; plan report names unsupported fast kernels.
6. ResourceArbiter computes dense weights+KV+scratch/OS headroom and reserves memory; load can refuse if model doesn't fit rather than silently changing quant.
7. Runtime prefills prompt in bounded chunks, builds KV state; decode uses direct compiled launches, no IR interpretation per token.
8. Target logits sampled with configured policy; frontier committed; HTTP stream emits.
9. Midstream cancellation invalidates provisional step and waits for events before buffer release.
10. Benchmark outcome measured versus llama.cpp on same GGUF; general correctness can be achieved before parity.

## Example 2 — Qwen3.8 Flash-Next on 128 GB gfx1151

**Input:** Qwen3.8 checkpoint with known qwen4exp recipe, supported quant, large n-gram table stored on local SSD, optional MTP; no automatic quant downgrade.

**Model load:** verify auxiliary n-gram token/row map, train-compatible MTP weights, 48 layers with GDN/QSA and 512 top-10 experts where config says so; allocate dense/expert and recurrent/attention/indexer pools using OS-safe global headroom. For model whose main weights and expert set fit, choose resident fast path. Separate SSD row reader for huge lookup table; a 95 GiB file does not become a 95 GiB resident mapping.

**Compile:** match fused norm/quant/projection/GDN and sparse attention regions; choose gfx1151 wave32 quantized grouped kernels for prefill and fast decode; verify MTP state variant; fallback to CPU reference on unsupported convolution op only if conversion/synchronization legal and benchmark shown acceptable.

**Prefill:** chunk prompt, hash token n-grams and asynchronously read needed embedding rows; device processes GDN, QSA and MoE using large-row specialized kernels; update provisional then committed sequence state; reuse previous immutable prefixes if compatible. Large prefill scratch leased and returned after GPU completion.

**Decode:** single-row quant MMVQ, GDN step, indexer and QSA selection execute GPU native without CPU route checks; trained MTP may propose 1–3 rows; target verifies, commits accepted states and drops others. If low acceptance, controller backs off without changing target sampling distribution.

**Multiple clients:** 4 unrelated requests may route into overlapping experts; route results grouped device-side; execution plan chooses grouped MoE where measured beneficial. Each GDN/indexer history remains per session. Memory manager reserves all live per-session states before admission.

**Metrics:** actual external same-checkpoint/quant comparison with Gufo and a closed tuned reference where legally runnable, fallback plan correctness; be explicit if that reference uses a different quant.

## Example 3 — GLM-5.3 Flash on 92/96 GB Strix Halo

**Artifact:** 321B-class GLM using a tested Q2 recipe, quant weight manifest. Model's 42 routed layers×288 experts cannot all necessarily fit beside dense/KDA/sparse MLA context and OS budget on smaller RAM.

**Resource plan:** commit dense, shared and recurrent/KDA/MLA structures; compute physical RAM reserve, prefill scratch and per-session frontier. Only remaining budget assigned to GPU resident expert slots. Missing expert blobs reside in file/mapped source; staging is bounded. Do not trust large HIP-reported GTT as free physical RAM.

**Decode layer:** device router selects top-8, consults table of `(layer, expert)->(slot,generation)`. Resident experts run immediately. On miss, service event requests exact blob, async read and safe slot replacement after readers complete; GPU waits at legal dependency, not arbitrary kernel synchronization. If transfer is expensive and CPU lane has demonstrated advantage, device/CPU work overlapped on discrete hardware, not assumed helpful on UMA.

**Context:** KDA fine-grain state and NoPE latent sparse MLA require exact position state. If a fallback attention kernel expects expanded KV, compiler performs explicit latent→expanded conversion at a legal boundary; otherwise reject candidate before launch.

**Perf:** compare cold and warm cache, stream and resident capacity, read miss stats and output correctness; Maya/DwarfStar quant recipes must be aligned or labeled incomparable.

## Example 4 — Shared-prefix coding agents with tool calls

- Eight clients submit 40K identical instructions/repository context, then diverge.
- Prefix cache key includes model SHA/quant/tokenizer/rendered history/pos policy and all modality fingerprints; load an immutable snapshot containing KV, GDN/indexer state and token-history.
- Each agent forks from immutable state, private mutable tail, independent sampler RNG.
- Tool JSON output and hidden reasoning retained in canonical token history; cannot retokenize changed formatting and claim exact prefix hit.
- One client changes images: fingerprint differs and affected multimodal spans/prefix state invalidate, while unrelated immutable prefix can be reused if valid.
- Another client cancels after one emitted token; no speculative uncommitted state survives. Unused leased scratch returned.
- Measure actual task latency/TTFT versus replaying 40K tokens independently.

## Example 5 — Training-free novel model op arrives

- New safetensors config declares operator `FooRecurrence` absent from registry.
- Loader recognizes artifact but returns `UnsupportedOperator(FooRecurrence)`, does not pretend ordinary attention works.
- Contributor adds a Rust reference state transition, tests and typed op schema; CPU model now functionally runs.
- Compiler generic GPU lowering added for supported backends; remaining known ops use existing extreme kernels.
- Later `gfx1151` fused kernel added; model semantics and server do not change.
- Offline quality/benchmarks validate and promote performance-qualified hardware cell only when evidence exists.

## Example 6 — Distributed TP failure during speculative verification

1. Two peers agree model/quant, stage graph and token frontier p.
2. Draft proposes 5 tokens, target verification begins over distributed expert/attention partials.
3. Worker disconnects after sending only some partial outputs.
4. Collective fails timeout/epoch check; no rank publishes frontier p+5 or emits proposals as accepted.
5. Quarantine incomplete buffers, close route; rebuild both peers from a common durable committed prefix p on a fresh handshake.
6. TCP/RDMA remoting must be explicitly trusted/authenticated per policy; no automatic cleartext public binding.

## Example 7 — Why compiler fusion might lose

A fused KDA + MoE kernel reduces launch count by 20 but spills registers, reduces occupancy and requires large shared scratch. Its per-operator microbench looks faster, but complete multi-client HTTP test reports worse P95 and no tokens/s gain. Retain the simpler plan as default, document rejection and leave candidate behind a lab build if useful for future GPUs. Numeric exactness is a separate gate from speed.

## Practical invariant crosswalk

Example 1 tests general fallback; example 2 tests model specialization and sampler rollback; example 3 tests physical memory and expert leases; example 4 tests full hybrid-prefix state; example 5 tests versioned semantic extensibility; example 6 tests distributed atomic frontier; example 7 tests proof-based plan selection.

Implementers should turn these scenarios into acceptance/integration fixtures as phases become active, preserving raw logs and actual performance evidence.

## 2026-10-10 integrated v2 addendum — async CIR and modern extension traces

**Trace 8: compilation and no hot-path IR**
1. Known model/quant tokenizer imported and validated into ModelRecipe.
2. SIR declares pure tensor ops + state-effect edges.
3. RIR optimizes legal subgraphs and reference fallback; KIR picks Rust-target CUDA/HIP provider candidates by shape/dtype and numeric contracts.
4. CIR assigns device streams, copy and IO timepoints, buffer lease generations, state-publish barriers, scratch lifetime.
5. Immutable executable bundle validated/cached. EngineCore schedules requests using StepPlan identifiers; GPU launches direct kernels/graph without walking SIR every token.
6. New driver invalidates code cache, **not** semantic model or trusted tokenizer.

**Trace 9: generalizable hybrid prefix intersection**
- Prefix state groups: full KV, sliding window KV, GDN recurrent matrix, compressed sparse QSA indexer, ngram token history and grammar RNG.
- Cache returns 50K token hash match, but compressed rows finalize only at a 2K-aligned frontier and one domain has only 44K valid state.
- Coordinator chooses a common legal frontier (at/before minimum valid constraints), restores compatible domains and recomputes remaining 6K in proper token order; no incorrect "50K hit" reported.
- COW fork to 8 agents remains immutable; adapter/image fingerprint invalidates incompatible branch state.
- Read source: `STATE_CACHE_AND_TRANSACTIONS.md`, `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`.

**Trace 10: new hardware with no model edits**
- A new NVIDIA target supports a custom Rust CUDA tile-kernel matmul but has no specialized KDA.
- DeviceProvider reports exact tile support; KIR compiles selected GEMM/projection regions, generic CPU/GPU KDA reference region at compatible conversion; CIR adds copy cost and context/phase plan.
- If KDA is too slow but correct, capability shown `functional, not qualified`. If no safe state conversion/kernel, reject selected backend for that model, then consider CPU whole plan.
- No changes needed to model recipe, tokenizer, tools or HTTP.

**Trace 11: async expert miss and cancellation**
- GPU route selects missing expert; CIR emits miss request with `model/layer/id/generation`; host reads exact immutable blob to pinned buffer under budget.
- User cancels while SSD read in flight; cancellation epoch invalidates dependent provisional step, but staging/slot leases remain until file/device completion. Completion with stale epoch is discarded, not published into a reused expert slot.
- EngineCore emits no further uncommitted output. Other unrelated sessions use hot expert cache, unaffected.
- A later request can resume from last committed full hybrid snapshot or from checked canonical token replay.

**Trace 12: an apparently fast kernel rejected**
- Grouped MoE kernel shows +12% isolated shape speed but full end-to-end at concurrency 8 worsens TTFT p95 by 18% due resident expert thrash and snapshot scratch pressure. Artifact/tactic remains in tuning lab but default KernelPlan retains simpler kernel; mode receipt names reason. No weakened numerical contract or silent precision change.

These traces are designed assertions, not evidence that zerogravity is already implemented; convert each into mock+real integration tests at the appropriate phase gate.


## 2026-10-12 trace 13 — Qwen cooperative prefill on an optionally qualified XDNA2 NPU

1. Device probe sees Strix Halo Radeon + XDNA2, XRT driver and exact matched firmware/power/FCLK qualification. If absent/unsafe, choose GPU-only plan and never wake NPU or modify host kernel settings.
2. ModelRecipe declares regular GDN z, QSA query/gate and shared FFN projections. RIR recognizes independent GEMM output row spans and the applicable 4096×2560×1024 BF8 program, preserving all source Q8_0 weight/quant metadata and numerical policy.
3. ResourceArbiter reserves pinned A/B/C and immutable NPU prepacked weight slices, instruction/queue and GPU state; CIR builds GPU pack→NPU enqueue versus GPU remaining projection work, with first-consumer external wait. Buffer generations must remain valid across HIP/XRT.
4. GPU continues GDN/attention/route work that does not depend on the NPU rows. NPU output arrives in mapped BO, GPU decodes/consumes in correct original row order, and all GDN/QSA/KV/indexer/sampler effects commit at a single causal frontier.
5. If unsupported chunk/numeric profile or NPU unqualified: immediate GPU-only kernel path, no extra model runtime. If XRT fails mid-compute: abort provisional token state and quarantine BO until truly quiescent; never fill missing rows with zeros or emit an invalid token.
6. Qualified benchmark captures source UD-IQ4_XS with matched held clocks and 16K prompt, actual first-use overlap, logit quality, P95/energy and safe device soak; author 1698→1821 tok/s remains research reference, not our result.

**Trace 14 — new media task:** An ASR model uses audio chunk preprocessing + encoder + model-native transcript generation; a TTS model uses acoustic stage/vocoder; an image model uses repeated denoise steps. Each TaskProgram runs through shared SIR/RIR/KIR/CIR and DeviceWorker but has independent stage frontier/output type, and optional NPU GEMM only after device and quality qualification. This prevents a media model from being forced into a fake text `decode_next_token` loop.

## v4 traces — prepared state, shape fallback and independent tasks

**Trace 15 — atomic all-domain state:** commit root R7 for KV+GDN+RNG. GPU prepares KV R8 and GDN R8; RNG Prepare fails or device event still pending. TaskCoordinator retains R7, emits no new output and quarantines pending buffers until quiescence. Recompute/RNG repair yields all prepared R8; coordinator swaps one root to R8 and then emits `OutputEvent(session,epoch,seq)`. Neither per-domain state nor physical event individually became public.

**Trace 16 — device-native StepProgram:** a text TaskDriver selects a legal compiled decode/verify island, binds persistent batch slot and current root epoch. GPU routes MoE and samples/accepts two provisional next tokens without host synchronizing IDs; event signals completed candidate and TaskCoordinator validates/publishes. If grammar needs host iteration or shape guard fails, choose correct smaller generic host-loop without claiming peak speed.

**Trace 17 — dynamic quant/shape and alternate task:** an ASR encoder stage with ragged audio frames chooses a guarded GPU kernel if `ShapeEnv` satisfies alignment, else CPU/GPU generic; it returns transcript chunks at audio time frontier, not fake target token budget. A future `TrainStep` produces new ParameterVersion without mutating existing inference session R7. Same resource/worker executor, no new model-specific serving engine.

## v4.2 distributed and tunneled execution traces

**Trace 18 — capacity-first layer split:** Node A owns TaskDriver and model layer group 0–k with its KV; authenticated Node B prewarms k+1–n weights and holds B-layer recurrent/KV. A sends only hidden activation at split boundary for each token. B returns next activation plus Prepared shard version; coordinator publishes one StateRoot referring to both after Ready. On B disconnect, stop/fence and recover from last canonical full-state checkpoint or fail clearly, with no wrong token output.

**Trace 19 — remote cache vs compute:** A compute GPU needs long-context attention state on B. Planner compares full compatible KV+GDN hydrate once to A, remote attention query execution on B returning small output, and naive per-step bulk KV remote read. On high-RTT tunneled link, last option fails cost; A can use its own local state/recompute or capacity-first layer placement. Immutable prefix cache hint stale yields verification miss, not wrong root.

**Trace 20 — attention token-range sharding:** three nodes own disjoint causal KV token ranges, compute local softmax max/denominator/weighted numerator and reduce stably to correct attention output, with source semantic positions/masks. A compressed/sparse/recurrent operator without registered PartitionAlgebra refuses same plan. All shard candidate versions/Ready must be prepared before TaskCoordinator root publication.

**Trace 21 — tunnel relay and failure:** enrolled peer path changes from direct to relayed while a layer group executes; current PlacementEpoch remains valid until safe barrier or attempt canceled. A new topology measurement proposes colocated or other peer at committed frontier; resource transfer/fence verifies model/tenant/state. No automatic arbitrary remote shell, no assumed RDMA over tunnel, no split-brain writer or phantom exactly-once network delivery.

## v4.3 multi-model router and one-command user traces

**Trace 22 — direct zero-config run:** user issues `zerogravity run ./local.gguf`. One engine verifies source model/tokenizer/quant, qualifies CPU/GPU, creates RegisteredModel and single ResidentModel lease, executes normal TaskCoordinator/TaskDriver and emits committed token stream; no HTTP service, model download or proxy per token. End closes/retains per configured in-process session and quiescent device leases.

**Trace 23 — many model aliases, limited resident capacity:** API `model:"coding"` resolves AliasRevision 7 to immutable source SHA/quant and finds warm GPU Replica 1. Simultaneously `model:"embedding"` requests an encoder load; ResidencyManager reserves physical RAM/VRAM and graph context, queues bounded if needed, without blocking warm coding output. A third huge model may force safe idle eviction or typed OutOfPhysicalBudget, never silent substitute/quant downgrade or active KV loss.

**Trace 24 — sticky replica, alias hot update:** two identical model replicas B and C each execute concurrent user sessions; a long-prefix session bound to B remains there for cache affinity. Admin changes alias `coding` to revision 8 targeting a different model; active revision 7 retains original tokenizer/KV/StateRoot and correct streamed `model` ID, new request resolves revision 8. Replica C may take an equivalent new session but cannot steal B's state without snapshot/epoch fencing.

**Trace 25 — OpenAI compatibility and native admin:** one `zerogravity serve` accepts local SDK Chat Completions SSE for `model:"coding"`, returns exact qualified model+usage/finish; embeddings endpoint only for actual encoder; Responses stateful/tool/cloud-only field unsupported => typed error. Native `models stop` requires admin capability and drains active model before physical unload. No model output causes server to invoke arbitrary tool, download model, publish private source paths or enroll remote peers.
