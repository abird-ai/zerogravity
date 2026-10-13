# zerogravity — Project-neutral experimental program, H01–H43 (v4 reviewed)

**15 Oct 2026 — SERVOPT 36 future concurrency/benchmark experiments:** [vLLM parity and combinatory source acceptance](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) adds SERVOPT-01..36: API queue and queued prefill token/byte/tenant capacity, source-correct KV/SWA/GDN/MLA hybrid prefix intersection, continuous decode + chunk prefill QoS/fairness, true physical kernel batch, target-only vs MTP/DFlash, SSD expert cache/source IO, captured graph pinning, distributed P/D state/collectives and multimodal task deadlines. Benchmark vLLM V1 and vLLM-Omni **with same exact model/quant/tokenizer/source semantics and comparable hardware/config**. Sweeps 1/2/4/8/16/32/64 requests, burst arrivals, prompt classes, 15-vs-18 grouped MoE rows, cache and cold/warm, full offered request denominator and rejected/timeouts. Optimize **valid committed SLO-goodput + fairness** not raw drafts or selected-work counts. No experiments were run under this documentation update.


**15 October 2026 — SSD STREAMING EXPERIMENTS (NOT YET RUN):** [SSD-01..32 source/physical tests](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) extend original H-series experiments: identical source tensor/StateRoot with full device residency, host RAM hot and cold SSD; real IO/H2D events and transient free, mock encoded-but-unexecuted GPU reader eviction race (DS4 #384), alternating Q2/Q4 expert sizes and prefill-cache wipe (DS4 #503), optional host hot slab vs O_DIRECT OS page-cache mode (#629), disk-only Qwen n-gram/Engram and private compatible disk KV, real OS MemAvailable/PSI/GTT plus active context, QoS P95. Benchmark **cold/warm, prefill/decode, 0/1/all resident, 1/4/8 sessions and direct-pointer versus compact-copy hit path**. Treat DS4/other prior-art published throughput as external baseline, not Zerogravity evidence; source-model and backend approval are required before disruptive 100GiB pressure tests.


**15 Oct experimental specialization extension (design, not executed):** [50 source-verified planned acceptance tests and experiment phases](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) add three complementary experiment families: **QTIP-01..14** packed derived quantization K2-K8 codecs, per-tensor shaped-noise calibration vs disjoint held-out source quality and original-vs-derived-vs-accelerated kernel error; **HYB-01..18** Qwen3.8/GLM5.3/MiMo2.6 source state, MTP/DFlash hidden-tap/partial RoPE/bidirectional/mask and rollback, prompt-specific acceptance and batch numeric near-ties; **GFX-01..18** gfx1151 Wave32/WMMA/LDS/int8/kernel tile, 15-vs-18 MoE row cliff, coherent-memory page-cache/host/driver headroom, source-valid first-image cold shape and physically canceled work. Required independent controls: same checkpoint+derived quant+tokenizer/template+source task, target-only versus draft+target, B=1/2/4/5/6/8, prompt types code/prose/chat/tools, active physical VRAM/OS MemAvailable, repeated ABBA and quality/accepted-token P95. The specialist-source author data are **external prior-art hypotheses** and its measured negative CPU MoE offload/graph/int8/lower-bit variants are retained as controls rather than automatic flags. No download, hardware pressure test or new architecture implementation in this documentation phase.


**2026-10-12 · NOT EXECUTED.** Every experiment below is a *falsifiable zerogravity hypothesis* and a means of deciding whether a capability belongs in the shared modular AI runtime. Source projects are cited separately as prior art in `../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` and historical research archives, not as components in our runtime. All performance figures discussed there are external author reports unless a real internal run exists.

## 0. Common scientific, safety and licensing protocol

Before every experiment define: exact hypothesis/mechanism, CPU/portable/reference baseline, candidate change, objective (TTFT/P95, committed outputs/time, memory/energy, quality), input/model/quant/task/position, device/topology, numerical modes and explicit failure/kill criteria.

**Comparable runs:** exact same checkpoint SHA, storage and effective compute quant mode or labeled quality delta, tokenizer/template, task inputs/context/prefill/output length/concurrency/draft acceptance, kernel/compiler/driver, GPU/NPU/CPU clock+power/firmware, physical available RAM and cache warm/cold. Warm/stabilize, use repeated ABBA/BAAB randomized blocks where relevant, report raw distributions with uncertainty and separate instrumentation overhead. An optional NPU trial must use the **same fixed fabric clock for ON and OFF arms**, with separate default-clock GPU-only result. Do not benchmark unqualified hardware that may hard hang or trigger physical OOM without consent and isolated safety measures.

**Evidence artifact:** test and source commit/diff, model/file/quant/adapter/program SHA, DeviceCapabilities, KernelPlan/CIR schema, topology/driver/firmware, task/regime, output/reference tensors and state hashes, numeric tolerances and quality, physical allocations/SSD bytes/cache hit/miss, event timeline, TTFT/P50/P95/P99/end-to-end, throughput in **committed outputs** (not proposed drafts), energy and power, repetitions and result `PASS|FAIL|INCONCLUSIVE|NEUTRAL`.

**License rule:** experiment about a prior-art algorithm may study any publicly available project and attribute it in the evidence atlas, but release Rust code must be independently written or explicitly reused from an audited MIT-licensed file after license/source-language/user policy approval. No extra model-specific engine may be embedded as fallback.

## Lane A — Kernel, compiler and execution legality

### H01 — Rust GPU architecture-specific kernel feasibility
**Compare:** scalar CPU oracle vs portable Rust GPU provider vs target-specialized Rust GEMV/GEMM and quant kernels on available real GPU. Shapes for decode1, 2–8 verification rows and large prefill; quant formats from actual source checkpoint. **Report:** generated ISA/occupancy/register/memory, numeric error, full model region speed and actual CPU overhead. **Kill:** impossible accurate ISA path or gross regression; keep generic provider.

### H02 — Compiled execution without IR hot-path tax
One model/operator graph through reference interpreter, compiled SIR→RIR→KIR/CIR bundle and hand-authored direct command control. Observe per-step host launch overhead and GPU idle; the typed compiler must not be traversed each token/frame. **Kill/pivot:** measurable unavoidable CPU/IR overhead, simplify direct bundle.

### H03 — Fusion vs occupancy and numerical legality
Benchmark norm→quant→projection, attention selector→compute, gated recurrent step, residual/hyperconnections, sampling/vocab head, decoder islands. Compare isolated micro and complete task; measure register pressure/LDS/occupancy, scratch, full-logit and state changes. Reject maximal fusion when slower or numerically unsafe.

### H04 — GPU-side routed expert row grouping
At same model/quant, vary uniform/skewed route overlap, C1/2/4/8/16/32 and decode/verify/prefill. Compare per-token expert execution vs device grouped assignments and numerically correct scatter. Capture unique expert counts and weight rereads, buffer use, P95 and compute. Reject if regroup overhead/latency dominates.

### H05 — Expert residency × batching × scratch tradeoff
Fixed resident/cold tier, CPU lane, SSD fetch, dynamic LFU/LRU/hotlist and scratch borrowing under one physical budget; vary cache size, route distribution and batch quantum. Measure *critical-path* expert misses and physical headroom rather than hit% alone. Revert any eviction or COW hazard.

### H06 — Huge immutable lookup tables with bounded IO
Random/scattered/repeated row reads from a very large table without demanding entire file in RAM; compare mmap/pread/async as supported, checksum/row hash and cold/warm SSD, dedup and GPU packing. Fuzz truncation/corruption; no silent fallback row zero or blocked GPU without error.

### H07 — Full hybrid-prefix cache sharing
40K repeated text/model prefix with 3+ state domains (KV, recurrence, indexer, n-gram, optional grammar/modality), 8 branches and modified tool/media spans. Compare full recompute vs verified immutable state snapshots vs COW/restore tiers. Same committed frontier/state/logits; quantify TTFT, cache bytes, cross-plan conversion and eviction/rebuild.

### H08 — Adaptive target speculation
No speculation vs trained multi-token/auxiliary draft, independent small draft and prompt-derived lookup at varied acceptance, context and concurrency. Measure *committed* tokens/total time, draft+verifier+rollback resource/energy, stochastic exactness and degradation on unpredictable output; disable when net worse.

### H09 — Mixed prefill/decode service fairness
C1/2/4/8/16/32 users with bursts, 256→32K input and output lengths; compare FIFO, token budget, chunking/priority and stage-aware group choices. Record TTFT/p95 inter-token/throughput/starvation with exact state.

### H10 — Long-context dense/sparse/latent/compressed attention
0/4K/8K/16K/32K/64K/128K/256K where physically possible, selection masks/ties, compression boundaries, chunk continuation. Compare GPU selector vs portable lowering and full official logits/long-retrieval quality. Do not reduce trained selection count to obtain speed.

### H11 — Physical unified/discrete memory and safe admission
Probe actual device RAM/VRAM/GTT/address space, OS/cgroup/caches/CMA, pinned staging and model/context estimates; simulate first then supervised real loads. No host OOM or kernel parameter changes. Compare CPU+GPU bandwidth contention against distinct PCIe dGPU tiers.

### H12 — Quant storage vs compute/derived model quality
Same source checkpoint with lossless packed format, lower-bit derived variant and temporary compute encodings; record per-tensor numerical errors, KL/NLL, greedy continuations, official task quality, memory and performance. Do not silently choose lower quality. Grouped expert/always-used router/mHC quality separately.

### H13 — Heterogeneous topology, activations and peer transport
Compare single device, layer/tensor/expert partition, fast/small GPU+large/slow GPU or CPU and optional remote links. Model exact mathematical expert selection. Measure activations/partial result traffic, collective cost, residency, communication and fault-recovery. A capacity improvement isn't necessarily a fair raw throughput comparison.

### H14 — Real multi-turn agent/task workload
Replay long common-prefix tool/reasoning/JSON sessions and optional images under model-native rendering, cache reuse, cancellation and concurrent requests; observe end-to-end task time, output quality and storage privacy. Don't publish private prompt content.

## Lane B — Compiler/state/architecture extension

### H15 — Multiple Rust kernel providers through one interface
Same semantic region and actual reference math compiled by distinct CPU/GPU codegen routes where available. Compare executable native instructions, semantic portability, compilation cost, ABI, safety and output. No model math/device coupling.

### H16 — Effect-token correctness under arbitrary graph rewriting
Generate legal and illegal operator reorderings with full KV, gated recurrent, sparse indexer and RNG/grammar. SIR verifier rejects effect conflict and preserves full state/logits. Test a zero accepted-prefix speculative branch and rollback after async IO failure.

### H17 — CIR async execution, liveness and deliberate failures
Fake multiple devices+IO with Acquire/Import/Pack/Dispatch/Wait/Join/Commit, random out-of-order completions, stale generations, early release, cancel and torn SSD reads; valid plans safe, invalid rejected. Physical overlap measured only after simulation.

### H18 — Actual eager vs device capture candidate choice
Identical source kernel variants under eager launch and graph replay with modeled pointer/dynamic residency; micro and full request may prefer different. Persist proven tactic keyed by source/driver/shape/mode. No captured stale allocation.

### H19 — Bounded pure-region optimizer/search
One pure fused region with no rewrite, finite hand heuristics and bounded equal-value rewrite search. Compare compilation RAM/time and full model performance, generated code size and legality. Turn off complex search when tiny or noisy gain.

### H20 — Common resumable prefix frontier for 3+ state groups
Frozen KV/recurrent/latent/indexer/media state at incompatible block finalization boundaries, model+quant/adapter/template+tenant hashes; select last common legal frontier and recompute suffix. Verify exact same-plan outputs and state; record cache cost.

### H21 — Modularity extension drill
Add a novel stateful operator, an unfamiliar device target and a new task program without editing unrelated HTTP, global allocator or model registries. Count changed crates/files/lines, compile time and all quality/fallback tests. **FAIL** if model-specific runtime fork needed.

### H22 — Product/task correctness and malformed inputs
ModelStore manifest/GC/download atomicity, GGUF dialect, tokenizer pre-type, grammar rollback, LoRA/adapter cache keys, embeddings/rerank pooling and image/audio spans. Fuzz malicious inputs, output streaming and privacy; task outputs only when supported.

### H23 — Accurate external benchmark comparison provenance
Separate same-model/checkpoint/quant/hardware/task from unmatched published source reports; record exact prior-art source URL and license in external research atlas and no false category/model matching. Do not use performance marketing thresholds without quality.

### H24 — Actual hardware safety profiles
Discover available device topology and driver instead of old assumed equipment; distinguish 64/92/96/128/192 GiB physical fit from virtual GPU GTT and latency. Optional multi-accelerator SoC requires firmware/clock/power risk gating and isolated tests. No automatic privilege escalation.

### H25 — Full model compiler runtime overhead
CPU reference/diagnostic IR interpreter vs precompiled direct bundle vs literal static runner across model prefill, decode and repeated task stages; measure hot-host dispatch, code cache, physical memory and full-model throughput with matched source. Must keep optimized model execution directly compiled.

## Lane C — Heterogeneous projection compute and additional task programs

### H26 — Intra-operator output row partition
A pure matrix projection partitioned by rows across synthetic independent compute providers, complete/ordered row-map join and optional first-use consumed spans; same tensor/state as unsplit oracle, including quant/bias/tails. No device/model name in operator math.

### H27 — Optional NPU-owned program/driver feasibility
Investigate Rust-authored accelerator program compilation/packaging and legal driver memory/callback API for a source-proven block-float tile shape. A foreign-language device kernel wrapped in Rust does **not** satisfy stated source-language policy. Report BLOCKED if no compatible Rust AIE/codegen exists; GPU-only remains normal.

### H28 — Cross-API DMA, machine-safety and power/fabric isolation
Fake mapped pinned buffers imported between GPU/NPU APIs, resource leases/event bridges/queue depth, deadline/quarantine, cancellations and stale completions. Real concurrent SoC GPU+NPU requires qualified firmware/driver/clock/power, dedicated recoverable test host and long soak; no automatic root clock/suspend changes.

### H29 — Same-source cooperative prefill benchmark
On physically qualified NPU+iGPU, compare selected independent projection row-slice variants vs GPU-only with identical model/checkpoint+quant, shapes, fabric clock/power/temperature and cache, including idle energy and p95. Separate clock-only effects; an upstream reported ~7% result is a hypothesis, not ours. Test long-context, C1/C4 and decode fallback.

### H30 — Block-float numerical and target task quality
Check packed exponent+mantissa bytes and reference dequant, full logits/recurrent/sparse state across partial GPU/NPU joins, strict numeric mode refusal and target sampled distribution where appropriate. Matching a small greedy-output set is not quality proof.

### H31 — Non-token TaskProgram pipeline reuse
One real or synthetic ASR/audio stream, embedding/pool, image-denoise iteration or TTS vocoder stage using the same SIR/RIR/KIR/CIR/ResourceArbiter and TaskCoordinator but **not** next-token-only state. Validate task-specific committed frontier, output schema, cancellation and device fallback. Separate model and codec release qualification.

## Promotion rules and evidence storage

- **PASS:** mathematically/legal/quality/resource-safe, and measured win or required correctness/usefulness with documented cost.
- **NEUTRAL:** correct but performance within noise; keep simpler default and explicit opt-in candidate.
- **FAIL:** changed model semantics/state, unsafe resource, unapproved numerical degradation, user-visible tail latency regression or source/licensing violation.
- **INCONCLUSIVE:** unsupported device, missing model checkpoint, insufficient repetitions or uncontrolled power/cache/hardware; no automatic promotion.
- **BLOCKED:** impossible required source-language, unavailable accelerator/driver or unqualified hardware safety; portable paths continue.

Future benchmark tree may include `bench/{inputs,configs,references,results,profiles,reports}`, with hash/provenance, redacted artifacts and machine metadata. Do not commit private prompts, licensed checkpoint weights, source-derived foreign kernels or credentials. For each completed experiment, update phase status, candidate registry, source prior-art catalog as needed and full agent handoff.

**Architectural success:** useful source-independent mechanisms can be tested and selectively plugged in; losing specialization does not leave model users without a runtime.

## v4 extension experiments — 2026-10-12, H32–H43

These tests are **NOT RUN** and extend existing H01–H31 without modifying their identifiers. They correspond to review findings R4-01…R4-24 and are written by independent mechanisms, not named external project internals. Use source artifact/quant/device/OS/license evidence; some tests need physical devices, others synthetic proof.

### H32 — Prepared StateRoot and atomic logical publication

Synthetic KV/recurrent/indexer/RNG plus a failed second `StateDomain::prepare`, out-of-order GPU events, cancellation and accepted speculative prefix length 0..k. Baseline old logical root read unchanged until all domain versions ready; exactly one root transition and stream output after publish. **Reject** domain-level public `commit`, false O(1) GPU recovery or partial output.

### H33 — GPU-native multi-step input progression

Compare per-token CPU router/sampler readback against prepared device-resident row map/stop/sampler and two or more device-native dependent provisional steps on a supported model/grammar mode. Measure host sync, GPU idle, P95, exact target sampling and GDN/indexer state. **Reject** fast-path claims when mandatory CPU synchronization still blocks every token; fall back honestly for unsupported modes.

### H34 — Hierarchical reserve and device-local allocation

Global ResourceArbiter reservation plus per-device asynchronous slab allocation, bounded 32-session concurrent loads, generation/epoch/lease correctness and delayed free. Compare single-global-mutex-per-operation baseline. Inject OS/cgroup pressure, driver buffer allocation failures and live pinned NPU buffers. **Reject** missing reservations, ABA and falsely claimed no-OOM guarantees.

### H35 — Dynamic shape polymorphism with guarded fallback

Rank-fixed bounded symbolic dimensions, ragged row offsets, sparse selected rows and prefill/decode/verify/encoder shape modes. Run 1,7,128,4096 rows with divisibility-required kernel and generic plan; invalid shape never reaches kernel. Measure code cache size and cold start. **Reject** unbounded variant explosion, unsafe padding or per-token compilation.

### H36 — Minimal TaskDriver versus generic workflow graph

Implement trusted text TaskDriver and mock image denoiser/encoder TaskDriver sharing semantic/kernel/resource/worker core. Compare CPU dispatch/alloc/complexity against a generic interpreted workflow DAG, count unrelated files touched. **Reject** a new model server or untrusted `ExternalAction` in core.

### H37 — Fused whole-region candidate with unchanged runtime

Register hand-optimized whole-layer/whole-model region as KIR candidate whose state/effects match generic SIR and whose driver handles are owned by DeviceWorker. Toggle candidate off and prove exact model/task continuation/fallback without new scheduler, sampler, service or allocator. Record full-task throughput and quality, not just tile GEMM.

### H38 — Output stream commit/reconnect semantics

TaskSession commits 1..N output chunks, client disconnects before/after acknowledgment, optional cursor replay and sequence dedup under output backpressure. Emit after PublishRoot; no uncommitted token/PCM/image chunk. Do **not** claim exactly-once network receipt without a durable ACK protocol.

### H39 — Distributed epoch fencing and partial failure

Two synthetic remote ranks prepare same model/adapter/state root, one ACK delayed/lost, coordinator restarts and stale attempt arrives. Ensure only fenced current epoch publishes state, output cannot be partially committed; document volatile vs durable recovery limits. This is P19 work, not Stage0 blocking GPU-only.

### H40 — Future parameter update compatibility (without training platform)

Synthetic `TrainStep` and GradientProgram update a new ParameterVersion plus optimizer state while pre-existing inference sessions stay pinned to immutable old model/adapter/compiled bundle. Compile reference forward/backward for one tiny op. **No** full autograd engine, training infra or public training promise required. Reject if SIR/TaskDriver/resource design must be rewritten.

### H41 — Key/cache migration and shape-band pruning

Upgrade driver/kernel compiler/semantic op schema and adapter generation. Verify separately versioned ArtifactKey, SemanticKey, StateCompatKey, BundleCacheKey and RuntimeBinding; invalidated code is rebuilt and old state safely converted/recomputed, no persisted raw pointers. Test bounded LRU and cold portable fallback.

### H42 — Fair mixed-task scheduling under resource/latency targets

Text prefill/decode, embedding encoder, synthetic denoiser and audio frame tasks share GPU/CPU/SSD; compare fixed batch vs dynamic target-time/fairness/pressure admission with same math and output. P50/P95/P99, throughput, memory, fairness, starvation and actual kernel overlap. Revert policy if no statistically significant whole-task benefit.

### H43 — v4 minimality and extension change-radius audit

Across H32–H42 count crates, direct implementation lines, new traits, circular deps, build time, hot-path allocation/CPU sync, actual codegen and model coverage. Add novel model op, new hardware provider and non-token task; prohibit duplicated model server/allocator/session owner. Compare one library+binary (or 2–4 crates) against over-modular alternative. **Reject** architectural complexity without needed correctness/performance benefit.

**Evidence:** all H32–H43 reports must list actual hardware/source SHA/model+quant/precision/state+shape+task, commands and failures. A mock does not qualify a physical NPU/GPU; a better microbenchmark does not qualify whole model performance, and a fully correct but slower provider can remain `FUNCTIONAL` without default selection.


## v4.1 cross-accelerator composability experiments H44–H55 (NOT RUN)

These experiments are **independent falsifiable source/hardware and fake-backend proof tasks**, not claims of working TPU/wafer/Graph ASIC support. Sources and product/protocol specifics are cited in `../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md`. Use one `RegionExecutionProvider` seam, exact model/task semantics and existing TaskCoordinator/ResourceArbiter; no separate model server or runtime.

### H44 — Coherent local ARM+GPU system versus HBM device admission

Simulate and (only if hardware accessible) physically probe a coherent local CPU+GPU ARM host with LPDDR/shared bandwidth and compare against discrete/HBM accelerator. Distinguish real OS/cgroup headroom, GPU addressability, zero-copy permissions and bandwidth. Cross-check source model/quant and native CUDA portable fallback. **Reject** single “Blackwell memory” capacity/throughput heuristic.

### H45 — Native block-scaled precision and multi-GPU topology

Reference-verify FP4 formats with different per-block FP8 vs power-of-two scaling and tensor-scale headers; separate source checkpoint identity, device packing and lossy arithmetic. Graph capture on a native GPU and 2-rank mock with peer/collective cost and missing rank failure. Optional physical discrete/hardware qualification, matched quality/P95/energy. **Reject** automatic lossy quant or falsely unified HBM.

### H46 — Fake graph-only accelerator (TPU-class execution contract)

Compile one **closed full model stage** through a simulated ahead-of-time graph compiler with static/guarded dynamic shapes and no exposed kernel-by-kernel launch. Return one program/event with prepared StateRoot and direct TaskDriver output; shape-miss fallback and cache fingerprint. **Reject** host micro-op interpreter or new service engine. Actual TPU XLA/PJRT/other compiler needs separate SDK+hardware approval.

### H47 — Fake wafer-scale programmed dataflow and streamed model weights

Simulate static tile/core placement and on-chip SRAM pipeline with immutable layer weight stream, compile/load/warm/weight IO critical path and stateless region output. Compare fake whole-stage stream to host issuing every GEMM. Correct semantic output with no new model code; stateful graph only if StateBridge passes. **No physical WSE claim without SDK/hardware.**

### H48 — Statically scheduled inter-chip dataflow and heterogeneous stages

Fake deterministic compiler-owned schedule across processing elements/on-chip SRAM with fixed link bandwidth and externally synchronized first-use barrier to CPU/GPU stage. Verify region cut, memory pressure, collective/RNG effect and no host per-op dispatch. **Reject** device pretending to support arbitrary dynamic op/branch when it does not.

### H49 — AOT accelerator backend with expensive graph compile

Simulate cloud ASIC graph compiler: bounded `CompileBudget`, static shape bands, optional custom low-level kernels, program upload, model/quant/source/revision pin, and runtime memory. Compare warm/cold TTFT and shape specialization explosion; ensure background compilation does not block first portable response. **Reject** claims that “model recognized” means `COMPILES`.

### H50 — Additional graph/low-level accelerator device classes

Add mock matrix/SRAM accelerator and low-level programmable dataflow device through same `RegionExecutionProvider` for one reference encoder+LLM subgraph; represent device ISA/firmware, graph and source constraints and optional Ethernet interconnect. Test provider qualification without copying foreign source/C++ compiler, and fallback. Actual physical target testing only after access.

### H51 — Opaque stateful graph refusal and exact snapshot bridge

One fake compiler graph with private provisional KV+GDN+RNG state, snapshot/restore/recompute and true event Ready/Failed; another with only unrecoverable opaque in-place model state. First can satisfy PreparedRoot, second **must not** be eligible for general stateful session/speculation. Stateless encoder can still work. Check accepted prefix, cancellation and late event.

### H52 — Region granularity and program residency optimizer

Same semantic model/tensor graph compiled into native small kernels vs captured GPU graph vs whole AOT stage vs static program with build/load/compilation amortization and external weight stream. Compare first output, P95, resident bytes, topology IO and full task; choose the lowest measured critical-path plan without inventing universal winner. Bound number of cached variants.

### H53 — Remote/cluster topology and epoch fencing

Fake rank/wafer/pod fabric, link congestion, collective groups, model/quant/state schema mismatch, lost ACK and coordinator epoch restart. No publication of partial rank state, no assumption of one HBM pointer across rack, no false exactly-once remote delivery. Remote creds/privacy/egress are admission restrictions.

### H54 — External model API versus native program boundary

Attempt to register a text-only remote inference API as native RegionExecutionProvider: it must be rejected for lacking lowerable model op/control and state compatibility. It may become explicit authorized `TaskServiceAdapter` with separate user privacy, task/stream model and cost, but must not be described as running zerogravity semantic kernels on remote hardware.

### H55 — Backend extensibility, licensing and alternate speculative algorithm

Add a **fake graph-only provider and fake static fabric provider** alongside a native CPU/GPU provider; verify no changes in ModelDefinition/TaskDriver, StateRoot owner, service/ResourceArbiter and no new durable IR. Source-language/driver/device compiler status must be truthful. As a distinct optional experiment, adaptive multihead parallel draft decoding (the Cerberus *algorithm*, not wafer-scale hardware) can register a DraftSource and run target distribution/cost tests without changing base model semantics or decoding engine.

**Promotion ladder:** architectural mock `ARCHITECTURE_COMPATIBLE` is not real SDK readiness, compilation, state safety or physical performance. Exact artifact/model/quant, driver, program compiler/SRAM/cache, shape and sample/quality, physical memory, measured p95 and power must be recorded for any genuine hardware claim. All H44–H55 remain NOT_STARTED until actual runs.

## v4.2 new independent distributed execution experiments H56–H70 (NOT RUN)

All experiments use one model/TaskDriver/StateRoot plus independent ComputePlacement, ParameterPlacement, StatePlacement, CachePlacement and TransferPlan, with authenticated peers and measured network. Model and topology names are descriptive capabilities, not imported source engines. No physical machines/tunnels may be accessed without explicit user authorization. Distinguish **simulated** from real source/checkpoint/physical measurements.

### H56 — Standalone zero-network fast path

Verify no peer discovery, tunnel handshake or global cluster allocation in standalone CPU/GPU work; compare StepProgram host/physical memory overhead before/after optional distribution module. Fail if cluster code is mandatory for one model.

### H57 — Two-node layer pipeline with layer-local state

Synthetic then actual two worker groups, disjoint layer ranges and matching per-layer KV/GDN/state, activation boundary transfer, same full logits/task output as single-node source oracle. Measure C1 sequential decode and chunked prefill overlap, extra RTT and capacity gain.

### H58 — Remote immutable prefix cache versus recompute

Remote prefix directory may lie/be stale, full-state source/quant/adapter/tenant hash validated, measure state hydrate bytes/time/p95 across direct and relayed links, compare local recompute and previous committed root when missing.

### H59 — Separate prefill and decode machines

Reserve compatible decode destination and transfer **entire required hybrid state** (KV+recurrent/indexer/ngram/grammar/modal), with chunk checksum/Ready/fail. Compare prefill/ITL tail isolation vs large network TTFT overhead under same task, not “always more throughput”.

### H60 — Remote attention beside KV versus remote KV read

GPU A does projections/FFN; node B owns KV and attention core. Compare per-layer compact query+output to raw KV bytes per-step and cache migration once, on simulated slow LAN/tunnel and real physically qualified links. Reject chatty plan with high p95.

### H61 — Sequence-token KV attention partition algebra

Shard causal KV token intervals among nodes; compute local max/Z/weighted numerator and global stable softmax result. Validate all-mask, head/GQA, position, numerical drift and state update; reject unknown sparse/compressed/global topk/GDN without separate algebra.

### H62 — Attention/FFN and routed expert owners

Source model with per-layer attention+KV state one node and FFN/MoE experts on another, GPU-side route stable, activations/partial returns, varying route overlap and hybrid attention type. Measure p95 and stage imbalance, reject slow-link negative.

### H63 — Tensor/head parallel and fast fabric collectives

Simulate and later physically qualify collective across 2+ GPU nodes, exact shape/model/quant, graph/driver/rank and epoch order, multi-hop bandwidth, lane congestion and wrong rank failure. Do not assume RDMA across ordinary encrypted tunnels.

### H64 — Task-stage distribution beyond text

One synthetic audio encoder/ASR and denoiser/vocoder TaskDriver across peer region stages with task-specific output/committed frontier, source/tenant permission and transport cost; no fake next-token stage or second model server.

### H65 — Encrypted tunnels, direct versus peer/relay topology

Transport emulator then approved real tunnel tests for direct, peer-relay, high RTT, loss/MTU/crypto CPU load and link failover. Compare bounded authenticated TCP/QUIC implementations, stream priority/credits and physical copy paths; VPN not GPUDirect.

### H66 — Trusted node enrollment and adversarial peers

Unknown peer/forged capability, revoked cert, wrong tenant/model license, malicious oversized frame, duplicated command, remote arbitrary shell attempt and untrusted device program all rejected before memory allocation/source export. Log redaction and scoped RDMA leases.

### H67 — Coordinator/network partition, epochs and state shards

Old coordinator lease expires, worker holds Prepared candidate, lost CommitNotice/ACK, late peer rejoin and provisional DMA. Exactly one session writer or fail-stop, old root intact/compatible replay; don't promise network exactly-once or durable failover without separate implementation.

### H68 — State shard migration and topology replan

Topology link degrades/disappears during active work, new plan proposed but not applied until committed TaskFrontier. Reserve new peer, transfer/verify state, fence old generation, PublishRoot ownership epoch and retire; inject failure before and after publish, no wrong output.

### H69 — Performance/capacity Pareto across multi-machine placements

Physical source checkpoint/quant/driver and measured link compare colocated if fits, contiguous layer split, remote prefix, attention/FFN, sequence shard and expert/tensor sharding at C1/C8 and 4K/32K/128K. Report capacity-only wins separate from raw tokens/s, TTFT/P95, energy and WAN crypto/egress.

### H70 — Future-topology composability change-radius audit

Add a new trusted accelerator node + TransportProvider + cache/state placement mode to **same** SIR/TaskDriver/RegionExecutionProvider/CIR/StateRoot/ResourceArbiter with no duplicate model service/allocator, no new durable IR, and zero-change standalone startup. Record changed modules, codegen/compile/run overhead and safety checks.

**Promotion levels:** model-architecture simulators cannot prove actual tunnel bandwidth; link-tested transfer isn't state-safe execution; mathematically correct model run isn't performance-qualified until physical p95/energy/batch tests. New remote worker profile remains opt-in and source/tenant-restricted until qualified. Normative docs: `../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md` and `../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md`.

## v4.3 multi-model local runtime, router, config and OpenAI API research H71–H86 (NOT RUN)

These are **unexecuted** independent tests for zerogravity's own built-in model router and user-facing experience. One TaskCoordinator/ResourceArbiter/RegionExecutionProvider, one TaskDriver per session and versioned exact model identities across `run`, SDK and `serve`; no external inference engine or mandatory gateway. Named CLI/API source comparisons are research-only in `../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md`.

### H71 — Zero-config one-command local model run

Clean supported machine and source-verified local GGUF/model sidecars, no config/account/cloud/daemon; `run <path>` uses CPU then actual eligible GPU, correct prompt rendering/logits/output state/usage with concise terminal. Compare CLI direct vs in-process library to detect hidden per-token proxy cost. Refuse unsupported model/quant/physical capacity honestly.

### H72 — ModelCatalog, immutable artifact and alias revision

Two different checkpoint SHA/quant/Tokenizer/Adapter artifacts, two profile aliases and one alias hot update during active stream; verify old TaskSession stays bound to original source/numerics/StateRoot, new request uses new revision, model deletion/GC preserves referenced blobs. No silent target change.

### H73 — Concurrent model load deduplication and cold-start fairness

100 simultaneous requests for one cold model produce **one verified load+compile/warm reservation**, bounded waiter queue/deadlines and correct first valid output. Meanwhile another warm model continues low-P95 serving. Simulate corrupt artifact, driver compile failure, cancel and partial load rollback; no 100x physical allocations.

### H74 — Many-model simultaneous physical residency

1/2/4/8 different registered models and 0–N physically resident models under finite HBM/UMA/RAM and per-session context pools; actual budget counted (weights+code+draft/adapter+KV+scratch+OS). Measure cache hit, warm time, TTFT, safe overcapacity queue/unload and no OOM or hidden lower-quant swap.

### H75 — Idle eviction, keepalive and pinned session safety

Warm/idle/pinned/scale-to-zero model with ongoing TaskSessions, speculative provisional state, device in-flight events and remote data; evict only after leases/quiescence and compatible session checkpoint retention. Alternate giant/small models to stress cache thrash; explicit pin can be refused if physical reserve cannot be granted.

### H76 — Equivalent replica balancing versus different-model choice

Two replicas exact same model/quant/adapter and a third different model; sticky long-session prefix StateRoot, request queue/P95 and device headroom. Compare round-robin to cache-aware/least predicted completion. Exact alias never chooses third model; user-configured policy alias may select it and reveals actual source.

### H77 — Sharded deployment versus independent source model replicas

One source model split across two trusted peers, two full replicas same artifact on different peers and two different models on same peer. Validate DeploymentId/ReplicaGroupId/ResourceArbiter accounting and remote StateShard owner affinity; no treating two shards as two independent load-balanced models.

### H78 — Model profile precedence and provenance

Built-in safe device defaults, global config, model/profile, alias/deployment, CLI and per-request sampler; test conflicting context/quant/adapter, invalid keys, hard model/tenant/physical limits, env precedence and ability to print `config show --effective` with value/source/guard. No silent immutable model mutation or ignored quality-affecting fields.

### H79 — OpenAI Chat Completions exact SDK conformance

Official pinned Python/JS client uses local base URL, `GET /v1/models`, chat nonstream/stream with correct ChatChunk SSE, model identifier, usage and stop/length/tool call data/finish. Compare raw generated sequence to direct `run` under same checkpoint/sampler and root. Different model roles rendered by exact verified templates; malformed/unsupported params get typed errors.

### H80 — Responses subset, stateful retention and event correctness

Stateless `/v1/responses` request and distinct typed `response.*` streaming events, correct created/item/delta/final output and failure. `store`, `previous_response_id`, conversation state and cloud built-ins refused when not implemented; when enabled later, authenticated StateRoot/retention/cursor exactly scoped. No hidden permanent history or tool execution.

### H81 — Embeddings and extra TaskDriver endpoint qualification

Actual source-verified encoder + pooling/dimensions/normalization and model alias/tenant, input batching, usage and correct SDK response; compare independent reference outputs. Rerank/audio/image endpoints unsupported unless their own TaskDriver/codec/model quality passes. Never fake vector embeddings from arbitrary text completion head.

### H82 — Local listener versus public API security and native admin

Loopback-only default/no network if `run`, remote bind denied unless explicitly configured auth/TLS/tenant policy; authorized inference cannot invoke model import/remove/unload/admin; unauthorized model names/paths/peer details hidden; reject malicious JSON, media URL SSRF, untrusted model code and auto remote pull. No dummy-key auth on public network.

### H83 — Multi-model parent TaskProgram roles and resource admission

One explicit target+draft and one encoder+reranker pipeline with independently named model artifacts and separate memory/compiled-plan identities; one parent task/session budget and one TaskCoordinator committed output. No replica-confusion, incompatible source artifacts rejected, semantic sampled verifier model intact.

### H84 — Warm/cold replica placement and dynamic per-model batching

C1/C2/C8/C16 active sessions across tiny model, giant MoE and encoder while a cold new model loads; measure TTFT/P50/P95/P99, real throughput, queue fairness, GPU device idle, resource headroom, native compatible row batching and per-model cold model load/evict policy. Kill global per-token routing mutex or starvation.

### H85 — Model/router hot reload, client disconnect and server restart

Change alias/source/profile/driver while streaming, unload/stop after cancellation and in-flight DMA, async SSE consumer disconnect, restart with immutable source catalog but **without** serialized raw GPU pointers. If safe SessionState persistence configured, validate full state restoration; else report session expiration rather than replay made-up token history.

### H86 — End-to-end minimality, parity and observability

One model `run` in process, `serve` same engine over loopback and Rust SDK TaskRequest produce same source-correct output and honest ModeReceipt under matched sampling. Measure per-token CPU/HTTP/router overhead and startup/config complexity. Add a second model/replica/embedding alias **without changing** SemanticProgram, KernelProvider, TaskCoordinator, DeviceBackend or independent serving engine; compare touched files/crates and CI. A straightforward user never requires choosing GPU CUDA tiling, NPU fabric mode or remote placement before running a supported model.

**Qualification:** a mocked catalog/router passes architecture-only; real CPU/GPU source model and official tested SDK versions required for functional/API marks; accurate P95/energy under same checkpoint/hardware for performance-qualified. All experiments H71–H86 are PLANNED, **none run**. New optional Stage S0-SERVE-A…J tests the small prerequisites.

