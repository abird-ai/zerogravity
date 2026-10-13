# vLLM Concurrency Prior Art and Zerogravity Architecture Reconciliation

**Researched 15 October 2026.** Primary references are vLLM and vLLM-Omni's published design, scheduler configuration and API documentation; this file is source research, **not** a Zerogravity benchmark. Models/configurations in upstream products evolve and exact baseline must be pinned at implementation.

## 1. Verdict

vLLM's concurrency performance comes from its whole serving system, not just PagedAttention. The relevant baseline includes:
- unified step scheduling across prompt, cached prefix and speculative work,
- continuous batching and decode-prioritized chunked prefill,
- controlled active sequences, queued requests and queued prefill tokens,
- physical KV block allocation, refcounted prefix cache and hybrid KV cache groups,
- asynchronous worker execution and bounded device snapshots/graphs,
- speculation, expert and parallel model-aware scheduling,
- disaggregated prefill/decode for independently tuning TTFT and ITL,
- vLLM-Omni multimodal/flow/AV task stages and async serving.

Zerogravity's existing architecture already accounts for these conceptually. **It does not yet implement equivalent high-concurrency production serving.** Its potential advantage is to co-optimize *compatible* kernel, quant, model state, routed expert cache, speculative row shape, GPU/RAM/SSD tier and hardware placement under one source-verified cost model, while retaining a simple resident/no-draft baseline.

## 2. vLLM V1 unified scheduling — real source contract

Official scheduler source/API: https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/

Current V1 schedule implementation states that there is no intrinsic “prefill phase” or “decode phase” inside core scheduler. Each request has computed tokens and the total needed tokens including speculative proposals. A bounded new-token budget advances computed work, subject to cache/residency and stage constraints. This naturally expresses chunked prefill and target speculation.

**What Zerogravity should preserve:**
- same source token positions and committed StateRoot, including per-model decoder and tokenizer semantics;
- track needed/computed/reused/accepted/draft work independently;
- allow compatible requests to enter a batch at safe per-token/stage frontier without waiting for other requests to finish;
- soft decode-first QoS while protecting long prefill from starvation, with source-legal chunk boundaries and valid per-chunk state;
- account required cache pages/physical worker occupancy before finalizing a batch;
- permit media stages only with their own atomic units and rational clocks.

Do not implement a second prefill-specific GPU inference engine or per-token global router. One coherent StepPlan controls existing prepared device region execution.

## 3. Three separate overload valves: current vLLM configuration

Official config: https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/

Current documented controls include:
- max_num_queued_reqs: global API-server bounded total queued/running requests, rejects excess with HTTP 503.
- max_num_queued_tokens: total tokens in requests still prefilling (coarse upper estimate across DP workers), guards TTFT against huge backlog.
- max_num_seqs: active sequences per serving rank/engine scheduling.
- max_num_batched_tokens: actual target step work/token budget, including relevant encoder/prefill/verification planning constraints.
- optional priority/FCFS, long-prompt thresholds and policies.

This is stronger than a single queue-length cap. Zerogravity should additionally bind memory, tenant fairness, typed encoder/audio-video work and external provider jobs without duplicate allocation. A queue estimate may intentionally overestimate prefill cost when cache reuse is not yet known, which is the safe direction for QoS; document observed estimate vs actual backlog.

**Negative test:** server with 20 waiting 200K prompts and only 2 active KV sessions must reject/defer long work before OOM, not wait until physical GPU occupancy fails.

## 4. Hybrid KV and source-domain cache compatibility

vLLM public design: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
Source API: https://docs.vllm.ai/en/latest/api/vllm/v1/core/kv_cache_coordinator/
Prefix cache design: https://docs.vllm.ai/en/latest/design/prefix_caching/

The hybrid design supports cache classes with different logical lifetimes and prefix replay rules: full attention, sliding-window and Mamba-style recurrent state. Unlike standard full-attention prefix caching, an SWA prefix may be valid when only the trailing source window is retained, while recurrent state may require boundary snapshots. Source document revisions differ on the number/group compositions supported; don't assert fixed two-group limitation across future/current releases without pinning code.

**Zerogravity design extension:** a generic bounded source-domain frontier meet across full KV, SWA, GDN, KDA, MLA/DSA/QSA indexers, mHC, draft and source-dependent media features, with exact processor, model/quant, numeric, content and tenant access keys. Where one required domain lacks a valid frontier, reuse the earlier complete joint root or recompute/restore; don't report a fake prefix hit based only on KV bytes.

Important: generic multi-state-domain support as a mathematical contract does NOT mean generic interchangeability of unrelated model state. Each StateDomain must provide its own source proof, restoration shape, causal dependency and trusted physical conversion receipt.

## 5. Chunked prefill and real-time tail protection

vLLM guidance: https://docs.vllm.ai/en/stable/configuration/optimization/

Decode work is usually prioritized for ITL. Remaining token budget is filled with prefill work, chunking long requests. Larger max_num_batched_tokens can help TTFT/batch throughput while smaller favors shorter decode inter-token delays; optimal depends on hardware/model and load.

Zerogravity must account for **three QoS constraints together**:
1. max queue wait and TTFT for newly admitted long prompts, including aging/protected minimum prefill service;
2. inter-token latency and per-tenant fairness for already decoding text;
3. soft realtime TTFA/playback packet deadlines and long-running solver-step yields for audio/video.

A low-latency speech stream cannot be protected simply by labeling all prefills lower-priority forever; admission and legal step boundaries must be measured. A P/D split may isolate tail but carries additional state transfer and memory.

## 6. Disaggregated prefill/decode

vLLM source: https://docs.vllm.ai/en/latest/features/disagg_prefill/

Published docs describe P/D disaggregation as experimental, mainly to independently tune TTFT/ITL and improve tail ITL control; explicitly note it does **not intrinsically improve throughput**. State transfer connectors include NIXL, Mooncake, an independent KV-cache layer and others, with source/lifecycle limitations.

Zerogravity must retain an equally first-class optional plan without duplicating inference engine:
- prefill and decode owners can run on different devices/peers;
- transfer ALL required hybrid state domains, not just full KV, with exact source model/numeric/epoch/tenant receipt;
- IO, peer transfer and actual device completion require real event/lease readiness before new StateRoot;
- compare T_prefill + T_network/state transfer + T_decode vs collocated mixed scheduling under tail latency and memory headroom;
- wrong model/quant/processors, interrupted transfer and incompatible state layouts must safely refuse/recompute, not silently read stale state.

Transport topology/permissions and authentic remote events remain core physical placement contracts, not API-layer tricks.

## 7. vLLM-Omni source capabilities and fair comparison

Official overview: https://docs.vllm.ai/projects/vllm-omni/en/latest/
Serving endpoints: https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/
Diffusion batching: https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching/
Video streaming input: https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/video_stream_api/

vLLM-Omni has native support for many text/image/audio/video/action model families, typed stage orchestration, async streaming, realtime and duplex variants. Its published model-serving docs describe **one loaded model per server instance** in primary API deployment examples, with per-model stage orchestration. However, do not extrapolate to a claim that a deployment of multiple instances cannot route across models, or that no plugin/multiple-model pattern exists elsewhere.

Diffusion step batching supports request-batching and iterative solver-step batching where exact pipeline source permits it. Different source image/video models may not be safely batched together, and a simultaneous TTS pipeline is not a native joint audio/video generator.

Zerogravity target differentiation:
- one physical device serving several independently registered models under shared source weight memory and global admission, without one inference engine/OS process per model;
- source-bound joint optimization across model, quantization, expert routing, speculation, storage tier and ISA;
- source-correct native media/decision TaskDrivers coordinated by latency and resource planner.

**Still future, not implemented.** Run original models and tasks under comparable source input, numerical/solver, codec and output finality requirements before promising any advantage.

## 8. Metrics and honest goodput

vLLM metrics: https://docs.vllm.ai/en/latest/design/metrics/

Core events distinguish queued, first scheduled, preempted and new output tokens. User-visible timings include TTFT, queue, prefill, ITL and end-to-end latency. Zerogravity should expose these plus true physical queued/issued/completed GPU and source state commit, rejected offers, tenant fairness, cache state/bytes, draft accepted vs proposed, energy, cold/warm, physical OS+GPU residency and audio/video clocks.

**Service goodput** only counts valid committed result under declared latency objectives, but its reporting denominator and service availability must include ALL offered requests. A strategy that rejects 80% of long prompts may deliver low tail latency to the remaining users, but is not automatically a better system. Include offered input rate, admitted/completed/rejected/expired share, p50/p95/p99 time, per-tenant fairness and source fidelity.

Quantized pack quality vs original FP8/BF16, GPU candidate vs same pack, and target+MTP vs target-only must be independently tested. At most claim “better in workload X at hardware/driver/config Y under comparable quality” after meaningful repeated trials.

## 9. Specific research and software risks

- Many vLLM performance optimizations rely on validated CUDA/HIP kernel assumptions; a Strix Halo shared UMA deployment has different bandwidth, OS page-cache and quant GEMV tradeoffs.
- “More draft tokens” can increase MoE unique experts and padded target verify rows, making output slower despite high proposal acceptance (one measured prior-art example).
- A longer prefill chunk can improve throughput and hurt a live user's ITL/TTFA; P/D disagg or adaptive batching is beneficial only after observed tail/goodput results.
- A copied prefix from a mismatched recurrent KDA/GDN schema can yield plausible outputs while making neural state incorrect.
- Physical GPU slot reused after encoding but before actual completion can corrupt tensor outputs even if every hardware command reports Ready (DS4 #384); resource/cache plan must cover entire pending graph.
- Diverse model assemblies compete for resident weights, draft cache, SSD and OS headroom. “Supports N model aliases” is not proof N models fit or run concurrently with deadlines.
- A latency-optimizing policy can starve long prefills or reject too much of load. Explicit fairness/admission governance is required.
- Any runtime optimization should retain a simple source-correct single-request/no-draft/reference plan and record negative speed experiments.

## 10. Phase owner map and evidence boundary

P09: API queue and bounded model/task/tenant admission; P10: hybrid state and prefix; P12: batched compatible source kernels; P15: MTP/DFlash; P16: finite multi-dimensional candidate scheduling; P20: offline tuner; P22: pinned vLLM/V1 and Omni benchmark. v5.1 adds timed media and different QoS task dimensions; distributed plan adds source-compatible P/D state transfer.

Read normative joint scheduling contract and SERVOPT-01..36. **None of these new tests or measured comparison results exists solely by authoring the design.**

Research reviewed against official upstream docs on 2026-10-15; URLs and source versions can change and must be reverified before implementation.
