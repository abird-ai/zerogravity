# Zerogravity — vLLM Parity and Combinatory Serving Acceptance Program

**15 October 2026 — DESIGN and planned future tests only, not executed.**
**Normative owner:** .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md.
**Goal:** implement the already designed P09/P10/P12/P15/P16/P20/P22 concurrent serving contracts with measured source-verified performance and the ability to independently reproduce gains or regressions against a pinned vLLM V1/vLLM-Omni baseline.

## 0. Status and boundaries

Zerogravity does NOT yet have a production OpenAI-compatible high-concurrency server or full-source GPU Qwen3.5 model engine. Shared immutable model source, CPU text execution, bounded fair physical-worker proofs, source-free quality and selected real CubeCL physical regions are narrower. **All SERVOPT tests below are PLANNED and must not be marked passing by documentation.**

No second serving engine, model-family GPU server, speculative sampling authority, vLLM Python runtime dependency or global resource allocator is permitted. At every phase the source mathematical reference, task-specific StateRoot and physical event/BufferLease must remain authoritative.

## 1. Development phases and dependency gates

| Phase | Artifact and integration seam | Definition of done |
|---|---|---|
| P09a request handling | ModelRouter, versioned protocol capabilities, tenant/request queue limits | bounded admission rejects overload before heavy state allocation; no false OpenAI compatibility |
| P09b per-request lifecycle | Requested→Queued→Admitted→Prepared→Issued→Completed, epoch cancellation, committed stream | typed failure/retry and no stale output after cancel/disconnect |
| P10a hybrid state page manager | exact full KV/SWA/GDN/KDA/MLA/indexer/grammar state and COW, per-domain prefix meet | independent reference replay and source-checked prefix state; tenant isolation |
| P10b physical/SSD residency | one global byte authority and real device+IO leases | no memory double-count, encoded pointer pin, source-exact cache restore |
| P12 compatible stage/route batches | true device source-compatible rows and grouped expert kernels | real physical concurrent native batched operations, exact quant/numeric policy |
| P15 target speculation | versioned MTP/DFlash/lookup and original target correction | deterministic and stochastic target-only equivalence, complete state rollback |
| P16 bounded joint planner | batch, prefill, draft, expert cache, kernel and placement candidates | best *measured* legal finite option under QoS/fairness, no online exhaustive search |
| P20 profiling/autotuning | offline measured per model+device+ISA+shape+driver candidate cache, cold/warm | no unbounded per-request kernel solver, persisted negative candidates |
| P22 benchmark/quality | pinned optimized vLLM/V1 + vLLM-Omni when comparable | apples-to-apples numerical quality, SLO goodput and tail latency under load |
| v5.1 serving + distribution | timed PCM/video/decisions, optional P/D and peer collectives | stage-typed QoS, safe event/clock semantics and actual transfer receipts |

Each phase remains subordinate to the existing original Qwen3.5→Gemma→Qwen3.8 practical source/hardware order. These are precise acceptance criteria, not authorization to implement huge model variants prematurely.

## 2. Quality and benchmark protocol

### Benchmark populations

1. **Text solo:** short/long prompt, short/long generation, B=1; warm and cold model/source/OS file cache, no draft.
2. **Text concurrency:** B={1,2,4,8,16,32,64} active sessions; source tokenizer/template same, stable source model; hit/miss KV pages and shared prefix.
3. **Bursty arrivals:** deterministic arrival trace with varied prompt length and staggered update/tool-like turns; offered QPS sweeps past knee until rejection/deadline miss, not just throughput under perfect saturation.
4. **Prompt mixes:** code/reasoning, prose/dialogue, retrieval/very long prefills, tools/JSON schema; actual expert routing entropy varies.
5. **MoE/low-bit:** Qwen3.8/GLM/MiMo target and source-derived quant (only when licensed/qualified), multiple draft depths and expert-row tile groups.
6. **Memory pressure:** fully resident vs legal GPU/host/SSD streaming, hot and cold experts, prefix restore from RAM/disk, long contexts; no disruptive 100GB stress on a shared interactive device without explicit permission.
7. **Multimodal:** mixed text, streaming ASR/TTS/duplex, image/video solver stages and finite decisions where actually implemented, per-class valid output + p95 deadline.
8. **Distributed P/D:** collocated mixed chunked prefill/decode vs P/D split across same-size devices and heterogeneous peers, actually measured transport/StateRoot transfer.
9. **Negative controls:** target-only, no prefix reuse, no batching, no draft, safe slow quant kernel, single-model resident CPU/GPU; retain negative/neutral evidence.

### Apples-to-apples source setup

Pin exact artifact revision and license, model/derived quant and numeric/quality contract, tokenizer chat template, rope/cache/draft state, prompt/input distribution and session scheduling, device ISA/compiler/driver, OS/cgroup/NUMA/SSD, power/thermal and model warmness. Compare similar vLLM/Omni capability/version with flags and connector/adapter differences recorded; if exact source model/format is unrecognized by competitor, label comparison non-equivalent, don't substitute another checkpoint or silently lower quality.

Repeat interleaved ABBA or BAAB runs after warmup. Report warm/cold and per-request raw events or reproducible aggregate with confidence bounds; do not claim a win if uncertainty overlaps noise or feature definitions differ. External vLLM metrics are baseline methodology, not Zerogravity results.

### Goodput and service-level availability

Text SLO-goodput uses only **committed correct output tokens** satisfying per-request TTFT and ITL bounds, divided by offered wall time, with **all offered attempts** accounted including rejected, timed out and canceled. Distinguish denominator choices and report attempted/accepted/rejected/expired/complete populations; a smaller served population is not evidence of higher service goodput.

For voice/AV/decisions use separately normalized, complete and valid task outputs inside their own deadlines, not a weighted sum of incompatible raw units. Report service-level fairness (per tenant latency/acceptance), model/capability correctness, tail under load, physical GPU/OS memory and energy. A throughput win that breaks source semantics, increases unexplained numerical error or starves other tenants is not qualified.

## 3. Source-free and actual-model acceptance: SERVOPT-01..36

| ID | Required scenario | Source-correct assertion / gate |
|---|---|---|
| SERVOPT-01 | Burst > global API queue cap | typed overload/retry; no large KV allocation, unbounded retained input or stalled server |
| SERVOPT-02 | Tenant A floods while tenant B has small interactive requests | quotas/fairness; bounded B tail and no data/cache leakage |
| SERVOPT-03 | Queued prompt tokens huge but only 2 active sessions | separate queued prefill/byte cap refuses before memory overcommit |
| SERVOPT-04 | API accepts request without physical GPU capacity | separate queued vs admitted state and honest latency/NoCapacity, not phantom Ready |
| SERVOPT-05 | 64 queued + 8 active sessions and scratch overbooking attempt | independent active sequence, row/token and physical byte admission caps |
| SERVOPT-06 | Canceled waiting request | no GPU fence; queue credits and input bytes released |
| SERVOPT-07 | Canceled submitted GPU task while other sessions active | affected physical service fenced only when unknown; other session state/Root not rewritten |
| SERVOPT-08 | Long prefill competing with 8 short decodes | legal prefill chunking protects decode ITL while prefills make eventual progress |
| SERVOPT-09 | 100 short decodes continuously arrive while long prefill waiting | aging/protected prefill progress, no permanent starvation |
| SERVOPT-10 | Source Qwen 0.8B/2B prefill B=1 and B=4 chunk mixes | exact original logits/StateRoot at same numeric mode and correct chunk boundaries |
| SERVOPT-11 | One mixed text-image input containing unsplittable image unit | no invalid half-image encoder work from token budget |
| SERVOPT-12 | Full KV + sliding KV group cache overlap | longest source-valid joint prefix frontier, not overclaimed prefix hit |
| SERVOPT-13 | GDN/KDA recurrent state absent at apparently cached KV prefix | lower reuse/recompute, no fabricated committed state |
| SERVOPT-14 | Three or more heterogeneous state groups, differing snapshot positions | generic bounded compatible frontier meet; no hardcoded only two groups |
| SERVOPT-15 | Prefix same tokens but different model quant/template/LoRA/media preprocess | cache miss or verified conversion, never wrong model state |
| SERVOPT-16 | Cross-tenant cache key hash same, ACL differs | no unauthorized state/cached private media leakage |
| SERVOPT-17 | Forked concurrent sessions reuse prefix and mutate one branch | COW exact isolation and real device page lease across queued work |
| SERVOPT-18 | State on RAM/SSD; compare restore vs recompute | measured cost-informed legal choice, same source outputs |
| SERVOPT-19 | MTP/DFlash/lookup no draft vs 0..8 proposals | target-only deterministic and stochastic sampler/stop/grammar/StateRoot parity |
| SERVOPT-20 | Prose low draft acceptance vs coding high acceptance | bounded candidate safely shrinks/disables draft when net slower, same target outputs |
| SERVOPT-21 | MoE R=15 vs R=18 rows across group-size16 candidate | cost accounts extra padded tile and expert union, no universal hardcoded 16 |
| SERVOPT-22 | One large batch + many model families on one card | physically compatible grouping only, mixed models interleave without fake GEMM merge |
| SERVOPT-23 | Different quant pack sizes QTIP K2 vs K3 with different kernel costs | no storage-bytes-only optimizer; source-derived quality separate |
| SERVOPT-24 | Fully resident model one token after warm compile | no global catalog/JSON/OS IO/RPC/readback synchronization per token |
| SERVOPT-25 | Overlap CPU tokenizer/H2D and GPU kernels under legal dependencies | real event-backed stage overlap measurable, no locks held across awaits |
| SERVOPT-26 | Captured GPU batch refers to expert slot before actual submission | slot cannot be reclaimed/changed until queued+physical reader events retire |
| SERVOPT-27 | Slow client streams output while GPU continues | bounded transport credits/backpressure, no unlimited GPU lease retention |
| SERVOPT-28 | Disaggregated prefill → decode on same source model | transfer full hybrid StateDomains with authenticated version/epoch and real Ready |
| SERVOPT-29 | Lost remote P/D state transfer while worker may run | no stale root/partial publication, correct recovery/fence and tenant guard |
| SERVOPT-30 | P/D disaggregated with WAN or slow KV transfer | benchmark can choose collocation; no unconditional throughput gain claim |
| SERVOPT-31 | 5 video jobs + 10 duplex audio sessions + text decode | per-class QoS/refusal and safe solver/frame yield; no audio starvation or fake service SLO |
| SERVOPT-32 | Worker selected vs kernel issued vs kernel actually completed | separate truthful counters, output commit only after actual event |
| SERVOPT-33 | P20 optimizer sees unknown new image shape/driver | bounded generic qualified fallback, no unbounded JIT/solver tuning in hot request |
| SERVOPT-34 | Plan candidate switch at hardware row/cache knee | repeated measured source-correct gain, bounded hysteresis and negative candidate retained |
| SERVOPT-35 | vLLM V1 text concurrency suite with same model/quant/device/runtime knobs | report acceptance, TTFT, ITL, SLO goodput, memory, error and confidence; no unverifiable win |
| SERVOPT-36 | vLLM-Omni comparable audio/video/decision tasks | typed-valid outputs/deadlines and truthfully stated unsupported modes; compare only same task/model quality |

All SERVOPT-xx cases are **future gates**, not implementation claims. Source-free scheduler unit tests cannot certify trained model/QoS; actual original checkpoint/source numerical tests must accompany GPU and benchmark evidence before qualifying a service.

## 4. Additional protocol, memory and failure variants

Each source-free state test above expands into:
- normal arrival, priority/tenant mix, duplicate req ID, cancel before/after submit, stale epoch and reconnect;
- CPU reference vs fake device and actual physical backend where authorized;
- independent KV/recurrent/encoder state subdomains, corrupted snapshot, wrong model/quant hash, near-tie numerical differences and required replay;
- prompt lengths 0/32/1K/8K/16K/32K/128K under source-qualified context limits, not assuming unsupported contexts passed;
- limited GPU memory + coherent UMA physical cache + SSD random expert miss + no-stale GPU slot, actual peak host/driver bytes;
- varied MTP acceptance 0..d and B1/2/4/5/6/8/16 with expert route histograms and optional draft assets, target-corrected source output;
- source license/privacy/tenant and provider-incompatible client events, exact API fidelity receipt;
- slow client and high GPU queue depth while physical kernel returns late Ready or unknown completion;
- regression to normal CPU Qwen3.5 Phase1 and prior 8K GDN numerical investigation remains independent.

## 5. vLLM competitive gate details

### Text
- vLLM V1 (not historic V0) scheduler config including max_num_seqs, max_num_batched_tokens, max queued reqs, chunked prefill, prefix cache and exact draft strategy where supported.
- Run same prompt sequence and concurrent arrival trace with pinned original/different derived quant (not mix); for unsupported exact model in one runtime, report as gap/capability comparison rather than apples-to-apples speed.
- Maintain target numerical sampling/grammar behavior; report draft acceptance, speculative state overhead and useful output.
- Source derived quant bytes, speed and quality all reported; no speed claim if fidelity differs silently.
- Compare disaggregated P/D only where network/topology equivalent and transfer state source-compatible; report ITL stability and total throughput separately.

### Multimodal
- vLLM-Omni model set and exact stream modes pinned; same audio codec, sample rate, image/video model, solver steps/guidance and output validity.
- Do not use generic text tokens/s to rank ASR or video; use TTFA, ASR lag/accuracy, first playable frame, synchronized AV, QoS and memory.
- If vLLM-Omni provides staged single-model inference whereas Zerogravity provides multi-model shared hosting, clearly distinguish *task orchestration* advantages from *same-model inference kernel* speed.

### Statistical promotion
- Run enough paired repeats across cold/warm and representative prompt classes to establish confidence; choose a predeclared service objective and stop criteria before experiment.
- Report all wins and regressions/neutral candidates, including real B/P95 and GPU physical headroom.
- Claim “outperforms vLLM in workload X on hardware Y at config Z” only if same model/quant/source and valid quality, with improvement outside uncertainty, no worse failure/rejection/fairness than declared acceptance threshold.
- Do not claim engine-wide superiority from a single source packed 2-bit decode or a different quantum of work.
- Unknown model/driver/feature combinations remain **UNQUALIFIED**, not default automatically optimized.

## 6. Responsibility, deliverables and change-radius constraints

| Responsible area | Inputs | Required deliverables |
|---|---|---|
| Router/API | models, tenants, capabilities, workload | bounded queue/credits, request lifecycle and typed overload with tests |
| Session/StateRoot | model state domains, snapshots, sampler | exact hybrid prefix meet, causal cache/COW, fork/replay and cancellation |
| DeviceBackend | source kernels/driver and physical loans | real batched Ready/event, source resident descriptors, no stale slot |
| TaskDriver | task-specific iteration | legal prefill/decode/codec/solver chunk boundaries and output validity |
| Planner/cost | prevalidated candidates and empirical profiles | bounded source-verified Pareto plan selection, no per-token heavy optimization |
| Resource/IO | global model+session memory and GPU/RAM/SSD | actual bytes/headroom, capacity and source identity admission |
| Distributed | peer/link capabilities, trusted state | safe P/D transfer and compute/communication overlap |
| Benchmark | exact source/corpus/runtime/hardware | pinned vLLM and vLLM-Omni parity and conditional evidence |

Introducing a new ModelRecipe, tensor codec, drafter or media task should NOT require rewriting central request parser, resource authority, StateRoot or hot-path scheduler. If it does, document measured minimality rationale before implementation.

## 7. Research sources and handoff

Governing contract: .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md.
Upstream study: .agents/prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md.
Original vLLM docs: https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/ , https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ , https://docs.vllm.ai/en/latest/design/prefix_caching/ , https://docs.vllm.ai/en/latest/features/disagg_prefill/ , https://docs.vllm.ai/en/stable/configuration/optimization/ .

This is a **design/acceptance** artifact: the user has authorized documentation and README updates, not new running serving workloads, large downloaded models, driver permissions or deployment.
