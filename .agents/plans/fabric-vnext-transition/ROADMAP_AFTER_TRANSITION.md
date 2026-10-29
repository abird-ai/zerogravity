# Fabric vNext — feature-expansion roadmap AFTER structural/functional transition

**Not part of first S0–F4 / Q0–Q2 migration completion.** This document preserves all current zerogravity product targets and full future design threads, with an implementation ordering that starts only after a correct model-neutral engine, public SDK, standalone serving library and thin CLI exist. Every new capability is separately source/task/numeric/state/physical/performance qualified before being advertised.

## 1. Base prerequisite

Tranche S: complete immediate compiled Rust module skeleton and migrate current exact codecs, family math, GPU provider and one state/resource protocol.

Tranche F: route all currently supported Qwen/Gemma/source/cache/device/pipeline/API features through new engine, exposed through zerogravity public Rust SDK and reusable zerogravity-serving crate. Delete old shadow runners/compatibility.

Tranche Q: one broad assembled check, targeted bugfixes and one final complete acceptance. Preserve original negative/held hardware/driver/model cases. Never claim vLLM-equivalent performance or full multimodal task support merely because the new packages compile.

Only then broaden physical/codecs/multimodal/distributed.

## 2. E01 — production-safe ordinary Qwen CPU and CUDA HF paths

- Finish Qwen3.5 0.8B independent source correctness for *fresh* prompt lengths 47/48/49/50 separate from cached decode at 48–50 and historical 47 prefix.
- Promote general original BF16/HF CUDA only through actual source qualified dynamic GDN/attention/KV/reduction with native model natural-feedback, every full-vocabulary affected state and physical lease/cancel/unknown; no historical finite captured selection override.
- Extend Qwen2B independently with source/numeric/whole-model/state profiles (0.8B does not qualify 2B), then Qwen3.8 27B source/data and independent full model gates.
- CPU/GGUF and original BF16 source remain different selected numerics; CPU/CUDA/HIP/Vulkan separate independent profiles. Cross-width batch bit equality only when profile explicitly promises it; default vLLM source may differ N1 vs B4.
- Long contexts 512/1K/2K/4K/8K and later 16K/32K independently tested for actual semantic state, RoPE, KV/GDN history, output and memory fit, not a position-only test.

Exit: actual ordinary source model run and API through public lib + serving, not finite test-only route.

## 3. E02 — vLLM-class continuous batching, async execution and goodput

- Persistent generated token / request slots with generation identity, bounded incremental metadata and GPU resident buffers; no model catalog/JSON/compilation/allocator per-token work.
- True mixed temporal prompt chunk prefill/decode scheduling with fair old/new session admission, deadline/aging, select task-compatible cohorts, valid ragged masks and independent root.
- Native B2/B4/B8 now works in finite cohort paths; extend actual independently qualified widths/concurrency, dynamic arrivals, vLLM Model Runner V2 lessons and reduce CPU↔GPU sync and first-use compile churn.
- Paged and **complete-domain** full KV/SWA/GDN/KDA/MLA/indexer/grammar/draft/media prefix caching/COW with common-restorable-frontier checks; no KV-only false hit.
- Dynamic shape-band cache, qualified capture/replay, CUDA graph source reader lifetimes, GPU-side sample and feedback where source/grammar allow.
- Unified task-aware scheduler and physical reserve: queued input bytes and uncomputed prompt, active state, verifier rows, prefill chunks, expert union/padding, output credits, streaming tails and tenant fairness, no unbounded queues.
- Joint finite prevalidated Pareto selection for batch/prefill/graph/kernel/SSD-hotset/optional drafting, not exhaustive online compiler search.
- Match current vLLM V1/MRV2 source/profile/hardware/perf in actual task throughput, TTFT, true ITL P50/P95/P99, offered-load rejection, concurrency/fairness and memory/energy.

Exit: workload-real native concurrency and correct state/output, reproducible complete-request benchmark with named winning AND losing cases; no blanket win statement.

## 4. E03 — full multi-model residency and source-aware routing

- One shared model catalog, immutable alias generation, replica groups, source fingerprint, valid TaskDriver and per-replica prepared executable plan. Multiple models/aliases in one process share physical budget without separate per-model server.
- Lazy model load, warm reuse, optional source-qualified unload/eviction/prewarming, component residency, exact request alias pin while hot reload, independent per-session state and output rights.
- Source-aware routing between eligible equivalent replicas/placements; persistent sessions cannot silently change quant/adapter/source/numerical.
- GPU/CPU/UMA and host/SSD complete peaks, OS headroom, multi-domain all-or-none reservation as actual consumer emerges; fairness across tenants/models.
- Serving API handles many qualified models with independent caps, not one Qwen sample model or forced process-per-model.
- Model-catalog metadata targets remain intent only; exact runtime capability receipt required.

Exit: two genuinely different trained model families source-qualified concurrently through same embedded SDK+HTTP and one ResourceArbiter.

## 5. E04 — large dense 27B, first-class SSD and virtual tensor residency

- Real Qwen3.8 27B original pinned indexed shards, complete model config/tensor roles/axis, tokenizer, BF16 model-level logits and all state.
- Bounded verified immutable SourceWeights/source windows, true resident/streamed owned extents (source hashes, FD mutation refusal, partial short read), CPU/GPU first-use event and reader leases.
- Physical O(1) source identity and direct fully-resident execution, optional SSD source window fault/prefetch with actual IO+DMA, no hot-path virtual page table fetch unless required.
- Dense layer-major streaming and activation spill optimized by physical bandwidth and source math, not incorrectly treated as sparse MoE experts. Startup, page cache and UMA memory effects measured.
- Honest capacity vs usable tokens/sec; streaming allows fit when safe, does not promise fast decode if every layer rereads huge weights.
- Later Qwen Flash large learned lookup/experts/draft uses same storage/IO substrate but different policies.
- Local multiple GPU layer split only with exact composed source/state/numeric/physical transfer and cost proof.

Exit: at least one source-exact large dense model across actual admitted memory/SSD envelopes, with full logits/StateRoot and honest cold/warm performance.

## 6. E05 — Gemma4 models and novel architectures

- Gemma4 E2B model-quality/PLE/attention/KV sharing/softcap exact source, CPU reference and independently qualified CUDA/HIP/Vulkan original model paths.
- Gemma4 26B-A4B MoE first, dense 31B after, exact source routing/shared expert/state schemas and boundaries; do not treat MoE and dense as one numerically interchangeable model.
- Expand to Llama, Mistral, GLM, DeepSeek, Phi, Kimi and released actual hybrid/sparse families only with verified artifacts and correct new ops/attention/SSM/quant; metadata target registry is not capability.
- Model integration drill: new architecture using existing semantic ops touches only its model family, one registration and tests. Novel stateful mathematical op extends exact core + CPU reference + optional provider; no server/scheduler/allocator edit.
- Held model payloads (DeepSeek/GLM etc.) remain held until explicit rights/authorization.

Exit: new real model paths demonstrate small change radius and same public SDK/server/session resource infrastructure.

## 7. E06 — future source encodings, quantization, compression and extreme low bit

- Implement exact released Q1/Q2/Q3/NVFP4 variants (bitstream/scales/layout/finite exceptions), distinct CPU reference and physically tested CUDA/HIP/Vulkan kernels; no generic "Q3" format without exact variant.
- Complete Q4_K source-exact CPU Q8_K dot and GPU kernels with real source numerical profile, separate from existing reference scalar Q4_K decoder.
- EXL3/QTIP/trellis/codebook/3.5-bpw source encodings and any other derived quant families: exact source tensor manifest, codebook/basis/bias/shape groups, original source license, offline deterministic conversion and independent held-out quality.
- Distinguish storage compression, reversible device packing, activation/KV temporary quant, lossy derived checkpoint, approximated attention or expert-budget changes and source sampler distribution. User expressly selected approximation profile may trade quality; no silent fallback from exact.
- Tune actual direct quant GEMV/GEMM batch/shape/cache, avoid mandatory gigantic FP16 dequant and measure that lower bpw can be slower due to decode/trellis tiling overhead.
- Source format and numeric profiles remain separate from model family, physical device layout and serving tasks.

Exit: exact per-codec support matrix, end-to-end source/derived quality and performance with real hardware/driver, no format-specific engine.

## 8. E07 — hybrid sparse models, expert/lookup streaming and speculation

- Qwen Flash-Next source-real recurrent GDN/KDA, QSA/sparse masks/indexer, MoE, n-gram/Engram trained lookup tables, independent MTP/drafters; verify all auxiliary sources/sidecars and memory.
- GLM/DeepSeek/Kimi other attention variants only after source equations, independent CPU reference and sampled task quality.
- Expert cache grouped model-selected routing IDs, lookup row IO and shared finite byte budgets; physical prefetch/eviction pins captured readers, SSD/UMA, CPU/GPU overlap as actual cost.
- Target-authoritative speculation, draft acceptance and full rollback for recurrent, KV, sparse indices, grammar and RNG, not KV-only; wrong drafter rejected all tokens can still yield correct target output, so drafter separately validated.
- Joint scheduler considers target verifier rows, acceptance probability, drafted sidecar weight+state memory, unique expert union and bandwidth/IO, not only accepted tokens.
- Support exact target distribution for required sampler mode and separate explicit approximate policies.

Exit: new hybrid model full semantic/state/source/batch/memory and complete-request quality/performance proofs; no new inference engine.

## 9. E08 — native multimodal task drivers and model assemblies

- Real encoder/embed/rerank/decision task as first non-autoregressive proof through current model-neutral runtime and typed task output; no second token scheduler.
- Source-verified ModelAssembly components for Qwen Image, encoder/transformer/scheduler/VAE, actual image/video diffusion/flow stages and typed latents.
- Source-real ASR: Whisper/Canary/Parakeet/Qwen ASR exact audio processors, windows, CTC/RNN-T/TDT/AR stage semantics, timestamps and transcript finality, speaker/diarization only if source-qualified.
- TTS: phoneme/prosody/acoustic flow, RVQ/codec depth, vocoder frames, priming delay, speech audio and model-specific timing; text→PCM cannot be fabricated by generated JSON.
- Native realtime audio/video and synchronized AV, duplex barge-in, capture clock, rational timebases, codecs and output credits; distinguish committed, delivered, ACK and played; no guaranteed human attention from network receipt.
- TaskPipeline of independently registered ASR→LLM→TTS stages on shared memory/device region links, type-checked transforms and staged commit groups. ModelAssembly internal stages remain distinct from cross-model task pipeline. No general tool/agent workflow DSL in engine.
- Public SDK supports tasks/typed event streams without HTTP; standalone serving adapts OpenAI/Responses/WebSocket/WebRTC protocols only as each model/task/transport is independently qualified.

Exit: real trained source task non-token stage and one compound pipeline with source-valid output, physical liveness and complete timing; architecture extended without new engine.

## 10. E09 — heterogeneous and distributed compute, state and memory

- GPU+XDNA2 NPU same-SoC safe partial projection row split/cooperative prefill only with correct source equations/output reduction and known power/driver safety. No hidden root clocks, suspend/IOMMU tuning or unsafe gfx90c GPU resets.
- Multi-GPU layer/tensor/expert/data-parallel qualified physical execution, graph-only device program/AOT/static dataflow with true external program lifecycle and state bridge, not a pretend CUDA raw kernel.
- Four placements compute, parameters, mutable state, reusable immutable cache, plus TransferPlan for state/copy/IO/collective and frozen PlacementEpoch.
- Trusted remote peer enrollment with per-source/tenant authorization, authenticated region-execution worker, actual bandwidth/RTT, epoch/fault/fence/unknown and private session root, no arbitrary remote data egress or mandatory local RPC.
- Disaggregated prefill/decode must transfer *all required hybrid state*, not merely KV, and choose measured QoS isolation over assumed throughput; remote cache hydrate vs remote stateful operator vs RDMA reads separately costed.
- No distributed consensus/HA or true exactly-once client delivery until independently implemented, with lost ACK/reconnect failure cases.

Exit: source/task/model quality parity across a genuinely qualified multi-device program with measured performance and failure safety, while local direct run remains simple.

## 11. E10 — autotuning, release and whole-system evaluations

- Bounded offline kernel/capture/shape/quant tuning with negative experiments and saved qualified evidence keys; never a giant per-token search.
- Hot path cost model includes compiler warmup, source bytes, physical resource stalls, latency and QoS fairness; full-task critical path and accepted-goodput objective.
- Compare selected models/tasks/numerical policy/hardware against pinned llama.cpp, vLLM V1/Model Runner V2 and vLLM-Omni baseline with matched actual source/quant/tokenizer/driver/context and performance class. Publish losing and neutral cases.
- Optional parameter-version/gradient/update TaskDriver future seam without training/autograd as mandatory inference dependency; immutable active sessions pin model version.
- CPU-only/optional-GPU/serving crate/Rust SDK + single binary, Nix/build/cross installer/release/license/security and docs/whitepaper updated only from real source/qualification.
- Security: untrusted source parsers, no unknown model Python, opt-in external listener auth/TLS, tenant/source privacy, snapshot integrity, network rights and retained GPU reader safety.

Exit: verified release and research evidence, not only a specification.

## 12. Crosswalk with existing P00–P23 program

Do not create a second implementation queue that loses original open tasks. Once transition is expressly implemented and accepted, map later E phases back to owning actual existing plans:
- P00–P08: source, math, artifact, semantic program, providers and first model;
- P09/P16: serving, scheduler, API and goodput;
- P10–P12: state, SSD, sparse/hybrid resource execution;
- P13–P15: extreme hybrid families and drafting;
- P17–P18: broad models and multimodal TaskDrivers;
- P19–P20: heterogeneous/distribution/autotune;
- P21–P23: security, release, verified performance/extension.
A plan number is NOT a capability state. Existing plans/TODO owners stay in effect; side plan only integrates architecture/ordering for future approved source work.

## 13. Ordering rationale and stop conditions

Why skeleton/engine first: without one real public Runtime and serve crate, adding codec/model/media features risks more CLI/model-specific dispatch and duplicated source/session code. Why actual source math stays first: general API structure without matching mathematical and physical state does not confer correctness. Why heavyweight qualification deferred: user explicitly prioritizes a rapid coherent transition and one integrated full test; per-phase checks still prevent broken code. Why source/model expansion later: it is independently large and frequently requires licensed model/device.

Halt any new capability if it:
- creates another inference runtime, cache/spendable physical ledger or StateRoot publisher;
- depends on test/fixture branch for ordinary execution authorization;
- silently substitutes source/quantization/approximation or treats old physical proof as fresh;
- introduces per-token model registry/compiler/JSON/allocator global contention;
- interprets audio/image/video work as fake text tokens;
- violates approved hardware/device/peer/source holds;
- cannot be accessed from public Rust zerogravity crate or requires model-specific serving implementation.

The next execution agent must implement the structural/foundation first (S0–F4/Q0–Q2), only then continue this E-series roadmap as independently approved capability development.
