# Recovered original pre-product README body

Historical data only. This is the actual former body; the later path became a pointer.
Do not apply its old commands or edit the current user-authored root README.

## Source: be95424837deb7919adf34503db71d27d3ad3adb:.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md

````text
# Zerogravity — Pre-product README implementation/history snapshot (15 October 2026)

**HISTORICAL DOCUMENT.** Original README content before the user-requested product-focused rewrite. This retains dated CPU/GPU qualification notes, release/NixOS instructions, command examples and implementation caveats verbatim for agent handoff. The new root README is the concise user-facing entry point; the live code and CURRENT_STATUS_AND_TRACKING govern current readiness. Do not promote older claims in this snapshot to current status. **Archival link note:** the original Markdown links below were written relative to repository-root `README.md`; when following them from this `.agents/plans/` snapshot, resolve them against the repository root, not the snapshot directory. The original text is intentionally unchanged.

---

# zerogravity

**CURRENT ARCHITECTURE UPDATE (15 Oct 2026):** Storage-tier-neutral model weight backing is a **first-class design**: the same verified logical tensor/ModelAssembly can use direct resident GPU memory, unified RAM, pinned host RAM, OS page cache or demand-streamed SSD extents, with one ResourceArbiter, actual device events and TaskCoordinator. Fully resident kernels pay no mandatory IO abstraction; SSD misses are asynchronously staged and honestly measured. Separate policies cover MoE expert slabs, Qwen n-gram/Engram lookup rows, committed private disk KV snapshots and slower optional dense trunk reload. Complete virtual residency specification; DS4/prior-art source findings; 32 **planned** SSD acceptance tests. No native SSD-streamed model execution or full GPU model support is claimed by these documents.


**CURRENT U-040 / P07-G3b (15 Oct 2026):** One original GGUF-backed `Arc<LoadedQwenModel>` now serves independent Qwen3.5 0.8B/2B sessions without reloading model bytes or immutable layer constants. New `ModelCatalog` provides explicit safe aliases, same-source concurrent load deduplication, lazy list/load, and revoke-new-lookups while existing sessions retain their own StateRoot. Real NVIDIA and AMD Vulkan tests verify *two sessions share one GPU weight upload and two actual source-bit-exact readbacks*. A new original-source-verified GPU region benchmark and physical driver VRAM/GTT telemetry distinguish performance/device allocator pooling from logical GPU envelope release. **Remaining:** complete-GPU Qwen GDN/attention/KV, native CUDA/HIP LLVM23, 8K CPU GDN source drift, physical memory recovery, and actual OpenAI-compatible router/serve API. No false full-GPU/server claims or model downloads. U-040 evidence and next gates.

**CURRENT U-039 · fresh adversarial correctness/design review (15 Oct 2026):** Fixed recoverable host error prematurely fencing GPUs, idle-weight LRU rejecting feasible source+scratch working sets, same-length GGUF pathname replacement mixing old metadata with new source bytes, wasted intermediate full-vocabulary CPU prefill projections, misleading GPU selected-vs-completed counters, unsafe untrusted huge head geometry and per-dispatch resource-snapshot allocation. Real Qwen0.8B/2B CPU chunk parity and both AMD/NVIDIA Vulkan physical source/abort safety tests pass. **Still NOT complete GPU inference:** native CUDA/HIP LLVM23, actual GPU GDN/attention/KV, physical memory pools, high-concurrency throughput and independent 8K GDN numerical drift remain open. Detailed issue-by-issue review, tests and follow-up priority. Full updated historical archive.

**CURRENT U-038 · P07-G3a (14 Oct 2026):** Source-trained **Qwen3.5 0.8B and 2B Q4_0×Q8_0 physically bit-exact on both AMD Renoir/RADV Vulkan and NVIDIA RTX3060/Vulkan** using an explicit `WideF64` per-block accumulation strategy; the original fast `NativeF32` remains an opt-in **source-bit-checked diagnostic** (NVIDIA passes, AMD fails). On AMD, real four-token hybrid Qwen prefills match the original Rust **full-vocabulary logits and StateRoot exactly**; dual-model physical GPU queue completes 4/4 jobs with source cache reuse and no retired logical lease leak. An isolated `hardware-abort-probe` feature physically submits one CubeCL command, intentionally drops the Future before readback, and verifies **five physical GPU handles quarantined, one 256 MiB logical provider charge retained, global PCI-card fenced, second model refused, Qwen root unchanged** on both GPUs. This is NOT driver-level cancellation/recovery or complete GPU inference; FP64 throughput impact unmeasured. CUDA/HIP compiler still lacks pinned LLVM23. CPU Phase1 **16/16 PASS**, four real 16K/32K contexts deferred; separate 8K CPU GDN numerical cause remains open and MM0 mocks unchanged. G3a exact hardware evidence and next gates.

**CURRENT U-037 P07-G2b · 14 Oct 2026:** Real NVIDIA RTX 3060 and AMD Renoir Vulkan GPU nodes are now available through dotlink-l5. The optional Rust CubeCL provider has one **bounded fair cooperative worker queue per physically attested GPU**, using the existing process-shared ResourceArbiter / source-resident P07-F/E completion and quarantine without double-charging. **NVIDIA Vulkan physical original-GGUF Qwen3.5 0.8B/2B selected Q4_0 regions are bit-exact**: four physical cross-model worker dispatches/Ready, two reused weights, clean lease retirement; separate one-token hybrid Qwen full-vocabulary logits/StateRoot parity passes both models. AMD Vulkan physically runs but has small FP32 mismatches under strict source arithmetic: **NUMERICALLY UNQUALIFIED, StateRoot refused**. CUDA/ROCm requires LLVM23 SDK closure and remains unverified; no complete GPU GDN/attention/recurrent/KV model execution yet. CPU Phase1 remains **16/16 PASS** through genuine 8K; 16K/32K extended **16/20 DEFERRED**. Latest P07-G2b physical evidence and exact next steps; full historical archive.
**Latest U-036 · P07-G2a physical memory admission (14 Oct 2026).** CubeCL's optional GPU source-resident adapter now binds an attested PCI/LUID/registry physical GPU identity and explicit operator/provider caps to the process-global Rust `ResourceArbiter`. Static weights + transient scratch stay within one conservative provider envelope, and a cancelled/unverified CubeCL readback globally fences budget reuse while P07-F retains the actual GPU handles. Strict root/GPU Clippy and source-real GGUF host tests pass; **no physical CUDA/ROCm/Vulkan inference has been qualified** in this connector. The 16/16 Phase1 CPU gate is unchanged; 16K/32K remain deferred. Fair GPU workers and detailed pool-level memory qualification remain pending. Exact P07-G2a handoff. Read the regenerated complete comprehensive archive for the full history.

**AUTHORITATIVE U-035 FULL COMPREHENSIVE IMPLEMENTATION HANDOFF (14 OCT 2026):** Read the complete full-history handoff **progressively in full before making further implementation changes**. This generated snapshot contains a current, source-verified executive status followed by the exact text of all 89 other Markdown source records, including every initial user decision, governing architecture/spec, original design research, P00-P23 roadmap, complete historical implementation phases, source validation and pending work, with per-source SHA-256. The latest committed code baseline is 4d1b82f (P07-G1 global admission/fair queue host control contracts). Qwen3.5 CPU Phase 1 16/16 passed; 16K/32K deferred; no new physical GPU full-model proof. The next agent prompt is in .agents/plans/NEXT_AGENT_PROMPT.md. This snapshot is **not** a substitute for inspecting current live Git and source/test evidence. Regenerate using python3 scripts/build-comprehensive-handoff.py --write then python3 scripts/update-doc-manifest.py --write after source documentation updates.


**CURRENT U-035 P07-G1 GLOBAL ADMISSION AND FAIR QUEUE (14 OCT 2026):** Revalidated committed P07-E and P07-F at baseline 602c246 (process-quarantined actual CubeCL handles and combined per-provider weight/scratch budget). New Rust-only source-neutral P07-G1 ResourceArbiter shares an explicitly configured physical-device byte budget across multiple models/sessions/providers, atomically admits typed memory bundles, and globally fences that device on uncertain submitted completion; a bounded FairDeviceQueue enforces per-model/session round-robin, deadline/epoch cancellation and backpressure while skipping memory-blocked requests. **8/8 new control-plane tests passed**, but there is NO integration into physical CubeCL workers or hardware attestation yet: P07-F remains the physical handle-lifetime authority, and native CUDA/ROCm/Vulkan full-model inference is UNVERIFIED. Qwen3.5 CPU Phase 1 **16/16 PASS** through real 8K; 16K/32K extended contexts remain **16/20 deferred**. Full comprehensive implementation handoff is being updated. See P07-G1 exact scope, tests and next integration gates.


**CURRENT U-034 P07-F PHYSICAL GPU KEEPALIVES (14 OCT 2026):** Revalidated P07-E committed state b283fb8 and added fail-closed physical GPU lifetime protection: an in-flight Rust region now retains every original CubeCL source/activation/scale/output Handle BEFORE launch; cancellation or unknown readback transfers all handles and their Client to process-wide quarantine that prevents subsequent use of the affected device service. Combined static resident weights + dynamic inputs + output share one explicitly checked per-provider VRAM budget; global multi-provider resource arbitration remains pending. Root quality workflow, strict GPU Clippy, 10 source-free GPU-package tests and 8 real Qwen checkpoint host tests pass. No new physical GPU execution is claimed: connected device nodes remain unavailable, and actual cancellation/teardown is unverified. CPU Qwen3.5 Phase 1 stays 16/16 PASS; extended 16K/32K still deferred 16/20; GDN accuracy and MM0 architecture streams unchanged. P07-F handoff.


**CURRENT U-033 P07-E CUBECL ASYNC READBACK BRIDGE (14 OCT 2026):** The optional source-resident CubeCL provider now calls the real CubeCL read_async Future, not a fake submitted==complete event. It verifies same-service weight buffers, admits dynamic/output logical leases to Rust DeviceEventLedger BEFORE launch, awaits actual readback, validates exact finite output and only then marks Ready/retirement. The new async-aware source-cache mutex avoids blocking executor threads while waiting but each provider remains serialized; MatVecKernel continues to drive the Future synchronously for Qwen source math and StateRoot compatibility. Pending Future cancellation, malformed completion and device errors fence source residency and refuse reuse. Five source-free bridge tests and an event-backed full-model host-mock test on both real Qwen3.5 0.8B/2B GGUFs pass, WITHOUT physical GPU device qualification. CPU Phase1 remains 16/16 PASS; extended 16K/32K tests DEFERRED. Physical GPU cancelled-buffer retention, multi-stream scheduling, GPU GDN/KV and full GPU model remain next phases. P07-E implementation handoff.


**CURRENT U-032 P07-D DECODE-ONCE / DEVICE EVENTS (14 OCT 2026):** The optional source-resident CubeCL path now decodes immutable original GGUF weights and f16 block metadata ONLY ONCE per source-identity cache miss; per-token preparation reuses compact static host descriptors and original GPU Handle without copying/walking the full trained matrix again. Six exact source tensors across Qwen3.5 0.8B/2B match the old source activation/scale/offset math over four inputs, including all-zero activation. The real three-token host-simulated resident model still matches all logits and StateRoot, with one 1/2 MiB source upload and two cache hits. New Rust-only DeviceEventLedger introduces checked resource admissions, read/write buffer leases, cancellation/epoch fences, physically signaled Ready and sole TaskCoordinator publication; 6 pure event tests and MM-06 duplex/barge-in mock pass, bringing multimodal mocks to 6/12. This is a **typed future asynchronous device control protocol**, not yet a physically integrated asynchronous CubeCL DeviceBackend. Current GPU provider still synchronously waits for readback, and no new physical CUDA/ROCm/Vulkan success is claimed. CPU Phase 1 16/16 PASS remains unchanged; 16K/32K extended contexts deferred. Full P07-D handoff.


**LATEST U-031 P07-C GPU RESIDENCY (14 OCT 2026):** Source-matched persistent GPU WEIGHT residency is now implemented in the OPTIONAL CubeCL provider, keyed by process-unique immutable model allocation identity, exact tensor, codec and geometry. Bounded physical memory leases, deterministic LRU, source-only unsupported CPU fallback, fail-closed uncertain-device-completion fencing, and a typed ReadbackReady receipt now guard the source matvec execution path. The provider reuses static CubeCL source weight handles across tokens, but dynamic Q8/scales still transfer, host GGUF preprocessing still copies weights per call, and the coordinator still runs synchronously. Real Qwen3.5 0.8B/2B three-token HOST-modeled cache tests each show one 1/2 MiB upload, two hits, complete logits and StateRoot parity; actual Vulkan/CUDA/ROCm GPU device dispatch is NOT verified because device access remains blocked. CPU Phase1 remains 16/16 functional evidence PASS; 16K/32K are still explicitly deferred 16/20. The v5 multimodal/accuracy streams remain separate. See P07-C implementation, safety and exact next phase.


**LATEST U-030 P07-B/MM0 PHASE (14 October 2026):** The opt-in native Rust/CubeCL source-mixed matvec provider now implements Q4_1/Q8_1, Q8_0/Q8_0, Q6_K/Q8_K and aligned F32 kernel numerics. All four original Qwen3.5 2B checkpoint matrices pass bitwise host source projections, and each modeled completed region yields identical 248,320 full-model logits/StateRoot. CubeCL Vulkan builds with strict lint; actual device execution is UNVERIFIED because GPU device nodes are not exposed, while CUDA/HIP compilation additionally needs LLVM23. The parallel MM0 stream now has 5/5 source-free rational-clock, typed-port, decision, denoiser and synchronized StateRoot mock tests. Both real CPU Qwen3.5 8K Phase1 gates retain 16/16 PASS, extended 16K/32K remains DEFERRED 16/20. Independent 8K source UB256 vs UB512: all 158 tensors identical for 0.8B; 149/158 differ for 2B, beginning with tiny layer1 GDN drift; no CPU arithmetic or thresholds changed. Full new phase evidence.


**CURRENT THREE-STREAM P07/ACCURACY/MULTIMODAL REVIEW (14 OCT 2026):** Source-verified native Qwen3.5 CPU remains 16/16 pass through real 8K, with 16K/32K explicitly deferred (extended gate 16/20). The P07 optional CubeCL Q4_0 x Q8_0 provider compiles and passes source-trained GGUF host numerical and whole-model completed-region simulation on BOTH small Qwen models; CUDA/HIP toolchains require LLVM 23 and physical GPU device nodes remain inaccessible, so there is NO current actual device-inference result. Independent trained 8K final layer comparisons find first GDN-region deviations at 0.8B layer4 and 2B layer1; reference 2B microbatch schedule varies by up to 0.151 shared top20 logprob, and no numeric acceptance limits were widened or risky source math changes made. An additive v5 multimodal architecture proposal/source atlas defines media timebases, compound components, non-token task drivers, synchronized multi-track StateRoot, decision heads and source codec transforms. This is DESIGN-ONLY for the new model families. Seven separate reviewer briefs are prepared; Codex/Claude CLI subagents are not authenticated on this connector and were NOT invoked successfully. See latest implementation and research handoff for exact evidence, owner boundaries and next actions.


**CURRENT U-027 PHASE1 8K ACCEPTANCE AND P07 INTEGRATION (14 October 2026 SGT):** The user explicitly authorized starting GPU integration after both Qwen3.5 models passed real CPU contexts through 8K; longer 16K and 32K contexts are DEFERRED local tests, not waived/passed. The explicit default Phase 1 CPU gate PASSES **16/16 checks**. The original extended-context gate remains BLOCKED at **16/20** with four real long-context checks unqualified. Both exact source GGUFs passed independent 8x32-token numerics, all trained 146-layer outputs, real 8K mixed-context next-token/top20 comparison, and CPU StateRoot recovery. P07 accelerator integration has **STARTED** with one source-typed Rust MatVecKernel provider boundary, original quantized weight bytes, narrow Unsupported-only fallback, and fail-closed device error/completion checks. A real 0.8B/2B checkpoint integration test confirms bit-identical fallback logits/state and rejection of malformed/incomplete device outputs. **No complete CUDA/ROCm/Vulkan GPU inference implementation or new physical device qualification is claimed**; current dotlink sandbox hides device nodes. Old 7/20–15/20 and 16K/32K-blocks-GPU statements below are historical, superseded only for Phase 1 scheduling. Latest authority and commands.


**CURRENT VERIFIED 8K CPU MILESTONE (14 October 2026 SGT):** **Qwen3.5 0.8B completed and independently passed a real 8,192-token mixed-content CPU context**, with exact next-token match (7993), all 20 top-20 IDs overlapping, max shared logprob error 0.203729244 vs 0.45 bound. **CPU acceptance 15/20**, 2B 8K and both models' 16K/32K whole-context runs remain unverified; full-model GPU work is still gated. The source-bound **2B 8K checkpoint is now running**; inspect `target/qualification/cpu-gate/qwen35-2b/context-8192-work` for the current committed prefix. Earlier 14/20 statements below are historical. Current source-verified CPU handoff.


**LIVE CPU LONG-CONTEXT CONTINUATION — U-026 (13 October 2026):** The native Rust Qwen3.5 0.8B and 2B CPU models have **both passed independently verified 8×32-token top-20 numerical suites** and independently captured trained-model 146-stage per-layer parity. Source-real checkpoint/rollback tests pass. **Both models have now passed a real 4,096-token mixed-content full-model next-token reference check:** 0.8B shared-logprob max error 0.204585, and 2B 0.130986 (20/20 shared top-20 IDs, matching next tokens). **14/20 functional CPU acceptance evidence gates are fulfilled; 8K/16K/32K on both models remain unfinished.** A resumable 8K Qwen3.5 0.8B test is in progress; inspect its exact committed prefix rather than assuming completion. The six outstanding actual full-model long-context gates block full-model GPU work. Historical 7/20, 9/20, 11/20 or 13/20 statements below are superseded, and some earlier 128/512/1024 reports lack current exact binary provenance and need same-build revalidation before an end-to-end release certification. See current live CPU-context handoff for hashes, exact commands, failure recovery, safety caveats, and remaining phases.

**CURRENT PHASE U-026 EXACT SHORT CPU NUMERICAL COMPLETION (2026-10-13):** Qwen3.5 0.8B and 2B run as one native Rust CPU inference engine on exact locally SHA-pinned GGUF files. **Both** latest-binary, eight-prompt x 32-token short numerical suites PASS independently against a separate version-pinned llama.cpp raw-token F32 non-flash oracle: 256/256 raw greedy choices per model, zero near-ties/hard mismatches, all top-20 logprob errors at most 0.000001, including the multilingual case. **Both** trained 24-layer independent 146-vector comparisons PASS, with no nonzero binary32 bit differences (2B had 48 harmless signed-zero differences). Both source-real CPU state/rollback test suites PASS. The fail-closed CPU readiness gate currently reports **11/20 verified**, with **nine real full-model context levels still unverified** (0.8B 4K/8K/16K/32K; 2B 1024/4K/8K/16K/32K). The next authorized phase is **real long-context CPU qualification**. Standalone CUDA/ROCm/Vulkan source quant kernels are physically proven separately, but no complete GPU model runner is qualified or integrated yet. All work remains uncommitted/unstaged on main at 53b360f. Earlier 7/20/failed-corpus entries below are historical, superseded by this independently validated update. See latest exact CPU numerical and context handoff.

**The high-performance AI execution fabric — built for composability and specialization at every layer, from individual kernels to distributed systems**

Zerogravity aims to make it as simple to execute a model locally as a single command, while offering a deeply optimized execution fabric when needed: exact model semantics, hardware-specialized kernels and compiled graphs, batching, multiple models and replicas, distributed state and caching, and native routing/API serving.

The architecture uses **one semantic model program, one compiled physical execution plan, one logical session-state authority, and one shared physical resource budget**. Specialization is intended to happen at any legal computation or task boundary without creating a new inference server per model, accelerator or topology.

Read the technical whitepaper, full engineering specification and agent implementation playbook for the design and phased delivery plan.

## Current status: Qwen3.5 text CPU inference + real GPU kernel qualification

**Latest CPU-first qualification (U-026, 13 October 2026):** The source-verified CPU Qwen3.5 0.8B/2B paths now support **immutable copy-on-write recurrent/KV state snapshots**, atomic chunked prefill, state rollback/replay, and opt-in complete f32 activation tracing of the final prefill token (146 stage vectors for the 24-layer dense model). A developer-only comparator and fail-closed **CPU acceptance gate** live in `scripts/`. **Physical GPU-model integration is still on hold.** The gate currently proves real 128/512/1024 input tokens on 0.8B, 128/512 on 2B, plus both models' checkpoint-based state invariants (**7/20 gates met; 13 unmet**). The 0.8B 8-prompt ×32-token rank/top-20 logprob suite passed its legacy threshold, but a new stricter probabilistic check rejects a **0.117 top-token probability difference**, while the **complete 2B 8-prompt ×32-token report is also now available: 6/8 cases pass, 1 fails due to worst shared top20 logprob error ~0.639, and 1 has six unexposed Unicode-fragment probability steps (32/32 raw greedy token choices still agree).** The numerical checker now saves each case atomically and supports an exact checkpoint/binary/parameter-checked --resume and --max-cases-per-run to avoid losing long runs. Independently captured per-layer oracles and **real 4K/8K/16K/32K full-model** contexts are missing. Synthetic 32K attention/GDN and high-position RoPE math tests **are not long-model proofs**. Detailed evidence: latest CPU qualification handoff.



**CPU comprehensive qualification is NOT complete (U-026).** The two small Qwen models run on CPU, but the user has explicitly prioritized independently verified numerical and long-context correctness **before** full-model GPU integration. We now have atomic state-checked CPU prefill chunks (default 128 tokens), private restore/replay/rollback tests, a raw-token aligned reference logprob audit, synthetic 32K operator-position proofs, and **real** next-token parity following **1,024 input tokens for 0.8B** and **512 input tokens for 2B**. The 2B strict 12/32-token top-20 numerical gate **fails**; independent per-layer reference tensors and full-model 4K, 8K, 16K, 32K prefill runs remain unverified. Prior CUDA/ROCm/Vulkan proof covers only selected source quantized kernels—not full models. See current CPU numerical and context evidence. GPU-model work is held until those CPU gates are cleared.



**A real Rust CPU inference path exists for Qwen3.5 0.8B and 2B.** The CPU decoder executes quantized GGUF weights directly and produces autoregressive tokens without wrapping a pre-existing inference engine. Its current numerical qualification is deliberately narrow: five independent greedy output comparisons, 12 tokens each on 0.8B and 8 on 2B. A near-tie at a later 2B token remains a recorded discrepancy; neither model is fully long-context or performance-qualified.

| Works now | Not yet implemented / qualified |
|---|---|
| `zerogravity run <local.gguf>` with co-located `config.json` and `tokenizer.json` for Qwen3.5 0.8B / 2B **text only**, CPU | Qwen3.5 full-model CUDA/ROCm/Vulkan generation and KV/recurrent GPU state |
| `zerogravity inspect <GGUF|safetensors|config.json>` for recognized source structures | Full ModelStore, sharded checkpoints, format breadth, secure source retrieval |
| F32/F16/BF16/Q4_0/Q4_1/Q4_K/Q8_0/Q6_K reference decoders and CPU quantized GEMV | Performance-optimized compiled generic SemanticProgram/ExecutableBundle |
| Rust Qwen dense-hybrid/GDN text prefill and incremental greedy output | Vision, trained MTP, general grammar/sampling parity, embeddings/other model families |
| CUDA RTX 4060, ROCm Radeon 890M, Vulkan on both GPUs: **physical source-derived Q4_0 projection probe PASS** | Device-resident full-model inference, safe async memory leases/state transactions and service-grade scheduling |
| CPU unit/integration tests, SHA-verified local checkpoint and independent CLI reference smoke scripts | OpenAI HTTP serving, built-in router, multi-model load balancing, distributed computation |

**The physical GPU proof is an isolated, opt-in Rust/CubeCL 0.11 execution probe, not a model inference backend.** Both small models' first-layer, 16-row Q4_0 gate projection has passed a true CUDA, HIP and NVIDIA/AMD Vulkan device test against an independent scalar CPU oracle, with ~1e-8 maximum error. These tests do not yet establish that Qwen's full tokenizer→layer/state→logits→sampling pipeline runs on GPU.

### Current local CPU usage

Only an existing, explicitly selected local GGUF plus the matching official source `config.json` and `tokenizer.json` are accepted. No background download or external model code execution occurs.

```sh
zerogravity inspect target/models/qwen35-0.8b/model.gguf
RAYON_NUM_THREADS=8 zerogravity run target/models/qwen35-0.8b/model.gguf --raw --prompt 'Hi' --max-tokens 8
RAYON_NUM_THREADS=8 zerogravity run target/models/qwen35-2b/model.gguf --raw --prompt 'The capital of France is' --max-tokens 8

# Chat-style user message (Qwen text role delimiters):
zerogravity run target/models/qwen35-0.8b/model.gguf --prompt 'Explain Rust briefly' -n 16

# CPU-only atomic prefill chunk-size variation and optional developer audit:
zerogravity run target/models/qwen35-0.8b/model.gguf --raw --prompt 'Hi' -n 8 --prefill-chunk 32
zerogravity run target/models/qwen35-0.8b/model.gguf --raw --prompt 'Hi' -n 8 --audit-logits-jsonl ./target/qualification/raw-logprobs.jsonl
```

Developer-only source/parity verification using pre-installed `llama-server` as **an independent oracle**, never as the Zerogravity runtime:

```sh
# Source/parity verification procedure (the former standalone scripts were retired;
# the checks below remain the required evidence):
#   - 0.8B independent greedy verification: 12 tokens
#   - 2B independent greedy verification: 8 tokens
#
# Strict raw-token aligned probability audit and actual context next-token checks:
#   - 0.8B SHA256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf: 32 tokens, 8 prompts -> target/qualification/numeric-08.json
#   - 2B SHA256 91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6: 512 target input tokens -> target/qualification/context-2b-512.json
```

See current Qwen CPU + GPU kernel handoff for checked model hashes, numerical caveats, physical test machines and remaining work. See opt-in GPU kernel and Q4_0 probe for exact CUDA/ROCm/Vulkan dependencies and hardware qualification; its large GPU/compiler dependency graph is **not** in the normal CPU binary.

### Developer CPU numerical acceptance gate

The GPU model path must not be treated as qualified until the strict numerical, physical full-model context, real-source state and **independent** per-layer tests all pass. Missing/untrusted evidence yields nonzero exit status and an actionable list of blockers:

```sh
python3 scripts/cpu-qualification-gate.py --evidence-dir target/qualification/cpu-gate --report target/qualification/cpu-gate-summary.json
# Exit 4 / BLOCKED_CPU_FIRST is the expected current status.

# Capture all source-model layer stages locally (does not itself prove parity):
zerogravity run target/models/qwen35-0.8b/model.gguf --raw -p 'Hi there!' -n 2 --audit-layers-jsonl target/qualification/candidate-layers.jsonl

# An independent llama/other reference capture is still needed:
python3 scripts/compare-qwen-layers.py --self-test
# State round-trip verification procedure retained (former standalone script retired).

# Run/recover a checked eight-prompt 32-token corpus in batches of 3.
# Exit 5 means INCOMPLETE, 4 means complete FAIL, 0 means complete PASS.
# 2B SHA256 91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6: 32 tokens, 8 prompts,
# max 3 cases per run, resumable -> target/qualification/cpu-gate/qwen35-2b/numeric-32.json
```

The private layer JSONL contains the entire f32 activation vectors, not the source weights, and must not be confused with an independent reference. Exact checkpoint hashes and the provisional probability tolerance criteria are in the CPU handoff.

### Confirmed implementation order

1. Qwen3.5 0.8B — CPU **numerical and real long-context qualification**, then full-model CUDA, ROCm and Vulkan.
2. Qwen3.5 2B — CPU **numerical and real long-context qualification**, then full-model CUDA, ROCm and Vulkan.
3. Gemma 4 E2B — full-model CPU first.
4. Gemma 4 26B-A4B — full-model CPU first.
5. Qwen3.8 27B — full-model CPU first.

The larger three models' GPU backends will be addressed only after their correct CPU implementation. A model fitting in GPU memory is not proof that its math, quantization, state effects, numerical contract and tokenizer are already supported.

### Still-future unified UX

```sh
zerogravity run coding
zerogravity serve
zerogravity inspect coding --explain-plan
```

Named model aliases, managed downloads, the router, HTTP APIs and multiple model replicas are **not implemented yet**. Today `run` accepts a local Qwen GGUF path; `serve` still returns an explicit not-implemented error.


## Build locally

Requirements: Rust **1.95.0** (pinned in `rust-toolchain.toml`) and Cargo. The CPU library uses pinned `serde`, `serde_json`, `tokenizers` and `rayon` crates in `Cargo.lock`. Offline builds work when all locked crate sources have been cached. The separate opt-in CubeCL probe has its own independent Cargo.lock.

```sh
cargo build --locked --offline
cargo run --locked --offline -- --help
cargo run --locked --offline -- --version
cargo test --locked --offline --all-targets
cargo fmt --all -- --check
cargo clippy --locked --offline --all-targets -- -D warnings
cargo build --release --locked --offline
./target/release/zerogravity --version

# Convenience wrapper for the local Rust, shell and Nix-format checks:
./scripts/check.sh

# Test the checksum installer against a local offline release fixture:
./scripts/test-install.sh ./target/release/zerogravity
```

On Nix/NixOS, the flake uses **pinned nixpkgs, crane and rust-overlay**. The native package is built with crane's separate dependency-build and package-build derivations, while tests/lint checks have independent outputs.

```sh
nix develop
nix build .#zerogravity
nix run .# -- --version

# Native tests, clippy, rustfmt, Nix formatting,
# shellcheck, offline installer fixture, workflow lint and NixOS module evaluation.
nix flake check
```

Nix may need to download pinned toolchains and input sources on first build. Offline Cargo builds do not. The existing tracked Git project and its Nix source definitions need fresh CI/crane/cross-target verification after adding inference dependencies. Untracked working-tree source files may not be included in flake Git source snapshots until staged/committed; do not treat local Cargo success as a passing six-target Nix release.

## Cross-platform releases

The `nix/release.nix` crane cross-build graph defines **six individual binary assets** and SHA-256 sidecars:

| Asset | Rust target | Build approach |
|---|---|---|
| `zerogravity-linux-x86_64` | `x86_64-unknown-linux-musl` | static MUSL, from Linux x86-64 |
| `zerogravity-linux-aarch64` | `aarch64-unknown-linux-musl` | static MUSL cross-build |
| `zerogravity-windows-x86_64.exe` | `x86_64-pc-windows-gnu` | MinGW cross-build |
| `zerogravity-windows-aarch64.exe` | `aarch64-pc-windows-gnullvm` | LLVM-MinGW ARM64 cross-build |
| `zerogravity-macos-x86_64` | `x86_64-apple-darwin` | pinned macOS SDK + LLVM/LLD cross-build |
| `zerogravity-macos-aarch64` | `aarch64-apple-darwin` | pinned macOS SDK + LLVM/LLD cross-build |

```sh
# Produce all six asset files + .sha256, VERSION, PLATFORMS.txt
nix build .#release-all
# Or copy and verify them into a local dist/ directory:
./scripts/build-release-artifacts.sh dist
```

The **definitions are provided**; adding pinned tokenizer/Rayon dependencies and source files does not constitute validation of Nix cross builds. Real execution on every Windows/macOS/Linux architecture, Apple SDK availability/licensing and code-signing/notarization are **not claimed complete** until their release builds and smoke tests pass on appropriate machines. macOS builds are currently **unsigned**, and full-model native GPU kernels are not yet part of the release graph; the optional GPU probe is separately qualified on pvl-a1. Each release tag must match the version in `Cargo.toml`.

### GitHub Actions

Build workflow runs on PRs and pushes to `main`:

1. Install Nix; run native crane tests, clippy, rustfmt, shellcheck, Nix formatter, workflow lint and module evaluation.
2. Cross-build all six release packages through `nix build .#release-all`.
3. Verify SHA-256 sidecars and the Linux binary version; upload CI artifacts.
4. **Only on a matching `v*` tag**, publish the six binaries and their checksums as GitHub Release assets. Existing release tags are **not silently overwritten**.

**The live repository now has a Git remote at `github.com/abird-ai/zerogravity`, with initial commit `53b360f`.** This does not establish that the newly implemented Phase 1 work has been pushed, that CI passes, or that a public binary release exists. Hosted CI and cross-target release gates must be verified independently before publishing a tag.

### Install from a published release

Two checksum-verifying installers are provided. They download the correct OS/architecture asset, verify its SHA-256 and reported binary version, then atomically replace the executable under a user-controlled directory. They do **not** install a daemon or system service.

```sh
# Set OWNER/REPO explicitly once an actual GitHub release exists.
ZEROGRAVITY_REPO=OWNER/zerogravity ./install.sh

# Choose version and destination, without root:
ZEROGRAVITY_REPO=OWNER/zerogravity \
  ZEROGRAVITY_VERSION=0.1.0 \
  ZEROGRAVITY_INSTALL_DIR="$HOME/.local/bin" \
  ./install.sh
```

Windows PowerShell (after publishing):

```powershell
.\install.ps1 -Repository "OWNER/zerogravity" -Version "0.1.0"
```

Both installers also accept an explicit release base URL override (environment variable `ZEROGRAVITY_RELEASE_BASE_URL`; on Windows, `-ReleaseBaseUrl`). No GitHub organization or repository is guessed or silently contacted.

**Supply-chain note:** a checksum fetched alongside a release asset prevents accidental corruption but **is not a cryptographic authenticity signature** against a compromised release host. Signed provenance and Windows/macOS signing are future release-policy choices.

## NixOS module

The flake exports `nixosModules.default` and `nixosModules.zerogravity` for eventual daemon operation:

```nix
{
  inputs.zerogravity.url = "github:OWNER/zerogravity"; # after publishing

  outputs = { nixpkgs, zerogravity, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        zerogravity.nixosModules.default
        {
          # Disabled until 'serve' is implemented.
          services.zerogravity.enable = false;
          # Future:
          # services.zerogravity.enable = true;
          # services.zerogravity.workingDirectory = "/srv/models";
          # services.zerogravity.extraArgs = [ ... ];
        }
      ];
    };
  };
}
```

The module is **disabled by default**, uses `DynamicUser`, a state/cache directory, journal logging, a restricted umask and no automatic firewall exposure. If manually enabled today, its `ExecStart` runs the deliberately unimplemented `serve` subcommand and **exits with an error**, without restarting indefinitely. The module does not implement a server by itself; when `serve` is implemented its service settings will be revisited and integration tested.

## Project structure

```text
Cargo.toml / Cargo.lock          Pinned Rust source/reference + Qwen3.5 text CPU runner
rust-toolchain.toml               Minimal Rust toolchain
src/main.rs                       Only --help, --version, honest reserved-command errors
tests/cli.rs                      Executable smoke/CLI tests
flake.nix / flake.lock            Pinned nixpkgs + crane + rust-overlay
nix/                              Native/checks/cross/release/dev/module
nix/tests/module.nix              NixOS service option/unit evaluation
.github/workflows/                PR/push quality + cross-build + tag release
scripts/build-release-artifacts.sh Cross-build and local asset verification
scripts/check.sh / test-install.sh  Local quality gate + offline installer test
install.sh / install.ps1          Portable, checksum-verified release installers
docs/                             Whitepaper, complete engineering spec, bibliography
.agents/docs/                     Detailed subsystem architecture + prior art
.agents/plans/                    Phase plan, test gates, decisions, agent handoff
```

The existing documentation index provides a guided reading order for technical reviewers and implementing agents.

## Engineering principles

- **Correct by construction:** the trained model's mathematics, quantization and every KV/recurrent/sampler state effect are authoritative; unsafe specialization is rejected or falls back.
- **Modular without hot-path penalty:** compiled closed-region kernels/graphs and one event/resource plan instead of one inference engine or general interpreter per specialization.
- **Hardware and topology independent:** CPU, GPU, NPU, AOT graph and future fabric providers, optionally across explicitly authorized network peers; model weights, compute, mutable state and caches are separately placeable when actually beneficial.
- **One state and resource authority:** workers prepare provisional versions, one TaskCoordinator publishes a complete logical root, and the physical resource arbiter admits safe memory while device-local pools stay fast.
- **One model catalog/router:** exact model aliases and versioned profiles, compatible physical replicas and safe model residency; no silent switch to a different checkpoint/quant to satisfy a request.
- **Simple first, completely explainable later:** one-command local run, one API server and deep configuration without forcing complexity on everyday users.
- **Verified claims only:** a design document, stub CLI, or cross-build derivation isn't proof of tested model or hardware performance.

## Implementation sequence

P00 packaging/bootstrap remains PARTIAL. **U-025 adds working Qwen3.5 0.8B/2B CPU text inference, real GPU kernel/Q4_0 source projection proofs, and a reference-only StateRoot primitive; the full SIR/ExecutableBundle/stateful GPU model executor is not implemented.** The packaging handoff and validation ledger distinguishes local passed checks from cross-platform/Nix-host checks still blocked or unrun. The original 24-phase roadmap and agent playbook remain the engineering authority.

Next phases, **only when explicitly requested**, start with a source-verified model/quant/tokenizer CPU reference, effect-aware SemanticProgram and TaskDriver, one prepared resource/event executor and first correct committed output. Only after that should `run` and `serve` become functional, using the **same** engine.

### License

The project's **code and document distribution license is not yet selected**. The project does not assume the reference project's MIT license is transferable to zerogravity. Do not publish this source under a license or incorporate proprietary model weights/device compiler components without the necessary owner/license decision.

````
