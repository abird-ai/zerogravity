# Prior-art archive — Historical v2 unified master architecture

**Archived 2026-10-12 by user's project-neutral redesign directive. RESEARCH ONLY / NON-NORMATIVE.** This is the historical pre-v3 complete master text, preserved for design provenance, implementation heuristics, older decisions and external references. **Do not follow source-project-named design instructions from here as canonical**; new authoritative system-independent contracts live in `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md` and the rewritten `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`. Research source and license evidence is separate under `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`.

---

# zerogravity — Comprehensive Unified Master Architecture and Product Plan (v2)

**Updated:** 2026-10-12 (v2.1: Gufo XDNA2 + upstream media task pipelines). **Path:** /home/pvl/spaces/abird/src/zerogravity/. **Status:** unified researched technical design; **no implementation or zerogravity benchmark exists yet**. Supersedes the original v1 interpretation where incompatible with the synthesis. Explicit user intent is authoritative in `DECISION_REGISTER.md`; unapproved product/license/library choices remain open.

**Essential sources:** `.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md`, `.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md`, `.agents/docs/SEMANTIC_IR_AND_COMPILER.md`, `.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md`; extensive pinned reference analysis at `/home/pvl/spaces/abird/research/inference/` (read-only source corpus). Implementation roadmap and handoff describe execution sequence, not completed work.

## 0. Mission

Build **one Rust-authored inference compiler + serving runtime** that can run broad known model semantics reliably through portable reference and generic optimized backends, while selecting extreme model/device/workload-specific kernels and dataflow when applicable. The intended combined strengths:

- **llama.cpp**: GGUF/safetensors interoperability, broad model adapters, native CPU/GPU deployment and approachable single-binary experience;
- **vLLM / SGLang**: token-aware continuous batching, prefix state sharing, asynchronous overlap, efficient KV and distributed serving;
- **Gufo / Maya / Strata / DwarfStar**: exceptional `gfx1151` hybrid/sparse/MoE computation, low-bit packed kernels, large model expert/lookup storage tiers, model-specific sampling and true state rollback;
- **An ahead-of-time ML compiler / a tensor compiler / MLIR / a Rust graph-search project / a kernel-dispatch library**: separation of semantic graph, legal tensor rewrites, async resource/stream plans, target-kernel tuning, model-load-time specialization and serving-mode-aware cost;
- **Rust ecosystem**: CubeCL, independent Rust tensor/serving projects, possible Rust GPU/CUDA Tile and SIMT compilers, Rust CPU SIMD and safe explicit device ownership.

**Ambition:** reach the Pareto-best performance/fidelity/latency/capacity frontier for each eligible model/hardware/workload and sometimes surpass specialized incumbents by co-optimizing kernel compute, request batching, expert/prefix residency, IO overlap and speculation. It is a hypothesis needing matched physical benchmarks, **not a guarantee that an IR magically outruns every runtime**.

**Foundational rule:** mathematical semantics once, state effects explicit, lower to specialized direct executable regions with no per-token graph interpretation; maintain correct fallback when specialization absent and openly refuse unknown semantics/insufficient resources.

## 1. User-confirmed scope versus working choices

**User-confirmed:** name zerogravity; code at exact root above; Rust engine and Rust-authored custom kernels intent; new general runtime fallback; Llama, Gemma 4, Qwen3.5+ and specialized Qwen3.8-Flash-Next/GLM-5.3-Flash and further modern families; `gfx1151` Strix Halo focus; target vLLM/SGLang throughput, llama.cpp convenience and Gufo/Maya/DwarfStar speed; maximize possibility of surpassing existing engines and maintain clean long-term adaptability; complete `.agents/{docs,plans}` handoff; combine prior `research/inference` and project plans.

**Working technical design, not user-approved product choices:** single logical engine owner, SIR/RIR/KIR/CIR, bounded compiler, typed state effects, general/exreme KV variants, global ResourceArbiter, route-aware batching, exact sampled speculation by default, model store and source manifest, async IO and per-response mode receipt, CPU oracle first. More details in reconciliation ledger.

**Unresolved:** license, binary name and public UX/paths, primary GPU kernel compiler and whether vendor math FFI is allowed, supported release backends/first models/quant, performance SLOs, release credentials/plugin security and distributed scope. These cannot be silently declared accepted solely because historical research labeled them RESOLVED.

## 2. Unified system architecture

```text
CLI / HTTP adapters / Rust SDK / model-management tools
     |
     +-> ModelStore / tokenizer / chat+grammar / multimodal processing
     |       |
     +-> Artifact parser & verified Architecture/ModelRecipe
                 |
                 v
        [SIR] Semantic Model IR
        pure SSA tensors + explicit versioned state/effect tokens
                 |
                 v
        [RIR] Legal Region Optimizer
        pure math rewrite/fusion, quant packing, symbolic shape/layout
                 |      \
                 |      [KIR] kernel providers:
                 |        Rust CPU, CubeCL, Rust HIP target, a Rust tile compiler,
                 |        a Rust SIMT/PTX compiler, Vulkan/Metal (as verified)
                 |       /
          [CIR] async resource/command planner
          memory leases + transfers + SSD rows/experts + GPU events
          graph capture/IO barriers + state commit points
                 |
                 v
         Immutable ExecutableBundles (verified variants)
                 |
                 v
 HardwareProbe -> pure HardwarePolicy+MemoryLedger <- ResourceArbiter
                 |                                    |
                 v                                    |
        EngineCore (single logical request owner) ----+
        token budget + admission + prefix cache + speculative policy
                 |
           StepPlan / typed WorkerDelta
                 |
                 v
        Device workers -> direct kernel/graph launches
                 |
        event-complete provisional state
                 |
       EngineCore target verification & atomic token frontier commit
                 |
       Stream committed tokens / metrics / mode receipt
```

**Important distinction:** KernelPlan records validated kernel options, HardwarePolicy sets machine-level placement/reservation, ExecutableBundle is the compiled region/command graph, StepPlan is actual requests/tokens. None is the semantic model. A model parser cannot choose vendor GPU kernels; a kernel cannot secretly alter routing/top-k/precision/sampling.

This architecture is based on an ahead-of-time ML compiler's Flow→Stream→HAL and a tensor compiler's Relax→TIR separation *conceptually* while staying a small Rust-native compiler, not embedding their C++/Python runtime or trying to implement a general MLIR clone.

## 3. Semantic IR and compiler legality

### 3.1 Model semantics

SIR represents: dense and sliding/local GQA/MQA, RoPE/M-RoPE, RMSNorm, quant projections and GLU, routed/shared MoE with exact route scoring/sum, Gated DeltaNet, Kimi Delta Attention, Qwen Sparse Attention, absorbed/noPE MLA+DSA, DeepSeek learned time-compressed attention, residual/hyperconnection/mHC, n-gram/Engram lookup, MTP/draft, vision/audio encoders, embedding/rerank pooling.

Each `SemanticOpSchema` declares:
- logical shapes and symbolic constraints, logical versus stored/packed dtype/quant;
- the precise formula and model version/position rules;
- state-domain reads/writes, causal ordering and initial/reset/transition/snapshot/fork/commit/abort;
- permitted pure algebraic rewrites and numerical policy;
- CPU reference and golden source logits/state; known backend lowerings and required resources.

Model recipes are built from validated config/tensor names and operator schemas, not hard-coded GPU targets. Novel model math without a reference op returns `UnsupportedOperator`. An entire model can fall back to generic execution **only when all semantic operators are implemented and resources suffice**.

### 3.2 Optimized Region IR

RIR marks pure subgraphs and mutable effect edges. Legal rewrites: projection/norm/quant fusion; no redundant materializations; local shared-activation quantization; attention/indexer fused prelude; layout packing; grouped routed expert execution; KDA/GDN chunk and decode form; state snapshot region cuts.

Optimization forbids hidden top-k changes, mHC approximation as generic gated residual, cross-quant prefix cache reuse and floating rearrangements outside numeric/quality policy. Fusion is chosen based on full-region critical path, not maximum operation count in one kernel.

### 3.3 Kernel authoring IR and registry

A target kernel provider advertises `KernelCapability`: region pattern/hash, hardware ISA, layout/quant/shape, numeric tolerance, state effects, scratch/leases, capture mode and performance evidence. A candidate may be a hand-tuned Rust `gfx1151` whole-layer implementation, generic CubeCL, Rust NVIDIA Tile/SIMT compiler or CPU reference; all satisfy same SIR contract.

No universal GPU kernel source required; **Rust-origin model and kernels** with backend-specific instruction control. External vendor driver/runtime/math FFI interpretation is an open user-policy detail; never quietly bundle another C++ model runtime.

### 3.4 Async Command/Resource IR

CIR records `Acquire,Release,Copy,Dispatch,PrefetchExpert,ReadLookup,PublishResidency,Wait,Signal,Commit,Abort,PeerCollective` with timepoint edges, buffer alias/lifetime, device affinity and error-state dependencies. It can legally overlap independent CPU/GPU/SSD work and reduce scratch peak before executable graph capture. Device pointer generations/leases guard against expert eviction and unsafe state/layout conversion.

Once compiled, executable variant is direct kernel/graph launch, **no semantic IR traversal during normal autoregressive steps**. Runtime policy picks among prevalidated variants; local JIT-on-miss is opt-in/bounded or runs off interactive critical path.

## 4. General path and extreme path — one implementation spine

Capability tiers:
- **Reference:** canonical Rust CPU tensor and state implementation; independent official model/llama.cpp fixture validation; can use scalar/FP64 diagnostics.
- **Optimized generic:** tuned CPU SIMD or GPU kernels for common projection/attention/MoE, paged KV, batching, compressed/latents when implemented.
- **Extreme:** compiled closed model-region on selected real hardware/quant/shape, fused recurrent/sparse/expert/pointer-table kernels, resident hot path, graph replay and nonresident IO if needed.

Fallback is **per-region where buffer layouts/state transitions are compatible**. Otherwise transition at an explicit canonical checkpoint with validated layout conversion/replay; runtime never silently reinterpret compressed MLA as expanded KV or advance GDN recurrence to rejected draft tokens. A CPU op fallback may be correct but too slow to claim peak performance; show it in `inspect` and mode receipt.

## 5. Physical resources and five+ lifetime classes

- model immutable weights/packed layouts;
- sparse expert slots with generation and source on SSD/RAM/GPU;
- session mutable KV/recurrent/indexer/ngram/sampler/multimodal;
- immutable shared full-prefix snapshots;
- execution scratch/provisional verify state/IO staging;
- distributed rank/transport buffers;
- compiler cached executable/capture graph and code artifacts.

One `ResourceArbiter` enforces real RAM/VRAM/UMA/cgroup/pinned/OS reserve and independent device addressability. **ROCm GPU-visible GTT capacity is not physical free memory.** Each cache has separate policy: always-used dense resident, routed MoE reuse tiers, ngram/Engram bounded disk row reads, paged attention or compressed indexer, state snapshots and scratch borrowing via leases.

Adopt research's stable expert slots/layer quotas, host-mapped ring, async pread/O_DIRECT when measured, hotlist hinting (not correctness), CPU lane only where measured, one-blob optional source pack and structured `ExpertStore` prefetch/publish. On one Strix Halo CPU and GPU contend for LPDDR; don't assume CPU expert lane helps.

**KV:** research's intrusive paged pool with chained hash and optional radix over same physical allocation, and extreme contiguous reserve-all variant, all under global reservation. For hybrid attention/recurrent models a general *state-group coordinator* computes common legal cached prefix frontier across any number of groups, including compressed finalization alignment and modality; vLLM hybrid state docs show incomplete generality for more-than-two groups. Cache identity binds canonical tokens/rendering/model/quant/position/state/schema/tenant+LoRA, not just rendered SHA1 bytes.

## 6. Runtime serving, scheduler and concurrency

Adopt one dedicated `EngineCore` logical owner and async `EngineMsg/EngineEvent`, persistent worker `InputBatch`, diff-based `StepPlan` and `WorkerDelta`. Worker owns device stream/alloc/captures; client never sees raw state. CPU pools handle tokenizer/tool grammar preprocessing; bounded host IO services handle SSD/prefix snapshots.

Scheduling uses vLLM-style unified target token budget plus input/encoder/physical memory/fairness budgets. Chunked prefill, 1-row decode and small speculative verification choose different kernels. Coalescing, 2-in-flight steps, native grouped MoE and expert affinity are measured optional policies, not permanent 2ms or 128/2048 hard constants. Queue priority and decode ITL protect user-facing latency.

**Cost-aware innovations**: plan admission weighs prefix reuse vs recomp/IO, expert residency and cache pressure, draft weights+verification cost, predicted route overlap, dynamic scratch and GPU/CPU+SSD transfers. Group selected routed experts on GPU where applicable; no host routing sync for every resident expert hit. Optimize aggregate committed output and P95 under quality, memory, fairness and power constraints. Peak performance may be different for single stream, concurrency 8, 32K cached context or disk misses.

## 7. Speculation, state transactions and cache correctness

`DraftSource` may be trained MTP, DSpark-compatible, EAGLE/draft model or zero-weight prompt/n-gram proposal. A dedicated `TargetVerifier` computes authoritative logits/states; `CommitPlan {FullDirect, Prefix(n), Replay(n)}` is a *physical execution mechanism* within a type-safe session transaction. Each verifier candidate has provisional multi-domain states. Commit publishes exactly accepted prefix and corrected sampler token, discarding rejected KV/GDN/KDA/indexer/ngram/modalities/RNG/grammar changes.

Sampled speculation should preserve target model distribution by default where implemented; approximation/opportunistic mode explicit. Exactness modes are distinct: fixed-plan greedy identity, reference numerical tolerance, cross-plan quality, stochastic distribution. No blanket bitwise identity across different GPU reduction orders. Acceptance policy must be measured per model/hardware and disabled if net worse.

Full hybrid-state immutable prefix caching and disk snapshots must be checkpoint/schema/version/quant and renderer compatible; copy-on-write or deferred materialization to avoid a full recurrence matrix copy per branch. A small frontier ring can be logical O(1) only if physical state representations allow it; otherwise copy/replay at bounded cost.

## 8. Formats, model store, grammar and new modalities

Input GGUF v2/v3 and supported safetensors/hub variants. Strict model adapter/quant dialect validation, no `trust_remote_code`, content-hash manifests, resumable atomic downloads and GC. Derived acceleration pack optional/regenerable with model fingerprint; research provisional public name "APK" is not automatically adopted (Android collision); own on-disk binary spec gets byte-checked sizes/version before writing code.

A complete convenience layer needs: byte-exact tokenizer and pre-type handling, Jinja-compatible model chat templates, incremental tool parse/reasoning, JSON-schema/grammar masking (a grammar library or verified alternative), sampled stop and grammar FSM transaction, one clean CLI/SDK, OpenAI API and Anthropic/Responses only with conformance, diagnostics `doctor/inspect`, mode receipts/metrics and no unsafe network defaults.

Dedicated `PoolingSpec` runner for embeddings/reranking (no decoder loop/KV), token/mean/CLS/LAST pooling, normalize and Matryoshka dims. LoRA adapters require per-request generation, distinct prefix cache keys, kernel shape and resource budget; defer high-concurrency multi-adapter fusion until tested. Multimodal encoder/processor/projector owns spans/content fingerprints and separate encoder budgets, avoids splitting noncausal spans; audio/video/diffusion can be future semantic extension with no automatic release promise.

## 9. Distributed and heterogeneous

Single physical GPU baseline lean. Optionally partition layers, tensor arithmetic, experts or prefill/decode service across devices/machines, chosen by interconnect/compute/state-transfer measurements. Worker rank must handshake model/quant/layout/semantic/schema/provisional frontier, protect buffers and commit same accepted token prefix or fail and rebuild. No implicit unauthenticated remote exposure. Copying only ordinary KV between hybrid prefill/decode ranks is insufficient.

## 10. Rust ownership, source policy and maintainability

Start with **few coherent Rust crates (~8–12)** for `artifact/model`, `ir/compiler`, `kernel/device`, `memory/state`, `runtime`, `service`, `validation`, plus binary; split as independent release/compile/security seams justify. Avoid historical 20–25 crate scaffold before one end-to-end vertical slice.

One authoritative trait/schema per axis: ArtifactReader, SemanticOp, CompilerPass/KernelProvider, DeviceBackend, ResourceArbiter/KvPool, ExpertSource, DraftSource/TargetVerifier, state codec, Encoder/Pooling, Placement/Transport, frontend adapter. No giant dynamic plugin host required: built-in statically linked Rust providers first, externally loaded executable kernels only after signed/trusted plugin security design.

Technology adaptation test: add a novel model family or target GPU compiler without editing unrelated HTTP/tokenizer/cache implementations. Plan/cache keys versioned by semantic/compiler/driver/quant/numeric/layout/shape. Document data-migration, deprecated operator version and change protocol. Compiler backend version upgrades require kernel parity and model-quality retests, not just Cargo build.

## 11. Physical benchmark + quality governance

Research comparisons must match exact checkpoint hash/quant/position/mode/device/driver/power/context/batch/acceptance and benchmark times. Source Gufo/Maya/DwarfStar numbers remain **external author reports**. For zerogravity, measure:
- kernel microbench and machine-instruction efficiency/occupancy;
- prefill (new tokens vs cached depth) and target AR/committed speculative t/s;
- TTFT, P95 ITL, aggregate throughput C=1/2/4/8/16/32;
- memory headroom, KV/SSM/indexer footprint, expert cache hits/miss critical path, NVMe bytes, power;
- CPU/reference tensor/logit parity, long contexts, tokenizer/tool/grammar, quality NLL/KL/retrieval;
- prefix snapshot, arbitrary speculative prefix acceptance, cancellations, OOM/fail-closed, rank failure.

Use ABBA/BAAB alternation and confidence bounds, full-model end-to-end validation and negative experimental records. One successful microkernel does not establish engine superiority. Dataflow cost model counts overlap and critical path, not naive sum. No silent quality upgrade/downgrade.

## 12. Correctness, security and failure invariants

1. No visible token without atomic publication of valid model and sampler state frontier.
2. No incompatible prefix hit across model/quant/adapter/modalities/position/state scheme.
3. No slot or buffer recycle while in-flight async work has a lease; generation checks prevent ABA.
4. No stateful rewrite/fusion across incompatible effects or unverified numeric assumptions.
5. No expert prefetch hint affects routed expert choice; cache miss aborts on IO error.
6. Physical memory reservation includes OS headroom and permanent/temporary allocations; GPU virtual address is not physical capacity.
7. Kernel fallback may only occur at type/state-conversion compatible boundary; otherwise replay from canonical committed prefix.
8. All model artifact and API input fields are bounded; unknown model ops cannot execute untrusted code.
9. Distributed rank mismatch or failure cannot publish half-committed target output.
10. Every throughput claim includes quality, model/quant/context/concurrency and verified source; no inflated draft tokens as completed output.
11. Serving auth defaults localhost, no unsafe tool execution or untrusted model extension code; snapshots private.
12. Model/kernel compiler/storage/HTTP adapter changes preserve independent owner boundaries and test evidence.

## 13. Research decisions reconciled

See `.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md` for exact historical research council D-01…D-20 dispositions. Key decisions:
- adopt research `EngineCore/StepPlan/WorkerDelta/KvPool/ExpertStore/PlanSource/KernelPlan` as **implementation guidance**, not direct frozen Rust API;
- modify `ModelSpec::kernel_plan(hw)` into semantic model recipe and compiler/kernel provider ownership;
- adopt speculative `CommitPlan` only inside typed full-state transactions;
- retain GGUF canonical and derived acceleration pack but revise broken 48-versus-54-byte disk-KV schema before implementation;
- keep strict Rust source requirement and broaden kernel candidate research to CubeCL, a Rust tile compiler and a Rust SIMT-PTX compiler as of September 2026;
- distinguish hardware policy, compiled execution bundle and per-step runtime scheduling;
- correct overly broad "any model any engine runs" claim;
- do not treat outdated local-box assumptions, old release dates, 75–125 engineer-month estimate or an unapproved license as user facts.

## 14. Implementation dependency strategy

**Do not start by implementing 25 crates or an unbounded compiler.** Follow `IMPLEMENTATION_ROADMAP.md` with the integrated Stage 0 proof:

- Physical `gfx1151` kernel/ISA and alternative Rust compiler bakeoff, CUDA representative second target.
- CPU GGUF tokenizer/quant/mathematical oracle and tiny dense model.
- Minimal typed SIR including GDN and one sparse/indexer state effect; RIR legalization.
- Minimal CIR with leases, two async tasks/device events and simulated I/O errors.
- Research paged KV/core scheduler/frontier fuzzer and `CommitPlan` typed rollback.
- Complete vertical slice: checkpoint → exact adapter/SIR → compiled CPU + one GPU → correct tokens via same EngineCore; forced specialization miss to legal fallback.
- Serving-mode-aware kernel tuning and model quality harness.
- Only after these succeed: advanced MoE tier, rich model breadth, MTP, full async graph capture, distributed and multimodal.

The existing detailed 24-phase ZG-P00…ZG-P23 remain stable IDs, with new consolidated Stage 0 proof mapping and gates in `IMPLEMENTATION_ROADMAP.md` and `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`. **All phases currently NOT_STARTED**.

## 15. Risks and disconfirming tests

Highest risks: Rust `gfx1151` compiler codegen; source-language/vendor FFI policy; semantic/effect legality and full-state rollback; runtime C IR overhead; physical UMA overcommit; prefix/cache alias; unreasonable breadth; too many unproven optimizations; benchmark mismatches; external code license and parser security.

A verified 2026 claim that no Rust compiler emits adequate code for a key operator does not require abandoning SIR. It requires improving or extending the kernel provider backend while retaining Rust-authored source and independent fallback. If a proposed whole-system scheduling innovation adds complexity but shows no repeatable net gain at representative latency, disable it by default and retain the simpler plan.

## 16. Authority and handoff

**Canonical reading:** the agent playbook → decision register → this master → reconciliation and comparative research → `ASYNC_RESOURCE_COMMAND_IR.md` and EngineCore protocol → domain specs → integrated Phase 0/24-phase roadmap → handoff. The research corpus is an important read-only evidence/benchmark library, not a competing authoritative implementation plan. Update decision register/roadmap/handoff with actual proofs when coding.

**Current state:** this is **documentation and source research**, not a functioning runtime, compiled GPU kernel, installed toolchain, benchmark or Git commit. Performance remains an ambitious goal to validate. The design preserves peak specialization freedom without sacrificing known-model semantic portability or system maintainability.

## 17. October 12, 2026 — Gufo fork, XDNA2 and generalized task pipelines (v2.1)

**Latest explicit user decision U-016:** include the implementation techniques in [mikealanni/gufo](https://github.com/mikealanni/gufo) and the [October 11 Strix Halo GPU+NPU prefill report](https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/), ensuring they fit **natively into our compiler and execution pipeline** rather than adding a separate specialized server. The upstream [gufo-org/gufo](https://github.com/gufo-org/gufo) is a distinct existing reference with additional ASR, TTS, image and video task pipelines. Source-pinned technical investigation: [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md). Full integration design: [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

### 17.1 Three heterogeneous execution strategies, one SIR

1. **Intra-operator projection-row partition:** on the same APU, GPU handles some output channels while XDNA2 NPU executes independent GEMM row groups. Source-demonstrated Qwen3.8 Flash-Next prefill partitions GDN z, attention Q/gate and shared expert gate/up; downstream values join at the first consumer. RIR proves exact row coverage and legality, KIR supplies device-specific candidates, CIR schedules NPU+GPU compute and cross-API fences. No XDNA name or hardware status appears in the trained model's semantic math.
2. **Routed-expert owner partition:** separately demonstrated by a discrete-GPU+iGPU expert-ownership approach with faster discrete GPU hot expert ownership, slower/larger UMA iGPU cold expert ownership and optional SSD tier. Sends selected activations and partial results across devices, not whole expert weights every token. Compute owner and residency/state owner are distinct; model's original MoE top-k and weights are unchanged.
3. **Layer/tensor/phase partition:** usual GPU/CPU/distributed pipeline, including prefill-vs-decode and multimodal encoder stages; requires explicit state and layout handoff and topology cost model. A plain GPU-only path remains correct and resource-efficient.

These are three **alternative/composable CIR/RIR placements**, not three independent runtimes. A qualified hardware profile may use several accelerators and storage tiers simultaneously, but the default never assumes availability or benefit.

### 17.2 Hardware support, numeric precision and safety

The supplied fork author reports Qwen3.8-Flash-Next **UD-IQ4_XS** on 128GB Strix Halo: **1,698 GPU-only → 1,821 GPU+XDNA2 prefill tokens/s at 16K** (+7.2%), 32K NPU 1,783 and 40K cold served 1,637. This is **author-reported prefill**, not decode, not a zerogravity benchmark and not proof of universal superiority. Limited 30 greedy-prompt matching outputs do not prove bitwise logits or stochastic distribution preservation. In the source, the NPU GEMM is a fixed **4096×2560×1024** block-float (`v8bfp16ebs8`) kernel with explicit XRT pinned BO and HIP callback dependencies; quantized compute error must be independently qualified. The AIE program source is Python MLIR-AIE/C++, not Rust. Rust source policy requires investigation of Rust-authored AIE lowering and separate XRT FFI license review before executable NPU support.

**Critical safety default:** the author reports hard whole-host machine-check hangs when the GPU fabric clock changes during NPU+GPU work. Manual held clock and power management are environment-specific and can increase idle power. **No automatic privileged tuning, no assumed BIOS/firmware safety, and no NPU offload enabled on unqualified machines.** DeviceCapabilities must represent SoC firmware/XRT/ROCm/safety state and support explicit experimental opt-in. If absent/unsafe/unsupported-shape, the compiler selects GPU-only region without user-visible model changes. A hardware hang is not recoverable by an ordinary inference timeout; physical tests belong on isolated approved machines.

### 17.3 TaskProgram for ASR/TTS/vision/diffusion/video

Upstream Gufo research demonstrates why we must not force every inference model into `EngineCore::next_token`. Introduce **`TaskProgram` with typed stage DAG + iteration + state + output contract** over the same SIR/RIR/KIR/CIR/ResourceArbiter:
- LM: prefill + causal tokens + optional MTP/DFlash2/DSpark speculation.
- ASR: audio preprocessing, speech encoder and model-specific decoder/transcription (potential NPU encoder GEMM, currently unproven).
- TTS: text conditioning, acoustic frame generation or flow, vocoder and audio streaming, speaker/style/voice state as required by model.
- Image generation/edit: conditioning + iterative denoise/flow/sample step + image decoder; separate iteration frontier from causal language tokens.
- Video/audio generation: temporal/frame state, multimodal encoder/denoiser/output-mux stages and bounded streaming resources.
- Embedding/rerank: encoder/pool/score without autoregressive KV/sampler.

These use a **common tensor/operator/kernel/async resource pipeline**, with different task loops and per-stage progress and media output; no media model becomes "supported" merely because upstream Gufo implements it. Each model/task/quant/device has its own acceptance and output fidelity. New accelerators and pipelines can be added without touching existing text server/ModelRecipe logic outside genuinely new operators.

### 17.4 Integration and acceptance

Maintain original P00–P23 identifiers; add an *optional but explicit* `S0-NPU-A…F` feasibility track in [INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md](../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md), model-specific `P07/P11/P13` NPU support and `P17/P18` TaskProgram extension gates. NPU hardware safety/source policy is **not a hard requirement to ship a portable CPU/GPU engine**. Tests include fake GPU/NPU execution graph with row partition/join, pinned buffer lifetime and XRT timeout, an actual Rust-owned AIE compiler/program feasibility trial, numerical source checkpoint parity, under-identical-clock ABBA performance, energy and long unattended safety qualification on isolated hardware. No automatic launch before validation.

### 17.5 Why this strengthens long-term flexibility

The accelerator's hardware details live in `DeviceCapabilities/KernelProvider`, the legal row split in RIR, cross-API resource and scheduling in CIR, and user/task workflow in EngineCore/TaskProgram. A new hardware NPU or model family may benefit from a previously verified projection-slice transformation **without adding another hard-coded model runtime**, while even a total NPU driver/compiler absence leaves the GPU-only model path intact. This is the concrete architectural meaning of "seamless", not a claim that the NPU program has already been implemented.