# zerogravity — NPU/GPU cooperative execution and multimodal TaskProgram design

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**Working technical design · 2026-10-12 · NOT IMPLEMENTED.** Adds the **AMD XDNA2 NPU as an eligible first-class compute target** for partial operator execution, inspired by the [mikealanni Gufo fork](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md). Preserves the model-independent, Rust-owned SIR→RIR→KIR/CIR→ExecutableBundle and the single authoritative EngineCore/resource arbiter. Covers NPU + iGPU on a single APU, discrete GPU + iGPU/CPU asymmetric expert ownership (heterogeneous routed-expert research) and future multi-accelerator combinations without creating a model-specific server.

**Read prerequisites:** `SEMANTIC_IR_AND_COMPILER.md`, `ASYNC_RESOURCE_COMMAND_IR.md`, `DEVICE_KERNEL_AND_CODEGEN.md`, `ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md`, `DISTRIBUTED_PARALLELISM.md`, `SECURITY_AND_RELIABILITY.md`, `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`.

## 1. The essential distinction — three hardware parallelism types

| Category | Integrated GPU + NPU example | Discrete GPU + unified iGPU example | Semantics |
|---|---|---|---|
| **Intra-operator row/slice partition** | Same GEMM projection split by output rows between RDNA iGPU and XDNA2 NPU | possible as future extension | Parallel independent linear projection subregions whose outputs are joined in exact original order |
| **MoE expert-owner partition** | NPU does *not* route/execute routed experts in first fork | R9700 fast hot experts + Strix cold experts concurrently; optional SSD tier | Router selects exact experts, owner split executes subsets and reduces unchanged mathematical output |
| **Layer/tensor/prefill-decode partition** | Gufo fork keeps attention/state on iGPU and accelerates chosen prefill slices | dGPU↔APU context/expert and phase-dependent placement | Device plan partitions larger layers, attention or stages with explicit state/layout handoff |

A generalized `ExecutionTopology` expresses CPU SIMD, APU iGPU, AIE/XDNA NPU, discrete GPU, pinned shared memory, SSD/remote as different nodes/edges, with **per-operator/per-phase execution owner**, memory ownership, transfer capabilities, synchronization and power/fabric domains. Never assume that a physically shared address implies cache coherence, zero-copy overhead or independent bandwidth.

## 2. SIR semantic preservation: output-row sliced GEMM

Model's source definition stays:
```
Y[t, m] = X[t, k] × W[m, k]^T (+ b[m] when present)
```
(The precise weight orientation, quant/dequant and accumulation are model/source-version dependent.) A pure RIR transformation may select disjoint row index sets `R_gpu` and `R_npu` covering `[0,m)`, with no overlap, no missing rows, and exact ordering:
```
Y_gpu = ProjectionRows(X, W, R_gpu, source_numeric_contract)
Y_npu = ProjectionRows(X, W, R_npu, source_numeric_contract)
Y     = AssembleRows(Y_gpu, Y_npu, ordered_map, type_conversion)
```

**Required legality checks:** no cross-row dependence inside the operator (an output-wise norm or softmax often cannot be split before its reductions), exact row/bias/output scales and quant/block boundaries, shape tail/padding, model effect dependencies, row-map uniqueness/coverage, numeric tolerance/quality gate and safe host/GPU/NPU memory access. If a downstream consumer only needs GPU rows early, CIR may schedule it early; if it needs both, the join occurs at the first dependency—not an indiscriminate barrier immediately after dispatch. **Do not move a state update or sampler past missing NPU rows.**

This optimization is *model-agnostic*: Qwen's GDN z projection, attention Q/gate groups and shared-expert gate/up are its first demonstrated candidates. Future Gemma/MoE/vision/FFN layer projections are eligible only after their mathematical independence, shape, memory and target kernel support are verified. The original model recipe must not contain `if gfx1151 { use_npu() }` logic.

## 3. Compiler representations and runtime data flow

### SIR
`Projection` semantic operation holds logical math/shape/dtype/routing and state effects. GDN/attention/shared FFN remain as original model ops; NPU is **not** a semantic op that changes the trained model.

### RIR
Create `ProjectionPartitionCandidate` with output row spans, shared input, separate compute dtype/precision and assembled destination. Region fusion can apply each partition independently and do canonical post-projection operations only at dependency-safe cuts. CPU reference and one-GPU implementation are always present for supported models.

### KIR
Kernel-provider `NpuKernelCandidate` declares:
- target `AcceleratorKind::XDNA2`, architecture generation, Rust-authored program identity and qualified native-driver or optional vendor-runtime ABI;
- supported exact `M,K,N` shape bands and tile/subbatch, group queue depth;
- input/output encoding and layout (`v8bfp16ebs8` nine-byte groups/8 values is Gufo example, not universal);
- weight prepacking and tensor quant dequant provenance, canonical output conversion;
- SRAM/L1/L2/host-BO limits, supported instruction bundles, compilation and kernel program identity/hashes;
- numerical accumulation/scale and acceptable error, state/effect independence;
- pinned buffers/aliases and map visibility, host progress/timeout/failure.
A `Rust-authored AIE/kernel` implementation is preferred; runtime wrappers around XRT driver/library can be evaluated under the previously OPEN vendor runtime FFI policy. Compiling/embedding Gufo's C++ AIE and Python generation scripts would violate the stated Rust-kernel-source objective unless explicitly authorized. Don't promise a Rust XDNA2 compiler exists without a successful S0 source/codegen feasibility experiment. If unavailable: `NpuKernelCandidate` capability reports unsupported and generic GPU path remains correct.

### CIR
An eligible prefill plan contains:
```
Acquire GPU/NPU-mapped buffers + pinned budget
        |
Prepare common input X (GPU layout)
        |
GPU PackNpuInput → event(E1) ───────────────────────────────┐
        |                                                   |
        v                                                   v
Enqueue XRT NPU run → token(N1)                    GPU ProjectionRows[R_gpu]
        |                                                   |
 NPU runs while GPU computes others                   GPU next independent ops
        |                                                   |
 XRT BO sync → event(N2)                                  |
        |___________________________                    ____|
                                    v                  v
                            FirstConsumer Join(R_gpu, R_npu)
                                      |
                        downstream norm/gate/sparse attention/
                        recurrent state update/output
                                      |
                full state transaction commit at causal frontier
```

Event graph must express host callback/worker queue, DMA publish/acquire and one buffer lease lifetime across HIP and XRT. Gufo uses HIP `hipLaunchHostFunc` and worker thread with later callback wait; zerogravity may lower CIR to that or a more explicitly event-driven bridge after proving native XRT progress, since blindly blocking a HIP callback can stall unrelated streams. **Host XRT completion token is not a GPU stream event until correctly bridged.** Never blindly assume synchronized CPU cache lines imply NPU DMA visibility. All memory aliases receive distinct `PhysicalRegion` and `LeaseGeneration` plus external-fence/copy semantics.

### ExecutableBundle/StepPlan
Prepared variants:
- GPU-only baseline: all rows on GPU, no NPU lease/callback/power dependency;
- GPU+XDNA2 cooperative prefill: eligible chunk sizes, weights prefetched, NPU power domain qualified, detailed first-use barriers;
- optional distinct GPU+XDNA2 encoder/pooler matrix jobs where task and hardware prove benefit.
StepPlan selects **only validated variants**, not dynamic arbitrary row split/JIT per token. Backend unavailability falls back to GPU before execution. Midflight failure invalidates provisional output and may require replay from last committed full-state frontier, not add partially computed NPU rows as zero. All CPU, GPU and NPU event/IO worker statuses flow through one typed `WorkerDelta` and single logical EngineCore.

## 4. Safety-critical unified SoC power/fabric model

The Gufo fork's source/test setup demonstrates a dangerous failure class: overlapping HIP+iGPU and XRT/XDNA2 commands can trigger a **whole-machine hang when fabric clock/power management transitions**. This is stronger than a single-request error. The architecture must explicitly model a `SharedPowerFabricDomain` (GPU/NPU share SoC power/fabric limits), `DeviceQuirks`, and `AcceleratorQualification`:

- `Unsupported` — absent device/driver/compatible AIE kernel, unknown firmware/clock safety; **never submit NPU work**.
- `ExperimentalOptIn` — user knowingly enabled after diagnostic; host preflight verifies selected firmware/XRT/ROCm/NPU power/clock conditions and a supported combo; document that risk remains.
- `Qualified` — real hardware/kernel/firmware matrix with repeated concurrent soak, suspend/resume, power/idle and model-quality tests; still user-controlled whether NPU is enabled, and regression can demote qualification.

**Safety default:** GPU-only. **Never automatically** run `sudo`, alter systemd boot services, change `pp_dpm_fclk`, disable IOMMU, `/sys/.../power/control`, set NPU power modes or change firmware. Prefer a diagnostic command that reports requested manual actions, available rollback, power impact and conservative workload guard. Holding fabric clock raises idle power and could affect other workloads; even with held clock, twelve no-hang test runs do not prove all machines stable. If safe concurrent clocks cannot be demonstrated on an unmodified host, keep NPU path disabled rather than "auto-tune through" OS hangs.

- Track idle-wake/keepalive power costs; no unbounded heartbeat unless explicitly part of user-selected safe profile.
- Do not initiate NPU wake under heavy GPU load unless physical hardware qualifies that transition; otherwise warm before or disable.
- GPU/XRT driver failure may leave active DMA. **Quarantine buffers until device reset or process end**; never call `free` merely because software deadline expired.
- Attempt timeouts/circuit breakers only for failures software can handle; machine-check lockups require external watchdog/hardware testing and cannot be promised to self-recover.
- Include hardware and firmware identity (AGESA/BIOS where available), device power state, FCLK, XRT/ROCm driver versions and thermal/power telemetry in mode receipt/benchmark reports.
- NUMA/host memory: all pinned/mapped BO bytes count as physical shared RAM and may contend with GPU memory bandwidth. Default global allocator headroom and cgroup limits still enforced.

## 5. Precision and correct mathematical output

Gufo's NPU kernel packs weights and activations in **block floating point** with shared exponent for each eight values. It is **not bit-exact equivalent** to BF16/FP32 projection mathematically. "Generated the same 30 greedy prompts" is a useful smoke signal, not an unconditional safe numerical contract.

Kernel qualification must report:
- per projected row relative/max/ULP error on real Q8_0 checkpoints, including extremes and padded rows;
- full logits, KL, NLL, top-k/tie differences, sparse indexer selected IDs and GDN state after mixed GPU/NPU prefill;
- greedy and stochastic exact-sampling distribution (if speculation enabled), tools/grammar/reasoning, 4K/16K/32K/128K context;
- whether NPU BFP compute changes selected model's **stored quantization** (it should not), but *compute arithmetic* does change; acceptable only under explicit `ValidatedTolerant` and full-model quality gate. `Reference` mode must bypass unqualified NPU computations.

Record per-operator `ComputePrecisionContract` distinct from `StoredQuant`. A zero performance improvement is acceptable; no silent quality regression. If selected NPU candidate cannot meet requested exact numerical mode, choose GPU-only plan.

## 6. Cost-aware ownership and utilization

Model device profile must report:
- **compute** GEMM/MMA peak vs measured shape-specific throughput;
- **NPU tile limit** and queue depth, A/B/C pack and decode bytes, weight prepack/model load time;
- **shared physical bandwidth contention** and pinned memory reservation;
- **serialization latency** via HIP callback/XRT worker vs independent GPU kernel launch;
- **critical first-use timepoint**, not necessarily whole layer end;
- FCLK and power draw; prefill-mode work available to overlap; decoder and short prompts often cannot benefit;
- performance differences under C1 vs batched prompts and under concurrent request/model loads.

A simple correct optimizer chooses NPU only if:
```
T_pack + max(T_gpu_remaining, T_npu_queue+run+sync) + T_decode_join
    < T_gpu_only_critical_path - meaningful_noise_margin
```
This schematic ignores shared bandwidth slowdown; the real cost model measures that and clock/power effects under ABBA tests. A faster NPU micro-GEMM may hurt whole model prefill. The prefill planner chooses row partitions from a small set of **prevalidated candidates**. The model adapter never hardcodes a particular number of attention heads delegated.

## 7. General multi-device ownership and sparse compute

Asymmetric routed-expert ownership and same-SoC accelerator-sliced GEMM are *different composable mechanisms*. Exact named source and benchmark details remain in the prior-art atlas. Support via common `ExecutionTopology` and `RegionPlacementPlan`:

```text
Tier 0: fast dGPU       Dense, shared, hot routed experts, draft, target KV
Tier 1: UMA iGPU        Cold routed experts, optional selected attention state
Tier 2: XDNA2 NPU      Independent projection slices / encoder kernels if qualified
Tier 3: CPU            SIMD fallback and auxiliary/IO
Tier 4: SSD            Immutable expert and Engram/ngram row sources
```

The actual host may have zero or some of these. Placement is not a mandatory 5-stage pipeline; planner maps model's region DAG to available nodes minimizing **measured critical-path latency** and resource/quality constraints, and rejects absent node without changing semantic model. GPU-GPU expert parallelism sends activations and partial expert sums, **not full expert matrices each token**. GPU+NPU projection slicing sends encoded A/B/C in pinned shared memory, and changes neither routed expert IDs nor main state placement. Intra-GPU fusion and NPU slicing are alternatives for a given region (do not attempt both unless compared). Media pipeline encoder may use NPU independently while GPU runs LLM prefill if legal and numerically qualified.

## 8. Extending beyond decoder-only pipelines

The upstream Gufo platform contains Qwen3-ASR, Qwen3-TTS, Qwen-Image and text-to-video/audio work. Model the common inference subsystem as a **TaskProgram** rather than assuming everything is a token-decoding loop:

```
TaskProgram {
  stages: [Preprocess, Encoder, AutoregressiveStep, VerifyDraft,
           PoolOrScore, Vocoder, DenoiseStep, DecodeMedia, Postprocess],
  state_effects,
  resource_reservations,
  output_stream_contract
}
```

These are *semantic task stage kinds*, not obligatory supported ops. Inference examples:
- **LM**: tokenizer → prompt prefill → causal decode/speculative verification → text output.
- **ASR**: audio preprocess/chunks → encoder → text (CTC/seq2seq depending model) with audio buffers/metadata and output timestamps; might place encoder GEMM on NPU and decoder on GPU.
- **TTS**: text tokenizer/encoder → acoustic token autoregression or flow/latent network → vocoder/PCM streaming → audio; needs audio framing/causal state separate from text KV.
- **Image diffusion/edit**: text/image conditioning → multi-step denoiser/flow sampling with timestep schedule → decode image; GPUs/NPU used per stage and shape; **not** simple `next token`.
- **Video+audio**: frame/temporal conditioning, spatial encoder and scheduler, flow/diffusion steps, cross-modal position mapping and output mux; separate memory budget and cancellation/partial-output rules.
- **Embeddings/rerank**: batch encoder/pooling/scorer, no target sampling or KV; multi-device GEMM offload possible where profitable.

All tasks share ArtifactReader/ModelRecipe/SIR math/state and KIR/CIR/ResourceArbiter/DeviceCapabilities; the task-specific loop and output adaptor differ. **Do not implement these models in this design-only task**; document their capability states (planned, reference, validated, optimized), and use the upstream Gufo model guides as research starting points. Future task stages can be added without forking core serving/device/memory implementation. Trainable model weights and evaluator/license remain per task.

## 9. Exact fallback contracts for a "seamless" experience

| Failure or unsupported feature | Correct response |
|---|---|
| No NPU/XRT or unknown AIE program | GPU-only compiled region remains runnable; inspect reports unavailable |
| Short/ragged prefill, unsupported M/K/N, or insufficient pinned RAM | GPU-only, no attempt to force padding unless correctness/size policy supports it |
| `Reference` numerical mode or unqualified BFP precision | GPU/CPU exact reference path; NPU disabled |
| NPU unhealthy before submitting first job | don't use NPU; automatic GPU plan possible at safe beginning |
| NPU times out mid-step / data partially written | abort provisional state, quarantine all shared DMA allocations, only replay from last committed full checkpoint on known-safe device; never mix partial C |
| NPU/GPU power-clock concurrency not qualified | refuse NPU variant preflight, GPU fallback; no privileged modifications |
| GPU-side kernel missing but NPU supports partial projection | can select only if remainder and consumer state are runnable and fully legal |
| NPU shared pinned buffers would push APU into OS pressure | resource arbiter reduces optional pool or refuses NPU, not the whole model |
| Runtime request starts speculative verification | NPU path typically GPU-only unless separately verified for tiny M, exact all-state rollback and benefit |
| Model has unknown novel attention/ASR/TTS operator | explicit `UnsupportedOperator`; new semantic model support required |
| Task acquires new adapter/media inputs | invalidate incompatible prefix/compiled cache key; no silent reuse of old state |

**Seamless** means *no incompatible graph rewrites or manual full-model runtime switch for validated models*, not that every device combination is enabled without testing. A correctly documented GPU-only fallback is a first-class success.

## 10. Acceptance tests and integration plan (proposed, not completed)

**Stage S0-NPU-A — generic IR proof (no hardware):** split a linear projection into GPU/NPU/CPU fake workers, prove row mapping and full tensor/logit/state math for Qwen GDN/QSA/shared expert and synthetic novel model. CIR mock `NpuQueue` with `Submit/Wait/FirstUse` and stale buffer generations; inspect plan cuts.

**Stage S0-NPU-B — Rust source/ABI feasibility:** qualified native-driver or optional runtime FFI and Rust-authored AIE program/codegen path, packaged under the existing optional provider with verified license/build identity. XRT and a particular external compiler are not mandatory architecture dependencies. Qualify Rust-authored AIE programs and native amdxdna or optional runtime transports against the same source, ABI and physical completion requirements. If no permitted Rust-owned device program path is feasible, report that candidate unsupported; do not vendor Gufo's C++ AIE as a hidden dependency.

**Stage S0-NPU-C — hardware safety gates:** dedicated test hardware with NPU and iGPU, firmware/XRT/ROCm/power identities, no destructive automated tweaking, validate concurrency and no hangs on a long soak including cold boot/resume; default disabled until qualified. Record known unsafe host configurations. Simulator cannot pass physical qualification.

**Stage S0-NPU-D — numerical/quality:** exact same Qwen checkpoint and quant, compare solo GPU and cooperative BFP8 projection at independent tensor/full-logit, indexer, long context, copied tokens, grammar and sampled distribution, without lowering trained MoE top-k. Test isolated GDN z, attention query/gate, shared expert and combinations.

**Stage S0-NPU-E — matched benchmarks:** ABBA with same fixed GPU fabric clock and power state for NPU OFF/ON; measure 4K/8K/16K/32K/64K chunks, TTFT/p95/throughput/energy/wall time, NPU queue overhead and prepacking/idle power. Separate GPU-only default-clock and NPU active from clock-change benefit. Compare reproducing author's claimed **1,698 → 1,821 tok/s** on UD-IQ4_XS, not mixing with upstream Gufo Q4_K_XL/MTP results.

**Stage S0-NPU-F — future model/hardware portability:** configure one non-Qwen arbitrary projection slice and one encoder (if supported) without touching EngineCore/ModelStore/HTTP. Unsupported NPU gracefully uses GPU kernel and logs reason; different NPU generation gets different KernelProvider and source compiler fingerprint.

**Release gates:** safety qualification is mandatory and **not inferred from 12 clean tests**. GPU-only remains production default until physical reliability and source policy approved. Quality band and whole-model end-to-end win required to enable NPU automatically on qualified hardware. Any machine-check freeze or unrecoverable DMA hazard blocks automatic NPU enablement.

## 11. Files/policy mapping

- **SIR**: `SEMANTIC_IR_AND_COMPILER.md`, `MODEL_SEMANTICS_AND_COMPATIBILITY.md`, `OPERATOR_AND_FUSION_CATALOG.md`.
- **RIR/KIR**: `DEVICE_KERNEL_AND_CODEGEN.md`, `COST_MODEL_AND_AUTOTUNING.md`.
- **CIR and lifetimes**: `ASYNC_RESOURCE_COMMAND_IR.md`, `MEMORY_RESIDENCY_AND_IO.md`.
- **Hardware/SoC**: `STRIX_HALO_OPTIMIZATION_DOSSIER.md`, `SECURITY_AND_RELIABILITY.md`, `DISTRIBUTED_PARALLELISM.md`.
- **Engine/metrics**: `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`, `SCHEDULER_BATCHING_AND_SPECULATION.md`.
- **Products/tasks**: `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`, `RUNTIME_SERVING_AND_UX.md`.
- **Plan**: `../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, `../plans/IMPLEMENTATION_ROADMAP.md`, `../plans/EXPERIMENT_PROGRAM.md`, `../plans/RELEASE_ACCEPTANCE_GATES.md`, `../plans/DECISION_REGISTER.md`.

**Final invariant:** new accelerator or new media task attaches at an explicit capability/semantic/task/command boundary without changing established, qualified model math. If a new XDNA2 kernel fails or causes a hardware stability problem, all earlier GPU-only execution remains intact and users get an honest diagnostic.

## v4 refinement — NPU is a KernelProvider and physical command dependency

The RIR row-slice candidate is a **pure closed projection transform** under the same SemanticProgram, not a reason to introduce a new model-specific XRT server. A KIR NPU program has fixed verified shape/numeric/compute encoding and physical pinned A/B/C data; CIR tracks actual GPU/NPU external API readiness and first-use Join inside ExecutableBundle. `TaskCoordinator` publishes the session StateRoot only after both device results and all recurrent/sampler/grammar state are Prepared. GPU-only bundle exists independently and is selected on shape/numeric/device-safety failures.

The NPU's hardware power/fabric/firmware risk is a device **safety veto**. A hard-locked shared SoC cannot be recovered by host software timeout, so unknown host configuration remains OFF and no automatic privileged clock/power changes. A Rust wrapper around foreign AIE kernel source doesn't satisfy user Rust-owned kernel source policy. The rest of the runtime and TaskDrivers must compile CPU/GPU-only without XRT; NPU is an optional qualified provider, not a permanent architecture layer.

## 12. U-048 — banked prefill and transport-neutral NPU programs

The [heterogeneous pipeline contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) adds a distinct **two-bank,
layer-major prompt-row** candidate alongside this document's original output-row
projection slicing. It specifies per-domain phase ordering (PLE Prepare A→B;
GDN/KV/indexer Core A→B), private live-across-bank values, last-consumer arena
retirement, complete interior snapshots, terminal error notification versus Ready,
realized W8/BF16/numeric schedule provenance and measured contention/pinning.
XRT examples above describe the Gufo source, not a compulsory transport. Native
amdxdna and Rust-authored AIE program builders are eligible feasibility paths
under the same provider; no external serving engine is introduced. Source-exact
mode and SoC safety veto remain authoritative. This is design-only; the 40 HP
acceptance gates are planned, not hardware-qualified by this amendment.
