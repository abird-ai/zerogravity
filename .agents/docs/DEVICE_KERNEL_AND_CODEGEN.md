# Device abstraction, Rust GPU kernels, code generation and backend specialization

**15 Oct exact gfx1151 specialization addendum:** [hardware candidate and physical safety contract](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) makes EXL3/QTIP packed WMMA GEMV, Wave32 register/fragment map, LDS bank-swizzle/padding, v_dot4 INT8 and SAD trellis, skinny GDN GEMM, grouped expert fusions, and exact first-image patch-merge convolution→GEMM **candidate implementations** only. Each requires model/source/derived-quant, m/n/k, ISA/driver/compiler, physical LDS scratch shared across all warp callers, accumulation/numerics and region event/lease guard. Neither padding=5 nor grouping=16 may enter SIR as source semantics. Use reference closed region or typed Unsupported when no verified candidate; do not silently enable CPU expert offload or graph capture from external performance claims.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**Objective:** support fully Rust-maintained inference logic and custom kernels with peak GPU performance where physical hardware/driver allows, while preserving portable canonical operators. CubeCL/CubeK is a leading *candidate*, not approved mandatory dependency; physical `gfx1151` feasibility gates selection.

## 1. Backend interface and invariant

A backend handles **devices, memory, kernels, stream/event lifetimes, graph capture and errors**. It is not responsible for model interpretation, scheduling, tokenizer, speculative acceptance or resource policy.

Conceptual traits:

```rust
trait DeviceBackend {
    fn capabilities(&self) -> Result<DeviceCapabilities>;
    fn allocate(&self, request: BufferRequest) -> Result<BufferLease>;
    fn copy_async(&self, cmd: CopyCommand, deps: &[DeviceEvent]) -> Result<DeviceEvent>;
    fn compile(&self, kernel: &KernelSpec, opts: &CompileOptions) -> Result<CompiledKernel>;
    fn launch(&self, kernel: &CompiledKernel, args: &LaunchArgs,
              deps: &[DeviceEvent]) -> Result<DeviceEvent>;
    fn capture(&self, region: &CaptureRegion) -> Result<ExecutableGraph>;
    fn wait(&self, event: &DeviceEvent, deadline: Deadline) -> Result<()>;
}
```

Exact async Rust ownership/Send/Sync/ffi types must be determined by implementation. A `DeviceEvent` includes device/stream/generation and must not accidentally call a blocking synchronize on the scheduler executor.

## 2. Capability model

Record vendor, chip/architecture target (`gfx1151`, `gfx1100`, `gfx1201`, `sm_89/90/100/120/121`, Apple GPU generations), waves/warps, max threads, local/LDS/shared memory, VGPR/SM registers, matrix instructions (WMMA/MFMA/FP4/INT8/BF16/FP16), subgroup/shuffle, FP8/FP4 support, atomics, graph support, peer access, RAM/VRAM/pinned mapping limits, transfer bandwidth and versioned driver/runtime.

**Capabilities are verified by runtime probes**, not solely static tables. Distinguish a reported feature, successful compilation and successful execution of a small parity kernel. Unsupported precision may use validated conversion without silently changing quantization and quality.

## 3. Rust kernel source design

The user asked for all Rust. The engine must not turn into a Rust wrapper around another C++ model server. Rust-written kernels should compile to device ISA through a Rust GPU DSL/backend or custom compiler path; normal host GPU runtimes may be accessed via bindings depending on final O-003 decision.

Candidates:
- CubeCL/CubeK for expression, tiling, matmul, autotuning, CUDA/HIP/Metal/Vulkan;
- dedicated `gfx1151` Rust kernel lowerer with architecture intrinsics/LLVM when CubeCL misses key instructions;
- Rust-written CPU SIMD reference/optimized kernels;
- vendor BLAS as an *optional execution candidate* if final all-Rust policy permits driver/library calls, without making model execution dependent on another engine.

Every kernel `KernelCandidate` advertises exact supported shapes/quantization/position/state/memory alignment/numerical behavior/capture and scratch. Plan only runs a kernel if all preconditions hold.

## 4. Output contract and numerical precision

A kernel must specify:
- input logical/stored dtype, block quant scale and packed layout, strides and pointer alignment;
- output dtype/accumulator precision and any reduction/fusion order constraints;
- required zeroed/preserved scratch and whether it writes beyond logical shape padding;
- valid model geometry and row-count ranges;
- state input version and output transition;
- async event and synchronization semantics;
- numerical parity oracle with tensor and full-logit tolerances;
- deterministic property under fixed device plan where relevant;
- expected critical-path performance regime and fallback equivalent.

A successful compile is never sufficient. Tests include bit patterns at min/max quant, block boundaries, ragged rows, masks, zero/invalid expert IDs, NaN/Inf policy and old output guards.

## 5. Kernel families and target-specific strategies

### Dense FFN and projection
Decode1/low rows: memory-bound GEMV/MMVQ, group output rows and minimize load traffic. Prefill: shape-tuned GEMM/WMMA, reuse weights in shared memory. Speculative verifier: 2–8 row MMQ/compact GEMM; use graph capture to reduce launches. Q8 or mixed Q4/Q5/IQ2 format decoding inside tile rather than full materialize where beneficial. Ensure padded K dimensions not mistaken for mathematical inputs.

### Routed MoE
Separate GPU router top-k and weight table lookup; handle scoring bias/sigmoid/softmax exactly. Build token→expert assignment map and expert→token compact worklist entirely on device where possible. For small rows, per-expert GEMV; for many rows, group by expert and use grouped GEMM. Shared expert may run concurrently with routed expert and/or disk prefetch if no ordering hazard. Scatter and reduce in numerical-safe order. Deterministic mode may use two-pass reduction rather than atomics; speed policy requires measured numeric drift/quality checks. Distinct expert dense/down quant paths can be selected.

### Recurrent GDN/KDA
One-token state update is latency and register dependent; chunk prefill is matrix-friendly. GDN and KDA use model-specific gate types (scalar/per-channel) and conv history. Special fused kernels must preserve state after each token required for speculative rollback. Prefill chunked scan has to pass independent recurrence oracles and multiple chunk boundaries. Snapshot tiles can be written to small side buffers only at required rollback cuts.

### Qwen Sparse Attention
GPU indexer score/selection with causal masks, block grouping/pooling, top-k and empty/masked cells. Avoid CPU fallback to top-k in critical decode path if it creates context-depth scaling/round trips. Bounded scratch and register allocation important at 128K. Selection, attention and numerical semantics must match model; proof cannot silently change selected budget.

### GLM sparse MLA + KDA
Compressed latent KV `c_kv` (NoPE where appropriate) versus expanded head KV must be explicit incompatible layout types; absorption saves KV bytes but changes projection shapes. DSA score top-k, query grouping, FP16 latent shared-memory tile boundaries, and KDA recurrent step need separate tuning. Documented GLM has 34 KDA + 11 sparse attention layers, mHC repeated. Optimize prefill and decode separately.

### DeepSeek compressed attention
128-token raw sliding KV, older 4:1 / 128:1 compressed rows depending on layer; eligible indexer layers select historical rows, other layers use compressed history directly. Matrix kernels may fuse gather+softmax/value, but cannot discard required pooling history. User-visible context extension performance is measured not inferred from claimed max context.

### Hyperconnections
mHC/Sinkhorn and gated residual operations have distinct formulas. Kernel needs full width, mixing/normalization behavior, snapshot legal boundaries and FP32-sensitive numeric policy. Fusion allowed only after full-model checks.

### Quantized LM head and sampling
Support wide vocabulary head with direct row tiling, transparent top-k/top-p/min-p and stop/penalty semantics; sampled MTP proposals may need different candidate selection from target's exact distribution, but verifier corrects it. Avoid unqualified approximate shortlist as target logits.

### N-gram/Engram
Hash and row lookup often better on CPU/IO with small staging; table may be enormous and not GPU-resident. Keep row IDs and history deterministic; async pread/read compaction can overlap GPU work but must carry content/position identity and errors. Treat lookups as data-dependent operations, not giant dense embedding matrix multiplications.

## 6. Strix Halo gfx1151

Plan for Ryzen AI Max+395/Radeon 8060S (RDNA 3.5) and unified LPDDR. Use wave32 and carefully tested rocWMMA shapes where supported, Q2/IQ2/Q4 vector dequant, Q8/F16 projections, GPU sparse selector. Cited DwarfStar tuned prefill using 32-head/80-key tiles, specific x64/y64 IQ2 MMQ and per-block Q2_K staging. These are research references; our actual optimal tile shapes and generated ISA must be measured.

**Physical memory:** ROCm GTT/hipMemGetInfo may exceed safely available RAM; kernel code cannot solve OS memory overcommit. Device planner must interact with ResourceArbiter. CPU and GPU share memory bandwidth, so simultaneous CPU expert compute can reduce overall speed on UMA. Compare with discrete GPU CPU expert lanes instead of blindly applying Maya strategy.

**Graph capture:** distinguish driver HIP capabilities from DwarfStar's current eager ROCm implementation. Prove capture+replay state pointer safety and event semantics, or use low-overhead eager streams if capture isn't beneficial.

## 7. CUDA / NVIDIA and other targets

CUDA architectures differ on native FP4, BF16/FP8, tensor core instruction layouts, asynchronous copies and shared memory. Blackwell FP4 can make specific MoE quant kernels superior where hardware supports native FP4; Ada/Hopper may need different Q8/FP16 alternatives. Multi-GPU model sharding and library kernels optional; avoid losing shape-specialization by using generic BLAS always.

Metal on unified Apple memory: Metal graph capture/tapes and simdgroup matrix kernels, portable state sessions. Apple UMA is not identical to AMD gfx1151; model-specific memory residency and codegen should remain backend-specific.

Vulkan: broad portability fallback with subgroup/precision feature probes; extreme performance only where actual kernel features exist, not promised as default. CPU: SIMD portable reference and optional tuned architecture-specific vector loops; keep correctness and CPU-only server functional even without GPU.

## 8. Kernel planning and autotuning

Compile only bounded shape families; select candidate based on measured whole operation and end-to-end plan. Keys include chip, vendor driver/runtime, kernel build, model/op geometry, tensor packing and quant, batch/decode/prefill/verify, context and numeric policy. Static heuristics provide safe default; auto-tune with explicit cost/time budget or offline profiles, not surprise time-consuming warmup.

For every candidate gather kernel time, memory bytes, register/LDS occupancy, launches, GPU clock and power; then run a complete real-model request. Reject optimizations whose isolated improvement does not translate or causes quality/latency regression. Keep publicly documented rejected experiment records.

## 9. Direct execution design

After plan selection, host prepares flat launch parameters with stable pointers/offsets and events. GPU graph replay, if used, sees device-resident control values for changing position/next tokens to avoid rebuilding graph each step. No string name searches, tensor trait-object dynamic dispatch or per-op memory allocation at every token. Host callbacks reserved for IO/cache misses, cancellation boundaries and scheduler cycles.

**Graph safety:** captured memory addresses stable for graph lifetime; pointer updates only where supported. All scratch buffers preleased; no host CPU waiting in stream callback; device-side wait kernels bounded and failure-aware. If cache miss detected, park safely or exit graph region to host-driven event, then resume with new validated slot table.

## 10. Error handling and tests

Device loss/timeout invalidates provisional states. Slow IO cannot lead to spin forever. Buffer generation prevents use-after-free/evict. Host/device races guarded by fences/events; cross-queue copies ordered explicitly. Test with simulator/stub first, then physical GPU, sanitizer under CUDA/HIP where possible and valgrind/Miri where suitable for host ownership. Ensure NixOS loader paths and GPU permissions are handled without privileged operations.

## 11. Need-to-measure technical unknowns

- Can CubeCL generate efficient `gfx1151` WMMA and quantized matmul code, at competitive occupancy?
- How to expose wave/lane permutations, shared-memory barriers and low-level half/bfloat operations entirely from Rust-authored kernels?
- Cost and availability of low-overhead HIP graph capture and dynamic control tables.
- FP4/FP8 kernel availability on multiple CUDA generations.
- Workload-accurate SIMD CPU expert throughput on UMA vs discrete.
- Whether a dedicated custom Rust GPU IR/lowering would be warranted over CubeCL for extreme paths.

Do not answer these by architectural confidence alone; close P01 with disassembly, numerical parity and physical-device evidence.

## 2026-10-10 integrated v2 addendum — Rust GPU provider landscape and portability

**No GPU compiler is a permanent engine ABI.** Evaluate portable Rust GPU tensor DSLs, a stable Rust CUDA tile-programming frontend and optional experimental Rust SIMT/PTX source compilation, each on actually supported hardware with pinned driver/compiler capability and numerical tests. For AMD `gfx1151`, compare CubeCL HIP generated ISA with a Rust-authored direct AMDGPU/LLVM target kernel if its WMMA/MFMA/LDS/tile capability is insufficient. Vulkan portability toolchains and Rust CPU SIMD remain fallback options; Metal is a separate hardware backend, not automatically equal performance.

Source: https://github.com/tracel-ai/cubecl . **Do not equate vendor GPU runtime FFI with user permission to import C++/HIP source kernels**. The historical research `zg-vendor` escape tier remains a **blocked policy choice** until explicit source-language/licensing decision; prioritize truly Rust-authored target-kernel escape paths.

Backend registry now uses `KernelCapability` with region semantic hash, state effects, supported numerical precision, quant/storage layout, shape regime, hardware features, scratch, C IR event/lease contract, capture/eager mode, source/compiler/driver revision and verified evidence IDs. One model can mix providers at safe region boundaries. Compilation may use provider-specific internals, but model SIR/EngineCore cannot import their native types. A `CompiledKernelHandle` is held by `ExecutableBundle`.

**G0 physical proof:** on true `gfx1151`, Q8/IQ2/Q4 GEMV, wave32 low-bit grouped MoE, GDN state update and attention selector vs independent CPU+legal HIP reference; inspect emitted ISA, register/LDS, occupancy and bytes/token. On NVIDIA compatible GPU, test a stable Rust CUDA tile-programming frontend and a Rust CUDA SIMT compiler frontend as separate experimental paths, not whole-workspace pinned nightly requirement. Use real model kernel results and at least one end-to-end request to qualify. Graph captured islands must be tested with changing expert-resident device tables and fail-retire semantics.

**No hardware name assumptions:** older research described an RTX 3060/Cezanne 27GB dev box. P00 must probe actual reachable hardware before selecting test variants or demanding a new Strix machine. A 92/96GB UMA host may have different practical model fit from published 128GB numbers. See `ASYNC_RESOURCE_COMMAND_IR.md` for target-independent event/lease planning.


## 2026-10-12 integration — XDNA2 AIE/Rust source feasibility provider

Add **`AcceleratorKind::XDNA2`** to the device-capability design and a provisional **`NpuKernelProvider`** with explicit programmable AIE tile/BO constraints: fixed matrix input/output shapes and batch multiples, supported block-float encoding (Gufo example `v8bfp16ebs8`, 9 bytes per 8 signed mantissas + exponent), 32 AIE cores (8×4 in source example), kernel precompiled program/instruction fingerprints, resource queue-depth and XRT hardware-context capabilities. The Gufo fork uses `M=4096,K=2560,N=1024` and allows four NPU jobs in flight; these are **one candidate's shape limits**, not a model or KIR ABI constant. Control plane CPU, GPU and NPU all produce typed device events.

**Source language requirement is a real blocker to investigate:** the fork's AIE program is implemented through Python-generated AIE IR and C++ device source, with a C++ XRT host bridge and compiled `.xclbin`. Existing zerogravity user intent says **owned engine and kernels authored in Rust**. A Rust XRT runtime FFI binding by itself is not a Rust-authored AIE kernel. Investigate Rust-generated valid AIE device IR or a supported Rust→AIE compiler path with physical correctness and performance; if no viable Rust-owned path exists, label `NpuKernelProvider` BLOCKED, without using Gufo's non-Rust kernel source or relaxing policy. Explicit user approval is required for any alternative.

**Hardware safety and shared memory:** driver and device capability probe records XRT and ROCm, XDNA generation, IOMMU requirements, GPU-NPU shared LPDDR and fabric power domain. The supplied Reddit/README report **hard machine hangs** during concurrent NPU/GPU clock transitions. GPU-only stays the default; no startup `sudo`, FCLK pin, suspend override, BIOS/AGESA/IOMMU alteration, long-lived heartbeat or systemd deployment without explicit user action and hardware-specific validated procedure. A software NPU timeout cannot recover a totally frozen host. Track pinned import limits and DMA lease lifetime after failure.

**Validation:** independently compare NPU block-float projections with CPU reference and GPU-only Q8_0 calculations (full layers and logits), per-device worst-case numeric and MoE router/indexer sensitive margins, long context and selected model task quality; benchmark realistic (M=4096/8192) under held *same clock* baseline/experimental paths and C1–C8, plus GPU-only fallback for unknown NPU or shape. Report prefill tokens/s, P95, memory/energy and state safety, not implied decode t/s. Source details in [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md).

## v4 refinement — true closed-region providers and device-native loop proof

A KernelProvider is selected at **compile/load time** by exact semantic region, numeric and state-effect contract, bounded shape/ragged constraints, compute encoding, target device/ISA and physical resource plan. It can implement an entire layer/model closed region as Rust-authored fused code, yet must **not own its own session/allocator/service**. KIR is a transient provider/codegen representation; compiled handles and device-resident parameter slots live in ExecutableBundle and DeviceBackend.

A performance-qualified device-native step must show that router IDs, draft acceptance, next-step token input and sampling/grammar data remain on device where semantically legal, not incur host sync each output token. `DeviceBackend` must support async param buffer updates, event readiness and local pool allocation under global reserve. Fallback safe on unsupported shape/context, arithmetic/driver/capture or unqualified SoC power state. GPU-only executable unaffected by absent NPU driver/source compiler.

Device/kernel Rust source policy stays strict; optional driver and vendor libraries are a separately unresolved user decision, and wrapping foreign-language device kernel doesn't itself make it Rust-authored.

## v4.1 — kernel backend is a specialized form of a region-program provider

The old `DeviceBackend::compile(kernel)` and `KernelProvider` sketches describe **native kernel execution**, not all target accelerators. The single `RegionExecutionProvider` contract now accepts a **closed semantic region of any validated scale** and returns either `NativeKernel`, `CapturedCommandGraph`, `AheadOfTimeDeviceGraph` or `StaticallyScheduledDataflow` (plus carefully gated managed-state programs). `DeviceBackend` handles low-level buffer and program submission/event/teardown for its target. **Don't emulate a TPU/wafer/graph compiler by host-interpreting each op** or require a device to offer a CUDA-like single-GEMM launch API.

Backend capability includes `SupportedRegionScope`, op/quant/numeric/shape/graph control, program resident memory, load/compile/warmup costs, topology/fabric placement, collectives and `StateBridgeContract`. A graph-only backend's compiler can accept an **ephemeral interchange graph** containing only known exact supported operations; it is not a new enduring zerogravity IR. An unsupported custom recurrent or sparse op must be partitioned at a legal state/layout cut or rejected with `NoLegalLowering`, not silently approximated. A new compiler/runtime or static device program requires separate Rust-source/vendor SDK/licensing review. Native CUDA/ROCm/Vulkan/Metal kernels remain useful high-performance provider candidates and CPU-only remains independent.

**Blackwell family tests:** coherent ARM+GPU SoC vs discrete HBM/GDDR vs high-bandwidth multi-GPU rack are separate `TopologyProfile`/power and memory classes; do not assume a 273 GB/s LPDDR host inherits datacenter HBM speed. Distinguish NVFP4 (16-value FP8 scale), MXFP4 (different block scaling) and FP8/FP6 compute, physical layout conversion and quality. Graph-only TPU-style accelerator tests need static/bounded shape and verified program state bridge; wafer/static SRAM device needs large-region placement/streamed weights and honest simulator-only status until real compiler/hardware.

Full normative specification: [accelerator execution families](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md). Source/hardware refs: [platform prior art](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md).
