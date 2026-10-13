# Zerogravity — Heterogeneous prefill and retained replay acceptance

**15 October 2026 · U-048/U-049 · DESIGN ONLY.** Implementation remains at P07-H6
(`eacf7ef`); the requirements and 40 acceptance gates below are not new runtime or
performance qualification. Root README and inference code remain unchanged.

## 1. Design objective and evidence boundary

Qualify native packed-weight/ISA kernels, retained replay, cooperative NPU/GPU
prefill, numeric/state provenance and feature composition within the existing
SemanticProgram → ExecutableBundle architecture. Preserve one Rust mathematical
engine, one physical resource authority and one TaskCoordinator/StateRoot publisher.
No additional model-specific runtime, scheduler or durable IR is required.

Existing design boundaries cover source kernels/encodings, closed model regions,
compiled command graphs, speculation, unified/tiered memory, multi-GPU placement,
continuous serving, and typed media/decision tasks. The requirements below make
banked prompt execution and replay ownership explicit at those same boundaries.

H6 qualifies selected source-exact Qwen3.5 0.8B/2B regions on Renoir/RTX3060 Vulkan
with per-region host readback. It does not qualify a resident GPU graph, batched
NPU prefill, large-MoE execution or competitive throughput. Current native CubeCL
CUDA/HIP compilation is blocked by LLVM23; future qualified native ISA/NPU program
paths need not depend on that specific compiler or XRT.

## 2. Capability ownership and qualification

All rows describe **design expressibility**, not implemented support. Exact default,
opt-in, experimental, unsupported and deprecated states are per source tuple.

| Capability / mechanism | Architecture owner | Qualification requirement |
|---|---|---|
| Rust host + HIP/ISA kernels; runtime without full ROCm SDK | DEVICE_KERNEL_AND_CODEGEN; accelerator provider | separately package compiler/build and driver/runtime; Rust kernel/code-object path, ABI/source/ISA tests |
| Packed affine/codebook quantization, role-specific recipes and overlays | FORMATS_QUANTIZATION_AND_LOADING; verified assemblies; derived quant | independent codec/scale/transform oracle and exact sidecar binding; reader support is not quality admission |
| Tuned GEMV/GEMM/verifier shapes, LDS tiling, fusions | GFX1151 contract; KIR/CIR | register/WMMA/barrier rounding and external GPU co-tenancy tests; no new model engine |
| Resident token graph; HipGraph / retained AQL / PM4 | execution families; CIR | complete recorded launches, real route receipts, patch bounds, code/pointer generations; keep transports distinct |
| Two-bank layer-major NPU trunk | cooperative NPU spec; closed ModelStage region | explicit bank/state/retirement algebra and direct arena consumer, beyond output-row splitting |
| Native amdxdna + Rust-authored AIE programs | optional NPU provider | remove XRT-only feasibility assumption; qualify actual program/driver transport, not require another serving runtime |
| NPU W8/BF16 direct compute | NumericContract; derived artifact/encoding policy | exact twin proof separately from source quality; realized segment/family regime in state/cache identity |
| Pinning, huge pages, shared LPDDR/power, calibrated crossover | physical arbiter; UMA safety; cost model | account driver pin vs mlock; full extra banks/panels; safety veto independent of perf prediction |
| Native MTP, DFlash/DSpark, lookup and gated EAGLE paths | target-authoritative speculation + ModelAssembly | per-source tap/head/RNG/rollback oracle, accepted tokens, truthful feature-combination admission |
| QSA/MLA/recurrent state and long-context KV | hybrid operators; state domains | actual source operators through whole-model128K/256K where model/hardware permit; no sparse-budget reduction |
| Q8/FP8/BF16/FWHT/Lloyd and VMM KV | memory/state numeric encodings | state compatibility/transform quality and capture-safe mappings; no blanket codec support |
| Session append, exact tool-turn tokens, prefix snapshots, disk tier | full-state prefix/SSD contract | original emitted IDs and canonical rendered history, all-domain frontier meet, crash/disk integrity |
| Continuous lanes, MTP rows, shared immutable weights | joint serving/goodput contract | exact per-session state, actual verifier-row budget, singleton fallbacks and model/feature restrictions |
| TP/EP/PP, heterogeneous GPU and mapped experts | placement/remote/expert contracts | topology/collective state proof, measured link cost; NPU+EP not inferred from separate support |
| Curated pull/list/run/chat/serve/bench/quantize/config, aliases, sidecars, TUI/UI | product, CLI/config, trust and packaging | declarative capabilities/profiles; explicit acquisition authorization; frontends share one engine |
| Chat SSE/tools/reasoning/logprobs, Messages/Responses adapters | canonical TaskRequest/API spec | schema/event/error/SDK subset tests; no tools executed by inference |
| Embeddings including late-interaction multi-vector query/document mode | verified task/encoder/pooling | preserve token matrix shape/truncation/markers; never silently replace with pooled embeddings |
| Finite decision readout | typed Decision TaskDriver | explicit label/method/calibration/session contract; explicit task and API capability declaration |
| Vision/OCR sidecars; rectified-flow generation/edit | v5.1 assemblies/media TaskDrivers | each checkpoint/component/solver/processor actually source-qualified; source family map is not a model support guarantee |
| Health/stats/metrics, queue limits, model reload, Nix/container, diagnostics | core/serving/security/release | truthful hardware/feature readiness; bounded backpressure, diagnostics, fault and service recovery tests |

Unsupported or unqualified precision, eviction and speculative combinations remain
explicit research candidates. They cannot become defaults merely because a kernel
exists. New mechanisms attach through the same model/task/encoding/provider
boundaries after semantic and quality qualification.

## 3. Required design contracts

1. Support same-op output-row slicing and two-bank prompt-row, layer-major
   pipelining as distinct transformations. Use a separately qualified legal
   region candidate, with within-session causal A→B state and bank-private data.
2. Support qualified native amdxdna/Rust AIE or optional runtime paths.
   Specify transport/program capabilities; retain current Rust-source policy.
3. Readiness must distinguish terminal error notification from
   successful data readiness; no signal/watchdog may waive DMA/reader lifetime.
4. Generic graph capture needs complete launch accounting, bounded
   patches, transport-specific route receipts and loaded-code lifetimes.
5. W8/BF16 paired and native unpaired tails realize different math.
   A generic tolerant-mode name is insufficient for snapshot/prefix compatibility.
6. Feature support must admit the composed tuple, not independently
   tick NPU, MTP, replay and EP boxes. Performance calibration cannot override
   correctness, source, topology or power/fabric safety.

Radeon 8060S has **40 CUs**; AMD’s [versioned GPU
specification](https://rocm.docs.amd.com/en/docs-10.0.0/reference/gpu-specs.html)
is the primary reference. Wave32 is a kernel profile, not the only hardware wave
mode. Measured costs must use the actual device profile.

These are specified in the [normative pipeline contract](../docs/HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md)
and linked into their existing owners. No runtime feature is marked complete.

## 4. Acceptance matrix — ALL PLANNED / NOT RUN

Existing H6 tests are separate evidence. None of these new gates passed merely by
adding this document. Each requires raw source/binary/model/config/route receipts.

| ID | Gate / rejecting case |
|---|---|
| HP-01 | import exact source packed quantization bits, scales, transforms and overlays with independent decode oracle |
| HP-02 | distinguish lossless repack from W8/requantized derivative and incompatible source identity |
| HP-03 | reject missing/wrong draft, vision, head, overlay or PLE sidecar and unsupported family recipe |
| HP-04 | source tensor/WMMA/LDS/rounding/guards across decode, verify and prefill shapes |
| HP-05 | compare eager/fused under same numeric contract; preserve recurrent state and source sparse IDs |
| HP-06 | co-tenant GPU clients stress LDS read retirement and shared scratch generations |
| HP-07 | all-source op and mutable-state coverage before a resident whole-token graph claim |
| HP-08 | recorded launch count reconciles with eager plus named external boundaries; omitted KV write rejected |
| HP-09 | eager / HipGraph / AQL / PM4 independently show actual executed route and source/state equality |
| HP-10 | wrong code-object ABI, artifact, alias, grid, kernarg or patch bounds fail prepare |
| HP-11 | idle prepared plans retain pointer/code identities or invalidate before eviction/rebind; in-flight reset/cancel waits for retirement |
| HP-12 | per-ISA acquire/release and multi-queue fan-in; unknown effects serialize |
| HP-13 | banked candidate proves exact prompt-range coverage and A-before-B causal domains at every layer |
| HP-14 | live-across-switch activations/HC gates private; scratch/arena reuse waits for last consumer |
| HP-15 | short/ragged/rebalanced pairs, padding zeros and private output row ranges match declared oracle |
| HP-16 | partial-core snapshot cuts capture all source state domains before later-bank overwrite |
| HP-17 | NPU integer/grid-scale/BF16 CPU oracle, GPU twin and actual NPU bytes agree |
| HP-18 | same candidate separately passes quality vs original/selected quant target and sampled distribution where claimed |
| HP-19 | numeric schedule/family/cuts/overlay change invalidates incompatible prefix and compiled artifacts |
| HP-20 | native amdxdna or optional runtime transport returns authentic readiness with target program identity |
| HP-21 | terminal signal on failure releases waits without authorizing success/publication |
| HP-22 | delayed DMA/error/unwind/cancel/quarantine keep arenas/registrations until genuine quiescence |
| HP-23 | bounded pinned-host/driver-pin/bank/panel/program memory admission; no APU double-count |
| HP-24 | SoC firmware/power/clock/IOMMU/co-tenancy reliability before automatic NPU; no hidden host tuning |
| HP-25 | keyed measured family/pair schedule, tail padding and LPDDR contention; unknown profile uses qualified GPU |
| HP-26 | no synchronous per-layer host wait, repeated layout conversion or hidden same-projection GPU duplicate |
| HP-27 | MTP/DFlash/DSpark/lookup/EAGLE source attach, greedy target verification and all-state rollback |
| HP-28 | sampled target distribution/RNG, deterministic plan selection, alternate legal routes and penalties |
| HP-29 | useful accepted tokens/target traversal; record speculation losses, fallback and padded expert work |
| HP-30 | long-context source QSA/MLA/GDN/KV/indexer, compressed cache quality and exact capability limits |
| HP-31 | exact tool-turn emitted-ID continuation; edited history misses; all-domain prefix/disk restore integrity |
| HP-32 | concurrent lanes, variable draft depths, real sum(1+d_i) verifier rows, fairness/TTFT/ITL under load |
| HP-33 | independently admit NPU×replay×speculation×TP/EP combinations; reject unqualified tuples |
| HP-34 | source-correct TP/EP/PP and heterogeneous expert traffic/collectives with no implicit full-weight copies |
| HP-35 | CLI/model lifecycle/config/install/sidecar/management behavior and capabilities match declared subset |
| HP-36 | Chat/Messages/Responses streaming, tools/reasoning/errors/disconnect and backpressure conformance |
| HP-37 | pooled vs multi-vector embeddings and query/document preprocessing; finite decision logits/calibration |
| HP-38 | vision/OCR/rectified-flow task frontiers, edits, component/solver provenance and source-specific outputs |
| HP-39 | matched pinned-baseline ABBA: cold/warm TTFT, decode AR/spec, context/quality/concurrency |
| HP-40 | whole-workload speed/goodput confidence, quality/SLO/memory/energy/reliability; negative results retained |

## 5. Integration order and comparison policy

P03 source/codec/assembly qualification; P05/P07 resident/native execution and
replay; P08/P11 complete model + causal state; P10/P12 state/weight/lookup tiers;
P15 speculation; P16 joint row-aware scheduling; P19 cooperative NPU/multi-GPU;
P17/P18 product/model/media breadth; P20 calibration; P21 packaging/trust;
P22/P23 matched performance and composition gates. Continue current H6 successor
in small verified phases; this review does not skip its incomplete operators.

To claim **faster**, first match exact deployed artifact/sidecars, sampler, prompt
IDs, KV/state numeric regime, context, device/RAM/kernel/firmware/driver/power and
features. Repeated interleaved fresh processes, identical output horizon, explicit
cold/warm cache and p50/p95/p99 uncertainty; count all accepted tokens and all
requests including errors/rejections. Separate source-equivalent comparison from
quality-normalized comparison across derived packs. Report load/prepack/pinning
and original-vs-derived quality independently. No lower precision, fewer experts,
shorter context, cached-prefix advantage, different speculative horizon or kernel
tuning may be concealed as an engine-only speedup.

Further gains are hypotheses: remove repeated host roundtrips/packing; reuse
activations across compatible fused kernels; choose resident/replay/native dispatch
by actual cost; couple expert-row grouping and speculation with NPU bank shape and
SLO; reduce shared-memory traffic without losing source math. Shared LPDDR/power
and Amdahl limits apply. A broader architecture creates options, not a performance
proof, and implementation/generalization overhead must be measured explicitly.

## 6. Review invariants and document validation

Per-domain causal edges attach to the phase that owns each mutation: PLE Prepare
A→B, GDN/KV/indexer Core A→B. Interior snapshots preserve every domain before later
bank writes. Retained pointers and loaded modules remain valid throughout an idle
prepared plan's lifetime or the plan is invalidated before eviction/rebind; submitted
work additionally requires real retirement. HP-11 checks both obligations.

U-049 retains these technical constructs and all 40 gates while removing external
implementation/release attribution, source-comparison narrative and imported
benchmark figures. Acceptance IDs use the heterogeneous-pipeline prefix HP.
Validation covers document links, unique gate IDs, archive inclusion and SHA
integrity, repository quality checks and the unchanged README byte hash. All new
physical/model/performance gates remain **NOT RUN**.

Document cleanup validation PASS: current versioned files/paths reference scan,
all 40 retained planned gates, 31 added local links, unchanged README/runtime,
repository `scripts/check.sh`, all 8 archive regressions and SHA integrity. These
document checks do not qualify the planned physical/model/performance gates.
