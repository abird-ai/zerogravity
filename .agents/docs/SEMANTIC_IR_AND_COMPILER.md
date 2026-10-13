# Semantic IR, state-effect system and ahead-of-time/JIT execution planning

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**This is the core architecture specification.** The IR does not itself improve GPU throughput. It creates a typed, optimization-safe bridge from exact mathematical model semantics to direct compiled GPU/CPU execution, with state transactions, tensor layouts and hardware specialization.

## 1. Principles and boundaries

1. Model **semantics** must be independent of model file, hardware backend, runtime scheduler and kernel implementation.
2. All state mutation is represented through explicit ordered effects, state versions and commit frontiers.
3. Compiler transformations must be verifiably legal under a selected numerical contract; moving two floating reductions or changing sparse selection may not preserve required behavior.
4. Backends may replace any *closed region* with a hand-written Rust target kernel or captured device graph if the region's type/effects/numerics and scratch boundaries match.
5. The semantic IR is not executed interpretively in the performance path. Model-load/compile creates executable launch programs.
6. Capability detection and plan variants are bounded to prevent a combinatorial JIT explosion; use meaningful regimes instead of per-token recompilation.
7. An optimized region is optional; a compatible canonical lowering exists for every supported semantic operator.
8. Core IR syntax must be small enough for exhaustive validation and human-readable dumps, yet expressive enough for modern recurrent/sparse/MoE models.

## 2. Three-stage representation

### SemanticModelIR (SIR)

Typed SSA tensor value graph **plus effect/state graph**. Captures mathematical operators, model-layer structure and resource requirements, model and tokenizer identity, assumptions about position/masking and API-visible semantics.

**Static:** model dimensions, weights, layer classes, quant format requirements, dependence graph, cache schema. **Dynamic inputs:** prompt tokens, modality spans, previous session state, actual positions, expert router results, sampling RNG, request batch size.

Stateful ops are not pure data-flow: a GDN step consumes one state version and produces another. A sparse indexer depends on previously compressed blocks. Speculative verification generates provisional state versions and must not advance externally visible state until committed.

### OptimizedExecutionIR (EIR)

Semantics lowered into region DAGs, tensors assigned to layout/storage/tiling, fusion groups, intermediate lifetime/scratch, typed device/host transfers, effect barriers, supported batch/context ranges and numeric constraints. Can represent multiple strategies (prefill GEMM vs decode GEMV vs verify grouped kernels, GPU resident vs cache miss) for same semantic region.

### DevicePlanIR (DPI)

No semantic interpretation in execution; immutable plan contains resolved kernel handles or GPU graph captures, kernel argument packing/shape variants, memory buffer leases, stream+event dependencies, host-driven IO callbacks only at necessary slow paths, state transaction begin/finish and fallback exits. Plan may use a direct whole-layer specialized kernel. Single physical GPU must not need server scheduler callbacks per node.

## 3. Suggested conceptual type system

```rust
type ValueId = u32;
type OpId = u32;
type StateId = u32;
type RegionId = u32;
type TensorId = u32;

enum ElementType { F32, F16, BF16, Fp8E4m3, I8, U8, I32, I64 }
enum QuantEncoding {
    None,
    Gguf { kind: GgufQuantType, block_elems: u16, block_bytes: u16 },
    BlockFp8 { block_m: u16, block_k: u16 },
    Mxfp4 { block_shape: Vec<u16> },
    CustomVerified { schema_id: String },
}
struct TensorType {
    logical_element: ElementType,
    stored: QuantEncoding,
    dims: Vec<SymbolicDim>,
    layout: LogicalLayout,
}
struct ValueRef { id: ValueId, ty: TensorType }
enum StateDomain {
    KvAttention, RecurrentGdn, RecurrentKda,
    SparseIndexer, CompressedAttention,
    TokenNgram, ModalityPositions, SamplerRng, DraftModel
}
struct StateRef { id: StateId, domain: StateDomain, version: u64 }
enum EffectKind {
    ReadState, Transition, Snapshot, Restore, Fork,
    CommitPrefix, AbortUncommitted, ConsumeRng,
    Transfer, Allreduce, Barrier
}
struct OpNode {
    id: OpId,
    semantic_op: OperatorId,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    state_inputs: Vec<StateRef>,
    state_outputs: Vec<StateRef>,
    effects: Vec<EffectKind>,
    numerical_contract: NumericContractId,
}
```

Code above is a **design sketch**; Rust enum representation/serialization and whether effect tokens are affine/linear is to be proven in Phase P04. Do not rely on this pseudo API as release ABI.

### Shape checking

Use symbolic dimensions with explicit constraints: e.g. head * head_dim, tokens ≤ context capacity, quant blocks divide padded storage stride (while logical stride differs), expert ID < n_experts, top_k <= n_experts, cache offset within allocated range. Checked u64 arithmetic for storage byte sizes; no `as usize` truncation on untrusted metadata. Valid constraints must be rechecked for dynamic batch shape and graph replay, not just compile time.

### Logical versus physical tensor

A TensorSpec describes mathematical shape/dtype and checkpoint quant. A DeviceTensor describes active device pointer, strides, packed layout and validated conversion provenance. No generic operator may treat compressed/latent attention KV as full KV or BF16 n-gram row as a Q8 dense matrix.

## 4. Stateful effects and exactly-once frontier semantics

Define `StateVersion(session, position, epoch)` for each domain. A transition consumes previous state and emits next. Multiple independent state domains may transition concurrently if the graph declares them independent, but externally visible `SessionCommit(position)` waits until **all** required domains are ready.

A speculative target verify produces `ProvisionalStates[0..D]`, one after every proposed token. Verification samples/chooses acceptance `a`. Commit means state at position `p+a` becomes current, other versions are released; rejected tail cannot affect KV, GDN, KDA, sparse indexer, n-gram, modality history or sampler RNG. Exact sampling correction needs its own RNG bookkeeping.

**State tokens are ordered resource effects.** Compiler may fuse several transitions into one kernel but must preserve logical snapshot candidates and rollback. A pure operator may be hoisted only across effects if its input values and relevant mutable state are independent.

### Effect rule examples

- `GdnStep(x, state_p) -> (y, state_p+1)`: state_p+1 must not be consumed before step finishes.
- `Snapshot(state_p) -> snap_p`: snapshot references immutably frozen state; later mutations require COW or copied affected regions.
- `Fork(snap_p) -> (stateA_p, stateB_p)`: either branch changes state without corrupting sibling.
- `VerifyDraft(x[p..p+k], state_p) -> provisional[1..k]`: no visible frontier advance yet.
- `CommitAccepted(a, provisional) -> state_p+a`: atomically publish chosen state version and release rejects.
- `Abort`: if failure before commit, all branch writes discarded or canonical state rebuilt; no false frontier.
- `RemoteCollective`: distributed ranks agree on same token/graph identity, otherwise fail-closed; no partial publish.

## 5. Typed operator families

**FullDenseAttention:** position/cached KvLayout, causal mask, Q/K/V scaling, RoPE config, head relation, sliding vs full, optional sink; verify numeric precision/softmax.

**QwenSparseAttention:** indexer compressed/pool state, block selection width/index sorting/causal mask and recurrent/cache position; separate from generic top-k which may have different ties/precision.

**NoPeSparseMla:** compressed latent KV, query absorbed vs expanded variants, no positional rotation in NoPE part, DSA indexer state, exact selected set semantics.

**GdnOperator:** gate variant and dtype, conv size, recurrent matrices, per-head dimensions and transition/snapshot formulas; prefill chunk rule and step rule are semantically equivalent within tolerance, not necessarily bit identical.

**KdaOperator:** per-channel/fine-grained gate and lower-bound parameter, conv, recurrence; cannot lower to GDN scalar decay.

**DeepSeekCompressedAttention:** short raw window, compress ratios (4/128 depending on model), indexer for selected compressed layers, causal and latency/state rules; not same as QSA.

**MoERoute/Group/Reduce:** routing function/scores, selected IDs, per-token weights, shared expert, summation precision/order; grouped backend may reorder rows but not route identities.

**HyperConnections:** gated residual vs mHC Sinkhorn with shape and iteration parameters; special numerical behavior.

**Ngram/EngramLookup:** history state, map/hash, row retrieval, output dtype/scales; disk I/O is an effectful data dependency, must error on stale/truncated reads.

**Speculation:** separate drafts, target execution, verification, rollback and RNG. Draft is not a semantic replacement for target logits.

## 6. Numerical policy / legality

Introduce explicit `NumericalContract`:
- `Reference`: canonical tested precision/order as model artifact demands;
- `ValidatedTolerant`: bounded max/relative/RMS error per op with full-model quality controls;
- `ExactReplay`: same graph/plan seed and schedule expected bitwise deterministic where implementation supports;
- `ApproximateOptIn`: changes quantization/selected attention set or sampling, user visible and separately evaluated.

Do not use unqualified terms `exact` or `equivalent` without specifying which level. FP16/BF16/F32/fused MMA may differ due reduction order; a kernel can produce matching next token for short prompts and still fail long-context quality.

Legalization rule examples:
- Fuse RMSNorm and activation quantization if intermediate rounding/numeric tolerance matches policy.
- Fuse router+MoE gather without skipping required indexer updates.
- Skip final unused projections only if no state or logits depend on those values.
- Repack Q4/GGUF weights losslessly for device but record mapping.
- Reject lowering mHC to ordinary gated residual.
- Reject MLAs expanded/absorbed variant if output precision drift fails target policy.
- Reject change of sparse top-k or attention mask unless explicit policy switch and quality proof.

## 7. Pattern matching and region formation

Candidates should match exact operator subgraphs plus state effects. A model with 48 layers may have repeated GDN/MoE patterns, but residual and sparse layer variants differ. Region formation groups operations around state-safe cut points. Fusion cost estimates must include:
- bytes loaded/stored and quant decomp;
- matrix instruction efficiency vs shape;
- launches/host sync;
- VGPR/LDS/reg pressure/occupancy and code size;
- intermediate scratch and graph lifetime;
- side streams and optional IO;
- numerical reordering and quality constraints.

Support whole-layer specialization without duplicating model structure: `RegionKey` covers model semantic operation pattern + dimensions + quant + backend + numeric policy. A specialized `gfx1151` Qwen decode island can replace several SIR ops; fallback returns to canonical region boundary where tensor and state formats are converted by a validated adapter.

Do not require exact same raw buffer layout at region cut points; implement explicit `LayoutAdapter` with measured conversion cost and state version match. Compile legal conversions ahead of time.

## 8. Device plan and persistent variant management

Suggested plan cache key:
```text
semantic_graph_hash
weights_artifact_hash + quant_manifest_hash
operator/region schema version
device vendor/arch and GPU count/topology
compiler backend and build SHA + driver/runtime version
numerical/sampling policy
context capacity tier + batch width band + prefill/verify mode
packed layout ID, state schema version and kernel capability set
```

Plan variant shapes might include `decode1`, `decode2-8`, `decode9-32`, `verify2-8`, `prefill<=256`, `prefill<=2048`, `prefill>2048`, with exact choices made by profiling, not canonical constants. Evict obsolete JIT variants and bound disk/in-memory cache size. JIT compilation must not occur synchronously inside an interactive token hot loop.

### Runtime plan selection

A `PlanSelector` operates on model identity, actual device resources, number of pending token rows, context frontier, currently leased buffers, numerical policy and optional expert cache statistics. It picks a previously validated plan and requests ResourceArbiter reservations. If a candidate cannot reserve scratch, choose a smaller legal variant before rejecting request. A runtime must never lower the quantization unilaterally to fit.

A device plan includes streams, event predicates, buffer lifetimes, launch order and rollback barriers; the server thread does not dynamically iterate semantic operations to execute them.

## 9. Cross-compilation and specialization choices

Rust `KernelSpec` should be independent of CubeCL framework APIs; CubeCL is a candidate compiler. A target extension may define architecture intrinsics and validate generated ISA. CPU baseline remains native Rust. Vendor runtime calls are packaged as `DeviceBackend` FFI but not an inference-engine fork. Metal/CUDA/ROCm/Vulkan capabilities can be unequal; expose unsupported region with fallback or explicit load-time error.

For HIP devices, graph capture support is a backend capability and may be available even if a specific source project (DwarfStar ROCm) executes eagerly. Do not encode an assumption `all ROCm cannot capture`.

## 10. Independent verifier and proof harness

Every optimization can be tested by executing:
1. SIR on canonical CPU/reference (synthetic and actual model).
2. EIR generic plan on same inputs.
3. Specialized region on same state/weights/quant.
4. State transition logs and full-vocabulary logits at each frontier.
5. Forced fallback and interrupted execution.
6. Benchmark isolated region and full HTTP request.

Store shape/quant and state traces as versioned fixtures (without private model weights/secret prompts). Test:
- invalid dimensions, unmatched quant blocks, ties and sparse masks;
- token branches, snapshots, accepted prefix lengths including zero/full;
- context and chunk boundaries;
- output differences at nonzero sampling temperature;
- recovery after IO/device error;
- multi-session concurrency and branch ABA hazards.

## 11. Practical model integration walkthrough

A GLM KDA block is declared as `RMSNorm → KDA(state) → mHC transition → router/top8 + shared expert → gated outputs`. Reference may execute separate tensors; optimizer recognizes model geometry, selects `KDA(gfx1151,decode1)`, `mHCfused`, and `MoEGroupedQ2`. A resident-path graph launches without host routing; on an expert miss an asynchronous IO stage produces a fresh validated slot generation before kernel continues. State is provisional until step ends. If ROCm cannot support a special `KdaOp` kernel, fallback reference KDA only at a legal region boundary while other operations remain specialized.

A Llama dense transformer instead lowers to optimized attention, FFN and KV. It does not pay for MoE expert-cache or GDN state machinery.

## 12. Performance and maintenance controls

IR canonicalization deterministic, region hashing stable within schema, reproducible compiler logs, readable plan dumps; all plan changes logged with reasons and benchmarks. No feature flag grows into an unbounded parallel semantic implementation. Keep kernels documented next to source and a machine-checkable capability catalog to prevent drift. Use explicit typed Rust errors rather than fallback-by-panicking. Versioned schema migrations required for persistent snapshots.

## 13. Pending design proof

- Formal effect/alias model expressivity and verification cost.
- Scalable symbolic-shape specialization without compile explosion.
- Ownership of state under mixed batched speculative execution.
- Rust GPU codegen quality and custom intrinsic path on `gfx1151`.
- Typed state layout conversion cost and when seamless fallback is impossible mid-token.
- Global plan optimization versus tractable local heuristics (choose cost model only when empirically supported).

These are P01/P04/P05/P10/P15/P16 research tasks, not pre-decided solutions.

## 2026-10-10 integrated v2 addendum — complete compiler partition

**Supersedes the three-stage-only interpretation where it omits asynchronous resource scheduling.** The authoritative design has SIR (typed model SSA + explicit state effects), RIR (legal pure-region fusion/layout/quant alternatives), **KIR** (target kernel codegen/registry branch), **CIR** (explicit async buffers, stream/timepoints, host/SSD/remote operations, graph capture legality), then immutable **ExecutableBundle**. Read `ASYNC_RESOURCE_COMMAND_IR.md`, `ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md`, and the comparative compiler research before implementing a compiler pass.

The previous conceptual `ModelDefinition::build_graph` remains; `ModelSpec::kernel_plan(hw)` from the research program is **not** a semantic-model responsibility. Use `KernelProvider::candidates(region, device, quant, numeric_policy)` and bind selected candidates to CIR with memory/event validation. The pure hardware/resource plan (`PlanSource`) and per-cycle `StepPlan` are distinct from the kernel registry; no monolithic `ExecutionPlan` may interpret model math during decode.

**State-effect legality:** model matrices, recurrent state, compressed attention indexer, sampler RNG, grammar and I/O token-history effects have separate write/snapshot/commit rules. E-graph/search techniques inspired by bounded graph-search work are permitted only in bounded **pure** regions with verified equivalence and compile budgets. Algebraic rewrites may not move a KDA/GDN state transition, token hash or tentative sampler RNG across commit. Tensor padding/quant packing must carry semantic shape and source checksum; numerical policy gates reassociation and state-compatible fallback.

**An important new acceptance:** create a synthetic 3-state-domain model (full KV, GDN, sparse indexer) and prove SIR→RIR→KIR+CIR eager/capture and CPU reference produce matching logits/frontiers after arbitrary `Fork`, `CommitAccepted(n)`, `Abort`, `Recompute` and layout fallback. Verify a poisoned buffer/stale event is rejected. On the first real dense model, compare compilation/dispatch CPU overhead with hard-coded call sequence: SIR/RIR/CIR must be eliminated from the normal token hot loop. Preserve a human-readable plan dump/validation errors for debugging, but no per-token optimizer.

**Compiler evolution:** version semantic op schemas independently of device kernels and serialized snapshot layouts; a new attention math family introduces CPU op+state contract, a new GPU introduces kernel provider+caps, and a new SSD tier introduces a CIR command+resource kind, not an HTTP or model-loader rewrite. Toolchain alternatives (portable Rust GPU DSL, Rust HIP target, CUDA tile/SIMT compiler, SPIR-V Rust provider) share one capability contract instead of multiplying IR dialects.


## 2026-10-12 integration — model-independent split-output projections and task stages

Add RIR's candidate `SplitProjectionRows`: for a pure matrix projection, divide output row map into disjoint, covering subsets assigned to GPU/NPU/CPU kernels, then explicitly assemble original tensor values at a first-use barrier. The model recipe/SIR **never names XDNA2, gfx1151, XRT or fabric-clock settings**; the numerical operator and GDN/QSA/shared expert effects remain unchanged. A transformation is legal only if every source/output row, biases, per-tensor scales and quant block geometry are covered; downstream consumers can run early only if their inputs do not depend on missing rows. A cross-row normalization or router top-k cannot be incorrectly split before the reduction. Test an equivalent unsplit reference for full logits and all state-domain positions.

The first concrete case is the `mikealanni/gufo` fork: eligible 4096-token-multiple prefill does GPU+NPU row-slice GEMMs for GDN **z**, attention Q/gate and shared expert gate/up. Its block-float `v8bfp16ebs8` is a **compute encoding**, not a source checkpoint Q8_0 type; a region may be `ValidatedTolerant` but not magically `Reference`. RIR/KIR candidate must declare encode/decode and complete quality policy; CIR supplies XRT import/events/first-use join, pinned leases and failure fence. Whole-model fallback GPU-only when unsupported or hardware safety unqualified.

**Beyond text:** SIR may represent reusable mathematical stages (encoder, projector, causal decoder, acoustic/vocoder, diffusion/flow denoiser, media decoder and pooling) but does not force all tasks into one `next_token` loop. A higher-level typed `TaskProgram` declares task's stage/control DAG, iteration frontier and stream output contract. An LM token loop, ASR recognition, TTS synthesis and image/video denoiser use different state/control graphs on the *same* SIR/RIR/KIR/CIR backend. Task stages remain capabilities until independently implemented and quality-qualified. See [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

## v4 refinement — two durable representations and explicit legality

Keep exactly **one canonical SemanticProgram and one compiled ExecutableBundle** with versioned identities. RIR is a **transient legal-region optimization view**, KIR a **kernel candidate/codegen view**, and CIR the **typed command/resource DAG within the compiled executable**; a separate serialized interpreter/runtime for each dialect is not required and would create maintenance and hot-path costs. A model adapter emits semantic equations/effects and never imports hardware/kernel source decisions.

`ShapeEnv` verifies bounded dynamic dimensions, fixed rank, ragged route offsets, strides, divisibility, batch/context and device/precision guard *before dispatch*; dynamic expert IDs are device data. An unknown shape selects prevalidated generic kernel rather than per-token recompilation. Closed fused model/layer candidate has the same externally observable value/state effects and numerical constraints as portable region. A compiler pass cannot publish logical TaskSession state: it can produce only **PreparedDomainVersion** and resource readiness. 

Future `GradientProgram` may define adjoint/optimizer effects, but is **not** mandatory compiler/autodiff machinery for inference-first MVP. Accept a synthetic future compatibility test while prioritizing real model + GPU direct-kernel vertical slice.
