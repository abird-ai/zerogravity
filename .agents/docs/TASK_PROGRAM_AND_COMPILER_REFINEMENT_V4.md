# v4 TaskProgram and compiler contracts — high throughput with a small semantic core

**Normative design · 2026-10-12 · not implemented.** This refines `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md` and `COMPOSITION_AND_EXTENSION_CONTRACTS.md` to avoid an overly broad task DSL and four separately serialized IR ecosystems. Companion physical execution/state ownership specification: `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`.

## 1. One reusable semantic representation, one physical executable plan

**Mandatory durable or stable in-memory representation #1: `SemanticProgram` (SIR).** Contains strongly typed mathematical values and minimal control-relevant effect annotations. It describes *what math and state transitions mean*, not how a GPU kernel dispatches. It is canonical for CPU reference/fallback, testing and legality checking. A semantic model can be assembled from versioned `SemanticOp` schemas and validated checkpoint tensor roles.

**Mandatory compiled representation #2: `ExecutableBundle`.** A closed set of prepared physical regions and command/event templates with guarded shape variants, kernel handles, buffer layouts/leases and cross-device synchronization contracts. This is the data consumed by the native DeviceWorker fast path. It may have cached serialized **metadata/source hashes** but never persisted raw pointers/events/driver handles. Runtime may rebuild after compiler/driver change.

**Only compiler-internal views, not mandatory new formats/runtimes**:
- `RIR` (Region view): pure-subgraph fusion, output-row partitions, layout/quant tiling alternatives and effect/barrier/fallback cuts.
- `KIR` (Kernel candidate view): target ISA/tile/register/precision/shape capability, source compiler implementation and numerical proof.
- `CIR` (Command view): the resource/timepoint DAG **inside the executable plan**; not an independent general-purpose host workflow scheduler or semantic-state publishing system.

This preserves **SIR→RIR/KIR→CIR→Bundle** as meaningful mental stages, but **does not force four separate serialized dialects or trait object graphs in code**. Implement a small `RegionBuilder`, `KernelRegistry` and `CommandPlanBuilder`, each reducible to one Rust module initially. The public model/task program interface needn't expose intermediate compiler implementation views.

## 2. Two levels of mathematical operations

### SemanticOp — source mathematical identity

Retain high-level ops where the distinct formula and state truly matter (full/sliding/sparse/latent attention, GDN/KDA/delta recurrence, MoE router/shared experts, 4-stream residual/hyperconnections, compressed/indexed history, learned n-gram lookup, tokenizer/media encoders and diffusion/flow/vocoder where defined). These carry **state effect declarations, numeric and position contracts** and a Rust reference implementation.

A model adapter may create new op graph from existing semantic operators without copying serving/runtime code. A novel trained attention or SSM formula gets a new exact semantic op with reference oracle, not reuse of a vaguely similar kernel.

### Pure tensor subset — optimization internals

Inside a mathematically closed semantic region, lower to typed `Add, Mul, MatMul, Dot, Norm, Gather, Transpose, Reduce, Quantize, Pack`, etc. with shape/quant/alias constraints. The compiler is free to fuse/tiling/specialize/reorder **only pure transformations allowed by numerical policy**. This avoids having to represent complex recurrence as one opaque GPU-only function *or* erasing its state semantics into 100 untyped primitive nodes too early.

One region may own entire fused model layer if exact inputs/outputs, state effects, memory/resource budget and comparable reference execution are available. This is our extension to beat specialized hand-tuned pipelines without having a separate model engine.

## 3. Value/dimension contracts and dynamic shape guards

```rust
struct SymbolicDim { id: DimId, min: u32, max: Option<u32>, divisible_by: u32 }
enum Dimension { Static(u32), Symbol(SymbolicDim) }
struct TensorType {
    rank: u8,
    dims: Vec<Dimension>,                  // only bounded rank initially
    dtype: LogicalDType,
    quant: StoredQuantType,
    positions: Option<PositionSpec>,
}
struct RuntimeShapeGuard {
    allowed: ShapePredicate,               // width, stride, alignment, ragged offsets, batch
    numeric: NumericContractId,
    effect_schema: StateEffectSchemaId,
    layout: PhysicalLayoutId,
}
```

These are design sketches. A given kernel may require `M % 4096 == 0`, supported K/N tile, `stride % alignment`, BF16 vs quant type and a particular device power profile. **ShapeEnv** binds actual input tokens/rows/stride at planning time and the kernel is submitted only if guard passes, otherwise a prepared generic region executes. In dynamic sparse MoE, router-selected expert IDs remain device-side and are **data** within a shape bounded ragged route layout, not a new compiled kernel per expert ID. Batch group changes can pick a prepared shape band.

**Avoid variant explosions:** only benchmarked hot bucket classes (decode1, verify-small, prefill-tiled, encoder-large, denoiser step with static shape/kv bands) are compiled/cached eagerly. On cold shape miss, use portable candidate with optional deferred compilation outside interactive critical path. LRU code cache and OS memory budget bound number of code/graph variants. Do not silently truncate a prompt or pad beyond allocated/reserved bounds merely to match a shape.

## 4. Effects and legal speculative execution

A `SemanticOp` reads/writes typed `EffectDomain` (attention cache, GDN recurrence, sparse/compressed indexer, token-history/grammar/RNG, media/timestep, optional parameter/optimizer state) with versioned input/output handles. Pure ops can be reordered, fused, recomputed or moved to different devices if their numerical and alias contracts allow. Stateful ops cannot be reordered across a causal boundary without a *validated state-effect graph transformation*.

**Compiler scheduling speculation** (execute math early) differs from **token prediction speculation** (propose future tokens and run authoritative target verifier). Both need explicit rollback boundaries, but only token speculation must preserve target output distribution under exact mode. Lossless device packing is different from temporary lossy compute representation; silent top-k reduction, changed sparse mask and changed diffusion schedule are model-changing approximations even when output “looks plausible”.

A compiled region is closed under all external dependencies. It may contain mutable effects if it returns prepared versioned candidate state plus readiness events; it cannot publish new TaskSession state or output by itself. A whole-model fused region must still implement the canonical state and output boundary.

## 5. Minimal task driver, not a workflow DSL

The previous v3 task graph defined a broad `TaskNode::{Invoke,Sequence,Parallel,Select,Iterate,Window,VerifyAndCommit,Emit,ExternalAction}`. This is far too close to a general workflow/orchestration language for a lean model runtime. It also risks interpreting a dynamic graph per token and bringing permissioned tool execution into the GPU scheduler.

**Use one small trusted `TaskDriver` abstraction** for AI *computation control*:

```rust
trait TaskDriver {
    fn validate(&self, req: &TaskRequest) -> Result<TaskSpec>;
    fn start(&self, req: TaskRequest, state: &TaskSnapshot) -> Result<TaskCursor>;
    fn next_stage(&self, cursor: &TaskCursor, completed: Option<&StageCompletion>)
        -> Result<StageIntent>;
    fn prepare_commit(&self, cursor: &TaskCursor, result: &ProvisionalResult)
        -> Result<TaskCommitCandidate>;
    fn after_commit(&self, root: &StateRoot, output: &CommittedOutput)
        -> Result<TaskProgress>;
}
enum StageIntent {
    Run(CompiledStageId, StageInputs),
    ParallelKnown(Vec<CompiledStageId>, JoinSpec),
    YieldForInput(InputWaitId),
    EmitCommitted(OutputRef),
    Done(FinishReason),
}
```

These signatures are **illustrative**: call `next_stage` only at true stage/step boundaries; inner tensor ops and optional repeated decode/denoise microiterations can execute entirely within a **device-native StepProgram** with loop bounds and provisional state. `ParallelKnown` is bounded/compile-time validated; arbitrary user-supplied callback/agent tool code is not a permitted `StageIntent`.

**No general `ExternalAction` inside core.** Application-level orchestration (tool calling, retrieval, RAG, multi-agent workflows, external APIs) is an independent permissioned service that produces typed input and consumes typed output of TaskPrograms. It may combine many inference tasks, but it does not gain OS/network authority from a model kernel or TaskProgram stage.

### Supported task drivers and their own frontier

- `LanguageGeneration`: token/prefix/target-verifier loop and grammar/sampler RNG; output tokens in exact committed order.
- `EncodePoolScore`: encoder/embeddings/cross-encoder plus model-specific pooling/normalization; output vector/score.
- `SpeechRecognize`: audio chunk/window preprocess and encoder, model-specific transcript decoder/CTC, output transcripts/timecodes.
- `SpeechSynthesize`: text conditioning/acoustic loop + vocoder; output complete verified PCM/codec frames with voice seed/style state.
- `DiffusionGenerate`: noise/timestep/denoise/flow loop, decode when done, output image/latent according to request.
- `VideoMedia`: temporal frame loop and synchronized audio/video encoder/decode, output complete ordered media chunks.
- `TrainStep` **optional future**: forward/backward/optimizer effect stage; no core training engine or general autodiff interpreter required for initial inference product.

Each task uses the same SIR/RIR/KIR/CIR compilation and DeviceWorkers but exposes its own `WorkUnit`, `TaskFrontier`, numeric and validation/failure contract. `TaskKind` is not a giant `match` in one serving GPU loop; task drivers are built-in providers selected at request admission or model load, and dispatch through compiled stage handles.

## 6. Interface compatibility and minimum provider set

**Exactly three expansion families required initially:**
1. **Semantic providers:** `ModelDefinition`, `SemanticOpRegistry` and `StateDomainSchema` paired with reference tests. A new model mostly adds an adapter/recipe; a novel mathematical family adds operator/reference/codec.
2. **Execution providers:** `KernelProvider` (pure per-region shape/capability/precision) and `DeviceBackend` (allocation/stream/event/copy). A device driver can support only one subset of operations and leave others portable.
3. **Task drivers:** model/task stage/iteration/stream semantics separate from tensor math and GPU devices.

**ResourceArbiter, TaskCoordinator and output service** are common *owners*, **not plugins whose object instances grow per model family**. Quant formats/codecs, external storage/IO adapters and model loaders can be plain enum/traits inside these three families. Do not create an abstract factory for every low-level struct.

Proof: a new 1-off GEMM kernel can be registered against an existing region without modifying TaskDriver, HTTP, state serializer, device storage and model source. A new task can call existing semantic operators and execution bundles without changing KV cache shape or adding model-name cases to GPU workers.

## 7. Compiled StepProgram and host/device scheduling cost

A `StepProgram` is an executable *mode-specific region/control island*, not a new semantic task. It includes:
- compiled device kernel/graph/loop handle, persistent batch rows and local resource slot map;
- parameter buffers (input lengths, token IDs, position/attention mask, expert pointer generations, stage counters, stop flags), each with explicit host/device producer and consumer epoch;
- shape and numeric guard for prefill vs decoder vs speculative verifier vs encoder/vocoder/denoiser;
- device-side routing, copying and sampling where validated; first-use waits on external IO/NPU/peer readiness without unconditional CPU synchronization;
- provisional output and state-root candidate ready record, bounds/max iterations and failure handling.

`TaskCoordinator` can prepare the next independent StepPlan while the GPU runs the current one, but cannot speculatively mutate the authoritative host root. CPU-GPU overlap is a benchmarked option with verified double buffering and race-free transfer; no rule promises always 2 in-flight steps if state/tool/sampling dependencies disallow it.

Hot-path budget measured: host scheduling instructions, GPU idle/wait, kernel launches, CPU↔GPU transfers, graph pointer writes and memory allocation count per step, plus throughput/TTFT/P95 under model/task match. **Not** a fictitious universal microsecond guarantee. A device-native loop not available for a particular grammar/operator must select a correct host-loop plan with ModeReceipt.

## 8. Region and whole-model specialized kernels

The core supports a high-performance specializer that takes **the same exact SemanticRegion** and returns a `KernelCandidate` for the entire fused sequence/layer/model. At compile time:
1. validate source model/weights, state effects, inputs/outputs and numeric/position/quant;
2. compare candidate vs canonical reference for full-logit/state/quality and deterministic/sampling where possible;
3. check target ISA/shape/thread/capture/virtual/physical budget and fallback at region boundary;
4. return compiled opaque code/graph handle and bounded parameter slots to CIR with resource lifetime/failure events;
5. ensure runtime TaskSession/TaskDriver/ResourceArbiter/ModelStore/Service remain common and untouched.

An extreme model-specific kernel can be handwritten/shape-specific **without becoming a model-specific inference engine**. The key test is whether it can be disabled and the generic model still generates correct outputs. A closed whole-model region can be large but must have compatible effect/snapshot state boundaries, or only be eligible for entire-session mode switching/replay.

## 9. Optional gradient/parameter update future without training bloat

“All AI execution” plausibly includes online learning, adapter fine-tuning and distributed training. The runtime should **reserve a legal upgrade path** but should **not** make training a dependency for the inference-first core.

Future `TrainStep` driver and `GradientProgram` may add:
- differentiability/adjoints for a subset of registered pure ops, or independently defined backward semantic operators;
- activation checkpoint/recompute and memory planner variants, dropout RNG and loss functions;
- `GradientAccumulator`, optimizer state (`Adam/SGD` or model-specific) and `ParameterVersion` updates;
- device collective/fault consistency, precision/loss-scaling and reproducibility under sharding;
- a **separate mutable parameter publication boundary**: inference sessions remain pinned to immutable model/adapter version until explicit cutover, preventing weight updates from silently corrupting cache, draft or compiled bundle identity.

Nothing in the original `SemanticProgram` should assume weights can never *be logically updated* (a model parameter is still a logical tensor), but standard inference execution takes an immutable versioned parameter view. Training-like tasks may become supported by these same operator/resource/device primitives **after** their own numerical/state/fault tests. This is a future capability seam, not a promise to rival specialized training compilers immediately.

## 10. Hardware/format-independent optimization search and cost

**Three timescales:**
- **On model load:** verify math and state; compile required baseline shape modes and reference fallback; reserve mandatory weights/state and obtain optional available hardware capabilities.
- **Offline/explicit tuning:** search bounded pure regions, measure kernel/register/critical-path vs full TaskProgram and quality; validate source/ISA/compiler/capture/device program; persist candidate evidence keyed by shape/regime/precision.
- **Online service:** choose only already qualified variants based on current stage/shape/memory/fairness/power; update cache/prefetch/speculated length with hysteresis at legal epochs. No equality-saturation/JIT in the inner decoding/denoise GPU loop.

Promotion requires matching checkpoint+quant, task output type, sample/attention state, concurrency, context, driver/firmware and physical memory/power; no optimization is universally optimal. A fast NPU projection requires pinned BO and event/clock safety and can lose to GPU-only at small shapes; a grouped routed expert kernel can worsen latency due to cache thrash despite reduced FLOPs.

**Portfolio of cost objectives:** TTFT/P95, committed output units/s, total task completion, device uptime, physical memory, energy and functional task quality; quality/safety are constraints, not scores hidden behind one tokens-per-second metric.

## 11. Evolving interop without external engine dependence

Model/task/device/quant representations may import **verified external artifact formats** if license and semantic equivalence are known. Do **not** require serializing an upstream compiler's internal IR to run. Any interoperability layer is a **lowerer/adapter** into our `SemanticProgram` after checking known operation semantics, shape rank/dynamism, custom/unknown op and side-effect tokens. Importing an unsupported custom op does not magically make it executable.

Versioned compatibility plan:
- semantic op schema and model program epoch;
- task driver protocol/frontier schema;
- resource/state snapshot schema and test vectors (persistent);
- executable bundle and backend kernel cache (expendable, invalidated on upgrades);
- artifact/model/quant/adapter bytes and logical weights;
- numerics/shape/provenance/hardware safety qualification;
- API stream output framing and replay/ACK (explicit optional).

Keep the number of **persistent** standards small. A public stable C ABI or cross-platform plug-in protocol should wait until proof and user decision; static linked Rust providers first.

## 12. Acceptance experiments proving high performance and smallness

1. Add novel model with no new pure operator → only ModelDefinition/fixture, no scheduler or kernel dispatch hack.
2. Add recurrent operator with new state schema → CPU oracle and one generic region; GPU provider optional, runtime/session output unchanged.
3. Add a target-only fused whole layer → same SemanticRegion and state effects, passes quality and can be disabled in one flag without losing model support.
4. Add NPU partial GEMM row slice or dGPU+iGPU expert split → old ModelDefinition unchanged, CIR correctness and unsafe-power fallback.
5. Add a non-language task driver (mock denoiser or encoder) with different committed frontier, stage scheduling and output, no general agent workflow engine.
6. Add synthetic `TrainStep` operator version and parameter update in reference mode while LLM inference stays pinned to old immutable parameter version.
7. Time one real model/shape in reference vs optimized whole region vs compiled direct island, measured CPU sync/instructions, allocation count, P95 and physical memory. If compiler scaffolding introduces unavoidable host bottleneck, simplify.
8. Bound code generation cache sizes and cold-shape fallback, with two compiled modes and several shape variants; never compile for every exact context token count.
9. Confirm source quant logical identity remains separate from low precision kernel compute and any derived requant checkpoint; exact/tolerant/approximate mode rules.
10. Confirm no core trait/module/type is named after an upstream project; prior-art licenses/research remain explicit source records.

**Release bar:** model/task correctness and safety come first; architectural modularity must be demonstrated with a *measured change radius*, and specialized throughput improvement must be shown on actual target hardware, not inferred from a nice compiler diagram.

## v4.1 addendum — large compiled graphs without a new IR

The `KernelProvider` of earlier sketches is generalized to one `RegionExecutionProvider`: choose a mathematically closed `SemanticRegion` and compile as a kernel, captured command graph, ahead-of-time whole subgraph or static fabric dataflow program. **RIR/KIR remain transient compiler views, and CIR remains inside the same ExecutableBundle**. We do **not** add a third durable TPU/wafer IR. A vendor graph lowering is a temporary *interchange* representation, permitted only for the subset of exact semantic/state/quant operations that that compiler understands.

Region-cut optimization must account for the hardware's **minimum efficient program granularity**: for graph/static hardware, a whole attention block, model stage or complete verified model may amortize compilation/loading/transfer overhead; slicing into hundreds of tiny host-submitted GEMMs can erase any advantage. Conversely a GPU fast tile can stay an isolated Rust-native kernel. TaskDriver owns **stage loops/frontier** and must not be replaced by a vendor's independent chat server. Compiler may lower a closed repeated step to a device-native program when explicit control/shape/state effects are supported.

**Dynamic shape reality:** graph-only devices can recompile for new batch/context/ragged layouts; cache a bounded set of guarded shape bands, compile off the request-critical path when possible, and fall back to a legal portable region or reject unsupported device context. Sparse route IDs remain runtime data where graph/compiler permits dynamic indexing. No conversion of a stateful unknown op to semantically different approximate attention just to reach an accelerator.

**Future training seam:** a gradient/optimizer TaskDriver might lower large pure forward/backward graphs and model parameter sharding to an AOT accelerator, but inference sessions must pin an immutable ParameterVersion. It is future-only and does not import external training framework into the inference core.

**Program capability and source policy** are separate: project Rust SemanticProgram and host execution can be independent of a closed vendor compiler, but that does not constitute a Rust-authored low-level device kernel/compiler. Do not claim true TPU/wafer support without authorized SDK, backend implementation, state/failure tests and physical reference/model quality. Full contract: [accelerator execution families](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## v4.2 — networked execution placement is a compiled physical plan, not a new TaskDriver

TaskDriver owns **what happens next** in its text/ASR/TTS/image/video/encoder task and source StateRoot; the compiler and topology planner decide **where** the closed stage, mathematical layer, expert or attention fragment executes and where its immutable weights, mutable state and cache copies reside. None of these location choices appear as model-name rules or new TaskDrivers. A `RegionExecutionProvider` can prepare local kernel, remote whole-layer, compiled multi-chip/fabric or remote attention beside state. TransferPlan, rank/peer events and resource reservations become CIR commands **inside the same ExecutableBundle**. SIR math/effect schema and exact task output frontier unchanged.

**Legal cuts:** layer boundaries with explicit activation/state version; attention vs FFN if source op/state exact and communication cost acceptable; sequence-sharded attention only with operator-specific verified normalized partial softmax merge. GDN/KDA recurrent/selected sparse/history can't inherit partition algebra from ordinary attention without new reference equations. P/D is a **phase transition** requiring full hybrid session state transfer, not just prompt tokens or text prefix.

Compiled PlacementPlan can be revised at a safe committed TaskFrontier after reserve/transfer/fence/recompute, not during a speculative unfinished device/remote group. A planner maintains low-cost static baseline for one host and bounded nearby measured candidates for private cluster. **No dynamic distributed search/interpreter per token** and no new semantic IR. See [placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md).
