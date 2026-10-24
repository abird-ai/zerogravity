# Canonical Sourcebook — volume 3

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/docs/HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md

Original UTF-8 bytes: 18090; SHA256 `0c5731fb275b8ea560ede89602c6883d35d7778829ce0f27878fe9e192d3c134`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — Heterogeneous prefill and retained replay contract

**15 October 2026 · U-048/U-049 · NORMATIVE DESIGN REFINEMENT; not implemented
capability or measured performance.** Architecture ownership, qualification
requirements and **40 PLANNED acceptance gates** live in the
[acceptance plan](../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md).

**Authority:** this additive contract refines existing v4.3 SemanticProgram →
ExecutableBundle, RegionExecutionProvider, CIR effects, v5.1 ModelAssembly and the
joint serving optimizer. There are still two enduring representations, one Rust
mathematical engine, one physical resource arbiter and one TaskCoordinator that
publishes StateRoot. Providers share this engine and its existing representations.
Existing source-exact H5/H6 numerical and physical gates remain unchanged.

## 1. Required capability and honest status

The design supports qualified native packed-weight/ISA kernels, resident/fused
execution,
driver graphs and attested retained queue replay, target-authoritative speculation,
cooperative NPU/GPU prefill, state/prefix/SSD tiers, model-specific long-context
attention/KV, multi-GPU, continuous lanes, model/quant tooling and task/API surfaces.
A model may combine a qualified NPU prefill phase with GPU MTP decode; this does
not imply that an NPU kernel, EP collective and PM4 tape can run interchangeably.

Each capability has separate **expressible / feasible / source-functional /
state-safe / performance-qualified / production-qualified** evidence, bound to:

`source assembly × task/phase × numeric regime × shape/context × physical topology
× compiler/driver/firmware/program × state schema × requested feature composition`.

The live H6 implementation is much narrower: original small Qwen CPU/hybrid
regions on Renoir/RTX3060 Vulkan. It has no whole resident graph, NPU, native large
MoE, production HTTP serving or competitive-speed qualification. Extensibility
permits a native implementation without another engine; it does not establish one
exists.

## 2. Existing boundaries carry every optimization

| Mechanism | Owner and representation | Constraint |
|---|---|---|
| model operators, causal recurrence/KV/QSA/PLE, task solver | SemanticProgram + trusted TaskDriver | exact source formulas, state domains and legal task frontier |
| projection fusion, prompt-bank plan, expert compaction | transient region planning within ExecutableBundle | dependency/effects/shape/numeric proof; no per-token AST walk |
| packed kernels, WMMA/LDS/ISA, Rust AIE programs | Kernel/RegionExecutionProvider candidates | immutable source/derived encoding, independent oracle and physical hardware profile |
| driver launch/graph, AQL, retained PM4 or external NPU submit | existing backend command transport | real route, ABI, timepoints, leases and actual completion |
| shared host/UMA/VRAM/NPU memory, weights and IO | existing physical arbiter and device-local pools | same physical bytes charged once; all aliases and pin limits exposed |
| prefill/decode/speculation/cache/batch/placement choice | existing bounded joint planner | compare prevalidated feasible candidates with SLO/fairness and contention costs |
| semantic state publication | TaskCoordinator only | all domains prepared at one causal frontier; no provider commits independently |

Transport and kernel source are independent: a Rust-authored ISA/AIE emitter may
produce code objects/PDI/TXN for a qualified native driver/runtime transport.
The design must not require XRT, an external AIE C++ runtime, hipcc or LLVM23 for
all possible NPU/ISA paths. Optional approved runtime FFI remains a transport,
not a foreign model engine. Zerogravity still needs its own feasible, tested Rust
program/source path and authentic device completion receipts.

## 3. Two distinct cooperative prefill transformations

Retain same-operator **output-row partitioning** from the existing NPU contract.
Add **banked prompt-row pipelining** as a distinct legal region candidate; never
encode one by pretending it is the other. A bounded closed ModelStage/ModelProgram
region may contain both only after their joint dependencies and costs qualify.

A bank plan partitions the uncached prompt segment into exact ordered half-open
row ranges A, B, then later pairs/tails. Per layer it overlaps selected independent
input projections for one bank with useful GPU work on the other. Bank capacity,
NPU wave size, panel width and selected projection families are **candidate-profile
facts**, not constants in generic model/TaskDriver code.
Different model/processor/layout/driver has a separately qualified candidate.

Example source-preserving schedule (dependencies, not measured timing):

~~~text
fill: prepare A(0), pack/kick A(0); prepare B(0), pack/kick B(0)
for layer l:
    terminal_wait/acquire A(l); GPU Core A(l); record C-reader retirement A
    GPU output/residual/HC/MoE A(l); prepare/pack/kick A(l+1)
    terminal_wait/acquire B(l); GPU Core B(l); record C-reader retirement B
    GPU output/residual/HC/MoE B(l); prepare/pack/kick B(l+1)
drain: complete all required state/capture/head work; validate status; Prepare
publish: TaskCoordinator publishes the one accepted StateRoot
~~~

Dependencies permit pre-enqueueing GPU fragments behind device waits. Avoid a
synchronous host wait after every layer. A bounded relay/host progress bridge may
serve an accelerator requiring it, without becoming a new inference scheduler.
Bind the A→B edge to each state domain's actual owning phase: for PLE
convolution updated in Prepare, require **Prepare A(l) → Prepare B(l)**; for
GDN/conv, KV and indexer updated in Core, require **Core A(l) → Core B(l)**.
A generic Core edge cannot order an earlier Prepare mutation. Each phase's B sees
exactly its source state after A. Terminal wait/acquire above is notification,
not a success receipt; §5 governs dependent work and publication. Different
sessions never share mutable bank or causal state.

**Bank/private storage:** every value live across a bank switch has its own view
and lifetime (residual/HC streams, mixed inputs, HC gates, token-dependent lookup
rows, capture outputs and registered A/C arenas as needed). Layer-local scratch
may alias only after its last physical consumer, not after host launch returns.
Do not copy an entire second model/scratch pool when a verified liveness plan can
share local temporaries. Bank count/memory is an admitted plan, never hidden growth.

**No duplicate projection:** NPU-covered families cannot also execute their GPU
projection to manufacture an overlap win. Uncovered families/tails keep their
qualified GPU path. Preserve fusion continuity across Prepare/Core/Out windows;
splitting a layer into windows must not discard HC gates or disable legal fusions.
Direct typed BF16/native arena consumers may remove a retile/copy, only if their
layout, padding, rounding, ownership and source consumers are independently proven.

**Interior snapshots:** a legal prefix cut needs the full vector of source domain
versions at that token frontier. In a layer-major pair, global scratch may already
contain B for some layers when A finishes elsewhere. Copy/freeze each required
A-domain checkpoint before its B overwrite, retain captures until the all-domain
frontier meet, and publish only a coherent root. A pointer to bank A or a snapshot
of only recurrent/KV state is not a complete prefix. Otherwise split/defer the pair
or recompute from an earlier valid frontier. No stateful operator moves across a
missing input or a required source snapshot boundary.

## 4. Numerical identity includes the realized schedule

Four proofs remain independent:

1. **artifact identity/quality:** chosen immutable source vs an explicitly derived
   weight/encoding/sidecar assembly;
2. **backend execution parity:** CPU oracle, GPU twin and actual NPU for the exact
   admitted packed input/weight/scale/rounding/epilogue contract;
3. **source/task quality:** that compute regime vs the requested model/quant source
   on logits, selected sparse IDs, recurrence, long context, tools and task metrics;
4. **speculation correctness:** target verification, sampling/grammar/RNG and
   rollback relative to that same admitted target regime.

An NPU candidate may requantize packed source weights to W8 with grid scale
sidecars and emit direct BF16. Its GPU emulation is an oracle for that **specific
compute contract**, not a proof of the original quantized or FP32 source execution.
Matching greedy text on measured fixtures does not establish identical logits or
sampled law. Direct BF16 output and int32 partials with a later fold are distinct
contracts, each requiring independent numerical and memory-traffic qualification.

Bind the actual selected families, source/overlay and derived packing hashes,
quantizer/scale/epilogue rules, pair/chunk/tail boundaries, state schemas and any
qualified compatibility class to existing NumericContract/StateCompatKey and
ModeReceipt. Paired W8/BF16 work and native GPU tails can realize different math;
a broad `ValidatedTolerant` label alone cannot certify reusable prefix/state.
Calibration that changes segment cuts therefore changes provenance unless an
independent compatibility proof allows the old state. Physical execution and numerical emulation can share state only under a proven
matching numeric schedule; GPU-only reuse is not presumed.
Use segmented provenance/frontier records already attached to state and snapshots,
not a third durable IR or a giant hash that invalidates unrelated proven prefixes.

Strict source mode rejects an incompatible candidate before submission. A lossy
physical encoding or different arithmetic requires an explicitly permitted named
numeric/derived-encoding policy and independent quality receipt. No H5/H6 tolerance
is widened, no source bytes mutated, and no auto mode silently re-quantizes a model.

## 5. External terminal signal is not successful readiness

Model three distinct facts: **notification became terminal**, **result succeeded
and is visible**, and **all producer/DMA/consumer references have retired**.
An error/watchdog/unwind may need to release a GPU wait to avoid a queue deadlock;
it must store sticky failure before its release signal. The waiting consumer must
not treat that signal as valid tensor data or semantic completion. If the backend
can release a GPU wait on error, its queued dependent work must either be gated or
remain confined to a poisoned private candidate with valid bounded memory access;
otherwise refuse that transport. Pair/stage-end status and all required success
receipts precede capture, sampling, output or ReadyForPublication.

Arena reuse requires:
`old C readers done → retirement event → next prepare/pack wait → A release/ready
→ NPU submission/read/write → output acquire → successful new consumers`.
Retirement and terminal signals are bank- and generation-bound; ABA/stale success
cannot clear an earlier error. A watchdog signal is never proof of canceled DMA.
Failure/cancellation retains or quarantines registered host memory, operands,
queues and physical charges until genuine quiescence. Unrecoverable shared-SoC
uncertainty keeps the appropriate physical/fabric fence; no transparent CPU retry
mixes partially written results into a committed root.

## 6. Retained native replay within the existing backend

Support qualified command transport variants without another math/runtime IR:
ordinary launch, vendor graph, retained per-dispatch queue packets, retained native
command body. A retained PM4 body submitted using a public ROCr/HSA AQL packet is
distinct from direct-KMD/libdrm control. Capabilities and route receipts identify
the actual submission transport and its qualification.
A custom transport is optional, per ISA/runtime, and never a prerequisite for CPU,
Vulkan, CUDA, native HIP or NPU-only stateless regions.

Before admitting replay:

- Reconcile **every** baseline dispatch and mutable effect with a recorded node or
  named external adapter boundary; distinguish launch fusion from transport replay.
- Bind loaded code-object digest/symbol/alias, padded/aligned kernarg ABI, scratch,
  grid/block/LDS, resource effects and dependencies. Kernel-name hashes are insufficient.
- Freeze geometry except for named, typed, owner-bound, offset-bounded patches
  (position, table/row pointer, grid limit) within a validated capture maximum.
- Retain embedded source/state/scratch/parameter/code/queue/completion identities
  for the **entire prepared plan lifetime**, including idle intervals before future
  replay. Eviction/rebind/teardown must invalidate the plan before another replay.
  Destruction must additionally await every submitted consumer's real retirement;
  capture/encoding already pins pointers even before first submission.
- Derive acquire/release, alias hazards and fan-in for the actual ISA and queue;
  unknown access serializes. Remove barriers only with effect proof and hardware
  parity/co-tenancy/reliability evidence, never because two kernel names differ.
- Count prepared, selected, physically submitted and actually retired replay
  separately, including residual ordinary launches. Graph handles or Ready flags
  are not measured use of that route.
- Reset/model swap/topology/program/numeric changes invalidate incompatible plans
  and dynamic patch generations before replay. Unknown completion fences resources.

First qualify one queue, original order and conservative visibility. Multi-queue,
persistent loops and co-resident phases are later candidates. The same tests cover
logits, every state domain, guard bytes, exact bindings and physical cancellation.
H6's physical region count is not an admission for any of these transport routes.

## 7. Joint schedule cost, safety and calibration

Choose among a small bounded set of **prevalidated** plans, then use cheap table/
model lookup at safe phase frontiers. Include eager/graph/replay, packed kernel,
NPU family mask/bank sizes, speculative depth/real verifier rows, routed-expert
histogram/padding, state snapshot cuts, prefix hits, IO and SLO—not independent
flags whose Cartesian product is assumed safe.

For a bank pair, use a dependency-DAG makespan with separate fill/drain, NPU chain
padding, GPU pack/consume, first-use stalls, HC/PLE/expert work, driver/host relay,
shared LPDDR/IOMMU/power interference, context-dependent attention and snapshots.
Do not replace this with the row-split `max(GPU,NPU)` equation or add the same wait
twice. A source-specific example stall is `max(0, NPU_chain - other_bank_GPU_window)`;
it is not a universal constant or an end-to-end prediction without calibration.

Calibration keys include artifact/encoding/numeric/layout/shape, program/compiler,
physical devices/driver/firmware and relevant memory/OS/power profile. Performance
confidence and a positive noise margin cannot waive semantic, numeric, pinned-memory
or hardware-safety eligibility. Advisory performance-profile fields cannot replace
SoC safety qualification. Unknown
profiles choose a qualified baseline; deterministic mode does not change routes
based on wall-clock noise during a sampled sequence. If an adaptive mode is later
qualified, its target distribution and replay determinism claims are separately scoped.

Charge driver pinning (`VmPin`), mlock (`VmLck`), all bank arenas, panel/program
residency, graph/cache/snapshot lifetimes and OS headroom without counting aliases
of the same physical RAM twice. Pin-limit quotas remain a separate constraint.
Huge pages/translation tuning is optional measured capability; it does not erase
bytes, justify unlimited memlock, or authorize modifying host privileges/clocks.
Compare same-clock GPU-only and cooperative routes before attributing NPU benefit.

## 8. Product/model completeness and faster-than claims

Express supported capabilities through registered source
adapters/tasks/codecs/frontends: chat/reasoning/tools, source-specific speculation,
vision/OCR, text-to-image generation/edit, pooled or multi-vector embeddings, finite decision
readout, model/sidecar/quant lifecycle, prefix/disk caches, multi-GPU, lanes, health/
metrics and deployable packaging. Preserve each source family's exact operators,
processor/template, auxiliary tap/state and supported feature combinations.
Unsupported or quality-rejected modes are advertised as such. Compatibility is an
explicit protocol subset, not cloning another engine's internal commands/settings.

Messages/Responses are adapters onto canonical TaskRequest, not extra schedulers.
A late-interaction token matrix is not a pooled embedding. Decision logits and
calibration are source TaskDriver semantics. Frontend UI/TUI remains an optional
client. Tool schemas produce data; the inference runtime does not execute tools.
Runtime-only deployment can use prebuilt attested kernels plus approved driver
libraries; distinguish build toolchain requirements from runtime requirements.

Capability and performance qualification must pass the plan's HP-01..HP-40 gates. Benchmark the same
model/derived pack, prompt IDs, sampler, context, state/KV numerics and enabled
routes on matched hardware/software/power; report load/prepack, cold/warm TTFT,
AR and accepted speculative output, latency tails, concurrency/goodput, quality,
memory, failures and energy. Use route-positive receipts and repeated interleaved
runs. A different quality-normalized model comparison is valid only when named
separately. Uncertainty/no win/negative results stay in the evidence ledger.

**Outperformance is an objective requiring measurements.** Eliminating host
roundtrips, better memory/ISA use, shared projection work and joint scheduling may
win; shared bandwidth, Amdahl limits, source precision constraints and generic
engine overhead may erase those gains. Neither Rust, a broader design nor a faster
microkernel proves a whole-workload advantage. Current strict FP64-assisted Vulkan
qualification kernels are correctness evidence, not a competitive-speed baseline.

````

## Source: .agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md

Original UTF-8 bytes: 20079; SHA256 `718b9aaac7535ccf79cf32cd58997fdc1dec5e6939944cbe41e0344363a940bc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Cross-cutting invariants, state lifetimes and failure semantics

**15 Oct 2026 — SERVOPT multi-tenant/state invariant:** [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) preserves ONE StateRoot publisher and physical resource authority across concurrent models, source-compatible hybrid prefix fork/COW, target-only exact speculative correction and true GPU events; request/turn canceled epochs must not advance card-wide physical epoch or destroy other tenants' state. Prefill/decode/media work runs only with positive bounded credits, and no queue/IO/transport operation may hold global arbiter locks while awaiting physical completion. An overload fast-rejection, stale cache/key, remote state loss or unknown GPU completion cannot counterfeit a committed output. See SERVOPT-01..36.


**15 October 2026 — NEW STORAGE-RESIDENCY INVARIANT SSD-INV-01:** An immutable source tensor retains exact semantic/quant identity across storage tiers. A physical expert slot must not be evicted/reused while ANY encoded, captured, queued, submitted or still-running GPU command may dereference its old contents. Read leases start at GPU encoding; generation-tagged slot table publication follows source IO verification + true DMA Ready. No host directory consistency check may substitute for whole-pending-graph reader proof (DS4 #384). A short/corrupt SSD read before GPU submission fails without fencing a healthy card; unknown submitted upload/kernel retains actual physical buffers and global fence. Mixed quant Q2/Q4 pools share a single bounded byte budget; one layer cannot reset another layer's warm set just because expert size changes. Frozen private disk StateRoot snapshots are not immutable model tensors. [Normative specification](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md); [32 negative/positive gate cases](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md).


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


Every subsystem must pass these requirements. They are **design invariants**, not suggestions. Reference other documents for concrete structures and test methods. In failures, prefer explicit typed error/replay over plausible but invalid results.

## Correctness and identity

**INV-001 Checkpoint provenance:** a runnable model links verified model weights, config, tensor manifest, tokenizer, chat template and quant policy. A model ID string alone cannot identify semantics.  
**INV-002 Semantic exactness:** all required operator families, state dependencies and numeric policy are declared; unsupported operators fail at load with actionable diagnostic.  
**INV-003 Quant integrity:** every quantized block's logical size, padding, scale and offset match the model/format. Repacking is separately proved lossless; requantizing is an explicitly different model artifact.  
**INV-004 Causal frontier:** all visible tokens reflect valid committed model state; no future/rejected speculative token contributes to visible recurrence, indexer, attention or RNG.  
**INV-005 Cross-session isolation:** requests never mutate another's logical state even if sharing immutable prefix/weights; race ordering proven.  
**INV-006 Exact template contract:** tokenization, BOS/EOS, hidden reasoning, tool rendering, multimodal positions and stop logic match the chosen model reference.

## Effects, ownership and memory

**INV-007 Linear state transition:** each state version consumed according to effect ordering. Graph optimization cannot drop or reorder writes across a live dependency.  
**INV-008 Lease generation:** GPU/host/IO references hold a valid `BufferLease`; eviction waits for completion, stale generations rejected.  
**INV-009 Expert slot safety:** resident/missing expert kernels cannot observe a partially populated or replaced slot; slot version + layer/expert mapping verified at use.  
**INV-010 Physical budget:** enforce OS headroom, pinned/committed physical memory, driver/UMA/GTT constraints and session scratch reservations before admission. Report virtual GTT capacity distinctly.  
**INV-011 IO integrity:** disk-backed table/expert read validates exact position/length/checksum as appropriate; EOF/errors invalidate dependent transaction. No stale bytes.  
**INV-012 Async synchronization:** events/fences order GPU writes, CPU reads, peer exchanges and slot reuse. No implicit reliance on an idle stream or incidental device synchronize.  
**INV-013 Fallback conversion:** an optimized representation may be replaced only at a validated layout/state/causal boundary; failures cannot pretend a different kernel with different caches continued the session correctly.

## Serving and speculative semantics

**INV-014 RNG fidelity:** sampler PRNG and penalty history correspond to committed tokens; rejected drafts do not perturb exact sampling.  
**INV-015 Speculative distribution:** when exact target sampling advertised, proposal acceptance/correction implements the target distribution; opportunistic sampling only explicit.  
**INV-016 No hidden latency starvation:** token-budget admission enforces fairness and bounded resident/preemptible resources; TTFT/P95 tracked. A batch path is qualified as native only if it truly computes multiple rows jointly.  
**INV-017 Cancellation atomicity:** cancellation releases only quiescent buffers, emits no tokens beyond committed frontier and preserves restart/replay capability.  
**INV-018 Persistent snapshot:** snapshot schema/version/checksum/model/quant/pos policy validated before state restore; prompt text privacy and permissions preserved.  
**INV-019 Remote frontier consistency:** distributed partial results cannot be committed on only one rank; fatal transport mismatch invalidates route and causes safe replay.

## Model/hardware compatibility

**INV-020 Capability truthfulness:** adapter support, numerical correctness, physical backend availability and performance qualification are distinct status fields; absence of extreme kernel is not model failure when generic backend supports all required ops.  
**INV-021 Hot-path autonomy:** direct specialized execution avoids per-op IR interpretation and avoids host routing lookup for resident routed experts where the plan promises GPU-side handling.  
**INV-022 Numerical transparency:** approximations, quant changes, sparse top-k changes and non-exact sampling clearly labeled; benchmark comparisons match precision/quality policy.  
**INV-023 Security boundary:** untrusted model artifacts and client requests cannot allocate unchecked sizes, execute arbitrary code or reach arbitrary host filesystem/network.  
**INV-024 Measured claims:** benchmark artifacts bind exact code/hardware/model/driver/power/context/temperature and complete execution; incomparable author claims stay attributed.

## Failure state machine

```text
ADMITTED -> RESERVED -> EXECUTING -> PROVISIONAL
                                |          |
                                |     VERIFY/VALIDATE
                                |          |
             FAILED <-----------+      COMMITTING -> COMMITTED -> EMITTED
                |                           |
     QUIESCE/QUARANTINE              durable frontier
                |
    RELEASE / REPLAY / REJECT
```

Emit only committed tokens. If a kernel fails after output from a previous commit has been emitted, report the failure without forging the rest of the stream. Internal rollbacks never retract already externally committed output. Cancellation epoch guards against late callbacks.

### Typed error classes

- `ArtifactError::{Corrupt,Checksum,UnsupportedQuant,WrongShape,TokenizerMismatch}`
- `ModelError::{UnknownArchitecture,UnsupportedOperator,StateSchemaMismatch}`
- `PlanError::{NoLegalKernel,LayoutConversionMissing,NumericalPolicy,GraphCaptureUnsupported}`
- `ResourceError::{UnsafeBudget,QuotaExceeded,OutOfMemory,LeaseConflict}`
- `DeviceError::{LaunchFailure,Timeout,LostDevice,InvalidEvent}`
- `StorageError::{UnexpectedEof,ReadFailed,ChecksumMismatch,CorruptSnapshot}`
- `ProtocolError::{PeerMismatch,TransportFailure,IncompleteCollective}`
- `RequestError::{InvalidInput,TooLarge,Cancelled,UnsupportedModality}`

These Rust names are illustrative; actual APIs may consolidate variants while preserving semantics and cause.

## Failure response / recovery matrix

| Failure | During | Required behavior | May retry? |
|---|---|---|---|
| Unknown op / malformed GGUF | load | refuse before GPU allocation, describe exact tensor/op | after compatible adapter/fixed file |
| Insufficient physical RAM | planning/admission | reject or try smaller context/batch/scratch, never silent requant | after resource changes |
| Unsafe expert eviction | route | prevent overwrite, wait on referenced event, stall safely | after lease releases |
| NVMe expert/Engram read short | execution | invalidate row/staging, abort dependent transaction | yes from clean checkpoint |
| Sparse attention selection invalid | compute | fail or execute verified fallback; no truncated wrong top-k | if canonical source valid |
| Device kernel timeout/loss | compute | quarantine event/buffers, invalidate provisional state, drain/reset | only verified restart |
| Draft reject mid-token | verify | commit accepted prefix and corrected token, restore all other state | normal path |
| Tool sampling policy changes inside draft | verify | rewind at policy boundary and verify again | yes from snapshot |
| Request disconnect | stream | cancel/reconcile no uncommitted output, cleanup after GPU quiescence | new request |
| Corrupted snapshot | restore | reject, rebuild from canonical history if available | yes recompute |
| Network TP peer disconnect | collective | fail-closed both ranks, no fake remote values | new matched route |
| Compiler/kernel tuning drift | plan | invalidate cached plan and revalidate candidate | compile with pinned toolchain |

## Ownership/lifetime diagrams (text)

```text
ModelArtifact (immutable, hashes)
     └── ModelExecutable (compiled regions, read-only weights)
           ├── ExpertCache slots [generation, GPU/IO leases]
           ├── Session A [committed state frontier p]
           │     ├── Snapshot p (immutable/shared)
           │     └── Tx p+1 (provisional, isolated)
           └── Session B [fork from Snapshot p, private continuation]

ResourceArbiter -> allocates BufferLease -> async Event -> safe recycle
```

State commits are serial per session, but independent sessions may execute concurrently in one GPU batch. Cache evictions operate over physical slots, not logical model tensor IDs. A lease must be tracked across all async CPU/GPU operations.

## Defensive tests / diagnostics

Fuzz malformed GGUF/safetensors/template/snapshot, integer overflow, truncated model file, random expert misses and slot reuse; force every possible speculative accepted-prefix length and kernel failure at each stage; stress cancellation against in-flight GPU events and server restarts; check per-state checksum at commit, physical memory watchdog and no host-wide OOM; replay same prefixes under altered chunking and log numerical difference. Distributed tests must inject wrong model identity and delayed/missing collective messages.

Every new optimization PR must explain which invariants it touches and include tests. A fallback path without an explicit state layout boundary violates INV-013, even if short prompts appear to work.

## 2026-10-10 integrated v2 addendum — cross-IR and worker-event invariants

Apply all existing INV-001…024 to the consolidated IR pipeline:
- **SIR/RIR**: only pure mathematically legal transforms; explicit state/effect token ordering; no unverified compiler speculation across possible failure/RNG/IO effect.
- **KIR/KernelPlan**: each provider advertises exact numerical, ISA, tensor/storage/state-layout, capture-mode and scratch constraints; no generic shape-only override.
- **CIR**: `Acquire/Dispatch/Copy/Read/Wait/Publish/Commit/Release` checked for alias, resource tier, event/timepoint ordering and buffer lease generation; no early expert eviction, no indefinite GPU wait on IO failure.
- **EngineCore**: one logical mutation owner; StepPlan and WorkerDelta epoch match, late completions cannot publish tokens or release referenced state. Separate worker and frontend never commit target output.
- **Prefix/LoRA/multimodal**: cryptographic model/quant/token/rendered-template/adapter/tenant/snapshot/finalization identities; cache "hit" must be fully restorable or recompute domain before use.

**Additional invariant INV-025 — source modularity:** adding a new supported semantic operator cannot require forking serving or ABI; adding new kernel provider cannot require changes to model math/HTTP; all such changes validated by extension test. **INV-026 — no uncontrolled serving optimizer:** every selected kernel variant was prevalidated for quant/model/hardware/numerics and command resources; real-time scheduler must not run unbounded rewrite/JIT. **INV-027 — mode receipt:** all user-visible capability degradations, kernel mode, quant changes requested and speculation fidelity are reported accurately. **INV-028 — package format correctness:** serialized header/magic/version/bytes/schema must be self-consistent and fuzzed, explicitly rejecting old research's defective disk-KV definition.

Failure semantics: an event may complete with error *after* a host issued cancellation; hold lease/quarantine until proper device quiescence; no double commit or retry with half-old recurrent state. Always distinguish source model precision from runtime compute and changed checkpoint quality, and fixed-plan numerical reproducibility from cross-backend approximations.


## 2026-10-12 integration — cross-accelerator safety and semantic invariants

INV-029: output-row projection split covers disjoint complete sets, restores original tensor row order and preserves model attention/recurrence/quant/routing semantics. INV-030: XRT/GPU imported pinned memory is protected by event and lease generations until *both* runtimes quiesce; timed-out external DMA may need terminal quarantine. INV-031: concurrent GPU+NPU is disabled for unknown/unsafe SoC power-fabric/firmware states, and no privileged FCLK/BIOS/IOMMU/suspend modification occurs automatically. INV-032: v8bfp16ebs8 compute is quality-qualified independently; strict numerical reference does not silently switch to NPU. INV-033: ASR/TTS/image/video TaskProgram stages have task-specific loop/output/state and do not masquerade as ordinary language autoregressive KV/sampler.

Required tests: mock NPU + GPU row join, stale BO/double free, timeout and partial result, cancellation after submit, late XRT completion, strict fallback to GPU, hardware-qualification fail-closed and 24h stability on isolated hardware. See [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

## v4 additional invariants and fault tests

**INV-034 Atomic session visibility:** `StateDomain::prepare` cannot mutate committed root; one TaskCoordinator writer validates all required domains/events and performs exactly one `PublishRoot`. Device/CIR event ready is not logical commit. **INV-035 Async host/device correctness:** validated StepProgram may progress device-native token/routing/sampling without mandatory per-token CPU synchronization; unsupported grammar/shape uses honest host-loop fallback. **INV-036 Hierarchical allocation:** global physical admission reservations and per-device local generation leases, no lock for each hot kernel scratch allocation, no impossible promise of OS-invariant free memory.

**INV-037 Dynamic shape legality:** rank, symbolic dimensions, ragged offsets/strides, numeric/quant/ISA and effect schema checked before kernel submission; cold missing variant chooses portable fallback, not uncontrolled JIT. **INV-038 Stream transport honesty:** emitted output has session/epoch/seq identity after root publish; live delivery attempt isn't exactly-once network delivery. **INV-039 Distributed fencing:** late epoch remote peer cannot cause publication of partially prepared state; durable exactly-once only if explicit protocol actually built.

**INV-040 TaskDriver boundary:** coarse trusted AI computation stage loop, no arbitrary model-supplied ExternalAction; non-token tasks use their own output frontier. **INV-041 Training future seam:** ParameterVersion mutations do not alter pinned inference sessions or model/bundle/prefix compatibility. **INV-042 Scope/minimality:** if a new specialization requires duplicated model server/allocator/scheduler or four independently persisted IR dialects, review/reduce before implementing.

Add fault tests: fail second domain prepare, late device DMA event, unsupported ragged GPU shape, remote lost ACK, slow consumer/stream reconnect, and trained model/adapter version change. Results remain NOT_STARTED until source and independent evidence exist.

## v4.3 multi-model routing and API invariants

**INV-043 — exact model binding:** alias resolution at admission yields pinned checkpoint/quant/tokenizer/template/adapter/ModelProfile revision and TaskDriver/numeric policy. A route/alias hot reload never mutates a live session. **INV-044 — replica semantic equivalence:** load balancing can only move a stateful session among compatible exact model deployments after a safe full StateRoot/placement transfer, never random per-token model or quant substitution.

**INV-045 — residency safety:** registered catalog models need no physical memory; a resident deployment has verified ResourceArbiter reservation and DeviceBackend graph leases. Active/unsnapshotable sessions, prepared speculative steps and live DMA prevent eviction. **INV-046 — router control-path:** ModelRouter performs coarse request/model admission, not source-graph traversal/JSON HTTP dispatch/locking on the GPU hot token path. Native batching uses existing TaskCoordinator.

**INV-047 — multi-model task roles:** target/draft/encoder/rerank/vocoder are separate registered models with separate admitted leases, not replicas or silent target alternatives. **INV-048 — config safety:** immutable source/model semantics and hard auth/memory/peer/numeric constraints cannot be relaxed by model profile or request; effective provenance deterministic and inspectable.

**INV-049 — local-first API:** no auto system daemon, public listener, remote peer or unknown source download on `run <local-model>`; external bind needs auth/TLS, model/admin scopes and tenant alias filtering. **INV-050 — API contract:** OpenAI-compatible is endpoint/field/stream/SDK tested subset, not implicit Responses conversation persistence, cloud built-ins or tool execution. Only committed StateRoot output emitted with sequenced event identity.

**Tests:** hot alias rename mid-stream, 100 concurrent cold requests dedup, tiny warm model protected during giant cold load, unload while in-flight KV/DMA, same source replica with incompatible numeric mode, alternate quant request refused, `/v1/models` privacy, official SDK streaming, Responses unsupported store/tool field typed refusal, verified direct CLI zero configuration.

````

## Source: .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md

Original UTF-8 bytes: 22925; SHA256 `d62f507a2a3bac341b753d0ff512c7f7a2f81c15cf95779903bebd4359272dde`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — Joint Concurrent Serving, Goodput and Combinatory Specialization Contract

**15 October 2026 — DESIGN ONLY; not production serving or a vLLM speed claim.**

**Authority:** Additive to v4.3 SemanticProgram → ExecutableBundle, TaskDriver, one TaskCoordinator/StateRoot publisher, one PhysicalResourceArbiter, DeviceWorkers, ModelCatalog/Router, and v5.1 realtime/multimodal TaskSessions. No new model-specific inference engine, second resource authority or durable compiler IR.

**Companions:** VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md and VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md in .agents/plans.

## 1. Design objective and competitive honesty

Preserve the concurrency fundamentals of vLLM-class systems—continuous token work scheduling, paged and hybrid state, prefix reuse, chunked prefill alongside decode, speculative verification, bounded admission, async worker pipelines, optional distributed P/D, and truthful metrics—and jointly specialize these **with**:
- exact model/source/processor and tensor encoding,
- quantized kernel geometry, fuse candidates and real device ISA,
- routed MoE expert overlap, padded physical row groups and source state,
- MTP/DFlash/lookup speculation and accepted tokens per expensive target step,
- GPU/UMA/host/SSD residency and real transfer completion,
- CPU/GPU/NPU/peer placement, power/thermal headroom and physical resource quotas,
- text TTFT/ITL, duplex audio latency, image/video deadline and tenant fairness.

The aim is to maximize **source-verified useful output meeting its declared service objective**, under safe physical budgets and latency, rather than optimizing unrelated component metrics independently.

Current Zerogravity implementation does NOT yet equal vLLM V1 production continuous batching; code for CPU Qwen3.5, catalog, fair resource arbitration and selected physical GPU regions is separate, narrower evidence. Any future assertion that Zerogravity outperforms vLLM requires a tuned comparable vLLM baseline with an identical model/quant/hardware/numerical setup and real concurrent request traces.

**Simplicity ceiling:** one native Rust engine, one request planner/control loop, one physical resource authority, prevalidated candidate StepPlans and cheap device-local batch execution. The joint optimizer is a planning/control view **within** the existing ExecutableBundle; not a new optimizer daemon, per-token graph search or giant dynamic dispatch hierarchy.

## 2. vLLM-class baseline — operational detail to preserve

1. **Continuous batching:** admit compatible work at safe token/stage frontiers instead of waiting for a whole request to finish.
2. **Unified token/work scheduling:** distinguish needed vs computed prompt/output/speculative tokens; allocate from bounded per-step work budget. For media, use source-typed encoder/acoustic/solver work rather than coercing every unit to text tokens.
3. **Chunked prefill/decode coexistence:** decode prioritized to protect ITL, long prefills served in legal chunks and aged to avoid starvation; chunk size may vary with available budget and source/state validity.
4. **Paged+hybrid KV/SSM:** lazily allocate physical KV/recurrent pages/state, protect in-flight reservations, reuse compatible committed prefix blocks and maintain exact source attention/recurrent state.
5. **Speculative decoding:** trained MTP/DFlash/lookup proposals remain provisional; target verifies whole state, sampling/grammar and accepted prefix before one StateRoot publication.
6. **Async worker:** overlap host preparation, source IO, transfers, GPU kernels and result delivery where mathematically legal; never hold global resource locks awaiting kernel/network.
7. **Distributed P/D or expert placement:** cost actual state transfer, network tail and warm/cold placement; latency isolation is distinct from increased throughput.
8. **Bounded overload:** request queue, live stateful sequences, prefill token backlog, device rows and memory each have separate admission limits.
9. **Multimodal coordination:** timed audio/video and finite decisions share global resource admission but retain own legal TaskDriver clock and output frontier.

vLLM V1 and vLLM-Omni already have sophisticated hybrid cache/expert/disaggregated/real-time capabilities; differentiation is **quality-grounded joint selection on specific devices/workloads**, not claiming those features are absent elsewhere.

## 3. Three independent admission limits

| Admission boundary | Bounded unit / owner | Before allocating what | Overload behavior |
|---|---|---|---|
| API and tenant queue | queued+inflight requests, payload bytes, tenant credits/weight, request deadline | any heavy model/session buffer | reject or bounded queue; typed 429/503-style overload with permissible retry advice |
| Model/state/prefill | immutable model assembly residency, private KV/GDN/KDA/draft, queued uncomputed prefill work, IO staging and OS+device quotas | active source session and state resources | defer, shrink legal prefill chunk, qualified preempt or refuse |
| Device StepPlan | active sequences, target verifier rows, token budget, media encoder units, physical kernel scratch/leases/driver depth | actual CPU/GPU/NPU submission | bounded batch selection or NoCapacity, never counterfeit Ready |

Queued HTTP requests are NOT active stateful sequences; a request with a 200K-token prompt is not one unit of GPU memory. Counting only active sequence slots allows unbounded prompt/work pressure. Counting only HTTP queue length ignores draft and video memory. Queued requests must not eagerly allocate full per-session KV.

Credits must be hierarchical for server/tenant/model/session and preserve fairness with aging. Decode and interactive audio may have protected latency priority, but long prefill cannot be starved permanently. Capacity admission must include active context memory, source weights/shared source ownership, DMA, host/SSD buffers and OS safety reserve. Releasing one reference to a shared loaded model does not free weights until all consumers genuinely quiesce.

**Cancellation contract:** waiting ticket cancel releases estimate; an unsubmitted prepared StepPlan releases preflight leases; a physically submitted command is retained until real completion or fenced/quarantined on unknown completion; request/turn cancellation never advances a card-wide physical epoch.

## 4. Unified request work frontier

The scheduling representation is an ephemeral typed **view** of existing model/session StateDomains; not a new persistent compiler IR.

~~~rust
// Non-binding Rust type sketch.
struct RequestWorkFrontier {
    session: SessionId,
    committed_root: StateRootId,
    source_model: VerifiedModelId,
    input_count: u64, verified_cached_prefix: u64,
    computed_count: u64, committed_output: u64,
    speculative_unverified: u32,
    input_media_stages: Vec<TypedWorkFrontier>,
    source_state: Vec<StateDomainFrontier>,
    generation: RequestEpoch,
}
struct StepBudget {
    active_sequences: u32,
    target_token_rows: u32,
    media_work_by_domain: Vec<DomainWorkBudget>,
    physical_scratch_bytes: u64,
    read_write_io_credits: u32,
    soft_deadline: Option<Deadline>,
}
~~~

At one scheduling opportunity:
1. Drop expired/stale/canceled waiting tasks before heavy admission.
2. Make progress on source-required already-running steps; inspect only ready/real events and source state frontiers.
3. Choose compatible decode, target verify and latency-constrained audio/video with protected credits; avoid one huge prefill blocking a live stream.
4. Fill spare target-row budget with legal prefill chunks after fairness/aging, respecting state/grammar/source processor boundaries.
5. Avoid splitting a codebook frame, video latent solver dependency or other atomic source task unit in the name of generic token budgeting.
6. Distinguish blocked-by-IO requests from executable work; SSD cache miss is a materialization Future, not a GPU source-ready token.
7. Construct a bounded compatible StepBatch; no source/quant/grammar conversion merely to merge rows from different models.
8. Dispatch with actual resource/event receipts; publish only source-valid committed frontier after device completion.

Rows proposed by MTP/DFlash are not output tokens. Required target verify rows for batch of B sequences with per-session depths d_i are conceptually R = sum(1+d_i), with physical expert-row/group padding and source state included in cost.

## 5. Hybrid prefix cache — exact state-domain frontier meet

A valid prefix hit must satisfy the *intersection of all source-required StateDomains*, not only global-attention KV hashes. Possible domains:
- full KV pages and position/RoPE masks;
- sliding window KV/rings and source lookahead/slot generation;
- GDN/KDA/SSM convolution/matrix state at a complete checkpoint boundary;
- MLA compressed KV, DSA/QSA indexer and required sparse selected positions;
- mHC stream state, draft model auxiliary tap/hidden and MTP state;
- source tokenizer/template/processor and multimodal encoder/codec alignment.

**Semantic key** (conceptual):
(original or derived artifact hash + exact component assembly, tokenizer/template/processor, source content/tokens, positional policy and state schema, numerical/reduction mode, tenant sharing authorization, causal StateRoot version and compatible restore recipe).

Physical device placement can change via a *verified state conversion/transfer*; pointer location is not the semantic key. A source-level prefix cache cannot simply share private user state because token hashes happen to match. Only immutable/share-authorized source features may be reused cross-tenant.

### Frontier meet algorithm
Each domain supplies a set/range of valid reusable committed checkpoints and what source context is necessary to resume there. Find the largest prefix frontier whose **full vector of domains** can be restored causally, with exact normalized input positions; if a GDN checkpoint is missing at that point, reduce length to previous complete joint checkpoint or recompute. For a sliding attention layer, retaining last W slots may be sufficient for source-compatible future continuation, but W alone is not an unconditional reusable prefix proof. For a source recurrent state, a single needed hidden matrix might force full replay despite global KV reuse.

A generalized multi-domain meet is required: avoid hardcoding that hybrid means just “full KV + one SWA cache”. Safe COW and reference-counted pages protect simultaneously queued/shared readers, including encoded-but-unexecuted GPU work. The candidate can choose **recompute vs RAM/SSD restore** by measured costs and integrity/tenant proof. An untrusted disk snapshot is never automatically a prefix hit.

## 6. Joint candidate selection: fast bounded Pareto lookup

Do not optimize batching, quant, MTP, cache and kernel independently. The planner filters a finite set of **prevalidated source/math/shape/device plans** and chooses among a small measured Pareto neighborhood at safe boundaries.

| Candidate dimension | Legal choices | Must preserve |
|---|---|---|
| Active work | B compatible sessions, target rows, prefill chunk length | tenant fairness, model and state domains |
| Speculation | none/MTP/DFlash/lookup, draft depth, exact target correction | accepted-token/source sampling distribution and full rollback |
| Physical weights | expert hot cache, GPU/RAM/SSD source extent, model component residency | source bytes/quant, cache readers pinned until Ready |
| Kernel | direct packed GEMV/GEMM, fused regions, row tile/ISA/numeric | qualified source tensor/output and reduction/effects |
| Placement | local CPU/GPU/NPU or trusted peer, collocated/disaggregated P/D | source artifact, state transfer validity and latency |
| QoS | TTFT, ITL, tail latency, audio first-chunk/jitter, video progress, energy | admitted advertised objective and error reporting |

Objective:

~~~text
p* = argmax over SOURCE-QUALIFIED feasible plans p:
    SLO-Goodput(p)
subject to:
    source/model/quant/numeric/effects correctness,
    physical GPU/OS/IO budgets and real completion,
    per-tenant fairness and active cancellation,
    declared tail latency and task deadlines.
~~~

**Text goodput** = useful accepted committed tokens meeting stated per-request TTFT+ITL/TPOT service targets over wall time, including overloaded/failed arrivals in reporting. **Media goodput** uses complete source-valid task units meeting its own latency objectives (PCM chunks, transcript segments, images or video segments); do not add heterogeneous unit counts as if one audio sample equaled one text token. A weighted multi-tenant score requires explicit normalized utility and per-class metrics; display class-specific goodput, admission losses and fairness even when optimizing a composite score.

Cost model dimensions: tensor codec/decode + physical expert-row padding + unique routed expert IDs; expected accepted target tokens and verifier rows; KV/GDN/MLA/indexer snapshot/COW; hot/cold GPU/RAM/SSD hit/miss and DMA; CPU/driver/IO stage overlap; batch/prefill interference; GPU graph reuse vs cold-shape JIT; physical resources and OS headroom; p95 queue/fairness, energy, backend/driver measured capability.

A small optimizer may update preferred candidates using offline/background evidence and hysteresis, not launch an exhaustive kernel search per token or during low-latency audio. Unknown hardware/model shapes fall back to simple qualified baseline. Repeatedly neutral/negative optimizations remain recorded, not forced.

## 7. Async worker pipeline and hot-path costs

Target architecture is bounded Prepare → Dispatch → Retire loops with source-verified state/physical isolation:
1. Host tokenizer/codec/verified preprocessing, IO and H2D may overlap with legal GPU compute on another admitted step; compiler/shape search must not block hot path.
2. Reuse preallocated per-shape command buffers, scratch arenas and model source descriptors; a fully resident token step must not require catalog JSON serialization or global arbiter snapshot.
3. A worker can have more than one GPU region in flight **only after** real per-stream event/lease isolation and driver recovery are physically tested. Current single worker region is *not* production continuous GPU concurrency.
4. Source backends supply true event Ready and exact output; scheduler selection is distinct from GPU submitted and actual retired kernel.
5. Provisional state for a later step may prepare ahead, but cannot publish ahead of its source predecessor or overwrite another session; root epoch/prefix validity checked at single TaskCoordinator.
6. Graph capture must protect ALL encoded/captured/queued source pointers and buffer generations until physical quiescence, not merely those in a presently executing layer (see DS4 issue #384).
7. Output network/media delivery is independently credit-bound; a slow socket never indefinitely retains GPU scratch.
8. Metrics distinguish parse/routing, queue, host prep, IO, H2D, kernel waiting, GPU execution, physical Done, StateRoot publication, media transport.

No mandatory CPU roundtrip for every layer if source-compatible device-native graph island can execute multiple steps; no false or fabricated readiness when IO/device events are missing.

## 8. Distributed prefill/decode and collective overlap

Retain the existing **single PlacementPlan** with compute, weight, mutable state and reusable cache owners and a typed TransferPlan. Disaggregated prefill/decode is **optional**: it can isolate TTFT from decode ITL, but is not automatically a throughput win.

- Prefill/decode model source+derived quant+template/numeric/state schema and causal positions must match; avoid duplicate source tokenization where permitted.
- Transfer **all** source-required state for hybrid models, not only global KV: GDN/KDA/SWA/MLA/mHC/indexer, sampler state, media track frontiers where relevant.
- Remote transfer receipt authenticates tenant, source model, state epoch, digest, version/layout and actual peer/device completion; no independent remote StateRoot publication.
- Stage/microbatch overlap of send/receive/collective and useful device compute must be legal in semantic dependency DAG.
- Compare tail ITL, TTFT, cross-device transfer bytes, queue and peak residency against collocated chunked mixed P/D; choose cheapest verified option.
- Local single-device hot path never opens a distributed RPC just to query placement; replicas/expert placement selected and pinned by request/phase.
- WAN/tunnel untrusted or high-RTT transfer may lose to local recomputation; explicit permissions and source identity control.

## 9. Failure, overload and fairness cases

- Large burst of queued requests: capacity valve, 429/503-like structured response or admitted bounded wait. No unbounded prefill work/state buffers.
- Decode-vs-prefill: protect ITL without starving long prompts, through aging/protected minimum work and bounded chunking.
- Prefix cached for KV but not recurrent/vision/codec state: exact bounded replay/restore, never report full hit.
- Draft target verifier full capacity, incompatible MTP sidecar or low acceptance: target-only fallback, identical stochastic distribution where lossless mode claimed.
- GPU cache slot being read by a *captured but not executed* command: cannot evict even if CPU cache directory marked idle; no premature reuse.
- SSD IO miss waiting: bounded staging and actual H2D Ready; cannot block all unrelated model requests under global lock.
- Target session canceled while other sessions on same physical card active: cancel only request/turn generation, not card epoch.
- Device uncertain submitted event: retain/quarantine real source/output handles and global fence until proven quiescent.
- Remote state transfer partially done: no prepared output publication; safe retry only if idempotence and model state/root support it.
- Malformed media or malicious payload: validation/refusal before expensive memory/device allocation.
- Hidden JIT/autotuning on first unfamiliar audio/image/video stage: bounded generic known-good candidate, not unbounded search.
- P95 latency misses, queue timeout and rejected requests: counted and disclosed, not silently excluded from goodput.

## 10. Metrics and honest performance claims

At minimum record per request: model artifact/quant/source processor version, task mode, arrival/admit/reject/timeouts, queue wait, live session count, committed tokens and tokens/s, TTFT/ITL/TPOT p50/p95/p99, actual prefix reused across each required domain and restoration cost, draft proposed/accepted/corrected, target verifier rows, unique experts/padded rows, GPU+OS+SSD physical pressure, IO/hot cache H2D, kernel scheduled/issued/ready counts, active/queued state, backend/driver/shape/numeric evidence, energy/power when available, tenant and tail fairness.

Avoid survivor bias: goodput denominator includes all offered work/attempted requests and rejection/timeout counts. Must gate service availability, fairness and rejection ratio alongside throughput. A server rejecting difficult requests is not “faster” by the goodput definition.

Benchmark current optimized vLLM V1 and vLLM-Omni on equal model/checkpoint/quant/tokenizer/input distribution/hardware/driver/capabilities and comparable latency policy. Different source quantization or lossy cache mode makes the result non-equivalent; report source quality alongside performance instead of treating it as a free optimization. Cold vs warm and single vs mixed workloads separate. vLLM-Omni may already support relevant multimodal scheduling, so don't infer new feature novelty solely from its task type.

## 11. Phase ownership and evidence status

| Roadmap | New explicit acceptance focus |
|---|---|
| P09 | three-level request/model/device admission; bounded queued prefill work; no unbounded API backlog |
| P10 | hybrid source-domain prefix frontier meet, tenant-safe COW/SSD restore, global memory credits |
| P12 | model/quant-compatible groups, source MoE expert count/padding and actual kernel row bands |
| P15 | exact MTP/DFlash/lookup proposals, target verification and measured acceptance/backoff |
| P16 | bounded measured joint Pareto candidates, batching/prefill fairness and goodput QoS |
| P20 | per-ISA/model/candidate cost evidence, cold-shape/negative results, no per-token tuning |
| P22 | pinned vLLM/vLLM-Omni benchmark parity and source-equivalent honest advantage proofs |
| v5.1 / distributed | realtime media QoS, remote state/collective overlap and protocol capability receipts |

Planned SERVOPT-01..SERVOPT-36 tests are specified in .agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md. **They are not implemented or passed.**

## 12. Primary sources and version caveat

- vLLM V1 scheduler: https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/
- vLLM hybrid KV manager: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
- vLLM prefix cache: https://docs.vllm.ai/en/latest/design/prefix_caching/
- vLLM chunked prefill: https://docs.vllm.ai/en/stable/configuration/optimization/
- vLLM optional experimental disaggregated prefill: https://docs.vllm.ai/en/latest/features/disagg_prefill/
- vLLM config request queue controls: https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- vLLM-Omni: https://docs.vllm.ai/projects/vllm-omni/en/latest/
- Prior-art specialization source atlas: the specialization source atlas in the plans directory (2026-10-15).

**Checked 15 October 2026.** vLLM's hybrid KV support, disaggregated connectors and decoder features evolve; review at pinned upstream commit/driver before running comparisons. No vLLM binaries were run, no end-to-end Zerogravity model-serving performance tested in this design update.

## 13. U-048 — NPU banks and retained replay participate in joint choice

The [heterogeneous pipeline contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) adds bank/family mask, paired
prompt cuts, direct arena layout and actual dispatch transport to the existing
finite candidate set. Price fill/drain, padded NPU waves, pack/consume, causal
snapshot cuts, other-bank GPU overlap and shared LPDDR/power interference with
expert-row padding, speculation and SLO. Separate same-op row-split cost from
layer-major pipeline makespan. Reuse calibration only within its qualified
correctness/safety envelope; it cannot approve unknown firmware or alter exact
mode. Realized numeric schedule belongs to state compatibility. Admit the whole
feature/phase tuple; separate NPU, MTP, EP and replay support is not composition
proof. Compare useful accepted output and real latency against a qualified baseline
route before any outperformance claim; these are planned design requirements.

````

## Source: .agents/docs/MEMORY_RESIDENCY_AND_IO.md

Original UTF-8 bytes: 29375; SHA256 `b45e4323babf1a655dc8a96dcde9266e7a0cfe9e1caa38e32f58fcedb9083176`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Global memory, expert residency, storage I/O and resource admission

Design requirements below. Implemented/tested scope and unfinished work are
tracked only in [the plan TODO index](../plans/TODO.md).

**15 Oct 2026 — multi-request physical memory and prefix-work accounting:** [Joint serving cost contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) makes admission count immutable source weights once, per-session KV/GDN/KDA/draft and prefix-COW state, real active + queued prefill work and actual device/pinned/OS/SSD scratch before kernels submit. Physical GPU/host caches may be shared under source/tenant-compatible leases; selected/published/Ready counters remain distinct. More speculative rows increase unique routed experts and physical byte pressure, potentially slowing output even when acceptance rises. A prefix restored from SSD is not zero-cost, and a cached expert used by encoded-but-not-yet-executed GPU commands cannot be evicted. SERVOPT-05/12..18/21/26 exercise those combinations.


**15 October 2026 — FIRST-CLASS STORAGE-BACKED EXECUTION (new user requirement):** [Unified Virtual Tensor Residency and SSD Streaming](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) elevates SSD streaming from an optional expert miss sketch into the same verified TensorView/physical BufferLease/DeviceWorker execution fabric as GPU-resident weights. One logical immutable SourceTensorId can materialize directly on GPU, mapped UMA, bounded host RAM, OS page cache or local SSD, with one global **byte-budgeted** ResourceArbiter and domain-local expert/row/frozen-state policies. Fast resident kernels incur no mandatory host/IO dispatch; SSD misses pay real bounded IO+DMA latency with asynchronous dependency overlap. Cache slots must be pinned from **GPU command encoding through actual completion**, including captured-but-unexecuted graph readers (DS4 #384); mixed Q2/Q4 size classes share ONE byte cap and prefill must not discard warm decode caches (DS4 #503). [SSD-01..SSD-32 acceptance and phase mapping](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md). Source integrity, file-descriptor identity, source bit output, UMA physical headroom, cache/QoS and actual device Ready are mandatory; no native SSD runner is claimed implemented.


**15 Oct UMA/source-loading precision addendum:** [gfx1151 and coherent-memory contract](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) distinguishes device address space/GTT max, driver reserved, host Linux MemFree, **MemAvailable**, cgroup limits/PSI, reclaimable clean source-file page cache, pinned H2D pages and physically resident model/draft/KV/SSM/lookup tables. On 128GiB-class coherent memory a buffered 80–100GiB model read can duplicate weight bytes in clean file cache before reclaim; must predict and preserve OS health reserve. Safe known-shard fadvise is optional/restricted, never global cache purge or default SIGKILL. First check physical headroom before GPU launch; model alias/source sharing and sidecars avoid duplicate resident bytes but *per-session* draft rings/states remain charged. QTIP-13 and GFX-09..11 are future physical stress gates.

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


## 1. Core architectural decision

Use **one authoritative memory-budget arbiter with specialized resource managers**. Model parameters, context state, experimental scratch, n-gram/Engram tables and disk I/O do not have the same lifecycle. A universal cache is wrong, but uncoordinated pools can fatally overcommit physical memory. This is especially critical for the 128 GB and 92/96 GB AMD Strix Halo targets.

This specification combines separately owned GPU/RAM/NVMe expert tiers, lease-safe fixed expert slabs, bounded immutable lookup-row IO, prefix-state cache tiers, hybrid snapshots and checked mapped quantized model sources. Named project provenance lives only in the prior-art atlas.

## 2. Distinguish addressability, residency and physical headroom

Measure and expose:
- installed and available physical DRAM;
- process RSS/working set, OS page cache and pinned/locked memory;
- device reported addressable VRAM, true separate VRAM, GTT/TTM and unified memory map limits;
- current HIP/CUDA/Metal allocations and driver-owned/staging bytes;
- cgroup/rlimit constraints, per-NUMA node memory and thread affinity;
- space needed for OS, file cache and untracked application load;
- host-to-device, device-to-host and peer bandwidth (possibly sharing same DRAM);
- local NVMe bandwidth, random latency and concurrency; filesystem/page cache influence.

**GPU virtual addressability is not free physical RAM.** For Strix Halo, the GPU may map over 100 GiB while `MemAvailable` and application safety margin are much smaller. Use conservative fail-closed preflight, dynamically recheck under external memory pressure and avoid mandatory unsafe firmware/boot tuning.

Illustrative budget (not fixed constants):
```text
safe_physical =
    max(0, MemAvailable_adjusted - OS_HEADROOM - safety_margin)
candidate_memory =
    mandatory_dense + mandatory_per_session_state + model_metadata
    + selected_expert_pool + IO_staging + temp_prefill + draft_weights
if candidate_memory > safe_physical OR any device limit:
    shrink optional pool / batch / context / prefill chunk in safe order
    else reject with exact shortfall
```

Use checked arithmetic and record request/lease context; never calculate with fixed total installed RAM and ignore other running processes.

## 3. Resource classes and default priorities

| Resource class | Owner / lifecycle | Demand characteristic | Placement principle |
|---|---|---|---|
| Dense model weights | model executable | read every layer/token | resident close to compute if possible |
| Shared expert and norms | model executable | active every applicable token | resident if beneficial |
| Routed MoE experts | model pool | sparse, skewed choices | dynamic cache by layer/class/cost |
| KV pages / sparse latent | session | grows with context | paged GPU then RAM/disk if valid |
| Recurrent GDN/KDA state | session | large mutable fixed shape | GPU resident active, frozen snapshots tiered |
| Sparse indexer history | session | grows/compresses with context | model-specific compressed ring/pools |
| MTP/draft weights | model option | only if activated | evaluate memory/benefit |
| Vision/audio weights | model option | sporadic | workload/budget tiered |
| N-gram/Engram table | large file | tiny row random accesses | disk primary; bounded row staging/cache |
| Prefill activation scratch | execution plan | short bursts | borrow from optional pools with leases |
| IO staging | device/IO plan | overlapping read/upload | fixed ring or slab with credits |
| Persistent prefix snapshots | prefix cache | shared immutable reads | GPU/host/disk hierarchy |

## 4. Authoritative resource arbiter

Proposed Rust interfaces:

```rust
struct ResourceEstimate {
    mandatory_bytes: u64,
    optional_bytes: u64,
    peak_scratch_bytes: u64,
    pinned_host_bytes: u64,
    device_addr_bytes: u64,
    gpu_compute_bytes: u64,
    num_sessions: u32,
}
enum ResidencyTier { Device, Unified, PinnedHost, PageableHost, FileBacked }
struct BufferLease {
    allocation_id: u64,
    generation: u64,
    tier: ResidencyTier,
    bytes: u64,
    /// physically released only after all referenced device/io events
}
trait ResourceArbiter {
    fn can_admit(&self, plan: &ResourceEstimate) -> Result<Reservation>;
    fn reserve(&mut self, plan: ResourceEstimate) -> Result<Reservation>;
    fn reclaim_optional(&mut self, target_bytes: u64) -> Result<ReclaimReport>;
}
```

These are **coarse global admission** sketches. In v4 real code, a DeviceBackend local allocator draws from an approved reservation and issues fast generation-tagged RAII leases with event-backed delayed release; the central ResourceArbiter must **not take a lock per token/kernel allocation**. Memory pressure notifies the coarse arbiter, which revises quotas/optional pools at safe boundaries. A reservation is a *cap*, not a guarantee that `malloc` cannot fail; execution must catch allocation failure and abort safely. A requested expert-cache size is not a physically guaranteed allocation.

Admission order:
1. validate artifact and desired context/modalities/quant; estimate mandatory tensor/state weights;
2. query physical/GPU topology and current use;
3. reserve OS/device headroom and mandatory buffers;
4. consider batch/session limits, context length, prefill chunk;
5. allocate remaining optional expert cache/draft/cache state based on expected benefit;
6. compile/capture plan with compatible stable pointers;
7. admit only when all mandatory leases ready; expose clearly chosen sizes and residual headroom.

On growing context or concurrency, rerun budget and shrink optional caches as legally possible; if mandatory state cannot fit, reject new request or provide explicit user-facing adjustment, not corruption.

## 5. Expert store organization

**Immutable backing source:** model file/GGUF shards, metadata offset/stride/type per `(layer, expert, projection)`, SHA/tensor manifest and optional imatrix/quant config. Read-only at runtime; page-aligned when useful.

**Mutable resident table:** `ExpertKey = (model_id, layer, expert_id, quant_layout_id)`; maps to slot `(allocation_id, generation, type/class, device addr, valid_until_event)`. GPU sees a compact current table for the model/region. The table should be updated atomically or via ordered generation publication so kernels never see new IDs with old pointers.

**Slot layout:** fixed-size class/layer slabs with deterministic padded expert sizes rather than allocate/free 11 MiB blobs per miss. Policies differ by backend: on PCIe discrete, pinned RAM can hold complementary expert set and CPU vector lane; on UMA, extra pinned copy often wastes scarce bandwidth and memory. Per-layer partitioning protects Maya-like hazards, but can reduce flexibility; measured capacity tradeoff vs pooled slots with leases.

**Residency policy:** use history (LFU/LRU/skew), miss and transfer cost, model layer/projection differences, multiple session working set and available memory. Preload known popularity profile where valid; dynamic adaptive predictions are hints not correctness. Correctness MUST NOT rely on expert being predicted. Pool can fill completely with all experts on large unified memory: avoid reserved idle spares unless needed for safe DMA replacement.

## 6. Fast resident expert path

```text
GPU router:
  route selected IDs & weights
  read device-resident slot table with generation validation
  group token rows by expert
  compute expert matrices from resident pointers
  aggregate results
  publish layer state/output event
```

No CPU copy of selected IDs, no per-layer host round trip, no global device synchronize merely to read routing. If all selected experts reside, CPU service thread can be asleep. Pointer table must be legal under captured graph replay (device-side data rather than stale baked host pointers, depending on graph design).

## 7. Expert miss path and async event protocol

```text
GPU router detects missing id(s)
  -> publish request (model/layer/IDs/expected table generation) to host-mapped ring
  -> safe dependency wait or exit current graph island
host IO service validates source + slot availability
  -> acquires lease on victim slot only after all readers/queued work are quiesced
  -> disk/pinned host read with checksum/length check (can use concurrent pread)
  -> async upload or CPU lane compute if measured superior on this topology
  -> update slot table + publish completion with release/acquire memory ordering
GPU resumes gated compute after valid completion, or aborts on failure/timeout
```

The GPU cannot spin indefinitely if host crashes or SSD fails; design bounded wait/progress counter and host-side cancellation/timeout recovery. Host and device memory ordering must be tested for HIP/CUDA specifically. Ring overflow, duplicate requests, eviction races and changing model/session epochs must be handled.

Do not over-evict active expert weights because a victim slot is being read by *another concurrent session*. Use generation leases and event completion, not only "victim from same layer" assumption, when batching across sessions.

## 8. Async SSD and NUMA

Options `mmap` + OS page cache, `pread`, `io_uring` and direct IO only after testing alignment/filesystem/queue behavior; portable fallback to bounded synchronous file reads on a worker pool is acceptable when async primitives unavailable.

For large prompt prefill, aggregate selected experts by layer and issue future reads while current layer is computing. Use multiple staged buffers only if their occupied RAM and queue depth yield net benefit. For decode, random expert misses are critical path; priority should favor current token and a known next layer.

NUMA-aware pinned memory placement and CPU pool threads can reduce PCIe costs on multi-socket machines. On Strix Halo, NUMA complexity often unnecessary; physical memory contention dominates.

Never automatically turn arbitrary model paths into network fetch operations. Downloaded model identity and checksums verified separately.

## 9. Disk-only lookup row service

Qwen n-gram and DeepSeek Engram tables are not giant MoE expert matrices. They require small indexed row reads, often BF16/native FP8 reconstruction. Data structures:
`LookupIndex{artifact_sha, tensor_offset, row_size, row_count, encoding, hash_table_schema}`
`LookupRequest{session_epoch, token_positions, row_ids, expected_generation}`
`LookupResult{verified_row_ids, staged_bytes, event}`.

Hash and row selection must match official model algorithm exactly (including history/window/padding). For repeated rows, deduplicate reads without changing output order. Use a bounded read cache sized by measured locality and physical RAM, not file size; on corruption or failed read, abort dependent transaction. Cache invalidation by model artifact/version, not file name.

DwarfStar Qwen Q2 example keeps ~95.37 GiB BF16 n-gram on disk while 41.73 GiB main/MTP weights can be resident; different model quant/quality than a closed tuned reference's lookup table. Never claim both have identical bandwidth need.

## 10. KV, prefix and recurrent state tiers

Attention KV may use page-size/batch optimized layout. Full context, sliding rings, latent compressed MLA and DeepSeek compressed history each have distinct address formula and eviction safety. Prefix cache uses immutable snapshot key and may share page memory across sessions; mutable tail COW protected. Recurrent state snapshot sizes and costs need explicit budget; offload only frozen copies, not in-flight mutable buffers. Offloaded state can often be cheaper to recompute from prefix than read from disk; scheduler compares costs.

GPU/host/disk movement is not always supported for every state type. Unsupported-tier fallback must be explicit. Never reinterpret expanded KV as latent KV. Do not evict a prefix still referenced by sibling session without COW/immutable ownership.

## 11. Dynamic scratch borrowing

Prefill may need GiB of temporary storage not used by decode. Arbiter can borrow cold optional expert cache capacity, but requires:
- prove no in-flight kernel holds victim leases;
- evict/demote optional experts safely;
- allocate bounded scratch with state generation and exact lifetime;
- run prefill;
- release scratch after GPU event;
- optionally prewarm experts for decode if justified;
- preserve mandatory state and OS headroom throughout.

Borrowing cannot be implemented as blindly mapping two overlapping buffers. Compare borrowing time/eviction cost versus smaller prefill chunk and reduced GEMM efficiency; select by measured benefit. Avoid second full scratch allocation if model can logically use tail of pool under strict exclusivity.

## 12. Safety and deployment

- default conservative memory budget, no OS-wide overcommit;
- warn on swap pressure; avoid unsafe `mlock` entire model;
- cgroup awareness and bandwidth competition;
- Linux `MemAvailable-CmaFree` adjustment where hardware kernels/CMA affect reporting; don't assume same signal on other OS;
- optional watchdog monitors physical memory threshold and aborts benchmark/load before host instability;
- protected cached prompt/session data on disk; user opts into persistence;
- pack source files immutable and checksum validated;
- model shards, staging/temp files placed under controlled paths with safe permissions.

## 13. Required test matrix

Synthetic 0/1/8/288/512 expert routes, duplicate and skewed routing, cold/hot alternation, cache sizes 0/1/half/all, forced corrupted SSD/read-only model, missing block, ABA slot reuse across concurrent requests, draft rollback during cache eviction, prefill scratch borrowing, context growth and server cancellation. Measure OOM-safe refusals. Physical tests on discrete PCIe, 92/96/128 GB UMA if available, and at least one different backend; record memory pool telemetry and clean teardown.

## 14. Known research hazards / unapproved constants

Default cache % (DwarfStar reports 80% safe in one ROCm setup), 1 GiB reserve/16 GiB headroom (Maya-specific examples), exact slot counts, pinned pool size, sparse top-k and prefill batch size are *reference configurations*, not portable zerogravity constants. Decide through measurements, hardware checks and explicit tested defaults.

## 2026-10-10 integrated v2 addendum — research-grade KV/expert algorithms governed by CIR

Adopt **research expert-tier specifics** as measured policy candidates: size-class slabs, per-layer allocations/ceilings, fill-only startup, decayed LFU+recency protections, profile-derived hotlists, device-side resident pointer table, pinned staging/O_DIRECT/io_uring vs mmap/read-ahead, one-blob expert source and host `publish` gate. Historical values such as 16 workers, 64 routes, 96 swaps, 20GiB reserve, 2ms coalescing and 3 misses are **research configurations**, not universal defaults; derive per machine profile by physical proof and source license.

**Allocator layering:** global `ResourceArbiter` controls physical GPU/UMA/CPU/OS/pinned reservations; separate `KvPool` implementations (paged baseline and contiguous extreme), expert slots, ngram/Engram disk row cache, prefix snapshots, encoder cache and scratch are all **clients**, each with domain-local eviction rules and global accounting. Radix prefix index points into the same paged pool; it must not create its own hidden allocator. The smallest valid cache frontier for hybrid models comes from `StateGroupCodec` intersection, not plain KV page count.

**CIR effect/timepoint integration:** `Acquire` and IO staging requests consume budget; `ReadArtifact` verifies immutable source identity and exact length; `Transfer` gets device event; `PublishResidency` updates slot pointer/generation only after all readers quiesce; `Dispatch` owns read leases; `Release` waits event. In a fully resident MoE decode, no per-token host route read. Miss path has bounded host ring and deadline; no GPU spin forever on IO failure. State snapshots are frozen session context and never treated as interchangeable model expert blobs.

**Policy decision:** host-visible mapped memory on Strix Halo shares LPDDR bandwidth with CPU; "promotion" from host to GPU may be a registration/alias change, **not** more physical RAM. Dynamic CPU expert lane only if measured net benefit. SSD row lookup table of 95/189GiB remains indexed file/optional tiny cache, not mandatory GPU residency. Scratch borrowing legal only when optional victim pool has no live leases and OS headroom remains valid.

**New experiment:** compare complete CIR scheduling of next-layer prefetch, resident/streamed expert grouping and prefill scratch reduction against handcoded host-stage baseline; measure critical-path latency and memory peak under cold/hot SSD and concurrent sessions. No claim of true asynchronous benefit until driver/event traces show overlap.


## 2026-10-12 integration — pinned AIE/NPU buffers and shared physical ownership

Extend global resource inventory for **XRT device/host buffer objects, GPU-mapped pinned host allocations, immutable prepacked NPU GEMM weight slices, A/B/C staging, instruction/program memory and outstanding DMA**. The Gufo fork imports `hipHostMallocMapped` pages into XRT user-pointer BO; an isolate mode copies to private XRT BO. Neither technique adds physical RAM: aliases may refer to the same LPDDR and **all pinned/duplicated allocations count toward OS-safe admission**. Shared-mapped memory requires explicit cross-API synchronization, not just a shared virtual address.

Each buffer has `(allocation, generation, owner, GPU/XRT mappings, last read/write event)` and stays pinned until both runtimes acknowledge quiescence. A timed-out NPU command can still be writing; quarantine the buffers or drain/reset correctly, never `free` based on elapsed time alone. NPU programs and weight slices may consume enough shared memory to reduce KV/expert/cache capacity, so cost planner chooses GPU-only when net TTFP/latency or safe physical memory deteriorates. The NPU can use its own program instruction BO and up to one driver-supported queue depth; do not share mutable staging inputs across active groups.

Mapped NPU buffer ownership is distinct from discrete-GPU/integrated-GPU *routed-expert ownership* and from SSD-backed n-gram tables; all coexist under one ResourceArbiter/ExecutionTopology without one universal residency policy. New `NpuBuffers` are an optional resource class, not a copy of model KV or sampler state. Early capability check can disable offload with no change to model semantics. Details in [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

## v4 refinement — reservation authority versus local pool

`PhysicalResourceArbiter` owns **coarse admission, peak bounds/headroom, tenant/model/session quotas and reclaim decisions**. Each `DeviceBackend` owns efficient **local suballocation** from an approved reservation: slab/page allocator, pinned buffers/IO staging and async event-backed retirement. A central mutex may be used for rare admission/rebalancing operations but **not for each emitted token/GEMM scratch tile**. Every allocation maps to an accountable reservation and BufferLease with generation, live readers and writer/completion event; optional cache eviction cannot touch active state.

Physical RAM/GTT/driver memory changes outside our control, so guarantee conservative engine admission + safe allocation failure and reclaim, **not** absolute no-OS-OOM behavior. Use local pressure watchers and release optional cache before changing mandatory task state. Imported GPU/NPU mapped buffers count physical RAM once per aliased allocation but require **both** API event/fence retirements; a timed-out DMA stays quarantined. A specialized contiguous cache can exist next to paged state pool under same global budget without a new model serving engine.

## v4.1 — weight/program residency spans GPU, graph and static dataflow

ResourceArbiter accounts **compiled program images**, graph code cache, on-chip SRAM scratch, device HBM/VRAM, shared coherent LPDDR (actual OS/headroom), pinned host staging, remote accelerator reserved quota, program warm/load cost, runtime collectives and streamed model weights independently. Native GPU kernel residency differs from **compiled graph program residency** and a **statically scheduled weight stream**; none should be forced to emulate page-by-page local KV. Static fabrics may keep compute program on chip while streaming immutable model layers, whose source, stage and transfer cost are explicit in CIR.

A `DeviceBackend` may own opaque physical handles that cannot be exported as native pointers; state snapshots must use typed `StateBridge` codec/replay or the region restricted to stateless/no-fork modes. The logical StateRoot remains authoritative and old versions unmutated before coordinator PublishRoot. A device with static on-chip memory and limited dynamic allocation requires **compiled reservation and bounded shape policy**; OS/device pressure and program upload must be gated before dispatch. Compiled device blobs and raw physical/collective handles are ephemeral and never serialized into a persistent prefix cache.

Coherent local host+GPU systems share physical RAM/bandwidth even if pointer alias works, while HBM rack GPUs and SRAM wafer arrays have different reuse/transfer costs. Reuse one hierarchy of global admission→device-local allocations and on-device compiler-owned layouts, **not** one cache policy for all. See [backend contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## v4.2 — one physical accounting discipline across remote model, state and cache owners

Global ResourceArbiter is a **logical aggregate quota/admission authority**, not an all-nodes one-pointer allocator. Per-node ResourceArbiter verifies its own DRAM/HBM/UMA/pinned/SSD/OS headroom, issues local reservation tickets and fast DeviceBackend pool leases. A distributed PlacementPlan must reserve **all** mandatory per-node regions, staging and transfer credits *before* launching a multi-peer collective/state migration; workers report real pressure and independent failures. Aggregate cluster memory capacity is only capacity if the transfer topology and required state/compute semantics allow the shard.

Track four distinct owners: immutable ParameterPlacement (dense weights, experts, adapter, lookup source), mutable StatePlacement (KV/GDN/indexer), immutable CachePlacement (prefix/encoder/compiled program copies) and compute RegionPlacement, all with model/quant/tenant rights. Cache locality hints can be eventually consistent; mutable remote state version/lease cannot. A remote prefix “hit” that needs GB-sized hydrate over relayed WAN may be worse than recompute, so physical IO/crypto/RTT and latency participate in admission/cost, not just byte presence.

State and expert transfer resources have bounded staging, per-flow priorities/credits, checksum/length and lease event completion; TCP/QUIC write acknowledgment is not GPU DMA quiescence. Trusted RDMA only on physically eligible registered memory, with scoped keys revoked/fenced. A lost tunnel cannot free in-flight DMA or release authoritative state; quarantine and recover from compatible committed checkpoint if possible. [Distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md).

## v4.3 — multiple resident models under one physical ResourceArbiter

The ModelCatalog may describe very many registered immutable artifacts **without allocating GPU/system model weights**. `ResidentModel` deployment is admitted by PhysicalResourceArbiter for mandatory dense/shared/routed weights, adapter/draft/encoder, compiled program images, KV/SSM/indexer/session headroom and scratch/pinned buffers **across all selected local/remote devices**. Separate **replicas of the same semantic model** need their own actual physical bytes unless device/model source sharing is explicitly proven; one source model **sharded across devices** is one deployment, not N replicas.

Life cycle: verify→reserve→compile/load→warm→ready→idle/draining→evict after all outstanding TaskSession/DeviceWorker/StateRoot/IO/DMA generation leases are quiescent. `RegisteredModel` remains available after unload. Model prewarm/pin/keepalive/scale-to-zero policies are soft except an explicitly admitted hard pin under quota; no unlimited resident model promise and no unsafe GPU virtual GTT RAM inflation. Before unloading idle source weights, ensure any desired restorable session state has a valid frozen full-state snapshot/replay or a clear session expiration policy; **never evict an active unsnapshotable session** to serve another model.

Cold-load and model-swap cost is an explicit admission/scheduling term. Prevent thrash from alternating huge models via load dedup, hysteresis, warm TTL, prewarm priority and per-tenant fairness. Evict optional model/prefix caches and safe cold idle models before refusing, but don't silently switch source quant, trained top-k or route to another model due to OOM. A model may use multiple GPUs or authorized remote nodes, whose reservations/tenants and state/weight/cache placements are already governed by v4.2. [Router contract](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).

````

## Source: .agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md

Original UTF-8 bytes: 27943; SHA256 `72a1f1941239b77e2dc3c0a0eb08d505d024e819a37ec49c777aeb94f25aa845`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Minimal execution spine v4 — state, asynchronous workers, resource ownership and hot-path performance

**Normative technical contract · 2026-10-12 · NOT IMPLEMENTED.** Read together with `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md` and the v4 master/reference architecture. This document resolves contradictions in the older `StateDomain::commit`, `Command::Commit`, one-owner/one-pool and worker protocol sketches. All proposed Rust signatures are illustrative **internal types**, not frozen public APIs or a promise of a full distributed database.

## 1. Exactly five ownership boundaries

**Immutable semantic owner:** trusted ModelDefinition + immutable source artifacts and typed semantic graph. It defines mathematics, state schemas and allowed layout conversions. It cannot allocate device cache on an unbounded basis, send service responses or choose architecture-specific code by model name.

**Compiled plan owner:** Compiler creates verified immutable `ExecutableBundle` from source semantics, task execution regime and `KernelCandidate`; it contains kernel handles, shape guards, resource command template, fallback edges and cache provenance. It does **not** own session frontiers or live devices.

**Logical session owner:** `TaskCoordinator` is the **single writer per TaskSession or per bounded session shard**, not a mandatory single global process-wide thread. It owns `TaskDriver` state, request queue/admission intent, current committed `StateRoot`, cancellation/attempt epoch, output sequence and policy. It alone publishes a new root and output.

**Physical resource authority:** `ResourceArbiter` decides globally admissible physical RAM/device memory/IO headroom for model sessions/tasks and outstanding draft/copy buffers. It issues **reservations at model/session/step boundaries**, rather than routing every GPU kernel temporary allocation through a central lock. Each DeviceWorker owns a **fast local subpool/suballocator** consuming a pre-reserved budget with generation-tagged leases; cache policies are independent clients.

**Device execution authority:** DeviceWorker/IOWorker owns context, code/graph handles, stream queues, buffer mapping and completion events. It executes direct physical work and **never mutates a user's committed state root or emits an unverified TaskResult**. Its reported completion cannot itself commit model semantics; it can only make a provisional candidate ready.

These boundaries are the minimal answer to the original question: *what changes when a new mathematical op, task, target device, memory tier or fused specialization is added?* There should be exactly one primary owner to extend for each.

## 2. State versioning — prepared, not independently committed

### 2.1 The core error to avoid

The v3 sketch allowed `StateDomain::commit(candidate)` returning a new handle while the surrounding narrative required an **atomic whole-session transaction** across KV, GDN/KDA, sparse index, n-gram, grammar, sampler/RNG and task state. Per-domain `commit` that mutates visible buffers immediately **cannot** provide such atomicity. Similarly, a `CIR Command::Commit` cannot publish logical session state while the host TaskCoordinator also claims that authority.

**v4 rule:** each domain **prepares** a candidate state; only the session owner **publishes one immutable versioned StateRoot**, after required device events/effects are ready and all domain versions agree on the same frontier.

### 2.2 Illustrative states, epochs and root ownership

```rust
struct SessionEpoch(u64);
struct StateVersionId(u128);
struct TaskFrontier {
    kind: FrontierKind,   // tokens, audio frames, denoise iteration, etc.
    value: u64,           // validated monotonic within this kind/epoch
}
struct StateRoot {
    version: StateVersionId,
    session_epoch: SessionEpoch,
    frontier: TaskFrontier,
    domains: Vec<DomainVersionHandle>,  // all required domains + generation
    provenance: SemanticAndNumericKey,
}
enum DomainWork {
    Prepared(PreparedDomainVersion),
    NeedsReplay(ReplayRequirement),
    Rejected(StateError),
}
struct PreparedCommit {
    epoch: SessionEpoch,
    from_root: StateVersionId,
    proposed_frontier: TaskFrontier,
    domain_versions: Vec<PreparedDomainVersion>,
    ready_events: Vec<EventId>,
    output: Option<ProvisionalOutput>,
}
trait StateDomain {
    fn snapshot(&self, root: &StateRoot) -> Result<SnapshotHandle>;
    fn fork(&self, root: &StateRoot) -> Result<ProvisionalDomain>;
    fn prepare(&self, provisional: &ProvisionalDomain,
               frontier: TaskFrontier) -> Result<DomainWork>;
    fn abort(&self, provisional: ProvisionalDomain);
    fn restore_or_recompute(&self, source: &CanonicalHistory) -> Result<ProvisionalDomain>;
}
```

**Logical operation:**
1. `Begin(epoch, read_root)` pins immutable snapshot/reference and reserves provisional buffers/state from global budget via local subpools.
2. Compiled kernel/TaskDriver stage executes against the snapshot/private COW tail, generating `ProvisionalDomain` and provisional output. GPU native speculative steps may generate many candidate frontier versions without publishing them.
3. `Prepare`: all required domain state versions have the same `frontier`, schema and valid numeric/position rules; CIR events, copy/IO and device writes are either **Ready** or subject to explicit compatible replay. If any invalid, `Abort`, keep committed root.
4. `PublishRoot`: the sole logical session writer checks epoch and expected old root (CAS-like **host logical** publication) then swaps the entire vector of prepared domain version handles as one root. **No independent domain becomes visible earlier.**
5. Emit committed outputs with monotonically increasing stream sequence. Retire old root/state buffers only when readers' leases and last physical device events are done.

This is **atomic visibility of logical state metadata**, not instantaneous physical atomicity of all GPU devices nor an automatically durable multi-host transaction. Physical mutable pages must use COW/private tail or an undo/recompute guarantee; *mutating the last committed page in place while GPU work can still fail is forbidden unless safe reconstruction proven*. Logical O(1) root swaps do not mean O(1) recurrent matrix copy.

### 2.3 On GPU-only decode, how to avoid per-token host barriers

A GPU-native `StepProgram` can compute multiple dependent row/token iterations in a **closed compiled execution island**:
- model forward, routing, penalties/grammar mask where GPU-qualified, sampler RNG, state update/verification, provisional token buffer;
- controls: max iterations, accepted frontier, device EOS/stop flag, legal interruption point and bounded output/step memory;
- device-side input buffers and index maps; run captured graph/loops without host fetching each router or draft token;
- explicit final `CompletionRecord` and `PreparedStateRoot` metadata sent to host when necessary for streaming/admission/next plan choice.

Not every model or sampling/grammar implementation can execute multiple steps on GPU. On those, the correct plan uses smaller islands or host transitions and reports mode; it must **never** speculate past incompatible grammar/RNG/cancellation state. A single target sample visible to user still requires a logical publication/stream decision, but this need not block the next independent GPU stream if device supports safe double-buffering and epochs.

A two-in-flight host scheduler is *optional* and only legal where step `N+1` inputs don't secretly depend on state or tokens not committed from step `N`, or the execution island handles that provisional dependency internally.

## 3. Worker step protocol and backpressure

```rust
struct WorkEnvelope {
    session: SessionId,
    attempt_epoch: SessionEpoch,
    expected_root: StateVersionId,
    bundle: ExecutableBundleId,
    shape_class: ShapeBandId,
    rows: RowBindings,            // stable slot indices + generations
    leases: LeaseBatchId,         // globally reserved/locally realized
    dependencies: Vec<EventId>,
    output_quota: WorkOutputBudget,
}
enum WorkerReply {
    Submitted { ticket: TicketId, event: EventId },
    Complete { ticket: TicketId, events: Vec<EventId>,
               candidate: ProvisionalCandidateId, stats: StepStats },
    Failed { ticket: TicketId, reason: WorkerError,
             may_have_live_dma: bool },
}
```

These types can be compact, pooled fixed-capacity tables rather than dynamic Vec in hot code. WorkEnvelope represents one precompiled region/stage assignment, **not an IR graph copy**. DeviceWorker maintains persistent batch slots and updates only deltas. Every slot/lease/ticket carries generation and session epoch; late duplicate worker completions are discarded, but buffers remain alive until actual device event/cancellation drain.

**Backpressure:** ingress bounded by request memory and timeout budget; event queue separates control/cancel from output chunks; output buffers bound by client acknowledgments or credits; device memory quotas allocated ahead of submission. A slow HTTP client cannot pin infinite GPU KV or an unbounded PCM/video queue. TaskCoordinator can continue unrelated sessions while one session is waiting on SSD/NPU/remote IO.

**Errors:** typed `UnsupportedShape`, `UnqualifiedDevice`, `StaleEpoch`, `BufferLeaseInvalid`, `DeviceLost`, `IoShortRead`, `StateIncompatible`, `OutOfBudget`, `OutputSchemaInvalid`, `Cancelled`; attach retry classification. Never "retry" failed partial output with incremented root blindly. All physical failures map to a valid last committed frontier or a typed unrecoverable response.

## 4. Device/command graph boundary

**Only these runtime primitives are mandatory initially**:
`Reserve`, `AllocateFromPool`, `Submit`, `Event`, `WaitAtFirstConsumer`, `Ready`, `Quarantine/Release`.

The planner may represent physical work in CIR commands:
`Acquire/Copy/Import/Pack/Dispatch/Wait/Join/PublishResidency/Release`. A semantic `Commit` is **not** a free-standing device-visible state publication command: CIR may provide a `BarrierBeforePublication` and an event proving all required writes are complete. TaskCoordinator still owns `PublishRoot`.

A CIR plan has typed buffer aliases and explicit availability/failure dependency DAG. Host↔device events, CPU/NPU/SSD async IO and captured GPU streams may require different primitive mapping; the semantic result is not ready merely because the host submitted a callback. Device-specific constraints are checked at compile time where known and at step binding for dynamic buffer generations/capabilities.

**Cross-accelerator equivalence:** A generic tensor op split into independent output rows may overlap two compute devices; routing expert subsets across discrete/iGPU may overlap. In both cases `Join` waits on true first consumer; source quant and model top-k/position/state remain unchanged. GPU+NPU power/firmware qualification can **veto** an otherwise legal computation plan. No root-clock/suspend/IOMMU changes automatically.

## 5. Hierarchical physical budget and local allocation (zero global per-token lock)

```text
Hardware/TopologyProbe + conservative OS/cgroup physical RAM snapshot
              |
     PhysicalResourceArbiter (global admission)
       reserve permanent weights and task/session peak
       reserve per-device subpool budget, tenant fairness/headroom
              |
  Device A pool       Device B pool        Host/IO pool
  fast slab/pages      async allocator      pinned/SSD staging
  local generations    local generations    local generations
       \                  |                    /
        completion/retirement events → release global reservation
```

**Global owner** controls who has reserved capacity and handles low-memory pressure/eviction coordination at coarse boundaries; it does not synchronously allocate every GEMM scratch tile. GPU/CPU/NPU contexts manage their own fast local pools with explicit accountable reservation ticket. Resources include model weights, KV/recurrent/indexer/prefix, expert slots, adapter/draft, encoder/denoiser scratch, graph programs, transferred activations, PCIe and pinned DMA, storage IO cache and OS reserve.

**No impossible OS promise:** concurrent processes, kernel/driver pinned allocations, compaction, driver arena growth and temperature/power may change available capacity after probing. Admission with safe headroom + failure handler + cgroup metrics is the guarantee; a model/task requiring memory cannot be admitted if actual capacity/pressure cannot be bounded. Device allocation can fail safely, and optional cache released/replay model if appropriate. Do not assume virtual GTT = free DRAM.

**Allocator contract:** local `Allocate(reservation,shape,layout)` returns `BufferLease{allocation_id,generation,range,tier,readers,writer,last_event}`, no silent in-place alias conflicts. `Retire` queues asynchronous release after final reader fence. NPU/remote DMA timeout may be **unsafe to free**, so retain/quarantine/reset. Resource ownership doesn't grant state publication authority.

## 6. Model/task/plan compatibility keys

Keep distinct canonical identities instead of one giant runtime hash:

- **ArtifactKey:** exact source checkpoint/shards, tokenizer/template, adapter generation, encoder/projector/vocoder/media and quantization schema.
- **SemanticKey:** normalized operator equations/version, position and state effects, source weight logical roles, approved numerical mode.
- **TaskKey:** task control schema, sampling/grammar/voice/noise/stream semantics and external-action policy boundary.
- **StateCompatKey:** artifact+semantic+task relevant prefixes, full required state schemas, position/quant, adapter/sampler/tenant salt/medium fingerprints and snapshot generation.
- **BundleCacheKey:** semantic region/candidate, physical layout, shape band, device ISA, kernel compiler/runtime driver, numeric mode, eager/capture, graph pointer-layout and program version.
- **RuntimeBinding:** session epoch, current root, batch slot indices/generations, current expert table/prefix cache state, physical reservation. **Do not include runtime binding in compilation cache key** or variant explosion follows.

Persist source blobs, schemas and state snapshots with versioned checked formats; **never serialize raw pointers, device events or compiled driver object handles** in a durable model snapshot. Bundles are expendable caches rebuilt on toolchain/driver upgrade; a state snapshot is separately schema-versioned and may require explicit converter/replay.

## 7. Logical output commit versus transport delivery

A TaskSession publishes valid `StateRoot` before making emitted `OutputEvent(session_id,epoch,output_seq,kind,payload_ref)` visible. For real streaming protocols:
- **Default semantics:** ordered, **at-most-once live emission attempt** per connection, no false promise of exactly-once client receipt on disconnect. An app may have received an event whose acknowledgment was lost.
- **Optional resumable delivery:** client cursors and a bounded/durable replay buffer, with explicit identity/retention/auth/privacy and idempotency. A resumed stream can supply already committed events with sequence numbers, but the client must deduplicate.
- **Cancellation:** previous complete committed events remain valid, pending tentative state aborted, leases retired after physical quiescence; media stages have their own "complete chunk" boundary.
- **Distributed:** each shard/rank stages provisional results under one coordinator epoch, sends `Prepared` with actual device completion, publishes authoritative root only after the required cohort ready. Rank timeouts and lost ACK require **fencing generation and recovery policy**, not assuming instant all-rank atomic memory. A durable exactly-once guarantee needs separate log/replay consensus and is out of inference MVP.

The TaskCoordinator may use application-level exactly-once identifiers, but it doesn't control an external network consumer that disappears before ACK. This is a correctness and honesty requirement, not a performance feature.

## 8. Subtask control without a giant workflow runtime

The core TaskDriver is trusted Rust orchestration at **stage-level/loop-level**, not arbitrary executable user-defined program. A stage invokes a compiled semantic bundle and specifies `NextStage` or `Repeat` plus resource/commit guards. Its `poll`/completion transition may allocate/submit to a worker, but steady token or denoiser arithmetic can run in device-native StepProgram without repeating full TaskDriver graph search for every math op.

Task examples:
- language: tokenize→prefill→decode/verify→commit tokens; host transition at true sampler/tool/fairness boundary;
- encoder/pooling: batch→encode→pool→commit vector;
- audio/TTS: window/acoustic frame loop→vocoder→PCM chunk commit;
- image: denoising step loop→decode→image emit; intermediate latent private;
- video: frame/window stage, temporal cache and mux output commit;
- **future training**: gradient/parameter update task with stable source `ParameterVersion` and optimizer state, separate authorized update/weight publication; not built into ordinary LLM decode.

External tool execution and multi-agent workflows belong above TaskRequest in a separately permissioned application scheduler; they may invoke TaskPrograms but cannot be turned into arbitrary kernel effects.

## 9. Failure semantics matrix

| Event | Logical response | Physical response | User-visible behavior |
|---|---|---|---|
| kernel returns UnsupportedShape before launch | choose prepared generic plan | no new device work | mode receipt fallback reason |
| device fails during a provisional recurrent update | abort new root candidate; replay if safe | wait/drain/quarantine dirty pages | no emitted failed token/frame |
| NPU/IO timeout with possible active DMA | discard candidate and mark provider unhealthy | quarantine buffers or safe device reset | GPU-only future request if safe; no false auto recovery |
| client disconnect during stream | retain committed root to defined session policy; cancel pending | retire leases after fences | previous emitted sequence may have been received |
| unavailable memory after reserve | refuse or evict optional cache/replan from root | free safe retired buffers; don't OOM attempt | typed budget/fallback |
| mismatch artifact/adapter/quant/task schema | no shared state/cache/plan mutation | invalidate incompatible cache handles | report state compatibility failure |
| remote rank lost after prepare | stop commit until epoch fenced/recovery policy | drain or discard provisional peer work | no partial new output |
| model/device program source untrusted | refuse execution | no device load | typed untrusted artifact |
| device power/firmware profile unqualified | GPU/CPU compatible plan | never auto-tune privileged system settings | mode receipt unavailable device |
| approximate numeric plan not authorized | return strict/generic candidate | no lossy input packing | unchanged model/sampler quality contract |

## 10. First vertical slice acceptance for this document

Required tests:
1. Mock two StateDomains + RNG; second prepare failure leaves exact original `StateRoot` visible and no output emitted, no UAF.
2. Fake CUDA/HIP/XRT and CPU streams with delayed out-of-order completion, stale `TicketId` and BufferLease generation; ready means correct memory visibility before physical memory release.
3. Per-device slab allocations under one global reservation; 32 in-flight sessions don't take one global allocation mutex per output operation; no unaccounted pinned buffers.
4. GPU `StepProgram` two dependent provisional target steps with *zero* host router lookup or mandatory sampled-token transfer where backend supports it; reference/quality and sampler state match.
5. Output disconnect/retry tests show stable event `output_seq` and no premature output; no assumption of network exactly-once delivery.
6. Shape-band guard selects legal candidate for 1,7,128,4096 and ragged inputs, with graceful generic fallback on unsupported exact tile; no unbounded JIT.
7. Forced specialized whole region disabled: session/task continues through generic path and no duplicate serving runtime.
8. Optional multi-rank protocol injects lost ACK/timeouts; logical writer cannot publish before successful prepared cohort and fencing rules.
9. Memory pressure while NPU pinned buffers potentially active: safely refuses/replans or quarantines without touching privileged driver power controls.
10. Time critical path end-to-end compared with simple handwritten direct-dispatch reference; bounded host CPU overhead and fast async event wake.

**Anti-goal:** there is no requirement to build all ten tests before one dense model run; implement 1–3 plus 7 first, then expand. Keep the system minimal; every new primitive must close a demonstrated correctness/performance gap.

## v4.1 addendum — graph/dataflow accelerator readiness and state bridge

A DeviceWorker is not necessarily a CUDA-like GPU kernel submitter. Its `ExecutableBundle` may contain `NativeKernel`, `CapturedCommandGraph`, `AheadOfTimeDeviceGraph` or a **StaticallyScheduledDataflow** program, each with a capability-checked `ProgramLifecycle`, memory residency and external event/fence interface. **One submission can represent a whole compiled model stage or statically scheduled accelerator network**, with device-internal tile/collective schedule delegated to the compiled target program; host TaskCoordinator should not simulate its internal GEMM operations.

**State is the limiting correctness contract:** If a TPU-like graph compiler or remote wafer program owns opaque recurrent/KV/sampler data, its adapter must produce a versioned **provisional** state candidate, support explicit snapshot/restore/recompute or declare a stateless-only region. A service returning final text and hiding all state is **not** a qualified RegionExecutionProvider; it is an external TaskServiceAdapter with different semantics. TaskCoordinator remains the only `PublishRoot` authority after ready/fail event, per-rank epoch checks and correct effect schema.

**Physical resource accounting** includes compile-time code cache, warmed/running program image, on-chip SRAM allocations, HBM/UMA buffers, streamed external weight sources, graph execution scratch, transport bandwidth, inter-chip collective occupancy, and network/privacy/egress budgets. A static program's claimed memory **must be observed/probed**; aggregate rack memory is not one ordinary pointer. A short-lived stateless program can be pipelined with local GPU segments only where transfer and numeric/state layout bridges exist.

**Lifecycle:** `Compile→Load/Upload→Warm→Reserve→Submit→Ready/Failed→PrepareState→PublishRoot→Retire`. Compilation and upload can dominate cold start. Bundle caches are disposable, **no persisted device pointers**, bounded shape variants and support fallback on shape miss. On hardware firmware/SDK/driver upgrade invalidate executable fingerprints and state-snapshot compatibility as necessary, not the mathematical model. No assertion a native Rust TPU/wafer/LPU compiler exists until source/toolchain+hardware proof. Related full contract: [accelerator execution families](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## 11. v4.2 — per-session root publication across remote state shard owners

**A StateRoot may refer to local and remote prepared DomainVersionHandles** with StateShardGeneration, storage/compute owner, lease epoch and state compatibility. TaskCoordinator remains exactly one logical writer per session/shard across the entire cluster. Each RemoteRegionWorker validates model/quant/semantic/numeric/program/position and the expected StateRoot and PlacementEpoch before executing into **private provisional** state; it returns Ready/Failed plus a lease/fence reference. Device completion or network ACK is never a new visible session root.

**Publication sequence:** (1) pin old root and placement epoch; (2) reserve all destination state/resources/network credits; (3) execute closed regions or transfer compatible state; (4) require all participant Ready and matching semantic/frontier epochs; (5) one TaskCoordinator PublishRoot and sequenced output; (6) send idempotent CommitNotice/Release after completion to remote peers. Failure before publication retains old root; failures after publication require old retained checkpoint or correct compatible replay, not an unsupported exactly-once machine transaction. Expiring a remote shard lease or losing coordinator must not permit stale peer writers. If no durable coordinator quorum exists, **fail-stop/reconstruct** rather than promise HA.

A RemoteRegionWorker can own a layer's KV/SSM physical buffers for many tokens, so per-token cross-machine work transfers activation tensors at layer-group boundaries and **not** the complete cache. Attention may execute on the KV host with compact query/output exchange, or state may migrate once at a committed frontier. Remote full KV reads on a tunnel are generally a costly candidate. Typed leases/delayed DMA retirement continue across transport loss. A missing remote state snapshot cannot be faked by local text prefix hash.

Global physical budget remains coarse; authorized peers reserve locally and report headroom. Cluster capacity is not one globally coherent pointer. Program ownership stays in RegionExecutionProvider; control/metadata transport can be tunneled while physical data chooses another qualified path. **No new global distributed lock per device operation.** Read [remote state protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

## 12. v4.3 — model routing does not own token state or physical allocations

`ModelCatalog` resolves one immutable model/alias/profile revision at **admission**; `ModelRouter` chooses an eligible warmed or safely loadable resident `DeploymentId` and pins a `ResolvedRoute` for TaskCoordinator. **TaskCoordinator remains one logical writer per TaskSession**, DeviceWorker/RegionExecutionProvider remain owners of compiled kernel/graph execution, and ResourceArbiter remains sole physical admission owner; adding more models must not create an independent `EngineCore`/sampler/KV allocation authority per model.

An active session retains exact `ArtifactKey+SemanticKey+Quant+Tokenizer/Adapter/Template+NumericPolicy` in its StateCompatKey. Replica balancing across the **same logical source model** may route a new stateless request to a different model replica; stateful continuation sticks to a compatible state owner unless a full committed frontier snapshot/lease/placement migration is validated. Hot alias changes never rebind a running state root. Multi-model TaskProgram target/draft/encoder/vocoder stages hold separate registered model leases under one task's budget.

Model lifecycle controls run **outside the device-native StepProgram**: deduplicated source verification, physical reservation, loading/plan compiling/warming, ready→idle→drain→unload after quiescence. Idle weight eviction cannot discard a live unsnapshotable KV/GDN, in-flight speculative partial or DMA; memory stress may evict safe optional cache/idle models or queue/refuse. Registered catalog models need no resident allocation. One `run` invocation can create local in-process engine; `serve` wraps that same engine with API adapters. No per-token HTTP/gateway hop.

**Responses/client streaming** uses the same root-first, sequenced output/at-most-once *attempt* contract; API compatibility translates schemas, never commits its own state. Tool-call generation is data, not an external tool execution authority. [Router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

````

## Source: .agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md

Original UTF-8 bytes: 33807; SHA256 `fbadae7f647be6be10eed8413fb288fc8d74948e049f454a000a16f3dc29cad4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Built-in model router, multi-model residency and serving contract (v4.3)

Design requirements below. Implemented/tested scope and unfinished work are
tracked only in [the plan TODO index](../plans/TODO.md).


**15 Oct 2026 — P09 three-stage admission and multi-model scheduling:** [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) distinguishes (1) bounded API/tenant queued request and input-byte credits, (2) per-model admitted sessions, **queued prefill token work** and actual immutable source/private state resource envelopes, and (3) device active sequence/verification row/scratch/IO budget. Do not allocate full KV for queued requests, confuse model registration with physical simultaneous residence, or silently change requested quant/model to fit. Per-tenant fairness and stable model session binding with typed overload/latency receipts precede new work. One ModelRouter resolves exact model and legal replica **once at admission**, not every token/audio frame; compatible models may time-share one physical worker but must never be fused into the same tensor GEMM without source proof. SERVOPT-01..07/22/31 test these boundaries.


**15 October 2026 — RESIDENCY POLICY IS NOT MODEL IDENTITY:** [First-class SSD tiered materialization](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) allows one registered verified model to run from physical GPU/UMA/RAM/SSD backing without a second model alias or extra service. The built-in Router binds exact ModelAssembly/derived quant and numeric policy; the storage planner automatically prefers full residency when mandatory weights+state fit with OS headroom, else source/backend-qualified streamed experts/lookup rows, else typed resource/latency refusal. Immutable source tensors may share loaded/streamed cache across sessions under physical leases; user KV/draft/recurrent state remains separate. A model too large for RAM can be *capacity-feasible* yet fail its interactive latency target—report effective cache/SSD QoS and no hidden model downgrade. SSD support remains backend/format/physical evidence scoped; tests SSD-01..32.

**AUTHORITATIVE DESIGN · 2026-10-12 · NOT IMPLEMENTED.** The model router is a **core capability of one zerogravity runtime**, not another inference engine or a mandatory front proxy/process. Read alongside [minimal semantic/task architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), [full physical budget](MEMORY_RESIDENCY_AND_IO.md), [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) and [local CLI/API/config contract](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). Named source-project research is isolated in [attributed prior-art research](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md).

**User goal U-021:** a blissfully simple `run <model>` and `serve` experience, **plus** a built-in load-balancing multi-model router, many registered/available models, multiple simultaneously resident/active model instances where physical resources permit, per-model and per-request configuration, native local/distributed placement, an OpenAI-compatible interface, full control through progressive disclosure. Simple defaults are a product requirement, not permission for silent model/quant/quality change.

## 1. One engine, four separable entities

**A. RegisteredModel** — an immutable validated `ModelDefinition` and `ArtifactKey` with checkpoint/shards, stored quant, tokenizer/template, adapters/sidecars/model license, supported TaskKinds/semantic op schemas and default safe config. It may have *zero* loaded weights or compiled programs.

**B. ModelAlias/ModelProfile** — a human-readable versioned name/tag and optional **configuration overlay** referencing one RegisteredModel/adapter tuple or a separately named policy-defined model group. `chat`, `code`, `vision`, `embedding` can be friendly aliases, but exact identity and capability are always inspectable. An alias is **not** a physical model or source checkpoint. Defaults may include context, sampler, speed/quality, persistent resident policy, compatible draft/adapter, per-tenant access and placement preferences; never mutate source model data silently.

**C. ResidentModel / Replica** — an independently admitted, loaded and warmed physical execution instance of one resolved model and source/adapter identity, using an `ExecutableBundle` and a `PlacementPlan` across qualified devices/peers. Several replicas of an **equivalent logical model** may exist for load balancing or capacity. A single replica may be **sharded across several machines** (one distributed instance), and multiple replicas may share immutable weights/cache storage where physical identity and leases permit.

**D. TaskSession / Request** — per-user/task control and mutable full model state, current `StateRoot`, output stream and workload requirements. Multiple TaskSessions share one ResidentModel/weights/compiled plan but **cannot** mutate each other's KV/SSM/sampler/grammar state. The same TaskDriver/TaskCoordinator and physical ResourceArbiter handles CLI and HTTP clients.

**Do not conflate:**
- many **registered** models vs many **resident** models vs many **concurrent sessions**;
- one model **split across peer devices** vs multiple **replicas of the whole model**;
- a **model alias/profile** vs `auto` **model-choice policy**;
- a **model stage graph with multiple compatible models** (e.g. encoder/target/drafter) vs **routing one request to a different target model**;
- a **compiled KernelPlan/ExecutableBundle** vs model artifacts or session state.

## 2. Minimal architecture — routing above shared execution, never hot-per-token proxy

```text
CLI run / native SDK / OpenAI-compatible API / typed media API
                          |
              Parse + Auth + Request validation
                          |
                 ModelCatalog + ModelAliases
            (versioned artifact/profile/capabilities)
                          |
                ModelRouter / Admission
          exact identity or explicitly configured policy
                          |
            ResidencyManager + ResourceArbiter
           resolve/load/warm/replica/evict/memory
                          |
               RequestPlacementSelector
          (task/model/phase/shape/cache/peers)
                          |
            TaskCoordinator + TaskDriver
                (same as ordinary run)
                          |
           ExecutableBundle + RegionWorkers
            local and/or authorized remote peers
                          |
      Prepared full-state versions → one PublishRoot
                          |
      Canonical committed TaskResult / stream / receipt
                          |
             CLI, SDK or API adapter formatting
```

There is **no** second token scheduler, duplicate model-weight allocator, model-specific HTTP process or proxy-in-proxy requirement. The router runs at **request admission and safe model/session/placement boundaries**, not once for every token/attention kernel. The device-native StepProgram's hot path remains intact and protected by the same physical state/lease semantics.

For a small embedded use, `run` may create the engine **in-process**, with no daemon. A configured long-running `serve` process can reuse the same engine objects. An already-running **same-user trusted local service** may be reused by CLI only after an explicit/verified local socket/process identity handshake, not by blindly connecting to an unrelated HTTP listener. No auto-install systemd/launchd service or privileged background daemon.

## 3. Typed identities, versioning and naming

Illustrative, not frozen Rust API:
```rust
struct RegisteredModelId(ArtifactAndSemanticKey);
struct AliasId(String);                // human-facing name, tenant scoped
struct AliasRevision(u64);            // atomic config generation
struct ProfileId(String);
struct ModelInstanceId(u128);
struct ReplicaGroupId(u128);
struct ModelGeneration(u64);          // resolves immutable artifact and overlays
struct DeploymentId(u128);            // one physical PlacementPlan (may span peers)
struct SessionRouteId(u128);

struct ModelBinding {
    registered: RegisteredModelId,
    profile_revision: AliasRevision,
    adapter_key: Option<AdapterGenerationId>,
    tokenizer_and_template: SourceTextRenderingId,
    allowed_numerical_mode: NumericContractId,
}
```

Aliases resolve under **one atomic catalog revision snapshot at request admission**. Subsequent alias updates never change an active session model/adapter/template/quant or invalidate its current StateRoot without explicit new session/replay. Hot reload/prewarm uses a newly published alias version; old active sessions continue on their old immutable source/runtime handles until drained or migrated by a **compatible, qualified state transformation**.

**Model naming constraints:** path files, offline content-addressed hashes, optional verified registry references and friendly names are separately parsed. User names must not bypass filesystem allowed roots, tenant scope or pull-source/license policy. A model alias cannot point to an arbitrary external API host without a separately configured trusted external TaskServiceAdapter.

## 4. Distinct route decisions: source identity, replica, phase and application model selection

### 4.1 Exact-model request (default, no substitution)

If the user issues `run mistral-small` or an API request `{"model":"coding"}`, resolve `coding@revision` to its **exact RegisteredModel checkpoint/quant/adapters/profile**. Then select an eligible *execution replica/placement* implementing **the same semantic/numerical contract**. A router must not silently change Llama↔Qwen, Q4↔Q8, a LoRA overlay, tokenizer roles or JSON-schema semantics because one is faster/loaded.

Aliases can be modified by their authorized owner; stable per-session bindings and `ModeReceipt` show the resolved immutable source identity and actual backend.

### 4.2 Replica load balancing (same logical model)

Choose among **equivalent** loaded/qualified replicas using measured:
- request task/operator/shape/context and requested output/mode;
- queue lengths, workload estimate and fair-share budget, predicted TTFT/P95/service time, source cache/prefix affinity;
- available model-weight resident bytes and per-session KV/state reservation, graph warm status;
- device temperature/power/safety/driver qualification and physical topology direct/relay links;
- tenant/data locality/allowed host, source-license rights;
- current session's state location, previous root/replica affinity and migration cost.

Prefer the replica holding the **compatible committed session StateRoot**. Stateless requests can route dynamically across replicas; stateful continuation is sticky unless a safe checkpoint/state migration is prepared and verified. Random or round-robin replica routing is a **baseline candidate**, not the right default for expensive stateful KV and long prompts. Hysteresis avoids ping-pong; no unbounded optimization cost per request.

Replicas may have different compiled execution tactics/accelerators, but only if both satisfy **the selected model/numerical/state compatibility policy**. A numerical-mode change that affects client-visible determinism must be opted into or pinned, not implicit because different node available. Strict deterministic requests can pin a specific qualified numerical plan.

### 4.3 Phase/task and multi-model pipeline routing

A single TaskProgram may explicitly bind several models: target plus trained MTP/draft, embedding model plus reranker, vision encoder and language model, speech encoder and decoder, or TTS conditioning and vocoder. These are **distinct named logical roles with separate artifacts and lifetimes**, all reserved under one TaskSession/application budget. The router schedules their stages/locations according to exact declared compatibility; it does not treat auxiliary model weights as an accidental replica of target.

Prefill/decode disaggregation or attention↔FFN placement is a **placement decision for one source model**, not request router silently changing target. The existing distributed PlacementPlan and ModelRouting manager must use the **same PhysicalResourceArbiter** to avoid capacity promises that conflict.

### 4.4 Optional intelligent target-model selector (explicitly configured)

A policy alias such as `auto` or `fast-private` may select among **different RegisteredModels**, but only if caller/admin config explicitly names allowed candidate pool and task/quality/privacy/latency conditions. Its result includes actual model selected and numeric/quality details in ModeReceipt; strict `exact` requests cannot fall back to another model or lower quant. This is an *application-level routing choice* and is **not** the same as load balancing equivalent replicas.

Policies cannot silently:
- choose a source model with different tokenizer/template/role semantics for an **ongoing session**;
- route sensitive input or weights to unauthorized external/cloud peers;
- decrease user-selected numeric/sampling quality or trained top-k;
- claim compatible token-level stream state across unrelated models;
- use remote proxy API as if it were a genuine local RegionExecutionProvider.

A failed exact model request gets typed insufficient-resources/no-compatible-replica error or an explicit user alternative suggestion, not wrong model output.

## 5. Model residency lifecycle, cold-start and scale-to-zero

```text
REGISTERED ───────► RESOLVING/VERIFYING ─► PREPARING/COMPILING
                     │                         |
                     ▼                         ▼
                   FAILED               ADMITTING MEMORY
                                                |
                                          LOADING WEIGHTS
                                                |
                                           WARMING PLAN
                                                |
                                              READY
                                             /   |   \
                                          BUSY IDLE DRAINING
                                             \   |   /
                                            COOLING
                                                |
                                            UNLOADING
                                                |
                                             UNLOADED
```

States are bounded, versioned and reported; optional side tasks `FETCHING`, `CONVERTING`, `UNSUPPORTED`, `BLOCKED_BY_POLICY` are **not** silently mapped to READY. Source downloads require explicit trusted source and user authorization (interactive consent or a user-requested pull). Loading means tensor/program verification and memory reservation before actual program execution. `READY` means enough validated devices/kernels/weights and model/task capabilities to accept the advertised request class; a cold compiled backend may be `PREPARING` while a portable variant is `READY`.

On-demand model access:
1. resolve alias/source and effective profile against hard security/numerical/device limits;
2. check catalog and residency of exact compatible artifact/adapter and TaskKind;
3. if not loaded, deduplicate concurrent load attempts by `ModelBinding`, reserve safe physical model+KV/graph+scratch resources, prepare ProgramLifecycle and load/warm once;
4. queue bounded waiting requests and let other resident models keep serving while load runs asynchronously; fail with useful retry/load state if deadline insufficient;
5. when ready, place requests on existing/new compatible replica and produce committed TaskResult;
6. when idle and no active pinned/snapshot-unreplayable session lease, eligible instance can cool/unload after configurable TTL or pressure; immutable catalog record remains.

**As many models as needed** means unbounded/large catalog and *as many concurrent resident instances as actual capacity policy supports*. ResourceArbiter checks **physical RAM, GTT versus true system memory, GPU HBM/VRAM, pinned buffers, per-context KV/SSM, graph/code cache, encoder/draft/adapter, distributed peers and OS reserve**. Do not promise infinite resident models. Saturation triggers safe queueing, idle replica unloading, allowed cache eviction, alternative qualified placement or explicit rejection—not host/system OOM.

### 5.1 Warm/keepalive policies

- `on_demand`: first admitted request loads; idle unload when beneficial.
- `keep_warm_for(duration)`: retain eligible weights/program after last request if resources permit; policy may be preempted by pressure unless explicitly pinned.
- `pin`: reservation for frequently used model/group, subject to owner permissions and hard headroom; cannot pin beyond safe physical budget.
- `preload/on_startup`: optional admin-selected only; don't automatically load every registered model.
- `scale_to_zero`: no resident weights after idle, with retained immutable source and any compatible **committed** state snapshots only under explicit session preservation/retention policy; otherwise warn that volatile sessions are lost.
- `max_resident_models`: optional administrative **safety/fairness bound**, not fixed universal 3/4/unlimited. Actual physical capacity is always the primary cap.

**Eviction must not corrupt active sessions or in-flight DMA**. The residency manager cannot unload an instance while TaskCoordinator has unresolved prepared state, unquiesced device graph or client-active unsnapshotable session. If a model is idle but its sessions have safely serialized/resumable state elsewhere, it may be evicted after owner policy and version/schema verification. On memory pressure, release optional cache/unused instances first; admit/reject safely rather than sneakily quantizing active weights. Avoid unload/reload thrashing for alternating large models via hysteresis, recent-load cost and fair-share.

### 5.2 Model replicas, sharding and placement

- **One deployment** may shard a model's tensor/layer/expert/state across several peers; router knows this is **one** replica's cohesive PlacementPlan.
- **Another replica** may independently load the same exact semantic checkpoint elsewhere with separate physical state roots; load balancer can choose it for new sessions.
- Source immutable tensors/adapters might be shared via content-addressed cache within a single host, but physical weight copies and driver allocations are budgeted separately.
- Local, trusted tunnel and rack nodes have independent physical reservation ledgers; router must not use a candidate placement that violates explicit model license/tenant/data locality.
- A new model can be loaded on a node already running another model if headroom, graph/kernel, memory and scheduling cost qualify. Scheduling data-parallel independent models vs sharding a single huge model are **different plan choices**.

## 6. One global physical budget, two scheduling timescales

**Model/replica level (coarse):** resource admission, warm/pin/unload, model placement, task/model pool quotas, prefetch/prefix/locality, idle budgets and power/temperature. The router must not hold a **global lock on every token** or perform per-token model alias resolution.

**Within an admitted model/replica:** existing TaskCoordinator and DeviceWorker schedule request rows with native batching, KV context and cross-session shared immutable weights, speculative verifier, async IO and fairness. Compatible request stage groups batch physically where exact source/task/adapters/numerics/shape permit; unrelated models may execute concurrently on separate streams/devices if resources allow. **Do not presume arbitrary same-GPU model kernels can batch as one mathematical model**, and don't serialize them through one global HTTP request mutex.

**Cross-model scheduler** may optionally amortize shared immutable base weights for compatible adapters and route compatible tasks to a warmed model. Tiny models with small contexts may coexist; giant models can use entire device/UMA/SSD+remote sharding. The `PhysicalResourceArbiter` remains source of truth for memory; a router-side “free VRAM estimate” is advisory, never authority to overcommit.

Queue fairness:
- per-model and global limits for pending bytes/tokens/tasks, separate model-load waiters vs active inference requests;
- per-tenant quotas and load-shedding policy, fairness/aging and deadline/SLO tags (interactive vs batch);
- **not** one huge FIFO that starves loaded interactive models while an unrelated giant model swaps in;
- bounded prefill/cold-load work and model-specific context/parallel reservation; avoid denial of service through model-switch thrashing;
- respond with known 429/503/429 retry metadata where appropriate, not hang forever.

## 7. Routing snapshot and request consistency

A canonical request is **resolved and pinned** once at admission:
```rust
struct ResolvedRoute {
    alias: AliasId,
    alias_rev: AliasRevision,
    binding: ModelBinding,
    task: TaskKind,
    generation_options: ValidatedRequestOptions,
    route_policy: RouterPolicyId,
    admitted_instance: Option<ModelInstanceId>,
    placement: Option<PlacementPlanId>,
    state_root: Option<StateRootId>,
    scope: TenantAuthorizationScope,
}
```

The `ResolvedRoute` is a **logical request control record**, not a per-token payload. Subsequent routing to another replica or backend during session continuation requires exact source/quant/adapter/state/schema compatibility and prepared state transfer under a new placement epoch at **committed TaskFrontier**, as specified in [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md). Streaming output model ID and usage correspond to the pinned resolved model, not an alias remapped halfway through generation.

Alias updates are transactional:
- create/edit new immutable revision, validate target source/template/TaskKind and profile config against policy;
- atomically publish alias map revision for *future* requests;
- existing streams/sessions continue under old binding or explicit safe migration;
- delete/unload waits on references/leases; alias removal doesn't destroy immutable model blobs that others reference;
- restart recovery reconstructs catalog from versioned manifest, not from transient warmed GPU pointers.

## 8. Per-model settings: explicit scope and safety

Every setting declares `Scope`, `Type`, `Validation`, `MutableWhen`, `SecurityFloor`, `CacheInvalidation` and `EffectiveSource`.

| Configuration class | Examples | When it can change |
|---|---|---|
| **Immutable source identity** | checkpoint SHA, tokenizer, tensor stored quant, model formula, fixed trained top-k | different RegisteredModel/alias revision; never silent runtime override |
| **Model-load plan** | devices, memory class, per-layer placement, quant pack, GPU fractions, offload, graph/kernel preferences, tensor/expert shards | before loading/warm; change requires new deployment or safe drain |
| **Resident instance** | min/target replicas, preload/pin/TTL/eviction, max concurrent models, device headroom, energy budget | admission policy at safe boundaries |
| **Per-session state** | context limit, KV pool, prefix cache, grammar/sampler source, LoRA adapter selection | at session start or only when compatible snapshot/replay proves legal |
| **Per-request generation** | temperature, top_p, top_k (where supported), max output, stop, seed, response_format, tool choice, reasoning visibility | validated per request; cannot break active session semantics or hard auth/physical limits |
| **Task pipeline** | encoder model, draft model, reranker, speech/vision sidecars, stage output codec | task admission or new TaskProgram revision |
| **Router policy** | exact model aliases, replica pool, sticky session, cache affinity, SLO, model selector opt-in, queue limits | policy revision; active routes remain bound |
| **Topology/network** | allowed peers, strict locality, preferred devices, transport, state/weight/cache placements | safe new PlacementEpoch only, explicit remote authorization |
| **Service/security** | listen bind, TLS/mTLS, keys, allowed registry sources, auth scopes, log privacy | admin config/restart or safe hot reload; **never downgraded by request** |

**No implicit source quant downgrade or cross-model fallback.** A profile can intentionally point to an explicit quantized derived artifact; router cannot create/use it without source/quality identity and user authorization. Existing task/memory hard limits cannot be defeated by per-request settings.

## 9. Multi-model coexistence and cache compatibility

- **Shared immutable weights:** a model plus compatible adapter generation may share base source/device weight buffers; different adapters can share base if exact math/activation implementation supports request-specific low-rank contribution and tenant isolation.
- **KV/recurrent/session state:** never shared across unrelated model artifacts/adapter/tokenizer/position/grammar/tenant. Prefix cache hit must pass existing full StateCompatKey and all relevant state-domain frontier—not just same human model alias or text.
- **Compiled bundles:** may reuse code for different model instances if SemanticKey, physical shape/quant, driver/compiler/capture and state layout match, and parameter slots correctly bind weight addresses; never persist raw pointers or leave stale captured graphs after unload.
- **Cross-model routing:** one request's target verifier can't use a draft trained for incompatible model without explicit verifiable compatibility. Encoder–reranker or target–draft pipelines require separate admitted model resources and cost budget, not silent placement.
- **Replicas in remote cluster:** shared prefix/snapshot migration only among authorized peers with exact model binding and a valid state bridge; if no safe transfer, stick to original instance or begin a new session.

## 10. Failure model and human-facing behavior

| Condition | Correct runtime behavior |
|---|---|
| Requested model registered but not loaded | automatic **local** authorized load once, bounded queue/progress and safe memory reservation |
| Requested name unknown | clear `ModelNotFound`, suggestions/list; **no arbitrary remote download** |
| Model available in registry but not downloaded | explain size/license/source; explicit `pull` or interactive consent; offline mode refuses gracefully |
| Artifact recognized but semantic op unsupported | explicit `UnsupportedModelOp`; not a random alternative model |
| Model fits only via optional peer | ask/require configured explicit remote policy; otherwise report placement options, no private data exfiltration |
| Multiple models exceed RAM | unload safely idle/unpinned, evict optional cache, queue or reject; no OS-wide OOM, no unapproved quant change |
| Model loads but compiling preferred fast graph fails | select known valid portable/generic candidate if semantics supported, explain fallback |
| Model already loaded under same source profile | share resident weights/compiled plan; create fresh TaskSession state |
| Alias edited while request streaming | pinned old model/profile revision until end; no mixed tokenizer/KV |
| Model stops/unloads during active request | graceful drain by default; forced cancel only explicit admin and correct last committed frontier |
| Replica/peer dies | failover only with proven compatible StateRoot snapshot/epoch fencing; otherwise typed interruption, never silent new model |
| Cache affinity route worse than fresh load | measured cost selector chooses qualified alternative replica; no network state transfer per token by default |
| User requests unavailable API option | typed `UnsupportedFeature/UnsupportedParameter`; not silently ignored or model-substituted |
| Client disconnect or slow reader | bounded backpressure, cancellation policy and preserved committed output sequence; no leaked provisional tokens |

A `ModeReceipt` reports resolved model source/quant/adapter, effective profile revision, actual replica/deployment, numerical/TaskDriver mode, load/warm/queue time, cache and fallback reasons, physical resources, and optional privacy-safe placement. Default CLI/API output remains concise, with diagnostics only on explicit inspect/explain or authorized headers.

## 11. Security, remote scope and API boundaries

**Router metadata is sensitive:** available models may reveal private research, custom weights/tenants, device inventory or internal peer topology. Native model admin/control API is **not** the same as public inference API, and exposed model IDs are filtered by tenant/role. Localhost default avoids accidental WAN binding; external bind requires auth/TLS strategy and explicit consent. Registry pull and model artifact shipment must respect license, private credentials and dataset/data policy. No model-supplied remote code or tool execution in model router.

Router is not a mandatory **arbitrary upstream cloud provider proxy**. Optional TaskServiceAdapters for approved external model APIs may exist at the application level, with explicit trust/credentials/quality/data egress, but cannot pretend their text-only response is a native zerogravity model `RegionExecutionProvider`.

Avoid an API control-plane call in the hot inference loop. User request body may contain arbitrary tool schemas/JSON/media URLs: tokenize/validate with limits and safe allowlisted fetch policies; executing tools is an independent permissioned workflow above TaskRequest.

## 12. Observability and perf admission proofs

Track at distinct levels:
- **Catalog:** registered aliases, pinned artifact and profile revisions, known tasks/qualifications; no private IDs exposed to unauthorized clients.
- **Residency:** per model replica warm/loaded/idled/loading/evicting, actual weight/program bytes by device, load/init graph time, adapter/drafter memory, pins and eviction attempts.
- **Router:** resolved alias→artifact and replica, queue delay / load wait, cold start/p95, load thrash rate, cache affinity hit/miss, model switch and per-tenant fairness.
- **Execution:** RequestPlacement/StepPlan, batching/graph mode, KV prefix state, per-phase TTFT/decode, real energy/device/IO, task output units, fallback reason.
- **Network:** peer direct/relay, resource reservations, state hydration and transfer bytes/time, retry/epoch failure, forbidden remote policy.
- **Quality:** exact source/quant/LoRA, sampler/grammar, task output/reference; no silent model substitution, tier downgrade or hidden tool invocations.

**Benchmarks:**
1. one-model single process `run` versus `serve` with no extra proxy/per-token dispatch overhead;
2. 1, 2, 4, 8 independent small models resident, with realistic context and physical memory and concurrent requests, C1/C8 fairness/P95;
3. giant model + two tiny models under memory pressure, idle eviction/thrash guard without corrupting active KV;
4. replica routing same exact model with prefix locality and sticky sessions; compare naive round robin, per-replica queues and measured critical path;
5. 100 concurrent requests for cold model deduplicated to one load and bounded waiting; no accidental extra copies;
6. hot alias/profile update mid-stream and concurrent streaming under old revision; no wrong model ID/state;
7. distributed one model sharded across two peers versus two replicas and two source models on same peer; no mistaken replica/cluster accounting;
8. error handling on untrusted or unknown model/quant, no silent fetch or quant downgrade; lost peer, cancellation and worker DMA fences;
9. OpenAI SDK chat/responses/embeddings smoke tests and exact streaming conformance, including UnsupportedFeature/unknown params;
10. model/task-specific parameters allowed/denied by profile policy and resolved provenance `config explain`.

An optimization can earn a default only after real model/quant/hardware tests; first proof is functional accuracy and safe resident scheduling with **no separate inference engine**.

## 13. Minimal implementation staging

**The first model vertical slice stays mandatory.** After a usable CPU/GPU reference:
- add `ModelCatalog` of content-addressed manifests and immutable alias bindings;
- add **one** `ResidentModel` lifecycle + safe on-demand load under existing ResourceArbiter;
- add bounded `ModelRouter` that chooses exact source binding and eligible resident replica/placement, reusing existing TaskCoordinator;
- expose `run` and `serve` with same internal engine, plus `ls/ps/inspect/pull` progressively;
- add equivalent replica routing, hot/idle eviction and per-model settings, then multi-model batch fairness/cold-start policies;
- only then model-choice policy aliases, distributed placement across peers and non-text TaskDriver pools if physical/model tests justify complexity.

**Do not** add a generic service mesh, separate gateway daemon, distributed registry DB, model-specific worker subprocess per model or hardcoded maximum loaded model count as an architectural requirement. A model process isolation mode may be a later security/fault tradeoff, still speaking the same engine worker protocol.

**Closure requirement:** a user can launch one supported model with no config and get a correct response, or expose several models and manage their residency through one API, without ever needing to understand CUDA tiles, RIR/KIR/CIR, sharding epochs or routing internals. Every hidden choice can be examined and overridden safely when explicitly requested.

````

## Source: .agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md

Original UTF-8 bytes: 19641; SHA256 `ffabf256a01a3c29b455593ba5e0162e1cb6a417bedce7fb7b13e00b15594341`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Model semantics, model adapters and compatibility qualification

**15 Oct model-family precision addendum:** [Qwen3.8/GLM5.3/MiMo2.6 recipes and drafting](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) makes **MiMo-V2.6-Flash a separate registered architecture to implement**, not another Qwen model alias. MiMo must identify source E8M0 scales, source-specific SWA/global layers and asymmetric value-head width, its independent DFlash partial RoPE/tap/mask/attention, and exact derived-quant paired layer overflow correction. Qwen's GDN/QSA/PLE/MTP and GLM's KDA/MLA/DSA/mHC/MTP remain distinct. Model recognition, host parser support and a single GPU matvec never imply runnable whole model. HYB-01..18 are required source-real acceptance fixtures before any capability promotion.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


## 1. Model support contract

A model descriptor is validated source metadata **plus** explicit semantic graph recipe and tokenizer/rendering contract. Config parsing alone is insufficient. Every new architecture must state:
- model/checkpoint revision, official config and tensor manifest;
- token ID/BOS/EOS/tool/reasoning special token representation, text and multimodal template;
- layer topology, stateful vs pure ops, special attention and MoE formulas;
- quantized or native precision per tensor family, numerical accumulation and storage;
- supported context ranges, positional schemes, masking and cache layouts;
- trained draft/MTP compatibility; snapshot/restore semantics;
- supported input modalities, encoder projection and image/audio position calculations;
- output head/logit and sampler numerical contract;
- CPU reference, backend capability requirements and quality vectors.

A loader must distinguish `UnknownModel`, `KnownModelUnsupportedVariant`, `UnsupportedOp`, `UnsupportedQuant`, `InsufficientResources` and `MalformedArtifact`. No silent generic transformer guess for a novel model.

## 2. Canonical operator registry

**Foundational:** embedding, norms (RMS/Layer/RMS variants), RoPE/GQA/MQA/local position, projections, gated activations, FFN, matmul, quantize/dequantize, softmax, route/topk/scatter, logits/sampling.

**Full attention:** causal MHA/GQA/MQA with paged KV, sliding/local and mixed global, model-specific RoPE scale and sections, quant KV, FlashAttention, grouped multi-query.

**Hybrid recurrence:** Gated DeltaNet with learned gates, state matrices, short convolution, chunk prefill/step decode; Kimi Delta Attention with fine-grained gates/decay and independent recurrence; keep gated recurrence variants separately versioned (not a generic SSM approximation). Both require exact state snapshot + rollback.

**Sparse compressed attention:** Qwen QSA with indexer-selected blocks; GLM DeepSeek Sparse Attention over MLA latent KV (NoPE where defined); DeepSeek V4 CSA/HCA time-axis compressed keys and special indexers; independent positional/causal, softmax and quantized attention semantics. Sparse indexer top-k changes model computation and must not be hidden as a performance optimization.

**MoE/residual:** explicit router logits/scoring (`softmax`/`sigmoid`, bias, scaling, noaux etc.), top-k or grouped selection, shared expert, gate/up/down weights and sum ordering; simple gated residual vs manifold-constrained hyperconnections (mHC) not interchangeable. Effect order must preserve causal layer frontier.

**Lookup/draft:** token n-gram/Engram lookup tables with historical token state and hashed-row read; separate trained MTP/DSpark/EAGLE models; target verifier maps speculative draft to authoritative state.

**Multimodal:** vision/video encoder, image patch/resize/merge, projector, modality token spans, 3D/MRoPE positions, audio encoder and special prompt template. No model is considered multimodal functional without full encoder/position/continuation tests.

## 3. Candidate first model families

### Llama and dense baseline
Common decoder-only dense causal/GQA, RoPE, RMSNorm, SwiGLU variants; baseline for portable graph, KV, tokenization and GPU GEMM performance. Specific new Llama variants need exact config; not all use identical RoPE/attention heads. Build complete CPU & GPU before deploying advanced IR.

### Gemma 4
Includes sliding-window vs global attention, proportional/other model-specific position rules, output transformations and, for 26B-A4B, routed MoE with 128 experts and top-8 + shared expert according to inspected model docs. Correct Gemma tokenizer/multimodal pathways as applicable. Read released config and official implementation for every exact variant (E2B/E4B/26B-A4B etc.), do not infer identical semantics from name.

### Qwen3.5+
A broad hybrid family with GDN and periodic full attention; both dense and MoE/vision variants exist. Treat Qwen3.5 and Qwen3.8-Flash-Next as different architecture IDs, even when sharing GDN primitives. Native model config must determine gate/state dimensions, number of attention layers, residual and positional scheme.

### Qwen3.8-Flash-Next
Research snapshot: language backbone ~125B, ~6B active/token (count excludes large auxiliary 51B n-gram parameter table + ~4B MTP); 48 layers, approx 36 Gated DeltaNet and 12 QSA, 512 routed experts with top 10, gated residual/hyperconnections, n-gram lookups and one MTP predictor. Native context ~262144 tokens in DwarfStar's documented graph; extensions such as YaRN are policy with possible quality changes. Different quant artifacts: DwarfStar Q2 main/MTP ~41.73 GiB resident and ~95.37 GiB BF16 n-gram disk-only; a closed reference native checkpoint ~62.1 GiB plus ~47.7 GiB paged n-gram; Gufo bench UD-Q4_K_XL target and Q8 draft sidecar. **Not equal precision or a controlled speed comparison**. DwarfStar Qwen supports Metal/CUDA, not ROCm at inspected snapshot; the closed reference and Gufo are Strix `gfx1151` references.

### GLM-5.3-Flash
From Maya's config study: ~321,323,031,390 total, 18B activated/token; `glm5-next`; 45 layers, hidden 4096; layers 0–2 dense, layers 3–44 routed MoE => 42 x 288 = 12,096 expert blobs; top 8, shared expert; 34 KDA and 11 sparse MLA+DSA (layers 3,7,...,43); NoPE latent MLA with 512-wide KV latent, DSA top-k 2048, 4-way mHC/Sinkhorn, 1 trained MTP. Quant Q2 Maya-S ~96.5 GiB and related options, DwarfStar GLM Q2 ~90 GiB, different quant recipes/quality. Attention and recurrent state cannot be lowered as ordinary GQA. Context up to 1M advertised does not imply cache or accuracy parity at 1M.

### DeepSeek V4 Flash / V4.1 Flash
DwarfStar's model research: V4 Flash ~284B total/~13B active, ~43 layers, 128-token raw sliding window and older history compressed by layer-specific 4:1 indexed (top-k up to 512) or 128:1 unindexed regimes; mHC, shared/routed MoE and trained DSpark support. V4.1 Flash has a large ~189 GiB Engram table in DwarfStar's Q2 file kept disk-only; different checkpoint/graph and tokenizer. Keep variants separate and checksum draft support files. Do not confuse DwarfStar quantized Q2 (81 GiB V4 Flash main) with official mixed FP4/FP8 model.

## 4. Initial capability matrix (PLANNED — not supported yet)

| Family | CPU semantic | AMD gfx1151 | Discrete AMD | NVIDIA CUDA | Apple Metal | Vulkan |
|---|---|---|---|---|---|---|
| Dense Llama | PLAN | PLAN | PLAN | PLAN | PLAN | PLAN |
| Gemma 4 dense/MoE | PLAN | PLAN | PLAN | PLAN | PLAN | PLAN |
| Qwen3.5 hybrid | PLAN | PLAN | PLAN | PLAN | PLAN | PLAN |
| Qwen3.8-Flash-Next | PLAN | extreme research target | PLAN | PLAN | PLAN | optional |
| GLM-5.3-Flash | PLAN | extreme research target | PLAN | PLAN | PLAN | optional |
| DeepSeek V4 family | PLAN | PLAN | PLAN | PLAN | PLAN | optional |
| Vision/audio | PLAN | separately qualified | separately qualified | separately qualified | separately qualified | separately qualified |

`PLAN` is not feature support. Exact model/quant/version/backend cells must migrate to `FUNCTIONAL`/`VERIFIED`/ `PERFORMANCE_QUALIFIED` only after tests. A generic GPU backend may be incomplete; CPU reference should be first for each semantic operator.

## 5. Model adapter shape (conceptual Rust)

```rust
struct ModelIdentity {
    family: ArchitectureId,
    checkpoint_sha256: [u8; 32],
    tokenizer_sha256: [u8; 32],
    quant_manifest_sha256: [u8; 32],
    semantic_version: u32,
}
struct ModelRecipe {
    identity: ModelIdentity,
    layers: Vec<LayerRecipe>,
    tensors: Vec<TensorSpec>,
    states: Vec<StateSpec>,
    positions: PositionRules,
    prompt_contract: PromptContract,
    modalities: ModalitySupport,
    draft: Option<DraftRecipe>,
}
trait ModelAdapter {
    fn recognize(&self, verified_metadata: &ModelMetadata) -> Result<ModelRecipe>;
    fn build_semantic_graph(&self, recipe: &ModelRecipe) -> Result<ModelGraph>;
}
```

Implementation should avoid storing model weight buffers inside `ModelRecipe`; it describes topology and stable TensorIds, while the ModelArtifact store owns backing bytes, and the runtime manager owns resident allocations. A `Session` owns only mutable state, not a copy of all weights.

## 6. Integration steps for an unfamiliar model

1. Pin official source model card/config and tokenizer code; hash the intended checkpoint and its legal distribution.
2. Produce a tensor manifest (names, shapes, stored quant, offsets, dimensions), reject mismatches.
3. Decompose layers into existing exact semantic ops; list missing ops explicitly.
4. Implement missing reference ops including state and multimodal positions; add deterministic small fixture.
5. Build Semantic IR recipe and validate shapes, masks and state effects.
6. Compare teacher-forced per-layer activations and full logits against independent reference in FP32/BF16/FP8 depending on artifact, then quantized format.
7. Add general backend lowerer and correct session prefill/decode, chunks, cache, sampling and cancellation.
8. Add optimized candidate kernels *after* measuring bottlenecks, validate output and quality.
9. Re-evaluate support across backend x context x quant x modality cells.
10. Publish known limitations; never claim blanket model family compatibility based on one small variant.

## 7. Numerical contracts to encode

LayerNorm/RMS epsilon and accumulator dtype; RoPE theta/sections/scaling; attention masks/top-k/ties/softmax reduction precision; router sigmoid/softmax/scaling/shared expert order; quant block types/scales/outlier layouts; hyperconnection and state update order; model's sample filtering and position advancement. A stored model's precision and a compute repack are separate. Fallback cannot arbitrarily reinterpret latent KV or recurrent matrices.

## 8. Future compatibility principle

A new model can run as soon as its novel math/state semantics have a verified reference implementation plus any required backend lowering. It need not wait for extreme kernels. But if no backend supports an operation, it must fail clearly rather than produce plausible nonsense.

## 2026-10-10 integrated v2 addendum — model recipe/compiler decoupling

The research original `ArchDef + ModelSpec` has been adopted as an excellent **model registration and validation** mechanism, but `ModelSpec::kernel_plan(hw)` is removed from authoritative ownership. A recipe declares the operator/state graph and exact tensor roles; the compiler/kernel provider selects implementation. Registry status distinguishes **recognized artifact**, **semantic model supported**, **runnable on selected backend**, **numerically validated**, **performance-qualified**.

**Modern explicit families remain required goals:** Llama, Gemma 4, Qwen3.5+, Qwen3.8-Flash-Next, GLM-5.3-Flash and DeepSeek V4 variants. The older research's baseline Llama/Qwen2/Qwen3/Gemma3/Mistral/Phi is a pragmatic early CPU/portability test set, **not** a replacement for user-required newer families. New model geometry comes from version-pinned official source/config; do not assume 48/45-layer examples apply to adjacent versions. Gufo is an additional `gfx1151` Qwen source absent in older research corpus.

**Model capability must include runner type:** decoder-only text generation, encoder-only embedding, cross-encoder rerank, multimodal processor/encoder/projector, and adapters (LoRA) with distinct per-session cache identities and possibly draft compatibility constraints. `PoolingSpec` requires no generation/KV loop. Vision/audio spans have explicit causal/bidirectional attention behavior and modality fingerprint; a new video/image model cannot just inject fake text token IDs.

**Loading invariant:** reading a valid GGUF/safetensors with 43+ known type IDs is *not* evidence any arbitrary trained op is implemented. For a truly unknown semantic op, report `UnsupportedOperator` and require a new reference op/state contract. No implicit HF remote-code execution. GGUF dialect, tokenizer pre-type, quant layout, template and source checkpoint must match the model recipe.

**Model admission proof:** per checkpoint+quant+backend produce golden layer activations/logits, tokenizer exactness, native positions and state-after-every-chunk, tool/grammar and (where applicable) image encoder/rerank quality. Include an explicit forced kernel-missing fallback test for each new semantic family. See `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`.


## 2026-10-12 integration — Gufo-family semantic reuse and task control graphs

**Upstream Gufo is source evidence, not a new mandatory runtime.** The original `gufo-org/gufo` model implementations include Qwen3.8 Flash-Next hybrid/GDN/QSA, Qwen3.8-27B and DFlash2, DeepSeek V4/DSpark, ASR, TTS and emerging diffusion/video workloads. A `ModelRecipe` still declares exact semantic operators, tensor shapes and mutable state independent of GPU/NPU/compiler. The specified `mikealanni/gufo` fork's XDNA2 offload is a **kernel placement choice for pure projection subsets** within one Qwen recipe, not a new `NpuQwen` mathematical architecture or different MoE top-k.

Add model-aware **`TaskProgram`** definition distinct from core mathematical SIR: `TaskKind::{TextGenerate,AudioAsr,AudioTts,ImageGenerate,ImageEdit,VideoGenerate,Embedding,Rerank}`, stage DAG and stage-specific control loop (autoregressive tokens, encoder segments, vocoder frames, flow/denoise timesteps), mutable states, output stream and quality contract. TaskKind is extensible and a status/capability, not evidence any one model works. `TaskProgram` shares SIR/RIR/KIR/CIR but does not force different models into causal text KV or default sampler semantics.

For multi-device NPU acceleration, a compatible `Projection` shape may be split by **output row spans** under RIR legality and a partial-result layout join; exact model config, quant scales and all downstream GDN/indexer state operations remain invariant. An unknown AIE kernel or unsupported numerics must fall back GPU/CPU, not change model semantics. Reference tests should include source Gufo Qwen hybrid projection/ASR/TTS model family vectors where redistributable, and full GP U-only parity before any NPU candidate.

## v4 refinement — registered exact mathematical models and future parameter versions

`ModelDefinition` builds one versioned `SemanticProgram` from verified checkpoint config, tensors, tokenizer/positions and registered exact operators, independent of target kernel/device/task. Two model variants with different attention gate, trained router, compressed indexer or vocabulary semantics require distinct exact op schemas and CPU reference; a shared marketing family is not enough. A novel model built from existing semantics needs only manifest/recipe and fixtures, not a new server or scheduler.

Model **compatibility matrix** now separates semantic source, TaskDriver stage/control, numeric precision, shape modes and kernel qualification; one `FUNCTIONAL` model/device pair may not be `PERFORMANCE_QUALIFIED`. An unsupported dynamic shape may lower to generic reference if correct operator implementation exists, not trigger uncontrolled JIT. Future optional ParameterVersion/TrainStep cannot mutate immutable weights of active inference sessions; new adapter/model generation creates a separate cache/compiled bundle compatibility key.

## Current metadata-only source profiles

These profiles bind configuration/layout, not executable support. Current
implementation and required trained tests live in [model-execution/TODO.md](../plans/model-execution/TODO.md).

**Qwen3.8 27B:** official `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, declared qwen3_5_text,64 layers
(48 GDN/16 full), hidden5120, FFN17408,24Q/4KV width256,16 GDN key/48 value
heads width128, vocabulary248320 and untied output head. The metadata index has
1199 tensors/18 shards, with851 ordinary text roles. This does not verify actual
shard headers/bytes; vision and MTP are separately admitted components.

Original HF grouped V axes differ from converted tiled axes. The converted
profile permutes V-related projections/constants/convolution/output, applies
-exp(A_log) and adds1 to non-GDN norm weights. Bind the selected profile
explicitly. `output_gate_type=swish` denotes GDN SiLU; full attention still uses
sigmoid. Exact shard-metadata integer parsing accepts `9007199254740993.0`
without precision loss. It rejects all exponent notation (including `1e3`),
nonzero fractional parts and u64 overflow.

**Gemma4:** preserve official source revisions in the
[family technical contract](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md).
E2B has35 layers,15 KV owners and20 consumers reusing compatible producers13/14,
plus packed per-layer embeddings and doubled later FFN width. A missing optional
weight index is a single-file observation, not complete header/name validation.

The26B variant has hidden2816,30 layers,128 experts/top8, expert width704 plus
dense2112 in every layer. Five sliding layers then one full: sliding16Q/8KV
width256/window1024; full16Q/2KV width512 with shared raw K/V and separate norms.
Full RoPE rotates128/512;26B has neither PLE nor shared-KV-layer ownership. Expert
gate_up `[128,1408,2816]`, down `[128,2816,704]` and router scale roles stay exact.

Routing dataflow is residual, unscaled RMSNorm, learned router scale,
inverse-sqrt-width, projection, FP32 full-row softmax, top8, selected
normalization and expert scale. Experts have separate pre-FFN norm and
GELU-tanh gate×up; dense/expert postnorms precede common postnorm, residual and
layer scalar. Framework-specific top-k/reduction/scale placement differ: choose and
independently qualify the original arithmetic before implementing routing.
Analytic test-local cases do not establish trained numerical parity.

These contracts retain the executable-source4GiB limit. Large sources need
admitted immutable owned shards with actual header/dtype/name/offset/content
binding, not a mutable mmap/private-COW shortcut or another source issuer.

````

## Source: .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md

Original UTF-8 bytes: 24661; SHA256 `52e5f1d1b113d1ff359a3eda0d533dc5fe17e55d69d90cbe85f475e35603c2cd`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — modular AI execution reference architecture v4.1

**2026-10-12 · authoritative v4.1, design only.** This is the **smallest architectural contract** for a high-performance, composable Rust AI runtime, including native kernels and compiler-centric/device-graph/dataflow execution families. Exact formal execution/state details are in `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`; model/task/compiler details in `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`; critical review and decisions in `ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md`. Rejected v3 design choices are summarized in the [critical review](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md). No code or benchmark has been implemented.

## 1. Prime rule — compose execution capabilities, never new engines

**A model describes mathematics; a task describes control; a provider supplies an implementation; a plan schedules computation and physical resources; a coordinator owns visible state.** No new model, task, quantization, accelerator or specialization is allowed to require a second independent model server, allocator, output service or task-session authority.

We target a practical **Pareto frontier** of mathematical/task quality, latency/P95, throughput, energy, memory capacity, ease of deployment and cost of extension. A general design does not automatically beat a specialized engine on one model/hardware workload; the ability to **substitute a complete hand-tuned region** while reusing the same runtime gives us a path to compete.

The runtime is Rust-owned. Vendor drivers/toolchains are optional devices with explicit language/FFI/license policy. It must always be possible to build a useful CPU-only reference path without any GPU/NPU compiler or external model server. Unknown mathematical ops are not magically runnable from a recognized checkpoint.

## 2. The minimal system vocabulary

| Canonical concept | Authority | Not responsible for |
|---|---|---|
| **Artifact/ModelDefinition** | immutable model weights, config, tokenizer/quant, exact semantic program/registered state effects | selecting a vendor kernel or serving requests |
| **SemanticProgram (SIR)** | mathematical operations + shapes/precision/position + typed mutable effect dependencies | physical device buffers |
| **TaskDriver** | task-specific stage/iteration control, stop/frontier/output type; one driver per task session | hardware kernel codegen and external privileged agent workflows |
| **RegionExecutionProvider** | one **closed semantic region** lowered to a native kernel, captured graph, ahead-of-time device graph or static dataflow program; native KernelProvider is one option | mutating session roots, duplicating a model runtime, using unknown opaque inference APIs as verified execution |
| **ExecutableBundle** | immutable prepared kernel handles + **CIR** command/resource DAG + guarded variants/fallback | live request queue or plan search on every output |
| **DeviceBackend/IOBackend** | physical execution, memory subpool, async stream/events/transfers, runtime capability | model semantics or user-visible results |
| **PhysicalResourceArbiter** | coarse global **admission/reservation/headroom** and quotas across physical tiers | per-kernel allocation lock or setting user's firmware |
| **TaskCoordinator** | one logical mutable writer **per TaskSession/shard**; commit/stream ordering, fair scheduling | direct unsynchronized device memory mutation |
| **StateRoot/StateTransaction** | immutable version tuple across **all** state domains; prepare/validate → one atomic logical root publication | magically atomically writing all GPU device memory |
| **ModeReceipt/Evidence** | exact supported backend/shape/precision/path, performance/quality and fallback provenance | claiming unmeasured performance |

These concepts can start as **modules within one Rust library plus binary**, not nine crates and a plugin ecosystem. The exact accelerator micro-schedule can be owned by its device/compiler; Rust retains model/task/state/reservation authority and a single program submit/ready boundary. Optional device providers can later be separated into a few crates when target/compiler build constraints justify it.

## 3. Two enduring representations; compiler-internal optimization views

```text
Artifacts + ModelDefinition            Trusted task request / TaskDriver
             |                                        |
             +--------------> SemanticProgram (SIR) <+
                           tensor math, typed effects
                                     |
                   transient RIR legal-region transformations
                     + provider candidates: tile, whole graph, static dataflow
                                     |
               physical CIR command/resource/timepoint lowering
                  (kernel, graph, fabric program or supported remote program)
                                     |
                           ExecutableBundle
               guarded shape bands + code handles + memory DAG
                                     |
                PhysicalResourceArbiter (coarse reserves)
                                     |
                TaskCoordinator -> StepPlan -> DeviceWorkers
                                     |
                   Prepared StateRoot + EventCompletion
                                     |
                  PublishRoot by single session owner
                                     |
                 Ordered committed task outputs/ModeReceipt
```

**Only the semantic program and physical executable plan have durable/architecturally stable identity.** SIR is the canonical executable mathematical reference; RIR and KIR are bounded compiler-local *views*, and CIR lives within the physical plan. These stages are helpful engineering vocabulary, **not a mandate for four independently serialized dialects or runtimes**.

At steady operation, the device worker submits **precompiled regions/kernels/graphs/static dataflow programs directly**, allowing a vendor accelerator compiler to own its micro-schedule without a host-side loop per tensor operation. It does not interpret the SIR or run optimization passes per language token/audio frame/denoise timestep. Dynamic router IDs and sampling can remain in device buffers where the target mathematical/grammar contract allows, avoiding mandatory host roundtrips.

A new specialized whole-layer or whole-model code path is simply a `RegionCandidate` satisfying a closed semantic region contract (same inputs, outputs, mutable state effects, numerical/correctness mode and bounded resources). The portable decomposition remains valid and can be forced on in tests. Fallback between incompatible physical state layouts occurs at a canonical committed checkpoint with conversion/replay, not by pointer reinterpretation.

## 4. Task control without an internal workflow platform

Use a **small trusted Rust `TaskDriver`** to issue stages and transitions for a task kind, not a general user-supplied DSL for arbitrary agent steps. Stage-level control can express a bounded branch/window/loop in the TaskDriver's implementation, but device-native `StepProgram` executes repeated mathematical steps where possible without frequent host wakeups.

Primary task types:
- autoregressive text with exact sampling/optional trained speculation and token frontier;
- embedding/encoder and rerank scoring without decoder-specific KV;
- ASR and streaming speech recognition with audio-window/sample frontier;
- TTS/voice acoustic generation + vocoder with complete PCM/audio chunks;
- image generation/edit with validated denoiser/flow/timestep loop and latent state;
- video/audio with frame/window/temporal state and synchronized outputs;
- compound **AI computation** with trusted stage composition, not arbitrary tool execution by default;
- **future optional** gradient/parameter-update training task, whose mutable model version/optimizer state are isolated from running inference sessions.

External RAG, tool use, agent/cross-product workflows and privileged actions belong in an **application/orchestrator above** the inference TaskRequest boundary, with authorization/sandbox. The inference runtime is a compute engine, not an unrestricted automation system.

Every task declares its own `WorkUnit`, `CommittedFrontier`, `StreamContract`, `CancellationPolicy`, resources and numerical/quality checks. A video frame isn't an LLM token. Model/task support remains status-gated by actual checkpoint and code implementation; listing a kind is not claiming support.

## 5. State and output transaction, exactly one logical publish

The earlier `StateDomain::commit()` was insufficient. v4 requires:
1. **Begin** from pinned immutable `StateRoot{session_epoch,frontier,domains}`.
2. **Execute** into private/COW provisional state; device/IO workers return events and candidate state.
3. **Prepare** each domain under one target frontier, verify shape/schema/quant, sampler/RNG/grammar/adapter and actual device event readiness; failure → abort and quarantine buffers where DMA still live.
4. **PublishRoot** exactly once by sole logical session writer when all required domains ready. No domain independently publishes committed state and no physical CIR command has that authority.
5. **Emit** ordered verified results identified by `(session,epoch,output_seq)`; live transport promises ordering/at-most-once *attempt*, not impossible exactly-once receipt across lost network ACK. Optional authenticated cursor replay can support resumable delivery.

This is atomic **logical root visibility**; physical GPU state updates use immutable/COW/private tail or safe undo/recompute. Distributed workers require versioned epochs/fences/readiness agreement; never claim instant distributed memory atomicity or durable exactly-once semantics without a real distributed log protocol.

Full-state prefix cache (KV, recurrence, sparse indexer, n-gram, sampler/grammar, adapters, modality) reuses the last **common compatible** frontier across all domains, not simply token text; immutable shared snapshot identity prevents cross-tenant/model/quant/position leakage.

## 6. Fast scheduling and resource ownership

One session/shard writer owns current TaskCursor and committed state, but not every hardware allocation in a global mutex. **Global admission is coarse; device-local subpools allocate fast under reservations** with generation-tagged BufferLeases and async retire; all permanent/temporary model, KV, recurrent, experts, encoder/media scratch, NPU staging and remote IO count against real physical RAM/VRAM/OS headroom.

A `StepPlan` binds compiled `ExecutableBundleId` with actual request/stage rows, device slot generations, runtime `ShapeGuard`, reserved leases and target state epoch. Persistent worker batches apply diffs; precompiled device-native step islands can route/specialize/advance dependent operations without host sync as allowed. Prefill, decode1, verify, encoder, denoise and vocoder are separate shape/cost regimes. A safe host-loop plan remains for features whose grammar/recurrent effects prohibit device-only progression.

Scheduler policy uses **typed stage cost/resource budgets** and fairness deadlines, not one universal token budget. Coarse stage coordination can overlap GPU and CPU work, route through sparse expert/SSD tiers and use NPU or remote accelerators after actual availability/numeric/fabric safety qualification. A slow client cannot accumulate infinite uncommitted GPU/PCM/frame output buffers; backpressure and cancellation are required.

**Physical safety is conservative, not omniscient:** OS memory changes independently of the engine. Probe and reserve headroom, monitor pressure and fail/evict/replan safely; don't promise zero OOM on an unrestricted multiprocess OS. GPU-visible GTT/virtual addressing isn't free physical memory.

## 7. Shape, numeric, cost and specialization guards

Every KernelCandidate/ExecutableBundle variant is bound to:
- semantic/StateEffect identity, numeric policy and source artifact/quant;
- symbolic **fixed rank + bounded dynamic dimension constraints**, alignment, ragged rows, supported batch/context/step band and output layout;
- device ISA/driver/compiler/program and optional graph/eager/safe-power profile;
- scratch/physical memory leases, availability/first-use fences and state-layout compatibility;
- numerical/quality evidence and measured stage/full-task performance.

Dynamic router IDs are **runtime data**, not a per-token compiled kernel variant; unsupported shapes always have a generic runtime guard/fallback if that operator is semantically supported. Compile/tune a small set of hot shape bands, not a codegen artifact for every context length. Offline bounded search; online only select approved variants at safe epochs with hysteresis. Compiled GPU plans are expendable caches invalidated on toolchain/driver/model changes; **never persist raw driver handles** inside state snapshots.

Qualify performance under identical model/quant/context/task/power/driver and exact numeric policy, with TTFT/P95, committed output units, quality, physical memory, energy and source-proven baseline. A mathematically model-changing approximation (reduced expert top-k/mask/altered diffusion schedule) requires explicit user opt-in, never automatic "optimization."

## 8. Minimal extension seams and evolution

| Adding | Extend | Must remain unchanged |
|---|---|---|
| New model with known ops | ModelDefinition/config/weights and independent reference fixtures | TaskCoordinator, DeviceBackend, ResourceArbiter |
| Novel recurrent or sparse op | semantic op + exact StateDomain + portable reference | unrelated models, CLI/server |
| New kernel/device or graph/dataflow accelerator | RegionExecutionProvider and compatible native kernel/graph compiler/runtime adapter, state bridge, physical/topology/shape and safety evidence | model mathematics, task frontiers and user-visible session authority |
| New quant or layout | checked codec/pack and numerical contract | generic session service |
| New AI task | TaskDriver stages/output/stream frontier and needed op semantics | device/resource/other TaskDrivers |
| New storage/peer link | CIR IO/transport and ResourceTier with lease/event semantics | tensor math and TaskDriver |
| New speculative algorithm | DraftSource/target verifier + full state transaction | ordinary model operations |
| New compiler optimizer | verified legal region transformation and bounded search | model source, user tools |
| New service adapter | canonical TaskRequest/Result mapper, auth and privacy | kernel providers, state effects |
| Future training | optional GradientProgram and ParameterVersion publication separate from inference | existing immutable inference model/session semantics |

The **modularity test** is practical: implement one new model operator, one accelerator provider and one non-token TaskDriver without editing unrelated API, memory, GPU driver or other task code; measure code churn/compile/load overhead and quality.

## 9. Deferred complexity and engineering discipline

**Initial implementation:** one library and one binary (or 2–4 cohesive crates if GPU toolchain demands it); static built-in providers, one model+TaskDriver vertical slice and device event/lease checker. Small versioned types, direct prepared execution, independent reference. Avoid bloated trait hierarchies and duplicated cache ownership.

**Defer:** generic plugin ABI, unrestricted workflow DSL, online distributed training optimizer, automatic dynamic model JIT every shape, equality-saturation in steady serving, all hardware types, media endpoints without reference checkpoints, distributed durable transaction WAL, third-party executable source code loading.

**Security:** no untrusted model remote-code execution or implicit tool privilege; no automatic `sudo`, BIOS/IOMMU/fabric-clock/suspend modification to speed an experimental GPU/NPU path; no cross-session private prefix/cache leak; executable device binaries/trusted code require independently reviewed source and precise driver capability. Rust-owned kernel source requirement persists, vendor driver FFI policy open.

## 10. Evidence, scope and authority

**Primary current docs:** `COMPOSITION_AND_EXTENSION_CONTRACTS.md` (corrected illustrative interfaces), `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` (state and async executor), `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md` (IR and TaskDriver), `ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md` (review and prior-art research), v4 master and decision register. All earlier subgroup documents are supporting details, subordinated where in tension with this v4 design.

**Release proof:** G0 real model and fake async event/shape/state faults, G1 CPU/GPU usable path, G2 multi-session and hybrid-state service, G3 extreme model/device specialization, G4 non-token model/task breadth and distributed, G5 reproducible release, G6 extension drill and sustainable evolution. All remain **NOT_STARTED** until code/tests/hardware evidence exists. `research/inference` is read-only evidence.

**Long-term win condition:** new semantic models, kernel hardware, state machines, iterated tasks and storage tiers attach without rewriting the runtime. When a fully specialized closed region is fastest, use it there; when it loses/unavailable/unsafe, preserve the portable model and a truthful receipt.

## 11. v4.1 execution granularity — native GPU, graph-only and dataflow accelerators

The single `RegionExecutionProvider` seam supports **multiple physical execution kinds** without adding another model or TaskDriver runtime:
- **NativeKernel** and **CapturedCommandGraph**: prepared direct CPU/GPU/edge accelerator computation;
- **AheadOfTimeDeviceGraph**: an entire legally closed model region/task stage compiled by a graph-centric accelerator, with shape, state, runtime, firmware and quant qualification;
- **StaticallyScheduledDataflow**: device compiler places entire operation pipelines/collectives and manages SRAM/on-chip program/weight streaming, preserving a large closed region boundary;
- **ManagedDeviceSession**: only when driver/SDK exposes enough program-state/snapshot/rollback/epoch and cost semantics for our StateRoot contract; opaque model-serving APIs are *application TaskService adapters*, not lowerable backend implementations.

The provider determines minimum efficient **region granularity** (tensor op, fused region, stage, whole model, multi-device partition) and micro-schedule owner. **Rust TaskCoordinator still owns model/task admission, one StateRoot PublishRoot, output sequence and global physical reservations**; a vendor graph compiler may own only internal device scheduling and memory layout. TPU-like graph programs need bounded dynamic shape groups and nontrivial compilation/warmup budget; wafer-scale/static pipelines need programmed fabric, streamed weight sources and stable on-chip residency; GPU rack domains need actual peer links/collective topology and fault fencing; coherent local SoCs need real physical UMA not assumed HBM.

A source compiler/runtime inaccessible or forbidden under Rust-source/FFI policy produces `CAN_MODEL_BUT_NOT_COMPILE` rather than a false `SUPPORTED` status. A device graph with opaque **in-place state** but no safe snapshot/recompute cannot legally replace a general stateful speculative region. Tiny host kernel launches should not be used to emulate the static pipeline; match the *target's efficient native granularity*. **All of this remains within one SemanticProgram→ExecutableBundle execution spine.**

**Detailed contract:** [accelerator execution families](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md). **External prior art:** [accelerator platform research](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md) (research only).

## 12. v4.2 — distributed machine and state placement as a core execution capability

**The universal runtime is local-first but network-composable.** Independent maps for compute, immutable model parameters, mutable per-layer/per-session state, and reusable immutable caches combine into **one PlacementPlan** within ExecutableBundle/CIR, with a costed TransferPlan and TopologyEpoch. This does not add a third enduring IR or an independent cluster inference engine. The same RegionExecutionProvider, DeviceWorker/RemoteRegionWorker, TaskDriver, ResourceArbiter, StateRoot Prepare/PublishRoot and output sequence apply on every machine.

Capabilities span single-host colocated, per-layer compute+KV colocated on remote peers, compute separated from attention+KV, dense/FFN/experts on other workers, prefill/decode state handoff, remote prefix hydrate, attention token-range shards, tensor/head/expert collectives, and different TaskDriver encoder/denoiser/vocoder stages. **Remote storage, remote stateful compute, and full remote state reads are different physical candidates.** Dynamic movement only at a safe committed frontier, and SIR operator-specific PartitionAlgebra is necessary before token-range attention splitting. Caches can be stale hints; authoritative mutable shard references require versioned leases and epoch fencing.

**Peer connectivity:** explicit authorized peer enrollment; control over authenticated direct/tunneled links; measured separate bulk data transport over physically eligible RDMA/peer or bounded encrypted streams. Tunnels are connectivity, not RDMA or low P95 latency guarantees; relayed WAN often favors whole layer+cache placement over chatty per-layer splits. Remote failure returns Prepared/Failed, never another session writer; a partition stops or replays safely, without claiming transparent leader failover/exactly-once output.

**Full specs:** [placement and physical cost](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [peer/tunnel and state protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md), and [subsystem entry](DISTRIBUTED_PARALLELISM.md). Source names/benchmark caveats remain [research-only](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md).

## 13. v4.3 — built-in multi-model request routing without another runtime

The project now has **one small ModelCatalog/ModelRouter/ResidencyManager control plane** integrated with the same TaskCoordinator, ResourceArbiter, RegionExecutionProvider/ExecutableBundle and DeviceWorkers used by single-model direct `run`. **Router request admission is not a hot-per-token execution path**. Standalone local model inference can run in-process without an HTTP server or daemon, and a long-lived `serve` instance exposes the same library engine across multiple registered models.

Separate `RegisteredModel` immutable checkpoint/quant/ModelDefinition, versioned `ModelAlias/ModelProfile`, actual resource-admitted `ResidentModel/Replica` (potentially one sharded DeploymentPlan across nodes), and `TaskSession` with its own full StateRoot. Registered model count isn't resident model count; replicas of same checkpoint aren't multiple distinct model sources; target/draft/encoder different models in one TaskProgram aren't replicas. One alias revision resolves to an exact immutable source/adapter/template/numeric identity for the session duration. **No automatic different-model or lower-quant substitute** under memory/queue pressure.

The router chooses compatible warm replicas using measured queue/cold load/physical capacity/prefix affinity and authorized device/peer placement; stateful sessions remain sticky unless full compatible StateRoot handoff at a committed frontier. Model residency (verify→admit→load→warm→ready→idle/drain→unload) is safe under physical reservations, per-device local pools and lease/drain gates. Pin/keepalive/scale-to-zero are optional. A configured `auto` alias may deliberately choose a different target model under an explicit user/model-quality policy and reports the actual source.

**Product surface:** proposed simple `zerogravity run <model>` and `zerogravity serve`, native `models list/ps/inspect/pull`, versioned model profiles and one loopback OpenAI-compatible `/v1` API subset with honest SDK/streaming/usage/tool-output conformance. Native model-management/admin rights are separate from public generation. Full reference: [model router and residency](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/config/API design](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). Source project comparisons in [non-normative research](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md).

````

## Source: .agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md

Original UTF-8 bytes: 25956; SHA256 `da65ffcafe3769127592dceda531685e3b7af416708f43fbfb628cb0e0b0c3b8`.
Resolve embedded relative paths against the original path, not this volume.

````text
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

````

## Source: .agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md

Original UTF-8 bytes: 30697; SHA256 `49e3a8885580ee665729bab9fd8d54015589cd849e42cb3b8751992d162db3ee`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Numerical correctness, model-quality evaluation and benchmark methodology

**15 Oct 2026 — mandatory vLLM parity-and-beyond qualification:** [36 concurrency/source tests and fair benchmark methodology](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) require pinned vLLM V1/vLLM-Omni version/config and identical original or derived quant artifact, tokenizer/template, task/numeric policy and comparable device/OS/cache state. Report cold/warm, admission and rejection, per-tenant fairness, p50/p95/p99 TTFT/ITL, accepted **committed** SLO goodput, actual bytes/driver and power, vLLM current continuous/hybrid/batch/disagg capability. Quant-model fidelity and target speculation exactness are distinct. “Outclasses vLLM” cannot be claimed on a different quant, cherry-picked prompt or from the existence of planned design.


**15 Oct specialization evidence clarification:** [derived quantization source qualification](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) and [original extreme-specialization evidence/method](../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md) require **three independent comparisons**: original FP8/BF16 model vs lossy derived quant; exact kernel/source math against the *same derived pack*; and target-only vs MTP/DFlash speculative acceptance/output distribution. Source reports Qwen derived quant top1 ~93.81% vs official FP8 yet lossless speculation relative to quantized target—no contradiction. PPL/argmax-only tests can miss invalid scratch/FP16 overflow, and parallel batch reduction may change near ties. Benchmark matched checkpoint/source/quant/prompt/acceptance/context/driver/power, repeated interleaved runs and negative candidates; never promote single prompt speed or confuse pack size with speed.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


This is the authority for deciding whether a kernel or execution optimization is valid and whether a performance claim is supported. The user requests peak performance *without losing general compatibility and fidelity*. No quality promise may be inferred from model loading or plausible answers.

## 1. Four independent notions of correctness

**Semantic formula correctness:** operation implements exact released architecture (e.g. KDA vs GDN, MLA NoPE, sparse top-k, mHC, time compression).  
**Numerical operator agreement:** tensor outputs match an independent reference to documented tolerance under chosen precision/reduction policy.  
**Model quality:** aggregate behavior stays acceptable versus the chosen source checkpoint/quantization and official or independent task/reference corpus.  
**Reproducible state/sampling:** same execution plan, seed, tokens and scheduler produce compatible states and outputs; exact target sampling distribution preserved where advertised.

One may pass while another fails. A 90/100 greedy first-token match against FP8 doesn't prove 90% overall reasoning equivalence; exact logits against the *previous optimized version* do not guarantee parity to original BF16 source.

## 2. Reference layers

- **A: scalar/FP64 mathematical oracle** for attention score/softmax/selection, routing and recurrence where calculable; use sufficiently stable range/reference for quantized integer dot exactness.
- **B: native reference model** (pinned HF Transformers or official source, with exact model and template) for teacher-forced layer activations/logits, sampling and image embeddings. Official reference may use different numerical reductions on GPU; document this.
- **C: zerogravity portable Rust CPU implementation** which implements exact chosen model formulas and quant layout; must agree with A and B by declared tolerance.
- **D: optimized device kernels/graph** compared with C and B under matching precision and chunk mode.
- **E: full serving lifecycle** state, prefix, tools, serialization and sampled/greedy outputs, concurrency and failure.

Never use our own optimized GPU kernel as the **sole** reference for itself. Reference kernels and corpus fixtures must be versioned independently.

## 3. Numeric error metrics

Given reference `r` and candidate `c`, record:
- max absolute `max |c_i-r_i|`;
- max relative with epsilon for near-zero denominators;
- RMS error and relative L2;
- cosine similarity for embedding vectors;
- KL divergence between logits-derived distributions and top-k agreement;
- max/mean error per tensor/layer/context/frontier;
- FP64 sparse attention error, selected-key IDs and score ties;
- first-token greedy agreement and longest matching continuation prefix;
- negative log likelihood / perplexity on checkpoint-matched corpus;
- model-specific official task quality and tool/structured output success;
- sampled distribution statistical tests where exact speculation is promised.

**Numerical tolerance values are not decided globally.** They must depend on op, dtype, quant, downstream sensitivity and reference. Keep stringent bounds for masking, index IDs, integer quant decode, positional state and sampler; allow bounded floating reduction differences only with high-level model quality proof.

## 4. Critical operator regression cases

| Operator | Required edge cases |
|---|---|
| Norm + RMS/Layer | tiny variance, zeros, large vector, BF16/F16 rounding, odd widths |
| RoPE/MRoPE | position 0, long context, extrapolation, multi-axis images, model scale |
| Attention softmax | all masked/empty, extreme logits, causal edges, head tails, ragged chunks |
| Sliding/KV | ring wrap, overwrite after branch, context eviction, page COW |
| Sparse QSA/DSA | selected width, top-k ties, negative scores, index masks, compression boundaries, 32K/128K |
| MLA | absorbed vs expanded compressed latent, NoPE, FP16 cache and head scaling |
| GDN | scalar decay variants, conv kernel 4, chunk vs step, state matrix snapshots |
| KDA | per-channel/fine-grained gate, lower bound, recurrent state, 34-layer geometry |
| mHC | Sinkhorn iterations, 4-stream mixing, normalization and rounding, phase offsets |
| MoE | router bias/scoring/top-k, duplicate experts, shared + routed order, 288/512 experts, quant tails |
| Quant matrix | all sign bits, block padding, scales, exact dequant, strange strides, type safety |
| Engram/n-gram | hashed token history, repeated IDs, invalid table row, truncated file, state restore |
| MTP/DSpark | verifier row grouping, 0..N accepted prefix, RNG rollback, tool boundary, EOS |
| Vision/audio | preprocessing, image resize, position maps, encoder cosine and ordering |

Tests must cover one-token decode, repeated 2/4/8 verification, 256/1024/2048+ prefill and continuation after deep prefixes. The best path for one shape may be slower/wrong for another.

## 5. Quantization and quality comparison

Keep three artifacts conceptually separate:
- original BF16/FP8 model reference and official model family;
- a low-bit checkpoint produced by particular imatrix/calibration recipe, with quality/precision fingerprint;
- a lossless device repack of the second checkpoint that should preserve semantic dequant output.

Comparing BF16 model vs Q2 changes quality even if source code same. Evaluate per-tensor sensitivity and per-family quality on a fixed corpus before accepting new quant. Mixed quant may retain routers, attention and dense always-active projections at higher precision while aggressively compressing routed expert matrices; document weights/bytes per category.

An optimization that changes sparse top-k, cache KV quant or sampler algorithm is not merely layout/performance; it requires its own numerical/quality contract.

## 6. Speculative decoding sampling rigor

For greedy target decoding, proposals are accepted only when matching target decisions under same reference/plan arithmetic; near-tied logits can legitimately diverge across FP precision or batch reduction. Record first divergence and logit delta, not only final answer.

For stochastic sampling, correct target-preserving speculation requires mathematically correct acceptance probabilities and residual redistribution. Verify proposed source `q`, target `p`, RNG consumption, filtering/top-p/top-k/min-p order, penalties and corrections using exact or high-precision CPU probability reference. Property tests over tiny categorical vocab exhaust probabilities and distributions; Monte Carlo validates empirical distribution (with confidence limits). Tests for zero-probability proposals and small `q(x)`, EOS/stop and batch index boundaries.

If an "opportunistic" mode accepts matching greedy proposals at nonzero temperature, label it **not distribution-preserving**, user opt-in only (proposed default policy). Record result differences, don't market as exact target.

## 7. Hybrid state replay equivalence

For a fixed model and reference plan:
- one full prefill vs chunks 1,31,256,1024,2048,non-even;
- decode after initial/continued prefill at 4K/8K/32K/128K and sparse thresholds;
- fork same immutable prefix into N branches, mutate one and ensure others identical;
- serialize/restore snapshot and compare logits/history position;
- run speculative proposals and commit every prefix length 0..d, compare to fresh token-by-token control;
- cancel between GPU events and re-run from committed prefix;
- tool-mode boundary and image/text history.

If output differs only because the execution plan uses different floating reduction order, classify separate "same-plan exact" and "cross-plan tolerance" test. Numerical drift should be bounded and model quality verified; no hidden byte-identity promise across Metal/CUDA/HIP.

## 8. Benchmark taxonomy

**Isolated kernel:** per op/shape latency, occupancy, memory throughput, numerical diff. Useful for diagnosis, not serving ranking.

**Model inference:** prefill tokens per second, target autoregressive decode tokens/sec, speculative **committed** tokens/sec, context depth and batch. Distinguish new prompt chunk length from existing cached prefix depth.

**Service benchmark:** client-observed TTFT P50/P95, inter-token latency P50/P95, total request latency and aggregate tokens/sec under concurrency and mixed arrivals, token correctness and cancellation.

**System benchmark:** cold/warm load, physical RAM peak and GTT vs VRAM, SSD read rate/hit rate, contention, network bytes and tokens per joule; no OOM/swap storms.

**Quality benchmark:** official/checkpoint-matched completions and tasks, numerical metrics and input modality. Never report speed improvement when outputs are silently lower quality.

## 9. Benchmark comparability matrix

Always match:
- **exact model family & checkpoint SHA**, quant bytes/recipe, tokenizer/template, enabled MTP support and auxiliary tables;
- GPU/CPU device model, DRAM/VRAM, PCIe/NUMA/GTT, kernel/runtime/compiler driver versions, memory clocks and power;
- prefill new-token count, existing depth, output tokens, stop policy, cancellation, beginning vs continued prefix, cached state;
- concurrency, warmup, explicit use of graph capture, batch/gpu prefill time and cache condition;
- sampling temperature/top-p/min-p, speculative exact/opportunistic mode, acceptance distribution and draft depth;
- timing method; host vs device timer, throughput vs end-to-end latency, repeat count and quality checks.

When matched comparison isn't feasible, report two separate numbers with visible caveat; do not compute misleading "engine X 10x better than engine Y" from different models.

## 10. Recorded external author baselines (research only)

- Gufo `gfx1151` Qwen3.8 Flash-Next UD-Q4_K_XL: reported pp2048 1700.52 and tg128 25.95 AR at d0; at 131K ~1478.66 pp / 24.92 tg; one-user repetitive MTP up to ~60.39 vs mixed ~31.32; C8 AR aggregate ~108.68. Project's source documents describe pinned llama.cpp comparator and one warmed sample.
- A specialized `gfx1151` Qwen3.8 native-quant engine: ~1584 pp 8K, ~1517 pp 131K, ~37.6 short-context serial, ~55–56 coding-agent speculation; different checkpoint/quant and benchmark methodology from Gufo.
- Maya GLM5.3 Flash Q2 on 128GB Strix: ~210–217 pp and ~17.8 tg when experts resident; dual-gpu AMD MTP ~33 tg with different machine.
- DwarfStar Strix DeepSeek V4 Flash Q2: ~187.26 to 292.86 pp after kernel tuning, context around 4K; model differs from Qwen/GLM.
- Prior-art evidence shows model- and operator-dependent performance gaps even when the host language is Rust; source-specific numbers belong in the research atlas, not as a universal design target.

Numbers show **mechanisms** worth reproducing, not targets achieved or ranked. Reinspect source bench documents and attach per-commit evidence before using them as acceptance thresholds.

## 11. Statistical method

Prefer at least several stable independent repeats with alternated A/B order (ABBA or randomized block), power/thermal stabilization and significance/uncertainty. Use bootstrap interval or robust median CI, show raw result distribution and context-specific outliers. For p95 need enough requests for meaningful estimates; 10 token samples do not give reliable tail percentiles. Time-of-day/OS load/SSD page cache should be controlled, and explicit test classes distinguish cold from warm and cache-hit from miss.

A win on individual kernel might be rejected if whole-model throughput worsens or P95 rises. Small single-run percentage differences should be reported as inconclusive.

## 12. Recording results

Record the command, relevant model/reference/device/toolchain profile, measured
quality/performance, actual result and remaining failures in the plan TODO.
Model content identities and independent numerical oracles remain required where
byte/source equivalence matters. Keep a bounded diagnostic capture only when it
helps investigate or assess that result; no blanket log inventory or receipt
schema is required. See [the TODO workflow](../../AGENTS.md),
[experiments](../plans/EXPERIMENT_PROGRAM.md) and
[release gates](../plans/RELEASE_ACCEPTANCE_GATES.md). Never commit model weights,
private prompts or credentials.

## 13. Quality gates and acceptance policy

A new specialized kernel only becomes default if:
1. required shape and semantic tests pass;
2. state rollback/snapshot/continuation tests pass;
3. no unauthorized change of quantization, sparse selected set or sampling distribution;
4. model quality remains within documented tolerance for the exact checkpoint;
5. matched end-to-end benchmark demonstrates gain beyond noise or a justified throughput/latency tradeoff;
6. memory and device failure/stress tests pass;
7. fallback tested when kernel disabled and compiler/driver cache invalidated.

If data incomplete, keep current simplest correct kernel; optional experimental flag permitted only with clear label. The team's optimization lab must preserve negative experimental findings to avoid rediscovering them.

## 14. Long context and quality

A model's declared max context is not proof of useful retrieval or stable latency at that depth. Exercise indexed sparse-attention threshold, latent compression and recurrent state, 8K/32K/64K/128K/256K/1M where hardware allows. Use needle/structured retrieval, code references and official continuation logprobs. Attention selector quality should be measured separately from overall model quant quality.

## 15. Acceptance report example (pseudocode)

```text
Experiment: gfx1151-quantized-moe-prefill
Model: exact GGUF SHA / official reference / quant
Hardware: AMD gfx1151, 128 GB, ROCm build, fixed power
A: kernel version 1, B: specialized version 2
Shapes: expert count, active top-k, 256/1024/2048 new rows
Correctness: maxabs/rms, selected ids, state parity, full logits
Quality: official corpus reference, greedy divergence and NLL
Performance: ABBA repeats, medians/CI, TTFT/P95, cache/SSD
Result: pass/neutral/fail/inconclusive
Next action: keep/revert/iterate, commit and source links
```

All fields filled from actual evidence, not from qualitative architectural reasoning.

## 2026-10-10 integrated v2 addendum — defensible Pareto performance and correction ledger

The long-term objective is to approach or improve the **measured Pareto frontier**, not claim faster-than-everything by architecture. Report per (model checkpoint, quant recipe, device, driver, context, batch/concurrency, acceptance, sampler, power) cells: throughput/TTFT/P95, RAM/cost/power, independent numerical fidelity/quality and availability. A win for C1 decode cannot imply a C32 serving win; repetitive MTP speed is not unconstrained reasoning t/s. External prior-art engine numbers (Gufo/Maya/DwarfStar and similar) are author reports and quant/workload-specific.

**Serving-mode-aware tactic selection:** tune candidate kernels against **actual eager vs captured graph execution**, not isolated GPU kernel timer alone. Offline schedule search demonstrates possible speedups but also a compilation-time explosion risk; use bounded experiments. Bound compile time/memory and warmup; ship measured plan, with reproducible default and no unchecked runtime JIT inside token loop.

Preserve research's detailed `S1–S12` GPU/compiler and CPU-oracle ABBA acceptance, but split tests:
- strict same-plan exact/bitwise on controlled exactly-representable cases;
- FP32/BF16/fused GPU numerical agreement at declared tolerance;
- exact target **sampling distribution** with speculative rejection/correction;
- checkpoint-level quality and long context/task tests;
- cross-plan/backends independent quality contract rather than universal bit identity.

**Source corrections:** the research's 8×L40S ~126 tok/s DwarfStar benchmark is a different model/workload and cannot stand as matched GLM-5.3 evidence. Historical 75–125 engineer-month and ~93 engineer-day Phase0/4–6wk estimates are staffing *hypotheses*, not guarantees. Research `D-14` MIT/Apache and old CUDA kernel versions are council recommendations, not user approvals.

**New test classes:** CIR lease/event failure, compiler effect rewrite legality, 3+ hybrid state group prefix share, model-version/LoRA tenant cache isolation, GPU cold/host memory/GTT and NVMe critical path, new backend codegen generated ISA. Publish and retain negative experiments; any unproven gain stays opt-in.


## 2026-10-12 integration — NPU BFP compute and performance comparability

The Gufo XDNA2 source converts Q8_0 model weights and GPU F16 activations into v8bfp16ebs8 block-float (8 signed mantissas and 1 shared exponent per nine bytes). This is a **compute-arithmetic change** rather than a new source GGUF checkpoint. Its 30 matching greedy prompts are a preliminary smoke test, not proof of equal logits, stochastic distribution, robust coding quality or GDN/QSA state. Require full tensor/logit/quality and model-continuation parity, attention selected-key identity and all-state/speculative rollback.

The claimed 16K prefill GPU-only 1698 → GPU+NPU 1821 tokens/sec was measured with held fabric clock and performance-mode NPU, so benchmark comparisons must hold clock/power/temperature constant in both arms, and measure GPU-only default vs held-clock as a separate experiment. Include pipeline first-consumer stalls, NPU BO sync, pinned RAM, energy/idle power, C1/C8 concurrency and long context. On hardware at risk of hard hang, do not conduct uncontrolled GPU-NPU load tests. Source numbers are author reports, **not** zerogravity achievements.

## v4 refinement — four distinct truth claims and task-relative units

Clearly distinguish: (1) **semantic model correctness** including trained routing/positions and state effects; (2) **numeric compute agreement** at a declared rounding/reassociation/temporary quant tolerance; (3) **checkpoint/task quality** under model+input+context/phase; (4) **distribution/state exactness** of sampled speculation and target verifier under supported modes. A faster stored quant or altered expert top-k/attention indexer is a **changed model/approximation**, not an automatic optimization. Strict bitwise replay may be a fixed-plan test; no blanket cross-driver GPU bit identity without proof.

For non-token tasks compare appropriately: ASR transcript/WER/timecodes, TTS audio output and latency, image/video sampled quality/completion, embedding/rerank accuracy, and optional future training loss/gradient. Use `WorkUnit` and `OutputUnit`, not fake t/s across all AI execution. Benchmark actual whole-task P95, time-to-first-valid-output, peak physical memory and energy in addition to operator kernels, under matched source checkpoint/quant/driver/power. Source-author claims and hypothetical architecture wins remain unverified.

## v4.1 — cross-accelerator precision and program validity

Qualify compute arithmetic and task quality per `RegionExecutionProvider`, not one generic “GPU supports FP4” flag. At minimum distinguish **native FP4 E2M1 with FP8 scale per 16 values and per-tensor scale**, power-of-two scaled microscale FP4 across 32 values, FP8/BF16 formats, integer/GGUF block encodings and device-specific compute accumulation. A temporary device compute format differs from source checkpoint quantization and a deliberately requantized checkpoint; **do not change model routing, trained attention or final sampling silently to fit a TPU/wafer/Blackwell backend**.

For graph-program backends, correctness must include **whole closed subgraph values, dynamic shape guards, device-managed recurrent/KV/indexer state, provisional snapshot/rollback or explicit stateless restriction**, and exact temporal task frontiers. A program that compiles but cannot reconstruct correct state after cancellation/spec rejection is not functionally qualified for general decoding. Published vendor tokens/s or TPU training peak does not establish same-model inference performance; compare exact checkpoint/quant/graph compiler/driver and cold compile amortization, full TaskDriver P95/quality/energy, topology/link and per-chip useful outputs. Temporary/simulated graph-only backend passes architecture proof **only**, not physical performance.

Read [accelerator backend contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) and research [platform evidence](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md).

## v4.2 — distributed mathematical parity and transport-aware benchmarks

Three correctness classes require **different** references: (a) contiguous layer parallel with per-layer KV/SSM co-location must match unpartitioned full model state and logits; (b) attention/FFN compute/state separation must preserve the exact source attention/FFN and numeric mode with explicit cross-network activations; (c) sequence-range KV sharding may merge per-shard attention summaries only if **the operator's PartitionAlgebra** reproduces mathematically valid causal/masked softmax and data-dependent selected-indexer semantics. Empty all-masked shard and floating overflow/underflow/tie cases need numerical reference tests. **No generic GDN/KDA recurrence sharding assumption.**

Metric matrix: C1 and C8/32 latency, TTFT/p95 ITL and committed t/s, prefill/verify and long context 4K/32K/128K; same source model/quant and device capacity, network direct LAN vs encrypted overlay direct vs peer-relayed vs high-RTT WAN, tunnel encryption CPU/MTU/loss and idle power. Measure **capacity enabling** separately from raw speed. KV state transferred must include hybrid recurrent/indexer/ngram/modal/grammar when required; a text-prefix hit without that state is invalid.

Fault/quality tests: wrong remote model hash/quant/adapter, corrupted/partial transfer, stale cache hint, remote replica serving wrong checkpoint, coordinator epoch loss/split brain, remote old-epoch StateShard generation and event/DMA quiescence, old root recover/replay. A physical network test needs actual machines/links; simulated cost doesn't prove performance. [Network research](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md).

## v4.3 exact-model routing and multi-model performance truthfulness

Routing a request to the wrong checkpoint, different stored quant, LoRA adapter generation, model-native tokenizer/template or sampler is **a semantic correctness regression**, even if output is plausible and the destination replica produces more tokens/sec. Default exact alias routes only among model/TaskDriver/numerical-compatible replicas. Explicitly configured `auto` source selection is a **changed target model policy** and reports actual resolved source, not an equivalent kernel optimization. Model profile revision/hot reload can't change live source StateRoot.

Report three distinct measurements: (1) one model `run` direct/in-process vs `serve` through ModelRouter under same artifact/task/sampler with negligible/per-request route work and **no per-token proxy tax**; (2) several resident source models/replicas with physical RAM/session KV and native batching, C1/C8 TTFT/P95, fair queue; (3) model-loading/unloading/cold start/anti-thrash and source quality/availability. “Many registered models” isn't a claim they all run simultaneously. A multi-model total output tokens/sec can be misleading if compared across different checkpoints/context/numeric or tasks; report per-source and per-tenant workload weights.

API conformance is a separate quality gate: tested SDK and endpoint versions, chat vs Responses SSE schemas, usage, model ID, stop/JSON/tool data, Unicode and cancellation, unsupported parameter errors, no assumed stateful Response/tool execution; no silent downgrade to different model or quant when overloaded. [Router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

## U-048 — separate NPU-twin parity, source quality and route performance

[heterogeneous pipeline qualification](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) distinguishes exact W8/grid-scale/BF16
CPU↔GPU-twin↔NPU execution from quality versus the selected immutable source
quantization and from target-lossless speculation. Greedy text equality alone
proves neither logits nor sampled law. Paired/tail arithmetic and selected family
cuts are bound to existing numeric/state provenance or a proved compatibility
class. H5/H6 source-bit gates remain intact; incompatible exact-mode candidates
are refused. A qualified tolerant/derived encoding remains explicit policy, never
a silent tolerance increase. Timing needs matched artifacts, quality, routes,
state/KV modes, prompt/sampling/context, power and repeated stationary comparisons;
all 40 new HP gates are planned, with no new performance result here.

## Original Qwen3.5 source arithmetic profile

This technical summary retains implementation constraints formerly described in
retired phase reports. The detailed executable contract is the existing CPU/GPU
source helpers, independent reference clients and regression tests. Accepted
results and open work are in [the numerics TODO](../plans/numerics-and-context/TODO.md).

- Pin native reference revision, CPU/F32 KV/nonflash mode and physical
  microbatch/chunk history. Raw teacher token IDs avoid incomplete UTF-8 HTTP
  fragments. Functional8K thresholds do not permit source-bit operator changes.
- Quantized projections preserve their source block and reduction schedules:
  Q4_0 uses Q8_0 activation blocks and the optimized eight-row32-integer-block
  FP32 FMA order; Q4_1 adds its separate zero-point path; Q8_0 retains eight
  accumulators; Q6_K uses signed-six/four-product/eight-lane order and contiguous16
  correction. Original F32 uses four FMA accumulators and its reduction tree.
  A standalone one-row dot is not automatically the same source schedule.
- FP32 products, recurrence iterations and reduction intermediates must round
  where the source rounds. Explicit stored buffers provide required device
  boundaries; a local cast need not prevent contraction or reassociation.
  GDN order is decay, prediction, beta, state update, readout and scaling.
  RMS uses rounded squares, ordered F64 sum, then the specified FP32 stages;
  Q/K L2 retains its RMS(epsilon/width) and scale schedule.
- WideF64 GEMV uses bounded F64 multiply/add then F32 conversion, with no
  residual buffers; its measured source qualification is not general IEEE FMA
  emulation. Separately, RoPE/QK/value and other qualified FMA consumers retain
  stored FastTwoSum residual correction and midpoint regressions because a
  generic F64 multiply/add then F32 conversion can double-round.
- Preserve softmax's two-step F64 reciprocal then F32 conversion. For example,
  denominator `0x3ff0000018000024` gives F64 `0x3fefffffd0000000`, then F32
  `0x3f7ffffe`; direct exact-real-to-F32 rounding is a different operation.
- Attention value accumulation includes exactly ceil(valid/256)*256 source
  FMAs with canonical future operands +0. Skipping padding or adding capacity
  blocks changes zero signs. Rewind/import must restore canonical future KV;
  capacity changes stride, not the arithmetic prefix.
- Partial IMRoPE pairs half-head axes with iterative FP32 theta/source FMA.
  Do not replace it with a closed-form FP64 phase or unqualified device trig.
  GGML polynomial SiLU differs from scalar-libm attention sigmoid/GDN gates;
  keep each original exp/trig, contraction, underflow and signed-zero policy.
- Activation quantization preserves Q8_K first signed maximum, independent
  rounding and Q8_1 sum from unrounded d before half conversion. Source-defined
  tiny blocks can contain permitted Inf/NaN scratch and saturating NaN-to-zero
  code conversion. Validate consumed scales/status/final values according to
  the operator; this is no exception for observable nonfinite model outputs.
- Greedy selection rejects nonfinite logits and preserves finite f32 total_cmp:
  +0 outranks -0; equal-bit maxima select the lowest token ID. Preserve source
  bits and subnormals. Never silently widen zero-sign or numerical policies.

````

## Source: .agents/docs/OPERATOR_AND_FUSION_CATALOG.md

Original UTF-8 bytes: 19814; SHA256 `87d160c1e67cd2cf47290a52ae5ee07f06c6ad6a0b4347e8d5e3816f9c60146d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Operator/region contracts, fusion patterns and model execution catalog

**15 Oct 2026 — batching specialization boundaries:** [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) adds the physical shape dimension to each existing source-qualified closed region candidate: compatible active sessions and rows, padded expert partitions, optional target verification depth, routed expert histograms, layout/precision, device event dependencies and scratch. Merge matrix rows only when all source model/codec/operator effects and numeric mode match; other models may share scheduling and immutable encoder but not an invalid fused matmul. Quant variants, cache warmth and draft acceptance inform the cost model, not semantic expert choice. Test SERVOPT-21..26 against reference state and true device submissions.


**15 Oct candidate-registry refinement:** [extreme hybrid semantic and draft rules](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) and [quant source codec](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) add **E8M0 source scale and paired convertible FP16 intermediate** as explicit tensor/semantic variants, plus derived-trellis/decode regions and separate MiMo attention. Norm→activation packing→quant GEMV, GDN/KDA recurrence and residual mixer, MLA indexer/tile, MoE dispatch→grouped quant expert→reduce can fuse only with original source effects, full state and legal reduction order. No whole-layer optimization can lose source mHC/Sinkhorn, QSA/DSA index or draft hidden-tap dependencies. Physical row tile size is KIR, not a model operator.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


This is the implementer's companion to `SEMANTIC_IR_AND_COMPILER.md` and `MODEL_SEMANTICS_AND_COMPATIBILITY.md`. It lists the semantic and execution choices a model adapter/kernel author must make, not fabricated finalized formulas. Exact mathematical equations must be derived from the pinned official model source and verified CPU oracles before implementation.

## 1. Canonical operator record

Every operator registry entry must include:
```text
operator_id, schema_version, aliases/model-family bindings
inputs and outputs: logical dims, dtype, quant stored vs compute layout
semantic parameters: eps, scale, mask, RoPE, gating, routing, compression...
state inputs/outputs + position/epoch and effects
preconditions: shape, sparse IDs, context, alignment, stable tensor identity
reference CPU function + external independent oracle fixture
backend kernel candidates and legal shape ranges
numerical rules: accumulation, rounding, selected-set invariants
allowed fusions/layout conversions/snapshot cuts
required scratch/lease/event and graph capture eligibility
test categories and performance profiles
```

No opaque "custom op" without a reference/state contract. If the full effect semantics are not representable, reject as unsupported instead of compiling a seemingly plausible graph.

## 2. Dense linear / embedding

**EmbeddingGather** maps token ID to a row; must handle model-tokenizer vocabulary, special token ID limits, quantized embedding storage where supported. N-gram/Engram lookups are separate ops.

**Matmul/Matvec/QuantizedProjection:** logical matrix `[out_dim,in_dim]` independent of stored block quant, padded K and transposed device layout. Backend may fuse dequant inside projection or prepack weights. Prefill GEMM, decode GEMV and verify grouped MMQ variants. Need A/B shape/error tests and explicit accumulator/rounding.

**RMSNorm/LayerNorm:** epsilon and reduction dtype, scale/bias order; fusion with quantization only if numeric policy passes independent oracle.

**Activation/GatedFFN:** GELU/SwiGLU/other exact model activation, gate/up/down row geometry and rounding, not interchangeably optimized.

**RoPE/MRoPE/Position:** position ID and axis/sections, static extension factor; long-context changes can affect quality. Need exact image/3D position mapping.

### Candidate region: Norm + Q8 activation packing + GEMV
Effect: pure tensor (unless intentionally fused with residual state). Inputs: current activation/weights. Output: same logical projected rows and declared numerical tolerance. Benefit: avoid extra activation write/read and kernel launch. Risk: register spill, FP32 rounding, block tails. Fallback: separate norm, pack and projection with explicit scratch.

## 3. Attention families

### Full causal attention / GQA
Q/K/V heads relation, causal mask, optional sliding window, position rotation, KV storage (paged). Prefill large attention tile, decode KV/gather, verification multirow. Fusion candidate QKV projection+RoPE+cache store where ordering/capture permits; dynamic KV page pointer reads require leases.

### Sliding-window / alternating local-global
Window is semantic (not arbitrary heuristic). Must preserve exact local positions and boundary; global layers cannot reuse same narrowed history. Gemma variants' attention scaling/positions declared by model recipe. Test sequence >window, uneven chunks and prefix fork.

### Qwen Sparse Attention (QSA)
Compressed/indexer rows, pooling, scoring, top-k block selection and masked attention; choose exact selection ID algorithm, tie ordering and scalar precision. At 128K sparse selection must remain on GPU for peak `gfx1151` plan. Separate candidate fused selected attention vs selector→attention; bounded scratch, no approximation to lower top-k without explicit quality mode.

### GLM latent MLA + DeepSeek Sparse Attention
Latent `c_kv` compressed state, NoPE where model says so, query absorbed projection option versus expanded K/V fallback (different numerical/physical layouts). Sparse indexer head geometry/top-k differs from QSA; do not use Qwen indexer kernel unmodified. Effect: append valid latent/indexer state and publish causal compressed rows. Legal conversion only at checkpointed frontier and test against model reference.

### DeepSeek time-compressed attention (CSA/HCA)
Raw most recent window and compressed rows indexed by time ratio (4 and 128 in DwarfStar research), some layers have indexer top-k. Distinct state update for compression pool and raw ring, exact causal reveal of completed compressed blocks. Never infer that ratio-128 compression is a performance tweak; it is trained model semantics.

### Candidate region: index score+topk+attention tile
Pros: fewer global scratch writes and launches. Cons: divergent dynamic top-k, register/LDS, numerical score ordering. Must test masked/empty selection, selection ties and long-context full logits, not just attention output relative error.

## 4. Recurrent and hybrid families

### Gated DeltaNet
State matrix and per-layer short-convolution history, gates, input projections; chunk prefill scan and one-step recurrence have distinct efficient kernels but semantically same transition. Model-specific heads and gating. Every speculative target row must have post-row checkpoint candidate, or an exact recomputation strategy. Causal dependency forbids parallelizing recurrent tokens as if independent ordinary attention.

### Kimi Delta Attention
Fine-grained per-channel gates/decay and potentially lower-bound scalar as released config requires; not GDN scalar-gate approximation. State large enough that eager snapshot per draft row may cost substantial memory. Fused scan should exploit algebraic associativity only if exact model recurrence and numerical policy allow. Long-context and chunk-edge parity mandatory.

### Hyperconnections
Qwen-style gated residual and GLM/DeepSeek mHC manifold-constrained mixing are distinct. mHC Sinkhorn iterations/normalization should be separate canonical op; fusion might change rounding; test full width, multiple stream slices and tool continuation quality.

### Candidate region: recurrent step + conv history + residual
Benefit: reduce small GPU launches and intermediate activations; preserve versioned recurrent state and snapshot access. Risk: no rollback snapshot at intermediate draft positions or scratch register spill. Legal only with explicit state outputs for each committed/provisional row.

## 5. MoE families

**Route:** logits→score transformation→top-k selected IDs/gates (signed/sigmoid/softmax/routed bias and scaling). Some models use auxiliary normalization; model recipe must dictate formula.

**Group:** token row assignments compacted into `expert -> row list`, with duplicate selected experts across rows computed together; preserve original row and gate association.

**RoutedExpert:** gate/up/down projections quantized per tensor; small decode and large prefill kernels can differ. Padded stored dims not semantic computation dims.

**SharedExpert:** often computed every token, may overlap selected expert IO/computation if dependencies permit; if output merged after routed contribution, reduction order/precision explicit.

**Aggregate:** deterministic or fast atomic reduce based on numerical contract. Atomic FP32 order may change logits, so exact-replay path uses stable ordering.

### Candidate fusion plans
- Router+selected IDs+GPU pointer table lookup: no host round trip on hit.
- Grouped expert gate/up and down: exploit input activation reuse and fewer launches, with scratch and occupancy budget.
- Shared expert running on independent stream while expert miss IO fetches missing routed weights.
- Prefill grouped GEMM vs decode GEMV/low-row MMVQ, choose measured row distribution.
- Speculative verification grouped experts across candidate rows and independent sessions.

**Hard guard:** expert-cache eviction cannot overwrite a slot with any still-queued reading kernel. Fused region must declare active leases and events or illegal.

## 6. N-gram / Engram row lookup

Operator has read-only huge backing file, small input token-history state and deterministic row-hash/lookup semantics. CPU may compute hash with bounded `pread`; GPU projection consumes tiny staged rows. Cache duplicate requests by `(artifact, row)`, no blindly mmap entire 95/189 GiB table as resident. A missing/corrupt row is an IO error that invalidates provisional model step; no zero-filled default or reused scratch.

## 7. Output/sampling/drafting

**LMHead:** large vocab projection with optional quant, shape-tail checks, model logit scaling and precision. **Sampler:** temperature/top-k/top-p/min-p/penalties and RNG, result emitted only upon state commit. **Drafter:** MTP/DSpark/small model/prompt lookup propose tokens, not authoritative outputs. **TargetVerifier:** computes logits/states for proposal rows, applies mathematically sound correction. Fusing sampling onto GPU is allowed only if matching numerical policy and RNG state/stop logic.

## 8. Operator capability matrix example (PROPOSED)

| Operator | CPU ref | Generic GPU | Special gfx1151 | Special CUDA | Notes |
|---|---|---|---|---|---|
| RMSNorm+project | required | candidate | wave32 fused | tensor-core suited | numerical rounding |
| Dense GQA | required | candidate | attention tile | Flash-style tile | paged KV/graph |
| GDN | required | candidate | recurrence+conv fused | recurrence fused | rollback every draft row |
| KDA | required | candidate | per-channel recurrence | per-channel recurrence | no scalar approximation |
| QSA | required | candidate | GPU top-k/sparse | sparse indexer | causal selection |
| Sparse MLA/DSA | required | candidate | latent/index tile | latent/index tile | NoPE and absorbed vs expanded |
| DeepSeek CSA/HCA | required | candidate | time-compressed tile | time-compressed tile | layer dependent ratios |
| Quant MoE | required | candidate | IQ2/Q2/Q4 grouped | FP4/INT8/Q kernels | route math fidelity |
| mHC | required | candidate | fused if valid | fused if valid | Sinkhorn/numeric policy |
| Ngram/Engram | required | CPU/IO candidate | async row stage | async row stage | disk table |
| MTP verifier | required | candidate | 2–8 grouped rows | grouped WMMA | exact commit |

All entries are implementation targets; **nothing in this table indicates zerogravity already runs them**.

## 9. Reference model and independent data flow

For each operation supply sample inputs and output SHA from official model where possible. If model code lacks a directly callable equivalent, implement independent pure scalar mathematical oracle and compare with at least two full-model reference points. Include state bytes or normalized state checksum after transitions. For quant, same stored bytes and dequant scales must be used by both candidates. Use official quality scores but avoid claiming official benchmark parity with different checkpoint/quant.

## 10. Region and fallback execution example

```text
SIR:
    norm -> gated_read -> GDN(state#7) -> gated_write -> router
        -> route_expert* + shared -> reduce -> residual

EIR candidate:
    fused_gdn_hc_region(state#7..state#8)
    routed_moe_gpu_region(quant IQ2, top10, rowcount 1..8)

If second kernel unavailable:
    preserve canonical activation at region cut
    generic routed MoE region executes known exact route/output
    state#8 remains owned and coherent
    continue next layer without full-model fallback
```

A fallback may require materialization and thereby lose performance, but **must not lose state correctness**.

## 11. Complexity budget

Add a new candidate only if it has a distinct valid hardware/workload regime. Avoid permanent flags for all experimental fusion combinations. Registry should expose a bounded priority list of winners and one portable fallback. Rejected variants may live in research fixtures but should not complicate release binary.

## 12. Missing official formula checklist

Before first implementation of GDN, KDA, QSA, DSA/MLA, mHC, CSA/HCA, n-gram/Engram and MTP:
- pin official model source/config to commit;
- derive precise equations and tensor shapes from released weight naming and reference executable;
- create tiny scalar oracle and actual-model vector fixture;
- document quantization sensitivity and state transition;
- only then write extreme kernels.

This stops model architecture names from becoming unsupported placeholders.

## 2026-10-10 integrated v2 addendum — effect-aware kernel and fusion ownership

Every operator catalog entry now has **four separate required contracts**: (1) exact SIR semantic formula and state effects, (2) RIR legalization/fusion/precision preconditions, (3) KIR backend candidate shapes/ISA/scratch and reference numerics, (4) CIR memory lifetime and async event/I/O dependencies. An operator is not performance-qualified simply because a generic graph can represent it.

Reusable operator families must remain first-class at SIR: ordinary GQA, sliding/full, GDN, KDA, Qwen sparse attention, GLM MLA/DSA, DeepSeek time-compressed attention and residual/mHC, routed+shared experts, model-native ngram/Engram, multimodal encoders and pooling. Only RIR may decompose these into lower-level primitives; a generic algebraic tensor rewrite must not erase a critical recurrent or indexing state effect.

**Candidate region boundaries:** fused norm→activation-quant→projection; QSA/DSA indexer select→masked attention; GDN/KDA recurrence+short-conv; router→stable counting-sort→grouped MMVQ/GEMM→deterministic reduction; trained MTP target verify+state snapshots. Each declares a canonical boundary and COW/convert fallback that preserves full state. For `gfx1151` grouped MoE, make `route` GPU-side, but don't assume greater group size wins when expert IDs are almost unique. For graph capture, dynamic expert table pointer generations must remain valid.

**Compile-stage tuning:** pure rewrite and target kernel candidates can be searched offline; avoid compiling dozens of combinatorial fusion variants per live request. A verified serving-mode-aware autotuning result means KernelPlan needs eager-versus-graph execution mode in its key. A selected candidate must beat baseline **at full-region or end-to-end level**, not merely microseconds in isolation. Custom Rust whole-model kernel accepted if and only if region semantics/state/numerics/resources are verified and the portable decomposition remains usable.

**New extension demonstration:** add a new trained sparse attention indexer under its own operator schema and CPU oracle; reuse existing GPU top-k/packed GEMM candidates where mathematically compatible, with no changes to `EngineCore` or tokenizer/HTTP.


## 2026-10-12 integration — `ProjectionRowSlice` and non-token stage optimization

New RIR candidate `ProjectionRowSlice` decomposes a pure linear projection's **disjoint output row groups** among capable compute owners. Precondition: all source rows covered once, compatible quant/scales/bias, independent per-output channel math until explicit join, no relocated causal state effect or whole-output layer norm before missing rows. `AssembleProjectionSlices` restores logical row order (possibly direct-to-layout), and downstream GDN, attention, shared FFN gates remain model-correct.

Source-demonstrated first slices: (1) GDN z rows in 16384-row SSM input, up to 6×1024 (default fork); (2) attention Q+gate of 16 heads via 8×1024 rows; (3) shared expert gate/up projections 2× eligible NPU GEMM. XDNA2 kernel is fixed 4096×2560×1024 BF8 example; GPU computes remaining rows. The exact NPU result first-use phase differs for each operator. Region choice may be GPU-only, GPU+NPU or other computation owners; no hidden hard-coded device name in SIR/ModelRecipe. Unsupported shape/batch/kernels/unsafe SoC → GPU-only lawful candidate.

**Separate mathematical correctness from quant encoding:** source model weights may be Q8_0, but NPU compute uses `v8bfp16ebs8` groups. Export per-op max error, first-use dependent state/logits and model quality under a declared numeric profile. Treat `TaskProgram::Vocoder`, `TaskProgram::DenoiseStep` and encoder pooling as future operator/task extensions with independent noncausal and iterative-state contracts, not as generic token projections by default.

## v4 refinement — useful op granularity and precise region substitution

Maintain high-level semantic op identity for stateful full/sparse/compressed attention, GDN/KDA/indexer, MoE routing, residual streams and model-specific media operators. Within a **closed region**, the compiler may lower into pure tensor primitives, fuse norms/quant/GEMM/router kernels, split independent projection output rows and use architecture-specific whole-layer kernels. **RIR/KIR are transient analysis/codegen views**; only canonical SIR and compiled ExecutableBundle need enduring identities initially.

A region candidate declares exact input/output, state effects, dynamic ShapeGuard/ragged offsets, tensor quant vs lossy compute encoding, scratch/leases and explicit safe fallback. A kernel may fuse multiple semantic operators but cannot hide state writes or publish TaskSession root. Numeric/effect legality and first consumer readiness precede adoption; measured whole model/TaskDriver performance dominates microkernel timing.

````

## Source: .agents/docs/PERFORMANCE_HYPOTHESES_AND_RISKS.md

Original UTF-8 bytes: 18664; SHA256 `2d2e1fed43f59dbf5ad1e36ceb7a668216bf0bef15c1fe4c568acf8921f5f060`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Architectural risks, opportunities, trade-offs and falsification criteria

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


Status: initial design research. Risks below are not implemented bugs. We must distinguish **observed external project evidence**, **deduced system requirement**, and **unproven zerogravity hypothesis**.

## 1. Risk register (severity × uncertainty)

| ID | Severity | Risk | Mitigation/proof | Kill or pivot trigger |
|---|---|---|---|---|
| R-01 | Critical | Rust GPU compiler cannot produce competitive `gfx1151` kernels | P01 ISA/occupancy/real kernels bakeoff, add target intrinsics | consistently poor unavoidable codegen in key ops: implement custom lowering or revise toolchain |
| R-02 | Critical | Semantic IR omits state effects, producing wrong hybrid/speculative outputs | typed effect tokens, CPU oracle, mandatory rollback tests | can't express snapshot/commit without hidden side effects: redesign IR before model work |
| R-03 | Critical | Physical APU memory/GTT overcommit hangs system | global arbiter, watchdog, conservative admission | unsafe default loads: freeze release and fail-closed |
| R-04 | Critical | Global planner complexity adds hot-path overhead | ahead-of-time region plans, no semantic interpretation during decode | measured launch/CPU scheduling latency material: simplify/compile direct plans |
| R-05 | High | Numerous specialized kernels create unmaintainable model forks | operator family registry, versioned model recipes, legal region interfaces | adding next model requires rewriting serving/session or many duplicate layers |
| R-06 | High | Quantized speed improvement degrades reasoning/tool accuracy | source/reference logits, KL/NLL, official tasks, explicit quant artifact | significant quality regression hidden by next-token smoke |
| R-07 | High | Dynamic expert-cache miss synchronization stalls GPU | device slot hits, async bounded ring, fixed slabs and lease safety | host round-trip dominates hit path or GPU hangs on IO failure |
| R-08 | High | Cross-request expert grouping costs more than reuse saves | route-overlap sweep and end-to-end P95/throughput | no repeatable net gain on representative workload -> disable policy |
| R-09 | High | Prefix COW snapshots consume too much memory | state-domain cost model, deferred snapshots, recompute alternative | branch creation/eviction costs exceed prefill saved |
| R-10 | High | Adaptively switching speculative policies harms latency/quality | exact target sampler, hysteresis, complete accepted-prefix rollback | low-acceptance workload worse than no-spec beyond tolerances |
| R-11 | High | Graph capture makes cache pointer/memory lifetimes invalid | stable device control table, event-backed lease, eager fallback | captured plan fails exact replay, graph disabled for region |
| R-12 | High | Distributed partial failure corrupts state frontier | handshake+epochs, collective commit and fail-closed | rank mismatch or stale partial reused -> stop deployment |
| R-13 | Medium | Every new format/architecture must be handcoded | semantic operator composition, explicit adapter and CPU oracle | parser accepts but cannot execute new op: report unsupported honestly |
| R-14 | Medium | Autotuner overfits to one shape/thermal condition | ABBA and end-to-end, cache version keys, regression suite | no real-model win or tail latency regression |
| R-15 | Medium | Broad API compatibility masks tool-tokenization mismatches | exact template/role/tool parsing, model-specific vectors | hidden reasoning/tool state leaks or incorrect prefix |
| R-16 | Medium | External source licensing and data rights | pinned attribution/license ledger, independent implementations | unavailable or incompatible license -> study behavior only |
| R-17 | Medium | Vendor driver versions break compiled plans | version fingerprint and cache invalidation, current physical tests | stale binary unsafe/wrong outputs -> invalidate/recompile |
| R-18 | Medium | Complex multi-crate setup slows contributors | few coherent crates, direct examples, decision/handoff docs | extending operator requires pervasive changes -> consolidate interfaces |

## 2. Central falsifiable hypotheses

**H-A Joint scheduling/residency:** dynamic batch/grouped experts will create extra weight reuse beyond separately tuned batcher+cache; predicts benefit at moderate concurrency with routing overlap, no benefit on one row/uniform routes. Falsify with H04/H05.

**H-B Semantic state-aware prefix sharing:** shared recurrent/indexer state prevents expensive hybrid prefill for repeated agent prefixes; predicts TTFT improvements on long common prompts if snapshot size and transfer less than recompute. Falsify with H07.

**H-C Adaptive speculation including memory:** choosing depth/drafter based on full verification+cache cost beats acceptance-only MTP policy across coding and reasoning workloads; expect little benefit on random text. Falsify with H08.

**H-D Rust-generated extreme kernels:** target-specific Rust kernels can equal or improve licensed reference kernels on `gfx1151` with enough backend lowering control; not established by Rust syntax. Falsify with H01.

**H-E No generality tax in hot path:** compiler/plan produces direct launches equivalent to hardcoded model runner, with no dynamic per-op SSA interpretation; falsify with H02.

**H-F Safe scratch borrowing:** flexible prefill scratch reusing cold expert-cache capacity improves real prefill without extra permanent GiBs or decode thrash; falsify with H05/H09.

**H-G Kernel variants per regime:** prefill/decode/verify tailored algorithms beat one generic kernel under same model quant and memory; strongly supported in a quant-aware inference project, Gufo and DwarfStar externally, but must re-prove implementation.

## 3. Key trade-off choices

### A. Semantic IR richness vs maintainability

Too little IR produces per-model manual code and unsafe state transformations; too much creates graph optimizer complexity/compile latency. Choose minimal typed SSA + explicit state effects needed by actual initial dense/GDN/KDA models; expand only with concrete proven operator. Preserve readable IR dump and per-node reference function.

### B. CubeCL vs direct hardware kernels vs vendor libraries

CubeCL: Rust syntax portability/cross-backend, fast development; possible codegen gaps. Direct AMDGPU/CUDA intrinsics: best control, more complexity; must preserve Rust maintenance. Vendor BLAS: peak tuned matmul and lower risk, but strict Rust-only interpretation and ABI/version questions; keep as optional until user resolves O-003. Compare actual shape/latency and source policies instead of dogma.

### C. One cache vs many caches

One budget arbiter required for host safety; one eviction policy across dense weights, experts, KV, recurrent and lookup rows is wrong. Each resource class needs distinct demand/reconstruction/cost and one global arbitration. Use typed leases and inventory to avoid independent hidden overcommit.

### D. Whole-layer fusion vs operation modularity

Big fused kernel reduces launches/global traffic but may spill registers, limit occupancy, obscure numerical testing and dynamic misses. Preserve decompose reference and choose candidate with full-model A/B; no mandatory maximal fusion.

### E. Full CPU reference vs portable but slow GPU backend

CPU reference essential to verify model math and unknown operator; a generic backend may be portable but lack needed FP8/atomics/context and thus unable to execute giant model practically. Support statuses separate; model works where device resources and all ops available, otherwise honest error.

### F. Automatic adaptation vs predictability

Runtime can adapt prefill quantum, batching, speculation and caches at request/step boundaries. But frequent adaptation can thrash plans/scratch and make reproducibility difficult. Use measured cost model, bounded variant count, hysteresis, deterministic quality mode and logging of decisions.

## 4. Common failure of reasoning to avoid

- "IR compiles model" therefore faster: false without kernels, correct layouts and memory.
- "125B parameter" means stream 125B every token: false for sparse MoE and disk-only lookup tables; active parameter/quant/traffic important.
- "All experts fit GPU-addressable GTT" means RAM safely available: false on UMA.
- "Continuous HTTP clients" means true native batched matrix: false when ordered per-session fallback.
- "60 tokens/s speculative" means 60 tokens/s serial reasoning: false; acceptance/workload-specific.
- "General GGUF reader" means any architecture supported: false; model semantics/position templates separate.
- "Bitwise same tokens" means same model quality: false, and cross-plan floating order can change close ties.
- "One high-quality kernel" means full model faster: false; real end-to-end critical path.
- "SSD memory backing" means all huge models interactive: false; random expert misses may dominate.
- "Linux driver supports graph capture" means our ROCm graph capture implementation works: test physical backend.

## 5. Decision process for research conflicts

When a new paper/project/repo claim contradicts this design, record exact commit/config/metric, identify if semantic contract actually differs, reproduce small test, compare required complexity and update decision register and risk. Benchmark evidence overrides attractive architecture diagrams. If can't reproduce, mark uncertainty. **Never silently rewrite history**: add decision superseding old ID, date and rationale.

## 6. Prioritize and prune complexity

If an optimization consumes substantial design complexity without a repeatable full-model win on representative workload, keep the simpler path and note the rejected experiment. Reserve long-term extensibility but build only proven interfaces and at least one complete model end-to-end before distributed/autotuning breadth. User's performance ambition is ambitious; over-engineering jeopardizes it.

## 7. Immediate actionable risks

Before implementation, resolve:
- exact vendor-driver/FFI interpretation of "entirely Rust";
- whether `gfx1151` available for repeatable kernel perf testing;
- first complete dense and hybrid model artifact/checkpoint choices;
- model quantization/checkpoint licensing for continuous integration;
- stable source pins of CubeCL, Gufo, Maya, DwarfStar, GGML;
- quality threshold policy and physical memory watchdog requirements.

These are open decisions or empirical feasibility gates, **not** instructions to ask the user to repeat already answered requirements.

## 2026-10-10 integrated v2 addendum — new research risks and falsifiable differentiation

**Reconciliation finding:** old research already identifies joint expert-affinity scheduling, MTP-as-group, scratch/slot solver, state frontiers and capture-stable expert table; those are not wholly novel zerogravity inventions. The *new potential* comes from embedding them in SIR/RIR/CIR's unified effect/resource-aware compiler, so one policy can legally reuse results across model families and hardware. No outperformance is guaranteed.

**Additional high-priority risks:**
- R-19: **CIR overbuilding** — attempt to recreate a general-purpose stream/IR runtime before a real model. Mitigation: minimum four commands+events+lease sim first; pivot if no legal overlap or too much compile/hot overhead.
- R-20: **GPU codegen lock-in** — selecting one Rust GPU DSL/compiler version before physical `gfx1151` feasibility or considering alternative Rust CUDA tile/SIMT compilers. Mitigation: capability interface, independent bakeoffs and late binding.
- R-21: **graph effect legality drift** — replacing closed region while GDN/DSA/mHC and grammar effects mismatched. Mitigation: state-domain verifier and forced op fallback tests.
- R-22: **too many optimizer rewrites** — bounded graph-search compile-time explosion. Mitigation: bounded pure-region search and offline only, evidence before default.
- R-23: **3+ hybrid prefix groups** — naive two-group hit-intersection algorithm generalized incorrectly or partial compressor row. Mitigation: exact `CanResume`/alignment/restore per group and synthetic 3+ group tests.
- R-24: **false Rust-only conformance** — vendor C++/HIP escape imported without explicit user decision. Mitigation: Rust owned kernel source policy and external API/driver distinction.
- R-25: **unapproved product/release decisions** — historical research council "resolved" license/MRSV/version treated as user approval. Mitigation: decision provenance IDs and versioned rulings.
- R-26: **false universal model fallback** — unknown model math mapped to plausible but wrong dense ops. Mitigation: registered semantic op and reference oracle required, no unsupported custom code execution.
- R-27: **incorrect binary serialization** — historical research 48/54-byte KV header and magic conflict. Mitigation: checked schema-first ABI fuzz and no adoption by copying.
- R-28: **model/user hardware assumption** — old research 6GB 3060/27GB and 128GB Strix purchase treated current. Mitigation: real capability probe and user-confirmed availability only.

**Highest-upside hypotheses and test:** (1) SIR/RIR/CIR lets us optimize asynchronous expert/lookup/kernels without model fork—prove 2 model families share compiled region implementations; (2) state-aware prefix sharing across GDN/KDA/indexer significantly reduces long-agent follow-up TTFT—prove C8 shared-prefix state vs recompute; (3) GPU route-aware batch×expert residency×MTP boosts committed tokens/sec at quality/P95 parity—compare baseline sequential and grouped; (4) offline rewrite/tactic search finds stable wins under captured serving mode—compare serving-aware kernel-tuning research-style tactic timing; (5) new Rust GPU compilers close ISA gap without C++ kernel source—physical gfx1151/CUDA parity evidence.

**Maintainability test:** ask external agent to add a new model attention state domain and alternate GPU backend without touching HTTP, model store, sampler, worker core outside stable interfaces. Document file count changed, compile time, added tests and duplicated code. Failure triggers interface redesign even if early benchmark favorable.


## 2026-10-12 Gufo XDNA2 opportunity and risks

**H-I — cross-accelerator independent GEMM slicing:** RIR can select disjoint output rows for a known GEMM projection, overlapping GPU and NPU with first-use join. Gufo fork author reports +7.2% prefill on one Qwen UD-IQ4_XS 16K case; shape/precision/safety/clock matter. Test exact source checkpoint, held clock equal ON/OFF, output/logits/state parity, real async overlap and pinned LPDDR critical path. Kill default NPU use if full-request gain inconclusive or precision quality inadequate.

**R-29 Critical — machine-wide GPU/NPU hard-hang safety:** reported GPU fabric-clock transitions simultaneous with NPU produce machine checks and freezes. Not recoverable by ordinary process watchdog. Require opt-in, isolated physical soak, firmware/XRT/ROCm/FCLK matrix and no automatic privileged system changes; GPU-only always safe fallback at admission.

**R-30 High — Rust AIE compiler/driver viability:** Gufo source is C++/Python with executable `.xclbin`. A Rust wrapper is insufficient for user Rust-owned kernel policy. Must prove Rust-maintained AIE codegen or explicitly record unsupported XDNA until user resolves source policy.

**R-31 High — NPU BFP8 numeric drift:** 9-byte shared exponent/8 mantissa compute can perturb GDN/attention indexer output despite matching greedy test. Add strict reference full-logit/quality/target verifier gate and deny offload under unsupported numerical mode.

**R-32 Medium — multi-task/control loop explosion:** upstream Gufo ASR/TTS/image/video cannot be forced into autoregressive next-token state; TaskProgram with task-local effects/loop/output under shared SIR/CIR needed. Prevent per-model competing inference servers or huge unbounded generic framework.

**R-33 Medium — FCLK power/thermal confounding:** held-clock alone may improve GPU prefill but increase idle consumption and physical risk. Bench same-clock ABBA and energy/24h, not NPU ON held vs GPU OFF automatic. Do not claim all `gfx1151` machines have identical AGESA/driver safety.

## v4 critical-path and complexity risks (review R4-01…R4-24)

**New critical risks:** partial state publication despite individual StateDomain commits, device kernel "done" confused with TaskSession commit; host↔GPU router/sample sync each token; global allocation lock per kernel under C32; shape specialization explosion; compiled kernel caches keyed accidentally on live request IDs; unchecked numeric/top-k changes; distributed rank lost ACK and false network exactly-once; workflow DSL hidden inside core and unneeded training/autograd bloat.

**Mitigation:** prepare all domains then one logical PublishRoot; GPU-native StepProgram only for supported exact state/sampler modes; per-device suballocators under coarse global admission; bounded symbolic ShapeGuards and bundle cache; source quant/compute precision/task quality independently versioned; fencing epochs; sequenced output with cursor policy; trusted minimal TaskDriver and external tools above TaskRequest; one library+binary initial rather than many crates. Hardware safety and Rust-owned kernel requirements remain independent of mathematical correctness.

**Falsification:** H32–H43, Stage0 S0-V4-A…J and a second-model/new-device/non-token-task change-radius drill. A new AI model/domain that forces edits to unrelated service/runtime/physical allocator fails the modularity test; a high-performance kernel that adds unbounded CPU work, P95 regression or unsafe memory/clock changes fails optimization qualification. The review document preserves detailed severity, evidence and decisions.

````

## Source: .agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md

Original UTF-8 bytes: 20666; SHA256 `a505cac71b058fdd05fe43edcdd785eca0ccf7cf482b16be9e8e66a47148b316`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Product, model loading and extension contract across the full pipeline

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**Status:** consolidated working design. Implements lessons from prior format-architecture and cross-cutting multimodal research and the project SIR/state/serving docs. Defines what an agent adding a new format, model, modality, endpoint or quantization must change—and what must remain untouched.

## 1. Source/format are not model mathematics

The model-loading contract should be:
```text
Resolver{path | verified HF revision | ModelStore tag}
  -> immutable ModelArtifact + complete SHA/manifest
  -> bounded GGUF/safetensors parser and tokenizer/rendering source
  -> ArchDef(ModelRecipe) maps declared tensors to verified semantic ops
  -> typed SemanticModelIR + StateContracts
  -> Compiler/ResourcePolicy/Kernels
  -> independent EngineCore/Frontend
```

No loader should directly choose `gfx1151` fused kernels. Model adapter declares exact trained formulas, all tensors and layout constraints (could be declarative for repetitive transformer families) while backends know only semantic ops/shapes/quant. Unknown model architecture, quant, tokenizer pre-type or new positional math gets a diagnostic refusal, never reinterpretation.

## 2. ModelStore and manifest versions

Borrow Ollama-like content-addressed blobs and manifest lists:
- immutable binary layers addressed by SHA-256;
- named model tag maps to manifest specifying base weights, shards, tokenizer/config/template, optional vision projector, LoRA/adapters, MTP sidecar, calibration/imatrix, license and **derived acceleration pack**;
- atomic resumable download with bounded size/cryptographic hash; explicit offline mode, no arbitrary code execution;
- store's garbage collector computes reachable blobs from referenced manifests, pins and active sessions—not file mtime alone;
- model switching/unload waits until GPU graph and buffers release their leases; model identity tied to actual bytes and recipe schema;
- user-effective file/cache locations XDG-aware with defaults subject to UX decision, private prefix snapshot cache separated from public model weights.

**Optional accelerated pack:** contains losslessly packed device tensors, per-expert one-blob index, layout descriptors, kernel plan provenance, and possibly explicit quality overlays which are *separate derived model artifacts*. Original GGUF/safetensors remains loadable without pack; pack invalidates if source checkpoint/compiler/layout revision changes. The research called it "APK"; retain only as historical label to avoid conflating with Android installation packages. Define a stable public extension after quality/format tests, not now.

## 3. GGUF and quantization schema

GGUF parser contract: v2/v3 header, LE integers, tensor metadata, name uniqueness, alignment power-of-two, checked 64-bit dimension product/offset+length, per-file dialect and tensor type registry, multi-shard completeness and tokenizer metadata. `ne[0]` is stored contiguous in GGUF, not necessarily a row-major host matrix. Must not treat GGUF type id from ik dialect as mainline without dialect detection. Unknown types refused with type id and precise fix.

Quant sources: native F32/F16/BF16/FP8/MXFP4, GGUF Q8_0/Q4_0/Q4_K/Q5_K/Q6_K and IQ2-family as needed; supported types are an evidence-based matrix, not every known type promised at v1. Store `logical_dims`, `stored_block_shape`, `scale/zero metadata`, `packed_layout_id` and `calibration_id` distinctly.

Internal packing:
- lossless reorder/transpose may be automatically selected when validation proves dequant/oracle parity;
- deliberate quantization changes produce a new source/derived checkpoint fingerprint and quality bands, never silently chosen to fit memory;
- model-sensitive precision allocations: always-on norm/router/mHC/indexer/latent state may require more precision than selected routed experts; not a one-bpw-for-all policy;
- imatrix calibration collects per-channel activations/routed expert histograms from an explicit offline `Tune` graph mode, not a surprise production telemetry path; independent calibration dataset license and reference quality measured;
- overlay quality/speed patch keyed to base artifact and declared sensitivity tradeoff, not silently applied.

## 4. Tokenizer, chat templates and constrained outputs

Product use depends on token fidelity as much as GEMM speed.

`Tokenizer`: `encode` with special policy, `decode` and byte-exact `token_to_piece`, vocabulary size, EOG and GGUF pre-tokenization type. Golden fixtures against official source and pinned llama.cpp; Unicode/invalid UTF-8/special-token injection/role boundaries.

`PromptRenderer`: model-native Jinja-compatible templates (`minijinja` possible), strict undefined behavior, control-token-safe input marking, tool and reasoning headers; returns structured rendered token IDs plus canonical hash. No arbitrary template code with host file/network access.

`ToolParser`: incremental model-specific Hermes/Qwen, Llama, Mistral, Harmony, DeepSeek/GLM, generic JSON/tool fenced formats as supported; separate hidden reasoning versus visible assistant response; stops across token/UTF-8 boundaries; versioned capability matrix.

`GrammarFSM`: constrained decoding from JSON schema/grammar (a grammar library via Rust binding is candidate), token trie/pretype matching, compiled asynchronous with bounded CPU time/memory; accept/mask/rollback state is part of Session transaction; masking order follows exact selected model sampler distribution. Unsupported arbitrary GBNF should refuse explicitly rather than silently accept incomplete grammar.

Strong user-facing requirement: `/props` or `inspect --explain-plan` returns template/tool/grammar/vision/embedding/quant/backend capability and important typed unsupported reasons. No fake feature claims.

## 5. Text generator and encoder/pooler are different runner modes

**Decoder-only generation:** KV/recurrent state, token-by-token sampling and speculative verification, token-budget batching and stream. **Embeddings/encoder-only models:** no autoregressive decoder/sampler/KV in ordinary pooling path; batch by length/encoder budget, hidden-state pooling, normalization/dimension truncation and precision reference. **Cross-encoder rerank:** model of query/document pair, score head, document batching, template length/truncation and calibration. These are semantic graph recipes that share kernels, devices, formats and resource arbiter, but not a fake common "generation loop".

`PoolingSpec` should encode `Cls | Last | Mean | Token | Multi`, last-token side, model-specific L2 normalize and Matryoshka dimension ordering, padding/masks and output dtype. API `/v1/embeddings` and `/v1/score`/`/rerank` only when exact model/response schemas tested. Streaming input semantics for video/audio encoders can reuse modality graph but is not guaranteed by a text executor.

## 6. Multimodal and modalities as typed state

For each input `MmItem`: content fingerprint, raw/preprocess URI or validated blob, modality type, pixel/sample limits, resize/patch/encode geometry; `MmSpan`: token positions, embeddings, M-RoPE/axes, causal/bidirectional window. Encoders/projectors are model-like graphs with their own `ModelRecipe`, `DevicePlan`, computation/memory budgets and exact state/cache fingerprints.

**Critical:** a noncausal/bidirectional image span cannot be split arbitrarily into prefill chunks as if it were ordinary decoder tokens. Scheduler declares `supports_partial` and span chunk boundaries; refer to the cross-cutting multimodal-embedding research. Vision embedding cache and KV/recurrent prefix cache have related but distinct identities; both must be available to claim full prefix reuse. Image edit invalidates affected snapshot, even when visible text prefix unchanged.

Speculation with image/video histories must be separately verified; can be disabled by capability until state restoration is correct. Do not promise MTP support merely because text-only model uses it.

**Future modalities:** audio input and live streaming, video temporal positions, new vision patch schemes, action/robotics/embeddings, potential diffusion/flow engines; integrate new semantic operators and loop/sampler policy without forcing old decoder-only code to become a monolithic switch statement. The product does not promise text-to-image/diffusion at initial release.

## 7. LoRA/adapter model (previous project gap)

LoRA must not be a simple mutable `adapter_name` global on a loaded model. Required contract:
- adapter artifact references exact compatible base model tensor layout, rank, scaling and affected projections;
- per-request/session `AdapterSet{generation, weights, weight_multiplier, compatibility_digest}`, plus safe LoRA kernel candidate or portable reference implementation;
- prefix cache key includes adapter identity/generation and any request-time blending; cannot share cached state across incompatible adapters;
- memory manager budgets adapters separately from base immutable weights and in-flight requests; hot swap preserves old sessions until references retire;
- batching may group compatible adapter rows for fused kernels; opposite adapter combinations cannot accidentally share projections;
- dual quant/LoRA transformations must be independently quality-tested with official known vectors;
- trained MTP/drafters may be incompatible with adapted target distributions, so speculative drafter needs explicit compatibility or falls back.

Defer complex multi-adapter batched fusion until base generation and state contracts validated; CPU reference for adapters comes first.

## 8. Platform and hardware extension contract

New AMD/NVIDIA/Apple/Intel/ARM/NPU/remote accelerator path adds `DeviceCapabilities` probe and driver+stream+event/memory tier backend. It can select generic kernel provider where supported or implement target-specific Rust-authored kernels. It never changes `ArchDef` or user prompt parsing. NPU sidecars for encode/prefill are only feasible if state movement, quant support, driver coexistence and physical bandwidth/latency are proven; do not disable IOMMU or security system for speculative speed gains automatically.

Hardware detection uses actual architecture/capabilities and OS/driver status, not marketing product names. Quant and device ISA matters at least as much as numerical model family.

## 9. Serving facade

One canonical `InferRequest` with optional `TextGeneration`, `Embedding`, `Rerank`, `Multimodal` task kind; frontends for OpenAI/Anthropic formats where conformance passes. Edge performs validation, deterministic rendering/tokenization, auth and quotas, then sends request to EngineCore. Engine core performs scheduling/commit; worker model execution; frontend streams only verified output. A tool executor/MCP app is an *independent high-risk permissioned subsystem*, not implied by text-generation support. Avoid web fetch/file access from a model input by default.

CLI has simple `run/serve/models/inspect/doctor/bench/convert/quantize/pack` candidates, default simple local use, offline model trust controls and optional XDG model store. Use one structured error/refusal taxonomy with fixes, no arbitrary low-level flag burden.

## 10. Conformance matrix for every extension

For a **new model**: checkpoint and tokenizer/template hashes, SIR operator registry and state effects, CPU oracle logits, model task quality, generic backend, optimized candidate, context/quant/multimodal limit and fallback.

For a **new format**: negative/malformed vectors, odd alignment/multishard/mapping, model conversion roundtrip exactness or labeled quality if quantized, data license.

For **new GPU**: actual physical driver setup, dtype/ISA capability, buffer alloc+copy/event/capture, generic kernel smoke, upstream comparable benchmark, explicit unsupported ops and memory cost.

For **new endpoint/grammar**: official client schema, data/model state and error tests, streamed output/stop/timeout, privacy, auth and capacity.

For **new adapter/modality**: state cache identity, downstream sampling and position logic, memory reservation, snapshots, exact parity and cancellation.

## 11. Explicit non-goals and open points

No direct execution of unknown HF Python architectures; no automatic quant quality downgrade; no untrusted executable model plugins; no unconditional claim that 156+ llama.cpp families work; no mixing session/adapter cache across tenants; no public server without auth by default; no promise of universal MTP/LoRA multimodal support. First model families and extensions should be prioritized by real demand and feasible validation—user explicitly selected modern families but not v1 support matrix.

**Strong recommendation:** preserve this as the single product extension map; use existing `MODEL_SEMANTICS_AND_COMPATIBILITY.md`, `FORMATS_QUANTIZATION_AND_LOADING.md`, `RUNTIME_SERVING_AND_UX.md`, `RUST_WORKSPACE_AND_ENGINEERING.md` for detailed subtype contracts, not duplicate entire parser/runtime implementations.

## 2026-10-12 integration — upstream Gufo task pipelines and reusable TaskProgram

The latest upstream Gufo model suite is broader than decoder-only text: **Qwen3.8-27B/DFlash2 vision, Flash-Next QSA/GDN/MTP, DeepSeek V4/DSpark, Qwen3-ASR, Qwen3-TTS (speech synthesis/voice cloning), Qwen-Image-2.1 generation/editing and MiniMax H3 text-to-video/audio in progress**. Pin each upstream model's version and quality/benchmark independently. A unified zerogravity `TaskProgram` adds explicit `Preprocess/Encoder/TokenDecode/VerifyDraft/Pooling/Vocoder/DenoiseStep/DecodeMedia/OutputMux` stage semantics and task-specific iteration/output frontiers, while reusing the **same SIR/RIR/KIR/CIR, KernelPlan, device registry, memory arbiter and artifact store**. Do not force all audio/image/video into `EngineCore::next_token` or declare work-in-progress upstream models supported in zerogravity.

**XDNA2 NPU as generic stage/slice owner:** no new model-specific NPU server. For a qualifying image/text/audio encoder, RIR may split independent projection output rows or whole eligible GEMM stages onto NPU while GPU performs unrelated computation, through typed task-stage dependencies and memory/fabric qualification. The currently source-demonstrated Gufo fork only accelerates specific **Qwen3.8 Flash-Next prefill GEMM slices**, so all other NPU uses are unverified hypotheses. Each stage must have its own valid numeric/model quality and model artifact fingerprint, cost/occupancy and checkpoint/cancellation policy.

Audio/image/video stage loops differ: streaming ASR with encoder/CTC or seq2seq as model-defined, TTS acoustic autoregression plus vocoder and PCM stream, image generation with iterative flow/diffusion/conditioning and denoise timestep state, video with frame/temporal conditioning and A/V sync. Some models have no KV cache, some have noncausal spans or long-lived latent state; `TaskProgram` models this explicitly and `CIR` schedules GPU/NPU/CPU resources. ModelStore supports encoder/projector/vocoder/decoder sidecars and individual licenses. All unsupported task paths refuse clearly while text-only remains usable.

See [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) for the full stage graph and [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md) for source citations.

## v4 refinement — TaskDriver, not universal workflow DSL

The shared product execution library exposes trusted built-in **TaskDrivers** for source-qualified text, encoder/embedding/rerank, ASR, TTS/vocoder, image denoise/edit and video temporal processing. Each has its own typed stage frontier and output stream; device step repetition can be compiled into a StepProgram instead of polling a generic workflow graph for every token/frame. `ExternalAction` from older TaskNode sketches is **removed from the inference core**: retrieval, tool execution, agent workflows and file/network actions are separate permissioned *application adapters* above TaskRequest.

Task requirements are model-independent but cannot imply a task works without source adapter, codecs, semantic operators and task quality tests. Optional future `TrainStep`/gradient update can operate on a new ParameterVersion and optimizer state, while ordinary inference keeps immutable model views and no mandatory autograd/training runtime. ModelStore remains a simple immutable manifest+blob source, not a workflow scheduler.

## v4.2 — TaskDriver distribution without forcing token-only topology

The same TaskDriver can run its stages locally or on authorized remote RegionExecutionProviders; the task's **committed frontier and output type remain unchanged**. Image/video/ASR/TTS may place encoder, transformer/denoiser and decoder/vocoder on different machines when stage output transfer is cheaper than weight loading or resource contention; no forced causal token/KV semantics for media. Distinct parameter/model sidecars and source rights matter when shipping staged model weights or private audio/images across peers.

For language tasks, prefill/decode disaggregation transfers **all required hybrid session state**, attention/FFN disaggregation transmits per-layer activations, remote prefix cache validates StateCompatKey, and token-range KV can be partitioned only for mathematical attention schema with a valid summary reduction. TaskRequest/application agent tool authorization does not expand because a tunnel peer exists; network execution remains an explicitly permitted compute action, no arbitrary remote process permissions.

Program and cache identity includes source checkpoint, adapter, quant, position, media fingerprint and state schema, not only model name. A device lacking remote state snapshot can execute a **stateless encoder** but cannot pretend full interactive decoding support. [Placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md).

## v4.3 — registry model profile vs immutable source and execution residency

The ModelStore's immutable content-addressed artifacts/tensor/quant/tokenizer/sidecars are **RegisteredModels**, while a human-friendly `ModelAlias/ModelProfile` is a versioned typed overlay and a warmed `ResidentModel/Replica` is a physical model deployment with executable bundle/resource leases. Model loader never owns HTTP routing, and a changed `alias` revision cannot change a live TaskSession's `ArtifactKey/StateCompatKey` source/adapter/tokenizer. Two profiles of the **same source** may differ in sampling/context or load placement but still obey exact model identity and compatible compiled/state/quant policy; a different quant is a different verified immutable artifact, not a runtime sampler option.

Native catalog operations proposed `import/pull/list/show/alias/profile/rm`, atomic manifest and digest, license/size/source policy and GC preserving referenced blobs; no implicit remote download of unknown model names or arbitrary model-supplied code. Model autoload is *only* from authorized locally available sources unless user explicitly initiates download. A `ModelProfile` may name a trained draft, LoRA adapter, encoder, reranker or vocoder as **separate logical model roles** under a TaskProgram admission; they are not replicas/automatic target-model switches.

Provider Device/Task capability matters: unsupported architecture/quant/TaskKind shouldn't appear READY. Registry can expose model source metadata without allocating weights, while ModelResidencyManager performs safe physical reserve/warm/unload via existing global arbiter. [Model router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [zero-config UX/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

````

## Source: .agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md

Original UTF-8 bytes: 31781; SHA256 `0a5412cf92701b7c6a4a8ab22fd75d8746a31b44ee882890c7e7509856ab6352`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity Real-Time Multimodal Serving, Sessions, QoS and API Contract — v5.1

**15 Oct 2026 — real-time concurrent-goodput scheduling:** [Joint serving and three-level admission contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) extends global scheduling beyond text: ASR/TTS/duplex/audio codec frames, video solver steps and typed decision batches keep independent clocks and atomic output readiness while sharing physical resources, credits, placement and admission. Interactive text ITL and first-audible/packet jitter have declared **soft** QoS, not promises of hard real-time. A video job yields only at source-valid solver boundaries; live barge-in revokes one turn generation, not the GPU card epoch. No text tokens/s summation with acoustic/video units in goodput; per-class SLO/attempt/rejection/fairness measured separately. SERVOPT-31/36 are planned source-real mixed-workload qualification.


**15 October 2026 — STORAGE TIER MUST RESPECT SOFT REALTIME:** [Same source TensorView with GPU/RAM/SSD backing](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) is a legal serving placement mechanism, not proof that random SSD misses can satisfy duplex audio or synchronized AV frame deadlines. QoS admission evaluates complete ModelAssembly state, codec, median/p95 cold-cache miss and physical staging/OS headroom; opt for resident model components or typed refusal/explicit weaker policy when latency cannot be met. Media delivery/ack/playback ledgers remain separate from immutable source cache and committed neural StateRoot; no SSD IO task can publish media finality. Resident audio hot path has no per-frame host IO callbacks. [SSD-30/31/32 acceptance](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md).


**15 October 2026 • DESIGN ONLY, no protocol implementation.**  
**Governing constraints:** single Rust EngineCore and model execution fabric; TaskCoordinator is sole logical model StateRoot publisher, physical device leases retire only after genuine completion, network transports never mutate neural state or silently substitute models.  
**Companions:** UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md, VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md, SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md, MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md.

## 0. Scope and goals

Support local Rust CLI/SDK, offline files, streaming HTTP, OpenAI/Anthropic-compatible **tested subsets**, WebSocket and WebRTC live conversations, optional telephony SIP gateway, image/video async jobs and non-token decision endpoints with ONE canonical execution/admission/result contract.

No assumption that HTTP protocol compatibility proves native model math equivalence; no model inference work or binary media serialization inside a transport's JSON encoder. A source-verified Moshi/Omni full-duplex model and a composed Whisper→Qwen→Kokoro assistant share the same task/session/transport infrastructure but their TaskDrivers remain distinct.

## 1. Serving decomposition without a second inference runtime

~~~text
User / SDK / CLI / Remote authorized client
          |
HTTP / SSE / WebSocket / WebRTC / optional SIP or native Rust callbacks
          |
ProtocolAdapter: auth, request/media limits, capability matching
          |
SessionGateway: session id, transport state, input clocks, resumability
          |
ModelRouter: model alias+revision, replica, allowed provider, requested QoS
          |
AdmissionManager: task cost, physical/card memory, buffer/queue credits
          |
EngineCore -> trusted TaskDriver -> prepared ExecutableBundle -> DeviceWorker(s)
          |
Prepared domains -> TaskCoordinator PublishRoot once -> CommittedMediaEvent
          |
DeliverySession: packetization, delivery/ACK/playback tracking, bounded credits
          |
Same input transport(s) or stored job output
~~~

Protocol adapters are stateless or lightly stateful formatting/handshake layers. They do NOT allocate a separate model server or schedule kernels. Request's model binding, source revision, processor/codec identities, task semantics and numeric policy become immutable for its current session epoch.

## 2. Canonical TaskRequest and response mode

~~~rust
struct TaskRequest {
    request_id: RequestId, tenant: TenantId,
    mode: TaskMode,                              // exact task kind, not a string heuristic
    model_binding: ResolvedModelBinding,         // exact artifact/assembly/profile revisions
    inputs: Vec<TimedInputPort>,                 // typed clocked values, security/size guards
    output: OutputContract,                      // required types, clock/codec/formats/finality
    qos: WorkloadQoS,                             // requested soft deadlines/admission mode
    authz: PermissionScope,                      // local/remote/provider and data egress
    lineage: Option<SessionCheckpointRef>,      // authenticated compatible state only
}
enum TaskMode {
    Text, Embed, Rerank, DecisionSchema, ASR, TTS, Image, Video,
    NativeAudioDuplex, NativeAudioVisual, CompoundTask,
}
struct OutputContract {
    modalities: Vec<OutputPortContract>,
    partial: PartialPolicy,
    ordering: OrderingPolicy,
    error_mode: StructuredErrorPolicy,
    delivery: DeliveryGuarantee,
}
~~~

**Per-task model support is not implied by a family name.** The same RegisteredModel can advertise text-in/audio-out and image-in/text-out while refusing unsupported video-out. Compatibility adapters MUST reject or explicitly report unsupported parameters (e.g. voices, sampling, duration, frame rate, JSON mode, beam, LoRA, response modalities, output format). No ignore-and-proceed.

## 3. RealtimeSession and transport state

~~~rust
struct RealtimeSession {
    id: SessionId,
    binding: ResolvedModelBinding,                  // pinned immutable source
    task: TaskDriverId,
    neural_root: StateRootId,                       // owned exclusively by TaskCoordinator
    transport_epoch: TransportEpoch,
    turn_epoch: TurnEpoch,
    lanes: Vec<MediaLane>,                          // capture, perception, response, etc.
    transport: DeliverySessionId,                   // independent from neural state
    qos: EffectiveQoS,                              // admitted soft SLO, not magical guarantee
}
struct MediaLane {
    id: LaneId, kind: LaneKind, timebase: RationalRate,
    capture_cursor: Option<MediaCursor>,
    commit_cursor: MediaCursor,                    // model-valid state
    delivery_cursor: Option<MediaCursor>,
    acknowledged_cursor: Option<MediaCursor>,
    playback_cursor: Option<MediaCursor>,
}
~~~

**Separate scopes of epoch:**
- PhysicalDeviceEpoch: globally invalidates all queued work on one physical card after driver failure/quiescence transition. NEVER advance this for a normal one-session barge-in.
- SessionEpoch: invalidates all old TaskDriver outputs in one session, preserving unrelated sessions.
- TurnEpoch/OutputGeneration: invalidates only a reply's future uncommitted playback after barge-in; other session input lanes may continue.
- TransportEpoch: reconnect/renegotiation generation; MUST NOT accidentally reset Neural StateRoot or grant replay of old private data.
- ModelGeneration: model/adapter hot reload pinned at request/session admission; old active sessions remain on original immutable revision unless trusted replay/migration.

## 4. Five distinct frontiers

| Frontier | Authoritative owner | What is guaranteed | What it does not guarantee |
|---|---|---|---|
| Prepared | device/codec provisional work | candidate buffers exist | source-valid, ready, committed or delivered |
| Committed | TaskCoordinator / committed root | exact task/model criteria passed; output eligible | bytes transferred or played |
| Delivered | DeliverySession transport | bytes handed to transport or accepted for transmission | client actually received or listened |
| Acknowledged | verified client ACK | client reports receiving/buffering to cursor | user actually heard or viewed |
| Played/presented | client/player signal where available | client claims media playback/display progress | guaranteed human attention; server cannot independently prove it |

Capture/input ingestion has analogous received/decoded/model-consumed frontiers. ACKs may be absent, delayed, reordered or untrusted; keep the reported watermark provenance and never infer played=delivered. No false exactly-once delivery promise over network partitions. Logical StateRoot publication is **not rolled back** merely because a client went offline; conversational presentation history is reconciled through a separate compatible policy.

## 5. MediaEvent stream

~~~rust
struct MediaEvent {
    version: SchemaVersion,
    session: SessionId, request: RequestId, turn: TurnId,
    event_seq: u64, lane: LaneId, generation: OutputGeneration,
    clock: ClockDomainId,
    capture_pts: Option<MediaTimestamp>,
    presentation_pts: Option<MediaTimestamp>,
    duration: Option<MediaDuration>,
    commit_group: CommitGroupId,
    phase: EventPhase,                          // provisional/update/committed/retracted/terminal
    payload: PayloadRef,                        // immutable leased bytes/URI, bounded
    media: ValueDescriptor,
    mode_receipt: CapabilityAndEvidenceReceipt,
}
~~~

Recommended event variants: SessionReady, InputAccepted, InputGap, ActivityStarted/Ended, TranscriptInterim(revision,parent), TranscriptFinal, DecisionCommitted, TextDelta/Final, AudioFramePrepared/Committed, VideoFrameCommitted, PacketReady, PlaybackProgress, ResponseStarted/Interrupted/Canceled/Completed, RootAdvanced, UsageUpdated, ToolCallProposed (not executed), ExternalJobStatus, TransportReconnected, TerminalError.

Ordering:
1. Every lane has monotonically increasing sequence and generation; cross-lane arrival is NOT globally ordered merely because frames share a WebSocket.
2. Causal dependency IDs and commit groups declare meaningful cross-track ordering.
3. Provisional ASR hypothesis may revise with the same segment identity and a higher revision; committed transcript cannot be rewritten without explicit correction event/version.
4. Output packet PTS continuity, codec priming delay and media clock must be verified before publishing a complete/playable chunk.
5. No retraction of audio already played; a barge-in terminates future scheduled output and updates conversation presentation state accordingly.
6. Output chunks are leased; final physical backing cannot be freed merely when serialized or ACKed while a device still owns DMA completion.

## 6. Input clocks and buffering

- Audio capture: monotonic local hardware clock where possible; PCM channel layout, sample format, duration and discontinuity markers; explicit resampling with checked rational conversions and retained fractional remainder.
- Video capture: frame PTS and decode/PREP timestamps, image orientation/color/metadata; frame drop selection is explicit task policy, never equate video capture with fixed 30Hz if upstream source jitter occurs.
- Input text: source tokens are not acoustic time, even when streamed alongside input PCM.
- Jitter buffer: bounded by bytes, duration and discontinuity policy. Too much latency becomes a first-class degraded service receipt or admission error, not a hidden infinite queue.
- Echo cancellation/VAD: own signal-processing state, activation/release threshold policy and source processor identity; do not let VAD become a trusted model authorizing unrelated filesystem/remote actions.
- Out-of-order input timestamp: reorder within allowed jitter window, otherwise mark gap/late/drop; never put late media into an earlier committed model frontier.
- Clock synchronization: use checked rational duration alignment and source-registered resampler delay, not magic media FPS constants.

## 7. Playback interruption/barge-in protocol

Canonical steps for server- or client-origin VAD:
1. Ingress event marks activity_start at exact input sample or best-known approximate watermark; attach source and trust level.
2. Session control marks current response output generation canceled and rejects new work from that generation. It does NOT call physical-card-wide epoch cancellation.
3. Drop **unsubmitted** queue tickets and bounded outgoing playback buffers. Already launched kernels keep physical leases until genuine readiness or terminal quarantine.
4. Stop client playback promptly; record best client observed played/acknowledged watermark, distinguishing true player-signal from approximate transport receipt.
5. Model conversation visibility is reconciled by a TaskDriver/source-compatible history policy. E.g. transcript for unplayed speech may be discarded or explicitly marked inaudible; cannot pretend emitted text was heard.
6. Retain already committed input samples and ongoing perception lanes when model permits simultaneous input/output.
7. New reply starts at a fresh TurnEpoch and source-verified model state; no stale prepared chunk, codec residue or old asynchronous tool callback may publish under new epoch.
8. Report interrupted terminal status for old response with high-water marks, dropped output count and uncertainty flags.

**No requirement to always replay the model on barge-in:** native duplex models may carry concurrent audio input state that cannot be reduced to text history. Use their source-verified state transformation or a safe full recompute; if no compatible replay exists, return typed refusal instead of corrupting state.

## 8. Reconnection and resume

Transport reconnect is not guaranteed session migration:
- Reconnect token binds authenticated tenant/session, model revision, output lane/sequence and expiry. Never reuse raw GPU handles or transmit private model state.
- A new connection negotiates codecs/capabilities and receives last *acknowledged committed* media items within bounded replay retention; it cannot request unlimited server-side audio history.
- Resumption must verify session epoch and state compatibility; if model session expired, offer a new session with a declared history import policy or refuse.
- Duplicated events are idempotent by (session, output_generation, lane, sequence, commit_group).
- Server cannot infer exact heard audio after a broken connection; preserve UnknownPlaybackProgress, not an invented complete acknowledgment.

Google Gemini Live publicly documents long-running session resumption, GoAway, compressed context and model-dependent speech interruption; these are API design *test cases*, not hidden engine dependencies. See research dossier.

## 9. Admission and QoS

~~~rust
enum QoSClass { RealtimeSoft, Interactive, Progressive, BatchMedia, Background }
struct WorkloadQoS {
    class: QoSClass,
    desired_first_output: Option<Duration>,
    max_inter_chunk_gap: Option<Duration>,
    max_end_to_end: Option<Duration>,
    weight: FairShareWeight,
    budget: TaskBudget,                          // RAM/VRAM, CPU, IO, output buffers
    graceful_miss: QoSMissPolicy,                 // refuse/degrade only when authorized
}
~~~

Admission must check **whole ModelAssembly** memory and resident components, codecs/encoders, session KV/prosody/latent scratch, driver allocation headroom, output rings, permitted peers and physical device interference. Do not promise hard real-time on general-purpose OS scheduling.

Two-level scheduling:
- Coarse: per-model/per-tenant fair queue + replica/placement choice at request/stage safe boundaries, reservation and deadline estimates.
- Fine: device-local streams/subpools and proved device-native step batches. Do not globally lock a model catalog/arbiter for every audio sample or token.
- Protected interactive windows: batching only compatible state/task/precision cohorts; reserve bounded resources for real-time audio so large video solves don't starve it.
- Stepwise interruption: diffusion model may yield at verified solver-step boundaries; not halfway through unknown physical GPU work.
- Queue capacity and first output latency are both bounded; codegen/tuning/cache loading must not block real-time hot path unexpectedly.
- If requested deadline cannot be met, refuse or choose an *explicitly authorized* alternative model/resolution/quality mode with ModeReceipt. NEVER silently lower precision/steps/voice/model to fake success.

## 10. Media-specific progress, metrics and SLOs

Text: TTFT, tokens/s, P95 decode, accepted-state correctness.  
ASR: real-time factor, word error rate (using source-aligned truth), time-to-first interim, endpoint-to-final latency, timestamp error, speaker diarization accuracy, revision count.  
TTS: first audible PCM latency, real-time factor for continued speech, chunk gap P50/P95/P99, packet underflow, speaker/style consistency, vocoder boundary artifacts.  
Live duplex: mouth-to-ear latency (with disclosed endpoint clocks), interruption detection-to-silence, delayed sample watermark divergence, inbound/outbound jitter and queue misses, full-session fairness.  
Image: full result latency, solver steps, source exactness, VRAM peak, qualitative reproducibility under pinned seed and setup.  
Video: first fully valid frame/segment, frames/s, causal window decode latency, source rate and A/V PTS skew, temporal consistency, peak VAE scratch.  
Decision: schema parse/admission and scored-choice latency, joint questions per batch, calibration error only if calibration exists.  
Hosted: actual provider response/request cost, job backlog, retries, upload/download egress and result validity.

Measure steady warm/cold start separately. Publish evidence hardware, driver, model variant, precision, input sequence, queue depth and percentile calculation. A research repo vendor claim is not a Zerogravity measurement.

## 11. Three fundamental delivery regimes

### 11.1 Local or one-shot request (sync)
- CLI/SDK in-process; no daemon required.
- File input/output with explicit resource source and permissions.
- API short jobs return full result if possible; not appropriate for multihour video.

### 11.2 Progressive streaming (HTTP/SSE or binary streaming)
- SSE for small structured committed events and text.
- PCM/Opus/audio/video bytes through binary frames, files or reference URLs; avoid base64 JSON for every 20–40ms live chunk on hot path.
- Transparent cap on event retention/queue length; server slow-client policy must not buffer unbounded device outputs.
- Transport backpressure propagates credits to task/session without forcing a full GPU lock across network send.

### 11.3 Bidirectional live (WebSocket/WebRTC)
- WebSocket: reliable control events and appropriate binary media framing. Client-managed playback ACK needed to truthfully truncate unplayed audio.
- WebRTC: preferred browser/mobile low-latency media path with negotiated codecs, RTCP feedback and separate reliable control channel; packet timestamps must map to model rational clocks.
- SIP: optional telephony gateway adapter; translate call lifecycle, DTMF, jitter, codec and hangup into the same session events. No direct neural model access.
- Native Rust callbacks: same canonical event contract without network serialization.

A user request for "gpt-realtime-2.5" is a **capability target**, not a verified publicly documented model ID as of this design date. OpenAI's public model catalog reviewed here lists GPT-Realtime-2.1 and GPT-Live variants; preserve model version labels only as live provider metadata. Do not hardcode proprietary internal architecture.

## 12. API matrix and versioning

| Endpoint family (illustrative) | Operation semantics | Must not fake |
|---|---|---|
| GET /v1/models | per-task/model/transport capability, exact artifact and revisions | inference support just from recognizable GGUF |
| POST /v1/chat/completions | source tokenizer/template, generation tools and streaming | fabricated tool call grammar or model substitution |
| POST /v1/completions | raw text continuation with bounded params | hidden chat templating |
| POST /v1/responses | only verified Responses-style event/tool schema | claiming full endpoint fidelity from text-only subset |
| POST /v1/embeddings | encoder pooling/normalization/dimensionality | auto using decoder logits |
| POST /v1/rerank | pairwise/cross-encoder scoring | text generative heuristic |
| POST /v1/audio/transcriptions | file/window ASR; language and timestamp policy | arbitrary partial ASR as final |
| POST /v1/audio/translations | actual source-target speech translation model | translate ASR via different model without disclosure |
| POST /v1/audio/speech | typed voice/speaker/sample-rate/format | partial invalid codec-frame response |
| WS /v1/realtime | strict implemented live session event subset | undocumented pseudo OpenAI fields |
| WebRTC realtime session negotiation | session/codec/token auth and ice/transport state | unbounded relayed media or leaked credentials |
| POST /v1/images/generations and edits | exact supported image model mode | imaginary unified image parameters |
| POST /v1/videos / jobs/{id} | async status/progress/cancel/output | long-running video blocking an HTTP request |
| POST /v1/decisions | typed schema question logits, choice and calibration | generated JSON masquerading as direct logits |
| POST /v1/tasks | native canonical TaskRequest with typed multimodal ports | executing arbitrary user workflow code |

**Compatibility profile** keys: protocol name/version; model identifier+revision; request field set and semantics; supported input/output modalities and codecs; streaming event types; limitation and refusal semantics; task/control algorithm; native/opaque service execution; quality evidence. Unknown request fields with semantic effect must fail or explicitly warn according to versioned compatibility policy; never silently discard safety-/quality-relevant fields.

## 13. Hosted opaque provider TaskServiceAdapter

External API providers: authorized hosted TTS/image/video generation services, proprietary decision services, hosted GPT-Realtime/Gemini Live and future models. A hosted provider exposes **capabilities and actual API observations**, not local SIR/TensorEncode semantics. No unverified backend region state or GPU memory leases. Provider model revisions can change; pin named snapshot if available, otherwise record alias/revision uncertainty.

~~~rust
trait TaskServiceAdapter {
    fn capabilities(&self) -> TrustedProviderCapabilityManifest;
    fn validate(&self, task: &TaskRequest, authz: &DataEgressPolicy) -> Result<ProviderRequest>;
    fn start(&self, req: ProviderRequest, key: IdempotencyKey) -> Future<ProviderJobHandle>;
    fn watch(&self, job: ProviderJobHandle, cursor: Option<ProviderCursor>)
        -> Stream<ProviderEvent>;
    fn cancel(&self, job: ProviderJobHandle) -> Future<CancelReceipt>;
    fn reconcile(&self, key: IdempotencyKey) -> Future<ProviderJobStatus>;
}
~~~

This is illustrative, not a fixed public trait. Provider job lifecycle: Created → Accepted → Queued → Running → Partial (only if provider guarantees validity) → Succeeded / Failed / Canceled / Unknown. A network timeout after submission does NOT imply job cancellation. Idempotency keys, cost bounds, upload retention and retries must avoid double billing.

Security:
- per-tenant secrets excluded from logs, model prompt and event payloads; no browser exposure of server API keys;
- explicit allowed external domains/region/data egress; signed URLs bound to scope and TTL; verify received asset MIME, size, integrity;
- upload voice samples and image/video with explicit consent/rights; avoid deceptive "on device" labels;
- content credentials and provider provenance recorded only when actually supplied, not invented;
- rate/cost limits and audit record without leaking private inputs.

## 14. Failure and availability rules

- **Transport disconnect:** stop/pause per declared session policy; physical CUDA work still held until Ready. Do not drop uncommitted StateRoot because socket died.
- **Client slow playback:** decrease producer credits, pause at legal frame/solver boundary or abort; no unlimited buffering.
- **Bad media:** reject excessive size, decompression ratio, corrupted timestamps, nonfinite samples, invalid color format, unsupported codec and impossible dimension before GPU launch.
- **Unsupported device:** portable source-verified candidate or typed refusal; no silently lowered precision.
- **One-session interruption:** cancel only its turn generation. Physical-device-wide epoch changes reserved for actual device-level failure/reinitialization.
- **Remote worker failure:** never trust remote "complete" without authenticated source/schema/event receipt; no locally forged StateRoot.
- **Hosted job 502/timeout:** job status Unknown and reconcile; don't retry an expensive video create unconditionally.
- **Observed API mismatch:** mark compatibility tier unsupported; explicitly reject that feature.

## 15. Media presentation receipts and truthful playback

A client "played up to sample N" event is weaker than physical playback proof; it may be absent/forged. Maintain:
- reported watermark and origin (client SDK, RTP feedback, player callback, inferred transport),
- delay bounds and uncertainty interval,
- output generation and codec clock,
- acknowledged vs reported played distinction,
- authenticated monotonic advancement; never decrease within one generation,
- action on client reconnect/late ACK (ignore stale generation).

In OpenAI Realtime WebSocket, documented conversation.item.truncate removes the unplayed audio part and corresponding transcript; in WebRTC/SIP output audio buffer handling is server-managed. These are adapter semantics, not universal model capabilities. Source: https://developers.openai.com/api/docs/guides/realtime-conversations (reviewed 2026-10-15).

## 16. Explicit compatibility prior art (not native implementation)

- OpenAI realtime: https://developers.openai.com/api/docs/guides/realtime-conversations
- OpenAI client event schema: https://developers.openai.com/api/reference/resources/realtime/client-events
- OpenAI public model catalog: https://developers.openai.com/api/docs/models
- Google Gemini Live: https://ai.google.dev/gemini-api/docs/live-api/capabilities
- Google session resumption, context and GoAway: https://ai.google.dev/gemini-api/docs/live-api/session-management
- Moshi (external reference removed during attribution cleanup)
- Hosted TTS live API and voice SDK (external reference removed during attribution cleanup)
- Speech toolkit prior art (external reference removed during attribution cleanup)
- Joint audio/video generation prior art (external reference removed during attribution cleanup).

All rate/latency/token values shown in other documents are per **upstream claim or model checkpoint**, not Zerogravity measured SLO or hard-coded defaults.

## 17. Acceptance drill extension

MM-13: duplex input while output; server and client VAD; turn cancellation stops audio without dropping input.  
MM-14: late playback ACK after new output generation is ignored.  
MM-15: 44.1k→16k/24k resampler and 30000/1001 video clock maintain exact timebase across prolonged stream.  
MM-16: no held GPU physical buffer released because network send resolved or timeout occurred.  
MM-17: slow WebSocket client causes bounded producer backpressure while other tenants remain live.  
MM-18: 5 video diffusion jobs plus 10 duplex streams share GPUs within declared safe quotas; fair admission without numerical downgrade.  
MM-19: websocket reconnect/resume with duplicate event ID returns idempotent committed history and consistent watermark.  
MM-20: hosted video-generation timeout after create remains Unknown and reconciles without double-submitting expensive job.  
MM-21: source-capability matrix rejects unsupported image edits/voice/decision fields and advertises actual backend.  
MM-22: no per-packet model routing or global resource snapshot allocation in critical live path.  
MM-23: source A/V joint generator cannot expose video without corresponding valid audio watermark when declared coupled.  
MM-24: malicious media, unauthorized egress and cross-tenant replay tokens rejected before heavy resource reservation.

**Before feature implementation:** design mocks MM-01..MM-24 with deterministic fault injection and explicit assertable counters. Then original source checkpoints and actual local device tests with observed latency. No raw model weights or untrusted live audio required for the mock phase.


## 18. Open policy decisions before live serving

The user requires broad native/authorized-hosted media and realtime capability,
not a particular proprietary implementation or an assumed numeric SLA. Resolve
these choices explicitly for the selected source/task/transport; type sketches
above do not settle policy or freeze an ABI.

| Open decision | Safe constraint until resolved |
|---|---|
| Media numerical/perceptual agreement | Preserve the selected source's declared arithmetic/quality contract; any tolerant mode requires separate approval and task-level tests. Existing exact Qwen gates do not relax. |
| First source/codec and actual streaming eligibility | Choose exact licensed feasible checkpoints/processors; distinguish source capability, native implementation and measured playable latency. Family labels and hosted responses do not qualify local execution. |
| Mouth-to-ear/first-audible/packet-gap targets | Measure under explicit device/load/transport/endpoint-clock conditions; no invented default P95, hard realtime or SSD-latency guarantee. Bound warmup and shape-band compilation outside the audio hot path. |
| Heard/presented watermark after interruption or reconnect | Retain origin, uncertainty and missing-feedback state. Authenticated ACK is still not proof of human hearing; never rewrite neural state from a transport callback. |
| Model, voice or codec changes mid-turn | Refuse unless an explicit new generation and source-compatible state transition/replay are supported. An old codec tail, callback or prior source cannot publish under the new generation. |
| Session persistence and media retention | Ephemeral bounded retention until duration/encryption/access/deletion policies are chosen. No silent recording or unlimited replay buffers. |
| Voice/face/reference-asset consent | Explicit rights and tenant-scoped authorization; separately validate G2P/codec/weight licenses and biometric-reference handling. No implicit cloning or remote egress. |
| Hosted auth, alias selection, cost and retry | Provider/profile/domain/region/egress must be explicit; a pathname or alias is not authorization. Cap upload/result lifetime and cost; reconcile ambiguous accepted jobs without blind replay. |
| WebRTC/SIP packaging and codec interoperability | Optional versioned transport adapters with reviewed licenses/auth, loopback-safe defaults and bounded media/control paths. Choose multiplexed versus separate tracks only with declared clock/epoch mappings. |
| Native component/plugin interface and cancellation | Start with static registered Rust components; no unreviewed dynamic code. Session/turn cancellation cannot advance a physical card epoch, release Unknown work or invent safe recovery. |

A future implementation must turn each applicable choice into an explicit plan
item and positive/negative test. The architecture permits either a direct native
duplex model or an explicitly composed speech pipeline; no mandatory cascade may
replace the selected model's simultaneous acoustic state. API version/model names
must be revalidated when implementing a provider; the dated examples here are
research references, not live catalog discovery.

````

## Source: .agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md

Original UTF-8 bytes: 16163; SHA256 `bd35928aa22cfd61e7e089c16a7f3899bfacb01dac9d73a5556edbfcd4f78715`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Remote peer transport, state handoff and failure protocol (v4.2)

**Normative design; 2026-10-12; no code written or network configured.** This is a **small peer/transfer capability**, not a separate distributed inference engine or Kubernetes-like control plane. Read with [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [minimal execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), and [universal backend contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md). Source/vendor mechanisms live separately in [network prior art](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md).

## 1. One process can run standalone or offer a peer role

**Standalone:** no cluster daemon required, no tunnel discovery, no network permissions and no remote state metadata. One TaskCoordinator/DeviceWorker/ResourceArbiter and direct local region kernel.

**Enrolled private peer:** the **same zerogravity library/binary** may run an authorized PeerAgent offering a **subset** of capabilities: advertise hardware/topology, cache verified model weights, execute an allowed closed region, own leased state shard buffers or immutable prefix cache, and transfer data. It is never a second model server or user request authority unless it is specifically the current TaskCoordinator for that user's session. PeerAgent's OS privileges are strictly less than a generic shell; it cannot read arbitrary local files or execute model-defined source.

**Suggested user-facing modes (illustrative, not frozen API):** standalone default; explicitly enroll trusted peer; list discovered authorized peer capabilities; show direct/relayed/overlay path and benchmark; inspect proposed placement; optionally lock/deny nodes and task/data classes; request disconnection/drain. Auto mode is only a **measured routing policy** within enrolled, authorized nodes, never network permission discovery.

## 2. Exact identities and separate epochs

- `ClusterMembershipEpoch`: authorized peer set and identity/certificate generation; dynamic discovery **does not** authorize a node.
- `TopologyEpoch`: observed device, memory, link, direct/relay RTT/bandwidth and fabric capability snapshot.
- `PlacementEpoch`: selected mapping of closed compute regions, parameters, state shards, caches and TransferPlan; frozen for current safe work group.
- `SessionEpoch`: authoritative TaskCoordinator writer generation, task root + last committed output seq.
- `StateShardGeneration`: remote memory allocation, version, owner node and physical lease; combines with logical StateRoot version and model/quant/sampler/state schema fingerprint.
- `TransferSequence`: per-operation reliable integrity-checked chunk stream; late/misordered/duplicate arrivals cannot become a new TaskSession root.

These epochs **serve distinct lifetimes**. A link reconnection may change TopologyEpoch without invalidating source model semantic identity, while moving a state owner changes PlacementEpoch/StateShardGeneration but not mathematical TaskFrontier. If topology changes during a compiled multi-rank collective, the current attempt **cannot silently recompute a new placement**; use safe abort/fence/replay boundary.

## 3. Wire control is small, bounded and authenticated

Illustrative commands (not wire ABI):
```text
Hello(ProtocolVersion, NodeIdentity, CapabilityNonce)
Authorize(PeerPolicy, TenantScope, ModelArtifactScope)
Capabilities(NodeSnapshot, LinkMeasurements, DriverAndMemory)
Reserve(ResourceTicket, MemoryFootprint, Deadline, ModelOrStage)
EnsureArtifact(ContentHash, LicensesAndPermissions)
PrepareProgram(SemanticRegionHash, ExecBundleHash, Shape, NumericMode)
PrepareTransfer(SourceResourceLease, DestReservation, TransferDescriptor)
ExecuteRegion(SessionEpoch, PlacementEpoch, InputBatchRef, StateShardRefs)
ReadyOrFailed(OperationId, CompletedEvents, PreparedStateShardVersions)
CommitNotice(SessionEpoch, CommittedStateRootHash, OutputSeq)
AbortOrFence(SessionEpoch, OperationId, Cause)
RenewStateLease/Release/Drain/Health
```

**All frames** have schema version, checked lengths, source/destination identity, per-session cryptographic-scoped operation authorization, deadline/idempotency token, integrity and replay protections. Unknown commands or optional capabilities explicitly rejected, not interpreted as shell or unsafe model extensions. **No raw GPU pointers, RDMA keys, public network identifiers, session secrets or tokenizer contents** are logged casually.

The control service is authenticated even if traversing encrypted WireGuard/private overlay. Tunnel is connectivity; an adversarial/compromised peer remains a data-access risk. Send only user-approved tensor/weight/hidden-state/activation classes. A new peer cannot steal work from an active session merely by advertising a faster GPU.

## 4. Bulk transfer and transport capabilities

`TransportProvider` is one contained implementation seam, selected by measured link rather than vendor flag:

| Logical mode | Examples | Transport capability contract |
|---|---|---|
| On-host | CPU ↔ GPU, UDS/shared-memory, mapped approved pages | memory lifetime, IOMMU/driver fences, buffer version |
| Trusted local peer | direct Ethernet/Thunderbolt, PCIe/NVLink if physically exposed | peer bandwidth/RTT, memory access/collectives, encryption policy |
| Fast registered remote | verified IB/RoCE/GPUDirect or compatible RDMA | registered segments, revocable memory keys, transport completion/lease quiescence, physically supported fabric |
| General remote | authenticated TCP/TLS or QUIC reliable streams | bounded buffers, multiplex/priority, congestion/RTT/loss and copy/crypto CPU budget |
| Private overlay tunnel | direct/peer-relayed/DERP-type overlay with WireGuard | identical application identity and reliable stream semantics; **overlay not RDMA** |
| Storage transfer | file/SSD/object through authorized IO provider | content hash, ranges, explicit destination physical memory and retry policy |

The **control plane may travel over a private tunnel while bulk data chooses a separate trusted direct fabric** where permissions and physical topology allow. On a WAN overlay relay, the solver may choose only occasional large state transfers or contiguous few-layer stage cuts. Other cases can fall back to local/offline queueing or reject the distributed plan.

Data transfers should carry only the **minimal required** payload. Tensor compression/packing must preserve selected numeric mode and source math. Explicit lossy activation compression creates a new candidate requiring model-level quality tests; it must not be silently enabled to hide a slow tunnel.

**Control/data credits:** reserve memory on receiver *before* sender prepares large payload, negotiate chunk size and bounded in-flight credits, check checksum/hash, release source and destination leases only after actual network/device completion, not after TCP write() returns. Prefer queue priorities for token-critical activations and state before background weights/prefix warming; prevent starvation via quotas. A peer may **decline** transfer if source/tenant/license authorization insufficient.

## 5. Remote model/layer execution flows

### 5.1 Layers and mutable cache colocated on peer B

Node A owns task cursor/sampler/root; B owns model layers [k..m] and those layers' persistent KV/recurrent shard buffers, warmed model tensors and compiled executable. For each token or prefill chunk:
1. A sends activation tensor and stable session/shape/position/StateRoot lease version.
2. B validates and executes its closed layer region on B, updating a **provisional state version**, returning next activation/result and Ready event.
3. Remaining layer groups run as prescribed by the same ExecutableBundle/PlacementEpoch; no B-side independent sampling or task service.
4. A verifies required group readiness and **publishes one logical root**, with B shard handles in that root; B retains committed version under lease and retires old state after actual fences.
5. On failure, no partial public state; replay/reset B from prior canonical shard checkpoint or choose other legal placement.

This retains **weights and KV local to the layer** and only incurs activation transfers at layer-group boundaries. Suitable first multi-machine capacity mode, but serial decode latency scales with number of cross-machine hops.

### 5.2 Layer compute on A, attention/KV state on B

**Alternative A — remote attention:** A sends normalized query/positions/attention mask and required output projection metadata to B. B executes the actual attention **beside KV**, returns a small attention result and PreparedStateShard. A performs dense projections/FFN. Requires exact op/math/quant state and per-layer cross-network latency can dominate.

**Alternative B — state hydrate:** A fetches authorized immutable/committed KV+SSM snapshot once from B, then executes attention locally under new state generation/placement epoch. Transfer bytes large at start but amortized across many future steps. Never use missing GDN/indexer state as a “complete KV” hit.

**Alternative C — direct remote state reads:** only where verified coherent/registered remote fabric can supply enough bandwidth and correct consistency. Relayed tunnel/ordinary WAN should normally reject this mode by cost, not fallback into accidental 10–100 GB KV reads per request.

### 5.3 Compute/FFN/experts on B and stateful attention on A

Attention/state on A where memory bandwidth and cache capacity are favorable; large MoE/FFN model parameters on B. Each affected layer has a bounded activation transfer and result return, with optional GPU-side expert IDs/route compactness. FFN worker doesn't own TaskCoordinator/RNG or KV. Sparse and hybrid models may need different split per layer; compiler proves exact operator boundaries. On WAN high RTT this is usually a *capacity/throughput at larger batches* strategy, not automatically best C1 latency.

### 5.4 Prefill on A, decode on B

Two-phase reservation:
1. prefill A computes candidate full hybrid state, B admission reserves **all destination mandatory state + graph scratch** or explicitly waits until prefill complete with risk/cost recorded;
2. transfer authenticated chunks of KV/recurrent/indexer/ngram/vision/task state, with checksum/schema and event completion; B declares PreparedStateBridge;
3. coordinator publishes new placement epoch and full root including B owners at a legal task frontier; decode begins from that exact prefix, not a text-only KV assumption;
4. partial transfer/cancel/failed B admission returns to old worker or deterministic replay from canonical checkpoint, with no duplicate published tokens.

The P/D split is a candidate for TTFT/ITL isolation, not a guarantee to increase throughput. On a slow tunnel, multi-GB cache migration may make it impractical.

### 5.5 Remote prefix cache and sequence KV shards

Remote **immutable prefix cache** is not a live state owner. It can provide verified content-addressed compatible snapshots with asynchronous partial download and recompute suffix; its cache presence directory can be stale. Publishing live StateRoot remains separate.

For sequence-sharded attention, B/C/D each hold disjoint token-range KV and compute local softmax maxima/denominators/numerators; A combines correctly scaled partials under the exact supported attention op. **Never** try this for unknown sparse/compressed/recurrent operator algebra; one shard may contain causal positions absent under another client's request. Model quality/numerical and attention mask tests are mandatory. A slow tunnel with per-layer reduction may still lose; prefer bounded local/phase plans.

## 6. Recovering from node failures and partitions

**Per-session leader, not universal consensus system:** TaskCoordinator holds one generation-scoped logical writer lease. Peer B cannot become a new task leader because its node heartbeats more recently; promotion/migration requires explicit safe leader fencing or a future quorum/leased metadata store. The initial private 2-node mode may **fail-stop and replay** rather than promise HA.

**Important ambiguous case:** A receives B's Prepared candidate and publishes a new StateRoot, then A crashes before B sees CommitNotice. B must keep its prepared candidate under a bounded lease/retain policy and respond idempotently to a later query for same root/epoch; if no durable root exists, safe recovery can use the previous committed checkpoint with replay, **not pretend user output was never sent or exactly-once delivered**.

**Network timeout does not cancel DMA:** remote device or RNIC may still be writing; physically fenced or quarantined buffers cannot be recycled until actual completion/reset. A stale transfer may not overwrite a new session's reused slot because `AllocationId+Generation+Epoch` checked on receiver.

**Network partition:** candidate state prepared on orphaned rank cannot be committed as a new root, and an old writer with expired epoch must not issue new writes. Control plane leader election/durable consensus is **deferred**, not silently provided by authenticated streams or private overlay connectivity. If a session cannot regain compatible state, return typed recovery failure and retain last known output cursor.

**Peer code trust:** do not execute arbitrary unverified kernels or remote shell commands to repair, “optimize” or join a cluster. Remote program and model artifacts allowed only under signed model/region/source policy.

## 7. Fault-injection and QoS gates

- invalid membership/tenant authorization, wrong checkpoint/LoRA/semantic/numeric/TaskDriver output, driver/backend ABI mismatch;
- direct → peer-relayed → DERP-like path transitions mid task, sudden bandwidth drop and high P95 jitter; verify solver degrades plan only at safe frontier;
- cache directory claims remote prefix but payload disappeared, mismatch or partial hash; recompute/not wrong prefix reuse;
- remote layer B produces correct activation but wrong GDN state version; reject publication;
- full KV transfer on tunnel would exceed deadline; planner must decline in favor of colocated/recompute/remote attention when legal;
- remote attention partial reducer with all-masked shard, lse overflow, global top-k and dense/GQA/rope mask; exact math and tolerance;
- duplicate CommitNotice, lost ACK, coordinator crash after publish, remote worker restart with stale lease;
- actual model+quant on 2 local machines with direct authenticated link, then overlay direct, then relayed path; distinguish capacity enabling from raw t/s improvement;
- 1/2/4/8+ request batches, 4K/32K/128K context, native decode/verify/preload, TTFT/P95/energy/egress/crypto and total task correctness;
- node drain/rejoin during prefilling/media stream, user cancellation, no leaked buffers or unacknowledged output.

**No proof of system durability or performance by simulation alone.** Actual transport and device tests follow user authorization; one model CPU/GPU vertical slice remains earlier priority.

## 8. Minimal implementation discipline

Use a few Rust modules within the existing engine: peer identity/capability, topology snapshot, placement solver, transport adapter, remote work envelope/state shard leases. These reuse existing DeviceBackend, ExecutableBundle/CIR, ResourceArbiter, TaskCoordinator and TaskDriver; **no Kubernetes, etcd, database, DHT, cluster-wide multi-leader election or new serving engine required initially**. Future scale may warrant richer control-plane services based on measured use cases, not as a mandatory project prerequisite.

**Prove a two-node capacity-first layer split before speculative distributed attention or WAN per-token FFN.** Explicit config/trust and low-risk direct encrypted transport first; retain standalone fast path. Higher-level features (distributed prefix directory, attention/FFN disaggregation, sequence KV shards, autoscaling, remote multi-task scheduling) remain independently qualified plan candidates.

**Status:** all implementation and transport gates are PLANNED, none run.

````

## Source: .agents/docs/RUNTIME_SERVING_AND_UX.md

Original UTF-8 bytes: 18788; SHA256 `dc295903499b72feb8914161adb9ade6c9e3c111d29c63bc24bc3cc62a76a7b9`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Runtime, APIs, developer experience and observability

**15 Oct 2026 — efficient serving UX and truthful metrics:** [Joint concurrency specification](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) requires three explicit capacity valves (API requests, queued prompt work/model/session budgets, active device step rows/scratch) with typed overload and no unbounded queue growth. One request binds exact model/quant/profile once; GPU-ready and root-committed outputs stream without per-token router lookup or hidden Python model-specific server. Report queue/admission/rejection fractions, committed tokens, TTFT/ITL/p95, cached state domains and actual saved recompute, draft accepted-vs-proposed, MoE expert/memory IO, energy and physical GPU completion. Claims of vLLM superiority require matching numeric/model/quant/hardware and loaded source; source-free demos do not qualify.


**15 October 2026 — NATURAL STORAGE MODE AND TRUTHFUL UX:** [Storage-neutral tensor residency](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) should be automatic: prefer full GPU/UMA residency when feasible and source/backend-qualified SSD expert/lookup streaming when a model cannot fit, without a second model/server/API alias. Report source identity, effective cache and mandatory memory, OS headroom, SSD hot/cold miss/copy budget, current residency ratio, source-compatible state snapshots and actual measured latency/QoS. Advanced controls only for cache bytes, IO policy, prefetch, user-selected capacity priority; no hidden quant, default SIGKILL or global page cache drop. A model may load from SSD but be too slow for live audio; state latency truthfully and refuse soft-realtime deadlines where appropriate. [SSD test matrix](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md).


**15 Oct extreme model serving precision addendum:** [extreme-specialization evidence/implementation source atlas](../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md) and [full specialization tests](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) reinforce one native Rust ModelRouter/TaskCoordinator/StateRoot, serving exact model/quant/sidecar and source tokenizer/think/tool template. Expose supported source mode, original-vs-derived quality, speculative accepted-only tokens, TTFT/latency/cold-start, model loading progress and physical OS headroom without claiming unknown success. A multi-session batch can alter near-tie greedy output through reduction-order changes: numeric capability must say so. Avoid dynamic solver search, graph compilation or global model catalog lookup on packet/token hot path; no new GLM/Qwen/MiMo Python proxy per family.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


The user expects llama.cpp-like simplicity despite a complex compiler and serving runtime. **Do not expose model kernel details as mandatory user flags.** Every public capability is either functional and tested or clearly experimental/unsupported.

## 1. Proposed executable and modes (illustrative, not finalized)

```sh
zerogravity run /models/model.gguf
zerogravity run hf:org/model --offline
zerogravity serve /models/model.gguf --host 127.0.0.1 --port 8080
zerogravity inspect /models/model.gguf --explain-plan
zerogravity bench /models/model.gguf --scenario single-ar
zerogravity devices
zerogravity models list
```

Auto plan selects backend/device, quant/layout, safe memory reservation, default context and supported token budget from model/hardware capabilities. CLI should print approximate model weight bytes, resident/SSD table bytes, context/recurrence state, scratch headroom and expected fallback reasons. Do not silently choose lower quant when requested weight file won't fit.

## 2. Rust library/SDK layer

Engine shared by CLI, HTTP and embedding in other Rust applications. Suggested API objects: `ModelRegistry`, `ModelHandle`, `Engine`, `Session`, `Prompt`, `GenerationOptions`, `TokenStream`, `ExecutionReport`, `BenchmarkRunner`. Engine owns devices, memory, plan registry and async executor. Sessions are mutable timelines; model handles immutable weights. Typestate or guarded APIs prevent using session after close or stale snapshot. No caller gets a raw device pointer by default.

## 3. HTTP compatibility matrix

Initial proposed endpoints:
- `GET /v1/models`: loaded model IDs, true capability and context.
- `POST /v1/chat/completions`: model-native template, tools, streaming.
- `POST /v1/completions`: raw prefix/text completion.
- `POST /v1/responses`: Responses-style continuation and tool history, if conformance complete.
- `POST /v1/messages`: Anthropic-style messages, if conformance complete.
- health/ready/metrics endpoints with policy-based information exposure.

OpenAI/Anthropic compatibility does not imply one text renderer for every model. Model adapter handles native tools, role sentinels and hidden reasoning. Traces must distinguish visible output from model-internal reasoning and prevent accidental disclosure of hidden state through APIs. Structured output grammar/JSON mode must not silently change sampling if unsupported.

Authentication, TLS and rate limits need safe defaults; local bind `127.0.0.1`, remote `0.0.0.0` explicit with warning/auth requirement. Requests cannot pass arbitrary local file path or remote image URL without explicit access policy and bounded fetch.

## 4. Streaming correctness

Stream only **committed** model output, not speculative draft proposals. Cancellation/stop must reconcile state frontier and output budget. Slow client streams have bounded queue and backpressure; never allow unlimited GPU buffer accumulation. Stop sequences can span token boundaries, so hold back appropriate text until detection; do not retract already streamed bytes. UTF-8 output can cross token boundaries, so incremental decode must handle incomplete codepoints.

Include deterministic request ID, position, terminal reason (eos/stop/length/cancel/failure), usage counters distinguishing prompt, cached and generated output, and speculative acceptance/latency in diagnostics (not necessarily every API response).

## 5. Tool and agent histories

Tool call delimiters and argument encoding must come from model-native template; use an intermediate structured message representation with opaque rendered token history. Reusing cached prefix based on only visible text can fail if previous hidden reasoning or tool IDs have different exact rendered tokens. Parse tool calls incrementally and validate escaping, invalid nesting, very large arrays/maps and unclosed blocks.

Agents may send same long system/tools repository context each turn; full hybrid prefix snapshots should avoid re-prefill where compatible. Preserve structured roles and consistent parent/branch session IDs. Optional persistent cache contains private conversations, so explicit user configuration and safe permissions are required.

## 6. Context and memory fit UI

`inspect --explain-plan` should show:
```text
artifact SHA / tokenizer / family / quant
device and backend, driver, architecture target
current memory: physical available, GPU addressable, OS reserve
weights: dense, shared, expert resident, expert backing storage
context: KV pages, GDN/KDA/indexer, snapshots, scratch per session
selected prefill/decode/verify kernel regions and numeric contract
speculative mode: none/MTP/other, exact sampled or approx opt-in
fallbacks and why (missing kernel, shape range, precision, memory)
maximum safe context and concurrency (estimated, not guaranteed)
```

On insufficient memory, explain specific bytes short, safe options (smaller context/batch/model/quant, SSD streaming if actually supported), and never automatically mutate system boot parameters or require root.

## 7. Multi-model and hot reload

A single engine can enumerate models and load one or several only if budgeted. Hot reload creates new immutable ModelHandle/plan and leaves old sessions bound to old model identity until retired or explicitly migrated by replay. Sessions cannot transparently reuse incompatible weights or tokenizer. Device graph/model unload waits on outstanding leases and events.

## 8. Metrics and telemetry

Core counters:
- queue, TTFT, prefill and decode duration, per-token latency P50/P95/P99;
- token counts by input/cached/output/draft/accepted/rejected;
- model/plan/backend/quant and context frontier; can be redacted or aggregated;
- expert cache hits/misses, unique routed experts and IO critical path;
- resident/pinned/virtual physical allocations and OOM/admission rejects;
- GPU execution vs idle, graph capture/launch count, kernel breakdown in debug;
- prefix reused tokens, COW snapshot size, persistent cache costs;
- errors/cancellations/restarts and distributed sync status.

Prompts, model personal data and tool outputs not sent to telemetry by default. Error messages must avoid dumping secrets and credentials. Profiling instrumentation changes performance; benchmark uninstrumented also.

## 9. Configuration hierarchy

Priority proposed: explicit request options > CLI flags > validated config file > environment for developer diagnostics > safe hardware/device defaults. Do not let undocumented environment flags change sampling/quality silently. Show effective configuration and explain any ignored option. Use versioned config schema with migrations; secrets handled separately from ordinary config and excluded from logs.

Power/perf modes: `interactive`, `throughput`, `balanced`, `memory-safe`, `quality-reference` (working names). Modes choose scheduling/resource strategy, not silent model quantization.

## 10. Local deployment and packaging

Portable CPU build and conditional GPU features, no Python required at serving runtime. Rust binaries and optional precompiled architecture-specific kernels cached with checksums; offline installation and deterministic reproduction. macOS Metal, Windows and NixOS packaging remain open priority; include artifact availability matrix and clear unsupported hardware errors.

NixOS considerations: GPU driver dynamic libraries/ld paths, system `/dev/kfd`, render group, ROCm version, glibc/LLVM linking. No assumption that conventional Ubuntu install instructions work on NixOS. A dev shell may supply build tools, but system drivers must remain host controlled.

## 11. Testing API usability

Run official API SDK smoke tests, malformed JSON/tokenizer error recovery, concurrency, very long tool histories, hidden/visible reasoning, image/text order and fingerprint, Unicode incremental output, network timeout, cancellation, server restart, old/new model hot reload, user-local bind default, auth and metrics. Agent integration needs real Pi/OpenCode/Codex or compatible request replay, but no embedded agent tool should access filesystem/network implicitly.

## 12. Pending policy

Executable flags/naming, stable Rust SDK/ABI, external authentication solution, default disk cache and retention, third-party plugin model schemas, logging redaction, default model selection and release channels remain open. Avoid treating CLI examples as approved final UX.

## 2026-10-10 integrated v2 addendum — concrete product and event contracts

Adopt research's precise serving backbone from `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`: frontends translate OpenAI/Anthropic/native requests into one canonical `InferRequest`; `EngineCore` is the single mutable owner; `EngineMsg/EngineEvent` and worker `StepPlan/WorkerDelta` channels provide asynchronous execution. Workers own device code/streams and only emit provisional outputs; stream clients see **committed** tokens after exact state/sampler checks. A `ModeReceipt` can be returned in diagnostics/metrics so true kernel tier, eager/capture, quant/graph, spec sampling fidelity and fallback are visible.

Research convenience features are part of the **product plan**: GGUF tokenizer pretypes, Jinja templates, a grammar library/JSON-schema grammar FSM, streamed family-specific tool parsers, content-addressed ModelStore with atomic pull/list/rm/GC/offline, model conversion/quantization/imatrix/optional acceleration pack, `/props`/doctor, and companion `run`/`serve`/Rust SDK. These are distinct from executing tools or MCP access: tool invocation requires a separate explicit authorization/sandbox system and is not promised by token generation.

The external API must distinguish task runners: text generation with KV/sampling; embeddings with encoder+PoolingSpec/Matryoshka normalization; rerank/score classification over paired inputs; multimodal encoder/projector with separate cost and span handling. LoRA adapters include request-specific generation in cache keys; unsupported methods like `best_of`/multi-sample are typed refusals until tested. `Responses`, `Anthropic` and tool schema fidelity must be implemented by conformance, not endpoint names alone.

**Mode/profile naming and version remain OPEN**, despite historical research selecting `zg`. Keep simple user-facing automatic plans with `inspect --explain-plan` and refusal naming one actionable adjustment. No startup user needs to know attention tile/WMMAs; no silent new quant or target sampler change. Do not accidentally expose hidden reasoning or private snapshot bytes through status endpoints or telemetry.

Integrate `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md` for model store/tokenizers/grammar/LoRA/embeddings and `ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md` for ownership and Rust-source policy.


## 2026-10-12 integration — NPU diagnostics and future media outputs

Device inspection should expose XDNA/NPU architecture, XRT presence, kernel/program compatibility, shared physical memory, and safety qualification state (unsupported, unqualified, experimental opt-in, qualified). **No automatic privileged change** to FCLK, power governor, runtime suspend, IOMMU or systemd. GPU-only works without XRT. ModeReceipt names GPU/NPU projection split, compute-encoding precision and fallback reason.

TaskProgram supports planned distinct client outputs for ASR transcript/audio timestamp, TTS PCM/audio frames, images/image editing, video frames and embedding/rerank scores, all under the existing request authorization and DeviceWorker/resource interfaces. An upstream Gufo model pipeline is research inspiration, not an already supported endpoint in zerogravity. Read [PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md](PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md).

## v4 refinement — truthful task streams and no hidden task workflow executor

One canonical TaskRequest maps protocol and SDK inputs to a built-in TaskDriver; it runs under the same coordinator/device/resource execution spine. Media and score outputs are typed independently of token streams. Session state is **published once after all domains Prepare**; only then may result/stream become visible. Live transport guarantees ordered **delivery attempts** with output_seq, not exactly-once remote receipt; optional authenticated cursor replay with retention/backpressure is a separate product feature. A slow client cannot indefinitely pin GPU KV or unbounded PCM/video buffers.

A TaskDriver cannot execute arbitrary filesystem/network/tool actions from model outputs; such external actions belong to an application-level authorized/sandboxed service. `inspect` and ModeReceipt should report whether a GPU-native step loop, host-controlled loop, compiled-region fast path or shape/precision/power safety fallback actually ran, without leaking prompt/media/private state. The mere presence of GPU/NPU/encoder data does not mean it is supported or performance-qualified.

## v4.3 authoritative multi-model service and zero-config UX refinement

**The current authoritative router/UX contracts** are [multi-model residency and routing](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [one-command CLI/config/OpenAI API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). Earlier `serve /model.gguf`, `config precedence`, standalone service and `OpenAI compatible` sketches in this document are historical supporting notes where they differ from v4.3.

**User-visible default:** `zerogravity run <model>` starts local interactive/single-request model processing directly in the same Rust engine **without a mandatory service**; `zerogravity serve` runs one loopback model-aware router+API over that same engine. Models list/pull/ps/inspect, per-model typed profiles and an effective-config explanation are progressively discoverable. A friendly alias resolves exactly to model bytes/quant/adapter/template/profile revision; a no-resident local model can on-demand load subject to real ResourceArbiter admission. No surprise registry download, public bind, model/quant substitute, remote peer, daemon or system setting changes.

**Multiple models/replicas:** catalog vs resident deployment vs active TaskSession different ownership; one model sharded across peers is one deployment. Preserve loaded state and sticky KV reuse where compatible. Replica load balance and model-choice `auto` policy are distinct, the latter explicit only. Model keepwarm/pin/idle unload never frees live DMA/unsnapshotable session; source model/alias hot reload affects only future bindings. Router isn't a token scheduler or another HTTP inference proxy.

**API:** qualified subset of `GET /v1/models`, `POST /v1/chat/completions`, embeddings with real encoder, later completions and versioned stateless Responses; typed SSE events, usage, roles/tools data and errors verified with official SDKs. Responses does NOT automatically provide cloud web search/file search/store or tool execution. Native model/admin control separate and restricted; local loopback by default, external bind requires explicit authenticated/TLS policy. Versioned user config with global/model/alias/CLI/request scopes, hard security/numeric/memory overrides and `config show --effective`; sample flags and port still proposed, not finalized.

````

## Source: .agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md

Original UTF-8 bytes: 25931; SHA256 `e6944af1831a086f17eba47be27339c6a69d39b1135003ff6b38e76b02931fa8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Rust workspace, interfaces, implementation practices and contributor workflow

**Current layout and model contracts ·16 October2026:** [H25](../plans/accepted-baseline/TODO.md) is accepted and committed as `8c39aa8`. The workspace now also contains the reviewed ninth Gemma4 contract library and Qwen3.8 source metadata APIs; their [separate main integration gates](../plans/model-execution/TODO.md) pass. These describe original configuration/tensor semantics and do not enable trained new-model inference. The ownership map below supersedes historical bootstrap sketches. One mathematical schedule, one source-identity issuer, independent session roots, sole physical admission and actual completion/quarantine remain mandatory.

| Workspace package | Responsibility |
|---|---|
| `zerogravity-core` | Shared typed source/program/state contracts, resource admission and device event/queue authority. Default std-only; optional serialization is explicit. |
| `zerogravity-artifacts` | Immutable source storage, checked GGUF parsing/codecs, validated Qwen source geometry/tensor roles and bounded duplicate-rejecting safetensors shard-index metadata. |
| `zerogravity-cpu` | Existing CPU numerical implementation borrowing source weights. |
| `zerogravity-model-qwen35` | Qwen3.5 schedule/state/tokenizer/checkpoint adapter; CPU and text are optional. |
| `zerogravity-model-gemma4` | CPU-free validated E2B and26B-A4B source configuration, tensor/KV/PLE/MoE contracts; no trained numerical execution route. |
| `zerogravity-gpu` | Generic resident GPU execution and physical lifetime ownership; independent Vulkan/CUDA/HIP features. Diagnostic qualification is explicit. |
| `zerogravity` | Small public facade and compatibility reexports; API examples and integration tests. |
| `zerogravity-cli` | The `zerogravity` executable, explicit CPU/GPU commands and text rendering. |
| `zerogravity-gpu-qualify` | Explicit diagnostic tool in `tools/gpu-qualify`, calling the same GPU library. |

The virtual workspace has one lockfile and version authority. Its default members are the facade and CLI with CPU/model/text defaults. The four narrowly patched upstream CubeCL packages are excluded workspace members; production GPU dependencies exclude CPU/family/text and LLVM code generation. Curated `all` selects implemented production components; `all-models` includes both family contract libraries and does not imply executable support for every described checkpoint. Diagnostics and abort probes require explicit features. A downstream path consumer must declare the four patch entries in its own workspace because Cargo patch tables are not transitive. Root release settings also apply to the promoted GPU products; H25 qualifies their frozen actual release binaries across the five healthy backend profiles. New source/feature changes retain separately scoped gates.

The layout gate removes the former `probes/` directory completely: production
GPU code and its notices belong under `crates/zerogravity-gpu`, independent
llama.cpp clients under `tests/reference/llama_cpp`, and the fixture extractor
under `tools/gpu-qualify`. Historical paths in dated reports remain historical.
Record retirement and any remaining work in the relevant plan TODO; obsolete
compiled objects and executable bodies are disposable build products. Audit sidetree ancestry, unique work and live users before
removing a merged or superseded worktree.

**Current work:** Follow [the plan TODO index](../plans/TODO.md) and
[AGENTS.md](../../AGENTS.md). Layout and contention-first concurrency review are
complete; no left-right migration was adopted. A1 cache is complete; A2/A3
acquisition precedes further GPU optimization. Model basics retain their
separately authorized CPU-first scope.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


## 1. Language and architecture constraint

The user has explicitly requested **entirely Rust** for the engine. Rust is the authoritative implementation language for model and serving logic and proposed owned GPU kernels. This is not a Rust wrapper around any external inference engine. Calls to GPU drivers, OS APIs and possible vendor math libraries via Rust FFI are a separate open decision (O-003); avoid claiming that writing Rust GPU kernel code eliminates ROCm/CUDA/Metal binary toolchains.

**Source language is not a speed feature**: architecture-specific instruction selection, memory traffic, graph capture and launch scheduling must be benchmarked.

## 2. Historical proposed Cargo ownership map

**v4 minimum:** start with **one library crate + one CLI binary** (or 2–4 crates if incompatible GPU compiler/source/toolchain restrictions justify). The older multi-crate tree below is a **possible eventual separation**, not tasks to scaffold immediately. Extra traits only when distinct provider code or architectural tests demand them; one Rust build must work CPU-only without GPU/NPU compilers.

```
zerogravity/
  Cargo.toml
  Cargo.lock
  rust-toolchain.toml
  flake.nix                     # candidate for reproducible NixOS setup
  README.md
  .agents/docs/ .agents/plans/
  crates/
    zg-model/                   # artifact parsing, adapters, tokenizer/template
    zg-ir/                      # semantic SSA / state effects, verifier
    zg-compiler/                # EIR, legalization, layout, kernel planning
    zg-kernels/                 # Rust CPU and GPU kernel source/registry
    zg-device/                  # GPU/CPU backends, streams/events/allocations
    zg-memory/                  # global arbiter, expert cache, IO, prefix storage
    zg-runtime/                 # engine/session/scheduler/drafter/commit
    zg-service/                 # APIs, CLI/lib, streaming, metrics
  tests/
    reference/
    models/
    integration/
    adversarial/
  bench/
    configs/ scenarios/ reports/ results/
  examples/
  tools/
    inspect/
    model-convert/
    quality/
```

This is a **proposed tree** and was not created in this planning step. If maintainability improves with fewer crates, merge crate boundaries and use Rust modules. Keep one-way dependencies:
`model + ir <- compiler <- runtime/service`; `device` and `memory` implement interfaces with no frontend dependency; kernels may depend on generated IR schema but never HTTP. Avoid circular crate deps or dozens of tiny traits in the hot path.

## 3. Ownership and concurrency rules

- `ModelArtifact` immutable and hash identified; optional mmap/SSD backing.
- `ModelExecutable` immutable compiled plan, owning kernel caches, tensor layouts and read-only weight residency handles.
- `Session` exclusive mutable owner of state frontier and one or more snapshots; share immutable prefix by typed COW.
- `ResourceArbiter` the sole physical allocation/admission authority. All subsystem caches request leases, never silently allocate giant buffers.
- `DeviceBackend` owns stream/event and kernel lifetimes; asynchronous completion tracked across worker threads.
- `Scheduler` may build multi-session batches but cannot expose mutable session state to unrelated requests.
- `Frontend` owns roles/messages/templates/HTTP streaming only; no raw device memory or direct model-specific kernel invocation.

Cross-thread messaging uses bounded channels; avoid blocking async executor with `hipDeviceSynchronize` inside request critical path. Rust `Send`/`Sync` impl on FFI GPU objects requires proof of driver/device-affinity safety; no blanket unsafe impl.

## 4. Public data contracts

`Result<T, ZgError>` or equivalent rich typed error hierarchy from INVARIANTS. Strong opaque IDs and generations: `ModelId`, `TensorId`, `SessionId`, `StateVersion`, `BufferLeaseId`, `DeviceEventId`, `PlanKey`. Avoid raw integer IDs crossing protocol layers without validation. Persisted snapshots have independent versioned schema, not Rust `repr(C)` memory dumps.

Model and kernel registries advertise capabilities rather than guessing from names. Registry operations support `can_lower(op, device, numeric_contract)` and return precise unsupported reasons. Dedicated reference implementation for all new semantic ops.

## 5. Core hot-path performance rules

- Compile/resolve graph regions and weight pointer layouts at model load; no per-token string lookup.
- Avoid boxed dyn trait dispatch inside GPU loops; a few calls at step region boundaries acceptable if benchmark cost negligible.
- Reuse scratch buffers; no per-expert allocation/free churn.
- CPU control to GPU command streams asynchronously; event batching and graph replay where supported.
- Pack kernel args into small verified struct; state pointer revisions/device table values may update safely without CPU graph rebuild.
- Prefill/verify/decode plan variants chosen by scheduler from bounded set.
- Hardware-specific Rust kernel source may use target intrinsics via backend extensions, with portability fallback.
- No global mutex covering all sessions during long GPU operation; separate session mutation lock, model immutable read state and cache lease operations.

## 6. Engineering ergonomics

Prefer explicit, small modules and easy-to-review changes. Every hot kernel includes a comment explaining shape constraints, numeric rounding and expected memory layout; every stateful op documents snapshot/rollback and causal position. Avoid generic framework architecture when concrete operator definitions suffice. Keep mutation APIs narrow; assert invariants in debug and return robust errors in release.

Use `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings` appropriately (GPU feature targets may differ), `cargo test`, Rustdoc, no undocumented `unsafe` blocks, sanitizer/Miri as support allows. Use `deny(unsafe_op_in_unsafe_fn)` and other idiomatic lints; FFI memory rules individually audited.

## 7. Unsafe and external code

GPU device pointers, memory registration, mapping, inline ISA and Rust compiler kernels may require `unsafe`. Concentrate unsafe code behind well-tested wrappers:
- provenance/size/alignment validated;
- device and stream identity attached;
- event-backed lifetime proves no use-after-free;
- no unchecked transmute to quantized packed layout;
- no dereference of stale GPU/host-mapped ring pointers;
- speculative cache snapshot stays immutable under child mutations;
- no unbounded allocation from untrusted model fields.

Licenses/attribution for any copied third-party kernel matter; Specialized engine cores aren't publicly reusable. Gufo/Maya/DwarfStar/GGML may offer reusable source subject to actual license, but user wants Rust source; porting from open source requires retaining required attribution and documenting derivative origin. Do not copy external proprietary code from binaries.

## 8. Toolchain and build variants

Candidate Rust toolchain pinned in `rust-toolchain.toml` and lockfile. For NixOS, use flake development shell if selected by user, pinned Rust compiler + GPU compiler toolchains, ROCm libraries through Nix overlays with host driver constraints. The shell should print exactly which environment vars/device permissions required; cannot assume Ubuntu `apt`, current user render group or environment globally.

Separate feature flags:
- `cpu-reference` baseline and `cpu-simd`;
- `rocm`, `cuda`, `metal`, `vulkan`, independently compile/condition;
- `distributed`, `multimodal`, `server`, `profiling`;
- device-specific experimental kernel pack features if needed but never incompatible semantic bypass.

Build should not require an AMD GPU simply to compile CPU tests. GPU availability probed at runtime; error if user selects an unavailable device, generic CPU used only if requested/allowed and model fits.

## 9. CI and physical GPU testing

**CPU CI on every PR:** fmt, clippy, unit/reference, property, format fuzz smoke, generated IR verifier, doctests, build of portable CLI, no model download/network. Model tests use tiny synthetic fixtures with legal redistribution and pinned checksum.

**GPU CI on supported runners:** at least ROCm `gfx1151` development device for extreme path, NVIDIA and Apple where available; kernel parity and sanitizer, full models only if user has access to licensed test weights. PRs changing a backend must report which hardware tests were skipped; compiling `--features rocm` is not execution proof.

**Performance CI:** repeatable alternate A/B with machine exclusive lock, fixed model quant, power and driver telemetry; not run concurrently with normal CI on same GPU. Baselines pinned; small gains require uncertainty report. Preserve raw artifacts and negative experiments.

**Security/fuzz CI:** GGUF/safetensors/tokenizer/cache parser, HTTP payload, distributed protocol and snapshot corruption. Model-file readers remain safe under untrusted inputs.

## 10. Dependency/version governance

Pin runtime dependencies and GPU toolchain versions; record SPDX licenses, source URLs and security patches. Prefer stable Rust crates for HTTP/serialization/IO, avoid dependency on PyTorch/Transformers for inference runtime. Python may be an offline reference/calibration tool (separate `tools`), but not mandatory for `zerogravity serve`.

Plan cache key includes compiler/kernel/backend revisions; do not reuse compiled GPU binaries blindly across driver/ISA changes. Offline mode must not contact HF or fetch executable plugins. Dynamic third-party kernels need an explicit trust/signature scheme, not automatic code execution.

## 11. Pull request checklist

- Purpose and affected semantic operators/sessions, prior baseline and exact benchmark setup.
- Independent oracle tests, model/quant/reference fixture, numerical difference and quality.
- Buffer/state effects and fallback legality; COW/leasing/cancellation tests if touched.
- Physical hardware tests and omitted devices recorded.
- Performance A/B complete model and CPU scheduler cost; no unjustified persistent complexity.
- Update the plan TODO with implemented/tested work and remaining items; change design docs only when the design changes.
- Source license and artifact provenance.
- No secrets or private prompt logs in commits.

A failed performance experiment can be valuable but should not leave dead code, unreachable flags and redundant buffers without good reason. Record failure in research log.

## 12. Build/release artifacts

Proposed CLI/library artifacts and crate publishing remain OPEN. If binary signed releases pursued, automate reproducible build profiles, checksum/SBOM, distro-specific GPU dependency notes and tests. An executable must report its runtime/kernel plan and driver identity for reproducibility, and reject unsupported GPU variants rather than silently execute wrong ISA.

## 13. Agent work tracking

Read [the plan TODO index](../plans/TODO.md) and [AGENTS.md](../../AGENTS.md).
Keep the plan's provenance commit, implementation/test checklist, completed
results and unfinished work in its TODO. Parallel agents use separate run TODOs
until the coordinator integrates them. Use actual tests appropriate to changed
source and hardware. No generated handoff or evidence inventory is required.

## 14. Open structural decisions

Exact crate split, Rust SDK surface, kernel compiler, OS support order, Git initialization, license, CI host hardware, code-generation strategy, checkpoint serialization lib, API auth crate and on-disk result schema are proposals and must be validated incrementally rather than prematurely frozen.

## 2026-10-10 integrated v2 addendum — minimal Rust crates and newer GPU compilers

The historical research implementation plan proposed ~20–25 separately versioned crates, Rust 1.87, cubecl 0.11.0 and a vendor C++/HIP escape. **These are historical council choices, not necessarily user-confirmed requirements.** Start with ~8–12 cohesive Rust crates/modules; preserve ownership of model artifacts/SIR, region/compiler/CIR, kernels/devices, memory/state, engine/scheduler, service/UX and quality, splitting when compile-time/release/unsafe isolation justifies it. No giant initial crate scaffold without a running vertical slice. Crate namespace and binary name remain OPEN except project name zerogravity.

GPU kernel source should remain Rust authored under user's instruction; using system CUDA/ROCm/Metal runtime APIs by FFI is a separate open policy from vendoring non-Rust kernel source. CubeCL is a candidate. **Additional Rust kernel compiler options** include a stable CUDA tile-language frontend and an experimental Rust SIMT/PTX compiler; evaluate them as independent KernelProvider candidates before choosing GPU compiler source policy. Their toolchains can live in separate optional crates/build steps; one experimental nightly path must not force nightly for CPU/core.

The multi-IR pipeline is **SIR+RIR+KIR/CIR**, with direct `ExecutableBundle` and no per-token IR walker. Effectful model state and async buffer/event lifetimes are Rust-typed; `unsafe` limited to reviewed device FFI/allocation/inline intrinsic wrappers, no unchecked Send/Sync across device affinity. One `EngineCore` logical owner and typed worker event protocol, no `Arc<Mutex<Session>>` in hot path.

**Build/CI qualification:** source-pin ledger from `research/inference/refs/COMMITS.txt`, compiler kernel/provider versions, Nix dev shell, CPU tests with no GPU required; CUDA/ROCm/Vulkan hardware lanes as available; fuzz SIR effect verifier, CIR timepoint/alias and untrusted GGUF/snapshot; actual physical `gfx1151` benchmark gate. Repo license, user-facing binary name, MSRV and vendored kernel policy are OPEN; keep old research pin list as comparison, not installation orders. Read `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`.


## 2026-10-12 integration — optional XRT/AIE backend and Rust source policy

The supplied Gufo fork implements the NPU program using Python-generated AIE IR and C++ AIE code and host C++ XRT bridge; compiled xclbin and instruction files are executable device artifacts. A Rust wrapper around that external kernel is **not** a Rust-authored owned kernel. Investigate Rust-generated AIE device IR or a Rust→AIE compiler path and XRT device runtime FFI, then confirm license, program hash and real hardware execution. Until feasible and explicitly policy-compatible, NPU KernelProvider must be disabled/blocked and GPU-only remains fully functional.

Module seams: device::xdna probe and BO/events, kernel::aie capability/program, compiler::rir output-row split, compiler::cir GPU/NPU timepoints, memory pinned leases, runtime worker/ModeReceipt. A new NPU kernel must not touch model math, tokenizer/HTTP or independent GPU kernels. CI without XRT compiles/runs CPU+GPU and mock cross-accelerator tests; physical hardware NPU qualification is a separate opt-in lane. Never add automatic privileged fabric clock controls or reusable untrusted device binary downloading.

## v4 refinement — minimum crates, stable Rust interfaces only where proven

The **initial codebase should be one Rust library plus one executable** (or 2–4 crates after proving a device compiler/unsafe-release boundary requires it), organized into modules for model/artifacts, math/effects, trusted TaskDriver, kernel/compiler, device/resource and coordinator/service. The old larger proposed crate tree is a **future optional decomposition**, not the bootstrap to implement. Avoid one crate/trait/dynamic plugin for every AI modality, IR view or accelerator.

Only SemanticProgram and compiled ExecutableBundle are enduring model execution representations; RIR/KIR are internal pass/candidate data, CIR a typed resource command DAG attached to the bundle. A single session/shard writer publishes prepared StateRoot; device suballocators consume global reservation without per-token mutex. `TaskDriver` is a trusted Rust stage controller, not a general workflow language or unrestricted ExternalAction. No mandatory training/autograd compiler in default inference product; future optional GradientProgram/ParameterVersion supported conceptually.

Use feature-isolated source/driver builds so CPU-only Cargo tests do not require GPU/NPU or downloading model weights. Physical kernels are Rust-authored; foreign compiler/device APIs remain policy-gated. Benchmark compiler cold start and host dispatch latency as well as device execution.

## v4.1 — vendor graph compiler adapter without kernel-source confusion

The project maintains **one Rust-owned SemanticProgram, TaskDriver, TaskCoordinator, ResourceArbiter and ExecutableBundle**. Its single `RegionExecutionProvider` interface has a native Rust `KernelProvider` form and optional graph/fabric program forms. No separate accelerator-specific inference server, model store or session manager, no TPU-only/wafer-only semantic IR and no mandatory new crate per vendor. Start CPU-only and one native GPU provider; add optional graph compiler/adapters as separately buildable modules or a few crates only if SDK/toolchain boundaries justify.

An external authorized vendor graph compiler consuming an **ephemeral, semantically checked graph interchange** can be an AOT backend while all logical task/runtime remains Rust. **This is not a Rust-authored target kernel/compiler by itself.** Track exact foreign compiler/driver/device program source-language/IP policy and license separately; missing compiler access results in architecture-compatible but unsupported native execution. A proprietary model text API is an external TaskServiceAdapter, not native backend equivalence.

Backend build/CI lanes: CPU-only smoke; mock graph-only/static-dataflow provider and opaque-state rejection; physical native CUDA/ROCm as available; **separate** real TPU/NPU/wafer/ASIC SDK/hardware verification only where accessible. No build-time downloads/credential requirement for basic Rust library. See [universal provider contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## v4.2 — optional authenticated peer feature, no new inference server

Build core CPU/GPU `zerogravity` with **zero cluster runtime overhead or dependencies** unless network execution enabled. One optional peer/transport module (or crate only if build/security isolation proves needed) reuses DeviceBackend, RegionExecutionProvider, ExecutableBundle/CIR, TaskCoordinator/StateRoot and ResourceArbiter. A RemoteRegionWorker receives checked typed work/StateShard refs and executes the same compiled closed SIR region; it is **not** a Rust wrapper around a separate model server or a new model-specific task loop.

**Minimal first protocol:** manually authorized two peers, supported reliable authenticated stream (TCP/TLS or Rust QUIC choice measured), bounded typed messages, buffer credit, source/destination physical tickets, and per-session generation/epoch idempotency; a local stub/simulator first. No mandatory Kubernetes, DHT, consensus/etcd service, public NAT relay, automatic remote shell or model-repo executable code. More advanced peer discovery/overlay tunnel integration and physically qualified RDMA are optional TransportProvider backends. A user-supplied tunnel adapter is not permission to change host firewall/keys/network settings.

Security source review applies to networking and licensed model/shard data rights. Native RDMA/UCX vendor FFI is separately open under the strict Rust-owned source policy; fallback buffered encrypted stream must be correct. CI: CPU-only/no remote, deterministic mock 3-node topology and fault/relay, then optional real 2-node LAN/tunnel and GPU/collective lanes. Tests mark simulated and hardware-qualified separately. [Placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md).

## v4.3 — one executable and one library for CLI, router and multi-model serving

`zerogravity run <model>` (proposed syntax) instantiates the same Rust ModelCatalog/ModelRouter/ResidencyManager/TaskCoordinator/ResourceArbiter/RegionExecutionProvider as the `serve` endpoint. **A CLI direct run does not require a local service process**, although an optional trusted same-user socket reuse mode can lower repeated model startup. No mandatory separate gateway binary, model-specific inference daemon, database, Python virtualenv, dynamic plugin ABI or system-level service installation.

Model catalog and alias/profile overlays can start as modules in the same library; real model loads and model-specific concurrency share physical/DeviceBackend event-pool accounting. Provide optional versioned typed config reader and `config show --effective`, but do not scaffold a large independent control-plane framework before one CPU/GPU source model is runnable. `serve` maps OpenAI-compatible HTTP schemas to canonical TaskRequest; it does not reimplement token sampler, model loader or run a subprocess per model by default.

CPU-only build remains minimal/offline; conditional GPU/NPU and optional authenticated distributed peer modules are independent build features. A remote model API proxy, if ever permitted, is TaskServiceAdapter and not a `ResidentModel` native RegionExecutionProvider. SDK conformance and safe localhost/remote auth/TLS/config precedence are separate quality gates. [CLI/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

````

## Source: .agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md

Original UTF-8 bytes: 24878; SHA256 `a420b6b64ea8a2616633b3ab009924a101c97f91e366ec402a95866b45c2b450`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Scheduler, continuous batching, prefix reuse and speculative execution

**15 Oct 2026 — precise concurrent-serving requirements:** [Joint serving, goodput and combinatory specialization](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) clarifies unified needed-versus-computed task frontiers, bounded per-step token/verification/media work, decode/duplex latency protection **and prefill starvation prevention**, separate queued requests/queued prefill tokens/active sequences and resource credits. P15 MTP/DFlash target proposals and P16 source/quant-compatible batch plans jointly account expert-row padding, unique routes, cache residency and full rollback—not batch or draft depth alone. A small **prequalified Pareto candidate set** is chosen at safe boundaries with hysteresis; default target-only/reference path remains. [SERVOPT-01..36 future tests](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md). vLLM already supports continuous batching, hybrid KV and disaggregation; these requirements aim to compete using demonstrated *joint* optimization, not assert feature novelty.


**15 Oct speculation/batching precision addendum:** [exact MTP/DFlash source draft contract](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) defines TrainedMTP/DFlash/lookup proposal-specific attention, tap, mask and mutable state; target alone verifies and commits accepted prefix, including stochastic sampler, grammar and full GDN/KDA/MLA index rollback. Work rows equal sum of (1+each draft depth), but actual cost includes grouped expert quant kernel tile/padding, unique routed experts and per-session state. Policy may back off to **target-only** on low acceptance without changing model/quant. Distinct batch B and near-tie numerical modes require explicit receipts; benchmark interactive P95 separately from aggregate committed throughput. HYB-06..18 and GFX-07/08 qualify this contract.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**Architectural core:** One request scheduler controls token budgets, priority, admission and complete state commit; one device-plan executor runs fused kernels; one global ResourceArbiter reserves physical memory. The original runtime scheduler jointly considers token work, prefix reuse, routed expert/lookup IO, physical memory, speculative verification and task fairness. No guarantee that complex scheduling is superior without measured thresholds.

## 1. Entities and lifecycle

```text
Incoming API request
  -> parse/render/tokenize & validate model/modality/sampling
  -> lookup compatible live/durable prefix
  -> estimate per-session mandatory state/scratch budget
  -> WAITING (may cancel/expire)
  -> ADMITTED (reservation and state fork)
  -> PREFILLING (chunks) <--> RUNNING/DECODING
  -> optionally DRAFTING/VERIFYING
  -> commit verified frontier and STREAM emitted tokens
  -> COMPLETED | CANCELLED | FAILED
  -> release or snapshot state by cache policy
```

A request may have pending prompt tokens while older requests decode; no exclusive "prefill engine". Scheduler prepares execution groups, chooses plan variant, preleases memory, dispatches async GPU work and processes completed events. GPU execution should proceed with no host decision between resident MoE ops within a captured/specialized region.

### Request ledger fields

```rust
struct Request {
    id: RequestId,
    model_id: ModelId,
    queue_class: QueueClass,
    s_lo: LatencyPolicy,
    input_tokens: Vec<TokenId>,
    committed_frontier: u64,
    computed_prefix: u64,
    max_new_tokens: u32,
    sampling: SamplingContract,
    mode: WorkMode, // prefill, decode, verify
    session: SessionId,
    cancellation_epoch: u64,
    deadline: Option<Instant>,
    resource_reservation: Option<ReservationId>,
}
```

Code sketch only. Request data and state access must avoid copying massive token histories on each schedule.

## 2. Unified token-budget scheduling

Given global capacity per cycle (tokens, running sessions, launch budget, memory scratch, device resources), choose currently running requests and new admission in one bounded pass. For each request compute required new tokens from `pending_prefix_end - computed_frontier`; allocate up to token budget and candidate plan's supported row shape; established decode requests usually require 1 new target token or a verify batch.

Chunked prefill is a service fairness mechanism and kernel-shape optimization simultaneously. A fixed 128/256 token quantum may be good for latency but bad for large GEMM utilization; choose per hardware/workload mode, profile P95 and throughput, and retain bounded maximum waiting time.

Scheduling policy must consider:
- per-model tokenizer/encoder work and modality;
- prefix cache hit probability and compatible state availability;
- per-request priority/deadline and fairness;
- model active weights and context/snapshot residency;
- GPU compute critical-path time, queues and outstanding copy/IO;
- token rows available for grouped MoE;
- scratch and per-session KV/recurrent state budget;
- speculative verification rows and acceptance;
- whether changing plan requires state-layout conversion (cost + safety).

### Baseline fairness and admission

Use stable waiting/running queues with aging/priority and optional per-tenant quotas. Running requests must not be perpetually starved by fresh long prompts; new requests cannot evict mandatory running session buffers without safe preemption/checkpoint. Out-of-memory admission fails with actionable budget report; do not preempt by corrupting sessions.

## 3. Execution groups

A group has compatible device model, kernel plan, precision/quant, operator-region state contract and batch rows. Several sessions can share immutable weights but retain private state. Mix:
- initial/continued prefill with token rows;
- target decode with one row per sequence;
- speculative verification with several rows per sequence;
- attention/recurrent state private by sequence and position;
- routed MoE expert grouping after device router selects IDs.

Do not force every row shape into same GEMM kernel. Use separate prefill, small-batch decode and verifier plans as justified by measured performance.

## 4. Route-aware MoE execution and cost

At layer L, each sequence token selects `top_k` experts. A grouped kernel can reuse expert weights across all rows routed to the same expert. Example four Qwen/GLM sequences may yield only ~15 distinct expert IDs for 32 route assignments; the actual distribution may be much less skewed and grouping may cost more than it saves.

**Mechanism:**
1. GPU route operations produce ids, gates, numerical scores.
2. GPU builds compact assignment list `(expert, session row, gate)`; optionally radix/count sort by expert within a bounded scratch and occupancy constraint.
3. Device-side table maps expert to resident slot or miss event. Missing IDs batched without stale pointer usage.
4. Resident grouped expert computation processes rows in appropriate MMVQ/WMMA tiles; layer's shared expert may overlap legally.
5. Scatter/sum to original row according to numeric contract; update layer state.
6. Profiling records distinct experts, group occupancy, byte reuse, misses, scatter and launch overhead.

**When not to group:** single row, too many unique experts, small workload, little overlap, expensive scratch or numerics requiring strict fixed reduction not achievable efficiently. Choose simple per-row path.

**Important:** scheduler generally cannot know route IDs ahead of previous layer activations. Memory decisions can use expert popularity as hints, but correctness requires on-device routing and miss handling; no blocking host lookup every token.

## 5. Prefix-aware scheduling

A token-prefix index supplies computed token length and exact compatible state snapshot identity. Rank new requests partly by reusable state and available memory only where fairness allows. A 40K shared prefix may justify admitting a request without 40K recomputation; if state is offloaded to disk and takes longer than replay, choose recomputation based on cost.

Prevent cache hoarding by low-value/one-off prefixes. Eviction score can include future reuse, state bytes, prefix length, reconstruct time, branches and disk IO. Keep LRU/LFU details local to PrefixCache; scheduler consumes estimated cost and handles memory reservations.

## 6. Speculative decoding architecture

A **Drafter** proposes token IDs but does not authorize them. A **TargetVerifier** runs the authoritative model on proposals, returns logits and provisional state snapshots. A **Sampler** chooses accepted prefix/correction under declared numerical/sampling policy. A **Controller** chooses drafter and draft length based on measured net benefit.

Drafters:
- native MTP (Qwen/GLM) with model-matched weights;
- DSpark for compatible DeepSeek version (support checkpoint verified);
- EAGLE-like trained drafter if supported;
- small independent auxiliary model;
- prompt-lookup/copied spans/n-gram suggestions;
- disabled.

No two proposals should be combined without an algorithm with correct target verification. If drafter weights unavailable or incompatible, disable rather than invent.

### Exact sampling reminder

For target distribution `p(t)` and proposal distribution `q(t)`, stochastic speculative sampling requires acceptance/correction rules that preserve `p`; a common one-draft form accepts candidate `x ~ q` with probability `min(1,p(x)/q(x))`, and upon rejection samples from normalized positive residual `max(p-q,0)`. Multi-token/tree variants additionally require careful conditioning and RNG handling. Deterministic greedy mode validates against target argmax within selected numerical plan, not necessarily bitwise cross-kernel equality.

The exact algorithm and FP32/FP64 probability arithmetic must be independently reference tested. **DwarfStar's default opportunistic sampled mode changes target sampling distribution**; zerogravity proposal is exact mode by default with explicit approximation opt-in, pending user approval.

### Transaction for verify block

```text
take snapshot of all required state domains at frontier p
draft up to d tokens; optional prompt lookup
run target forward over d proposals in one state-provisional region
compute acceptance/correction and valid a
commit all KV/GDN/KDA/indexer/history/RNG at chosen a
discard rejected state versions, reclaim scratch, emit accepted/corrected output
maybe adjust drafter policy at request/step boundary
```

Correctness for stop/EOS/structured output/tool-call boundary: stop may occur inside verified block. Accept only up to actual semantic boundary, roll back rest; if sampling mode changes, restore appropriate pre-verify snapshot and re-evaluate after boundary rather than pretending the draft has same policy.

## 7. Speculation cost model and guardrails

Estimated benefit:
```text
committed_tps =
    E[useful committed tokens per cycle] /
    E[draft_compute + verify_compute + state_snapshot/rollback + memory/IO +
      scheduler overhead and critical-path non-overlapped time]
```

Use actual measured critical-path time, not sum where stages overlap. Controller policies: initialize conservative, sample acceptance for small number of cycles, expand/shrink depth gradually with hysteresis, bypass on poor acceptance or memory pressure, re-evaluate at context or workload transitions; do not disable speculation based on one noisy token. Minimize per-token controller CPU work and toggles that invalidate graph captures.

An extreme kernel plan can optimize verification groups using MoE weight reuse and shared projection data, but must not change sampler semantics. Draft weights compete with expert/prefix caches for physical RAM, so speculation is a resource-policy decision too.

## 8. Active batch vs queued concurrency

Explicit distinction:
- **HTTP concurrency:** multiple incoming requests being accepted/queued.
- **Session concurrency:** multiple live state buffers present.
- **Native GPU batch:** multiple independent rows computed together, potentially reusing weights.
- **Throughput:** aggregate committed token rate, with P50/P95 per-user latency.

Some source projects report concurrent sessions but execute sequentially; zerogravity must label `ordered-fallback` distinctly from `native-batched`.

## 9. Multi-model hosting and eviction

User goal implies running many model families as options; not necessarily resident simultaneously on 96 GB Strix Halo. Model manager can unload/evict model-specific immutable weights when inactive, using reference-counted session frontiers and optional checkpoint/replay. Do not evict a model that still has in-flight kernels or live sessions without explicit admission policy. Cold switching time and downloaded model disk size must be visible to the user.

## 10. Distributed scheduling and prefill/decode disaggregation

Optional P/D split and remote expert parallel when network and throughput justify transfers. Every request includes model/plan/quant and committed frontier identity; remote workers may not invent independent stochastic samples or proceed after a lost collective. Scheduler sizes network and memory budgets, and can keep local single-device fast path free from transport overhead.

## 11. Cancellation, priority and stop conditions

Scheduler assigns cancellation epochs. A request cancelled in WAITING simply releases admission estimate; ACTIVE request marks pending transaction invalid, waits for current kernel/IO events to settle, emits already committed tokens only, releases state or persists valid prefix if policy. Stop sequences apply to committed token stream; never emit speculative drafts. Slow streaming clients have bounded queues and can cause request cancellation/backpressure without retaining unlimited GPU scratch.

## 12. Metrics required for correct tuning

Per request: queue time, TTFT, prefill tokens/rate, decode and total tokens/rate, per-token P50/P95, context depth, prefix cache reused tokens and state domains, speculative drafter and acceptance distribution, num verifier forwards, output quality/error, terminal reason. Per engine: GPU active %, memory allocations, expert hit/miss, unique expert grouping, IO bytes/latency, graph capture vs eager, task fairness, request rejection, P95 under load, per-model weights resident.

Do not report generated drafts as output tokens; report committed tokens only. A 60 tokens/s repetitive draft workload does not imply 60 tokens/s unpredictable reasoning.

## 13. Tests and acceptance

- C1/2/4/8/16/32 mixed workload with different cached prefixes and route distributions.
- Simulated high-variance prompt arrivals, context depths 0/4K/32K/128K and output lengths.
- Cancellation during prefill, router miss, MTP verify and after partial tool boundary.
- Equal input model state with different scheduler interleavings: same-plan state/logits with numeric tolerances and deterministic sampling configuration.
- Draft lengths 0..8, acceptance 0..d, EOS early, RNG exact/correction distribution, draft mismatch, model mismatch.
- Forced OOM admission and scratch shrinking; no running state corruption.
- HTTP P50/P95 and aggregate throughput compared with raw engine independent of network.
- Cross-session isolation with shared weights and immutable prefix fork under simultaneous GPU batch.

## 14. Open questions for empirical resolution

Optimal fairness policy, prefill quantum by GPU and batch mix, GPU route grouping critical-path tradeoff, cost model sample frequency and hysteresis, prefix eviction vs expert cache priority, speculative exactness vs optional opportunistic modes, and time to checkpoint/restore for serving preemption. Avoid prematurely locking these constants.

## 2026-10-10 integrated v2 addendum — concrete research event loop and scheduler ownership

The implementation core is research's `EngineMsg/EngineEvent/StepPlan/WorkerDelta` single logical engine owner plus per-device worker with persistent `InputBatch` diffs, **not** a global locked session list. See `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`. A `StepPlan` references already-compiled `ExecutableBundle` variant handles, state epochs, row assignment, KV block leases and CIR timepoint dependencies. A device worker may complete computation but EngineCore alone commits externally visible token frontier across all state domains.

Use two independently constrained budgets: target compute work (new prefill/decode/verify rows) and input/encoder work; then physical resource reservations and fairness/latency rules. Research's 128/2048 mixed-prefill quanta and 2ms coalescing are **measured candidates**, not unconditional settings. A generic paged token-budget `KvPool` and low-concurrency extreme contiguous ReserveAll mode are explicit different admission/preemption policies, requiring different worker output and state region mobility; switching them is *not* a pure flag change.

Speculation retains research `DraftSource`, `VerifyMode`, per-hardware acceptance EMA+backoff, and `CommitPlan/Frontier` optimizations, but must obey the project's full typed state/RNG/grammar transaction. Adaptive depth uses **committed** tokens per critical-path time, and draft weights/extra verifier rows must be charged against global memory. Exact target-distribution sampled mode is the working fidelity default, with explicit opt-in when a mode intentionally changes output statistics. Model-specific MTP and DSpark source-artifact compatibility is mandatory.

**Joint optimization experiment:** GPU routes target/verifier candidate rows by expert; group rows and reuse matrix source only when more efficient and numerically legal. Scheduler may consider cache pressure, expert union, scratch and prefix hit, but it cannot know future expert selection before router activation; device table and bounded miss service remain authoritative. serving-aware kernel-tuning research 2026 demonstrates execution-mode tactic variation: record captured-versus-eager mode per plan.

New backpressure/failure tests: two overlapping StepPlans with grammar/tool boundary, cancel after first visible token, 0..k accepted draft, GPU cache miss and slot ABA, 32 concurrent requests and long prefill. No output until all state domains commit at same causal frontier. Emit ModeReceipt showing native vs ordered batch, actual draft acceptance, numeric policy and reason for fallback.


## 2026-10-12 integration — prefill-specific GPU+NPU candidate

Eligible large prefill matrix projections may use RIR output-row split across RDNA iGPU and XDNA2 when source quant, batch shape, NPU code, numeric quality and SoC concurrency are qualified. Gufo's fixed GEMM requires prompt groups of 4096 tokens; short/ragged prefill, decode1 and most speculative verifier batches should remain GPU-only unless new NPU kernels prove a gain. Scheduler selects among prevalidated plans, and CIR owns asynchronous GPU/NPU first-use joins.

Physical pinned memory, shared LPDDR bandwidth and NPU queue/power state count against the global budget. ModeReceipt reports actual row offload and wall-time benefit, not simply NPU presence. Speculative decoding target verifier and complete GDN/QSA/KV/RNG state remain authoritative. For ASR/TTS/image/video use typed TaskProgram stage budgets and output events without pretending all tasks are token generation.

## v4 refinement — mixed task cost, prepared state and bounded overlap

TaskCoordinator schedules **ready task stages**, not necessarily text tokens, using typed `WorkUnit` and physical budgets. Text generation stages can use target token/prefill/verification budgets, while image denoise, audio encoder/vocoder, video frames and embeddings need their own cost/frontier controls. Preserve fairness/age/latency targets and backpressure so one long prefill/task cannot starve ongoing interactive output. A request accepted draft sequence is only a provisional state until target verifier and all domain Prepare checks pass.

**No independent domain commit during speculation:** draft and verifier kernels produce prepared versions; logical `PublishRoot` and output happen once after GPU events and target distribution check. Device-native verification loops can avoid host trips when grammar/sampler state supported; host loop remains a correct fallback. Kernel candidates with unsupported shape, compute quant or state layout are gated before launch or cause checkpoint replay.

Physical resource reservations are coarse; one shared global lock per token/expert cache fill is prohibited. Dynamic prefetch/route results are **data**, not requests to JIT one kernel per expert ID. No unapproved top-k/mask/sampler reduction is an optimization.

## v4.2 — cross-machine task admission and stage scheduling without central hot-path RPC

TaskCoordinator owns **one session/shard TaskDriver and StateRoot** regardless of how many physical peers execute it. Stage admission checks per-node physical reservations and network credits before submitting a multi-peer StepPlan, preferring colocated execution when network cannot meet p95/SLO. Remote workers keep their layer weights/KV persistent; only bounded stage-boundary activation and Ready/Failed messages transit each work unit. The coordinator need not synchronize every tensor op across nodes if a remote RegionExecutionProvider owns a compiled whole-layer/stage/fabric program.

For P/D, reserve decode destination memory and transfer complete compatible KV/recurrent/indexer/modal state at phase boundary; for A/FFN split, route hidden activations and compact partials with per-layer queue/RTT accounting; for long-context attention shard, only mathematically qualified softmax partial summary reduction. Immutable remote prefix cache is used only when hydration beats recompute. Preempt/cancel on slow peer must **fence placement/session epochs** and keep authoritative root, not orphan selected tokens or draft recurrent state.

Fair sharing across nodes: per-task typed WorkUnits, weighted deadlines/credits, tenant access scope and backpressure; topology may become unavailable mid-session and require planned replay at last committed frontier. Speculative target correction remains authoritative and distributed draft branches cannot publish state on a worker. Transport metadata and peer-safety do not enter generic one-host token hot loop. [Remote protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

## v4.3 — multi-model route admission and native within-model batching

Keep two **timescales, not two inference schedulers**. `ModelRouter` resolves exact named source and selects/waits for compatible ResidentModel/Replica under ResourceArbiter at request admission, using load/warm cost, queue/p95, prefix/StateRoot affinity, effective memory, tenant/authorized peers and TaskKind. `TaskCoordinator/DeviceWorker` performs ongoing per-model/per-task decode/prefill/speculative verification batching in persistent kernel slots; do **not** select a new source/checkpoint or call HTTP router per token.

Multiple different models with unrelated state/weights can run concurrently on qualified devices if physical reservations permit; mathematically compatible sessions under *same model/adapter/precision* may batch. Never merge unrelated model token ID/position/grammar semantics into a GPU batch simply because shapes coincide. Target-draft or embedding-rerank stages are separate registered model role leases under **one TaskProgram budget**, not load balancing equivalent replicas.

Policies: bound global and per-model pending queues, cold model load wait and SLO priority to protect already warm low-latency traffic; deduplicate 100 concurrent requests for one model load; pin/cache-affine active sessions and drain before model eviction; automatic model-choice aliases only explicitly authorized. Model hot reload or replica migration requires full StateRoot/StateCompatKey and committed frontier, not a queue-level `model` string change. [Router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).

````

## Source: .agents/docs/SECURITY_AND_RELIABILITY.md

Original UTF-8 bytes: 23521; SHA256 `b66dbb79129fe5ad3e1400a9fd187258f80e99c3e6157f4c50fd0ffde8555c17`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Security, reliability, resilience and safe defaults

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


zerogravity accepts untrusted model artifacts, tokenizer metadata, prompts, images and potentially network requests. GPU drivers, memory mapping and high-memory MoE loads create host-safety risks beyond ordinary HTTP services. Security must be designed from the beginning, not added after fast kernels ship.

## 1. Threat model

Assets: host files/config/secrets, model checkpoints/provenance, GPU/CPU memory, user prompt/tool histories, cached token/recurrent state, network service/auth credentials, benchmark/model licensing, device stability and availability.

Potential adversaries:
- malicious/untrusted GGUF/safetensors/tokenizer/config/vision assets;
- unauthorized local/remote HTTP clients, very large or deliberately pathological prompts;
- code injected through user/tool content attempting to trigger host execution (not a model-runtime privilege);
- compromised network/distributed worker or stale RDMA memory key;
- crafted cache/snapshot file causing oversized allocations or state reuse under different model;
- untrusted optional plugin/kernel bundle trying to execute arbitrary host/device code;
- non-malicious workloads that exhaust physical RAM/GTT and hang an APU host.

## 2. Untrusted artifact parsing

Every input integer and offset is validated before allocating or dereferencing:
- limit tensor count, dimensional rank, string length and metadata nested structures;
- check all `count * elem_size`, offset+size, shard total and alignment with 64-bit checked arithmetic;
- reject overlapping or contradictory regions and duplicate mandatory tensor names;
- reject unrecognized quant type or unknown architecture rather than guessing;
- handle invalid UTF-8/template/tokenizer tables without UB;
- canonicalize local paths and apply user authorization rules to symlinks/traversal;
- no executing HF `trust_remote_code`, model postprocessing scripts or embedded executable payload;
- downloads to temporary files, verify source/checksum, atomic publish; protect from partial reads.

Use property/fuzz tests with ASan/UBSan where FFI is involved and Rust fuzz/prop tests for parsers. GPU kernel dispatch must not take unchecked dimensions from user model file.

## 3. GPU memory lifetime and physical safety

The GPU can read mapped/pinned host memory directly. A stale pointer or generation mismatch can corrupt model state and in severe cases destabilize driver. Enforce typed BufferLeases across IO/CPU/GPU events, checked state versions and no overlap between in-flight source/evict destination.

On Strix Halo: GPU-visible GTT may be larger than safely free RAM. Respect adjusted `MemAvailable`, `CmaFree` where relevant, OS/cache margin and worst-case prompt scratch. Physical benchmarks require watchdog and exclusive GPU use. Avoid causing host OOM/forced reboot during standard tests. Never auto-edit `amdgpu.gttsize`/TTM kernel parameters, disable IOMMU or change firmware UMA without explicit user action and hardware-specific warning.

Kernel spin waits for expert misses must have bounded progress/timeouts and failure notification; driver error results must propagate. No unsafe memory overcommit to try to fit 321B model.

## 4. Service exposure and request admission

Default bind localhost only; `--host 0.0.0.0` requires explicit choice and warning about authentication/TLS. Protect remote endpoints with authentication, rate limits, per-client token/context/quota controls and bounded upload sizes. CORS headers are **not** authentication.

Request parsers limit JSON nesting, string sizes, tool lists, multimodal bytes/pixel count, context tokens, max output, concurrent sessions and timeouts before allocation. A slow client should not retain unbounded GPU memory or block all request scheduling. Streaming may be cancelled safely after committed frontier.

No arbitrary file reading through `image_url`, local model loaders, tools or `/read` commands without explicit policy. Model serving and an optional agent tool executor are different security principals. If an agent is added, tools must be sandboxed with separate consent and filesystem/network limits.

## 5. Model and prompt privacy

Full rendered prompts, hidden reasoning, tool results, embedding data and KV/recurrent state can contain sensitive conversation data. Debug traces and durable prefix snapshots are therefore private by default. Use restrictive file permissions, configurable expiry/size limit, versioned cache identity and opt-in persistence. No uploading benchmark/model data or prompts to external telemetry by default. If content hashes are used, avoid logging material that allows easy correlation with private inputs.

When exposing traces, redact credentials and secret request fields. Hidden reasoning must not leak through streaming even if model-native template encodes it in output.

## 6. Snapshot integrity and cache poisoning

Prefix cache keys must bind model weights, quant, tokenizer/rendered message structure, multimodal fingerprint, position/numeric policy and state schema. A hash collision or corrupted metadata cannot result in unsafe arbitrary state restore. Serialization bounds check offsets and checksums before memory use, with no persisted raw device pointers. Reject snapshot from different schema/backend layout unless a tested conversion exists. Cached state persistence may need encryption based on deployment threat model (OPEN).

If snapshot loading fails, reconstruct from trusted token history or reject; never continue with half-restored recurrence/KV.

## 7. Distributed transport

RDMA/TCP protocols require authenticated peers or explicitly restricted trusted deployment, checked frame lengths and model/plan/quant identity handshake, monotonically sequenced operation epochs, bounded allocations and timeouts. Registered memory region keys must be scoped and revoked at disconnect. No default public unauthenticated RDMA/TCP cluster on shared internet. Remote worker failures require atomic frontier failure and rebuild, not accept incomplete allreduces.

Do not log credential/session payload or NIC keys. Malicious or stale peer commands cannot induce arbitrary GPU memory writes; validate region keys, offsets and expected tensor shapes.

## 8. Plugins and generated kernel trust

A Rust-authored GPU kernel may compile to device code with potentially serious memory safety consequences. Do not auto-download and execute arbitrary third-party kernels or executable compiler IR from model repositories. Ship built-in signed/built code; if plugins added, define a strict user-consented trust/sandbox/signature scheme, versioned operator ABI and resource limits. Runtime compiler inputs must be validated and code cache isolated to trusted/user-owned directories with checksums.

A model checkpoint is **data**, not an executable plugin.

## 9. Reliability and operational state

- crash-safe model artifact download and cache snapshot writes via temporary file + fsync/atomic rename where needed;
- device error marks provisional state invalid; no silent continuation;
- empty/masked attention selection handled as defined, not NaNs or uninitialized memory;
- bounded IO queues and host-mapped event rings with credits/backpressure;
- strict startup memory-fit and fallback plan selection;
- thread and device teardown waits on in-flight events and cleans cached slabs;
- no incompatible model hot reload while active session running;
- independent operation metrics and structured errors, not hung promise/worker.

## 10. Supply chain and licensing

All third-party code (GGML quant code, Gufo, Maya, DwarfStar, CUDA snippets), model checkpoints, conversion recipes and evaluation datasets require license/provenance review before redistribution. DwarfStar license was inspected as MIT with GGML notices, but other repos' licenses must be rechecked at pinned commit. Some engine sources are not public; do not copy or derive from proprietary implementation. Document source/attribution when independently porting legal algorithms into Rust.

Avoid installing unofficial external binaries or unsigned models without review. Nix/Cargo lockfiles and release SBOM are proposed controls. Avoid committing large proprietary checkpoint data or access tokens in test fixtures.

## 11. Reliability testing before release

**Model inputs:** fuzz GGUF/safetensors/quant scales/shards and broken vision encoder; no crashes/hangs/excess RSS.  
**Runtime:** simulate out-of-memory, HIP device lost, failed event, cancelled kernel, poisoned buffer generation and per-session race.  
**Storage:** partial expert read, corrupted Engram row, truncated snapshot, disk full, read-only file, temp rename failure.  
**Serving:** malformed JSON, 1000 simultaneous invalid requests, tool boundary parse, HTTP disconnect, auth rejects, stream backpressure, large uploads.  
**Distributed:** wrong model handshake, dropped frames, TCP half-close, RDMA reject, worker crash during speculation.  
**APU:** physical memory watchdog under increasing session/context/scratch pressure; admit/reject predictably without system-wide OOM.  
**Numerics:** every safety fallback returns correct result (if safe) or clear error, not plausible invalid tokens.

## 12. Release security checklist

- [ ] Defined model file trust and network resolver policy.
- [ ] All external source/data licenses reviewed.
- [ ] No unchecked parser arithmetic or format downgrades.
- [ ] Public HTTP exposure is explicit and protected.
- [ ] Model-state snapshots private, bounded and versioned.
- [ ] Physical RAM admission and APU watchdog exercised on hardware.
- [ ] Fault injection on expert slots, snapshots, kernels and cancellation.
- [ ] No execution of arbitrary Python model/plugin scripts.
- [ ] Distributed transport identity/credential and fail-closed policy.
- [ ] GPU memory/FFI unsafe blocks audited and sanitizer tests recorded.
- [ ] Secrets/model traces excluded from CI and release artifacts.
- [ ] Performance and quality claims link reproducible evidence.

## 13. Open security decisions

TLS/auth implementation and tokens, snapshot encryption/retention defaults, model download allowlist, kernel/plugin extension policy, optional agent filesystem/network grants, remote execution deployment, release signing keys and public bug reporting/security contact. These must not be silently assumed done.

## 2026-10-10 integrated v2 addendum — compiler, async executor and plugin threat boundaries

The new pipeline creates trust boundaries in addition to the original artifact reader/API: `ArtifactReader` may parse malicious model files, `SemanticOp` adapter may define unknown math, `RewritePass` may perform illegal state change, `KernelProvider` may generate unsafe ISA, `CIR` may schedule stale device/SSD buffers, and distributed/LoRA/vision paths may reuse state across tenants. Only trusted built-in Rust code providers should execute initially; dynamic third-party executable plugins require source/signing/trust and driver memory isolation policy before exposure. Model files are **data**, never permission to download/run HF Python remote code.

**CIR security:** all allocations consume ResourceArbiter reservations, every async DeviceEvent/Timepoint has source epoch and buffer lease generation, and expert IO `pread` validates artifact identity/offset/bytes before publish. Graph capture must not dereference host pointers whose lifetimes expired. No infinite GPU spin while waiting for host ring completion; host service error/timeout leads to safe provisional abort. Model cache and prefix snapshot lifetimes are private, tenant-salted and copied/serialized only under validated state schema.

**Binary format correction:** research's listed "48-byte" disk-KV header is actually 54 bytes with incompatible five-character-u16 magic. Reconstruct a new checked versioned format and fuzz it before writing or accepting snapshot files; never persist process pointers. Hash identity must include exact token and model/quant/pos/adapter/modality/renderer state, cryptographic digest rather than bare SHA1 prompt bytes.

**Source language / license risk:** historical vendored HIP/C++ kernel proposal must not bypass user Rust-owned source requirement. Optional Rust GPU tile/SIMT compiler backends each bring version/ISA attack surfaces; audit codegen artifacts, driver load, unsupported dynamic compiler cache and kernel source trust. Using vendor GPU math via FFI is an open user policy, not permission to load an entire untrusted inference runtime.

**Control API:** default localhost, bounded authentication/authz, model download checksum, prompt/hidden reasoning privacy, tool/MCP permission separation, per-tenant cache namespace, encrypted/permissioned durable state policy, no remote RDMA without explicit trust. A new device/IO transport must pass event/lease and peer-epoch fault tests as well as ordinary kernel parity. Add a compiler pass negative corpus that intentionally moves `ConsumeRng` or `GdnTransition` across commit and proves rejection.


## 2026-10-12 integration — GPU+XDNA2 SoC hard-hang risk and NPU artifact trust

**Critical external report:** the user's Reddit source `1x3gtdh` and `mikealanni/gufo` fork report **whole-machine freezes / machine-check errors** when Radeon GPU fabric clock changes while the XDNA2 NPU operates concurrently. Fork workaround pins `pp_dpm_fclk` to top, configures XRT NPU performance mode, disables runtime suspend and sends keepalive; 12 clean author test runs **do not prove** general production safety. Some community reports note increased idle power and BIOS/AGESA behavior.

**Safe default and rule:** NPU offload stays **disabled by default** in zerogravity until an exact hardware/firmware/driver and compiler program combination passes controlled physical qualification. Detect and expose `Supported/Experimental/Qualified/Unsafe` capabilities, with typed refusal/fallback GPU-only. **Never issue privileged automatic clock/power/BIOS/sysfs/systemd/IOMMU modifications.** A diagnostic may describe manual opt-in commands, their scope, increased power, reversibility and risks; auto mode must not silently lock clocks. Avoid testing machine-check triggers on an everyday workstation: use an isolated, recoverable bench box and explicit user permission. A daemon timeout cannot recover a frozen SoC.

**XRT/HIP memory and malicious bytecode:** XRT `.xclbin` and instruction stream are **executable device programs** with potential DMA and fault risks. Only trusted bundled/checksummed hardware-qualified programs with declared source/compiled ISA provenance; a downloaded GGUF must never specify an executable kernel URL. A Rust runtime wrapper around Gufo's C++/Python AIE source is **not** proof of Rust-owned kernels; honor source policy and verify licensing per file. GPU-pinned BOs imported into XRT are resources requiring generation-tagged ownership and explicit device sync. Failed/timed-out commands may keep DMA alive; quarantine pages rather than freeing them. Bound BO dimensions, stride, program size, queue depth, device temperature and power.

**Security tests:** software-level simulations for stale buffer generation, XRT BO map failure/short import, HIP callback ordering, worker crash/cancel, partial NPU writes; physical safe tests for driver/firmware contention, 24h run, boot/resume/thermal/power and fallback. CPU/GPU model remains correct with `--npu off` or absent XRT, and no new network or filesystem privilege becomes necessary for normal serving.

## v4 refinement — one logical root and output delivery security

Every provisional state update is isolated; **no independently mutating per-StateDomain commit** or CIR device-side session publish exists. TaskCoordinator checks prepared versions of KV/recurrent/indexer/RNG/media/grammar and required GPU/IO fences then atomically swaps the logical StateRoot, after which a sequenced committed output may be attempted for delivery. A lost remote ACK means exactly-once client receipt is **not** guaranteed; optional replay cursor requires explicit authenticated retention and dedup. Never expose tentative tokens, audio frames or latent state.

The global ResourceArbiter is admission/pressure authority; local device pools own fast allocations and event-backed retirement under its reservation, with stale generation quarantine. External memory pressure can still occur; fail safely rather than claiming OS memory cannot change. No model-supplied code, device bytecode or privileged agent actions in untrusted inputs. GPU/NPU power/fabric risk is a hard device qualification veto; no automatic root/clock/IOMMU/firmware setting. Gradient/parameter updates (if ever supported) require new immutable ParameterVersion and inference session isolation.

## v4.1 — compiled graph/wafer programs, remote sessions and license trust

A graph compiler or programmed fabric produces **executable device code/program images** with model/data access; it is not safe just because our Rust host passed a typed program to the SDK. Verify provenance/driver/compiler/program signatures or trusted build source, op/shape/quant validity, buffer limits and device safety before load. Runtime device access may require vendor closed binaries/FFI; user's Rust-authored kernel source policy still applies separately and remains open for graph targets—**do not claim a native Rust TPU/wafer/LPU compiler exists without proof**.

A managed accelerator service that only offers model input/output and no trusted snapshot/KV/RNG/gradient and effect semantics is not a Native `RegionExecutionProvider`; classify as an external authenticated **TaskServiceAdapter** with explicit user consent, tenant/session/output contracts, remote data-privacy and egress/security cost. Do not leak prompts, checkpoint weights, embeddings or media to a remote device implicitly. Device program/collective groups must be fenced on failures and versioned against actual model/driver/firmware; no exactly-once network delivery claims. Safe host and internal firmware/power restrictions still override optimizer placement.

An opaque graph with unrecoverable in-place state cannot be selected for general speculative/fork/resumable session execution. A stateless encoder subgraph can still qualify with correct input/output and client privacy constraints. Unknown op/quant or unsafe toolchain => fail closed/fallback. Source citations stay in prior-art research.

## v4.2 — trusted network peers, tunnel isolation and remote state safety

**Explicit enrollment before any remote execution**: each peer has cryptographically authenticated identity, tenant/model/weight rights and allowed compute/data classes. Tunnel discovery and IP reachability do not imply access; a new node cannot execute arbitrary binaries, host shell commands or read a model/store outside approved artifacts, nor can the runtime implicitly export prompts, hidden activations, KV cache, embeddings, audio/video or checkpoint weights. Remote cache/task placement must honor source licenses and sensitive-data policy.

Control/metadata and bulk tensor state transfer are **distinct logical planes** with identity, versioned bounded frames, deadlines/idempotency, encrypted transport and optional physically qualified RDMA. A private WireGuard-type tunnel does not provide GPU RDMA or reliable low RTT; direct/relay/cryptographic cost must be measured. Registered RDMA/RNIC memory keys, GPU pointers and tenant secrets cannot be in logs; source/destination memory lease generations and DMA fence must remain valid through timeouts. Remote machine compromise is a separate trust assumption even if link encrypted.

**Single session publisher across network:** remote workers Prepare state, coordinator root contains versioned owned StateShard refs and epoch; only it publishes. Cluster membership/lease fencing prevents old rank/coordinator writes after partition; no false exactly-once client network delivery or durable transparent failover promise. Without verified quorum/heartbeat/checkpoint, fail-stop and reconstruct last known committed state safely; session availability is conditional. Cache directory entries can be stale hints and must be revalidated. Missing transfer bytes, bad model/version or slow peer cannot become a committed target token. [Remote protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

## v4.3 model router, registry and API security

**Serving listener defaults to loopback**, and one local `run model` needs no public server. A remote/API listener requires deliberate bind plus authentication/TLS/proxy policy, tenant quotas, model alias visibility and native admin separation. A local SDK sending a dummy key for syntactic compatibility does **not** authenticate a remote caller; never ignore externally presented invalid keys or log credentials. Admin model pulls/imports/profiles/loaded model controls must not be available to every chat completion caller.

Model names/aliases can reveal private checkpoint research and tenant data. Expose only authorized `/v1/models` aliases; don't reveal local artifact file paths, peer/tunnel addresses, quant private sidecars, GPU topology or last user prompts to an unauthorized client. Model source import/pull requires user intent/allowlisted source, content hash/license/size and cannot run arbitrary Python/model code. Unknown names never trigger automatic cloud model guessing/download; ModelRouter must not leak source prompt/hidden KV/weights to an unauthorized remote peer or TaskServiceAdapter.

Model changing mid-session is a correctness and security risk: pinned immutable checkpoint/quant/adapter/template/alias revision and StateCompatKey remain valid through streaming, request config can't bypass numerical/tenant/memory controls. Router load/unload must hold BufferLease/event fences and shouldn't evict an active unsnapshotable session. Resource exhaustion attacks via model-switch thrash and many cold loads are bounded by queue/load dedup/headroom/per-tenant fairness, with typed rate/overload errors.

OpenAI-compatible fields don't grant inference code tool/web/file execution; **tool calls are structured generated data only** until an external app separately authorizes and executes a tool. Stateless Responses first; durable store, previous-response state and cloud built-ins only with explicit implemented security/privacy/conformance. Local persistent conversation memory is privacy-first opt-in. CORS/media URL fetch/JSON/body sizes and streaming backpressure validated. [Full CLI/API contract](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md), [router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).

````

## Source: .agents/docs/SEMANTIC_IR_AND_COMPILER.md

Original UTF-8 bytes: 23564; SHA256 `44c4050aa02dbb70309dfc1321df74ca20d2db7cc7767dc43b8a9d05a2002338`.
Resolve embedded relative paths against the original path, not this volume.

````text
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

````

## Source: .agents/docs/STATE_CACHE_AND_TRANSACTIONS.md

Original UTF-8 bytes: 23890; SHA256 `7b7d3ca70986f488280c1d309128159e049b7c4f9b1fae0b3c5311b298522763`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Session state, prefix caching, snapshots and transactional semantics

**15 Oct 2026 — generalized hybrid prefix frontier:** [Joint scheduling/prefix contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) requires the **largest source-complete, compatible prefix across every required StateDomain**, not just token hashes or an intersection of two KV groups: dense KV, SWA, GDN/KDA/SSM, MLA/QSA/DSA indexing, mHC/MTP/DFlash, processor/media state and sampler. Cache identity binds original/derived artifact, tokenizer/template/processor, positions and numerical mode, exact committed state version and tenant ACL; restore from RAM/SSD competes economically with recompute. Missing recurrent checkpoint means shorten reuse or replay, never claim all cached tokens reusable. Forks are COW, with encoded GPU readers pinned through actual completion. SERVOPT-12..18 and 26 test N-domain cache mismatch and tenant security.


**15 October 2026 — DISK STATE IS NOT EXPERT SOURCE:** [Virtual residency/SSD contract](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) shares physical IO/BufferLease but keeps **immutable routed expert bytes** separate from **mutable, tenant-private committed KV/GDN/SWA/prefix StateDomain snapshots**. Disk KV/prefix restoration needs exact model/quant/tokenizer/source/session/epoch compatibility, integrity and optional encrypted/private persistence; saved StateRoot cannot be restored merely by mapping a GPU pointer. A disk save after committed root may append asynchronously, but cannot publish uncommitted speculative state or free in-use GPU cache. DS4/specialized-engine disk resume prior art is not proof of source-compatible constant-time GPU restore. Tests SSD-27/28; unresolved retention/consent policy explicit.


**15 Oct hybrid cache/draft addendum:** [MiMo/GLM/Qwen state contract](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) requires independent source per-layer full paged KV, sliding-window **fixed per-session rings**, GDN/KDA/MLA/mHC/indexer and trained draft caches, with generation, source position, window lookahead and rollback. SWA ring bytes are **not KV per-token**; DFlash has its own ring/hidden tap and physical charge. Complete accepted StateRoot includes draft target verifier effects, RNG/grammar/tool response, and optional MTP/eh_proj identity. No sharing mutable cache among sessions; single TaskCoordinator retains publish authority. Tests HYB-06..12 and existing snapshot/version/ABA tests qualify.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


This specification unifies paged KV, token-prefix indexes, hybrid recurrent/indexer snapshots, serialized session histories and event-safe resource ownership under one full-state transaction. **A correct prefix snapshot must represent every required state domain at the same logical frontier.**

## 1. The distinction between state and model parameters

**Immutable model lifetime:** checkpoint tensors, converted packed layouts, dense/shared weights, routed expert source metadata and lookup table sources. Multiple sessions share these. Expert residency may change physically; tensor identity and quant semantics do not.

**Mutable session lifetime:** logical token history/frontier, KV/attention caches, recurrent state/conv buffers (GDN/KDA), sparse indexer/compressor state, n-gram/Engram history, vision/audio positions, RNG/sampler, draft/MTP state and cancellation epoch. No session can mutate another session's committed buffers.

**Short-lived invocation lifetime:** prompt batch activations, quantized activation staging, scratch, provisional speculative rows, in-flight GPU/IO events, draft proposal buffers. These may be pooled only under lease/generation rules.

## 2. Session data model (suggested, not final code)

```rust
struct SessionIdentity {
    model: ModelIdentity,
    semantic_schema: u32,
    state_layout_schema: u32,
    numeric_plan_hash: [u8; 32],
    rendered_prefix_hash: [u8; 32],
    position_policy_hash: [u8; 32],
}
struct SessionFrontier {
    committed_tokens: u64,
    request_epoch: u64,
    state_generation: u64,
}
struct SessionState {
    identity: SessionIdentity,
    frontier: SessionFrontier,
    kv: Vec<KvStateHandle>,
    recurrence: Vec<RecurrentStateHandle>,
    indexers: Vec<IndexerStateHandle>,
    ngrams: Vec<NgramHistoryHandle>,
    modalities: Vec<ModalityStateHandle>,
    sampler: SamplerState,
    draft: Option<DraftStateHandle>,
}
```

Each handle owns a typed lease and an exact `version`. A `Session` controls all mutation; cache and device managers own physical allocation. A finalized prefix is immutable and reusable across sessions if and only if model/quant/pos/template/layout/numeric policies are compatible.

## 3. State domains, required saved content and hazards

| Domain | Checkpoint payload | Typical hazard |
|---|---|---|
| Dense attention KV | page table, K/V blocks, position/mask/rope metadata | stale page after prefix fork, incorrect causal mask |
| Sliding-window KV | visible-ring position and overwritten region or reconstruction plan | parent/sibling ring corruption |
| Gated DeltaNet | per-head recurrent matrix, per-session short-conv history, gates as needed | speculative rejected tail changes later logits |
| Kimi Delta Attention | KDA fine-grain state and conv, clipping/decay mode | treating KDA as scalar GDN |
| Sparse QSA | raw/pooled index keys, attention KV, selection boundaries, positions | wrong top-k after rewind, stale compressed rows |
| Sparse MLA / DSA | latent compressed KV, indexer pools/keys, masks, NoPE assumptions | expanded vs absorbed layout mismatch |
| DeepSeek CSA/HCA | raw recent window, compressed historical rows, indexer for selected layers | wrong compression ratio/history |
| N-gram/Engram | recent token IDs, hashing/convolution history and metadata | wrong lookup rows after fork |
| Multimodal | encoder embeddings or fingerprinted source, visual spans/positions/rope maps | identical text prefix but different image |
| Sampler | PRNG algorithm/state, filtering history, penalties and logit frontier | repeated/consumed RNG after rejected draft |
| MTP/DSpark draft | trained predictor hidden/cache/history + proposals | draft state ahead of target |

A state snapshot that contains just ordinary KV is invalid for GDN/KDA/QSA/MTP models. A text-rendering cache key that ignores image identity is invalid. A SHA of rendered bytes alone cannot replace checking model/quant/position/format compatibility.

## 4. Transactional generation protocol

Each externally visible action is tied to a monotonically increasing committed token frontier. Session operations:
- `begin_step(parent_frontier)`: verify generation, allocate provisional state and scratch reservations.
- `execute_target`: launch state transitions into provisional/isolated writable buffers.
- `verify`: read target logits, proposals and sampling, decide accepted prefix `a`.
- `commit(a)`: publish all state-domain handles and updated RNG/token history for position `parent+a` **atomically at session level** after requisite GPU events.
- `abort`: invalidate all provisional buffers/events, restore prior committed state or force safe replay from immutable snapshot.
- `cancel`: prevent further commits after cancellation epoch; wait for or safely abandon in-flight device events before recycling buffers.

The commit must correspond to actual verified tokens; do not claim state was updated at `p+a` if one indexer or recurrence remains at `p+k`. All committed domains must have same logical position even if they store compressed representations with different physical row counts.

### Pseudocode

```text
frontier = session.committed_frontier
tx = begin(session, frontier)      // reserves buffers + branch versions
proposals = drafter.maybe_propose(frontier, policy)
verified = target.execute(frontier, proposals, tx.provisional_state)
if error OR cancelled:
    tx.abort_and_quarantine_inflight()
    return failure
decision = sampler.verify(proposals, verified.logits, tx.rng_branch)
if !tx.can_publish_all_domains(decision.accepted_prefix):
    tx.abort()
    return safe_failure
tx.commit(decision.accepted_prefix)
emit_only_committed_tokens()
```

Speculation may verify an initial seed and several proposals, so `accepted_prefix` indexing must unambiguously state whether the target seed is included. Test lengths 0..k, stop at tool/end boundaries and eos in middle.

## 5. Snapshot and copy-on-write (COW) policy

Snapshots are **immutable references to logical state**, not necessarily eager copies of all buffers. Techniques:
- KV pages frozen with reference counts; new continuation allocates new tail page or COW on mutation.
- Recurrent dense matrices: private per-session live state, deferred snapshot materialization; partially copied state tiles if proven correct; COW or simple copy chosen based on actual cost.
- Sparse indexer: append-only committed block keys can be referenced; mutable current block/ring needs private copy and exact row counts.
- Ngram/token-history: small value copy.
- Multimodal position/fingerprint: stable immutable descriptors; large vision embeddings may reuse frozen cache under matching fingerprint and model.
- Sampler state: exact value copy before proposals, but parallel draft RNG sequence must not leak into target RNG when rejected.

Snapshot lifetime extends through consumers' in-flight GPU events, disk serialization, branch forks, cache eviction and error handling. A released parent must not invalidate a still-used child.

### ABA protection

Leases include `(allocation_id, generation)`. Replacing expert/KV/snapshot slots increments generation; async callbacks must validate this generation, not just reuse address. COW mutation prohibited while frozen readers exist unless a new physical buffer is used. GPU event completion required before recycle.

## 6. Prefix cache identity and compatibility

A prefix key must include at least:
- model checkpoint and tokenizer fingerprint (not only human model name);
- quantization artifact/layout and numerical/position policy;
- exact rendered token IDs and prefix boundary, tool/historical bytes where semantics require;
- multimodal spans plus content fingerprints/encoder config and positional transforms;
- state schema and graph/effect version, compiler kernel plan compatibility constraints where exact restoration depends on them.

Plan-specific buffers may be converted to canonical state for reuse across different kernel plans. If conversion is not validated, reject cross-plan reuse and recompute from tokens; do not silently load invalid layout. Similar quantizations may yield different logits: cross-quant prefix reuse should be off by default unless deliberately qualified.

Prefix sharing needs a radix-tree or equivalent token-prefix index and a separate storage layer for *full state snapshots*. Indexing should be bounded; scores for eviction may consider reuse hits, prefix length, recompute cost, lifetime, in-memory and disk sizes, context relation and model popularity.

A model with a 40K shared prefix and eight agents should be able to reuse recorded final recurrent state rather than replaying GDN 40K times when compatible.

## 7. Snapshot persistence schema

Proposed file format:
```text
magic + schema/version + endian + checked sizes
model/checkpoint/quant/tokenizer/semantic hashes
numeric/position policy + modality fingerprint
rendered prefix identity + token IDs + commit frontier
buffer table: state domain, layer, dtype, layout, logical shape,
              codec, bytes, offset, checksum, version
optional sampler/RNG, draft state and tool/rendering history
end checksum/AEAD if policy requires
```

Do not use unchecked on-disk pointers, file-sized allocations before validating header, unbounded nesting or host-native layout without explicit version. Use strong hash to detect corruption and prevent model-mismatch reuse. Snapshot files may contain private user prompts/tool data: secure file permissions and opt-in persistence/retention policy are mandatory. Serialize without blocking GPU mutability beyond necessary event-backed snapshot freeze.

When a snapshot schema changes, either migrate with reference tests or explicitly invalidate and rebuild from text. Do not load stale states for the sake of speed.

## 8. Prefix replay and chunk boundaries

The canonical state for prefix `tokens[0..p]` must be equivalent under:
- initial prefill in one large chunk;
- chunks of 1/256/1024/2048/uneven sizes;
- snapshot load at p;
- full replay from tokens;
- branching and editing after an earlier checkpoint;
- waiting/serving queue interleaving;
- speculative verification with all rejected tokens discarded.

Numerical tolerances or plan-specific behavior may make *bit identity* unsupported across different chunk algorithms; record and separately test same-plan exact replay and cross-plan closeness/quality. Do not claim universal bitwise equivalence where tensor reduction order changes.

### Attention state compression

Some ops retain only compressed or sliding history; snapshot must hold sufficient compressed metadata and recent raw rows. A restored state that has correct physical KV allocation but wrong causal selected-index IDs is invalid. Test just before and after 4:1, 128:1 and QSA/DSA sparsity activation boundaries.

## 9. Tiered cache and memory interaction

Global arbiter can demote prefix state GPU → host RAM → disk, but must know:
- rebuild cost from token history vs bytes copied vs disk load;
- snapshot type and compression, latent vs expanded KV;
- other sessions' COW references;
- swap/staging overhead and physical available RAM;
- interleaving with expert cache and prefill scratch.

An expert cache stores model weights; its eviction cannot corrupt user state. Prefix cache stores mutable/logical context frozen at a position; its eviction may require recomputation. They share a global budget but **must not share one naive LRU policy**.

## 10. Failure and cancellation handling

- GPU launch error: mark provisional region invalid, prevent commit, quarantine GPU buffers until event resolution or device reset; do not emit unverified text.
- IO/truncated table: fail closed; re-fetch or restart safely only when state clear and row identity preserved.
- Corrupt disk snapshot: reject, delete/mark bad after logging, rebuild from canonical token history if available; no partial restore.
- Application stream disconnect: either finish a bounded committed step or abort; never reuse session slot before quiescence.
- Mid-speculation stop/tool marker: choose valid accepted prefix before state commit, rollback remainder. Retokenization of tool rendering cannot silently alter cache frontier.
- Distributed worker loss: all ranks fail and need consistent committed prefix or replay; a local snapshot is not a valid global commit by itself.
- Resource eviction in flight: lease forbids freeing until last GPU/IO reader completes; when memory demand exceeds reservation, backpressure/reject or use smaller plan.
- Power/driver reset: skip optimistic resumption unless driver and runtime can prove state integrity; replay from durable checkpoint if compatible.

## 11. Test suite design

Property-based generated token trees (fork depth, 0..k accept), deterministic scalar oracle, random batch interleavings, stale pointer slot reuse, page eviction during prefill, snapshot serialize/load corrupted byte positions, 40K shared prefix branching, image fingerprint mismatches, exact RNG and penalties. Debug mode logs state domain version/checksum at each commit and forbids invisible writes. Compare per-layer stable intermediate/last logits with reference.

## 12. API invariants for implementers

- `commit` consumes a transaction exactly once, immutable parent remains valid for all siblings.
- All held state pointers protected by a lease whose generation matches.
- One session can't be concurrently mutated by unrelated requests; request scheduler may read immutable prefix while a sibling mutates its own branch.
- Cache hit is usable only if fully validated; partial hits must reconstruct missing domains via a deterministic costed path.
- Any optimizer-generated state representation must declare canonical conversion at every fallback/snapshot boundary.
- No speculative or failed operation may advance external token counter or visible output beyond committed state.

## 13. Unresolved choices

COW granularity per state domain, durable snapshot codec, cache index persistence, quantitative thresholds for demotion and cross-plan reuse, maximum snapshot memory tax and exact cross-backend reproducibility mode. All require benchmarks and explicit policy decision; avoid hardcoding unmeasured constants in the core architecture.

## 2026-10-10 integrated v2 addendum — generalized hybrid prefix state + research frontier mechanics

The original research's `FrontierStore`/`CommitPlan { FullDirect, Prefix(n), Replay(from) }` and bounded frontier ring are now the **physical implementation strategies within** this document's complete state transaction. They are not a replacement for SIR's effect/state contract. A prefix can be committed or shared only if *every* GDN/KDA, KV, sparse/compressed-indexer, token-history, RNG, grammar, modality and LoRA-sensitive state group has an exact compatible representation at that logical frontier.

Implement `StateGroupCodec` with `state_schema_id`, `can_resume(candidate_frontier, stored_rows)`, `snapshot`, `restore`, `recompute_from`, `required_suffix`, `finalization_alignment` and `estimated_cost`; an intersection coordinator finds **common legal prefix frontier across arbitrary groups**. A design limited to two cache groups is insufficient for arbitrary hybrid attention/recurrent/indexer models; do not assume exactly two groups. Compressors that finalize only at 2048-token boundaries may require rounding down and recomputing; never claim an illegal partial snapshot hit. Canonical token IDs/rendered prompt policy, model/quant, adapter generation, multimodal fingerprint and tenant salt participate in collision-resistant identity.

**Paged and contiguous state pools:** historical research paged `KvPool` with intrusive block metadata, chained hashes, optional radix *index* over same pool and extreme contiguous reserve-all `KvPool` are complementary. Both allocate through one ResourceArbiter and are isolated by typed state leases. They do not imply separate user model runtimes; transitions happen at safe checkpoint/replay with layout compatibility.

O(1) `restore_prefix` means logical pointer/frontier change only where physical GPU buffer state has already been safely retained. GDN/KDA dense recurrence, compressed indexer rows and COW mutations may need copy/deferred materialization or replay. Rejected drafts cannot alter grammar FSM, sampled penalties or RNG. Tool/Reasoning template and hidden-state renderer must be snapshotted where necessary. Read `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md` for session owner and `ASYNC_RESOURCE_COMMAND_IR.md` for buffer/event/commit ordering.

**Persistent cache:** do not port research's defective "48-byte" disk-KV struct, which totals 54 stated bytes and declares a 5-character magic as u16. Design a new field-by-field serialization with explicit fixed byte counts and checked lengths, model+token+tenant/version identity, cryptographic hash and no raw pointers. Include corruption/truncation/partial-write/fuzz coverage and migration or explicit invalidation. Snapshot contains private user content; opt-in persistence/private permissions remain mandatory.


## v4 refinement — authoritative prepared-root publication

**Supersedes any independent StateDomain commit wording:** an individual domain returns a `PreparedDomainVersion`; it **must not mutate the committed root visible to other readers**. A `StateTransaction` starting at `StateRoot(epoch,frontier)` checks all required domains, a common legal resume/acceptance frontier and all GPU/IO readiness events. **Only the session TaskCoordinator publishes one new StateRoot.** Logical atomic publication does not mean a remote/GPU physical memory write is atomic; private/COW state, deferred copies or validated replay preserve old roots until publication. An unready/failed device blocks or aborts all new public state, not individual domain updates.

A full prefix hit must match exact source model/quant/adapter/task/token/pos/modality/tenant and a common valid `CanResume` frontier across KV/recurrent/indexer/grammar/RNG. Uncommitted sampler histories and media states never enter a shared prefix. Persist verified schema data, not raw GPU pointers/driver events. Old root references retire only after all device/host readers' generation-tagged leases and events quiesce.

**Required test:** inject failure during second domain Prepare after first succeeds; old root still produces exactly its previous output/state, and no emitted new token/audio chunk leaks. Then test accepted speculative prefix 0..N and cancellation with stale completion.

## v4.2 — remote mutable state owner is distinct from immutable shared cache

**StatePlacement is not CachePlacement.** Every active TaskSession per-layer KV/SSM/sparse-indexer/ngram/modal state has a versioned primary shard handle, owner peer, lease/generation/placement epoch and full StateCompatKey. TaskCoordinator alone publishes a StateRoot referencing those immutable committed versions; peers execute provisional updates and return Ready before publication. A peer directory may advertise a cached model prefix or encoded feature, but remote hit is only valid after content, schema, position, quant, adapter, tenant and *all required domains* have been confirmed and safely imported/recomputed.

**Layer co-location:** ordinary pipeline stage peer owns the layer's KV/recurrent state beside layer compute. **Layer and state decoupling:** if attention runs at state owner, send query and compact result; if compute remains local, migrate compatible state once; direct per-token remote KV read is a distinct, usually expensive topology. `PartitionAlgebra` is required to shard softmax attention KV by token range; ordinary GDN/KDA and global selected indexers don't inherit this automatically. A shared prefix cache cannot mutate a live authoritative state root, and an old peer's lease cannot survive StateRoot owner migration.

**Migration/checkpoint:** at committed TaskFrontier reserve destination, transfer entire compatible state group (KV, recurrent, indexer, modality, RNG if relevant), validate and fence old generation, publish new root/placement epoch or retain old root on failure. Network partition/crash may lose volatile proposed/committed-but-not-durable work; exactly-once client delivery and transparent coordinator failover are not promised. All remote DMA/registered buffers remain leased/quarantined until safe physical quiescence. Details: [placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [remote protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

````

## Source: .agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md

Original UTF-8 bytes: 24409; SHA256 `61411e66c812dc40a9ed37617cb9deff54ee209b9f548aed91409b3275a45142`.
Resolve embedded relative paths against the original path, not this volume.

````text
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

````

## Source: .agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md

Original UTF-8 bytes: 21748; SHA256 `55fec7a0edcd1b48a2e47eee17c8f7f63f63c6c6f5c5c98a0e34cb13001f7d01`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — Unified multimodal execution contracts (v5 design proposal)

**14 October 2026 | Proposed refinement, NOT implemented multimodal model support.**
This is additive to the v4.3 governing docs: MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md, COMPOSITION_AND_EXTENSION_CONTRACTS.md, TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md, and MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md. See companion model-family research in the plans directory.

## MM0 executable mock acceptance — 14 October 2026

**Six of twelve proposed MM acceptance drills now have passing
source-free mock tests** under tests/multimodal_mm0.rs: rational AV clocks
(MM-01), typed component ports (MM-02), atomic fake iterative solver
(MM-03), duplex barge-in with device epoch/lease fencing (MM-06),
synchronized media publication (MM-07), and finite-choice scoring (MM-08).
MM-06 uses the new Rust-only DeviceEventLedger and the existing
TaskCoordinator/StateRoot: a cancelled prior audio stream cannot free
device buffers before signaled completion or publish stale output after a
later user turn. These are mock backend events, not actual GPU hardware,
and add **no** new production inference engine or source-media codec.
The remaining six MM drills and source-verified real media/decision
inference remain unimplemented.

## Executive conclusion

**The v4.3 execution spine is correct but underspecified for comprehensive image/audio/video/decision/multimodel support.** Preserve one native Rust SemanticProgram/ExecutableBundle, trusted TaskDriver, TaskCoordinator sole PublishRoot, resource authority, region provider, executable placement graph and model catalog. Do NOT add independent model-family servers, a general workflow interpreter, or a new persistent IR for each modality.

Six narrow architectural amendments are required:

1. **Typed media values and rational timebases.** Tokens, image/video latents, audio PCM/mel/codebooks, structured decisions, compressed streams and stream clocks must have explicit meanings and constraints.
2. **Verified component assemblies.** A model may comprise encoders, transformers, VAEs, codecs, vocoders, acoustic heads, decision heads and diffusion solvers that can be placed independently while retaining one exact artifact identity.
3. **Generalized task and iteration semantics.** Distinguish decoder token AR, parallel classifier scoring, CTC/RNN-T/TDT transcription, nested audio codebook generation, diffusion/flow solvers, and temporal video reconstruction.
4. **Synchronized multi-track StateRoot and streaming.** Maintain independent audio/video/sample/codec/denoise frontiers with legal coordinated publication, backpressure, low-latency duplex cancellation and replay.
5. **Extensible physical tensor encodings and legal activation transforms.** Model topology must be independent of Q4/Q6/ternary bit packing, Hadamard transforms, RVQ audio codebooks, FP4 or numerical reduction schedule.
6. **Honest composition and service boundaries.** Native model regions, compound local models, multiple independent models, replicas and proprietary external TaskServiceAdapters are all different concepts.

These are concrete refinements to existing types/registries, not six independently owning services.

## MM-01. Typed value and clock contract

The existing generic ValueType must specify the physical/semantic axes. Suggested *illustrative* Rust data contracts:

~~~rust
enum ValueDescriptor {
    Tensor(TensorDesc),           // rank, dtype, layout, axes, source codec
    TokenSequence(TokenDesc),     // text / image / acoustic / interleaved
    CodebookSequence(CodebookDesc), // outer frame, inner RVQ codebook
    AudioSignal(MediaDesc),       // PCM, channels, sample rate, layout
    VideoFrames(MediaDesc),       // dimensions, pixel/color space, timestamps
    MediaLatents(LatentDesc),     // spatial × temporal VAE, source compressor
    EncodedPackets(CodecStreamDesc), // codec ID/config, PTS/DTS
    StructuredValue(SchemaKey),  // typed object or finite answers
    DecisionLogits(ChoiceSchema),// per-question option logits/calibration
}
struct Timebase { ticks_per_second_num: u64, ticks_per_second_den: u64 }
struct TrackCursor { track_id: TrackId, tick: u64, clock: ClockDomainId }
enum ClockDomainKind {
    TextToken, AudioSample, AcousticFrame, RVQCodebookDepth,
    VideoFrame, VideoLatentFrame, SolverStep, ExternalWallTime,
}
~~~

Rational sample/frame times are required. 44.1k/48kHz audio, 24/29.97/50fps video, 12.5/21/25Hz acoustic tokens and text token counts are NOT interchangeable clocks. Resampling/codec decoding must carry explicit source-registered transforms, bounded overlap buffers and rounding residue. Image color space/pixel format, video frame order and audio channel layout cannot silently disappear inside opaque byte blobs. BufferLease can still hold all values physically.

Treat decoded and encoded media as distinct semantic types. A VAE latent image or unfinished RVQ frame is not automatically a playable output. Transport-level media container, mux, packet timestamp, license/rights and finality must be validated.

## MM-02. Verified model-component graphs

A source model can have many components without becoming many independent serving engines:

~~~rust
struct ModelAssembly {
    source: ArtifactKey, semantic_key: SemanticKey,
    components: Vec<VerifiedComponent>,
    typed_edges: Vec<TypedPortEdge>,
    supported_tasks: TaskCapabilitySet,
}
enum ComponentClass {
    Tokenizer, TextEncoder, VisionEncoder, AudioFeatureExtractor,
    AudioEncoder, DecoderLM, SparseExperts, RecurrentCore,
    AcousticDecoder, MultiCodebookDecoder, FlowPredictor, Denoiser,
    VAEEncoder, VAEDecoder, Vocoder, WaveformCodec, UpScaler,
    TranscriptionHead, DecisionHead, TemporalSynchronizer,
}
struct TypedPortEdge {
    from: ComponentPortId, to: ComponentPortId,
    type_and_clock: ValueDescriptor,
    conversion: Option<RegisteredCodecId>,
    state_effects: StateEffectContract,
}
~~~

Every component carries an independently hash-verified weight source, trusted processor/codecs, numerical semantics, shape/precision guards, adapter version and license. Unknown/custom executable checkpoint code is never imported as arbitrary Rust host callbacks.

Examples: FLUX.2 separates a *checkpoint-specific* conditioner ([dev] Mistral-class, [klein] Qwen3 embedding) from flow DiT and VAE; one autoregressive multimodal MoE image model is described as such, but the exposed source generation path must still be audited for diffusion or other non-AR stages; not simply a generic image denoiser; Qwen3-Omni has separable Thinker/Talker and audio codecs; an audio generation model has slow temporal AR, fast RVQ depth AR and waveform decoding; a joint audio/video model has two cross-attending audio/video latent streams; a typed decision model has a Qwen backbone plus a typed schema-scoring head.

A component may be resident on a different CPU/GPU/peer with explicit typed edge transfers, event leases and admission. The TaskDriver chooses when a component runs; the compiler and placement policy decide where its closed semantic regions execute. Keep **component assembly** distinct from a user-authored external workflow and distinct from **replicas** of one model.

## MM-03. Distinct iteration policies under one TaskDriver

Keep one minimal trusted TaskDriver interface but register stage/iteration kinds rather than assuming one token loop:

~~~rust
enum StageIntent {
    Encode(RegionGroup), DecodeText(TokenPhase),
    DecodeAcoustic(CodebookPhase), DecodeTranscript(CTC_RNNT_TDT_BeamSpec),
    ScoreFiniteSchema(ChoiceSchema),
    Denoise(DenoisePhase), FlowStep(FlowPhase),
    Reconstruct(CodecStage), Synchronize(TrackBarrier),
    ParallelKnown(RegionGroup),
}
struct IterativePolicy {
    solver: RegisteredSolverId, schedule: RegisteredScheduleId,
    numeric_policy: NumericPolicyId, rng: RngDomainId,
    shape_bounds: IterationBounds, rollback: StateEffectContract,
}
struct TaskFrontier {
    epoch: TaskEpoch, tracks: Vec<TrackCursor>,
    iterations: Vec<StageCursor>, published_group: CommitGroupId,
}
~~~

Token decode, encoder/embedding pooling, classifier inference, streaming transcription, nested audio codebooks, iterative flow/diffusion, causal VAE and duplex conversation have genuinely distinct termination and validity conditions. A Device-native StepProgram may run many internal microsteps without host round trips, but host TaskCoordinator remains the ONLY publisher of committed state/output.

Diffusion/flow scheduler, timestep ordering, seed and RNG, guidance branches, noise prediction parameterization, distilled step count and conditional expert routing are part of the **task/model semantic contract**. Altering these for speed is a new quality mode/model, not a transparent fused kernel. Source-registered solver families can cover Euler, Heun, DPM-type or learned schedule where exact math/evidence is known.

Speech: an outer 12–25Hz acoustic temporal token and inner multi-codebook RVQ depth loop are two coordinates. Whisper's encoder-decoder audio/text attention is not the same task as CTC/RNN-T/TDT streaming transduction, including blank and beam state. A typed decision head scores finite options with NO free-form token generation.

## MM-04. Multitrack StateRoot, synchronized commit and interruption

Extend StateDomains, preserving **Prepare each → await all physical ready events → one TaskCoordinator PublishRoot**:

| Typed domain | Examples | Valid commit boundary |
|---|---|---|
| TextToken | token history, grammar, sampler, top-k RNG | accepted token or verified prefix |
| AttentionRecurrent | KV, sliding cache, GDN/SSM | model-specific complete step |
| AudioFrontend | resampler/STFT overlap, mel windows, VAD | exact input sample/window offset |
| Acoustic | codec frames, codebook depth, voice/style embedding | complete reconstructible acoustic frame |
| Transcript | text hypotheses, diarization, timestamps | committed transcript segment |
| SolverLatent | image/video latent, timestep, guidance, solver history, RNG | complete denoise/flow iteration |
| TemporalVideo | 3D VAE tiles, latent temporal cache, frame sequence | valid temporal frame window |
| AudioVideoSync | track cursors, lip-sync alignment, ready watermark | declared playable AV sync group |
| TypedDecision | schema, option logits, calibrated probabilities | atomic scored record |
| CrossModelEdge | source-verified stage outputs / ownership | complete typed edge |
| OutputDelivery | stream sequence, retention, credits, receipts | committed packet or chunk |

A scalar StateRoot position is insufficient for multiple clocks. CommitGroup declares which audio samples and video frames become final together; alignment is computed from exact rational clocks, source delay and resampler residual. It may publish one audio segment before video only if the model/stream format explicitly permits that validity, and cannot expose partially denoised frames or incomplete codec frames.

**Full duplex:** simultaneous audio input and output lanes with capture timestamp, playback watermark, VAD/endpointing, incremental ASR, TTS buffer, speaker-turn and interruption epoch. On user barge-in: cancel all uncommitted output after the turn boundary; already played audio cannot be recalled; fence in-flight GPU/codec writes; preserve/replay source-compatible conversation context and release leases only after actual completion. Bound input jitter, queued playback, and all latency/deadline budgets; no blanket hard-real-time guarantee on shared OS.

## MM-05. Denoising, temporal specialization, and task performance

Decompose image/video tasks into **conditioning encoders**, **denoiser/flow predictor**, **registered sampler+guidance**, and **VAE/media reconstruction**. Text/VLM conditioners can be independently resident; image and video latents have ranks and time axes, not fake token identifiers.

- Qwen-Image and SD3.5: MMDiT/DiT latent predictors, text encoder(s), guidance and VAE.
- FLUX.2: source-variant conditioner ([dev] Mistral-class; [klein] Qwen3 text embedding) + rectified flow transformer + independent VAE and multiple image references.
- An autoregressive multimodal image model: described as a unified multimodal autoregressive MoE; its released generate_image path needs audit for diffusion/other stages before choosing a mandatory AR-only or MMDiT-only workflow.
- Wan 2.1: spatiotemporal DiT/3D VAE; Wan 2.2: high/low noise-phase **whole-model expert** selection by denoising step, not an ordinary token-level FFN MoE.
- An autoregressive multimodal video model: 3D causal VAE plus selective/sliding tile sparse attention and optional superresolution. Sparse mask is SEMANTIC; cannot silently replace with a different local window.
- Mochi: asymmetric multimodal attention with non-square projections and separate video VAE.
- Joint audio/video models and similar: jointly conditioned, synchronized audio/video streams; cannot be modeled as independent TTS plus video with an arbitrary merge.

Media performance policy includes latency-to-first-playable-frame/audio, true real-time factor, peak latent scratch, VAE tile size, audio/video synchronization error, completed sample quality, context streaming, denoise step count and bounded device leases. Multi-model resident conditioner+denoiser+codec sharing requires resource budgets and eviction safety across concurrent tasks.

## MM-06. Orthogonal tensor encodings and ternary adversarial compatibility

**Model topology and source tensor encoding must not be conflated.** A ternary 27B model preserves the hybrid Qwen3.8 model architecture, but the PTQ1_0/PQ2_0 original source weights and runtime block Hadamard transformed activations differ fundamentally from Q4/Q6. A decoder recognizing Qwen3.8 attention does NOT imply it can load/execute that ternary model.

Proposed TensorEncoding contract (illustrative):

~~~rust
struct TensorEncoding {
    source_format: TensorSourceFormat,
    physical_codec: WeightCodecId, // Q4_0, PTQ1_0, PQ2_0, FP4...
    block_geometry: BlockShape,
    scale_encoding: ScaleEncoding,
    activation_transform: Option<RegisteredTransformId>,
    activation_codec: Option<ActivationCodecId>,
    accumulator: AccumulationContract,
    semantic_kernel_key: SemanticKernelKey,
}
enum RegisteredTransform {
    Identity, Cast, LayoutPermutation,
    BlockHadamard{group_size:usize,sign_scheme:SignedBasisId},
    PackCodebooks, CodecResample,
}
~~~

g128 ternary weights use group scales, runtime matching Hadamard activation transformation, direct bitpacked GEMV; source mismatch is hard load failure. Allow source-qualified backend kernels that fuse Hadamard+ternary dot without unpacking entire original checkpoint into FP16. Applicable equally to BitNet-style low-bit, FP4, dynamic activation formats, audio codec RVQ, and mixed GEMM quant policies. Current Phase 1 Qwen QuantKind enum is legitimately narrow; extend through a verified codec/transform registry when an actual new source checkpoint is admitted, rather than adding an unbounded trait zoo today.

## MM-07. Typed decision scoring, not JSON generation

Proprietary decision services and Qwen-backed decision models require a ScoreFiniteSchema TaskDriver. Inputs: JSON/text/media state and finite schema of yes/no, choice or score questions. A model returns an option-logit vector per question, with explicit normalization/calibration and schema version:

~~~rust
struct DecisionResult {
    schema: SchemaKey, questions: Vec<QuestionScores>,
    calibration: Option<CalibrationId>, source: ArtifactKey,
}
struct QuestionScores {
    question_id: QuestionId, options: Vec<ChoiceId>,
    logits: Vec<f32>, probabilities: Vec<f32>,
    confidence_semantics: ScoreInterpretation,
}
~~~

A ModernBERT-based decision encoder and a Qwen multimodal backbone+joint schema head are distinct assemblies under one task kind. A proprietary decision service is exposed only via a separately authorized external adapter. Never generate JSON tokens and call it source-equivalent to direct classification scores; never invent calibrated confidence if the model supplies only raw logits. Many questions may be processed jointly in one forward; task routing is allowed to choose a true decision model instead of a bigger text generator when policy/quality support it.

## MM-08. Native versus opaque external models

Hosted image, video and TTS generation services generally require **approved TaskServiceAdapter**, not native SemanticProgram execution, unless source weights/math become legally accessible. These adapters exist ABOVE TaskRequest and have independent credential/egress/privacy controls, asynchronous job polling/webhooks, file/bucket staging, cost accounting, provider model revision, optional C2PA and rights policies. They may participate in a higher application workflow, but are not a DeviceBackend and may never claim to run on local GPU or publish local neural StateRoots.

Keep OpenAI-style audio/image/video/response APIs only where endpoint semantics genuinely align. Use native typed endpoints for decisions, synchronized AV and compound tasks. Unsupported parameters are typed errors, not dropped. Typed output events carry lane, sequence, source timebase, media timestamp, commit group, encoding, finality and payload reference.

## MM-09. Four orthogonal dimensions of specialization

Any candidate closed-region implementation is legal only across these four checked dimensions:

1. **Semantic:** MMDiT, flow, sparse mask, hybrid GDN, AR codec, audio text alignment, attention variants, ternary transform and head semantics.
2. **Physical:** CPU SIMD/CUDA/HIP/Vulkan/Metal/NPU/AOT, tiling, compression, matmul group FMA, graph capture, tensor layout.
3. **Task/control:** encoder, decode1, acoustic codebook loop, CTC beam, denoise step, long temporal window, latency/deadline, typed decision batch.
4. **Placement/resources:** UMA/discrete device, resident conditioner/denoiser/codec, KV/latents/prosody state, staged transfers, remote privacy, memory/IO quotas.

Plan keys also include source checkpoint, semantic/codec version, numeric policy, legal state bridge, resource/event budgets and exact device evidence. A driver supporting only a native kernel or whole AOT graph can still implement a legally closed region; there is no model-name conditional in the core router.

## MM-10. Minimal fake acceptance drills, before feature implementation

These tests are **proposed, not currently implemented**.

- MM-01: 44.1kHz PCM and 29.97 FPS rational clocks, lossless resampler offset and exact AV mux ordering.
- MM-02: verified conditioner→flow predictor→VAE graph, independent device placement/admission, artifact revocation.
- MM-03: 20-step flow/DiT with seed, noise schedule, guidance and partial-step failure; wrong sampler must not pass.
- MM-04: audio-codebook-style 21 Hz slow AR + 10-codebook depth AR and complete codec frame commit.
- MM-05: Whisper encoder-decoder versus Parakeet CTC/RNN-T/TDT transducer; blank and beam state, timestamps, streaming recovery.
- MM-06: full duplex input/output, partial ASR, barge-in, in-flight device completion, bounded latency credits.
- MM-07: joint-audio/video-style synchronized generation with dual-stream attention, different clocks and matched output watermark.
- MM-08: Qwen-backed decision model / ModernBERT-based encoder with 20 typed schema questions in one forward pass, calibrated distribution shape and rejection of invalid choices.
- MM-09: Ternary PTQ1_0/PQ2_0 source bitpack and Hadamard activation transform, exact CPU/physical backend parity, reject missing transform.
- MM-10: fake CPU, native GPU, graph AOT and external service provider on same TaskRequest, no duplicate StateRoot publication.
- MM-11: long media streams with bounded memory, cancellation replay, backpressure, physical lease/event fencing.
- MM-12: support status distinguishes recognized source vs mathematical reference vs CPU running vs per-kernel GPU vs full-model GPU vs real-time qualified vs opaque provider capability.

**Architecture acceptance law:** adding a source-verified new audio codec, TTS generator, video predictor, low-bit tensor format or typed decision head must not require rewriting the central Router, TaskCoordinator, BufferLease/DeviceBackend event authority or HTTP inference runtime. Extend only narrow source model/codec/TaskDriver/region providers.

## MM-11. Implementation order without derailing P07

- P07 GPU native Qwen3.5 specialization proceeds as planned; do not preempt with arbitrary media work.
- MM0: design/reference atlas, typed ValueDescriptor+Timebase+TrackFrontier schemas and fake source fixtures.
- MM1: one non-token task (encoder/typed decision) and one multi-step flow/denoise fake task under the existing coordinator; test effect/failure invariants.
- MM2: one small real ASR (Whisper or Parakeet), Kokoro TTS; then a small open flow/DiT image model such as FLUX.2 Klein where feasible.
- MM3: dual-codebook TTS, Qwen3 Omni, Wan/autoregressive video and synchronized joint-audiovisual generation, based on actual checkpoint/hardware/resources.
- MM4: Full ternary activation transforms + exact source packing and remaining specialist hardware providers.
- Opaque proprietary task adapters can be added independently above native execution, with explicit credential/user-consent/data-egress policies.

The **implementation reality** today is the native Rust Qwen3.5 CPU model, independently passed Phase 1 8K source qualification, and source-correct P07-A partial CubeCL GPU region implementation pending physical tests. **None of the above image/audio/video/decision/ternary models are implemented in Zerogravity today.**

````

## Source: .agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md

Original UTF-8 bytes: 22533; SHA256 `d39bfcaaf68165aeb9c3ac4d053a7332d840cac925dce2562df6b0236a2a284f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — First-Class Virtual Tensor Residency and High-Performance SSD Streaming

**15 Oct 2026 — cache residency × concurrency as one optimization:** [Joint serving and SLO-goodput contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) couples model/source weights, per-session recurrent/paged prefix state, routed expert hit/miss and actual SSD/host IO credits to batch/speculative verification and p95 admission. Prefetch hints never change selected experts or target math. A fully resident compatible kernel avoids per-token host IO; a cache miss legitimately blocks only its source-dependent stage and cannot evict expert slots referenced by encoded/captured/submitted GPU work. Larger MTP draft may increase expert union and lower net goodput even with higher acceptance. Serve transparent source/quant and honest memory/latency QoS receipts; no hidden re-quantization to meet deadlines.


**15 October 2026 · Governing design refinement, NOT implemented support.**
**Architecture:** v4.3 SemanticProgram → ExecutableBundle, verified ModelAssembly and SourceTensorId, one PhysicalResourceArbiter, per-device lease/event authority, one TaskCoordinator StateRoot publisher. Specialized asynchronous IO providers and cache policy clients, but NO additional inference engine.

## 0. User requirement and an exact promise

The user requires SSD streaming to be a first-class, natural capability: an immutable neural tensor can be fully GPU resident, host/unified-memory resident, cached in pinned RAM, file-backed, or asynchronously loaded on demand from SSD **without changing its mathematical identity or model operator**. Models too large to fit in GPU/CPU memory should load only needed experts, lookup rows or source-bound media components under bounded capacity and predictable performance.

**Near-zero-cost abstraction means near-zero software overhead on the fully resident path. It cannot mean zero physical latency for an SSD miss.** The resident fast path has no mandatory host routing readback, global model map lock, per-token filesystem stat, host scheduler wakeup, HTTP/JSON dispatch, dynamic allocation or source IR walk. The miss path pays actual SSD latency, source verification, staging and transfer costs, which are bounded and measured; may overlap safely with earlier layers.

Keep one model source, one tensor view, one legal source kernel and one task result. Physical residency is a property of materialization and lease, never a different model type or a silent quantization policy.

## 1. Independent identities and ownership

| Item | Owner | Exact identity/effect |
|---|---|---|
| SourceTensorId | ArtifactCatalog / verified ModelAssembly | immutable checkpoint or derived-quant SHA, component/tensor role, logical axes, source codec, transform, shape |
| TensorSlice | source tensor descriptor | bounds-checked logical byte/element subview: selected expert, projection, lookup row, image/video tile |
| SourceExtent | bounded read-only file descriptor | pinned artifact+shard ID, validated offset/length/alignment, integrity and source generation |
| ResidentSlot | physical storage manager | physical device/host address, layout, slot generation, readiness and read leases |
| TensorLease | existing BufferLease/event authority | source identity plus stable materialized slot and real dependency/fence until last GPU reader retires |
| Prepared plan | ExecutableBundle/CIR | source-qualified kernel, hit-only or miss-capable path and physical resource dependency graph |
| Mutable TaskSession | TaskCoordinator/StateDomain | KV/recurrent/draft/codec state, causal session epoch, distinct from immutable expert weights |

An immutable tensor can change backing tiers without invalidating its SourceTensorId. Requantizing weights produces a NEW derived artifact with its own numerical/quality evidence; reordering a physical layout without changing values is a separately verified lossless packing transform. A frozen session snapshot must carry state-format and session-compatibility identity, not reuse the immutable expert key.

### Illustrative Rust vocabulary — conceptual, not a frozen public ABI

~~~rust
struct TensorKey {
    artifact: VerifiedArtifactId,
    component: ComponentId,
    tensor: TensorId,
    source_codec: SourceCodecId,
    transform: Option<VerifiedTransformId>,
    geometry: TensorGeometry,
}
struct TensorView { key: TensorKey, offset: u64, size: u64, layout: LayoutId }
struct SourceExtent { artifact: VerifiedArtifactId, shard: ShardId,
    offset: u64, length: u64, alignment: u32, integrity: ExtentDigest }
enum BackingTier {
    DeviceLocal(DeviceId), UnifiedMapped(PhysicalPoolId),
    HostPinned(HostPoolId), HostPageCache(FileId), LocalSsd(SourceExtent),
}
struct ResidentSlot {
    source: TensorViewId, allocation: PhysicalAllocationId,
    generation: u64, layout: PhysicalLayoutId,
    upload_event: DeviceCompletionEvent,
    readers: PhysicalReaderLeaseSet,
}
struct ReadyTensorLease { slot: SlotId, generation: u64,
    source: TensorViewId, event: VerifiedCompletionId }
~~~

Do not implement a trait-object indirection per matrix element or media frame. These concepts are compiled/instantiated at source-loading/stage-admission boundaries, and become stable GPU-native descriptors plus already-held leases during hot execution.

## 2. Source-preserving fast and miss paths

~~~text
Exact model + immutable TensorView
      │
Precompiled source-equivalent region / legal KernelCandidate
      │
Validated resident slot directory / device pointer table
      ├── READY HIT
      │     Same GPU pointer or mapped memory address
      │     Existing generation/read lease, no OS IO
      │     Kernel launch → real event → retire lease
      │
      └── MISSING EXTENT
            Coalesced bounded request to source materializer
            Verified local SSD / host-cache read
            Exact length, source digest and codec verification
            Host staging or direct compatible mapped memory
            Async H2D transfer / verified event
            New resident slot table generation published
            Same source-equivalent kernel → real event
            Release only after every reader retires
      │
TaskCoordinator independently verifies causal state and publishes ONE StateRoot
~~~

**Resident hits:** ideally direct device address, validated source table ID or precompiled device-resident mapping. An SSD service thread can sleep. No per-layer synchronous CPU route ID copy or mandatory host callback. A prepared all-resident graph entirely omits the IO branch.

**Misses:** GPU routing may first generate real selected expert IDs into a compact device-visible list; if a missing ID occurs, yield at a validated graph-island boundary, post a bounded request ring, and resume when actual host IO and GPU DMA Ready. No unbounded GPU busy-spin or uncaught page fault. Predicted routes are hints only; routing cannot choose a different expert because one is cached.

A captured GPU graph must not hold an old raw slot pointer after reuse. Either keep all captured pointers pinned for the graph's full lifetime or dereference a correctly ordered generation-tagged device table. If capture cannot safely cope with misses, select a precompiled miss-capable eager island; no dynamic kernel compilation per token.

## 3. One residency fabric, domain-specific allocation policies

| Source data class | Default candidate tier and cost | Safe policy |
|---|---|---|
| Dense trunk, routers, layer norms, attention/shared projections | GPU/UMA when feasible; used by every step | keep resident first; slow staged whole-layer reload is opt-in capacity mode |
| Routed MoE expert matrices | SSD source + bounded GPU/host expert slab | demand selected experts, per-layout size class and adaptive cache |
| Gigantic Qwen n-gram or DeepSeek Engram table | immutable SSD file and OS page cache | tiny indexed row cache, dedup/coalesce reads; never upload all |
| Tokenizer/critical small sidecars | host/GPU resident | avoid predictable per-token SSD latency |
| Trained MTP/DFlash/vision encoders | workload-enabled component | stage residency by measured benefit; independently budget auxiliary state |
| Image/video VAE, diffusion predictors, upscalers | GPU while active, host/SSD between stages | source-verified stage/tile streaming; respect temporal overlap |
| Input audio/video files | bounded timed decoded buffers and codec IO | source media chunks, not MoE slab policy |
| Immutable committed prefix state | GPU while active; RAM/disk snapshot | tenant-private, source-compatible restart/recompute |
| Mutable active KV/GDN/KDA/mHC/codec state | device-local current step | never evict unconfirmed GPU state based only on disk write |

Different lifetimes require different **cache policies**, not separate engines: RoutedExpertCache, LookupRowCache, FrozenStateStore and MediaTileStore use the same SourceExtent, event/lease and global physical budget, but own specialized eviction and access patterns. One universal LRU for all is incorrect.

## 4. Expert source and cache: lifecycle and data layout

Key immutable expert by (model artifact and derived quant revision, layer, expert ID, gate/up/down projection group, tensor codec, physical layout). Source GGUF shards are opened by verified file descriptor with bounded exact offsets and independent source integrity, not by mutable pathname.

Expert residency lifecycle:

~~~text
Cold → Requested → Reading → Verified → Uploading/Mapping
     → READY published (generation increment)
     → Captured/Encoded reader leases
     → Queued/submitted/physical GPU readers
     → Complete (all device events retired)
     → Idle/Evictable → reused for another source, generation increment
~~~

Separate IO and GPU transfer events: SSD completion does not imply GPU upload Ready, and upload Ready does not imply all future readers quiesced.

### Mandatory correctness: lease includes ENCODED but UNEXECUTED GPU work

DS4 issue #384 isolated wrong logits under SSD expert cache pressure: an early layer's GPU command buffer was **encoded** referring to a slot; while encoding a later layer, the cache evicted/reused that slot; GPU executed the first command and read the *later* expert's bytes. The pointer was valid and CPU cache state internally consistent, so no hardware fault occurred.

**Zerogravity requirement: a slot becomes pinned as soon as a GPU command/captured graph records its reference, and stays protected until every possibly reading submitted command genuinely completes.** Merely protecting the currently encoding layer's experts is unsafe.

- All graph/batch commands pin their referenced slots across entire unexecuted, queued and submitted lifetime.
- If no safe victim, wait/flush at a legal graph island boundary, reduce staging/parallelism, use another safe backing tier or refuse with backpressure.
- No slot reuse until physical completion and generation checking; pointer-table publish uses device acquire/release ordering and actual Ready.
- Graph replay must revalidate slot generations or keep stable addresses pinned through replay.
- Unknown hardware completion means entire actual slot/physical handles quarantined; no timed release.

## 5. Mixed expert sizes and one shared byte budget

Alternating Q2/Q4/FP8/derived-trellis layers may store expert gate/down matrices with different sizes and layouts. DS4 issue #503 found a single-size CUDA cache repeatedly cleared its entire hot set on alternating mixed-quant expert sizes; a separate prefill path also threw away hot decode cache on short user prompts. The issue additionally warns that per-size-class count limits can collectively exceed real memory.

Use:
- small bounded size-class pools keyed by exact codec/shape, **one authoritative total byte reservation across ALL classes**;
- admission covers required scratch, pending staging, session KV and model nonrouted weights first;
- low-traffic class must not collapse or globally clear another class;
- prefill may use temporary independent staging or fill-only cache behavior, with evictions only after actual last reader;
- decode may access already ready slot directly, avoiding whole-expert device-to-device compact copy for every hit where legal and useful.

A high cache hit rate while repeatedly copying 2GB/token between device buffers is NOT low overhead. Test path with direct resident pointer and explicit copy path to evaluate correctness and speed.

## 6. Multi-tier IO implementation and platform reality

Default logical tiers: device-local or unified mapped, bounded pinned/hot host RAM, OS buffered file cache, SSD immutable source. Any intermediate tier can be absent. A host RAM second-level expert cache is useful when random O_DIRECT SSD access bypasses page cache; DS4 issue #629 discusses such a tier as proposed work, NOT necessarily merged production behavior.

IO provider choices (not runtime model semantics):
- Linux io_uring with alignment/capability validation and bounded submissions; fallback bounded pread worker pool.
- Linux O_DIRECT where alignment/device/filesystem and measured read size justify it; explicit compatible buffered fallback on Unsupported/invalid alignment.
- macOS dispatch IO/pread/mmap/Metal mapping under its own event and residency contract.
- Windows overlapped IO/IOCP or mapped file with exact lifetime and error handling.
- CUDA GPUDirect Storage/vendor direct SSD→GPU only after actual backend proof; no assumption that page-cache-free implies free GPU transfer.
- UMA mapped CPU/GPU handles may share one DRAM physical allocation (single physical byte charge), but contention and cache coherence remain real.
- Discrete GPU consumes host staging and PCIe/H2D budget; pinned cache doubles physical copies and cannot be called zero-copy.

Read tasks carry source offset, exact length, integrity, owner/tenant/source generation, cancel epoch, output buffer lease and deadline. Partial/torn reads and corrupted tensors never publish Ready. IO cancel cannot reclaim pinned bytes until the IO operation actually retired. Coalesce identical immutable source requests across model sessions (per-source singleflight) without sharing mutable state.

## 7. Prefill, decoding, and speculative expert reuse

### Prefill policy
Gather source-selected unique experts by layer across many token rows, coalesce reads, overlap future layers where dependencies permit, and protect existing hot decode slots. Large prefill scratch can borrow **idle** optional cache only after real event retirement. Small subsequent user turns must not indiscriminately flush all warmed expert cache. Consider shrinking legal prefill chunk rather than displacing mandatory dense weights/KV.

### Decode policy
The random expert miss often lies on critical path; keep route-hot experts warm, cost next-layer lookahead, and favor short bounded IO queues. Speculative decoding may increase distinct routed expert IDs and total candidate rows, so speculation width and batch depth must include miss probability, cache churn and storage-transfer cost, not merely model FLOPs. Prefetch hints never change mathematical router selection.

### Dynamic cache policies
- Full residency is special case with zero misses: no host IO or source-materialization callbacks in hot path.
- Across-layer dynamic hot expert cache with LFU/recency/byte cost under remaining budget.
- Fully resident prefix layers plus dynamic later experts when source stage/cost warrants. This is policy, not model type.
- Per-task adaptive cache partition/cold experts with bounded hysteresis; no unsafe eviction under queued GPU graph.
- May use host RAM as second tier, or keep source file page cache warm where OS headroom supports it.

Always reserve nonrouted trunk + mandatory state/OS health reserve before optional expert-cache bytes. Larger SSD cache is not universally faster when it displaces dense trunk/KV and forces expensive mandatory reload.

## 8. Four distinct uses of SSD under one substrate

**Routed weights:** cache expert slabs and materialize missing routed matrices as real device buffers. DS4 supports selected models/backends and cache sizes; it explicitly does NOT promise every arbitrary GGUF.

**Indexed giant lookup:** A specialized engine's large Qwen FP8 n-gram table and DS4 Qwen PLE/DeepSeek Engram table remain disk-file-backed and return only required source-registered hash rows. Use verified row layout, dedup, batched coalesced reads, OS page-cache or bounded row RAM cache. No full-table GPU residency expected.

**Frozen committed session/KV:** DS4/specialized-engine disk KV/prompt caches store source-compatible token/session state for reconnect or restart. Snapshot is mutable-derived private state with epoch/COW/compatibility and checksum, not just expert bytes; restore may require significant SSD→GPU transfer or source-compatible replay. Durable write can be asynchronous behind a committed prefix and must not block ordinary output unless durability was requested.

**Unpinned dense trunk:** One specialized engine documents disabling trunk pinning to free memory, trading substantial decode speed for memory fit and potentially 7–50s additional cold prompt reload. Treat as an explicit capacity/latency tradeoff, not a default "equally fast" tier mode.

These patterns share IO buffers, artifact identities, resource accounting and event contracts, but use distinct policies. The source model architecture stays independent.

## 9. Memory budgets and source performance cost

Physical capacity before loading:
mandatory dense/shared weights + Qwen/GLM/MiMo source state + active sessions KV/GDN/MLA/SWA + draft/vision/audio + graph/code + codec latents + output and scratch + OS/cgroup health reserve + pinned staging ring + optional expert/row caches + expected page-cache pressure. For UMA, GPU and CPU aliases to the same DRAM charged once physically, with separate budget policy; GPU virtual address max != safe RAM.

A cache count N cannot be applied independently to every expert size class. Global byte reservation and staging peaks must prevent physical driver/OOM hangs.

Latency model:
T_miss = queue + random SSD seek/IO + length/effective throughput + verification + optional dequant + host staging/mapping + GPU transfer and event wait + eviction/repair - **measured and legal overlap**.
T_hit = existing device pointer lookup + GPU compute, with no mandatory per-layer host thread.
Whole task cost is the dependency-graph **critical path**, not sum of nominal streams; expose effective hit/miss per layer and actual transfer/stall.

The scheduler should prefer fully resident when feasible automatically, then pick legal streamed placement when memory isn't sufficient, with truthful ModeReceipt: resident vs streamed ratio, effective byte budget, predicted/observed latency/QoS, cache/SSD fallback. If model will fit but be very slow, explicitly disclose; don't auto-change original quant or model identity.

## 10. Reliability, error handling and cancellation

| Event | Required result |
|---|---|
| Source extent/hash missing or invalid | pre-submission refuse, no false device fence |
| SSD short/corrupt read | cancel dependent stage, no published slot/StateRoot |
| O_DIRECT alignment or unsupported filesystem | measured buffered fallback or typed Unsupported |
| Cache full, every slot pinned in encoded GPU batch | legal wait/flush/backpressure or refuse, never overwrite |
| Pointer table ABA/reuse | generation/read lease prevents old graph seeing new expert |
| Wrong model/codec layout | exact model-specific Unsupported, no implicit requantization |
| Transfer partially submitted, canceled | retain staging/device buffer until actual completion or quarantine |
| GPU command reads streamed slot then canceled | physical lease stays until genuine quiescence |
| Slow SSD / rising p95 / OS page-cache thrash | backpressure/admission, optional quality-declared QoS downgrade only |
| Mixed quant source | byte-class-specific pool and one global budget, no cache reset |
| Slow client after output commit | bounded transport credits, no forced device buffer release |
| Disk snapshot incompatible with model/tokenizer/tenant | fail secure or authorized recompute, no foreign KV |

## 11. Verification and zero-overhead discipline

Target minimal implementation seams:
1. Verified source extent description in existing ArtifactCatalog, preserving open-descriptor lifetime and hashes.
2. Bounded materialization Future/result compatible with existing BufferLease and typed actual IO/device events, not another executor.
3. Compact physical resident pointer/generation table under existing DeviceWorker; no per-element trait objects.
4. Domain-local cache policies (expert slab, lookup rows, immutable encoder/latent, frozen state) with one global resource reservation.
5. Prepared resident-only and bounded miss-capable command variants; no dynamic IR interpretation per token.
6. Existing TaskCoordinator publishes root only after all actual IO+DMA+compute dependencies and source algorithms are proven.

Test ID SSD-01..SSD-32 in companion plan. No model/format/backend can advertise SSD streaming just because an arbitrary file path is present. Physical source test must include original trained model and full token/state parity in both fully resident and deliberately constrained cache cases.

## 12. Source and provenance references (reviewed 2026-10-15)

- DS4 SSD streaming: https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md
- DS4 SSD controller: https://github.com/antirez/ds4/blob/main/ds4_ssd.c
- DS4 GPU reader lifetime bug: https://github.com/antirez/ds4/issues/384
- DS4 mixed quant/cache clearing: https://github.com/antirez/ds4/issues/503
- DS4 proposed host RAM cache tier: https://github.com/antirez/ds4/issues/629
- DS4 session/KV serving: https://github.com/antirez/ds4/blob/main/docs/SERVER.md
- Specialized-engine resident trunk, lookup and disk KV (external reference removed during attribution cleanup).
- Specialized-engine detailed config (external reference removed during attribution cleanup).
- Specialized-engine quant/lookup (external reference removed during attribution cleanup).

All numerical figures and bug reports in companion research are original author/issue evidence, NOT native Zerogravity SSD performance or implementation proof. **Specified first-class architecture does not mean already implemented SSD execution.**

````

## Source: .agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md

Original UTF-8 bytes: 28589; SHA256 `4b2e7195c28e6d4813dae9da5bc07b4b9080388295d8b6bf6b5458e38c790637`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity Universal Multimodal Execution Architecture — v5.1 Design Specification

**Date:** 15 October 2026 (Asia/Singapore)  
**Status:** Comprehensive, implementation-ready DESIGN PROPOSAL; no new runtime/code/hardware qualification.  
**Lineage:** v4.3 core execution authority + additive 14 October v5 multimodal proposal + 15 October modality/real-time research.  
**Authority:** This document is the preferred design for newly considered multimodal scope; the user explicitly requested design and documentation, **not a unilateral final approval of every illustrative ABI choice**. Where conflicts exist, preserve current user decisions and the v4.3 single-owner invariants.  
**Companions:** REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md, VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md, the detailed research tracks under .agents/plans/multimodal-research, and MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md.

## 0. Nonnegotiable goals and non-goals

### Goals
1. A **single Rust-authored execution fabric** that understands verified mathematical semantics rather than guessing algorithms from model family names.
2. Text generation, text embedding, ranking, finite schema decisions, images, image editing, audio ASR/translation, TTS, full-duplex speech, video, synchronized AV, native multimodal perception and multimodel compositions all representable using one small set of contracts.
3. Per-model and per-stage **best-legal specialization** on CPU SIMD, CUDA, ROCm/HIP, Vulkan, Metal, suitable NPUs, static-dataflow/AOT accelerators and allowed remote peers, subject to source math, numerical and physical resource proofs.
4. Streaming with independent clocks and bounded backpressure; explicit customer-facing distinction between prepared, committed, delivered and played media.
5. One process may register many distinct models, load many within physical quotas, share immutable component weights and execute independent mutable TaskSessions without leaking source/model or user state.
6. Portable CPU mathematical reference path wherever a model is claimed runnable; unsupported source operations are typed refusal, not silently delegated to unrelated inference products.
7. Native built-in model router and unified CLI/SDK/HTTP/realtime/job serving; an adapter-specific API protocol MUST NOT become an execution engine.

### Non-goals
- No automatic claim to reproduce proprietary model internals: GPT-Realtime, Gemini Live, hosted image/video/TTS generation APIs and proprietary decision-service APIs are opaque external services unless verified locally executable sources become available.
- No replacing all media tasks with the same token-AR loop, and no equating CTC, RNN-T, diffusion, rectified flow, codec RVQ and direct decision heads.
- No four independently serialized intermediate representations. Stable semantic source program and prepared executable bundle remain the two durable identities; region/kernel/command views stay compiler-internal.
- No general user-programmable agent workflow DSL in a kernel worker. Privileged retrieval/tools/action execution belongs in a separately permissioned application orchestrator.
- No implicit remote credential forwarding, unknown checkpoint code execution, downloaded model weights, vendor runtime installation, networking or services as a side effect of reading a checkpoint.
- No forced synchronous CPU roundtrip per token, acoustic frame, video frame or denoising microstep; prepared Device-native step islands may advance locally within validated effect/budget bounds.
- No target model substitution, changed quantization, sparse attention, solver schedule, codec resampling, guidance or answer-calibration without a separately named numerical/semantic policy and evidence.

## 1. Architectural reconciliation and ownership boundaries

**Keep the v4.3 vocabulary**, extended conservatively:

| Authority | Owns | Must never own |
|---|---|---|
| ModelCatalog / immutable ArtifactManifest | exact checkpoint files, hashes, tokenizer/media processors, source semantics, artifact versions, license and trusted components | user sessions, networking, GPU command readiness |
| ModelDefinition + ModelAssembly | verified source component graph, tensor roles, port schemas, exact mathematical definitions and codec/transform identities | ad-hoc family-specific HTTP server |
| SemanticProgram (SIR) | semantic ops, tensor axes, modalities, mathematical equivalence, numerical constraints and state effects | driver pointers, sockets, mode routing |
| TaskDriver | coarse task-specific stage order, algorithm, iteration, checkpoints, output validity and stop conditions | physical allocator, remote authentication, general user workflows |
| RegionExecutionProvider / DeviceBackend | semantically closed and source-qualified CPU/GPU/NPU/AOT region candidates; physical buffers, command stream, completion events | StateRoot publication, automatic model substitution |
| ExecutableBundle | immutable prepared region candidates, guarded shape/codec/numeric variants and event/lease command templates | mutable session state or per-packet model search |
| PhysicalResourceArbiter / DeviceWorker | coarse global physical admission and fast local bounded allocations, deadlines/fairness and real command retirement | semantic model policy, fake ready events |
| TaskCoordinator | one logical writer per TaskSession or authorized shard; state root publication, media commit groups, epoch and output-order authority | raw driver reclamation before hardware quiescence |
| DeliverySession / PresentationLedger | transport output sequence, packet ACK, playback evidence, reconnect/replay cursors | altering neural model state; claiming ACK means listened/viewed |
| ModelRouter | exact model binding, task capability, replica/placement selection at safe boundaries | hidden hot-path per-token scheduling, hidden degradation |
| TaskServiceAdapter | opt-in provider API with async jobs, privacy, credential and billing policy | local DeviceBackend and claims of native model math |
| Client protocol adapter | external OpenAI/Anthropic/native/WebSocket/WebRTC/SIP semantics | model-specific numerical definitions, StateRoot writes |

A set of logical authorities does **not** require separate OS processes or crates. Initial implementation may be cohesive Rust modules and a few small traits. No universal abstract object per PCM sample or per GPU op.

## 2. Type-safe multimodal values

The existing v5 ValueDescriptor is adopted with explicit validation and ownership fields.

~~~rust
// Illustrative type-level design, NOT compiled Rust/API compatibility promise.
enum ValueDescriptor {
    Tensor(TensorDesc),             // logical rank/axes/layout/dtype/shape
    TokenSequence(TokenDesc),       // text/image/acoustic/interleaved units
    CodebookSequence(RvqDesc),      // time index × depth index, completion policy
    AudioPcm(PcmDesc),              // rate, integer/float format, channels
    AudioFeatures(FeatureDesc),     // mel/STFT embeddings, windows/overlap state
    VideoFrames(VideoFrameDesc),    // dimensions, pixel/color space, PTS/DTS
    ImagePixels(ImageDesc),         // crop, alpha, ICC, color space, gamma
    MediaLatents(LatentDesc),       // spatial/temporal axes and VAE identity
    EncodedPackets(EncodedStreamDesc), // codec config / framing / extradata
    Transcript(TranscriptDesc),     // speaker/timing/confidence/finality
    DecisionScores(ChoiceSchemaId), // logits and calibration identity
    Structured(SchemaId),          // strict validated result shape
}
struct RationalRate { numerator: NonZeroU64, denominator: NonZeroU64 }
struct TimePoint { tick: u64, domain: ClockDomainId }
struct TimedValue { value: ValueRef, capture: Option<TimePoint>,
                    presentation: Option<TimePoint>, duration: Option<TimePoint> }
~~~

**Semantic invariants:**
- A text token count is never an audio sample count; acoustic codebook depth is neither audio time nor video frame number.
- Clock domain MUST name source origin, epoch, rational rate, optional delay and resampler rounding residue. Use checked u128 intermediate products or equivalently proven exact arithmetic; no cumulative floating point drift.
- Audio sample rates are source-specific (e.g. 16k, 24k, 44.1k, 48k), with validated PCM bit depth, endian and channel ordering. Codec packet timestamps have an independent rate.
- Video 24000/1001 and 30000/1001 PTS values require exact rational comparisons with audio timestamps and explicit continuity/discontinuity markers.
- Color conversion preserves declared pixel format, primaries, transfer curve, matrix, range and alpha handling; latent VAE channels and temporal compression MUST carry verified decoder identity.
- A latent is not a playable image/frame. An incomplete RVQ depth group is not playable audio. A partial/stale transcript is not equivalent to a committed transcript.
- Tensor dimensions are bounded and checked for rank, strides, byte multiplication and allocation pressure before host/device submission.
- Value references carry a lease/lifetime and immutable content identity; packets carry transport-specific delivery references without changing underlying model meaning.

## 3. ModelAssembly: physically separable, semantically unified

~~~rust
struct ModelAssembly {
    assembly_key: VerifiedAssemblyId,       // manifest hash and semantic revision
    components: Vec<VerifiedComponent>,     // each hash/license/config/weights pinned
    typed_edges: Vec<TypedPortEdge>,        // validated descriptor and clocks
    capabilities: TaskCapabilitySet,        // truthful per mode/transport
}
struct VerifiedComponent {
    role: ComponentRole,                     // model-specific, not global name heuristic
    artifact: VerifiedArtifactId,
    source_adapter: SemanticAdapterId,
    processor: Option<VerifiedProcessorId>,
    state_schema: ComponentStateSchema,
    inputs: Vec<PortSchema>, outputs: Vec<PortSchema>,
    source_license: LicenseAndAccessReceipt,
}
struct TypedPortEdge {
    source: PortId, destination: PortId,
    value: ValueDescriptor,
    transform: Option<VerifiedTransformId>,
    effect: StateEffectContract,
}
~~~

A component may be a text/vision/audio encoder, tokenizer, semantic decoder, expert bank, flow predictor, latent decoder, G2P, VAE, neural codec, vocoder, schema head, temporal synchronizer, refiner or upscaler. The list is not a promise of implemented kernels. Treat an assembly as **one exact source model** when its author defines components jointly, even if placed across separate devices.

**Not the same as:**
- a group of replicas serving the same logical model;
- a cross-model workflow combining separately registered ASR → LLM → TTS;
- user action/tool orchestration outside the inference engine.

Per-edge legal conversions are explicit: cast/layout, VAE encode/decode, resample with delay/remainder, tokenizer, feature extraction, spatial resize, normalized schema conversion. Unknown conversions are rejected, never guessed from matching tensor widths.

Assembly invariants:
1. Validate the complete component manifest and license against requested execution. Do not assume checkpoint directory structure implies semantic graph.
2. Version every processor/tokenizer/G2P/codec/vocoder independently; immutable assembly identity binds all required revisions.
3. Allow component-level residency and placement without splitting logical state publication.
4. Each source role has a unique verified model+component+adapter key; don't confuse a shared conditioner with a replica.
5. When sharing an immutable encoder across different models, allow only if exact source weights, media processor semantics and numerical policy are identical.
6. Treat captioner, prompt rewriter or safety model as explicitly optional/required assembly components, not hidden side effects.

## 4. TaskDriver families and termination

A TaskDriver describes algorithmic **control**, not device kernel code. Candidate task families:

| Driver | Main stages | Critical state | Valid output frontier |
|---|---|---|---|
| CausalText | tokenize, prefill, decode, verify/speculate, sample | KV, SSM/GDN, grammar, RNG, history | accepted token/text boundary |
| EncodePoolRank | frontend, encoder, pooling/reranker | source attention, normalization, batch | vector/score with schema |
| WhisperASR | STFT, mel, encoder, AR text decoder, merge windows | acoustic overlap, prompt/task headers, timestamps | committed transcript window |
| CtcSpeech | feature frontend, encoder, CTC collapse + align | blank/beam/timestamp state | stabilized transcript segment |
| TransducerSpeech | streaming FastConformer, predictor/joiner or TDT duration | per-chunk acoustic and decoder caches | emitted hypothesis with revision/finality |
| DeterministicTTS | phonemes, align/duration/pitch, acoustic network, vocoder | speaker embedding, align, vocoder overlap | complete PCM/encoded chunk |
| CodecAutoregressiveTTS | slow temporal codebook AR, fast RVQ depth, codec | time × codebook state, prosody/style, codec | complete reconstructible codec group |
| AcousticFlowTTS | text→acoustic context→flow predictor→codec | solver RNG/timestep, chunk overlap | complete decoded acoustic chunk |
| DuplexAudio | input acoustic encoder and simultaneous text/output codec lanes | capture, playback, turn epochs, echo/barge-in | model+playback-valid chunk |
| LatentImageFlow | conditioners, seeded latent, solver/guidance, VAE | timestep, noise/CFG RNG, latents | completed decoded image |
| ImageARCodec | interleaved multimodal AR, image codebook, decoder | token/codec positions and verification | complete frame/image unit |
| TemporalVideo | spatial-temporal latents, solver, 3D VAE, mux | tile/temporal cache, frame cadence | valid ordered frames/segments |
| JointAudioVideo | dual coupled latents and synchronized solver/codec | independent clocks and solver state, A/V barrier | legal AV commit group |
| TypedDecisions | encode, finite-choice logits/head, calibrate if justified | question/option schema and normalization | atomic answer set |
| ExternalTask | request serialization, async provider job, receipt | job/retry/idempotency, privacy, cost | provider-verified result |

**Algorithm-specific mandatory details:**
- CTC: blank semantics, collapse rules, repetition, alignment and optional LM fusion.
- RNN-T: predictor/joiner state, hypotheses, beam semantics; TDT: duration outputs and timestamp contracts. Must NOT be normalized to Whisper token decode.
- Neural speech codec: frame rate versus RVQ depth, source-trained quantizers, delay, overlap, packet reconstruction and voice/speaker conditioning.
- Rectified flow vs diffusion: predictor parameterization (epsilon, velocity, x0), sigma/timestep schedule, solver integrator, guidance/CFG/STG, seed/Philox-like RNG identity, count, step ordering and precision. Altering these changes the model task.
- Video temporal: sparse tile/selective masks and 3D RoPE/lattice, temporal VAE receptive field, tile overlap, frame latency, super-resolution branches.
- Decision: closed schema, raw option logits and only genuinely calibrated probabilities, joint multi-question inference; never invent probability calibration or simulate with free-text JSON.
- Compound multimodal: known bounded task joins inside trusted TaskDriver; no implicit arbitrary tool execution.

## 5. StateRoot, multitrack commit and physical event authority

Keep exactly one TaskCoordinator-authorized PublishRoot per TaskSession epoch (or explicitly authorized state shard). StateDomain implementations PREPARE versions only. An audio writer, codec decoder, transport thread, DeviceWorker and remote peer cannot independently publish the model's logical session state.

TaskFrontier extends from scalar token position to a vector:
- text accepted-token cursor and grammar/sampler/hidden history,
- recurrent/KV and prefix/branch version,
- input audio sample cursor and frontend overlap,
- output acoustic frame and **RVQ depth completeness**,
- ASR hypothesis revision and stable transcript segment,
- image/video solver iteration and latent version,
- video temporal tile and frame PTS,
- audio/video decode progress and playable commit-group watermark,
- typed-decision schema generation,
- external stage committed result identity.

~~~text
Committed Session StateRoot
  -> provisional sub-steps/regions, potentially on many devices
  -> physical events prove each submitted dependency completed
  -> typed StateDomains prepare compatible versions
  -> model/task algorithm validates complete output boundary
  -> TaskCoordinator compares epoch + expected predecessor root
  -> ONE PublishRoot
  -> committed output sequence becomes eligible for transport
~~~

**Do not conflate logical commit with physical RAM/VRAM DMA completion or with successful client delivery.** Physical handles remain leased until genuine readback/event retirement. Unknown cancellation quarantines indefinitely until proved device quiescence; the global device card need not be canceled because one session requests barge-in if its other sessions are still active.

Synchronized A/V can publish a joint legal group only when its declared audio and video track watermarks, solver iteration, codec state and timestamps are compatible. Permit legitimately independent output lanes only when the exact model/protocol contract permits them. Partial video denoise latents or incomplete codec codebooks may never be labeled final media.

## 6. Numerical/specialization legality dimensions

Specialization is orthogonal across **FOUR** axes:
1. **Semantic**: attention variant, GDN/SSM, transducer predictor/blank, diffusion/flow parameterization, sparse temporal mask, autoregressive codec, multi-head schema scoring.
2. **Physical**: CPU, CUDA/HIP/Vulkan/Metal/NPU/AOT; tensor dtype/packing, FP64/FP32 reduction, tile/warp subgroup, captured graph, PCI fabric, stream/event scheduling.
3. **Task control**: prefill/decode, RVQ frame/depth, CTC beam, flow solver step, video window, low-latency bidirectional audio, decision batch.
4. **Placement/QoS**: assembly component residency, async transfer, memory cap, calibrated soft deadlines, fairness, remote consent/data locality.

Candidate legality key (conceptual): artifact+assembly revision, semantic graph/op hash, exact tensor codec and activation transform, numerical policy, shape/stride and state-effect schema, precision reduction order, device driver/compiler/ISA, resource and event plan, required request QoS and verification EvidenceId.

Kernels can fuse or replace a closed region ONLY if source semantics/effects/numeric contract and guarded shape match. An unqualified model or media transform returns a typed unsupported/failed operation, not silent approximation. For user-selected approximate quality modes, report provenance and measurable distribution/quality differences; never label those equivalent.

## 7. TensorEncoding and novel low-bit models

The stored weight representation does not define model topology. Example: a ternary 27B model maintains a Qwen-derived hybrid model architecture but needs PTQ1_0/PQ2_0 packed ternary weights, 128-group scales and matching 1024-block Hadamard rotation on runtime activation. Loading only its Qwen topology and treating these formats as regular Q2 is incorrect.

~~~rust
struct TensorEncoding {
    source_format: VerifiedFormatId,
    weight_codec: WeightCodecId,
    block: BlockGeometry,
    scale: ScaleCodecId,
    activation_transform: Option<RegisteredTransformId>,
    activation_quant: ActivationCodecId,
    accumulation: NumericReductionContract,
}
~~~

Codec/transform/shape/source identity are all verified at load and plan selection. A backend can execute packed ternary dot directly; it need not expand to FP16 weights merely because reference semantics are expressed independent of packing. Equally applicable to FP4/NVFP4, indexed memory, RVQ audio embeddings and future quant families, but no blind numerical fallback.

## 8. Media component placement, resource accounting, and scheduling

One global physical arbiter and device-local bounded subpools allocate weights, state, H2D/D2H, pinned codec buffers, attention/video latents, replay snapshots and queues. Global admission is coarse and infrequent; per-step stage scheduling uses prepaid envelope leases without a global lock or snapshot allocation per audio frame. Kernel completion and actual driver retirement authorize physical reuse.

**QoS kinds**:
- RealtimeSoft: bounded deadlines, jitter, first audible token/codec chunk, short device quanta, request preemption at validated safe boundaries, explicit admission refusal if no budget.
- Interactive: chat, decisions, short synthesis, prioritized low queue time.
- StreamProgressive: bulk ASR, long TTS, progressive valid output and bounded buffers.
- BatchMedia: images/videos, throughput and joules/quality, denoise-step scheduling, VAE tiling and offload.
- Background: prefetch/tuning/quantization only when non-interfering.

No preemption inside an unconfirmed physical kernel unless backend has proven physical cancellation and state rollback. Deadline misses are observable, not grounds to mutate model semantics. Adaptive placement must preserve the client's exact artifact/numerical/source license, unless a verified policy explicitly permits substitution and records it.

## 9. Distinct stages of composition

A. **Intra-model assembly**: FLUX [dev] conditioner → flow DiT → VAE; audio temporal/depth/vocoder; Qwen-Omni Thinker/Talker; a coupled A/V model. These are one verified assembly where source declares joint weights/state.

B. **Multi-model inference task**: Whisper → Qwen → Kokoro; embedding+rerank+decision; independent image enhancer plus video generator. Separate RegisteredModel identities and output tokens/leases are joined by a trusted bounded task policy or external orchestrator. Each stage must declare exact ports and privacy budget.

C. **Application/agent workflow**: retrieval, tool calls, approvals, external API, business decisions. Authorization/side effects remain above TaskRequest, not inside a GPU kernel or arbitrary TaskDriver callback. Device/computation may remain in the same native engine.

## 10. Artifact/model capability status and truthful UX

A catalog must distinguish:
1. Recognized name or format,
2. complete verified semantic recipe,
3. runnable exact native CPU reference,
4. source-qualified native GPU subregion(s),
5. complete original model/task qualified on device,
6. streaming/playable result certified,
7. soft-realtime objective measured under declared load,
8. external hosted provider capability with auth/consent.
A model card may simultaneously have several statuses by task, variant and backend. Never infer full model support from a successful Q4 matrix projection or a v5 mock.

## 11. Failure classes and recovery

| Error | Source action | Physical resource | Public behavior |
|---|---|---|---|
| Unsupported semantic op/codec/shape | fail or choose proven equivalent reference | no launch | precise Unsupported response |
| Malformed source/input/clock/schema | refuse before submit | no device fence | 400-like validation error |
| Insufficient memory/credits | reject/admit smaller declared batch | no unsafe allocation | typed admission error with available safe options |
| Future canceled before device submission | discard queue ticket and provisional state | no device fence | canceled request |
| Future canceled after possible physical command | mark unknown, quarantine buffers/leases | fence affected physical service until proven quiescent | no false Ready, no stale publication |
| Model numerical non-equivalence | source-grounded failure, no silent epsilon increase | safe completed buffers eventually retire | unsupported/quality-gate failure |
| Transport disconnect | retain bounded committed output cursor, cancel or pause by session policy | never assume client received media | reconnect policy / explicit terminal event |
| Playback interruption | suppress unplayed queued output and reconcile actual playback evidence | do not cancel other sessions/card globally | truncation/turn event |
| Remote provider timeout | status uncertain until provider job receipt, idempotent reconciliation | no local neural DeviceBackend state claim | job status unknown/retry policy |
| Partial coupled A/V solver failure | abort incompatible commit group, retain previous root | correct event/lease quarantine | no partial mislabeled final AV |

## 12. Complexity ceilings and future-proofing

- Do not add an enum variant to one universal hot-path router for each new model; use registered semantic schemas and TaskDriver selection at admission.
- Do not make a central interpreter evaluate general user-defined workflow graphs per media frame.
- Keep public interfaces small (verified ModelAssembly, TaskDriver, ValueDescriptor/Timebase, StateRoot, RegionProvider, ModeReceipt, native task stream).
- Keep model-specific semantics in one source adapter plus reusable operators; no separate model-serving subprocess.
- Component graph and stage control must be understandable by human maintainers and independent LLM agents without runtime-generated magic.
- No mandatory trait objects for each tensor or audio sample; capability registries and generics should compile into direct dispatch at prepared-plan boundaries.
- Separate reference math from fast kernel selection; staged evidence and model-specific numerical tests are non-negotiable.
- A genuinely new hardware class may execute AOT graphs, static dataflow or native kernels at its own granularity under one Event/Lease/PreparedRoot boundary.
- Document every additional persistent abstraction with a measured change-radius/latency justification and an explicit removal path if it fails.

## 13. Design acceptance (not yet implementation)

A new model/codec/TaskDriver must be introduced with changes confined to source adapter, task algorithm, semantic math, processor/codec and the best matching provider—not rewrite core Router, TaskCoordinator, BufferLease, or HTTP protocol handling. Test on source-free mocks first, then original checkpoint source, then actual hardware.

Required adversarial drills:
- Qwen-AR text and direct finite-scored decisions share state/runner but not output semantics.
- Whisper versus TDT streaming ASR share transport but not decoding algorithm.
- Kokoro without token LM; multi-depth RVQ audio; Moshi full duplex without intermediate text requirement.
- FLUX variants require correct conditioner; wrong variant, solver, guidance and VAE are rejected.
- Autoregressive multimodal image-model AR-versus-hybrid stages are based on inspected checkpoint, not family label.
- Wan2.2 phase-dependent denoising experts; Mochi unequal text/video projection widths; autoregressive video-model temporal tile masks.
- Mutually coupled A/V solver steps and exact clocks; arbitrary TTS-over-video composition is not equivalent.
- Packed ternary plus Hadamard; loading ternary weights without matching transform hard fails.
- Correct pre-submit rejection, physical cancellation quarantine, multi-model fairness and no false StateRoot publication.
- Host native inference versus independently permissioned external hosted jobs never conflated.
- A single request may include text+image+audio+video and valid typed output, with precise capability/transport refusal for unsupported combinations.

**Plan relation:** Existing MM-01..MM-12 remain governing initial drills; extend with explicit MM-13..MM-24 for real-time QoS, presentation, hosted job and API compatibility. See acceptance/roadmap document for exact fixtures and evidence gates.

## 14. Source references / provenance

Local normative prior documents: MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md; COMPOSITION_AND_EXTENSION_CONTRACTS.md; MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md; TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md; UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md; MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md; DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md.

External original-source research and specific variant cards are indexed in the modality dossiers and SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md. Source claims are dated to this research pass; repository updates or different checkpoint revisions must be reverified. Upstream open-source runtime code is prior art, not automatically authorized for embedding or redistribution.

````

