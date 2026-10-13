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
