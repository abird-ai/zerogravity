# Next implementation iteration — backend parity and execution-performance architecture

Status: **design only / dormant**.

This plan is the execution-performance iteration **after** the current live work,
`../cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md`, and
`../serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md` are closed.
It is intentionally isolated from the live TODOs so concurrent agents can finish
current backend/correctness work without this document changing their ownership or
order.

This document does not grant implementation, performance, numerical, backend or
serving acceptance. Current source, CURRENT_BASELINE.md, live owning TODOs and
their completed physical evidence remain authoritative until this iteration is
explicitly activated.

## 1. Objective

Make ZeroGravity's physical execution layer competitive with or faster than the
best matched vLLM path on selected qualified model/device profiles **without
turning ZeroGravity into a vLLM clone** and without weakening its stronger
ownership, completion, state-publication, memory or composition semantics.

The performance objective is deliberately narrower than "beat vLLM everywhere":

- preserve one reusable execution fabric across models and backends;
- make a well-qualified ZeroGravity physical plan able to match or beat vLLM on
  the model/device/profile combinations we deliberately optimize;
- use mature vendor/backend primitives when they are better than ours;
- specialize aggressively only when the specialization stays explicit,
  maintainable and measurable;
- keep the control plane, ownership model, memory model and pipeline composition
  general enough for LLM, ASR, TTS, image and mixed single-process pipelines.

The iteration is successful if ZeroGravity can generate an execution plan whose
**physical geometry, launch overhead, batching efficiency and device occupancy**
are no longer the dominant reason it loses to vLLM. It does not require owning
every kernel or reproducing every vLLM feature.

## 2. Why this iteration exists

The current baseline shows two different facts.

First, the historical conclusion that ZeroGravity was much faster than vLLM was
real for the retained **eager** vLLM baseline, but it did not establish superiority
over optimized vLLM. The current vLLM baseline uses its normal optimized execution
and is the correct engineering target.

Second, ZeroGravity's current numbers do **not** look like a fundamentally slower
architecture.

On L5, current 0.8B diagnostics are approximately:

- C1: ZeroGravity 39.621 tok/s vs vLLM 55.062 tok/s;
- C4: ZeroGravity 119.613 tok/s vs vLLM 125.637 tok/s;
- C8: ZeroGravity 133.676 tok/s vs vLLM 227.948 tok/s.

C4/C8 ZeroGravity are currently diagnostic only because the fused CUDA path fails
lane invariance, but their timing shape is still useful for architecture analysis.

For the same 47-token prompt + 3-token completion, approximate total wall time is:

- ZeroGravity C4: ~100 ms;
- ZeroGravity C8: ~180 ms;
- vLLM C4: ~96 ms;
- vLLM C8: ~105 ms.

The striking defect is therefore not merely "our GEMM is slower." The physical
cost of widening from four to eight requests grows far more in ZeroGravity than
in vLLM.

The 2B shape says the same thing:

- ZeroGravity C4: ~145 ms;
- ZeroGravity C8: ~250 ms;
- vLLM C4: ~119 ms;
- vLLM C8: ~152 ms.

The practical hypothesis for this iteration is:

> ZeroGravity's ownership/control architecture is already strong enough; the
> physical execution architecture is not yet exploiting enough token-level
> density, persistent device state, replay, overlap and mature backend kernels.

That is a tractable problem.

## 3. Mandatory ordering

### 3.1 Earlier iterations close first

Do **not** begin the performance slices in this document until:

1. the current live backend/correctness iteration is closed;
2. the CLI/model-lifecycle UX iteration is closed;
3. the full OpenAI-compatible serving iteration is closed.

The activation gate is semantic backend parity:

1. CUDA remains the current semantic/reference implementation where it is already
   qualified.
2. HIP reaches the same relevant public serving and model-neutral primitive
   capability.
3. Vulkan reaches the same relevant capability, using Vulkan-native mechanisms
   where CUDA/HIP-specific machinery has no direct equivalent.
4. Current deterministic CUDA fused C4/B8 lane divergence is fixed or the
   affected path remains disabled; an invalid path cannot be a performance base.

"Parity" here means **semantic and capability parity**, not byte-identical backend
implementation and not equal throughput.

A backend may use different kernels, layouts, graph/replay mechanisms or vendor
libraries while preserving the same higher-level contracts.

### 3.2 Once parity is closed, every performance slice is backend-complete

After activation, work proceeds one vertical slice at a time:

1. **HIP first.**
2. **CUDA second**, either through the shared implementation or a CUDA-specific
   specialization when measured evidence justifies it.
3. **Vulkan third**, using the same semantic primitive with Vulkan-native
   execution where appropriate.
4. Close the slice across all three backends.
5. Only then move to the next slice.

Do not run several half-complete backend optimizations in parallel and leave one
backend permanently behind.

HIP-first is intentional:

- it forces the shared primitive to be model/backend-neutral rather than silently
  encoding CUDA assumptions;
- X2 has enough VRAM to expose real batching and resident execution behavior;
- ROCm provides mature vendor primitives without CUDA's exact APIs;
- a primitive that is clean on HIP first is more likely to remain clean when CUDA
  specialization and Vulkan lowering are added.

CUDA remains free to specialize aggressively when a CUDA-only mechanism
materially wins, but the specialization sits behind the same shared semantic
contract.

## 4. Non-negotiable architecture

Performance work must preserve the following.

### 4.1 State publication remains authoritative

StateRoot, session/root lineage and complete-domain ownership remain the
logical state authority.

No graph, vendor kernel, persistent batch or asynchronous flight may publish
state by itself.

The sequence remains conceptually:

~~~text
authenticated current roots + candidate owners
        ↓
physical execution
        ↓
authoritative backend completion
        ↓
Ready / completed candidate
        ↓
existing checked StateRoot publication
~~~

### 4.2 Completion and unknown-completion semantics stay exact

Cancellation does not manufacture completion.

Unknown submitted work remains fenced/quarantined according to the existing
physical lifetime contract.

Performance work may overlap **preparation** and independent flights, but it may
not race, drop or infer completion.

### 4.3 ResourceArbiter and physical identity remain the memory authority

DeviceBufferIdentity, original backing/range identity, provider/device admission
and explicit lifetime ownership remain authoritative.

Persistent execution structures are consumers of admitted memory; they do not
become a second allocator or second ownership graph.

### 4.4 One scheduler and one state engine

Do not build:

- a Qwen-only scheduler;
- a second batching engine;
- a graph-specific publication path;
- a separate paged-KV engine;
- a backend-specific root/state model.

The existing fair worker, cohort formation, paged state, prefix cache, candidate
state and publication system remain the control substrate.

### 4.5 Reusable primitives survive rejected consumers

A negative A/B removes the selector/consumer, not a correct reusable primitive.

Preserve model-neutral:

- kernels;
- layouts;
- packed transports;
- graph/replay helpers;
- persistent metadata structures;
- state gather/scatter;
- scheduling transforms;
- completion/event mechanics;
- backend vendor adapters;

when they remain correct and cleanly reusable.

Delete a primitive only for correctness/safety/resource-contract failure or
strict supersession.

## 5. Performance design principles

### 5.1 Zero-cost Rust by default

Shared abstractions should compile away in the hot path where naturally
possible.

Prefer:

- enums resolved at load/plan time;
- monomorphized/static backend implementations;
- precomputed offsets and geometry;
- fixed-capacity/preallocated scratch;
- slices and POD descriptors;
- direct function/plan dispatch outside token loops;
- no per-token heap allocation;
- no per-token string/dynamic-map work;
- no locks in the device hot path unless measured and unavoidable.

Do not introduce elaborate type machinery merely to claim zero cost. If a
straightforward enum/match is simpler and outside the hot loop, use it.

### 5.2 Optimize the physical plan, not the control-plane syntax

A request-oriented scheduler is compatible with a token-oriented physical
executor.

Keep requests, sessions, roots and cancellation request-oriented.

Make arithmetic token-oriented where that improves device efficiency.

### 5.3 Specialize at load/plan time

Device/model/profile specialization should happen when a prepared model or
physical execution plan is built.

The hot loop should not repeatedly rediscover:

- tensor geometry;
- backend capability;
- kernel family;
- graph eligibility;
- workspace offsets;
- row/token packing;
- state-domain mappings.

### 5.4 Use mature primitives without ideology

ZeroGravity does not need to own every arithmetic kernel.

The physical plan may select, after qualification:

- ZeroGravity native kernels;
- cuBLAS/cuBLASLt;
- hipBLAS/hipBLASLt;
- FlashAttention/FlashInfer/AITER or another qualified attention primitive;
- Vulkan cooperative-matrix or other qualified Vulkan kernels;
- future backend libraries.

The selector must be load-time/static for the prepared plan where practical.

If a vendor primitive is materially faster and does not block a higher-value
fusion, use it.

## 6. Target physical architecture

The end-state is not "vLLM rewritten in Rust."

It is:

~~~text
requests / sessions
        │
        ▼
existing bounded fair scheduler
        │
        ├── StateRoot / ownership
        ├── paged state / prefix reuse
        └── cancellation / completion authority
        │
        ▼
logical scheduled spans
        │
        ▼
model-neutral physical batch description
        │
        ├── request cohort B1/B2/B4/B8
        ├── token spans per request
        ├── sequence offsets/frontiers
        ├── candidate-state destinations
        └── stable device metadata
        │
        ▼
prepared backend physical plan
        │
        ├── packed/ragged token work
        ├── persistent metadata
        ├── replayable execution regions
        ├── bounded multi-flight
        └── best qualified native/vendor kernels
        │
        ▼
backend completion
        │
        ▼
existing authoritative publication
~~~

Physical request cohorts remain bounded at B1/B2/B4/B8.

Offered C16/C32/C64 remains queue/refill/regroup pressure, not a reason to build
B16/B32/B64 model kernels.

## 7. Activation prerequisite — HIP and Vulkan parity with CUDA

This is primarily current-iteration work, but the next iteration must define its
entry condition precisely.

Parity is required for the **shared substrate that later performance slices will
modify**.

### 7.1 Required semantic parity surface

For the current Qwen source-serving path, the shared target surface includes,
where applicable:

- source-authenticated resident model loading;
- exact provider/device admission;
- session/root creation and retirement;
- contiguous and paged state policy;
- prefix restore/reuse with complete-domain compatibility;
- B1/B2/B4/B8 request cohorts;
- multi-token prefill;
- decode;
- attention + KV update;
- recurrent/GDN state;
- MLP/projection path;
- compact greedy/head output;
- private candidate state;
- state gather/scatter;
- cancellation before launch;
- cancellation after submitted work;
- unknown completion quarantine/fencing;
- physical identity/alias validation;
- clean resource retirement.

Backend parity does **not** require identical kernel names or launch counts.

### 7.2 HIP parity

HIP should become the first complete non-CUDA implementation of this semantic
surface.

Reuse existing shared primitives wherever possible.

ROCm-specific implementations may use:

- hipBLASLt/hipBLAS;
- AITER;
- HIP graphs;
- ROCm-specific wave/layout tuning;
- ROCm-specific memory or launch APIs;

behind the shared semantic layer.

Do not retag CUDA kernels or layouts as HIP-compatible without physical
qualification.

### 7.3 Vulkan parity

Vulkan should implement the same semantic surface using Vulkan-native mechanisms.

Do not force CUDA concepts literally onto Vulkan.

For example:

- CUDA/HIP graph replay may correspond to reusable/pre-recorded Vulkan command
  buffers plus stable descriptor/pipeline state;
- CUDA/HIP stream/event semantics map to explicit Vulkan queue/submission/fence
  ownership;
- cooperative-matrix or subgroup-specific kernels remain device-profile
  implementations, not shared arithmetic authority.

Respect actual Vulkan binding/buffer/subgroup limits. Large monolithic bindings
must page or select rows through authenticated ranges rather than assuming CUDA
addressability.

### 7.4 Parity acceptance

Keep acceptance intentionally small.

For each newly completed shared capability:

1. existing shared/unit coverage if the primitive has pure logic;
2. one smallest physical 0.8B path that proves the capability;
3. 2B only where geometry/model family materially differs;
4. 27B only where the shared primitive is specifically needed and the model fits;
5. clean retirement.

Do not create a backend × model × width × context fixture explosion merely to
declare parity.

The existing population and serving workloads are preferred evidence.

## 8. Slice 1 — physical scaling diagnosis and dispatch-budget closure

### Goal

Explain and remove the pathological cost growth between physically useful
cohort widths before adding more sophisticated execution machinery.

The key question is:

> Why does a wider qualified cohort consume much more wall time than the extra
> useful arithmetic should require?

The current CUDA B4→B8 timing shape is the motivating case, but this slice begins
on HIP after parity and then closes on CUDA and Vulkan.

### Shared design

Add only coarse physical-stage accounting needed to answer the question.

Useful stage classes are:

- input/token packing and embedding;
- projection families;
- attention;
- recurrent/GDN;
- MLP;
- state gather/scatter;
- head/greedy;
- explicit copies/zeroing;
- host synchronization/readback;
- total dispatch count.

Do **not** instrument every kernel permanently.

Prefer existing completion/dispatch counters and a small prepared-plan stage
classification.

Any temporary detailed probe should be removed once the bottleneck is understood,
unless it is broadly useful health telemetry.

### HIP first

Once HIP B4/B8 is correct:

- measure B1/B4/B8 with the existing canonical population workload;
- identify whether widening is limited by launch count, memory traffic, serial
  lane work, poor GEMM shape, recurrent state movement or host synchronization;
- remove accidental lane-serial work when a shared batch primitive already
  exists;
- retain backend-neutral fixes.

### CUDA second

Repeat the same coarse accounting on the corrected CUDA B4/B8 path.

CUDA may specialize where the data shows a CUDA-only issue:

- cuBLASLt shape selection;
- SM86 launch geometry;
- stream/event behavior;
- CUDA-specific packed layouts.

Do not optimize an invalid lane path.

### Vulkan third

Run the same semantic measurement at B1/B4/B8.

Focus on:

- dispatch count;
- descriptor/buffer rebinding;
- command-buffer construction;
- subgroup/cooperative-matrix geometry;
- unnecessary host/device synchronization.

### Slice closure

Close the slice only when:

- exact outputs/state remain correct;
- B8 cost growth is explained;
- avoidable serial work has been removed;
- the accepted implementation is simpler than alternative experimental paths;
- no speculative instrumentation framework remains.

The objective is not a fixed ratio, but B8 should no longer look like two nearly
serial B4 executions unless the model mathematics or hardware prove that is
unavoidable.

## 9. Slice 2 — token-oriented physical batching and ragged prefill

### Goal

Keep logical request cohorts but make dense arithmetic operate on the largest
useful token batch rather than repeatedly executing tiny temporal/request
fragments.

This is the most important vLLM-inspired architectural change.

### Shared semantic primitive

Introduce one model-neutral physical token-batch description.

It conceptually contains:

- bounded request members;
- per-request token span;
- sequence offsets/frontiers;
- packed-token base/count;
- mapping from packed token rows back to request/session owners;
- candidate-state destinations;
- readout requirements;
- exact source/model compatibility.

Avoid a second scheduler.

The scheduler still decides **which requests and token spans are allowed to
advance**.

The physical batch only describes **how those already-authorized spans are packed
for execution**.

### Physical layout

For compatible prefill work:

~~~text
request A: tokens a0..aN
request B: tokens b0..bM
request C: tokens c0..cK
          ↓
packed token rows
[a0 ... aN b0 ... bM c0 ... cK]
          ↓
large dense operators
          ↓
ragged sequence-aware state/attention handling
          ↓
scatter candidate state to original request owners
~~~

Dense GEMMs/RMS/elementwise work should see M = total packed token rows where
possible.

Attention/recurrent state receives sequence offsets and private state ownership;
it must not accidentally flatten causality across requests.

### HIP first

Build the shared packed-token primitive around HIP.

Prefer existing primitives:

- ragged attention;
- private KV/state;
- multi-token GDN;
- grouped recurrent transport;
- packed BF16 values;
- shared-input projections.

The first target is not every ragged shape.

Start with the existing equal/small canonical population workload and enough
raggedness to prove that packing is not tied to equal prompt lengths.

### CUDA second

Reuse the same physical batch description.

CUDA may select different packed kernels or cuBLASLt plans.

Where CUDA B4/B8 already has specialized fused paths, those may remain as fast
subplans if they beat the generic token-packed route.

The shared batch description must permit either implementation.

### Vulkan third

Lower the same token-batch semantics into Vulkan buffers/descriptors.

Use explicit offsets and checked binding ranges.

Do not require one giant buffer binding when device limits make paged/segmented
access more natural.

### Slice closure

Required evidence:

- existing canonical prompt workload exact;
- at least one unequal/ragged prompt shape through existing serving machinery;
- StateRoot/frontier ownership exact;
- clean retirement;
- measured reduction in prefill wall time or dispatch count.

No new fixture corpus is required.

## 10. Slice 3 — persistent device batch metadata

### Goal

Stop rebuilding and retransmitting batch metadata that is stable across decode
steps or can be patched cheaply in place.

vLLM's persistent batch is useful prior art here, but ZeroGravity should express
the concept through its own typed ownership model.

### Shared design

A prepared model may own a bounded persistent physical-batch workspace whose
device addresses remain stable for the prepared plan.

Potential persistent fields:

- slot/session-to-row mapping;
- sequence lengths/frontiers;
- token IDs / next-token input;
- KV page/block references;
- state-domain pointers/indices;
- packed-token offsets;
- output packet destinations.

The persistent structure is **not** logical state authority.

It is a cache/physical representation derived from authoritative current roots
and prepared candidate owners.

Every flight must still validate the root/session/epoch inputs it consumes.

### Update strategy

Prefer tiny bounded patches:

- changed token ID;
- changed frontier/length;
- changed page reference;
- changed active mask/slot mapping.

Avoid reconstructing large host arrays each token.

Where metadata can be produced by the previous GPU flight, keep it on device.

### HIP first

Use HIP device buffers and bounded async copies or native kernels for metadata
patching.

Measure whether host metadata preparation or upload is material before adding
clever GPU mutation.

### CUDA second

Use the same semantic structure.

CUDA-specific graph replay will benefit strongly from stable addresses.

If CUDA can keep more metadata GPU-resident because of graph requirements, that
is a backend specialization, not a new logical batch type.

### Vulkan third

Use stable descriptor/buffer layouts and update only changed ranges.

Where descriptor indexing or dynamic offsets are appropriate and qualified, use
them; otherwise prefer straightforward fixed layouts.

### Slice closure

- no new state authority;
- no per-token heap allocation in the hot physical path;
- stable bounded memory;
- exact cancellation/retirement;
- measured reduction in CPU preparation/H2D work before keeping complexity.

## 11. Slice 4 — replayable production execution regions

### Goal

Remove repeated host/driver launch overhead from stable hot regions while keeping
completion/publication outside the replay object.

This is the primary answer to vLLM's CUDA-graph advantage.

### Shared abstraction

Do **not** create a fake lowest-common-denominator "graph API."

The shared concept is a prepared **replayable physical region**:

- fixed admitted workspace;
- stable or patchable argument locations;
- known dependency boundaries;
- one authoritative submitted/completed receipt;
- explicit backend capability.

Backend implementations differ.

### HIP first

Evaluate HIP Graphs for stable decode/B4/B8 regions.

Capture only regions whose addresses/control flow are genuinely stable.

If HIP graph replay does not beat the normal launch path, retain the shared
region concept but reject the HIP graph consumer.

### CUDA second

Use CUDA Graphs where they materially reduce launch overhead.

Likely initial targets:

- B1 decode;
- B4 decode;
- B8 decode;
- stable head/greedy tail.

Piecewise regions are acceptable when full-model capture is awkward.

Do not capture StateRoot publication or ownership transitions into the graph.

Graph completion produces the same Ready/completion authority expected by the
existing publisher.

### Vulkan third

Use Vulkan's natural equivalent:

- reusable command buffers;
- stable pipeline/descriptor state;
- minimal per-step patches/submissions.

Do not force a CUDA-graph-shaped abstraction into Vulkan.

### Failure semantics

Unknown submitted replay remains unknown submitted GPU work.

The graph/command buffer cannot be "forgotten" because cancellation happened.

Existing fencing/quarantine rules apply.

### Slice closure

For each backend:

- one correct B1 replay path first;
- B4/B8 only after B1 shows a real benefit;
- compare normal launch vs replay on the same current workload;
- reject replay if the gain is noise or maintenance cost is disproportionate;
- keep only broadly useful replay primitives.

## 12. Slice 5 — bounded two-flight overlap

### Goal

Overlap CPU scheduling/preparation of the next physical flight with current GPU
execution without weakening completion authority.

Do not build arbitrary asynchronous execution.

Start with **two** bounded flights.

### Shared model

~~~text
CPU: prepare N+1  ───────────────┐
                                 │
GPU: execute N ───────► complete │
                                 ▼
GPU:                 execute N+1
~~~

Each flight owns:

- original current roots;
- candidate owners;
- physical input/output ranges;
- backend completion object;
- cancellation observation;
- prepared physical batch metadata.

No flight may reuse an allocation that remains owned by an unresolved previous
flight unless the physical access ledger explicitly permits it.

### Worker integration

Keep the existing worker as the scheduling authority.

The worker may have:

- one executing flight;
- one fully prepared but not yet submitted flight;

or, where safe and beneficial:

- two submitted independent flights on separate admitted resources.

Do not let asynchronous preparation reorder StateRoot publication.

### HIP first

Use HIP streams/events through the existing completion/retirement mechanics.

The first proof should be simple:

- independent compatible batches;
- no shared write ranges;
- exact reverse/normal completion;
- cancellation before submit and after submit;
- no resource leak.

Then measure overlap on real serving.

### CUDA second

Reuse the same bounded flight model.

CUDA may use multiple streams, graph launches, event dependencies and the
existing multi-flight substrate.

### Vulkan third

Use queue submissions/fences according to actual queue-family/device support.

One queue with overlapping host preparation is still useful; multi-queue is not
required.

### Slice closure

Keep two-flight overlap only if real-model wall/goodput improves.

Do not expand to arbitrary N-flight concurrency without new evidence.

Two is a mechanism, not a permanent magic constant; increasing it requires a
measured bottleneck and bounded admission proof.

## 13. Slice 6 — qualified best-primitive selection

### Goal

Stop spending engineering effort reimplementing mature kernels when a qualified
vendor/backend primitive is faster, while retaining ZeroGravity kernels where
fusion or semantics give us an advantage.

### Operator families to compare

At minimum:

- dense BF16 GEMM/projection;
- attention;
- RMS/normalization where a fused opportunity exists;
- recurrent/GDN stages;
- head/greedy;
- KV copy/append/gather/scatter.

### Selection rules

A primitive may become production-selected only if it preserves:

- numerical contract;
- exact relevant source/state semantics;
- bounded workspace;
- cancellation/completion/lifetime contract;
- backend/device qualification.

Selection occurs at prepared-plan/load time where possible.

### HIP first

Compare ZeroGravity paths against available ROCm primitives such as hipBLASLt,
AITER and qualified attention implementations.

Do not replace model-semantic fusion with a generic library call when the fused
ZeroGravity path is faster end-to-end.

### CUDA second

Compare against cuBLASLt and qualified attention libraries.

The CUDA implementation may use a different winner than HIP.

Do not treat one backend's winner as model-neutral truth.

### Vulkan third

Compare native Vulkan kernels against cooperative-matrix/subgroup/pipeline
variants supported by the actual device.

Vulkan may keep more ZeroGravity-owned kernels because the vendor-library
ecosystem differs.

### Evidence policy

Use tiny operator A/Bs only when a whole-model stage total indicates the operator
matters.

Do not create a standing exhaustive microbenchmark farm.

Whole-model end-to-end improvement is the final selector.

## 14. Slice 7 — targeted semantic fusion and kernel tuning

### Goal

Only after execution geometry, replay, overlap and vendor selection are sound,
spend effort on lower-level kernel optimization.

This is where the existing dormant candidate library becomes most valuable.

### Existing reusable candidates

Preserve and reconsider, where measured hot-path data points to them:

- lane regroup;
- copy + inactive-tail zero;
- 17 BF16 GEMM strategies;
- recurrent tiled transpose;
- GDN occupancy variants;
- grouped recurrent/Conv transport;
- packed BF16 greedy;
- head→greedy partials;
- dual Q/K RMS;
- certified recurrent export;
- shared-input projection composition;
- ragged attention/private KV/rotary/stateless primitives;
- CUDA/HIP transfer-retirement primitives.

### Fusion priority

Favor semantic fusions that remove material traffic or launches:

- RMS → projection;
- shared-input Q/K/V;
- dual Q/K RMS;
- projection → elementwise activation where numerically valid;
- recurrent state import → GDN → export;
- head → greedy partial reduction;
- KV append with adjacent state work where ownership/ranges permit it.

Do not fuse simply because two kernels are adjacent.

A fusion must preserve numerical ordering required by the selected numerical
contract.

### Backend order

HIP first identifies whether the shared semantic fusion is valuable.

CUDA may implement a specialized fused kernel/CUTLASS/cuBLASLt epilogue path.

Vulkan receives an equivalent semantic fusion only if its pipeline/layout makes
the fusion beneficial.

### Slice closure

Keep a tuned kernel only when:

- it wins a measured hot path;
- the end-to-end model path improves or the primitive is clearly reused by
  another qualified workload;
- maintenance cost is proportionate.

Otherwise leave the correct primitive dormant or reject the selector.

## 15. Slice 8 — sustained serving closure

### Goal

After B1/B4/B8 physical execution is efficient, verify that high offered
concurrency keeps the best physical width continuously occupied.

Do not add B16/B32/B64 model kernels.

### Expected behavior

For offered populations larger than the active physical cohort:

- queue fills bounded active slots;
- completions free slots;
- pending requests refill immediately;
- compatible rows regroup into B8/B4/B2/B1;
- prefill/decode fairness remains bounded;
- output backpressure parks only affected rows;
- GPU remains supplied with useful work.

### What to compare

Use the existing population benchmark and comparator.

At C16/C32/C64 measure:

- output tok/s;
- request completion rate;
- physical cohort-width distribution;
- GPU occupancy/utilization if already cheaply available;
- queue/active health;
- exact IDs;
- clean retirement.

Do not add bespoke long-running harnesses.

### Closure objective

For the selected model/device profiles, sustained throughput should approach the
best stable B8 physical goodput instead of flattening due to CPU scheduling or
repeated drain/refill gaps.

If vLLM still scales materially better after B8 parity, investigate scheduling
overlap and physical occupancy before adding kernel width.

## 16. Performance goals and decision gates

The purpose is to **beat** the matched baseline where practical, but the
architecture must not contort itself to win one benchmark cell.

### Milestone A — correctness

Mandatory:

- exact expected output;
- exact required StateRoot/state domains;
- lane invariance for identical requests;
- no source/ownership weakening;
- clean completion/retirement.

No performance claim exists without this.

### Milestone B — architectural parity

For target 0.8B/2B profiles:

- C1 no worse than ~10% behind the matched vLLM result;
- B4/B8 scaling no longer shows obvious serial-width pathology;
- high offered concurrency is limited primarily by physical GPU execution, not
  queue/refill bubbles.

This is a diagnostic milestone, not the final objective.

### Milestone C — target performance

For deliberately optimized model/device profiles:

- match or beat vLLM C1;
- match or beat vLLM B4/B8 physical goodput;
- match or beat sustained offered C16/C32/C64 throughput where the same resource
  envelope and correctness contract apply.

The preferred target is a clear repeatable win, not a 1% noise claim.

A useful engineering threshold for declaring a selector winner is normally a
repeatable ≥3–5% end-to-end benefit, unless the change removes major complexity
or resource use.

### 27B

Do not let 27B dominate early iteration latency.

Use 0.8B first, 2B for geometry/model confirmation, then 27B once the shared
mechanism is already credible.

27B is important for:

- memory behavior;
- large-model arithmetic;
- resident-vs-streamed policy;
- confirming that a shared primitive scales.

It should not be the first debugging vehicle for every optimization.

## 17. Minimal evidence policy

This iteration explicitly rejects evidence overengineering.

### Reuse existing evidence

If a shared primitive and its contracts are unchanged, reuse accepted evidence.

Do not re-run large historical matrices merely because a nearby selector changed.

### For each slice/backend

Default evidence should be:

1. focused compile/lint for changed surface;
2. existing small logic tests if shared pure logic changed;
3. one real 0.8B physical canary;
4. one matched performance cell;
5. 2B only when needed to confirm geometry/model scaling;
6. 27B only when relevant;
7. clean retirement.

Expand only when a real failure demands it.

### No fixture explosion

Do not add:

- synthetic width matrices merely to increase coverage;
- permanent debug probes for one investigation;
- duplicate correctness archives;
- elaborate benchmark receipt pinning;
- a second comparison framework.

Use:

- CURRENT_BASELINE.md;
- existing population benchmarks;
- existing serve-compare.py / serve-matrix.py;
- existing source/reference authorities.

### Performance evidence

Prefer a few matched real-model samples over many microbenchmarks.

Warm/cold/resource envelopes must be comparable when claiming a win.

Do not average correctness-invalid rows into a performance claim.

## 18. Backend-specific freedom under one semantic contract

Parity does not mean forcing the same machinery everywhere.

### HIP may specialize through

- AITER;
- hipBLASLt;
- HIP Graphs;
- ROCm wave32 geometry;
- ROCm-specific async copy/event behavior.

### CUDA may specialize through

- cuBLASLt;
- CUDA Graphs;
- FlashAttention/FlashInfer where qualified;
- SM-specific kernels;
- CUDA stream/event optimizations.

### Vulkan may specialize through

- reusable command buffers;
- cooperative matrices/subgroup operations;
- descriptor indexing/dynamic offsets;
- Vulkan-specific storage tiling;
- device-specific pipeline specialization.

These remain implementation details beneath shared semantic primitives.

A backend specialization must not leak backend-specific assumptions into:

- session/root ownership;
- scheduler semantics;
- prefix compatibility;
- paged-state authority;
- cancellation/completion semantics.

## 19. Where ZeroGravity should seek structural wins

Do not judge the project only by a short standalone LLM benchmark.

The architecture has several places where it can plausibly exceed vLLM.

### 19.1 Exact prepared specialization

ZeroGravity can know at load time:

- exact model;
- exact device;
- exact dtype;
- exact state layout;
- qualified request bands;
- memory envelope;
- selected kernel families.

Exploit that to build a static physical plan with minimal hot-path discovery.

### 19.2 Model-semantic fusion

Because ZeroGravity owns explicit model/state semantics, it can fuse across
boundaries that a generic compiler may not safely infer.

GDN prefill is already evidence that semantic restructuring can produce orders
of magnitude more value than small kernel tuning.

### 19.3 Stable ownership and memory topology

Explicit physical identity, leases, candidate roots and paged-state ownership
allow stable device addresses and aggressive precomputation without giving up
safety.

Use that stability for persistent metadata and replay.

### 19.4 Single-process pipeline composition

Longer term, ZeroGravity's strongest product-level performance advantage may be
outside standalone LLM serving:

~~~text
ASR → LLM → TTS
image encoder → LLM → image decoder
audio → reasoning → speech
~~~

One process can retain device memory, state and scheduler ownership across model
boundaries.

Avoid:

- HTTP serialization;
- process crossings;
- duplicate tokenization/transport;
- duplicate GPU staging;
- independent schedulers fighting for the same device.

The execution primitives in this iteration should remain reusable for that goal.

## 20. What not to do

Do not:

- rewrite the engine as vLLM in Rust;
- create B16/B32/B64 model kernels because offered concurrency is 64;
- fork scheduler/state/publication per backend;
- add another paging/COW mechanism;
- add graph-owned state publication;
- keep correctness-invalid fast paths enabled;
- insist on ZeroGravity-native kernels when mature vendor kernels win;
- micro-tune kernels before fixing execution geometry;
- replace clear code with template/type complexity for theoretical zero cost;
- optimize one model by baking its dimensions into supposedly shared primitives;
- expand test matrices without a concrete uncovered risk;
- use a performance win to justify weaker cancellation/lifetime semantics.

## 21. Slice implementation protocol

When this iteration becomes active, each slice follows the same small loop.

### Step 1 — state the measured deficit

One paragraph, one current baseline.

Example:

> HIP B8 prefill spends X ms in serial recurrent work; total B8 wall is Y vs
> target Z.

No speculative optimization backlog.

### Step 2 — choose the smallest shared primitive

Prefer a reusable model-neutral mechanism.

If a backend-specific specialization is required, keep the semantic interface
shared and the backend implementation local.

### Step 3 — HIP implementation

Implement and validate the shared mechanism on HIP first.

Use smallest real-model feedback.

### Step 4 — CUDA implementation

Reuse the shared primitive.

Add CUDA specialization only when measured.

### Step 5 — Vulkan implementation

Implement the same semantics through Vulkan-native mechanisms.

### Step 6 — close the slice

Record:

- correctness status;
- small real-model timing;
- accepted selector;
- rejected selectors;
- any reusable dormant primitive retained.

Then move on.

Do not continue polishing a closed slice because another possible 1% optimization
exists.

## 22. Recommended slice order

After current backend parity closes:

1. **Physical scaling diagnosis / B4→B8 cost closure.**
2. **Token-oriented physical batching + ragged prefill.**
3. **Persistent device batch metadata.**
4. **Replayable production regions.**
5. **Bounded two-flight overlap.**
6. **Best qualified native/vendor primitive selection.**
7. **Targeted semantic fusion + kernel tuning.**
8. **Sustained C16/C32/C64 serving closure.**
9. Re-evaluate the matched baseline and decide the next bottleneck from data.

This order is intentional.

Do not begin by tuning GEMM tiles or adding more concurrency.

The largest expected gains come from executing the right amount of useful work
per flight with less launch/host overhead.

## 23. Exit criteria for the iteration

The iteration may close when all of the following hold for the deliberately
qualified target profiles.

### Architecture

- HIP/CUDA/Vulkan share the same logical batching/state/completion model.
- Each backend has an efficient physical implementation.
- No backend requires a separate scheduler or state engine.
- token-packed/ragged execution is reusable by other models;
- replay and overlap are bounded ownership-aware primitives;
- vendor-kernel selection is explicit and replaceable.

### Correctness

- C1/B4/B8 exact on supported current profiles;
- identical lanes are invariant;
- paged/prefix/candidate state remains exact;
- cancellation/unknown completion retains existing authority;
- clean retirement.

### Performance

- singleton overhead is competitive with the matched baseline;
- B8 scales efficiently from B4;
- offered C16/C32/C64 keeps the best physical cohort supplied;
- selected target profiles match or beat the current vLLM baseline, or any
  remaining deficit is localized to a specific arithmetic primitive rather than
  scheduler/launch/memory architecture.

### Maintainability

- rejected selectors removed;
- reusable correct primitives retained;
- no duplicate benchmark framework;
- no unnecessary fixture corpus;
- no hot-path dynamic abstraction that can be cleanly resolved at plan time;
- the resulting architecture is simpler to reason about than the collection of
  experiments that produced it.

## 24. First activation prompt

When the current iteration is fully closed, the next agent should begin with:

> Read CURRENT_BASELINE.md, the live serving/backend parity TODOs and
> NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md. Revalidate that
> HIP and Vulkan have reached the required CUDA semantic parity and that the
> current fused CUDA lane-correctness defect is closed or disabled. Do not start
> performance work if those entry conditions are false. If they are true, start
> Slice 1 on HIP: use the existing population workload and coarse existing
> dispatch/stage accounting to explain B4→B8 physical cost growth. Prefer the
> smallest shared model-neutral fix; no new fixture matrix or benchmark
> framework. After a coherent HIP result, port/qualify the same slice on CUDA
> (specialize only if measured), then Vulkan, close the slice in durable docs and
> only then proceed to token-oriented physical batching.