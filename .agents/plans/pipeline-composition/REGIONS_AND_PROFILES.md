Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# Recursive pipelines and composable optimization profiles

Design proposal, 21 Oct 2026. Interfaces and new recursive/profile behavior remain
unimplemented; no runtime or performance qualification follows from this document.
Parent: [pipeline composition](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/pipeline-composition/README.md).
The coordinator reconciles shared plans; current source/tests and live TODOs
outrank this proposal. Existing qualified execution remains unchanged.

Owner addendum: [profile policy, replaceable components and native reuse](PROFILE_POLICY_AND_COMPONENT_REUSE.md)
specifies explicit approximation, interchangeable implementations, shared policy
context and native correctness gates. References below to one allocator/runtime
preserve common ownership/accounting; they do not freeze an allocator algorithm
or prohibit audited statically linked components and faithful ports.

## 1. Decision

Use **semantic primitives → recursively composed regions/tasks → profiles**.
Preserve one mathematical engine and the existing ownership spine.

- A primitive names an exact operation, value type and effect contract. A native
  kernel is one implementation of a primitive or a whole closed region.
- A pipeline node may contain a subpipeline. Nesting describes reusable
  structure, typed boundaries and optimization opportunities; it does not add a
  recursive interpreter, scheduler, allocator or state publisher.
- A profile assembles compatible task/region pipelines and prepared execution
  strategies for a workload and target. It is a named policy recipe, not proof
  that its source, arithmetic, shape, device or runtime is qualified.
- A highly specialized model may lower to the smallest efficient **kernel path**:
  one kernel, several fused kernels, a captured graph or a whole-model device
  program. Minimize measured complete-request cost, rather than kernel count.
- Keep two enduring execution representations: the semantic program and its
  prepared physical executable plan. Hierarchy, region optimization and command
  lowering are parts/views of those two representations.

The existing `RegionExecutionProvider`/`ExecutableBundle` contract already
permits closed region and whole-model replacement. Implement that seam
incrementally when needed; do not introduce a competing framework.

## 2. Current source versus design vocabulary

Inspected MAIN and the explicitly named ISO implementation; no tests rerun.
These anchors identify source, not immutable runtime qualification.

| Existing contract | Current scope | Proposed use |
| --- | --- | --- |
| [shared layer traversal](/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/forward.rs:19) | `run_layer_sequence` is the family-neutral traversal seam; profile-specific closures retain layer mathematics | Continue recording/evaluating through this shared seam; no second optimized model traversal |
| [typed operations and plan](/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/accelerator/program.rs:513) | `PlanOperation`, flat `PlanNode` list, `ForwardPlan`; source/profile/value precision, state versions and frontier | Preserve canonical operations; add private nested region scopes only when a real lowering needs them |
| [numerical contracts](/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/accelerator/program/typed.rs:32) | `ProgramNumericalProfile` is explicit and closed; its presence grants no backend eligibility | Extend with independently qualified arithmetic contracts; do not turn workload profiles into this enum |
| [resident lowering](/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-gpu/src/resident_program.rs:1) | Exact plan-to-physical bindings/preflight/preparation and launch | Initial concrete home for qualified region strategies, layout and scratch planning |
| [configuration structure](/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/pipeline.rs:284) | `ModelInstance`, `PipelineNode`, edges, flat stage lists | Source-neutral task recipe; no nested execution, source admission or device effects today |
| [pure validator](/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/pipeline.rs:418) | `VerifiedShape.ordered` is a topological stage order | Structural validation must remain effect-free; successful recursion checking would still not authorize execution |
| [context owner](/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/context_snapshot.rs:219) | Revocable grants and frozen authorized committed records | Outer tasks can pass read-only context snapshots; they cannot manufacture grants or neural caches |
| [freshness boundary](/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/context_snapshot.rs:530) | `ContextReader::require_fresh` rejects stale/foreign snapshots | Recheck at actual consuming admission/publication/delivery boundaries |
| [state publication](/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-gpu/src/resident_inference.rs:1257) | Prepared candidate validated, original service fence, authority transition, state swap, token bound to actual published root | Preserve this ownership boundary through any inner nesting/fusion |
| [event ownership](/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/device_events.rs:1) | Ready requires actual completion; unknown completion forbids fallback/reuse/publication | Every new region, capture, transfer and task cancellation obeys it |

`RegionExecutionProvider`, `ExecutableBundle`, SIR/CIR and the generic
TaskDriver sketches are architectural vocabulary in the
[reference architecture](/home/pvl/spaces/abird/src/zerogravity/.agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md:30)
and [composition contracts](/home/pvl/spaces/abird/src/zerogravity/.agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md:63).
Do not describe them as a general recursive runtime already implemented.

Observed qualification boundary at this design snapshot:

- Real original GGUF B2/B4/B8, independent roots and bounded private graph/state
  failure cases exist; this does not qualify arbitrary source/profile/shape.
- Isolated original .8B forced five-phase full24/all42/fullV proof and faults
  passed, followed by the complete gate17. Its actual
  [positive log](/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-upstream-cuda-root-20261020/root-full24-positive-3/execution.log:3)
  explicitly says no model-root publication, natural feedback, performance or
  serving qualification. Subsequently ISO natural binding/publication passed
  finalhelper43's seven actual physical modes, strictcombined Clippy2 and complete
  gate18. All24/all42/fullV/actual feedback and original roots match for five fixed
  1/1/64/2/1 calls. Reusable calls and canonical benchmark shapes remain open.
- Original 2B natural/all42 independent reference passed; native model/serving
  qualification is still open. See the
  [live numerical queue](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/numerics-and-context/TODO.md:389).
- Production graph admission, full temporal GPU prefill, paged complete-domain
  prefix/COW, large dense model execution, media and distributed runtime remain
  open. The [current batching design](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md)
  is a follow-up contract, not implementation proof.

## 3. Three conceptual layers, two execution representations

```text
source/config + model mathematical schedule + trusted task definition
                                  |
               semantic program, with nested region boundaries
                    primitive or region(subnodes + typed ports)
                                  |
           compiler-local expansion / inlining / outlining / fusion
                   compatible region implementation candidates
                                  |
             prepared physical executable plan / ExecutableBundle
             kernels, graph programs, transfers, lifetime/event DAG
                                  |
            original worker + arbiter + late per-attempt bindings
                                  |
              actual Ready -> original coordinator PublishRoot
                                  |
                    ordered committed task/media output
```

### Primitive layer

Start with existing `PlanOperation`, typed source operands and registered media
contracts. A primitive is the smallest **useful semantic unit**, not necessarily
an elementary scalar operation or one GPU launch. Examples: source-typed
projection, specified RMS, append/read state, GDN recurrence, a verified codec
transform. Novel semantics require a source formula, portable reference and
tests; a new optimized kernel alone does not add a new semantic primitive.

Each contract specifies input/output types, shape rules, finite-precision math,
read/write effects, source provenance and valid failure behavior. Physical
implementation details—tile size, memory swizzle, graph capture, CPU ISA—belong
to candidate lowering. A view/alias is an explicit contract, not an unrecorded
copy-elision assumption.

### Recursive pipeline layer

Use the same composition idea at two scopes with different port/effect types:

| Scope | Nodes/ports | Legal control | Visibility |
| --- | --- | --- | --- |
| Mathematical region | Tensor/scalar/state-view ports; operation or subregion | Acyclic dependency/effect graph, bounded static repeated layers; trusted source-qualified device loops if needed | All intermediate state provisional under one model/task attempt |
| Compound AI task | Text/media/context ports; registered task or subpipeline | Bounded acyclic stage dependencies; trusted TaskDriver owns loops, input waits and legal yields | Explicit stage/segment/turn commit frontier; see section 7 |

Nesting is structural. Reject recursive definition cycles and unbounded expanded
node/edge/depth/constant sizes before source/resource access. Reused definitions
may form a DAG; each invocation has distinct scoped values and mutable bindings.
Sharing a definition is not sharing a session, random stream or StateRoot.

Initially, store region scopes as private metadata over existing plan nodes and
scoped value IDs; flatten deterministically to the current semantic operation
schedule. The flat plan remains executable reference; scopes are optimization
boundaries. If later the semantic program stores nested bodies directly, its
deterministic expansion must reproduce that same reference schedule. Do not keep
a second serialized nested IR alongside a separate authoritative flat IR.

The YAML/CLI task config remains source configuration, just as today. A later
bounded subpipeline reference can expand at validation/compile time into typed
task stages. It is not a third executable representation or an interpreter for
arbitrary workflows. Do not change `zerogravity.pipeline.v1` or imply new syntax
works as part of this design-only task.

### Profile layer

A profile composes reusable pipelines and compatible policy choices. It does
not redefine primitive equations or load a different model implicitly. Resolve
it once at model/session/task preparation; workers use prepared IDs and bounded
data bindings rather than recursively looking up profile strings per token.

## 4. Typed closed region contract

Use a small private Rust descriptor before introducing public traits or crates.
The following fields are contract requirements, not proposed public ABI:

| Field | Meaning and required check |
| --- | --- |
| Identity | Source-bound model/semantic region revision; ordered operation/effect identity, child scope identities |
| Ports | Named typed inputs/outputs; tensor rank, dimension relations, strides/layout, precision, valid/ragged extents, semantic media type/clock where relevant |
| Source | Immutable artifact identity and live original issuer; codec/scales/grouping, exact tensor members/axes/offsets, tokenizer/processor/adapter as applicable |
| Numerics | Profile, phase, cast/store points, reduction/FMA order, padding/mask, nonfinite/status, tie rules and allowed equivalence class |
| Effects | Read and provisional-write state versions, alias rules, ordered host/IO effects, required observations, output frontier and cancellation boundaries |
| Guard | Bounded shape/device/compiler/state-schema/layout/placement compatibility; unsupported versus invalid versus unsafe failure |
| Resources | Peak live and candidate state, scratch, code/module/graph memory, tables, transfer/IO staging, encoded reader and event dependencies |
| Fallback | Exact original decomposition, compatible accepted alternative or typed refusal; no invented reference for unknown semantics |

A child exports the complete effect summary of its body. The parent cannot hide
a state write, source read, synchronization requirement or output observer.
Preserve per-domain version dependency edges across scope boundaries; do not
infer independence merely from different region names.

Port compatibility has separate semantic and physical checks. Equal shape or
byte size is insufficient for source, precision, state schema, position policy,
media clock, ownership or authorization compatibility. Convert only through an
explicit qualified operation. A same-source contiguous materialization may be
physical lowering; re-encoding precision changes needs the selected arithmetic
or derived-source contract.

## 5. Lowering and specialization

Prepare a small candidate set outside the request hot loop:

1. **Expand and validate.** Resolve static source/model/task definitions, bounded
   nested scope references, ports, value types and complete effect graph.
2. **Retain baseline.** The original source/numerical decomposition is forcible
   for comparison and fallback; unknown semantic operations refuse.
3. **Inline.** Expose producer/consumer regions where compatible ports/effects
   permit cross-boundary layout/fusion. Hierarchy does not force a dispatch.
4. **Outline.** Keep a reusable prepared region when it reduces code/capture
   duplication or marks a required host/device/codec/state boundary. An outlined
   mathematical region is not a new runtime session.
5. **Select closed replacements.** Existing RegionExecutionProvider seam offers
   per-op, fused-region, layer, stage or whole-model candidates. Hand-written
   kernels and compiler-produced programs satisfy the same contract.
6. **Lower physical lifetime.** Derive aliases, copies, buffer live intervals,
   encoded readers, transfer/stream dependencies, status and Ready requirements
   from the **selected** physical path, after fusion/outlining.
7. **Prepare once.** Admit resources, compile/load bounded shape bands, validate
   immutable source and code identity, prewarm only admitted variants.
8. **Bind late.** Original worker validates actual attempt/root/row/resource/event
   bindings, submits a compatible prepared path, and original coordinator alone
   decides publication.

No token-by-token graph search, generic pipeline interpreter, plugin DSL,
second allocator or independent tuning runtime. Offline bounded search and
load-time compilation may use transient region/tile views. Runtime choice is a
small selection among **accepted compatible** variants under current admission,
fairness and deadline constraints. A cost model can decline or choose K=0; it
cannot waive a failed source, arithmetic, ownership or physical gate.

Autotuner choices belong to the prepared implementation identity, including
selected tile/reduction configuration and code. Current original2B and
canonical47 investigations show both configuration variation and exact-state
comparison failures; causality remains under investigation. A policy cannot
treat all shape-compatible choices as numerically interchangeable. Qualify the
relation declared by the selected exact or approximate arithmetic contract;
preserve that relation when composing, caching or replacing a region.

Preparation may hoist pure source-only subregions, such as learned gamma
widening followed by its specified `+1`, out of per-token execution. Preserve
the exact arithmetic, cast/store observations and source/device/code identity;
the derived value belongs to the original admitted model lifetime. Treat it as
a prepared physical implementation of the same semantics, with explicit
resident bytes and completion, rather than a new learned source or cache
authority. Keep root-, position-, input- and request-dependent work in its
proper binding scope. A changed reduction or rounding order requires its own
numerical contract and qualification.

Whole-model replacement is legal only when every port/effect/required numerical
observation is covered, all state remains provisional, physical resources are
visible to original admission and a tested state bridge permits cancellation,
fork/restore/rejection as required. An opaque model service is not such a
candidate. A fast device program lacking safe state bridge remains ineligible
for stateful or speculative replacement.

### Resident values and cross-stage lowering — owner21 Oct

Optimize compatible producer/consumer chains across nested region and task-stage
boundaries. A logical pipeline boundary must not by itself require a host return,
serialization, allocation, tensor materialization, copy, synchronization or
physical launch. Keep hierarchy useful for reasoning while preparing the
smallest useful physical path through the existing engine.

```text
host input -> explicit ingress -> device stage A -> device stage B -> device stage C
                               leased values + ordered device dependencies
                 original authorities observe required completion/status metadata
                                      -> requested output / explicit egress
```

- Carry a typed value identity with a leased physical binding where eligible:
  original backing owner/device/context, allocation/range/generation, dtype,
  shape/stride/alignment, valid extents, access mode, source/profile/root scope,
  semantic media/position metadata and real producer dependencies. Reuse existing
  value IDs, handles and event owners; no new tensor runtime or serialized IR.
- Share the admitted backing through a view when provenance, alias/layout,
  access and consumer compatibility permit. A stage's host metadata can remain
  cheap while its payload stays on device. Equal shapes or pointers alone do
  not prove semantic compatibility, safe aliases or readiness.
- Keep device-resident input/state, intermediates, readout and next-token
  feedback on compatible paths. Read back requested output and required bounded
  control/status information; debug/full-logit observations remain explicit
  consumers. No implicit download/reupload or text encode/decode between stages.

The prepared physical plan chooses a bounded qualified candidate:

| Candidate | What it removes / retains |
| --- | --- |
| Resident handoff | Reuse a compatible leased output in the next stage, retaining separate kernels and original state authorities |
| Prepared device chain / graph | Order several kernels/copies with device dependencies and fewer host submissions, retaining every required effect/observation |
| Fused region | Combine legal producer/consumer work and optionally remove intermediate storage/traffic while preserving the selected arithmetic contract |
| Explicit bridge | Perform a necessary layout/codec/device transfer, host operation or materialization through an admitted, observable operation |

These choices compose and have independent costs. Graph replay does not imply
one kernel; resident handoff does not imply kernel fusion. Preparation resolves
ports, placements, layouts and finite shape candidates outside the token hot
loop. An accepted decomposition remains forcible for comparison/fallback.
Cost these separately: host/device payload transfers, host submission/wait
overhead and intermediate device-memory reads/writes. Label measured traffic
and logical byte estimates independently; none is a substitute for task latency.

- Plan placement/layout and the joint live set over the eligible connected
  region, including multiple consumers, old/candidate roots, staging, captured
  readers and graph pools. Reuse private scratch after actual last use. A copy
  or smaller region can win by shortening lifetimes or improving layout/occupancy;
  keeping everything resident is not an unlimited-memory requirement.
- Reused native components expose compatible caller-owned buffers/workspace and
  stream/event dependencies where available. Account hidden copies, pools and
  waits; a component lacking the required seam needs an explicit bridge or a
  suitable port, not a second allocator/runtime outside original accounting.
- Use real device dependency/event ordering instead of a blocking host wait per
  stage. Overlap admitted copies/IO with independent work only after readiness
  and lifetime checks. Preserve bounded cancellation/fairness/yield and output
  credit points; no unbounded captured chain or unchecked queued output.
- Preserve exact host decisions, sampler/grammar semantics, context grants and
  per-model publication boundaries. Split a graph at such a boundary until a
  separately qualified device-native control region can preserve its contract.
  Logical publication may exchange metadata while payloads remain resident;
  fusion never grants a kernel independent root-publication authority.
- Across different models, require an actual typed trained/codec bridge. A
  transcript, token IDs, acoustic features and hidden states are distinct ports;
  two tokenizers or equal-width activations are not interchangeable. Cross-model
  state roots remain independent and staged commits remain observable.
- Every actual materialization, H2D/D2H/device copy, synchronization, required
  boundary and graph split has an explainable reason and attributed byte/cost
  counters. Examples: layout/codec mismatch, different device/context, requested
  host observation, publication/control boundary, pressure or unsupported path.
  Keep detailed tracing separate from primary unprofiled performance runs.

Implement the first measured same-model compatible chain through private
extensions to current ports/providers. Later processor/model/media chains use
the same contracts after their own source/runtime qualification. NVIDIA first,
then Strix Halo: UMA does not by itself prove zero-copy, coherence or low cost.
No generic optimizer framework is a prerequisite for the first useful region.

## 6. Fusion legality and physical limits

Fusion must pass all of these, before timing:

- Same source operands and semantics; retain actual tensor member order/codec.
- Same required casts and independently observable stores. For example, an
  unrounded F32 residual sum and separately rounded BF16 residual side output
  must both survive when the profile requires them. A BF16 sum widened back to
  F32 is not equivalent. Preserve selected reduction tree/FMA and source phase
  order, even when real-number algebra would allow reassociation.
- Same row/head/position/causal/branch masks, inactive/padded tail handling,
  signed zero, status propagation, selected-ID ties and finite checking.
- Complete external outputs and state effects: no dropping required state,
  observations or an output consumer merely because it is absent from one test.
- No alias conflict, write-before-last-read, state version violation or lost
  device/transfer dependency. Preserve immutable prior roots and distinct
  candidates. Values retained by output/debug/graph readers stay live.
- Same legal yields/cancellation/output-credit boundaries, or independently
  qualified bounds for a larger non-preemptible region. A long fused prefill
  cannot hide an unbounded cancellation or fairness delay.

Fusion can remove intermediate storage only after recalculating **physical**
liveness. A graph, queued command, copy or peer transfer can retain a reader
after its last semantic use. A scratch slot cannot be recycled until the last
actual reader retires; cancellation or scope exit alone does not retire it.

Do not fuse freely across sampling/grammar decisions, host callbacks, external
tools, unverified model/codec state, required stream/event barriers, device or
peer boundaries, output commits, deadlines or backpressure waits. Some of these
can later be enclosed in a qualified device-native step island with explicit
control/state contracts. A label saying “pipeline” supplies no such permission.

Fewer kernels can lose through register pressure/spills, shared-memory demand,
low occupancy, poorer tensor-core geometry, redundant loaded values, serialized
independent branches or bigger graph/scratch pools. A fused MoE/shared branch
may be slower than an independently overlapped branch. One giant kernel is an
experiment, not a design goal.

Report separate quantities:

- semantic primitive/node count;
- fused region count and selected native kernel bodies;
- host submission/graph-replay count;
- actual physical kernel dispatches, copies/transfers and synchronization;
- complete task latency, goodput, memory and energy.

One graph submission does not mean one physical kernel. Static launch-site
counts and source instrumentation are not automatically a driver enumeration.

## 7. State, nesting and commit boundaries

### Inner mathematical pipeline

All subregions execute under one original attempt from the expected immutable
StateRoot. They return provisional values/state plus actual completion/status.
They can prepare domain versions, never publish them. The original coordinator
validates complete required domains and exact root/epoch/resource binding and
performs the one logical PublishRoot for that transition.

Subregion completion is not a union of child Ready flags. A multi-stream path
must join actual producers through the original service/event protocol and
certify the exact written-output cohort expected by publication. Preserve all
producer/read/copy owners across the join; an unknown child fences the affected
attempt rather than yielding a fabricated composite Ready.

Do not turn each nested node into a child session that independently publishes.
Do not use private device state updates, a Ready event or a static profile as a
substitute for the original session authority.

### Compound task pipeline

Default future mode: **explicit staged commits**, with model/codec/task state
and output frontiers identified at every legal stage/segment boundary.

- ASR can commit an accepted transcript segment. A later LLM/TTS failure does
  not pretend that accepted segment or already played audio never existed.
- LLM token state, complete TTS codec chunks and playback acknowledgments are
  distinct frontiers. Neural commit, transport delivery and playback differ.
- One trusted parent task driver schedules nested task work through the
  original coordinator/worker. Each actual session has its original sole
  writer. A subpipeline is composition of those authorized transitions, not
  an autonomous recursive publisher/scheduler.
- The parent tracks ordered immutable child outcomes and compound progress.
  A compound task frontier does not silently union unrelated StateRoots or
  grant authority over another model's session.

If a future task requires **atomic multi-model logical publication**, declare
that transaction explicitly. All participating model-domain candidates must
remain unpublished under one coordinating owner until every required Ready
and epoch/grant check passes. This needs a tested combined-root/prepare protocol
using the existing transaction seam. It is not implied by nesting or available
because existing per-model `commit` methods can be called in sequence. No
atomic rollback is promised across prior staged commits, external tool effects,
network delivery or audio playback. Do not implement this optional mode first.

Target and drafter use independent roots and source identities, with explicit
read-only trained feature/tensor sharing only where exact architecture permits.
Verifier acceptance chooses a target-correct prefix, prepares every required
domain and publishes through the target's original authority. Reject/cancel
restores or recomputes recurrence, conv, KV, draft taps, RNG/grammar and any
other participating domains. Deleting future KV positions alone is insufficient.

### Immutable definition, late live binding

Distinguish a reusable content/code recipe from its **source-owner-resolved
prepared object**. Portable artifact/code fingerprints can locate a recipe;
the live process-local SourceIdentity and retained original owner authorize its
prepared resources. An existing source-owned library cannot be rebound to a
different owner because bytes/hashes happen to match. Reprepare against that
owner when necessary; never manufacture a live issuer from a cache key.

Reuse immutable source-bound graph/code/prepared parameter definitions. Bind
actual call inputs, token/position spans, source/model owner, root/attempt,
session/graph/grant epoch, row/slot/buffer generation, reservations, stream and
completion events at the original per-call preparation boundary.
Late binding checks the same original source owner; it does not replace the
source issuer captured by prepared records or transfer leases.

The [isolated native body](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_program/native_full.rs:32)
already separates a source-owned shared library from once-bound phase state.
That is a useful direction, not general reuse proof. Fresh calls require fresh
immutable wrappers/original input uploads and ownership checks; never clear a
`OnceLock`, retag a teacher record/old Ready/result or keep borrowed prior-root
state inside a supposedly immutable reusable library.

## 8. Profiles: orthogonal selectors, resolved recipes

Keep these dimensions distinct in config, identity and diagnostics:

| Dimension | Example role | Authority |
| --- | --- | --- |
| Model semantics/source | Exact source/encoding/assembly, tokenizer/template/processor/adapter, trained sparse/multimodal behavior | Original source/model owner |
| Arithmetic | Casts/reductions/activation/state precision, phase behavior, tie/status contract | Explicit independently accepted numerical contract |
| Task/workload policy | Interactive latency, throughput, long/shared prefix, output/grammar/proposal limits | Task/session config, scheduler fairness and admission |
| Device/placement strategy | Native kernel family, graph/eager, topology, residency/IO and allowed shape bands | Prepared physical plan plus live physical arbiter |
| Observed qualification | Source/model/backend/shape/runtime/physical/quality/performance evidence and limits | Executed tests/results; cannot be enabled by a config name |

Do not collapse these into named external-engine modes or a single “fast” flag.
Keep accepted GGUF and original-HF arithmetic lanes explicitly distinct; a
source container, arithmetic policy and workload are independent dimensions.
Arithmetic profile names identify contracts; workload profile names express
intent. An approximate model policy must not masquerade as either exact path.

Suggested user-facing recipes (all proposed, subject to independent gates):

| Recipe | Pipelines/mechanisms combined | Prerequisites | Conflicts/fallback | Current status |
| --- | --- | --- | --- | --- |
| `resident-interactive` | Prepared decode region, compact requested readout/device feedback, small fair cohorts, legal local fusion, bounded replay; optional cheap proposal | Natural whole-model/source/state/runtime parity, admitted stable resources, strict cancellation/output limits | Deep drafts/large prefill can harm ITL; safe direct/smaller accepted shapes, K=0 | Existing pieces; new original arithmetic/runtime composition open |
| `resident-throughput` | True temporal prefill + independent-request batches, stable metadata, tuned projections, aged chunks/decode protection, optional graph | Accepted batch/phase numerics and ragged masks, fair admission and real temporal prefill | Larger scratch and queue delay; choose smaller accepted quanta/bands without starving cold prompts | GGUF B2/4/8 foundation real; full temporal/production replay open |
| `long-shared-prefix` | Typed paged state + complete hybrid snapshots/COW, source/tenant-authorized prefix reuse, bounded chunking | Independently qualified long context, all-domain restorable frontier, page/capture reader lifetime | KV-only hit invalid; miss/partial/common-frontier recompute, no cache reusing mismatched arithmetic | Design/open; CPU reference long-context success is not all GPU profile support |
| `memory-bounded-dense` | Authenticated weight windows, coalesced IO, bounded staging/prefetch, layer/row reuse and explicit CPU/GPU placement | Actual dense model semantics/runtime + source/cancel/IO/physical closure; bandwidth feasibility | Source reads/copies can dominate; smaller chunks or refusal within cap; no silent quant/model change | Original 27B source foundation only, trained runtime open |
| `sparse-capacity` | Source-selected experts/learned tables, typed size classes, protected hotset, demand-first prefetch, grouped rows, later target-correct draft | Qualified actual MoE/indexer/lookup architecture and every participating state domain | Dense model ineligible; speculative IO/cache borrowing can lose; no expert skipping in exact mode | Later; does not solve dense 27B |
| `derived-precision` | Explicit derived source recipe, exact codec/packing kernels, independently selected activation/KV precision | Authorized derived model/source, calibration+held-out quality, numerical and physical acceptance | Different source/quality/cache compatibility; explicit user selection/refusal, no automatic pressure fallback | Later/explicit; not a performance waiver for current parity |
| `media-interactive` | Typed ASR/language/TTS or native speech task regions, clocks, finite stream credits and barge-in | Source-real processors/codecs/media tasks and trained quality/latency/physical tests | Buffers/playing audio cannot globally roll back; stage commits and bounded fail/stop | Structural registry/config only today |
| `distributed-placement` | Same prepared regions with topology/transfer/peer dependencies, appropriate tensor/layer/expert partitions | Authenticated topology, state bridges, real links/collectives, remote completion/quarantine | Link cost/reduction topology may invalidate performance/numerics; coherent task-level abort/refusal | Later; local context/capacity does not prove remote execution |

Recipe resolution must be deterministic and finite: explicit defaults, explicit
overrides, canonical expanded values, no arbitrary inheritance/merge graph.
Initially use built-in Rust presets and a small validated config section only
when implementation needs it. Unknown/unsupported combinations refuse before
loading resources. A selected recipe cannot silently substitute a different
checkpoint, codec, arithmetic contract or model under pressure.

Nested recipe example, **illustration only; no current YAML schema change**:

```text
voice-interactive recipe
  task pipeline = conversational-voice
  ASR stage      = exact selected ASR source + its arithmetic + bounded windows
  language stage = exact selected Qwen source + resident-interactive strategies
  TTS stage      = exact selected TTS source + its codec/complete-chunk strategy
  outer policy   = staged commits + clocks + credits + barge-in + source grants
```

This is how profiles combine pipelines without conflating their models or
numerics. Each child exposes eligibility/resource/effect summaries; the parent
admits their joint peak and permitted overlap. Static named recipes carry no
Ready token, resource reservation, source grant or state-publish capability.

## 9. Examples and boundaries

### Qwen single-token, prefill and verify

```text
text task (trusted TaskDriver)
  prompt pipeline
    exact tokenizer/template -> authorized exact token spans
    prefill region [bounded chunk, true temporal path only after qualification]
      embedding
      layer sequence [nested, same canonical model schedule]
        normalization/residual region
        GDN OR full-attention region [source-defined layer kind]
        gated FFN region
      state-only output OR required final head
  decode1 pipeline [accepted shape band]
    embedding -> same nested layer regions -> final norm/head
    selected exact sampler/grammar -> actual device feedback/output candidate
  optional proposal + verify pipeline [later]
    authorized accepted-history lookup OR actual trained draft task
    target verify region [accepted verify numerics, bounded token rows]
    target acceptance + all-domain restoration/preparation
  original Ready/root/epoch/credit checks -> PublishRoot -> ordered output
```

Prefill/decode/verify reuse semantic components but retain phase-specific cast,
reduction, mask and state contracts. Independent-request B4 is not a four-token
temporal prefill or a target verify proof. Source-native layer kinds remain
explicit; the profile cannot choose GDN instead of attention.

Candidate physical paths include portable decomposition, fused residual/RMS,
projection prologue/epilogue, QK/RoPE/store or GDN local fusion, per-layer capture
and whole-step replay. A whole-model implementation can replace the layer
subpipeline only under the same closed contract and original publication owner.
State-only known teacher tokens may omit output-only head work when the task
does not request it; preserve the final readout and all state checks required
by the chosen task/numerical contract.

The source/model/numerics determine semantics; a profile selects the accepted
physical path and work budget. Sampling/grammar/cancel/output-credit/timer
frontiers remain visible unless a separately accepted device loop preserves
their behavior and finite response bounds.

### ASR → language → TTS

```text
audio input, rational sample clock
  ASR subpipeline: processor -> encoder/decoder -> complete transcript segment
  optional acoustic-first diarization / registered transcript conversion
  language subpipeline: exact context + tokenize -> Qwen task -> accepted text
  TTS subpipeline: processor/acoustic/codebook -> codec/vocoder -> complete PCM
audio delivery: credits -> ordered complete chunks -> playback ACK frontier
```

Every learned model has its own exact source and model state contract. Reuse
immutable weights/prepared code across tasks when identical and admitted;
request state stays private. Authorized committed chat context is a read-only
snapshot, revalidated at use. Different models may share authorized text/media
bytes. Token/processor outputs can be shared only when exact identity, settings,
position/format and authorization match. Matching text or “same tokenizer name”
does not authorize sharing KV, embeddings, recurrent state or arbitrary tokens.

Partial ASR hypotheses are explicitly provisional; never treat them as
committed transcript silently. Barge-in advances cancellation epochs, stops new
publication and drains/quarantines existing physical readers. Backpressure
limits all neural, codec and transport queues. Complete chunk publication does
not claim the user heard it.

### Native speech alternative

A source-qualified native audio-in/audio-out model may expose one speech task
subpipeline instead of ASR+language+TTS. It must preserve its actual learned
audio token/codebook/processor/state/clock contract and meet its own quality
gates. It is an explicitly different task/model choice, not an algebraically
equivalent fused replacement for three independently chosen models. A wrapper
can offer the same typed application boundary once separately qualified.

Cross-model feature sharing or simultaneous media/text decoding must be
architecture-defined and source-qualified; it is never inferred from enclosing
both in one process. External tools and application actions stay above the
trusted AI task boundary, with explicit input/result stages and authorization.

## 10. Failure, capture and fallback rules

| Condition | Allowed response |
| --- | --- |
| Unsupported candidate/shape, before submission and no uncertain work | Select another accepted same-contract variant after its own admission checks; otherwise refuse |
| Compile/capture unavailable with proven closed safe failure and no live/fenced resources | Separately accepted direct/eager path may run; account preparation/capture costs |
| Capture failure/partial replay/unknown device completion or fenced service | Preserve original completion ownership and quarantine; refuse work/publication; no eager/CPU retry on that service |
| Source/semantic/profile/root/epoch/generation/alias mismatch | Reject binding, never retag old state/results to fit |
| Output credit exhausted/input wait/timer deadline | Legal stage yield; bounded retention under original fair worker and wakeup protocol |
| Peer timeout/loss after possible submit | No assumption DMA stopped; retain local/remote reader ownership and coherent attempt failure; no rank-local incompatible fallback |
| Candidate state layout differs from baseline | Switch only at a compatible committed checkpoint with tested conversion/replay; do not reinterpret pointers |

Graph replay keys include source/model/numerics/phase, shape/layout, selected
kernel/library/compiler/driver/device, code/module identity, actual storage and
row/table/workspace generations, stream/placement/residency and output mode.
Reserve graph objects/native pools, metadata and captured readers before use.
Padding cannot write/advance/sample an inactive row or observe poison tails.
Capturing, queuing and encoding already pin readers; completion and graph
destruction/retirement are distinct lifetimes.

Cancellation, logical cache eviction, graph config reload or a source epoch
change revokes future eligibility, not past physical readers. Unknown completion
never creates reusable memory credit. The original arbiter/fence remains live
authority even if a profile/tuning entry says the path is supported.

## 11. Additive implementation sequence

This proposal does not move recursive runtime work onto the critical path. Use
existing owners and small verified changes; no framework-wide prerequisite.

1. **Current correctness first.** Complete .8B natural source binding, real
   original publication, native 2B/full-state/natural feedback and actual serving
   route acceptance. Keep original GGUF regression and exact arithmetic gates.
2. **Matched short baseline.** Source/profile/cap-matched C1/2/4/8 complete
   requests, offered-load traces and unprofiled paired runs. Attribute actual
   CPU prepare, copies, dispatch, model math, readout and delivery. Choose the
   largest demonstrated generic bottleneck, not a preferred abstraction.
3. **One legal region increment.** Add private region scopes/candidate metadata
   only for the first useful replacement. Keep canonical plan expansion and
   forced decomposition; close exact arithmetic/effects/lifetime and complete
   performance gates. Prepared reuse, compact readout, local fusion or bounded
   replay can land independently; do not require a generic optimizer first.
4. **Shared serving improvements.** Qualify stable metadata, real temporal
   prefill, finite batch/phase bands, production capture, aged fair chunking,
   lifetime-aware storage and compatible optional speculation in evidence order.
   Each has independent acceptance and material combination ablations.
5. **Long context/state scaling.** Independently qualify contexts, complete
   domain checkpoints/paged attention/COW/prefix restoration under exact selected
   profiles and physical budgets. CPU reference or KV-only success is bounded.
6. **Dense original 27B.** Establish source-real model/runtime correctness with
   admitted placement; the first correct path may use synchronous streamed
   windows. Then optimize bounded authenticated dense IO/copy/compute and prove
   complete state before cold/warm performance. Full residency is not a
   prerequisite. Original source is roughly 55GB versus current relative 5GiB
   device target; this is capacity/bandwidth work, not MoE hot-expert reuse.
7. **Gemma E2B → 26B-A4B → dense31B.** Add exact PLE/attention/shared-KV/
   routing/soft-cap semantics and source-real references; reuse infrastructure,
   not Qwen mathematical assumptions. Later trained drafters/quant variants need
   their own sources and gates.
8. **Then broader sparse, derived precision, media/distributed composition.**
   Source-qualified operators/tasks extend this same seam. Held DeepSeek/GLM
   payloads remain held; no model/SDK acquisition or gfx90c launch follows from
   a profile being documented.

The first profile implementation can be a resolved struct of existing accepted
choices. Add a new interface only when at least two concrete clients need the
same capability and an extension drill demonstrates less coupling. Keep
unfinished combinations visibly unqualified rather than expose many toggles.

## 12. Acceptance and extension drills

All gates below are **proposed**, not executed by this design task.

| Gate | Required proof |
| --- | --- |
| Source/reproducibility | Exact checkpoint/index/shards/codec/scales, config/tokenizer/processor/adapter, original live issuer and acquired rights; selected arithmetic/phase, device/compiler/driver/code/shape/caps. Repeat immutable binding checks; wrong/stale/foreign identities fail before access |
| Semantic/hierarchy | Flat versus nested/inlined/outlined/replaced path preserves typed ports, full operation/effect order, observations and state frontier; cyclic/oversized nesting and incompatible ports refuse without effects |
| Resident stage edges | Qualified compatible same-device producer/consumer fixture retains payload on device with zero intermediate H2D/D2H payload bytes and no stage-only host wait; separately name control/status traffic, actual copies/launches and reason for every required bridge. Incompatible codec/layout/device/grant paths explicitly convert or refuse |
| Cross-stage lowering | Forced decomposition versus resident chain, graph and fused candidates agrees under the selected contract; odd/ragged/tail, multiple-consumer/debug, alias, root/epoch and reload cases. Recalculate actual joint peak/retirement and test cancellation, backpressure and unknown completion at each preserved boundary |
| Numerical | Actual selected path, full affected tensors/logits/state and following-token continuation; adversarial casts/reduction widths/zeros/ties/tails/masks/nonfinite behavior. Original exact gates remain exact; separately selected changed profiles get independent fidelity/quality gates |
| Full model | Original .8B/2B natural trajectory and full vocabulary/all42 required domains; batch/reorder/refill and initial/continued-prefill/decode/verify only where claimed. 27B/all112 and Gemma domains separately established; no inheritance from a small-model result |
| State/cancel | Independent sessions/roots; COW forks, common-frontier miss/partial/full restore, every speculative accepted length/reject, stale epoch/grant/reload; cancel before submit/in flight/Ready-before-publish/delivery. No partial state or future-row exposure |
| Physical | Original arbiter admission before allocations/IO/submit; peak old+candidate+scratch+code/graph+metadata+host/transfer/page charges; encoded/captured/queued/in-flight readers, source mutation/short reads, real completion/fault quarantine, safe retire and final zero only where actually proved |
| Runtime | Actual original worker/CLI/HTTP route, raw IDs/text/EOS/usage/finish, slow/disconnected consumers, finite credits, real timers/wakeups, cancellation response and fairness/starvation limits under mixed prefill/decode and source epochs |
| Performance | Independent ablations then material combined profile: complete requests, repeated ABBA/BAAB or justified random order, cold/warm/compile/capture/prefix states named; offered/admitted/refused/failed/canceled/completed counts, TTFT/true ITL p50/p95/p99, SLO-goodput, fairness, memory, IO, energy, actual physical work |
| Phase promotion | Appropriate real source/state/cancel/physical tests plus the complete isolated AGENTS gate: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`. Successful compilation/mocks/docs are insufficient |

Sampling requires a declared claim: greedy exact identity, same selected sampler
stream or distribution equivalence are different tests. Natural-feedback and
serving closure cannot be replaced with forced captured trajectories. Physical
diagnostic profiling runs are distinct from primary unprofiled performance.

Required extension drills before treating the design as stable:

1. Replace one nested pure region with a fused implementation, then force its
   decomposition; unchanged source/model/task/publication, measured code churn.
2. Add one source-real stateful primitive and model layer use; show unrelated
   workers/routes/tasks need no model-name branches.
3. Add one device strategy through the existing region/device seam; preserve
   the canonical semantic schedule, source owner and ResourceArbiter.
4. Add one non-token task with typed media frontier/clock and a nested task
   stage; no fake token frontier or recursive independent publisher.
5. Add one profile combining two accepted strategies, with an intentionally
   incompatible arithmetic/state/shape candidate rejected before submit.
6. Exercise source/model reload, canceled nested task, graph-held alias reuse,
   missing full-domain prefix, peer/unknown completion and output-credit waits.
7. Compose two compatible device stages without a new host payload copy or
   stage-local allocator; qualify graph/fused alternatives, then introduce an
   incompatible consumer or required host decision and show its explicit bridge.
   Measure complete-request benefit and negative lifetime/occupancy cases.

Profile promotion means useful source-correct committed output meeting the
declared complete-task SLO under admitted physical/fairness bounds. Report
neutral/slower variants and rejected combinations. Neither fewer kernels,
higher cache hit rate, higher speculative acceptance nor isolated throughput is
a substitute for that result.

## 13. References

- [Existing composition specification](README.md) and [live queue](TODO.md).
- [Composition/extension contracts](../../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md).
- [Execution spine](../../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md).
- [Task/compiler refinement](../../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md).
- [Current batching and complete state](../serving-and-speculation/BATCHING_AND_PAGED_STATE.md).
- [Independent mechanism research and dispositions](../../research/engine-optimization-comparison-2026-10-21/README.md).
- [Mechanism-to-profile crosswalk](../../research/engine-optimization-comparison-2026-10-21/profile-coverage.md).
- [Critical review](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/composable-pipelines-profiles-design-20261021/REVIEW.md).
- [Independent source and coverage review](../../research/engine-optimization-comparison-2026-10-21/composition-review.md).

The semantic/effect/state/resource rules above stand independently of the
research corpus. No foreign runtime or source import is required by the design.
