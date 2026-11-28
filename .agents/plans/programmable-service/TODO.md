# Programmable service and execution stack

Base: `6986470e46c1cc5d0ac4601dc9163cf041d73e7b`
Status: UX-11D accepted; provider discovery landed; foundation API/protocol contracts specified; implementation not started.
Owner: @pvl; root coordinates shared contracts and the queue.

See [service/SDK design](README.md),
[composition and adaptive execution](COMPOSITION_AND_ADAPTIVE_EXECUTION.md),
[foundation phases and parallel lanes](IMPLEMENTATION_PHASES.md),
and the authoritative [API/protocol contracts](API_AND_PROTOCOLS.md).

The user set the order on 28 Nov: UX-11D → PS-F1/F2/F3 → independent
ROCm, CUDA, MCP/native attachment and capability lanes. Full MCP, generalized
composition, tuning, continuation and arbitrary-code isolation do not gate
backend parity. This TODO is the live completion authority for this effort;
backend-specific work stays in its existing owning plans.

## Committed

### `d8b55472` — Programmable stack design and developer README

Documentation batch; runtime implementation has not started.

- [x] Inspect original service/control/host, source preparation, SDK, semantic
  program, state/resource/capture and persistence boundaries.
- [x] Specify tagline, reusable Service/Client, native MCP, simple SDK,
  live revisions, experiments, observation and continuation.
- [x] Define typed task/model authoring, execution mappings, SDK/MCP parity
  and workload-driven qualified policies.
- [x] Record accepted design direction and next priority after UX-11D.
- [x] Split shared foundations from independent implementation lanes, including
  SDK-hosted MCP, backend independence, owners and concrete phase exits.
- [x] Propose concise README positioning, quick SDK/live-revision examples and
  subtle branding changes in [README proposal](README_PROPOSAL.md).
- [x] Apply the explicitly authorized root README and architecture SVG update:
  quick SDK, live revisions, supported fusion and expandable typed kernel chain,
  current support/roadmap and subtle live-programmable positioning.
- [x] Lead with installed-binary `zerogravity` commands and add authentic native
  terminal/Web snapshots; link the portable source-install path while the first
  public binary release remains pending.

### `06ce454f` — UX-11D exit and initial foundation preparation

- [x] Reconcile accepted UX-11D package, physical launch and assembled checks
  recorded at 683ac065; retain packaging fixes and normal dependency acquisition.
- [x] Prepare the original host/source/receipt extraction boundaries and leave
  runtime implementation to the separately assigned foundation owner.

### `f4e4d76b` — Provider handoff and PS-F1 preparation

- [x] Audit the incoming native discovery/configuration boundary, current
  frontend-reaped operations, original receipts and existing qualification
  coverage. Record the extraction map and five bounded acceptance scopes in
  [PS-F1 implementation entry](IMPLEMENTATION_PHASES.md#ps-f1--reusable-service-and-simple-sdk).

### `be6f07cb` — API/protocol contracts and Rust implementation policy

- [x] Specify native inputs/results and builder/run/stream ownership; distinguish
  acquired sources, immutable revisions, live references and durable commands.
- [x] Define typed authoring/activation, native HTTP/Unix attachment and optional
  SDK-hosted rmcp projection, independent versioning and bounded observations.
- [x] Align SDK/composition examples and phase ownership; make library reuse,
  practical zero-cost design and proportionate end-to-end validation explicit.
- [x] Reconcile landed provider discovery at fda84c16: native source configuration,
  browse/repository pages/cursors and existing qualification. Broader discovery
  adapters remain independent; acquisition/eligibility retain original owners.

### Resource scheduling, paging and cache integration — this documentation commit

- [x] Reconcile the new memory/serving design with native Service/SDK/MCP:
  one resource facade, existing configuration/operation/continuation owners.
- [x] Specify unique backing/credit accounting, real reclamation, bounded
  operation demand and enforceable progress under constrained concurrency.
- [x] Define portable cache descriptors, admitted asynchronous transfer/
  publication/retirement and the full-continuation boundary for live parking.
- [x] Keep F1/F2/F3 bounded; common memory work and independent backend
  consumers follow that exit without MCP, optimizer or all-backend gates.

## Uncommitted

### Documentation and prerequisite

- [ ] Retain API-preview labels until the examples' actual public SDK and
  execution capabilities are qualified; remaining package/Web/About metadata
  proposals are separate from the applied README and diagram.

**PS-F1 entry — API/protocol contracts prepared (28 Nov):**

- [ ] Coordinate current control/preparation/lifecycle file ownership before
  extraction; read the latest source and retain the ongoing GPU lifetime fixes.
  Reuse applicable owner checks without another provider/backend matrix or a
  historical source freeze.

Start with current AGENTS.md, CURRENT_HANDOFF, this checklist, service design
§§2–4 and 12–14, the [API contract](API_AND_PROTOCOLS.md), and the linked PS-F1
entry. The provider design and acceptance remain in [model acquisition](../model-acquisition/TODO.md) and
[CLI source discovery](../cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md).
Source-provider definitions already live in the SDK facade. Reuse them and
move their orchestration into Service; retain execution-provider contracts
in their original layer. No separate provider registry or UI-private SDK.

This preparation changes plans only. All PS-F1 implementation items remain
open; the next implementation unit is embedded Service/Client plus the
migrated frontends and actual load/run/stream/serve/shutdown acceptance.

### PS-F1 — Reusable Service and simple SDK

Contract: [public SDK](API_AND_PROTOCOLS.md#3-public-sdk). Apply the shared
[Rust reuse/performance](API_AND_PROTOCOLS.md#2-rust-design-and-library-reuse)
and [proportionate validation](API_AND_PROTOCOLS.md#8-validation-without-process-overhead)
rules throughout the implementation; each checkbox is part of the coherent unit.

- [ ] Extract host plus shared configuration, accepted source discovery,
  source selection, explicit acquisition and resident preparation together;
  reuse existing discovery/configuration/cache authorities and keep UI glue local.
- [ ] Give Service independent management supervision/reaping, operational
  event/sink ownership and reliable joined shutdown.
- [ ] Provide Service owner and cloneable direct Client with exact source,
  registration, operation, listener and client lifetimes.
- [ ] Expose pull/load/run/stream/serve/inspect over original operations; preserve
  effective settings, explicit acquisition and actionable native errors.
- [ ] Use original committed outputs/usage/results with explicit usage scope;
  preserve reliable terminal cancellation/worker-loss outcomes and one drained
  receipt across run/stream/start modes, without bounded UI copies as results.
- [ ] Provide wakeable async waits and a shared blocking driver without token
  scheduling duplication, UI polling dependence or per-call async runtimes.
- [ ] Migrate CLI/TUI/Web to the same service behavior; embedding installs no
  process-global logging, signals, socket or UI as an implicit side effect.
- [ ] Complete the real embedded SDK → run/stream → shared serving → shutdown
  exit. Sequential live registration remains required in PS-F3.

### PS-F2 — Programmable contracts, records and observation

Contract: [typed authoring](API_AND_PROTOCOLS.md#4-programmable-authoring-and-live-control)
and [commands/results/observations](API_AND_PROTOCOLS.md#5-operations-errors-and-observation).

- [ ] Add distinct task/model/mapping drafts, scoped IDs, atomic bounded edits,
  validation, immutable freeze/derive and exact prepared references.
- [ ] Exercise existing source/model definitions, a serial pipeline and one
  actual closed-region preparation; unsupported general edits refuse precisely.
- [ ] Preserve canonical ModelProgram/ForwardPlan, complete ports/shapes/source/
  parameter/numerical/state/effect/observation contracts and original binding.
- [ ] Expose owned native operation/result schemas, capabilities and typed
  errors through Client; borrowed native interfaces remain valid Rust APIs.
- [ ] Establish adapter grant context and Service-side mutation/private-detail
  enforcement without introducing a general identity-management framework.
- [ ] Atomically deduplicate scoped command intent before resolving floating
  inputs/defaults; retain resolved recipe and operation before durable dispatch.
  Distinguish attempt identity, ephemeral/durable acknowledgement, current grants
  and bounded result/deduplication retention.
- [ ] Persist immutable recipes/artifact references before durable acknowledgement;
  reuse atomic storage and add an index only for actual consistency needs.
- [ ] Reconcile service/domain epochs, interrupted and completion-unknown work;
  affected admission remains fenced until original trusted retirement proof.
- [ ] Keep queryable terminal results separate from bounded notification delivery;
  define expiry, loss/gaps, clocks, units, correlation and field provenance.
- [ ] Establish basic compact counters, optional observation budgets and separate
  raw-input/tensor/output capture choices.
- [ ] Establish the [resource facade](API_AND_PROTOCOLS.md#45-resource-policy-and-cache-control)
  using current native inspection/controls; distinguish saved defaults, desired/
  applied policy, advisory demand and actual original admission. Unsupported
  elasticity/tiering remains explicit; no allocator implementation in F2.
- [ ] Establish checkpoint envelope/compatibility contracts using existing codecs;
  distinguish records/rerun from resumable task state.
- [ ] Establish recognized primitive/artifact/provider extension boundaries;
  implement worker IPC only with the actual PS-X consumer.
- [ ] Complete native Client/adapter conformance, supported draft preparation,
  idempotent recovery and truthful unsupported-capability acceptance.

### PS-F3 — Live ownership and common foundation exit

Contract: [references and activation](API_AND_PROTOCOLS.md#4-programmable-authoring-and-live-control).

- [ ] Implement bounded live add-registration through original catalog/admission;
  loading B preserves registered A, its source owner and existing sessions.
- [ ] Atomically publish exact new registrations only after preparation;
  failed unpublished work retires through original owners.
- [ ] Add generation-guarded route/pipeline activation for an existing qualified
  prepared revision and exact public alias retirement.
- [ ] Retain all original owners for an admitted composition/session, including
  later-stage models; admit compute workspace at actual stage boundaries.
- [ ] Account old/new coexistence, fail preparation without damaging active
  service, and expose explicit drain/reprepare when coexistence cannot fit.
- [ ] Preserve capture/cache/buffer/queued-command lifetimes through actual
  completion/quarantine and original per-session StateRoot publication.
- [ ] Apply generation/lifetime rules to supported resource-policy changes;
  lower-cap drain preserves spendable admitted guarantees/progress credit.
- [ ] Persist activation intent separately from observed generation; reconcile
  crashes/retries without duplicate publication or invented device completion.
- [ ] Qualify supported existing checkpoint compatibility as a concrete consumer;
  leave general migration/full task resume in PS-S.
- [ ] Complete load A → run A → load B → direct/serial task → serve workflow,
  region inspection, retained-stage change, conflicts/refusals and rerun.
- [ ] Accept the assembled foundation on its verified scope, then unblock the
  parallel lanes immediately; no full MCP or performance parity gate.

### Parallel backend lanes — existing owners

- [ ] Resume retained ROCm P3C work after the foundation exit through
  [CPU/GPU optimization](../cpu-gpu-optimization/TODO.md) and
  [model execution](../model-execution/TODO.md); preserve original evidence.
- [ ] Run CUDA correctness/parity independently after the same exit through
  those existing backend owners and [serving](../serving-and-speculation/TODO.md).
- [ ] Coordinate shared source/build/device capacity; neither backend's result
  nor MCP completion is a prerequisite for the other backend's work.

Memory/serving work follows the [shared dependency split](../memory-and-streaming/ELASTIC_CONTEXT_AND_RESIDENCY.md#7-delivery-and-acceptance):
one common accounting/pool/scheduler owner, independent HIP/CUDA direct
consumers, then integrated elasticity on each ready backend. Basic bounded
scheduling does not depend on PS-E; live parking needs its PS-S continuation
scope. Concrete implementation items stay in the owning memory/serving TODOs.

These are scheduling/integration items, not duplicate backend acceptance lists.
Vulkan and optional compiled providers keep the later priority in the full plan.

### PS-M — SDK-hosted MCP and native attachment

Contract: [MCP projection](API_AND_PROTOCOLS.md#7-mcp-projection-m1)
and [native attachment](API_AND_PROTOCOLS.md#6-native-attachment-protocol-m2).
Use optional rmcp and existing HTTP/async/serde libraries; no custom framing or
CLI-private SDK. M1/M2 remain independent over the native foundation.

- [ ] M1: implement optional MCP host directly over Client, usable by embedded
  SDK applications without CLI/daemon dependency.
- [ ] M1: expose supported capabilities, model/control/authoring/inspect operations,
  resource inspection/policy/settings/maintenance, operation get/cancel/results
  and bounded observations with native grants. Missing memory/tier capabilities
  do not block MCP on already supported operations.
- [ ] M1: negotiate supported core/extension features; ordinary polling works
  without optional Tasks/Events support.
- [ ] M2: implement authenticated versioned native endpoint/Client transport,
  exact service epoch/identity and bounded subscriptions.
- [ ] M2: add foreground headless service and serialized attach-or-embed;
  auth/version/busy/live-owner errors cannot trigger a second Runtime.
- [ ] M2: preserve explicit embedded/connect modes, service-relative paths and
  explicit upload/import when client-local data must cross the boundary.
- [ ] Integrate M1/M2: equivalent supported embedded/attached SDK and MCP
  operations, simultaneous clients, reconnect and correct host shutdown.
- [ ] Keep public inference, browser-safe control and privileged experiments
  separate under the common native authorization rules.

### PS-C — task and model composition

- [ ] C1: simple chain plus bounded typed acyclic task graph; terminal text
  fan-out/join/format and final streaming over original admitted model calls.
- [ ] C1: typed edge identities/capacities/backpressure/context checks, independent
  stage sessions, legal cancellation and staged publication.
- [ ] C2: source-bound authored model with recognized primitives/reusable regions,
  actual parameter roles/numerics/state and one generic catalog/load adapter.
- [ ] C2: generalize canonical values/outputs/state only for the concrete authored
  model; no second graph interpreter or canonical executable IR.
- [ ] C2: validate intended model against independent specification/reference,
  then candidate lowering and all affected state against accepted semantics.
- [ ] Joint exit: authored model in a useful task graph, through SDK first and
  MCP projection when available; C1 and C2 can progress independently.

### PS-K — execution mapping and transforms

- [ ] K1: extend requested/prepared/executed many-to-many provenance with actual
  artifacts, placements, lifetimes, dependencies, bridges and unavailable detail.
- [ ] K1: explicit typed resident kernel chains/layout/capture recipes through
  original providers and prepared ownership.
- [ ] K2: execute one supported replacement covering at least two canonical
  semantic operations, preserving all escaping values/state/observations.
- [ ] K2: preserve finite-precision rules, post-mapping physical liveness and
  qualified cancellation boundaries; compare the connected plan.
- [ ] Distinguish source operations/submissions/dispatches, logical/measured bytes
  and fused-region timing; never invent missing internal measurements.
- [ ] Save/rerun actual mappings. Unsupported requests stay explicit; a slower
  correct primitive can remain reusable without production selection.
- [ ] Work on existing models immediately; new-definition mapping waits only
  for its particular C2 model, not generic task graph/MCP/parity completion.

### PS-E — profiles, experiments and policies

- [ ] E1: record supported workload observations with exact/sampled/aggregate/
  unavailable provenance; metadata profiles are not exact replay inputs.
- [ ] E1: authorized replay or declared synthetic workloads, full offered-work
  outcomes, correctness/state, latency/goodput/fairness and resource costs.
- [ ] E1: bound trials, compilation, time, retained artifacts/traces, workspace,
  concurrent trials and live-serving interference.
- [ ] E1: persist recipes/attempts/results/lineage and truthful no-gain outcomes;
  initial comparisons use existing qualified variants.
- [ ] E2: finite prepared policies with actual phase/cohort/shape/layout/state/
  target/resource guards; common allowed variants for cohort members.
- [ ] E2: pin session policy membership; qualify transitions and pre-submit
  baseline fallback while retaining submitted failure/retirement behavior.
- [ ] E2: reuse F3 activation/coexistence and account retained code/captures/
  parameters plus peak concurrently live scratch/state.
- [ ] E2: actual guarded selection, capacity/no-match, old sessions, generation
  conflict and bounded observation on a declared trial route; no speedup required.
- [ ] Keep production policy unchanged without qualified gain. Add background
  bounded retuning with hysteresis only as a consumer of this same loop.

### PS-S — qualified continuation

- [ ] Extend qualified neural checkpoints with task/RNG/input/output/session
  frontiers and explicit source/backend/layout compatibility.
- [ ] Implement supported continue/resume/rerun/fork using original state and
  durable command recovery; preserve emitted versus consumed output.
- [ ] Qualify a real single-model stop/restart/continue first; add concrete
  pipeline stage/edge continuation when C1/C2 contracts support it.
- [ ] Add state migration only for actual useful qualified pairs; no universal
  restoration claim from saved recipes or metadata.

### PS-X — isolated extension

- [ ] Isolate bounded compilation/artifact intake and a real native trial.
- [ ] Implement parent-issued resource envelopes, actual IPC admission/supervision/
  completion/retirement with the first resource-consuming child.
- [ ] Preserve unavailable/quarantined domains on unknown completion; child
  process_shared and process death cannot mint/release global resource credit.
- [ ] Qualify an honest dedicated/maintenance GPU scope first where needed;
  shared-device and GPU-fault containment claims require their own acceptance.
- [ ] Execute a newly supplied known-contract implementation, record checks and
  benchmark results, and preserve its isolation during ordinary preparation.
- [ ] Separate numerical/performance qualification from trusted in-process
  promotion or live activation; enforce the explicit trust boundary.
- [ ] Extend missing semantics/providers by controlled normal build/host
  replacement unless a real supported extension ABI exists.

## Tests and verification

The initial README update passed link/anchor, Markdown, SVG and visual checks.
The release-focused follow-up replaces checkout-relative commands with the
installed executable, verifies command/build scope against current help and
recipes, and captures actual terminal/Web catalog views. Publication checks passed
for 210 relative links/anchors across 11 Markdown files, Markdown structure,
SVG XML, both PNGs and whitespace; the diagram and screenshots were visually
reviewed. No runtime build or GPU qualification is implied.

During implementation, extend these shared scenarios; API rows are not separate
test/fixture/evidence gates. Reuse accepted unchanged-path evidence, combine
related behavior, and add narrow checks only for concrete unresolved risks.
Performance checks target real hot paths/workloads, not every accessor.

- [ ] F1: original source/output/usage and request/owner shutdown semantics across
  embedded SDK and migrated native clients.
- [ ] F2: bounded drafts, semantic/source/state rejects, adapter grant denial,
  command conflicts/durability, expiry and observer gaps.
- [ ] F3: sequential loading, complete admitted task pinning, live activation,
  coexistence refusal and original reader/capture/physical retirement.
- [ ] F3: crash windows around accepted commands/activation receipts and exact
  dirty-domain fencing; eligible recorded rerun with fresh bindings.
- [ ] Foundation final assembled repository gate and required affected physical
  checks; retain failed/interrupted/unavailable scopes explicitly.
- [ ] M: embedded/attached/MCP parity for supported native operations, bootstrap
  races, auth/version refusal, disconnect and process-owner exit.
- [ ] C: real authored-model reference/state acceptance, task edge/backpressure/
  session isolation, bounded fan-out/join and failure publication.
- [ ] K: actual multi-operation replacement numerical/state/escaping-output
  acceptance, liveness/fault behavior and explained physical mapping.
- [ ] E: complete workload denominators/measurement boundaries, budget/observer
  costs and actual variant/transition selection independent of speedup.
- [ ] S: actual qualified continuation and corrupt/stale/incompatible refusal,
  committed output/input frontiers and original publication.
- [ ] X: actual artifact/worker execution, budgets, isolation, cancellation,
  unknown completion and original physical authority for advertised scope.
- [ ] Joint native SDK/MCP author → compose → map → measure → activate → save/
  reopen/rerun scenario once the corresponding lanes are available.
- [ ] Complete scripts/check.sh on each accepted assembled implementation batch
  as required by current AGENTS.md; avoid repetitive per-helper gates.
