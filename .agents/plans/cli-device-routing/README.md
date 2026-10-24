Author: Project owner
Co-authors: Astra HIGH, ROOT coordinator
Owner: @pvl

# CLI and device routing

**Implementation authorized · parallel worktree lane · 24 October 2026.**
The user explicitly requested implementation and merge into main. This contract
still grants no unimplemented model/backend/composition support. Current work,
source-form additions and applicable acceptance remain in [TODO.md](TODO.md).

## Outcome

`zerogravity run <model>` should choose a qualified route using resources
available **now**:

1. Prefer a complete comfortable fit on NVIDIA, then AMD, then another qualified
   GPU. Consider all eligible single GPUs before splitting the model.
2. Otherwise place as many legal layer groups as feasible on the preferred GPU,
   then the next GPUs, then CPU. Require a qualified complete composition.
3. With no usable GPU route, use a qualified CPU route if CPU is allowed and
   host resources suffice; otherwise explain why execution cannot start.

“Best” here means this explicit preference and fit policy, not a claim that a
vendor always benchmarks faster. Never assume an occupied card is free, change
the source/quant/arithmetic to make it fit, or infer execution from a model name.
Future NPUs and remote peers participate through the same placement machinery,
only after their source/state/transport/physical capabilities are qualified.
`run <config>` selects the same request from a configuration; selector-free
`run` loads the default configuration or reports an explicit missing-config error.

## Current source and boundaries

The source review found:

| Current seam | Present behavior | Proposed change |
| --- | --- | --- |
| `zerogravity-cli/src/main.rs`, `Command::Run` | CPU GGUF path; GPU commands have a separate argument path | Normalize commands into one shared library request |
| `GpuBackend::device` | CUDA/HIP ordinal 0; Vulkan AMD/NVIDIA mapped to integrated/discrete ordinal 0 | Resolve actual enumerated identity and vendor; no vendor inference from device class |
| `zerogravity-gpu/src/physical_service.rs` | PCI/LUID/registry deduplication; `max_memory()` checked against explicit operator/provider caps | Preserve identity/caps; add separately qualified current-availability discovery |
| `ResourceArbiter::process_shared` | Original shared ledger; immutable matching limits, transactional single-domain admission, fences | Reuse for every chosen domain; extend original admission for a domain bundle where needed |
| Targets and prepared execution adapters | Metadata-only/catalogued models can remain unqualified; concrete source paths have separate capability limits | Derive eligibility from executable source/backend/profile tuples, not catalog flags |

Live free-memory discovery, generic multi-device placement, mixed-backend layer
execution, remote execution and NPU inference are **not established by these
seams**. CPU support for a family also does not establish a CPU spill bridge
for a GPU instance. The local Renoir/gfx90c launch hold applies to the physical
device across **all** backend views: choosing Vulkan instead of HIP cannot
bypass it. Inventory may report the held device; execution refuses until explicit
hold removal. Current work in other plans proceeds independently.

Keep one Rust mathematical engine, immutable source identity and selected math
profile, independent session StateRoots, the original physical ledgers/services,
fair workers and genuine completion/quarantine. This plan adds selection and
physical lowering; it adds no second dispatcher, allocator or state publisher.

## CLI contract

Canonical command:

```text
zerogravity run [MODEL|CONFIG.yaml] [model/task/graph options]
    [--config CONFIG] [--model MODEL]
    [--backend auto|cpu|cuda|rocm|vulkan|REGISTERED_BACKEND]
    [-d|--device auto|DEVICE[,DEVICE...]]
    [--headroom PERCENT|BYTES] [--explain] [--dry-run]
```

- `--backend auto` and `--device auto` are defaults.
- `run-cpu`, `run-cuda`, `run-rocm`, `run-vulkan` are aliases for `run` with
  that backend. Future `run-<x>` names resolve only registered backend IDs;
  they do not load arbitrary executables/plugins. `rocm` is the public name
  for the existing HIP provider.
- One parser and identical model/task flags apply to every alias. An alias
  plus the identical `--backend` is accepted; a different value, including
  `auto`, is an error. Repeated backend values must agree.
- Preserve existing `run-vulkan-amd`/`run-vulkan-nvidia` as deprecated aliases
  for `--backend vulkan` plus a real vendor eligibility filter. They must not
  preserve the incorrect integrated/discrete selection heuristic.
- A forced backend constrains **every neural stage**, including target,
  encoders, auxiliary heads, MTP and a separate drafter. Required CPU tokenizing,
  orchestration, compilation and transfer staging are host support, not neural
  fallback. `run-cpu` therefore means all neural computation on CPU.
- Forced GPU backends may use several allowed GPUs of that backend only when
  that complete split is qualified. No CPU neural spill. Vulkan can address
  eligible vendors, but a shared API alone does not qualify a cross-card split.
- Missing backend/build support, model eligibility, compatible device or
  sufficient resources fails explicitly. Never silently rewrite a forced run
  into CPU execution or a different backend.
- Resolve source from an explicit local path or already acquired immutable
  cache reference. A catalog slug must resolve to one locally available exact
  artifact/variant or refuse with acquisition guidance. No automatic download,
  implicit quant choice/conversion or model substitution.

### Configuration selection and one schema

The current `pipeline.rs` accepts bounded YAML `zerogravity.pipeline.v1` and
compiles `model`, `pipeline`/`step`/`link`, and `route` into one typed graph.
It currently rejects mixing `--config` with graph flags, and graph execution
remains unimplemented. The following override/run behavior is a **proposed
versioned extension**, not something accepted by that parser today.

Use `schema: zerogravity.pipeline.v2`: extend that same document/compiler with
an optional `run` block and source-bound model options. Do not create a separate
run-config engine. `model[]` is the sole instance table for direct and graph
runs; `run.entrypoint` is exactly one of `{model: ID}` or `{pipeline: NAME}`.
A direct entrypoint uses the existing prepared model adapter; absent `pipeline`
and `route` lists default to empty in v2. Current v1 `Config::validate` requires
nonempty pipelines: implementation must make this validation entrypoint-aware
and lower a direct selection to the same canonical single-model execution
region, not pass an empty v1 graph as already validated. All source/task/port
and policy checks still apply. A graph entrypoint
uses the same canonical graph and qualified recursive regions when implemented.
No second model declaration belongs under `run`, a route or a nested stage.
An explicit direct CLI model normalizes to one synthesized `main` instance and
its model entrypoint; source metadata resolves its actual family through the
same adapter. This is normalization, not a second instance catalog.
Existing v1 graph documents retain structural-check compatibility and normalize
through this compiler, but need explicit `--entrypoint pipeline:NAME` to run.
Never infer an entrypoint from the first model, sole pipeline or route order.

Selection is deterministic:

- `run FILE.yaml` or `FILE.yml` selects configuration, regardless of content.
  All other positional arguments select models; there is no content sniff or
  magic model-payload scan to guess configuration. Use `--config PATH` for
  other config filenames and
  `--model PATH_OR_REF` to force model interpretation, including a `.yaml` name.
  A selected config with malformed content or an unsupported/malformed schema
  version is a config error, never a model. An extensionless config therefore
  requires `--config`; this explicit classification does not depend on version.
- At most one positional argument; it conflicts with either explicit selector.
  `--config PATH --model SOURCE` is deliberately valid: select that config and
  override the source of its **direct model entrypoint**. It refuses a graph
  entrypoint; graph models are changed by named declarations, not a global source.
- With no explicit source/config selector or graph definition flags, `run`
  (also `run-cpu`, etc., with optional task flags) reads exactly
  `$XDG_CONFIG_HOME/abird/zerogravity/config.yaml` when that XDG base is absolute;
  otherwise use `$HOME/.config/abird/zerogravity/config.yaml`. Require a valid
  absolute HOME fallback. No cwd/parent search, alternate filenames, implicit
  creation, empty-config success or automatic acquisition.
- Missing default: name the expected file and suggest explicit `--model` or
  `--config`. Unreadable/invalid default: report that error without searching
  elsewhere. An explicit model/config/graph request never overlays a default
  config. Help/version do not read configuration.
- CLI file paths resolve against invocation cwd; paths supplied by a config
  resolve against the directory of the selected config path, not cwd. Apply
  that origin to model files, sidecars and diagnostic output destinations before
  merging. Cache/artifact IDs and device/host selectors are not filesystem paths.
  A config symlink does not change its selected-path base to the target directory.

Repurpose `run --config` exclusively for the run document. Current `main.rs`
uses it for model metadata JSON; migrate that option to `--model-config`, with
`model[].model_config` in v2. Keep `--tokenizer` as the paired source option.
This is an explicit pre-0.1 CLI change: an old model-JSON argument gets a targeted
`use --model-config` diagnostic, never content-based reinterpretation as an
override or fallback. The old positional-model-plus-`--config` combination is
also rejected with that migration hint. No runtime change is made by this draft.

### Semantic parity and precedence

Every semantic flag component has a typed config field. Reuse the
existing graph declarations rather than storing command-line strings in YAML:

| CLI component | v2 config component |
| --- | --- |
| Direct model/source, `--model-config`, `--tokenizer` | Referenced `model[]` entry: `target`, `source`, `model_config`, `tokenizer` |
| `--backend`, `-d/--device`, `--headroom` | `run.backend`, `run.device`, `run.headroom` |
| Prompt/context/output bounds, max tokens, raw, teacher IDs, prefill chunk | Typed `run.task` fields: `prompt`, `context`, `max_tokens`, `raw`, `teacher_ids`, `prefill_chunk` |
| Selected numerical profile, speculation, draft-device constraint | `run.numerical`, `run.speculation`, `run.draft_device` |
| Explain, dry-run, `--audit-logits-jsonl`, `--audit-layers-jsonl` | `run.diagnostics`: `explain`, `dry_run`, `audit_logits_jsonl`, `audit_layers_jsonl` |
| Named model, pipeline name/input/output, steps, links, routes and their existing options | Original `model[]`, `pipeline[]`, `step[]`, `link[]`, `route[]` typed fields |
| `--entrypoint model:ID` or `pipeline:NAME` | `run.entrypoint` tagged selection |

Help/version, positional interpretation and `--config` file selection are
invocation controls, not recursively configurable fields. Config cannot select
another config. Existing model/task option validation (for example teacher-ID
count versus max tokens and required audit output) applies identically to both
frontends. Built-in source-specific numerical defaults are resolved explicitly;
an omitted field never invents a newly qualified arithmetic profile.

Keep `pipeline --model ID=TARGET[@SOURCE]` semantics in that existing subcommand.
For `run`, reserve `--model SOURCE` for explicit direct-source selection;
use repeatable `--define-model ID=TARGET[@SOURCE]` for graph model declarations,
alongside existing `--name/--input/--output/--step/--link/--route` graph syntax.
Both declaration flags lower to the same `model[]` entries. Graph flags address
the explicitly selected/named pipeline; conflicting names or an ambiguous scope
refuse. Richer future recursive graph fields must receive equivalent typed CLI
bindings in that compiler, not a separate routing language.

Resolve **explicit CLI > selected config > built-in defaults**, recording the
winning value and origin for explain/dry-run. Do not pre-fill parser defaults:
absence differs from explicit `false`, `0`, `""`, `[]` or `auto`. Preserve an
explicit value and validate its field semantics: empty prompt can be valid;
zero output tokens or an empty device list can refuse, never mean “use default.”
Booleans accept `--flag=true|false` as well as positive shorthand, so config
`raw: true` or `dry_run: true` can be explicitly disabled. `null` is rejected
unless the typed field expressly defines it; it is not a universal reset token.

- Scalars replace their config counterpart. Alias backend is an explicit CLI
  value: `run-cuda --config x.yaml` equals `run --backend cuda --config x.yaml`
  even if config backend is ROCm. Two contradictory explicit CLI backend values
  still fail. Then validate the result: an AMD-only config device list conflicts
  with either CUDA invocation and fails before preparation.
- Repeated CLI `-d` values append **within the CLI occurrence list**; if present,
  that complete list replaces the config device list, never appends to it.
  Other ordered lists use the same replace-whole-list rule.
- Graph CLI declaration lists replace the corresponding whole list at their
  declared scope: `--define-model` replaces `model[]`; `--step/--link` replace
  selected pipeline lists; `--route` replaces `route[]`. Unmentioned lists and
  scalar fields remain from config. No implicit by-ID deep merge; validate all
  remaining references and reject dangling IDs. Explicit empty list spelling
  `--define-model=[]`, `--step=[]`, `--link=[]`, `--route=[]` clears only that
  list and cannot mix with ordinary occurrences; normal graph validation follows.
- Unknown schema versions/keys, duplicate YAML keys/instance IDs, ambiguous
  entrypoints and contradictory flags refuse. A model-specific hard backend or
  role constraint, if introduced, must be a distinct typed constraint rather
  than an ordinary default that unexpectedly defeats precedence.

Retain the current loader's bounded single-document YAML, scalar anchors and
shallow mapping merges, with duplicate/recursive/oversize rejection. No TOML,
custom tags, includes, environment/shell interpolation, executable constructors
or external resolvers. Loading config is read-only. Complete pure validation of
the merged document, all typed references and the selected entrypoint precedes
model preparation, allocations, compiler work, audit writes or network effects.
Parsed routes are inert declarations: `run` executes its selected entrypoint,
never opens every configured route. Endpoint binding remains an explicit serving
operation with its own authorization. Parse/load/validation errors do not pull
artifacts or fall back to another config/model.

### Proposed YAML examples

Both examples are valid **proposed v2** documents, not runnable by today's v1
parser. Sources still require independent qualification and actual local files.

```yaml
schema: zerogravity.pipeline.v2
model:
  - id: assistant
    target: qwen3.5-2b
    source: ./models/model.gguf
run:
  entrypoint: {model: assistant}
  backend: auto
  device: [gpu:0, cpu:0]
  headroom: 10%
  task: {prompt: "Hello", max_tokens: 64, raw: false, prefill_chunk: 1}
  speculation: "off"
  diagnostics: {explain: true, dry_run: false}
```

Graph reuse keeps one model instance, with an explicitly selected pipeline:

```yaml
schema: zerogravity.pipeline.v2
model:
  - id: assistant
    target: qwen3.5-2b
    source: ./models/model.gguf
pipeline:
  - name: chat
    input: text
    output: text
    step: [{id: reply, kind: llm, model: assistant}]
    link:
      - {from: input.value, to: reply.text}
      - {from: reply.text, to: output.value}
  - name: summarize
    input: text
    output: text
    step: [{id: summary, kind: llm, model: assistant}]
    link:
      - {from: input.value, to: summary.text}
      - {from: summary.text, to: output.value}
route:
  - {path: /v1/chat/completions, protocol: openai-text, pipeline: chat}
run:
  entrypoint: {pipeline: chat}
  task: {prompt: "Hello", max_tokens: 64}
  diagnostics: {dry_run: true}
```

`run chat.yaml --dry-run=false --prompt "Hi"` replaces those two fields;
it selects `chat` and does not bind the declared HTTP route. Reusing `assistant`
does not duplicate weights or share mutable request roots. Multiple named models
and future recursive qualified graph regions use this same instance table.

### Device references and ordering

Use `[HOST/]SELECTOR`, with omitted host meaning `local`:

| Example | Meaning |
| --- | --- |
| `local/pci:0000:01:00.0` | Physical PCI device on this host |
| `local/luid:ENCODED_ID`, `local/registry:ID` | OS physical identity where applicable |
| `gpu:0`, `gpu:1`, `npu:0` | Convenient class index in the current inventory snapshot |
| `cpu:0` | Host CPU compute domain backed by the host memory domain; not CPU core 0 |
| `lab/pci:0000:41:00.0` | Future explicitly enrolled host alias and its physical device |

Physical selectors are stable within their OS/host identity scope; PCI addresses
can change after topology changes and must be revalidated. `HOST` resolves to
an authenticated host identity, not a free-form address or SSH command. Backend
ordinals remain internal snapshot-scoped bindings shown in diagnostics; never
persist them as physical identity or infer hardware from an adapter name.

`-d gpu:1,gpu:0,cpu:0` is an ordered **allowlist**, not a suggestion:

- Repeat `-d` to append in encounter order; comma lists and repeated flags have
  identical normalization. `auto` must stand alone. Empty/unknown selectors fail.
- Resolve every selector first. An explicitly incompatible device fails the
  request, even if another listed device would work. Thus `run-cuda -d AMD_ID`
  fails before model preparation, compilation or model/device allocation.
- Deduplicate aliases of the same host/physical device, preserving its first
  position. One GPU visible through CUDA and Vulkan is not two layer targets,
  two free-memory pools or two reservations. Retain its backend candidates.
- Scan the ordered GPUs for a **whole-model fit first**; order does not require
  splitting onto the first card when a later card fits the full model.
- If no GPU fits, the order is the layer-spill preference. `cpu:0`, if supplied,
  is the terminal fallback; reject CPU before another device. Use `-d cpu:0`
  or `run-cpu` to request CPU alone. Multiple CPU domains/remote CPU stages are
  a later explicit composition, not this initial spill syntax.
- Explicit GPU-only lists prohibit CPU compute fallback. Omitting `-d` enables
  local qualified GPUs in automatic order and local CPU last. Forced backend
  filtering removes CPU spill from the automatic list; explicitly listing CPU
  with a forced GPU backend is a contradiction and fails.
- No unlisted local or remote device is appended. An unavailable listed device
  is reported as unavailable; malformed, missing or backend-incompatible
  explicit references fail rather than silently changing user intent. A valid
  but busy device may be skipped for a later allowed route.

Automatic GPU order is NVIDIA, AMD, other qualified GPUs. Within a vendor tier,
use fresh applicable measured request cost (queue + preparation + prefill +
bounded decode) when comparable; otherwise use stable physical-ID order and
label cost unknown. On one card, rank qualified backend candidates by applicable
measured cost; deterministic ties prefer CUDA on NVIDIA, ROCm on AMD, then
Vulkan. Vendor preference is policy; eligibility and measured performance are
separate facts. Explicit list order overrides vendor order.

### Diagnostics

- `zerogravity devices list [--json]`: local read-only inventory, physical and
  convenient IDs, vendor/class, backend availability, memory domains, current
  observation age/quality, configured caps, original ledger commitments and
  exclusions. “Backend found” does not mean “model runnable.”
- `run ... --dry-run`: resolve available local metadata and snapshots, validate
  policy and show a proposed placement or precise refusal. No model execution,
  weight upload, kernel compilation, allocation reservation, cache mutation or
  network access. Bounded read-only metadata inspection is allowed. Missing
  source facts or probes requiring context creation remain `unknown`; do not
  construct a context to make the dry run look complete.
- `run ... --explain`: execute normally and print the resolved source/profile,
  allowlist, rejected routes, selected layers/stages/backends/domains, fit
  breakdown and fallback reason to stderr. Keep generated output clean.
- Return structured distinctions: invalid policy/reference, backend unavailable,
  source unavailable, unqualified model/shape/composition, availability unknown,
  insufficient resources, resources changed, preparation failed, unsafe/unknown
  completion. A dry-run proposal is never a reservation or execution guarantee.

## Shared request and placement flow

CLI, future serving/config adapters and library callers lower into the same
policy representation. No CLI-only route table should grow into another engine.
Conceptual types below are **pseudocode**, not new runtime implementations:

```rust
struct RunRequest {
    program: ValidatedCanonicalGraph,     // one model table, direct or graph entry
    entrypoint: ModelOrPipelineId,
    task: BoundedTaskRequest,            // prompt/context/output/phase bounds
    numerical: SelectedNumericalProfile,
    backend: AutoOrRequiredBackend,      // applies to every neural stage
    devices: AutoOrOrderedAllowlist,
    headroom: HeadroomPolicy,
    role_constraints: StageRoleConstraints,
    diagnostics: DiagnosticOptions,
    provenance: ResolvedFieldOrigins,     // config path / CLI / built-in
}
struct DeviceRef { host: HostIdentity, physical: PhysicalIdentity }
struct BackendBinding { device: DeviceRef, backend: BackendId, epoch: SnapshotEpoch }
struct ResourceSnapshot {
    epoch: SnapshotEpoch, observed_at: Instant,
    devices: DeviceCapabilities,
    domains: MemoryDomainObservations,   // topology, caps, ledger, physical use
    links: QualifiedTransferCapabilities,
}
struct ModelRequirements {
    source: VerifiedSourceIdentity, numerical: NumericalIdentity,
    regions: LegalIndivisibleGroups, states: ExactStateSchemas,
    phases: BoundedPhaseRequirements, qualified: ExecutionCapabilities,
}
struct PlacementProposal {              // description; grants no permission
    compute: ComputePlacement, parameters: ParameterPlacement,
    state: StatePlacement, cache: CachePlacement, transfers: TransferPlan,
    costs: PerDomainPhasePeaks, bindings: CapabilityAndTopologyEpochs,
}
```

The flow is:

1. Select explicit/default input; bounded-load config if selected. Normalize
   CLI/config to one graph/request, apply field precedence and path origins,
   then validate entrypoint, source selection, task bounds and backend/device
   contradictions before effects. Read bounded metadata; bind real source
   identity at preparation.
2. Resolve source-specific semantic/shape/state requirements through the existing
   model adapter. Catalog metadata alone cannot mint executable capability.
3. Discover read-only hardware/backend capabilities and current resources;
   distinguish compiled, discoverable, eligible, numerically/state qualified and
   performance-qualified tuples. Preserve runtime/compiler/driver/shape limits.
4. Build a bounded logical proposal over existing semantic regions, using the
   single-device scan then qualified layer groups. Compute/weights/state/cache
   are separate maps connected by exact transfers, not separate schedulers.
5. Freshly revalidate identities, source/profile/shape, topology, ledger fences
   and available resources. Admit through original physical owners. Prepare
   source-bound executables/buffers using those leases and existing adapters.
6. Run the prepared physical plan through original fair workers. Actual Ready
   events feed the original `SessionAuthority` publication path, including
   direct runs. TaskCoordinator orchestrates where applicable; it does not
   replace that authority. No policy lookup per token.

Cache semantic/cost templates only with their complete source/profile/toolchain
keys. A cached topology or placement never caches allocation permission. Device,
memory pressure and observation-age changes require a fresh binding/admission.

Eligibility precedes memory ranking and covers the **actual workload envelope**:
tokenized prompt and context extent, output bounds, cache/state kind, active
concurrency/batch shape, phases, tensor/activation/state precision, effective
arithmetic profile and draft/MTP policy. Existing finite short-context learned
GPU proofs or a family diagnostic cannot qualify an arbitrary `run` prompt.
Reject a request outside the supported envelope even when its weights fit.
Switching backend must preserve the selected source and effective numerical
contract; a differently rounded implementation needs its own explicitly selected
and qualified profile, not an automatic relabeling.

## Current resources and comfortable fit

Keep these quantities distinct **per physical memory domain**:

| Quantity | Meaning |
| --- | --- |
| Physical/operator ceiling | Fixed conservative registered domain limit; driver maximum is only a ceiling |
| Current availability | Timestamped driver/OS/cgroup observation including external users; may be unknown |
| Original commitments | Existing arbiter/provider leases, including reserved but not yet materialized bytes |
| Existing backing | Actual owned resident allocations, retained pools/native contexts/graphs and reusable bytes |
| Candidate incremental peak | Additional ledger charge and physical growth after proven same-owner reuse |
| Headroom | Explicit spare capacity beyond all modeled costs, applied once per physical domain |

Never register a new physical limit from current free memory, grow a cap when
free memory rises, or resize a persistent context/envelope to make a route fit.
`register_matching` remains immutable. Bare `run MODEL` must not require the
user to calculate and supply resource caps. For a **new** process, resolve one
bootstrap policy before context effects: intersect attested physical-domain
capacity with OS/cgroup constraints, explicit operator limits when supplied,
and mandatory backend safety ceilings. The proposed default operator ceiling
is that attested intersection; comfortable headroom is withheld separately.
Resolve and record it once, including its policy identity and provenance.
Backend/provider/native-context sub-envelopes still require their qualified
finite policy and cannot take all remaining capacity by assumption. Unknown
physical limits or unbounded bootstrap effects refuse with an explanation.

This default is a proposed new CLI bootstrap policy, not permission to resize
any already registered domain or persistent envelope. Existing configured and
benchmark/serving caps remain unchanged. Live free memory constrains a new
reservation only, never domain registration or later cap growth.

For each candidate check **both** original ledger allowance and physical
availability. They are intersecting constraints, not two quantities to add.
The planner must map charges to actual backing/owners to distinguish:

- Bytes already physically resident: already excluded from driver free memory;
  do not subtract them again from that observation.
- Unmaterialized promises to other owners: not necessarily excluded from driver
  free memory; protect their potential growth before accepting a new candidate.
- Candidate-owned existing backing: count once in its total peak; only new
  growth consumes newly available physical bytes. Reuse needs exact ownership,
  layout, lifetime and original admission proof, not equal tensor hashes.
- Prepaid provider envelope and its internal allocations: one outer physical
  charge, not envelope plus children. Preserve conservative reserved slack and
  native context partitions; report stranded capacity rather than spending it
  as another model's free pool. Reuse inside an existing envelope must remain
  within its original sub-admission.

Conceptually, new ledger charge must fit remaining original allowance; physical
growth plus protected unbacked promises plus headroom must fit observed usable
availability. Matching requires an authoritative backing/charge mapping. When
that mapping or current availability cannot be established, mark fit unknown
and exclude that route from automatic comfortable-fit selection. Do not invent
free bytes from `max_memory`, GTT addressability or a configured provider cap.

UMA CPU/GPU/NPU compute domains can share **one** physical memory domain. Apply
host/OS/cgroup limits and headroom once to the union of live backing, including
display/OS/external use. Do not sum host RAM, VRAM and GTT counters or count the
same host-visible allocation on both sides of a transfer. Coherency still needs
actual qualified synchronization; shared RAM does not imply zero-copy safety.

Proposed comfortable-fit default: spare `max(256 MiB, 10% of domain ceiling)`
for discrete device memory; `max(1 GiB, 10%)` for host/UMA. `--headroom 15%`
or `--headroom 2GiB` explicitly replaces that routing margin for every selected
domain; configured mandatory safety floors still apply. Validate finite checked
values and report resolved bytes. This is an adjustable policy proposal, not a
measured universal safe margin; it never substitutes for an omitted cost.

Account complete request peaks across cold preparation, prefill, steady decode,
verification, and teardown/migration overlap where applicable:

- Source metadata/scans/windows, host mappings/materialization and upload staging;
  page cache accounting uses OS semantics rather than double-counting RSS.
- Exact retained weights/packs/replicas, tied tensors and source codecs.
- KV, sliding/recurrent/conv/indexer state; committed and provisional/COW roots,
  retained checkpoints/readers, requested context/output bounds and concurrency.
- Activations, scratch, logits/output credits, host readback and transfer buffers.
- Native context/code/modules, compiler working set, JIT/cache retention, graph
  capture/replay storage, allocator pools/fragmentation and bootstrap overhead.
- Optional or required draft/MTP weights, own state, taps, candidate tokens,
  target verification scratch, rejected-state retention and resynchronization.

Take the maximum **joint live set** of each permitted phase/overlap, not the sum
of mutually exclusive peaks. Unknown native/compiler cost needs a separately
qualified finite conservative allowance. Charge context bootstrap before creating
it where supported; explicitly model the existing pre-admission bootstrap gap
and refuse a claimed hard bound until its owner closes that gap. Do not assume
hot caches, imminent model eviction or another process exiting. Initial automatic
routing uses presently available resources and does not preempt other work.

## Selection, splitting and safe admission

Full-fit means the complete requested neural pipeline fits on that GPU within
its qualified shape/profile, plus required admitted host support. Optional
speculation may be omitted only when the resolved policy permits `auto`;
a requested draft/MTP feature cannot disappear merely to claim a full fit.

If no single GPU fits, consider legal contiguous layer-group assignments in
allowlist order. Maximize the number of groups on the first GPU **subject to a
feasible complete remainder**, then the next, then CPU. Bounded backtracking is
needed: blindly filling the first GPU can strand an indivisible final group or
head. Never divide a layer merely by proportional weight bytes.

The source adapter defines legal boundaries and indivisible dependencies:
tied/shared weights, embeddings, final norm/head, recurrent state and native
MTP groups may require joint placement or explicit charged replicas/transfers.
Weights and per-layer state remain colocated with compute initially. Store
authoritative state-shard references under the original StateRoot; transfers
carry exact activation layout/precision, source/profile/root/frontier/generation
and producer completion. Keep source, destination, staging and readers alive
through actual transfer and consumer completion. Host sampling/control is not
permission to evaluate unsupported model layers on CPU.

Allow a split only if every group **and composition** is qualified: closed
interfaces, exact state bridges, numerical behavior, completion and costs.
CUDA-qualified plus ROCm-qualified does not imply a qualified CUDA→ROCm model.
If no split is qualified, try the allowed qualified full CPU route; otherwise
fail with the missing capability. Do not silently invoke streaming, tensor
parallelism, a different quant or approximate math to rescue the fit.

Automatic order remains capacity/preference driven. Within equivalent layer
coverage choose fewer boundaries and lower measured transfer/queue cost. A
qualified GPU group that is demonstrably detrimental including transfers may
be excluded with an explanation; unknown performance is reported, not labeled
a speedup. More cards or shards is not inherently faster. Cross-host RTT,
staging, bandwidth contention and target latency limits can make a technically
fitting split unusable. CPU-only remains explicitly selectable for comparison.

Admission is transactional at the existing ownership boundary:

1. Re-read selected-domain observations and original ledger state immediately
   before admission. Snapshot epochs, freshness bounds and identity are checked;
   this still cannot prevent an unrelated process racing an allocation.
2. Obtain the complete set of original reservations before preparation/launch.
   Add an all-or-none domain-bundle operation to the original arbiter if needed;
   current `admit` proves only one-domain atomicity. Do not introduce a parallel
   reservation service. Aliased domains are merged before this operation.
3. Preparation allocates/compiles only under admitted owners. If it fails with
   provably known-safe teardown and no submitted work, release only retired
   resources and replan within the same request/allowlist. Bound this to one
   fresh-snapshot replan; further churn yields `resources changed`, not a loop.
4. Unknown native load/unload, transfer, capture or launch completion retains
   owners and fences/quarantines through the existing mechanism. Do not retry
   submitted model work elsewhere, manufacture Ready or publish a partial root.
   Future recovery from committed checkpoints is a separate explicit protocol.

Initial selection is a startup decision, not automatic live migration under
memory pressure. Shape/context growth outside the admitted envelope refuses or
uses an already qualified bounded policy; it never silently consumes the margin.

## Future stage roles, NPUs and peers

Keep host, physical device, backend, memory domain and **stage role** separate.
Target prefill/decode, draft proposal, target verification, encoder and other
stages are semantic roles; an NPU is hardware, not a synonym for “drafter.”

Reserve one extension: `--draft-device DEVICE[,DEVICE...]` constrains the draft
role, while `-d` remains the entire run's allowlist. Draft selectors must be in
that allowlist and compatible with forced backend constraints. An NPU-only
draft override under `run-cuda` or `run-cpu` is a contradiction. Target placement
still follows the qualified GPU/CPU policy; NPU target execution needs its own
explicitly qualified future policy before it enters automatic target ranking.

`--speculation auto|off|mtp|draft:MODEL` is the proposed role-policy spelling.
The new `run` default is `auto`: select only an already qualified method with
applicable measured end-to-end benefit, otherwise use ordinary decode. This
does not change existing benchmark/serving speculation controls. `mtp` and
`draft:MODEL` require that method to be available, qualified and admitted;
required source/model features also cannot be disabled by the `auto` default.
With `auto`, `--draft-device` restricts an optional method's placement, not a
requirement to perform drafting. This spelling reserves no current capability.
Native MTP depth and draft length are different: one trained
MTP layer can be applied recursively for multiple proposed tokens only when the
actual source algorithm and carried state support that recurrence. Never map
“one MTP layer” to “one token maximum” or invent untrained extra heads.

Select an NPU proposal route automatically only with applicable evidence that
complete speculative execution is faster than available GPU drafting **and**
the permitted non-speculative baseline: include activation/tap transfer,
conversion, compile/warmup amortization, synchronization, expected acceptance,
target verification, rejected-prefix rollback and resynchronization, queue and
shared UMA bandwidth/power contention. Unknown/stale evidence cannot qualify
“faster.” Keep target/drafter private roots and original single target commit.

Remote references are future opt-in placement, not automatic network discovery:

- Bare `auto` stays local. Only explicit enrolled peer references authorize a
  future execution attempt there under existing tenant/source/data permissions.
  Host aliases resolve pinned authenticated identities and capability epochs;
  a matching display name is insufficient.
- Read-only dry runs use already available peer metadata and label freshness
  unknown; they do not open sockets or enumerate remote machines. A future live
  run can contact explicitly selected enrolled peers, not enroll them implicitly.
- Coordinator requests all-or-none bounded holds from the **original** per-host
  ledgers before preparing a multi-host plan. Reservation IDs carry host,
  session/placement/capability epochs and expiry. This coordinating operation is
  not a new global allocator. Safe unsubmitted holds can expire/release;
  expiry/heartbeat silence never releases possibly submitted work.
- Cost directional links, host/device copies, encryption, bandwidth, RTT/jitter,
  relay paths and preparation/artifact availability. No implicit remote download,
  weight redistribution, SSH, shell execution or assumption that a tunnel is
  RDMA/coherent memory. First remote unit is a contiguous layer group with
  local state, not arbitrary per-operator WAN delegation.
- TaskCoordinator joins every required shard's Ready under valid epochs before
  the original session's `SessionAuthority` publishes one logical root. Layer
  workers cannot commit independent visible roots. Loss, stale replies and
  uncertain DMA retain/fence original
  ownership; no partial root or local retry of possibly submitted remote work.

## Examples of the proposed behavior

These commands illustrate the proposed API; they are not current support claims.
`model.gguf` denotes a locally available, source-qualified variant, and displayed
memory numbers must include all phase costs and resolved headroom.

| Command and situation | Required result |
| --- | --- |
| `zerogravity run model.gguf` — NVIDIA and AMD both comfortably fit | NVIDIA full-model route; no needless split |
| `zerogravity run` — default `config.yaml` exists | Load only the XDG-selected default, validate its explicit entrypoint and route normally |
| `zerogravity run` — default missing or invalid | Explicit missing/load/validation error; no cwd search or fallback |
| `zerogravity run ./chat.yaml --backend cuda` | Read selected config; CLI backend wins, then validate the complete request |
| `zerogravity run model.gguf` — default config also exists | Explicit model only; no implicit default overlay |
| `zerogravity run --model ./odd.yaml` | Treat argument as model source, never as run config |
| Same — NVIDIA capacity 24 GiB but only 3 GiB currently available; AMD has a complete 10 GiB fit | AMD full fit; never assume all 24 GiB are free |
| Same — each NVIDIA has less usable space than the full model; AMD fits | AMD full fit before NVIDIA splitting |
| Same — no GPU fits; legal groups can use NVIDIA, AMD and CPU | Qualified groups on preferred NVIDIA, next AMD, CPU remainder; fail/allowed full CPU if composition unqualified |
| Same — no eligible GPU, adequate host memory and qualified CPU model | CPU route with explicit explanation |
| `zerogravity run-cpu model.gguf` | All neural target/draft/MTP stages on CPU |
| `zerogravity run-cuda model.gguf -d local/pci:0000:03:00.0` — ID resolves AMD | Backend/device contradiction before preparation/allocation |
| `zerogravity run-rocm model.gguf` — available AMD memory insufficient and no qualified all-ROCm split | Resource refusal; no CPU neural fallback |
| `zerogravity run model.gguf -d gpu:1,gpu:0,cpu:0` | Scan GPU1 then GPU0 for full fit; otherwise split in that order, CPU allowed |
| `zerogravity run model.gguf -d gpu:1,gpu:0` | Same GPU preference; CPU spill prohibited |
| `zerogravity run model.gguf -d cpu:0 --dry-run` | Read-only CPU fit proposal/refusal; no reservation |
| `zerogravity run model.gguf -d local/gpu:0,lab/pci:0000:41:00.0,local/cpu:0` | Future authorized peer candidate in explicit order; current unsupported implementation refuses |
| `zerogravity run model.gguf -d gpu:0,npu:0,cpu:0 --speculation auto --draft-device npu:0` | Future NPU draft only if qualified and measured beneficial; otherwise permitted ordinary target execution |
| `zerogravity run-cuda model.gguf -d gpu:0,npu:0 --speculation mtp --draft-device npu:0` | Contradiction; required MTP is not permission to escape the forced backend |

## Decisions for user review

The recommendation keeps the requested `run-*` convenience commands and `-d`
ordering, with one canonical CLI/config request. Review:

1. Ordered lists are strict allowlists; CPU must be listed last to allow explicit
   spill. Full-GPU-fit scanning precedes splitting, even with explicit order.
2. Forced backend constrains every neural stage, not only the target model.
3. Comfortable headroom defaults/override and fail-closed unknown availability;
   no automatic preemption, downloads, quant changes or cap growth.
4. Physical host-scoped IDs plus snapshot convenience indices; local-only auto.
5. Optional speculation uses measured benefit; explicit methods remain required.
6. Implement local single-device routing first; split, mixed-backend, remote and
   NPU stages become executable only through separately qualified larger units.
7. Extend the existing pipeline schema to v2; default only selector-free runs to
   XDG `config.yaml`; explicit CLI overrides config without implicit list merges.
   Migrate model metadata `--config` to `--model-config`, and require explicit
   runnable entrypoints without opening declared routes.

Acceptance of this draft does not itself schedule implementation or lift device,
source-download, remote-execution or qualification holds.

## References

- [Plan layout](../README.md); [proposed work and scenarios](TODO.md).
- [Pipeline composition](../pipeline-composition/README.md),
  [regions/profiles](../pipeline-composition/REGIONS_AND_PROFILES.md),
  [source-grounded execution architecture](../pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md).
- [Distributed placement](../../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md),
  [peer/state protocol](../../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md),
  [accelerator backend contract](../../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).
- [Distributed work](../distributed-execution/TODO.md),
  [memory/residency work](../memory-and-streaming/TODO.md).
- Source: `crates/zerogravity-cli/src/main.rs`,
  `crates/zerogravity-cli/src/pipeline.rs`,
  `crates/zerogravity-core/src/pipeline.rs`,
  `crates/zerogravity-gpu/src/physical_service.rs`,
  `crates/zerogravity-gpu/src/backend/contract.rs`,
  `crates/zerogravity-core/src/resource_arbiter.rs`,
  `crates/zerogravity-core/src/session_root.rs`,
  `crates/zerogravity-core/src/targets.rs`.
