Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# Profile policy, replaceable components and native reuse

Owner directives, 21 Oct 2026. Design and pending acceptance only; no new
implementation, physical qualification or performance result. Extends
[regions/profiles](REGIONS_AND_PROFILES.md); live tasks remain in the
[composition](TODO.md), [optimization](../cpu-gpu-optimization/TODO.md) and
[platform](../platform-and-release/TODO.md) plans.

## Explicit approximation

- Support exact and deliberately approximate profiles. Approximation is a
  first-class selectable strategy for measured speed/capacity/quality tradeoffs.
  Keep artifact/source, arithmetic, model approximation, workload and placement
  as separate resolved dimensions; approximation is broader than quantization.
- Candidate mechanisms include weight/activation/KV compression, reduced
  precision and model-compatible attention/context or expert-budget variants.
  Each declares its changed equations/state/distribution and supported scope;
  no universal quality or speed claim follows from a mechanism name.
- A resolved profile names permitted methods/ranges, evaluation domain and
  accepted quality limits, memory/latency goals, fallback and failure behavior.
  Adaptive approximation may select only within those explicit permissions.
  No silent promotion from an exact profile under pressure.
- Check correctness against the declared variant and independent quality against
  the original target. Preserve an exact selected baseline for comparison.
  Approximation never relaxes ownership, memory safety, cancellation, source
  identity, tenant isolation or completion/publication requirements.
- Bind effective approximation and state format to cache/snapshot/graph keys.
  Cross-profile reuse needs an explicit qualified conversion or recomputation.
  Diagnostics expose the selected variant, effective overrides and quality limits.

## Replaceable implementations

Keep one coherent mathematical engine and ownership spine while permitting
different implementations of allocators/pools, kernel/region providers, storage
codecs, residency/eviction, scheduling and transfer strategies. Rebuild module
boundaries when concrete implementations demonstrate the need; do not preserve
an unsuitable implementation merely because it was first.

- Separate an implementation from its contract. A resource ledger/arbiter owns
  admission and actual physical accounting; an allocator strategy supplies
  storage under that contract. Device/workload-specific pools may coexist
  without becoming independent unaccounted authorities.
- Allocator replacement must preserve alignment, alias/view provenance, actual
  backing size, generations, initialized ranges, failure behavior and every
  encoded/captured/queued reader. The issuing allocator retains its allocations
  through actual completion/quarantine and last-reader retirement.
- Select replacements at preparation or a defined drained boundary. Never free
  old backing merely because the current configuration names a new allocator.
  Prove old/new coexistence, pressure, failed allocation, cancellation and unknown
  completion. Define compatibility before migrating persistent state or handles.
- Imported native components run through these same contracts. They cannot
  introduce a separate model loop, session publisher or hidden resource budget.
  Preserve a forcible baseline and accepted alternative where meaningful.
- Start with private concrete seams and the first useful implementation; add
  public abstractions only when real clients and extension drills justify them.

## Shared policy context and module-specific decisions

All participating modules can inspect the same versioned effective configuration
and permitted observations. Share what is needed for optimization without
granting unrelated prompt/state access or new resource authority.

| Shared view | Contents and lifetime |
| --- | --- |
| Resolved configuration | Source/arithmetic/approximation bindings, workload goals, deadlines/fairness, hard host/device/IO caps, allowed providers, module overrides, batching/draft/prefetch ranges; immutable for an admitted attempt |
| Operational observations | Bounded, timestamped load/queue/cost/residency/headroom information published by existing owners; advisory snapshots do not grant memory or completion credit |
| Decision explanation | Configuration revision, default/override provenance, selected candidates, rejected alternatives and reasons; bounded diagnostics outside the token hot loop |

- Module-specific policies optimize local tiling, cache admission, batching,
  draft depth or prefetch within the shared constraints. Policies are explicit
  and configurable, with deterministic precedence and conflict validation.
- Decisions spanning modules belong to existing shared coordination: compare
  batching, graphs, speculation and prefetch against their joint live memory,
  IO, fair progress and latency cost. Do not independently spend the same slack.
- Resolve static choices outside the hot loop; runtime selection is bounded
  among qualified variants. Modules request admission through existing owners.
  Handle stale observations and declined reservations without hidden overcommit.
- Configuration reload creates a new revision for future preparations/attempts.
  Existing attempts retain compatible bindings and readers; semantic/state-format
  changes require explicit migration or a fresh root. Expose effective settings
  and fallback decisions consistently through explain/diagnostic output.

## Reuse first across all implementations and harnesses

Apply the owner-approved [reuse/provenance policy](../../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md)
to mathematical kernels, allocators, codecs, compiler pieces, scheduling helpers
and correctness/benchmark tooling alike:

1. Identify a suitable existing component. Pin source and review exact file,
   dependency, license/NOTICE and build requirements. Prefer MIT or Apache-2.0
   components that can satisfy the native binary and required contracts.
2. Reuse a suitable Rust library or statically link a suitable C ABI component
   first. A narrow audited C ABI shim for a self-contained C++ component is
   acceptable. Verify actual static linkage, target toolchain, transitive runtime,
   allocator ownership, thread/error/cancellation behavior and binary-size cost.
   A repository containing C++ is not proof of a usable standalone ABI.
3. If direct native linkage is unavailable or incompatible, directly port the
   smallest useful implementation and its fixtures into Rust/native device
   source. Preserve attribution, exact algorithm, casts/reductions, layout,
   rounding/ties, malformed-input behavior and relevant negative controls.
4. Establish a small faithful port against independent pinned fixtures/oracles,
   then integrate through the original engine contracts and optimize. The port
   may use a temporary test-only baseline; it must not become another serving
   stack. If adapting changes behavior, declare and test the changed contract.
5. Retain upstream licenses/notices and mark modifications. A translation remains
   derived code; our MIT license does not erase Apache or file-level obligations.
   Existing separately approved components retain their dispositions.

Production paths must fit the Rust native binary without a Python interpreter,
PyTorch runtime or foreign inference server. Independent offline reference tools
remain valid test oracles. Preserve already required optional GPU driver/compiler
interfaces; do not describe an artifact as fully static without inspecting its
actual link dependencies. Separate portable host fixture checks from real-device
gates and supported-target packaging checks.

## Correctness harness and implementation acceptance

Promote [the existing kernel-contract research](../../research/engine-optimization-comparison-2026-10-21/tensorfold.md)
into explicit implementation tasks. Reuse/port the harness and useful kernels
according to the same rule; importing test names without executable checks is
insufficient. Build shared comparison/contract logic in native Rust; provide a
native fixture/diagnostic entry point as needed. Deliberately broken variants
remain test-only and cannot be selected for serving.

| Gate | Required positive and negative evidence |
| --- | --- |
| Independent oracle | Compare against independently obtained source/codec fixtures and the selected numerical contract; candidate and oracle cannot share the faulty implementation. FP64/error bounds complement, not replace, an exact pinned-profile oracle |
| Arithmetic | Correct implementation passes; wrong rounding, cast/store point, reduction/FMA order, tie handling, NaN/status handling and precision controls fail where forbidden by that profile |
| Extents/writes | Odd/ragged/tail shapes and minimal/max supported extents; poisoned output/scratch and canaries catch no-op kernels, unwritten tails, invalid aliases and writes outside declared ranges |
| Dispatch legality | Wrong device/image, shape, alignment, scratch/shared-memory size and unsupported capability refuse before launch; never intentionally issue an unsafe hardware launch to prove a guard |
| Identity/state/lifetime | Wrong/stale source, row, profile, root and generation fail; exercise COW, allocator replacement, captured readers, cancellation, pressure and unknown completion using existing isolation/fault controls |
| Harness sensitivity | Maintain named deliberately broken controls and verify the harness detects each intended defect. Run these controls in CI; a harness that accepts a planted defect fails acceptance |
| Model/quality/runtime | After leaf checks, full logits, all required state and natural continuation; approximate variants also meet separately declared held-out quality limits; real-device and full isolated check.sh gates remain required |

For each selected source/profile/phase/backend, maintain an explicit equivalence
matrix: solo/concurrent, B1/batched with reorder/refill, draft off/on and every
accepted prefix, fresh/restored/COW state, ordinary/graph/fused execution, and
contiguous/paged storage. Mark required bitwise equality, bounded error,
distribution equality or intentionally changed semantics per cell. A sampled
distribution guarantee does not imply an identical seeded trace. Do not impose
cross-profile bit equality where the contracts differ. Test rejected/incompatible
combinations and compare complete-request performance, including losing cases.

## Hardware order and implementation sequence

- Long-term target: broad model/device/platform compatibility comparable in
  scope to the portable and serving engines under study, with truthful per-target
  qualification and specialized paths alongside portable implementations.
- Hardware order: specialize first for the NVIDIA cards on both existing
  machines; **Strix Halo/gfx1151 is the highest-priority subsequent hardware
  target**, ahead of other new devices and cluster machinery. Capability guards
  and actual tests determine eligibility; keep the separate Renoir/gfx90c hold.
- For Strix Halo, qualify actual UMA backing/accounting, host/device contention,
  copy/coherence behavior, wave/MMA/kernel layouts and complete-request results;
  port CUDA assumptions only after checking the corresponding AMD contracts.
- Preserve the work order: current parity → matched short C1/C2/C4/C8 → measured
  generic improvements and short reruns → long contexts → dense27B → subsequent
  model work. The hardware order governs target selection, not permission to
  bypass correctness or to acquire models/SDKs. Cluster work remains deferred.

All implementation, static-link, mutation-control, physical, quality and
performance gates above remain pending until executed and recorded by their
owning plans. Documentation checks do not qualify any of them.
