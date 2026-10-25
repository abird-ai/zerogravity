# Platform and release

Provenance commit: `6d74f76`.
Scope: remaining P00/P04/P05/P08/P20/P21/P22/P23.
Contracts: [P00–P23 roadmap](../IMPLEMENTATION_ROADMAP.md),
[minimal execution spine](../../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md),
[compiler/task refinement](../../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md),
[release gates](../RELEASE_ACCEPTANCE_GATES.md).

## Uncommitted

- [ ] Owner23Oct: bring Vulkan execution to CUDA feature/correctness parity;
  use ROCm/HIP for AMD including pvl-x2 gfx1151. Vulkan source catalogue
  landed in `19ff059`; the real offline build PASSED on x2 25Oct (v-build,
  1,006.6s, three artifacts) and v-admin/v-bootstrap closures are green;
  V cases are blocked on the v-resident validator/producer field fix
  (PM handoff §2.3) — no Vulkan parity or performance claim yet. Reuse existing physical
  identity/arbiter/service/queue/quarantine. Separate compiled, available and
  qualified source/layout/math capabilities; refuse before allocation/launch
  when unsupported. Preserve gfx90c launch hold and source/SDK acquisition rules.
- [ ] Apply the owner-approved [static-native reuse/direct-port rule](../pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md#reuse-first-across-all-implementations-and-harnesses)
  to all implementations and correctness tooling: exact MIT/Apache file and
  transitive license/NOTICE review, suitable Rust/static C ABI library first,
  otherwise faithful smallest-component port with attribution and fixtures.
- [ ] Prove actual final binary linkage for selected reused components across
  supported targets: native toolchain and ABI/error/thread/allocator boundaries,
  no accidental Python/PyTorch/foreign-server runtime, declared optional device
  interfaces, native fixture/diagnostic support and feature-isolated packaging.
  Statically linked components do not alone prove the entire binary is static.
- [ ] Verify shared effective-configuration explanation/revision handling and
  replacement compatibility across allocator/provider implementations; preserve
  original resource/session/completion authorities and old-reader retirement.
- [ ] Generalize the working typed forward plan into verified effectful semantic regions/task control, closed provider substitution, bounded shape/kernel caches and explicit command/resource legality; no new persistent IR/runtime without a real need.
- [ ] Prove artifact→model→compiled task→coordinator→output and explain-plan for supported tasks, including source-qualified preselected fallback and no selected-GPU-error CPU substitution.
- [ ] Run extension drills for one stateful semantic op, device/provider, non-token TaskProgram, quant/storage/API adapter and optional ParameterVersion/TrainStep seam; future training is not thereby implemented.
- [ ] Finish per-file source/model/license/NOTICE/SBOM policy, malformed-input fuzzing and ownership/security review, authenticated APIs/private caches and qualified driver/device safety.
- [ ] Verify Nix/module/native/cross packages and installer/release behavior on actual Linux/Windows/macOS target access; packaging definitions alone do not prove platform support.
- [ ] Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
- [ ] Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
- [ ] Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
- [ ] Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.

- [ ] Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
- [ ] Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.

## Committed

### 25Oct evidence records — `docs(plans)` (this commit)

- [x] User-requested obsolete worktree cleanup:43 registered +5 redundant
  extracted trees retired after source/evidence preservation; branch refs and
  active/reference/build trees retained. ROOT reverified all44 new archives and
  four preexisting extract archives; main/rootREADME unchanged.
  [Recovery and remaining live scope](../../runs/resume-all-parallel-root-20261024/WORKTREE_CLEANUP.md).

- [x] Retire nine more obsolete trees: all source/evidence and Git metadata
  preserved; ROOT compared all18 TAR archives against actual source before
  removal. Two large transport archives retained by same-inode hardlinks,
  without duplicate copy/rehash. Eight registrations remain: main, five active
  candidates and two reference roots still consumed by current checks.
- [x] Remove29 obsolete merged local branches in one atomic old-OID transaction.
  ROOT verified actual ancestry/ownership and exact final refs; packed refs,
  all29 reflogs and recreation commands preserved. Seven protected heads remain;
  main/source/index/config/README, active/reference trees and remotes preserved.
- [x] Retire two historical Git registrations and their remaining merged local
  branch25Oct. Verified recovery copies and renamed originals; preserve all
  ordinary reference files at their existing paths. Cumulative30 local branches
  removed; six registered trees/heads remain (main plus five active candidates).
  Main HEAD/index/rootREADME and all13 other resolved refs unchanged.

- [x] Retire the five active lane units' trees and branches; their source is
  integrated (`8f01635`..`68e13a8`, full CPU gate rc0 25Oct). Keep three donor
  drafts until their work is recovered; preserve lane-unique evidence first.
  DONE25Oct: trees and branches retired; unique evidence preserved under
  zerogravity-worktrees/.preserved-lanes-20261025/ (SHA256SUMS + RECORDS.md);
  OLD-OIDs recorded; donor/reference trees kept.



### `ef9cffd413df` — assembled feature gates and platform scope

Related source commits: `4bc7c429d585` (vendor-hip-runtime), `63583681ae8c` (gpu-native-policy-identity).
Aggregate source21 qualification applies to the assembled tree; no per-commit
build or general model/backend grant. Historical failures keep their source scope.

- [x] Reviewed implementation and existing bounded validation committed.

### 1af2ac4

- [x] NixOS module eval check evaluates a complete base system (root filesystem and boot loader) so the whole-system assertions hold on current nixpkgs-unstable; the CI `quality` job was failing at this evaluation.

### 6d74f76

- [x] Cohesive workspace, ordinary CPU/product packaging, one shared GPU library and accepted narrow original-model vertical slice.
- [x] These completed slices do not mark every original P00–P23 goal complete.

## Tests and verification

- [ ] Qualify all six [CPU release definitions](../../../docs/DEVELOPMENT.md#cpu-cross-release-definitions) with actual target execution, installer checksum/version/refusal cases and platform signing/authenticity policy. Build definitions and same-host checks do not establish other-target support.
- [ ] Test enabled NixOS service lifecycle and runtime-secret handling after serving is implemented; the currently disabled service definition and module evaluation are not a running-server claim.
