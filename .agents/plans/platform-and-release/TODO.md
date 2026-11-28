# Platform and release

Provenance commit: `6d74f76`.
Scope: remaining P00/P04/P05/P08/P20/P21/P22/P23.
Contracts: [P00–P23 roadmap](../IMPLEMENTATION_ROADMAP.md),
[minimal execution spine](../../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md),
[compiler/task refinement](../../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md),
[release gates](../RELEASE_ACCEPTANCE_GATES.md).

## Uncommitted

- [x] L5 native launch environment follow-up: default x86-64 Linux Nix
  development shell shares the ordinary package GPU runtime closure;
  explicit portable shell remains CPU-only and GPU benchmarking inputs
  remain in the optional gpu shell. Actual shell realization and full
  native Cargo build passed, with no lock changes. The raw launch with
  SDK/loader variables stripped reproduced the three user errors on the
  exact cached 0.8B. Actual signed 48c00b22 package is realized at
  target/native-package (native release, unchanged dependencies).
  Fresh Auto and explicit CUDA each generated 11+3 tokens, joined
  retirement, 14.711s and 14.356s action durations (canaries, not perf).
  Corrected default dev shell also generated 11+3 tokens and retired
  (unoptimized 185.329s). Explicit ROCm reached physical admission and
  refused 9885299712/8870966887 bytes, not missing libraries. Final idle
  NVIDIA37MiB/0%, no compute owners. User launch command/log are outside
  connector read scope; reproduction is independent. Use ordinary nix
  run or nix develop, not the unwrapped Cargo binary. No host changes.
- [x] X2 ordinary native ROCm launch qualification (28 Nov user
  direction: X2 primary ROCm execution host). Exact L5 dcfa8fde source
  synchronized by incremental Git bundle, no source authored on X2.
  Normal Nix realization built the same package output as L5
  (48c00b22 source, target/native-package); no lock, trust, driver or
  SDK overrides. Existing Qwen snapshots only. Physical gfx1151,
  PCI 0000:c6:00.0, 96 GiB VRAM, isolated Web control host, ordinary
  numerical policy, default headroom, no device-mib override, context
  1024/active 1/output 3. Explicit ROCm 0.8B, Auto 2B and explicit
  ROCm 27B each loaded, generated 11+3 tokens (raw "Continue: 1, 2, 3,"
  → " 4,", length), and unloaded. All 24/24/64 layers on the AMD
  GPU; observed loaded VRAM 2508787712/4547153920/56435105792 bytes.
  Cold load actions 19.009/23.859/153.301s (canaries, not performance
  comparators). Each host retired; final 274452480 bytes/busy0/no KFD
  owners, exact initial idle. 0.8B success was part of the rejected
  sequential reload test below; 2B/27B fresh-process runs exited 0.
  Logs on X2: target/x2-native-launch-build.log,
  x2-native-launch-physical-retry.log, x2-native-launch-2b.log,
  x2-native-launch-27b.log; per-case results and physical receipts in
  matching run directories. No backend optimization or PS-F1 promotion.
- [ ] Fix ordinary same-process HIP load/unload/reload on X2 before
  claiming reusable native model lifecycle. After the actual 0.8B
  inference/unload, Auto 2B reached HIP but failed 500
  inference_error: conflicting physical device budget attestation.
  Source diagnosis: hip_budget derives an operation envelope from
  current free memory; begin_single_source attempts fresh physical
  preadmission and register_matching against the initial process cap.
  The live NativeContextBudget intentionally survives model retirement;
  configure_single_stream and NATIVE_CONTEXT_CONSTRUCTED are further
  one-time bootstrap boundaries. Do not relax equality, reset global
  state, release retained context credit or bypass fencing merely to
  pass reload. Separate stable physical capacity/context ownership from
  current model admission, reuse the existing actor and allocate a new
  source/model lease with current actual capacity and exact retirement.
  Qualify 0.8B→2B→0.8B in the same process, physical reuse, stale owner
  refusal and final exact idle. Coordinate with the ordered service/
  physical-owner migration; existing fixed-budget tests stay valid.
  Fresh 2B succeeds, so this is not a device capacity refusal.
- [ ] Populate missing optional original-HIP loaded-model snapshot
  provenance in the owner diagnostics: actual Web capabilities.snapshot
  is null for these three exact cached loads; original action correlation,
  source resolver and all physical layer/device observations passed.
  Do not synthesize the snapshot from UI selection; project the existing
  retained source identity. Initial test harness rejected this optional
  observation; corrected harness independently verified each requested
  digest in its own load event and actual provider/placement.
- [ ] Investigate same-process CUDA retirement/reload capacity: successful
  Auto load and unload left the next explicit CUDA cold observation with
  5335148135 bytes after headroom, just below the qualified 5 GiB device
  envelope. Fresh explicit CUDA passed; process retirement restored idle
  device state. Retain the original bound; reconcile shared compiler/
  context ownership and admission with measured CUDA backend work.

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
  UX-11D native package realized and physically qualified on L5
  at 3ec574ab (CUDA12.9/ROCm7.2.3). Normal trusted Nix realization
  fetched the declared closure; no lock changes. 28573dc0 preserves
  real registry path patches in shared Crane dependency builds;
  3ec574ab preserves complete declared package source trees. Exact
  source integrity, native release, clean wrapper startup, discovery,
  Auto CUDA original 0.8B generation, precise HIP capacity and source
  refusals, joined retirement passed. Portable and six cross-target
  feature boundaries remain CPU-only. Wider platform realization
  remains open. See CURRENT_HANDOFF. Ordinary declared build dependency
  fetching is authorized under AGENTS.md; do not disable substitution
  or require new approval for cache misses. Keep locked/offline
  compilation, not unnecessarily offline Nix acquisition.
- [ ] Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
- [ ] Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
- [ ] Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
- [ ] Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.

- [ ] Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
- [ ] Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.

## Committed

### 25Oct evidence records — `docs(plans)` (this commit)

- [x] User-requested obsolete checkout cleanup:43 registered +5 redundant
  extracted trees retired after source/evidence preservation; branch refs and
  active/reference/build trees retained. ROOT reverified all44 new archives and
  four preexisting extract archives; main/rootREADME unchanged.
  Recovery and remaining live scope.

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
  Git history (SHA256SUMS + RECORDS.md);
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

- [x] Separate full native Cargo CLI product from explicit portable CPU
  Nix/cross-release assets. Full native source build/install uses the
  production default; L5 isolated linked debug launch/version/help/doctor
  and native/Web smoke verified. Existing portable feature selections stay
  unchanged. Loader availability and qualified execution are distinct from
  compiled features.
- [ ] Future all-backend distributed native artifacts: qualify toolchain,
  runtime library availability, target linkage and installer authenticity
  independently of the existing portable CPU releases; no silent change to
  those release definitions or claim of cross-target execution.

- [ ] Qualify all six [CPU release definitions](../../../docs/DEVELOPMENT.md#cpu-cross-release-definitions) with actual target execution, installer checksum/version/refusal cases and platform signing/authenticity policy. Build definitions and same-host checks do not establish other-target support.
- [ ] Test enabled NixOS service lifecycle and runtime-secret handling after serving is implemented; the currently disabled service definition and module evaluation are not a running-server claim.
