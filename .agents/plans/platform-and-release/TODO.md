# Platform and release

Provenance commit: `6d74f76`.
Scope: remaining P00/P04/P05/P08/P20/P21/P22/P23.
Contracts: [P00–P23 roadmap](../IMPLEMENTATION_ROADMAP.md),
[minimal execution spine](../../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md),
[compiler/task refinement](../../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md),
[release gates](../RELEASE_ACCEPTANCE_GATES.md).

## Uncommitted

- [ ] Generalize the working typed forward plan into verified effectful semantic regions/task control, closed provider substitution, bounded shape/kernel caches and explicit command/resource legality; no new persistent IR/runtime without a real need.
- [ ] Prove artifact→model→compiled task→coordinator→output and explain-plan for supported tasks, including source-qualified preselected fallback and no selected-GPU-error CPU substitution.
- [ ] Run extension drills for one stateful semantic op, device/provider, non-token TaskProgram, quant/storage/API adapter and optional ParameterVersion/TrainStep seam; future training is not thereby implemented.
- [ ] Finish per-file source/model/license/NOTICE/SBOM policy, malformed-input fuzzing and ownership/security review, authenticated APIs/private caches and qualified driver/device safety.
- [ ] Verify Nix/module/native/cross packages and installer/release behavior on actual Linux/Windows/macOS target access; packaging definitions alone do not prove platform support.
- [ ] NixOS module eval check evaluates a complete base system (root filesystem and boot loader) so the whole-system assertions hold on current nixpkgs-unstable; the CI `quality` job was failing at this evaluation.
- [ ] Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
- [ ] Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
- [ ] Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
- [ ] Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.

- [ ] Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
- [ ] Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.

## Committed

### 6d74f76

- [x] Cohesive workspace, ordinary CPU/product packaging, one shared GPU library and accepted narrow original-model vertical slice.
- [x] These completed slices do not mark every original P00–P23 goal complete.
