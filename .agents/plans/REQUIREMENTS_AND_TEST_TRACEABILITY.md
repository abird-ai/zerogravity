# Requirements and test traceability

This is the stable repository target for acceptance/evidence navigation. It is
not a historical matrix of every prior run. Cross-window context starts at
[CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).

## Governing rule

A requirement is satisfied only when current source implements it and the
appropriate current tests or physical qualification pass. Git history preserves
older evidence; deleted handoffs, checkouts and transient run directories are
not acceptance inputs.

## Traceability map

| Area | Current requirement / implementation owner | Evidence location |
|---|---|---|
| Architecture and boundaries | [plan index](README.md), [active queue](TODO.md), [Fabric vNext](fabric-vnext/README.md) | architecture checks, crate tests, owning subsystem TODO |
| Model execution | [model execution](model-execution/TODO.md) | model/unit/integration tests plus required trained-model physical gates |
| Numerics and state | [numerics and context](numerics-and-context/TODO.md) | exact/reference tests, state/fault/cancellation gates |
| CPU/GPU optimization | [CPU/GPU optimization](cpu-gpu-optimization/TODO.md) | backend-specific qualification and matched benchmarks |
| Serving | [serving and speculation](serving-and-speculation/TODO.md) | CLI/server tests, concurrency/cancellation and physical serving gates |
| Memory/streaming | [memory and streaming](memory-and-streaming/TODO.md) | ownership, bounded-memory, cache/SSD and fault tests |
| Acquisition/cache | [model acquisition](model-acquisition/TODO.md) | artifact/cache/acquisition transaction and provider tests |
| Pipeline composition | [pipeline composition](pipeline-composition/TODO.md) | composition/configuration/state-isolation tests |
| Platform/release | [platform and release](platform-and-release/TODO.md), [release gates](RELEASE_ACCEPTANCE_GATES.md) | repository checks, packaging/install/release gates |
| Public architecture | [Engineering Specification](../../docs/ENGINEERING_SPECIFICATION.md), [Whitepaper](../../docs/WHITEPAPER.md) | current source and the above gates override dated design assumptions |

## Evidence discipline

- Keep stable fixtures and reproducible test inputs when they are needed by
  current tests.
- Do not make machine-local absolute paths, checkout names or transient run
  directories part of a permanent acceptance contract.
- Historical benchmark/result prose may explain a decision but cannot override
  current source, tests or a later qualification.
- Physical/hardware claims remain scoped to the exact device/backend/source
  cell actually tested.

## Current Fabric and retained Qwen acceptance

- SDK/CLI source/host checkpoint `d4e94b1`: typed artifact/component/cache operations,
  original fixed source/transfer domains, direct CPU planning/audit/output lifetime,
  finite CUDA/Gemma diagnostic owners and a public-only Rust example. Exact original
  math/physical owners remain; dead alternate CLI paths removed. Focused artifact/
  CLI/approved tiny CPU tests and native/Gemma host controls pass.
  Final gate:2,315 Rust/0fail/97ignored/302 suites (70 empty);17 Python suites/
  182 cases/10 skips. Real trained Gemma and GPU execution remain unrun, not inferred
  from compilation or local source presence. Active contracts and whole-transition
  F4/Q acceptance remain open; see owning TODO and current-source reconciliation.

- Shared scheduler source/host checkpoint `ae7ec11`: one CPU/GPU Router/worker and
  transport path, typed original physical owners, admitted final-reader/tokenizer
  controls; CLI scheduler copies removed.85 engine controls and4 genuine CPU
  SDK/HTTP fixture cases pass; CUDA/HIP/Vulkan compile/host scopes pass.
  Final gate: 2,306 Rust passes/0fail/94ignored/300 suite runs;
  17 Python suites/182 cases/10 skips.
  Moved GPU execution still requires separate physical numerical/state/cancel/
  ownership/serving regression; historical source/ABI evidence is not retagged.

- S0 engine/serving/SDK contracts compile and preserve one ForwardPlan/physical
  arbiter. Refusal/authority/API tests pass; final isolated gate2,159 Rust/0fail/
  87ignored/292 suites,17 Python suites/182 cases plus packaging. New operations
  refuse before execution/listener side effects; no model/physical grant.
  See [transition](fabric-vnext-transition/TODO.md) and its
  [feature matrix](fabric-vnext-transition/FEATURE_PARITY_MATRIX.md).
- Existing model/codec/source/CPU/GPU/CLI/HTTP behavior and deliberate refusals
  remain required through S/F moves. Structural visibility is not migration
  completion, physical evidence or production eligibility.
- Signed pre-transition gate`e2559b6`:2,142 Rust/0fail/87ignored/283 suites;
  17 Python suites/178 cases plus packaging; shared HIP/Vulkan strict pass.
- CUDA0.8B cached47→50 fullV/all42 is exact. Capacity/lifecycle48-active-token
  runs do not qualify full contexts or2B numerics; exact2B/wider authority open.
- HIP0.8B/2B bounded64 single-row full-model/two replays/all state pass;
  canonical healthy/fault/unknown final readers accepted in`85f33c7`.
  Finite27B full64/fullV/all112 is accepted only for documented diagnostic calls.
- CPU/HIP chunk GDN:19 cases/380 fields/64,345,520 words exact; native four
  numerical refusals expose status only;20 modules retire. Initialized completed
  writes are not the same as Ready exports or logical publication.
- Original multi-row47–50/512→514 references are not ZG execution qualification.
  Multi-row HIP integration, general27B, populated contexts, C1..C64, matched
  vLLM and measured superiority remain deferred requirements, not dropped work.
- Accepted unchanged evidence is retained; changed source/ABI/numerical/owner
  paths require affected qualification. Keep original failed and unrun scopes.
  No implicit resume of cancelled recurrence or model/device campaign.
