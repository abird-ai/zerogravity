# Rust workspace, interfaces, implementation practices and contributor workflow

**Current layout and model contracts ·16 October2026:** [H25](../plans/accepted-baseline/TODO.md) is accepted and committed as `8c39aa8`. The workspace now also contains the reviewed ninth Gemma4 contract library and Qwen3.8 source metadata APIs; their [separate main integration gates](../plans/model-execution/TODO.md) pass. These describe original configuration/tensor semantics and do not enable trained new-model inference. The ownership map below supersedes historical bootstrap sketches. One mathematical schedule, one source-identity issuer, independent session roots, sole physical admission and actual completion/quarantine remain mandatory.

| Workspace package | Responsibility |
|---|---|
| `zerogravity-core` | Shared typed source/program/state contracts, resource admission and device event/queue authority. Default std-only; optional serialization is explicit. |
| `zerogravity-artifacts` | Immutable source storage, checked GGUF parsing/codecs, validated Qwen source geometry/tensor roles and bounded duplicate-rejecting safetensors shard-index metadata. |
| `zerogravity-cpu` | Existing CPU numerical implementation borrowing source weights. |
| `zerogravity-model-qwen35` | Qwen3.5 schedule/state/tokenizer/checkpoint adapter; CPU and text are optional. |
| `zerogravity-model-gemma4` | CPU-free validated E2B and26B-A4B source configuration, tensor/KV/PLE/MoE contracts; no trained numerical execution route. |
| `zerogravity-gpu` | Generic resident GPU execution and physical lifetime ownership; independent Vulkan/CUDA/HIP features. Diagnostic qualification is explicit. |
| `zerogravity` | Small public facade and compatibility reexports; API examples and integration tests. |
| `zerogravity-cli` | The `zerogravity` executable, explicit CPU/GPU commands and text rendering. |
| `zerogravity-gpu-qualify` | Explicit diagnostic tool in `tools/gpu-qualify`, calling the same GPU library. |

The virtual workspace has one lockfile and version authority. Its default members are the facade and CLI with CPU/model/text defaults. The four narrowly patched upstream CubeCL packages are excluded workspace members; production GPU dependencies exclude CPU/family/text and LLVM code generation. Curated `all` selects implemented production components; `all-models` includes both family contract libraries and does not imply executable support for every described checkpoint. Diagnostics and abort probes require explicit features. A downstream path consumer must declare the four patch entries in its own workspace because Cargo patch tables are not transitive. Root release settings also apply to the promoted GPU products; H25 qualifies their frozen actual release binaries across the five healthy backend profiles. New source/feature changes retain separately scoped gates.

The layout gate removes the former `probes/` directory completely: production
GPU code and its notices belong under `crates/zerogravity-gpu`, independent
llama.cpp clients under `tests/reference/llama_cpp`, and the fixture extractor
under `tools/gpu-qualify`. Historical paths in dated reports remain historical.
Record retirement and any remaining work in the relevant plan TODO; obsolete
compiled objects and executable bodies are disposable build products. Audit sidetree ancestry, unique work and live users before
removing a merged or superseded worktree.

**Current work:** Follow [the plan TODO index](../plans/TODO.md) and
[AGENTS.md](../../AGENTS.md). Layout and contention-first concurrency review are
complete; no left-right migration was adopted. A1 cache is complete; A2/A3
acquisition precedes further GPU optimization. Model basics retain their
separately authorized CPU-first scope.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


## 1. Language and architecture constraint

The user has explicitly requested **entirely Rust** for the engine. Rust is the authoritative implementation language for model and serving logic and proposed owned GPU kernels. This is not a Rust wrapper around any external inference engine. Calls to GPU drivers, OS APIs and possible vendor math libraries via Rust FFI are a separate open decision (O-003); avoid claiming that writing Rust GPU kernel code eliminates ROCm/CUDA/Metal binary toolchains.

**Source language is not a speed feature**: architecture-specific instruction selection, memory traffic, graph capture and launch scheduling must be benchmarked.

## 2. Historical proposed Cargo ownership map

**v4 minimum:** start with **one library crate + one CLI binary** (or 2–4 crates if incompatible GPU compiler/source/toolchain restrictions justify). The older multi-crate tree below is a **possible eventual separation**, not tasks to scaffold immediately. Extra traits only when distinct provider code or architectural tests demand them; one Rust build must work CPU-only without GPU/NPU compilers.

```
zerogravity/
  Cargo.toml
  Cargo.lock
  rust-toolchain.toml
  flake.nix                     # candidate for reproducible NixOS setup
  README.md
  .agents/docs/ .agents/plans/
  crates/
    zg-model/                   # artifact parsing, adapters, tokenizer/template
    zg-ir/                      # semantic SSA / state effects, verifier
    zg-compiler/                # EIR, legalization, layout, kernel planning
    zg-kernels/                 # Rust CPU and GPU kernel source/registry
    zg-device/                  # GPU/CPU backends, streams/events/allocations
    zg-memory/                  # global arbiter, expert cache, IO, prefix storage
    zg-runtime/                 # engine/session/scheduler/drafter/commit
    zg-service/                 # APIs, CLI/lib, streaming, metrics
  tests/
    reference/
    models/
    integration/
    adversarial/
  bench/
    configs/ scenarios/ reports/ results/
  examples/
  tools/
    inspect/
    model-convert/
    quality/
```

This is a **proposed tree** and was not created in this planning step. If maintainability improves with fewer crates, merge crate boundaries and use Rust modules. Keep one-way dependencies:
`model + ir <- compiler <- runtime/service`; `device` and `memory` implement interfaces with no frontend dependency; kernels may depend on generated IR schema but never HTTP. Avoid circular crate deps or dozens of tiny traits in the hot path.

## 3. Ownership and concurrency rules

- `ModelArtifact` immutable and hash identified; optional mmap/SSD backing.
- `ModelExecutable` immutable compiled plan, owning kernel caches, tensor layouts and read-only weight residency handles.
- `Session` exclusive mutable owner of state frontier and one or more snapshots; share immutable prefix by typed COW.
- `ResourceArbiter` the sole physical allocation/admission authority. All subsystem caches request leases, never silently allocate giant buffers.
- `DeviceBackend` owns stream/event and kernel lifetimes; asynchronous completion tracked across worker threads.
- `Scheduler` may build multi-session batches but cannot expose mutable session state to unrelated requests.
- `Frontend` owns roles/messages/templates/HTTP streaming only; no raw device memory or direct model-specific kernel invocation.

Cross-thread messaging uses bounded channels; avoid blocking async executor with `hipDeviceSynchronize` inside request critical path. Rust `Send`/`Sync` impl on FFI GPU objects requires proof of driver/device-affinity safety; no blanket unsafe impl.

## 4. Public data contracts

`Result<T, ZgError>` or equivalent rich typed error hierarchy from INVARIANTS. Strong opaque IDs and generations: `ModelId`, `TensorId`, `SessionId`, `StateVersion`, `BufferLeaseId`, `DeviceEventId`, `PlanKey`. Avoid raw integer IDs crossing protocol layers without validation. Persisted snapshots have independent versioned schema, not Rust `repr(C)` memory dumps.

Model and kernel registries advertise capabilities rather than guessing from names. Registry operations support `can_lower(op, device, numeric_contract)` and return precise unsupported reasons. Dedicated reference implementation for all new semantic ops.

## 5. Core hot-path performance rules

- Compile/resolve graph regions and weight pointer layouts at model load; no per-token string lookup.
- Avoid boxed dyn trait dispatch inside GPU loops; a few calls at step region boundaries acceptable if benchmark cost negligible.
- Reuse scratch buffers; no per-expert allocation/free churn.
- CPU control to GPU command streams asynchronously; event batching and graph replay where supported.
- Pack kernel args into small verified struct; state pointer revisions/device table values may update safely without CPU graph rebuild.
- Prefill/verify/decode plan variants chosen by scheduler from bounded set.
- Hardware-specific Rust kernel source may use target intrinsics via backend extensions, with portability fallback.
- No global mutex covering all sessions during long GPU operation; separate session mutation lock, model immutable read state and cache lease operations.

## 6. Engineering ergonomics

Prefer explicit, small modules and easy-to-review changes. Every hot kernel includes a comment explaining shape constraints, numeric rounding and expected memory layout; every stateful op documents snapshot/rollback and causal position. Avoid generic framework architecture when concrete operator definitions suffice. Keep mutation APIs narrow; assert invariants in debug and return robust errors in release.

Use `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings` appropriately (GPU feature targets may differ), `cargo test`, Rustdoc, no undocumented `unsafe` blocks, sanitizer/Miri as support allows. Use `deny(unsafe_op_in_unsafe_fn)` and other idiomatic lints; FFI memory rules individually audited.

## 7. Unsafe and external code

GPU device pointers, memory registration, mapping, inline ISA and Rust compiler kernels may require `unsafe`. Concentrate unsafe code behind well-tested wrappers:
- provenance/size/alignment validated;
- device and stream identity attached;
- event-backed lifetime proves no use-after-free;
- no unchecked transmute to quantized packed layout;
- no dereference of stale GPU/host-mapped ring pointers;
- speculative cache snapshot stays immutable under child mutations;
- no unbounded allocation from untrusted model fields.

Licenses/attribution for any copied third-party kernel matter; Specialized engine cores aren't publicly reusable. Gufo/Maya/DwarfStar/GGML may offer reusable source subject to actual license, but user wants Rust source; porting from open source requires retaining required attribution and documenting derivative origin. Do not copy external proprietary code from binaries.

## 8. Toolchain and build variants

Candidate Rust toolchain pinned in `rust-toolchain.toml` and lockfile. For NixOS, use flake development shell if selected by user, pinned Rust compiler + GPU compiler toolchains, ROCm libraries through Nix overlays with host driver constraints. The shell should print exactly which environment vars/device permissions required; cannot assume Ubuntu `apt`, current user render group or environment globally.

Separate feature flags:
- `cpu-reference` baseline and `cpu-simd`;
- `rocm`, `cuda`, `metal`, `vulkan`, independently compile/condition;
- `distributed`, `multimodal`, `server`, `profiling`;
- device-specific experimental kernel pack features if needed but never incompatible semantic bypass.

Build should not require an AMD GPU simply to compile CPU tests. GPU availability probed at runtime; error if user selects an unavailable device, generic CPU used only if requested/allowed and model fits.

## 9. CI and physical GPU testing

**CPU CI on every PR:** fmt, clippy, unit/reference, property, format fuzz smoke, generated IR verifier, doctests, build of portable CLI, no model download/network. Model tests use tiny synthetic fixtures with legal redistribution and pinned checksum.

**GPU CI on supported runners:** at least ROCm `gfx1151` development device for extreme path, NVIDIA and Apple where available; kernel parity and sanitizer, full models only if user has access to licensed test weights. PRs changing a backend must report which hardware tests were skipped; compiling `--features rocm` is not execution proof.

**Performance CI:** repeatable alternate A/B with machine exclusive lock, fixed model quant, power and driver telemetry; not run concurrently with normal CI on same GPU. Baselines pinned; small gains require uncertainty report. Preserve raw artifacts and negative experiments.

**Security/fuzz CI:** GGUF/safetensors/tokenizer/cache parser, HTTP payload, distributed protocol and snapshot corruption. Model-file readers remain safe under untrusted inputs.

## 10. Dependency/version governance

Pin runtime dependencies and GPU toolchain versions; record SPDX licenses, source URLs and security patches. Prefer stable Rust crates for HTTP/serialization/IO, avoid dependency on PyTorch/Transformers for inference runtime. Python may be an offline reference/calibration tool (separate `tools`), but not mandatory for `zerogravity serve`.

Plan cache key includes compiler/kernel/backend revisions; do not reuse compiled GPU binaries blindly across driver/ISA changes. Offline mode must not contact HF or fetch executable plugins. Dynamic third-party kernels need an explicit trust/signature scheme, not automatic code execution.

## 11. Pull request checklist

- Purpose and affected semantic operators/sessions, prior baseline and exact benchmark setup.
- Independent oracle tests, model/quant/reference fixture, numerical difference and quality.
- Buffer/state effects and fallback legality; COW/leasing/cancellation tests if touched.
- Physical hardware tests and omitted devices recorded.
- Performance A/B complete model and CPU scheduler cost; no unjustified persistent complexity.
- Update the plan TODO with implemented/tested work and remaining items; change design docs only when the design changes.
- Source license and artifact provenance.
- No secrets or private prompt logs in commits.

A failed performance experiment can be valuable but should not leave dead code, unreachable flags and redundant buffers without good reason. Record failure in research log.

## 12. Build/release artifacts

Proposed CLI/library artifacts and crate publishing remain OPEN. If binary signed releases pursued, automate reproducible build profiles, checksum/SBOM, distro-specific GPU dependency notes and tests. An executable must report its runtime/kernel plan and driver identity for reproducibility, and reject unsupported GPU variants rather than silently execute wrong ISA.

## 13. Agent work tracking

Read [the plan TODO index](../plans/TODO.md) and [AGENTS.md](../../AGENTS.md).
Keep the plan's provenance commit, implementation/test checklist, completed
results and unfinished work in its TODO. Parallel agents use separate run TODOs
until the coordinator integrates them. Use actual tests appropriate to changed
source and hardware. No generated handoff or evidence inventory is required.

## 14. Open structural decisions

Exact crate split, Rust SDK surface, kernel compiler, OS support order, Git initialization, license, CI host hardware, code-generation strategy, checkpoint serialization lib, API auth crate and on-disk result schema are proposals and must be validated incrementally rather than prematurely frozen.

## 2026-10-10 integrated v2 addendum — minimal Rust crates and newer GPU compilers

The historical research implementation plan proposed ~20–25 separately versioned crates, Rust 1.87, cubecl 0.11.0 and a vendor C++/HIP escape. **These are historical council choices, not necessarily user-confirmed requirements.** Start with ~8–12 cohesive Rust crates/modules; preserve ownership of model artifacts/SIR, region/compiler/CIR, kernels/devices, memory/state, engine/scheduler, service/UX and quality, splitting when compile-time/release/unsafe isolation justifies it. No giant initial crate scaffold without a running vertical slice. Crate namespace and binary name remain OPEN except project name zerogravity.

GPU kernel source should remain Rust authored under user's instruction; using system CUDA/ROCm/Metal runtime APIs by FFI is a separate open policy from vendoring non-Rust kernel source. CubeCL is a candidate. **Additional Rust kernel compiler options** include a stable CUDA tile-language frontend and an experimental Rust SIMT/PTX compiler; evaluate them as independent KernelProvider candidates before choosing GPU compiler source policy. Their toolchains can live in separate optional crates/build steps; one experimental nightly path must not force nightly for CPU/core.

The multi-IR pipeline is **SIR+RIR+KIR/CIR**, with direct `ExecutableBundle` and no per-token IR walker. Effectful model state and async buffer/event lifetimes are Rust-typed; `unsafe` limited to reviewed device FFI/allocation/inline intrinsic wrappers, no unchecked Send/Sync across device affinity. One `EngineCore` logical owner and typed worker event protocol, no `Arc<Mutex<Session>>` in hot path.

**Build/CI qualification:** source-pin ledger from `research/inference/refs/COMMITS.txt`, compiler kernel/provider versions, Nix dev shell, CPU tests with no GPU required; CUDA/ROCm/Vulkan hardware lanes as available; fuzz SIR effect verifier, CIR timepoint/alias and untrusted GGUF/snapshot; actual physical `gfx1151` benchmark gate. Repo license, user-facing binary name, MSRV and vendored kernel policy are OPEN; keep old research pin list as comparison, not installation orders. Read `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`.


## 2026-10-12 integration — optional XRT/AIE backend and Rust source policy

The supplied Gufo fork implements the NPU program using Python-generated AIE IR and C++ AIE code and host C++ XRT bridge; compiled xclbin and instruction files are executable device artifacts. A Rust wrapper around that external kernel is **not** a Rust-authored owned kernel. Investigate Rust-generated AIE device IR or a Rust→AIE compiler path and XRT device runtime FFI, then confirm license, program hash and real hardware execution. Until feasible and explicitly policy-compatible, NPU KernelProvider must be disabled/blocked and GPU-only remains fully functional.

Module seams: device::xdna probe and BO/events, kernel::aie capability/program, compiler::rir output-row split, compiler::cir GPU/NPU timepoints, memory pinned leases, runtime worker/ModeReceipt. A new NPU kernel must not touch model math, tokenizer/HTTP or independent GPU kernels. CI without XRT compiles/runs CPU+GPU and mock cross-accelerator tests; physical hardware NPU qualification is a separate opt-in lane. Never add automatic privileged fabric clock controls or reusable untrusted device binary downloading.

## v4 refinement — minimum crates, stable Rust interfaces only where proven

The **initial codebase should be one Rust library plus one executable** (or 2–4 crates after proving a device compiler/unsafe-release boundary requires it), organized into modules for model/artifacts, math/effects, trusted TaskDriver, kernel/compiler, device/resource and coordinator/service. The old larger proposed crate tree is a **future optional decomposition**, not the bootstrap to implement. Avoid one crate/trait/dynamic plugin for every AI modality, IR view or accelerator.

Only SemanticProgram and compiled ExecutableBundle are enduring model execution representations; RIR/KIR are internal pass/candidate data, CIR a typed resource command DAG attached to the bundle. A single session/shard writer publishes prepared StateRoot; device suballocators consume global reservation without per-token mutex. `TaskDriver` is a trusted Rust stage controller, not a general workflow language or unrestricted ExternalAction. No mandatory training/autograd compiler in default inference product; future optional GradientProgram/ParameterVersion supported conceptually.

Use feature-isolated source/driver builds so CPU-only Cargo tests do not require GPU/NPU or downloading model weights. Physical kernels are Rust-authored; foreign compiler/device APIs remain policy-gated. Benchmark compiler cold start and host dispatch latency as well as device execution.

## v4.1 — vendor graph compiler adapter without kernel-source confusion

The project maintains **one Rust-owned SemanticProgram, TaskDriver, TaskCoordinator, ResourceArbiter and ExecutableBundle**. Its single `RegionExecutionProvider` interface has a native Rust `KernelProvider` form and optional graph/fabric program forms. No separate accelerator-specific inference server, model store or session manager, no TPU-only/wafer-only semantic IR and no mandatory new crate per vendor. Start CPU-only and one native GPU provider; add optional graph compiler/adapters as separately buildable modules or a few crates only if SDK/toolchain boundaries justify.

An external authorized vendor graph compiler consuming an **ephemeral, semantically checked graph interchange** can be an AOT backend while all logical task/runtime remains Rust. **This is not a Rust-authored target kernel/compiler by itself.** Track exact foreign compiler/driver/device program source-language/IP policy and license separately; missing compiler access results in architecture-compatible but unsupported native execution. A proprietary model text API is an external TaskServiceAdapter, not native backend equivalence.

Backend build/CI lanes: CPU-only smoke; mock graph-only/static-dataflow provider and opaque-state rejection; physical native CUDA/ROCm as available; **separate** real TPU/NPU/wafer/ASIC SDK/hardware verification only where accessible. No build-time downloads/credential requirement for basic Rust library. See [universal provider contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## v4.2 — optional authenticated peer feature, no new inference server

Build core CPU/GPU `zerogravity` with **zero cluster runtime overhead or dependencies** unless network execution enabled. One optional peer/transport module (or crate only if build/security isolation proves needed) reuses DeviceBackend, RegionExecutionProvider, ExecutableBundle/CIR, TaskCoordinator/StateRoot and ResourceArbiter. A RemoteRegionWorker receives checked typed work/StateShard refs and executes the same compiled closed SIR region; it is **not** a Rust wrapper around a separate model server or a new model-specific task loop.

**Minimal first protocol:** manually authorized two peers, supported reliable authenticated stream (TCP/TLS or Rust QUIC choice measured), bounded typed messages, buffer credit, source/destination physical tickets, and per-session generation/epoch idempotency; a local stub/simulator first. No mandatory Kubernetes, DHT, consensus/etcd service, public NAT relay, automatic remote shell or model-repo executable code. More advanced peer discovery/overlay tunnel integration and physically qualified RDMA are optional TransportProvider backends. A user-supplied tunnel adapter is not permission to change host firewall/keys/network settings.

Security source review applies to networking and licensed model/shard data rights. Native RDMA/UCX vendor FFI is separately open under the strict Rust-owned source policy; fallback buffered encrypted stream must be correct. CI: CPU-only/no remote, deterministic mock 3-node topology and fault/relay, then optional real 2-node LAN/tunnel and GPU/collective lanes. Tests mark simulated and hardware-qualified separately. [Placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md).

## v4.3 — one executable and one library for CLI, router and multi-model serving

`zerogravity run <model>` (proposed syntax) instantiates the same Rust ModelCatalog/ModelRouter/ResidencyManager/TaskCoordinator/ResourceArbiter/RegionExecutionProvider as the `serve` endpoint. **A CLI direct run does not require a local service process**, although an optional trusted same-user socket reuse mode can lower repeated model startup. No mandatory separate gateway binary, model-specific inference daemon, database, Python virtualenv, dynamic plugin ABI or system-level service installation.

Model catalog and alias/profile overlays can start as modules in the same library; real model loads and model-specific concurrency share physical/DeviceBackend event-pool accounting. Provide optional versioned typed config reader and `config show --effective`, but do not scaffold a large independent control-plane framework before one CPU/GPU source model is runnable. `serve` maps OpenAI-compatible HTTP schemas to canonical TaskRequest; it does not reimplement token sampler, model loader or run a subprocess per model by default.

CPU-only build remains minimal/offline; conditional GPU/NPU and optional authenticated distributed peer modules are independent build features. A remote model API proxy, if ever permitted, is TaskServiceAdapter and not a `ResidentModel` native RegionExecutionProvider. SDK conformance and safe localhost/remote auth/TLS/config precedence are separate quality gates. [CLI/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).
