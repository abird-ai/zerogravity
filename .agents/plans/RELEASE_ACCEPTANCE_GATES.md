# zerogravity — Model-neutral release, quality and composability gates (v4 reviewed)

**2026-10-12 · NOT EXECUTED.** All stage and release gates are proposed verification rules for the original modular AI architecture, not declarations that code exists. Source-specific historical gate versions are preserved in `../prior-art/PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md`. Use `IMPLEMENTATION_ROADMAP.md`, `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, the modular AI reference architecture and composition contracts as authority.

## 1. Approval states

Each capability, not only each model name, has `PLANNED`, `REFERENCE`, `EXPERIMENTAL`, `FUNCTIONAL`, `VERIFIED`, `PERFORMANCE_QUALIFIED`, `UNSAFE_DISABLED` or `BLOCKED` status by (model architecture/checkpoint/quant, task, device/backend/driver, numeric contract, context/phase). A GPU kernel may be performance-qualified for prefill but not decode or a different quant, and an accelerator physically present but power-unsafe is not eligible for automatic scheduling.

A source repo claims a speedup ≠ zerogravity measured a speedup. Unknown trained operator and unsupported model artifact must fail honestly.

## 2. Foundational release gates

| Gate | Evidence | Hard failure |
|---|---|---|
| **G0-A — semantic correctness** | safe real checkpoint/config/tokenizer/quant, Rust scalar/CPU oracle and exact trained ops/position/state | wrong math, unknown op silently guessed, mutable state leak |
| **G0-B — kernel and command resources** | physical GPU Rust-authored kernel parity, legal RIR fallback, CIR alias/lease/timepoint checks and compiled direct dispatch | stale buffer/device pointer, illegal rewritten state, interpreted IR in token hot path |
| **G0-C — useful vertical slice** | model→SIR→compiled CPU/GPU→TaskCoordinator→committed task output and accurate ModeReceipt; forced missing optimized kernel | no real verified output, separate model-specific server required |
| **G0-D — composability** | novel model op + device provider + non-token TaskProgram added without unrelated service/allocator/core rewrite | architecture requires new inference engine |
| **G1 — portable product** | supported models/formats, real task outputs, local safe SDK/API, CPU+validated GPU and strict fallback | incomplete adapter, unsafe input, fake support |
| **G2 — serving/memory** | multi-session and heterogeneous task stages with physical admission, prefix sharing, fair P95 and cancellation recovery | host overcommit, state cross-talk, starvation or uncommitted output |
| **G3 — extreme specialized** | representative dense/hybrid/sparse/MoE/multidevice profile with independent quality and matched speed, device safe, no other engine process | hidden approximation, unsupported/unsafe kernel, false speed ranking |
| **G4 — broader AI tasks/distribution** | accepted ASR/TTS/image/video/embedding/rerank TaskPrograms and remote/multi-device only per verified cell | fake next-token task, irrecoverable task state, partial peer commit |
| **G5 — sustainable release** | security/ABI/schema/source license/reproducibility/performance evidence and extension drill | untrusted executable model code, unreviewed source, broken upgrade |

**No gate passes from synthetic simulation alone where physical GPU/NPU or official model quality is required.**

## 3. Required semantic and numerical quality controls

- Source checkpoint hash, tensor manifest, tokenizer/template and quant dialect verified before inference.
- Mathematical operator/position/masking/routing/compression/recurrence semantics independently implemented and tested.
- New stateful op has state transition/fork/snapshot/restore/commit/abort tests and SIR effect verification.
- Strict `Reference` vs validated numerical-tolerance vs approximate opt-in must be visible and not conflated with stored quant.
- Full logits and layer metrics where available (maxabs, RMS, KL, greedy match, sampled target distribution, context/long-task quality).
- Encoder pooling/reranking, ASR audio timestamps/WER/quality, TTS speaker/acoustic/PCM quality, image generation/edits, and video fidelity require model/task-specific independent acceptance. Do not use language perplexity as universal task score.
- Ongoing tool/reasoning/grammar and partial output/state read/write boundaries tested for cancellation and safety.
- Every specialization can be disabled to compare against the exact portable model; safe fallback may require layout conversion/replay and will report cost.

## 4. Resource and async safety

- Physical RAM/VRAM/OS headroom and real unified/device addressability distinguished, cgroup and pinned staging included; no dangerous default overcommit.
- All GPU/NPU/CPU/SSD/remote buffers have validated ResourceLease and generation, read/write aliases, timepoint/event quiescence and failure scope.
- Expert source loads, disk lookup rows, snapshot files and media streams validate exact artifact/offset/length/schema; EOF or cache generation mismatch fails closed.
- A trained routed expert or sparse attention selection cannot be skipped merely because its parameter tier is cold.
- Temporary prefill/diffusion scratch overlaps cache slabs only after old readers retired and lease fencing.
- Device lost/timeouts may leave DMA outstanding; quarantine until safe, no stale pointer reuse.
- Distributed collectives must agree model/quant/semantic state/accepted frontier or abort/rebuild.
- NPU and GPU sharing a known unsafe fabric/power domain stay OFF until physical safety qualification; no automatic root/sysfs/firmware/BIOS/IOMMU/clock changes. A process watchdog cannot recover whole-machine hangs.

## 5. Task and service acceptance

A task must declare `TaskKind`, logical input/output ValueTypes, stage DAG, iteration/control and output frontier/cancel behavior, model/adapter/codec identity and resource budgets. At least one non-token task must pass reuse of the same compiled region/device/TaskCoordinator without a model-specific runner fork.

Service/SDK:
- canonical TaskRequest and typed output, model/error/capability reports;
- streamed text/audio/image/video/structured results only when committed/valid according to task frontier;
- sampled text proposal never emitted before authoritative target verification;
- request cancellation/timeout leaves previously emitted valid output intact, rolls back uncommitted state and returns correct terminal reason;
- tool external actions require their own permission sandbox; no arbitrary model-file remote executable code;
- local-only bind by default, remote authentication/quota, private prefix/media state and redacted metrics;
- correct token counts versus audio frames, video frames and diffusion steps, accurate ModeReceipt for kernel/device/quality/fallback.

## 6. Long-run composability release test

Required demonstrator under controlled real code:
1. Add a model architecture composed of existing SemanticOps with only ModelDefinition/tensor manifest addition.
2. Add a genuinely new recurrent/attention operator with CPU reference, state schema and at least one generic backend.
3. Add a kernel provider for a new device ISA, with no ModelDefinition or service type modifications.
4. Add an encoder or iterative TaskProgram with distinct stage unit/output and no decoder-only loop.
5. Add a new storage/IO tier or async command mapping without changing math model and task code.
6. Disable extreme providers; semantically supported model/task runs portable/reference with correct state or honest physical capacity refusal.
7. Quantify unrelated modules changed, schema churn, Cargo build time, runtime CPU overhead, portability degradation and kernel performance.

**Pass rule:** functionality/quality/state/resources correct, no duplicate serving engine and no project-named core types, and ordinary serving remains operational after each extension. If patch changes every layer, revise the interface rather than add a model-name exception.

## 7. Performance qualification

Every performance claim has:
- matched source model, checkpoint/quant and task/control/sampler, prompt/context/regime and output units;
- actual CPU/GPU/NPU/device/fabric/driver/firmware and physical memory/cache state;
- direct baseline (same general path) and external peer reference when legitimately comparable;
- repeated paired ABBA/BAAB, raw time distribution, p50/p95/p99 TTFT/interunit/completion, energy, memory and source quality;
- distinct tokens-per-second for autoregression vs prefill vs speculative accepted output; distinct audio/video/encoder and image throughput units;
- explicit numerical error and resource safety failure logs; no favorable hardware-clock confounding.

**Promotion:** performance-qualified specialization becomes eligible for plan selection only in its validated shape/driver/model/quality/safety cell. Unqualified kernel remains experimental and not default. Worsened p95 or quality can outweigh isolated GEMM speedup. Source-author claims remain in prior-art evidence, never baseline "achieved" by zerogravity.

## 8. Source/license/security qualification

- Exact code file license/SPDX/header, required NOTICE and source provenance/commit reviewed independently of repository root badge.
- MIT-licensed candidates considered only after audit; any different embedded file/model-weight/device executable license handled separately.
- Non-MIT source-specific APIs/code do not become normative interfaces or copied implementation without explicit user policy resolution. Public mechanism research can inspire independent design.
- User's Rust-owned implementation policy: a foreign GPU/NPU kernel wrapped in Rust does not alone qualify as Rust-authored owned kernel.
- Compiler source/driver device-program trust, toolchain provenance, cache invalidation, snapshot schema and model artifact hash verified.
- No hidden third-party inference engine. Source project names retained in prior-art records, not architecture types.
- No release despite incomplete critical parser/security/physical-memory/async ownership checks.

## 9. Evidence/report schema (conceptual)

```text
result: PASS | FAIL | NEUTRAL | INCONCLUSIVE | BLOCKED
phase, gate, run_timestamp, git_sha or source_hash
model/task/quant/tokenizer/semantic_op/state_schema IDs
kernel/plan/driver/device/clock/fabric program identity
input/output task geometry, concurrency and mode receipt
reference numerical and full task quality outcomes
committed units/time, TTFT/P95, physical/pinned memory, energy
resource IO/cache metrics, safety/failure, privacy/authorization
raw artifacts paths, repetitions, uncertainty, skipped reasons
next decision / guarded fallback / reviewer source license
```

An unknown metric remains null/omitted rather than a misleading zero. Tests, raw data and hardware/OS error reports link the exact implementation state. No user private prompts or copyrighted weights in public artifacts.

**Documentation closure:** every passing gate updates roadmap/status, relevant source/policy decision, handoff and integrity manifest. Until tested, all phases remain NOT_STARTED, regardless of how complete the plans look.

## v4 mandatory architecture acceptance refinements — 2026-10-12

These rules supersede older shorthand claims of “all domains commit atomically” and “one owner for all allocations” where those phrases did not specify physical mechanics. Add to G0/G1/G2/G5 without inventing new product release dates.

1. **Single publication owner:** no mutating per-domain public `commit`; prepared versions of all required state domains validate same TaskFrontier, event readiness, schema and numerical contract; TaskCoordinator/shard performs one `PublishRoot`, only then emits committed output sequence. CIR/device commands do not publish semantic session state.
2. **Async-first hot path:** performance-qualified device-native StepProgram avoids mandatory host router/sampler/token readback and semantic IR interpretation per inner iteration when model/support permits; correct host-loop fallback is allowed and honestly labeled `FUNCTIONAL`.
3. **Hierarchical memory:** global admissions reserve conservative physical memory, device-local suballocators own fast generation-tagged leases; no global lock per op/token. OS memory can change externally; no absolute no-OOM promise. Inject allocation failure, stale lease and delayed DMA.
4. **Shape guards and bounded compilation:** fixed-rank dynamic dimensions, ragged offsets, divisibility/context/quant/ISA constraints checked before launch; no unbounded JIT/variant explosion. Cold unsupported shape selects safe prepared portable plan or explicit diagnostic.
5. **One minimal TaskDriver:** no uncontrolled agent/workflow language or `ExternalAction` in core. Non-token model stages have their own frontier/output and share worker/resource backend, without fake LM sampling state.
6. **Fast specialization containment:** a full fused layer/model candidate is replaceable at a closed region with equivalent values/state/effects/numeric contract and can be forcibly disabled with portable model still supported.
7. **Distributed and transport honesty:** coordinator epoch fencing and Ready cohort precede logical root publish; output event sequence/backpressure/cursor documented, no network exactly-once delivery claim without explicit durable protocol.
8. **Training is future optional:** architecture allows ParameterVersion/GradientProgram/optimizer state TaskDriver without modifying inference state owner or forcing autograd/training dependencies into first release. Synthetically prove the seam, do not claim trained models run.
9. **Fewest mandatory abstractions:** SIR and compiled ExecutableBundle are the only enduring representations; region/kernel/command views are compile internals. One library+binary (or few crates) preferred over a nine-crate stub hierarchy; extra abstractions require measured value.
10. **Project neutrality and evidence:** source-name/foreign kernel dependencies confined to audited prior art, actual GPU driver/FP8/BFP8 numerical/provider choices validated and explicitly qualified; no silent trusted model approximation or unsafe privileged host changes.

**Dedicated tests:** `S0-V4-A…J` in integrated Stage0 and `H32…H43` in experiment program. Inability to do one of these tests at current hardware is **BLOCKED/SKIPPED**, not VERIFIED. Architecture is still **design only**.


## v4.1 backend portability and honesty gates — 2026-10-12

1. **Universal executable kind:** a `RegionExecutionProvider` candidate can cover a native kernel, captured graph, AOT whole-region graph or static dataflow fabric program, returning an `ExecutableBundle` and typed async ready/fail event. No new model-specific server/allocator/session writer and no device-specific semantic IR.
2. **Full state bridge:** stateless region can be qualified without mutable snapshot; **stateful region** requires privately prepared state, snapshot/restore or bounded recompute/checkpoint under exact schema/epoch/cancellation. Opaque in-place unrecoverable state must be rejected for general speculative/forkable decoding even if program compiles. Device program cannot publish StateRoot.
3. **Graph shape/precision proof:** bounded fixed-rank dynamic dimensions, ragged/sparse offsets, native quant scale/block representation and compiler region shape; unsupported layout/op/quant gets legal smaller region or typed refusal, not uncontrolled AOT recompilation or wrong model arithmetic.
4. **Program lifecycle:** compile/codegen/upload/warm/resident/teardown measured and budgeted, compiler/runtime/driver/firmware versioned; recompilation/cache invalidation tested. No leaked compiled handles or raw pointers in persisted snapshots.
5. **Physical topology and collectives:** coherent SoC vs discrete GPU vs multi-GPU rack vs on-chip SRAM/static fabric have distinct memory/IO/transport/link resources, rank/epoch fault fencing; one system marketing memory capacity cannot be treated as one pointer.
6. **Remote API separation:** vendor inference endpoint without trusted exact model op lowering and StateBridge is an independently authorized external **TaskServiceAdapter**, with visible privacy/credential/egress/model-sampling differences. It is **not** proof of native zerogravity inference on that accelerator.
7. **Rust-source/license capability:** using a vendor graph compiler for device executable is not evidence low-level kernels/compiler were written in Rust; classify it as optional authorized external toolchain policy, with exact source/program provenance and license. Missing SDK/compiler/hardware reports `ARCHITECTURE_COMPATIBLE` not `FUNCTIONAL`.
8. **Physical qualification:** each hardware/model/quant/context/task/driver/compiler variant needs official/independent semantic/state/quality and matched real end-to-end throughput/p95/energy tests before `PERFORMANCE_QUALIFIED`. Theoretical peak and source-vendor reported speedups aren't our evidence.

**New optional tests:** S0-FABRIC-A…H and experiments H44…H55. The portable CPU/GPU model remains valid when no external compiler/wafer/TPU/ASIC hardware can be accessed. Details: `../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md` and source-attributed `../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md`.

## v4.2 distributed placement and networked execution acceptance

**New distributed release cells are optional and separately qualified** from local model/kernel execution. Architecture support for a placement topology is not real distributed functionality.

1. **Independent placement proof:** same exact SemanticProgram/TaskDriver can independently map compute, immutable parameters, mutable KV/recurrent/indexer task state and immutable prefix caches to distinct authorized nodes, with one TransferPlan and StateRoot schema/lease.
2. **No runtime forks:** RemoteRegionWorker runs the same compiled closed region and DeviceBackend as local, without another API server/model-specific allocator/TaskCoordinator. Single host runs with zero cluster daemon/discovery/network hot-path overhead.
3. **Layer + cache correctness:** per-layer-group remote model worker keeps the layer's KV/recurrent state, receives only necessary activation boundary; model logits/full state match all-local reference after chunked prefill and C1 decoder, including cancellation.
4. **Separated compute and state:** remote cache hydrate vs remote attention beside KV vs direct remote KV reads have different cost/StateBridge requirements and must not be conflated. Slow/relayed WAN per-token KV readers ordinarily refused when worse, not silently used.
5. **Operator-specific token sharding:** mathematically exact masked/causal attention softmax shard summaries reference-tested, no unproved generic GDN/KDA or sparse selected indexer partition.
6. **Full hybrid P/D:** destination reserve and verified complete KV+recurrent+indexer+RNG/grammar+modal state transfer with committed frontier, or explicit replay/fallback.
7. **Tunnel, peer trust and privacy:** explicit authorized node identities, private transfer policy for source weights/prompts/KV/media; authenticated bounded control and reliable bulk, direct/relay cost measured and no assumed RDMA over VPN. No automatic remote shell, public service, credentials or model weights redistribution.
8. **Epoch/failure:** stale StateShard lease, wrong quant/model/adapter, lost ACK, network partition, stale coordinator or live DMA failure cannot publish incorrect TaskSession root or leak output. No unimplemented durable exactly-once or automatic HA claims.
9. **Topology plan lifecycle:** compiled plan frozen during active collective/attempt, safe relocation only at committed TaskFrontier with reserve/transfer/hash/epoch fence, prepared cohort and physical buffer retire; fallback to old root or replay.
10. **Real benchmarks:** two authenticated physical nodes and direct LAN and optional tunnel variants require exact model/quant/task/hardware, bytes/step, P95 RTT/bandwidth/jitter, CPU encryption overhead, TTFT/ITL/committed throughput, energy, device headroom, failure and quality. A larger fitting model is a **capacity** achievement even when C1 t/s lower.

**New optional Stage0 S0-DIST-A…L** and **experiments H56…H70**, all PLANNED/NOT RUN. Node/tunnel connection/deployment requires explicit user authorization and separate security test. Source research remains non-normative; see [distributed placement](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md).

## v4.3 multi-model router, simple CLI and API conformance release gates

**Product correctness is an explicit gate, not just a convenient CLI sketch.** A release must show the **same inference engine** serves direct in-process CLI and one multi-model HTTP endpoint with independent physical model lifecycles, no extra per-model serving engine.

1. **Blissfully simple local:** one installed Rust binary plus a supported local checkpoint runs with `run model` and no config, daemon, internet, remote/tunnel/root change or mandatory model registry authentication. Hardware auto-plan is qualified; unfit model gets precise actionable refusal, not silent quant/model downgrade.
2. **Exact model catalog/aliases:** immutable source SHA+model math+stored quant+tokenizer/template+adapter, separate ModelAlias/Profile revision, ResidentModel/Replica and TaskSession. Hot alias changes affect only future requests; old active sessions keep exact source and full StateCompatKey.
3. **Many-model physical safety:** catalog may hold many models; simultaneously resident models/replicas only when actual RAM/HBM/UMA/OS reserve+graph programs+per-session context fit. Load/warm/idle unload deduplicated and bounded, never evict active unsnapshotable state, live DMA or prepared speculative buffers. User keepwarm/pin cannot override safety.
4. **Exact replica load balancing:** route among equivalent semantic/quant/adapter models with sticky StateRoot/cache locality, queue/P95 and permitted device/tenant/peer; safe explicit full-session migration at committed root only. A configured different-model selector cannot masquerade as an exact model request.
5. **Fair multi-model execution:** cold giant-model wait cannot starve warm small interactive traffic, no unbounded model-load thrashing, bounded per-model and global queue with typed overload errors, existing native within-model batching and per-task scheduler unchanged.
6. **Per-model configuration:** versioned human-editable effective defaults/global/model/alias/CLI/request scopes, deterministic precedence and provenance, invalid/unsupported option diagnostics, immutable/security/physical/precision guards never overridden. Profile changes requiring recompile/load create new deployment; no in-place live model mutation.
7. **OpenAI-compatible subset conformance:** actual tested `GET /v1/models`, `POST /v1/chat/completions` streaming/nonstream JSON/usage/finish/tool-data, Python/JS SDK current pinned versions; embedding if real encoder qualified, Responses stateless subset with genuine distinct event schema if implemented, stateful store/tools separately approved. Unsupported request fields not silently ignored.
8. **API security/privacy:** loopback default, remote binding requires configured auth/TLS/tenant/cors and separate authorized native admin; no private model names/paths, filesystem/media SSRF, user prompt telemetry/credential leakage, automatic remote/tunnel/cloud enrollment.
9. **Multi-model TaskProgram:** target/draft, embedding/reranker, vision encoder/text model and TTS/vocoder are separately bound model artifacts with shared parent task budget; cannot treat different source/quant as replicas.
10. **Cost/performance:** direct local `run` vs router `serve` must not incur a **per-token gateway/second-inference-scheduler** overhead; measured real model/quant/driver C1/C8 TTFT/p95, cold/warm load, memory/energy, cache affinity, distributed placement and denied unknown source responses required.
11. **Client state/stream honesty:** model ID/alias/profile revision and committed output seq pinned, slow client backpressure/cancel, no rejected speculative output or fabricated exactly-once delivery. Stateful Responses/conversation storage needs explicit authenticated retention.
12. **Status discipline:** failed API conformance = unsupported endpoint/field/capability, not “fully OpenAI-compatible”. No model download, daemon/service installation or public API bind without user-approved policy.

**Evidence:** S0-SERVE-A…J and H71…H86; real supported model/TaskDriver/hardware+SDK needed before functionality or performance marks, synthetic architecture proofs only qualify design. The prior-art source UI/SDK research is non-normative; see [model router](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [CLI/API](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

## P00 packaging-only Rust binary and release artifact gates — U-023

This is a **separate bootstrap acceptance class**, not a claim the model runtime or the main product release is ready.

| Gate | Status (2026-10-12) | Required test |
|---|---|---|
| Cargo toolchain and package metadata | **PASS (local)** | Rust 1.95.0, Cargo.lock offline, no runtime dependencies, version 0.1.0 |
| CLI honest bootstrap behavior | **PASS (local)** | `--help/--version` success; `run/serve` explicit nonzero unimplemented refusal |
| Rust format/unit/integration/lint | **PASS (local)** | 4 unit + 3 CLI tests, rustfmt and clippy -D warnings |
| Local Linux x86-64 release build | **PASS (local)** | `cargo build --locked --offline --release`; `zerogravity --version` exact |
| Shell and Nix source static checks | **PASS (local)** | Shellcheck four scripts; nine Nix file parse and Alejandra checks |
| POSIX installer offline integrity | **PASS (fixture)** | file:// checksum, successful install/repeat, malformed sidecar and wrong version refused |
| Windows installer runtime | **NOT RUN** | Windows PowerShell x86-64/ARM64 source fixture and real release |
| Nix flake build/check/module eval | **BLOCKED BY CONNECTOR** | `nix flake check`, `nix build .#zerogravity`, `nix eval` hit read-only Nix-store source lock |
| Crane six cross release outputs | **NOT RUN** | actual Linux MUSL x86/ARM, Windows GNU/ARM64, macOS x86/ARM and per-OS smoke |
| GitHub Actions/release publish | **NOT RUN** | repo/owner/license and real tag not established; successful hosted CI/tag release required |
| NixOS future service | **DESIGN/SYNTAX ONLY** | module disabled default, no restart loop; real `serve` not implemented |

**Completion rule:** P00 packaging may be described as a *working local Rust CLI bootstrap*, **not** as a fully cross-platform published release or complete P00 source-policy/model-runtime foundation until the outstanding gates are exercised. Installer SHA256 is corruption detection, not a release authenticity signature. Public project license, owner/repository and Apple/Windows signing policy still open O-072…O-074. [Bootstrap handoff](platform-and-release/TODO.md).

