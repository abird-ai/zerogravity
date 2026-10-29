# Fabric vNext transition — FIRST full Rust skeleton and complete code relocation

**The user's highest-priority implementation milestone** is visible, compilable code structure of the ENTIRE Fabric vNext architecture immediately, with all crate/module ownership layers and small typed interfaces ready for implementation. That must precede the detailed implementation migration and expensive qualification. Design only in this current conversation; no Rust was edited.

## 1. S0 — complete structural skeleton in one signed logical commit

The first source implementation milestone must introduce:

- NEW `crates/zerogravity-engine`: model-neutral Runtime, ModelCatalog/ModelInstance, Router, TaskCoordinator/TaskDriver, Scheduler, PreparedExecutable/ExecutableBundle, Residency, Placement and committed output handling. Uses existing one shared physical ResourceArbiter, not a separate allocator.
- NEW `crates/zerogravity-serving`: independently reusable server library with ServerBuilder/ServerHandle, HTTP/OpenAI/SSE, auth/ingress/output lifecycle, metrics and future media protocol modules. Depends on engine/core, never facade/CLI/Qwen/Gemma/GPU.
- EXISTING `crates/zerogravity`: fully visible public SDK sections Runtime, models, artifacts/pull/cache, devices/placement/dry-run, direct run and events, pipelines/config, serving builders/reexports, diagnostics, built-in registration.
- EXISTING `zerogravity-cli`: target source-level thin shape args/commands/output/signals/entrypoint. Temporary old CLI implementations can coexist only during S0–F2; F3 deletes old runtime/HTTP/business implementation, and CLI source after F3 uses only public zerogravity library for all project work.
- EXISTING `core`: semantic program based on current ForwardPlan, source/quant/numerics/identity/task/state/provider/value/placement modules visible, including typed future media, NPU and remote capability schemas but not fake backends.
- EXISTING `artifacts`: containers, exact codecs and source reads, immutable model store/cache/acquisition and future derived quant manifest.
- EXISTING `CPU/GPU`: clear operator/quantization/precision/physical execution/transfer/memory/capture/qualifier ownership; existing bodies retained and moved only if needed.
- EXISTING `model-Qwen/Gemma`: authoritative family config/roles/source binding/forward/state/tokenizer/specialization/reference homes, not another engine.

Each source-visible top-level module must have at least one meaningful Rust type, function, enum or contract, not empty directories. All target crates and feature selections must compile, have one real acyclic Cargo DAG, and document their responsibilities. Import or link the complete expected tree from ARCHITECTURE_AND_CRATE_CONTRACTS.md.

**S0 is a single structural milestone** and should be one coherent signed commit following only fast architecture, Cargo-check, formatting and small contract/refusal tests. It does NOT claim new model functionality.

## 2. Safe TODO strategy in real Rust

The user explicitly allows `todo!()` placeholders where needed for the first structural skeleton. Enforce the safety boundary:

- Every ordinary public/CLI/HTTP/live Runtime operation must either delegate to a proven existing implementation or return a clear typed `Error::Unsupported` / `NotImplemented` including missing capability and status.
- A `todo!()` macro is acceptable only inside an explicitly unreachable, private, development-only or future-disabled module. It MUST NOT be on a production, release, default, "all" feature, supported device, or live request route. Never use it to construct fake Ready, tensor bytes, logits, source permits, working TaskSession, buffer lease, admission credit or authenticated model support.
- Future Q1/Q2/Q3/NVFP4 codec, multimodal TaskDriver, remote worker or graph-only accelerator APIs can have **types and explicit refusal** in S0 without physically implementing them.
- S0 may leave current supported HTTP/runners on the existing CLI implementation temporarily; F0–F3 migrate them without compatibility branches remaining at completion.
- A module exists and compiles in S0 even when its exact source/model/backend operation has not yet been implemented. Library capability reports must distinguish structural API availability from actual model/physical support.

Example permitted public contract:

~~~rust
pub fn prepare_unsupported_media(spec: &MediaTaskSpec) -> Result<PreparedTask> {
    let _ = spec;
    Err(Error::Unsupported("media TaskDriver not source-qualified".into()))
}

#[cfg(feature = "internal-stub-only")]
mod never_selected_by_runtime {
    fn future_kernel_lowering() -> () {
        todo!("private planned implementation")
    }
}
~~~

The internal-stub feature must not be included in release/all-features build surfaces unintentionally; best use typed refusal for most skeleton interfaces. Avoid a new dead code tree requiring clippy allowances everywhere: favor small public types over hundreds of placeholder functions.

## 3. S0 mechanical action order and integration

Before coding:
1. Recheck actual current HEAD/status, AGENTS.md, active plan ownership and concurrent agents; source may have advanced beyond the historical fadf2b1 snapshot. Never overwrite unrelated or concurrently owned work.
2. Decide exact final crate/type names once against current source and avoid unnecessary public compatibility aliases. The user explicitly permits breaking pre-v1 APIs.
3. Freeze input source identities, existing actual capability inventory and the lightweight migration smoke commands. Do not run heavyweight model/device suites yet.

Structural source unit:
4. Add engine and serving crates to workspace with safe default feature graph; no concrete model/GPU import in engine; no facade/CLI import in serving; CLI depends only on facade for project behavior.
5. Add core semantic/model/task/encoding/numeric/provider/placement/root type homes, reusing original ForwardPlan and low-level ResourceArbiter/SessionAuthority implementations as the one actual source.
6. Add facade Runtime/TaskRequest/Models/Devices/Artifacts/Pipeline/Serving API surfaces and cold builtins registration; new methods either delegate or fail closed.
7. Add engine Scheduler/ModelInstance/TaskCoordinator/ExecutableBundle/CommittedEvent types and self-contained lifecycle; no automatic source loads or sockets.
8. Add serving ServerBuilder/ServerHandle/HTTP/SSE placeholder modules and a safe builder; do not bind without explicit start and actual engine capability.
9. Add typed task/media/clock/snapshot/stage/remote/optional accelerator descriptors and explicit future capability statuses; no functional media/remote claim.
10. Add CLI target args/commands/output/signals modules and path-map; no forced switch of all current production until corresponding migration phase.
11. Extend existing architecture checker to enforce all target crate edges and no production source imports of model test fixtures. Update relevant Cargo feature checks to include new packages, without making CPU-only build pull CubeCL/native GPU compiler, HTTP or acquisition dependencies by default.
12. Run small checks; review no panic on reachable route; signed structural commit, explicit S0 result in side plan.

**Exit:** all target structural layers *exist in actual compiled Rust*, module ownership recognizable, public crate exposes a coherent API with support-aware errors and independent serving compile capability. Existing supported code has not been silently discarded. Zero new physical/model runtime acceptance from skeleton.

## 4. S1 — codec/precision/source ownership

Move real source code and callers, not just declarations.

- core/src/quant.rs -> typed `ScalarDType`, `QuantScheme`, `StoredTensorEncoding` with exact GGML/safetensors mappings, checked byte length and unknown refusal.
- artifacts/src/quant.rs, quant/q4.rs, quant/q6.rs, quant/q8.rs -> exact artifacts source codecs. Retain F16/BF16 edge handling, source offsets, scales, malformed shape tests.
- CPU source.rs/projection exact original GGUF dynamic-activation Q8_0/Q8_1/Q8_K arithmetic and reduction -> CPU owned quantization/precision operators. Preserve pure F32 dequantized oracle as separately named diagnostic, not exact source dot.
- GPU quantization/ q4_0/q4_1/q6_k/q8_0, activation, batch projection -> keep physical kernel bodies; adapt code to shared encoding/numerics type and qualified source/shape candidate.
- Q1/Q2/Q3/NVFP4/EXL3 README-only homes remain unimplemented; Q4_K original CPU matvec still deliberately refuses despite scalar decode.
- Migrate all source/quant parser/tests and callsites coherently; delete obsolete pre-v1 legacy names once new APIs serve all callers, no shadow codecs.

Suggested one S1 signed commit or split S1.a artifacts, S1.b CPU, S1.c GPU only if each subunit compiles and has targeted fixture coverage.

## 5. S2 — family semantics and GPU/qualification

- Continue already accepted a5ed7de/263ff79 Qwen/Gemma family containment. Do not reintroduce model→GPU→model cycle or move all model-specific source into core.
- Keep one Qwen source-exact traversal and Gemma math with independent PLE/softcap/shared-KV/MoE as appropriate.
- Build one canonical SemanticProgram from existing ForwardPlan, capable of host reference or prepared closed-region implementations; no duplicate model traversal.
- GPU physical provider implements reusable actual kernels/ABI/leases; family source adapter owns original model eligibility and tests; application facade cold registration joins families and physical candidates.
- Preserve recent parameterized CUDA GDN Conv/KKT/Recompute/State/Output and fresh-prefill source code/image hashes without rewriting real CUDA math just for new module naming.
- Separate production implementation config from historical finite prompt/recipe cases; test-only source/fixture orchestration in tool/tests, production always requires actual semantic/source/profile/device/shape permits.
- Keep physical CUDA/HIP/Vulkan qualification independent, no code rename giving false capability. Existing frozen/held gfx90c and physical device scopes remain held.
- Delete obsolete direct model-specific routing from physical worker only when generic adapter exists and tests compile.

Suggested S2 signed commit or S2.a family+semantic, S2.b physical provider+qualification if needed for reviewable change.

## 6. S3 — canonical session + resource + event semantics

- Reconcile core/state.rs TaskCoordinator/StateRoot/PreparedDomain/CommittedOutput with core/session_root.rs SessionAuthority/RootBinding/RootAttempt into one logical StateRoot/attempt/epoch/provisional-domain/publication contract, keeping original physical completion, source identity and semantics.
- One publisher per TaskSession/shard. Source identity, expected previous root, epoch, frontier, domain schema and real ReadyWitness required. Device events/IO callbacks never publish logical root themselves.
- No new engine-only ResourceArbiter; existing process-shared low-level ledger stays below engine; introduce all-or-none bundle reservation only if actual multi-domain consumer/proof requires it, with one alias accounting and no duplicate host budgets.
- Preserve all physical readers encoded/captured/unsubmitted/in-flight and quarantine. Separate physical Ready, logical Commit, transport Delivery and client ACK/playback.
- Adapt CPU/Qwen/Gemma/GPU state callsites once to common contract; no permanent CPU-vs-GPU second logical root runtime. Preserve real source/driver-specific state capabilities and unsupported modes.
- Compile and run small test matrix for source/state authority and reservation/lifetime, without GPU device/model long suites.

One S3 signed commit or S3.a root, S3.b reservations if independent unit.

## 7. S0–S3 lightweight exit table

| Phase | Minimum check before commit | Deferred |
| --- | --- | --- |
| S0 | Cargo package/feature graph, fmt, architecture checker, new type/builder/refusal smoke, CPU-only cargo check | Full check.sh, hardware, trained models, performance |
| S1 | artifacts+CPU+GPU source-only checks and exact q4/q6/q8 malformed/golden fixtures | Full model/driver GPU and slow codec benchmark |
| S2 | family+GPU source/feature checks and representative semantics/source/ABI/capture fixture | Fresh actual CUDA full model, Gemma/27B, C16/C32 |
| S3 | CPU state/root/resource source-free targeted tests and affected crate compile | GPU fault injection, full root migration stress, long context |

Fix obvious failures before each commit, but one expensive integrated qualification only after F4. Do not waive source, state, memory or root safety merely because a structural test passes.

## 8. Transition halfway point

At the end of S0 architecture is VISIBLE; at end of S3 the lower boundaries are MOVED. Old legacy CLI serving and source business logic will be migrated in F0–F3, not retained as permanent competing runners.

Existing source correctness, acceptance identities, model-specific test results, original non-support and format guards must be preserved. A pre-v1 Rust type/name change is allowed; changing mathematical arithmetic, buffer lifetime or authorized capability without explicit new qualification is not.

After S3 proceed immediately to the model-neutral engine migration F0, public SDK F1, independent serving F2, thin CLI F3, present-feature coverage F4 and one consolidated Q0–Q2 integration gate. Full details in PHASED_IMPLEMENTATION.md.
