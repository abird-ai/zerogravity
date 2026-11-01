# Fabric vNext transition — current-source reconciliation

## Current source/host checkpoint — `ae7ec11`

- Foundation/lifecycle through `d78eb55` retained; shared engine Router/worker now
  owns CPU/GPU admission, scheduling, presentation and cohorts. Facade owns typed
  ordinary/captured adapters; serving consumes the same Router. CLI copies removed.
- Original IR, issuers, SessionAuthority, physical arbiter, Ready/commit, native
  feedback and final readers remain. New metadata/scratch ownership is admitted.
- Actual source retains fixed-prompt64-token wave credit and captured-Qwen
  72×max(native_rows,1); prompt and B4 eligibility remain source-owned. The older
  blanket67/72-over64 note was not an authority to narrow or widen these routes.
- CPU SDK/HTTP fixtures4 pass; host/feature/strict scopes pass. Final gate:
  2,306 Rust passes/0fail/94ignored/300 suite runs;
  17 Python suites/182 cases/10 skips.
- No CPU/GPU/model-crate math, kernel/image/manifest changes. Moved GPU execution
  remains physically unqualified at this source; no historical evidence retagging.
- Remaining SDK/CLI, active contracts and F4/Q acceptance stay in [TODO](TODO.md).
  The dated source observations/crosswalk below are historical, not restart orders.

## Historical S0 landing — `7396984`

S0 now exists in source: eleven workspace packages including engine/serving;
public facade contracts, truthful cold Unsupported, original owners preserved,
byte-identical CLI responsibility moves. DAG18/strict/focused tests and final
isolated gate pass2,159 Rust/0fail/87ignored/292 suites,17 Python suites/182 cases
plus packaging. S1 is next. No F0/F2 execution migration, physical qualification
or production activation from this structural milestone.

## 31 Oct 2026 implementation reconciliation

- Signed starting point: `1c35172`, clean main, following `e2559b6` native GDN.
  Root README remains untouched. External detached Kilo checkout is not ours.
- Latest instruction activates Fabric S0 before returning to Qwen. Plan 1 owns
  the complete architecture; Plan 2 adds concrete crate/SDK/server/CLI migration
  and S/F/Q ordering. Its corrected low-level ResourceArbiter placement wins.
- Pre-transition`1c35172`: nine workspace packages; engine/serving then absent.
  No GPU→model production dependency needs repeating. CLI already depends on
  the facade by Cargo name but still owns reusable runtime/HTTP/business logic.
- Keep actual `ForwardPlan`, `SourceInvocation`, `SourceOperatorTemplate` and
  typed source/state lowering. Do not add a parallel SemanticProgram IR or
  expand position-indexed captured catalogues into production execution.
- `2983bb7`: owned CUDA GEMM/RMS/rotary/attention; exact0.8B cached47→50,
  all248,320 logits/all42 domains; cancellation/quarantine and retirement.
  Thirteen capacity cells ran48 active tokens, not populated long contexts.
  2B32K refuses by8,980,841B; 2B numerical/reference authority remains open.
- `99f0232`/`825ade8`: source rank preservation and finite27B HIP natural/teacher
  full64/fullV/all112 domains plus known/fault/unknown owner qualification.
  This is not general27B execution or serving.
- `d4d5df4`/`f6fa4b9`: original CPU PV and shared CPU-free Qwen traversal;
  independent CPU/HIP attention exact1,774,240 words with physical retirement.
- `8a1c6b1`/`1f29a42`: genuine bounded0.8B/2B HIP,64 single-row positions,
  two natural replays, fullV/all24 layers/all state exact. `85f33c7` retains
  canonical SnapshotLease and qualifies healthy/fault/unknown final readers.
- `55536ce`/`e2559b6`: CPU and actual HIP padded64 chunk GDN each match
  19 cases/380 fields/64,345,520 words. Four HIP refusal controls export status
  only; all20 modules retire. Physical completed Writes may be initialized
  without math being exported by Ready. This boundary must survive S3.
- Original fresh47–50 and populated uniform/ragged512→514 captures exist;
  they do not qualify ZG multi-row models. Original cross-chunk states/logits
  differ; replay, chunk invariance, backend equality and quality remain distinct.
- Pre-transition accepted isolated gate:2,142 Rust/0fail/87ignored/283 suites,
  17 Python suites/178 cases plus packaging; shared HIP/Vulkan strict checks.
  No repeated baseline/capture/physical runs for unchanged paths during S0.

### Preserved unfinished mission

After structural/feature-preserving migration and applicable acceptance, return
first to genuine multi-row0.8B/2B HIP (projection/Conv4/attention/BF16 residuals,
chunk GDN and call-free lowering), then general27B and genuinely populated fit
contexts. Preserve l5 CUDA0.8B/2B qualification, x2 HIP0.8B/2B/27B,
C1/C4/C8/C16/C32/C64 isolation/cancel/shutdown, matched vLLM and measured
optimization. Keep losses/refusals/failed/unrun cells; do not replace this mission
with a generic future roadmap. Wider CUDA reference authority and an approved
ROCm vLLM runtime remain unresolved. Production routes remain disabled.

Recurrence`17aafa71` stays cancelled. S0 grants no model download, SDK replacement,
device privilege/reset, gfx90c launch, push or release. CPU/CUDA parity means
functional model coverage, not identical numerical contracts.

### S0 design choices against live source

- Keep current implementation homes when they already express the owner; the
  target tree is an ownership contract, not a demand for empty synonym modules.
- Core adds meaningful semantic/encoding/numerical/model/task/provider/placement
  surfaces; semantic refers to the existing ForwardPlan type. State and physical
  exports refer to original owners, without constructing a second publisher.
- Engine is a cold, model-neutral lifecycle/API shell until F0. No worker,
  admitted source, executable bundle, Ready or committed event is fabricated.
- SDK/serving pending operations return typed Unsupported. Current supported
  low-level SDK and CLI paths remain intact until their coherent migration.
- Separate scalar/quant encoding descriptions from numerical contracts and
  derived artifacts. Existing decoder mappings remain the sole codec authority
  during S0; S1 removes old names with callers, not via permanent compatibility.
- Existing CPU/GPU/model implementation modules satisfy their physical/family
  roles; S2 moves only remaining misplaced ownership/qualification support.
- Architecture checker extends the existing DAG policy: engine cannot reach a
  concrete provider/model, serving cannot reach facade/CLI/provider/model,
  CLI reaches project functionality through the facade only.

## Plan 1 → Plan 2 → current mission crosswalk

| Requirement retained from Plan 1 | Binding Plan 2 / implementation placement |
|---|---|
| One semantic program and prepared executable | Existing ForwardPlan + ExecutableBundle; no parallel IR/traversal; S0/S2/F0 |
| Five authorities, no duplicate runtime | Source/family, compiler, original session publisher, core physical arbiter, actual device worker; S0/S3/F0 |
| Closed-region legality | Required observations/effects; source/numeric/shape/ABI guards; committed-frontier state conversion or recompute; refuse opaque unrecoverable state; proven fallback or typed refusal; S2/F0 |
| Whole public SDK and independent server | S0 types; F1/F2 real load/run/pull/cache/config/pipeline/device/diagnostic/server APIs; CLI uses only facade at F3 |
| Embedded lifecycle/security | Explicit bind/start/address/shutdown/reconfiguration, caller-owned execution; redacted credentials; stage/retryability-aware failures; admin unexposed by default; F1/F2/Q |
| ModelAssembly ≠ TaskPipeline | Family components versus cross-model typed stages; metadata/validation ≠ execution; S0/F0/E08 |
| Task-aware rather than token-only scheduling | Text/encoder/acoustic/codebook/denoise/decision stages, output credits, shared admission; F0/E02/E08 |
| Realtime exactness | Received/decoded/consumed ingress; rational conversion residue; revision-bound provisional output; turn/output generations distinct from device epochs; Prepared/Committed/Delivered/ACK/Played separate; reconnect grants no neural root; S0 descriptors/E08 implementation |
| Safe adaptive specialization | Finite qualified candidates, switching at legal boundaries, hysteresis and truthful selected mode; no per-token compiler search; S0 contract/E02/E10 |
| Storage/numerics/derived quant separation | Existing exact codecs/source profiles; independent activation/layout/derived lineage; Q4_K operator refusal and future-codec holds; S0/S1/E06 |
| Physical placement and lifetime | Four placements + verified transfers, complete-domain prefix state, captured/in-flight source readers, actual Ready/quarantine, SSD/remote/NPU disabled until qualified; S3/E02/E04/E09 |
| Broad model/task roadmap | mHC and other new math require real source/reference; finite decision schemas require honest calibration/quality; E05/E08, not metadata support |
| Extension proof drills | Known-op family, novel stateful op, fusion disabled, new provider, non-token task, remote state bridge; measure shared-code change radius. Q2 certifies transition only; unimplemented drills remain E-series gates |
| Initial Qwen mission | Accepted CUDA/HIP/operator/lifetime scopes preserved; multi-row/general27B/populated contexts/C1..C64/matched vLLM remain first resumed capability work after migration |

Plan 2 corrects Plan 1's proposed upward arbiter move: the live physical ledger
stays in core. Ordinary CPU already uses SessionAuthority in current paths;
S3 reconciles the remaining reference coordinator rather than replacing CPU
with a new publisher. Root README stays byte-identical. Use typed refusal for
all pending operations; a private feature-gated todo is not safe merely because
it is private (all-features can select it). Existing Cargo checker enforces DAG;
review source/test separation without new placement-policy tests.

## S0 concrete source map

- Core: new identity/encoding/numerics/semantic/model/task/provider/physical/
  placement contracts; original accelerator::program, session_root/state,
  device events/queues and resource_arbiter remain authoritative implementations.
- Artifacts: artifact/sharded are containers; quant is the one current codec;
  source_weights/cache/acquisition retain actual owners. Derived lineage is
  metadata only. S1 moves implementations/callers, not duplicate decoders.
- CPU: existing ops/numerics/source/reference_text/runtime cover operators,
  selected arithmetic, references and physical CPU work. GPU: existing backend,
  cuda/hip/vulkan, quantization, resident_program/inference/leases and transfers
  retain kernels/ABI/ownership. No cosmetic kernel/image renames in S0.
- Families: existing config/source/forward/plan/reference and Gemma text/tensors/
  execution remain math owners. No runtime catalogue dispatch added to GPU.
- Engine: runtime/catalog/router/model_instance/prepare/executable/scheduler/
  session/task/state_cache/residency/placement/request/output/diagnostics compile;
  no loaded model, source, executable, event, worker or inference success issued.
- Serving: server/http/ingress/openai/responses/stream/auth/metrics/admin/realtime
  compile independently; cold loopback builder validates limits; start refuses
  without bind. Original CLI serving remains working until F2.
- Facade: runtime/models/artifacts/run/devices/pipelines/serving/diagnostics/
  builtins expose one public dependency. Existing low-level APIs retained during
  migration; new high-level engine operations refuse until adapters land.
- CLI: actual args/output/commands/signal code moved, not a second stub command
  tree. Existing model/HTTP/business owners remain temporary F0–F3 move targets.

Names reuse current responsibility homes rather than manufacturing empty synonym
directories. Skeleton compile visibility is not F4 feature migration acceptance.

The sections below retain the **27-Oct historical source inventory** for move
planning, not current execution/capability claims.

## 1. Historical 26–27 Oct baseline and commit lineage

The original file-tree/revalidation research inspected HEAD 195099ca583ace766d075682927f590360c823a5 with a then-large moving uncommitted checkout. The first vNext source refresh inspected eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe, **39 commits** after 195099ca, 1,220 files touched, 157,086 inserted and 103,429 deleted lines (includes moves, generated source and handoffs). Two later commits ba7fb03 and fadf2b1 bring the observed gap to **41 commits**. Earlier whole-repository history contained approximately 289 commits; do not mistake the last 41 for all project work.

| Commit | Concrete change | Fabric transition consequence |
| --- | --- | --- |
| 405a3e6 | Exact source codec and CPU/GPU quantization homes; BF16 precision separate | Retain format split, do not mint support from empty README |
| d9a8ecf | Name actual prefill/input/cache request geometry | Use shape/data, not fixture or ordinal names |
| fa844a6 | Family/model and backend ownership instructions | Enforce actual source/ABI/control boundaries |
| a5ed7de | Qwen/Gemma source/config/math moved into family owners | Do not redo already completed extraction |
| c90b8cd | GPU provider/native prepared bindings grouped | Reuse real physical GPU work, source evidence and lifecycle |
| fc73457 | Application model routes composed | Move orchestration up to model-neutral engine later |
| fe17625 | Four-tap GDN scalar math consolidated in CPU | Preserve tested arithmetic, no new duplicate code |
| 263ff79 | GPU→Qwen normal/build/optional model dependency removed; tests use dev dependency | Already satisfies the central Cargo boundary |
| 50abaf7 | HTTP worker join and shutdown ownership | Preserve in serving extraction |
| cd4159d / 5a18b5b | Native submitted commands and captured source leases retained | No premature free on cancellation/stop |
| 61f095c / d74f59a | Historical finite Qwen CUDA prompt and recipe fixtures | Diagnostics, not general production routing |
| 3612ecb | Production vs test/qualification code rules in AGENTS.md | Keep compiled production independent of fixture whitelist |
| 1f08e55, a1c51f9, 91a2958 | Fixed rotary preparation/capture separated and source identity guarded | Retain exact ABI and read-only source leases |
| 9ad9aac, d087012 | Parameterized CUDA GDN Conv and correct image/code identity | Reuse specialized kernel body and guard |
| 05baab5 | Prompt source semantics separated from qualification | Valid source geometry cannot be faked from testcase |
| 0c63d89, ae936a8, 7a910d9, ca0d2bb, f61a3ca | Qwen CUDA finite qualification separate, declarative implementation selection | Distinct source permission and backend kernel eligibility |
| 85b2724, 1fe6151, 2e3bbec | Prefill GDN stages parameterized | Preserve KKT/solve/recurrent/output variant contract |
| 1b898ff, c0bdbb1 | Sealed explicit kernel combinations, no recipe bridge in execution | Don't reintroduce test-case dispatch in production |
| eb38dd5 | Fresh-48 output variant | Not proof of general dynamic shape/model inference |
| ba7fb03 | Added fresh-prefill capture and generated source | Retain qualification source hashes/compiler/lease semantics |
| fadf2b1 | Fresh-prefill reference layout/oracle fixture corrections | Historical final 27-Oct design snapshot; real hardware/model gate remained separate |

## 2. Actual current Cargo workspace, dependencies, ownership

~~~text
crates/
  zerogravity-core/
  zerogravity-artifacts/
  zerogravity-cpu/
  zerogravity-model-qwen35/
  zerogravity-model-gemma4/
  zerogravity-gpu/
  zerogravity/              public facade but not yet complete high-level SDK
  zerogravity-cli/          owns HTTP/service work + run/acquisition/pipeline CLI
tools/gpu-qualify/
~~~

The proposed zerogravity-engine and zerogravity-serving crates DO NOT EXIST at the snapshot. The user explicitly wants them and all other target structural layers visible in code in the first architectural skeleton commit.

Normal/build dependencies:
- core imports no model or backend.
- artifacts -> core.
- CPU -> core+artifacts.
- GPU -> core+artifacts, optional CPU; Qwen source fixtures use dev-dependency only.
- Qwen model -> core+artifacts, optional CPU.
- Gemma model -> core, optional artifacts/CPU/GPU for specific preparation/qualification.
- facade -> core+artifacts, optional CPU/GPU/families.
- CLI -> facade plus CLI-specific crates. It still owns core application behaviors even though it only depends on facade by name.
scripts/check-architecture.py remains the single architecture-boundary checker and passed again during the 29-Oct review. Extend this checker in skeleton S0 for engine/serving/facade/CLI rather than creating a second policy checker. The earlier fadf2b1 pass is historical evidence only.

Historic approximate source counts at refresh: GPU ~590 Rust files/~310,274 LoC (~120,414 generated source); artifacts 57/~32,268; Qwen 95/~36,517; Gemma 24/~6,677; CPU 43/~11,774; core 43/~19,383; facade 27/~5,924; CLI 46/~26,407. Includes test/qualification sources. These are indicators of refactor ownership, not numerical/model capability.

## 3. Existing portions that should be reused

### Exact semantic model and source

- core/src/accelerator/program.rs contains real ForwardPlan, PlanOperation, typed SourceOperator, state effects, value/parameter references, explicit source/precision and MathematicalBackend. This is the foundation for future canonical SemanticProgram, not a new redundant IR.
- model-qwen35/src/forward.rs performs single family mathematical traversal; plan/ and source/ own source input geometry, config/tensor interpretation and variants.
- model-gemma4/src/{config,tensors,text,execution}/ contains source/math/PLE/attention/MoE and candidate source integration; no unrelated Qwen inference authorization.
- core/src/pipeline.rs contains structural typed pipelines/ports; pipeline run is explicitly NOT IMPLEMENTED.
- core/src/targets.rs contains metadata-only model targets; binding family slug does not certify model execution.

### Source formats, scalar and physical computation

- core/src/quant.rs contains the old QuantKind representation with exact GGML GGUF codes and byte length: F32/F16/BF16, Q4_0, Q4_1, Q4_K, Q6_K, Q8_0. This broad enum mixes scalar and quantized storage identity and should be refined into two typed variants without weakening errors.
- artifacts/src/quant.rs and quant/q4.rs, quant/q6.rs, quant/q8.rs perform actual scalar decoding (including Q4_K); Q1/Q2/Q3/NVFP4 directories are only README placeholders, not Rust bodies. BF16 currently has real scalar precision conversion in shared source, but new precision/bf16 folders mainly house documentation.
- CPU source source.rs/projection.rs owns true source-specific activation quantization and GGML reduction (Q4_0 x Q8_0, Q4_1 x Q8_1, Q6_K x Q8_K, Q8_0 x Q8_0). Q4_K full source matvec explicitly refuses because the required original GGUF Q4_K x Q8_K physical dot has not passed independent qualification.
- GPU quantization.rs and quantization/{activation,activation_fused,projection,batch_projection,q4_0,q4_1,q6_k,q8_0} plus backend CUDA/HIP/Vulkan use actual packed kernels and physical queues/ABI, separately qualified by backend. No general Q1/Q2/Q3/NVFP4 computation granted.
- New CUDA GDN includes explicit TextGdnImplementation and GdnKernelConfig for KKT/Recompute/H/Output, source-bound geometries and retained code-image hashes; actual source/ABI/shape must be proven before general source promotion.

### State/resources and real completion

- core/resource_arbiter.rs: one process-shared physical ResourceArbiter, GlobalResourceLease, PhysicalDeviceKey, allocation classes. This is the existing low-level owner and stays below the engine.
- core/state.rs: reference TaskCoordinator/StateRoot/PreparedDomain/CommittedOutput; core/session_root.rs: SessionAuthority/RootAttempt/RootBinding. They need one logical root/epoch/publication protocol, preserving original test and real physical safety.
- GPU resident_program.rs, resident_inference.rs, resident_leases.rs, physical_service.rs, physical_queue.rs have accepted physical work, one selected StateRoot publication, ready witnesses, image/code ownership, unknown/cancel quarantine and finite multi-row programs. Do not discard in favor of a hypothetical executor.
- Device source and activation data may remain GPU-resident over legal stages; source reader leases must be held from encoding/capture through actual final completion.

### High-level still needing extraction

- zerogravity/src/lib.rs is mainly module/backend reexports with some model-specific Qwen CUDA adapters under qwen35_cuda.rs. It is not yet a complete public Runtime/Server/ModelStore API.
- CLI/serve/{mod.rs,engine.rs,gpu_engine.rs,gpu_batch.rs,pending.rs,http.rs} implements CPU/GPU HTTP routes, Qwen-only model/tokenizer fields, bounded queues/workers, real B2/B4/B8 native request cohorts, JSON/SSE/output credits and graceful worker shutdown.
- CLI/pull.rs + pull_azure/s3/gcs/object/modelscope/xet/oci etc. implements high-level provider selection and UX despite many transport primitives already living in artifacts/acquisition. Make library programmatically useful without changing source rights.
- CLI/pipeline.rs does bounded YAML v1/v2 parse and source-neutral graph check, but actual pipeline run refuses. Move library logic up without implying native execution.
- CLI/run_cli/source, execute, devices, config, qwen35_cuda contain model source/fit/selection and preflight business behavior; move reusable policy to facade/engine while retaining argument/env/printing in CLI.
- The serving API is a qualified subset only, not full OpenAI Responses/embedding/tool/multimodal support.

## 4. Historical outcomes and explicitly open claims

The current plan records include prior check.sh CPU/library/CUDA source/test passes and real finite GPU runs with specific model/driver/certificates, e.g. 263ff79 full 1,914 Rust tests with 87 explicit ignored cases and no new physical model run. That does not qualify fresh latest code or all models. Other plans contain current scoped B2/B4/B8 GPU API cases, historical C1/C4/C8 wins and regressions, and long-context gaps. They remain grounded historical evidence and must not be relabeled "all performance done."

Current truthful gaps:
- Q1/Q2/Q3/NVFP4: document homes, no real codec/CPU/GPU implementations.
- Q4_K: scalar decoder exists; source-exact CPU matvec explicitly Unsupported; no general trained model execution permit follows.
- Original HF CUDA: finite captured source diagnostic permits only; ordinary dynamic Qwen CUDA explicitly rejects unsupported GDN/source shapes. Fresh48 kernel source/capture/oracle updates are insufficient to promote.
- Qwen27B: sharded/source windows and finite CPU/HIP/qualification; not arbitrary trained whole-model, SSD optimized serving.
- Gemma4: family source, E2B etc. reference and finite HIP/Vulkan candidates; not generalized all model variants/tasks.
- Paged complete-domain KV/GDN/MLA prefix state, generic production chunked temporal prefill, unrestricted vLLM continuous goodput parity: not finished.
- Native image/video/audio/ASR/TTS/TTS-codec and peer-distributed inference: design/scaffolding only or separately bounded provider experiments.
- CPU GGUF serving and selected Qwen CUDA fused HTTP routes: valuable existing qualified subsets, preserve and revalidate.
- Source/parser/device/data/HTTP refusal and physical lifecycle tests: no waiver during transition.

## 5. Historical old-to-new owner crosswalk

| Current source | Final owner/responsibility | Must preserve |
| --- | --- | --- |
| core/accelerator/program.rs | core/semantic/program and typed contracts | one plan, exact arithmetic/effects |
| core/quant.rs | stored encoding and typed scalar/quant schemes | byte length, GGML IDs, unknown refusal |
| artifacts/quant.rs & q4/q6/q8 | artifacts/codecs + source encoding | exact scalar decoder and malicious input guards |
| CPU source.rs/projection | CPU source-exact numerical operators | activation quant/reduction and reference |
| GPU quantization/ | GPU packed kernels and ABI | physical/kernel tests, per-backend qualifiers |
| Qwen source/plan/forward | family semantics and qualified recipes | original trained math/source maps |
| Gemma config/tensors/execution | family semantic assembly/candidate | true PLE/MoE/softcap/attention |
| GPU CUDA GDN/text and generated cases | reusable CUDA implementations + qualification-tool oracle | actual code/ABI/testcase provenance |
| core/resource_arbiter | shared low-level physical ownership | single process ledger, aliasing, real leases |
| core/state and session_root | canonical state protocol + optional engine coordination | one publisher per session, full domain version |
| GPU resident_program/inference/leases | device executor/physical service, compiled bundles | actual Ready/cancel/quarantine |
| facade prepare/qwen35_cuda/model | application model registry and public runtime adapters | no false production CUDA permit |
| CLI serve engine/gpu_engine/pending | engine model-neutral scheduling/admission; ingress-specific part serving | independent sessions, fairness, native cohorts |
| CLI serve http/protocols | separate zerogravity-serving | OpenAI JSON/SSE, error mapping, shutdown |
| CLI run/pull/cache/devices/pipeline | public library/engine/artifacts and truly thin CLI | same functionality and deliberate refusals |

## 6. Boundaries for successor agents

The source is actively changing and the latest observed model/GPU evidence can shift between turns. At future code start:
- Read HEAD/status/active checkouts, all .agents/plans owner TODO updates and AGENTS.md.
- Preserve signed commit history, retained source hashes, source model/driver/kernel code IDs and explicit negative tests.
- Don't rerun already completed low-level dependency extraction just to match a new filename; move responsibility only where still genuinely misplaced.
- Never execute held gfx90c GPU or unsafe system power tuning; model downloads and remote peers need separate authorization.
- No source code changes or Git commits are part of writing this plan. The structural-code phases require a later explicit implementation instruction.
