# ZeroGravity — full comprehensive agent handoff (25 October 2026, night)

**Purpose:** One source-grounded takeover entrypoint spanning the initial Zerogravity design, implementation phases, user decisions, tests/negative results, current hardware/backend boundaries, and next work. This handoff is accompanied by two exhaustive, repository-local inventories: **[all 245 existing reachable commits](HANDOFF_ALL_COMMIT_LINEAGE_2026-10-25.md)** and **[all 360 non-completed plan items (351 open, 8 active, 1 blocked)](HANDOFF_ALL_OPEN_PLAN_ITEMS_2026-10-25.md)**. These inventories are snapshots for coverage; the live Git tree and plan TODOs always win. Earlier large handoffs, canonical sourcebooks, specs and original decision tables are retained, not replaced.

| Item | Grounded state when this handoff began |
|---|---|
| Repository | `/home/pvl/spaces/abird/src/zerogravity`, via connected `dotlink---l5` |
| Branch | `main`, HEAD `d84853337bbd2830f50a9c4e704d266990fd2ce4` |
| Working state | Clean tree; 36 commits ahead of `origin/main` at inspection; no push |
| Remote | `https://github.com/abird-ai/zerogravity` (not permission to push) |
| Language/build | Rust 1.95, Cargo workspace 0.1.0, MIT, Nix/crane scaffolding |
| Protected README | User-owned `README.md`; SHA256 `6c00873ec0ac2c59bc73703b55a9bfa4d93703334a03fa16bcac2d204cda5b15`; **never edit** |
| Source truth | Live source + original physical/output receipts > subsystem TODOs > normative design > historical handoffs |
| Binding immediate order | **3: B4 complete original physical/model certification → 4: fresh matched .8B/2B C16 → 2: Qwen3.8-27B x2 original capture** |
| What this handoff did | Documentation/index/source review only: no code change, model download, GPU inference, full build, x2 access, push or release |

**Important timestamp caveat:** Some imported historical Git records bear dates *later than this 25 October 2026 review*. Source ancestry/commit SHA is reliable for reconstruction; do **not** use those dates as evidence that work ran in the future. The companion commit index is explicitly ordered oldest-to-newest by Git ancestry without relying on dates.

## 0. Immediate takeover: how to get fully oriented

1. Enter the repository and independently inspect `git status --short --branch`, `git rev-parse HEAD`, `git log -8 --oneline`, and `git diff --check`. Never presume this document's snapshot is still HEAD.
2. Read [`AGENTS.md`](../../AGENTS.md) **in full** (safety, signed commits, user-owned README, source permissions, hardware holds, coordination). Then [plan rules](../plans/README.md) and [current queue](../plans/TODO.md).
3. Read this handoff completely. Review the [all-commit history](HANDOFF_ALL_COMMIT_LINEAGE_2026-10-25.md) and [all unchecked tasks](HANDOFF_ALL_OPEN_PLAN_ITEMS_2026-10-25.md), then read the **actual source TODO** of your chosen subsystem. Those source TODOs preserve multiline dependencies and failures; do not edit the snapshot index to mark work done.
4. For original architecture read [master architecture](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md), [whitepaper](../../docs/WHITEPAPER.md), [engineering specification ES-00…15](../../docs/ENGINEERING_SPECIFICATION.md), and [shared execution design](../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md).
5. For decisions from the beginning read the **archived** [explicit-user vs proposed decision register](../plans/archive/history/DECISION_REGISTER.md), [requirements traceability](../plans/archive/history/REQUIREMENTS_AND_TEST_TRACEABILITY.md), [P00–P23 agent playbook](../plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md), [roadmap](../plans/IMPLEMENTATION_ROADMAP.md), [release gates](../plans/RELEASE_ACCEPTANCE_GATES.md), and [experiments](../plans/EXPERIMENT_PROGRAM.md).
6. For latest changes read [B4/C16 continuation](B4_C16_EXECUTION_CONTINUATION_2026-10-25.md), [adversarial correctness review](ARCHITECTURE_CORRECTNESS_REVIEW_2026-10-25.md), [post-4bf source audit](POST_4BF3673_AUDIT_AND_RESUME_2026-10-25.md), [ABI/family/CLI refactor](ABI_FAMILY_CLI_REFACTOR_2026-10-25.md), and [Linux /dev device nodes](LINUX_DEV_NODE_SELECTORS_2026-10-25.md).
7. Bound searches to literal source/run paths. `.agents/runs` contained approximately **433 GiB** at audit; never globally recurse or grep through it. Only named old runs/source pins support old claims. Keep *every* failing or interrupted run.
8. Confirm physical ownership, exact model permissions, source/artifact hashes, runtime SDK and authorization **before** any model/gpu/remote side effect. If blocked, progress independently ready safe work and report exact missing prerequisites without inventing results.

## 1. Original vision and immutable design intent

**User-visible positioning:** “Run any model locally with the single binary Rust execution fabric for **text, image, video, audio, real-time, and combined multimodal inference**.” This is the stated product goal, **not a current universal executable-model guarantee**. Model-specific mathematics, kernels, memory placement, scheduling and serving compose under **one engine**, rather than a new runtime per model or modality. The project name is lowercase **zerogravity**; MIT open source. The user favors human-led architecture/design with AI-assisted implementation; the README has explicit user-owned wording and must not be modified.

**Initial requirements and evolving scope:**

- **Source-exact model truth first:** identifiable checkpoints, index/layout, quantized weights, tokenizer/chat template, operator mathematics including exact casts/reductions/FMA, session state, sampling and reference outputs. A model name, downloaded file, metadata parser or Cargo feature is **not** proof of runnable inference.
- **Portable CPU reference + specialized GPU/accelerator implementations:** Rust-owned semantic/StateRoot engine with fast candidate regions on NVIDIA CUDA, AMD HIP/ROCm, Vulkan AMD/NVIDIA and future NPU/ASIC without splintering source execution. Vendor SDKs and audited native kernel code may be used *inside* owned physical providers; no separate hidden C++/Python model engine.
- **Real physical ownership and heterogeneous backing:** GPU/RAM/SSD unified tensor identity, source/index page windows, typed liveness, GPU memory budgets, actual event/reader leases, source-compatible cache eviction/prefetch, paged KV plus recurrent state and branch prefix copy-on-write; never double-count GTT/UMA or release unknown in-flight physical memory.
- **vLLM/SGLang-class serving ambition:** same Rust executor powers CLI and local multi-model OpenAI-compatible API; resident model router, simultaneous sessions, fair token-bounded continuous batching, native fused B1/B2/B4/B8 compatible waves, chunked prefill, prefix reuse, prefill/decode priority, exact output/HTTP SSE, target-authoritative MTP/drafter/n-gram/suffix speculative verification and rollback. Optimize actual complete-request goodput/p95, not nominal kernel launch reduction.
- **One-process composable model pipelines:** YAML typed model/stage graph, separate source/tokenizer/processor/codec, ASR/diarization→LLM/decision/filter/tools→TTS; real-time audiovisual clocks; image diffusion/video pipelines, source-bound iterated tasks. Also support separately owned processes as a deployment choice.
- **Multi-device/machine future:** independently place computation, model parameters, mutable state, immutable prefix/expert caches and task stages on local/remote authenticated peers over approved tunnels, with actual network/IO costs and one valid logical state publication. No cloud account/server/network required for ordinary local use.
- **Simple UX:** one binary, `zerogravity run MODEL`, `serve`, `devices list`, `-d` for physical choice, `--dry-run`/`--explain`, XDG config and optional advanced `--backend`. No hidden source download or unqualified backend/precision/model substitution. Rust/Nix/crane builds, cross-release, installer and release workflow are tracked separately from platform proof.
- **Model order initially:** Qwen3.5 0.8B/2B CPU→GPU, Qwen3.8 27B, Gemma4 E2B/26B-A4B and other Gemma4 variants, later Qwen Flash-Next ~125B, GLM Flash, DeepSeek and additional dense/MoE, audio/image/video/decision families. Later explicit user **execution priority** is the B4/C16/x2 order in section 8, not a replacement of the long-range roadmap.
- **Acceptance:** per exact model/revision/quant/shape/numerics/context/sampler/backend/driver/hardware and workload; include full-logit/StateRoot, real failure/cancellation/retention, cold and warm complete-request throughput, tail, memory and energy, fair matched stock baseline and all negative experiments. No invented benchmark percentage or generic source-capability equivalence.

## 2. Original design hierarchy, supersessions, and surviving historical sources

**Governing principle:** *A model describes mathematics; a task describes control; a provider supplies an implementation; a plan schedules computation and physical resources; a coordinator owns visible state.* There must be one semantic engine, one physical resource authority per real domain, and one session StateRoot publisher.

| Complete source | Content and why to read |
|---|---|
| [Master architecture v4.3](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) | Original vision, six composable concerns, v4.1 accelerator granularity, v4.2 distributed placement/tunnels, v4.3 one-command router/serving/UX. |
| [Whitepaper](../../docs/WHITEPAPER.md), [engineering spec](../../docs/ENGINEERING_SPECIFICATION.md), [bibliography](../../docs/REFERENCES.md), [publication checklist](../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md) | Externally coherent **proposal**, normative requirements ES-00 through ES-15, acceptance/limitations/references. A manuscript is not a benchmark claim. |
| [Decision register](../plans/archive/history/DECISION_REGISTER.md), [traceability](../plans/archive/history/REQUIREMENTS_AND_TEST_TRACEABILITY.md), [agent playbook P00–P23](../plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md), [roadmap](../plans/IMPLEMENTATION_ROADMAP.md) | Original U-series user-approved versus merely recommended versus open choices, supersessions, phase numbering and test prerequisites. This is the *real* surviving location. |
| [Modular reference architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [minimal execution spine v4](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), [composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [TaskProgram/IR refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) | Later minimality decisions supersede incompatible initial multi-engine/multi-authority drafts; regional compiler views are transient, not new public runtimes. |
| [Semantic IR](SEMANTIC_IR_AND_COMPILER.md), [async resource/command IR](ASYNC_RESOURCE_COMMAND_IR.md), [memory residency](MEMORY_RESIDENCY_AND_IO.md), [state transactions](STATE_CACHE_AND_TRANSACTIONS.md), [fail-safe invariants](INVARIANTS_AND_FAILURE_SEMANTICS.md), [native kernels](DEVICE_KERNEL_AND_CODEGEN.md) | Exact contracts for typed effects, source/producer identity, physical pages/readers, event/unknown handling and state publication. |
| [v5.1 multimodal](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md), [real-time](REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md), [model assemblies](VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md), [SSD backing](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) | Latest broad proposed family/media/pipeline/source/streaming semantics, not automatic implemented support. |
| [Research prior art](../prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md), [architecture critical review](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md), [final architecture review](../prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md), [prior-art/license rules](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md) | Why the architecture declined duplicate engines, state publishers, per-kernel global admission, unverified source math or uncited vendor source reuse. |

**Historical link relocation:** old `docs/README.md` and earlier handoffs refer to `.agents/plans/DECISION_REGISTER.md`, `AGENT_IMPLEMENTATION_PLAYBOOK.md` and `REQUIREMENTS_AND_TEST_TRACEABILITY.md` at nonexistent former roots. Actual files survive under **`.agents/plans/archive/history/`**. `.agents/plans/TODO-SPEC.md` also no longer exists; use current `AGENTS.md`, `.agents/plans/README.md` and the archived playbook instead. Do not invent files or treat broken old references as lost design decisions.

**Complete lossless spec/plan/research manifest:** [all existing source files, hashes and headings](HANDOFF_ALL_DESIGN_AND_SPEC_SOURCES_2026-10-25.md). This includes every present `.agents/docs`, `.agents/plans`, `.agents/research`, `.agents/prior-art` and top-level `docs` Markdown document at this checkpoint, including sourcebooks not named below.

**Lossless documentation history:** [17 Oct handoff](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md) (~377 KiB), [19 Oct](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md) (~603 KiB), [20 Oct](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-20.md) (~1.37 MiB) and the archived [20 Oct recovered sourcebook](handoff-2026-10-20/HISTORY_DESIGN_AND_COVERAGE_AUDIT.md). These retain the original U-series decision/approved/proposal categories, negative experiments, all previous assistant context and generated plans; don't collapse U-001 onward into only recent priorities. [25 Oct morning](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25.md) and [afternoon](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md) preserve original H/G/V and B4/C16/x2 worker context but describe **earlier source**. [Post-integration audit](POST_4BF3673_AUDIT_AND_RESUME_2026-10-25.md) and recent code reviews override their superseded bug findings; actual failures remain preserved.

## 3. Actual architecture / ownership (one execution fabric)

~~~text
immutable source + verified model family + source numerical schedule
                              |
                typed semantic ForwardPlan
         mathematical operations / versioned state effects
                              |
     bounded prepared physical implementations / regions
   source, ABI, profile, geometry, available HW, compiler identity
                              |
    one global-per-process physical ResourceArbiter domain
       actual provider subleases / fair queues / IO owners
                              |
         actual submitted GPU/CPU/IO work and events
       completion/readback or retained unknown/quarantine
                              |
       exactly one SessionAuthority/TaskCoordinator
          publishes one current logical StateRoot
                              |
    committed CLI / HTTP / stream / model pipeline output
~~~

- `SourceIdentity`, versioned trained weights/index/quant/tokenizer, `ProgramNumericalProfile`, prepared executable identity, and real physical `PhysicalDeviceKey` are **different types of truth**; none can stand in for another.
- `StateDomain` only prepares data, `Device/CIR` only returns actual Ready/Failed, one coordinator/session authority checks source+attempt+epoch+frontier and publishes. Output after cancellation/Unknown cannot guess a commit.
- `ResourceArbiter::process_shared()` collapses multiple CUDA/HIP/Vulkan views of **one PCI card** within a process. It is **not** a cross-process/fleet scheduler. Real allocated backing, envelopes, generations, captured/encoded readers, retained modules and worker credits remain held until device completion or terminal quarantine. “VRAM max”/GTT/UMA is not “free/owned VRAM.”
- `ForwardPlan` is the common mathematical representation. Finite source/ABI/numerics-specific kernels may cover an op, layer or whole legal region. No hot per-token family catalog or general-purpose graph optimization requirement. If a source shape/numerical profile differs, prepared implementation must refuse; a different qualified candidate is legal *before submission*, never silently after an unknown GPU fault.
- Typed task clock/frontier and staged dataflow allow speech, images and video without pretending a frame is a token; separate application authorization governs tools/network/files. Same-device compatible stage values can retain leased GPU memory and event dependencies without host copies; cross-device/processor/codec paths use explicit bridges.
- CPU direct run and OpenAI-compatible serving use the same source-defined semantic engine and original session owner; no second router state authority or model-specific executor is allowed by the design.

**Actual code map:** `crates/zerogravity-core` owns `accelerator/program.rs`, `session_root.rs`, `source.rs`, `state.rs`, `resource_arbiter.rs`, `device_queue.rs`, `device_events.rs`, `run_policy.rs` and `pipeline.rs`. `crates/zerogravity-artifacts` owns verified immutable cache/source; `-cpu` portable math; `-model-qwen35` and `-model-gemma4` family semantics; `-gpu` native CUDA/HIP/Vulkan physical service, leases, source/program/typed transport and quarantine; `crates/zerogravity` facade; `crates/zerogravity-cli` owns direct `run` and `serve`; `tools/gpu-qualify` and `tools/` independent reference/quality probes. `nix/`, `flake.nix`, `install.{sh,ps1}` and `.github/workflows/build-binaries.yml` cover scaffolding, **not universally verified released binaries**.

**Recent source refactors:** `83a3174` made one stack-fixed (max 30 args) B4 CUDA ABI table derive physical handle order, byte access/extents, entrypoint and cold metadata; CUDA native source signatures have a separate test. `095bf8a` moved closed Qwen3.5 geometry/BF16 recipes into `crates/zerogravity-core/src/accelerator/program/family_qwen35/` **while preserving original Rust canonical type identities**. `7362699` fixed cancellation hint and decode context span; `a39ca96` fixed independent-source GPU oracle. The implementation did not add a second runtime or change CUDA kernel math.

## 4. Full implementation chronology and key decisions (Git ancestry, not inferred from dates)

The [all-245-commit lineage](HANDOFF_ALL_COMMIT_LINEAGE_2026-10-25.md) records every reachable source step with SHA and subject, oldest first. Consult the actual Git diff and evidence rather than asserting all such changes physically passed.

### 4.1 Original product/research → bootstrap

- **10–12 October design history** (older documentation/proposals): modular multi-model inference, hardware-specialized math and kernels, one typed semantic program, transient region/codegen views, physical memory/root transactions, SSD/offload, tokenizer/cache/source identity, SGLang/vLLM-inspired scheduling, router/API/CLI, multimodal future, peers/tunnels; research compared llama.cpp/Ollama/vLLM, accelerator/quant/SSD/graph/Gufo/NPU and distributed systems. [Master architecture](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md), [research index](../prior-art/RESEARCH_SOURCE_ATLAS.md) and [original design register](../plans/archive/history/DECISION_REGISTER.md) retain the details and superseded options. Bootstrap first: Rust 1.95 single binary, Nix/crane, lint/test/install/cross workflow. Initially **not an inference runtime**.
- **`be95424`** established design/prior art in Git. **`1e10b5c`** introduced immutable source identity, session StateRoot and resource ownership. **`af9a5d3` / `b276fd0`** introduced sharded source artifacts and transactional cache; source manifest/checkpoint/rights are separate from execution.
- **`782e6fc` / `1fd12da`** built CPU math and source-verified Qwen3.5 forward/tokenizer; source correctness and independent CPU reference improved through qualification. **`e6ecf02`** began Gemma4 family/source contracts, later expanded to E2B/26B and subtypes. Historical CPU qualification reported **8K functional context**, with longer 16K/32K tests explicitly deferred (not passed). Residual GDN numeric accuracy, independent teacher-force/f32 mismatch remain separate plan work.

### 4.2 GPU residency, physical safety and exact mathematical islands

- Vendored audited CubeCL 0.11 components and extensions for native CUDA/HIP/WGPU/IR/CPP; later `7d768a0` / `a9723bc` introduced GPU resident source and facade/semantic/state foundations. This is a single Rust-owned engine with vendor physical backends, not multiple independent inference engines.
- P07-G1/G2 started source-neutral card admission, PCI identity/alias control, fair workers, resident leases, readback and fail-closed post-submission abandonment. Physical CUDA/HIP/Vulkan identity is from driver/OS BDF/LUID/registry, not adapter name or virtual memory total.
- Real local NVIDIA RTX3060 and AMD Renoir/RADV Qwen GGUF partial source tests distinguished correct computation from complete model execution. The original AMD RADV Q4_0×Q8_0 `NativeF32` kernel did **not** match source bits; an explicitly specialized `WideF64` implementation did. Historical two-card partial stage StateRoots/vocabulary and physical aborted-Future quarantine passed under their exact source, without proving native driver reset/recovery or general physical GPU serving.
- H1/H2 physical milestones established **187 projections + 49 norm operations/token = 236 source-bit-equivalent GPU operations** for specified Qwen3.5 .8B/2B partial source scope. H3/H4/H5/H6 extended grouped head norms, GDN causal convolution/recurrent, attention, RoPE/QK/state and fault probes. Second-token F32 multiply/source-reduction rounding differences required **separate correctly rounded kernels**, never a relaxed numerical tolerance. Historic H3/H4 results include many region completions/StateRoot parity, but cannot be reinterpreted as universal all-model GPU inference.
- Historical U-036…U-047 decisions and exact physical receipts live in [archived decision register](../plans/archive/history/DECISION_REGISTER.md), [17 Oct handoff](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md) and [accepted-baseline history](../plans/archive/history/accepted-baseline/TODO.md) if that path exists; the 17/19/20 Oct archives contain recovered copies if older plan entries have since moved. **Local AMD Renoir/gfx90c launch/reset is now held** after a later abort: do not reattempt an unrelated physical probe simply because old RADV pass exists.

### 4.3 Composition, acquisition/SSD, serving and learned-model families

- The Rust facade and shared typed `ForwardPlan` recorded model/operator mathematics and task/pipeline scaffolds; `zerogravity.pipeline.v1/v2` YAML and one-process multi-model pipeline controls were designed/partially implemented. Audio/sample/frame/video/decision model support remains behind separate source, codec, processor and physical gates.
- Immutable source/cache/acquisition extended bounded local HF/GitHub and optional OCI/Azure/GCS/S3/ModelScope/Xet providers with protected staged transactions and source pins. This does **not** grant license to use a checkpoint, remote unapproved download, quant/format support or SSD-trained inference.
- Initial OpenAI-compatible HTTP/1.1 `serve` included bounded ingress, local auth, `/health`, `/v1/models`, chat/text completions, SSE/events, output/cancel/backpressure and honest unsupported-field refusal; CPU session/state and finite GPU queues were kept inside one engine. Model router, current config and reservation policy must not become independent state authorities.
- Native CUDA short-HF source paths grew from isolated N1 into certified finite B1/B2/B4/B8 source-compatible operations, source-root current frontier and B4 typed state transport; C1/C4/C8/C16 testing was requested **against matched stock vLLM**. File/spec dependencies are tracked in the active [serving](../plans/serving-and-speculation/TODO.md), [numerics](../plans/numerics-and-context/TODO.md) and [optimization](../plans/cpu-gpu-optimization/TODO.md) plans.
- Parallel model lanes introduced Qwen3.8-27B source-role/reference tools, Gemma4 E2B/26B learned source, H/G/V HIP+Vulkan specializations and source page/event ownership. **Qualified partial operations and source reports do not prove a complete large-model API.** Keep Qwen3.8 reference capture and Gemma4 feedback/PLE independent from current small-model benchmark gate.

### 4.4 Late October integration and subsequent corrections

Selected dated integration ancestry (consult every intervening SHA in the full commit index):
- **`4bf3673`**: integrated CLI, compute and composition plan checkpoint.
- **`8f01635`**: shared portable admission/thread control; **`5d40cf5`**: common pending queue and ingress credit; **`52f76a0`**: tokenizer and 8GiB HOST bound; **`efd7a34`**: CUDA resource census. **`19ff059`**: Vulkan native catalog; **`29a98c1`**: HIP source-page event probe; **`e8db111`**: Qwen27B matrix/reference tools; **`332038e`**: Gemma HIP owner tools.
- **`2304f8b`**: CLI catalog/schema precedence, explicit device/source selection, dry-run resource explanation and source-bound pre-Client admission. **`d3d21a3`**: B4 grouped GPU transport/certification scaffolds. The [25 Oct afternoon handoff](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md) describes this older integration, including real H/G/V passes and failed fixture/controller cases.
- **`488e0ea` / `93b611a` / `9e67b74`**: repaired feature isolation and H/G/V qualification fixture/report correctness, independently recorded in [post-integration audit](POST_4BF3673_AUDIT_AND_RESUME_2026-10-25.md). Do not rerun superseded “open bug” instructions from the afternoon handoff without reading this audit.
- **`03f2e66`**: repaired grouped KV ABI controls to source-exact **row-major `base0,end0,block0,...`** (22 import/30 export bindings), packed BF16 fixture, exact output/write census and complete final logical/device retirement. On **that specific source**, actual RTX3060 **18-case grouped model-free physical PASS** and separate **301 finite model-free cases PASS**; a 4344-record original CPU preflight **timed out**, and model-dependent B4 cases/optimized current C16 remained unqualified. [B4 continuation evidence](B4_C16_EXECUTION_CONTINUATION_2026-10-25.md).
- **`a39ca96`**: found a critical oracle flaw: grouped kernels were running on **already-updated per-row buffers**, so missing writes could falsely pass. New code uses *separately initialized/retired* host-exact per-row and grouped flights. Also requires identical shared native KV cache allocation/view/stream across four rows and removes duplicate readback copies. **Old physical pass does not certify this stronger new code**.
- **`7362699`**: fixes cancellation race (reserve→cancel→empty sweep→enqueue lost hint; boolean retry) and capacity overflow (decode frontier 512+1 accepted). Scoped pending20/20, batch14/14, HTTP8/8 PASS; no broad real-GPU proof. **`5229e53`** documents adversarial source/architecture review.
- **`83a3174`**: one stack-fixed typed CUDA ABI (source/native entry, pointer slot, byte extent, readable/writable mode) feeds prepared metadata **and** actual bound handles; native CUDA signature separately checked against source, wrong kernel variant refused. **`095bf8a`**: source files for Qwen3.5 .8B/2B closed `full_model_profile`/`short_hf`/`vllm_model`/`vllm_native` moved into `program/family_qwen35/` while **original Rust type identity paths** deliberately remain unchanged; one shared `ForwardPlan` not replaced. **`2a47d25`/`d9c5872`**: concise CLI/user-facing memory observation and source/evidence design report.
- **`5399ab9`**: removed redundant `--on`: normal device choice uses **`-d`**, optional advanced **`--backend`** only to force CUDA/ROCm/Vulkan when the same device supports alternatives. Do not reintroduce `--on`.
- **Current `d848533`**: Linux exact physical `-d /dev/dri/renderD128` / `/dev/dri/cardN` / `/dev/nvidiaN` / supported `/dev/dri/by-path` symlinks. Read-only `stat`/sysfs/driver-proc resolve a real character node to the **existing full PCI BDF observation**, preserving original exclusions/holds/vendor/availability. Shared `/dev/kfd`, `nvidiactl`, `nvidia-uvm` are **not valid per-GPU selectors**, but are **not prohibited runtime interfaces**. Vendor ROCm/CUDA may need them to actually compute. No universal pre-context permission/prerequisite checker or auto container mounts currently exist. [Full device decision](LINUX_DEV_NODE_SELECTORS_2026-10-25.md).

**Performance caveat:** an earlier frozen source21 2B C16 matched attempt recorded ~78.7859 tokens/s ZeroGravity versus ~95.9974 stock vLLM (a loss), 16 offers/128 output tokens. An .8B diagnostic had an extra sampler and not a quiet matched win. Other historical C1/C4 numbers are source-scoped. None qualifies **new current HEAD**. There is no justified fresh C16 advantage until section 8 gates complete.

## 5. State of each subsystem at handoff

| Subsystem | What exists now | What is still *not* established |
|---|---|---|
| Core identity/state | Immutable source, `SessionAuthority`, typed `ForwardPlan`, condition-checked logical StateRoot, physical resource arbiter/queued work and retained event state. 208/208 recent core tests PASS. | Independently qualified all-family semantics, general distributed state writer, automatically safe physical driver recovery. |
| Artifacts and acquisition | SHA/manifest/source graph, protected cache snapshots and provider adapters with scoped tests. | Every provider/layout/rights case, broad SSD trained-source conversion, arbitrary format/model name execution. |
| Qwen CPU | Qwen3.5 .8B/2B GGUF paths, tokenizer, bounded reference/physical math and source-specific BF16 tools. | Full 8K numerical long-prefix parity in every independent reference profile, 16K/32K gates, all Qwen3.8 learned end-to-end. |
| Gemma model | Gemma4 E2B/26B source/typed operators, PLE, source-page and HIP/Vulkan partial evidence/fixtures. | Complete trained 26B MoE routed inference or every E4B/12B/31B source backend with natural outputs. |
| GPU service | CubeCL-patched CUDA/HIP/Vulkan, source-attested PCI resource budgets, retained physical buffers, failure quarantine, finite grouped kernels. | Current-head B4 physical requalification; all 26 original cases; local Renoir/gfx90c reset/recovery; general all-GPU graph support. |
| Serving | Bounded CPU HTTP OpenAI subset, SSE/queue, finite source-aware CUDA B4 cohort logic, dynamic prompt/decode frontiers, lifecycle checks. | Full vLLM parity, broad sampling/tools/grammars/logprobs, high-concurrency fair goodput and fresh matched C16 across trained source. |
| Memory/SSD | Immutable cache, source window/extents, bounded residency, retained leases and profile design. | End-to-end large trained SSD streamed model, full-domain paged KV/COW/prefix, stable source IO fairness/driver pool reclaim. |
| Pipelines/media | Typed YAML/config, source-bound model/stage interfaces, context/filter/decision/media task scaffolds. | Real live streamed ASR/TTS/image/video with precise clocks/codecs/processor/license/driver gates. |
| CLI/devices | `run MODEL`, `-d auto/cpu/gpu:N/pci:/dev`, optional `--backend`, `devices list` with observed memory, `--dry-run/--explain`, XDG config and alias provenance. 15/15 latest focused CLI tests PASS, CUDA/short-HF typecheck PASS. | General smart multi-GPU layer split, automatic unqualified backend use, complete OS prerequisite checker for /dev support nodes and all-platform routing acceptance. |
| Distributed | Proven source/epoch/topology designs, preparatory actor/IPC concepts. | Authenticated peer semantics and actual two-host model/state/collective proof. |
| Packaging | Rust1.95, 0.1.0, MIT, reproducible Nix/crane config, installers and GitHub workflow definitions. | Final cross-OS/hardware release validation, public perf and source license/security matrices; no push/release performed. |

**Newest focused proof:** after `d848533`, **15/15** run CLI/device tests, **208/208** core tests, default CLI binary smoke, CUDA/short-HF CLI typecheck and default strict Clippy PASS. During the preceding ABI/family refactor, typed GPU ABI **7/7** PASS, cross-backend source/feature checks PASS. In contrast, the combined all-backend cold compile of the newest CLI **timed out**; two broad `scripts/check.sh` attempts on prior post-refactor source each hit **rc124** after ~112–113s (second in real CPU `run_unified`) with **no recorded assertion failure**. **Current full workspace acceptance is OPEN**. These historical and current facts must not be merged into one “all passed” sentence.


## 6. All original delivery phases P00–P23 (capability gates, NOT completion claims)

Original normative phases and the exact test/acceptance details are preserved in the [P00–P23 implementation playbook](../plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md) and [living roadmap](../plans/IMPLEMENTATION_ROADMAP.md). Their numbers describe the original architecture program, **not** the user's later “3 / 4 / 2” execution order.

| Phase | Original contract | Present interpretation / unclosed qualification |
|---|---|---|
| **P00** | Lean reproducible Rust project/toolchain, CLI/source/build/Nix release scaffold | Exists; full native cross-platform installer, signing and release acceptance still open. |
| **P01** | Architecture-specific codegen/device compiler feasibility and source checks | Multiple CUDA/HIP/Vulkan source/physical probes; not a grant for any GPU model. |
| **P02** | Independent original mathematical oracle, state effects, reference numerical traces | Qwen/Gemma controls; long prefix/8K independent mismatches and full learned models remain. |
| **P03** | Verified artifact, quantization codecs, tensor schemas, tokenizer and model identity | GGUF, HF immutable sources and cache exist; derived codecs/providers/rights incomplete. |
| **P04** | Typed semantic program, source operators and trusted task controls | ForwardPlan, source-specific recipes and typed pipeline/task scaffolding; broad stages later. |
| **P05** | Legal closed-region optimization and async resource/command program | Finite prepared candidates with exact ABI; general optimizer/cache transformations incomplete. |
| **P06** | Useful portable CPU inference before accelerator speed | Qwen3.5 source CPU and Gemma candidate, exact numerical source/context gates still scoped. |
| **P07** | Real GPU physical device workers, local pools, leases, completion/fencing | Source-exact partial GPU models; latest complete B4 model/physical gate outstanding. |
| **P08** | First complete trained model/task end-to-end proof | Some source-scope full CPU and finite GPU results, not all model/driver/context combinations. |
| **P09** | Coordinator, request workers, OpenAI-compatible APIs, model routing | CPU HTTP subset and GPU finite serving; full API/sampler/model router proof open. |
| **P10** | Unified physical state residency and safe complete-domain prefix caching | Pools/readers/leases partially done; paged hybrid KV/GDN/COW and snapshots open. |
| **P11** | Stateful GDN/recurrent/attention and hybrid model operators | Exact source numeric islands and source experiments; full learned recurrent variants open. |
| **P12** | Sparse experts, routed parameters and SSD streaming | Source/policy/codec research and some actors; real trained MoE/SSD optimization open. |
| **P13** | First extreme large hybrid sparse family specialization | Qwen Flash-Next/GLM source ambition; no all-layer large-model service proof. |
| **P14** | Substantially different second family on same engine | Gemma4 source and partial CPU/HIP/Vulkan; full26B sparse route not done. |
| **P15** | Target-authoritative MTP/drafter/speculation and full rollback | Design/finite hooks; full verifier/distribution/checkpoint state acceptance open. |
| **P16** | vLLM-class continuous batching, multi-model scheduling and SLO goodput | CPU bounded and native CUDA finite waves exist; fresh C16 + C32/C64 and fairness open. |
| **P17** | Broader dense/encoder/embedding/reranking/LoRA/model roles | Scaffolds/source inventory; per-family trained source acceptance required. |
| **P18** | Real audio/ASR/TTS, image/video, realtime and non-token tasks | Typed assembly/control/clock work; native playable/seeded models not certified. |
| **P19** | Multiple accelerators, collectives, remote authenticated peers and placement | Designs and identity seams; actual two-host state + source proof needed. |
| **P20** | Bounded offline autotuning/costing and negative experiment recording | Per-source tests/profiles; universally optimal policy not claimed. |
| **P21** | Security/license/provenance and reproducible signed releases | Policy and some tooling; full fuzz/SBOM/rights/cross-target gates open. |
| **P22** | Release-level quality/performance matrix and matched complete requests | Partial physical/numerical evidence, current-head B4/C16/large-model matrix open. |
| **P23** | Extension proof without framework growth (new model/kernel/task/peer) | User-required architectural maintainability proof remains active. |

## 7. All current subsystem backlogs, open-item preservation and dependencies

The [live queue](../plans/TODO.md) has **8 active** plans and **3 icebox** plans; the [exhaustive 360-item index](HANDOFF_ALL_OPEN_PLAN_ITEMS_2026-10-25.md) enumerates every `- [ ]`, `- [~]` and `- [!]` source checkbox under its plan, section and line number, including `Uncommitted`, `Tests and verification`, `Deferred` and unchecked items inside old `Committed` sections. Never delete or mark a source checkbox merely because a single historical test ran. Original multiline descriptions and risk controls remain in the linked plan itself.

| Priority/lane | Unchecked items at snapshot | Highest remaining work and evidence boundary |
|---|---:|---|
| **P0 Serving/speculation** [TODO](../plans/serving-and-speculation/TODO.md) | 29 | Source/state/full-vocab API parity; actual same-attempt quiet C16 before larger concurrency; genuine resident GPU batching, fair prefill/decode, paged KV, source-held MTP/draft, bounded SSE/auth/overload. |
| **P1 Numerics/context** [TODO](../plans/numerics-and-context/TODO.md) | 26 | Canonical47 source arithmetic/ordinary states, independent BF16/FLA/stock refs, 2B KKT/GDN partial outliers, nonzero cached trajectories, clean true 8K reference and deferred16/32K. |
| **P1 Memory/streaming** [TODO](../plans/memory-and-streaming/TODO.md) | 25 | Immutable source windows, GPU pool/reader lifetime, phase lending, SSD real row windows, packed experts, safe pager full-domain KV/GDN/COW, meaningful hardware/IO/latency, unknown fence state. |
| **P1 CPU/GPU optimization** [TODO](../plans/cpu-gpu-optimization/TODO.md) | 46 | Matched C1/C4/C8 source/stock records, urgent C16, later C32/C64, FMA/source numerical codegen, cold compiler reuse, NVIDIA/AMD/Vulkan parity, real physical decision based on complete cost. |
| **P1 Pipeline composition** [TODO](../plans/pipeline-composition/TODO.md) | 49 | One shared family/semantic operation/physical owner, existing ForwardPlan profiles, per-stage source/lease/clock, real two-family operation, zero unjustified H2D/D2H, correct fusion/graph liveness, accepted provider reuse. |
| **P1 CLI/devices** [TODO](../plans/cli-device-routing/TODO.md), [design](../plans/cli-device-routing/README.md) | 60 | Source-scheme/effective YAML v2 config precedence, exact device inventory/refusal, actual backend fit/admission, source-qualified single-device before any split, future NPU/remote, full real run/conformance and backend prerequisite checks. |
| **P2 Model execution** [TODO](../plans/model-execution/TODO.md) | 42 | Genuine Qwen3.5 BF16 and Qwen3.8-27B original full window; Gemma4 E2B/E4B/12B/26B-A4B/31B geometry+MoE; all-state/typical natural outputs, trained/source-bound CPU/GPU and exact hardware. |
| **P2 Model acquisition** [TODO](../plans/model-acquisition/TODO.md) | 18 | Immutable source variants, XDG snapshots, typed image/tokenizer sidecars, full provider/format/rights/cleanup correctness, missing/changed bytes, no implicit provider downloads or hidden migrations. |
| **Icebox Multimodal** [TODO](../plans/multimodal/TODO.md) | 27 | True Whisper/Canary/Parakeet/Qwen ASR, Qwen/Kokoro TTS, RVQ speech codec, image generation/editing, Wan/LTX-style video and real-time AV, per-frame/sample clocks/ack/barge-in; no mock→product inference leap. |
| **Icebox Distributed** [TODO](../plans/distributed-execution/TODO.md) | 12 | Distinct compute vs parameters vs live state vs cache placement, authenticated peers/tunnels, typed transfer/collective/epoch, costs+Unknown and first full numerical two-host execution. |
| **Icebox Platform/release** [TODO](../plans/platform-and-release/TODO.md) | 17 | Nix/crane/cross platforms, real vendor/OS legality, provenance/signing/SBOM/licenses, source-minimal feature matrix, compiler/autotune guards and actual releases. |

**Key architectural crosscutting decisions (do not regress):**

- A single semantic `ForwardPlan` and one prepared physical plan/`SessionAuthority`; no new engine for Qwen vs Gemma vs audio.
- Shared immutable source and physical resource state; per-session roots are independent; actual fences and held readers outlive in-flight GPU work.
- Source/quant/numerical/backend compatibility is an explicit finite permit; source filename or GPU family/vendor is not proof.
- Optional plan policy and profile can change physical algorithm but not source mathematics or task target distribution without separate approximate-quality opt-in.
- CUDA, ROCm and Vulkan can address one physical PCI identity through distinct services; never double-admit the card or treat one allocator's free pages as globally reclaimable.
- CLI flags represent user device constraints; vendor support nodes such as `/dev/kfd` or `/dev/nvidia-uvm` are shared driver access requirements, not individually selectable GPUs. Container/sandbox must expose them; future prerequisite diagnostic goes under backend admission rather than multi-`-d` user arguments.
- Historical user choices also prohibit automatic driver resets on local gfx90c, unverified downloads/network routing, silent private source use, unapproved force-push, source equality relaxation and unmeasured “faster than vLLM” statements.
- Run lists and test suites may be reused, but current accepted evidence must be rebound to changed source/binary/model/driver and interpreted *per case*. No stale passing lease/census when new code changed the oracle.

## 8. Mandatory next sequence: user items 3 → 4 → 2 (NOT P03/P04/P02)

These numbers refer to the latest user work ordering, **not** the original 24-phase roadmap.

### 8.1 Item 3 first — full B4 trained/model/physical certification on local RTX3060

**Start with:** [B4/C16 executable continuation](B4_C16_EXECUTION_CONTINUATION_2026-10-25.md), [latest source review](ARCHITECTURE_CORRECTNESS_REVIEW_2026-10-25.md), [typed ABI contract](ABI_FAMILY_CLI_REFACTOR_2026-10-25.md), and live test files `crates/zerogravity/tests/short_hf_cohort_physical.rs` and `crates/zerogravity-gpu/src/resident_program/native_cohort/transport_qualification.rs`. Consult original `.agents/runs/resume-all-takeover-b4-gpu-c16-glm-20261025/`, `.agents/runs/takeover-independent-validation-20261025/` and actual frozen source authorities in the earlier PM handoff.

**Known state, not an acceptance assertion:**
- Original historical `03f2e66` diagnostics on actual RTX3060 passed grouped state transport **18 inner cases** and separate N1 **301 inner finite cases** using **fast diagnostic “release” opt-level0**; genuine zero logical device/events checks. Those are **two test scopes with many inner cases**, not “319/26 cases completed”.
- The current `a39ca96` oracle requires **two independently initialized physical fixtures**, each compared to independent host equations and retired to zero. The newer `83a3174` unified ABI also changed source/metadata. **Old 18-case physical pass does NOT certify current HEAD**.
- Complete **26-scope** actual B4 acceptance still missing: 23 scopes depend on pinned trained source; the independent original B4 **4344-record** CPU archived preflight was **interrupted** at 75s and 101s (not a numerical FAIL/PASS). Verify the current harness list/manifest before claiming counts.
- Qwen3.5 .8B and 2B original snapshots physically exist outside sandbox at `/home/pvl/.cache/abird/zerogravity/` but `dotlink---l5` **cannot read** them under its currently mounted read rules. Ask for **specific read-only path grants**, no model substitute or broad privilege bypass. Follow source SHA/checkpoint/tokenizer exactness from the recorded pin file; don't invent snapshot basenames/hashes.
- The **canonical optimized** release with workspace `[profile.release] lto="thin", codegen-units=1, strip="symbols"` and old preflight requires a **supervised execution session longer than this connector's ~100–120s command lifetime**. Repeated timed commands are incomplete. Diagnostic environment `CARGO_PROFILE_RELEASE_OPT_LEVEL=0` / `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256` / `CARGO_PROFILE_RELEASE_LTO=off` is useful for physical correctness investigation but **cannot** support performance comparisons.

**Required method:** verify no live owner, exact RTX3060 BDF/SDK/driver and original pinned source; enumerate every actual ignored physical test, run each in its own isolated process under the approved command policy/`scripts/native-gpu.py`; preserve stdout/stderr/rc/source sha/binary sha, device event and ledger, fault test root/no-publication, final physical zero or expected retained-fenced unknown. Validate true per-row and grouped fresh-flight bit parity, nonzero normal source traces, source state and natural completion, all original owners and no stale future views. Only sign off after every relevant scope is green. The local AMD Renoir/gfx90c **launch/reset hold** remains binding; do not move this suite to AMD for convenience.

### 8.2 Item 4 second — genuinely matched, quiet, optimized C16 for BOTH .8B and 2B

**Only after B4 accepted on source HEAD.** Build original **optimized** ZeroGravity and stock vLLM arms with exact matching trained checkpoint/source quant/tokenizer/heads, logical numerical/sampler mode, prompts/stop, context, model/batch population (**C16, active8/B4**), GPU and controlled thermals/current capacity. Record actual request offer/complete/error counts, TTFT, decode ITL, complete wall/throughput, p50/p95 where measurable, compiler0/hot preparation and physical service/root/source retirement. Preserve **both arms from the same attempted run** rather than combining best independent historic values or adding an extra sampler.

Historical source21 clean 2B result was **slower** ZeroGravity ~78.7859 token/s versus stock ~95.9974; .8B differential diagnostic had an extra sampler. These are historical bounded observations in [optimization plan](../plans/cpu-gpu-optimization/TODO.md), **not a fresh current performance claim**. C32/C64 are permitted only after the exact complete C16 source/quality/performance gate; **no C2** per user decision. If the canonical source/stock arms cannot match, report that honestly and preserve failed attempts.

### 8.3 Item 2 third — Qwen3.8 27B original full-window capture on x2

**Only after 3 and 4** unless the user explicitly changes ordering. Target is `pvl-x2`, the AMD Strix Halo Ryzen AI Max+395 machine (previously reported high-RAM ~92 GiB; current capacity must be re-observed), **not local l5**. Exact historical source/artifact pin, container recipe and allowed operations are in [PM handoff `4](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md) and `.agents/runs/resume-all-takeover-qwen27b-x2-capture-glm-20261025/{TODO.md,LOG.md}` plus source-pin metadata. Historical user authorization for this specific image/weights workflow exists, **not** permission for different weights, driver resets or disabled access policies.

- Strict x2 SSH from this connector is **blocked by lack of a trusted authenticated ED25519 host key**. Do not turn off StrictHostKeyChecking, guess a fingerprint or claim device access without authenticated identity.
- Historical UD6 foreign Podman container evidence includes `exit 137 / SIGKILL`. **Cause unknown**; not demonstrably OOM without kernel/container evidence. Resolve this and the actual image identity before remote capture; do not infer permission from source-only planning.
- Original base image identity is historically recorded as `docker.io/vllm/vllm-openai@sha256:b18abb2df97b…21e66ac12f` (ellipsis **not an executable digest**); original derived image identity, torchcodec wheel and 31 original weight pins live in the exact run `source-pin.json`. Prefer an identity-preserving content-addressed save/load if verified remote conditions permit; don't rebuild/download and pretend the digest is unchanged.
- Run original full reference/nonzero natural state/logits/feedback with complete checkpoint/container/process/EOF closure on a genuine GPU-free quiet remote owner. This yields original Qwen27B evidence, **not** automatic complete ZeroGravity Qwen27B GPU serving permission.

**If blocked:** progress independent source-only documentation, model/numerics/fixture checks and related safe code work without reordering user priority. State exact missing connector path grant, verified SSH key, or supervised command route rather than bypassing the sandbox or fabricating test receipts.

## 9. Hardware, sandbox, runtime device nodes and tool commands

### 9.1 Host specifics and authorization

- **Connected l5:** `dotlink---l5` repo at `/home/pvl/spaces/abird/src/zerogravity`. The tunnel shell may run with `HOME=/tmp/home` and restrict outside-repo paths. It currently exposes l5's 6 GiB **NVIDIA RTX3060 Laptop GPU**, CUDA driver + locally installed NVRTC/library paths, AMD Renoir RADV/gfx90c (**physical launch hold**) and DRM nodes. Device/driver/current memory must be rechecked; the historical idle RTX3060 value ~41 MiB is not guaranteed current.
- **Remote x2:** AMD Strix Halo Ryzen AI Max+395/iGPU and high-RAM host; its real capability/SDK/sandbox vary from l5 and no live access was proven at latest checkpoint. Do not transplant NVIDIA ordinal or current host free-space values.
- **Linux per-device selectors:** `-d /dev/dri/renderDN`, `-d /dev/dri/cardN`, `-d /dev/nvidiaN`, approved DRM by-path symlinks, `-d pci:DDDD:BB:DD.F`, `-d gpu:N`. `devices::dev` verifies char rdev, DRM sysfs or NVIDIA exact kernel minor and full PCI, then routes through *the same* existing `Observation`. Forced `--backend cpu/cuda/rocm/vulkan` constraints and original physical admission/exclusion still apply. No magic acceptance from a device's name.
- **Backend supporting interfaces:** ROCm runtime may require shared `/dev/kfd` plus the selected DRM render node; CUDA requires `/dev/nvidiactl` plus the per-GPU device and may require `/dev/nvidia-uvm` for managed/Unified Virtual Memory. These are **driver access/support**, **not** unique selected GPU identities. ZeroGravity doesn't open them via the `-d` resolver or mount them inside containers; vendor backend/OS process permissions control use. There is **not yet** a unified read-only backend-specific node prerequisite checker before context creation. This is future design work, not proof of universal access.
- **No machine changes:** no unapproved driver, ROCm/CUDA SDK, GPU privilege/device-node recovery, network download or performance power/governor/clock tuning. The project’s bubblewrap/tooling and XDG permissions are enforced independently of CLI parser validity.

### 9.2 Standard safe commands (inspect executable scripts before executing)

~~~bash
cd /home/pvl/spaces/abird/src/zerogravity
git status --short --branch
git rev-parse HEAD
git log -8 --format='%h %G? %s'
git diff --check

# Narrow source/CLI/numerics controls; omit nonexistent features on any future HEAD:
cargo fmt --all -- --check
CARGO_BUILD_JOBS=2 cargo test --locked --offline -p zerogravity-core --lib
CARGO_BUILD_JOBS=2 cargo test --locked --offline -p zerogravity-cli --bin zerogravity run_cli::
CARGO_BUILD_JOBS=2 cargo test --locked --offline -p zerogravity-cli --features serve --bin zerogravity serve::pending::tests
CARGO_BUILD_JOBS=2 cargo test --locked --offline -p zerogravity-gpu --no-default-features --features vllm-cuda-qualification --lib resident_state_transport::typed::
bash scripts/check-gpu-feature-boundaries.sh
CARGO_BUILD_JOBS=2 cargo check --locked --offline -p zerogravity-cli --features cuda,short-hf-run --tests
CARGO_BUILD_JOBS=2 cargo clippy --locked --offline -p zerogravity-core -p zerogravity-cli --bin zerogravity -- -D warnings

# Genuine isolated whole-gate invocation ONLY with an owned long enough session:
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/native-gpu.py -- bash scripts/check.sh

# Inspect the canonical *optimized* release settings before GPU/perf:
sed -n '/^\[profile.release\]/,+5p' Cargo.toml
~~~

- `scripts/native-gpu.py` supplies checked network isolation; it **does not** download/install SDKs or automatically expose missing GPU device nodes. GPU/model tests require their particular CUDA_PATH, LD_LIBRARY_PATH, source and runtime profile from original per-run documented pins. Do not copy historical Nix store paths blindly; validate they still exist.
- Some SDK/CubeCL builds are huge; one connector shell call may be cut at ~100–120s. Preserve `rc124` as **INCOMPLETE** instead of success or an assertion fail; no unsupervised detached task. For real long release/physical work arrange a truly supervised long-lived owner and preserve exact report/driver/source/binary identities.
- `.agents/runs` can be enormous (~433 GiB). Limit commands to named run folders and file types; never recursively `rg` over all archived runs, and do not perform broad clean/prune of source/outstanding GPU logs.
- Cache and config locations are under XDG absolute bases: `abird/zerogravity`. The pinned model snapshots at user `/home/pvl/.cache/abird/zerogravity/` are currently **not readable** through l5 profile. Read-only grant must be explicitly authorized for exact required paths; do not substitute HOME or copy source through an unauthorized tunnel.
- Repo signing config observed `commit.gpgsign=true`, `gpg.format=ssh`. `git show %G?` returned `U` in sandbox (signature validation/trust **not** established here, not necessarily unsigned). Preserve signing; do not disable to make a commit work. No push, tag, release or history rewrite without explicit instruction.
- Save evidence under an agent-owned scoped `.agents/runs/<run>/` TODO/log path and reconcile only the plan owned by your coordinator. Don’t mutate the user's root README. The connected l5 shell did not expose a `pi` command at the earlier check; delegate only using **actually available** authorized tools and coordinate build/GPU slots.

## 10. Explicit evidence ladder, negative experiments and review rules

A future agent must keep these labels distinct:

1. **DESIGN/PLANNED**: a whitepaper, typed Rust struct, YAML field, source atlas or named model family. No runnable claim.
2. **SOURCE/COMPILE QUALIFIED**: exact source/checkpoint/hash/config/parser and build feature gates; may have no live GPU execution.
3. **FUNCTIONAL**: one bounded real CPU/GPU call, native reference numeric agreement, source-valid state and complete outputs for specific tuple.
4. **STATE/PHYSICAL SAFE**: every required original domain/feedback/attempt/pending/snapshot, write/read event, fault/abort/quarantine/owned retirement and real memory credit/capacity checked; no unowned same-card IO.
5. **SERVING QUALIFIED**: genuine independent sessions, fair admission, multiple requests, HTTP/event/output/cancel closure, exact source and no dropped tail, matching requested OpenAI fields/stream semantics.
6. **PERFORMANCE QUALIFIED**: quiet identical source/profile/client/workload/hardware baseline and real complete-request latency/throughput/p95/energy, numerical/state/closure already accepted; report losses, uncertainty and negative cases.
7. **PRODUCTION/RELEASE QUALIFIED**: OS/hardware/driver/model/format/SDK/rights/security/fuzz/sandbox/operational failure and artifact signatures/installer all independently covered. Compiled Linux binaries do not grant Windows/macOS release acceptance.

**Mathematical authority:** immutable trained model source; correct BF16 exact round/accumulation/FMA, Q4xQ8 vs WideF64, shared GDN/H/KV source state, original known/unknown event root, exact token IDs and full vocab logits, even signed zeros/NaN where relevant. No relaxed tolerances or teacher substitution after seeing a mismatch.

**Physical authority:** `ResourceArbiter` original PCI/host domains, real snapshots and queued/captured/encoded readers, `SessionAuthority` source/attempt/frontier, genuine readback/Event Ready and terminal post-submit fences. Dropping a Future is not driver cancellation, a diagnostic cleanup is not permission to release physically unknown pages, and driver pool VRAM usage is not automatically a leaked logical reservation.

**Benchmark authority:** same-attempt source/stock runs, no active compiler/source/cache scan overlap, cold/warm and thermal/pressure recorded, actual active8/B4/C16, full request latency/throughput and error counts, no cherry-picked local snapshots or synthetic “zero compiler” inference. A source/ABI unit test and the old physical301 fixture are not a throughput proof.

**Secure deployment authority:** local loopback by default for HTTP; auth/TLS proxy if externally exposed; explicit application egress/tool grants; no secret/key leakage in logs; remote host keys authenticated rather than waived; distinct tenant/model/source/draft/peer epochs; source rights/license and trusted immutable variant identity.

## 11. Exact final handoff inventory and provenance checklist

- [Original architecture](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md), [ES-00…ES-15](../../docs/ENGINEERING_SPECIFICATION.md), [historical decisions](../plans/archive/history/DECISION_REGISTER.md), [P00–P23 roadmap](../plans/IMPLEMENTATION_ROADMAP.md), [v5.1 proposals](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md).
- [Latest live active/icebox plan queue](../plans/TODO.md) and **[all 351 unchecked plan entries](HANDOFF_ALL_OPEN_PLAN_ITEMS_2026-10-25.md)**. The live plan list and unchecked item counts can change after the next agent commits; this appendix is a fixed source SHA checkpoint and is intentionally not a parallel queue.
- [Complete 252-file design, plan, research, prior-art and publication inventory](HANDOFF_ALL_DESIGN_AND_SPEC_SOURCES_2026-10-25.md), including every original surviving Markdown source and its SHA256. This index is descriptive, not an approval record.
- [All 245 source commits](HANDOFF_ALL_COMMIT_LINEAGE_2026-10-25.md) with ancestry order. A new handoff documentation commit belongs *after* the indexed source HEAD, not in the older snapshot.
- [17 Oct complete historical handoff](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md), [19 Oct](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md), [20 Oct](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-20.md), [25 Oct AM](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25.md), [25 Oct PM](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md). Read those only for detailed history/authorizations; never promote their older status labels to this source.
- [Post-4bf source audit](POST_4BF3673_AUDIT_AND_RESUME_2026-10-25.md), [B4/C16 precise results](B4_C16_EXECUTION_CONTINUATION_2026-10-25.md), [adversarial source review](ARCHITECTURE_CORRECTNESS_REVIEW_2026-10-25.md), [ABI/family/CLI source refactor](ABI_FAMILY_CLI_REFACTOR_2026-10-25.md), [Linux /dev device selector](LINUX_DEV_NODE_SELECTORS_2026-10-25.md). These are the latest concrete engineering deltas.
- **No file/model/work lost:** old failed/interrupted hardware and numerical logs remain under their named `.agents/runs` directories (often ignored and giant). Old frozen benchmark data are not current benchmark output. Source-derived binary SHA and model/cfg/driver pins must be validated independently when an agent actually runs them.
- **User constraints:** root README bytes unchanged, no unapproved model/SDK downloads, no local Renoir/gfx90c launch/reset, no x2 SSH bypass, no hidden source fallback, no push/tag/release/rewrite, no fabricated “source-safe” math, new runtime, unverified speedup or unqualified model support.

## 12. Ready-to-paste next agent starting prompt (also retained in .agents/plans)

~~~text
Take over ZeroGravity using the connected dotlink---l5 repository at
/home/pvl/spaces/abird/src/zerogravity.

Read ALL of .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-NIGHT.md
as the newest comprehensive entrypoint. Then read AGENTS.md, the .agents/plans
README and TODO index, the owning subsystem plan TODOs, and the actual
archived .agents/plans/archive/history/DECISION_REGISTER.md, original
v4.3 master plan, whitepaper/ES-00...15 and latest shared ForwardPlan
architecture. Use the TWO exhaustive companion handoff indexes for all
245 reachable Git source-history commits and all 360 non-completed plan markers (351 unchecked, 8 active, 1 blocked).
Older 17/19/20/25AM/PM handoffs remain essential historical provenance
but do not override current source, superseding reviews or user decisions.

Start by INDEPENDENTLY checking Git HEAD/status, signed commits, concurrent
worktrees/jobs, device/SDK and model-source permissions and current hardware,
then mapping tests and incomplete receipts to their exact source/binary pins.
Preserve the user-owned README byte-for-byte; no unapproved model/SDK/network
downloads, driver changes, GPU resets, privilege changes, x2 SSH host-key
bypass, pushes/tags/releases/history rewrites or silent source/numerics/model
substitutions. The local AMD Renoir/gfx90c launch/recovery hold remains.

Follow the latest explicit user sequence, not the numeric phase names:
(3) requalify the current independently initialized B4 grouped CUDA oracle,
complete the original trained .8B/2B 26-scope physical/CPU source suite
on l5 RTX3060 with proper original pinned source and exact final ownership;
THEN (4) build an actually OPTIMIZED release and run fresh SAME-ATTEMPT,
quiet C16 stock vLLM vs ZeroGravity for BOTH models (no C2; C32/C64 later);
THEN (2) Qwen3.8 27B original natural full-window reference capture on x2,
only after authentic strict SSH trust and UD6 exit137 forensics.

Important blockers: l5 dotlink sandbox currently has NO read grant to exact
pinned Qwen .8B/2B snapshots under /home/pvl/.cache/abird/zerogravity;
the canonical optimized build and original 4344-record CPU preflight
need a supervised long-lived process beyond the ~100s relay; the new
B4 oracle has not yet been rerun physically, so historical 18-case
physical plus finite301 old-source diagnostics DO NOT certify it;
the x2 trusted ED25519 SSH host key is unavailable and the historical
podman exit137 cause is UNKNOWN, not confirmed OOM. Never relax the sandbox,
guess host-key identity or mark an interrupted run green. No current matched
C16 speedup may be claimed.

Use one resource arbiter / one ForwardPlan / one StateRoot publisher /
exact source numerical semantics and original per-device physical owner.
Keep CUDA/HIP/Vulkan source/ABI/driver identities separate. Model metadata
does not authorize inference; -d /dev/... is only a unique GPU selector,
not /dev/kfd/nvidia-uvm driver-support permission provisioning. Preserve
every rejected case and all independent reference numerical/physical facts.

While access is legitimately blocked, execute safe independent source-only
review/fixtures and keep the canonical plan TODOs (only your owned lane)
and run-local evidence exact. Use available authorized subagents only where
independent scopes and GPU/build exclusivity can be enforced. Do narrow
controls, then the full scripts/check.sh gate on assembled source with an
actually supervised enough session. Record binary/model/driver/SDK SHA,
full stdout+stderr+rc/timeout, failures/fencing/retirement and matched
baseline from the SAME source. Sign scoped commits; never push.

Return clear delivered changes, validated tests, true remaining blockers
and exact source commits. Do not repeat old work without revalidating source
identity, and do not claim asynchronous/background completion.
~~~

**Final review reminder:** This handoff creates no new hardware execution permission, model qualification, inferred speedup, or release claim. The user's working order and original complete design/decision provenance are preserved through the canonical links above and the exhaustive two companion indices.
