# ZeroGravity — comprehensive handoff, 26 October 2026

Author: ROOT / GPT-6.1 Sol xhigh
Co-authors: GPT-6.1 Sol xhigh history, open-work and design audit lanes
Owner: @pvl

This is the newest comprehensive entrypoint through implementation commit
**d74f59a18c6345dbea963861acd519c1282a26fa**. Read it completely, then use the
companions and original sources below. It preserves the original project vision,
decisions, complete reachable history, every current unfinished plan block,
recent source changes and actual acceptance boundaries. It creates no new
runtime, model qualification, performance claim or recurring tracking duty.

The immediate next source work is **actual removal of captured-testcase dispatch
and qualification-harness coupling from ordinary inference**, including
parametric GDN and honest model/backend admission. Then resume current-source
.8B fresh/cached47–50 correctness, genuine context/backend routes, affected
.8B/2B B4 and shutdown qualification, optimized matched C16, and finally the
original Qwen27B/x2 full-window reference under its own gates.

## 0. Snapshot, scope and completeness

| Item | Verified state / scope |
|---|---|
| Repository | /home/pvl/spaces/abird/src/zerogravity; connected l5 repository, originally dotlink---l5. Current tool permissions must be independently observed. |
| Branch/code cutoff | main; d74f59a18c6345dbea963861acd519c1282a26fa. The documentation commit containing this handoff follows that code cutoff. |
| Git before handoff | Main tracked/index clean;63commits ahead of locally recorded origin/main; no push/fetch performed. All272reachable commits verify G under current local allowed-signers configuration. |
| Side worktree | /home/pvl/spaces/abird/src/.zerogravity-worktrees/qwen35-prompt-reference-20261026, branch codex/qwen35-prompt-reference-20261026; same code cutoff,0/0divergence, clean tracked/index. |
| Only pending side overlay | tools/cuda-gdn-recipes/:11untracked files/64860B; preserved unstashed, unintegrated and untested. Full non-executable text snapshot and original custody are linked below. |
| Stashes/jobs | No stashes, conflicts, unique side commits or active owned build/model/device tool sessions. Handoff subagents only inspect/write disjoint documentation. Re-observe concurrency before work. |
| Protected README | SHA256 6c00873ec0ac2c59bc73703b55a9bfa4d93703334a03fa16bcac2d204cda5b15. User-owned; leave its bytes unchanged. |
| Latest source gate | Strict CUDA/Vulkan all-target Clippy0;288affected host tests pass; complete isolated scripts/check.sh0,2050Rust executions/0fail/91ignored,12Python suites, release/installer/shell/Nix. Full streams/group/reap closure. These are source/host checks. |
| Latest physical scope | Original .8B fresh47/48/49/50 full-ABI reference captures complete. Current refactored native binary has no fresh model/physical/C16 qualification. Earlier accepted results remain tied to earlier source/binaries. |
| Documentation coverage |272reachable commits;374noncompleted live-plan markers;11queue markers separately;5closed archival TODOs;264original document/supplement files. |
| Historical coverage | Original245commit and360noncompleted-marker companion indexes remain unchanged. All360old marker identities are retained;14new ones account for374current. |
| Date/time | Handoff date26Oct2026, user-facing client timezone Asia/Calcutta. Raw execution receipts keep original UTC/offset metadata. Imported Git author dates do not establish execution chronology. |

This task writes documentation only. It does not rerun physical/model gates,
download inputs, change source/plans/drivers/permissions, contact x2, or publish.
The last read-only l5 observation was kernel7.2.9, RTX3060Laptop6144MiB,
driver595.91.07,41MiB reported used and no listed compute applications. That is
an observation, not a future reservation.

### The complete companion set

| Read | Purpose |
|---|---|
| [Starting prompt](NEXT_AGENT_PROMPT_2026-10-26.md) | Copyable next-agent instruction, with current sequence and safety boundaries. |
| [User instructions and supersessions](handoff-2026-10-26/USER_DECISIONS.md) | Current controlling decisions and the takeover-session instruction sequence, including the rejection of convention-test proliferation. |
| [Full design and decisions](handoff-2026-10-26/DESIGN_AND_DECISIONS.md) | V1→V4.3→V5.1→current shared architecture, original U/P/O/A decisions, ES-00…15, P00…23, subsystem contract route and all surviving original documents. |
| [All272commits](handoff-2026-10-26/SOURCE_HISTORY.md) | Full SHA/date/signature/subject/actual-path classification and owner summaries for every reachable commit, including documentation and imported history caveats. |
| [All unfinished work](handoff-2026-10-26/OPEN_WORK.md) | Every374live noncompleted occurrence copied with its complete multiline block, exact source span/hash/heading; queue and archived TODOs separately inventoried. |
| [Exact evidence and continuation](handoff-2026-10-26/EVIDENCE_AND_CONTINUATION.md) | Source/binary/model/driver/SDK identities, all four fresh original captures, failed predecessors, current rotary/GDN debt, supervision, old B4/C16 and x2 gates. |
| [Complete264-file source catalog](handoff-2026-10-26/SOURCE_CATALOG.md) | Every tracked original under .agents/docs, .agents/plans, docs, research and prior-art, including full heading routes, bytes, SHA and non-Markdown supplements. |
| [Preserved continuation reports](handoff-2026-10-26/continuation/README.md) | Eight byte-identical recent design/audit/result snapshots plus the full eleven-file pending GDN draft as non-executable text. |

The59original publication/research/prior-art documents outside the requested
roots are also copied byte-identically,1331632B, under
handoff-2026-10-26/reference-sources/{publication,research,prior-art}/.
Originals remain canonical. These dated copies ensure their full bodies are
available inside .agents/docs; they are not competing live specifications.
Internal links in copied bodies retain original bytes and resolve from the
canonical source path named in the catalog.

No claim is made that unavailable original history was recovered. The original
33-node Git graph, later129/191-source historical aggregate versions and full
70-report set were not recovered by the earlier audit. Surviving94-source
history, recovered volumes, original decisions and reachable272-commit ancestry
are preserved. Do not invent missing SHAs, source bodies or acceptance results.

### Terms used in retained evidence

| Term | Meaning |
|---|---|
| Prefill | Process a fresh prompt or actual prompt chunk; report its token count and prior cached tokens. |
| Decode / cached continuation | Consume an actually generated token with an existing prefix; frontier is consumed context, not output count. |
| N1 / B4 | Historical shorthand for single-request execution / grouped four-request execution. New fields use request_batch_size explicitly. |
| C16 | Sixteen concurrent offered client requests. It does not mean a native sixteen-row kernel; older measurements had native peak8 and grouped4 execution. |
| GDN | Gated-delta recurrent operators and their state/control/kernel family; distinguish them from ordinary attention/KV. |
| BF16 / F32 / quantized source | Precision or exact packing contracts, not interchangeable model identities or complete numerical profiles. |
| Original reference | The pinned independent upstream implementation and actual source/shape/schedule; matching its text alone does not establish its full math/state. |
| Stock | Pinned original vLLM in the current matched serving benchmarks. llama.cpp supplies separate comparison/reference practices and explicit GGUF scopes. |

## 1. How a successor obtains context

1. Independently inspect HEAD, status/index, signatures, worktrees, stashes and
   active process/device owners. Do not assume this snapshot remains current.
2. Read [AGENTS.md](../../AGENTS.md) completely, then the
   [plan rulebook](../plans/README.md) and [live queue](../plans/TODO.md).
   Current user instructions supersede older Pi/Astra/process text.
3. Read this handoff and the complete user-decision, design, history, open-work
   and evidence companions. If a tool truncates output, read the omitted spans;
   do not claim a truncated response was a full read.
4. Read the actual [archived decision register](../plans/archive/history/DECISION_REGISTER.md),
   [master v4.3](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md),
   [whitepaper](handoff-2026-10-26/reference-sources/publication/WHITEPAPER.md)
   and [engineering specification ES-00…15](handoff-2026-10-26/reference-sources/publication/ENGINEERING_SPECIFICATION.md).
   Original publication bodies also remain under root docs/.
5. Read current [shared ForwardPlan architecture](../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md),
   [structure/naming](../plans/pipeline-composition/STRUCTURE_AND_NAMING.md),
   [format homes](../plans/pipeline-composition/FORMATS.md),
   [correctness policy](../plans/numerics-and-context/CORRECTNESS_POLICY.md)
   and the owning plan TODO plus its linked specifications and actual code.
6. Use the source catalog to recover earlier rationale and every task's full
   contract, not merely the latest short status. The older17/19/20/25AM/25PM/
   25NIGHT handoffs remain essential historical provenance. Their prompts are
   not commands to execute and their old status does not override later source.
7. Before model/GPU/remote execution, authenticate current source/model/ABI/SDK,
   exact hardware/access and actual ownership. Inspect the source and complete
   semantic/cleanup outcomes behind any accepted or failed receipt you reuse.
8. Continue the authorized next substantial source unit; do not create another
   onboarding ledger, per-helper acceptance bureaucracy or convention test suite.
   Context can be read in bounded chunks with retained notes; no broad recursive
   scan of giant ignored run trees is needed.

Authority order: current explicit user decisions and actual permissions;
live source/tests and executed evidence for their exact scope; owning plan
TODOs; applicable normative contracts with supersessions; historical snapshots.
A source test cannot override a failed physical result. Old prose cannot turn
a new unsupported model/backend into a supported one.

## 2. Original product vision and immutable architecture

ZeroGravity aims to run supported models locally through one Rust-owned
execution fabric for text, image, video, audio, real-time and composed multimodal
tasks. The ambition includes ordinary small models and extreme dense/MoE models
whose storage/residency spans GPU, RAM and SSD. This is the product direction,
not a statement that every named model/format/backend currently works.

The original design evolved from portable semantic math and specialized kernels
to a minimal shared architecture:

~~~text
authenticated immutable checkpoint / tokenizer / codec / source roles
                              |
           model-family mathematical traversal and typed task control
                              |
                   one semantic ForwardPlan
          exact numerical profile and versioned state effects
                              |
        admitted prepared physical regions / source / ABI / costs
         CUDA, HIP, Vulkan, CPU, native graph/AOT where qualified
                              |
      one ResourceArbiter authority for each real physical domain
             fair queues, pools, source windows and readers
                              |
      submitted work -> real completion/readback OR fenced Unknown
                              |
            one session/task authority publishes StateRoot
                              |
                committed CLI / HTTP / pipeline output
~~~

- A model describes mathematics; a task describes control; a provider supplies
  an implementation; a plan schedules physical resources; one coordinator owns
  visible state. Model-specific runtimes, second IRs and competing publishers
  were explicitly rejected.
- Source identity, semantic source roles, numerical implementation, compiled
  kernel/ABI, physical device and driver/SDK are independent identities.
  A model slug, metadata, enum, feature or filename grants none of the others.
- Shared immutable weights may serve many independent sessions. Each session
  retains genuine source/attempt/epoch/frontier-bound roots. Domains prepare;
  only the original coordinator publishes.
- One process-shared ResourceArbiter unifies views of a real PCI device even if
  CUDA/HIP/Vulkan expose different services. It is not a cross-process/fleet
  lock. Unknown work retains actual backing/readers/modules and fences reuse.
  Dropping a Future is not driver cancellation.
- ForwardPlan is the durable mathematical representation; region/compiler views
  are transient. A prepared implementation can cover an operator, fused region,
  captured graph, AOT/static fabric or qualified whole subgraph. Prepared shape
  guards are real implementation limits, not authorization from a test matrix.
- Four distributed placements remain distinct: computation, parameters, mutable
  state and immutable caches. Transfers have costs/epochs/rights/ownership; an
  authenticated tunnel is not free RDMA or a distributed atomic GPU transaction.
- Typed task values/clocks support non-token media. Model commit, transport ACK
  and audio/video playback are different events. Real codecs/processors and
  source-specific tasks need actual quality and synchronization proof.
- Fast serving means complete-request goodput/latency/fairness with compatible
  batching, chunked prefill, shared prefixes and target-authoritative speculative
  verification/rollback. Kernel launch reduction is not a measured speedup.

### Surviving original specifications and rulings

The design companion maps every ES-00…15 and P00…23 contract, U-001…U-070,
P-001…P-096, O-001…O-074 and A-001…A-006 to its original source. Those namespaces
are not interchangeable. A proposal/open question is not a user approval.

Important supersessions:

- V4 minimality narrows earlier multi-authority/permanent-IR sketches.
- V4.1 allows providers at natural kernel/graph/AOT/static region granularity.
- V4.2 adds distributed placement/peers under the same state/resource fabric.
- V4.3 adds one-command local execution and same-engine multi-model routing.
- V5/V5.1 extend typed media, duplex/QoS, verified assemblies and tensor/SSD
  backing; they do not grant implemented multimodal capability.
- MIT was selected for project source/docs; third-party/model/SDK rights remain.
- YAML supersedes historical proposed TOML for current pipelines/config.
- The21Oct owner reuse directive permits suitable audited MIT/Apache Rust,
  static C ABI/native components and narrow C++ shims, or faithful attributed
  ports. It supersedes older blanket from-scratch/Rust-only foreign-kernel
  restrictions. Rust-owned semantics/runtime/authority remain.
- A later explicit resume supersedes old planning-only pauses. New task scope,
  source permissions, device holds and publication authorization still apply.
- Plan/run TODOs replaced mandatory per-file SHA inventories/generated handoffs.
  This comprehensive snapshot is explicitly requested now; do not impose its
  construction as a routine future prerequisite.

## 3. Source ownership and the current refactor

| Owner | Responsibility |
|---|---|
| zerogravity-core | Semantic program/ForwardPlan, source identities, state/root contracts, resource arbiter, device events/queue, task/pipeline and generic optimization contracts. |
| zerogravity-artifacts | Immutable source/cache/manifests and actual codec/layout interpretation shared across models. Family-specific config/tensor roles belong to models. |
| zerogravity-cpu | Shared operations, exact numerical contracts, generic source access and CPU runtime. |
| Existing Qwen and Gemma model crates | Family config, tensor roles/ties, authenticated semantic source binding, mathematical traversal and model state/tokenizer/execution adapters. |
| zerogravity-gpu | Shared CUDA/HIP/Vulkan physical implementation, compiler/ABI, leases, pools, transport, completion/quarantine. No production model-crate dependency. |
| zerogravity facade / CLI | Prepared model/application adaptation, sessions, worker/HTTP/run/serve and user-facing device/source routing under original owners. |
| Rust tests / explicit diagnostic tools | Cases, expected values, reference adapters, independent comparison and qualification orchestration. |

Real family responsibilities moved out of shared-core/artifact/GPU interpretation;
GPU production dependencies on Qwen were removed in263ff79. Model-backed unit
fixtures may retain dev-only dependencies. Model references in tests are not
evidence that production coupling remains.

CUDA, HIP/ROCm and Vulkan have separate modules/providers with separate
source/driver/ABI identities. Quantized-source code has contained homes for
Q1/Q2/Q3/Q4/Q6/Q8/NVFP4; BF16 is precision. Q1/Q2/Q3/NVFP4 still have explicit
missing implementation work. Q4_0/Q4_1/Q4_K and Q6_K/Q8_0 are different byte
formats. A scalar codec does not imply a complete operator/model route;
the existing source-projection Q4_K refusal remains meaningful.

Names describe behavior and explicit fields: phase, prompt_tokens, cached_tokens,
input_tokens, request_batch_size, generated_tokens, context_capacity_tokens.
Primary operators are gdn/attention/rms_norm; distinct alternatives use useful
implementation suffixes. Historical names remain only where genuine upstream
numerical identity or succinct grep provenance requires them.

<!-- Previous labels: short_hf, original_hf, canonical47, case5, initial48–50, vllm_gdn, UD6, exit137. -->

### What remains architecturally wrong

The recent user complaint is valid: finite capture/test policy still controls
ordinary model/backend execution. Moving files or renaming a whitelist as a
capability descriptor does not solve that.

Remaining paths include model prompt admission and eager templates, expected
GDN capture slices, GPU observed-specialization selection, packet materializers
and vendor/cohort call caps. The shared rotary core still carries diagnostic
fault orchestration, and fixed-prompt loading still invokes a qualifier.
Actual kernel mathematics/ABI constraints must be preserved while separating
those policies and implementing real supported dynamic shapes.

The latest1f08e55 checkpoint removes ordinary text loading's qualifier call,
prepares rotary source for configured context and derives exact immutable native
upload/resource charges from real bindings. It moves actual test-only helpers/
report facts into Rust test support. It does not close the remaining debt.

The next GDN design is concrete: reuse existing typed scalar-block transport,
derive controls/strides from real retained geometry, preserve genuine numerical
kernel variants and replace case dispatch with real implementations. Pure
reversible proof covers all96original full CUDA bodies;17instruction/ABI bodies
remain after accounting for identical wrappers. No dynamic GDN runtime has been
implemented or physically qualified. Details and pending draft custody are in
[the evidence companion](handoff-2026-10-26/EVIDENCE_AND_CONTINUATION.md).

## 4. Correctness method across all models and backends

The user explicitly requested the approach of both pinned vLLM and llama.cpp.
The [policy](../plans/numerics-and-context/CORRECTNESS_POLICY.md) defines four
separate claims:

1. Exact semantic/source/state/ownership/cancellation/physical invariants.
2. Per-operator accuracy under the declared dtype, packing, reduction and layout;
   existing source-bit contracts stay exact. A separately declared tolerance
   profile needs independently established limits before measurement.
3. Independent model/task accuracy for the same checkpoint/tokenizer/template/
   inputs/settings, with aligned output/distribution/quality evidence.
4. Determinism, batch/schedule invariance and cross-hardware bit equality only
   where separately promised and tested.

There is no universal epsilon and no universal bit-identity requirement across
CPU/CUDA/HIP/Vulkan. Appropriate independent references do not erase structural
or exact-source requirements. Matching tokens/top1 alone does not establish
state/distribution identity; model metadata and synthetic operators establish
no missing full-model adapter.

The historical strict single-request-vs-batched cached-decode comparison
fails101 and deliberately quarantines before Ready. Independently authenticated
original vLLM N1/B4 also differs in six BF16 QKVZ words at identical frontier47
inputs. Preserve both facts. Default model accuracy and optional bit-invariance
must be reported separately; neither a new name nor a cleanup0 makes the old
failed experiment pass.

Fresh47/48/49/50-token prompts differ from continuation of a47-token request.
The original reference captures now cover all four fresh prompts and their
actual cached histories with full state/vocabulary/source/ABI evidence.
Native fresh48–50 is still unsupported pending genuine source/ABI/control/math/
physical integration. Do not merely append those lengths to runtime tables.

Genuine context qualification requires real full-model prefixes/continuations
at512/1024/2048/4096/8192, then16K/32K, with state/replay/chunk/fault coverage.
Large rotary positions, empty padding, another model or a different GGUF
numerical profile do not qualify original BF16 CPU/ROCm/Vulkan routes.

## 5. What landed, what actually ran, and what did not

Every step from the beginning is in the272-row history index. The recent
transition is summarized here without flattening evidence levels:

1. Stronger independently initialized B4 oracle replaced the flawed shared/
   already-updated-buffer comparison; later fixture lifetime/accounting repairs
   allowed a real optimized A3 build and physical execution.
2. On that earlier A3 source, grouped oracle/CPU4344/finite301 and finite
   width/schedule/API/HTTP scopes passed. Original26 as a whole did not:
   24pass, one unqualified export, one strict N1/B4 rejection. SDK observer
   attempts1–4 remain rejected; V5 has its bounded live-snapshot authority.
3. Earlier same-attempt quiet C16 completed both arms for both models:
   .8B stock98.2827tok/s versus ZG111.1738, rate/wall pass;
   2B stock96.8627 versus ZG84.1017, rate failure/pair incomplete.
   No both-model win or current-source speedup follows.
4. Architecture/model/backend/format/naming refactors landed in logical units;
   preserved stashes were reviewed, missing source guards recovered, completed
   worktrees retired and GPU production model dependencies removed.
5. The l5 benchmark container termination was causally narrowed to probable
   five-second cleanup escalation against an old PID1 server without serving
   shutdown. Historical fatal signal/sender/OOM remain unproven.
6. New serving shutdown and owned-stop evidence landed. Actual model-free
   process/PID1/container controls pass; trained-model/backend replay remains.
7. Shared command ownership and finite original reference adapters/bundle
   staging landed. All23approved small-model files became legitimately readable;
   long supervised source/build owners completed. Old relay/access blockers
   are historical, not reasons to bypass permissions or re-run obsolete work.
8. Fresh original47–50 paired/full-ABI captures completed,408archives and
   31008launch packets with all required state/vocabulary/source/cleanup checks.
   Failed and incomplete predecessors remain preserved separately.
9. Configured rotary source preparation and real binding-derived upload/costs
   replaced ordinary text loader test assertions. Test-only report/helper
   removal changed the honest2B aggregate source fingerprint; executable packet
   and GDN payload bytes stayed exact.
10. Offline recipe generation for accepted48/49/50 captures completed; candidates
    remain diagnostic data, without native admission or model qualification.
11. The assembled source batch passed strict CUDA/Vulkan Clippy,288affected host
    tests and full scripts/check.sh. Source changes were signed as3612ecb,
    1f08e55,d74f59a. The single post-batch worktree reconciliation finished;
    only the unique GDN draft remains pending.

Exact result paths, complete raw streams, failures, binary/model/SDK hashes,
source identities and supervision commands are in the evidence companion.
No current refactored optimized inference binary, physical .8B47–50 acceptance,
long-context/backend expansion or current matched C16 has been claimed.

## 6. Complete subsystem backlog and broader product goals

The live queue contains8active and3icebox plans. Every current noncompleted
multiline block is retained in OPEN_WORK, including6under old Committed headings,
57under test headings,16deferred and295under Uncommitted. These are374source
occurrences, not374independent engineering units. Counts do not measure progress.

| Plan | Noncompleted markers | Existing foundation and remaining larger outcome |
|---|---:|---|
| [Serving/speculation](../plans/serving-and-speculation/TODO.md) |30|Same-engine CPU/finite CUDA HTTP/SSE and shutdown foundation. Remaining trained-model shutdown/faults, fair continuous batching, full source-aware API/sampling/overload, paged state, MTP/draft verification and goodput.|
| [Numerics/context](../plans/numerics-and-context/TODO.md) |31|Shared correctness policy and exact finite/source/reference tools. Remaining native47–50/cached,2B outliers, full context/backend routes and actual state/physical closure.|
| [Memory/streaming](../plans/memory-and-streaming/TODO.md) |25|Immutable windows, residency/leases/readers and bounded source work. Remaining real trained SSD streaming, safe complete-domain pager/KV/GDN/COW/prefix, IO fairness, liveness and measured fit/latency.|
| [CPU/GPU optimization](../plans/cpu-gpu-optimization/TODO.md) |55|Prepared CUDA/source/SDK/physical evidence and historical measurements. Remaining correctness dependencies,2B cost diagnosis, fresh optimized C16, laterC32/C64 and separately qualified HIP/Vulkan/native strategies.|
| [Pipeline composition](../plans/pipeline-composition/TODO.md) |53|Shared ForwardPlan, model containment, backend/format organization and dependency cleanup. Remaining actual runtime/test separation, generic prepared composition/physical costs, reusable resident stages and full two-family extension proof.|
| [CLI/device routing](../plans/cli-device-routing/TODO.md) |60|Unified run/source/config/device observation, explain/dry-run and physical selectors. Remaining actual route/admission coverage, source-specific fit, support-node diagnostics, multi-device/NPU/remote and full command conformance.|
| [Model execution](../plans/model-execution/TODO.md) |46|Qwen/Gemma family source/contracts and finite CPU/HIP/Vulkan/CUDA proofs. Remaining complete learned natural/reference/context/backend cells, Qwen27capture, Gemma PLE/lifetime/feedback/MoE and larger families.|
| [Model acquisition](../plans/model-acquisition/TODO.md) |18|Verified immutable source/cache/provider/layout foundation. Remaining variants/sidecars/rights/provider/format/cleanup cases and exact authorized establishment; acquisition is not inference.|
| [Multimodal](../plans/multimodal/TODO.md) |27|Typed values/tasks/clocks/assemblies and research. Remaining real ASR/TTS/image/video/direct-decision/codec/playback/duplex/barging quality and full source/device ownership. Icebox.|
| [Distributed](../plans/distributed-execution/TODO.md) |12|Placement/peer/epoch/transfer designs and seams. Remaining authenticated actual multi-host numerical/state/collective/IO proof, rights and complete costs. Icebox.|
| [Platform/release](../plans/platform-and-release/TODO.md) |17|Rust/Nix/crane/installers/cross-workflow scaffolding. Remaining real OS/hardware/SDK/license/security/SBOM/fuzz/artifact/release matrix. No push/tag/release authorized. Icebox.|

Longer-range model ambitions remain in the master/registry/contracts: Qwen
larger dense/hybrid/Flash-Next, Gemma E2B/E4B/12B/26B-A4B/31B and MoE, GLM/
DeepSeek and other text/embedding/vision/audio/video/decision families. No
registry slug or source metadata closes their actual full-model gates.
Speculation, SSD extreme-model execution, cooperative GPU/NPU, multi-host
placement and broad vLLM-class serving remain larger goals, not completed
features inferred from this refactor.

## 7. Latest executable sequence and acceptance boundaries

Follow the actual user sequence, not original P-phase numbers or stale
performance-first headings:

1. Finish the current substantive architecture unit across .8B/2B/27B/Gemma:
   actual model containment, real parametric GDN/kernel/config/source contracts,
   ordinary loading/execution separated from qualification/fault/case policy,
   and clean consumers. Keep one engine/ForwardPlan/arbiter/publisher.
   Full27B/Gemma model suites are waived for this refactor only; source/feature
   checks still cover them. No new convention tests.
2. Build an actually optimized current binary and complete .8B fresh47–50
   plus genuine cached continuation against its appropriate pinned independent
   profile. Exercise affected2B and actual B4 source/CPU/physical coverage,
   cancellation/Unknown and trained-model/backend shutdown/serving. Preserve
   optional strict invariance failures and all structural/physical predicates.
3. Implement and qualify genuine long contexts and missing original BF16
   CPU/CUDA/ROCm(HIP)/Vulkan full-model routes. Respect the local gfx90c hold.
   Do safe independent source work where device/source access is blocked.
4. After correctness, run fresh quiet SAME-ATTEMPT optimized C16 stock vLLM vs
   ZeroGravity for BOTH small models. Match source, tokenizer, template,
   numerical/sampler contract, prompts/context/stop/population, nativeactive8/B4,
   GPU and capacity/thermal/owner conditions. No C2; C32/C64 follow later gates.
   Report loss/error/uncertainty honestly; do not combine best historical arms.
5. Only after those prerequisites, run the original Qwen3.8-27B natural
   full-window reference on x2 with exact original source/image/tool/math and
   independent validator. No short-context, teacher, changed image or model
   substitute. It does not automatically certify ZeroGravity27B serving.

### x2 and incident gates

The old trusted-key blocker was superseded by an actual strict ED25519 trust
verification, with unchanged known_hosts. Last authenticated preflight instead
found missing exact31model members/image, available CPU RAM below16GiB, missing
prescribed PATH Python and an owner/validator source_open signature mismatch.
Actual host/resource/source/owner/validator conditions must be re-observed.
Do not double-count96GiB GPU addressability as free CPU RAM.

The historical l5 container incident's supported cleanup-escalation inference
and preventive shutdown fix do not supply missing fatal-signal/OOM proof.
Preserve its hold/evidence and all applicable capture/benchmark gates; do not
guess causality or relabel it as an x2 crash.

## 8. Safety, resource custody and efficient collaboration

- Preserve the root README exactly. Never overwrite other agents' staged,
  uncommitted or ignored work. Keep the partial GDN side tree until its unique
  work is resolved; it is not an empty merged tree.
- Signed scoped commits only. No push/tag/release/deploy/history rewrite,
  reset--hard/branch deletion, broad cleanup or unapproved external messaging.
- No unapproved model/SDK/network downloads, privilege/driver/device recovery,
  resets, power/clock changes or x2 trust bypass. The Renoir/gfx90c launch/
  recovery hold remains. Source metadata is not inference authorization.
- -d chooses one physical GPU via PCI/ordinal/character node. /dev/kfd,
  nvidiactl and nvidia-uvm are shared backend support interfaces, not additional
  selectable GPUs or automatically provisioned permissions.
- Keep CUDA/HIP/Vulkan compiler/source/ABI/driver identities separate. Actual
  source numerical casts/reductions/rounding/FMA and codec variants are not
  silently exchangeable.
- Use GPT-6.1 Sol xhigh for every lane. Declare disjoint file ownership, one
  writer for shared plans and explicit build/GPU/memory ownership. A subagent's
  final message is not independent evidence: coordinator reviews actual bytes,
  executed results and failures.
- Work in substantial batches; use narrow diagnostics only to resolve concrete
  blockers. Fix related failures together. Run required affected numerical/
  source/state/physical/serving checks, then the complete assembled gate last.
  Reuse accepted unchanged-path evidence. Do not invent per-helper/per-commit
  validation cycles or repeated broad gates with adjacent work unfinished.
- Commit ready logical units before a huge dirty tail. Finish MAIN first;
  delegate a single useful side-worktree reconciliation after the batch.
  Copy integrated bytes with fresh mtimes so Cargo cannot silently reuse a
  stale library. Preserve unique pending overlays before any synchronization.
- Set PYTHONDONTWRITEBYTECODE=1; use python3 -B. Bounded rg searches in named
  source/run directories only. Historical .agents/runs is hundreds of GiB;
  never recursively scan, prune or copy it wholesale.
- Keep live tool-session jobs with their owner through actual completion and
  retrieval. No detached/background completion claims or session-ID-only
  ownership transfer. Timeout/interruption remains incomplete, not green.
- No new architecture placement tests: use AGENTS/rules/review. Meaningful
  numerical/runtime/ownership/cancellation tests remain required.

## 9. Verification, documentation preservation and known stale prose

The owning TODOs preserve history and sometimes retain earlier status paragraphs
below later updates. OPEN_WORK includes a discrepancy register rather than
silently deleting/ticking them. In particular:

- NIGHT's inaccessible small-model snapshots, missing x2 trusted key and
  never-rerun independent oracle were true at its source snapshot. Later actual
  access, A3physical/SDK results and strict trust supersede those status claims.
- Earlier timeout-limited full gates remain incomplete; later supervised full
  gates pass on their exact source. Do not reuse obsolete short relay commands.
- Old Pi/Astra role prose, proposed TOML, pre-reuse foreign-code bans and
  planning-only pauses are not current instructions.
- Run-local historical checklist entries may remain open after a later exact
  result. The latest owning plan/evidence controls; never use one stale checkbox
  or the last passing case to flatten a whole batch.
- Old register/playbook/traceability root paths moved under plans/archive/history.
  TODO-SPEC no longer exists. Existing broken sourcebook/retired-worktree links
  are identified, not fabricated or aliased back into existence.
- All264original document/supplement files and the245/360historical indexes are
  unchanged by this documentation task. The source catalog and snapshots preserve
  their bytes; new authored links/coverage are checked separately from inherited
  historical link defects.
- This handoff reuses the accepted latest source gate because no runtime/source
  implementation changes. Documentation coverage/link/byte checks do not add
  policy tests or establish fresh inference support.

The new documents contain complete route/index/data, not a claim that all past
experiments were successful or all missing original history was recovered.
No raw model/tensor archive, private key/credential or giant event stream is
copied into Git. Exact large evidence remains under named original run paths.

## 10. Starting prompt and expected successor report

Pass the full [starting prompt](NEXT_AGENT_PROMPT_2026-10-26.md) to the next
agent. It names this entrypoint and companions, current architecture/correctness
sequence, independent checks and safety boundaries.

The successor reports concrete working behavior/changes, exact source commits,
meaningful validation and physical limits, unresolved blockers and the next
larger executable unit. No model support, speedup, asynchronous completion,
permission or historical recovery is inferred from documentation or test counts.
