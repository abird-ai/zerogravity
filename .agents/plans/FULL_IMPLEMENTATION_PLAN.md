# ZeroGravity — implementation plan

Author: Project owner
Co-authors: GPT-6.2
Owner: @pvl
Decision: 24 Nov 2026

**One execution fabric. Native models first. Mature operators wherever possible.
Compiled execution where it earns its place.**

This is the current cross-plan implementation order. It supersedes older
iteration ordering, not the mathematics, source/lifetime requirements or
accepted results in the owning plans. [Plan queue](TODO.md) remains the
ownership/status index; its linked TODOs retain the detailed open and completed
work. Do not recreate their histories here or treat a design as qualification.
Read [current handoff](../docs/CURRENT_HANDOFF.md) and
[current baseline](../docs/CURRENT_BASELINE.md) before implementation.

**README convention:** describe the intended engineering architecture directly.
Keep implementation status, pending integrations, qualification limits and
delivery dates in the roadmap and owning plans, not in architectural descriptions
or diagrams. Concrete usage examples use supported paths; performance and
availability claims belong to measured status.

## Delivery order

| Phase | Objective | Exit |
| --- | --- | --- |
| 1. Foundations | Complete the shared execution, provider, source, state, memory and control contracts | Real native consumers on a coherent common foundation; optional compiled-region seam ready |
| 2. CLI, TUI and Web | One usable product and control surface, then OpenAI-compatible serving | Native terminal and embedded Web clients use the same SDK/runtime; public serving and benchmark paths ready |
| 3. ROCm | Match or beat optimized vLLM first on AMD | Matched qualified native source-model serving workloads |
| 4. CUDA | Match or beat optimized vLLM next on NVIDIA | Matched qualified native source-model serving workloads |
| 5. Vulkan | Native backend feature/correctness and performance parity | Real-model execution and measured native Vulkan competitiveness |
| 6. Compiled providers | Optional MIGraphX, TensorRT, IREE and ONNX Runtime | Real supported models or closed regions, measured benefit, original ownership |

The order is deliberate. Finish the common foundations before frontend work;
finish CLI/TUI/Web before the performance campaigns. Finish the ROCm campaign
before the CUDA campaign, and then Vulkan. Graph/compiler scaffolding belongs
in Phase 1; concrete third-party graph compiler integrations belong in Phase 6.
Native command capture, native operator libraries and native optimization
remain part of their respective backend phases. Independent bounded correctness
or compatibility fixes may proceed when needed to keep accepted paths usable;
do not start a competing performance campaign or revive an old implementation
hold merely because a specification is linked here.

### Model and source priorities

Transformers, hybrid GDN/recurrent models, MoE, diffusion and audio models
describe **model architectures**. Safetensors, GGUF and ONNX describe different
**artifact representations**; Hugging Face is a distribution/source ecosystem.
Native model support is not contingent on exporting to ONNX.

- Primary: original authenticated Hugging Face checkpoints, particularly indexed
  Safetensors with source config, tokenizer, processor, learned roles and exact
  tensor axes. BF16/F16/F32 and explicitly supported quantized formats.
- Primary: GGUF, with its own verified metadata/codec/quantization semantics and
  matching config/tokenizer where required. Do not silently convert a checkpoint
  or assume a quantized model is numerically equivalent to an original model.
- First performance family: original Qwen3.5 0.8B/2B and Qwen3.8 27B. Use the
  small models for fast feedback and the large model for residency/throughput.
  Extend to source-qualified Qwen3.5+/3.8, Gemma4 (E2B, MoE and dense variants),
  Qwen Flash/Next and other dense/hybrid/MoE families through the same contracts.
  Keep Qwen image/audio, Whisper, Parakeet, TTS and other media pipelines in
  the model expansion queue. Other families (DeepSeek, GLM, Mistral, Llama,
  etc.) are source- and task-qualified additions, not an implicit support list.
- ONNX, exported graph formats, compiled engines and imported executables:
  optional interoperability/acceleration after the native backend campaigns.
  No mandatory graph conversion, Python model runtime or ONNX dependency for
  native Hugging Face/GGUF inference.

### Current starting point

At source checkpoint `aabc3d4f` the common dense geometry
(`Gemm`, `UnitLowerSolve`) and borrowed native command seam are implemented.
The original exact HIP source path remains the default. Explicit native HIP
V2 uses owned rocBLAS BF16 projections, FP32 batched GDN products and triangular
solve under the same resident state/actor/completion owner. Actual X2 47+3
serving gives 0.8B C8 43.638 tok/s, 2B C8 30.949 tok/s and 27B C1 2.228 tok/s.
The last two small-model C8 vLLM reference values are 219.612/140.847 tok/s
(retained comparisons, not fresh paired results). CUDA has substantial native
and paged/cohort execution, but deterministic fused-lane correctness remains
open and current physical performance must be revalidated. Vulkan has bounded
operator evidence, not whole-model parity. The assembled source-preparation
checkpoint now passes the full workspace gate, including CPU/cache/serving,
optional acquisition, release/installer and isolation checks. The pinned X2 vLLM closure is currently unavailable;
restore the approved local environment before claiming fresh parity.
See CURRENT_BASELINE for measured scope; none of these observations grants
arbitrary model, hardware or numerical support.

## 1. Foundations

Accepted foundation slice: `8c34c472` makes source-owned preparation generic
over actual borrowed/window-backed source owners and routes production Qwen
HIP/CUDA through the shared whole-program seam. Physical X2 source/replay/
continuation and retirement checks pass. The complete assembled
`scripts/check.sh` gate is green under mandatory isolation. Acceptance fixes
preserve typed physical-resource refusals, configured limits, ownership and
rollback; empty model declarations refuse before backend selection. No
additional physical CUDA qualification was claimed for that earlier slice.
Accepted next foundation slice: `82f895da` implements bounded stateless
closed-region shape/binding/preparation on the same plan and typed source,
with CUDA projection ownership and HIP admission consumers. Focused
tests/Clippy, the full isolated workspace gate, X2 native/exact/continuation
and L5 CUDA C1 physical canaries pass with original resource retirement.
Standalone partial execution, stateful region effects/bridge and full
capability/lifecycle scaffolding remain open. Continue those foundations
through existing physical owners. Accepted `0db60ede` adds prepared
physical region ports (selected provider/mode, explicit row band, per-port
encoding and checked byte layout), native CUDA/HIP consumers and original
lease-view and invocation-issuer checks. Full isolated gate and physical
Qwen 0.8B native/exact/continuation on X2 and CUDA C1 on L5 pass.
Accepted `77e8b4ae` adds incremental provider-local workspace and
opaque device requirements, original-flight resource binding and captured
CUDA stream/retirement ownership. Shared context/module accounting stays
with the original owners. Physical native/exact/continuation and captured
graph lifecycle checks and the full isolated workspace gate pass.
Accepted `262712f1` derives stateful region ports and the original state
writer/read chain, reuses native CUDA/HIP state plane preflight/binding,
and binds same-layout provisional state to the exact forward attempt.
No second publisher or state allocator. Focused and physical native,
continuation and CUDA contiguous/paged state checks and the full isolated
workspace gate pass. Independent
partial execution and general state conversion remain open. Accepted
`6eae81c6` binds managed private-buffer transfer and same-context CUDA
secondary-stream completion to original physical flights; no implicit copy,
host wait, allocation or publication. Physical certified GEMM/copy/interrupt,
event lifecycle, native CUDA/HIP canaries and full isolated gate pass.
Accepted `7e86aad7` closes contiguous text token/position and producer
observation ports, phase-explicit borrowed call/state binding and original
whole-text preparation for HIP/CUDA singleton/cohort calls. Observation
support is explicit (HIP at producer, CUDA refusal before submission).
Physical Qwen 0.8B observation/refusal, native/exact/continuation and
CUDA contiguous/paged canaries and full isolated gate pass.
Accepted `f4746c23` reconciles explicit native provider/device selection,
shared headroom fit, pre-Client HIP admission and actual effective native
preparation reporting through the original SDK owner. HIP pressure
refusal/retry and native/exact canaries, L5 CUDA singleton/paged/contiguous
and original retirement pass; full isolated gate passes. CUDA still
clamps/adopts the actual envelope on the Client actor; Vulkan guarded
preadmission is compile checked, not newly physically qualified.
Source `057683d1` (test-only correction `139bdbdf`) shares CPU/GPU local
GGUF held-descriptor host admission and canonical-path alias ownership,
validates complete alias declarations also on HF, and implements exact
sole-owner catalog unload/drain/join and cold reload through the original
lifecycle. CPU SDK/HTTP unload/reload, X2 native HIP C1/C8 and L5 native
CUDA C1 plus ordinary physical GGUF unload pass with retained-reader
checks. Combined CPU/CUDA/HIP all-target compile passes. Full isolated
gate: 2388 passed, 0 failed, 101 explicit skips; release/installer/scripts pass.
Signed `2b6656b3` implements selective retirement of independent prepared
CPU/ordinary GPU owners using the same registry, request-lifetime readers,
workers and lifecycle. Existing accepted work drains; unrelated owners
continue. Ordinary GGUF GPU cleanup uses the original physical queue.
Real CPU GGUF SDK/HTTP, L5 two-owner CUDA GGUF selective unload and native
CUDA C1, X2 native HIP V2 C1/C8 pass with original retirement.
Full isolated gate passes (2388 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass). Combined CPU/CUDA/HIP all-target compile passes. Optional adapters without an independent
retirement hook still refuse mixed-owner unload; no live insertion,
replacement or backend context reinitialization.
Signed `bffad06a` exposes typed runtime/catalog snapshots from actual loaded
owners, effective native preparation, scheduler and physical arbiter.
Retirement states and exact alias groups reuse the lifecycle; resource
observations do not retain executables or add per-token telemetry. Legacy
HTTP diagnostics remain with additive typed catalog/scheduler fields.
CPU SDK/HTTP, CUDA mixed-owner GGUF and contiguous/paged native C8, HIP V2
C1/C8 and original retirement pass. Full isolated gate:
2389 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed `4477f20c` binds the existing serial language graph to explicitly
selected loaded registrations and executes through the original runtime.
Committed intermediate text moves into the next original admitted request;
only the final stream/usage is exposed. Stale plans refuse, current request
cancellation and selective retirement are preserved, future stages are not
pre-admitted. CPU fixture SDK/HTTP, CUDA GGUF distinct-owner chain/retirement,
native CUDA C1 and HIP V2 C1/C8 plus alias chain pass. Full isolated gate:
2393 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed `2df8cbb3` binds scoped prefix lookup to the exact prepared resident
state identity and snapshot owner, replaces numeric placeholders with typed
key equality, shares complete-domain restore/fork validation and original
publication, and charges retained cache/snapshot backing. State-only reuse
leaves final-token feedback work. L5 physical fork/restore, B1 contiguous/
paged exact, scoped reuse/exact-length replay/eviction/HTTP and X2 native
HIP V2 C1/C8 plus alias chain pass. Full isolated gate: 2393 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed `7b36a879` shares original authenticated matrix-window validation
among resident packing/matrices and source pages, uses the existing owned
CUDA transfer for cold packing, and merges singleton/cohort projection
transport while retaining direct resident access and numerical selection.
`0342e103` fixes CUDA-only device selection; `fdfb467f` compares physical
resident/streamed fits in fresh admitted contexts. L5 B1 full logits/all42,
native B4 token/score/all42, streamed cancellation/retry, ordinary CUDA C1
and X2 native HIP V2 C1/C8 plus alias chain pass. Full isolated gate:
2393 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed 687d9a24/06df027f/43928332/f4e09428 consolidate authenticated parameter
materialization and original initialization-owned transient backing. CUDA
retains sealed descriptor/lease authority without raw/conversion staging;
HIP keeps its aligned F32 layout and the same shared materializer. Use the
original logical Write with shared Bytes, including padded physical views;
the native full-allocation matrix/page primitive remains strict and reusable.
Artifact 30/30, conversion 3/3, real CUDA resident/streamed B1 logits/all42,
native B4 token/score/all42, cancel/discard/retry, actual host staging
retirement and native HIP V2 C1/C8 replay/retirement pass.
X2 HIP portable typed slice 1024/2048 matches existing CPU/PyTorch
references and original retirement (test-only typed refusal correction).
Full isolated gate: 2393 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed 3ca16abd/a53a3446 close the bounded SDK-free compiled-provider
scaffold: RegionSource binds original encoded references or verified
window descriptors; ordered ModelRegion operators borrow original nodes.
Explicit Compiled(provider) is distinct from native Graph capture;
prepared selection must match cold selection and exact source/plan/scope.
Combined call/physical-shape binding retains existing state and completion
owners. Engine preparation 9/9, tensor windows 30/30, strict Clippy,
combined SDK check and actual Qwen 0.8B CUDA C1/HIP V2 C1/C8 replay and
retirement pass. No fresh captured or legacy GGUF physical qualification;
capture code unchanged. No external compiler execution or new IR.
Full isolated gate: 2394 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed 8eb17bc2 separates authenticated Qwen HF source from task/arithmetic
selection, makes approved snapshot metadata backend-independent, shares HF
cold registration and GGUF format/family/geometry checks over held descriptors.
Original request variants, source authority, numerical/device policy and
registry retained. Real CPU 0.8B/2B cached/direct and HF-import comparisons
match full vocabulary/all42 and source retirement (test-only 00f57e7a/9374e0ca).
CUDA HF C1/GGUF pipeline+unload and HIP V2 C1/C8 pass. No new performance claim.
Full isolated gate: 2396 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed 9ae6dd1e/e06213a5: ordinary Linux CUDA and pinned source preparation
reuse the original HIP/Vulkan preadmission owner; exact context-free PCI and
configured provider charge precede Client. Original actor adopts the same
lease, live downward clamp releases unused credit before model admission;
source-shaped fit and compiler lifetime preserved. Real HF peer refusal
(no new observed compute owner)/retry, pinned live-fit/retirement, CUDA HF C1,
same-process contiguous/paged native C8 exact tokens and context reuse,
GGUF pipeline/selective unload and HIP V2 C1/C8 pass. Logical admission is
not an OS-wide VRAM guarantee; live observation is post-bootstrap and
configured-cap conflicts may refuse conservatively.
Full isolated gate: 2396 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed cbd2c119/197c2e26: shared aggregate host backing over original scopes,
immutable first ceiling plus latest known cgroup hard constraint, common
cold host fit and actual driver/local VRAM intersection on qualified HIP.
Keep lifetime source-shaped logical upper separate from incremental live
checks; initial over-refusal corrected to actual context reservation.
CUDA HF peer refusal/retry/C1 and GGUF pipeline+unload; HIP V2 C1/C8 and
32k capacity refusal before native context pass. Fresh release real GGUF CPU
direct run passes. Full isolated gate: 2394 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
No additive UMA/VRAM/GTT
or hard OS memory claim.
Signed a6548863: ordinary Linux HIP pre-Client physical envelope adoption,
same shared driver/local VRAM fit; guarded Vulkan exact adapter/local fit
before original registration. Same declared physical-domain placement has
no inter-domain transfer plan; other domains explicitly refuse. Actual
portable shared HIP and Vulkan healthy/unknown and peer refusal/retry pass.
Real HF HIP V2 C1/C8 replay and retirement pass. Full isolated gate: 2394 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Generic
named Vulkan repeated initialization needs original registry support;
non-Linux and actual host-visible GPU backing remain open.
Signed 1d5c4ada: optional production native GGUF CPU committed persistence
(state-checkpoint). Actual immutable source/geometry/numerical compatibility,
original v1 numerical body plus separate v2 SHA256/extent/private atomic IO,
original host admission retained by imported state readers, local replacement
through SessionAuthority. Actual 0.8B/2B fresh-allocation restore and full
vocabulary exact continuation, corrupt/wrong-model/budget refusal and
stale-attempt/last-reader behavior pass; original diagnostic v1 wire and
real domain bridge preserved. Full isolated gate: 2395 executions, zero
failures, 102 explicit skips; release/installer/scripts pass. No sampler/prompt-prefix/authentication or
provider-conversion permit.
Signed 804d4882/d9b25205: optional committed native HIP V2 contiguous F32
persistence through original source-owned semantic revision, prepared
identity, complete committed snapshot, bounded SHA256/atomic IO and
admitted native plane uploads/root pool. Actual HF 0.8B same/fresh-process
exact full-vocab continuation/all-domain replay, corrupt/stale refusal,
known retirement and actual unknown H2D retention/fence pass; HF V2 C1/C8
replay/retirement with persistence enabled pass. Full isolated gate: 2395 Rust test executions, zero failures, 102 explicit skips; release/installer/scripts pass.
Other GPU formats/contexts/profile conversion remain unsupported.
Signed ebda04c4: bounded source-declared language preparation with explicit
execution policy, selected graph models only, actual local source geometry
or exact single approved HF snapshot, original loader/alias ownership.
Admit exact plan and router together before Ready; failed binding/pressure
rolls back original router. Actual CPU 0.8B/2B mismatch/retry and chain,
CUDA GGUF three stages/two owners/retirement, native HIP V2 HF alias chain
and C1/C8 replay/retirement pass. Full isolated gate: 2396 Rust test executions, zero failures, 104 explicit skips; release/installer/scripts pass.
Signed b25b5c02: common borrowed model/prepared-pipeline TextTask,
explicit controls, original bounded receiver poll/cancel, selected exact
registration/config observations and structured service refusal. Actual CPU
0.8B/2B chain + direct framed nonstream task, CUDA GGUF three-stage/two-owner
chain, HIP V2 HF same-owner chain/C1/C8 final token observations and retirement
pass. Legacy APIs and original scheduler/transport retained. Full isolated gate: 2396 Rust test executions, zero failures, 104 explicit skips; release/installer/scripts pass.
Signed 7987f70a: common selected GGUF local/cache location, held
descriptor and family/target/context checks through direct CPU and
SDK/pipeline; TextLoadPolicy and original catalog/graph lifecycle,
exact cache digest and verified raw lease (no duplicate charge).
Actual cached 0.8B CPU/CUDA SDK/reload/chain, ref retarget refusal and
earlier selected direct materialization; existing local CPU/CUDA GGUF
and native HF HIP C1/C8 pass. 2ec06026 preserves selector refusal diagnostics.
Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
Signed 8e3b8993/33118f16: original recorded and retained selected region
observations, actual physical/logical ports/state/requirements and exact
native CUDA projection owner. Cold explicit same-context queue/resources
and identical dense view compatibility; original completion/retention,
shared actual GEMM orientation. Actual CUDA resident/streamed B1/C4
full-domain/full-logit equality, GGUF chain, native HIP V2 C1/C8 and
physical private bridge interruption/refusal pass. Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
Signed 3b5a520d/fc38db03/e1decff5: original CubeCL registry noncreating
lookup/atomic fallible factory and exact Client; ordinary/guarded Linux
Vulkan selected PCI/current local fit and original lease. Known reuse
does not arm new bootstrap. Actual RADV peer refusal, raced/warm service
reuse and independent envelope retirement, portable healthy/unknown and
HIP healthy and real HF HIP V2 C1/C8 replay/retirement pass. Locked
common/runtime sources; no dependency version
change. Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
Next reconcile actual captured-owner observations and minimum
Phase 1 exit requirements; keep general UMA aliasing, foreign buffers/
streams, non-Linux and unqualified backend/family/codec/task explicit.
No duplicate catalog, estimator, allocator or runtime.
General DAG/media/context authority, multi-source HF and mixed-backend
pipeline loading, CLI/API transport activation, generalized persistent
session storage and cross-provider restore remain unsupported.
None of these slices closes Phase 1.

Complete the engineering contracts and their real consumers, including every
foundation below. This is not a requirement to finish every backend optimization,
model family or optional provider before UX. Close the minimum useful shared
implementation of each seam; defer concrete specializations to their phases.

### 1.1 Semantic model and execution composition

- [ ] Reconcile model/source/format/codec/task registration and crate boundaries
  against the live Fabric transition. Keep family-specific mathematics in
  model crates, shared mathematical operators in core/shared libraries, and
  physical execution in backends. Remove only obsolete wiring; retain useful
  correct primitives and numerical variants.
- [ ] Preserve one authoritative source-bound `ModelProgram` / `ForwardPlan`
  and prepared physical executable. Typed values, state effects, layouts,
  regions and task dependencies determine valid execution. Distinguish a
  mathematical region, a native ModelAssembly and a multi-model TaskPipeline.
  Recursive composition is a semantic/planning view, not a recursive second
  runtime.
- [ ] Complete the existing `RegionExecutionProvider` / `PreparedExecutable`
  / `ExecutableBundle` seam for closed regions and whole-model regions.
  Preserve the original model/source identity, numerical contract, shape
  constraints, input/output bindings and state effects. No arbitrary
  subgraph replacement that bypasses the authoritative plan.
- [ ] Separate cold policy from physical capability: explicit backend,
  implementation/provider identity, numerical/approximation profile,
  execution mode and resolved region choices. Select a qualified executable
  at preparation or an explicitly supported committed-root boundary. Shape
  misses use a qualified pre-enqueue decomposition or refuse. Submitted
  failures cannot be retried as another provider on the same root.

### 1.2 Native and compiled provider architecture

![Execution architecture](../../docs/architecture-overview.svg)

Keep four concepts separate:

| Layer | Responsibility |
| --- | --- |
| Execution fabric | Serving, batching, scheduling, lifecycle, memory, persistent state, pipelines, placement and publication |
| Model semantics and planning | Verified source, typed mathematical/task regions, numerical policy, preparation and bindings |
| Execution implementations | Native composed operators, captured command graphs, compiled regions or whole-model executables |
| Device backends | CPU/CUDA/HIP/Vulkan and future device/peer capabilities, physical storage and completion |

Operator libraries (rocBLAS, hipBLASLt, cuBLAS/Lt, CUTLASS, MIOpen, cuDNN,
FlashAttention, FlashInfer, cooperative-matrix shaders) implement math inside
native execution. MIGraphX, TensorRT, IREE and ONNX Runtime can realize complete
computations at the region boundary, often using the same native libraries.
A graph compiler is not a device backend. Capturing a command sequence is
not graph optimization, and both can coexist.

Keep ingestion, semantics and executable lowering independent:

| Representation | Responsibility |
| --- | --- |
| Source artifact | Verified weights, metadata, tensor encodings, configuration and model provenance |
| Semantic program | Native model definitions, ModelProgram / ForwardPlan, typed regions and state effects |
| Prepared executable | Native commands, captured graphs or provider-compiled regions with admitted bindings and lifetimes |

Source artifacts are inputs, not a third execution IR. Provider-specific graphs
and compiler IRs are internal lowering representations, not additional canonical
model IRs. Compiled artifacts retain original source and numerical identity.
The [closed-region design](pipeline-composition/REGIONS_AND_PROFILES.md) defines
this boundary. Direct provider lowering from native regions and optional graph
import (e.g. ONNX) are independent capabilities. Neither ONNX conversion nor
a Python runtime is mandatory.

- [ ] Generalize only proven shared mathematical contracts: keep the dense
  seam and add bounded native row-wise, normalization/gating, rotary, attention,
  recurrent, convolution and quantized operators as real consumers require.
  Keep compact row/batch layouts and explicit precision; avoid forcing every
  operator into GEMM or a universal dispatch object.
- [x] Complete the bounded optional compiled-provider **SDK-free scaffolding** (3ca16abd/a53a3446): a closed
  semantic region or whole model with explicit source, typed bindings,
  layout/precision/shape/shape-miss, state effects and prepared executable
  ownership. Accept the authoritative native region and verified source
  bindings directly, without requiring graph export/import. A provider reports
  supported geometry, actual workspace and persistent-resource needs,
  external-buffer/stream compatibility, preparation/retirement and completion
  behavior. Reuse original physical owners. No extra canonical IR, model
  conversion pipeline, general compiler framework or independent ONNX
  subsystem in Phase 1. Prove the boundary with native/captured consumers
  and a minimal graph-only preparation/binding test as needed.
- [ ] Make borrowed device views and external execution/completion integration
  possible without promising universal zero-copy interop. A provider may
  plan/reuse local intermediate storage inside its admitted region. When it
  cannot import caller buffers or use the original stream, admit its internal
  memory and actual synchronization/transfers explicitly. No provider may
  publish StateRoot, schedule independent sessions or hide unaccounted memory.
- [ ] Keep provider features independently optional. Native execution and all
  primary model loading remain usable with no graph compiler installed.
  Concrete MIGraphX/TensorRT/IREE/ONNX Runtime build, runtime bindings and
  model/region support are Phase 6 work, not foundation exit requirements.

### 1.3 Immutable source and model lifecycle

- [ ] Complete shared source acquisition/import/cache, verified shard/index
  and tensor-window identities, license/authorization, exact snapshot
  selection, config/tokenizer/processor/adapter validation and finite
  encoding/codec support. Preserve HF/Xet, GGUF, local and existing remote
  acquisition mechanisms; unsupported providers/format variants refuse
  explicitly. Native weights load without ONNX conversion or remote model code.
- [ ] Preserve one catalog/SDK source-lifecycle authority, transactional cache
  admission, cancellation, download resume, last-reader retirement and safe
  cleanup. Retain source provenance, checked metadata and byte bounds;
  avoid duplicate allocations for aliases and components. Make actual file
  descriptors, mappings and immutable readers the authority.
- [ ] Consolidate pre-Client fit/admission, stable device identity and source
  eligibility. Derive complete cold/prefill/decode/graph/provider/state/
  output cost and headroom from real geometry/physical observations; keep
  UMA overlap and host limits explicit. A typed capacity miss is distinct
  from a lifecycle error. No silent CPU/backend/profile substitution after
  work is submitted.

### 1.4 State, memory and scheduling

- [ ] Close shared physical backing/range identity, original leases,
  immutable source readers, actual readiness and event-backed completion/
  quarantine. One ResourceArbiter; optional local allocator strategies and
  provider workspaces may vary under its authority. No global reset or
  reclamation of unrelated owners.
- [ ] Complete the common independent session-state and candidate-publication
  contract across KV, recurrent/conv and all required state domains.
  Paged/hybrid state, complete-domain prefix reuse and COW, bounded snapshots,
  cancellation/recovery and cache compatibility use the same StateRoot.
  Only actual completion and the original authority publish; unknown work
  retains all affected owners and fences the unsafe service.
- [ ] Complete shared resident/paged/streamed tensor materialization and
  transfer-retirement boundaries. Preserve resident direct access with no
  mandatory host dispatch; actual SSD windows, page/weight cache, prefetch,
  bounded async IO and eviction policies mature in later native workloads.
- [ ] Finish model-neutral invocation/cohort, fair worker, batching/refill,
  backpressure, cancellation and task/pipeline control seams. Continuous
  batching, chunked/ragged prefill, speculation/MTP and multi-model placement
  share these owners. Do not grant unqualified execution modes from policy.
- [ ] Preserve typed media ports/clocks, realtime lanes and remote placement/
  transport boundaries without implementing every media codec or network
  topology now. Local single-process execution requires no network.

### 1.5 Library, build, verification and observability

- [ ] Complete public Rust SDK and standalone serving library contracts;
  CLI/TUI/Web use the same runtime, catalog, configuration and control
  events, not a separate frontend-owned execution path. Keep model and
  backend features orthogonal, ordinary `all` free of diagnostic features,
  backend-only/model-only/serve-only dependency isolation and mixed GPU
  development shell. Integrate provider capabilities without requiring
  optional SDKs in default CPU builds.
- [x] Resolve the historical CPU cache-admission assertion (00f57e7a,
  9374e0ca): typed host capacity refusal and original host test key;
  real cache comparisons and subsequent full gates pass.
- [ ] Close remaining applicable architecture/feature/test-boundary debt.
  Keep source, model arithmetic,
  quality, cancellation, physical readiness, state/root and real execution
  checks distinct. Qualify the existing working native paths; CUDA unresolved
  parity is an explicit gate before CUDA performance, not a pretext for
  starting its optimization campaign ahead of ROCm.
- [ ] Expose a common effective configuration, capabilities, resource
  snapshot, plan/region identity, structured error and execution metrics
  contract. Keep telemetry on demand, bounded and out of hot-path ownership
  decisions where possible. The UI builds on this rather than deriving
  semantics from logs.

**Exit:** one source-authenticated native run via public SDK/CLI; real supported
backend/provider paths use the common mathematical and physical contracts;
closed-region prepare/bind/retire seam accepts native semantic regions and
verified source bindings independently of artifact format, with explicit
unsupported capabilities; native paths require no ONNX or graph SDK; no new
ownership plane; feature and applicable full workspace
gates pass. Document remaining model/device/quality holds, not implied
support. Detailed owners: [composition](pipeline-composition/TODO.md),
[regions](pipeline-composition/REGIONS_AND_PROFILES.md),
[profile/reuse policy](pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md),
[Fabric transition](fabric-vnext-transition/README.md),
[model execution](model-execution/TODO.md), [memory](memory-and-streaming/TODO.md),
[acquisition](model-acquisition/TODO.md), [numerics](numerics-and-context/TODO.md)
and [platform](platform-and-release/TODO.md).

## 2. CLI, TUI and Web

Make ZeroGravity one small, coherent tool: minimal by default and transparent on
demand. Preserve the existing detailed
[CLI/model lifecycle/terminal UX specification](cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md)
(including UX-1 through UX-10 and §46A).

### 2.1 Shared SDK, control plane and CLI

- [ ] Close typed command tree, ConfigDocument/profile merge and shared
  presentation/action/snapshot/event model. One resolver for local paths,
  model names, explicit source locators, backend aliases and canonical
  source identity. CLI overrides selected config overrides defaults;
  no implicit remote/download/approximation authority.
- [ ] Complete model/download/cache lifecycle, progress, pause/resume/cancel,
  device discovery, headroom, qualified auto-selection, one-line run/serve,
  config/profile management, pipeline inspection and doctor/explain.
  Explicit backend aliases mean selected backend or factual refusal.
- [ ] Stable scripting output, JSON and plain non-TTY behavior; beautiful,
  responsive terminal mode. Show backend, provider, numerical/quality profile,
  shape and execution plan, supported features and actual selected/refused
  modes. Expose benchmark and diagnostics through the same SDK.

### 2.2 Native TUI, then embedded Web

- [ ] Build native TUI (UX-8A–D): bare interactive command opens a responsive
  shell; Home/Running/Serve, Models/Downloads/Devices, Profiles/Config/Logs.
  Shared events, command palette, keyboard/navigation, cancellation,
  resize/accessibility; no action on mere visual selection.
- [ ] Build embedded SolidJS/Vite Web (UX-9A–E), following the existing AW
  reuse/design direction and same visual vocabulary. Same high-value
  views/actions as TUI; `zg web` and opening Web from TUI reuse original
  runtime/control plane. Keep presentation state local; allow simultaneous
  clients without duplicating owner state.
- [ ] Keep one ConfigDocument and control protocol; secure local control
  auth and browser transport independently of inference auth. No separate
  model/scheduler/cache implementation in a frontend. Complete UX-10
  doctor/pipeline polish, compatibility cleanup and real simultaneous-client
  lifecycle acceptance.

### 2.3 Public serving and performance interface

- [ ] Close the [OpenAI-compatible serving iteration](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md):
  model catalog/router, OpenAI Chat/Responses and supported completion/
  embedding/tool/structured-output contracts, usage, errors, JSON/SSE,
  streaming/cancellation/timeout/backpressure and real concurrent requests.
  Unsupported tasks/options must refuse rather than simulate compatibility.
- [ ] Preserve a separate control surface and inference API with typed
  configuration, explicit route/tenant scope, model lifecycle and shared
  scheduling. Add future WebSocket/WebRTC/media endpoints through the
  existing task contracts; do not wait on unimplemented media models.
- [ ] Establish ordinary public CLI/SDK/server benchmark invocation,
  effective config and end-to-end metrics, not a qualification-only serving
  process. Retain one process/multiple model instances and pipeline routing.

**Exit:** ordinary one-line run/serve; coherent model/device/config lifecycle;
both TUI and Web use one control model and real owner events; public serving
works for qualified model/task/backend paths; API behavior and actual
concurrency can be benchmarked through normal product surfaces.
Owners: [CLI/device routing](cli-device-routing/TODO.md),
[UX design](cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md),
[serving](serving-and-speculation/TODO.md),
[platform](platform-and-release/TODO.md).

## 3. ROCm — match vLLM first

Use X2 (gfx1151) as the first measured performance target, then other qualified
AMD hardware. Source authoring on L5; execution/testing on X2/A1 per the
handoff. Observe shared capacity and retire only our own allocations.

- [ ] Restore the approved pinned optimized vLLM provider and establish
  fresh matched original-source 0.8B/2B/27B baselines. Use retained V2
  results for engineering direction, not fresh comparisons. Get quick real
  C1/C8 feedback; extend the declared campaign after working results.
- [ ] Native row-wise and attention regions: eliminate unnecessary intermediate
  materialization and exact-reference scalar passes within explicitly
  qualified native arithmetic. Compose complete normalization/residual,
  gating, QK/rotary, attention and recurrent regions; reuse mature native
  kernels/providers where capability and model geometry fit. Keep GDN
  solve/products and source/reference implementations.
- [ ] Use rocBLAS/hipBLASLt for dense products, appropriate CK, MIOpen and
  mature attention/flash-decode kernels, model-specific kernels where needed.
  Select measured GEMV/GEMM, quantized and batch shapes, not one blanket
  provider choice. Native command capture and larger execution regions
  amortize host/actor dispatch, without requiring MIGraphX.
- [ ] Improve genuine shared prefill, independent stateful cohort work,
  continuous ragged refill/regroup, paged/hybrid KV and prefix/COW.
  Device-resident sampling/feedback, overlap, fair scheduling, native
  graph execution where material. Qualify cancellation and state effects.
- [ ] Scale native resident and streamed weights, long context, real
  supported low-bit GGUF/source quantization, KV formats and SSD execution.
  Keep exact/source/native profiles separate; quantify quality against
  reference and compare equivalent precision/workload. Extend family support
  through existing registries and source bindings, not separate model loops.
- [ ] Close the matched performance acceptance below on the required feasible
  AMD cells. Do not wait until the entire grid is green to measure and fix
  the dominant gaps. No actual graph compiler integration in this phase.

**Exit:** match or exceed optimized matched vLLM on the declared feasible
ROCm original model/serving campaign while preserving accepted quality,
memory/state/cancellation semantics. Report missing cells and refusals
honestly; no claim based on synthetic GEMM or incomparable eager baselines.
Owners: [execution](model-execution/TODO.md), [backend parity](cpu-gpu-optimization/TODO.md),
[batching/state](serving-and-speculation/BATCHING_AND_PAGED_STATE.md),
[serving](serving-and-speculation/TODO.md), [memory](memory-and-streaming/TODO.md),
[numerics](numerics-and-context/TODO.md).

## 4. CUDA — match vLLM next

- [ ] Revalidate actual current NVIDIA source paths before performance claims.
  Close deterministic C4/B8 identical-lane divergence, independent populated
  all-state-domain correctness and missing approved active-context/2B
  references. Preserve prior physical acceptance only within its scope.
- [ ] Promote the existing owned cuBLAS/cuBLASLt dense adapter into ordinary
  execution through shared geometry and original physical owners. Use native
  tensor-core/CUTLASS and FlashAttention/FlashInfer where they win; preserve
  existing specialized GDN/state and transport alternatives.
- [ ] Reuse ROCm phase's mathematical regions and native execution improvements;
  tune CUDA-specific attention/paging, continuous batching, compact
  device-resident sampling, larger captured graphs and dispatch amortization
  only from actual measurements. Keep pre-enqueue qualified misses distinct
  from unknown native submissions. Production selection must be physically
  accepted, not inferred from diagnostic graph/timing support.
- [ ] Run matched optimized vLLM through the same public serving campaign on
  approved NVIDIA devices and feasible models/context/concurrency. Extend
  source/GGUF model coverage, native quantization and residency according to
  measured capacity. TensorRT remains optional Phase 6 work.

**Exit:** same matched correctness, serving and performance criteria as ROCm.
Owners: [execution](model-execution/TODO.md),
[backend parity](cpu-gpu-optimization/NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md),
[numerics](numerics-and-context/TODO.md), [serving](serving-and-speculation/TODO.md).

## 5. Vulkan — native parity and portability

- [ ] Reuse shared typed operations, region/shape contracts and device
  lifetime semantics. Close actual Vulkan model execution, independent
  recurrent/KV state, prefill/decode, cohort and resource safety, then
  required CPU/CUDA/HIP-compatible capabilities where available.
- [ ] Evaluate mature ggml Vulkan cooperative-matrix GEMM implementations,
  capability-gated VK_KHR_cooperative_matrix and vendor variants, portable
  compute shaders for unsupported geometries, and applicable optimized
  quantized/attention kernels. Audit license/provenance and actual backend
  layout compatibility before reuse. No fictional universal Vulkan BLAS
  and no mandatory compute-engine fork.
- [ ] Compare real model latency, throughput, memory and quality against
  native Vulkan baselines (e.g. llama.cpp/ggml) on the same device/format;
  use matching native/vLLM hardware comparisons only where meaningful.
  No unsupported vLLM Vulkan baseline. Extend Linux and other targets by
  actual capability and package qualification; retain clean backend feature
  isolation. IREE integration is Phase 6.

**Exit:** practical native whole-model Vulkan execution, source/format parity
for the chosen supported families, measured competitiveness with the
appropriate Vulkan baseline and explicit unavailable capabilities.
Owners: [backend parity](cpu-gpu-optimization/TODO.md),
[platform](platform-and-release/TODO.md), [execution](model-execution/TODO.md).

## 6. Optional compiled execution providers — after native parity

The foundation already exposes the region boundary; now implement real
optional provider integrations. No dependence of native HF/GGUF execution
on graph export. MIGraphX and TensorRT are planned concrete integrations
on their native backends. IREE and ONNX Runtime extend portable/graph-model
coverage where useful. Integrate in useful, independently qualified slices;
optional build support is not a reason to skip the actual planned adapters.

Prefer direct programmatic graph/network construction from source-bound
native regions for MIGraphX and TensorRT. Treat ONNX import separately;
a successful import, compilation or export is not numerical or performance
qualification. Explicitly admit any repacking, dequantization, layout
conversion or additional persistent storage required by a selected provider.
Do not silently change the model's numerical contract or source identity.

- [ ] MIGraphX on HIP: lower native semantic regions through programmatic
  graph construction, qualify runtime/hardware and external device arguments/
  stream interop, compile/prepare a closed FFN or other dense region, then
  compare native and compiled paths at actual model/serving level. Consider
  whole encoders/embeddings/ASR where source format and model semantics fit.
- [ ] TensorRT on CUDA: lower native semantic regions through programmatic
  network construction, same closed-region/whole-model boundary, compatible
  caller-bound tensor addresses/context memory/stream, admitted engine
  compilation/cache and explicit supported shape/profile. Benchmark a real
  region or conventional model, not another complete LLM serving engine.
- [ ] IREE: evaluate compiler/HAL interop on supported device targets, in
  particular Vulkan, and compare with the native cooperative-matrix path.
  Verify actual import/export/synchronization before promising zero-copy.
- [ ] Optional graph import: authenticate ONNX graphs and external weights;
  establish their typed region, numerical and effect contracts through the
  existing source/preparation boundary. ONNX Runtime executes supported
  conventional models, encoder/vision/audio/embedding graphs, with device
  I/O binding or explicitly budgeted materialization. MIGraphX/TensorRT
  import support is independently qualified; it is not a prerequisite for
  direct native-region lowering. Other graph accelerators/TVM only when a
  concrete workload justifies them.
- [ ] Native/compiled selection happens at admitted preparation, with source,
  numerical/quality, layout, shapes, device/runtime, actual costs and state
  compatibility explicit. Bound compilation/autotune/artifact caches and
  retain useful alternatives. Require actual external-resource lifetime and
  completion; a failed submission never falls back by hiding its failure.
- [ ] Preserve one engine, state authority, resource fabric and model/pipeline
  interfaces. Whole-model opaque managed remote APIs are TaskService adapters,
  not native kernels or proof of local physical execution.

**Exit:** each planned compiled provider has a real supported consumer
through the original ownership boundary, or an explicit documented capability
blocker/owner-approved deferral. Measure actual consumers against the native
path; optional provider feature, source/license, capability, memory and
lifecycle checks pass. An initial useful integration is a checkpoint, not
completion of every provider. Unavailable providers remain optional.

## Further native breadth and cross-cutting work

Do not discard deferred work while delivering the ordered campaigns. Reconcile
its scheduling at phase boundaries. Native model breadth remains primary;
compiled providers need not precede native expansion. The current icebox and
specific hardware/download/implementation holds remain in force.

- **Model breadth and modalities:** extend the authenticated family registry and
  quantization/codec matrix; CPU/reference and native backends for Gemma4,
  hybrid/MoE Qwen, text/embedding/decision tasks, ASR/TTS, image/video and
  multi-model pipelines. Preserve explicit approximation permissions,
  media clocks, genuine streaming and rollback/barge-in. See
  [multimodal](multimodal/TODO.md), [composition](pipeline-composition/TODO.md)
  and [model execution](model-execution/TODO.md).
- **Memory tiers and topology:** bounded SSD materialization/cache, expert
  residency, state persistence, safe victim/reader retirement; then local
  layer/CPU/GPU/NPU partitioning, multi-GPU and authorized peer execution,
  tensor/context/expert/pipeline parallelism and prefill/decode disaggregation
  where actual transfer costs justify them. See
  [memory](memory-and-streaming/TODO.md),
  [distributed execution](distributed-execution/TODO.md).
- **Serving extensions:** source-qualified speculation/MTP/draft, multi-model
  QoS/tenant isolation, realtime transport, embeddings/tools/structured
  output, external TaskService and observability; one state/publication
  authority. See [serving](serving-and-speculation/TODO.md).
- **Platform/release/security:** feature isolation, actual linkage, Nix and
  target-specific installers, source/license/NOTICE/SBOM, bounded parsers,
  authenticated APIs/private caches and explicit hardware support. Cross-target
  release is not implied by source definitions or compilation. See
  [platform](platform-and-release/TODO.md) and [release gates](RELEASE_ACCEPTANCE_GATES.md).

## Acceptance and engineering cadence

**Performance:** compare the same model/revision, tokenizer/prompt/output
workload, precision or equivalent qualified quality, context, offered load,
hardware, capacity and optimized provider settings. Use actual public execution
and serving, distinguish cold from warm, and report throughput, TTFT/ITL,
latency/goodput, admission, memory and failures. Start with 47+3 C1/C8 and
actual long-decode/context/refill, then the existing C1/4/8/16/32/64 campaign
where feasible, including 8K/16K/32K and longer supported context. Do not
compare a quantized model to an unrelated BF16 baseline as a parity result.
Measure actual wins at the largest useful blocks rather than launch counts,
tiny synthetic fixtures or evidence bookkeeping. A parity claim requires
matched measurements, not retained reference ratios.

**Correctness:** exact/reference and explicitly native/approximate profiles
have different declared arithmetic and quality obligations. Preserve actual
state/root/source identity, replay, lane invariance, independent reference
checks appropriate to the changed math, cancellation, actual completion and
retirement. Keep unsupported/held scopes explicit. Test representative
real-model, ownership and fault paths; do not require a new exhaustive
fixture grid per reusable helper.

**Implementation:** Rust, simple, composable, low-overhead and zero-cost where
practical. Prefer existing optimized libraries and faithful reusable primitives
over rebuilding general-purpose math. Specialize genuine hot regions where
measurements justify maintenance. Keep useful correct alternatives even if
the current selector loses; remove temporary probes and failed consumer
wiring, not reusable kernels. Batch work and validate the assembled slice;
no speculative framework or evidence bureaucracy. Each implementation turn
targets a coherent signed checkpoint within about 30 minutes. Source authoring
on L5; X2/A1 physical runs use the original capacity and isolation rules.
No push without explicit authorization.

## Plan crosswalk

This document sets priority; every detail and open item stays in its owner.
Historical orders in the following designs are subordinate to the delivery
order above, but their valid constraints and decisions are preserved.

| Existing plan/spec | Primary placement |
| --- | --- |
| [Master plan](ZEROGRAVITY_MASTER_PLAN_2026-10-10.md), [P00–P23 roadmap](IMPLEMENTATION_ROADMAP.md), [integrated delivery gates](INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md) | Overall design; Phase 1 foundations, remaining work in applicable phases |
| [Fabric vNext](fabric-vnext/README.md), [transition](fabric-vnext-transition/TODO.md) | Phase 1; accepted migration retained; deferred scopes follow this plan and explicit holds |
| [Pipeline composition](pipeline-composition/TODO.md), [shared execution](pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md) | Phase 1, then native regions and later breadth |
| [Model acquisition](model-acquisition/TODO.md), [immutable shard spec](IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md), [acquisition spec](MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) | Phase 1 and 2 lifecycle; additional sources/models qualified later |
| [Memory/streaming](memory-and-streaming/TODO.md) | Phase 1 common ownership, Phases 3–5 optimization, later topology |
| [CLI/device routing](cli-device-routing/TODO.md), [CLI UX](cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md) | Phase 1 SDK/routing prerequisites, Phase 2 complete UI |
| [Serving/speculation](serving-and-speculation/TODO.md), [OpenAI serving](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md) | Phase 1 contracts, Phase 2 APIs, Phases 3–5 native performance |
| [Model execution](model-execution/TODO.md), [CPU/GPU optimization](cpu-gpu-optimization/TODO.md), [backend parity](cpu-gpu-optimization/BACKEND_PARITY.md) | Phase 1 contracts, Phases 3/4/5 in order, Phase 6 compiled providers |
| [Numerics/context](numerics-and-context/TODO.md), [correctness policy](numerics-and-context/CORRECTNESS_POLICY.md), [Qwen API/CLI frontier](qwen-parity/) | All phases; source/quality correctness precedes respective performance and capability claims |
| [Multimodal](multimodal/TODO.md), [distributed execution](distributed-execution/TODO.md) | Phase 1 boundaries; deferred qualified implementation/expansion |
| [Platform/release](platform-and-release/TODO.md), [traceability](REQUIREMENTS_AND_TEST_TRACEABILITY.md), [experiment program](EXPERIMENT_PROGRAM.md) | Cross-cutting acceptance and later platform breadth |

The dated handoff/archive packs remain the historical record. Do not erase
their detailed pending items or duplicate them into a second task ledger.
