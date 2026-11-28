# ZeroGravity — implementation plan

Author: Project owner
Co-authors: GPT-6.2
Owner: @pvl
Decision: 24 Nov 2026
Scheduling update: 28 Nov 2026 — foundations then parallel backend/adapter lanes

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
| 1. Foundations | Shared execution, provider, source, state, memory and control contracts | Accepted native consumers and optional compiled-region seam; preserve the bounded exit |
| 2. CLI, TUI and Web | One usable product/control surface and public serving | Complete the separately owned UX-11D implementation and required qualification |
| 2.4. Programmable stack foundations | Reusable Service/SDK, typed programming contracts, durable operations and safe live ownership | PS-F1/F2/F3 common foundation exit; later lanes can extend the same contracts independently |
| 3. ROCm — parallel lane | Match or beat optimized vLLM on AMD | Matched qualified native source-model serving workloads |
| 4. CUDA — parallel lane | Match or beat optimized vLLM on NVIDIA | Matched qualified native source-model serving workloads |
| 5. Vulkan | Native backend feature/correctness and performance parity | Real-model execution and measured native Vulkan competitiveness |
| 6. Compiled providers | Optional MIGraphX, TensorRT, IREE and ONNX Runtime | Real supported models or closed regions, measured benefit, original ownership |

**28 Nov user scheduling decision:** finish UX-11D acceptance, then complete
[PS-F1/F2/F3](programmable-service/IMPLEMENTATION_PHASES.md). After that bounded
shared foundation exit, ROCm parity, CUDA parity, SDK-hosted MCP/native
attachment and capability lanes may proceed independently. A backend campaign
does not wait for another backend's parity result or for MCP implementation.
Full composition, optimization, checkpoint expansion and arbitrary-code
isolation are separate consumers, not prerequisites for reopening parity.

This replaces the earlier ROCm-before-CUDA scheduling requirement. Existing
phase numbers identify scopes; phases 3 and 4 are now parallel. Vulkan and
concrete third-party compiler integrations retain later priority. Native
command capture, mature operator libraries and native optimization remain in
their backend lanes. Accepted foundations/UX work and historical measurements
remain accepted within their actual scopes.

Common interface/resource/state changes have one integration owner. Each
lane retains its correctness-before-performance requirements, explicit holds,
authorized sources and physical admission/retirement rules. Shared hardware
or build contention is coordinated scheduling, not a new architectural
dependency between the lanes.

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

**Minimum shared foundation exit accepted — 25 Nov 2026.**
See the bounded exit audit below and CURRENT_HANDOFF for accepted scope.
The broader checkboxes remain the expansion queue; Phase 2 is next.

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
Named Linux Vulkan registry reuse is closed below at 3b5a520d/fc38db03/
e1decff5; bounded actual coherent mapped producer at 14bbbd48.
Non-Linux and general host-visible GPU/UMA backing remain open.
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
Signed ce83cc18: actual original captured profile/source/stream/receipt,
cache/pin/service observation and successful cohort mode, distinct from
native or compiled selection. Actual GGUF CUDA B2 fullV/all42 equality,
direct/cache graph observations, native fallback, retirement/replacement
and unknown retention/fence; native HF HIP V2 C1/C8 pass. No new graph
selection policy. Full assembled isolated gate: 2398 Rust test executions (2396 workspace gate + 2 focused graph tests), zero failures, 106 explicit skips; release/installer/scripts pass.
Signed 14bbbd48/5d952364/5a7535e7: actual original Vulkan coherent
device-local mapped producer through exact backing/managed resource,
original range Write/queue and actual prior barrier. Fresh private
scratch/activation only; synchronous borrowed host copy, no staging
fallback or duplicate charge. Actual RADV full/subrange GPU consumption,
original accounting/retirement and refusal; portable Vulkan healthy/
unknown and HIP healthy pass. No new learned/performance or generalized
UMA/foreign buffer claim. Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
Final accounting correction revalidated on actual RADV and Vulkan-only/
combined qualification compile.

### Phase 1 bounded exit audit — 25 Nov

The minimum shared foundation is implemented for the existing qualified
local text/native and portable execution scope. See CURRENT_HANDOFF for
the final verification and qualification boundaries.

| Seam | Exit coverage |
| --- | --- |
| 1.1 Semantic composition | Original source-bound ModelProgram/ForwardPlan, typed closed-region/whole-text binding, original model/task/pipeline identity and explicit prepared selection. |
| 1.2 Operators/providers | Real native typed math/ports/state, managed private CUDA buffers/secondary queue, actual graph observation, bounded coherent Vulkan mapped backing; SDK-free compiled provider scaffold, explicit refusal. |
| 1.3 Source/lifecycle | Authenticated original HF and held local/verified cached GGUF, shared declared SDK/pipeline/CLI source eligibility, cache/source/alias/last-reader authority and original physical admission. |
| 1.4 State/memory/control | Original backing/range/lease, completion/quarantine, independent state and sole publication, paged/streamed/prefix/persistence and cohort/queue/task/pipeline paths, typed media/clock/placement contracts. |
| 1.5 SDK/verification | Actual public source-authenticated native SDK and real CLI local/cache runs, original runtime/catalog/control/poll and selected plan/region/capture facts, feature/architecture and full gate. |

The broader engineering work listed below remains the durable expansion
queue, not additional mandatory Phase 1 exit gates. No second catalog,
allocator, publisher or execution IR. General UMA/foreign/context
interop, non-Linux fit, general DAG/media/context, multi-source HF/mixed
backends, generalized persistence/async SSD and unqualified model/
device/codec/quality paths remain explicitly open. Native parity
requirements belong to their backend phases; concrete graph SDKs Phase 6.
The full gate and affected final physical/feature checks pass.
**Phase 1 exited into Phase 2.1 UX-1; UX-1 is now accepted (see Phase 2.1).**

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

- [x] UX-1 shared presentation foundation accepted: terminal detection,
  rich/plain, color/NO_COLOR, quiet/verbose/progress, typed reporting and
  failures/hints, original machine stdout/stderr contracts. Full isolated
  gate 2431 pass/0 fail/106 skips, release PTY/device JSON and existing
  CPU run equivalence. No source/backend/runtime policy change.
  [CLI owning checkpoint](cli-device-routing/TODO.md).
- [x] UX-2 typed command tree + shared cold model/config resolver accepted:
  canonical hierarchy and compatibility dispatch, shared globals/help/static
  completion, exact digest and explicit by-name selector through original
  cache metadata/leases, one selected run document/profile and original
  overrides, model/device reads and explicit lifecycle forwarding.
  Full isolated gate 2451 pass/0 fail/106 skips; synthetic CPU equivalence
  and release completion/JSON smoke pass. No new runtime, acquisition,
  physical or numerical qualification.
- [x] UX-3 shared native ControlPlane: one facade control state, original
  runtime/cache/device/config/serving observations and typed actions,
  bounded transient event cursors with resync, exact registration/generation
  guards and original task output/cancellation. Canonical model/device CLI
  consumer; real approved Qwen 0.8B GGUF two-client lifecycle and loopback
  integration pass. Full gate 2457 pass/0 fail/108 skips (+1 focused
  model integration); no duplicate execution or transport owner.
- [x] UX-4 bounded canonical single-plan lifecycle: shared download
  observation/actions with exact operation IDs, original strong-validator
  checkpoint/range pause/resume and safe-boundary fallback, cancellation,
  cold partial inventory, interactive/plain/quiet/JSONL and actual receipt.
  Direct provider plans share progress; advanced forced modes and aggregate
  component controls remain open. Full isolated gate 2459 pass/0 fail/109
  skips; separate SDK acquisition integration and five PTY/JSONL fixture
  runs pass. No duplicate transfer owner or new provider grant.
- [x] UX-5 bounded canonical one-line run: shared durable name/approved
  catalog selection, exact provider/snapshot pin, positional/file/stdin
  prompts, original CPU/native Qwen owner retained across independent
  TTY requests, actual selected model/backend/device status. Explicit
  consented 2B/27B exact-source recipes, original shared acquisition
  operation and precommit digest guard. Legacy 0.8B manifest recreation
  and unsupported routes refuse. Synthetic CPU equality and acquisition
  guard, final combined lint/feature build pass; full gate 2465 Rust
  executions pass/0 fail/109 skips, release/installer/scripts pass; real
  Qwen 0.8B GGUF CPU PTY two-prompt one-shot equality and clean exit pass.
  No remote transfer or physical GPU requalification.
- [x] UX-6 stable canonical serve: positional/explicit aliases and shared
  run/source/config/device policy; typed serve settings and named profiles.
  CPU GGUF and single approved original HF snapshot alias sets, same
  eligible acquisition consent and original loader. Legacy serving profiles
  retained; incompatible route sets refuse cold. Original ControlPlane
  listener with actual startup, sampled status and joined shutdown. Real
  cached Qwen 0.8B GGUF release alias equality/one-owner/SIGTERM test and
  cold refusal pass; combined lint and GPU feature compile pass. No protocol
  rewrite or physical/remote acquisition evidence. Full isolated gate
  2466 pass/0 fail/109 skips; release/installer/scripts pass.
- [x] UX-7 unified configuration/profile management: existing v2 Document
  is ConfigDocument, one selected document with implicit default fallback
  and CLI precedence. Typed UI/offer preferences, cold validation/
  provenance, atomic ConfigFile replacement, profile copy/create/remove
  and temporary validated editor, basic inline managers. Focused SDK/CLI,
  PTY and combined lint/feature builds pass. Corrected full isolated gate
  2473 pass/0 fail/109 skips; release/installer/scripts pass. Serial
  combined release run_unified 6/0/0 including real cached GGUF serving
  with unchanged admission bound. No new source, network or execution authority.
- [x] UX-8A native terminal shell: bare capable TTY opens read-only native
  view over shared Runtime/ControlPlane and cold owner observations.
  Responsive navigation/palette, scoped terminal/signal lifecycle, optional
  CLI-only Ratatui/Crossterm (ordinary default/all and product releases).
  Actual PTY and cell rendering/reducer tests, strict lint pass; full
  isolated gate 2477 pass/0 fail/109 skips, release PTY and installer/
  scripts pass. No implicit execution or new backend qualification.
- [x] UX-8B native Home / Running / Serve: shared canonical resident route
  planning, original source/device/admission and listener policy, one
  runtime and owned action worker. Load/quick-run, exact registration
  invoke/unload, generation start/stop, bounded output and cooperative
  cancellation; confirmed foreground retirement. First actual cached
  GGUF TUI/HTTP lifecycle and combined release 6/0/0, strict lint and
  combined feature compile pass. Final repeat-invoke/paste real PTY/HTTP
  rerun and remaining release tests pass; full isolated gate passes
  (2,479 passed, zero failed, 109 ignored), release PTY/installer/scripts
  pass. UX-8B accepted.
- [x] UX-8C native Models / Downloads / Devices: exact cached digest
  actions, static model catalog/partial observations, non-replacing
  alias and original reclaim outcomes, physical device source-policy
  check/load, coalesced inventory reader off presentation loop. Same
  runtime original hub/recipe acquisition, explicit pinned file/quant
  consent, bare offline, exact operation pause/resume/cancel and observed
  progress. Reducers/intent, real controlled-transfer PTY, strict lint
  pass; actual cached lifecycle and remaining release tests pass; full
  isolated gate 2,481/0/109, release PTY/installer/scripts pass. UX-8C
  accepted.
- [x] UX-8D native Profiles / Config / Logs: shared ConfigFile atomic
  snapshot-based editor, typed search/provenance and raw repair,
  create/copy/activate/confirmed remove, all cold reads on original
  observation worker, bounded operational action/control-event log with
  search/level/component/follow. Profile activation is process-local
  for future actions, no new runtime or settings owner. Focused reducers,
  strict lint and actual settings PTY pass; serial cached release lifecycle/
  profile-switch passes (22.95s), remaining release tests pass. Full isolated
  gate 2,483/0/109, release PTY/installer/scripts pass. UX-8D accepted.
- [x] UX-9A embedded SolidJS/Vite Web shell: committed locally reproducible
  assets/license, optional frontend-only feature, shared foreground
  Runtime/Selection/retirement, loopback independent random bearer,
  numeric Host/origin boundary, bounded read-only snapshot/SSE, responsive
  shell and singular palette. CLI web and TUI same-process private URL,
  exact owner identities, no new model/cache/inference owner or actions.
  Actual HTTP/Firefox responsive smoke, JS tests/strict lint and serial
  cached TUI/HTTP/Web shared-owner release 6/0/0 pass. Complete isolated
  rerun 2,485/0/109, release Web/TUI/installer/scripts pass. UX-9A accepted.
- [x] UX-9B Web Home / Running / Serve: original resident worker promoted
  into one shared host action slot, exact identities and native-private
  results, bounded authenticated JSON POST/action receipt, explicit
  cached/configured source load/run/invoke/unload and listener start/stop,
  cancellation and original streaming output. Solid actual forms and
  terminal usage, original runtime/inference owners. Strict lint,
  JS/focused/cold HTTP/Firefox pass; real optimized cached browser/
  TUI/Web/HTTP lifecycle passed (46.14s), full isolated gate
  2,489/0/109 plus release Web/TUI/installer/scripts. Accepted.
- [x] UX-9C Web Models / Downloads / Devices: accepted;
  shared cache/pull/device actions, authenticated cold inventory, exact
  identities, explicit pinned acquisition/consent and offline refusal,
  original transfer pause/resume/cancel and native/Web invalidations.
  Actual controlled HTTP/native PTY/Firefox transfer, cold browser and
  strict lint/dependency isolation passed. UX-9C optimized cached integration passed 1/0/0 (48.13s, five filtered); full isolated gate 2,489/0/109 (318 Rust suites), release Web/TUI/installer/scripts passed. UX-9C accepted.
- [x] UX-9D Web Profiles / Config / Logs: shared ConfigFile/Selection/action
  slot, safe typed and normalized raw YAML, protected removal and exact
  stale refusal; bounded host operational log, native/Web invalidations.
  Strict lint, terminal-free Web, JS, actual HTTP/native attached-Web PTY
  and Firefox pass. Initial interrupted gate completed in UX-9E below.
- [x] UX-9E shared control events and simultaneous TUI/Web acceptance:
  native log revision redraw, native safe pause -> attached Firefox
  resume/durable or cancel/partial, same original operation and joined
  retirement. Real cached combined integration 1/0/0 (54.16s), native
  active-policy edit/stale Web refusal/Web restore; unchanged admitted
  scheduler, registration, listener and exact HTTP inference output.
  Combined release CLI 50/0/3, actual settings PTY, Full isolated workspace gate 2,491/0/109 (318 Rust suites), release Web/TUI/installer/scripts passed.
  UX-9D/E accepted.
- [x] UX-10 core direct CLI implementation: cold doctor, model-specific
  physical device dry-run, validated pipeline list/inspect/check, fresh
  positional model route, help/completions/compatibility and CLI guide.
  Strict minimal/combined Clippy and isolated focused 59/0/1 pass.
  Isolated formatting/architecture and HIP/Vulkan compilation pass;
  comprehensive gate stopped during CUDA native transport rebuild.
- [x] UX-10 final acceptance: existing CUDA driver, HIP runtime/RTC,
  Vulkan loader probes; shared original cooperative quota charge/live
  reservations under existing-only nonblocking publication ownership.
  Busy unknown; no new locks, files, runtime or dependency. Focused
  44/0/2, strict minimal/cache/combined lint. Complete isolated gate sequence passed 2,505/0/109 (318 Rust suites), release Web/TUI/installer/scripts. Original run stopped at two transient busy-observation fixture assertions in acquisition/cache transactions; test-only bounded busy retry, reran affected cache-only/diagnostic/acquisition suites and exact remaining gate commands. No production change after the initial run.
  Graph inspection is direct text/JSON, general pipeline execution refuses.
  **UX-1–10 accepted. Next: public serving (§2.3).**

- [x] Close typed command tree, ConfigDocument/profile merge and shared
  presentation/action/snapshot/event model. One resolver for local paths,
  model names, explicit source locators, backend aliases and canonical
  source identity. CLI overrides selected config overrides defaults;
  no implicit remote/download/approximation authority.
- [x] Complete supported model/download/cache lifecycle, progress, pause/resume/cancel,
  device discovery, headroom, qualified auto-selection, one-line run/serve,
  config/profile management, pipeline inspection and doctor/explain.
  Explicit backend aliases mean selected backend or factual refusal.
  Advanced direct-provider modes and aggregate component control remain
  explicit deferred work in the CLI/device owner TODO.
- [x] Stable scripting output, JSON and plain non-TTY behavior; beautiful,
  responsive terminal mode. Show backend, provider, numerical/quality profile,
  shape and execution plan, supported features and actual selected/refused
  modes. Cold diagnostics reuse the SDK; public benchmark interface
  remains §2.3.

### 2.2 Native TUI, then embedded Web

- [x] Build native TUI (UX-8A–D): bare interactive command opens a responsive
  shell; Home/Running/Serve, Models/Downloads/Devices, Profiles/Config/Logs.
  Shared events, command palette, keyboard/navigation, cancellation,
  resize/accessibility; no action on mere visual selection.
- [x] Build embedded SolidJS/Vite Web (UX-9A–E), following the existing AW
  reuse/design direction and same visual vocabulary. Same high-value
  views/actions as TUI; `zg web` and opening Web from TUI reuse original
  runtime/control plane. Keep presentation state local; allow simultaneous
  clients without duplicating owner state.
- [x] Keep one ConfigDocument and control protocol; secure local control
  auth and browser transport independently of inference auth. No separate
  model/scheduler/cache implementation in a frontend. Complete UX-10
  doctor/pipeline polish, compatibility cleanup and real simultaneous-client
  lifecycle acceptance.

### 2.2A Native TUI interaction and visual polish (UX-P)

**Accepted (26 Nov):** user-requested full mouse interaction,
resizable/focusable panes, independent scrolling and polished visuals
completed before public serving. See the owning
[TUI polish specification](cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md).
UX-1–10 remains accepted; this is additional presentation work.

- [x] UX-P1: semantic palette, Dark/Light/Auto, typed appearance preview/save,
  quieter native chrome and Web theme/token parity. Strict lint and focused
  Rust 81/0/3, JS 3/0, real PTY and Firefox regressions passed. Native Auto
  uses terminal hint/dark fallback; active probe deferred.
- [x] UX-P2: scoped mouse capture and ui.mouse policy, render-derived
  exact inert selection, focus, draggable splits, maximize/restore,
  responsive collapse and independent visual-row scrolling. Strict lint,
  focused Rust 84/0/3, real SGR PTY and original native/Firefox smoke pass.
- [x] UX-P3: action bars/overflow, bound commands and modal controls,
  typed setting/log choices, local shelf search, grapheme caret/selection
  and independent editor scrolling. Focused UX-P3A/B accepted; P3B
  86/0/3 plus real cached lifecycle 1/0/0, native editor/settings/mouse
  regressions and stale refusal passed. Remaining final view affordances,
  stable anchors and visual/focus edge cases are UX-P4A; P4B full gate.
- [x] UX-P4: visual/terminal-depth/accessibility polish, actual PTY and
  simultaneous-client lifecycle acceptance; full isolated gate.
  P4A1 user screenshot-driven minimalism: at most three quick actions,
  one navigation rail, concise summaries, on-demand exact Details and
  smart units. Follow-up removes toolbar entirely: selected exact-row
  context menus and inline live local search, explicit Alt shortcuts.
  P4A2 identity scroll anchors, separate response, safe log details,
  compact scrollable dialogs/editor drag edges focused-qualified
  (91/0/3 and native/shared transfer PTY). User follow-up: CLI Cargo
  default full production, preserve typed errors and safe public/opt-in
  authenticated diagnostic projection and structured logs. User follow-up:
  [live model performance and layer inspection](cli-ux/LIVE_MODEL_PERFORMANCE.md),
  registration-owned TPS/TTFT and selected object details before P4B.
  P4B focused real resident/config/listener/transfer regressions, native
  full-default linked launch/strict lint and actual CUDA layer/counter
  observations pass. User-directed M2 reference-aligned dashboard and
  exact layer strip, visible details and activity refinement focused-
  qualified; standalone serving refusal assertions corrected and pass.
  Final assembled private-network gate on ad7a4178 passed: 2,539 passed, zero failures, 109 explicit skips across 318 Rust suites.
  Release Web/TUI/installer/scripts passed. UX-P accepted; proceed to §2.3.
  Optional deeper profiling follows actual provider capabilities.

  User follow-up: model-local native/Web Execution selector accepted; Run,
  Load and Load for Serve inherit profile by default, with optional explicit
  backend/device draft over the original resolver. Actual cached CPU lifecycle
  and assembled gate 2,543/0/109 (318 suites) pass; OA-1 remains next.

No new runtime, cache, configuration or execution owner. No inference
protocol or backend performance work was admitted before this bounded exit.

### 2.3 Public serving and performance interface

OA-1A bounded async text transport is accepted (complete isolated gate
2,538/0/109, 318 suites; real cached CLI/TUI/Web and SDK passed). OA-1B
bounded decoded-body staging and owned WebSocket primitives implemented
and accepted (2,544/0/109 across 318 isolated Rust suites; real cached
CLI/TUI/Web and SDK passed). Public chunked text now uses the same
bounded reader and a narrow cached Hyper server framing patch; focused
33/0/0, real cached CLI/TUI/Web/SDK and full isolated retry 2,546/0/109
(318 suites) passed. OA-1B multipart foundation implemented using approved
checksum-verified Multer and original bounded ingress/staging owners;
focused 38/0/0, real cached CLI/TUI/Web and SDK, and full isolated gate
2,551/0/109 (318 suites) passed. OA-1 accepted. OA-2 shared semantic
request/model-template boundary is implemented (core typed input,
original semantic queue/events and model-owned Qwen template/pricing;
focused and actual tokenizer checks passed). Actual cached consumers
and assembled gate passed (2,553/0/109, 318 suites, 878.6s). OA-2
accepted; OA-3 Models/retrieve and stateless Responses text JSON/SSE
implemented over the original typed receipt, focused 167/0/0 and strict
actual cached official SDK canary passed. Native/browser/Rust SDK
regressions and assembled gate (2,557/0/109, 318 suites, 617.5s)
passed. OA-3 accepted. OA-4A foreground response storage/retrieval,
deletion/input pages and continuation implemented using the original
semantic request and listener-owned bounded charged protocol store;
focused 170/0/0, strict lint and real installed official SDK/CLI/TUI/Web/
Rust SDK and full isolated gate 2,560/0/109 (318 suites, 690.5s)
passed. OA-4A accepted. OA-4B joined nonstream background polling/
cancel implemented on original receipt and shared quota; focused
172/0/0 and strict lint passed; real official SDK/CLI/TUI/Web and Rust
SDK passed, including actual cancelled. Full isolated assembled gate
2,562/0/109 (318 suites, 377.3s) passed. OA-4B accepted.
OA-4C conversation/item resources and atomic context append accepted:
focused 177/0/0 and strict lint, actual official SDK/CLI/TUI/Web (124.91s)
and Rust SDK (15.56s) passed; full isolated gate 2,567/0/109 (318 suites,
342s) passed. Same protocol quota and original semantic admission; no
additional model state or execution owner. OA-4D exact input-token count
accepted: focused 179/0/0 and strict lint, actual official SDK/CLI/TUI/Web
(162.61s) and Rust SDK (18.79s) passed; full isolated gate 2,569/0/109
(318 suites, 783s) passed. No generation admission or conversation mutation.
OA-4E bounded background streaming/replay/resumption is implemented:
focused 182/0/0, strict lint, actual official SDK 2.33.0 cached-model
stream/replay/cancel/atomic append and Rust SDK (19.96s) passed. Same
joined receipt, shared charged protocol log and exclusive cursor.
Full isolated assembled gate 2,572/0/109 (318 suites, 548s) passed.
Final serial resident Firefox/CLI/TUI retry passed 83.29s on unchanged
37723346; earlier model-load timeout did not recur. OA-4 accepted.
OA-5 typed Chat/legacy text compatibility accepted: focused 184/0/0,
strict lint, actual official SDK/CLI/TUI/Firefox (184.78s), Rust SDK
(13.99s) and full isolated gate 2,574/0/109 (318 suites, 306s) passed.
Keep unavailable tool/structured-output/media/sampling capabilities
explicitly refused. OA-6A: native BERT source-bound embedding candidate,
shared CPU noncausal primitives and verified single-file Safetensors
owner, synthetic numerical/ownership qualification and complete unchanged
workspace gate (2,581/0/109). OA-6B native shared Runtime/registry/FIFO/workers
and typed charged receipt integrated; actual synthetic source-bound SDK,
mixed scheduling and refusal tests pass (246/0/1 focused, strict lint).
Assembled gate: 2,584/0/109 (321 Rust suites, 834s), complete isolated gate
passed. OA-6C public embedding wire adapter implemented over the same
registered encoder and receipt; explicit qualification gate, four input
forms, exact usage/float/base64, bounded projection and actual synthetic
BERT/installed SDK consumer passed (246/0/1 focused, strict lint).
Assembled gate: 2,585/0/109 (322 Rust suites, 575s), complete isolated gate passed. OA-6 remains open for authorized actual
MiniLM trained source, independent tokenizer/vector parity and SDK wire
acceptance. Continue ready OA-7 audio work using existing original Gemma
4 E2B checkpoint: OA-7A native CPU F32 waveform frontend, all 12 learned
audio layers and language projector; actual independent Transformers
parity (4/16 x1536 rows), cancellation and ownership pass. Focused
125/0/13, strict lint. Assembled: 2,587/0/110 (322 Rust suites, 570s), complete isolated gate passed. OA-7B shared original window source
(74.4 MB retained combined), native same traversal/StateRoot audio
conditioned prefill and cached decode, actual independent full vocabulary
logit parity (524288 values, max abs0.000320435), cancellation and
retirement pass; focused222/0/14 strict lint. Assembled:
2,591/0/111 (322 Rust suites, 679s), complete isolated gate passed. OA-7C original model prompt/tokenizer and bounded native speech task
through existing Runtime; independent local synthesized speech transcript
and exact native prompt. Native learned: actual native SDK "Hello world", input=41, output=3 including stop, finish=Stop; 263.484s completion, full real test passed 292.62s (including cancellation/retirement).
Focused247/0/2 strict lint. Assembled: 2,593/0/112 (324 Rust suites, 617s), complete isolated gate passed.
OA-7D capability-gated multipart PCM16 WAV, original public registration
and receipt, JSON exact usage/text, strict controls/length and output
limits. Focused249/0/3 strict lint. Actual learned: actual installed OpenAI SDK2.33.0 strict JSON "Hello world", input41/output3/total44, exact independent reference; 264.264s SDK completion, real learned wire test passed292.73s including retirement.
Assembled: 2,595/0/112 (324 Rust suites, 904s), complete isolated gate passed. OA-7E source-supported AST through same speech task/receipt and
capability-gated multipart translations, original model-owned bilingual
prompt and target projection, independent Spanish "Hola mundo" ->
"Hello world" reference. Actual learned SDK: actual installed OpenAI SDK2.33.0 strict JSON {"text":"Hello world"}, exact independent English target and native input56/output8 usage; SDK completion378.481s, real learned test passed399.51s including retirement.
Focused314/0/6 strict lint. Assembled: 2,596/0/112 (324 Rust suites, 698s), complete isolated gate passed.
OA-9A bounded text Realtime over original registration, semantic Runtime,
worker and receipt. Explicit CPU Qwen eligibility, text-only conversation,
native usage, complete/incomplete/failure/cancellation, joined retirement.
Actual installed SDK2.33.0/websockets16.0 cached Qwen0.8B GGUF two turns exact
native text/usage, both length-limited; focused62/0/0 strict lint,
final assembled 2,597/0/112 (326 Rust suites, 546s), complete isolated gate passed. Corrected malformed generated AST
terminal classification. No approved TTS/image generator/MiniLM source
found; keep missing-media holds. OA-10 supported-capability closeout
corrects cold markers, help, standard safe errors and actual refusal
behavior; canonical [endpoint matrix](../../docs/OPENAI.md), focused63/0/0
strict lint. SDK audit passed (installed SDK2.33.0, cached CLI/native/attached-Web 125.30s, Realtime 20.05s); assembled 2,603/0/112 (326 Rust suites, 463s), complete isolated gate passed.
Ordinary benchmark acceptance: existing serve-bench/compare/matrix
support Responses/Chat/raw Completions, same workloads and normalized
actual usage, strict terminal SSE validation. Actual cached ordinary CPU
Qwen0.8B canonical listener,2x1 workers,128 context/8 output,
4 varied t0 arrivals per cell, Chat C1/C2, Responses C2, matched raw
Completions C2 all4/4 exact output/usage/finish; 16 total completed,
native peak2, request/listener retirement. Functional CPU only;
no natural-speech, voice, true token ITL or physical GPU parity claim.
Final 2,603/0/112 (326 Rust suites, 235s), complete isolated gate passed.

- [x] Close the [OpenAI-compatible serving iteration](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md) for actually qualified capabilities:
  model catalog/router, OpenAI Chat/Responses and supported completion/
  embedding/tool/structured-output contracts, usage, errors, JSON/SSE,
  streaming/cancellation/timeout/backpressure and real concurrent requests.
  Unsupported tasks/options must refuse rather than simulate compatibility.
- [x] Preserve the separate control surface and inference API with typed
  configuration, explicit route/trusted sharing scope, model lifecycle and
  shared scheduling for the qualified scope. Future WebRTC/media/hosted
  tenancy remain capability-gated expansion through existing task contracts.
- [x] Establish ordinary public CLI/SDK/server benchmark invocation,
  effective config and end-to-end metrics, not a qualification-only serving
  process. Retain one process/multiple model instances and qualified
  pipeline routing. Actual native/SDK lifecycle and benchmark pass;
  final assembled gate passed.

**Exit:** ordinary one-line run/serve; coherent model/device/config lifecycle;
both TUI and Web use one control model and real owner events; public serving
works for qualified model/task/backend paths; API behavior and actual
concurrency can be benchmarked through normal product surfaces.
Owners: [CLI/device routing](cli-device-routing/TODO.md),
[UX design](cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md),
[serving](serving-and-speculation/TODO.md),
[platform](platform-and-release/TODO.md).

**Bounded Phase 2 accepted (27 Nov).** Ordinary qualified CLI/TUI/Web,
control/serving, SDK and public benchmark paths passed. The actual
benchmark recheck and complete isolated gate establish the supported
interface/lifecycle exit. Future model/media, hosted tenancy and general
pipeline/backend expansion remain in their owning plans, not implicit
acceptance. Current resumption follows §2.4; no optimized GPU/vLLM result inferred
from functional CPU benchmark evidence.

## 2.4 Programmable stack foundations and parallel lanes

**Queued immediately after UX-11D acceptance — 28 Nov user decision.**
Implementation has not started for this scope. See the
[service/SDK design](programmable-service/README.md),
[composition design](programmable-service/COMPOSITION_AND_ADAPTIVE_EXECUTION.md),
[phase/dependency plan](programmable-service/IMPLEMENTATION_PHASES.md) and
[live checklist](programmable-service/TODO.md).

The shared foundation block has three substantial outcomes:

1. **PS-F1:** extract reusable Service/Client plus source/preparation,
   independent operation supervision and owned observations; make ordinary
   SDK run/stream/serve and existing frontends use the same native behavior.
2. **PS-F2:** implement typed task/model/mapping revisions, checked authoring
   and provider binding over existing consumers; owned adapter contracts,
   capability/grant checks, minimal durable operations/recipes and observation.
3. **PS-F3:** implement actual bounded live registration/activation, complete
   admitted composition retention, coexistence admission and safe recovery
   boundaries; qualify the assembled foundation through real supported paths.

The SDK is usable directly and supplies the native contracts for an optional
SDK-hosted MCP server. Full MCP transports/protocol implementation happen in
their own lane after this gate and require no CLI-private host.

At foundation exit, reopen **ROCm and CUDA independently** alongside
**MCP/native attachment**. Task DAGs and model authoring can progress
independently; mapping/experiments can start with existing models and qualified
implementations. Continuation and isolated extension use the same ownership
contracts but have separate useful acceptance scopes.

Keep work in its owning plan. The programmable-service checklist tracks
shared integration and service capabilities; the existing backend, numerical,
source and serving TODOs retain their detailed work and evidence. One common
contract writer coordinates changes; disjoint backend/adapter paths have
explicit owners. A common-authority failure gates affected consumers; a
backend- or protocol-local failure does not stop unrelated ready work.

No performance result, new model capability, general checkpoint scope or GPU
isolation claim follows from these plans. The user-authored root README and
the other agent's UX-11D source/work remain unchanged.

## 3. ROCm — AMD parity

Use X2 (gfx1151) as the first measured performance target, then other qualified
AMD hardware. Source authoring on L5; execution/testing on X2/A1 per the
handoff. Observe shared capacity and retire only our own allocations.

P3A fresh readiness review: canonical original GPU profile previously
hard-coded cohort width0 (sequential HIP). Cold automatic/explicit
cohort and original ROCm exact device-envelope policy now implemented
through existing config/CLI/control/SDK preparation; comparison
requires full input/output/total usage. X2 exact Phase2 bundle
synchronized; missing local Cargo graph restored offline from
verified L5 cache. Signed 50f6e276 passes full isolated assembled
gate (2,603/0/112; 326 suites; 519s) and actual public native
47+3 canaries: 0.8B C1/C8 23.26/41.28, 2B 10.59/29.47,
27B C1 2.22 tok/s; exact IDs/usage/envelopes, actual width8
publications on both small C8 cells and explicit sequential0
control (0.8B C8 23.89 tok/s), joined retirement. See current
handoff for scope. Pinned vLLM0.33.0+rocm72 restored on X2,
GC-rooted and real optimized BF16 public 47+3 C1/C8 qualified for all
three original models, 81/81 exact IDs/usage and joined retirement.
Fresh standalone medians 53.59/214.71, 36.33/141.58, 3.80/13.48
tok/s; no fresh paired parity claim. Retain source/numerical/ownership
rules. P3B compact HIP V2 greedy feedback at f79dce28 physically
validated: original singleton/cohort exact full-logit differential,
compact packet/committed-device embedding and three original model
C1/C8 public canaries. Existing source/reference profiles preserved.
Fresh owned 0.8B C1/C8 paired native/vLLM medians 24.18/65.22 and
42.62/226.57 tok/s, all exact response IDs/text/usage/finish, joined
retirement. Configured envelope matched; no independent state or SLO
parity claim. Fresh pre-region 2B pair: native/vLLM 10.53/40.18 C1,
30.68/144.61 C8, exact response identity and retirement.
P3C native row regions 54b2c910: exact RMS differential and native
A/B for 0.8B/2B C1/C8, 47+3/16, all exact output IDs/usage and
retirement; short C8 +20%/+15%, long C8 +9.75%/+6.12%.
Fresh paired 27B C8 at 54b2c910: 5.8652/13.51425 tok/s, 32
exact timed requests and all four provider retirements.
28 Nov takeover: retained 54b2c910 fresh paired 27B C1
2.25575/3.82090 native/vLLM tok/s (59.037%), 32 exact timed
requests, all four provider retirements, original idle VRAM.
Initial C1/C8 paired campaign complete (small-model pairs at
pre-row-region checkpoint); current matched acceptance still open.
Complete assembled gate remains open: takeover 1792/0/97 (235 Rust suites);
GPU and main CPU gates passed; resume deadline in combined CPU
product integration, then acquisition/serving/release suffix. See handoff.
P3C broad gate 1785/0/97 (237 suites), strict production/CPU
Clippy passed; 1200s timeout in serving-only rebuild after
cache and CPU preparation checks, no reported test failures.
P3D exploratory trace after P3C on X2 0.8B finds C8 recurrent key
reductions ~12.86ms of ~49.72ms incremental device-kernel step;
BF16 BBS GEMMs ~18.43ms. Profiler shutdown rejected, so diagnostic
only. Exact private-state recurrent region is the next candidate;
see CURRENT_BASELINE and model-execution TODO. No new selection.
Prefix/paged/ragged/remaining temporal/attention regions remain ordered
measured work.

- [x] Restore the approved pinned optimized vLLM provider and qualify
  actual original-source 0.8B/2B/27B C1/C8 serving on X2.
- [x] Initial fresh paired original-source 0.8B/2B/27B C1/C8
  baselines through the existing owned comparator. Small-model pairs
  pre-row-region, 27B pairs post-region. Retain exact checkpoint and
  workload distinctions in CURRENT_BASELINE.
- [ ] Extend fresh paired coverage as optimized source/regions land;
  use standalone and historical cells for direction, not paired
  claims. Get quick real C1/C8 feedback.
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

## 4. CUDA — NVIDIA parity

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
| [Programmable service](programmable-service/README.md), [foundation/lane plan](programmable-service/IMPLEMENTATION_PHASES.md) | After UX-11D: PS-F1/F2/F3, then independent SDK/MCP/composition/experiments/continuation/extension lanes |
| [Pipeline composition](pipeline-composition/TODO.md), [shared execution](pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md) | Phase 1 contracts; PS foundations and independent composition/mapping capability lanes |
| [Model acquisition](model-acquisition/TODO.md), [immutable shard spec](IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md), [acquisition spec](MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) | Phase 1 and 2 lifecycle; additional sources/models qualified later |
| [Memory/streaming](memory-and-streaming/TODO.md) | Phase 1 common ownership, Phases 3–5 optimization, later topology |
| [CLI/device routing](cli-device-routing/TODO.md), [CLI UX](cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md) | Phase 1 SDK/routing prerequisites, Phase 2 complete UI |
| [Serving/speculation](serving-and-speculation/TODO.md), [OpenAI serving](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md) | Phase 1 contracts, Phase 2 APIs, Phases 3–5 native performance |
| [Model execution](model-execution/TODO.md), [CPU/GPU optimization](cpu-gpu-optimization/TODO.md), [backend parity](cpu-gpu-optimization/BACKEND_PARITY.md) | Phase 1/PS common contracts; Phases 3 and 4 in parallel, then Phase 5 and optional Phase 6 |
| [Numerics/context](numerics-and-context/TODO.md), [correctness policy](numerics-and-context/CORRECTNESS_POLICY.md), [Qwen API/CLI frontier](qwen-parity/) | All phases; source/quality correctness precedes respective performance and capability claims |
| [Multimodal](multimodal/TODO.md), [distributed execution](distributed-execution/TODO.md) | Phase 1 boundaries; deferred qualified implementation/expansion |
| [Platform/release](platform-and-release/TODO.md), [traceability](REQUIREMENTS_AND_TEST_TRACEABILITY.md), [experiment program](EXPERIMENT_PROGRAM.md) | Cross-cutting acceptance and later platform breadth |

The dated handoff/archive packs remain the historical record. Do not erase
their detailed pending items or duplicate them into a second task ledger.
