# Cross-cutting invariants, state lifetimes and failure semantics

**15 Oct 2026 — SERVOPT multi-tenant/state invariant:** [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) preserves ONE StateRoot publisher and physical resource authority across concurrent models, source-compatible hybrid prefix fork/COW, target-only exact speculative correction and true GPU events; request/turn canceled epochs must not advance card-wide physical epoch or destroy other tenants' state. Prefill/decode/media work runs only with positive bounded credits, and no queue/IO/transport operation may hold global arbiter locks while awaiting physical completion. An overload fast-rejection, stale cache/key, remote state loss or unknown GPU completion cannot counterfeit a committed output. See SERVOPT-01..36.


**15 October 2026 — NEW STORAGE-RESIDENCY INVARIANT SSD-INV-01:** An immutable source tensor retains exact semantic/quant identity across storage tiers. A physical expert slot must not be evicted/reused while ANY encoded, captured, queued, submitted or still-running GPU command may dereference its old contents. Read leases start at GPU encoding; generation-tagged slot table publication follows source IO verification + true DMA Ready. No host directory consistency check may substitute for whole-pending-graph reader proof (DS4 #384). A short/corrupt SSD read before GPU submission fails without fencing a healthy card; unknown submitted upload/kernel retains actual physical buffers and global fence. Mixed quant Q2/Q4 pools share a single bounded byte budget; one layer cannot reset another layer's warm set just because expert size changes. Frozen private disk StateRoot snapshots are not immutable model tensors. [Normative specification](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md); [32 negative/positive gate cases](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md).


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


Every subsystem must pass these requirements. They are **design invariants**, not suggestions. Reference other documents for concrete structures and test methods. In failures, prefer explicit typed error/replay over plausible but invalid results.

## Correctness and identity

**INV-001 Checkpoint provenance:** a runnable model links verified model weights, config, tensor manifest, tokenizer, chat template and quant policy. A model ID string alone cannot identify semantics.  
**INV-002 Semantic exactness:** all required operator families, state dependencies and numeric policy are declared; unsupported operators fail at load with actionable diagnostic.  
**INV-003 Quant integrity:** every quantized block's logical size, padding, scale and offset match the model/format. Repacking is separately proved lossless; requantizing is an explicitly different model artifact.  
**INV-004 Causal frontier:** all visible tokens reflect valid committed model state; no future/rejected speculative token contributes to visible recurrence, indexer, attention or RNG.  
**INV-005 Cross-session isolation:** requests never mutate another's logical state even if sharing immutable prefix/weights; race ordering proven.  
**INV-006 Exact template contract:** tokenization, BOS/EOS, hidden reasoning, tool rendering, multimodal positions and stop logic match the chosen model reference.

## Effects, ownership and memory

**INV-007 Linear state transition:** each state version consumed according to effect ordering. Graph optimization cannot drop or reorder writes across a live dependency.  
**INV-008 Lease generation:** GPU/host/IO references hold a valid `BufferLease`; eviction waits for completion, stale generations rejected.  
**INV-009 Expert slot safety:** resident/missing expert kernels cannot observe a partially populated or replaced slot; slot version + layer/expert mapping verified at use.  
**INV-010 Physical budget:** enforce OS headroom, pinned/committed physical memory, driver/UMA/GTT constraints and session scratch reservations before admission. Report virtual GTT capacity distinctly.  
**INV-011 IO integrity:** disk-backed table/expert read validates exact position/length/checksum as appropriate; EOF/errors invalidate dependent transaction. No stale bytes.  
**INV-012 Async synchronization:** events/fences order GPU writes, CPU reads, peer exchanges and slot reuse. No implicit reliance on an idle stream or incidental device synchronize.  
**INV-013 Fallback conversion:** an optimized representation may be replaced only at a validated layout/state/causal boundary; failures cannot pretend a different kernel with different caches continued the session correctly.

## Serving and speculative semantics

**INV-014 RNG fidelity:** sampler PRNG and penalty history correspond to committed tokens; rejected drafts do not perturb exact sampling.  
**INV-015 Speculative distribution:** when exact target sampling advertised, proposal acceptance/correction implements the target distribution; opportunistic sampling only explicit.  
**INV-016 No hidden latency starvation:** token-budget admission enforces fairness and bounded resident/preemptible resources; TTFT/P95 tracked. A batch path is qualified as native only if it truly computes multiple rows jointly.  
**INV-017 Cancellation atomicity:** cancellation releases only quiescent buffers, emits no tokens beyond committed frontier and preserves restart/replay capability.  
**INV-018 Persistent snapshot:** snapshot schema/version/checksum/model/quant/pos policy validated before state restore; prompt text privacy and permissions preserved.  
**INV-019 Remote frontier consistency:** distributed partial results cannot be committed on only one rank; fatal transport mismatch invalidates route and causes safe replay.

## Model/hardware compatibility

**INV-020 Capability truthfulness:** adapter support, numerical correctness, physical backend availability and performance qualification are distinct status fields; absence of extreme kernel is not model failure when generic backend supports all required ops.  
**INV-021 Hot-path autonomy:** direct specialized execution avoids per-op IR interpretation and avoids host routing lookup for resident routed experts where the plan promises GPU-side handling.  
**INV-022 Numerical transparency:** approximations, quant changes, sparse top-k changes and non-exact sampling clearly labeled; benchmark comparisons match precision/quality policy.  
**INV-023 Security boundary:** untrusted model artifacts and client requests cannot allocate unchecked sizes, execute arbitrary code or reach arbitrary host filesystem/network.  
**INV-024 Measured claims:** benchmark artifacts bind exact code/hardware/model/driver/power/context/temperature and complete execution; incomparable author claims stay attributed.

## Failure state machine

```text
ADMITTED -> RESERVED -> EXECUTING -> PROVISIONAL
                                |          |
                                |     VERIFY/VALIDATE
                                |          |
             FAILED <-----------+      COMMITTING -> COMMITTED -> EMITTED
                |                           |
     QUIESCE/QUARANTINE              durable frontier
                |
    RELEASE / REPLAY / REJECT
```

Emit only committed tokens. If a kernel fails after output from a previous commit has been emitted, report the failure without forging the rest of the stream. Internal rollbacks never retract already externally committed output. Cancellation epoch guards against late callbacks.

### Typed error classes

- `ArtifactError::{Corrupt,Checksum,UnsupportedQuant,WrongShape,TokenizerMismatch}`
- `ModelError::{UnknownArchitecture,UnsupportedOperator,StateSchemaMismatch}`
- `PlanError::{NoLegalKernel,LayoutConversionMissing,NumericalPolicy,GraphCaptureUnsupported}`
- `ResourceError::{UnsafeBudget,QuotaExceeded,OutOfMemory,LeaseConflict}`
- `DeviceError::{LaunchFailure,Timeout,LostDevice,InvalidEvent}`
- `StorageError::{UnexpectedEof,ReadFailed,ChecksumMismatch,CorruptSnapshot}`
- `ProtocolError::{PeerMismatch,TransportFailure,IncompleteCollective}`
- `RequestError::{InvalidInput,TooLarge,Cancelled,UnsupportedModality}`

These Rust names are illustrative; actual APIs may consolidate variants while preserving semantics and cause.

## Failure response / recovery matrix

| Failure | During | Required behavior | May retry? |
|---|---|---|---|
| Unknown op / malformed GGUF | load | refuse before GPU allocation, describe exact tensor/op | after compatible adapter/fixed file |
| Insufficient physical RAM | planning/admission | reject or try smaller context/batch/scratch, never silent requant | after resource changes |
| Unsafe expert eviction | route | prevent overwrite, wait on referenced event, stall safely | after lease releases |
| NVMe expert/Engram read short | execution | invalidate row/staging, abort dependent transaction | yes from clean checkpoint |
| Sparse attention selection invalid | compute | fail or execute verified fallback; no truncated wrong top-k | if canonical source valid |
| Device kernel timeout/loss | compute | quarantine event/buffers, invalidate provisional state, drain/reset | only verified restart |
| Draft reject mid-token | verify | commit accepted prefix and corrected token, restore all other state | normal path |
| Tool sampling policy changes inside draft | verify | rewind at policy boundary and verify again | yes from snapshot |
| Request disconnect | stream | cancel/reconcile no uncommitted output, cleanup after GPU quiescence | new request |
| Corrupted snapshot | restore | reject, rebuild from canonical history if available | yes recompute |
| Network TP peer disconnect | collective | fail-closed both ranks, no fake remote values | new matched route |
| Compiler/kernel tuning drift | plan | invalidate cached plan and revalidate candidate | compile with pinned toolchain |

## Ownership/lifetime diagrams (text)

```text
ModelArtifact (immutable, hashes)
     └── ModelExecutable (compiled regions, read-only weights)
           ├── ExpertCache slots [generation, GPU/IO leases]
           ├── Session A [committed state frontier p]
           │     ├── Snapshot p (immutable/shared)
           │     └── Tx p+1 (provisional, isolated)
           └── Session B [fork from Snapshot p, private continuation]

ResourceArbiter -> allocates BufferLease -> async Event -> safe recycle
```

State commits are serial per session, but independent sessions may execute concurrently in one GPU batch. Cache evictions operate over physical slots, not logical model tensor IDs. A lease must be tracked across all async CPU/GPU operations.

## Defensive tests / diagnostics

Fuzz malformed GGUF/safetensors/template/snapshot, integer overflow, truncated model file, random expert misses and slot reuse; force every possible speculative accepted-prefix length and kernel failure at each stage; stress cancellation against in-flight GPU events and server restarts; check per-state checksum at commit, physical memory watchdog and no host-wide OOM; replay same prefixes under altered chunking and log numerical difference. Distributed tests must inject wrong model identity and delayed/missing collective messages.

Every new optimization PR must explain which invariants it touches and include tests. A fallback path without an explicit state layout boundary violates INV-013, even if short prompts appear to work.

## 2026-10-10 integrated v2 addendum — cross-IR and worker-event invariants

Apply all existing INV-001…024 to the consolidated IR pipeline:
- **SIR/RIR**: only pure mathematically legal transforms; explicit state/effect token ordering; no unverified compiler speculation across possible failure/RNG/IO effect.
- **KIR/KernelPlan**: each provider advertises exact numerical, ISA, tensor/storage/state-layout, capture-mode and scratch constraints; no generic shape-only override.
- **CIR**: `Acquire/Dispatch/Copy/Read/Wait/Publish/Commit/Release` checked for alias, resource tier, event/timepoint ordering and buffer lease generation; no early expert eviction, no indefinite GPU wait on IO failure.
- **EngineCore**: one logical mutation owner; StepPlan and WorkerDelta epoch match, late completions cannot publish tokens or release referenced state. Separate worker and frontend never commit target output.
- **Prefix/LoRA/multimodal**: cryptographic model/quant/token/rendered-template/adapter/tenant/snapshot/finalization identities; cache "hit" must be fully restorable or recompute domain before use.

**Additional invariant INV-025 — source modularity:** adding a new supported semantic operator cannot require forking serving or ABI; adding new kernel provider cannot require changes to model math/HTTP; all such changes validated by extension test. **INV-026 — no uncontrolled serving optimizer:** every selected kernel variant was prevalidated for quant/model/hardware/numerics and command resources; real-time scheduler must not run unbounded rewrite/JIT. **INV-027 — mode receipt:** all user-visible capability degradations, kernel mode, quant changes requested and speculation fidelity are reported accurately. **INV-028 — package format correctness:** serialized header/magic/version/bytes/schema must be self-consistent and fuzzed, explicitly rejecting old research's defective disk-KV definition.

Failure semantics: an event may complete with error *after* a host issued cancellation; hold lease/quarantine until proper device quiescence; no double commit or retry with half-old recurrent state. Always distinguish source model precision from runtime compute and changed checkpoint quality, and fixed-plan numerical reproducibility from cross-backend approximations.


## 2026-10-12 integration — cross-accelerator safety and semantic invariants

INV-029: output-row projection split covers disjoint complete sets, restores original tensor row order and preserves model attention/recurrence/quant/routing semantics. INV-030: XRT/GPU imported pinned memory is protected by event and lease generations until *both* runtimes quiesce; timed-out external DMA may need terminal quarantine. INV-031: concurrent GPU+NPU is disabled for unknown/unsafe SoC power-fabric/firmware states, and no privileged FCLK/BIOS/IOMMU/suspend modification occurs automatically. INV-032: v8bfp16ebs8 compute is quality-qualified independently; strict numerical reference does not silently switch to NPU. INV-033: ASR/TTS/image/video TaskProgram stages have task-specific loop/output/state and do not masquerade as ordinary language autoregressive KV/sampler.

Required tests: mock NPU + GPU row join, stale BO/double free, timeout and partial result, cancellation after submit, late XRT completion, strict fallback to GPU, hardware-qualification fail-closed and 24h stability on isolated hardware. See [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

## v4 additional invariants and fault tests

**INV-034 Atomic session visibility:** `StateDomain::prepare` cannot mutate committed root; one TaskCoordinator writer validates all required domains/events and performs exactly one `PublishRoot`. Device/CIR event ready is not logical commit. **INV-035 Async host/device correctness:** validated StepProgram may progress device-native token/routing/sampling without mandatory per-token CPU synchronization; unsupported grammar/shape uses honest host-loop fallback. **INV-036 Hierarchical allocation:** global physical admission reservations and per-device local generation leases, no lock for each hot kernel scratch allocation, no impossible promise of OS-invariant free memory.

**INV-037 Dynamic shape legality:** rank, symbolic dimensions, ragged offsets/strides, numeric/quant/ISA and effect schema checked before kernel submission; cold missing variant chooses portable fallback, not uncontrolled JIT. **INV-038 Stream transport honesty:** emitted output has session/epoch/seq identity after root publish; live delivery attempt isn't exactly-once network delivery. **INV-039 Distributed fencing:** late epoch remote peer cannot cause publication of partially prepared state; durable exactly-once only if explicit protocol actually built.

**INV-040 TaskDriver boundary:** coarse trusted AI computation stage loop, no arbitrary model-supplied ExternalAction; non-token tasks use their own output frontier. **INV-041 Training future seam:** ParameterVersion mutations do not alter pinned inference sessions or model/bundle/prefix compatibility. **INV-042 Scope/minimality:** if a new specialization requires duplicated model server/allocator/scheduler or four independently persisted IR dialects, review/reduce before implementing.

Add fault tests: fail second domain prepare, late device DMA event, unsupported ragged GPU shape, remote lost ACK, slow consumer/stream reconnect, and trained model/adapter version change. Results remain NOT_STARTED until source and independent evidence exist.

## v4.3 multi-model routing and API invariants

**INV-043 — exact model binding:** alias resolution at admission yields pinned checkpoint/quant/tokenizer/template/adapter/ModelProfile revision and TaskDriver/numeric policy. A route/alias hot reload never mutates a live session. **INV-044 — replica semantic equivalence:** load balancing can only move a stateful session among compatible exact model deployments after a safe full StateRoot/placement transfer, never random per-token model or quant substitution.

**INV-045 — residency safety:** registered catalog models need no physical memory; a resident deployment has verified ResourceArbiter reservation and DeviceBackend graph leases. Active/unsnapshotable sessions, prepared speculative steps and live DMA prevent eviction. **INV-046 — router control-path:** ModelRouter performs coarse request/model admission, not source-graph traversal/JSON HTTP dispatch/locking on the GPU hot token path. Native batching uses existing TaskCoordinator.

**INV-047 — multi-model task roles:** target/draft/encoder/rerank/vocoder are separate registered models with separate admitted leases, not replicas or silent target alternatives. **INV-048 — config safety:** immutable source/model semantics and hard auth/memory/peer/numeric constraints cannot be relaxed by model profile or request; effective provenance deterministic and inspectable.

**INV-049 — local-first API:** no auto system daemon, public listener, remote peer or unknown source download on `run <local-model>`; external bind needs auth/TLS, model/admin scopes and tenant alias filtering. **INV-050 — API contract:** OpenAI-compatible is endpoint/field/stream/SDK tested subset, not implicit Responses conversation persistence, cloud built-ins or tool execution. Only committed StateRoot output emitted with sequenced event identity.

**Tests:** hot alias rename mid-stream, 100 concurrent cold requests dedup, tiny warm model protected during giant cold load, unload while in-flight KV/DMA, same source replica with incompatible numeric mode, alternate quant request refused, `/v1/models` privacy, official SDK streaming, Responses unsupported store/tool field typed refusal, verified direct CLI zero configuration.
