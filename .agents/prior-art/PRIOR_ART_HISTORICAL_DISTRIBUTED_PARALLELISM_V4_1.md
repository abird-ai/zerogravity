# Historical distributed-parallelism design before topology v4.2

**NON-NORMATIVE ARCHIVE · 2026-10-12.** Preserves every detailed original layer/tensor/expert and P/D distributed design note from v2–v4.1 as prior art and historical context. The **current normative** specification is `DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md`, with wire/peer detail in `REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md` and source research at `DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md`. Do not adopt older participant-level `ranks commit` examples where inconsistent with TaskCoordinator's single logical PublishRoot and remote state leases.

---

# Heterogeneous devices, distributed execution, topology and recovery

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


All strategies must preserve one logical Semantic IR and one committed session frontier. Distribution is optional: a lone Strix Halo must never pay for network orchestration when it can run resident.

## 1. Placement modes

**Single device:** one CPU/GPU handles model and state; baseline, no collective tax. **Layer/pipeline parallel:** contiguous disjoint layer groups across devices/machines, one activation handoff per stage, good capacity and multi-chunk prefill; cannot automatically speed single-token serial generation. **Tensor parallel:** matvec/GEMM and possibly attention components split over peers, partial sums exchanged; useful when interconnect fast enough. **Expert parallel:** route selected token rows to expert owner GPUs and aggregate contributions; high benefits for MoE at throughput scale but has dispatch/comm cost. **Prefill/decode disaggregation:** separate prefill workers and decode workers with KV/recurrent state transfer, only when service economics justify.

Heterogeneous plans may combine pipeline across CPU/GPU/machines and tensor/expert within a stage. Composition constrained by state and expected communication cost. Not all variants supported at first release.

## 2. Cost and topology model

`TopologyGraph` devices and edges: device vendor/arch, free VRAM/unified RAM, memory speed, compute capacities by kernel shape, PCIe lanes/generation, P2P availability, NVLink/RDMA/TCP bandwidth and latency, CPU NUMA locality, physical host RAM/SSD. Benchmark representative transfers rather than trusting brochure bandwidth.

For each placement, estimate:
- each tensor/quant packed bytes and mandatory model/optimizer state;
- per-stage compute time for prefill/decode/verify, not just FLOPs;
- weights duplicated vs sharded, expert residency and cache misses;
- per-token activation/collective bytes and launch/stream synchronization;
- per-session KV/recurrent state mapping and concurrent memory pressure;
- context length and speculative states that grow per active session;
- failure/recovery and cold startup.

A larger GPU can receive more layers, but load balance requires actual layer complexity (dense vs sparse vs MoE), not equal layer count.

## 3. Layer pipeline

Assign model layers [0..k], [k+1..m] etc with exact model recipe and subgraph boundary. Each worker owns its slice and state for those layers; coordinator owns input tokens, sampling policy and API. Prefill chunks overlap stages when multiple chunks can be in flight and downstream state consistency allows. Decode one token must pass all layers before next target token sampled; pipeline latency often bounded by sum stage timings and links.

Wire data: model/quant hash, session ID+epoch, token frontier, shape and activation dtype, exact stage IDs, optional modality positions, checksums. Compressing transmitted activations F32→F16/INT8 changes numerical contract; explicit opt-in and quality test.

## 4. Tensor parallel

Two ranks may split routed experts by owner and exchange partial contributions every relevant layer; dense and shared tensors may remain replicated. A peer identity handshake validates checkpoint size/hash, tokenizer/config, quant, positional policy, layer/collective schedule, context size, draft and RNG protocol, hardware capability and runtime version. DwarfStar's `ds4_tp.h` is a useful example of a narrow command protocol and GPU-visible RDMA slab separate from frontend logic.

Some workloads favor sharding attention heads/vocab, others experts only; choose based on operation semantics and device link. Prohibit a rank from applying unacknowledged logits or speculative proposals as committed. All ranks agree on exact collective shape, epoch, sequence and model layer gate schedule.

## 5. Expert parallel

GPU-side router produces token-expert assignments; local/remote expert owners process compact row groups; reduce contributions deterministically per configured policy. Network AllToAll/dispatch overhead must be overlapped with on-device shared expert and independent work where safe. For small per-token batch on slow PCIe, routing remotely can be worse than local CPU or expert fetch; admission should choose measured regime. Keep expert source residency separate from session KV locality.

## 6. Prefill/decode disaggregation

Offload prefill to workers optimized for big GEMM, then transfer **all required session state**, not only KV (GDN/KDA/indexer/history/modality state), to decode workers. Transfer cost and incompatible layout conversions may erase benefit. Prefix cache state versions and ABI must agree. This is not guaranteed feature on every model or backend.

## 7. Wire protocol and transport

Candidate transports RDMA (verbs/RoCE/Thunderbolt if supported), authenticated TCP or local IPC. **No unauthenticated remote serving by default.** Even trusted RDMA systems should bind to allowed devices/networks and reject arbitrary ranks. Control frames use versioned types, checked lengths and content hash; data transfers register explicit memory regions with lifetime leases, not persistent unchecked raw pointers. Streaming failure/partial sends must be distinguishable from complete ACK.

Proposed frame identity:
```text
protocol_version, cluster_run_id, model/checkpoint hash, quant/layout hash
source/destination rank, session_id, state_epoch, operation_id
committed_token_frontier, graph_region_id, expected_payload_bytes
sequence_no, request_nonce, checksum/authentication
```

Validate before allocating network payload. RDMA memory keys/addresses are sensitive; protect logs and handle lifecycle. Use end-to-end optional integrity checks for partial sums/activation hashes during QA.

## 8. Distributed state transaction

```text
coordinator selects operation at committed p
ranks verify same model/plan/epoch and reserve buffers
execute subgraphs/collectives, produce provisional state
collect successful completion acknowledgments
coordinator samples/chooses accept-prefix a
ranks commit same p+a and ack
coordinator emits only committed output
```

If any collective fails, rank times out, memory lease invalidates or checksum mismatches, **the operation fails globally**. Peers must not keep generating independently. Recovery requires new consistent route and a canonical committed prefix snapshot or full verified replay. Leader-only restore cannot pretend the worker's recurrent state is restored.

### Idempotency

Each operation tagged by `session_epoch, request_id, op_seq`. Duplicate/reordered frames rejected or explicitly idempotent. No partially completed MoE allreduce can be reused after reconnect. Cancellation generation must propagate and await quiescence. Session locking prevents concurrent distributed writers.

## 9. Multi-GPU single-host handling

On one host, GPU topology can be heterogeneous. Layer splits often best for incompatible cards or thin PCIe; tensor parallel works better with fast peer access; allow CPU spill only when semantic and numeric kernel supports it and measured performance remains viable. A GPU's reported free VRAM is not enough: reserve per-device graphs, cached weights, temporary prefill and sessions before computing placement.

For AMD Strix Halo, no exposed multiple GPUs assumed unless hardware actually enumerates them; single-device path must remain simple.

## 10. Performance and quality evaluation

Bench C1 and C8+ throughput, single-token latency, 8K/32K/128K prompt prefill, network bytes/step, rank memory/OS pressure, cold reload, snapshots and failure recovery. Compare CPU/GPU partition vs two-GPU vs single-GPU where quant/capacity equal; if multi-host enables larger quantization, classify as **capacity improvement**, not fair raw engine speed comparison. Distributed stochastic sampler and quantized collective changes can alter numerical behavior; record per-rank logits and quality.

## 11. Fault-injection and security

Tests: wrong model/checkpoint, mismatched quant, incompatible draft, wrong collective order, interrupted TCP send, RDMA completion failure, slow worker, rank process crash, network partition, cancellation during exchange, stale registered slab, replay duplicated command, out-of-memory on one rank. In every case either all ranks have a valid committed token frontier or execution stops and safely reconstructs from checkpoint.

## 12. Open choices

Collective/RDMA crate, topology discovery stack, stable protocol version, remote credentials/transport security, per-GPU kernel compiler transport compatibility, TP+MTP combined state handling, P/D staging format, checkpoint restore semantics and which modes should be supported at first distributed release. Requires physical hardware evidence before promising support.

## 2026-10-10 integrated v2 addendum — distributed inference is CIR + model state protocol

Research's DS4 TP/RDMA and layer pipeline dataflow is kept, and project transaction rule controls correctness. A distributed model plan is composed of **SIR region partition**, RIR tensor/layout/quant conversion, **CIR** peer transfer/collective/timepoint commands and explicit resource leases; only rank/session identity and accepted frontier are maintained by the coordinator. The kernel/runtime may use RCCL/NCCL or direct RDMA/TCP as a backend policy, but must not require a non-Rust inference server.

Separate data-parallel **replicas** (independent sessions, no per-layer collective) from TP/EP/PP and prefill/decode disaggregation. v1 should retain single-device fast path unless actual hardware demand and tests justify distributed overhead; P19 is later. Research original v1 multi-tier/TP exclusions are scope recommendations, not product limitation forever. Compile topology-specific plan variants only after measuring PCIe/NVLink/RDMA activation traffic and exact buffer formats, not simply choosing "two GPUs = twice throughput".

**State across ranks:** hybrid model requires transfer/restore of KDA/GDN, sparse/compressed indexer, modality and RNG/grammar state—not ordinary KV alone. CIR collective `TimepointId` and rank epoch/generation must all agree before commit, or fail globally/replay from a common validated checkpoint. Buffer lease pin/overrun, stale RDMA keys, wrong quant/checkpoint/semantic hashes, partial capture failure and slow rank must abort, not publish inconsistent frontier. No public unauthenticated TP port.

A new transport (RDMA, PCIe peer, TCP, future shared-memory interconnect) should plug in as `Collective/TransportProvider` with physical cost and safety contracts, not modify model mathematics, tokenizer or HTTP. Add fault-injection and a 2-rank official model parity test before any parallel performance claim.


## 2026-10-12 integration — one-host heterogeneous computation beyond GPU expert owners

The placement contract must distinguish (A) **dGPU↔iGPU routed MoE expert owners** (activation/partial result transfer), (B) **iGPU↔XDNA2 NPU independent projection output-row slices** (the user's Gufo fork), and (C) multi-host TP/PP/EP/P-D collectives. (B) is **intra-op heterogeneous execution within one model layer on one SoC**, not distributed MPI-like tensor parallelism, and it should not require RDMA, a remote server or multiple model copies.

`TopologyGraph` now includes `Node::XDNA2` with AIE kernel/instruction capability, pinned shared-LPDDR BO import path and nonuniform compute shapes, and `PowerFabricDomain` shared with RDNA GPU. `RegionPlacementPlan` assigns compute owner by *projection slice*, separate from `WeightResidencyOwner`, `KvStateOwner` and the `EngineCore` logical session owner. Per-device `CIR Timepoint` and lease generations bridge HIP/XRT/CPU/SSD and any remote queue, ensuring first-consumer joins and state-commit only after all required partials arrive.

The portable fallback is GPU-only if NPU unavailable/unsupported/unsafe. Automatic NPU enablement cannot be inferred from `device present`; it requires physical power/driver/firmware qualification and numerical validation. A new model may expose a projector that the NPU can accelerate without changes to TCP/remote collective code, while a new NPU generation adds provider capabilities and AIE program schema rather than changing model SIR. See [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

## v4 refinement — epoch fencing and limits of atomicity

The protocol guarantees **logical publication consistency**: coordinator epoch identifies one session StateRoot, ranks execute into provisional private versions and report `Prepared/Ready` only after their device/transport fences complete; coordinator issues one logical publish when required cohort is ready. Late results from old epochs are **fenced**. A rank lost between prepare/commit may require recovery/replay from a retained canonical committed root; device memory and network delivery are **not physically atomic transactions**. Exactly-once delivery would require a separate durable log/ACK/consensus protocol, outside the initial distributed execution path.

TP/PP/EP/prefill-decode split and GPU/NPU on-host operators all use the same staged regions and CIR resource/event contracts. Cross-device KDA/GDN/indexer/adapter/media state must accompany true handoff, not only attention KV; resource transfers carry explicit layout/quant/position identity. Smaller portable single-device plan remains clean with zero remote transport dependency.

## v4.1 — distributed topology includes compiler-owned fabric programs

Separate **device/host execution control** from **chip-internal or collective micro-schedules**. GPU peer ring, NVLink switch, Infinity-style all-to-all, TPU interchip collective, statically mapped wafer dataflow and remote program submission have different topology, compiler capability, cost and failure semantics. A `PartitionedProgram` can own several ranks/chips inside **one compiled region candidate**. TaskCoordinator still owns session admission, attempt epoch, common model/StateRoot schema and single logical PublishRoot after every required rank reports actual Ready/Prepared. Stale/partial remote rank must be fenced/replayed, not considered a successful user-visible commit.

A graph compiler may schedule inside a pod/fabric more efficiently than a Rust host issuing tiny separate operations. The provider publishes a `TopologyId`, `CollectiveGroup`, `InternalScheduleOwner`, artifact/quant hashes, memory/link placement and external async event. **Do not model aggregate rack memory as one dereferenceable GPU pointer.** A remote inference API with opaque state/result is **TaskService interoperability**, not equivalent to native SIR→subgraph lowering or guaranteed compatible KV+RNG snapshots.

Source and platform comparison is contained in prior-art records; normative code uses resource/fabric/collective capabilities. See [backend contract](../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).