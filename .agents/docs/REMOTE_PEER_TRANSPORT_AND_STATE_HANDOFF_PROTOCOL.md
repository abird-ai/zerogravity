# zerogravity — Remote peer transport, state handoff and failure protocol (v4.2)

**Normative design; 2026-10-12; no code written or network configured.** This is a **small peer/transfer capability**, not a separate distributed inference engine or Kubernetes-like control plane. Read with [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [minimal execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), and [universal backend contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md). Source/vendor mechanisms live separately in [network prior art](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md).

## 1. One process can run standalone or offer a peer role

**Standalone:** no cluster daemon required, no tunnel discovery, no network permissions and no remote state metadata. One TaskCoordinator/DeviceWorker/ResourceArbiter and direct local region kernel.

**Enrolled private peer:** the **same zerogravity library/binary** may run an authorized PeerAgent offering a **subset** of capabilities: advertise hardware/topology, cache verified model weights, execute an allowed closed region, own leased state shard buffers or immutable prefix cache, and transfer data. It is never a second model server or user request authority unless it is specifically the current TaskCoordinator for that user's session. PeerAgent's OS privileges are strictly less than a generic shell; it cannot read arbitrary local files or execute model-defined source.

**Suggested user-facing modes (illustrative, not frozen API):** standalone default; explicitly enroll trusted peer; list discovered authorized peer capabilities; show direct/relayed/overlay path and benchmark; inspect proposed placement; optionally lock/deny nodes and task/data classes; request disconnection/drain. Auto mode is only a **measured routing policy** within enrolled, authorized nodes, never network permission discovery.

## 2. Exact identities and separate epochs

- `ClusterMembershipEpoch`: authorized peer set and identity/certificate generation; dynamic discovery **does not** authorize a node.
- `TopologyEpoch`: observed device, memory, link, direct/relay RTT/bandwidth and fabric capability snapshot.
- `PlacementEpoch`: selected mapping of closed compute regions, parameters, state shards, caches and TransferPlan; frozen for current safe work group.
- `SessionEpoch`: authoritative TaskCoordinator writer generation, task root + last committed output seq.
- `StateShardGeneration`: remote memory allocation, version, owner node and physical lease; combines with logical StateRoot version and model/quant/sampler/state schema fingerprint.
- `TransferSequence`: per-operation reliable integrity-checked chunk stream; late/misordered/duplicate arrivals cannot become a new TaskSession root.

These epochs **serve distinct lifetimes**. A link reconnection may change TopologyEpoch without invalidating source model semantic identity, while moving a state owner changes PlacementEpoch/StateShardGeneration but not mathematical TaskFrontier. If topology changes during a compiled multi-rank collective, the current attempt **cannot silently recompute a new placement**; use safe abort/fence/replay boundary.

## 3. Wire control is small, bounded and authenticated

Illustrative commands (not wire ABI):
```text
Hello(ProtocolVersion, NodeIdentity, CapabilityNonce)
Authorize(PeerPolicy, TenantScope, ModelArtifactScope)
Capabilities(NodeSnapshot, LinkMeasurements, DriverAndMemory)
Reserve(ResourceTicket, MemoryFootprint, Deadline, ModelOrStage)
EnsureArtifact(ContentHash, LicensesAndPermissions)
PrepareProgram(SemanticRegionHash, ExecBundleHash, Shape, NumericMode)
PrepareTransfer(SourceResourceLease, DestReservation, TransferDescriptor)
ExecuteRegion(SessionEpoch, PlacementEpoch, InputBatchRef, StateShardRefs)
ReadyOrFailed(OperationId, CompletedEvents, PreparedStateShardVersions)
CommitNotice(SessionEpoch, CommittedStateRootHash, OutputSeq)
AbortOrFence(SessionEpoch, OperationId, Cause)
RenewStateLease/Release/Drain/Health
```

**All frames** have schema version, checked lengths, source/destination identity, per-session cryptographic-scoped operation authorization, deadline/idempotency token, integrity and replay protections. Unknown commands or optional capabilities explicitly rejected, not interpreted as shell or unsafe model extensions. **No raw GPU pointers, RDMA keys, public network identifiers, session secrets or tokenizer contents** are logged casually.

The control service is authenticated even if traversing encrypted WireGuard/private overlay. Tunnel is connectivity; an adversarial/compromised peer remains a data-access risk. Send only user-approved tensor/weight/hidden-state/activation classes. A new peer cannot steal work from an active session merely by advertising a faster GPU.

## 4. Bulk transfer and transport capabilities

`TransportProvider` is one contained implementation seam, selected by measured link rather than vendor flag:

| Logical mode | Examples | Transport capability contract |
|---|---|---|
| On-host | CPU ↔ GPU, UDS/shared-memory, mapped approved pages | memory lifetime, IOMMU/driver fences, buffer version |
| Trusted local peer | direct Ethernet/Thunderbolt, PCIe/NVLink if physically exposed | peer bandwidth/RTT, memory access/collectives, encryption policy |
| Fast registered remote | verified IB/RoCE/GPUDirect or compatible RDMA | registered segments, revocable memory keys, transport completion/lease quiescence, physically supported fabric |
| General remote | authenticated TCP/TLS or QUIC reliable streams | bounded buffers, multiplex/priority, congestion/RTT/loss and copy/crypto CPU budget |
| Private overlay tunnel | direct/peer-relayed/DERP-type overlay with WireGuard | identical application identity and reliable stream semantics; **overlay not RDMA** |
| Storage transfer | file/SSD/object through authorized IO provider | content hash, ranges, explicit destination physical memory and retry policy |

The **control plane may travel over a private tunnel while bulk data chooses a separate trusted direct fabric** where permissions and physical topology allow. On a WAN overlay relay, the solver may choose only occasional large state transfers or contiguous few-layer stage cuts. Other cases can fall back to local/offline queueing or reject the distributed plan.

Data transfers should carry only the **minimal required** payload. Tensor compression/packing must preserve selected numeric mode and source math. Explicit lossy activation compression creates a new candidate requiring model-level quality tests; it must not be silently enabled to hide a slow tunnel.

**Control/data credits:** reserve memory on receiver *before* sender prepares large payload, negotiate chunk size and bounded in-flight credits, check checksum/hash, release source and destination leases only after actual network/device completion, not after TCP write() returns. Prefer queue priorities for token-critical activations and state before background weights/prefix warming; prevent starvation via quotas. A peer may **decline** transfer if source/tenant/license authorization insufficient.

## 5. Remote model/layer execution flows

### 5.1 Layers and mutable cache colocated on peer B

Node A owns task cursor/sampler/root; B owns model layers [k..m] and those layers' persistent KV/recurrent shard buffers, warmed model tensors and compiled executable. For each token or prefill chunk:
1. A sends activation tensor and stable session/shape/position/StateRoot lease version.
2. B validates and executes its closed layer region on B, updating a **provisional state version**, returning next activation/result and Ready event.
3. Remaining layer groups run as prescribed by the same ExecutableBundle/PlacementEpoch; no B-side independent sampling or task service.
4. A verifies required group readiness and **publishes one logical root**, with B shard handles in that root; B retains committed version under lease and retires old state after actual fences.
5. On failure, no partial public state; replay/reset B from prior canonical shard checkpoint or choose other legal placement.

This retains **weights and KV local to the layer** and only incurs activation transfers at layer-group boundaries. Suitable first multi-machine capacity mode, but serial decode latency scales with number of cross-machine hops.

### 5.2 Layer compute on A, attention/KV state on B

**Alternative A — remote attention:** A sends normalized query/positions/attention mask and required output projection metadata to B. B executes the actual attention **beside KV**, returns a small attention result and PreparedStateShard. A performs dense projections/FFN. Requires exact op/math/quant state and per-layer cross-network latency can dominate.

**Alternative B — state hydrate:** A fetches authorized immutable/committed KV+SSM snapshot once from B, then executes attention locally under new state generation/placement epoch. Transfer bytes large at start but amortized across many future steps. Never use missing GDN/indexer state as a “complete KV” hit.

**Alternative C — direct remote state reads:** only where verified coherent/registered remote fabric can supply enough bandwidth and correct consistency. Relayed tunnel/ordinary WAN should normally reject this mode by cost, not fallback into accidental 10–100 GB KV reads per request.

### 5.3 Compute/FFN/experts on B and stateful attention on A

Attention/state on A where memory bandwidth and cache capacity are favorable; large MoE/FFN model parameters on B. Each affected layer has a bounded activation transfer and result return, with optional GPU-side expert IDs/route compactness. FFN worker doesn't own TaskCoordinator/RNG or KV. Sparse and hybrid models may need different split per layer; compiler proves exact operator boundaries. On WAN high RTT this is usually a *capacity/throughput at larger batches* strategy, not automatically best C1 latency.

### 5.4 Prefill on A, decode on B

Two-phase reservation:
1. prefill A computes candidate full hybrid state, B admission reserves **all destination mandatory state + graph scratch** or explicitly waits until prefill complete with risk/cost recorded;
2. transfer authenticated chunks of KV/recurrent/indexer/ngram/vision/task state, with checksum/schema and event completion; B declares PreparedStateBridge;
3. coordinator publishes new placement epoch and full root including B owners at a legal task frontier; decode begins from that exact prefix, not a text-only KV assumption;
4. partial transfer/cancel/failed B admission returns to old worker or deterministic replay from canonical checkpoint, with no duplicate published tokens.

The P/D split is a candidate for TTFT/ITL isolation, not a guarantee to increase throughput. On a slow tunnel, multi-GB cache migration may make it impractical.

### 5.5 Remote prefix cache and sequence KV shards

Remote **immutable prefix cache** is not a live state owner. It can provide verified content-addressed compatible snapshots with asynchronous partial download and recompute suffix; its cache presence directory can be stale. Publishing live StateRoot remains separate.

For sequence-sharded attention, B/C/D each hold disjoint token-range KV and compute local softmax maxima/denominators/numerators; A combines correctly scaled partials under the exact supported attention op. **Never** try this for unknown sparse/compressed/recurrent operator algebra; one shard may contain causal positions absent under another client's request. Model quality/numerical and attention mask tests are mandatory. A slow tunnel with per-layer reduction may still lose; prefer bounded local/phase plans.

## 6. Recovering from node failures and partitions

**Per-session leader, not universal consensus system:** TaskCoordinator holds one generation-scoped logical writer lease. Peer B cannot become a new task leader because its node heartbeats more recently; promotion/migration requires explicit safe leader fencing or a future quorum/leased metadata store. The initial private 2-node mode may **fail-stop and replay** rather than promise HA.

**Important ambiguous case:** A receives B's Prepared candidate and publishes a new StateRoot, then A crashes before B sees CommitNotice. B must keep its prepared candidate under a bounded lease/retain policy and respond idempotently to a later query for same root/epoch; if no durable root exists, safe recovery can use the previous committed checkpoint with replay, **not pretend user output was never sent or exactly-once delivered**.

**Network timeout does not cancel DMA:** remote device or RNIC may still be writing; physically fenced or quarantined buffers cannot be recycled until actual completion/reset. A stale transfer may not overwrite a new session's reused slot because `AllocationId+Generation+Epoch` checked on receiver.

**Network partition:** candidate state prepared on orphaned rank cannot be committed as a new root, and an old writer with expired epoch must not issue new writes. Control plane leader election/durable consensus is **deferred**, not silently provided by authenticated streams or private overlay connectivity. If a session cannot regain compatible state, return typed recovery failure and retain last known output cursor.

**Peer code trust:** do not execute arbitrary unverified kernels or remote shell commands to repair, “optimize” or join a cluster. Remote program and model artifacts allowed only under signed model/region/source policy.

## 7. Fault-injection and QoS gates

- invalid membership/tenant authorization, wrong checkpoint/LoRA/semantic/numeric/TaskDriver output, driver/backend ABI mismatch;
- direct → peer-relayed → DERP-like path transitions mid task, sudden bandwidth drop and high P95 jitter; verify solver degrades plan only at safe frontier;
- cache directory claims remote prefix but payload disappeared, mismatch or partial hash; recompute/not wrong prefix reuse;
- remote layer B produces correct activation but wrong GDN state version; reject publication;
- full KV transfer on tunnel would exceed deadline; planner must decline in favor of colocated/recompute/remote attention when legal;
- remote attention partial reducer with all-masked shard, lse overflow, global top-k and dense/GQA/rope mask; exact math and tolerance;
- duplicate CommitNotice, lost ACK, coordinator crash after publish, remote worker restart with stale lease;
- actual model+quant on 2 local machines with direct authenticated link, then overlay direct, then relayed path; distinguish capacity enabling from raw t/s improvement;
- 1/2/4/8+ request batches, 4K/32K/128K context, native decode/verify/preload, TTFT/P95/energy/egress/crypto and total task correctness;
- node drain/rejoin during prefilling/media stream, user cancellation, no leaked buffers or unacknowledged output.

**No proof of system durability or performance by simulation alone.** Actual transport and device tests follow user authorization; one model CPU/GPU vertical slice remains earlier priority.

## 8. Minimal implementation discipline

Use a few Rust modules within the existing engine: peer identity/capability, topology snapshot, placement solver, transport adapter, remote work envelope/state shard leases. These reuse existing DeviceBackend, ExecutableBundle/CIR, ResourceArbiter, TaskCoordinator and TaskDriver; **no Kubernetes, etcd, database, DHT, cluster-wide multi-leader election or new serving engine required initially**. Future scale may warrant richer control-plane services based on measured use cases, not as a mandatory project prerequisite.

**Prove a two-node capacity-first layer split before speculative distributed attention or WAN per-token FFN.** Explicit config/trust and low-risk direct encrypted transport first; retain standalone fast path. Higher-level features (distributed prefix directory, attention/FFN disaggregation, sequence KV shards, autoscaling, remote multi-task scheduling) remain independently qualified plan candidates.

**Status:** all implementation and transport gates are PLANNED, none run.
