# zerogravity — Distributed placement, network and state execution contract (v4.2)

**15 Oct 2026 — P/D disaggregation and communication/compute overlap:** [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) treats separate prefill/decode stage placement as a **measured QoS/latency isolation candidate**, not an assumed throughput improvement. TransferPlan includes **all source-required** KV, SWA, GDN/KDA, MLA/mHC/indexer, RNG/grammar and media state with model/artifact/quant/epoch/tenant-compatible receipt. Async network/collectives may overlap useful model compute only when semantic effect DAG, physical device events and peer trusts allow it; remote publisher cannot independently alter TaskCoordinator StateRoot. WAN/tunnel RTT and burst tail may make collocated mixed P/D faster; local single-GPU fast path has no RPC overhead. See SERVOPT-28..30 and vLLM prior-art study for current baseline.


**AUTHORITATIVE ARCHITECTURE · 2026-10-12 · DESIGN ONLY.** This makes distributed and heterogeneous execution **a first-class property of one AI runtime**, not a second cluster engine. Existing normative foundations remain the single SemanticProgram, transient legal region/provider lowering, one ExecutableBundle/CIR, TaskDriver, PhysicalResourceArbiter and **one logical TaskCoordinator/StateRoot publisher per session/shard**. The universal [RegionExecutionProvider](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) can span devices or machines as one compiled PartitionedProgram. The detailed [source-attributed network research](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md) is non-normative.

**User goal U-020:** run any *semantically supported model/task* efficiently across available hardware, alone or together, including networked machines reached over explicitly authorized tunnels. Independently place compute layers, model weights, live KV/recurrent state and immutable/shared caches; optimize based on topology, workload, memory, network and failure cost. Support current and near-future execution patterns, yet keep the single-machine default as simple and high performing as possible.

## 1. Central design rule — four independent placement maps

A normal transformer layer contains compute, trained parameter tensors and mutable session state that are often colocated but are **not the same resource**. They must be independently placeable:

1. **ComputePlacement:** assigns closed semantic regions, operators, layer groups, attention cores, routed expert work, encoder/decoder/denoiser stages and partial tensor operations to qualified compute domains. An AOT graph or statically scheduled program can internally own multiple ranks.
2. **ParameterPlacement:** assigns immutable model weights, adapters, sparse routed experts, embedding/lookup rows, packed source tensors and graph program images to memory/storage and possible replicas. The execute owner may differ from the source storage owner.
3. **StatePlacement:** assigns **mutable authoritative per-session state shards** (layer KV, sliding/latent/compressed history, GDN/KDA conv/matrix, sparse indexer, n-gram, sampler/RNG/grammar and modality/task state) to memory and qualified stateful compute. Each has an immutable committed version and a provisional mutation scope.
4. **CachePlacement:** assigns *immutable/reusable* prefix snapshots, encoder features, checkpoint copies and other recomputable cached artifacts to local/peer/remote RAM/SSD/object storage; cache-directory entries are **hints** rather than an authority to publish state. Replicas need content/schema/tenant hashes and leases.

A **fifth derived object, TransferPlan**, connects mismatched compute, parameter, state and cache owners over the physical network topology, with explicit shape, quant, copy, conversion, source/tenant policy and completion events. Together they form **one PlacementPlan**, not four independent uncoordinated global schedulers.

*Why this matters:* “layer on GPU-A and cache on machine B” describes **at least three execution implementations**:
- **remote cache hydrate:** copy B's needed KV/SSM/indexer state to A once, then compute locally;
- **remote stateful operator:** send A's query/activation to a qualified state compute worker on B, execute attention beside KV, return compact output;
- **remote state reads:** A directly pulls B's cache bytes every step; may be valid on fast coherent/RDMA fabrics but generally disastrous across a tunnel or WAN.

The planner must distinguish and cost each; never treat remotely *stored* cache as if it were coherent GPU-local memory.

## 2. Three graphs, two enduring runtime representations

**SemanticExecutionGraph:** existing SIR + TaskDriver stage/region DAG and explicit effect/state boundaries (canonical, device independent).

**ResourceAndPlacementGraph:** a **view within the compiled physical plan** binding compute regions and exact parameter/state/cache resource handles to physical owners, copies/collectives and remote event edges. It is assembled by a bounded placement solver from SIR, resource estimates and current topology. **Not a third durable IR.**

**TopologyGraph:** runtime snapshot of machines, devices, memory, stores, transports, trust/power domains and measured links, with generation/time validity. It may change dynamically; a StepPlan binds a **validated placement epoch**, not a raw mutable global topology map.

```
ModelDefinition + TaskDriver           Hardware/Peer Topology Discovery
            |                                      |
         SemanticProgram (math/state effects)  Node/Link Capabilities
            |                                      |
     legal closed regions -------------------------+
            |                                      |
     PlacementCandidate(compute, weights, state, immutable caches)
            |                 + routing/phase/shape/cost/rights
     compiled ExecutableBundle + CIR transfers/collectives
            |
     TaskCoordinator(session root + placement epoch)
            |
     StepPlan across local and RemoteRegionWorkers
            |
     Prepared state shards + event-ready cohort
            |
     Coordinator alone PublishRoot -> sequenced committed output
```

Only SemanticProgram and physical ExecutableBundle endure as compilation representations. A RemoteRegionWorker is **the same trusted compiled-region executor** deployed on a peer, not a cloned inference server/API/task engine.

## 3. Physical topology, device graph and measured transport

Each node declares a signed/validated **NodeCapabilityRecord** with node identity, trusted execution policy, available executable providers, topology snapshot generation, CPU/GPU/NPU model/driver/compiler/program ABI, physical RAM/HBM/UMA/pinned/headroom, SSD/object cache capacity, link interfaces, OS/cgroup constraints, thermal/power, in-flight queue/resource quotas and optional model/weight caches. **Claims and observed measurements are separately recorded.**

Network **LinkCapability** contains both direction metrics (may be asymmetric), throughput by transfer size and concurrency, RTT and P95 jitter, packet loss/retransmits, MTU, direct vs relay, data-plane eligibility, encryption and host CPU crypto cost, collective/RDMA prerequisites, egress charges and versioned trust policy. Measure continuously/periodically outside the inference inner hot path; do not believe advertised 100G port means effective 100G payload after overlays.

Device and network form **different but connected graph layers**: one physical machine can contain CPU, integrated GPU, discrete GPU, NPU and shared memory with mixed coherency; two machines may connect via LAN, Thunderbolt, PCIe, NVLink-scale fabric, RoCE/IB, QUIC/TCP direct or encrypted overlay. A data path can have multiple hops and bottleneck links.

### 3.1 Separate control plane from data plane

**Control/metadata plane:** peer identity, discovery/approval, capabilities, placement epochs, lease renewal, heartbeat, failure, transfer rendezvous and per-session authority. Over authenticated/tunneled reliable channels, bounded messages. It need not carry every tensor.

**Bulk state/activation plane:** negotiated data transfers of model blobs, per-layer activations, parameter shards, KV/SSM snapshots, expert results, device partial attention summaries and collectives. On an eligible trusted local fabric choose direct peer copy/RDMA where actual NIC/driver memory access and security allow; on general WAN/tunnels choose bounded authenticated stream-based copies, with explicit staging if GPU direct isn't available.

**Telemetry/hints plane:** optional lower-priority link probes/cache-directory updates, never a source of mathematically authoritative state or raw credentials. Do not allow a lossy datagram to become a committed StateRoot or an unacknowledged checkpoint.

### 3.2 Tunnel requirements

Support **existing trusted encrypted overlay/tunnels or direct authenticated peer links** as transport *options*, not core engine security assumptions:
- private direct peer through an encrypted overlay using a standard secure tunneling protocol, with NAT traversal where available;
- peer-relayed or DERP-like relay if direct blocked, automatically **classified as high-RTT/constrained** and re-costed; never silently keep RDMA/collective assumptions;
- authenticated TCP/TLS or QUIC streams as eligible transport implementations with versioned protocol and stream priority, but don't mandate nested QUIC-over-VPN when inferior;
- on-host UDS/shared memory and trusted fast peer/RDMA/Thunderbolt/NVLink/IB/RoCE where qualified;
- explicit user-approved remote endpoints, identity pinning/rotation/revocation and per-node access scope. Discovery **does not authorize** arbitrary execution, user prompt sharing or model-weight redistribution.

**Hard rule:** a tunnel does **not** create RoCE, GPUDirect RDMA, cache-coherent remote VRAM or near-zero P95 RTT. Use buffered send/receive with strict bounds and encryption/authorization unless physical network+driver prove special transports. A relayed WAN tunnel may be useful for capacity-first layer split or asynchronous snapshot migration but usually cannot sustain per-layer per-token attention↔FFN calls at datacenter-like latency.

**Security:** control channel and data channel both cryptographically peer-authenticated/authorized (overlay encryption alone isn't blindly taken as application identity); no open inbound public HTTP inference server by default; no remote code execution or arbitrary shell just because machine joins. Remote worker only runs already trusted, allowed compiled program/model artifact IDs. RDMA memory registration capabilities scoped to approved segments/epochs and revoked on disconnect/fencing; don't log memory keys or raw pointer values. Tenant/model weight rights and user data privacy are explicit placement policy restrictions.

## 4. Placement modes all represented by the same compiler and remote worker

| Mode | Compute assignment | State / weight / cache placement | Typical network usage | Caution |
|---|---|---|---|---|
| **Colocated** | entire model/TaskProgram on one worker | weights, mutable state and hot caches local | none | performance baseline; no network tax |
| **Device-local partition** | tensor/layer/expert parallel across local CPU/GPU/NPU | state per owning op or local shared memory | peer DMA/PCIe | device state/alias/precision |
| **Layer pipeline** | contiguous or noncontiguous legal layer group on distinct machines | each layer's KV/recurrent state **colocated** with its compute by default | hidden activations and stage control at split boundaries | C1 AR decode waits serial hops; throughput via microbatch |
| **Remote immutable cache** | model remains local | prefix snapshots/encoder features stored elsewhere | one-shot hydrate or asynchronous prefetch | transfer can cost more than recompute |
| **P/D disaggregated** | prefill on one node/pool, decode elsewhere | **entire compatible KV+SSM+indexer+RNG/modalities** handed off under resource handshake | bulk full state at handoff | TTFT transfer cost, placement admission |
| **Attention vs FFN** | attention operator on bandwidth/cache owner, FFN/MoE on compute-rich machine | **attention KV colocated with attention**, experts resident with FFN | query/activation/result each affected layer | repeated inter-node RTT; async overlap/batch needed |
| **Remote state reads** | attention compute on A | live KV on B | large KV reads each token/step | often prohibit on LAN/WAN unless true direct coherent/read fabric |
| **Sequence-state sharding** | per-token-range attention partial on multiple state workers | disjoint KV sequence shards, weightless attention nodes | query broadcast plus compact softmax summaries/reduction | exact supported attention math; not arbitrary recurrence/index |
| **Expert sharding** | routed experts across peers | expert weights near their compute, KV elsewhere | selected activation rows, partial sums, routing metadata | tiny decode C1 may lose vs local fetching |
| **Tensor/head sharding** | matmul/attention heads split per layer/rank | weights and state per shard | allreduce/allgather every layer | requires very fast links, generally not relayed WAN |
| **Task/encoder/decoder split** | audio/image encoder, text decoder, vocoder/denoiser separate | stage-specific parameters and state colocated with stage | stage output/features, plus state if needed | distinct output/frontier/codec |
| **Replica/tenant pool** | independent full or part replicas | optional remote full-prefix share | cache location hints and task routing | consistency, memory duplication |
| **Static graph/fabric** | one `RegionExecutionProvider` owns compiled multi-device program | device compiler-owned SRAM/HBM/streamed weights | internal static collective + external program events | don't host-micro-schedule device; StateBridge mandatory |

These **compose**: e.g. node A prefill, node B attention+KV, node C FFN/expert, node D remote prefix object store, with tensor/head sharding within B/C on fast fabrics. A chosen plan must be a **legal and cost-qualified composition**, not an arbitrary Cartesian product enabled without tests.

## 5. Per-layer state vs compute placement is first-class

`RegionPlacement` can be **per layer/per operator group**:
- `LayerOwner(layer_range, compute_node)`;
- `StateDomainOwner(session, layer, state_kind, shard, node, epoch)`;
- `ParameterShardOwner(artifact, layer, tensor, replica, node)`;
- `CacheReplicaOwner(prefix_or_encoder_key, node, immutable_epoch)`;
- `ExecutionRole(node, task_stage_or_phase)`.

State owner is the device/machine physically holding **the authoritative mutable candidate and committed logical version** for that state shard—not a second independent TaskCoordinator. The global session root stores a versioned manifest of references to owned shard versions, with location + lease/fencing and compatible schema. The **TaskCoordinator alone publishes the new root** after required participant Ready responses.

For a stage/layer on a remote node, the minimal execution protocol is the *same* `ExecuteClosedRegion` work with source/versioned input activation, state shard handles and output to the next stage; the receiver validates semantic program, quant/precision, shape/layout, state epoch and physical reservations. A weight shard can be fetched once and retained, while mutable state remains at this layer worker over many tokens. **No full model or whole KV must be transferred per token** for ordinary layer-pipeline.

**Noncontiguous layer placement** is representable if source math and state boundaries allow and cost model tolerates extra hops; prefer contiguous groups when sparse networks because each extra boundary adds latency. Highly nonuniform hybrid layers might justify different per-group owners on fast fabrics.

## 6. Live state separate from compute: three physical plans

### 6.1 Cache hydrate once then local execution

Fetch an immutable compatible prefix snapshot or transfer a committed mutable state version from remote cache owner to compute node. Check `StateCompatKey`, resource reservations, page alignment, transport checksum and secure tenant scope; publish new location/version only at a committed task frontier. The new compute node owns local copy/lease for subsequent steps.

**Use when:** repeated future local compute will amortize a bulk transfer, link fast, and state layout compatible. **Reject when:** state huge, remote bandwidth constrained, recent prompt cache has short remaining useful life or full hybrid snapshot missing; recompute or remote operator may win.

### 6.2 Execute stateful operator beside remote state

Keep KV/SSM or sparse indexer on B and send **small activation/query** from A to the qualified attention/recurrent operator program on B. B executes into provisional state, returns compact output and readiness events. FFN/dense projections can stay on A.

**Use when:** remote cache is large, per-step query/result smaller, B has required math/ISA and per-layer network RTT + queue can be hidden or amortized by batching. **Reject when:** RTT per layer/token makes ITL unacceptable, B only storage without compute, state mutation isn't rollback-safe or slow tunnel relay path.

**For recurrent GDN/KDA,** do not replace exact time-dependent recurrence with ordinary remote softmax. The operator's own StateDomain schema, step ordering and snapshot support govern what may execute remotely. A state-only cache server may **not** be capable of running attention/SSM math; capability discovery must prove it.

### 6.3 Remote live cache byte reads per step

Only legal when physical remote memory access/read consistency/latency and owner leases can satisfy the exact operator, ideally a verified direct high-throughput fabric. Naively streaming full growing KV across ordinary network every token is **not** a default. Planner must estimate bytes and reject when bandwidth/latency costs defeat other placements or violate SLO. A tunnel path generally favors hydrate/recompute or remote compute rather than per-token full-KV reads.

## 7. Distributed sequence attention: compact partial-result composition

For mathematically ordinary softmax attention, token-range KV shards may execute attention partials **at the KV owners**. Let a fixed query have per-shard score values. Each worker produces:
- `m_i`: maximum valid attention logit for its causal/masked shard;
- `z_i`: sum of exponentials of score minus `m_i`;
- `n_i`: vector sum of exponentials times values under the same scale.

Combiner takes global `m = max_i(m_i)`, total `z = sum_i(exp(m_i-m) * z_i)`, total `n = sum_i(exp(m_i-m) * n_i)`, output `n / z`. This is **mathematically the same softmax weighted sum** in real arithmetic when shards partition valid keys with correct position/mask/scale. Finite precision/reduction order may differ; declared numeric and full-model quality gate required. Empty shards (all masked) have special zero contribution and well-defined `-infinity` maximum handling.

**Caveats:** multihead/GQA geometry, RoPE positional phase, causal mask, attention bias, sliding/local windows, latent indexer-selected keys and compressed KV semantics must be honored. Sparse global top-k selection may require global score candidate aggregation before sharded attention; a generic local fixed top-k can be wrong. Full softmax partition doesn't automatically extend to GDN/KDA recurrent state or deep learned compressor/indexer. **Operator registry must declare `PartitionAlgebra` and proof/reference**, else sequence sharding prohibited. This is a **capability extension**, not a baked-in mandatory attention algorithm.

Benefits: KV storage scales across attention nodes, full KV need not move every token, compact query/summary travels instead. Costs: broadcast/collective at each attention layer, P95 RTT, partial state update and memory pressure. May be profitable on fast local/rack fabric for long contexts, not automatically tunneled wide-area networking.

## 8. Placement plan validity, versioning and state migration

`PlacementPlan` binds:
- `SemanticProgramKey` and task stage/regime;
- `TopologyEpoch` and participating allowed nodes/link capabilities;
- `PlanEpoch` and immutable compiled RegionExecutionProvider candidates/program IDs;
- exact ComputePlacement, ParameterPlacement, StatePlacement, CachePlacement and TransferPlan;
- `StateCompatKey`, per-shard `StateShardLease` with ownership/fencing version;
- physical memory and network reservations, output/source numeric policy and tenant security;
- `FallbackPlan`: colocated, simpler layer split, portable provider, migrate/replay/refuse on failure.

A plan stays frozen for a **safe microstep/collective group**. Dynamic topology discovery can propose a new plan asynchronously; switching during an active speculative/remote collective would corrupt state unless a compiled protocol explicitly supports it. Plan changes happen at a **committed TaskFrontier** or an operator-specific snapshot/phase boundary where all old cohort events are quiesced, candidate shards copied/validated and owner leases atomically replaced in the logical StateRoot.

**Two-phase relocation (not a database):**
1. Reserve new owners + transfer immutable/committed compatible model/state snapshots, warm needed program.
2. Verify hashes/state codec and readiness, stop or fence old writer for that region at approved TaskFrontier, publish new placement and new remote state-shard leases in one root epoch. Retire old location only after late operations invalidated/fenced.
3. If migration fails: old authoritative committed root and old qualified placement remain, or reconstruct from latest canonical checkpoint/committed history; never mix half-old and half-new state.

A single cold **KV prefix-cache hint** may be stale without affecting correctness: revalidate bytes/hashes/availability on use, fetch elsewhere or recompute. A **mutable state shard owner** must have stronger lease/fencing epoch and cannot be selected from gossip without verification.

## 9. Minimal control plane and remote worker protocol

**Default installation:** one process on one host, zero remote peer discovery/network threads or unnecessary collective tax. A user may explicitly create an authorized private peer group; node participation is a **capability/authorization**, not automatic trust of subnet/broadcast/tunnel. Early implementation may use configured peer IDs and addresses; optional mDNS/overlay discovery later.

Minimal **PeerAgent** on each authorized machine:
- exposes signed `NodeCapabilities`, version, quotas and heartbeat;
- accepts scoped `PrepareExecutable(artifact/program/hash/shape)`, `ReserveResources`, `ExecuteRegion`, `TransferState/Activation/Weight`, `Cancel/EpochFence`, `Release`;
- uses same local `RegionExecutionProvider`, `DeviceBackend` and `ResourceArbiter` as standalone zerogravity;
- does **not** run a separate chat API/server, model orchestrator, state-root publisher or arbitrary remote shell on behalf of peers.

Illustrative wire identity, **not frozen ABI**:
```text
protocol/schema version + cluster membership epoch + peer identity
model artifact + semantic op + quant/shape/numeric and program hash
session id (opaque capability) + TaskFrontier + coordinator generation
placement plan id/epoch + state shard id/version + resource lease token
region id, input/output shape and transfer descriptors + operation sequence
deadline, cancellation epoch, checksum/authenticated payload and ACK level
```

Encode bounded typed frames with explicit endian/version/limits, replay protection, per-stream quota and cryptographic transport identity. Peer only executes verified/allowed model programs and tensor operations; **joining a cluster is never permission to run arbitrary arbitrary model-supplied code or arbitrary process commands**.

Control plane can use authenticated QUIC/TCP over an existing private tunnel, with coarse session/task metadata. On suitable fabric a distinct data plane can use direct memcpy/IPC/P2P/RDMA. `TransportProvider` chooses physical implementation from measured `LinkCapability`. Prioritize state transfer and token-critical activation over optional cache warming/telemetry via credits. No transfer may outlive its source/destination BufferLease or stale RDMA registration.

## 10. Coordinator, split brain and transport failure

**One logical TaskCoordinator session writer** remains the authority for root publication; it may be placed on a chosen machine independently of model compute. Local peer workers only `Prepared/Ready/Failed` state versions for the same expected `StateRoot`, attempt/placement epochs and required participant cohort. A remote buffer ready event does not mean a new token is committed; task output sequences are emitted **after** coordinator PublishRoot.

**Network partition/leader loss:** losing the coordinator can lose volatile prepared work and possibly recent published outputs not durably replicated. We do **not** promise transparent failover or exactly-once delivery. To permit recovery:
- **SessionLease / fencing epoch** from a trusted coordinator or future quorum/lease service; a node with expired/older epoch must reject writes.
- if supported, persist or replicate **immutable committed checkpoints** and output cursor, with explicit durability level and privacy; reconstruct from last safely known checkpoint and compatible StateCompatKey, replay where exact and affordable;
- in a partition, **fail-stop or serve only sessions whose coordinator lease remains valid**; do not create two independent writers with competing roots; late old coordinator fenced on rejoin;
- worker/collective timeout after partial GPU/remote DMA: quarantine affected buffers and abort prepared candidate, not “release because socket closed”;
- queue/graceful drain/rebalance at safe stage/frontier, avoid hot plan thrash based on transient P95 jitter.

**Consistency distinction:** immutable weight/model/prefix artifacts can be content-addressed and eventually discoverable; **mutable authoritative live KV/recurrent state must be linearizable under its single session-root writer** within the promised volatile epoch semantics. Durable exactly-once distributed output/consensus is optional future scope, not implicit from network tunnels.

## 11. Network-aware cost planner and performance policy

Choose the physically best plan **by task phase/workload, not static number of GPUs**. Input includes actual state bytes and model/quant, activation boundary bytes, model layer FLOPs and measured kernel time, cache hits/age and acquisition cost, output/token/denoise work unit, concurrency/fairness, link throughput/RTT/P95/jitter/loss, transfer staging+crypto+double tunnel overhead, power/energy/egress and failure risk.

Useful qualitative cost model:
```text
Per-step critical path =
    longest dependency-path of device compute + remote region waiting +
    peer control latency + data movement + state snapshot/fences
    minus legally overlapped independent async work

Placement objective =
    minimize quality-constrained P95 task latency / resource cost
    subject to physical/transport/tenant rights/StateBridge/SLO budgets.
```

A transfer `T_move` at minimum includes `bytes / effective_link_rate + RTT_or_handshake + buffer_copy + crypto/IO costs`; **real** asynchronous overlap is measured, not assumed. Multiple separate per-layer interactions multiply link latency; single prefill/decode state transfer has large bytes only once. The measured link and cache transfer are directional and affected by other tasks.

**Plan modes:**
- **Latency**: local/state-colocated, minimum serial network cuts, no unnecessary remote per-token action.
- **Capacity**: contiguous layer groups, cached shards and longer latency acceptable if model otherwise cannot fit.
- **Throughput**: multi-session pipeline, expert/tensor/sequence sharding on fast links, prefill/decode pools and multiple replicas.
- **Cost/Energy**: prioritize existing idle nodes/storage and model sharing, but never violate correctness/SLO/privacy or turn off CPU/GPU safety.
- **Private**: only nodes authorized for specific model and tenant content; remote bridge may be denied even if fastest.

**Minimum performance qualification:** no generic “more machines = faster” claim. Compare all-local same model/quant when it fits, forced offload, layer split, remote cache hydrate, attention/FFN split, sequence shards, P/D and hybrid placement with controlled link emulation and real LAN/tunnel/relay links at C1 and C8+ with long context. A capacity-only win can be useful while decode t/s decreases; label it honestly.

## 12. Distributed placement by task kind

- **Language generation**: layer and attention KV/SSM state, MoE experts, prefill/decode, trained MTP verification and tokens; coherent causal checkpoints and sampler/grammar owner.
- **Long agent context**: immutable full-prefix snapshots shared across requests, decode KV sequence shards, remote attention/core versus local FFN, reducing repeated KV movement; application/tool calls still permissioned outside core.
- **Embeddings/rerank**: stateless or windowed encoders sharded/replicated per batch, remote static graph program likely efficient; no language KV required.
- **ASR/TTS**: audio preprocess/encoder/acoustic/vocoder stages on different machines if feature/output size makes sense; per-PCM/window frontier, link codec/latency, private speaker/audio data.
- **Image/video diffusion**: denoiser/encoder/decoder stages can be placed separately with large latent tensors or frame streams; model weights may dominate RAM, transfer cost vs compilation/step loop and media data rights must be measured. Stage full-program graph on target accelerators when appropriate.
- **Future training/adapter fine-tuning**: ParameterVersion/gradient/optimizer sharding and fault semantics are independent future TaskDriver capabilities; inference sessions pinned to immutable parameter versions, no silent cross-task weight mutation.

A `RegionExecutionProvider` can own a remote closed program across multiple local chips and return one Ready event; the host **must not emulate every internal TPU/wafer graph operation over the tunnel**.

## 13. Security, model rights and user experience

Local-first: no outside hosts used without explicit user enrollment. Maintain **node identity, authorized model/artifact set, tenant data permissions, signed program/source/quant provenance, transport encryption, certificate/key rotation, lease expiry and clear revocation**. Different administrative domains are mutually distrustful by default; no automatic prompt/embedding/model weight export, despite valid tunnel routing. Weight licenses may prohibit redistribution; a host's spare GPU capacity doesn't override model rights.

Monitoring exposes per-peer capability, direct/relay transport, egress bandwidth and cost, actual layer/attention/FFN/weight/cache/state placement, memory reservations, qualifying state schema, observed network P95, output/compute owner, and why a requested topology is rejected or slower. The CLI/SDK can offer simple **one-machine / add trusted peers / auto measured topology / pin group / explain plan** modes, while advanced placement can be declarative/locked. Never expose raw user data or RDMA memory keys in traces. A tunnel peer need not open a public API.

**No implicit uptime promise:** a tunnel relay may drop, NAT change or worker vanish; fail-stop/recover from last valid committed root and report why. No distributed dynamic load balancer, membership consensus database or Kubernetes dependence required for first 2–3 private nodes. Scale control-plane sophistication only with measured need.

## 14. Proof/test program and staged delivery

**Stage D (design/simulator, independent of physical network):**
1. 3 fake heterogeneous nodes, different compute/weight/state/cache placements, exact same SIR/TaskDriver and StateRoot, no cloned inference service.
2. Layer pipeline with KV colocated per layer; capacity-first and C1 serial latency, two chunk prefill overlaps.
3. Remote immutable prefix cache: hit/miss/partial state, full hybrid GDN/SSM/indexer compatibility, transfer once vs recompute.
4. Attention versus FFN: KV colocated with attention node, selected activations/results over slow and fast links, hybrid attention cost.
5. Sequence-token KV shards: mathematically exact softmax shard summary combine and numerical quality, reject unsupported sparse/indexer/recurrence.
6. Per-step remote full KV read negative test: cost planner must reject tunnel/WAN unless measured valid, compare query remote attention.
7. Transport topology: direct LAN vs high-RTT/tunnel peer relay; RDMA capability never inferred from overlay; bandwidth/crypto/MTU effects.
8. Coordinator fail/reconnect, duplicate frames, old-epoch remote writes, lost Ready ACK, network partition, no split brain or uncommitted output.
9. State migration across machines at committed frontier, disabled fast path and recovery old placement, no memory leaks or zombie DMA.
10. Different TaskDriver (encoder/ASR/image) stage placement and privacy/tenant restrictions without new backend server.

**Physical qualification (future, only with permission and accessible machines):**
- 2 local heterogeneous machines on Ethernet or direct high-performance link, actual model/checkpoint/quant and generated tokens, multi-node layer split;
- same through explicitly enrolled private tunnel with direct/relay path fingerprint, measured P50/P95 and energy, correct fallback;
- independent remote cache/prefix + optional remote attention/FFN on verified link, snapshot migration and per-state math reference;
- higher-bandwidth rack RDMA/collectives/topology and eventual TPU/wafer/ASIC compiled partition only if SDK/device available;
- mixed node drain/rejoin/failures without leaked state, at-most-once live delivery attempt or explicit cursor resume.

**Implementation staging:** P00 local CPU/GPU no network overhead → P07 optional transport seam/fake peer → P09 trusted peer enrollment/metadata → P10 per-state handles/cache directory → P16 work planner cost → P19 remote region worker / layer split first, then remote prefix/P-D, attention/FFN and sequence shards on qualified links → P22 matched hardware tests → P23 new topology extension drill. Nonessential service orchestration/consensus/distributed KV datastore is **deferred** until needed. Existing 24-phase IDs preserved.

**Plan acceptance statuses:** ARCHITECTURE_ONLY, SIMULATED, TRANSPORT_TESTED, SEMANTICALLY_VALIDATED, STATE_SAFE, PHYSICALLY_QUALIFIED, PERFORMANCE_QUALIFIED, PRODUCTION_QUALIFIED. Do not mark functional because a network library crate compiles. A model/task unsupported by any remote accelerator must still work in available qualified local plan when possible.

## 15. Architectural completeness test

A new machine with a different GPU, NPU or topology should require **only** NodeCapabilities, RegionExecutionProvider backend and a qualified peer TransportProvider; it must not require a new model math description, TaskDriver, API server, StateRoot coordinator or independent distributed execution engine.

A new placement strategy (KV remote attention, per-layer cache shard, P/D transfer, token-range attention or expert route) should require **only** a legal region/StateDomain partition strategy plus TransferPlan and cost/failure policy, leaving other models/task kinds untouched.

**Core invariant:** distribute *where it helps* and colocate *where it is necessary*, but never trade away source mathematical truth, full causal state, physical memory safety, authenticated remote access or a minimal performant single-host fast path.

## 16. v4.3 — model router load balancing versus distributed one-model sharding

**ModelRouter operates at TaskRequest/model admission; distributed PlacementPlan operates within an already resolved exact model binding.** A single physical `ResidentModel` may shard one checkpoint across several machines (pipeline/tensor/expert/attention/state), whereas two different `ResidentModel` deployments of the same exact checkpoint/quant/TaskDriver/numerical policy form an **equivalent replica group** eligible for request-level load balancing. Two distinct model checkpoints on different nodes are **not** equivalent replicas, even if user alias or API format seems similar.

Route choice factors: source/tenant authorization and model weight redistribution rights, per-peer RAM/VRAM/prefix cache/StateShard locality, measured link direct/relay RTT/P95, cold load/graph warm and KV migration cost, available replica queue, model-parameter/quant source fidelity, task phase and requested SLO. Request/session binding to an already live distributed StateRoot remains sticky unless compatible full-state and placement epoch migration. A helper remote peer executing only one layer doesn't expose its own OpenAI server or independent model routing. When a target/draft/encoder pipeline uses several different models, the parent task reserves each model deployment under one admission budget and separate state/quant identities.

No automatic remote use from a local OpenAI API request: explicit permission to offload user content and licensed model weights required. A runtime may expose many remote qualified models through **one** local loopback/API endpoint, with `GET /v1/models` filtered by current tenant and actual task support. A model alias hot reload never changes active peer-owned StateShard generation or model source. [Router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).
