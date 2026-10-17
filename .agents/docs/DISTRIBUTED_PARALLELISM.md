# Distributed and heterogeneous execution — v4.2 authoritative index

**2026-10-12 · DESIGN ONLY; no distributed runtime implemented.** This file is the stable subsystem entrypoint. Full *normative* technical specifications are [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [peer/tunnel/state protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md), [universal region execution](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) and [single-root state/worker protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md). Named prior art in [network research](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md). Earlier distributed sketches are superseded by these placement and single-publisher contracts; their original bodies remain in Git history.

## Primary goal

**One AI model/task execution engine** can use a single CPU/GPU, several local accelerators, or multiple explicitly enrolled peer machines (including private encrypted tunnels) by choosing where **compute regions, model parameters, mutable model state and immutable reusable caches** live independently, and selecting verified communication. A distributed session never creates another source-model-specific server, tensor store, TaskDriver, ResourceArbiter or authority to publish output.

**Always preserve a zero-cluster fast path.** One-machine execution should not start discovery/remote heartbeats, peer control sockets or require consensus/transport libraries at CPU/GPU model launch. Distribution is optional and only selected if measured task performance/capacity/energy/cost/rights justify it.

## Four independent placement maps and one physical plan

- **ComputePlacement:** per-layer group, attention core, FFN/MoE, tokenizer/audio/vision stage, tensor/head and token-range partitions, optional whole AOT/fabric regions; one RegionExecutionProvider for each closed region.
- **ParameterPlacement:** source/packed trained weights, adapters, sparse experts and immutable massive lookup rows resident/streamed/cached per device and allowed node.
- **StatePlacement:** per-layer/session KV, sliding/latent/compressed attention, GDN/KDA, indexer/ngram, RNG/grammar/sampler, draft and TaskDriver media state. Remote shard is prepared/versioned, not another coordinator writer.
- **CachePlacement:** content-addressed immutable full-prefix snapshots, encoder features, compiled program caches and replicated model data across permitted peers/storage; directory presence is only a hint.
- **TransferPlan:** ordered/overlapped activations, KV snapshots, state migrations, expert partials and collectives with physical TransportProvider, bandwidth/RTT/priority/memory leases, source/numeric security and cancellation.

Together they are one candidate **PlacementPlan within ExecutableBundle/CIR** tied to a TopologyEpoch and TaskSession's placement epoch, not a new enduring IR.

## Required topologies

| Topology | Main property | Qualification caution |
|---|---|---|
| Single-device colocated | compute and live KV near weights | baseline, minimal latency |
| Local multi-device tensor/expert/model pipeline | peer GPU/CPU/NPU work | actual PCIe/UMA and collective cost |
| Layers on remote machines with their per-layer cache | contiguous layer shards, activation handoff | decode adds serial hops; prefill chunk pipeline can overlap |
| Layer compute and KV/state on different nodes | remote attention near KV or state hydrate once | naive remote full KV reads per token usually slow |
| Attention+KV separate from FFN/MoE | memory-bounded attention vs compute-rich experts | per-layer roundtrips, hybrid layer imbalance |
| P/D disaggregation | prefill produces full compatible state for decode | large transfer/TTFT and destination reservation |
| Token-range KV shards | execute partial attention beside state, merge numerical softmax summaries | only where op PartitionAlgebra proves legality; not generic GDN |
| Remote immutable prefix/encoder cache | reuse state across requests/workers | stale directory/hashes and transfer vs recompute |
| Multi-model/task distributed stage pipeline | encoder, denoiser, text decode, vocoder on different nodes | individual task frontier/privacy/output |
| Multi-GPU rack/static programmed fabric | compiler-owned internal collective/program | rank/epoch fencing and StateBridge |
| Trusted encrypted tunnel peers | private small cluster across NAT/WAN | direct vs relay RTT; overlay isn't RDMA; explicit data authorization |

The legal plan set is bounded by **semantic effects, model/quant/position, state snapshot/restore, numerical quality, device/transport permissions, physical reservations, latency and failure recovery**. The performance optimizer can favor **capacity** even when raw C1 decode gets slower, but must label that tradeoff.

## Authoritative publication and node failure

**Exactly one logical TaskCoordinator writer per TaskSession/shard**. Participating RemoteRegionWorkers execute under SessionEpoch/PlacementEpoch/StateShardGeneration, return Prepared/Ready/Failed and hold safe leases. TaskCoordinator validates all necessary region/state groups, then one logical StateRoot PublishRoot and sequenced output. Device/peer can neither individually commit session state nor emit provisional output as real.

A disconnected peer or stale rank must be fenced; partial DMA cannot be recycled until safe quiescence; failure may require replay from last compatible committed root. Losing coordinator can lose volatile work and leave network delivery uncertain. **Do not promise distributed exactly-once or transparent failover without durable quorum/log/ACK machinery**, which is deferred. Immutable prefix caches may be eventually discovered but cannot override authoritative state shard versions.

## Transport and security

Peer enrollment requires explicit authorized identity and model/tenant/data permissions. **No public remote inference endpoint or arbitrary remote shell by default**. A control plane can connect over authenticated QUIC/TCP through a private tunnel, while data can use a different permitted fast LAN/RDMA or bounded encrypted stream; keys/registered memory regions are scoped to peer/session/lease and never logged.

The planner measures **direct vs relayed** paths, RTT/P95 jitter, actual payload throughput, MTU/packet loss, staging/copy/crypto/egress cost. Tunnel does not confer cache coherence or GPUDirect. Slow relays are inappropriate for many per-token exchanges even when hardware memory is abundant. A remote prefix transfer or small number of layer boundaries may be acceptable.

## Minimal staging and tests

**No cluster implementation before real local model CPU/GPU vertical slice.** Then first distributed milestone: one explicit trusted peer, one remote layer group with its own state, activation boundary transfer, simple reliable encrypted stream, correct TaskSession/root after network delay and peer loss. Second: remote immutable prefix and full hybrid state transfer; third: independent attention/FFN and sequence sharded attention where fast links/model math justify. High-bandwidth collectives/rack, automatic placement/migration and robust leader recovery are **independent later milestones**, not core baseline.

Tests: exact model/logits/state across local vs split; 3 different state domains and proper source/adapter hashes; C1 and C8 P95 on direct LAN/private tunnel direct/relay; wrong quant/state version, partial transfer, duplicate frame, node restart, cancellation, coordinator failure and slow client. GPU-only fallback must remain unaffected.

**Full contracts, costs, wire identity, security and test plans are in the linked normative specs.** Nothing here indicates code or performance has been validated yet.
