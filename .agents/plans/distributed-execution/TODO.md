# Distributed execution

Provenance commit: `6d74f76`.
Scope: P19 plus P04/P05/P09/P10/P16/P23 extensions.
Contracts: [placement](../../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md),
[peer/state protocol](../../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md),
[accelerator families](../../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## Uncommitted

23Oct architecture reconciliation: [shared execution architecture](../pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md).
Shards and expert placement lower the same semantic program and retain existing
per-device admission/readers/completion. Pin weights separately from mutable
state; no distributed runtime or async expert-cache qualification is claimed.

### Deferred mechanism reconciliation — 21 Oct

Research: [families18–20](../../research/engine-optimization-comparison-2026-10-21/README.md#cross-engine-mechanism-comparison).
No new device/peer/transport execution authorized by this research.

- [ ] Evaluate layer, tensor/head, context and expert partitions as independent
  placement choices under source-specific partition algebra. Distributed full
  softmax summaries and recurrent state are different contracts; changed reduction
  or8/16-bit activation wire formats need explicit numerical profiles.
- [ ] Admit shared collective/IPC/RDMA backing, staging, graph and encoded-reader
  ownership through original per-physical-resource ledgers. UMA host-visible
  transfer is not discrete GPUDirect; port ratings/tunnels do not attest throughput.
- [ ] Later MoE EP/replicas/EPLB/communication overlap: preserve logical expert
  source IDs, topology/version guards, old/new migration readers and rank-agreed
  batch/graph/verification shapes. PDL requires actual eligible hardware; DBO is
  not generic one-GPU prefill/decode overlap.
- [ ] Later produced-state P/D/encoder separation and remote prefix hydrate:
  complete hybrid source/profile/tenant/root-frontier handoff, authenticated
  producer/consumer epochs, completion notifications and failure fencing. Heartbeat
  expiry or peer silence cannot reclaim unknown DMA or publish partial roots.
- [ ] Compare placement against local batching and capacity needs using actual
  directional links/host copies/crypto/collectives/stragglers and offered-load
  tails/goodput. Retain two-device/tiny-model and grouped-pipeline negative cases;
  no universal speedup from more ranks or deferred global-service descriptions.

- [ ] Preserve zero-network standalone execution; add capability/transport/placement seams through the same model/task/coordinator/resource authorities.
- [ ] Qualify optional local CPU/GPU/NPU split projection, expert ownership and prefill/decode placement with actual event/memory/power-fabric safety; Stage N is independent of CPU/GPU release.
- [ ] Begin authorized two-node layer-group plus local-state execution and immutable full-state hydration; then evaluate P/D and attention+state versus FFN/expert disaggregation.
- [ ] Extend tensor/head/pipeline/data/context/expert/sequence parallelism and collectives only where exact state partition algebra, source numerics and transport cost support them.
- [ ] Authenticate/version source/state/tenant/session epochs and complete transfers; loss/timeout/stale peers cannot publish partial roots or release unknown in-flight storage.
- [ ] Qualify topology changes only at committed frontiers and compare collocation against fast/slow/tunneled links, including real transfer/copy/crypto cost and fairness.
- [ ] Add a peer/transport/placement without rewriting model, task, coordinator or API; fleet SSH qualification alone is not distributed inference.

## Committed

### 6d74f76

- [x] Hardware fleet discovery and independent per-host profiles accepted; distributed/heterogeneous model execution is not established by those results.
