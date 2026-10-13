# Distributed execution, state disaggregation and tunnel transport: prior-art research

**Research snapshot: 2026-10-12. Non-normative, source-attributed.** Project-specific names are retained here with exact research URLs, and MUST NOT become zerogravity's architectural classes, duplicated inference engines or vendor-dependent type names. Normative distributed design: DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md. No source from prior-art projects was copied unless directly vendored (for example, CubeCL (vendored)), which retains its upstream license; the Zerogravity engine is a from-scratch rebuild informed by prior-art concepts, learnings and research. No network was connected, no cluster benchmark was run, and no external research repository was modified.

## 1. New axes beyond model/tensor sharding

Contemporary distributed AI execution includes **independently placing**: mathematical operations and task stages, immutable weight shards, stateful KV/recurrent/indexer data, full-prefix reusable snapshots, compressed historical token ranges, sparse routed experts, scratch/IO, and transport/program execution locations. The best partition depends on whether computation or memory is the bottleneck and whether moving **activations** or **state** across a link is less expensive. Do not assume a tunnel has datacenter RDMA latency.

Important combinations: colocated layer+own KV, model layers on separate workers, stateful attention at a KV owner with weight-heavy FFN on another, prefill/decode with full-state transfer, remote prefix-only cache hydrate, attention KV token-range sharding with mathematical softmax merge, and heterogeneous GPU/NPU/CPU expert ownership. All must share one semantic model and transaction/ResourceArbiter contract.

## 2. Prefill/decode, KV transfer, cache-aware routing

- [vLLM disaggregated prefilling docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/) explicitly describe independent prefill and decode workers, async KV connectors (NIXL, Mooncake, and other independent connector implementations), and say disaggregated prefill is NOT inherently throughput-improving. Useful distinction: tail inter-token latency/TTFT and placement flexibility versus state handoff costs.
- [Practical source-specific guide published Sep 29, 2026](https://vllm-project.github.io/2026/09/29/disaggregated-serving-guide.html) discusses CPU frontend separation, P/D, remote KV and encoder/attention-FFN. One reported example is roughly 3 GB KV for 10k Llama-70B BF16 tokens; not every model/quant has this size.
- [SGLang P/D docs](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx) explicitly differentiate reserving decode KV slots before prefill versus only once prefill is complete; safe destination admission, deadlock avoidance and latency matter.
- First-party P/D serving architecture documentation documents KV-aware routing and nonblocking VRAM-to-VRAM transfers. An independent routing design describes cache-aware routing, shared prefix from a peer CPU tier and SLO-aware scaling.
- [Mooncake KV-centric paper](https://arxiv.org/abs/2407.00079) and independent multi-tier prefix-cache documentation motivate multi-tier remote prefix caches and cache-aware placement. Research claims are not our benchmarks.

**Transferable mechanisms:** independent prefill/decode or encoder pools; explicit whole-state Schema/StateCompatKey (not only simple KV for hybrid models); source/destination reservation handshake; cache presence hints and cost-aware routing; optional remote prefix hydration; measured P95 and state transfer.

## 3. Attention/FFN disaggregation and remote KV execution

- [vLLM AFD experimental plugin, Jul 23, 2026](https://vllm-project.github.io/2026/07/23/vllm-afd-plugin.html) separately places stateful attention/KV workers and FFN/MoE worker groups. Per-layer activations and execution metadata move, while FFN ranks can scale independent of KV capacity.
- [CMU CSD operator-level disaggregation, May 15, 2026](https://www.cs.cmu.edu/~csd-phd-blog/2026/operator-level-disaggregated-serving/) compares attention memory cost to compute-rich FFN and explains why fixed attention/FFN splits can LOSE efficiency if stage latencies mismatch or hybrid attention layers vary.
- [Attention/FFN provisioning theory, Jan 2026](https://arxiv.org/abs/2601.21351), and [disaggregation design space, May 2026](https://arxiv.org/abs/2605.28302) examine conditional cost/scaling. These are mathematical/simulator/source studies, not universal wins.

**Critical distinction:** locating KV on node B while running attention computation on node A requires continual *remote reads or cache transfers*, and can be disastrously expensive. An alternative is **compute attention on node B, beside KV**, transferring only queries and compact results to node A where projection/experts run. This difference must be explicit in the planner, not a single boolean named offload.

## 4. Token-range KV sharding and long-context elastic attention

- [KernelFlume, Jun 2026](https://arxiv.org/abs/2606.29207) studies weight-heavy projection/FFN nodes and **weightless attention nodes holding token-range KV shards**, query-first attention and inter-layer pipelining. Published physical P99 and projected larger simulations are author reports, not our results.
- [LoongServe elastic sequence parallelism](https://arxiv.org/abs/2404.09526) motivates varying sequence-attention partition degree by context and workload, with cost of moving KV shards and group communication.
- The attention math allows normalized softmax summaries (local max, exponential normalization denominator, weighted value numerator) to be merged across token-range shards using a global max. It requires correct masks, positions, numeric agreement and **special handling or refusal** for sparse top-k, compressed attention and recurrent GDN/KDA—none inherit ordinary full-softmax partition legality by default.

**Generalized mechanism:** separate query/FFN computation and KV state ownership, optionally shard sequence state over attention nodes and send compact query/partial outputs instead of K/V tensors every token. Route updates only at valid committed frontier; session state and compression/position remain typed.

## 5. LAN and consumer-machine distribution

- [llama.cpp RPC backend](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) supports local+remote weight/KV distribution, tensor split, remote tensor cache and negotiated TCP/RDMA links when physical RDMA eligible. Exact source ABI/flags remain prior art; MIT-repo candidate still needs file-level license.
- An independent automatic-discovery and topology-aware sharding design claims heterogeneous peer discovery and topology-aware parallel distribution, RDMA over eligible Thunderbolt systems; an earlier revision includes ring-style contiguous layer groups across network machines.
- Collaborative internet-scale model block execution research and its [ACL 2023 paper](https://aclanthology.org/2023.acl-demo.54/) demonstrate model layers across independent network hosts, including privacy and peer trust challenges. Public, untrusted contributor swarms expose private inputs/hidden states; not a default product trust model.

**Transferable:** a small private node cluster with explicitly authorized peers and dynamic capability discovery, capacity-first layer partition where WAN latency tolerates few stage boundaries, and independent checkpoint/tensor rights. Auto discovery does NOT imply auto sharing user prompts/weights.

## 6. Transport and encrypted tunnels

- [NIXL architecture](https://docs.nvidia.com/nixl/getting-started/architecture/) and [transport matrix](https://docs.nvidia.com/nixl/user-guide/backend-selection/) expose typed registered VRAM/DRAM/File/Object/Block memory segments and nonblocking transfers using local peer links, IB/RoCE/GPUDirect/UCX/TCP/storage according to actual capability. Its implementation is not a required project dependency.
- Mesh-overlay connection-type documentation: direct UDP peer connections, peer relay and DERP relay differ greatly in latency/throughput; all are WireGuard-encrypted. Published performance guidance and hard-NAT warnings document relay throughput and possible head-of-line constraints.
- Rust QUIC library documentation describes encrypted, authenticated multiplexed reliable streams and optional datagrams. QUIC is a candidate wire implementation and not universally faster than tuned TCP or QUIC-inside-WireGuard.
- A VPN/WireGuard/tunnel exposes network connectivity and encrypted traffic, **not** real IB/RDMA/GPUDirect. Physical registered-memory access needs verified NIC/capability/trust; an overlay usually uses stream/buffered copies. Double encryption, relay hops, CPU crypto, MTU and packet loss count in actual effective latency/bandwidth.

**Transferable:** an explicitly trusted peer control plane (node identity/authorization/discovery/health), a data plane whose transport is chosen per physical link, and separate capability for direct, overlay-direct and overlay-relayed paths. Tunnel-only links may be suitable for **layer capacity, remote immutable/prefix caching or migration**, while per-layer per-token attention↔FFN over relayed WAN may be strictly refused.

## 7. Safety, numerical, economic and product boundaries

Pipeline parallel's few activation hops can fit limited links but may worsen serial decode ITL; tensor/attention/FFN per-layer all-to-all needs very fast fabric and workload batching. Remote cache hydration has large initial transfer cost and should be compared to recompute. Remote attention can reduce transferred state but adds query/return RTT. Mixed quant/compute must retain model sampling quality; reduction order can vary.

Failure modes: incorrect model/quant/position/adapter, remote rank loss, split-brain coordinator, expired capability token, canceled session with live RDMA buffers, packet duplication/reordering, stale peer cache directory, dangling device program, tail-latency outliers and secret leakage. Proper single logical StateRoot publication, epoch fencing and replan/recompute from durable checkpoint are needed. A network tunnel cannot guarantee remote machine won't read weights/prompts; use trusted peers and explicit authorization.

## 8. Conclusion

The architecture should represent (a) **logical model/task execution graph**, (b) **physical compute/weight/state/cache storage assignments**, and (c) **network topology and transfer plans**, with no new model-specific runtime. Source projects and performance claims above remain attributed prior art. The ability to *describe* a placement mode is not proof it outperforms colocation; physical link throughput, model/quant/full-state reference, reliability and security must qualify every plan.
