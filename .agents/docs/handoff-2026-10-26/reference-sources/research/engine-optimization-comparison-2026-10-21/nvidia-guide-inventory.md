# Deployment recipe mechanism dispositions

Coordinator extraction of the independent [NVIDIA recipe report](nvidia-guide.md).
Accessible recipe pin `16dace6ce9afd9ce6384a838e9b08d2bf2291f69`; historical June
deployment, later open PR and current documentation remain separate versions.
These16 entries are source-scoped deployment/capability observations, not16
distinct optimizations or measured Zerogravity gains. Primary citations and
missing evidence are in the report.

| ID | Mechanism / evidence | Local disposition and integration |
|---|---|---|
| NV01 | Two-rank TP across two GB10 UMA systems | Deferred placement/transport; no local RDMA/topology/collective qualification. |
| NV02 | FP8 KV/block256 | Explicit source/state precision profile; no default BF16/F32 cache substitution. |
| NV03 | Mixed routed-expert FP4/FP8/BF16 checkpoint roles | Trained codec/source identity, not uniform FP8 or automatically applicable dense kernels. |
| NV04 | Trained MTP2; later checkpoint uses DSpark | Source-compatible proposal stage after target correctness; versions/head contracts cannot be mixed. |
| NV05 | Full+piecewise graphs/capture ceiling8 | Existing private graph basis; production physical admission/reader/drop and shape gates open. |
| NV06 | Bounded sequences and batched-token budgets | Extend existing fair native workers; qualify temporal prefill and source profile separately. |
| NV07 | Prefix cache | Full-domain exact frontier/COW with source/profile/tenant keys; KV-only hit insufficient for GDN. |
| NV08 | InstantTensor/loader preparation | Compare with existing verified source windows and admitted packing/upload; no executable acquisition. |
| NV09 | CSA/HCA/local windows/indexer/YaRN | Trained DeepSeek semantics deferred; not generic compression for dense Qwen/Gemma. |
| NV10 | RoCE/UCX registration-cache lifetime controls | Later authorized transport investigation; author revised firmware diagnosis and long soak incomplete. No host mutation. |
| NV11 | SM12x fallback/backend selection | Capability-specific kernel candidate; SM121/SM100/SM86 are different support/precision contracts. |
| NV12 | Kernel build/cache checksum and warmup | Source/compiler/driver/ABI keyed admitted artifact cache; moving refs are not reproducible immutable inputs. |
| NV13 |1M versus256K memory/sequence/token profiles | Capacity, concurrency and scratch tradeoff; configured1M limit does not prove real1M correctness/performance. |
| NV14 | Short nonstreaming throughput probe | Use matched original short profiles, genuine token counts/true ITL/all offered work; no1M prefill claim. |
| NV15 | Approved dense Qwen27B metadata includes MTP | Head metadata/tensors are not draft runtime or whole-model support. Dense layer-window capacity path first. |
| NV16 | Gemma assistant sharing/low-QPS opportunities | Qualified trained Gemma target first; architecture-sanctioned sharing and measured negative MoE cases retained. |

See the [cross-engine families](README.md#cross-engine-mechanism-comparison)
and existing numerical/serving/memory/model/distributed plans for acceptance.
