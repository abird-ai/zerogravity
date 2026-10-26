# References and source provenance

*Zerogravity whitepaper supporting bibliography · literature review snapshot: 12 October 2026.*

This bibliography supports the **design motivations and prior-art comparisons** in [the whitepaper](WHITEPAPER.md). Its links are to publications or primary documentation, not to incorporated code. **No claim of authorship or first invention is made for the cited techniques.** Source names appear here as attributed research, not as the names of zerogravity's normative interfaces. Public materials may change after the snapshot date; the exact reference should be revalidated at submission.

## Peer-reviewed and academic research

### R01
Kwon, W. et al. (2023). **Efficient Memory Management for Large Language Model Serving with PagedAttention.** *Proceedings of the ACM Symposium on Operating Systems Principles (SOSP).* https://doi.org/10.1145/3600006.3613165 — Dynamic state paging, efficient memory sharing and batching. Published performance improvements belong to the authors and their test setup.

### R02
Dao, T., Fu, D. Y., Ermon, S., Rudra, A. & Ré, C. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** *Advances in Neural Information Processing Systems.* https://arxiv.org/abs/2205.14135 — Exact attention tiling and hierarchy-aware memory traffic. Distinguish exact dense attention from its separately described approximate block-sparse extension.

### R03
Leviathan, Y., Kalman, M. & Matias, Y. (2023). **Fast Inference from Transformers via Speculative Decoding.** *International Conference on Machine Learning.* https://proceedings.mlr.press/v202/leviathan23a.html — Target-distribution-preserving speculative sampling under specified algorithmic conditions. The paper's speedups are not measurements for zerogravity.

### R04
Zheng, L. et al. (2024). **SGLang: Efficient Execution of Structured Language Model Programs.** *NeurIPS 2024.* https://arxiv.org/abs/2312.07104 — Structured language program execution and prefix-aware cache reuse. Distinct from zerogravity's deliberately minimal, trusted TaskDriver rather than an embedded arbitrary workflow language.

### R05
Zhong, Y. et al. (2024). **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.** *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’24)*, pp. 193–210. [Peer-reviewed proceedings](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin); [preprint](https://arxiv.org/abs/2401.09670). Prefill/decode interference, SLO-based placement and transfer costs. Published benefits depend on workload/interconnect and are not reproduced here.

### R06
Lattner, C. et al. (2020). **MLIR: A Compiler Infrastructure for the End of Moore's Law.** https://arxiv.org/abs/2002.11054 — Motivation for typed compiler layering and target extensibility. Zerogravity does not propose shipping that compiler as a required runtime component.

### R07
Zheng, L. et al. (2020). **Ansor: Generating High-Performance Tensor Programs for Deep Learning.** *14th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’20)*, pp. 863–879. [Peer-reviewed proceedings](https://www.usenix.org/conference/osdi20/presentation/zheng); [preprint](https://arxiv.org/abs/2006.06762). The later auto-scheduler introduction is an implementation background source, not the publication venue. Motivates search, real target measurement and bounded optimizer complexity; no claim a generated kernel always beats hand-written code.

### R08
**LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism.** https://arxiv.org/abs/2404.09526 — Elastic sequence parallelism and dynamic KV/state placement for long contexts; no automatic generalization to recurrence.

### R09
**Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.** https://arxiv.org/abs/2407.00079 — Distributed context caching and admission/transfer, with source-author benchmarks.

### R10
**KernelFlume: Elastic Core-Attention Scaling for Agentic Long-Context Decoding.** https://arxiv.org/abs/2606.29207 — Weight-vs-attention-state separation and token-range KV ownership. Reported experimental and simulated results must be distinguished.

### R11
**Cerberus: Efficient Inference with Adaptive Parallel Decoding and Sequential Knowledge Enhancement.** https://arxiv.org/abs/2410.13344 — A *parallel decoding algorithm*, not the similarly spelled wafer-scale hardware manufacturer. Any exact sampling-distribution claim requires algorithm-specific proof.

## Primary compiler and hardware documentation

### R12
**StableHLO compatibility and dynamism contracts.** https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md and https://github.com/openxla/stablehlo/blob/main/docs/dynamism.md — Versioned operator and bounded shape semantics, used for research into compiler interoperability. This is not a required zerogravity binary format.

### R13
**Stream-dialect compiler IR.** Research on resource lifetimes, asynchronous work and explicit timepoints. No source-dialect or runtime dependence implied.

### R14
**NVIDIA Inference Xfer Library (NIXL), overview and backend selection.** https://docs.nvidia.com/nixl/getting-started/overview/ and https://docs.nvidia.com/nixl/user-guide/backend-selection/ — Transfer abstractions across DRAM/VRAM/storage, transports and registered-resource constraints. The mere presence of an encrypted tunnel does not create physical RDMA capabilities.

### R15
**Google Cloud TPU7x (Ironwood), architecture and topology.** https://cloud.google.com/tpu/docs/tpu7x — Compiler, on-chip memory and interchip placement motivation. Hardware marketing specifications are not zerogravity performance results.

### R16
**Cerebras Software Documentation, How Cerebras Works and Weight Streaming.** https://training-api.cerebras.ai/en/1.7.1/cerebras-basics/how-cerebras-works.html and https://training-api.cerebras.ai/en/1.8.0/wsc/cerebras-basics/cerebras-execution-modes.html — Graph-compiled wafer-scale execution and weight-streaming prior art, with source caveats about training modes and hardware access.

### R17
**NVIDIA Blackwell native FP4 and inference precision.** https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ — Research motivation for distinct low-bit block-scaling and reference quality, not blanket compatibility of all FP4 variants.

### R18
**NVIDIA DGX Spark product specifications.** https://www.nvidia.com/en-us/products/workstations/dgx-spark/ — Coherent CPU/GPU LPDDR system profile, distinct from datacenter GPU HBM/rack topologies. Vendor headline throughput is not a measured zerogravity benchmark.

## Serving, APIs, local user experience and networking

### R19
**Ollama CLI, Modelfile and model residency FAQ.** https://docs.ollama.com/cli ; https://docs.ollama.com/modelfile ; https://docs.ollama.com/faq — Local one-command interaction, named model presets, load/unload and memory-bound multi-model residency as user-experience prior art. Exact source licenses/implementation revisions require review before reuse.

### R20
**llama.cpp native server and model router documentation.** https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md — Published router mode, model aliases/loaded models and compatibility APIs. Zerogravity's proposed design deliberately retains one shared model executor rather than using an existing runtime as a mandatory implementation dependency.

### R21
**OpenAI API official reference**, including [Chat Completions](https://developers.openai.com/api/reference/chat-completions/overview), [Responses](https://developers.openai.com/api/reference/responses/overview) and [Responses streaming](https://developers.openai.com/api/docs/guides/streaming-responses). Public API compatibility is a **versioned, tested endpoint-and-feature subset**; a locally built endpoint cannot claim every cloud-only feature or execute tools merely by naming itself compatible.

### R22
**Encrypted overlay network direct/relay distinctions.** Link latency and path-dependent throughput important for private tunneled peer planning. Overlay networking is not equivalent to GPU memory coherence or RDMA.

### R23
**Rust QUIC transport documentation.** Optional reliable stream/datagram implementation candidate for authenticated, bounded peer transport; it is not an architectural dependency or a guarantee of better performance on every link.

## Additional project research dossiers

The historical research corpus provides deeper contemporaneous implementation comparisons and license records that should be retained as **non-normative**:
- [Prior-art license and provenance policy](../.agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md)
- [Compiler/runtime comparison](../.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md)
- [GPU/NPU cooperative compute source report](../.agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md)
- [Accelerator family research](../.agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md)
- [Distributed placement and tunnel research](../.agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md)
- [Model router/API comparative research](../.agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md)
- [Original source atlas](../.agents/prior-art/RESEARCH_SOURCE_ATLAS.md)

**License/attribution policy:** reference content is cited and analyzed, not copied into executable source. Actual use of third-party code, compiler/runtime programs, pretrained weights, model assets and datasets must be separately reviewed at an exact revision/file and comply with the user's Rust source-ownership policy. Patent and trademark questions are independent of repository root licenses.

**Literature limitations:** this is a selective engineering bibliography, not an exhaustive survey or a claim that the combined system is the first implementation of a technique. The underlying architecture is a proposed independent design and all performance hypotheses require reproduction.
