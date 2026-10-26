# Source atlas — architectural research preserved for implementers

> **Research/provenance only (v3, 2026-10-12).** Project names, URLs, specific implementation techniques and source-author benchmarks in this document are **attributed prior art, not normative zerogravity runtime or compiler types**. The independent governing architecture is [Modular AI systems](../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [Composition contracts](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md), and [Prior-art license policy](PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md). External projects never become a requirement to clone a model-specific inference engine; technical/measured evidence remains available for independent validation.


Research consolidated as of **2026-10-10**. Source URLs are starting points, not immutable references unless a SHA is given. At implementation time obtain a commit pin, review current code and license, record checked-out SHA and test data hash. External benchmark numbers here are project-author-reported, **not zerogravity results**. Never compare different model family/quant/context numbers as a ranking.

## A. Main engines — direct user selections

### vLLM — https://github.com/vllm-project/vllm
**Study:** V1 token-budget scheduler and request computed-token count, waiting/running admission, KV block/paged cache, prefix lookup, graph capture, per-shape kernel backend selection, tensor/pipeline/expert parallelism, concurrency. Relevant candidate components: the V1 scheduler, KV cache manager, worker/model runner and attention backend interfaces; verify paths after checkout. **Transfer:** token-aware unified request scheduling, typed memory ownership, execution variant selection, clean distributed resource accounting. **Avoid:** depending on Python/PyTorch as zerogravity engine and assuming all vLLM kernels optimized equally across AMD/NVIDIA/model. Core P09/P10/P16/P19.

### SGLang — https://github.com/sgl-project/sglang
**Study:** RadixAttention/radix prefix cache, HiCache GPU/RAM/storage tiers, overlap scheduler, speculative decode (MTP/EAGLE/DFlash...), P/D disaggregation, expert-parallel grouped GEMM and communication overlap. **Transfer:** prefix-aware admission, cache-aware balancing and async execution, independent prefill and decode resource decisions. **Avoid:** conflating KV/CPU prefix-cache tiers with Maya's *model expert* tiers, or believing P/D network split improves one APU automatically. P09/P10/P15/P16/P19.

### llama.cpp / GGML — https://github.com/ggml-org/llama.cpp
**Study:** GGUF format, tensor metadata and quant layouts, model registry and graph build, CPU SIMD, CUDA/HIP/Metal/Vulkan backends, partial GPU offload, mmap and CLI/HTTP ergonomics. Read the pinned GGML and llama model source and GGUF/GGML backend implementations. **Transfer:** common model packaging, many architecture forms, quantized block exactness, native UX, CPU reference/optimized variants. **Avoid:** copy-pasting unsupported backend kernels without licenses, using a general graph interpreter in hot path and assuming a GGUF loader validates mathematics. P03/P06/P08/P17.

### Specialized gfx1151 engine
**Study:** published benchmark/config/quant/quality docs for Qwen3.8-Flash-Next optimized for AMD Strix Halo `gfx1151`. Reported example: ~1584 prefill tok/s at 8K, ~1517 at 131K, ~37.6 short-context serial tok/s and ~55–56 coding-agent speculative tok/s on 128 GB Strix Halo; other published versions/configs differ. Native checkpoint ~62 GiB, n-gram lookup ~47.7 GiB paged, strong attention and mixed quant tuning. **Source access caveat:** published repository does not provide core engine source; proprietary engine/EULA must not be copied or reverse-engineered unlawfully. Use output/perf methodology as black-box reference and independently implement known public model mathematics. P01/P13/P15.

### Maya — https://github.com/mw00/project-maya
**Study:** the GLM fast path, AMD/GLM platform notes, expert source, cache/CPU expert implementation and tests. Based on Strata-derived concepts, adds GLM-5.3-Flash port with KDA, sparse MLA, mHC; GPU device-side (layer,expert)->slot lookup, host mapped route miss ring, pinned staging, async disk reads, per-layer slot partition/generation-safe eviction. Reported Strix 128 GB GLM Maya-S ~210–217 prefill and ~17.8 decode; 12,096 experts nearly all in pool on cited run. Dual AMD two-GPU ~33 tok/s with MTP; not equivalent to single GPU. **Transfer:** detailed cache safety/IO handoff and model-specific reference parity, selective CPU expert lane. **Limit:** not a general multi-client continuous-batching vLLM peer; AMD experimental text-only paths in inspected snapshot. GitHub connector inspection saw main blob SHA `56bbd3af0f14720294a8ff72046a7bf91860bf4e` for indexed results; capture current commit before writing code. P12/P14/P16.

### Project Strata — https://github.com/mach10x/strata
**Study:** originally Qwen3.8-Flash-Next specialized model with VRAM expert cache, pinned RAM and SSD, GPU/CPU expert overlap, large n-gram table, speculation and heterogeneous layer splitting. **Transfer:** dynamic placements depend on PCIe bandwidth, CPU SIMD and batch shape; n-gram lookup has distinct IO policy from MoE matrix weights. **Limit:** specialized runtime; older/newer engines may diverge. P10/P12/P13.

### Gufo — https://github.com/gufo-org/gufo
**Study:** published model benchmarks, quality and experiment documents, the ROCm executor, GDN/sparse attention, quant MoE, GPU state snapshots, MTP sampling, multisequence batch. October 8–9 author-reported `gfx1151` 128 GB Qwen3.8 UD-Q4_K_XL: AR pp2048 ~1700.52 tok/s, tg128 ~25.95 at d0; d128K pp ~1478.66, tg ~24.92; MTP repetitive ~60.39 vs mixed ~31.32; C8 AR aggregate ~108.68. Historical llama.cpp comparator pinned and one warmed sample/point; *not* independent verification. **Transfer:** session-private recurrent/indexer state, speculative rollback per accepted prefix, long context, real workload benchmarks and rejected-kernel experiment discipline. **Limit:** GPU kernels C++/HIP today; check license before code copying. P01/P11/P13/P15/P16.

### DwarfStar (antirez/ds4) — https://github.com/antirez/ds4
**Study:** `ds4.h`, `ds4.c`, `ds4_rocm_compat.cu`, `rocm/ds4_rocm_runtime.cuh`, `ds4_engram.c`, `ds4_ssd.c`, `ds4_tp.h`, platform/model/speculative-decoding/distributed notes. GitHub connector read main whose newest visible commit was `fc80bd695da76ee14bcb86820bf900a5cdfda806` (2026-10-07); current head may differ. MIT `LICENSE` with GGML attribution. Native C/ObjC/CUDA/HIP without full GGML runtime, custom GGUF packs and model graphs, Metal/CUDA/ROCm, DSpark, time-compressed DeepSeek attention (128 raw local window, 4:1 indexed /128:1 unindexed older history), large SSD n-gram/Engram row tables, TP+PP and tool-aware session snapshots. `gfx1151` DeepSeek V4 Q2 prefill tuned ~187.26 → ~292.86 tok/s warm 4K frontier, specific ROCm model and version. **Critical:** Qwen3.8 path documented Metal/CUDA only, **not ROCm** in inspected Qwen docs; `ds4_rocm_compat.cu` returns no support for decode-island graph capture; single-device ROCm eager. Server modes vary between native batches and ordered fallback; concurrency != native batching. **Transfer:** compact narrow Rust API, model-specific graph, SSD tiering and fixed slabs, state and distributed failure contracts. P03/P09/P10/P13/P14/P15/P19.

## B. Complementary engines and tools

### A CPU/GPU heterogeneous engine
CPU AVX-512 + GPU heterogeneous execution, grouped/layerwise prefill and native large-model weights; GLM-5.3 official FP8 fits machines with hundreds of GB RAM, not automatically Strix 128 GB. Integration into SGLang is an architectural lesson about specialization + generic serving. P12/P14.

### A quant-aware inference project
Quant-aware MoE separate decode GEMV and large-prefill grouped kernel paths; its Qwen Flash-Next report measured ~16% prefill gain on cited kernel change. Understand row-count thresholds, per-expert launch overhead, grouped padded GEMM and atomic reduction/numerical tradeoffs. P12/P16.

### A kernel-dispatch library
Kernel library: paged/sparse/MLA attention, MoE grouped kernels, sampling, JIT/autotune by shape. Primarily CUDA/NVIDIA reference for kernel-dispatch architecture. P05/P07/P20.

### A vendor graph compiler
Model graph extraction/recognition (AutoDeploy), fused attention/MoE transforms, device graph compiler, parallel/disaggregated serving. Study legalization and typed graph optimizations; do not require a vendor graph backend. P04/P05/P19.

### Apple unified-memory frameworks
Apple unified memory and lazy tensor execution, useful analogy for CPU/GPU shared addressability. AMD UMA physical behavior is different. P07/P10.

### An independent Rust serving project
Rust inference+serving, continuous batching, many models, a Rust tensor library; competitive dense examples but reported substantial gap on certain Gemma 4 MoE prefills. Lesson: Rust is not intrinsically slow, kernel/operator coverage remains crucial. P00/P03/P09/P17.

### A Rust tensor library
Rust tensors/models and CPU/CUDA/Metal paths. Useful numerical reference/portability ideas; not a substitute for whole server optimizer. P02/P06.

### CubeCL — https://github.com/tracel-ai/cubecl
Rust GPU kernel language + compiler/intermediate representation, CUDA/HIP/Metal/Vulkan/WebGPU backends, tile/reduction/matmul primitives. Alpha/evolving; judge `gfx1151` generated instructions before selecting as mandatory. Need potential low-level architecture intrinsics for peak kernels. P01/P07/P20.

## C. Research interpretation constraints

- **Do not reproduce proprietary engine code**. Public algorithm documentation and original model papers can inform independent implementation.
- Strong source-level examples include Gufo/Maya/DwarfStar; the performance figures are maintained by those projects and may be single-run/config-specific.
- Specialized-engine and Gufo model quantization likely differ; do not conclude engine-vs-engine win from unmatched checkpoint speed.
- Maya 321B total model routes only 18B active parameters/token; working set/residency and quantization drastically affect throughput.
- A CPU/GPU shared address is not free memory bandwidth; unified RAM, GTT and PCIe must be measured separately.
- vLLM/SGLang throughput on datacenter GPUs does not establish same throughput on Strix Halo without kernel parity.
- Model families and tensor formats evolve quickly. Pin sources when building against them. Check legal code/data licenses, distribution rights and trademarks separately.

## D. Minimum reproducibility manifest for future research additions

```text
name / url / checkout SHA / license reviewed on date
model checkpoint + quant SHA / tokenizer + template version
hardware GPU arch / VRAM + physical RAM / driver / power
algorithm source path + call graph / model semantics contracts
bench inputs / context / concurrency / sampling / acceptance
measured prefill, decode, TTFT, P95, memory, error/quality
what remains unverified / differences from other studies
```

On adding a new project, record **transferable mechanism**, **limitations**, **source status**, **performance evidence** and **license restrictions**, then map it to the relevant subsystem spec and experiment rather than simply adding to a long list of names.

## 2026-10-10 integrated v2 addendum — additional first-party compiler and source programs

The canonical **local research corpus** is the external read-only research library and its pinned-revision manifest, containing source-pinned vLLM, SGLang, llama.cpp, Strata/Maya, DwarfStar and Rust ecosystem evidence. Its historical master and implementation plan are **reference research**, not a second authoritative project implementation plan. The Gufo source, modern Gemma4 and Qwen3.5+ requirements in this project are important complementary additions not fully addressed by old council.

**New first-party compiler evidence, accessed 2026-10-10:**
- An ahead-of-time ML compiler's Stream/HAL (external reference removed) — explicit async resources, device affinity, command/timepoints and schedule-allocation, inspiration for small Rust `CIR`.
- A tensor compiler's Relax→TensorIR and MetaSchedule (external reference removed) — split semantic operator graph from target tiling/schedule and measured candidate search.
- MLIR effect/conditional speculation model, https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/ — pure SSA reordering versus mutable resource effects; optimizer legality model.
- A Rust graph-search project and an e-graph compile-search warning (external references removed) — minimal lower-level primitives, symbolic shapes, bounded pure-region rewriting, compilation time as a risk.
- A kernel-dispatch autotuner v2 (external reference removed) — 2026 eager-vs-captured execution can change best tactic; benchmark whole serving regime, not bare GPU device time.
- NVIDIA Rust GPU codegen announcement Sep 8 2026 (external references removed) — a Rust tile compiler (tile stable Rust) and a Rust SIMT/PTX compiler (SIMT PTX early alpha), supplement CubeCL rather than freezing a CUDA vendor C++ kernel escape.
- vLLM Hybrid KV cache docs https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ — group-specific cache hit and intersection; documented implementation limitations inspire generalized 3+ state domain coordinator.

**Research defects flagged:** earlier "48 byte" disk-KV ABI internally sums to 54 and declares impossible 5-byte magic in u16; its "any model any engine runs" fallback scope is logically impossible for unknown ops; dated local box, v1 phase/release mismatches, compiler pin and unapproved license are not authoritative. Use `ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md` for exact D-01…D-20 ruling. Pin exact current compiler/project commits before implementation, verify licenses; the specialized engine source is not public.


## 2026-10-12 integration — distinct upstream Gufo and XDNA2 fork

New user sources: https://github.com/mikealanni/gufo (default fork-enhancements, commit 8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7) and https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/ (October 11, 2026). Upstream Gufo is https://github.com/gufo-org/gufo (main inspected f37fb012281bbd58054b1082933153bbec303106). Treat the fork and upstream as separately pinned sources. The upstream includes Qwen3.8/DeepSeek LLMs, ASR/TTS and emerging image/video models; the fork adds a **different, same-SoC GPU+XDNA2 cooperative prefill** capability.

Verified fork source: partial Qwen3.8-Flash-Next GDN z, attention query/gate, shared-expert gate/up projection rows run on an XRT/AIE NPU while remaining rows and consumer GPU kernels continue. Its NPU GEMM fixed geometry is 4096×2560×1024 using 9 bytes/eight block-float values; HIP-mapped pinned host/XRT userpointer BO, host submit/wait callbacks and first-use output join. Author reports UD-IQ4_XS 16K prefill GPU-only 1,698 vs NPU+GPU 1,821 tok/s (+7.2%), with only 30 greedy prompts checked. This is *not* a decode speed or zerogravity result.

**Safety:** author reports machine-check/host hard hangs during fabric-clock transitions; 12 clean subsequent runs with manual held FCLK do not prove general safety. Never enable automatically on unqualified host or run privileged clock/power tweaks. **Rust policy:** AIE kernel comes from C++/Python source and compiled xclbin, not a Rust-authored kernel; a Rust-owned device codegen path must be proven or stay blocked. Full source/ABI/quant/clock analysis: [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md). Generalized design: [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

## 2026-10-12 addendum — heterogeneous accelerator execution families

**New comprehensive first-party research index:** [ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md](ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md), referencing DGX Spark/GB10, other workstation and datacenter Blackwell/B200/B300/GB200/GB300, Google's TPU6e/TPU7x and graph/PJRT stack, Cerebras wafer-scale CS-3/WSE-3, Groq deterministic LPU, AWS Trainium/Inferentia, AMD CDNA MI350/MI400, Intel Gaudi, Tenstorrent, SambaNova RDU and other graph/dataflow ASIC families. Distinguishes the **Cerberus** adaptive parallel decoding paper (not Cerebras hardware). All external source names, repository licenses, URLs, vendor claims and hardware specifics remain **research-only provenance**.

**Project-neutral adaptation:** [ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md](../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) sets one RegionExecutionProvider across NativeKernel/CapturedGraph/AOTDeviceGraph/StaticDataflow. Source analysis alone is not proof a particular vendor compiler or StateBridge can support the runtime. Opaque model API services are different from native model-region lowering. No SDK or source was installed/copied or tested during this documentation pass.


## 2026-10-12 new distributed state/cache/tunnel research

Full source-linked prior-art study: [DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md](DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md). It compares P/D and KV connector frameworks, remote stateful attention versus remote KV reads, attention/FFN and heterogeneous MoE split, token-range sharded softmax, cache-centric state tiering, consumer peer layer clusters, authenticated TCP/QUIC/direct/relayed encrypted overlays and physically validated RDMA. Source models and published speedups remain external author research, NOT zerogravity achievements.

**Normative derived design:** [DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) and [REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md](../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md). Single TaskCoordinator, Source SemanticProgram, Prepared StateRoot, four orthogonal compute/parameter/state/cache placements and TransferPlan. No mandatory network runtime for standalone.

## 2026-10-12 additional model-router/local CLI and API source research

Source-specific [multi-model router and API research](MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md) compares current local model CLIs, model keepalive/load policy, model-specific settings/presets, built-in multi-model router and official OpenAI Chat Completions/Responses/embeddings and streaming SDK schemas. **These are external prior art**, not source copied or capabilities tested by zerogravity.

Normative independent interface targets: [ModelRouter/ResidentModel lifecycle](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [one-command CLI/config/OpenAI API contract](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). The router handles coarse exact-model admission and equivalent replica selection; TaskCoordinator/DeviceWorker remains the only native model execution/spaces state authority. No model-specific inference engine or gateway required.
