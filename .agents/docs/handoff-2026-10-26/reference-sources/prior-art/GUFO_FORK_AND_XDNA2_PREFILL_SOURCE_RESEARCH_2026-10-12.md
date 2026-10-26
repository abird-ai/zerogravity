# Gufo XDNA2 cooperative-prefill fork and October 2026 Strix Halo report — primary-source research

> **Research/provenance only (v3, 2026-10-12).** Project names, URLs, specific implementation techniques and source-author benchmarks in this document are **attributed prior art, not normative zerogravity runtime or compiler types**. The independent governing architecture is [Modular AI systems](../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [Composition contracts](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md), and [Prior-art license policy](PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md). External projects never become a requirement to clone a model-specific inference engine; technical/measured evidence remains available for independent validation.


**Research date:** 2026-10-12 (Asia/Singapore). **Scope:** the exact user-supplied [mikealanni/gufo](https://github.com/mikealanni/gufo) fork and [r/StrixHalo 1x3gtdh](https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/) post, compared with [upstream gufo-org/gufo](https://github.com/gufo-org/gufo). This is a READ/SOURCE research record, **not** a zerogravity benchmark or executed NPU test. GitHub was inspected via the connected GitHub app; Reddit was independently read via public search. Distinguish this fork from original Gufo and from a separate dGPU+iGPU asymmetric MoE path.

**Verified GitHub fork ref:** `mikealanni/gufo`, default branch `fork-enhancements`, commit **`8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`** when inspected. Its `README.md` blob SHA was `af75c7a7211a0514935d940ed2374e7a56e75845`. Upstream `gufo-org/gufo` has default branch `main`; the two should be independently source-pinned. GitHub files below were inspected at this exact fork commit. Upstream HEAD and fork may change: re-fetch and diff before code/benchmark integration.

## 1. What the user's sources actually establish

**Author's claim:** 128 GiB GMKtec EVO-X2 Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) with Qwen3.8 Flash-Next **UD-IQ4_XS GGUF**:
- 16,384-token prompt, GPU-only **1,698 prefill tok/s**, GPU + XDNA2 NPU **1,821 prefill tok/s**; approximately **+7.24%** throughput at matched reported benchmark shape.
- 32,768-token prefill **1,783 tok/s** with NPU (no GPU-only value in fork README).
- 40K cold served prompt, MTP+cache: **1,637 tok/s** prefill with NPU (different workload/flags).
- README reports 8,192-token prefill chunks, held maximum GPU fabric clock, NPU performance mode and no competing load.
- Post reports 30 longer greedy prompts + a copy task produced identical *generated tokens* with and without NPU. This **does not prove** bit-identical full logits, sampled distribution, long-context correctness, input rounding parity or corpus-wide quality.
- Post reports prior **hard machine freezes / machine-check failures** when GPU fabric clock changes during concurrent NPU and GPU work; 12 subsequent clean runs after holding fabric clock. This is author experience on a particular firmware/driver/power configuration, not a general hardware fix guarantee. Some replies discuss AGESA version requirements and idle power increases from clock hold.

**Do not misattribute to upstream Gufo:** older `gufo-org/gufo` engine includes `gfx1151` Qwen3.8 optimized GPU inference, MTP, sparse attention, concurrent sessions and other model/media pipelines. The new feature in `mikealanni/gufo` is **partial projection GEMM offload onto Strix Halo's XDNA2 NPU simultaneously with its Radeon iGPU**, using the same *model forward pass*. It is **not**:
- a separate NPU draft language model taking turns with the GPU;
- two discrete GPUs with MoE expert partitioning;
- general XDNA inference of the full transformer/attention/SSM model;
- a GPU decode speed claim of 1,821 tokens/s (the number is **prefill**);
- a guaranteed speedup at every chunk size or on CPU/NVIDIA/Apple.

Sources:
- [fork README](https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/README.md)
- [user's Reddit thread](https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/)
- [upstream README](https://github.com/gufo-org/gufo/blob/main/README.md)

## 2. Architecture in actual fork code

### 2.1 Hardware and programming interfaces

**Compute engines:** AMD RDNA 3.5 GPU and XDNA2 AIE NPU on same Ryzen AI Max+395 SoC. NPU accessed through **Xilinx Runtime (XRT)** and an `.xclbin` hardware program plus matching instructions `insts_*.txt`. NPU AIE source uses a **Python MLIR-AIE layout generator and C++ mixed block-float tile kernel**. The host bridge uses C++ XRT and HIP APIs, controlled by optional CMake `GUFO_ENABLE_NPU=ON`; without it the GPU-only path is available. Fork source components:
- NPU GEMM host-bridge header [blob `2911538b5a8313f0b731d2592e72a7ca859027b0`]
- NPU GEMM host-bridge implementation [blob `289c98808ad14f0ec52ecd96bcb8dd7c7291df91`]
- device model (weight slicing/capability/activation)
- executor (GPU-NPU scheduling, submits, waits and GPU continuation)
- HIP kernels (device-side NPU block-float activation/weight input staging and output decoding)
- CMake NPU optional integration
- AIE program inputs: placed-core generator, mixed block-float kernel, `.xclbin` binary and instructions, and fabric-clock helper.

The .xclbin is *not* written in Rust. Its presence is evidence of a feasible XDNA2 hardware strategy, **not** permission to embed non-Rust kernels into zerogravity. The Python AIE generator carries Apache-2.0 WITH LLVM-exception provenance, the fork overall says MIT; individual file/header/licensing and compiled artifacts need audit before reuse. User's Rust-owned kernels constraint remains explicit.

### 2.2 One fixed NPU GEMM geometry (initial implementation, NOT fundamental model identity)

`NpuGemm::Config` has defaults **M=4096 prompt tokens, K=2560 input channels, N=1024 output channels**. Tile geometry `128x64x128` as named by the binary. AIE network uses **8 columns × 4 rows = 32 cores**. Kernels use a 9-byte group for eight values: one shared exponent plus eight signed 8-bit mantissas, `v8bfp16ebs8` (not exactly E4M3 FP8, GGUF Q8_0 or BF16). Recorded decoding: `value = mantissa * 2^(exp-127)/64`. GPU packs dense activations and Q8_0 expert/projection weights into that NPU-specific layout, including padded tails.

**Model-load prepacking:** the device model extracts selected rows from known Q8_0 projection tensors into immutable, NPU-friendly pinned/XRT buffers. The NPU operator is then fed narrow slices rather than converting the entire model to an NPU graph.

**Shape gate:** most active paths require `n_tokens >= 1024` but also `n_tokens % npu->tokens() == 0`, so the particular fixed kernel actually fires for **4096-token multiples** within max batch and 1–4 active groups. Other batch sizes and unsupported projections keep their GPU path. This is a useful example of **shape- and phase-specific provider capability**, not a general GPU+NPU promise.

### 2.3 Exactly which mathematical projections are split

The NPU is used for three independent GEMM families while the GPU performs the rest of the same layer:

1. **Gated DeltaNet (GDN) linear-attention input projection — `z` rows.** The device model defaults to 6 slices × 1024 output rows of the last part of a 16384-row input matrix; model route checks Q8_0 and `K=2560`. The GPU executes remaining projection/conv/scan while the NPU computes `z`. The NPU output is consumed just before **GDN gated-norm/epilogue**, not before the whole layer.
2. **Qwen sparse/full attention query and gate projections.** Input `[13312,2560]`, `num_heads=24`, `num_kv_heads=2`, `head_dim=256`, `rotary_dim=64` in the fork's guarded path. Default `GUFO_NPU_GATE_SLICES=8`, each slice 1024 output rows = query+gate of two attention heads, or 16 heads for 8 slices. GPU computes the remaining heads/keys/values and can continue indexer work. First NPU-result consumer is the query/gate preparation/attention phase.
3. **Shared expert gate/up projections.** When eligible Q8_0, dimensions fit NPU slice, and `GUFO_NPU_SHEXP` not 0, NPU runs two GEMMs for gate and up while GPU computes routed experts. NPU gate/up outputs feed a GPU-side SwiGLU then GPU shared-expert down projection; the result joins the MoE residual. This is **not moving routed experts to NPU**.

These are legal because the parent projection can be partitioned by *output rows* and the two parts have no dependence until the known join. Correctness still depends on stored quant, model score/order, output scaling and kernel rounding. Such partitions must be modeled in RIR and the resource/timepoint graph, **not hard-coded in Qwen ModelRecipe**.

### 2.4 Asynchronous execution and memory sharing

- NPU weight buffers are prebuilt at model load.
- GPU produces input activations; a GPU kernel encodes them into block-float staged A.
- `NpuGemm::Submit` enqueues HIP `hipLaunchHostFunc` callback into the GPU stream. Callback sends work to dedicated CPU thread, which pushes XRT NPU commands. GPU stream can continue independent GPU kernels; compiler/human schedules the first required result read later.
- `NpuGemm::Wait` enqueues another HIP host callback that waits for worker completion, introducing the dependency at **first true consumer**.
- NPU has **up to four outstanding jobs**; it uses one long-lived XRT hardware context to avoid millisecond kernel/context switches.
- GPU-visible pinned host pages (`hipHostMallocMapped`) are **imported** as XRT user pointer BO. In an isolation mode the NPU has private BOs with explicit copy-in/out around runs. A GPU kernel can decode NPU C block-float streams directly or through a staging/decode kernel, in some cases fusing activation like SwiGLU.
- This is **shared mapped memory with explicit XRT synchronization**, not proof of universally coherent zero-overhead reads or an extra VRAM pool. Count all pinned pages and staged duplicate weights in global physical memory.
- On timeout/XRT failure the fork marks NPU backend unhealthy and aborts affected batch. Timeouts can leave DMA unsafe to deallocate; code retains/quarantines affected buffers rather than freeing them immediately. **Zerogravity must never recycle a possibly active DMA buffer.**

This is an excellent source-level proof that our planned **CIR Async Resource/Command IR** should express execution dependencies between two accelerators in *one host*, including `Pack, SubmitNpu, ContinueGpu, WaitNpu, DecodeOrConsume, Join, Commit`.

### 2.5 Runtime/safety limitations actually observed

Fork README and post require **held fabric clock** on the author's firmware/driver to avoid hard host freezes, NPU `performance` power mode, and disabling runtime suspend to avoid NPU power transition contention. Internally it runs a periodic zero-operand XRT heartbeat to keep the NPU awake (otherwise driver suspend/reinitialize interacts with GPU); code also supports `GUFO_NPU_KEEPALIVE=0`. Root-level clock-hold tooling changes `power_dpm_force_performance_level` and `pp_dpm_fclk`. Do **not** run or recommend automatic privileged tuning at zerogravity startup, and do **not** assume changing frequency is universally safe or permanent.

The Reddit conversation contains reports of increased idle power with NPU/fabric-locked setups and firmware/AGESA-specific behavior. **These are user reports**, not independent root-cause validation. A reported 12 clean repeated runs is far too little for a general-purpose "safe to enable automatically" guarantee. On an unsupported system, the default must be GPU-only without launching NPU jobs or changing firmware/clock/power state. If NPU is unqualified, opt-in experiment with explicit preflight and emergency disabled fallback **before** any simultaneous GPU+NPU command. A software timeout cannot necessarily recover a machine hard-locked by an upstream power-management race.

**Do not assume NPU use improves decoder t/s, serial batch=1, or even prefill after considering 24h power/thermal effects.** NPU offload is justified only where its real critical-path benefit exceeds packing, XRT queue, shared-memory bandwidth and join costs, and the device is proven stable.

## 3. Relation to upstream gufo-org/gufo model pipelines

The upstream Gufo project already demonstrates more than Qwen Flash-Next:
- Qwen3.8-27B, image input, autoregressive and **DFlash2** speculative drafting;
- Qwen3.8 Flash-Next, **GDN/QSA/MTP**, large memory and long-context prefill;
- DeepSeek V4 Flash, **DSpark**, compressed attention;
- Qwen3-ASR speech recognition, **Qwen3-TTS** voice generation;
- developing Qwen-Image-2.1 image generation/editing and MiniMax H3 text-to-video/audio, with separate task models and numerical evidence.
- Native multi-request streaming, cache/quality reports and Nix+ROCm deployment.

**Design lesson**: do not model all these tasks as decoder-only `StepPlan::next_token`. They share SIR tensor/kernel/event/resource core, but different **TaskProgram** loops: LM autoregression+MTP, block-diffusion verifier, encoder-only ASR/pooling, TTS acoustic stream/vocoder, image diffusion/denoise iterations, video temporal/frame steps and multimodal pre/postprocessing. Typed effects, task-local state and `CIR` allow these to be added without rewriting the text server or memory ownership. Media pipelines are future-capability targets, not already functional in zerogravity. Learn numerical tests and measured bottlenecks per task.

See upstream `gufo-org/gufo` [README](https://github.com/gufo-org/gufo/blob/main/README.md), its model guides, benchmark documentation, upstream [TESTING](https://github.com/gufo-org/gufo/blob/main/docs/TESTING.md). The specific fork's new feature remains **XDNA2 partial projection offload**, not a replacement for the entire upstream multi-modal roadmap.

## 4. Observed strengths, weaknesses and potential zerogravity improvements

| Dimension | Fork proof | Not established | Zerogravity opportunity |
|---|---|---|---|
| Math-domain partition | selected output-row GEMMs of three independent projections | arbitrary new model/operator/quant | RIR `SplitProjectionRows` transformation with exact producer/consumer effects |
| Accelerators | gfx1151 + XDNA2 | Intel NPU, NVIDIA NPU, multi-host or other shapes | DeviceCapabilities NPU + generic AcceleratorProvider, no model-specific hardware hooks |
| NPU layout | 9-byte BFP8 (8 mantissas + exponent), fixed 4096×2560×1024 | arbitrary FP8/block shapes or exact BF16 parity | explicit compute QuantEncoding and quality gate, shape-specialized candidate |
| Async handoff | XRT long-lived context, HIP callbacks, shared mapped BO | no host sync overhead, guaranteed safe coherence | CIR release/acquire timepoints, bounded queued ops, source/target resource domains |
| Efficiency | author's +7.2% prefill for one UD-IQ4_XS workload | causal gain independent of clock pin and thermal policy | ABBA same FCLK/temp/power with NPU active/inactive, idle energy and 24h soak |
| Correctness | same generated tokens in 30 greedy prompts | full logits/quality, sampled distribution, long context, mixed prefill/decode | official checkpoint oracle, KL/logit/quality, every partial-join boundary |
| Reliability | 12 clean runs after fabric hold | unattended workstation stability across firmware | OFF-by-default until hardware-qualified, reversible external tuning, no sudo in serving |
| Adaptability | model-specific hard-coded slices | automatic choice for new models | per-model semantic region traits; generic `ProjectionSplitCandidate` planner and fallback |
| Maintenance | fork of C++ engine with AIE C++/Python code | Rust-only compiled kernel path | independent Rust DeviceBackend/KernelProvider; ABI+licensing proof before using XRT/AIE artifacts |

## 5. Required experiments before enabling any NPU offload

- **CPU simulation:** partitioned matrix projection join equals unsplit SIR oracle (full logits and output state), including arbitrary row split, padding, tensor quant scales and fused activation, dozens of shapes.
- **XDNA2 capability probe:** exact XRT version, XDNA revision, compatible precompiled AIE kernel ABI, physical pinned memory/map limits, IOMMU requirement, host driver/power/BIOS reliability and device-specific safety profile.
- **Rust kernel feasibility:** can AIE/NPU program be authored or generated/compiled from Rust-owned source while respecting user requirement? A Rust wrapper around existing `mm_bfp_mixed.cc`/Python AIE kernel is **not** a Rust-source implementation. If no viable Rust-owned provider exists, record as `blocked-by-policy`; GPU generic fallback works.
- **Bridge semantics:** Stage A/B/C bytes and scale/rounding, GPU→XRT shared BO ownership, explicit DMA and fences, host callback effects, timeouts and quarantine; no stale generation, host pointer or swallowed completion error.
- **First-use overlap:** measure actual (T_{GPU remaining}), (T_{NPU pack+queue+compute+sync+decode}), and join/transfer; proof gain equals critical-path reduction, not a second device running in name only.
- **Workloads:** Qwen3.8 Flash-Next exact UD-IQ4_XS baseline 16K/32K prompt under 8192 chunks; C1/C4, cold/warm cache, context depth and verify; GDN projection, attention queries/gates and shared expert toggled independently. Test arbitrary unsupported prompt length (GPU fallback), short one-token decode (GPU), model variants and other quant types.
- **Quality:** layer activation/logit relative error and sparse top-k IDs; 30 prompt same greedy token match is one smoke test, not a release gate. Target sampled speculation distribution, tool/reasoning, NLL/KL and official held-out continuation.
- **Fair performance:** ABBA NPU ON/OFF with **same held clock, NPU power state, driver/quant, prompt, device memory, temperature and CPU load**; compare GPU-only same-clock versus GPU-only default-clock separately. Report throughput, TTFT/P95, energy, pinned memory and system idle draw.
- **Safety:** normal boot/suspend/resume, idle GPU, mixed concurrent GPU and NPU, driver loss/power transitions, shutdown, failure/cancel, and >24-hour unattended soak on an isolated qualified test machine **with no destructive clock or BIOS modifications**. If any hard hang occurs, quarantine configuration; a software watchdog may not recover a frozen SoC. Never automatically run privileged clock-hold tooling, disable IOMMU or install a systemd clock-hold service.
- **Correct fallback:** absent XRT, NPU allocation failure, unsafe clock, no suitable shape, unsupported quant, too much NPU memory, XRT sync failure before job -> GPU compiled plan (from valid committed frontier); error after partial execution -> abort/quarantine/replay, not return incorrect result.

## 6. Source-by-source references and provenance

**User supplied**
- https://github.com/mikealanni/gufo (branch `fork-enhancements`, inspected commit `8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`)
- https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/ (author report published 2026-10-11)

**Fork source (immutable commit links)**
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/kernels/rocm/npu_gemm.cpp
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/kernels/rocm/device_model.cpp
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/kernels/rocm/executor.cpp
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/CMakeLists.txt
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/tools/npu/n32_core_placed.py
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/tools/npu/mm_bfp_mixed.cc
- the fork's privileged clock-hold script at pinned commit `8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`.

**Upstream**
- https://github.com/gufo-org/gufo
- https://github.com/gufo-org/gufo/tree/main/docs/models

**Other complementary research**
- A model-agnostic GPU+GPU/CPU expert owner split — this is a *different* heterogeneous parallelism from the Gufo fork NPU partial GEMM.
- /home/pvl/spaces/abird/src/zerogravity/.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md — target-independent timepoints, allocations and first-use join.
- /home/pvl/spaces/abird/src/zerogravity/.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md — Rust-owned provider and capability contract.
- /home/pvl/spaces/abird/src/zerogravity/.agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md — APU physical memory, GTT, GPU kernels.
- /home/pvl/spaces/abird/src/zerogravity/.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md — source-author figures versus reproducible quality/throughput.

**Conclusion:** the user-supplied fork contributes a *specific and important missing specialization* to zerogravity: **heterogeneous operator-slice pipelining on one APU with XDNA2 NPU**, including its own data-packing, queue, dependency, safety and precision policies. It fits cleanly under SIR/RIR/KIR/CIR and global ResourceArbiter if these contracts are formalized. It is not yet implementable/automatic and must default OFF until Rust-source kernel feasibility and hardware reliability are demonstrated.
