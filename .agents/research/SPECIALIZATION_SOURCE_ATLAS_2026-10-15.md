# Zerogravity Specialization Prior Art — Specialist Source Atlas

**15 October 2026 · DESIGN/RESEARCH.** No benchmarks, model downloads, application code, drivers or running inference changed in this research. Read together with the three new authoritative specialization contracts, existing v4.3 architecture and v5.1 ModelAssembly/serving. GitHub sources were inspected through the GitHub connector, not experimentally reproduced.

## 1. Original public sources

| Source | Relevant learning |
|---|---|
| Specialist-source README | 100GB-class derived quantization model packs on 128GB Strix Halo; source quant and speculative verification |
| gfx1151 report | Wave32 WMMA, LDS bank conflicts, MoE packing, batching cliffs, accepted versus proposed tokens, negative optimizations |
| Quant optimizer experiment | per-tensor LDLQ-shaped sensitivity, self-sampled in-domain calibration and disjoint evaluation, constrained bit allocation |
| Measured benchmark methodology | actual prompt classes, contexts, sampling, hardware, single/aggregate rates and numerical FP8 quality controls |
| ROCm decode kernel notes | derived-quantization packed GEMV, gfx12 vs gfx1151 ISA, source fallback and negative kernel optimizations |
| MiMo adaptation/runbook | E8M0 scales, 39 SWA/9 global, DFlash exact attention, layer47 FP16 overflow, UMA page-cache loading risk |
| Qwen serving notes | trained MTP and prompt n-gram, model cache, multiple concurrent sessions and reduction-order near-tie differences |
| GLM serving notes | MLA/KDA/MTP and optional 64MiB BF16 eh_proj, image input, exact tool/think streaming |
| MiMo serving notes | native DFlash target speculation, per-request gating, lossless greedy testing and realistic slowdowns |
| First-image optimization | avoid unbounded first-use MIOpen solver search; legally source-equivalent stride2 patch-merge conv→GEMM |
| Runtime behavior hook | explicit optional activation-space steering, new behavior variant not tensor/kernel equivalence |
| Behavior steering presets | direction+layer strengths and hashes, optional model behavior, safety/quality are separate gates |
| Organization profile | actual source weights and tested AMD hardware scope |
| Qualitative gallery | qualitative agent-made page comparison, NOT controlled numerical benchmark |
| Derived quantization codec/kernel prior art | codec/kernel source prior art, NOT a mandated foreign execution engine |
| Community evidence pointer (supplied by user) | community evidence pointer; text not reliably retrievable in this read-only pass, so no unsupported claims based on comments |

**Pinned inspection note:** observed public Git tree SHA cb0d162ba71d1ff35b0362b73f5c09ef8420eda6; it is a tree digest, **not a Git commit ID**. Source blob SHAs included gfx1151 report c65506b54e7f200945b4985b49ab5e0abde485f0, quant optimizer 9ec2dbe5637f4b32ba4a486cab0c0ab809797372, benchmark methodology 1ae541455f4ba9cd079c83603cb9eb0170ac0552, MiMo runbook 8241095573d532cc6227f2e105e2777e498a37b7. Re-fetch/check exact commit before source-code porting.

## 2. Published benchmark claims, source-specific

Authors report on a Ryzen AI Max+ 395 / Radeon 8060S gfx1151 with 128GB unified memory:

| Derived quant/model | Packed size | Prefill 8K | Speculative decode by prompt class |
|---|---:|---:|---:|
| Qwen3.8-Flash-Next | 95GB | 1481 tokens/s | 45.7–66.8 tokens/s |
| GLM-5.3-Flash | 99.73GB | 634 tokens/s | 26.2–33.4 tokens/s |
| MiMo-V2.6-Flash (1.5) | 105.8GB | 760 tokens/s | 28.9–37.0 tokens/s |

These figures come from upstream README/benchmarks, with differences in prompt categories, context, sampler and measurement method. **Not Zerogravity measurements and not a controlled model-quality-normalized ranking.**

Published quantization-fidelity comparison to original FP8 Qwen reports around 93.81% top1 agreement and KL ~0.0216 on 3084 scored positions for one specialist Qwen derived pack. Its target-lossless speculation refers to **the derived quantized target**, not mathematical equivalence with original FP8 and not guaranteed batching determinism.

## 3. The strongest engineering findings

### Packed GPU kernels
- Direct derived-quantization/QTIP low-bit GEMV with verified RDNA Wave32 WMMA fragment mapping produced a substantial acceleration relative to reconstruct+GEMM fallback on its tested source.
- Source-verified LDS staging stride 5 instead of 4 reduced 32-bank collisions, but making every caller's scratch width match a small MoE kernel broke another 16-warp GEMV; output checks must include model-native complete rows and not just PPL.
- GDN skinny GEMM/quant residual mixing were useful on tested shapes. Some INT8 DOT4 variants and cache/graph capture modifications were neutral or slower.

### MoE and speculative batching
- Representative verify forward cost T_ms about 32.4 + 9.04 × physical rows, due partly to extra distinct routed-expert bytes per additional row. Do not assume extra draft rows are free.
- A grouped-MoE 16-row kernel cliff caused batch B5 at depth2 = 15 physical rows to outperform B6 = 18 physical rows, which triggered another nearly-empty group.
- Single-stream ~47 tokens/s in a favorable measured configuration versus realistic sustained B5 aggregate ~74–75 tokens/s: latency and throughput are different objectives.
- Static depth2 appeared best on one cherry-picked prompt yet multi-prompt runs did not justify changing dynamic policy default.

### Quant and calibration
- 2.05bpw Qwen pack ran around 25.7 tokens/s versus ~44.1 tokens/s for a 3.05bpw variant due to K2 trellis decode/tiling. Less quant storage can cost more execution.
- Experimental per-tensor sensitivity optimizer uses quantizer-shaped perturbation, held-out quant quality and representative trace; explicitly labeled **experimental and untested on sparse models**.

### Exact MiMo and source semantics
- MiMo source variant's 39 sliding-window and 9 global-attention layers use different per-slot fixed rings and global paged KV; codebook/draft also has an independent ring.
- DFlash partial RoPE, layer tap index, bidirectional block, learned mask embedding, sink bias and value scaling cannot be represented as generic default MTP.
- A particular MiMo derived pack had FP16 layer47 intermediate overflow and used paired conversion/runtime rescaling; not universal numerical equivalence and not license to clamp.
- Drafting helped code/reasoning, slowed some prose; model-specific confidence gating and disabled baseline required.

### Unified memory and first-image
- The specialist source's GB10 MiMo loading notes report roughly 86GiB clean page cache competing with about 86GiB GPU weights in one ~121GiB coherent memory pool after buffered read. Driver MemFree is not the same as OS MemAvailable or safe physical headroom. Their external guard included optional known-shard page eviction and process kill under configured threshold, **not recommended as a generic automatic engine policy**.
- GLM's first new image resolution triggered expensive first-use convolution/GEMM autotune searches; a verified patch merge conv→GEMM avoids shape-specific solver searches, with source numerical gating.

## 4. Architecture/source mapping

| Source idea | Zerogravity authoritative home | New spec |
|---|---|---|
| Derived-quantization/QTIP packed trellis/codebook, derived quality, calibration | FORMATS_QUANTIZATION_AND_LOADING, P03/P20 | EXL3_QTIP_AND_DERIVED_QUANTIZATION_CONTRACT |
| Qwen GDN/QSA/PLE; GLM KDA/MLA/DSA/mHC | MODEL_SEMANTICS_AND_COMPATIBILITY, P11–P14 | EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS |
| MiMo E8M0/SWA/v_head_dim/DFlash/FP16 intermediate | MODEL_SEMANTICS_AND_COMPATIBILITY, P11/P14/P15 | EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS |
| MTP sidecars/acceptance, dynamic draft cost | SCHEDULER_BATCHING_AND_SPECULATION, P15 | EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS |
| gfx1151 WMMA/LDS/DOT4, latency shape | DEVICE_KERNEL_AND_CODEGEN, STRIX_HALO dossier, P07/P20 | GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT |
| MoE route-aware row/shape batching | COST_MODEL_AND_AUTOTUNING, P12/P16 | GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT |
| UMA headroom/pagecache/driver pressure | MEMORY_RESIDENCY_AND_IO, P10 | GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT |
| First-image cold shape, tuning | COST_MODEL_AND_AUTOTUNING, P20 | GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT |
| Steering behavior mode | COMPOSITION_AND_EXTENSION_CONTRACTS, v5.1 ModelAssembly | EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS |

## 5. Explicitly NOT adopted

- No mandatory Python/tensor-framework/external serving runtime or model family-specific serving process.
- No universal CUDA cp.async translated verbatim to AMD; choose legal ISA variants.
- No global fixed 16-row batching rule or universal ladder of draft depths.
- No automatic CPU expert offload/graph capture/INT8 activation or deeper drafts without measured source-gated benefit.
- No assumption smaller bitrate improves performance; no new derived quant auto-downshift under memory pressure.
- No automatic process SIGKILL, system-wide page-cache eviction, unapproved ROCm driver edits or user credential downloads.
- No activation-steering preset silently bundled/activated: behavior-changing model mode must be user-selected, audited and separately tested.
- No qualitative gallery comparison used as proof of source model equivalence or objective model superiority.

## 6. Evidence limits and next audit

This research inspected the specialist source documentation and repository tree, not the original full trained Qwen3.8/GLM5.3/MiMo2.6 checkpoints on Zerogravity, and not every GPU kernel's codegen. Before implementation pin model source/derived quant/aux processors, original mathematical reference and source commit, then run scalar/host/original model and real hardware. Source names and engineering conclusions are correctly useful, but physical throughput claims are not transferable across model precision, OS, GPU, firmware or context workload.

**All technical claims from the source study belong in the new specialization contracts, acceptance matrix and handoff; this atlas is evidence, not another engine authority.**
