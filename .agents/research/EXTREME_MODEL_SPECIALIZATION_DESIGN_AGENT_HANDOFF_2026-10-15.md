# Zerogravity — Complete Specialist-Source Specialization Design Handoff for External Agents

**15 October 2026 SGT • DESIGN/PLANS ONLY; no application/kernel/CI/model/environment changes.**
**Repository:** /home/pvl/spaces/abird/src/zerogravity, dotlink-l5 root /home/pvl/spaces/abird.
**Observed baseline:** clean main at 4a1c9fdac46fc94665fb8e22b24a284e73f9497d, 18 commits ahead of origin/main. Revalidate current HEAD and dirty/staged status before implementation, and do not push.

## 0. Progressive full reading order

1. Original governing v4.3: .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md, COMPOSITION_AND_EXTENSION_CONTRACTS.md, MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md, TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md.
2. Original quant/model/device/serving: .agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md, MODEL_SEMANTICS_AND_COMPATIBILITY.md, OPERATOR_AND_FUSION_CATALOG.md, DEVICE_KERNEL_AND_CODEGEN.md, STATE_CACHE_AND_TRANSACTIONS.md, SCHEDULER_BATCHING_AND_SPECULATION.md, COST_MODEL_AND_AUTOTUNING.md, MEMORY_RESIDENCY_AND_IO.md, STRIX_HALO_OPTIMIZATION_DOSSIER.md, NUMERICS_QUALITY_AND_BENCHMARKS.md.
3. **New exhaustive specialization designs (read fully):**
   - .agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md
   - .agents/docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md
   - .agents/docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md
4. **Original evidence:** .agents/research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md, then inspected upstream specialist-source links.
5. **Implementation/test phase plan:** .agents/research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md and QTIP-01..14, HYB-01..18, GFX-01..18.
6. Prior v5.1 multimodal/realtime: .agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md, VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md, REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md.
7. Complete previous decisions and history: .agents/plans/DECISION_REGISTER.md, FULL_COMPREHENSIVE_IMPLEMENTATION_HANDOFF_2026-10-14.md and COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md. Read large documents progressively in full, not just headings, to preserve original design and work.
8. Current live Rust source/driver/actual model SHA, tests and selected runtime features. Later agent work must observe current Git independently.

## 1. User intent and decision provenance

The current user directive, following a detailed read-only specialist-source review, is to strengthen specialization contracts and implementation plans **cleanly, robustly and elegantly**, retaining architecture simplicity.

Binding user intent from original conversation:
- One Rust-owned inference engine and one TaskCoordinator/root publication owner; no hidden external/Python serving product.
- Source-mathematical + physical kernel/sidecar/codec/hardware-specific optimizations only when accurate/effect-safe and evidence qualified.
- Multi-model built-in router and heterogeneous devices, model-specific extreme sparse/hybrid/flow/audio/video/decision specializations.
- Current practical Qwen3.5 small-model/Gemma/Qwen3.8 CPU-first priority and 8K real gate retained; GDN 8K accuracy investigation independent.
- No default lower-quant substitution, unapproved model/driver downloads, arbitrary system memory purge, unexpected external egress, fabricated performance or GPU state.
- This task changes *plans/docs* only, not Rust/kernel tests/build/model artifacts.

The integration plan's **SYK-001..SYK-010 are DESIGN PROPOSALS**, not yet frozen public ABI; **SYK-011..SYK-016 are OPEN** licensing/precision/SDK/headroom/policy questions. Do not silently mark them user-approved or solved. Historical decision-number collision: the shared decision register already uses U-040 for a separate shared-model-source implementation, while the earlier v5.1 addendum independently reused U-040/U-041 for multimodal user requests. This handoff uses unique SYK IDs to avoid overwriting/renumbering authentic user history. A future provenance-only cleanup can reconcile under one authoritative index with explicit original IDs.

## 2. Architectural verdict: no extra inference engines

Existing v4.3 already supports verified source semantics, two durable identities (SemanticProgram and ExecutableBundle), transient RIR/KIR/CIR compiler views, source/quant shapes, one ResourceArbiter, device-local leased scheduling, model residency and route/cost adaptation, one TaskCoordinator and target-authoritative speculation.

The precision refinements live in:
- independent derived-quantization/QTIP/codebook/TensorEncoding and derived artifact/calibration manifest;
- MiMo2.6 separate exact ModelRecipe with E8M0, SWA/global and logical key/value dimensions plus source overflow-scaled derived pack;
- trained auxiliary DraftSource variants: MTP, DFlash, independent smaller model and prompt/PLE lookup, with exact source tap/mask/attn and target verification;
- gfx1151 guarded ISA/WMMA/LDS/DOT4/int8/source Q4-QTIP quant fused GEMV/GEMM and GDN/KDA/MLA/mHC region codegen;
- physical expert-row/padding/verification scheduling cost, measured net accepted output/p95/energy;
- OS/driver co-owned unified-memory headroom, file page-cache pressure, model/draft/cache and physical buffer safety;
- bounded cold-start codegen/conv layout avoiding unknown image resolution multi-minute tuner;
- quality receipts for original vs derived artifact, same-derived quant CPU/GPU, target-only vs speculative target and multi-session numerical behavior;
- optional explicit steering treated as behavior-changing source variant, OFF by default, separate quality/safety assessment.

The application remains one model engine, one router, one per-session state authority, one physical admission and one execution fabric.

## 3. Precise researched lessons to preserve

- **Architecture-specific GEMV**: gfx1151 RDNA3.5 Wave32 WMMA vs CUDA/gfx12; fragment mapping, register/LDS bank width, 4→5 staging padding reported useful for one kernel but not universal; wrong small scratch breaks 16-warp caller and may leave PPL misleading.
- **Quant cost**: lower 2.05-bpw K2 trellis Qwen pack was measured slower than 3.05-bpw K3 pack in the inspected source. Stored bytes and matrix throughput are separate objectives; source derived artifacts are separate quantized models.
- **MoE row cost**: an inspected source configuration had T_ms≈32.4+9.04×verifier_rows, with 16-row grouping; B5 depth2 = 15 rows vs B6 = 18 introduced expensive padding. This is measured one-kernel evidence, never a hardcoded SIR batch constant.
- **One vs many sessions**: authors' ~47 tokens/s single stream and ~74–75 aggregate under a representative full batch measure different objectives; source model/quant and output prompt categories affect comparison.
- **Exact Qwen/GLM/MiMo**: Qwen GDN/QSA/PLE; GLM KDA/MLA/DSA/mHC; MiMo E8M0 block scale/SWA/global/value-head dimensions, all source-specific state and attention.
- **DFlash**: partial RoPE, learned mask, bidirectional right-window, sink/value scaling and hidden tap index; a broken draft may merely be rejected by a correct target and waste performance, so independent draft semantic goldens and acceptance are mandatory.
- **FP16 overflow**: one MiMo layer47 converted pack used matched conversion-time/interm_div and runtime scaling instead of blindly clamping. This is a derived pack variant with numerical/quality proof; no universal magic transform.
- **Physical coherent RAM**: on GB10, the inspected source reported 86GiB clean source shard page cache contending with ~86GiB GPU-resident weights in 121GiB coherent pool. Linux MemFree, MemAvailable, GTT max and actual pinned allocations differ; no invisible SIGKILL or global page-cache discard.
- **First-image cost**: source-equivalent stride2 patch merge convolution→GEMM can bypass slow MIOpen shape search; no unbounded first-request autotune.
- **Negative results**: CPU MoE offload, some int8 decode/prefetch/CUDA graphs, lower bitrate and oversized draft trees can be neutral/negative. Do not auto-enable.
- **Behavior steering**: behavior presets alter the model's residual activations; not source-equivalent optimizer and not an automatic default.

## 4. Implementation support truth

At this design snapshot native Rust Qwen3.5 0.8B/2B CPU Phase1 16/16 through 8K passed, four 16K/32K contexts deferred; physical Vulkan selected Q4 source regions on AMD and NVIDIA are source qualified under safe precision. Shared immutable model/session code and physical measurements were independently committed before this design update. **There is NO complete native Qwen3.8, GLM5.3 or MiMo2.6 model, no derived-quantization codec, no DFlash/MTP target verifier runtime, no full Qwen3.5 GPU GDN/attention/KV and no production audio/video/realtime support.**

The new 50 tests are proposed **not run**. ModelRecognition is weaker than source math, CPU reference, GPU region, whole-GPU, streaming and measured realtime tiers. Current source/quant/model status must be revalidated live rather than guessed from docs.

## 5. Phased implementation when authorized

- P03 verified original/derived artifact/derived-quantization codebook plus optional offline calibration identity and quant reference.
- P05/P07 native kernel families, gfx1151 WMMA/LDS/codebook/physical event numeric/evidence.
- P10 physical OS MemAvailable/pagecache/driver/UMA headroom, model/draft/SWA ring memory, safe loading.
- P11–P14 original source Qwen3.8/GLM5.3/MiMo2.6 operators/state and MoE/MLA/GDN/KDA/mHC correctly.
- P15 source-bound MTP/DFlash and lookup sidecars, target-only+target-verification exact sample state, no hidden draft output.
- P16 source measured expert-aware batch/verify rows/candidate cost and QoS fairness, no max-B guessing.
- P20 optional quantizer calibration/sensitivity, shape/driver kernel autotune bounded and cache invalidation.
- P22 full trained checkpoints, actual physical hardware and original-vs-derived/target-vs-draft/sample/near-tie tests plus end-to-end latency and energy.

Future agent process: verify live Git and owned source files; pin original licensed model/version/source math; source-free tests; scalar CPU reference; physical source GPU quality/leases; repeatable per-task benchmarking; promotion only after user-approved quality policy and safe resource gates; commit owned files atomically without touching other agents' staging. No code edits now.

## 6. Documentation archive generation and historical hash caveat

The latest new specialization source documents were committed independently and are authoritative editable Markdown. The earlier October 15 self-contained v5.1 multimodal handoff and the older October 14 implementation archive remain historical snapshots: **they cannot be silently assumed to contain subsequent source changes**.

The existing scripts/build-comprehensive-handoff.py generator currently attempts to archive the entire already-self-contained v5.1 handoff inside another nine-tilde Markdown source fence. It refuses this with a known literal archive fence collision. This failure predates the new specialization contracts and is due to nested full-text archive delimiting, not a source/math/weight issue.

**Design-only disposition:** do not change scripts or rewrite historical source archives in this task. Preserve both historical snapshot SHA identities; read the current specialization specs and integration plan as **separate authoritative addenda**, with their current live hashes from the repository-wide integrity manifest. The manifest inventory generator can run independently and correctly hash all committed documentation.

**Future maintenance fix for a code-owning/document-build agent, only after authorization:** update the historical archive generator to exclude already-self-contained derived archives from its scanned input, or dynamically choose a collision-free fence per source, with a strict recorded SHA and a regression test for nested Markdown. Never delete user decisions or discard the v5.1 handoff simply to silence the collision. Do not claim the historical full archive has been successfully refreshed until its --check actually passes.

## 7. Primary inspected upstream references

- Specialist-source README (reference removed).
- gfx1151 hardware report (reference removed).
- Benchmark methodology (reference removed).
- Quantization optimizer experiment (reference removed).
- MiMo adaptation/runbook (reference removed).
- First-image convolution optimization (reference removed).
- GLM serving notes (reference removed).
- Qwen serving notes (reference removed).
- MiMo serving notes (reference removed).
- Behavior steering presets (reference removed).
- Derived quantization codec/kernel prior art (reference removed).
- Hardware discussion thread (reference removed).

**The specialist-source numbers are external original author reports, not Zerogravity results.** The Reddit text was not reliably retrievable during this read-only research; do not invent comment contents. No native checkpoint downloads, benchmark executions or hardware changes were performed.

**Architecture preserved:** one verified ModelAssembly/source identity → one mathematical SemanticProgram → guarded legal tensor/ISA/state/quant specialization → one shared physical ResourceArbiter and fair worker → one TaskCoordinator StateRoot publisher → one Router/serving API fabric. No second inference engine.
