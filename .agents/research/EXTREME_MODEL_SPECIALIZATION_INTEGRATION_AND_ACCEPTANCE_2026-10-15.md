# Zerogravity — Extreme Model Specialization Integration Roadmap and Executable Acceptance Design

**15 October 2026 — EXTREME MODEL SSD STORAGE FIRST-CLASS ADDENDUM:** Derived-quantization/QTIP derived quantized routed experts and MiMo/Qwen/DeepSeek sparse n-gram/Engram tables can remain SSD-backed through **the same** verified ModelAssembly/SourceTensorId and ready-leased source-compiled GPU kernels. [Normative virtual residency contract](../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) and [SSD-01..SSD-32 detailed plan](SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) extend original P03/P05/P07/P10/P11/P15/P16/P20/P22 with async dynamic expert materialization, size-class caches, source-verified physical slot generations, prefill/decode reuse, per-session target/draft ring and actual SSD latency/OS headroom. Do not conflate disk KV snapshot with expert weights, source lower-bitrate quant with memory tier, or SSD capacity with interactive latency. Borrowed GPU kernel code remains identical source math; stream policy is physical placement with real event lifetime.


**2026-10-15 | DESIGN PLAN, not runtime implementation, performance proof or new user authorization to download weights.**
**References:** new derived-quantization/QTIP quant, hybrid/DFlash and gfx1151/UMA contracts under .agents/docs; detailed specialist source atlas. Existing P00–P23 phase order and user-selected Qwen3.5/Gemma4/Qwen3.8 CPU-first model order remain unchanged.

## 1. Success definition

Any future extreme sparse/hybrid architecture (Qwen3.8 Flash-Next, GLM5.3 Flash, MiMo2.6 Flash, new MOPD/BitNet/ternary variants) must be added through **one VerifiedModelAssembly + exact model semantic adapter + optional DerivedQuantArtifact + TaskDriver/DraftSource + StateDomains + guarded KernelCandidates**, not a parallel runtime.

Pass criteria:
- original checkpoint source math and independent state oracle;
- quant pack source codec and exact tensor mapping; optional lossless layout repack vs lossy derived quality clearly distinct;
- target CPU reference and GPU backend numerical/source correctness on original trained tensors;
- all-state speculative target verification including zero accepted tokens, RNG/grammar/tool stop, auxiliary sidecars and parent root;
- peak physical weight/state/OS page cache and driver pool admission on unified memory;
- real per-model, per-task, per-ISA performance and cold/warm/long-context checks;
- no global router, TaskCoordinator, public HTTP or resource arbiter new code just because this model is registered.

No model is "supported" until appropriate source, physical, serving and quality evidence tier is achieved.

## 2. Specification owners and minimal architecture

| Stage | Existing authority | Extension allowed | Must not add |
|---|---|---|---|
| Source artifact/derived pack | Verified ModelStore/ArtifactKey | codec/trellis/codebook calibration/semantic-derived hash | arbitrary source Python execution |
| Model math | ModelRecipe/SemanticProgram | new source attention/recurrent/mHC/MiMo and transforms | new model-specific inference engine |
| Draft | DraftSource + target verifier TaskDriver | trained MTP, auxiliary DFlash, lookup; source state | sampler/Root writer in drafter |
| Device math | RegionExecutionProvider/KernelCandidate | gfx1151 WMMA/LDS/DOT4, quant fused GEMV, source-aware fallback | independent AMD allocator |
| Batch/cost | Scheduler + physical ResourceArbiter | measured row/effective expert/QoS cost candidate | hardcoded batch16 in semantic math |
| Model and OS residency | ModelCatalog + arbiter and local leased pools | UMA headroom/page cache/resident rings | artificial virtual GTT=free physical RAM |
| Serving | existing router and canonical task event | truthful derivation, reasoning, metrics/API | Python side server |
| Optional steering | versioned ModelAssembly/profile transform | explicit user-selected behavior effect | hidden "uncensor" mode labeled source-exact |

## 3. Phase dependency map — additive to original project

### ZG-P03 (quant formats and artifacts)
Plan:
1. Read exact derived-quantization/QTIP source code under pinned commit and derive block/codebook/trellis bitstream layout. Separate K/codebook/scale/basis logical tensor identity from layout swizzle.
2. Add source CPU decode oracle for K2..K8 where available and guard mixed per-tensor precision, FP16/BF16 heads, router/indexer and sidecars. Reconcile source HF/GGUF+derived-quant packs; derived and original hashes distinct.
3. Record a separate derived quantization artifact recipe, calibration trace provenance and held-out quality matrix, not silent lower-bit mode.
4. Cache file-backed table source and auxiliary model artifacts by model/quant/source hash and checked offset/size, without full duplication.
5. Report missing derived-quantization or MiMo E8M0 as UnsupportedCodec/UnsupportedModel and permit original source CPU fallback only when a correct reference exists.
Gate: QTIP-01..05, 08..12 and HYB-04; required original CPU reference for exact first model.

### ZG-P05/P07 (compiler and physical GPU)
Plan:
1. Shape-guard Wave32/WMMA fragment and register/LDS layout; source reference must test m=1,2,3,4,8,16+ and aligned/ragged codebook blocks.
2. Source-trust candidate fused derived-quantization packed decoder GEMV versus reconstruct+matmul, with cost and exactness receipts.
3. ROCm gfx1151 specialized v_dot4 and SAD trellis and skinny GEMM candidates; optional CUDA equivalents generated independently.
4. Tune LDS physical stride and scratch warp coverage without altering semantic logical row dimensions. Validate all callers and both partial/full trained model logits.
5. Qualified fallback when native codegen unavailable or slower; no compiler explorer loop on every request.
Gate: GFX-01..06, GFX-14,17 and QTIP-06/07.

### ZG-P10 (global physical admission, OS and state)
Plan:
1. Distinguish physical GPU residency from OS available headroom, file-backed clean page cache, pinned staging, driver pool and UMA GTT.
2. Add an OS-safety admission policy for huge models+draft+sparsely fetched lookup: host total/MemAvailable/cgroup/PSI when supported; platform-specific conservative fallback.
3. Correct multiple model and per-session state: 39 SWA/9 global + DFlash ring example is a model-source configuration, not a hard-coded memory formula.
4. Record optional safe clean-shard page-cache hints and bounded file reads; refuse memory-threatening requests **before** GPU dispatch, never kill unrelated processes.
5. ModelStore singleflight and immutable model weight reuse, per-session root/draft isolation; process-wide unknown GPU event quarantine remains.
Gate: GFX-09..11,15,18, HYB-11/12 and QTIP-13.

### ZG-P11–P14 (specialized model math and MoE)
Plan:
1. Qwen3.8 QSA+GDN+PLE and gate/residual full CPU semantic oracle, then GPU.
2. GLM5.3 KDA/MLA/DSA/mHC source reference, sparse indexer and prompt/vision input variants, then GPU.
3. **MiMo2.6** exact source config, SWA/global KV, E8M0 scales, different key/value widths, MoE/routing and paired overflow correction.
4. Closed region candidates for norm→activation pack→projection, GDN/KDA recurrence+short conv, MLA sparse indexer/tile, residual mixer, routed expert compact grouped quant GEMV/GEMM and safe residual reductions.
5. Token routing/expert destination remains model semantic; aggregation packing and DMA prefetch are physical scheduling only. Independently original model state source oracle.
Gate: HYB-01..05,16, GFX-05/06, and full trained model edge cases (long context, incomplete cache, invalid mask/scale).

### ZG-P15 (target-authoritative speculation)
Plan:
1. Distinct TrainedMTP, DFlash, independent small model and prompt-lookup proposal sources; validated compatibility with target exact checkpoint/derived pack/tokenizer/tap layers.
2. Verify MiMo DFlash partial RoPE, learned mask, bidirectional draft block, sink/value scale and tap_shift; target-only exact acceptance and draft intermediate goldens.
3. Bind optional GLM MTP BF16 eh_proj and Qwen MTP/n-gram sidecars; independent artifact hashes and resource/draft state.
4. Accept only verified target prefix; all provisional GDN/KDA/MLA/indexer, RNG/grammar, output and model state changed atomically or rolled back.
5. Dynamic depth based on net target-only vs draft benefit; no positive benefit -> no draft. Correct stochastic target distribution and tools/stop.
Gate: HYB-06..10,14..15,17, plus exact target sampling distribution/adversarial StateRoot tests.

### ZG-P16 (batching and multi-model QoS)
Plan:
1. Plan rows R=sum(1+d_i), coupled with per-expert count, padded groups, source quant/ISA and state/draft bandwidth.
2. B1/2/4/5/6/8/10/16, depth0..8 and route entropy sweeps; explicit shape "cliffs" and physical queue throughput/fairness.
3. Select among finite prevalidated plans under TTFT/p95/energy/latency/throughput; maintain simple predictable fallback; no model router/catalog scanning per-token.
4. Preserve per-session root, source-specific cache buffers and correct sampled-target distribution.
Gate: HYB-13/14/18, GFX-07/08/16/18 and P16 existing fairness matrix.

### ZG-P20 (bounded autotune / offline quant calibration)
Plan:
1. Compile/reference known correct generic kernels, choose finite shape candidates for new image sizes without unbounded solver search.
2. Put shape/driver/quant/ISA/numeric/graph cache keys and negative candidate evidence into a small persistent candidate database.
3. Optional, source-licensed offline optimizer for calibration trace, per-tensor LDLQ-shaped noise, sensitivity and bitrate allocation, held-out KLD/quality; sparse-model claims remain unverified until tested.
4. Benchmark hardware codebook decode, LDS bank collisions, GDN/MLA and MoE route bands, graph capture, CPU offload and NPU cooperative work as mutually independent experiments.
Gate: QTIP-08/09, GFX-12..14, full A/B negative cases and no protected realtime hot-path compile.

### ZG-P22 (release quality, API and performance)
Plan:
1. Strict evidence tier: recognized-format vs source-exact CPU vs selected physical GPU region vs whole model GPU vs streaming vs measured soft realtime, per model/artifact/device/numeric/shape.
2. Compare derived pack vs original source and speculative-on vs derived target **separately**, plus cross-batch numerical drift.
3. Report cold/warm first response, prefill, physical valid committed output, p50/p95/99 decode, aggregate throughput, OS MemAvailable floor, power/thermal/energy, actual buffer completions and quality.
4. Mixed text/chat/prose/code/tool/multimodal prompts and precise source tokenizer reasoning/stop parser tests; no format-level claim of complete GPT-4 style multimodality.
Gate: completed all relevant QTIP/HYB/GFX suites, existing Qwen CPU8K/extended gate truth and no claimed quality unsupported by real input/source.

## 4. Acceptance register (50 planned tests, NOT implemented)

- QTIP-01..QTIP-14: defined concretely in DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md.
- HYB-01..HYB-18: defined in EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md.
- GFX-01..GFX-18: defined in GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md.
- Source/physical reproductions refer to exact licensed Qwen3.8 Flash-Next, GLM5.3 Flash, MiMo2.6 checkpoint and corresponding specialist pack only if user authorizes downloads/evaluation.
- These 50 tests are **future design acceptance**; they do not add to currently passing Qwen3.5/G3 proof or six MM0 multimodal mock tests.
- Negative tests must fail closed for malformed quant, wrong transform/drafter/source, invalid state, physical unknown completion, memory pressure or unsupported device. No fake Ready/event or conditional lowering to another model.
- Adjacent v5.1 media MM-01..24 remain governed by prior plans; no audio/image/video/decision TaskDriver semantics are changed by specialist-specific optimization.

## 5. Experiment methodology and promotion policy

Each promotion candidate has one exact source digest, original/derived pack, device and OS, driver/kernel compiler/ISA, numeric mode, task/shape/context, source latency/throughput/quality oracle and memory figures. A measured gain on one prompt does not become the default.

| Result | Action |
|---|---|
| Source numerical failure, StateRoot mismatch or unsafe GPU resource | hard FAIL, candidate disabled, error cause recorded |
| Derived artifact quality loss beyond approved policy | reject derived pack as target, regardless of kernel speed |
| Hardware capability missing | source-correct fallback or Unsupported with exact reason |
| Whole task speed neutral/noisy | keep simplest known-correct baseline, record negative experiment |
| One prompt faster but representative P95 worse | do not auto-promote; separate explicit mode if justified and approved |
| Memory pressure close to OS reserve | refuse admission before destructive load, no silent SIGKILL |
| Repeated ABBA shows source-correct whole task win across representative prompts | register guarded KernelCandidate with evidence; no universal model-name hardcode |
| Change in firmware, driver/compiler, quant pack or shape family | invalidate only affected compiled/evidence cache cells |

**Benchmark fairness:** match actual derived packs and original reference precision, tokenizer, source prompt templates, context, sampling, MTP depth, active sessions and device power/fabric clocks. Report per-session TTFT/tokens per sec AND aggregate committed output. The specialist prior-art external results are targets for experimental comparison, not facts about Zerogravity.

## 6. Small design decisions and open questions

| ID | Disposition | Decision |
|---|---|---|
| SYK-001 | Existing user constraint reaffirmed | Rust-owned one engine and one TaskCoordinator, no hidden external/Python process |
| SYK-002 | Proposed precision contract | New source quant codec and derived artifact are orthogonal to model mathematical identity and packed hardware kernel |
| SYK-003 | Proposed implementation gate | MiMo-V2.6 a distinct source-qualified ModelRecipe and DFlash semantics; no Qwen alias |
| SYK-004 | Proposed specialization | Kernel physical codebook/tile/ISA/LDS, guarded by exact source numerical and shape evidence |
| SYK-005 | Proposed QoS | MoE batch/verification depth uses measured expert work, row packing and accepted output, not max B/draft |
| SYK-006 | Proposed source policy | Trained MTP, DFlash, lookup and small drafter are versioned DraftSources with target-only authoritative commit |
| SYK-007 | Proposed OS safety | Coherent-memory OS+GPU physical headroom, bounded source read/page-cache pressure, no invisible unsafe process killer |
| SYK-008 | Proposed offline research | Optional representative trace, shaped tensor sensitivity and held-out quant quality before derived artifact approval |
| SYK-009 | Proposed behavioral gate | Activation steering is OFF by default, explicit behavior-changing model version, separately assessed for safety/quality |
| SYK-010 | Proposed cold shape policy | No indefinite first-image request tuner; source-equivalent generic fallback and finite offline autotune |
| SYK-011 | Open | exact derived-quantization/QTIP source codec schema, licensed original checkpoints and conversion toolchain choice |
| SYK-012 | Open | acceptable approximate quant/perceptual loss vs original; user has not chosen universal threshold |
| SYK-013 | Open | native CUDA/ROCm LLVM23 source build availability and operator permission |
| SYK-014 | Open | target startup latency, per-task soft SLO and high-availability/process recovery policy |
| SYK-015 | Open | authorized model downloads/large 100GiB physical pressure experiments and page-cache hints |
| SYK-016 | Open | whether any steering/activation-transform feature should be exposed at all; default remain disabled |

Do not treat a proposal as a user-approved implementation requirement or an open question as resolved. The actual new user request is to **strengthen plans and specifications** cleanly, not to launch experiments.

## 7. Handoff for independent agent

Read:
1. existing v4.3 governing architecture and v5.1 ModelAssembly/semantic contract;
2. this integration roadmap;
3. each of three new specialization contracts **in full**;
4. specialist original source atlas, including negative results/method;
5. original user decision register/current implementation handoff and live Git;
6. existing P00–P23 and experimental Qwen3.5 CPU8K/partial GPU quality evidence.

Then independently implement smallest source-free acceptance within its assigned phase only after user authorization; quantify all errors and prevent cross-agent file staging conflicts. Report model/codec support honestly. **Never substitute a new local engine, hidden quant format, unapproved BF16 behavior steering, or fake GPU event.**

This plan is documentation and test design only; the v4.3 architecture and P00–P23 implementation sequence remain authoritative.
