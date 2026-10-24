# Recovered History — volume 3

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/research/EXTREME_MODEL_SPECIALIZATION_DESIGN_AGENT_HANDOFF_2026-10-15.md

Original UTF-8 bytes: 13533; SHA256 `3506887d4fc6f44d7d7ffd73dd5b916e37698df400ae8a2acce23f43a18c8b8b`.
Resolve embedded relative paths against the original path, not this volume.

````text
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

````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/research/MULTIMODAL_PARALLEL_RESEARCH_WORKSTREAMS_2026-10-14.md

Original UTF-8 bytes: 6707; SHA256 `765d0182594bec62ce3ab457c8bcbb75123bbfd4b37048b638d6c984e9988c00`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity multimodal architecture — parallel reviewer and subagent workstreams

**14 October 2026 · v5 design-only · coordination record**
Base: main / a965737 (source-verified Qwen3.5 Phase 1 CPU). P07 GPU and 8K numerical investigation proceed independently and must not be edited by these architectural reviewers.

## Runtime status of subagent execution

Independent Codex and Claude Code binaries are present on pvl-l5, but standalone Codex fails HTTP 401 without a valid login/bearer and Claude reports "Not logged in." Therefore no separate autonomous architecture subagent was successfully started through this connector. **The seven slices below are scoped reviewer/subagent briefs, not a claim that seven agents actually ran.** An authorized user/agent environment can execute them separately later. The v5 architecture study and official-source research were directly completed in this session.

Do not bypass auth with private keys/secrets, mount SSH agents or execute arbitrary agent binaries with unsafe permissions to simulate concurrency. No source/model weights are prerequisites for design-only reviews.

## Parallel workstream contracts (each independent, docs-only)

| Workstream | Exclusive output location if separately delegated | Source questions and required proof |
|---|---|---|
| A — Core execution contracts | .agents/plans/multimodal-review/core/ | Preserve SemanticProgram / ExecutableBundle / TaskDriver / StateRoot / ResourceArbiter; enumerate exactly which ValueType/TaskFrontier versions need amendment; reject new persistent IR/server/central scheduler. Include 12 fake model acceptance drills. |
| B — Image generation/editing | .agents/plans/multimodal-review/image/ | Qwen-Image, rectified-flow [dev] vs [klein] conditioners, SD3.5 and autoregressive multimodal image MoE checkpoint-specific image generator; source tensor/processor/flow/AR/codec evidence, licensing and memory. Audit AR-vs-diffusion stage ambiguity using actual implementation rather than model marketing. |
| C — ASR and streaming speech understanding | .agents/plans/multimodal-review/asr/ | Whisper window/encoder/AR, Parakeet CTC/RNN-T/TDT, Canary translation, VibeVoice ASR, Qwen-Omni Thinker; audio sample clock, acoustic state, chunked replay and partial/final transcript. |
| D — TTS and real-time conversation | .agents/plans/multimodal-review/tts/ | Kokoro duration/vocoder, dual-AR slow+fast AR residual codec, Qwen3-TTS, VibeVoice codec/flow, Qwen-Omni Talker, Hosted TTS. Model-specific audio token clock, voice/speaker state, duplex interruption and playback watermark. |
| E — Video / coupled A/V | .agents/plans/multimodal-review/video/ | Wan2.1/2.2 denoise expert schedule, dual-stream video DiT selective tiles, asymmetric-attention DiT (AsymmDiT), coupled audio/video, provider-only hosted generation services. 3D VAE/temporal motion/sync, rational clocks, completion and memory tests. |
| F — Typed decisions and source quant codecs | .agents/plans/multimodal-review/decision-quant/ | Open joint-schema decision head exact option-logit joint head, open encoder classification/decision model (NO: it is an encoder classification/decision model, not CTC), external decision service; ternary PTQ1_0/PQ2_0 and required Hadamard activation transform. Distinguish codec from underlying Qwen topology and semantic decision head from text JSON generation. |
| G — Integration and adversarial review | .agents/plans/multimodal-review/reconcile/ | Reconcile A-F against v4.3 and the v5 proposal; find undocumented state writers, fake tokens, missing codec boundaries, impossibly broad ABI, unsafe media resampling, shape/clock conflicts, licensing and unverified product claims. Produce a minimal diff list with acceptance fixtures, no speculative refactor. |

## Rules for actual independent agents, if authorized later

1. **Read governing design**: .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md; COMPOSITION_AND_EXTENSION_CONTRACTS.md; MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md; TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md; .agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md; .agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md.
2. **Current status is authoritative:** native Rust Qwen3.5 0.8B/2B CPU validated through 8K, 16/16 phase1; P07 CubeCL Q4×Q8 source-correct provider built and host-tested but not physically tested. No native speech/image/video/decision checkpoint runs in Zerogravity yet.
3. **Source validation**: original GitHub/Hugging Face/official paper/API and exact model-specific version; cite source URLs and publication dates where present. Record "unverified" rather than inventing release/model internals. Differentiate online proprietary provider from locally loadable checkpoint and its license.
4. **Ownership**: write docs in your own exclusive directory above, never edit shared source, Cargo.toml, whitepaper, decision register, target/model files or other agent's work; no Git staging/stashes/reset/rebase/commit while other agents work. The coordinator alone reconciles reviews and commits only reviewed files atomically at one boundary.
5. **Task result**: max ~8–20 pages of source-grounded technical analysis per track; a concise verdict, proposed smallest contract changes, three source model implementation traces, hidden invariants, negative tests, performance hypotheses and explicit "open questions". Avoid inventing new enduring IR layers.
6. **Validation**: reviewer G evaluates conflicts and tests all 12 MM-01..MM-12 mock scenarios. A result is DESIGN COMPLETE only when other model TaskDrivers can be added without changing the core router, event authority, resource ledger or StateRoot sole publisher.

## Integrated current outputs

- Proposed governing additive design: .agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md
- Source-family model research and implementation order: .agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md
- Public framing in docs/WHITEPAPER.md and docs/ENGINEERING_SPECIFICATION.md is marked **proposed**, not implemented or tested.
- CPU accuracy investigation remains separately owned by .agents/plans/QWEN35_8K_NUMERICAL_DRIFT_INVESTIGATION_2026-10-14.md.
- GPU development remains separately owned by probes/gpu-cubecl/ and src/accelerator.rs, with no permission for architecture reviewers to alter its numerical/kernel semantics.

**Next architecture milestone:** MM0 fake typed values + rational timebases + model-component/iteration acceptance fixtures, after the design is reviewed and the user authorizes that implementation phase; do not disrupt P07 GPU rollout.

````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/research/MULTIMODAL_V5_1_RESEARCH_WORKSTREAMS_AND_AGENT_BRIEFS_2026-10-15.md

Original UTF-8 bytes: 12366; SHA256 `c8b4e54496f56e84c5684c1721777cde231698b683070163fa3ff73ebadef83c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity v5.1 — Independent Research Workstreams, Agent Instructions and Integration Ownership

**15 October 2026 · Design/research coordination only.** No source code, runtime, external model, provider account or GPU implementation changed by this work. No separate autonomous subagents were actually launched: prior October 14 documentation reported unauthenticated external Codex/Claude binaries; this conversation's available tools have not supplied authenticated independent subagents. **Eight named workstreams are a division of responsibility and ready-to-use agent briefs, not evidence of independent parallel execution.**

## 0. One coordinating lead, independent source research

The coordination lead is the design/research integrator, not a new inference service, runtime scheduler or model broker. It owns:
- historical user decision preservation; explicit USER vs PROPOSAL vs OPEN labeling;
- v4.3/v5/v5.1 architecture authority and minimal compatibility contracts;
- independent source-model code/weight/processor/codec/license references;
- source algorithm, state, clocks, hardware, streaming and numerical cross-check;
- practical implementation acceptance roadmap and precise support tiers;
- adversarial cross-stream review and minimality test (no duplicate engines/allocators/state);
- deterministic handoff and SHA manifest across old historical records and new docs;
- no mutation of active source agents' worktree, stage, scripts or GPU source.

Every substream writes to its own unique docs path, keeps original facts/source URL/version and reports uncertainties rather than inventing precision or release facts.

## 1. Research lane A — Core engine and semantic/state design

**Read:** .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md, COMPOSITION_AND_EXTENSION_CONTRACTS.md, TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md, UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md.  
**Current deliverable:** .agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md.
**Focus:** SemanticProgram, Verified ModelAssembly/ports, TaskDriver algorithms, StateDomain PREPARE vs TaskCoordinator PUBLISH, physical event/lease correctness, two durable representation rule, four legal specialization axes.
**Required adversarial models:** autoregressive image MoE ambiguity; dual-AR nested RVQ; Wan2.2 timestep experts; coupled dual streams; joint-schema decision head; ternary; full duplex.
**Stop if:** need second inference engine, allocator, public plugin ABI for each op, or generic workflow DSL in device workers.

## 2. Research lane B — ASR, translation and speech understanding

**Dossier:** .agents/research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md.
**Models:** Whisper; NVIDIA Parakeet FastConformer with CTC/RNN-T/TDT variants; Canary AST; Nemotron streaming; VibeVoice rich/streaming ASR; Qwen3-Omni Thinker.
**Questions:** exact feature frontend/resampler, decoder blank/beam/duration/timestamp policy, source language translation, streaming versus offline checkpoint, diarization provisional corrections, chunk/lookahead and speaker state.
**Required outputs:** source-by-variant matrix, three algorithm traces (AR, CTC, RNN-T/TDT), exact clocked Transcript schema, negative fixtures MM-ASR-01..12, source-evidence limits.
**Allowed implementation seam later:** source adapter, TaskDriver, frontend/encoder/source kernels, typed output. No separate ASR serving stack.

## 3. Research lane C — TTS, codecs, voice and native duplex

**Dossier:** .agents/research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md.
**Models:** Kokoro, dual-AR, Qwen3-TTS, VibeVoice TTS/Realtime, neural-codec full duplex, Qwen3-Omni Talker, a hosted TTS service.
**Questions:** text normalization/G2P licensing, duration/prosody, temporal acoustic token and RVQ depth clocks, source codec frame completeness, voice identity/consent, simultaneous audio capture/output, source state recovery on barge-in.
**Required outputs:** non-AR Kokoro trace, dual-AR trace, full-duplex trace, source sample/codec/rate table, adverse latency/backpressure tests MM-TTS-01..12.
**Stop if:** forces every TTS into token AR, publishes partial codebook frame as audible PCM or hides cloud egress.

## 4. Research lane D — Image/flow/editing

**Dossier:** .agents/research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md.
**Models:** Qwen-Image, rectified-flow dev/Klein, SD3.5 Large/Medium/Turbo, autoregressive image MoE variants and proprietary frontier image APIs.
**Questions:** distinct conditioning architectures (Mistral vs Qwen3), MMDiT/AR/flow path, guidance/noise/seed/source solver, VAE/edit/mask/color, variant-distilled step policy, checkpoint license and execution memory.
**Required outputs:** exact assembly examples, source-qualified legal flow stage, published autoregressive image generate_image call graph audit request (do not invent), negative tests MM-IMG-01..12.
**Stop if:** model family name alone selects text encoder, wrong solver deemed equivalent or uncommitted latent exposed as image.

## 5. Research lane E — Video, temporal VAE and native coupled AV

**Dossier:** .agents/research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md.
**Models:** Wan2.1/2.2, dual-stream video DiT/1.5, asymmetric-attention video DiT, coupled A/V, hosted generation services.
**Questions:** spatiotemporal latent axes, 3D causal VAE stride/overlap, sparse tile mask, asymmetric projection shapes, Wan denoise expert switch, joint A/V solver/codec clocks, frame/mux finality, large-model scratch/offload.
**Required outputs:** source model traces and cross-attn constraints, exact AV commit group examples, negative fixtures MM-VID-01..15, source vs hosted distinction.
**Stop if:** Wan MoE treated as token FFN routing, coupled A/V reduced to unrelated TTS+video, or invalid temporal VAE output labeled final.

## 6. Research lane F — Typed decisions and low-bit specialist text

**Dossier:** .agents/research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md.
**Models:** proprietary hosted decision service, open typed-decision models, open joint-schema decision models, ternary and future BitNet/FP4/linear/hybrid families.
**Questions:** choice/score/noul schema, calibrated vs raw logits, direct scoring heads, ModernBERT/mmBERT/Qwen backbones, PTQ1_0/PQ2_0 128-group scales, mandatory 1024 Hadamard source transform, vision mmproj.
**Required outputs:** explicit TaskDriver and QuantEncoding examples, false fallback matrix, MM-DEC-01..06 / MM-BON-01..08 golden fixtures, source licensing.
**Stop if:** generated JSON claimed source-equal to direct decision logits, stock Q2 loads the ternary model without activation transform or the hosted decision service treated as local weights.

## 7. Research lane G — Serving, real-time QoS, APIs and opaque providers

**Dossier:** .agents/research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md.  
**Proposed spec:** .agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md.
**Prior art:** OpenAI Realtime/Audio/Responses/Image/Video and model catalog, Gemini Live, vLLM, SGLang, hosted voice and generation services.
**Questions:** WS/WebRTC/SIP/HTTP/SSE binary and control paths, request capability conformance, model source vs hosted identity, playback ACK vs committed output, reconnect/resume, jitter/clock credits, async provider job Unknown/reconcile and idempotency, soft deadline admission and cross-workload fairness.
**Required outputs:** RealtimeSession, PresentationLedger, MediaEvent/API profiles, security/egress threat model, MM-SRV-01..12 and MM-13..24.
**Stop if:** per-audio-packet global router/snapshot/JSON hot path, second GPU resource authority, client ACK modifies neural StateRoot or unauthorized provider egress.

## 8. Research lane H — Adversarial reconciliation / quality gate

**Dossier:** .agents/research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md.
**Purpose:** reconcile lanes A-G against v4.3 governing authority and v5 proposal, ensure no duplicated owner or latent false model-support claim, record OPEN and PROPOSAL rather than silently approving changes.
**Required outputs:** conflict/disposition matrix, smallest contract change list, original-source vs native hardware vs hosted status matrix, phase/test dependencies and negative tests, reproducible handoff.
**Hard questions:**
1. Does adding this model require modifying central Router/TaskCoordinator/ResourceArbiter/transport parser?
2. Does this model support its claimed output/streaming mode or only have recognizable weights?
3. Are all audio/video/sample/codec/solver clocks genuinely typed and compatible?
4. Does per-model numerical/codebook/source processing remain equivalent?
5. Could a stale provisional GPU result publish or release active physical handles?
6. Does a protocol reconnect lie about what was heard or the model's actual state?
7. Is license/privacy/consent handled per asset and external provider?
8. Is the new abstraction simpler than adding a TaskDriver/component/region adapter?
9. Is the reported performance measured on a real source checkpoint/device/load, not upstream marketing?
10. Can another agent implement/reproduce the design without the original chat?

## 9. Copyable design-only agent takeover brief

~~~text
You are an independent DESIGN-ONLY reviewer for the Zerogravity Rust-native
universal AI execution fabric. Use the connected dotlink-l5 repository at
/home/pvl/spaces/abird/src/zerogravity. Do NOT change code, Cargo, tests,
build/CI/Nix, model weights, permissions or remote services. Do NOT reset,
stash, stage or commit other agents' files; write only your exclusive assigned
new .agents/{docs,plans} research path, preserving shared source.

Read full historical handoff progressively, governing v4.3 architecture,
original 2026-10-14 v5 multimodal proposal, v5.1 core/serving/assembly docs,
and all user decisions before altering any design conclusions.

For each assigned source model/variant: cite original repo/paper/API URL,
checkpoint/license/processor/codec version, mathematical execution algorithm,
model-component assembly, tensor shapes/low-bit transforms, per-task state
and rollback, sample/frame/token clock and output validity, GPU/CPU/device
region specialization, physical residency, serving modality/transport,
negative/positive source tests, actual versus unverified status and precise
open questions. Separate provider marketing from verified source behavior.

Propose ONLY minimal changes under one SemanticProgram/ExecutableBundle,
one TaskCoordinator publisher, one PhysicalResourceArbiter, trusted TaskDrivers
and native/external TaskServiceAdapter boundaries. No extra model engine.
Deliver source-backed Markdown with full acceptance fixtures, no fabricated
real tests or autonomous subagent claims; send lead a conflict/change-radius
summary and keep your file ownership explicit.
~~~

## 10. Parallel work and merge discipline

- Separate research tracks can be reviewed simultaneously by truly authenticated external agents; this document does **not** start them.
- Each agent must own one **new unique file**, avoid modifying the current shared historical archive, decision register, docs manifest or live code.
- Lead alone reconciles content and builds a new self-contained handoff/manifest, never overwrite another agent's staged or dirty files.
- Implementation, when explicitly authorized later, has independent phases/branch/file ownership; source-real CPU/physical GPU/media/endpoint gates precede claiming support.
- Atomic Git operations may operate only on explicitly owned paths and only after status verification; do not include unrelated staged changes.

## 11. Completion standard

The architecture research program is DESIGN complete when the three v5.1 governing specs, seven model/API research dossiers, independent reconciliation, decisions/OPEN register, MM01..24 plus per-family test fixtures, roadmap and SHA-verified handoff exist and agree. **It is not RUNTIME complete.** Future agent implementation must not treat the documents as loaded model proof.

This workstream brief is a persistent research plan and accurate documentation of the research completed in the conversation, not an assertion of eight launched model agents.

````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/research/SSD_STREAMING_FIRST_CLASS_ARCHITECTURE_AGENT_HANDOFF_2026-10-15.md

Original UTF-8 bytes: 10537; SHA256 `976f2df97b96f5e0c9e29e64e6fdcea341e4acda01343551452a28b46dec60a7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — SSD Streaming as First-Class Model Residency: Comprehensive Next-Agent Handoff

**2026-10-15 • DOCUMENTATION / ARCHIVE TOOLING ONLY. NOT A NEW INFERENCE IMPLEMENTATION.**
**Source repository:** /home/pvl/spaces/abird/src/zerogravity, dotlink-l5 root /home/pvl/spaces/abird.
**User instruction:** user explicitly granted permission to resolve the nested comprehensive-archive generator and strengthen architectural docs/plans, and required SSD streaming to feel natural with the **same tensor/materialization structures** backing GPU/RAM or SSD. No source model, SSD benchmark, huge weight download, ROCm/CUDA driver change, running deployment or model inference code was requested.
**Baseline:** main was clean at bb93449 at start of the work; verify live HEAD/worktree before any future code. Existing native CPU/GPU Qwen3.5 Phase1/physical partial-region evidence remains unaffected. Prior user model order and no-foreign-inference-engine policy remain.

## 1. Required reading before implementation

1. Read existing v4.3 authority: .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md; MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md; COMPOSITION_AND_EXTENSION_CONTRACTS.md; TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md.
2. Read **new governing storage contract**: .agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md in full.
3. Read independent original-source research: .agents/prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md (DS4 expert SSD, file-only n-gram lookup, disk KV, source issue #384/503/629, prior offload projects).
4. Read implementation and all 32 planned tests: .agents/research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md.
5. Read existing .agents/docs/MEMORY_RESIDENCY_AND_IO.md, ASYNC_RESOURCE_COMMAND_IR.md, STATE_CACHE_AND_TRANSACTIONS.md, FORMATS_QUANTIZATION_AND_LOADING.md, COST_MODEL_AND_AUTOTUNING.md, INVARIANTS_AND_FAILURE_SEMANTICS.md and MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md. These now explicitly point at new contract.
6. Read v5.1 verified ModelAssembly and RealtimeSession/QoS specs, EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE plan, DECISION_REGISTER and original complete historical archive **progressively in full**. On true takeover, revalidate most recent Git, artifact hashes, driver and actual device.
7. Read the current modified doc generator and offline regression suite before regenerating archival docs; source archives are snapshots, not editable primary source.

## 2. Architecture diagram — one source tensor, several legal physical homes

~~~text
Verified ModelAssembly / SourceTensorId / TensorView / exact quant + shape
                         |
          Prepared compiled region / KernelCandidate
                         |
                ReadyTensorLease binding
                     /          \
             Direct READY       Miss/provisional
             device slot         source extent file
                  |                   |
        same compiled kernel       bounded async IO
             dispatch               + source verify
                  |                   |
       true completion event     pinned RAM/mapped bytes
                  |                   |
            lease retire          H2D+true upload event
                  |                   |
            output-ready         new generation atomically published
                  |                   |
          TaskCoordinator <---- same region kernel and physical event
                  |
             ONE StateRoot
~~~

**Resident path:** all required source pointers already physically Ready; no per-token filesystem/IO worker wakeup, no global cache lookup, no extra model-specific scheduler, no hidden blocking OS read. The abstraction is intended to add negligible overhead to direct GPU computation when all weights fit.

**SSD path:** same source tensor, quant, model semantics and final mathematical output; bounded, measured SSD IO/mapping and true upload completion. Actual miss latency cannot be zero; prefetch and staged overlap are performance candidates under source and physical effect guards.

## 3. One source/four policies

- **Routed experts:** dynamic per-expert/projection slabs, source quant/size class, valid routes from model router, hotness/IO cost, pinned from GPU command encoding through actual completion (DS4 #384).
- **Huge lookup rows:** n-gram/Engram table remains immutable disk primary with tiny coalesced selected-row cache, source hash/address and row codec. Different from expert bulk slabs.
- **Frozen session StateRoot:** committed private KV/GDN/SWA/codec snapshots to disk with exact model+session+format identity, immutable frozen state, encryption/TTL policy and compatible replay; no mutable active state sharing or false zero-copy restore.
- **Unpinned dense trunk:** optional, explicitly QoS-downgraded capacity mode if dense mandatory weights cannot fit; can be dramatically slower (file-backed lookup source). Not an automatic equal-speed source kernel mode.

All share the original SourceExtent physical IO provider and ResourceArbiter, but a unified LRU across mutable state, experts, lookup, media and trunk is NOT accepted.

## 4. The two highest-priority external regression fixtures

### Expert slot corrupting a queued GPU graph (DS4 issue #384)
- Create A/B layers and cache capacity one. Encode GPU reader for expert X from slot S but do not execute yet.
- Request Y from later layer. Cannot recycle S while earlier GPU encoding/capture/queued reader exists, even if CPU cache directory thinks X is idle.
- Flush/wait at legal boundary, use another verified tier or backpressure; overwrite forbidden.
- Compare original full logits/StateRoot under extreme cache pressure, not only cache table invariants.

### Mixed Q2/Q4 cache thrash and budget overcommit (DS4 issue #503)
- Alternate unequal byte-size expert matrices across layers, then send a short prefill following a warmed decode.
- Preserve active size classes, enforce one global byte cap across all, avoid unconditional cache-clear on format switch/prefill, reserve H2D/staging before GPU submit.
- Cache hit ideally permits direct resident pointer use without copying multiple GiB/token; measure actual DMA bytes and latency even at 97% hit rate.
- Smaller cache/cold SSD may slow tremendously, but must not corrupt source math or hang.

## 5. Phased implementation and honest gates

| Phase | Future owner | Output and test gates |
|---|---|---|
| SSD-P0 / P04 | semantic/CIR | fake GPU/RAM/SSD logical tensor, real generation/lease simulation, SSD-01..04/08/12 |
| SSD-P1 / P03 | ArtifactCatalog+IO | source extent/open descriptor, byte integrity/format, portable async IO, SSD-05..07/17..20 |
| SSD-P2 / P05/P07 | DeviceWorker | real GPU slot, H2D event, encoded/queued reader lease, graph-safe indirection, SSD-09..16 |
| SSD-P3 / P10/P12 | cache/resource | mixed expert size and total bytes, direct pointer hits, protected prefill/cache, SSD-21..24/26/29 |
| SSD-P4 / P11/P10 | model state/lookup | giant sparse lookup and private committed disk KV/codec snapshot, SSD-25/27/28/30 |
| SSD-P5 / P16/P20/P22 | admission/benchmark | full physical original source GPU/SSD parity and memory, cold/warm latency and QoS, SSD-31/32 |

All 32 SSD-xx tests **PLANNED**. No exact model/backend promised until a trained full source checkpoint, actual disk/driver, numerical and StateRoot gate passes. Existing Qwen3.5 original CPU through 8K 16/16, remaining 16K/32K deferred; no more native GPU full-model or audio/video execution is asserted.

## 6. Documentation archive generator repaired in this phase

The former full archive builder failed on nested prior v5.1 complete handoff with a fixed nine-tilde Markdown fence. The minimal permitted tooling fix chooses deterministic per-source delimiter longer than any literal tilde run; source bytes, including UTF-8 and trailing newlines, remain recoverable by indexed original byte length; output is stable and hash-indexed. It adds offline regression tests for nested archives, source bytes, determinism and previous committed handoff. The existing scripts/check.sh source-only Python gate runs these tests. Regenerate **after** all documentation is committed and source list is stable; manifest last; use --check without guessing success.

Do not remove the previous self-contained multimodal handoff, discard user decisions or copy its entire content manually. The deterministic archive must include prior archive as a nested **source document**, with dynamic safe fences; the repo-wide SHA manifest excludes itself and records exact current source/archived bytes. This is a documentation-tooling bug fix only, not an inference engine change.

## 7. Original external sources

- DS4 SSD streaming https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md
- DS4 expert slot queued command race https://github.com/antirez/ds4/issues/384
- DS4 mixed Q2/Q4 thrash https://github.com/antirez/ds4/issues/503
- DS4 host RAM cache proposal https://github.com/antirez/ds4/issues/629
- DS4 server disk KV https://github.com/antirez/ds4/blob/main/docs/SERVER.md
- File-backed disk lookup/trunk and KV prior art (reference removed).
- File-backed lookup flags prior art (reference removed).
- Prior offload background paper https://proceedings.mlr.press/v202/sheng23a.html
- Activation-aware expert-offload prior art (reference removed).

Upstream measured numbers are source author reports, not new Zerogravity benchmarks. Prior-art source licenses and versions require re-verification before implementing algorithms.

## 8. Decision provenance and scope

**VST-USER-001** in decision register records user approval for docs/plans/documentation-tool repair and requirement for first-class SSD residency. Do not confuse this with authorization to run huge model loads or a live remote provider. The implementation must preserve user design goals: simple intuitive one Rust engine, no duplicate resource/state writer, composable virtual tensor backing, zero-extra software work on resident path, and truthful slow/storage-bound modes.

A future implementation agent must follow the smallest source-free acceptance first, then real original source and physical GPU, and only then publish a status. Keep other agents' work staged/unstaged untouched and commit only owned paths; no whole-tree reset/stash.

**This handoff is implementation-ready design and documentation validation, not running native SSD inference.**

````

