# Zerogravity v5.1 Multimodal Decisions, User Requirements, Design Proposals and Open Questions

**15 October 2026 · user intent and model-research provenance addendum.**  
**Important:** This file ADDS user decisions U-040/U-041 and new design-proposal/open IDs without modifying the concurrent agent's shared .agents/plans/DECISION_REGISTER.md. For full project user decisions U-001..U-039 consult the existing decision register and the historical comprehensive implementation archive. The new comprehensive v5.1 handoff incorporates that history verbatim as a dated snapshot.

## 0. How to interpret the decision ledger

- **USER**: directly stated or explicitly reaffirmed by user in this conversation; preserve unless later user supersedes.
- **PROPOSAL**: architecture recommendation from research/assistant; recorded to enable independent review, NOT an extra user approval.
- **OPEN**: requiring explicit user policy, exact source audit or performance/hardware evidence before finalizing.
- **ASSUMPTION**: reversible interpretation, documented so implementation agents don't invent authorization.
- **HISTORICAL**: prior source fact/implementation state at a dated baseline; not current software support assertion.
- **SUPERSEDED**: an earlier user decision whose exact scope was later revised, preserving previous rationale.

The user explicitly asked for deeply researched, composable architecture and comprehensive documents, not a blanket waiver of source licensing/secret permissions or automatic native execution of proprietary models.

## 1. New explicit user instructions U-040 and U-041

### U-040 | USER | 15 October 2026 — universal audio/image/video/decision/ternary/realtime design research, strictly no code

The user requested another **architecture pass in design-only mode, no code change at all**, including:
- standalone voice models, speech-to-text, text-to-speech, low-latency streaming and native real-time full-duplex audio models, their codecs, speech processing, model variants and exact task algorithms;
- **Image:** Qwen Image, rectified-flow image models, autoregressive multimodal image MoE, Stable Diffusion 3.5 and frontier/new image architectures;
- **Audio:** a hosted TTS service, dual-AR speech, NVIDIA Canary/Parakeet, Whisper, Kokoro, VibeVoice, Qwen TTS and other top open or hosted models;
- **Video:** Wan 2.1/2.2, hosted generation services, dual-stream video DiT 1.5, coupled A/V models, asymmetric-attention video DiTs and future model architectures;
- **Decision:** proprietary and open typed finite-choice/score decision families;
- **Text special:** ternary-compressed and other source-encoded low-bit, unconventional/hybrid language models, without copying/rewriting inference engine code;
- **Multimodel/multimodal:** one fabric that can run combined models, models with native multiple modalities, and model-specific specialization at every legal computational, hardware, task and placement layer;
- **Serving:** modular, high-performance multi-model router/serving; all relevant APIs and transport types needed for media, realtime, combinatory models, including patterns represented by the user-supplied example "gpt-realtime-2.5";
- **Research organization:** one lead and distinct deep model-family research streams, as many as useful, source-backed and reconciled to minimal composable design; research may be delegated only when real subagent tooling exists.

**Critical interpretation:** "gpt-realtime-2.5" is a user-supplied example, not verified to be a currently released official OpenAI API model. Official model catalog consulted on 2026-10-15 documented GPT-Realtime-2.1 and GPT-Live families. Preserve desired **capabilities** (duplex native speech, barge-in, playback reconciliation, tool events, low latency), not an invented proprietary model implementation.

**User forbids app code edits in this design phase.** No model download, runtime binary, API server, GPU driver permission change, live phone call, provider API credentials or remote deployment authorized.

### U-041 | USER | 15 October 2026 — preserve full comprehensive architecture and all research as on-repo implementation/handoff artifacts

The user requested:
- write **all** multimodal and realtime design plans, full architecture specifications and any other useful artifacts into .agents/{docs,plans};
- comprehensive attention to each modality/model/algorithm, not just executive summary;
- record each individual research stream's learned methods, source links, release/codec/version caveats, comparisons, negative tests, open questions, implementation guidance;
- preserve **all prior user decisions**, design rationale, rejected alternatives and pending work across original project and this research;
- author sufficiently detailed comprehensive implementation handoff so independent future external LLMs/agents can recover exact project intent, governing design, source status, phase ordering, acceptance gates and what remains unimplemented **without relying on the conversation**;
- keep design-only scope: documentation permitted, **no Rust/application/build/CI/test changes**, and preserve other active agents' changes/staging.
- no implied approval to run a model or proprietary service or modify running systems.

**Fulfillment plan:** new uniquely named v5.1 doc set, individual modality and serving research dossiers, DMM decisions/open ledger, acceptance roadmap, source-provenance append-only master handoff and SHA manifest. Shared original decision register and old comprehensive archive stay untouched because other agents may be concurrently editing them; they are incorporated into the v5.1 handoff by dated hash-verified snapshot.

## 2. Prior user decisions explicitly retained (not newly re-approved)

| Existing ID(s) | Binding original intent | This plan preserves |
|---|---|---|
| U-001, U-003, U-017 | lowercase zerogravity; Rust-owned original engine and general architecture; attributed prior art without unwanted clones | one Rust engine, no C++/Python hidden inference backend, source/license verification |
| U-004..U-009 | standard/hybrid/MoE massive model families, semantic IR, extreme hardware optimization while preserving portable fallback | no model family special-cased into global router or separate engine |
| U-011, U-015, U-018, U-022 | comprehensive plans, source evidence, independent LLM handoff, simple/maintainable/extensible architecture | detailed docs and falsifiable criteria, decisions annotated as USER/PROPOSAL/OPEN |
| U-019..U-020 | NVIDIA/AMD/NPU/AOT/rack-scale devices and heterogeneous remote/tunnel execution | unified physical provider/placement/events, not per-modality device stack |
| U-021 | native built-in model router, many models and profiles, OpenAI-compatible serving and simple one-command UX | one EngineCore and bounded task/protocol adapters, no automatic proxy-per-model |
| U-025..U-027 | Qwen3.5 0.8B then 2B CPU→CUDA/ROCm/Vulkan, Gemma4 E2B/26B then Qwen3.8 27B CPU-first; 8K Phase1 16/16; extended 16K/32K deferred | MM design does NOT silently reorder practical implementation gates |
| U-028..U-029 | independent long-context GDN accuracy investigation and three parallel workstreams including multimodal research | separate CPU math/GPU/architecture ownership, no invented subagent execution |
| U-030..U-039 | source-correct GPU P07-A→G3a and critical review gates, true device readback/quarantine, single StateRoot, no false hardware proof | existing partial GPU status honest; source-free media mocks ≠ media support |
| All previous explicit user decisions | full record in .agents/plans/DECISION_REGISTER.md | new handoff snapshots it, no duplication/rewriting of history here |

## 3. Research-derived design proposals DMM-001..DMM-032

| ID | Status | Proposed decision and reason |
|---|---|---|
| DMM-001 | PROPOSAL | Adopt v5's six narrowly scoped multimodal contract refinements as implementation design over v4.3, not a separate inference architecture |
| DMM-002 | PROPOSAL | Use one SemanticProgram/ExecutableBundle; RIR/KIR/CIR are compiler-internal views, not extra externally committed runtimes |
| DMM-003 | PROPOSAL | Keep one TaskCoordinator PublishRoot per session/shard, one global PhysicalResourceArbiter and per-card bounded device workers |
| DMM-004 | PROPOSAL | Typed ValueDescriptor supports tokens, PCM, acoustic features/RVQ, pixels, latents, video frames/packets, transcript, decisions and structured outputs |
| DMM-005 | PROPOSAL | Exact rational Timebase/clock domains and codec/resampler delay; no fake equivalence between audio samples, codebook depth, frame PTS and token positions |
| DMM-006 | PROPOSAL | Verified ModelAssembly with component artifacts/ports/processor/codec/license/version, separate from replicas and application workflows |
| DMM-007 | PROPOSAL | Distinct trusted TaskDriver algorithms for causal AR, CTC, RNN-T, TDT, finite choices, duration-prosody TTS, nested RVQ, flow/diffusion, joint AV |
| DMM-008 | PROPOSAL | Source-registered scheduler/solver/guidance/seed is semantic and may not silently change as hardware speed optimization |
| DMM-009 | PROPOSAL | Model topology orthogonal to TensorEncoding/activation transform/accumulation: ternary PTQ/PQ2+Hadamard is compulsory example |
| DMM-010 | PROPOSAL | Source CPU reference where claimed runnable and original checkpoint+processor goldens before accelerator/full-model support status |
| DMM-011 | PROPOSAL | Media model state is typed multi-track TaskFrontier, but one root publisher and state effect/fork/rollback authority persists |
| DMM-012 | PROPOSAL | AV coupled architectures publish valid joint CommitGroups, not independent arbitrary TTS mux |
| DMM-013 | PROPOSAL | Add RealtimeSession with independent capture/perception/response/audio/video lanes and Session/Turn/Transport/Physical epochs |
| DMM-014 | PROPOSAL | Separate prepared, committed, delivered, acknowledged and played/presented frontiers. Playback is evidence from client, never inferred from transport alone |
| DMM-015 | PROPOSAL | Add canonical MediaEvent stream with payload lease, clock, source model, lane, sequence, commit group, finality and cancellation epoch |
| DMM-016 | PROPOSAL | Soft real-time QoS and stage-aware admission/priorities, not hard real-time promise; audio responsive while video yields at proven solver boundaries |
| DMM-017 | PROPOSAL | Transport adapters for CLI/SDK, HTTP/SSE, WebSocket, WebRTC, optional SIP and async media jobs all share one native EngineCore |
| DMM-018 | PROPOSAL | Versioned model/task/protocol capability manifest; OpenAI-compatible only for tested observable fields and event behaviors |
| DMM-019 | PROPOSAL | OpenAI Realtime barge-in and Gemini Live reconnect are external protocol conformance scenarios, not reasons to copy proprietary model internals |
| DMM-020 | PROPOSAL | Hosted generation and voice services, an external decision service, proprietary realtime services live at explicitly authorized TaskServiceAdapter, not DeviceBackend |
| DMM-021 | PROPOSAL | External job Unknown/reconcile/idempotency/asset rights/cost/privacy are first-class; no blind retry of accepted remote creates |
| DMM-022 | PROPOSAL | Hardware providers may specialize at op, layer, model-stage or closed whole-model graph only with exact source/shape/numeric/state proof |
| DMM-023 | PROPOSAL | Do not force single global token scheduler, one media codec rate, generic CLIP text encoder or model-name-based device routing |
| DMM-024 | PROPOSAL | Source-specific Wan2.2 high/low denoise experts are timetable-based, not token-level sparse FFN expert routing |
| DMM-025 | PROPOSAL | Direct finite option-logit TaskDriver for open decision models with raw/calibration provenance; generated JSON is not source-equivalent |
| DMM-026 | PROPOSAL | Native full-duplex (or Qwen-Omni) model supported without obligatory ASR→LLM→TTS intermediary; separate cascade also supported |
| DMM-027 | PROPOSAL | Do not broadly label "streaming" until PCM/codec valid chunk latency and interruption are actually measured under source checkpoint |
| DMM-028 | PROPOSAL | Per-model loaded immutable weights may be shared across sessions only when full exact source identity and state isolation permit |
| DMM-029 | PROPOSAL | Add MM13–MM24 real-time/serving deterministic mock tests to MM01–MM12, then model-specific fixtures and physical support tiers |
| DMM-030 | PROPOSAL | Document support status tiers Recognized→Semantic→CPU→PartialPhysical→FullPhysical→Streaming→Realtime plus separate OpaqueHosted |
| DMM-031 | PROPOSAL | Preserve default zero-config simple UX; advanced model/codec/device/transport knobs progressive disclosure and explicit numerical receipts |
| DMM-032 | PROPOSAL | Permit adding a new model via source adapter, task driver, codec and qualified kernel without editing central Router, TaskCoordinator or protocol parser |

None of DMM-001..032 has independently been ratified as frozen code ABI by the user. The user asked us to specify these to enable later implementation decisions and review.

## 4. Open design decisions OMM-001..OMM-022

| ID | Priority | Exact unresolved question | Default until resolved |
|---|---|---|---|
| OMM-001 | High | Strict bit-exact versus source-acceptable perceptual/numeric tolerance for media backends | enforce source-verified model's existing numeric acceptance; no silent widening |
| OMM-002 | High | First legally/hardware feasible ASR source checkpoint Whisper or Parakeet? | choose only after exact model/processor license and CPU feasibility |
| OMM-003 | High | First TTS: Kokoro small non-AR followed by dual-AR depth and full-duplex? | architectural plan, not implementation authorization |
| OMM-004 | High | First image flow checkpoint and original source/VRAM/license? | source-free mock then verifiable small model |
| OMM-005 | High | Exact autoregressive image MoE release generator AR/solver path | audit selected published entrypoint before claiming |
| OMM-006 | High | Native CUDA/HIP LLVM23 vendor SDK availability/approval | remain unqualified, do not download/change permissions |
| OMM-007 | High | Real video 3D VAE tiling/temporal shape and memory fit targets | no unsupported full-model GPU or realtime claim |
| OMM-008 | High | Physical GPU allocator headroom, UMA/GTT and cross-process contention accounting | conservative operator cap and measured evidence |
| OMM-009 | High | Trusted safe physical GPU quiescence and unknown-work recovery | retain quarantine/fence; process restart policy |
| OMM-010 | High | Default real-time P95 mouth-to-ear/TTFA/jitter target by hardware/load | configurable soft QoS, no numeric SLO invented |
| OMM-011 | High | Playback acknowledged vs actually heard after reconnect | represent uncertainty/provenance explicitly |
| OMM-012 | High | Voice cloning, biometric voice reference, face/video content consent and retention | deny remote egress/clone unless explicit authorized |
| OMM-013 | Medium | Async hosted provider cost, idempotency and acceptable retry policy | Unknown/reconcile after ambiguous create |
| OMM-014 | Medium | WebRTC/SIP optional transport packaging/licensing and auth default | transport adapters optional; loopback safe defaults |
| OMM-015 | Medium | Whether same session can change model/voice/codec mid-turn | require source-compatible checkpoint+explicit new generation |
| OMM-016 | Medium | Live session persistence and encrypted checkpoint/codec/buffer retention | ephemeral bounded by default, no silent media persistence |
| OMM-017 | Medium | Source licensing for dual-AR/rectified-flow/various commercial model weights | per-checkpoint license receipt and refuse unlicensed use |
| OMM-018 | Medium | Vendor API exact released model name “gpt-realtime-2.5” | **unverified name**; versioned capability contract not hardcoded |
| OMM-019 | Medium | ModelAssembly codegen component ABI freeze and dynamic plugin policy | initially one Rust crate and static registered modules |
| OMM-020 | Medium | Fine-grained per-session cancellation vs GPU process/physical-card epoch | session/turn cancel only, card epoch reserved for card failures |
| OMM-021 | Medium | What constitutes authorization to external provider via CLI alias? | explicit model/provider/profile/egress opt-in, no path string magic |
| OMM-022 | Medium | Continuous compilation/autotuning overhead versus audio latency | precompile warm shape bands, no hot path dynamic search |

Each question must be resolved by user instruction or measured/source-qualified research; a future agent cannot silently mark it solved because a source README claims performance.

## 5. Inherited exact model-priority and project-path requirements

Canonical repository: /home/pvl/spaces/abird/src/zerogravity; connector root /home/pvl/spaces/abird. Do not read files from unrelated aw/abird-health projects. Important model order: Qwen3.5 0.8B → 2B CPU through 8K, then NVIDIA CUDA, AMD ROCm/HIP, Vulkan; after those CPU-first Gemma4 E2B, Gemma4 26B-A4B, Qwen3.8 27B. Separate future multimedia research does not reorder. Rust-owned engine, no foreign inference runtime hidden under Rust, source verified semantic math, no fake GPU proof or unsafe early StateRoot publication. Built-in multi-model router and high-performance serving remain user goals, not implementation claims.

Previous user-approved original research topics (Strix Halo cooperative NPU/GPU, DGX Spark, TPU/wafer-scale AOT device, distributed placement/tunnels, extreme MoE and quant specialization) remain first-class architecture goals; the v5.1 review adds modality/task/serving semantics but does not supersede those commitments.

## 6. Research and source provenance

Official source URLs and claimed model mechanisms are individually recorded in:
- multimodal-research/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md
- multimodal-research/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md
- multimodal-research/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md
- multimodal-research/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md
- multimodal-research/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md
- multimodal-research/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md
- multimodal-research/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md

Verified against publicly available upstream docs as of 2026-10-15, not tested on all exact source checkpoints. No source facts from third-party blog/forum are considered stronger than original repositories/papers/verified APIs. The original comprehensive implementation archive and decision register contain full historical user decisions and remain separately hashed in new master handoff.

## 7. Changes explicitly forbidden during this documentation task

- No source code, Rust/Cargo, tests, scripts, Nix/CI, live network APIs, model weights, driver permissions, daemon configuration or release modification.
- No staging/committing other agents' files, no whole-tree stashes/reset/rebase.
- No rewriting older user decisions or changing their status to approved.
- No fabricated subagents, upstream benchmarks, provider model IDs or real hardware media inference proofs.
- No new runtime architecture implemented merely because diagrams and type sketches are written.
- No automatic external egress or provider account installation.

New design files and a separate manifest/handoff are the authorized deliverables.
