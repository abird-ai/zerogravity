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
