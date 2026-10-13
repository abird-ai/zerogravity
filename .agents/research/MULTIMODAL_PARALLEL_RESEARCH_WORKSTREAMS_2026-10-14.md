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
