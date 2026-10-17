# TODO — Single-process model and pipeline composition

Status: active
Provenance: fee0bb2
Design: [README.md](README.md)
Owners: model-acquisition, model-execution, serving-and-speculation, multimodal,
memory-and-streaming. All model/source/media execution remains subject to
their existing numeric/source/physical acceptance gates.

## Uncommitted

- [x] Source-neutral ordered model target matrix for Qwen3.5+/3.5/3.8,
  Gemma4, DeepSeek V4.1 Flash, GLM5.3 Flash/full, Muse Spark, Mistral/Llama,
  additional OSS, QwenImage2.1/QwenTTS/QwenASR/Whisper/Canary/Parakeet.
  Source name/layout/release status distinguished from executable support.
  Exact local 0.8B/2B Qwen GGUF baseline not promoted to arbitrary HF repo.
  DeepSeek V4.1 Flash/GLM5.3 Flash payload hold represented explicitly.
- [x] Pure typed ComponentContract, ModelInstance, Pipeline, Edge and ApiRoute
  types in existing neutral core. Compile bounded DAG with matching semantic
  ports, required producers, topology, source-to-output productivity,
  unambiguous exposed routes and shared model IDs. Output is NOT a server,
  loaded model or qualified media graph.
- [x] Default CLI `models [list|inspect]` static truthful registry and
  fail-closed `models run|stream`. TOML schema `zerogravity.pipeline.v1`
  and repeated `--model`/`--step`/`--link`/`--route` flags lower into
  exactly the same graph validator; config file size ≤256KiB. Both single-
  model and multi-model diagrams/example configs exist. `pipeline check`
  is pure and `pipeline run` is explicitly NOT IMPLEMENTED.
- [x] Bounded pinned HF/GitHub `--bundle --include` opaque Auxiliary-only
  data selection for Diffusers/multi-stage speech assets, reusing one Hub
  pin/tree, existing filters/admission/cache publisher and exact offline
  bundle-only source identity. No invented model/codec completeness, native
  Xet bundle, unknown-size cap, arbitrary subdir recursion or executable
  checkpoint interpretation.
- [x] Implement source-neutral explicit optional chat-context policy:
  per-pipeline enabled toggle (default absent/off), shared text LLM model
  instance, max1..16384 UTF-8 bytes, typed synthetic prior-turn
  context.value, optional ASR/diarization/transcript-text/speakers-text/
  translation stage context inputs. Same-session prior committed visible
  turns only; compiled graph has no cyclic/current-turn history flow or
  raw KV injection. Disabled policy leaves edges but delivers None.
  Source-real contextual speech/diarization/translation, privacy-safe
  snapshot capture and context A/B quality remain NOT IMPLEMENTED.
  TOML and repeated flags compile identical shape; include full
  contextual-voice-assistant example and adversarial negative tests.
- [x] Frozen source-only acceptance: exact MAIN
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passed 1,032 Rust executions, zero failures, 69 intentional ignores
  across 230 suite runs, plus 18 namespace Python tests, release CLI,
  Nix/shell formatting, installer, CPU numeric self-tests and strict
  default/production/diagnostic Clippy. Actual Nix filtered source-integrity
  includes the new core/CLI modules and five TOML example configs.
  Original Qwen3.5 release cache_prepare source/model parity independently
  passes for both models, all 248320 logits, all42 StateRoot domains,
  2 tokens, session/source leases, mutation/reclaim: direct 21.05s and
  HF-import 24.10s. Both successful after an initial aborted connector
  tool invocation and explicitly corrected two test-local shadowing errors
  plus a bad nested help quote; no numeric/source/runtime tolerances relaxed.
  Acceptance uses an exact-path signed local source commit. No device
  launch, data download, live server, license approval or real inference extension.
- [ ] Source execution: bind actual learned model, tokenizer, processor
  assembly and port formats in original ModelCatalog/TaskDriver before
  enabling any pipeline run or route listening.
- [ ] Single-process multi-model scheduler: immutable source residency,
  exact shared instance reuse/replicas, typed cross-model buffer leases,
  physical ready fencing, bounded ingress/admission/cancellation and
  model-specific batch/fairness. One TaskCoordinator publishes StateRoots.
- [ ] Pipeline runtime/hot reload: per-session model identity, graph generation,
  buffered media/source epoch, legal stage yields, failure rollback and
  atomic route transition. No stale audio, image or cross-tenant leakage.
- [ ] API serving: explicit auth/bind/route config and tested OpenAI
  chat, ASR, TTS, images; SSE/WS/WebRTC soft-realtime with content/size
  bounds and accurate NOT IMPLEMENTED capability reports until qualified.
- [ ] Context-aware source-real voice pipeline: stage-specific ASR prompt/
  vocab support or separate contextual transcript reranker, acoustic-first
  diarization and source-preserving contextual translation. Implement
  prior-committed same-session snapshot freezing, cancellation epochs,
  explicit consent, redaction, bounded history and no cross-user leaks.
  A/B compare WER/CER/DER/translation adequacy/latency for context
  ON/OFF; never override contradicted acoustic evidence. This does not
  follow from shape validation.
- [ ] Actual staged ASR + diarization (speaker-tagged timestamps), ASR
  -> Qwen -> TTS one coherent command; audio sample clocks/codebook
  playback/VAD/barge-in and image latent/complete preview frames with
  backpressure. Validate codec/latency/source on real media hardware.
- [ ] Optimize model execution: Qwen3.5 2B, Qwen3.8 27B
  resident+SSD, Gemma E2B/26B/31B, then Qwen Flash-Next QSA/GDN/MoE/
  learned n-gram embeddings/MTP and separate speculative n-gram/draft,
  with measured model-real speed/quality/physical safety gates.
- [ ] Complete remaining exact Q2/Q3.x/Q4 quant source pipelines,
  full native image/audio/NeMo assemblies, token/codec quality and
  range/resume/cancellation before declaring "all models supported".
  Keep DeepSeek V4.1 Flash and GLM5.3 Flash strictly pipeline-only now.

## Committed

- [x] Prior accepted source and family contracts remain in
  model-acquisition, model-execution, multimodal, serving, memory and
  earlier signed history; none of those prior unqualified model/runtime
  implementations are retroactively promoted.

## Tests and verification

- [x] Opt-in previous-turn context *shape* passes TOML and repeated flags,
  with toggling ON/OFF on the same graph, absent-by-default channel, and
  static bounds/unknown model/non-language source/no consumer/spoofed
  current-turn feedback refusal. No hidden prompts or cross-user data
  can be obtained by configuration checking (it has no data/transport
  access). Runtime content/biometric/clock and privacy tests remain open.
- [x] Core shape validation passes: one-stage Qwen, model sharing across
  two routes, ASR+diarization+optional contextual translation+LLM+TTS
  and image flow. Stages are semantic placeholders only.
- [x] Source-free negative tests pass: cycles, orphan stages, wrong
  typed ports, missing speaker source, duplicate model ID/API/path/input,
  unknown TOML, unsupported model/media routes, invalid quant/role hold,
  context spoof/size/provider/current-turn linkage and disabled output.
- [x] Default CPU & acquisition CLI tests: offline `models list/inspect`,
  `pipeline check --config` and equivalent flags/ON-OFF context toggle;
  `run/stream` refuse without source/physical/API effects.
- [x] Acquisition and CLI gates: original HF/GH closed weights selection
  unchanged; `--bundle` separate Auxiliary role/source, original cache
  publisher and exact offline cache. Native Xet bundle refuses before
  Hub/CAS. No invented FirstSeen trust or guessed release payload;
  conflicting options fail before credentials.
- [x] Full frozen isolated `scripts/check.sh` 1,032/0 failures,
  four strict artifact/CLI profile lint and core CI, Nix filtered-source
  integrity and original two Qwen reference release regressions pass.
- [ ] Later: source-real trained multi-model numerical, media correctness,
  physical device completion, realtime SLO/fairness and zero-copy lease gates.

## Deferred

- [ ] OSS release/distribution naming and "latest" pointers remain
  observational until an exact immutable source and codec version is bound.
- [ ] Remote routed pipelines/peers, encrypted hosted gateways and
  implicit zero-copy cross-device are not enabled by source-free shapes.
