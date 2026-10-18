# TODO — Single-process model and pipeline composition

Status: active
Provenance: fee0bb2
Design: [README.md](README.md)
Owners: model-acquisition, model-execution, serving-and-speculation, multimodal,
memory-and-streaming. All model/source/media execution remains subject to
their existing numeric/source/physical acceptance gates.

## Uncommitted

- [~] Replace fixed prior-visible-only history with a reusable typed
  ContextFilterPlan for independently configurable user/assistant messages,
  tool calls, tool results, application-owned system/developer prompts and
  explicitly named shared/granted records. Default remains session
  user+assistant; filters include/exclude by record kind, exact tool name,
  source origin or text substring in deterministic order. Typed calibrated
  NanoJev-style decision stage references a separately declared Decision
  model, accepts threshold in basis points and configurable unavailable
  policy (omit/reject/keep_authorized). The latter never reintroduces
  excluded, missing or ungranted records. Support TOML and repeatable CLI
  flags; bound source selectors/controls and configurable 4KiB default
  through 1MiB context budget. Configuration *requests* source rights;
  runtime source owner must verify actual authorization, not a hard-coded
  block against application prompts or authorized sharing. No model
  inference/automatic context history access is claimed.
- [x] Source-grant checked context owner/snapshot service: the Rust core
  now holds committed immutable records and exact revocable session,
  app-prompt and shared/cross-tenant source grants behind one trusted
  `ContextOwner`; separately minted read-only `ContextReader` capabilities
  cannot mutate grants or rebind to another principal. Per-session prior
  turn check, bounded latest-whole window, provenance IDs/source labels,
  source epoch retry/freshness validation, and filter-only exposure to
  verified candidates are implemented. Decision adapter receives record
  text only after source grant + previous filters; it returns a typed
  include/omit verdict or explicit unavailable/error. Snapshot records
  original-source attributes and decision audit (not raw text in audit).
  This is an in-process source owner, not an external chat connector
  or pretrained model executor. External source grants must be issued by
  the trusted application before minting a read-only reader. Separate
  tests prove that a policy selector alone cannot grant access.
- [ ] Connect authenticated application/conversation history providers to
  ContextOwner, with durable and revocable grants, owner-authenticated
  tenant+application+session identity, event/graph epoch binding to the
  existing TaskCoordinator and physical memory/residency admission. Wire
  source-real calibrated NanoJev/other decision heads and streaming media
  A/B tests before enabling an actual context-fed ASR/translation router.
  Do not infer inference support from the in-memory source fixture.
- [x] Audited all 6 old detached side worktrees against signed C3/C4/C5,
  v1 and pipeline acceptance commits; preserved earlier unfinished portable
  source patches under .agents/docs and copied nine side-only run artifacts
  into existing MAIN .agents/runs before removing all six detached
  worktrees. One new active, clean worktree
  zerogravity-worktrees/agents-conventions at unmerged e0410c7 is
  intentionally retained. No branch or unrelated user work deleted.

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
  instance, originally max1..16384 UTF-8 bytes and prior committed
  same-session visible turns (historical 4bf624e semantics, now superseded
  by the configurable source selection/filtering policy above).
  Typed optional context.value inputs reach ASR/diarization/
  transcript-text/speakers-text/translation. The graph still prevents
  same-turn cyclic feedback and unowned raw KV injection. Disabled policy leaves edges but delivers None.
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
  explicit consent, redaction, bounded history and no unauthorized
  cross-user disclosures.
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

- [x] Source-authoritative snapshot tests pass: exact tenant/application/
  session isolation, current-turn exclusion, cross-tenant shared grants,
  same-application prompts, independent tool-result filtering,
  authorized-only decision inputs, optimistic revoke during callbacks
  without deadlock, stale/foreign reader refusal, erasure/record provenance,
  bounded latest-whole 4MiB candidate and 1MiB output windows,
  full-or-empty source behavior, decision error propagation and
  secret-free decision audit. Full original scripts/check.sh frozen
  isolated gate passed 1,058 Rust executions/0 failures/69 ignored/230
  suite runs plus 18 Python namespace fixtures, strict lint, release
  build/installer, Nix-format and CPU comparator selftests.
  Independent release cache_prepare exact approved Qwen3.5 0.8B+2B
  parity passes full vocab/42 StateRoot domains/2 token sessions/leases:
  cache-direct17.36s and HF-import21.91s. Actual Nix offline
  source-integrity passed with new Rust module, root README and
  dependencies/lock unchanged; no live model/GPU/provider access.

- [x] Frozen isolated updated-context-filter source-only check.sh
  passed 1,047 Rust executions, 0 failed, 69 intentionally ignored
  across 230 suites, plus 18 Python namespace fixtures, four strict
  production/diagnostic feature lint gates, Nix formatting, release
  installer and CPU comparator gates. Two independently invoked original
  Qwen3.5 release cache_prepare routes pass both approved models,
  full248320 logits/all42 state domains/two tokens/source-lease/
  mutation/reclamation bit-exact: direct18.66s, HF-import23.24s.
  Real offline Nix filtered-source integrity includes two new paths;
  root README, workspace deps/Cargo.lock unchanged.
  This qualifies only config shape plus pure filtering over explicitly
  *provided* already-granted record views. Live source authorization,
  history retrieval, calibrated NanoJev model inference and real media
  execution are not implemented.

- [x] Filter pipeline: independent tool-call and tool-result opt-in/
  exclusion, context source selection for app prompts/shared grants,
  no silent permission acquisition, ordered predicates, user preference
  filters and decision confidence/fallback. Include/omit verdicts must
  not resurrect previously filtered entries or invent inaccessible text.
- [x] TOML + equivalent CLI filters and NanoJev-style typed decision
  references. No inferred upstream checkpoint or executable model. Reject
  duplicate/unknown scopes, malformed origin names, unsourced decision
  models, non-decision heads, invalid thresholds, invalid return variants,
  excessive source/filter/budget and unsupported executable pipelines.
- [x] Opt-in previous-turn context *shape* passes TOML and repeated flags,
  with toggling ON/OFF on the same graph, absent-by-default channel, and
  static bounds/unknown model/non-language source/no consumer/spoofed
  current-turn feedback refusal. Configuration checking has no
  data/transport access; a later runtime may include explicitly
  requested authorized app prompts or shared user records only under
  real source grants. Runtime content/biometric/clock tests remain open.
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
