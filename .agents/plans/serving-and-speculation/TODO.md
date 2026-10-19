# Serving and speculation

Provenance commit: `6d74f76`.
Scope: P09/P15/P16/P20/P22.
Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md),
[36 serving gates](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md),
[Qwen parity and decoding](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md).

## Uncommitted

- [x] **First real OpenAI HTTP/CPU router milestone:** opt-in
  `zerogravity serve` with original Qwen3.5 GGUF sources, alias reuse
  (one resident immutable model/lease per canonical path), model list,
  health/admission, text and chat greedy completions and SSE usage.
  Bounded N CPU workers, independent source-specific StateRoots,
  round-robin 32-token prefill / single decode quanta, finite
  queue+active+connection admission, shared host ResourceArbiter
  weight and per-session state envelopes. Loopback-only with
  optional Bearer token; strict duplicate-key JSON and request
  framing, unsupported sampling/roles/media refusal, cancellation,
  SSE keepalives/deadlines and bounded backpressure. Real live 0.8B
  multi-alias + two concurrent text requests, authenticated health,
  host weight/state reservation release, unsupported parameter
  refusal and streaming passed. Full frozen gate:
  1,147 Rust passed/0 failed/69 ignored across 241 suites,
  18 Python fixtures and release/installer/CPU comparator checks.
  Both approved Qwen3.5 0.8B/2B original direct and HF-import
  release parity tests passed exact full 248320 logits/all 42
  StateRoot domains/two tokens/source lease and mutation.
  Strict default and composed serve+SSD/cache Clippy pass.
  The separate Nix offline source-integrity derivation is blocked:
  pinned nixpkgs 7a0f122 is not cached and the connector cannot
  fetch it with available SSL trust anchors; Nix formatting DID pass.
  Not fused GPU continuous batching or full OpenAI/vLLM API parity.
- [ ] **Next genuine multi-request GPU execution:** physically admitted
  grouped prefill/decode over compatible source/position/shape and
  actual GPU completion/quarantine, paged KV/prefix cache, source
  generation+tenant/epoch admission, sampled decoding, and measured
  committed-token p50/p95 latency/goodput versus pinned vLLM.

- [ ] Connect accepted core `ContextOwner`/`ContextReader` snapshot
  capability to the real router's authenticated session/turn frontier:
  source-scoped application/shared grants, per-application and per-tenant
  ownership, immutable complete context records, explicit ON/OFF context
  selection, short-lock optimistic source epochs and cancel/revoke
  invalidation. Recheck `require_fresh` at TaskCoordinator delivery and
  bind source/graph generations to model/media StateRoot publication.
  DecisionAdapter is a typed callback boundary, not trained NanoJev
  logits or actual ASR context support. Cross-session/host source sync,
  bounded CPU staging and QoS, media quality A/B and revocation/stream
  cancellation remain open.

- [ ] One router must snapshot each enabled, source-granted context
  stream at the declared turn/graph epoch before live ASR. Default only
  committed user/assistant session content; explicitly selected tool
  call/result, application-owned system/developer prompts and authorized
  shared/cross-user records are eligible. Strict source ownership and
  requested grants, not a blanket category blacklist, govern access.
  No same-turn LLM feedback loop or unauthorized sources/tenants, raw
  unowned KV pointers or tool-secret leaks. The ON/OFF toggle leaves
  typed graph topology stable; OFF performs no retrieval. Bound
  configurable 4KiB..1MiB plus physical admission, ordered filters,
  typed decision models, credit/backpressure, stale ACKs and source-real
  accuracy/consent tests before live mode.

- [ ] Make YAML/repeatable-flag compiled pipeline configs a first-class
  source-bound router input: many model IDs, many typed component DAGs,
  many explicit protocol/route bindings within ONE process; one model
  can serve standalone chat and ASR→diarization→LLM→TTS. Preserve
  direct one-model CLI execution with no config requirement. Current
  pipeline check is pure scaffolding; implement model residency,
  admission, TaskCoordinator execution, generation/hot reload and
  authenticated OpenAI/SSE/WebSocket/WebRTC transport, then enable
  pipeline run/serve only after source/physical tests.
- [ ] Distinguish model-trained MTP weights and Qwen Flash-Next
  learned n-gram embedding bank from runtime n-gram/suffix
  speculative decoding (no extra source required). Verify
  accept/reject and complete rollback/grammar RNG across
  QSA/GDN/MoE before any speed claim.

- [ ] Implement canonical TaskRequest/coordinator workers and CLI/SDK/API integration, model routing/residency lifecycle, explicit supported OpenAI/Anthropic interfaces, streaming/tools/reasoning/schema/sampling and truthful capabilities.
- [ ] Enforce separate API/tenant payload, queued-prefill/residency and physical active-step admission; bounded overload, cancellation, disconnect, stale epochs and output backpressure.
- [ ] Add real compatible continuous/grouped batches, aged chunked prefill, decode/media fairness and finite joint batch/draft/expert/kernel/placement plans; no per-token catalog/JSON/global-lock work.
- [ ] Finish Qwen basics before optimizations; enumerate the complete pinned llama.cpp/vLLM union, public and internal surfaces. Preserve 105 provisional groups/145 document seeds and CLI/API frontiers as unreviewed inputs, not exhaustive parity.
- [ ] Implement every applicable sampler/logits-processor/RNG/logprob/grammar/tool/stop/reasoning composition and its independently verified outputs.
- [ ] Qualify target-authoritative MTP, draft model, n-gram/suffix, EAGLE, DFlash/DFlash2, DSpark and related registered strategies; DSpark2 exact primary source/version remains unresolved.
- [ ] Test all accepted-prefix lengths, target-distribution correction, target/draft RNG/grammar ownership, full KV/recurrent/index/media rollback, physical branch admission and authentic cancellation/quarantine.
- [ ] Jointly benchmark material combinations, adaptive depth/backoff and all offered/rejected/canceled work: committed SLO-goodput, TTFT/ITL tails, fairness, memory and energy versus equivalent pinned vLLM/V1/Omni.
- [ ] Preserve zero untriaged frontier/disposition requirements before any full-parity claim; synthetic acceptance, fewer launches and competitor measurements do not establish local parity or speed.

## Committed

### 6d74f76

- [x] Shared immutable source, independent StateRoots, bounded fair physical workers and actual completion exist as foundations; production serving/parity/speculation is not complete.
