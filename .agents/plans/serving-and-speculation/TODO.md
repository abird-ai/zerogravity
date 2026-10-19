# Serving and speculation

Provenance commit: `6d74f76`.
Scope: P09/P12/P15/P16/P20/P22.
Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md),
[36 serving gates](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md),
[Qwen parity and decoding](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md).

## Uncommitted

- [x] **P12-B4 native CUDA greedy/device feedback:** same full-model
  batch ForwardPlan plus four native vocabulary-reduction stages,
  independent12-byte packets and one native device-input gather.
  Original packet producer is stamped only at all-row checked commit;
  exact source/session/epoch/root/frontier and physical lease bind
  consumption. Mutable public host ID/max fields are non-authoritative.
  Approved trained0.8B/2B mixed GGUF B2/B4/B8 RELEASE passed six
  source steps/four GPU-feedback steps per row: CPU/B1/native
  full-logit shadow all248320 logits and all42 domains exact;
  compact greedy ID/max bits/all42 domains exact,28/52/100 bytes
  per cohort, no warm value uploads and final reservation0.
  Swapped/duplicate/identical-ID foreign packets, isolated same-epoch
  old roots, restored epochs, invalid IDs, abandoned/reordered/stale
  proposals refuse without publication. Full-vocabulary synthetic
  ties/zero/subnormal/dense-bit/EOS winners and nonfinite/sticky-status
  refusals passed; trained EOS input is exact, HTTP EOS stop unqualified.
  Actual feedback launch/drop retained1827/1722 original handles,
  including input/output packets, all42 CPU prior domains and future
  KV zero tails exact; old Ready/further work/new service refused,
  original4GiB envelope fenced/charged after owners drop, no recovery.
  Both models repeated B2 imported-prefix valid1/256/257/512 boundary,
  exhausted-capacity and reclamation proof; no GPU full-prefix claim.
  GPU53 passed/0 failed/3 ignored; strict default/CUDA qualification
  and CUDA/Vulkan serve+cache Clippy passed. Final isolated check.sh
  exit0:1147 Rust passed/0 failed/69 ignored over241 suites,18 Python,
  releaseCLI/installer/Nixfmt; both approved cache-direct/HF-import
  release routes and benchmark/comparator self-tests passed.
  HTTP remains interleaved; no fused-serving/vLLM throughput acceptance.

- [ ] **P12 fused continuous GPU serving:** next P12-B5 wires
  native CUDA B2 one-token prompt tails and decode through the
  existing `gpu_engine.rs`/worker waves. Attempt each committed
  packet emission at most once; retire EOS/length/canceled rows
  and park backpressured rows BEFORE feedback. Pair compatible
  eligible survivors and use B1 for unmatched rows. Preserve
  all-row publication; abandoned/refused cohorts must retry survivors
  without duplicate SSE text/usage/terminal events. Physically prove
  B1-produced→native-consumed and native-produced→B1 packets,
  genuine EOS, unequal request lengths, disconnect/backpressure,
  exact output/state and resource reclamation before enabling HTTP
  fused mode. Then extend serving to B4/B8, unequal frontiers,
  paged hybrid KV/recurrent cache, physically safe overlap and fairness.
  Extend real full-logit/StateRoot qualification to advertised
  profiles/backends and full GPU prefill. Measure comparable p95
  TTFT/true ITL/SLO-goodput versus pinned vLLM V1 with equivalent
  source/quant/tokenizer/hardware. HTTP stays interleaved until
  its own acceptance; fewer launches do not prove performance.

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

### 35ebc5b — native full-model CUDA ForwardPlan

- [x] **P12-B3 native full-model CUDA ForwardPlan:** one traversal
  over B2/B4/B8, original shared source handles, admitted packed
  scratch and private per-session prior/candidate leases. One actual
  completion yields exact source/session/epoch/root-bound Ready
  receipts; existing all-row checked commit remains the publisher.
  Original trained0.8B/2B mixed GGUF profiles passed all248320
  logits/all42 domains per row versus independent CPU AND B1 over
  two decode steps, distinct histories, invalid/duplicate/foreign/
  reordered rows, abandoned proposals, stale epoch/root rollback,
  sibling continuation, no warm source/value uploads and reclamation.
  Release B2 for BOTH models passed CPU-hydrated prefix0/255/511
  decode at valid1/256/257/512, exact canonical KV tails, attention
  padding dispatch delta and capacity refusal. Actual launch/drop
  tests retained all1820/1715 original handles and the full4GiB
  envelope behind the existing quarantine; prior roots/all42 domains
  stayed exact, old Ready/next step/other model refused, no recovery.
  Source-free GPU51 passed/0 failed/3 ignored; strict default GPU,
  CUDA qualification and CUDA/Vulkan serve+cache Clippy passed.
  Frozen isolated check.sh exit0:1147 Rust passed/0 failed/69 ignored
  over241 suites,18 Python fixtures, release CLI/installer/Nixfmt.
  Both approved original0.8B/2B direct-cache and HF-import RELEASE
  full-logit/all42-domain parity routes passed; benchmark and
  comparator self-tests passed after private loopback setup.
  Same-frontier full-logit CUDA diagnostic only: no native row
  greedy/EOS, unequal-prefix/paged state, GPU full-prefix prefill,
  fused HTTP or vLLM performance acceptance. HTTP stays interleaved.

### 3ec431a — native quantized batch projections

- [x] **P12-B2 source-quantized native batch projections:**
  Q4_0, Q4_1, Q8_0 and Q6_K source-exact B2/B4/B8
  kernels call the SAME original B1 integer/F32 dot.
  Each physical RTX3060 CUDA test used one native B>1
  launch, bit-identical 128/256/512 F32 outputs versus
  original per-row B1 GPU reference. Original row-specific
  dynamic Q8 scales/offsets/codes, strict typed layouts,
  source-free bad-geometry tests and no CPU dot fallback.
  THESE are synthetic-source operator fixtures only.
  Strict source-free and CUDA Clippy pass; frozen gate
  1,147 Rust tests/0 failed over 241 suites, and trained
  Qwen3.5 .8B/2B direct-cache/HF-import source parity
  remains full logits/all42 StateRoot domains. Next:
  batch-native resident ForwardPlan integration and
  physical B2/4/8 trained full-model root parity.

### 08be9c6 — resident cohort publication

- [x] **P12-A1 source-bound resident GPU cohort:** ONE
  `ResidentModel` original-source method stages B2/B4/B8
  independently completed `ResidentReady` per-session proposals,
  checks all session RootAttempts and complete domain leases
  before one `ResidentService` fence-guarded publication.
  Invalid/foreign/abandoned/stale rows refuse all sibling
  root changes. Physical RTX3060 CUDA Qwen3.5 B2/B4/B8 matched
  full 248320 logits and all42 state domains per row against
  independent CPU; B4/B8 memory returned to baseline.
  Live OpenAI CUDA B2 short-prompt wave published two roots
  in one checked cohort; four separate GPU full-model forwards,
  six completed quanta, zero errors and `gpu_fused_batches=0`.
  Real batch-native ForwardPlan kernels remain pending.
  Acceptance: physical CUDA resident B2/B4/B8 exact all248320
  logits and all42 StateRoot domains per independent session,
  source/session state snapshots, abandoned/invalid/stale rollback,
  one source upload and lease reclamation; HTTP B2 cohort
  publication and active disconnect/survivor/replay pass.
  Strict CPU/CUDA/Vulkan feature Clippy -D warnings; full
  scripts/check.sh exit0: 1,147 Rust tests/0 failures/69 ignored
  across 241 suites plus release CLI/installer, 18 Python fixtures
  and Nix formatting; trained Qwen3.5 .8B/2B direct-cache and
  HF-import release parity both pass exact full logits/all42 domains.
  Separate offline Nix source-integrity cannot evaluate because
  pinned nixpkgs is absent from the local store and connector SSL
  CA path is unavailable. NOT a source-test failure.

### 9790fa2 — resident GPU OpenAI serving

- [x] **P09 GPU OpenAI serving / finite batch-wave scheduler:**
  opt-in CUDA/Vulkan/HIP resident model with independent physical
  StateRoots and device-side greedy feedback; no CPU inference
  fallback. Shared HTTP/SSE/greedy output, cancellation and
  original completion/quarantine. Bounded one-GPU worker,
  active/queue limits, source/phase/frontier wave grouping.
  0 fused GPU batches (no batch-native kernels yet).
  Typed GPU profile, CUDA/CCCL fail-fast preflight and reusable
  Nix split-SDK include view. Metrics separate GPU forward API calls (not kernel launches),
  successful quanta, attempted quanta, active/queued/canceled/error.
  Physical RTX3060 Qwen0.8B CUDA: 2 concurrent aliases under one
  model source, six completed quanta/four GPU forward API calls/zero errors,
  canceled active session retired safely and new session succeeded.
  SSE benchmark: 2/2 requests, 4 tokens, 20.02s debug wall,
  0.1998 token/s (NOT optimized/vLLM parity).

### 35b9093 — initial OpenAI CPU router and verified SSD extents

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

### 6d74f76

- [x] Shared immutable source, independent StateRoots, bounded fair physical workers and actual completion exist as foundations; production serving/parity/speculation is not complete.
