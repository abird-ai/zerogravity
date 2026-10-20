# Serving and speculation

Provenance commit: `6d74f76`.
Scope: P09/P12/P15/P16/P20/P22.
Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md),
[36 serving gates](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md),
[Qwen parity and decoding](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md).

## Uncommitted

- [x] **Genuine projector and stock0.8B loader prerequisite:** unchanged
  pinned llama converter produces source-audited BF16 visual projectors for
  both original models; independent complete tensor-byte checks pass.
  Stock pinned vLLM V1/official GGUF plugin loads original0.8B language-only
  weights; all320 backbone roles validate. Plain/observed B1 teacher runs
  preserve both full vocabularies,496640 native BF16 logit words, exactly.
  Native dtype and separate F32 observer representation are explicit; no
  sampler-cast, all42, cross-engine arithmetic or performance qualification.
  Owned process/group/container/EOF cleanup passes. Both projectors are
  genuine source conversions, not dummy parameters or a modified engine.
- [ ] **Stock2B loader and matched benchmark:** first actual2B loader reaches
  model initialization but its observer refuses335 tensors versus the assumed
  320. Original GGUF contains15 extra MTP tensors; the additive audit binds
  all335 descriptors, exact stock exclusions and all320 backbone roles.
  All19 diagnostic fixtures pass; fresh plain capture completes, observed
  capture and paired validation remain pending. Preserve the failed attempt.
  Then run unconstrained OpenAI workloads with
  actual common prompt IDs/stop policy, equivalent hardware/resources and
  cold/warm conditions via canonical tools. No vLLM speed result is accepted.

- [ ] **Production CUDA graph admission:** close every same-card constructor,
  transport, allocation, external event/storage and final-drop path through
  original queue/completion ownership; admit graph metadata/driver resources.
  Private graph replay does not enable production graph selection. Test-only
  host protocol14 cases/three ownership negatives pass. Separate original
  transport14 cases/two intended ownership failures, strict Clippy and no_std
  also pass in an isolated harness. Prepaid retirement and complete integration
  remain separate work, with participation unproven.

- [ ] **P12 prefix scheduling performance:** measure concurrent native prefix
  cold-process throughput/TTFT against the accepted max32-prefix implementation.
  Warm observations below cover both models plus explicit B1 scheduling cost;
  genuine equivalent-source vLLM comparison remains open.

- [ ] **P12 fused continuous GPU serving:** extend the accepted CUDA B2/B4/B8
  HTTP path with paged hybrid KV/recurrent cache, physically safe
  prefill/decode overlap and fairness. Unequal decode frontiers and
  compatible refill are accepted within the B7 source/context scope.
  Extend full-logit/StateRoot qualification to other advertised trained
  profiles/backends and full GPU prefill. Measure actual p50/p95 TTFT,
  true token ITL, committed SLO-goodput, fairness, memory and physical
  dispatch counts against genuine pinned vLLM V1 with equivalent
  source/quant/tokenizer/hardware, cold/warm conditions and caps.
  Local release smoke and fewer launches do not qualify performance.

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

### 9a19901 — batched graph and baseline qualification

- [x] **Private full-model CUDA graph qualification:** both original models
  B2/4/8, StateOnly teacher prefixes (including genuine EOS input), future full
  logits and outside-body greedy sampling match independent CPU/B1/native
  full248320/all42 state. All six normal cases and18 fresh unknown-completion
  cases pass actual replay, stale/Ready/pressure refusal and reclamation or
  original quarantine. RoPE launch reduction additionally passes both models'
  170 ordered-angle cases and full native B2/4/8 normal/greedy comparisons.
  Physical32-case result87b85eff. Composed strict CUDA/Vulkan, complete
  isolated1243-test phase, both original release routes,14 worker and12 HTTP
  CUDA regressions pass; composed result e76248e8.

- [x] **Genuine GGUF baseline build prerequisite:** official out-of-tree
  vLLM GGUF plugin e2b8ad532b8b builds unchanged against pinned vLLM0.31/
  Torch2.13/CUDA12.9; wheel af9de736,129 RECORD entries/111 packaged Python
  files and full151-source pre/post pins validate. Owned container/group/EOF
  cleanup passes; no GPU/model ran. Stock text loading with a genuine visual
  projector and language-model-only mode, activation/cast/reduction profiles,
  numerical quality and matched benchmarks remain unqualified. Lack of an
  in-tree GGUF registry does not exclude this official plugin route.

- [x] **Bounded native-prefix warm comparison:** 0.8B strict B8 ABBA and 2B
  natural concurrency8 BAAB, each plus B1,16 serialized samples/72offers/
  72completed/576 output tokens. Exact full IDs/text/usage/finish and cleanup.
  Mean concurrent waves32.301205→18.33249s(0.8B,-43.2%) and
  42.19858→28.457115s(2B,-32.6%); actual forward dispatches
  1722640→302160(-82.5%) and1668496→294384(-82.4%). B1 means
  4.43954→4.42998s/5.60352→5.652635s, unchanged kernel counts.
  Both concurrent workloads still miss10s TTFT; no consistent ITL-tail gain,
  source-correct goodput unset. Original strict2B B8 attempt failed its
  declared scheduling guard and remains unqualified; fresh naturalC8 protocol
  permits actual2/4/8+B1, and both successful new samples happened to use54B8.
  Conditions/limits and observed memory in FUSED_GPU_BATCHING; no vLLM claim.

### f788929 — native teacher-prefix serving

- [x] **Bounded same-engine cold/warm comparison complete:** both original
  models, warm ABBA(0.8B)/BAAB(2B) and cold-process ABBA,16 serialized samples,
  128offers/128completed/1024 output tokens. Every sample has8 genuineB8
  cohorts, zero refusals/errors/discards, and identical complete IDs/text/usage/
  finish reason across providers. Provider4GiB/device5GiB on BOTH builds.
  Mean warm waves improve15.3%/6.1%; cold13.8%/6.0%. Actual dispatches fall
  16.9%/14.4%. No consistent ITL-tail improvement or10s TTFT-SLO success;
  observed goodput0/source-correct goodput unset. Exact values/conditions in
  FUSED_GPU_BATCHING. Initial equal4GiB attempt remains rejected evidence.
- [x] **P12 native teacher-prefix serving:** same full-model BatchProgram
  consumes one teacher token/request/wave with StateOnly readout, independent
  causal positions and original all-root publisher. Both modelsB2/4/8 pass
  per-teacher CPU+B1/all42/future full248320, trained EOS input, abandoned Ready,
  stale/invalid rows and final0. Two physical pressure and two fresh unknown
  completion cases pass; unknown retains1817/1712 handles and4GiB envelope,
  unchanged roots/all42 and fence, without recovery. All14 worker and12 HTTP
  scopes pass, including ragged prefix/tail partition, no prefix-generated IDs,
  Ready cancellation/regroup, output-credit fairness and final active/queue0.
  Three readout-phase health counters close against actual native bands and
  four sampler stages/greedy cohort. Complete isolated check1241 Rust/0failed/
 70ignored/248suites+18Python/release/installer/Nix; strict CUDA/Vulkan/qualifier
  and both approved original cache-direct/HF-import release parity routes pass.
  Two test-fixture/assertion failures and one missing-CUDA_PATH prelaunch
  refusal are preserved; no production arithmetic or acceptance limit changed.


### d0a6084 — source activation preparation and physical acceptance

- [x] **P12 block-local CUDA activation preparation:** one native32-lane
  workgroup preserves source ordered finite/max/scale/rounding/code/sum
  arithmetic for Q4_0/Q4_1/Q8_0/Q6_K projections. Original seven buffers,
  composite/status stages, leases, queue and completion remain authoritative.
  Cold checked bindings remove repeated warm handle/capability/alias setup;
  only explicit non-CUDA selection retains the old lowering. CUDA errors
  propagate. Projections use5 launches versus10 (Q6_K:8), before certification.
- [x] **Current physical acceptance:**272 preparation cases/all7 buffers/
  1,483,608 words exact; both original0.8B/2B complete B2/4/8 unequal-frontier
  graphs pass all248320 logits/all42 domains versus independent CPU AND B1,
  feedback/refill/abandoned Ready/stale packets/final reservation0. Two real
  pressure cases pass prelaunch refusal, B4 progress and B8 after release.
  Eight serving-worker and six HTTP cases pass both models B2/4/8, EOS,
  credit/cancel/disconnect/replay and final0. Four fresh graph/sampler unknown
  completions retain1824/1719 and1828/1723 handles, unchanged roots/all42,
 4GiB envelope, fence and no recovery. Complete isolated gate1218 Rust/0fail/
 69ignored/248 suites plus18Python/release/installer/Nix; strict CUDA/Vulkan/
  qualifier and both original release parity routes pass.
- [x] **Bounded same-engine0.8B warmed ABBA:**32 offers/32 completed/256
  output tokens across four fresh serialized samples; all eight requests/sample
  have individual token events and identical IDs/text/usage/finish reason.
  Each sample executes eight B8 cohorts with zero refusal/failure/cancel.
  Mean wave39.14206→33.163665s (-15.3%); dispatches2,072,360→1,722,640
  (-16.9%). Mean of per-sample client ITL p95:0.3911→0.3933s, no tail
  improvement established. Both miss the declared10s TTFT SLO; goodput0.
  Source-correct SLO goodput remains unset. This is ZG versus ZG only.
### 1b599df — serialized owned benchmark provider lifecycle

- [x] **P22 serialized owned provider lifecycle:** optional strict
  `--provider-lifecycle` specs launch one fresh foreground process per
  actual ABBA/BAAB sample, with literal loopback vacancy checks, bounded
  startup/shutdown/logs, exact model readiness, finite optional warmup and
  isolated receipts. `cold_process` means a fresh process without warmup;
  it does not mean evicted OS/storage/device caches. `warmed_process`
  adds one ordinary unmeasured request. Direct argv only; Podman binds
  the first fresh CID and stops only that owned container via the same
  local runtime prefix. Cleanup precedes evidence writes; receipt/log/
  reader/process/endpoint failures block the next sample. Environment
  names and redacted argv are recorded; private raw logs are capped/drained.
  Existing already-running-endpoint comparison remains available.
- [x] **Lifecycle acceptance:** source-free actual localhost ABBA/BAAB,
  exclusive ownership, warm/cold, startup/model/hang/warmup/measurement
  failures, TERM-to-KILL, CID overwrite/missing/stop failure and five
  persistent receipt-I/O failures passed; all owned readers/processes/
  endpoints retired before any next sample. Root independently passed
  comparator/benchmark self-tests and pycompile. Two real GPU providers
  also passed SERIAL startup, one warmup request and complete shutdown:
  accepted B7 ZG mixed-GGUF CUDA, then immutable pinned vLLM0.31.0-cu129
  original-HF BF16. Each returned GPU telemetry to baseline41MiB; live
  samples1059/4203MiB are advisory samples, not peaks or reservations.
  Those sources/quant/math are explicitly unmatched: this is lifecycle
  qualification ONLY, not a benchmark or numerical/performance parity.
  Complete isolated gate1202 Rust/0failed/69ignored/248 suites plus18Python,
  strict CUDA/Vulkan/qualifier Clippy and both approved release parity
  routes passed. Matched source-correct SLO goodput remains unset.


### dbb5f4f — unequal native decode frontiers and bounded token observations

- [x] **P12-B7 unequal-frontier native CUDA decode:** per-row positions,
  RoPE/KV append/masking and packed attention softmax lower
  the same full-model ForwardPlan. B2/B4/B8 retain independent original
  authorities, roots, packets, leases and all-row Ready publication.
  Native decode accepts unequal frontiers; prompt-tail grouping stays
  strict and other prefill uses B1. Actual original0.8B/2B engine tests
  passed all248320 logits/all42 domains versus independent CPU AND B1
  at mixed positions255/256/0/31/32/1/510/511, host/device feedback,
  B1 handoffs/refill, abandoned Ready, stale/foreign packets, capacity512,
  inactive AV poison/signed-zero/subnormal cases and final reservation0.
  This engine run preceded metadata/observability-only composition;
  fresh composed worker/HTTP/pressure/abort suites passed separately.
- [x] **P22 bounded publication/delivery observations:** opt-in
  `--enable-per-request-metrics` plus `return_token_ids:true`; bounded
  ASCII request IDs, at most512 non-EOS token records, exact-once IDs
  independent of UTF-8 emission. Observe commit immediately after original
  root publication, queue receive and socket-write completion separately;
  retain the original observation through output-credit parking. GPU
  producer frontiers and existing successful native publication sequence/
  width bind each row without creating a new authority. Standard vLLM
  scheduler-origin TTFT/queue fields stay unset; parsed-request timing is
  named separately. Tools validate IDs, clocks, usage, frontiers/cohorts;
  true client ITL requires one actual token per arrival. Grouped arrivals
  remain unresolved; source-correct SLO goodput remains unset.
- [x] **B7 physical composed serving acceptance:** eight release worker
  cases and six release HTTP cases passed both models at B2/4/8, exact
  CPU/B1 packets/all42 internally, CPU IDs/text/usage on wire, trained EOS,
  limits, Ready cancellation/regroup, capacity1 parking, disconnect,
  survivors/replay and final reservation/active/queued/failed0. New
  unequal-arrival cases offer the short request AFTER the long request's
  first actual ID and observe a shared native B2 publication at distinct
  frontiers. Each model completed5 native cohorts/10rows/935 native
  projection-dot launches/20 native sampler stages in that wire case.
  Two fresh real-pressure cases passed prelaunch B8 refusal without
  mutation, B4 progress under pressure and B8 after release. Four fresh
  full-graph/sampler unknown-completion probes retained all original
  handles in quarantine (1824/1719;1828/1723), unchanged prior roots/all42,
  refused old Ready/further work/other model, retained4GiB and no recovery.
  First abort invocation used a nonexistent package-named binary and
  launched no GPU work; only the four probes were rerun successfully.
- [x] **B7 verification:** complete isolated check1164 Rust passed/
  0failed/69ignored over245 suite runs plus18Python/release/installer/Nix;
  strict CPU/CUDA/Vulkan/qualification Clippy passed. Fresh source-free
  release CLI35/GPU58 passed; focused composed CPU/Core/Model/facade/
  serving185 passed/2ignored. Both approved original cache-direct and
  HF-import release routes passed full248320/all42 for both models.
  Independent launch audit reconciles the unequal0.8B HTTP counter exactly:
  26 B1 state-only graphs*5489 +6 B1 greedy*5510 +5 nativeB2*5728
  =204414 actual completed launches. This includes private certification
  and state/packet copies, excludes cold initialization/maintenance, and
  counts a native cohort once. No throughput claim follows from it.


### d9fa829 — native CUDA B4/B8 HTTP bands

- [x] **P12-B6 native CUDA B4/B8 serving:** finite unpadded8/4/2/1
  bands, live EOS/length/cancel/credit survivor regrouping and actual
  per-band Ready counters. Explicit `--batch-rows 2|4|8` (default2),
  CUDA/active>=rows checks before model/device; default interleaved.
  Checked physical capacity refusal is typed None BEFORE allocations or
  submission; only that result permits smaller bands. Errors/unknown
  completion retain original fence/quarantine. Real scratch leases prove
  cold/warm B8 refusal leaves queue/uploads/allocations/roots/all42/idle
  workspace untouched; B4 progresses under pressure, B8 after release,
  full248320 CPU/B1 logits/packets/all42 exact, final reservation0.
  Six same-process release worker cases (.8B/2B × B2/4/8) passed exact
  CPU/B1 state, unequal output limits, handoffs, actual after-Ready cancel,
  bounded capacity1 parking/resume and trained EOS. Initial combined run
  failed CUDA OOM after old-stream cached pools accumulated; cold-load
  cleanup now runs under original initialization/queue guards. Repetition
  physically released completed idle pools (2.51GB/2.13GB→0), all6 pass.
  Four release HTTP tests prove both models B2/B4/B8 JSON/SSE CPU text/
  usage, trained EOS, one terminal, active disconnect, survivors and replay;
  final active/queued/failed0. Source-free CLI26 and GPU53 passed;
  strict CPU/CUDA/Vulkan/qualifier Clippy and full isolated check1148/0/
  69ignored/245suites+18Python passed, both approved release cache-direct/
  HF-import routes exact full logits/all42. Initial direct HTTP invocation
  found a non-serve binary left by the full check; rebuilt explicit CUDA
  server and repeated all4 successfully. Same-frontier serving only;
  unequal-frontier/paged state, full-prefix native prefill and matched
  vLLM source-correct performance remain open.

### 6bfcdbe — bounded offered-work benchmark accounting

- [x] **P22 bounded benchmark accounting:** `serve-bench.py` records
  offered arrivals, client backlog/overload, HTTP acceptance/rejections,
  completion/timeout/cancel outcomes, tenant tails and the full offered
  denominator. SSE chunk intervals remain explicit proxies. Optional
  provider token traces require exact run/wire/provider IDs, complete
  ordered token IDs/timestamps and terminal usage equality; malformed or
  incomplete traces cannot qualify true ITL/SLO goodput. Raw loopback JSON
  snapshots remain provider claims. `serve-compare.py` executes actual
  alternating ABBA/BAAB orders, compares per-request text/usage/token IDs
  and keeps caller source declarations separate from independent parity.
  Both isolated self-tests/pycompile and full check.sh passed. Actual pinned
  vLLM0.31.0-cu129 V1 CUDA startup and four-request warm smoke passed with
  original HF0.8B; this is NOT a matched comparison. Common trained source
  arithmetic, tokenizer/generation policy, caps and real token observations
  remain open; qualified source-correct SLO goodput stays unset.

### 2677d8b — native CUDA B2 HTTP waves

- [x] **P12-B5 native CUDA B2 HTTP:** existing worker dispatches the
  original full-model native plan for compatible prompt tails/decode;
  unmatched rows and other prefill remain B1. Independent session/root/
  packet/physical leases and the all-row publisher remain authoritative.
  Present each committed packet once, retire EOS/length/canceled rows,
  park bounded output before feedback and retry discarded Ready cohorts
  without repeated text/usage/terminal. Actual original0.8B/2B RELEASE
  worker tests passed CPU/B1 compact ID/max/all42 parity, B1↔native
  handoffs, output limits2/5, real trained EOS, pre-feedback/after-Ready
  cancellation, capacity1 credit parking/recovery and final reservation0.
  Actual owned CUDA HTTP alias JSON/SSE tests passed exact CPU text/usage,
  trained EOS stop/EOS-excluded usage, one terminal/usage/Done, active
  disconnect, survivor/replay and final active/queued/failed0. Native
  completed cohorts9/8, projection launches1683/1496, sampler stages36/32;
  one Ready cohort discarded per model. Opt-in fused is CUDA B2 only,
  active≥2; default interleaved and non-CUDA refusals are preserved.
  Strict CPU/CUDA/Vulkan serve+cache and CUDA-qualifier Clippy passed;
  CPU serve14 and GPU53 tests passed (GPU3 intentionally ignored).
  Full isolated check.sh exit0:1147 Rust passed/0 failed/69 ignored over
  245 suite runs,18 Python, release/installer/Nix formatting. Both approved
  cache-direct/HF-import RELEASE routes passed full logits/all42 for both
  models. Real release serve-bench smoke4/4 requests16 usage tokens each,
  eight actual native cohorts each; comparator self-test passed. No true
  ITL/SLO/fairness or matched vLLM performance acceptance; vLLM absent.

### 54fd580 — native CUDA greedy/device feedback

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
