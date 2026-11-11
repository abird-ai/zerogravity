# Model execution

## Current direction — signed shared composition; resident HIP comparison

[Fabric transition](../fabric-vnext-transition/TODO.md) is accepted through source
`8c43844`, docs `1c99929`. Genuine multi-row and populated1K–16K0.8B/2B HIP now pass native/SDK/HTTP
correctness and original retirement; final isolated gate passes.
Signed `8ffb511` closes the populated0.8B/2B HIP checkpoint.
Owner prioritizes roadmap items1/2: item1 is signed complete; advance CUDA
reference/state/executable closure on l5 while ordinary27B remains open.
Missing legacy CUDA authority remains distinct from new vLLM references.
Native batching remains subsequent; matched baselines need not wait for it.
Preserve accepted source/operator/lifetime evidence; no repeated Fabric extraction.
Main only; no new or resumed subagents. Last existing reviewer finished.
Assemble substantial shared surfaces, then batch numerical/state/lifecycle/serving
checks; narrow diagnostics only for concrete blockers, complete gate last.
Owner correction: prioritize working implementation and quick matched-vLLM feedback.
Reuse existing end-to-end checks/evidence; add fixtures only for concrete bugs or
uncovered contracts, not speculative coverage or more evidence administration.
Prefer simple shared fixes; optimize measured hot paths. Benchmark each completed,
qualified comparable path without waiting for all models/backends/batching.
Owner resumed items2/3: finish implementation quickly with real-model/serving
feedback, reusing accepted results and avoiding extensive synthetics.
Existing approved model inputs/devices; approved x2 ROCm vLLM runtime retained.
No gfx90c, recovery, push or recurrence restart. Recurrence`17aafa71` remains cancelled.

Owner architecture correction: models compose established shared primitives; do
not build another per-size engine. Family source/config owns geometry, roles,
ties and traversal; backend implementations own explicit numerical/layout domains.
Reuse accepted primitive/ownership evidence; compact real-model composition and
state checks first, deeper diagnostics for concrete gaps or changed optimizations.
CUDA composition is signed in`4f1e827` (signatureG): source-derived layouts/geometry, parameterized
attention/rotary, grouped-head GDN using existing16-head leaves, and bounded
source-page weight streaming on the6GiB l5 device. Fresh small-model resident
and streamed numerical/state diagnostics pass. Shared public SDK/HTTP and streamed
cancellation/quarantine scopes now pass. Large-source composition smoke and the
assembled final gate pass; independent large numerical qualification stays open. Preserve explicit primitive domains; do not
remove guards without implementing their capability. New-shape vendor bitwise
parity is separate from correctness under an established operator contract.
Owner approved a fresh approximately75GiB total native27B HIP GPU budget for the
resident comparison;16GiB host and4GiB context remain. Explicit budget plumbing
is implemented. Resident C1 and all66 original observer-free state/logit calls
pass with independently verified original retirement. Offered C1..64 completes
with matching outputs; final isolated gate passes. Signed`d16f096` (signatureG),
no push; larger CUDA/HTTP/context/batching debt remains separate.

Closeout: no live build/device campaign or owner to inherit. The latest user
request is a signed all-changes/handoff checkpoint, not a new physical launch.
Original budget R1/R2 are complete; do not rerun them or restart cancelled watches.
Use the continuation queue below over historical milestone notes later in this file.

Provenance commit: `6d74f76`.
Scope: P02/P03/P08/P11–P14/P17; CPU truth before GPU specialization.
Contracts: [large immutable sources](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md),
[Qwen basics](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md),
[Gemma family](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md).

## Uncommitted

### Next continuation — no live campaign

- [ ] Establish independent large-CUDA distribution/state accuracy (session task44)
  under its declared numerical contract. Three matching IDs/112 finite domains
  and CPU-logit NMSE0.124241/0.090048/0.043837 are diagnostic, not a post-hoc
  tolerance grant. Preserve exact small-model references and original failures.
  The existing large composition diagnostic now reuses the independently accepted
  source-call archive reader to measure full-vocabulary and all112 state-domain
  cross-profile error together; CUDA Conv3 is compared explicitly with the retained
  history tail of original Conv4, while recurrent/KV retain their logical mappings.
  Results remain `same_contract_reference=false`/`full_reference_acceptance=false`:
  no27B CUDA-contract oracle exists on l5, and the retained22,126,120,608B original
  archive needed for the physical rerun is no longer present on this host.
- [x] Extend small CUDA to genuine populated contexts. Qualification-only resident
  trajectories now populate both0.8B and2B through exact1K/2K frontiers using the
  existing64-row chunk contract: all four cells finish with finite terminal logits,
  all42 state domains present/finite and clean model retirement. This closes the
  prior short-trajectory/65-token-tail gap; existing SDK/HTTP lifecycle evidence is
  unchanged and no new numerical tolerance/pinning was introduced.
- [ ] Localize ROCm2B C4 warmup and C16/32/64 warmup/timed divergence (task39),
  without repinning expected IDs or attributing an unlocalized failure to kernels.
- [ ] Close ordinary27B HIP HTTP504 and populated1K/2K references/serving (task11).
  R10 audit is complete and mixed; R11 remains staged/unlaunched. Review its older
  binary/budget before any fresh successor. Both context captures remain interrupted;
  the waived12GiB host-available threshold is not a current blocker.
- [ ] Continue genuine shared inter-session batching beyond current B8 decode/chunked
  prefill cohorts. Aligned prompt chunks up to the model's admitted chunk size share
  projection GEMMs while retaining private roots; B8 decode reuses the qualified B4
  stateful kernels as two groups. Dynamic grouping remains open, but do **not**
  resurrect the rejected global-active/native-B16 experiment as the next step: a
  same-path0.8B B16 measurement was slightly worse than two B8 groups
  (78.389 vs79.380 tok/s), while2B B16 regressed catastrophically (~44.9s/~42.0s
  warm waves). `MAX_ACTIVE_PER_WORKER=8` and native4|8 therefore remain deliberate
  until a materially different architecture/measurement justifies revisiting them.
  Bounded queueing already handles larger offered populations without widening the
  resident/native ceiling: same non-LTO2B47+3 run stayed exact/native-width8 at
  offered8/16/32 with66.24/68.10/69.06 output tok/s respectively;16 and32 are
  simply two/four queued B8 groups, not hidden B16/B32 execution.
- [ ] Optimize measured CUDA2B/HIP hot paths and run controlled comparisons after
  relevant correctness. Existing short eager/direct-API rates are not general
  HTTP/context/graph performance or proof of vLLM superiority.

### Current real-model feedback — items2/3

- CUDA0.8B/2B: both48-call native schedules now match the unchanged accepted
  references exactly:11919360 vocabulary words/model, all42 state domains/call,
  genuine feedback/frontiers and two repeats. Every case and original invocation/
  group/EOF/device retirement inspected. `native-output-fix-{08,2b}-r1/`;
  frozen binary`ac3b55e3`. Diagnostic milestone, not serving/full-reference or
  matched-performance acceptance. Earlier2B pass and0.8B failure remain.
- Fixed shared output-projection arithmetic: the source uses two interleaved
  K slices below three-quarter64-row tile occupancy, ordered accumulation above.
  Actual47-row output projection matched;48-row projection had43 BF16 differences.
  Typed handle/lifetime/resource/image joins and associated SASS confirm the two
  accumulation chains; no name-only kernel substitution or token-value patches.
  `output-projection-lineage-r{1,2}/`, `output-projection-images-r1/` retain evidence.
  Current recorder's standalone launches still have unsupported-schema argument
  gaps (zero ENTRY bytes); no aggregate-ABI/model-wide lineage grant.
- Extra early-operator capture failed atstep20 when its unchanged GPU snapshot
  arena filled. Its GDN autotuning/state differs from the accepted reference;
  it was not adopted as a replacement. Only fully persisted/hash-checked actual
  projection operands were used diagnostically. `early-operators-08-r1/` and both
  failed/passing native projection scopes retained. First disassembly dependency
  failure and first build-feature error retained; no SDK acquisition.
- Old Nix NVRTC/header paths no longer exist. Fresh native runners use retained
  approved CUDA12.9 package headers/libraries plus installed host C++ runtime.
  Both pre-inference loader failures remain; no SDK download or source repin.
- Original27B latest validation audited: both66-call numerical/state trajectories,
  SDK completion/disconnect and10 lifecycle scopes verified. HTTP504 and original
  BrokenPipe remain. `ordinary-semantic-disposition-r10.json`; not an overall pass.
- Shared source cache now reuses authenticated integrity chunks for adjacent
  page windows within its unchanged budget.30 existing integration tests pass;
  CUDA/HIP release builds pass. No measured serving speedup yet.
- Fresh27B HTTP/SDK scope is staged (`ordinary-semantic-inputs-r11.json`), not
  launched: latest read-only x2 guard12130418688B <12884901888B required.
  Watch`960f23ca` timed out. Owner now waives the12GiB host-available threshold
  and permits swap for new scopes; original guard failures retained. No device change. CUDA regressions
  and the complete assembled gate pass. Signed fixes: CUDA`a57590e`, cache`b9d8355`.
- [x] CUDA C1 quick benchmark: same approved sources/47–50-token inputs,
  greedy3 outputs, BF16/context1024, eager vLLM;8 warmups +16 timed requests/model.
  Mean request latency:0.8B native65.22ms/vLLM90.64ms (1.39× faster);
  2B native114.29ms/vLLM84.34ms (1.36× slower). All outputs and original retirement
  verified in all four runs. No observers or new reference/fixture campaign.
  Load times separately: native6.39/11.96s, vLLM113.13/114.08s; boundaries differ
  (native includes source validation). Short direct-API comparison, not HTTP,
  compiled/CUDA-graph vLLM, long context, batching or general superiority.
  `qwen-cuda-regeneration-tDW0EJ1I/c1-benchmark-summary-r1.json` retains samples.
- [x] CUDA serving hot path now selects directly from packed BF16 logits, returns
  a12-byte greedy packet and reuses the committed device token for decode embedding;
  full-logit readback remains for diagnostics. Same current optimized0.8B feedback
  binary: compact65.944ms vs legacy67.614ms mean, identical IDs/output (~2.5% win).
  Current2B compact feedback111.865ms versus retained native114.29ms/vLLM84.34ms:
  readback helps but is not the principal2B gap. Installed CUDA13.4 NVRTC now emits
  exact-SM cubin under the finite compiler context, avoiding driver PTX-ISA refusal
  without SDK acquisition or driver change. Feedback only; no broader speed claim.
  Next measured optimization target remains2B execution; prior native decode averaged30.97ms/token.
- [x] Shared BF16 GEMV now specializes the common16-lane256/512-term reductions
  at compile time while preserving the exact FMA/segment/lane order. Two quick
  release-feedback samples (LTO disabled only to shorten iteration):0.8B59.494–
  62.306ms vs retained vLLM90.64ms;2B85.383–87.441ms vs retained vLLM84.34ms,
  with identical IDs/output. Existing physical GEMM arithmetic/guard/retirement
  coverage passes. The measured2B C1 gap is now only low-single-digit percent;
  do not add complexity merely to chase ~1–3ms. Controlled standard-release/
  batched comparisons stay separate.
- [x] First ordinary CUDA inter-session decode cohort batches stateless math and
  the vocabulary head across four independent roots while retaining existing
  singleton stateful GDN/KV/attention primitives and atomic publication. Public
  Runtime feedback produced identical `[248068,198,90700]` on every lane:0.8B
  C4 50.50 tok/s vs prior45.04/vLLM60.85;2B C4 41.23 vs prior26.26/vLLM60.55.
  These are quick non-LTO feedback rates, not final matched benchmarks. A B8
  extension was exact on0.8B but flat at50.74 tok/s and was not retained; an
  experimental batched-GDN path diverged after the first token and was removed.
- [x] Narrow cached-GDN convolution batching preserves private recurrent state:
  four decode lanes use distinct retained cache slots for Conv only, then return
  to the existing per-session recurrent path. Rebuilt public Runtime feedback is
  exact `[248068,198,90700]`;0.8B B4 58.08 tok/s,2B warmed B4 ~45.8 tok/s versus
  prior committed50.50/41.23 and vLLM60.85/60.55. Batch code is admitted/prepared
  only when `native_rows=4`; sequential CUDA loads keep the original cold path.
  Quick non-LTO feedback only; broader stateful attention/recurrent batching remains open.
- [x] Cached GDN Decode+Gated now batches the same four cache slots while recurrent
  StateRoot copies stay private. Retained Decode uses gridY batch (`[4,64,1]`),
  Gated uses dense64 head rows; no arithmetic replacement. Public Runtime remains
  exact `[248068,198,90700]`:0.8B B4 63.90 tok/s versus58.08/vLLM60.85;2B B4
  45.50 tok/s, effectively flat versus~45.8 and still below vLLM60.55. Quick
  non-LTO feedback; full-attention/KV remains the next per-session CUDA lane.
- [x] Aligned cached full-attention now shares one B4 physical launch while each
  lane retains its own candidate K/V buffers and publication root. `blockIdx.z`
  selects the private K/V owner; the existing online-softmax arithmetic is unchanged.
  Public Runtime is exact `[248068,198,90700]`:0.8B B4 peaked65.03 tok/s and2B
  46.30 tok/s versus prior63.90/45.50 and eager-vLLM60.85/60.55. The B4 entry is
  precompiled only for `native_rows=4`; sequential loads keep the original path.
  Remaining2B gap is predominantly dense projection/GEMM scaling, not attention launch count.
- [x] Ordinary BF16 projection GEMMs now certify finite BF16 at the existing
  rounded store and OR the same status flag, removing the separate post-GEMM scan.
  Legacy three-buffer GEMM entries remain unchanged for other callers. Public B4
  Runtime remains exact `[248068,198,90700]`:0.8B65.63 tok/s vs65.03 and2B47.17
  vs46.30; eager-vLLM C4 remains60.85/60.55. Quick non-LTO feedback only.
- [x] B4 lane-strided output projection now reuses each BF16 weight load across
  four independent rows while preserving the exact16-lane/512-term per-row FMA
  and lane-add order. Existing physical GEMM qualification passes every synthetic
  case, including N=4/K=1024 at aligned/+4B views with zero mismatches and clean
  retirement. Public Runtime stays exact `[248068,198,90700]`:0.8B67.59 tok/s
  vs65.63/vLLM60.85;2B49.89 vs47.17/vLLM60.55. Quick non-LTO feedback only;
  the remaining2B gap is in broader dense projection execution, not output readback.
- [x] Aligned B4 full-prompt CUDA prefill now shares only row-independent projection
  GEMMs across four private activation/state lanes; embedding, causal GDN/KV/attention,
  candidate roots and publication remain per-session. Admission owns a bounded ~23.5–
  23.8MiB B4 activation arena and sequential loads allocate none. Public Runtime
  returned the same `"<think>"` as sequential on0.8B/2B; quick non-LTO paired warm
  waves:0.8B144.8vs203.2ms and182.3vs330.9ms across two runs,2B186.2vs230.5ms.
  Four distinct equal-length prompts also matched sequential 16-token continuations.
  This initial checkpoint covered equal full prompts<=64; generalized chunking is below.
- [x] B8 extends the same aligned decode/full-prompt path without inventing B8 state
  math: stateless projections and the exact lane-strided output head widen to8 rows,
  while GDN and full-attention run as two qualified B4 subcohorts over distinct roots.
  Quick non-LTO eight-request47+3 feedback is exact `[248068,198,90700]`:0.8B
  76.77 tok/s vs native4 71.90/vLLM-C8 72.20;2B62.59 vs native4 58.22/vLLM-C8
  73.70. Eight equal-length one-token-different prompts matched native4 lane-for-lane
  through16-token continuations. Activation/admission supports only4|8; wider dynamic
  cohort ceilings remain open.
- [x] Chunked/ragged CUDA prefill keeps intermediate native chunks state-only and emits
  compact greedy readout only on each prompt's terminal tail. Scheduler bands are
  compatibility-aware (same frontier/chunk geometry), so unequal tails regroup/fall back
  without padding or root aliasing; shared wave admission uses each model's context rather
  than a hardcoded512-token frontier. Quick non-LTO153-token x8 A/B stayed exact:0.8B
  777.87ms native8 vs961.53ms sequential (~19.1% faster);2B808.91ms vs1204.58ms
  (~32.8% faster). Ragged140/140/155/155/170/170/185/185-token prompts matched
  sequential lane-for-lane. A separate public-serving0.8B641-token x8 check crossed the
  former512 frontier and stayed exact `[561,10993,12681]`/`" The execution fabric"`;
  observed native width8 and warm3072.58ms sequential vs2903.86ms native8 (~5.5%).
  No new numerical tolerance or permanent fixture.
- [x] Partial offered populations now regroup through a shared B2 subcohort instead
  of failing when an admitted B8/B4 wave leaves two aligned rows. B2 shares only
  stateless projection work; stateful GDN/KV/attention stay on the existing private
  singleton path, while B4/B8 retain their qualified stateful cohorts.2B47+3 quick
  feedback stayed exact `[248068,198,90700]`: Count2 45.27 tok/s vs prior singleton
  36.12, Count6 now completes at49.50 tok/s instead of the previous unsupported
  refusal, and Count7 (4+2+1) completes at51.01 tok/s. Router/cohort unit gates,
  CUDA-serving compile and strict Clippy pass. Native model admission remains4|8.
- [x] Ordered large-K prefill GEMMs now use a64-row MMA tile only when rows>64 and
  K>=2048; smaller-K and all rounded/interleaved/lane reductions keep their existing
  kernels. The K/MMA accumulation order is unchanged; only independent physical rows
  share a larger tile. Controlled same-HEAD/non-LTO B4 47+1 prefill A/B on `ccf441d`:
  0.8B138.10->136.25ms and2B147.42->141.21ms, with the same public output. Geometry,
  CUDA-serving compile and strict GPU Clippy pass. This is a shape capability, not a
  Qwen-model selector; keep further row/K tuning measurement-driven.
- [x] Ordered B8 decode GEMMs with K>=2048 now stage64 K terms per tile instead of32,
  preserving the exact MMA sequence while halving K-stage synchronization. C1/B4,
  smaller-K and non-ordered reductions keep the prior kernels. Controlled same-HEAD
  non-LTO2B B8 A/B on `67586eb` stayed lane-exact:47+3 rose68.31->69.11 tok/s;
  decode-heavier47+16 rose147.62->153.61 tok/s (~4.1%). This is a shared shape
  specialization, not a model selector; retain only while longer decode keeps the gain.
- [x] Rounded-segment B8 GEMMs with64-aligned segment boundaries now reuse the same
  K64 staging while preserving the original MMA order and BF16 rounding points.
  C1/B4, interleaved/ordered/lane reductions and non-64-aligned segments keep their
  previous entries. Existing physical GEMM qualification with an added N=8/K=1024
  selected case passes aligned/+4B views with zero mismatches and clean retirement.
  Same-HEAD non-LTO2B B8 feedback stayed exact:47+3 68.83->69.49 tok/s and47+16
  152.93->157.28 tok/s (~2.8% on the decode-heavier run). Shared shape specialization only.
- [x] Interleaved-pair B8 GEMMs now use the same K64 staging for64-aligned K while
  preserving the two original FP32 accumulation chains and32-term chain-switch points.
  Physical GEMM qualification passes the added N=8/K=64 case at aligned/+4B views
  with zero mismatches and clean retirement. Same-HEAD non-LTO2B47+16 stayed exact
  and improved155.36->158.08 tok/s (~1.75%). C1/B2/B4 and other reductions are unchanged.
- [x] Rejected consolidating the four B4 canonical<->native GDN state-copy launches
  into one grouped typed-transport launch in the ordinary CUDA hot path. The candidate
  stayed lane-exact for the accepted2B B8 47+16 continuation, but a same-HEAD non-LTO
  A/B regressed157.47->151.82 output tok/s (~3.6%). The production integration was
  removed rather than retaining slower complexity. Keep the existing checked per-row
  state transport and the already-qualified grouped transport primitives available for
  other model/layout consumers; do not retry this exact launch consolidation without a
  materially different transport/fusion architecture and measurement.
- [x] Small CUDA populated-context qualification now covers0.8B/2B at1K and2K.
  Each cell starts from the accepted47-token prefill, advances by genuine64-token
  state-only chunks to the exact context frontier, emits finite terminal vocabulary
  logits, reads all42 finite state domains, and shuts down cleanly. Initial greedy
  token remained248068 and terminal greedy token248046 in all four cells. The0.8B
  legacy directory used its already-retained source receipt;2B used the compiled
  inventory-authenticated source directory. Temporary feedback runner was not retained.
- [x] CUDA offered C1/4/8/16/32/64 sweep:0.8B native45.59/45.04/45.46/45.55/44.80/refused,
  vLLM36.94/60.85/72.20/87.79/91.22/95.56 output tok/s;2B native26.24/26.26/26.32/
  refused/refused/refused, vLLM34.88/60.55/73.70/87.17/90.55/88.01. All measured
  outputs and original retirement verified. Native singleton, independent live
  sessions; vLLM eager active8, higher offered populations queue.47–50 input/3output
  tokens, context1024; exploratory direct-API timing. Native throughput does not
  scale; capacity refusals preserve private prior/candidate roots and budgets.
  `qwen-cuda-regeneration-tDW0EJ1I/population-benchmark-summary-r1.json`.
- [x] x2 ROCm runtime acquired under owner approval; fixed12GiB available-host
  threshold waived, swap≤8GiB permitted in fresh scopes. Identity/ownership,
  native budgets/deadlines, driver and host privileges unchanged. Official RDNA
  image digest`19ad8dc5fb3012f2d5810995f73e8bf069302056d2b4aa6fdf2d2ae5fa9a68ab`
  supports gfx1151. Actual packages: Torch2.11.0+rocm7.14.1,
  vLLM0.23.1.dev1+g9ddef7117.d20260901.rocm714, Transformers5.16.1.
  Device-free LLM import failed (GPU-dependent platform detection); retained.
  Metadata probe and all three GPU model executions subsequently ran.
- [x] Native HIP SDK timing scope:0.8B/2B C1..64 and27B C1. All39 waves,
  expected IDs, distinct session roots and original per-model/unit retirement
  verified. Native27B timed requests243.61/239.08s; higher populations not run,
  not memory refusals. No native fused batching.
- [x] ROCm vLLM0.8B/27B offered C1..64: all outputs and original retirement
  verified.2B C1/C8 whole cells match; C4/16/32/64 remain failed comparisons.
- [ ] Resolve ROCm2B batch-dependent output mismatch. Retained
  `qwen-rocm-benchmark-3gteresx/vllm-2b-summary-r1.json` confirms C4 warmup
  wave0/request_index3 differs, despite0/8 timed mismatches. C16/C32/C64 have
  2/32,2/64,6/128 timed mismatches, plus1/1/3 warmup mismatches. Expected
  [95726,26516,220], actual[95726,98019,271]. Warmup failures count; earlier
  “C1/C4/C8 match” shorthand was incorrect. Original summaries/logs unchanged.
  Failed run/source bytes/shutdown/original retirement retained. Cause unlocalized;
  not established as a ROCm kernel bug. Do not repin expected IDs or count these
  cells as accepted matched speedups.

Historical24GiB-total native sweep; output tok/s including prefill, offered clients.
Resident27B75GiB results are separate under signed`d16f096` below:

| C | HIP0.8B | vLLM0.8B | HIP2B | vLLM2B | HIP27B | vLLM27B |
|---|---:|---:|---:|---:|---:|---:|
|1|12.51|57.81|6.85|34.14|0.01243|3.85|
|4|12.28|79.12|6.96|60.23†|not run|7.18|
|8|12.14|125.70|6.86|74.85|not run|9.25|
|16|12.02|140.86|6.83|86.51†|not run|10.74|
|32|12.07|150.25|6.77|92.56†|not run|11.72|
|64|12.01|154.04|6.77|95.35†|not run|12.07|

- † Warmup or timed output mismatch; whole cell is not an accepted matched comparison.
- Same canonical47 input/3 greedy outputs/model, context1024, one warmup and two
  timed waves/cell. Both active limits8; above8 queues. Native physical batch1;
  vLLM actual peak-active/batch width uninstrumented.
- Native raw-text SDK delivery vs vLLM token-ID direct API; native CPU-source
  arithmetic vs stock ROCm BF16 execution. Not bitwise equivalence/full qualification.
- Native27B default24GiB TOTAL (including4GiB context) streams weights; vLLM uses78%
  of96GiB and loads50.22GiB weights. Not resource-matched. Native small scopes
  overlap runtime acquisition; device-free probes overlap part of27B. Exploratory.
- Load timers native/vLLM seconds:0.8B19.54/183.41,2B26.78/187.11,
  27B111.79/209.05. Different boundaries, not matched process cold starts.
- Native summary: `qwen-cuda-regeneration-tDW0EJ1I/hip-native-benchmark-summary-r1.json`.
  Runtime/results/audits: XDG state `qwen-rocm-benchmark-3gteresx`,
  `vllm-{08,2b,27b}-summary-r1.json`. All GPU owners retired; no active campaign.
  Benchmark helpers release-built/executed; subsequent assembled CPU gates passed
  (`composition-final-gate-r2.log`, `budget-final-gate-r1.log`).
  Reader preallocation repair passed and landed in`4f1e827` with its assembled
  final gate and ownership/dependency review.

### Ordinary27B — retained qualification chronology

Source/geometry/storage/streaming and numerical/SDK/lifecycle work below landed
by`4f1e827`; resident-budget qualification is signed`d16f096`. Preserve historical
failures and scope limits, but use the current continuation queue above rather
than old “remaining” milestones to select work. HTTP/context closure is still open.

- [ ] Complete actual27B ordinary HIP/SDK/serve, not the finite two-position route.
  Authenticated `Qwen/Qwen3.8-27B` geometry now drives the same ordinary template:
  64layers,112domains, untied head,851text roles/18shards. Finite CUDA profiles
  remain closed; source geometry grants no execution or numerical acceptance.
- [ ] Qualify bounded weight streaming and source/physical final readers.
  Whole-source BF16 storage is priced before upload; use existing source-page
  transport when resident weights plus workspace/roots/native owners exceed the
  default24GiB total including4GiB context. Explicit residency uses the separately
  admitted75GiB total; host ledger remains16GiB. Original
  source extent is separate from resident allocation; no float kernels changed.
- [ ] Independently audit the original CPU capture before native comparison.
  Existing approved source/image,66 genuine calls/two repeats, full64layers,
  fresh47/48/49/50 and uniform/ragged512→514. Capture completed:66 calls,
  19848 records,22126120608 raw bytes; contemporaneous foreground/group/container/
  EOF retirement recorded before owner receipt. Independent audit now passes
  all6 cases/6533182464 float words/851 roles, full digests, natural feedback and
  repeats. Accepted manifest`f226ad21`; raw`d9690f46`. No HIP parity, serving or
  CUDA-oracle claim follows. Accepted archive staged on x2 with full stream hashes.
  NativeR4 observed child exited0:66calls/6533182464 exact words, full vocabulary,
  natural feedback, independent roots and source/native final readers independently
  audited in `native-observed-reviewed-r4.json`. The supervisor then failed its
  stale descriptor-proxy PID/executable guard, before x2 rebooted. Observer-free
  never started; original terminal/results are absent. Preserve this failed
  campaign and limited observed acceptance; no full-R4/serving acceptance.
  Observer-free-only continuation keeps frozenR3/references; no observed rerun.
  Fresh guard pins boot/PID/starttime/executable without changing device/KFD/busy
  or12GiB checks. R6 staged but superseded unexecuted after
  review: mainPID0 did not ensure descendant retirement; post-start errors could
  skip watcher attachment. Watch`566a9940` cancelled. R7 reviewed/fixed/staged;
  pure/output controls and four stubbed-OS failure injections pass: launch
  timeout, receipt failure, transient status and interrupted wait. Observation
  survives these errors; signals stop only the exact owned invocation; attempted/
  indeterminate scopes are not labelled unrun. R7 manifest`730bcdeb`; watch
  `e89ad3c8` succeeded. Fresh guards/pins/unit/path checks passed with13650759680B
  available. Observer-free R7 launched2026-11-05T12:41:41Z, invocation
  `a31ca21457a24796ad21fb1ad0e3798e`. Only `state-parity`, unchangedR3/reference.
  Original terminal2026-11-05T14:37:14Z proves nonce/invocation, absent owned
  cgroup/KFD[]/busy0 and no supervision errors. Independent
  `native-state-reviewed-r7.json` passes all66calls/3137737728 state-logit words,
  611450880 KV-tail words, independent roots/natural feedback/final owners.
  Job2666 completed; monitor2667 stopped after retrieval/audit. R4 supervisor
  failure and missing original terminal remain; no full-R4 acceptance. Do not
  rerun observed numerics, duplicate launch/restage R7 or revive R6.
- [ ] Run actual-source admission, full-vocabulary/state parity, observer-free
  execution, cancellation/replay/final-reader and ordinary serving checks; then
  the complete assembled gate. Host tests/strict feature checks pass15 tests,
  including retained finite-source boundaries and exact managed manifest.
  Two earlier host failures (stale helper argument/new frontier assertion) are
  retained and corrected; call-free frontier stays0, runtime calls own extents.
  Evidence: XDG state `qwen-ordinary-large-rHMls4mk`; implementation integrated
  by`4f1e827`, with HTTP/context exceptions preserved above.
- [ ] Finish additive BF16/F32 GDN source storage qualification. Canonical host
  and x2 hostR2 exposed the same exact-type refusal: all48 original large-source
  `A_log` and GDN norm tensors are BF16; small-source F32 remains required.
  Shared CPU-source GDN descriptors now accept both independently, retaining
  original dtype/bytes and exact widening through existing upload; CUDA's F32
  contract stays closed. Model-source validation still requires its pinned types.
  Shared descriptor/ordinary-plan/storage regressions and strict checks pass;
  R7 release diagnostics/SDK built, but actual-source final-reader test failed
  on the tokenizer's old single-source pin. R6's test-helper type error remains.
  Tokenizer JSON differs only by seven added-token entries (33 total; max21B):
  source-specific BPE/automata accounting now shares the same authenticated
  NFC/Split/ByteLevel machinery and global owner; original26-token pricing stays.
  R9 assembled validation passes:22 tokenizer plus8 owner-accounting tests per
  exact source, strict HIP/CUDA consumers, release diagnostics/SDK and actual27B
  canonical source/plan/tokenizer/final-reader test. Small-source pricing stays
  unchanged; R8's incorrect special-token test expectation remains as a failure.
  Canonical514/active2 plan streams53786705920B original weights; host14720655712B,
  physical device15439853568B including4GiB context; estimates, not reservations.
  Repaired binariesR2 frozen unqualified. x2 host-capacityR3 passes all28 cells:
  contexts512/1K/2K active1/2/4/8 admitted as estimates;4K–32K refused under the
  original host/device caps. Source recheck and exact-invocation idle retirement
  pass. Frozen binariesR1 remain unqualified and predate both repairs.
  x2 hostR1 failed before source IO on noclobber lock opening; R2 source refusal
  retired its exact invocation/cgroup with KFD empty; both were host-only.
  Independent1K/2K original capture was interrupted by SIGTERM, child137; owned
  foreground/group/container/EOF retirement is recorded. Partial archive remains
  unqualified:104calls/3888records/all104 digests were the target, not a result.
  Host-private successorR2 preserved byte-identical inputs and R1's missing
  original-name absence evidence; it also received SIGTERM/child137, after11/104
  logged calls. Cleanup recorded; both partial archives remain unqualified.
  No active capture or accepted populated-context reference. A successor requires
  fresh identity and reviewed supervision. Shared scopes preserve small-model
  1K–16K and add large1K/2K schedules. R10 passes19 host reader controls/strict
  checks and release assembly; frozenR3 production bytes equal accepted hostR2.
  Capacity/refusal and reader preparation do not qualify actual contexts.
  Sequential C1 HTTP JSON/SSE/disconnect/replay coverage is authored separately
  from concurrency; existing population scopes stay unchanged. R11 stopped on
  a test-only constant-assert lint; runtime guard corrected. R12 strict/release
  assembly passes (1 host test,8 physical scopes ignored); SDK-onlyR4 frozen.
  Only HTTP qualification support differs from frozenR3; production and native
  example bytes are unchanged. No new HTTP physical acceptance.
  R5 lifecycle/SDK/sequential-HTTP campaign and terminal retrieval/audit are
  prepared only:13 fresh scopes. Its original full-R4 gate cannot pass after
  interruption. Successor R8 consumes limited observedR4 + accepted observer-freeR7
  honestly; same nativeR3/SDK-onlyR4 and deadlines/limits. Exact-owned supervisor
  failure controls and attempted/indeterminate/unrun runner controls pass;
  manifest`e8a35f90`, never staged/launched. Review found inter-case descendant,
  signal-handler lock and unit-name signal races. R8 preserved; R9 preparation
  adds case-cgroup checks, lock-free flags and pidfd-bound cancellation. Eleven
  ownership/four supervision controls pass; independent review found no launch
  blocker. Fresh exact guard/pins/paths passed; R9 exclusively staged and launched
  once2026-11-05T16:20:39Z, invocation`26c8c471c5f44b63896052bae63d596e`,
  nonce`9753bbec952d432abcf9562a33ddf8c8`; manifest`05e6cd11`, supervisor2729,
  monitor2730. SDK-complete/disconnect and healthy/session-known/snapshot-known/
  pending-known/readback-known/snapshot-readback-known exited0 with pass markers.
  HTTP-sequential failed101/HTTP504 at `qwen_hip_http.rs:98`, unchanged180s deadline;
  303.74s is cell duration. Stable live snapshot binds exact invocation and the
  HTTP after-boundary runner-only cgroup/KFD[]/busy0. All13 scopes then completed;
  independent retrieval/audit verifies12 semantic scopes plus HTTP101/504,
  authentic original retirement and no supervision errors.52 ordered case events/
  26 runner-only boundaries checked; indeterminate/unrun empty. Disposition:
  `ordinary-semantic-disposition-r9.json`. Original auditor's clean-supervision
  predicate also required semantic process success; supplemental disposition
  separates successful retirement from HTTP failure without changing old bytes.
  Staging log's inherited uppercase R8 labels are stale; filenames/hashes are R9.
  Never manufacture R4 terminal evidence. Retained-service bytes include state/
  scratch; do not confuse them with weight residency.
  Main now implements shared strided projection output/grouped partial windows
  for the concrete HTTP failure. Float operations/reductions/BF16, page/full-K,
  owners/cancellation and budgets are unchanged; all direct/prepared callers
  updated. Eight host mapping/status/scratch/extent controls and strict HIP/CUDA/
  Vulkan consumers and corrected R14 release assembly pass. Preserve R13's
  test-only integer-inference failure. FrozenR5 executed in R10: all15 scopes
  attempted,14 exited0 and HTTP101/504 retained. Logs report66 observed plus66
  observer-free calls, zero mismatches; independent retrieval/audit remains open.
  Original terminal records retirement for invocation`6448edba84244c1b9eb339908a7a2618`,
  nonce`42cfc1bc2aa8407ba8b1ceb07143254c`, owned cgroup absent/KFD[]/busy0;
  output errors include `BrokenPipeError(32, 'Broken pipe')`. No clean-supervision,
  whole-campaign or numerical acceptance before independent disposition.
  R10 is finished, not live; never restage/relaunch or overwrite its artifacts.
  Grouping did not resolve HTTP504; localize measured latency without changing
  deadlines or budgets. Row grouping is not cross-session native batching.
  No measured speedup, populated-context acceptance, current full gate or signature.
  Owner requires shared, additive capabilities—not per-model forks.
- [ ] Complete independent CUDA references on l5; Main owns implementation and
  execution, no new or resumed subagents. Authenticate runtime/provider/source/kernel identities; fresh
  captures do not replace missing legacy executable authority.
  XDG run `qwen-cuda-regeneration-tDW0EJ1I`: exact RTX3060 UUID/SM86, driver595.91.07;
  original retained image passed a CUDA tensor probe. Runtime-addressable memory
  is6088359936B, distinct from physical6GiB; initial equality assertion failed and
  is retained. Upstream vLLM image then failed LLM import on CUDA13 torchcodec.
  Existing CPU-codec image`f57d77f5` (no installation/download) runs both0.8B/2B:
  two actual47-token greedy requests/model produce `[264,264,264]`; source hashes/
  identities before/after and original foreground/group/container/EOF inspected.
  Earlier controller mount exposed guest-writable CID/receipts; review limitation
  retained with supplemental current-name absence, never replacement receipts.
  New ownerR4 mounts only payload/cache, keeps control files host-private and
  independently checks the original container name after cleanup.
  Torch2.13/cu129, vLLM0.31, Transformers5.17, Triton3.7.1; eager/V2/FA2/Triton-GDN.
  Full-vocabulary R1 refused callable serialization; R2's FP32-only assertion
  rejected native BF16. Deferred async exception reporting masked that first
  error as uncleared pending state, not a proven hook bypass/forward overlap.
  R3 snapshots public `compute_logits` native returns and actual input buffers,
  defers CPU drain, returns the same object and exactly restores hooks/methods.
  No head dtype, arithmetic, async or serialization-security changes. Three
  CPU-only controls pass; both models'48calls/16requests independently audited:
  fresh47–50 uniform/alphabet, two repeats, genuine feedback, all248320 vocabulary
  words/call,11919360BF16 words/23838720rawB per model; exact repeats/source pins/
  foreground-group-container-EOF retirement. Raw08`36350f39`,2b`f0b70f7a`;
  `full-vocabulary-{08,2b}-reviewed-r3.json` are discovery-only acceptance.
  Both genuinely unobserved16-request runs independently match public outputs
  and termination, with exact source/options/retirement verified:
  `unobserved-{08,2b}-reviewed-r1.json`. No plain full-vocabulary/state-bit claim.
  Bounded async transport passes host and independent actual-CUDA native-bit/
  stride/mutation controls:40MiB GPU arena +24MiB minimal snapshots,128MiB pinned
  host; nonblocking producer, independent GPU/host retirement. Both expanded
  captures completed and retired; `expanded-{08,2b}-reviewed-r1.json` audit every
  archived byte (5925722312 /10564249800B), all24 original layer/deferred-add
  boundaries and4032 physical state snapshots/model. Captured valid prefixes
  preserve continuity/repeat bits; live views are disjoint. All320 original text
  parameters associate with actual loaded packing/casts, and parameters/buffers
  match before/after. This producer casts18 original F32 GDN norms/model to BF16;
  native source contracts unchanged. Full vocabulary equals R3, public output/
  termination equals unobserved runs. These expanded captures are not full-reference
  passes: they omit actual context slot mapping, owner-boundary mutable metadata
  and classification inputs; consumer successors below extend that coverage.
  Constructor buffers are not independently regenerated.
  KV layout is head×token×(K256|V256), not a leading K/V axis. Fresh Mamba backing
  is invalid history, not assumed zero. Producer/native CUDA use history3;
  CPU/HIP history4 is separate. Slab padding/plain-internal invariance remain
  unclaimed. At this expanded-capture stage CUPTI was inspected only; later
  standalone lineage diagnostics below do not grant model-wide ABI, native CUDA,
  matched performance or later roadmap acceptance. R1 review found incomplete
  chunk-storage attribution, metadata-reference provenance and lifecycle order,
  not capture corruption. R2 strengthens all three; both actual metadata archives
  and9 adversarial controls pass; both full R2 audits passed. Review then found
  inventory subview order was not linked to archive logical offsets. R3 fixes
  that link and rejects a coordinated descriptor/inventory swap: both metadata
  archives and10 adversaries pass. `expanded-{08,2b}-reviewed-r3.json` rehash both
  unchanged native/vocabulary archives, bind792/1334 chunks and explicitly reuse
  pinned R2 byte/layer/source checks. R1/R2 remain immutable; current scoped
  acceptance is R3, not exhaustive effective-state/reference/native closure.
  Read-only R3 review found no remaining concrete order hole; not reexecution.
  Actual-consumer observer captures KV/context mapping, conv/recurrent arguments,
  gather/mask/scatter and classification CPU/GPU inputs;12 real-Torch CPU controls
  pass. 08 consumerR2 completed48calls/37848records/6873374472B with original
  foreground/group/container/name/EOF retirement; vocabulary/public outputs match.
  Independent08 consumer auditR3 now passes all48 calls/8432 events, actual
  classification/counters/mappings, conv mutation and gather/mask/chunk/scatter
  bytes/order. Earlier audits wrongly treated padded GPU schedules as active CPU
  lists and prompt progress as total decode progress. Source-derived corrections
  check full padded allocations/roles, not weakened expectations;19 adversaries
  pass. `consumer-08-reviewed-r3.json`; failed auditors/logs remain immutable.
  2B consumerR2 failed at step28 after9 public requests; partial5987762129B and
  original cleanup retained. R3 removed duplicate chunk-return copying but failed
  at step1. Both original tracebacks identify host-ring admission, not GPU or
  record-budget exhaustion. Neither failure has a complete archive/contemporaneous
  GPU-after receipt; later idle observations never backfill those omissions.
  GPU retirement was already independent of disk work. Corrected transport splits
  GPU snapshot admission from coordinator-owned host admission/copy; host leases
  retire after raw consumption, record credits after bounded manifest writes.
  Drain acknowledges both GPU and exact host retirement. First failure is latched
  before diagnostics; uncertainty fences/quarantines before logging and survives
  query/close races. No producer waits, async/options/cap changes or extra temporaries.
  Fresh2B R4 completed48calls/38102records/11209912072B; all original foreground/
  group/container/name/EOF retirement independently verified, GPU before/after41MiB.
  Full audit verifies source bytes, all24 layers/4032 states/8432 consumer events,
  exact vocabulary/public outputs, masks/conv/scatter and all transport frontiers.
  `consumer-2b-reviewed-r4.json`;288 real scatter links plus7 adversaries pass.
  GPU transport peak11034368B, host90827008B; final used/padding/pending0, no
  failure/unknown/quarantine. Limits remain40MiB GPU+24MiB minimal/128MiB host.
  Transport controls:24 host scopes/20000 allocator operations/600records;
  actual CUDA9 native-bit/stride/mutation records and3 stalled-writer records.
  Latest `tensor-archive-r6.py` / `expanded-consumer-observation-r7.py` fixes a
  subsequent minimal-error publication race with deterministic interleaving and
  callback-fallback controls. Successful GPU/producer paths are byte-identical;
  reuse physical success evidence explicitly, no model recapture or physical
  uncertainty claim for that delta. Original clean R4 capture remains unchanged.
  `consumer-boundaries-reviewed-r5.json` binds both scoped audits and controls.
  Raw K/V candidateR8 and audit bodyR4 now observe actual entry arguments and
  compare logical page mutation through the existing transport. Both actual
  layouts/24 adversaries and original-once/same-return wrapper controls pass;
  no new model execution, backing-padding or complete effective-readset acceptance.
  Native callback recorder/offline typed lineage/checked ELF decoder assembled
  separately as qualification support. Runtime-compatible build replaces the
  failed host-libc probe; all failed bytes/logs stay. R3 repairs rejected nested
  calls incorrectly popping outer frames;15 native host scopes/400 concurrent
  calls/eight wire adversaries pass. No callback CUDA/GIL/file IO/capacity waits.
  Standalone RTX3060 R3 independently verifies13 primitive cases/launches, three
  public-edge-associated executable images,5544 callback records, original
  foreground/group/container/name/EOF retirement and no inter-case synchronization
  in its driver trace. Cubin-derived extents match eight successful parameter
  queries plus three terminal invalid-index responses;34 metadata/eight query
  adversaries pass. `cuda-parameter-lineage-reviewed-r4.json`. Original CUDA-info
  metadata has two explicitly checked link formats; retain R1's decoder refusal.
  Earlier R2 control staging waits and unload-fixture failure remain recorded.
  Independent constructor regeneration now passes for both source configs:
  `constructors-lineage-reviewed-r3.json`,268435488 fresh original CUDA bytes,
  all30 named buffers/model before/after (25 unique,5 aliases/phase), complete
  fresh archive/source/config and original retirement. Accepted consumer raw-byte
  audits explicitly reused; no observed tensor bytes supplied as constructor input.
  Its24 launches/10 images have typed lineage, but **zero captured arguments**:
  producer-version129 was unsupported. These24 ENTRY gaps are permanent for R3.
  Earlier constructor R1 missing-config-context and R2 enum-check failures stay;
  both precede kernels and lack contemporaneous GPU-after receipts.
  Native argument recorder `cuda-lineage-r6/`, approved-runtime buildR5, now
  supports witnessed123/128/129 metadata, live typed handles, declared host extents,
  packed extra/Ex attributes and explicit unmarked threads.32 native scopes,
  400 concurrent calls,14 actual schemas,166 malformed metadata cases and8 wire
  adversaries pass on CPU; source/binary pins and original retirement inspected.
  Binary`22a8dcaa2bc33f6e96a5bb141c4f60a0631b3ff96999a78d682f9730b8309f5f`.
  Initial native parser refusal remains. No GPU/model run of the new recorder,
  worker bootstrap/event attribution integration or actual GPU argument acceptance.
  Candidate `argument-wire-r1.py` still points to R5 readers; prepare/test an
  explicit successor, not an in-place rewrite of executed capture/audit inputs.
  Paused here by owner; no next campaign launched. Resume combined model-wide
  argument/image observation, effective-state read/write and recurrent-axis proof,
  then native CUDA/serving. Native CUDA history3 already exists; do not impose
  CPU/HIP history4. No aggregate-field/model-ABI/full-reference/native acceptance.
  No retained package found in the inspected readable local/remote scopes;
  five local container housekeeping paths denied.
  Exact2B cubin and original FLA/causal-conv/FlashAttention qualification remain
  unavailable. No CPU substitution or source repin; accepted new discovery is
  not full CUDA reference or legacy-authority acceptance.

### Next end-to-end milestone

- [ ] Extend populated contexts beyond accepted0.8B/2B HIP scopes to remaining
  CUDA/general27B routes; retain full-logit/state truth, cancellation and physical
  ownership at actual extents.32K refusal remains a capacity result, not inference.
- [ ] Extend/qualify serving concurrency through C1..C64 and general27B/CUDA;
  preserve missing-source/kernel/runtime refusals. Offered C64/active≤8 singleton
  execution is only a baseline, not native-batch acceptance.
- [ ] Measure matched vLLM workloads as each implemented path passes its relevant
  correctness/capacity checks; do not wait for the whole roadmap or batching.
  Match workload/settings/concurrency, report losses and limits. Unqualified
  experiments stay exploratory; no superiority claim without matched measurements.

### Deferred until Fabric and model/context correctness are complete

- [ ] Owner requirement: independently configure queue depth, active admission,
  batch members and per-step token budget; enable genuine continuous batching,
  including all eligible admitted work when it fits. Remove the universal
  eight-session/singleton policy ceiling, not physical/source/kernel safeguards.
  Reuse shared cohorts, original flights/roots and the arbiter. Cross-session
  kernels/state mappings and partial cancellation must work before promotion;
  no metadata-only batching or silent relabeling of serial work. This is queued,
  not part of the current populated-context correctness implementation.

### Current physical checkpoint — 31 Oct

- [x] Exact `.8B` native/reference inputs recovered offline; fresh phase-1
  cached47→50 reference capture and all 1,008 source-window retirements pass
  after selecting the existing explicit 8 GiB reference host profile.
- [x] Signed `2983bb7` replaces ordinary cuBLAS dependency with owned execution;
  exact `.8B` cached47→50, physical lifecycle/capacity and source gates pass.
  The historical private-provider refusal remains intact.
  [Current numerical evidence](../numerics-and-context/TODO.md).
- [ ] Owner30Oct explicitly resumes Qwen ROCm (HIP) on `pvl-x2` independently
  of wider CUDA reference gaps; approved2B/9B/27B model downloads into canonical
  XDG cache are allowed if needed. No subagents, source substitutions, SDK
  downloads or device-privilege changes. Current27B finite source/natural reports
  pass exact full64/fullV/all112-state with supervised exit/retirement in signed
  `825ade8`. Ordinary production/serving and general27B contexts remain open;
  bounded small-model HIP results follow below.
- [ ] Implement ordinary HIP through shared operators/physical owners, not a
  backend flag on the CUDA plan. Declare actual numerical/storage contracts;
  generalize real model geometry beyond24 layers/42 domains for27B. Preserve
  source-priced admission, independent roots, flights and physical retirement.

### Ordinary HIP continuation — 31 Oct

- [ ] Generalize actual geometry/backend execution for27B. Finite27B still
  pins2 positions/112 domains; it is not the ordinary small-model source path.
The former0.8B/2B populated-context continuation is complete in`8ffb511` below;
its1K–16K native/SDK/HTTP acceptance is not reopened by the27B extension.
[Numerics](../numerics-and-context/TODO.md) retains independent evidence.

### Ordinary Qwen execution checkpoint — 27 Oct

- [x] `ad82feb`/`f7ff4f9` add one call-free source-template plan for authenticated
  Qwen3.5 `.8B` and `2B`, retaining the exact source owner/workspace through plan
  and invocation lifetimes. The shared 24-layer schedule has42 state domains,
  chunks<=64 and context<=32768; old captured resident/batch paths refuse it.
- [x] `fe228ab`/`52bfb51` add the complete parameterized GDN prefill/cached
  schedule with explicit transport/state operations and no public second launch
  authority. Configuration/source/ABI remain explicit; source/host checks do not
  establish whole-model numerics.
- [x] Independently complete the interrupted ordinary-primitive review. The
  source-derived primitive series is integrated as
  `ea66b3a`/`68393e5`/`6b4a2ec`/`441fa64`, with RMS overflow/status, scalar ABI,
  alias, theta/KV-layout and CUDA compile/reflection gaps closed. It remains
  unselected and has no backend or launch authority.
- [x] The ordinary non-GDN projection/attention lowerer is integrated through
  the existing physical owner with authenticated tensors, queue/flight,
  status/completion/quarantine and `StateRoot`; the 28-Oct ordinary checkpoint
  below supersedes this earlier open item. No generated packet table,
  prompt/call/node identity or qualification fallback authorizes execution.
- [ ] Qualify `.8B` then `2B` full model, cached continuation, faults, shutdown
  and exact resource fit before enabling an ordinary serving route. Long-context,
  CPU, HIP and Vulkan cells remain independent gates.

### Current owner direction — 26 Oct

- [~] Every existing/future registry target follows the shared
  [correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md): declared
  per-op accuracy, pinned independent task/model quality, separate determinism
  and exact source/state/physical ownership. Registry metadata is no execution
  or model-access authorization; unsupported adapters remain unqualified.
- [ ] Implement/qualify missing full-model/context/backend cells, beginning
  .8B47–50 then genuine long context on CPU/CUDA/ROCm/Vulkan. ROOT coordinates
  exclusively GPT-6.1 Sol xhigh lanes. Older Pi/Astra assignments and performance
  sequence below are superseded. The later30Oct mission resumes independent ROCm
  and correctness-gated timing; older blanket deferrals below are historical.

Historical24Oct: CLI and shared model units are integrated; Pi implements/executes,
Sol coordinates/reviews/merges, Astra designs/reviews architecture and math.
Local source21 shared state-certificate preparation passes the complete1793 CPU
gate, combined strict, five readiness contracts and optimized release build.
Current physical23/full2Bmath and both .8B/2B API/HTTP pass ROOT review;
current .8B/2B C1/C4/C8 singlepass rate AND wall wins are accepted. These
finite scopes do not grant fullmatrix/generalparity. Source20 x2's complete
shared HIP unit passes below. Independent Qwen natural feedback and learned
Gemma PLE hit/transfer coverage remain open; next whole-unit design proceeds in
parallel with local .8B→2B benchmark work. All78 signed commits are complete.
Latest user direction resumes all tracks now: separate Pi Qwen27B/Gemma4/
Vulkan/HIP/queue lanes alongside external C16 correction. C32/C64 follow shared
queue acceptance; physical jobs and quiet timings retain coordinated ownership.

Latest owner merge direction: complete the current Qwen27B/Gemma4/HIP/Vulkan/
queue units to a tested, mergeable checkpoint, assemble and reconcile shared
source, then integrate into main and retire completed checkouts. Keep original
reference roots while current checks use them. Broken source or a draft
controller is not a merge checkpoint; preserve pending physical scopes honestly.


### H/G/V physical checkpoint — 25 Oct pass 3

- [~] Three passes executed on pvl-x2.
  Root causes of passes 1-2 fixed and verified on hardware
  (`06b89da` E0308, `c217c35` V parents, `e6ab074` census,
  `3941a3c` Gemma envelope). Pass 3: builds/admins/binds all PASS;
  h-known-qwen-reload + h-qwen-final-reader-unknown PASS (full
  semantic closures); three real failures preserved
  (h-gemma-feedback-unknown: session-state identity/policy/fence
  assert + `scope closes only between completed proposals`;
  g-resident: parity row-upload fixture contract vs legitimate PLE
  row dedup; v-resident: adapter validator requires a false-permit
  field the producer never emits — physical case itself passed);
  19 cases stopped as dependents. No product defect established
  (qualification-gated paths).

### Current shared source20 hardware checkpoint — 24 Oct

- [~] Qwen natural2-position independent original comparison and Gemma learned
  PLE row-hit/H2D/full-state scope, retaining existing quotas/owners. ROOT reviewed
  Astra's complete next-unit design; two isolated Pi implementation/controller
  preparation lanes active, no GPU/reference execution before review. Qwen old
  teacher capture requires independently verified original fullV/prior/source
  join; otherwise fresh natural reference. Gemma reuses its genuine natural
  matrix and adds an actual-hit full35/fullV comparison plus final-owner scopes.
  No new public
  serving, source21 x2 artifact, general-fit or speed grant from source20 fixture.
- [ ] Gemma PLE twofixture source draft and prepared controller need corrective
  review before merge/physical work. ROOT finds diagnosticD2H exactly1B gate
  instead of positive and regressed dataFD/bounds. Original valid source20
  math/lifetime results unchanged; new all-hit/final-reader scope unexecuted.


### Qwen27B and Gemma4 optimization audit — 24 Oct

- [ ] Close isolated source113 as one checked model unit, then actual x2 source/
  state/fault/lifetime qualification: Qwen full64/all112, Gemma resident20/all35/
  fullV/all15KV and observer-free natural feedback. Grouped isolated CPU/strict
  checks PASS:5strict configurations,907CPU library tests/2comparators,46explicitly
  ignored cases. Exact source113/P1/P2 and original400v4 base remain unchanged;
  contextual410 continuous/compiler reconciliation is reviewed and affected
  strict/166boundary/7compiler-owner checks pass with all jobs closed;21physical
  cases explicitly ignored. Later RMS3/source6 excluded. Gemma's finite actual
  Resident20 and Qwen full64/armed-producer fault results follow below; Qwen's
  final Pending/Snapshot reader fault coverage is now closed on frozen469 below.
  Keep current C8 assembly separate.
- [ ] Reconcile frozen models113/P1 hooks and Gemma final-feedback fix with current
  shared CUDA readiness source, preserve arithmetic/backend boundaries, perform
  affected assembled checks and final complete gate before model merge acceptance.
- [ ] Qwen27: qualify natural and temporal decode/prefill, residency/admitted
  prefetch, packed projections, eligible optimized GEMM/attention/recurrent GDN,
  hybrid prefix checkpoints and serving through shared engine primitives.
  Current native route requires streamed weights and finite teacherfrontiers0/1;
  shared64-lane CPU-order BF16 dot is not TensorCore/BLAS. Preserve source/cast/
  reduction profiles rather than inheriting small-model CUDA permits.
- [ ] Gemma E2B final-reader ownership: source466 review finds Session/Snapshot
  drop `loaded` before final committed feedback's independent12-byte Output
  lease. Native/modules/managed-source guard can retire before a true final
  feedback release refusal. Pending already has safe ordering. Isolated Sol
  fix+fixture unit authorized: place `loaded` after inner+feedback; Session/
  Snapshot × healthy/exact final Output-lease refusal, Weak-only diagnostics and
  original actual service/native/source/quarantine owners. Preserve frozen466,
  completed Qwen P1 and readiness sources. Frozen469 ten-path fix+fixture passes
  grouped6control/4probe/all-target strict checks; independent Sol review and ROOT
  complete source/control/cost/archive review pass. ROOT authorized one bounded
  fresh admin/offlinebuild/four-case unit. Admin33537/offline GPU-free build73520
  CLOSED0; first Session-known28952 FAILED_CLOSED at semantic verifier's exact
  `device_with_six_roots_bytes` formula after native child exited0. Actual groups/
  one-reap/fullEOF/source35FD/cache-parent closure healthy. ROOT and independent
  Sol identify verifier-only omission: greedy12336B+six256B alignment guards
  missing from both exact expected totals; production sums/caps are correct.
  Corrected local verifier6controls PASS; new saved Session-known revalidation
  PASS without GPU replay, original FAILED receipts unchanged. ROOT reviewed
  full225line local collector delta/17pins and authorized remaining3 serial
  native cases, exact-known-error-only/source35/cache/binary/originalowner
  closure required. All4actual scopes now close with corrected local acceptance;
  ROOT independently verifies every report/source/cache/original+reader owner.
  Known:184→0modules/sourcecloseOK/model0/context4GiBdevice+2GiBhost. Fault:
  actual12Bfinalreader refusal precedes native retirement,184modules/617handles
  retained/full20GiBfenced/sourceclose refused. Every full35/30planes/fullV/
  one natural commit; original failed remote collectors and SSH1 remain explicit.
  No Session replay/Rust rebuild/serving or performance grant. Existing source466
  cache reused with only actual publication/activity locks writable; no second
  10GB import. Current11+118path isolated combined8package/alltarget strict and
  189CPU PASS/33explicitphysicalignored; original source/owner/EOF closures pass,
  first controller-only compilerfile cap failure preserved. Current12 differs
  only by test-fixture rustfmt outside118. ROOT now adopted models118 plus the
  source-bound2B cachedB4 unit together:173selected/172effective/46new paths,
  fresh mtimes, original README/lock and all unrelated source preserved. Current13
  has1629full files/492selected; combined strict/20sharedCPU/5ABI-source controls/
  authentic2B CPU inventory/reopen pass. Original excluded-vendor Cargo refusals
  remain failed, not executed vendor-test evidence. Optimized13 CLOSED0 and all
  original19physical scopes/healthy2Btrajectory PASS; original2B same-prior30calls/
  8112sections/fullstate/fullV exact. New2B fault global-vs-flight count mismatch
  is diagnosed; fixture-only15 correction builds, preserving every graph/source/
  fence guard. Current assembled phase and final
  assembled gate still required. Prior.8B measured checkpoint remains source11/12.
  Combined assembly results.
  Four-case report.
  Independent source evidence.
- [ ] Gemma E2B/E4B: dependency-based shared-tail fast prefill, PLE row reuse,
  optimized packed projections and eligible attention, physical SWA reclamation.
  E2B15owners/20shared tail exists; E4B needs24/18 and a new functional route.
  Optional vLLM fast prefill is off by default, not a universal baseline feature.
- [ ] Add functional Gemma E4B and12B routes;12B uses Unified geometry/MM mapping,
  no PLE/shared tail. Extend source/profile contracts before learned qualification.
- [ ] Gemma26B-A4B: learned router/grouped experts/dense+expert composition, then
  admitted expert sharing/hotsets/EP. Gemma31B is dense; qualify its learned dense
  path separately. Reuse the existing single raw K projection for configured
  K=V layers, preserving distinct K/V transforms and state. Select the exact
  optimized-router tie/reduction/scale profile; full-softmax fallback is not an
  identical floating-point schedule. Geometry/analytic contracts alone do not
  implement either learned route.
- [ ] Both families: shared continuous serving/temporal work, paged KV/COW,
  full-domain compatible prefix intersection and graph/capture ownership after
  actual source qualification. Shared target verifier/all-prefix rollback before
  Qwen trained MTP or Gemma assistant/KV sharing/sparse draft vocabulary.
- [ ] Later explicit codec/distributed/MM/LoRA profiles: do not substitute lossy
  FP8/FP4/KV codecs for original BF16 or assume source/hardware eligibility.
  CUDA-only fusion and CDNA/gfx12 AITER are not unconditional gfx1151 gaps;
  current ROCm Triton speculative GDN needs actual eligibility checks. Generic
  MXFP4 registry falls back for dense Linear; DBO rejects relevant hybrid/MM routes.

### Current model/backend checkpoint — 23 Oct

- Owner clarification: Qwen27B/Gemma4 must reuse the established .8B/2B code
  wherever applicable. Lift common allocation/residency/transfer/native ownership,
  model operations and family behavior to their existing shared layers; compose
  those primitives in each model. Before merging, inspect callers and a concrete
  reuse map, remove parallel implementations, and justify remaining specialization
  by actual source layout, numerical contract or necessary optimized kernel path.
  Distinct arithmetic/backend eligibility must not be merged by superficial
  operator similarity. Sol HIGH implements; ROOT reviews/reconciles whole units.
  Actual audit found duplicated family orchestration in27B. Sol extracts one
  substantive HybridDecoder composition consumed by BOTH small-model and27B
  recorders, with shared role groups/bindings/state schema. Packed-vllm delayed
  residual versus separate Torch/BF16 arithmetic remain explicit adapters.
  Astra initial review
  accepts the design; settled source/caller reuse and actual checks remain open.
- Shared physical successor now integrated MAIN/ISO345v3. Required local release
  and both small models' original full source/state/fault/API matrices PASS;
  x2 native3/portable6/rawtransport12 plus corrected shared4 HIP/Vulkan healthy/
  unknown PASS in finite scopes. Healthy source retirement and unknown source
  retention were rerun after fixture-only Oracle lifetime correction; production
  ownership unchanged. Original interrupted-controller/missing local13 proof
  remains explicit; fresh13 replay passed. No new learned large-model/performance
  grant. Qwen27B reachable native+managed facade compiles with Gemma; complete
  single-source qualifier/strict validation in progress. Next Gemma residency
  and multiple-native-set work follows Astra architecture-only review and Sol
  HIGH implementation; final assembled gate remains pending.
- Gemma optimized residency unit active in isolated Sol HIGH assembly, following
  Astra architecture-only design:
  reuse shared authenticated matrices/row transport, cold-select repeated full
  projections and tied embedding/head, widen output-row launches without changing
  full-K arithmetic, normal observer=None before D2H construction, original
  private packet/Ready/state publication. Complete source-shaped cold host/device
  inventory precedes fixed profile selection. PLE selected-row transport remains
  explicit. Implementation/actual numerical/serving/performance acceptance open.
- Shared native/context+Qwen opt-in Conv4 integrated301selected paths. Both small
  Qwen models' required release full physical/state/fault/API scopes and explicit
  streamed2B faults PASS; full isolated gate14 LAST PASS1747Rust/0failed/allchecks.
  Original x2 native3+HIP/Vulkan transport12 also pass their finite scopes only.
  Conv4 transport is not learned27B numerical closure.
- Astra HIGH architecture/design only; Sol HIGH owns implementation. Gemma44v5
  now guarded/fresh-mtime integrated MAIN/ISO343selected paths. Unsafe lowlevel
  source-retention contract has one validated safe facade; source-neutral finite
  HIP policy, full seven-cell/state/lifetime qualifier and shared round probe
  use the original engine/source/flight/publication owners. All parents/donors/
  modes/rootREADME preserved. Original CPU128MiB profile unchanged.
  Actual frozen MAIN314+44 executable on x2 PASS:7source-tokenized prompt cases,
  55completed comparisons/52commits/150rows/all35block boundaries/full262144V/
  all15KV final+prior pairs,46,858,240derived compared words/zero mismatches;
  real private feedback/decoded output, independent sessions/pressure/cancel/
  Ready discard/refusal/restore/reset/source close PASS. Actual shared-round16
  signed-subnormal/tie/cancellation cases and186native modules pass. Fresh
  unknown process:1completed producer/0roots/full16GiB fenced/source-close
  refused; original owners retain until process retirement. Healthy model close
  retains contextdevice4GiB/host2GiB until process exit. Both original local+
  remote source35/group/one-reap/fullEOF closures pass; sampled shared memory is
  not an isolated peak/bound. Three fresh-process ELF hashes differ explicitly.
  Complete actual checkpoint.
  Single1MiB synchronous source transport is a correctness baseline; no serving/
  performance/general HIP permit. Binder316 and the new shared physical-owner
  split were not executed by that proof. Next assembled gate/affected physical
  checks follow the coherent successor; gate16 accepted preceding316 only.
- Qwen27B actual original MKL dispatch identified SSE EXHAynn. Sol's shared Rust
  explicit normal-domain128split-table/non-FMA profile matches the authenticated
  connected H48 cached producer, including both exp calls and all786432state
  words bitexact; legacy2/48decay +5539state negative control preserved. Five
  reviewed paths integrated MAIN/ISO302selected paths with fresh mtimes; CPU/Qwen
  strict and existing operator tests pass. No learned
  model/native/serving/performance grant from this fixture closure. Subsequent
  actual full64 replay3 now PASS: two independent sessions each one INITIAL
  token ID11 then one cached token ID353, all1040original records/240896000float words bitexact, all112
  state domains/fullV/ALL4Conv frames,12position words. Source/archive/owned
  closures, precancel/actualpressure/output-retention/finalhost0/unfenced pass.
  Generic profile/wholemodel/native/serving/performance flags remain false;
  natural feedback, wider contexts and GPU qualification still open. Four
  qualification-example corrections now integrated MAIN/ISO with fresh mtimes,
  preserving exact immutable replay evidence. Full assembled gate15 LAST PASS:
  1752Rust/0failed/89ignored/269suites/allscriptchecks,314sourcepaths/rootREADME
  unchanged.
  Prepared-recipe successor also passes both small models' full CUDA/state/
  fault/API scopes. Latest prepared-binding release1 quiet .8B C1+78.62%/
  C4+26.67%/C8−1.09% remains singlepass; final gate16 PASS1752Rust/0failed/
  89ignored/269suites/allchecks accepted316before Gemma integration.

### Parallel model work — contained integration, 22 Oct

- [ ] Continue Qwen27B/Gemma source/runtime integration independently; merge
  contained completed units promptly. Use pvl-x2 Radeon8060S/gfx1151 ROCm/HIP
  for their GPU qualification/benchmarks; preserve exact HF source identities,
  source-derived complete allocations and full logits/state/cancellation gates.
  Existing remote Ollama GGUF inputs are not approved-HF equivalents. Current
  Vulkan/CUDA parity work is a separate implementation/qualification obligation.
- [ ] Gemma follow-on: qualify tokenizer/ordinary learned route and broader
  prompt/cached geometry plus learned cancellation/state lifecycle through the
  same engine; selected diagnostic exactness alone grants no public route.
- [ ] Independent follow-on units continue while short-HF validation runs:
  Qwen authentic H48/128 GDN producer and full64/112-domain schedule, plus Gemma
  family source-bound CPU completion. Preserve actual source geometry and exact
  numerical limits; unavailable learned arithmetic must not use legacy fallback.
- [ ] Gemma trained Rust execution requires exact learned BF16 arithmetic and
  independent source/state comparison. Payload/tokenizer/original CPU oracle
  now exist; native/serving/performance acceptance remains separate.


### Source-specific optimization backlog — 21 Oct, deferred behind current order

Source details and limits: [independent comparison](../fabric-vnext/PRIOR_ART_AND_REVALIDATION.md).
These are future operators/strategies in the same engine, not payload approval
or model support. Qwen parity/short benchmark/long→dense27B→Gemma remains binding.

- [ ] Dense27B: bind full trained source/state through bounded row/layer windows;
  amortize immutable reads across compatible prompt/request rows, account
  activation spill and genuine upload readiness. MTP metadata/tensors alone
  do not qualify target or drafter runtime; expert hotsets do not sparsify it.
- [ ] Gemma4 variants: source-real PLE, normalization/logit caps, SWA/global
  attention, configured K=V/cross-layer sharing, head256/512 and26B-A4B routing.
  Check graph padding cannot route experts or advance roots. Matching trained
  assistants may share target KV/embedding only under actual architecture maps;
  first qualify target execution and complete branch/precision ownership.
- [ ] Later sparse Qwen: stable finite top-k/tie/block ordering, source-selected
  expert grouping, hyperconnections and deterministic learned PLE lookup lane;
  runtime n-gram proposals are a separate stage. Source-dependent sparse masks
  and reduction/cast profiles cannot be grafted onto ordinary dense attention.
- [ ] Later DeepSeek: exact CSA/HCA/MLA/indexer/compressor/mHC/YOCO/Engram and
  checkpoint-matched DSpark/Markov/confidence contracts; typed dependency/liveness
  proof may eliminate dead prompt work only after every future state read is
  covered. Truncated bounded replay and perturbed sampling are explicit changed
  profiles, not exact defaults. Existing payload acquisition hold persists.
- [ ] Later GLM: distinguish KDA from GDN, MLA/DSA index/open-tail state, mHC and
  own trained MTP/taps. CPU/GPU placement, cached-expert group order and FP16
  prefill must have declared arithmetic; geometry or source names do not prove
  interchangeability. Existing payload/device holds persist.

- [ ] **Independent trained BF16/FLA checkpoint gate:** original
  Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 acquired
  through the existing bounded Rust cache pipeline; independent31-file/
  18-shard full hashes and1199-tensor header/index checks passed.
  Whole-model27B execution remains unqualified; bounded trained operator
  qualifications below do not establish full logits or112 StateRoot domains.
  Genuine original layer0 GDN reference now validates initial chunk64 and true
  cached continuation for teachers11/353,48 value heads and original BF16
  parameters. Plain/observed complete outputs, convolution/recurrent state,
  raw source readsets, full before/after source hashes and original owned
  retirement match. The serializer's V2 stride failure remains preserved;
  V3 raw-bit transport passes12 real PyTorch layouts. This is an independent
  public target. An isolated native composition now passes the authentic H48
  cached slice:804960 input and798720 output words, including the complete
  recurrent state, match independently before/after callers drop. The distinct
  BF16-gamma gated RMS slice also passes61632 intermediate,12288 final and
  24576 retained input words. Source hashes, original ownership, cancellation,
  real pressure and final reservations0 pass; strict/release builds and237
  focused tests pass. These operator changes remain isolated from MAIN;
  initial chunk64, native exp, complete GDN,112-domain state and full inference
  remain unqualified. A separate genuine public reference passes32 pure
  operator cases over16 roles,881856 raw BF16 input/output words, including
  complete248320-row head and17408-column down projection. The isolated
  native wide-role replay now matches all881856 words exactly; real pressure,
  pre-cancel, retained head/source after caller drop and final reservations0
  pass. Strict/release builds and17 source/relocation fixtures pass. External
  probes for other roles do not establish trained producers or full-model
  parity; these changes also remain isolated from MAIN.
  Six actual bounded native-provider startup/lifetime cases pass with the
  provider unloaded. Physical provider allocation/math admission remains open.
  Original
  Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc is now acquired
  by the same bounded Rust pipeline; independent official Git/LFS full12-file
  hashes,632 headers/320 text shapes,596BF16+36F32 tensors pass. Its4.548GB
  shard remains above the default4GiB complete-source guard. Genuine pinned
  official CPU two-profile reference capture now passes archive and owned
  cleanup validation: tokens11/353, four full248320 logits,168 canonical
  domains and60 actual intermediates. Official BF16 loading has617 BF16
  parameters; restoring only36 original F32 tensors gives581 BF16+36 F32.
  The profiles differ in47/49 raw tensors per step despite identical argmax;
  neither establishes Rust2B HF numerical qualification. First capture's
  cleanup-name bug is preserved; the repaired fresh run proves actual nonce
  container absence and process reap. Original Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 pinned
  Transformers5.17/Torch2.13 CPU captures passed archive validation:
  tokens11,353, full248320 logits/all42 domains, actual layer0 stages.
  Keep official473-BF16 loader and source437-BF16+36-F32 profiles distinct;
  first-token argmax321 vs198. Explicit new CPU64 diagnostic vs source-mixed
  reference remains UNQUALIFIED: embeddings/direct layer0 norms and both
  QKV projections exact; layer0 convolution state exact,41/42 domains still
  differ. Full248320 logits differ214245/214285 words, maxabs .22119140625/
  .1875. First token exposes conv/SiLU signedzero; second exposes GDN state.
  Complete pinned convolution, chunk64/cached recurrent, coefficients and
  gated RMS arithmetic, then original GPU/FLA/state/streaming gates before enabling
  trained HF execution or27B serving. User authorizes required runtime/model
  acquisition/testing; dated source-free slice restrictions below are history.
- [ ] **Next trained reference gate:** qualify exact original HF BF16
  rounding/FLA order and independent source-trained logits/StateRoot;
  then bind the original profile to the existing per-op
  MathematicalBackend/SessionAuthority, with full physical
  device/SSD resource admission. Earlier source-free slices establish no
  full27B logits,112-domain state, native provider admission or GPU support;
  current owner authorization covers the required acquisition and testing.

- [ ] **Following ME-P0 real trained integration:** fuse the
  accepted analytical original source operators into the ONE
  existing forward_token/MathematicalBackend and
  SessionAuthority execution schedule, qualify original HF BF16
  intermediate casts and FLA GDN accuracy, source-trained exact
  full logits and StateRoot, tokenizer, physical mutable-state
  memory accounting, cancellation and device completion. No
  SSD/physical inference claim follows from the diagnostic bridge.

- [ ] **ME-P0 next after diagnostics:** wire the analytically verified
  Qwen original operators into the ONE existing forward-plan and
  TaskDriver/SessionAuthority (no second token loop or state issuer),
  with exact source BF16 intermediate cast and original FLA GDN kernels,
  real trusted tokenizer and full 248320 logits/42-or-more StateRoot
  domains against the pinned checkpoint. Only then begin
  SSD-resident streaming and measured Qwen3.8-27B physical throughput.

- [ ] **ME-P0 next numerical stage:** source-exact original Qwen
  full attention, GDN/MLP, grouped Q/K→V replication, A_log/gate
  transformation and BF16 intermediate numerical schedule using the
  shared original TaskDriver/StateRoot. Qualify tiny independent
  analytic operator/state gates before real approved trained Qwen3.8
  27B original full-vocabulary reference. Measured original Qwen3.5
  2B CPU/GPU end-to-end performance also remains open.

- [ ] **ME-P0 next:** bind source-approved original Qwen3.8 27B
  tensor/index/config and tokenizer to the existing mathematical engine
  with authentic head/axis geometry, strict trained CPU full logits/
  StateRoot, then source-verifiable immutable SSD extents with stable
  leased memory and measured decode/prefill. First finish measured
  Qwen3.5 2B CPU/approved GPU end-to-end performance and
  cancellation/physical readiness. Preserve no model/SDK downloads
  and held device launch policies without explicit authorization.

- [ ] **ACTIVE NEXT: ME-P0 Qwen3.5 2B then Qwen3.8 27B.**
  Begin with original qualified Qwen3.5 .8B/2B CPU and resident GPU
  source/state oracles, inspect release source ownership and exact
  model-2B profiling on the permitted CPU/GPU devices. Keep tests
  bounded to existing 8K context now; 16K/32K extensions remain
  separate numerical gates. Qualify a stable single-model fast path,
  original tokenizer/math/KV/recurrent/StateRoot and actual device
  completion first. Then implement checked immutable source shard leases
  above the present 4GiB executable-source gate and Qwen3.8 27B
  complete trained CPU logits, including SSD streaming and quant-aware
  physical placement. Do not fetch weights or launch held Renoir/gfx90c
  hardware without the corresponding permission and source integrity.
- [ ] **NEXT after Qwen: ME-P1 Gemma4 E2B→26B-A4B→31B**, independently
  source-bind tensors/reference semantics, PLE/MoE/dense quantization
  and trained CPU→GPU correctness before measured throughput
  optimization. **ME-P2 Qwen3.8 Flash-Next** follows with separately
  verified learned n-gram embedding/4B MTP heads, QSA/GDN/MoE and
  explicit Q2/~3.5bpw/Q4 source quant choices (NOT guessed files),
  speculative n-gram as a distinct runtime optimization. Other OSS,
  ASR/TTS/image/video/decision execution and multi-model router remain
  on their own deferred source-real gates.
- [ ] Preserve direct one-model-per-process execution and one
  shared-instance-many-pipelines as **one Rust fabric**; the accepted
  YAML DAG/compiler and extension schemas add no inference backend.
  Latest21Oct owner request authorizes parallel recursive pipeline/profile
  design after independent engine research. It must not delay ME-P0/P1 or
  imply new trained context/media/router execution.

- [ ] Enforce new owner priority: Qwen3.5 .8B regression, Qwen3.5
  2B highest-perf trained CPU/GPU, Qwen3.8 27B with SSD
  streaming/owned shard tensors, Gemma4 E2B, 26B-A4B, dense 31B,
  then Qwen3.8 Flash-Next QSA/GDN/MoE/gated residual/MTP
  and 51B learned n-gram embeddings. Every additional DeepSeek,
  GLM/Muse/Mistral/Llama family has a typed NOT IMPLEMENTED
  source contract until distinct trained CPU and GPU gates pass.
  No subfamily is generalized from metadata-only source support.
- [ ] Treat single-model local GGUF execution and multi-model
  pipeline-qualified execution as two frontends of ONE TaskDriver/
  SourceIdentity/StateRoot fabric. A reusable model instance
  serves multiple routes without another model process or implicit
  source promotion; named replicas/adapters require separate
  physical memory admission.

- [ ] After the accepted source-acquisition v1 cutover, use the existing
  pinned HF index-first path to inspect explicit original Qwen3.8-27B and
  Gemma4 26B-A4B index/config/tokenizer/chat/processor identities
  (see [acquisition v1 handoff](../model-acquisition/README.md)).
  Select an authorized exact revision and admitted local host. Do not treat
  synthetic >51GB transfer-plan tests, public model pages or cached bytes
  as authorization, execution-ready source, or trained CPU parity.
  2026-10-18 tokenless live metadata-only HF source observation resolved
  Qwen3.8-27B to `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  (18 shards, 55,586,113,293 selected bytes) and Gemma4-26B-A4B-it
  to `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
  (2 shards, 51,644,338,194 selected bytes); Gemma4 E2B-it commit
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
  (1 unsharded safetensors, 10,278,848,001 selected bytes).
  No trained weights downloaded, locally owned, header-audited or
  model-execution-qualified yet.

- [ ] Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
- [ ] Qualify real large (>4GiB) original trained shard payload on an
  authorized capable host using the already verified owned-shard
  implementation from 93a5622; independently measure resident
  allocation/physical RAM pressure and immutable lease lifetime. Next,
  develop the separately admitted SSD streaming extent architecture.
  The bounded source-only synthetic tests do not establish real large
  learned weight loading.
- [ ] Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
- [ ] Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
- [ ] Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
- [ ] Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
- [ ] Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
- [ ] Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.

- [ ] Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
- [ ] Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.

## Committed

### `d16f096` — explicit HIP total budget and resident comparison

- [x] Thread an explicit total-device budget through source admission, native
  context/cache, physical service and the existing shared serving adapter.
  Approximately75GiB total, including4GiB context; retain16GiB host. No changed
  historical defaults, numerical profile, source pins or recovery permissions.
- Implemented `HipSourceBudget`, explicit source preflight/load and
  `ModelLoadRequest::QwenHipWithBudget` through the existing owners.75GiB includes
  4GiB context;71GiB remainder includes native cache, weights, roots and workspace.
  Default API/24GiB total/64GiB physical admission remain unchanged.
- Initial controls:3 budget/placement tests and production HIP/SDK/Gemma-HIP
  strict checks pass;16 original physical/native/foreign-owner controls also pass.
  R1 exposed legacy-wrapper feature gating, fixed without enabling qualification
  in production. Mixed checkR3 uses the approved CUDA SDK path (R2's copied path
  typo preserved). Frozen SDK`9009f874` is qualification-free; diagnostic`57376b34`.
  No large-CUDA numerical acceptance implied.
- Authoritative correction: `ResidentPhysical::new_with_native_context` partitions
  context from the original provider envelope. Legacy `ResidentText` envelope is
  24GiB TOTAL (20GiB model remainder +4GiB context), with64GiB physical-admission
  ceiling. Earlier24+4 narrative was inaccurate; preserve original receipts and
  timings, but do not use that narrative as resource accounting.
- [x] Verify resident C1 and the unchanged66-call source/state reference; original
  invocation/binaries/nonces, per-case EOF/reap/group and device retirement joined.
  `qwen-rocm-benchmark-3gteresx/budget-c1-reviewed-r1b.json`; initial auditR1 missed
  the exact libtest prefix, preserved and fixed in a fresh auditor, not by changing
  outputs/references.66 calls,3137737728 state/logit words +611450880 KV-tail words,
  zero mismatches; source/ref unchanged, independent roots and genuine feedback.
- Actual source:53786705920B weights resident. Live total envelope80530636800B
  includes4294967296B context; model/native/root/workspace remainder76235669504B.
  Original resident ledger62502630529B includes opaque native charges; it is NOT
  measured VRAM. Observed C1 device peak56445587456B (~52.57GiB), returned to the
  original274448384B baseline. Host policy16GiB, cgroup12GiB RAM+8GiB swap.
- C1:5517.731/5512.832ms (mean5515.281ms),0.543943outputtok/s; load189657.308ms.
  Historical streamed mean241346.590ms →43.76x latency ratio; retained vLLM
  mean779.270ms →native7.08x slower. Historical contrast, NOT controlled ABBA.
  Native raw-text SDK/full-logit host sampling/CPU-source arithmetic differs from
  stock ROCm token-ID API/arithmetic.75GiB cap vs~74.88GiB baseline, actual usage
  differs; active8 and physical batch1. No full qualification/vendor parity claim.
- [x] Offered C1/4/8/16/32/64 completes with unchanged production SDK binary;
  all375 requests/1125 output IDs (250 timed requests) match. All original scopes
  independently retired: matching binaries/nonces/invocations, no signals, reaped
  children, absent groups/cgroups, closed writers/EOF, KFD empty/busy0.
  `budget-populations-reviewed-r2b.json`; R2b clarifies “two timed waves”, retaining
  R2's correct measurements. PopulationR2 starts only after accepted C1/stateR1.
- Output tok/s (47-input/3-output; one warmup +two timed waves per offered count):

  | Offered | Native HIP | Retained ROCm vLLM |
  |---:|---:|---:|
  |1|0.543943|3.849756|
  |4|0.542490|7.183117|
  |8|0.537208|9.248896|
  |16|0.534479|10.742837|
  |32|0.531398|11.720846|
  |64|0.529817|12.071477|

- Native active limit8, actual physical batch1 throughout; no fused throughput
  scaling. C64 baseline22.78x faster. Population device peak58989559808B
  (~54.94GiB), back to274448384B; startup189504.319ms separate from timing.
  Resident buffers/native opaque ledger/75GiB reservation/measured device use
  remain distinct. Diagnostic label corrected to `resident_ledger_bytes` after
  review; original frozen logs retain their old label, no arithmetic/allocation
  change. Affected strict example check passes.
- [x] Final isolated/offline `scripts/check.sh` runs last and passes:2360 Rust
  passes/0 failures/101 ignored across309 suites;17 Python suites/182 cases.
  `budget-final-gate-r1.log`; separate19 accounting/owner controls and physical
  source/population scopes above. No new source/fault/serving claims from CPU gate.
- Signed checkpoint`d16f096` (signatureG), no push. `qualification=false`,
  `full_reference_acceptance=false`; HTTP504 and populated1K/2K closure remain
  separate. No new download, recovery, subagents or push.

### `4f1e827` — shared source geometry, CUDA composition and serving

- Source geometry retained through the facade; backend layout/state/alias counts
  derive from typed operators. Attention, rotary and KV extents use actual heads.
- Grouped GDN composes existing16-head leaves with word-exact gather/scatter;
  direct16/16 path retained. A_log supports exact BF16→F32 widening; dt_bias stays
  BF16 per executable ABI. This is a new transport domain, not accepted evidence.
- Resident/streamed matrix bindings share authenticated sources. Admission can
  select one bounded page; strided GEMM keeps full-projection reduction selection.
  Streamed calls retain the original flight through completion/candidate discard.
- Isolated facade/test compilation passed (`composable-cuda-check-r4.log`,42.04s).
  Earlier API errors retained; R2 was intentionally stopped by owner to raise
  Cargo parallelism. Current default8 build jobs; runtime/test limits unchanged.
  25 focused host checks passed;2 physical tests remained ignored. The initial
  serial-transport filter matched zero tests; enabling the existing test support
  for CUDA produced6 executed page/cache/charge passes. Strict R1 found a kernel
  `is_multiple_of` lint; corrected using the supported CubeCL operation.
  Shared GPU and facade/test strict checks then passed with `-D warnings`
  (`composable-cuda-strict-r2.log`). Subsequent physical results below; the final
  assembled gate and wider qualification remain. No new performance claim.
- [x] Reader preallocation debt repaired: regular-file size checked before fallible
  allocation; exact read plus EOF growth probe. Review/reference JSON and32MiB
  state manifest use the same bound. Exact/oversize/sparse/hash controls pass
  (`reference-bounds-r1.log`); original reference pins unchanged.
- [x] Fresh release composition R2 passes both48-call resident trajectories and
  both3-call streamed trajectories (47-row prefill + two nonzero-state decodes).
  Streaming uses512MiB provider capacity, not a512MiB total-GPU claim. All248320
  vocabulary words and42 state domains/call match unchanged references exactly:
  102 calls/25328640 vocabulary words, zero logit/state mismatches. Original
  invocation/group/EOF/GPU retirement verified separately for every scope.
  `composition-{resident,streamed}-{08,2b}-r2/`, `composable-small-results-r2.json`;
  frozen binary`9a67353f`. Diagnostic evidence, not full serving qualification.
- First streamed2B scope failed before its first result; original failed scope
  and successful process/GPU retirement retained.85-column pages exposed a
  four-byte binding-alignment assumption at BF16 output byte170. GEMM now declares
  natural two-byte alignment; other pinned kernels retain four-byte checks.
  Alignment participates in ABI identity; exact extent/service checks remain.
  Diagnostic helper now reports initiating and shutdown errors separately.
  Seven pinned-kernel host controls, shared GPU/facade strict checks and the fresh
  release build pass (`composition-align-r2.log`); all four successor scopes pass.
- [x] Reuse the existing source-call session/root/readback/publication owner for
  production CUDA. Shared `qwen_source` adapter serves both backends; public
  `ModelLoadRequest::QwenCuda`, `QwenSourceProfile`, compatible `QwenHipProfile`.
  Raw text only; aliases cannot bypass chat refusal. Admission prices all
  `2*max_sessions+2` roots before resident/streamed placement. Original source owner
  survives uncertain cold initialization; retained calls finish before discard.
- [x] Eight small-model public scopes pass:0.8B/2B × resident/streamed × SDK/HTTP.
  Unchanged47-token/3-output reference; independent sessions/roots, shared aliases,
  chat refusal, SDK cancel/replay, HTTP JSON/SSE/disconnect/replay and final-reader
  source lifetime. Worker active limit2; physical batch1, not fused batching.
  Model/namespace retirement leaves the intended process-retained compiler
  reservations:512MiB device/384MiB host. Every original log and invocation,
  group/reap/EOF/GPU boundary inspected independently. R7 binary`3987e38b`;
  `composition-serving-results-r2.json`. No full-reference/performance claim.
- [x] Four streamed lifecycle scopes pass:0.8B actual-page cancellation/completed
  discard/replay and unknown-completion quarantine;2B normal state/restore and
  one-byte-provider prelaunch refusal. Unknown retains282 quarantined owners/
  89293157 bytes; peer reuse refused, no recovery. Original retirement verified.
  `composition-semantic-batch-r1.log` retains these passes and separate failures.
- Preserve failed scopes: first page-cut test never suspended after real launch;
  a diagnostic-only yield now cuts after queued consumers, before completion.
  Cancellation must return its exact expected error. Initial0.8B SDK scopes lacked
  the canonical snapshot; existing approved local bytes were subsequently imported
  through CacheStore, no download/repin. First HTTP launcher had a `Path` NameError
  before invocation; no HTTP execution/retirement claim for that empty scope.
- [x] Qualification-free CUDA serving build, strict facade/GPU tests and three
  root-accounting controls pass. HIP-only/mixed-feature strict checks plus six
  page/cache/charge controls and one source-call refusal control pass; no new HIP
  physical campaign. `composition-checks-build-r7.log`,
  `composition-shared-features-r1.log`.
- Large CUDA R1 refused a stale sharded-source routing guard; R2 refused core GDN
  parameter storage before inference. Both failures/retirement remain. Source
  routing now preserves the approved inventory; core accepts authentic BF16/F32
  A_log/norm storage for both arithmetic contracts, retaining BF16-only dt_bias,
  exact source/shape/bytes and F16 rejection. Backend conversion stays explicit;
  storage permission is not vendor parity. Core/large-plan regressions, strict
  checks and R8 release build pass (`composition-checks-build-r8.log`).
- [x] R8 binary`81ffe668`: approved large-source composition smoke passes genuine
  47-row prefill and feedback at48/49; IDs343/288/830,112 populated finite state
  domains, exact immutable-prior digests and clean shutdown/original retirement.
  Load83648ms; calls115707/110039/110078ms, not a performance qualification.
  `composition-large-tail-results-r3.json`. This is not full27B accuracy acceptance:
  CPU-reference logit NMSE0.124241/0.090048/0.043837, maxabs9.34375/5.494141/4.570313.
  Different declared arithmetic profiles; no distribution bound or vendor parity
  inferred from matching IDs. Grouped transport executes; numerical scope stays
  explicit, `qualification=false`, `full_reference_acceptance=false`.
- [x] Public generic `run` and65-token (64+1) chunk-tail independent replay pass
  for0.8B resident/2B streamed on R8, including normal serving retirement. Tail
  replay has no independent numerical reference;47-token checks retain theirs.
- [x] Complete isolated final gate R2 passes:2360 Rust passes/0 failures/101 ignored
  across309 suites;17 Python suites/182 cases. `composition-final-gate-r2.log`.
  GPU feature/physical results are the separate scopes above, not inferred from
  this CPU workspace gate. R1 formatting failure retained; only the owned
  `head_tile` declaration ordering changed before R2.
- [x] Signed checkpoint`4f1e827` (signatureG); only reviewed owned paths.
  User AGENTS and root README preserved; no push.
- Keep backend capabilities reusable for Gemma/DeepSeek/Mistral; source family
  semantics stay in model crates. Concrete prepared types/static dispatch;
  no per-size engine or speculative universal framework. HIP follows CUDA.

### `a57590e` — CUDA output-reduction repair

- [x] Preserve the original sliced-K/ordered accumulation boundary at
 three-quarter occupancy of a64-row output tile. Shared typed role/shape rule;
 no prompt IDs, testcase ordinals, value patches or public capability change.
- [x] Assembled-tree0.8B/2B each pass48 genuine calls against unchanged accepted
 references:11919360 vocabulary words/model, all42 populated state domains/call,
 native feedback/frontiers and two repeats. Original invocation/group/EOF/device
 retirement independently checked. `native-output-fix-{08,2b}-r1/`, binary`ac3b55e3`.
- [x] Eight real-operand projection regressions, arithmetic host assertion and
 strict CUDA consumer check pass. Complete gate `output-fix-final-gate-r2.log` passes;
 no production changes followed that gate. First formatting/lint failures retained.
- Typed original executable joins and associated SASS establish the accumulation
 chains; unsupported-schema ENTRY argument gaps remain. No serving/full-reference
 or matched-performance grant. Extra operator capture failed later and was not
 adopted as a replacement reference. Broader diagnostic helper remains uncommitted.

### `b9d8355` — authenticated source-window reuse

- [x] Reuse already authenticated chunk spans for adjacent tensor windows;
 4MiB maximum entry, unchanged8MiB cache cap, source-stamp checks and prepaid
 ownership. Outstanding readers retain their charge after eviction.
- [x] 30 source integration tests; current-tree isolated final gate passes:
 2360Rust passes/0failures/101ignored/309suites;17Python suites/182cases/10skips.
 Evidence: `qwen-cuda-regeneration-tDW0EJ1I/output-fix-final-gate-r2.log`.
- 27B HTTP improvement remains unmeasured: staged successor is blocked by the
 unchanged12GiB x2 available-memory guard. No model/serving qualification grant.

### `8ffb511` — populated Qwen HIP serving

- [x] Original0.8B/2B multi-row and16K parity: R28 accepts both66-call schedules
  (873649152/939144192words) and both516-call trajectories (955163648/956867584
  float words;128133120digest/448512tail words each),0mismatches. Original
  feedback/source/native retirement and exact physical cleanup verified.
- [x] R29 audit accepts26/26: SDK1K–16K completion/cancel/replay,16K HTTP
  JSON/SSE/disconnect/replay,32K pre-native refusal. Both16K SDK cancellations
  observed once,7tokens/49150model-token steps. Every original final reader passes;
  actual binaries, receipt and executed scripts authenticated. HTTP180s/SDK1200s
  unchanged. No repeated matrix after the monitor stopped.
- [x] R29 physical terminal is an explicit later observation of the SAME successful
  invocation. Missing contemporaneous terminal receipt/supervision gap retained,
  separately from verified per-case semantics/readers. Original capture and R12
  precheck limitations remain explicit.
- [x] Twenty small-context population cases: offeredC1..C64, active≤8, singleton.
  Profile/transport context checks and raw-text-only chat refusal preserved.
  32K refusal is not inference; prompt/head-row grouping is not native batching.
- [x] Source-exact trig/digest coverage, bounded attention/RMS work and priced
  scratch, weight-local traversal, integer status tree. Full-call PV, original
  reductions/BF16/+0, caps, roots, flights, source seals and arbiter unchanged.
  RejectedR26 tiling removed; no matched performance claim.
- [x] Final isolated gateR2:2350Rust/0failed/101ignored;309suites including74empty;
  17Python suites/182cases/10skips. Combined HIP gate exposed stale RMS preparation
  arguments in a separate consumer; corrected without changing qualified kernels.
  Numerical/physical acceptance is separate from the host gate.
- [x] [Qualification history and retained failures](archive/populated-hip-serving/TODO.md).
  External evidence remains under XDG state `qwen-context-serving-8SXpaZGt`.


### `bb400c4` — ordinary multi-row HIP SDK

- [x] Connect original `SourceInvocation`/`SourceOperatorTemplate` recording to
  the actual CPU-source contract, full4 convolution history, padded GDN chunks,
  full-call attention contraction and existing physical/session owners. No
  position-body catalogue or cached-one substitute for genuine chunk execution.
- [x] Confirm retained two-model original66-call references: fresh47/48/49/50 and
  uniform/ragged512→514, full logits/layers/state; hashes unchanged. No recapture.
- [x] Independent original F.linear49 cases: existing AVX2 dot matches20,674 BF16
  output words, including cancellation-sensitive rows and scalar tails. CPU row
  primitive and three focused tests pass. Operator evidence only, not model/HIP.
- [x] CPU full-call arithmetic matches unchanged original source:
  - Conv4:12 cold/continued cases,2,831,232 exact words, complete four-word cache.
  - Attention:20 cases,4,024,472 exact words; genuine multi-query causal calls and
    actual populated contraction lengths through32K. Operator evidence only.
  - Complete synthetic GDN layers:6 actual original forward calls,4,413,632
    exact words. Rust projections → Conv4 → chunk/cached recurrence → norm/gate →
    output projection; later calls consume actual Rust prior outputs, never
    expected states. Rows47→1→17→1 and64→1; heads16/48, mixed norm codecs.
  - Reference containers terminated/removed; parameters and inputs unchanged.
- [x] Same source seal/traversal/`ForwardPlan` now records a distinct CPU-source
  contract and MKL exp identity, preserving CUDA spellings/behavior. Core9 and
  model4 tests passed; CPU101 passed/12 ignored, plus the new masking/cancel/
  extent regression and exact Conv4/attention/GDN follow-up. These establish metadata/math,
  not native execution. Explicit cached-one declarations cannot become chunks.
- [x] Qualify the assembled native resident body. `source_call` now connects
  multi-row projection, complete Conv4, chunk/cached GDN, full-call attention,
  residual/RMS/rotary and last-row full-vocabulary readout. It traverses the
  original sealed plan through `ResidentModel`/`ResidentSession`/
  `PreparedInference`, native sets, ResourceArbiter, root pool and publication.
  No separate scheduler, publisher, position-body catalogue or repeated decode
  substitute. Shared recurrence/activation/parameter uploads were extracted,
  not duplicated. Whole-call sigmoid preserves vector/scalar-tail selection.
  Submitted CPU-contract calls complete before cancellation discards publication.
- [x] Validate original-flight observation and trained-call consumer. Completed
  producer frames use the existing bounded source-priced checkpoint. Diagnostic
  mapping reads actual residual-add producers for all24 layer outputs; no host
  reconstruction. The new test-support reader authenticates the retained66-call
  archives; `qwen_source_calls_hip` uses original resident execution, two
  interleaved roots and actual-logit feedback. Initial-root/unused-KV zero
  controls and position metadata stay separate from trained float comparisons.
  Last-row/full-vocabulary readout matches the original capture; no policy change.
- Host diagnostics pass: assembled HIP library/all-target strict check; both
  new/old source examples strict; CUDA all-target strict; two native helper
  geometry tests (one each, no ignored/empty suite). Both real models authenticate
  all66 calls/7,608 records, materialize every original invocation and derive the
  native layout at capacity514 with six roots. No GPU initialization/execution.
  - `.8B`: weights1,503,789,056B/workspace32,588,032B;
    model upper11,399,708,928B/physical upper15,694,676,224B.
  - `2B`: weights3,762,552,832B/workspace34,033,920B;
    model upper13,659,918,592B/physical upper17,954,885,888B.
  - Both: root33,275,904B; physical upper includes original4GiB context.
  - `source-call-host-r3.log` passes. R1 preserved2B's source-seal refusal:
    diagnostic reused a4MiB loader; the original seal requires1MiB. New caller
    selects1MiB; existing loader defaults and source guards stay unchanged.
  - Original one-MiB native catalogue admission preserved after review caught
    an incorrect caller reservation. Compilation/lint/API failures retained;
    HIP unwired and CUDA observer-only warnings resolved without lint suppression.
  These host preflights authenticate data, not native outputs. Later numerical,
  SDK and lifetime evidence is scoped below.
- Source lifetime integration (physical qualification below):
  - `SourceCallModel/Session/Pending/Snapshot/Readback` retain the same original
    resident objects. Private inner access prevents unpriced reader escape;
    immutable readback access keeps the actual Vec and reservation together.
  - Reserve32MiB before original invocation materialization/control/readback;
    64KiB model/session/snapshot controls. Snapshot clones share one allocation
    and reservation; reads reserve4×root+64KiB. Root initialization keeps its
    existing3×root charge; conservative host preflight now includes both.
    Configured session count is a nonblocking permit bound before allocation.
  - Same source/model/host guards retained on actual initialization, forward,
    snapshot-read flights and uncertainty. Reset/restore/commit use the original
    methods/authority; no second publisher. Source shutdown refuses live readers.
  - Proven prelaunch capacity/cancel differ from completed private Ready;
    cancellation at commit discards with actual completion metrics. Submitted
    failures stay fatal even with cancellation-looking errors. CUDA cancellation
    checks unchanged.
  - `source-call-ownership-r2.log`: HIP all-target, source example, CUDA
    all-target and minimal HIP strict pass; one nonempty prelaunch/submitted
    classification test passes. R1 large-enum lint retained; Pending boxes its
    original prepared value under the existing call charge. R3 HIP/all-target
    and source-example strict pass after final session limits, conservative host
    estimate and directly retained guard cleanup.
- Runtime source feature extracted; strict/dependency checks:
  - GPU `hip-source` depends on HIP + original artifact cache only;
    `qwen-source-qualification` depends on it, never the reverse. Normal/build
    dependency closure contains no CPU/model crates or qualification feature.
  - Shared arithmetic, resident matrices/parameters/pages, root pools and native
    context/cache ownership compile independently. Finite source bodies,
    portable/model harnesses, matrix fault cuts/row probes and diagnostic kernel
    injection remain explicitly diagnostic; no warning suppression added.
  - Weak original native/service observation moved to `source_native_probe`;
    runtime + hardware probe does not require finite source/fault orchestration.
    Legacy root exports and original cold/warm/environment/owner guards retained.
  - Ordinary-only builds omit successful native kernel/code/config telemetry.
    Actual native host/device inventory is checked against the original bound;
    call termination uses the original page `finish` guard/actor (serial `None`
    remains the same construction). Physical evidence is scoped below.
  - `runtime-feature-strict-r3`: runtime-only all-target strict passes; R5:
    runtime+weak probe all-target and runtime+CUDA+Vulkan library strict pass;
    R6: three shared-HIP/source example consumers and CUDA all-target strict
    pass; normal/build dependency tree inspected. R1–R5 failed scopes retained,
    including missing cache binding, diagnostic reachability and moved-state
    import; no native launch or float comparison in these checks.
- Canonical source/SDK integration; initial host checks:
  - Facade `qwen35-hip` + `ModelLoadRequest::QwenHip` use the existing registry,
    `Router::with_executor` and worker. One cold source,1..16 aliases, active1..8,
    context512..32768 subject to actual backend admission. Raw text only; chat,
    warm reconfiguration and native cross-session cohorts are not enabled.
  - Original CUDA verifier/held descriptor/inventory code moved to shared
    `qwen35_source`; legacy directory receipt, exact source membership/SHA,
    source limits and CUDA seal preserved. Canonical input uses actual
    `SnapshotLease` descriptors/digest/provenance; no reopen/import/acquisition.
    Same original family owner/seal/plan; tokenizer uses the already held bytes.
  - `ServingHost::from_process_admission` borrows the original16GiB system-RAM
    arbiter/key, not another wallet. Real64MiB source/cache/plan control follows
    source/native/tokenizer readers; tokenizer bootstrap/constructor/retained
    guards remain original. Device/root/call reservations remain library-owned.
    Diagnostics leave unobserved backend/source host subdivisions null.
  - Source executor keeps actual-logit feedback bound to its committed root,
    explicitly selects Decode for initialized single-row prompt tails, and uses
    genuine chunks up to64. Core finite greedy preserves first numeric ties and
    signed-zero bits. Same primitive used by the source trajectory diagnostic.
    Completed discard counts actual work without publication; submitted errors
    remain fatal. The closing shared worker invokes original native shutdown
    after active states retire; join observes that outcome.
  - `source-sdk-assembly-r3`: runtime-only/no-default facade all-target strict,
    combined HIP+legacy CUDA and CUDA-only library strict pass. Core greedy1,
    original host-domain1, source adapter4 tests pass; four filtered engine
    integration suites are empty, not additional evidence. Canonical2B host
    test separately enabled: real source/tokenizer/plan/preflight, last tokenizer
    reader retains cache/source/credit, final host returns to bootstrap baseline.
    No GPU execution or mathematical comparison in this test.
  - `source-sdk-public-host-r1`: public SDK refusal/limit/lifecycle registration
    test1 and target strict pass. Runtime normal/build dependency tree has no
    CPU/qualification edge. R1 vocabulary type and R2 borrowed-str/unused
    directory-helper failures retained; no warning suppression.
- Trained ordinary native calls pass on approved x2/gfx1151: both models66calls/
  7608records, capacity514, true47/48/49/50/64/17-row chunks and cached decode,
  all24layers/full248320 vocabulary/all42state domains (48physical planes).
  Actual feedback, two independent roots, uniform/ragged512→514 exact. Float
  words:0.8B873,649,152;2B939,144,192. Per model separate controls:99,827,712
  initial-zero words,206,512,128 unused-KV-tail words,7380position metadata words.
  Every case/extent/count independently checked against the original manifest;
  no reference recapture or repinning. Known retirement releases95/90 original
  modules, leaves resident bytes0 and original4GiB context/unfenced.
  - Evidence `native-results-r4`/`native-calls-r4-08-verified.json` and
    `native-results-r5`/`native-calls-r5-2b-verified.json` under the multi-row run.
    R4 model exits0; immediate sampled-utilization guard stops its supervisor
    before2B, later verified idle/KFD empty/MainPID0/empty cgroup. R5 contains
    only2B (no redundant0.8 replay), exits0 with terminal ownership verified.
  - R1 protected-process-FD guard, R2 stale Python PATH and R3 noclobber lock
    setup failures retained; no model executed in these scopes. Readable KFD
    process inventory supplies authoritative idle ownership without privileges;
    unrelated descriptor-forwarding service remains untouched. Retrieval banner
    parse failure retained; corrected retrieval checks exact sizes/digests.
- Public SDK native qualification exposed two integration gaps; fixes assembled:
  - SDK R1: finite wave guard correctly rejects context514. Adapter now selects
    existing sequential/fair worker with original64-row prefill, not enlarged
    wave limits or a new scheduler. Runtime-only all-target strict passes.
  - SDK R2: actual token arrives, but sequential presentation drops supplied
    GPU publication metadata. Shared `accept_observed` now preserves the
    original observation and counts actual presented tokens; CPU compatibility
    and backpressure remain unchanged. Engine strict + regression pass;85 other
    router tests already pass. Initial regression fixture47/7 error retained.
  - New runtime-only `tests/qwen_hip_sdk.rs`: approved manifest-pinned IDs,
    aliases/independent roots, public generic run,65-row prompt tail replay,
    disconnect/cancel and native/host/tokenizer final-reader checks. All four
    fresh native cases now pass: both models × complete/disconnect. Exact pinned
    IDs, generic run, independent alias sessions, prompt-tail replay, native4GiB
    context/unfenced and last tokenizer/source/cache/host reader assertions pass.
    Runtime-only binary; no qualification execution dependency. Native SDK R1/R2
    failures retained. R3's first completion passed; immediate KFD retirement
    sampling stopped its batch supervisor, not the test. Remaining three cells
    passed in fresh R4 processes, without repeating the successful first cell.
    All four logs inspected; terminal KFD/cgroups empty/GPU idle. Evidence:
    `native-results-sdkbatch-r3` and `native-results-sdk-{08-disconnect,2b-complete,2b-disconnect}-r4`.
- Observer-free/state/lifecycle native batch accepted, both models:
  - All66 original calls/model:450,794,496 exact post-commit state/full-logit
    words/model; all42domains/48planes, independent sessions and actual feedback.
    Original typed snapshot readers separately check114,647,040 KV-tail words.
    No per-layer observer or reconstructed feedback; observed trajectory evidence
    above remains unchanged and was not repeated.
  - Healthy controls: prelaunch capacity/cancel, session limit, cross-session
    refusals, submitted cancellation → known completed discard with1862dispatches,
    snapshot clone charging/restore/reset/replay and independent roots pass.
  - Original final session/snapshot/pending readers: actual lease release before
    native retirement, known/fault pairs. Immutable inference/snapshot readbacks
    retain model/native/source until the actual final reader. Known paths leave
    native/resident0, device4GiB, host2GiB +64MiB reference support, unfenced.
    Release faults retain95/90 original modules,543/539 handles, source/cache
    and24GiB device fence; physical peer admission refused.
  - Unknown completion:495/491 submitted handles,1pending event,543/539 retained
    handles and completion owners; unchanged published roots, peer/reset/read
    refusal, original native/source/cache/charges retained after caller drops.
    Pre-fault5,449,728 state words exact/model. Post-fault device state and pending
    outputs unavailable, not numerical comparisons or recovery evidence.
  - R1:20cases pass; two unknown diagnostics incorrectly attempted a post-fence
    GPU read. Original native policy correctly refused. Removed only that new
    diagnostic path; R2 asserts the original snapshot fence refusal and both
    unknown cases pass. No production fence relaxation; no passed case repeated.
    Every case/count/cleanup independently inspected; terminal KFD/cgroups empty,
    GPU idle. Evidence: `native-results-state-lifecycle-r{1,2}` and
    `state-lifecycle-verified-r{1,2}.json`. Earlier compile/lint failures retained.
  - Affected bare HIP-source+hardware, CUDA+hardware and runtime-only SDK strict
    checks pass. Diagnostic support remains outside the production dependency
    closure; no native arithmetic, source pins or accepted trajectories changed.
- [x] Final complete isolated `scripts/check.sh` passes:2348Rust passes,0failures,
  101ignored,309suites including74empty;17Python suites/182cases/10skips.
  Two existing unused HIP probe-hook warnings remain in the Vulkan-peer compile
  scope; no suppression or runtime change. `final-checkpoint-gate-r1.log` and
  `final-checkpoint-gate-summary-r1.json` retain the complete result.
- [x] Land signed`bb400c4`; signature verified G with the expected key. No push.
  Next larger milestone: populated contexts/serving concurrency, then matched
  performance; no Fabric reopening or per-helper acceptance cycles.
  Run-local evidence: XDG state `abird/zerogravity/qwen-multirow-4DkQcm4c/`.

### `e2559b6` — original-source native chunk GDN

- [x] Shared GPU chunk arithmetic and a test-only original-owner comparison now
  compile. Durable x2 `gdn-native` r1 refused its incorrect4MiB catalogue owner
  against the existing exact1MiB contract before numerical cases; parent101,
  empty descendants, evidence saved and owned unit stopped. Caller fixed to1MiB;
  original guard/arithmetic unchanged. R2 matches all19 cases/380 fields/
  64,345,520 words and immutable inputs exactly, per-case physical baseline intact.
  Negative fixture incorrectly confused completed-write initialization with Ready
  exports; first assertion failed,3 controls/final retirement unrun. Failed101
  scope preserved/closed. Correct diagnostic checks actual status-only Ready
  exports and original completion identity; runtime/arithmetic unchanged. R3
  now passes all19/380/64,345,520 words plus4 status-only Ready exclusions and
  actual20-module/native/namespace retirement. Independent all-scope reconciliation,
  parent0/empty descendants saved; owned unit stopped. HIP/Vulkan strict passes.
  Final isolated gate passes2,142 Rust/0fail/87ignored/283 suites and17 Python
  suites/178 cases plus packaging. Initial formatting-only refusal retained;
  module-order correction changes no arithmetic/owner. No model lowering,
  source/profile permit or ordinary/serving grant.

### `55536ce` — multi-row references and connected CPU GDN

- [x] Capture/inspect independent original0.8B/2B call references through514
  populated positions: fresh47/48/49/50 and uniform/ragged512-token chunks,
  two natural decodes, two replays,66 calls/7,608 records/model. All source,
  full last-rowV, layer/state, replay and durable closure scopes inspected;
  owned unit stopped. Schedules/IDs are test data, never runtime routing.
  Same-plan replay is exact; original cross-chunk equality is false.
- [x] Implement source padded64 CPU GDN: actual triangular reductions,
  initialized-C matrix arithmetic, FP64 cumsum/rare decay and BF16 output.
  All19 independent original operator cases/380 fields/64,345,520 words match
  exactly. Source state remains private; one-row chunk is not cached recurrence.
  Normal exp-profile guard unchanged. Retain all prior failed scopes.
- [x] Final reviewed comparison/CPU/diagnostic strict and isolated gate pass:
  2,142 Rust/0fail/87ignored/283 suites,17 Python suites/178 cases plus packaging.
  Unchanged native arithmetic/ownership reuse accepted evidence; this checkpoint
  adds no multi-row HIP/model/serving eligibility. Full scope/evidence in
  [numerics TODO](../numerics-and-context/TODO.md).

### `85f33c7` — canonical source/native lifetime qualification

- [x] Qualify canonical `CacheStore` snapshot/native lifetime ownership. The new
  source owner retains the actual snapshot and duplicates its pinned descriptors;
  native staging paths refuse, while cold preflight still permits staging.
  Two schema/provenance/path controls and standalone/shared-HIP/shared-Vulkan
  strict checks pass. Durable `source-lifetimes` r1 tests both models: canonical
  pin controls, Session/Snapshot/Pending known/fault readers and abandoned
  original pending completion. All14 scopes inspected; parent exit0/empty
  descendants, completed owned unit stopped. Healthy readers release all
  785/781 modules and source/cache pins; faults retain them with a24GiB physical
  fence. Unknown retains752/747 handles,189 completion owners and one event;
  cached logits stay unobserved and no root publishes. Evidence:
  `ordinary-hip-20261031.Jc5XWh/source-lifetime-all-scopes-r1.log`.
  Six healthy r2 scopes additionally prove physical fence clearance and other-owner
  admission, retaining only the original4GiB context/2GiB host charges plus the
  detached64MiB reference lease. All scopes inspected, parent0/empty descendants;
  completed unit stopped. Final isolated gate:2,136 Rust/0fail/85ignored/283 suites,
  17 Python suites/178 cases; protected roots unchanged. No serving enablement.
- [x] Qualify unknown-completion and final-reader release faults on actual
  small-model execution through original primitives; r1/r2 above, not finite27B
  substitution. Unknown cached logits/root publication remain unavailable.

### `1f29a42` — source-bound small-model HIP

- [x] Assemble real source-state lowering, shared resident weights/workspace,
  source-priced cold admission and existing session publication for0.8B/2B.
  Bounded implementation derives state from typed SSA producers, shares one resident
  matrix/workspace set across actual frontiers, and reuse the original session,
  pool, queue, flights and publication. Finite guards remain separate; explicit
  F32/BF16-widened storage, not a renamed CUDA seal/flag. Core assembly passed
  strict compilation. Standalone actual-source preflight now passes both models:
  64 ordered records,11 refusal/accounting controls and root pools4/6/130.
  This is cold pricing, not C64 inference. HIP all-target and shared Vulkan strict
  checks pass. The first x2 physical attempt prepares785 modules, then refuses
  premature resident binding before original initialization Ready. Move pending
  operand binding after that barrier; do not relax initialization checks. Parent
  exit1/MainPID0/empty cgroup verified; in-process native retirement and model
  parity are not established. Failed log:`source-model-physical-r1-08.log`.
  Post-Ready r2 reaches0.8B full-model comparison but fails303 words; first
  after.recurrent/layer0 word50816 is negative zero versus original positive zero.
  All785 modules retire and namespace/source close; parent exit1/empty cgroup
  verified, completed unit stopped. Original initial recurrence adds its decayed
  empty state; CPU/HIP had omitted that addition. Restore actual arithmetic,
  not output canonicalization/tolerance. Four focused host tests and standalone
  strict check pass; initial test's unsupported one-head fixture refusal retained
  and corrected to genuine16-head geometry. Richer per-field mismatch reporting
  stays in test support. Fresh r3 passes both models as detailed below. Review
  also requires every position's capacity to equal the complete record count,
  independently of whether the source has KV domains; both fresh actual-source
  cold controls pass. Source/session remains position-bound/single-row through64.
  Final isolated gate:2,136 Rust/0fail/85ignored/283 suites;17 Python suites/178
  cases, release/installer/shell/Nix. CPU/diagnostic/HIP/shared-Vulkan strict checks
  pass; narrower historical Vulkan lint refusal remains a separate failed scope.
- [x] Capture independent small-model whole-source references from the approved
  cached Torch2.13/Transformers5.17 original runtime, AVX2 CPU, raw published
  parameters and original full forward. Each model:64 genuinely populated
  single-token positions, two independent exact replays,128 calls/15,776 records;
  fullV, layer outputs, rotary positions and every populated Conv4/recurrent/KV
  state. Inspect every record/digest, natural selection, source role and closure.
  `.8B`:1,396,408,320 float words/5,189,864,448B;
  `2B`:1,399,816,192 float words/5,196,680,192B. Original source members remain
  11/12 with authentic indexes,488/632 tensors. Durable local owner exits0 with
  no descendants; both containers/groups/reaps/EOF close. Completed unit stopped.
  These are original-reference results, not HIP parity or serving acceptance.
  Evidence:`ordinary-hip-20261031.Jc5XWh/small-original-r1-{08,2b}` and
  `small-source-reference-r1-validation.log`. No source/image substitution.
- [x] Transfer29 source/reference members,16,744,116,588B, to bounded x2 staging
  with complete size/hash/EOF checks. Import0.8B through canonical `CacheStore`:
  snapshot`d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c`.
  Legacy provenance remains Unspecified, not upgraded. No download or inference
  claim. `2B` now imports12 exact members into canonical snapshot
  `88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52`; no download.
- [x] Physical r3:0.8B/2B full24/full248,320V/all populated Conv4/recurrent/KV
  state,64 positions × two independent natural replays. All15,776 records/model
  inspected;1,396,408,320/1,399,816,192 words exact. Pre/late cancellation,
  proposal discard, session/snapshot pressure, live-owner shutdown refusal,
  independent complete roots, restore/reset and context64 refusal pass. Actual
  native785/781 modules retire, namespaces/source owners close; host unfenced.
  Durable parent exits0/MainPID0/empty descendants; completed owned unit stopped.
  Evidence:`source-model-all-scopes-r3.log`, physical logs and original pinned
  references. Earlier r1/r2 failures remain separate. Assembled gate passes above;
  no production/serving/performance permit or concurrent-root throughput claim.


### `f6fa4b9` — shared source execution foundation

- [x] Share source traversal and optional observations independently of CPU
  execution/layer9 diagnostics. `record_source_token(position,capacity)` derives
  actual full4 Conv, recurrent and KV domains/extents from authenticated geometry,
  retaining original metadata/source ownership. Bound64 tokens/layers; the finite
  wrapper still requires2 positions/64 layers/112 domains. Host194/0fail/1ignored;
  source-only CPU-free and shared-HIP all-target strict checks pass.
- [x] Separate shared HIP kernel preparation from explicit diagnostic probes.
  Preserve existing Qwen/Gemma probe order, issuer/metadata checks, original
  service/queue/flight/native owners and unchanged cold ceilings.
- [x] Share actual GPU attention execution, geometry-sized scratch, source-service
  range/alias checks and matching kernel constructors. Extract sigmoid and KV
  append/output gate; keep the finite attention adapter's position<=1 guard.
- [x] Independent original Torch2.13/cf30153/HF5.17 AVX2 reference covers40 calls:
  8Q/2KV at16 active lengths through64;24Q/4KV at1/2/47/64, random and arithmetic
  cancellation. CPU and real gfx1151 each match all1,774,240 words across360
  fields. Original reference and durable HIP worker both exit0 with verified
  owner/descendant closure. Every17-launch case inspected,150 native modules
  retire, original model namespace closes, completed supervisor unit stops.
  No trained-model, root publication, fault-cancellation or serving claim.
- [x] Final isolated `scripts/check.sh`:2,135 Rust executions/0fail/85ignored,
  283 suites;17 Python suites/178 reported cases, release/installer/shell/Nix.
  Existing non-strict feature warnings match the preceding accepted gate;
  selected CPU/HIP strict checks are warning-free. Initial format, CPU-free
  observer, diagnostic callback/namespace and remote-image failures retained.
  Evidence: external `ordinary-hip-20261031.Jc5XWh`; no repeated finite27B run.



### `d4d5df4` — original eager CPU attention

- [x] Fix PV four-part BMM reduction; retain the distinct projection/QK schedule.
  Original32-case/65,536-word and complete40-case/1,774,240-word comparisons
  are exact;16 analytic arithmetic-cancellation cases cover lengths through64.
  Original seven-token failure retained; no tolerance or source change.

### 6d85559 — common comparison tools, 26 Oct

- [x] Shared model-output schema/checks and exact/default comparison policies,
  independent distribution/operator measurements, strict negative controls;
  complete isolated source gate0/EOF. Both .8B/2B retained C16 output comparisons
  pass; this adds no fresh execution, full-model or cross-backend permit.
  [Policy](../numerics-and-context/CORRECTNESS_POLICY.md) and
  evidence.

### `29a98c1` — HIP source-page events and witness (25Oct assembly)

25Oct integration: accepted H14 pending-event source plus final sticky witness
error projection is now copied into MAIN with fresh mtimes. ROOT independently
matched all14 files/524,658B and modes; README/index preserved. New diagnostic
bound2,640,926B consumes the existing128MiB fixture allowance; conditional cold
host33,530,474,220B is unchanged. Enabled-HIP typing and actual H6/G8/V10 physical
cases remain pending. V's fixed SSH adapter is source-reviewed only. CPU queue
16/32/64, cap/cancellation/HOST and actual P16 tokenizer checks pass separately;
they do not qualify GPU models or performance. The collected CUDA direct-route
ownership/host-cost repair landed in `52f76a0`/`efd7a34`; the final full CPU
gate passed rc0 25Oct and the five-lane batch landed as signed commits
(`8f01635`..`68e13a8`). Five-tree retirement remains pending.

- H14 accepted source is the transferred state behind this commit;
  enabled-HIP typing and the actual H6/G8/V10 physical cases remain open in
  Uncommitted scopes. Vulkan lane source landed in `19ff059` (final five-fix
  strict-SSH adapter source-reviewed; V10 physical pending); Qwen27B audit
  corrections folded into `e8db111`; Gemma4 examples/tools into `332038e`;
  queue serving into `5d40cf5`/`52f76a0`.


### `225fc5e1a520` — shared family sources and finite learned-model diagnostics

Related source commits: `bebfeaf2d817` (gemma4-text-reference), `9af407bf3ff5` (qwen35-27b-window-replays), `2058dcb1ea3f` (facade-qwen27b-hip-source), `71f496b190dc` (facade-gemma4-hip-drivers), `752642361cb4` (facade-shared-hip-lifecycle).
Aggregate source21 qualification applies to the assembled tree; no per-commit
build or general model/backend grant. Historical failures keep their source scope.

#### Current shared source20 hardware checkpoint — 24 Oct

- [x] ROOT accepts the actual x2 offline build/binary binding and complete
  finite known3Q/4G reload plus two unknown1Q/1G lifecycle cases. Revalidated
  original typed receipts, exact1650 source,70dataFD, artifact51808f9e…,
  global fence, both process-group/reap/EOF layers; all earlier failures kept.
  Actual scope.

- [x] ROOT authenticates original Qwen teacher receipt/manifest and independently
  scans allfour248320V BF16 records. Both original call0 select220, while held
  call1 consumes353; compositional natural continuation refuses correctly.
  Fresh natural reference producer/validator/controller preparation required,
  never relabel teacher call1. ROOT+Astra reject correction draft: signedzero
  stillwrong, fresh package table stale, scopes/staging incomplete; replacement
  controller has invented caps/reap-before-signals. No merge/physical grant.
  Coherent corrective design.

#### Qwen27B and Gemma4 optimization audit — 24 Oct

- [x] Separate parallel read-only audit of actual vLLM0.31 benchmark sources,
  stable0.32 and frozen current upstream;20Qwen/20Gemma/15shared comparisons.
  Report, exact model matrix and source links.
  Neither family currently has all applicable vLLM optimizations. Source dispatch,
  planning inventory and source113 assembly do not grant physical/serving/perf.

- [x] Independent full Gemma4 series review: G01–G20 × E2B/E4B/Unified12B/
  26B-A4B/31B, plus shared S01–S15 eligibility. Existing raw K=V projection
  reuse is already in the family schedule; final normalized/rotated KV stays
  distinct. Pinned optimized Triton MoE routing and Torch fallback have different
  reduction schedules. Current E2B GPU12,288-column limit does not admit12B/
  31B down projections; extend source-bound profiles/inventory, not inherited
  limits or permits. Variant KV owner/PLE costs and MAIN versus frozen118
  boundaries are explicit. No new runtime/performance qualification.
  Complete matrix, corrections and reuse map.

- [x] Source466 x2 readiness/control unit reviewed and executed through actual
  packs/builds/GPU-free imports; Gemma Resident20 healthy+fresh unknown PASS.
  Old namespaces/controls immutable; source466 excludes later RMS3/source6.
  Commands, costs and limitations.

- [x] Fresh Gemma source466 source-only pack reviewed:32,286,720B archive
  `28fcd32410715277668534efc14ce18aadf998b0fd5bbde6e759480ad482226d`,
  sourceclosure`b404cb9a831b896d5a179a07a08b27f9fe0ccf3ce9f8e2d5ec2f7839f5d19330`.
  All1576regular tar members' bytes/SHA, selected466 source/modes,11controls and
  13protected guards exact; source/tar owners closed. Local prep grants no fit.

- [x] Gemma fresh source upload/bootstrap/build and complete original P2/source
  closure PASS; actual binary/source identities retained below and in run results.

- [x] Qwen genuine local data/source preparation closed:35member data archive
  56,521,973,760B`8109be7cd03b1232569f312fac75496ee6dfa6da6ccffad13e30d8ec63b09036`,
  source32,276,480B`39ae07bde9a08afc0e8203b09762c4682e93aae934841620e2802dbb4ffdc2c8`.
  Original held source reads/copy hashes stable; all1575source tar members exact,
  selected466modes/10controls/13protected guards exact; shared1562source records
  match Gemma. No downloaded weights or data-derived expected outputs.

- [x] Both original fresh admin/bootstrap/build units CLOSED0; preparation/build
  alone do not establish GPU fit. Actual model scopes remain separate below.

- [x] Both source466 fresh x2 builds passed with actual held artifacts/source and
  complete original owner/EOF closure. Gemma readonly35FD authentication and
  genuine GPU-free new96GiB CacheStore import passed:8payloads10,278,848,001B,
  snapshot`ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1`.
  Installed user/PID/net/bwrap readonly35data+binaryFD namespace executed; old
  source/cache unchanged. Subsequent actual GPU dev-bind/Resident20 scope passed
  below; no whole-family or serving/performance grant.

- [x] Gemma E2B frozen466 actual x2 HIP Resident20 healthy+fresh unknown CLOSED0.
  All55diagnostic calls/all35blocks/full262144V/all15KV exact,52commits150rows;
  50normal observer=None calls/50commits145rows preserve actual state/private
  feedback with zero optional producer/KV D2H and repeated selected-weight H2D.
  PLE selected-row H2D remains2,598,400B; explicit state snapshots separate.
  Healthy model/source/native close retains honest4GiBdevice/2GiBhost context
  until process retirement. Freshunknown1producer/0roots/full20GiB fenced/source
  close refused and original owners retained until process retirement. Actual
  184native modules independently priced; complete device17,469,201,928B under
  unchanged20GiB profile.16physical round cases pass; all source/data/binary/
  original groups/one-reap/fullEOF closures pass. Shared sampledVRAM is not an
  isolated peak; seven fresh ELF hashes differ explicitly. No serving/speed or
  current late-RMS/source8 grant.
  Complete actual scope.

- [x] Qwen466 actual original receive+GPU-free new96GiB CacheStore import CLOSED0:
  31payloads55,586,113,293B, snapshot`98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e`.
  Same held binary1fec59d5/sourcea9d86cb4; all35dataFD/fullSHA/source and original
  local/remote owner/EOF closures stable. No model/GPU call from import.

- [x] Qwen466 original x2 healthy sixfull64/all112/two teacher-session matrix,
  then dependent freshunknown authorized. Healthy localSSH18683 CLOSED1 after
  wireless route loss; original local group/reap/EOF closed but remote GPU is NOT
  retired by that receipt. Subsequent configured strict-hostkey Tailscale read-only
  collection confirms original remote healthy PASS/1097.656s: sixfull64calls,
  fourcomparisons/commits/two independent teacher sessions,240,896,000floatwords/
  12positions/zero mismatches, fullV/all112domains128planes/Conv4/priors/finals.
  Ready discard/latecancel/pressure/restore/reset/lastoutput/sourceclose pass;
  117actual native modules/16physical round cases pass. Remote original supervisor/
  child absent, group/one-reap/fullEOF/source/data closure healthy. Local original
  transport failure stays FAILED_CLOSED255, not an end-to-end collector pass.
  Freshunknown88524 CLOSED0 through reviewed LOCAL-only configured Tailscale
  transport derivative:1actual producer/0roots/full16GiB fenced/sourceclose
  refused, original source/native/armedflight retained until process retirement.
  Both original local+remote owner/source/group/one-reap/fullEOF closures pass.
  No healthy replay/recovery. Mandatory native-gpu user/net and held
  binaryFD; supervisor-held35dataFD SHA/lifetime. Gemma's nested readonly-host/PID/
  dev-bind/dataFD namespace is not present in this original Qwen GPU controller;
  import's private GPU-free bwrap is separate. P1 final-owner Pending/Snapshot
  actual HIP fault coverage remains open regardless of finite matrix outcome.
  Complete physical scopes and transport distinction.

- [x] Frozen469 alias-v2 actual P1 final Pending/Snapshot reader retirement:
  all4 fresh HIP cases PASS; ROOT independently checked every actual report and
  original local/remote group/reap/fullEOF/source/binary/35FD/cache-parent closure.
  Internal aliases3/3/2/2, commits0/0/1/1, Candidate1/model1/workspace-target0;
  exactly one true original lease release precedes one native retirement.
  Known:117native modules retire/model0/sourcecloseOK/context4GiBdevice+2GiBhost.
  Fault:117modules/1602handles/service/source retained/full16GiBfenced and
  sourceclose refused until actual process retirement. Each real64layers/
  128proposedplanes/full248320V; unchanged numerical matrix reused, no replay/
  old-order negative/natural-temporal/serving/speed grant. Original failed469
  incomplete internal alias census remains failed; exact equality preserved in
  repair. Genuine own P1 cache reused with actual locks, no second55GB import.
  Four-case results.

#### Current model/backend checkpoint — 23 Oct

- [x] **Original27B final63 tail, external boundary only:** pinned public CPU
  reference and the existing Rust `OriginalWindowOperators::execute_output_tail`
  agree exactly on all2,609,152 BF16 words: four external post-attention probes,
  independent1+3 rows, two repeats, nine complete streams including248320 head.
  Actual31-file/18-shard full source identities, held reference FDs, pre-cancel,
  real exhausted-host refusal, output retained after caller drops and final
  host0/unfenced pass. Nine Python fixtures, Rust example3/strict Clippy and
  complete isolated scripts/check.sh pass. Expected pin is sealed after actual
  independent validation; no expected bytes supply computed stages. Scope ends
  at this tail: full64-layer producer,112-domain StateRoot, GPU/serving and
  performance remain open. Source now integrated in MAIN and ROOT; fresh MAIN
  full checkgate passes. The external-tail numerical proof remains the original
  ROOT replay above; no fresh full27B execution follows from source convergence.

#### Parallel model work — contained integration, 22 Oct

- [x] Completed combined40-path CPU donor integrated into ROOT ISO with fresh
  bytes: Qwen27B final63 attention→existing output tail plus Gemma4 genuine
  26B-A4B MoE through the same mathematical backend/source-window/session paths.
  Reciprocal source review, combined CPU libraries/replays and strict all-target
  Clippy pass in isolation. Learned Qwen rotary/BMM parity remains unqualified
  (reference pinNone); Gemma learned checkpoint/oracle not yet qualified.
  Source readiness does not establish trained/native/serving/performance support.
  This and the follow-on CPU batch now converge into MAIN; final assembled
  isolated MAIN gate PASS1531 Rust/0failed/81ignored/260suites plus18 namespace
  fixtures and complete scripted release/installer/lint/qualification checks.

- [x] Follow-on CPU model source batch integrated into ROOT ISO: Gemma31B
  exact60-layer/60-owner analytic schedule and21504-column windows (113isolated
  CPU tests/strictClippy PASS); Qwen27B original64-layer traversal/shared H48
  GDN hook/private SessionAuthority diagnostic (176/82/74libraries+6replays,
  strictClippy PASS). Existing public trained/native routes remain refused.
  Qwen initial-one is not originalchunk64; minimal Conv history3 is distinct
  from original HF4. Real retired H48 exp differs2/48 and recurrent state
  5539/786432 words; generic f64 exp also fails. Preserve fatal exact comparison,
  no tolerance/correction or complete112-domain/full64 numerical claim.

- [x] Complete owner-authorized managed HF Gemma4 E2B pull at pinned
  3e22461f65e89153144f8adb70e3b8c2cc9845a7. Tokenizer/config sidecars are
  acquired and verified. Owner-approved canonical cache64→96GiB applied,
  preserving models; full8file/10,278,848,001B managed checkpoint closure
  published and acquisition hashes verified. Offline full closure verified;
  authentic263,952B header/2,011BF16 tensors and derived index prepared, all600
  text members match registered geometry. Acquisition grants no Rust/native or
  serving qualification; original learned reference/replay results follow below.

- [x] Gemma E2B original learned CPU reference executed and independently
  validated: full checkpoint, BF16/eager, prompt4 then two natural continuations,
  two independent cache cohorts,93 records/full262144 logits/all15KV owners and
  exact consumed priors. Original8-file source and owned lifetime close healthy
  under12GiB/4CPU. Strict Rust AnalyticF32TextV1 replay completes the full35-block
  first call, rejects logits[0] (`c1ce0000` vs `c1ce9ba0`), publishes no root,
  retains unchanged source and releases reservations to0. Neither numerical
  parity nor native/serving/performance support follows from this failure.

- [x] Gemma BF16 whole35-block candidate and producer-boundary tracing merged
  into ROOT ISO (20 paths); default analytic arithmetic and trained refusals
  preserved. Source-defined embedding/PLE scales, direct-gamma RMS, rotary,
  projection/BMM, residual and final casts use the same ReferenceText engine.
  CPU tests/strict Clippy pass. Fresh original reference2 passes all natural
  calls, fullV/all15KV/repeats/priors/source/lifetime,1981 producer records.
  Strict whole35 replay rejects first original-order boundary at layer0,
  position0,ffn.gelu[416]: source -0 (80000000), candidate b4240000.
  Root/frontier remain0, final reservations0; this is not learned parity.

- [x] Resolve Gemma candidate transcendental arithmetic at its source:
  faithful pinned BF16 AVX2 GELU/tanh schedule, same full-range formula and
  source saturation; no expected operands or input correction table. CPU batch
  passes; strict whole35 replay against unchanged healthy reference2 closes the
  selected producer prefix through810 complete records, then rejects layer12,
  position0,ple.post_norm[880] (3b830000 vs3b820000). Same1981 observations,
  root/frontier0, unchanged source/final0 and original owned closure. Seven
  source paths now converge MAIN/ISO. Pinned ATen pow(-.5) selects1/sqrt;
  candidate general powf differs. Shared source-shaped RMS correction is now
  merged MAIN/ISO; CPU batch/strict Clippy PASS. Actual whole35 replay matches
  all1981 selected initial producer records and initial+first cached fullV/all15
  KV pairs, commits greedy886→2915/frontiers4→5. Second cached call fatally
  differs logit0 (c1b00000 vs c1b10000); prior root2/frontier5 unchanged, source
  and actual owned closure healthy/final0. Its KV comparison was not reached.
  Detailed cached producers were not captured, so initial matches do not
  establish cached arithmetic. Extend all-call diagnostic comparison/tracing
  within existing caps, then compare a fresh independent producer reference.
  This was the intermediate diagnostic limit; later PV closure below supersedes
  its selected three-call failure. General trained/native/serving stays refused.

- [x] Qwen27B original full64 producer executed healthy: teachers11→353 with
  two independent cache cohorts, all851 raw source roles/67modules,1040 records/
  934,776,928B, authentic Conv4/FP32 recurrent/head-majorKV. All851 observed source roles
  are BF16, no source casts. All31 whole source
  identities/mapped bytes unchanged, owned exit0/group/container/EOF close.
  Eleven additive producer tools plus contained observed-exp causal test merged
  into ROOT ISO; source-free Python protocol8+4 fixtures PASS. Independent fresh
  original archive/source validation PASS all1040 records/exact repeats/priors/
  fullV/all112 domains and source/lifetime. Rust parity remains false.

- [x] Qwen original full64 archive connected to the existing Rust diagnostic,
  explicit HF Conv4→minimal3 mapping and complete available-record comparison.
  INITIAL1 earliest producer divergence and correction remain tracked below.

- [x] Separate cached Qwen GDN exponential correction, finite original profile:
  actual original SSE HA dispatch reconstructed with mathematically derived
  split128table/separateF32 operations; both A_log→A and logdecay→decay computed.
  Authenticated H48 connected producer/all786432state words bitexact; legacy
  2/48decay and5539state negative control preserved. Actual full64 replay3 closes
  initial+cached/2independent sessions/all1040records/240896000words including
  all4Conv/completeKV/fullV, source/ownership healthy. No golden operands or
  looser limit. Numerical primitive's explicit finite normal domain remains;
  general mode/tails/natural/long/native/serving/performance stay unqualified.

- [x] Qwen full64 Rust replay/source-owned scratch unit merged into ROOT ISO
  (13 paths): same traversal/SessionAuthority, explicit Conv4→minimal3 and
  head-major→token-major mappings, expected-only completed-boundary observer.
  H48 transient scratch retires after synchronous compute; output/state retains
  original ownership. Whole-replay host envelope768MiB, derived upper651296768B;
  old128MiB leaf limits unchanged. CPU105 checks/strict Clippy pass. ROOT sealed
  only the expected-target reference pin after validating actual retired raw
  archive/manifest/invocation/source. No kernel operand or runtime permission.

- [x] Recover the first learned Rust replay's pre-boundary GDN output capacity
  refusal with bounded exact output allocation below; no raised caps. Initial
  refusal opened no expected archive and kept root/final0/source/owned closure
  healthy. Subsequent full64 replays now reach complete numerical comparison;
  startup-only namespace/affinity refusals establish no numerical result.

- [x] Qwen bounded GDN output correction integrated MAIN/ISO: declare/reserve
  exact output before fallible collection, retain capacity and element bounds,
  reject short/excess/original errors. Covers activated/gated and analogous
  cached recurrent BF16 output, unchanged arithmetic/caps. H48 reproducer
  confirms prior capacity8192/len6144; related CPU47+whole-window5 and strict
  Clippy pass. Corrected learned replay5 completes the actual64-layer proposal,
  then strictly rejects layer9 output word13 (3cc80000 vs3cc60000). Embedding,
  layers0..8 and first13 words of layer9 match,51213 words in this reader order;
  no post-state/internal-stage comparison was reached, no cached call ran.
  Source before/after/full expected SHA/EOF, pre-cancel/real pressure and actual
  owned closure pass; root unchanged/commits0/final host0/unfenced. Observed
  boundary reservation517760000B is not an exact peak. No causal attribution
  to older cached exp results. Next same completed callback must compare all
  available state/boundary/fullV records before refusing, with bounded summaries.

- [x] Complete Qwen proposal comparison source merged MAIN/ISO;9 fixtures and
  strict Clippy PASS. Actual learned6 compares all196 initial records/39842304
  float words plus3 position metadata words. All20 Conv3/H/KV tensors through
  layer9 exact; first differing state is layer10 Conv after known layer9 output
  divergence. First bad layer9 output differs381/5120 words. FullV differs
  197638/248320; no commit/root unchanged/final host0/unfenced and full source/
  expected/owned closure pass. No hidden earlier state mismatch was found.

- [x] Qwen layer9 causal diagnostic source merged MAIN/ISO (20 paths): opt-in
  fifteen borrowed producers from the same full64 Rust traversal, completed
  full-proposal and48-record prefix comparison before numerical refusal. No
  arithmetic/default/native change; conservative upper664928256B under same
  768MiB cap. Python6/session6/example10/strict Clippy PASS. Original full64 pin
  unchanged; separate prefix pinNone until independent actual validation.

- [x] Original64-object prefix independently validated: two fresh initial
  caches/135 used roles/18 logical domains/96records/52355504B, all66 overlaps
  byte-exact to unchanged accepted full64 original. Stop after layer9 source
  unload; source whole checks/31FDs/actual process+container+group/reap/EOF close.
  ROOT checked artifacts and sealed distinct expected-only prefix receipt
  14593B SHA4caffb384f0a67b20bb3a81aded8fc151b4bfd949659aab28cb2d5bf7a02b657.
  Old full64 pin unchanged; no original final/head/full-model claim.

- [x] Actual full64 layer9 causal replay closes with strict numerical refusal:
  input/InputNorm/QKV/Z/B/A exact; first error GDN core2/6144 BF16 words,
  word806 actualb6d60000 vsoriginalb6d70000. NormGate exact, gated2→Out32→
  postresidual3/postnorm3→MLP divergences→381/5120 output differences. All20
  layer0..9 state tensors exact; first state error layer10Conv downstream.
  Full196-record comparison reproduces learned6. Root unchanged/commits0,
  source/31FDs/expected/actual owned closure healthy, finalhost0/unfenced.
  Observed boundary530355200B<derived664928256B<768MiB is not a peak.

- [x] Correct Qwen INITIAL1 GDN core at its source. Genuine original pads to64
  BEFORE intra-chunk F32 matmul; current product+sum_inner128 is not a proven
  equivalent reduction. Actual MKL SGEMM_BATCH(T,N)/four-thread dispatch and
  learned inner producers identify sequential FP32 FMA as the correction.
  Shared primitive and same-iterator diagnostics integrated; complete initial
  full64 replay passes below. Cached MKL exponential remains separate/open.

- [x] Qwen actual inner-prefix causal comparison closes: convolution/activation/
  Q/K/normalizedQ/K/scaledQ/betaV all exact; first mismatch QK36/48 F32 values,
  then4549/6144 rawcore and2 BF16core words. FMA candidate using only genuinely
  computed Rust scaledQ/K/betaV matches all48 QK and6144 rawcore words exactly.
  Original116-record capture joins all96 oldprefix+66full records; ROOT reviewed
  expected-only SHA9355c45eef22726c25710faaaf897777fa6f9ab47f7dec9780262289a7873c48
  and12 actual marked MKL dispatch intervals. Actual unpublishedprefix callback1/
  commits0/rootunchanged, source31FD/ownedclosure/final0/unfenced. Report
  SHA51b5d18f6209b2f2472e034814a593cbbce8baf522e2ec79e9e1e3c0b454851f.
  Shared INITIAL1 sequential-FMA primitive now wired and30 reviewed source
  paths merged MAIN/ISO. Actual full64 replay: two independent INITIAL1
  teacher11 proposals,196 records/39,842,304float words each; all79,684,608
  words exact including fullV and mapped state. Two callbacks/commits, actual
  pre-cancel/pressure/retained-output/source31FD/owned closure/finalhost0 pass.
  Replay204551B SHA34c81e5118c8df4ad275ff0833e372cbf70eca54def149f76c14476626e1465b.
  Complete Conv4 equivalence, cached353/MKLexp, natural/native/serving/performance
  remain unqualified. MAIN gate5 PASS1645Rust/0failed/86ignored/265suites;
  later common cache/backend drafts are outside this gate.

- [x] Gemma all-call diagnostics integrated MAIN/ISO (10 paths):3965 producers/
  22598144B within unchanged caps, complete numeric scans before refusal. Fresh
  original v2 capture/independent validators PASS; ordinary fullV/KV raw and all
  1981 initial producer records equal immutable reference2. Gemma19/example5/
  Python11/strict Clippy PASS. Actual Rust calls0/1 match all1981+992 producers,
  fullV/all15KV; call2 compares all992 producers/all31 fullV+KV records. Earliest
  actual mismatch is layer9/position5 attention.pv[2421] (head4/channel373), one
  4096-element PV word:3d460000 vs3d470000, with exact observed six-term inputs.
  Callback rejects despite matching greedy886; root2/frontier5 unchanged,
  source/owned closure healthy/final0. No full three-call parity promotion.

- [x] Gemma source-shaped BF16 PV correction merged MAIN/ISO (5 paths): pinned
  GEMM-NN four FP32 partials/tail/final order, source small-BMM branch and full
  masked-prefill contraction from actual RootAttempt call geometry. Default
  analytic/QK/GEMV, source identity, caps and native eligibility unchanged.
  PV4/reference11/Gemma19/example5/strict Clippy PASS. One learned whole35 replay
  against unchanged healthy v2 matches all3965 selected producers and all93
  fullV/KV records across three natural calls886→2915→886; roots1/2/3,
  frontiers4/5/6, source unchanged/final_reserved0 and actual owned closure.
  Report SHA199ca15df9d6d02017b2fb3fff94bcaeb1bd4dbc6a669bd4f5850486bc3fee6f.
  Scope: authentic4-token prompt plus two natural cached calls only; general
  trained/native/serving/performance support remains unqualified.

- [x] Gemma selected learned lifecycle source5 paths merged MAIN/ISO; existing
  completion callback handles late refusal/cancel with no new engine hook.
  Two actual sessions, pre-cancel/layer9 observer refusal/completed refusal/
  post-comparison cancel/same-root retry/natural survivor; full producer/fullV/
  all15KV comparison and actual root/KV/reservation invariants. Gemma20/example6/
  strict Clippy/release PASS. Same128MiB+8MiB limits; one actual learned run active.

- [x] Actual Gemma learned lifecyclePASS: two natural sessions886→2915→886,
  roots1/2/3/frontiers4/5/6. Pre-cancel/actuallayer9PV refusal have0completed
  callbacks; completed refusal+latecancel each compare992producers/fullV/all15
  KV then preserve root1/allKV hashes/reserved79441036B. Retry+survivor pass;
  all8callbacks/9914producers/248completed records exact. Source/actual owned
  closure/finalreserved0 healthy; selectedCPU only, generalflagsfalse.

- [x] Assembled108-path MAIN isolated fullcheckgate4PASS:1560Rust/0failed/
  82ignored/261suites,18Python namespace checks, complete scriptClippy/release/
  installer/lint/qualification/shell/Nix. Model learned runs above are separate
  real-source checks; neither test durations nor Qwen refusal grant support.

- [x] Gemma bounded tokenizer/text preparation source merged MAIN/ISO:
  approved32,169,626B tokenizer uses BPE262144 vocab/514906 merge pairs,
  bytefallback/fuse_unk, Replace spaces→▁/Split and identity single-sequence
  postprocessor; no implicit BOS. Reuse tokenizers0.22.2 algorithms via bounded
  typed streaming construction; full generic JSON graph is not a fit proof.
  Actual original/Rust comparison passes46 text and15 byte/control decode
  cases in both special-token modes. Requested GlobalAlloc construction peak
  94099192B/retained57113812B, notRSS; original source/owner closure and final0
  pass. Construction116MiB+8MiB control plus4MiB caller fits128MiB admission.
  Source review fixes serialize allocating operations nonblockingly, retire
  provenance before admission and verify final source stamps after hashing.
  Tokenize→drop leaves only64KiB IDs/provenance before model load. Focused
  Gemma24/featureexample7/legacyexample6/strictClippy/Python6 PASS. Opt-in
  --plain-text wiring uses actual tokenizer IDs and the same learned35-block
  engine. One actual replay PASS: all3965 producers and93 fullV/KV records exact,
  natural886→2915→886, roots1/2/3/frontiers4/5/6, unchanged source/final0 and
  heldbinary/process/group/reap/EOF closure. Report SHA
  716ee16ec29e1b80855055c873254f2421e57029a1766bb9bbf4abd569998354.
  Final assembled gate pending; default CPU gate now includes optionaltokenizer
  feature and six source-only Python contract tests.
  No chat/multimodal/generaltrained/native/serving/performance promotion.

- [x] Gemma ordinary plainCPU31-path candidate integrated MAIN/ISO: managed
  held SnapshotLease/source windows, bounded selected-ID decoder reusing source
  decoder algorithms without retained BPE graph, private same-session greedy
  feedback/EOS[1,106,50], explicit CLI profile. Derive whole128MiB coexistence;
  whole derived108773100B fits128MiB; bounded decoder61 actual cases in both
  special-token modes exact, original source/final0/owned closure healthy.
  Full assembled gate6b PASS1713Rust/0failed/88ignored/266suites plus scripted
  checks, including candidate CLI feature. This is source/decoder acceptance;
  broader ordinary runtime qualification remains open below.

- [x] Complete frozen actual seven-cell Gemma ordinary matrix: genuine plain
  M1/2/3/4/32/57 plus UnicodeM3, natural extent64/fullV/all15KV/consumedpriors,
  source/cancel/refusal/retry/sibling checks. Original captures and lossless
  BF16 archives within unchanged caps. Actual Rust PASS50natural committed
  calls +2completed rejected proposals,52 fullV/all15KV comparisons and10710
  actual35-block row-boundary events,0mismatchedwords. Pre-cancel/refusal/late-
  cancel/retry/independent survivor/pressure pass. Sourceafter true, prepared
  3670016B→caller3145728B→final0. Report255906B SHA
  cef7dd0dddc18710caff6e6a23c6b8046c401507e635b8acd68a8745b62ace1e;
  ROOT independently checked numerical counts/frontier64/report/source/owned
  closure. Frozen binary unchanged/FDclosed, groupabsent/reaped/EOF/cleanup[];
  same original7200s deadline including bounded855.9s pause, no recapture or
  expected operands. Finite ordinary CPU candidate scope only; generic trained,
  chat/native/HTTP/performance flags remain false.

- [x] CLI dispatch successor integrated MAIN/ISO: option-position scan preserves
  existing Qwen --prompt '--source-profile' and every existing value flag/alias,
  while selecting the explicit Gemma profile. Eight matching-feature CLI tests
  and strict Clippy PASS; both original parsers and frozen matrix unchanged.
  Assembled isolated gate7 PASS1715Rust/0failed/88ignored/266suites plus scripted
  checks; actual Gemma matrix retains its immutable earlier binary scope.

- [x] Qwen27B CPU source-window output tail integrated into ROOT checkout: final FFN norm → gate/up → staged BF16 SiLU/multiply → down → residual → final norm → full vocabulary, through existing MathematicalBackend/HostMath. Fourteen owned paths merged with fresh mtimes after exact parent checks; integrated model67/core17 tests PASS. Explicit post-attention input remains the boundary; no trained whole27B/112-domain/serving parity claim. Scope and checks.

- [x] Complete isolated CPU integration gate now covers Qwen27B + Gemma: scripts/check.sh PASS,1490 Rust test passes/0failed/80ignored across257 suites,18Python checks, strictClippy/release/installer/shell/Nix PASS,0warnings. Full trained inference/native/serving/performance qualifications remain open.

- [x] Gemma E2B source-bound AnalyticF32TextV1 CPU reference integrated: one generic full35-layer schedule,15KV owners/shared consumers,PLE/proportional rotary/sliding attention/finalnorm/tied full vocabulary and original SessionAuthority transactions. Eighteen owned paths contextually merged with fresh mtimes, preserving27B. Tiny genuine source-window fixtures/nonzero synthetic continuation/rollback and complete CPU integration gate PASS. Scope and checks.

### fff1842 — trained BF16 operator qualification

- [x] **Native CUDA CPU-profile BF16 projection:** both original Qwen3.5
  sources, layer3 q/k/v windows and teachers11/353, B1/independent B1/B2/4/8:
  163840 raw F32 accumulation words match the accepted CPU64 operator and
  163840 rounded BF16 words match authentic public archives exactly.
  Same sealed full-source windows, original CUDA queue/service/leases/Ready;
  pressure, pre-cancel, source/axis refusals, final physical0 and fresh actual
  abandoned-flight source/completion/fence retention pass. Synthetic328
  case/band pairs cover source widths64..8192, signed zero, subnormals, RNE,
  FMA/reduction order and canaries; eight fresh FP32/BF16 overflow flights
  withhold Ready and retain original quarantine. Strict Clippy/five focused
  tests and independent trained/synthetic raw-byte validators pass. This is
  the explicit CPU-target profile, not vLLM CUDA arithmetic, full HF/GDN,
  model StateRoot or a throughput claim. Original failed build/import checks
  and their additive harness corrections remain preserved.
- [x] **Original GDN convolution/SiLU fragment:** both authentic HF sources,
  layer0 teachers11/353, initial padding and cached-tail inputs: all122880
  BF16 words across raw convolution, activated output and canonical next3
  history match independent public Torch operators exactly. Public outputs
  first reproduce original activated/K4 archives. Same ForwardPlan/HostMath,
  sealed raw BF16[C,1,4] SourceParameter and externally authenticated inputs;
  actual pressure/pre-cancel/wrong-role, retained output after callers drop,
  full parameter stage accounting and final physical0 pass. Core10/artifact2/
  model48 tests and strict CPU/CUDA/Vulkan checks pass; trained replay has
  independent all12-file signed-bit validation. Standalone JSON-macro syntax
  failure is preserved; fresh ROOTV7 changes only harness parentheses.
  Full HF/session/checkpoint guards remain closed. Initial FLA chunk64,
  cached recurrent state, coefficients/gated norm and full logits/StateRoot
  remain required before enabling original2B/27B inference.
- [x] **Composed BF16/conv phase:** complete isolated1251 Rust tests,
  zero failures/70 ignored/248 suites,18 Python tests, release/installer/Nix
  checks pass. Composed strict CUDA/qualifier/Vulkan and CPU-free binding8/
  GPU cold-refusal2 tests pass. Both approved original release cache-direct
  and HF-import routes reproduce full248320/all42 for both models and retain
  original source/lease/mutation/reclamation guarantees.

### fe2a342 — original attention and reference operators

- [x] **Original CPU attention fragment:** both authenticated sources at
  layer3, teachers11/353, source-36-F32-restored profile: all47104 raw BF16
  words across seven projection/view/QK-norm outputs match authentic original
  captures, including signed zero. Same HostMath/ForwardPlan and sealed source
  windows; real source-pool pressure, pre/mid-cancel, foreign input/provider,
  retained output after callers drop and final reservations0 pass. Explicit
  external hidden inputs establish no preceding producer or StateRoot.
  Focused16 tests, strict CPU/CUDA/Vulkan, both approved original release
  routes and complete isolated1243-test phase pass. Fresh helper strict build,
  full source scans and owned process/group/EOF cleanup also pass.
  Earlier metadata-list and omitted second4MiB binding-accounting failures
  remain preserved; corrections change only the replay harness. RoPE, KV,
  attention BMM/softmax, GDN/FLA, full-HF logits/StateRoot and27B stay open.

- [x] **Authentic vLLM projection operands:** read-only extraction from the
  accepted four-call eager-B1 capture preserves all original full496640-logit,
  320-source-audit, provider and cleanup guards. Six raw BF16 files,36992 bytes,
  bind actual teacher11/353 inputs/results, public GEMM pointers/settings and
  learned QKVZ(qkv,z)/BA(b,a) row hashes. Ten transport tests and independent
  rederivation pass; result f0981e62. No new model/GPU execution or dtype cast.
  Native BF16 arithmetic, opaque provider allocation admission and full-HF
  execution remain unqualified.

- [x] **Original2B paired attention observation:** fresh plain/observed CPU
  captures preserve all196 raw tensors/four full248320 logits/168 canonical
  domains/60 prior intermediates against the accepted two-profile reference.
  Observed capture adds216 genuine BF16 tensors across six attention layers,
  nine stages and teachers11/353 in both original loader profiles,688128 bytes.
  Exact signed-bit/shape/stride/source checks, unchanged archive validator,
  all632 source/617 loaded parameter audits and owned group/container/EOF
  cleanup pass. Ten parser fixtures pass. Diagnostic elapsed44.56s/42.01s;
  no benchmark or physical RAM admission claim. Result e45e1903 qualifies
  transparent reference-stage availability, not Rust attention output,
  full HF execution/StateRoot, GPU/FLA, long context or trained27B.

- [x] **Genuine pinned vLLM V1 eager B1 public GEMM observation:** four
  unchanged original0.8B layer0 QKVZ/BA projections at teachers11/353 use
  cublasGemmEx, BF16 A/B/C, FP32 compute, requested DEFAULT_TENSOR_OP,
  default math/host pointers/atomics disallowed, alpha1/beta0 and explicit
  8519680-byte workspace. Actual full tensor/workspace allocation extents,
  original provider versions/mapped SHA and loaded320 source roles before/
  after pass; plain/observed all496640 logit words are exact. Strict observer
  build and all three owned process/container/group/EOF cleanups pass.
  Original header-build and generic resolver failures remain preserved;
  additive exact already-loaded provider binding has no alternative lookup
  or math fallback. Prior resolver failure cause remains unproven.
  This records public requests, not hidden selected kernels, all42 state,
  wider/default graphs, Zerogravity GPU math/profile or matched performance.
  Qualifier result55d68c7a; explicit workspace alone does not qualify opaque
  library allocation admission. Pinned default vLLM loader/quant registries
  have no GGUF entry; original BF16 remains the matched-source route.

### bfe62df — leased source-window execution

- [x] **Shared source-window HostMath fragment:** original27B851-role
  namespace and same PlanBuilder/HostMath embedding→ordinary RMS→full-column
  projection now execute genuine trained bytes without source materialization.
  Both tokens11/353 match all40,960 independent BF16-widened reference words,
  including full5120-wide embedding/norm and10240-row QKV. All18 shards/
 55,563,006,776B are verified through retained FDs. All80 authentic128-row
  windows preserve5120 input columns; maximum real QKV staging charge8,388,656B
  and raw norm staging4,194,328B are measured, not inferred from requested bytes.
  Each output retains admitted64KiB Output+960KiB Scratch and source issuer
  through last drop. Real stage-triggered mid-cancel, pre-cancel, pool pressure,
  foreign-source/config/index/token refusal and retained output comparisons pass;
  final reservations0/unfenced. Actual92.74s includes90.79s source scan;
 18,736KiB observed RSS is not physical free-RAM admission evidence.
  Fresh MAIN-linked original0.8B three-operator regression also matches all
 16,384 words/final source0. Focused17 artifact/1CPU/5model, strict CPU/CUDA/
  Vulkan/qualifier Clippy and full isolated1243 Rust/0failed/70ignored/248suites
  plus18Python/release/installer/Nix pass. Both approved original cache-direct
 17.22s/HF-import22.74s release routes pass fullV/all42/reclamation for both
  models. Only four pointless borrowed-value test drops needed repair. This is three pure state-free operators,
  frontier delta0: full112-domain27B logits/StateRoot and streamed inference
  remain unqualified.


### 5350a4d — typed ordinary math and source windows

- [x] **Typed ordinary RMS in the shared schedule:** the same PlanBuilder/
  HostMath embedding→raw InputNorm(0)→GdnQkv(0) fragment preserves source
  parameter identity, BF16 precision, epsilon1e-6 and explicit packing.
  Fresh MAIN replay matches all16384 genuine original0.8B reference words,
  two tokens11/353, with final source reservation0. Unaware providers and
  GPU plans refuse before callbacks/source admission; full HF session,
  forward and checkpoint guards remain closed.
- [x] **Sealed rank1 parameter windows:** BF16/F16/F32 raw ranges retain
  original full descriptors, issuer, opened source and actual staging charge;
  source/cancel/corruption/capacity/drop tests pass. Partial ranges never
  become complete TensorSource views. This is transport, not27B inference.
- [x] **Closed original BF16 row-window projection:** authentic original
  full-matrix descriptors and complete input columns reuse the accepted
  CPU64 dot without changing its arithmetic. Output-row partition, width5120,
  foreign source/range/dtype/nonfinite/cancel/provider and lease tests pass.
  No partial matrix/provider substitution or final BF16 cast; caller must
  admit private output/scratch. Same-HostMath27B integration remains open.
- [x] **Closed pinned F32 softmax:** source Vec8 maximum/sum order, unchanged
  SLEEF exponential and F32 reciprocal/multiply, width1..512. Genuine public
  full3599-case replay matches1838736 output words and3 refusals; authenticated
  default72-case golden preserves11280 expected words/3 refusals. NaN/+Inf/
  all-masked rows refuse. No legacy GGUF or full HF attention enablement.
- [x] **Composed acceptance:** focused core9/artifact16/CPU47/model6 pass,
  compact default5 pass; full isolated gate1241 Rust/0failed/70ignored/248
  suite runs plus18Python/release/installer/Nix passes. Strict composed CUDA+
  HF+CPUcache, Vulkan serving and CUDA qualifier pass; CPU-free HF binding8
  and GPU cold refusal2 pass. Both original cache-direct/HF-import RELEASE
  routes pass(18.88s/24.75s), full248320/all42 and final source reclamation.
  Full27B Rust reference comparison, streamed model state and CUDA/FLA remain
  open. Compilation/type and Clippy-only first failures were fixed; no
  mathematical tolerance or full-profile guard changed.


### c5eb1f8 — closed pinned sigmoid and ordinary RMS

- [x] **Closed pinned BF16 sigmoid:** explicit contiguous iterator spans,
  vector32 prefix plus the original scalar tail, private SLEEF and licensed
  Arm/glibc-FMA arithmetic. Genuine public Torch and Rust replay match all
  326753 output words, including all65280 finite BF16 values in vector/scalar/
  mixed call shapes and both actual original beta16 calls. No host exp,
  fitted output table, numerical-profile enablement or GGUF math change.
- [x] **Closed pinned ordinary RMS:** raw immutable rank1 BF16/F32 learned
  weights, exact BF16 input, source four-Vec8 reduction/carry order at widths
  256/1024/2048/5120, F32 division and runtime(1+raw weight), then final BF16
  cast after the F32 product. Actual original .8B122 norm calls preserve98
  prior full-logit/cache checks;32 public width/dtype fixtures cover the
  larger geometries. Rust matches all154 cases/1694 full stage fields/
  1680488 words exactly. Width fixtures do not qualify2B/27B inference.
- [x] **Composed source-free gate:** complete isolated check1218 Rust passed,
  0failed/69ignored/248 suite runs plus18Python, release/installer/Nix;
  ordinary RMS7 focused tests and strict CPU Clippy pass. Independent source
  review authenticates every6792 compact golden word from genuine public
  archives. Strict final composed CUDA+HF, Vulkan serving and CUDA qualifier
  Clippy pass; both approved cache-direct/HF-import release routes pass
  (20.09s/24.88s), both original models, full248320/all42, independent sessions,
  source leases and reclamation. Original-HF full session/forward/checkpoint
  guards remain closed; coefficient exp, initial chunk64, attention and
  physical model/state integration are still required.

### 2a32aff — typed source composition and vector softplus

- [x] **Typed original-source operator integration:** explicit program
  profile and activation precision, BF16 OutputInput embedding/projection
  contracts and raw SourceParameter metadata use the existing PlanBuilder
  and HostMath. Private operator slices carry no domains, logits receipt
  or frontier advance. Unknown physical providers and resident GPU plans
  refuse before source reads/admission. Raw learned parameters stay borrowed
  from the same owner; existing RootBinding descriptors distinguish initial/
  cached phase, BF16 conv/KV and FP32 recurrent axis layouts. These metadata
  views grant no freshness, state transport or publication capability.
  Complete original-HF session/forward/checkpoint guards remain closed.
- [x] **Genuine typed operator qualification:** pinned public Torch
  F.embedding for original tokens11/353 followed directly by independent
  GdnQkv(0), FfnGate(0) and tied Output projections matches every Rust bit,
  including all248320 words of both output projections. Eight reference
  payloads preserve518144 values; same loaded source, raw matrix pointers,
  two-node recorder, zero initialized state and final source reservation0
  are checked. No input normalization, complete layer/model logits or
  GPU/27B inference is claimed by this pure operator composition.
- [x] **Closed pinned FP32 vector softplus:** source beta1/threshold20,
  finite general-F32 contiguous spans divisible16, explicit SLEEF3.8
  compensated log1p/exp/FMA order and BSL notice. Actual32 original biased
  alphas and28464 adversarial F32 values, reordered values and16/32/48/64
  spans match public Torch bit for bit. No scalar-tail, MKL coefficient
  exp, general profile or default GGUF arithmetic change.
- [x] **Typed/softplus composed gate:** full isolated check1213 Rust
  passed/0failed/69ignored/248 suite runs plus18Python, installer/release/Nix;
  focused core/CPU/model158 passed/2ignored, CPU-free binding10 passed,
  resident cold refusal2 passed. Strict composed CUDA+HF, CPU-free HF,
  Vulkan serving and CUDA qualification Clippy pass. Both approved original
  cache-direct/HF-import release routes pass (17.15s/23.12s), both original
  models, full248320/all42, independent sessions/source lifetime/reclaim.

### cadcbd9 — closed pinned original-HF CPU operators

- [x] **Closed pinned CPU original-HF operators:** source-bound BF16
  four-tap convolution, FP32 width128 cached GDN, raw-F32-gamma gated RMS
  prefix and contiguous vector SiLU. No model/session/initial-chunk permit;
  default GGUF/scalar profiles stay unchanged. Convolution preserves four
  independent FMA partials followed by direct adds and BF16 output. Cached
  GDN uses direct query division by sqrt128, actual key-major FP32 state,
  materialized key reductions and separate update multiply/add; observed
  decay is an explicit input. RMS casts normalized core before raw original
  F32 gamma, retaining the FP32 product. SiLU ports pinned SLEEF3.8 AVX2
  exp/FMA/scaling and FP32 division with source attribution/license, only
  finite exactly BF16-widened contiguous inputs with length divisible16.
  Twenty-eight focused Rust tests pass including old projection/reducers.
- [x] **Actual closed-operator Rust qualification:** untouched pinned
  Torch2.13 public SiLU with observed SLEEF AVX2 provider supplies exact
  outputs for ALL65280 finite BF16 inputs plus4096 actual source gates;
  Rust matches every F32 bit. Full16-head cached replay matches262144
  state words plus5*2048 normalized/scaled/core/output words. Gated RMS
  prefix matches8256 words across both actual tokens/all16heads.
  Convolution and composed vector-SiLU/BF16 match all6144 channels at
  four actual causal positions (49152 output words). These are pure
  operator replays of genuine source operands, not full-model/profile,
  GPU/FLA or27B numerical acceptance. Initial chunk64, coefficient exp/
  sigmoid/softplus, ordinary RMS and eager attention remain open.
- [x] **Composed verification:** complete isolated check1202 Rust passed/
  0failed/69ignored/248 suite runs plus18Python/release/installer/Nix.
  Strict CUDA+HF-binding+portable, Vulkan serving and CUDA-qualification
  Clippy passed; CPU-free original binding6 and CPU serving22 passed.
  Both approved original0.8B/2B cache-direct/HF-import RELEASE routes
  retain all248320 logits/all42 domains, source lifetime and reclamation.
  Independent long-context GGUF GDN divergence remains a separate required
  gate in [numerics-and-context](../numerics-and-context/TODO.md).

### 9815504 — original HF roles and closed CPU projection/reducers

- [x] **Original HF source roles and closed CPU profiles:** authentic rank/axes,
  raw source dtype and tied/untied heads bind to the same loaded owner;
  Safetensors output/input rows remain untransposed. Typed original-HF
  metadata carries the actual config/index/bundle and independent state
  geometry. Its execution/session/checkpoint-v1 entry points refuse before
  allocation/backend/I/O; metadata is not an executable-model promotion.
  Explicit Torch CPU AVX2 BF16 projection diagnostics retain the pinned
  64-lane FMA/reduction order and separate state/trace schema; the original
  scalar/GGUF profiles remain distinct. Actual original .8B layer0 QKV
  is exact6144/6144 for each of tokens11,353. Closed FP32 width128 L2 and
  key-axis reducers pass authentic compact bitwise goldens and adversarial
  head/order/finite/overflow cases. Real attention-input norm observations
  now cover both full and GDN layers. Focused composed suites185/0/2ignored,
  CPU-free binding8/0, strict CPU/CUDA/Vulkan/qualifier Clippy and full
  isolated check1164/0/69ignored/245suites+18Python passed. Both approved
  original .8B/2B cache-direct/HF-import release routes remain exact full
  logits/all42. Full original HF mathematics remains UNQUALIFIED.


### f26232e — authentic tied original output heads

- [x] **Original HF tied output head:** verified `tie_word_embeddings`
  selects the original embedding or independent `lm_head` projection;
  no alias tensor or fallback source. Source-free authenticated tied/untied
  fixtures test opposite-sign independent heads, exact F32/BF16 diagnostic
  logits, immutable source identity and final lease0. Full isolated gate:
  1148 Rust passed/0 failed/69 ignored/245 suites,18 Python. Both approved
  original0.8B/2B cache-direct/HF-import release routes retain exact full
  logits/all42. This fixes the diagnostic head; no trained HF promotion.

### 1c0bb41 — pinned numerical reference seam

- [x] **ME-P0 pinned original numerical reference gate:**
  extend one immutable source owner with canonical source-bundle
  SHA256 over pinned index and all authenticated shard bytes.
  Capture a bounded subset of original Qwen GDN, attention, FFN
  and logits through shared PlanObservation, with opt-in 32MiB
  private trace and a single selected GDN recurrent state layer.
  Strict typed independently SHA256-pinned JSON comparisons bind
  config/index/bundle, exact stage order/geometry/position/dtype,
  detect first mismatched element, and distinguish truly bit-exact
  BF16 output from bounded, tolerated FP32 state deviations.
  Source-free adversarial tests verify same-index/different-weight
  digest changes, duplicate/missing/reordered reference content,
  bad token/source/dtype, selected GDN state and immutable roots.
  Fix full-model BF16 residual result-cast omission by reusing
  its source-typed existing residual binary operator.
  This stage DOES NOT prove independence merely because a
  reference JSON asserts trained-framework origin.
  Complete isolated scripts/check.sh passed 1,135 Rust tests, 0 failures,
  69 intentionally ignored across 238 suites plus 18 Python fixtures,
  release CLI/installer and CPU numeric comparators. Strict default and
  composed CPU/cache/Qwen/acquisition Clippy -D warnings pass.
  Independent approved Qwen3.5 .8B/2B release cache/direct 17.45s and
  HF-import 21.97s preserve exact full 248320 logits, all 42 StateRoot
  domains, two-token independent sessions and source/lease correctness.
  Offline Nix source-integrity passed with the checked trace files.
  No trained Qwen3.8 reference or Torch/FLA runtime was accessed.

### 6d74f76

- [x] Qwen3.8 27B and Gemma4 E2B/26B configuration, tensor and source-index contracts; no new trained execution, tokenizer or checkpoint loading.

### Source contracts — 023703f

- [x] Qwen3.8 metadata/shard roles and CPU-free Gemma4 family contracts,14 Gemma tests and official metadata/config cases passed within metadata scope. No tokenizer, new learned weights, physical shard-header binding or trained inference follows.
- [x] Preserve Qwen/Gemma source axes, metadata precision and routing/KV facts in the [model semantics reference](../../docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md) and [Gemma technical contract](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md). The executable-source4GiB cap remains.

### 93a5622 — original verified immutable source shard owner

- [x] **ME-P0 LS-B01–B05, verified owned sharded-source allocation:**
  implement `SourceWeights::prepare_verified_shards` in the existing
  artifacts owner with exactly opened original Safetensors file
  descriptors, caller-expected index SHA256 and whole-shard SHA256/length,
  one original SafetensorsIndex parser and checked header name→shard,
  dtype/shape/offset/payload binding. Preflight limits and reserve whole
  captured file allocation plus a conservative metadata envelope via the
  original caller-attested host `ResourceArbiter` before copying. Reparse
  exact owned bytes, compare authenticated headers against preflight
  (including same-length mutation) before issuing the single original
  SourceIdentity. Arc owner lifetime pins all original slices and RAM
  charge; default GGUF `SourceWeights::open` keeps the 4GiB cap and
  original Qwen execution path.
  10 source-only integration tests pass: positive F32/F16/BF16, complete
  index closure, foreign tensor metadata rejection, wrong SHA/length,
  missing/extra/duplicate/wrong shard, same-descriptor pathname
  replacement, original inode mutation, verified copied immutability,
  multiple admitted/shared original leases and exact resource rollback.
  Sparse >5GiB header tests refuse before reads under both 4GiB
  default and physically insufficient explicit >4GiB budgets.
  **No actual >4GiB weight load, SSD streaming, trained original
  Qwen27B/Gemma math, tokenizer or physical GPU support is claimed.**

### b796a2e — original Qwen text source geometry and CPU value binding

- [x] **ME-P0 original Qwen3.8 source/value binding (small original
  format, not trained 27B):** bind caller-authenticated original
  QwenTextConfig SHA256 and pinned indexed-source SHA256 on the *same*
  immutable SourceWeights allocation. Check every ordinary-text
  name and original [output,input] tensor shape/encoding, including
  the untied output head; distinguish source extra vision/MTP from
  unregistered text roles. Reuse the existing CPU source MatVecRegion
  for HF packed F32/F16/BF16 output-first matrices with no transpose,
  plus the existing embedding/parameter-vector decoder through a
  source-bound QwenOriginalCpu façade. Small complete 4-layer hybrid
  original-format integration tests and negative hash/shape/missing-
  head/source-provider tests pass. Oversized HF source matrices
  refuse before the optional accelerator is invoked (observed no-dispatch).
  The complete frozen isolated
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passes 1,103 Rust tests, 0 failures, 69 ignored across 238 suites,
  plus 18 Python namespace fixtures, strict lint, installer/Nix
  formatting and CPU numeric selftests.
  Both independent Qwen3.5 0.8B/2B original trained
  source-parity release routes pass AFTER the final safety guard:
  cache/direct 16.70s and HF-import 22.88s, each full 248320-vocabulary
  logits, all 42 StateRoot domains, two tokens, independent sessions
  and lease/mutation/reclamation exact. Default and
  acquisition/cache-CPU strict `cargo clippy --locked --offline
  --all-targets -- -D warnings` checks also pass. The original
  GPT-family GGUF CPU and session path is unchanged.
  Does NOT run trained 27B, create a Qwen session, implement
  grouped GDN permutations, or stream SSD.

### b11e58c — original source-bound Qwen analytical one-layer proposals

- [x] **ME-P0 original Qwen3.8 analytic one-layer operators:**
  source-bound ordinary zero-centered RMS gamma (1+weight),
  HF head-interleaved full Q/gate projection, per-head Q/K norm,
  source-configured partial RoPE, 24Q/4KV grouped lookup,
  QK/softmax/AV/sigmoid/out projection and candidate KV slabs under a
  source-and-layer-bound immutable `OriginalFullPrefix`. Source GDN
  four-tap causal conv and SiLU, exact 16→48 repeat_interleave of
  original Q/K, per-head L2, stored A_log→-exp(A_log),
  dt_bias/softplus/sigmoid beta through the existing coefficients and
  recurrent step, output GDN gated norm/out projection. An immutable
  `OriginalGdnPrefix` holds exact source/layer and private tentative
  conv/recurrent state, never replacing SessionAuthority. Source FFN
  post-attention RMSNorm, SwiGLU and residual reuse old math.
  Small independently derived analytical checks include zero-centered
  norms, 16→48 head signed-zero mapping, 24Q/4KV GQA six-query
  grouping, ln2 decay beta0.5/gain0.5, full attention gates/partial
  RoPE, GDN two-token private repeat/branch, source identity mismatch,
  finite and dimensional errors. The original 8K diagnostic bound and
  original default GGUF behavior remain. **No trained HF BF16 intermediate
  rounding, full layer/model StateRoot, tokenization, actual 27B learned
  weights or GPU/SSD support claimed.** Strict source-config-specific
  numerical reference and official checkpoint tests remain open.
  Final isolated `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4
  python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passed 1,111 Rust test executions, 0 failures, 69 ignored across
  238 suites, plus 18 Python namespace fixtures, release CLI/installer
  and CPU numeric-comparator scripts; strict default and composed
  CPU/cache/Qwen/acquisition Clippy `-D warnings` passed.
  Actual offline Nix source-integrity passed. Two independently run
  Qwen3.5 .8B and 2B trained-source release routes remain exact:
  cache/direct 17.36s and HF-import 22.49s, each original complete
  248320 logits/42 StateRoot domains/two tokens/source leases.
  This diagnostic-only milestone is recorded in signed commit b11e58c.

### 6329cff — source-bound Qwen one-layer TaskCoordinator

- [x] **ME-P0 source-bound original Qwen StateRoot admission
  (one-layer diagnostic only):** the sole existing TaskCoordinator
  now supports an opt-in original immutable source identity plus
  a unique coordinator/session ID while retaining its generic
  publication API. A stateless source-bound one-layer bridge maps
  canonical ForwardPlan Conv/Recurrent/Kv StateSlots to versioned
  GDN or full-attention StateRoot domain bytes, with exact model,
  epoch/session, layer, root, schema and frontier verification.
  Immutable source-pinned numerical proposals publish via the
  existing TaskCoordinator only after complete finite-domain
  preparation, 8K token/64MiB diagnostic caps and early next-KV
  admission. Independent tests cover GDN two-domain and causal
  attention KV commit, abandoned and stale competing steps,
  previous immutable roots, corrupted/nonfinite state refusal,
  source/session/schema mismatch and generic coordinator parity.
  There is NO separate multi-layer token loop, model scheduler or
  new state issuer; this is not yet trained BF16 reference parity.
  Final isolated full scripts/check.sh passed 1,116 Rust tests,
  0 failed, 69 intentionally ignored across 238 suites plus
  18 Python namespace fixtures, release CLI, installer, shell/Nix
  formatting and CPU comparator selftests. Strict default and
  composed CPU/cache/Qwen/acquisition Clippy -D warnings passed;
  independently repeated Qwen3.5 .8B/2B release cache/direct
  17.11s and HF-import 22.27s retained exact 248320 logits,
  42 StateRoot domains, two tokens and source lease behavior.
  Offline Nix source-integrity passed before signing.

### e43ba4c — original Qwen full-model FP32 analytical forward

- [x] **ME-P0 full original-HF model analytical forward:** extract
  the one Qwen layer traversal from the existing GGUF forward_token,
  preserving its complete MathematicalBackend operator order.
  Use it also for original source-bound GDN/full-attention/FFN,
  each layer's private Conv/Recurrent/Kv state and residual output,
  and final RMSNorm/untied head. One opt-in TaskCoordinator
  owns **all** model state domains, not one root per layer.
  Validate all source identities/schema/finite bytes, 8K/128-layer/
  128MiB diagnostic bounds and complete logits before atomic
  publication. Tests: complete synthetic four-layer original
  checkpoint with seven state domains/two tokens, matching
  independent one-layer GDN, abandoned/competing/foreign session
  and late-corrupted state refusal, derived final logit ~8, and
  shared GGUF traversal ordering/fail-fast. The method is explicitly
  named `prepare_diagnostic_token_f32`: this is **not** trained BF16/
  FLA inference, production TaskDriver or an alternate runtime.
  Complete isolated scripts/check.sh passed 1,119 Rust test
  executions, 0 failed, 69 intentionally ignored across 238
  suites, plus 18 Python namespace fixtures, release installer/CLI,
  Nix/shell formatting and numeric comparator selftests.
  Strict default and composed CPU/cache/Qwen/acquisition Clippy
  -D warnings passed. Both independent original Qwen3.5 .8B/2B
  trained-source release routes preserve exact 248320 full logits,
  42 StateRoot domains, two tokens and source/session leases:
  cache/direct 17.88s and HF-import 22.59s, both exit0.
  Offline Nix source-integrity passed before the signed commit.

### aaf9ba6 — original Qwen BF16 source dtype diagnostics

- [x] **ME-P0 explicit original BF16 dtype diagnostic:**
  implement finite nearest-even BF16 rounding with signed-zero and
  overflow refusal; introduce optional BF16 source stages into the
  existing original Qwen source binder/CPU operators, without
  mutating the default FP32 path. Cast after ordinary FP32
  (1+gamma) RMSNorm, per-head Q/K norm, HF output-first
  projection, SwiGLU, residual, RoPE, BF16 QK matmul *and*
  scaling, post-FP32 softmax, attention value and gates.
  GDN follows BF16 conv/SILU, FP32 A_log/softplus, BF16 beta,
  FP32 recurrent state and BF16 core output, source gated RMS
  casts before gamma and at gated output. StateRoot schema
  variants disallow cross-dtype replay even for one allocation.
  Two-token four-layer synthetic BF16 whole-model integration
  plus source-independent NumPy BF16 bit-golden RMSNorm and
  gated RMSNorm, FLA two-step recurrent output/FP32 state
  oracle, sign/tie/overflows and F32 parity are required.
  This is explicit BF16 **diagnostics** only: NOT trained
  reference equivalence, a new model-run route or GPU/SSD.
  Source-independent NumPy BF16 goldens match exact bit patterns
  for ordinary/gated RMS and two-step GDN returned cores, with
  FP32 recurrent state tested within 2e-7. The full isolated
  scripts/check.sh gate passed 1,128 Rust test executions with
  0 failures and 69 intentional ignores across 238 suite runs,
  plus 18 Python fixtures, release installer/CLI and CPU numeric
  comparators. Rerunning on the final implementation still
  passes 1,128/0 across 238 suites. Both independent Qwen3.5
  .8B and 2B trained release routes retain complete 248320
  logits, all 42 StateRoot domains and source lease/mutation:
  direct16.26s and HF-import21.78s on the final source.
  Strict default/composed Clippy and Nix source-integrity pass;
  production model run is not enabled.

## Deferred — after Qwen/Gemma and Flash execution

- [ ] Qualify independently selected NanoJev-like calibrated decision
  model sources under the same model/source/cache/residency authority.
  Register finite choice/boolean/score or include/omit heads without
  assuming a specific research checkpoint. Validate calibrated
  probabilities and per-question source-bound state, batching, thresholds,
  accepted/rejected context records and decision-fallback semantics
  before enabling generic context-filter execution; pipeline check alone
  provides only typed static intent.

### 28 Oct 2026 — ordinary Qwen CUDA integrated, qualification-only

- [x] Source-derived ordinary .8B/2B CUDA execution is assembled under the existing ResidentProgram / ResidentInference owner: one call-free ForwardPlan, request-local materialization, owned cuBLAS projections, source-exact attention, arbitrary-row GDN, F32/F32/BF16 two-root state, one queue/flight/completion/quarantine path and the existing StateRoot publication authority. Prompt IDs, lengths, ordinals, packets, captures and recipe hashes do not authorize ordinary execution.
- [x] Source-free lifecycle/build gate passes: qualification ordinary tests, strict qwen35-cuda-qualification Clippy, plain qwen35-cuda, default and all-features checks. Explicit shutdown now requires unique model ownership, retires the BLAS provider/pool before lease release, verifies zero resident bytes/events, then closes the model namespace; source-free close-order tests cover managed allocations, opaque foreign charge and in-flight event retirement.
- [ ] Real current-source qualification remains mandatory before production routing: .8B natural/current-root parity including cached 47→50, physical fault/cancellation/retirement and real-model shutdown, then .8B 512/1K/2K/4K/8K and 16K/32K. Admit 2B contexts only where the exact two-root plan fits.
- [ ] Current physical blocker: the approved .8B source is now visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`; its 1,746,942,600-byte safetensor SHA256 is exactly `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The transient `.agents/runs` native bundle was intentionally purged and no `cuda-kernels.json`/legacy `short-hf-native.json` bundle remains under the repo or user cache. Reacquire/regenerate the qualified native/reference bundle through the maintained capture path; do not restore deleted run directories or substitute another model source.

### 28 Oct 2026 — exact ordinary CUDA context admission matrix

- [x] Diagnostic source-semantic Qwen plans now feed the same production
  OrdinaryCudaAdmission analyzer, with no source IO, GPU allocation or execution
  authority. Under the unchanged 4608 MiB provider allowance inside the 5120
  MiB physical envelope, exact cold + two-root totals are:
  - .8B: 512=1,652,916,305 B; 1K=1,667,661,905; 2K=1,697,153,105;
    4K=1,756,135,505; 8K=1,874,100,305; 16K=2,110,029,905;
    32K=2,581,889,105. All fit admission.
  - 2B: 512=3,911,845,969 B; 1K=3,926,591,569; 2K=3,956,082,769;
    4K=4,015,065,169; 8K=4,133,029,969; 16K=4,368,959,569;
    32K=4,840,818,769. 512 through 16K fit; 32K exceeds 4608 MiB
    (4,831,838,208 B) by 8,980,561 B and must refuse before allocation.
- [ ] These are fit/admission results only. Physical correctness, continuation,
  replay, faults and state-history qualification remain required for each cell
  before production use; .8B real-model qualification is blocked on a qualified
  native/reference bundle, not on model-source visibility.

### 28 Oct 2026 — ordinary CUDA source-free cancellation/fault ownership

- [x] Ordinary request cancellation is explicit at the resident owner: checks run
  before candidate-root allocation, before flight arming, between source nodes,
  after launch/completion, and immediately before commit. Cancellation on an
  armed flight returns through the existing ResidentSubmission Drop path, so
  completion becomes unknown, the service is fenced and original access/native
  owners are retained for quarantine; no alternate cancellation authority or
  root publication path exists.
- [x] Source-free lifecycle regression set passes: ordinary cancellation helper,
  independent model close/reload with global IDs/events, delayed event reader
  source/quota retention until actual retirement, unknown module release
  sibling fencing, explicit ordinary provider shutdown, strict cuda-cublas and
  qwen35-cuda-qualification Clippy, and all-features build.
- [ ] Real physical after-launch cancellation/fault and real-model shutdown
  remain required with the approved .8B source; source-free ownership success
  does not substitute for those device tests.

### 28 Oct 2026 — typed committed-root snapshot and canonical KV tail

- [x] Ordinary KV candidate planes are now fully defined inside the same model
  flight: each BF16 candidate key/value plane is zeroed, the committed prefix is
  copied exactly, then new rows are appended. Future KV tail therefore remains
  exact +0 instead of inheriting uninitialized allocation bytes.
- [x] ResidentSnapshot readback is physical-type aware. F32 roots decode as F32;
  BF16 roots widen exact BF16 bits to F32 for comparison, reject nonfinite
  encodings, derive active KV elements from typed element count/capacity, and
  require the untouched physical tail bytes to be zero.
- [x] Typed snapshot unit tests, exact admission matrix, strict cuda-cublas and
  qwen35-cuda-qualification Clippy, and all-features build pass.
- [ ] This provides the committed-root comparison surface for the real .8B
  47→50 parity harness; it does not substitute for the blocked physical run.

### 28 Oct 2026 — cached47→50 two-process real-parity harness ready

- [x] Qualification-only phase 1 captures the approved historical current-source
  .8B path at frontiers 47→48→49→50, including full 248320 logits, greedy
  ID/max bits, and all 42 canonical state domains. Reference KV checks exact
  future +0 and BF16-widened F32 semantics. After explicit historical model
  shutdown, it writes bounded SHA256-protected host comparison evidence.
- [x] Same-process reference→ordinary was rejected during audit: the historical
  path loads cuBLAS, while ordinary bounded BLAS activation must precede every
  cuBLAS DSO mapping. Phase 2 therefore runs in a fresh process. It independently
  authenticates the approved source, fixed 512 context and provider paths;
  evidence cannot choose the plan, context, prompt or next token. Ordinary
  48→50 uses its own greedy feedback and stops at the first exact-parity failure.
- [x] Two ignored release-CUDA integration tests compile and strict-Clippy clean:
  ordinary_cached47_to50_capture_reference_evidence then, in a separate process,
  ordinary_cached47_to50_compare_reference_evidence. Both use the same absolute
  ZG_QWEN35_ORDINARY_PARITY_EVIDENCE path. Phase 1 additionally requires
  ZG_QWEN35_CUDA_KERNEL_DIR; phase 2 requires explicit ZG_QWEN35_CUBLAS and
  ZG_QWEN35_CUBLAS_LT. Source-free parity/evidence tests and all-features build
  pass. This is qualification infrastructure only.
- [ ] Execute both phases serially as soon as the approved .8B safetensor shard
  becomes visible to the connector. Do not download or substitute source.

### 28 Oct 2026 — ordinary RoPE switched to source-exact trig

- [x] Clean RTX 3060 qualification with explicit CUDA 12.9 headers/NVRTC
  reproduced a real legacy ordinary RoPE error at position 1:
  query[3] ordinary 0xc4dd vs source 0xc4dc (1 BF16 ULP).
- [x] Production ordinary execution no longer calls the legacy CUDA sincosf
  primitive. It now uses the existing source angle reduction, Arm-derived
  source sincos schedule, source rotary apply, and exact BF16 widen/pack
  transport under the same resident flight/status/completion owner.
- [x] Panic-free physical NVIDIA parity of the production raw rotary helper
  passes positions 0,1,47,48,49,50,511,512,1023,2047,4095,8191,16383,32767
  for query/key BF16 bits exactly.
- [x] Context admission totals were re-derived after the rotary scratch change;
  the fit matrix is unchanged (.8B through 32K, 2B through 16K; 2B 32K refuses).

### 28 Oct 2026 — ordinary owned-cuBLAS physical qualification harness

- [x] Added an isolated source-independent NVIDIA qualification for the exact
  ordinary owned-cuBLAS path at dynamic rows 1, 2, 17 and 64. It uses identity
  BF16 weights, verifies exact output bytes, retains the original resident
  flight/provider/pool owners, then explicitly retires the BLAS provider and
  verifies zero resident bytes/events before namespace close.
- [x] The harness compiles/discovers under qualification,cuda-cublas, strict
  Clippy passes, and all-features builds.
- [ ] Physical execution is currently blocked by provider identity, not code:
  the verifier is pinned to libcublas.so.12 bytes=105,140,976 SHA256
  d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048
  and libcublasLt.so.12 bytes=749,205,904 SHA256
  96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541.
  Historical validated runs used
  /usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/, but those held
  DSOs are not visible through the current connector. The available Nix CUDA
  12.9.2.10 DSOs have different sizes/hashes. Do not repin or substitute them
  without an explicit source/provider qualification decision.

### 28 Oct 2026 — source-independent ordinary primitive numerical parity

- [x] Clean RTX 3060 physical parity against source-derived arithmetic passes
  ordinary .8B full-width RMS for rows 1, 2, 17 and 64, including exact BF16
  output bits and zero shared status.
- [x] Query-head RMS (8x256) and key-head RMS (2x256) from the real packed
  5120-row attention layout pass exact source parity for rows 1, 2, 17 and 64.
- [x] Ordinary FFN SiLU*up and attention sigmoid pass exact source SLEEF BF16
  parity for rows 1, 2, 17 and 64. The tested paths cover the production packed
  GateUp and attention-gate layouts.
- [x] These gates are source/model independent and do not replace the mandatory
  current-source full-model parity run. Together with the source-exact RoPE
  correction they remove the known primitive-level numerical drift before the
  blocked .8B 47->50 campaign.

### 28 Oct 2026 — ordinary residual-RMS physical source parity

- [x] Clean RTX 3060 physical parity passes ordinary residual-RMS for rows
  1, 2, 17 and 64 against source arithmetic, including the in-place BF16
  residual update, normalized BF16 output, and zero shared status.
- [x] The source-free harness is discoverable under
  qualification,cuda-cublas,qwen-source-qualification; strict combined Clippy,
  all-features, formatting and diff checks pass.
- [x] This closes the last ordinary RMS normalization variant with distinct
  alias/rounding semantics before the blocked current-source full-model run.

### 28 Oct 2026 — ordinary multiply and KV append physical qualification

- [x] Clean RTX 3060 qualification passes the production post-sigmoid multiply
  against source arithmetic exactly for rows 1, 2, 17 and 64, including BF16
  output bits and zero shared status.
- [x] Production KV append passes rows 1, 2, 17 and 64 at base 31/capacity 512:
  key/value BF16 payloads land in exactly [base, base+rows), all cache words
  outside that interval remain untouched sentinels, and shared status stays 0.
- [x] The harness is source/model/provider independent and strict
  qualification,cuda-cublas Clippy passes. It does not substitute for the
  blocked full-model state-history/attention continuation campaign.

### 28 Oct 2026 — ordinary embedding physical qualification

- [x] Clean RTX 3060 qualification passes the production .8B embedding gather
  at the exact 248,320 x 1,024 BF16 table extent for rows 1, 2, 17 and 64.
  Selected rows are populated through exact device subviews and gathered output
  matches the expected BF16 rows bit-for-bit with zero shared status.
- [x] Token ID == vocabulary is also qualified: the production kernel writes an
  all-zero output row and reports exactly STATUS_OUT_OF_RANGE.
- [x] This closes the remaining simple ordinary transport primitive that can be
  physically qualified without the blocked model/provider artifacts.

### 28 Oct 2026 — ordinary attention wrapper physical qualification

- [x] Clean RTX 3060 qualification passes the private production
  OrdinaryCudaAttention wrapper for rows 1, 2, 17 and 64 at base 31/capacity
  512 using authentic materialized .8B FlashAttention descriptors.
- [x] Candidate KV future tail is deliberately BF16-NaN poisoned. The wrapper
  widens only the active prefix, keeps shared status at zero, and matches an
  independently scheduled source-attention oracle bit-for-bit for every row's
  causal frontier base+row+1.
- [x] This qualifies wrapper prefix/row/layout/BF16 transport semantics in
  addition to the already source-derived attention arithmetic.

### 28 Oct 2026 — ordinary canonical GDN state transport physical qualification

- [x] Clean RTX 3060 qualification passes the production canonical state
  adapters ConvIn->ConvOut and RecurrentIn->RecurrentOut with exact round-trip
  bytes, zero shared status, and zero resident bytes/events before namespace
  close.
- [x] Conv transport covers canonical F32 BF16-widened roots <-> native BF16
  [frame,channel] layout; recurrent transport covers the exact F32 H/K/V <->
  H/V/K transpose.
- [x] A deliberately non-BF16 canonical conv root sets the transport fault bit,
  proving fail-closed validation rather than silently truncating malformed root
  state.

### 28 Oct 2026 — generic ordinary GDN runtime physical qualification

- [x] Added and ran a source/model/provider-independent RTX 3060 smoke for the
  production parameterized GDN runtime with the exact ordinary stage tuple.
  Initial prefill rows 1, 2, 17 and 64, continued prefill frontier 64 / rows 17,
  and cached decode frontier 71 all execute both convolution and recurrent
  partitions successfully with zero-fixture outputs.
- [x] The smoke found a real cached Gated launch-geometry bug before the
  full-model campaign: the retained cached PTX writes 128 BF16 outputs per
  block and immutable original metadata for PTX SHA a97d127b... records grid
  [16,1,1], while the parameterized runtime launched [4,1,1]. That wrote only
  512 of the declared 2,048 BF16 cached Gated outputs.
- [x] Cached Gated now launches [16,1,1]; prefill keeps its source-derived
  rows/4 grid. A source-free regression pins cached [16,1,1] and 47-token
  prefill [188,1,1]. The physical zero-fixture now passes cached decode as well
  as every prefill/continued-prefill cell.
- [x] Strict qualification,cuda-cublas Clippy, all-features build, formatting
  and diff checks pass after the correction.

### 28 Oct 2026 — deterministic serial ordinary cached47->50 run wrapper

- [x] Added scripts/qualify-qwen35-ordinary-cached47-50.sh. It performs no
  acquisition, substitution, repinning, permission changes or GPU recovery.
  Before either process starts it requires explicit absolute paths and
  authenticates: the complete approved .8B HF source census from source08.json,
  the complete non-provider .8B native-kernel census from the checked-in bundle
  pins, the exact pinned libcublas.so.12/libcublasLt.so.12 SONAME/size/SHA
  identities, an explicit CUDA include view, NVRTC visibility, a fresh evidence
  path, and an idle NVIDIA compute queue.
- [x] The wrapper runs the existing two ignored integration tests in separate
  release processes, reference evidence first and ordinary comparison second,
  with --no-default-features and defaults of CARGO_BUILD_JOBS=8 /
  RAYON_NUM_THREADS=8. GPU idleness is rechecked between phases and after the
  comparison.
- [x] Release-mode qwen35_cuda integration-test cargo check passes for the exact
  qwen35-cuda-qualification feature surface. bash syntax/shell checks pass.
  Missing-env preflight is explicit/fail-closed.
- [x] Current-host preflight with valid CUDA 12.9/NVRTC paths performs no GPU
  work and refuses immediately because the available target/models/qwen35-0.8b
  directory is not the approved HF source tree (first missing member: LICENSE).
  This is expected and confirms the wrapper stops before provider/kernel/GPU
  execution when held artifacts are absent.
- [ ] When the approved source tree, pinned native bundle payloads and pinned
  provider DSOs reappear, use this wrapper with a fresh evidence path; no manual
  phase ordering is required.

### 28 Oct 2026 — ordinary parity evidence/restart hardening

- [x] Reference evidence publication is now crash-safe at the final filename:
  phase 1 writes/hashes/fsyncs a create-new sibling temporary file, atomically
  hard-links it into the create-new final evidence path, fsyncs the directory,
  then removes the temporary name. A partial phase-1 crash cannot publish a
  partial final evidence file.
- [x] Evidence publication has a source-free regression proving the final name
  is absent before publish, complete bytes appear at publish, and a second
  publisher cannot overwrite an existing final evidence file.
- [x] The serial wrapper now supports --compare-existing for deterministic
  phase-2 restart after a successful reference capture. It requires an existing
  regular non-symlink evidence file, reruns the complete source/kernel/provider/
  CUDA/NVRTC/GPU-idle preflight, then runs only the fresh ordinary comparison
  process. Corrupt evidence remains SHA/schema rejected before ordinary load.
- [x] Wrapper mode tests pass: compare-existing rejects missing evidence,
  fresh/preflight mode rejects an existing evidence path, and compare-existing
  with an existing file advances to artifact authentication. Release integration
  cargo check, strict qualification Clippy, parity unit tests, bash syntax and
  shell checks all pass with CARGO_BUILD_JOBS=8 / RAYON_NUM_THREADS=8.
