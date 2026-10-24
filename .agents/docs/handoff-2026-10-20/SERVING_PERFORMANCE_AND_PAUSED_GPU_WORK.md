# Serving, performance and paused GPU work — 20 October 2026

Author: batch_design_review, for ROOT's comprehensive handoff.
Scope: continuation after c85ad25; serving fusion, numerical boundaries,
benchmark methodology/results, GPU drafts and the exact paused SiLU checkpoint.
Documentation only. Live source, executed results and owning TODOs outrank this
snapshot. No new code/build/test/provider/model/GPU execution was performed to
write it. ROOT alone owns aggregation, shared plans, commits and execution.

## 1. Stop state and authority

The owner paused implementation/runtime work at a reasonable checkpoint, then
authorized investigation/planning and this comprehensive documentation. These
requests do **not** authorize silently resuming the prepared GPU matrix.

- Last accepted implementation HEAD: `fb68d8f` — combined state scatter and
  certification. Accepted source/server evidence precedes the SiLU overlay.
- MAIN has an **unaccepted ordered-SiLU integration and diagnostic overlay**.
  It is not clean; do not reset it, overwrite newer parent files, stage others'
  changes or infer that a successful build accepted the model/serving change.
- Current candidate release build completed successfully. New frozen library,
  qualifier, server, worker and HTTP artifacts exist, but the prepared 60-case
  physical matrix was **not started**. Loaded-code/SASS proof, model/state,
  serving/full gate/routes and new performance evidence remain pending.
- `target/release/zerogravity` may now be the unaccepted candidate. Do not use
  its filename as an accepted benchmark identity. Use the explicit frozen
  artifact and source/build association for each experiment.
- Current dirty paths observed while writing: coordinator TODO updates,
  `lib.rs`, `resident_program.rs`, `resident_program/batch.rs`, vendor CUDA
  `compute/context.rs`; new `ordered_silu.rs`, `ordered_silu/`,
  `resident_program/silu.rs`, `resident_program/silu/`; new correctness-first
  plan. This list is a snapshot, not permission to modify them.

Preserve one Rust mathematical engine, original immutable source identities,
TaskCoordinator/fair queue, independent SessionAuthority/StateRoot ownership,
ResourceArbiter, physical leases, Ready/completion/quarantine and checked atomic
publication. Preserve root README bytes and Renoir/gfx90c AMD launch hold.
Signed owned commits only; no push/tag/deploy/history destruction or device
privilege changes without applicable authorization. No code/runtime resumes
merely because a diagnostic command appears below.

## 2. Context entry points and supersession

Read the original
[FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md)
progressively for October 10 onward, V1–V5.1, numerical engine/resource design,
failed approaches and original milestones. At c85ad25, complete model graphs
were still separate per request despite accepted batch operators/cohort
publication. That limitation was genuinely superseded by 35ebc5b, not by a
scheduler counter or renamed API.

Owning documents:

- [Serving TODO](../../plans/serving-and-speculation/TODO.md),
  [FUSED_GPU_BATCHING](../../plans/serving-and-speculation/FUSED_GPU_BATCHING.md).
- [Model TODO](../../plans/model-execution/TODO.md),
  [Numerics TODO](../../plans/numerics-and-context/TODO.md).
- [Correctness-first vLLM plan](../../plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md):
  newly reviewed planning document, not executed acceptance.
- Joint serving/goodput, 36 serving gates and Qwen basics specifications remain
  linked by the serving TODO. This handoff does not replace their contracts.

Later named gates supersede only their own earlier limitations. For example,
real opt-in fused HTTP is accepted now, but simultaneous temporal prefill,
trained original-HF full-model execution, production graphs and matched
source-correct vLLM performance are still separate open work.

## 3. Post-c85ad25 implementation chronology

The signed implementation sequence is the durable provenance. Docs-only
904a190/570d865/7fc9ad4 record their corresponding acceptance boundaries.

| Commit | What actually advanced | Boundary retained |
| --- | --- | --- |
|35ebc5b|P12-B3 one batch-native complete ForwardPlan for B2/4/8, both approved mixed-GGUF models|Full-logit same-frontier diagnostic initially; no HTTP/greedy performance claim|
|54fd580|P12-B4 native greedy reduction and independent12B device packets, native feedback gather|No fabricated host packet/source-root authority; HTTP still separate then|
|2677d8b|P12-B5 actual CUDA B2 HTTP/SSE waves, bounded resumable presentation and retirement|Default interleaved; native opt-in only; other prefill initially B1|
|f26232e|Authentic original head tying follows trained configuration|Original HF diagnostic fix, not a full original-HF engine|
|6bfcdbe|Offered-load/token-observation benchmark accounting|No inferred true ITL/equivalence/goodput|
|d9fa829|P12-B6 finite unpadded8/4/2/1 HTTP bands and physical prelaunch capacity fallback|Typed None only permits smaller bands; errors/unknown completion never fallback|
|9815504|Original HF diagnostic source/profile binding|Diagnostics explicitly do not enable production HF|
|dbb5f4f|P12-B7 true unequal-frontier decode and bounded per-request publication/delivery observations|Source/profile/context qualification is explicit; no temporal block prefill|
|cadcbd9|Pinned CPU HF operator work|Closed operators, not full trained source execution|
|1b599df|Owned serialized benchmark lifecycle|Lifecycle smoke is not matched arithmetic/performance|
|d411634 /2a32aff|Verified tensor windows and typed source operators|No fake reduced matrix/source descriptors|
|c5eb1f8 /d0a6084|Source sigmoid/RMS and fused activation preparation|Exact original codecs/status/owners preserved|
|5350a4d /bfe62df /fe2a342|Typed source windows, leased execution and trained attention fragment|CPU/reference slices do not enable full HF/27B|
|f788929|P12-B8 actual native StateOnly teacher-prefix serving|One teacher/request/wave, not simultaneous multi-token prefill|
|9a19901|Ordered RoPE and private full-model CUDA graph qualification|Production global capture/driver allocation admission remains closed|
|fff1842|Trained BF16 projection/conv slices and genuine stock prerequisites|CPU-target CUDA profile distinct from stock CUDA; full HF remains closed|
|fb68d8f|Row-native state scatter plus original finite certification; accepted physical/serving phase and native-only A/B|Fewer launches not a vLLM win; candidate SiLU follows as unaccepted overlay|

### B3/B4: what physical fusion means

Original0.8B uses Q4_0/Q8_0 plus F32; original2B additionally uses Q4_1/Q6_K.
These are the already accepted packed files, not newly requantized weights.
One original source-bound plan is lowered across native B2/4/8 kernels for
embedding/projections/quant preparation/norm/RoPE/conv/GDN/attention/FFN.
No copied Qwen token/layer loop, CPU math fallback or separate batch engine.

- Every row retains its actual independent prior/candidate BufferLeases and
  root attempt. Packed device state is scratch, not another state authority.
- One real completion produces ordered independent Ready proofs; all-row
  checked publication validates every root before any sibling changes.
- Full248320 logits and all42 domains matched **independent CPU AND accepted
  B1 GPU**, distinct histories, two decode steps at B3; B4 adds six steps with
  four genuine device-feedback steps and complete-logit shadows.
- B3 aggregate two-step native launches .8B B2/4/8:11442/11994/13076 versus
 22024/44048/88096 separate B1. 2B:10782/11334/12416 versus20704/41408/82816.
  These are original source launch sites, not optimized throughput claims.
- B4 full-vocabulary selector preserves lowest-ID ties, finite ordering,
  signed zero/subnormals and sticky status. Four native sampling stages at
  V248320 plus B original packet copies; readback4+12B=28/52/100 bytes.
- Device packet consumption binds exact source/session/epoch/root/frontier and
  initialized lease. Mutable public host ID/max metadata is non-authoritative.
- Imported CPU-hydrated prefixes0/255/511 cover valid1/256/257/512, exact
  future KV positive-zero tails, capacity refusal and reclamation. This is
  decode-from-imported-state proof, not full GPU prefill of those prefixes.
- Actual dropped native work retains the original4GiB envelope, handles and
  source/packet/state owners in quarantine; prior roots/all42 remain exact,
  old Ready/further work/new same-card service refuse. Diagnostic readback or
  sync never constitutes recovery or permission to clear a fence.

### B5–B8 serving semantics that must survive optimization

- GeneratedText GPU presentation stages a token once and resumes only pending
  bounded transport events. If final Text queues but Done is blocked, retry
  cannot duplicate Text, token usage or terminal. CPU accept remains fail-closed.
- A per-packet presented mark survives discarded candidate retries. Presentation
  precedes packet feedback. Blocked output rows park with bounded timeout and
  are excluded from GPU feedback; cancellation/disconnection retires them.
- Actual B1-produced→native and native-produced→B1 packet handoffs are tested.
  Genuine trained EOS, unequal output limits, after-Ready cancellation,
  capacity1 backpressure, survivor replay and final active/queue/failed0 pass.
- Native widths2/4/8 are ceilings, not padded row counts. Bands8/4/2/1 regroup
  surviving eligible rows after EOS/length/cancellation. Real prelaunch
  admission None can lower a band; other errors propagate to original owners.
- B6 initially hit CUDA OOM after completed old-stream memory pools accumulated
  across models. Cold cleanup now runs under original initialization/queue
  guards. Actual repetition freed completed idle pools; no unsafe reset/recovery.
- B7 per-row positions drive RoPE, KV append, causal softmax and AV. Mixed
  positions255/256/0/31/32/1/510/511, refill and feedback compare fullV/all42.
  Padded AV must skip inactive FMA lanes; canonical future zeros remain exact.
- B8 partitions StateOnly prefix work from final prompt-tail greedy work before
  banding. Every teacher advances its own state/cursor but emits no generated
  ID/Text/usage/Done. EOS **as a teacher** is ordinary input. Ragged tails may
  finish through B1; prefix HTTP tests must not demand unrelated greedy fusion.
- Current native scheduler consumes one teacher per row/wave, including unmatched
  B1 native-mode rows. Interleaved retains max32 prefix quantum. This exposes
  host/root/readback overhead and motivates the paused temporal proposals.

Current source anchors: `resident_program.rs`, `resident_program/batch.rs`,
`resident_inference{,/fused.rs}`, `resident_batch.rs`, original `resident_leases`
and physical queue/quarantine; CLI `serve/gpu_engine.rs`, `gpu_engine/cohort.rs`,
`gpu_batch.rs`, shared `engine.rs`; internal qualifiers and `tests/serve_gpu.rs`
including prefix child. Never infer internal all42 evidence from HTTP text.

## 4. Last accepted state-scatter phase

Run directory:
`.agents/runs/serving-and-speculation-state-scatter-certification-20261020`.
Accepted commit fb68d8f follows exact U32 row transport and persistent finite
certification fusion; original42-domain checked preflight remains before flight.
Inactive fixed ABI aliases are safe only because inactive bindings are never
accessed; real spans/source/service/alias checks are preserved.

Accepted phase receipts include:

- `root-full-model-matrix-1.json`:46 physical model/owner/private-graph/failure
  cases; `root-operator-matrix-1.json`:10 operator/failure cases.
- `root-full-check-result-1.json`: complete isolated1251-test phase, strict
  checks and both approved original cache-direct/HF-import release routes.
- `root-serving-physical-2.json` SHA
 186259602e1debf69df208c5f12be0f60687a6fb81c43a699c34f5a55dcd0431:
 14 worker cases953.94s and12 HTTP cases593.68s, exact frozen source/build,
  original owner/group/EOF closure. All26 scopes passed; no claim this timing
  is serving performance. GPU returned idle after the phase.
- Compile-fix-v2 originally repaired four BufferArg unsafe contexts and explicit
  qualifier imports only. No shader/math/layout change. Preserve first strict
  failure and derivative e5caecdd metadata, rather than pretending first-pass
  compilation. Later actual receipts supersede the original source-only proposal.

Accepted server used for the latest native A/B:
87e1d727f908368f508eeb1e4dd53456f14348b8c72c4519cbee5e52b977c61e.
Previous accepted baseline:
4a7455277b89aa48e92bd9d30b15a939935017368918c0e304a4f699687aae3f.
Both labels are native Zerogravity; neither belongs in a vLLM comparison slot.

## 5. Performance history and interpretation

### Native versus native

Fused activation preparation uses the same original7 intermediate buffers and
all codec/status bits. Physical272 cases/all4 codecs/B1/2/4/8 compare1,483,608
words exactly; original projections need5 launches instead of10 (Q6_K8),
before outer certification. Source-bound cold PreparedBlocks avoid repeated warm
range/capability/alias setup; CUDA refusal propagates, no heuristic CPU fallback.

Bounded native0.8B warm ABBA initially measured mean wave39.14206→33.163665s
(-15.3%), dispatches2072360→1722640(-16.9%); ITL p95 .3911→.3933s did not
improve. Both samples miss10s TTFT. Broader warm/cold both-model observations
later pass with equal provider4GiB/device5GiB; an initial equal4GiB capacity
attempt remains refused, not silently rescued on one side.

Native StateOnly prefix observations (small finite samples):

| Case | Old→new wave seconds | Source launches old→new | Limits |
| --- | --- | --- | --- |
|.8B concurrent8|32.301205→18.33249|1722640→302160|StrictB8; TTFT still>10s|
|2B naturalC8|42.19858→28.457115|1668496→294384|First strictB8 attempt failed; fresh naturalC8 accepts genuine bands|
|.8B B1|4.43954→4.42998|unchanged246314|No broad non-regression proof|
|2B B1|5.60352→5.652635|unchanged238538|No consistent ITL-tail gain|

The failed2B strictB8 report5da6dc90 must remain unqualified. The fresh reviewed
naturalC8 protocol validates actual2/4/8+B1 membership rather than forcing B8;
successful new samples happened to have46 StateOnly B8 +8 greedy B8 cohorts.

Latest accepted state-certification A/B:
`.agents/runs/serving-and-speculation-native-before-after-20261020/root-native-v2-1`.
Exactly32 fresh native processes: ABBA then BAAB for each model/C1 orC8;
144 offers/completions,1152 output IDs and32 separate ordinary canonical warmups.
All IDs/text/usage/finish agree across versions. All144 responses contain8
single-ID delivery groups and7 real intervals:1008 ITLs, no coalescing proxy.
Original owner/process/group/container/EOF closure passes in every arm.

| Model/C | Rate tokens/s baseline→candidate | Offered TTFT p50(s) | Client ITL p95(s) | Source launches |
| --- | --- | --- | --- | --- |
|.8B C1|1.69640→1.69750|4.08717→4.09925|.10324→.11145|236270→236270|
|.8B C8|2.25060→2.28945|25.77409→25.32761|.40800→.40623|292116→239952|
|2B C1|1.36985→1.37575|4.74065→4.71593|.16056→.16569|228494→228494|
|2B C8|1.62680→1.65430|32.66844→32.05865|.97987→.97747|284340→232176|

Four samples/label, medians of per-sample summaries, not pooled percentiles or
confidence-certified wins. About18% fewer C8 source launches correspond to only
about1.73%/1.69% descriptive median rate gains. C1 unchanged. ActualC8 samples
have54 B8 cohorts/432 model-token steps/roots, source error/refusal/discard/
cancel deltas0. All128 C8 requests miss the10s TTFT SLO despite ITL<2s;
qualified source-correct goodput is unset. C1 wire SLO passes.

Point GPU memory stays1059/3045MiB(.8B C1/C8),1857/3509MiB(2B), not exact
allocator peaks. Temperatures61..85°C/clocks varied. One ordinary C1 warmup
does not fully warm B8/JIT/graph shapes. Eight simultaneous offers measure a
burst, not sustained capacity. Two workload labels do not prove tenant isolation.

### Genuine pinned stock vLLM baseline

Pinned V1 environment is real: vLLM module0.31.0, distribution0.31.0+cu129,
commitgdb9527a46; Torch2.13.0+cu129/gitcf30153; local pinned image f57d77f5…
with manifest eec72f47…. Do not confuse module/distribution version strings.
Official out-of-tree GGUF plugin e2b8ad532b8b builds unchanged against this
runtime (wheelaf9de736). Genuine source-converted BF16 visual projectors and
language-only loading are used, never dummy weights or model math patches.

- Stock0.8B loader: all320 backbone roles, plain/observed two full248320 BF16
  vocabularies496640 words exact. Stock2B inventory335=320 backbone+15 genuine
  MTP exclusions following unchanged source; all335 descriptors validated.
  Source/header/dtype/byte audits, raw logits and owned cleanup pass.
- This proves stock-loader observation transparency, not all42/ZG arithmetic,
  full generation quality, default graph route or throughput parity.
- Original HF stock eager B1 first-layer observations also preserve all320
  source/cast roles and496640 raw BF16 logits. Actual loaded0.8B profile:
 284 rawBF16→BF16;18 A_log F32→F32;18 gated gamma F32→BF16. It is distinct
  from the restored-F32 CPU target. Do not generalize those counts to2B/27B.
- Authentic public cuBLAS observation proves four QKVZ/BA GemmEx calls for
  teachers11/353, A/B/C BF16, compute68, requestedalgo99, alpha1/beta0,
  transposeA1/B0, math/atomics/pointer modes0, stream0 and explicit8519680B
  workspace. Hidden selected reduction kernel and opaque provider allocation
  admission are still open. API settings alone are not exact native arithmetic.

Actual V8 unprofiled eager/cold-process/solo comparisons use canonical47 prompt
IDs/max8, greedy/EOS source policy, same GPU UUID, CPU4/RAM12GiB/swap0 and
owned serialized providers. Cold means first request after readiness, not
storage-cache cold or complete startup latency.

| Model | Native median TTFT | Stock median TTFT | Numerical/quality limit |
| --- | ---: | ---: | --- |
|.8B|13.400s|1.561s|Cross-engine IDs/text/usage differ; one stock sample coalesces individual delivery|
|2B|14.540s|1.562s|Canonical one-prompt IDs/text/usage/finish match; seven-prompt quality still unqualified|

Each model has8 total ABBA/BAAB samples, four requests/32 raw output tokens per
provider. 2B all samples have individual delivery; native p95ITL .186–.199s
versus stock .026–.030s. Five of seven2B quality fulltexts match, but that is
not full quality/numerical equivalence. .8B seven-prompt quality also remains
unqualified. Stock is faster on these raw bounded points; no neutralizing
claim, source-correct goodput or broader vLLM parity is justified.

Evidence root: `.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020`:
`root-compare-{08b,2b}-eager-cold-solo-v8-1*`, matching pre-benchmark,
policy/quality validation and tokenizer records. V8 protocolb0c50e8a retains
the genuine launch/workload. Earlier V3–V7 failures are not replaced by V8.

## 6. Benchmark/profiler lifecycle lessons and known failures

- Original generic lifecycle implementation had reap-before-PGID-signal risks.
  Actual diagnostic/native A/B paths use reviewed53ee reserved-leader WNOWAIT
  whole-group cleanup and00d1 record-only TERM+INT guard. Preserve these exact
  seams when copying helpers; comments about finite cleanup are not proof.
- Never raise asynchronously across Popen assignment/owner handoff or cleanup.
  Guard records a flag; explicit checkpoints raise only after ownership exists.
  Original drain is restored before retirement; failure/receipt/log errors
  prevent the next provider. Every nonzero group signal precedes first reap.
- Podman cleanup authenticates nonce/name/CID, retries bounded stop/force removal,
  reaps original foreground/group and rediscovers owned startup races. Unknown
  HPIDs establish no ownership; GPU membership must match numeric owned PIDs.
- Private native-only A/B needs full rootless map and existing reviewed
  `podman unshare → unshare --net --mount → root-private-loopback-exec-2.py`.
  The single-UID native-gpu map breaks rootless overlay ownership. Do not retry
  this known bad path or force permissions. Preserve `_CONTAINERS` roots.
- Initial FlashInfer read-only `/root/.cache` failure was fixed through the
  public pinned FLASHINFER_WORKSPACE_BASE=/tmp setting for both arms, not HOME
  replacement or a model math change. Initial GPU membership failures were
  diagnosed by bounded raw evidence, not relaxed ownership predicates.
- Actual vLLM module/dist version and Podman image bareId versus sha256 prefix
  mismatches were corrected from authenticated runtime source; failed attempts
  remain. Do not broaden version/provider matching to make a run pass.
- Stock2B loader first rejected335 versus assumed320 stored tensors. Fresh V4
  validates all335/exact15 MTP exclusions plus320 original source audits; it does
  not hide unknown parameters. Early serializer singleton-stride failures and
  original transparent-capture evidence remain separate preserved failures.

## 7. Actual hardware counts, not source-counter fiction

Official CUPTI already exists locally in the pinned environment; no SDK download
or device privilege repair was needed. Host collector V2 separately passes two
requests/C2/max4 `Say hello.` after a one-token warmup:64328 actual kernel records,
64328 successful direct driver launches,0 graphs,0 loss, exact source delta,
plain/observed complete outputs and original owner/EOF closure. This is not the
canonical benchmark workload and must not be substituted into its denominator.

Canonical container diagnostic adds a separate bounded512MiB/4M-record
collector revision and finalization checks. LD_PRELOAD is injected **only at
the final native exec**, never outer Python/Podman (double constructor/stale
socket hazard). Full mapped FD/inode/hash closure is bound through authenticated
nonce/CID, init hostPID, pidfd/start/ns and SO_PEERCRED. Raw CUPTI driver PID1
or authenticated hostPID is preserved, never rewritten; foreign/mixed IDs refuse.

Known mapped-device discrepancy: stat device encoding versus `/proc/maps`
kernel key differed. V3 retained opened FD, full SHA/fstat and an actual private
readonly page mapping to derive the kernel key; it did not ignore device
identity. V4 supplies the record-only cancellation wrapper. Both metadata-only
and harmless owner fixture results are separate from GPU collection.

Accepted canonical diagnostic:
`.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/root-container-c1-v4-1/validation.json`.
Four actual native/stock plain/observed owners close; native records236270
kernels/direct launches,0 graph APIs,0 lost retained records, exact source delta
and outputs. Stock public profiler records9318 activities/8810 launches but
losslessness/all-worker coverage is **not qualified**. Device profiler internal
allocation admission is also open. No complete matched hardware-count ratio.

Short-window count histogram (not time): SiLU10080/15.67%, finite9178/14.27%,
merge5230/8.13%, copies3704/5.76%. It corroborates many tiny launches but cannot
rank latency. Never treat count fractions as speed fractions or use profiled
diagnostics as unprofiled timing samples. Keep raw large traces; continuation
can inspect compact metadata first, then schedule bounded offline analysis.

## 8. Exact paused ordered-SiLU checkpoint

### Operator and integration sources

All original immutable bundles remain:

| Run / freeze | Meaning |
| --- | --- |
|`serving-and-speculation-ordered-silu-20261020`, a7939ba7…|Initial CUDA-only ordered leaf, old pipeline oracle,17 checkpoints/adversaries|
|same/followup-v2,70c5947f…|Qualifier MiB budget fixes and cold-preflight/lifecycle correction|
|`serving-and-speculation-ordered-silu-scalar-input-20261020`, 16198590…|All15 asm scalar inputs bit-materialized across9 helpers|
|`serving-and-speculation-ordered-silu-integration-20261020`, 9110ef76…|Original B1/native emitter integration, unchanged old fallback and budgets|
|same/followup-visibility-v2,3bc2bf33…|Only pub(super) on preflight_silu/launch_silu; binding helper remains private|
|`serving-and-speculation-ordered-silu-compiler-witness-review-20261020`|Opt-in vendor dump collision repair and source review; ROOT composed it|

The generic closed operator preserves the original12-stage SiLU math (13 with
gate) in one kernel, all original FP32 store/rounding/FMA/sign/FTZ semantics.
Trace variant exposes17 checkpoints; compact variant must match. Independent
old GPU pipeline and original CPU arithmetic remain the oracles. No model/math
profile change or global substitute for original-HF SiLU is allowed.

Integration adds no persistent binding table/heap/device scratch. Exact same
immutable handles/rows/width are cold-preflighted before original flight and
stack-prepared again before the node's first launch. Known CUDA/profile/geometry
selects the new path; unsupported geometry/non-CUDA retains exact original
fallback. Any unexpected binding/service error propagates through original
flight/quarantine, never Err-triggered hot fallback. B8 aggregate is not B1.
Original interval/outer certificates, scratch admission and status persist.

### Actual compile failure and scalar repair

First physical normal attempt failed before numerical Ready. CubeCL asm
push_input used HasValue.value instead of read_value; mutable prior asm outputs
were pointers. Actual CPP fed float/double pointers with constraint'l'; mov.b32
then failed size checking, while some64-bit operations could wrongly consume
address bits without failing. Do not merely repair the final mov instruction.

Source repair adds15 scalar materializations across9 helper inputs:
`f32::reinterpret(u32::reinterpret(value))` and f64/u64 equivalent. Frontend
reinterpret forces read_value/load, then carries bits unchanged. No arithmetic,
asm template, rounding, output, lease/queue/status change. Generic backend
pointer dereference would break legitimate address operands and was rejected.

ROOT applied exact corrected module3d789c06…; operator source freeze2:
158be58c846d390de61f08ac751ae6cdb968458ca6490ad82a0f4a68825f40ab.
Five CPP-parser fixtures pass, including rejecting both actual failed sources.
One peer mistakenly reported a missing assertRaises, re-read nesting and
explicitly retracted it; **no parser fix is needed** for that false finding.

### Operator physical result and incomplete instruction witness

`.agents/runs/serving-and-speculation-ordered-silu-20261020/root-physical-2.json`: six actual fresh processes PASS.
Normal204 cases171562710 bit comparisons; all17 intermediate checkpoints,
old-GPU/CPU/compact equality, status/canaries and original ownership pass.
Known-Ready abandonment compares5808 pairs. Unknown/preexisting/nonfinite/range
processes report0 numerical pairs and test their refusal/status/ownership scope;
do not read their shared summary as extra numerical comparisons.

ROOT compiler audit:
`.agents/runs/serving-and-speculation-ordered-silu-20261020/root-compiler-audit-2/report.json`.
All144 authentic ordered CPP specializations have correct scalar constraints.
Four count41120 plain/gated×traced/compact CPP variants are identified, but
only compact-gated entry ordered_6be19757 retains matching PTX. Vendor dump used
last180 sanitized KernelId characters, overwriting settings-distinct kernels.
This is **one specialization's PTX witness**, not all144 or loaded SASS.

ROOT composed diagnostic-only fix: full Debug KernelId sidecar, StableHasher128
index +PID +AtomicU64 sequence, short create_new filenames, finite collisions;
raw PTX bytes and NVRTC/module-load/queue paths unchanged. Missing/partial/orphan
files still fail witness acceptance; dumping remains best effort. No loaded
SASS inspection is supplied by CPP/PTX token scripts.

Initial direct vendor Cargo-unit invocation failed because the excluded vendor
package/features were not workspace targets. `root-composition-1/release-1.json`
preserves exit101 before new release artifacts. ROOT extracted the exact actual
write_ptx_dump/static/whole unit, linked the actual existing StableHasher and
ran one harmless file-transport unit successfully. That is no CUDA/math/SASS
proof; `dump-transport-isolated-1/{result,manifest}.json` bind its scope.

### Current MAIN candidate build — compiled, unaccepted

Directory:
`.agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1`.

- `composition.json`: original integration +geometry +lib wiring +visibility.
  Never copy9110's old full ordered module over scalar-corrected3d789c source.
- `diagnostic-composition-1.json`: vendor collision-safe diagnostic patch only.
- `strict-1.json`: CUDA/Vulkan/default strict and2 row-geometry tests PASS.
- `source-freeze-3.json` SHA
 30fdcfa2b67df26a82a272e14f80b22c315bae7b048784156adb04f52ccba244.
- `release-2.json`: PASS_RELEASE_BUILD_REQUIRES_PHYSICAL_OPERATOR_MODEL_SERVING_AND_FULL_GATE.
  CUDA strict/library/qualifier build, serving strict and release worker/HTTP/
  server build all0. This is the final completed build at user pause.

| Frozen artifact | SHA256 | Meaning |
| --- | --- | --- |
|frozen-libtest-2|f603eb47ef26853e85fd77e7f69b42e86b63eebdebb9b84a07d7601383309bcc|Candidate test binary|
|frozen-qualifier-2|fe9604655aeb92e7f7c0cfc668964cba1e2ade2a608f546e1764ef170dded16a|Candidate physical CLI|
|frozen-server-2|af0f04902559d13cc817138f973ea517ca8279150e12a22941fc955730e04faa|Unaccepted serving binary|
|frozen-worker-2|f1f2071db1e1d9ec6fcef99568047880530a40cdce3cd9cbac8873976c86695c|Candidate internal worker tests|
|frozen-http-2|02c4b4eda6b6eedb8670b2ea68335645f15a0ca326e8be1d07f4466380a44f7c|Candidate HTTP tests|

Prepared `root-physical-1.py` was source-reviewed but **not run**. It enumerates
60 distinct fresh processes:6 operator +42 qualifier CLI +12 exact prefix/
admission tests =54 model cases. Graph abort state/logits/greedy modes and
rows/source/caps match actual CLI parsing; exact ignored-test names require1
passed, preventing zero-test false success. Original53ee/00d1 ownership and
source/artifact/README checks surround every case. Only operator-normal enables
private compilation logging/PTX dump; profiling remains disabled. Generic PASS
markers still require independent per-mode result review.

Loaded SASS/code path remains an explicit open acceptance requirement. On
resumption decide the source-supported loaded-code witness method before claiming
it closed. A fresh collision-safe PTX run must join full KernelId +exact CPP
entry to exact PTX entry for every claimed specialization, reject incomplete
pairs and preserve the original failed/truncated evidence. Token presence does
not prove loaded binary identity or complete compiler control/dataflow.

Remaining integration gates after explicit resume: fresh60 physical/compiler
cases, worker14/HTTP12, both approved release routes, strict/full isolated gate,
source/lease/cancel/unknown completion review and fresh native A/B plus separate
hardware diagnostic. No faster/vLLM claim exists for this candidate.

## 9. Temporal, graph and other pending source drafts

These drafts are not implementations accepted by existing physical receipts.
Preserve exact parent source snapshots and narrow hunks; MAIN has evolved.

- `serving-and-speculation-native-prefix-chunk-20261020`, cc3a62c9…:
  sequential native B2/4/8 T2/T4 StateOnly. Original body runsT times; gather
  once, two existing private banks, each intermediate candidate fully certified,
  latched status, final scatter/one4B Ready and begin_advance(T). It amortizes
  sync/ownership/transport, **not** temporal matrix work or body kernel count.
- `serving-and-speculation-native-temporal-grouping-20261020`,6be63adf…:
  true closed B2/T2, four activation rows[T0R0,T0R1,T1R0,T1R1], only two roots.
  Stateless original nodes onB4; causal full attention/GDN onB2 in teacher order.
  Exact SSA/state-version dependencies, same plan lineage, U32 bridges, complete
  P2+P4 physical budgets, intermediate certification and one final publication.
  Physical16 row/B8×T2 is not supported. No CLI/worker route is enabled.
- `serving-and-speculation-native-solo-temporal-20261020`,4f2a9d5f…:
  direct one-session B1/T2; genuine original B1 causal emitter and middle/final
  all42 private leases, B2 stateless lowering, full P1/P2/middle/final admission,
  begin_advance2/one4B Ready/root1. No BatchProgram1/fake second session.
  Original ROOT_WIRING.patch is malformed; use additive followup-v2
  CORRECT_ROOT_WIRING.patch17726d0e… with proper prefixed blank/mod solo lines.
  Source reviews pass after that repair; no compile/physical acceptance.
- Each draft prepares genuine CPU+B1 fullV/all42 per teacher, intermediate state
  inspection through a separate original read-only flight after real Ready,
  EOS teachers, nonzero/boundary histories, real pressure, stale/abandoned Ready,
  fresh unknown completion and synthetic intermediate-NaN sticky-status tests.
  Synthetic poison proving latching is not a trained nonfinite-model result.

Private CUDA graph proof landed 9a19901: both models B2/4/8, StateOnly and Logits
with outside greedy, fullV/all42/CPU+B1, pressure/cancel/stale/final0.18 fresh
unknown graph cases preserve source/executable/state owners and fences. Source
keys bind program/source generation/band/capacity/profile/readout. Actual graph
host API enqueue1 is never described as underlying kernel1. Device-feedback
graph replay remains refused without its own proof.

Production capture still lacks complete same-card exclusion and physical
graph/driver metadata admission. Initialization/alloc/H2D/sync outside ordinary
fair-step scope, external event/storage/GC, constructors and final Drop/service
retirement must all participate. Fresh-process caller isolation used in private
qualification is insufficient for production.

Spine proposal runs to preserve:

- `serving-and-speculation-cuda-capture-exclusion-spine-20261020`: original
  QueueState/FairDeviceQueue protocol, permanently closed participation/issuer
  outside tests. Initial drain-waker/revocation and mutable borrow holes were
  repaired additively in followup-v2. ROOT14 host tests/3 intended negative
  ownership compiles pass; that is host protocol, not CUDA completeness.
- `serving-and-speculation-cuda-command-admission-spine-20261020`: optional
  pre-enqueue common transport hooks on original channel/mutex/reentrant actor;
  defaultNone path unchanged; only Unproven participation. ROOT separate14 host
  tests/2 intended ownership negatives, strict/no_std pass; no Complete bridge.
- `serving-and-speculation-cuda-retirement-spine-20261020`: prepaid bounded
  original-runner retirement slots, publish-only last Drop, source/executable
  keepalive until known completion. No actual QueueState/terminal bridge or
  production Complete issuer; author source-only/unexecuted.
- Initial graph source baseline/CompletionOwner/quarantine and actor PIN work
  in `p12-cuda-graph-20261020` and `p12-hf-cuda-boundary-20261020` worktrees.
  Source visibility repair and fail-closed graph memory-node queries were
  necessary; preserve final followups, not just original245e2956 source freeze.

`CompletionOwner` is retained BEFORE first enqueue/init operation and even
before a possible `ensure_usable` refusal when prior work may exist. Armed Drop
moves owners and original handles/envelopes into original fence/quarantine;
success releases only after real completion/readback/ledger validation. Actor
pinning must precede any own allocation/clone/submission; same actor identity
does not imply another same-card service cannot submit. Do not add a second
CUDA stream/queue or arbitrary raw-pointer callback to bypass these gaps.

Unfinished DEFAULT-bulk relocation is explicitly non-runnable:
`.agents/runs/numerics-and-context-default-bulk-isolated-20261020/{CHECKPOINT,TODO}.md`.
ROOT-owned worktree `root-default-bulk-20261020`, detached fb68, contains exact
primitive689+attributiona7+typed0d11+GPU refusal deltas only, uncompiled. ROOT463
source freeze a4891041… is known; copied original helper baselines/design await
new source/model namespace separation and actual compiler attestation. It is a
132096-word DEFAULT CPU leaf replay, not GPU prefill/full8K or serving work.

## 10. Resume procedure and preservation priorities

Until explicit runtime/implementation resume, documentation/investigation only.
After resume, revalidate HEAD/worktrees/dirty owners/build processes/CUDA state
and live TODOs before choosing a gate. Do not restart from c85ad25 or reenact
accepted fusion work. Current source/frozen binaries may differ from accepted
server paths; verify the intended exact scope before invoking a helper.

For SiLU continuation, first inspect source-freeze3/release2, original failures,
CPP/PTX coverage gap and collision fix; retain source/owner identity, fresh
mtimes on any copied bytes and the user's pause disposition. Do not run the
60-case helper automatically from a handoff. ROOT decides loaded-code witness,
serial matrix and subsequent full gates. Only reviewed/verified changes may be
signed/committed, with owning TODO reconciliation by ROOT.

Preserve at least these source/small-metadata run families when ROOT archives
ignored work; binary/model/DSO/large raw traces need not be duplicated in docs,
but their existing authentic paths/references must not be silently removed:

1. State-scatter/certification all source freezes/check/matrix/build/serving
   receipts and compile-fix failures; accepted87e1 and prior4a source association.
2. Native A/B original+V2/V3 namespace/owner followups, accepted serving bindings,
   raw32 samples/validator/review and source/launch/workload protocols.
3. Genuine stock V1 environment/plugin/projector/loader, matched V8 launches,
   all failed V3–V7 startup/membership/cache attempts, policy/quality/source
   validators, original complete IDs and raw comparison records.
4. CUPTI V1/V2 collector source/provider pins, owner repairs, capacity/finalization/
   FD-map/signal followups, actual accepted diagnostic metadata and raw trace
   references; no incomplete old trace promoted to matched hardware counts.
5. All SiLU original a793/70c/1619/9110/3bc freezes, actual failed CPP/physical
   logs, corrected operator result, partial compiler audit and current ROOT
   composition/source/build artifacts/unexecuted runner. Keep accepted and
   candidate binaries distinct; do not prune the only frozen source copies.
6. Sequential/true-temporal/SOLO source baselines, patches, corrections and
   prepared qualifiers; graph/capture/common-hook/retirement source followups
   and host-only results. No physical completion claim from their metadata.
7. This durable chapter and embedded performance report below, plus ROOT's
   consolidated correctness-first plan and companion mathematical handoff.

## 11. Independent performance investigation — durable prose copy

The following report was prepared under investigation/planning-only authority.
It consolidates measured bottlenecks versus hypotheses, shared-engine priorities,
benchmark matrix/tool gaps and27B capacity. It is reproduced here so its prose
survives eventual ignored-run cleanup. Original location:
`.agents/runs/serving-and-speculation-performance-audit-batch-20261020/REPORT.md`.

### Shared-engine performance audit — planning only

Owner: batch_design_review. Parent: serving-and-speculation; ROOT consolidates.
No code, tests, builds, model/provider/GPU/container execution or payload scans.
This is an independent recommendation, not a new implementation tracker.

#### Conclusion

The strongest immediate target is the repeated single-teacher execution path,
not a larger advertised batch flag. Native multi-request B2/4/8 is real, but
prefill still advances one teacher per request/wave and emits thousands of
kernels per model-token. A credible route toward vLLM performance combines
temporal prefill, exact leaf fusion, substantially better projection kernels
and eventually production-safe graph replay. None alone establishes a win.

Numerical correctness and physical ownership are release gates for every cut.
The current GGUF CPU-target profile and stock's loaded BF16 arithmetic are
different targets. Faster execution of one does not establish cross-engine
equivalence; the common GPU profile remains a critical-path requirement.

#### What is measured, and what is inferred

| Observation | Evidence | Meaning / remaining uncertainty |
| --- | --- | --- |
| Cold-process eager C1 native TTFT13.400s/.8B and14.540s/2B; stock about1.56s | V8 cold-solo accepted raw comparisons, both models | Large raw latency gap. First request after readiness; startup/storage-cold latency excluded. Numerical/quality equivalence remains open. |
| 2B canonical IDs/usage/text match on one prompt, seven-prompt quality still unqualified | V8 policy/quality/pre-benchmark records | Matching one greedy response does not qualify hidden states or numerical profile. .8B canonical responses differ. |
| Warmed native C1 TTFT about4.09s/.8B and4.72s/2B |32-process native-only A/B | Cold/warm gap makes lazy compilation/allocation/initialization a necessary diagnostic. Different experiments; do not subtract them as a measured JIT duration. |
| Native prefix batching cuts earlier concurrent source launches about82%, improving finite-wave time43%/.8B and33%/2B | Native-prefix warm observations | Multi-request prefix fusion matters; C1 changes little. All concurrent samples still miss10s TTFT. |
| Latest C8 source launches292116→239952/.8B and284340→232176/2B; median rates+1.73%/+1.69% | Accepted native-only32-process before/after | About18% fewer launches did not yield18% speed. No causal per-family time breakdown or confidence-certified improvement. |
| 64328 actual native kernels in short B2 window; SiLU15.67%, finite14.27%, merge8.13%, copies5.76% of count | Validated native CUPTI +small family histogram | Real count inventory, **not latency fractions** and not canonical C1/current candidate. |
| Canonical native C1 has236270 actual kernels/direct launches; stock trace has9318 activities/8810 launches | Separate diagnostic records | Native window is qualified. Stock losslessness/all-process coverage remains open; no qualified hardware-count ratio. |
| Native prefill_count returns1; B1 and batch execute/certify the original node list | Current gpu_engine.rs/resident_program.rs/batch.rs | Actual source bottleneck opportunity. Sequential per-teacher body dispatch remains. Its critical-path share needs timing. |
| Existing projections map one request/output row to one GPU thread and reuse exact source dot loops | batch_quant_projection.rs/mixed_ops.rs | Genuine B batching, not a tiled high-throughput matrix multiply. TensorCore/CMMA use is absent from these paths. Compute/BW/occupancy dominance is not measured. |

The proposed ordered SiLU operator passed204 old-GPU/CPU cases and171562710 bit
comparisons; production integration remains unaccepted at the paused checkpoint.
Its original PTX filename collision invalidates complete specialization coverage,
not the retained physical comparisons. The collision-safe diagnostic and new
model/serving gates must close before measuring it as accepted production.

#### Priorities and measurable gates

##### 0. Close the comparable numerical target and benchmark accounting

- Keep native GGUF regression optimization against accepted independent CPU/B1
  full-vocabulary/all42 references. Keep original-HF official loader, restored
  F32 CPU profile and stock CUDA loaded-cast profile explicitly distinct.
- For genuine common-HF CUDA comparison, qualify the actual loaded parameter
  casts, BF16 projection reductions, GDN/FLA initial versus cached schedule,
  RoPE/attention, KV layout and complete state/logits. .8B B1 stock subset and
  public GemmEx settings are useful antecedents, not full-model equivalence.
- Extend existing comparison tools minimally with an authenticated independently
  validated parity record and original physical publication/committed-token
  binding. Record must bind weights/quantization, tokenizer/template, numerical
  profile, graph/program generation, generation policy, source/test scope and
  hardware. Output hashes or caller declarations alone cannot create it.
- Current serve-compare deliberately leaves source-correct SLO goodput unset and
  equivalence NOT_ESTABLISHED. Add paired process/block statistics after these
  gates; preserve legacy usage-throughput fields and never relabel them goodput.

Acceptance: full required mathematical/StateRoot/physical regression suites;
comparison rejects stale/incompatible parity records and unverifiable commits.
No source-correct performance claim until these prerequisites hold.

##### 1. Obtain a critical-path decomposition before choosing further kernels

Use separate diagnostic processes/windows, never profiled main timing samples.
Derive kernel-duration totals and interval unions, launch/API blocking time,
host gaps, H2D/D2H and device-copy bytes, synchronization count, allocation,
compile/cache misses and shape reuse. Sum of overlapping durations is not wall
time. Associate data with prefill/decode/maintenance and exact request/window.

- Compare cold-process versus declared warm conditions; explicitly attribute
  compile, initialization and H2D rather than moving cost behind readiness and
  claiming it disappeared. Report startup-to-ready separately.
- Close native/stock collector coverage, losses, child-process/PID attribution
  and device/RAM overhead. Native counter equality applies only to its exact
  workload; graph enqueue1 is not underlying kernel count1.
- Establish a simple roofline for projection/state traffic from **measured**
  bytes and time. Counts, low point GPU utilization and code inspection alone
  cannot distinguish bandwidth, arithmetic, launch bubbles or host stalls.

Acceptance: diagnostic output transparency, zero lost/unowned activity and
closed owners; explain the dominant measured latency components before a major
kernel rewrite. Existing bounded trace may support offline analysis by ROOT;
this investigation did not scan its raw payload.

##### 2. Finish small exact fusion already supported by source

- Qualify ordered SiLU in both original B1 and lifted B2/4/8 emitters, including
  plain/gated real widths, independent old operator/CPU arithmetic, status,
  canaries, source profile, exact buffer leases and unknown/Ready cancellation.
- Then rank RMS, coefficients/sigmoid, convolution/reduction, attention small
  chains and producer/certification/status merge by actual critical-path time.
  Fuse only while preserving every original rounding/store/FTZ/sign boundary
  and predicate, persistent error OR, cancellation and quarantine.
- Reuse fixed original workspace initially. Hoist validation/binding cold;
  avoid per-token metadata allocation. Never remove finite/range checks merely
  to improve counts, clear prior failure status, or silently retry after launch.

Acceptance: all intermediate/final bits versus independent original pipeline;
full model/all42, worker14/HTTP12, original release routes and isolated gate.
Ordered SiLU predicts11 plain or12 gated launch-site savings per executed node;
verify actual work and measure latency. No percentage speed forecast from count.

##### 3. Temporal prefill, including the dominant solo case

Two distinct milestones must not be conflated:

1. Sequential T2/T4 state-only chunk: gather once, existing B-body repeatedT,
   certify every produced state before consumption, persistent status once,
   final scatter/one4B Ready and one begin_advance(T). This amortizes host,
   transport and terminal sync, **not projection/MLP kernel traversal count**.
2. True temporal stateless lowering: original B1/T2 through existing B2 math,
   B2/T2 through B4, then B4/T2 through B8. Batch stateless projections/MLP
   across actual teacher rows; execute complete attention/GDN causal regions
   in original teacher order. Same original plan lineage, no fake requests or
   authorities. This can reduce repeated mathematical dispatch and reuse weights.

Existing cc3 sequential,6be temporal and SOLO proposals are source-only. Qualify
them in that dependency order, including solo B1; optimizing only C8 cannot fix
the observed C1 gap. Do not label a B-loop or sequentialT loop native temporal
matrix batching. B8/T2 needs qualified tiling or new geometry; B16 is not implied.

Preserve real teacher EOS as input, final output-tail reservation, nonzero
histories, state versions, canonical future KV zeros, intermediate failure
latching, all-before-flight capacity admission and one independent authority
per real request. T2 first; bounded T4 only after evidence/fairness. Keep parked
outputs/cancellation responsive and cap the scheduler quantum by actual work.

Acceptance: CPU+B1 all42 after each teacher, native intermediate/final all42 and
future fullV, exact logits/packets, pressure/refusal/unknown completion and final0.
Measure C1 TTFT/host calls/readbacks/copied bytes and C2/4/8 mixed-load decode
tails; require measurable gain without correctness or declared-SLO regressions.

##### 4. Rework the projection data path, not just its launch wrapper

- First exact GGUF route: tile output/request rows, coalesce packed weight loads,
  share activation codes/scales and retain the same per-block integer dots,
  zero points, scale arithmetic, eight FP32 lane accumulators and final tree.
  Investigate DP4A or bounded integer MMA only when integer subresults can match
  original lane/block grouping and overflow limits exactly. Do not reassociate
  FP scale reductions, replace activation quantization or narrow precision.
- Precompute immutable source scale metadata only under the source identity and
  physical budget; avoid persistent expanded weights exceeding admission.
- Separate common-HF CUDA route: native BF16 tiled/TensorCore projection in the
  SAME operator/provider seam is essential research for vLLM-class arithmetic
  intensity. Public GemmEx settings do not identify its hidden reduction kernel.
  Original-actor stream/handle/workspace/internal allocation/Ready ownership and
  authentic operand parity must close before a cuBLAS provider is admitted.
- Existing exact CPU-target BF16 CUDA kernel intentionally reproduces CPU64
  order and is not evidence of stock CUDA arithmetic or TensorCore throughput.

Acceptance: operator timing, achieved BW/occupancy/reuse, all codec scale/code/
intermediate/status bits, source-independent adversaries, complete model/state
and resource safety. TensorCore math requires its declared independently proven
GPU profile; it is not a universal replacement for CPU-exact GGUF reductions.

##### 5. Production graph reuse and synchronization reduction

Private graph proof exists; production admission does not. Close participation
for every same-card command, constructor/init H2D/allocation, external event/
storage, sync and final Drop. Same original actor/stream and fair queue only;
no second context, global mutex held across await or caller isolation fiction.

- Capture fixed owned math workspace; dynamic original-root gather/candidate
  scatter and independent publication stay outside. Keys bind source/program
  generation, numerical profile, row band/capacity/readout and relevant shapes.
- Bound graph/executable/driver metadata, cache entries and retirement owners.
  Eviction waits for real completion; unknown completion quarantines owners.
- Device-token feedback and graph variants need their own proof; current private
  host-teacher proof does not authorize them. Cold admission refusal can choose
  another qualified path; post-launch errors cannot become fallback.

Acceptance: physical exclusivity under competing init/Drop/commands, fairness,
epoch revocation, no capture allocation leaks, CPU/B1/native all42/fullV,
abort/pressure/cache invalidation and final0. Measure host enqueue/bubbles/TTFT;
underlying kernels remainK per replay even when host graph API calls become1.

##### 6. State traffic, attention and scheduling capacity

- Measure bytes first: current packed gather/scatter remains real state traffic
  even after row transport cut its kernel count. Consider direct per-row leased
  state bindings for uncaptured kernels, or immutable paged KV/prefix sharing
  under the original source/session lineage and physical ownership. GDN private
  candidate state still needs genuine independent writes. Do not alias writable
  roots or infer unpublished candidates are safe to overwrite.
- Graph fixed buffers and direct dynamic state bindings have different ownership
  constraints; benchmark both after qualification instead of silently capturing
  mutable published-state pointers.
- Fused attention/softmax and more parallel GDN/chunk prefill can be major long
  context work. FlashAttention/FLA cannot simply replace the CPU-target schedule:
  casts, reductions, masking, padded lanes and initial/cached order need their
  own independently verified common GPU contract.
- Continue ragged band/refill/age-aware decode-prefill scheduling; retain real
  typed capacity refusal and2/4/8→smaller bands, output credits, at-most-once SSE
  and prompt/decode fairness. Measure offered load and aging, not just full waves.
  Raise concurrency/geometry only after actual active-state/metadata/graph/cache
  admission; no silent bump to fill the GPU.

#### Fair staged benchmark matrix

Use unchanged canonical serve-bench.py/serve-compare.py and a new explicit
protocol derivative when metrics/parity hooks are added. Existing compare caps:
requests<=128, concurrency1..8, repetitions1..4; bench has other bounds, which
do not authorize changing compare/admission. Start at C1/2/4/8. Keep all failures.

| Stage | Work | Conditions / purpose |
| --- | --- | --- |
| A: exact regression | Existing47-ID prompt/max8 plus seven-prompt quality, genuine EOS and unequal outputs | .8B/2B, complete state/logits first; preserve current comparability/history. |
| B: startup/solo | Canonical C1 cold-process and warm-process | First after readiness versus exactly one canonical ordinary warmup. Separately report process startup-to-ready. Neither means storage-cold. |
| C: concurrent | Canonical C2/4/8 finite waves, then bounded repeated waves | Test true fusion, shape/JIT warm effects and tail retirement; ordinary C1 warmup does not fully warm C8. Add an explicitly separate shape-warmed protocol if desired, symmetrically for both. |
| D: prefill/decode shape | Short1/32/47 input, medium128 input, long448 input; outputs8/64 where capacity fits | Current accepted GPU capacity512 constrains prompt+actual generated frontier. Do not schedule512+64 or advertise8K GPU serving. Report actual IDs/EOS-dependent work. |
| E: interference/SLO | Open-loop fixed arrivals, long-prefill plus short decode, ragged refills, cancellation/backpressure and two workload labels | Sweep offered rates through saturation, finite queue and fixed deadline; test starvation, overload and committed goodput, not a closed-loop throughput illusion. Labels are not authenticated tenant isolation. |
| F: best route | Genuine pinned stock default/graphs versus qualified ZG best path | Eager/eager is a diagnostic row. Do not handicap stock to eager and call it best vLLM, or enable private ZG graphs prematurely. |

Keep equivalent weights/quantization/loaded casts/tokenizer/template/generation,
actual raw token IDs, EOS/usage/finish, source/hardware/context and policy limits.
Use the same original rootless/private isolation, GPU UUID, CPU4/12GiB RAM/swap0
and declared device envelope as current .8B/2B protocol; collect actual readbacks.
Changing caps or admission is a new point for BOTH providers, not a rescue run.
Counterbalanced ABBA/BAAB fresh-process blocks remain useful. More evidence can
use repeated bounded outer blocks without silently raising tool limits.

Report offered/accepted/rejected/timed-out/cancelled/completed requests, committed
and delivered tokens, p50/p95/p99 TTFT and **actual individual-token** ITL,
SLO-qualified committed goodput per offered second, queue/client backlog,
memory allocation peaks versus point samples, power/clocks/temperature, fairness
and actual hardware kernels/direct APIs/graph enqueues separately. Coalesced IDs
leave per-token delivery unresolved; mean ITL or SSE chunk timing cannot repair it.

Add paired process/block confidence intervals for predeclared primary metrics.
Do not treat tokens within one process as independent repetitions or median
per-sample p95 as a pooled p95. Check thermal/order effects and all losing points.
A speed claim needs a confidence-supported improvement on its named workload
with source-correct SLO-goodput and predeclared latency/resource/no-starvation
constraints. A finite-wave win is not sustained capacity or broad vLLM parity.

#### Model feasibility and dependencies

| Model | Honest next scope | Main blockers |
| --- | --- | --- |
| Qwen3.5 .8B | Accepted GGUF exact serving optimization and common-HF GPU qualification; bounded matched C1/2/4/8 after parity | Current cross-profile quality/state gap; token-major prefill, scalar projection path, production graphs/stock collector closure. |
| Qwen3.5 2B | Same engine/cuts, actual2B per-shape/resource proof rather than .8B extrapolation | HF shard4.548GB exceeds old4GiB complete-source guard; differing original/loaded casts, initial/attention/FLA closure; tighter active/workspace capacity. |
| Original Qwen3.8 27B | Finish initial exp/provider/core +common typed full model/112-domain CPU truth, then source-window streamed CPU/GPU and matched feasible offload | Isolated wide/cached/gamma operators pass but full model absent. Full55.563GB source cannot reside in current12GiB process cap or6GiB GPU. No stock exact-architecture/offload equivalence is established. |

Actual retained hardware is an RTX3060 **Laptop6GiB**, not a12GiB desktop card.
Current GPU operator envelope5GiB/provider4GiB is smaller again. Nominal27B
2-bit weights alone are about6.75GB before block metadata/state/workspace, so
even that is not an assumed fully resident solution. No such qualified source
is implied by a metadata listing. BF16 needs streamed/offloaded windows or a
separately authorized larger hardware configuration.

For27B, first prove the same engine's full original math and state; then measure
SSD/RAM/GPU bytes, throughput, source leases, I/O completion and overlap with
actual ResourceArbiter readiness. Multi-teacher prefill can amortize each staged
weight tile; streaming decode must disclose bandwidth/cache limits. Compare only
against stock's genuinely supported same source/profile/offload policy and caps.
If stock or ZG cannot execute that point, mark it infeasible/unqualified rather
than substitute another model/quantization/device or invent a benchmark.

#### Source/evidence entry points

- `.agents/plans/serving-and-speculation/{TODO,FUSED_GPU_BATCHING}.md` and
  `.agents/plans/model-execution/TODO.md`: live gates and accepted scope.
- Current `crates/zerogravity-cli/src/serve/gpu_engine.rs:507`, cohort.rs and
  `crates/zerogravity-gpu/src/resident_program/{batch.rs,silu.rs}`: scheduling,
  repeated body, status/certification and candidate integration seams.
- `crates/zerogravity-gpu/src/{batch_quant_projection,mixed_ops,source_accumulation}.rs`:
  original scalar native source reductions; CPU-target BF16 kernel expressly
  excludes TensorCore math.
- Native-only32-process A/B and independent review:
  `.agents/runs/serving-and-speculation-native-before-after-20261020/root-native-v2-1`,
  `../serving-and-speculation-native-ab-peer-review-batch-20261020/REVIEW.md`.
- Stock raw V8 cold-solo validation/pre-benchmark/quality records under
  `.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020`.
- Actual short-window family histogram:
  `.agents/runs/serving-and-speculation-cupti-activity-20261020/root-native-kernel-families-1.json`.
- Sequential cc3, temporal6be and SOLO source proposals remain unexecuted;
  no implementation/performance acceptance follows this planning report.
