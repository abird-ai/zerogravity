# Short HF performance — current source21 measurement

24Oct source21 passes CPU/strict/build, physical23, full2B numerical composition,
both .8B/2B API/HTTP and the following fresh matched measurements:

| Client concurrency | ZG output tok/s | vLLM output tok/s | ZG ratio |
|---|---:|---:|---:|
| .8B C1 |67.7432|38.8470|1.74×|
| .8B C4 |95.6083|42.8328|2.23×|
| .8B C8 |92.9820|54.8241|1.70×|
| 2B C1 |45.0305|37.7431|1.19×|
| 2B C4 |80.8278|50.2514|1.61×|
| 2B C8 |80.7265|62.4442|1.29×|

Both providers use local NVIDIA CUDA/RTX3060Laptop6GiB, original pinned vLLM0.31
eager/BF16 recipe and eight-offer/eight-natural-output wave. All48offers384tokens
exact; six original process/container/group/one-reap/fullEOF/source closures
pass. All ZG compiler deltas0; cold448 remains448 through API first requests,
including five prompt lengths×eight outputs. C4/C8 each16actualB4flights64rows;
clientC8 is not a B8kernel claim. Native4GiB provider/5GiBphysical versus stock
requested4GiB/512MiBKV retain their differing enforcement; no equal-hard-cap
claim. One pass/band, unbalancedC8 provider order; preliminary, no confidence,
latest-tuned-vLLM or broad parity claim. Current2B C1/C4/C8 passes ROOT review:
all48offers384tokens/sixclosed owners exact, compiler0. C1 walls1.42126/1.69567s;
C4 walls.79181/1.27360s. 2B C4 has14cachedB4flights56rows plus8N1prefills.
C8 walls.79280/1.02492s;14cachedB4flights56rows plus8N1prefills. All three
bands win rate AND wall. 2B actual API24/192 per provider,
fresh genuine stock5cap HTTP5/36 and all4owners close. Cold396 equals whole,
compiler0; provider4608MiB/physical5120MiB/HOST4096MiB unchanged.
Failed20/compiler intervals remain preserved.

[Current C1/C4 raw](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/current-quiet08-21-1/run/quick.json),
[current C8 raw](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/current-c8-08-21-1/run/quick.json),
[current2B C1/C4 raw](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/current-quiet2b-21-1/run/quick.json),
[current2B C8 raw](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/current-c8-2b-21-1/run/quick.json),
[current queue](TODO.md).

Latest user bands: C1/C4/C8, then C16/C32/C64 for both models; no C2 requested.
The78 signed logical commits are complete; five isolated Pi lanes are active:
Qwen27B, Gemma4, Vulkan, HIP and the shared pending queue. Finish their current
tested units, reconcile and integrate into main, then retire completed trees.
The queue target is unlimited pending count with an optional configured cap and
actual HOST admission; C32/C64 follow its acceptance. Higher bands require genuine
offered populations and honest nativecapacity/kernelwidth. Best historical results remain distinct from
latest and keep their own matched stock arm/source/profile.

## C16 source21 — completed experiments, distinct timing scopes

| Model/scope | ZG tok/s | vLLM tok/s | ZG/vLLM wall seconds | Disposition |
|---|---:|---:|---|---|
| 2B, original measurement |78.7859|95.9974|1.62466 / 1.33337|Measured loss; retain |
| .8B, first measurement |102.0478|81.4600|1.25431 / 1.57132|Audit scan overlaps both arms; nonquiet |
| .8B, one repeat |105.7331|96.6544|1.21060 / 1.32431|Extra sampler; diagnostic only |

Each arm completed16offers/128tokens with0errors and original source/semantic/
owner closure. Native remains active8/B4, not B16; compiler deltas0. The2B
timed windows followed the known audit-scan termination. Its qualification
driver refuses the rate loss after both arms complete; numerical/cleanup
success does not convert it to a speed win. The .8B repeat added332 GPU/listener
samples across both arms, changing observation scope; neither .8B attempt is an
accepted quiet/best C16 result. No further unchanged rerun is scheduled.

[Original C16 timeline and rates](../../runs/resume-all-c16-executor-pi-20261024/C16_TIMELINE_AND_RAW_SUMMARY.json),
[final observation-scope review](../../runs/resume-all-parallel-source-supervision-sol-20261024/C16_FINAL_SCOPE_REVIEW.json).
After five-tree integration, implement shared prepared B4 validation/state-copy
submissions per [Astra design](../../runs/cpu-gpu-optimization-astra-c16-shared-gap-20261024/DESIGN.md),
then run a fresh matched comparison. Fewer software enqueues alone do not prove
wall improvement. C32/C64 remain unexecuted.

## Highest verified retained pairs

ROOT rechecked the paired rates and offered windows in the retained comparable
wave8 records. Highest ZG rate and stock rate below always come from one attempt;
this is a historical record, not a repeated-run ranking or confidence claim.

| Model | Band | Highest ZG tok/s | Same-attempt vLLM tok/s | Source |
|---|---|---:|---:|---|
| .8B | C1 |67.7996|37.4208|11|
| .8B | C4 |101.9840|49.0230|11|
| .8B | C8 |100.4600|61.9783|11|
| 2B | C1 |45.0305|37.7431|21|
| 2B | C4 |80.8278|50.2514|21|
| 2B | C8 |80.7265|62.4442|21|
| 2B | C16 |78.7859|95.9974|21; single measured loss|
| .8B | C16 |excluded|excluded|nonquiet/extra-sampler diagnostics above|
| Both | C32/C64 |pending|pending|unexecuted|

The .8B C1 retained maximum is about67.8tok/s; no verified70tok/s C1 appears
in this bounded set. Historical C4/C8 exceeded100tok/s. Source11 C1 differs
from current21 by0.083%; one pass cannot establish a performance change.
[Paired audit](../../runs/cpu-gpu-optimization-best-results-audit-pi-20261024/BEST_RESULTS.json).

## Historical source11 measurement

24Oct: shared admitted CUDA readiness, measured CLI10/11; fixture11 accounting
and current12 formatting do not change production bytes.

| Concurrency | ZG output tok/s | vLLM output tok/s | ZG ratio |
|---|---:|---:|---:|
| .8B C1 |67.7996|37.4208|1.81×|
| .8B C4 |101.9840|49.0230|2.08×|
| .8B C8 |100.4600|61.9783|1.62×|

Local RTX3060 Laptop6GiB, NVIDIA CUDA for both; pinned vLLM0.31 eager/BF16
recipe, original benchmark39 controls. Eight offers/eight natural tokens per
arm, four ordinary B1 warmups; opposite first provider across C1/C4 bands.
All48offers384tokens exact, six original source/process/container/group/reap/
fullEOF closures healthy. All offered time counts. One pass per arm: preliminary,
no balanced repeated-run confidence, latest fully tuned vLLM or broad parity
claim. Native4GiB provider/5GiB physical and stock4GiB request/512MiB KV retain
their original differing enforcement; stock has no equivalent hard device cap.

C4/C8 each execute16actual B4 flights/64rows; C8 peak8 independent sessions,
not an eight-row kernel claim. All three native arms report zero live compiler
attempt/key-source/PTX deltas. Cold exact-recipe compilation is admitted at
startup; actual512MiB DEVICE/384MiB HOST backend charges remain until process
retirement. This closes the measured first-use C4 regression on this profile;
it does not qualify arbitrary shapes or long contexts.

Current19physical scopes PASS: standalone301 finite checks, both models' full
40call source/state/natural matrices, B4 matrix/lifecycle/three submitted faults,
all N1 fault5/model and forced-streamed2B head cuts. API24offers192tokens/provider
PASS; genuine HTTP5N1prefills→mixedB4/refill/36natural tokens PASS with explicit
same-cap actual stock reuse and zero compiler deltas. Original same-prior B4
fullstate/fullV proof remains unchanged/reused. Old failed ABI/fresh-context and
provider-only fixture assertions remain preserved; no cap or production change
in the accounting correction. Fullgate17 stopped at fixture formatting only;
owned-file rustfmt applied. Fullgate18 PASS on current12:1756Rust/0failed/
89explicitignored/271suites/allscriptchecks; full1583MAIN/frozenWORK and
420selectedISO/rootREADME unchanged after the gate. GPU/model/serving grants
remain the separate frozen-binary scopes above.

[C1/C4 actual comparison](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/actual-readiness11-c1-c4-1/run/quick.json)
and [C8 actual comparison](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/actual-readiness11-c8-1/run/quick.json).

Historical continuous-path regression (superseded above):

24Oct source8 fixture proof +unchanged source6/CLI6, explicit finite B4:

| Concurrency | ZG output tok/s | vLLM output tok/s | Outcome |
|---|---:|---:|---|
| .8B C1 |66.7151|37.3407|N1; preliminary faster pass|
| .8B C4 |17.5611|49.0326|B4 regression; do not promote|
| .8B C8 |—|—|Held until both C1/C4 superior|

Original benchmark39 policy, eight offers/eight natural tokens, local RTX3060
Laptop6GiB CUDA, same original B1-only warmup. All32offers256tokens exact; four
original source/process/container/group/one-reap/fullEOF closures healthy.
One pass per provider/band; no repeated-run confidence claim. ROOT74852 CLOSED0.
[Actual measurement](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/actual-c1-c4-8-cli6-reused-1/quick.json).
Separate full API8 PASS24offers192tokens/provider; wave16B4flights64rows starts
with B4 prefill, ragged remains N1. Separate actual event-driven HTTPv2 passes
five genuine N1prefills→mixedcachedB4→completedA/replacementE/refilledB4,
five actualB4publications20rows/36natural outputs exact against actual same-cap
stock; all source/original owner closures healthy. A4+BCD/E8 correctness case,
not a timing comparison or general cancellation/wider-shape grant.
The native15-transition same-prior original comparison now passes all30calls,
fullstate/fullV/24sums/control/natural bits; explicit compositional proof, not an
independently generated original public trajectory or universal N1/B4 equality.
Source6 two-key KV/compiler lifetime/RMS geometry are present. Separate direct
compiler diagnostic4 now closes185trips/1480spans with exact8/64/source/owner and
monotonic-clock coverage. Its3.72236s wave contains79first-use trips with parent
compile union3.10167–3.10218s; nested NVRTC2.18533–2.18585s, module load
.84863–.84915s. Firstprefill45trips/1.66453–1.66482s within1.77291s; firstcached
34trips/1.43714–1.43736s within1.47071s. Later decode and secondwave contain zero
compiler spans. Child/parent scopes are not additive. Instrumented diagnostic,
not the quiet-run fraction or a faster benchmark. Exact source/artifact kept
separate; shared admitted cold compiler readiness is next.
[Direct attribution](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/actual-c4-compiler-diagnostic-4/timing-analysis.json).
Final assembled gate pending; gate16 below
accepts preceding316 only.

Historical finite B4 measurement (superseded above):

24Oct candidate400v4, explicit finite B4 (not default/promoted):

| Concurrency | ZG output tok/s | vLLM output tok/s | Outcome |
|---|---:|---:|---|
| .8B C1 |67.2592|37.1156|N1; preliminary faster pass|
| .8B C4 |11.7555|49.4752|B4 regression; do not promote|
| .8B C8 |—|—|Held until C4 superiority|

Same original eight-offer/eight-token controls, stock0.31 and local CUDA GPU.
All32offers256tokens exact, four original owners/source closures healthy. C4
executes16genuineB4 flights/64fused rows; this is not a counter-only batch claim.
First four service~5.128s, second four~0.315s; **all eight count in11.7555**.
Lazy first-use compilation and60position-specialized KV recipes are confirmed
in source, but no compiler wall fraction is measured. The separate Basic
diagnostic retains no compile events with compilation caching disabled; its
physical matrix passes, timing attribution remains unresolved. Dynamic KV and
continuous refill successors remain unmeasured. The source6 two-key KV and
compiler-lifetime successor passes finite B4/fullstate/lifecycle/three unknown
cuts. Shared geometry fixes mixed-frontier admission, but mixed N1→B4 fails the
extra N1-equivalence comparator. Diagnostic7 isolates QKVZ projection as the
first differing producer with identical embedding/normalized operands and all
consumed priors. Exact original B4 same-prior/fullstate comparison is pending;
cross-width kernel differences alone do not qualify it. No fresh API/timing or
speedup grant for this successor.
Actual release4 source/state/lifecycle/healthy retirement/three unknown cuts
and fresh API pass. Final assembled gate remains pending; gate16 below covers
preceding316 only. [Actual comparison](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/timing-b4-c1-c4-4/quick.json).

Prior accepted source checkpoint:

23Oct: .8B frozen prepared-binding release1; 2B row retains storage/rings release2.
Actual local NVIDIA CUDA execution on the RTX3060 Laptop6GiB, approved HF sources,
canonical47-token prompt, eight natural output tokens/request. Eight offered
requests per arm; concurrency limits1/4/8. Same original quality, source, warmup,
population and cleanup controls. Both engines executed on the same GPU.

| Concurrency | ZG output tok/s | vLLM output tok/s | ZG difference |
|---|---:|---:|---:|
| .8B C1 |67.2018|37.6230|+78.62%|
| .8B C4 |63.8830|50.4311|+26.67%|
| .8B C8 |60.9713|61.6415|−1.09%|
| 2B C1 |43.3738|37.6126|+15.32%|

- One measured pass/provider/band, no balanced repeated-run confidence or
  stable-tail claim. Throughput denominator includes the complete offered wave.
- .8B48offers/384output tokens and2B16offers/128tokens exact; all8 original
  source/process/container/EOF closures healthy. Real native peaks1/4/8;
  request math remains N1, batchrowsmax1 and zero fused rows. C8 superiority
  remains unmet. No stable superiority or complete parity claim.
- .8B native actual caps:4GiB host/provider,5GiB physical device. Stock original
  requested4GiB and explicit512MiB KV; it has no equivalent5GiB hard device cap.
  2B native provider is explicitly4608MiB; host4096MiB/device5120MiB unchanged.
  These are matched-workload results, not equal-resource enforcement.
- Separate frozen-binary correctness for EACH model:301 finite device cases,
  40 natural calls across47/57/62/67/72,
  full vocabulary/all42states/consumed priors/natural IDs, independent roots,
  five actual unknown/cancel fault cuts and matched API24offers192tokens per
  provider passed. Forced-streamed2B headunknown/headcancel additionally passes
  actual ring lifetime/quarantine checks. Current prepared-binding release1
  repeats both models' full physical/fault/API scopes successfully. Shared
  prepared semantics retain source-issued invocation provenance and original
  physical authority; per-call record bound512KiB replaces4MiB. Source-owned
  analyzed-site cache40 retains its complete witness and charged metadata;
  actual invocation/provenance validation runs outside the cache lock. GPU551
  and default/provider strict pass. Final assembled gate16 PASS:
  1752Rust/0failed/89ignored/269suites/allscriptchecks;316selected sourcepaths
  and both rootREADME identities unchanged.
  Wider numerical/profile/long-context parity remains open.
- Actual [.8B comparison](../../runs/cpu-gpu-optimization-prepared-binding-root-20261023/timing-08-1/quick.json)
  and [2B C1](../../runs/cpu-gpu-optimization-storage-rings-root-20261023/actual-batch-2/timing-2b-c1-4608/quick.json).
  Earlier storage/rings release2 observed .8B C1+63.70%/C4+21.68%/C8−7.08%.
  Prepared-recipe release1 observed C1+70.73%/C4+28.81%/C8−2.58%.
  Previous pair9 C8+2.03% was also onepass; its small margin was not stable
  confidence. Current C8 remains below vLLM; genuine batching is still pending.

## 2B head residency and transfer repair

Earlier frozen9 at4096MiB streamed the1017118720B vocabulary head through970
one-MiB uploads and actual stream waits perforward. Its exact preliminary C1
was ZG0.5298 vsstock37.0680tok/s (stock69.97x); preserved
[original comparison](../../../../zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/serving-and-speculation-short-hf-08-bench-live-state-20261022/prepared2b-pair9-c1-comparison.json).

Current source shares six mutually exclusive geometry workspaces, reducing
source-derived2B workspace allocation590622720→139673088B; this is an inventory
saving, not an observed peak. Source-shaped resident upload/native geometry and
an explicit4608MiB allowance admit the complete1017118720B table. Whole provider
bound4598390272B fits; no source/head upload occurs perresidentforward.

The 2B C1 row above measures storage/rings release2; it does not attribute a
separate speedup to each component. Both actual full source/state/fault/API
matrices pass. Explicit forced-streamed2B partial faults retain at most two
authenticated host windows, actual native events and original quarantine;
production streaming no longer synchronizes after every window. No healthy
all40 forced-streamed result or stable repeated timing claim is made.

Source-neutral prepared head/family extraction is accepted. Next: genuine BF16
B4/B8 independent-request lowering, then .8B C1/C4/C8 and2B/C16 as authorized.
C16 needs a real16-offer/admission profile; changing only client concurrency
cannot establish it with the current8-offer/active8 protocol.

## Prior causal evidence — actor8

Earlier actor8 diagnostic is closed: eight offers/64 exact natural outputs,
64 native forwards/roots,432 steps,peak8,zero fused rows. Original service and
both profiler owners close with source/group/reap/full-EOF checks. Its timing
is instrumented and does not replace the quiet timings above.

| Measured scope,64 forwards | Seconds |
|---|---:|
| Host forward total |0.860439|
| Host bind |0.138115|
| Host preflight |0.034819|
| Host enqueue |0.516009|
| Host completion/readback |0.136296|
| Host post-completion |0.032024|
| Whole device-clock envelope |0.615946|
| Nested vocabulary-head envelope |0.099558|

Device-clock envelopes include submission gaps; Head is inside whole, and host
prepare includes forward. Do not add these overlapping scopes. Median Head is
1.553472ms; it has zero H2D/checkpoints and one native launch per forward.
Process CPU is1.02user+.21system seconds over1.376918s profiling window;
the benchmark loop and perf-stat windows have different denominators.

258 user-cycle samples/zero lost now show allocator scan8.67%self,
driver mutex lock6.21%,worker memcmp5.44%,kernel-key xxhash4.11% and allocator
coalescing3.28%. Fractions describe sampled CPU cycles, not wall/GPU time or
removable gains. The exact xxhash callers are the server and driver KernelId
lookups, hashing generated full source text in that frozen binary. Pair9 shares
prepared content digests with full equality/collision fallback, cold GDN/FA copy
definitions and sealed immutable source metadata. Original device dispatches,
finite groups/scans and fault scalars remain unchanged; cold35-cohort metadata
increases1400B as priced within the same256KiB bound. Actual physical/API and
quiet measurements pass above; full isolated gate11 passes1724Rust/0failures/
88ignored/266suites/allscriptchecks over260unchangedsourcepaths/rootREADME.
Two parallel allocation/boundary audits rank root-backing reuse + immutable
layout/access sharing next, then owned transfer/frame retirement. No sample
fraction establishes a wall speedup. [Shared design](SHARED_MEMORY_AND_BOUNDARIES.md).
[Actual nested timing](../../../../zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/serving-and-speculation-short-hf-08-bench-live-state-20261022/actor8-sampled-phase-c8-1/nested-phase-and-gpu-envelope-summary.json).

Earlier frozen pair6 C8 diagnostic, exact64forwards/8offers and complete original+
profiler/source closure: nested forward1.134798s, bind.131585, preflight.036283,
enqueue.796107, complete/readback.137668 and post.031198. Head nested.005718s,
64launches,zeroH2D/checkpoints.233usercycle samples/0lost; explicit driver-launch
and queue/mutex activity, actor idle-spin~29.48%self,memcmp~9.29%self.

This diagnostic predates finite cohorts and actor placement. These are
instrumented host phases and CPU samples, not GPU durations or
removable wall percentages. They support investigating fewer actual commands;
they do not attribute the entire C8 gap to batching or a particular kernel.

The integrated source groups eligible finite checks into bounded cohorts.
The previous path launched327 separate BF16 checks/forward. Deferred checks require
exact physical range/sole producer/concrete-write/status-consumer proofs;
recycled or uncertified ranges stay immediate. Actual NaN/Inf/sticky-status/
range-edge tests and original40-call/five-fault/API qualification pass below.

Frozen pair7 now passes301 actual device edge cases,40 exact natural calls,
all independent-root lifecycle checks and allfive submitted fault cuts. Every
call observes279 deferred values in35groups plus48 immediate checks:83 finite
certificate enqueues rather than327,244 fewer per forward. These are original
client enqueue acknowledgements, not a hardware activity enumeration. Matched
API24offers192tokens per provider and original owner/source/EOF closure pass.
Assembled gate8b passes1715Rust/0failures/allscriptchecks over210selected
changed-source paths; gate8's test-module formatting-only failure is preserved.
Quiet matched timing is the table above. All48offers/384tokens and original
source/group/container/EOF closures pass; stable superiority remains open. The
reduced check count alone does not establish a causal speedup. Reviewed actor6
then moves original whole registration/enqueue onto the existing device runner,
keeping caller completion and streamed-head fallback unchanged. Frozen pair8
passes the original physical/API suite; all dispatch/finite/fault scalars equal
pair7. New quiet results are above; the current actor diagnostic and next
source unit are described at the start of this section.

## Independent work

- Earlier210selected changed-source checkpoint: isolated full gate8b passes1715Rust/
  zero failures/88ignored/266suites plus scripted checks; includes shared reuse,
  backend admission, CLI routing, finite cohorts and portable foundation tests.
  Pair7 binaries precede the later reviewed portable matrix11 integration.
  That source's full gate9 passes1715Rust/0failures/allscriptchecks over217
  selected paths. First actual HIP mode found a row allocation-range guard
  failure after exact full-table upload; corrected2path BufferRange successor
  now passes all12 actual HIP/Vulkan transport modes, independently reconciled
  by ROOT. Original failure preserved. Current220selected paths also include
  reviewed actor6. Frozen pair8 passes301deviceedge+40natural/all5faults and
  matchedAPI24offers192tokens perprovider, original source/ownedclosure exact.
  Every physical dispatch/finite/fault scalar equals pair7. Full isolated
  gate10 passes1715Rust/0failed/88ignored/266suites/allscriptchecks, source and
  rootREADME unchanged. Quiet C1/C4→C8 closed with the results above; ROOT
  independently reconciled all6 original owner/source/group/container/log
  records and actual counters. Current actor causal diagnostic is closed above.
  Transport results grant no arithmetic/model/serving eligibility.
- Gemma ordinary CPU candidate: real7cell matrix,50natural commits+2completed
  rejected proposals/fullV/all15KV/10710boundary row events exact,final0.
  GPU serving/performance remains open.
- Qwen27B CPU full64 replay: two independent sessions, each one INITIAL token
  ID11 then one cached token ID353; all1040records/240896000float words bitexact, fullV/all112domains/
  complete4-frame convolution state. Actual source/lifetime/finalhost0 pass.
  Natural feedback, long contexts, GPU, serving and performance remain open.
- [Vulkan/CUDA parity and AMD ROCm/HIP units](BACKEND_PARITY.md) proceed on
  pvl-x2 independently. Tiny stateful and BF16 transport tests do not grant
  whole-model numerical or performance eligibility.
