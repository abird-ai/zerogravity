# Current takeover context — 20 October 2026

Author: pvl
Co-authors: Codex and the independent numerical, serving and history reviewers
Status: documentation-only handoff; implementation remains paused

## Latest instruction and authority

The owner originally requested full GPU fusion, actual matched vLLM benchmarks,
correct Qwen3.5-0.8B/2B and original Qwen3.8-27B, and shared-engine optimization
until faster than vLLM. The owner added the older independent long-context
GDN divergence as required work, then requested a pause at a reasonable
checkpoint. The already-running release build drained; no next physical
matrix began.

The owner briefly resumed GDN correctness investigation and requested clean
ports from pinned vLLM. **The subsequent instruction superseded implementation
authorization: planning and investigation only; do not touch code.** The
latest request is this comprehensive handoff, full retained context and a
starting prompt. It does not resume builds, model runs, GPU qualification,
benchmarks, acquisition or implementation.

Keep the earlier broad goals and approval history as context, but obey this
latest narrower scope. A new agent should onboard and report the exact
checkpoint, then remain paused until implementation is explicitly resumed.
No push, tag, deployment, publication or destructive Git action is authorized
by this handoff. Signed commits and owned-path staging rules remain.

Authority order: latest owner instruction → live AGENTS.md and plan rules →
current source and executed results → current owning TODOs/specifications →
this current-state summary → dated copies and historical design.

The handoff and its sourcebooks are a **one-time preservation requested by
the owner**, not a new required receipts/archive workflow. Existing TODOs
remain the only work queues. Historical unchecked snapshots are not extra
work items to recreate.

## Exact repository state at this inspection

- Repository: `/home/pvl/spaces/abird/src/zerogravity`.
- Branch `main`, signed HEAD
  `fb68d8fa165de16e21d1bd37afef3790b80da971`,
  `perf: combine GPU state scatter certification`.
- Local `origin/main` comparison: **34 ahead, 0 behind**; no fetch or remote
  verification was performed. Do not describe the remote server as current.
- Index empty. MAIN has the earlier unaccepted ordered-SiLU implementation
  overlay and documentation changes; do not reset, overwrite or stage them
  wholesale.
- **29 worktrees including MAIN**, with pending changes in every inspected
  tree. Older “only one clean worktree” statements are historical.
- Two especially important ROOT-owned detached trees:
  `../zerogravity-worktrees/root-bf16-composition-20261020` at `fff1842`;
  `../zerogravity-worktrees/root-default-bulk-20261020` at `fb68d8f`.
- Root README Git blob remains
  `a7be69c35cf61d14e8810f9cb5a6a43d1397bf27`; leave its bytes unchanged.
- Read-only process inspection found no Cargo/rustc/inference job; the dotlink
  bridge process remains. GPU point observation: RTX3060 **Laptop**, 6,144MiB,
  41MiB used, 0% utilization, 50°C. Point telemetry is not a new health or
  mathematical qualification.
- GPU UUID `GPU-89d28939-1000-40b8-e243-b18f72b3c381`.
  Preserve the local Renoir/gfx90c AMD launch hold.

Recheck all of these facts before resuming. Exact worktree/branch/status
inventory and post-handoff commit order appear in the assembled handoff.

The owner requested `@dotlink-l5`. Prior tool discovery did not expose a
callable connector in this session; direct tools accessed the shared local
host. Do not falsely attribute commands to the connector. Use it if exposed
to the next agent, otherwise identify the actual available access path.

## What changed after the October 19 handoff

The old handoff was based on `3ec431a`, followed by documentation-only
`c85ad25`. Its full body is retained unchanged in the new aggregate. Its
current-status sections are superseded as follows:

| Earlier assertion | Current fact |
|---|---|
| Complete native B>1 model graphs not implemented | Accepted complete native B2/B4/B8 execution exists for both approved small GGUF models, including unequal frontiers and independent roots |
| Fused HTTP disabled/interleaved only | Native fused HTTP/worker cohorts and device greedy feedback are physically qualified within their source/context scope |
| vLLM absent | Genuine pinned vLLM V1 and official GGUF plugin have run both models; bounded eager benchmarks exist |
| All independent8K evidence unresolved | Fresh AR1 and explicit reference-control8K comparisons match full vocabulary and all42 domains; default bulk/historical questions remain open |
| Original27B unavailable/source-only metadata | Original source acquired and verified; bounded source-window and trained operator work is real, but complete27B inference remains unqualified |
| No production graph work | Private captured graph replay is qualified; production graph resource/lifetime admission remains open |
| Clean MAIN, ten ahead, no side trees | Dirty preserved MAIN,34 ahead of local origin,29 worktrees |
| Current priority immediately implement P12 | Current instruction is documentation/planning only; next implementation priority, when resumed, is numerical correctness |

Native cross-request fusion is **not** temporal prefill. Prompt processing
still advances one teacher token through a complete model step per request/
wave. Sequential chunk preparation does not establish token-matrix execution.

## Proven, unproven and newly proposed work

### Accepted small-model path

Qwen3.5 original GGUF0.8B and2B CPU/resident CUDA, independent sessions,
full-vocabulary/all42-domain comparisons, native row bands2/4/8, ragged
frontiers, refill, device greedy feedback and physical cancellation/retirement
are accepted only within the recorded profiles, context capacities and tests.
GPU context beyond the qualified512 consumed-token scope is open.

At `fb68d8f`, the row/scatter certification phase passed46 physical
model/owner/private-graph/failure cases,10 operator/failure cases, full
isolated1251-test gate, relevant strict checks and both original release
routes, plus14 worker and12 HTTP cases. Its native-only paired comparison
completed144 measured requests/1152 tokens across32 processes: observed
C8 raw token rates improved about1.7%; C1 was nearly unchanged.
These historical executions are not rerun or extended by this handoff.

### Numerical issue

There is **no demonstrated single recurrence algebra bug** accounting for all
differences. Fresh first64 default-versus-AR1 comparisons locate:

- .8B: layer3 attention AV at position2, before the old layer4 GDN boundary.
- 2B: layer0 F32 alpha projection at position0; beta differs before recurrence.
- Default tinyBLAS and AR1 use different FP32 partial-accumulator/reduction
  schedules; later2B Q8 dequantized BLAS versus activation-quantized dots is
  a separate route difference.

Fresh8K AR1 and explicit `use_ref` controls already match the accepted Rust
profile. The old historical raw corpus is unrecovered. Do not call the old
issue closed merely because a new corpus passes; do not call all8K wrong.

Latest source investigation identifies further **profile differences** with
pinned vLLM: fused conv activation/store, prefill versus decode Q/K casts,
FP32 beta, scale placement, fused gated norm and mixed-precision chunk64/WY.
The old diagnostic also unconditionally rounds a weighted gamma product where
raw-F32-gamma CPU arithmetic should promote. The detailed arithmetic audit
and reviewed15-finding plan are preserved in full.

Source inspection alone has not identified the first actual full-model
vLLM→Rust mismatch or proven that changing those operations closes it.
Effective loaded dtypes, GGUF loader log/exp transforms, temporary activation
quantizers and selected compiled routes must be compared first.

### Original HF and27B

The common typed engine already has many genuine trained CPU operator
qualifications. Additional cached H48, BF16-gamma and wide-role results
exist in the isolated ROOT tree; they are not merged whole-model support.
Initial chunk, live coefficients, native-exp/provider admission, full trained
producer composition and complete112-domain/logit parity remain open.

Original27B has about55.56GB of BF16 weight bytes; it does not fit the local
6GiB GPU or the current bounded benchmark process configuration.
Source windows are not yet full streamed inference. Do not bypass executable
source, full-K projection or physical memory limits to claim support.

### Actual vLLM comparisons

Pinned vLLM0.31+cu129
`db9527a46873454610df6dbedf79a36d6bf1a7f6`, Torch2.13+cu129 and official
GGUF plugin `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8` ran genuine comparisons.
Median eager first-request-after-readiness TTFT:

| Model | Zerogravity | vLLM |
|---|---:|---:|
| .8B |13.400s|1.561s|
|2B|14.540s|1.562s|

These are bounded cold-process eager C1 results. Source-loaded arithmetic
profiles differ; .8B generated output differs, and matching2B output on one
prompt is not numerical qualification. Warm/default/concurrent matched
competitive gates remain open. Native canonical trace counted236,270 actual
kernels; stock's9,318 recorded activities do not have proven complete/lossless
coverage, so no qualified exact count ratio exists.

## Paused implementation that must survive

The release build finished at `2026-10-20 06:31:23 UTC` after the pause request.
The ordered-SiLU private operator passed204 physical cases/171,562,710 bit
comparisons and failure tests. Its MAIN integration is **unaccepted**.

Candidate source freeze461:
`30fdcfa2b67df26a82a272e14f80b22c315bae7b048784156adb04f52ccba244`.
Candidate release binary at `target/release/zerogravity` is **not the accepted
baseline**: SHA256
`af0f04902559d13cc817138f973ea517ca8279150e12a22941fc955730e04faa`.
The prepared60-case physical matrix was never started. Worker/HTTP, both
release parity routes, full isolated gate and fresh timing remain required
for this integration. A PTX filename-collision failure and source-only loaded
SASS collector must remain distinguished from physical instruction proof.

Default bulk ROOT tree:463-source freeze
`a4891041aa4fb57d3afac7f180212eddc65d735790dfea0f22f4e61eb7cca1b0`;
not compiled/replayed/merged. The faithful replay relocation has only six
baseline copies and an unfinished checkpoint.

Original-BF16 ROOT tree:521-source freeze
`036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615`;
isolated operator results pass, full model remains closed. Combined525
derivative is prepared, not applied/built/replayed. Do not relabel old521
binary evidence as525.

All paused source locations, exact qualification boundaries and next tests
are detailed in the numerical/model and serving/paused-work supplements.
Textual MAIN/ROOT-tree diffs, untracked source and selected unique draft/
harness files are preserved as inert documentation, not patches to apply
blindly. Existing worktrees and run artifacts must also remain intact.

## Next work, only after explicit implementation resumption

Follow the reviewed correctness-first plan:

1. Revalidate live state, preserve other work and select one MAIN Cargo/GPU
   coordinator. Give independent agents disjoint source/review scopes.
2. Prefer original .8B HF with the actual vLLM GPU loaded profile for the
   first GDN diagnosis; keep GGUF-plugin and Torch CPU targets separate.
3. Find the first differing operation on identical real inputs. Port its
   pinned arithmetic into the existing typed plan/backend; compose producer
   and consumer immediately.
4. Qualify cached core, then actual initial/continued chunk64 and complete
   initial→cached layer. Progress to full .8B,2B and original27B state/logits.
5. Keep default tinyBLAS/old-corpus work bounded and parallel; completing the
   separate Torch CPU native-provider path is not a prerequisite to the
   primary vLLM CUDA port.
6. After correctness, measure prefill/decode critical paths and improve true
   temporal prefill, projections, exact fusion, production-safe graphs,
   memory/state traffic and fair scheduling.
7. Produce model and OpenAI-serving comparisons using the existing benchmark
   tools; predeclare quality/SLO/load/placement and paired statistical gates.

Keep SSD/RAM/GPU source streaming, Gemma4 E2B/26B-A4B/31B, other model
families, speculation, multimodal, distributed and release work in their
existing owning plans. Nothing is canceled by prioritizing GDN.

## Working and verification rules

Use one Rust mathematical engine, immutable SourceIdentity, one
TaskCoordinator/SessionAuthority root publication authority, the original
ResourceArbiter and physical GPU queue/lease/event/quarantine ownership.
No separate model engine, false Ready, metadata-only execution claim,
post-launch fallback or cross-session writable state aliasing.

Implementation acceptance requires focused tests, relevant strict Clippy,
real source/state/cancellation/physical CUDA gates and both approved original
cache-direct/HF-import release parity routes. Those import routes do not
themselves qualify complete original-HF BF16 inference. Complete gate:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

The native isolation wrapper and rootless Podman have different namespaces.
Do not run full rootless Podman inside the wrapper's single-UID mapping.
Reuse the retained approved private-loopback/owned-process helper, record-only
signal guard and exact source/env binding when runtime work is resumed.
No asynchronous exception across process creation; no reap-then-kill of a
possibly recycled process group. Unknown device completion retains ownership.

When merging from worktrees, use fresh file modification times; preserving
old timestamps previously allowed stale Cargo libraries. Verify the actual
artifact/profile before treating a `target/release` executable as accepted.

No runtime tests were run to create this handoff. Documentation validation
does not extend any numerical, device, model, serving or performance claim.
