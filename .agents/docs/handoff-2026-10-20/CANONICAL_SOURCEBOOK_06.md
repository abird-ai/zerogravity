# Canonical Sourcebook — volume 6

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/plans/serving-and-speculation/FUSED_GPU_BATCHING.md

Original UTF-8 bytes: 53671; SHA256 `68b8f1707452914384170b5fa6fb5271617ed83fac3be18b73ea8f5837e76977`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# P12 — native fused GPU batch acceptance

**Current HTTP acceptance — P12-B8:** opt-in `batch_mode=fused`
uses the original full-model native CUDA B2/B4/B8 plan for teacher prefixes,
prompt tails and decode at independent causal frontiers. Prefix StateOnly and
last-prompt-token greedy barriers are partitioned before native banding.
Explicit `--batch-rows 2|4|8`, default2; interleaved remains the default mode.
Surviving rows regroup without padding. Both approved original Qwen3.5 models
pass actual release JSON/SSE, trained EOS, cancellation and exact internal
CPU/B1 state tests. Engine full248320-logit/compact packet/all42 evidence is
distinct from HTTP wire evidence. Paged state, simultaneous multi-token prefill,
trained-HF arithmetic and matched vLLM performance remain open. Later named
checkpoints supersede only their corresponding historical limitations below.

## Ownership boundaries

- `zerogravity-gpu/src/resident_inference.rs`: a single
  `ResidentModel` owns immutable source/physical `ResidentService`;
  `ResidentSession` owns each logical `SessionAuthority` and StateRoot.
- `zerogravity-gpu/src/resident_program.rs`: current
  `ResidentProgram::launch_impl` accepts one RuntimeParameters,
  one source-specific StateBinding cohort, and one device token.
  Its `resident_program/batch.rs` child lowers the SAME plan to
  multi-row kernels; it introduces no model schedule or root owner.
- `zerogravity-core/src/accelerator/program.rs`: the existing
  `ForwardPlan`/`PlanOperation` is the mathematical authority.
  Never fork Qwen into a separate batched mathematical runtime.
- `zerogravity-cli/src/serve/gpu_batch.rs`: scheduling membership,
  source/phase/frontier and finite credits ONLY. Co-scheduling is
  **not** a physical fused kernel, device Ready or StateRoot commit.

## P12-B0 — physically qualified source F32 multi-row projection

19 Oct 2026, RTX 3060 CUDA: one CubeCL kernel launch per B=2/4/8
projection (R=C=64), comparing 128/256/512 output words bit-exactly
against the previously accepted B1 CUDA GPU kernel. The B1 and
B>1 kernels share the same original Qwen 32-lane F32 dot reduction.

A first run without CUDA_PATH failed in CubeCL compiler retries and
is NOT numerical evidence. With already-installed CUDA runtime/CCCL
headers configured, the physical B2/4/8 tests exited 0. Strict CUDA
Clippy passed. Frozen repository suite 1147 passed/0 failed/69 ignored
over 241 suites. Original Qwen3.5 .8B/2B direct/HF-import parity
preserved exact full logits and all42 StateRoot domains.

This proves one fused PROJECTION OPERATOR, **not** batched full-model
inference or GPU-serving fusion. Native GDN, attention, KV, FFN,
token routing and per-session StateRoot batch execution remain open.
Existing HTTP batch_mode=fused must continue to refuse and report
gpu_fused_batches=0.

## P12-B1 — physical CUDA batch operator acceptance

19 Oct 2026, RTX 3060, source-qualified CUDA runtime and CCCL/NVRTC
headers from existing local Nix packages. B=2,4,8 tests passed
bit-for-bit against the original B1 GPU math for:

- Shared projection (previous P12-B0).
- RMSNorm, elementwise residual/SILU with B1-compatible arithmetic.
- GDN recurrence, including output and complete candidate-state bits.
- Immutable-origin KV candidate append (positions 0/7/31).
- Grouped QK/softmax/AV attention (valid positions 1/17/32).

One physical GPU operation dispatch covers all B lanes in each
native batch. These are isolated kernel/operation qualifications,
**not** a complete model-forward GPU batch. A source-bound
`ResidentBatchRows` admission contract rejects duplicate/foreign
sessions, stale frontier and overlapping row ownership without
publishing state. The actual `ResidentProgram` still operates one
session at a time and `gpu_fused_batches=0` is accurate for HTTP.

Accepted checks: 49 GPU library tests, 46 passed/3 intentionally
ignored; strict CUDA qualification Clippy `-D warnings`, frozen
workspace 1,147 passed/0 failed/69 intentionally ignored across
241 Rust suite runs, plus both approved trained Qwen3.5 0.8B/2B
direct-cache and HF-import full-logit/42-domain parity routes.
The physical tests were repeated after the lint-only refactor.

An isolated vLLM comparator is now available at
`tools/serve-compare.py`; its source-free self-test passed.
**No vLLM runtime was installed** and no source-matched external
vLLM endpoint was provided, so there is no legitimate P22 latency
or throughput comparison.

Remaining P12-critical work is batch-native *full-model*
`ResidentProgram` lowering: all operator nodes, GGUF codec
variants, coherent per-row session state publication, actual
Ready/fence receipts, concurrent cancellation, paged hybrid KV,
and source-real B2/4/8 complete full-vocabulary/StateRoot parity.
Do not enable `batch_mode=fused` or claim vLLM parity yet.

## P12-A1 — physically Ready resident multi-root publication

19 Oct 2026, RTX 3060 CUDA, original pinned Qwen3.5 0.8B.
`ResidentModel::prepare_logits_cohort` and `prepare_greedy_cohort`
reuse the accepted ONE resident GPU forward program, independent
original `ResidentSession` authorities and physical `ResidentReady`
completions. Rows share identical source/frontier, unique trusted
session IDs and valid tokens. The private `PreparedResidentCohort`
publishes NO StateRoot until its checked `commit`.

- Before publication, **all** source/service/epoch, Ready events,
  proposed state-domain lease cohorts and RootAttempts are verified.
  One `ResidentService::publish_if_usable` supplies a real
  device-fence publication barrier. ALL original root transitions
  are obtained before mutating any state. Abandoned/invalid/stale/
  foreign rows do not publish partial siblings.
- Physical CUDA Qwen3.5 B=2, B=4 and B=8 returned **exact complete
  248,320 logits and all 42 StateRoot domains per session** versus
  independent CPU sessions. B2 additionally passed dropped proposals,
  invalid-token refusal, stale-root all-row rollback and a valid
  surviving sibling continuation. B4/B8 returned private GPU state
  leases to baseline after session drop.
- The OpenAI CUDA worker uses this same cohort for source-compatible
  one-token final prefill tails. Two actual simultaneous HTTP requests
  on one resident model returned identical two-token responses and
  one checked state-cohort publication across both sessions.
  Health records `state_cohort_publications` and
  `state_cohort_rows` separately from actual GPU forwards.

**Limitation:** each row's complete model graph still launches
SEPARATELY; no fused B>1 full-model GPU execution has been proved.
`gpu_fused_batches=0` and refusal of `batch_mode=fused` are
mandatory. Next: lower physically qualified B>1 kernels through
the one resident ForwardPlan and per-row device buffers.

Repeat on a qualified NVIDIA CUDA host, with installed CUDA
toolkit/CCCL/NVRTC supplied and sufficient real memory:

~~~bash
python3 scripts/native-gpu.py -- cargo run --locked --offline   -p zerogravity-gpu-qualify --features qualification,cuda --   resident-cohort-cuda target/models/qwen35-0.8b/model.gguf   4096 5120 8
~~~

The final arguments are provider cap MiB, device cap MiB and
row count (2, 4 or 8). This is a diagnostic tool, **not** an
HTTP fused-batching enablement.

## P12-B2 — physically accepted quantized B>1 GPU projection

19 Oct 2026, RTX 3060 CUDA, original GGUF packed codecs
`Q4_0`, `Q4_1`, `Q8_0` and `Q6_K`. Every codec now has a
native B2/4/8 GPU projection kernel sharing the EXACT
original B1 dot operation. Source arithmetic includes
signed nibble/6-bit correction, dynamic Q8 per-row codes,
scales, Q4_1 offsets, ordered integer sum and original
F32 accumulation/reduction. One shared immutable packed
matrix; request-private activations.

**Physical qualifier:** each codec × B2/4/8 matched all
128/256/512 F32 output bits against the accepted B1 CUDA
kernel, with exactly ONE multi-row projection launch.
The GGUF matrices here are small deterministic *synthetic
source fixtures*, not trained full-model parity. Q8/Q4
64×64; Q6_K 64×256. In this qualification the input
dynamic quantizer prepares each row through the original
source code; actual resident batched GPU quant preparation,
full-model ForwardPlan lowering and model-state publication
remain pending.

Source-free geometry refusals and default/CUDA Clippy
`-D warnings` passed; isolated scripts/check.sh passed
1,147 Rust tests/0 failures over 241 suites and both
approved trained Qwen3.5 .8B/2B direct-cache and HF-import
regressions retained full logits/all42 state domains.
The existing OpenAI GPU serving batch mode remains
`interleaved`; `gpu_fused_batches=0` is still correct.

Physical command on a qualified CUDA host, with installed
CUDA runtime, CCCL and NVRTC libraries already configured:

~~~bash
python3 scripts/native-gpu.py -- cargo run --locked --offline   -p zerogravity-gpu-qualify --features qualification,cuda --   batch-q8-projection-cuda
~~~

The original qualifier name is retained for compatibility,
but now covers Q8_0, Q4_1, Q6_K and Q4_0.

## P12-B3 — batch-native resident ForwardPlan

Implementation: `35ebc5b`, 19 Oct 2026. RTX3060 Laptop CUDA;
approved original 0.8B/2B mixed GGUF inputs only.

`ResidentModel::prepare_logits_fused_cohort` lowers the original
source-bound ForwardPlan in ONE traversal over CUDA B2/B4/B8.
Embedding, native source-codec projections, dynamic activation
quantization, norm, RoPE, convolution, GDN, attention and FFN
operate on batch-shaped values. Original reduction/rounding and
attention padding stay unchanged. Immutable tensors and numerical
tables share the original source handles; warm forwards upload no
model or activation values.

- Independent session prior/candidate BufferLeases remain the state
  owners. Admitted packed state is temporary scratch; KV gathers
  and scatters preserve each private source/epoch/root payload.
- One real completion validates device status and logits, then
  creates exact ordered per-row Ready proofs. Existing cohort commit
  checks ALL RootAttempts before publishing any row. Unknown
  completion uses the existing physical fence/quarantine owner.
- Copy/certificate work is chunked under the original bounded ABI,
  including B8 vocabulary and lifted KV buffers. `dispatches` counts
  actual launch sites; `fused_dispatches` conservatively counts only
  native multi-row projection launches.

Approved trained mixed GGUF inputs include Q4_0/Q8_0 (0.8B) and
Q4_0/Q4_1/Q8_0/Q6_K (2B), alongside F32. These are the original files
pinned by the cache-direct/HF-import parity fixtures, not new
requantizations. Physical RTX3060 CUDA acceptance (two decode
steps, distinct two-token prior histories; every row exact CPU AND
independent B1):

| Original source | B | Full logits/row | State domains/row | Batch launches | Separate B1 launches | Native projection launches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 0.8B | 2 | 248320 | 42 | 11442 | 22024 | 374 |
| Qwen3.5 0.8B | 4 | 248320 | 42 | 11994 | 44048 | 374 |
| Qwen3.5 0.8B | 8 | 248320 | 42 | 13076 | 88096 | 374 |
| Qwen3.5 2B | 2 | 248320 | 42 | 10782 | 20704 | 374 |
| Qwen3.5 2B | 4 | 248320 | 42 | 11334 | 41408 | 374 |
| Qwen3.5 2B | 8 | 248320 | 42 | 12416 | 82816 | 374 |

Both original models passed release B2 CPU-hydrated prefixes0/255/511
with native decode at valid1/256/257/512: complete logits/all42 domains
exact against CPU and independent B1, canonical future KV positive
zero, exhausted512 refusal, unchanged roots and final reservation0.
Crossing valid256→257 adds exactly144 fused versus288 separate B1
launches, preserving source attention padding. This does NOT qualify
full-prefix GPU prefill.

Both release B2 abandonment tests dropped after an actual native
full-model launch: completion stayed unknown; all1820 original
0.8B handles and all1715 original2B handles were quarantined. Both
original roots/all42 prior domains and KV tails stayed exact.
Previously completed Ready, subsequent steps
and a new same-card model were refused; the original4294967296-byte
envelope stayed charged/fenced after user owners dropped. Diagnostic
sync/readback did not constitute recovery or clear the quarantine.

Verified gates: source-free GPU library51 passed/0 failed/3 ignored;
strict default GPU, CUDA qualification and CUDA/Vulkan serving
Clippy `-D warnings`; isolated `scripts/check.sh` exit0 with1147
Rust passed/0 failed/69 ignored across241 suite runs,18 Python
fixtures, release CLI/installer and Nix formatting. Both approved
0.8B/2B original direct-cache and HF-import release parity routes
passed complete logits/all42 domains, source/lease/mutation checks.
`tools/serve-bench.py` and `tools/serve-compare.py` self-tests passed;
the comparator's first private-namespace invocation had loopback
down and was rerun successfully after guarded private lo setup.
No local vLLM installation or supplied reference endpoint; no matched
benchmark.
Launch counts establish physical fusion, not optimized latency
or throughput.

Repeat with the already installed CUDA/CCCL/NVRTC environment:

~~~bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo run --locked --offline -p zerogravity-gpu-qualify \
  --features qualification,cuda,hardware-abort-probe,portable-checkpoint -- \
  resident-fused-cuda target/models/qwen35-0.8b/model.gguf 4096 5120 2
~~~

For release boundary/abandonment checks, use the same feature flags
and caps with `--release`, replacing the subcommand/row argument by
`resident-fused-boundary-cuda PATH 4096 5120` or
`resident-fused-abort-cuda PATH 4096 5120`. Each uses B2; run abort
in its OWN process because unknown completion intentionally retains
that process's physical quarantine until exit.

P12-B3 accepted a same-frontier, full-logit diagnostic. Additional trained
quantization profiles, unequal prefixes, native per-row greedy/EOS
feedback, paged hybrid KV, continuous HTTP integration and the
matched vLLM performance gate were still open at that checkpoint;
P12-B4 below supersedes the engine greedy/feedback gap. `batch_mode=fused`
continues to refuse; HTTP `gpu_fused_batches=0` remains accurate.

## P12-B4 — native greedy packets and device feedback

Implementation: `54fd580`, 19 Oct 2026. RTX3060 Laptop CUDA;
approved original0.8B/2B mixed GGUF profiles only.

`prepare_greedy_fused_cohort` executes the same complete batch plan,
then performs native per-row reduction over the original vocabulary.
The original finite-bit ordering, lowest-ID tie rule and sticky graph
status govern selection. For V248320, four native sampler stages
cover every row; B original three-word copies transport the results
into independent 12-byte packet leases. Status plus packets require
4+12B readback bytes; full logits remain device-resident.

Each private packet joins its row's candidate-state Ready outputs.
Only the existing checked all-row commit stamps its producing
RootBinding. `prepare_device_tokens_fused_cohort` requires that exact
source/session/epoch/root/frontier and initialized original physical
lease, then gathers packet first words in one GPU dispatch. Public
host ID/max fields do not select embedding rows. Previous-root,
foreign-row and restored-epoch packets must refuse before launch.
The original flight pins input packets read-only and output packets
for writing; unknown completion retains them with state and source
handles behind the existing quarantine and charged physical envelope.

Both approved original trained mixed-GGUF models passed RTX3060 CUDA
B2/B4/B8 over six source steps: two host seed steps and four original
GPU-packet feedback steps, following distinct two-token histories.
Independent CPU and B1 GPU compare all248320 logits and all42 state
domains. A separate native full-logit shadow verifies the complete
vocabulary; compact greedy calls compare exact selected ID/max bits
and all42 domains without transferring their complete logits.
These counts cover the six committed greedy steps only:

| Original source | B | D2H bytes/cohort | Total D2H bytes | Total physical launches |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 0.8B | 2 | 28 | 168 | 34362 |
| Qwen3.5 0.8B | 4 | 52 | 312 | 36030 |
| Qwen3.5 0.8B | 8 | 100 | 600 | 39300 |
| Qwen3.5 2B | 2 | 28 | 168 | 32382 |
| Qwen3.5 2B | 4 | 52 | 312 | 34050 |
| Qwen3.5 2B | 8 | 100 | 600 | 37320 |

Each six-step run issued1122 actual native projection launches and24
native sampler stages for the entire cohort. Per-row packet copies are
included in total physical launches, separately from native stages.
No warm model/value uploads; private packets, states and events
reclaim, and the final physical reservation returns to0.

Refusal probes cover swapped/duplicate packets, identical-ID packets
from a foreign row, same-epoch old roots with fresh sibling packets,
restored epochs, invalid host IDs, reordered commits and abandoned
completed proposals. Changing public host ID/max metadata does not
change device feedback. B1's actual greedy packet independently
matches CPU/B1 full-logit selection and complete state at every step.

Separate full-vocabulary native selector fixtures cover lowest-ID
ties across tile boundaries, signed zero, subnormals, dense finite
bits and a genuine tokenizer EOS ID as a private synthetic winner.
One-row NaN and preexisting graph error status invalidate every row;
aliased packet allocations refuse before launch. Invalid fixture
records never become ResidentTokens or published roots. A trained
model EOS input is source-exact; trained EOS generation/HTTP stop
handling is not established by the synthetic winner fixture.

Both release B2 feedback-abandonment probes dropped after actual
model AND sampler launches. All1827 original0.8B handles and1722
original2B handles, including incoming/outgoing packets, remained
quarantined. Incoming packet bits, all42 independently computed CPU
prior domains and canonical future KV zero tails stayed exact.
The raw verifier matches unique StateSlots across CPU layer order
and physical slot order; it rejects missing/duplicate domains and
reads every original state lease's full byte range. Roots stayed unchanged;
old completed Ready, further work and new same-card service admission refused.
The original4294967296-byte envelope stayed charged/fenced after user
owners dropped. Diagnostic sync/readback did not establish recovery.

The full-logit B2 boundary regression passed again for both models:
CPU-hydrated prefix0/255/511, native valid1/256/257/512, all248320
logits/all42 domains exact against CPU and B1, positive-zero KV
tails, exhausted512 refusal and final reservation0. This qualifies
native decode from imported prefixes, not full-prefix GPU prefill.

Verified gates: GPU library53 passed/0 failed/3 ignored; strict
default GPU, CUDA qualification and CUDA/Vulkan serve+cache Clippy
`-D warnings`; final isolated `scripts/check.sh` exit0 with1147 Rust
passed/0 failed/69 ignored over241 suite runs,18 Python fixtures,
release CLI/installer and Nix formatting. Both approved original
0.8B/2B cache-direct and HF-import release routes passed complete
logits/all42 domains/source leases/mutation checks. Benchmark and
comparator self-tests passed. No vLLM executable/module is available
in the checked environment; no matched serving benchmark or
performance acceptance is claimed.

Acceptance commands use the existing installed CUDA/CCCL/NVRTC
environment and the same original model paths/caps as P12-B3:

~~~bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo run --release --locked --offline -p zerogravity-gpu-qualify \
  --features qualification,cuda,hardware-abort-probe,portable-checkpoint -- \
  resident-fused-greedy-cuda target/models/qwen35-0.8b/model.gguf 4096 5120 2
~~~

Use B2/B4/B8 for both approved original models. The separate-process
`resident-fused-greedy-abort-cuda PATH 4096 5120` probe uses B2 and
intentionally retains unknown-completion quarantine until exit.

Greedy is an engine selection operation. Request EOS/stop policy,
row retirement/refill, unequal frontiers, paged hybrid state and
continuous HTTP integration retain their separate acceptance gates.
Do not enable `batch_mode=fused` or infer serving throughput from
these sampler/dispatch counts.

## P12-B5 — native CUDA B2 HTTP waves

Implementation: `2677d8b`, 19 Oct 2026, RTX3060 Laptop CUDA; the same two approved original
mixed-GGUF sources as B3/B4. The existing GPU worker now feeds
matching-source/phase/frontier pairs into the original native model
and greedy methods. Each request retains its independent authority,
root, KV/GDN leases, Ready receipt and producer-bound device packet.
One all-row checked publisher remains the only cohort commit path.

- Multi-token prefill remains B1 in ≤32-token quanta, preserving the
  last prompt token for native B2 when a compatible sibling exists.
  Unmatched rows use the original B1 GPU path; no CPU inference fallback.
- A committed packet is presented once before feedback. EOS/length
  rows retire, canceled rows drop and output-credit-blocked rows park
  BEFORE wave formation. A parked row retains its original packet/root
  and a bounded two-event pending output; it owns no submitted command,
  queue slot or workspace guard. A 30s credit deadline bounds stalls.
- Cancellation after actual Ready discards every private candidate
  before publication. Survivors retry from original roots/packets,
  retaining presentation progress; no repeated text, usage or terminal.
- Fused mode requires CUDA and ≥2 active slots. CPU/Vulkan/HIP,
  active1 and unknown modes refuse before model/device lookup.
  Native mode currently dispatches pairs even with >2 active requests.
  Default interleaved behavior remains available.

Internal RELEASE worker tests for BOTH models physically proved:
B1-produced→native-consumed and native-produced→B1 packet handoffs,
unequal output limits2/5, cancellation before feedback and after Ready,
a real capacity1 output channel parked across sibling progress, and
recovery without duplicate events. Compact token ID/max bits and all42
state domains matched independent original CPU AND B1 at each checked
frontier. A bounded source-real CPU search found trained EOS; actual
GPU replay proved native final-prompt execution, stop/EOS-excluded
usage, all42 parity and an independent continuing sibling. No synthetic
EOS fallback can pass. Final active0/reservation0, device unfenced.
The separate B3/B4 diagnostics establish complete248320-logit parity;
B5's HTTP wire checks establish exact decoded output and usage.

Actual owned HTTP servers used two aliases of ONE resident source,
real JSON/chunked SSE requests, distinct output limits, genuine EOS
through both wire formats, active disconnect, survivor and fresh replay.
Both tests passed with one terminal/usage/Done, final active/queued0
and failed GPU quanta0. These cumulative counters include three canceled
prefill gate requests and the disconnected target; they are not a latency
benchmark:

| Original source | Completed native B2 cohorts | Native projection launches | Native sampler stages | Discarded Ready cohorts | Total completed kernel launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 0.8B | 9 | 1683 | 36 | 1 | 1155210 |
| Qwen3.5 2B | 8 | 1496 | 32 | 1 | 1080879 |

Health counter meanings:

- `completed_forward_kernel_dispatches`: actual completed launches,
  including private Ready cohorts later discarded, counted once/cohort.
- `gpu_fused_batches`/`gpu_fused_rows_completed`: actual completed native
  full-model cohorts/rows, not scheduler membership or committed output.
  `native_projection_dispatches_completed` counts projection-dot launches,
  excluding their activation-quantization, staging and status kernels;
  `native_sampler_dispatches_completed` counts the four native selector
  stages, excluding private packet copies.
- `completed_model_token_steps`: physically completed model work,
  including discarded work; one B1 API call may prefill multiple tokens.
  `gpu_forward_calls_completed` counts API calls; legacy
  `gpu_forward_steps_completed` retains that meaning.
- `committed_gpu_roots`/`state_cohort_publications`/`state_cohort_rows`:
  actual root transitions and all-row logical publications.
- `completed_gpu_quanta`: request work including terminal presentation;
  `co_scheduled_members`: scheduler membership only.
- `accepted_output_tokens`: staged committed-packet output, not transport
  delivery, acknowledged committed goodput or an SLO metric.
  `output_credit_parked_requests` exposes parked rows separately.

Repeat physical worker tests with the installed CUDA/CCCL/NVRTC environment:

~~~bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --release --locked --offline -p zerogravity-cli \
  --features serve,cuda,cache --bin zerogravity \
  serve::gpu_engine::qualification:: -- --ignored --nocapture --test-threads=1
~~~

HTTP tests need loopback UP inside the private network namespace. Use
`scripts/check.sh`'s guarded private-loopback setup in that SAME namespace
before the following command; never change a host interface:

~~~bash
cargo test --release --locked --offline -p zerogravity-cli \
  --features serve,cuda,cache --test serve_gpu \
  -- --ignored --nocapture --test-threads=1
~~~

Verified B5 gates: strict CPU/CUDA/Vulkan serve+cache and CUDA-qualifier
Clippy `-D warnings`; CPU serve14 passed, GPU53 passed/3 intentionally
ignored. Full isolated `scripts/check.sh` exit0:1147 Rust passed/0 failed/
69 ignored across245 suite runs,18 Python fixtures, release CLI/installer
and Nix formatting. Both approved original0.8B/2B cache-direct and
HF-import RELEASE routes retained full-vocabulary/all42/source-lease/
mutation/reclamation parity. Root README bytes and source hashes remain
unchanged; no AMD launches, device changes, recovery or publication.

### Local release tool smoke; no matched performance acceptance

`tools/serve-bench.py` ran against actual owned fused CUDA servers.
One separate B1 warmup request preceded four measured requests at
concurrency2/max4, using `Say hello.` through the chat route. New
processes reused existing source/CUDA caches; native-plan/JIT work
may occur in the measured run. This is a tiny correctness/tool smoke,
not cold/warm performance qualification:

| Original source | Offered/completed | Usage-reported tokens | TTFT p50/p95 seconds | SSE chunk p95 seconds | Sampled total-device peak MiB | Completed kernel delta | Native cohort delta |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Qwen3.5 0.8B | 4/4 | 16 | 1.771/10.851 | 0.294 | 1513 | 265376 | 8 |
| Qwen3.5 2B | 4/4 | 16 | 2.083/6.823 | 0.396 | 2235 | 249616 | 8 |

Memory samples include startup/warmup/measurement and are not exact peaks
or ResourceArbiter reservations. SSE chunk intervals do NOT establish
true token ITL. SLO goodput, fairness and performance remain unqualified.
`tools/serve-compare.py --self-test` passed; no vLLM executable/module or
matched external endpoint exists in the checked environment. No models,
SDKs or vLLM packages were downloaded; no vLLM comparison is claimed.

## P12-A — batch-native parameter, source and state bindings

1. Introduce a typed `ResidentBatchStep` with finite row count
   (start B=2), one verified model/quant/plan/dtype per source
   group, per-row independent session/epoch/root/frontier and
   token or device-greedy packet. Reject duplicate session IDs.
2. Allocate real batched rows with explicit strides, offsets,
   capacity and typed read/write domains. Validate all overlaps
   and source identity *before* physical admission and queueing.
3. Use existing ResourceArbiter/BufferLease/ResidentService
   for weight, scratch, prior and candidate state. Completion
   terminal event or fence must precede any allocation reuse.
4. Commit each row's logical StateRoot only on qualified actual
   device completion, with correct cancellation/epoch guard.
   One bad row must not mutate successful sibling sessions.

## P12-B — native kernels (not a host loop)

1. Launch batched F32 projection on B=2 independent input rows
   in **ONE physical kernel**; compare bit-exact or bounded
   source-qualified reduction against accepted B=1.
2. Extend source GGUF codec projections, RMSNorm, SiLU/binary,
   RoPE, GDN recurrence/coefficients, causal QK/softmax/AV and
   final output/greedy into the same ForwardPlan order, using
   typed row-state bindings and per-row finite/status results.
3. Never concatenate mutable sessions into unowned shared KV.
   B2/4/8 tests must compare full-vocab logits, every KV/GDN
   StateRoot domain, source leases, EOS, unequal prefixes,
   cancellation, stale epoch and per-row rollback.
4. Compact device-side greedy feedback must remain row-bound
   without full-vocab host readback, retaining terminal buffer
   leases until the consuming next step is qualified.

## P12-C — true continuous batching and paged hybrid KV/GDN

- Source-safe paged/copy-on-write KV, sliding state and GDN
  recurrence, cache prefix reuse only when all causal state
  domains match the same verified source and epoch.
- Physical row/token/byte admission and fair chunked prefill
  versus decode with aging. No artificial memory credits or
  global model lock on the token-critical path.
- Per-row queued/admitted/issued/completed/canceled/rejected
  populations and physical Ready/fence receipts. A failed
  GPU batch cannot falsely publish a partial output.

## P22 — vLLM-equivalent validation

- Same pinned original model/quant/tokenizer, device/driver,
  prompt distribution, numeric policy and generation sampling
  as vLLM V1; non-equivalent model shapes flagged explicitly.
- B={1,2,4,8} initially: offered/committed tokens, p50/p95
  TTFT, **true token ITL**, SLO goodput, fairness, memory
  peak/evictions and physical dispatch counts (not wave counts).
- Compare warm/cold repeated ABBA/BAAB workloads with all
  rejected/canceled work included and uncertainty recorded.
- Enable a backend/profile's fused path only after actual B>1 native
  dispatch and source-exact Root/Ready acceptance. P12-B5 qualifies
  CUDA B2 for the two approved sources; other backends refuse fused.
  P12-B6/B7 additionally qualify CUDA B4/B8 and unequal decode frontiers.
  A fused flag or fewer launches does not establish vLLM performance.

Changing a scheduler counter to "fused" without native multi-row
execution is forbidden. P12-C/P22 remain pending; the sections
above delimit the physically accepted operator and model scopes.

## Native CUDA B4/B8 HTTP acceptance — P12-B6

Actual RTX3060 release worker tests pass both original0.8B/2B × B2/4/8
in one process: CPU/B1 exact compact packets/all42 domains, limits1..rows,
B1/native handoffs, post-Ready cancellation/regroup, capacity1 output
parking/recovery and actual trained EOS. All end at reservation0.

Four actual HTTP tests pass JSON/SSE text/usage against the independent
CPU oracle, real trained EOS, live disconnect, survivors/replay and one
terminal. Each width4/8 wave uses actual queued source aliases and proves
native completion before disconnect. Final active/queued/failed0:

| Original model | Configured maximum rows | Completed cohorts B2/B4/B8 | Native projections | Native sampler stages |
| --- | ---: | ---: | ---: | ---: |
| 0.8B | 2 | 9/0/0 | 1683 | 36 |
| 0.8B | 4 | 8/7/0 | 2805 | 60 |
| 0.8B | 8 | 10/10/7 | 5049 | 108 |
| 2B | 2 | 8/0/0 | 1496 | 32 |
| 2B | 4 | 9/5/0 | 2618 | 56 |
| 2B | 8 | 11/11/5 | 5049 | 108 |

Counters include physically completed Ready cohorts discarded before
publication. They are not delivered-token throughput. HTTP wire tests do
not expose private StateRoot bytes; exact all42 evidence is internal.

Actual lease pressure separately proves checked capacity refusal before
submission/allocation, unchanged prior roots/packets/idle workspace,
B4 progress under pressure and B8 after release, full248320 CPU/B1/native
logits/packets/all42 exact. Only typed prelaunch None permits smaller
bands; other errors preserve original completion/fence/quarantine.

The first combined worker run exposed retained idle CUDA allocator pools
and failed OOM. Cold model initialization now drains completed pools under
the original queued guard; live/graph-pinned and quarantined owners remain
retained. Actual telemetry showed retired2.51GB/2.13GB pools return to0;
repeated same-process six-case worker suite passed. Telemetry is advisory,
not the ResourceArbiter admission authority.

Strict CPU/CUDA/Vulkan and qualifier Clippy pass. Final isolated gate:
1148 Rust passed/0failed/69ignored over245 suites plus18Python, release,
installer and Nix checks. Source-free CLI26/GPU53 pass. Both approved
cache-direct/HF-import release routes retain full-logit/all42 parity.

Genuine pinned vLLM0.31.0-cu129 V1 executes the original HF0.8B checkpoint
on this card: weights, Torch compile, CUDA graphs, health and warm smoke.
Its stock TorchCodec CUDA13 import issue was resolved by an isolated,
reproducible CPU-TorchCodec wheel replacement; core inference package
versions/RECORDs remain identical. Source arithmetic, tokenizer policy,
caps, true token observations and matched performance remain unqualified.

Example wider native configuration adds `--batch-rows 8` and
`--active-per-worker 8` to the accepted CUDA command. This phase still
requires equal causal frontiers; other prefill/unmatched work uses B1.

## Unequal-frontier native CUDA decode — P12-B7

Both approved original0.8B/2B models pass B2/B4/B8 complete ForwardPlan
execution at independent row positions255/256/0/31/32/1/510/511. Native
RoPE/KV append, masking, packed softmax and value reduction retain each
session's own causal prefix. Inactive attention-value lanes preserve bits
without arithmetic on poisoned, signed-zero or subnormal future values.
Independent CPU AND B1 comparisons cover all248320 logits/all42 state
domains, device feedback, B1 handoffs/refill, stale/foreign packets,
abandoned Ready, capacity512 and final reservation0. No separate model
graph is executed per row. Full-prefix native prefill remains open.

Fresh composed release tests:8 worker cases and6 actual HTTP cases pass
both models B2/4/8, trained EOS, limits, cancellation/credit parking,
disconnect/survivors/replay and one terminal/usage/Done. The new arrival
cases offer a short request only after the long request's first actual
ID, then observe one successful native publication sequence shared by
rows at different producer frontiers. Each model records5 native B2
cohorts/10 rows,935 native projection launches and20 native sampler
stages. Internal tests retain exact CPU/B1 packets/all42; wire checks
compare exact CPU IDs/text/usage and bounded observation clocks.

Opt-in `--enable-per-request-metrics` and `return_token_ids:true` expose
at most512 once-only non-EOS IDs, independent of UTF-8 chunk emission.
Commit-observed, output-queue-received and socket-write-completed clocks
have distinct names. The commit observation follows original root
publication and survives credit parking; GPU frontiers and existing
successful cohort sequence/width identify the physical producer.
No metric creates a session authority or changes default responses.
Standard vLLM scheduler-origin TTFT/queue/rate fields stay unset where
that origin is unavailable. Benchmark true client ITL requires one actual
ID per observed arrival; grouped arrivals are unresolved. Provider mean
ITL and SSE chunk intervals remain separate from true client token ITL.

Independent source audit reconciles the0.8B unequal-arrival HTTP launch
count exactly:26 B1 state-only graphs*5489 +6 B1 greedy*5510 +5 nativeB2
cohorts*5728 =204414. The counter includes actual private certification,
state gather/scatter and packet-copy launches, excludes cold model
initialization/maintenance and counts each native cohort once. A native
projection uses10 actual launches including activation preparation and
the dot, before output certification, for this mixed Q4_0/Q8_0 source.
The935 projection-dot launches are the named subset of those kernels.
These costs remain real
optimization work; fewer model traversals do not establish performance.

Fresh pressure cases pass typed prelaunch B8 refusal with unchanged
allocations/queue/roots/all42, B4 progress under pressure and B8 after
release. Four actual full-graph/sampler completion-failure probes retain
1824/1719 and1828/1723 original handles in quarantine; prior roots/all42
and future KV zero tails stay exact, old Ready/further work/other model
refuse and the4GiB envelope remains charged. No recovery performed.
An initial wrong executable-name invocation launched no abort work;
corrected invocations passed without repeating accepted pressure tests.

Complete isolated gate:1164 Rust passed/0failed/69ignored/245 suites,
18Python, release/installer/Nix. Strict CPU/CUDA/Vulkan/qualification
Clippy passes; fresh release CLI35/GPU58 source-free tests pass. Both
approved original cache-direct/HF-import release routes remain exact
full248320/all42 for both models. The earlier full-logit ragged engine
binary predates metadata/observability composition; fresh worker/HTTP/
pressure/abort acceptance above covers the composed implementation.

P12-C/P22 remain open: paged hybrid state, full-prefix native prefill,
physical prefill/decode overlap and measured fairness, other trained
profiles/backends, a common original-HF numerical profile and matched
pinned-vLLM cold/warm workloads. Genuine vLLM startup/smoke is accepted;
source-correct SLO goodput and vLLM-equivalent throughput are unqualified.


## P22 owned process isolation

The comparator can own a strict foreground provider lifecycle: one fresh
process per ABBA/BAAB sample, explicit warmup policy, bounded logs/readiness/
shutdown, loopback vacancy, immutable owned Podman CID and cleanup before
receipt writes. Uncertain cleanup blocks the next provider. Real accepted
ZG GGUF CUDA and pinned-vLLM original-HF BF16 startup/one warm request/
shutdown passed sequentially; both returned device telemetry to41MiB.
Live1059/4203MiB samples are neither peaks nor ResourceArbiter reservations.
Sources/quant/arithmetic differ; this is no matched benchmark. Fresh-process
coldness is separate from OS/storage/device cache conditions. The independent
8K GGUF GDN issue also remains a required correctness gate for both models.


## P12 block-local source activation preparation

The same B1/native projection path now binds original scratch views cold and
prepares dynamic source Q8 blocks in one CUDA32-lane workgroup/block. Lane0
retains the original finite check, ordered first signed maximum, exact source
scale/division and half rounding. Lanes retain original scaled-word/code
rounding and code-sum semantics. Composite scale, sticky status merges and
final source dot remain separate. No new persistent device buffer, upload,
lease, queue, publication or completion authority is introduced. Non-CUDA
selection is explicit; CUDA errors cannot silently choose another lowering.

Each Q4_0/Q4_1/Q8_0 projection uses5 actual launches versus10; Q6_K uses5
versus8, before output certification. The actual pressure B8 graphs report
5616/5472 total launches for0.8B/2B. These are kernel execution counts, never
one count per full graph or a throughput inference.

Physical CUDA acceptance on the RTX3060 Laptop:

- 272 source-preparation cases: all7 buffers/1,483,608 words exact.
- Both original models B2/4/8, unequal frontiers, full248320 logits/all42
  domains exact against independent CPU AND B1; host/device feedback,
  refill, abandoned Ready, stale/foreign packets and final reservation0.
- Two pressure cases: real admitted leases, cold/warm prelaunch refusal,
  B4 progress under pressure, B8 after release, exact full state and logits.
- Eight worker and six actual HTTP cases pass B2/4/8, unequal arrivals,
  trained EOS, credit parking, Ready cancellation, disconnect/survivors/replay
  and final active/queued/failed/reserved0. Wire checks do not expose state;
  the internal tests establish all42-domain equality.
- Four fresh unknown-completion graph/sampler probes retain1824/1719 and
 1828/1723 original handles in quarantine, unchanged prior roots/all42 and
  canonical future KV. Old Ready/further work/another model refuse; the
 4GiB envelope remains charged and no recovery occurs.

Current complete isolated check:1218 Rust passed/0failed/69ignored over248
suites plus18Python/release/installer/Nix. Strict CUDA/Vulkan/qualifier checks
and both approved original cache-direct/HF-import release routes pass.
Same-engine cold/warm release A/B observations are now accepted below;
source-correct SLO goodput and genuine matched vLLM performance stay open.
The first A/B invocation used equal4GiB provider/device limits, leaving no
unreserved global staging capacity. Its native refusals correctly prevented
acceptance. Repeats use the already accepted physical HTTP limits4GiB/5GiB
for BOTH providers, with the original numerical/native/no-refusal gates.

Full-prefix native prefill, paged hybrid state, overlap/fairness, original-HF
math and the independent two-model8K issue remain separate required gates.


The first valid warmed0.8B ABBA measurement completed32/32 offered requests,
256 output tokens, with identical per-request IDs/text/usage/finish reason
across the two builds. Each sample executed8 native B8 cohorts with zero
refusal/failure/cancellation. Four CPU affinity members, exact original GGUF/
tokenizer, greedy generation/output8/context512 and4GiB provider/5GiB device
limits were common; processes started/exited serially and returned to41MiB.

| Mean over two samples/provider | Accepted B7 | Block preparation |
| --- | ---: | ---: |
| Eight-request wave seconds |39.14206|33.163665|
| Per-sample TTFT p95 seconds |36.4701|30.5368|
| Per-sample client token ITL p95 seconds |0.3911|0.3933|
| Actual forward kernel dispatches/sample |2072360|1722640|

This bounded wave improved15.3%; dispatches decreased16.9%. No client ITL-tail
improvement is established. Both builds miss the declared10s TTFT SLO, hence
observed delivery-SLO goodput0; source-correct goodput remains unset. Sampled
GPU memory reached3045MiB in the first old/new pair;0.5s observations can miss
peaks and are separate from admission. The two workload labels are not an
authenticated tenant-isolation proof. The remaining model/order/cold observations are recorded below;
genuine equivalent-source vLLM comparisons remain required.

The launch delta independently reconciles with the source traversal:368 B1
state-only calls plus8 B8 greedy calls. Old counts5489/6551 become4559/5616;
368*(5489-4559)+8*(6551-5616)=349720 fewer actual launches. Model-token steps
432, committed roots80 and generated output tokens64 are different quantities.


## Completed bounded activation-preparation comparison

Sixteen serialized samples cover both original models, warmed-process ABBA
(0.8B), BAAB(2B), and cold-process ABBA(both). Each model/policy has32offers/
32completed/256 committed output tokens, with identical full IDs/text/usage/
finish reason across builds;8 genuineB8 cohorts/sample, zero refusal/failure/
cancel/discard. Total128offers/128completed/1024 output tokens. Same source/
tokenizer/greedy policy/context512/output8,4CPU affinity members,4GiB provider/
5GiB device caps. Both providers stop and device telemetry returns41MiB before
the next sample. Cold means fresh process/no request warmup; OS/storage/compiler
caches are not evicted. Model readiness/startup is separate from request-wave
time. The original invalid4GiB-device attempt is retained, never averaged in.

Values below are means of the two per-provider sample summaries; the reported
p95 values are means of sample p95s, not a pooled population percentile.

| Model/process | Wave seconds old→new | Client TTFT p95 old→new(s) | Client token ITL p95 old→new(s) |
| --- | ---: | ---: | ---: |
|0.8B warm|39.14206→33.163665|36.4701→30.5368|0.3911→0.3933|
|2B warm|45.81464→43.04175|39.0828→36.3616|0.9822→0.9736|
|0.8B cold process|45.657195→39.34474|42.9849→36.7244|0.3944→0.3905|
|2B cold process|51.963075→48.845175|45.3282→42.2650|0.9618→0.9515|

Warm wave changes: -15.3%/-6.1%; cold -13.8%/-6.0%. Actual dispatches/sample
are2072360→1722640(0.8B,-16.9%) and1949016→1668496(2B,-14.4%) for BOTH
policies. No consistent token ITL-tail improvement is established. Every sample
misses10s TTFT; observed delivery-SLO goodput0 and source-correct goodput unset.
Memory observations sample maxima3045MiB(0.8B) and3509MiB(2B) on both builds;
0.5s telemetry is not an exact allocator peak/admission proof. Two workload
labels do not establish authenticated tenant fairness. These are small bounded
ZG-versus-ZG observations, not a matched vLLM benchmark or a throughput-parity
claim. Native full-prefix work and original-HF/CUDA numerical qualification
remain the next performance prerequisites.


## Native teacher-prefix serving — P12-B8

The original BatchProgram now advances one teacher token per row with
ResidentOutput::StateOnly, omitting output projection/sampling/logits and
reading only shared4B status. All42 independent candidate state domains,
physical leases, Ready events and checked atomic publication are unchanged.
Each teacher advances its own cursor/root; it produces no generated ID,
TokenText, terminal event, EOS decision or output usage. A trained EOS teacher
is ordinary input. Prefix/tail partition precedes2/4/8/1 banding; unmatched B1
native-mode rows also consume one teacher per wave. Interleaved scheduling
retains its max32 prefix quantum. Capacity refusal occurs before launch;
original errors propagate, and post-Ready cancellation discards the whole
candidate cohort before survivors regroup.

Both original sources pass release B2/B4/B8 teacher steps against independent
CPU AND accepted B1 GPU: all42 domains after every teacher, then complete
248320 future logits. Tests include unequal history, four teacher steps with
trained EOS, invalid/stale/abandoned candidates and final reservation0.
Two pressure tests prove actual B8 refusal/B4 progress/B8 after release;
two fresh unknown-completion processes retain1817/1712 original handles,
unchanged prior roots/all42,4GiB envelope and fence without recovery.
All14 serving-worker and12 HTTP scopes pass. New prefix HTTP cases use
ragged prompts, actual StateOnly cohort/row counters and exact source-bound
SSE IDs/text/usage/frontiers; prefix roots cannot appear as generated IDs.
Ragged B2 tails may finish through B1: genuine StateOnly fusion is required,
while matching-tail/worker cases separately establish greedy fusion.

Three health fields expose completed StateOnly batches/rows and greedy
batches. Total native batches equal the two readout phases; width-weighted
rows close independently; native sampler dispatches equal4*greedy batches.
Completed discarded GPU work remains distinct from committed publication.

Complete isolated check:1241 Rust passed/0failed/70ignored over248 suites,
18 Python tests, release CLI/installer/Nix pass. Strict combined CUDA/HF/cache,
qualification and Vulkan Clippy pass. Approved original cache-direct22.20s
and HF-import29.94s release routes pass both models/fullV/all42/source/reclaim.
Initial ragged-fixture and unrelated greedy-counter assertions were repaired
only in tests; raw failed attempts remain preserved. A final launch missing
CUDA_PATH refused before GPU execution; the identical frozen artifacts passed
with the approved environment. No source math/tolerance change.

Frozen production server b1846c3e was built from5350a4d plus this reviewed
prefix overlay. This is one native teacher
per row/wave, not simultaneous multi-token prefill or matched vLLM performance.

## Bounded native-prefix warm observations

Quiet serialized release measurements compare frozen b1846c3e with accepted
activation-preparation server a896a455 (actual c5eb1f8 plus reviewed overlay,
landed d0a6084). Same original GGUF/tokenizer, greedy policy, context512,
output8, four CPU affinity members, provider4GiB/device5GiB and fresh owned
process per sample with one request warmup. OS/storage/compiler caches remain
warm; request time excludes startup/warmup. All16 samples close against exact
IDs/text/usage/finish/frontiers, actual provider counters and owned process/
endpoint/reader cleanup, followed by idle device observations. Total72offers/
72completed/576 measured output tokens; warmup work is separate.

0.8B uses strict B8 ABBA plus B1; report bd3a77d4. Its concurrent samples have
46 native StateOnly B8 passes and8 greedy B8 passes,432 model-token steps and
432 independent root publications versus368 B1 teacher steps plus8 B8 greedy
passes/80 root publications. Dispatches1722640→302160(-82.5%).

The first2B strict-B8 attempt failed its predeclared scheduling guard after
actual mixed2/4/8+B1 work; report5da6dc90 remains unqualified and excluded.
A separately reviewed naturalC8 protocol accepts whatever2/4/8+B1 scheduling
actually occurs and authenticates every output packet/cohort; no forcedB8
condition or invented StateOnly membership. Its BAAB plus B1 report f6869712
passes. Both measured new concurrent samples happened to use46 StateOnly B8
and8 greedy B8 passes,432 steps/roots and294384 dispatches versus368 B1
teacher steps plus8 B8 greedy passes/80 roots and1668496 dispatches(-82.4%).

Each value below is the mean of two per-provider summaries; p95 means sample
p95s, not pooled population percentiles. These are small bounded observations.

| Model/workload | Wave seconds old→new | Client TTFT p95 old→new(s) | Actual ID-delivery ITL p95 old→new(s) |
| --- | ---: | ---: | ---: |
|0.8B B8|32.301205→18.33249|29.6884→15.7265|0.3851→0.3832|
|2B naturalC8|42.19858→28.457115|35.6887→21.9980|0.9380→0.9312|
|0.8B B1|4.43954→4.42998|3.8142→3.7957|0.1043→0.1007|
|2B B1|5.60352→5.652635|4.5128→4.5488|0.1651→0.1679|

Concurrent mean waves decrease43.2%/32.6%; B1 changes-0.2%/+0.9%, with
unchanged246314/238538 dispatches. No consistent ITL-tail gain is established.
Concurrent samples still miss10s TTFT (delivery-SLO goodput0); all B1 samples
meet declared10s TTFT/2s delivery-ITL limits. Qualified source-correct goodput
remains unset. B1 new one-teacher scheduling makes54 calls/roots versus old10,
while preserving54 model steps and8 generated tokens. This exposes its host
scheduling cost; finite noisy samples do not establish no-regression broadly.

Sampled GPU maxima old→new3045→3043MiB(0.8B),3509→3507MiB(2B);
owned RSS maxima1654288→1650324KiB and2719732→2719836KiB. Sampling can
miss allocator peaks and does not establish physical RAM admission. Two
workload labels do not establish authenticated tenant fairness. Cold-process
prefix measurements, original-HF/CUDA numerical qualification and genuine
matched vLLM V1 benchmarks remain open. No vLLM speed/parity claim.

## Private CUDA graph and ordered RoPE qualification

20 Oct 2026, original RTX3060/CUDA, original0.8B/2B GGUF sources.
Production graph selection remains unavailable.

- One ordered native angle kernel replaces the previous per-angle launches,
  preserving F32/F64 recurrence and exact signed bits. Each model passes170
  physical cases/28296 words, B1/2/4/8, half-width1..32 and positions through32768;
  trained RoPE stress passes separately. Full native B2/4/8 normal and greedy
  decoding matches independent CPU/B1 full248320 logits and all42 domains.
  This removes186 launch sites per complete cohort; no speed measurement follows.
- The original actor captures the native full-model body with original buffer
  owners. Typed graph keys bind source/program generation, band/capacity,
  AV-padding profile and StateOnly versus Logits body. Greedy sampling stays
  outside the captured Logits body. Host-supplied teacher input is qualified;
  device-feedback graph replay is refused before admission.
- Six normal cases cover both models×B2/4/8 at capacity512/AV-padding256, unequal
  histories, two StateOnly teachers including genuine EOS input, independently
  checked future full logits and outside-body greedy output. CPU/B1/normal native
  comparisons cover all42 domains; pressure, abandoned Ready, stale epoch and
  final reservation0 pass. Recorded body launch sites: StateOnly4323/4335/4347
  (0.8B) and4179/4191/4203 (2B); Logits4334/4346/4359 and4190/4202/4215.
- Eighteen fresh unknown-completion processes cover both models×three bands×
  StateOnly/Logits/Greedy. Each actually enqueues a graph, retains executable
  and original4GiB envelope, leaves roots unchanged and refuses later work.
  No post-abort readback, driver destruction or trusted recovery is attempted.
  The32-case physical result87b85eff also includes native B4/B8 regressions.
- Physical global capture exclusion and graph/driver metadata admission remain
  UNQUALIFIED. Original queue host-protocol14 cases and three intended ownership
  compile failures pass only in an isolated checkout. Optional transport and
  prepaid nonblocking retirement integration must cover every constructor,
  allocation, event/storage and final Drop before production admission.

Combined strict CUDA/Vulkan, complete isolated1243-test phase, both original
release parity routes,14 worker and12 HTTP CUDA regressions pass; composed
result e76248e8. Body counters record source launch sites, not driver-enumerated
graph nodes. Production graph admission, trained-HF/27B execution, matched vLLM
throughput and performance acceptance remain open.

````

## Source: .agents/plans/serving-and-speculation/README.md

Original UTF-8 bytes: 13805; SHA256 `ef95d4b041b61639b0c8c8d78a61664e89fff63c90a05a5efc9c466dd1a50715`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# Native serving and request scheduling

Current P12-B7: opt-in CUDA B2/B4/B8 HTTP prompt-tail/decode waves execute the
original native full-model ForwardPlan. Same-source eligible decode
rows share immutable weights; each retains its own SessionAuthority,
StateRoot, causal frontier, device packet and physical leases. Unequal
decode frontiers use native row-specific RoPE/KV/attention. Unmatched rows and
multi-token prefill use the accepted B1 GPU path. Default batch mode
remains `interleaved`.

Approved original Qwen3.5 0.8B/2B passed real release JSON/SSE,
trained EOS, unequal output limits, disconnect/survivor/replay and
internal CPU/B1 exact packets/all42 state domains. The earlier native
B2/B4/B8 engine diagnostics retain full248320-logit acceptance;
HTTP qualifies native B2/B4/B8, including actual later-arriving requests
sharing a native cohort at different frontiers. Opt-in bounded token-ID,
publication and delivery observations preserve distinct clock boundaries.
Paged state, full-prefix native GPU prefill, original-HF numerical
compatibility and matched vLLM performance remain open.
Exact scopes: [TODO.md](TODO.md), [FUSED_GPU_BATCHING.md](FUSED_GPU_BATCHING.md).

## Implemented: first OpenAI HTTP/CPU router slice

- **One process:** optional `zerogravity serve` feature on original Qwen3.5 GGUF CPU only. Each canonical model path loads once under one immutable SourceIdentity and ResourceArbiter lease. Multiple aliases share original bytes/tokenizer; each request owns an independent StateRoot.
- **HTTP/1.1:** loopback-only; remote access must use a separate TLS/auth proxy. Optional `ZEROGRAVITY_API_KEY`. `GET /health` exposes configured CPU admission-budget counters; `GET /v1/models`; `POST /v1/completions` and `POST /v1/chat/completions`. Reject user chat strings containing raw Qwen role delimiters.
- **Supported semantics:** greedy temperature 0, top_p 1, n 1 and zero penalties, bounded max_tokens or max_completion_tokens; plain string chat/content only. JSON/SSE with finish reason and optional usage; strictly reject unsupported sampling/tools/structured output/roles, duplicate JSON keys and HTTP framing ambiguities. **Not full OpenAI API parity.**
- **Concurrency:** finite CPU worker threads, per-worker round-robin quanta of ≤32 prefill tokens or one decode token, bounded active and queued requests. Distinct sessions share immutable weights. One configured host-resource ledger charges weight residency and per-request state/scratch. This is a configured budget, **not measured available physical RAM**.
- **Cancellation:** dropped request receivers stop future worker quanta, including queued tasks. HTTP disconnect and SSE keepalives with deadlines, bounded response backpressure. CPU worker session is released on completion. No GPU work launched.
- **Bounds:** default 2 workers × 2 active requests per worker, 16 queued per worker, 64 connections, 1 MiB JSON body, maximum configurable context 8192/output 512 tokens. Distinct long-context accuracy limits still apply.

## Run

From the repository root, with approved GGUF/config/tokenizer already present:

~~~bash
cargo run --release --locked --offline -p zerogravity-cli --features serve -- \
  serve --config examples/serving/qwen-cpu.yaml
~~~

Or explicit model and resource caps:

~~~bash
cargo run --release --locked --offline -p zerogravity-cli --features serve -- \
  serve --model qwen3.5-0.8b=target/models/qwen35-0.8b/model.gguf \
  --workers 2 --active-per-worker 2 --max-context 512 --max-output 32 \
  --host-mib 4096 --listen 127.0.0.1:8000
~~~

~~~bash
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"Hello!"}],"temperature":0,"max_tokens":8}'
~~~

Set configured host budget responsibly based on physical RAM and other host/cgroup workloads. The server does not automatically measure all OS headroom. With `ZEROGRAVITY_API_KEY` set, every request requires an Authorization Bearer token.

## Not implemented / next

- Paged KV/cache, full-prefix native GPU prefill, cross-machine load balancing and any vLLM goodput parity claim.
- All sampling/grammar/reasoning/tool/audio/image/video/embedding/provider-specific API shapes beyond the above bounded subset; real pipeline/media execution, multi-tenant shared context, hot reload or persistent sessions.
- Gemma4/Qwen3.8 trained execution, SSD model inference, exact trained 27B StateRoot or tokenizer support.

Next: qualify original-HF arithmetic in the same engine and bounded source streaming, then run matched pinned-vLLM workloads. Preserve independent per-request StateRoots, finite occupancy and original BufferLease completion; measure p50/p95 TTFT/true ITL, committed SLO goodput, fairness, memory and actual dispatch counts. Paged hybrid state and physical prefill/decode overlap remain open.

## Resident GPU OpenAI serving — accepted CUDA slice

- One `ResidentModel` per canonical GGUF; aliases share source.
  One independent `ResidentSession`/StateRoot per request and the
  existing physical service/queue/BufferLease readiness.
- Explicit `serve,cuda`, `serve,vulkan` or `serve,hip`. No CPU
  inference fallback. GPU greedy token feedback stays on the
  original device path; the host handles tokenization and text.
- One finite GPU worker, ≤8 active/16 queued. Default `interleaved`
  uses B1 graphs. Opt-in `--batch-mode fused` requires CUDA and
  active>=rows and `--batch-rows 2|4|8` (default2): matching-source
  decode rows use actual native multi-row kernels at independent frontiers.
  Prompt-tail grouping still requires matching frontiers.
  Other prefill uses ≤32-token B1 quanta;
  unmatched rows use B1. No padding stopped or blocked requests.
- Shared CPU/GPU output owner handles EOS, UTF-8, SSE, usage,
  backpressure/cancel. Credit-blocked GPU rows park before feedback
  with one retained packet; a 30s deadline bounds stalls. Ready-cohort
  cancellation discards every candidate before retrying survivors.
  Health distinguishes scheduler quanta, completed model-token work,
  actual kernel dispatches, native cohorts, committed roots and
  queued/active/canceled/error work; accepted output is not goodput.
- CUDA fail-fast prerequisite: `CUDA_PATH` contains runtime
  headers and CCCL device headers. Split installed Nix packages
  can be staged using `scripts/cuda-nix-include-view.sh`:
  link existing header paths; no SDK download/driver modification.

### Run qualified NVIDIA CUDA HTTP source

```bash
# Supply already-installed CUDA runtime and CCCL include directories
# as CUDART_INCLUDE and CCCL_INCLUDE; no packages are downloaded.
export CUDA_PATH="$(scripts/cuda-nix-include-view.sh \
  "$CUDART_INCLUDE" "$CCCL_INCLUDE" \
  "$HOME/.cache/abird/zerogravity/cuda-toolkit-view")"

# Put the locally installed NVIDIA driver and NVRTC library directories
# on LD_LIBRARY_PATH, then launch:
cargo run --release --locked --offline \
  -p zerogravity-cli --features serve,cuda -- \
  serve --config examples/serving/qwen-cuda.yaml
```

The GPU YAML profile specifies backend=cuda, workers=1,
active=2, context=512 and bounded provider/device memory.
Use your XDG cache root instead of HOME/.cache if configured.
See the CPU YAML example for unchanged CPU serving.
Local Renoir/gfx90c AMD launch hold remains intact.

### Initial interleaved evidence: RTX 3060 Laptop / CUDA, 19 Oct 2026

- Qwen3.5 0.8B: resident device model, HTTP 200, GPU-generated
  greedy token. Initial debug process readiness ~7.65s.
- Two requests, two aliases, one immutable source: both produced
  identical deterministic 2-token completions. 6 completed
  scheduling quanta, 3 source-compatible scheduling groups,
  4 successful model-forward API calls (NOT single kernel launches),
  0 failed, 0 fused, 0 active
  sessions after return. ~16.56s debug wall in first test.
- Disconnect while active: in-flight GPU step finished, session
  canceled and retired, no further launches; fresh session worked.
- `tools/serve-bench.py` tested 2 offered/2 complete SSE requests:
  4 usage-reported tokens, 20.02s debug wall, ~0.1998
  tokens/offered-second, p95 first streamed text ~19.78s.
  No pinned vLLM baseline or trained accuracy comparison;
  SSE chunk intervals are NOT verified per-token ITL.
- CUDA 12.9 split packages were already in Nix store. Vulkan
  builds/tests pass, but CubeCL exposed 0 usable local discrete
  adapters despite `vulkaninfo` seeing NVIDIA. No Vulkan physical
  acceptance, AMD launch or GPU reset performed.

Detailed implementation/acceptance contract: [FUSED_GPU_BATCHING.md](FUSED_GPU_BATCHING.md).

## Native CUDA B2 HTTP acceptance — P12-B5

Implementation: `2677d8b`.

Release tests over BOTH approved original models prove native
source-alias JSON/SSE, exact independent CPU text/usage, real trained
EOS stop with EOS excluded from usage, one terminal/usage/Done,
active disconnect, survivor and fresh replay. Internal real GPU
worker tests additionally compare compact token ID/max bits and all42
state domains against CPU and B1, B1↔native packet handoffs,
unequal output limits, cancellation before feedback/after Ready,
actual bounded-channel parking and final reservation0.

| Original source | Completed native cohorts | Native projections | Native sampler stages | Discarded Ready cohorts | Final active/queued/failed |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5 0.8B | 9 | 1683 | 36 | 1 | 0/0/0 |
| Qwen3.5 2B | 8 | 1496 | 32 | 1 | 0/0/0 |

Cohort counts include actual completed work discarded before root
publication. They do not measure delivered tokens or throughput.
HTTP wire checks do not expose private StateRoot/lease contents;
those exact checks belong to the internal worker qualification.

For the optional native mode, use the same qualified CUDA environment
and resource caps, for example:

~~~bash
cargo run --release --locked --offline -p zerogravity-cli \
  --features serve,cuda -- serve --backend cuda --batch-mode fused \
  --model qwen3.5-0.8b=target/models/qwen35-0.8b/model.gguf \
  --workers 1 --active-per-worker 4 --max-context 512 --max-output 8 \
  --provider-mib 4096 --device-mib 5120 --listen 127.0.0.1:8000
~~~

## Pending: P12/P16/P22

1. Unequal frontiers, dynamic compatible refill
   and full-prefix GPU prefill; qualify additional trained profiles
   and backends before advertising them.
2. Device-resident paged hybrid KV + GDN/COW state, source-safe
   prefix reuse, physical admission and cancellation/quarantine.
3. Fair prefill/decode/media occupancy and safe overlap across
   GPU/host/SSD; no phantom memory credit.
4. Actual p50/p95 TTFT/token ITL, committed SLO goodput, fairness,
   memory and dispatch measurements against a genuine pinned,
   source/quant/tokenizer/hardware-matched vLLM V1 environment.
   Pinned vLLM0.31.0-cu129 V1 CUDA startup/compile/graphs and original
   HF0.8B warm smoke pass in an isolated immutable derived runtime.
   These runs do not establish a matched comparison or numerical parity.

Earlier B0/B1/A1/B2 operator and serialized-cohort evidence is retained
in [FUSED_GPU_BATCHING.md](FUSED_GPU_BATCHING.md); its historical
limitations are superseded only by the later named acceptance scope.

## Native CUDA B4/B8 — P12-B6

Both original0.8B/2B qualify actual CUDA HTTP rows2/4/8, unpadded
survivor regrouping, trained EOS and disconnect/replay. Internal worker
and real lease-pressure tests preserve exact CPU/B1 packets/logits/all42
and final reservation0. Only checked prelaunch capacity refusal retries
smaller bands; other errors preserve the original fence/quarantine.
Cold-load cleanup physically releases retired allocator pools.

Use `--batch-rows 8 --active-per-worker 8` with the qualified CUDA command.
Default rows2/interleaved mode remain unchanged. Exact acceptance and
remaining scope: [FUSED_GPU_BATCHING.md](FUSED_GPU_BATCHING.md).

## Unequal decode frontiers and token observations — P12-B7

Both approved original0.8B/2B sources pass full248320-logit/all42-state
CPU/B1/native comparisons at B2/4/8 and mixed frontiers through511.
Eight fresh release worker and six actual HTTP tests cover both models,
including a later-arriving short request joining native decode at a
separate frontier. Pressure and actual unknown-completion quarantine
probes also pass. Prefill tails still require matching frontiers.

`--enable-per-request-metrics` plus `return_token_ids:true` opts into
bounded token IDs, producer frontier/native publication membership and
separate commit/queue/socket-write observations. This does not establish
standard scheduler-origin timing, true ITL for grouped client arrivals,
source-correct SLO goodput or matched vLLM performance. Exact scope and
remaining work: [FUSED_GPU_BATCHING.md](FUSED_GPU_BATCHING.md).


## Serialized benchmark provider ownership

`tools/serve-compare.py --provider-lifecycle SPEC.json` optionally starts and
stops one fresh owned provider per ABBA/BAAB sample. The strict spec in
[serve-lifecycle.py](../../../tools/serve-lifecycle.py) binds direct argv,
source/runtime/tokenizer/profile/policy/cap declarations and finite deadlines.
Use `--lifecycle-output-dir` to choose private logs/receipts. An existing
listener or incomplete cleanup refuses the next launch.

`warmed_process` performs one unmeasured request per fresh process;
`cold_process` omits it. Neither policy evicts or attests OS/storage/device
cache coldness. Real ZG and pinned-vLLM GPU startup/warmup/teardown passed
serially and returned telemetry to baseline. Their source profiles remain
unmatched; performance and independent long-context numerical acceptance
remain open. See [numerics-and-context](../numerics-and-context/TODO.md).

````

## Source: .agents/plans/serving-and-speculation/TODO.md

Original UTF-8 bytes: 38407; SHA256 `eec2f868421e5fedff223da60e43e8ade2b5b63e46a90fbd83ed3b43cd7b9fa3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Serving and speculation

Provenance commit: `6d74f76`.
Scope: P09/P12/P15/P16/P20/P22.
Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md),
[36 serving gates](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md),
[Qwen parity and decoding](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md).

## Uncommitted

- [ ] **P12 ordered SiLU integration:** private CUDA B1/B2/B4/B8 operator
  passes204 physical old-GPU/CPU differential cases,171562710 exact bit
  comparisons and five fresh failure/abandonment processes. All17 intermediate
  checkpoints, status, canaries and original lease/quarantine ownership pass.
  Independent source review and candidate CUDA/Vulkan/default strict checks
  plus two row-partition tests pass. Candidate library, qualifier, server,
  worker and HTTP release artifacts compile; the exact-source diagnostic
  file-transport unit passes. Paused at the owner's request before starting
  the prepared60-case physical matrix. Full-model integration remains unaccepted:
  complete PTX/loaded-code witnesses, model/StateRoot/serving regressions, both
  original release parity routes, isolated full gate and fresh timing/count
  comparisons remain pending. Existing truncated PTX dump filenames overwrite
  specializations; retain that failed evidence and qualify a collision-safe
  diagnostic before claiming complete instruction coverage.

- [ ] **Stock2B loader and matched benchmark:** genuine pinned stock2B
  V4 plain/observed loader and independent comparison now pass: all335
  stored tensors inventoried, exact15 MTP exclusions and320 backbone roles;
  both complete vocabularies,496640 native BF16 logit words, remain exact.
  Source/runtime and owned process/group/container/EOF cleanup pass; retain
  the original V3 observer refusal. Both model loaders are prerequisites,
  not cross-engine/full42/default-graph or throughput qualification.
  Actual0.8B eager HTTP policy passes for both providers: identical47 prompt
  IDs, original Rust tokenizer, effective greedy/EOS settings, observed
  cgroup12GiB/swap0/CPU4 caps and authenticated GPU process membership.
  Original owned cleanup closes. Podman unresolved HPIDs establish no
  ownership; every sampled GPU PID must match the numeric owned subset.
  Generated IDs differ under the declared activation/cast profiles. Actual
  0.8B full-output quality covers seven prompts; quality equivalence remains
  unqualified. First unprofiled eager/cold-process/solo comparison completes
  all eight ABBA/BAAB samples: four requests/32 raw tokens per provider.
  Median TTFT13.400s native versus1.561s stock; stock is faster here.
  Cold-process means first request after readiness, not storage-cache cold
  or complete startup latency. Both providers are internally stable, but
  cross-engine IDs/text/usage differ. One stock sample coalesces delivery;
  individual-token ITL and source-correct SLO-goodput remain unqualified.
  The2B eager/cold-process/solo run also completes all eight samples:
  four requests/32 tokens per provider, identical raw IDs/output hashes/
  usage/finish on that one prompt. Median TTFT14.540s native versus1.562s
  stock; all samples have individual token delivery, native ITL p95
  0.186–0.199s versus stock0.026–0.030s. Seven-prompt2B quality remains
  unqualified despite five matching full texts; this prompt establishes no
  numerical equivalence. Finish default-route, warm/concurrent workloads
  and physical counts;
  this bounded observation does not establish numerical or serving parity.

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
  Separate native0.8B B2 CUPTI diagnostic records64328 actual kernels and
  64328 successful direct launch calls, zero dropped records and exact source
  counter delta; plain/observed complete outputs and owned retirement match.
  A fresh accepted-baseline0.8B canonical C1 diagnostic also validates all
  four native/stock plain/observed processes:236270 actual native kernels
  and direct launches equal the source delta, zero dropped records/errors;
  exact provider-internal outputs and original owned retirement pass.
  Stock public profiler records9318 kernel activities/8810 host launches,
  but its losslessness/all-process coverage remains unverified. Independent
  validators and ROOT/peer metadata reviews pass. These observations qualify
  the native windows only; complete matched hardware counts, profiler device
  allocation admission, source-correct goodput and profiled timing remain
  unqualified. Equivalent default-route/warm/concurrent vLLM timing stays open.

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

### fb68d8f — exact state transport and certification

- [x] **Row-native state transport and scatter certification:** exact U32
  vector/KV transport and persistent finite/status certification preserve the
  original all42 preflight, leases, flight, Ready and atomic publication.
  All46 physical model/owner/private-graph/failure cases,10 operator/failure
  cases, complete isolated1251-test gate, strict checks and both approved
  original release parity routes pass. Fresh14 worker/12 HTTP regressions
  pass with exact frozen source/build and original owner/EOF retirement.
  Native-only unprofiled before/after runs complete32 fresh ABBA/BAAB
  processes,144/144 measured offers,1152 output tokens and32 ordinary B1
  warmups. Full IDs/text/usage/finish match exactly; all144 requests have
  individual-ID delivery/1008 actual intervals. C8 uses54 actual B8 cohorts
  per sample,432 independent steps/roots. Median raw token rates improve
  2.2506→2.28945/s(0.8B,+1.73%) and1.6268→1.6543/s(2B,+1.69%);
  C1 is nearly unchanged. Four samples/label, finite bursts, ordinary B1
  warmup and point memory samples establish no confidence-certified win,
  fully warm B8 scope, source-correct goodput or vLLM parity. Source dispatch
  counts292116→239952/284340→232176 are not hardware enumeration. All owned
  process/group/container/EOF retirement and independent ROOT/peer reviews pass.

### fff1842 — stock baseline prerequisites

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

````

## Source: .agents/research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md

Original UTF-8 bytes: 20113; SHA256 `30a30d0dc41a4aa37829ac92687c0cb4feaa0c2c3cc368767b18ddfa17b7bb02`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — Extreme Model Specialization Integration Roadmap and Executable Acceptance Design

**15 October 2026 — EXTREME MODEL SSD STORAGE FIRST-CLASS ADDENDUM:** Derived-quantization/QTIP derived quantized routed experts and MiMo/Qwen/DeepSeek sparse n-gram/Engram tables can remain SSD-backed through **the same** verified ModelAssembly/SourceTensorId and ready-leased source-compiled GPU kernels. [Normative virtual residency contract](../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) and [SSD-01..SSD-32 detailed plan](SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) extend original P03/P05/P07/P10/P11/P15/P16/P20/P22 with async dynamic expert materialization, size-class caches, source-verified physical slot generations, prefill/decode reuse, per-session target/draft ring and actual SSD latency/OS headroom. Do not conflate disk KV snapshot with expert weights, source lower-bitrate quant with memory tier, or SSD capacity with interactive latency. Borrowed GPU kernel code remains identical source math; stream policy is physical placement with real event lifetime.


**2026-10-15 | DESIGN PLAN, not runtime implementation, performance proof or new user authorization to download weights.**
**References:** new derived-quantization/QTIP quant, hybrid/DFlash and gfx1151/UMA contracts under .agents/docs; detailed specialist source atlas. Existing P00–P23 phase order and user-selected Qwen3.5/Gemma4/Qwen3.8 CPU-first model order remain unchanged.

Current work is tracked in the [CPU/GPU optimization TODO](../plans/cpu-gpu-optimization/TODO.md), [model execution TODO](../plans/model-execution/TODO.md) and [serving/speculation TODO](../plans/serving-and-speculation/TODO.md). This reference preserves proposed acceptance, not a support or performance claim. The [source atlas](SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md) supplies the dated research context; recheck exact source revisions and licensing before implementation.

## 1. Success definition

Any future extreme sparse/hybrid architecture (Qwen3.8 Flash-Next, GLM5.3 Flash, MiMo2.6 Flash, new MOPD/BitNet/ternary variants) must be added through **one VerifiedModelAssembly + exact model semantic adapter + optional DerivedQuantArtifact + TaskDriver/DraftSource + StateDomains + guarded KernelCandidates**, not a parallel runtime.

Pass criteria:
- original checkpoint source math and independent state oracle;
- quant pack source codec and exact tensor mapping; optional lossless layout repack vs lossy derived quality clearly distinct;
- target CPU reference and GPU backend numerical/source correctness on original trained tensors;
- all-state speculative target verification including zero accepted tokens, RNG/grammar/tool stop, auxiliary sidecars and parent root;
- peak physical weight/state/OS page cache and driver pool admission on unified memory;
- real per-model, per-task, per-ISA performance and cold/warm/long-context checks;
- no global router, TaskCoordinator, public HTTP or resource arbiter new code just because this model is registered.

No model is "supported" until appropriate source, physical, serving and quality evidence tier is achieved.

## 2. Specification owners and minimal architecture

| Stage | Existing authority | Extension allowed | Must not add |
|---|---|---|---|
| Source artifact/derived pack | Verified ModelStore/ArtifactKey | codec/trellis/codebook calibration/semantic-derived hash | arbitrary source Python execution |
| Model math | ModelRecipe/SemanticProgram | new source attention/recurrent/mHC/MiMo and transforms | new model-specific inference engine |
| Draft | DraftSource + target verifier TaskDriver | trained MTP, auxiliary DFlash, lookup; source state | sampler/Root writer in drafter |
| Device math | RegionExecutionProvider/KernelCandidate | gfx1151 WMMA/LDS/DOT4, quant fused GEMV, source-aware fallback | independent AMD allocator |
| Batch/cost | Scheduler + physical ResourceArbiter | measured row/effective expert/QoS cost candidate | hardcoded batch16 in semantic math |
| Model and OS residency | ModelCatalog + arbiter and local leased pools | UMA headroom/page cache/resident rings | artificial virtual GTT=free physical RAM |
| Serving | existing router and canonical task event | truthful derivation, reasoning, metrics/API | Python side server |
| Optional steering | versioned ModelAssembly/profile transform | explicit user-selected behavior effect | hidden "uncensor" mode labeled source-exact |

## 3. Phase dependency map — additive to original project

### ZG-P03 (quant formats and artifacts)
Plan:
1. Read exact derived-quantization/QTIP source code under pinned commit and derive block/codebook/trellis bitstream layout. Separate K/codebook/scale/basis logical tensor identity from layout swizzle.
2. Add source CPU decode oracle for K2..K8 where available and guard mixed per-tensor precision, FP16/BF16 heads, router/indexer and sidecars. Reconcile source HF/GGUF+derived-quant packs; derived and original hashes distinct.
3. Record a separate derived quantization artifact recipe, calibration trace provenance and held-out quality matrix, not silent lower-bit mode.
4. Cache file-backed table source and auxiliary model artifacts by model/quant/source hash and checked offset/size, without full duplication.
5. Report missing derived-quantization or MiMo E8M0 as UnsupportedCodec/UnsupportedModel and permit original source CPU fallback only when a correct reference exists.
Gate: QTIP-01..05, 08..12 and HYB-04; required original CPU reference for exact first model.

### ZG-P05/P07 (compiler and physical GPU)
Plan:
1. Shape-guard Wave32/WMMA fragment and register/LDS layout; source reference must test m=1,2,3,4,8,16+ and aligned/ragged codebook blocks.
2. Source-trust candidate fused derived-quantization packed decoder GEMV versus reconstruct+matmul, with cost and exactness receipts.
3. ROCm gfx1151 specialized v_dot4 and SAD trellis and skinny GEMM candidates; optional CUDA equivalents generated independently.
4. Tune LDS physical stride and scratch warp coverage without altering semantic logical row dimensions. Validate all callers and both partial/full trained model logits.
5. Qualified fallback when native codegen unavailable or slower; no compiler explorer loop on every request.
Gate: GFX-01..06, GFX-14,17 and QTIP-06/07.

### ZG-P10 (global physical admission, OS and state)
Plan:
1. Distinguish physical GPU residency from OS available headroom, file-backed clean page cache, pinned staging, driver pool and UMA GTT.
2. Add an OS-safety admission policy for huge models+draft+sparsely fetched lookup: host total/MemAvailable/cgroup/PSI when supported; platform-specific conservative fallback.
3. Correct multiple model and per-session state: 39 SWA/9 global + DFlash ring example is a model-source configuration, not a hard-coded memory formula.
4. Record optional safe clean-shard page-cache hints and bounded file reads; refuse memory-threatening requests **before** GPU dispatch, never kill unrelated processes.
5. ModelStore singleflight and immutable model weight reuse, per-session root/draft isolation; process-wide unknown GPU event quarantine remains.
Gate: GFX-09..11,15,18, HYB-11/12 and QTIP-13.

### ZG-P11–P14 (specialized model math and MoE)
Plan:
1. Qwen3.8 QSA+GDN+PLE and gate/residual full CPU semantic oracle, then GPU.
2. GLM5.3 KDA/MLA/DSA/mHC source reference, sparse indexer and prompt/vision input variants, then GPU.
3. **MiMo2.6** exact source config, SWA/global KV, E8M0 scales, different key/value widths, MoE/routing and paired overflow correction.
4. Closed region candidates for norm→activation pack→projection, GDN/KDA recurrence+short conv, MLA sparse indexer/tile, residual mixer, routed expert compact grouped quant GEMV/GEMM and safe residual reductions.
5. Token routing/expert destination remains model semantic; aggregation packing and DMA prefetch are physical scheduling only. Independently original model state source oracle.
Gate: HYB-01..05,16, GFX-05/06, and full trained model edge cases (long context, incomplete cache, invalid mask/scale).

### ZG-P15 (target-authoritative speculation)
Plan:
1. Distinct TrainedMTP, DFlash, independent small model and prompt-lookup proposal sources; validated compatibility with target exact checkpoint/derived pack/tokenizer/tap layers.
2. Verify MiMo DFlash partial RoPE, learned mask, bidirectional draft block, sink/value scale and tap_shift; target-only exact acceptance and draft intermediate goldens.
3. Bind optional GLM MTP BF16 eh_proj and Qwen MTP/n-gram sidecars; independent artifact hashes and resource/draft state.
4. Accept only verified target prefix; all provisional GDN/KDA/MLA/indexer, RNG/grammar, output and model state changed atomically or rolled back.
5. Dynamic depth based on net target-only vs draft benefit; no positive benefit -> no draft. Correct stochastic target distribution and tools/stop.
Gate: HYB-06..10,14..15,17, plus exact target sampling distribution/adversarial StateRoot tests.

### ZG-P16 (batching and multi-model QoS)
Plan:
1. Plan rows R=sum(1+d_i), coupled with per-expert count, padded groups, source quant/ISA and state/draft bandwidth.
2. B1/2/4/5/6/8/10/16, depth0..8 and route entropy sweeps; explicit shape "cliffs" and physical queue throughput/fairness.
3. Select among finite prevalidated plans under TTFT/p95/energy/latency/throughput; maintain simple predictable fallback; no model router/catalog scanning per-token.
4. Preserve per-session root, source-specific cache buffers and correct sampled-target distribution.
Gate: HYB-13/14/18, GFX-07/08/16/18 and P16 existing fairness matrix.

### ZG-P20 (bounded autotune / offline quant calibration)
Plan:
1. Compile/reference known correct generic kernels, choose finite shape candidates for new image sizes without unbounded solver search.
2. Put shape/driver/quant/ISA/numeric/graph cache keys and negative candidate evidence into a small persistent candidate database.
3. Optional, source-licensed offline optimizer for calibration trace, per-tensor LDLQ-shaped noise, sensitivity and bitrate allocation, held-out KLD/quality; sparse-model claims remain unverified until tested.
4. Benchmark hardware codebook decode, LDS bank collisions, GDN/MLA and MoE route bands, graph capture, CPU offload and NPU cooperative work as mutually independent experiments.
Gate: QTIP-08/09, GFX-12..14, full A/B negative cases and no protected realtime hot-path compile.

### ZG-P22 (release quality, API and performance)
Plan:
1. Strict evidence tier: recognized-format vs source-exact CPU vs selected physical GPU region vs whole model GPU vs streaming vs measured soft realtime, per model/artifact/device/numeric/shape.
2. Compare derived pack vs original source and speculative-on vs derived target **separately**, plus cross-batch numerical drift.
3. Report cold/warm first response, prefill, physical valid committed output, p50/p95/99 decode, aggregate throughput, OS MemAvailable floor, power/thermal/energy, actual buffer completions and quality.
4. Mixed text/chat/prose/code/tool/multimodal prompts and precise source tokenizer reasoning/stop parser tests; no format-level claim of complete GPT-4 style multimodality.
Gate: completed all relevant QTIP/HYB/GFX suites, existing Qwen CPU8K/extended gate truth and no claimed quality unsupported by real input/source.

## 4. Acceptance register (50 planned tests, NOT implemented)

- QTIP-01..QTIP-14: defined concretely in DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md.
- HYB-01..HYB-18: defined in EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md.
- GFX-01..GFX-18: defined in GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md.
- Source/physical reproductions refer to exact licensed Qwen3.8 Flash-Next, GLM5.3 Flash, MiMo2.6 checkpoint and corresponding specialist pack only if user authorizes downloads/evaluation.
- These 50 tests are **future design acceptance**; they do not add to currently passing Qwen3.5/G3 proof or six MM0 multimodal mock tests.
- Negative tests must fail closed for malformed quant, wrong transform/drafter/source, invalid state, physical unknown completion, memory pressure or unsupported device. No fake Ready/event or conditional lowering to another model.
- Adjacent v5.1 media MM-01..24 remain governed by prior plans; no audio/image/video/decision TaskDriver semantics are changed by specialist-specific optimization.

## 5. Experiment methodology and promotion policy

Each promotion candidate has one exact source digest, original/derived pack, device and OS, driver/kernel compiler/ISA, numeric mode, task/shape/context, source latency/throughput/quality oracle and memory figures. A measured gain on one prompt does not become the default.

| Result | Action |
|---|---|
| Source numerical failure, StateRoot mismatch or unsafe GPU resource | hard FAIL, candidate disabled, error cause recorded |
| Derived artifact quality loss beyond approved policy | reject derived pack as target, regardless of kernel speed |
| Hardware capability missing | source-correct fallback or Unsupported with exact reason |
| Whole task speed neutral/noisy | keep simplest known-correct baseline, record negative experiment |
| One prompt faster but representative P95 worse | do not auto-promote; separate explicit mode if justified and approved |
| Memory pressure close to OS reserve | refuse admission before destructive load, no silent SIGKILL |
| Repeated ABBA shows source-correct whole task win across representative prompts | register guarded KernelCandidate with evidence; no universal model-name hardcode |
| Change in firmware, driver/compiler, quant pack or shape family | invalidate only affected compiled/evidence cache cells |

**Benchmark fairness:** match actual derived packs and original reference precision, tokenizer, source prompt templates, context, sampling, MTP depth, active sessions and device power/fabric clocks. Report per-session TTFT/tokens per sec AND aggregate committed output. The specialist prior-art external results are targets for experimental comparison, not facts about Zerogravity.

### Research cautions retained from the retired handoff

- A gfx1151 Wave32 WMMA/LDS layout is a capability- and caller-specific candidate. A useful staging-stride change in one kernel can invalidate another caller's wider scratch requirement. Check every caller, complete trained rows and final logits; aggregate perplexity alone can hide a defect.
- Smaller quantized storage can decode more slowly. Likewise, one source's grouped-MoE row boundary made a larger batch slower, and single-stream versus aggregate rates measured different objectives. Reproduce these mechanisms on the selected source/device; do not copy a batch constant, row-cost formula or advertised token rate into model semantics.
- A correct target can reject a broken draft and conceal its semantic error. Independently test draft masks, taps, partial RoPE, sink/value scaling and acceptance, including workloads where drafting slows execution.
- A source pack's FP16 overflow workaround paired conversion-time scaling with runtime scaling. Treat that as a separately identified derived artifact with quality tests; do not generalize it into clamping or a source-exact transform.
- Clean source-shard page cache can compete with resident GPU weights in one coherent memory pool. Driver free memory, GTT addressability and OS MemAvailable are different observations. Admission must include both uses and OS headroom, without automatic process killing or global page-cache discard.
- CPU expert offload, some INT8/prefetch/graph variants, lower bitrates and deeper draft trees produced neutral or negative results in the source study. Keep a correct disabled baseline and require complete-request gains. First-image shape search also needs a bounded policy; any convolution-to-GEMM alternative still needs source equivalence.
- Activation steering changes model behavior. Keep it off by default and separately versioned, authorized and quality-assessed rather than presenting it as a kernel optimization.

## 6. Small design decisions and open questions

| ID | Disposition | Decision |
|---|---|---|
| SYK-001 | Existing user constraint reaffirmed | Rust-owned one engine and one TaskCoordinator, no hidden external/Python process |
| SYK-002 | Proposed precision contract | New source quant codec and derived artifact are orthogonal to model mathematical identity and packed hardware kernel |
| SYK-003 | Proposed implementation gate | MiMo-V2.6 a distinct source-qualified ModelRecipe and DFlash semantics; no Qwen alias |
| SYK-004 | Proposed specialization | Kernel physical codebook/tile/ISA/LDS, guarded by exact source numerical and shape evidence |
| SYK-005 | Proposed QoS | MoE batch/verification depth uses measured expert work, row packing and accepted output, not max B/draft |
| SYK-006 | Proposed source policy | Trained MTP, DFlash, lookup and small drafter are versioned DraftSources with target-only authoritative commit |
| SYK-007 | Proposed OS safety | Coherent-memory OS+GPU physical headroom, bounded source read/page-cache pressure, no invisible unsafe process killer |
| SYK-008 | Proposed offline research | Optional representative trace, shaped tensor sensitivity and held-out quant quality before derived artifact approval |
| SYK-009 | Proposed behavioral gate | Activation steering is OFF by default, explicit behavior-changing model version, separately assessed for safety/quality |
| SYK-010 | Proposed cold shape policy | No indefinite first-image request tuner; source-equivalent generic fallback and finite offline autotune |
| SYK-011 | Open | exact derived-quantization/QTIP source codec schema, licensed original checkpoints and conversion toolchain choice |
| SYK-012 | Open | acceptable approximate quant/perceptual loss vs original; user has not chosen universal threshold |
| SYK-013 | Open | native CUDA/ROCm LLVM23 source build availability and operator permission |
| SYK-014 | Open | target startup latency, per-task soft SLO and high-availability/process recovery policy |
| SYK-015 | Open | authorized model downloads/large 100GiB physical pressure experiments and page-cache hints |
| SYK-016 | Open | whether any steering/activation-transform feature should be exposed at all; default remain disabled |

Do not treat a proposal as a user-approved implementation requirement or an open question as resolved. The actual new user request is to **strengthen plans and specifications** cleanly, not to launch experiments.

SYK IDs retain their local proposal/open-question meaning; they do not renumber the separate U-040/U-041 histories. SYK-013 is the original question about an LLVM code-generation path and permissions. The subsequently accepted native C++ NVRTC/HIPRTC profiles do not qualify that LLVM path, and this historical question is not a claim that current native execution requires LLVM23. Current backend and permission limits remain in the linked TODOs.

## 7. References and implementation boundary

The detailed requirements are in the [derived quantization contract](../docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md), [hybrid-model and drafter semantics](../docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md), [gfx1151 cost and unified-memory contract](../docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) and [source atlas](SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md). Use the [plan index](../plans/TODO.md) for current work and the [repository workflow](../../AGENTS.md) for coordination.

Implement only the assigned bounded slice, beginning with source-free acceptance and an independent original-source CPU reference before actual trained GPU and performance gates. Report the achieved model/codec/device scope explicitly. Never substitute a new local engine, hidden quant format, behavior steering or fake GPU event. This reference supplies proposed design and acceptance; it does not establish implementation or qualify a new source model.

````

## Source: .agents/research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md

Original UTF-8 bytes: 4536; SHA256 `4a7a0da4210cf82c6c69521875e0a6271199f9899127bbb593b0ecc77ba994f5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Gemma 4 family source contracts — 2026-10-16

Technical reference: configuration, tensor-shape and KV ownership contracts for E2B and26B-A4B. It does not establish trained inference, artifact loading or tokenization.

The CPU-free `zerogravity-model-gemma4` family library exposes these metadata
contracts through the existing shared source/state/resource boundaries.

## Authoritative sources

Official metadata/source revisions below define the current technical profile.
Completion/results and pending execution work live in [model-execution/TODO.md](../plans/model-execution/TODO.md).

- [Official E2B-it configuration](https://huggingface.co/google/gemma-4-E2B-it/blob/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json), revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- [Official 26B-A4B-it configuration](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/config.json), revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`; base configuration at `24548b62aa021d562695c04aaf7758a1ea47990b` parses to the same configuration value.
- [Gemma 4 reference implementation](https://github.com/google-deepmind/gemma/tree/9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3/gemma/gm/nn/gemma4), revision `9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3`.
- Framework Gemma 4 implementation, revision `683b721971417b3b2c64fba9001f525607547e5a`.
- [llama.cpp Gemma 4 metadata mapping](https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/models/gemma4.cpp), revision `b6334e37e6ef286fc5091a41b6561b2113d915fb`; consulted as an additional layout reference, without adopting its runtime.

The optional E2B weight index returned404; it is a single-file observation,
not independent qualification of headers or the complete learned-tensor closure.

## Implemented API and invariants

`Gemma4Config::from_json` accepts bounded official outer/text configurations and stores checked geometry privately. Read-only accessors expose variant, per-layer attention/head/rotary geometry, vocabulary, scalar policies, per-layer input dimensions, EOS and MoE geometry. Unsupported activation, schedule, sharing, rotary and scalar overrides return typed errors before deriving storage contracts.

`tensor_specs()` describes original HF conditional-model tensor names and row-major shapes. `validate_tensor_shapes()` checks required declared metadata and rejects missing/transposed shapes. It does not authenticate bytes, codecs, checkpoint identity or decoded values. Optional tied heads and source-omitted shared/K=V projections remain explicit.

`kv_state_owners()` returns existing core KV-domain descriptions. E2B has 15 owning layers and 20 consumers reusing the last compatible local/global producers, layers 13/14; local/global head widths differ. E2B also retains its packed per-layer embeddings and doubled later FFN width. The 26B variant has distinct full-attention KV widths, learned routing/expert axes, K=V projection semantics and extra MoE branch normalization weights.

`MoeGeometry` and `variant_26b::validate_config` provide checked metadata only. The contract library has no production routing arithmetic or CPU executor.
Five test-local analytic routing cases do not establish framework source parity.

## Remaining work and qualification limits

The shared plan/backend needs source-defined GELU, elementwise multiply/scale, final tanh soft-capping, sliding attention and shared KV, Gemma proportional RoPE and value RMS normalization. Implement these as shared operators with immutable source binding and existing session/physical ownership. Do not express Gemma as Qwen, add a duplicate numerical runtime or substitute CPU output for an unsupported GPU region.

The current source-storage API still has its existing 4 GiB owned-byte cap. The 26B artifact and E2B's large per-layer embedding storage require a separately designed cold-storage/admission seam; this unit does not raise the cap. Metadata-only tensor sizes do not allocate model payloads.

Model acquisition, exact tokenizer/template behavior, checkpoint-codec mapping, source-bit operator qualification, CPU/GPU forward plans, real trained multi-token state, cancellation, rollback, full-model logits and context gates remain pending. Framework reduction and per-expert scaling schedules differ, so a later routing implementation must choose and independently qualify its original-source arithmetic. Neither fixture tests nor this contract library establishes runnable Gemma support or performance.

````

## Source: .agents/research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md

Original UTF-8 bytes: 15281; SHA256 `fd3dfc14e583780556b32eaa5ad65e0003f39a7ff9152d1f2e6429382e837661`.
Resolve embedded relative paths against the original path, not this volume.

````text
# GPU tokenization, execution control and text decoding roadmap

16 October 2026. User-requested roadmap addition following the discussion of
GPU tokenization, device execution scheduling and detokenization.

**Status: planned; none of the capabilities below is qualified by this document.**
This addendum extends the existing phase definitions without renumbering them.
Current implementation ordering and completion live in the
[CPU/GPU optimization TODO](../plans/cpu-gpu-optimization/TODO.md); this design
does not reopen completed backend work or qualify the capabilities below.

## Existing architecture and phase ownership

Use the [implementation roadmap](../plans/IMPLEMENTATION_ROADMAP.md),
[backend execution contract](../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md)
and [single-owner execution spine](../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md).
This makes existing device-native StepProgram and graph-execution ideas concrete;
it does not introduce another model runtime, scheduler authority or public DSL.

| Existing phase | Added deliverables |
| --- | --- |
| ZG-P03: artifacts, codecs and rendering | Exact tokenizer/decoder contracts, asset identity, CPU oracle and GPU-compatible vocabulary/merge representations. |
| ZG-P05: legal regions and async commands | Reusable command graphs and persistent execution candidates with explicit effects, resources, guards and rounding boundaries. |
| ZG-P07: accelerator providers | Capability-qualified CUDA/HIP graph replay, equivalent Vulkan command reuse where supported, device queues and persistent execution. |
| ZG-P08: complete task slice | End-to-end prompt bytes to committed output bytes through the same model engine, with declared CPU/GPU text placement. |
| ZG-P09: coordinator and workers | Bounded device-driven generation, asynchronous output delivery, CPU/GPU overlap and authoritative root/output publication. |
| ZG-P10: physical memory | Admission and original leases for tokenizer tables, text buffers, graph captures, device queues, decoder state and output rings. |
| ZG-P15: speculation | GPU stop/grammar/verification work where qualified, preserving target authority, RNG semantics and accepted-prefix rollback. |
| ZG-P16: batching and placement | Batch tokenization/detokenization and measured CPU/GPU placement across prompt sizes, concurrency and live model load. |
| ZG-P20/P22: tuning and qualification | Bounded profiling, negative results and complete latency/throughput/quality/resource comparisons. |

## Shared execution contract

Keep one Rust mathematical engine, immutable source/tokenizer identity,
independent per-session StateRoots and the sole SessionAuthority publisher.
Workers may compute private state/output proposals and execute prepared work;
GPU execution does not independently publish a root or a client response.

Reuse the existing physical ResourceArbiter, fair workers, original buffer
leases, actual completion and unknown-completion quarantine. Budget all added
device tables, graph storage, input/output staging and queue capacity. Report
host metadata and driver allocations separately where measurement is available.

Keep exact token IDs, output bytes and model numeric contracts. Graph replay or
fusion must preserve required FP32 rounding, physical stores and state ordering.
Never widen numerical tolerances to qualify a placement or execution strategy.

CPU and GPU text processing are explicit, separately qualified placements of
the same semantics. Choose placement before dispatch and report the choice.
An explicitly selected GPU path propagates failures; it does not silently
substitute CPU work. Qualified CPU paths remain available for short inputs,
unsupported tokenizer configurations and other declared placement decisions.

## 1. Profile the complete request before selecting optimizations

Measure tokenization, normalization/template work, admission/queue delay,
host submission, GPU execution, synchronization, sampling/stop processing,
detokenization and response delivery separately and end to end.

Cover short interactive prompts, long fresh prompts, reused prefixes, Unicode,
multiple simultaneous sessions and sustained batches. Report cold versus warm
tables/graphs, first-use compilation, host core usage, logical and observed
physical memory, TTFT, inter-token latency and completed useful throughput.
Include p50/p95/p99, cancellation, rejection and fairness in serving workloads.
Measure GPU text processing while model inference is active, not only in an
isolated tokenizer benchmark where compute and memory bandwidth are idle.

**Gate:** identify a reproduced critical-path cost and retain an equivalent
baseline. Kernel timing, host-only lookup timing and GPU utilization alone
cannot establish a request-level speedup.

## 2. Reuse prepared execution through captured command graphs

Prepare bounded graph variants from the existing typed model/operation plan.
Use CUDA Graphs and HIP graphs on qualified native providers; evaluate recorded
commands/replay on Vulkan according to actual capabilities. Record unsupported
features explicitly rather than equating native and Vulkan mechanisms.

Preallocate admitted parameter/input/output buffers. Bind token, position,
valid-prefix length and current private state safely for each execution. Guard
source identity, numeric policy, geometry/capacity, device/context, compiler and
all captured allocation generations. Invalidate or safely rebind graphs after
restore, reset, buffer replacement, device fencing or incompatible configuration.
Retain original leases through replay and actual completion, including graphs
whose contents refer to buffers skipped by a state-only prefix.

**Gate:** compare replay with the existing uncaptured path on the same trained
sources: full logits, all required state domains, ordered diagnostic traces and
zero signs. Exercise nonzero causal state, changing valid lengths, shape-band
boundaries, multiple sessions, snapshots, rollback, admission refusal and real
post-launch cancellation/quarantine. Report graph construction/cache cost and
extra memory alongside end-to-end latency; preserve qualified context limits.

## 3. Move bounded generation and task scheduling onto the device

Progress from graph replay to bounded device-side decode loops/conditional
graphs where supported, then evaluate persistent kernels with device task
queues and dependency events when profiling justifies them. Keep implementations
as provider execution variants of the same semantic plan.

Keep sampled token feedback, supported EOS/stop checks, per-step parameter
updates and eligible grammar/sampling work on device. Research device-side
batch/KV metadata updates within already admitted resource grants. GPU workers
must not invent global memory admission or a second session-state authority.

Use bounded work windows and output queues so cancellation, fairness and
backpressure remain observable. Define actual completion at the authorized
transaction frontier; return ordered provisional state/output receipts to the
host SessionAuthority. Publish only validated committed prefixes. Preserve
exact speculative acceptance, rejection, RNG and all-domain rollback semantics.
Evaluate CPU/GPU overlap as a separate strategy when host control remains useful.

**Gate:** exact results against the existing step-by-step executor, including
early EOS, token limits, stop/grammar boundaries, partial accepted prefixes,
concurrent sessions, queue saturation, stale epochs and real cancellation at
multiple device-loop points. Unknown completion retains all referenced originals
and fences the device. Measure latency, useful throughput and fairness; an
unbounded persistent kernel or fewer host wakeups is not an acceptance result.

## 4. Add exact GPU tokenization as an optional placement

Define source-bound prepared vocabulary, merge ranks and byte/normalization
tables from the original tokenizer assets. Qualify the complete configured
pipeline: normalization, added/special-token matching, pre-tokenization, BPE or
other selected algorithm, token-ID assembly, BOS/EOS and request policies.
Separate GPU BPE primitives from a qualified complete model tokenizer.

Start with an explicit supported configuration and extend by semantic rules,
not model-name guesses. Preserve official chat-template behavior. Any retained
host normalization or request-policy stage must be visible in execution receipts
and included in complete-call measurements. Feed GPU-produced IDs directly to
GPU prefill where possible; CPU ID readback is a diagnostic or declared caller
requirement, not an automatic dependency of the GPU consumer.

Batch independent inputs without merging their semantic boundaries. Evaluate
declared CPU placement for short inputs and GPU placement for long/batched
inputs. Base any adaptive crossover on measured complete-call costs, concurrency,
transfer/topology and available resources; no universal character-count threshold.

**Gate:** complete ordered token-ID equality with the official CPU tokenizer
loaded from the same exact assets. Cover empty/short/long text, all byte classes,
Unicode normalization and combining marks, whitespace/newlines, digits, special
tokens adjacent to ordinary text, byte fallback, merge-rank ties, batch seams,
changing/reused workspaces and malformed-input policy. Test relevant normalization
and pre-tokenizer variants rather than assuming that a similarly named Qwen
family shares identical assets. Qualification on one vocabulary or GPU does not
qualify arbitrary tokenizer.json files, Qwen3.5, AMD or another device.

## 5. Add exact GPU detokenization and output processing

Evaluate vocabulary-byte lookup, output-length prefix scans, byte assembly and
eligible incremental decoder/stop processing on GPU. Keep tokens and prepared
decoder tables resident where useful. Transfer bounded completed output chunks
at the API/console boundary, accounting for queueing and copies in measurements.
Batch independent sessions while retaining separate decoder state and frontiers.

Match the source decoder's byte mapping, special-token options, byte fallback,
normalization/cleanup behavior and incomplete UTF-8 handling. Stop strings can
span token or output-chunk boundaries. Decoder state must follow the authorized
committed prefix through rewind, speculative rejection, snapshot and cancellation;
never expose speculative text or emit a prefix twice after replay.

Keep the reference CPU decoder as a declared placement when it is cheaper.
Distinguish neural-network autoregressive decoding from detokenization: moving
token IDs into text bytes does not add or remove model GPU compute coverage.

**Gate:** byte-identical final text and equivalent documented streaming behavior
against the source CPU decoder across adversarial token/chunk boundaries,
Unicode fragments, special tokens, cleanup settings, stop trimming and rejected
drafts. Require bounded output admission, no session cross-talk and correct
rollback/cancellation. Prefer complete-request gains over isolated bulk byte-copy
speedups that increase interactive latency.

## 6. Joint placement and release acceptance

Retain independent choices for text encoding, model execution, execution control
and text decoding. A fully resident GPU model, GPU tokenizer, GPU detokenizer,
captured graph and device-driven generation are separate reported capabilities.
Avoid a single full-GPU flag that implies unqualified frontend/control support.

Compare relevant CPU/GPU text placements, eager versus replayed execution,
bounded device loops and persistent scheduling under identical model/tokenizer
assets, numeric policy, workload, hardware, power and cold/warm conditions.
Report native CUDA, native HIP and Vulkan results separately. Preserve failed
experiments and measured regressions; retain simpler qualified strategies where
they win. Do not claim that these studies already demonstrate an improvement
over another runtime or permit broader contexts/model families.

Implementation order is profiling, captured replay, bounded device execution,
then measured GPU tokenizer/decoder candidates and joint qualification. Text
codec research may proceed independently when it does not interfere with native
backend bring-up. Persistent scheduling follows a measured need and complete
ownership/fairness gates. Track each bounded implementation slice, source/model
identities, actual hardware/numerical results, failures and unfinished gates in
the owning plan/run TODOs under [TODO-SPEC.md](../plans/TODO-SPEC.md). Commit
verified implementation, tests and plan updates together; no generated handoff,
archive or documentation SHA closeout is required. Existing phase IDs remain.

## Research evidence and its limits

These sources motivate execution constructs, not dependencies to install or
foreign runtimes to embed. Recheck versions, licenses, actual source behavior
and measured claims before implementation; no download or SDK/device-access
authorization is granted by this roadmap.

- [vendor GPU tokenization](https://docs.rapids.ai/api/libcudf/legacy/group__nvtext__tokenize)
  supplies GPU BPE/WordPiece operations. Its primitives alone do not establish
  exact full-pipeline support for our tokenizer assets.
- [GraphTok research paper, v1](https://arxiv.org/html/2610.13325v1) describes GPU
  pre-tokenization, BPE and ID assembly with vLLM and other serving-stack
  integrations. Its tested
  configurations, CPU routing for short inputs, host Qwen NFC normalization and
  H100 NVL evaluation limit what can be inferred for our models/hardware.
- [llama.cpp CUDA Graphs](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
  demonstrates reduced submission overhead through graph replay. It does not
  mean that tokenization or the complete serving scheduler runs on GPU.
- A vendor overlap scheduler
  overlaps CPU scheduling/response/stop work with GPU computation; evaluate
  overlap independently from actually moving that work onto the device.
- [Mirage Persistent Kernel paper, v1](https://arxiv.org/html/2512.22219v1)
  demonstrates in-kernel dependency scheduling and device batch/KV metadata
  updates. Reuse the ideas only under our numerical/resource/state contract;
  its results do not qualify source-exact fusion or our GPUs automatically.
- [vLLM tokenizer configuration](https://docs.vllm.ai/en/stable/configuration/optimization/)
  documents its standard Hugging Face tokenizer and optional Rust replacement;
  that does not imply default GPU tokenization.
- [SGLang detokenizer implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/detokenizer_manager.py)
  illustrates a host detokenization process with incremental text state. It is
  a semantic/placement reference, not evidence of a GPU text decoder.

## Implementation integration

Use the current plan TODO for ownership and applicable gates, with separate
agent run TODOs until coordinator integration. Preserve unrelated work and the
user-authored root README. Commit only verified, scoped changes and carry every
unfinished item forward. This roadmap remains a design reference; reading or
linking it does not establish implementation or qualification.

````

## Source: .agents/research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md

Original UTF-8 bytes: 15112; SHA256 `86de7d64a9bf8246fa1fbd071d4193db425f4323ed52794cfed0db9a6c205bdb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — Heterogeneous prefill and retained replay acceptance

**15 October 2026 · U-048/U-049 · DESIGN ONLY.** Implementation remains at P07-H6
(`eacf7ef`); the requirements and 40 acceptance gates below are not new runtime or
performance qualification. Root README and inference code remain unchanged.

## 1. Design objective and evidence boundary

Qualify native packed-weight/ISA kernels, retained replay, cooperative NPU/GPU
prefill, numeric/state provenance and feature composition within the existing
SemanticProgram → ExecutableBundle architecture. Preserve one Rust mathematical
engine, one physical resource authority and one TaskCoordinator/StateRoot publisher.
No additional model-specific runtime, scheduler or durable IR is required.

Existing design boundaries cover source kernels/encodings, closed model regions,
compiled command graphs, speculation, unified/tiered memory, multi-GPU placement,
continuous serving, and typed media/decision tasks. The requirements below make
banked prompt execution and replay ownership explicit at those same boundaries.

H6 qualifies selected source-exact Qwen3.5 0.8B/2B regions on Renoir/RTX3060 Vulkan
with per-region host readback. It does not qualify a resident GPU graph, batched
NPU prefill, large-MoE execution or competitive throughput. Current native CubeCL
CUDA/HIP compilation is blocked by LLVM23; future qualified native ISA/NPU program
paths need not depend on that specific compiler or XRT.

## 2. Capability ownership and qualification

All rows describe **design expressibility**, not implemented support. Exact default,
opt-in, experimental, unsupported and deprecated states are per source tuple.

| Capability / mechanism | Architecture owner | Qualification requirement |
|---|---|---|
| Rust host + HIP/ISA kernels; runtime without full ROCm SDK | DEVICE_KERNEL_AND_CODEGEN; accelerator provider | separately package compiler/build and driver/runtime; Rust kernel/code-object path, ABI/source/ISA tests |
| Packed affine/codebook quantization, role-specific recipes and overlays | FORMATS_QUANTIZATION_AND_LOADING; verified assemblies; derived quant | independent codec/scale/transform oracle and exact sidecar binding; reader support is not quality admission |
| Tuned GEMV/GEMM/verifier shapes, LDS tiling, fusions | GFX1151 contract; KIR/CIR | register/WMMA/barrier rounding and external GPU co-tenancy tests; no new model engine |
| Resident token graph; HipGraph / retained AQL / PM4 | execution families; CIR | complete recorded launches, real route receipts, patch bounds, code/pointer generations; keep transports distinct |
| Two-bank layer-major NPU trunk | cooperative NPU spec; closed ModelStage region | explicit bank/state/retirement algebra and direct arena consumer, beyond output-row splitting |
| Native amdxdna + Rust-authored AIE programs | optional NPU provider | remove XRT-only feasibility assumption; qualify actual program/driver transport, not require another serving runtime |
| NPU W8/BF16 direct compute | NumericContract; derived artifact/encoding policy | exact twin proof separately from source quality; realized segment/family regime in state/cache identity |
| Pinning, huge pages, shared LPDDR/power, calibrated crossover | physical arbiter; UMA safety; cost model | account driver pin vs mlock; full extra banks/panels; safety veto independent of perf prediction |
| Native MTP, DFlash/DSpark, lookup and gated EAGLE paths | target-authoritative speculation + ModelAssembly | per-source tap/head/RNG/rollback oracle, accepted tokens, truthful feature-combination admission |
| QSA/MLA/recurrent state and long-context KV | hybrid operators; state domains | actual source operators through whole-model128K/256K where model/hardware permit; no sparse-budget reduction |
| Q8/FP8/BF16/FWHT/Lloyd and VMM KV | memory/state numeric encodings | state compatibility/transform quality and capture-safe mappings; no blanket codec support |
| Session append, exact tool-turn tokens, prefix snapshots, disk tier | full-state prefix/SSD contract | original emitted IDs and canonical rendered history, all-domain frontier meet, crash/disk integrity |
| Continuous lanes, MTP rows, shared immutable weights | joint serving/goodput contract | exact per-session state, actual verifier-row budget, singleton fallbacks and model/feature restrictions |
| TP/EP/PP, heterogeneous GPU and mapped experts | placement/remote/expert contracts | topology/collective state proof, measured link cost; NPU+EP not inferred from separate support |
| Curated pull/list/run/chat/serve/bench/quantize/config, aliases, sidecars, TUI/UI | product, CLI/config, trust and packaging | declarative capabilities/profiles; explicit acquisition authorization; frontends share one engine |
| Chat SSE/tools/reasoning/logprobs, Messages/Responses adapters | canonical TaskRequest/API spec | schema/event/error/SDK subset tests; no tools executed by inference |
| Embeddings including late-interaction multi-vector query/document mode | verified task/encoder/pooling | preserve token matrix shape/truncation/markers; never silently replace with pooled embeddings |
| Finite decision readout | typed Decision TaskDriver | explicit label/method/calibration/session contract; explicit task and API capability declaration |
| Vision/OCR sidecars; rectified-flow generation/edit | v5.1 assemblies/media TaskDrivers | each checkpoint/component/solver/processor actually source-qualified; source family map is not a model support guarantee |
| Health/stats/metrics, queue limits, model reload, Nix/container, diagnostics | core/serving/security/release | truthful hardware/feature readiness; bounded backpressure, diagnostics, fault and service recovery tests |

Unsupported or unqualified precision, eviction and speculative combinations remain
explicit research candidates. They cannot become defaults merely because a kernel
exists. New mechanisms attach through the same model/task/encoding/provider
boundaries after semantic and quality qualification.

## 3. Required design contracts

1. Support same-op output-row slicing and two-bank prompt-row, layer-major
   pipelining as distinct transformations. Use a separately qualified legal
   region candidate, with within-session causal A→B state and bank-private data.
2. Support qualified native amdxdna/Rust AIE or optional runtime paths.
   Specify transport/program capabilities; retain current Rust-source policy.
3. Readiness must distinguish terminal error notification from
   successful data readiness; no signal/watchdog may waive DMA/reader lifetime.
4. Generic graph capture needs complete launch accounting, bounded
   patches, transport-specific route receipts and loaded-code lifetimes.
5. W8/BF16 paired and native unpaired tails realize different math.
   A generic tolerant-mode name is insufficient for snapshot/prefix compatibility.
6. Feature support must admit the composed tuple, not independently
   tick NPU, MTP, replay and EP boxes. Performance calibration cannot override
   correctness, source, topology or power/fabric safety.

Radeon 8060S has **40 CUs**; AMD’s [versioned GPU
specification](https://rocm.docs.amd.com/en/docs-10.0.0/reference/gpu-specs.html)
is the primary reference. Wave32 is a kernel profile, not the only hardware wave
mode. Measured costs must use the actual device profile.

These are specified in the [normative pipeline contract](../docs/HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md)
and linked into their existing owners. No runtime feature is marked complete.

## 4. Acceptance matrix — ALL PLANNED / NOT RUN

Existing H6 tests are separate evidence. None of these new gates passed merely by
adding this document. Each requires raw source/binary/model/config/route receipts.

| ID | Gate / rejecting case |
|---|---|
| HP-01 | import exact source packed quantization bits, scales, transforms and overlays with independent decode oracle |
| HP-02 | distinguish lossless repack from W8/requantized derivative and incompatible source identity |
| HP-03 | reject missing/wrong draft, vision, head, overlay or PLE sidecar and unsupported family recipe |
| HP-04 | source tensor/WMMA/LDS/rounding/guards across decode, verify and prefill shapes |
| HP-05 | compare eager/fused under same numeric contract; preserve recurrent state and source sparse IDs |
| HP-06 | co-tenant GPU clients stress LDS read retirement and shared scratch generations |
| HP-07 | all-source op and mutable-state coverage before a resident whole-token graph claim |
| HP-08 | recorded launch count reconciles with eager plus named external boundaries; omitted KV write rejected |
| HP-09 | eager / HipGraph / AQL / PM4 independently show actual executed route and source/state equality |
| HP-10 | wrong code-object ABI, artifact, alias, grid, kernarg or patch bounds fail prepare |
| HP-11 | idle prepared plans retain pointer/code identities or invalidate before eviction/rebind; in-flight reset/cancel waits for retirement |
| HP-12 | per-ISA acquire/release and multi-queue fan-in; unknown effects serialize |
| HP-13 | banked candidate proves exact prompt-range coverage and A-before-B causal domains at every layer |
| HP-14 | live-across-switch activations/HC gates private; scratch/arena reuse waits for last consumer |
| HP-15 | short/ragged/rebalanced pairs, padding zeros and private output row ranges match declared oracle |
| HP-16 | partial-core snapshot cuts capture all source state domains before later-bank overwrite |
| HP-17 | NPU integer/grid-scale/BF16 CPU oracle, GPU twin and actual NPU bytes agree |
| HP-18 | same candidate separately passes quality vs original/selected quant target and sampled distribution where claimed |
| HP-19 | numeric schedule/family/cuts/overlay change invalidates incompatible prefix and compiled artifacts |
| HP-20 | native amdxdna or optional runtime transport returns authentic readiness with target program identity |
| HP-21 | terminal signal on failure releases waits without authorizing success/publication |
| HP-22 | delayed DMA/error/unwind/cancel/quarantine keep arenas/registrations until genuine quiescence |
| HP-23 | bounded pinned-host/driver-pin/bank/panel/program memory admission; no APU double-count |
| HP-24 | SoC firmware/power/clock/IOMMU/co-tenancy reliability before automatic NPU; no hidden host tuning |
| HP-25 | keyed measured family/pair schedule, tail padding and LPDDR contention; unknown profile uses qualified GPU |
| HP-26 | no synchronous per-layer host wait, repeated layout conversion or hidden same-projection GPU duplicate |
| HP-27 | MTP/DFlash/DSpark/lookup/EAGLE source attach, greedy target verification and all-state rollback |
| HP-28 | sampled target distribution/RNG, deterministic plan selection, alternate legal routes and penalties |
| HP-29 | useful accepted tokens/target traversal; record speculation losses, fallback and padded expert work |
| HP-30 | long-context source QSA/MLA/GDN/KV/indexer, compressed cache quality and exact capability limits |
| HP-31 | exact tool-turn emitted-ID continuation; edited history misses; all-domain prefix/disk restore integrity |
| HP-32 | concurrent lanes, variable draft depths, real sum(1+d_i) verifier rows, fairness/TTFT/ITL under load |
| HP-33 | independently admit NPU×replay×speculation×TP/EP combinations; reject unqualified tuples |
| HP-34 | source-correct TP/EP/PP and heterogeneous expert traffic/collectives with no implicit full-weight copies |
| HP-35 | CLI/model lifecycle/config/install/sidecar/management behavior and capabilities match declared subset |
| HP-36 | Chat/Messages/Responses streaming, tools/reasoning/errors/disconnect and backpressure conformance |
| HP-37 | pooled vs multi-vector embeddings and query/document preprocessing; finite decision logits/calibration |
| HP-38 | vision/OCR/rectified-flow task frontiers, edits, component/solver provenance and source-specific outputs |
| HP-39 | matched pinned-baseline ABBA: cold/warm TTFT, decode AR/spec, context/quality/concurrency |
| HP-40 | whole-workload speed/goodput confidence, quality/SLO/memory/energy/reliability; negative results retained |

## 5. Integration order and comparison policy

P03 source/codec/assembly qualification; P05/P07 resident/native execution and
replay; P08/P11 complete model + causal state; P10/P12 state/weight/lookup tiers;
P15 speculation; P16 joint row-aware scheduling; P19 cooperative NPU/multi-GPU;
P17/P18 product/model/media breadth; P20 calibration; P21 packaging/trust;
P22/P23 matched performance and composition gates. Continue current H6 successor
in small verified phases; this review does not skip its incomplete operators.

To claim **faster**, first match exact deployed artifact/sidecars, sampler, prompt
IDs, KV/state numeric regime, context, device/RAM/kernel/firmware/driver/power and
features. Repeated interleaved fresh processes, identical output horizon, explicit
cold/warm cache and p50/p95/p99 uncertainty; count all accepted tokens and all
requests including errors/rejections. Separate source-equivalent comparison from
quality-normalized comparison across derived packs. Report load/prepack/pinning
and original-vs-derived quality independently. No lower precision, fewer experts,
shorter context, cached-prefix advantage, different speculative horizon or kernel
tuning may be concealed as an engine-only speedup.

Further gains are hypotheses: remove repeated host roundtrips/packing; reuse
activations across compatible fused kernels; choose resident/replay/native dispatch
by actual cost; couple expert-row grouping and speculation with NPU bank shape and
SLO; reduce shared-memory traffic without losing source math. Shared LPDDR/power
and Amdahl limits apply. A broader architecture creates options, not a performance
proof, and implementation/generalization overhead must be measured explicitly.

## 6. Review invariants and document validation

Per-domain causal edges attach to the phase that owns each mutation: PLE Prepare
A→B, GDN/KV/indexer Core A→B. Interior snapshots preserve every domain before later
bank writes. Retained pointers and loaded modules remain valid throughout an idle
prepared plan's lifetime or the plan is invalidated before eviction/rebind; submitted
work additionally requires real retirement. HP-11 checks both obligations.

U-049 retains these technical constructs and all 40 gates while removing external
implementation/release attribution, source-comparison narrative and imported
benchmark figures. Acceptance IDs use the heterogeneous-pipeline prefix HP.
Validation covers document links, unique gate IDs, archive inclusion and SHA
integrity, repository quality checks and the unchanged README byte hash. All new
physical/model/performance gates remain **NOT RUN**.

Document cleanup validation PASS: current versioned files/paths reference scan,
all 40 retained planned gates, 31 added local links, unchanged README/runtime,
repository `scripts/check.sh`, all 8 archive regressions and SHA integrity. These
document checks do not qualify the planned physical/model/performance gates.

````

## Source: .agents/research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md

Original UTF-8 bytes: 10023; SHA256 `5449aae6e719ee9b314c94cd588373f3dfadc4251034d55432c4fc4e00060745`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Initial hardware targets and capability optimization

16 October 2026. **User decision:** the initial target fleet is the current
machine plus SSH targets `pvl-a1` and `pvl-x2`: three CPUs and five physical GPUs.
Optimize all of them after native CUDA/ROCm enablement. The user explicitly
requests subagents for implementation work going forward. This document records
the target decision and planned acceptance; it does not qualify new backends,
models, context sizes, ISA kernels or performance superiority.

## Target fleet inventory

These are the initial discovered identities/capabilities, not a current free
memory budget or a new qualification. Recheck the selected profile before
hardware work. Completion and open work are in the [GPU/CPU TODO](../plans/cpu-gpu-optimization/TODO.md).

| Machine | CPU, topology and reported cache | Observed CPU instructions |
| --- | --- | --- |
| Current machine | Ryzen 7 5800H, 8 cores/16 threads, one NUMA node; L1d 32 KiB/core, L2 512 KiB/core, L3 16 MiB | AVX2, FMA, F16C; AVX512 absent |
| `pvl-a1` | Ryzen AI 9 HX 370, 12 cores/24 threads, one NUMA node; L1d 48 KiB/core, L2 1 MiB/core, L3 16+8 MiB. Topology reports four cores with 5.158 GHz maximum and eight with 3.289 GHz maximum | AVX2/FMA/F16C, AVX512F/DQ/BW/VL/IFMA, AVX-VNNI, AVX512-VNNI/BF16/VBMI/VBMI2/VPOPCNTDQ |
| `pvl-x2` | Ryzen AI MAX+ 395, 16 cores/32 threads, one NUMA node; L1d 48 KiB/core, L2 1 MiB/core, L3 2×32 MiB | AVX2/FMA/F16C, AVX512F/DQ/BW/VL/IFMA, AVX-VNNI, AVX512-VNNI/BF16/VBMI/VBMI2/VPOPCNTDQ |

| Physical GPU | Observed architecture and execution width | Memory and relevant limits |
| --- | --- | --- |
| Current RTX 3060 Laptop, PCI `0000:01:00.0` | CUDA compute capability 8.6 (`sm86`) | 6144 MiB dedicated VRAM |
| Current AMD Renoir | Actual HIP/COMGR target `gfx90c`, 8 CU, wave64 | Shared system memory; no WMMA/MFMA capability inferred |
| `pvl-a1` RTX 4060 Laptop, PCI `0000:64:00.0`, `10de:28e0` | Compute capability 8.9 (`sm89`); Vulkan subgroup32 | 8188 MiB dedicated VRAM; Vulkan shared workgroup memory 48 KiB, maximum 1024 invocations |
| `pvl-a1` Radeon 890M, PCI `0000:66:00.0`, `1002:150e` | Actual ROCm `gfx1150`, 16 CU, wave32; RADV subgroup64 default, controllable 32–64 | Shared RAM; sysfs 512 MiB VRAM aperture and 15.23 GiB GTT, not additive capacity; LDS 64 KiB, maximum 1024 invocations |
| `pvl-x2` Radeon 8060S, PCI `0000:c6:00.0` | Actual ROCm `gfx1151`, 40 CU, wave32; RADV subgroup64 default, controllable 32–64 | Nominal 64 GiB shared GPU pool; usable host RAM 67.04 GB decimal; LDS 64 KiB, maximum 1024 invocations |

Both remote AMD Vulkan devices report Mesa 26.1.8/Vulkan 1.4.354, FP64/F16,
I8/I16/I64, integer dot8 and cooperative matrices, plus FP32 signed-zero/NaN,
denormal and round-to-nearest preservation. The remote NVIDIA driver is
595.91.07; Vulkan 1.4.329 reports FP64, subgroup32 and FP32 denormal-preserve
false. These are advertised capabilities; actual kernel behavior remains a
separate gate. Vulkan CPU llvmpipe is excluded from the five-GPU count.

The qualified native a1 HIP profile uses its existing ROCm/runtime closure with
an explicit glibc2.44/GCC16.2 loader. The default2.42 loader cannot replace that
profile. Exact approved model identities are in [the accepted baseline](../plans/accepted-baseline/TODO.md).
SDK/compiler versions, usable memory, swap/load and power telemetry must be
measured for a new test; no global cache purge, downloads or device recovery is
implied. Old H24 discovery JSON/scripts/stderr are retired and are not required
execution inputs. The local gfx90c launch hold remains.

## Existing CPU execution and concrete opportunities

Read-only source audit: `crates/zerogravity-cpu/src/weights.rs` prepares source Q8 activation blocks
once per projection, reads original packed GGUF rows and uses Rayon parallel
rows. The integer dots and F32 dots express the qualified source's lane/FMA
reduction as portable Rust; there is no explicit runtime AVX2/AVX512 dispatcher,
CPU intrinsic implementation or model-specific topology-aware worker pool.
Compiler autovectorization is possible and must be inspected/measured rather
than assumed absent. `crates/zerogravity-model-qwen35/src/lib.rs` owns ordered GDN/attention dots and norms; the math module owns the source polynomial/FMA exponential schedule.

Implement capability-selected kernels inside those existing numerical
boundaries, retaining a qualified portable path. AVX2 and AVX512/VNNI can decode
packed weights, form exact integer lanes and process independent rows without
changing source scale/FMA/reduction order. Benchmark vector activation max,
round-to-even conversion, half conversion and quantization; preserve first
signed-maximum ties, NaN-to-integer behavior, subnormals and signed zeros.
Avoid saturating integer dot instructions unless the exact intermediate bounds
prove saturation impossible. BF16 availability does not authorize replacing
source FP32 operations with lower-precision arithmetic.

Measure thread counts, SMT on/off placement, cache-local row tiles, immutable
source-derived repacking and activation reuse. Admission must include derived
layouts; immutable source identity remains the original checkpoint. Evaluate
four fast-core versus all-core placements on the HX 370 using observed
frequency/cache groups and measured load; preserve small-operation serial
thresholds and reserve CPU capacity for GPU submission/tokenizer/serving work.
Tune the two L3 groups on MAX+ 395 independently of NUMA (one node was observed).
Do not add another scheduler authority or pin processes/change governors as an
unrequested persistent machine configuration.

## Ordered implementation and acceptance

After native qualification, follow the approved
[workspace/layout redesign](../plans/accepted-baseline/TODO.md),
then the immediately following [left-right redesign](../plans/cpu-gpu-optimization/TODO.md).
Implement measured optimization variants within that user-approved ordering.

1. **Complete native enablement first.** Follow
   [P07-H24](../plans/accepted-baseline/TODO.md); qualify compilation,
   physical identity and exact operators before complete CUDA/HIP models.
   Reuse one Rust model schedule and existing physical arbiters/leases/events.
   Never infer `gfx1150`/`gfx1151` support from a different RDNA target or count
   runtime initialization as a trained-model pass. Establish remote assets and
   their hashes before remote model qualification. The fleet request authorizes
   implementation and bounded temporary binary/evidence transfers and qualification
   workloads over SSH. Model/SDK downloads, persistent service/configuration or
   driver/device privilege changes, push and release retain their approval limits.
2. **Record end-to-end baselines on all targets.** Pin source/tokenizer,
   compiler/runtime/backend, numeric policy, topology, power policy, workload
   and resource limits. Separate cold compilation/load/prefill from warm cached
   prefill/decode, fresh versus reused prefixes, interactive versus batched
   concurrency, and CPU/GPU/mixed placement. Measure TTFT, inter-token latency,
   completed useful tokens/s, p50/p95/p99, queue/admission delay, host core time,
   actual transfer bytes, memory residency and cancellation latency. Retain
   short and long contexts only within independently qualified boundaries.
3. **Tune CPU variants and GPU kernels in bounded units.** Qualify AVX2 then
   AVX512/VNNI candidates, cache/thread policies and source-quantized GEMV;
   tune GPU row/block tiling, subgroup widths, vector loads, admitted packing,
   occupancy and memory reuse per observed architecture. Compare exact fused
   and prepared execution before adding complexity. Consider tensor/MMA/WMMA
   or accelerated integer dot only when their accumulation, saturation and
   rounding match the selected source contract; FP16/BF16/TF32 substitution
   cannot satisfy the current exact FP32 contract by changing tolerances.
4. **Qualify shared-memory and heterogeneous placement.** Couple CPU and APU
   pressure against the same physical RAM pool; Vulkan heaps, HIP free memory,
   VRAM apertures and GTT counters may overlap and must never be summed. Charge
   immutable sources, candidate/committed roots, staging, compiler/workspace
   and quarantine; retain OS/desktop headroom. Exercise CPU/APU/dGPU concurrency,
   bandwidth contention, cross-backend access to the same PCI device and fair
   bounded service. Multiple GPUs are placement choices, not permission to
   shard a stateful model without the existing distributed/ownership contracts.
5. **Reduce complete-request overhead after measured bottlenecks.** Use the
   [GPU text/control roadmap](GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md)
   for graph replay, bounded device generation, tokenizer/decoder placement and
   overlap. All additions remain placements of one engine and session authority.

Every optimization must pass unchanged independent original-source criteria,
complete logits/all state domains/ordered traces and zero-sign reporting,
multi-token nonzero state, replay/checkpoint/rollback, stale ownership refusal,
physical readback/admission and real cancellation/fencing/original quarantine.
Preserve known independent CPU 8K drift and deferred larger-context boundaries.
Then require a repeated end-to-end benefit under comparable conditions, retain
negative results and report scope per CPU/GPU/backend/workload. Kernel timing,
peak utilization, fewer launches or advertised hardware features cannot prove
that the hardware is fully used or another runtime is outperformed.

Runtime selection keys must be capabilities: ISA/OS vector support, topology,
cache groups, PCI identity, backend/architecture/subgroup, numeric features,
memory domain and qualified kernel/compiler version. Hostnames belong only in
fleet discovery and benchmark receipts. The goal is the best demonstrated
useful performance for these targets; each improvement remains empirically gated.

````

## Source: .agents/research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md

Original UTF-8 bytes: 24738; SHA256 `0a6d40f9e51aafb016b8379f5485bab3b6d29f30a0c524147db7e2a2617d77c1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Model pull, XDG artifact cache and pinned acquisition parity

**16 October 2026 — DESIGN ONLY.** No pull command, transport dependency, model download or acceptance result is implemented by this document. User-directed order: finish the [workspace redesign](../plans/accepted-baseline/TODO.md), complete the [contention-first concurrency review](../plans/cpu-gpu-optimization/TODO.md), then implement this pull milestone, before returning to GPU optimization. Existing local-file CPU/GPU inference and its qualification scope remain authoritative.

This refines the [routing/residency contract](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and acquisition scope of the [serving parity plan](VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md). It introduces no second inference engine, scheduler, catalog publisher or resource allocator. The user explicitly requested GitHub and Hugging Face pull plus source-pinned acquisition capability parity; implementing this plan may acquire only artifacts the subsequent pull invocation requests. This research fetched public source/docs and repository metadata only, never model weights or SDKs.

## 1. Primary-source pins and interpretation

| Reviewed project | Exact source revision | Commit timestamp (UTC) |
|---|---|---|
| vLLM | `a8be1500bad93d3c459e435bf71194aa8062e49d` | 2026-10-16 00:21:24 |
| llama.cpp | `2851b3bcbf8d993a2243ae751c0a7dfa8e944a40` | 2026-10-15 23:46:42 |

The research pins below remain the reference baseline, independent of the numerical oracle revision. Raw research captures were retired under the [accepted checkpoint](../plans/accepted-baseline/TODO.md); primary links remain below. The [bounded cache/transport specification](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) and its 57-row provisional acquisition inventory define future work. Their gates remain NOT RUN; revalidate selected upstream surfaces before claiming parity. Optional delegated packages remain separately unqualified.

| Key | Pinned source / primary documentation |
|---|---|
| V-W | [vLLM weight download helpers](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/weight_utils.py) |
| V-R | [vLLM revision, repository and non-weight snapshot helpers](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/transformers_utils/repo_utils.py) |
| V-L | [vLLM default loader](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/default_loader.py), [load configuration](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/config/load.py), [registered built-in loaders](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/__init__.py) |
| V-O | [Run:ai object schemes and sidecar pull](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/transformers_utils/runai_utils.py), [streamer loader](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/runai_streamer_loader.py) |
| V-T | Serialized-tensor URI/credential boundary, [ModelExpress delegation](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/modelexpress_loader.py) |
| L-A | [llama.cpp acquisition arguments and task construction](https://github.com/ggml-org/llama.cpp/blob/2851b3bcbf8d993a2243ae751c0a7dfa8e944a40/common/arg.cpp) |
| L-D | [llama.cpp HTTP, split-file, sidecar and Docker acquisition](https://github.com/ggml-org/llama.cpp/blob/2851b3bcbf8d993a2243ae751c0a7dfa8e944a40/common/download.cpp) |
| L-C | [llama.cpp Hub cache and revision metadata](https://github.com/ggml-org/llama.cpp/blob/2851b3bcbf8d993a2243ae751c0a7dfa8e944a40/common/hf-cache.cpp), [endpoint configuration](https://github.com/ggml-org/llama.cpp/blob/2851b3bcbf8d993a2243ae751c0a7dfa8e944a40/common/common.cpp) |
| H | [Hub file/snapshot/revision/filter API](https://huggingface.co/docs/huggingface_hub/en/guides/download), [Hub environment/auth/offline settings](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables) |
| G | [GitHub release assets](https://docs.github.com/en/rest/releases/assets), [repository contents and size limits](https://docs.github.com/en/rest/repos/contents), [Git LFS pointers](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) |
| X | [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir/latest/) |

## 2. Acquisition inventory and explicit parity disposition

Every row below is planned. Complete pull acceptance requires the applicable acquisition rows, with a precise Unsupported response for any intentionally unimplemented provider. A partial first transport must not be announced as complete reference-tool parity. Loading strategies and arbitrary plugin behavior have separate dispositions below.

| Capability class | Evidence at the pins | Zerogravity disposition / acceptance boundary |
|---|---|---|
| Hub repository, exact file, subfolder and revision | V-W accepts revision/subfolder/filter selection; H documents commit/branch/tag and single-file or snapshot acquisition. | Native Rust Hugging Face source resolver; resolve mutable reference once, persist exact revision and file list, support explicit files and bounded include/exclude filters. |
| Quant selection | L-A offers case-insensitive quant shorthand, default Q4_K_M and documented fallback, with explicit file overriding shorthand. | Offer explicit quant/file selection and discoverable candidates; refuse missing or ambiguous requested quant. Any convenience default must print resolved filename/quant and obey explicit policy; never silently change a caller's numeric contract. |
| Checkpoint shards and index | V-W uses safetensors weight-map filenames and downloads index metadata; L-D identifies numbered split GGUF parts. | Resolve complete required shard closure before transfer/publication; validate unique ordered parts, index paths and total bounds. Acquiring a format does not make its loader executable. |
| Config/tokenizer/template/license/quant metadata | V-R obtains non-weight snapshots; V-W obtains quant metadata separately. | Artifact manifest declares model-required config, tokenizer vocabulary/merges/special tokens/template, quant config, model card/license and applicable adapter metadata at their own pinned revisions. Required sidecar absence prevents Ready. |
| Auxiliary and speculative artifacts | L-D selects mmproj, MTP, DFlash, Eagle3 and DSpark; L-A separately accepts draft Hub/URL inputs and a preset file. | Explicit component roles and dependency manifests; pull/cache these bytes without claiming multimodal/draft execution. Remote preset/config remains bounded data, cannot authorize more sources, override local limits or execute code. |
| Direct model and auxiliary HTTP URL | L-A constructs model/mmproj/draft URL tasks; L-D handles HTTP transfer. | Explicit HTTPS source with expected digest when supplied; normalized origin and final content hash recorded. Plain HTTP only through an explicit local/testing or separately configured insecure-source policy. |
| Retry, progress, cancellation and resume | L-D has bounded retry, progress callback cancellation, partial files, range requests with 206 validation, ETag reuse and atomic file move. | Bounded configurable retry/backoff/timeouts/concurrency, monotonic byte receipts, validator-bound resume and deterministic cancellation. ETag is a change validator, not a cryptographic content identity. |
| Cache and offline mode | L-C uses Hub-compatible blobs/snapshots/refs and cache environment precedence; L-D supports cached-only reads. V-W/V-R pass Hub offline settings. | Native XDG cache plus explicit cache-dir override; inspect/list and scoped removal; genuinely zero network in offline mode including metadata, DNS, auth and background refresh. Import existing external cache files read-only through verification, without ownership/deletion rights over that cache. |
| Private/gated repositories and endpoints | L-A uses HF_TOKEN; L-C endpoint configuration accepts MODEL_ENDPOINT with HF_ENDPOINT compatibility. H documents auth/cache/offline environment. | Origin-scoped credentials, private cache permissions, license/gate error distinctions, explicit mirror endpoint/provenance policy, no automatic license acceptance. Support documented compatibility aliases only with clear precedence. |
| an alternative model hub | V-W optionally delegates snapshot download with revision, filters, cache and offline arguments. | Separate optional provider adapter and contract tests; preserve native manifest/source rules, do not add a Python runtime dependency. |
| S3, GCS and Azure objects | V-O explicitly recognizes s3://, gs:// and az://; its loader distinguishes object paths from local/Hub paths and pulls metadata through a delegated package. | Individually qualified object-provider adapters: exact object version/generation or digest, listing pagination, authenticated ranges and complete manifest publication. S3-compatible endpoint support is explicit, not an assumption that all clouds share one API. |
| Docker Hub / OCI GGUF layers | L-D resolves Docker Hub token/manifest and selects recognized GGUF layer media types; L-A exposes docker-repo. | Optional OCI resolver pins manifest and layer digests, verifies complete blob, stores model artifact only; never starts a container. Do not claim generic registry/auth parity before those protocols are tested. |
| Remote serialized-tensor streams | V-T validates local or s3/http/https URIs and delegates stream opening; this is coupled to its serialization loader. | The applicable URL/S3 acquisition capability is covered by native transports. Serialized-tensor deserialization, encryption and tensor streaming require a separate qualified format/loader proposal; cached opaque bytes are not runnable weights. |
| GitHub repository and release artifacts | G documents immutable ref selection, content API limits, release asset metadata/digest and LFS pointer indirection. | Explicit user-requested extension: pinned repository file/tree and pinned release asset with exact identity/digest; no claim that a built-in vLLM loader provides generic GitHub pull. Resolve LFS pointer objects through a separately tested flow or refuse clearly; never publish the pointer as a weight file. |
| Dry-run, cache inventory and removal | H exposes dry-run file/size planning; L-D exposes cached listing, path resolution and removal helpers. | Resolve/print bounded transfer plan without body transfer, cache list/inspect/verify and explicit removal with active-lease protection. Offline dry-run resolves only retained metadata and must label unavailable sizes. |

**Loader classification:** V-L registers auto/hf, safetensors, pt, npcache, mistral, fastsafetensors, instanttensor, runai_streamer, runai_streamer_sharded, sharded_state, a serialized-tensor loader, ipc_cache, modelexpress and dummy. Auto format choice, lazy/eager/prefetch/torchao, CPU/GPU placement, quant reconstruction, distributed shard loading, CUDA IPC cache and random dummy weights are loading/execution mechanisms, not additional pull protocols. ModelExpress delegates download/load behavior to an optional external package (V-T); unbounded behavior of that package or custom loader plugins is not a finite acquisition parity target. Inventory a concrete missing transport before proposing it. No Python plugins or remote-code execution are introduced.

The current source-qualified Qwen GGUF formats remain the inference boundary. Safetensors/PT/serialized-tensor/OCI-contained formats, sidecars and new families may be acquired/inspected only as data until their parser, model schedule, numerics, state and physical execution are independently accepted. Default CPU/toolchain isolation must remain intact.

## 3. Ownership and public command contract

1. Keep acquisition, manifest validation and cache storage in the artifact ownership layer established by H25; optional network/provider features do not enter std-only core, model mathematics, CPU/GPU kernels or the minimal inspection CLI. Add a separate cohesive artifact-transport crate only if real dependency/platform separation warrants it.
2. CLI owns `pull` syntax and progress; facade exposes the same typed operation. Proposed selectors distinguish `hf:`, `github:` and direct/object/OCI URLs; exact spelling is an implementation decision. No family-specific downloader, shell subprocess chain, git checkout hooks or imported Python engine is needed.
3. `pull` is explicit acquisition authorization. `run`/`inspect`/catalog miss never implicitly accesses the network. A later intentional auto-pull policy would need a separately visible user choice. Offline mode overrides provider settings and background helpers.
4. A bounded `PullPlan` resolves provider, original selector, immutable resolved revision/object IDs, component roles, exact normalized relative paths, expected lengths/digests, license/access metadata and transfer limits. Resolve means metadata acquisition only; dry-run must distinguish missing metadata from zero bytes.
5. `PullReceipt` reports manifest digest, actual bytes transferred/reused/verified, resolved source, cache path, completed/failed/canceled status and unsupported execution capabilities. Fetching and verifying are never model-engine Ready.
6. Artifact-store immutable manifest/ref publication is distinct from the current Qwen executable `ModelCatalog::register(gguf, config)` / `load` publisher. Acquired unsupported formats remain artifact inventory only. Register a model through the existing sole executable catalog only after normal supported-family/config validation; keep artifact inventory publication with its actual store transaction owner. A different concurrent index requires representative contention evidence and semantic fit; left-right adoption is not a cache prerequisite. No read guard spans network, hashing, fsync, parsing, model loading, awaits or GPU work. Preserve single-flight loads and source/weak-lifetime semantics.
7. A durable artifact manifest digest is distinct from the process-local core `SourceIdentity`. Loading still uses the single checked issuer and original immutable source owner; no synthetic identity restoration, new session publisher or per-token provider dispatch.

## 4. Cache, validation and transaction design

Use `${XDG_CACHE_HOME:-$HOME/.cache}/abird/zerogravity` by default with a documented explicit override; ignore relative XDG values as required by X. Never repurpose HOME. Linux is the first qualified platform; other platforms need equivalent path/atomicity tests before portability claims. User configuration/credentials and durable alias policy belong in their appropriate config/data directories, outside a cache users may delete.

A proposed internal layout is `models/blobs/sha256/<digest>`, `models/manifests/<manifest-digest>.json`, `models/snapshots/<manifest-digest>/`, `models/refs/<provider+repository-key>/<selector-key>.json` and `models/staging/<attempt-id>/`. Names are implementation details; use collision-resistant encoded keys rather than remote paths as cache roots. Provider endpoints and authorization scope participate in resolution keys. Immutable content can be shared only where cache ownership/access policy permits.

- Validate bounded manifest/index sizes and JSON depth/counts, normalized relative paths, duplicate/case collisions, path traversal, symlinks/reparse behavior, object sizes, checked aggregate byte arithmetic and full required dependency closure before allocating transfer buffers or exposing snapshots. Preserve file extensions as metadata, not format proof.
- Metadata commits, object versions and expected digests stay fixed across a pull. A changed branch, revision, shard list, ETag or object generation cannot append into an earlier partial. Restart safely or fail with the original manifest untouched.
- Stream into private same-filesystem staging with incremental SHA256 and bounded buffers. Validate content length/digest and any available provider integrity metadata. A computed digest proves content identity; it is not proof of publisher authenticity when no trusted expected digest/signature exists. Record that distinction.
- For a URL without immutable source identity, a supplied expected digest gives reproducible acquisition. Otherwise publish a first-seen content-addressed snapshot and label its provenance weaker; subsequent resolution cannot silently mutate that snapshot. Never reinterpret ETag or Git blob SHA as SHA256 without its specified hash framing.
- Atomic publication order: complete and sync blob bytes; install verified immutable blobs without replacing conflicting contents; write/sync complete manifest and snapshot; atomically switch the small ref/alias binding; sync containing directories where supported. Readers see the old complete artifact or the new complete artifact, never partial shards. Define the supported filesystem durability boundary explicitly.
- Resume metadata includes provider/source identity, validators, expected length/hash, completed byte ranges and attempt ownership. Revalidate saved bytes before use. Verify Content-Range/status and never concatenate a fresh 200 response to a partial 206 target. Bounded recovery ignores orphan staging until verified.
- Cross-process coordination serializes publication for the same resolved object/manifest without keeping a catalog read guard. Concurrent writers deduplicate verified blobs; single-flight cancellation must not cancel another caller's live attempt. A failed publisher releases only its own staging/transfer resources.
- Secrets are obtained from explicit environment/config credential providers, never model files or command output. Scrub URLs containing credentials/query tokens and headers from receipts/errors/traces. Reauthentication and redirects follow origin/provider policy; no credential forwarding to arbitrary redirect or mirror hosts.
- `TransferStaging` and host-buffer allocations use the existing physical arbiter. Disk free-space/quota and concurrent transfer admission need a bounded artifact-store policy; the current GPU arbiter does not imply an implemented disk quota. Account temporary plus final publication peaks and manifest overhead honestly. Refuse over-capacity before body transfer when sizes are known; enforce streaming limits when unknown.
- Content store ownership/lease bookkeeping is separate from session StateRoot. Eviction/removal respects active source descriptors, loads, snapshots and encoded device uses; no unlink-based capacity claim until actual owned storage is releasable. Charge physical shared payload once for its full lifetime; index/log metadata and partial staging have their own charges.
- Preserve same-descriptor source materialization: validate and load the verified opened object, not an unchecked pathname reopened after catalog publication. Cache mutation/replacement must not change bytes beneath an existing immutable source owner.
- Required config/tokenizer/template/license metadata is data-only. Do not execute remote code, deserialize arbitrary pickle into the engine, run repository install hooks, auto-apply remote serving arguments or fetch transitive URLs merely because a config mentions them. Explicitly selected dependency components remain within the resolved plan and local policy.

## 5. Implementation sequence and finite completion gates

| Unit | Concrete result before proceeding |
|---|---|
| PULL-A cache transaction | Bounded manifest/path validation, XDG paths, immutable blobs/snapshots/refs, crash recovery, local import and offline inspect/verify/removal; no network needed in tests. |
| PULL-B first vertical slice | Explicit HF and GitHub file/release pull, HTTPS transfer, origin-scoped auth, dry-run, full sidecars/shards, progress/cancel/resume, original-model local-cache load through existing facade/catalog. |
| PULL-C pinned acquisition coverage | Optional alternative-model-hub, object-provider and Docker/OCI rows individually implemented and qualified; reference cache import behavior; finite source inventory updated with observed unsupported cases. No delegated Python loader parity claim. |
| PULL-D closure | CLI/API ergonomics, CPU-only/minimal dependency matrix, acquisition-to-load parity and concurrency/space/crash gates below; publish exact supported provider/format table and pending loader capabilities. |

The sequence starts after accepted layout and the concurrency review has a disposition for current owners. A justified no-migration decision satisfies that review; do not invent a left-right target to unblock acquisition. Provider branches may be implemented/tested in parallel with disjoint ownership once the shared cache contract is frozen. No acceptance depends on downloading new real checkpoints in this design task. Future physical acquisition of selected user-requested artifacts must use explicit pins and bounded expected sizes, never an unrestricted repository mirror.

All acceptance IDs below are **PLANNED / NOT RUN**:

| ID | Required gate |
|---|---|
| PULL-01 | Unset/empty/absolute/relative XDG, explicit override, private permissions, missing home and offline cache hit/miss; no hidden network. |
| PULL-02 | HF branch/tag/full commit/exact file/subfolder/filter and GitHub revision/release-asset selection; branch/tag moves during transfer cannot mix revisions. |
| PULL-03 | Requested quant missing/ambiguous, malformed or incomplete shard/index, metadata count/size/overflow and path/case/symlink/LFS-pointer refusal; no published alias changes. |
| PULL-04 | Multi-file sidecars at pinned revisions, tokenizer/template/license roles, unsupported component execution and remote preset/code references; cacheable data never implies runnable capability. |
| PULL-05 | Interrupted/resumed transfer, changed validator, 200-after-range, bad Content-Range, truncated/oversized/chunked stream and checksum mismatch; old artifact byte-exact. |
| PULL-06 | Fault injection before/after each blob/manifest/ref sync/rename plus restart; every visible artifact complete or absent, no mixed shards or lost prior ref. |
| PULL-07 | Threads/processes race same/different selectors; exactly one logical binding publication, no duplicate in-flight ownership, bounded queue/fairness, canceled caller cannot destroy a sibling pull. |
| PULL-08 | Private/gated auth, revoked/absent tokens, rate limits/backoff, HTTPS redirects and configured mirror changes; no credential leak or unintended cross-origin authority. |
| PULL-09 | Transfer/host/disk admission limits, full disk and unknown-size stream; verified releases of owned resources, no fake capacity or StateRoot/device-service mutation. |
| PULL-10 | Cache list/inspect/verify, scoped removal and orphan recovery while model/source/session/snapshot remains pinned; old loaded source survives alias update and re-registration. |
| PULL-11 | Each alternative-model-hub/S3/GCS/Azure/S3-compatible/OCI adapter: pagination/version/digest/auth/offline/cancel/range fixtures and explicit supported endpoint/profile receipt. External loader internals remain outside this gate. |
| PULL-12 | Local reference HTTP fixtures prove zero external network; offline tests additionally use actual network namespace isolation. Tests never depend on public weights, credentials, new SDKs or provider availability. |
| PULL-13 | Default CPU/minimal inspect/core feature closures remain SDK/transport-free where selected; no Python/runtime-engine dependency and one Cargo lock/source issuer/catalog publisher. |
| PULL-14 | Existing pinned trained Qwen .8B/2B fixtures imported into a fresh cache reproduce exact checkpoint bytes, source identity ownership, CPU output and qualified GPU state/logits through the same engine; no fallback to a different quant/model. |
| PULL-15 | Cold/hit/resume/reuse byte/time receipts with bounded concurrency, writer/read tails under the actual store protocol and fair admission; no inference throughput claim from download measurements. |

Completion requires exact implemented scope and limitations in the handoff, source/transport test receipts, strict build/lint/tests and documentation-integrity gates. Default numerical policy, CPU 8K GDN divergence, genuine 16K/32K qualification and fleet optimization remain separate work; acquiring longer-context or different-format artifacts does not close them.

````

## Source: .agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md

Original UTF-8 bytes: 25200; SHA256 `31e2e73b5228da9328f48bea4c0a76d8e5b31467bf20ee01b83ba55e88158660`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — Model-family research atlas and multimodal implementation roadmap

**14 October 2026 (Singapore).** Research-only model capability inventory and design decisions.
**Companion:** [Proposed unified multimodal execution contract](../docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md).
**Truth:** model names, released weight status, numerical method and API contracts are source-specific and time-sensitive. The external sources below identify the models; Zerogravity **does not currently run any listed media/decision/ternary checkpoint natively**. Only native Qwen3.5 0.8B and 2B CPU inference/8K acceptance and source-correct partial GPU region code exist. No model weights were downloaded.

## Research method and source hierarchy

Prefer (1) model authors' official open checkpoints, source repositories and papers, (2) upstream framework inference implementations, and (3) original vendor API contracts. A vendor statement about image quality/latency or license is not an independently reproduced Zerogravity result. Proprietary hosted model details below describe **API behavior**, not verifiable internal neural math. Some names have several architectures/variants; identify the **specific model checkpoint and release**, never just the family string. Multimodal timestamps/codecs can change independently of neural weights; pin both.

### Image generation / editing

| Model and availability | Source-backed execution family | Requirements for verified native model support | Primary source |
|---|---|---|---|
| **Qwen-Image 20B**, open-weight | Multi-modal Diffusion Transformer (MMDiT), conditioned latent image generation and editing; text rendering; text and image processors | MMDiT dual streams, image VAE and processor, text conditioning, sampler/flow schedule, guidance, multi-image editing, source diffusion noise | [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) |
| **Rectified-flow image model [dev] 32B**, weights under a non-commercial license; **[klein] 4B** Apache-2.0 (9B klein different license) | Variant-specific flow models: **[dev] uses a Mistral-class image/text conditioner, while [klein] 4B uses a Qwen3-4B embedding conditioner**; latent flow transformer + VAE, image references, and differing distillation/schedules | Bind conditioner/embedding type and dimensions to the exact checkpoint (do not reuse dev's Mistral configuration for klein's Qwen3 embedding), independent residency, latent packing, model-registered flow solver and guidance, VAE, source step budget | — |
| **Autoregressive multimodal image MoE / Instruct / distilled**, source and weight availability vary by variant | **The vendor describes a unified autoregressive multimodal 80B MoE** (64 experts, about 13B active), but **the exposed image-generation source path requires deeper auditing before declaring it purely AR versus diffusion/hybrid stages** | Register exact underlying image-generation stage semantics only after following the official generate_image and source causal-MM implementations; support both AR token/codecs and denoising components where actually present, rather than hardcoding either | — |
| **Stable Diffusion 3.5 Large/Turbo/Medium**, open-weight release under a source license | MMDiT-X / QK normalization, Large dual attention layers, multiple conditioner encoders, autoencoder and noise/timestep scheduler; Turbo distilled few-step | Per-variant checkpoint and licensing, multiple text encoders, MMDiT, VAE, guidance, scheduler, exact noise policy; distilled turbo steps are not transparent fast mode | — |

**Important split:** Qwen-Image, the rectified-flow image model and SD3.5 have distinct source latents/transformer/sampler semantics. The vendor describes the autoregressive image MoE as an AR multimodal MoE, but an upstream issue identifies diffusion-style paths in the published generation code; inspect the exact checkpoint and actual execution graph before claiming a pure autoregressive image decoder. An image TaskDriver must permit the correct source stages instead of forcing all image models into either one global denoising loop or one AR codec loop. Rectified-flow [dev] and [klein] differ by conditioner architecture, checkpoint, release terms and distillation; no imaginary unified tensor parser. Optional hosted frontier image models go through external TaskServiceAdapters with explicit egress, not untrusted model code.

### Speech-to-text, streaming recognition and speech understanding

| Family | Execution semantics | Specialization/quality contracts | Primary source |
|---|---|---|---|
| **OpenAI Whisper**, open source/weights | Log-Mel frontend, audio encoder and autoregressive text decoder with cross-attention; multilingual transcription/translation, timestamp and language tokens; default 30s windows | STFT/mel CPU/GPU DSP, encoder batching, cross-attn state, window/stride/overlap and timestamp alignment, dynamic beam, language task headers | [OpenAI Whisper](https://github.com/openai/whisper), [architecture](https://openai.com/index/whisper/) |
| **NVIDIA Parakeet**, vendor open weights | FastConformer audio encoder with **CTC, RNN-T or TDT** alternative decoding; v2/v3 language variants | cached encoder states, convolution subsampling, blank/predictor/joiner or duration transducer, beam decoding/timestamps, streaming state, no fake decoder LM | vendor speech toolkit, [checkpoint docs](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_checkpoints.html), [Parakeet model](https://build.nvidia.com/nvidia/parakeet-tdt-0_6b-v2/modelcard) |
| **NVIDIA Canary**, vendor open weights | Multitask multilingual ASR/translation with prompt/task slots and encoder-decoder transcription; variants may differ in features | language-conditioned decoder policy, source/target language slots, timestamp/prosody task flags, transcoding | vendor inference guide |
| **VibeVoice-ASR / Streaming / BitNet**, Microsoft published model/repo | Long-form transcription with diarization+timestamp+content; streaming recognition emits chunks; BitNet variant edge quantization | long audio source sample cursor, diarization identity, contextual vocabulary, chunk partial/finality, specialized I8/I2 tensor codec for BitNet | [Microsoft VibeVoice official](https://github.com/microsoft/VibeVoice) |
| **NVIDIA Nemotron speech / streaming voice**, release-dependent | streaming cache-aware FastConformer, deadline-adjusted WER/latency; duplex voice variants couple ASR/TTS | per-stream persistent acoustic cache, variable endpointing/latency modes, 80ms–1s output policies, barge-in/echo/codec clock | vendor speech updates |
| **Qwen3-Omni** audio/video understanding | Audio/video processor, MoE **Thinker** with optional **Talker**; native real-time interactive multimodal input | dual-stage model residency, stable A/V timebases, streaming audio/video input partials, Thinker→Talker handoff, optional text-only Talker eviction | [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) |

Do not collapse all ASR into Whisper or all encoder scoring into next-token generation. In particular CTC, RNN-T and TDT have different input/output/state and blank/duration decoding semantics. Distinguish **partial** ASR hypotheses that might be revised from committed transcript segments; user-visible streaming must tag revision/finality and sample-accurate time spans.

### TTS, codec-token generation, real-time speech and voice cloning

| Family | Execution family | Required model/codec constraints | Primary source |
|---|---|---|---|
| **Kokoro-82M**, open weights, checkpoint terms | style-based **G2P/phonemes → text conditioning/embedding → text prosody/duration/F0 → acoustic generator/ISTFT vocoder**, no token-AR LM required | registered G2P and licenses, phoneme normalization, duration/alignment, prosody voice embedding, FFT/ISTFT overlap-add, exact PCM sample clock | official checkpoint family |
| **Dual-AR speech model**, open code/weights under a source research license | **Dual autoregressive** speech codec: ~21Hz slow temporal primary codebook plus 9 fast residual codebooks; waveform decoder; multi-speaker/prosody | independent AR clocks (time vs RVQ depth), codebook tensor/event shape, codec decoder, voice references, fine-grained control, source sample rate and streaming finality | — |
| **Qwen3-TTS**, open weights under model terms | Dual-track real-time LM; published 12Hz/25Hz speech tokenizers, low-latency multicodebook generation and/or block-wise diffusion reconstruction | bidirectional streaming text/audio, phoneme/voice descriptions, multi-level codec state, packet timebase, separate speech tokenizer and waveform decoder, delay/rollback | [Qwen3-TTS official](https://github.com/QwenLM/Qwen3-TTS), [technical paper](https://arxiv.org/abs/2601.15621) |
| **VibeVoice TTS / Realtime**, Microsoft repo caveat | Continuous acoustic/semantic speech tokens (~7.5Hz) with language context LM and diffusion acoustic head; streaming realtime 0.5B | codebook/continuous representation, diffusion step per acoustic chunk, multi-speaker, long-form ~90min; **original VibeVoice-TTS code was removed from repo in 2025**, inspect variant availability before claiming runnable | [VibeVoice official and availability notes](https://github.com/microsoft/VibeVoice) |
| **A hosted TTS service**, proprietary API | Hosted streaming TTS and separate text-to-dialogue / multi-speaker WebSocket behavior, model dependent | authorized external TaskServiceAdapter, voice ID registry, SSML/turn boundary, stream chunk flush/cancel, credentials and content consent, egress/tracking/cost; **not** a native weight-execution model | Realtime TTS, TTS vs TTD |
| **Qwen3-Omni Thinker/Talker**, open models | Native real-time audio/video understanding and audio output; MoE-based Thinker and multi-codebook Talker | double model stage, optional 10GB-class Talker memory, text+audio synchronized output, dynamic turn and codecs, full duplex/interrupt handling | [Qwen3-Omni official](https://github.com/QwenLM/Qwen3-Omni) |

**Streaming availability ≠ all runtimes implement streaming.** Qwen3-TTS project maintainers explicitly distinguished model architecture support from the early demo package's mainly offline path: [streaming implementation issue](https://github.com/QwenLM/Qwen3-TTS/issues/10). A source model capable of block-wise audio generation does not guarantee Zerogravity can emit early playable audio until the entire upstream codec/decoder path and state checkpoint have been implemented.

### Video and synchronized audiovisual generation

| Model | Source architecture/availability | Native runtime requirements | Primary source |
|---|---|---|---|
| **Wan2.1** 1.3B/14B and I2V, open weights | Spatiotemporal DiT, causal 3D Wan-VAE, text-guided diffusion, offload/sequence parallel examples | temporal latent shape, 3D VAE incremental/tiling, spatial/temporal position axes, scheduler/CFG/seed, source text conditioner | [Wan2.1 official](https://github.com/Wan-Video/Wan2.1) |
| **Wan2.2** T2V/I2V/TI2V, open weights | Diffusion **denoising-phase MoE**: high-noise and low-noise specialized transformer blocks or models selected by timestep; dense TI2V option | source noise/timestep guard for expert switch, separate expert residency and transition events, correct scheduler, 3D VAE, video output timestamp | [Wan2.2 official](https://github.com/Wan-Video/Wan2.2), [model card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| **Dual-stream video DiT 1.5**, open code/weights | 8.3B DiT, causal 3D VAE with 16x spatial and 4x temporal compression, Selective/Sliding Tile Attention, super-resolution stage | registered **attention mask semantics**, spatial+temporal tiles, 3D VAE state, conditioned SR, possible FP8 GEMM, overlapping model residency/offload | — |
| **Original dual-stream video DiT**, open weights | Dual-stream/single-stream multimodal DiT, causal 3D VAE, MLLM text encoder | accurate dual→single transition, processor, large conditioner, latent VAE, source model-specific attention | — |
| **Asymmetric-attention video DiT**, Apache 2.0 open weights | 10B Asymmetric Diffusion Transformer (AsymmDiT), 3D video AsymmVAE, unequal text/video hidden size and non-square projections | unequal tensor dimensions and joint attention, 3D positions, causal 3D VAE, context/tensor parallel and tiling | — |
| **Coupled audio/video model**, open model with license caveats; model-specific variants | Coupled **audio+video dual-stream DiT**, bidirectional A/V cross attention, jointly conditioned timestep, media-aware CFG | two clock domains, audio+video latents, synchronized cross-attention/solver, decoder output watermark and coupled resource residency | — |
| **A hosted video service 3.1**, proprietary Google hosted | Text/image-guided audiovisual video generation via asynchronous hosted API; supports sound, reference frames/image | external provider job lifecycle/authorized media upload, quota, billing, result retrieval, optional C2PA; cannot supply native semantic regions without weights | A hosted video service 3.1 |
| **A hosted creative-media image/video service**, proprietary hosted | Hosted creative media API, synchronous/asynchronous image and video job families | external TaskServiceAdapter, OAuth/client credentials, async asset IDs and retries, consent, rights and content credentials | Hosted creative-media API, asynchronous API |
| **A hosted joint audio-video service 1.5 Pro**, proprietary ByteDance hosted | Native joint audio-video two-branch MMDiT-style generator with cross-modal conditioning | remote job task adapter; when a future open comparable model exists, coupled A/V clocks + attention and diffusion model assembly | ByteDance announcement, technical abstract |
| **A hosted video service 3.0 / Omni**, proprietary hosted video-service API | Hosted audio/video generation, storyboarding, image/video references and async jobs | remote typed capability adapter; do not infer internal neural architecture from an API model brand | hosted video service developer API |

**Important accuracy note:** The cited coupled-A/V paper describes synchronized dual streams; its arXiv abstract and HTML differ in the exact substream parameter breakdown in source versions. Do not hardcode those counts as trusted weight geometry; load actual weight model and source manifest. Similarly Wan2.2's timestep MoE is NOT an ordinary per-token MoE route. The asymmetric-attention model's large token space needs context/spatial-temporal tiling and strong memory admission; “GPU fit” is not equivalent to true device throughput.

### Typed decision engines and model selection

| Model | Nature | How it maps to Zerogravity | Primary source |
|---|---|---|---|
| **Proprietary decision service** | Proprietary decision-only service for yes/no/choice/score under an explicit allowed-option schema and calibrated decision semantics | external TaskServiceAdapter; score schema, no generated-text parser, calibration/threshold metadata, independent privacy/policy | [decision study](https://arxiv.org/html/2609.28940v1) |
| **Open multilingual typed-decision model**, open code/weights, checkpoint-dependent license | Non-AR ModernBERT-large 421M / mmBERT-base 322M encoder and task scoring heads; choice/score/no-ul (yes/no) schema, router chooses per request | locally executable TaskDriver ScoreFiniteSchema, normal encoder residency and independent head, batched state/question logits | — |
| **Open joint-schema decision model**, open weights | 27B Qwen3.8-27B or 9B Qwen3.5-9B **multimodal Qwen backbone + joint-schema decision transformer head**; one logit per allowed option, multiple questions in one forward pass | reusable Qwen and vision model semantic components plus registered typed question scoring head, calibrated score output; not token-AR JSON | model card and source head (references removed) |

A decision model must carry raw-logit, probability-calibration, option enumeration and schema ownership distinctly. A 27B multimodal backbone does not automatically make its decision head interchangeable with a 421M encoder classifier. The decision router may choose based on observed accuracy, cost and task latency, but *model type* does not grant unsupported calibration claims.

### Ternary-compressed and source-native hybrid text models

**A ternary-compressed 27B hybrid model** (source family Qwen3.8 27B) is a direct test of the separation between **model topology** and **tensor representation**. It has about 75% linear attention and 25% full attention, with vision as an optional component, like the underlying hybrid model—but the weights are packed in native ternary PTQ1_0 or PQ2_0 encodings with F16 group scales, and the runtime applies a matching blockwise Hadamard transform to activations.

The source model card explicitly specifies:

- PTQ1_0 dense trit packing versus PQ2_0 2-bit trit slots; group size 128 and original source scales.
- 1024-value block Hadamard rotation and fixed sign scheme; matching runtime transformation is **required**.
- Custom ternary kernels for GPU and CPU rather than decompressing the entire model to high-precision tensors.
- Optional separately resident Q8_0 vision mmproj, and model/version compatibility checks.

Primary sources: model card and release notes (references removed).

**Design consequence:** a new quantization codec/activation-transform provider can support the ternary model using the same Qwen3.8 SemanticProgram. Model-dispatch code that hardcodes GGUF Q4_0/Q6_K into the model topology is an obstacle. Distinguish a source-compatible compressed artifact from an ordinary low-bit quantized derivative that silently changes quality; checkpoint identity and activation-basis metadata are mandatory.

Other hybrid/recurrent families (Gated DeltaNet, Mamba, SSM, partial full attention, grouped MoE) are operators/state domains, not new independent inference servers. Every new weight codec and attention variant is separately source-qualified; no generic “ternary fallback to llama.cpp” route.

## Cross-model composition examples

### Live voice assistant

Authenticated live PCM ingress (48kHz) → VAD/resampler and acoustic windows → streaming ASR encoder/cache (Parakeet/Canary/Whisper) or native Thinker → optional typed decision router (proprietary or open decision models) → text/richer reasoning model → streaming TTS (Kokoro/dual-AR/Qwen3-TTS/Talker) → completed codec frames/PCM with backpressure and low-latency playback.

The application-level *orchestration* connecting distinct models is above TaskRequest; each source model has its own exact identity/admission, and its own source math program. If a single **trained compound model** like Qwen3-Omni bundles Thinker/Talker, its internal assembly belongs to the verified ModelDefinition, not the external agent scheduler.

Barge-in must cancel old not-yet-played audio and fence in-flight GPU work without duplicating the committed session state root. Audio clock conversion and streaming revision are first-class, not inferred from text length.

### Image studio

Typed prompt and optional multi-image reference → condition encoders (VLM, CLIP/T5 or Mistral depending on source) → seeded latent/flow source step with registered CFG/noise policy → VAE decode/tiling → image color metadata and requested format. If prompt rewrite uses a separate LLM, it is an explicitly visible optional application composition, not an invisible change of user prompt.

### Video with soundtrack

Wan/dual-stream-video/hosted joint-AV-like source conditioner → temporal latent/DiT schedule (and optionally timestep MoE experts) → causal 3D VAE video frames. With an intrinsically joint A/V model such as the coupled audio/video architecture, audio/video cross-attention and shared timestep outputs are one coupled source assembly, and the StateRoot sync frontier must commit audio samples and video frames consistently. For a staged separate TTS+video workflow, an explicit synchronization adapter and separate quality metrics are needed; do not imply intrinsic lip-sync is guaranteed.

### Decision-heavy application

Typed request with 20 allowed questions → low-latency open decision-model logits → exact calibrated option result. Based on policy, low confidence can trigger a separate Qwen/Gemma reasoning TaskRequest; returned typed decision state and explanation have different source provenance. No arbitrary script/tool execution is embedded in tensor kernels.

## Proposed support tiers, not invented implementation

| Tier | Meaning | Example today |
|---|---|---|
| Recognized | artifact or API name recognized, architecture/weights may be untrusted | catalog research only for all listed media models |
| Semantics registered | exact source operators, codec/time, state domains have reference implementation | Qwen3.5 text only |
| CPU verified | trained checkpoint executes, exact source/mode numeric evidence | Qwen3.5 0.8B/2B only |
| Device region verified | specific GPU kernel on physical device with source-model numeric evidence | historical Q4_0 F32 projection probe, NOT complete GPU model |
| Hybrid execution verified | full model runs with at least one real physical GPU region and CPU fallback | P07-A provider compiles and has CPU host oracle; physical GPU not yet accessible |
| Full native accelerated model | complete model operates on GPU with timing, state and accepted outputs | none yet |
| Stream/duplex qualified | latency, bounded buffers, cancellation, synchronization, quality | none yet |
| External API supported | explicitly authorized, typed remote adapter with credentials/egress and lifecycle | none yet |

Do NOT equate “open-source code,” “open-weight checkpoint,” “commercially unrestricted license” or “hosted API” when choosing runtime support. License and availability can change per model release.

## Implementation roadmap — staged extensions, not a new engine

**P07 remains current:** finish physical CubeCL/CUDA/ROCm/Vulkan source-Q4_0×Q8_0 partial GPU-model test; add Q4_1/Q6_K/F32, persistent weight residency, buffer leases and full-model GPU step planning. The new multimodal design is a *parallel specification effort*, not a reason to reset this implementation.

**MM0 — source semantics & fake tests:** typed value/media clocks, component graph/ports, task frontier coordinates, seed/solver schema, known codec transforms and unsupported-provider mode receipts. Build one fake non-token encoder/decision, one fake denoiser, one streaming audio timeline under the same StateRoot coordinator. Verify 12 acceptance drills in the architecture contract.

**MM1 — practical open audio:** Whisper ASR then small streaming Parakeet/CTC/TDT and/or Kokoro/ISTFT TTS for a non-AR acoustic model. End-to-end waveform + transcript + timestamps against author reference.

**MM2 — image:** a manageable Qwen-Image or rectified-flow-Klein/SD3.5 pipeline using verified conditioner → DiT → scheduler → VAE, actual image output, end-to-end source comparison and peak scratch budgeting. Preserve sampler choices and license.

**MM3 — difficult speech:** dual-AR RVQ, Qwen3-TTS dual-track codec, VibeVoice continuous-token acoustic diffusion, Qwen3-Omni live Thinker/Talker and duplex barge-in. Actual time-to-first-audio, codebook decode and sample clock quality required.

**MM4 — video:** Wan2.1/2.2 (3D VAE, denoising phase MoE), dual-stream video DiT 1.5 (SSTA), asymmetric-attention DiT (AsymmDiT), coupled A/V (joint synchronized). Model-class task drivers remain reusable; introduce only source codecs/operators/conditioners/solver variants.

**MM5 — ternary/decision:** ternary PTQ1_0/PQ2_0 transformed source GEMV, open typed-decision option logits plus multimodal state, transparent small-model routing. Closed-model APIs only through authorized TaskServiceAdapter independent of the CPU/GPU model runtime.

## Architectural acceptance decision

**Keep the Zerogravity v4 execution spine and clarify six contracts; do not rewrite the core.** If the proposed value/clock/codec/assembly/TaskDriver extensions cannot support a new open source model without writing another model server, the extension design failed. The strongest litmus models are:

1. **Dual-AR codec model** — two simultaneous AR axes and codec/waveform staging.
2. **Coupled A/V model** — synchronized dual audio/video streams and source timestep semantics.
3. **Typed decision head** — non-token structured decision head on a multimodal backbone.
4. **Ternary-compressed model** — same Qwen model topology with incompatible packed ternary weight/activation-basis semantics.
5. **Autoregressive image MoE** — advertised autoregressive multimodal MoE with a disputed source-generation path; exact checkpoint/operator tracing is required before concluding AR-only or diffusion-only.
6. **Wan2.2** — source denoising timestep expert switch, not token router.

Passing these architectural mock drills would demonstrate composability across fundamentally different execution types; implementing named weights and physical backends requires separate real validation and licensing.

````

## Source: .agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md

Original UTF-8 bytes: 24952; SHA256 `cdbdd47321034fdab00031c21541c7f49dfddbe89e52d8c826fb7b0dbf471953`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity Multimodal v5.1 — Complete Implementation Roadmap, Dependency Gates and Acceptance Matrix

**15 October 2026 — MODEL COMPONENT STORAGE SUPPORT:** The [virtual tensor + SSD first-class contract](../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) applies to immutable model components, video temporal VAE and frozen private session state without changing any model algorithm/codec/clock. Audio/AV high-performance realtime may require full resident components and MUST reject SSD placement if it violates measured QoS. SSD stage admission and actual completion are independent of playable media finality and TaskCoordinator StateRoot. Companion [SSD-P0..P5 planned acceptance](SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) has SSD-01..32; all planned, not executed, and do not supersede MM01–24.


**15 Oct additive extreme-text/quant implementation precision:** The source-bound [QTIP + MiMo/DFlash + gfx1151/UMA integration roadmap](EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) supplies **50 distinct planned QTIP/HYB/GFX acceptance fixtures** under existing P03–P22 without altering media MM-01..24 or any model implementation priority. Large video/audio/decision TaskDrivers remain independent; source artifact/codec+TensorEncoding/region kernel support remains orthogonal to each task. The specialist source atlas is source evidence, NOT hardware/quality proof for Zerogravity; vendor-specific memory page-cache and row grouping are candidate kernel/cost properties, not generic multimodal task semantics.


Technical implementation and acceptance reference. Live work and completed results
belong to [the multimodal TODO](../plans/multimodal/TODO.md), under
[TODO-SPEC](../plans/TODO-SPEC.md). The requirements below do not grant permission
to acquire checkpoints, invoke a provider or change devices. Real media/decision
support requires its original source, task/state and eligible-device gates.

## 0. Universal execution truth and success definition

The user requires broad, high-performance model/hardware support with strong specialization, simple UX, first-class multimodal/realtime serving, a built-in router and maintenance-light Rust-native architecture. A model is not "supported" by importing its name or format; support must be labeled at precise capability tier.

Support tiers (each scoped by model artifact, component assembly, task, backend, numerical mode, input shape and stream transport):
- **T0 Recognized:** model name/artifact format recognized, no source semantics qualification.
- **T1 Semantically verified:** exact source tensors/processor/ops/driver and reference recipes identified.
- **T2 CPU reference running:** trained checkpoint/source fixture and exact CPU task correctness including state effects.
- **T3 Partial accelerator:** selected closed semantic region physically executed with source-matched receipt; NOT whole model.
- **T4 Whole-model accelerator:** full task semantics/state root, original complete checkpoint, physical backend under exact numerical quality contract.
- **T5 Streaming/playable:** valid output media chunks and clocks/codec finality, state/cancel/teardown verified.
- **T6 Soft-realtime measured:** declared latency/jitter/memory/throughput under pinned hardware, load and transport, with admitted QoS.
- **TX External hosted:** provider endpoint and authorized lifecycle validated; never reported as native T2–T6 merely because API responds.

The catalog and CLI/SDK/API must show exact tier and missing gates, not a vague "multimodal supported" flag.

## 1. Scope and source authority

Preserve the user-directed breadth: standalone and native duplex speech, image
creation/editing, temporal video and coupled AV, finite decisions, unusual source
codecs, composable model assemblies, and high-performance serving in one Rust
engine. A direct neural duplex model and an ASR→language→TTS composition remain
distinct tasks. Hosted services expose authorized observable APIs, not invented
local weights or proprietary internals. No family name implies streaming,
commercial licensing, source-equivalent math or a runnable checkpoint.

Use the [retained source atlas](MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md),
the per-family dossiers below, the governing execution/assembly contracts and
current plan TODOs. Do not revive old baseline statuses or change the authorized
CPU-first model/acquisition priorities because a media design is ready. User
requirements remain distinct from assistant proposals and unresolved policy.

## 2. Design readiness gates MM-D0..MM-D5

### MM-D0 — source and design consistency
Deliver:
- v5.1 unified execution, model assembly/codec and real-time serving specifications.
- research dossiers for ASR/translation, TTS/duplex, image, video/AV, decision/ternary, protocol/provider prior art.
- U40/U41 explicit user decision provenance, research/design recommendation register and open options.
- independent reviewer reconciliation with remaining conflicts and falsification criteria.
- linked original-source references, unresolved decisions and modality-specific adverse fixtures in their retained technical owners; no duplicate handoff or checksum inventory.

Gate: contracts agree on one state/resource authority, algorithms/clocks and exact capability scope; references resolve and no model support is invented. A design consistency review does not qualify runtime behavior. Current TODOs record implementation and actual tests.

### MM-D1 — typed media and clocks (later CODE only if authorized)
- ValueDescriptor with exact sample/frame/codebook/latent/decision types and registered parser limits.
- ClockDomain, RationalRate, checked u128/sample/PTS/DTS conversions and residual resampler delay.
- independent per-lane capture/codec/solver clocks; strict invalid output finality.
- no new persistent IR; adapt existing SemanticProgram ValueType contracts.

Gate:
- all MM-01 and timebase cases; 44.1k/48k PCM to 16k/24k audio plus 24000/1001 and 30000/1001 video;
- no drift over millions of ticks, invalid packet format/clock fails, bounded buffers and overflow refuses;
- no dependence on GPU vendor runtime for source-free tests.

### MM-D2 — verified ModelAssembly and source codec registry
- source artifact + processor + per-component assembly manifest, typed ports, verified transforms, license/version/hashes.
- TensorEncoding independent of model topology, accumulation/activation transform, exact padding/shape and codec identities.
- source model can have separate conditioner/VAEs/vocoder or multiple text/audio heads without another EngineCore.

Gate:
- MM-02 typed components and port incompatibility tests;
- intentionally mismatched rectified-flow [dev]/[klein] conditioner fails;
- ternary codec without Hadamard fails;
- no arbitrary checkpoint Python execution;
- model capture variants and license ownership shown in receipts;
- one component resident across two sessions without mutable state leakage.

### MM-D3 — task algorithms and transactional multi-domain root
- trusted drivers for decoder token, encoder pooling, CTC/RNNT/TDT, codec depth AR, non-AR TTS, flow, video, schema choices.
- legal repeat/parallel-known/codec/mux stage control with source-defined iteration and state effects.
- TaskFrontier multi-axis and CommitGroup atomic publication.

Gate:
- MM-03 flow and MM-04 RVQ/ASR addition, MM-07 AV and MM-08 finite schema; wrong solver/codec/choice fails;
- interrupted device events never publish stale StateRoot;
- no arbitrary external action inside TaskDriver.

### MM-D4 — RealtimeSession and MediaEvent/PresentationLedger
- live input/output lanes, separate neural/transport/turn epochs, committed/delivered/acked/played watermarks;
- VAD/barge-in, revisioned transcript, output codec chunks and bounded buffers; per-session cancellation scope.
- make recording/ack knowledge honest when client isn't providing accurate playback feedback.

Gate:
- MM-06 duplex barge-in plus MM-13..MM-17 and MM-19; no cross-session GPU poisoning, no underflow/latency lie, no output media emitted before complete codec;
- sequence, generation, timer and clock assertions with deterministic mock Waker.

### MM-D5 — canonical serving and protocol conformance
- EngineCore reused by CLI, SDK, HTTP, SSE, WS, WebRTC, optional SIP gateway;
- per-model/per-transport capability profiles and refusal semantics;
- external TaskServiceAdapter asynchronous jobs with auth, privacy/egress, rate/cost/idempotency and result receipts.

Gate:
- MM-18..MM-24, official tested SDK API subset, malformed request/codec/media tests;
- accepted remote provider job timeout remains Unknown and reconciles without automatic double charge;
- no duplicate router/TaskCoordinator/DeviceBackend, no per-frame global hot-path snapshot or JSON-only binary transport.

## 3. Source-real implementation slices MM-I0..MM-I7 (proposed, sequencing requires approval)

### MM-I0 — small non-token model and codec mock under same core
Implement one encoder/finite direct decision head plus one iterative latent solver mock using source-free deterministic fixture. Reuse TaskCoordinator/ResourceArbiter/RegionProvider. Validate no change to actual Qwen CPU/GPU results. This is the first architecture *change-radius* proof, not a media product.

### MM-I1 — original ASR checkpoint, CPU first
Select one sufficiently small legally available Whisper or Parakeet source checkpoint. Register exact feature extractor/processor, model semantic op, ASR TaskDriver and transcript clock. CPU source golden tests, then GPU source-equivalent attention/conv/frontend on one physical backend. Publish accurately whether model supports offline only or real-time streaming. No RTT claims until measured.

### MM-I2 — Kokoro non-AR TTS, CPU first
G2P/phoneme, duration/F0/prosody, source vocoder. Source-defined PCM sample rate, complete codec chunk and real WAV output. This deliberately falsifies designs hardcoded around decoder LM. Validate source accuracy and speaker privacy and measure first audible sound. Reuse warmed source weights across sessions only under source/version match.

### MM-I3 — dual-AR/Qwen TTS depth AR and full-duplex mock
- First source-free RVQ nested time/depth and two-lane audio consumer mock; then licensed dual-AR or Qwen3-TTS original checkpoint as permitted.
- Prove source codebook, chunk/overlap, voice state and timebase.
- Add native full-duplex neural-codec or Qwen-Omni source model if feasible, with actual loaded input/output streams and interrupt handling; don't force text-only cascade.
- Hosted TTS adapter can proceed independently behind consent without native weights.

### MM-I4 — image flow generator and editing
Pick a small source-audited flow/DiT model (rectified-flow Klein if exact weights/license permit) and exact conditioner/VAE. CPU reference fixtures and actual GPU source region/whole-image validation, seeded solver/guidance, memory tiling. Then SD3.5 and Qwen-Image as variants, not one fake image backend. The autoregressive image MoE AR/hybrid source recipe only after careful generation code audit. A hosted creative-media service/GPT-Image adapters independent of local support.

### MM-I5 — video temporal / Wan expert switching
3D VAE and temporal latent reference, Wan2.1 small proof if feasible, then Wan2.2 high/low noise expert switching with source threshold. Source-correct model/video clocks, VAE overlap, tile/sparse attention and real end-to-end output. Add dual-stream video DiT 1.5 / asymmetric-attention attention/shape variants as verified components.

### MM-I6 — joint audio/video and frontier job workflows
Coupled A/V source stages, independent audio/video latents and synchronizing one commit group. Actual source/AV codec output and PTS sync evidence on feasible GPU(s), no post-hoc TTS masquerade. Separate hosted generation-service asynchronous TaskServiceAdapters, source/license/rights/cost consent, unknown timeout reconcile.

### MM-I7 — Ternary compression, specialist text, typed decisions at scale
- Ternary PTQ1/PQ2 + Hadamard transform original source golden reference, then CPU bitpacked/physical GPU legal specialization; no second Qwen engine.
- One open source-typed decision checkpoint; a shared Qwen backbone and trained schema head, source logits and versioned calibration.
- The hosted decision API adapter with consent and provenance, direct finite typed results.
- Additional low-bit/MTP/linear-attention families adopt same independent codecs, transforms and StateDomains.

**These later slices do not override user's already approved Qwen3.5→Gemma→Qwen CPU/GPU roadmap.** They are an extension program after or interleaved only when explicitly authorized and do not interrupt existing phase safety work.

## 4. Complete MM-01..MM-12 inherited acceptance matrix

| ID | Fixture and minimum observable assertion |
|---|---|
| MM-01 | exact rational 44.1k PCM↔29.97FPS clocks, correct residual over long durations |
| MM-02 | verified conditioner→generator→decoder ports; wrong processor/shape/format refused |
| MM-03 | 20-step dummy flow with seed, RNG/guidance and per-step atomic StateRoot |
| MM-04 | Dual-AR slow temporal ~21Hz × ten RVQ codebooks; only complete frame can decode |
| MM-05 | Whisper encoder/AR vs Parakeet CTC/RNNT/TDT blank/duration/timestamps |
| MM-06 | full-duplex input/out + barge-in/epoch and device lease/mock completion fencing |
| MM-07 | Coupled-style synchronized two-track A/V publication and failure |
| MM-08 | multiple typed finite schema decisions atomically scored, nonfinite/refusal |
| MM-09 | Ternary PTQ1/PQ2+Hadamard exact codec mock and source shape guard |
| MM-10 | one TaskRequest supports fake CPU/GPU/AOT/hosted with distinct truth tiers |
| MM-11 | long streams bounded, partial playback/cancel/rollback and event/resource safety |
| MM-12 | recognized vs CPU-only vs GPU-region vs full GPU vs realtime vs hosted capability status |

Existing synthetic foundations are in `crates/zerogravity/tests/multimodal_mm0.rs`. The matrix is a requirement inventory, not a new test result. Check current code and the multimodal TODO before extending it; synthetic clocks or dummy flow do not qualify trained media.

## 5. MM-13..MM-24 new realtime/API acceptance matrix (all not implemented)

| ID | Deterministic scenario | Required assertion |
|---|---|---|
| MM-13 | Native duplex audio while response streaming; VAD barge-in | old output stops, input state continues, unrelated GPU sessions unaffected |
| MM-14 | Playback ACK from prior turn arrives after turn epoch changed | ignored; never rewrites new played watermark |
| MM-15 | 48k/44.1k PCM and 30000/1001 video PTS over hour | exact residual/rounding, no invented integer FPS |
| MM-16 | Transport acknowledges packet while GPU command still outstanding | physical BufferLease remains until verified Ready |
| MM-17 | Slow WS/SSE media client with tiny credits | bounded queue/memory, backpressure/admission error not OOM |
| MM-18 | 10 live speech sessions plus 5 large video solver jobs | fair physical admission and honest missed-SLO, no silent quant/solver changes |
| MM-19 | WS connection dies and reconnects with duplicate cursor/event | bounded idempotent replay of committed media, source auth and epoch check |
| MM-20 | Remote hosted-video-service-like POST accepted but response times out | status Unknown; reconcile ID, no automatic duplicate billable create |
| MM-21 | Model capability supports TTS but not realtime interruption/voice clone | unsupported transport/task fields explicitly rejected |
| MM-22 | Every audio packet goes through prepared path | no global catalog/router/arbiter snapshot allocation per frame |
| MM-23 | Coupled A/V joint solver step with audio lagging behind video | no video/audio final group until declared ready/compatible |
| MM-24 | malformed media path, hostile dimensions/codec and unauthorized voice egress | rejected before GPU allocation or outbound provider call |

Mock tests are **precondition** for source-real media implementation; they are not evidence of source model quality or commercial product readiness.

## 6. Model-specific detailed acceptance suites

Research dossiers define named tests:
- MM-ASR-01..12 in ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md.
- MM-TTS-01..12 in TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md.
- MM-IMG-01..12 in IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md.
- MM-VID-01..15 in VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md.
- MM-DEC-01..06 and MM-BON-01..08 in DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md.
- MM-SRV-01..12 in SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md.

These roughly 77 more focused fixtures supplement MM01–24, not replace them. An agent must import *all* relevant suites for a source model, and explicitly classify source-free mock, original source trained CPU, physical GPU, hosted API, or soft-realtime measurement evidence.

## 7. Source binding and review obligations

For each actual model/variant, identify original config/tensors, tokenizer,
processor, codec, scheduler/solver and licenses; keep code, weights, voice/G2P
assets and provider rights separate. Model/content identities remain required.
Source code is reference input, not executable checkpoint Python. A source matrix
must say which processor/codecs/checkpoint APIs are still unverified.

Keep the distinct gates from the retired review briefs:

| Slice | Required source trace and negative cases |
|---|---|
| ASR/translation | Whisper window/encoder/AR; CTC blank-collapse; RNN-T predictor/joiner and TDT duration advances. Check frame/subsampling clocks, streaming checkpoint lookahead, language/task headers and provisional diarization corrections. Offline FastConformer is not automatically streaming; encoder classification is not CTC. |
| TTS/duplex | Non-AR phoneme/duration/prosody/vocoder, nested time/depth RVQ and truly simultaneous codec input/output traces. Bind G2P/voice assets separately; incomplete codebooks, stale overlap or changed voice/codec generation cannot become audible PCM. |
| Image/editing | Check exact conditioner, latent packing, VAE/crop/color/mask, seed/guidance and source solver. Dev/distilled conditioners and schedules differ. Audit the selected AR/hybrid image generation entrypoint instead of choosing AR or diffusion from its name. |
| Video/coupled AV | Check temporal 3D-VAE stride/overlap, rectangular attention, sparse tile selection and timestep expert transitions. Audio/video cross-attention and independent clocks must meet their joint commit group; unrelated TTS over video is a compound task. |
| Decisions/low-bit | Check direct option/joint-schema head, option order, null/refusal and calibrated versus raw scores; text JSON is not equivalent. PTQ1_0/PQ2_0 group scales and the source1024 Hadamard/sign transform must match; ordinary Q2 is not a valid substitute. |
| Serving/providers | Check model/task/transport field conformance, prepared binary packet path, bounded credits, reconnect epochs and actual playback uncertainty. Ambiguous accepted remote jobs stay Unknown until reconciled; no duplicate billable create or implicit egress. |

Each implementation review identifies private state and its sole publisher,
source/numeric reference, clocks/finality, cancellation/readback/lease retention,
missing support tier and extension change radius. Adding a model must not require
a second router, allocator, runtime, arbitrary workflow DSL or public plugin ABI
per operator. Record actual command/result and failures in the plan/run TODO;
ordinary evidence-file hashes, archive construction and generated status copies
are not acceptance requirements. Physical or provider tests remain required for
their own claims; documentation or a compiled kernel is insufficient.

## 8. Perf/maintainability gates

Test both speed and extension cost:
- Throughput: GPU utilization, inference tokens/s, image steps/s, video frames/s, TTS audio seconds/s with exact model and output quality, not meaningless mixed-unit throughput.
- Tail: TTFT/TTFA, ASR lag, user barge-in silence delay, audio packet dropout, video first playable frame, P50/P95/P99 by load/transport and thermal regime.
- Memory: weights+latent state+codec+buffers+peak driver allocation/fragmentation, UMA/GTT vs resident physical bytes.
- Multi-session: start/stop/stale session isolation; source weights shared where exactly same artifact; no shared sampler/SSM/acoustic state corruption.
- Routing: exact model only by default, no hidden downgrade; per-replica placement sticky state and safe migration.
- Change-radius: adding one new codec/head/task provider should not edit the central Router/TaskCoordinator/serving socket code; document any exception.
- Build: useful CPU-only local Rust binary without extra CUDA/ROCm/Python prerequisites; compiler/vendor features optional.
- Simplicity: all new interfaces are expressible in bounded model+task+physical and streaming contracts; no plugin trait per sample and no unnecessary service microprocess.

## 9. Dependencies and stop rules

| Work | Depends on | Stop if |
|---|---|---|
| Universal typed media values | source ValueDescriptor/Timebase proposal | raw float time can't be made exact, risk of ambiguous clocks |
| ModelAssembly | artifact hash/config processors/licensing, semantic adapters | source component cannot be verified or license disallows use |
| TaskDriver | source-stable model algorithm and reference oracle | task fallback changes answer meaning |
| GPU specialization | real hardware/support, physical buffer/effect leases, numeric source proof | kernel produces unapproved distribution drift |
| Realtime serving | typed committed MediaEvent + bounded presentation state | client/transport can commit neural model state |
| WebRTC/SIP | media auth/transport/clock/codec/ACK contract | media transport exposes credentials or unbounded buffers |
| Hosted providers | explicit allowed egress, cost, credential, idempotency | source model is opaque but claimed local, or provider job replay ambiguous |
| Full AV generator | video/audio decoder and joint latent source graph | source cross-attention ignored or timestamps don't align |
| Ternary compressed | source codec+Hadamard validated | activation transform absent |
| Whole native GPU model | all source semantics, recurrent/KV/multimodal and state actually on GPU | merely one Q4 region or fake Ready source-only evidence |

**Release gate:** a feature cannot move from proposal to "supported" without exact source math, output and state/clock verification and device/API evidence. The user has not approved any numerical latency SLA, provider account linking or proprietary weight substitution.

## 10. Implementation ownership and pending policy

Assign narrow source/driver/codec/provider work to disjoint owners; the coordinator
integrates reviewed changes and test results. Use the current multimodal TODO and
TODO-SPEC, not copied agent briefs or a static aggregate. Named research lanes are
not evidence that independent reviewers ran, and no future test is passed by
writing its specification.

Unanswered choices remain explicit: first licensed feasible ASR/TTS/image/video
checkpoints; source-bit versus a separately approved media-quality policy; exact
AR-image entrypoint; temporal VAE/memory-fit limits; physical headroom/quiescence;
measured mouth-to-ear/first-output/jitter objectives; and packaging of optional
transports or future component ABI. None follows from model marketing or user
examples of API names. The detailed live-session questions are in
[the realtime policy section](../docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md#18-open-policy-decisions-before-live-serving).

## 11. Technical references

- [Universal execution](../docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md), [verified assemblies](../docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md), [realtime QoS](../docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md).
- [ASR](multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md), [TTS/duplex](multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md), [image](multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md), [video/AV](multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md).
- [Decisions/ternary](multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md), [serving prior art](multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md), [adversarial reconciliation](multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md).

The original authors' source/license references in those owners remain intact.
These are technical research and future gates, not claims of native trained media
execution or completed API/realtime acceptance.

````

## Source: .agents/research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md

Original UTF-8 bytes: 19547; SHA256 `8d9a6fd6b7d4f7aa8498eae4cf25aa06a424b7515d3399ee145f446de2abbf7d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity v5.1 Multimodal Decisions, User Requirements, Design Proposals and Open Questions

**15 October 2026 · user intent and model-research provenance addendum.**  
This retained ledger distinguishes historical user decisions U-040/U-041,
research proposals and unresolved policy. The [decision register](../plans/DECISION_REGISTER.md)
retains the wider project decisions. Current work uses [the plan TODOs](../plans/TODO.md)
and [TODO-SPEC](../plans/TODO-SPEC.md); historical handoff/manifest instructions
below are superseded tracking methods, not current duties. Git retains history.

## 0. How to interpret the decision ledger

- **USER**: directly stated or explicitly reaffirmed by user in this conversation; preserve unless later user supersedes.
- **PROPOSAL**: architecture recommendation from research/assistant; recorded to enable independent review, NOT an extra user approval.
- **OPEN**: requiring explicit user policy, exact source audit or performance/hardware evidence before finalizing.
- **ASSUMPTION**: reversible interpretation, documented so implementation agents don't invent authorization.
- **HISTORICAL**: prior source fact/implementation state at a dated baseline; not current software support assertion.
- **SUPERSEDED**: an earlier user decision whose exact scope was later revised, preserving previous rationale.

The user explicitly asked for deeply researched, composable architecture and comprehensive documents, not a blanket waiver of source licensing/secret permissions or automatic native execution of proprietary models.

## 1. New explicit user instructions U-040 and U-041

### U-040 | USER | 15 October 2026 — universal audio/image/video/decision/ternary/realtime design research, strictly no code

The user requested another **architecture pass in design-only mode, no code change at all**, including:
- standalone voice models, speech-to-text, text-to-speech, low-latency streaming and native real-time full-duplex audio models, their codecs, speech processing, model variants and exact task algorithms;
- **Image:** Qwen Image, rectified-flow image models, autoregressive multimodal image MoE, Stable Diffusion 3.5 and frontier/new image architectures;
- **Audio:** a hosted TTS service, dual-AR speech, NVIDIA Canary/Parakeet, Whisper, Kokoro, VibeVoice, Qwen TTS and other top open or hosted models;
- **Video:** Wan 2.1/2.2, hosted generation services, dual-stream video DiT 1.5, coupled A/V models, asymmetric-attention video DiTs and future model architectures;
- **Decision:** proprietary and open typed finite-choice/score decision families;
- **Text special:** ternary-compressed and other source-encoded low-bit, unconventional/hybrid language models, without copying/rewriting inference engine code;
- **Multimodel/multimodal:** one fabric that can run combined models, models with native multiple modalities, and model-specific specialization at every legal computational, hardware, task and placement layer;
- **Serving:** modular, high-performance multi-model router/serving; all relevant APIs and transport types needed for media, realtime, combinatory models, including patterns represented by the user-supplied example "gpt-realtime-2.5";
- **Research organization:** one lead and distinct deep model-family research streams, as many as useful, source-backed and reconciled to minimal composable design; research may be delegated only when real subagent tooling exists.

**Critical interpretation:** "gpt-realtime-2.5" is a user-supplied example, not verified to be a currently released official OpenAI API model. Official model catalog consulted on 2026-10-15 documented GPT-Realtime-2.1 and GPT-Live families. Preserve desired **capabilities** (duplex native speech, barge-in, playback reconciliation, tool events, low latency), not an invented proprietary model implementation.

**User forbids app code edits in this design phase.** No model download, runtime binary, API server, GPU driver permission change, live phone call, provider API credentials or remote deployment authorized.

### U-041 | USER | 15 October 2026 — preserve full comprehensive architecture and all research as on-repo implementation/handoff artifacts

The user requested:
- write **all** multimodal and realtime design plans, full architecture specifications and any other useful artifacts into .agents/{docs,plans};
- comprehensive attention to each modality/model/algorithm, not just executive summary;
- record each individual research stream's learned methods, source links, release/codec/version caveats, comparisons, negative tests, open questions, implementation guidance;
- preserve **all prior user decisions**, design rationale, rejected alternatives and pending work across original project and this research;
- author sufficiently detailed comprehensive implementation handoff so independent future external LLMs/agents can recover exact project intent, governing design, source status, phase ordering, acceptance gates and what remains unimplemented **without relying on the conversation**;
- keep design-only scope: documentation permitted, **no Rust/application/build/CI/test changes**, and preserve other active agents' changes/staging.
- no implied approval to run a model or proprietary service or modify running systems.

**Current fulfillment:** retained architecture/assembly/serving contracts,
individual source research dossiers and this USER/PROPOSAL/OPEN ledger preserve
the substantive design. The multimodal plan TODO carries implementation/tests;
Git preserves earlier documents. Append-only handoffs, SHA manifests and dated
snapshot copies are superseded by the current TODO workflow and are not required.

## 2. Prior user decisions explicitly retained (not newly re-approved)

| Existing ID(s) | Binding original intent | This plan preserves |
|---|---|---|
| U-001, U-003, U-017 | lowercase zerogravity; Rust-owned original engine and general architecture; attributed prior art without unwanted clones | one Rust engine, no C++/Python hidden inference backend, source/license verification |
| U-004..U-009 | standard/hybrid/MoE massive model families, semantic IR, extreme hardware optimization while preserving portable fallback | no model family special-cased into global router or separate engine |
| U-011, U-015, U-018, U-022 | comprehensive plans, source evidence, independent LLM handoff, simple/maintainable/extensible architecture | detailed docs and falsifiable criteria, decisions annotated as USER/PROPOSAL/OPEN |
| U-019..U-020 | NVIDIA/AMD/NPU/AOT/rack-scale devices and heterogeneous remote/tunnel execution | unified physical provider/placement/events, not per-modality device stack |
| U-021 | native built-in model router, many models and profiles, OpenAI-compatible serving and simple one-command UX | one EngineCore and bounded task/protocol adapters, no automatic proxy-per-model |
| U-025..U-027 | Qwen3.5 0.8B then 2B CPU→CUDA/ROCm/Vulkan, Gemma4 E2B/26B then Qwen3.8 27B CPU-first; 8K Phase1 16/16; extended 16K/32K deferred | MM design does NOT silently reorder practical implementation gates |
| U-028..U-029 | independent long-context GDN accuracy investigation and three parallel workstreams including multimodal research | separate CPU math/GPU/architecture ownership, no invented subagent execution |
| U-030..U-039 | source-correct GPU P07-A→G3a and critical review gates, true device readback/quarantine, single StateRoot, no false hardware proof | existing partial GPU status honest; source-free media mocks ≠ media support |
| All previous explicit user decisions | full record in .agents/plans/DECISION_REGISTER.md | current decisions and Git retain history; plan TODOs track implementation without snapshot copies |

## 3. Research-derived design proposals DMM-001..DMM-032

| ID | Status | Proposed decision and reason |
|---|---|---|
| DMM-001 | PROPOSAL | Adopt v5's six narrowly scoped multimodal contract refinements as implementation design over v4.3, not a separate inference architecture |
| DMM-002 | PROPOSAL | Use one SemanticProgram/ExecutableBundle; RIR/KIR/CIR are compiler-internal views, not extra externally committed runtimes |
| DMM-003 | PROPOSAL | Keep one TaskCoordinator PublishRoot per session/shard, one global PhysicalResourceArbiter and per-card bounded device workers |
| DMM-004 | PROPOSAL | Typed ValueDescriptor supports tokens, PCM, acoustic features/RVQ, pixels, latents, video frames/packets, transcript, decisions and structured outputs |
| DMM-005 | PROPOSAL | Exact rational Timebase/clock domains and codec/resampler delay; no fake equivalence between audio samples, codebook depth, frame PTS and token positions |
| DMM-006 | PROPOSAL | Verified ModelAssembly with component artifacts/ports/processor/codec/license/version, separate from replicas and application workflows |
| DMM-007 | PROPOSAL | Distinct trusted TaskDriver algorithms for causal AR, CTC, RNN-T, TDT, finite choices, duration-prosody TTS, nested RVQ, flow/diffusion, joint AV |
| DMM-008 | PROPOSAL | Source-registered scheduler/solver/guidance/seed is semantic and may not silently change as hardware speed optimization |
| DMM-009 | PROPOSAL | Model topology orthogonal to TensorEncoding/activation transform/accumulation: ternary PTQ/PQ2+Hadamard is compulsory example |
| DMM-010 | PROPOSAL | Source CPU reference where claimed runnable and original checkpoint+processor goldens before accelerator/full-model support status |
| DMM-011 | PROPOSAL | Media model state is typed multi-track TaskFrontier, but one root publisher and state effect/fork/rollback authority persists |
| DMM-012 | PROPOSAL | AV coupled architectures publish valid joint CommitGroups, not independent arbitrary TTS mux |
| DMM-013 | PROPOSAL | Add RealtimeSession with independent capture/perception/response/audio/video lanes and Session/Turn/Transport/Physical epochs |
| DMM-014 | PROPOSAL | Separate prepared, committed, delivered, acknowledged and played/presented frontiers. Playback is evidence from client, never inferred from transport alone |
| DMM-015 | PROPOSAL | Add canonical MediaEvent stream with payload lease, clock, source model, lane, sequence, commit group, finality and cancellation epoch |
| DMM-016 | PROPOSAL | Soft real-time QoS and stage-aware admission/priorities, not hard real-time promise; audio responsive while video yields at proven solver boundaries |
| DMM-017 | PROPOSAL | Transport adapters for CLI/SDK, HTTP/SSE, WebSocket, WebRTC, optional SIP and async media jobs all share one native EngineCore |
| DMM-018 | PROPOSAL | Versioned model/task/protocol capability manifest; OpenAI-compatible only for tested observable fields and event behaviors |
| DMM-019 | PROPOSAL | OpenAI Realtime barge-in and Gemini Live reconnect are external protocol conformance scenarios, not reasons to copy proprietary model internals |
| DMM-020 | PROPOSAL | Hosted generation and voice services, an external decision service, proprietary realtime services live at explicitly authorized TaskServiceAdapter, not DeviceBackend |
| DMM-021 | PROPOSAL | External job Unknown/reconcile/idempotency/asset rights/cost/privacy are first-class; no blind retry of accepted remote creates |
| DMM-022 | PROPOSAL | Hardware providers may specialize at op, layer, model-stage or closed whole-model graph only with exact source/shape/numeric/state proof |
| DMM-023 | PROPOSAL | Do not force single global token scheduler, one media codec rate, generic CLIP text encoder or model-name-based device routing |
| DMM-024 | PROPOSAL | Source-specific Wan2.2 high/low denoise experts are timetable-based, not token-level sparse FFN expert routing |
| DMM-025 | PROPOSAL | Direct finite option-logit TaskDriver for open decision models with raw/calibration provenance; generated JSON is not source-equivalent |
| DMM-026 | PROPOSAL | Native full-duplex (or Qwen-Omni) model supported without obligatory ASR→LLM→TTS intermediary; separate cascade also supported |
| DMM-027 | PROPOSAL | Do not broadly label "streaming" until PCM/codec valid chunk latency and interruption are actually measured under source checkpoint |
| DMM-028 | PROPOSAL | Per-model loaded immutable weights may be shared across sessions only when full exact source identity and state isolation permit |
| DMM-029 | PROPOSAL | Add MM13–MM24 real-time/serving deterministic mock tests to MM01–MM12, then model-specific fixtures and physical support tiers |
| DMM-030 | PROPOSAL | Document support status tiers Recognized→Semantic→CPU→PartialPhysical→FullPhysical→Streaming→Realtime plus separate OpaqueHosted |
| DMM-031 | PROPOSAL | Preserve default zero-config simple UX; advanced model/codec/device/transport knobs progressive disclosure and explicit numerical receipts |
| DMM-032 | PROPOSAL | Permit adding a new model via source adapter, task driver, codec and qualified kernel without editing central Router, TaskCoordinator or protocol parser |

None of DMM-001..032 has independently been ratified as frozen code ABI by the user. The user asked us to specify these to enable later implementation decisions and review.

## 4. Open design decisions OMM-001..OMM-022

| ID | Priority | Exact unresolved question | Default until resolved |
|---|---|---|---|
| OMM-001 | High | Strict bit-exact versus source-acceptable perceptual/numeric tolerance for media backends | enforce source-verified model's existing numeric acceptance; no silent widening |
| OMM-002 | High | First legally/hardware feasible ASR source checkpoint Whisper or Parakeet? | choose only after exact model/processor license and CPU feasibility |
| OMM-003 | High | First TTS: Kokoro small non-AR followed by dual-AR depth and full-duplex? | architectural plan, not implementation authorization |
| OMM-004 | High | First image flow checkpoint and original source/VRAM/license? | source-free mock then verifiable small model |
| OMM-005 | High | Exact autoregressive image MoE release generator AR/solver path | audit selected published entrypoint before claiming |
| OMM-006 | High | Native CUDA/HIP LLVM23 vendor SDK availability/approval | remain unqualified, do not download/change permissions |
| OMM-007 | High | Real video 3D VAE tiling/temporal shape and memory fit targets | no unsupported full-model GPU or realtime claim |
| OMM-008 | High | Physical GPU allocator headroom, UMA/GTT and cross-process contention accounting | conservative operator cap and measured evidence |
| OMM-009 | High | Trusted safe physical GPU quiescence and unknown-work recovery | retain quarantine/fence; process restart policy |
| OMM-010 | High | Default real-time P95 mouth-to-ear/TTFA/jitter target by hardware/load | configurable soft QoS, no numeric SLO invented |
| OMM-011 | High | Playback acknowledged vs actually heard after reconnect | represent uncertainty/provenance explicitly |
| OMM-012 | High | Voice cloning, biometric voice reference, face/video content consent and retention | deny remote egress/clone unless explicit authorized |
| OMM-013 | Medium | Async hosted provider cost, idempotency and acceptable retry policy | Unknown/reconcile after ambiguous create |
| OMM-014 | Medium | WebRTC/SIP optional transport packaging/licensing and auth default | transport adapters optional; loopback safe defaults |
| OMM-015 | Medium | Whether same session can change model/voice/codec mid-turn | require source-compatible checkpoint+explicit new generation |
| OMM-016 | Medium | Live session persistence and encrypted checkpoint/codec/buffer retention | ephemeral bounded by default, no silent media persistence |
| OMM-017 | Medium | Source licensing for dual-AR/rectified-flow/various commercial model weights | per-checkpoint license receipt and refuse unlicensed use |
| OMM-018 | Medium | Vendor API exact released model name “gpt-realtime-2.5” | **unverified name**; versioned capability contract not hardcoded |
| OMM-019 | Medium | ModelAssembly codegen component ABI freeze and dynamic plugin policy | initially one Rust crate and static registered modules |
| OMM-020 | Medium | Fine-grained per-session cancellation vs GPU process/physical-card epoch | session/turn cancel only, card epoch reserved for card failures |
| OMM-021 | Medium | What constitutes authorization to external provider via CLI alias? | explicit model/provider/profile/egress opt-in, no path string magic |
| OMM-022 | Medium | Continuous compilation/autotuning overhead versus audio latency | precompile warm shape bands, no hot path dynamic search |

Each question must be resolved by user instruction or measured/source-qualified research; a future agent cannot silently mark it solved because a source README claims performance.

## 5. Inherited exact model-priority and project-path requirements

Canonical repository: /home/pvl/spaces/abird/src/zerogravity; connector root /home/pvl/spaces/abird. Do not read files from unrelated aw/abird-health projects. Important model order: Qwen3.5 0.8B → 2B CPU through 8K, then NVIDIA CUDA, AMD ROCm/HIP, Vulkan; after those CPU-first Gemma4 E2B, Gemma4 26B-A4B, Qwen3.8 27B. Separate future multimedia research does not reorder. Rust-owned engine, no foreign inference runtime hidden under Rust, source verified semantic math, no fake GPU proof or unsafe early StateRoot publication. Built-in multi-model router and high-performance serving remain user goals, not implementation claims.

Previous user-approved original research topics (Strix Halo cooperative NPU/GPU, DGX Spark, TPU/wafer-scale AOT device, distributed placement/tunnels, extreme MoE and quant specialization) remain first-class architecture goals; the v5.1 review adds modality/task/serving semantics but does not supersede those commitments.

## 6. Research and source provenance

Official source URLs and claimed model mechanisms are individually recorded in:
- multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md
- multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md
- multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md
- multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md
- multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md
- multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md
- multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md

Research references were reviewed against publicly available upstream docs as of 2026-10-15, not tested on all exact source checkpoints. No third-party blog/forum is stronger than original repositories, papers or verified APIs. Retain source/license attribution in the linked technical owners and the decision register; historical archives/manifests are not current acceptance requirements.

## 7. Changes explicitly forbidden during this documentation task

- No source code, Rust/Cargo, tests, scripts, Nix/CI, live network APIs, model weights, driver permissions, daemon configuration or release modification.
- No staging/committing other agents' files, no whole-tree stashes/reset/rebase.
- No rewriting older user decisions or changing their status to approved.
- No fabricated subagents, upstream benchmarks, provider model IDs or real hardware media inference proofs.
- No new runtime architecture implemented merely because diagrams and type sketches are written.
- No automatic external egress or provider account installation.

The original scope was design-only; it did not authorize runtime/provider execution. Current deliverables and actual tests belong in the applicable plan TODO, without a separate manifest/handoff.

````

## Source: .agents/research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md

Original UTF-8 bytes: 8183; SHA256 `10aa7ce134a6d8876833a2483f3826d599c74e51ae73c563b13ee1fa33dfa445`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Proposed native FP32 GEMV FMA qualification

Status: proposal only; no implementation, policy enablement, performance gain, or acceptance is claimed. H24 frozen executables and their measured WideF64 arithmetic remain unchanged. This candidate reserves no next phase ID. It is queued after the user-ordered workspace/layout redesign and its immediately following left-right concurrency redesign pass; an independent bounded optimization may proceed only when compatible with that ordering and explicitly coordinated. Qualification of the existing native paths across the authorized hardware inventory remains the preceding acceptance boundary.

## Bounded change and current source

`crates/zerogravity-gpu/src/source_accumulation.rs` already defines `WideF64` and a diagnostic `NativeF32`. `source_mul_add` selects either a widened multiply/add followed by FP32 conversion or the backend's explicit FP32 `fma`. The widened path documents checkpoint qualification rather than universal IEEE equivalence. `model_workers.rs` exposes `model-workers-native-cuda`, but that fixture exercises one chosen matrix per model; it does not establish all-codec or resident whole-model eligibility.

The resident program currently checks WideF64 capability at construction and passes the constant `true` at each Project kernel launch: original F32, Q4_0, Q4_1, Q8_0, and Q6_K matrix paths. The provider device-activation execution path already derives the same kernel boolean from `SourceAccumulation`. The smallest candidate change is a cold, immutable resident projection policy using these existing kernels and quantization bindings. No separate runtime, source cache, state authority, or numerical engine is needed.

Default policy remains WideF64 until every applicable gate below passes. Initially expose NativeF32 only as an explicit qualification choice on the tested native backend/device, with source-bit comparisons mandatory. Refuse unsupported or unqualified combinations before launch. A selected region failure propagates and preserves existing StateRoot rollback and quarantine; it must never silently execute on CPU or switch accumulation policy after failure.

## Identity and physical ownership

Carry the immutable policy in the prepared model/program configuration. Include it in every relevant prepared-program/cache identity and generated kernel comptime variant key, together with existing backend/device/source identity. Confirm that source weight ownership can be shared across variants while generated kernels and any policy-specific prepared metadata cannot be confused. An existing prepared program must not change policy between launches.

Change only Project accumulation selection. Preserve original source quantization, integer dot/reduction topology, intermediate FP32 boundaries, finite/status certificates, static projection bindings, source leases, resource admission, actual completion events, cancellation cohorts, and root publication. Keep all other FP64 use required by independently qualified operations: normalization and L2, GDN coefficients and recurrence, SiLU, RoPE, attention QK/value residual correction, softmax and its two-round reciprocal. Native FP32 GEMV eligibility does not imply that the device-resident pipeline has become FP32-only.

## Numerical acceptance

First qualify an actual device FP32 FMA operation against the pinned host source `fmaf` arithmetic, comparing all 32 bits and retaining physical readback receipts. Cover finite normal values, positive/negative zeros in all operand positions, cancellation to zero, subnormal operands and results, underflow ties, minimum/maximum subnormals, minimum normal and adjacent values, maximum finite and overflow ties, halfway cases, widely separated exponents, sign permutations, and deterministic randomized bit patterns. Include the previously discovered coupled midpoint construction where widened multiplication/addition can double-round; a native FMA must match the original fused source result. Check signed zero explicitly. Record hardware/backend/compiler/driver and floating-point mode; do not infer denormal preservation from language syntax or an instruction name. Separate nonfinite refusal fixtures from admitted finite numeric fixtures.

Then qualify all five actual source matrix kinds (F32, Q4_0, Q4_1, Q8_0, Q6_K) using original trained compressed bytes and activations for both pinned Qwen3.5 models. Include every used codec, multiple matrix shapes and reduction lengths, genuine nonzero token-dependent activations, and adversarial rows/activations. Compare source quantization intermediates/statuses and full output vectors exactly, so a changed quantizer or dequantization ordering cannot conceal a difference. Synthetic stress supplements trained tensors and is not model-context proof.

Required end-to-end regressions for every proposed eligible backend/device profile:

- Both real models: full four-token resident pipeline, every logit, all 42 physical state domains, and ordered diagnostic trace; state-only prefix specialization and all finite/status certificates remain exact.
- Genuine sequential official-tokenizer 512-token contexts with the existing 12 milestones, including 255/256/257 padding transitions, full logits/state/trace, private chunk replay, restore, and exhausted-capacity refusal. A constructed 4096-element operator fixture is not whole-model context acceptance.
- Concurrent independent sessions; shared immutable source ownership; stale/foreign root, epoch and generation rejection; reset, restore, and snapshot lifetime; canonical unused KV tails; admission failure before publication.
- Post-launch graph cancellation across the 256 boundary and checkpoint upload cancellation: actual unknown completion, authentic handle retention/quarantine, device fence and unchanged committed root/state. No fake Ready or trusted recovery.
- Portable CPU/device checkpoint round trips, cross-backend replay where available, exact resumed suffixes, and final physical charge retirement on successful paths.
- Product raw prompt/output parity against the fresh pinned CPU candidate, device-token feedback, warm transfer counters, and existing prelaunch limits; default SDK-free Vulkan regression remains intact.

A failure blocks NativeF32 eligibility for that profile or input contract. Retain the first failing fixture, logs, hashes, differing bits and source arithmetic before proposing a repair. Do not widen tolerances or redefine source mathematics. If native arithmetic flushes denormals, identify the exact affected instruction and source operation before considering a separately qualified correction.

## Performance evidence and closure

Measure the existing WideF64 and candidate NativeF32 variants using comparable frozen source/build inputs, hardware clocks/power conditions, CPU thread counts, source cache and warm-up policy, context/prompt/token counts, diagnostic settings and transfer boundaries. Report cold model preparation separately from warm prefill and decode. Alternate repeated measurements and retain dispersion and raw timing receipts. Include end-to-end product inference and the dispatch count; host launch/validation overhead may dominate. Report host CPU time/counts and actual transfer/allocation counters where available. Neither an FMA flag, a native instruction, nor an isolated kernel timing proves model throughput improvement.

Run complete build/lint/tests, independent source gates, physical acceptance and relevant documentation checks before enabling any default. Record actual commands/results, exact eligible source/backend/device/compiler profiles, model-content identities, failures, measured timing limits and remaining work in the [CPU/GPU optimization TODO](../plans/cpu-gpu-optimization/TODO.md) and assigned run TODO. Follow [TODO-SPEC.md](../plans/TODO-SPEC.md): commit verified work atomically with explicit paths, group completion under its landing commit and carry unfinished gates forward. No comprehensive handoff or documentation SHA manifest is required. Until these gates pass this file remains a proposal; existing acceptance retains its recorded source/hardware/numerical scope.

````

## Source: .agents/research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md

Original UTF-8 bytes: 5633; SHA256 `ef8fc19168c602ce843dc19e95842294579bc0985e625c28dd5b84a19788d5ab`.
Resolve embedded relative paths against the original path, not this volume.

````text
# A1 local cache — protocol and API reference

Implemented and tested in `3787519`. Current implementation/test status and
remaining acquisition work live in [model-acquisition/TODO.md](../plans/model-acquisition/TODO.md).
This file is a technical reference, not another handoff or progress checklist.

## Protocol decisions from source/design review

1. Reuse optional already-locked/cached `sha2=0.10.9` and `rustix=1.1.5`; keep
   normal core/CPU/inspection dependencies unchanged. Linux capability-relative
   `openat`/NOFOLLOW operations protect every directory component. Read-only
   handles and cooperative immutability do not freeze same-UID external writes.
   A3 must hash actual owned source-copy bytes before SourceIdentity issuance.
2. Transfer admission uses a caller-supplied existing ResourceArbiter, registered
   domain and owner. Admit bounded staging before allocation/body reads; never
   instantiate or register another arbiter inside a pull. This does not establish
   global host/UMA or cross-process memory admission.
3. Disk quota covers conservative cooperative cache-owned usage and live-attempt
   reservations, including partials, orphan verified blobs and bounded metadata/
   allocation allowances. Persist root policy and refuse conflicting limits.
   Filesystem free space is an observation, not a reservation; unrelated writers
   are outside that accounting scope. Inject ENOSPC rather than filling the host.
4. Before atomic ref rename, a failed transaction leaves the previous binding
   exact. If rename succeeds but parent-directory sync fails, rollback cannot
   establish the old durable state. Return a cache-specific explicit
   DurabilityUnknown outcome containing the visible complete binding and cause.
   Never return ordinary success or pretend nothing was published. Child crash
   recovery may observe an old or new complete snapshot, never a partial closure.
   Existing core Error and GPU actualReady semantics remain unchanged.
5. Stable lock inodes are never unlinked/recreated while another process can
   hold or open them. Attempt-lock ownership, not PID existence, establishes
   whether a reservation is abandoned. Bound lock/metadata growth. Long object
   transfers do not hold the short global publication lock.
6. A1 has no HTTP/provider interface. Exercise actual network-isolated local
   hit/miss/import/inspect/verify and audit the selected dependency closure.
   The provider-constructor panic subfixture belongs to A2 when that interface
   exists; do not invent a provider runtime to manufacture this test result.
7. Snapshot leases retain validated metadata and the exact opened object handles.
   Removal/eviction remains typed Unsupported, including while leases exist.
   Artifact publication is distinct from executable eligibility or ModelReady.

## Public API and bounds

The existing artifact owner exports Linux-only `cache` and explicit
`diagnostic-cache` features. Use `CacheOptions::new(root, disk_limit_bytes)`,
`CacheStore::open`, `begin(manifest, &TransferAdmission, cancel)`,
`import_object`, then `commit(reference)`. `CommitOutcome` distinguishes Durable
from DurabilityUnknown. `resolve_ref`, `open_snapshot` and `verify_snapshot`
return a SnapshotLease retaining validated metadata and exact read-only files.
`usage`, `root_identity` and `filesystem_profile` expose bounded observations;
`remove_snapshot` returns Unsupported. No new public runtime/authority is added.

Defaults:1MiB manifest (8MiB hard ceiling),4096 files,1TiB/file,4TiB aggregate,
JSON depth8,64KiB transfer buffer and30-second lock deadline. These logical
artifact limits do not raise the existing4GiB executable-source limit. Paths
are unchanged case-sensitive UTF8 bytes, <=4096 bytes/255 per component/64
components; no case/Unicode normalization. Inventory is capped at131072 nodes
and8MiB retained path characters. Transfer admission reserves64MiB control
headroom +8×configured manifest bound +transfer buffer +64KiB verification
buffer before begin's control copies, transfer allocation or body read.
Filesystem block/fragment units must be nonzero powers of two <=4096 bytes;
other allocation profiles are explicitly Unsupported.

## Important implementation limits

- Disk accounting covers cooperating cache writers, partials/orphans/metadata
  and live future reservations; it does not reserve filesystem-wide capacity.
- An existing final snapshot cannot consume the reservation for a new private
  snapshot rebuild. Reference versions retain named backing even after alias
  replacement and on identical-record dedup, keeping live bytes charged.
- Verification hashes bounded bytes and checks length/inode/mtime. Hardlink-only
  ctime changes are allowed; same-UID external payload writes remain possible.
  A3 must hash actual owned source-copy bytes before source identity issuance.
- Publication tests covered18 operation types/38 callbacks with76 injected I/O
  and76 actual child-exit/restart cases. They establish same-kernel process
  recovery, not power-loss durability. Initial root policy creation was outside
  fault injection. Post-ref-rename sync uncertainty is explicitly reported.
- Entry/import cancellation and a genuinely held publication-lock deadline are
  tested. Cancellation inside an observed EWOULDBLOCK iteration was not claimed.
- A1 has no HTTP/provider/CLI pull/eviction/loading; cache data is not ModelReady,
  DeviceReady or StateRoot publication. No new checkpoint/SDK was downloaded and
  no GPU was launched. Actual network-isolated local hit/miss passed; provider
  constructor/credential offline fixtures belong to A2.

````

## Source: .agents/research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md

Original UTF-8 bytes: 16722; SHA256 `b45d031e0c058de6777bf652e2b0092ac1ea72c6c6c48b9157dc561b388b2839`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Qwen basics, complete reference parity and advanced decoding

**16 October 2026 — implementation roadmap; new gates below are NOT RUN.**
The user explicitly orders: get the Qwen basics working, match every capability
and optimization of llama.cpp and vLLM, then enable further appropriate methods,
including MTP, speculative decoding, n-gram, DSpark2 and DFlash2. This extends
the existing serving and specialization plans; it does not replace the current
source-correct engine or restart its architecture. Parallel model source work
continues under [the model addition plan](../plans/model-execution/TODO.md).

## Ordered implementation and honest completion

1. **Qwen basics:** first finish source/configuration/tensor admission and real
   CPU execution for the exact new Qwen3.8 27B checkpoint. Keep the original
   Qwen3.5 0.8B/2B paths exact. Then qualify the new model's full resident GPU
   execution through the same mathematical schedule and backend interfaces.
   Baseline decoding, tokenizer/template/stop behavior, independent sessions,
   rollback and physical ownership must work before speculative optimizations.
2. **Reference capability and optimization parity:** enumerate and implement
   the complete union of both pinned upstream capabilities applicable to Qwen,
   including public interfaces and internal execution optimizations. Follow the
   shared layout → measured left-right → XDG acquisition sequence. An interface
   shim, parser or renamed option is not behavior or performance parity.
3. **Advanced decoding and additional optimizations:** implement and physically
   qualify applicable MTP, draft models, n-gram/suffix lookup, EAGLE variants,
   DFlash/DFlash2, DSpark and any separately source-verified DSpark2, parallel
   drafting, dynamic depth and confidence-based verification. Evaluate their
   combinations with batching, caching, quantization and placement.

Several named methods already exist in the reference tools. Their rows remain
open in the parity ledger until their implementation and advanced-decoding
gates pass; moving work to step3 never counts as completing those rows early.
"Full parity" is reserved for the complete ledger, including these overlaps.
Additional optimization ideas remain separate from verified upstream behavior.

## Pinned source baseline and coverage denominator

| Project | Source revision | Purpose |
| --- | --- | --- |
| llama.cpp | `2851b3bcbf8d993a2243ae751c0a7dfa8e944a40` | Capability/optimization baseline; same pin as acquisition research |
| vLLM | `a8be1500bad93d3c459e435bf71194aa8062e49d` | Capability/optimization baseline; same pin as acquisition research |
| DeepSpec | `005e03b81cec38b7da6399833d609ee89a2587f2` | Primary DSpark source and exact drafter compatibility research |
| DFlash | `07ebd93db9f472af339b644bb70221ad8428328a` | Primary DFlash/DFlash2 source and exact drafter compatibility research |

These are parity research pins, not replacements for the independent numerical
oracle or each model's own immutable checkpoint/reference revisions. Refresh
upstream deliberately: review a pinned delta and add new rows, keeping the old
comparison reproducible. A mutable "latest" label cannot define completion.

Historical research captures and repository-tree receipts are retired under the
[accepted checkpoint](../plans/accepted-baseline/TODO.md). The preserved documentation frontier contains 145
**unreviewed** document paths. It is a seed, not an exhaustive capability
inventory or a passing result. No checkpoint, tokenizer body or SDK was acquired.

The 105 provisional capability groups, the retained
[CLI candidates](../plans/qwen-parity/CLI_OPTION_FRONTIER.csv) and
[API candidates](../plans/qwen-parity/API_ROUTE_FRONTIER.csv) frontier lists remain open
planning inputs.
The historical audit counted16 classes,12,989 pinned paths and11,615 source
blobs awaiting semantic disposition. Its raw tree captures are retired; regenerate
the complete frontier from the explicit source revisions before closing parity.
The twelve QPA-C closure gates remain proposed, not passed. These inventories
are a broader starting point, never an exhaustive-parity claim.

The exhaustive ledger must enumerate **all** public CLI/API/configuration
options, model/task and codec registries, sampler/logits processors, attention
and state kernels, backend/build switches, loaders, cache and scheduler modes,
parallel/connector interfaces, documentation and their reference tests. Record
each source item at its exact path/symbol/revision and map aliases or equivalent
implementations to canonical capability IDs. Generated arguments and delegated
components require their own inspected source boundary. No unexplained source
frontier item may disappear from the denominator.

Each capability needs: reference project/pin, source evidence, exact behavior,
Qwen variant/task applicability, shape/codec/device restrictions, dependencies,
owner/phase, local implementation location, acceptance IDs, actual test receipts,
performance measurement and outstanding gaps. Statuses distinguish UNREVIEWED,
PLANNED, IMPLEMENTED, VERIFIED and NOT-APPLICABLE-WITH-REASON. The latter needs
an independently reviewed semantic or physical incompatibility; it cannot hide
unfinished implementation or erase the broader supported-hardware roadmap.

## Required capability inventory

The categories below are coverage obligations, not claims that both reference
tools support every item or that Zerogravity already implements them. Split
every concrete upstream mechanism into its own sourced ledger entry.

| Area | Complete coverage obligation |
| --- | --- |
| Model semantics and tasks | Dense/MoE/hybrid Qwen variants, exact layer gates/norms/RoPE, GDN/QSA/PLE where source-defined, logits, tokenization/chat templates, reasoning control, tool parsing, embeddings/scoring/pooling and released image/video/audio pathways |
| Artifacts and adapters | GGUF/split GGUF, safetensors/shards/sidecars, all registered loaders and Qwen-applicable weight codecs, source/derived identity, conversion/quantization, LoRA/adapters, draft/vision components, pinned acquisition/cache/offline behavior |
| Sampling and output | Every sampler/logits processor and composition order, RNG/seed/ties, logprobs, stop/EOS, penalties/bias, grammar/JSON/schema/regex, tools/reasoning parsing, streaming and accepted-prefix output |
| CPU execution | Threading/affinity/NUMA, ISA-selected and BLAS paths, tiling/packing, weight reuse, quantized GEMV/GEMM, prefill/decode specialization, batch shapes and memory mapping/prefetch/locking policies |
| GPU and mixed execution | CUDA/HIP/Vulkan and remaining backend goals, all attention/projection/activation/quantization kernels, source-exact fusions, flash/paged/sliding/hybrid attention, matrix instructions, graph capture/replay, device sampling, streams/events, overlap and placement/offload/splitting |
| Memory and caches | KV/state formats, paging/COW/prefix sharing, hybrid recurrent snapshots, quantized caches, context extension, sparse/index/encoder caches, cache keys/eviction, CPU/SSD offload, sleep/preload/snapshots and pinned/UMA/zero-copy transfers |
| Scheduling and concurrency | Continuous/chunked/async prefill and decode, batching/bucketing, preemption/priorities/fairness, admission, context budgets, cancellation, multi-model residency, encoder/decoder scheduling and cold/warm load behavior |
| Distributed execution | Tensor/pipeline/data/context/expert parallelism, collective/communication overlap, expert routing/load balancing, multi-device split modes, RPC/disaggregation/connectors and transfer/cache ownership |
| Serving and integration | Offline/online/batch operation, relevant OpenAI/Anthropic interfaces, model listing/routing, tools/multimodal/reasoning, metrics/tracing/profiling/health, request validation, lifecycle and documented integration behavior |
| Speculation and joint tuning | Every upstream proposer/verifier/scheduler mode, dynamic depth/trees/confidence, heterogeneous vocab rules where supported, cache/sampler/grammar interactions, draft placement and useful goodput under mixed concurrent traffic |

In particular, preserve hybrid SSM/recurrent prefix checkpoints and replay,
bounded sliding-window replay, persistent decoder/monokernel options, kernel
autotuning and compilation/AOT warm-start/cache identity as distinct sourced
optimization rows. A prepared Rust plan is not driver graph capture. The
speculative inventory also keeps LiLiCoRR and all registered rejection policies;
each requires its own target/version/quality contract before promotion.

Use the existing [joint serving acceptance plan](VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md),
[artifact pull contract](MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md),
[hybrid/drafter semantics](../docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md)
and [specialization gates](EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md).
Reuse their owners and tests; do not build a second serving engine or duplicate
completion/session/resource authority for parity.

## Basics gates

| ID | Required acceptance before baseline promotion |
| --- | --- |
| QBASE-01 | Exact checkpoint/config/tokenizer/template/reference identities; source-defined GDN and full-attention gates distinguished; no semantics inferred from family name |
| QBASE-02 | Exact tensor headers, shape/dtype/codec/shard closure and untied output binding; bounded malformed/overflow/duplicate rejection before allocation |
| QBASE-03 | Explicit source storage and physical admission for artifacts larger than the current4GiB source materialization cap; no silent cap increase or duplicate whole-model copies |
| QBASE-04 | All mathematical operators and reductions match the chosen source; grouped Q/K→V GDN geometry and intermediate rounding tested independently |
| QBASE-05 | Real trained CPU full-vocabulary/state, multi-token and nonzero replay tests against an independent executable reference; fixtures alone do not qualify inference |
| QBASE-06 | Tokenizer/chat/reasoning/stop boundaries, greedy behavior and supported sampling agree with the declared contract |
| QBASE-07 | Independent sessions/shared immutable source, reset/snapshot/stale attempt, resource refusal and rollback preserve the sole committed root |
| QBASE-08 | Full GPU model graph on each applicable initial hardware profile: actual physical source/state/readback, admission, cancellation and original-handle quarantine |
| QBASE-09 | Genuine context/corpus and shape boundaries qualified explicitly; original512 GPU capacity and existing8K divergence/16K/32K gaps remain visible |
| QBASE-10 | Default/minimal/composed features, build/lint/tests, source qualification and documentation integrity pass; unsupported combinations fail explicitly |

The [large-source basic design](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md)
proposes verified, explicitly admitted owned shards inside the existing source
owner. Its six acceptance gates are unrun. Keep the current4GiB default;
read-only/private mappings alone do not freeze externally writable files.
Route existing and new host owners through a coherent configured physical pool
before claiming global admission, and account for UMA sharing with GPU memory.
Original large BF16 payloads exceed local RAM; select a capable authorized host
or an explicitly identified and independently qualified encoded source profile.

## Speculative and advanced execution contracts

MTP is a trained source head; prompt n-grams are history-based proposals;
hashed PLE/source lookup tables are model parameters. Keep all three distinct.
EAGLE, DFlash2 and DSpark are checkpoint-specific proposal programs, not a
generic switch compatible with any Qwen checkpoint. Pin target, drafter,
tokenizer/vocabulary mapping, hidden-state taps, embeddings/head sharing,
position/mask/block limits, precision and any confidence/Markov components.

Primary DFlash source lists a Qwen3.8-27B DFlash2 checkpoint. DeepSpec documents
DSpark and DFlash; a separate **DSpark2** version has not yet been established
from primary source in this research. Keep the requested DSpark2 track open,
resolve its exact source/version and evaluate it when available; do not rename
DSpark or assume it is DFlash2. No auxiliary weights are downloaded here.

The target verifier remains authoritative. Every proposed branch owns its
private target/draft KV, GDN/QSA/index state, positions, token history,
sampler/RNG/grammar/stop state and output frontier. Publish only the accepted
legal prefix through the existing session authority. Zero acceptance, rejection,
EOS/grammar termination, cancellation and stale completion must discard the
remaining branch without corrupting the committed root. GPU cancellation keeps
all original handles/charges until actual completion or quarantine/fencing.

For greedy lossless mode require exact target-only tokens and accepted state
under the declared tie policy. For stochastic mode implement the correct
accept/reject/correction distribution, including zero-probability and numerical
boundaries, with explicit target/draft RNG ownership. Distributional correctness
does not imply identical fixed-seed draws for algorithms consuming RNG in a
different order; document/test any stronger reproducibility promise separately.
Never widen source arithmetic tolerances or use predicted confidence as token
acceptance authority. Synthetic acceptance is a labeled benchmark diagnostic,
never valid model output. Any target-only policy is explicit before execution;
a failed GPU region never silently substitutes CPU math.

All gates below are **PLANNED**, with real model/device fixtures required:

| ID | Required acceptance |
| --- | --- |
| QSPEC-01 | Native MTP tensor/head/tap/position/state and target compatibility |
| QSPEC-02 | Separate draft model plus vocabulary/embedding mapping and placement |
| QSPEC-03 | Every source n-gram/cache/map/suffix variant, repeat boundaries, bounded memory and cross-request isolation |
| QSPEC-04 | Exact EAGLE variant and hidden-state tap/vocabulary contract |
| QSPEC-05 | Exact DFlash/DFlash2 mask, block diffusion, positions/taps and target pairing |
| QSPEC-06 | DSpark Markov/confidence scheduling and separately verified DSpark2 semantics if available |
| QSPEC-07 | Accepted lengths0..K, empty proposals, all-reject/all-accept and bonus token boundaries |
| QSPEC-08 | Greedy target-only token/state equality and stochastic correction/distribution properties |
| QSPEC-09 | Hybrid recurrent/KV/index rollback across chunk/window/padding/checkpoint boundaries |
| QSPEC-10 | Grammar/tools/reasoning/stop/EOS/logprobs and streaming emit accepted output only |
| QSPEC-11 | Wrong/missing/stale drafter, source/codec/tap mismatch and unsupported vocabulary refuse safely |
| QSPEC-12 | Physical target+draft+branch memory admission and concurrent shared-source accounting |
| QSPEC-13 | Actual post-launch cancellation, stale completion, original handle quarantine and device fence |
| QSPEC-14 | Bounded adaptive depth/trees/confidence with load-aware scheduling, fairness and single-stream/saturated goodput |
| QSPEC-15 | Pairwise then material higher-order combinations with quantization, caches, graph capture, batching and placement |
| QSPEC-16 | Controlled target-only/reference/new-path benchmarks, quality and complete-request counters on all applicable initial devices/CPUs |

## Parity and performance closure

Complete parity requires zero untriaged frontier items, a disposition for every
enumerated reference capability, passing functional/source/ownership tests for
each applicable row, and a separately documented performance result for every
optimization row. A documented incompatibility remains visible; it does not
authorize claiming that missing functionality is implemented. Initial hardware
qualification covers the existing five-GPU/three-CPU fleet; unsupported or held
devices remain explicit and the wider backend roadmap remains open.

Measure identical checkpoint/derived precision, context/corpus, sampling,
concurrency and power/thermal conditions. Report TTFT, TPOT/tail latency,
prefill/decode throughput, accepted useful output, saturated goodput, peak
physical memory, source transfers, draft acceptance/reject work and fairness.
Include cold/warm cases and both baseline and combined optimizations. Record
regressions and select paths by measured workload/capability rather than enabling
every optimization unconditionally. No speedup or parity claim follows from a
lower launch count, a tiny operator benchmark or a reference project's claim.

````

## Source: .agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md

Original UTF-8 bytes: 522; SHA256 `a0832269bd377daa5652b21efa93f749aa85803efb5dc853fdae28eca238bf21`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Development, builds and releases

Current build, installer, release-definition and NixOS guidance lives in the
[development guide](../../docs/DEVELOPMENT.md). See the
[plan index](../plans/TODO.md) for current implementation and qualification, and
the [GPU library runbook](../../crates/zerogravity-gpu/README.md) for optional
backend builds and their limits.

This legacy path remains for existing links. Previous snapshot text is available
in Git history; its dated status and probe commands are not current guidance.

````

## Source: .agents/research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md

Original UTF-8 bytes: 10291; SHA256 `bb4602272d9f666762f28f3c3d1b7c45a3f53beece49e949934158486103dedc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity Specialization Prior Art — Specialist Source Atlas

**15 October 2026 · DESIGN/RESEARCH.** No benchmarks, model downloads, application code, drivers or running inference changed in this research. Read together with the three new authoritative specialization contracts, existing v4.3 architecture and v5.1 ModelAssembly/serving. GitHub sources were inspected through the GitHub connector, not experimentally reproduced.

## 1. Original public sources

| Source | Relevant learning |
|---|---|
| Specialist-source README | 100GB-class derived quantization model packs on 128GB Strix Halo; source quant and speculative verification |
| gfx1151 report | Wave32 WMMA, LDS bank conflicts, MoE packing, batching cliffs, accepted versus proposed tokens, negative optimizations |
| Quant optimizer experiment | per-tensor LDLQ-shaped sensitivity, self-sampled in-domain calibration and disjoint evaluation, constrained bit allocation |
| Measured benchmark methodology | actual prompt classes, contexts, sampling, hardware, single/aggregate rates and numerical FP8 quality controls |
| ROCm decode kernel notes | derived-quantization packed GEMV, gfx12 vs gfx1151 ISA, source fallback and negative kernel optimizations |
| MiMo adaptation/runbook | E8M0 scales, 39 SWA/9 global, DFlash exact attention, layer47 FP16 overflow, UMA page-cache loading risk |
| Qwen serving notes | trained MTP and prompt n-gram, model cache, multiple concurrent sessions and reduction-order near-tie differences |
| GLM serving notes | MLA/KDA/MTP and optional 64MiB BF16 eh_proj, image input, exact tool/think streaming |
| MiMo serving notes | native DFlash target speculation, per-request gating, lossless greedy testing and realistic slowdowns |
| First-image optimization | avoid unbounded first-use MIOpen solver search; legally source-equivalent stride2 patch-merge conv→GEMM |
| Runtime behavior hook | explicit optional activation-space steering, new behavior variant not tensor/kernel equivalence |
| Behavior steering presets | direction+layer strengths and hashes, optional model behavior, safety/quality are separate gates |
| Organization profile | actual source weights and tested AMD hardware scope |
| Qualitative gallery | qualitative agent-made page comparison, NOT controlled numerical benchmark |
| Derived quantization codec/kernel prior art | codec/kernel source prior art, NOT a mandated foreign execution engine |
| Community evidence pointer (supplied by user) | community evidence pointer; text not reliably retrievable in this read-only pass, so no unsupported claims based on comments |

**Pinned inspection note:** observed public Git tree SHA cb0d162ba71d1ff35b0362b73f5c09ef8420eda6; it is a tree digest, **not a Git commit ID**. Source blob SHAs included gfx1151 report c65506b54e7f200945b4985b49ab5e0abde485f0, quant optimizer 9ec2dbe5637f4b32ba4a486cab0c0ab809797372, benchmark methodology 1ae541455f4ba9cd079c83603cb9eb0170ac0552, MiMo runbook 8241095573d532cc6227f2e105e2777e498a37b7. Re-fetch/check exact commit before source-code porting.

## 2. Published benchmark claims, source-specific

Authors report on a Ryzen AI Max+ 395 / Radeon 8060S gfx1151 with 128GB unified memory:

| Derived quant/model | Packed size | Prefill 8K | Speculative decode by prompt class |
|---|---:|---:|---:|
| Qwen3.8-Flash-Next | 95GB | 1481 tokens/s | 45.7–66.8 tokens/s |
| GLM-5.3-Flash | 99.73GB | 634 tokens/s | 26.2–33.4 tokens/s |
| MiMo-V2.6-Flash (1.5) | 105.8GB | 760 tokens/s | 28.9–37.0 tokens/s |

These figures come from upstream README/benchmarks, with differences in prompt categories, context, sampler and measurement method. **Not Zerogravity measurements and not a controlled model-quality-normalized ranking.**

Published quantization-fidelity comparison to original FP8 Qwen reports around 93.81% top1 agreement and KL ~0.0216 on 3084 scored positions for one specialist Qwen derived pack. Its target-lossless speculation refers to **the derived quantized target**, not mathematical equivalence with original FP8 and not guaranteed batching determinism.

## 3. The strongest engineering findings

### Packed GPU kernels
- Direct derived-quantization/QTIP low-bit GEMV with verified RDNA Wave32 WMMA fragment mapping produced a substantial acceleration relative to reconstruct+GEMM fallback on its tested source.
- Source-verified LDS staging stride 5 instead of 4 reduced 32-bank collisions, but making every caller's scratch width match a small MoE kernel broke another 16-warp GEMV; output checks must include model-native complete rows and not just PPL.
- GDN skinny GEMM/quant residual mixing were useful on tested shapes. Some INT8 DOT4 variants and cache/graph capture modifications were neutral or slower.

### MoE and speculative batching
- Representative verify forward cost T_ms about 32.4 + 9.04 × physical rows, due partly to extra distinct routed-expert bytes per additional row. Do not assume extra draft rows are free.
- A grouped-MoE 16-row kernel cliff caused batch B5 at depth2 = 15 physical rows to outperform B6 = 18 physical rows, which triggered another nearly-empty group.
- Single-stream ~47 tokens/s in a favorable measured configuration versus realistic sustained B5 aggregate ~74–75 tokens/s: latency and throughput are different objectives.
- Static depth2 appeared best on one cherry-picked prompt yet multi-prompt runs did not justify changing dynamic policy default.

### Quant and calibration
- 2.05bpw Qwen pack ran around 25.7 tokens/s versus ~44.1 tokens/s for a 3.05bpw variant due to K2 trellis decode/tiling. Less quant storage can cost more execution.
- Experimental per-tensor sensitivity optimizer uses quantizer-shaped perturbation, held-out quant quality and representative trace; explicitly labeled **experimental and untested on sparse models**.

### Exact MiMo and source semantics
- MiMo source variant's 39 sliding-window and 9 global-attention layers use different per-slot fixed rings and global paged KV; codebook/draft also has an independent ring.
- DFlash partial RoPE, layer tap index, bidirectional block, learned mask embedding, sink bias and value scaling cannot be represented as generic default MTP.
- A particular MiMo derived pack had FP16 layer47 intermediate overflow and used paired conversion/runtime rescaling; not universal numerical equivalence and not license to clamp.
- Drafting helped code/reasoning, slowed some prose; model-specific confidence gating and disabled baseline required.

### Unified memory and first-image
- The specialist source's GB10 MiMo loading notes report roughly 86GiB clean page cache competing with about 86GiB GPU weights in one ~121GiB coherent memory pool after buffered read. Driver MemFree is not the same as OS MemAvailable or safe physical headroom. Their external guard included optional known-shard page eviction and process kill under configured threshold, **not recommended as a generic automatic engine policy**.
- GLM's first new image resolution triggered expensive first-use convolution/GEMM autotune searches; a verified patch merge conv→GEMM avoids shape-specific solver searches, with source numerical gating.

## 4. Architecture/source mapping

| Source idea | Zerogravity authoritative home | New spec |
|---|---|---|
| Derived-quantization/QTIP packed trellis/codebook, derived quality, calibration | FORMATS_QUANTIZATION_AND_LOADING, P03/P20 | EXL3_QTIP_AND_DERIVED_QUANTIZATION_CONTRACT |
| Qwen GDN/QSA/PLE; GLM KDA/MLA/DSA/mHC | MODEL_SEMANTICS_AND_COMPATIBILITY, P11–P14 | EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS |
| MiMo E8M0/SWA/v_head_dim/DFlash/FP16 intermediate | MODEL_SEMANTICS_AND_COMPATIBILITY, P11/P14/P15 | EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS |
| MTP sidecars/acceptance, dynamic draft cost | SCHEDULER_BATCHING_AND_SPECULATION, P15 | EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS |
| gfx1151 WMMA/LDS/DOT4, latency shape | DEVICE_KERNEL_AND_CODEGEN, STRIX_HALO dossier, P07/P20 | GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT |
| MoE route-aware row/shape batching | COST_MODEL_AND_AUTOTUNING, P12/P16 | GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT |
| UMA headroom/pagecache/driver pressure | MEMORY_RESIDENCY_AND_IO, P10 | GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT |
| First-image cold shape, tuning | COST_MODEL_AND_AUTOTUNING, P20 | GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT |
| Steering behavior mode | COMPOSITION_AND_EXTENSION_CONTRACTS, v5.1 ModelAssembly | EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS |

## 5. Explicitly NOT adopted

- No mandatory Python/tensor-framework/external serving runtime or model family-specific serving process.
- No universal CUDA cp.async translated verbatim to AMD; choose legal ISA variants.
- No global fixed 16-row batching rule or universal ladder of draft depths.
- No automatic CPU expert offload/graph capture/INT8 activation or deeper drafts without measured source-gated benefit.
- No assumption smaller bitrate improves performance; no new derived quant auto-downshift under memory pressure.
- No automatic process SIGKILL, system-wide page-cache eviction, unapproved ROCm driver edits or user credential downloads.
- No activation-steering preset silently bundled/activated: behavior-changing model mode must be user-selected, audited and separately tested.
- No qualitative gallery comparison used as proof of source model equivalence or objective model superiority.

## 6. Evidence limits and next audit

This research inspected the specialist source documentation and repository tree, not the original full trained Qwen3.8/GLM5.3/MiMo2.6 checkpoints on Zerogravity, and not every GPU kernel's codegen. Before implementation pin model source/derived quant/aux processors, original mathematical reference and source commit, then run scalar/host/original model and real hardware. Source names and engineering conclusions are correctly useful, but physical throughput claims are not transferable across model precision, OS, GPU, firmware or context workload.

**All technical claims from the source study belong in the new specialization contracts, acceptance matrix and handoff; this atlas is evidence, not another engine authority.**

````

## Source: .agents/research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md

Original UTF-8 bytes: 20887; SHA256 `cf46b1e593ed5cf118ede4b676ef187f4dcec1987b7f58ae8d3e3b8f63cd10f3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — First-Class SSD Streaming Implementation Roadmap and Acceptance Tests

**15 October 2026 · DESIGN/PLANS ONLY, 32 planned tests, NOT run.**
**Status:** explicit user architectural requirement to treat SSD-backed model residency and streaming as first class across the same model and tensor structures; no user authorization to download enormous models or run physical memory-pressure experiments.
**Authorities:** .agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md and existing v4.3 ModelAssembly/SemanticProgram, CIR, DeviceBackend, ResourceArbiter, ModelRouter, TaskCoordinator. No model-specific disk engine, no new persistent IR.

Current implementation work is tracked in the [memory and streaming TODO](../plans/memory-and-streaming/TODO.md). This document retains the design and the SSD acceptance scenarios; their presence does not establish streamed-model support.

### Reading the design without the retired handoff

Use the [virtual tensor residency contract](../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) for the source/materialization/lease protocol and the [source research](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md) for attributed findings. VST-USER-001 records the requirement for first-class SSD residency through the same tensor structures as RAM/GPU backing; it does not authorize large downloads, memory-pressure experiments or provider execution.

One verified source tensor binds to either an already-ready resident lease or a provisional source extent. A miss requires checked IO, admitted staging and actual upload completion before the same source kernel can consume it. Host read completion alone cannot publish a ready device slot or StateRoot. The resident path requires no per-token filesystem access, IO worker wakeup, cache lookup or hidden bulk copy; these are acceptance conditions to measure.

Keep four policies distinct over that shared substrate: routed expert slabs, bounded selected lookup rows, immutable committed session snapshots, and optional slower dense-trunk reload. A universal LRU across these lifetimes is not the design. The encoded-reader and mixed-codec regressions below remain required even when a cache reports high hit rates. Prefetch or lower storage use cannot change source routes, quantization, numerical policy or the committed result.

## 0. Implementation principle and current status

The old design already covered expert SSD miss path, disk-only n-gram/Engram rows, prefetched experts and disk KV snapshots conceptually. What needed strengthening is ONE verified immutable TensorView and source tensor identity independent of backing, clear materialization/IO/physical GPU lease contracts, zero-extra-IO resident fast path, whole-batch encoded reader pin lifetime, mixed-codec byte budgeting, IO provider portability and performance-truth admission. All are specified now; no SSD streaming path is hereby implemented in the current Rust binary.

Do not equate existence of an SSD-backed file with whole-model SSD streaming qualification. Physical backend support must be stated per exact source format, quant, checkpoint, task, region and real device.

## 1. Minimal changes to the existing engine — no new layers

- P03 ArtifactCatalog: SourceExtent and verified immutable file backing, logical TensorKey/View identity, exact tensor slice/shape/codec, file descriptor and derived quant hash.
- P05 prepared CIR/ExecutableBundle: one MaterializationEffect with ready/dependency/read-lease, and validated resident/miss-capable command alternatives; no new serialized compiler IR.
- P07 optional physical IO/DeviceBackend providers: bounded file/host IO worker and real H2D upload event, generation-tagged ResidentSlot and protected reader sets, atomic table publish; reuse device events and BufferLease instead of another lease authority.
- P10 ResourceArbiter and domain-cache clients: total OS/DRAM/VRAM/UMA/pinned/staging bytes, expert size-class cache, lookup-row cache, frozen StateSnapshot tier, per-shard source singleflight, OS page-cache headroom.
- P11/P12 exact source model operations: router, source expert and source hashed lookup access stay mathematical authority, independent of physical placement.
- P15/P16 physical miss-aware target/draft/batch cost, ready frame and safe prefetch, no hidden source routing change or cross-session mutable state sharing.
- P20/P22 tuning/validation: source original full model, forced cache miss and IO failure, real physical source parity, no uncaught memory-pressure failure, cold/warm performance and P95 QoS.
- v5.1 image/video/audio: same typed source extents, component-stage residency and media-file chunks where legal, but codec/window/finality policies distinct from expert cache. Audio source real-time QoS can REFUSE SSD placement that violates bound.

## 2. Implementation slices and gate definitions

### SSD-P0 — source-only contracts and a fake tiered executor
Define a source-verifiable logical TensorKey/View, immutable SourceExtent, BackingTier and lease-generation vocabulary. Create simulated GPU/host/SSD with controlled IO ordering (no actual source model or GPU). Test same source kernel outputs with 0/1/all resident and controlled misses; no hidden approximate data conversion. ModelRoot effect remains one owner.

**Gate:** SSD-01..04, 08 and 12: fake concurrent command queues, logical tensor identity preserved, source data exact, wrong shard/quant refused, no per-element trait dispatch.

### SSD-P1 — immutable file-backed tensor and bounded IO
Keep original ArtifactCatalog source descriptor open against same-size path swap; checked offsets, on-disk sorted expert/lookup/codec metadata, exact read length/integrity. Add a small Rust portable buffered pread worker pool; source may optionally support Linux io_uring, O_DIRECT where legal and macOS/Windows mapped/overlapped adapters, but no mandatory new OS dependencies.

**Gate:** SSD-05..07 and 17..20: partial, O_DIRECT invalid alignment, short read, simultaneous same-offset coalescing, cancellation IO lifetime and optional clean file-cache hints.

### SSD-P2 — resident GPU plus source-preserving demand materialization
On a real supported backend, bind already verified source tensor to either ready direct GPU pointer or asynchronously H2D-loaded slot, with exact transfer event and generation. A captured command must pin all expert buffers it references even before submission. On no safe victim, flush/quiesce/retry or bounded backpressure.

**Gate:** SSD-09..16 including source-real DS4 #384 encoded-not-executed regression; keep source logits/StateRoot invariant. No "Ready" from a completed host read alone.

### SSD-P3 — mixed-quant MoE streaming and nonrouted source residency
Implement small bounded per-quant-size-class expert slabs under ONE total byte budget and source route cache hotness policy, with optional resident prefix layers. Mandatory dense/attention/router/state stays resident where possible. Prefill staging can borrow only retired idle optional cache without wiping hot decode working set. On all-resident path, direct compiled kernel needs no host cache lookup.

**Gate:** SSD-13, 21..24, 26 and 29: Q2/Q4 alternating layers, cache warm after short follow-up, GPU direct-hit copies measured and honest cost, no unbounded host prefetch.

### SSD-P4 — disk-only indexed tables and frozen state
Source-specific huge Qwen PLE/DeepSeek Engram lookup row index with bounded coalescing and tiny RAM cache. Separately serialize exact committed StateDomain KV/SWA/GDN snapshots into private versioned disk cache with corruption/tenant/quant/processor and fork/ABA checks. These are separate cache policy clients over same source IO and ResourceArbiter/StateRoot.

**Gate:** SSD-25, 27, 28 and 30: row ordering, exact source hash mapping, disk KV private restore and invalid source rejection.

### SSD-P5 — adaptive placement, QoS and end-to-end integration
On 64/92/96/128GiB coherent RAM and at least one discrete GPU + NVMe machine, measure full model resident and forced streamed outputs, OS MemAvailable/PSI, driver pool, staging, cache ratios and end-to-end latency. Test multiple Qwen/GLM/DeepSeek specialist source recipes as source/weights permit; source manual approvals before 100GiB memory pressure. At model load choose direct resident when feasible, streaming if needed and source/device verified. Refuse real-time deadline that SSD misses cannot meet; never change model/quant silently.

**Gate:** SSD-31, SSD-32, entire corpus with actual source-trained model parity, memory headroom and cold/warm measured QoS. No blanket "every GGUF SSD supported" claim.

## 3. SSD-01..SSD-32 planned acceptance matrix

| ID | Independent scenario | Required source/physical safety result |
|---|---|---|
| SSD-01 | same immutable tensor in GPU, host RAM and file-only fake tier | same TensorKey/SourceId, logical math and state outputs |
| SSD-02 | fully resident model with instrumented filesystem/IO worker | zero per-token IO requests, cache worker wakeups and new allocations |
| SSD-03 | cache miss during source model region | same prepared semantic kernel after true materialization; no changed router |
| SSD-04 | file-backed tensor view with offsets and padded format | exact checked source bytes/geometry, no generic reinterpretation |
| SSD-05 | same-length atomic pathname replacement after source open | opened inode + verified model hash preserved; no mixed metadata/data |
| SSD-06 | short/torn read, corrupt model scale/codebook and negative extent | fail before GPU submit, no false Ready/Root or unrelated fence |
| SSD-07 | simultaneous sessions request same cold immutable expert | singleflight IO, both receive correct distinct reader leases |
| SSD-08 | cancel queued IO before any device submission | IO buffer lifetime valid until IO completion; no GPU fence |
| SSD-09 | CPU source exact model outputs at cache capacities 0,1,all | same model semantics, numeric comparison and accepted tokens |
| SSD-10 | GPU route hits ready slot 10K times | no mandatory CPU readback, per-token global lock, disk read or DMA |
| SSD-11 | cache miss H2D pending; CPU marks SSD read Done | slot unavailable until genuine GPU copy event Ready |
| SSD-12 | stale slot generation/pointer table ABA | old expected TensorView cannot read new resident tensor data |
| SSD-13 | model with 0, 1, 288/512 experts, long hot/cold/skewed routes | bounded pool and exact top-k routing; no guessed warm expert substitution |
| SSD-14 | DS4 #384: two-layer encoded batch, capacity one, later expert miss | earlier unexecuted GPU reader prevents slot reuse; full logits bit/source parity |
| SSD-15 | capture/graph replay with evicted pointer and stage-split fallback | stable pinned pointer or validated indirection generation, no stale address |
| SSD-16 | unknown GPU completion / cancel after upload or dispatch | physical buffers and staged IO retained until proved device quiescence |
| SSD-17 | O_DIRECT short alignment / unsupported filesystem | validated buffered fallback or typed Unsupported, no incorrect bytes |
| SSD-18 | Linux io_uring/worker pread, macOS mapped and Windows IOCP mocks | same original artifact/read/lease result, bounded queues and cancellations |
| SSD-19 | host RAM cache vs O_DIRECT/no page cache, repeated hot experts | source parity, byte budget and honest reduced NVMe misses |
| SSD-20 | concurrent IO cancellation while SSD read in flight | original host pinned IO buffer not freed before completion |
| SSD-21 | DS4 #503 alternating Q2/Q4 expert sizes across layers | bounded size-class caching without hot cache teardown or memory overcommit |
| SSD-22 | prefill after warmed decode working set | preserve hot entries when safe; scratch preflight and no unconditional flush |
| SSD-23 | 97% expert hit but hidden full packed compact copy | instrumentation catches physical copy bytes; direct hit variant compared |
| SSD-24 | cache budget N per different format sizes | one global **byte** cap prevents allocating N slots in each class beyond budget |
| SSD-25 | giant FP8 BF16 ngram/Engram lookup rows on SSD | exact source hash/row IDs, dedup and right reorder; no full table residency |
| SSD-26 | dense trunk pinned vs optional unpinned/reload | same source result but explicit latency/capacity ModeReceipt; no false zero-cost claim |
| SSD-27 | persist session KV after committed root and resume | exact compatible model/tokenizer/quant/tenant; no replay of uncommitted state |
| SSD-28 | disk KV encrypted/permissioned and source revision mismatch | reject foreign tenant/unsupported state; proper error/recompute policy |
| SSD-29 | Qwen target+DFlash/MTP and shared warm experts across 4 sessions | correct expert union, draft/target cache state and model SessionRoot isolation |
| SSD-30 | source VAE/video-tile/audio frontend offloaded by legal stage | exact frame/sample clock/codec output, no giant unsafely resident intermediate |
| SSD-31 | coherent 128GiB with hot 80–100GiB source page cache and active KV | admission keeps OS health reserve; no OOM killer/page-cache thrash |
| SSD-32 | full original trained MoE model SSD streamed vs fully resident on physical backend | complete logits/StateRoot source parity, cold/warm TTFT, P95, memory and actual IO receipts |

**Every SSD-XX above is PLANNED / NOT EXECUTED.** Source-free fake IO tests are not original model GPU qualification; actual hardware support must remain per device+model+quant.

## 4. Required negative-fixture detail — DS4 #384 / #503

### Encoded GPU use-after-slot-reuse (SSD-14, SSD-15)
- Construct two layers A and B, one reusable expert slab S, immutable source expert X then different expert Y.
- Encode A reading X from S, without submitting or waiting; start encoding B and request Y; allocator must see A's **encoded reader lease** and refuse to reclaim S.
- Only when A physically finishes may S be reused for Y; publisher generation must change and no A stale pointer may read Y.
- Compare full source tensor logits and state; simple CPU directory/key check is insufficient because wrong bytes produce valid GPU completion.
- Exercise queued, submitted, captured graph and cancellation of encoded-but-unsubmitted command. No timed release.

### Mixed-format thrash (SSD-21/22/24)
- Q2 expert gate/down 7MiB, Q4 expert 14MiB, alternate A-B-A-B across layers, maintain one global 64MiB byte budget.
- Per-class pools may persist simultaneously; no group switch clears unrelated source residency, and both pools combined never exceed 64MiB plus separately accounted mandatory scratch.
- Start short prefill on warm decoder; no unconditional cache purge; staging may evict truly idle optional victims only if it cannot fit, and must record chosen policy.
- Real device copy bytes on cache hits tracked; an extra 2GB/token device-to-device copy with 97% hits fails a zero-extra-IO claim even though source numerics pass.

## 5. Admission/placement and high-performance UX

Default profile: Auto, source identity and exact numerical mode unchanged, resident first when mandatory weights+all active state+OS reserve fit; otherwise source-qualified dynamic tiered mode where available, or typed Unsupported/InsufficientResources. Don't expose an additional user-facing model alias "ssd_model" unless a distinct quant/artifact is actually chosen.

Advanced optional controls (design proposals, not flags yet):
- operator maximum host/device memory and minimum OS headroom,
- expert cache byte allowance, pinned prefix/host slab, read queue depth,
- prefetch lookahead bytes/depth and chosen SSD buffered/direct tier,
- QoS class: throughput/batch, interactive, soft-realtime (admit/refuse based on measured miss),
- model/task cache warmup only when verified safe.

User-visible summary must say exact model, source/derived quant, resident mandatory bytes, effective optional GPU/host SSD cache bytes, source file table bytes, active state budget, estimated cold miss and expected QoS; don't claim SSD access itself is zero latency or use driver GTT addressability as free DRAM.

## 6. Benchmark and evidence reporting

Fields: SourceTensorId+artifact hash, size/quant classes, device driver/compiler/ISA, filesystem/SSD model+protocol, physical topology, OS MemAvailable/cgroup/PSI, initial cache hotness and expected lookup/expert routes, route entropy/unique expert count, prefill/decode/draft accepted tokens, actual IO bytes/ops/latency and staging/H2D bytes, cache hits/misses and evictions, protected encoded+GPU reader leases, pagecache or O_DIRECT mode, p50/p95/p99 TTFT and decode, Qwen/GLM/DeepSeek complete logits/StateRoot source parity and full startup/teardown resources.

Do **not** promote upstream DS4/prior-art author numbers into local performance promises. Compare independent resident, pagecache-warm, cold SSD, host-tier hot and mixed-quant modalities with equivalent full model/numeric task. Document negative/neutral experiments for later agents.

## 7. Dependency and next engineering plan

This roadmap is additive to original Zerogravity P00–P23, and existing Qwen3.5 0.8B/2B CPU→physical GPU followed by Gemma4/Qwen3.8 order is unchanged. Do not start 200GiB downloads before approved test/hardware/resources.

Ownership:
- Artifact source owner: P03, source-extents and schema.
- CIR/Device owner: P05/P07, IO/transfer/slot lease/graph correctness.
- Resource/storage owner: P10, bytes/headroom/cache and safe immutable sharing.
- Model math owner: P11–P14, source-exact routing and lookup.
- Draft and batching owner: P15/P16, acceptance, IO-aware expert union and QoS.
- Perf/release gate owner: P20/P22, physical original model source+telemetry.
- Realtime media owner: v5.1 TaskDriver and serving contracts; stage-specific offload cannot corrupt playable frontiers.

Other agents may be working in repository; use precise owned paths, no unrelated staging/stashes. Test via source-free mocks then original checkpoint CPU, then physical GPU, then measured SSD stress only on authorized machine. A separate SSD engine is a design failure.

## 8. User intent and design decisions

**Binding user instruction, 2026-10-15:** fix documentation archive generator cleanly and make SSD streaming natural, first-class, highly performant, reusable with GPU/RAM/SSD backing via similar structures. User authorized docs/plans and necessary minimal documentation tooling, not implementing a new native SSD inference engine or downloading large model weights.

**Design proposals:**
- SSDD-001: Same immutable logical TensorView across tiers, ready lease in existing BufferLease/CIR.
- SSDD-002: Resident direct fast path has effectively zero per-token software IO overhead; SSD miss paid honestly.
- SSDD-003: Expert, lookup and frozen state share physical/IO substrate but have separate eviction/finality policies.
- SSDD-004: Full encoded/captured/submitted GPU reader set pins slot before eviction; generation ABA test.
- SSDD-005: Mixed quant size classes share one global byte cap; preserve hot decode cache across prefill.
- SSDD-006: Physical OS/driver headroom before cache; direct IO and mmap chosen by measured platform.
- SSDD-007: Source route and quant/numeric identity never change due to cache status.
- SSDD-008: Committed StateSnapshot disk persistence private, source/version compatible and independent of weight cache.
- SSDD-009: Auto resident first, stream when required/qualified, with QoS/cold/warm mode receipt; no change of target model.
- SSDD-010: Same model and API independent of backing, no additional storage inference engine.

**OPEN:** which cache/popularity and IO algorithm performs best on each actual target; default minimum physical OS headroom; pressure stress authorization; optional GDS/remote storage policy; disk KV retention/encryption; mandatory realtime QoS; source model/backend/quant streaming qualification. These need measurements and/or explicit authorization and must not be filled with DS4/prior-art-specific constants.

## 9. References and implementation boundary

Use the [current plan](../plans/memory-and-streaming/TODO.md) and [repository workflow](../../AGENTS.md) for an assigned implementation slice. The [virtual residency contract](../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md), [source research](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md), [physical memory design](../docs/MEMORY_RESIDENCY_AND_IO.md), [command/event contract](../docs/ASYNC_RESOURCE_COMMAND_IR.md) and [state transaction contract](../docs/STATE_CACHE_AND_TRANSACTIONS.md) supply the detailed technical requirements. Start with the bounded source-free scenarios, then original checkpoint CPU, actual GPU and authorized SSD-pressure gates. Mock IO cannot establish streamed-model or physical-lifetime qualification.

````

## Source: .agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md

Original UTF-8 bytes: 18981; SHA256 `41e0d994ce6c52039e0f441486772db8a7caaff474a1a4992a62fc71e9bc6d54`.
Resolve embedded relative paths against the original path, not this volume.

````text
# AMD Strix Halo dossier — gfx1151 extreme inference targets

**15 Oct extreme-specialization research reconciliation:** [extreme-specialization source atlas](SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md), [gfx1151/UMA spec](../docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) and [EXTREME P03–P22 tests](EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) detail architecture-specific WMMA fragments/LDS bank padding, source trellis codebook, sparse MoE batching/target speculation and memory-safe 100GiB model admission. Quant 2.05 vs 3.05 bpw can invert decode speed; CPU expert offload and some graph optimizations proved negative on tested upstream. Host MemAvailable, reclaimable file cache, pinned allocations and GPU GTT are different. These are **external measured gfx1151/GB10 observations**, not tested Zerogravity performance. No hidden ROS daemon/process kill or universal tuning flags for that project.


> **Research/provenance only (v3, 2026-10-12).** Project names, URLs, specific implementation techniques and source-author benchmarks in this document are **attributed prior art, not normative zerogravity runtime or compiler types**. The independent governing architecture is [Modular AI systems](../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [Composition contracts](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md), and [Prior-art license policy](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md). External projects never become a requirement to clone a model-specific inference engine; technical/measured evidence remains available for independent validation.


**Hardware:** Ryzen AI Max+ 395 / Radeon 8060S, RDNA 3.5 `gfx1151` APU, often 64/96/128 GB unified LPDDR configurations. Distinguish machine-specific firmware and Linux kernel GTT/TTM behavior from actual available physical memory. Our architecture must also run on non-Strix GPUs via other kernels or portable fallback.

## 1. Why this APU is special

Substantial unified memory can accommodate weights that do not fit consumer discrete VRAM. CPU/GPU share physical RAM, but bandwidth, cache coherency, page pinning and competing CPU activity determine practical performance. GPU ROCm/HIP may report large virtual GTT capacity even while physical `MemAvailable` is insufficient. Kernel launch overhead, wave32 occupancy, quantized weight streaming and sparse attention selection can dominate; generic GPU graph coverage is often uneven.

**Design principle:** `gfx1151` is an eligible optimization target, not a separate hard-coded engine. Specialized regions only when semantic op/shape/quant/capacity/numerical contracts match; other operations fall back without rewriting model definition.

## 2. Models and primary cost sources

### Qwen3.8-Flash-Next
Research: 48 backbone layers, 36 Gated DeltaNet and 12 sparse attention layers, 512 routed experts top 10, gated residual, n-gram lookup and trained MTP, ~125B language backbone and ~6B active/token. The advertised 51B n-gram parameters form a different access pattern than MoE; support disk-only read of original BF16 rows where checkpoint provides them.

Main bottlenecks:
- small decode rows repeatedly stream dense and selected expert weights;
- QSA indexer/top-k/sparse attention at long context; CPU fallback can cause drastic depth scaling;
- GDN recurrence and convolution require correct state and fast small-row kernels;
- n-gram disk lookup must avoid forced giant resident allocation;
- MTP verification and rollback state must be efficient and numerically valid.

External reference:
- **Specialized gfx1151 engine**: native model-specific quant, `gfx1151` highly tuned kernels, ~1584 pp at 8K, ~1517 pp at 131K, ~37.6 short-context serial tg and ~55–56 coding spec (project claims); core engine source not published. Its ~62 GiB native checkpoint and ~47.7 GiB paged lookup differ from Gufo/DwarfStar.
- **Gufo**: `UD-Q4_K_XL` 128 GiB APU, ~1700.52 pp/25.95 AR tg d0, ~1478.66 pp/24.92 AR tg d128K; MTP mixed ~31.32, repetitive ~60.39; GPU attention and state snapshot source inspectable. One warmed sample/point caveat.
- **DwarfStar**: Qwen Metal/CUDA, not ROCm as of 2026-10-10 inspected snapshot; Q2 main/MTP 41.73 GiB and original BF16 n-grams 95.37 GiB disk-only; cannot claim its Strix Qwen performance.

### GLM-5.3-Flash
Approx 321B total/18B active, 45 layers; 34 KDA recurrence + 11 sparse MLA/DSA, 42 routed MoE layers x288 experts top8, mHC, 1 MTP. `gfx1151` 128GB can hold aggressive Q2 expert checkpoint under certain resource policies; 92/96GB requires careful scratch/expert trade-off and may stream from NVMe.

External reference:
- Maya-S Q2 ~96.5 GB GGUF, reported ~210–217 pp and ~17.8 tg on 128GB Strix with ~99.9% expert hits and 12096 experts in pool; tuning hipBLASLt/pre-fill chunk changed speed drastically.
- DwarfStar GLM-5.3 Flash Q2 about 90GiB, Strix ROCm supported including SSD expert streaming; same model family, not necessarily same quant/layout or numerical fidelity.
- llama.cpp supplies useful portable kernel/operator references, but specific sparse top-k and host round-trip paths require benchmark verification.

### DeepSeek V4 Flash
DwarfStar's research: 284B total/13B active, 43 layers; raw 128-token sliding KV + compressed historical attention at 4:1/128:1 depending on layer; 512 top-k index selection on selective layers, mHC, MoE and trained DSpark. ROCm `gfx1151` model-specific kernel improvements: 4K warm prefill ~187.26→292.86 tok/s with IQ2 MMQ/rocWMMA/attention kernel changes. This demonstrates large gains from shape-specific kernels on the same checkpoint, not Qwen/GLM direct benchmark.

## 3. Physical resource profiles (illustrative, must measure)

| RAM tier | Likely constraints | Planning response |
|---|---|---|
| 64 GiB | large Qwen Q4/GLM may not fit beside state and OS | low-bit model if user selected; disk tables; smaller contexts; explicit refusal when mandatory bytes exceed safe capacity |
| 92/96 GiB | Qwen low-bit + context feasible in some configs; GLM Q2 likely needs careful residency/tier | safe physical headroom, bounded context/prefill, expert cache sized after mandatory allocations, prioritize dense resident |
| 128 GiB | more aggressive expert-resident Qwen/GLM feasible; still OS and prefill scratch pressure | maximize useful resident pool, avoid duplicate staging, tune memory borrowing |
| Larger | weight residency easier, device bandwidth/compute/attention remain limits | adapt to actual device and controller rather than linear speed claim |

Numbers are tiers, not guaranteed working model fits. Physical availability, model quant, context, vision, draft and other processes matter. Design must use real preflight measurements instead of fixed 1GB/16GB reserves from individual reference projects.

## 4. ROCm kernel feasibility tasks

1. `hipcc`/CubeCL compile for `gfx1151`, driver/ROCm versions and codegen ISA inspected.
2. Q8_0/Q4_K/Q2_K/IQ2_XXS decode GEMV with vectorized block loads and wave32 reductions; exact GGUF quant structure.
3. Pre-fill MMQ/GEMM with rocWMMA or directly emitted matrix instructions, correct tile padding/K block reads. Compare DwarfStar's tuned IQ2 x64/y64 and Q8/F16 WMMA shapes as independent references.
4. GDN recurrent `step` and `chunk` plus short conv; optimize state update without per-token host barriers, snapshot fast path.
5. KDA fine-grained per-channel gates and state layout; measure compact recurrent representation and chunked matrix operations.
6. Qwen QSA indexer/selection/attention entirely on device; compare bounded scratch, tile occupancy, sparsity scaling to >=128K.
7. GLM MLA latent NoPE attention, DSA top-2048 and dynamic indexer, FP16 latent shared tiles; separate absorbed/expanded correctness paths.
8. Grouped MoE route+compact list+quantized gate/up/down, small-row verify and large prefill; reused expert weights across rows.
9. MTP verification compact batched rows and full state rollback; exact sampler path, no stale cache on accept/reject.
10. N-gram and Engram disk service with bounded read buffers, partial reads/corruption handling and no huge mapping.

For each track measure occupancy, register/LDS, weight bytes, instruction mix, launch count, full-model critical path and quality. A faster microkernel is not automatically retained if full HTTP gets worse.

## 5. Resident routing and streaming expert model

```text
read-only GGUF tensor manifest -> expert source
    |                  |
  dense weights     device resident cache slabs
    |                  |
GPU selected expert IDs -> device pointer table hits
    |                        |
    |                 GPU grouped kernels
    v
missing IDs -> host-mapped bounded request ring -> async read/pinned stage
                                        -> lease-safe slot eviction/upload
                                        -> device completion publish
```

All slot pointers carry a generation and active-reader lease. Avoid 100s of `hipMalloc`/free per token. For 128GB all-expert-resident case, no host trip on hit path. For 92GB cold-miss use error/timeout and explicit latency. CPU expert computation may compete for the same LPDDR bandwidth, so do not enable by default just because discrete PCIe benchmarks benefit.

## 6. Unified-memory and OS safety

Inspect GPU-visible memory pool via `rocminfo`, actual `hipMemGetInfo`, Linux kernel/device `/dev/kfd`, `MemAvailable`, `CmaFree`, cgroup available, RSS, pinned and scratch. Do not assume `rocm-smi` VRAM count is physical budget. `amdgpu.gttsize`, TTM page limits and BIOS UMA reservation can affect reported addressability but tuning them is a system-specific user action, not a runtime default or proof of physical RAM.

Failure tests must not freeze user's machine: use simulated capacity, model-free GPU allocation probes under a watchdog, then gradual measured physical loads. No disabling IOMMU. Device failure should abort request and quarantine leases, not spin indefinitely.

## 7. Large model residency and bandwidth reasoning

Approximate lower bound if `active_params * stored_bits/8` bytes must be streamed per token: Qwen 6B active at 4 bits gives ~3 GB weights before metadata and other traffic; GLM 18B at 2 bits gives ~4.5 GB. These are rough illustrative active payloads and **not** measured bytes (shared/dense inclusion, repeated experts, working-set cache and quant overhead vary). Effective t/s upper bound depends on memory bandwidth plus compute, launch, attention and SSD misses. Speculation can accept multiple tokens per target verification, changing output tokens per pass.

Memory tuning must not select an illegally small sparse-attention top-k to achieve speed. Preserve semantic selection and quant quality.

## 8. Different operating modes on gfx1151

**Latency mode:** dense/experts resident as budget allows, decode1 lowest launches, minimal host sync, spec only if net benefit. **Throughput mode:** larger native batches and grouped MoE, fairness/P95 guard, prefix shares. **Memory-safe mode:** conservatively reduced context/quant chosen explicitly and expert SSD tier as available, strict OS reserve. **Reference mode:** portable operators, diagnostic precise trace, no undocumented approximation.

Kernel and plan variants selected at model load and scheduling boundaries; avoid constant recompile/repack under changing request mix.

## 9. Benchmark scenarios to qualify

- Exact checkpoint/quant and reference logit comparison against Gufo/Maya/DwarfStar or official model, not cross-family speed.
- Qwen AR pp2048/tg128 at 0/4K/32K/128K; MTP same depths mixed/repetitive/code, reported acceptance; concurrent 1/2/4/8 sessions and model quality.
- GLM 8K context warm/cold expert cache, Q2 vs Q4 quality, initial/continued 2K/4K/8K prompt, 128/512 decode; SSD misses measured.
- DeepSeek V4 Flash 2K/4K warm prefill and DSpark low/high acceptance, no speculative baseline.
- Physical RAM and latency at 64/92/96/128GB where hardware available; no hidden swap/host OOM.
- CPU lane test on UMA vs discrete; prefetch next-layer effectiveness and IO bandwidth.
- Captured graph vs eager: CPU launch overhead, graph pointer safety and device state.

## 10. Actual speed gap ownership

If performance lags:
- prefill -> inspect tile shapes/GEMM, expert grouping, staging/dequant, quant format, occupancy/launches;
- decode -> inspect actual streamed weight bytes, dense vs sparse layer time, GPU top-k, GDN/KDA state, host sync;
- deep context -> inspect selected attention complexity, indexer/latent cache, CPU fallback, scratch spills;
- large MoE -> expert hit/miss, disk random latency, pinned copy, slot churn and quant;
- speculation -> drafter overhead, accepted length, verifier state snapshots and target rollback;
- concurrency -> scheduler head-of-line blocking, too little grouped work, too much scratch or bandwidth contention.

Do not attribute all speed loss to Rust/IR if generated GPU ISA and critical-path memory operations are the problem.

## 11. Implementation priority for this hardware

After CPU/reference IR and general dense fallback, implement small `gfx1151` kernel bakeoff, then a Qwen/GLM representative hybrid op and routed MoE, and only then full model with tiering and speculation. This order is a **proposal**; user hasn't chosen first model release priority. Strix optimization must not force model semantics or memory layouts incompatible with CPU/CUDA/Metal fallbacks.

## 12. Uncertainties to maintain openly

Qualifying model quant/checkpoint and actual real-world code quality; CubeCL codegen competitive with hand HIP; full 128GB model fit after runtime/OS reserve; 92GB expert working set and SSD bandwidth; HIP graph capture practicality; state prefix sharing cost; MTP ROI under diverse agent workloads; distribution/accuracy trade-offs. Close by physical tests and pinned reference evidence, not narrative.

## 2026-10-10 integrated v2 addendum — hardware optimization under unified compiler

**New hardware/toolchain priority:** `gfx1151` Strix Halo remains a primary target of user intent, but no purchased or attached 128GB reference machine is assumed. The historical research plan's RTX3060/Cezanne/27GB "local hardware" is *not verified current inventory*; P00 must probe actual GPUs/ROCm versions, physical RAM, cgroups, UMA firmware and KFD gfx target before claiming tests. A 92/96GB host is not equivalent to published 128GB Qwen/GLM quant context. Never blindly change IOMMU, kernel parameters or reserve GTT as physically safe RAM.

**Rust GPU provider benchmark:** `CubeCL` gfx1151 and future Rust-native AMDGPU direct instruction paths evaluated for wave32/rocWMMA or MFMA, Q2/IQ2 low-bit GEMV/GEMM, fused GDN/KDA, sparse indexer/topk and grouped experts. New NVIDIA Rust Tile/SIMT compiler choices affect CUDA portability only; don't confuse CUDA sm100 FP4 support with AMD RDNA3.5 INT8/WMMA capabilities. KernelPlan ties generated ISA/compiler/driver and packed quant layout to verified model state/quality.

**CIR differentiator:** represent device resident expert table read as typed command, host-mapped miss doorbell, bounded `pread`/IO staging and GPU upload/publish events, followed by expert GEMV. A fully resident Qwen/GLM pass has **no CPU synchronization every routed MoE layer**; a cache miss can interrupt at legal graph region boundary and resume only after valid generation. Prefill scratch borrowing must have verified release fences; CPU/GPU LPDDR contention measured.

**Model-target matching:** Qwen3.8-Flash-Next compare Gufo with exact quant/state/draft acceptance or label incomparable. GLM5.3 use Maya/DwarfStar with different Q2 recipes/quality accounted; MTP may lose on one `gfx1151` and should not default on unconditionally. DeepSeek V4 learned time-compressed attention and SSD-only Engram distinct from Qwen sparse and GLM KDA. General Llama/Gemma4/Qwen3.5+ and other AMD APU/discrete devices remain supported by generic kernels and capability negotiation.

**Real performance proof:** same checkpoint/quant/hardware and expected context, single stream vs C8 grouped, cold/warm expert/IO, long context, p95 TTFT and memory watchdog, speculative acceptance/quality. A 20% kernel microbench improvement does not establish full request/agent task t/s dominance. See `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md` S0-01/10/12 and the expanded research note.


## 2026-10-12 integration — XDNA2 NPU + Radeon iGPU on the same Strix Halo SoC

**New source:** `mikealanni/gufo` `fork-enhancements` pinned at `8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`, user Reddit 2026-10-11 on EVO-X2 MAX+395 128GB. Reports Qwen3.8 Flash-Next **UD-IQ4_XS** prefill 16K GPU-only **1,698 tok/s** vs GPU+NPU **1,821** (+7.2%), 32K NPU 1,783 and 40K cold served 1,637; do not compare numbers against upstream Gufo Q4_K_XL without checking quant, model and flags. Source actually computes *row slices* of GDN z, attention query/gate, shared expert gate/up on XDNA2 while GPU computes other projection rows, routed experts and attention. The NPU **does not** hold a second model, and this is prefill not a 1,821 tok/s decode figure.

**Performance pipeline:** RIR splits independent 4096-token-multiple projection GEMMs; KIR selects 4096×2560×1024 BF8/NPU program with source-quant and numeric provenance; CIR schedules GPU input packing, XRT BO pin/import, NPU dispatch, concurrent iGPU work, wait at first consumer, ordered output join and all-state commit. Generic/optimized GPU-only remains legal for other shapes, numerical policies and other Ryzen hardware. Optimizer should measure pinned memory/LPDDR bandwidth, XRT launch/sync overhead and load-time prepacked weights; NPU offload is not a free performance multiplier.

**Severe hardware qualification issue:** author reports machine-check hard hangs if GPU fabric clock changes during NPU+GPU activity. Their workaround requires elevated FCLK hold, NPU power mode and runtime-suspend controls; a user may see extra idle watts and firmware dependency. **Do not automatically perform privileged system tuning or turn on NPU by default.** Detect XDNA2/XRT/firmware/clock profile and expose `unqualified/experimental/validated`; GPU-only for unknown hosts. A program timeout cannot guarantee recovery of a frozen SoC. No silent IOMMU disable (can break NPU). Test isolated, fully instrumented hardware only and abort automatic enablement after any hang. See [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

**Strix validation matrix:** 64/92/96/128/192GB if physically available; `gfx1151+XDNA2` generation, GPU driver ROCm, XRT, kernel/BIOS/AGESA, idle and held FCLK, cold boot/resume, CPU+GPU+NPU contended bandwidth, NPU disabled vs enabled under identical clocks, long context and C8. A host lacking such hardware cannot have a PASSED NPU performance or safety gate. No actual zerogravity NPU implementation/benchmark exists.

````

## Source: .agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md

Original UTF-8 bytes: 17781; SHA256 `064b63c2cd342a464f8ff3325189df5ca35b83803e5c5bc45ed84e9738119d17`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — vLLM Parity and Combinatory Serving Acceptance Program

**15 October 2026 — DESIGN and planned future tests only, not executed.**
**Normative owner:** .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md.
**Goal:** implement the already designed P09/P10/P12/P15/P16/P20/P22 concurrent serving contracts with measured source-verified performance and the ability to independently reproduce gains or regressions against a pinned vLLM V1/vLLM-Omni baseline.

## 0. Status and boundaries

Zerogravity does NOT yet have a production OpenAI-compatible high-concurrency server or full-source GPU Qwen3.5 model engine. Shared immutable model source, CPU text execution, bounded fair physical-worker proofs, source-free quality and selected real CubeCL physical regions are narrower. **All SERVOPT tests below are PLANNED and must not be marked passing by documentation.**

No second serving engine, model-family GPU server, speculative sampling authority, vLLM Python runtime dependency or global resource allocator is permitted. At every phase the source mathematical reference, task-specific StateRoot and physical event/BufferLease must remain authoritative.

## 1. Development phases and dependency gates

| Phase | Artifact and integration seam | Definition of done |
|---|---|---|
| P09a request handling | ModelRouter, versioned protocol capabilities, tenant/request queue limits | bounded admission rejects overload before heavy state allocation; no false OpenAI compatibility |
| P09b per-request lifecycle | Requested→Queued→Admitted→Prepared→Issued→Completed, epoch cancellation, committed stream | typed failure/retry and no stale output after cancel/disconnect |
| P10a hybrid state page manager | exact full KV/SWA/GDN/KDA/MLA/indexer/grammar state and COW, per-domain prefix meet | independent reference replay and source-checked prefix state; tenant isolation |
| P10b physical/SSD residency | one global byte authority and real device+IO leases | no memory double-count, encoded pointer pin, source-exact cache restore |
| P12 compatible stage/route batches | true device source-compatible rows and grouped expert kernels | real physical concurrent native batched operations, exact quant/numeric policy |
| P15 target speculation | versioned MTP/DFlash/lookup and original target correction | deterministic and stochastic target-only equivalence, complete state rollback |
| P16 bounded joint planner | batch, prefill, draft, expert cache, kernel and placement candidates | best *measured* legal finite option under QoS/fairness, no online exhaustive search |
| P20 profiling/autotuning | offline measured per model+device+ISA+shape+driver candidate cache, cold/warm | no unbounded per-request kernel solver, persisted negative candidates |
| P22 benchmark/quality | pinned optimized vLLM/V1 + vLLM-Omni when comparable | apples-to-apples numerical quality, SLO goodput and tail latency under load |
| v5.1 serving + distribution | timed PCM/video/decisions, optional P/D and peer collectives | stage-typed QoS, safe event/clock semantics and actual transfer receipts |

Each phase remains subordinate to the existing original Qwen3.5→Gemma→Qwen3.8 practical source/hardware order. These are precise acceptance criteria, not authorization to implement huge model variants prematurely.

## 2. Quality and benchmark protocol

### Benchmark populations

1. **Text solo:** short/long prompt, short/long generation, B=1; warm and cold model/source/OS file cache, no draft.
2. **Text concurrency:** B={1,2,4,8,16,32,64} active sessions; source tokenizer/template same, stable source model; hit/miss KV pages and shared prefix.
3. **Bursty arrivals:** deterministic arrival trace with varied prompt length and staggered update/tool-like turns; offered QPS sweeps past knee until rejection/deadline miss, not just throughput under perfect saturation.
4. **Prompt mixes:** code/reasoning, prose/dialogue, retrieval/very long prefills, tools/JSON schema; actual expert routing entropy varies.
5. **MoE/low-bit:** Qwen3.8/GLM/MiMo target and source-derived quant (only when licensed/qualified), multiple draft depths and expert-row tile groups.
6. **Memory pressure:** fully resident vs legal GPU/host/SSD streaming, hot and cold experts, prefix restore from RAM/disk, long contexts; no disruptive 100GB stress on a shared interactive device without explicit permission.
7. **Multimodal:** mixed text, streaming ASR/TTS/duplex, image/video solver stages and finite decisions where actually implemented, per-class valid output + p95 deadline.
8. **Distributed P/D:** collocated mixed chunked prefill/decode vs P/D split across same-size devices and heterogeneous peers, actually measured transport/StateRoot transfer.
9. **Negative controls:** target-only, no prefix reuse, no batching, no draft, safe slow quant kernel, single-model resident CPU/GPU; retain negative/neutral evidence.

### Apples-to-apples source setup

Pin exact artifact revision and license, model/derived quant and numeric/quality contract, tokenizer chat template, rope/cache/draft state, prompt/input distribution and session scheduling, device ISA/compiler/driver, OS/cgroup/NUMA/SSD, power/thermal and model warmness. Compare similar vLLM/Omni capability/version with flags and connector/adapter differences recorded; if exact source model/format is unrecognized by competitor, label comparison non-equivalent, don't substitute another checkpoint or silently lower quality.

Repeat interleaved ABBA or BAAB runs after warmup. Report warm/cold and per-request raw events or reproducible aggregate with confidence bounds; do not claim a win if uncertainty overlaps noise or feature definitions differ. External vLLM metrics are baseline methodology, not Zerogravity results.

### Goodput and service-level availability

Text SLO-goodput uses only **committed correct output tokens** satisfying per-request TTFT and ITL bounds, divided by offered wall time, with **all offered attempts** accounted including rejected, timed out and canceled. Distinguish denominator choices and report attempted/accepted/rejected/expired/complete populations; a smaller served population is not evidence of higher service goodput.

For voice/AV/decisions use separately normalized, complete and valid task outputs inside their own deadlines, not a weighted sum of incompatible raw units. Report service-level fairness (per tenant latency/acceptance), model/capability correctness, tail under load, physical GPU/OS memory and energy. A throughput win that breaks source semantics, increases unexplained numerical error or starves other tenants is not qualified.

## 3. Source-free and actual-model acceptance: SERVOPT-01..36

| ID | Required scenario | Source-correct assertion / gate |
|---|---|---|
| SERVOPT-01 | Burst > global API queue cap | typed overload/retry; no large KV allocation, unbounded retained input or stalled server |
| SERVOPT-02 | Tenant A floods while tenant B has small interactive requests | quotas/fairness; bounded B tail and no data/cache leakage |
| SERVOPT-03 | Queued prompt tokens huge but only 2 active sessions | separate queued prefill/byte cap refuses before memory overcommit |
| SERVOPT-04 | API accepts request without physical GPU capacity | separate queued vs admitted state and honest latency/NoCapacity, not phantom Ready |
| SERVOPT-05 | 64 queued + 8 active sessions and scratch overbooking attempt | independent active sequence, row/token and physical byte admission caps |
| SERVOPT-06 | Canceled waiting request | no GPU fence; queue credits and input bytes released |
| SERVOPT-07 | Canceled submitted GPU task while other sessions active | affected physical service fenced only when unknown; other session state/Root not rewritten |
| SERVOPT-08 | Long prefill competing with 8 short decodes | legal prefill chunking protects decode ITL while prefills make eventual progress |
| SERVOPT-09 | 100 short decodes continuously arrive while long prefill waiting | aging/protected prefill progress, no permanent starvation |
| SERVOPT-10 | Source Qwen 0.8B/2B prefill B=1 and B=4 chunk mixes | exact original logits/StateRoot at same numeric mode and correct chunk boundaries |
| SERVOPT-11 | One mixed text-image input containing unsplittable image unit | no invalid half-image encoder work from token budget |
| SERVOPT-12 | Full KV + sliding KV group cache overlap | longest source-valid joint prefix frontier, not overclaimed prefix hit |
| SERVOPT-13 | GDN/KDA recurrent state absent at apparently cached KV prefix | lower reuse/recompute, no fabricated committed state |
| SERVOPT-14 | Three or more heterogeneous state groups, differing snapshot positions | generic bounded compatible frontier meet; no hardcoded only two groups |
| SERVOPT-15 | Prefix same tokens but different model quant/template/LoRA/media preprocess | cache miss or verified conversion, never wrong model state |
| SERVOPT-16 | Cross-tenant cache key hash same, ACL differs | no unauthorized state/cached private media leakage |
| SERVOPT-17 | Forked concurrent sessions reuse prefix and mutate one branch | COW exact isolation and real device page lease across queued work |
| SERVOPT-18 | State on RAM/SSD; compare restore vs recompute | measured cost-informed legal choice, same source outputs |
| SERVOPT-19 | MTP/DFlash/lookup no draft vs 0..8 proposals | target-only deterministic and stochastic sampler/stop/grammar/StateRoot parity |
| SERVOPT-20 | Prose low draft acceptance vs coding high acceptance | bounded candidate safely shrinks/disables draft when net slower, same target outputs |
| SERVOPT-21 | MoE R=15 vs R=18 rows across group-size16 candidate | cost accounts extra padded tile and expert union, no universal hardcoded 16 |
| SERVOPT-22 | One large batch + many model families on one card | physically compatible grouping only, mixed models interleave without fake GEMM merge |
| SERVOPT-23 | Different quant pack sizes QTIP K2 vs K3 with different kernel costs | no storage-bytes-only optimizer; source-derived quality separate |
| SERVOPT-24 | Fully resident model one token after warm compile | no global catalog/JSON/OS IO/RPC/readback synchronization per token |
| SERVOPT-25 | Overlap CPU tokenizer/H2D and GPU kernels under legal dependencies | real event-backed stage overlap measurable, no locks held across awaits |
| SERVOPT-26 | Captured GPU batch refers to expert slot before actual submission | slot cannot be reclaimed/changed until queued+physical reader events retire |
| SERVOPT-27 | Slow client streams output while GPU continues | bounded transport credits/backpressure, no unlimited GPU lease retention |
| SERVOPT-28 | Disaggregated prefill → decode on same source model | transfer full hybrid StateDomains with authenticated version/epoch and real Ready |
| SERVOPT-29 | Lost remote P/D state transfer while worker may run | no stale root/partial publication, correct recovery/fence and tenant guard |
| SERVOPT-30 | P/D disaggregated with WAN or slow KV transfer | benchmark can choose collocation; no unconditional throughput gain claim |
| SERVOPT-31 | 5 video jobs + 10 duplex audio sessions + text decode | per-class QoS/refusal and safe solver/frame yield; no audio starvation or fake service SLO |
| SERVOPT-32 | Worker selected vs kernel issued vs kernel actually completed | separate truthful counters, output commit only after actual event |
| SERVOPT-33 | P20 optimizer sees unknown new image shape/driver | bounded generic qualified fallback, no unbounded JIT/solver tuning in hot request |
| SERVOPT-34 | Plan candidate switch at hardware row/cache knee | repeated measured source-correct gain, bounded hysteresis and negative candidate retained |
| SERVOPT-35 | vLLM V1 text concurrency suite with same model/quant/device/runtime knobs | report acceptance, TTFT, ITL, SLO goodput, memory, error and confidence; no unverifiable win |
| SERVOPT-36 | vLLM-Omni comparable audio/video/decision tasks | typed-valid outputs/deadlines and truthfully stated unsupported modes; compare only same task/model quality |

All SERVOPT-xx cases are **future gates**, not implementation claims. Source-free scheduler unit tests cannot certify trained model/QoS; actual original checkpoint/source numerical tests must accompany GPU and benchmark evidence before qualifying a service.

## 4. Additional protocol, memory and failure variants

Each source-free state test above expands into:
- normal arrival, priority/tenant mix, duplicate req ID, cancel before/after submit, stale epoch and reconnect;
- CPU reference vs fake device and actual physical backend where authorized;
- independent KV/recurrent/encoder state subdomains, corrupted snapshot, wrong model/quant hash, near-tie numerical differences and required replay;
- prompt lengths 0/32/1K/8K/16K/32K/128K under source-qualified context limits, not assuming unsupported contexts passed;
- limited GPU memory + coherent UMA physical cache + SSD random expert miss + no-stale GPU slot, actual peak host/driver bytes;
- varied MTP acceptance 0..d and B1/2/4/5/6/8/16 with expert route histograms and optional draft assets, target-corrected source output;
- source license/privacy/tenant and provider-incompatible client events, exact API fidelity receipt;
- slow client and high GPU queue depth while physical kernel returns late Ready or unknown completion;
- regression to normal CPU Qwen3.5 Phase1 and prior 8K GDN numerical investigation remains independent.

## 5. vLLM competitive gate details

### Text
- vLLM V1 (not historic V0) scheduler config including max_num_seqs, max_num_batched_tokens, max queued reqs, chunked prefill, prefix cache and exact draft strategy where supported.
- Run same prompt sequence and concurrent arrival trace with pinned original/different derived quant (not mix); for unsupported exact model in one runtime, report as gap/capability comparison rather than apples-to-apples speed.
- Maintain target numerical sampling/grammar behavior; report draft acceptance, speculative state overhead and useful output.
- Source derived quant bytes, speed and quality all reported; no speed claim if fidelity differs silently.
- Compare disaggregated P/D only where network/topology equivalent and transfer state source-compatible; report ITL stability and total throughput separately.

### Multimodal
- vLLM-Omni model set and exact stream modes pinned; same audio codec, sample rate, image/video model, solver steps/guidance and output validity.
- Do not use generic text tokens/s to rank ASR or video; use TTFA, ASR lag/accuracy, first playable frame, synchronized AV, QoS and memory.
- If vLLM-Omni provides staged single-model inference whereas Zerogravity provides multi-model shared hosting, clearly distinguish *task orchestration* advantages from *same-model inference kernel* speed.

### Statistical promotion
- Run enough paired repeats across cold/warm and representative prompt classes to establish confidence; choose a predeclared service objective and stop criteria before experiment.
- Report all wins and regressions/neutral candidates, including real B/P95 and GPU physical headroom.
- Claim “outperforms vLLM in workload X on hardware Y at config Z” only if same model/quant/source and valid quality, with improvement outside uncertainty, no worse failure/rejection/fairness than declared acceptance threshold.
- Do not claim engine-wide superiority from a single source packed 2-bit decode or a different quantum of work.
- Unknown model/driver/feature combinations remain **UNQUALIFIED**, not default automatically optimized.

## 6. Responsibility, deliverables and change-radius constraints

| Responsible area | Inputs | Required deliverables |
|---|---|---|
| Router/API | models, tenants, capabilities, workload | bounded queue/credits, request lifecycle and typed overload with tests |
| Session/StateRoot | model state domains, snapshots, sampler | exact hybrid prefix meet, causal cache/COW, fork/replay and cancellation |
| DeviceBackend | source kernels/driver and physical loans | real batched Ready/event, source resident descriptors, no stale slot |
| TaskDriver | task-specific iteration | legal prefill/decode/codec/solver chunk boundaries and output validity |
| Planner/cost | prevalidated candidates and empirical profiles | bounded source-verified Pareto plan selection, no per-token heavy optimization |
| Resource/IO | global model+session memory and GPU/RAM/SSD | actual bytes/headroom, capacity and source identity admission |
| Distributed | peer/link capabilities, trusted state | safe P/D transfer and compute/communication overlap |
| Benchmark | exact source/corpus/runtime/hardware | pinned vLLM and vLLM-Omni parity and conditional evidence |

Introducing a new ModelRecipe, tensor codec, drafter or media task should NOT require rewriting central request parser, resource authority, StateRoot or hot-path scheduler. If it does, document measured minimality rationale before implementation.

## 7. Research sources and handoff

Governing contract: .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md.
Upstream study: .agents/prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md.
Original vLLM docs: https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/ , https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ , https://docs.vllm.ai/en/latest/design/prefix_caching/ , https://docs.vllm.ai/en/latest/features/disagg_prefill/ , https://docs.vllm.ai/en/stable/configuration/optimization/ .

This is a **design/acceptance** artifact: the user has authorized documentation and README updates, not new running serving workloads, large downloaded models, driver permissions or deployment.

````

## Source: .agents/research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md

Original UTF-8 bytes: 15110; SHA256 `dc04cf0a1fb0b4cd1dd47be8416cb3da079a39af84c5f5a69064c6e60e8e69db`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Research Track B — ASR, Speech Translation and Streaming Speech Understanding

**Date:** 15 October 2026 | **Mode:** source-grounded DESIGN/RESEARCH, not Zerogravity implementation.  
**Owner scope:** original mathematical source/processor/decoder models, stream/clock/state semantics and acceptance fixtures. Do not own transport server, GPU allocator, TaskCoordinator or model routing.  
**Read first:** .agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md; .agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md.

## 1. Verdict and central lesson

A speech-to-text task is not universally a causal text generator. **Whisper's encoder-decoder autoregression, CTC blank collapse, RNN-T streaming joiner/predictor, TDT duration transduction, source-language speech translation and joint diarization have different mathematical effects and legitimate partial-output boundaries.** They share frontend DSP, convolution/matmul/attention/normalization kernels, resource admission and typed transcript events, but require separate trusted TaskDriver algorithms. Native ASR must support both file-window and continuous streaming semantics without inventing timestamps.

## 2. Authoritative source matrix

| Family / exact variant risk | Original method and task | State/protocol consequence | Official source |
|---|---|---|---|
| Whisper (size/language/English vs multilingual vary) | log-mel encoder + cross-attending AR text decoder, language/task/time tokens, 30-second source window method | rolling windows, prompt carry, exact time offset merging, timestamp token grammar | original public model release and research |
| NVIDIA Parakeet FastConformer (CTC, RNNT, TDT vary BY CHECKPOINT) | acoustic FastConformer encoder; CTC blank collapse OR recurrent transducer predictor/joiner OR token/duration transducer | separate decoder and state schema; align/cached convolution and timestamp rules | public ASR toolkit and checkpoint matrix |
| NVIDIA Parakeet TDT 0.6B v3 | offline multilingual TDT checkpoint, not automatically cache-aware streaming | **do not advertise live streaming** simply because the FastConformer family has other streaming variants | public model-variant documentation |
| NVIDIA Canary variants | multitask encoder-decoder speech recognition and speech translation, language task slots | source-target language conditioning, translation/not just ASR, task label and capitalization | public ASR toolkit |
| NVIDIA Nemotron streaming speech variants | cache-aware FastConformer RNNT with limited lookahead, specific streaming configurations | persistent acoustic encoder/predictor cache and chunk/latency policy | public model-variant documentation |
| VibeVoice-ASR | long-form ASR with diarization and timestamping (who/when/what) | speaker identity, interval alignment, long acoustic context and structured transcript | public model card |
| VibeVoice-ASR-Streaming | incremental transcription while PCM arrives; chunk/lookahead from checkpoint's preprocessor config | emit provisional text/chunk revisions; do not hardcode arbitrary chunk length | public streaming ASR documentation |
| Qwen3-Omni Thinker | native multimodal audio/video/text understanding, MoE architecture varies by variant | synchronized input media clocks and perception state; optional Talker separate | public model documentation |

**Source caution:** Model names and checkpoints are not interchangeable. Some Parakeet models support full-utterance only; some use source-trained chunked caches. Whisper large versus tiny changes weights/performance, not the generic driver algorithm. Canary task/translation pairing is a model capability, not inferred from a request string. VibeVoice ASR format and streaming lookahead must come from its versioned preprocessor config, not a global 20ms constant.

## 3. Whisper reference execution trace

1. Accept PCM/file under size/duration/rights restrictions. Decode compressed audio via registered/permissioned codec with bounded decompression and checked channels/sample format.
2. Resample to the source-required audio rate using a registered resampler; record delay, source sample clock, exact rational sample positions and resampling remainder.
3. Compute STFT/log-mel features with exact window/hop/window function, normalization, pad/truncate/window contract. Codec/processor version is part of assembly identity.
4. Execute encoder using standard conv/attention/GEMM regions; encoder output is immutable within the window and may be cached under exact source audio feature hash.
5. Decoder starts with model-specific start/language/task/prompt header tokens and performs AR token generation with encoder cross attention; tokenizer is source-verified.
6. Timestamp token grammar, no-speech probability, temperature fallback/beam behavior, compression/segment selection and window merging follow exact upstream behavior or a named alternative task quality mode.
7. Prepare transcript segments with exact window/sample offsets and possible overlapping-word adjudication; publish committed segment only when source policy says stable.
8. Stream interim transcript if the checkpoint/TaskDriver actually supports it; an offline-only window result should not be deceptively streamed as source-native word-by-word real-time recognition.
9. Device memory includes encoder activations, decoder KV, mel buffers, source text history and bounded overlap; pending frontend buffers have their own SampleClock domain.

**Negative cases:** wrong sample rate interpreted without resampling; timestamp token treated as natural-language text; reused encoder cache with different mel preprocessing; missing language task header; late segment from stale capture epoch; bad audio frame counts before GPU allocation.

## 4. CTC source-specific execution trace

1. Capture, validate and window audio. Run source-trained Conv/FastConformer with cached/recurrent state if model supports it.
2. Produce per-frame logits with declared vocabulary including **blank** class.
3. Apply model-specific CTC decoding: collapse repeats and blanks with correct ordering. Optional beam/LM fusion changes decoding semantics and must have named task options.
4. Align label frames to exact source feature-frame and audio-sample clocks including subsampling and lookahead delays.
5. Emit provisional segment revisions while frames may revise, then publish final stable transcript segment according to decoder/checkpoint policy.
6. Validate WER, time alignment and stream chunk invariance against original source fixture. Do not write output through a causal-text decoder unless the model truly contains one.

**Counterexample:** source frame predictions [a,a,blank,a] collapse to [a,a] under CTC rules, not a GPT token stream. Blank id and padding mask semantics must not change.

## 5. RNNT and TDT source-specific trace

1. Feature frontend yields frames and validated subsampled encoder state.
2. RNN-T uses a prediction network conditioned on previous labels and a joiner combining acoustic and predictor representations.
3. Decoder can emit multiple labels per acoustic frame until blank; beam state, blank probability and recurrent prediction caches belong in provisional StateDomains.
4. TDT includes token **duration**, permitting semantic frame jumps; its duration-cursor semantics cannot be simulated with a fixed frame increment.
5. A source checkpoint may allow cache-aware streaming with bounded lookahead or may only support full-utterance inference; advertised realtime capabilities are evidence-gated.
6. Time stamps are formed from trained subsampling, duration and observed sample-clock positions, never solely from token index.
7. Mid-frame failure reverts entire proposed acoustic/predictor state; only a valid transcript frontier publishes.

**Performance candidates:** fused encoder conv+norm/GEMM, cached streaming attention, predictor/joiner scalar/tiny-batch GPU vs CPU selection, duration-aware stage batching. Fusing is legal only if blank/duration probabilities and state are preserved.

## 6. Canary ASR + AST / language conditioning

- A request must explicitly name task and source/target language constraints if required; never silently translate when caller requested verbatim transcription.
- Language prompts/task slots, tokenizer, audio segment sizes, punctuation/casing and acoustic features are source processor components.
- Translate with source checkpoint decoder where actually trained; if a separate text translation model is used, describe it as a multi-model composition with a different ModeReceipt.
- Return transcription/translation output type with language version, alignment policy and confidence provenance.

## 7. VibeVoice and diarization

- Long audio requires chunked/bounded feature residency; do not allocate an entire hour-long unbounded GPU attention matrix as an accidental general frontend.
- Speaker IDs are provisional within a recording/session and often not equivalent to universal real-world identity. Merge/renumber only through explicit diarization versioned correction.
- ASR content, word/time spans and speaker turns are distinct event streams that may require an atomic committed segment group.
- Streaming chunk/lookahead must come from each checkpoint preprocessor_config rather than a family-wide fixed number.
- A change in speaker segmentation after text has been streamed requires explicit revision events; it cannot silently rewrite a previously committed transcript.

## 8. Qwen-Omni audio input and live perception

The Thinker is a verified multimodal model component using audio and potentially video inputs. Its frontend codec, source text/audio representation and time alignment must be bound to its exact source variant. A recognition-only task can choose to omit/unload a Talker when assembly permits, without inventing speech output. If internal audio tokens and video frames are interleaved, the TaskDriver maintains the appropriate source order and typed timebases; do not collapse everything to text.

## 9. Reusable ASR contract (illustrative)

~~~rust
struct SpeechRecognitionRequest {
    source_audio: TimedPcmOrEncodedRef,
    task: TranscribeOrTranslate,
    languages: SpeechLanguageSelection,
    timestamps: TimestampGranularity,
    diarization: DiarizationPreference,
    interim: InterimPolicy,
    decoder: RegisteredDecoderConfig,
    deadline: Option<SoftDeadline>,
}
struct TranscriptSegment {
    segment: StableSegmentId,
    revision: u64,
    text: String,
    sample_range: CheckedSampleRange,
    speaker: Option<SpeakerTrackId>,
    language: Option<LanguageId>,
    confidence: Option<ConfidenceWithProvenance>,
    finality: InterimOrCommitted,
}
~~~

**Capabilities cannot be inferred from enum alone.** Exact model binding reports which decoder, translation languages, timestamps, diarization and streaming modes have source/test evidence. External recognizer service is TaskServiceAdapter, not a native ASR DeviceBackend.

## 10. State, scheduling and memory

| State domain | Source-relevant contents | Safe commit |
|---|---|---|
| AudioIngress | capture sample cursor, gaps, channels | consumed window/ack |
| Resampler/STFT | fractional residue, overlap, sample/window index | complete exact frontend window |
| EncoderCache | attention/conv cache, subsampling, mask | source-defined encoder chunk |
| DecoderBeam | RNNT predictor/joiner, TDT duration, or Whisper KV and tokenizer | validated text hypothesis version |
| Diarization | speaker cluster labels/embeddings and interval versions | stable speaker assignment |
| Transcript | segment, revision, word timestamps and language | complete declared transcript segment |

Low-latency ASR uses bounded cache-aware frames and CPU/GPU crossover cost, not maximum GEMM throughput only. Whole-file ASR can batch windows and prioritize throughput. Model/router should expose task-specific QoS and streamed/whole-file modes separately. Other ASR sessions have disjoint mutable state but can share immutable model weights and processor kernels.

## 11. Independent verification and negative fixtures

**MM-ASR-01**: mono 16kHz audio and 44.1k→16k resampled exact time mapping across many windows with residue; no drift.  
**MM-ASR-02**: Whisper 30s window boundary and valid timestamp token header, exact overlap/merge.  
**MM-ASR-03**: synthetic CTC frames with blanks/repeats, beam selection and timestamp alignment; no token AR flattening.  
**MM-ASR-04**: RNNT predictor/joiner emits multi-label frame then blank, cancellation rolls back both caches.  
**MM-ASR-05**: TDT source duration step advances correct number of acoustic frames; wrong duration schema fails.  
**MM-ASR-06**: checkpoint that disallows streaming explicitly rejects WS live request, despite family name Parakeet.  
**MM-ASR-07**: Canary source→target language/task flags correct, no silent transcript/translation change.  
**MM-ASR-08**: VibeVoice ASR streaming revised hypothesis with speaker update; immutable committed past unaffected.  
**MM-ASR-09**: interrupted input includes stereo gap, sample discontinuity and frontend overlap; reject impossible clocks.  
**MM-ASR-10**: long audio bounded input buffers; malicious decompression ratio rejected before large allocation.  
**MM-ASR-11**: ModelSession=two recordings same weights; no cache/diarization leakage.  
**MM-ASR-12**: mismatched processor/feature normalization to checkpoint rejected rather than producing plausible text.

Before marking model implemented: obtain licensed exact checkpoint, source feature extractor/decoder configuration, CPU independent original oracle, token/time WER tests, actual GPU source parity and latency. Recognition of model metadata is not execution support.

## 12. Implementation sequencing and priorities

1. Tiny source-validated log-mel/CTC mock TaskDriver and typed transcript events.
2. One real small Whisper or Parakeet checkpoint, original source feature/tokenizer/decoder verification, CPU first.
3. Add transducer variant only when RNN-T/TDT semantics have isolated test oracle; do not approximate with AR generator.
4. Add Canary multilingual translation and VibeVoice rich/long ASR with licensed code/weights and real tests.
5. Join ASR with a separately modeled LLM/TTS under trusted user application composition, then native Omni perception/duplex when verified.

Maintain one ModelDefinition, one TaskDriver per algorithm, one TaskCoordinator. No separate ASR router/allocator/server.

## 13. Primary sources and provenance

- Original Whisper implementation.
- Whisper research publication.
- Parakeet/FastConformer ASR toolkit source.
- Parakeet/FastConformer checkpoint matrix.
- Parakeet/FastConformer model-variant and streaming caveat documentation.
- VibeVoice model card.
- VibeVoice streaming ASR documentation.
- Qwen3-Omni model documentation.

Read/tested as publicly available documentation on 15 Oct 2026. **Variant-specific code/config and licenses must be rechecked** when choosing real artifacts. No tests/downloads of the listed checkpoints were performed during this design research.

````

## Source: .agents/research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md

Original UTF-8 bytes: 16022; SHA256 `850fe29f213c681bf5ff928d1058787f67acedc024e28e8e2b8a04b6565e27ba`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Research Track H — Cross-Stream Adversarial Reconciliation and Minimal Architecture Delta

**15 October 2026 · final design integration, no implementation.**  
**Inputs:** the five family research dossiers (ASR, TTS/duplex, image, video/AV, decision/ternary), the serving/API prior-art dossier, existing v4.3 core and v5 multimodal proposal, the user's universal specialization/serving requirements.

## 1. Scope and governance

This is a review of **design claims**, not verification of every source-model neural checkpoint. Official source URLs, inference algorithm evidence, mathematical identities, current release status and licenses must be re-checked per version at implementation. A sample API from an opaque provider is evidence of an API event contract, not a disclosed source model.

There is no actual autonomous multi-agent orchestration tool in this conversation environment and no separate research subagent execution was claimed. These named streams represent independent analysis and preserved task briefs. The previous 14 October workstream plan explicitly noted local Codex/Claude standalone authentication failures; we did not attempt credential workarounds.

## 2. Overarching verdict

**The v4.3 execution architecture is substantially correct; v5 predicted nearly all multimodal model type requirements. A narrow v5.1 refinement is needed for live transport/presentation frontiers, soft-realtime QoS/admission, canonical MediaEvent/API profiles and external async jobs.** Preserve one EngineCore/TaskCoordinator/PhysicalResourceArbiter/ModelRouter and native Rust-owned reference semantics. Do not turn every modality into a new inference server, fork the resource authority or make a general workflow interpreter.

## 3. Cross-stream conflict and disposition matrix

| Conflict / failure temptation | Disposition | Proof demanded |
|---|---|---|
| "Image output means latent denoising" vs an AR image-codec source | TaskDriver may run AR codec, flow/denoise or verified hybrid under one assembly | trace exact published generate_image source; refuse unknown stages |
| "Video is independent visuals + optional TTS" vs joint neural AV | coupled dual-stream model must share source solver/attention/effect frontiers | MM07/MM23 valid joint CommitGroup and perturbation negative test |
| "Every speech model can stream" vs offline Parakeet TDT variants | capability per checkpoint, decoder and implementation | unsupported streaming returns typed refusal |
| "One audio frame equals one audio token" vs RVQ codec depth and streaming neural codecs | rational acoustic frame clock × independent RVQ depth × decoded sample clock | incomplete RVQ frame never published |
| "Transcript is final as generated" vs streaming ASR revision/diarization | provisional revisioned hypotheses and stabilized segment publication | no stale or out-of-order final transcript |
| "A model response is what a user heard" vs WS playback reality | presentation ledger distinct from committed neural StateRoot | interruption truncation based on actual/unknown played watermark |
| "gpt-realtime-2.5 must exist and expose model internals" vs public API | preserve as user-supplied target name, verify actual public model catalogue, model as opaque provider | no invented local neural semantics |
| "All decisions are LLM JSON" vs hosted and encoder decision heads | dedicated finite-option TaskDriver + direct logits/typed schema | reject generated JSON as source-equivalent direct decision |
| "Qwen topology loads ternary weights" vs PTQ1/PQ2/Hadamard | orthogonal verified TensorEncoding and runtime transform, source-matched numeric reference | missing Hadamard/codec hard fails |
| "Open code means commercially redistributable weights" | independent per-checkpoint/processor/codec license and rights receipt | deny disallowed distribution/egress |
| "Vulkan GPU availability means FP32 exact numerics" vs AMD FMA | source/device/driver/precision evidence keys; do not silently widen tolerance | exact source vectors on actual device |
| "A future canceled means card can be freed" | await proven completion or quarantine submitted unknown buffers | real post-launch abort retains global fence/leases |
| "One physical epoch per session" vs shared GPU | PhysicalDeviceEpoch is card-wide, Session/TurnEpoch independent | barge-in does not cancel other session/card |
| "Global router runs per audio chunk" | bind model/task/replica once at admission; local device/native StepProgram fast loop | no JSON/catalog/physical global snapshot per audio frame |
| "External hosted video job becomes source StateRoot" | explicit TaskServiceAdapter job lifecycle; no local neural region claim | timeout after accepted POST remains Unknown |
| "We should add one generic TaskKind::Anything" | bounded trusted TaskDrivers with types/stages, not a giant dynamic graph DSL | adding model doesn't edit global router/worker |
| "One sample rate for audio" | source-specific rational clock and codec/processor version | 44.1/48k resample and 29.97FPS no drift |
| "Commit means delivered means played" | independent model commit / transport delivery / ACK / playback ledger | late ACK after barge-in cannot rewrite old generation |
| "Solver fewer steps is transparent fast kernel" | source model/quality mode/version distinct; seeded numerical policy | wrong schedule/CFG rejected |
| "GPU fit means latency target met" | assembly-wide resources plus measured physical headroom and QoS admission | mixed video/duplex stress and soft latency receipts |

## 4. Minimal delta against v4.3 + v5, not a redesign

Already correct in v4.3:
- SemanticProgram mathematical identity + prepared ExecutableBundle.
- Transient RIR/KIR and CIR embedded inside command/resource bundle; not several separately serialized dialects.
- TaskDriver stage/iteration control under one StateRoot publisher.
- RegionProvider/device execution families, shape/numeric/state compatibility.
- PhysicalResourceArbiter coarse admission and local leased fast subpools.
- built-in ModelRouter over same EngineCore, model identity/source exactness, legal distributed placement.

Already proposed in v5 (promote into future governing implementation design, do not claim implemented):
- ValueDescriptor with precise media axes and rational timebase;
- verified ModelAssembly and component ports;
- source-specific codecs and TaskDriver algorithm families;
- multitrack TaskFrontier and legal AV commit groups;
- orthogonal TensorEncoding/activation transforms and ternary low-bit case;
- explicit hosted TaskServiceAdapter separation.

**New v5.1 proposals only:**
1. RealtimeSession with independent capture/response lanes, TurnEpoch and TransportEpoch.
2. PresentationLedger tracking prepared/committed/delivered/acknowledged/played separately from StateRoot.
3. Canonical MediaEvent schema with lane/sequence/timebase/commit group/finality/payload reference.
4. WorkloadQoS for soft deadlines, fair-sharing, bounded media buffer/queue admission and typed deadline refusal.
5. ProtocolCapabilityManifest describing per-model/per-task/per-event fidelity under versioned HTTP/WS/WebRTC/SIP/async API.
6. ExternalTaskJob ledger with Unknown/reconcile/idempotency and explicit egress/credential/license costs.
7. MM-13..MM-24 mock conformance suite for barge-in, AV clocks, QoS, reconnect, hosted jobs and refusals.

**Do not create** a MediaEngine, VideoEngine, AudioEngine, BrokerService, GPUAllocator2, WorkflowRuntime or TaskRoot2. Logical helper types/modules can be introduced within current Rust crate and proven later.

## 5. Source-to-execution map

| New feature | Source/model registry delta | TaskDriver delta | Kernel/provider delta | Common engine delta |
|---|---|---|---|---|
| Whisper ASR | mel/encoder/decoder config | 30s window AR+timestamp merge | DSP/attention source kernels | none beyond typed audio/clock |
| Parakeet CTC/RNNT/TDT | feature encoder+heads | blank/predictor/duration | fused source conv/encoder/joiner | none |
| Kokoro TTS | G2P/speaker/vocoder | duration/prosody→PCM | FFT/vocoder operators | media output descriptor |
| Dual-AR RVQ TTS | 10-codebook RVQ codec | time/depth nested AR | embedding/codec/acoustic op | codebook clocks/finality |
| Full-duplex codec model | streaming neural codec and source attention | simultaneous input/output | stream codec/temporal kernels | RealtimeSession/played ledger |
| Rectified-flow image | conditioner+DiT+VAE | flow/guidance iteration | MMDiT/flow/latent VAE | typed latent result |
| Qwen-Image | variant-specific MMDiT + edit | denoise or exact source stage | image/attention/VAE | typed image result |
| AR/hybrid image | audited AR/hybrid components | checkpoint-specific AR+codec/flow | source ops | none beyond flexible driver |
| Wan2.2 video | two denoise phase experts | source noise phase switch | temporal DiT+3D VAE | video typed track |
| Asymmetric video | asymmetric text/video attention | source denoise | non-square projection, AsymmVAE | none |
| Joint AV diffusion | joint dual latents and codecs | coupled solver/AV join | audio/video cross-attn, dual VAE/vocoder | synchronized CommitGroup |
| Encoder decisions | encoder + schema head | finite choice score | source encoder/score math | decision output |
| Joint-schema decision | Qwen backbone+joint head | multi-question score | joint head | decision output |
| Ternary low-bit | packed ternary and Hadamard | Qwen-family AR | new codec/transform/kernel | no router/core rewrite |
| Hosted media providers | external provider manifest | external task status only | none | TaskServiceAdapter at task layer |

**Modularity acceptance law:** a new registered model adds only required semantic adapter/operator/driver/codec/provider, not rewrites of Router/TaskCoordinator/ResourceArbiter/network protocol core. Any exception needs a written measured change-radius justification.

## 6. Hard invariants to keep

### Single authority
Every model/task including multimodal speech/video publishes only through TaskCoordinator's epoch-checked PublishRoot. A codec/transport can prepare and deliver eligible committed buffers but cannot change neural state. Remote worker/authenticated device events are merely readiness receipts.

### Physical buffer safety
All submitted device commands carry physical card identity, numeric capability, source/model identity and real event/lease keepalives. A canceled/unknown submitted Future retains original buffers and fence, no timer-based memory reclaim. No transport cancel request can automatically "unfence" device.

### Semantics
Processor/tokenizer/codec/solver/guidance/quantization/activation transform and model specific masking are part of source semantics and quality policy; unspecified transformations cannot be silently optimized across GPU shape bands.

### Time
Audio sample, acoustic semantic frame, RVQ codebook depth, text token, solver iteration and video frame/latent clocks distinct; exact rational conversions and codec delays. Published AV groups obey both lanes' legal readiness.

### API truth
A named model can be recognized but not executable; CPU reference may be qualified while GPU source region only partial; a hosted provider can answer API but doesn't become locally loaded weights. The capability matrix must distinguish each case.

### Security
No untrusted custom checkpoint host code; no default remote egress; separately authorized voice cloning/face/rights; tenant session state isolation; sensitive media metadata redacted in logs by default.

## 7. Additional design recommendations still OPEN, not silently approved

- Which codec/media reference implementation family to start first (Whisper vs Parakeet; Kokoro vs dual-AR RVQ; source-real video checkpoint size)? The architecture can accept multiple, but practical first choice needs artifact/license/hardware feasibility.
- Should numeric-equivalence mode be strict bit-exact for every media decoder pixel/PCM sample, or define a source-verified task-specific quality tolerance? **Default reference exactness remains governing for current Qwen**; don't assume all media codecs have bit-exact cross-device outcomes. An explicit user-approved quality policy can be introduced with evidence.
- What is the default audio buffer QoS: desired mouth-to-ear P95 under a given device/load? User supplied no numeric SLA; design only requires measured configurable soft objective.
- What durable playback event retention and encryption policy is appropriate? Need privacy/cost decisions and measured memory tradeoffs before persistence.
- How to expose hosted provider cost/egress rules? Default deny until trusted authorization; no implicit provider install.
- Do multi-provider audio/video synchronous clients need one multiplexed WebRTC connection or distinct tracks/sessions? Allow both in protocol without binding EngineCore to a single transport.

## 8. Adversarial review questions to require on every implementation PR

1. Which actual source tensor/algorithm/processor proves this task?
2. Which source artifact/codec/assembly revision is immutable for a running session?
3. Which state domain is provisional? Who alone can commit it?
4. Is this a real GPU readback or only a host simulation/compiled shader?
5. Does another model on the same GPU retain its own state and fairness?
6. Does this callback use a global snapshot/lock/JSON encoder per packet/token?
7. Is the source API genuinely compatible or are parameters silently discarded?
8. Is this a native model, a trained multi-component assembly, an application pipeline or a hosted provider?
9. Do output clocks/timebase/codec and actual playback frontier agree?
10. Is a low-bit codec and activation transform mathematically required and checked?
11. Does disabling a streaming feature preserve correctness rather than output fake chunks?
12. Could cancellation/retry free an unconfirmed physical buffer or double bill a remote job?
13. Are immutable weights and codecs shared only when source hashes/legal terms match?
14. Is the new abstraction necessary, or can it be an existing TaskDriver/Provider extension?
15. Can a second LLM agent reconstruct required golden fixtures and tests from documents alone?
16. Did benchmark include end-to-end media tasks rather than isolated tokens/s or FLOPs?
17. Could an attacker supply unbounded input media, malformed codec, infinite revised transcript or arbitrary file path?
18. Was exact original device/driver/source checkpoint and capability tier recorded in a ModeReceipt?

## 9. Proposed acceptance result classification

A feature is **DESIGN SPECIFIED** when typed ports/algorithm/clocks/ownership/error and adversarial mock tests are documented. It becomes **SOURCE-MATH VERIFIED** when original model/processor/codecs and CPU reference vectors run; **PARTIAL PHYSICAL** when only a GPU region executes; **FULL PHYSICAL** when end-to-end source-equivalent model and state output execute on exact backend; **STREAMING QUALIFIED** when playable chunks, abort, memory and latency tests pass; **HOSTED API VALIDATED** when actual approved provider request/response lifecycle conforms. **No future model family is already runnable solely because a document exists.**

## 10. Reconciliation deliverables

Authoritative for agents:
- .agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md
- .agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md
- .agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md
- Independent ASR, TTS, image, video, decision/ternary, serving research dossiers (this folder).
- .agents/research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md
- .agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md
- Current [multimodal checklist](../../plans/multimodal/TODO.md) and [TODO rules](../../plans/TODO-SPEC.md); earlier aggregate bodies remain in Git history.

All choices remain subject to explicit user approval where they are recommendations rather than user directives.

````

## Source: .agents/research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md

Original UTF-8 bytes: 15661; SHA256 `76aa831a66ff7cb675f6ccf43d0bf0c131508527e5ee6cb630db93349d380094`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Research Track F — Typed Decision Models, Ternary Formats and Specialist Text Architectures

**15 October 2026 · design/research only.**  
**Models:** hosted typed-decision services, encoder decision-model variants, Qwen-derived joint-schema decision models, ternary-packed low-bit models, other source-verified low-bit/hybrid/linear attention model families. No locally executable source for the hosted services is claimed.  
**Key architectural question:** can novel task heads and tensor formats coexist with a familiar Transformer backbone without creating another engine? **Yes, if source math, task algorithm, tensor codec and runtime activation transformation remain independent and verified.**

## 1. Why decision heads need a native task kind

Decision-focused models answer a **finite question** about a state, not necessarily a natural language response. Examples: Is this transaction suspicious? Which of five tools should execute? Score whether an image matches a medical document? A model may return an option logit vector or probability for multiple schema questions in a single forward pass.

A generative text model can be prompted to produce JSON, but that is **not the same** as a dedicated trained head with calibrated yes/no/choice/score semantics. Never route a typed-decision request into ordinary language sampling and claim source-equivalent option probabilities.

## 2. Source variants and execution family

| Model | Source-verified nature | Required model/task semantics | Official source |
|---|---|---|---|
| Hosted typed-decision service | proprietary hosted decision-only model with schema questions and typed responses | external TaskServiceAdapter; source input state, choice/score/noul schema, provider version, probability meaning | provider documentation |
| Encoder decision model (English) | ModernBERT-large 421M non-AR encoder | text encoder + direct decision classification/scoring head, finite schema and text normalization | model card |
| Encoder decision model (multilingual) | mmBERT-base 322M multilingual, checkpoint context differs | language/processor, source encoder, schema output and per-variant rate/limit | model card |
| Encoder decision model (typed decisions) | ModernBERT-large with typed choice/score/yes-no tasks | multiple typed question heads or shared classifier, calibrated score only when sourced | model card |
| Joint-schema decision model (27B) | multimodal Qwen3.8-27B-derived backbone with joint schema attention/decision head | Qwen multimodal processor and joint schema scoring, many questions per forward | model card |
| Joint-schema decision model (lighter) | lighter Qwen3.5-9B-derived multimodal backbone and decision head | exact different checkpoint, source processor/schema, not interchangeable with 27B | model card |
| Ternary-packed low-bit model (27B) | Qwen3.8-derived hybrid attention topology but PTQ1_0/PQ2_0 ternary weights with required runtime Hadamard transform | source register for codec + activation transform + accumulation, familiar Qwen semantic backbone | model card |

**Source caution:** The joint-schema source head layout and specific foundation variants must be bound to their exact weight config; do not infer every Qwen checkpoint automatically has a joint decision head. The hosted typed-decision service is opaque and absent released executable model weights; do not import a guessed neural formula. The encoder decision-model variants are distinct registered checkpoints, not simply one model with a language flag.

## 3. DecisionSchema model

~~~rust
struct DecisionTask {
    source_state: TypedStateRef,                 // text/media/structured context
    schema: VerifiedDecisionSchema,              // stable question/choice IDs
    output_policy: RawLogitsOrCalibrated,
    model: ResolvedModelBinding,                  // exact source/processor/semantic revision
}
struct QuestionSpec {
    id: QuestionId,
    semantic_prompt: BoundedText,
    answer: AnswerDomain,                        // Noul, finite choices, scalar score
}
enum AnswerDomain {
    Noul,                                        // source-defined yes/no probability semantics
    Choice(Vec<ChoiceId>),                       // choices ordered, source schema bound
    Score { minimum: f32, maximum: f32 },       // scale+direction defined by source
}
struct QuestionResult {
    question_id: QuestionId, schema_version: SchemaId,
    raw_logits: Option<Vec<f32>>,
    scores: Vec<ScoreWithProvenance>,
    calibration: Option<CalibrationId>,
    finality: DecisionFinality,
}
~~~

**Schema invariants:**
- Question IDs, answer option IDs and option order are versioned; sorting choices alphabetically may change model output.
- Raw logits are distinct from softmax probabilities and temperature/calibrated probabilities.
- Calibration metadata may include source calibration dataset, validation temperature/isotonic/logistic link, thresholds and task domain; absence means "not calibrated", never confidence=0.9 invented.
- A score may not be a probability; preserve units, range, direction and trained head source.
- 20 questions can be jointly processed only if checkpoint supports joint schema; otherwise register a bounded batch of separate score tasks.
- Nonfinite logits, invalid option count or missing schema answer must reject publication before StateRoot commits.

## 4. Encoder decision source trace

1. Bind exact variant (English, multilingual or typed decisions), tokenizer, language/profile and allowed schema.
2. Normalize bounded textual context and question descriptions per source processor; no arbitrary text truncation if it changes answer.
3. Execute ModernBERT/mmBERT encoder through shared attention+norm+GEMM providers.
4. Apply checkpoint-trained direct scoring or classification heads (not a causal token sampler), respecting question representations and answer type.
5. Produce option logits/score distribution and apply only source-validated calibration/normalization.
6. Validate schema cardinality and output value ranges, then atomic TaskCoordinator publish the decision result.
7. Benchmark batch sizes/memory, latency and actual per-domain accuracy. Device placement can be CPU/accelerator independently of model type.

**Negative:** no decoder LM available → no need to fabricate BOS/EOS text or KV decode loop. Router may still select between encoder decision models if caller authorized quality/language policy, but must report actual variant.

## 5. Joint-schema decision head trace

1. Bind Qwen-derived multimodal backbone+vision processor and the source-trained joint schema head as verified ModelAssembly.
2. Validate input text/image roles and finite schema (questions, options, scoring semantics).
3. Encode/source-pass multimodal context using checkpoint-correct attention/recurrent state.
4. Feed shared representations and all requested bounded schema questions to joint scoring head within one forward where source-trained.
5. Obtain raw per-option logits, apply source-specified normalization, calibration only if exact head supports it.
6. TaskCoordinator publishes complete question/answer schema atomically; output is a typed decision record rather than free-text JSON.
7. Share immutable foundation weights where absolutely identical to a local Qwen model, but separate joint-head/processor semantics and per-session state. Never assume plain Qwen3.8-27B results equal the joint-schema model.
8. Return device/model/mode receipt and schema revision for audit/replay.

**Specialization:** joint question scoring and medium/high batch matrix ops; source sparse/multimodal attentions may use existing Qwen providers if equivalence proven. A new decision head is a semantic op/reference plus model component, not an independent LLM inference server.

## 6. Hosted typed decisions

Current hosted service official description: state plus typed questions; source response may include probability/choice/score with provider model alias/version. API requires server-held bearer credentials and task-specific choice schema. Treat this as **opaque hosted TaskServiceAdapter**:
- consent for source state/image/doc egress and geography;
- model alias revision uncertainty; pin version when provider supports it;
- provider-calibration semantics only as officially promised;
- token/usage/cost, retries and tenant limits;
- typed response schema validation and actual service errors, rather than missing tensor logits invented locally;
- never mix local GPU source evidence with hosted API measurement.

A typed hosted decision and a local encoder decision model may share a public decision request/response schema where semantics truly align, but if one returns only a probability and the other returns raw logits, preserve that distinction.

## 7. Ternary low-bit source architecture: why simple Qwen format recognition is dangerous

The ternary low-bit model's source card describes 27.36B Qwen3.8-derived topology, about 75% linear attention / 25% full attention, with ternary packed weights in PTQ1_0 or PQ2_0; matching blockwise Hadamard activation transform is essential. Stock loaders can mishandle a superficially similar low-bit format and silently produce garbage without an error.

- Ternary values in {-1,0,+1}, g128 per-group scaling; codec forms have different bit packing and speed/memory tradeoffs.
- The source card states 1024-value block Hadamard rotation with fixed sign scheme folded into stored weights; a **matching transform MUST be applied to runtime activations** before dot.
- Codec-specific group scales/packing/unpacking, zero padding, signed representation and exact numerical accumulation are independently versioned.
- Vision encoder can be separate Q8_0 mmproj component; do not load/cache it for text-only tasks unless source assembly requires it.
- Source GGUF tensor type PTQ1_0/PQ2_0 must fail gracefully on generic stock Q4/Q2 kernels. A plausible-looking output is not an acceptable test.

This is a first-class proof of four orthogonal axes: **same semantic model topology; novel source physical codec; novel activation basis transformation; independent kernel backend specialization**.

## 8. Ternary TensorEncoding design

~~~rust
struct TernaryCodec {
    ternary_packing: TernaryPack,               // PTQ1_0 / PQ2_0 exact source
    group_size: u32,                           // source declared 128
    scale_type: ScaleEncoding,
    basis: HadamardBasisId,                    // source block 1024 and sign scheme
    activation_transform: VerifiedTransformId, // mandatory at runtime
    accumulation: NumericReductionContract,
}
struct QuantizedMatvecRegion {
    semantic_weight_id: TensorIdentity,
    encoded_weights: VerifiedByteSpan,
    input_activation: LogicalTensorRef,
    codec: VerifiedCodecId,
    transform: VerifiedTransformId,
    legal_kernel: SourceQualifiedKernelId,
}
~~~

Reference math:
- parse source weight blocks, scales and transform metadata with checked sizes and source license/version;
- evaluate original ternary + activation transform semantics in small scalar reference; verify independent golden vectors;
- on physical CPU/GPU kernels, match exact source numeric/quality contract and quantizer outputs; never omit Hadamard or convert it into an unqualified approximation;
- optimize direct bitpacked GEMV, activation Hadamard fusion, tensor packing and placements without expanding entire weights unless memory/perf justified;
- support optional separate vision component under exact processor identity and media port.

## 9. Extensibility to other specialist text models

Although the ternary low-bit model is a concrete adversarial target, the same architecture must accept:
- BitNet-style ternary/1.x bitlinear ops with source activation normalization;
- mixed FP4/NVFP4 weights with per-block/per-channel scales;
- linear/DeltaNet/KDA/GDN recurrent states distinct from Transformer KV;
- learned indexers/compression/non-token memory state (Qwen flash/GLM/DeepSeek-like variants);
- LoRA/adapter/rank-specific deltas with exact model/source/key identity;
- multimodal Qwen-derived variants with vision/acoustic processors;
- block-sparse experts and dynamic output projection where numerical identity preserved.

New semantics get a registered SemanticOp and ReferenceOp, new packed formats get a VerifiedCodec, new state gets a StateDomain; router, TaskCoordinator, server and physical arbiter remain unchanged.

## 10. Negative test matrix

**MM-DEC-01:** Encoder decision model 3 questions via encoder, no text sampling; exact option labels/order.  
**MM-DEC-02:** Joint-schema 20-question head, same source image/text representation and typed schema, atomic result.  
**MM-DEC-03:** option logits [-100,100] with stable normalized softmax; no NaN probability; calibration absent remains absent.  
**MM-DEC-04:** model lacking true probabilistic calibration cannot claim reliable confidence, even after softmax.  
**MM-DEC-05:** edited question option set after task start invalidates running schema; stale TaskRoot refused.  
**MM-DEC-06:** hosted typed-decision service unauthorized egress refused before API call; timeout job status Unknown instead of fabricated answer.  
**MM-BON-01:** source packed PTQ1_0/PQ2_0 blocks decode against golden reference and exact scales.  
**MM-BON-02:** missing Hadamard transform fails at model load/plan, not after poor generated tokens.  
**MM-BON-03:** wrong Hadamard block width/sign scheme fails source golden vector.  
**MM-BON-04:** generic Qwen architecture can share attention semantic graph without reusing wrong tensor codec.  
**MM-BON-05:** CPU scalar reference vs native bitpacked SIMD/GPU at multiple original source-trained vectors and all shape guards.  
**MM-BON-06:** two concurrent text sessions share immutable packed source weights but not recurrent/grammar/RNG state.  
**MM-BON-07:** vision tower absent text-only works where source permits; image input without verified vision mmproj refuses.  
**MM-BON-08:** memory reporting accounts exact compressed bytes AND Hadamard transform scratch, not just disk size.

## 11. Qualification and implementation order

1. Mock typed decision schemas and direct-logit normalization with source-free deterministic fixtures.
2. One small source-verified encoder+decision head CPU model, then source-qualified GPU; use no new serving engine.
3. Exact joint-schema model assembly with Qwen head integration and source multimodal state; verify original trained checkpoint on licensed hardware.
4. Ternary PTQ1/PQ2 source parsers, Hadamard/reference scalar kernel and source-golden vectors; then specialized devices/fast path.
5. Hosted service adapter only when user/admin explicitly connects account, grants egress/consent and job schema; no arbitrary external provider calls from native GPU.
6. Other low-bit text models follow same codec/transform/semantic policy.

Existing user-mandated practical Qwen3.5→Gemma4→Qwen3.8 source CPU/GPU order is preserved; this is a later extension plan and architecture acceptance sequence.

## 12. Original sources reviewed

- Hosted typed-decision service introduction.
- Hosted typed-decision service API quickstart.
- Encoder decision model source/config.
- Joint-schema decision model source card (27B).
- Joint-schema decision model source card (lighter variant).
- Joint-schema implementation reference.
- Ternary-packed low-bit model source card.
- Ternary-packed model demonstration source.

Source claims reviewed on 15 Oct 2026, subject to exact checkpoint/license/version verification. No decision or ternary low-bit model was run or downloaded in Zerogravity during this research.

````

## Source: .agents/research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md

Original UTF-8 bytes: 17669; SHA256 `c80956cefff5b9b18dc10e2eccfc80eae0a17bd849aafde05d3f3fa57edbbdb3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Research Track D — Image Generation, Editing, Flow/Diffusion and Autoregressive Image Models

**15 October 2026 · Design/source research, not runnable model support.**  
**Scope:** Qwen-Image, rectified-flow image variants (large dev and small distilled/base), AR/hybrid image models, multimodal DiT latent-diffusion models and newer/future frontier image providers. Source variants are not interchangeable.  
**Architecture authority:** Unified v5.1 model assembly, source-legal TaskDriver iteration, typed ValueDescriptor and single TaskCoordinator.

## 1. Verdict

An image model is **not** equivalent to "input text, iterate diffusion and return JPEG." Contemporary open architectures include MMDiT diffusion/flow, rectified flow with variant-specific LLM text conditioners, image edit reference branches, causal/multimodal autoregressive image/codebook generation and potentially mixed-stage reconstruction. The reusable engine must support semantic latent tensors, source-registered samplers and complete decoded images, without a separate image inference runtime.

The most important insight: **flow schedule, denoising prediction target, seed/RNG, guidance and conditioner/decoder identities are source-model semantics**. Changing step count or text encoder to make a smaller GPU run faster may alter output distribution, not constitute an equivalent physical kernel.

## 2. Source model matrix and component inventory

| Family and variant | Verified public architecture characteristics | Assembly and exact-variant risks | Official sources |
|---|---|---|---|
| Qwen-Image original (~20B MMDiT) | image generation/editing with multimodal diffusion transformer and text conditioning, strong rendered text | image editing image-reference processor and latent VAE, MMDiT exact sequence, Qwen family source tensors | model documentation |
| Qwen-Image later 2.0 variants | upstream release adds new image model/configuration variants | do NOT assume original 20B checkpoint attention, processor or numeric schedule apply unchanged | model documentation |
| Rectified-flow image model (large dev variant, 32B) | rectified-flow multimodal image generator/editing, Mistral-class text conditioner, latent generator and VAE | large conditioner residency and reference images; non-commercial weight license; source step rules | model documentation |
| Rectified-flow image model (small distilled/base variants, 4B/9B) | different conditioner based on Qwen3 embedding, distilled 4-step vs base many-step recipes; variant licensing | Do not pair the [dev] Mistral encoder with the distilled variant's Qwen embedding, do not silently treat distillation as equivalent solver | model documentation |
| Multimodal DiT latent-diffusion model (large/medium/turbo variants) | multimodal DiT with multiple text encoders (CLIP-L/OpenCLIP-G/T5-XXL), dedicated VAE latent decoder; version-distinct ControlNet | exact text encoders, QK/norm and timestep/CFG semantics, 16-channel VAE path, source license for weights | model documentation |
| Unified multimodal AR MoE image model | upstream documentation states unified multimodal AR MoE; source generation graph is checkpoint/branch specific | needs exact source audit; don't hardcode either pure AR image-codec or generic diffusion stages | model documentation |
| Hosted GPT-Image and other hosted image providers | opaque provider generation/editing APIs | TaskServiceAdapter with exact input asset/rights/usage/cost/async policy, not native regions | provider documentation |

**Release and license verification required:** The Qwen-Image source repository includes multiple generations; the small rectified-flow variants may use different terms from the larger and dev variants; source code availability ≠ commercially redistributable image weights. The AR/hybrid image model has divergent discussions of its AR image generation pipeline; inspect exact generate_image call tree for the **selected checkpoint revision** before freezing source semantics.

## 3. Verified image component assembly contract

Typical flow/diffusion assembly:
~~~text
Source input text (+ mask/reference images)
  -> validated tokenizer / text conditioner(s) (CLIP / T5 / Mistral / Qwen family)
  -> conditioned embeddings (+ image reference encoding / attention masks)
  -> sampled initial latent + source RNG/seed
  -> source timestep/sigma schedule, solver integrator and guidance branch(es)
  -> MMDiT / DiT / flow predictor repeated at validated stages
  -> final semantically valid latent state
  -> checkpoint-bound VAE decoder / image processing
  -> pixels with explicit color primaries, channels and output dimensions
  -> encode image packet/container if requested
  -> StateRoot single publication and output receipt
~~~

**No automatic architectural assumption**: for an AR image/codebook model, replace the solver loop with the source-verified image token/codec loop and decoder. For a hybrid AR→latent predictor, the assembly has both semantic stages. TaskDriver chooses the correct algorithm, not the endpoint named images/generations.

Version and hash required for every node: checkpoint, encoder processor/tokenizer(s), VAE, scheduler, optional edit encoder, LoRA/ControlNet, prompt optimizer and any external host dependencies. Allowed type edges may include pixel→VAE latent or raw text→tokenized embed; dimension match alone does NOT mean compatible.

## 4. Latent tensor and image value semantics

~~~rust
struct ImageValue {
    width: u32, height: u32, channels: ChannelLayout,
    component_type: PixelType, color_space: ColorSpaceId,
    alpha: AlphaMode, orientation: Orientation,
    data: PixelBufferLease,
}
struct ImageLatent {
    batch: u32, channels: u32, latent_width: u32, latent_height: u32,
    patch_layout: RegisteredLatentLayout, vae: VerifiedDecoderId,
    source_rng: RngStateId, iteration: SolverCursor,
}
~~~

Image tensor shapes need bounded element count, alignment, stride, patch packing, masks/crops and safe decoded-byte checks before device allocation. Image edits may have one or more reference images with different dimensions and region masks. Pixel color space, ICC, gamma and alpha cannot be silently discarded when decoding from latent.

A latent is **not** final output; never stream partial denoising state as a finished image. Progressive previews are allowed only when client explicitly opts into provisional preview semantics and model TaskDriver can construct them without leaking uncommitted neural state.

## 5. Flow and denoising semantics must be exact

For each checkpoint store:
- prediction target: velocity / epsilon / x0 or registered alternative;
- timestep/sigma schedule and indexing; exact start/end and interpolation;
- integrator (Euler, Heun, DPM family or source schedule), update order and required precision;
- noise RNG algorithm, counter/key, seed, shape, negative-conditioning and guidance branches;
- CFG weight and any "guidance distillation" convention; text/image/reference conditioning masks;
- learned embeddings and positional/rotary coordinate semantics;
- VAE latent scaling/shifting and source decoder/crop;
- whether step distillation/turbo supports alternate step count and under which quality mode.

**Wrong but plausible image is a failure.** A GPU candidate that uses Euler where original reference uses Heun, flips noise direction, omits guidance, changes VAE scaling or silently truncates a reference image must fail source qualification, even if visually acceptable.

A source TaskDriver may run many solver steps in a prepared device-native region or graph, then return prepared latent version and actual GPU completion. TaskCoordinator alone publishes root and final image. When a request is canceled midway, device kernel completes or quarantines; previous committed image/state remains uncorrupted.

## 6. Qwen-Image source-specific trace

1. Validate exact Qwen-Image variant/release, 20B original versus later Qwen-Image2.0 family.
2. Load source-bound text processor/conditioner and image editing reference processor if applicable.
3. Encode prompt preserving text layout/casing/Unicode direction needed for typography.
4. Prepare source latent representation, timestep/guidance and RNG state.
5. Execute MMDiT with correct cross-modal attention, normalization, positional embeddings and input/reference masks.
6. Run exact diffusion/flow source scheduler; after each full legal iteration, mark provisional latent state with a monotonically checked solver cursor.
7. Decode through matching VAE; source output dimensions/color and trim only by declared processor.
8. Publish one final image and source/numeric evidence; optional previews tagged provisional and never conflated with final.
9. For editing, changes to reference image or mask invalidate related source conditioning cache keys; no reuse of incompatible embeddings.

**Negative:** wrong image processor/text layout, mismatched VAE, truncated reference array, changed guidance schedule, incorrect timestep ordered expert/head.

## 7. Rectified-flow image source-specific trace

**Large dev variant, 32B**: source-backed flow transformer with a Mistral-class conditioner and substantial memory requirement. Text/image reference processing and prompt upsampling must be **explicit**. A remote conditioner can be a separately authorized stage, not covert egress.

**Small distilled/base variants, 4B/9B**: conditioner based on Qwen3 embeddings and distinct model sizes. Distilled 4-step variants and base ~50-step variants change training/sampler assumptions. Must pin exact Qwen3 embedding revision and dimension; using the large variant's encoder is an ABI error, not acceptable fallback.

Common trace:
1. Resolve source manifest → variant-specific VerifiedModelAssembly.
2. Encode prompt and references; retain original conditioning embeddings with verified source/processor keys.
3. Seed noise under allowed random policy; bind latents to source VAE geometry.
4. Predict flow velocity with source DiT and timestep conditioning, apply registered integrator and any source guidance.
5. Decode/crop/format final image and publish. Cache warmness/residency across requests must not leak user reference images.
6. Track conditioner GPU residency independently of denoiser to support offload and device placement; admission accounts the whole assembly.

**Falsification:** selecting the image model by name alone and injecting the [dev] Mistral embedding into the distilled variant (or vice versa) must refuse before any kernel. Distilled/base scheduler cannot silently be swapped because same tensor hidden size.

## 8. Multimodal DiT latent-diffusion source-specific trace

The original reference implementation identifies CLIP-L/14, OpenCLIP bigG, T5-XXL, 16-channel VAE and MM-DiT core. Not every deployment keeps every encoder resident or uses identical prompt pooling; source variant manifest must identify required components.

- Distinguish Large, Medium and Large-Turbo weight/semantic/numeric configurations.
- Text encoders are independent verified models with their own tokenizer, mask, pooling and output axes; combine exactly per source.
- VAE decode latent normalization, image channels, output cropping and color range are source-defined.
- QK normalization and attention-layer variants matter to mathematical identity; a generic Transformer block must not silently omit them.
- ControlNets and adapters are source-versioned components, not arbitrary tensor patches; their injection paths and residual scales are part of assembly state.
- Turbo fewer-step outputs are distinct quality model modes; do not market as equivalent fused implementation.

## 9. AR/hybrid image model ambiguity protocol

The upstream README describes AR multimodal generation with MoE. However general high-level media architecture descriptions are insufficient to decide the exact released \`generate_image\` pipeline for all variants.

Before native implementation:
1. Pin exact branch/commit, weight files, variants (Instruct/distilled/source).
2. Trace official entrypoint into model forward, image special token/codebook decoding and any extra denoise/flow/refiner or reconstruction.
3. Capture source equations and necessary processors for each stage.
4. Determine exactly which operations are AR, which are one-shot decoder, which are iterative solver; document mandatory RNG/state effects.
5. Register a ModelAssembly and TaskDriver matching that trace. **Never** force a global DiT solver or pure decoder AR based only on branding.
6. Verify synthetic source-specific shape/time/codec tests followed by original checkpoint tests. Until complete mark algorithm details Unknown/UnderAudit rather than inventing.

This model is an adversarial architectural proof that one coherent fabric can support distinct algorithms under one request without a new server.

## 10. Image editing and composability

Potential components: prompt language understanding, mask segmentation, text/vision encoders, image-to-latent, reference-conditioned generator, inpaint/outpaint attention/mask, VAE and optional super-resolution/upscaler. One task may combine these under a verified original assembly or app-level workflow depending on source. An external "remove background then generate" is NOT silently equivalent to a trained joint editing model.

Typed edits should include:
- input image(s) count, required ordering and roles;
- dimensions, crop coordinates, alpha/mask semantics, orientation and color space;
- allowed transformation, output dimensions/aspect and source step/guidance;
- whether input content leaves device/local machine;
- output rights and provider receipt for hosted APIs.

## 11. Physical memory and performance strategy

Cost dimensions: conditioner weights, denoiser weights, VAE weights, reference encoder, source latent+CFG doubling, attention scratch, tile reconstruction and output pixels. A 20B/32B text+image model cannot fit solely because the DiT weights fit; the complete assembly must be budgeted. Discrete/UMA storage and offload should be planned at stage boundaries.

Potential optimizations, all subject to proof:
- Independently resident conditioner shared by exact source manifest, memoized on permissible public prompt/cache key.
- Device-native iterative steps in compiled bundle to reduce host per-step scheduling while retaining typed solver state.
- VAE tiling with verified overlap/padding and deterministic output comparison.
- Tensor/sequence parallel large MMDiT with correct cross-device ordered dependencies.
- Quantized checkpoint original encoding and source numerical policy; not silent "use a cheaper quant".
- Long-running batch job QoS with video/voice tasks coexisting via legal scheduling step boundaries.
- Explicit warm/cold performance and peak physical resource reporting.

## 12. Required image acceptance

**MM-IMG-01:** Qwen-Image original vs later source variant require distinct manifests; mixed processors fail.  
**MM-IMG-02:** [dev] Mistral and distilled-variant Qwen3 conditioner mismatch refuses at load.  
**MM-IMG-03:** multimodal DiT multiple text encoders with different shape/mask; skip required encoder hard fails.  
**MM-IMG-04:** exact solver/CFG/seed/step count changes reproduce expected source reference; different schedule reported as distinct quality mode.  
**MM-IMG-05:** incorrect VAE crop/scaling produces error before final image; no unverified preview publication.  
**MM-IMG-06:** inpaint input mask/alpha/ICC survives image processor semantics, doesn't leak ignored/reference frames.  
**MM-IMG-07:** whole ModelAssembly physical memory admission; conditioner/latents/VAE all charged, never double-admitted.  
**MM-IMG-08:** dropped GPU Future mid-solver fences physical handles without stale StateRoot, retains last valid image.  
**MM-IMG-09:** parallel images from two users share immutable conditioner weights but not latent noise/RNG/private references.  
**MM-IMG-10:** AR/hybrid image model exact generate_image graph audited before labeling AR/Diffusion support.  
**MM-IMG-11:** output PNG/JPEG/WebP color/orientation and alpha validated against pixel type.  
**MM-IMG-12:** unsupported external hosted "frontier image" source not masquerading as local inference; egress consent enforced.

## 13. Recommended implementation order

1. Source-free validated conditioner→flow predictor→VAE mock assembly with source-bound RNG/solver.
2. Small verifiable open flow model checkpoint with original CPU reference and GPU-tiling correctness; candidate small distilled rectified-flow checkpoint if license/hardware permits.
3. Multimodal DiT latent-diffusion and Qwen-Image variant adapters using reusable ops while preserving their distinct source processors.
4. Editing/multi-reference and source-specific ControlNet/LoRA components.
5. The AR/hybrid image model's exact audited AR/hybrid recipe and optimizer proof, not before source semantic clarification.
6. Hosted frontier providers through TaskServiceAdapter independently from native image math.

The existing user CPU→GPU Qwen3.5/Gemma/Qwen3.8 implementation priority is NOT automatically reordered by this research roadmap.

## 14. Primary sources checked

- Qwen-Image model documentation.
- Rectified-flow image model documentation and license.
- Multimodal DiT latent-diffusion source.
- AR/hybrid image model documentation.
- Hosted image provider API documentation.
- OpenAI model/API catalog: https://developers.openai.com/api/docs/models

Research gathered and reconciled 15 Oct 2026. No checkpoint weights downloaded/tested, no license inferred from GitHub repo existence, no native image support implemented.

````

## Source: .agents/research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md

Original UTF-8 bytes: 17156; SHA256 `c0514c341b64d20d07ad610945ba89d3fd4c32c8dd10eb159dc92e9de3355747`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Research Track G — Real-Time and Multimodal Serving / API Prior Art and Design Reconciliation

**15 October 2026; source/architecture research, no Zerogravity network/API execution.**  
**Focus:** OpenAI Realtime/Responses/audio/image/video, hosted live multimodal APIs, local vLLM/SGLang patterns, WebRTC/WS/SIP, hosted voice/media providers and provider job adapters.  
**Governing proposed spec:** .agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md.

## 1. Executive verdict

A high-performance universal inference engine must not expose every task solely through /v1/chat/completions. Text generation, asynchronous image/video, file ASR, streaming TTS, native duplex speech and typed decisions have different request, stream, cancellation, finality, cost and state lifecycles.

The right implementation is a **canonical TaskRequest and typed MediaEvent** beneath multiple outward protocol adapters, connected to one ModelRouter/TaskCoordinator/ResourceArbiter. Each adapter translates only verified observable semantics, never creates its own model engine.

## 2. OpenAI public realtime model/protocol reality

At review time, OpenAI's public API model catalog (https://developers.openai.com/api/docs/models) lists GPT-Realtime-2.1, GPT-Live and additional realtime/transcription models. The user supplied example "gpt-realtime-2.5" could not be independently verified as an official published model ID; treat as desired future/realtime capability class, **not** a known architecture or immutable API name.

Official realtime docs: https://developers.openai.com/api/docs/guides/realtime-conversations
- WebRTC and SIP flows may have a server-side playback/output buffer whose interrupted audio can be truncated automatically.
- WebSocket clients manage their own audio playback; they must stop it and send an explicit conversation.item.truncate with actual played duration to reconcile the conversation.
- With VAD, speech_start may cancel current response; client should not confuse generated audio, sent audio and actually heard audio.
- A canceled response can still have earlier committed text/audio, and tools may need independent cancellation/late result policy.
- Session and response events have explicit event IDs, item/content indices and generation boundary; these cannot be represented as raw token strings.

Official client-event schema: https://developers.openai.com/api/reference/resources/realtime/client-events
- conversation.item.truncate is an observable protocol contract, NOT an instruction for GPU DeviceBackend to roll back physical DMA.
- Truncation of unplayed speech transcript is a **presentation/history** policy and should not physically pretend a user heard earlier generated words.
- Unsupported events or fields must result in explicit capability refusal, rather than fabricated transparent compatibility.

**Lesson:** One session needs neural committed root AND transport/presentation playback ledger, plus source-defined state replay for barge-in.

## 3. Hosted live multimodal API reference

Provider documentation:
- Hosted live API capability documentation.
- Hosted live API session management documentation.
- Hosted live API best-practices documentation.
- Hosted live API stream reference.

A public hosted live API streams audio/video/text on a persistent WebSocket, advertises VAD interruption, session resumption handles, periodic connection GoAway and context compression. The public documentation reviewed 2026-10-15 illustrates a hosted live model variant; exact model IDs, session limits and capabilities vary by date/model and must be obtained from provider metadata rather than compiled into Zerogravity.

Live media behavior highlights:
- Audio chunks can be 16kHz PCM inbound, 24kHz PCM outbound depending on selected model/API; do NOT hardcode as universal media clocks.
- Video capture frame rate may be throttled/model-limited, distinct from the model's semantic video-processing clock.
- Asynchronous function calls may be permitted by some models and disallowed/serialized by others; model capability manifests must distinguish this.
- Session resumption and content compression preserve conversation under certain provider rules, but not guaranteed lossless neural local StateRoot migration.
- Multiple modality streams have independent ordering; timestamp and causal ordering must be explicitly handled.

**Lesson:** A protocol adapter can implement hosted-live-API-like session semantics over canonical MediaEvents without altering the native TaskDriver or TaskCoordinator.

## 4. Native streaming audio prior art

An independent full-duplex audio project provides native full-duplex audio and a streaming neural codec. It has source Python-framework code, an Apple-silicon framework and a Rust implementation, demonstrating that audio decoding and playback often require a strongly typed codec frame/time state, not conventional next-token output. A correct native backend must keep simultaneous input and output lanes and codec state. Its Rust source is useful prior art, not a mandate to embed a second independent model-serving runtime in Zerogravity.

**Lesson:** Real-time endpoint architecture must support source-native duplex models **and** composed Whisper→LLM→TTS under the same session transport.

## 5. Hosted voice streaming

A hosted WebSocket TTS input can accumulate text and use chunk_length_schedule and flush; those settings trade text buffering/latency against output. Some provider behavior responds to end-of-turn and inactivity; use provider adapter version, not universal client assumption. Word-level alignment may be available for certain outputs. The API provider is opaque; source model architecture and native GPU execution cannot be inferred from WebSocket frame shape.

**Lesson:** A TaskServiceAdapter must map provider-specific text flush, generated audio chunks, voice settings and stop/usage events into canonical committed/delivery events with receipt. Remote cancellation may be best-effort; never assume dropped socket means billable job stopped.

## 6. Hosted long-running media jobs

A hosted video generation service uses a long-running generate_videos operation and explicit polling until done, then asset download. Hosted async image/generative APIs return a job ID/status URL and later asset URLs; cancellation and rate handling exist as separate stages.

**Serving implication:**
- JobStarted is not ImageCommitted or VideoPlayable.
- A network 504 after POST can mean remote service accepted job; status Unknown must be reconciled with provider operation/idempotency key before reissuing potentially billable work.
- Output assets need bounded signed download, MIME/size/codec integrity checks, license and expiry tracking.
- Remote provenance includes provider model alias/version and charged cost if reported; no local GPU mode receipt.
- User-required rights/content credentials should be represented where actually provided, not fabricated.

## 7. vLLM and SGLang useful patterns and caveats

Relevant public docs (version-specific):
- vLLM documentation: https://docs.vllm.ai/
- SGLang documentation: https://docs.sglang.ai/
- llama.cpp HTTP serving/prior art: https://github.com/ggml-org/llama.cpp

Research uses them for architectural principles rather than claiming Zerogravity's feature parity:
- continuous batching and paged state management, prefix caching, model-aware scheduling;
- adapters/processor pipelines for multimodal image/audio inputs;
- source-specific compile/cache policies and asynchronous local HTTP streaming;
- graph capture/fused-kernel fast path while keeping model/session state separate;
- distinctions between inference engine capability and HTTP API compatibility.

**What not to copy:** hidden Python engine wrappers, per-model HTTP subprocess, mandatory user flags for every kernel, one global GPU lock per PCM packet, blindly interpreted model blobs, conflation of text token streaming with playable audio. "vLLM supports this model" does NOT prove it exposes full-duplex speech through its standard endpoint.

## 8. Canonical API compatibility and behavior matrix

| Capability | Native model runner needs | External API needs | Failure mode to test |
|---|---|---|---|
| Text/completions | tokenizer/template, exact samples/grammar state | streaming SSE/HTTP, source model alias | unsupported tools/JSON silently ignored |
| Embeddings/rerank | source encoder+pool normalization | vector size, batch schema, errors | generic decoder logits used as embeddings |
| File STT | source frontend+decoder+clock/diarization | files, timestamps, translation langs | partial text mislabeled final |
| Realtime STT | cached encoder/transducer, VAD, interim/final | WS/WebRTC, gap/revision events | out-of-order segments, fake latency |
| TTS | acoustic+vocoder and complete frame | binary chunk/format, voice ID, playback status | incomplete RVQ group emitted as playable |
| Duplex voice | independent capture/generation states | live session handshake, barge-in, ACK | output cancel corrupts input/other session |
| Images/edit | verified predictor/solver/VAE | async/sync, input image masks, output URI | invalid mask/color/schedule silently changed |
| Video/AV | temporal latents, 3D VAE/codec, AV sync | async job status, assets, progress | raw incomplete latents delivered as final |
| Decisions | schema-conditioned head raw logits | typed choice/noul/score | made-up calibrated probability |
| Hosted provider | NONE if no weights/source | consented TaskServiceAdapter | vendor host mistaken for local GPU |

**Compatibility receipt:** protocol/profile revision, actual task mode, model artifact or remote provider identity, requested and effective parameters, input/output modalities, supported stream events, error policy, numerical/quality source evidence, latency-qualified mode (if any).

## 9. Transport-specific lifecycle templates

### HTTP file ASR
Receive bounded file → sniff/check MIME, decode sample clock → admit verified ASR model → process windows → commit final transcript/timestamps → return response. Large files may need async job; do not keep file bytes indefinitely in memory.

### HTTP SSE text
Admit exact model/session → committed token/text events from TaskCoordinator → SSE adapter buffers at most configured small window → stop/length/cancel terminal frame. UTF-8/stop sequences can span token boundaries; hold back uncommitted pieces.

### WebSocket live voice
Handshake auth + source model/codec capability → choose input/output lanes → binary audio frames and control events with independent sequence numbers → VAD/ASR native TaskDriver → source-valid model output → complete codec chunk → bounded playback buffer → client ACK/truncate on barge-in → reconcile turn epoch and conversation.

### WebRTC live voice/video
Secure ephemeral session/negotiation plus ICE/transport setup → RTP audio/video; data channel for reliable events → transport jitter/RTCP feedback maps into DeliverySession only → model TaskDriver receives exact timed media input and response. No direct RTP socket manipulation of neural StateRoot or GPU BufferLease.

### Async hosted video
Authenticate/authorize egress, license and cost → create provider job with idempotency key → persist provider receipt/job status → poll or verified webhook → validate output asset → canonical committed job result/reference → client downloads within TTL. Retry only after reconcile Unknown.

## 10. Security, rate, billing and privacy research learnings

- Remote bind defaults to safe loopback; public interface must require appropriate auth/TLS, scope and rate limiting.
- Binary media decompress/parse under bounded input size, time, resolution, frame count and entropy ratio; beware zip bombs and forged codecs.
- No arbitrary local file path or media URL fetch based solely on model prompt; user/admin supplies explicit allowed paths/domains/credentials.
- Separate local/inference resource quotas and external provider cost/rate limits. API request budget does not give permission to upload private speech/face to vendor.
- Voice cloning/ref audio requires verified consent and source-voice access policy; log hashes/metadata not raw voice samples by default.
- Provider secrets never embedded in checkpoint, generic task IR, diagnostics or client/browser payload; ephemeral tokens for allowed client direct provider APIs.
- Tenant session state/decoder/acoustic/attention caches isolated; immutable source weights may be shared only with license/provenance checks.
- Reconnect tokens expire and are tenant-bound; playback ACK isn't trusted evidence of human hearing.
- Observability redacts private prompts, source image pixels, raw audio and video unless explicit opt-in.

## 11. Minimality and performance learnings

To avoid multiplying complexity:
- one canonical TaskRequest and ModelBinding/ModeReceipt;
- one actual TaskCoordinator and physically qualified worker/arbiter;
- protocol adapters in separate small Rust modules sharing the same SessionGateway;
- one MediaEvent schema for native and provider outputs with explicit source category;
- per-device/prepared-bundle hot path with no catalog lookup, JSON serialization or global snapshot allocation per packet;
- bounded live ring buffers and credits to prevent GPUs outrunning playback while memory grows unbounded;
- model-specific TaskDriver handles source math; API transport merely moves verified values.

A "zero-copy" claim should be measured at the actual codec/PCI/network boundaries; a reference-counted BufferLease may still require one conversion or copy for transport security/format. Avoid forcing DMA buffers to outlive malicious or broken clients.

## 12. Adversarial serving acceptance

**MM-SRV-01:** official OpenAI SDK short text, streaming, tools subset; unsupported fields explicit.  
**MM-SRV-02:** real-time WebSocket playback ACK is behind delivered, then barge-in truncates only unheard audio and matching transcript.  
**MM-SRV-03:** WebRTC RTP clock conversion to source PCM/video timestamp exact across frame jitter; client interruption/RTCP cannot commit model state.  
**MM-SRV-04:** hosted-live-API-like session reconnect GoAway and output stream replay with client resume cursor; no invented full neural migration.  
**MM-SRV-05:** hosted voice-provider incremental text chunks require source flush/end; model variant doesn't silently change speaker/voice.  
**MM-SRV-06:** hosted video provider accepted POST, network timeout, retry/reconcile does not create duplicate external billable job.  
**MM-SRV-07:** hosted async image job callbacks validate host/signature/TTL; wrong asset MIME rejected.  
**MM-SRV-08:** two sessions on same PCI GPU, one barge-in does NOT advance card-wide physical epoch; other completes.  
**MM-SRV-09:** slow client and 5 video jobs leave bounded PCM/video packet buffers; fair audio response under declared SLO or typed admission miss.  
**MM-SRV-10:** no local DeviceBackend receives proprietary provider raw credential or claims opaque model GPU execution.  
**MM-SRV-11:** versioned native schema decision endpoint consumes finite options, returns raw logits/provenance not generated JSON heuristics.  
**MM-SRV-12:** model hot reload/alias update cannot alter active Session model revision without legal migration.

## 13. Implementation recommendation and scope

First implement source-free MediaEvent, SessionGateway, rational clocks, authoritative vs delivery frontiers and explicit capability matrix. Then protocol adapters and tests. No new model inference engine, no uncontrolled media pipeline DSL.

Prioritize:
1. text + file ASR/TTS adapters sharing EngineCore;
2. WS true duplex with source-free duplex mocked TaskDriver and playback cancellation;
3. WebRTC live media once clock/jitter/egress security model passes adversarial tests;
4. async image/video/hosted adapters, provider consent and cost;
5. high-concurrency workloads with real checkpoint/hardware QoS evidence.

Protocol support is separately qualified from native model support and from current provider marketing.

## 14. External sources / dates

- OpenAI official catalog and model status: https://developers.openai.com/api/docs/models (reviewed 2026-10-15)
- OpenAI Realtime conversations/truncation: https://developers.openai.com/api/docs/guides/realtime-conversations
- OpenAI event schema: https://developers.openai.com/api/reference/resources/realtime/client-events
- Hosted live API capability documentation.
- Hosted live API session management documentation.
- Hosted live API stream ordering documentation.
- Native streaming audio prior-art repository.
- Hosted real-time TTS provider documentation.
- Hosted long-running video provider operation documentation.
- Hosted asynchronous media provider documentation.
- vLLM docs: https://docs.vllm.ai/
- SGLang docs: https://docs.sglang.ai/
- llama.cpp: https://github.com/ggml-org/llama.cpp

The research is about API-observable semantics, not proof of vendor model weights or internal implementation. No service login/test calls were made during this design pass.

````

## Source: .agents/research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md

Original UTF-8 bytes: 18756; SHA256 `efd81080053aad474923d1cd1f57ffff37df91812397baea2d4fd797d42041b1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Research Track C — Text-to-Speech, Neural Audio Codecs, Voice and Full-Duplex Conversation

**15 October 2026 · research/design only.** This track owns source-specific phoneme/prosody/acoustic/codebook/codec semantics and full-duplex state. Serving transport is separately specified; no model is claimed to run natively in Zerogravity now.

## 1. Verdict

A universal "text → wav" routine conceals at least four radically different implementations:
1. **Duration/prosody acoustic synthesis** (Kokoro): G2P+prosody+vocoder without decoder LM.
2. **Autoregressive acoustic codebook generation** (a dual-AR codebook model, some Qwen TTS): one or several nested time/depth loops with a decoder.
3. **Streaming acoustic generation with flow/diffusion chunks** (VibeVoice, model-specific Qwen paths): solver loops inside acoustic frame batches.
4. **Native full-duplex audio-language models** (a native full-duplex speech model, Qwen-Omni): model simultaneously ingests ongoing speech and generates speech/text; not necessarily an ASR→text-LM→TTS sequence.

Each is a trusted TaskDriver/ModelAssembly under the same source semantics, physical buffer/event leases, StateRoot and media serving system. Do NOT create a separate TTS backend server per family.

## 2. Primary source and method matrix

| Exact model/variant | Source method | Mandatory model/codec boundaries | Primary references |
|---|---|---|---|
| Kokoro 82M / voice variants | style-based text/phonemes, prosody and duration modeling, acoustic synthesis/ISTFT vocoder | an independent G2P engine or verified alternative, phoneme tokenization, speaker embeddings, duration/F0, PCM reconstruction | https://huggingface.co/hexgrad/Kokoro-82M |
| Dual-AR speech model (source variant) | Dual AR: slow temporal transformer predicts primary codebook, fast depth transformer predicts nine residual codebooks; 10-codebook RVQ codec ~21Hz | nested temporal/depth clocks, codec frame completion, voice/prosody instructions, waveform decoder | — |
| Qwen3-TTS | variant-specific streaming text/audio decoder; Qwen acoustic tokenizers 12.5 Hz/25Hz and potentially multi-codebook/flow chunk reconstruction | token frames+sample rate, codebook depth, speech tokenizer checksum, chunk/overlap, model-specific voice conditioning | https://github.com/QwenLM/Qwen3-TTS |
| VibeVoice TTS | long-form speaker-consistent generation with source-specific acoustic decoder/flow | 4-speaker or variant-defined voice state, long-duration chunking, semantic acoustic state | https://github.com/microsoft/VibeVoice |
| VibeVoice-Realtime 0.5B | streaming text input, low-latency acoustic synthesis, checkpoint-specific codec/flow | incremental text cursor, bounded output chunks, model-specific lookahead/reconstruction | https://github.com/microsoft/VibeVoice |
| Native full-duplex speech model + neural codec | native simultaneously inbound/outbound audio and text streams, 24kHz codec to ~12.5Hz audio frames, two-stage temporal/depth models | one full-duplex TaskSession with independent capture, generated codebook, text and playback cursors | — |
| Qwen3-Omni Thinker/Talker | separate perception/reasoning Thinker and speech Talker components, coupled text/audio output | verified Thinker→Talker edge, text/acoustic synchronization, option to omit Talker on input-only tasks | https://github.com/QwenLM/Qwen3-Omni |
| Hosted streaming TTS | hosted streaming TTS, voice/speaker selection and dialogue APIs with model-specific WebSocket semantics | explicit provider credentials/egress/consent/cost, text flush/end, cancel and provider voice lifecycle | provider documentation |

**Evidence note:** The dual-AR source model documentation describes its Dual-AR architecture and codec; model weights may use a research-only license different from code. For Qwen-TTS, 12.5Hz and 25Hz refer to **specific tokenizer variants**, not a universal single model rate. For VibeVoice, original repo release/availability has changed historically: inspect exact checkpoint and code location before loading. Hosted voice-provider internals are not independently verifiable; represent endpoint behaviors only.

## 3. Audio state and clocks: minimum trusted schema

~~~rust
struct SpeechSynthesisRequest {
    text: TimedTextStreamOrTextRef,
    voice: VerifiedSpeakerOrStyleId,
    language: Option<LanguageId>,
    sample_rate: RequestedOrSourceRate,
    output_codec: RegisteredAudioOutput,
    prosody: ModelSpecificProsodyControls,
    seed: OptionalModelRngSeed,
    streaming: StreamingCapabilityRequest,
}
struct AcousticFrontier {
    input_text_position: TextTokenCursor,
    time_frame: AcousticFrameCursor,     // 12.5Hz, 21Hz, 25Hz etc
    codebook_depth: Option<RvqDepthCursor>,
    codec_delay: SampleOffset,
    decoded_sample_cursor: SampleCursor,
    speaker_style_version: SpeakerStateVersion,
}
~~~

Time-domain frame and depth-domain codebook count MUST NOT be collapsed into one scalar position; 10 codebooks at 21Hz does not mean 210 acoustic frames per second. Codebook completion and PCM sample clock finality are different. A codec frame is publishable only when its declared number/dependencies of codebooks and waveform reconstruction history are valid.

Separate clocks:
- text Unicode/grapheme/token/phoneme positions (not necessarily aligned);
- acoustic semantic frames and residual RVQ depth;
- resampler input/output sample counts with checked rational ratios and fractional residue;
- vocoder overlap/add state and emitted samples;
- output codec packets and PTS; transport transmitted/acknowledged/played positions.

Voice/speaker state persists within a source-qualified session; changing voice mid-chunk may require explicit new session, codec reset, or model-defined transition. External voice cloning requires verified consent/rights policy, not just a model parameter.

## 4. Kokoro: non-token-AR TTS execution trace

1. Validate raw text language, Unicode normalization, punctuation and length before allocating weights.
2. Run source-required grapheme-to-phoneme frontend (an independent G2P engine or validated equivalent) and phoneme tokenizer; record G2P model/version/license separately from Kokoro weights.
3. Apply verified phoneme-aware text conditioning and speaker/style embedding.
4. Compute duration/alignment/F0/prosody with source-specific inference; normalize and bound predicted frame length.
5. Run acoustic generation and vocoder with source-defined convolution, phase/ISTFT transform, overlap and output sample rate.
6. Apply postprocessing only if source-defined or explicitly chosen audio-quality policy (normalization, loudness, trim).
7. Publish complete PCM/encoded packet at safe vocoder overlap boundary and account for latency.
8. On cancellation, discard uncommitted samples; retain submitted GPU buffers until Ready; playback ACK ledger remains separate.

**Adversarial fixture:** an input with no whitespace (numbers/foreign language) must exercise correct text normalization, not silently strip characters. Changing G2P or voice embedding changes source semantics even if PCM sounds plausible. Two independent listeners share immutable Kokoro weights but never the other person's speaker or text state.

## 5. Dual-AR codebook generation trace

1. Bind text+speaker/style/reference voice, source codec and exact source checkpoint.
2. Slow AR creates primary semantic RVQ codebook tokens along time, approximately 21Hz per source version.
3. Fast AR produces residual codes per temporal frame (nine residual codebooks in the source model's official description). This inner loop is **not** nine extra time frames.
4. Only when the full required depth group is complete may the audio codec decoder consume that frame.
5. Neural waveform decoder reconstructs sample intervals under its overlap/receptive field; resulting PCM cursor may lag codebook generation.
6. Commit and deliver complete audio chunks with exact timestamps. Separate text output, if any, is not implicitly guaranteed to align word-perfectly to codec frames.
7. Barge-in cancels uncommitted future codebook/PCM frames by output generation; already heard output requires playback-boundary accounting.
8. A source reference codec version change requires new assembly identity, not silent reuse of cached frame embeddings.

**Specializations:** device-native slow+fast batch, small depth loops, direct codebook embedding gather, fused acoustic decoder; each must preserve codebook probabilities and state. A fast GPU kernel that reorders RVQ depth will change speech.

## 6. Qwen3-TTS: variant/streaming cautions

Upstream publishes 12.5Hz and 25Hz neural speech tokenizers; checkpoint-specific inference can use distinct codec models, temporal chunk policies and acoustic reconstruction. A single CLI model alias is insufficient to identify the tokenizer/codec and semantics.

- Verifiable components: text tokenizer, speaker/style/reference encoder, acoustic token predictor, speech tokenizer/decoder, optional diffusion/acoustic refinement.
- Codec sample rate, quantizer count, token frame rate, chunk latency and special text/audio sentinel positions come from exact source model.
- A model README may claim streaming-capable architecture while a particular inference wrapper buffers full output before decoding. Zerogravity may advertise playable streaming only when implemented and measured source-compatible codec chunk completion.
- Audio/text synchronized generation requires declared alignment mapping and TaskFrontier coordinates. Do not assert one text token maps to one speech frame.
- Voice design and voice cloning must be opt-in capability with privacy/consent and locally bound reference-audio identity; never silently send source voice to cloud service.

## 7. VibeVoice long-form and realtime

- Long-form voice state may include speaker consistency, conversation context and prosodic trajectory across multiple speech chunks.
- Realtime 0.5B model uses streaming text input: incomplete text words/punctuation and interruptive edits must be tracked, not simply concatenated after generation.
- Acoustic diffusion/flow stages (where present per selected checkpoint) need source solver/timestep; do not substitute direct waveform vocoder.
- Long generation cannot accumulate every PCM sample and every GPU latent in resident memory. Stream complete chunks, preserve bounded overlap/prosody state and release verified retired buffers.
- Speaker changes are explicit track events; multiple voices may share one model but not mutable per-speaker generation state.
- Profile real-time factor, voice consistency and first audible latency separately from GPU batch throughput.

## 8. Native full duplex with a neural codec

A native full-duplex model is a critical test of architecture rather than merely another TTS implementation. Its neural codec publishes ~12.5Hz neural audio frames from 24kHz audio with streaming causal behavior, and the model maintains simultaneous user speech input and its own output plus text stream.

**Trace:**
1. Capture incremental user audio and decode/encode with the neural codec into acoustic input frames, source sample/capture clock and latent delay.
2. Model consumes incoming audio even while its earlier response is still being generated or played, under the source-trained duplex attention/state semantics.
3. Model predicts text and outgoing acoustic tokens, potentially through a temporal backbone and depth decoder.
4. Neural codec transforms only complete output acoustic groups into continuous audio samples with known overlap.
5. Commit neural state and complete output chunks at legal source boundaries. Delivery and playback are tracked separately.
6. Barge-in advances the affected outgoing turn generation, not the entire input acoustic state and not the shared physical device epoch.
7. Model-specific conversation state update on interruption is mandatory. It may **not** be possible to reconstruct an exact state by splicing a text transcript into the stream.
8. Treat user silence, VAD activity, echo and dropped PCM packets as distinct input events; do not infer all responses from text-only prompts.

The neural codec has a real Rust implementation in an independent source repository, useful as **prior art**. It is not permission to embed a separate independent Rust inference engine under the Zerogravity serving stack; adopt mathematical/codec semantics through independently reviewed source/licensing.

## 9. Qwen3-Omni Thinker/Talker

A verified Qwen3-Omni assembly binds audio/video frontend, Thinker, Talker, speech codec/tokenizers and typed text/acoustic outputs. Thinker may have expert routing and state differing from Talker; if input-only ASR/perception is requested, a compatible assembly may omit Talker residency. Conversely, audio output requires valid Thinker→Talker edge and codec output semantics, not a generic language model + speaker file.

Candidate optimization:
- separate residency/offload of Talker and codec decoder for non-audio tasks,
- overlap source-permitted Thinker computation with output codec preparation,
- reuse shared text conditioning under exact artifact identity,
- handle interleaved temporal audio+video inputs and independent clocks.
Any pipelined generation must preserve source causal dependencies and single committed root semantics.

## 10. Hosted voice providers

A hosted voice provider is an explicit TaskServiceAdapter:
- validated voice ID/model ID, output format, language/latency options;
- WebSocket streaming input/output frames and text flush/end markers per provider schema;
- provider-side rate, quota/cost, API key and data-egress permissions;
- client disconnect, barge-in and cancellation receipts; remote operation may continue after local network error;
- source voice consent and user privacy classification;
- provider-reported bytes/seconds, not a fabricated neural StateRoot or local GPU runtime.

Do not label API-hidden model internals "known codec-AR" without official source. Native portability requires actual licensed weights, math and source codec.

## 11. Serving contracts / output finality

- /v1/audio/speech: bounded TTS request with declared output codec/sampling, source-bound voice, zero unsupported-field swallowing.
- Stream events: TextAccepted, AcousticFramePrepared, CodecFrameComplete, PcmChunkCommitted, AudioPacketDelivered, PlaybackReported, InterruptionConfirmed, ResponseTerminated.
- WebRTC: media track and control channel; WS: binary media + control metadata; CLI: file output and optional progressive chunk callback, all from the same committed media event.
- Explicit output-side backpressure: never unbounded PCM buffer. A slow speaker/network can pause at frame boundary or cancel; model cannot overwrite immutable already-committed output.
- Exact sample duration and output decoder delay are required for speech/video synchronization and lived playback history.

## 12. Latency, quality and risk

Measure:
- TTFA (time to first audible valid PCM) and capture-to-first-response,
- P50/P95/P99 chunk gap/jitter, real-time factor (<1 means faster-than-audio-duration, state definition),
- perceptual metrics when allowed plus transcript alignment, speaker similarity, artifacts at chunk boundaries,
- normal/interrupt/low-resource long-form memory consumption,
- default and degraded modes with exact model/codec/numeric receipt,
- voice data retention/security and explicit user permissions.

A vendor-reported 300ms first audio on one hardware class is not a guarantee Zerogravity achieves it on Strix Halo or RTX. Performance evidence must include checkpoint, device, load/temperature, transport, codec and setup.

## 13. Required adversarial acceptance fixtures

**MM-TTS-01:** Kokoro phoneme/duration path generates exact sample count and valid wav without any decoder text LM.  
**MM-TTS-02:** Dual-AR frame with nine of ten codebooks remains unplayable; final depth completes one frame, not ten time frames.  
**MM-TTS-03:** Qwen12.5Hz and 25Hz tokenizer assemblies refuse cross-swapped codec components; clocks round exactly.  
**MM-TTS-04:** voice style updated mid-chunk under an unsupported variant must fail before state corruption.  
**MM-TTS-05:** Full-duplex simultaneous user audio in + assistant audio out preserves independent cursors; no obligatory transcript.  
**MM-TTS-06:** user barge-in truncates unplayed speech, does not erase consumed user input or cancel other GPU sessions.  
**MM-TTS-07:** learned/vocoder overlap buffers persist across samples, but not across unrelated sessions; exact PCM continuity.  
**MM-TTS-08:** slow client backpressure bounded for 10-minute long generation; no infinite output queue.  
**MM-TTS-09:** disconnect/resume playback cannot claim user heard delivered bytes without ACK; maintain UnknownPlaybackProgress.  
**MM-TTS-10:** no untrusted remote voice-cloning/hosted-provider egress without explicit authorization/consent.  
**MM-TTS-11:** partial codec GPU command failure fences physical handle if unknown and refuses StateRoot publication.  
**MM-TTS-12:** source-trained multiprompt/sample-rate and speaker tests on actual hardware before production media claim.

## 14. Incremental implementation plan

1. Introduce source-free distinct sample and acoustic-frame clocks + complete codec-frame finality mocks (no actual TTS weights).
2. Native Kokoro small non-AR TTS with original checkpoint and verifiable G2P/vocoder, CPU reference then qualified GPU.
3. Dual-AR small codec/RVQ fixtures to prove time×depth; then original licensed model if allowed.
4. Qwen3-TTS exact variant assembly and chunked source codec on CPU; only qualified realtime playback advertised.
5. Native full-duplex neural-codec end-to-end latency/interrupt evidence, or Qwen3-Omni Thinker/Talker with original codecs.
6. Hosted voice-provider TaskServiceAdapter independently tested against sandbox/mock API policy, not a second native model engine.

Any new TTS architecture should add a source model assembly and TaskDriver, never a duplicate GPU ResourceArbiter/TaskCoordinator.

## 15. Primary sources checked

- Kokoro: https://huggingface.co/hexgrad/Kokoro-82M
- Rust-language independent source study.
- Dual-AR speech source model.
- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- VibeVoice: https://github.com/microsoft/VibeVoice
- Native full-duplex/neural-codec source model.
- Qwen3-Omni: https://github.com/QwenLM/Qwen3-Omni
- Hosted real-time TTS WebSocket documentation.
- Hosted TTS versus text-to-dialogue documentation.

Checked against upstream docs and earlier model atlas on 15 Oct 2026. All detailed model internals/weights and licensing must be independently bound to the exact implementation/version before declaring Zerogravity support.

````

## Source: .agents/research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md

Original UTF-8 bytes: 17977; SHA256 `e6e9510e073aa5bd0b58c04356b3044d5f7cd91917cb5f81bfc5cceefc891448`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Research Track E — Video, Temporal Diffusion, Synchronized Audio-Video and Frontier Providers

**15 October 2026 · design/source research only; no original video model executes in Zerogravity.**  
**Scope:** Wan 2.1/2.2, original/1.5 dual-stream video DiTs, asymmetric-attention video DiTs, coupled audio/video model families and hosted frontier video providers. User intent includes both open native video and hosted frontier providers.  
**Dependencies:** v5.1 typed media clocks/model assembly, source-registered solver and joint StateRoot/CommitGroup; realtime-serving for live AV output.

## 1. Verdict

Video workloads require the same shared execution spine, but they stress it differently from text:
- a denoising transformer iterates over **spatiotemporal latent grids** rather than a text token prefix;
- source attention may be temporally sparse, sliding, asymmetric or dual-stream;
- a 3D VAE introduces temporal compression, receptive-field overlap, pixel/frame reconstruction and potentially separate tile state;
- Wan 2.2 expert choice is **timestep/denoising-phase routing**, not token-level sparse FFN MoE;
- Coupled audio/video model families and advanced hosted systems can generate coupled **audio and video** with cross-modal neural state;
- output can be valid only at complete decoded frame/segment and synchronizable audio clocks, not at arbitrary latent iteration.

No universal FPS, fixed video latent temporal compression or universal sampler assumption is safe. TaskDriver and ModelAssembly are exact per checkpoint while CUDA/Vulkan/GPU tensor kernels and physical resource ownership are reused.

## 2. Source-backed family matrix

| Model/checkpoint | Distinct semantic path | Required native extension | Primary source |
|---|---|---|---|
| Wan 2.1 T2V/I2V 1.3B/14B variants | spatiotemporal diffusion/DiT + source video VAE, prompt conditioning | temporal latent positions, source flow solver, 3D VAE, image-reference conditioning, exact stages | https://github.com/Wan-Video/Wan2.1 |
| Wan 2.2 A14B / TI2V variants | high-noise vs low-noise denoising-phase experts selected by timestep; model variants differ | step-phase expert/sigma switch, verified phase-resident experts, source 3D VAE/schedule | https://github.com/Wan-Video/Wan2.2 |
| Original dual-stream video DiT | text→large conditioner, dual-stream→single-stream video DiT, causal 3D VAE | distinct attention phase, temporal compression/reconstruction and source prompt optimization | — |
| Dual-stream video DiT 1.5 | ~8.3B DiT, 3D causal VAE (4 temporal/16 spatial compression), selective/sliding tile attention and optional SR | exact sparse attention masks, causal/tiled VAE, SR component and phase+latency planning | — |
| Asymmetric-attention video DiT | ~10B AsymmDiT, non-square QKV/output projections, large visual stream, temporal AsymmVAE | asymmetric hidden dimensions, joint-attention layouts, temporal 3D VAE and context-parallel candidates | — |
| Coupled audio/video dual-stream model | audio and video **coupled** asymmetric DiT streams, two source VAEs, vocoder, bidirectional cross-modal attention | two latent clocks/time-dependent cross-attn, exact sync barriers/AV state root, separate decoder placement | — |
| Hosted video generation service (3.1 and related) | hosted generation of video/audio via long-running API jobs; no public exact neural source | explicitly authorized TaskServiceAdapter with job status, asset retrieval and rights | provider documentation |
| Hosted creative media service | hosted creative media API with async submission/status | provider task lifecycle, auth, references/media asset boundaries, cost/privacy | provider documentation |
| Hosted joint audiovisual service (1.5 Pro) | closed audiovisual generation; public report describes joint AV denoising branches | hosted service today; source-independent joint A/V solver contract for future verified checkpoint | provider documentation |
| Hosted creative video service | hosted creative video/storyboarding/AV generation | provider API version, async jobs, input/reference media and rights; internal neural math Unknown | provider documentation |

**Source distinction:** The requested frontier models are **capability/API targets**, not verified open native weights. Different video model releases have distinct licenses and potentially incompatible VAE/tile/sampler schemas. The listed upstream author claims are not Zerogravity measurements.

## 3. Video value and clock contract

~~~rust
struct VideoInput {
    frames_or_packets: TimedVideoValue,
    fps: RationalRate,                    // e.g. 24000/1001, 30000/1001
    orientation: Orientation,
    color: ColorSpaceDescriptor,
    track_epoch: TrackEpoch,
}
struct VideoLatent {
    batch: u32, latent_channels: u32,
    temporal_length: u32,
    spatial_width: u32, spatial_height: u32,
    vae_identity: VerifiedDecoderId,
    temporal_compression: VerifiedTemporalStride,
    tile_plan: Option<VerifiedTileGeometry>,
    solver_step: SolverCursor,
}
~~~

Pixel/latent/media packet timebases differ: true image frame count, temporal VAE latent frame number, solver step index and codec packet DTS/PTS are separate dimensions. For 30000/1001 FPS, a single frame spans 1001/30000 s, not a floating 33.33ms rounding repeated forever. Audio 48k samples cannot be aligned by assuming integer samples/frame. Resampler/codec delay and fractional remainder must participate.

Temporal VAE output cannot be considered final until receptive-field overlap and crop on all necessary adjacent windows are complete. Frame set is publishable only when all component readiness events and source numerical/output constraints pass.

## 4. Source-valid video-generation execution trace

1. Validate task kind (T2V, I2V, V2V, TI2V, extension, keyframes), exact checkpoint variant, aspect, number of frames, total decoded byte budget, and license.
2. Decode input/keyframes/references with exact color/alpha/orientation; bind source processor identity and references in assembly.
3. Encode text and image conditioning via verified source encoders; prompt rewriter is separate component (and privacy/egress policy if remote).
4. Seed initial spatiotemporal latent grid using the exact registered noise RNG and scheduler config; no silent replacement of seed or time bounds.
5. Run DiT/flow/denoiser iterative steps, preserving timestep/sigma order, sparse mask, cross-modal attention and source guidance.
6. Under Wan2.2 choose high- or low-noise expert according to declared source switch threshold and each step's actual timestep. Do not choose expert by token/latent index.
7. Source VAE decodes valid latent temporal windows with causal/tile overlap, optional registered upscaler and pixel transform.
8. Encode output video packets (and optional audio) under source target frame rate, stream codec, color and requested container; track DTS/PTS and mux metadata.
9. TaskCoordinator publishes complete, playable frames/segments only after all required model/codec events Ready, and only in correct output generation/track epoch.
10. Long job status/progress is independent of model state and transport: queued, running, completed/canceled/failed/unknown; no fabricated partial final video before VAE/mux completion.

## 5. Wan 2.1 specifics

- Source spatiotemporal DiT/3D VAE with T2V/I2V variants; validate exact trained length/resolution and input image conditioning.
- Denoise/rectified flow schedule is model/variant specific and must be captured by solver identity.
- Device-native step loop may avoid repeated host roundtrips only while preserving source RNG/timestep/guidance and bounded latent/event memory.
- VAE time dimension may require padding/extra frame semantics at start/end; shape guard must include causal window and reassembly crop.
- Long video with high resolution requires tile plans, temporal window and offload accounting; no model "fits in VRAM" claim based on weights alone.

## 6. Wan 2.2 denoising-phase MoE

Wan 2.2 introduces MoE by splitting the denoising trajectory into specialized experts with high-noise and low-noise responsibilities. This is **whole denoising-phase** switching, unlike LLM sparse FFN expert selection per token.

Required source semantics:
1. Registered timestep/sigma schedule in TaskDriver.
2. Source-defined threshold/phase selection rule; no heuristic "choose the expert that is currently warm" override.
3. Proper dual expert weight source identities, potentially different LoRA/quant or model variants.
4. Residency pipeline can prefetch the later expert when schedule safely predicts phase, but cannot evict an earlier expert while GPU buffers using it remain pending.
5. Actual memory plan includes transition spike (both expert sets may be simultaneously resident), not merely max(single expert).
6. Denoiser output continuity and RNG across phase boundary must be source-equivalent; no ad-hoc blending or interpolating the expert parameters.
7. TI2V variants that are dense do not automatically use the same MoE plan.

**Negative:** With source timetable [high,high,low], a worker executing high at the last step should fail the source variant quality gate even if output visually plausible.

## 7. Dual-stream video DiT original and 1.5

Original model combines textual conditioning with dual-to-single stream multimodal video transformer and causal 3D VAE. The 1.5 source highlights:
- selective/sliding tile attention (SSTA): attention mask/selection is a mathematical constraint; cannot silently replace with dense attention whose outputs differ numerically or a different sparse window;
- causal 3D VAE compression and stateful tile/overlap behavior;
- optional super-resolution and distillation steps are separate component/quality mode;
- source attention backend/FP8 GEMM can be optimized only within validated output numerical/quality contract.

Proposed specialization: source-trained sparse-tile region with precompiled allowed tile shapes and safe fallback. High-level SparseAttn registry needs explicit selected block IDs/mask and spatial/temporal axes; a generic "local window" key is insufficient.

## 8. Asymmetric-attention DiT and AsymmVAE

The original asymmetric-attention model has unequal text and video hidden widths, non-square QKV and output projections, and asymmetrically sized visual/text subnets. This is a direct test of composability: a runtime with an implicit equality constraint on cross-modal hidden dimensions cannot support it correctly.

- Semantic model operator must allow rectangular QKV and cross-stream attention projections.
- Source tokenizer/T5-XXL text conditioner and video VAE are separate verified components.
- 3D VAE temporal/spatial compression and causal decode/dense attention masks are checkpoint-specific.
- Context/tensor parallel candidate requires correct cross-device rank/ordering, scratch and source numerical proof; not inferred from vendor benchmark claims.
- Precompiled shapes can specialize common spatial/temporal sizes without editing TaskCoordinator or public HTTP API.

## 9. Coupled joint audio/video, exact synchronized group

Source documentation describes **asymmetric dual-stream audio and video DiT** with bidirectional attention, audio and video VAEs, vocoder and Gemma-3-based text conditioner in certain source variants. Internal model dimensions must be loaded from exact checkpoint rather than copying one technical paper's parameter split.

~~~text
Text + optional image/audio/video references
       -> verified conditioner embedding(s)
       -> video noisy latents, video spatial/temporal positions
       -> audio noisy latents, audio timebase and neural codec representation
       -> JOINT dual-stream transformer iteration:
             video self-attention + audio self-attention
             text conditioning + bidirectional A↔V cross-attention
             timestep-conditioned heads / source solver
       -> synchronized complete audio+video latent group
       -> 3D Video VAE + Audio VAE/vocoder
       -> aligned complete A/V packet groups with rational PTS
       -> ONE TaskCoordinator synchronized PublishRoot
~~~

Neither output may publish as independent final AV if source requires cross-attention through an unfinished iteration. The task frontier includes (video latent timestep, audio latent timestep, video PTS, audio sample/frame cursor, decode readiness watermark) and a CommitGroupId with legal cross-track readiness.

**Important counterexample:** generating Wan video and attaching unrelated Kokoro speech afterward is a valid *different multimodel application* but not a native inference implementation of the coupled A/V model.

## 10. Frontier proprietary video services

A public hosted report describes joint audiovisual diffusion; exact internal neural source weights/math are not available in this design inventory. Hosted adapters preserve:
- request model/version, text/images/references, duration/frame rate/audio flags and license/rights;
- async job id, progress, expected polling intervals, provider output asset ref and expiry;
- upload staging and authenticated callback/webhook verification;
- idempotency keys, rate quotas, cost controls and unknown status after connection failure;
- no hidden external egress or implied local GPU/VRAM budget.

Other hosted providers similarly provide API observable semantics, but not a native executable SIR recipe unless separately verified open weights exist. TaskServiceAdapter must not claim numerical equivalence to local Wan or coupled A/V.

## 11. Physical memory and QoS

Whole video assembly budgets must cover text/vision conditioner, denoiser experts, VAE/codec, temporal state, CFG branches, latent+tiling scratch, video+audio output packetization, buffers and physical transfer leases. In mixed workloads prioritize interactive duplex audio at bounded soft deadlines without allowing unbounded video starvation; schedule video at **legal solver-step boundaries**.

Cost/perf metrics:
- first valid playable video frame/segment latency, not just first noisy latent;
- fps after reconstruction, real frame timestamps and A/V sync skew;
- peak real HBM/VRAM/UMA usage, offload and multi-device transfer;
- noise-step latency, denoiser/latent residency, VAE tile cache hit and SR overhead;
- overall video quality on pinned source prompts/seeds and desired resolution;
- context/temporal window drift and looping boundary artifacts;
- cost, job egress and retries for hosted services.

Any "realtime video" request must have a measured admit/refuse policy; a 50-frame batch finishing eventually is not real-time interactive video.

## 12. Test matrix for independent agent implementation

**MM-VID-01:** Wan1 source temporal latent shape/3D VAE reconstruction with exact final frame count and crop.  
**MM-VID-02:** Wan2.2 source fixed high/low-noise phase transition and dual expert residency; wrong step expert rejected.  
**MM-VID-03:** source original dual-stream video DiT dual→single stream transition and conditioning masks, valid compiler region split.  
**MM-VID-04:** Dual-stream video DiT 1.5 SSTA mask/selected tiles equality against slow reference; no silent sparse-mask relaxation.  
**MM-VID-05:** Asymmetric-attention DiT non-square text/video projection and source-specific VAE temporal compression; rectangular dimensions valid.  
**MM-VID-06:** Coupled two-stream A/V solver with audio/video cross-attention dependencies and legal A/V CommitGroup barrier.  
**MM-VID-07:** 48k PCM + 30000/1001 FPS video duration alignment over 60 minutes, exact residual arithmetic.  
**MM-VID-08:** slow codec/mux returns backpressure, prepared video frames retained but not exposed as final.  
**MM-VID-09:** cancel one video denoise task after physical submit; no unrecovered buffer, preserved old StateRoot.  
**MM-VID-10:** 5 video jobs + live ASR/TTS and text requests share physical GPU budget without false QoS guarantee.  
**MM-VID-11:** hosted video provider timeout after create reconciles idempotent job; no duplicate billable submission.  
**MM-VID-12:** streamed video no invalid GOP/packet timestamps; clock discontinuities are explicit.  
**MM-VID-13:** model source decoder/source font/text/video language license verified independently of repository.  
**MM-VID-14:** artifact hot reload cannot reuse old temporal VAE state/codec buffers with new semantics.  
**MM-VID-15:** cross-model Wan video + external Kokoro TTS is identified as compound task, not a native coupled A/V model.

## 13. Implementation order, not a changed user model priority

1. Source-free spatiotemporal latent/3D VAE contracts and tiny fake noise-step loop.
2. One small Wan2.1 or similar open-video original checkpoint with source-trained CPU reference; favor feasible shape/hardware.
3. Wan2.2 two-expert denoising transition and actual measured residency/physical GPU scheduler pressure.
4. Asymmetric attention and dual-stream video DiT sparse tile masks, separate numerical equivalence tests.
5. Coupled joint synchronized audio/video for real coupled generation, not a post-hoc mux.
6. Hosted provider adapters independently behind explicit authorization.

No step implies original GPU Qwen3.5/Gemma4/Qwen3.8 implementation priority was altered.

## 14. Primary sources reviewed 15 October 2026

- Wan2.1: https://github.com/Wan-Video/Wan2.1
- Wan2.2: https://github.com/Wan-Video/Wan2.2
- Original dual-stream video DiT source model.
- Dual-stream video DiT 1.5 source model.
- Asymmetric-attention video DiT source model.
- Coupled audio/video source model.
- Coupled audio/video component architecture.
- Hosted video generation provider documentation.
- Hosted creative media provider documentation.
- Hosted joint audiovisual provider release documentation.
- Hosted creative video provider API documentation.

No video models were downloaded or run. Release, license, Tensor role, scheduler and artifact source code must be revalidated at implementation time.

````

## Source: .agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md

Original UTF-8 bytes: 19654; SHA256 `141281cd8fb7e52b4e21b6ef545a046427852b698c4371d58e50e22f19502b5c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Accelerator execution prior art — Blackwell, TPU, wafer-scale, static dataflow and specialized runtimes

**Research snapshot:** 2026-10-12. **Role:** attributed, non-normative **prior-art evidence** only. Project-neutral architecture is in `ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md`, `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` and the master plan. The separate source research directory `/home/pvl/spaces/abird/research/inference` is read-only and unaffected.

**User question:** account for NVIDIA DGX Spark/GB10, other Blackwell systems, Google TPU, “cerberus” (plausibly **Cerebras** wafer scale, but **Cerberus** is also a *different published speculative-decoding algorithm*), plus other accelerators/runtimes. The real issue is not a missing GPU product enum; it is whether the core can execute **kernel, captured graph, AOT graph, compile-time scheduled dataflow and remote session-managed regions** without cloning a server.

**Evidence classes:** [VERIFIED-DOC] first-party public architecture documentation as inspected; [AUTHOR-REPORTED] published specifications/claims not independently reproduced on project hardware; [INFERRED] architecture implications; [OPEN] native compiler/runtime/IP capability or policy not yet demonstrated. No zerogravity kernel, TPU program, Cerebras wafer program, Blackwell benchmark or cluster experiment exists.

## 1. NVIDIA Blackwell is multiple hardware *systems*, not one GPU profile

### DGX Spark / GB10 (local ARM + coherent low-power Blackwell)

- [First-party DGX Spark product specification](https://www.nvidia.com/en-us/products/workstations/dgx-spark/): **Grace Blackwell GB10**, 20-core Arm CPU, Blackwell GPU, 64 GB or 128 GB LPDDR5x coherent unified **system memory**, listed 273 GB/s memory bandwidth and up to 1 PFLOP FP4 theoretical. These are vendor peak/platform claims, not an actual measured memory headroom; even with common addressability, sharing bandwidth and OS memory is real.
- [DGX Spark release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html): software/firmware/driver and unified-memory pressure behavior are version-sensitive; current document's release table can change independently of OEM partner systems. The device probe needs **a system profile not just `compute_capability=Blackwell`**.
- [NVIDIA DGX Spark native inference tutorial](https://build.nvidia.com/spark/llama-cpp/overview) documents device-specific native CUDA/offloading paths, but its C++ serving runtime is **prior art**, not an engine dependency.

**Transferable mechanisms:** CPU/accelerator coherent memory with explicit physical headroom; ARM64 device+host executable support; tensor low-bit cache/packing and decoded/prefill shape specific strategies; avoid assuming HBM7TB/s from datacenter Blackwell. Hypotheses: small batch memory-bound decode can prioritize device bandwidth and resident expert policies differently from Blackwell B200 large batch. Need actual GB10 reference hardware to qualify.

### Blackwell RTX/PRO discrete vs B200/B300 and GB200/GB300 rack domains

- [NVIDIA Blackwell architecture/inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) documents `NVFP4`: E2M1 FP4 payload with **FP8 E4M3 scale per 16 values** and an additional per-tensor scale; `MXFP4` is different (shared power-of-two block scale of 32). **Don't conflate NVFP4, MXFP4, GGUF IQ2/Q4 and on-SoC shared-exponent BF8 as one codec or instruction path.** Stored checkpoint quant, transformed physical representation and compute arithmetic remain separate identities and quality gates.
- [B200 and GB200 hardware and network](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/) documents standalone 8-GPU NVLink/NVSwitch DGX B200 and rack-scale GB200 NVL72. [NVLink topology](https://developer.nvidia.com/blog/low-latency-inference-chapter-2-blackwell-is-coming-nvidia-gh200-nvl32-with-nvlink-switch-gives-signs-of-big-leap-in-time-to-first-token-performance) demands true GPU peer/fabric dataflow; B200/GB200/GB300/HBM, GDDR workstation RTX and Spark shared LPDDR cannot share one memory/bandwidth/resident tensor assumption.
- [NVIDIA current generational NVFP4 and hardware](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) and [low precision diffusion](https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/) show why **kernel/region execution specialization must cover *both* LLM/decoding and repeated diffusion/flow tasks**, graph captured steps, low-bit tiling and multi-GPU sharding. Published speedups depend on quant and selected algorithm (including optional denoiser cache changes), not necessarily identical model quality.

**Prior-art runtimes/techniques:** CUDA graph/captured eager runners, vendor GEMM/attention kernel libraries, current autotuned attention-kernel research and open local C/C++ runners. Project-neutral mechanism = supported `NativeKernel` or `CapturedGraph` region variants, with local device managers/collective topology and real shape/numeric/precision/quality proof. Do not wrap one of those inference *engines* as our required runtime.

## 2. Google TPU v5e/v6e/v7x — compiled graph + fabric, not ordinary CUDA commands

- [Google TPU7x (Ironwood) hardware](https://cloud.google.com/tpu/docs/tpu7x) and [2026 Ironwood hardware+system architecture](https://cloud.google.com/blog/products/compute/training-large-models-on-ironwood-tpus/) characterize chip/pod topologies and HBM/ICI, multiple dimensions of device placement and collective communication.
- [Cloud TPU software/PJRT](https://docs.cloud.google.com/tpu/docs/runtimes) explains that TPU v5e and newer use **PJRT**, with host/framework API and compiler-managed device execution. The user cannot assume a raw CUDA-like per-op `KernelProvider` in a common address space.
- First-party 2026 TPU software architecture documentation describes a **model graph→StableHLO→XLA** compiler route and custom framework-native kernels, plus the real problem of **dynamic sequence-length recompilation**. First-party TPU inference backend documentation covers PyTorch support and separate tested v6e/v7x model recipes/support matrices. First-party TPU kernel-language documentation warns of restricted experimental custom code semantics; not all arbitrary GPU ops are native TPU kernels.
- First-party Google TPU execution stack documentation describes a compiler-centric device path with on-chip tiling, memory movement and topology partitioning; host-runner control and compiled graph loops differ from GPU explicit per-op launches.

**Transferable mechanisms:** a **large closed SIR region/subgraph** passed to a trusted *AOT Graph Compiler Provider*, with explicit graph-boundary input/output/state and collective topology; dynamic shape constraints, compile cache, HBM and rendezvous, asynchronous program launches and state snapshot capability. Optional TPU specific custom region lowering requires vendor compiler/IR/runtime interfaces. A working high-performance TPU native backend may **not** be possible in pure Rust kernel-source mode without an approved external compiler that emits device binaries; record this as OPEN, don't pretend TPU is already supported.

## 3. Cerebras WSE3 / CS3 — wafer-scale static dataflow and streamed weight model

User said “cerberus”; this likely references **Cerebras**, a wafer-scale AI architecture, but see §4 for a separate similarly named decoder.

- [Cerebras wafer-scale cluster](https://training-api.cerebras.ai/en/latest/wsc/Concepts/how-cerebras-works.html) describes WSE-3 / CS-3 with a fabric of large numbers of processing elements rather than CUDA blocks.
- [Weight streaming](https://training-api.cerebras.ai/en/latest/wsc/Concepts/cerebras-execution-modes.html) describes streaming one layer's weights from an external MemoryX service to WSE when total weights don't fit on-chip, including backward/gradient streaming in training. **The accessible document primarily describes a training execution mode**; don't assume identical contracts or access for arbitrary inference workloads.
- [Cerebras compiler and low-level SDK](https://www.cerebras.ai/product-software) and [SDK dataflow kernels](https://www.cerebras.ai/blog/supercharge-your-hpc-research-with-the-cerebras-sdk) describe graph compiler placement/communication and Cerebras Software Language (CSL), a **C-like** low-level WSE kernel language, with simulator and supported hardware generation limitations. Source language/proprietary runtime constraints prevent us from claiming a Rust-owned native WSE compiler exists.

**Transferable mechanisms:** `CompileStaticDataflowProgram` across an entire closed model or stage region, compile-time placement/routing and on-chip buffers, external streamed tensor/weight sources, large resident model program/fabric partition, bulk collective capacity and long **device program lifetime**. The host orchestrator owns task/session/state; the programmed fabric handles its internal micro-schedule. Expose compiled subgraph or a **managed remote service boundary**, but don't conflate calling a Cerebras inference API with truly executing our SIR under our native provider. If only a model-serving endpoint is available, that is **external TaskService interoperability**, not native kernel/graph integration.

## 4. Cerberus — a distinct parallel-decoding algorithm, not wafer-scale hardware

- [Cerberus: Efficient Inference with Adaptive Parallel Decoding and Sequential Knowledge Enhancement](https://arxiv.org/abs/2410.13344) describes adaptive parallel decoding and sequential knowledge-enhanced draft heads; author reports acceleration up to 2.12× on benchmark settings. **Do not confuse it with Cerebras WSE**.
- Our architecture already supports this *class* through `DraftSource`, `TargetVerifier`, `CommitPlan` and full SIR StateRoot speculative rollback. Whether published Cerberus decoding **preserves exact target distribution** for each decoding mode requires separate algorithm-specific proof. Do not copy model weights or source code based on an unverified license or marketing result.

## 5. AWS Trainium/Inferentia — specialized AOT accelerator compiler and runtime

- [AWS Neuron SDK](https://aws.amazon.com/ai/machine-learning/neuron/) is a distinct compiler/runtime/kernel system for Trainium and Inferentia, using graph compiler and custom Neuron Kernel Interface (NKI), runtime memory and collective management.
- [Neuron 2.30, May 26 2026](https://aws.amazon.com/about-aws/whats-new/2026/05/aws-announce-neuron-2-30-0/) includes Trainium3 kernels, segmented attention, KV-parallel prefill, FP8/MXFP8 and model porting/quality validation, runtime zero-copy host-device features.
- [Neuron 2.27, December 2025](https://aws.amazon.com/about-aws/whats-new/2025/12/announcing-aws-neuron-2-27/) explicitly details SDK integration, custom compiler and newer Trainium3 support.

**Transferable:** `AOTGraph` and optional fine-grained `AcceleratorKernel` candidates, graph compiler fingerprint, compiler-owned high bandwidth memory/collective layouts, real **compile/warmup costs** and model/config/precision-specific quality/shape qualification. No claim direct Rust source can yet produce equivalent Trainium ISA without vendor toolchain.

## 6. Deterministic dataflow, SRAM processors and hybrid GPU+LPU

- [Groq LPU architecture](https://groq.com/blog/the-groq-lpu-explained): compiler-managed deterministic static dataflow schedule, on-chip SRAM, inter-chip communication planned at compilation. A static `ExecutableBundle` can represent a host submission to an opaque **on-device scheduled program**, not every micro-op as a host command.
- [NVIDIA Groq 3 LPX March 2026 architecture](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform) presents a **heterogeneous GPU+deterministic LPU pipeline**, where different model regions can run on accelerator families specialized for prefill, attention and FFN/MoE. Vendor projected hardware speedups aren't measured by zerogravity.
- [SambaNova reconfigurable dataflow](https://sambanova.ai/products/dataflow-architecture) describes programmable compute+SRAM memory grid and long fused operator pipelines; [additional execution modes](https://sambanova.ai/blog/accelerating-scientific-applications-with-sambanova-reconfigurable-dataflow-architecture) are a prior-art concept for dataflow graphs/programs.

**Transferable:** compile-time **placement/schedule**, internal SRAM-local buffers, deterministic or bounded shape/branch control, streaming transfers and host-orchestrated task boundaries; one backend ProgramProvider with static program region extent. **Not** permission to claim Groq or RDU native compilation without SDK/source/driver access.

## 7. Other hardware families informing provider constraints

| Hardware/program family | Verified first-party docs | Specific abstraction test |
|---|---|---|
| **AMD Instinct CDNA4 MI350/MI355 and CDNA5 MI400/Helios** | [ROCm MI350 microarchitecture](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html), [AMD CDNA families](https://www.amd.com/en/technologies/cdna.html) | MXFP4/6/8 scaling, MI350 partitioning, Infinity Fabric and rack collective topology; distinct from RDNA3.5 iGPU |
| **Intel Gaudi 3 / Ethernet accelerator fabric** | [Intel Gaudi 3](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html) | graph compiler and runtime, Ethernet/RoCE topology, distributed collective failure |
| **Tenstorrent Wormhole/Blackhole** | [TT system software](https://docs.tenstorrent.com/) | Tensix local SRAM/dataflow and graph/custom kernel compilation, licensed low-level C++ kernel toolchain; needs independent Rust-source feasibility |
| **Qualcomm Cloud AI 100 / edge NPUs** | [Qualcomm Cloud AI SDK](https://www.qualcomm.com/developer/software/cloud-ai-sdk) | compiled graph accelerator with licensed runtime, static kernels and quantization/model converter; standalone managed device, not CUDA |
| **Apple Silicon / Metal and Neural Engine** | source references in previous research | unified physical memory, GPU shader/graph and NPU subset with optional OS/framework device program interface; source/OS restrictions |
| **Huawei Ascend / CANN** | external infrastructure ecosystem source pins to be verified if prioritised | new compiler and matrix precision/collective source adapter, driver and firmware qualification |
| **FPGA / DSP / ASIC future** | architecture class (not a concrete vendor implementation claim) | bitstream/program identity and state/layout, program-load time and trust, streaming latency/tile constraints |

The **architecture** must be capable of modeling these families, but the **implementation** cannot be universal without licensed compiler/runtime/driver, reference model and real hardware tests. A remote appliance that only exposes an inference API cannot promise intermediate tensors, KV/SSM state snapshots, device-resident draft verification or custom operator lowering.

## 8. Why this changes the design beyond v4

v4's `KernelProvider` and `DeviceBackend` naturally express explicit CPU/GPU launches and on-SoC NPU tiles but are **not sufficient as currently written** to guarantee high-performance SIR execution on graph-only/distributed/statically scheduled fabrics. Required neutral refinement:

1. **`RegionExecutionProvider`** replaces "KernelProvider only" as the *single* capability-discovery/lowering seam. Candidate compiled implementation kinds: `NativeKernel`, `CapturedCommandGraph`, `AheadOfTimeGraph`, `StaticallyScheduledDataflow` (and only optional `ManagedSessionRegion` where semantics/state API permit). Native KernelProvider becomes an internal variant, not a new top-level engine.
2. **Compiler-owned region cuts:** choose a *large closed subgraph/model task stage* for graph compiler/static dataflow, with full state input/output, effect/position, quant/shape guards and external memory residency. A graph-only backend may be unable to execute one arbitrary tiny operator with acceptable cost. Compiler avoids expensive host↔accelerator transfers and per-token graph recompilation.
3. **`ProgramLifecycle` and `PlacementContract`**: compile/warmup/upload/load/resident program memory, param/weight IO, static topology/tile placement and collective/rank membership, micro-scheduler owner, program cache+source version, exact supported shape/precision/bitstream/source-language policy.
4. **`StateBridge`**: provider must explicitly support copy-on-write/provisional state, snapshot/export/import/restore or safe replay so TaskCoordinator's **one Prepare→PublishRoot** is not defeated by an opaque stateful remote program. A black-box API service not giving an acceptable state bridge must stay above computation as a TaskService adapter with weaker/explicit session semantics.
5. **Host versus device scheduler boundary:** TaskCoordinator manages fairness, admission and session logical roots, while TPU graph compiler or static dataflow fabric owns its **device-internal** tile placement and execution timing. Do not recreate each micro operation as a host GPU command—this would lose the hardware's performance advantage.
6. **Topology and quant are first class:** Spark UMA, HBM GPU and rack-scale NVLink, TPU ICI, WSE on-chip SRAM+weight stream, LPU on-chip SRAM/fabric have distinct tiers/links/power, native FP4/FP8 scaling and shape/compilation sensitivities. A one-dimensional `gpu_memory_bytes` or universally "fastest FP4" plan is wrong.
7. **Ownership and honesty:** Rust-owned `SemanticProgram`/TaskCoordinator/ResourceArbiter can use a vendor AOT compiler or device program if explicitly allowed by source-language/vendor runtime policy. That is **not** equivalent to building a fully Rust-authored native TPU/Cerebras/LPU kernel compiler; label capabilities `designed/needs-toolchain/needs-hardware/functional/qualified` distinctly.
8. **Future-proof exit gates:** independent capability tests for a graph-only fake device, static dataflow simulator, opaque state refusal, topology partition failure and precision/shape fallback **without implementing separate model server**.

## 9. Status and scope discipline

Source evidence dates/speeds and model support are as of 2026-10-12; online pages may describe experimental preview, release timing, theoretical peak hardware, author-tested subset or vendor-specific deployment. Never equate those to an implemented zerogravity runtime. **No hardware was connected, benchmarked or compiled during this research.** No source from prior-art projects was copied unless directly vendored (for example, CubeCL (vendored)), which retains its upstream license; the Zerogravity engine is a from-scratch rebuild informed by prior-art concepts, learnings and research. Only `src/zerogravity/.agents` plans/docs are in scope; external `research/inference` remains read-only.

**Key conclusion:** To absorb genuine backend diversity, the engine must own **semantic model/task correctness and resource/state publication** and delegate the **smallest efficient closed physical execution unit required by the accelerator**, which ranges from a tensor kernel to an entire compiled distributed dataflow program. That is the necessary and sufficient **architectural extension contract**; performance and native execution rights require actual proven backends.

````

