Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# P12 — native fused GPU batch acceptance

**Current HTTP acceptance — P12-B6:** opt-in `batch_mode=fused`
uses the original full-model native CUDA B2/B4/B8 plan for compatible
same-frontier prompt tails/decode. Explicit `--batch-rows 2|4|8`, default2;
interleaved remains the default mode. Surviving rows regroup without padding.
Both approved original Qwen3.5 models passed actual release JSON/SSE,
trained EOS, cancellation and exact internal CPU/B1 state tests.
Engine full248320-logit and compact packet/all42 evidence remains distinct
from HTTP wire evidence. Unequal frontiers, paged state, full-prefix native
prefill and matched vLLM performance remain open. Named later checkpoints
supersede only the corresponding historical limitations below.

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
  `native_projection_dispatches_completed` is the projection subset;
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
