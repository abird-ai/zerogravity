Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# P12 — native fused GPU batch acceptance

**HTTP baseline:** original Qwen3.5 resident CUDA with independent
OpenAI HTTP sessions, verified GPU completions, cancellation, and
source-compatible scheduling waves. Each member still performs its
OWN full-model GPU program. **HTTP fused full-model batches: zero.**
The separate native ForwardPlan diagnostic is tracked under P12-B3.

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

Same-frontier, full-logit diagnostic only. Additional trained
quantization profiles, unequal prefixes, native per-row greedy/EOS
feedback, paged hybrid KV, continuous HTTP integration and the
matched vLLM performance gate remain open. `batch_mode=fused`
continues to refuse; HTTP `gpu_fused_batches=0` remains accurate.

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
- Promote `batch_mode=fused` only after B>1 native physical
  dispatch and source-exact Root/Ready acceptance on every
  advertised GPU backend; until then return
  `unsupported_batch_mode` and `gpu_fused_batches=0`.

Changing a scheduler counter to "fused" without native multi-row
execution is forbidden. P12-C/P22 remain pending; the sections
above delimit the physically accepted operator and model scopes.
