Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# P12 — native fused GPU batch acceptance

**Accepted baseline:** original Qwen3.5 resident CUDA with independent
OpenAI HTTP sessions, verified GPU completions, cancellation, and
source-compatible scheduling waves. Each member still performs its
OWN full-model GPU program. **Actual B>1 fused kernels: zero.**

## Ownership boundaries

- `zerogravity-gpu/src/resident_inference.rs`: a single
  `ResidentModel` owns immutable source/physical `ResidentService`;
  `ResidentSession` owns each logical `SessionAuthority` and StateRoot.
- `zerogravity-gpu/src/resident_program.rs`: current
  `ResidentProgram::launch_impl` accepts one RuntimeParameters,
  one source-specific StateBinding cohort, and one device token.
  PlanNode kernels currently execute one request at a time.
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

Changing a scheduler counter to "fused" without one native
multi-row GPU dispatch is forbidden. This file is pending
implementation and tests, not a completed acceptance claim.
