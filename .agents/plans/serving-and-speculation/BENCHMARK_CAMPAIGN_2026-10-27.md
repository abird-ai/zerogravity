# Qwen3.5 matched serving campaign — 27 Oct 2026

Author: @pvl

## Purpose

One serial, correctness-gated comparison of ZeroGravity and pinned vLLM for the
approved Qwen3.5 `.8B` and `2B` sources. This plan excludes Fabric vNext.

No cell runs until its exact model/source/numerical/context/physical route is
qualified. Refusal is a valid result; a finite capture, source proof, compiled
kernel, or synthetic queue does not grant a benchmark route.

## Fixed identities

- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB, one campaign owner.
- vLLM: `0.31.0`, commit
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Pinned image: `sha256:f57d77f5…b4b0dbe` (full identity remains in the
  retained lifecycle package).
- vLLM profile: eager BF16, TP1/PP1, prefix cache off, chunked prefill on,
  `max_num_seqs=8`, explicit 512 MiB KV allocation.
- `.8B`: revision `2fc06364715b967f1860aea9cf38778875588b17`.
- `2B`: revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Tokenizer/source hashes must be supplied to `tools/serve-compare.py` and match
  between providers. Route names alone prove nothing.

The retained vLLM package and complete hashes are indexed by
`../../docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-27.md`.

## Required implementation gates

1. One ordinary, source-derived Qwen CUDA route for both models:
   - shared call-free `ForwardPlan`;
   - authenticated source owner retained for the plan/invocation lifetime;
   - no prompt ID, prompt length, call ordinal, node ID, packet contents,
     captured recipe hash, or nearest-case lookup as execution authority;
   - chunked prefill and cached continuation through the existing queue, flight,
     completion/quarantine, arbiter, and `StateRoot`.
2. Complete non-GDN source-derived CUDA lowering and physical owner.
3. Parameterized GDN schedule and state transport, numerically qualified.
4. Initial and cached correctness for each admitted context, including all 24
   layer boundaries, all 42 state domains, full 248,320 logits, natural greedy
   feedback, cancellation/faults, retirement, and source/resource conservation.
5. Real-model shutdown replay on the exact benchmark binary/source.
6. Complete `scripts/check.sh` on the assembled source.
7. C32/C64 additionally require real production pending/admission/active-worker
   qualification. The source-free production Router test passed on current
   source on 27 Oct; model-specific C32/C64 still require the ordinary route.

## Workloads

All requests use the exact approved chat template/tokenizer and request eight
natural output tokens unless EOS ends earlier. Store exact prompt token IDs,
reported prompt/completion usage, output token IDs, finish reason, source and
binary identities.

### Short canonical

- Exact canonical 47-token prompt.
- C1, C4, C8, C16.
- C16 means 16 offered clients over the admitted active-8/B4 implementation;
  it is not a B16-kernel claim.

### Short ragged

- Exact 47/57/62/67/72-token prompt set.
- Deterministic balanced populations at C1, C4, C8, C16.
- Preserve per-request identity; do not compare unlike prompt outputs as one
  aggregate string.

### Queue scaling

- Canonical and ragged C32/C64 only after the production queue and exact model
  route pass their gates.
- Report offered, admitted, queued, active, rejected, canceled and completed
  populations; actual peak active width and native cohort width; fairness and
  queue residence.
- Do not imply B32/B64 kernels. Current target remains active at most 8 and
  native Qwen batching at most B4 unless later physical evidence says otherwise.

### Context ladder

- 512, 1K, 2K, 4K, 8K; then 16K and 32K.
- Begin at C1. Add concurrent long-context cells only after exact physical fit
  and correctness at that context.
- Prompt text must be generated against the shared tokenizer and verified by
  terminal reported prompt-token usage. A byte length is not a token length.
- `.8B` 32K is only a candidate specialized N1 profile: two roots, streamed
  vocabulary head, shared serial attention workspace/cache, exact admission.
- `2B` 32K is currently expected not to fit the 6 GiB device under F32 roots and
  two-root semantics. Refuse it unless a newly derived exact plan fits without
  changing source, precision, ownership, or root semantics.

## Measurement protocol

- One GPU owner; no build, model, benchmark, profiler, or provider overlap.
- One cell at a time; close and verify each provider before the next sample.
- Two repetitions minimum, balanced ABBA then BAAB. One wave is preliminary.
- Keep cold-process and warmed-process campaigns separate.
- Keep prefill and decode separate:
  - TTFT and provider-observed prefill/first-token completion;
  - decode provider commit ITL, client delivery ITL, output-token rate;
  - total offered-work throughput and SLO-goodput only when source/output gates
    pass.
- Record p50/p95 and individual runs, not only a median ratio.
- Retain full per-cell reports and the aggregate; never overwrite inputs or an
  existing result without explicit `--overwrite`.
- Report resource asymmetries. Do not call unequal caps “equal hard cap”.

## Tooling

- `tools/serve-bench.py`: bounded SSE/OpenAI measurements, workloads up to 1,024
  requests and concurrency up to 64.
- `tools/serve-compare.py`: paired source-attested ABBA/BAAB comparison,
  concurrency up to 64.
- `tools/serve-lifecycle.py`: owned provider process/container lifecycle.
- `tools/serve-matrix.py`: serial canonical/ragged C1/C4/C8/C16/C32/C64 plus
  explicitly listed context cells. It retains the complete report for every cell
  as a sibling JSON and writes an aggregate index. It refuses path aliases and
  existing outputs unless `--overwrite`.

The matrix tool is orchestration, not qualification. Its fixed short list includes
C32/C64, so do not invoke it until those cells are authorized, or provide a
separate staged matrix after extending the strict schema deliberately.

## Pending tooling needed before long-context timing

- Retain and validate terminal prompt-token and total-token usage.
- Bind provider-side prefill/first-token timing so TTFT is not mislabeled as pure
  prefill compute.
- Prepare strict canonical, ragged and exact-token context workload files under
  the XDG state run directory, not production source.
- Prepare pinned short and long provider lifecycle inputs without copying the
  retained package's historical receipt machinery.

## Historical baseline, not current acceptance

Pinned preliminary single-wave output-token rates (tok/s):

| model | C1 ZG/vLLM | C4 ZG/vLLM | C8 ZG/vLLM | C16 ZG/vLLM |
|---|---:|---:|---:|---:|
| `.8B` | 67.7432 / 38.8470 | 95.6083 / 42.8328 | 92.9820 / 54.8241 | 111.1738 / 98.2827 |
| `2B` | 45.0305 / 37.7431 | 80.8278 / 50.2514 | 80.7265 / 62.4442 | 84.1017 / 96.8627 |

These results have no confidence interval, broad parity, equal-hard-cap, p99, or
latest-vLLM claim. The `2B` C16 loss must be diagnosed, not hidden.

## Acceptance output

For every cell publish:

- exact source/tokenizer/runtime/binary/device identities and launch flags;
- prompt/output IDs, text digest, usage and finish agreement;
- actual active/cohort widths and queue outcomes;
- complete lifecycle and source/resource closure;
- TTFT/prefill, decode ITL/rate, total throughput and qualified SLO-goodput as
  separate fields;
- all samples plus p50/p95, failure classification and explicit limitations.

A cleanup pass or the last passing cell cannot overwrite an earlier failed or
interrupted scope.
