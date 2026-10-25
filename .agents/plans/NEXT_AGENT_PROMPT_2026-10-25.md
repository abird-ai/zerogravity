# NEXT AGENT PROMPT — 2026-10-25

Take over `/home/pvl/spaces/abird/src/zerogravity` (ZeroGravity: Rust AI execution
fabric; goal: beat vLLM on measured throughput with a fully composable
primitives→pipelines→profiles architecture across CPU/CUDA/HIP/Vulkan).

## Read first (in this order)

1. `AGENTS.md` — repo rules. Keep the safety/correctness rules (signed commits,
   no push/destructive ops without explicit user confirmation, preserve other
   work, root README bytes never change, no downloads/privilege changes without
   authorization, gfx90c launch hold). **Disregard the previous orchestration
   delegation rules** (Sol/Astra/Pi-agent ceremony) — use your own orchestration.
2. `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25.md` — the full
   handoff: goals, progress, validation status, all user decisions, evidence
   locations, exact resume queue. This is your primary context.
3. `.agents/plans/README.md` then `.agents/plans/TODO.md` (plan queue), then the
   live `## Uncommitted` sections of cpu-gpu-optimization, model-execution,
   platform-and-release, serving-and-speculation, cli-device-routing.

## Situation

The previous session ("Review ZeroGravity handoff", Codex 01a158d8, 2026-10-20 →
2026-10-24) was interrupted mid-flight by a usage limit at 2026-10-24T21:12Z,
along with all its subagents. HEAD is `4bf3673`; main's working tree holds a
large **uncommitted assembled batch** (70 modified + 21 untracked paths: the
five-lane assembly, HIP H14, CUDA11 corrections, queue tokenizer/VM-credit fix,
qualification tools). The five lane worktrees (qwen27b, gemma4, hip, vulkan,
queue under `worktrees/resume-all-20261024-*`) still carry their candidate
changes and remain registered; retire them only after their work is committed
from main.

Keep the 5-lane parallel structure; work the lanes independently and merge at
checkpoints. Work in end-to-end units; don't over-validate per helper.

## Resume queue (in order)

1. Fix the two strict-cli clippy failures in the assembled batch and rerun
   strict-cli: (a) unused import `fixed_size` at
   `crates/zerogravity-cli/src/serve/gpu_engine.rs:9`; (b) collapsible_if at
   `crates/zerogravity-cli/src/run_cli/short_hf.rs:444` (arbiter snapshot
   condition — collapse, don't change semantics). Evidence:
   `.agents/runs/resume-all-cuda8-affected-execution-pi-20261025/strict-cli.err`
   (builds core/gpu/cli/product all rc0; strict core/gpu rc0).
2. Run the complete gate on the assembled main batch:
   `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`.
3. Finish the Vulkan final package: V census predicates are fixed
   (`…resume-all-hgv-checkpoint-execution-sol-20261025/v-census-boundary-binding-1/`),
   but `V_BYTES`/`V_SHA` release constants are still `None` (final five-fix
   adapter SHA `1c12cfe1…76833a6`); produce the `vulkan-final` package; never
   retag old package identities.
4. Execute the prepared H/G/V physical checkpoint on pvl-x2 exactly per
   `.agents/runs/resume-all-hgv-checkpoint-execution-sol-20261025/TODO.md`
   ("Released execution order" and "Bounds / stop policy"): serial H6/G8/V10
   cases, strict SSH, inspect every case's semantic + cleanup result before the
   next; remote names `resume-all-five-{hip,gemma4,vulkan}-20261025-1`; no cap
   growth, no weight transfer/download.
5. If the gate passes: commit the assembled batch as logical signed units with
   explicit paths, reconcile the plan TODOs, then diff each lane worktree
   against main and retire fully-landed lanes (trees + branches). Keep the two
   reference roots and three donor extracts.
6. Implement the shared prepared-B4 certification/state transport per
   `.agents/runs/cpu-gpu-optimization-astra-c16-shared-gap-20261024/DESIGN.md`
   (C16 loss: 2B ZG 78.79 vs vLLM 96.0 tok/s; up to 22.13% dispatch reduction
   is a ceiling, not a claim), then run fresh matched C16 for 0.8B and 2B;
   then genuine C32 and C64 (no C2; same-attempt arms; no cherry-picking).
7. Unblock Qwen27B natural reference capture per
   `…resume-all-queue-production-bound-astra-review-20261024/Q_REFERENCE_ROUTE_DECISION.md`
   (no nested-rootful retries; rootless route needs full original-container
   verification), then numerical acceptance; Gemma4 learned-PLE coverage open.
8. Finish cli-device-routing open items (source-scheme precedence, ambiguous
   variant rejection, dry-run read-only, pre-Client transactional admission).
9. Deferred backlog (user-approved, after benchmarks): graceful + forced
   cancellation with timeout and stuck-loop breaking; NPU routing; DeepSeek and
   other model families (deferred on TODO).

## Hard rules

- GPU/model claims require real source/state/cancellation/physical evidence —
  compilation or mocks prove nothing. Keep evidence scopes distinct; a passing
  last case never certifies earlier cases.
- Preserve the failed/negative results (C16 loss, failed20 first-use refusal,
  diagnostic-only .8B C16) — never silently replace them.
- No 8GiB CUDA HOST fit claim, no Vulkan/HIP physical claim, no Qwen27B
  numerical acceptance exists yet; don't state otherwise.
- User decisions are catalogued in handoff §6 — they bind unless the user
  changes them. The user's standing directive is to continue all five parallel
  tracks autonomously to their mergeable units.
- Report progress on the larger goals, blockers, and next milestones; test
  counts don't substitute for working behavior.