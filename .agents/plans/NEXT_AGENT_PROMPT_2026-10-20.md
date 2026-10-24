# Starting prompt — Zerogravity comprehensive takeover, 20 October 2026

Copy the following text to the next agent. It preserves the current
planning/investigation-only scope. Add an explicit new instruction if you
want that agent to resume implementation.

~~~text
Take over the EXISTING Zerogravity Rust repository:

  /home/pvl/spaces/abird/src/zerogravity

Use @dotlink-l5 if its tools are available; otherwise identify the actual
available shared-workspace access. Never claim connector execution without it.
This is an established implementation, not a new architecture or engine.

CURRENT SCOPE: onboarding, investigation and planning only. Implementation is
PAUSED. Earlier broad implementation approval and older takeover prompts do
not override the latest pause. Do not change code, build/run models, perform
GPU qualification or benchmarks, acquire dependencies/models, or start
implementation until I explicitly resume it.

Read progressively, without truncating or skipping sections:

  .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-20.md

Use its reading map and bounded consecutive line ranges; retain your cursor
through context compaction. Re-read any truncated range. It contains the current
checkpoint, all new numerical/model/serving investigations, the exact previous
handoff, the full correctness/performance plan, current TODO snapshots and the
starting prompt. Its dated appendices are history, not current instructions.

Then read:
  AGENTS.md
  .agents/plans/README.md
  .agents/plans/TODO.md
  .agents/plans/numerics-and-context/{TODO,CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20}.md
  .agents/plans/model-execution/{README,TODO}.md
  .agents/plans/serving-and-speculation/{README,TODO,FUSED_GPU_BATCHING}.md
  .agents/plans/memory-and-streaming/{README,TODO}.md
  .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md
  .agents/plans/IMPLEMENTATION_ROADMAP.md

Read the remaining live plan TODOs and canonical contracts indexed by the
handoff, including numerical acceptance, minimal V4 spine, source/state/resource
ownership, joint serving/goodput and the 36 vLLM acceptance gates.

For complete original V1–V5.1 design, decisions, research and earlier steps:
  .agents/docs/handoff-2026-10-20/CONTEXT_CATALOG.md
  .agents/docs/handoff-2026-10-20/HISTORY_DESIGN_AND_COVERAGE_AUDIT.md

The catalog locates byte-complete canonical sourcebooks, all retained run
Markdown, draft patches, important uncommitted source and recovered historical
bodies under .agents/docs. Read the original source or its exact preserved
body, not just its catalog title. The original 94-source nested archive is
preserved; this does not restore the missing original33-node Git history,
later129/191-source aggregates or every old report. Do not fabricate them.

REVALIDATE READ-ONLY:
- HEAD, branch, index, pending tracked/untracked changes, worktrees and owners.
- Existing Cargo/rustc/server/container jobs and permitted-device point status.
- Relevant current code, actual failed/interrupted tests and accepted artifacts.
- Do not fetch, reset, remove worktrees, overwrite drafts or stage others' work.

Checkpoint at handoff: signed main fb68d8fa165de16e21d1bd37afef3790b80da971,
34 ahead of local origin/main, empty index,29 worktrees, preserved dirty MAIN.
The root README is user-authored: leave its bytes unchanged.
The MAIN ordered-SiLU candidate compiled but is UNACCEPTED; the prepared60-case
physical matrix never started. target/release/zerogravity is that candidate,
not the accepted benchmark baseline. Preserve root-bf16-composition-20261020
and root-default-bulk-20261020 plus every other agent-owned tree.

CURRENT TRUTH:
- Both approved small GGUF models have physically qualified complete native
  CUDA B2/B4/B8 graphs, ragged frontiers, device feedback and independent roots.
  Full temporal prefill remains incomplete.
- Fresh .8B/2B AR1 and explicit reference-control8K comparisons match full logits
  and all42 domains. Default bulk/historical divergence remains open.
  The fresh earliest differences occur BEFORE GDN: .8B layer3 attention AV,
  2B layer0 alpha/beta projection.
- Pinned vLLM is the primary new CUDA arithmetic target. CPU/GGUF, Torch CPU
  and vLLM loaded/cast/quantization schedules differ; do not blend them.
  Latest source research identifies conv/SiLU, phase-specific Q/K, beta,
  scale, gated norm and chunk64/WY boundaries. No single universal GDN bug
  or completed native/vLLM numerical parity has been proven.
- Original27B source and genuine isolated trained operators exist. Complete
  model/all112 domains, original trained GPU and streamed inference do not.
- Genuine pinned vLLM eager benchmarks exist and are substantially faster
  in the measured bounded C1 workload. They do not establish matched numerical
  quality or optimized default-route/warm/concurrent parity.

WHEN IMPLEMENTATION IS EXPLICITLY RESUMED:
Follow the reviewed correctness-first plan: actual first differing operation →
complete GDN initial/cached layer → full .8B/2B/27B parity → measured shared-engine
performance. Use the same MathematicalBackend/ForwardPlan, immutable source,
TaskCoordinator/SessionAuthority, ResourceArbiter and physical GPU completion/
quarantine owners. Do not create another engine or relax failed tolerances.

Then pursue temporal prefill, efficient projections, fusion, production-safe
graphs, state traffic and fair scheduling. Use tools/serve-bench.py and
tools/serve-compare.py for genuine source/profile/hardware-matched comparisons,
offered/completed work, committed SLO goodput, actual token ITL, memory, fairness
and physical dispatch counts. Claim only measured workload-specific wins.

Preserve the other priorities: source-verified SSD/RAM/GPU streaming, Gemma4
E2B/26B-A4B/31B, other families, speculation, multimodal and distributed work.
Use subagents with disjoint scopes and one MAIN Cargo/test/GPU coordinator.
Keep current TODOs authoritative; archives are not duplicate work queues.

Implementation acceptance, once authorized, requires focused tests, strict
Clippy, physical source/state/cancel tests, both approved original release
parity routes and:
  CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh

Preserve Renoir/gfx90c AMD launch hold and root README. No push/tag/deploy,
device privilege/recovery changes or destructive Git operations without the
applicable explicit authorization. Commits must be signed and owned.

After onboarding, report exactly what is proven, what remains unfinished and
the smallest recommended next correctness step. Remain planning-only unless
I explicitly resume implementation.
~~~
