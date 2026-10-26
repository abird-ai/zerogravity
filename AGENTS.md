# AGENTS.md

<!-- BEGIN common -->

## Shared conventions

These are byte-identical across all abird repositories. Repo-specific rules
follow the markers below.

- **Source of truth.** Live source, tests and executed results outrank dated
  prose. Start at `.agents/plans/README.md` (overview + spec), then work from
  `.agents/plans/TODO.md` (the queue).
- **XDG paths.** Where the project owns user config, data, cache or state,
  resolve it under `$XDG_*_HOME/abird/<project>` — an absolute XDG base, falling
  back to `HOME` when unset. No legacy or alternate root.
- **Shared `.gitignore`.** The block at the top of `.gitignore` is identical
  across repos; repo-specific entries follow it. Keep it mechanical.
- **Preserve other work.** Never revert or overwrite changes you did not make.
- **Owned changes only.** Commit only the paths you changed for the request.
  Before committing, inspect `git diff --cached --stat`; if it lists files you
  did not touch, stop and ask — unless the user explicitly asked you to include
  them.
- **Ownership.** A plan directory belongs to the `@owner` on the queue line, and
  the coordinator alone reconciles `.agents/plans/TODO.md`; do not edit another
  owner's plan or worktree unless the user explicitly asks.
- **Clean pushes.** Do not push from a dirty worktree unless the user explicitly
  asked to include those changes; otherwise commit/stash your own change first,
  or push from a clean temporary worktree.
- **Terse writing.** Keep code comments and docs succinct. Comments need not be
  full sentences; prefer symbols and fragments unless the user asked for an
  explanation or the topic is design-level or otherwise complex. Prefer bullets
  and engineering focus over prose in other docs.
- **Terse output.** Keep replies short and on point, preferably bulleted; avoid
  prose unless the user asked for an explanation. Be terse without dropping
  detail.
- **Git safety.** Commits are signed; never disable signing. Never push,
  force-push, tag, publish, deploy, or run destructive operations — history
  rewrite, `reset --hard`, branch deletion, or touching another agent's staged
  or uncommitted work — without explicit user confirmation for each action. A
  request to "commit" or "apply" is not authorization to push.

<!-- END common -->

<!-- BEGIN repo-specific -->

## Repository

Start with [.agents/plans/TODO.md](.agents/plans/TODO.md), choose the relevant
plan, and read its `TODO.md` plus the specifications it links. These TODO files
replace the old handoffs, status ledgers, evidence workflow and SHA inventories.
Older documents remain design/history references, not extra tracking duties.

Read [.agents/plans/README.md](.agents/plans/README.md) before opening a plan —
it is the rulebook for the layout. A plan is `.agents/plans/<plan>/` with an
optional `README.md` (what it is and why it exists, plus design, decisions and
references sections, opened with its `Author:`/`Co-authors:`) and a required,
extensive `TODO.md` whose completed work is grouped under the commit that landed
it and whose open work sits under `## Uncommitted`.

- **XDG project:** `abird/zerogravity`.

## Plans and parallel work

Architecture, structure and naming follow
[STRUCTURE_AND_NAMING.md](.agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md).
Model semantics belong to model crates; shared operators and physical owners
remain shared. Workload sizes and test ordinals are data, not production names.
Existing boundary debt is tracked in the owning plan; preserve numerical/source
identities and historical evidence while migrating actual consumers.

- `.agents/plans/TODO.md`: one checkbox and link per plan. Check it only when
  the whole plan is implemented and tested; partial milestones stay in its file.
- `.agents/runs/<plan>-<session-or-agent-id>/TODO.md`: each agent's own working
  checklist. Record its parent plan, owned scope, tests, results, failures and
  unfinished items, so another agent can continue an interrupted run. Do not
  edit another agent's checklist.
- The coordinator alone reconciles the shared plan: one writer, one serialized
  atomic update; parallel appends must not overwrite each other. A finished run
  file can be removed once its items are in the main plan; keep an interrupted
  run file until its work is recovered.
- Commit verified implementation, tests and plan updates together with explicit
  paths. After each commit, move completed items under that commit and carry
  unfinished items forward; do not create commits merely to record their own
  IDs. On a finished plan, check its one line in the plan index.
- When a plan grows long, move only completed sections into
  `.agents/plans/<plan>/archive/<date-or-milestone>/TODO.md` and link them from
  the live TODO.

No mandatory run receipts, source snapshots, per-file hashes, archives or
generated handoffs. Actual model/content hashes and numerical acceptance limits
remain part of the implementation wherever needed.

## Execution cadence

- Work in substantial end-to-end units with a concrete user-visible outcome.
  Bindings, readers, helpers and individual layers are usually internal steps,
  not separate acceptance phases or coordination cycles. Isolate a component
  when a concrete failure needs diagnosis, then return to the larger unit.
- Use subagents for substantial independent implementation, investigation and
  review. A correction or validation blocker in one task must not stop unrelated
  ready work; advance other models, backends or features in parallel where safe.
- Respect the user's agent roles and model settings. Architecture/design agents
  review and propose redesigns; the requested implementation agents apply them.
  Keep those roles explicit when delegating and handing off work.
- Current delegation: GPT-6.1 Sol xhigh for coordination, architecture,
  implementation, execution and review, per the user's26Oct instruction.
  Keep file, build and device ownership explicit across independent lanes.
- Return concise final reports. Keep intermediate tool streams and diagnostics
  in run-local logs; review changed source and executed evidence separately.
- Set `PYTHONDONTWRITEBYTECODE=1` for agent processes and use `python3 -B` for
  read-only preflight: imports must not add bytecode to sealed source packages.
- Use focused independent lanes where safe. Optimize time to the larger outcome;
  coordinate shared build/GPU capacity. Give implementers concrete invariants,
  interfaces, checks and completion criteria.
  The coordinator independently inspects code, results and failures before
  accepting or merging; an agent's report alone is not evidence of correctness.
- When fixes become circular or repeated fix/test cycles stall a task, assign
  focused root-cause work in a separate worktree and keep other work moving.
  Coordinate shared GPU, compiler and memory use so concurrent jobs do not
  invalidate results or contend needlessly.
- Use separate worktrees for independent changes or overlapping investigations.
  Declare ownership and interfaces, preserve existing work, and keep the main
  agent responsible for reconciliation and integration.
- Integrate completed, contained changes promptly, even within a larger unit.
  Reconcile active worktrees at periodic useful checkpoints to limit drift and
  catch interface conflicts early; do not wait for every parallel task to finish.
- Commit ready changes in small logical units at useful checkpoints. Fix major
  source/refactor bugs first; track smaller qualification bugs separately.
  Reuse applicable batch evidence instead of inventing per-commit acceptance
  cycles; run the comprehensive gate on the assembled changes.
- After a commit batch, delegate side-worktree reconciliation where independent.
  Preserve unique/unintegrated work and active ownership; synchronize cleanly
  without discarding changes or rewriting history.
- Batch related implementations and parallel contributions before validation.
  Bindings, readers, helpers and individual changes do not each require a test,
  review, physical-regression or acceptance cycle.
- During implementation, use compilation or narrow diagnostics when needed to
  resolve a concrete blocker. Avoid routine validation after every change.
- Validate the assembled batch with the required numerical/source/state,
  physical ownership/fault and serving checks. Gather failures, fix related
  failures together and rerun affected coverage; keep evidence scopes distinct.
- Run the complete `scripts/check.sh` gate last, before accepting the final
  tested batch. Avoid repeated full gates while adjacent planned implementations
  remain. Repeat broad checks only for relevant changes, failures or unresolved
  concerns; preserve all required real-source and physical validation.
- Reuse applicable accepted evidence for unchanged paths. Avoid redundant
  baseline runs and per-helper freeze, receipt or release administration.
- Keep active tool-session jobs with the initiating agent until completion.
  A session ID alone is not an ownership transfer. Verify live supervision and
  result retrieval before ending the old owner; otherwise keep its turn active.
- Inspect every case's semantic and cleanup results before reporting a batch
  pass. Successful cleanup or the last passing case cannot establish earlier
  results; preserve interrupted and failed scopes separately.
- Continue already authorized substeps without requiring another user prompt.
  Report completed end-to-end behavior, the concrete blocker and the next larger
  milestone; test counts and receipts do not substitute for working behavior or
  measured performance.

## Implementation boundaries

Preserve one Rust mathematical engine, immutable source identity, independent
StateRoot ownership, the physical ResourceArbiter, fair workers and actual
completion/quarantine. Run checks appropriate to the change and the complete
`scripts/check.sh` gate before accepting an implementation phase. Use
`CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
for actual user/network isolation with already approved inputs. GPU/model
changes also require their relevant real source, state, cancellation and
physical tests; compilation or mocks do not establish inference support. Inspect
changed code and failed/interrupted tests when resuming; Git IDs are source
references.

When merging source from a worktree, copy bytes with fresh modification times.
Preserving older timestamps can make Cargo reuse a stale library; unchanged
bytes may need a narrow timestamp refresh before rerunning the same affected
checks.

The root README is user-authored: leave its bytes unchanged. Do not download
models/SDKs or change device privileges or recovery without applicable user
authorization. The local Renoir/gfx90c AMD launch hold remains. Use subagents
for independent implementation/review/test work; the main agent coordinates,
reconciles, decides and merges. Preserve unrelated user work.

<!-- END repo-specific -->
