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
