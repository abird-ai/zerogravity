# Working in Zerogravity

Start with [.agents/plans/TODO.md](.agents/plans/TODO.md), choose the relevant
plan, and read its `TODO.md` plus the specifications it links. These TODO files
replace the old handoffs, status ledgers, evidence workflow and SHA inventories.
Older documents remain design/history references, not extra tracking duties.

## One checklist per plan

- `.agents/plans/TODO.md`: one checkbox and link per plan. Check it only when
  that whole plan is implemented and tested; partial milestones stay in its file.
- `.agents/plans/<plan>/TODO.md`: the main implementation and test checklist.
  Record a known provenance commit, an `Uncommitted` section for current work,
  and completed work/results under their actual commit IDs.
- `.agents/runs/<plan>-<session-or-agent-id>/TODO.md`: each agent's own working
  checklist. Include its provenance commit, main-plan link, owned scope, tests,
  results, failures and unfinished items. Update it as work proceeds so another
  agent can continue an interrupted run. Do not edit another agent's checklist.

Use ordinary Markdown checkboxes and short notes. A test item says what to test
and what passes; check it only after actually running it. Record the command and
result briefly. Failed, blocked and unrun work stays unchecked with its reason.
Keep a useful diagnostic log beside the run TODO when needed. No mandatory run
receipts, source snapshots, per-file hashes, archives or generated handoffs.
Actual model/content hashes and numerical acceptance limits remain part of the
implementation wherever they are needed; this changes only work tracking.

## Integrate, commit and continue

Agents work in their own run files. When an agent finishes its assigned work,
the coordinator reviews its changes/results and incorporates completed and
unfinished items into the latest main plan in one serialized, atomic file
update. Use one writer for that final update; parallel appends must not overwrite
each other. A finished run file can be removed once all useful items are in the
main plan. Keep an interrupted run file until its work is recovered.

Commit verified implementation, tests and plan updates together using explicit
paths. After each commit, record its ID, move completed items/results under that
commit, and carry every unfinished item into `Uncommitted`. That small rolling
TODO update belongs with the next commit; do not create commits merely to record
their own IDs. On a fully finished plan, check its one line in the plan index.

When a plan becomes long, move only completed sections into
`.agents/plans/<plan>/archive/<date-or-milestone>/TODO.md` and link them from the
live TODO. Keep unfinished work in the live file. Git provides source history;
there is no separate archive verification/sealing process.

## Implementation boundaries

Preserve one Rust mathematical engine, immutable source identity, independent
StateRoot ownership, the physical ResourceArbiter, fair workers and actual
completion/quarantine. Run checks appropriate to the change and the complete
`scripts/check.sh` gate before accepting an implementation phase. Use
`CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
for actual user/network isolation with already approved inputs. GPU/model changes
also require their relevant real source, state, cancellation and physical tests;
compilation or mocks do not establish inference support. Inspect changed code
and failed/interrupted tests when resuming; Git IDs are source references.

The root README is user-authored: leave its bytes unchanged. Do not download
models/SDKs, change device privileges or recovery, push, tag or release without
applicable user authorization. The local Renoir/gfx90c AMD launch hold remains.
Use subagents for independent implementation/review/test work; the main agent
coordinates, reconciles, decides and merges. Preserve unrelated user work.
