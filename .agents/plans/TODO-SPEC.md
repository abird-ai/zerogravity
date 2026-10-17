# TODO spec — plans, checklists and evidence

The rulebook for work tracking in zerogravity. Read it before opening a plan.
Older documents (`.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md`, the master
plan, status ledgers) remain design/history references, not tracking duties. The
git history is the evidence, and a user-approved checkpoint commit is the
accepted proof of record.

## Plan layout

A plan is a directory under `.agents/plans/<plan>/`. `TODO.md` is required; the
optional `README.md` is the single home for everything else:

- `README.md` — the plan details: what it is and why it exists, plus goals,
  scope, and links, and sections as needed:
  - **Design** — interfaces, invariants, alternatives considered, migration and
    rollback.
  - **Decisions** — user decisions that bind the plan, with their rationale, so
    agents follow them instead of re-litigating settled questions.
  - **References** — research and source references: papers, specs, prior art,
    source files, benchmarks, with enough provenance to reproduce the reasoning.

  Open it with the plan's authorship: when human-led, `Author: <user>` plus
  `Co-authors: <comma-separated AI models used>`; when AI-led,
  `Author: <the AI models only>`. Keep it one file while it stays readable. When
  a section grows large, split it into `DESIGN.md`, `DECISIONS.md`, or
  `REFERENCES.md` beside `README.md` and link back from `README.md`.
- `TODO.md` — the detailed todo: implementation, tests, verification, gates, and
  evidence. Required, and it needs to be extensive — every step, test, and gate
  gets its own item. Completed work is grouped under the commit that landed it;
  work without a commit yet goes under `## Uncommitted`.

```text
.agents/plans/TODO.md                      index: one checkbox and link per plan
.agents/plans/<plan>/README.md             plan details (optional)
.agents/plans/<plan>/DESIGN.md             detailed architecture (optional)
.agents/plans/<plan>/DECISIONS.md          user decisions (optional)
.agents/plans/<plan>/REFERENCES.md         research and source references (optional)
.agents/plans/<plan>/TODO.md               the plan's implementation checklist
.agents/runs/<plan>-<agent-id>/TODO.md     one agent's working checklist (in flight)
```

Keep plans flat: a plan is a top-level directory under `.agents/plans/`, not
nested inside another, and one flat plan per effort is preferred while its
`TODO.md` stays tractable. Only an extremely complex or large design splits into
smaller plans that stand on their own — added as sibling plan directories, with
the larger plan's `TODO.md` referring to them by link instead of carrying every
item.

## TODO shape

```markdown
# TODO — <plan>

Status: <active | done | blocked>
Provenance: <base-sha>            # commit the plan started from
Checkpoint: <checkpoint-sha>      # user-approved tip, when closed

## Committed
<!-- grouped by the commit that landed each set; newest last -->
### `<sha>` — <what this set landed>
- [x] <item>
- [x] <item>

## Uncommitted
<!-- the working list; the only place new work appears -->
- [ ] <item> — <owner / reason>
- [~] <item> — in progress
- [!] <item> — blocked outside this repository

## Tests and verification
<!-- every test and gate the plan must pass; all [x] before Status: DONE -->
- [ ] <test or gate> — <command / evidence>

## Deferred
<!-- optional; accepted, fail-closed, with the reason -->
- [ ] <item> — <reason>
```

- Completed work is grouped under a `### <sha>` heading; the heading is the
  provenance for every item beneath it.
- An item moves from `## Uncommitted` into a `## Committed` group only after the
  commit exists and its gates pass. Never reorder or rewrite a committed group.
- `## Tests and verification` lists every test and gate the plan must pass, each
  with the command or evidence that proves it. A plan is not done while any of
  these is open; all must be `[x]` on the closing commit.
- Tick an item only after its checks pass, on the commit you are marking. The
  commit and its passing gate are the evidence. Never label planned support as
  implemented or unverified results as source-equivalent.

## Lifecycle

1. **Open** — create `plans/<plan>/TODO.md` with a `## Tests and verification`
   list, add its line to `plans/TODO.md`.
2. **Work** — each agent keeps its own checklist under
   `runs/<plan>-<agent-id>/TODO.md`.
3. **Commit** — add a `### <sha>` group under `## Committed` and move the
   items it landed into it; unfinished work stays in `## Uncommitted` with its
   reason.
4. **Integrate** — the coordinator merges a finished run into the plan TODO in
   one serialized write, then deletes the run file.
5. **Close** — when `## Uncommitted` is empty and every `## Tests and
   verification` item is `[x]`, check the plan's line in `plans/TODO.md` and
   record the checkpoint sha.

## Checkpoint rule

At a user-approved milestone, record the sha as the plan's `Checkpoint:` and
delete prior evidence files; that checkpoint is the accepted source of truth.

## Gates

`scripts/check.sh` (under
`CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
for user/network isolation) is the gate. GPU/model changes also require their
relevant real source, state, cancellation and physical tests. The gate result is
not stored — the commit is the proof.
