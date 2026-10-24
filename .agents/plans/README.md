# Plans

This folder is the agent planning entry point. Start here, then work from
`TODO.md`. Git is the authoritative history (author, time, diff, provenance sha);
this folder keeps the live queue and per-plan checklists.

## Layout

```text
.agents/plans/
  README.md                 overview + spec
  TODO.md                   queue: Active / Icebox / Closed
  <plan>/README.md          plan details + author/co-authors
  <plan>/TODO.md            Base + Committed + Uncommitted + Tests
  <plan>/HISTORY.md         optional chronology (heavy plans)
  archive/<year>-Q<n>.md    quarterly rollup of closed plans
```

Plans are directories, flat under this folder; do not nest plans. One plan per
effort; split into sibling plans only when a design is too large for one TODO.

## Queue — `TODO.md`

One line per plan:

```markdown
## Active
- [~] P0 <plan> — @owner — <one-line> — [details](<plan>/README.md)

## Icebox
- [ ] <plan> — [details](<plan>/README.md)

## Closed
- [x] <plan> — <closing-sha> — <date>
```

- Active is sorted P0 → P2; moving a plan from Icebox to Active is the act of
  prioritising it.
- `[ ]` open · `[~]` active · `[x]` done · `[!]` blocked.
- P0 requires human promotion and a named incident coordinator.

## Plan — `README.md`

Open with the plan's authorship and ownership:

```markdown
Author: <user>
Co-authors: <comma-separated AI models used>
Owner: @<user|agent>
```

Then what the plan is and why it exists, plus **Design**, **Decisions** and
**References** sections as needed. Split a section into `DESIGN.md`,
`DECISIONS.md` or `REFERENCES.md` only when it grows large, and link back.

## Plan — `TODO.md`

```markdown
# <plan>

Base: `<sha>`

## Committed
### `<sha>` — <what this set landed>
- [x] <item>

## Uncommitted
- [ ] <item>
- [ ] <item> — <blocker/reason>

## Tests and verification
- [x] <test or gate> — `<sha>`
- [ ] <test or gate>
```

- `Base:` is the commit the plan started from.
- Completed work is grouped under the commit that landed it; append new groups.
- Tests are a flat list; tick each with the sha when the commit's provenance
  group is added.
- A plan is done when `## Uncommitted` is empty and every test is `[x]`.

## History — optional `HISTORY.md`

For heavy or multi-agent plans, keep an append-only chronology:

```markdown
- 2026-10-18 09:14 — @owner — opened; claimed P2
- 2026-10-18 09:40 — abc1234 — implemented parser; tests pass
- 2026-10-18 10:02 — @reviewer — reviewed; requested X
```

Light plans can skip it; the commits are the history.

## Parallel work

The owner on the queue line owns the plan directory; no one else edits it.
Parallel agents keep scratch under `.agents/runs/<plan>-<agent>/TODO.md` and
append back only when finished. The coordinator reconciles the queue line and
the plan's committed groups.

Pi implementation/execution/review launches use
`--provider ollama-cloud --model deepseek-v4.1-flash:1017 --thinking xhigh`.
Return only the concise final report. Redirect intermediate messages, reasoning,
tool calls/results and diagnostics to run-local logs; never import raw event or
session streams into coordinator context. The coordinator separately reviews
changed source and concrete verification evidence before accepting work.
Prefer `--print --mode text`: installed Pi emits only the final assistant text,
excluding thinking/tool events, and reports truncated/error completions as
failures. Keep stdout/stderr local; forward only the final report. JSON event
mode is private diagnostic storage, never coordinator input.

Implementation Pi runs start in their actual candidate directory, with MAIN and
other worktrees mounted read-only; candidate and owned run are writable. A
prompt or shell `cd` is insufficient isolation: a24Oct worker edited MAIN and
the original physical source guard correctly refused the batch. Preserve stray
edits before restoring a verified baseline. Keep Pi provider credentials/settings
in a private temporary config when locks need writes; never expose their values
or change global configuration. No GPU devices in preparation-only namespaces.
Retain the current user ID in this outer namespace; do not force UID0/GID0
and drop all capabilities, which prevents the mandatory nested native-gpu
user/network mapping. The current-user readonly boundary passes that wrapper.
An isolation setup failure stops checks; direct execution cannot replace it.
Search with `rg` in named source directories; bound exploratory searches and
exclude `target`, weights, raw tensors and Pi event/session streams. Do not run
recursive `grep` across the repository: a24Oct executor spent over312CPUseconds
there before benchmark launch. Read explicit reviewed paths before searching.

## Closeout

All tests `[x]` and `## Uncommitted` empty → move the queue line to `## Closed`
with the closing sha and date. Quarterly, roll older `## Closed` entries into
`archive/<year>-Q<n>.md` and prune their plan directories; git retains them.
