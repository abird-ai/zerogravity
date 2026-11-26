# Plans

**Current delivery order:** [Full implementation plan](FULL_IMPLEMENTATION_PLAN.md) — foundations → CLI/TUI/Web → ROCm parity → CUDA parity → Vulkan → optional compiled providers. This supersedes older ordering in linked designs; detailed owner TODOs and source/quality/lifetime constraints remain authoritative.

This folder is the agent planning entry point. Start here, then work from
`TODO.md`. Git is the authoritative history (author, time, diff, provenance sha);
this folder keeps the live queue and per-plan checklists.

Current source, tests, and TODO.md are authoritative. Git retains retired history; deleted takeover snapshots are not live instructions.

Repository naming follows the short canonical vocabulary in
[TERMINOLOGY.md](../docs/TERMINOLOGY.md). New code must name the engineering
concept rather than historical qualification shorthand.

Cross-window continuation starts at [the canonical current handoff](../docs/CURRENT_HANDOFF.md).
For the complete current takeover through accepted UX-9B, read the
[26 Nov comprehensive handoff](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md) §§0–13;
the historical continuation was UX-9C (now accepted). The live handoff
records accepted UX-9D/E and the next coherent unit UX-10.
The [standalone UX-9C prompt](../docs/NEXT_AGENT_2026-11-26_UX9C.md) is historical.
Appendix D preserves every Phase 2 landing. Appendix E and the
[25 Nov Phase 1 exit takeover](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
retain the complete original design, foundation history and prior evidence.
The [25 Nov foundations pack](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md)
and [24 Nov foundations pack](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-24_FOUNDATIONS.md)
remain historical authority for their accepted scope.
Read the main body first and use the appendices and earlier packs as needed.
The live ordered [full implementation plan](FULL_IMPLEMENTATION_PLAN.md),
source, tests and owner TODOs remain authoritative; the takeover is a
navigation snapshot, not a separate tracking system.

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

`main` is the only persistent checkout and branch. On l5, do not create additional
Git worktrees for normal implementation, profiling or review: the canonical
`target/` and `tmp/` trees are very large and must not be duplicated. Parallel
agents may inspect the same checkout or use bounded ephemeral external scratch,
but durable repository checkouts/branches are not coordination state. Reconcile
finished work into main, remove temporary `_tmp` runners/isolated build output
when done, commit logical units, and keep transient logs/artifacts out of tracked
documentation unless they become stable fixtures.

## Closeout

All tests `[x]` and `## Uncommitted` empty → move the queue line to `## Closed`
with the closing sha and date. Quarterly, roll older `## Closed` entries into
`archive/<year>-Q<n>.md` and prune their plan directories; git retains them.
