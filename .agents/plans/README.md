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
The latest [comprehensive takeover through OA-4B](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-27_OA4B.md)
§§0–13 describes original architecture, accepted Phase 1/UX-P and serving
OA-1–3/OA-4A/B, actual evidence, limitations and next coherent work.
The [OA-4C takeover prompt](../docs/NEXT_AGENT_2026-11-27_OA4C.md) is
historical; read the live current handoff for accepted OA-4C and next work.
Current unit: [remaining OA-4 response/conversation resources](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-4--responseconversation-resources):
exact input-token count, then bounded background stream/replay/resumption.
OA-4B background polling/cancel and OA-4C conversation/item resources and
atomic append are accepted; deeper M2 inspection stays optional and
capability-bound. §10.2 records remaining packaging and physical
qualification caveats (old P4B blockers resolved).
Appendices A–E preserve the engineering index, unresolved checklist,
complete reachable Git ledger, every post-M1 landing and plan delta,
and the complete previous main body. The retained
[UX-P-M1 pack](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md),
[UX-9B pack](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md),
[Phase 1 exit pack](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md),
[25 Nov foundations](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md)
and [24 Nov foundations](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-24_FOUNDATIONS.md)
link the original October design, exact chronology and accepted/rejected
experiments. Old prompts are historical. Read the new main body first,
then relevant appendices. The [full ordered plan](FULL_IMPLEMENTATION_PLAN.md),
live source, executed tests and owning TODOs remain authoritative.
The handoff is a requested navigation snapshot, not a separate tracking system.

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
