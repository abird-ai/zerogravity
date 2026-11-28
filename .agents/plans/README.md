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
The latest [comprehensive takeover through OA-7D](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-27_OA7D.md)
§§0–13 and Appendix F preserve original architecture, chronology, accepted
work and the recovered OA-7E work. The
[OA-7E takeover prompt](../docs/NEXT_AGENT_2026-11-27_OA7E.md) is historical;
OA-7E landed at signed 1585b579, actual learned SDK and assembled gate passed.
User-priority [UX-11 A/B/C/D presentation, diagnostics and source-aware launch](cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md#28-nov--ux-11-follow-up-user-priority-before-rocm-continuation) source implementation landed at the D checkpoint. Focused CLI/UI/runtime validation passed; pinned native Nix package realization and complete assembled gate remain open. Actual L5 GPU environment and eligibility diagnosis in CURRENT_HANDOFF. Pause measured ROCm work at P3C; resume after outstanding qualification without reopening accepted operator evidence.
Prior unit: [Phase 3 ROCm readiness](FULL_IMPLEMENTATION_PLAN.md#3-rocm--match-vllm-first) on X2 after bounded Phase 2 acceptance and [OA-10 closeout](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-10--compatibility-closeout).
P3A ordinary cold cohort/exact device envelope and comparator usage
completeness implemented at 50f6e276; X2 exact commit sync and locked
offline Cargo restore complete. Full gate 2,603/0/112 (326 suites,
519s) and real 0.8B/2B C1/C8, 27B C1 native public cells pass;
explicit sequential control passed. Pinned vLLM restored and rooted;
all three original models C1/C8 optimized HTTP, 81/81 exact requests
and physical retirement pass. P3B compact feedback and P3C exact RMS
row regions/flattened GDN gate physically validated. Fresh paired
0.8B/2B C1/C8 before region optimization; native A/B 47+3/16 gains
up to 20%. Fresh 27B C8 pair at 54b2c910, 5.8652/13.51425 tok/s,
32 exact timed requests and joined retirement. 27B C1 pair and
assembled gate pending (P3C 1785/0/97 in 237 suites, 1200s
timeout during serving-only rebuild); next measured
temporal/recurrent and attention regions.
OA-9A bounded native text Realtime accepted at 4735621b. OA-10 corrected
cold endpoint markers/help, documented actual supported routes and
capability limits, and focused-qualified safe error/refusal behavior.
SDK audit: passed (installed SDK2.33.0, cached CLI/native/attached-Web 125.30s, Realtime 20.05s). Assembled: 2,603/0/112 (326 Rust suites, 463s), complete isolated gate passed.
Ordinary public benchmark: actual CPU Qwen0.8B canonical listener,
all three routes exact native output/usage/finish,16/16 and native
peak2; source-free benchmark/comparator/lifecycle/matrix pass.
Final 2,603/0/112 (326 Rust suites, 235s), complete isolated gate passed. Next ordered ROCm on X2.
OA-6 trained MiniLM, OA-7 synthesis/streaming and OA-8 image generation
require actual authorized source and capability acceptance. No new model download.
OA-4 fully accepted after the serial resident lifecycle retry; OA-5
focused/SDK/assembled qualified. Deeper M2 inspection stays optional and
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
