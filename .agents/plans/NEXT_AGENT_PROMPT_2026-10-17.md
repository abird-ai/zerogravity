# Starting prompt — Zerogravity takeover, 17 October 2026

Copy the following into the next agent. It onboards and revalidates the paused
repository; implementation resumes only when the owner explicitly requests it.

```text
You are taking over the live Zerogravity Rust repository at
/home/pvl/spaces/abird/src/zerogravity. This is an existing implementation,
not a new project or architecture rewrite. If using dotlink-l5 from
/home/pvl/spaces/abird, use ./src/zerogravity as the working directory.

Before implementing anything, read IN FULL:
.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md
It is large: read bounded chunks without truncation, retain working notes,
and follow its complete document/decision/history/plan source catalog.

Then read the current AGENTS.md, .agents/plans/TODO-SPEC.md,
.agents/plans/TODO.md, model-acquisition/TODO.md and README.md, and both
linked completed-TODO archives. Read the relevant core architecture/specs,
all active plan TODOs and the portable paused-draft context referenced
by the handoff before changing their code. TODO files remain the live queue;
the handoff is a dated context snapshot, not a second tracking system.

Revalidate live Git HEAD, branch/index/worktree changes, the two unfinished
SIDE worktrees, active processes/Cargo ownership and the exact source delta.
The last signed implementation checkpoint was f197c9c (explicit GGUF quant),
preceded by f22f759 (candidate discovery). Later handoff documentation and
rolling TODO changes may be uncommitted. Preserve them; do not reset or
blindly copy an old SIDE over MAIN. Some old owner-approved checkpoint labels
no longer resolve after history rewrite. Use the handoff's tested recovery
refs and explicit missing-history limits, not nonexistent original commits.

Current accepted original-model resident GPU/native CUDA/HIP support is
bounded to its documented profiles/512 consumed tokens. CPU functional8K,
independent8K GDN drift, deferred16K/32K and the local Renoir/gfx90c launch
hold remain distinct. Qwen3.8/Gemma4 metadata contracts are not trained
inference. Serving, advanced decoding, streaming and distributed/multimodal
runtime goals remain open. Do not restart the already completed H5 sequence.

C3 filters and C4 native Xet indexes are incomplete, unmerged and untested.
Read their exact paused TODOs and portable source patches under
.agents/docs/handoff-2026-10-17/. C3 has only an unwired, not-compile-ready
filters.rs. C4 has only three draft production files, no CLI integration
or integration tests. Preserve them. On a future explicit resume, finish
and qualify C3, reconcile the accepted baseline, then qualify C4 through
one existing provider/cache/engine path; continue other live plans without
inventing completeness or numerical/performance claims.

Use subagents for independent implementation/review/test work; act mainly
as coordinator, gate, reconciler, decider and merger. Give each agent an
own runs/<plan>-<agent>/TODO.md and explicit paths. Only one MAIN Cargo owner;
no SIDE builds without a serialized grant. Merge bytes with fresh mtimes.
Use the documented locked/offline native user/network isolation and actual
focused/full/source/model/physical checks for the change being qualified.

Preserve one Rust mathematical engine, immutable source identity,
independent StateRoots, one physical ResourceArbiter, fair workers,
actual completion/fencing and quarantine. Never widen tolerances,
substitute CPU after a selected GPU failure or claim unqualified support.
The root README.md is user-authored and must remain byte-identical.
Use the simple TODO workflow; do not restore evidence archives,
SHA inventories or staged tree/blob receipts. Actual model/content hashes
and source/numerical integrity checks remain necessary.

Do not download models/SDKs, modify device privileges/recovery, mounts,
persistent services/live caches, push/force-push/tag/publish/deploy or perform
destructive Git operations without applicable explicit authorization.
Do not launch or reset held local AMD hardware.

Get fully up to context, report the live checkpoint, any source/draft drift
and the next concrete work. Keep runtime implementation PAUSED until I
explicitly request resumption. This takeover prompt authorizes reading
and revalidation, not a new implementation or publication phase.
```
