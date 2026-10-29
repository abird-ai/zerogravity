# Current status and tracking

This file is the stable status target used by the repository README. Cross-window context starts at [CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).

## Authority

Current source and tests are the implementation truth. The active work queue is
[TODO.md](TODO.md), with plan conventions in [README.md](README.md). Git history
retains superseded plans, handoffs, checkouts and run evidence; they are not
live authority.

## Reviewed implementation baseline — 29 Oct 2026

- Reviewed implementation/source baseline: `ae8cd89`; later handoff-document commits do not change execution support.
- Ordinary Qwen3.5 `.8B`/`2B` CUDA is integrated under the existing resident owner but remains qualification-only; production routing still refuses before source/device/native load.
- Approved `.8B` source revision `2fc06364715b967f1860aea9cf38778875588b17` is visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`. The 1,746,942,600-byte safetensor hashes to `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- No complete current/legacy Qwen CUDA native bundle remains under the repo or user cache after transient run cleanup. Some GDN CUDA sources are tracked, but the complete bundle also depends on captured/precompiled module images. Reacquire/regenerate it through the maintained qualification/capture path; do not restore deleted run trees as authority.
- Source-free ordinary cancellation/fault/shutdown ownership, exact context admission and serving/cache/acquisition gates are current; real source+native physical parity/lifecycle/context qualification remains open.
- Fresh review fixed environment-sensitive PID1 serving shutdown control and added the maintained Qwen27 layer9 source-contract suite to `scripts/check.sh`.

## Repository topology

- Persistent checkout: main only.
- Local branch: main only.
- Accepted implementation is committed on main.
- Transient agent/run artifacts are not tracked as provenance.

## Current product boundary

The root README's implementation-status table is the concise public capability
summary. Detailed open/closed work remains in the subsystem TODOs linked from
[TODO.md](TODO.md). A capability is accepted only by current source plus its
tests/physical qualification where required; model names, metadata, old run
receipts or historical branches do not grant support.

## Current architecture direction

The maintained architecture transition is under
[fabric-vnext](fabric-vnext/README.md) and
[fabric-vnext-transition](fabric-vnext-transition/README.md). Current code wins
over older architecture prose when they disagree.

## Updating status

Land implementation and tests in logical signed commits, update the owning
subsystem TODO when the status materially changes, and keep this file concise.
Do not add generated handoffs, branch inventories, checkout paths or transient
run directories as status authority.
