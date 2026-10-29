# Current status and tracking

This file is the stable status target used by the repository README.

## Authority

Current source and tests are the implementation truth. The active work queue is
[TODO.md](TODO.md), with plan conventions in [README.md](README.md). Git history
retains superseded plans, handoffs, checkouts and run evidence; they are not
live authority.

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
