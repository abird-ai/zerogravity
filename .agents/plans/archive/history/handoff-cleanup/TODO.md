# Simplify agent handoffs

Provenance commit: `bfe672c`.

## Uncommitted

None for this completed plan. Continue [model acquisition](../model-acquisition/TODO.md).

## Committed

### bfe672c — simple TODO handoffs

- [x] Replace the handoff process with the owner's plan/run TODO hierarchy.
- [x] Define per-commit completed sections and a rolling unfinished checklist in root AGENTS.md.
- [x] Remove obsolete archive/generator/checksum scripts and their check.sh calls; preserve implementation tests.
- [x] Consolidate all material pending work into the plan index and per-plan TODOs.
- [x] Retire redundant handoff/status/provenance records and fix live navigation.
- [x] Reconcile independent agent checklists into this plan in one coordinator update.
- [x] Validate links, preserved architecture/source/root README and full repository checks.
- [x] Commit the cleanup, record its commit and finish this plan in the index.


### 3787519 — starting implementation

- [x] A1 cache implementation and tests exist. This cleanup changes only
  work tracking, documentation and obsolete tracking tools; it adds no inference
  capability and changes no mathematical engine or artifact implementation.

## Integrated agent results

- Plan inventory: nine plans retain all material P00–P23 work and original
  numerical/source/model/hardware limits. A1 is complete; A2/A3 are next.
- Tooling: removed five archive/generator/SHA tools and their four check.sh
  invocations. Exact comparison retained every real implementation check.
  Shell syntax, ShellCheck and AST parsing of18 remaining Python scripts passed.
- Preservation review: generated aggregate duplicated191 individual sources.
  Removed that aggregate and generated manifests; kept substantive current
  design documents and static V2/V3/multimodal historical versions. Authored
  retired handoff/status bodies remain available in Git at `3787519`; old
  entrypoint paths are small compatibility links to the TODO index.
- Navigation: all links in live index,11 plan TODOs, AGENTS.md, contributor
  guide and public document index resolve. No Rust, Cargo, GPU tools, model
  fixtures, Nix, CI or root README source has changed in this cleanup.
- Old A1 evidence recorder/patches/receipts are retired. Its actual194-test
  completion, publication/recovery cases and important limits are retained in
  model-acquisition/TODO.md and the A1 technical reference.

## Final checks

- [x] `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`: exit0;194 Rust executions passed,27 optional executions ignored. Strict Clippy, release build, installer, shell/Nix formatting, reference self-tests and18 native-wrapper fixtures passed.
- [x] One-off local-link check of35 current/new/modified documents: all targets resolve. The first pass found seven old pre-layout source paths in requirements; corrected to their crate owners and reran successfully.
- [x] Both independent final reviews: all11 plans preserve scope and pending work; all44 links in the13 TODO/workflow documents resolve. No review blocker.
- [x] Root README bytes and all crates/tests/tools/Cargo/Nix/CI sources match `3787519`. Only tracking docs, attributes and obsolete tracking scripts changed.
- [x] Finished per-agent run notes are integrated above and retired. The next agent starts from the plan index, not those completed working files.

Next implementation: model-acquisition/TODO.md — A2 shared bounded HTTPS,
pinned Hugging Face/GitHub resolution and explicit CLI, then A3 loader bridge.
