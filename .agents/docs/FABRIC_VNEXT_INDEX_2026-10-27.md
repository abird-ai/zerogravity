# Fabric vNext — frozen design archive index (27 October 2026)

**Read-only architectural reference and handoff.** The user asked to archive the complete research/redesign context before returning to the moving implementation. Do not start resync, code changes, tests or release from this archive until separately asked.

## Single-file complete archive

[Complete consolidated research, architecture, public-library/serving amendments, migration and handoff](FABRIC_VNEXT_COMPLETE_REFERENCE_2026-10-27.md)

This single file contains the **complete texts** of all files in the companion plan. The source plan directory remains the canonical multi-file authoring reference.

## Side plan (canonical)

- [Handoff and exact next-agent starting prompt](../plans/fabric-vnext/HANDOFF.md)
- [Verbatim user decisions, rejected alternatives and open issues](../plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md)
- [Full Fabric vNext architecture](../plans/fabric-vnext/README.md)
- [Library-first public Rust API and standalone serving library](../plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md)
- [llama.cpp/vLLM research plus zerogravity design revalidation](../plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md)
- [Historical source-to-target crosswalk, migration and acceptance](../plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md)
- [Held future-phase checklist](../plans/fabric-vnext/TODO.md)

No changes were made to the active source, tests, existing plans, shared TODO queue or current running implementation. Do not confuse the historical 26 October checkout HEAD 195099ca with the later 27 October observation (eb38dd5e). The actual current code must be reaudited only when the user explicitly initiates the next phase.

## Verbatim component integrity inventory

SHA-256 digests are of these *documentation files only* at the moment of archive creation. They do not attest any model, compiled kernel, GPU or test evidence.

| File | UTF-8 bytes | SHA-256 |
| --- | ---: | --- |
| HANDOFF.md | 11914 | 23e87d87333bdd1cc5d5a893680caed11a96f69b0807930858b354376282dfe7 |
| DECISIONS_AND_OPEN_QUESTIONS.md | 21159 | 0a5dcfc66fdc03ff8c7d8f05aa115bd7ed9e1a065d402a2b0084757ac9613734 |
| README.md | 45064 | 01c278ed1ba84c7c43be0196822f2135c782bbb56ed19b1589d7f49b5d5dc4cd |
| PUBLIC_LIB_AND_SERVING.md | 25619 | 5203a8a7bad766f0cab92d5db59512da4780c7be97729905402b9224a45ffaea |
| PRIOR_ART_AND_REVALIDATION.md | 45401 | 0ec6fdf3d07407a840a92d72f58aba16bfe620c1fdfcee9fb250bd139f303656 |
| IMPLEMENTATION_AND_ACCEPTANCE.md | 39617 | dc4654c636daa35236591e768134a132f8600a79826596e67d11dda5df8d8478 |
| TODO.md | 7758 | 1ee4ed90098170a00879ebdca35142f4b8156a8d5dcd7d1d0c559e653ea95c68 |

The combined copy is made by concatenating the UTF-8 contents in the order shown. If a future author changes an individual plan file, reconcile/regenerate the combined archive only with explicit appropriate documentation authorization and recorded provenance.
