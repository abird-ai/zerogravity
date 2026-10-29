# Fabric vNext — COMPLETE transition handoff index, 27 October 2026

**Side architecture/implementation docs, not an active implementation queue.** Current source, tests, active plans, Git state and physical GPU/model qualifications are untouched by preparing these files.

## Complete single-file handoff

[All current transition documents AND the previous complete vNext plan reproduced verbatim](FABRIC_VNEXT_TRANSITION_COMPLETE_REFERENCE_2026-10-27.md)

This one file includes **all 12 latest transition design/handoff files and all 7 original frozen architectural research files**, each with original full text, not just references. Newer transition-phase instructions appear first and take priority over older archive's "await resync" wording. For next source work, agent should recheck HEAD and get explicit implementation permission.

## New current implementation plan

- [Next-agent prompt and full handoff](../plans/fabric-vnext-transition/HANDOFF.md)
- [Overview, latest user decisions, structural-first architecture](../plans/fabric-vnext-transition/README.md)
- [Current Git/refactor source reconciliation](../plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md)
- [Complete crate/module graph and public APIs](../plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md)
- [Exact codec/quant/numerical/derived source design](../plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md)
- [First compiled full Rust skeleton and source relocation](../plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md)
- [Commit-by-commit implementation phases and acceptance](../plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md)
- [Existing features to preserve, actual support versus gaps](../plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md)
- [Cheap per-phase tests versus consolidated Q0/Q1/Q2 qualification](../plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md)
- [Future serving, codecs, models, multimodal, SSD and distribution](../plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md)
- [Accepted/rejected decisions, risks and completion criteria](../plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md)
- [Held per-phase implementation checklist](../plans/fabric-vnext-transition/TODO.md)

## Previous frozen design still preserved

[Original full architecture and previous research archive](FABRIC_VNEXT_COMPLETE_REFERENCE_2026-10-27.md) — includes the prior complete llama.cpp/vLLM/zerogravity source analysis, full abstract architecture, library-first user amendments, old architectural decisions and former handoff. The new complete file ALSO reproduces all its seven constituent original source files.

## Component exact text integrity

Digests below are **documentation-file bytes only**, NOT model/kernel/trained-source qualification. Compiled bundles/hardware claims must be independently tested.

| File | UTF-8 bytes | SHA-256 |
| --- | ---: | --- |
| .agents/plans/fabric-vnext-transition/HANDOFF.md | 18909 | cc264082d71baa5e0fa5424c2669af6a424120cb4e88f83f06d6b630a69cbc04 |
| .agents/plans/fabric-vnext-transition/README.md | 16297 | b1c7449f81b5fd7817a1a7beced17a3a4a2823d002f1be7cc04de94ab2aa3582 |
| .agents/plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md | 14309 | eff83eb3538ea31bc90455c72cf99e4f63042fafc730dfaed62c540dbcbc0a93 |
| .agents/plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md | 14603 | dfc2ac70b2e2f16116577c12d87629fa8e5e8d667bc11f35e40bfcf7bdcde69c |
| .agents/plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md | 19292 | 208060ae97186c14e77a3a9b6f58ce727b0707021bb0fd2cd83de6eef594513f |
| .agents/plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md | 16925 | fc95204a409f2e6d66bbc8f62b49ddd1950310376dd33e65721fef1a069ef9a2 |
| .agents/plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md | 13489 | beec474d1b8d18de7a17b532fd1183aa116cb792a1ba6f94971e288e94005e1a |
| .agents/plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md | 26243 | 26d8885ac5b98971d5541c22a6f3162d270b775263f7bbb332b2c9b5b93c7096 |
| .agents/plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md | 16401 | 9b3e83e3a131461a8b13c99098e9533cb32e47513f6eb3d543d428210f86cb7a |
| .agents/plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md | 15742 | 3335ac31c9b119721b79f866b26450b37d0dd4f919943131f6f42dbf402360cb |
| .agents/plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md | 16922 | 1d474de44c619573579a1cf610723f2e08d9b94fbf3f0c049d6aa7cb2d407a23 |
| .agents/plans/fabric-vnext-transition/TODO.md | 11461 | c27d8e4e6ca1323a8a6ee66e2c12ee8f46e0538f0deb940d3bcdb7e39d50ff96 |
| .agents/plans/fabric-vnext/HANDOFF.md | 11914 | 23e87d87333bdd1cc5d5a893680caed11a96f69b0807930858b354376282dfe7 |
| .agents/plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md | 21159 | 0a5dcfc66fdc03ff8c7d8f05aa115bd7ed9e1a065d402a2b0084757ac9613734 |
| .agents/plans/fabric-vnext/README.md | 45064 | 01c278ed1ba84c7c43be0196822f2135c782bbb56ed19b1589d7f49b5d5dc4cd |
| .agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md | 25619 | 5203a8a7bad766f0cab92d5db59512da4780c7be97729905402b9224a45ffaea |
| .agents/plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md | 45401 | 0ec6fdf3d07407a840a92d72f58aba16bfe620c1fdfcee9fb250bd139f303656 |
| .agents/plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md | 39617 | dc4654c636daa35236591e768134a132f8600a79826596e67d11dda5df8d8478 |
| .agents/plans/fabric-vnext/TODO.md | 7758 | 1ee4ed90098170a00879ebdca35142f4b8156a8d5dcd7d1d0c559e653ea95c68 |

This index and combined copy are documentation-only records. The source plan files remain the canonical editable materials; the combined file is a full archival read convenience. Original frozen files are not modified.
