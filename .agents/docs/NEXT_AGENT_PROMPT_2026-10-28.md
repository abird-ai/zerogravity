# Next agent prompt — 28 Oct 2026

Work in /home/pvl/spaces/abird/src/zerogravity and continue only the non-Fabric ZeroGravity execution workstream.

Read, completely and in order:

1. AGENTS.md
2. .agents/plans/README.md
3. .agents/plans/TODO.md
4. .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-28.md
5. .agents/docs/handoff-2026-10-26/USER_DECISIONS.md
6. .agents/docs/handoff-2026-10-26/DESIGN_AND_DECISIONS.md
7. .agents/docs/handoff-2026-10-26/EVIDENCE_AND_CONTINUATION.md
8. .agents/docs/handoff-2026-10-26/OPEN_WORK.md
9. .agents/docs/handoff-2026-10-26/SOURCE_CATALOG.md
10. .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md
11. .agents/plans/model-execution/TODO.md
12. .agents/plans/numerics-and-context/TODO.md
13. .agents/plans/cpu-gpu-optimization/TODO.md
14. .agents/plans/serving-and-speculation/TODO.md
15. .agents/plans/memory-and-streaming/TODO.md
16. .agents/plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md

Before changing anything run:

    export CARGO_BUILD_JOBS=8
    export RAYON_NUM_THREADS=8
    export PYTHONDONTWRITEBYTECODE=1
    git status --short --branch
    git log --show-signature --oneline -30
    git worktree list --porcelain
    nvidia-smi

Current truth:

- work only on main unless a genuinely new isolated scope requires otherwise;
- only the main worktree should be registered;
- ordinary source-derived Qwen3.5 .8B/2B CUDA is integrated but qualification-only;
- one ForwardPlan, one ResourceArbiter, one queue/flight/completion/quarantine path and the existing StateRoot authority must remain the only execution/publication authority;
- source-derived execution must never be selected by prompt IDs/lengths, call/node ordinals, packet contents, capture tables or recipe/catalogue hashes;
- the source-independent hot path has already been physically qualified on the RTX 3060: source-exact RoPE, all RMS variants, SiLU/sigmoid/multiply, KV append, .8B embedding, attention wrapper, canonical GDN state transport and generic GDN initial/continued prefill + cached decode;
- two real defects were already found and fixed: source-inexact legacy RoPE and cached GDN Gated grid [4,1,1] writing only 512/2048 outputs; cached Gated is now [16,1,1];
- exact admission currently fits .8B through 32K and 2B through 16K; 2B 32K must refuse before allocation;
- do not redo primitive work unless a new failing qualification proves it necessary.

Immediate next goal:

1. Revalidate main and exact artifact availability.
2. If the exact approved .8B HF source tree, pinned native bundle payloads and pinned cuBLAS/cuBLASLt DSOs are available, run:

       scripts/qualify-qwen35-ordinary-cached47-50.sh --preflight-only

3. If preflight passes, run the same script without a mode. It must execute historical reference capture first and ordinary comparison in a fresh second process.
4. Require exact cached 47→48→49→50 parity for all 248,320 logits, selected token/max bits and all 42 canonical state domains.
5. If phase 1 succeeded but phase 2 must be retried, use:

       scripts/qualify-qwen35-ordinary-cached47-50.sh --compare-existing

6. After exact parity, run after-launch cancellation/fault, retirement and real-model shutdown qualification.
7. Then run .8B physical context correctness 512/1K/2K/4K/8K/16K/32K; admit 2B only where the exact two-root admission fits.
8. Run the final assembled repository gate.
9. Only then run the one serial matched vLLM campaign in the benchmark plan.

Current artifact blockers recorded at handoff:

- approved Qwen/Qwen3.5-0.8B revision 2fc06364715b967f1860aea9cf38778875588b17;
- weight shard model.safetensors-00001-of-00001.safetensors, 1,746,942,600 bytes, SHA256 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696;
- pinned libcublas.so.12: 105,140,976 bytes, SHA256 d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048;
- pinned libcublasLt.so.12: 749,205,904 bytes, SHA256 96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541;
- the connector-visible target/models/qwen35-0.8b tree and current Nix cuBLAS build are not substitutes.

If those exact held artifacts are still unavailable, report the exact missing identity and stop at that boundary. Do not download, repin or substitute them. Do not skip ahead to context timing or vLLM benchmarks.

Preserve:

- root README bytes;
- unrelated untracked Fabric vNext files/plans;
- tools/cuda-gdn-recipes;
- all historical failure records;
- independent StateRoots;
- exact source/provider identities;
- signed scoped commits.

Do not run Fabric vNext, push, tag, publish, change device permissions/recovery, or launch held AMD hardware. Use at least 8 Cargo jobs by default. Keep physical GPU work serial. Independently inspect any delegated work before integrating it. Report substantial verified checkpoints with current goal, what is complete, blockers and the exact next step.
