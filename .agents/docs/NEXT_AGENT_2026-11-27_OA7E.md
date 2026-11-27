You are taking over ZeroGravity. Continue implementation in:

/home/pvl/spaces/abird/src/zerogravity

### 1. Acquire complete project context

Read in this order:

1. AGENTS.md
2. .agents/plans/README.md
3. .agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-27_OA7D.md — §§0–13 first, then Appendix F (OA-7E recovery).
4. .agents/plans/FULL_IMPLEMENTATION_PLAN.md
5. .agents/docs/CURRENT_HANDOFF.md and .agents/docs/CURRENT_BASELINE.md
6. .agents/plans/TODO.md
7. .agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md — especially §§6–10, 17–25, 30–37 and OA-6/OA-7.
8. .agents/plans/serving-and-speculation/TODO.md
9. .agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md; .agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md; relevant state, physical resource, scheduler, model/source/numerical and multimodal contracts.
10. Read the relevant prior comprehensive pack and owning plans before revisiting any rejected approach or frozen evidence.

The new pack preserves every post-OA-4B landing (11 signed implementation checkpoints), original narratives and exact engineering-document changes in Appendix D. Appendix E preserves the previous complete main body and links every older comprehensive pack through the original October design. Appendix A indexes the engineering corpus; B snapshots unresolved markers; C is the full reachable Git ledger; F preserves interrupted OA-7E investigation and a time-stamped diagnostic WIP snapshot. Earlier next-agent prompts are historical, not instructions. Live source, actual executed evidence, AGENTS.md and owner TODOs remain authoritative.

### 2. Revalidate actual repository and execution authority

Last independently confirmed accepted implementation:
05c54e5d73f193543c725dfba64d1a039f2075e6 — OA-7D.
OA-7C: 83c79f1eda16b28401c0bf6728637bdfd2a625ab.
OA-7B: 642b72f38db8dcb77a758d10526ba9aa5a3a4fa8.
OA-7A: cec65c4f5cda4aaa807c21e8c69ab3c9b8491914.
OA-6C: fa269df16d24cfaab9ff2c4f4863f4160f10d87f.
OA-5: 70e9809c209794b4b98689dbf6f78e57dc6555f0.
Accepted UX-1–10: 1b8de1cb87fa4815655cbb6700e6633946f93305.
Accepted bounded Phase 1 exit: d5030ad4ce476b913a46056d492992ccf1b8edf7.

Do NOT assume a clean checkout or that HEAD is still 05c54e5d. During handoff, OA-7E changes appeared progressively in the reference script, native model, engine, SDK, serving and test files; the worktree may have an active writer or a durable test job. The independent reference succeeded and a development compile passed; check the live status of any subsequent tests/commits. Coordinate/establish the owner and review staged, unstaged and untracked changes before touching source. Do not replay the captured patch on top of existing or newer work. If OA-7E has since landed, verify it and continue the next open coherent unit instead.

Use **dotlink---l5 as the sole source-authoring connector**. L5 main is the only persistent checkout/branch. Revalidate HEAD, local tracking ref (not remote live state), worktrees, durable jobs, resource owners, available RAM and disk. X2/A1 execution/testing only; sync exact authorized L5 commits via verified incremental Git bundles when needed. Preserve all unrelated and in-flight work. No destructive Git, unapproved model/SDK downloads, privilege changes, GPU resets, unrelated cache/process cleanup or push. Sign commits; never disable signing. Root README is user-authored and stays unchanged.

### 3. Accepted scope and important open boundaries

Phase 1 bounded foundations; UX-1–10 and UX-P4B, live model metrics and model-local backend/device selector; OA-1 through OA-5 supported text contracts are accepted. OA-6A/B/C embedding implementation, shared runtime and wire/SDK fixture are qualified, NOT trained MiniLM parity or built-in public eligibility. MiniLM original Safetensors (~90.9 MB plus config/tokenizer/module closure) acquisition remains subject to explicit authorization. Do not interpret bare "next" as download permission.

OA-7A/B real cached Gemma4 E2B F32 audio encoder, one shared verified-window source, audio-conditioned prefill/decode qualified. Correct PLE semantics: PAD token lookup, learned PLE projection from merged audio embedding. OA-7C native transcription through shared Runtime and OA-7D strict authenticated multipart PCM16 mono16k WAV JSON/text endpoint accepted. Actual installed OpenAI SDK 2.33.0 on synthesized "Hello world": input 41, output 3 including stop, total 44; 264.264s CPU reference, 292.73s real lifecycle test. Full isolated gate 2,595/0/112 across 324 suites (904s), strict lint and release/installer checks passed. Explicit test-only public adapter, built-in Gemma candidate remains native-only. No general natural-speech, throughput, GPU audio, speech synthesis, translation or streaming acceptance follows.

### 4. Next coherent implementation unit — recover/finish OA-7E

Read Appendix F and live diff/source. A local synthesized Spanish speech reference already ran on original cached Gemma4/PyTorch/Transformers: 9,558 samples, 15 audio rows, reference output "Hola mundo\nEnglish: Hello world", original stop token. The WIP adds distinct speech task and native/public translation eligibility, source model prompt and target-line extraction, original shared receipt/scheduler and multipart /v1/audio/translations. Inspect actual implementation and results; do not assume acceptance from compilation or the reference. Retain distinct task/registration/limit identity, real output token accounting (including generated source transcript and stop), cancellation/length/failure semantics, no unlabeled/source-only fallback masquerading as English. Verify exact reference prompt and actual native English result, explicit public test adapter and official SDK, malformed/unsupported controls, atomic complete outputs, disconnect/cancellation, shared physical admission and joined retirement. Reuse OA-7D unchanged evidence; regression test affected transcription and mixed-task consumers. Run the mandatory private-network assembled gate on the coherent final batch. Keep natural speech/general quality and production eligibility gated.

If source/SDK acquisition or natural-speech quality remains blocked, document it and advance independent ready work, never simulate a model capability. TTS requires real learned synthesis source; OA-8 real generation/edit/variation (not just a vision encoder); OA-9 bounded Realtime over actual common capabilities; OA-10 endpoint/SDK/lifecycle/help closeout.

### 5. Remaining implementation order

Finish remaining applicable public-serving capabilities (including OA-6 trained acceptance when authorized), ordinary benchmark/serving acceptance; then match or beat optimized matched vLLM on ROCm, CUDA, native Vulkan; only then concrete optional MIGraphX/TensorRT/IREE/ONNX Runtime providers. Original Hugging Face/Safetensors and GGUF primary; no mandatory ONNX conversion or external graph compiler in native execution. Respect current holds (including Renoir/gfx90c) and earlier rejected selectors; no performance claim from compilation, synthetic fixtures or reference CPU times.

### 6. Engineering contract and cadence

Preserve one Runtime, ControlPlane, shared Actions slot, process-local Selection, typed ConfigDocument, source/cache/transfer owners, semantic admission, registry, scheduler, workers, listener, original request receipt, StateRoot/SessionAuthority and physical ResourceArbiter. Protocol resources must not retain model sessions or own another scheduler. Model semantics in model crates, model-neutral operators shared, physical backends own their resources. Elegant, simple, clear, intuitive Rust; favor zero-cost abstraction/direct hot paths, target measured major bottlenecks, reuse mature math. Preserve correct reusable primitives even when individual production selectors lose.

Avoid overengineered test/evidence/pinning; meaningful focused checks, changed real consumers and assembled final gate, reuse accepted evidence where unchanged. Locked/offline builds; mandatory private isolation via existing scripts/native-gpu.py --require-private-network and explicit bwrap --unshare-net when needed; preserve original loopback setup, limits and admission. Account for observed free disk/memory and GPU owners. Do not purge unrelated caches. Use subagents for independent work where available with explicit boundaries.

Aim for one substantial, coherent validated signed implementation checkpoint per user turn within ~30 minutes. At checkpoint update owning TODOs, current handoff and baseline; report actual completed work, exact commits, accepted scope, blockers, limitations and next unit; pause. No push without explicit instruction.
