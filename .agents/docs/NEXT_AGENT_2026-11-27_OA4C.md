You are taking over ZeroGravity. Continue in:

`/home/pvl/spaces/abird/src/zerogravity`

### 1. Acquire complete project context

Read these in order:

1. `AGENTS.md`
2. `.agents/plans/README.md`
3. `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-27_OA4B.md` — §§0–13 first
4. `.agents/plans/FULL_IMPLEMENTATION_PLAN.md`
5. `.agents/docs/CURRENT_HANDOFF.md`
6. `.agents/docs/CURRENT_BASELINE.md`
7. `.agents/plans/TODO.md`
8. `.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md` — especially §§6–10, 17–25, 30–37 and OA-4
9. `.agents/plans/serving-and-speculation/TODO.md`
10. `.agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md`, the current router/state/resource/SDK contracts and `.agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md`.
11. Relevant owning CLI/UX, acquisition, platform, model execution and numerical specifications as the actual change requires.

The comprehensive handoff's Appendix D preserves every post-M1 landing with exact Git references, original narratives and plan changes. Appendix E retains the previous complete main body and links the full historical chain through the initial October design. Appendix A indexes the original engineering corpus; Appendix B preserves unresolved checklist markers and Appendix C the complete reachable Git ledger. Read the historical evidence before revisiting rejected designs or experiments. Earlier continuation prompts are historical. Live source, tests and owning plans remain authoritative.

### 2. Revalidate repository and execution authority

Latest accepted implementation: `89822f4038b2048affd8b92dac6c7b2bcbeb219e` — OA-4B.

Accepted OA-4A: `0898734b`; OA-3: `cea0f18c`; OA-2: `a76ddbfe`; OA-1: `3f8329f2`.
Accepted UX-P4B gate: `dc9b1956`, implementation `ad7a4178`; model-local execution selector: `240bd05e`.
Accepted UX-1–10: `1b8de1cb87fa4815655cbb6700e6633946f93305`.
Accepted bounded Phase 1 exit: `d5030ad4ce476b913a46056d492992ccf1b8edf7`.
The latest comprehensive handoff is a documentation-only commit above OA-4B; inspect actual HEAD rather than assuming it is the implementation SHA.

Use **dotlink---l5 as the sole source-authoring connector**. Main is the only persistent branch/checkout. Revalidate HEAD, staged/unstaged/untracked changes, local tracking ref, worktrees, active durable jobs, physical GPU/host owners and free disk/memory. X2/A1 are execution/testing hosts only; sync exact authorized L5 commits with verified incremental Git bundles and clean fast-forward when necessary. No destructive Git, unapproved download, privilege change, GPU reset or unrelated process cleanup. Sign commits; do not push unless separately instructed. Root README is user-authored; leave it unchanged.

### 3. Current accepted position

Phase 1 bounded common foundations, UX-1–10, UX-P4B (including toolbar-free design, live metrics and exact reported layer spans) and the model-local backend/device chooser are accepted. OA-1 async transport, OA-2 semantic text input and model-owned templates, OA-3 stateless Models/Responses, OA-4A stored foreground resources and OA-4B joined background polling/cancellation are accepted.

OA-4B already committed and passed: 172 focused engine/serving test executions, strict lint, actual cached Qwen3.5 official OpenAI SDK 2.33.0 (including an actual cancelled background response), real CLI/TUI/Firefox lifecycle and Rust SDK, plus the full isolated gate 2,562 passed / 0 failed / 109 skipped across 318 Rust suites. Do not redo the interrupted turn.

Keep one Runtime, ControlPlane, shared Actions slot, original Selection/ConfigDocument, semantic admission, model registry, scheduler, worker, StateRoot, ResourceArbiter, listener, source/cache/transfer owners and original request receipt. Store only protocol records, not model sessions. Current listener trust scope is not a multi-tenant security claim. Foreground store=false retains nothing; background store=false retains temporary polling records only (10 minutes, no continuation). Stored terminal records 30 minutes. Shared quota 64 record slots, 32 MiB conservative backing; pending/residual readers count; background JoinSet also hard-bounded 64 handles. Logical cancellation is not physical completion.

### 4. Next coherent unit — OA-4 conversation resources

Continue the remaining OA-4 work. Start with conversations/items CRUD and atomic append of context on successful generation using the same protocol store/admission authority. Inspect actual accepted `crates/zerogravity-serving/src/http/responses/{store,input,background}.rs`, `responses.rs`, `transport.rs`, engine tokenizer, request and owner contracts. Recheck the current API reference for supported fields, resource shapes and lifecycle. Do not reconstruct the architecture from assumptions.

Qualify exact authorization and identity, bounded pagination, mutation/expiry/deletion and retained-reader lifetimes, conflict/cancellation/failure semantics and original semantic text replay. Avoid partial append, stale retargeting, lost concurrent updates or fake unsupported items. Preserve previous_response_id behavior and replacement of request-local instructions. Keep response and conversation accounting under the same quota/physical ledger, no independent authority or model-state retention. Support only real typed text capabilities. Make the largest coherent tested landing rather than overengineering tiny evidence stages.

Then complete exact input-token counting via the selected original charged tokenizer without a physical generation session, and bounded background streaming/event replay/resumption with the same joined job/receipt. Keep unsupported routes/options explicit until qualified. Do not claim all OA-4 accepted until its remaining contracts are implemented and validated. OA-5 compatibility close and later endpoint families remain ordered; enable new modalities only through real model capabilities.

### 5. Roadmap and engineering contract

After public serving and ordinary CLI/SDK/server benchmark acceptance, match or exceed optimized matched vLLM on ROCm first, CUDA next, native Vulkan third. Concrete optional MIGraphX/TensorRT/IREE/ONNX Runtime later. HF original Safetensors and GGUF primary, no mandatory ONNX conversion. Preserve prior numerical/source/physical/quality and state/completion semantics. No new hardware/performance claims from compile or CPU UI tests.

Keep Rust elegant, simple, clear and intuitive; zero-cost abstractions where practical; use mature optimized mathematics and focus on real bottlenecks. Preserve correct model-neutral primitives when a selector loses. Batch meaningful tests, actual changed-consumer checks and the assembled mandatory private-network workspace gate; reuse applicable accepted evidence. No overengineered pinning/fixtures/receipts. Run locked/offline; use private loopback via the existing `scripts/native-gpu.py --require-private-network` harness and explicit Bubblewrap network namespace on the L5 connector. See handoff §10 for exact commands. Keep build concurrency within available memory and disk (4 jobs per lane was used for OA-4B); do not purge unrelated caches. No new downloads unless authorized.

Aim for one substantial coherent validated signed checkpoint per user turn within approximately 30 minutes. Recover interruptions from actual source and results. Update owning TODOs, current handoff and baseline; report accepted scope, exact commit, blockers and next unit; pause. Do not push unless explicitly requested.

**Revalidate the current state and continue implementation.**
