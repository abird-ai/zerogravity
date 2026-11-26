You are taking over ZeroGravity. Continue in:

`/home/pvl/spaces/abird/src/zerogravity`

### 1. Acquire complete project context

Read in order:

1. `AGENTS.md`
2. `.agents/plans/README.md`
3. `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md` §§0–13 first
4. `.agents/plans/FULL_IMPLEMENTATION_PLAN.md`
5. `.agents/docs/CURRENT_HANDOFF.md`
6. `.agents/docs/CURRENT_BASELINE.md`
7. `.agents/plans/TODO.md` and relevant owning TODOs
8. `.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md` (especially §46A)
9. `.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md` and `.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md`
10. Relevant CLI/device-routing, acquisition, serving, SDK, platform and structure/naming specifications and TODOs.

The comprehensive handoff Appendix D preserves every checkpoint since UX-9B, including original landing narratives, decisions and exact changed paths. Appendix E and the linked earlier complete handoffs preserve the original architecture, every historical foundation/UX implementation step, accepted evidence, rejected experiments, reusable primitives and pending work going back to the October design. Use the detailed sources and previous exact experiment conditions; do not reconstruct from assumptions or repeat rejected trials without materially new evidence. Appendix B is a historical pending-marker snapshot, not a second task queue.

### 2. Repository and execution authority

Implementation checkpoint: `e96cdab98d47c5c84001c1afaaebd8f126b9e0b3` (UX-P-M1).
Accepted UX-1–10 exit: `1b8de1cb87fa4815655cbb6700e6633946f93305`.
Accepted bounded Phase 1 exit: `d5030ad4ce476b913a46056d492992ccf1b8edf7`.
The new comprehensive handoff is a later documentation-only commit; find its exact current SHA with Git. Revalidate actual HEAD, working tree, staging, tracking ref, worktrees, jobs, available resources and original shared physical owners. Live source, executed tests and owning plans take precedence if advanced.

Use **dotlink---l5 as the sole source-authoring connector**. Main is the only persistent branch/worktree. X2/A1 execution/testing only; synchronize exact L5 commits via verified incremental Git bundles and clean fast-forward when necessary. Preserve unrelated work and shared physical owners. No destructive Git, unapproved downloads, privilege changes, GPU resets, unrelated process cleanup or push. Sign commits; do not disable signing.

### 3. Current implementation and product direction

Phase 1 bounded foundations and UX-1–10 are accepted. UX-P1/P2/P3A/B/P4A1 (including toolbar-free correction)/P4A2, full native Cargo CLI defaults/error diagnostics, and UX-P-M1 live metrics are implemented and focused-qualified. The interrupted M1 turn had actually committed; do not redo it. Latest complete full isolated gate was UX-10: 2,505 Rust test executions, zero failures, 109 explicit skips, 318 suites, release Web/TUI/installer/scripts. This is not final acceptance of later changes.

Latest M1 approved cached Qwen 0.8B GGUF CPU/browser/native/HTTP lifecycle passed (55.61s, five filtered), shared registration counters/reset, exact 24 CPU layers, original 8 GiB admission, exact output and stale/shared settings retained. Engine 109/0/0, combined CPU CLI 64/0/3, Web 4/0, focused strict lint and full default cargo check. GPU layer placement compile-only. No performance parity claim.

Preserve one Runtime, ControlPlane, shared Actions slot, process-local Selection, typed ConfigDocument, original source/cache/transfer/registration/scheduler/listener owners, physical ResourceArbiter and sole StateRoot publication. Browser control auth is separate from inference API auth.

User-directed UX: no permanent action toolbar; one rail, inline live local search (ordinary printable keys filter), exact selected-object ellipsis/right-click/context menu, separate session menu, concise normal summaries and details on demand, smart IEC sizes, Dark/Light/Auto, complete native mouse/keyboard and resizable independent panes/editors. Running is the quiet live metrics dashboard. Preserve exact registration TPS/TTFT semantics and actual layer maps, independent response viewport, no fictitious telemetry or hidden extra ownership.

Default Cargo CLI is production all (all current model features, CPU/CUDA/ROCm/Vulkan, acquisition, serve, TUI/Web); no diagnostic/experimental features. Preserve explicit minimal/portable and embedding builds. Full default check passed, linked default cargo run timed out in final compilation; verify actual launch. Full-GPU Clippy debt remains. Portable Nix/release explicitly CPU; do not imply all-backend distribution without verifying platform intent. Error chain retains typed resource/I/O causes, accurate InvalidInput/Unsupported vs Malformed. HTTP/SSE safe public message/stable status+code, optional authenticated loopback debug details, safe structured operational logs regardless of HTTP setting. No pre-auth leakage or browser-safe toggle of trusted diagnostic exposure.

### 4. Next coherent unit — UX-P4B required assembled acceptance

Read comprehensive handoff §11, especially §10.2 known gate/default/build caveats. Reconcile actual full production link/launch, explicit feature-gate coverage (old cache-only gate commands inherited changed defaults), existing GPU lint debt and real changed-consumer regressions. Do not reduce admission, remove valid dormant primitives to silence lint, weaken tests or bypass private-network checks. Qualify the new exact listener error-event assertion and actual cached browser/native/HTTP/shared transfer/config behavior under unchanged owner policies and 8 GiB bound. Real GPU layer reporting requires actual already-qualified adapter observation before claiming it; unsupported remains explicit.

Optional UX-P-M2 is preserved in LIVE_MODEL_PERFORMANCE.md. If justified by actual existing boundaries or explicitly requested, implement bounded exact response/request identity, phase timings and deeper actual layer/device/memory inspection. No fake offload/split, no estimated layer time, no default profiling sync/overhead, no second runtime/request registry. Do not make optional profiler architecture a blocker to P4B. The current ordered plan/queue requires P4B next; no public serving before UX-P acceptance.

Use the existing isolated harness. On outer shared-network connector:
~~~sh
export CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
bwrap --unshare-net --bind / / --dev-bind /dev /dev -- \
  python3 -B scripts/native-gpu.py --require-private-network -- bash scripts/check.sh
~~~
Reduce build jobs for observed resource pressure. Use locked/offline approved inputs, real cached owner fixture serially with explicit CPU/model/text/TUI/Web/serve/acquisition features and --no-default-features, original admission; existing private loopback setup. Full assembled gate last. Reuse applicable evidence; no premature phase exit or claim from compilation alone.

### 5. Remaining delivery order

1. UX-P4B comprehensive acceptance (optional capability-bound M2 as appropriate).
2. Public OpenAI-compatible serving and normal CLI/SDK/server benchmark acceptance (owning dormant serving specification).
3. Match or beat optimized matched vLLM on ROCm first.
4. Match or beat optimized matched vLLM on CUDA.
5. Native Vulkan parity.
6. Concrete optional MIGraphX, TensorRT, IREE and ONNX Runtime integrations.

Original Hugging Face/Safetensors and GGUF remain primary source paths. Native execution must not require ONNX conversion or external graph compiler SDKs. Do not reopen bounded Phase 1 or start premature backend performance tuning. Preserve historical rejected approach/held device/model/quality conditions; restore approved comparator only with actual authorization.

### 6. Engineering principles and cadence

Elegant, simple, clear, intuitive. Rust library/execution fabric: zero-cost abstractions/direct hot paths and high performance when clean; simple maintainable design first where complexity is not justified, optimize measured hot areas. Reuse mature optimized mathematics and attack large bottlenecks first. Preserve useful correct model-neutral primitives even when an individual consumer loses; remove losing selector and temporary probes, not reusable implementations. Preserve exact source/numerical/layout/state/resource/ownership/completion/cancellation. No overengineered test/evidence/receipt/pinning administration. Reuse still-applicable evidence and meaningful focused/actual changed-consumer checks; assembled mandatory-private-network gate last.

Aim for one substantial coherent, validated, signed checkpoint per turn within approximately 30 minutes. Update owning TODOs/current handoff/baseline as appropriate. Recover interrupted work from actual source and results, not assumptions. Report completed scope, exact commits, tests, limitations, blockers, next unit and clean worktree, then pause. Do not push unless explicitly instructed.

**Revalidate current state and continue the next required coherent acceptance unit, UX-P4B.**
