# ZeroGravity — next agent starting prompt (UX-9C)

You are taking over ZeroGravity. Continue in:

/home/pvl/spaces/abird/src/zerogravity

### 1. Acquire complete project context

Read in order:
1. AGENTS.md
2. .agents/plans/README.md
3. .agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md
4. .agents/plans/FULL_IMPLEMENTATION_PLAN.md
5. .agents/docs/CURRENT_HANDOFF.md
6. .agents/docs/CURRENT_BASELINE.md
7. .agents/plans/TODO.md
8. .agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md
9. Relevant CLI/device-routing, acquisition, serving, public SDK and platform specifications/TODOs linked by the handoff.

Read §§0–13 first. Appendix D contains every exact Phase 2 landing
narrative, including UX-7 implementation and later acceptance. Appendix E
and the 25 Nov Phase 1 exit pack preserve the original design, full
foundation implementation, prior chronology, physical/source evidence,
reusable primitives and rejected experiments. Appendix B is a dated
literal open-item snapshot; live owning TODOs take precedence. Do not
reconstruct architecture or repeat rejected experiments without materially
new evidence.

### 2. Revalidate actual repository and execution authority

Latest accepted implementation: 4f23983a5bb33a3ab2e9ef124342a0cc33df2088 (UX-9B).
Accepted Phase 1 exit: d5030ad4ce476b913a46056d492992ccf1b8edf7.
Preceding comprehensive Phase 1 takeover: 450cea5530dad567ff7706c30e1a1872ff01bda7.

The new handoff is a documentation-only descendant; determine its SHA
from Git. Revalidate actual HEAD, staged/working tree, tracking ref,
worktrees, jobs and shared physical resource owners. Live source,
executed tests and owning plans take precedence if advanced.

Use dotlink---l5 as the sole source-authoring connector. Main is the only
persistent branch and checkout. X2 and A1 are execution/testing only;
synchronize exact L5 commits by verified incremental Git bundles when
needed. Preserve unrelated agents' work/jobs/resources. No destructive
Git, privilege/recovery changes, GPU resets, unapproved downloads, or
push without explicit user approval.

### 3. Current implementation position

Minimum Phase 1 shared foundations accepted within bounded qualified
scope. UX-1–7, UX-8A–D, UX-9A and UX-9B accepted. Latest full isolated
workspace gate 2,489 passed, zero failed, 109 explicit skips; release
Web/TUI, installer and scripts pass. Actual cached Qwen 0.8B GGUF
browser/TUI/HTTP lifecycle passed with original 8 GiB host envelope.
No new physical GPU, public-network, numerical or performance grant.

One Runtime/ControlPlane, one process-local Selection, one shared Actions
slot, original model/registration/OutputStream/cache/download/listener
owners and one typed ConfigDocument. Resident native and Web actions
share the worker. Browser control auth is independent of inference auth.
Exact wire IDs are canonical strings. Native cache/download/device and
settings operations still require their respective ordered Web slices.

### 4. Next coherent unit — UX-9C: Web Models / Downloads / Devices

Read canonical UX §46A.9, §46A.13–29 and the UX-8C/9A/9B narratives.
Inspect control_host.rs, control_host/{actions,work}.rs,
tui/{library,work,view}.rs, tui.rs, web/{actions,http,wire}.rs, web.rs,
resident.rs, pull_native.rs and pull.rs, original SDK control/cache.rs,
control/downloads.rs, device policy and artifact/cache/acquisition owners.
Inspect SolidJS web/src/{main.jsx,resident.jsx,api.js,control-state.js,
work-state.js,styles.css} and actual tests.

Implement bounded native-equivalent Web model/catalog/partial, download
and device actions and observations through the original owners:
- Promote the relevant native request variants and cold observations into
  the existing host-neutral seam; no duplicate cache/registry, terminal
  dependency in Web-only builds, or new action queue.
- Exact digest inspect/verify/load/run/load-for-Serve, non-replacing alias,
  dry-run/confirmed reclaim, original referenced/busy/durable/incomplete
  outcomes. No arbitrary cache root/path, raw diagnostics or provider
  credentials. Catalog/partial metadata is not execution/resume authority.
- Use original pinned explicit HF/GitHub files/quant or approved Qwen
  recipe, consent, cache quota/admission/expected digest and publication.
  Offline before transport; no automatic download, quota increase,
  tag/quant guess, opaque checkpoint resume or automatic load.
- Display original download phase/status, actual bytes/total/maximum/rate/
  checkpoint/resumed and terminal durability. Pause/resume/cancel targets
  exact operation ID even while pull owns the shared slot, not a second
  action queue.
- Show observed physical device identity, memory/backend/provenance/
  exclusion, exact source/device policy check or explicit load through
  original resident planner. No inferred support, implicit fallback or
  persisted preference on visual selection.
- Preserve bounded deny-unknown-fields JSON, independent loopback bearer
  and same-origin auth, exact action/registration/generation/operation
  IDs, confirmation/stale refusal, native-private result handling and
  original event invalidations. No arbitrary filesystem/argv/credential
  or unrestricted execution grant. Client disconnect is not retirement.
- Reuse actual cached lifecycle, controlled transfer, Web HTTP/Firefox and
  native PTY fixtures. Cover real cache lifecycle, offline/feature/auth/
  stale refusal, original pause/range resume/cancel, shared owner
  observations and joined retirement. Meaningful focused checks then the
  complete mandatory isolated gate. No remote transfer or physical GPU
  requalification unless actually needed and authorized.

Do not start UX-9D/E, UX-10, public serving expansion or backend
performance prematurely.

### 5. Ordered plan and engineering principles

1. Shared foundations — bounded exit accepted.
2. CLI, native TUI, embedded SolidJS Web, compatibility and public serving.
3. Match or beat optimized vLLM on ROCm.
4. Match or beat optimized vLLM on CUDA.
5. Native Vulkan feature/correctness/performance parity.
6. Optional concrete MIGraphX, TensorRT, IREE and ONNX Runtime.

After UX-9C: UX-9D Profiles/Config/Logs, UX-9E simultaneous client
acceptance, UX-10 doctor/pipeline/compatibility, then public serving and
normal benchmark acceptance. One SDK/runtime/catalog/ConfigDocument/
actions/snapshots/events. Separate inference/control auth. Original
HF/Safetensors and GGUF primary; no mandatory ONNX conversion, second
canonical IR or external graph SDK for native execution.

Elegant, simple, clear and intuitive Rust. Favor zero-cost abstractions
and clean direct hot paths when practical; targeted optimization of
measured bottlenecks when complexity otherwise grows. Reuse mature math.
Preserve correct reusable model-neutral primitives even if a production
selector loses. Remove losing consumer/probe wiring, not useful
implementations. Preserve exact source/numerical/layout/state/resource/
ownership/completion/cancellation.

No overengineered granularity, evidence pinning or redundant qualification.
Reuse accepted evidence and actual changed-consumer checks, then the
assembled full private-network gate. Use locked/offline approved inputs,
existing scripts/check.sh, and coordinate build/memory/device owners.
Keep the original 8 GiB real cached serving admission bound.

### 6. Cadence and closeout

One substantial coherent validated signed checkpoint per user turn,
approximately 30 minutes. Recover interrupted work from actual source
and job results. Update owning TODOs, CURRENT_HANDOFF and CURRENT_BASELINE
as appropriate. Report completed work, exact commits, accepted scope,
limitations, failed/interrupted checks and concrete next unit. Preserve
unrelated work and resources. Do not push.

Revalidate current state and implement UX-9C.
