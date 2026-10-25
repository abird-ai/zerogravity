# B4/C16 execution continuation — 2026-10-25

Source parent: `9e67b74000bc085a844d8844d954d82f8e196493` (`main`). Audit interval begins at `4bf3673`. Run evidence: `.agents/runs/takeover-independent-validation-20261025/`. This document distinguishes GPU-correctness, model, and performance scopes. Earlier failures remain preserved; no model download, provenance substitution, push, GPU permission change, or x2 capture was attempted.

## Changes implemented

- Corrected B4 grouped KV control ABI. CUDA kernel arguments are row-major `base0,end0,block0,base1,end1,block1,...` (8/4/4 bytes per row), not control-major `base0..3,end0..3,block0..3`. Both `kv_definition` size metadata and `PreparedTypedTransport::prepare` checked handles now use the source ABI. Added a source-only test assertion for exact binding size order, guarding both import (22) and export (30) bindings. No limit was enlarged and no unvalidated pointer accepted.
- Preserved and completed the GPU-fixture correction already active in the continuation worktree: packed native BF16 halfwords, per-row candidate expectations, exact immutable read-only vs writable submission census, launch population, and lifetime/retirement checks. Prior failed attempts show respectively out-of-bounds fixture, incorrect access, wrong submitted writes, and binding extent mismatch.
- Corrected fixture lifetime check: after complete readbacks, drop the two remaining status leases `status_in/status_out` before checking zero resident/event counts. Only after verified zero events/bytes and `memory_cleanup` drop the original `ResidentService` provider envelope, then require actual physical arbiter `reserved=0` and not fenced. Previous check compared final0 while provider owner was still alive. Zero-retention contract is **not relaxed**.

## Actual results

| Scope | Result | Evidence |
| --- | --- | --- |
| CUDA grouped state transport, **real RTX3060**, 18 cases | **PASS** (rc0; one ignored libtest selected; 4.90s). Per-row vs grouped exact, canaries, sticky status, guard, future padding, native event/root retirement | `physical/model-free-grouped-fast-8/{rc,binary.sha256,stdout.log,stderr.log,gpu-before}` |
| CUDA model-free finite301, **real RTX3060** | **PASS** (rc0; 301 finite cases; one ignored libtest selected; 0.96s) | `physical/n1-finite301-fast-1/{rc,binary.sha256,stdout.log,stderr.log,gpu-before}` |
| Fast release profile build for physical diagnostic | **PASS** (rc0) `CARGO_PROFILE_RELEASE_OPT_LEVEL=0 CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256 CARGO_PROFILE_RELEASE_LTO=off`; therefore **not** optimized C16/performance-profile acceptance | `a2-fast-status-retire.{rc,stdout.log,stderr.log}` |
| CUDA GPU lib including new test code | **PASS typecheck** `cargo check --locked --offline -p zerogravity-gpu --no-default-features --features vllm-cuda-qualification --tests` | `b4-test-typecheck.{rc,stdout.log,stderr.log}` |
| HIP/Vulkan/CUDA isolation + G/V regressions | **PASS rc0**, 5 G + 2 V source tests | `b4-feature-boundaries.{rc,stdout.log,stderr.log}` |
| Full CPU workspace gate | **PASS rc0** post-correction | `final-cpu-gate.{rc,stdout.log,stderr.log,started,finished}` |
| Original 4344-record B4 CPU archived consumer preflight | **INCOMPLETE**: hard timeout 75s and 101s; no test assertion recorded | `physical/b4-original-cpu-preflight-fast-{1,2}/` |
| Original optimized A2 release build | **INCOMPLETE**: prior canonical build timed out after 1700s; this continuation used *unoptimized release* for diagnostic tests | Original `resume-all-takeover-b4-gpu-c16-glm-20261025/LOG.md`; local A2 attempts |
| 24 model-dependent B4/N1 scopes, fresh matched .8B and 2B C16 | **NOT RUN / NOT CERTIFIED** | B4 original 26-case gate not yet green |

Physical execution used original `scripts/native-gpu.py` isolation; inherited installed CUDA 12.9 include-view `.agents/runs/p12-full-fusion-root-2026-10-19/cuda-view` and a Nix CUDA 12.9 NVRTC library with RTX3060 driver library. No CUDA SDK/image/model download. RTX3060 returned to idle **41 MiB** after both passing cases.

## Remaining blockers in mandated user order

**3 — B4 26-case model/physical acceptance:** Original pinned Qwen3.5 .8B and 2B model snapshots are under `/home/pvl/.cache/abird/zerogravity/`. The connected `dotlink---l5` sandbox has no read permission there. The original model-backed harnesses require those exact bytes and model/source pins; using a substitute path/model, earlier binaries or stale results would violate the accepted source contract. Additionally the immutable optimized release A2 build and original 4344-record preflight require a genuinely long-lived execution owner, not a ~100s shell-call ceiling. Need legitimate **read-only grants** for the two frozen model snapshot directories (plus any original external-reference baseline if required by selected owner), and a supervised long-running command route retaining original isolated, one-case-at-a-time guarantees. A user-granted connector profile extension or already-authorized matching profile is needed; do **not** disable the sandbox or silently widen permissions.

**4 — Matched C16:** **held** pending all 26 physical cases and exact optimized binary pin. Benchmark runner, source/profile/stock recipes and 0.8B/2B same-attempt envelopes stay as recorded in the canonical B4/C16 handoff. No new ZeroGravity vs vLLM speedup, latency or model-parity claim. No C32/C64.

**2 — Qwen27B x2 reference capture:** strictly *after* 3/4 per current user's order. Existing separate blockers persist: trusted strict-SSH host key not exposed to connector; UD6 unexplained prior exit 137. Never accept a new fingerprint without independent authentication; never infer OOM from SIGKILL. Keep x2 weight/image downloads paused. Prefer exact content-preserving derived-image save/load after authorization + verified host access.

No other source files were changed.
