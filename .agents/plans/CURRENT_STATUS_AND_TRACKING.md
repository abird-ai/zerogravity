# Current status and tracking

This file is the stable status target used by the repository README. Cross-window context starts at [CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).

## Authority

Current source and tests are the implementation truth. The active work queue is
[TODO.md](TODO.md), with plan conventions in [README.md](README.md). Git history
retains superseded plans, handoffs, checkouts and run evidence; they are not
live authority.

## Owned CUDA checkpoint — `2983bb7`

- Owner authorized continuation and completion. Signed owned CUDA execution
  checkpoint `2983bb7` is on the sole main worktree; nothing pushed.
- Owned driver-only GEMM replaces ordinary opaque BLAS under the original
  queue/compiler/arbiter. Pinned reference/source/limits stay unchanged.
  180 synthetic and eight authenticated projection cases pass exactly, with
  intact guards and retired leases/events.
- Corrected rotary admission (+280 bytes), GDN partition binding, FP32 residual
  normalization, fused Q/K reduction and CUDA rotary FMA rounding. CPU contracts
  remain unchanged. 211 GPU host tests and14-position rotary comparison pass;
  strict CUDA GPU/facade all-target Clippy and formatting checks pass.
- Fresh ordinary `.8B` cached47→48→49→50 now passes exactly: all248,320 logits,
  all42 state domains and natural feedback at every step, against the unchanged
  independently authenticated reference. Real-model normal shutdown also passes.
- Allocation-free tile-64 CUDA attention replaces the CPU-profile lowering;
  independent prefill/cached attention operands match exactly. Nine analytic
  causal/multitile cases through a32K active prefix and three nonfinite-refusal
  cases pass with intact guards and retirement. Admission totals are unchanged.
- 16 independent staged CUDA RMS cases pass. Real post-launch cancel/abandon
  preserves original quarantine/prior state. Normal lifecycle passes all13
  admitted capacities at48 active tokens;2B32K refuses before physical creation.
  Full active-context/2B numerical coverage remains open; production is disabled.
  Final source gate passes2,129/0failed/83ignored; CUDA/native host675/0failed,
  plus strict CUDA/HIP/Vulkan feature checks. Earlier failed scopes remain failed.
- Remaining input edge: exact2B reference image and trusted active-context
  comparison inputs; no model/SDK download, substitution or reference relaxation.
- ROCm on `pvl-x2` is unqualified. Read-only SSH confirms AMD PCI1586, cached
  ROCm/Rust/Python toolchains and prior qualification inputs; no current checkout.
  No physical AMD run or permission change. No benchmark detour; see
  [numerics](numerics-and-context/TODO.md).

## Historical pre-owned-execution baseline — 29 Oct 2026

- Historical reviewed implementation/source baseline: `ae8cd89`; the 29-Oct continuation adds reference-fixture/diagnostic repairs, not ordinary execution support.
- Ordinary Qwen3.5 `.8B`/`2B` CUDA is integrated under the existing resident owner but remains qualification-only; production routing still refuses before source/device/native load.
- Approved `.8B` source revision `2fc06364715b967f1860aea9cf38778875588b17` is visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`. The 1,746,942,600-byte safetensor hashes to `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- The `.8B` N1 native/reference bundle is recovered offline under XDG state: retained canonical native images plus the two exact SM80 FlashAttention modules extracted from the retained pinned reference image, with the original cuBLAS providers. All 28 resources match current Rust identities; all 11 approved model members verify. Maintained bundle validation and isolated wrapper structural preflight pass. No downloads, repinning or deleted run-tree restoration.
- Fresh phase-1 cached47→50 capture passes after selecting the existing 8 GiB reference-fixture host profile; all 1,008 source windows retire. Ordinary host and all device limits remain unchanged. The original 4 GiB refusal stays failed.
- At this historical checkpoint, opaque cuBLAS initialization failed before math: status1, refused private export UUID `6bd5fb6c5bf4e74a8987d93912fd9df9`, zero allocation attempts/charges; legacy graph lookup also refused. That failure remains failed. `2983bb7` replaces ordinary execution's dependency, not the refusal or reference.
- That checkpoint's source-free lifecycle/admission/serving checks did not qualify real inference; current physical scopes are listed above.
- Fresh review fixed environment-sensitive PID1 serving shutdown control and added the maintained Qwen27 layer9 source-contract suite to `scripts/check.sh`.

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
