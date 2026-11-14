# Plan queue

Current source, tests, and this queue are authoritative; Git retains retired history.
Cross-window entrypoint: [CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).

Latest implementation checkpoint is signed `f1ce532`
(`feat(cuda): bind graph replay to resident flight`). A captured graph now owns
the exact `(ResidentLease, DeviceAccess)` set for its internal storage, and replay
is permitted only when the current armed `ResidentSubmission` retains those exact
allocations with compatible access authority. This closes the same-card
storage/event-authority seam without enabling graph serving. The current
conservative policy charge is15,356,224 bytes; the original0.8B B2 physical graph
qualification passes with exact fullV/all42 state, six graph enqueues, final
reserved0 and41MiB→41MiB. The prior single transient post-capture mismatch was under
the older15,339,840-byte policy and remains an explicit watch item. Production
serving still selects Native. Earlier graph checkpoints through docs `b12e8f5`
are pushed; `origin/main` is `b12e8f5` at this refresh. Revalidate live l5 `HEAD`/ahead
count because this plan refresh may add a docs-only commit. X2 is
the approved AMD large-VRAM execution/test host. Tiny-HIP runtime health is green;
exact tokenizer, isolated source binding/plan, and the full canonical host runtime
test now pass; the latter completed in75.20s once the external execution wrapper
kept emitting progress, proving the prior silent host termination was not a Qwen
source bug. `b0cf0b2` remains the shared duplicate-hash fix. A fresh explicit75GiB
public-HTTP liveness probe reached ~31.376GB total device usage but its isolated
process then disappeared before any HTTP status. Host-global KFD sysfs exposes
owner `2406838`:31,098,232,832 bytes VRAM, six queues, `pasid=0`, zero
faults/occupancy. X2 also runs Ollama/model workloads outside the sandbox, so that
owner is **shared-host baseline unless positively attributed**, not evidence of a
stranded ZeroGravity allocation. After any newer docs-only checkpoint X2
may again be behind l5; sync it to exact current l5 `main` before
new physical tests without push transport. Selected CUDA2B B8
serving still uses the canonical-boundary private B8 recurrent core (`60b048d`),
ragged private B8 KV transition (`775dfa6`) and B8 private-KV CTA256 (`399e43a`).
`874e67d` retains a model-neutral shared-input Ordered+LaneBlocked B8 projection
pair; its ordinary-Qwen consumer is now explicitly rejected after order-sensitive
loaded-model A/B, not merely held. `main` is the only persistent
worktree per host. Detailed CUDA/HIP primitive history, exact measurements,
rejections and blockers remain in [model-execution/TODO.md](model-execution/TODO.md);
do not duplicate them into a second evidence ledger.

## Owner priority — current continuation

**Fabric functional transition accepted:** source `8c43844`, CUDA repair
`8d761c4`; scoped model/GPU/SDK/HTTP/cache acceptance and final isolated gate pass.
Original unsupported/unavailable cells and Qwen expansion remain explicit.
Main only; no new or resumed subagents or recurrence restart.
Approved x2 ROCm vLLM acquisition is complete. The former12GiB available-host
guard was waived with swap for fresh scopes; device safeguards remain unchanged.
Explicit75GiB-total HIP admission is signed`d16f096`; historical defaults unchanged.
Complete substantial implementation batches before combined validation; full gate last.
Prioritize working paths + early matched-vLLM feedback; reuse checks/evidence,
add fixtures only for concrete gaps. Baselines need not await all models/batching.
On l5, `5bf440d` opened the production CUDA graph ownership substrate,
`db7be60` added prepaid opaque graph-resource admission, `6a5c610` closed the
driver-PTX compatibility blocker, `81b7c71` made charge lifetime explicit, and
`f1ce532` now binds replay to the exact resident-flight storage/access authority.
The original0.8B B2 graph path passes physically; one older transient post-capture
byte-accounting mismatch remains a watch item. No production serving selector is
enabled; lifecycle/cache + selector integration is next.
Owner direction: l5 owns repository/source changes; x2 is only for AMD
large-VRAM execution/testing. Tiny HIP health and the full canonical27B host
runtime source/tokenizer path are green. X2 is a **shared GPU host** with Ollama
and other model workloads: existing KFD owner `2406838` / ~31.1GB is baseline,
not a launch blocker and not safely attributable to ZeroGravity from sandboxed
`/proc`. Before our physical work record the current VRAM/KFD baseline, run only
within available capacity, and verify our incremental delta retires afterward.
Never reset/reclaim/restart/kill other owners. Preserve the unchanged180s contract
and missing-reference blockers; no X2 build/device campaign is live at this
checkpoint.

Fabric migration is complete; the unfinished Qwen campaign remains next.
Preserve signed native GDN`e2559b6`/pause`1c35172` and all accepted bounded
model/lifetime evidence. Recurrence`17aafa71` remains cancelled; no implicit
hardware restart. S0 → S1–S3 → F0–F4 → Q0/Q1/Q2 are accepted within their named
scopes. Return to the Qwen model/context/C1..C64/vLLM mission without silently
activating held routes.

## Active

- [ ] P1 serving-and-speculation — @pvl — l5 CUDA `.8B`/`2B`, x2 ROCm `.8B`/`2B`/`27B`: correctness/context fit, C1..C64, then measured vLLM superiority; finite27B checkpoint passed; populated0.8B/2B HIP SDK/HTTP accepted; final gate passes; next CUDA/general27B/native batching; no recurrence restart — [details](serving-and-speculation/TODO.md)
- [~] P1 numerics-and-context — @pvl — exact cached47→50, lifecycle/capacity and final gate pass; active-context/2B reference inputs missing — [details](numerics-and-context/TODO.md)
- [~] P1 memory-and-streaming — @pvl — exact two-root admission: `.8B` fits through 32K, `2B` through 16K, `2B`32K refuses before allocation — [details](memory-and-streaming/TODO.md)
- [~] P1 cpu-gpu-optimization — @pvl — ordinary kernel/transport checks pass; exact whole-model parity before measured tuning and the matched benchmark campaign — [details](cpu-gpu-optimization/TODO.md)
- [~] P1 pipeline-composition — @pvl — recursive primitives/pipelines/profiles; shared execution adapters active, reuse current adapters in Fabric; Qwen expansion deferred — [details](pipeline-composition/TODO.md)
- [~] P1 cli-device-routing — @pvl — unified run/config/source/device selection; preserve behavior through Fabric SDK/CLI migration — [details](cli-device-routing/TODO.md)
- [~] P2 model-execution — @pvl — l5 CUDA2B B8 currently selects exact independent reductions, same-frontier ragged/chunked cohorts, private B8 recurrent core `60b048d`, ragged private B8 KV `775dfa6` and CTA256 KV `399e43a`; `874e67d` retains the exact shared-input Ordered+LaneBlocked B8 projection pair and its serving consumer remains rejected after non-repeatable loaded-model A/B. `6a5c610` fixes nonpersistent NVRTC driver compatibility and physically accepts the B2 CUDA graph resource lifecycle above `db7be60`. Large same-contract CUDA numerical acceptance remains blocked by missing approved reference material. Populated small HIP is `8ffb511`; resident27B C1..C64 is `d16f096`; `c282f20`/`134a571` repair HIP cross-feature boundaries; `b0cf0b2` removes duplicate Qwen snapshot hashing with an explicitly non-authoritative descriptor lease. X2 exact tokenizer/source binding and full canonical host runtime pass. X2 is shared with Ollama/model workloads: existing KFD/VRAM is baseline, so physical27B/ROCm2B work may proceed when the requested configuration fits current available capacity, while preserving other owners and verifying only ZeroGravity's incremental resource delta retires — [details](model-execution/TODO.md)
- [~] P2 model-acquisition — @pvl — [details](model-acquisition/TODO.md)

## Icebox

- [ ] multimodal — [details](multimodal/TODO.md)
- [ ] distributed-execution — [details](distributed-execution/TODO.md)
- [ ] platform-and-release — [details](platform-and-release/TODO.md)

## Closed

- [x] fabric-vnext-transition — @pvl — source `8c43844`; S/F and scoped Q0/Q1/Q2 accepted, deferred expansion unchanged — [details](fabric-vnext-transition/TODO.md)
- [x] accepted-baseline — `6d74f76` — 2026-10-18
- [x] handoff-cleanup — `bfe672c` — 2026-10-18
- [x] historical-records — `9239719` — 2026-10-18
