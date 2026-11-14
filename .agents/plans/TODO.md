# Plan queue

Current source, tests, and this queue are authoritative; Git retains retired history.
Cross-window entrypoint: [CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).

Latest implementation checkpoint is signed `b0cf0b2`
(`fix(cache): avoid duplicate Qwen snapshot hashing`), above `134a571`, docs
`d68b649` and the earlier `874e67d` CUDA / `c282f20` HIP checkpoints.
`origin/main` is `d68b649` at this refresh; revalidate live l5 `HEAD`/ahead
count because this plan refresh may add a docs-only commit. X2 is
the approved AMD large-VRAM execution/test host. Tiny-HIP runtime health is green;
exact tokenizer, isolated source binding/plan, and the full canonical host runtime
test now pass; the latter completed in75.20s once the external execution wrapper
kept emitting progress, proving the prior silent host termination was not a Qwen
source bug. `b0cf0b2` remains the shared duplicate-hash fix. A fresh explicit75GiB
public-HTTP liveness probe reached real HIP allocation (~31.376GB VRAM) but its
isolated process then disappeared before any HTTP status. Host-global KFD sysfs
now identifies the sole retained owner `2406838`:31,098,232,832 bytes VRAM,
six queues, `pasid=0`, zero faults/occupancy. X2 has not returned to baseline.
After any newer docs-only checkpoint X2
may again be behind l5; sync it to exact current l5 `main` before
new physical tests without push transport. Selected CUDA2B B8
serving still uses the canonical-boundary private B8 recurrent core (`60b048d`),
ragged private B8 KV transition (`775dfa6`) and B8 private-KV CTA256 (`399e43a`).
`874e67d` retains a model-neutral shared-input Ordered+LaneBlocked B8 projection
pair, but its ordinary-Qwen consumer is not selected. `main` is the only persistent
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
Owner direction: l5 owns repository/source changes; x2 is only for AMD
large-VRAM execution/testing. Tiny HIP health is green and the full canonical27B
host runtime source/tokenizer path now passes. The next explicit75GiB public HTTP
liveness attempt reached HIP allocation but the isolated process disappeared before
HTTP status. Host-global KFD sysfs now shows the sole retained owner `2406838`
with31,098,232,832 bytes and six queues; sandboxed `/proc` had hidden it. A
matched35s non-GPU native-gpu control is green, so generic execution lifetime is
not the10..12s boundary. Treat X2 as physically dirty: no27B or ROCm2B launch until
that KFD owner/VRAM is explicitly recovered or naturally disappears. Preserve the
unchanged180s contract and missing-reference blockers; do not implicitly reset
GPU/service/host state. No build/device campaign is live at this checkpoint.

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
- [~] P2 model-execution — @pvl — l5 CUDA2B B8 currently selects exact independent reductions, same-frontier ragged/chunked cohorts, private B8 recurrent core `60b048d`, ragged private B8 KV `775dfa6` and CTA256 KV `399e43a`; `874e67d` retains the exact shared-input Ordered+LaneBlocked B8 projection pair with serving held. Large same-contract CUDA numerical acceptance remains blocked by missing approved reference material. Populated small HIP is `8ffb511`; resident27B C1..C64 is `d16f096`; `c282f20`/`134a571` repair HIP cross-feature boundaries; `b0cf0b2` removes duplicate Qwen snapshot hashing with an explicitly non-authoritative descriptor lease. X2 exact tokenizer/source binding and full canonical host runtime pass; a fresh explicit75GiB public HTTP liveness probe reached partial resident-weight HIP upload then lost its isolated process before HTTP status. Host-global KFD owner `2406838` retains31,098,232,832 bytes/six queues and explains nearly all ~31.376GB usage. Stop physical27B/ROCm2B until that owner/VRAM is explicitly recovered or disappears — [details](model-execution/TODO.md)
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
