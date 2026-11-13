# Plan queue

Current source, tests, and this queue are authoritative; Git retains retired history.
Cross-window entrypoint: [CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).

Last implementation/probe-bearing l5 checkpoint before the 14 Nov 2026 handoff
refresh is signed `cb9e349`; `origin/main` remains `c12acab`. The handoff refresh
may be a newer docs-only signed commit, so takeover must revalidate live HEAD and
ahead count; nothing newer than `c12acab` is pushed at handoff creation.
Selected CUDA2B B8 serving includes the canonical-boundary private B8 recurrent
core (`60b048d`), ragged private B8 KV transition (`775dfa6`) and B8 private-KV
CTA256 (`399e43a`). Later local commits retain useful shared primitives or record
rejected consumers/probes; production source is restored after the docs-only probe
checkpoints. `main` is the only persistent worktree. Detailed CUDA/HIP primitive
history, exact measurements, rejections and blockers remain in
[model-execution/TODO.md](model-execution/TODO.md); do not duplicate them into a
second evidence ledger.

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
Owner direction: CUDA + ordinary27B, quick real-model/serving feedback. Shared
CUDA and resident HIP checkpoints are signed; no build/device campaign is live.
This closeout reconciles notes and commits current changes, without a new launch.

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
- [~] P2 model-execution — @pvl — l5 CUDA2B B8 currently selects exact independent reductions, same-frontier ragged/chunked cohorts, private B8 recurrent core `60b048d`, ragged private B8 KV `775dfa6` and CTA256 KV `399e43a`; independently useful grouped transport/greedy/head/Conv/QK/certification primitives remain retained while unprofitable serving consumers stay rejected. Large same-contract CUDA numerical acceptance is blocked by missing approved reference material on l5. Populated small HIP is `8ffb511`; resident27B C1..C64 is `d16f096`, while HTTP/context closure and ROCm2B mismatch localization remain x2 work — [details](model-execution/TODO.md)
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
