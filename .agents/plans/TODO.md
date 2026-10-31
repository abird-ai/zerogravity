# Plan queue

Current source, tests, and this queue are authoritative; Git retains retired history.
Cross-window entrypoint: [CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).

## Owner priority — 31 Oct

Fabric vNext is now authorized and precedes the unfinished Qwen campaign.
Preserve signed native GDN`e2559b6`/pause`1c35172` and all accepted bounded
model/lifetime evidence. Recurrence`17aafa71` remains cancelled; no implicit
hardware restart. S0 → S1–S3 → F0–F4 → Q0/Q1/Q2, then return to the Qwen
model/context/C1..C64/vLLM mission. No production activation from refactoring.

## Active

- [~] P0 fabric-vnext-transition — @pvl — S0 signed`7396984`; S1 exact codec moves next, then feature-preserving migration and assembled qualification — [details](fabric-vnext-transition/TODO.md)
- [ ] P1 serving-and-speculation — @pvl — l5 CUDA `.8B`/`2B`, x2 ROCm `.8B`/`2B`/`27B`: correctness/context fit, C1..C64, then measured vLLM superiority; finite27B checkpoint passed; ordinary HIP remains open; deferred behind Fabric — [details](serving-and-speculation/TODO.md)
- [~] P1 numerics-and-context — @pvl — exact cached47→50, lifecycle/capacity and final gate pass; active-context/2B reference inputs missing — [details](numerics-and-context/TODO.md)
- [~] P1 memory-and-streaming — @pvl — exact two-root admission: `.8B` fits through 32K, `2B` through 16K, `2B`32K refuses before allocation — [details](memory-and-streaming/TODO.md)
- [~] P1 cpu-gpu-optimization — @pvl — ordinary kernel/transport checks pass; exact whole-model parity before measured tuning and the matched benchmark campaign — [details](cpu-gpu-optimization/TODO.md)
- [~] P1 pipeline-composition — @pvl — recursive primitives/pipelines/profiles; shared execution adapters active, reuse current adapters in Fabric; Qwen expansion deferred — [details](pipeline-composition/TODO.md)
- [~] P1 cli-device-routing — @pvl — unified run/config/source/device selection; preserve behavior through Fabric SDK/CLI migration — [details](cli-device-routing/TODO.md)
- [~] P2 model-execution — @pvl — ordinary CUDA .8B cached47→50/fault/shutdown pass; broader context/2B numerical coverage remains before route activation — [details](model-execution/TODO.md)
- [~] P2 model-acquisition — @pvl — [details](model-acquisition/TODO.md)

## Icebox

- [ ] multimodal — [details](multimodal/TODO.md)
- [ ] distributed-execution — [details](distributed-execution/TODO.md)
- [ ] platform-and-release — [details](platform-and-release/TODO.md)

## Closed

- [x] accepted-baseline — `6d74f76` — 2026-10-18
- [x] handoff-cleanup — `bfe672c` — 2026-10-18
- [x] historical-records — `9239719` — 2026-10-18
