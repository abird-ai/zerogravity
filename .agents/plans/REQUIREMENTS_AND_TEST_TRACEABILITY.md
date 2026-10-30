# Requirements and test traceability

This is the stable repository target for acceptance/evidence navigation. It is
not a historical matrix of every prior run. Cross-window context starts at
[CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).

## Governing rule

A requirement is satisfied only when current source implements it and the
appropriate current tests or physical qualification pass. Git history preserves
older evidence; deleted handoffs, checkouts and transient run directories are
not acceptance inputs.

## Traceability map

| Area | Current requirement / implementation owner | Evidence location |
|---|---|---|
| Architecture and boundaries | [plan index](README.md), [active queue](TODO.md), [Fabric vNext](fabric-vnext/README.md) | architecture checks, crate tests, owning subsystem TODO |
| Model execution | [model execution](model-execution/TODO.md) | model/unit/integration tests plus required trained-model physical gates |
| Numerics and state | [numerics and context](numerics-and-context/TODO.md) | exact/reference tests, state/fault/cancellation gates |
| CPU/GPU optimization | [CPU/GPU optimization](cpu-gpu-optimization/TODO.md) | backend-specific qualification and matched benchmarks |
| Serving | [serving and speculation](serving-and-speculation/TODO.md) | CLI/server tests, concurrency/cancellation and physical serving gates |
| Memory/streaming | [memory and streaming](memory-and-streaming/TODO.md) | ownership, bounded-memory, cache/SSD and fault tests |
| Acquisition/cache | [model acquisition](model-acquisition/TODO.md) | artifact/cache/acquisition transaction and provider tests |
| Pipeline composition | [pipeline composition](pipeline-composition/TODO.md) | composition/configuration/state-isolation tests |
| Platform/release | [platform and release](platform-and-release/TODO.md), [release gates](RELEASE_ACCEPTANCE_GATES.md) | repository checks, packaging/install/release gates |
| Public architecture | [Engineering Specification](../../docs/ENGINEERING_SPECIFICATION.md), [Whitepaper](../../docs/WHITEPAPER.md) | current source and the above gates override dated design assumptions |

## Evidence discipline

- Keep stable fixtures and reproducible test inputs when they are needed by
  current tests.
- Do not make machine-local absolute paths, checkout names or transient run
  directories part of a permanent acceptance contract.
- Historical benchmark/result prose may explain a decision but cannot override
  current source, tests or a later qualification.
- Physical/hardware claims remain scoped to the exact device/backend/source
  cell actually tested.

## Current open physical qualification edge

- Approved Qwen3.5 .8B model bytes are visible and exact.
- `.8B` native/reference inputs are recovered offline with exact current Rust payload/provider identities; fresh phase-1 cached47→50 capture and source retirement pass.
- Owned GEMM/RMS/rotary/attention pass exact `.8B` cached47→50: full logits/all42/natural feedback.180 synthetic and eight authenticated GEMMs, two authenticated attentions,12 analytic/refusal attention cases and16 staged CUDA RMS cases pass with guards/retirement. Original cuBLAS private exports remain refused; no opaque passthrough.
- Owner resumed. Real post-launch cancellation/abandonment preserves prior bits/tails/root and original quarantine charges. Normal lifecycle passes all13 admitted capacities at48 active tokens;2B32K refuses at pure preflight. Long-active-context/2B numerical coverage is separate and open.
- Current complete repository gate passes2,129/0failed/83ignored; CUDA/native host675/0failed and strict CUDA/HIP/Vulkan feature checks pass. Production stays disabled. Missing exact2B/long-context reference inputs block broader acceptance; ROCm (`pvl-x2`) has only read-only readiness inspection. See the owning numerics TODO.
