# ZeroGravity terminology

Use names that describe the engineering concept, not the qualification step or
historical fixture that introduced it. Keep names short, but make dimensions and
authority explicit.

## Canonical vocabulary

- **Model variant** — immutable model geometry/family member. For Qwen use
  `QwenModelVariant::{Qwen800M,Qwen2B}`. Do not call model size a source
  profile.
- **Serving profile** — request/runtime capacity policy such as active requests,
  context, output and cohort width. Use `QwenServingProfile`; backend choice
  belongs to the load request, not the profile type.
- **Numerical profile / numerical contract** — arithmetic semantics only. Use
  `NumericalProfile` and `NumericalContract`; names such as
  `CudaSm86Bf16ReferenceV1` describe the actual arithmetic identity and do not
  imply backend eligibility.
- **Cohort width** — number of rows/requests executed together by one fused
  path. Use `cohort_width`, **singleton**, and **2-row / 4-row / 8-row
  cohort**. Do not use `native_rows` or bare `B1/B2/B4/B8`.
- **Fused cohort** — scheduler/backend work that actually executes multiple
  rows together. Scheduler policy uses `Cohorts::Fused`; membership planning
  uses `partition_cohorts` / `partition_supported_cohorts`.
- **Reference capture** — immutable qualification/reference material. Arrange
  it structurally by model variant, phase and shape, e.g.
  `reference/qwen800m/prefill/47`, rather than opaque names such as
  `qwen08_47`.
- **Text invocation** — one phase-qualified token span. The semantic pipeline is
  `InvocationInput -> BoundInvocation -> TextInvocation { TokenSpan,
  InvocationPhase }`. Use `tokens()` and `phase()`, not `rows()` / `mode()` on
  this type.
- **Model invocation** — model-family semantic metadata layered above the core
  text invocation. Qwen uses `PromptInvocation`, `QwenGdnInvocation` and
  `QwenModelInvocation` rather than `*Call` names.
- **Resident text** — live model/session/pending/readback/snapshot ownership for
  source-backed text execution. Use `ResidentText*`; the backend program body is
  `text_execution`, and observation uses `TextFrame` / `TextObserver`.

## Reserved words

- **native** is only for genuinely backend/platform-native concepts. Prefer
  `fused`, `cubin`, `device`, or another exact mechanism when that is what
  is meant.
- **forward** is the canonical term for the normal source-backed model
  execution path. Qwen model semantics use `QwenForward*`, authenticated source
  authority uses one backend-neutral `ForwardSourceSeal`, and CUDA mechanics use
  `cuda_forward` / `CudaForward*`. The historical `ordinary*` Qwen execution
  vocabulary is retired; do not reintroduce it.
- **standard RMSNorm** is a separate arithmetic distinction from the forward
  execution path. Historical `ordinary RMS` names are not mechanically
  `forward RMS`; rename them only when their exact numerical distinction has a
  clear replacement.
- **autoregressive** is the canonical term for committed model output fed into
  the next decoding step. Runtime/input/state/publication/reuse ownership uses
  `Autoregressive*` and `autoregressive_*`; the historical `natural*` runtime
  vocabulary is retired.
- **all state domains** is the canonical term for the complete model state-domain
  set. Do not encode the current Qwen domain count into semantic names; `all42`
  is retired outside frozen archive/schema/provenance coordinates.
- **full model** is the canonical term for complete model execution/body/program
  ownership. Use `full_model` in identifiers; historical `full24` is retired.
- **all layers** is the canonical term for coverage across every model layer.
  Use `all_layers` in machine-readable coverage labels; historical `all24` is
  retired outside frozen provenance fields.
- **full vocabulary** is the canonical term for complete vocabulary/logit
  coverage. Use `full_vocabulary` in machine-readable labels; historical
  `fullV` / `fullv` are retired outside frozen provenance fields.
- **prefill 47** is the canonical name for the fixed 47-token prefill trajectory;
  use `prefill47` in identifiers/receipt labels. Describe continuation explicitly
  by decode bases/frontiers or, for the accepted parity path, **decode 48–50**.
  Historical `canonical47`, `cached47`, `true47` and `case5` are retired outside
  frozen provenance fields.

Historical commit messages, externally stable schema strings and provenance
identifiers are not rewritten merely for style. Active APIs, modules, tests,
diagnostics and current documentation should use the canonical vocabulary.

The historical `SourceCall*` family is retired. Do not reintroduce `source
call` as a semantic or runtime-owner term. Externally stable historical schema
or status values may retain the old spelling behind their existing compatibility
boundary; active APIs, env vars, diagnostics and tests use invocation/resident-text
terminology.

Frozen external reference packages may retain `natural` in versioned schemas,
artifact filenames, receipt fields and historical diagnostic entrypoint names;
those are compatibility/provenance boundaries, not runtime vocabulary.

The historical `all42` shorthand is retired from active code, tests and
current documentation. Frozen reference archive coordinates/keys may retain it
at their compatibility boundary.

The historical `full24` / `all24` / `fullV` shorthand is retired from active
code, tests and current documentation. `full24` maps to full-model execution,
`all24` to all-layer coverage, and `fullV` to full-vocabulary coverage. Frozen
acceptance/schema keys may retain the old spelling at their compatibility boundary.

The historical `canonical47` / `cached47` / `true47` / `case5` trajectory
shorthand is retired from active code and current documentation. The frozen
`canonical47-repeated-v1` oracle profile remains unchanged at its compatibility
boundary.

The historical provider/workload shorthand `active8`, `temporal4` and
`quantum32` is retired from active terminology. The fixed-prompt lifecycle uses
**eight-session benchmark profile** / `EightSessionBenchmark`; the smaller lifecycle
profile is `TwoSession`. Production temporal
prefill is described as **4-token equal-frontier prefill chunks**; and generic
scheduler configuration uses the explicit **32-token prefill quantum** when that
value matters.

The next isolated terminology debt is residual row-width/concurrency shorthand
(`B2`/`B4`/`B8`, `native8`, `C1`/`C4`/`C8`, etc.) where it is active engineering
vocabulary rather than an ABI, artifact, or historical benchmark coordinate.
Keep the separate standard-RMS naming problem isolated until its exact arithmetic
distinction has a clear replacement.
