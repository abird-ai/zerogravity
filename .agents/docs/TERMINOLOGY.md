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

## Reserved words

- **native** is only for genuinely backend/platform-native concepts. Prefer
  `fused`, `cubin`, `device`, or another exact mechanism when that is what
  is meant.
- **forward** is the canonical term for the normal source-backed model
  execution path. Historical `ordinary*` code is being migrated to
  `forward*`; do not introduce new `ordinary` names.
- **autoregressive** is preferred over historical `natural` for committed
  model output fed into the next decoding step.
- **all state domains** is preferred over `all42`; the number is a model
  property, not the concept.
- Describe fixed qualification trajectories semantically, e.g.
  **prefill 47 + decode 48–50**, rather than `cached47`, `canonical47`, or
  `case5`.

Historical commit messages, externally stable schema strings and provenance
identifiers are not rewritten merely for style. Active APIs, modules, tests,
diagnostics and current documentation should use the canonical vocabulary.
