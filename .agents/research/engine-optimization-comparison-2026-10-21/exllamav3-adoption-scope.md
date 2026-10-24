# ExLlamaV3 adoption investigation: scope and initial evidence

21 Oct 2026. Requested follow-up to the eight completed engine/recipe studies.
This is **preliminary scoping**, not a ninth completed audit, an extension of the
416-entry count, code incorporation or runtime qualification.

## Pinned initial observations

- Repository: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3).
  Inspected master identity `da68d57de61781e5b74ad68698ac4caf36408d7a`,
  committed 20 Oct 2026 23:51:51 UTC, subject `Bump to v1.6.4`.
- Root [LICENSE](https://github.com/turboderp-org/exllamav3/blob/da68d57de61781e5b74ad68698ac4caf36408d7a/LICENSE)
  is MIT. Only this root license was inspected here; file-level/transitive
  license and NOTICE closure is still pending before any implementation reuse.
- Inspected build/dependency lines in
  [setup.py](https://github.com/turboderp-org/exllamav3/blob/da68d57de61781e5b74ad68698ac4caf36408d7a/setup.py),
  [ext.py](https://github.com/turboderp-org/exllamav3/blob/da68d57de61781e5b74ad68698ac4caf36408d7a/exllamav3/ext.py)
  and [pyproject.toml](https://github.com/turboderp-org/exllamav3/blob/da68d57de61781e5b74ad68698ac4caf36408d7a/pyproject.toml).
  Packaging depends on torch and uses its C++/CUDA extension tooling. This is
  not proof of a self-contained statically linkable C/Rust library. Audit leaf
  dependencies before choosing direct linkage, a narrow C shim or a Rust port.
- No complete kernel/model call chain, standalone ABI, imported file license,
  format fixture, test suite or physical device behavior was established here.
  No upstream source was written into the workspace or executed.

## Full investigation still pending

1. Trace codec reader → transforms/codebooks/scales → packing → dense/grouped
   expert consumers. Record supported widths, tails, layouts, architectures,
   source/model restrictions and malformed-format handling; build independent
   scalar fixtures for exact decode and projections.
2. Inspect GEMV/GEMM regime selection, fused boundaries, graph/buffer reuse,
   low-bit KV, attention/cache layouts, dynamic batches and speculation. Classify
   exact representations, changed arithmetic and explicit quality compromises.
3. Inspect tensor/expert partitioning and CPU offload for transferable leaf
   components; defer the cluster/model scope rather than import another engine.
4. Produce a component-by-component reuse disposition: pinned files/notices,
   direct Rust/static C ABI/shim viability and dependencies, or a direct port
   with upstream fixtures, independent checks and target architecture seam.
5. Map findings into existing plans without duplicating the already designed
   [derived codec/calibration workflow](../../docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md).
   Numerical/quality/static-link/state/physical/performance gates stay separate.

Queue: [optimization follow-up](../../plans/cpu-gpu-optimization/TODO.md).
Integration contract: [profiles, components and reuse](../../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md).
