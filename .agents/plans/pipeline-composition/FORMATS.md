Author: Project owner
Co-authors: GPT-6.1 Sol xhigh
Owner: @pvl; ROOT coordinates integration

# Source formats and precision homes

These homes organize existing code and explicit TODOs. They add no source tag,
codec, operator, backend capability or complete model support. A bit count does
not identify a byte layout. BF16 belongs to precision; NVFP4 requires its own
actual floating code/scale format rather than an integer Q4 substitution.

## Ownership

| Responsibility | Home |
|---|---|
| Source metadata and bounded scalar codec | `artifacts/src/quant/`, existing artifact readers and core `QuantKind` |
| CPU source operators | `cpu/src/source/quantization/`, existing `source.rs` and exact `numerics/` contracts |
| Shared GPU packed-source arithmetic | `gpu/src/quantization/` |
| Device compiler, ABI and physical implementation | Existing GPU `cuda/`, `hip/`, `vulkan/` owners |
| BF16 precision | `artifacts/cpu/gpu` `src/precision/bf16/` documentation and linked real implementations |
| Family roles, source eligibility and complete model traversal | Existing model crate |

Here `artifacts`, `cpu` and `gpu` mean the existing `zerogravity-*` crates.
Reserved homes contain documentation only; add Rust modules when real code
exists. Do not create a format registry, alternate graph/runtime, unsupported
compute stub or compatibility alias.

## Current source inventory

“Body exists” describes live source, not executed qualification. CUDA, HIP and
Vulkan are separate implementations/qualification scopes even when sharing a
CubeCL body. This inventory does not revoke or expand earlier bounded evidence.

| Exact format | Metadata/scalar codec | CPU source projection | CUDA | HIP | Vulkan | Complete model support |
|---|---|---|---|---|---|---|
| Q1 family | Not implemented; exact variant unspecified | None | TODO | TODO | TODO | Not granted |
| Q2 family | Not implemented; exact variant unspecified | None | TODO | TODO | TODO | Not granted |
| Q3 family | Not implemented; exact variant unspecified | None | TODO | TODO | TODO | Not granted |
| Q4_0 | Existing: 32 elements / 18 bytes | Source and distinct 8-row repack bodies | Shared body; qualify selected path | Shared body; qualify selected path | Shared body; qualify selected path | Adapter/source/profile specific |
| Q4_1 | Existing: 32 / 20 | Source body, explicit offset | Shared body; qualify selected path | Shared body; qualify selected path | Shared body; qualify selected path | Adapter/source/profile specific |
| Q4_K | Existing: 256 / 144 | Explicitly Unsupported in source projection | No kernel here; TODO | No kernel here; TODO | No kernel here; TODO | Not granted by codec |
| Q6_K | Existing: 256 / 210 | Source body, signed dynamic Q8_K | Shared body; qualify selected path | Shared body; qualify selected path | Shared body; qualify selected path | Adapter/source/profile specific |
| Q8_0 | Existing: 32 / 34 | Source body; separate closed OpenBLAS F32 contract | Shared body; qualify selected path | Shared body; qualify selected path | Shared body; qualify selected path | Adapter/source/profile specific |
| NVFP4 | Not implemented; exact packing/scales unspecified | None | TODO | TODO | TODO | Not granted |
| BF16 precision | Existing: exact 16-bit decode | Explicit numerical bodies/windows | Native/shared bodies; qualify selected path | Shared bodies; qualify selected path | Shared bodies; qualify selected path | Adapter/source/profile specific |

The existing CPU Q4_K refusal remains required despite scalar decoding and a
diagnostic dequantized dot. Q4_0/Q4_1/Q4_K, Q6_K and Q8_0 are exact variants;
no generic Q4/Q6/Q8 packing is recognized. Dynamic Q8_1/Q8_K activation
coding does not add corresponding source-storage enum variants.

## Concrete family homes

| Family | Source codec / metadata TODO | CPU operator TODO | GPU CUDA/HIP/Vulkan TODO |
|---|---|---|---|
| q1 | [source](../../../crates/zerogravity-artifacts/src/quant/q1/README.md) | [CPU](../../../crates/zerogravity-cpu/src/source/quantization/q1/README.md) | [GPU](../../../crates/zerogravity-gpu/src/quantization/q1/README.md) |
| q2 | [source](../../../crates/zerogravity-artifacts/src/quant/q2/README.md) | [CPU](../../../crates/zerogravity-cpu/src/source/quantization/q2/README.md) | [GPU](../../../crates/zerogravity-gpu/src/quantization/q2/README.md) |
| q3 | [source](../../../crates/zerogravity-artifacts/src/quant/q3/README.md) | [CPU](../../../crates/zerogravity-cpu/src/source/quantization/q3/README.md) | [GPU](../../../crates/zerogravity-gpu/src/quantization/q3/README.md) |
| q4 | [source](../../../crates/zerogravity-artifacts/src/quant/q4/README.md) | [CPU](../../../crates/zerogravity-cpu/src/source/quantization/q4/README.md) | [GPU](../../../crates/zerogravity-gpu/src/quantization/q4/README.md) |
| q6 | [source](../../../crates/zerogravity-artifacts/src/quant/q6/README.md) | [CPU](../../../crates/zerogravity-cpu/src/source/quantization/q6/README.md) | [GPU](../../../crates/zerogravity-gpu/src/quantization/q6/README.md) |
| q8 | [source](../../../crates/zerogravity-artifacts/src/quant/q8/README.md) | [CPU](../../../crates/zerogravity-cpu/src/source/quantization/q8/README.md) | [GPU](../../../crates/zerogravity-gpu/src/quantization/q8/README.md) |
| nvfp4 | [source](../../../crates/zerogravity-artifacts/src/quant/nvfp4/README.md) | [CPU](../../../crates/zerogravity-cpu/src/source/quantization/nvfp4/README.md) | [GPU](../../../crates/zerogravity-gpu/src/quantization/nvfp4/README.md) |
| BF16 precision | [source](../../../crates/zerogravity-artifacts/src/precision/bf16/README.md) | [CPU](../../../crates/zerogravity-cpu/src/precision/bf16/README.md) | [GPU](../../../crates/zerogravity-gpu/src/precision/bf16/README.md) |

## Acceptance before support changes

- [ ] Pin the exact released variant: original container metadata, byte order,
  block sizes, codes, scales/offsets, exceptional values and source hashes.
  Add recognition only with independently checked codec fixtures and bounded
  malformed/truncated/overflow refusals. Unknown variants remain Unsupported.
- [ ] Declare each operator's weight/input/output layout, activation coding,
  accumulation/reduction, rounding/cast and geometry contract before measuring.
  Preserve exact existing source-bit limits; a different contract is separately
  qualified rather than silently substituting dequantized F32 or BF16.
- [ ] Validate immutable source backing/identity/ranges, actual producer and
  TeacherSpan/SSA/state joins, ordered matrix segments and complete outputs.
  Descriptors and folder names mint no source owner, root or native permit.
- [ ] Qualify CPU, CUDA, HIP and Vulkan independently using actual original
  source and the selected compiler/SDK/driver/device/source/ABI identities.
  Exercise finite status, source refusals, cancellation, resource accounting,
  completion/fences, quarantine and healthy retirement. Keep gfx90c on hold.
- [ ] Qualify each complete model/checkpoint/profile/context against its pinned
  independent reference with declared model accuracy and state checks; operator
  fixtures do not fill missing model cells. Preserve failed/unsupported records.
- [ ] ROOT reviews assembled source, affected tests and final `scripts/check.sh`.
  Moving identical arithmetic changes source/type/binary/cache identity where
  applicable; do not attach old physical evidence to a renamed binary.

[The common correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md) governs distinct semantic,
operator/model accuracy and determinism claims. This layout adds no tolerance
and no requirement for universal cross-batch or cross-backend bit equality.
