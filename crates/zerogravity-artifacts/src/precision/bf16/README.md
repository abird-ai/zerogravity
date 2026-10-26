# BF16 precision

BF16 is a floating-point precision, separate from integer quantization and
NVFP4 source-format contracts. File encoding, activation/output casts and
accumulation/reduction policy remain distinct fields.

Existing scalar BF16 decoding remains in [quant.rs](../../quant.rs):
little-endian 16-bit words are widened by their exact bit representation.
It retains floating special values; consumers enforce their own finite-input
policy. [QuantKind](../../../../zerogravity-core/src/quant.rs) recognizes BF16 metadata;
[artifact readers](../../artifact.rs) validate bounded byte geometry.

## TODO

- [ ] If relocating existing float utilities, move all actual callers together,
  preserve exact bit/cast semantics and requalify new source/type identities.
- [ ] Retain independent special-value, byte-order, truncation and overflow
  fixtures. Do not infer operator or model support from a scalar decoder.

Qualification: [format matrix](../../../../../.agents/plans/pipeline-composition/FORMATS.md) and
[common correctness policy](../../../../../.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md).
