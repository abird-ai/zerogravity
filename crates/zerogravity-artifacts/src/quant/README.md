# Source quantization formats

[Format ownership and qualification matrix](../../../../.agents/plans/pipeline-composition/FORMATS.md).

`quant.rs` keeps the existing public scalar decoder, byte-length checks and
allocation bound. Private `q4.rs`, `q6.rs` and `q8.rs` own the integer decoding
bodies; each family directory documents exact variants and pending work.

- [q1](q1/README.md)
- [q2](q2/README.md)
- [q3](q3/README.md)
- [q4](q4/README.md)
- [q6](q6/README.md)
- [q8](q8/README.md)
- [nvfp4](nvfp4/README.md)

[BF16 precision](../precision/bf16/README.md)
is separate from quantized formats. F16/F32 scalar utilities remain at their
existing public paths. `QuantKind` is the existing source storage enum; this
layout introduces no tags, capabilities, codecs or model eligibility.

Metadata parsing belongs to [artifact readers](../artifact.rs) and
[QuantKind](../../../zerogravity-core/src/quant.rs); immutable backing belongs to
[SourceWeights](../source_weights.rs). An unknown encoding remains
`Unsupported`. Successful scalar decoding does not authorize execution.
