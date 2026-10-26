# BF16 precision

BF16 is a floating-point precision, separate from integer quantization and
NVFP4 source-format contracts. File encoding, activation/output casts and
accumulation/reduction policy remain distinct fields.

Existing arithmetic/transport owners:

- [source.rs](../../source.rs): explicit `TorchCpuBf16Avx2` projection contract
- [source/projection.rs](../../source/projection.rs): checked output-first source binding
- [source/window_projection.rs](../../source/window_projection.rs): authenticated full-width row windows
- [numerics/torch_cpu.rs](../../numerics/torch_cpu.rs): operation-specific numerical implementations

## TODO

- [ ] Keep BF16-exact activation, source/shape, F32 accumulation and final-cast
  responsibilities explicit; an unaware provider must continue to refuse.
- [ ] Qualify every selected operator/model/source/context profile independently;
  no generic BF16 model or CPU/GPU equivalence grant.

Qualification: [format matrix](../../../../../.agents/plans/pipeline-composition/FORMATS.md) and
[common correctness policy](../../../../../.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md).
