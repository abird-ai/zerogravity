# BF16 precision

BF16 is a floating-point precision, separate from integer quantization and
NVFP4 source-format contracts. File encoding, activation/output casts and
accumulation/reduction policy remain distinct fields.

Existing implementation owners (unchanged):

- [bf16_projection_kernel.rs](../../bf16_projection_kernel.rs): explicit CPU-order full-K dot lowering
- [resident_program/bf16_projection.rs](../../resident_program/bf16_projection.rs): prepared authentic streamed/resident operand and original flight
- [cuda/native/prepared_bf16_head.rs](../../cuda/native/prepared_bf16_head.rs): selected native CUDA head

## TODO

- [ ] CUDA: preserve each selected native/shared kernel, actual source/ABI and
  numerical contract; qualify changed identities without retagging old evidence.
- [ ] HIP: qualify the selected implementation independently; preserve the
  local gfx90c launch hold and unsupported path refusals.
- [ ] Vulkan: qualify the selected full-K implementation, row tiling, shader
  arithmetic and binding/grid limits independently.
- [ ] Retain immutable backing, resource admission, leases, actual completion,
  state publication, cancellation/quarantine and healthy retirement tests.

Qualification: [format matrix](../../../../../.agents/plans/pipeline-composition/FORMATS.md) and
[common correctness policy](../../../../../.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md).
