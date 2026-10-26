# GPU source quantization

Shared packed-source arithmetic and activation coding live here. Provider
compiler/ABI/runtime policy stays under `cuda/`, `hip/` and `vulkan/`; format
names grant no provider capability. Existing kernel files remain in place.

- [q1](q1/README.md)
- [q2](q2/README.md)
- [q3](q3/README.md)
- [q4](q4/README.md)
- [q6](q6/README.md)
- [q8](q8/README.md)
- [nvfp4](nvfp4/README.md)

BF16 has a separate [precision home](../precision/bf16/README.md).

Shared owners:

- [Projection host preparation and physical readback](projection.rs)
- [Dynamic activation coding](activation.rs) and [fused preparation](activation_fused.rs)
- [Explicit batch layouts and kernels](batch_projection.rs)
- [Declared accumulation/capability policy](../source_accumulation.rs)

Preserve the exact source binding, activation codec, reduction/cast schedule,
feature gates and selected workload guards. Shared CubeCL source is not proof
of CUDA, ROCm (HIP) or Vulkan execution. Complete model support remains model
adapter/source/state/physical qualification under the [format matrix](../../../../.agents/plans/pipeline-composition/FORMATS.md).
