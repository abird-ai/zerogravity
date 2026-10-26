# Q8 GPU operators

Existing shared kernel owners:

- `Q8_0`: [q8_0.rs](../q8_0.rs)

Q8_0 uses dynamic Q8_0 activations and the existing eight-lane accumulation.
Other eight-bit formats and alternate reductions require separate contracts.

## TODO

- [ ] CUDA: implement missing exact variants or qualify the existing shared
  body with actual compiler/ABI/source/profile/geometry and device capability.
- [ ] HIP: implement missing exact variants or qualify the existing shared
  body independently; preserve the local gfx90c launch hold.
- [ ] Vulkan: implement missing exact variants or qualify the existing shared
  body independently, including shader arithmetic and binding/grid limits.
- [ ] For each selected path, test authentic source bytes, adversarial codec
  cases, complete output, state/SSA joins, cancellation, completion/fencing,
  resource accounting, quarantine and healthy retirement.
- [ ] Qualify full-model accuracy/state separately; operator source or a folder
  is not a serving capability. Reuse applicable unchanged evidence explicitly.

Source layout: [Q8 codec](../../../../zerogravity-artifacts/src/quant/q8/README.md).
Required acceptance: [format matrix and qualification](../../../../../.agents/plans/pipeline-composition/FORMATS.md).
