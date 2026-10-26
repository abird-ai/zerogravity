# Q4 GPU operators

Existing shared kernel owners:

- `Q4_0`: [q4_0.rs](../q4_0.rs)
- `Q4_1`: [q4_1.rs](../q4_1.rs)
- `Q4_K`: no source-matched GPU kernel in this format owner.

Q4_0 × dynamic Q8_0 and Q4_1 × dynamic Q8_1 are distinct operations.
Q4_K scalar decoding cannot select either kernel. Preserve the Q4_0
complete-eight-row eligibility and the Q4_1 zero-point contribution.

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

Source layout: [Q4 codec](../../../../zerogravity-artifacts/src/quant/q4/README.md).
Required acceptance: [format matrix and qualification](../../../../../.agents/plans/pipeline-composition/FORMATS.md).
