# Q6 CPU operators

Current owner: [source.rs](../../../source.rs).

Q6_K × signed dynamic Q8_K uses its existing integer groups, local scales
and eight-lane F32 reduction. Other six-bit variants remain unrecognized.

## TODO

- [ ] Qualify the exact variant, input coding, reduction/cast schedule and
  supported geometry against independent source-reference operators.
- [ ] Preserve bounded workspace, source-owner joins, cancellation and finite
  results if moving the existing arithmetic into this family directory.
- [ ] Qualify complete models separately; no bit-width-wide support claim.

Source layout: [Q6 codec](../../../../../zerogravity-artifacts/src/quant/q6/README.md).
Required acceptance: [format matrix and qualification](../../../../../../.agents/plans/pipeline-composition/FORMATS.md).
