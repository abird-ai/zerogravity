# Q4 CPU operators

Current owner: [source.rs](../../../source.rs).

Q4_0 and Q4_1 source GEMV bodies exist. Q4_0 also has the distinct
8-row repack schedule. Q4_K scalar decoding exists, but
`source_reference_projection_with_profile` explicitly returns `Unsupported`
for Q4_K: a diagnostic dequantized dot is not a qualified Q4_K × Q8_K runner.

## TODO

- [ ] Qualify the exact variant, input coding, reduction/cast schedule and
  supported geometry against independent source-reference operators.
- [ ] Preserve bounded workspace, source-owner joins, cancellation and finite
  results if moving the existing arithmetic into this family directory.
- [ ] Implement and independently qualify Q4_K × Q8_K before removing its
  explicit CPU projection refusal; scalar codec tests cannot authorize it.
- [ ] Qualify complete models separately; no bit-width-wide support claim.

Source layout: [Q4 codec](../../../../../zerogravity-artifacts/src/quant/q4/README.md).
Required acceptance: [format matrix and qualification](../../../../../../.agents/plans/pipeline-composition/FORMATS.md).
