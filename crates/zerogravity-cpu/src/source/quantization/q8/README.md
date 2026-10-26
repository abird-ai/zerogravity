# Q8 CPU operators

Current owner: [source.rs](../../../source.rs).

Q8_0 × dynamic Q8_0 source GEMV exists. The separately closed
[OpenBLAS 0.3.33 ZEN arithmetic](../../../numerics/zen_blas_q8_f32.rs)
is limited to its declared geometry and retained workspace; it is not a
replacement for arbitrary Q8_0 GEMV.

## TODO

- [ ] Qualify the exact variant, input coding, reduction/cast schedule and
  supported geometry against independent source-reference operators.
- [ ] Preserve bounded workspace, source-owner joins, cancellation and finite
  results if moving the existing arithmetic into this family directory.
- [ ] Qualify complete models separately; no bit-width-wide support claim.

Source layout: [Q8 codec](../../../../../zerogravity-artifacts/src/quant/q8/README.md).
Required acceptance: [format matrix and qualification](../../../../../../.agents/plans/pipeline-composition/FORMATS.md).
