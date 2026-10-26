# Q8 source formats

| Exact variant | Logical elements/block | Bytes/block |
|---|---:|---:|
| `Q8_0` | 32 | 34 |

Existing private scalar bodies: [q8.rs](../q8.rs).
Common byte-length/allocation checks: [quant.rs](../../quant.rs).

Q8_0 preserves signed codes and its F16 scale. Dynamic Q8_1/Q8_K activation
coding in operators is separate from this recognized source storage variant.

## TODO

- [ ] Extend independent source fixtures for each variant's scale/sign/offset
  edge cases; retain existing nonfinite-scale and exact-length refusals.
- [ ] Add any further variant only with its own pinned byte contract and
  malformed/truncated/overflow tests; keep other encodings unsupported.
- [ ] Qualify each consuming operator separately from scalar decoding.

Shared metadata authority: [QuantKind](../../../../zerogravity-core/src/quant.rs).
Execution/qualification: [format matrix](../../../../../.agents/plans/pipeline-composition/FORMATS.md).
