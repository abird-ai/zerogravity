# Q4 source formats

| Exact variant | Logical elements/block | Bytes/block |
|---|---:|---:|
| `Q4_0` | 32 | 18 |
| `Q4_1` | 32 | 20 |
| `Q4_K` | 256 | 144 |

Existing private scalar bodies: [q4.rs](../q4.rs).
Common byte-length/allocation checks: [quant.rs](../../quant.rs).

Q4_0 low/high nibble order, Q4_1 offset and Q4_K packed local scales/minima
are distinct contracts. Neither bit count nor a shared folder permits conversion.

## TODO

- [ ] Extend independent source fixtures for each variant's scale/sign/offset
  edge cases; retain existing nonfinite-scale and exact-length refusals.
- [ ] Add any further variant only with its own pinned byte contract and
  malformed/truncated/overflow tests; keep other encodings unsupported.
- [ ] Qualify each consuming operator separately from scalar decoding.

Shared metadata authority: [QuantKind](../../../../zerogravity-core/src/quant.rs).
Execution/qualification: [format matrix](../../../../../.agents/plans/pipeline-composition/FORMATS.md).
