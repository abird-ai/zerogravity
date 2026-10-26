# Q6 source formats

| Exact variant | Logical elements/block | Bytes/block |
|---|---:|---:|
| `Q6_K` | 256 | 210 |

Existing private scalar bodies: [q6.rs](../q6.rs).
Common byte-length/allocation checks: [quant.rs](../../quant.rs).

Q6_K preserves its packed low/high bits, signed local scales and global scale.
No other six-bit variant is implied.

## TODO

- [ ] Extend independent source fixtures for each variant's scale/sign/offset
  edge cases; retain existing nonfinite-scale and exact-length refusals.
- [ ] Add any further variant only with its own pinned byte contract and
  malformed/truncated/overflow tests; keep other encodings unsupported.
- [ ] Qualify each consuming operator separately from scalar decoding.

Shared metadata authority: [QuantKind](../../../../zerogravity-core/src/quant.rs).
Execution/qualification: [format matrix](../../../../../.agents/plans/pipeline-composition/FORMATS.md).
