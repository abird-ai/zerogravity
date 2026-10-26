# NVFP4 source format

Reserved documentation home. No NVFP4 source codec, metadata tag or packing
contract is implemented. The family name specifies no container layout.

## TODO

- [ ] Select a released, pinned exact variant and authoritative byte layout.
- [ ] Specify packing/order, signedness or codebook, scale/offset encoding,
  block geometry, special values, endianness and checked byte extents.
- [ ] Add only that variant's metadata recognition and bounded scalar codec,
  with independent byte fixtures and malformed/truncated/overflow cases.
- [ ] Keep unknown variants `Unsupported`; codec availability alone grants
  no CPU/GPU operator or complete model support.

NVFP4 must retain its actual floating code/scale contract; do not substitute
an integer Q4 codec or invent a scale hierarchy from the family name.

Shared metadata authority: [QuantKind](../../../../zerogravity-core/src/quant.rs).
Execution/qualification: [format matrix](../../../../../.agents/plans/pipeline-composition/FORMATS.md).
