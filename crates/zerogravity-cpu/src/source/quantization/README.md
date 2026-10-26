# CPU source quantization

CPU format homes document operators over authenticated immutable source bytes.
Current arithmetic remains in [source.rs](../../source.rs); this layout
does not duplicate or move its shared activation preparation and dot schedules.

- [q1](q1/README.md)
- [q2](q2/README.md)
- [q3](q3/README.md)
- [q4](q4/README.md)
- [q6](q6/README.md)
- [q8](q8/README.md)
- [nvfp4](nvfp4/README.md)

BF16 has a separate [precision home](../../precision/bf16/README.md).
[The format matrix](../../../../../.agents/plans/pipeline-composition/FORMATS.md) separates codec, operator, device
and full-model support. Preserve source identity/backing, exact numerical
profiles, finite-input/output checks and admission before any future extraction.
