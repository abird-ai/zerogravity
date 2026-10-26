# Qwen 3.5 family

- `config`: validated family geometry and layer schedule.
- `source`: interpret authenticated safetensors/GGUF bytes, roles and ties.
- `source::binding`: source-issued finite calls and prepared ForwardPlan bindings.
- `forward`, `hybrid_decoder`: mathematical traversal and state semantics.
- `plan`: family source-operation recording and finite eligibility.
- `reference`: CPU source arithmetic, state bridge and diagnostic traces.
- `bulk_f32_slice`: model-owned GGUF recipes over shared F32 operators.
- `tokenize`: tokenizer/input framing.

The 0.8B, 2B and 27B variants share this ownership and qualification process.
Their actual source, numerical contracts, supported shapes and backend status
remain distinct. Exact-source recipes stay explicit rather than widening a
finite implementation based on matching dimensions.

## Correctness and extension

Use the shared [correctness policy](../../.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md)
and [comparison protocol](../../tools/model-correctness/README.md). Keep four
separate results: exact architecture/source/state/ownership, declared operator
accuracy, independent full-model accuracy, and explicitly supported determinism.
A missing backend or incomplete prefix capture remains unqualified.

A new variant with the same architecture supplies validated geometry and source
roles; it does not duplicate traversal or metrics. A new architecture implements
its own config, source binding, traversal and state inside its model crate. Reuse
the existing mathematical backend, ForwardPlan, physical ResourceArbiter and
StateRoot publication. Register the family once in application wiring. See
[structure and naming](../../.agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md)
for the small module template and dependency rules.

Run source/contract tests without model access first. For real qualification,
collect independent pinned reference and candidate outputs with the same actual
checkpoint, tokenizer/template, precision and input/history. Record prefill/decode,
prompt/input/cached token counts and request batch size explicitly. Use the shared
checker; retain failures and physical cleanup independently of accuracy. Feature
selection and metadata parsing do not establish inference support.

Use the shared [format implementation map](../../.agents/plans/pipeline-composition/FORMATS.md)
when adding a checkpoint codec or precision. Bind this family's tensor roles to
an exact format; reuse the artifact codec and backend operators. A reserved
format home or scalar decoder does not authorize model execution.
