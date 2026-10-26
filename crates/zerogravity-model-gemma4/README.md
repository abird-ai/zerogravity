# Gemma 4 family

- `config`, `tensors`: architecture, tensor roles and variant geometry.
- `text`: family traversal, PLE/shared-KV/MoE semantics and source binding.
- `plan`: model-owned finite BF16 policy over shared semantic operators.
- `execution`: source-owned CPU candidate and optional backend composition.
- `text` tokenizer adapter: tokenizer and input framing where enabled.

Use the same extension and correctness process as Qwen. Gemma-specific PLE,
shared-KV, norm and vocabulary rules remain here; they do not become generic
backend assumptions. Analytic fixtures, source inspection and finite candidates
have separate acceptance scopes.

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
for exact codec, precision and backend ownership. Gemma's tensor roles and
numerical eligibility stay here; shared codecs and operators are reused. Missing
format/backend implementations remain explicit TODOs and refuse execution.
