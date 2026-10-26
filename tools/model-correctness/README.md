# Shared correctness comparisons

Standalone qualification tools; no inference, model loading or downloads.
Applies to every language-model family, source format and backend through
independently collected outputs. Model-specific adapters retain their original
source, state, physical-owner and numerical-profile checks.

- `accuracy.py`: exact generation, pinned vLLM default generation comparison,
  full-vocabulary distribution measurements and llama.cpp backend NMSE.
- `check.py`: checks compatible declared identities, complete ordered cases,
  input/history lengths, context bounds and owner closure before comparison.
- Tests use synthetic data only; they cannot qualify a model or backend.

## Policies

`exact` requires identical generated IDs and output text (or explicitly supplied
text SHA256). `vllm-default` follows the pinned helper: compare generated IDs,
then require mutual membership in the returned top-five-plus-selected maps at
the first differing token. Later tokens have different histories and are not
semantically compared. Equal IDs with different text produce a warning in this
mode. Structural checks still cover every supplied row. Missing maps at a
divergence fail. We additionally reject truncated populations and malformed,
nonfinite or impossible map data rather than inheriting upstream zip truncation.

Full-vocabulary measurements report reference→candidate KL, target NLL,
delta NLL, probability RMS, logit error and top1 agreement. `vocabulary_size`
must match the complete aligned logits. Top-k packets cannot establish KL/PPL.
Average aligned target NLL to obtain corpus log-perplexity; a single position
is not corpus perplexity. Raw floating-point KL roundoff is retained.

`backend_nmse(reference, candidate)` uses llama.cpp's candidate-energy
denominator. Both-zero vectors return0; nonzero-reference/zero-candidate refuses.
There is no global threshold: llama.cpp itself overrides thresholds by operator.
Existing exact codec/index/mask/state and source-specific arithmetic checks stay
exact. Distribution metrics are measurements until a model/profile-specific
acceptance contract supplies its limits.

## Input and command

```sh
python3 -B tools/model-correctness/check.py \
  --reference original.json --candidate zerogravity.json \
  --policy vllm-default --report fresh-comparison.json
```

Both inputs use `schema: "zg.model-output.v2"` and contain:

- `complete: true`, `owner_closure: {"complete": true, "errors": []}`.
- `text_representation`: `utf8` for actual text, or `sha256` for actual stored
  output digests. Digest comparison never reconstructs or claims retained text.
- `model`: `source_sha256`, `tokenizer_sha256`, `template_sha256`, `precision`,
  `quantization`. All five values must match. Digests are64 lowercase hex digits.
- `execution`: actual `backend` (`cpu/cuda/hip/vulkan`), `runtime_revision`,
  `numerical_profile`, `runtime_artifact_kind` (`executable/container-config`),
  `runtime_artifact_sha256`, `context_capacity_tokens`, `vocabulary_size`,
  `batch_invariant`. Runtime artifacts/backends/profiles are reported separately;
  model identities, context capacity, vocabulary and invariance mode must match.
- Nonempty ordered `cases`, each with unique descriptive `id`, actual
  `input_ids`, `phase`, `prompt_tokens`, `input_tokens`, `cached_tokens`,
  `request_batch_size` and `generation`. Input+cached counts equal input length;
  input+requested output must fit context. `prefill` completes the prompt
  (possibly its last chunk); `decode` consumes one token after the prompt.
  A fresh 48-token prompt differs from a 47-token prompt followed by decode.
  Reference and candidate batch sizes are reported separately; comparing model
  accuracy across widths does not require or establish byte invariance.
- `generation`: nonempty `token_ids`, `text` (actual text or declared SHA256),
  and explicit `top_logprobs` (`null`, or one list of `{token_id,logprob}` per
  output). Every selected ID is in its own returned map; each map has5 or6
  entries. IDs must fit the declared vocabulary; scores are finite/nonpositive.
- Optional paired `full_logits` and `target_id` in a case refer to one identical
  teacher-forced history, independent of later natural-feedback divergence.

Reports are created exclusively; an existing file is never overwritten. Exit0
means the selected output-comparison policy passed; exit4 means refusal/failure.
Reported metadata is not independently authenticated by this tool. Reports
explicitly grant no source, operator/state/physical or inference qualification.
Missing contexts/backends and interrupted originals remain unqualified.

Every model uses this same protocol. Model adapters collect their own source,
tokenizer, state and owner facts; they do not duplicate comparison algorithms.
Keep family-specific capture tools beside their reference contracts and pass
their complete outputs to this shared checker. A partial layer/prefix capture
cannot be exported as a completed model generation. `hip` identifies the actual
ROCm API backend; driver/SDK/source identities remain independently recorded.

<!-- Previous protocol/labels: zg.model-output.v1, new_input_tokens,
context_capacity, rocm, initial48, cached48. No input aliases retained. -->

## Upstream basis

- [vLLM exact/default helpers, db9527a46](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/tests/models/utils.py).
- [vLLM reproducibility](https://docs.vllm.ai/en/v0.31.0/usage/reproducibility/):
  default batching can change results; optional invariance is hardware/version
  bounded. Enabling it changes the contract and requires fresh qualification.
- [llama.cpp backend tests, d812350](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tests/test-backend-ops.cpp).
- [llama.cpp model PPL/KL comparisons](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/perplexity/README.md).

The repository policy lives in
[CORRECTNESS_POLICY.md](../../.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md).
