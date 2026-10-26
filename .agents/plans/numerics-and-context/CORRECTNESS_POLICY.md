# Correctness policy — all models and backends

Owner: @pvl. Coordinator: ROOT / GPT-6.1 Sol xhigh.
User decision26Oct2026: use the correctness approach of both vLLM and llama.cpp
throughout all models. .8B prompts of47–50 tokens precede genuine long-context work;
CPU/CUDA/ROCm/Vulkan results remain separately source/device qualified.

This applies to every existing/future target in `zerogravity-core::targets`,
including metadata-only entries. It changes no target's execution status,
payload authorization or existing accepted numerical profile. Root README,
source/model/SDK identity and driver/recovery holds remain unchanged.

## Required independent checks

1. **Semantics and safety:** released architecture/source codec, complete
   parameters, exact shapes/indexing/masks/positions, causal state/history,
   source ownership, one ResourceArbiter/ForwardPlan/StateRoot publication,
   cancellation/completion/quarantine and final physical ownership. These are
   hard invariants; a quality score cannot waive them.
2. **Operator accuracy:** compare with an independent original/mathematical
   reference under a declared op/dtype/quant/reduction/layout profile. Existing
   source-bit contracts remain exact. For a separately declared tolerance
   profile, use per-op error measures/limits established before measurement.
   No universal epsilon or inherited permit from another backend/model.
3. **Model accuracy:** same checkpoint/quant artifact, tokenizer/template,
   inputs and context/generation settings against a pinned independent runtime.
   Compare outputs and distributions/quality under the task-specific contract.
   Different batch reductions can pass accuracy without byte equality.
4. **Determinism:** same-plan replay, batch invariance, schedule invariance and
   cross-hardware bit equality are distinct claims. Require them only where
   explicitly supported/advertised, with their own tests. They never follow
   from a seed, matching text, enum name or a passing operator fixture.

All four appear in reports, including failed/incomplete/unsupported scopes.
Interrupted originals never become passing evidence. Every physical run retains
its exact source/binary/model/driver/SDK identities and original owner's result.
No second engine or independent publisher is introduced for qualification.

## Upstream practice used

- Pinned vLLM `db9527a46873454610df6dbedf79a36d6bf1a7f6` tests exact token/text
  equality or mutual membership in returned top-N maps at the first differing
  generated token. Its default helper does not assert numeric allclose or
  compare subsequent different histories. Default batching need not be
  invariant; opt-in invariance is a separate runtime/hardware contract.
- Pinned llama.cpp `d81235049384534c167caea52b85a694f6103d14` compares backend
  operators with CPU reference output and per-op NMSE limits. Its NMSE is
  `sum((candidate-reference)^2) / sum(candidate^2)`. Default1e-7, matmul/attention
  5e-4 and softmax1e-6 in that source are distinct operator contracts, not
  blanket model thresholds. Our undefined zero-energy/nonfinite handling is
  deliberately stricter: the pinned upstream comparison can let NaN through.
- llama.cpp model-quality work measures aligned target NLL/PPL, KL divergence,
  probability differences and top-token agreement against retained logits.
  Neither PPL nor top1 alone proves full distributions/state. PPL comparisons
  require the same tokenizer/corpus/settings; limits are task/profile specific.
- Our common helpers live at `tools/model-correctness/`. They strengthen input
  completeness/type/nonfinite checks rather than copy zipped/truncated cases.
  Measured KL/NLL/NMSE do not silently acquire acceptance thresholds.

Sources: [vLLM tests](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/tests/models/utils.py),
[vLLM reproducibility](https://docs.vllm.ai/en/v0.31.0/usage/reproducibility/),
[llama.cpp operators](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tests/test-backend-ops.cpp),
[llama.cpp PPL/KL](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/perplexity/README.md).

## Application across the registry

| Model/task | Independent model-quality evidence |
|---|---|
| Qwen/Gemma and other language models | Pinned original HF/vLLM or same-GGUF llama.cpp; matching source/precision/context; generated IDs/text and aligned full-vocabulary/corpus metrics |
| Quantized variants | Same-pack operator checks plus separately measured source-vs-derived quality; lossless repack is distinct from lossy quantization |
| Vision/image generation | Source preprocessing/geometry and model-specific intermediate/output/quality corpus; a text-only pass cannot qualify image execution |
| ASR/TTS/audio | Exact source/frontend/alignment/state and task-specific transcript/audio quality; token metrics alone do not qualify acoustic output |
| Decision/embedding/other tasks | Released task semantics and independently declared score/vector/decision metrics; no generated-text substitute |

Unsupported/missing adapters remain unqualified. A registry slug, source metadata,
synthetic GPU test or another model's acceptance cannot fill a missing cell.
CPU/GGUF, HF/CUDA, HF/HIP and HF/Vulkan remain distinct. No quantized checkpoint
may silently substitute for the requested original BF16/F32 model.

## .8B first milestone and long context

- Preserve the historical strict single-request/batched decode FAIL101 and
  deliberate quarantine. It asks byte equality at request batch sizes1/4.
  Independently authenticated original vLLM N1/B4 already
  differ on identical frontier47 operands: the same six QKVZ BF16 differences.
  Metadata also establishes width differences at48/49; no paired original N1
  frontier50 exists in that fifteen-transition archive. Do not fabricate one.
- Default-profile acceptance compares each implementation with its appropriate
  original numerical profile and actual model outputs. Preserve complete
  same-width operator/state proofs and exact structural safety. A batch-invariant
  kernel would be a new explicitly qualified profile, not a fix to relabel the
  default original's result. The failed historical result never turns green.
- Initial prompt lengths47/48/49/50 differ from a47-token prompt followed by
  cached decode at48/49/50. Declare and test both requested paths; the current
  finite HF route's unsupported fresh48–50-token prompts are not covered by
  cached evidence.
- Genuine full-model contexts512/1024/2048/4096/8192 and then16384/32768 need
  actual independently run prefixes/continuations within admitted capacity,
  plus state/replay/chunk/fault coverage. Large RoPE positions or empty padding
  cannot qualify them. Keep current accepted ranges and missing cells explicit.
- Current HF .8B route is finite CUDA; corresponding CPU/ROCm/Vulkan full-model
  paths remain to be implemented/qualified. Existing synthetic operator,
  different GGUF and finite Qwen27B proofs are separate evidence.

Future model/backend changes use this policy and shared metrics alongside their
existing original-source and ownership checks. Complete the focused numerical,
source/state/physical/serving checks and assembled `scripts/check.sh` before
accepting an implementation; performance work follows correctness.

<!-- Previous labels: case5, initial48–50. Use explicit phase/token/batch fields. -->
