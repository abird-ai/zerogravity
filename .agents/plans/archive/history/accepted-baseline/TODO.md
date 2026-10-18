# Accepted implementation baseline

Provenance commit: `6d74f76`.
Owner-approved completion is recorded below; original reports are available in Git before their retirement.

## Uncommitted

None. Remaining features are separate plans in [the index](../TODO.md).

## Committed

### 6d74f76 — owner-approved tested baseline

- [x] Original Qwen3.5 CPU0.8B/2B functional Phase1:16/16 through genuine8K.
- [x] Complete original-model resident GPU pipeline and source/state/resource/cancellation boundaries through the qualified512-token scope.
- [x] CUDA local RTX3060/a1 RTX4060 and HIP a1 gfx1150/x2 gfx1151; SDK-free NVIDIA Vulkan is the fifth backend profile.135 profile checks plus12 host fingerprints accepted; this is not five fully qualified physical cards.
- [x] Nine-package layout, reference relocation and superseded worktree/cache cleanup; root README unchanged.
- [x] Qwen3.8/Gemma4 source contracts; no trained new-model inference.
- [x] Contention-first concurrency review completed; no left-right migration justified. Require measured contention and ownership fit for future proposals.
- [x] Completed historical evidence retired by owner authority. Preserve tests and technical designs; Git retains previous source/history.

Independent CPU8K GDN drift, larger contexts, held local AMD hardware, actual new
models, acquisition, serving and further optimization remain unchecked in their
own plans. This completed baseline does not close those tasks.

### Accepted detail retained from retired reports — 6d74f76

- [x] Original-model resident execution includes compressed embeddings, device activation quantization, GDN/attention/FFN and all42 private state domains; one mathematical schedule and compact token feedback.
- [x] State-only prefill, exact whole-range Contiguous aliases, prepared range/binding metadata and fused dense View certification are delivered. This is not driver graph capture, GPU tokenization or a general device scheduler.
- [x] Portable CPU/device v1 replay is qualified within its trusted diagnostic scope; authenticated persistence and full tokenizer/sampler/grammar/media serialization remain open.
- [x] Independent native-source512 checks compare146 stages/model: zero nonzero-FP32 differences and56 existing2B signed-zero distinctions. Internal full-logit/all-state/trace comparisons remain strictly bit-exact; independent full-vocabulary source equality is not inferred.
- [x] Main model-merge regression:135 ordinary Rust executions; separate23 original-model executions/22 distinct names, two exact eight-token CLI outputs and one official Qwen metadata case. Earlier phase counts are not additive accepted totals.
- [x] Native CPP profiles retain their measured scalar/libm/loader scope, including a1 gfx1150's explicit glibc2.44/GCC16.2 closure. Separate LLVM device code generation remains unqualified.

Approved original model content identities (still required for actual model inputs):

- Qwen3.5 0.8B SHA256: `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`.
- Qwen3.5 2B SHA256: `91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6`.
