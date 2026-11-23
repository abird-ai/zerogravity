# ZeroGravity full comprehensive handoff — 23 Nov 2026

This is the authoritative takeover pack for
`/home/pvl/spaces/abird/src/zerogravity` as of 23 Nov 2026.

It is intentionally a **comprehensive current takeover delta**, not another
byte-for-byte copy of the already very large 22 Nov pack.

Historical authority is chained as follows:

1. this document — everything current on 23 Nov, including the post-22-Nov
   implementation/design/benchmark work and active dirty ownership;
2. [FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md)
   — complete 22 Nov authority and embedded prior lineage;
3. the 21/20/18/15/14 Nov packs transitively embedded/referenced there;
4. live source, tests and owning TODOs override any stale dated prose.

The next agent should read the short ordered list in §1, then use this document
as its takeover map.

---

## 0. Executive state

### Source authority

- **L5 / `dotlink---l5` is the only source-authoring connector.**
- Repository: `/home/pvl/spaces/abird/src/zerogravity`.
- Persistent branch/worktree: **main only**.
- Do not create another persistent worktree or branch for normal work.
- X2 is execution/testing only for large-VRAM AMD/ROCm work. Synchronize exact
  L5 commits by Git bundle; do not push merely for transport.
- A1 has now been explicitly brought back into scope for mixed-GPU build/runtime
  validation. It is **not** an independent source-authoring host. Use exact L5
  commits there.
- All hosts use cache `/home/pvl/.cache/abird/zerogravity`.
- Cargo:
  `CARGO_BUILD_JOBS=8`,
  `RUSTC_WRAPPER=/run/current-system/sw/bin/kache`.
- Physical GPU work uses
  `python3 -B scripts/native-gpu.py -- ...` when the owning qualification
  requires the native-GPU envelope.
- TCP/UDP loopback work uses the accepted private-network envelope and
  `--require-private-network`; never weaken sandbox/network/TLS/SSH/ownership
  checks merely to make a gate pass.

### Git authority at handoff creation

Latest signed **source** checkpoint before this handoff document is:

`f3f872cb567a6885918ae2b4866ee6d535d736d8`
— `feat(cuda): retain B1 fused head greedy primitives`.

Tracked `origin/main` is:

`2a595be84bc7c5dccf7509e35e4f9b3a341662a2`
— `docs: record matched rocm vllm small models`.

Local main is therefore 25 source/docs commits ahead of origin before the
handoff-document commit.

The handoff-consolidation docs commit may sit above `f3f872c`; revalidate
actual HEAD first.

### Critical concurrent dirty work — do not overwrite

At handoff creation, another agent owns an uncommitted CUDA FlashInfer + paired
GDN-state slice. The following paths are dirty/untracked and must be preserved
exactly unless ownership is explicitly transferred:

- `crates/zerogravity-cli/Cargo.toml`
- `crates/zerogravity-gpu/Cargo.toml`
- `crates/zerogravity-gpu/src/attention_provider.rs`
- `crates/zerogravity-gpu/src/cuda/gdn/state_transport.rs`
- `crates/zerogravity-gpu/src/resident_program/cuda_forward/attention.rs`
- `crates/zerogravity-gpu/src/resident_program/cuda_forward/cold.rs`
- `crates/zerogravity-gpu/src/resident_program/cuda_forward/execute.rs`
- `crates/zerogravity-gpu/vendor/cubecl-cuda/Cargo.toml`
- `crates/zerogravity-gpu/vendor/cubecl-cuda/build.rs`
- `crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/server.rs`
- `crates/zerogravity-gpu/vendor/cubecl-cuda/src/lib.rs`
- `crates/zerogravity-gpu/vendor/cubecl-cuda/src/owned_command.rs`
- `crates/zerogravity/Cargo.toml`
- `nix/dev.nix`
- untracked `crates/zerogravity-gpu/src/cuda/gdn/resident/state_pair_in.cu`
- untracked
  `crates/zerogravity-gpu/src/cuda/gdn/resident/state_pair_out_certified.cu`
- untracked
  `crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/server/flashinfer.rs`
- untracked
  `crates/zerogravity-gpu/vendor/cubecl-cuda/src/owned_command/flashinfer.rs`
- untracked
  `crates/zerogravity-gpu/vendor/cubecl-cuda/src/owned_command/flashinfer/bridge.cu`

The visible intent is:

- add `AttentionProvider::FlashInferDecode`;
- combine FA2 prefill + FlashInfer decode in the prepared attention plan;
- add paired GDN canonical/native import/export primitives
  `GdnStateIn` and `GdnStateOutCertified`;
- preserve exact service/extent/alias/status ownership.

Do not reset, unstage, overwrite or partially commit this concurrent work.

---

## 1. Read first, in order

1. `AGENTS.md`
2. `.agents/plans/README.md`
3. this file:
   `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md`
4. `.agents/docs/CURRENT_HANDOFF.md`
5. `.agents/docs/CURRENT_BASELINE.md`
6. `.agents/plans/TODO.md`
7. `.agents/plans/model-execution/TODO.md`
8. `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`
9. `.agents/plans/serving-and-speculation/TODO.md`
10. `.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md`
11. `.agents/plans/memory-and-streaming/TODO.md`
12. `.agents/plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md`
13. `.agents/plans/pipeline-composition/TODO.md`
14. the three next-iteration specs:
    - `.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md`
    - `.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md`
    - `.agents/plans/cpu-gpu-optimization/NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md`
15. any owning docs linked by the current slice.

For history before 23 Nov, read the 22 Nov comprehensive pack rather than
reconstructing old conversations.

---

## 2. Revalidate before editing

Always inspect:

```text
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git log -1 --show-signature
git worktree list
```

Also inspect:

- live Cargo/rustc/model/native-gpu/vLLM/nvcc/ptxas/Bubblewrap processes;
- physical GPU state;
- source/cache identity;
- disk space and temporary `_tmp` / benchmark artifacts;
- current dirty ownership before touching shared CUDA/GDN/provider files.

Do not assume the handoff source HEAD is still current.

---

## 3. Non-negotiable architecture

Preserve all of the following.

### State and lifetime

- `StateRoot` publication remains authoritative.
- Cancellation never manufactures completion.
- Unknown submitted GPU work remains fenced/quarantined.
- `ResourceArbiter` and physical lifetime ownership remain authoritative.
- `DeviceBufferIdentity` is canonical physical allocation/range identity.
- Handle ranges remain executable binding geometry, not ownership truth.
- Exact partial-tail paged COW is already solved. Do not build another COW engine.
- Prefix reuse requires exact compatibility and complete-domain ownership.
- No graph/vendor kernel/persistent batch may become a second state authority.

### Scheduler and batching

- One shared scheduler/state engine.
- Request/session ownership remains request-oriented.
- Physical arithmetic may be token/batch oriented.
- Physical serving cohorts remain B1/B2/B4/B8.
- Offered concurrency 16/32/64 is queue/refill/regroup pressure, not a reason
  to invent B16/B32/B64 model kernels.

### Rust/performance style

- Model-neutral primitives stay clean and composable.
- Prefer zero-cost Rust abstractions where natural.
- Resolve backend/model/provider choices at load/prepare time where possible.
- Optimize measured hot paths aggressively when the fast design remains simple.
- Otherwise prefer the simpler maintainable design.
- Do not build elaborate evidence/fixture systems for every experiment.
- Reuse accepted evidence for unchanged primitives.
- Real-model feedback comes before speculative fixture matrices.

### Durable rejected-experiment rule

This is now a hard project rule:

> **Reject the selector/consumer, not a correct reusable primitive.**

A negative A/B for one model/shape/backend/device removes or disables the
production consumer/selector. It does **not** justify deleting a clean reusable:

- kernel;
- transport;
- layout;
- scheduling transform;
- provider;
- ownership/completion primitive;
- fusion/composition primitive.

Delete the primitive only when:

- it is incorrect or unsafe;
- it violates numerical/ownership/resource contracts;
- it is strictly superseded by an equivalent retained primitive;
- it is intrinsically one-off/model-specific maintenance debt with no useful
  composable domain.

Temporary benchmark drivers, selectors, evidence glue and model-specific
adapters should still be removed normally.

This rule already exists in `AGENTS.md` and `CURRENT_HANDOFF.md`; preserve it.

---

## 4. Host state

### L5

Primary source host and current CUDA benchmark host.

Current CUDA device used by the matched baseline:

- NVIDIA RTX 3060 Laptop, 6 GiB;
- SM86;
- current Qwen native CUDA source path is qualified here.

### X2

Execution-only large-VRAM AMD host:

- Radeon 8060S;
- gfx1151;
- ROCm;
- ~96 GiB-class addressable pool in current tests.

Use X2 for:

- HIP/ROCm physical qualification;
- 27B resident work;
- matched ROCm vLLM baselines.

Exact L5 source commits go over by Git bundle.

### A1 — explicitly re-enabled for mixed-GPU runtime/build validation

Observed hardware:

- NVIDIA RTX 4060 Laptop GPU, 8 GiB, SM89;
- AMD Radeon 880M/890M-class iGPU, gfx1150, integrated/UMA.

Authoritative HIP probe on A1 reported approximately:

- `integrated=true`;
- HIP total pool: 16,355,614,720 bytes;
- HIP free pool: 16,342,921,216 bytes.

This is why DRM's nominal 512 MiB "VRAM" must **not** be used as HIP admission
authority on integrated AMD. Production HIP now observes the driver allocator
pool instead.

The current 0.8B/2B HIP source envelope is roughly 13.07 GB before normal
headroom, so both should fit A1's HIP pool under the current default 10% headroom
when the exact source checkpoint is synchronized.

CUDA caveat:

- the new FA2 provider itself compiles for SM89;
- the full current Qwen CUDA chain still contains SM86-closed native/PTX
  contracts;
- therefore A1 RTX 4060 full-model CUDA is **not yet qualified** merely because
  `cuda + model-qwen35` compiles.

Do not weaken architecture guards to pretend SM89 support. Generalize/qualify
the remaining closed leaves properly.

---

## 5. Build and feature architecture — Priority-0 work completed on 23 Nov

The desired product remains **one `zerogravity` binary / one process**.

The mixed GPU Nix shell is a build/development environment, not a runtime
sidecar architecture.

### Feature algebra

Backends are independent system capabilities:

```text
cpu
cuda
hip
vulkan
```

Models are independent additive contributions:

```text
model-qwen35
model-gemma4
...
```

Intersection semantics:

```text
cuda + model-qwen35  -> Qwen CUDA contribution
hip  + model-qwen35  -> Qwen HIP contribution
cuda alone           -> CUDA substrate, no Qwen model
model-qwen35 alone   -> Qwen model/task/source support, no GPU backend
```

Cargo weak optional-dependency features implement this relationship.

Old pair names such as `qwen35-cuda` / `qwen35-hip` may remain compatibility
aliases; they are not the architecture.

`serve` is also orthogonal. It means the serving frontend is compiled; it does
not secretly pull CPU/Qwen.

### Production `all`

`all` means normal production capabilities:

- normal model families;
- CPU/CUDA/HIP/Vulkan;
- cache/acquisition;
- serving/frontends.

It excludes qualification captures, abort probes, fault injection and diagnostic
evidence features.

Canonical build:

```bash
nix develop .#gpu

cargo build --locked --release \
  -p zerogravity-cli \
  --no-default-features \
  --features all
```

### Mixed GPU shell

`nix develop .#gpu` is validated on L5 and provides the required combined
CUDA + ROCm/HIP development/runtime user-space:

- CUDA runtime headers;
- CCCL;
- NVRTC;
- cuBLAS;
- nvcc;
- HIP/ROCm;
- hipcc;
- hipBLAS;
- hipBLASLt;
- rocBLAS;
- runtime library paths;
- Nix-specific CUDA include view.

CUDA unfree policy is isolated to the optional GPU shell; ordinary CPU package
evaluation remains unchanged.

### Runtime packaging

The product still runs as one ZeroGravity binary. The host supplies GPU driver /
minimal user-space runtime/provider libraries required by selected backends.

Do not introduce:

- Python runtime;
- PyTorch runtime;
- vLLM process;
- Triton daemon;
- CUDA/HIP sidecars.

AOT/native providers may be statically compiled into the ZeroGravity binary.

### Public one-shot GPU path

The natural CLI now routes through the same source-bound production Runtime seam
used by SDK/serving:

```text
ModelRef
  -> authenticated immutable cache snapshot
  -> Runtime::models().load(...)
  -> Qwen CUDA/HIP production adapter
  -> Runtime::request_text(...)
```

Canonical smoke commands:

```bash
zerogravity run-cuda qwen3.5-0.8b --prompt hello
zerogravity run-rocm qwen3.5-0.8b --prompt hello
zerogravity run      qwen3.5-0.8b --prompt hello
```

Semantics:

- `run-cuda`: CUDA or fail;
- `run-rocm`: HIP/ROCm or fail;
- `run`: qualified automatic selection only.

Friendly model names resolve to the model contribution's canonical immutable
source digest; users do not need internal revision-qualified cache alias names.

### Accepted implementation checkpoint

`491cf99` — `feat: unify production GPU build and run paths`.

Validated:

- model-only;
- backend-only;
- model+backend;
- production `all`;
- serve-only isolation;
- mixed GPU shell;
- real public CUDA Qwen 0.8B inference;
- truthful HIP admission;
- default/all target compile.

---

## 6. Attention architecture and provider work

### Semantic cleanup

Model semantics no longer name a physical algorithm as the operator.

The model plan now describes semantic `Attention`.

Physical provider selection is prepared separately.

`6f8f071` — `refactor(gpu): separate attention semantics from providers`.

The shared provider plan is phase-aware:

- prefill provider;
- decode provider.

Provider choice is load/preparation-time, not per-token dynamic framework logic.

### Terminology

Keep these meanings distinct:

- **native**: ZeroGravity directly owns/launches the GPU work;
- **builtin**: kernel authored/owned by ZeroGravity;
- **FlashAttention**: FlashAttention algorithm/kernel provider;
- **FlashInfer**: FlashInfer provider;
- **AITER**: AMD provider;
- **Triton-AOT**: Triton-origin compiled provider.

These can compose:

```text
native + builtin
native + FlashAttention
native + FlashInfer
native + AITER
native + Triton-AOT
```

Using an external algorithm does not create a parallel scheduler/runtime.

### Native FA2 provider

`64c06e2` — `feat(cuda): add opt-in FlashAttention provider`.

Implementation:

- pinned `vllm-project/flash-attention`
  commit `b273a02621d3c94bc747df67204f591b67f38fae`;
- pinned CUTLASS
  `62750a2b75c802660e4894434dc55e839f322277`;
- forward-only BF16/head-256 causal FA2;
- AOT/native build;
- SM86 + SM89 + compute_80 PTX;
- no Python/Torch/libtorch runtime;
- raw pointers and CUDA stream stay inside `cubecl-cuda` owned-command actor;
- existing resident leases / ExecuteScope / completion ownership remain
  authoritative;
- ZeroGravity certification remains after FA2 so error semantics are unchanged.

The provider remains opt-in via `cuda-flash-attn`; normal production `all`
does not silently promote it.

### vLLM provider behavior

For the exact L5 Qwen shape (BF16, head size 256, ordinary causal attention),
vLLM 0.33 validates the major CUDA providers and prioritizes roughly:

1. FlashAttention;
2. FlashInfer;
3. Triton Attention;
4. later fallbacks.

The vLLM bundled FlashAttention extension itself is not a valid ZeroGravity
runtime dependency: it drags Torch/CUDA libraries and is hundreds of MB. The
ZeroGravity provider instead compiles the underlying forward kernel directly.

### FA2 A/B decision

FA2 was tested against the builtin provider on real Qwen 0.8B.

Observed:

- 256/768-ish prefill workloads: essentially neutral (~0–0.4% difference);
- selecting FA2 for decode made the 64-token generation workload about 5–6%
  slower.

Therefore:

`ed0382a` — `perf(cuda): keep FlashAttention off decode path`.

Current committed policy when FA2 feature is enabled:

- FA2 may own prefill;
- builtin stays on decode.

This is exactly the durable rejected-selector rule:
the FA2 primitive remains available even though decode selection lost.

The current dirty concurrent slice is exploring FlashInfer specifically for
decode. Do not collide with it.

---

## 7. CUDA prefill and decode optimization after FA2

### Fused Q/K normalization + rotary prefill

`0a35f17` — `perf(cuda): fuse qk normalization and rotary prefill`.

`2805c37` — `perf(cuda): enable fused qk rotary prefill`.

These are committed current-path prefill optimizations.

### Why GDN is now the dominant focus

Qwen3.5 0.8B config contains:

- 24 total layers;
- 18 linear-attention/GDN layers;
- 6 full-attention layers.

This is why attention-provider replacement alone cannot close the full-model
vLLM gap.

The GDN path has much larger repeated state/transport/launch cost.

### Fused recurrent export + certification

`179bb15` — `perf(cuda): fuse GDN recurrent export certification`.

Production B1 decode now reuses the existing `RecurrentOutCertified` primitive
instead of:

```text
recurrent computation
 -> separate BF16 certification
 -> separate recurrent canonical export
```

It now does export + gated-output certification in one launch.

This removes one launch per GDN layer, up to **18 launches per decoded token**.

Real Qwen 0.8B 8-token-prompt + 64-token-generation workload:

- before: ~906.4 ms;
- after: ~855.3 ms;
- improvement: ~5.6%;
- greedy output remained exact.

This is a strong signal that GDN launch/state mechanics are a better current
optimization target than more full-attention abstraction.

### Retained losing primitives

`c829aa9` — `feat(cuda): retain B1 direct GDN conv primitive`.

The Qwen production selector/consumer lost its A/B and was removed, but the
clean B1-capable cached-GDN convolution primitive remains for future
models/shapes/backends.

`f3f872c` — `feat(cuda): retain B1 fused head greedy primitives`.

The Qwen selector/receipt wiring was essentially neutral/slightly worse
(~856.9 ms vs ~855 ms), so the selector was rejected.

Retained dormant reusable primitives include:

- single-row `LaneStrided<16,512>` projection->greedy kernel;
- generic partial-record continuation reducer.

Do not delete them merely because current Qwen did not benefit.

### Broader reusable performance library already retained

Continue preserving:

- lane regroup;
- copy + inactive-tail zero;
- 17 BF16 GEMM strategies;
- recurrent tiled transpose;
- GDN occupancy variants;
- grouped recurrent/Conv transport;
- packed BF16 greedy;
- head->greedy partials;
- dual Q/K RMS;
- certified recurrent export;
- shared-input projection composition;
- ragged attention/private KV/rotary/stateless primitives;
- CUDA/HIP transfer-retirement primitives;
- native FA2 provider;
- B1 direct cached-GDN conv primitive;
- B1 fused head-greedy primitives.

A negative Qwen A/B is not a deletion order.

---

## 8. Direct matched vLLM result that changed the CUDA optimization strategy

A 23 Nov direct Runtime/vLLM same-snapshot comparison was run on L5 using the
canonical Qwen3.5 0.8B BF16 snapshot.

These are diagnostic full-model warm measurements, not a replacement for the
public serving matrix in `CURRENT_BASELINE.md`.

### 768-token-ish prompt + 1 generated token

- vLLM 0.33: ~67.8 ms;
- ZeroGravity committed FA2-prefill path: ~284.7 ms.

ZG is still about 4.2x slower in that long-prefill cell.

### 8-token prompt + 64 generated tokens

- vLLM 0.33: ~426.4 ms;
- ZeroGravity after `179bb15`: ~855.3 ms.

ZG is still about 2.0x slower in that generation-heavy cell.

Important conclusion:

> **Attention provider selection is not the main remaining bottleneck.**

Because 18/24 layers are GDN, state transport/certification and recurrent/conv
execution are currently much more valuable targets.

For direct vLLM Python baseline work in the Nix environment:

- `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` avoids Triton's hard-coded
  `/sbin/ldconfig` assumption on NixOS;
- use a real script file rather than stdin because vLLM V1 worker spawning must
  reopen `__main__`;
- `VLLM_USE_FLASHINFER_SAMPLER=0` avoids unrelated FlashInfer sampler JIT when
  measuring the model/attention path in the lean vLLM shell.

Do not turn these benchmark-environment details into production requirements.

---

## 9. ROCm/HIP current baseline and work

The active baseline remains `CURRENT_BASELINE.md`.

### X2 matched 24 GiB envelope

vLLM 0.33 vs ZeroGravity HIP baseline:

| Offered | vLLM 0.8B | ZG HIP 0.8B | vLLM 2B | ZG HIP 2B |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 51.966 | 12.406 | 36.054 | 6.169 |
| 4 | 135.358 | 12.048 | 96.233 | 6.897 |
| 8 | 219.612 | 12.139 | 140.847 | 6.771 |
| 16 | 245.256 | 11.881 | 150.059 | 6.752 |
| 32 | 271.828 | 11.848 | 156.828 | 6.735 |
| 64 | 284.356 | 11.797 | 159.801 | 6.721 |

All retained outputs exact.

### Current HIP decode cohort improvement

Committed HIP sequence above `2a595be` includes:

- `94a7e2d` — batch source decode projections;
- `48d3d94` — size decode cohort head scratch;
- `cb923cc` — price decode cohort projection scratch;
- `189c718` — accept fused cohort observations;
- `99dcbbe` — batch row-wise decode operators;
- `e35e650` — compare decode cohort semantics structurally;
- `9ee877c` — batch aligned decode rotary prep;
- `a5ec4c3` — seal batched decode rotary variants;
- `1de99ef` — batch independent decode conv state;
- `c33016d` — preserve conv coefficient input contract;
- `57cc384` — document stateful HIP conv cohort.

Current retained 0.8B HIP:

- C4 ~13.297 tok/s;
- C8 ~13.351 tok/s;
- exact tokens;
- clean retirement.

The Conv4/stateful cut moved C4 only modestly and C8 nearly flat.

Current interpretation:

- recurrent GDN state work remains the next meaningful HIP batching target;
- HIP singleton efficiency is also independently behind vLLM;
- batching alone does not explain the gap.

### X2 27B matched 75 GiB

| Offered | vLLM | ZG HIP |
| ---: | ---: | ---: |
| 1 | 3.803 | 0.540921 |
| 4 | 9.917 | 0.536684 |
| 8 | 13.491 | 0.534072 |
| 16 | 13.533 | 0.530675 |
| 32 | 13.492 | 0.528366 |
| 64 | 13.461 | 0.529800 |

All retained rows exact/replay-stable/lane-invariant.

Podman/FUSE is obsolete for the current baseline. The vLLM baseline is Nix-native.

---

## 10. Current L5 CUDA baseline

From `CURRENT_BASELINE.md`:

### vLLM 0.33 optimized

| Offered | 0.8B tok/s | 2B tok/s |
| ---: | ---: | ---: |
| 1 | 55.062 | 42.264 |
| 4 | 125.637 | 100.917 |
| 8 | 227.948 | 157.678 |
| 16 | 257.028 | 170.158 |
| 32 | 277.783 | 178.899 |
| 64 | 292.064 | 182.560 |

### Historical current-path ZG matrix recorded 22 Nov

| Offered | 0.8B tok/s | 2B tok/s |
| ---: | ---: | ---: |
| 1 | 39.621 | 34.802 |
| 4 | 119.613 | 83.038 |
| 8 | 133.676 | 96.099 |
| 16 | 137.410 | 95.255 |
| 32 | 135.918 | 96.091 |
| 64 | 136.385 | 95.962 |

The 22 Nov baseline marks C4+ diagnostic because that fused path had a
lane-invariance defect at that snapshot.

Do not merge the 23 Nov direct-runtime timings with this public population matrix
as if they were the same workload.

The direct-runtime 23 Nov measurements are useful for bottleneck direction only.

---

## 11. L5 27B streamed mode

Current smart startup on 6 GiB L5 chooses SSD-streamed weights for 27B.

Retained 47+3-token exact result:

- warmup ~311.557 s;
- timed ~305.902 / 306.003 s;
- median ~305.952 s;
- ~0.009805 output tok/s;
- clean shutdown.

Treat this as a separate memory-constrained streaming problem, not a resident
GPU arithmetic comparison.

---

## 12. Product interface design completed on 23 Nov

Three new dormant design specs define the next product iterations.

### 12.1 CLI / model lifecycle / TUI / Web

Authoritative spec:

`.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md`.

Iteration order:

1. current live implementation work;
2. CLI/model-lifecycle/TUI/Web iteration;
3. full OpenAI-compatible serving iteration;
4. later backend execution-performance iteration.

The immediate feature/build/run prerequisite in §2A has already largely landed
in `491cf99`.

#### Canonical normal CLI

Target surface:

```text
zg
zg run [MODEL] [PROMPT...]
zg serve [MODEL...]
zg model ...
zg device ...
zg pipeline ...
zg profile ...
zg config ...
zg doctor
```

`zerogravity` remains canonical binary; `zg` is a first-class alias.

#### One model resolver

Run/serve/model inspect/pull/verify/remove use the same typed model reference.

Friendly model names are normal UX; immutable snapshot identity remains source
authority.

#### Model lifecycle

Normal surface:

```text
zg model list
zg model search
zg model inspect REF
zg model pull REF
zg model verify REF
zg model remove REF
zg model import PATH
zg model alias ...
```

Advanced exact cache maintenance remains under `zg cache`.

#### Downloads

The design includes true:

- resolve -> plan -> transfer -> verify -> publish phases;
- resumable partials;
- pause/resume/cancel;
- bounded async/native operation model;
- rich TTY progress;
- plain non-TTY output;
- JSON/JSONL machine output.

Do not add a daemon merely for download control.

#### One config language

There is no separate settings database.

```text
~/.config/abird/zerogravity/
  config.yaml
  profiles/
    laptop.yaml
    workstation.yaml
```

Rules:

- `config.yaml` is the implicit/default profile;
- `profiles/<name>.yaml` uses the exact same `ConfigDocument` schema;
- `--profile NAME` selects exactly one named document;
- `--config FILE` selects exactly one arbitrary document;
- no profile inheritance/stacking;
- no hidden merge with `config.yaml`;
- CLI overrides the selected document.

#### Default TUI

Bare `zg` / `zerogravity` on a capable TTY opens the native full-screen TUI.

Primary navigation:

```text
Home
Running
Serve

Models
Downloads
Devices

Profiles
Config
Logs
```

The TUI is:

- minimal;
- keyboard-first;
- restrained colors/symbols;
- responsive;
- transparent about backend/device/model state;
- one main workspace plus contextual detail only where useful.

#### SolidJS Web control plane

`zg web` starts the embedded browser control UI.

Architecture intentionally borrows AW V1.5:

- SolidJS + Vite;
- assets embedded/served by Rust;
- no Node runtime in production;
- client-local selection/navigation state;
- one shared Rust `ControlPlane` underneath TUI/Web/CLI;
- snapshot + action + event model;
- frontends are projections, never runtime authority.

Namespaces remain separate:

```text
/v1/...          OpenAI-compatible inference
/zg/control/...  ZeroGravity control API
/zg/events       control event stream
/ui/...          embedded assets
```

TUI and Web may control the same process/runtime simultaneously.

---

## 13. Full OpenAI-compatible serving design

Authoritative spec:

`.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md`.

This comes after the CLI/TUI/Web surface stabilizes.

Current server is only a partial bootstrap:

- /health;
- /v1/models;
- /v1/completions;
- /v1/chat/completions;
- JSON/SSE;
- bounded ingress;
- auth/cancellation/shutdown.

The target local **inference** compatibility surface includes, as runtime
capabilities actually exist:

- Models;
- Responses;
- Conversations;
- Chat Completions;
- legacy Completions;
- Embeddings;
- Audio transcription/translation/speech;
- Images generation/edit/variation;
- Realtime WebSocket.

Do not clone hosted OpenAI billing/account/fine-tune/eval control-plane products.

### One semantic request layer

OpenAI protocol objects, CLI run and future native clients lower into one typed
semantic input/conversation/generation request.

HTTP must not own Qwen ChatML or model-specific role delimiters.

Model/tokenizer adapters own conversation lowering.

### Transport

Do not grow the current hand-written one-request HTTP parser into a general web
stack.

Target a mature native async Rust serving layer (likely Tokio + Hyper, with Axum
only where it genuinely simplifies routing/extraction).

Keep these types out of core/model/scheduler/state/GPU crates.

Preserve:

- explicit resource bounds;
- backpressure;
- disconnect cancellation;
- server/task lifetime ownership;
- graceful shutdown.

OpenAI auth and ZeroGravity control-plane auth are separate.

---

## 14. Later backend execution-performance design

Authoritative spec:

`.agents/plans/cpu-gpu-optimization/NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md`.

This is not “rewrite vLLM in Rust.”

The architecture remains:

```text
request/session scheduler + StateRoot authority
  -> logical authorized spans
  -> model-neutral physical batch
  -> prepared backend physical plan
  -> best qualified kernels/providers
  -> authoritative completion
  -> checked publication
```

Once activated, each performance slice closes backend-completely in order:

1. HIP first;
2. CUDA second;
3. Vulkan third;
4. then next slice.

Planned slices:

1. physical scaling diagnosis / B4->B8 cost closure;
2. token-oriented physical batching + ragged prefill;
3. persistent device batch metadata;
4. replayable production regions;
5. bounded two-flight overlap;
6. qualified best-primitive selection;
7. semantic fusion + kernel tuning;
8. sustained C16/C32/C64 serving closure.

Use vendor primitives without ideology when they win:

- hipBLASLt/AITER;
- cuBLASLt/FlashAttention/FlashInfer;
- Vulkan-native equivalents.

The control/state architecture remains ZeroGravity's.

---

## 15. Full post-origin chronology

Tracked origin is `2a595be`.

The local 25-commit lineage above it is:

1. `b34c025` — establish clean current baseline.
2. `94a7e2d` — batch HIP source decode projections.
3. `48d3d94` — size HIP decode cohort head scratch.
4. `79de45a` — observe HIP decode cohorts in the benchmark path.
5. `cb923cc` — price HIP decode cohort projection scratch.
6. `189c718` — accept HIP fused cohort observations.
7. `b6c60ef` — record HIP decode cohort checkpoint.
8. `99dcbbe` — batch HIP row-wise decode operators.
9. `e35e650` — compare HIP decode cohort semantics structurally.
10. `537e5e8` — advance HIP decode cohort baseline docs.
11. `9ee877c` — batch aligned HIP decode rotary prep.
12. `a5ec4c3` — seal batched HIP decode rotary variants.
13. `a390837` — advance HIP aligned decode baseline docs.
14. `1de99ef` — batch independent HIP decode conv state.
15. `c33016d` — preserve HIP conv coefficient input contract.
16. `57cc384` — record stateful HIP conv cohort.
17. `491cf99` — unify production GPU build/run paths.
18. `6f8f071` — separate attention semantics from providers.
19. `64c06e2` — add opt-in native FlashAttention provider.
20. `ed0382a` — keep FlashAttention off decode path.
21. `0a35f17` — fuse Q/K normalization and rotary prefill.
22. `2805c37` — enable fused Q/K rotary prefill.
23. `179bb15` — fuse GDN recurrent export certification.
24. `c829aa9` — retain B1 direct GDN conv primitive after losing selector A/B.
25. `f3f872c` — retain B1 fused head-greedy primitives after neutral/lost selector A/B.

Re-run `git log origin/main..HEAD` for the authoritative sequence if HEAD has
advanced.

Do not use this chronology as a substitute for source/tests/TODOs.

---

## 16. What remains open

### Immediate concurrent work

First priority is to let the current owner finish/reconcile the dirty FlashInfer
decode + paired GDN state transport slice.

Do not independently edit the same files.

When it lands:

- revalidate current HEAD;
- inspect selector policy;
- run the smallest real Qwen A/B;
- preserve reusable primitives even if the selector loses;
- update live baseline/handoff plan state.

### CUDA performance

The current matched direct-runtime gap is still large.

Highest-value areas:

- GDN paired state import/export/certification;
- recurrent/conv state mechanics;
- unnecessary per-layer dispatches;
- execution geometry and persistent/replay later;
- FlashInfer decode only if it wins real Qwen A/B;
- SM89 full-chain qualification for A1 after actual remaining SM86-closed leaves
  are generalized.

Do not spend time adding more attention-provider abstraction; the provider seam
already exists.

### HIP

- recurrent GDN cohort/state work remains the next major batching target;
- singleton efficiency remains a separate problem;
- run real A1 gfx1150 HIP 0.8B/2B once the exact L5 checkpoint is synchronized;
- continue X2 27B/large-VRAM work only from exact L5 commits.

### Vulkan

Still needs semantic/capability parity with the CUDA/HIP production surface
before the later backend-complete performance program can truly activate.

### Numerical closure

Independent populated larger-context all-state-domain numerical closure remains
open where the approved independent reference archive is absent.

Do not manufacture a reference merely to close it.

### Product interface

CLI/TUI/Web plan is designed, not implemented.

After current live low-level work closes, implement the UX iteration before the
full OpenAI-server rewrite and later performance architecture iteration unless
the owner reprioritizes.

### OpenAI server

Responses/Embeddings/Audio/Images/Realtime full compatibility is designed but
not yet implemented.

### Pipelines / broader mission

Continue toward one reusable Rust execution fabric for:

- Qwen text;
- Gemma;
- ASR;
- TTS;
- image;
- ASR -> LLM -> TTS;
- multimodal single-process pipelines.

Do not create per-model engines where shared primitives suffice.

---

## 17. Evidence discipline

Do not overengineer tests/evidence.

Default implementation loop:

1. state the measured/observed deficit;
2. choose the smallest reusable primitive;
3. compile/focused unit validation if needed;
4. one real small model canary;
5. one matched performance cell;
6. only expand when a real failure demands it;
7. checkpoint a coherent implementation unit.

Reuse existing:

- `tools/serve-bench.py`;
- `tools/serve-compare.py`;
- `tools/serve-matrix.py`;
- public SDK/population tests;
- existing source/context qualifiers.

Do not create a second benchmark framework.

Do not turn temporary benchmark scripts into tracked infrastructure unless they
become a genuinely reusable product tool.

---

## 18. Exact next-agent safety rules

- L5 only for source authoring.
- Preserve the current dirty FlashInfer/GDN work.
- Do not reset/clean/checkout over it.
- Do not push unless explicitly requested.
- Commits remain signed.
- Stage only paths you own.
- Do not weaken CUDA/HIP/Vulkan device/source/numerical gates for convenience.
- Do not revive Podman/FUSE for vLLM; current baseline is Nix-native.
- Do not resurrect historical eager-vLLM numbers as current engineering input.
- Do not build B16/B32/B64 kernels to satisfy offered concurrency.
- Do not add another paging/COW/state authority.
- Do not let frontend/control-plane state become runtime/model authority.
- Do not delete reusable primitives solely because current Qwen loses an A/B.

---

## 19. Starting prompt for the next agent

Use the following verbatim unless current source state has obviously advanced:

~~~text
Use dotlink---l5 as the only source-authoring connector.

Continue in /home/pvl/spaces/abird/src/zerogravity on the single persistent main
worktree. Use dotlink---x2 only for AMD/ROCm large-VRAM execution/testing and
synchronize exact L5 commits by Git bundle rather than pushing merely for
transport. A1 is now explicitly available for mixed NVIDIA SM89 + AMD gfx1150
build/runtime validation, but remains execution/testing only; source changes land
on L5 first.

Read first, in order:
1. AGENTS.md
2. .agents/plans/README.md
3. .agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md
4. .agents/docs/CURRENT_HANDOFF.md
5. .agents/docs/CURRENT_BASELINE.md
6. .agents/plans/TODO.md
7. .agents/plans/model-execution/TODO.md
8. .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md
9. .agents/plans/serving-and-speculation/TODO.md
10. .agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md
11. .agents/plans/memory-and-streaming/TODO.md
12. .agents/plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md
13. .agents/plans/pipeline-composition/TODO.md
14. .agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md
15. .agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
16. .agents/plans/cpu-gpu-optimization/NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md
17. any owning docs linked from the live TODO for the current slice.

Revalidate before editing:
- git status --short --branch
- git rev-parse HEAD
- git rev-parse origin/main
- git log -1 --show-signature
- worktrees
- live Cargo/rustc/model/native-gpu/vLLM/nvcc/ptxas/Bubblewrap processes
- physical GPU state
- source/cache identities
- disk space and _tmp artifacts.

At handoff creation the latest signed source checkpoint is
f3f872cb567a6885918ae2b4866ee6d535d736d8 —
feat(cuda): retain B1 fused head greedy primitives.
The handoff docs commit may sit above it; revalidate actual HEAD first.

Tracked origin/main at handoff creation is
2a595be84bc7c5dccf7509e35e4f9b3a341662a2.

IMPORTANT: another agent owns dirty FlashInfer-decode + paired-GDN-state work in
the CUDA/provider/state-transport files listed in the 23 Nov handoff. Preserve it.
Do not reset, overwrite, partially stage or independently edit those overlapping
paths until the owner lands/relinquishes them.

Use CARGO_BUILD_JOBS=8 and
RUSTC_WRAPPER=/run/current-system/sw/bin/kache for Cargo.

Preserve the architecture:
- exact StateRoot publication;
- cancellation/unknown-completion authority;
- ResourceArbiter ownership and physical lifetime;
- DeviceBufferIdentity as physical ownership truth;
- exact paged COW/prefix compatibility;
- one scheduler and one state engine;
- physical cohorts B1/B2/B4/B8;
- prepared/load-time provider selection;
- frontend/control state is never runtime authority.

Durable experiment rule:
Reject the selector/consumer, not a correct reusable primitive.
Retain clean model-neutral kernels, transports, layouts, scheduling transforms,
providers and ownership/completion mechanics for future models/shapes/devices.
Delete a primitive only for correctness/safety/resource-contract failure or true
strict supersession.

Current key facts:
- production feature algebra is backend capability × model contribution;
- one --features all binary supports normal CPU/CUDA/HIP/Vulkan/model/cache/serve;
- nix develop .#gpu supplies mixed CUDA+ROCm tooling;
- run-cuda/run-rocm/run route through the production Runtime model-load/request seam;
- A1 HIP gfx1150 reports ~16.36 GB driver pool and should admit current 0.8B/2B;
- A1 RTX4060 is SM89; FA2 compiles for it but the full Qwen CUDA chain still has
  SM86-closed leaves and must not be falsely advertised;
- semantic Attention is separate from physical providers;
- native FA2 is retained, prefill-only when selected; decode FA2 lost its A/B;
- Qwen3.5 0.8B has 18 GDN layers and only 6 full-attention layers;
- fused recurrent export+certification improved the real 8+64 workload about
  906.4 ms -> 855.3 ms;
- current direct same-snapshot vLLM is ~67.8 ms for 768+1 and ~426.4 ms for 8+64,
  so GDN/state/dispatch work is a higher-value target than more attention abstraction;
- B1 direct cached-conv and B1 fused head-greedy primitives are intentionally
  retained dormant after their current Qwen selectors lost/failed to win;
- matched vLLM is Nix-native; do not return to Podman/FUSE.

First action:
revalidate whether the concurrent FlashInfer + paired-GDN-state dirty slice is
still present or has landed. If still dirty, do not touch it; inspect only and
advance a non-overlapping ready item. If it landed, review it against the durable
primitive-retention rule, run the smallest real Qwen correctness/performance A/B,
and checkpoint only a coherent win or retained reusable primitive.

Keep each implementation turn around 30 minutes. Prefer implementation + quick
real-model feedback over speculative fixture/evidence machinery. Use the existing
benchmark tools; do not create a second framework. Preserve simple composable
zero-cost Rust abstractions and aggressively optimize measured hot paths only
when the fast design stays clear and maintainable.

Do not push unless explicitly requested.
~~~
