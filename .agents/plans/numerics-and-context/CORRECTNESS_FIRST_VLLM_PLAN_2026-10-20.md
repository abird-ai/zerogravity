# Correctness first, then measured vLLM performance

Author: pvl
Co-authors: Codex, baseline_review, spine_review, batch_design_review
Owner: @pvl
Date: 2026-10-20
Source checkpoint: `fb68d8fa165de16e21d1bd37afef3790b80da971`
Status: investigation and proposed implementation plan only; implementation paused.

## 1. Decision and scope

First make Qwen3.5-0.8B, Qwen3.5-2B and original Qwen3.8-27B numerically
correct under explicit reference arithmetic. Then optimize the same engine
and demonstrate a workload-specific advantage over pinned vLLM.

Use pinned vLLM as the primary target for the new CUDA execution profile.
Treat it as correct unless an independent counterexample establishes otherwise.
Keep the accepted llama.cpp reference and Torch CPU profiles as regression
oracles. Their differing reductions and casts must not be silently combined
into an invented fourth reference.

This document records the implementation sequence and acceptance decisions.
The existing [numerics TODO](TODO.md), [model execution TODO](../model-execution/TODO.md)
and [serving TODO](../serving-and-speculation/TODO.md) remain the only owning
queues. No implementation or numerical gate is completed by writing this plan.

**Fastest route:** reproduce the earliest differing operation with identical
inputs; port its actual reference arithmetic; immediately compose its real
producer and consumer; progress to a complete layer and model. Avoid extended
standalone helper work that never reaches a real token.

## 2. What the evidence establishes

| Area | Established | Still missing |
|---|---|---|
| Small-model GGUF reference profile | Fresh mixed/Unicode corpus through 8K; both models match pinned llama.cpp AR1 and explicit `use_ref` control at the recorded checkpoints: 730 selected records, full 248,320 logits, all 42 causal domains; 55,382,016 state words/model at 8K | Default bulk arithmetic and the unrecovered historical corpus |
| Native small-model CUDA | Complete B=2/4/8 model graphs, independent row positions, CPU/B1 logits and all 42 domains within accepted source/context scope | Broader independent numeric profiles, genuine long-context GPU, temporal prefill |
| Original HF CPU | Numerous source-bound operators and isolated compositions pass; original reference captures exist | Complete trained .8B/2B/27B model composition and full-state/logit parity |
| Original 27B | Original source acquired; H48 cached recurrence, gated norm and wide-role isolated tests pass | Live producers, complete initial chunk, complete model/112 domains, admitted physical execution |
| vLLM | Genuine pinned V1 and official GGUF plugin ran both small models | Cross-engine full numerical/state parity, optimized default-route warm/concurrent comparison |
| Performance | Bounded eager C1 comparison shows a large native deficit | Numerical qualification of compared profiles and representative performance advantage |

The original assertion that complete B>1 GPU fusion is absent is superseded
by accepted P12-B7 and later work. Cross-request full-model fusion exists.
Temporal prefill still executes a full model step per prompt token.

No evidence currently proves a single defective GDN recurrence equation is
the cause of every discrepancy. “First difference observed in GDN” is not
equivalent to “GDN produced the first numerical error.”

## 3. Divergence inventory and fixes

“Proven difference” below means the cited source or executed comparison
establishes that difference. It does not automatically establish a defect in
a separately declared arithmetic profile. This is the complete currently
identified inventory, not a claim that undiscovered bugs cannot remain.

| ID | Finding and confidence | Correct layer to change | Proposed fix and decisive test |
|---|---|---|---|
| N1 | **Executed:** fresh .8B first64 default/AR1 difference begins at layer3 attention AV, position2: 530/2,048 words. Probability operands and causal V values match. This precedes layer4 GDN. | F32 attention-value reduction and temporal dispatch selection | Reproduce original AV inputs through the pinned tinyBLAS reduction; compare every output. Select that contract only for the matching bulk profile. Then rerun layer3→layer4 and the complete prefix. |
| N2 | **Executed:** fresh 2B first difference is layer0 F32 alpha projection, position0: 13/16 words; beta also differs before recurrence. Earlier compared rows and alpha operands match. | F32 projection reduction | Same bulk primitive with original alpha weights/inputs. Require exact replay, then alpha/beta→coefficients→GDN state. |
| N3 | **Executed/source identified:** default bulk enters tinyBLAS's one-Vec8 accumulator path; AR1/reference uses four Vec8 partials. Float reassociation changes results. Only first named compiled entries were observed. | Numerical contract and schedule, not a global `gdn_step` replacement | Add closed bulk schedule without changing `GgufSourceV1` defaults. Reuse the prepared implementation only after compiling and replaying it. Full default-route model closure is a separate gate. |
| N4 | **Recorded route difference:** later 2B `ssm_out` uses Q8 weight dequantization followed by F32 BLAS in one route, versus activation-Q8/integer-dot arithmetic in AR1. | Quantized projection dispatch and temporary activation format | Reproduce both routes with the same raw source, explicit dequant/activation quantization, block scaling and accumulation. Fix route selection for the declared target; an F32 alpha fix alone is insufficient. |
| N5 | **Source-proven profile difference:** original-HF diagnostic rounds convolution to BF16 before SiLU; pinned vLLM fused convolution applies SiLU to the FP32 accumulator before BF16 store. | Typed causal-convolution/activation operation | Add a vLLM CUDA contract with the reference tap order, contraction and one output cast. Keep the independently qualified Torch CPU convolution→BF16→SiLU route intact. |
| N6 | **Source-proven profile differences:** vLLM prefill Q/K normalization stores BF16; its fused beta path stores F32. Current diagnostic keeps normalized Q/K in F32 and rounds beta to BF16. | Post-convolution normalization/coefficient preparation | Explicit phase-specific dtypes; compare Q/K, beta, log-decay and state using identical inputs. Do not change the Torch CPU target, where BF16 beta is intentional. |
| N7 | **Source-proven profile difference and diagnostic defect:** vLLM gated RMS keeps arithmetic F32 through gamma and SiLU until output store. The old diagnostic also rounds the weighted product even when raw F32 gamma should promote the CPU product to F32 (`original_qwen_precision.rs:140–143`). | Gated norm, with gamma dtype retained | Port selected vLLM norm; replace the diagnostic's unconditional weighted-product cast with the already qualified dtype-specific CPU contract when composing that CPU target. Keep these fixes separate. |
| N8 | **Executed/source-proven:** original HF initial chunk and cached step use different algorithms. Cached CPU query divides by rounded sqrt(128); initial CPU query multiplies rounded inverse sqrt. CPU initial cumsum has F64 carry/F32 stores. | Phase-aware GDN core, scale and prefix operations | Implement and select initial/continued-prefill/cached-decode contracts explicitly. A cached step on zero state is not an initial chunk qualification. |
| N9 | **Unfinished producer:** authentic recurrence replays import decay and other intermediates. Standalone Torch CPU `exp` uses a different provider from SLEEF expressions embedded in sigmoid/SiLU. | Coefficient producer and native math admission | Connect live `A_log→exp→negative`, alpha/dt/softplus, log-decay and decay; use the actual profile's exp implementation. Never substitute host `exp` or recorded lookup values. |
| N10 | **Executed/source-proven:** official BF16 loading and restoration of original F32 tensors produce different outputs. Accepted GGUF projections prepare Q8_0/Q8_1/Q8_K activations by codec; the stock plugin uses its own Q8_1 path. Storage equality does not imply matching activation/cast/reduction schedules. | Source binding, loader, projection and numerical profile identity | Pin effective loaded dtype and activation quantizer by role. Separate official-loaded, mixed-source, GGUF-AR1 and vLLM CUDA comparisons. Authentic HF cuBLAS operands do not qualify the GGUF-plugin projection. |
| N11 | **Integration gap:** isolated correct operators do not constitute a correct model. Ordinary norm, attention, residuals, MLP and final head can introduce the next difference after GDN. | Existing complete ForwardPlan and HostMath/resident backends | Trace in causal order and compose real producers. Check every layer's residual, complete logits and every state domain, then test the next teacher token. |
| N12 | **Geometry/layout risk, not newly demonstrated corruption:** 27B has 16 key heads→48 value heads, source/canonical state orientation differences, four-frame source conv cache versus canonical newest-three history. | Source adapter and typed layout/state codec | Explicit repeat-interleave, axis transpose and history mapping; asymmetric per-head fixtures and genuine 27B captures. No reinterpretation based merely on equal byte count. |
| N13 | **Evidence gap:** original historical 8K raw token corpus is unrecovered. Fresh exact 8K results cannot retrospectively replay it. | Reference corpus recovery/reporting | Make one bounded recovery attempt from retained original inputs; otherwise retain the historical item as unreproduced and close only the explicitly named fresh corpus. |
| N14 | **Measurement gap:** stock plain/observed exact logits prove observer transparency, not native/stock parity. Matching text or argmax hides large tensor differences. | Comparator and qualification boundary | Require independent full-vocabulary and canonical state comparison; bind benchmark validity to that qualification. |
| N15 | **Source-proven loader transform:** plugin reconstructs `A_log=log(-ssm_a)` from GGUF negative exponentiated decay; later exp need not recover identical stored bits. Ordinary norm subtracts one; GDN gamma is excluded. Grouped heads also require explicit permutations. | Effective loaded-parameter adapter, before GDN | Compare actual loaded parameters and faithfully reproduce the selected loader. Preserve raw source identity and derived representation separately; do not “fix” this with extra negation or in-place weight mutation. |

No new evidence justifies changing RoPE, global signed-zero treatment, the
accepted GGUF recurrence, or tolerances as a blanket “GDN fix.”

## 4. Numerical contracts

### 4.1 Keep four targets explicit

| Target | Purpose | Required identity |
|---|---|---|
| Accepted GGUF AR1/reference | Preserve working CPU/CUDA behavior and independently exact fresh8K results | Original GGUF, existing reduction/quant policy, F32 KV, FA off in reference |
| Pinned llama.cpp default bulk | Explain and reproduce the default comparison discrepancy | `d812350`, actual CPU/BLAS route, batch/ubatch, strides, thread count, temporary quant format |
| Pinned Torch CPU original HF | Independent trained-source oracle and existing CPU operator work | Effective loaded tensor dtypes, Torch/provider versions, vector width/tails, initial/cached schedule |
| Pinned vLLM CUDA | Primary new correctness and performance target | vLLM, plugin, source/loader, CUDA kernels, phase, activation/state dtype, compiler/math modes |

Extend `ProgramNumericalProfile` and typed operator contracts in the existing
core; the proposed vLLM and bulk profiles do not exist in MAIN today.
Profile identity belongs in compiled-plan/cache keys and state compatibility.
Reject incompatible reuse rather than choosing a dtype from a display name.

### 4.2 Mathematical GDN definition

For each value head, let `S[k,v]` be its FP32 recurrent matrix, with independent
convolution history and causal frontier. Original 27B uses 128 key/value
dimensions and repeats each of 16 Q/K heads into three adjacent value heads.
Converted GGUF layouts require their own verified mapping.
This mathematical notation is the transpose of vLLM's actual contiguous-key
`[head,value,key]` cache; it is not permission to reinterpret the bytes.

The semantic recurrence is:

```text
u_t = depthwise_conv4(project_qkv(x_t), prior_history)
(q_t, k_t, v_t) = split(SiLU(u_t))
qhat = q_t / sqrt(sum(q_t*q_t) + eps_l2)
khat = k_t / sqrt(sum(k_t*k_t) + eps_l2)
qscaled = qhat / sqrt(key_dim)
beta = sigmoid(project_b(x_t))
g = -exp(A_log) * softplus(project_a(x_t) + dt_bias)
d = exp(g)
Sdecayed[k,v] = d * Sprevious[k,v]
prediction[v] = sum_k(khat[k] * Sdecayed[k,v])
delta[v] = beta * (v_t[v] - prediction[v])
Snext[k,v] = Sdecayed[k,v] + khat[k] * delta[v]
core[v] = sum_k(qscaled[k] * Snext[k,v])
gated = RMS(core, eps_norm) * gamma * SiLU(project_z(x_t))
output = project_out(gated)
```

These equations specify algebra, not floating-point instruction order.
For the pinned target, the required implementation also includes each cast,
reduction tree, FMA boundary, exponential approximation and layout.
Pinned vLLM Q/K L2 epsilon is fixed `1e-6`; gated RMS uses
`config.rms_norm_eps`. These are separate fields.
Do not reuse GGUF grouped-RMS lowering as proof of the original FLA algorithm.

### 4.3 Exactness rules

- BF16 stores: round-to-nearest, ties-to-even at the reference boundary;
  preserve signed zero. Widening BF16 to F32 is exact and is not an extra
  rounding step.
- Raw learned parameters retain their actual dtype and source role. Ordinary
  Qwen RMS's `1 + weight` semantics do not apply to GDN's raw learned gamma.
- FP32 state remains FP32; BF16 core/output does not permit BF16 recurrent
  storage. Preserve source accumulation order and every materialized boundary.
- Source FMA is one rounding; separate multiply/add is two. Do not enable
  blanket fast-math or silently contract a CPU tensor expression.
- Padding is masked according to the reference. An intentional `-Inf`
  triangular mask is valid only inside that closed operator; it does not
  weaken general finite-value validation. Do not multiply poisoned future
  values by zero and assume the result is harmless.
- Pin denormal/FTZ behavior, exponential/reciprocal implementation, matrix
  compute/output types and TF32 settings. A cuBLAS API's “FP32 accumulation”
  does not reveal its reduction tree.
- Kernel source alone cannot establish the exact generated CUDA instruction
  stream. Where bit identity is promised, replay actual compiled kernels on
  authentic operands. Record unresolved instruction/provider details as
  unresolved rather than filling them in from algebra.

### 4.4 Initial prefill versus decode

Use the actual selected pinned vLLM kernel, not a kernel guessed from an
environment-variable name. On SM86 the inspected automatic prefill branch
selects Triton. With packed decode enabled, ordinary non-speculative decode
selects `_forward_core_decode_non_spec` and Triton packed recurrence.
`GDN_DECODE_KERNEL=cuda` alone does not prove the guarded C++ MTP path ran.
Confirm branch/environment at runtime when implementation resumes.

**Normal packed decode, all temporaries F32:** widen conv output; divide Q/K
by `sqrt(sum(square)+1e-6)`; multiply Q by `K**-0.5` before readout; compute
`x=a+dt_bias`, `softplus=select(x<=20,log(1+exp(x)),x)`,
`g=-exp(A_log)*softplus`, `beta=sigmoid(b)`; decay state, subtract its K dot
from V, multiply residual by beta, update state and reduce against scaled Q.
Store core BF16 and recurrent state in its explicitly configured FP32 cache.
The state-decay exp alias is `tl.exp` by default and `fast_expf` under
`FLA_USE_FAST_OPS=1`; coefficient exp/log/sigmoid are direct Triton operations.
Preserve the selected setting and validate generated arithmetic.

**CUDA initial/continued prefill, chunk64:**

1. Conv FP32 accumulation → `acc/(1+exp(-acc))` → BF16 store.
   Normalize Q/K in F32 using reciprocal sqrt, multiply, then store BF16.
   V remains BF16; beta and log-gates are F32.
2. Prefill softplus uses the stable positive branch
   `x+log(1+exp(-x))`, nonpositive `log(1+exp(x))`, then threshold20.
   This differs from the decode expression; do not collapse the two.
3. Compute chunk-local F32 prefix log-gates `G=cumsum(g)`.
   Form the strict-lower system
   `A[i,j]=dot(beta[i]*k[i],k[j])*exp(G[i]-G[j])` in the source graph.
   Beta multiplies the operand before the dot; moving it outside changes
   rounding. Keep the promoted operand dtype selected by the CUDA branch.
4. `solve_tril` forms `(I+A)^-1`, including its source negation and unit
   diagonal. Preserve the 16×16 solve and block-merge order rather than
   substituting a generic inverse. It returns **BF16 K dtype**.
   Pin `FLA_TRIL_PRECISION` (source default `ieee`) and actual dot policies.
   The WY construction casts scaled V/K operands to BF16 before its dots;
   `u` and `w` outputs are BF16.
5. Running state stays F32 `[head,value,key]`; each chunk's saved entering
   state `h` is BF16. Compute `v_new=u-w@h` and store BF16. Apply chunk decay
   to F32 state; cast the decay-adjusted delta BF16 before its K/delta dot.
   Final recurrent state is F32.
6. Output uses BF16 Q/K/h/v_new dots with F32 accumulators and causal masks;
   gate-adjusted attention coefficients are cast BF16 before the V dot.
   Scale both output terms, add and store BF16.
7. Gated RMS widens the BF16 core, computes norm/gamma/SiLU product in F32,
   then stores once to BF16. No CPU intermediate BF16 gamma-product cast.

This is a mixed-precision chunk/WY algorithm. Keep sequential recurrence as
an independent semantic oracle; it is not an instruction-exact substitute.
Reproduce actual reductions, masks and boundary state from the pinned kernels.
No newly executed vLLM→Rust operator comparison is asserted by this table.

The existing Torch CPU target has a distinct known schedule:

1. Widen and normalize Q/K; initial query multiplies F32 `0x3db504f3`.
   Cached query divides by rounded F32 sqrt(128).
2. Initial short input pads to64; materialize beta*K and beta*V.
3. Prefix scan uses F64 carry with an F32 store at each element.
4. Form masked prefix differences, exponentials, Gram products and
   unit-lower solves; retain materialized intermediate boundaries.
5. Preserve the six recorded BMM families and two solve groups, column-major
   solve outputs and final valid-row extraction. State remains FP32.
6. Cached update uses materialized products/reductions, separate subtraction,
   multiplication and addition. Its tested state uses key-major orientation.

This CPU schedule must not be copied into vLLM's Triton path merely because
both implement GDN. Conversely, vLLM's fused casts must not replace the
already qualified CPU profile.

### 4.5 Small-model default bulk arithmetic

The minimal existing candidate implements one eight-lane FP32 FMA accumulator
per output, processing K in eight-element groups, including the source's
zero-filled tail operations. Its horizontal sum follows the pinned low/high
lane additions, then `(low0 + low2) + (low1 + low3)`. AR1's four partial
accumulators remain a separate contract.

Validate the original strided AV and F32 alpha captures, together covering
132,096 output words. Then advance through downstream routes, including
Q8/dequant/BLAS. The prepared `root-default-bulk-20261020` worktree is
uncompiled and unqualified; its existence is not a fix.

## 5. Porting strategy and implementation boundaries

Port small mathematical operations and the reference's phase/layout logic.
Reuse the existing `MathematicalBackend`, `ForwardPlan`, source descriptors,
`HostMath`, resident GPU queue, `ResourceArbiter` and `SessionAuthority`.

| Existing area | Intended change after implementation resumes |
|---|---|
| `zerogravity-core/src/accelerator/program/typed.rs` and typed children | Explicit profile, phase, dtype and layout contracts; only qualified operations become selectable |
| `zerogravity-cpu/src/original_hf/` and `src/math/` | Independent semantic/source-profile reference operations; preserve accepted CPU schedules |
| `zerogravity-model-qwen35/src/forward.rs`, `host_math/`, source bindings | Compose actual trained producers in the existing plan; preserve source-derived geometry and state effects |
| `zerogravity-gpu/src/resident_program.rs` and operator modules | vLLM-profile CUDA operations and temporal kernels through the original queue |
| `resident_inference.rs`, `resident_batch.rs` | Phase-aware temporal prefill, independent rows, state/currentness and physical completion |
| `zerogravity/src/original_qwen_math.rs` | Keep diagnostic labeling explicit; do not promote it to a second production engine |

Do not port vLLM's Python model/session engine or allocator. If a provider
kernel is reused directly, it must fit the original physical admission,
buffer lifetime, event and quarantine model. A Python subprocess is an
independent oracle, not a production inference fallback.

Preserve upstream license/SPDX notices and exact source revision when porting.
Prefer the actual selected upstream kernel over a convenient unused helper.
No new broad plugin framework is needed for a handful of typed operations.

## 6. Fast implementation sequence, when resumed

### C0 — Lock the comparison, not another framework

Use existing raw teacher IDs, retained captures and pinned runtimes.
For each run record the model/source, effective loaded dtypes, phase,
chunk size, state layout and operation route in the existing test output.
Reuse existing process ownership and reference tools.

Choose the lane explicitly. Recommended primary lane: original .8B HF with
vLLM's actual loaded GPU dtypes, avoiding the extra GGUF log/exp and temporary
quantization transformations during the first GDN diagnosis. Use that same
profile for 2B/27B when admitted. Keep the existing GGUF-plugin benchmark lane
separate; it must reproduce its own loader/quantizer to qualify numerically.
Neither lane's pure operator captures may be relabeled as the other's.

For .8B, begin with teachers11/353 and the first64 corpus. Capture projected
QKV/Z/a/b before GDN to determine whether a projection mismatch already
precedes the known cast differences. Do not assume the diagnostic matches
vLLM at its GDN inputs.
Bind actual phase and partition metadata: teacher353 must really take the
claimed cached/continued route. Repeat the exact reference partition in each
comparison; different chunkings or GEMM shapes are not automatically bit-equal
because they share a profile name.

Exit: the first differing operation has identical compared inputs and source
weights, or the earlier source/loader/projection mismatch is precisely located.

### C1 — Close the earliest small-model default discrepancies

Compile and test the existing bulk primitive; replay AV and alpha.
Compose real producer→operator→consumer and run the same first64 tokens.
Follow the next actual difference, including beta and Q8/BLAS output routes.

Keep this a bounded CPU side task while the vLLM CUDA profile is developed.
Do not delay the primary vLLM route on recovering an unavailable historical
file or on completing an unrelated generic arithmetic framework.

Exit: exact authentic leaf replays and a declared complete default schedule
comparison; accepted AR1 results unchanged. Historical closure remains
separate if its raw inputs cannot be recovered.

### C2 — Complete one vLLM-profile GDN layer

After projection parity, implement convolution+SiLU → phase-specific Q/K
normalization/replication → coefficients. Qualify the smaller cached core
and gated RMS first using authentic nonzero prior state, then implement
initial/continued-prefill chunk64 and connect initial→cached continuation
plus output projection. Start with .8B, then use 2B and H48/27B for geometry.
Completing the separate Torch CPU MKL/BLAS provider is not a prerequisite
for the vLLM CUDA port.

For every fix require a failing-before/passing-after genuine-operand case.
Compare full intermediates and state, not just one output element.
Connect each accepted operation immediately; imported reference operands may
isolate a bug but cannot replace the native producer in the composed gate.

Exit: initial layer output/state followed by genuine cached continuation
matches the target; no reference intermediate is injected into the final test.

### C3 — Complete .8B, then 2B in the same plan

Compose all layers: embedding, norms, full attention, GDN, residuals, MLP,
final norm and head. Trace the first new mismatch rather than repeatedly
dumping every later error. Inspect the full 248,320 logits and all42 domains.

First B1, then B2/4/8, ragged positions, refill and B1↔batch transitions.
Do not carry an assumption that one-token and bulk GEMM use identical
reductions; qualify the actual selected algorithms/shapes.
The source-correct safe chunk path is itself part of correctness; its initial
implementation need not yet be the fastest temporal-prefill implementation.

Exit: small-model full correctness matrix below, including initial→decode
and whole-prefill→chunked-prefill continuations.

### C4 — Complete original27B without relaxing resource guards

Use the same recipe/operations with source-derived geometry: 64 layers,
48 GDN/16 full-attention layers, 112 state domains, hidden5120,
MLP17408, vocabulary248320 and an untied head.

Reuse verified source windows and isolated H48/wide-role work, including the
already tested full17408-column CPU down-projection role. Finish live
coefficients, initial chunk and trained producer/full-model composition.
The corresponding native GPU wide-column path remains unqualified;
geometry refusal must not be removed without the corresponding arithmetic.

Original BF16 weights are about55.56GB. They cannot fit the local6GiB GPU;
current executable-source and GPU-width guards also remain. First establish
complete CPU/reference execution on an admitted capable host or correctly
accounted source-window path. For GPU serving choose either a capable
matched resident device or complete physical RAM/SSD/GPU streaming.
Compare vLLM under the same capacity/offload constraints; an unsupported
comparison is reported as unavailable.

Exit: original27B full logits/all112 domains and continuation qualify.
Operator-only success, acquired metadata and a source window are not this gate.

### C5 — Accept and checkpoint

Run focused regressions, relevant strict Clippy, physical CUDA/state/cancel
tests, both approved original cache-direct/HF-import release routes and:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

Use one MAIN Cargo/GPU coordinator; independent agents own disjoint source
cuts/reviews. Preserve root README, unrelated overlays and the Renoir/gfx90c
launch hold. Review and sign only owned verified changes. Reconcile existing
plan TODOs; no extra status ledger or source-inventory framework.

## 7. Numerical parity matrix and acceptance

| Axis | Small first gate | Expansion required for the claim |
|---|---|---|
| Sources | .8B original GGUF and official HF, explicitly separate | 2B equivalents; original27B, then individually qualified quant profiles |
| Teacher inputs | Tokens11/353; retained mixed/Unicode first64 | Fixed raw IDs for prose, code, Unicode, repeats, zeros/extreme operator inputs; held-out prompts |
| Phase boundaries | Empty→first prefill→cached step | 1/2/31/32/63/64/65/127/128/129 tokens; ragged chunks and continued prefill |
| Context | Short complete model | 512/1024/2048/4096/8192; 16K/32K only after actual capacity and predecessor gates |
| Batch | B1 | B2/4/8, independent positions, refill, row reorder, canceled neighbor |
| Outputs | Full affected tensor/state | Every layer residual, full248320 logits, all42 or112 domains and next-token continuation |
| State lifecycle | Initial and cached | Snapshot/restore, fork/COW, cancellation before/after submission, failure/quarantine, final reclaim |

Required distinguishing tests, using independently generated expected values:

- Conv sums straddling a BF16 rounding boundary before SiLU; one-store CUDA
  and two-store CPU results must remain distinguishable.
- Nonzero beta that changes under premature BF16 rounding; positive/negative
  softplus inputs and values on either side of20, separately for both phases.
- Zero/tiny/nontrivial Q/K sums; prefill BF16 versus cached F32 normalized
  vectors and fixed L2 epsilon independent of model norm epsilon.
- Asymmetric `HK=2,HV=6` sentinels proving `[0,0,0,1,1,1]` grouping; genuine
  16→48 source/canonical permutation and quantized output-layout checks.
- Nonuniform signed prior state and Q exposing transpose and scale-before/
  scale-after reduction; a zero-state-only fixture would miss both.
- F32 and BF16 raw gamma cases distinguishing promotion from intermediate
  BF16 rounding; compare the relevant CPU and CUDA targets separately.
- Length63/64/65 with nonzero initial state and invalid padded lanes;
  compare solved-A, u/w, saved h, v_new, final state and complete next output.

Three separate verdicts are required:

1. **Same source and same arithmetic contract:** exact bits for promised
   leaf replays, deterministic internal CPU/B1/batch state/logits and integer
   layouts. Zero-sign differences are reported, not silently normalized.
   Phase, shape, partition and selected kernel are part of that contract;
   cross-partition algorithm changes use a separately qualified comparison.
2. **Independent implementations with deliberately different arithmetic:**
   report max absolute/relative error, RMS/relative-L2, ULP distribution,
   full-vocabulary distribution metrics, top-k/rank and per-domain drift.
   Preserve existing numerical limits: historical functional shared-top20
   logprob0.45 and top-token probability0.05 are only those specific tests,
   not permission for arbitrary state error or a new universal tolerance.
3. **Model behavior:** teacher-forced loss/perplexity, continuation and task
   quality on held-out checkpoint-matched inputs; stochastic policy and
   sampling distribution separately when advertised.

There is no existing universal BF16 state/logit tolerance that can honestly
be asserted to qualify all three models against every vLLM kernel.
For a newly different arithmetic route, define its per-operator/state/error
budget from the pinned reference and sensitivity study **before evaluating
the candidate for promotion**; record it in the existing numerical contract.
Until that budget exists and passes, it is unqualified. Do not choose bounds
after seeing the candidate error or relabel a failed exact test as approximate.

For the fastest initial implementation, retain the target's own casts and
operation order and aim for exact genuine-operand replays. This minimizes
the need to justify new approximation budgets. A cuBLAS reduction difference
must still be investigated rather than hidden by the high-level equations.

## 8. Why performance is currently far behind

The actual eager first-request-after-readiness comparison measured median
TTFT13.400s versus1.561s for .8B and14.540s versus1.562s for2B.
The2B individual-token delivery p95 was approximately0.186–0.199s native
versus0.026–0.030s stock. This is a real observed deficit in that bounded
configuration, not evidence of numerical equivalence or every workload.
Separate native warm experiments show substantially lower C1 latency.
Their difference from the cold experiment is not a measured JIT duration:
profile lazy compilation, initialization and allocations explicitly.

Concrete causes/limitations to address:

- `resident_inference.rs` still iterates prompt tokens and awaits a complete
  model execution for each. Cross-request batching does not supply a
  token-matrix prefill or a chunk-parallel GDN algorithm.
- The canonical .8B C1 window launched236,270 native kernels for54 model
  steps. Native launch coverage was verified. Stock recorded9,318 kernel
  activities, but complete/lossless stock coverage was not established;
  do not claim an exact hardware-count ratio.
- Source launch attribution includes41,148 SiLU launches,58,144
  certification/merge launches and20,768 RMS launches. These are launch
  counts, not measured percentages of runtime.
- The latest accepted C8 change cut source dispatch counts by about18% but
  improved observed median raw token rates only about1.7%; C1 barely changed.
  Fewer launches alone are a poor predictor of the remaining speedup.
- Both compared routes were eager. Missing native production CUDA graphs
  cannot explain this particular eager-versus-eager deficit by itself.
- Tiny step projections, reduction kernels, synchronization, allocations and
  state movement are likely important; rank them by actual stage/device
  timing and bytes, not by kernel count alone.

There is no basis for saying all vLLM optimizations are already implemented.
Existing source/state safety and fusion foundations do not guarantee faster
execution. Correctness and a Rust host implementation alone imply no speedup.

## 9. Performance sequence after correctness

The order below targets structural work before local tuning. A bounded,
already-prepared fusion can land earlier if it proves a useful gain without
delaying the correctness or temporal-prefill path.
If measured host gaps dominate, graph admission may proceed alongside
projection work; the priorities are evidence-driven, not a fixed calendar.

| Priority | Change in shared machinery | Why / measurement | Required guard |
|---|---|---|---|
| P0 | Establish accepted numeric-profile baseline, phase timing, device events, bytes, CPU submission gaps and physical counts | Identify prefill/decode bottlenecks and separate GPU work from launch/host waits | Unprofiled paired timings remain the performance result; instrumented outputs match plain outputs |
| P1 | True temporal prefill: token-matrix projections/MLP, causal attention over a token block, chunked GDN, one bounded state transaction | Replace O(prompt tokens) host-driven full model steps; target TTFT first | Same-profile chunk/step parity or explicitly qualified chunk contract; no intermediate token publication |
| P2 | Efficient projection kernels/provider path, appropriate GEMV for decode and tensor-core GEMM for token/batch matrices | Improve actual arithmetic/memory throughput, not merely row count | Source codec/effective activation/output types and compute accumulation fixed; opaque workspace physically admitted |
| P3 | Fuse convolution+SiLU, Q/K normalize+coefficient preparation, gated RMS, residual+norm and SwiGLU where the profile permits | Reduce intermediate writes, launches and global synchronization | Preserve reference casts/reductions, finite checks and meaningful failure semantics |
| P4 | Reuse shape-bound scratch/commands; remove redundant scans/copies by constructing certified outputs and combining checks | Reduce certification, allocator and state-transport overhead | Retain required checks; one original authority, correct unknown-completion quarantine, no lease reuse before completion |
| P5 | Finish production CUDA graph admission and bounded decode graph replay | Reduce host overhead after the kernel structure is stable | Account for graph metadata, captured pointers/generations, all allocations/events and lifetime through retirement |
| P6 | Joint chunked prefill/decode scheduler, ragged refill, bounded overlap and per-tenant aging | Raise SLO goodput while controlling TTFT/ITL tails | Actual independent session/state/event ownership; no starvation or hidden dropped requests |
| P7 | Paged hybrid KV/GDN cache and exact complete-prefix reuse | Lower capacity/copy cost and exploit repeated work | Include recurrent/conv state, source/profile/tenant identity and causal frontier; cold no-hit baseline retained |
| P8 | Source-verified RAM/SSD/GPU streaming for capacity-bound27B | Enable the actual large model and overlap useful transfers | Stable source extents, bounded staging, true Ready events and physical resource accounting |
| P9 | Shape specialization and measured tuning of qualified kernels | Pursue a specific-device advantage after basic throughput is competitive | Offline bounded choices, held-out workload, no unbounded per-token JIT/search |
| P10 | Speculation/prefix combinations only after target execution is competitive | Additional useful committed tokens when acceptance pays for overhead | Exact target distribution/state rollback and full cost of rejected work; target-only comparison |

P1 has three distinct cuts: sequential T2/T4 amortizes gather/scatter,
readback and transaction overhead while repeating the mathematical body;
true temporal stateless lowering combines real teacher rows (including
solo B1/T2) across projections/MLP while respecting causal regions; the
fully qualified chunk attention/GDN path supplies larger temporal parallelism.
Measure each separately. A sequential loop is not token-matrix fusion.

For P2, preserve exact GGUF behavior by first investigating tiled/coalesced
weight loads, shared activation preparation and integer DP4A work within
the existing codec/block/lane reductions. Keep scale addition order and
overflow bounds. BF16 tensor-core GEMM belongs to the independently qualified
vLLM GPU profile; it is not a drop-in replacement for CPU-exact GGUF math.

Do not begin with speculative decoding to hide a slow or incorrect target.
Do not quantize the model/cache more aggressively and call the resulting
quality change a free speed improvement. Do not remove state isolation to
improve a benchmark.

Potential advantages over vLLM are hypotheses: lower host overhead on this
small GPU, efficient small-shape quantized kernels, fused state operations,
and workload-aware hybrid caching/scheduling. Promote only measured wins.

## 10. Two benchmark deliverables

### A. Model execution benchmark

Separate pure prefill and decode after a fixed prefix. Report model/source/
numeric profile, prompt length, teacher-input tokens/s separately from
generated/committed output tokens/s, per-step
latency, physical memory, kernel counts, transferred bytes and device time.
No HTTP/network overhead is attributed to kernels. A pure operator
microbenchmark remains a diagnostic, not the model result.

Preserve canonical47-input/8-output continuity. First admitted GPU matrix:
.8B and2B, B1/2/4/8, prompt lengths1/32/47/128/448 and output lengths8/64,
subject to the actual consumed-token frontier fitting capacity512.
Never run512 prompt plus32/128 generated tokens against that capacity or
silently truncate. After real capacity/numerical qualification, expand to
512/2048/8192 contexts and32/128 outputs within the enlarged envelope.
Cold-first-request and steady warm remain separate.
Add original27B only on a physically admitted, equivalent placement. Report
capacity-enabling streaming separately from resident performance.

### B. OpenAI serving benchmark

Use [serve-bench.py](../../../tools/serve-bench.py) and
[serve-compare.py](../../../tools/serve-compare.py). Preserve their existing
owned-provider lifecycle and alternating ABBA/BAAB sampling.

- Same exact source/quant, effective loader/numeric policy, tokenizer/template,
  raw IDs, generation/EOS/stop/sampling rules and output budget.
- Same device/driver, resource caps, power/thermal conditions and comparable
  residency/cache state; serialized arms on the shared GPU.
- Report startup separately from first request after readiness. Separate
  storage-cache cold, process cold, warmed model and prefix-cache hit/miss.
  One B1 warmup does not warm every B8/JIT/graph shape; declare an additional
  shape-warmed protocol symmetrically when that is what is measured.
- Compare eager routes for diagnosis and actual optimized/default vLLM V1
  routes for the competitive claim. Pin graph, prefix cache, chunked prefill,
  token/sequence budgets and attention/GDN backends.
- Start concurrency1/2/4/8 within current tool bounds. Use fixed open-loop
  arrival traces, short/long mixes and multiple tenants; sweep offered load
  through saturation. Expand concurrency only after admission/tool support.
  Workload labels alone measure class balance; authenticated tenant fairness
  additionally requires the actual tenant identity/admission boundary.
- Report offered/admitted/rejected/canceled/timed-out/completed requests,
  committed correct tokens, p50/p95/p99 TTFT and token ITL, SLO goodput,
  per-tenant completion/latency, memory and complete device counts.
- SSE chunk spacing is not individual-token ITL. When delivery coalesces,
  use authenticated token commit events for engine ITL; report client delivery
  independently and leave unavailable token-level measurements unset.

Current `serve-compare.py` deliberately leaves qualified source-correct
SLO-goodput null and equivalence `NOT_ESTABLISHED`; it also does not estimate
confidence intervals. The minimal follow-on is to bind an independently
accepted numerical qualification and physical commit evidence, then compute
paired statistics from retained raw runs. Merely supplying equal hashes or
matching generated text must not unlock that result.

Before measurement freeze workload/SLO/quality conditions. Retain the existing
10s TTFT/2s delivery-ITL workload as a historical comparator; define practical
interactive and long-prefill SLO classes separately before running them.
Use at least two balanced ABBA/BAAB cycles for screening. For promotion,
predeclare a larger fixed repeat budget, analyze independent run-level paired
ratios and95% confidence intervals, and do not treat correlated token samples
as independent repetitions.

**Win condition:** on a named qualified workload/hardware/configuration,
the95% confidence interval for correct committed SLO-goodput ratio is above1,
with no regression beyond predeclared latency, quality, rejection, completion
or fairness budgets. Report neutral/loss cases alongside wins. No engine-wide
“faster than vLLM” claim follows from one successful prompt or model.
If baseline goodput is zero, a ratio is undefined: use an absolute goodput
difference and its confidence interval, never an infinite-speedup claim.

## 11. Stop conditions and immediate next work

Stop promotion at the first incorrect tensor, incomplete state mapping,
unadmitted resource, unknown physical completion or missing independent
oracle. Continue diagnosis at that operation; do not advance a claim.

When the owner resumes implementation, the first concrete work is:

1. Revalidate paused overlays and accepted artifacts; use the existing bulk
   worktree for the bounded CPU reproduction.
2. Establish actual .8B vLLM projection→convolution inputs and select the
   first complete GDN layer's pinned phase/backend.
3. Port the smallest differing operation with genuine before/after evidence,
   immediately compose it, then walk forward to complete layer/state parity.
4. Expand to complete .8B,2B,27B and the parity matrix.
5. Implement temporal prefill and efficient projections before claiming that
   small launch reductions close the observed throughput gap.

The paused ordered-SiLU candidate remains unaccepted until its model/physical
gates pass. Do not use `target/release/zerogravity` as an accepted baseline
without checking which build produced it.

## 12. References and evidence boundaries

- [Numerical contract](../../docs/NUMERICS_QUALITY_AND_BENCHMARKS.md).
- [Independent llama.cpp runbook](../../../tests/reference/llama_cpp/README.md).
- [Joint serving/goodput contract](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md).
- [36 serving acceptance gates](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md).
- [Accepted batching and performance evidence](../serving-and-speculation/FUSED_GPU_BATCHING.md).
- Existing pinned runtime: vLLM0.31+cu129
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`, Torch2.13+cu129
  `cf30153c4…`, official GGUF plugin
  `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`; retained image
  `sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe`.
  Exact local runtime source is authoritative for this comparison; current
  online documentation is not a substitute for those pinned bytes.
- Pinned vLLM source paths: `model_executor/models/qwen3_5.py`;
  `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`;
  `model_executor/layers/mamba/ops/causal_conv1d.py`;
  `third_party/flash_linear_attention/ops/{fused_recurrent,
  fused_gdn_prefill_post_conv,chunk,chunk_scaled_dot_kkt,solve_tril,wy_fast,
  chunk_delta_h,chunk_o,layernorm_guard,op}.py`.
  Relevant source ranges: packed recurrence317–340; prefill prep78–149;
  gated norm119–176. Preserve Apache-2.0 and vendored FLA MIT notices
  (Songlin Yang, Yu Zhang) for later ports.
- Pinned plugin source: `weights_adapter/qwen3_5.py` and
  `quantization/layout.py`; loaded-role transforms are part of the comparison.
- Original27B revision:
  `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Numerical evidence and current open work are in the owning TODOs linked
  above. Run directories are supporting diagnostics, not additional queues.

This planning pass changes documentation only. It adds no runtime acceptance,
new benchmark result, implemented arithmetic profile or fixed-bug claim.
