# Numerics and context

Provenance commit: `6d74f76`.
Scope: P02/P11/P22, independently of optimization.
Contracts: [numerical/source profile](../../docs/NUMERICS_QUALITY_AND_BENCHMARKS.md),
[independent reference runbook](../../../tests/reference/llama_cpp/README.md).

## Uncommitted

- [ ] **Required three-model correctness gate (owner,20 Oct):** resolve
  independent original-source CPU8K GDN divergence for BOTH original
  Qwen3.5 models without loosening tolerances or changing declared source
  arithmetic. This is required alongside original Qwen3.8-27B correctness,
  matched serving/model benchmarks and shared-engine performance work.
  Short-context CPU/B1/native exactness and closed original-HF operator
  proofs do not close this independent GGUF long-context issue. Recover
  exact raw teacher prefixes and pinned native arithmetic/chunk settings,
  locate the first state/stage discrepancy, then rerun independent long
  context and the accepted short corpus after any implementation fix.
- [ ] Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
- [ ] Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
- [ ] Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
- [ ] Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.

- [x] **Fresh0.8B whole-runtime reference control:** original8192-ID corpus,
  CPU8/F32 KV/FA off, batch/ubatch64 and public CPU use_ref control pass all730
  selected records/full248320 logits and all42 active causal domains,
  55,382,016 state words, zero bit differences including signed zero versus
  accepted Rust. Actual630 public CPU calls bind630 inner plans with use_ref
  true and exact TLS ownership; two BLAS lifecycle contexts execute no math.
  Original source/build/provider/observation and owned cleanup guards pass.
  This qualifies the explicit reference-control schedule, not default bulk
  arithmetic, historical corpus recovery or production arithmetic changes.
- [ ] **2B whole-runtime control continuation:** the same V6 control correctly
  refuses actual BLAS compute before delegation (exit125, no accepted capture).
  Source/build/provider and owned cleanup binding pass. Preserve this failure;
  investigate genuine public scheduling controls without bypassing compute
  guards or substituting unobserved arithmetic. Earlier AR1 all8K exactness
  remains accepted; required two-model long-context closure stays open.

- [ ] Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
- [ ] Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.

- [x] **Full-runtime reference-control prerequisite:** actual public metadata
  identifies the rejected constructor as pinned BLAS (ACCEL), before CPU
  initialization. The diagnostic keeps its original pre-setter exit125 and
  produces no numerical capture. Strict build, exact source/build/invocation
  binding and owned process/group/EOF cleanup pass. This diagnoses a control
  harness refusal; it does not explain or close the long-context arithmetic
  divergence. Guarded public compute-route coverage remains pending; no BLAS
  disable, source arithmetic or tolerance change.

## Committed

### bfe62df — independent source and reference-path qualification

- [x] Fresh mixed/Unicode teacher prefix: both original GGUF models at
  512/1024/2048/4096, pinned native AR1/F32 KV/FA off versus Rust: all730
  selected actual stage records/full248320 logits and independently decoded
  all42 active causal state domains bit-exact. Each4K state comparison covers
  30,216,192 words. Fresh native8K captures/55,382,016-word state exports are
  complete. BOTH fresh8K comparisons now pass all730/full248320 and all42
  active causal domains,55,382,016 words/model with zero differences;
  Rust6K captures alone are not independent6K comparisons. Historical8K raw
  corpus was not recovered; this fresh corpus does not establish its replay.
- [x] Unchanged0.8B first64 bulk-versus-AR1 capture narrows the first observed
  result gap to layer3 attention AV, position2 (530/2048 words), before the
  previously observed layer4 GDN gap. All131072 probability operands and
  1,064,960 causal V comparisons are exact;114432 future probabilities are
  positive zero. Earlier captured layers0–2/GDN states match. Plain/observed
  whole state and next AR1 full-vocabulary guards prove capture transparency.
- [x] Genuine public ggml AV replay:65 defaults/262144 words match their own
  captures. Public reference-control bulk differs57823 words from default
  bulk and matches all64 default AR1 outputs exactly. Transparent exported
  entry observation preserves all130 default/control outputs (524288 words)
  in both runs and records576 real calls:bulk64 takes64 true returns,
  AR1 takes512 false returns, reference control takes none. Exact opaque ABI,
  pinned DSO origin and bounded owned cleanup pass independent review.
  Internal compiled kernel/full-runtime cause is unqualified; no production
  arithmetic, source profile or tolerance changed.
- [x] Unchanged2B first64 bulk/AR1 capture: plain/observed whole post-state
  and next AR1 full248320/whole-state guards pass. All28 original physical
  learned operands remain unchanged after both modes; eight opaque CPU_REPACK
  payloads are transport-only, with no codec equivalence claim. The first
  observed result gap is layer0 F32 alpha projection, position0 (13/16 words).
  All960 preceding compared rows and131072 actual alpha-input words match;
  its full F32[2048,16] weight equals the original131072 GGUF bytes. Beta
  projection also differs before GDN. No production fix or tolerance change
  is inferred from this capture.
- [x] Genuine2B public alpha replay:65 defaults/2048 output words match their
  own original captures. Public reference-control bulk matches all64 default
  AR1 outputs exactly; reference AR1 outputs are unchanged. Transparent
  exported-entry observation preserves all130 default/control outputs,
 4096 words/run;520 calls comprise8 bulk true,512 AR1 false and zero control
  calls. Original inputs/full raw F32 weights, source/DSO identity and bounded
  owned cleanup pass. This narrows native batch-dependent arithmetic before
  GDN; internal compiled route/full-runtime cause and historical issue closure
  remain open. Rust/source arithmetic and numerical limits are unchanged.

- [x] Actual compiled-route observation for BOTH unchanged public replays:
  all130 default/control outputs, operand checks, exported-entry traces and
  complete before/after mapped-file sets/hashes remain exact. Each dataset
  records132 first named entries across392 stops. Default bulk enters the
  pinned tinyBLAS F32 outer/full/tail families; the other129 scopes enter
  ggml_vec_dot_f32. Static source/code pins identify one Vec8 partial/output
  versus four Vec8 partials/reduction, respectively. Only first named entries
  were observed, not every hit/instruction or private numerical arguments.
  Five actual owned-SID/pidfd/subreaper failure fixtures and actual debugger/
  inferior/group/EOF cleanup pass. First default-scheduler guard failure is
  preserved; additive explicit scheduler-locking off retains original math.
  This strengthens the fresh first64 batch-dependent arithmetic explanation;
  it does not recover the lost historical corpus, qualify full host-runtime
  closure or change production source math/tolerances.


### 6d74f76

- [x] Original CPU Phase 1 functional 16/16 through genuine 8K; bounded GPU genuine 512-token internal comparisons accepted.
- [x] Neither result closes the open independent numerical or larger-context gates.

### Accepted investigation findings — 6d74f76

- [x] Independent final-token divergence first appears in0.8B layer4 GDN and2B layer1 GDN, after preceding layers match. The separate historical AMD FMA failure is not a demonstrated CPU cause/fix.
- [x] At native ubatch512, shared-top20 logprob differences were0.20372924/0.19678696 (0.8B/2B); matching greedy7993 did not close numerical agreement.2B changes with native scheduling;0.8B did not. These partial-top20/activation observations are not full-vocabulary/KL/total-variation proof.
- [x] A large-position RoPE or padded-state fixture does not qualify a genuine long-context whole model. Canonical future-KV padding and zero signs remain source semantics.
