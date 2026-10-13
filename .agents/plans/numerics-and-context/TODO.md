# Numerics and context

Provenance commit: `6d74f76`.
Scope: P02/P11/P22, independently of optimization.
Contracts: [numerical/source profile](../../docs/NUMERICS_QUALITY_AND_BENCHMARKS.md),
[independent reference runbook](../../../tests/reference/llama_cpp/README.md).

## Uncommitted

- [ ] Resolve independent original-source CPU 8K GDN numerical divergence without loosening tolerances or changing the declared source arithmetic.
- [ ] Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
- [ ] Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
- [ ] Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
- [ ] Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.

- [ ] Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
- [ ] Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.

## Committed

### 6d74f76

- [x] Original CPU Phase 1 functional 16/16 through genuine 8K; bounded GPU genuine 512-token internal comparisons accepted.
- [x] Neither result closes the open independent numerical or larger-context gates.

### Accepted investigation findings — 6d74f76

- [x] Independent final-token divergence first appears in0.8B layer4 GDN and2B layer1 GDN, after preceding layers match. The separate historical AMD FMA failure is not a demonstrated CPU cause/fix.
- [x] At native ubatch512, shared-top20 logprob differences were0.20372924/0.19678696 (0.8B/2B); matching greedy7993 did not close numerical agreement.2B changes with native scheduling;0.8B did not. These partial-top20/activation observations are not full-vocabulary/KL/total-variation proof.
- [x] A large-position RoPE or padded-state fixture does not qualify a genuine long-context whole model. Canonical future-KV padding and zero signs remain source semantics.
