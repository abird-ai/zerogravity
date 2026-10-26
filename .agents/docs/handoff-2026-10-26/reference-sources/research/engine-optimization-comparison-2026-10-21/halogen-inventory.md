# Halogen per-optimization inventory

Pinned public text: `3f2adcfca6be623c8ec4f69331cbf97beff55251` / `v0.18.13`,
2026-10-21. This table inventories public substantive mechanisms; it does not
claim internal kernels were inspected or any foreign runtime was executed.

Evidence: public documents are claims unless public executable source below was read.
`U` = core implementation uninspectable; `B` = existing bounded local implementation
and recorded qualification; `I` = isolated unpublished work; `P` = open plan;
`A` = applicable implementation absent; `NA` = inapplicable model/hardware now.
Every external engine math/state mechanism is `U` even when its public claim is
specific. Z1-Z10 and G0-G5 are defined in [REPORT.md](halogen.md). `B` never means
full vLLM parity, arbitrary context, broad backend or production graph qualification.

| ID | Public mechanism | Primary source | Zerogravity disposition | Composable stage, priority and gate |
|---|---|---|---|---|
| H01 | gfx1151/model-family specialization; no portable fallback | [H][H] [LIMIT][LIMIT] | U; P Z1/Z2 | Capability-selected lowering in existing plan; defer Strix physical work; never hard-code one engine. |
| H02 | Trained GDN, QSA/indexer, routed MoE and residual kernels | [C][C] | B/I Z1/Z2; A Flash-Next | Qwen3.5 GDN shares only semantic subregions; QSA/512-expert source contracts remain separate. |
| H03 | QSA scores 4-token blocks; default selects512/2048 tokens | [CFG][CFG] | A Z2 | Trained sparse selection belongs to source semantics; exact masks/ties/order before optimization. |
| H04 | Distributed register-bisection selection replaces serial workgroup scan | [C][C] | A Z2; P Z10 | Long-context candidate after27B; preserve complete block lists including ties and order. |
| H05 | Indexer scoring retains128 block keys/workgroup and reuses them | [C][C] | A Z2; P Z10 | Typed tiled region; same product/reduction order; bandwidth/occupancy and full-state gates. |
| H06 | Stable counted expert grouping replaces CPU comparison sort | [C][C] | A Z2; P Z10 | Finite-key exact stable grouping is reusable; public account still includes CPU grouping. |
| H07 | Routed-expert prefill kernel revision changes rounding | [C][C] | A Flash-Next; B dense Z2 | New arithmetic profile unless exact oracle passes; kernel speed cannot waive source schedule. |
| H08 | Deterministic per-shape GEMM library plan | [F][F] [PERF][PERF] | B prepared plans Z1; P tuning Z10 | G1 tune B1/decode/temporal/verify separately; freeze compiler/device/source/profile and selected algorithm. |
| H09 | Decode/prefill/verify use different documented row regimes | [TOOLS][TOOLS] [F][F] | B B2/4/8; I native profile Z1/Z2 | G1 finite shape specialization; do not equate request batch with temporal prefill or verify. |
| H10 | Fusion, HIP graphs, WMMA/MFMA, LDS layouts: unpublished | [F][F] | U external; B fusion/private graph Z1 | No claim Halogen uses graphs or a specific ISA; qualify our own candidates under G0-G5. |
| H11 | v2 mixed4/6/8-bit checkpoint; FP8 lookup separate | [Q][Q] [P][P] | A .hgn; B GGUF quant Z3 | Derived source+calibration identity separate from source semantics; no implicit requantization. |
| H12 | ht43 reduces resident weights; can slow prefill/decode | [P][P] | A ht43; P derived quant Z10 | Later explicit precision option; measure memory and quality plus goodput, never fewer bits = faster. |
| H13 | w4b calibrated overlay promotes sensitive output projections | [Q][Q] | A overlay; P quality profiling Z10 | Study per-family error attribution; independently derive permitted source packs, preserve base identity. |
| H14 | Higher-precision draft projections improve proposal acceptance | [HF][HF] | A MTP Z8 | Draft pack may differ if target remains authoritative; source-bound head, quality/ROI gates. |
| H15 | GGUF quant values repacked without requantization | [G][G] | B Q4_0/Q4_1/Q8_0/Q6_K Z3; A Halogen formats | G1 lossless layouts where useful; exact dequant/code/scale/source proof before changed kernels. |
| H16 | Unsupported GGUF block types refuse at startup | [G][G] | B explicit QuantKind/refusals Z3 | Declare exact tensor-role codec matrix; no silent lossy fallback or name-based compatibility claim. |
| H17 | Some affine dense/ssm_out tensors stage as BF16 | [G][G] | A Halogen path; P lowering Z10 | A load fallback can increase per-token bytes; cost/profile explicit, never assume packed file stays packed. |
| H18 | ssm_out128-column reorder conflicts with256-wide scale groups | [G][G] | A Halogen layout; I GDN typed layout Z2 | Whole source grouping/reduction must survive repack; this foreign layout is not a required ZG ABI. |
| H19 | Parallel GGUF repack; optional persisted repack | [G][G] | A persistent derived-pack cache; B parsing Z3 | G2 cache by full source+pack/compiler/device schema; size/mtime alone cannot establish immutable source. |
| H20 | Native checkpoint verify/inspect includes geometry, checksums, scales | [TOOLS][TOOLS] | B source/index/hash Z3/Z4 | Reuse existing bounded verifier; no .hgn parser or engine executable adoption needed. |
| H21 | GPU registered host weights; table remains file-backed | [HOST][HOST] | P complete UMA accounting Z4 | One arbiter must charge overlapping physical backing once, plus OS/cgroup/pinning headroom. |
| H22 | Optional mlock avoids weight-page reclaim cycles | [HOST][HOST] | P UMA backing Z4 | Deferred hardware policy; distinguish GPU registration, mlock, page pinning, cache and actual OS release. |
| H23 | w4b unpinned mode copies experts and is much slower | [HOST][HOST] | A dynamic expert executor; P Z4 | Negative control for placement; not evidence of efficient expert LRU/SSD pooling. |
| H24 | Pool/scratch auto-fit, CMA/GTT/host-fragmentation diagnostics | [CFG][CFG] [ENTRY][ENTRY] | B arbiter; P OS/UMA integration Z4 | G2 physical host plus device admission; driver counter is not immediately available RAM. |
| H25 | Optional persistent unpacked trunk costs5.5GiB | [F][F] | P derived-pack residency Z3/Z4 | Choose repack vs memory per workload; lower-memory SM86 target must charge persistent form and peak staging. |
| H26 | Fixed startup slots(up to64); wait for pool room | [LIMIT][LIMIT] | B active<=8 HTTP Z5 | G2 independent roots+bounded overload; generalize fair worker without importing fixed slot architecture. |
| H27 | Shared batched target step; plain batching beyond4 | [LIMIT][LIMIT] [CFG][CFG] | B GGUF B2/4/8 Z1/Z5; P profile integration | G1/G2 joint batch planning after accepted profile; no claim trained vLLM profile batch parity yet. |
| H28 | Chunked prefill alternates other streams decode ticks | [CFG][CFG] | B one-teacher prefix waves; P temporal Z5 | G1/G2 true multi-token temporal producer, then fair chunks; source-fixed chunk schedule. |
| H29 | 0.18.13 splices decode during long prefill | [C][C] | P temporal/fair overlap Z5 | G2 legal layer/region yield; preserve owned state and actual event completion, qualify tails and TTFT tradeoff. |
| H30 | 0.18.13 nests<=1024-new-token request at layer boundary | [C][C] [F][F] | P fair temporal scheduling Z5 | G2 bounded extra workspace admitted first; nested work must never alias long request mutable state. |
| H31 | Interactive-first queue ages agent turns after30seconds | [CFG][CFG] | B worker fairness; P HTTP class policy Z5 | G2 fair progress+aging+deadline gates; client classification cannot cause starvation. |
| H32 | No paging or preemption; throughput flattens past8 | [LIMIT][LIMIT] | P paged hybrid allocator Z4/Z5 | Our paged state is independent work; private graph/batching cannot be relabeled PagedAttention. |
| H33 | Full hybrid prompt snapshots; strict cache1 vs faster cache2 | [QUAL][QUAL] [CFG][CFG] | B checkpoint foundations; P full-prefix reuse Z6 | G2 require cache-disabled full-logit/all-domain continuation identity for default exact profile. |
| H34 | Attention stays in place; small position-free snapshot | [CFG][CFG] | B immutable roots/checkpoints; P sharing Z6 | G2 typed hybrid checkpoint+KV frontier; cross-session reuse needs source/profile/tenant authorization. |
| H35 | Pool relocation/packing/clamping rather than paging | [CFG][CFG] | P hybrid page allocator/COW Z4/Z6 | Exact concurrent readers/generations; capacity clamp must stay visible and cannot bypass branch ownership. |
| H36 | Incremental direct-I/O disk prompt-cache restore | [CFG][CFG] | B portable diagnostic; P durable product Z6 | G2 corruption/atomicity/compatibility/admission tests; exact restore means same saved bytes, not cold equivalence. |
| H37 | Cache eviction gives long conversations length-based credit | [C][C] | P complete-prefix manager Z6 | G2 workload-aware policy only after exact cache; measure avoided recompute, fairness and physical cost. |
| H38 | Common project-leading prefix and branch-aware resume points | [C][C] | P full-prefix manager Z6 | G2 authorized common text reuse; greatest jointly-restorable hybrid frontier, independent state branches. |
| H39 | Keep/fork/join handles and optional thread mapping | [SESS][SESS] | B roots/checkpoint; P serving handles Z6 | G2 preserve tenant/app/session grants; global shared-name design cannot authorize other users in ZG. |
| H40 | Composable-context preview reuses moved message work approximately | [CFG][CFG] | A approximation; B text snapshot API Z7 | Deferred explicit numerical option. It is unrelated to ONE shared execution fabric/context owner. |
| H41 | Opt-in KV8 lowers KV memory and changes numerical results | [F][F] | A KV8 profile; P quant policy Z10 | Deferred explicit state precision; quality/reference/rollback/cache identity needed. No default adoption. |
| H42 | Opt-in static YaRN extends native262144 toward1M | [CFG][CFG] | P longer context Z2/Z10 | Source-approved alternate model configuration, not a cache trick; preserve short-context regression and long oracles. |
| H43 | Raised QSA budget trades retrieval, perplexity and speed | [CFG][CFG] | A QSA; P source contract Z2 | Explicit model semantics; never lower trained budget for speed or present increased budget as source-exact. |
| H44 | Native trained MTP layer proposes; full target verifies | [HF][HF] [QUAL][QUAL] | A trained MTP; P Z8 | G3 after Qwen basics+benchmarks; target/draft state ownership and every acceptance-prefix rollback. |
| H45 | Prompt lookup proposes request-text continuations, greedy/solo | [F][F] | A runtime lookup; P Z8 | G3 generic no-new-model drafter; separate learned lookup bank; target-authoritative commit and adaptive ROI. |
| H46 | Greedy confidence gate and acceptance-window backoff | [F][F] | A adaptive spec; P Z8 | G3 finite plan/depth budget; rejected work and draft overhead included in complete-request goodput. |
| H47 | Sampled variable-depth draft; exact target distribution claimed | [SAMP][SAMP] [C][C] | A sampled speculative runtime; P Z8 | G3 verify full correction distribution/RNG/grammar ownership; seed equality is a narrower separate guarantee. |
| H48 | End-of-turn terminates draft chain early | [C][C] | A speculative runtime; P Z8 | G3 prevent useless final verification; terminal/grammar/stop/RNG cases, including one-token replies. |
| H49 | GGUF drafter scores shortlist for greedy proposals | [C][C] | A MTP shortlist; P Z8 | G3 shortlist only proposal path; target full vocabulary remains authority; validate discarded target candidates. |
| H50 | Schema masks on GPU; grammar-forced text drafts up to8rows | [F][F] [C][C] | A grammar/spec execution; P Z8 | G3 schema automaton/candidate masks+RNG rolled back with hybrid state; no target vocabulary truncation. |
| H51 | Joint2/3/4-stream MTP verification with cost fallback | [C][C] [PERF][PERF] | B target batches; A joint spec Z1/Z8 | G3 compose fair batch+branch admission; output per root matches solo; adapt from measured total useful work. |
| H52 | Learned FP8 n-gram bank stays paged on disk | [Q][Q] [HF][HF] | A learned lookup runtime; P SSD Z4 | Later Flash-Next, not dense27B; bounded immutable row leases and real asynchronous copy readiness. |
| H53 | Concurrent bounded row gather avoids serial SSD waits | [F][F] | B extents/windows only; A gather executor Z4 | G2 for actual27B window pipeline, later sparse bank; coalesce reads, preserve hashes and cancel ownership. |
| H54 | Lookahead of decode lookup rows; additional launch reduction claimed | [C][C] | A lookup executor; U exact method | Later semantic cost stage; no inference of persistent kernels/graph/async IO from prose. |
| H55 | Incremental detokenization replaces repeated full-answer decode | [C][C] | B tokenizer; P output parity/perf Z9 | G1 measure actual native output cost first; Unicode/special-token/tool boundaries and tokenizer identity. |
| H56 | Disconnect cancellation plus retry grace retains paused state | [LIMIT][LIMIT] | B bounded cancellation; P expanded lifecycle Z5/Z6 | G2 do not conflate disconnect with completed work; paused reservations count, unknown completion quarantines. |
| H57 | Public benchmark uses named prompts, hashes, timings and cold flags | [B1][B1] [B2][B2] | B control harness; P full comparison Z9 | G1 real output IDs/full-state accepted first; record true per-token ITL, SLO-goodput, source/cold/warm and lossless counts. |
| H58 | Quality tooling: paired PPL, bounded-support KL, retrieval | [TOOLS][TOOLS] [QUAL][QUAL] | B source/full-state tests; P broad quality Z9/Z10 | G1 retain exact operator/state gates plus functional quality; support-limited KL and short retrieval are not universal fidelity. |
| H59 | NPU side models behind shared HTTP with separate binary/programs | [NPU][NPU] [NOTICE][NOTICE] | A NPU runtime; B pure pipeline/context Z7 | Deferred after Qwen/Gemma; ONE engine typed capability stages, not foreign sidecars or second allocator. |
| H60 | NPU cost-aware8x512 buckets and serial bounded pass queue | [NPU][NPU] | A NPU; B worker foundation Z5/Z7 | Deferred heterogeneous policy; amortization threshold, load mix, fair latency and shared-memory pressure gates. |
| H61 | NPU source conversion cache, prompt decision resume and idle-part release | [NPU][NPU] | A NPU; P residency/context Z4/Z7 | Deferred source-bound pack cache, role-specific complete state; per-model idle/unload must retain real readers. |
| H62 | NPU/GPU requires IOMMU+fabric-clock qualification | [NPU][NPU] [FAB][FAB] | A qualified NPU hardware; P heterogeneous Z7 | Deferred, no device/privilege/recovery mutation; validate power/bandwidth contention and physical failures first. |
| H63 | NPU command faults return503 and restart side worker | [C][C] | A NPU; B quarantine Z4/Z5 | Do not transplant restart as proof safe reuse; original device completion/quarantine contract still governs. |
| H64 | Vision tower only for uncached images; pixel-cost bound | [IM][IM] [C][C] | A trained vision; P model/media Z7 | Deferred; source-bound media preprocessing/positions/all-state cache identity, cancel and physical residency. |
| H65 | Runtime counters expose cache, queue, draft-source and driver stalls | [API][API] [C][C] | B native counts; P unified serving diagnostics Z9 | G1/G2 accountable accepted/rejected/canceled work, full physical dispatch/copy counts, all-source coverage. |

## Pinned primary sources

[H]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/README.md
[C]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/CHANGELOG.md
[F]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/FLAGS.md
[Q]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/QUANT.md
[P]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/precision.md
[G]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/gguf.md
[CFG]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/configuration.md
[PERF]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/performance.md
[QUAL]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/quality.md
[TOOLS]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/checkpoint-tools.md
[HOST]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/host-setup.md
[NPU]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/NPU.md
[LIMIT]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/limits.md
[SESS]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/sessions.md
[SAMP]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/sampling.md
[API]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/api.md
[ENTRY]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/deploy/entrypoint.sh
[FAB]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/deploy/host/halogen-fabric-clock
[B1]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/tools/bench-serving.py
[B2]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/tools/halogen-bench.py
[IM]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/images.md
[TOK]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/token-budgets.md
[TROUBLE]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/troubleshooting.md
[LIC]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/LICENSE.md
[NOTICE]: https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/THIRD-PARTY-NOTICES.md
[HF]: https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next/blob/ff0830694ec8461edd827d7300134e346bf03eb1/README.md
