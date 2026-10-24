# E2B learned text diagnostic

- Source: `google/gemma-4-E2B-it` @ `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- Acquired managed snapshot: `ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1`.
- Eight original members, whole checkpoint SHA/length, config and tokenizer pins are compiled in `source-pin.json`.
- `derived-model.safetensors.index.json` is the authenticated **unsharded header-derived closure** over all2011 original tensors. It is not an official HF sidecar; never write it into the managed snapshot. Original source contains600 exact text tensors; 35 blocks use15 independent KV owners.

The original reference runs the existing immutable approved offline image with original `Gemma4ForConditionalGeneration`, eager attention, CPU BF16/AVX2, Torch2.13.0+cu129 and Transformers5.17.0. Image ID/digest fixes the package; actual installed Gemma/cache source hashes and tokenizer class are recorded. No source patch, remote code, network, image pull or download. Full conditional checkpoint loading may honestly refuse the12GiB containment; no increase/fallback is implied.

Actual source tokenizer turns raw text `One plus one is` into original IDs (no chat-template claim). One complete prompt invocation followed by up to two naturally computed greedy continuations, stopping at original EOS. Two independent fresh-cache repetitions must have identical whole logits/owner state/feedback. Captures every full262144-way selected-last-row vocabulary and all15 original owner K/V caches; shared consumers15..34 are not extra owners. The independent archive validator checks BF16 widening, full-V greedy (numeric signed-zero ties), exact record/EOF/shape/source joins and byte-identical consumed-prior continuity.

Rust uses the existing `SourceWeights::prepare_verified_windows` → `OriginalGemma4Text::bind` → same `ReferenceTextMath` schedule and `SessionAuthority`. One1MiB source scan before issuance; subsequent authenticated windows; one final whole held-source pass. Actual original prompt IDs are input tokens. Original logits/KV are borrowed only inside the completed-result callback. Continuation operands come from **Rust's actual full-V argmax**. Callback failure or cancellation after callback refuses original publication; no expected token substitutes for computed feedback.

`AnalyticF32TextV1` remains a distinct unqualified diagnostic. Exact expected-word comparisons fail on every bit difference, including output and KV; a failure is retained with current frontier and causes nonzero exit. BF16 casts/scaled embedding, GEMM/BMM reduction, RMS/softmax/GELU/libm rotary and activation rounding have not been trained-qualified. No learned tokenizer/runtime/native/serving/performance acceptance follows from acquisition, source construction or synthetic fixtures.

## One coordinated untimed batch

The initial20-path CPU batch passed, then actual healthy original reference3 completed the full three-call repeated trajectory within the unchanged12GiB bound. The first strict analytic-F32 Rust replay computed all35 blocks/fullV but failed at logits[0]; root stayed unchanged and final reservations returned0. Failed original attempts1/2 are preserved separately. Neither result trains or qualifies Rust.

The BF16 successor uses the same engine and token traversal. Its constructor is explicit `Gemma4ReferenceSession::new_bf16_candidate`; default and recorded analytic routes remain unchanged. BF16 values/KV have a separate arithmetic seal. GEMM/BMM reduction and pow/GELU/rotary/tanh bits remain unqualified, especially true initialM4 vs sequentialB1 projection. CPU/schema fixtures do not settle those seams.

Optional `--producers` now records borrowed actual initial rows0/3 and every naturally completed cached call through the same35 blocks: scaled embedding/PLE; projections, direct-gamma RMS, rotary, masked QK/softmax/PV, FFN/GELU, PLE and final norm/head/logits. Original wrappers return the same tensors. Per-call context is the actual input rows and pre-call frontier; hooks are deactivated between calls and removed before the independent fresh-cache repetition. No continuation is forced after EOS. Existing fullV/all15 KV/feedback/repetition/source validation still applies.

New producer schema v2 carries explicit current-root call contexts. Old healthy v1 initial-only references remain readable and immutable; they never become cached producer authority. With the genuine4-row prompt and two continuations, the maximum is3965 records/22598144BF16 bytes under the original4096-record/32MiB file bounds. Raw records stream; compact metadata fits1MiB under the existing2MiB reader ceiling. Original12GiB/4CPU and Rust128MiB (including8MiB diagnostic control) remain unchanged.

The Rust example accepts `--arithmetic bf16-candidate` (otherwise original analytic F32). Expected words only inspect borrowed actual results. Numeric differences are counted across every word of every available selected producer, fullV and all15 complete KV pairs in the completed call before refusing publication. Reports retain a count and first word per differing record, with original-record indices for producer summaries; they do not retain whole activation copies. Structural/source/nonfinite/schema failures remain fatal. Failed completion does not publish, choose a new token or replace an operand. All trained/native/serving/performance predicates remain false.

Actual earlier producer reference2 has detailed producers only for initial rows0/3. The reviewed GELU+RMS fixes closed all1981 initial selected producer records and the first cached fullV/KV comparison, but the second cached call remains numerically unqualified. Its old logits-first comparison did not reach cached KV. The new all-call diagnostic source awaits one coordinated fresh original capture and strict replay; no new result is implied by the added tracing.

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test -p zerogravity-model-gemma4 --features text-reference --lib --examples
python3 -m unittest discover -s tools/gemma4-e2b-reference -p 'test_*.py'
```

ROOT launches the original runner directly: rootless Podman supplies its own admitted user/network containment. An outer `native-gpu.py` network namespace would prevent the existing rootless overlay mount. The tool imports no model until its explicit isolated capture entry runs; original helper cleanup retains unreaped group identity until TERM/KILL, then reap/absence/EOF.

```bash
python3 tools/gemma4-e2b-reference/run_reference.py \
  --source /home/pvl/.cache/abird/zerogravity/models/snapshots/ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1/files \
  --new-output /ABS/FRESH-ORIGINAL-ATTEMPT
# Only after the healthy whole original finishes, bind its exact invocation SHA:
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo run --release -p zerogravity-model-gemma4 --features text-reference \
  --example gemma4_e2b_text -- \
  --source /home/pvl/.cache/abird/zerogravity/models/snapshots/ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1/files \
  --reference /ABS/FRESH-ORIGINAL-ATTEMPT/invocation.json \
  --reference-sha256 ACTUAL_WHOLE_INVOCATION_SHA --report /ABS/FRESH-RUST-REPORT.json
```

The Rust example has one finite128MiB host admission and8MiB caller control, independently of10.25GB checkpoint size. It streams W rather than copying the checkpoint. Runtime sources/activation/state windows remain charged by their existing factories; a real admission refusal is fatal. Source64MiB metadata ceiling,2MiB model metadata, rank1 weights,66-row15-domain state,1MiB full-V/record buffers and bounded activations are independent of whole W. This is a declared diagnostic resource budget, not measured host peak or production fit. Source/report live outside each other; no managed cache writes. End-of-run ledger must return0 after source/state/control drop.

Primary reference:
[HF Gemma4 implementation v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/gemma4/modeling_gemma4.py),
[HF cache implementation v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py).
The older architecture review pin683b7219 is not asserted byte-identical to this independent framework target.
