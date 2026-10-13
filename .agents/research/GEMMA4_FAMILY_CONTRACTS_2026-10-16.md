# Gemma 4 family source contracts — 2026-10-16

Technical reference: configuration, tensor-shape and KV ownership contracts for E2B and26B-A4B. It does not establish trained inference, artifact loading or tokenization.

The CPU-free `zerogravity-model-gemma4` family library exposes these metadata
contracts through the existing shared source/state/resource boundaries.

## Authoritative sources

Official metadata/source revisions below define the current technical profile.
Completion/results and pending execution work live in [model-execution/TODO.md](../plans/model-execution/TODO.md).

- [Official E2B-it configuration](https://huggingface.co/google/gemma-4-E2B-it/blob/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json), revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- [Official 26B-A4B-it configuration](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/config.json), revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`; base configuration at `24548b62aa021d562695c04aaf7758a1ea47990b` parses to the same configuration value.
- [Gemma 4 reference implementation](https://github.com/google-deepmind/gemma/tree/9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3/gemma/gm/nn/gemma4), revision `9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3`.
- Framework Gemma 4 implementation, revision `683b721971417b3b2c64fba9001f525607547e5a`.
- [llama.cpp Gemma 4 metadata mapping](https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/models/gemma4.cpp), revision `b6334e37e6ef286fc5091a41b6561b2113d915fb`; consulted as an additional layout reference, without adopting its runtime.

The optional E2B weight index returned404; it is a single-file observation,
not independent qualification of headers or the complete learned-tensor closure.

## Implemented API and invariants

`Gemma4Config::from_json` accepts bounded official outer/text configurations and stores checked geometry privately. Read-only accessors expose variant, per-layer attention/head/rotary geometry, vocabulary, scalar policies, per-layer input dimensions, EOS and MoE geometry. Unsupported activation, schedule, sharing, rotary and scalar overrides return typed errors before deriving storage contracts.

`tensor_specs()` describes original HF conditional-model tensor names and row-major shapes. `validate_tensor_shapes()` checks required declared metadata and rejects missing/transposed shapes. It does not authenticate bytes, codecs, checkpoint identity or decoded values. Optional tied heads and source-omitted shared/K=V projections remain explicit.

`kv_state_owners()` returns existing core KV-domain descriptions. E2B has 15 owning layers and 20 consumers reusing the last compatible local/global producers, layers 13/14; local/global head widths differ. E2B also retains its packed per-layer embeddings and doubled later FFN width. The 26B variant has distinct full-attention KV widths, learned routing/expert axes, K=V projection semantics and extra MoE branch normalization weights.

`MoeGeometry` and `variant_26b::validate_config` provide checked metadata only. The contract library has no production routing arithmetic or CPU executor.
Five test-local analytic routing cases do not establish framework source parity.

## Remaining work and qualification limits

The shared plan/backend needs source-defined GELU, elementwise multiply/scale, final tanh soft-capping, sliding attention and shared KV, Gemma proportional RoPE and value RMS normalization. Implement these as shared operators with immutable source binding and existing session/physical ownership. Do not express Gemma as Qwen, add a duplicate numerical runtime or substitute CPU output for an unsupported GPU region.

The current source-storage API still has its existing 4 GiB owned-byte cap. The 26B artifact and E2B's large per-layer embedding storage require a separately designed cold-storage/admission seam; this unit does not raise the cap. Metadata-only tensor sizes do not allocate model payloads.

Model acquisition, exact tokenizer/template behavior, checkpoint-codec mapping, source-bit operator qualification, CPU/GPU forward plans, real trained multi-token state, cancellation, rollback, full-model logits and context gates remain pending. Framework reduction and per-expert scaling schedules differ, so a later routing implementation must choose and independently qualify its original-source arithmetic. Neither fixture tests nor this contract library establishes runnable Gemma support or performance.
