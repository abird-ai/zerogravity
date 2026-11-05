# Independent CPU reference probes

This directory holds **test-only C++ clients** of a separately installed and
version-pinned llama.cpp/ggml library. They serve solely as independent
numerical oracles for Zerogravity's Rust CPU implementation. They are **not**
dependencies of Zerogravity's production library, binary, or GPU backends.

## Probes

- **llama_reference_logits.cpp** emits unfiltered top-20 logits and
  full-vocabulary-normalized log-probabilities **at every raw teacher token**,
  avoiding the llama-server HTTP response's merged UTF-8 fragments.
- **llama_reference_layers.cpp** captures independently evaluated trained GGUF
  embedding, attention, FFN, and final-norm tensors via llama.cpp's evaluation
  callback. For the unnamed post-FFN residual, qualification reconstructs the
  source result with exactly one IEEE-754 f32 addition of two independently
  captured tensors. This reconstruction is explicitly recorded in evidence.

Reference settings: CPU only, F32 KV cache, Flash Attention disabled.

## P07-H9 independent RoPE capture

`llama_reference_rope.cpp` is another test-only client, not a runtime dependency.
It captures the first full-attention block's complete Q/K RoPE outputs at four
genuine causal positions by decoding raw IDs 11,353,17,42 individually. Signed
zeros are serialized explicitly. The comparison requires all eight
unique records, exact dimensions, a matching GGUF SHA and every FP32 bit; it
allows no signed-zero exception. Both original checkpoints pass 10,240 values.

The qualification used the existing pinned llama.cpp d812350 library/headers
and its separate glibc closure, without downloading or rebuilding the library.
From the repository root, with those existing paths:

```sh
c++ -std=c++17 -O2 -I target/qualification/reference-llama/include \
  tests/reference/llama_cpp/llama_reference_rope.cpp \
  -L /nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib \
  -L /nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib \
  -Wl,-rpath,/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib:/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib \
  -Wl,--dynamic-linker,/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib/ld-linux-x86-64.so.2 \
  -lllama -lggml -lggml-base -lm \
  -o target/qualification/reference-llama/rope_capture
target/qualification/reference-llama/rope_capture \
  target/models/qwen35-0.8b/model.gguf target/h9-independent-rope-qwen35-0.8b.jsonl
cargo build --locked --offline --release --example trace_tokens --features diagnostic-gdn-trace
ZG_TRACE_LAYER=3 RAYON_NUM_THREADS=8 target/release/examples/trace_tokens \
  target/models/qwen35-0.8b/model.gguf target/models/qwen35-0.8b/config.json \
  target/h9-rust-rope-qwen35-0.8b.jsonl 11 353 17 42
```

Repeat with qwen35-2b and its original SHA
`91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6`.
These Nix paths describe the qualified local closure; other machines need their
own matching pinned headers/library and loader, not an unpinned replacement.
The [accepted baseline](../../../.agents/plans/accepted-baseline/TODO.md)
records GPU qualification and its limits.

## P07-H10 native QK dots and independently captured scale

`llama_reference_qk.cpp` captures all six full-attention layers at four genuine
causal tokens. It selects valid positions using actual tensor strides, preserving
zero signs. Raw dots are native source tensor values. Scaled scores are explicitly
reconstructed using exactly one FP32 multiplication
of native dots and the independently read source softmax operation scale; the
source does not expose a separately materialized scaled tensor. Both models pass
480 raw dots plus 480 reconstructed scores, without a signed-zero exception.

Use the existing pinned d812350 headers/library and separate loader closure:

```sh
qk_reference_lib=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
qk_reference_libc=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
c++ -std=c++17 -O2 -I target/qualification/reference-llama/include \
  tests/reference/llama_cpp/llama_reference_qk.cpp -L "$qk_reference_lib" -L "$qk_reference_libc" \
  -Wl,-rpath,"$qk_reference_lib:$qk_reference_libc" \
  -Wl,--dynamic-linker,"$qk_reference_libc/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -o target/qualification/reference-llama/qk_capture
target/qualification/reference-llama/qk_capture \
  target/models/qwen35-0.8b/model.gguf target/h10-independent-qk-qwen35-0.8b.jsonl
cargo build --release --locked --offline --example trace_tokens --features diagnostic-gdn-trace
ZG_TRACE_ALL_LAYERS=1 RAYON_NUM_THREADS=8 target/release/examples/trace_tokens \
  target/models/qwen35-0.8b/model.gguf target/models/qwen35-0.8b/config.json \
  target/h10-rust-qwen35-0.8b.jsonl 11 353 17 42
```

Repeat with qwen35-2b and its SHA listed above. These helpers remain external
qualification clients, never dependencies of the Rust mathematical engine.
See the [accepted baseline](../../../.agents/plans/accepted-baseline/TODO.md)
for physical qualification and the [numerical reference](../../../.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md)
for source arithmetic.

## P07-H11 native attention probabilities

`llama_reference_softmax.cpp` captures native SOFT_MAX output probabilities from
the original pinned library, across all six attention layers and raw causal IDs
11,353,17,42. It reads valid positions by actual tensor strides, excludes padded
slots and preserves signed zeros. The comparison checks all
24 unique records/480 probabilities directly to Rust `full_probs`. No probability
reconstruction or numerical tolerance is used. The helper remains external to
the Rust engine and GPU backend.

Use the existing pinned d812350 headers/library and separate loader closure:

```sh
softmax_reference_lib=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
softmax_reference_libc=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I target/qualification/reference-llama/include tests/reference/llama_cpp/llama_reference_softmax.cpp \
  -L "$softmax_reference_lib" -L "$softmax_reference_libc" \
  -Wl,-rpath,"$softmax_reference_lib:$softmax_reference_libc" \
  -Wl,--dynamic-linker,"$softmax_reference_libc/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -o target/qualification/reference-llama/softmax_capture
target/qualification/reference-llama/softmax_capture \
  target/models/qwen35-0.8b/model.gguf target/h11-independent-softmax-qwen35-0.8b.jsonl
cargo build --release --locked --offline --example trace_tokens --features diagnostic-gdn-trace
ZG_TRACE_ALL_LAYERS=1 RAYON_NUM_THREADS=8 target/release/examples/trace_tokens \
  target/models/qwen35-0.8b/model.gguf target/models/qwen35-0.8b/config.json \
  target/h11-rust-qwen35-0.8b.jsonl 11 353 17 42
```

Repeat with qwen35-2b and its original SHA listed above. The
[accepted baseline](../../../.agents/plans/accepted-baseline/TODO.md)
records the separate physical and source-profile gates.

## Separate build

Compile with C++17 and the **matching** llama.cpp/ggml headers and libraries.
With the correct toolchain available in standard search paths:

    mkdir -p target/qualification/reference-llama
    c++ -std=c++17 -O2 tests/reference/llama_cpp/llama_reference_logits.cpp \
        -lllama -lggml -lggml-base -lm \
        -o target/qualification/reference-llama/sequence_logits
    c++ -std=c++17 -O2 tests/reference/llama_cpp/llama_reference_layers.cpp \
        -lllama -lggml -lggml-base -lm \
        -o target/qualification/reference-llama/layer_capture

Nonstandard installations may require explicit header/library paths, rpath,
or a compatible C++ and glibc toolchain. The validated pvl-l5 test library was
llama.cpp v0.6.0, build 11429, source revision d812350. Always build against
your actual reference revision and record its binary SHA-256.

## Run

Example raw-token direct numerical audit for Qwen3.5 2B: the check requires the
model and its exact SHA-256, reference attention off, the independent
`sequence_logits` binary, 32 tokens, 8 prompts, at most 2 cases per run and a
JSON report destination.

Example all-trained-layers independent parity: the check requires the model
key, model and exact SHA-256, hidden width 2048, reference revision d812350 and
the independent `layer_capture` binary.

Both require local exact-model SHA-256, independently compiled
reference binaries and native Rust results. No model download is performed.

**Scope:** Short CPU numerical and per-layer proof is not a real 4K–32K
whole-model context proof, and it does not qualify GPU whole-model inference.

## P07-H12 native attention values and canonical padding

`llama_reference_value.cpp` captures native `attn_pregate` output across all six
attention layers and causal IDs11,353,17,42. `compare-attention-value.py` requires
the original checkpoint SHA, all24 unique records*2048 values, and every finite
FP32 bit including zero signs. There is no reconstruction or tolerance.

The optional `--padding-audit` decodes257 genuine tokens in the pinned CPU source
with context512 and inspects actual value-MUL_MAT operands/strides at
T1,31,32,33,224,225,255,256,257. It asserts source V/P padding to256-slot boundaries
and bitwise +0 unused slots. `llama_reference_value_padding.cpp` independently
executes original ggml MUL_MAT and asserts nine signed-zero/subnormal fixtures,
including the difference from the former valid-only Rust schedule. This qualifies
a canonical +0 future-cache profile, not arbitrary dirty native cache rewinds.

Using the same pinned library/loader closure as H11:

```sh
value_reference_lib=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
value_reference_libc=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
for value_helper in llama_reference_value llama_reference_value_padding; do
  c++ -std=c++17 -O2 -Wall -Wextra -Werror \
    -I target/qualification/reference-llama/include "tests/reference/llama_cpp/$value_helper.cpp" \
    -L "$value_reference_lib" -L "$value_reference_libc" \
    -Wl,-rpath,"$value_reference_lib:$value_reference_libc" \
    -Wl,--dynamic-linker,"$value_reference_libc/ld-linux-x86-64.so.2" \
    -lllama -lggml -lggml-base -lm -o "target/qualification/reference-llama/$value_helper"
done
target/qualification/reference-llama/llama_reference_value_padding
target/qualification/reference-llama/llama_reference_value \
  target/models/qwen35-0.8b/model.gguf target/h12-independent-value-qwen35-0.8b.jsonl
target/qualification/reference-llama/llama_reference_value \
  target/models/qwen35-0.8b/model.gguf target/h12-independent-value-padding-qwen35-0.8b.jsonl --padding-audit
cargo build --release --locked --offline --example trace_tokens --features diagnostic-gdn-trace
ZG_TRACE_ALL_LAYERS=1 RAYON_NUM_THREADS=8 target/release/examples/trace_tokens \
  target/models/qwen35-0.8b/model.gguf target/models/qwen35-0.8b/config.json \
  target/h12-rust-qwen35-0.8b.jsonl 11 353 17 42
python3 scripts/compare-attention-value.py --self-test
python3 scripts/compare-attention-value.py \
  --reference target/h12-independent-value-qwen35-0.8b.jsonl \
  --candidate target/h12-rust-qwen35-0.8b.jsonl \
  --model target/models/qwen35-0.8b/model.gguf \
  --sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf
```

Repeat source captures/audits/comparison for2B with its original SHA above.
These external test clients never become production engine dependencies.

## P07-H13 native attention scalar-sigmoid gate

The same strict-warning value helper also accepts `--gated` (separate from
`--padding-audit`). It captures native `attn_gated` at all six layers across raw
causal IDs11,353,17,42. Run its H12 compilation command above, then:

```sh
target/qualification/reference-llama/llama_reference_value \
  target/models/qwen35-0.8b/model.gguf target/h13-native-gated-qwen35-0.8b.jsonl --gated
# Use the fresh all-layer H13 Rust trace from the same original raw IDs.
python3 scripts/compare-attention-value.py --stage full_gated \
  --reference target/h13-native-gated-qwen35-0.8b.jsonl \
  --candidate target/h13-rust-qwen35-0.8b.jsonl \
  --model target/models/qwen35-0.8b/model.gguf \
  --sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf
```

Repeat2B with its original SHA. Both checkpoints pass49152 finiteFP32 values
with zero signs, without reconstruction or tolerance. Pregate remains the
comparator's default. Both stages require complete24-record source/Rust coverage.
