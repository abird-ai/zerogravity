# Zerogravity CLI

This package owns the sole `zerogravity` executable, argument handling and text
rendering. Default workspace `cargo build` and `cargo run -- ...` retain the CPU
frontend. Source installation uses `cargo install --path crates/zerogravity-cli`.

The binary also builds with `--no-default-features`: help, version and bounded
artifact/config inspection remain available. Inference requires its selected
backend, `model-qwen35` and `text`; disabled capabilities refuse before source
I/O or device initialization.

Backend commands are explicit:

- `run` uses CPU execution.
- `run-vulkan-amd` selects Vulkan integrated GPU0.
- `run-vulkan-nvidia` selects Vulkan discrete GPU0.
- `run-cuda` selects CUDA device0.
- `run-rocm` selects HIP/ROCm device0.

The Vulkan command names preserve the qualified frontend's device-kind mapping;
they do not discover or prove a vendor identity. Qualification records bind the
actual physical device separately. Execution never falls back to another backend.

Features forward through the library facade: `cpu`, `model-qwen35`, `text`,
`vulkan`, `cuda`, `hip`, `all-models`, `all-backends` and `all`. Default is
`cpu,model-qwen35,text`. `all` excludes qualification and fault-injection tools.
Existing diagnostic feature names remain explicit forwarders for API examples.

The CPU frontend preserves its existing argument/audit limits. The resident GPU
frontend accepts at most 512 consumed tokens, checks worst-case prompt plus
generation before GPU preparation, and requires colocated original GGUF,
`config.json` and `tokenizer.json`. It uses the existing generic GPU model/session
API and preserves EOS handling and the device-token loop.
