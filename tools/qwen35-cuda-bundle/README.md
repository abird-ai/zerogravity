# Qwen CUDA legacy bundle compatibility

This helper is deliberately not a trust root. It only translates the old
short-hf-native.json shape into the current cuda-kernels.json shape and copies
regular files byte-for-byte. The Rust Qwen CUDA loader remains the sole
authority for kernel, activation and provider byte authentication.

No model reads, framework imports, downloads, device access, pin catalogues or
source-code hashes are involved.

Read-only structural checks:

  PYTHONDONTWRITEBYTECODE=1 python3 -B tools/qwen35-cuda-bundle/stage.py --check-legacy /ABS/legacy-bundle

  PYTHONDONTWRITEBYTECODE=1 python3 -B tools/qwen35-cuda-bundle/stage.py --check-current /ABS/current-bundle

Fresh compatibility copy:

  PYTHONDONTWRITEBYTECODE=1 python3 -B tools/qwen35-cuda-bundle/stage.py --legacy-root /ABS/legacy-bundle --output /ABS/FRESH/current-bundle

The legacy translation changes only the manifest filename/schema and four
historical kind labels:

- Canonical47CutlassBf16Sm80 -> FixedPromptCutlassBf16Sm80
- ShortHfCutlassBf16Sm80 -> TextCutlassBf16Sm80
- ShortHfFaCausalCombineSm80 -> TextFaCausalCombineSm80
- ShortHfFaSplitCombineSm80 -> TextFaSplitCombineSm80

Everything else is structural validation: bounded canonical relative paths,
regular files only, no symlink traversal, distinct resources/kinds, bounded
inventory/extent, exact source-byte copying, fsync, and cleanup of failed fresh
outputs.

stage-result.json from older staged bundles is accepted as optional historical
audit metadata, but it is never required or trusted for execution.
