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

## Offline recovery from retained reference artifacts

A missing assembled bundle need not require a new model capture. Check the
project's separate container store, not only the default Podman store.

For the `.8B` historical N1 reference:

- A retained `canonical47-native.json` bundle supplies 14 model images, nine
  rotary images and the activation image. It is not a complete current bundle.
- The exact pinned reference image's `vllm_flash_attn/_vllm_fa2_C.abi3.so`
  contains the two missing whole SM80 modules. Extract into fresh external
  scratch with the retained CUDA tool:

      cuobjdump --gpu-architecture sm_80 --extract-elf all /ABS/_vllm_fa2_C.abi3.so

- Select `TextFaCausalCombineSm80` and `TextFaSplitCombineSm80` by exact extent
  and content identity from `ProjectionImage` in
  `crates/zerogravity-gpu/src/cuda/native/cubin_manifests.rs`, never extraction
  ordinal/name alone. Do not patch, rebuild or repin a nonmatching module.
- Copy the required images and original authenticated cuBLAS/cuBLASLt files
  into a fresh XDG qualification directory. Write `cuda-kernels.json` using the
  current schema and kind names; all resource paths must be relative for this
  structural helper. The complete N1 bundle has 28 distinct resources.
- Run `stage.py --check-current`, then the parity wrapper's structural preflight
  and fresh phase-1 capture. Rust remains the authority for every payload and
  provider identity. Extraction/structural success does not qualify inference.

The current required image set is owned by `text_required_images_for_cohort`;
`2B` and B4 require their own inventories. Do not silently reuse the `.8B` set.
Phase 1 uses the existing 8 GiB historical-reference host profile; phase 2 keeps
ordinary's 4 GiB host profile. Provider/device limits remain 4,608/5,120 MiB.
