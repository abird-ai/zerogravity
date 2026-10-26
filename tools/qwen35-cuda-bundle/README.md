# Qwen CUDA bundle staging

Explicit file migration for the retained approved .8B/2B N1 CUDA bundles.
No model reads, framework imports, build, device, provider loading or downloads.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/qwen35-cuda-bundle/test_stage.py
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/qwen35-cuda-bundle/stage.py \
  --legacy-root /ABS/retained/native-bundles/0.8b --output /ABS/FRESH/bundle
```

- Accept only either exact retained manifest SHA/extent in `pins.json`.
- Authenticate current loader/catalogue/image/provider contract source bytes.
- Hold and hash every original image, rotary, activation and provider file;
  reject unknown/missing fields/resources, extra directories and symlinks.
- Copy regular files into a fresh private root; preserve relative resource paths
  and exact bytes. No hardlinks, aliases or edits to retained inputs.
- Emit `cuda-kernels.json`, schema `zerogravity.cuda-kernels.v2`. Four closed
  kind-label renames only: `Canonical47CutlassBf16Sm80`→`FixedPromptCutlassBf16Sm80`,
  `ShortHfCutlassBf16Sm80`→`TextCutlassBf16Sm80`,
  `ShortHfFaCausalCombineSm80`→`TextFaCausalCombineSm80`,
  `ShortHfFaSplitCombineSm80`→`TextFaSplitCombineSm80`.
- Rehash staged resources and recheck held source paths/FDs before success.
  `stage-result.json` records new manifest, original pins and unchanged control
  source lineage. Failures after output creation retain `stage-failure.json`;
  discard no failed output and reuse no existing output directory.
- Current Rust loader still authenticates original source/images/ABI/provider
  contracts. Staging enables no admission and grants no inference/ABI/math,
  physical memory, serving or performance qualification.
- .8B N1/B4 share this image cohort. This2B bundle has the original17-image N1
  cohort; it does **not** supply the separately required cached2B B4 image
  `Cohort2BAmpere64SlicedSm80`. Unknown resources are refused rather than added.
- Pure controls use only tiny temporary synthetic files. ROOT owns running the
  controls and real staging; controls do not stand in for actual native checks.
