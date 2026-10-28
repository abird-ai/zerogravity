# Qwen CUDA bundle staging

Explicit file migration for the retained approved .8B/2B N1 CUDA bundles.
No model reads, framework imports, build, device, provider loading or downloads.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/qwen35-cuda-bundle/test_stage.py
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/qwen35-cuda-bundle/stage.py \
  --legacy-root /ABS/retained/native-bundles/0.8b --output /ABS/FRESH/bundle
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/qwen35-cuda-bundle/verify_current.py \
  --root /ABS/FRESH/bundle
```

- Accept only either exact retained manifest SHA/extent in `crates/zerogravity/src/qwen35_cuda/bundle_migration_pins.json`.
- Authenticate current loader/catalogue/image/provider contract source bytes, including the staging and verification code. The receipt preserves the original staging lineage; verify_current.py reports whether that historical lineage is current while independently authenticating today's bytes/contracts.
- Hold and hash every original image, rotary, activation and provider file;
  reject unknown/missing fields/resources, extra directories and symlinks.
- Copy regular files into a fresh private root; preserve relative resource paths
  and exact bytes. No hardlinks, aliases or edits to retained inputs.
- Emit `cuda-kernels.json`, schema `zerogravity.cuda-kernels.v2`. Four closed
  kind-label renames only: `Canonical47CutlassBf16Sm80`→`FixedPromptCutlassBf16Sm80`,
  `ShortHfCutlassBf16Sm80`→`TextCutlassBf16Sm80`,
  `ShortHfFaCausalCombineSm80`→`TextFaCausalCombineSm80`,
  `ShortHfFaSplitCombineSm80`→`TextFaSplitCombineSm80`.
- Rehash staged resources and recheck held source paths/FDs before success. verify_current.py refuses missing/extra files, malformed receipts, current contract-source drift or resource mutation in an already staged current bundle; older valid staging lineage is reported rather than treated as execution authority.
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

## Parity wrapper integration

The ordinary cached47-to-50 parity wrapper accepts either a current staged
bundle or one of the exact retained legacy bundles. Legacy input is never edited.
The wrapper stages it into ZG_QWEN35_CUDA_STAGED_DIR when provided, otherwise
into a catalogue-versioned <evidence>.native-bundle-<pin-prefix> directory.
An existing derived directory is reused only
after verify_current.py reauthenticates its complete resource inventory and
current pinned contract sources; historical receipt lineage is reported separately.

### Current-byte authority versus historical staging lineage

A current bundle is reusable after loader/verifier hardening when its current
manifest and every resource still match the checked-in reference contract.
verify_current.py authenticates those bytes and current contract sources on
every use. The stage receipt remains preserved audit provenance; an older
migration-catalogue digest is reported with stage_lineage_current=false rather
than forcing a gigabyte-scale recopy. Parity evidence records the current
reference-contract digest, not the historical staging-tool version.
