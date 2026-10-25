# qwen27b-hip-qualification

Source-only controller for the Qwen 3.8 27B HIP qualification unit. The
controller freezes one explicit source root plus the eleven-file natural tool;
it never discovers Git state, a baseline, or an overlay.

## Four explicit operands

All operands are absolute. The payload and archive must be fresh and must not
overlap the source, the source manifest, or the tracked tool.

```sh
python3 -B tools/qwen27b-hip-qualification/prepare.py \
  --source-root /absolute/fresh-source \
  --source-manifest /absolute/final-native-manifest.json \
  --payload /absolute/fresh-payload \
  --archive /absolute/fresh/source-payload.tar
```

`--source-manifest` maps each repo-relative member to
`{kind, bytes, mode, sha256}`. Counts are derived from the copied union:
native members plus exactly eleven natural-tool members. Each native source
member is limited to 64 MiB, so the declared 3,814,536-byte `generated.rs`
member is admitted. The archive is a fresh code bundle consumed by
`payload/runner/bootstrap.py`; name it `source-payload.tar` at the remote base.

The archive contains only the controller and the frozen source union. It does
not contain model weights, corpora, caches, prior bundles, sessions or
receipts.

## Accepted natural receipt

Without an external receipt the installed authority is
`AWAITING_ACCEPTED_ORIGINAL_CPU_CAPTURE` and every natural mode refuses.

An accepted receipt is installed only from an external caller-pinned file. The
three operands are required together and are never inferred:

```sh
python3 -B tools/qwen27b-hip-qualification/prepare.py \
  --source-root /absolute/fresh-source \
  --source-manifest /absolute/final-native-manifest.json \
  --payload /absolute/fresh-payload \
  --archive /absolute/fresh/source-payload.tar \
  --accepted-receipt /absolute/external/receipt.json \
  --accepted-receipt-bytes 12345 \
  --accepted-receipt-sha256 <64 hex digits>
```

The installer opens the receipt with one bounded `O_NOFOLLOW` descriptor,
checks its extent, EOF, stable stamp and SHA-256, and validates the original
producer schema, scope, image, snapshot and nine-file package. It records the
receipt pin and the four declared artifact pins; it does not read corpus bytes,
model weights or the artifacts themselves, and it never pins itself.

## Source identity

`payload/source/tools/qwen27b-natural-full-window-reference/` remains the
separate eleven-file natural authority. The original owner and cancellation
helpers are byte-identical:

- `owned_container_v4.py`: `53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d`
- `sigterm_cancellation.py`: `00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128`

No capture, model execution, build, GPU run or remote effect is claimed by
this source tree. The combined CPU verification batch runs after the Q/V/G
source owners close; the nine runtime package pins in
`crates/zerogravity/examples/qwen27b_hip_matrix/expected.rs` are updated once
by that executor from the actual runtime files.
