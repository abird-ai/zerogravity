# E2B ordinary plain CPU matrix

- Existing approved offline image/source; no acquisition/GPU/serving route.
- Seven genuine source-tokenized prompts:1/2/3/4/32/57 rows andUnicode3.
- Up to8 natural greedy outputs, actual EOS[1,106,50]; no forced tokens,
  padding, expected operands or replacement state. True maximum frontier64.
- One original BF16 model; two independent caches/cell. Observed first cache
  stores all35 block input/output planes; plain second compares every fullV
  and all15 complete owner KV pairs byte-exact. Boundary repeat count is1.
- BF16 archives reinterpret original storage losslessly; no numeric cast.
  Per-cell raw16MiB/boundaries32MiB/4096records unchanged. Worst57→64:
  raw13,115,392B +boundaries13,762,560B. Original4CPU/12GiB limits unchanged.
- Direct host-owned rootless Podman supervisor supplies networknone/read-only,
  approved image/full source before+after, bounded logs/group/reap/EOF cleanup.
  Do not put Podman under the separate native-gpu user namespace.

Run `python3 tools/gemma4-e2b-ordinary-reference/run_reference.py --source
CANONICAL_SNAPSHOT/files --new-output FRESH_DIRECTORY` in an untimed CPU/I/O
window. The independent contract checks the complete matrix, recordSHA/EOF,
actual private feedback, all consumed prior prefixes and genuine frontier64.

Rust example `gemma4_cpu_matrix` requires the explicit fresh healthy invocation
SHA. Inputs come from actual text through the managed SnapshotLease tokenizer;
reference data is used only for comparison. It uses the same ReferenceText,
SessionAuthority and private computed feedback as the explicit CPU CLI.
Fatal per-word differences roll back that proposal and fail the whole matrix;
other independent cells remain available for diagnosis. Source-close/final0 and
real session/input pressure/cancel/refusal/retry are part of the same unit.

No general trained/chat/native/HTTP/performance qualification. Existing selected
three-call reference and prior tokenizer/text anchors remain unchanged.
