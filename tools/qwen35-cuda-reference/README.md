# Qwen3.5 original CUDA reference capture

- Finite original BF16 .8B fresh prompts47/48/49/50; single request, natural
  greedy max8, EOS248046, context512. Cached decode follows the actual produced
  tokens; fresh prompts remain separate requests.
- `discover` authenticates42 retained `original-source` files; `capture`
  authenticates50 retained `capture-source` files. Their10 changed files and8
  additions are explicit stage ancestry. The original42-file manifest remains
  unchanged. Both independently preserve the20 primary runtime/model/operator
  source pins; collector/source/header identities must match their stage.
  The successor request contract and native validator's prompt guard change.
  Original model capture, all24/all42, full-vocabulary/sampler/source/lifetime
  validators, remaining native checks and collector stay byte-identical.
  `source-inheritance.json` records that distinction.
- An owned CPU-only container verifies the complete pinned checkpoint and derives
  IDs from its tokenizer. The original V1 tokenizer independently rechecks them
  before `generate`. Wrong prompt length refuses; no padding or forced teachers.
- `discover` retains actual whole images, launches and reflected layouts.
  `capture` requires a passing discovery for the same exact request and uses its
  whole-image layouts to collect packed arguments. Optional additional passing
  discovery receipts contribute a finite exact-image/formal union. Every layout
  joins the validated actual launch, reflected function and whole module bytes;
  a caller-supplied catalog is not accepted. Unknown images/layouts fail.
- Both modes run fresh plain and observed originals and preserve strict paired
  bytes/state/source/cleanup validation. Packed capture additionally compares
  every common/final/fullV, all24/all42 active-state, consumed-prior and observed
  parameter byte against the same request's discovery, with held input identities
  before and after execution. A failure stays failed. Original batch
  reduction differences grant no bit-invariance or implementation waiver.
- Reference collection does not qualify native recipes, execution or performance.
  No checkpoint download, driver mutation, tuner override or second publisher.

Request:

```json
{
  "schema": "zerogravity.qwen35-cuda-reference.request.v1",
  "model": "Qwen/Qwen3.5-0.8B",
  "source_revision": "2fc06364715b967f1860aea9cf38778875588b17",
  "phase": "prefill",
  "prompt_tokens": 48,
  "cached_tokens": 0,
  "request_batch_size": 1,
  "prompt": "Actual text whose rendered input has exactly 48 tokens",
  "generation": {"max_new_tokens": 8, "eos_token_id": 248046, "temperature": 0.0}
}
```

Run under ROOT supervision; the original owned Podman runtime supplies user
and network isolation. Do not nest rootless Podman inside `native-gpu.py`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/qwen35-cuda-reference/run_reference.py \
  --reference-root /absolute/retained/original-source \
  --owner /absolute/retained/owned_container_v4.py \
  --observer /absolute/retained/libzg_gemm_observer.so \
  --cancellation /absolute/retained/sigterm_cancellation.py \
  --lifecycle-source /absolute/reviewed/tools/serve-lifecycle.py \
  --lifecycle-sha256 REVIEWED_LIFECYCLE_SHA256 \
  --collector /absolute/retained/original-build-5/build \
  --source /absolute/approved/Qwen3.5-0.8B/revision \
  --generation /absolute/approved/common-generation \
  --request /absolute/request.json --output /absolute/fresh-output --mode discover
```

For packed capture, use a new output, replace the source/collector/mode arguments:

```bash
  --reference-root /absolute/retained/capture-source \
  --collector /absolute/retained/original-build-6/build \
  --mode capture --discovery /absolute/same-request-passing-discovery
```

Repeat `--layout-discovery /absolute/other-passing-discovery` up to3 times when
other fresh47–50 discoveries supply additional independently observed images.
An incomplete source/layout/packet/control/image proof refuses. No packet is
reconstructed from an entry name or raw digest. Full-ABI capture retains the
original selected-Triton typed-formal and prelaunch whole-image join checks.
All paths are explicit; legacy missing
worktree paths are never recreated as aliases. Exact approved observer/helper
bytes are required; a similarly named library is insufficient.
An explicitly rebuilt observer may use `--observer-receipt`: it binds the new
artifact to the exact original C++ source, compiler, dependencies and build
command. Its actual digest remains distinct from the historical artifact.
Original nonce-bound cleanup uses shared `owned_command` evidence for every
Podman command, retaining intent, PID/PGID, one reap, full separate streams and
results. Original model stdout/stderr also remain separate retained streams.
The discovery and full-ABI collectors are authenticated separately; neither can
stand in for the other. The original capture controls and complete native packet,
resource, scope and extent validators remain unchanged except the explicit
finite prompt membership guard. Discovery alone does not establish complete
packed arguments. A separately postvalidated discovery may retain an explicit
outer-controller failure; passing underlying references do not relabel that run.

If an otherwise closed capture refuses only an absent layout, another fresh47–50
capture may additionally use these explicit inputs:

```bash
  --late-layout-capture /absolute/failed-capture \
  --late-layout-controller /absolute/failed-controller/result.json \
  --late-layout-controller-sha256 ACTUAL_FAILED_CONTROLLER_SHA256 \
  --late-layout-pins /absolute/original-controller-source-pins.json \
  --late-layout-pins-sha256 ACTUAL_ORIGINAL_PINSET_SHA256
```

The held `--discovery`/`--layout-discovery` receipts must contain exactly one
matching the failed parent's complete request and tokenized operands. Its raw
diagnostic stays bound to that discovery. The current capture retains its own
request/discovery and every mathematical comparison; only the four-field exact
image/entry/callback/formal layout is reusable across these finite requests.
No scalar packet, output, numerical identity or qualification transfers.
This finite reader authenticates predecessor source pins, failed stage/owner/log
closure, unchanged original source, every event/scope/present packet and pointer
query count. It requires only collector INCOMPLETE64, an actually absent layout,
complete late reflection, exact held whole-image/ELF entry and successful public
library/load/resource/getter/launch joins. Unknown failures and ambiguous joins
refuse. Only the actual reflected layout joins the future registration union;
the failed parent's packets, files, status and qualification remain unchanged.
The failed finish can leave worker/result receipts, loaded-parameter after-audit
and the small buffered norm archive unpublished. The reader records these gaps
explicitly and compares every retained complete raw/state/fullV/prior stream
against the same-request discovery. It creates no missing archive or packet and
issues no mathematical acceptance. The new run still needs complete original
plain/observed, source/state/fullV/natural-history and full-ABI validation.

Pure checks: `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tools/qwen35-cuda-reference -v`.
Set `ZG_REFERENCE_SOURCE=/absolute/retained/original-source` to include actual
42-file ancestry, unchanged validator-body and fresh47–50 scope controls. Add
`ZG_CAPTURE_SOURCE=/absolute/retained/capture-source` for the corresponding
50-file/full-ABI controls and mismatched-collector rejection. These are pure
source/contract checks; no model or device is invoked.
The retained original-source package descends from the previous short-HF matrix;
legacy labels remain only inside its authenticated unchanged implementation.
