# Draft Patches — volume 8

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay-internal-draft-2/REPLAY_PREPARATION.patch

Original UTF-8 bytes: 78415; SHA256 `d9b1b93c3e942e6249b7948a104cedc61dd89538ea76158e8cdca2e7ed5f507a`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/Cargo.toml
@@ -0,0 +1,23 @@
+[package]
+name = "original-cached-gdn-archive-replay"
+version = "0.0.0"
+edition = "2024"
+publish = false
+
+[workspace]
+
+[dependencies]
+zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-artifacts", features = ["cache"] }
+zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core" }
+zerogravity-model-qwen35 = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35", features = ["cpu", "original-hf-binding"] }
+serde_json = "=1.0.151"
+sha2 = "=0.10.9"
+libc = "=0.2.190"
+
+[profile.release]
+opt-level = 3
+codegen-units = 1
+
+[[bin]]
+name = "original-cached-gdn-archive-replay"
+path = "main.rs"
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/ROOT.md
@@ -0,0 +1,80 @@
+# ROOT-only archived cached GDN gate
+
+Author: spine_review. Parent: model-execution.
+
+- Source-only preparation. No author build, tests, operator/provider/model,
+  container, GPU, MAIN or shared-plan changes. Preserve original57a4e64 freeze.
+- Parent `STRIDE_FOLLOWUP.patch` changes ONLY rows256→2 and bad.query→bad.decay.
+  Original F32 backing, same issuer and valid two-element decay shape now reach
+  the intended contiguous guard. No production change in that followup.
+- `.8B` actual public step1 archive provides all six cached inputs and all three
+  results. Whole archive SHA, selected byte SHA, dtype/shape/stride, previous
+  real chunk return and98 historical fullV/cache checks are mandatory.
+- This is a replay against an already captured authentic public call. Exporter
+  reads actual original archive bytes; it calls no Torch/operator/model and
+  derives no expected value. It does not claim a new public runtime execution.
+- Imported exp(g) is the actual archived decay. No native exp, live coefficient
+  producer, initial-token substitute, state publication, canonical transpose,
+  source mutation, model/session admission or new engine is introduced.
+- Native helper opens and verifies the genuine complete `.8B` shard through
+  existing SourceWeights/OriginalWindowOperators and calls ONLY the public
+  execute_cached_gdn_slice. All266240 output words are compared to independent
+  archive bits, plus268320 retained input words. Signed zero/subnormals survive.
+- Existing owned_lifetime.py copied byte-exact; owned_native.py retains original
+  source_files/execute bodies, changing only scope wording and removing unused
+  coefficient main. Exec-FD SHA, exact four CPUs, noGPU environment,512MiB AS,
+  finite900s child/2400s outer deadline,4MiB combined logs, WNOWAIT/group-bound
+  TERM/grace/two-KILL-before-any-reap/independent reap/EOF remain.
+- Host ledger128MiB:64MiB sealed source +two4MiB bindings +8MiB qualifier control.
+  Aggregate six-input and three-result payloads and conservative helper scratch
+  use original same-source host reservations. Real pool pressure/pre-cancel/
+  wrong-layer refusal, exact live/retained accounting and last-drop0 required.
+  Configured ledger and AS limits are not a whole-host physical-RAM claim.
+- ROOT must independently review/merge both prepared source patches, record the
+  complete actual build/source list and attest the resulting native binary.
+  `--implementation` is an explicit ROOT review/build attestation, not proof
+  obtained merely by providing hashes. All changed paths are required; exact
+  new child and unchanged original arithmetic pins are checked independently.
+  Parent Program/HostMath merges must preserve concurrent coefficient nodes.
+
+## Availability limits
+
+- 2B actual paired generic archive has final recurrent state and QKV/conv, but
+  lacks cached q/k/v/beta/decay/core-call inputs/targets. Do not reconstruct them.
+- 27B actual public reference contains embedding/InputNorm/QKV only. No authentic
+  H48 cached operands/results exist in the inspected archives; historical27B
+  runs do not supply them. Current source-level cached_heads also requires F32
+  GdnNorm while original27B has BF16 gamma. Thus H48 dimension compatibility
+  does not make the current original27B slice executable. No dtype gate changed.
+- `availability.json` records actual inspected archive headers/reference scope.
+  No synthetic, repeated `.8B`, pre-convolution QKV or guessed state substitutes.
+  Future2B/H48 need actual original public initial+cached capture with immutable
+  source/runtime pins and independently transparent full-model evidence first.
+
+## ROOT sequence
+
+Run only after current benchmark/GPU holds and source integration permit it.
+Set `RUN` to this `replay/` directory. All output directories are fresh.
+
+```bash
+python3 "$RUN/test_contracts.py"
+python3 "$RUN/export_reference.py" --new-output "$RUN/root-reference-1"
+CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
+  cargo build --release --offline --manifest-path "$RUN/Cargo.toml"
+python3 "$RUN/run_replay.py" --binary "$ROOT_ATTESTED_BINARY" \
+  --binary-sha256 "$ROOT_BINARY_SHA" \
+  --reference-directory "$RUN/root-reference-1" \
+  --implementation "$ROOT_SOURCE_ATTESTATION" --new-output "$RUN/root-native-1"
+python3 "$RUN/validate_replay.py" --output "$RUN/root-native-1" \
+  --reference-directory "$RUN/root-reference-1" --implementation "$ROOT_SOURCE_ATTESTATION"
+```
+
+ROOT source attestation JSON: `status=ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE`,
+`prepared_manifest_sha256=<this prepared-manifest.json SHA>`,
+`full_model_or_StateRoot_or_exp_or_GPU_qualified=false`, and `files` containing
+the complete reviewed actual build-source rows `{path,bytes,sha256}`. Required
+paths and exact child/dependency pins live in prepared-manifest.json. Preserve
+actual Cargo lock/artifact/build/source receipts independently.
+
+Actual acceptance still requires focused/strict feature tests, original source
+routes and full isolated repository gate. No new acceptance is claimed here.
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/availability.json
@@ -0,0 +1,191 @@
+{
+  "0.8b": {
+    "archive": {
+      "bytes": 6472400,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-1/actual-gdn-frames.safetensors",
+      "sha256": "c8b7c72d0644c418dfa5c74e974678f7ee76e3d7cc9e83a7c157c3a9f836c0fa"
+    },
+    "available": true,
+    "heads": 16,
+    "tensor_names": [
+      "l2_key.input.x",
+      "l2_key.return.actual_return_0",
+      "l2_key.return.inv_norm",
+      "l2_query.input.x",
+      "l2_query.return.actual_return_0",
+      "l2_query.return.inv_norm",
+      "recurrent.decay_and_prior.decay_t",
+      "recurrent.decay_and_prior.last_recurrent_state",
+      "recurrent.decayed_state.last_recurrent_state",
+      "recurrent.delta.delta",
+      "recurrent.fp32_core.core_attn_out",
+      "recurrent.fp32_transposed.beta",
+      "recurrent.fp32_transposed.decay",
+      "recurrent.fp32_transposed.key",
+      "recurrent.fp32_transposed.query",
+      "recurrent.fp32_transposed.value",
+      "recurrent.input.beta",
+      "recurrent.input.g",
+      "recurrent.input.initial_state",
+      "recurrent.input.key",
+      "recurrent.input.query",
+      "recurrent.input.value",
+      "recurrent.memory_prediction.beta_t",
+      "recurrent.memory_prediction.k_t",
+      "recurrent.memory_prediction.kv_mem",
+      "recurrent.memory_prediction.q_t",
+      "recurrent.memory_prediction.v_t",
+      "recurrent.proposed_state.last_recurrent_state",
+      "recurrent.qk_normalized.key",
+      "recurrent.qk_normalized.query",
+      "recurrent.query_scaled.query",
+      "recurrent.return.actual_return_0",
+      "recurrent.return.actual_return_1",
+      "recurrent.return.beta",
+      "recurrent.return.core_attn_out",
+      "recurrent.return.decay",
+      "recurrent.return.key",
+      "recurrent.return.last_recurrent_state",
+      "recurrent.return.query",
+      "recurrent.return.value"
+    ]
+  },
+  "27b": {
+    "actual_public_archive": {
+      "bytes": 65417,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-typed-oracle-20261020/root-reference-1/captured/reference.json",
+      "sha256": "794765650ff40a52b821cebf85889cb46593b337ea79b94b50f4bad4d244e0aa"
+    },
+    "actual_schema": "zerogravity.original27B.public-embedding-norm-qkv.v1",
+    "actual_scope": "Actual original27B full embedding/matrix raw mmap views, selected two independent public F.embedding\u2192unchanged original RMS\u2192F.linear calls only. No model constructor/token loop/cache/StateRoot/FLA/GPU/performance/full-model equivalence/production RAM admission. Dispatch metadata observes framework ops, not internal machine kernel selection. Original learned tensors stay BF16; .8B source36F32-restored is a different source policy.",
+    "actual_steps": [
+      {
+        "token": 11
+      },
+      {
+        "token": 353
+      }
+    ],
+    "available": false,
+    "current_model_slice_refusal": "cached_heads requires F32 GdnNorm; authentic27B raw GdnNorm is BF16. Core dimensional compatibility alone is not source-level eligibility.",
+    "heads": 48,
+    "missing": [
+      "all actual H48 cached q/key/value/beta/decay/prior and outputs"
+    ],
+    "no_synthetic_or_tiled_substitute": true,
+    "source_header_gate": {
+      "actual_header_descriptor": {
+        "data_offsets": [
+          168847552,
+          168847808
+        ],
+        "dtype": "BF16",
+        "shape": [
+          128
+        ]
+      },
+      "header_bytes": 45392,
+      "header_sha256": "7bdeb03df566aa804dca2cb48e420908cb1864e5222858ad04918299883287e0",
+      "index_sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
+      "prior_source_pins": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-typed-oracle-20261020/source-pins.json",
+      "prior_source_pins_sha256": "470b29c0587d9f24da4fdd1eadcfe70e4b4ee47d645e7198573ae5e9f8f9a5fb",
+      "raw_parameter_name": "model.language_model.layers.0.linear_attn.norm.weight",
+      "shard": "model-00001-of-00018.safetensors",
+      "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
+      "source_weight_payload_read": false
+    }
+  },
+  "2b": {
+    "available": false,
+    "missing": [
+      "actual BF16 recurrent-call q/key/value/beta",
+      "actual F32 exp(g) decay",
+      "actual cached core_f32 output"
+    ],
+    "no_reconstruction": true,
+    "observed_header": {
+      "header_bytes": 1528,
+      "header_sha256": "493a207536ab0e68260ec7d44537f70028b204a1030e9ec5fcb37b30ff95b709",
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/observed/captured-reference/source-36-f32-restored/step-1/observed-intermediates.safetensors",
+      "scope": "actual header inventory only; no reconstructed cached operands",
+      "tensor_names": [
+        "embedding",
+        "final_logits",
+        "final_norm",
+        "layer-00.attention_input_norm",
+        "layer-00.attention_output",
+        "layer-00.ffn_fused",
+        "layer-00.ffn_gate",
+        "layer-00.ffn_input_norm",
+        "layer-00.ffn_output",
+        "layer-00.ffn_up",
+        "layer-00.gdn_conv_silu",
+        "layer-00.gdn_qkv",
+        "layer-00.gdn_recurrent_fp32",
+        "layer-00.post_attention_residual",
+        "layer-00.post_ffn_residual"
+      ],
+      "whole_archive_bytes": 1640960
+    },
+    "raw_cache_header": {
+      "header_bytes": 4552,
+      "header_sha256": "e16e9a714d9f8fdd82e0e6aea38d2ded57024e089a487a5941aac05aff0abdbe",
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/observed/captured-reference/source-36-f32-restored/step-1/raw-official.safetensors",
+      "scope": "actual header inventory only; no reconstructed cached operands",
+      "tensor_names": [
+        "layer-00.conv",
+        "layer-00.recurrent",
+        "layer-01.conv",
+        "layer-01.recurrent",
+        "layer-02.conv",
+        "layer-02.recurrent",
+        "layer-03.keys",
+        "layer-03.values",
+        "layer-04.conv",
+        "layer-04.recurrent",
+        "layer-05.conv",
+        "layer-05.recurrent",
+        "layer-06.conv",
+        "layer-06.recurrent",
+        "layer-07.keys",
+        "layer-07.values",
+        "layer-08.conv",
+        "layer-08.recurrent",
+        "layer-09.conv",
+        "layer-09.recurrent",
+        "layer-10.conv",
+        "layer-10.recurrent",
+        "layer-11.keys",
+        "layer-11.values",
+        "layer-12.conv",
+        "layer-12.recurrent",
+        "layer-13.conv",
+        "layer-13.recurrent",
+        "layer-14.conv",
+        "layer-14.recurrent",
+        "layer-15.keys",
+        "layer-15.values",
+        "layer-16.conv",
+        "layer-16.recurrent",
+        "layer-17.conv",
+        "layer-17.recurrent",
+        "layer-18.conv",
+        "layer-18.recurrent",
+        "layer-19.keys",
+        "layer-19.values",
+        "layer-20.conv",
+        "layer-20.recurrent",
+        "layer-21.conv",
+        "layer-21.recurrent",
+        "layer-22.conv",
+        "layer-22.recurrent",
+        "layer-23.keys",
+        "layer-23.values",
+        "logits"
+      ],
+      "whole_archive_bytes": 20284880
+    }
+  },
+  "author_math_runtime": false,
+  "schema": "zerogravity.cached-gdn.available-authentic-inputs.v1"
+}
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/contracts.py
@@ -0,0 +1,140 @@
+"""ROOT-only byte transport; no operator/model/provider calls on import."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import struct
+
+HERE = Path(__file__).resolve().parent
+INPUTS = ('query', 'key', 'value', 'beta', 'decay', 'previous_state')
+OUTPUTS = ('core_f32', 'output_bf16', 'next_state')
+STAGES = INPUTS + OUTPUTS
+PROFILE = 'source-36-f32-restored'
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+def digest(raw):
+    return hashlib.sha256(raw).hexdigest()
+
+def sha(path):
+    with Path(path).open('rb') as opened:
+        return hashlib.file_digest(opened, 'sha256').hexdigest()
+
+def unique(pairs):
+    value = {}
+    for key, item in pairs:
+        require(key not in value, 'duplicate JSON field')
+        value[key] = item
+    return value
+
+def identity(meta):
+    return (meta.st_dev, meta.st_ino, meta.st_size, meta.st_mtime_ns, meta.st_ctime_ns)
+
+def read(path, cap=2*1024*1024, row=None):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as opened:
+        before = os.fstat(opened.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular input')
+        raw = opened.read(cap+1)
+        after = os.fstat(opened.fileno())
+    require(len(raw) <= cap and identity(before) == identity(after), 'unchanged bounded input')
+    if row is not None:
+        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'exact input bytes/SHA')
+    return raw
+
+def pin(path, row, cap=2*1024*1024):
+    read(path, cap, row)
+
+def read_json(path):
+    return json.loads(read(path), object_pairs_hook=unique)
+
+def write(path, value):
+    with Path(path).open('x') as opened:
+        json.dump(value, opened, indent=2, sort_keys=True)
+        opened.write('\n')
+
+def frozen():
+    document = read_json(HERE/'prepared-manifest.json')
+    for row in document['files']:
+        pin(HERE/row['file'], row)
+    parent = read_json(document['parent_freeze']['path'])
+    pin(document['parent_freeze']['path'], document['parent_freeze'])
+    parent_root = Path(document['parent_freeze']['path']).parent
+    for row in parent['files']:
+        pin(parent_root/row['file'], row)
+        pin(parent_root/'frozen-prepared'/row['file'], row)
+    for row in document['dependency_source']:
+        pin(row['path'], row)
+    return sha(HERE/'prepared-manifest.json')
+
+def implementation(path):
+    """Mandatory ROOT review/build attestation, not independent build proof."""
+    value = read_json(path)
+    require(value['status'] == 'ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE' and
+            value['prepared_manifest_sha256'] == frozen() and
+            value['full_model_or_StateRoot_or_exp_or_GPU_qualified'] is False,
+            'ROOT must review merged implementation/build closure first')
+    rows = value['files']
+    require(type(rows) is list and 1 <= len(rows) <= 512, 'bounded complete ROOT build-source list')
+    require(len({row['path'] for row in rows}) == len(rows), 'unique ROOT source paths')
+    expected = read_json(HERE/'prepared-manifest.json')['expected_merged_children']
+    by_path = {row['path']: row for row in rows}
+    require(set(read_json(HERE/'prepared-manifest.json')['required_build_paths']) <= set(by_path),
+            'ROOT attestation includes all changed parents/children and original numerical source')
+    for row in rows:
+        require(Path(row['path']).is_absolute(), 'absolute ROOT source path')
+        pin(row['path'], row)
+    for row in expected:
+        require(by_path.get(row['path']) == row, 'exact reviewed cached child/unchanged CPU arithmetic')
+    return sha(path)
+
+def finite(raw, dtype):
+    require(dtype in ('BF16', 'F32'), 'closed source dtype')
+    size, fmt, mask = (2, '<H', 0x7f80) if dtype == 'BF16' else (4, '<I', 0x7f800000)
+    require(raw and len(raw) % size == 0, 'complete raw finite words')
+    require(all(bits & mask != mask for bits, in struct.iter_unpack(fmt, raw)), 'finite raw input/output')
+
+def archive(row):
+    raw = read(row['path'], 32*1024*1024, row)
+    n, = struct.unpack_from('<Q', raw)
+    require(1 < n <= 1024*1024 and 8+n < len(raw), 'bounded original archive header')
+    header = json.loads(raw[8:8+n], object_pairs_hook=unique)
+    tensors, spans = {}, []
+    for name, item in header.items():
+        if name == '__metadata__':
+            continue
+        dtype, shape = item['dtype'], item['shape']
+        require(dtype in ('BF16', 'F32'), 'original cached archive dtypes')
+        count = 1
+        require(type(shape) is list and len(shape) <= 8, 'bounded source rank')
+        for axis in shape:
+            require(type(axis) is int and 0 < axis <= 262144, 'bounded source axis')
+            count *= axis
+        lo, hi = item['data_offsets']
+        require(type(lo) is type(hi) is int and 0 <= lo < hi and
+                hi-lo == count*(2 if dtype == 'BF16' else 4) and 8+n+hi <= len(raw),
+                'complete typed original archive extent')
+        payload = raw[8+n+lo:8+n+hi]
+        finite(payload, dtype)
+        tensors[name] = {'dtype': dtype, 'shape': shape, 'raw': payload}
+        spans.append((lo, hi))
+    end = 0
+    for lo, hi in sorted(spans):
+        require(lo == end, 'complete nonoverlapping original archive')
+        end = hi
+    require(8+n+end == len(raw), 'no trailing original archive bytes')
+    return tensors
+
+def raw_file(root, row):
+    require(Path(row['file']).name == row['file'], 'contained operand filename')
+    raw = read(root/row['file'], 4*1024*1024, row)
+    finite(raw, row['dtype'])
+    return raw
+
+def widened(raw, dtype):
+    if dtype == 'BF16':
+        return b''.join((word << 16).to_bytes(4, 'little') for word, in struct.iter_unpack('<H', raw))
+    require(dtype == 'F32', 'source F32 transport')
+    return raw
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/export_reference.py
@@ -0,0 +1,110 @@
+"""ROOT ONLY archived genuine public reference extraction; no arithmetic."""
+import argparse
+import json
+from pathlib import Path
+from contracts import HERE, INPUTS, OUTPUTS, PROFILE, STAGES, archive, digest, frozen, pin, read_json, require, sha, write
+
+NAMES = {
+    'query': ('recurrent.input.query', 'BF16', [1,1,16,128]),
+    'key': ('recurrent.input.key', 'BF16', [1,1,16,128]),
+    'value': ('recurrent.input.value', 'BF16', [1,1,16,128]),
+    'beta': ('recurrent.input.beta', 'BF16', [1,1,16]),
+    'decay': ('recurrent.decay_and_prior.decay_t', 'F32', [1,16,1,1]),
+    'previous_state': ('recurrent.input.initial_state', 'F32', [1,16,128,128]),
+    'core_f32': ('recurrent.fp32_core.core_attn_out', 'F32', [1,16,1,128]),
+    'output_bf16': ('recurrent.return.actual_return_0', 'BF16', [1,1,16,128]),
+    'next_state': ('recurrent.return.actual_return_1', 'F32', [1,16,128,128]),
+}
+
+def authenticated():
+    binding = read_json(HERE/'source-bindings.json')
+    for key in ('capture_manifest','capture_validation','step0_metadata','step1_metadata'):
+        pin(binding[key]['path'], binding[key])
+    capture = read_json(binding['capture_manifest']['path'])
+    validation = read_json(binding['capture_validation']['path'])
+    require(capture['profile'] == PROFILE and capture['tokens'] == [11,353] and
+            capture['source_reverified_after_capture'] is True and
+            capture['modeling_sha256'] == '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b',
+            'original pinned source/function/profile capture')
+    require(validation['validated'] is True and validation['raw_prior_capture_checks'] == 98 and
+            validation['second_token_initial_state_is_first_token_return_bits'] is True,
+            'accepted actual fullV/cache transparency and first→cached link')
+    before, actual = archive(binding['step0_archive']), archive(binding['step1_archive'])
+    metadata = read_json(binding['step1_metadata']['path'])
+    require(metadata['step'] == 1 and metadata['input_token'] == 353 and
+            metadata['actual_return_matches_cache'] is True and
+            metadata['actual_operator_calls'] == {'l2norm':2,'torch_recurrent_gated_delta_rule':1},
+            'one actual cached call, not initial chunk or synthetic state')
+    prior = before['chunk.return.actual_return_1']
+    require(prior['dtype'] == 'F32' and prior['shape'] == [1,16,128,128] and
+            prior['raw'] == actual['recurrent.input.initial_state']['raw'] ==
+            actual['recurrent.decay_and_prior.last_recurrent_state']['raw'],
+            'full actual initial chunk returned state equals original cached prior')
+    selected = {}
+    for name, (key, dtype, shape) in NAMES.items():
+        tensor, description = actual[key], metadata['tensors'][key]
+        require(tensor['dtype'] == dtype and tensor['shape'] == shape and
+                description['shape'] == shape and description['contiguous'] is True and
+                description['raw_sha256'] == digest(tensor['raw']) and
+                description['bytes'] == len(tensor['raw']), 'actual full typed cached stage: '+name)
+        selected[name] = (tensor, description)
+    return binding, selected
+
+def validate(directory):
+    binding, selected = authenticated()
+    inputs, reference = read_json(directory/'inputs.json'), read_json(directory/'reference.json')
+    require(inputs['schema'] == 'zerogravity.cached-gdn.archived-inputs.v1' and
+            reference['schema'] == 'zerogravity.cached-gdn.archived-reference.v1' and
+            inputs['source_bindings_sha256'] == reference['source_bindings_sha256'] == sha(HERE/'source-bindings.json') and
+            reference['inputs_sha256'] == sha(directory/'inputs.json') and
+            inputs['model'] == reference['model'] == '0.8b' and
+            inputs['heads'] == reference['heads'] == 16 and inputs['layer'] == reference['layer'] == 0 and
+            inputs['profile'] == reference['profile'] == PROFILE and
+            reference['expected_output_words'] == 266240 and
+            reference['external_decay_no_native_exp'] is True,
+            'exact genuine cached artifact namespace/profile')
+    from contracts import raw_file
+    require(set(inputs['operands']) == set(INPUTS) and set(reference['targets']) == set(OUTPUTS),
+            'all six actual inputs and all three actual outputs')
+    for name in STAGES:
+        row = inputs['operands'][name] if name in INPUTS else reference['targets'][name]
+        tensor, description = selected[name]
+        require(row['original_tensor'] == NAMES[name][0] and row['dtype'] == tensor['dtype'] and
+                row['shape'] == tensor['shape'] and row['original_stride'] == description['stride'] and
+                raw_file(directory, row) == tensor['raw'], 'all signed original cached words: '+name)
+    require({p.name for p in directory.iterdir()} == {'inputs.json','reference.json',*(name+'.raw' for name in STAGES)},
+            'complete immutable extracted artifact tree')
+    return {'status':'PASS_ARCHIVED_PUBLIC_CACHED_OPERANDS_AND_TARGETS_ONLY',
+            'inputs_sha256':sha(directory/'inputs.json'), 'reference_sha256':sha(directory/'reference.json'),
+            'source_bindings_sha256':sha(HERE/'source-bindings.json'), 'expected_output_words':266240,
+            'full_model_or_live_decay_or_StateRoot_or_H48_qualified':False}
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--new-output', type=Path, required=True)
+    args = parser.parse_args()
+    frozen()
+    binding, selected = authenticated()
+    require(args.new_output.resolve().parent == HERE, 'fresh own-run export only')
+    args.new_output.mkdir(mode=0o700)
+    rows = {}
+    for name, (tensor, description) in selected.items():
+        raw = tensor['raw']
+        file = name+'.raw'
+        with (args.new_output/file).open('xb') as opened:
+            opened.write(raw)
+        rows[name] = {'file':file,'bytes':len(raw),'sha256':digest(raw),'dtype':tensor['dtype'],
+                      'shape':tensor['shape'],'original_stride':description['stride'],
+                      'original_tensor':NAMES[name][0]}
+    common = {'model':'0.8b','heads':16,'layer':0,'teacher_tokens':[11,353],'profile':PROFILE,
+              'source_bindings_sha256':sha(HERE/'source-bindings.json'),
+              'external_decay_no_native_exp':True,'no_new_public_operator_or_model_execution':True}
+    write(args.new_output/'inputs.json',dict(common,schema='zerogravity.cached-gdn.archived-inputs.v1',
+          operands={name:rows[name] for name in INPUTS}))
+    write(args.new_output/'reference.json',dict(common,schema='zerogravity.cached-gdn.archived-reference.v1',
+          inputs_sha256=sha(args.new_output/'inputs.json'),targets={name:rows[name] for name in OUTPUTS},
+          expected_output_words=266240))
+    print(json.dumps(validate(args.new_output),sort_keys=True))
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/main.rs
@@ -0,0 +1,521 @@
+//! ROOT ONLY: actual archived cached operands/outputs through shared HostMath.
+use serde_json::{Value, json};
+use sha2::{Digest, Sha256};
+use std::collections::BTreeMap;
+use std::error::Error;
+use std::fs::OpenOptions;
+use std::io::{Read, Write};
+use std::os::unix::fs::OpenOptionsExt;
+use std::path::{Component, Path};
+use std::sync::atomic::AtomicBool;
+use zerogravity_artifacts::artifact::Artifact;
+use zerogravity_artifacts::cache::TransferAdmission;
+use zerogravity_artifacts::source_weights::{
+    ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits,
+};
+use zerogravity_core::accelerator::program::ProgramNumericalProfile;
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, DeviceBudgetSnapshot, PhysicalDeviceKey, ResourceArbiter,
+    ResourceOwner,
+};
+use zerogravity_model_qwen35::operator_slice::cached_gdn::{
+    CachedGdnSliceInput, CachedGdnSliceOutput,
+};
+use zerogravity_model_qwen35::original_binding::{OriginalWindowOperators, QwenNumericalProfile};
+type Proof<T> = Result<T, Box<dyn Error>>;
+const MIB: u64 = 1024 * 1024;
+const HOST: u64 = 128 * MIB;
+const SOURCE_META: u64 = 64 * MIB;
+const CONTROL: u64 = 8 * MIB;
+const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
+const STAGES: [&str; 9] = [
+    "query",
+    "key",
+    "value",
+    "beta",
+    "decay",
+    "previous_state",
+    "core_f32",
+    "output_bf16",
+    "next_state",
+];
+const INPUT_BYTES: u64 = 16 * (16384 + 3 * 128 + 2) * 4;
+const OUTPUT_BYTES: u64 = 16 * (16384 + 2 * 128) * 4;
+const INPUT_SCRATCH: u64 = 64 * 1024;
+const CORE_SCRATCH: u64 = 16 * (6 * 16384 * 4 + 16 * 128 * 4) + 256 * 1024;
+fn ensure(ok: bool, message: impl Into<String>) -> Proof<()> {
+    if ok {
+        Ok(())
+    } else {
+        Err(message.into().into())
+    }
+}
+fn text(value: &Value) -> Proof<&str> {
+    value
+        .as_str()
+        .ok_or_else(|| "required string missing".into())
+}
+fn sha(bytes: &[u8]) -> String {
+    format!("{:x}", Sha256::digest(bytes))
+}
+fn read(path: &Path, cap: usize) -> Proof<Vec<u8>> {
+    let file = OpenOptions::new()
+        .read(true)
+        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+        .open(path)?;
+    ensure(
+        file.metadata()?.is_file() && file.metadata()?.len() <= cap as u64,
+        "bounded regular input",
+    )?;
+    let mut raw = Vec::new();
+    file.take(cap as u64 + 1).read_to_end(&mut raw)?;
+    ensure(raw.len() <= cap, "bounded input grew")?;
+    Ok(raw)
+}
+fn pinned(path: &Path, pin: &Value, cap: usize) -> Proof<Vec<u8>> {
+    let raw = read(path, cap)?;
+    ensure(
+        pin["bytes"].as_u64() == Some(raw.len() as u64) && pin["sha256"] == sha(&raw),
+        "exact input file digest/length",
+    )?;
+    Ok(raw)
+}
+fn relative(value: &Value) -> Proof<&Path> {
+    let path = Path::new(text(value)?);
+    ensure(
+        !path.as_os_str().is_empty()
+            && path
+                .components()
+                .all(|part| matches!(part, Component::Normal(_))),
+        "contained input name",
+    )?;
+    Ok(path)
+}
+fn write_json(path: &Path, value: &Value) -> Proof<()> {
+    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
+    serde_json::to_writer_pretty(&mut output, value)?;
+    output.write_all(b"\n")?;
+    Ok(())
+}
+fn snapshot(host: &TransferAdmission) -> Proof<DeviceBudgetSnapshot> {
+    Ok(host.arbiter.snapshot(&host.resource)?)
+}
+fn class(row: &DeviceBudgetSnapshot, kind: AllocationClass) -> u64 {
+    row.per_class.get(&kind).copied().unwrap_or(0)
+}
+fn ledger(row: &DeviceBudgetSnapshot) -> Value {
+    json!({"reserved":row.reserved,"remaining":row.remaining,"reservations":row.reservations,
+        "fenced":row.fenced,"per_class":row.per_class.iter()
+            .map(|(k,v)|(format!("{k:?}"),*v)).collect::<BTreeMap<_,_>>()})
+}
+fn same_ledger(host: &TransferAdmission, before: &DeviceBudgetSnapshot) -> Proof<()> {
+    let after = snapshot(host)?;
+    ensure(
+        after.reserved == before.reserved
+            && after.per_class == before.per_class
+            && after.reservations == before.reservations
+            && !after.fenced,
+        "all private failed-call buffers released",
+    )
+}
+fn source(binding: &Value, host: &TransferAdmission) -> Proof<(VerifiedTensorSource, Vec<u8>)> {
+    let root = Path::new(text(&binding["source_directory"])?);
+    let files = binding["source_files"]
+        .as_array()
+        .ok_or("original source file pins")?;
+    let get = |name: &str| -> Proof<&Value> {
+        files
+            .iter()
+            .find(|row| row["path"] == name)
+            .ok_or_else(|| "required source file pin".into())
+    };
+    let config = pinned(&root.join("config.json"), get("config.json")?, MIB as usize)?;
+    let index = pinned(
+        &root.join("model.safetensors.index.json"),
+        get("model.safetensors.index.json")?,
+        MIB as usize,
+    )?;
+    let parsed: Value = serde_json::from_slice(&index)?;
+    let shard_names = parsed["weight_map"]
+        .as_object()
+        .ok_or("complete original index")?
+        .values()
+        .map(text)
+        .collect::<Proof<std::collections::BTreeSet<_>>>()?;
+    ensure(shard_names.len() == 1, "approved single source shard")?;
+    let mut shards = Vec::new();
+    for name in shard_names {
+        relative(&json!(name))?;
+        let expected = get(name)?;
+        let file = OpenOptions::new()
+            .read(true)
+            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+            .open(root.join(name))?;
+        ensure(
+            file.metadata()?.is_file()
+                && file.metadata()?.len() == expected["bytes"].as_u64().ok_or("shard length")?,
+            "same original full regular shard",
+        )?;
+        shards.push(ExpectedSourceShard::new(
+            name.into(),
+            Artifact::from_file(file, Path::new(name))?,
+            expected["bytes"].as_u64().ok_or("shard length")?,
+            text(&expected["sha256"])?.into(),
+        )?);
+    }
+    let source = SourceWeights::prepare_verified_windows(
+        &index,
+        text(&get("model.safetensors.index.json")?["sha256"])?,
+        shards,
+        WindowOpenLimits {
+            max_source_bytes: 5 * 1024 * MIB,
+            max_metadata_bytes: SOURCE_META,
+            max_shards: 1,
+            max_tensors: 1024,
+            scan_chunk_bytes: MIB as usize,
+        },
+        host,
+    )?
+    .open()?;
+    ensure(
+        source.metadata_reserved_bytes() == SOURCE_META && source.source_identity().is_bound(),
+        "sealed original issuer and admitted metadata",
+    )?;
+    Ok((source, config))
+}
+fn row<'a>(inputs: &'a Value, reference: &'a Value, name: &str) -> &'a Value {
+    if STAGES[..6].contains(&name) {
+        &inputs["operands"][name]
+    } else {
+        &reference["targets"][name]
+    }
+}
+fn raw(root: &Path, descriptor: &Value, name: &str) -> Proof<Vec<u8>> {
+    let bf16 = matches!(name, "query" | "key" | "value" | "beta" | "output_bf16");
+    let words = match name {
+        "beta" | "decay" => 16,
+        "previous_state" | "next_state" => 262144,
+        _ => 2048,
+    };
+    ensure(
+        descriptor["dtype"] == if bf16 { "BF16" } else { "F32" },
+        "exact source stage dtype",
+    )?;
+    let bytes = pinned(
+        &root.join(relative(&descriptor["file"])?),
+        descriptor,
+        2 * MIB as usize,
+    )?;
+    ensure(
+        bytes.len() == words * if bf16 { 2 } else { 4 },
+        "complete all-head raw stage",
+    )?;
+    Ok(bytes)
+}
+fn views(value: &CachedGdnSliceOutput) -> Proof<[&[f32]; 9]> {
+    let inputs = value.original_inputs()?;
+    Ok([
+        inputs[0],
+        inputs[1],
+        inputs[2],
+        inputs[3],
+        inputs[4],
+        inputs[5],
+        value.core_f32()?,
+        value.output_bf16()?,
+        value.next_state_key_major()?,
+    ])
+}
+fn check(
+    value: &CachedGdnSliceOutput,
+    inputs: &Value,
+    reference: &Value,
+    reference_root: &Path,
+    out: &Path,
+    after_drop: bool,
+) -> Proof<Value> {
+    ensure(
+        value.layer() == 0 && value.heads() == 16 && value.numerical_profile() == PROFILE,
+        "actual same-engine source profile/layer/head binding",
+    )?;
+    let mut stages = Vec::new();
+    let mut different = 0;
+    let mut words = 0;
+    for (name, actual) in STAGES.into_iter().zip(views(value)?) {
+        let descriptor = row(inputs, reference, name);
+        let expected = raw(reference_root, descriptor, name)?;
+        let bf16 = descriptor["dtype"] == "BF16";
+        let width = if bf16 { 2 } else { 4 };
+        ensure(
+            actual.len() * width == expected.len(),
+            "all actual words before zip",
+        )?;
+        let mut delta = 0;
+        let mut first = Value::Null;
+        for (index, (actual, bytes)) in actual.iter().zip(expected.chunks_exact(width)).enumerate()
+        {
+            let bits = if bf16 {
+                u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16
+            } else {
+                u32::from_le_bytes(bytes.try_into()?)
+            };
+            if actual.to_bits() != bits {
+                delta += 1;
+                if first.is_null() {
+                    first = json!({"word":index,"actual":format!("{:08x}",actual.to_bits()),"expected":format!("{bits:08x}")});
+                }
+            }
+            ensure(
+                actual.is_finite() && (!bf16 || actual.to_bits() & 0xffff == 0),
+                "finite exact original output precision",
+            )?;
+        }
+        let bytes = actual
+            .iter()
+            .flat_map(|v| v.to_bits().to_le_bytes())
+            .collect::<Vec<_>>();
+        let file = format!("{name}.f32le");
+        if !after_drop {
+            OpenOptions::new()
+                .write(true)
+                .create_new(true)
+                .open(out.join(&file))?
+                .write_all(&bytes)?;
+        }
+        stages.push(json!({"stage":name,"words":actual.len(),"different_words":delta,
+            "first_difference":first,"file":file,"bytes":bytes.len(),"sha256":sha(&bytes),"reference":descriptor}));
+        words += actual.len();
+        different += delta;
+    }
+    let record = json!({"after_callers_drop":after_drop,"words":words,"different_words":different,"stages":stages});
+    write_json(&out.join(format!("comparison-{after_drop}.json")), &record)?;
+    ensure(
+        words == 534560 && different == 0,
+        "actual cached signed bits differ; preserve first mismatch",
+    )?;
+    Ok(record)
+}
+fn exercise(
+    binding: &Value,
+    inputs: &Value,
+    reference: &Value,
+    reference_root: &Path,
+    out: &Path,
+) -> Proof<Value> {
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "cached-gdn-archive-replay")?;
+    arbiter.register(key.clone(), HOST)?;
+    let host = TransferAdmission {
+        arbiter,
+        resource: key,
+        owner: ResourceOwner::new("0.8b", "cached-gdn-private-replay")?,
+    };
+    let control = host.arbiter.admit(
+        &host.resource,
+        host.owner.clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Other,
+            bytes: CONTROL,
+        }],
+    )?;
+    let (source, config) = source(binding, &host)?;
+    let files = binding["source_files"]
+        .as_array()
+        .ok_or("source file pins")?;
+    let source_hash = |name: &str| -> Proof<&str> {
+        text(
+            &files
+                .iter()
+                .find(|v| v["path"] == name)
+                .ok_or("source pin")?["sha256"],
+        )
+    };
+    let operators = OriginalWindowOperators::bind(
+        source.clone(),
+        &config,
+        source_hash("config.json")?,
+        source_hash("model.safetensors.index.json")?,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )?;
+    ensure(
+        operators.binding().text_tensor_count() == 320
+            && operators
+                .binding()
+                .config()
+                .geometry()
+                .recurrent_value_heads
+                == Some(16),
+        "genuine complete source 320/H16",
+    )?;
+    let before = snapshot(&host)?;
+    ensure(
+        before.reserved == SOURCE_META + 8 * MIB + CONTROL
+            && before.reservations == 4
+            && class(&before, AllocationClass::Scratch) == SOURCE_META + 8 * MIB
+            && class(&before, AllocationClass::Other) == CONTROL,
+        "source/two-binding/control steady admission",
+    )?;
+    let originals = STAGES[..6]
+        .iter()
+        .map(|name| raw(reference_root, row(inputs, reference, name), name))
+        .collect::<Proof<Vec<_>>>()?;
+    let parts: [&[u8]; 6] = originals
+        .iter()
+        .map(Vec::as_slice)
+        .collect::<Vec<_>>()
+        .try_into()
+        .map_err(|_| "six original raw operands")?;
+    let make_input =
+        || CachedGdnSliceInput::from_bytes(source.source_identity(), PROFILE, 16, parts);
+    ensure(
+        operators
+            .execute_cached_gdn_slice(0, make_input()?, Some(&AtomicBool::new(true)))
+            .is_err(),
+        "pre-cancel refuses",
+    )?;
+    same_ledger(&host, &before)?;
+    ensure(
+        operators
+            .execute_cached_gdn_slice(3, make_input()?, None)
+            .is_err(),
+        "wrong full-attention layer refuses",
+    )?;
+    same_ledger(&host, &before)?;
+    let mut pressure = Vec::new();
+    while snapshot(&host)?.remaining > 0 {
+        pressure.push(source.reserve_workspace(snapshot(&host)?.remaining.min(32 * MIB) as usize)?);
+    }
+    let pressured = snapshot(&host)?;
+    ensure(
+        pressured.reserved == HOST
+            && operators
+                .execute_cached_gdn_slice(0, make_input()?, None)
+                .is_err(),
+        "real original pool refuses before private operand allocation",
+    )?;
+    same_ledger(&host, &pressured)?;
+    drop(pressure);
+    same_ledger(&host, &before)?;
+    let result = operators.execute_cached_gdn_slice(0, make_input()?, None)?;
+    ensure(
+        result.source_identity() == source.source_identity(),
+        "original issuer unchanged",
+    )?;
+    let comparison = check(&result, inputs, reference, reference_root, out, false)?;
+    let live = snapshot(&host)?;
+    ensure(
+        live.reserved
+            == before.reserved + INPUT_BYTES + OUTPUT_BYTES + INPUT_SCRATCH + CORE_SCRATCH
+            && class(&live, AllocationClass::Output) == INPUT_BYTES + OUTPUT_BYTES
+            && class(&live, AllocationClass::TransferStaging) == 0,
+        "all six retained inputs/three results and conservative scratch charged",
+    )?;
+    drop(operators);
+    drop(source);
+    drop(config);
+    let retained = snapshot(&host)?;
+    ensure(
+        retained.reserved
+            == SOURCE_META + CONTROL + INPUT_BYTES + OUTPUT_BYTES + INPUT_SCRATCH + CORE_SCRATCH
+            && class(&retained, AllocationClass::Output) == INPUT_BYTES + OUTPUT_BYTES,
+        "same sealed source/all inputs/results survive caller drop",
+    )?;
+    check(&result, inputs, reference, reference_root, out, true)?;
+    drop(result);
+    ensure(
+        snapshot(&host)?.reserved == CONTROL,
+        "all source/data reservations retire at output last drop",
+    )?;
+    drop(control);
+    let final_row = snapshot(&host)?;
+    ensure(
+        final_row.reserved == 0 && final_row.reservations == 0 && !final_row.fenced,
+        "original arbiter final zero",
+    )?;
+    Ok(
+        json!({"model":"0.8b","heads":16,"imported_words":268320,"output_words":266240,
+        "compared_words":534560,"comparison":comparison,"steady_ledger":ledger(&before),
+        "live_ledger":ledger(&live),"retained_ledger":ledger(&retained),"final_ledger":ledger(&final_row),
+        "pre_cancel":true,"wrong_layer":true,"real_pressure_refusal":true,
+        "external_decay_no_exp_or_initial_or_StateRoot":true}),
+    )
+}
+fn run() -> Proof<Value> {
+    if cfg!(debug_assertions) {
+        return Err("release-only cached operator proof".into());
+    }
+    let args = std::env::args().skip(1).collect::<Vec<_>>();
+    ensure(
+        args.len() == 6 && args[5] == "0.8b",
+        "only source-authentic available .8B cached archive",
+    )?;
+    let input_bytes = read(Path::new(&args[0]), MIB as usize)?;
+    let reference_bytes = read(Path::new(&args[2]), MIB as usize)?;
+    ensure(
+        sha(&input_bytes) == args[1] && sha(&reference_bytes) == args[3],
+        "actual input/reference metadata SHA",
+    )?;
+    let inputs: Value = serde_json::from_slice(&input_bytes)?;
+    let reference: Value = serde_json::from_slice(&reference_bytes)?;
+    for document in [&inputs, &reference] {
+        ensure(
+            document["model"] == "0.8b"
+                && document["heads"] == 16
+                && document["layer"] == 0
+                && document["profile"] == "source-36-f32-restored"
+                && document["teacher_tokens"] == json!([11, 353])
+                && document["external_decay_no_native_exp"] == true
+                && document["source_bindings_sha256"]
+                    == sha(include_bytes!("source-bindings.json")),
+            "authentic cached-only source/profile/phase",
+        )?;
+    }
+    ensure(
+        inputs["schema"] == "zerogravity.cached-gdn.archived-inputs.v1"
+            && reference["schema"] == "zerogravity.cached-gdn.archived-reference.v1"
+            && reference["inputs_sha256"] == args[1],
+        "original public reference links actual operands",
+    )?;
+    ensure(
+        Path::new(&args[0]).parent() == Path::new(&args[2]).parent(),
+        "same immutable raw reference directory",
+    )?;
+    let out = Path::new(&args[4]);
+    std::fs::create_dir(out)?;
+    let bindings: Value = serde_json::from_str(include_str!("source-bindings.json"))?;
+    let result = exercise(
+        &bindings["source"],
+        &inputs,
+        &reference,
+        Path::new(&args[2]).parent().ok_or("reference root")?,
+        out,
+    )?;
+    let affinity = std::fs::read_to_string("/proc/self/status")?
+        .lines()
+        .find(|v| v.starts_with("Cpus_allowed_list:"))
+        .ok_or("actual CPU affinity")?
+        .to_owned();
+    Ok(
+        json!({"status":"PASS_PRIVATE_ARCHIVED_CACHED_GDN_ONLY","result":result,
+        "inputs_sha256":args[1],"reference_sha256":args[3],"actual_affinity":affinity,
+        "full_model_or_live_coefficients_or_StateRoot_or_H48_or_GPU_qualified":false}),
+    )
+}
+fn main() {
+    let result = run();
+    let report = match &result {
+        Ok(value) => value.clone(),
+        Err(error) => json!({"status":"FAIL_UNQUALIFIED","error":error.to_string()}),
+    };
+    let written = std::env::args()
+        .nth(5)
+        .ok_or("output path missing")
+        .and_then(|p| {
+            write_json(&Path::new(&p).join("report.json"), &report)
+                .map_err(|_| "report write failed")
+        });
+    println!("{report}");
+    if result.is_err() || written.is_err() {
+        std::process::exit(1);
+    }
+}
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/owned_lifetime.py
@@ -0,0 +1,91 @@
+"""Exact accepted owned-native leader cleanup functions; no child on import."""
+import os
+import resource
+import signal
+import time
+from contracts import require
+CAP = 4 * 1024 * 1024
+
+
+def unreaped(child):
+    require(child.returncode is None, "leader already reaped")
+    value = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if value is None:
+        return None
+    require(value.si_pid == child.pid, "owned child identity")
+    return value.si_status if value.si_code == os.CLD_EXITED else -value.si_status
+
+
+def cleanup_owned(child, record, errors, *, observe=unreaped, pause=time.sleep):
+    """Independent bounded stages; never signal after the first reap attempt."""
+    def bound():
+        observe(child)
+        require(os.getpgid(child.pid) == child.pid and os.getsid(child.pid) == child.pid,
+                "owned session/group binding")
+        return child.pid
+
+    try:
+        os.killpg(bound(), signal.SIGTERM)
+        record["group_term_before_reap"] = True
+    except ProcessLookupError:
+        record["group_term_absent"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
+    try:
+        grace = time.monotonic() + 0.5
+        while time.monotonic() < grace:
+            observe(child)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # TERM/grace/one KILL error cannot skip the separately bound KILL stage.
+    # Both attempts occur before ANY wait, while the owned leader is unreaped.
+    for attempt in range(2):
+        try:
+            os.killpg(bound(), signal.SIGKILL)
+            record["group_kill_before_reap"] = True
+            break
+        except ProcessLookupError:
+            record["group_kill_absent_before_reap"] = True
+            break
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    # ALWAYS attempt reap even after signal/ownership failures. Popen.wait only
+    # targets this child. An interrupted first wait cannot skip the final wait.
+    # No signal operation occurs below this point, including after timeout.
+    try:
+        record["reaped_exit_code"] = child.wait(timeout=5)
+        record["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned initial reap: " + repr(error))
+    finally:
+        if child.returncode is None:
+            try:
+                record["reaped_exit_code"] = child.wait(timeout=5)
+                record["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final reap: " + repr(error))
+        else:
+            record["reaped_exit_code"] = child.returncode
+            record["leader_reaped"] = True
+    if record.get("leader_reaped"):
+        try:
+            deadline = time.monotonic() + 5
+            while True:
+                try:
+                    os.killpg(child.pid, 0)  # Read-only absence check, NEVER kill after reap.
+                except ProcessLookupError:
+                    record["group_absent"] = True
+                    break
+                require(time.monotonic() < deadline, "owned process group remains")
+                pause(0.05)
+        except BaseException as error:
+            errors.append("owned group absence: " + repr(error))
+
+
+def launch_limits():
+    # Python is single-threaded at Popen; no foreign runtime is imported.
+    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
+    resource.setrlimit(resource.RLIMIT_CPU, (3601, 3601))
+    resource.setrlimit(resource.RLIMIT_FSIZE, (CAP, CAP))
+    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/owned_native.py
@@ -0,0 +1,114 @@
+"""ROOT ONLY native CPU helper orchestration. No Torch/model/container imports."""
+import argparse
+import json
+import os
+from pathlib import Path
+import selectors
+import signal
+import stat
+import subprocess
+import time
+
+from contracts import HERE, frozen, pin, read_json, require, sha, write
+from owned_lifetime import cleanup_owned, launch_limits, unreaped
+
+CAP = 4 * 1024 * 1024
+
+
+def source_files(binding):
+    """ROOT performs complete source scans; never provider/opaque-memory math."""
+    root = Path(binding["source_directory"])
+    rows = []
+    for item in binding["source_files"]:
+        path = root / item["path"]
+        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
+        with os.fdopen(os.open(path, flags), "rb") as opened:
+            before = os.fstat(opened.fileno())
+            require(stat.S_ISREG(before.st_mode) and before.st_size == item["bytes"], "original source regular FD/full length")
+            import hashlib
+            digest = hashlib.file_digest(opened, "sha256").hexdigest()
+            after = os.fstat(opened.fileno())
+        identity = lambda value: [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
+        require(identity(before) == identity(after) and digest == item["sha256"], "complete original source FD digest/identity")
+        rows.append({"path": str(path), "bytes": before.st_size, "sha256": digest,
+                     "fd_identity": identity(before)})
+    return rows
+
+
+def execute(binary, binary_fd, inputs, inputs_sha, reference, reference_sha, output, model, cpus, freeze, source_before):
+    argv = [str(binary),str(inputs),inputs_sha,str(reference),reference_sha,str(output),model]
+    environment = {key: value for key, value in os.environ.items()
+                   if key in ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "XDG_CACHE_HOME")}
+    environment.update(RAYON_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
+                       CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void",
+                       ATEN_CPU_CAPABILITY="avx2")
+    record = {"status": "UNQUALIFIED", "argv": argv, "freeze_manifest_sha256": freeze,
+              "binary_sha256": sha(binary), "cpu_affinity": cpus, "scope": "private archived original cached GDN only",
+              "address_space_cap_bytes": 512 * 1024**2, "wall_cap_seconds": 900,
+              "log_cap_bytes": CAP, "source_before": source_before,
+              "no_model_or_session_or_GPU_execution": True}
+    child = selector = pipe = log = None
+    errors, primary, eof, retained = [], None, False, 0
+    started = time.monotonic()
+    try:
+        log = (output.parent / (model + "-lifecycle") / "execution.log").open("xb")
+        child = subprocess.Popen(argv, executable=f"/proc/self/fd/{binary_fd}", pass_fds=(binary_fd,),
+            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
+            env=environment, start_new_session=True, preexec_fn=launch_limits)
+        # Every post-spawn operation is covered by cleanup below.
+        record["owned_cli_pid"] = child.pid
+        pipe = child.stdout
+        os.set_blocking(pipe.fileno(), False)
+        selector = selectors.DefaultSelector()
+        selector.register(pipe, selectors.EVENT_READ)
+
+        def drain(timeout):
+            nonlocal retained, eof
+            for key, _ in selector.select(timeout):
+                try:
+                    data = os.read(key.fd, 65536)
+                except BlockingIOError:
+                    continue
+                if not data:
+                    selector.unregister(key.fileobj)
+                    eof = True
+                else:
+                    require(retained + len(data) <= CAP, "bounded child combined log")
+                    log.write(data)
+                    retained += len(data)
+        while True:
+            drain(0.05)
+            status = unreaped(child)
+            if status is not None:
+                record["exit_code"] = status
+                break
+            require(time.monotonic() - started < 900, "finite helper deadline")
+    except BaseException as error:
+        primary = repr(error)
+    finally:
+        if child is not None:
+            cleanup_owned(child, record, errors)
+        if selector is not None:
+            try:
+                deadline = time.monotonic() + 2
+                while not eof and time.monotonic() < deadline:
+                    drain(0.05)
+                require(eof, "owned pipe EOF deadline")
+            except BaseException as error:
+                errors.append("final drain: " + repr(error))
+        for name, owned in (("selector", selector), ("pipe", pipe), ("log", log)):
+            if owned is not None:
+                try:
+                    owned.close()
+                    record[name + "_closed"] = True
+                except BaseException as error:
+                    errors.append(name + " close: " + repr(error))
+        record.update(primary_error=primary, cleanup_errors=errors, pipe_eof=eof, log_bytes=retained,
+                      elapsed_seconds_not_benchmark=time.monotonic() - started)
+    success = primary is None and not errors and record.get("exit_code") == record.get("reaped_exit_code") == 0 \
+        and record.get("leader_reaped") and record.get("group_absent") and eof
+    record["status"] = "CHILD_EXECUTED_REQUIRES_VALIDATION" if success else "FAIL_UNQUALIFIED"
+    write(output.parent / (model + "-lifecycle") / "invocation.json", record)
+    require(success, "owned native helper failed; preserve fresh attempt")
+    return record
+
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/run_replay.py
@@ -0,0 +1,68 @@
+"""ROOT ONLY fresh owned native child; no provider/model/operator on import."""
+import argparse
+import hashlib
+import os
+from pathlib import Path
+import signal
+from contracts import HERE, frozen, implementation, read_json, require, sha, write
+from export_reference import validate as validate_archived
+from owned_native import execute, source_files
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--binary', type=Path, required=True)
+    parser.add_argument('--binary-sha256', required=True)
+    parser.add_argument('--reference-directory', type=Path, required=True)
+    parser.add_argument('--implementation', type=Path, required=True)
+    parser.add_argument('--new-output', type=Path, required=True)
+    args = parser.parse_args()
+    def deadline(_signal, _frame):
+        raise TimeoutError('finite original source audit and cached operator proof deadline')
+    signal.signal(signal.SIGALRM, deadline)
+    signal.alarm(2400)
+    freeze = frozen()
+    source_attestation = implementation(args.implementation)
+    archived = validate_archived(args.reference_directory)
+    require(args.reference_directory.resolve().parent == HERE and
+            args.new_output.resolve().parent == HERE, 'own bounded reference/attempt only')
+    require(not args.binary.is_symlink() and args.binary.is_file() and
+            sha(args.binary) == args.binary_sha256, 'ROOT reviewed actual native binary')
+    cpus = sorted(os.sched_getaffinity(0))[:4]
+    require(len(cpus) == 4, 'four available CPUs')
+    os.sched_setaffinity(0, cpus)
+    args.new_output.mkdir(mode=0o700)
+    (args.new_output/'0.8b-lifecycle').mkdir(mode=0o700)
+    inputs, reference = args.reference_directory/'inputs.json', args.reference_directory/'reference.json'
+    binding = read_json(HERE/'source-bindings.json')['source']
+    fd = os.open(args.binary.resolve(), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
+    try:
+        with os.fdopen(os.dup(fd), 'rb') as opened:
+            require(hashlib.file_digest(opened, 'sha256').hexdigest() == args.binary_sha256, 'actual exec-FD SHA')
+        before = source_files(binding)
+        output = args.new_output/'0.8b'
+        execute(args.binary.resolve(), fd, inputs.resolve(), sha(inputs), reference.resolve(), sha(reference),
+                output.resolve(), '0.8b', cpus, freeze, before)
+        after = source_files(binding)
+        require(before == after, 'all full original source files unchanged')
+        require(implementation(args.implementation) == source_attestation and frozen() == freeze and
+                sha(args.binary) == args.binary_sha256 and validate_archived(args.reference_directory) == archived,
+                'all actual build/source/public-reference bindings unchanged')
+        write(output/'source-after.json', {'source_before':before,'source_after':after,'all_equal':True})
+        report = read_json(output/'report.json')
+        require(report['status'] == 'PASS_PRIVATE_ARCHIVED_CACHED_GDN_ONLY' and
+                report['result']['output_words'] == 266240 and report['result']['compared_words'] == 534560,
+                'complete actual cached results require independent raw validation')
+        write(args.new_output/'completed.json', {
+            'status':'EXECUTED_REQUIRES_INDEPENDENT_RAW_VALIDATION',
+            'freeze_manifest_sha256':freeze,'binary_sha256':args.binary_sha256,
+            'ROOT_build_source_attestation_sha256':source_attestation,
+            'archived_reference_validation':archived,
+            'report_sha256':sha(output/'report.json'),
+            'source_after_sha256':sha(output/'source-after.json'),
+            'invocation_sha256':sha(args.new_output/'0.8b-lifecycle/invocation.json'),
+            'full_model_or_live_decay_or_StateRoot_or_H48_or_GPU_qualified':False})
+    finally:
+        os.close(fd)
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/source-bindings.json
@@ -0,0 +1,172 @@
+{
+  "capture_manifest": {
+    "bytes": 110084,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/manifest.json",
+    "sha256": "6ad1051588c2f2239577f512e27597397c0fcd8929f4082ead77ccedfc81743b"
+  },
+  "capture_validation": {
+    "bytes": 1011,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/validation.json",
+    "sha256": "95ee2462a13ac6d92c856a94af1f090f45ffbefbf6b326fe122d8b39ed77dbfe"
+  },
+  "heads": 16,
+  "layer": 0,
+  "model": "0.8b",
+  "producer_scope": "Actual archived original public cached call; native decay remains external observed exp(g), never recomputed",
+  "profile": "source-36-f32-restored",
+  "schema": "zerogravity.cached-gdn.archived-source-bindings.v1",
+  "source": {
+    "capture_manifest": {
+      "bytes": 310033,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/manifest.json",
+      "sha256": "e007d5e8058a218614a6135d829d35ad5a3b72d4dbdc4916a39498426057cd76"
+    },
+    "heads": 16,
+    "hidden": 1024,
+    "modeling_source": {
+      "path": "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py",
+      "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+    },
+    "repo": "Qwen/Qwen3.5-0.8B",
+    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
+    "runtime": {
+      "cpu_capability": "AVX2",
+      "cuda_available": false,
+      "default_dtype": "torch.float32",
+      "machine": "x86_64",
+      "mkldnn_enabled": true,
+      "processor": "x86_64",
+      "python": "3.12.3",
+      "safetensors": "0.8.0",
+      "torch": "2.13.0+cu129",
+      "torch_interop_threads": 1,
+      "torch_num_threads": 4,
+      "transformers": "5.17.0",
+      "vllm": "0.31.0+cu129"
+    },
+    "source_directory": "/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17",
+    "source_files": [
+      {
+        "bytes": 11544,
+        "path": "LICENSE",
+        "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
+      },
+      {
+        "bytes": 7755,
+        "path": "chat_template.jinja",
+        "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
+      },
+      {
+        "bytes": 2907,
+        "path": "config.json",
+        "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
+      },
+      {
+        "bytes": 3353259,
+        "path": "merges.txt",
+        "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
+      },
+      {
+        "bytes": 1746942600,
+        "path": "model.safetensors-00001-of-00001.safetensors",
+        "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
+      },
+      {
+        "bytes": 50900,
+        "path": "model.safetensors.index.json",
+        "sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e"
+      },
+      {
+        "bytes": 390,
+        "path": "preprocessor_config.json",
+        "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
+      },
+      {
+        "bytes": 12807982,
+        "path": "tokenizer.json",
+        "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
+      },
+      {
+        "bytes": 16709,
+        "path": "tokenizer_config.json",
+        "sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
+      },
+      {
+        "bytes": 385,
+        "path": "video_preprocessor_config.json",
+        "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
+      },
+      {
+        "bytes": 6722759,
+        "path": "vocab.json",
+        "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
+      }
+    ],
+    "steps": [
+      {
+        "input_token": 11,
+        "metadata": {
+          "bytes": 53537,
+          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-0/manifest.json",
+          "sha256": "d35aaea5180947a4efe02704563cc90a4ae5a0708af60e982740dbdb2e2e2958"
+        },
+        "observed": {
+          "bytes": 1646080,
+          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-0/observed-intermediates.safetensors",
+          "pin_origin": "existing actual capture manifest; author did not hash payload",
+          "sha256": "87d6434fae1a29bac9b308fda8b0054a12bcfa38b01a5a44041fce9443006201"
+        },
+        "raw": {
+          "bytes": 20272592,
+          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-0/raw-official.safetensors",
+          "pin_origin": "existing actual capture manifest; author did not hash payload",
+          "sha256": "d46a4d2115aafc5ff74bc5807f3174900f9f841409418aef6852825c4dabb478"
+        }
+      },
+      {
+        "input_token": 353,
+        "metadata": {
+          "bytes": 54310,
+          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-1/manifest.json",
+          "sha256": "020ac70910386cb3c8e91d1b99d43ad36f4f9b89852c1b5f8a104f74cc861577"
+        },
+        "observed": {
+          "bytes": 1609216,
+          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-1/observed-intermediates.safetensors",
+          "pin_origin": "existing actual capture manifest; author did not hash payload",
+          "sha256": "c730449ea0cc432696f5adabc26efcf9461479a19bd533301f730b7ce060ef27"
+        },
+        "raw": {
+          "bytes": 20284880,
+          "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/source-36-f32-restored/step-1/raw-official.safetensors",
+          "pin_origin": "existing actual capture manifest; author did not hash payload",
+          "sha256": "8d4f50dc3c4ebcefed24b178a66aa86543a7dbf652158382accc8fc5b93f781b"
+        }
+      }
+    ]
+  },
+  "step0_archive": {
+    "bytes": 12453400,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-0/actual-gdn-frames.safetensors",
+    "sha256": "a493dd31bf1c511edee9198ac4e8eb30f7141f79207d2ce836dfb8ef02ad0450"
+  },
+  "step0_metadata": {
+    "bytes": 46710,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-0/manifest.json",
+    "sha256": "ea855e806fa7a0001b943d3dc56ca5f73a35814bda5f15a79c87418572731947"
+  },
+  "step1_archive": {
+    "bytes": 6472400,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-1/actual-gdn-frames.safetensors",
+    "sha256": "c8b7c72d0644c418dfa5c74e974678f7ee76e3d7cc9e83a7c157c3a9f836c0fa"
+  },
+  "step1_metadata": {
+    "bytes": 41192,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-1/manifest.json",
+    "sha256": "f146eba12662fc9acf7d1cb647244024c2c3747fb9e90d241afeb8a6a9beb8a7"
+  },
+  "tokens": [
+    11,
+    353
+  ]
+}
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/test_contracts.py
@@ -0,0 +1,47 @@
+"""Prepared source-free byte-validation fixtures; no provider/model calls."""
+import json
+from pathlib import Path
+import struct
+import tempfile
+import unittest
+from contracts import archive, digest, finite, read_json, widened
+from validate_replay import affinity
+
+class ByteContracts(unittest.TestCase):
+    def test_signed_zero_subnormal_and_bf16_transport(self):
+        raw = bytes.fromhex('008001000000')
+        finite(raw, 'BF16')
+        self.assertEqual(widened(raw, 'BF16'), bytes.fromhex('000000800000010000000000'))
+        fp = bytes.fromhex('0000008001000000')
+        finite(fp, 'F32')
+        self.assertEqual(widened(fp, 'F32'), fp)
+        for dtype, bad in [('BF16',bytes.fromhex('807f')),('F32',bytes.fromhex('000080ff'))]:
+            with self.assertRaises(ValueError):
+                finite(bad, dtype)
+
+    def test_actual_archive_extent_and_trailing_byte_refusals(self):
+        payload = bytes.fromhex('0000008001000000')
+        header = json.dumps({'x':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}).encode()
+        raw = struct.pack('<Q',len(header))+header+payload
+        with tempfile.TemporaryDirectory() as root:
+            path = Path(root)/'small.safetensors'
+            path.write_bytes(raw)
+            pin = {'path':str(path),'bytes':len(raw),'sha256':digest(raw)}
+            self.assertEqual(archive(pin)['x']['raw'], payload)
+            path.write_bytes(raw+b'\0')
+            with self.assertRaises(ValueError):
+                archive(dict(pin,bytes=len(raw)+1,sha256=digest(raw+b'\0')))
+
+    def test_duplicate_metadata_and_actual_cpu_binding(self):
+        with tempfile.TemporaryDirectory() as root:
+            path = Path(root)/'metadata.json'
+            path.write_text('{"x":1,"x":2}')
+            with self.assertRaises(ValueError):
+                read_json(path)
+        self.assertEqual(affinity('Cpus_allowed_list:\t0-3'), [0,1,2,3])
+        for bad in ('Cpus_allowed_list:\t0-2','Cpus_allowed_list:\t0-4','Cpus_allowed_list:\t0,1,1,2'):
+            with self.assertRaises(ValueError):
+                affinity(bad)
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ b/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/validate_replay.py
@@ -0,0 +1,111 @@
+"""ROOT ONLY independent full signed-bit/source/ledger/native-lifetime check."""
+import argparse
+import json
+from pathlib import Path
+from contracts import INPUTS, OUTPUTS, STAGES, frozen, implementation, raw_file, read, read_json, require, sha, widened, write
+from export_reference import validate as validate_archived
+
+def affinity(text):
+    require(type(text) is str and text.startswith('Cpus_allowed_list:'), 'actual proc CPU affinity')
+    values = []
+    for item in text.split(':',1)[1].strip().split(','):
+        span = item.split('-')
+        require(1 <= len(span) <= 2 and all(v.isdecimal() for v in span), 'strict CPU span')
+        lo, hi = int(span[0]), int(span[-1])
+        require(0 <= lo <= hi < 4096, 'bounded actual CPU span')
+        values.extend(range(lo,hi+1))
+    require(len(values) == 4 and values == sorted(set(values)), 'four actual distinct CPUs')
+    return values
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--output', type=Path, required=True)
+    parser.add_argument('--reference-directory', type=Path, required=True)
+    parser.add_argument('--implementation', type=Path, required=True)
+    args = parser.parse_args()
+    freeze = frozen()
+    archived = validate_archived(args.reference_directory)
+    build_source = implementation(args.implementation)
+    done = read_json(args.output/'completed.json')
+    require(done['status'] == 'EXECUTED_REQUIRES_INDEPENDENT_RAW_VALIDATION' and
+            done['freeze_manifest_sha256'] == freeze and
+            done['ROOT_build_source_attestation_sha256'] == build_source and
+            done['archived_reference_validation'] == archived and
+            done['full_model_or_live_decay_or_StateRoot_or_H48_or_GPU_qualified'] is False,
+            'exact original public/source/build receipt closure')
+    directory, lifecycle = args.output/'0.8b', args.output/'0.8b-lifecycle'
+    for file, key in [('report.json','report_sha256'),('source-after.json','source_after_sha256')]:
+        require(sha(directory/file) == done[key], 'exact native result/source receipt')
+    require(sha(lifecycle/'invocation.json') == done['invocation_sha256'], 'exact original owned-child receipt')
+    report = read_json(directory/'report.json')
+    invocation = read_json(lifecycle/'invocation.json')
+    source = read_json(directory/'source-after.json')
+    require(report['status'] == 'PASS_PRIVATE_ARCHIVED_CACHED_GDN_ONLY' and
+            report['inputs_sha256'] == archived['inputs_sha256'] and
+            report['reference_sha256'] == archived['reference_sha256'] and
+            report['full_model_or_live_coefficients_or_StateRoot_or_H48_or_GPU_qualified'] is False,
+            'pure archived cached operator only')
+    require(invocation['status'] == 'CHILD_EXECUTED_REQUIRES_VALIDATION' and
+            invocation['binary_sha256'] == done['binary_sha256'] and invocation['freeze_manifest_sha256'] == freeze and
+            invocation['primary_error'] is None and invocation['cleanup_errors'] == [] and
+            invocation['exit_code'] == invocation['reaped_exit_code'] == 0 and
+            all(invocation[k] is True for k in ('leader_reaped','group_absent','pipe_eof','selector_closed','pipe_closed','log_closed')),
+            'actual independently retired original owned child/PGID/pipe')
+    require(affinity(report['actual_affinity']) == invocation['cpu_affinity'], 'actual four-CPU native execution')
+    require(source['all_equal'] is True and source['source_before'] == source['source_after'] == invocation['source_before'],
+            'actual whole original source unchanged')
+    result = report['result']
+    require(result['model'] == '0.8b' and result['heads'] == 16 and result['imported_words'] == 268320 and
+            result['output_words'] == 266240 and result['compared_words'] == 534560 and
+            all(result[k] is True for k in ('pre_cancel','wrong_layer','real_pressure_refusal','external_decay_no_exp_or_initial_or_StateRoot')),
+            'all H16 stages and actual negative receipts')
+    mib = 1024*1024
+    input_bytes, output_bytes = 268320*4, 266240*4
+    scratch = 65536 + 16*(6*16384*4+16*128*4)+256*1024
+    steady, live, retained, final = [result[k+'_ledger'] for k in ('steady','live','retained','final')]
+    require(steady['reserved'] == 80*mib and steady['reservations'] == 4 and
+            steady['per_class'].get('Scratch',0) == 72*mib and steady['per_class'].get('Other',0) == 8*mib,
+            'source/two original binding/control reservations')
+    require(live['reserved'] == 80*mib+input_bytes+output_bytes+scratch and
+            retained['reserved'] == 72*mib+input_bytes+output_bytes+scratch,
+            'aggregate payload/scratch/source owners remain through caller drop')
+    for ledger in (live,retained):
+        require(ledger['per_class'].get('Output',0) == input_bytes+output_bytes and
+                ledger['per_class'].get('TransferStaging',0) == 0 and ledger['fenced'] is False,
+                'all actual cached payload reservations, no transfer/root work')
+    require(final['reserved'] == final['reservations'] == 0 and final['fenced'] is False, 'actual last-drop zero')
+    inputs = read_json(args.reference_directory/'inputs.json')
+    reference = read_json(args.reference_directory/'reference.json')
+    comparison = result['comparison']
+    require(comparison['after_callers_drop'] is False and comparison['words'] == 534560 and
+            comparison['different_words'] == 0 and [v['stage'] for v in comparison['stages']] == list(STAGES),
+            'complete actual stage ordering/counts')
+    total = 0
+    files = {'report.json','source-after.json','comparison-false.json','comparison-true.json'}
+    for record in comparison['stages']:
+        name = record['stage']
+        row = inputs['operands'][name] if name in INPUTS else reference['targets'][name]
+        expected = widened(raw_file(args.reference_directory,row),row['dtype'])
+        require(record['reference'] == row and record['different_words'] == 0 and
+                record['first_difference'] is None and record['words']*4 == len(expected) and
+                record['file'] == name+'.f32le' and read(directory/record['file'],4*mib,record) == expected,
+                'independent every signed actual/expected cached word: '+name)
+        files.add(record['file'])
+        total += record['words']
+    for dropped in (False,True):
+        record = read_json(directory/('comparison-'+str(dropped).lower()+'.json'))
+        require(record['after_callers_drop'] is dropped and record['words'] == 534560 and
+                record['different_words'] == 0 and record['stages'] == comparison['stages'],
+                'same full archived words remain accessible after source caller drop')
+    require(total == 534560 and {p.name for p in directory.iterdir()} == files and frozen() == freeze,
+            'all import/output words and native artifact tree accounted')
+    value = {'status':'PASS_ARCHIVED_H16_CACHED_GDN_PRIVATE_SLICE_ONLY',
+             'imported_words':268320,'independent_output_words':266240,'all_compared_words':total,
+             'completed_sha256':sha(args.output/'completed.json'),'freeze_manifest_sha256':freeze,
+             'ROOT_build_source_attestation_sha256':build_source,
+             'full_model_or_live_decay_or_StateRoot_or_H48_or_GPU_qualified':False}
+    write(args.output/'validation.json',value)
+    print(json.dumps(value,sort_keys=True))
+
+if __name__ == '__main__':
+    main()

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/frozen-prepared/GPU_REFUSAL.patch

Original UTF-8 bytes: 1917; SHA256 `0a49ceafd9426512204d13bf4686c7a2f50ea01a22c80e3e7f7a56c8a26a6897`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -349,6 +349,8 @@
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
+                        | PlanOperation::TypedCachedGdnCore { .. }
+                        | PlanOperation::TypedCachedGdnBf16Return { .. }
                 )
             })
         {
@@ -1431,7 +1433,9 @@
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
-                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedCausalConvSilu { .. }
+                | PlanOperation::TypedCachedGdnCore { .. }
+                | PlanOperation::TypedCachedGdnBf16Return { .. } => Err(Error::Unsupported(
                     "resident typed operation has no qualified lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -839,7 +839,9 @@
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
                 | PlanOperation::TypedOrdinaryRms { .. }
-                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedCausalConvSilu { .. }
+                | PlanOperation::TypedCachedGdnCore { .. }
+                | PlanOperation::TypedCachedGdnBf16Return { .. } => Err(Error::Unsupported(
                     "resident batch has no qualified typed lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/frozen-prepared/NEW_DELTA.patch

Original UTF-8 bytes: 45177; SHA256 `061cfc0d9fe7842713f6b549ec9f468a87936ba88fc32e74a4b8a52b01a91e2e`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-core/src/accelerator/program.rs
+++ b/crates/zerogravity-core/src/accelerator/program.rs
@@ -12,6 +12,8 @@
 use std::sync::Arc;
 use std::sync::atomic::{AtomicU64, Ordering};
 
+mod cached_gdn;
+pub use cached_gdn::{CachedGdnContract, CachedGdnOperands, CachedGdnValues};
 mod gdn_conv;
 pub use gdn_conv::{CausalConvSiluContract, CausalConvValues};
 mod typed;
@@ -282,6 +284,18 @@
             "backend does not implement explicit ordinary RMS contract".into(),
         ))
     }
+    /// Cached single-token arithmetic only; decay is an explicit external input.
+    fn cached_gdn_profiled(
+        &mut self,
+        _source: SourceIdentity,
+        _heads: usize,
+        _input: &CachedGdnOperands<Self::Value>,
+        _contract: CachedGdnContract,
+    ) -> Result<CachedGdnValues<Self::Value>> {
+        Err(Error::Unsupported(
+            "backend does not implement explicit cached GDN contract".into(),
+        ))
+    }
     /// Pure original convolution window; no state owner or implicit fallback.
     fn causal_conv_silu_profiled(
         &mut self,
@@ -466,6 +480,16 @@
     TypedProject {
         tensor: SourceTensor,
         contract: ProjectionContract,
+    },
+    TypedCachedGdnCore {
+        source: SourceIdentity,
+        heads: usize,
+        contract: CachedGdnContract,
+    },
+    TypedCachedGdnBf16Return {
+        source: SourceIdentity,
+        heads: usize,
+        contract: CachedGdnContract,
     },
     TypedCausalConvSilu {
         parameter: SourceParameter,
@@ -818,6 +842,16 @@
                 contract.validate(parameter, inputs[0].elements)?;
                 Ok(ActivationPrecision::Bf16WidenedF32)
             }
+            PlanOperation::TypedCachedGdnCore {
+                source,
+                heads,
+                contract,
+            } => cached_gdn::result_precision(self, *source, *heads, *contract, inputs, false),
+            PlanOperation::TypedCachedGdnBf16Return {
+                source,
+                heads,
+                contract,
+            } => cached_gdn::result_precision(self, *source, *heads, *contract, inputs, true),
             PlanOperation::TypedCausalConvSilu {
                 parameter,
                 contract,
@@ -839,6 +873,8 @@
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
                         | PlanOperation::TypedCausalConvSilu { .. }
+                        | PlanOperation::TypedCachedGdnCore { .. }
+                        | PlanOperation::TypedCachedGdnBf16Return { .. }
                 ) =>
             {
                 Ok(ActivationPrecision::F32)
@@ -1105,6 +1141,15 @@
             input.shape,
             None,
         )
+    }
+    fn cached_gdn_profiled(
+        &mut self,
+        source: SourceIdentity,
+        heads: usize,
+        input: &CachedGdnOperands<PlanValue>,
+        contract: CachedGdnContract,
+    ) -> Result<CachedGdnValues<PlanValue>> {
+        cached_gdn::record(self, source, heads, input, contract)
     }
     fn causal_conv_silu_profiled(
         &mut self,
--- /dev/null
+++ b/crates/zerogravity-core/src/accelerator/program/cached_gdn.rs
@@ -0,0 +1,301 @@
+//! Pure cached-token core; explicit decay boundary, never initial chunk64.
+use super::*;
+
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum CachedGdnContract {
+    TorchCpuBf16Avx2Width128,
+}
+impl CachedGdnContract {
+    pub fn validate(self, heads: usize, lengths: &[usize; 6]) -> Result<(usize, usize)> {
+        let vector = heads.checked_mul(128);
+        let state = vector.and_then(|n| n.checked_mul(128));
+        let (Some(vector), Some(state)) = (vector, state) else {
+            return Err(Error::Limit("cached GDN geometry overflow".into()));
+        };
+        if heads == 0 || state > 1_048_576 {
+            return Err(Error::Limit("cached GDN head/element bound".into()));
+        }
+        if *lengths != [vector, vector, vector, heads, heads, state] {
+            return Err(Error::Malformed(
+                "cached GDN explicit operand geometry".into(),
+            ));
+        }
+        Ok((vector, state))
+    }
+}
+
+/// Q/K are already repeated to value heads. No implicit head/layout conversion.
+#[derive(Clone)]
+pub struct CachedGdnOperands<V> {
+    pub query: V,
+    pub key: V,
+    pub value: V,
+    pub beta: V,
+    /// Explicit finite FP32 exp(g); this type grants no coefficient eligibility.
+    pub decay: V,
+    /// FP32 [head,key128,value128], distinct from canonical value-major state.
+    pub previous_state: V,
+}
+#[derive(Clone)]
+pub struct CachedGdnValues<V> {
+    pub core_f32: V,
+    pub output_bf16: V,
+    pub next_state: V,
+}
+
+pub(super) fn result_precision(
+    builder: &PlanBuilder,
+    source: SourceIdentity,
+    heads: usize,
+    contract: CachedGdnContract,
+    inputs: &[PlanValue],
+    bf16_return: bool,
+) -> Result<ActivationPrecision> {
+    if !builder.operator_slice
+        || builder.source != source
+        || builder.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    {
+        return Err(Error::Unsupported(
+            "cached GDN source/profile/private slice".into(),
+        ));
+    }
+    if bf16_return {
+        let vector = heads
+            .checked_mul(128)
+            .filter(|_| heads > 0 && heads <= 64)
+            .ok_or_else(|| Error::Limit("cached GDN return head bound".into()))?;
+        if inputs.len() != 1
+            || inputs[0].elements != vector
+            || inputs[0].precision != ActivationPrecision::F32
+        {
+            return Err(Error::Malformed("cached GDN return F32 core".into()));
+        }
+        Ok(ActivationPrecision::Bf16WidenedF32)
+    } else {
+        let lengths: [usize; 6] = inputs
+            .iter()
+            .map(|v| v.elements)
+            .collect::<Vec<_>>()
+            .try_into()
+            .map_err(|_| Error::Malformed("cached GDN operand arity".into()))?;
+        contract.validate(heads, &lengths)?;
+        if inputs[..4]
+            .iter()
+            .any(|v| v.precision != ActivationPrecision::Bf16WidenedF32)
+            || inputs[4..]
+                .iter()
+                .any(|v| v.precision != ActivationPrecision::F32)
+        {
+            return Err(Error::Unsupported(
+                "cached GDN BF16/F32 operand precision".into(),
+            ));
+        }
+        Ok(ActivationPrecision::F32)
+    }
+}
+
+pub(super) fn record(
+    builder: &mut PlanBuilder,
+    source: SourceIdentity,
+    heads: usize,
+    input: &CachedGdnOperands<PlanValue>,
+    contract: CachedGdnContract,
+) -> Result<CachedGdnValues<PlanValue>> {
+    let inputs = [
+        input.query,
+        input.key,
+        input.value,
+        input.beta,
+        input.decay,
+        input.previous_state,
+    ];
+    let (vector, state) = contract.validate(heads, &inputs.map(|v| v.elements))?;
+    // Existing record validates ALL original builder/value/stride memberships
+    // before recording either pure node. No record precision policy changes.
+    let values = builder.record(
+        "original.cached_gdn.core",
+        PlanOperation::TypedCachedGdnCore {
+            source,
+            heads,
+            contract,
+        },
+        &inputs,
+        &[ValueShape::Vector(vector), ValueShape::Vector(state)],
+        None,
+    )?;
+    let [core_f32, next_state]: [PlanValue; 2] = values
+        .try_into()
+        .map_err(|_| Error::State("cached GDN core output arity".into()))?;
+    let output_bf16 = builder.one(
+        "original.cached_gdn.return_bf16",
+        PlanOperation::TypedCachedGdnBf16Return {
+            source,
+            heads,
+            contract,
+        },
+        &[core_f32],
+        ValueShape::Vector(vector),
+        None,
+    )?;
+    Ok(CachedGdnValues {
+        core_f32,
+        output_bf16,
+        next_state,
+    })
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    fn input(builder: &mut PlanBuilder, n: usize, bf16: bool) -> PlanValue {
+        if bf16 {
+            builder
+                .embed_profiled(
+                    SourceTensor {
+                        source: builder.source,
+                        name: format!("synthetic.operand.{n}"),
+                        kind: QuantKind::Bf16,
+                        axes: MatrixAxes::OutputInput,
+                        columns: n,
+                        rows: 1,
+                    },
+                    PlanBuilder::inputs().token,
+                    EmbeddingContract::Bf16OutputInput,
+                )
+                .unwrap()
+        } else {
+            // Test-only F32 operand metadata; no executable source/model claim.
+            builder
+                .one(
+                    "synthetic.fp32_operand",
+                    PlanOperation::Observe {
+                        layer: 0,
+                        stage: "synthetic_fp32_operand",
+                    },
+                    &[],
+                    ValueShape::Vector(n),
+                    None,
+                )
+                .unwrap()
+        }
+    }
+    fn fixture(b: &mut PlanBuilder) -> CachedGdnOperands<PlanValue> {
+        CachedGdnOperands {
+            query: input(b, 128, true),
+            key: input(b, 128, true),
+            value: input(b, 128, true),
+            beta: input(b, 1, true),
+            decay: input(b, 1, false),
+            previous_state: input(b, 16384, false),
+        }
+    }
+    fn builder() -> PlanBuilder {
+        PlanBuilder::new_operator_slice(
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        )
+        .unwrap()
+    }
+    #[test]
+    fn cached_core_and_return_have_separate_precisions_without_state_effects() {
+        let mut b = builder();
+        let operands = fixture(&mut b);
+        let source = b.source;
+        let result = b
+            .cached_gdn_profiled(
+                source,
+                1,
+                &operands,
+                CachedGdnContract::TorchCpuBf16Avx2Width128,
+            )
+            .unwrap();
+        assert_eq!(result.core_f32.precision(), ActivationPrecision::F32);
+        assert_eq!(result.next_state.precision(), ActivationPrecision::F32);
+        assert_eq!(
+            result.output_bf16.precision(),
+            ActivationPrecision::Bf16WidenedF32
+        );
+        let recorded = b.finish_operator_slice(result.output_bf16).unwrap();
+        assert_eq!(recorded.plan().nodes().len(), 8);
+        assert!(recorded.plan().required_domains().is_empty());
+        assert!(recorded.plan().proposed_domains().is_empty());
+        assert_eq!(recorded.plan().frontier_delta(), 0);
+    }
+    #[test]
+    fn foreign_inputs_and_valid_geometry_strides_refuse_before_first_core_node() {
+        let mut b = builder();
+        let mut operands = fixture(&mut b);
+        let source = b.source;
+        let mut other = builder();
+        operands.decay = input(&mut other, 1, false);
+        assert!(
+            b.cached_gdn_profiled(
+                source,
+                1,
+                &operands,
+                CachedGdnContract::TorchCpuBf16Avx2Width128
+            )
+            .is_err()
+        );
+        assert_eq!(b.nodes.len(), 6);
+        operands.decay = input(&mut b, 1, false);
+        let base = input(&mut b, 256, true);
+        operands.query = b
+            .view(
+                &base,
+                TensorView {
+                    offset: 0,
+                    rows: 128,
+                    width: 1,
+                    row_stride: 2,
+                },
+            )
+            .unwrap();
+        let error = b
+            .cached_gdn_profiled(
+                source,
+                1,
+                &operands,
+                CachedGdnContract::TorchCpuBf16Avx2Width128,
+            )
+            .err()
+            .unwrap();
+        assert!(error.to_string().contains("contiguous"));
+        let nodes = b.nodes.len();
+        operands.query = b.contiguous(&operands.query).unwrap();
+        let decay = operands.decay;
+        operands.decay = operands.beta;
+        assert!(
+            b.cached_gdn_profiled(
+                source,
+                1,
+                &operands,
+                CachedGdnContract::TorchCpuBf16Avx2Width128
+            )
+            .is_err()
+        );
+        assert_eq!(b.nodes.len(), nodes + 1);
+        operands.decay = decay;
+        b.cached_gdn_profiled(
+            source,
+            1,
+            &operands,
+            CachedGdnContract::TorchCpuBf16Avx2Width128,
+        )
+        .unwrap();
+    }
+    #[test]
+    fn closed_bounds_cover48heads_without_head_padding() {
+        let c = CachedGdnContract::TorchCpuBf16Avx2Width128;
+        assert_eq!(
+            c.validate(48, &[6144, 6144, 6144, 48, 48, 786432]).unwrap(),
+            (6144, 786432)
+        );
+        assert!(c.validate(0, &[0; 6]).is_err());
+        assert!(
+            c.validate(65, &[8320, 8320, 8320, 65, 65, 1064960])
+                .is_err()
+        );
+        assert!(c.validate(usize::MAX, &[0; 6]).is_err());
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs
@@ -0,0 +1,327 @@
+//! Synthetic ownership/schedule checks; genuine archived replay is separate.
+use super::*;
+use crate::accelerator::program::{CachedGdnContract, CachedGdnOperands};
+use crate::operator_slice::cached_gdn::CachedGdnSliceInput;
+use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
+
+fn cached_source(fixture: &Fixture) -> Loaded {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    cfg["text_config"]["hidden_size"] = json!(256);
+    cfg["text_config"]["linear_key_head_dim"] = json!(128);
+    cfg["text_config"]["linear_value_head_dim"] = json!(128);
+    source_with_config(
+        fixture,
+        serde_json::to_vec(&cfg).unwrap(),
+        true,
+        |_, _, v| v,
+        |_| {},
+    )
+}
+fn raw() -> [Vec<u8>; 6] {
+    let bf16 = |values: Vec<f32>| {
+        values
+            .into_iter()
+            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
+            .collect()
+    };
+    let fp32 = |values: Vec<f32>| values.into_iter().flat_map(f32::to_le_bytes).collect();
+    [
+        bf16(
+            (0..256)
+                .map(|i| if i % 3 == 0 { -0.0 } else { 0.5 })
+                .collect(),
+        ),
+        bf16(
+            (0..256)
+                .map(|i| if i % 2 == 0 { 0.25 } else { -0.5 })
+                .collect(),
+        ),
+        bf16((0..256).map(|i| if i < 128 { 1.0 } else { -2.0 }).collect()),
+        bf16(vec![0.5, 0.25]),
+        fp32(vec![0.75, 0.5]),
+        fp32(
+            (0..32768)
+                .map(|i| {
+                    if i % 11 == 0 {
+                        -0.0
+                    } else {
+                        (i % 9) as f32 * 0.125
+                    }
+                })
+                .collect(),
+        ),
+    ]
+}
+fn input<'a>(
+    source: crate::accelerator::SourceIdentity,
+    raw: &'a [Vec<u8>; 6],
+) -> CachedGdnSliceInput<'a> {
+    CachedGdnSliceInput::from_bytes(
+        source,
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        2,
+        raw.each_ref().map(|v| v.as_slice()),
+    )
+    .unwrap()
+}
+fn imported_bytes() -> u64 {
+    2 * (16384 + 3 * 128 + 2) * 4
+}
+fn result_bytes() -> u64 {
+    2 * (16384 + 2 * 128) * 4
+}
+
+#[test]
+fn cached_fragment_reuses_closed_core_keeps_signed_bits_and_retains_all_outputs() {
+    let f = Fixture::new();
+    let loaded = cached_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let raw = raw();
+    let actual = ops
+        .execute_cached_gdn_slice(0, input(source.source_identity(), &raw), None)
+        .unwrap();
+    let originals = actual.original_inputs().unwrap();
+    for (index, (raw, values)) in raw.iter().zip(originals).enumerate() {
+        if index < 4 {
+            assert_eq!(raw.len(), values.len() * 2);
+            for (bytes, value) in raw.chunks_exact(2).zip(values) {
+                assert_eq!(
+                    value.to_bits(),
+                    u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16
+                );
+            }
+        } else {
+            assert_eq!(raw.len(), values.len() * 4);
+            for (bytes, value) in raw.chunks_exact(4).zip(values) {
+                assert_eq!(
+                    value.to_bits(),
+                    u32::from_le_bytes(bytes.try_into().unwrap())
+                );
+            }
+        }
+    }
+
+    let expected = zerogravity_cpu::original_hf::recurrent::cached_step_128(
+        zerogravity_cpu::original_hf::recurrent::CachedGdn128Inputs {
+            source: source.source_identity(),
+            heads: 2,
+            query: originals[0],
+            key: originals[1],
+            value: originals[2],
+            beta: originals[3],
+            decay: originals[4],
+            previous_state: originals[5],
+        },
+    )
+    .unwrap();
+    same(&expected.core_f32, actual.core_f32().unwrap());
+    same(&expected.output_bf16, actual.output_bf16().unwrap());
+    same(&expected.state, actual.next_state_key_major().unwrap());
+    assert_eq!(actual.layer(), 0);
+    assert_eq!(actual.heads(), 2);
+    let scratch = crate::host_math::cached_gdn::CACHED_INPUT_SCRATCH as u64
+        + crate::host_math::cached_gdn::cached_core_scratch(2).unwrap() as u64;
+    assert_eq!(
+        reserved(&admission),
+        baseline + imported_bytes() + result_bytes() + scratch
+    );
+    let snapshot = admission.arbiter.snapshot(&admission.resource).unwrap();
+    assert_eq!(
+        snapshot.per_class[&AllocationClass::Output],
+        imported_bytes() + result_bytes()
+    );
+    assert_eq!(
+        snapshot
+            .per_class
+            .get(&AllocationClass::TransferStaging)
+            .copied()
+            .unwrap_or(0),
+        0
+    );
+    let (model, _, _) = bind(loaded);
+    assert!(model.new_session().is_err());
+    assert!(model.forward_plan(8, true, false).is_err());
+    drop(ops);
+    drop(source);
+    same(&expected.state, actual.next_state_key_major().unwrap());
+    same(&expected.output_bf16, actual.output_bf16().unwrap());
+    assert!(reserved(&admission) > 0);
+    drop(actual);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn bad_raw_phase_representation_and_finite_boundaries_refuse() {
+    let id = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
+    let original = raw();
+    input(id, &original);
+    for (index, bad) in [
+        (0, vec![0; 511]),
+        (3, vec![0, 0x40, 0, 0x3f]),
+        (4, f32::NAN.to_le_bytes().repeat(2)),
+        (4, 2f32.to_le_bytes().repeat(2)),
+        (5, f32::INFINITY.to_le_bytes().repeat(32768)),
+    ] {
+        let mut changed = original.clone();
+        changed[index] = bad;
+        assert!(
+            CachedGdnSliceInput::from_bytes(
+                id,
+                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+                2,
+                changed.each_ref().map(|v| v.as_slice())
+            )
+            .is_err()
+        );
+    }
+    // Zero state is legal cached input; it never becomes initial chunk64.
+    let mut zero = original;
+    zero[5] = vec![0; 32768 * 4];
+    input(id, &zero);
+}
+#[test]
+fn real_pressure_pre_cancel_and_after_import_cancel_release_only_owned_leases() {
+    let f = Fixture::new();
+    let loaded = cached_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let raw = raw();
+    let value = input(source.source_identity(), &raw);
+    let cancelled = AtomicBool::new(true);
+    assert!(
+        ops.execute_cached_gdn_slice(0, value, Some(&cancelled))
+            .is_err()
+    );
+    assert_eq!(reserved(&admission), baseline);
+    let core_scratch = crate::host_math::cached_gdn::cached_core_scratch(2).unwrap() as u64;
+    let needed = imported_bytes()
+        + result_bytes()
+        + core_scratch
+        + crate::host_math::cached_gdn::CACHED_INPUT_SCRATCH as u64;
+    let pressure = admission
+        .arbiter
+        .admit(
+            &admission.resource,
+            admission.owner.clone(),
+            &[AllocationRequest {
+                kind: AllocationClass::Scratch,
+                bytes: 128 * 1024 * 1024 - baseline - needed + 1,
+            }],
+        )
+        .unwrap();
+    let before = reserved(&admission);
+    assert!(ops.execute_cached_gdn_slice(0, value, None).is_err());
+    assert_eq!(reserved(&admission), before);
+    drop(pressure);
+    cancelled.store(false, Ordering::Release);
+    let mut math =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, Some(&cancelled), None);
+    let operands = math.import_cached_gdn_input(value, 2).unwrap();
+    let imported = reserved(&admission);
+    cancelled.store(true, Ordering::Release);
+    assert!(
+        math.cached_gdn_profiled(
+            source.source_identity(),
+            2,
+            &operands,
+            CachedGdnContract::TorchCpuBf16Avx2Width128
+        )
+        .is_err()
+    );
+    assert_eq!(reserved(&admission), imported);
+    cancelled.store(false, Ordering::Release);
+    let results = math
+        .cached_gdn_profiled(
+            source.source_identity(),
+            2,
+            &operands,
+            CachedGdnContract::TorchCpuBf16Avx2Width128,
+        )
+        .unwrap();
+    let clone = results.next_state.clone();
+    assert!(clone.clone().into_vec().is_err());
+    drop(results);
+    assert_eq!(clone.as_slice().unwrap().len(), 32768);
+    drop(clone);
+    assert_eq!(reserved(&admission), imported);
+    drop(operands);
+    assert_eq!(reserved(&admission), baseline);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
+#[test]
+fn foreign_issuer_strided_inputs_unaware_provider_and_wrong_layer_refuse() {
+    struct Unaware;
+    impl crate::accelerator::MatVecKernel for Unaware {
+        fn try_execute(
+            &self,
+            _: crate::accelerator::MatVecRegion<'_>,
+        ) -> Result<crate::accelerator::RegionExecution> {
+            panic!("cached GDN must refuse before unaware provider callback")
+        }
+    }
+    let f = Fixture::new();
+    let loaded = cached_source(&f);
+    let (source, admission) = windows(&f, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let baseline = reserved(&admission);
+    let raw = raw();
+    let value = input(source.source_identity(), &raw);
+    assert!(ops.execute_cached_gdn_slice(3, value, None).is_err());
+    let provider = Unaware;
+    let unaware =
+        crate::host_math::HostMath::for_window_operator_slice(&ops, None, Some(&provider));
+    assert!(unaware.import_cached_gdn_input(value, 2).is_err());
+    assert_eq!(reserved(&admission), baseline);
+    let mut math = crate::host_math::HostMath::for_window_operator_slice(&ops, None, None);
+    let operands = math.import_cached_gdn_input(value, 2).unwrap();
+    let before = reserved(&admission);
+    let strided = math
+        .view(
+            &operands.previous_state,
+            TensorView {
+                offset: 0,
+                rows: 256,
+                width: 1,
+                row_stride: 2,
+            },
+        )
+        .unwrap();
+    let mut bad = operands.clone();
+    bad.query = strided;
+    let error = math
+        .cached_gdn_profiled(
+            source.source_identity(),
+            2,
+            &bad,
+            CachedGdnContract::TorchCpuBf16Avx2Width128,
+        )
+        .err()
+        .unwrap();
+    assert!(error.to_string().contains("contiguous"));
+    assert_eq!(reserved(&admission), before);
+    let foreign_source = windows(&f, &loaded).0;
+    let foreign_ops = bind_windows(foreign_source, &loaded);
+    let mut foreign =
+        crate::host_math::HostMath::for_window_operator_slice(&foreign_ops, None, None);
+    assert!(
+        foreign
+            .cached_gdn_profiled(
+                source.source_identity(),
+                2,
+                &operands,
+                CachedGdnContract::TorchCpuBf16Avx2Width128
+            )
+            .is_err()
+    );
+    drop(foreign_ops);
+    drop(bad);
+    drop(operands);
+    assert_eq!(reserved(&admission), baseline);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&admission), 0);
+}
--- a/crates/zerogravity-model-qwen35/src/forward.rs
+++ b/crates/zerogravity-model-qwen35/src/forward.rs
@@ -650,3 +650,55 @@
     }
     Ok(values)
 }
+
+/// Pure cached-token arithmetic over explicit inputs; no StateEffect or exp.
+pub(crate) fn cached_gdn_slice<B: MathematicalBackend>(
+    source: &impl OperatorSliceSource,
+    math: &mut B,
+    layer: usize,
+    input: &crate::accelerator::program::CachedGdnOperands<B::Value>,
+) -> Result<crate::accelerator::program::CachedGdnValues<B::Value>> {
+    use crate::accelerator::program::{ActivationPrecision, CachedGdnContract};
+    let heads = crate::operator_slice::cached_gdn::cached_heads(source, layer)?;
+    let operands = [
+        &input.query,
+        &input.key,
+        &input.value,
+        &input.beta,
+        &input.decay,
+        &input.previous_state,
+    ];
+    let contract = CachedGdnContract::TorchCpuBf16Avx2Width128;
+    contract.validate(heads, &operands.map(|v| math.elements(v)))?;
+    for (index, value) in operands.iter().enumerate() {
+        let expected = if index < 4 {
+            ActivationPrecision::Bf16WidenedF32
+        } else {
+            ActivationPrecision::F32
+        };
+        if math.precision(value) != expected {
+            return Err(Error::State("cached GDN source operand precision".into()));
+        }
+    }
+    let values = math.cached_gdn_profiled(source.source_identity(), heads, input, contract)?;
+    for (value, elements, precision) in [
+        (&values.core_f32, heads * 128, ActivationPrecision::F32),
+        (
+            &values.output_bf16,
+            heads * 128,
+            ActivationPrecision::Bf16WidenedF32,
+        ),
+        (
+            &values.next_state,
+            heads * 128 * 128,
+            ActivationPrecision::F32,
+        ),
+    ] {
+        if math.elements(value) != elements || math.precision(value) != precision {
+            return Err(Error::State(
+                "cached GDN explicit result geometry/precision".into(),
+            ));
+        }
+    }
+    Ok(values)
+}
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/gdn_cached_slice.rs
@@ -0,0 +1,211 @@
+//! Observed cached-token operands only; no coefficient producer/cache owner.
+use super::OperatorSliceSource;
+use crate::accelerator::SourceIdentity;
+use crate::accelerator::program::{
+    CachedGdnContract, CachedGdnOperands, CachedGdnValues, PlanBuilder, PlanValue,
+    ProgramNumericalProfile,
+};
+use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
+use crate::{Error, LoadedQwenModel, Result};
+
+pub(crate) fn cached_heads(source: &impl OperatorSliceSource, layer: usize) -> Result<usize> {
+    if !source.source_identity().is_bound()
+        || source.program_numerical_profile()
+            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
+    {
+        return Err(Error::Unsupported(
+            "cached GDN original profile/layer".into(),
+        ));
+    }
+    let heads = usize::try_from(source.geometry().recurrent_value_heads.unwrap_or(0))
+        .map_err(|_| Error::Limit("cached GDN value heads".into()))?;
+    let keys = usize::try_from(source.geometry().recurrent_key_heads.unwrap_or(0))
+        .map_err(|_| Error::Limit("cached GDN key heads".into()))?;
+    if keys == 0 || !heads.is_multiple_of(keys) || heads == 0 || heads > 64 {
+        return Err(Error::Unsupported(
+            "cached GDN repeated-head geometry".into(),
+        ));
+    }
+    let qkv = source.matrix_descriptor(QwenMatrixRole::GdnQkv(layer))?;
+    let output = source.matrix_descriptor(QwenMatrixRole::GdnOutput(layer))?;
+    let norm = source.parameter_descriptor(QwenParameterRole::GdnNorm(layer))?;
+    if qkv.source != source.source_identity()
+        || output.source != source.source_identity()
+        || qkv.rows != (keys * 2 + heads) * 128
+        || output.columns != heads * 128
+        || norm.source != source.source_identity()
+        || norm.shape != [128]
+        || norm.kind != zerogravity_core::quant::QuantKind::F32
+        || norm.bytes != 512
+    {
+        return Err(Error::Unsupported(
+            "cached GDN original width128/source geometry".into(),
+        ));
+    }
+    Ok(heads)
+}
+
+/// Raw source-function inputs AFTER Q/K repeat; qualifier authenticates archive
+/// provenance. Finite FP32 decay is explicitly supplied, never computed here.
+#[derive(Clone, Copy)]
+pub struct CachedGdnSliceInput<'a> {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    heads: usize,
+    words: [&'a [u8]; 6],
+}
+impl<'a> CachedGdnSliceInput<'a> {
+    /// q/k/v/beta are BF16 LE; decay and key-major prior state are FP32 LE.
+    /// A zero prior still selects CACHED arithmetic, never initial chunk64.
+    pub fn from_bytes(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        heads: usize,
+        words: [&'a [u8]; 6],
+    ) -> Result<Self> {
+        let result = Self {
+            source,
+            profile,
+            heads,
+            words,
+        };
+        result.validate()?;
+        Ok(result)
+    }
+    pub fn source_identity(self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    pub fn heads(self) -> usize {
+        self.heads
+    }
+    pub fn raw_operands(self) -> [&'a [u8]; 6] {
+        self.words
+    }
+    pub(crate) fn validate(self) -> Result<()> {
+        if !self.source.is_bound()
+            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported("cached GDN input source/profile".into()));
+        }
+        let mut lengths = [0; 6];
+        for (index, bytes) in self.words.iter().enumerate() {
+            let size = if index < 4 { 2 } else { 4 };
+            if bytes.len() % size != 0 {
+                return Err(Error::Malformed("cached GDN partial raw word".into()));
+            }
+            lengths[index] = bytes.len() / size;
+        }
+        CachedGdnContract::TorchCpuBf16Avx2Width128.validate(self.heads, &lengths)?;
+        for (index, bytes) in self.words.iter().enumerate() {
+            if index < 4 {
+                for raw in bytes.chunks_exact(2) {
+                    let bits = u16::from_le_bytes([raw[0], raw[1]]);
+                    let v = f32::from_bits(u32::from(bits) << 16);
+                    if !v.is_finite() || index == 3 && !(0.0..=1.0).contains(&v) {
+                        return Err(Error::Malformed("cached GDN finite BF16/beta".into()));
+                    }
+                }
+            } else {
+                for raw in bytes.chunks_exact(4) {
+                    let v = f32::from_bits(u32::from_le_bytes(raw.try_into().unwrap()));
+                    if !v.is_finite() || index == 4 && !(0.0..=1.0).contains(&v) {
+                        return Err(Error::Malformed("cached GDN finite FP32/decay".into()));
+                    }
+                }
+            }
+        }
+        Ok(())
+    }
+}
+
+impl LoadedQwenModel {
+    pub fn append_cached_gdn_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &CachedGdnOperands<PlanValue>,
+    ) -> Result<CachedGdnValues<PlanValue>> {
+        self.require_operator_slice()?;
+        crate::forward::cached_gdn_slice(self, builder, layer, input)
+    }
+}
+
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+pub struct CachedGdnSliceOutput {
+    source: SourceIdentity,
+    layer: usize,
+    heads: usize,
+    input: CachedGdnOperands<crate::host_math::HostValue>,
+    values: CachedGdnValues<crate::host_math::HostValue>,
+}
+#[cfg(all(feature = "cpu", feature = "original-hf-binding", target_os = "linux"))]
+impl CachedGdnSliceOutput {
+    pub fn source_identity(&self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+    }
+    pub fn layer(&self) -> usize {
+        self.layer
+    }
+    pub fn heads(&self) -> usize {
+        self.heads
+    }
+    pub fn original_inputs(&self) -> Result<[&[f32]; 6]> {
+        Ok([
+            self.input.query.as_slice()?,
+            self.input.key.as_slice()?,
+            self.input.value.as_slice()?,
+            self.input.beta.as_slice()?,
+            self.input.decay.as_slice()?,
+            self.input.previous_state.as_slice()?,
+        ])
+    }
+    pub fn core_f32(&self) -> Result<&[f32]> {
+        self.values.core_f32.as_slice()
+    }
+    pub fn output_bf16(&self) -> Result<&[f32]> {
+        self.values.output_bf16.as_slice()
+    }
+    pub fn next_state_key_major(&self) -> Result<&[f32]> {
+        self.values.next_state.as_slice()
+    }
+}
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+impl crate::original_binding::OriginalWindowOperators {
+    pub fn append_cached_gdn_slice(
+        &self,
+        builder: &mut PlanBuilder,
+        layer: usize,
+        input: &CachedGdnOperands<PlanValue>,
+    ) -> Result<CachedGdnValues<PlanValue>> {
+        crate::forward::cached_gdn_slice(self, builder, layer, input)
+    }
+    #[cfg(feature = "cpu")]
+    pub fn execute_cached_gdn_slice(
+        &self,
+        layer: usize,
+        input: CachedGdnSliceInput<'_>,
+        cancelled: Option<&std::sync::atomic::AtomicBool>,
+    ) -> Result<CachedGdnSliceOutput> {
+        let heads = cached_heads(self, layer)?;
+        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
+        let operands = math.import_cached_gdn_input(input, heads)?;
+        let values = crate::forward::cached_gdn_slice(self, &mut math, layer, &operands)?;
+        if cancelled.is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire)) {
+            return Err(Error::State("cached GDN fragment cancelled".into()));
+        }
+        Ok(CachedGdnSliceOutput {
+            source: self.source_identity(),
+            layer,
+            heads,
+            input: operands,
+            values,
+        })
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/host_math.rs
+++ b/crates/zerogravity-model-qwen35/src/host_math.rs
@@ -507,6 +507,25 @@
         #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
         let output = output.retain_workspace(workspace)?;
         Ok(output)
+    }
+    fn cached_gdn_profiled(
+        &mut self,
+        source: crate::accelerator::SourceIdentity,
+        heads: usize,
+        input: &crate::accelerator::program::CachedGdnOperands<HostValue>,
+        contract: crate::accelerator::program::CachedGdnContract,
+    ) -> Result<crate::accelerator::program::CachedGdnValues<HostValue>> {
+        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+        {
+            self.execute_cached_gdn(source, heads, input, contract)
+        }
+        #[cfg(not(all(feature = "original-hf-binding", target_os = "linux")))]
+        {
+            let _ = (source, heads, input, contract);
+            Err(Error::Unsupported(
+                "cached GDN requires leased original windows".into(),
+            ))
+        }
     }
     fn causal_conv_silu_profiled(
         &mut self,
@@ -1053,3 +1072,6 @@
 
 #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
 mod gdn_conv;
+
+#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
+pub(crate) mod cached_gdn;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/host_math/cached_gdn.rs
@@ -0,0 +1,194 @@
+//! Admitted private transport/core; no provider exp, state owner or publication.
+use super::*;
+use crate::accelerator::program::{CachedGdnContract, CachedGdnOperands, CachedGdnValues};
+use crate::operator_slice::cached_gdn::CachedGdnSliceInput;
+
+pub(crate) const CACHED_INPUT_SCRATCH: usize = 64 * 1024;
+pub(crate) fn cached_core_scratch(heads: usize) -> Result<usize> {
+    heads
+        .checked_mul(128 * 128 * 4 * 6 + 128 * 4 * 16)
+        .and_then(|n| n.checked_add(256 * 1024))
+        .ok_or_else(|| Error::Limit("cached GDN conservative scratch bound".into()))
+}
+impl HostMath<'_, '_, '_> {
+    fn cached_windows(&self) -> Result<&crate::original_binding::OriginalWindowOperators> {
+        if self.region_kernel.is_some()
+            || self.engine.is_some()
+            || self.state.is_some()
+            || self.source.program_numerical_profile()
+                != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported(
+                "cached GDN private pinned CPU window contract".into(),
+            ));
+        }
+        match &self.source {
+            HostSource::Windows(source) => Ok(source),
+            HostSource::Loaded(_) => Err(Error::Unsupported(
+                "cached GDN requires admitted window backing".into(),
+            )),
+        }
+    }
+    fn check_cached_cancel(&self) -> Result<()> {
+        if self
+            .cancelled
+            .is_some_and(|v| v.load(std::sync::atomic::Ordering::Acquire))
+        {
+            return Err(Error::State("cached GDN private fragment cancelled".into()));
+        }
+        Ok(())
+    }
+    pub(crate) fn import_cached_gdn_input(
+        &self,
+        input: CachedGdnSliceInput<'_>,
+        heads: usize,
+    ) -> Result<CachedGdnOperands<HostValue>> {
+        let source = self.cached_windows()?;
+        input.validate()?;
+        if input.source_identity() != source.source_identity()
+            || input.heads() != heads
+            || input.numerical_profile() != self.source.program_numerical_profile()
+        {
+            return Err(Error::State(
+                "cached GDN input source/profile/head count".into(),
+            ));
+        }
+        self.check_cached_cancel()?;
+        let words = heads
+            .checked_mul(128 * 128 + 3 * 128 + 2)
+            .and_then(|n| n.checked_mul(4))
+            .ok_or_else(|| Error::Limit("cached GDN aggregate input bytes".into()))?;
+        // ALL six retained payloads admitted before the first private Vec.
+        let workspace = Arc::new(
+            source
+                .source()
+                .reserve_operator_buffers(words, CACHED_INPUT_SCRATCH)?,
+        );
+        let mut imported = Vec::with_capacity(6);
+        for (index, raw) in input.raw_operands().iter().enumerate() {
+            let values = if index < 4 {
+                raw.chunks_exact(2)
+                    .map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16))
+                    .collect()
+            } else {
+                raw.chunks_exact(4)
+                    .map(|v| f32::from_bits(u32::from_le_bytes(v.try_into().unwrap())))
+                    .collect()
+            };
+            let precision = if index < 4 {
+                ActivationPrecision::Bf16WidenedF32
+            } else {
+                ActivationPrecision::F32
+            };
+            imported.push(
+                HostValue::typed(
+                    values,
+                    precision,
+                    source.source_identity(),
+                    self.source.program_numerical_profile(),
+                )?
+                .retain_workspace(Some(workspace.clone()))?,
+            );
+        }
+        let [query, key, value, beta, decay, previous_state]: [HostValue; 6] = imported
+            .try_into()
+            .map_err(|_| Error::State("cached GDN imported operand arity".into()))?;
+        self.check_cached_cancel()?;
+        Ok(CachedGdnOperands {
+            query,
+            key,
+            value,
+            beta,
+            decay,
+            previous_state,
+        })
+    }
+    pub(super) fn execute_cached_gdn(
+        &self,
+        identity: crate::accelerator::SourceIdentity,
+        heads: usize,
+        input: &CachedGdnOperands<HostValue>,
+        contract: CachedGdnContract,
+    ) -> Result<CachedGdnValues<HostValue>> {
+        let source = self.cached_windows()?;
+        let operands = [
+            &input.query,
+            &input.key,
+            &input.value,
+            &input.beta,
+            &input.decay,
+            &input.previous_state,
+        ];
+        let lengths = operands.map(|v| v.elements);
+        let (vector, state) = contract.validate(heads, &lengths)?;
+        if identity != source.source_identity()
+            || self.source.metadata().geometry().recurrent_value_heads != Some(heads as u64)
+        {
+            return Err(Error::State(
+                "cached GDN source/configured head count".into(),
+            ));
+        }
+        for (index, value) in operands.iter().enumerate() {
+            let precision = if index < 4 {
+                ActivationPrecision::Bf16WidenedF32
+            } else {
+                ActivationPrecision::F32
+            };
+            if value.source != Some(identity)
+                || value.profile != Some(self.source.program_numerical_profile())
+                || value.precision != precision
+            {
+                return Err(Error::State(
+                    "cached GDN operand source/profile/precision".into(),
+                ));
+            }
+            let workspace = value.data.workspace.as_ref().ok_or_else(|| {
+                Error::State("cached GDN operand missing physical backing".into())
+            })?;
+            if !source.source().same_issuer(workspace.source()) {
+                return Err(Error::State("cached GDN operand foreign issuer".into()));
+            }
+            precision.validate_values(value.as_slice()?)?;
+        }
+        self.check_cached_cancel()?;
+        let bytes = state
+            .checked_add(vector * 2)
+            .and_then(|n| n.checked_mul(4))
+            .ok_or_else(|| Error::Limit("cached GDN aggregate output bytes".into()))?;
+        // Covers every temporary materialized state/product and small reducer
+        // arrays of the existing helper. No native opaque allocator is called.
+        let workspace = Arc::new(
+            source
+                .source()
+                .reserve_operator_buffers(bytes, cached_core_scratch(heads)?)?,
+        );
+        let completed = zerogravity_cpu::original_hf::recurrent::cached_step_128(
+            zerogravity_cpu::original_hf::recurrent::CachedGdn128Inputs {
+                source: identity,
+                heads,
+                query: input.query.as_slice()?,
+                key: input.key.as_slice()?,
+                value: input.value.as_slice()?,
+                beta: input.beta.as_slice()?,
+                decay: input.decay.as_slice()?,
+                previous_state: input.previous_state.as_slice()?,
+            },
+        )?;
+        let make = |values, precision| -> Result<HostValue> {
+            HostValue::typed(
+                values,
+                precision,
+                identity,
+                self.source.program_numerical_profile(),
+            )?
+            .retain_workspace(Some(workspace.clone()))
+        };
+        let result = CachedGdnValues {
+            core_f32: make(completed.core_f32, ActivationPrecision::F32)?,
+            output_bf16: make(completed.output_bf16, ActivationPrecision::Bf16WidenedF32)?,
+            next_state: make(completed.state, ActivationPrecision::F32)?,
+        };
+        self.check_cached_cancel()?;
+        Ok(result)
+    }
+}
--- a/crates/zerogravity-model-qwen35/src/operator_slice.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice.rs
@@ -234,3 +234,6 @@
 
 #[path = "gdn_conv_slice.rs"]
 pub mod gdn_conv;
+
+#[path = "gdn_cached_slice.rs"]
+pub mod cached_gdn;
--- a/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs
@@ -382,3 +382,6 @@
 
 #[path = "gdn_conv_window_tests.rs"]
 mod gdn_conv;
+
+#[path = "cached_gdn_window_tests.rs"]
+mod cached_gdn;

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/SOURCE_DELTA.patch

Original UTF-8 bytes: 38521; SHA256 `8ed8dfb4bbc1b35af74cf85740dcf05ee954e8e43d2aaeb95e47a9dadfeaddeb`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs
@@ -0,0 +1,102 @@
+//! Actual same-source host backing, not a producer/session/provider authority.
+use super::*;
+use std::sync::Arc;
+use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
+
+/// External authenticated operator bytes. Caller proves archive/producer origin;
+/// importing them never claims a current state, model projection or decay math.
+#[derive(Clone, Copy)]
+pub struct InitialPreludeInput<'a> {
+    contract: InitialPreludeContract,
+    raw: &'a [u8],
+}
+impl<'a> InitialPreludeInput<'a> {
+    pub fn from_f32le(contract: InitialPreludeContract, raw: &'a [u8]) -> Result<Self> {
+        if raw.len() != contract.elements() * 4
+            || raw
+                .chunks_exact(4)
+                .any(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x7f80_0000 == 0x7f80_0000)
+        {
+            return Err(Error::Malformed(
+                "initial GDN external finite raw FP32".into(),
+            ));
+        }
+        Ok(Self { contract, raw })
+    }
+    pub fn contract(self) -> InitialPreludeContract {
+        self.contract
+    }
+}
+struct Backing {
+    contract: InitialPreludeContract,
+    input: Vec<f32>,
+    output: Vec<f32>,
+    // Actual buffers retire before their admission/source owner.
+    workspace: WindowWorkspace,
+}
+/// Clones retain the SAME source/backing/lease; no uncharged Vec escape.
+#[derive(Clone)]
+pub struct InitialPreludeOutput(Arc<Backing>);
+impl InitialPreludeOutput {
+    pub fn contract(&self) -> InitialPreludeContract {
+        self.0.contract
+    }
+    pub fn imported_input(&self) -> &[f32] {
+        &self.0.input
+    }
+    pub fn output(&self) -> &[f32] {
+        &self.0.output
+    }
+    pub fn charged_bytes(&self) -> usize {
+        self.0.workspace.bytes()
+    }
+    pub fn same_issuer(&self, source: &VerifiedTensorSource) -> bool {
+        self.0.workspace.source().same_issuer(source)
+    }
+    pub fn required_bytes(contract: InitialPreludeContract) -> usize {
+        contract.elements() * 8 + 256
+    }
+}
+
+/// One pure synchronous leaf. All payload allocation follows aggregate admission.
+/// No GPU/BLAS/provider path and no fallback; cancelled output is discarded.
+pub fn execute_initial_prelude(
+    source: &VerifiedTensorSource,
+    input: InitialPreludeInput<'_>,
+    cancelled: Option<&AtomicBool>,
+) -> Result<InitialPreludeOutput> {
+    not_cancelled(cancelled)?;
+    let contract = input.contract;
+    if contract.source_identity() != source.source_identity() {
+        return Err(Error::State("initial GDN foreign raw source issuer".into()));
+    }
+    let bytes = contract.elements() * 4;
+    let workspace = source.reserve_operator_buffers(bytes, bytes + 256)?;
+    not_cancelled(cancelled)?;
+    let mut imported = Vec::with_capacity(contract.elements());
+    imported.extend(
+        input
+            .raw
+            .chunks_exact(4)
+            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
+    );
+    let mut output = vec![0.0f32; contract.elements()];
+    if imported.capacity() != contract.elements() || output.capacity() != contract.elements() {
+        return Err(Error::Limit("initial GDN exact payload capacities".into()));
+    }
+    initial_prelude_into(contract, &imported, &mut output, cancelled)?;
+    let result = InitialPreludeOutput(Arc::new(Backing {
+        contract,
+        input: imported,
+        output,
+        workspace,
+    }));
+    completed(result, cancelled)
+}
+pub(super) fn completed(
+    result: InitialPreludeOutput,
+    cancelled: Option<&AtomicBool>,
+) -> Result<InitialPreludeOutput> {
+    not_cancelled(cancelled)?;
+    Ok(result)
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs
@@ -0,0 +1,257 @@
+use super::*;
+fn contract(operation: InitialPreludeOperation, heads: usize) -> InitialPreludeContract {
+    InitialPreludeContract::new(
+        SourceIdentity::for_new_allocation().unwrap(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        GdnPreludePhase::InitialChunk64,
+        operation,
+        heads,
+    )
+    .unwrap()
+}
+#[test]
+fn prefix_retains_f64_carry_and_stores_each_f32_prefix() {
+    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
+    let mut input = vec![0.0; c.elements()];
+    input[0] = 1.0;
+    input[1] = f32::from_bits(0x3380_0000); // 2^-24, exact half ULP at1.
+    input[2] = -1.0;
+    input[64] = -0.0;
+    input[128] = -1.0;
+    let mut output = vec![9.0; input.len()];
+    initial_prelude_into(c, &input, &mut output, None).unwrap();
+    assert_eq!(output[0].to_bits(), 0x3f80_0000);
+    assert_eq!(output[1].to_bits(), 0x3f80_0000); // Store rounded, carry not rounded.
+    assert_eq!(output[2].to_bits(), 0x3380_0000);
+    assert_eq!(output[63].to_bits(), 0x3380_0000);
+    assert_eq!(output[64].to_bits(), 0); // +0 accumulator + -0 is +0.
+    assert_eq!(output[127].to_bits(), 0);
+    assert_eq!(output[128].to_bits(), 0xbf80_0000);
+    assert_eq!(output[191].to_bits(), 0xbf80_0000);
+}
+#[test]
+fn query_multiply_preserves_signed_zero_and_subnormal_rounding() {
+    let c = contract(InitialPreludeOperation::QueryScale128, 48);
+    let mut input = vec![1.0; c.elements()];
+    input[..6].copy_from_slice(&[
+        1.0,
+        -1.0,
+        0.0,
+        -0.0,
+        f32::from_bits(1),
+        f32::from_bits(0x8000_0001),
+    ]);
+    let mut out = vec![0.0; input.len()];
+    initial_prelude_into(c, &input, &mut out, None).unwrap();
+    assert_eq!(
+        out[..6].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
+        [0x3db5_04f3, 0xbdb5_04f3, 0, 0x8000_0000, 0, 0x8000_0000]
+    );
+    assert!(out[6..].iter().all(|v| v.to_bits() == 0x3db5_04f3));
+}
+#[test]
+fn cached_profile_source_geometry_and_complete_span_refuse() {
+    for (source, profile, phase, heads) in [
+        (
+            SourceIdentity::unbound(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::GgufSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::Cached,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            32,
+        ),
+    ] {
+        assert!(
+            InitialPreludeContract::new(
+                source,
+                profile,
+                phase,
+                InitialPreludeOperation::DecayPrefix64,
+                heads
+            )
+            .is_err()
+        );
+    }
+    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
+    let mut input = vec![0.0; c.elements()];
+    let mut out = vec![7.0; c.elements()];
+    assert!(initial_prelude_into(c, &input[..input.len() - 1], &mut out, None).is_err());
+    input[0] = f32::NAN;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+    input[0] = f32::INFINITY;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+    assert!(out.iter().all(|v| *v == 7.0));
+    input[0] = 0.0;
+    assert!(initial_prelude_into(c, &input, &mut out, Some(&AtomicBool::new(true))).is_err());
+    assert!(out.iter().all(|v| *v == 7.0));
+    input[0] = f32::MAX;
+    input[1] = f32::MAX;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+}
+
+#[cfg(all(feature = "cache", target_os = "linux"))]
+mod physical {
+    use super::*;
+    use std::fs::{self, File};
+    use std::path::{Path, PathBuf};
+    use std::sync::atomic::AtomicU64;
+    use zerogravity_artifacts::artifact::Artifact;
+    use zerogravity_artifacts::cache::TransferAdmission;
+    use zerogravity_artifacts::source_weights::{
+        ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits,
+    };
+    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
+    const META: u64 = 16 * 1024 * 1024;
+    // Existing accepted SYNTHETIC source; no trained fixture or new header alias.
+    const BYTES: &[u8] = include_bytes!("../window_projection/fixture.safetensors");
+    const INDEX: &[u8] = include_bytes!("../window_projection/fixture.index.json");
+    struct Fixture(PathBuf);
+    impl Fixture {
+        fn new() -> Self {
+            static NEXT: AtomicU64 = AtomicU64::new(1);
+            let p = std::env::temp_dir().join(format!(
+                "zg-initial-prefix-{}-{}",
+                std::process::id(),
+                NEXT.fetch_add(1, Ordering::Relaxed)
+            ));
+            fs::create_dir(&p).unwrap();
+            fs::write(p.join("model.safetensors"), BYTES).unwrap();
+            Self(p)
+        }
+        fn source(&self, h: &TransferAdmission) -> VerifiedTensorSource {
+            SourceWeights::prepare_verified_windows(
+                INDEX,
+                "42e2f4619abb5431bccce39b473b23eaa90679c9947d26c9d439308c78a86149",
+                vec![
+                    ExpectedSourceShard::new(
+                        "model.safetensors".into(),
+                        Artifact::from_file(
+                            File::open(self.0.join("model.safetensors")).unwrap(),
+                            Path::new("model.safetensors"),
+                        )
+                        .unwrap(),
+                        BYTES.len() as u64,
+                        "973fca86c0585f6b11d0b686c051bad06b896f13b2b3d10ef0299eea4510f033".into(),
+                    )
+                    .unwrap(),
+                ],
+                WindowOpenLimits {
+                    max_source_bytes: 1024 * 1024,
+                    max_metadata_bytes: META,
+                    max_shards: 1,
+                    max_tensors: 6,
+                    scan_chunk_bytes: 1024,
+                },
+                h,
+            )
+            .unwrap()
+            .open()
+            .unwrap()
+        }
+    }
+    impl Drop for Fixture {
+        fn drop(&mut self) {
+            let _ = fs::remove_dir_all(&self.0);
+        }
+    }
+    fn host() -> TransferAdmission {
+        let arbiter = ResourceArbiter::new();
+        let resource = PhysicalDeviceKey::new("host", "synthetic-initial-prefix").unwrap();
+        arbiter
+            .register(resource.clone(), 64 * 1024 * 1024)
+            .unwrap();
+        TransferAdmission {
+            arbiter,
+            resource,
+            owner: ResourceOwner::new("synthetic", "initial-prefix").unwrap(),
+        }
+    }
+    fn reserved(h: &TransferAdmission) -> u64 {
+        h.arbiter.snapshot(&h.resource).unwrap().reserved
+    }
+    fn raw(c: InitialPreludeContract) -> Vec<u8> {
+        1.0f32.to_le_bytes().repeat(c.elements())
+    }
+    #[test]
+    fn admission_foreign_cancel_pressure_clone_and_lastdrop() {
+        let f = Fixture::new();
+        let h = host();
+        let source = f.source(&h);
+        let foreign = f.source(&h);
+        let c = InitialPreludeContract::new(
+            source.source_identity(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            InitialPreludeOperation::QueryScale128,
+            48,
+        )
+        .unwrap();
+        let raw = raw(c);
+        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
+        let before = reserved(&h);
+        assert!(execute_initial_prelude(&foreign, input, None).is_err());
+        assert!(execute_initial_prelude(&source, input, Some(&AtomicBool::new(true))).is_err());
+        assert_eq!(reserved(&h), before);
+        let need = InitialPreludeOutput::required_bytes(c) as u64;
+        let hold = h.arbiter.snapshot(&h.resource).unwrap().remaining - (need - 1);
+        let pressure = source.reserve_workspace(hold as usize).unwrap();
+        let pressured = reserved(&h);
+        assert!(execute_initial_prelude(&source, input, None).is_err());
+        assert_eq!(reserved(&h), pressured);
+        drop(pressure);
+        assert_eq!(reserved(&h), before);
+        let output = execute_initial_prelude(&source, input, None).unwrap();
+        let clone = output.clone();
+        assert!(output.same_issuer(&source));
+        assert!(!output.same_issuer(&foreign));
+        assert_eq!(reserved(&h), before + need);
+        assert_eq!(output.charged_bytes() as u64, need);
+        drop(foreign);
+        drop(source);
+        assert_eq!(reserved(&h), META + need);
+        assert_eq!(clone.imported_input().len(), 48 * 128);
+        assert!(clone.output().iter().all(|v| v.to_bits() == 0x3db5_04f3));
+        drop(output);
+        assert_eq!(reserved(&h), META + need);
+        drop(clone);
+        assert_eq!(reserved(&h), 0);
+    }
+    #[test]
+    fn completed_result_cancel_drops_real_admitted_backing() {
+        let f = Fixture::new();
+        let h = host();
+        let source = f.source(&h);
+        let c = InitialPreludeContract::new(
+            source.source_identity(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            InitialPreludeOperation::DecayPrefix64,
+            16,
+        )
+        .unwrap();
+        let raw = raw(c);
+        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
+        let value = execute_initial_prelude(&source, input, None).unwrap();
+        assert!(reserved(&h) > META);
+        assert!(super::super::admitted::completed(value, Some(&AtomicBool::new(true))).is_err());
+        assert_eq!(reserved(&h), META);
+        drop(source);
+        assert_eq!(reserved(&h), 0);
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude.rs
@@ -0,0 +1,155 @@
+//! Closed initial-GDN leaves; no L2/exp/chunk solver, state or model permit.
+//! Cumsum schedule: Torch cf30153 ReduceOpsKernel.cpp79–95,
+//! AccumulateType.h64–66/155. Attribution: initial_prelude.PYTORCH-LICENSE.
+use crate::accelerator::{SourceIdentity, program::ProgramNumericalProfile};
+use crate::{Error, Result};
+use std::sync::atomic::{AtomicBool, Ordering};
+
+/// Phase is explicit; cached query division is deliberately unsupported here.
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum GdnPreludePhase {
+    InitialChunk64,
+    Cached,
+}
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum InitialPreludeOperation {
+    /// One already-normalized contiguous `[heads,1,128]` FP32 query.
+    QueryScale128,
+    /// One already-padded contiguous `[heads,1,64]` FP32 decay chunk.
+    DecayPrefix64,
+}
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub struct InitialPreludeContract {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    operation: InitialPreludeOperation,
+    heads: usize,
+}
+impl InitialPreludeContract {
+    pub fn new(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        phase: GdnPreludePhase,
+        operation: InitialPreludeOperation,
+        heads: usize,
+    ) -> Result<Self> {
+        if phase != GdnPreludePhase::InitialChunk64
+            || profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported("initial GDN leaf phase/profile".into()));
+        }
+        if !source.is_bound() || !matches!(heads, 16 | 48) {
+            return Err(Error::Malformed("initial GDN leaf source/heads".into()));
+        }
+        Ok(Self {
+            source,
+            profile,
+            operation,
+            heads,
+        })
+    }
+    pub fn source_identity(self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    pub fn phase(self) -> GdnPreludePhase {
+        GdnPreludePhase::InitialChunk64
+    }
+    pub fn operation(self) -> InitialPreludeOperation {
+        self.operation
+    }
+    pub fn heads(self) -> usize {
+        self.heads
+    }
+    pub fn elements(self) -> usize {
+        self.heads
+            * match self.operation {
+                InitialPreludeOperation::QueryScale128 => 128,
+                InitialPreludeOperation::DecayPrefix64 => 64,
+            }
+    }
+}
+fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
+    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        return Err(Error::State("initial GDN leaf cancelled".into()));
+    }
+    Ok(())
+}
+fn environment() -> Result<()> {
+    #[cfg(target_arch = "x86_64")]
+    {
+        let mut mxcsr = 0u32;
+        // SAFETY: stmxcsr writes exactly four bytes to this valid local word.
+        // Passive read only; exception status bits may change during arithmetic.
+        unsafe {
+            std::arch::asm!("stmxcsr [{p}]", p = in(reg) (&mut mxcsr as *mut u32),
+                options(nostack, preserves_flags));
+        }
+        if !std::arch::is_x86_feature_detected!("avx2") || mxcsr & 0xffc0 != 0x1f80 {
+            return Err(Error::Unsupported(
+                "initial GDN pinned CPU FP controls".into(),
+            ));
+        }
+        Ok(())
+    }
+    #[cfg(not(target_arch = "x86_64"))]
+    Err(Error::Unsupported(
+        "initial GDN pinned CPU architecture".into(),
+    ))
+}
+
+/// No allocation/provider callback. Caller owns both complete, disjoint slices.
+/// Output remains private on error; successful completion grants no state effect.
+pub fn initial_prelude_into(
+    contract: InitialPreludeContract,
+    input: &[f32],
+    output: &mut [f32],
+    cancelled: Option<&AtomicBool>,
+) -> Result<()> {
+    not_cancelled(cancelled)?;
+    environment()?;
+    if input.len() != contract.elements()
+        || output.len() != input.len()
+        || input.iter().any(|v| !v.is_finite())
+    {
+        return Err(Error::Malformed(
+            "initial GDN complete finite FP32 span".into(),
+        ));
+    }
+    match contract.operation {
+        InitialPreludeOperation::QueryScale128 => {
+            // Source query * (128 ** -0.5), scalar converted once to FP32.
+            // No reciprocal/division, FMA, BF16 cast or normalization rewrite.
+            const SCALE: f32 = f32::from_bits(0x3db5_04f3);
+            for (row, target) in input.chunks_exact(128).zip(output.chunks_exact_mut(128)) {
+                not_cancelled(cancelled)?;
+                for (&value, result) in row.iter().zip(target) {
+                    *result = value * SCALE;
+                }
+            }
+        }
+        InitialPreludeOperation::DecayPrefix64 => {
+            for (row, target) in input.chunks_exact(64).zip(output.chunks_exact_mut(64)) {
+                not_cancelled(cancelled)?;
+                let mut carry = 0.0f64;
+                for (&value, result) in row.iter().zip(target) {
+                    carry += f64::from(value);
+                    *result = carry as f32; // Store each prefix; retain unrounded F64 carry.
+                    if !result.is_finite() {
+                        return Err(Error::Malformed("initial GDN prefix FP32 overflow".into()));
+                    }
+                }
+            }
+        }
+    }
+    not_cancelled(cancelled)
+}
+
+#[cfg(all(feature = "cache", target_os = "linux"))]
+mod admitted;
+#[cfg(all(feature = "cache", target_os = "linux"))]
+pub use admitted::{InitialPreludeInput, InitialPreludeOutput, execute_initial_prelude};
+#[cfg(test)]
+mod tests;
--- a/crates/zerogravity-cpu/src/original_hf.rs
+++ b/crates/zerogravity-cpu/src/original_hf.rs
@@ -4,6 +4,7 @@
 //! This does not enable complete trained Qwen3.8 inference.
 
 pub mod coefficient_preparation;
+pub mod initial_prelude;
 pub mod conv;
 pub mod norm;
 pub mod recurrent;
--- a/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs
@@ -56,13 +56,18 @@
     }
     let dt = source.parameter_descriptor(QwenParameterRole::GdnDt(layer))?;
     let a_log = source.parameter_descriptor(QwenParameterRole::GdnA(layer))?;
+    // A_log is metadata only; no exp/g or learned-parameter cast is granted.
+    let a_log_codec_valid = match a_log.kind {
+        QuantKind::F32 => a_log.bytes == heads * 4,
+        QuantKind::Bf16 => heads == 48 && a_log.bytes == 96,
+        _ => false,
+    };
     if dt.source != source.source_identity()
         || dt.name != format!("{namespace}.dt_bias")
         || a_log.source != source.source_identity()
         || a_log.name != format!("{namespace}.A_log")
-        || a_log.kind != QuantKind::F32
         || a_log.shape != [heads as u64]
-        || a_log.bytes != heads * 4
+        || !a_log_codec_valid
     {
         return Err(Error::State(
             "coefficient preparation original raw A_log/dt_bias roles".into(),
--- a/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
@@ -269,3 +269,6 @@
     assert!(model.new_session().is_err());
     assert!(model.forward_plan(8, true, true).is_err());
 }
+
+#[path = "coefficient_preparation_h48_tests.rs"]
+mod h48;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/coefficient_preparation_h48_tests.rs
@@ -0,0 +1,390 @@
+//! Sealed synthetic H48 source/metadata/lease checks; no trained math claim.
+use super::*;
+use crate::accelerator::program::{ProjectionContract, SourceParameter, SourceTensor};
+use crate::operator_slice::OperatorSliceSource;
+use zerogravity_core::quant::QuantKind;
+
+fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    let text = &mut cfg["text_config"];
+    text["num_hidden_layers"] = json!(2);
+    text["full_attention_interval"] = json!(2);
+    text["layer_types"] = json!(["linear_attention", "full_attention"]);
+    text["hidden_size"] = json!(256);
+    text["linear_num_key_heads"] = json!(16);
+    text["linear_num_value_heads"] = json!(48);
+    text["linear_key_head_dim"] = json!(128);
+    text["linear_value_head_dim"] = json!(128);
+    let cfg = serde_json::to_vec(&cfg).unwrap();
+    let mut header = BTreeMap::new();
+    let mut map = BTreeMap::new();
+    let mut payload = Vec::new();
+    for spec in QwenTextConfig::from_json(&cfg)
+        .unwrap()
+        .tensor_specs()
+        .unwrap()
+    {
+        let begin = payload.len();
+        let value = if spec.name.ends_with("dt_bias") {
+            0
+        } else {
+            0x3e80u16
+        };
+        for _ in 0..spec.shape.iter().product::<u64>() {
+            payload.extend_from_slice(&value.to_le_bytes());
+        }
+        header.insert(
+            spec.name.clone(),
+            json!({"dtype":"BF16", "shape":spec.shape,
+            "data_offsets":[begin,payload.len()]}),
+        );
+        map.insert(spec.name, "model.safetensors");
+    }
+    let header = serde_json::to_vec(&header).unwrap();
+    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
+    bytes.extend(header);
+    bytes.extend(&payload);
+    assert!(bytes.len() < 16 * 1024 * 1024);
+    let path = fixture.0.join("model.safetensors");
+    fs::write(&path, &bytes).unwrap();
+    let index = serde_json::to_vec(&json!({"metadata":{"total_size":payload.len()},
+        "weight_map":map}))
+    .unwrap();
+    let index_sha = digest(&index);
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "coefficient-h48-source-test").unwrap();
+    arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
+    let host = TransferAdmission {
+        arbiter,
+        resource: key,
+        owner: ResourceOwner::new("synthetic", "coefficient-h48").unwrap(),
+    };
+    let artifact =
+        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).unwrap();
+    let source = SourceWeights::prepare_verified_windows(
+        &index,
+        &index_sha,
+        vec![
+            ExpectedSourceShard::new(
+                "model.safetensors".into(),
+                artifact,
+                bytes.len() as u64,
+                digest(&bytes),
+            )
+            .unwrap(),
+        ],
+        WindowOpenLimits {
+            max_source_bytes: 16 * 1024 * 1024,
+            max_metadata_bytes: 32 * 1024 * 1024,
+            max_shards: 1,
+            max_tensors: 200,
+            scan_chunk_bytes: 64 * 1024,
+        },
+        &host,
+    )
+    .unwrap()
+    .open()
+    .unwrap();
+    (source, host, cfg, index_sha)
+}
+
+struct MetadataFault<'a> {
+    source: &'a OriginalWindowOperators,
+    a_log: SourceParameter,
+    profile: ProgramNumericalProfile,
+}
+impl OperatorSliceSource for MetadataFault<'_> {
+    fn geometry(&self) -> &crate::model::ModelGeometry {
+        self.source.geometry()
+    }
+    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
+        self.source.source_identity()
+    }
+    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    fn rms_epsilon(&self) -> f32 {
+        self.source.rms_epsilon()
+    }
+    fn matrix_descriptor(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
+        self.source.matrix_descriptor(role)
+    }
+    fn parameter_descriptor(&self, role: QwenParameterRole) -> Result<SourceParameter> {
+        if role == QwenParameterRole::GdnA(0) {
+            Ok(self.a_log.clone())
+        } else {
+            self.source.parameter_descriptor(role)
+        }
+    }
+    fn embedding_contract(&self) -> EmbeddingContract {
+        self.source.embedding_contract()
+    }
+    fn projection_contract(&self) -> ProjectionContract {
+        self.source.projection_contract()
+    }
+}
+
+#[test]
+fn only_genuine_h48_bf16_unevaluated_a_log_extent_is_admitted() {
+    use crate::operator_slice::coefficient_preparation::coefficient_roles;
+    let old = Fixture::new();
+    let loaded = coefficient_source(&old);
+    let (source, host) = windows(&old, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let roles = coefficient_roles(&ops, 0).unwrap();
+    assert_eq!(
+        (roles.heads, roles.a_log.kind, roles.a_log.bytes),
+        (16, QuantKind::F32, 64)
+    );
+    let mut unsupported = roles.a_log;
+    unsupported.kind = QuantKind::Bf16;
+    unsupported.bytes = 32;
+    assert!(
+        coefficient_roles(
+            &MetadataFault {
+                source: &ops,
+                a_log: unsupported,
+                profile: ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+            },
+            0
+        )
+        .is_err()
+    );
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&host), 0);
+
+    let fixture = Fixture::new();
+    let (source, host, cfg, index) = h48_source(&fixture);
+    let ops = OriginalWindowOperators::bind(
+        source.clone(),
+        &cfg,
+        &digest(&cfg),
+        &index,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )
+    .unwrap();
+    let roles = coefficient_roles(&ops, 0).unwrap();
+    assert_eq!(
+        (
+            roles.heads,
+            roles.a_log.kind,
+            roles.a_log.shape.clone(),
+            roles.a_log.bytes
+        ),
+        (48, QuantKind::Bf16, vec![48], 96)
+    );
+    assert_eq!(
+        (roles.dt.kind, roles.dt.shape, roles.dt.bytes),
+        (QuantKind::Bf16, vec![48], 96)
+    );
+    let before = reserved(&host);
+    for (kind, bytes, shape, foreign, name, profile) in [
+        (
+            QuantKind::F16,
+            96,
+            vec![48],
+            false,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            192,
+            vec![48],
+            false,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::F32,
+            96,
+            vec![48],
+            false,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            96,
+            vec![47],
+            false,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            96,
+            vec![48],
+            true,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            96,
+            vec![48],
+            false,
+            "foreign.A_log".into(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            96,
+            vec![48],
+            false,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::GgufSourceV1,
+        ),
+    ] {
+        let mut parameter = roles.a_log.clone();
+        parameter.kind = kind;
+        parameter.bytes = bytes;
+        parameter.shape = shape;
+        parameter.name = name;
+        if foreign {
+            parameter.source = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
+        }
+        assert!(
+            coefficient_roles(
+                &MetadataFault {
+                    source: &ops,
+                    a_log: parameter,
+                    profile
+                },
+                0
+            )
+            .is_err()
+        );
+        assert_eq!(reserved(&host), before);
+    }
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&host), 0);
+}
+
+#[test]
+fn h48_preparation_preserves_source_lease_cancel_pressure_and_all_private_words() {
+    let fixture = Fixture::new();
+    let (source, host, cfg, index) = h48_source(&fixture);
+    let ops = OriginalWindowOperators::bind(
+        source.clone(),
+        &cfg,
+        &digest(&cfg),
+        &index,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )
+    .unwrap();
+    let a = 0x41a8u16.to_le_bytes().repeat(48);
+    let b = 0u16.to_le_bytes().repeat(48);
+    let make = |identity| input(identity, &a, &b);
+    let before = reserved(&host);
+    for (layer, identity, cancel) in [
+        (1, source.source_identity(), None),
+        (
+            0,
+            crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
+            None,
+        ),
+        (0, source.source_identity(), Some(AtomicBool::new(true))),
+    ] {
+        assert!(
+            ops.execute_gdn_coefficient_preparation_slice(layer, make(identity), cancel.as_ref())
+                .is_err()
+        );
+        assert_eq!(reserved(&host), before);
+    }
+    let mut pressure = Vec::new();
+    loop {
+        let remaining = host.arbiter.snapshot(&host.resource).unwrap().remaining;
+        if remaining == 0 {
+            break;
+        }
+        pressure.push(
+            source
+                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
+                .unwrap(),
+        );
+    }
+    let pressured = reserved(&host);
+    assert!(
+        ops.execute_gdn_coefficient_preparation_slice(0, make(source.source_identity()), None)
+            .is_err()
+    );
+    assert_eq!(reserved(&host), pressured);
+    drop(pressure);
+    assert_eq!(reserved(&host), before);
+    let output = ops
+        .execute_gdn_coefficient_preparation_slice(0, make(source.source_identity()), None)
+        .unwrap();
+    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 48]);
+    assert_eq!(output.beta_input_bf16().unwrap(), &[0.0; 48]);
+    assert_eq!(output.biased_alpha_f32().unwrap(), &[21.0; 48]);
+    assert_eq!(output.softplus_f32().unwrap(), &[21.0; 48]);
+    assert_eq!(output.beta_bf16().unwrap(), &[0.5; 48]);
+    assert_eq!(
+        output.unevaluated_raw_a_log_parameter().kind,
+        QuantKind::Bf16
+    );
+    assert_eq!(reserved(&host), before + 2 * 1024 * 1024);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&host), 34 * 1024 * 1024);
+    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 48]);
+    assert_eq!(output.beta_bf16().unwrap(), &[0.5; 48]);
+    drop(output);
+    assert_eq!(reserved(&host), 0);
+}
+
+#[test]
+fn h48_recording_adds_only_existing_two_preparation_nodes() {
+    let fixture = Fixture::new();
+    let (source, host, cfg, index) = h48_source(&fixture);
+    let ops = OriginalWindowOperators::bind(
+        source.clone(),
+        &cfg,
+        &digest(&cfg),
+        &index,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )
+    .unwrap();
+    let mut builder = PlanBuilder::new_operator_slice(
+        source.source_identity(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+    )
+    .unwrap();
+    // Same-builder views exercise transport; no authentic projection is claimed.
+    let embed = builder
+        .embed_profiled(
+            ops.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    let a = builder.view(&embed, TensorView::vector(0, 48)).unwrap();
+    let b = builder.view(&embed, TensorView::vector(48, 48)).unwrap();
+    let values = ops
+        .append_gdn_coefficient_preparation_slice(&mut builder, 0, &a, &b)
+        .unwrap();
+    let plan = builder.finish_operator_slice(values.softplus).unwrap();
+    assert_eq!(plan.plan().nodes().len(), 5);
+    assert_eq!(
+        plan.plan()
+            .nodes()
+            .iter()
+            .filter(|n| matches!(
+                n.operation,
+                PlanOperation::TypedGdnAlphaPreparation { .. }
+                    | PlanOperation::TypedGdnBetaPreparation { .. }
+            ))
+            .count(),
+        2
+    );
+    assert!(plan.plan().required_domains().is_empty());
+    assert!(plan.plan().proposed_domains().is_empty());
+    assert!(plan.plan().logits().is_none());
+    assert_eq!(plan.plan().frontier_delta(), 0);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&host), 0);
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude.PYTORCH-LICENSE
@@ -0,0 +1,84 @@
+From PyTorch:
+
+Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
+Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
+Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
+Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
+Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
+Copyright (c) 2011-2013 NYU                      (Clement Farabet)
+Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
+Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
+Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
+
+From Caffe2:
+
+Copyright (c) 2016-present, Facebook Inc. All rights reserved.
+
+All contributions by Facebook:
+Copyright (c) 2016 Facebook Inc.
+
+All contributions by Google:
+Copyright (c) 2015 Google Inc.
+All rights reserved.
+
+All contributions by Yangqing Jia:
+Copyright (c) 2015 Yangqing Jia
+All rights reserved.
+
+All contributions by Kakao Brain:
+Copyright 2019-2020 Kakao Brain
+
+All contributions by Cruise LLC:
+Copyright (c) 2022 Cruise LLC.
+All rights reserved.
+
+All contributions by Tri Dao:
+Copyright (c) 2024 Tri Dao.
+All rights reserved.
+
+All contributions by Arm:
+Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
+
+All contributions from Caffe:
+Copyright(c) 2013, 2014, 2015, the respective contributors
+All rights reserved.
+
+All other contributions:
+Copyright(c) 2015, 2016 the respective contributors
+All rights reserved.
+
+Caffe2 uses a copyright model similar to Caffe: each contributor holds
+copyright over their contributions to Caffe2. The project versioning records
+all such contribution and copyright details. If a contributor wants to further
+mark their specific copyright on a particular contribution, they should
+indicate their copyright solely in the commit message of the change when it is
+committed.
+
+All rights reserved.
+
+Redistribution and use in source and binary forms, with or without
+modification, are permitted provided that the following conditions are met:
+
+1. Redistributions of source code must retain the above copyright
+   notice, this list of conditions and the following disclaimer.
+
+2. Redistributions in binary form must reproduce the above copyright
+   notice, this list of conditions and the following disclaimer in the
+   documentation and/or other materials provided with the distribution.
+
+3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
+   and IDIAP Research Institute nor the names of its contributors may be
+   used to endorse or promote products derived from this software without
+   specific prior written permission.
+
+THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
+AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
+IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
+ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
+LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
+CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
+SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
+INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
+CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
+ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
+POSSIBILITY OF SUCH DAMAGE.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/frozen-prepared/SOURCE_DELTA.patch

Original UTF-8 bytes: 38521; SHA256 `8ed8dfb4bbc1b35af74cf85740dcf05ee954e8e43d2aaeb95e47a9dadfeaddeb`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs
@@ -0,0 +1,102 @@
+//! Actual same-source host backing, not a producer/session/provider authority.
+use super::*;
+use std::sync::Arc;
+use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
+
+/// External authenticated operator bytes. Caller proves archive/producer origin;
+/// importing them never claims a current state, model projection or decay math.
+#[derive(Clone, Copy)]
+pub struct InitialPreludeInput<'a> {
+    contract: InitialPreludeContract,
+    raw: &'a [u8],
+}
+impl<'a> InitialPreludeInput<'a> {
+    pub fn from_f32le(contract: InitialPreludeContract, raw: &'a [u8]) -> Result<Self> {
+        if raw.len() != contract.elements() * 4
+            || raw
+                .chunks_exact(4)
+                .any(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x7f80_0000 == 0x7f80_0000)
+        {
+            return Err(Error::Malformed(
+                "initial GDN external finite raw FP32".into(),
+            ));
+        }
+        Ok(Self { contract, raw })
+    }
+    pub fn contract(self) -> InitialPreludeContract {
+        self.contract
+    }
+}
+struct Backing {
+    contract: InitialPreludeContract,
+    input: Vec<f32>,
+    output: Vec<f32>,
+    // Actual buffers retire before their admission/source owner.
+    workspace: WindowWorkspace,
+}
+/// Clones retain the SAME source/backing/lease; no uncharged Vec escape.
+#[derive(Clone)]
+pub struct InitialPreludeOutput(Arc<Backing>);
+impl InitialPreludeOutput {
+    pub fn contract(&self) -> InitialPreludeContract {
+        self.0.contract
+    }
+    pub fn imported_input(&self) -> &[f32] {
+        &self.0.input
+    }
+    pub fn output(&self) -> &[f32] {
+        &self.0.output
+    }
+    pub fn charged_bytes(&self) -> usize {
+        self.0.workspace.bytes()
+    }
+    pub fn same_issuer(&self, source: &VerifiedTensorSource) -> bool {
+        self.0.workspace.source().same_issuer(source)
+    }
+    pub fn required_bytes(contract: InitialPreludeContract) -> usize {
+        contract.elements() * 8 + 256
+    }
+}
+
+/// One pure synchronous leaf. All payload allocation follows aggregate admission.
+/// No GPU/BLAS/provider path and no fallback; cancelled output is discarded.
+pub fn execute_initial_prelude(
+    source: &VerifiedTensorSource,
+    input: InitialPreludeInput<'_>,
+    cancelled: Option<&AtomicBool>,
+) -> Result<InitialPreludeOutput> {
+    not_cancelled(cancelled)?;
+    let contract = input.contract;
+    if contract.source_identity() != source.source_identity() {
+        return Err(Error::State("initial GDN foreign raw source issuer".into()));
+    }
+    let bytes = contract.elements() * 4;
+    let workspace = source.reserve_operator_buffers(bytes, bytes + 256)?;
+    not_cancelled(cancelled)?;
+    let mut imported = Vec::with_capacity(contract.elements());
+    imported.extend(
+        input
+            .raw
+            .chunks_exact(4)
+            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
+    );
+    let mut output = vec![0.0f32; contract.elements()];
+    if imported.capacity() != contract.elements() || output.capacity() != contract.elements() {
+        return Err(Error::Limit("initial GDN exact payload capacities".into()));
+    }
+    initial_prelude_into(contract, &imported, &mut output, cancelled)?;
+    let result = InitialPreludeOutput(Arc::new(Backing {
+        contract,
+        input: imported,
+        output,
+        workspace,
+    }));
+    completed(result, cancelled)
+}
+pub(super) fn completed(
+    result: InitialPreludeOutput,
+    cancelled: Option<&AtomicBool>,
+) -> Result<InitialPreludeOutput> {
+    not_cancelled(cancelled)?;
+    Ok(result)
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs
@@ -0,0 +1,257 @@
+use super::*;
+fn contract(operation: InitialPreludeOperation, heads: usize) -> InitialPreludeContract {
+    InitialPreludeContract::new(
+        SourceIdentity::for_new_allocation().unwrap(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        GdnPreludePhase::InitialChunk64,
+        operation,
+        heads,
+    )
+    .unwrap()
+}
+#[test]
+fn prefix_retains_f64_carry_and_stores_each_f32_prefix() {
+    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
+    let mut input = vec![0.0; c.elements()];
+    input[0] = 1.0;
+    input[1] = f32::from_bits(0x3380_0000); // 2^-24, exact half ULP at1.
+    input[2] = -1.0;
+    input[64] = -0.0;
+    input[128] = -1.0;
+    let mut output = vec![9.0; input.len()];
+    initial_prelude_into(c, &input, &mut output, None).unwrap();
+    assert_eq!(output[0].to_bits(), 0x3f80_0000);
+    assert_eq!(output[1].to_bits(), 0x3f80_0000); // Store rounded, carry not rounded.
+    assert_eq!(output[2].to_bits(), 0x3380_0000);
+    assert_eq!(output[63].to_bits(), 0x3380_0000);
+    assert_eq!(output[64].to_bits(), 0); // +0 accumulator + -0 is +0.
+    assert_eq!(output[127].to_bits(), 0);
+    assert_eq!(output[128].to_bits(), 0xbf80_0000);
+    assert_eq!(output[191].to_bits(), 0xbf80_0000);
+}
+#[test]
+fn query_multiply_preserves_signed_zero_and_subnormal_rounding() {
+    let c = contract(InitialPreludeOperation::QueryScale128, 48);
+    let mut input = vec![1.0; c.elements()];
+    input[..6].copy_from_slice(&[
+        1.0,
+        -1.0,
+        0.0,
+        -0.0,
+        f32::from_bits(1),
+        f32::from_bits(0x8000_0001),
+    ]);
+    let mut out = vec![0.0; input.len()];
+    initial_prelude_into(c, &input, &mut out, None).unwrap();
+    assert_eq!(
+        out[..6].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
+        [0x3db5_04f3, 0xbdb5_04f3, 0, 0x8000_0000, 0, 0x8000_0000]
+    );
+    assert!(out[6..].iter().all(|v| v.to_bits() == 0x3db5_04f3));
+}
+#[test]
+fn cached_profile_source_geometry_and_complete_span_refuse() {
+    for (source, profile, phase, heads) in [
+        (
+            SourceIdentity::unbound(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::GgufSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::Cached,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            32,
+        ),
+    ] {
+        assert!(
+            InitialPreludeContract::new(
+                source,
+                profile,
+                phase,
+                InitialPreludeOperation::DecayPrefix64,
+                heads
+            )
+            .is_err()
+        );
+    }
+    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
+    let mut input = vec![0.0; c.elements()];
+    let mut out = vec![7.0; c.elements()];
+    assert!(initial_prelude_into(c, &input[..input.len() - 1], &mut out, None).is_err());
+    input[0] = f32::NAN;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+    input[0] = f32::INFINITY;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+    assert!(out.iter().all(|v| *v == 7.0));
+    input[0] = 0.0;
+    assert!(initial_prelude_into(c, &input, &mut out, Some(&AtomicBool::new(true))).is_err());
+    assert!(out.iter().all(|v| *v == 7.0));
+    input[0] = f32::MAX;
+    input[1] = f32::MAX;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+}
+
+#[cfg(all(feature = "cache", target_os = "linux"))]
+mod physical {
+    use super::*;
+    use std::fs::{self, File};
+    use std::path::{Path, PathBuf};
+    use std::sync::atomic::AtomicU64;
+    use zerogravity_artifacts::artifact::Artifact;
+    use zerogravity_artifacts::cache::TransferAdmission;
+    use zerogravity_artifacts::source_weights::{
+        ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits,
+    };
+    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
+    const META: u64 = 16 * 1024 * 1024;
+    // Existing accepted SYNTHETIC source; no trained fixture or new header alias.
+    const BYTES: &[u8] = include_bytes!("../window_projection/fixture.safetensors");
+    const INDEX: &[u8] = include_bytes!("../window_projection/fixture.index.json");
+    struct Fixture(PathBuf);
+    impl Fixture {
+        fn new() -> Self {
+            static NEXT: AtomicU64 = AtomicU64::new(1);
+            let p = std::env::temp_dir().join(format!(
+                "zg-initial-prefix-{}-{}",
+                std::process::id(),
+                NEXT.fetch_add(1, Ordering::Relaxed)
+            ));
+            fs::create_dir(&p).unwrap();
+            fs::write(p.join("model.safetensors"), BYTES).unwrap();
+            Self(p)
+        }
+        fn source(&self, h: &TransferAdmission) -> VerifiedTensorSource {
+            SourceWeights::prepare_verified_windows(
+                INDEX,
+                "42e2f4619abb5431bccce39b473b23eaa90679c9947d26c9d439308c78a86149",
+                vec![
+                    ExpectedSourceShard::new(
+                        "model.safetensors".into(),
+                        Artifact::from_file(
+                            File::open(self.0.join("model.safetensors")).unwrap(),
+                            Path::new("model.safetensors"),
+                        )
+                        .unwrap(),
+                        BYTES.len() as u64,
+                        "973fca86c0585f6b11d0b686c051bad06b896f13b2b3d10ef0299eea4510f033".into(),
+                    )
+                    .unwrap(),
+                ],
+                WindowOpenLimits {
+                    max_source_bytes: 1024 * 1024,
+                    max_metadata_bytes: META,
+                    max_shards: 1,
+                    max_tensors: 6,
+                    scan_chunk_bytes: 1024,
+                },
+                h,
+            )
+            .unwrap()
+            .open()
+            .unwrap()
+        }
+    }
+    impl Drop for Fixture {
+        fn drop(&mut self) {
+            let _ = fs::remove_dir_all(&self.0);
+        }
+    }
+    fn host() -> TransferAdmission {
+        let arbiter = ResourceArbiter::new();
+        let resource = PhysicalDeviceKey::new("host", "synthetic-initial-prefix").unwrap();
+        arbiter
+            .register(resource.clone(), 64 * 1024 * 1024)
+            .unwrap();
+        TransferAdmission {
+            arbiter,
+            resource,
+            owner: ResourceOwner::new("synthetic", "initial-prefix").unwrap(),
+        }
+    }
+    fn reserved(h: &TransferAdmission) -> u64 {
+        h.arbiter.snapshot(&h.resource).unwrap().reserved
+    }
+    fn raw(c: InitialPreludeContract) -> Vec<u8> {
+        1.0f32.to_le_bytes().repeat(c.elements())
+    }
+    #[test]
+    fn admission_foreign_cancel_pressure_clone_and_lastdrop() {
+        let f = Fixture::new();
+        let h = host();
+        let source = f.source(&h);
+        let foreign = f.source(&h);
+        let c = InitialPreludeContract::new(
+            source.source_identity(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            InitialPreludeOperation::QueryScale128,
+            48,
+        )
+        .unwrap();
+        let raw = raw(c);
+        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
+        let before = reserved(&h);
+        assert!(execute_initial_prelude(&foreign, input, None).is_err());
+        assert!(execute_initial_prelude(&source, input, Some(&AtomicBool::new(true))).is_err());
+        assert_eq!(reserved(&h), before);
+        let need = InitialPreludeOutput::required_bytes(c) as u64;
+        let hold = h.arbiter.snapshot(&h.resource).unwrap().remaining - (need - 1);
+        let pressure = source.reserve_workspace(hold as usize).unwrap();
+        let pressured = reserved(&h);
+        assert!(execute_initial_prelude(&source, input, None).is_err());
+        assert_eq!(reserved(&h), pressured);
+        drop(pressure);
+        assert_eq!(reserved(&h), before);
+        let output = execute_initial_prelude(&source, input, None).unwrap();
+        let clone = output.clone();
+        assert!(output.same_issuer(&source));
+        assert!(!output.same_issuer(&foreign));
+        assert_eq!(reserved(&h), before + need);
+        assert_eq!(output.charged_bytes() as u64, need);
+        drop(foreign);
+        drop(source);
+        assert_eq!(reserved(&h), META + need);
+        assert_eq!(clone.imported_input().len(), 48 * 128);
+        assert!(clone.output().iter().all(|v| v.to_bits() == 0x3db5_04f3));
+        drop(output);
+        assert_eq!(reserved(&h), META + need);
+        drop(clone);
+        assert_eq!(reserved(&h), 0);
+    }
+    #[test]
+    fn completed_result_cancel_drops_real_admitted_backing() {
+        let f = Fixture::new();
+        let h = host();
+        let source = f.source(&h);
+        let c = InitialPreludeContract::new(
+            source.source_identity(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            InitialPreludeOperation::DecayPrefix64,
+            16,
+        )
+        .unwrap();
+        let raw = raw(c);
+        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
+        let value = execute_initial_prelude(&source, input, None).unwrap();
+        assert!(reserved(&h) > META);
+        assert!(super::super::admitted::completed(value, Some(&AtomicBool::new(true))).is_err());
+        assert_eq!(reserved(&h), META);
+        drop(source);
+        assert_eq!(reserved(&h), 0);
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude.rs
@@ -0,0 +1,155 @@
+//! Closed initial-GDN leaves; no L2/exp/chunk solver, state or model permit.
+//! Cumsum schedule: Torch cf30153 ReduceOpsKernel.cpp79–95,
+//! AccumulateType.h64–66/155. Attribution: initial_prelude.PYTORCH-LICENSE.
+use crate::accelerator::{SourceIdentity, program::ProgramNumericalProfile};
+use crate::{Error, Result};
+use std::sync::atomic::{AtomicBool, Ordering};
+
+/// Phase is explicit; cached query division is deliberately unsupported here.
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum GdnPreludePhase {
+    InitialChunk64,
+    Cached,
+}
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum InitialPreludeOperation {
+    /// One already-normalized contiguous `[heads,1,128]` FP32 query.
+    QueryScale128,
+    /// One already-padded contiguous `[heads,1,64]` FP32 decay chunk.
+    DecayPrefix64,
+}
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub struct InitialPreludeContract {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    operation: InitialPreludeOperation,
+    heads: usize,
+}
+impl InitialPreludeContract {
+    pub fn new(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        phase: GdnPreludePhase,
+        operation: InitialPreludeOperation,
+        heads: usize,
+    ) -> Result<Self> {
+        if phase != GdnPreludePhase::InitialChunk64
+            || profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported("initial GDN leaf phase/profile".into()));
+        }
+        if !source.is_bound() || !matches!(heads, 16 | 48) {
+            return Err(Error::Malformed("initial GDN leaf source/heads".into()));
+        }
+        Ok(Self {
+            source,
+            profile,
+            operation,
+            heads,
+        })
+    }
+    pub fn source_identity(self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    pub fn phase(self) -> GdnPreludePhase {
+        GdnPreludePhase::InitialChunk64
+    }
+    pub fn operation(self) -> InitialPreludeOperation {
+        self.operation
+    }
+    pub fn heads(self) -> usize {
+        self.heads
+    }
+    pub fn elements(self) -> usize {
+        self.heads
+            * match self.operation {
+                InitialPreludeOperation::QueryScale128 => 128,
+                InitialPreludeOperation::DecayPrefix64 => 64,
+            }
+    }
+}
+fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
+    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        return Err(Error::State("initial GDN leaf cancelled".into()));
+    }
+    Ok(())
+}
+fn environment() -> Result<()> {
+    #[cfg(target_arch = "x86_64")]
+    {
+        let mut mxcsr = 0u32;
+        // SAFETY: stmxcsr writes exactly four bytes to this valid local word.
+        // Passive read only; exception status bits may change during arithmetic.
+        unsafe {
+            std::arch::asm!("stmxcsr [{p}]", p = in(reg) (&mut mxcsr as *mut u32),
+                options(nostack, preserves_flags));
+        }
+        if !std::arch::is_x86_feature_detected!("avx2") || mxcsr & 0xffc0 != 0x1f80 {
+            return Err(Error::Unsupported(
+                "initial GDN pinned CPU FP controls".into(),
+            ));
+        }
+        Ok(())
+    }
+    #[cfg(not(target_arch = "x86_64"))]
+    Err(Error::Unsupported(
+        "initial GDN pinned CPU architecture".into(),
+    ))
+}
+
+/// No allocation/provider callback. Caller owns both complete, disjoint slices.
+/// Output remains private on error; successful completion grants no state effect.
+pub fn initial_prelude_into(
+    contract: InitialPreludeContract,
+    input: &[f32],
+    output: &mut [f32],
+    cancelled: Option<&AtomicBool>,
+) -> Result<()> {
+    not_cancelled(cancelled)?;
+    environment()?;
+    if input.len() != contract.elements()
+        || output.len() != input.len()
+        || input.iter().any(|v| !v.is_finite())
+    {
+        return Err(Error::Malformed(
+            "initial GDN complete finite FP32 span".into(),
+        ));
+    }
+    match contract.operation {
+        InitialPreludeOperation::QueryScale128 => {
+            // Source query * (128 ** -0.5), scalar converted once to FP32.
+            // No reciprocal/division, FMA, BF16 cast or normalization rewrite.
+            const SCALE: f32 = f32::from_bits(0x3db5_04f3);
+            for (row, target) in input.chunks_exact(128).zip(output.chunks_exact_mut(128)) {
+                not_cancelled(cancelled)?;
+                for (&value, result) in row.iter().zip(target) {
+                    *result = value * SCALE;
+                }
+            }
+        }
+        InitialPreludeOperation::DecayPrefix64 => {
+            for (row, target) in input.chunks_exact(64).zip(output.chunks_exact_mut(64)) {
+                not_cancelled(cancelled)?;
+                let mut carry = 0.0f64;
+                for (&value, result) in row.iter().zip(target) {
+                    carry += f64::from(value);
+                    *result = carry as f32; // Store each prefix; retain unrounded F64 carry.
+                    if !result.is_finite() {
+                        return Err(Error::Malformed("initial GDN prefix FP32 overflow".into()));
+                    }
+                }
+            }
+        }
+    }
+    not_cancelled(cancelled)
+}
+
+#[cfg(all(feature = "cache", target_os = "linux"))]
+mod admitted;
+#[cfg(all(feature = "cache", target_os = "linux"))]
+pub use admitted::{InitialPreludeInput, InitialPreludeOutput, execute_initial_prelude};
+#[cfg(test)]
+mod tests;
--- a/crates/zerogravity-cpu/src/original_hf.rs
+++ b/crates/zerogravity-cpu/src/original_hf.rs
@@ -4,6 +4,7 @@
 //! This does not enable complete trained Qwen3.8 inference.
 
 pub mod coefficient_preparation;
+pub mod initial_prelude;
 pub mod conv;
 pub mod norm;
 pub mod recurrent;
--- a/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs
+++ b/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs
@@ -56,13 +56,18 @@
     }
     let dt = source.parameter_descriptor(QwenParameterRole::GdnDt(layer))?;
     let a_log = source.parameter_descriptor(QwenParameterRole::GdnA(layer))?;
+    // A_log is metadata only; no exp/g or learned-parameter cast is granted.
+    let a_log_codec_valid = match a_log.kind {
+        QuantKind::F32 => a_log.bytes == heads * 4,
+        QuantKind::Bf16 => heads == 48 && a_log.bytes == 96,
+        _ => false,
+    };
     if dt.source != source.source_identity()
         || dt.name != format!("{namespace}.dt_bias")
         || a_log.source != source.source_identity()
         || a_log.name != format!("{namespace}.A_log")
-        || a_log.kind != QuantKind::F32
         || a_log.shape != [heads as u64]
-        || a_log.bytes != heads * 4
+        || !a_log_codec_valid
     {
         return Err(Error::State(
             "coefficient preparation original raw A_log/dt_bias roles".into(),
--- a/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs
@@ -269,3 +269,6 @@
     assert!(model.new_session().is_err());
     assert!(model.forward_plan(8, true, true).is_err());
 }
+
+#[path = "coefficient_preparation_h48_tests.rs"]
+mod h48;
--- /dev/null
+++ b/crates/zerogravity-model-qwen35/src/coefficient_preparation_h48_tests.rs
@@ -0,0 +1,390 @@
+//! Sealed synthetic H48 source/metadata/lease checks; no trained math claim.
+use super::*;
+use crate::accelerator::program::{ProjectionContract, SourceParameter, SourceTensor};
+use crate::operator_slice::OperatorSliceSource;
+use zerogravity_core::quant::QuantKind;
+
+fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {
+    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
+    let text = &mut cfg["text_config"];
+    text["num_hidden_layers"] = json!(2);
+    text["full_attention_interval"] = json!(2);
+    text["layer_types"] = json!(["linear_attention", "full_attention"]);
+    text["hidden_size"] = json!(256);
+    text["linear_num_key_heads"] = json!(16);
+    text["linear_num_value_heads"] = json!(48);
+    text["linear_key_head_dim"] = json!(128);
+    text["linear_value_head_dim"] = json!(128);
+    let cfg = serde_json::to_vec(&cfg).unwrap();
+    let mut header = BTreeMap::new();
+    let mut map = BTreeMap::new();
+    let mut payload = Vec::new();
+    for spec in QwenTextConfig::from_json(&cfg)
+        .unwrap()
+        .tensor_specs()
+        .unwrap()
+    {
+        let begin = payload.len();
+        let value = if spec.name.ends_with("dt_bias") {
+            0
+        } else {
+            0x3e80u16
+        };
+        for _ in 0..spec.shape.iter().product::<u64>() {
+            payload.extend_from_slice(&value.to_le_bytes());
+        }
+        header.insert(
+            spec.name.clone(),
+            json!({"dtype":"BF16", "shape":spec.shape,
+            "data_offsets":[begin,payload.len()]}),
+        );
+        map.insert(spec.name, "model.safetensors");
+    }
+    let header = serde_json::to_vec(&header).unwrap();
+    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
+    bytes.extend(header);
+    bytes.extend(&payload);
+    assert!(bytes.len() < 16 * 1024 * 1024);
+    let path = fixture.0.join("model.safetensors");
+    fs::write(&path, &bytes).unwrap();
+    let index = serde_json::to_vec(&json!({"metadata":{"total_size":payload.len()},
+        "weight_map":map}))
+    .unwrap();
+    let index_sha = digest(&index);
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "coefficient-h48-source-test").unwrap();
+    arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
+    let host = TransferAdmission {
+        arbiter,
+        resource: key,
+        owner: ResourceOwner::new("synthetic", "coefficient-h48").unwrap(),
+    };
+    let artifact =
+        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).unwrap();
+    let source = SourceWeights::prepare_verified_windows(
+        &index,
+        &index_sha,
+        vec![
+            ExpectedSourceShard::new(
+                "model.safetensors".into(),
+                artifact,
+                bytes.len() as u64,
+                digest(&bytes),
+            )
+            .unwrap(),
+        ],
+        WindowOpenLimits {
+            max_source_bytes: 16 * 1024 * 1024,
+            max_metadata_bytes: 32 * 1024 * 1024,
+            max_shards: 1,
+            max_tensors: 200,
+            scan_chunk_bytes: 64 * 1024,
+        },
+        &host,
+    )
+    .unwrap()
+    .open()
+    .unwrap();
+    (source, host, cfg, index_sha)
+}
+
+struct MetadataFault<'a> {
+    source: &'a OriginalWindowOperators,
+    a_log: SourceParameter,
+    profile: ProgramNumericalProfile,
+}
+impl OperatorSliceSource for MetadataFault<'_> {
+    fn geometry(&self) -> &crate::model::ModelGeometry {
+        self.source.geometry()
+    }
+    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
+        self.source.source_identity()
+    }
+    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    fn rms_epsilon(&self) -> f32 {
+        self.source.rms_epsilon()
+    }
+    fn matrix_descriptor(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
+        self.source.matrix_descriptor(role)
+    }
+    fn parameter_descriptor(&self, role: QwenParameterRole) -> Result<SourceParameter> {
+        if role == QwenParameterRole::GdnA(0) {
+            Ok(self.a_log.clone())
+        } else {
+            self.source.parameter_descriptor(role)
+        }
+    }
+    fn embedding_contract(&self) -> EmbeddingContract {
+        self.source.embedding_contract()
+    }
+    fn projection_contract(&self) -> ProjectionContract {
+        self.source.projection_contract()
+    }
+}
+
+#[test]
+fn only_genuine_h48_bf16_unevaluated_a_log_extent_is_admitted() {
+    use crate::operator_slice::coefficient_preparation::coefficient_roles;
+    let old = Fixture::new();
+    let loaded = coefficient_source(&old);
+    let (source, host) = windows(&old, &loaded);
+    let ops = bind_windows(source.clone(), &loaded);
+    let roles = coefficient_roles(&ops, 0).unwrap();
+    assert_eq!(
+        (roles.heads, roles.a_log.kind, roles.a_log.bytes),
+        (16, QuantKind::F32, 64)
+    );
+    let mut unsupported = roles.a_log;
+    unsupported.kind = QuantKind::Bf16;
+    unsupported.bytes = 32;
+    assert!(
+        coefficient_roles(
+            &MetadataFault {
+                source: &ops,
+                a_log: unsupported,
+                profile: ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+            },
+            0
+        )
+        .is_err()
+    );
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&host), 0);
+
+    let fixture = Fixture::new();
+    let (source, host, cfg, index) = h48_source(&fixture);
+    let ops = OriginalWindowOperators::bind(
+        source.clone(),
+        &cfg,
+        &digest(&cfg),
+        &index,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )
+    .unwrap();
+    let roles = coefficient_roles(&ops, 0).unwrap();
+    assert_eq!(
+        (
+            roles.heads,
+            roles.a_log.kind,
+            roles.a_log.shape.clone(),
+            roles.a_log.bytes
+        ),
+        (48, QuantKind::Bf16, vec![48], 96)
+    );
+    assert_eq!(
+        (roles.dt.kind, roles.dt.shape, roles.dt.bytes),
+        (QuantKind::Bf16, vec![48], 96)
+    );
+    let before = reserved(&host);
+    for (kind, bytes, shape, foreign, name, profile) in [
+        (
+            QuantKind::F16,
+            96,
+            vec![48],
+            false,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            192,
+            vec![48],
+            false,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::F32,
+            96,
+            vec![48],
+            false,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            96,
+            vec![47],
+            false,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            96,
+            vec![48],
+            true,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            96,
+            vec![48],
+            false,
+            "foreign.A_log".into(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        ),
+        (
+            QuantKind::Bf16,
+            96,
+            vec![48],
+            false,
+            roles.a_log.name.clone(),
+            ProgramNumericalProfile::GgufSourceV1,
+        ),
+    ] {
+        let mut parameter = roles.a_log.clone();
+        parameter.kind = kind;
+        parameter.bytes = bytes;
+        parameter.shape = shape;
+        parameter.name = name;
+        if foreign {
+            parameter.source = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
+        }
+        assert!(
+            coefficient_roles(
+                &MetadataFault {
+                    source: &ops,
+                    a_log: parameter,
+                    profile
+                },
+                0
+            )
+            .is_err()
+        );
+        assert_eq!(reserved(&host), before);
+    }
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&host), 0);
+}
+
+#[test]
+fn h48_preparation_preserves_source_lease_cancel_pressure_and_all_private_words() {
+    let fixture = Fixture::new();
+    let (source, host, cfg, index) = h48_source(&fixture);
+    let ops = OriginalWindowOperators::bind(
+        source.clone(),
+        &cfg,
+        &digest(&cfg),
+        &index,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )
+    .unwrap();
+    let a = 0x41a8u16.to_le_bytes().repeat(48);
+    let b = 0u16.to_le_bytes().repeat(48);
+    let make = |identity| input(identity, &a, &b);
+    let before = reserved(&host);
+    for (layer, identity, cancel) in [
+        (1, source.source_identity(), None),
+        (
+            0,
+            crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
+            None,
+        ),
+        (0, source.source_identity(), Some(AtomicBool::new(true))),
+    ] {
+        assert!(
+            ops.execute_gdn_coefficient_preparation_slice(layer, make(identity), cancel.as_ref())
+                .is_err()
+        );
+        assert_eq!(reserved(&host), before);
+    }
+    let mut pressure = Vec::new();
+    loop {
+        let remaining = host.arbiter.snapshot(&host.resource).unwrap().remaining;
+        if remaining == 0 {
+            break;
+        }
+        pressure.push(
+            source
+                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
+                .unwrap(),
+        );
+    }
+    let pressured = reserved(&host);
+    assert!(
+        ops.execute_gdn_coefficient_preparation_slice(0, make(source.source_identity()), None)
+            .is_err()
+    );
+    assert_eq!(reserved(&host), pressured);
+    drop(pressure);
+    assert_eq!(reserved(&host), before);
+    let output = ops
+        .execute_gdn_coefficient_preparation_slice(0, make(source.source_identity()), None)
+        .unwrap();
+    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 48]);
+    assert_eq!(output.beta_input_bf16().unwrap(), &[0.0; 48]);
+    assert_eq!(output.biased_alpha_f32().unwrap(), &[21.0; 48]);
+    assert_eq!(output.softplus_f32().unwrap(), &[21.0; 48]);
+    assert_eq!(output.beta_bf16().unwrap(), &[0.5; 48]);
+    assert_eq!(
+        output.unevaluated_raw_a_log_parameter().kind,
+        QuantKind::Bf16
+    );
+    assert_eq!(reserved(&host), before + 2 * 1024 * 1024);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&host), 34 * 1024 * 1024);
+    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 48]);
+    assert_eq!(output.beta_bf16().unwrap(), &[0.5; 48]);
+    drop(output);
+    assert_eq!(reserved(&host), 0);
+}
+
+#[test]
+fn h48_recording_adds_only_existing_two_preparation_nodes() {
+    let fixture = Fixture::new();
+    let (source, host, cfg, index) = h48_source(&fixture);
+    let ops = OriginalWindowOperators::bind(
+        source.clone(),
+        &cfg,
+        &digest(&cfg),
+        &index,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )
+    .unwrap();
+    let mut builder = PlanBuilder::new_operator_slice(
+        source.source_identity(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+    )
+    .unwrap();
+    // Same-builder views exercise transport; no authentic projection is claimed.
+    let embed = builder
+        .embed_profiled(
+            ops.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
+            PlanBuilder::inputs().token,
+            EmbeddingContract::Bf16OutputInput,
+        )
+        .unwrap();
+    let a = builder.view(&embed, TensorView::vector(0, 48)).unwrap();
+    let b = builder.view(&embed, TensorView::vector(48, 48)).unwrap();
+    let values = ops
+        .append_gdn_coefficient_preparation_slice(&mut builder, 0, &a, &b)
+        .unwrap();
+    let plan = builder.finish_operator_slice(values.softplus).unwrap();
+    assert_eq!(plan.plan().nodes().len(), 5);
+    assert_eq!(
+        plan.plan()
+            .nodes()
+            .iter()
+            .filter(|n| matches!(
+                n.operation,
+                PlanOperation::TypedGdnAlphaPreparation { .. }
+                    | PlanOperation::TypedGdnBetaPreparation { .. }
+            ))
+            .count(),
+        2
+    );
+    assert!(plan.plan().required_domains().is_empty());
+    assert!(plan.plan().proposed_domains().is_empty());
+    assert!(plan.plan().logits().is_none());
+    assert_eq!(plan.plan().frontier_delta(), 0);
+    drop(ops);
+    drop(source);
+    assert_eq!(reserved(&host), 0);
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude.PYTORCH-LICENSE
@@ -0,0 +1,84 @@
+From PyTorch:
+
+Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
+Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
+Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
+Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
+Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
+Copyright (c) 2011-2013 NYU                      (Clement Farabet)
+Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
+Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
+Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
+
+From Caffe2:
+
+Copyright (c) 2016-present, Facebook Inc. All rights reserved.
+
+All contributions by Facebook:
+Copyright (c) 2016 Facebook Inc.
+
+All contributions by Google:
+Copyright (c) 2015 Google Inc.
+All rights reserved.
+
+All contributions by Yangqing Jia:
+Copyright (c) 2015 Yangqing Jia
+All rights reserved.
+
+All contributions by Kakao Brain:
+Copyright 2019-2020 Kakao Brain
+
+All contributions by Cruise LLC:
+Copyright (c) 2022 Cruise LLC.
+All rights reserved.
+
+All contributions by Tri Dao:
+Copyright (c) 2024 Tri Dao.
+All rights reserved.
+
+All contributions by Arm:
+Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
+
+All contributions from Caffe:
+Copyright(c) 2013, 2014, 2015, the respective contributors
+All rights reserved.
+
+All other contributions:
+Copyright(c) 2015, 2016 the respective contributors
+All rights reserved.
+
+Caffe2 uses a copyright model similar to Caffe: each contributor holds
+copyright over their contributions to Caffe2. The project versioning records
+all such contribution and copyright details. If a contributor wants to further
+mark their specific copyright on a particular contribution, they should
+indicate their copyright solely in the commit message of the change when it is
+committed.
+
+All rights reserved.
+
+Redistribution and use in source and binary forms, with or without
+modification, are permitted provided that the following conditions are met:
+
+1. Redistributions of source code must retain the above copyright
+   notice, this list of conditions and the following disclaimer.
+
+2. Redistributions in binary form must reproduce the above copyright
+   notice, this list of conditions and the following disclaimer in the
+   documentation and/or other materials provided with the distribution.
+
+3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
+   and IDIAP Research Institute nor the names of its contributors may be
+   used to endorse or promote products derived from this software without
+   specific prior written permission.
+
+THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
+AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
+IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
+ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
+LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
+CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
+SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
+INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
+CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
+ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
+POSSIBILITY OF SUCH DAMAGE.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/BOUNDED_ORACLE_IO_V2.patch

Original UTF-8 bytes: 3395; SHA256 `edd6d71d802a44e24aa7590c1f0ac2a4ffb3f54f4e9ed70078563137f6c2ba72`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/reference.py
+++ b/reference.py
@@ -7,6 +7,7 @@
 import os
 from pathlib import Path
 import struct
+import stat
 
 
 def require(ok, message):
@@ -18,12 +19,23 @@
     return hashlib.sha256(raw).hexdigest()
 
 
+def small(path, cap):
+    with os.fdopen(os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW),'rb') as opened:
+        before=os.fstat(opened.fileno())
+        require(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'bounded regular operator metadata/operand')
+        data=opened.read(cap+1)
+        after=os.fstat(opened.fileno())
+    identity=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
+    require(len(data)<=cap and identity(before)==identity(after),'bounded immutable selected input FD')
+    return data
+
+
 def checked(root, record):
     name=record['file']
     require(type(name) is str and Path(name).name == name, 'contained selected operand')
     path=root/name
     require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 128*1024, 'bounded raw operand')
-    raw=path.read_bytes()
+    raw=small(path,128*1024)
     require(len(raw) == record['bytes'] and sha(raw) == record['sha256'], 'exact selected operand pin')
     return raw
 
@@ -59,8 +71,8 @@
     require(torch.backends.cpu.get_cpu_capability() == 'AVX2' and not torch.cuda.is_available()
             and torch.backends.mkldnn.enabled, 'original CPU-only AVX2/mkldnn policy')
     source=Path('/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py')
-    require(sha(source.read_bytes()) == '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b', 'unchanged official convolution caller')
-    document=args.inputs.read_bytes()
+    require(sha(small(source,1024*1024)) == '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b', 'unchanged official convolution caller')
+    document=small(args.inputs,128*1024)
     require(len(document)<=128*1024 and sha(document)==args.inputs_sha256,'exact prepared inputs metadata')
     inputs=json.loads(document)
     require(inputs['schema']=='zerogravity.original-conv-fragment.inputs.v1' and inputs['profile']=='source-36-f32-restored','genuine input protocol')
--- a/run_replay.py
+++ b/run_replay.py
@@ -36,12 +36,12 @@
 
 
 def frozen():
-    manifest = read_json(HERE / "freeze-manifest.json")
+    manifest = read_json(HERE / "freeze-manifest-v2.json")
     for item in manifest["files"]:
         pin(HERE / item["file"], item)
     for item in manifest["main_sources"]:
         pin(item["path"], item)
-    return sha(HERE / "freeze-manifest.json")
+    return sha(HERE / "freeze-manifest-v2.json")
 
 
 def execute(binary, binary_fd, inputs, inputs_sha, reference, reference_sha, output, model, cpus, freeze, source_before):
--- a/ROOT-QUALIFICATION.md
+++ b/ROOT-QUALIFICATION.md
@@ -1,6 +1,7 @@
 # ROOT-only genuine qualification
 
-Status: PREPARED / UNQUALIFIED. Author parsed/formatted source only. No raw
+Status: PREPARED / UNQUALIFIED. Latest additive freeze: `freeze-manifest-v2.json`;
+initial `freeze-manifest.json`/`frozen-prepared/` retained unchanged. Author parsed/formatted source only. No raw
 model/tensor payload was hashed or operator/reference/harness executed.
 `bindings.json` copies expected source/capture hashes from genuine existing
 receipts; author independently read/pinned only small source/metadata.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/GPU_REFUSAL.patch

Original UTF-8 bytes: 1645; SHA256 `f09d057c3c65d75907f85ade140c45c05465d20b9b831a254b86475b8299e419`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -838,7 +838,8 @@
             match &node.operation {
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
-                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedOrdinaryRms { .. }
+                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                     "resident batch has no qualified typed lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -348,6 +348,7 @@
                     PlanOperation::TypedEmbed { .. }
                         | PlanOperation::TypedProject { .. }
                         | PlanOperation::TypedOrdinaryRms { .. }
+                        | PlanOperation::TypedCausalConvSilu { .. }
                 )
             })
         {
@@ -1429,7 +1430,8 @@
             match &node.operation {
                 PlanOperation::TypedEmbed { .. }
                 | PlanOperation::TypedProject { .. }
-                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
+                | PlanOperation::TypedOrdinaryRms { .. }
+                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                     "resident typed operation has no qualified lowering".into(),
                 )),
                 PlanOperation::Embed { tensor, .. } => {

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/HOSTMATH_DROP_TEST_ONLY_V3.patch

Original UTF-8 bytes: 578; SHA256 `c06dc2324bf450e23ce0fb20592d9bf29a1b80d17254e2c2475d16db98a20de1`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-model-qwen35/src/gdn_conv_window_tests.rs
+++ b/crates/zerogravity-model-qwen35/src/gdn_conv_window_tests.rs
@@ -241,7 +241,6 @@
     assert_eq!(clone.as_slice().unwrap().len(), 32);
     drop(clone);
     assert_eq!(reserved(&admission), after_import);
-    drop(math);
     drop(activation);
     drop(history);
     assert_eq!(reserved(&admission), baseline);
@@ -320,9 +319,7 @@
             )
             .is_err()
     );
-    drop(foreign);
     drop(foreign_ops);
-    drop(math);
     drop(strided);
     drop(activation);
     drop(history);

````

