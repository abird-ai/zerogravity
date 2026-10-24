# Draft Patches — volume 25

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/NATURAL_C8_ADDITIVE.patch

Original UTF-8 bytes: 226969; SHA256 `ba76f30083b6741ecc9328e76cd766dded1df16c0253bbd5b54b5f4881eafc5c`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/measure_natural_c8.py
@@ -0,0 +1,809 @@
+#!/usr/bin/env python3
+"""ROOT ONLY: bounded natural concurrency8 same-Zerogravity observations.
+
+Both providers are Zerogravity. This never invokes serve-compare, supplies a
+vLLM slot, evicts caches, or infers source/model/performance qualification.
+Preparation is source-only; --run explicitly starts the owned GPU processes.
+"""
+from __future__ import annotations
+
+import argparse
+from datetime import datetime, timezone
+import hashlib
+import importlib.util
+import json
+import math
+import os
+from pathlib import Path
+import re
+import socket
+import subprocess
+import tempfile
+import threading
+import time
+import urllib.request
+
+ROOT = Path(__file__).resolve().parents[3]
+OWN = Path(__file__).resolve().parent
+ROOT_RUN = ROOT / ".agents/runs/serving-and-speculation-parity-root-2026-10-19"
+LABELS = {"accepted_d0prep_zerogravity", "native_prefix_zerogravity"}
+TOOL_PINS = {
+    "serve-bench.py": "e119af67a0b3dd384c76b4d2745f8ebf6807cb27832b132ad2341580e9c5373f",
+    "serve-lifecycle.py": "b3f9c76f872711edbe8eea5731f6f85c3ba119e86386b8dd50798b507fae9cd3",
+}
+MODELS = {
+    "0.8b": {"source": "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
+             "config": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"},
+    "2b": {"source": "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6",
+           "config": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"},
+}
+TOKENIZER_SHA = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
+ACCEPTED_SHA = "a896a455ec5d9093e88706751a9450170cb68b277658268ba30082f85e7ce598"
+ACCEPTED_RECEIPT_SHA = "148f990a06da8a94661304e2b89834c631fc6fc2662bd130d9e2f8dda9358ffb"
+PHASE_COUNTERS = ["gpu_fused_state_only_batches", "gpu_fused_state_only_rows", "gpu_fused_greedy_batches"]
+# Separate predeclared workload; original strict B8 driver and its failed
+# attempt remain immutable. This script has no import/report-requalification mode.
+COUNTER_SOURCE_CONTRACT = {'accepted_head': 'c5eb1f8e717c79146937e585a145a0052eff747b',
+ 'accepted_head_files': {'crates/zerogravity-cli/src/serve/gpu_engine.rs': 'e03886e60c1ea8426b11fe20a827454ea51295aa93fd40811a18ac8c6b073c86',
+                         'crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs': '16783bd008babcecd600b18ede3ae8e598407673e01c6dbab3c8be003660739f',
+                         'crates/zerogravity-cli/src/serve/mod.rs': 'e10cd99e6b174139f8949477726a89f1d04fab5eec6568bc1d4bd30e937c444e'},
+ 'accepted_files': {'crates/zerogravity-cli/src/serve/gpu_engine.rs': 'e03886e60c1ea8426b11fe20a827454ea51295aa93fd40811a18ac8c6b073c86',
+                    'crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs': '16783bd008babcecd600b18ede3ae8e598407673e01c6dbab3c8be003660739f',
+                    'crates/zerogravity-cli/src/serve/mod.rs': '84e30f0a8909dcf677c8d6444d8a3b21ebe297e2e69acd7ca90b9fdc118afc37'},
+ 'current_files': {'crates/zerogravity-cli/src/serve/gpu_engine.rs': 'f44c60e4d3360825480aa9ec970280aedba7deac2d152cdf799cf025a0ac4279',
+                   'crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs': 'f57b08e6b48390f129334436168a2a32f421a9b54c113f092b0d07bcfd8ff14a',
+                   'crates/zerogravity-cli/src/serve/mod.rs': '9430fb01cc7d971ccc936e91ac3f40ee71f20c1d3afcf6710bc7962bedb9f2ad'}}
+ORIGINAL_PREPARED_SHA = "be0bbee8a028fbe123478297aa57f1a495de1343a69ecb70100dff6cc48605ee"
+
+CURRENT_SOURCE_FIELDS = ["crates/zerogravity-cli/src/serve/gpu_engine.rs",
+    "crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs", "crates/zerogravity-cli/src/serve/mod.rs",
+    "crates/zerogravity-gpu/src/resident_inference/fused.rs"]
+COUNTERS = ["completed_forward_kernel_dispatches", "gpu_forward_calls_completed",
+            "completed_model_token_steps", "committed_gpu_roots",
+            "native_projection_dispatches_completed", "native_sampler_dispatches_completed",
+            "accepted_output_tokens", "gpu_fused_batches", "gpu_fused_rows_completed",
+            "native_admission_refusals", "discarded_ready_cohorts", "failed_gpu_quanta",
+            "cancelled_requests", "state_cohort_publications", "state_cohort_rows"]
+GAUGES = ["queued_requests", "active_sessions", "output_credit_parked_requests"]
+PROMPT = "Continue this list of integers, separated by commas, for many more terms: 1, 2, 3, 4, 5, 6, 7, 8,"
+
+
+def require(condition, message):
+    if not condition:
+        raise ValueError(message)
+
+
+def sha(path):
+    with Path(path).open("rb") as source:
+        return hashlib.file_digest(source, "sha256").hexdigest()
+
+
+def json_read(path, cap=1024 * 1024):
+    with Path(path).open("rb") as source:
+        raw = source.read(cap + 1)
+    require(len(raw) <= cap, "JSON input exceeds bound")
+    return json.loads(raw)
+
+
+def module(name):
+    path = ROOT / "tools" / name
+    require(sha(path) == TOOL_PINS[name], f"measurement/lifecycle source changed: {name}; root must review and repin")
+    spec = importlib.util.spec_from_file_location(name.replace("-", "_").replace(".", "_"), path)
+    require(spec is not None and spec.loader is not None, "module loader")
+    value = importlib.util.module_from_spec(spec)
+    # Dataclass module identity must exist before loading serve-bench.
+    import sys
+    sys.modules[spec.name] = value
+    spec.loader.exec_module(value)
+    return value
+
+
+def local_opener():
+    class NoRedirect(urllib.request.HTTPRedirectHandler):
+        def redirect_request(self, *_args, **_kwargs):
+            raise ValueError("local provider redirects refused")
+    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
+
+
+def get_json(url):
+    with local_opener().open(url, timeout=2) as response:
+        require(response.status == 200, "provider readiness HTTP status")
+        raw = response.read(1024 * 1024 + 1)
+    require(len(raw) <= 1024 * 1024, "provider JSON response bound")
+    return json.loads(raw)
+
+
+def health(url, label):
+    value = get_json(url + "/health")
+    require(value.get("status") == "ok" and value.get("per_request_metrics_enabled") is True,
+            "actual per-request metrics not enabled")
+    admission = value.get("admission")
+    require(isinstance(admission, dict) and admission.get("kind") == "resident_gpu"
+            and admission.get("batch_mode") == "fused" and admission.get("native_batch_rows_max") == 8,
+            "actual configured GPU fused max8 differs")
+    for name in COUNTERS + GAUGES:
+        require(type(admission.get(name)) is int and admission[name] >= 0, f"health field absent/invalid: {name}")
+    visibility = [field in admission for field in PHASE_COUNTERS]
+    require(all(visibility) if label == "native_prefix_zerogravity" else not any(visibility),
+            "actual phase-counter visibility differs from frozen source: legacy absence is not observed zero")
+    for field in PHASE_COUNTERS:
+        if field in admission:
+            require(type(admission[field]) is int and admission[field] >= 0, "invalid observed phase counter")
+    bands = admission.get("gpu_fused_batches_by_rows")
+    require(isinstance(bands, dict) and all(type(bands.get(str(width))) is int and bands[str(width)] >= 0
+                                          for width in [2, 4, 8]), "actual native row-band counters absent")
+    return value
+
+
+def quiescent(url, label):
+    deadline = time.monotonic() + 10
+    prior = None
+    while time.monotonic() < deadline:
+        current = health(url, label)
+        admission = current["admission"]
+        if admission["queued_requests"] == admission["active_sessions"] == 0:
+            if current == prior:
+                return current
+            prior = current
+        else:
+            prior = None
+        time.sleep(.1)
+    raise ValueError("bounded provider health did not become quiescent; no next sample")
+
+
+def deltas(before, after):
+    left, right = before["admission"], after["admission"]
+    result = {}
+    for name in COUNTERS:
+        require(right[name] >= left[name], f"counter decreased during owned sample: {name}")
+        result[name] = right[name] - left[name]
+    for field in PHASE_COUNTERS:
+        require((field in left) == (field in right), "phase-counter visibility changed during sample")
+        if field in left:
+            require(right[field] >= left[field], "phase counter decreased")
+            result[field] = right[field] - left[field]
+        else:
+            result[field] = None
+    result["phase_counter_observability"] = (
+        "observed_native_prefix_phase_counters" if all(field in left for field in PHASE_COUNTERS)
+        else "legacy_absent_not_observed")
+    result["native_batches_by_rows"] = {}
+    for width in ["2", "4", "8"]:
+        a, b = left["gpu_fused_batches_by_rows"][width], right["gpu_fused_batches_by_rows"][width]
+        require(b >= a, "native band counter decreased")
+        result["native_batches_by_rows"][width] = b - a
+    return result
+
+
+def gauge_observations(before, after):
+    return {name: dict(before=before["admission"][name], after=after["admission"][name])
+            for name in GAUGES}
+
+
+def gpu(uuid):
+    def query(fields, apps=False):
+        option = "--query-compute-apps=" if apps else "--query-gpu="
+        command = ["nvidia-smi", "--id=" + uuid, option + fields, "--format=csv,noheader,nounits"]
+        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                                stderr=subprocess.PIPE, timeout=3, check=True)
+        require(len(result.stdout) <= 32768 and len(result.stderr) <= 32768, "bounded NVIDIA observation output")
+        return result.stdout.decode().strip()
+    parts = [item.strip() for item in query("uuid,memory.used,memory.total,utilization.gpu,temperature.gpu").split(",")]
+    require(len(parts) == 5 and parts[0] == uuid, "exact NVIDIA UUID observation")
+    result = dict(uuid=parts[0], memory_used_mib=int(parts[1]), memory_total_mib=int(parts[2]),
+                  utilization_percent=int(parts[3]), temperature_c=int(parts[4]))
+    result["compute_apps_raw"] = query("pid,used_gpu_memory", True)
+    require(all(0 <= result[field] <= 1_000_000 for field in ["memory_used_mib", "memory_total_mib",
+            "utilization_percent", "temperature_c"]), "finite GPU observation domain")
+    result["observed_monotonic_seconds"] = time.monotonic()
+    return result
+
+
+def gpu_idle(uuid, ceiling=None):
+    deadline, samples = time.monotonic() + 30, []
+    consecutive = 0
+    while time.monotonic() < deadline:
+        sample = gpu(uuid)
+        samples.append(sample)
+        limit = sample["memory_used_mib"] if ceiling is None else ceiling
+        if not sample["compute_apps_raw"] and sample["utilization_percent"] <= 5 and sample["memory_used_mib"] <= limit:
+            consecutive += 1
+            if consecutive == 3:
+                return samples
+        else:
+            consecutive = 0
+        time.sleep(.25)
+    raise ValueError("GPU not observably idle after owned cleanup; preserve other processes and stop")
+
+
+class Sampler:
+    """Read-only GPU/proc samples, including process startup; bounded retention."""
+    def __init__(self, uuid):
+        self.uuid, self.pid = uuid, None
+        self.samples, self.error, self.peak_gpu_mib, self.peak_rss_kib = [], None, 0, 0
+        self.stop_event = threading.Event()
+        self.thread = threading.Thread(target=self.observe, daemon=True)
+        self.thread.start()
+
+    def observe(self):
+        while not self.stop_event.is_set():
+            try:
+                row = gpu(self.uuid)
+                if self.pid is not None:
+                    try:
+                        status = Path(f"/proc/{self.pid}/status").read_text()
+                        row["process_memory_kib"] = {name: int(match.group(1))
+                            for name in ["VmRSS", "VmHWM", "VmPeak"]
+                            if (match := re.search(r"^" + name + r":\s*([0-9]+) kB$", status, re.M))}
+                        self.peak_rss_kib = max(self.peak_rss_kib, row["process_memory_kib"].get("VmRSS", 0),
+                                                row["process_memory_kib"].get("VmHWM", 0))
+                    except FileNotFoundError:
+                        pass
+                self.peak_gpu_mib = max(self.peak_gpu_mib, row["memory_used_mib"])
+                if len(self.samples) < 2048:
+                    self.samples.append(row)
+            except BaseException as error:
+                self.error = type(error).__name__
+                return
+            self.stop_event.wait(.5)
+
+    def finish(self):
+        self.stop_event.set()
+        self.thread.join(8)
+        require(not self.thread.is_alive(), "owned read-only sampler did not finish")
+        require(self.error is None, f"memory observation failed: {self.error}")
+        return dict(samples=self.samples, retained_sample_cap=2048, peak_gpu_device_mib=self.peak_gpu_mib,
+                    peak_owned_process_rss_kib=self.peak_rss_kib,
+                    scope="nvidia-smi device/proc observations; not physical ResourceArbiter admission or exact allocator peak; .5s sampling can miss peaks")
+
+
+def plan_type(lifecycle):
+    class SameEnginePlan(lifecycle.LifecyclePlan):
+        """Honest labels; inherited owned start/sample/cleanup implementation.
+
+        Only initialization is adapted. No global PROVIDERS mutation, synthetic
+        vLLM provider, alternate child shutdown or arbitrary stop commands.
+        """
+        def __init__(self, providers, endpoints, policy, directory):
+            require(set(providers) == set(endpoints) == LABELS, "same-Zerogravity provider labels")
+            require(policy in {"cold_process", "warmed_process"}, "explicit process policy")
+            self.endpoints = {label: lifecycle.endpoint(url) for label, url in endpoints.items()}
+            require(len(set(self.endpoints.values())) == 2, "distinct owned endpoint ports")
+            for label, provider in providers.items():
+                require(set(provider) == {"argv", "cwd", "env", "declarations"}, "fixed direct source provider spec")
+                lifecycle.argv(provider["argv"])
+                require(Path(provider["argv"][0]).is_absolute() and Path(provider["cwd"]) == ROOT,
+                        "explicit source binary/cwd")
+                require("vllm" not in provider["declarations"]["runtime"].lower(), "both providers must be Zerogravity")
+            self.spec = {"policy": policy, "startup_timeout_seconds": 240,
+                         "shutdown_timeout_seconds": 30, "log_limit_bytes": 4 * 1024 * 1024,
+                         "providers": providers}
+            directory.mkdir(mode=0o700)
+            self.directory = directory
+            self.current, self.blocked, self.receipts = None, False, []
+            self.manifest = {"schema": "zerogravity.same-engine.owned-lifecycle.v1", "policy": policy,
+                "receipts": self.receipts, "providers": sorted(LABELS),
+                "implementation_sha256": TOOL_PINS["serve-lifecycle.py"],
+                "semantics": "Both providers ARE Zerogravity; original OwnedProvider/sample/cleanup implementation; fresh process per sample; OS/storage/device caches are not evicted or claimed cold"}
+            self.save()
+    return SameEnginePlan
+
+
+def workload(max_tokens, count):
+    return [dict(request_id=f"request-{index}", prompt=PROMPT, arrival_seconds=0.0,
+                 tenant=f"workload-label-{index % 2}", max_tokens=max_tokens) for index in range(count)]
+
+
+def settled_result(result, count, max_tokens):
+    require(all(type(result.get(field)) is int and result[field] == count for field in
+                ["offered_requests", "completed_requests", "http_accepted_requests"])
+            and all(type(result.get(field)) is int and result[field] == 0 for field in
+                ["unsuccessful_requests", "completions_without_usage", "http_429_requests", "http_503_requests",
+                 "rejected_requests", "cancelled_requests", "timed_out_requests", "client_overload_requests", "launch_deadline_requests"])
+            and result["classifications"] == {"completed": count} and result["errors"] == {}
+            and type(result["usage_reported_completion_tokens"]) is int
+            and result["usage_reported_completion_tokens"] == count * max_tokens
+            and len(result["requests"]) == count,
+            "all offers/HTTP outcomes/terminal usage accounted; no excluded or failed requests")
+    identities, providers = set(), set()
+    for row in result["requests"]:
+        identity, provider = row["request_id"], row["provider_response_id"]
+        require(identity not in identities and row["wire_request_id"] == result["run_id"] + ":" + identity
+                and isinstance(provider, str) and 0 < len(provider) <= 256 and provider not in providers,
+                "unique bound actual offered/wire/provider response identities")
+        identities.add(identity)
+        providers.add(provider)
+        require(row["completed"] is True and row["classification"] == "completed" and row["http_status"] == 200
+                and row["http_accepted"] is True and row["http_attempted"] is True and row["error"] is None
+                and row["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
+                and type(row["max_tokens"]) is int and row["max_tokens"] == max_tokens
+                and type(row["usage_reported_completion_tokens"]) is int
+                and row["usage_reported_completion_tokens"] == max_tokens and row["finish_reason"] == "length",
+                "same actual prompt and complete HTTP terminal/usage; EOS-shortened samples refuse")
+    require(identities == {f"request-{i}" for i in range(count)}, "every offered identity retained")
+
+
+def signature(result, count):
+    settled_result(result, count, 8)
+    metrics = result["wire_token_metrics"]
+    require(metrics["requested"] is True and metrics["completed_requests_with_individual_token_delivery"] == count
+            and metrics["completed_requests_with_provider_commit_observations"] == count,
+            "all offers have individually observed IDs and distinct provider commit observations")
+    rows = {}
+    for row in result["requests"]:
+        observed = row.get("wire_token_observation") or {}
+        ids = observed.get("token_ids")
+        require(observed.get("status") == "provider_bound_complete_ids"
+                and observed.get("client_delivery_status") == "individual_token_ids_observed"
+                and observed.get("provider_commit_observation_status") == "provider_bound_complete_commit_observations"
+                and isinstance(ids, list) and len(ids) == row["max_tokens"] == 8
+                and all(type(token) is int and 0 <= token < 248320 for token in ids)
+                and len(ids) == row["usage_reported_completion_tokens"]
+                and isinstance(row["output_sha256"], str) and re.fullmatch(r"[a-f0-9]{64}", row["output_sha256"]),
+                "complete actual IDs/delivery/commit/usage required")
+        records = observed["provider_token_records"]
+        require(len(records) == 8 and all(isinstance(token.get("gpu"), dict) for token in records),
+                "all exact GPU packet-producing committed frontiers required")
+        frontiers = [token["gpu"]["producer_frontier"] for token in records]
+        require(type(frontiers[0]) is int and 1 < frontiers[0] <= 504
+                and all(type(frontier) is int for frontier in frontiers)
+                and frontiers == list(range(frontiers[0], frontiers[0]+8)), "actual consecutive producer roots")
+        identity = row["request_id"]
+        require(identity not in rows, "duplicate measured request identity")
+        rows[identity] = dict(prompt_sha256=row["prompt_sha256"], max_tokens=row["max_tokens"],
+            token_ids=ids, usage_completion_tokens=row["usage_reported_completion_tokens"],
+            producer_frontiers=frontiers, output_sha256=row["output_sha256"], finish_reason=row["finish_reason"])
+    require(set(rows) == {f"request-{i}" for i in range(count)}, "all offered identities retained")
+    require(len({tuple(row["producer_frontiers"]) for row in rows.values()}) == 1, "same actual teacher length")
+    require(all(row == next(iter(rows.values())) for row in rows.values()),
+            "same-prompt greedy IDs/text/frontiers/usage/finish agree across all offered requests")
+    return rows
+
+
+def output_cohorts(result, count):
+    require(count in [1, 8], "closed offered concurrency scope")
+    cohorts, singles, prior_sequence = {}, [], {}
+    for row in result["requests"]:
+        identity = row["request_id"]
+        observed = row["wire_token_observation"]
+        for index, token in enumerate(observed["provider_token_records"]):
+            gpu_packet = token["gpu"]
+            require("native_cohort" in gpu_packet,
+                    "explicit actual native-or-B1 packet provenance required")
+            member = dict(request_id=identity, output_index=index,
+                          producer_frontier=gpu_packet["producer_frontier"],
+                          token_id=observed["token_ids"][index],
+                          wire_request_id=row["wire_request_id"],
+                          provider_response_id=row["provider_response_id"])
+            cohort = gpu_packet["native_cohort"]
+            if cohort is None:
+                singles.append(member)
+                continue
+            require(count == 8 and isinstance(cohort, dict)
+                    and set(cohort) == {"width", "publication_sequence"}
+                    and type(cohort["width"]) is int and cohort["width"] in [2, 4, 8],
+                    "actual native2/4/8 cohort metadata; c8 is not a width8 assertion")
+            sequence, width = cohort["publication_sequence"], cohort["width"]
+            require(type(sequence) is int and 0 < sequence < 2**64,
+                    "actual bounded native publication sequence")
+            require(sequence > prior_sequence.get(identity, 0),
+                    "per-request native packet publication order")
+            prior_sequence[identity] = sequence
+            group = cohorts.setdefault(sequence, dict(width=width, members={}))
+            require(group["width"] == width and identity not in group["members"],
+                    "one consistent width and distinct actual session/request per cohort")
+            group["members"][identity] = member
+    bands = {"2": 0, "4": 0, "8": 0}
+    for group in cohorts.values():
+        require(len(group["members"]) == group["width"],
+                "every actual native greedy cohort has complete distinct width membership")
+        bands[str(group["width"])] += 1
+    native_rows = sum(group["width"] for group in cohorts.values())
+    require(native_rows + len(singles) == 8 * count,
+            "every accepted output belongs to one actual native cohort or B1 packet")
+    return dict(observed_greedy_cohorts=len(cohorts), native_greedy_rows=native_rows,
+                observed_greedy_bands=bands, cohort_members=cohorts,
+                observed_B1_packet_outputs=len(singles), B1_packets=singles,
+                scope="actual source-bound post-publication packet observations; independent frontiers may have different output indices inside one native cohort; no independent state/math proof")
+
+
+def counters(result, label, count):
+    require(label in LABELS and count in [1, 8], "closed same-engine source/workload scope")
+    outputs = signature(result, count)
+    observed = output_cohorts(result, count)
+    snapshots = result["same_engine_quiescent_health"]
+    delta = snapshots["delta"]
+    require(all(type(delta.get(field)) is int and delta[field] >= 0 for field in COUNTERS)
+            and set(delta["native_batches_by_rows"]) == {"2", "4", "8"}
+            and all(type(value) is int and value >= 0 for value in delta["native_batches_by_rows"].values()),
+            "actual finite nonnegative completed-work counters")
+    if label == "native_prefix_zerogravity":
+        require(delta["phase_counter_observability"] == "observed_native_prefix_phase_counters"
+                and all(type(delta.get(field)) is int and delta[field] >= 0 for field in PHASE_COUNTERS),
+                "new phase counters observed, never inferred")
+    else:
+        require(delta["phase_counter_observability"] == "legacy_absent_not_observed"
+                and all(delta.get(field) is None for field in PHASE_COUNTERS),
+                "legacy phase absence remains unobserved")
+    require(all(delta[field] == 0 for field in ["failed_gpu_quanta", "cancelled_requests",
+            "native_admission_refusals", "discarded_ready_cohorts"]),
+            "failed/refused/discarded work in timing sample")
+    p = next(iter(outputs.values()))["producer_frontiers"][0]
+    model_steps, prefix_tokens = count * (p + 7), count * (p - 1)
+    native_greedy = observed["observed_greedy_cohorts"]
+    greedy_rows = observed["native_greedy_rows"]
+    b1_outputs = observed["observed_B1_packet_outputs"]
+    bands = delta["native_batches_by_rows"]
+    fused, fused_rows = sum(bands.values()), sum(int(w) * n for w, n in bands.items())
+    require(delta["accepted_output_tokens"] == count * 8
+            and delta["completed_model_token_steps"] == model_steps
+            and delta["completed_forward_kernel_dispatches"] > 0
+            and fused <= model_steps // 2 and fused_rows <= model_steps,
+            "actual complete output/model-step/native bounds")
+    require(fused == delta["gpu_fused_batches"] == delta["state_cohort_publications"]
+            and fused_rows == delta["gpu_fused_rows_completed"] == delta["state_cohort_rows"],
+            "all native band/rows/atomic-publication counters close")
+    require(delta["native_sampler_dispatches_completed"] == 4 * native_greedy
+            and delta["completed_forward_kernel_dispatches"] >=
+            delta["native_projection_dispatches_completed"] + delta["native_sampler_dispatches_completed"]
+            and ((delta["native_projection_dispatches_completed"] > 0) if fused
+                 else (delta["native_projection_dispatches_completed"] == 0)),
+            "actual native greedy sampler and projection-dot subsets close")
+    left = snapshots["before"]["admission"]["state_cohort_publications"]
+    right = snapshots["after"]["admission"]["state_cohort_publications"]
+    require(type(left) is int and type(right) is int and 0 <= left <= right < 2**64
+            and right - left == fused, "source-pinned publication counter interval")
+    require(all(left < sequence <= right for sequence in observed["cohort_members"]),
+            "each native output sequence belongs to this owned sample counter domain")
+    if label == "native_prefix_zerogravity":
+        prefix_bands = {w: bands[w] - observed["observed_greedy_bands"][w] for w in bands}
+        require(all(n >= 0 for n in prefix_bands.values())
+                and delta["gpu_fused_greedy_batches"] == native_greedy
+                and delta["gpu_fused_state_only_batches"] == sum(prefix_bands.values())
+                and delta["gpu_fused_state_only_rows"] == sum(int(w)*n for w,n in prefix_bands.items())
+                and delta["gpu_fused_state_only_batches"] + native_greedy == fused
+                and delta["gpu_fused_state_only_rows"] + greedy_rows == fused_rows
+                and delta["gpu_fused_state_only_rows"] <= prefix_tokens,
+                "actual phase/band conservation; no fabricated StateOnly membership")
+        b1_prefix_tokens = prefix_tokens - delta["gpu_fused_state_only_rows"]
+        b1_model_steps = b1_prefix_tokens + b1_outputs
+        expected_calls = fused + b1_model_steps
+        require(delta["committed_gpu_roots"] == model_steps
+                and delta["gpu_forward_calls_completed"] == expected_calls
+                and b1_model_steps == model_steps - fused_rows,
+                "new one-teacher native/B1 model steps, API calls and independent roots close")
+        observed["derived_StateOnly_bands"] = prefix_bands
+        observed["StateOnly_membership"] = "unobserved; aggregate difference only"
+        prefix_calls = b1_prefix_tokens
+    else:
+        require(bands == observed["observed_greedy_bands"] and fused == native_greedy
+                and fused_rows == greedy_rows,
+                "legacy native cohorts are exactly all observed greedy packets")
+        prefix_calls = count * ((p - 1 + 31) // 32)
+        b1_prefix_tokens = prefix_tokens
+        b1_model_steps = prefix_tokens + b1_outputs
+        expected_calls = native_greedy + b1_outputs + prefix_calls
+        require(delta["gpu_forward_calls_completed"] == expected_calls
+                and delta["committed_gpu_roots"] == count * 8 + prefix_calls,
+                "legacy max32 prefix API chunks, actual native/B1 greedy calls and roots close")
+        observed["derived_StateOnly_bands"] = None
+        observed["StateOnly_membership"] = "legacy phase counters absent; source declares B1 prefix"
+    if count == 1:
+        require(fused == fused_rows == native_greedy == greedy_rows == 0 and b1_outputs == 8,
+                "B1 remains one actual request with no native cohort relabeling")
+        if label == "native_prefix_zerogravity":
+            require(all(delta[field] == 0 for field in PHASE_COUNTERS), "observed B1 native phase counters")
+    observed["publication_counter_interval"] = dict(before=left, after=right,
+        unobserved_native_publications=fused-native_greedy,
+        scope="same source-pinned WaveStats AtomicU64; warmup excluded; no StateOnly membership or numeric GPU ServiceId inferred")
+    observed["actual_scheduling"] = dict(offered_requests=count, client_concurrency=count,
+        configured_native_rows_max=8, actual_teacher_frontier=p,
+        declared_prefix_policy="one actual teacher per scheduled pass" if label == "native_prefix_zerogravity"
+            else "legacy max32 teacher chunks with final tail separate",
+        observed_forward_API_calls=delta["gpu_forward_calls_completed"],
+        observed_committed_roots=delta["committed_gpu_roots"],
+        actual_B1_packet_outputs=b1_outputs, derived_B1_prefix_tokens=b1_prefix_tokens,
+        derived_B1_prefix_API_calls=prefix_calls, derived_B1_model_steps=b1_model_steps,
+        actual_native_rows=fused_rows, completed_model_token_steps=model_steps,
+        physical_forward_kernel_dispatches=delta["completed_forward_kernel_dispatches"])
+    return outputs, observed
+
+
+
+
+def build_receipt(path, expected_sha, binary, current=False):
+    require(sha(path) == expected_sha, "exact reviewed build receipt SHA")
+    receipt = json_read(path)
+    artifact = receipt["artifacts"]["server"]
+    frozen = Path(artifact["frozen_path"])
+    frozen = frozen if frozen.is_absolute() else ROOT / frozen
+    require(frozen.resolve() == binary and artifact["sha256"] == sha(binary)
+            and artifact["bytes"] == binary.stat().st_size, "actual frozen build artifact binding")
+    require(re.fullmatch(r"[a-f0-9]{40}", receipt["source_head"]) is not None
+            and isinstance(receipt.get("source_hashes"), dict) and receipt["source_hashes"]
+            and all(isinstance(name,str) and not Path(name).is_absolute() and ".." not in Path(name).parts
+                and re.fullmatch(r"[a-f0-9]{64}", value) is not None
+                for name,value in receipt["source_hashes"].items()), "actual source head/overlay hash attestation")
+    if current:
+        require(set(CURRENT_SOURCE_FIELDS) <= set(receipt["source_hashes"]), "current prefix owner source hashes required")
+    else:
+        require(receipt["source_head"] == "c5eb1f8e717c79146937e585a145a0052eff747b"
+                and artifact["sha256"] == ACCEPTED_SHA, "accepted cold-preparation compiled source/overlay receipt")
+    return {"receipt_sha256": expected_sha, "receipt": receipt,
+            "scope": "reviewed source/build attestation; not an independent numerical or reproducible-build proof"}
+
+
+def counter_source_contract(receipt, label):
+    require(label in LABELS, "source contract requires one declared same-engine label")
+    contract = COUNTER_SOURCE_CONTRACT
+    accepted = label == "accepted_d0prep_zerogravity"
+    expected = contract["accepted_files" if accepted else "current_files"]
+    if accepted:
+        require(receipt["source_head"] == contract["accepted_head"], "accepted counter-domain HEAD")
+    actual = {}
+    for path, digest in expected.items():
+        fallback = contract["accepted_head_files"][path] if accepted else None
+        actual[path] = receipt["source_hashes"].get(path, fallback)
+        require(actual[path] == digest, "source-pinned worker/health counter domain changed: " + path)
+    return dict(source_sha256=actual,
+        domain="one engine-owned WaveStats Arc passed to worker and read by health; sequence increment only after cohort.commit Ok",
+        reservations_and_failures="no sequence reserved before commit; abandoned/failed/Ready-discarded attempts do not increment; samples reject those counters",
+        B1="single_completion never increments this cohort counter", warmup="excluded by quiescent before/after interval",
+        numeric_GPU_ServiceId_or_SourceIdentity="not exposed and not inferred")
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--run", action="store_true", help="root coordinator explicitly launches actual GPU providers")
+    parser.add_argument("--current-server", type=Path, required=True)
+    parser.add_argument("--current-sha256", required=True)
+    parser.add_argument("--current-source-id", required=True, help="root actual source/build receipt identity, never inferred from binary name")
+    parser.add_argument("--current-receipt", type=Path, required=True)
+    parser.add_argument("--current-receipt-sha256", required=True)
+    parser.add_argument("--workloads", default="c8,b1", help="natural eight offers/concurrency8 c8 and/or single actual request b1")
+    parser.add_argument("--gpu-uuid", required=True)
+    parser.add_argument("--accepted-server", type=Path, default=ROOT_RUN / "activation-cold-final-server")
+    parser.add_argument("--accepted-receipt", type=Path, default=ROOT_RUN / "activation-cold-final-serving-artifacts.json")
+    parser.add_argument("--models", default="0.8b,2b")
+    parser.add_argument("--order", choices=["ABBA", "BAAB"], default="ABBA")
+    parser.add_argument("--policy", choices=["cold_process", "warmed_process"], default="warmed_process")
+    parser.add_argument("--max-tokens", type=int, choices=[8], default=8)
+    parser.add_argument("--timeout", type=float, default=90)
+    parser.add_argument("--ttft-slo", type=float, default=10)
+    parser.add_argument("--itl-slo", type=float, default=2)
+    parser.add_argument("--ports", default="18020,18021")
+    parser.add_argument("--cpus", help="exactly four currently allowed logical CPUs; default first four allowed")
+    parser.add_argument("--output-parent", type=Path, default=OWN / "actual-natural-c8-measurements")
+    args = parser.parse_args()
+    require(args.run, "PREPARED ONLY: --run is required; this agent has not launched any provider")
+    require(sha(OWN / "prepared-manifest.json") == ORIGINAL_PREPARED_SHA,
+            "original strict B8 prepared freeze changed; stop, never reclassify its failure")
+    require(sha(OWN / "measure_same_engine.py") == "755c0c526422b510f26ea0d5360d8cc74daae44932c75056bc02c993d623a667",
+            "original strict B8 driver changed")
+    # This private process owns urllib's opener before benchmark threads start.
+    urllib.request.install_opener(local_opener())
+    require(re.fullmatch(r"[a-f0-9]{64}", args.current_sha256) is not None, "actual current binary SHA256 required")
+    require(re.fullmatch(r"[A-Za-z0-9_.:/+-]{1,256}", args.current_source_id) is not None, "bounded actual source receipt ID")
+    require(re.fullmatch(r"GPU-[a-fA-F0-9-]{16,64}", args.gpu_uuid) is not None, "exact observed GPU UUID required")
+    require(math.isfinite(args.timeout) and 1 <= args.timeout <= 180 and all(math.isfinite(value)
+            and .001 <= value <= ceiling for value,ceiling in [(args.ttft_slo,10), (args.itl_slo,2)]), "finite predeclared SLO limits; no widening")
+    models = args.models.split(",")
+    require(1 <= len(models) <= 2 and len(set(models)) == len(models) and all(name in MODELS for name in models), "approved source models only")
+    workloads = args.workloads.split(",")
+    require(1 <= len(workloads) <= 2 and len(set(workloads)) == len(workloads)
+            and all(name in {"c8", "b1"} for name in workloads), "explicit bounded workload cases")
+    require(re.fullmatch(r"[a-f0-9]{64}", args.current_receipt_sha256) is not None, "pinned current build receipt SHA")
+    ports = [int(value) for value in args.ports.split(",")]
+    require(len(ports) == 2 and len(set(ports)) == 2 and all(1024 <= value <= 65535 for value in ports), "two bounded loopback ports")
+    allowed = os.sched_getaffinity(0)
+    cpus = sorted(allowed)[:4] if args.cpus is None else [int(value) for value in args.cpus.split(",")]
+    require(len(cpus) == len(set(cpus)) == 4 and set(cpus) <= allowed, "exactly four allowed logical CPU affinity members")
+    bench, lifecycle = module("serve-bench.py"), module("serve-lifecycle.py")
+    accepted = args.accepted_server.resolve(strict=True)
+    current = args.current_server.resolve(strict=True)
+    accepted_build = build_receipt(args.accepted_receipt, ACCEPTED_RECEIPT_SHA, accepted)
+    current_build = build_receipt(args.current_receipt, args.current_receipt_sha256, current, True)
+    counter_contracts = {
+        "accepted_d0prep_zerogravity": counter_source_contract(accepted_build["receipt"], "accepted_d0prep_zerogravity"),
+        "native_prefix_zerogravity": counter_source_contract(current_build["receipt"], "native_prefix_zerogravity")}
+    require(accepted != current and sha(accepted) == ACCEPTED_SHA and sha(current) == args.current_sha256,
+            "actual separately frozen accepted/current server hashes")
+    root_models = {}
+    for name in models:
+        parent = ROOT / "target/models" / ("qwen35-" + name)
+        require(sha(parent / "model.gguf") == MODELS[name]["source"]
+                and sha(parent / "config.json") == MODELS[name]["config"]
+                and sha(parent / "tokenizer.json") == TOKENIZER_SHA, "actual original source/config/tokenizer bytes changed")
+        root_models[name] = parent
+    env = {"CUDA_PATH": str(ROOT / ".agents/runs/p12-full-fusion-root-2026-10-19/cuda-view"),
+           "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
+           "LD_LIBRARY_PATH": "/nix/store/cs50fahgnln1k5cx7zs0dnyikzqzf4iq-cuda12.9-cuda_nvrtc-12.9.86-lib/lib:/run/opengl-driver/lib",
+           "RAYON_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
+    require(Path(env["CUDA_PATH"]).is_dir(), "approved existing CUDA view missing")
+    args.output_parent = args.output_parent.resolve()
+    require(OWN in args.output_parent.parents, "all new run artifacts must stay in this owned ignored directory")
+    args.output_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
+    directory = Path(tempfile.mkdtemp(prefix="same-zg-c8-", dir=args.output_parent))
+    report = dict(schema="zerogravity.same-engine.natural-concurrency8.before-after.v1", status="RUNNING",
+        scope="accepted cold-preparation Zerogravity versus candidate Zerogravity only; never a vLLM comparison or source/model qualification",
+        providers={"A": "accepted_d0prep_zerogravity", "B": "native_prefix_zerogravity"},
+        binary_sha256={"accepted_d0prep_zerogravity": ACCEPTED_SHA, "native_prefix_zerogravity": args.current_sha256},
+        source_ids={"accepted_d0prep_zerogravity": {"compiled_head": accepted_build["receipt"]["source_head"],
+            "reviewed_build_receipt_sha256": ACCEPTED_RECEIPT_SHA, "landed_reference": "d0a6084",
+            "overlay_sha256": accepted_build["receipt"]["source_hashes"]},
+            "native_prefix_zerogravity": {"caller_receipt_id": args.current_source_id,
+                "compiled_head": current_build["receipt"]["source_head"],
+                "reviewed_build_receipt_sha256": args.current_receipt_sha256,
+                "overlay_sha256": current_build["receipt"]["source_hashes"]}},
+        actual_build_receipts={"accepted_d0prep_zerogravity": accepted_build, "native_prefix_zerogravity": current_build},
+        order=args.order, process_policy=args.policy, workloads=workloads, cpus=cpus, samples=[],
+        workload_semantics={"c8":"eight offers, client concurrency8, configured max8; actual native2/4/8 and B1 packets retained without fixed-band promise",
+                            "b1":"one actual request, original B1 math; old max32/new one-teacher scheduling"},
+        source_pinned_counter_contracts=counter_contracts,
+        prior_strict_B8_failure="preserved UNQUALIFIED; not imported, retried or converted by this new workload",
+        tools=TOOL_PINS, measurement_driver_sha256=sha(Path(__file__)),
+        source_pins={name: MODELS[name] for name in models}, tokenizer_sha256=TOKENIZER_SHA,
+        temperature=0, top_p=1, n=1, max_tokens=args.max_tokens, context_capacity=512, active_rows_max=8,
+        logical_caps_mib={"host": 4096, "provider": 4096, "device": 5120},
+        OS_storage_device_caches_evicted=False, physical_RAM_admission_qualified=False,
+        kernel_counter_scope="actual completed source-forward kernel receipts, including native prefix+greedy bodies; initialization/DMA/unknown-completion excluded; projection counter is projection-dot subset; sampler counter is 4-kernel B>1 greedy subset, not all B1 samplers",
+        counter_visibility={"accepted_d0prep_zerogravity": "legacy phase fields absent: unobserved, never zero",
+                            "native_prefix_zerogravity": "all three phase fields required observed"},
+        B1_policy="one actual request, same prompt/caps/fused-enabled configuration: old max32 teacher API chunks versus new one-teacher scheduled passes; original B1 GPU math",
+        predeclared_SLO_limits={"TTFT_seconds":args.ttft_slo,"actual_single_ID_delivery_ITL_seconds":args.itl_slo},
+        memory_scope="nvidia-smi and proc observations, not ResourceArbiter reservation telemetry; provider health does not currently expose GPU budget bytes",
+        http_transport="private loopback; proxies disabled and redirects refused for benchmark and health requests",
+        started_utc=datetime.now(timezone.utc).isoformat())
+    write = lambda: lifecycle.write_json(directory / "report.json", report)
+    write()
+    os.sched_setaffinity(0, cpus)
+    try:
+        idle = gpu_idle(args.gpu_uuid)
+        report["initial_GPU_idle_observations"] = idle
+        idle_ceiling = idle[-1]["memory_used_mib"] + 64
+        canonical_by_model = {}
+        for name in models:
+            for workload_name in workloads:
+                count = 8 if workload_name == "c8" else 1
+                alias = "qwen35-" + name + "-same-source"
+                endpoints = dict(zip(["accepted_d0prep_zerogravity", "native_prefix_zerogravity"],
+                                    [f"http://127.0.0.1:{port}" for port in ports]))
+                providers = {}
+                for label, binary in [("accepted_d0prep_zerogravity", accepted), ("native_prefix_zerogravity", current)]:
+                    command = [str(binary), "serve", "--model", alias + "=" + str(root_models[name] / "model.gguf"),
+                        "--backend", "cuda", "--workers", "1", "--active-per-worker", "8", "--max-context", "512",
+                        "--max-output", "8", "--host-mib", "4096", "--provider-mib", "4096", "--device-mib", "5120",
+                        "--batch-mode", "fused", "--batch-rows", "8", "--enable-per-request-metrics", "--listen",
+                        "127.0.0.1:" + str(ports[0 if label == "accepted_d0prep_zerogravity" else 1])]
+                    providers[label] = dict(argv=command, cwd=str(ROOT), env=env,
+                        declarations=dict(runtime="Zerogravity " + label + " binarySHA " + report["binary_sha256"][label],
+                            source_sha256=MODELS[name]["source"], tokenizer_sha256=TOKENIZER_SHA,
+                            numerical_profile="accepted original GGUF source kernels; measured token/text/usage checked, independent mathematical acceptance remains separate",
+                            quantization="unchanged original mixed GGUF source bytes", generation_policy="greedy temperature0 top_p1 n1; same source EOS; finite max8",
+                            hardware="caller-pinned NVIDIA UUID " + args.gpu_uuid,
+                            resource_limits=dict(context=512, configured_rows=8, offered_requests=count,
+                                host_mib=4096, provider_mib=4096, device_mib=5120,
+                                logical_CPU_affinity=cpus, Rayon_threads=4, OS_CPU_or_physical_RAM_quota_claimed=False)))
+                plan = plan_type(lifecycle)(providers, endpoints, args.policy, directory / ("owned-" + name + "-" + workload_name))
+                reference = None
+                for sequence, letter in enumerate(args.order):
+                    label = report["providers"][letter]
+                    url, binary = endpoints[label], accepted if letter == "A" else current
+                    require(sha(binary) == report["binary_sha256"][label], "frozen provider changed before spawn")
+                    require(sha(root_models[name] / "model.gguf") == MODELS[name]["source"]
+                            and sha(root_models[name] / "config.json") == MODELS[name]["config"]
+                            and sha(root_models[name] / "tokenizer.json") == TOKENIZER_SHA,
+                            "source/config/tokenizer changed before owned provider")
+                    before_idle = gpu_idle(args.gpu_uuid, idle_ceiling)
+                    sampler = Sampler(args.gpu_uuid)
+                    sample = None
+                    try:
+                        def ready():
+                            require({item["id"] for item in get_json(url + "/v1/models")["data"]} == {alias}, "exact ready model alias")
+                            health(url, label)
+                        with plan.sample(label, ready) as owned:
+                            sampler.pid = owned.process.pid
+                            require(Path(os.readlink(f"/proc/{owned.process.pid}/exe")).resolve() == binary,
+                                    "owned foreground executable differs from frozen binary")
+                            if args.policy == "warmed_process":
+                                warm = bench.bench(url, alias, PROMPT, count, count, 1, args.timeout, None,
+                                    workload=workload(1, count), snapshot_url=url + "/health", return_token_ids=True,
+                                    max_pending=count, run_id=f"prefix-zg-warm-{name}-{workload_name}-{sequence}")
+                                owned.record_warmup(warm)
+                                settled_result(warm, count, 1)
+                            before = quiescent(url, label)
+                            measured = bench.bench(url, alias, PROMPT, count, count, 8, args.timeout, None,
+                                workload=workload(8, count), ttft_slo_seconds=args.ttft_slo,
+                                itl_slo_seconds=args.itl_slo, snapshot_url=url + "/health", return_token_ids=True,
+                                max_pending=count, run_id=f"prefix-zg-{name}-{workload_name}-{sequence}")
+                            after = quiescent(url, label)
+                            measured["same_engine_quiescent_health"] = dict(before=before, after=after,
+                                delta=deltas(before, after), gauges=gauge_observations(before, after))
+                            owned.record_measurement(measured)
+                            sample = dict(model=name, workload=workload_name, offered_count=count,
+                                sequence=sequence, provider=label, measurement=measured,
+                                lifecycle_receipt_path=str(owned.receipt_path), GPU_idle_before_spawn=before_idle)
+                    finally:
+                        memory = sampler.finish()
+                    # Original owner closes and proves cleanup before any receipt validation/next provider.
+                    require(sample is not None and not plan.blocked and plan.current is None, "owned lifecycle did not finish")
+                    sample["memory_observations"] = memory
+                    sample["lifecycle_receipt"] = json_read(sample["lifecycle_receipt_path"])
+                    owned_pid = sample["lifecycle_receipt"]["pid"]
+                    owned_gpu_observed = False
+                    for observed_sample in memory["samples"]:
+                        for application in observed_sample["compute_apps_raw"].splitlines():
+                            process = application.split(",", 1)[0].strip()
+                            require(process.isdecimal() and int(process) == owned_pid,
+                                    "another GPU compute process appeared; preserve it and stop")
+                            owned_gpu_observed = True
+                    require(owned_gpu_observed, "selected GPU owned compute process was not physically observed")
+                    sample["GPU_idle_after_owned_cleanup"] = gpu_idle(args.gpu_uuid, idle_ceiling)
+                    require(sha(binary) == report["binary_sha256"][label], "frozen provider changed during sample")
+                    require(sha(root_models[name] / "model.gguf") == MODELS[name]["source"]
+                            and sha(root_models[name] / "config.json") == MODELS[name]["config"]
+                            and sha(root_models[name] / "tokenizer.json") == TOKENIZER_SHA,
+                            "source/config/tokenizer changed during owned provider")
+                    require(sha(args.accepted_receipt) == ACCEPTED_RECEIPT_SHA
+                            and sha(args.current_receipt) == args.current_receipt_sha256,
+                            "reviewed build receipt changed during comparison")
+                    report["samples"].append(sample)
+                    write()
+                    outputs, closed = counters(sample["measurement"], label, count)
+                    closed["sample_namespace"] = dict(provider=label, model=name, model_alias=alias,
+                        measurement_run_id=sample["measurement"]["run_id"], owned_foreground_pid=owned_pid,
+                        binary_sha256=report["binary_sha256"][label], source_sha256=MODELS[name]["source"],
+                        tokenizer_sha256=TOKENIZER_SHA, gpu_uuid=args.gpu_uuid,
+                        source_pinned_counter_contract=counter_contracts[label],
+                        scope="singleton declared model in one owned source-bound engine/process; numeric GPU ServiceId and SourceIdentity unavailable, never invented")
+                    sample["actual_output_and_counter_closure"] = closed
+                    canonical = next(iter(outputs.values()))
+                    if name not in canonical_by_model:
+                        canonical_by_model[name] = canonical
+                    require(canonical == canonical_by_model[name],
+                            "same model/prompt B1 and c8 exact IDs/text/frontiers/usage/finish differ")
+                    if reference is None:
+                        reference = outputs
+                    require(outputs == reference, "actual offered token IDs/frontiers/text/usage/finish changed; no next sample")
+                    sample["same_source_outputs_equal"] = True
+                    write()
+        report["comparison_observations"] = []
+        for model_name in models:
+            for workload_name in workloads:
+                selected = [sample for sample in report["samples"]
+                            if sample["model"] == model_name and sample["workload"] == workload_name]
+                require([sample["provider"] for sample in selected] == [report["providers"][letter] for letter in args.order],
+                        "actual serialized natural-c8/B1 sample order equals predeclared ABBA/BAAB")
+                report["comparison_observations"].append({"model":model_name,"workload":workload_name,
+                    "actual_order":[sample["provider"] for sample in selected],
+                    "samples":[{"provider":sample["provider"],"sequence":sample["sequence"],
+                        "wall_seconds":sample["measurement"]["wall_seconds"],
+                        "text_TTFT_p50_seconds":sample["measurement"]["ttft_p50_seconds"],
+                        "text_TTFT_p95_seconds":sample["measurement"]["ttft_p95_seconds"],
+                        "actual_ID_delivery_and_provider_commit_clocks":sample["measurement"]["wire_token_metrics"],
+                        "all_offered_denominators_and_workload_label_fairness":sample["measurement"]["fairness_by_workload_tenant"],
+                        "physical_forward_and_scheduling_receipts":sample["measurement"]["same_engine_quiescent_health"],
+                        "memory_sampled_peak_device_mib":sample["memory_observations"]["peak_gpu_device_mib"],
+                        "memory_owned_process_RSS_peak_kib":sample["memory_observations"]["peak_owned_process_rss_kib"]}
+                        for sample in selected],
+                    "scope":"bounded observations only; B1 records explicit one-teacher versus max32 scheduling cost alongside actual kernels; no automatic causal speed/SLO/vLLM eligibility"})
+        report["status"] = "PASS_BOUNDED_NATURAL_CONCURRENCY_SAME_ZEROGRAVITY_OBSERVATIONS_ONLY"
+        report["actual_outputs_identical_across_samples_per_model_and_workload"] = True
+        report["same_prompt_B1_and_c8_outputs_identical_per_model"] = len(workloads) == 2
+        report["source_correct_SLO_goodput_qualified"] = False
+        report["speed_or_fullmodel_eligibility_inferred"] = False
+    except BaseException as error:
+        report["status"] = "UNQUALIFIED"
+        report["failure_type"], report["failure_message"] = type(error).__name__, str(error)[:1024]
+        raise
+    finally:
+        os.sched_setaffinity(0, allowed)
+        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
+        # All owned contexts close before this final evidence I/O. A lost
+        # receipt still surfaces as failure, and cannot start another provider.
+        write()
+        print(json.dumps({"status": report["status"], "report": str(directory / "report.json")}), flush=True)
+
+
+if __name__ == "__main__":
+    main()
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/test_natural_c8.py
@@ -0,0 +1,242 @@
+#!/usr/bin/env python3
+"""ROOT ONLY source-free accounting fixtures; author AST parsing only."""
+import ast
+import copy
+import hashlib
+import importlib.util
+from pathlib import Path
+import unittest
+
+OWN = Path(__file__).resolve().parent
+SPEC = importlib.util.spec_from_file_location("natural_c8_driver", OWN / "measure_natural_c8.py")
+DRIVER = importlib.util.module_from_spec(SPEC)
+SPEC.loader.exec_module(DRIVER)
+A, B = "accepted_d0prep_zerogravity", "native_prefix_zerogravity"
+
+# Only membership/shape from preserved failed strict-B8 report5da6dc90;
+# identities/IDs/text/clock payload below are synthetic accounting fixtures.
+CAPTURED_MIXED_LAYOUT = [
+    (95, 2, [(0, 0), (2, 0)]),
+    (96, 4, [(3, 0), (4, 0), (6, 0), (7, 0)]),
+    (97, 2, [(0, 1), (2, 1)]),
+    (98, 8, [(0, 2), (1, 2), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]),
+    (99, 8, [(0, 3), (1, 3), (2, 3), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2)]),
+    (100, 8, [(0, 4), (1, 4), (2, 4), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3)]),
+    (101, 8, [(0, 5), (1, 5), (2, 5), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4)]),
+    (102, 8, [(0, 6), (1, 6), (2, 6), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5)]),
+    (103, 8, [(0, 7), (1, 7), (2, 7), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6)]),
+    (104, 4, [(3, 7), (4, 7), (6, 7), (7, 7)]),
+]
+
+
+def fixture(label, count=8, native=True):
+    p, requests = 47, []
+    for request in range(count):
+        records = [{"gpu": {"producer_frontier": p + index, "native_cohort": None}}
+                   for index in range(8)]
+        requests.append(dict(request_id=f"request-{request}", max_tokens=8,
+            usage_reported_completion_tokens=8, wire_request_id=f"fixture:request-{request}",
+            provider_response_id=f"chatcmpl-fixture-{request}", completed=True,
+            classification="completed", http_status=200, http_accepted=True,
+            http_attempted=True, error=None,
+            prompt_sha256=hashlib.sha256(DRIVER.PROMPT.encode()).hexdigest(),
+            output_sha256="2"*64, finish_reason="length",
+            wire_token_observation=dict(status="provider_bound_complete_ids", token_ids=list(range(8)),
+                client_delivery_status="individual_token_ids_observed",
+                provider_commit_observation_status="provider_bound_complete_commit_observations",
+                provider_token_records=records)))
+    if count == 8 and native:
+        for sequence, width, members in CAPTURED_MIXED_LAYOUT:
+            for request, index in members:
+                requests[request]["wire_token_observation"]["provider_token_records"][index]["gpu"]["native_cohort"] = {
+                    "width": width, "publication_sequence": sequence if label == B else sequence-94}
+    mixed = count == 8 and native
+    greedy, greedy_rows, singles = (10, 60, 4) if mixed else (0, 0, count*8)
+    delta = {field: 0 for field in DRIVER.COUNTERS}
+    delta.update(completed_forward_kernel_dispatches=10000,
+        completed_model_token_steps=count*(p+7), accepted_output_tokens=count*8,
+        phase_counter_observability="observed_native_prefix_phase_counters" if label == B
+            else "legacy_absent_not_observed",
+        native_sampler_dispatches_completed=4*greedy)
+    for field in DRIVER.PHASE_COUNTERS:
+        delta[field] = 0 if label == B else None
+    if label == B:
+        state_batches, state_rows = (47, 366) if mixed else (0, 0)
+        bands = {"2": 3, "4": 3, "8": 51} if mixed else {"2": 0, "4": 0, "8": 0}
+        delta.update(gpu_fused_state_only_batches=state_batches,
+            gpu_fused_state_only_rows=state_rows, gpu_fused_greedy_batches=greedy,
+            committed_gpu_roots=count*(p+7),
+            gpu_forward_calls_completed=state_batches+greedy+count*(p-1)-state_rows+singles)
+    else:
+        state_batches, state_rows = 0, 0
+        bands = {"2": 2, "4": 2, "8": 6} if mixed else {"2": 0, "4": 0, "8": 0}
+        prefix_calls = count*((p-1+31)//32)
+        delta.update(committed_gpu_roots=count*8+prefix_calls,
+                     gpu_forward_calls_completed=greedy+singles+prefix_calls)
+    fused, fused_rows = state_batches+greedy, state_rows+greedy_rows
+    delta.update(gpu_fused_batches=fused, gpu_fused_rows_completed=fused_rows,
+        state_cohort_publications=fused, state_cohort_rows=fused_rows,
+        native_batches_by_rows=bands, native_projection_dispatches_completed=100 if fused else 0)
+    before = 47 if mixed and label == B else 0
+    return dict(run_id="fixture", offered_requests=count, completed_requests=count,
+        http_accepted_requests=count, usage_reported_completion_tokens=8*count,
+        unsuccessful_requests=0, completions_without_usage=0, http_429_requests=0,
+        http_503_requests=0, rejected_requests=0, cancelled_requests=0, timed_out_requests=0,
+        client_overload_requests=0, launch_deadline_requests=0,
+        classifications={"completed": count}, errors={}, requests=requests,
+        wire_token_metrics=dict(requested=True,
+            completed_requests_with_individual_token_delivery=count,
+            completed_requests_with_provider_commit_observations=count),
+        same_engine_quiescent_health=dict(delta=delta,
+            before={"admission": {"state_cohort_publications": before}},
+            after={"admission": {"state_cohort_publications": before+fused}}))
+
+
+def packet(value, request=0, index=0):
+    return value["requests"][request]["wire_token_observation"]["provider_token_records"][index]["gpu"]
+
+
+class NaturalAccounting(unittest.TestCase):
+    def test_actual_mixed_layout_closes_all_57_batches_426_rows_and_four_B1_outputs(self):
+        old, new = fixture(A), fixture(B)
+        left, a = DRIVER.counters(old, A, 8)
+        right, b = DRIVER.counters(new, B, 8)
+        self.assertEqual(left, right)
+        self.assertEqual(b["observed_greedy_bands"], {"2": 2, "4": 2, "8": 6})
+        self.assertEqual(b["derived_StateOnly_bands"], {"2": 1, "4": 1, "8": 45})
+        self.assertEqual(b["observed_B1_packet_outputs"], 4)
+        self.assertEqual(b["actual_scheduling"]["derived_B1_prefix_tokens"], 2)
+        self.assertEqual(b["actual_scheduling"]["derived_B1_model_steps"], 6)
+        self.assertEqual(b["actual_scheduling"]["observed_forward_API_calls"], 63)
+        self.assertEqual(b["actual_scheduling"]["observed_committed_roots"], 432)
+        self.assertIsNone(a["derived_StateOnly_bands"])
+
+    def test_independent_frontiers_keep_different_output_indices_in_one_cohort(self):
+        _, observed = DRIVER.counters(fixture(B), B, 8)
+        members = observed["cohort_members"][98]["members"]
+        self.assertEqual({m["output_index"] for m in members.values()}, {1, 2})
+        self.assertEqual({m["producer_frontier"] for m in members.values()}, {48, 49})
+        self.assertEqual({(m["request_id"], m["output_index"]) for m in observed["B1_packets"]},
+                         {("request-1", 0), ("request-1", 1), ("request-5", 0), ("request-5", 7)})
+
+    def test_original_strict_B8_validator_still_refuses_this_layout(self):
+        spec = importlib.util.spec_from_file_location("strict_original", OWN / "measure_same_engine.py")
+        original = importlib.util.module_from_spec(spec)
+        spec.loader.exec_module(original)
+        with self.assertRaisesRegex(ValueError, "B8"):
+            original.counters(fixture(B), B, 8)
+
+    def test_missing_whole_or_partial_native_membership_refuses(self):
+        for whole_group in [False, True]:
+            bad = fixture(B)
+            packet(bad, 0, 0)["native_cohort"] = None
+            if whole_group:
+                packet(bad, 2, 0)["native_cohort"] = None
+            with self.subTest(whole_group=whole_group), self.assertRaises(ValueError):
+                DRIVER.counters(bad, B, 8)
+        bad = fixture(B)
+        del packet(bad, 1, 0)["native_cohort"]
+        with self.assertRaises(ValueError):
+            DRIVER.counters(bad, B, 8)
+
+    def test_duplicate_inconsistent_or_noninteger_cohort_refuses(self):
+        for field, value in [("width", 4), ("width", True), ("publication_sequence", True),
+                             ("publication_sequence", 2**64), ("publication_sequence", 47)]:
+            bad = fixture(B)
+            packet(bad)["native_cohort"][field] = value
+            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
+                DRIVER.counters(bad, B, 8)
+        bad = fixture(B)
+        packet(bad, 0, 1)["native_cohort"] = copy.deepcopy(packet(bad, 0, 0)["native_cohort"])
+        with self.assertRaises(ValueError):
+            DRIVER.counters(bad, B, 8)
+
+    def test_counter_domain_interval_cannot_reuse_warmup_or_foreign_sequences(self):
+        for boundary, value in [("before", 95), ("after", 103), ("before", True)]:
+            bad = fixture(B)
+            bad["same_engine_quiescent_health"][boundary]["admission"]["state_cohort_publications"] = value
+            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
+                DRIVER.counters(bad, B, 8)
+
+    def test_phase_rows_bands_sampler_roots_APIs_and_failures_all_close(self):
+        for field, value in [("gpu_fused_greedy_batches", 9), ("gpu_fused_state_only_batches", 48),
+                ("gpu_fused_state_only_rows", 365), ("state_cohort_rows", 425),
+                ("gpu_forward_calls_completed", 62), ("committed_gpu_roots", 431),
+                ("completed_model_token_steps", 431), ("native_sampler_dispatches_completed", 228),
+                ("native_admission_refusals", 1), ("discarded_ready_cohorts", 1),
+                ("failed_gpu_quanta", 1), ("cancelled_requests", 1)]:
+            bad = fixture(B)
+            bad["same_engine_quiescent_health"]["delta"][field] = value
+            with self.subTest(field=field), self.assertRaises(ValueError):
+                DRIVER.counters(bad, B, 8)
+        bad = fixture(B)
+        bad["same_engine_quiescent_health"]["delta"]["native_batches_by_rows"]["4"] = 2
+        with self.assertRaises(ValueError):
+            DRIVER.counters(bad, B, 8)
+
+    def test_all_B1_is_reported_honestly_without_inventing_native_work(self):
+        for label in [A, B]:
+            _, observed = DRIVER.counters(fixture(label, native=False), label, 8)
+            self.assertEqual(observed["observed_B1_packet_outputs"], 64)
+            self.assertEqual(observed["observed_greedy_cohorts"], 0)
+
+    def test_single_request_retains_oldmax32_vs_newone_teacher_policy(self):
+        left, a = DRIVER.counters(fixture(A, 1), A, 1)
+        right, b = DRIVER.counters(fixture(B, 1), B, 1)
+        self.assertEqual(left, right)
+        self.assertEqual(a["actual_scheduling"]["observed_forward_API_calls"], 10)
+        self.assertEqual(b["actual_scheduling"]["observed_forward_API_calls"], 54)
+        bad = fixture(B, 1)
+        packet(bad)["native_cohort"] = {"width": 2, "publication_sequence": 1}
+        with self.assertRaises(ValueError):
+            DRIVER.counters(bad, B, 1)
+
+    def test_legacy_missing_phase_fields_are_never_observed_zero(self):
+        for field in DRIVER.PHASE_COUNTERS:
+            bad = fixture(A)
+            bad["same_engine_quiescent_health"]["delta"][field] = 0
+            with self.subTest(field=field), self.assertRaises(ValueError):
+                DRIVER.counters(bad, A, 8)
+
+    def test_full_wire_usage_identity_frontier_and_outcome_guards_retained(self):
+        for field, value in [("wire_request_id", "foreign"), ("provider_response_id", ""),
+                ("output_sha256", "bad"), ("usage_reported_completion_tokens", 7),
+                ("completed", False), ("finish_reason", "stop")]:
+            bad = fixture(B)
+            bad["requests"][0][field] = value
+            with self.subTest(field=field), self.assertRaises(ValueError):
+                DRIVER.counters(bad, B, 8)
+        bad = fixture(B)
+        packet(bad, 1, 0)["producer_frontier"] = 48
+        with self.assertRaises(ValueError):
+            DRIVER.counters(bad, B, 8)
+        bad = fixture(B)
+        bad["offered_requests"] = 9
+        with self.assertRaises(ValueError):
+            DRIVER.counters(bad, B, 8)
+
+    def test_source_contract_rejects_unknown_counter_domains(self):
+        c = DRIVER.COUNTER_SOURCE_CONTRACT
+        accepted = {"source_head": c["accepted_head"], "source_hashes": {
+            "crates/zerogravity-cli/src/serve/mod.rs": c["accepted_files"]["crates/zerogravity-cli/src/serve/mod.rs"]}}
+        DRIVER.counter_source_contract(accepted, A)
+        current = {"source_hashes": dict(c["current_files"])}
+        DRIVER.counter_source_contract(current, B)
+        for label, value in [(A, accepted), (B, current)]:
+            bad = copy.deepcopy(value)
+            bad["source_hashes"]["crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs"] = "0"*64
+            with self.subTest(label=label), self.assertRaises(ValueError):
+                DRIVER.counter_source_contract(bad, label)
+
+    def test_underlying_lifecycle_wire_source_memory_and_SLO_helpers_are_exact(self):
+        def functions(path):
+            text = path.read_text()
+            return {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body
+                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
+        old, new = functions(OWN / "measure_same_engine.py"), functions(OWN / "measure_natural_c8.py")
+        for name in old.keys() - {"main", "output_cohorts", "counters", "health"}:
+            self.assertEqual(old[name], new[name], name)
+
+
+if __name__ == "__main__":
+    unittest.main()
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural_c8_functions.pyfrag
@@ -0,0 +1,147 @@
+def output_cohorts(result, count):
+    require(count in [1, 8], "closed offered concurrency scope")
+    cohorts, singles, prior_sequence = {}, [], {}
+    for row in result["requests"]:
+        identity = row["request_id"]
+        observed = row["wire_token_observation"]
+        for index, token in enumerate(observed["provider_token_records"]):
+            gpu_packet = token["gpu"]
+            require("native_cohort" in gpu_packet,
+                    "explicit actual native-or-B1 packet provenance required")
+            member = dict(request_id=identity, output_index=index,
+                          producer_frontier=gpu_packet["producer_frontier"],
+                          token_id=observed["token_ids"][index],
+                          wire_request_id=row["wire_request_id"],
+                          provider_response_id=row["provider_response_id"])
+            cohort = gpu_packet["native_cohort"]
+            if cohort is None:
+                singles.append(member)
+                continue
+            require(count == 8 and isinstance(cohort, dict)
+                    and set(cohort) == {"width", "publication_sequence"}
+                    and type(cohort["width"]) is int and cohort["width"] in [2, 4, 8],
+                    "actual native2/4/8 cohort metadata; c8 is not a width8 assertion")
+            sequence, width = cohort["publication_sequence"], cohort["width"]
+            require(type(sequence) is int and 0 < sequence < 2**64,
+                    "actual bounded native publication sequence")
+            require(sequence > prior_sequence.get(identity, 0),
+                    "per-request native packet publication order")
+            prior_sequence[identity] = sequence
+            group = cohorts.setdefault(sequence, dict(width=width, members={}))
+            require(group["width"] == width and identity not in group["members"],
+                    "one consistent width and distinct actual session/request per cohort")
+            group["members"][identity] = member
+    bands = {"2": 0, "4": 0, "8": 0}
+    for group in cohorts.values():
+        require(len(group["members"]) == group["width"],
+                "every actual native greedy cohort has complete distinct width membership")
+        bands[str(group["width"])] += 1
+    native_rows = sum(group["width"] for group in cohorts.values())
+    require(native_rows + len(singles) == 8 * count,
+            "every accepted output belongs to one actual native cohort or B1 packet")
+    return dict(observed_greedy_cohorts=len(cohorts), native_greedy_rows=native_rows,
+                observed_greedy_bands=bands, cohort_members=cohorts,
+                observed_B1_packet_outputs=len(singles), B1_packets=singles,
+                scope="actual source-bound post-publication packet observations; independent frontiers may have different output indices inside one native cohort; no independent state/math proof")
+
+
+def counters(result, label, count):
+    require(label in LABELS and count in [1, 8], "closed same-engine source/workload scope")
+    outputs = signature(result, count)
+    observed = output_cohorts(result, count)
+    snapshots = result["same_engine_quiescent_health"]
+    delta = snapshots["delta"]
+    require(all(type(delta.get(field)) is int and delta[field] >= 0 for field in COUNTERS)
+            and set(delta["native_batches_by_rows"]) == {"2", "4", "8"}
+            and all(type(value) is int and value >= 0 for value in delta["native_batches_by_rows"].values()),
+            "actual finite nonnegative completed-work counters")
+    if label == "native_prefix_zerogravity":
+        require(delta["phase_counter_observability"] == "observed_native_prefix_phase_counters"
+                and all(type(delta.get(field)) is int and delta[field] >= 0 for field in PHASE_COUNTERS),
+                "new phase counters observed, never inferred")
+    else:
+        require(delta["phase_counter_observability"] == "legacy_absent_not_observed"
+                and all(delta.get(field) is None for field in PHASE_COUNTERS),
+                "legacy phase absence remains unobserved")
+    require(all(delta[field] == 0 for field in ["failed_gpu_quanta", "cancelled_requests",
+            "native_admission_refusals", "discarded_ready_cohorts"]),
+            "failed/refused/discarded work in timing sample")
+    p = next(iter(outputs.values()))["producer_frontiers"][0]
+    model_steps, prefix_tokens = count * (p + 7), count * (p - 1)
+    native_greedy = observed["observed_greedy_cohorts"]
+    greedy_rows = observed["native_greedy_rows"]
+    b1_outputs = observed["observed_B1_packet_outputs"]
+    bands = delta["native_batches_by_rows"]
+    fused, fused_rows = sum(bands.values()), sum(int(w) * n for w, n in bands.items())
+    require(delta["accepted_output_tokens"] == count * 8
+            and delta["completed_model_token_steps"] == model_steps
+            and delta["completed_forward_kernel_dispatches"] > 0
+            and fused <= model_steps // 2 and fused_rows <= model_steps,
+            "actual complete output/model-step/native bounds")
+    require(fused == delta["gpu_fused_batches"] == delta["state_cohort_publications"]
+            and fused_rows == delta["gpu_fused_rows_completed"] == delta["state_cohort_rows"],
+            "all native band/rows/atomic-publication counters close")
+    require(delta["native_sampler_dispatches_completed"] == 4 * native_greedy
+            and delta["completed_forward_kernel_dispatches"] >=
+            delta["native_projection_dispatches_completed"] + delta["native_sampler_dispatches_completed"]
+            and ((delta["native_projection_dispatches_completed"] > 0) if fused
+                 else (delta["native_projection_dispatches_completed"] == 0)),
+            "actual native greedy sampler and projection-dot subsets close")
+    left = snapshots["before"]["admission"]["state_cohort_publications"]
+    right = snapshots["after"]["admission"]["state_cohort_publications"]
+    require(type(left) is int and type(right) is int and 0 <= left <= right < 2**64
+            and right - left == fused, "source-pinned publication counter interval")
+    require(all(left < sequence <= right for sequence in observed["cohort_members"]),
+            "each native output sequence belongs to this owned sample counter domain")
+    if label == "native_prefix_zerogravity":
+        prefix_bands = {w: bands[w] - observed["observed_greedy_bands"][w] for w in bands}
+        require(all(n >= 0 for n in prefix_bands.values())
+                and delta["gpu_fused_greedy_batches"] == native_greedy
+                and delta["gpu_fused_state_only_batches"] == sum(prefix_bands.values())
+                and delta["gpu_fused_state_only_rows"] == sum(int(w)*n for w,n in prefix_bands.items())
+                and delta["gpu_fused_state_only_batches"] + native_greedy == fused
+                and delta["gpu_fused_state_only_rows"] + greedy_rows == fused_rows
+                and delta["gpu_fused_state_only_rows"] <= prefix_tokens,
+                "actual phase/band conservation; no fabricated StateOnly membership")
+        b1_prefix_tokens = prefix_tokens - delta["gpu_fused_state_only_rows"]
+        b1_model_steps = b1_prefix_tokens + b1_outputs
+        expected_calls = fused + b1_model_steps
+        require(delta["committed_gpu_roots"] == model_steps
+                and delta["gpu_forward_calls_completed"] == expected_calls
+                and b1_model_steps == model_steps - fused_rows,
+                "new one-teacher native/B1 model steps, API calls and independent roots close")
+        observed["derived_StateOnly_bands"] = prefix_bands
+        observed["StateOnly_membership"] = "unobserved; aggregate difference only"
+        prefix_calls = b1_prefix_tokens
+    else:
+        require(bands == observed["observed_greedy_bands"] and fused == native_greedy
+                and fused_rows == greedy_rows,
+                "legacy native cohorts are exactly all observed greedy packets")
+        prefix_calls = count * ((p - 1 + 31) // 32)
+        b1_prefix_tokens = prefix_tokens
+        b1_model_steps = prefix_tokens + b1_outputs
+        expected_calls = native_greedy + b1_outputs + prefix_calls
+        require(delta["gpu_forward_calls_completed"] == expected_calls
+                and delta["committed_gpu_roots"] == count * 8 + prefix_calls,
+                "legacy max32 prefix API chunks, actual native/B1 greedy calls and roots close")
+        observed["derived_StateOnly_bands"] = None
+        observed["StateOnly_membership"] = "legacy phase counters absent; source declares B1 prefix"
+    if count == 1:
+        require(fused == fused_rows == native_greedy == greedy_rows == 0 and b1_outputs == 8,
+                "B1 remains one actual request with no native cohort relabeling")
+        if label == "native_prefix_zerogravity":
+            require(all(delta[field] == 0 for field in PHASE_COUNTERS), "observed B1 native phase counters")
+    observed["publication_counter_interval"] = dict(before=left, after=right,
+        unobserved_native_publications=fused-native_greedy,
+        scope="same source-pinned WaveStats AtomicU64; warmup excluded; no StateOnly membership or numeric GPU ServiceId inferred")
+    observed["actual_scheduling"] = dict(offered_requests=count, client_concurrency=count,
+        configured_native_rows_max=8, actual_teacher_frontier=p,
+        declared_prefix_policy="one actual teacher per scheduled pass" if label == "native_prefix_zerogravity"
+            else "legacy max32 teacher chunks with final tail separate",
+        observed_forward_API_calls=delta["gpu_forward_calls_completed"],
+        observed_committed_roots=delta["committed_gpu_roots"],
+        actual_B1_packet_outputs=b1_outputs, derived_B1_prefix_tokens=b1_prefix_tokens,
+        derived_B1_prefix_API_calls=prefix_calls, derived_B1_model_steps=b1_model_steps,
+        actual_native_rows=fused_rows, completed_model_token_steps=model_steps,
+        physical_forward_kernel_dispatches=delta["completed_forward_kernel_dispatches"])
+    return outputs, observed
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural-c8-counter-contract.json
@@ -0,0 +1,18 @@
+{
+  "accepted_head": "c5eb1f8e717c79146937e585a145a0052eff747b",
+  "accepted_head_files": {
+    "crates/zerogravity-cli/src/serve/gpu_engine.rs": "e03886e60c1ea8426b11fe20a827454ea51295aa93fd40811a18ac8c6b073c86",
+    "crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs": "16783bd008babcecd600b18ede3ae8e598407673e01c6dbab3c8be003660739f",
+    "crates/zerogravity-cli/src/serve/mod.rs": "e10cd99e6b174139f8949477726a89f1d04fab5eec6568bc1d4bd30e937c444e"
+  },
+  "accepted_files": {
+    "crates/zerogravity-cli/src/serve/gpu_engine.rs": "e03886e60c1ea8426b11fe20a827454ea51295aa93fd40811a18ac8c6b073c86",
+    "crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs": "16783bd008babcecd600b18ede3ae8e598407673e01c6dbab3c8be003660739f",
+    "crates/zerogravity-cli/src/serve/mod.rs": "84e30f0a8909dcf677c8d6444d8a3b21ebe297e2e69acd7ca90b9fdc118afc37"
+  },
+  "current_files": {
+    "crates/zerogravity-cli/src/serve/gpu_engine.rs": "f44c60e4d3360825480aa9ec970280aedba7deac2d152cdf799cf025a0ac4279",
+    "crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs": "f57b08e6b48390f129334436168a2a32f421a9b54c113f092b0d07bcfd8ff14a",
+    "crates/zerogravity-cli/src/serve/mod.rs": "9430fb01cc7d971ccc936e91ac3f40ee71f20c1d3afcf6710bc7962bedb9f2ad"
+  }
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/NATURAL_C8.md
@@ -0,0 +1,116 @@
+# Additive natural concurrency8 workload
+
+Parent: `.agents/plans/serving-and-speculation/TODO.md` (P12).
+Author: `/root/baseline_review`; source-only preparation, root executes.
+
+## Separate scope
+
+- `measure_natural_c8.py`: explicit `--workloads c8,b1` (default), finite
+  predeclared ABBA or BAAB. c8 means8 offers/client concurrency8/configured native
+  maximum8. Actual native2/4/8 and B1 packet outputs are retained independently.
+- Original strict B8 driver, six-file be0bbee8 freeze, all existing attempts and
+  failure remain unchanged. No report-import/reclassification or retry mode.
+- Fresh outputs use `actual-natural-c8-measurements/same-zg-c8-*`, a distinct
+  schema/status and explicit workload semantics. Natural c8 may honestly show
+  all B1 if that is what occurred; it cannot establish native width8 execution.
+- The prior2B strict report `actual-measurements/same-zg-18c5xx5l/report.json`
+  stays UNQUALIFIED, SHA256
+  `5da6dc90184419ae53e467a4b86a9e35b117cc17255b6027ada4f9c19669b1c3`.
+  Its complete8 offers/64 outputs did not meet that different fixed-B8 guard.
+
+## Actual packet conservation
+
+- Every output has explicit `gpu.native_cohort`: actual sequence/width2/4/8,
+  or None for B1. No missing/unknown metadata is treated as B1.
+- For each actual native sequence require one consistent width and exactly that
+  many distinct request members. Retain each request/output index/producer
+  frontier/token ID/wire ID/provider response ID. Different generation indices
+  inside the same cohort are valid independent causal frontiers.
+- Per-request native sequence order must increase. Every output belongs to
+  exactly one complete native cohort or one listed B1 packet, including native
+  tail→B1 and B1 tail→native transitions.
+- Derive greedy band histogram only from observed packets. Subtract it from
+  health total bands to obtain aggregate StateOnly bands for the new source.
+  Require phase counts/rows to close. StateOnly membership is unobserved and is
+  never reconstructed or claimed.
+- Sampler count=4×observed native greedy cohorts, never4×all fused batches.
+  Full kernel count covers disjoint native projection-dot and sampler subsets;
+  actual kernel totals remain the source-bound physical receipts, not estimates.
+
+For prompt frontier p and N requests (N=8 or1), model steps=N×(p+7), outputs=8N.
+
+- New prefix: native StateOnly rows+B1 prefix tokens=N×(p−1). All B1 work consumes
+  one token per scheduled pass. API calls=fused batches+(model steps−fused rows);
+  committed roots=model steps. B1 greedy count is the actual listed B1 outputs.
+- Accepted old source: prefix remains B1 max32 with last prompt tail separate;
+  prefix API calls=N×ceil((p−1)/32). Calls=observed native greedy cohorts+B1
+  outputs+prefix API calls; committed roots=8N+prefix API calls. Old phase fields
+  remain absent/unobserved, never zero.
+
+## Counter domain and namespace
+
+The old compiled receipt identifies c5eb1f8 plus six exact overlays. GPU worker
+and cohort source are not overlaid: their matching HEAD bytes are retained in
+`natural-c8-counter-source/accepted/`. The exact health serializer overlay
+matches d0a6084 bytes SHA84e30f0a. Current three source hashes are retained under
+`current/` and match the source receipt in the preserved failed2B attempt.
+`natural-c8-counter-contract.json` pins these small source files. Runtime checks
+each reviewed build receipt against that exact worker/cohort/serializer contract
+before starting providers. This is a source/build attestation, not an independent
+reproducible-build proof.
+
+In both inspected sources:
+
+1. GpuServe creates one `Arc<WaveStats>` and passes its clone to the worker.
+2. After successful `cohort.commit`, its `cohort_publications.fetch_add(1)+1`
+   becomes `NativeCohortObservation.publication_sequence`.
+3. Metrics reads that same atomic; serializer exposes
+   `state_cohort_publications`. No earlier sequence reservation exists.
+4. Abandoned, failed, cancelled-after-Ready and stale proposals do not increment
+   it. B1 `single_completion` increments calls/model steps/roots, not this counter.
+5. Samples reject failures/refusals/discards and use quiescent before/after
+   health. Warmup is excluded. Each observed greedy sequence must lie in the
+   exact same counter interval; its size closes against total native batches.
+
+The counter belongs to the engine/worker stats, not a wire-exposed numeric GPU
+ServiceId. Each sample namespace records owned foreground PID, measurement run
+ID, singleton ready model alias, exact original source/tokenizer/binary hashes,
+GPU UUID and source contract. The launcher declares only that one model and
+requires readiness to expose exactly that alias. No SourceIdentity or ServiceId
+number is fabricated; sequences are never merged across samples/providers/models.
+
+## Preserved measurement/lifecycle rules
+
+All underlying workload, settlement, signature, source/config/tokenizer/build
+pins, provider argv/caps, greedy generation, memory sampling, lifecycle and
+cleanup helpers remain byte-identical to the original. Health's text label now
+says configured max8. Main changes only c8 naming/schema/output location,
+source-domain binding and the new actual grouping/accounting closure.
+
+- Same max8 tokens, context512, active/configured max8,4/4/5GiB caps, CPU4.
+- Same full IDs/text hashes/usage/terminal/provider/wire/prompt/frontier guards
+  and exact canonical equality across both providers and B1/c8 for each model.
+- Same offered/completed/outcome denominators, true individual-ID ITL/commit
+  clocks, fairness, fixed TTFT/ITL SLOs and raw health/memory/startup receipts.
+- Same serialized owned lifecycle, no proxies/redirects, bounded logs,
+  startup/warmup/measurement/shutdown receipts and GPU idle after cleanup.
+- No production tool/source change, user process termination or privilege change.
+- No claim of source-correct SLO eligibility, full model/state equivalence,
+  causal speedup or vLLM parity from this measurement script.
+
+## Prepared checks, not executed by author
+
+`test_natural_c8.py` contains13 source-free methods: actual mixed layout and B1
+remainder conservation; independent member indices/frontiers; original strict
+B8 refusal; missing whole/partial cohorts; duplicate/width/type/sequence errors;
+counter-domain interval; phase/band/sampler/API/root negatives; honest all-B1 c8;
+old/new B1 policy; legacy observability; wire/outcome/usage/frontier guards;
+source-contract refusal; unchanged underlying helpers.
+
+The fixture's10-cohort membership is copied from the preserved failed capture;
+other fixture payloads are explicitly synthetic. Tests do not promote that
+failed attempt. Author only parsed ASTs and inspected static source/diffs;
+no tests, providers, processes, models, operators, containers or GPU launches.
+
+Root reviews, runs source-free fixtures, then chooses fresh pinned binaries and
+an explicit finite ABBA/BAAB invocation. No automatic adjustment follows failure.
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/NATURAL_C8_TODO.md
@@ -0,0 +1,14 @@
+# Natural c8 variant
+
+Parent: `.agents/plans/serving-and-speculation/TODO.md` (P12).
+Owner: `/root/baseline_review`; additive source-only files in root-approved run.
+
+- [x] Read original driver/tests progressively; preserve be0 six-file freeze.
+- [x] Inspect preserved strict2B failure's small metadata without reclassifying.
+- [x] Bind sequence counter domain to exact accepted/current source+receipts.
+- [x] Prepare explicit c8 actual native2/4/8 +all B1 packet conservation.
+- [x] Retain original lifecycle, source, wire, memory, SLO and B1/c8 equality.
+- [x] Prepare13 source-free fixture methods; AST/static checks only, not run.
+- [ ] Independent/root source review and root source-free test execution.
+- [ ] Root fresh bounded c8+B1 ABBA/BAAB; no prior failure converted to PASS.
+- [ ] Root measurement interpretation and shared-plan reconciliation.
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural-c8-static-checks.json
@@ -0,0 +1,49 @@
+{
+  "status": "SOURCE_ONLY_AST_STATIC_NO_TESTS_IMPORTS_OR_RUNTIME",
+  "AST_parsed": [
+    "measure_natural_c8.py",
+    "test_natural_c8.py",
+    "natural_c8_functions.pyfrag"
+  ],
+  "fixture_methods_prepared_not_executed": [
+    "test_actual_mixed_layout_closes_all_57_batches_426_rows_and_four_B1_outputs",
+    "test_independent_frontiers_keep_different_output_indices_in_one_cohort",
+    "test_original_strict_B8_validator_still_refuses_this_layout",
+    "test_missing_whole_or_partial_native_membership_refuses",
+    "test_duplicate_inconsistent_or_noninteger_cohort_refuses",
+    "test_counter_domain_interval_cannot_reuse_warmup_or_foreign_sequences",
+    "test_phase_rows_bands_sampler_roots_APIs_and_failures_all_close",
+    "test_all_B1_is_reported_honestly_without_inventing_native_work",
+    "test_single_request_retains_oldmax32_vs_newone_teacher_policy",
+    "test_legacy_missing_phase_fields_are_never_observed_zero",
+    "test_full_wire_usage_identity_frontier_and_outcome_guards_retained",
+    "test_source_contract_rejects_unknown_counter_domains",
+    "test_underlying_lifecycle_wire_source_memory_and_SLO_helpers_are_exact"
+  ],
+  "byte_identical_underlying_helpers": [
+    "Sampler",
+    "build_receipt",
+    "deltas",
+    "gauge_observations",
+    "get_json",
+    "gpu",
+    "gpu_idle",
+    "json_read",
+    "local_opener",
+    "module",
+    "plan_type",
+    "quiescent",
+    "require",
+    "settled_result",
+    "sha",
+    "signature",
+    "workload"
+  ],
+  "warmup_and_measurement_bench_call_ASTs_identical": true,
+  "original_six_files_and_be0_freeze_unchanged": true,
+  "original_strict_B8_failed_report_unchanged": {
+    "path": ".agents/runs/serving-and-speculation-native-prefix-before-after-20261020/actual-measurements/same-zg-18c5xx5l/report.json",
+    "sha256": "5da6dc90184419ae53e467a4b86a9e35b117cc17255b6027ada4f9c19669b1c3"
+  },
+  "models_binaries_DSO_or_payloads_hashed_by_author": false
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural-c8-counter-source/accepted/crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
@@ -0,0 +1,263 @@
+//! Batch submission uses the original ResidentModel and all-row publisher.
+use super::*;
+
+pub(super) fn finish_prompt_cohort(
+    members: Vec<GpuActive>,
+    active: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+    native_rows: usize,
+) {
+    regroup(members, active, stats, Phase::Prefill, native_rows);
+}
+
+pub(super) fn advance_decode_cohort(
+    members: Vec<GpuActive>,
+    active: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+    native_rows: usize,
+) {
+    regroup(members, active, stats, Phase::Decode, native_rows);
+}
+
+fn regroup(
+    members: Vec<GpuActive>,
+    active: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+    phase: Phase,
+    native_rows: usize,
+) {
+    // EOS, limits, cancellation and output credit can remove any wave row.
+    // Repartition eligible survivors, never pad a native batch with a stop.
+    let eligible = members
+        .into_iter()
+        .filter_map(|row| eligible_member(row, active, stats))
+        .collect::<Vec<_>>();
+    if native_rows == 0 {
+        if phase == Phase::Prefill && eligible.len() > 1 {
+            execute(eligible, active, stats, phase, 0);
+        } else {
+            for row in eligible {
+                advance_member(row, active, stats);
+            }
+        }
+        return;
+    }
+    let bands = match gpu_batch::native_bands(eligible.len(), native_rows) {
+        Ok(bands) => bands,
+        Err(error) => {
+            for row in eligible {
+                stats.fail(&row, error.clone());
+            }
+            return;
+        }
+    };
+    let mut remaining = eligible.into_iter();
+    for width in bands {
+        if width == 1 {
+            advance_member(remaining.next().expect("planned native row"), active, stats);
+        } else {
+            let members = remaining.by_ref().take(width).collect();
+            execute(members, active, stats, phase, width);
+        }
+    }
+}
+
+fn execute(
+    mut members: Vec<GpuActive>,
+    active: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+    phase: Phase,
+    native_rows: usize,
+) {
+    if members
+        .iter()
+        .any(|row| row.cancelled.load(Ordering::Acquire))
+    {
+        regroup(members, active, stats, phase, native_rows);
+        return;
+    }
+    let Some(first) = members.first() else { return };
+    let model = Arc::clone(&first.resident);
+    let roots = members
+        .iter()
+        .map(|row| row.session.binding())
+        .collect::<Vec<_>>();
+    let valid = ResidentBatchRows::new(&roots, first.hidden_width).and_then(|batch| {
+        if members.iter().any(|row| {
+            !Arc::ptr_eq(&row.resident, &model)
+                || row.hidden_width != batch.row_elements()
+                || (phase == Phase::Prefill
+                    && (row.cursor + 1 != row.prompt.len() || row.device_token.is_some()))
+                || (phase == Phase::Decode
+                    && (row.cursor != row.prompt.len()
+                        || !row.packet_presented
+                        || row.device_token.is_none()
+                        || row.blocked_since.is_some()))
+        }) || (native_rows != 0
+            && (!matches!(native_rows, 2 | 4 | 8) || members.len() != native_rows))
+        {
+            return Err(zerogravity::Error::State(
+                "incompatible original GPU cohort".into(),
+            ));
+        }
+        batch.require_current(&roots)
+    });
+    if let Err(error) = valid {
+        let error = ServiceError::inference(error);
+        for row in members {
+            stats.fail(&row, error.clone());
+        }
+        return;
+    }
+    stats
+        .attempted
+        .fetch_add(members.len() as u64, Ordering::Relaxed);
+    let prepared = if phase == Phase::Decode {
+        let inputs = members
+            .iter()
+            .map(|row| {
+                (
+                    &row.session,
+                    row.device_token.as_ref().expect("validated packet"),
+                )
+            })
+            .collect::<Vec<_>>();
+        block_on(model.prepare_device_tokens_fused_cohort_admitted(&inputs))
+    } else {
+        let inputs = members
+            .iter()
+            .map(|row| (&row.session, row.prompt[row.cursor]))
+            .collect::<Vec<_>>();
+        if native_rows != 0 {
+            block_on(model.prepare_greedy_fused_cohort_admitted(&inputs))
+        } else {
+            block_on(model.prepare_greedy_cohort(&inputs)).map(Some)
+        }
+    };
+    let cohort = match prepared {
+        Ok(Some(value)) => value,
+        Ok(None) => {
+            stats
+                .native_admission_refusals
+                .fetch_add(1, Ordering::Relaxed);
+            // The admitted API guarantees no launch or root/packet mutation.
+            // Each capacity retry strictly reduces the attempted native band.
+            regroup(members, active, stats, phase, native_rows / 2);
+            return;
+        }
+        Err(error) => {
+            let error = ServiceError::inference(error);
+            for row in members {
+                stats.fail(&row, error.clone());
+            }
+            return;
+        }
+    };
+    let native = cohort.fused_dispatches() != 0 && cohort.native_greedy_dispatches() != 0;
+    record_completion(stats, &cohort, native);
+    if native_rows != 0 && !native {
+        drop(cohort);
+        stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
+        for row in members {
+            stats.fail(
+                &row,
+                ServiceError::new(
+                    500,
+                    "missing_native_receipt",
+                    "GPU cohort lacks native model/sampler receipts",
+                ),
+            );
+        }
+        return;
+    }
+    #[cfg(test)]
+    {
+        let hook = AFTER_COHORT_READY.with(|slot| slot.borrow_mut().take());
+        if let Some(hook) = hook {
+            hook();
+        }
+    }
+    if members
+        .iter()
+        .any(|row| row.cancelled.load(Ordering::Acquire))
+    {
+        // Known completion: discard private results, preserve original roots
+        // and packet-presented marks, then regroup only eligible survivors.
+        drop(cohort);
+        stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
+        regroup(members, active, stats, phase, native_rows);
+        return;
+    }
+    let mut originals = members
+        .iter_mut()
+        .map(|row| &mut row.session)
+        .collect::<Vec<_>>();
+    let published = cohort.commit(&mut originals);
+    let commit_observed_at = Instant::now();
+    drop(originals);
+    match published {
+        Ok(receipts) => {
+            let count = members.len() as u64;
+            let publication_sequence = stats
+                .cohort_publications
+                .fetch_add(1, Ordering::Relaxed)
+                .checked_add(1);
+            let native_cohort = if native {
+                publication_sequence.map(|publication_sequence| NativeCohortObservation {
+                    publication_sequence,
+                    width: members.len(),
+                })
+            } else {
+                None
+            };
+            stats.cohort_rows.fetch_add(count, Ordering::Relaxed);
+            stats.committed_roots.fetch_add(count, Ordering::Relaxed);
+            for (mut row, read) in members.into_iter().zip(receipts) {
+                if phase == Phase::Prefill {
+                    row.cursor += 1;
+                }
+                match row.install(read, commit_observed_at, native_cohort) {
+                    Ok(()) => active.push_back(row),
+                    Err(error) => stats.fail(&row, error),
+                }
+            }
+        }
+        Err(error) => {
+            stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
+            let error = ServiceError::inference(error);
+            for row in members {
+                stats.fail(&row, error.clone());
+            }
+        }
+    }
+}
+
+fn record_completion(stats: &WaveStats, cohort: &PreparedResidentCohort, native: bool) {
+    let count = cohort.len() as u64;
+    stats.completed.fetch_add(count, Ordering::Relaxed);
+    stats
+        .gpu_forwards
+        .fetch_add(if native { 1 } else { count }, Ordering::Relaxed);
+    stats
+        .kernel_dispatches
+        .fetch_add(cohort.dispatches() as u64, Ordering::Relaxed);
+    stats.model_tokens.fetch_add(count, Ordering::Relaxed);
+    if native {
+        stats.fused_batches.fetch_add(1, Ordering::Relaxed);
+        stats.fused_rows.fetch_add(count, Ordering::Relaxed);
+        stats
+            .native_projections
+            .fetch_add(cohort.fused_dispatches() as u64, Ordering::Relaxed);
+        stats
+            .native_sampler
+            .fetch_add(cohort.native_greedy_dispatches() as u64, Ordering::Relaxed);
+        if let Some(index) = match cohort.len() {
+            2 => Some(0),
+            4 => Some(1),
+            8 => Some(2),
+            _ => None,
+        } {
+            stats.native_bands[index].fetch_add(1, Ordering::Relaxed);
+        }
+    }
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural-c8-counter-source/accepted/crates/zerogravity-cli/src/serve/gpu_engine.rs
@@ -0,0 +1,933 @@
+//! Resident GPU OpenAI serving: one loaded model source, independent device
+//! StateRoots, one physical workspace and terminal completion per GPU step.
+//! Opt-in CUDA B2/B4/B8 waves use the original native ForwardPlan; other work
+//! interleaves through B1. Device feedback retains each SessionAuthority.
+
+use super::engine::{
+    EVENTS_PER_REQUEST, GeneratedText, GpuTokenObservation, MAX_ACTIVE_PER_WORKER,
+    NativeCohortObservation, Presentation, QUEUED_PER_WORKER, Request, RequestEvents,
+    ResponseEvent, Scheduled, ServiceError,
+};
+use super::gpu_batch::{self, Limits, Phase, Ready};
+use crate::GpuBackend;
+use futures_lite::future::block_on;
+use std::collections::{BTreeMap, VecDeque};
+use std::path::{Path, PathBuf};
+use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
+use std::sync::{Arc, mpsc};
+use std::thread;
+use std::time::{Duration, Instant};
+use zerogravity::gpu::{
+    InferenceReadback, PreparedResidentCohort, ResidentModel, ResidentSession, ResidentToken,
+    resident_batch::ResidentBatchRows,
+};
+use zerogravity::qwen35::LoadedQwenModel;
+use zerogravity::tokenize::QwenTokenizer;
+
+const GPU_CAPACITY: usize = 512;
+const PREFILL_QUANTUM: usize = 32;
+const OUTPUT_STALL_LIMIT: Duration = Duration::from_secs(30);
+mod cohort;
+use cohort::{advance_decode_cohort, finish_prompt_cohort};
+#[cfg(all(test, feature = "cuda"))]
+mod qualification;
+#[cfg(test)]
+thread_local! {
+    static AFTER_COHORT_READY: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
+        const { std::cell::RefCell::new(None) };
+}
+pub(super) struct GpuProfile {
+    pub workers: usize,
+    pub active: usize,
+    pub context: usize,
+    pub output: usize,
+    pub provider_mib: usize,
+    pub device_mib: usize,
+    pub native_rows: usize,
+}
+
+/// CubeCL CUDA's C++ backend requires a toolkit INCLUDE root in addition
+/// to the NVIDIA driver/NVRTC library. Validate before the device client:
+/// without this, a driver thread can panic/retry indefinitely.
+fn check_cuda_toolkit(path: &Path) -> Result<(), ServiceError> {
+    for header in ["include/cuda_runtime.h", "include/cuda_fp16.h"] {
+        if !path.join(header).is_file() {
+            return Err(ServiceError::new(
+                503,
+                "cuda_toolkit_missing",
+                format!("CUDA_PATH={} does not contain {header}", path.display()),
+            ));
+        }
+    }
+    if ![
+        path.join("include/cccl/cuda/std"),
+        path.join("include/cuda/std"),
+    ]
+    .iter()
+    .any(|p| p.is_dir())
+    {
+        return Err(ServiceError::new(
+            503,
+            "cuda_cccl_missing",
+            format!(
+                "CUDA_PATH={} has no CUDA CCCL device headers",
+                path.display()
+            ),
+        ));
+    }
+    Ok(())
+}
+fn preflight_cuda_toolkit(backend: GpuBackend) -> Result<(), ServiceError> {
+    if backend != GpuBackend::Cuda {
+        return Ok(());
+    }
+    let installed = std::env::var_os("CUDA_PATH")
+        .map(PathBuf::from)
+        .or_else(|| {
+            ["/usr/local/cuda", "/opt/cuda"]
+                .into_iter()
+                .map(PathBuf::from)
+                .find(|p| p.is_dir())
+        })
+        .or_else(|| {
+            Path::new("/usr/bin/nvcc")
+                .exists()
+                .then(|| PathBuf::from("/usr"))
+        })
+        .ok_or_else(|| {
+            ServiceError::new(
+                503,
+                "cuda_toolkit_missing",
+                "CUDA NVRTC requires CUDA_PATH pointing to toolkit include headers",
+            )
+        })?;
+    check_cuda_toolkit(&installed)
+}
+
+/// This is one host route to an already resident GPU source/graph.
+struct GpuModel {
+    resident: Arc<ResidentModel>,
+    tokenizer: Arc<QwenTokenizer>,
+    group: usize,
+    hidden_width: usize,
+    vocabulary_size: u64,
+    native_rows: usize,
+}
+#[derive(Default)]
+struct WaveStats {
+    attempted: AtomicU64,
+    completed: AtomicU64,
+    gpu_forwards: AtomicU64,
+    failed: AtomicU64,
+    cancelled: AtomicU64,
+    queued: AtomicUsize,
+    active: AtomicUsize,
+    groups: AtomicU64,
+    co_scheduled: AtomicU64,
+    cohort_publications: AtomicU64,
+    cohort_rows: AtomicU64,
+    kernel_dispatches: AtomicU64,
+    model_tokens: AtomicU64,
+    committed_roots: AtomicU64,
+    fused_batches: AtomicU64,
+    native_bands: [AtomicU64; 3],
+    native_admission_refusals: AtomicU64,
+    fused_rows: AtomicU64,
+    native_projections: AtomicU64,
+    native_sampler: AtomicU64,
+    discarded_cohorts: AtomicU64,
+    output_tokens: AtomicU64,
+    parked: AtomicUsize,
+}
+struct SessionCharge(Arc<WaveStats>);
+impl Drop for SessionCharge {
+    fn drop(&mut self) {
+        self.0.active.fetch_sub(1, Ordering::AcqRel);
+    }
+}
+pub(super) struct GpuMetrics {
+    pub attempted: u64,
+    pub completed: u64,
+    pub gpu_forwards: u64,
+    pub failed: u64,
+    pub cancelled: u64,
+    pub queued: usize,
+    pub active: usize,
+    pub groups: u64,
+    pub co_scheduled: u64,
+    pub cohort_publications: u64,
+    pub cohort_rows: u64,
+    pub kernel_dispatches: u64,
+    pub model_tokens: u64,
+    pub committed_roots: u64,
+    pub fused_batches: u64,
+    pub native_bands: [u64; 3],
+    pub native_admission_refusals: u64,
+    pub fused_rows: u64,
+    pub native_projections: u64,
+    pub native_sampler: u64,
+    pub discarded_cohorts: u64,
+    pub output_tokens: u64,
+    pub parked: usize,
+    pub native_rows: usize,
+}
+pub(super) struct GpuRouter {
+    models: Arc<BTreeMap<String, Arc<GpuModel>>>,
+    sender: mpsc::SyncSender<Scheduled>,
+    next_id: AtomicU64,
+    stats: Arc<WaveStats>,
+    pub(super) max_output: usize,
+    native_rows: usize,
+}
+impl GpuRouter {
+    pub(super) fn validate_profile(
+        workers: usize,
+        active: usize,
+        max_context: usize,
+        max_output: usize,
+        provider_mib: usize,
+        device_mib: usize,
+    ) -> Result<(), ServiceError> {
+        if workers != 1
+            || !(1..=MAX_ACTIVE_PER_WORKER).contains(&active)
+            || max_context != GPU_CAPACITY
+            || !(1..=GPU_CAPACITY).contains(&max_output)
+            || !(256..=65536).contains(&provider_mib)
+            || !(256..=65536).contains(&device_mib)
+            || provider_mib > device_mib
+        {
+            return Err(ServiceError::new(
+                400,
+                "invalid_gpu_profile",
+                "GPU serve requires workers=1, context=512, active=1..8, \
+                 output=1..512, provider/device 256..65536 MiB and provider<=device",
+            ));
+        }
+        Ok(())
+    }
+    pub(super) fn new(
+        declared: Vec<(String, PathBuf)>,
+        backend: GpuBackend,
+        profile: GpuProfile,
+    ) -> Result<Self, ServiceError> {
+        let GpuProfile {
+            workers,
+            active,
+            context,
+            output,
+            provider_mib,
+            device_mib,
+            native_rows,
+        } = profile;
+        Self::validate_profile(workers, active, context, output, provider_mib, device_mib)?;
+        if native_rows != 0
+            && (!matches!(native_rows, 2 | 4 | 8)
+                || backend != GpuBackend::Cuda
+                || active < native_rows)
+        {
+            return Err(ServiceError::new(
+                400,
+                "unsupported_batch_mode",
+                "fused serving requires CUDA, rows=2/4/8 and active>=rows",
+            ));
+        }
+        preflight_cuda_toolkit(backend)?;
+        if declared.is_empty() || declared.len() > 16 {
+            return Err(ServiceError::new(
+                400,
+                "invalid_models",
+                "GPU model routes require 1..16 IDs",
+            ));
+        }
+        // CPU fallback is forbidden: unlinked or absent provider must refuse
+        // before a resident model is constructed or HTTP announces readiness.
+        let provider_cap = provider_mib.checked_mul(1024 * 1024).ok_or_else(|| {
+            ServiceError::new(400, "provider_budget", "GPU provider cap overflow")
+        })?;
+        let device_cap = (device_mib as u64) * 1024 * 1024;
+        let mut models = BTreeMap::new();
+        let mut unique = BTreeMap::<PathBuf, Arc<GpuModel>>::new();
+        for (alias, path) in declared {
+            if alias.is_empty()
+                || alias.len() > 96
+                || !alias
+                    .bytes()
+                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
+                || models.contains_key(&alias)
+            {
+                return Err(ServiceError::new(
+                    400,
+                    "invalid_model",
+                    "bad/duplicate GPU route name",
+                ));
+            }
+            let path = std::fs::canonicalize(path).map_err(|_| {
+                ServiceError::new(400, "missing_model", "GGUF source does not exist")
+            })?;
+            if let Some(resident) = unique.get(&path) {
+                models.insert(alias, Arc::clone(resident));
+                continue;
+            }
+            let parent = path.parent().unwrap_or(Path::new("."));
+            let tokenizer = QwenTokenizer::from_file(&parent.join("tokenizer.json"))
+                .map_err(ServiceError::inference)?;
+            let source = LoadedQwenModel::open(&path, &parent.join("config.json"))
+                .map_err(ServiceError::inference)?;
+            let hidden_width = usize::try_from(source.geometry().hidden_size).map_err(|_| {
+                ServiceError::new(400, "source_hidden", "invalid Qwen source hidden width")
+            })?;
+            let vocabulary_size = source.geometry().vocab_size;
+            let plan = source
+                .forward_plan(GPU_CAPACITY, true, true)
+                .map_err(ServiceError::inference)?;
+            let device = backend.device().map_err(ServiceError::inference)?;
+            let resident = block_on(ResidentModel::load(
+                device,
+                source,
+                plan,
+                provider_cap,
+                device_cap,
+            ))
+            .map_err(ServiceError::inference)?;
+            let shared = Arc::new(GpuModel {
+                resident,
+                tokenizer: Arc::new(tokenizer),
+                group: unique.len(),
+                hidden_width,
+                vocabulary_size,
+                native_rows,
+            });
+            unique.insert(path, Arc::clone(&shared));
+            models.insert(alias, shared);
+        }
+        let models = Arc::new(models);
+        let (sender, receiver) = mpsc::sync_channel(QUEUED_PER_WORKER);
+        let sources = Arc::clone(&models);
+        let stats = Arc::new(WaveStats::default());
+        let worker_stats = Arc::clone(&stats);
+        thread::Builder::new()
+            .name("zg-resident-gpu".into())
+            .spawn(move || worker(receiver, sources, active, worker_stats, native_rows))
+            .map_err(|e| ServiceError::new(500, "worker_spawn", e.to_string()))?;
+        Ok(Self {
+            models,
+            sender,
+            next_id: AtomicU64::new(1),
+            stats,
+            max_output: output,
+            native_rows,
+        })
+    }
+    pub(super) fn metrics(&self) -> GpuMetrics {
+        GpuMetrics {
+            attempted: self.stats.attempted.load(Ordering::Relaxed),
+            completed: self.stats.completed.load(Ordering::Relaxed),
+            gpu_forwards: self.stats.gpu_forwards.load(Ordering::Relaxed),
+            failed: self.stats.failed.load(Ordering::Relaxed),
+            cancelled: self.stats.cancelled.load(Ordering::Relaxed),
+            queued: self.stats.queued.load(Ordering::Acquire),
+            active: self.stats.active.load(Ordering::Acquire),
+            groups: self.stats.groups.load(Ordering::Relaxed),
+            co_scheduled: self.stats.co_scheduled.load(Ordering::Relaxed),
+            cohort_publications: self.stats.cohort_publications.load(Ordering::Relaxed),
+            cohort_rows: self.stats.cohort_rows.load(Ordering::Relaxed),
+            kernel_dispatches: self.stats.kernel_dispatches.load(Ordering::Relaxed),
+            model_tokens: self.stats.model_tokens.load(Ordering::Relaxed),
+            committed_roots: self.stats.committed_roots.load(Ordering::Relaxed),
+            fused_batches: self.stats.fused_batches.load(Ordering::Relaxed),
+            native_bands: std::array::from_fn(|i| {
+                self.stats.native_bands[i].load(Ordering::Relaxed)
+            }),
+            native_admission_refusals: self.stats.native_admission_refusals.load(Ordering::Relaxed),
+            fused_rows: self.stats.fused_rows.load(Ordering::Relaxed),
+            native_projections: self.stats.native_projections.load(Ordering::Relaxed),
+            native_sampler: self.stats.native_sampler.load(Ordering::Relaxed),
+            discarded_cohorts: self.stats.discarded_cohorts.load(Ordering::Relaxed),
+            output_tokens: self.stats.output_tokens.load(Ordering::Relaxed),
+            parked: self.stats.parked.load(Ordering::Relaxed),
+            native_rows: self.native_rows,
+        }
+    }
+    pub(super) fn model_ids(&self) -> impl Iterator<Item = &str> {
+        self.models.keys().map(String::as_str)
+    }
+    pub(super) fn request(&self, request: Request) -> Result<RequestEvents, ServiceError> {
+        if !self.models.contains_key(&request.model) {
+            return Err(ServiceError::new(
+                404,
+                "model_not_found",
+                "unregistered GPU model",
+            ));
+        }
+        if request.input.is_empty()
+            || request.input.len() > 1024 * 1024
+            || request.max_tokens == 0
+            || request.max_tokens > self.max_output
+        {
+            return Err(ServiceError::new(
+                400,
+                "invalid_request",
+                "GPU request bounds",
+            ));
+        }
+        let model = request.model.clone();
+        let token_metrics = request.token_metrics.clone();
+        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
+        if id == u64::MAX {
+            return Err(ServiceError::new(
+                503,
+                "id_exhausted",
+                "request IDs exhausted",
+            ));
+        }
+        let cancelled = Arc::new(AtomicBool::new(false));
+        let (tx, rx) = mpsc::sync_channel(EVENTS_PER_REQUEST);
+        self.stats.queued.fetch_add(1, Ordering::AcqRel);
+        self.sender
+            .try_send(Scheduled {
+                request,
+                events: tx,
+                cancelled: Arc::clone(&cancelled),
+            })
+            .map_err(|err| {
+                self.stats.queued.fetch_sub(1, Ordering::AcqRel);
+                match err {
+                    mpsc::TrySendError::Full(_) => {
+                        ServiceError::new(429, "queue_full", "GPU queue at capacity")
+                    }
+                    mpsc::TrySendError::Disconnected(_) => {
+                        ServiceError::new(503, "worker_unavailable", "GPU executor unavailable")
+                    }
+                }
+            })?;
+        Ok(RequestEvents::new(id, model, rx, cancelled).observed(token_metrics))
+    }
+}
+struct GpuActive {
+    _charge: SessionCharge,
+    resident: Arc<ResidentModel>,
+    session: ResidentSession,
+    group: usize,
+    hidden_width: usize,
+    cancelled: Arc<AtomicBool>,
+    errors: mpsc::SyncSender<ResponseEvent>,
+    output: GeneratedText,
+    prompt: Vec<u32>,
+    cursor: usize,
+    device_token: Option<ResidentToken>,
+    token_commit_observed_at: Option<Instant>,
+    token_gpu_observation: Option<GpuTokenObservation>,
+    native_rows: usize,
+    packet_presented: bool,
+    blocked_since: Option<Instant>,
+}
+impl GpuActive {
+    fn new(
+        job: Scheduled,
+        model: Arc<GpuModel>,
+        stats: Arc<WaveStats>,
+    ) -> Result<Self, ServiceError> {
+        let prompt = model
+            .tokenizer
+            .prompt_tokens(&job.request.input, job.request.chat)
+            .map_err(ServiceError::inference)?;
+        if prompt.is_empty()
+            || prompt.len() > GPU_CAPACITY
+            || prompt
+                .len()
+                .checked_add(job.request.max_tokens - 1)
+                .is_none_or(|total| total > GPU_CAPACITY)
+        {
+            return Err(ServiceError::new(
+                400,
+                "context_overflow",
+                "resident GPU profile admits at most 512 consumed tokens",
+            ));
+        }
+        let output = GeneratedText::new(
+            Arc::clone(&model.tokenizer),
+            job.events.clone(),
+            job.request.stream,
+            job.request.max_tokens,
+            prompt.len(),
+        )
+        .observed(job.request.token_metrics.clone(), model.vocabulary_size);
+        // ResidentSession allocates independent, physical source-bound GPU
+        // state. Neither its tensor source nor its StateRoot is CPU fallback.
+        let session = block_on(model.resident.session()).map_err(ServiceError::inference)?;
+        stats.active.fetch_add(1, Ordering::AcqRel);
+        Ok(Self {
+            _charge: SessionCharge(stats),
+            resident: Arc::clone(&model.resident),
+            session,
+            group: model.group,
+            hidden_width: model.hidden_width,
+            cancelled: job.cancelled,
+            errors: job.events,
+            output,
+            prompt,
+            cursor: 0,
+            device_token: None,
+            token_commit_observed_at: None,
+            token_gpu_observation: None,
+            native_rows: model.native_rows,
+            packet_presented: false,
+            blocked_since: None,
+        })
+    }
+    fn ready(&self) -> Ready {
+        let remaining = self.prompt.len().saturating_sub(self.cursor);
+        Ready {
+            source: self.group,
+            phase: if remaining > 0 {
+                Phase::Prefill
+            } else {
+                Phase::Decode
+            },
+            frontier: self.session.binding().frontier() as usize,
+            tokens: if remaining > 0 {
+                self.prefill_count()
+            } else {
+                1
+            },
+            // This is finite scheduler credit, NOT a claim that this is
+            // the true physical GPU workspace footprint.
+            scratch_bytes: 32 * 1024 * 1024,
+        }
+    }
+    fn prefill_count(&self) -> usize {
+        let remaining = self.prompt.len() - self.cursor;
+        if self.native_rows != 0 && remaining > 1 {
+            (remaining - 1).min(PREFILL_QUANTUM)
+        } else {
+            remaining.min(PREFILL_QUANTUM)
+        }
+    }
+    /// Presentation precedes consumption. A Ready-cohort retry retains this
+    /// mark and the original packet; only a newly committed packet resets it.
+    fn present_decode(&mut self, stats: &WaveStats) -> Result<Presentation, ServiceError> {
+        let presented = if self.packet_presented {
+            self.output.resume()?
+        } else {
+            let token = self.device_token.as_ref().ok_or_else(|| {
+                ServiceError::new(
+                    500,
+                    "missing_device_token",
+                    "resident GPU feedback is missing",
+                )
+            })?;
+            let before = self.output.generated_token_count();
+            let committed = self.token_commit_observed_at.ok_or_else(|| {
+                ServiceError::new(
+                    500,
+                    "missing_commit_observation",
+                    "resident packet has no publication observation",
+                )
+            })?;
+            let status =
+                self.output
+                    .present_observed(token.id, committed, self.token_gpu_observation)?;
+            stats.output_tokens.fetch_add(
+                (self.output.generated_token_count() - before) as u64,
+                Ordering::Relaxed,
+            );
+            self.packet_presented = true;
+            status
+        };
+        if presented == Presentation::Blocked {
+            let since = self.blocked_since.get_or_insert_with(Instant::now);
+            if since.elapsed() >= OUTPUT_STALL_LIMIT {
+                return Err(ServiceError::new(
+                    503,
+                    "backpressure_timeout",
+                    "client output credit deadline expired",
+                ));
+            }
+        } else {
+            self.blocked_since = None;
+        }
+        Ok(presented)
+    }
+    fn install(
+        &mut self,
+        read: InferenceReadback,
+        commit_observed_at: Instant,
+        native_cohort: Option<NativeCohortObservation>,
+    ) -> Result<(), ServiceError> {
+        let token = read.device_token.ok_or_else(|| {
+            ServiceError::new(
+                500,
+                "missing_device_token",
+                "GPU completion has no original greedy packet",
+            )
+        })?;
+        // install follows successful original publication; this is the exact
+        // packet-producing committed binding, including a B1 retry.
+        self.token_gpu_observation = Some(GpuTokenObservation {
+            producer_frontier: self.session.binding().frontier(),
+            native_cohort,
+        });
+        self.device_token = Some(token);
+        self.token_commit_observed_at = Some(commit_observed_at);
+        self.packet_presented = false;
+        self.blocked_since = None;
+        Ok(())
+    }
+    /// Original B1 source execution, including all unmatched native-wave rows.
+    fn step(&mut self, stats: &WaveStats) -> Result<MemberStep, ServiceError> {
+        if self.cursor < self.prompt.len() {
+            let end = self.cursor + self.prefill_count();
+            let read = if end < self.prompt.len() {
+                block_on(
+                    self.session
+                        .prefill_without_logits(&self.prompt[self.cursor..end], false),
+                )
+            } else {
+                block_on(self.session.forward_greedy(&self.prompt[self.cursor..end]))
+            }
+            .map_err(ServiceError::inference)?;
+            let committed = Instant::now();
+            stats.single_completion(&read);
+            if end == self.prompt.len() {
+                self.install(read, committed, None)?;
+            }
+            self.cursor = end;
+            return Ok(MemberStep::Forwarded);
+        }
+        match self.present_decode(stats)? {
+            Presentation::Finished => return Ok(MemberStep::Finished),
+            Presentation::Blocked => return Ok(MemberStep::Parked),
+            Presentation::Ready => {}
+        }
+        if self.cancelled.load(Ordering::Acquire) {
+            return Ok(MemberStep::Cancelled);
+        }
+        let token = self.device_token.as_ref().ok_or_else(|| {
+            ServiceError::new(
+                500,
+                "missing_device_token",
+                "resident GPU feedback is missing",
+            )
+        })?;
+        let read =
+            block_on(self.session.forward_device_token(token)).map_err(ServiceError::inference)?;
+        let committed = Instant::now();
+        stats.single_completion(&read);
+        self.install(read, committed, None)?;
+        Ok(MemberStep::Forwarded)
+    }
+}
+
+enum MemberStep {
+    Forwarded,
+    Finished,
+    Parked,
+    Cancelled,
+}
+impl WaveStats {
+    fn single_completion(&self, read: &InferenceReadback) {
+        self.gpu_forwards.fetch_add(1, Ordering::Relaxed);
+        self.kernel_dispatches
+            .fetch_add(read.dispatches as u64, Ordering::Relaxed);
+        self.model_tokens.fetch_add(
+            (read.state_only_steps + read.logit_steps) as u64,
+            Ordering::Relaxed,
+        );
+        self.committed_roots.fetch_add(1, Ordering::Relaxed);
+    }
+    fn fail(&self, row: &GpuActive, error: ServiceError) {
+        if error.status == 499 && error.code == "cancelled" {
+            self.cancelled.fetch_add(1, Ordering::Relaxed);
+        } else {
+            self.failed.fetch_add(1, Ordering::Relaxed);
+        }
+        let _ = row.errors.try_send(ResponseEvent::Error(error));
+    }
+}
+fn advance_member(next: GpuActive, active: &mut VecDeque<GpuActive>, stats: &WaveStats) {
+    let Some(mut next) = eligible_member(next, active, stats) else {
+        return;
+    };
+    stats.attempted.fetch_add(1, Ordering::Relaxed);
+    match next.step(stats) {
+        Ok(MemberStep::Parked) => active.push_back(next),
+        Ok(MemberStep::Cancelled) => {
+            stats.cancelled.fetch_add(1, Ordering::Relaxed);
+        }
+        Ok(outcome) => {
+            stats.completed.fetch_add(1, Ordering::Relaxed);
+            if matches!(outcome, MemberStep::Forwarded) {
+                active.push_back(next);
+            }
+        }
+        Err(error) => {
+            stats.fail(&next, error);
+        }
+    }
+}
+/// Resume bounded output before building new GPU waves. Parked rows own
+/// no submitted command, queue slot or physical workspace guard.
+fn eligible_member(
+    mut row: GpuActive,
+    parked: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+) -> Option<GpuActive> {
+    if row.cancelled.load(Ordering::Acquire) {
+        stats.cancelled.fetch_add(1, Ordering::Relaxed);
+        return None;
+    }
+    if row.cursor < row.prompt.len() {
+        return Some(row);
+    }
+    match row.present_decode(stats) {
+        Ok(Presentation::Ready) => Some(row),
+        Ok(Presentation::Blocked) => {
+            parked.push_back(row);
+            None
+        }
+        Ok(Presentation::Finished) => {
+            stats.attempted.fetch_add(1, Ordering::Relaxed);
+            stats.completed.fetch_add(1, Ordering::Relaxed);
+            None
+        }
+        Err(error) => {
+            stats.attempted.fetch_add(1, Ordering::Relaxed);
+            stats.fail(&row, error);
+            None
+        }
+    }
+}
+
+fn worker(
+    rx: mpsc::Receiver<Scheduled>,
+    models: Arc<BTreeMap<String, Arc<GpuModel>>>,
+    active_limit: usize,
+    stats: Arc<WaveStats>,
+    native_rows: usize,
+) {
+    let mut active = VecDeque::<GpuActive>::new();
+    loop {
+        active.retain(|row| {
+            if row.cancelled.load(Ordering::Acquire) {
+                stats.cancelled.fetch_add(1, Ordering::Relaxed);
+                false
+            } else {
+                true
+            }
+        });
+        while active.len() < active_limit {
+            let next = if active.is_empty() {
+                match rx.recv_timeout(Duration::from_millis(25)) {
+                    Ok(v) => Some(v),
+                    Err(mpsc::RecvTimeoutError::Timeout) => break,
+                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
+                }
+            } else {
+                rx.try_recv().ok()
+            };
+            let Some(job) = next else { break };
+            stats.queued.fetch_sub(1, Ordering::AcqRel);
+            if job.cancelled.load(Ordering::Acquire) {
+                stats.cancelled.fetch_add(1, Ordering::Relaxed);
+                continue;
+            }
+            let Some(model) = models.get(&job.request.model) else {
+                continue;
+            };
+            let errors = job.events.clone();
+            match GpuActive::new(job, Arc::clone(model), Arc::clone(&stats)) {
+                Ok(session) => active.push_back(session),
+                Err(e) => {
+                    stats.failed.fetch_add(1, Ordering::Relaxed);
+                    let _ = errors.try_send(ResponseEvent::Error(e));
+                }
+            }
+        }
+        if active.is_empty() {
+            continue;
+        }
+        let mut eligible = VecDeque::new();
+        let mut parked = VecDeque::new();
+        for row in active.drain(..) {
+            if let Some(row) = eligible_member(row, &mut parked, &stats) {
+                eligible.push_back(row);
+            }
+        }
+        active = parked;
+        stats.parked.store(active.len(), Ordering::Relaxed);
+        if eligible.is_empty() {
+            // No locks/queue slot while waiting for bounded transport credit.
+            if !active.is_empty() {
+                thread::sleep(Duration::from_millis(2));
+            }
+            continue;
+        }
+        let ready = eligible.iter().map(GpuActive::ready).collect::<Vec<_>>();
+        let bounds = Limits {
+            max_members: if native_rows != 0 {
+                native_rows
+            } else {
+                active_limit
+            },
+            max_tokens: 64,
+            max_scratch: 256 * 1024 * 1024,
+        };
+        let planned = if native_rows != 0 {
+            gpu_batch::form_decode(&ready, bounds)
+        } else {
+            gpu_batch::form(&ready, bounds)
+        };
+        let waves = match planned {
+            Ok(groups) => groups,
+            Err(error) => {
+                for failed in eligible.drain(..) {
+                    stats.failed.fetch_add(1, Ordering::Relaxed);
+                    let _ = failed.errors.try_send(ResponseEvent::Error(error.clone()));
+                }
+                continue;
+            }
+        };
+        let mut pending = eligible.drain(..).map(Some).collect::<Vec<_>>();
+        for wave in waves {
+            // Validate source/epoch/unique committed session roots BEFORE
+            // issuing even the first physical GPU member. P12 source
+            // batch admission is shared by native bands and original B1 paths.
+            let admitted = wave
+                .indices
+                .iter()
+                .filter_map(|&i| pending[i].as_ref())
+                .filter(|row| !row.cancelled.load(Ordering::Acquire))
+                .collect::<Vec<_>>();
+            if let Some(first) = admitted.first() {
+                let roots = admitted
+                    .iter()
+                    .map(|row| row.session.binding())
+                    .collect::<Vec<_>>();
+                let verification = ResidentBatchRows::new(&roots, first.hidden_width);
+                let verification = verification.and_then(|cohort| {
+                    if admitted
+                        .iter()
+                        .any(|row| row.hidden_width != cohort.row_elements())
+                    {
+                        return Err(zerogravity::Error::State(
+                            "GPU model wave hidden widths differ".into(),
+                        ));
+                    }
+                    cohort.require_current(&roots)?;
+                    Ok(cohort)
+                });
+                if let Err(error) = verification {
+                    stats
+                        .failed
+                        .fetch_add(admitted.len() as u64, Ordering::Relaxed);
+                    let refusal = ServiceError::inference(error);
+                    for index in wave.indices {
+                        if let Some(row) = pending[index].take() {
+                            let _ = row.errors.try_send(ResponseEvent::Error(refusal.clone()));
+                        }
+                    }
+                    continue;
+                }
+            }
+            stats.groups.fetch_add(1, Ordering::Relaxed);
+            if wave.indices.len() > 1 {
+                stats
+                    .co_scheduled
+                    .fetch_add(wave.indices.len() as u64, Ordering::Relaxed);
+            }
+            // Exact prompt tails share the existing checked root publisher.
+            // Native mode uses multi-row kernels; interleaved mode keeps B1 graphs.
+            let batch_ready = wave.indices.len() > 1
+                && wave.indices.iter().all(|&i| {
+                    pending[i].as_ref().is_some_and(|row| {
+                        !row.cancelled.load(Ordering::Acquire)
+                            && row.cursor + 1 == row.prompt.len()
+                            && row.device_token.is_none()
+                    })
+                });
+            if batch_ready {
+                let members = wave
+                    .indices
+                    .iter()
+                    .map(|&i| pending[i].take().expect("unique admitted wave row"))
+                    .collect::<Vec<_>>();
+                finish_prompt_cohort(members, &mut active, &stats, native_rows);
+                continue;
+            }
+            if native_rows != 0
+                && wave.indices.len() > 1
+                && wave.indices.iter().all(|&i| {
+                    pending[i]
+                        .as_ref()
+                        .is_some_and(|row| row.cursor == row.prompt.len())
+                })
+            {
+                let members = wave
+                    .indices
+                    .iter()
+                    .map(|&i| pending[i].take().expect("unique admitted decode row"))
+                    .collect();
+                advance_decode_cohort(members, &mut active, &stats, native_rows);
+                continue;
+            }
+            for index in wave.indices {
+                let next = pending[index]
+                    .take()
+                    .expect("GPU wave planner must cover unique ready slots");
+                advance_member(next, &mut active, &stats);
+            }
+        }
+        stats.parked.store(
+            active
+                .iter()
+                .filter(|row| row.blocked_since.is_some())
+                .count(),
+            Ordering::Relaxed,
+        );
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    #[test]
+    fn cuda_requires_runtime_and_cccl_headers_before_device_startup() {
+        use std::sync::atomic::{AtomicU64, Ordering};
+        static NEXT: AtomicU64 = AtomicU64::new(0);
+        let root = std::env::temp_dir().join(format!(
+            "zerogravity-cuda-prerequisite-{}-{}",
+            std::process::id(),
+            NEXT.fetch_add(1, Ordering::Relaxed)
+        ));
+        std::fs::create_dir_all(root.join("include")).unwrap();
+        assert_eq!(
+            check_cuda_toolkit(&root).unwrap_err().code,
+            "cuda_toolkit_missing"
+        );
+        std::fs::write(root.join("include/cuda_runtime.h"), b"").unwrap();
+        std::fs::write(root.join("include/cuda_fp16.h"), b"").unwrap();
+        assert_eq!(
+            check_cuda_toolkit(&root).unwrap_err().code,
+            "cuda_cccl_missing"
+        );
+        std::fs::create_dir_all(root.join("include/cccl/cuda/std")).unwrap();
+        assert!(check_cuda_toolkit(&root).is_ok());
+        std::fs::remove_dir_all(root).unwrap();
+    }
+
+    #[test]
+    fn gpu_profile_is_bounded_before_any_device_lookup() {
+        assert!(GpuRouter::validate_profile(1, 2, 512, 16, 2048, 4096).is_ok());
+        for bad in [
+            (2, 2, 512, 16, 2048, 4096),
+            (1, 0, 512, 16, 2048, 4096),
+            (1, 9, 512, 16, 2048, 4096),
+            (1, 2, 8192, 16, 2048, 4096),
+            (1, 2, 512, 513, 2048, 4096),
+            (1, 2, 512, 16, 4096, 2048),
+            (1, 2, 512, 16, 0, 4096),
+        ] {
+            assert!(GpuRouter::validate_profile(bad.0, bad.1, bad.2, bad.3, bad.4, bad.5).is_err());
+        }
+    }
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural-c8-counter-source/accepted/crates/zerogravity-cli/src/serve/mod.rs
@@ -0,0 +1,511 @@
+//! Opt-in local OpenAI-compatible HTTP serving.
+//! Uses original Qwen CPU source and private independent model sessions.
+
+mod engine;
+#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+mod gpu_batch;
+#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+mod gpu_engine;
+mod http;
+mod token_metrics;
+
+#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+use crate::GpuBackend;
+use engine::{Request, RequestEvents, ServiceError};
+
+enum Router {
+    Cpu(engine::Router),
+    #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+    Gpu(gpu_engine::GpuRouter),
+}
+impl Router {
+    fn request(&self, request: Request) -> Result<RequestEvents, ServiceError> {
+        match self {
+            Self::Cpu(v) => v.request(request),
+            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+            Self::Gpu(v) => v.request(request),
+        }
+    }
+    fn max_output(&self) -> usize {
+        match self {
+            Self::Cpu(v) => v.max_output,
+            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+            Self::Gpu(v) => v.max_output,
+        }
+    }
+    fn model_ids(&self) -> Box<dyn Iterator<Item = &str> + '_> {
+        match self {
+            Self::Cpu(v) => Box::new(v.model_ids()),
+            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+            Self::Gpu(v) => Box::new(v.model_ids()),
+        }
+    }
+    fn admission(&self) -> serde_json::Value {
+        match self {
+            Self::Cpu(v) => {
+                let (limit, reserved, remaining) = v.memory_snapshot().unwrap_or((0, 0, 0));
+                serde_json::json!({"kind":"configured_host_cpu",
+                    "host_limit_bytes":limit,"reserved_bytes":reserved,
+                    "remaining_bytes":remaining})
+            }
+            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+            Self::Gpu(v) => {
+                let m = v.metrics();
+                serde_json::json!({"kind":"resident_gpu",
+                    "physical_accounting":"ResidentService/ResourceArbiter",
+                    "batch_mode":if m.native_rows != 0 {"fused"} else {"interleaved"},
+                    "native_batch_rows_max":m.native_rows.max(1),
+                    "gpu_step_attempts":m.attempted,
+                    "completed_gpu_quanta":m.completed,
+                    "completed_quantum_unit":"request work, including terminal presentation",
+                    "gpu_forward_steps_completed":m.gpu_forwards,
+                    "gpu_forward_calls_completed":m.gpu_forwards,
+                    "completed_model_token_steps":m.model_tokens,
+                    "completed_forward_kernel_dispatches":m.kernel_dispatches,
+                    "committed_gpu_roots":m.committed_roots,
+                    "native_projection_dispatches_completed":m.native_projections,
+                    "native_sampler_dispatches_completed":m.native_sampler,
+                    "gpu_fused_rows_completed":m.fused_rows,
+                    "gpu_fused_batches_by_rows":{"2":m.native_bands[0],"4":m.native_bands[1],"8":m.native_bands[2]},
+                    "native_admission_refusals":m.native_admission_refusals,
+                    "discarded_ready_cohorts":m.discarded_cohorts,
+                    "accepted_output_tokens":m.output_tokens,
+                    "output_credit_parked_requests":m.parked,
+                    "failed_gpu_quanta":m.failed,
+                    "cancelled_requests":m.cancelled,
+                    "queued_requests":m.queued,
+                    "active_sessions":m.active,
+                    "scheduling_groups":m.groups,
+                    "co_scheduled_members":m.co_scheduled,
+                    "state_cohort_publications":m.cohort_publications,
+                    "state_cohort_rows":m.cohort_rows,
+                    "gpu_fused_batches":m.fused_batches})
+            }
+        }
+    }
+}
+
+use serde::Deserialize;
+use std::net::{SocketAddr, TcpListener};
+use std::path::{Path, PathBuf};
+use std::sync::Arc;
+
+#[derive(Debug, Deserialize)]
+#[serde(deny_unknown_fields)]
+struct YamlConfig {
+    #[serde(default = "default_listen")]
+    listen: String,
+    models: Vec<YamlModel>,
+    #[serde(default = "default_workers")]
+    workers: usize,
+    #[serde(default = "default_active")]
+    active_per_worker: usize,
+    #[serde(default = "default_context")]
+    max_context: usize,
+    #[serde(default = "default_output")]
+    max_output: usize,
+    #[serde(default = "default_host")]
+    host_mib: usize,
+    #[serde(default = "default_backend")]
+    backend: String,
+    #[serde(default = "default_provider_mib")]
+    provider_mib: usize,
+    #[serde(default = "default_device_mib")]
+    device_mib: usize,
+    #[serde(default = "default_batch_mode")]
+    batch_mode: String,
+    #[serde(default = "default_batch_rows")]
+    batch_rows: usize,
+    #[serde(default)]
+    enable_per_request_metrics: bool,
+}
+#[derive(Debug, Deserialize)]
+#[serde(deny_unknown_fields)]
+struct YamlModel {
+    id: String,
+    path: PathBuf,
+}
+fn default_listen() -> String {
+    "127.0.0.1:8000".into()
+}
+fn default_workers() -> usize {
+    2
+}
+fn default_active() -> usize {
+    2
+}
+fn default_context() -> usize {
+    1024
+}
+fn default_output() -> usize {
+    128
+}
+fn default_host() -> usize {
+    4096
+}
+fn default_backend() -> String {
+    "cpu".into()
+}
+fn default_provider_mib() -> usize {
+    2048
+}
+fn default_device_mib() -> usize {
+    4096
+}
+fn default_batch_rows() -> usize {
+    2
+}
+fn default_batch_mode() -> String {
+    "interleaved".into()
+}
+
+impl YamlConfig {
+    fn cli(args: &[String]) -> Result<Self, ServiceError> {
+        let mut config = Self {
+            listen: default_listen(),
+            models: Vec::new(),
+            workers: default_workers(),
+            active_per_worker: default_active(),
+            max_context: default_context(),
+            max_output: default_output(),
+            host_mib: default_host(),
+            backend: default_backend(),
+            provider_mib: default_provider_mib(),
+            device_mib: default_device_mib(),
+            batch_mode: default_batch_mode(),
+            batch_rows: default_batch_rows(),
+            enable_per_request_metrics: false,
+        };
+        if args.first().is_some_and(|x| x == "--config") {
+            if args.len() != 2 {
+                return Err(ServiceError::new(
+                    400,
+                    "invalid_config",
+                    "--config must stand alone",
+                ));
+            }
+            let path = Path::new(&args[1]);
+            let raw = std::fs::read(path).map_err(|_| {
+                ServiceError::new(400, "config_unreadable", "cannot read serving YAML")
+            })?;
+            if raw.is_empty() || raw.len() > 64 * 1024 {
+                return Err(ServiceError::new(
+                    400,
+                    "config_limit",
+                    "serving YAML exceeds 64 KiB",
+                ));
+            }
+            let mut parsed: Self = serde_yaml::from_slice(&raw).map_err(|_| {
+                ServiceError::new(400, "config_invalid", "invalid typed serving YAML")
+            })?;
+            let base = path.parent().unwrap_or(Path::new("."));
+            for model in &mut parsed.models {
+                if model.path.is_relative() {
+                    model.path = base.join(&model.path);
+                }
+            }
+            return Ok(parsed);
+        }
+        let mut it = args.iter();
+        while let Some(flag) = it.next() {
+            if flag == "--enable-per-request-metrics" {
+                config.enable_per_request_metrics = true;
+                continue;
+            }
+            let value = it.next().ok_or_else(|| {
+                ServiceError::new(400, "invalid_flag", format!("{flag} requires a value"))
+            })?;
+            match flag.as_str() {
+                "--model" => {
+                    let (id, path) = value.split_once('=').ok_or_else(|| {
+                        ServiceError::new(400, "invalid_model", "--model requires ID=PATH")
+                    })?;
+                    config.models.push(YamlModel {
+                        id: id.into(),
+                        path: PathBuf::from(path),
+                    });
+                }
+                "--listen" => config.listen = value.clone(),
+                "--workers" => config.workers = parse_usize(value, flag)?,
+                "--active-per-worker" => config.active_per_worker = parse_usize(value, flag)?,
+                "--max-context" => config.max_context = parse_usize(value, flag)?,
+                "--max-output" => config.max_output = parse_usize(value, flag)?,
+                "--host-mib" => config.host_mib = parse_usize(value, flag)?,
+                "--backend" => config.backend = value.clone(),
+                "--provider-mib" => config.provider_mib = parse_usize(value, flag)?,
+                "--device-mib" => config.device_mib = parse_usize(value, flag)?,
+                "--batch-mode" => config.batch_mode = value.clone(),
+                "--batch-rows" => config.batch_rows = parse_usize(value, flag)?,
+                _ => {
+                    return Err(ServiceError::new(
+                        400,
+                        "invalid_flag",
+                        format!("unknown serve flag {flag}"),
+                    ));
+                }
+            }
+        }
+        Ok(config)
+    }
+}
+fn parse_usize(value: &str, flag: &str) -> Result<usize, ServiceError> {
+    value.parse::<usize>().map_err(|_| {
+        ServiceError::new(
+            400,
+            "invalid_integer",
+            format!("invalid integer for {flag}"),
+        )
+    })
+}
+fn validate_batch_mode(
+    mode: &str,
+    backend: &str,
+    active: usize,
+    rows: usize,
+) -> Result<(), ServiceError> {
+    match mode {
+        "interleaved" => Ok(()),
+        "fused" if backend == "cuda" && matches!(rows, 2 | 4 | 8) && active >= rows => Ok(()),
+        _ => Err(ServiceError::new(
+            400,
+            "unsupported_batch_mode",
+            "use interleaved, or fused with CUDA, rows=2/4/8 and active>=rows",
+        )),
+    }
+}
+
+pub fn run(args: &[String]) -> Result<(), ServiceError> {
+    if args.len() == 1 && args[0] == "--help" {
+        println!(
+            "USAGE: zerogravity serve --config config.yaml | \
+             --model ID=MODEL.gguf [--listen 127.0.0.1:8000] \
+             [--backend cpu|cuda|vulkan-nvidia|vulkan-amd|rocm] \
+             [--workers 2] [--active-per-worker 2] \
+             [--max-context 1024] [--max-output 128] \
+             [--host-mib 4096] [--provider-mib 2048] [--device-mib 4096] \
+             [--batch-mode interleaved|fused] [--batch-rows 2|4|8] \
+             [--enable-per-request-metrics]\n\
+             ENDPOINTS: GET /health, GET /v1/models, \
+             POST /v1/chat/completions, POST /v1/completions.\n\
+             Qwen3.5 greedy text CPU or resident GPU, no inference fallback. \
+             GPU: workers=1, context=512. Default interleaved; fused enables \
+             CUDA B2/B4/B8 compatible prompt tails and independent decode frontiers \
+             with active>=rows. Default rows=2; survivors regroup without padding. \
+             Prefill groups require equal frontiers; unmatched rows use B1. \
+             CUDA needs CUDA_PATH with runtime+CCCL headers. \
+             Opt-in metrics permit return_token_ids=true with bounded token IDs, \
+             publication-observed times and separate HTTP delivery observations. \
+             Loopback only; optional ZEROGRAVITY_API_KEY."
+        );
+        return Ok(());
+    }
+
+    let options = YamlConfig::cli(args)?;
+    let address: SocketAddr = options.listen.parse().map_err(|_| {
+        ServiceError::new(400, "invalid_listen", "--listen requires numeric IP:PORT")
+    })?;
+    let key = std::env::var("ZEROGRAVITY_API_KEY")
+        .ok()
+        .filter(|v| !v.is_empty());
+    // No native TLS; a Bearer token over unencrypted non-loopback
+    // sockets must not be mistaken for secure remote authentication.
+    // Use a trusted TLS reverse proxy forwarding to local loopback.
+    if !address.ip().is_loopback() {
+        return Err(ServiceError::new(
+            403,
+            "remote_bind_unsupported",
+            "HTTP service is loopback-only; expose through a TLS/auth reverse proxy",
+        ));
+    }
+    validate_batch_mode(
+        &options.batch_mode,
+        &options.backend,
+        options.active_per_worker,
+        options.batch_rows,
+    )?;
+    let declared = options.models.into_iter().map(|m| (m.id, m.path)).collect();
+    let backend = options.backend.as_str();
+    let router = match backend {
+        "cpu" => Router::Cpu(engine::Router::new(
+            declared,
+            options.workers,
+            options.active_per_worker,
+            options.max_context,
+            options.max_output,
+            options.host_mib,
+        )?),
+        "vulkan-nvidia" | "vulkan-amd" | "cuda" | "rocm" => {
+            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+            {
+                let selected = match backend {
+                    "vulkan-nvidia" => GpuBackend::VulkanNvidia,
+                    "vulkan-amd" => GpuBackend::VulkanAmd,
+                    "cuda" => GpuBackend::Cuda,
+                    _ => GpuBackend::Hip,
+                };
+                if !selected.compiled() {
+                    return Err(ServiceError::new(
+                        400,
+                        "backend_not_compiled",
+                        "selected GPU backend was not compiled into this binary",
+                    ));
+                }
+                Router::Gpu(gpu_engine::GpuRouter::new(
+                    declared,
+                    selected,
+                    gpu_engine::GpuProfile {
+                        workers: options.workers,
+                        active: options.active_per_worker,
+                        context: options.max_context,
+                        output: options.max_output,
+                        provider_mib: options.provider_mib,
+                        device_mib: options.device_mib,
+                        native_rows: if options.batch_mode == "fused" {
+                            options.batch_rows
+                        } else {
+                            0
+                        },
+                    },
+                )?)
+            }
+            #[cfg(not(any(feature = "vulkan", feature = "cuda", feature = "hip")))]
+            {
+                return Err(ServiceError::new(
+                    400,
+                    "backend_not_compiled",
+                    "compile with --features serve,vulkan or serve,cuda or serve,hip",
+                ));
+            }
+        }
+        _ => {
+            return Err(ServiceError::new(
+                400,
+                "invalid_backend",
+                "backend must be cpu, vulkan-nvidia, vulkan-amd, cuda or rocm",
+            ));
+        }
+    };
+    let router = Arc::new(router);
+    let listener = TcpListener::bind(address)
+        .map_err(|e| ServiceError::new(500, "bind_failed", e.to_string()))?;
+    eprintln!(
+        "zerogravity: serving {:?} Qwen on {address}, {} worker(s); max-context {}, max-output {}",
+        backend, options.workers, options.max_context, options.max_output
+    );
+    http::serve(listener, router, key, options.enable_per_request_metrics)
+        .map_err(|e| ServiceError::new(500, "http_error", e.to_string()))
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    #[test]
+    fn native_serving_mode_is_bounded_before_model_or_device_lookup() {
+        for backend in ["cpu", "cuda", "vulkan-nvidia", "vulkan-amd", "rocm"] {
+            assert!(validate_batch_mode("interleaved", backend, 1, 2).is_ok());
+            if backend != "cuda" {
+                let err = run(&[
+                    "--backend".into(),
+                    backend.into(),
+                    "--batch-mode".into(),
+                    "fused".into(),
+                    "--model".into(),
+                    "unqualified=missing.gguf".into(),
+                ])
+                .unwrap_err();
+                assert_eq!(err.code, "unsupported_batch_mode");
+            }
+        }
+        assert!(validate_batch_mode("fused", "cuda", 2, 2).is_ok());
+        assert!(validate_batch_mode("fused", "cuda", 4, 4).is_ok());
+        assert!(validate_batch_mode("fused", "cuda", 8, 8).is_ok());
+        for rows in [0, 1, 3, 5, 6, 7, 9, 16] {
+            assert!(validate_batch_mode("fused", "cuda", 8, rows).is_err());
+        }
+        assert!(validate_batch_mode("fused", "cuda", 4, 8).is_err());
+        for mode in ["fused", "fused-b4", "unknown"] {
+            let err = run(&[
+                "--backend".into(),
+                "cuda".into(),
+                "--batch-mode".into(),
+                mode.into(),
+                "--active-per-worker".into(),
+                "1".into(),
+                "--model".into(),
+                "unqualified=missing.gguf".into(),
+            ])
+            .unwrap_err();
+            assert_eq!(err.code, "unsupported_batch_mode");
+        }
+    }
+
+    #[test]
+    fn serve_cli_is_bounded_and_configured_only_explicitly() {
+        let cfg = YamlConfig::cli(&[
+            "--model".into(),
+            "a=weights.gguf".into(),
+            "--workers".into(),
+            "3".into(),
+        ])
+        .unwrap();
+        assert_eq!(cfg.models.len(), 1);
+        assert_eq!(cfg.workers, 3);
+        assert_eq!(cfg.backend, "cpu");
+        assert!(!cfg.enable_per_request_metrics);
+        let observed = YamlConfig::cli(&[
+            "--enable-per-request-metrics".into(),
+            "--model".into(),
+            "a=weights.gguf".into(),
+        ])
+        .unwrap();
+        assert!(observed.enable_per_request_metrics);
+        assert!(YamlConfig::cli(&["--model".into(), "missing".into()]).is_err());
+        assert!(YamlConfig::cli(&["--workers".into(), "no".into()]).is_err());
+        assert!(
+            YamlConfig::cli(&[
+                "--config".into(),
+                "config.yaml".into(),
+                "--workers".into(),
+                "2".into()
+            ])
+            .is_err()
+        );
+        let example = include_str!("../../../../examples/serving/qwen-cpu.yaml");
+        let parsed: YamlConfig = serde_yaml::from_str(example).unwrap();
+        assert_eq!(parsed.listen, "127.0.0.1:8000");
+        assert_eq!(parsed.models.len(), 2);
+        assert_eq!(parsed.models[0].path, parsed.models[1].path);
+        assert_eq!((parsed.workers, parsed.active_per_worker), (2, 2));
+        assert_eq!(parsed.host_mib, 4096);
+        let gpu_config = include_str!("../../../../examples/serving/qwen-cuda.yaml");
+        let gpu: YamlConfig = serde_yaml::from_str(gpu_config).unwrap();
+        assert_eq!(gpu.backend, "cuda");
+        assert_eq!(gpu.batch_mode, "interleaved");
+        assert_eq!(gpu.max_context, 512);
+        assert_eq!((gpu.workers, gpu.active_per_worker), (1, 2));
+        assert_eq!((gpu.provider_mib, gpu.device_mib), (2048, 4096));
+        assert_eq!(gpu.models.len(), 2);
+        assert!(
+            serde_yaml::from_str::<YamlConfig>(
+                "listen: 127.0.0.1:8080\\nmodels: []\\nunknown_provider: yes"
+            )
+            .is_err()
+        );
+        let err = run(&[
+            "--listen".into(),
+            "0.0.0.0:9911".into(),
+            "--model".into(),
+            "unqualified=missing.gguf".into(),
+        ])
+        .unwrap_err();
+        assert_eq!(err.code, "remote_bind_unsupported");
+        let unsupported = run(&["--batch-mode".into(), "fused".into()]).unwrap_err();
+        assert_eq!(unsupported.code, "unsupported_batch_mode");
+        // Refuse uncompiled GPU backend before initializing a device.
+        #[cfg(not(any(feature = "vulkan", feature = "cuda", feature = "hip")))]
+        {
+            let missing = run(&["--backend".into(), "cuda".into()]).unwrap_err();
+            assert_eq!(missing.code, "backend_not_compiled");
+        }
+    }
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural-c8-counter-source/current/crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs
@@ -0,0 +1,331 @@
+//! Batch submission uses the original ResidentModel and all-row publisher.
+use super::*;
+
+pub(super) fn finish_prompt_cohort(
+    members: Vec<GpuActive>,
+    active: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+    native_rows: usize,
+) {
+    regroup(members, active, stats, Phase::Prefill, native_rows);
+}
+
+pub(super) fn advance_decode_cohort(
+    members: Vec<GpuActive>,
+    active: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+    native_rows: usize,
+) {
+    regroup(members, active, stats, Phase::Decode, native_rows);
+}
+
+fn regroup(
+    members: Vec<GpuActive>,
+    active: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+    phase: Phase,
+    native_rows: usize,
+) {
+    // EOS, limits, cancellation and output credit can remove any wave row.
+    // Repartition eligible survivors, never pad a native batch with a stop.
+    let eligible = members
+        .into_iter()
+        .filter_map(|row| eligible_member(row, active, stats))
+        .collect::<Vec<_>>();
+    if native_rows == 0 {
+        if phase == Phase::Prefill && eligible.len() > 1 {
+            execute(eligible, active, stats, phase, 0, false);
+        } else {
+            for row in eligible {
+                advance_member(row, active, stats);
+            }
+        }
+        return;
+    }
+    if phase == Phase::Prefill {
+        // A teacher prefix has no generated output. Keep that readout barrier
+        // distinct from the last prompt token before choosing native widths.
+        let (prefix, tail): (Vec<_>, Vec<_>) = eligible
+            .into_iter()
+            .partition(|row| row.prompt.len().saturating_sub(row.cursor) > 1);
+        bands(prefix, active, stats, phase, native_rows, true);
+        bands(tail, active, stats, phase, native_rows, false);
+    } else {
+        bands(eligible, active, stats, phase, native_rows, false);
+    }
+}
+
+fn bands(
+    eligible: Vec<GpuActive>,
+    active: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+    phase: Phase,
+    native_rows: usize,
+    state_only: bool,
+) {
+    let bands = match gpu_batch::native_bands(eligible.len(), native_rows) {
+        Ok(bands) => bands,
+        Err(error) => {
+            for row in eligible {
+                stats.fail(&row, error.clone());
+            }
+            return;
+        }
+    };
+    let mut remaining = eligible.into_iter();
+    for width in bands {
+        if width == 1 {
+            advance_member(remaining.next().expect("planned native row"), active, stats);
+        } else {
+            let members = remaining.by_ref().take(width).collect();
+            execute(members, active, stats, phase, width, state_only);
+        }
+    }
+}
+
+fn execute(
+    mut members: Vec<GpuActive>,
+    active: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+    phase: Phase,
+    native_rows: usize,
+    state_only: bool,
+) {
+    if members
+        .iter()
+        .any(|row| row.cancelled.load(Ordering::Acquire))
+    {
+        regroup(members, active, stats, phase, native_rows);
+        return;
+    }
+    let Some(first) = members.first() else { return };
+    let model = Arc::clone(&first.resident);
+    let roots = members
+        .iter()
+        .map(|row| row.session.binding())
+        .collect::<Vec<_>>();
+    let valid = ResidentBatchRows::new(&roots, first.hidden_width).and_then(|batch| {
+        if members.iter().any(|row| {
+            !Arc::ptr_eq(&row.resident, &model)
+                || row.hidden_width != batch.row_elements()
+                || (phase == Phase::Prefill
+                    && (row.cursor >= row.prompt.len()
+                        || (row.prompt.len() - row.cursor > 1) != state_only
+                        || row.device_token.is_some()))
+                || (phase == Phase::Decode
+                    && (row.cursor != row.prompt.len()
+                        || !row.packet_presented
+                        || row.device_token.is_none()
+                        || row.blocked_since.is_some()))
+        }) || (state_only && (phase != Phase::Prefill || native_rows == 0))
+            || (native_rows != 0
+                && (!matches!(native_rows, 2 | 4 | 8) || members.len() != native_rows))
+        {
+            return Err(zerogravity::Error::State(
+                "incompatible original GPU cohort".into(),
+            ));
+        }
+        batch.require_current(&roots)
+    });
+    if let Err(error) = valid {
+        let error = ServiceError::inference(error);
+        for row in members {
+            stats.fail(&row, error.clone());
+        }
+        return;
+    }
+    stats
+        .attempted
+        .fetch_add(members.len() as u64, Ordering::Relaxed);
+    let prepared = if phase == Phase::Decode {
+        let inputs = members
+            .iter()
+            .map(|row| {
+                (
+                    &row.session,
+                    row.device_token.as_ref().expect("validated packet"),
+                )
+            })
+            .collect::<Vec<_>>();
+        block_on(model.prepare_device_tokens_fused_cohort_admitted(&inputs))
+    } else {
+        let inputs = members
+            .iter()
+            .map(|row| (&row.session, row.prompt[row.cursor]))
+            .collect::<Vec<_>>();
+        if state_only {
+            block_on(model.prepare_state_fused_cohort_admitted(&inputs))
+        } else if native_rows != 0 {
+            block_on(model.prepare_greedy_fused_cohort_admitted(&inputs))
+        } else {
+            block_on(model.prepare_greedy_cohort(&inputs)).map(Some)
+        }
+    };
+    let cohort = match prepared {
+        Ok(Some(value)) => value,
+        Ok(None) => {
+            stats
+                .native_admission_refusals
+                .fetch_add(1, Ordering::Relaxed);
+            // The admitted API guarantees no launch or root/packet mutation.
+            // Each capacity retry strictly reduces the attempted native band.
+            regroup(members, active, stats, phase, native_rows / 2);
+            return;
+        }
+        Err(error) => {
+            let error = ServiceError::inference(error);
+            for row in members {
+                stats.fail(&row, error.clone());
+            }
+            return;
+        }
+    };
+    let native = cohort.fused_dispatches() != 0
+        && if state_only {
+            cohort.native_greedy_dispatches() == 0
+        } else {
+            cohort.native_greedy_dispatches() != 0
+        };
+    record_completion(stats, &cohort, native, state_only);
+    if native_rows != 0 && !native {
+        drop(cohort);
+        stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
+        for row in members {
+            stats.fail(
+                &row,
+                ServiceError::new(
+                    500,
+                    "missing_native_receipt",
+                    "GPU cohort lacks native model/readout receipts",
+                ),
+            );
+        }
+        return;
+    }
+    #[cfg(test)]
+    {
+        let hook = AFTER_COHORT_READY.with(|slot| slot.borrow_mut().take());
+        if let Some(hook) = hook {
+            hook();
+        }
+    }
+    if members
+        .iter()
+        .any(|row| row.cancelled.load(Ordering::Acquire))
+    {
+        // Known completion: discard private results, preserve original roots
+        // and packet-presented marks, then regroup only eligible survivors.
+        drop(cohort);
+        stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
+        regroup(members, active, stats, phase, native_rows);
+        return;
+    }
+    let mut originals = members
+        .iter_mut()
+        .map(|row| &mut row.session)
+        .collect::<Vec<_>>();
+    let published = cohort.commit(&mut originals);
+    let commit_observed_at = Instant::now();
+    drop(originals);
+    match published {
+        Ok(receipts) => {
+            let count = members.len() as u64;
+            let publication_sequence = stats
+                .cohort_publications
+                .fetch_add(1, Ordering::Relaxed)
+                .checked_add(1);
+            let native_cohort = if native {
+                publication_sequence.map(|publication_sequence| NativeCohortObservation {
+                    publication_sequence,
+                    width: members.len(),
+                })
+            } else {
+                None
+            };
+            stats.cohort_rows.fetch_add(count, Ordering::Relaxed);
+            stats.committed_roots.fetch_add(count, Ordering::Relaxed);
+            for (mut row, read) in members.into_iter().zip(receipts) {
+                if phase == Phase::Prefill {
+                    row.cursor += 1;
+                }
+                if state_only {
+                    // Prefix publication changes only the independent root.
+                    // No token packet, output presentation or usage is created.
+                    if read.state_only_steps != 1
+                        || read.logit_steps != 0
+                        || !read.logits.is_empty()
+                        || read.device_token.is_some()
+                    {
+                        stats.fail(
+                            &row,
+                            ServiceError::new(
+                                500,
+                                "invalid_prefix_readout",
+                                "state-only GPU completion produced a generated readout",
+                            ),
+                        );
+                    } else {
+                        active.push_back(row);
+                    }
+                } else {
+                    match row.install(read, commit_observed_at, native_cohort) {
+                        Ok(()) => active.push_back(row),
+                        Err(error) => stats.fail(&row, error),
+                    }
+                }
+            }
+        }
+        Err(error) => {
+            stats.discarded_cohorts.fetch_add(1, Ordering::Relaxed);
+            let error = ServiceError::inference(error);
+            for row in members {
+                stats.fail(&row, error.clone());
+            }
+        }
+    }
+}
+
+fn record_completion(
+    stats: &WaveStats,
+    cohort: &PreparedResidentCohort,
+    native: bool,
+    state_only: bool,
+) {
+    let count = cohort.len() as u64;
+    stats.completed.fetch_add(count, Ordering::Relaxed);
+    stats
+        .gpu_forwards
+        .fetch_add(if native { 1 } else { count }, Ordering::Relaxed);
+    stats
+        .kernel_dispatches
+        .fetch_add(cohort.dispatches() as u64, Ordering::Relaxed);
+    stats.model_tokens.fetch_add(count, Ordering::Relaxed);
+    if native {
+        stats.fused_batches.fetch_add(1, Ordering::Relaxed);
+        if state_only {
+            stats
+                .fused_state_only_batches
+                .fetch_add(1, Ordering::Relaxed);
+            stats
+                .fused_state_only_rows
+                .fetch_add(count, Ordering::Relaxed);
+        } else {
+            stats.fused_greedy_batches.fetch_add(1, Ordering::Relaxed);
+        }
+        stats.fused_rows.fetch_add(count, Ordering::Relaxed);
+        stats
+            .native_projections
+            .fetch_add(cohort.fused_dispatches() as u64, Ordering::Relaxed);
+        stats
+            .native_sampler
+            .fetch_add(cohort.native_greedy_dispatches() as u64, Ordering::Relaxed);
+        if let Some(index) = match cohort.len() {
+            2 => Some(0),
+            4 => Some(1),
+            8 => Some(2),
+            _ => None,
+        } {
+            stats.native_bands[index].fetch_add(1, Ordering::Relaxed);
+        }
+    }
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural-c8-counter-source/current/crates/zerogravity-cli/src/serve/gpu_engine.rs
@@ -0,0 +1,945 @@
+//! Resident GPU OpenAI serving: one loaded model source, independent device
+//! StateRoots, one physical workspace and terminal completion per GPU step.
+//! Opt-in CUDA B2/B4/B8 waves use the original native ForwardPlan; other work
+//! interleaves through B1. Device feedback retains each SessionAuthority.
+
+use super::engine::{
+    EVENTS_PER_REQUEST, GeneratedText, GpuTokenObservation, MAX_ACTIVE_PER_WORKER,
+    NativeCohortObservation, Presentation, QUEUED_PER_WORKER, Request, RequestEvents,
+    ResponseEvent, Scheduled, ServiceError,
+};
+use super::gpu_batch::{self, Limits, Phase, Ready};
+use crate::GpuBackend;
+use futures_lite::future::block_on;
+use std::collections::{BTreeMap, VecDeque};
+use std::path::{Path, PathBuf};
+use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
+use std::sync::{Arc, mpsc};
+use std::thread;
+use std::time::{Duration, Instant};
+use zerogravity::gpu::{
+    InferenceReadback, PreparedResidentCohort, ResidentModel, ResidentSession, ResidentToken,
+    resident_batch::ResidentBatchRows,
+};
+use zerogravity::qwen35::LoadedQwenModel;
+use zerogravity::tokenize::QwenTokenizer;
+
+const GPU_CAPACITY: usize = 512;
+const PREFILL_QUANTUM: usize = 32;
+const OUTPUT_STALL_LIMIT: Duration = Duration::from_secs(30);
+mod cohort;
+use cohort::{advance_decode_cohort, finish_prompt_cohort};
+#[cfg(all(test, feature = "cuda"))]
+mod qualification;
+#[cfg(test)]
+thread_local! {
+    static AFTER_COHORT_READY: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
+        const { std::cell::RefCell::new(None) };
+}
+pub(super) struct GpuProfile {
+    pub workers: usize,
+    pub active: usize,
+    pub context: usize,
+    pub output: usize,
+    pub provider_mib: usize,
+    pub device_mib: usize,
+    pub native_rows: usize,
+}
+
+/// CubeCL CUDA's C++ backend requires a toolkit INCLUDE root in addition
+/// to the NVIDIA driver/NVRTC library. Validate before the device client:
+/// without this, a driver thread can panic/retry indefinitely.
+fn check_cuda_toolkit(path: &Path) -> Result<(), ServiceError> {
+    for header in ["include/cuda_runtime.h", "include/cuda_fp16.h"] {
+        if !path.join(header).is_file() {
+            return Err(ServiceError::new(
+                503,
+                "cuda_toolkit_missing",
+                format!("CUDA_PATH={} does not contain {header}", path.display()),
+            ));
+        }
+    }
+    if ![
+        path.join("include/cccl/cuda/std"),
+        path.join("include/cuda/std"),
+    ]
+    .iter()
+    .any(|p| p.is_dir())
+    {
+        return Err(ServiceError::new(
+            503,
+            "cuda_cccl_missing",
+            format!(
+                "CUDA_PATH={} has no CUDA CCCL device headers",
+                path.display()
+            ),
+        ));
+    }
+    Ok(())
+}
+fn preflight_cuda_toolkit(backend: GpuBackend) -> Result<(), ServiceError> {
+    if backend != GpuBackend::Cuda {
+        return Ok(());
+    }
+    let installed = std::env::var_os("CUDA_PATH")
+        .map(PathBuf::from)
+        .or_else(|| {
+            ["/usr/local/cuda", "/opt/cuda"]
+                .into_iter()
+                .map(PathBuf::from)
+                .find(|p| p.is_dir())
+        })
+        .or_else(|| {
+            Path::new("/usr/bin/nvcc")
+                .exists()
+                .then(|| PathBuf::from("/usr"))
+        })
+        .ok_or_else(|| {
+            ServiceError::new(
+                503,
+                "cuda_toolkit_missing",
+                "CUDA NVRTC requires CUDA_PATH pointing to toolkit include headers",
+            )
+        })?;
+    check_cuda_toolkit(&installed)
+}
+
+/// This is one host route to an already resident GPU source/graph.
+struct GpuModel {
+    resident: Arc<ResidentModel>,
+    tokenizer: Arc<QwenTokenizer>,
+    group: usize,
+    hidden_width: usize,
+    vocabulary_size: u64,
+    native_rows: usize,
+}
+#[derive(Default)]
+struct WaveStats {
+    attempted: AtomicU64,
+    completed: AtomicU64,
+    gpu_forwards: AtomicU64,
+    failed: AtomicU64,
+    cancelled: AtomicU64,
+    queued: AtomicUsize,
+    active: AtomicUsize,
+    groups: AtomicU64,
+    co_scheduled: AtomicU64,
+    cohort_publications: AtomicU64,
+    cohort_rows: AtomicU64,
+    kernel_dispatches: AtomicU64,
+    model_tokens: AtomicU64,
+    committed_roots: AtomicU64,
+    fused_batches: AtomicU64,
+    fused_state_only_batches: AtomicU64,
+    fused_state_only_rows: AtomicU64,
+    fused_greedy_batches: AtomicU64,
+    native_bands: [AtomicU64; 3],
+    native_admission_refusals: AtomicU64,
+    fused_rows: AtomicU64,
+    native_projections: AtomicU64,
+    native_sampler: AtomicU64,
+    discarded_cohorts: AtomicU64,
+    output_tokens: AtomicU64,
+    parked: AtomicUsize,
+}
+struct SessionCharge(Arc<WaveStats>);
+impl Drop for SessionCharge {
+    fn drop(&mut self) {
+        self.0.active.fetch_sub(1, Ordering::AcqRel);
+    }
+}
+pub(super) struct GpuMetrics {
+    pub attempted: u64,
+    pub completed: u64,
+    pub gpu_forwards: u64,
+    pub failed: u64,
+    pub cancelled: u64,
+    pub queued: usize,
+    pub active: usize,
+    pub groups: u64,
+    pub co_scheduled: u64,
+    pub cohort_publications: u64,
+    pub cohort_rows: u64,
+    pub kernel_dispatches: u64,
+    pub model_tokens: u64,
+    pub committed_roots: u64,
+    pub fused_batches: u64,
+    pub fused_state_only_batches: u64,
+    pub fused_state_only_rows: u64,
+    pub fused_greedy_batches: u64,
+    pub native_bands: [u64; 3],
+    pub native_admission_refusals: u64,
+    pub fused_rows: u64,
+    pub native_projections: u64,
+    pub native_sampler: u64,
+    pub discarded_cohorts: u64,
+    pub output_tokens: u64,
+    pub parked: usize,
+    pub native_rows: usize,
+}
+pub(super) struct GpuRouter {
+    models: Arc<BTreeMap<String, Arc<GpuModel>>>,
+    sender: mpsc::SyncSender<Scheduled>,
+    next_id: AtomicU64,
+    stats: Arc<WaveStats>,
+    pub(super) max_output: usize,
+    native_rows: usize,
+}
+impl GpuRouter {
+    pub(super) fn validate_profile(
+        workers: usize,
+        active: usize,
+        max_context: usize,
+        max_output: usize,
+        provider_mib: usize,
+        device_mib: usize,
+    ) -> Result<(), ServiceError> {
+        if workers != 1
+            || !(1..=MAX_ACTIVE_PER_WORKER).contains(&active)
+            || max_context != GPU_CAPACITY
+            || !(1..=GPU_CAPACITY).contains(&max_output)
+            || !(256..=65536).contains(&provider_mib)
+            || !(256..=65536).contains(&device_mib)
+            || provider_mib > device_mib
+        {
+            return Err(ServiceError::new(
+                400,
+                "invalid_gpu_profile",
+                "GPU serve requires workers=1, context=512, active=1..8, \
+                 output=1..512, provider/device 256..65536 MiB and provider<=device",
+            ));
+        }
+        Ok(())
+    }
+    pub(super) fn new(
+        declared: Vec<(String, PathBuf)>,
+        backend: GpuBackend,
+        profile: GpuProfile,
+    ) -> Result<Self, ServiceError> {
+        let GpuProfile {
+            workers,
+            active,
+            context,
+            output,
+            provider_mib,
+            device_mib,
+            native_rows,
+        } = profile;
+        Self::validate_profile(workers, active, context, output, provider_mib, device_mib)?;
+        if native_rows != 0
+            && (!matches!(native_rows, 2 | 4 | 8)
+                || backend != GpuBackend::Cuda
+                || active < native_rows)
+        {
+            return Err(ServiceError::new(
+                400,
+                "unsupported_batch_mode",
+                "fused serving requires CUDA, rows=2/4/8 and active>=rows",
+            ));
+        }
+        preflight_cuda_toolkit(backend)?;
+        if declared.is_empty() || declared.len() > 16 {
+            return Err(ServiceError::new(
+                400,
+                "invalid_models",
+                "GPU model routes require 1..16 IDs",
+            ));
+        }
+        // CPU fallback is forbidden: unlinked or absent provider must refuse
+        // before a resident model is constructed or HTTP announces readiness.
+        let provider_cap = provider_mib.checked_mul(1024 * 1024).ok_or_else(|| {
+            ServiceError::new(400, "provider_budget", "GPU provider cap overflow")
+        })?;
+        let device_cap = (device_mib as u64) * 1024 * 1024;
+        let mut models = BTreeMap::new();
+        let mut unique = BTreeMap::<PathBuf, Arc<GpuModel>>::new();
+        for (alias, path) in declared {
+            if alias.is_empty()
+                || alias.len() > 96
+                || !alias
+                    .bytes()
+                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
+                || models.contains_key(&alias)
+            {
+                return Err(ServiceError::new(
+                    400,
+                    "invalid_model",
+                    "bad/duplicate GPU route name",
+                ));
+            }
+            let path = std::fs::canonicalize(path).map_err(|_| {
+                ServiceError::new(400, "missing_model", "GGUF source does not exist")
+            })?;
+            if let Some(resident) = unique.get(&path) {
+                models.insert(alias, Arc::clone(resident));
+                continue;
+            }
+            let parent = path.parent().unwrap_or(Path::new("."));
+            let tokenizer = QwenTokenizer::from_file(&parent.join("tokenizer.json"))
+                .map_err(ServiceError::inference)?;
+            let source = LoadedQwenModel::open(&path, &parent.join("config.json"))
+                .map_err(ServiceError::inference)?;
+            let hidden_width = usize::try_from(source.geometry().hidden_size).map_err(|_| {
+                ServiceError::new(400, "source_hidden", "invalid Qwen source hidden width")
+            })?;
+            let vocabulary_size = source.geometry().vocab_size;
+            let plan = source
+                .forward_plan(GPU_CAPACITY, true, true)
+                .map_err(ServiceError::inference)?;
+            let device = backend.device().map_err(ServiceError::inference)?;
+            let resident = block_on(ResidentModel::load(
+                device,
+                source,
+                plan,
+                provider_cap,
+                device_cap,
+            ))
+            .map_err(ServiceError::inference)?;
+            let shared = Arc::new(GpuModel {
+                resident,
+                tokenizer: Arc::new(tokenizer),
+                group: unique.len(),
+                hidden_width,
+                vocabulary_size,
+                native_rows,
+            });
+            unique.insert(path, Arc::clone(&shared));
+            models.insert(alias, shared);
+        }
+        let models = Arc::new(models);
+        let (sender, receiver) = mpsc::sync_channel(QUEUED_PER_WORKER);
+        let sources = Arc::clone(&models);
+        let stats = Arc::new(WaveStats::default());
+        let worker_stats = Arc::clone(&stats);
+        thread::Builder::new()
+            .name("zg-resident-gpu".into())
+            .spawn(move || worker(receiver, sources, active, worker_stats, native_rows))
+            .map_err(|e| ServiceError::new(500, "worker_spawn", e.to_string()))?;
+        Ok(Self {
+            models,
+            sender,
+            next_id: AtomicU64::new(1),
+            stats,
+            max_output: output,
+            native_rows,
+        })
+    }
+    pub(super) fn metrics(&self) -> GpuMetrics {
+        GpuMetrics {
+            attempted: self.stats.attempted.load(Ordering::Relaxed),
+            completed: self.stats.completed.load(Ordering::Relaxed),
+            gpu_forwards: self.stats.gpu_forwards.load(Ordering::Relaxed),
+            failed: self.stats.failed.load(Ordering::Relaxed),
+            cancelled: self.stats.cancelled.load(Ordering::Relaxed),
+            queued: self.stats.queued.load(Ordering::Acquire),
+            active: self.stats.active.load(Ordering::Acquire),
+            groups: self.stats.groups.load(Ordering::Relaxed),
+            co_scheduled: self.stats.co_scheduled.load(Ordering::Relaxed),
+            cohort_publications: self.stats.cohort_publications.load(Ordering::Relaxed),
+            cohort_rows: self.stats.cohort_rows.load(Ordering::Relaxed),
+            kernel_dispatches: self.stats.kernel_dispatches.load(Ordering::Relaxed),
+            model_tokens: self.stats.model_tokens.load(Ordering::Relaxed),
+            committed_roots: self.stats.committed_roots.load(Ordering::Relaxed),
+            fused_batches: self.stats.fused_batches.load(Ordering::Relaxed),
+            fused_state_only_batches: self.stats.fused_state_only_batches.load(Ordering::Relaxed),
+            fused_state_only_rows: self.stats.fused_state_only_rows.load(Ordering::Relaxed),
+            fused_greedy_batches: self.stats.fused_greedy_batches.load(Ordering::Relaxed),
+            native_bands: std::array::from_fn(|i| {
+                self.stats.native_bands[i].load(Ordering::Relaxed)
+            }),
+            native_admission_refusals: self.stats.native_admission_refusals.load(Ordering::Relaxed),
+            fused_rows: self.stats.fused_rows.load(Ordering::Relaxed),
+            native_projections: self.stats.native_projections.load(Ordering::Relaxed),
+            native_sampler: self.stats.native_sampler.load(Ordering::Relaxed),
+            discarded_cohorts: self.stats.discarded_cohorts.load(Ordering::Relaxed),
+            output_tokens: self.stats.output_tokens.load(Ordering::Relaxed),
+            parked: self.stats.parked.load(Ordering::Relaxed),
+            native_rows: self.native_rows,
+        }
+    }
+    pub(super) fn model_ids(&self) -> impl Iterator<Item = &str> {
+        self.models.keys().map(String::as_str)
+    }
+    pub(super) fn request(&self, request: Request) -> Result<RequestEvents, ServiceError> {
+        if !self.models.contains_key(&request.model) {
+            return Err(ServiceError::new(
+                404,
+                "model_not_found",
+                "unregistered GPU model",
+            ));
+        }
+        if request.input.is_empty()
+            || request.input.len() > 1024 * 1024
+            || request.max_tokens == 0
+            || request.max_tokens > self.max_output
+        {
+            return Err(ServiceError::new(
+                400,
+                "invalid_request",
+                "GPU request bounds",
+            ));
+        }
+        let model = request.model.clone();
+        let token_metrics = request.token_metrics.clone();
+        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
+        if id == u64::MAX {
+            return Err(ServiceError::new(
+                503,
+                "id_exhausted",
+                "request IDs exhausted",
+            ));
+        }
+        let cancelled = Arc::new(AtomicBool::new(false));
+        let (tx, rx) = mpsc::sync_channel(EVENTS_PER_REQUEST);
+        self.stats.queued.fetch_add(1, Ordering::AcqRel);
+        self.sender
+            .try_send(Scheduled {
+                request,
+                events: tx,
+                cancelled: Arc::clone(&cancelled),
+            })
+            .map_err(|err| {
+                self.stats.queued.fetch_sub(1, Ordering::AcqRel);
+                match err {
+                    mpsc::TrySendError::Full(_) => {
+                        ServiceError::new(429, "queue_full", "GPU queue at capacity")
+                    }
+                    mpsc::TrySendError::Disconnected(_) => {
+                        ServiceError::new(503, "worker_unavailable", "GPU executor unavailable")
+                    }
+                }
+            })?;
+        Ok(RequestEvents::new(id, model, rx, cancelled).observed(token_metrics))
+    }
+}
+struct GpuActive {
+    _charge: SessionCharge,
+    resident: Arc<ResidentModel>,
+    session: ResidentSession,
+    group: usize,
+    hidden_width: usize,
+    cancelled: Arc<AtomicBool>,
+    errors: mpsc::SyncSender<ResponseEvent>,
+    output: GeneratedText,
+    prompt: Vec<u32>,
+    cursor: usize,
+    device_token: Option<ResidentToken>,
+    token_commit_observed_at: Option<Instant>,
+    token_gpu_observation: Option<GpuTokenObservation>,
+    native_rows: usize,
+    packet_presented: bool,
+    blocked_since: Option<Instant>,
+}
+impl GpuActive {
+    fn new(
+        job: Scheduled,
+        model: Arc<GpuModel>,
+        stats: Arc<WaveStats>,
+    ) -> Result<Self, ServiceError> {
+        let prompt = model
+            .tokenizer
+            .prompt_tokens(&job.request.input, job.request.chat)
+            .map_err(ServiceError::inference)?;
+        if prompt.is_empty()
+            || prompt.len() > GPU_CAPACITY
+            || prompt
+                .len()
+                .checked_add(job.request.max_tokens - 1)
+                .is_none_or(|total| total > GPU_CAPACITY)
+        {
+            return Err(ServiceError::new(
+                400,
+                "context_overflow",
+                "resident GPU profile admits at most 512 consumed tokens",
+            ));
+        }
+        let output = GeneratedText::new(
+            Arc::clone(&model.tokenizer),
+            job.events.clone(),
+            job.request.stream,
+            job.request.max_tokens,
+            prompt.len(),
+        )
+        .observed(job.request.token_metrics.clone(), model.vocabulary_size);
+        // ResidentSession allocates independent, physical source-bound GPU
+        // state. Neither its tensor source nor its StateRoot is CPU fallback.
+        let session = block_on(model.resident.session()).map_err(ServiceError::inference)?;
+        stats.active.fetch_add(1, Ordering::AcqRel);
+        Ok(Self {
+            _charge: SessionCharge(stats),
+            resident: Arc::clone(&model.resident),
+            session,
+            group: model.group,
+            hidden_width: model.hidden_width,
+            cancelled: job.cancelled,
+            errors: job.events,
+            output,
+            prompt,
+            cursor: 0,
+            device_token: None,
+            token_commit_observed_at: None,
+            token_gpu_observation: None,
+            native_rows: model.native_rows,
+            packet_presented: false,
+            blocked_since: None,
+        })
+    }
+    fn ready(&self) -> Ready {
+        let remaining = self.prompt.len().saturating_sub(self.cursor);
+        Ready {
+            source: self.group,
+            phase: if remaining > 0 {
+                Phase::Prefill
+            } else {
+                Phase::Decode
+            },
+            frontier: self.session.binding().frontier() as usize,
+            tokens: if remaining > 0 {
+                self.prefill_count()
+            } else {
+                1
+            },
+            // This is finite scheduler credit, NOT a claim that this is
+            // the true physical GPU workspace footprint.
+            scratch_bytes: 32 * 1024 * 1024,
+        }
+    }
+    fn prefill_count(&self) -> usize {
+        let remaining = self.prompt.len() - self.cursor;
+        if self.native_rows != 0 {
+            // Native cohorts and unmatched B1 rows consume the same truthful
+            // one-token wave credit, then release the worker to other rows.
+            remaining.min(1)
+        } else {
+            remaining.min(PREFILL_QUANTUM)
+        }
+    }
+    /// Presentation precedes consumption. A Ready-cohort retry retains this
+    /// mark and the original packet; only a newly committed packet resets it.
+    fn present_decode(&mut self, stats: &WaveStats) -> Result<Presentation, ServiceError> {
+        let presented = if self.packet_presented {
+            self.output.resume()?
+        } else {
+            let token = self.device_token.as_ref().ok_or_else(|| {
+                ServiceError::new(
+                    500,
+                    "missing_device_token",
+                    "resident GPU feedback is missing",
+                )
+            })?;
+            let before = self.output.generated_token_count();
+            let committed = self.token_commit_observed_at.ok_or_else(|| {
+                ServiceError::new(
+                    500,
+                    "missing_commit_observation",
+                    "resident packet has no publication observation",
+                )
+            })?;
+            let status =
+                self.output
+                    .present_observed(token.id, committed, self.token_gpu_observation)?;
+            stats.output_tokens.fetch_add(
+                (self.output.generated_token_count() - before) as u64,
+                Ordering::Relaxed,
+            );
+            self.packet_presented = true;
+            status
+        };
+        if presented == Presentation::Blocked {
+            let since = self.blocked_since.get_or_insert_with(Instant::now);
+            if since.elapsed() >= OUTPUT_STALL_LIMIT {
+                return Err(ServiceError::new(
+                    503,
+                    "backpressure_timeout",
+                    "client output credit deadline expired",
+                ));
+            }
+        } else {
+            self.blocked_since = None;
+        }
+        Ok(presented)
+    }
+    fn install(
+        &mut self,
+        read: InferenceReadback,
+        commit_observed_at: Instant,
+        native_cohort: Option<NativeCohortObservation>,
+    ) -> Result<(), ServiceError> {
+        let token = read.device_token.ok_or_else(|| {
+            ServiceError::new(
+                500,
+                "missing_device_token",
+                "GPU completion has no original greedy packet",
+            )
+        })?;
+        // install follows successful original publication; this is the exact
+        // packet-producing committed binding, including a B1 retry.
+        self.token_gpu_observation = Some(GpuTokenObservation {
+            producer_frontier: self.session.binding().frontier(),
+            native_cohort,
+        });
+        self.device_token = Some(token);
+        self.token_commit_observed_at = Some(commit_observed_at);
+        self.packet_presented = false;
+        self.blocked_since = None;
+        Ok(())
+    }
+    /// Original B1 source execution, including all unmatched native-wave rows.
+    fn step(&mut self, stats: &WaveStats) -> Result<MemberStep, ServiceError> {
+        if self.cursor < self.prompt.len() {
+            let end = self.cursor + self.prefill_count();
+            let read = if end < self.prompt.len() {
+                block_on(
+                    self.session
+                        .prefill_without_logits(&self.prompt[self.cursor..end], false),
+                )
+            } else {
+                block_on(self.session.forward_greedy(&self.prompt[self.cursor..end]))
+            }
+            .map_err(ServiceError::inference)?;
+            let committed = Instant::now();
+            stats.single_completion(&read);
+            if end == self.prompt.len() {
+                self.install(read, committed, None)?;
+            }
+            self.cursor = end;
+            return Ok(MemberStep::Forwarded);
+        }
+        match self.present_decode(stats)? {
+            Presentation::Finished => return Ok(MemberStep::Finished),
+            Presentation::Blocked => return Ok(MemberStep::Parked),
+            Presentation::Ready => {}
+        }
+        if self.cancelled.load(Ordering::Acquire) {
+            return Ok(MemberStep::Cancelled);
+        }
+        let token = self.device_token.as_ref().ok_or_else(|| {
+            ServiceError::new(
+                500,
+                "missing_device_token",
+                "resident GPU feedback is missing",
+            )
+        })?;
+        let read =
+            block_on(self.session.forward_device_token(token)).map_err(ServiceError::inference)?;
+        let committed = Instant::now();
+        stats.single_completion(&read);
+        self.install(read, committed, None)?;
+        Ok(MemberStep::Forwarded)
+    }
+}
+
+enum MemberStep {
+    Forwarded,
+    Finished,
+    Parked,
+    Cancelled,
+}
+impl WaveStats {
+    fn single_completion(&self, read: &InferenceReadback) {
+        self.gpu_forwards.fetch_add(1, Ordering::Relaxed);
+        self.kernel_dispatches
+            .fetch_add(read.dispatches as u64, Ordering::Relaxed);
+        self.model_tokens.fetch_add(
+            (read.state_only_steps + read.logit_steps) as u64,
+            Ordering::Relaxed,
+        );
+        self.committed_roots.fetch_add(1, Ordering::Relaxed);
+    }
+    fn fail(&self, row: &GpuActive, error: ServiceError) {
+        if error.status == 499 && error.code == "cancelled" {
+            self.cancelled.fetch_add(1, Ordering::Relaxed);
+        } else {
+            self.failed.fetch_add(1, Ordering::Relaxed);
+        }
+        let _ = row.errors.try_send(ResponseEvent::Error(error));
+    }
+}
+fn advance_member(next: GpuActive, active: &mut VecDeque<GpuActive>, stats: &WaveStats) {
+    let Some(mut next) = eligible_member(next, active, stats) else {
+        return;
+    };
+    stats.attempted.fetch_add(1, Ordering::Relaxed);
+    match next.step(stats) {
+        Ok(MemberStep::Parked) => active.push_back(next),
+        Ok(MemberStep::Cancelled) => {
+            stats.cancelled.fetch_add(1, Ordering::Relaxed);
+        }
+        Ok(outcome) => {
+            stats.completed.fetch_add(1, Ordering::Relaxed);
+            if matches!(outcome, MemberStep::Forwarded) {
+                active.push_back(next);
+            }
+        }
+        Err(error) => {
+            stats.fail(&next, error);
+        }
+    }
+}
+/// Resume bounded output before building new GPU waves. Parked rows own
+/// no submitted command, queue slot or physical workspace guard.
+fn eligible_member(
+    mut row: GpuActive,
+    parked: &mut VecDeque<GpuActive>,
+    stats: &WaveStats,
+) -> Option<GpuActive> {
+    if row.cancelled.load(Ordering::Acquire) {
+        stats.cancelled.fetch_add(1, Ordering::Relaxed);
+        return None;
+    }
+    if row.cursor < row.prompt.len() {
+        return Some(row);
+    }
+    match row.present_decode(stats) {
+        Ok(Presentation::Ready) => Some(row),
+        Ok(Presentation::Blocked) => {
+            parked.push_back(row);
+            None
+        }
+        Ok(Presentation::Finished) => {
+            stats.attempted.fetch_add(1, Ordering::Relaxed);
+            stats.completed.fetch_add(1, Ordering::Relaxed);
+            None
+        }
+        Err(error) => {
+            stats.attempted.fetch_add(1, Ordering::Relaxed);
+            stats.fail(&row, error);
+            None
+        }
+    }
+}
+
+fn worker(
+    rx: mpsc::Receiver<Scheduled>,
+    models: Arc<BTreeMap<String, Arc<GpuModel>>>,
+    active_limit: usize,
+    stats: Arc<WaveStats>,
+    native_rows: usize,
+) {
+    let mut active = VecDeque::<GpuActive>::new();
+    loop {
+        active.retain(|row| {
+            if row.cancelled.load(Ordering::Acquire) {
+                stats.cancelled.fetch_add(1, Ordering::Relaxed);
+                false
+            } else {
+                true
+            }
+        });
+        while active.len() < active_limit {
+            let next = if active.is_empty() {
+                match rx.recv_timeout(Duration::from_millis(25)) {
+                    Ok(v) => Some(v),
+                    Err(mpsc::RecvTimeoutError::Timeout) => break,
+                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
+                }
+            } else {
+                rx.try_recv().ok()
+            };
+            let Some(job) = next else { break };
+            stats.queued.fetch_sub(1, Ordering::AcqRel);
+            if job.cancelled.load(Ordering::Acquire) {
+                stats.cancelled.fetch_add(1, Ordering::Relaxed);
+                continue;
+            }
+            let Some(model) = models.get(&job.request.model) else {
+                continue;
+            };
+            let errors = job.events.clone();
+            match GpuActive::new(job, Arc::clone(model), Arc::clone(&stats)) {
+                Ok(session) => active.push_back(session),
+                Err(e) => {
+                    stats.failed.fetch_add(1, Ordering::Relaxed);
+                    let _ = errors.try_send(ResponseEvent::Error(e));
+                }
+            }
+        }
+        if active.is_empty() {
+            continue;
+        }
+        let mut eligible = VecDeque::new();
+        let mut parked = VecDeque::new();
+        for row in active.drain(..) {
+            if let Some(row) = eligible_member(row, &mut parked, &stats) {
+                eligible.push_back(row);
+            }
+        }
+        active = parked;
+        stats.parked.store(active.len(), Ordering::Relaxed);
+        if eligible.is_empty() {
+            // No locks/queue slot while waiting for bounded transport credit.
+            if !active.is_empty() {
+                thread::sleep(Duration::from_millis(2));
+            }
+            continue;
+        }
+        let ready = eligible.iter().map(GpuActive::ready).collect::<Vec<_>>();
+        let bounds = Limits {
+            max_members: if native_rows != 0 {
+                native_rows
+            } else {
+                active_limit
+            },
+            max_tokens: 64,
+            max_scratch: 256 * 1024 * 1024,
+        };
+        let planned = if native_rows != 0 {
+            gpu_batch::form_native(&ready, bounds)
+        } else {
+            gpu_batch::form(&ready, bounds)
+        };
+        let waves = match planned {
+            Ok(groups) => groups,
+            Err(error) => {
+                for failed in eligible.drain(..) {
+                    stats.failed.fetch_add(1, Ordering::Relaxed);
+                    let _ = failed.errors.try_send(ResponseEvent::Error(error.clone()));
+                }
+                continue;
+            }
+        };
+        let mut pending = eligible.drain(..).map(Some).collect::<Vec<_>>();
+        for wave in waves {
+            // Validate source/epoch/unique committed session roots BEFORE
+            // issuing even the first physical GPU member. P12 source
+            // batch admission is shared by native bands and original B1 paths.
+            let admitted = wave
+                .indices
+                .iter()
+                .filter_map(|&i| pending[i].as_ref())
+                .filter(|row| !row.cancelled.load(Ordering::Acquire))
+                .collect::<Vec<_>>();
+            if let Some(first) = admitted.first() {
+                let roots = admitted
+                    .iter()
+                    .map(|row| row.session.binding())
+                    .collect::<Vec<_>>();
+                let verification = ResidentBatchRows::new(&roots, first.hidden_width);
+                let verification = verification.and_then(|cohort| {
+                    if admitted
+                        .iter()
+                        .any(|row| row.hidden_width != cohort.row_elements())
+                    {
+                        return Err(zerogravity::Error::State(
+                            "GPU model wave hidden widths differ".into(),
+                        ));
+                    }
+                    cohort.require_current(&roots)?;
+                    Ok(cohort)
+                });
+                if let Err(error) = verification {
+                    stats
+                        .failed
+                        .fetch_add(admitted.len() as u64, Ordering::Relaxed);
+                    let refusal = ServiceError::inference(error);
+                    for index in wave.indices {
+                        if let Some(row) = pending[index].take() {
+                            let _ = row.errors.try_send(ResponseEvent::Error(refusal.clone()));
+                        }
+                    }
+                    continue;
+                }
+            }
+            stats.groups.fetch_add(1, Ordering::Relaxed);
+            if wave.indices.len() > 1 {
+                stats
+                    .co_scheduled
+                    .fetch_add(wave.indices.len() as u64, Ordering::Relaxed);
+            }
+            // Prefix and tail readouts share the original checked publisher.
+            // The cohort owner partitions state-only and greedy barriers.
+            let batch_ready = wave.indices.len() > 1
+                && wave.indices.iter().all(|&i| {
+                    pending[i].as_ref().is_some_and(|row| {
+                        !row.cancelled.load(Ordering::Acquire)
+                            && row.cursor < row.prompt.len()
+                            && (native_rows != 0 || row.cursor + 1 == row.prompt.len())
+                            && row.device_token.is_none()
+                    })
+                });
+            if batch_ready {
+                let members = wave
+                    .indices
+                    .iter()
+                    .map(|&i| pending[i].take().expect("unique admitted wave row"))
+                    .collect::<Vec<_>>();
+                finish_prompt_cohort(members, &mut active, &stats, native_rows);
+                continue;
+            }
+            if native_rows != 0
+                && wave.indices.len() > 1
+                && wave.indices.iter().all(|&i| {
+                    pending[i]
+                        .as_ref()
+                        .is_some_and(|row| row.cursor == row.prompt.len())
+                })
+            {
+                let members = wave
+                    .indices
+                    .iter()
+                    .map(|&i| pending[i].take().expect("unique admitted decode row"))
+                    .collect();
+                advance_decode_cohort(members, &mut active, &stats, native_rows);
+                continue;
+            }
+            for index in wave.indices {
+                let next = pending[index]
+                    .take()
+                    .expect("GPU wave planner must cover unique ready slots");
+                advance_member(next, &mut active, &stats);
+            }
+        }
+        stats.parked.store(
+            active
+                .iter()
+                .filter(|row| row.blocked_since.is_some())
+                .count(),
+            Ordering::Relaxed,
+        );
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    #[test]
+    fn cuda_requires_runtime_and_cccl_headers_before_device_startup() {
+        use std::sync::atomic::{AtomicU64, Ordering};
+        static NEXT: AtomicU64 = AtomicU64::new(0);
+        let root = std::env::temp_dir().join(format!(
+            "zerogravity-cuda-prerequisite-{}-{}",
+            std::process::id(),
+            NEXT.fetch_add(1, Ordering::Relaxed)
+        ));
+        std::fs::create_dir_all(root.join("include")).unwrap();
+        assert_eq!(
+            check_cuda_toolkit(&root).unwrap_err().code,
+            "cuda_toolkit_missing"
+        );
+        std::fs::write(root.join("include/cuda_runtime.h"), b"").unwrap();
+        std::fs::write(root.join("include/cuda_fp16.h"), b"").unwrap();
+        assert_eq!(
+            check_cuda_toolkit(&root).unwrap_err().code,
+            "cuda_cccl_missing"
+        );
+        std::fs::create_dir_all(root.join("include/cccl/cuda/std")).unwrap();
+        assert!(check_cuda_toolkit(&root).is_ok());
+        std::fs::remove_dir_all(root).unwrap();
+    }
+
+    #[test]
+    fn gpu_profile_is_bounded_before_any_device_lookup() {
+        assert!(GpuRouter::validate_profile(1, 2, 512, 16, 2048, 4096).is_ok());
+        for bad in [
+            (2, 2, 512, 16, 2048, 4096),
+            (1, 0, 512, 16, 2048, 4096),
+            (1, 9, 512, 16, 2048, 4096),
+            (1, 2, 8192, 16, 2048, 4096),
+            (1, 2, 512, 513, 2048, 4096),
+            (1, 2, 512, 16, 4096, 2048),
+            (1, 2, 512, 16, 0, 4096),
+        ] {
+            assert!(GpuRouter::validate_profile(bad.0, bad.1, bad.2, bad.3, bad.4, bad.5).is_err());
+        }
+    }
+}
--- /dev/null
+++ b/.agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural-c8-counter-source/current/crates/zerogravity-cli/src/serve/mod.rs
@@ -0,0 +1,514 @@
+//! Opt-in local OpenAI-compatible HTTP serving.
+//! Uses original Qwen CPU source and private independent model sessions.
+
+mod engine;
+#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+mod gpu_batch;
+#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+mod gpu_engine;
+mod http;
+mod token_metrics;
+
+#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+use crate::GpuBackend;
+use engine::{Request, RequestEvents, ServiceError};
+
+enum Router {
+    Cpu(engine::Router),
+    #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+    Gpu(gpu_engine::GpuRouter),
+}
+impl Router {
+    fn request(&self, request: Request) -> Result<RequestEvents, ServiceError> {
+        match self {
+            Self::Cpu(v) => v.request(request),
+            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+            Self::Gpu(v) => v.request(request),
+        }
+    }
+    fn max_output(&self) -> usize {
+        match self {
+            Self::Cpu(v) => v.max_output,
+            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+            Self::Gpu(v) => v.max_output,
+        }
+    }
+    fn model_ids(&self) -> Box<dyn Iterator<Item = &str> + '_> {
+        match self {
+            Self::Cpu(v) => Box::new(v.model_ids()),
+            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+            Self::Gpu(v) => Box::new(v.model_ids()),
+        }
+    }
+    fn admission(&self) -> serde_json::Value {
+        match self {
+            Self::Cpu(v) => {
+                let (limit, reserved, remaining) = v.memory_snapshot().unwrap_or((0, 0, 0));
+                serde_json::json!({"kind":"configured_host_cpu",
+                    "host_limit_bytes":limit,"reserved_bytes":reserved,
+                    "remaining_bytes":remaining})
+            }
+            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+            Self::Gpu(v) => {
+                let m = v.metrics();
+                serde_json::json!({"kind":"resident_gpu",
+                    "physical_accounting":"ResidentService/ResourceArbiter",
+                    "batch_mode":if m.native_rows != 0 {"fused"} else {"interleaved"},
+                    "native_batch_rows_max":m.native_rows.max(1),
+                    "gpu_step_attempts":m.attempted,
+                    "completed_gpu_quanta":m.completed,
+                    "completed_quantum_unit":"request work, including terminal presentation",
+                    "gpu_forward_steps_completed":m.gpu_forwards,
+                    "gpu_forward_calls_completed":m.gpu_forwards,
+                    "completed_model_token_steps":m.model_tokens,
+                    "completed_forward_kernel_dispatches":m.kernel_dispatches,
+                    "committed_gpu_roots":m.committed_roots,
+                    "native_projection_dispatches_completed":m.native_projections,
+                    "native_sampler_dispatches_completed":m.native_sampler,
+                    "gpu_fused_rows_completed":m.fused_rows,
+                    "gpu_fused_state_only_batches":m.fused_state_only_batches,
+                    "gpu_fused_state_only_rows":m.fused_state_only_rows,
+                    "gpu_fused_greedy_batches":m.fused_greedy_batches,
+                    "gpu_fused_batches_by_rows":{"2":m.native_bands[0],"4":m.native_bands[1],"8":m.native_bands[2]},
+                    "native_admission_refusals":m.native_admission_refusals,
+                    "discarded_ready_cohorts":m.discarded_cohorts,
+                    "accepted_output_tokens":m.output_tokens,
+                    "output_credit_parked_requests":m.parked,
+                    "failed_gpu_quanta":m.failed,
+                    "cancelled_requests":m.cancelled,
+                    "queued_requests":m.queued,
+                    "active_sessions":m.active,
+                    "scheduling_groups":m.groups,
+                    "co_scheduled_members":m.co_scheduled,
+                    "state_cohort_publications":m.cohort_publications,
+                    "state_cohort_rows":m.cohort_rows,
+                    "gpu_fused_batches":m.fused_batches})
+            }
+        }
+    }
+}
+
+use serde::Deserialize;
+use std::net::{SocketAddr, TcpListener};
+use std::path::{Path, PathBuf};
+use std::sync::Arc;
+
+#[derive(Debug, Deserialize)]
+#[serde(deny_unknown_fields)]
+struct YamlConfig {
+    #[serde(default = "default_listen")]
+    listen: String,
+    models: Vec<YamlModel>,
+    #[serde(default = "default_workers")]
+    workers: usize,
+    #[serde(default = "default_active")]
+    active_per_worker: usize,
+    #[serde(default = "default_context")]
+    max_context: usize,
+    #[serde(default = "default_output")]
+    max_output: usize,
+    #[serde(default = "default_host")]
+    host_mib: usize,
+    #[serde(default = "default_backend")]
+    backend: String,
+    #[serde(default = "default_provider_mib")]
+    provider_mib: usize,
+    #[serde(default = "default_device_mib")]
+    device_mib: usize,
+    #[serde(default = "default_batch_mode")]
+    batch_mode: String,
+    #[serde(default = "default_batch_rows")]
+    batch_rows: usize,
+    #[serde(default)]
+    enable_per_request_metrics: bool,
+}
+#[derive(Debug, Deserialize)]
+#[serde(deny_unknown_fields)]
+struct YamlModel {
+    id: String,
+    path: PathBuf,
+}
+fn default_listen() -> String {
+    "127.0.0.1:8000".into()
+}
+fn default_workers() -> usize {
+    2
+}
+fn default_active() -> usize {
+    2
+}
+fn default_context() -> usize {
+    1024
+}
+fn default_output() -> usize {
+    128
+}
+fn default_host() -> usize {
+    4096
+}
+fn default_backend() -> String {
+    "cpu".into()
+}
+fn default_provider_mib() -> usize {
+    2048
+}
+fn default_device_mib() -> usize {
+    4096
+}
+fn default_batch_rows() -> usize {
+    2
+}
+fn default_batch_mode() -> String {
+    "interleaved".into()
+}
+
+impl YamlConfig {
+    fn cli(args: &[String]) -> Result<Self, ServiceError> {
+        let mut config = Self {
+            listen: default_listen(),
+            models: Vec::new(),
+            workers: default_workers(),
+            active_per_worker: default_active(),
+            max_context: default_context(),
+            max_output: default_output(),
+            host_mib: default_host(),
+            backend: default_backend(),
+            provider_mib: default_provider_mib(),
+            device_mib: default_device_mib(),
+            batch_mode: default_batch_mode(),
+            batch_rows: default_batch_rows(),
+            enable_per_request_metrics: false,
+        };
+        if args.first().is_some_and(|x| x == "--config") {
+            if args.len() != 2 {
+                return Err(ServiceError::new(
+                    400,
+                    "invalid_config",
+                    "--config must stand alone",
+                ));
+            }
+            let path = Path::new(&args[1]);
+            let raw = std::fs::read(path).map_err(|_| {
+                ServiceError::new(400, "config_unreadable", "cannot read serving YAML")
+            })?;
+            if raw.is_empty() || raw.len() > 64 * 1024 {
+                return Err(ServiceError::new(
+                    400,
+                    "config_limit",
+                    "serving YAML exceeds 64 KiB",
+                ));
+            }
+            let mut parsed: Self = serde_yaml::from_slice(&raw).map_err(|_| {
+                ServiceError::new(400, "config_invalid", "invalid typed serving YAML")
+            })?;
+            let base = path.parent().unwrap_or(Path::new("."));
+            for model in &mut parsed.models {
+                if model.path.is_relative() {
+                    model.path = base.join(&model.path);
+                }
+            }
+            return Ok(parsed);
+        }
+        let mut it = args.iter();
+        while let Some(flag) = it.next() {
+            if flag == "--enable-per-request-metrics" {
+                config.enable_per_request_metrics = true;
+                continue;
+            }
+            let value = it.next().ok_or_else(|| {
+                ServiceError::new(400, "invalid_flag", format!("{flag} requires a value"))
+            })?;
+            match flag.as_str() {
+                "--model" => {
+                    let (id, path) = value.split_once('=').ok_or_else(|| {
+                        ServiceError::new(400, "invalid_model", "--model requires ID=PATH")
+                    })?;
+                    config.models.push(YamlModel {
+                        id: id.into(),
+                        path: PathBuf::from(path),
+                    });
+                }
+                "--listen" => config.listen = value.clone(),
+                "--workers" => config.workers = parse_usize(value, flag)?,
+                "--active-per-worker" => config.active_per_worker = parse_usize(value, flag)?,
+                "--max-context" => config.max_context = parse_usize(value, flag)?,
+                "--max-output" => config.max_output = parse_usize(value, flag)?,
+                "--host-mib" => config.host_mib = parse_usize(value, flag)?,
+                "--backend" => config.backend = value.clone(),
+                "--provider-mib" => config.provider_mib = parse_usize(value, flag)?,
+                "--device-mib" => config.device_mib = parse_usize(value, flag)?,
+                "--batch-mode" => config.batch_mode = value.clone(),
+                "--batch-rows" => config.batch_rows = parse_usize(value, flag)?,
+                _ => {
+                    return Err(ServiceError::new(
+                        400,
+                        "invalid_flag",
+                        format!("unknown serve flag {flag}"),
+                    ));
+                }
+            }
+        }
+        Ok(config)
+    }
+}
+fn parse_usize(value: &str, flag: &str) -> Result<usize, ServiceError> {
+    value.parse::<usize>().map_err(|_| {
+        ServiceError::new(
+            400,
+            "invalid_integer",
+            format!("invalid integer for {flag}"),
+        )
+    })
+}
+fn validate_batch_mode(
+    mode: &str,
+    backend: &str,
+    active: usize,
+    rows: usize,
+) -> Result<(), ServiceError> {
+    match mode {
+        "interleaved" => Ok(()),
+        "fused" if backend == "cuda" && matches!(rows, 2 | 4 | 8) && active >= rows => Ok(()),
+        _ => Err(ServiceError::new(
+            400,
+            "unsupported_batch_mode",
+            "use interleaved, or fused with CUDA, rows=2/4/8 and active>=rows",
+        )),
+    }
+}
+
+pub fn run(args: &[String]) -> Result<(), ServiceError> {
+    if args.len() == 1 && args[0] == "--help" {
+        println!(
+            "USAGE: zerogravity serve --config config.yaml | \
+             --model ID=MODEL.gguf [--listen 127.0.0.1:8000] \
+             [--backend cpu|cuda|vulkan-nvidia|vulkan-amd|rocm] \
+             [--workers 2] [--active-per-worker 2] \
+             [--max-context 1024] [--max-output 128] \
+             [--host-mib 4096] [--provider-mib 2048] [--device-mib 4096] \
+             [--batch-mode interleaved|fused] [--batch-rows 2|4|8] \
+             [--enable-per-request-metrics]\n\
+             ENDPOINTS: GET /health, GET /v1/models, \
+             POST /v1/chat/completions, POST /v1/completions.\n\
+             Qwen3.5 greedy text CPU or resident GPU, no inference fallback. \
+             GPU: workers=1, context=512. Default interleaved; fused enables \
+             CUDA B2/B4/B8 compatible prompt tails and independent decode frontiers \
+             with active>=rows. Default rows=2; survivors regroup without padding. \
+             Prefill groups require equal frontiers; unmatched rows use B1. \
+             CUDA needs CUDA_PATH with runtime+CCCL headers. \
+             Opt-in metrics permit return_token_ids=true with bounded token IDs, \
+             publication-observed times and separate HTTP delivery observations. \
+             Loopback only; optional ZEROGRAVITY_API_KEY."
+        );
+        return Ok(());
+    }
+
+    let options = YamlConfig::cli(args)?;
+    let address: SocketAddr = options.listen.parse().map_err(|_| {
+        ServiceError::new(400, "invalid_listen", "--listen requires numeric IP:PORT")
+    })?;
+    let key = std::env::var("ZEROGRAVITY_API_KEY")
+        .ok()
+        .filter(|v| !v.is_empty());
+    // No native TLS; a Bearer token over unencrypted non-loopback
+    // sockets must not be mistaken for secure remote authentication.
+    // Use a trusted TLS reverse proxy forwarding to local loopback.
+    if !address.ip().is_loopback() {
+        return Err(ServiceError::new(
+            403,
+            "remote_bind_unsupported",
+            "HTTP service is loopback-only; expose through a TLS/auth reverse proxy",
+        ));
+    }
+    validate_batch_mode(
+        &options.batch_mode,
+        &options.backend,
+        options.active_per_worker,
+        options.batch_rows,
+    )?;
+    let declared = options.models.into_iter().map(|m| (m.id, m.path)).collect();
+    let backend = options.backend.as_str();
+    let router = match backend {
+        "cpu" => Router::Cpu(engine::Router::new(
+            declared,
+            options.workers,
+            options.active_per_worker,
+            options.max_context,
+            options.max_output,
+            options.host_mib,
+        )?),
+        "vulkan-nvidia" | "vulkan-amd" | "cuda" | "rocm" => {
+            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
+            {
+                let selected = match backend {
+                    "vulkan-nvidia" => GpuBackend::VulkanNvidia,
+                    "vulkan-amd" => GpuBackend::VulkanAmd,
+                    "cuda" => GpuBackend::Cuda,
+                    _ => GpuBackend::Hip,
+                };
+                if !selected.compiled() {
+                    return Err(ServiceError::new(
+                        400,
+                        "backend_not_compiled",
+                        "selected GPU backend was not compiled into this binary",
+                    ));
+                }
+                Router::Gpu(gpu_engine::GpuRouter::new(
+                    declared,
+                    selected,
+                    gpu_engine::GpuProfile {
+                        workers: options.workers,
+                        active: options.active_per_worker,
+                        context: options.max_context,
+                        output: options.max_output,
+                        provider_mib: options.provider_mib,
+                        device_mib: options.device_mib,
+                        native_rows: if options.batch_mode == "fused" {
+                            options.batch_rows
+                        } else {
+                            0
+                        },
+                    },
+                )?)
+            }
+            #[cfg(not(any(feature = "vulkan", feature = "cuda", feature = "hip")))]
+            {
+                return Err(ServiceError::new(
+                    400,
+                    "backend_not_compiled",
+                    "compile with --features serve,vulkan or serve,cuda or serve,hip",
+                ));
+            }
+        }
+        _ => {
+            return Err(ServiceError::new(
+                400,
+                "invalid_backend",
+                "backend must be cpu, vulkan-nvidia, vulkan-amd, cuda or rocm",
+            ));
+        }
+    };
+    let router = Arc::new(router);
+    let listener = TcpListener::bind(address)
+        .map_err(|e| ServiceError::new(500, "bind_failed", e.to_string()))?;
+    eprintln!(
+        "zerogravity: serving {:?} Qwen on {address}, {} worker(s); max-context {}, max-output {}",
+        backend, options.workers, options.max_context, options.max_output
+    );
+    http::serve(listener, router, key, options.enable_per_request_metrics)
+        .map_err(|e| ServiceError::new(500, "http_error", e.to_string()))
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    #[test]
+    fn native_serving_mode_is_bounded_before_model_or_device_lookup() {
+        for backend in ["cpu", "cuda", "vulkan-nvidia", "vulkan-amd", "rocm"] {
+            assert!(validate_batch_mode("interleaved", backend, 1, 2).is_ok());
+            if backend != "cuda" {
+                let err = run(&[
+                    "--backend".into(),
+                    backend.into(),
+                    "--batch-mode".into(),
+                    "fused".into(),
+                    "--model".into(),
+                    "unqualified=missing.gguf".into(),
+                ])
+                .unwrap_err();
+                assert_eq!(err.code, "unsupported_batch_mode");
+            }
+        }
+        assert!(validate_batch_mode("fused", "cuda", 2, 2).is_ok());
+        assert!(validate_batch_mode("fused", "cuda", 4, 4).is_ok());
+        assert!(validate_batch_mode("fused", "cuda", 8, 8).is_ok());
+        for rows in [0, 1, 3, 5, 6, 7, 9, 16] {
+            assert!(validate_batch_mode("fused", "cuda", 8, rows).is_err());
+        }
+        assert!(validate_batch_mode("fused", "cuda", 4, 8).is_err());
+        for mode in ["fused", "fused-b4", "unknown"] {
+            let err = run(&[
+                "--backend".into(),
+                "cuda".into(),
+                "--batch-mode".into(),
+                mode.into(),
+                "--active-per-worker".into(),
+                "1".into(),
+                "--model".into(),
+                "unqualified=missing.gguf".into(),
+            ])
+            .unwrap_err();
+            assert_eq!(err.code, "unsupported_batch_mode");
+        }
+    }
+
+    #[test]
+    fn serve_cli_is_bounded_and_configured_only_explicitly() {
+        let cfg = YamlConfig::cli(&[
+            "--model".into(),
+            "a=weights.gguf".into(),
+            "--workers".into(),
+            "3".into(),
+        ])
+        .unwrap();
+        assert_eq!(cfg.models.len(), 1);
+        assert_eq!(cfg.workers, 3);
+        assert_eq!(cfg.backend, "cpu");
+        assert!(!cfg.enable_per_request_metrics);
+        let observed = YamlConfig::cli(&[
+            "--enable-per-request-metrics".into(),
+            "--model".into(),
+            "a=weights.gguf".into(),
+        ])
+        .unwrap();
+        assert!(observed.enable_per_request_metrics);
+        assert!(YamlConfig::cli(&["--model".into(), "missing".into()]).is_err());
+        assert!(YamlConfig::cli(&["--workers".into(), "no".into()]).is_err());
+        assert!(
+            YamlConfig::cli(&[
+                "--config".into(),
+                "config.yaml".into(),
+                "--workers".into(),
+                "2".into()
+            ])
+            .is_err()
+        );
+        let example = include_str!("../../../../examples/serving/qwen-cpu.yaml");
+        let parsed: YamlConfig = serde_yaml::from_str(example).unwrap();
+        assert_eq!(parsed.listen, "127.0.0.1:8000");
+        assert_eq!(parsed.models.len(), 2);
+        assert_eq!(parsed.models[0].path, parsed.models[1].path);
+        assert_eq!((parsed.workers, parsed.active_per_worker), (2, 2));
+        assert_eq!(parsed.host_mib, 4096);
+        let gpu_config = include_str!("../../../../examples/serving/qwen-cuda.yaml");
+        let gpu: YamlConfig = serde_yaml::from_str(gpu_config).unwrap();
+        assert_eq!(gpu.backend, "cuda");
+        assert_eq!(gpu.batch_mode, "interleaved");
+        assert_eq!(gpu.max_context, 512);
+        assert_eq!((gpu.workers, gpu.active_per_worker), (1, 2));
+        assert_eq!((gpu.provider_mib, gpu.device_mib), (2048, 4096));
+        assert_eq!(gpu.models.len(), 2);
+        assert!(
+            serde_yaml::from_str::<YamlConfig>(
+                "listen: 127.0.0.1:8080\\nmodels: []\\nunknown_provider: yes"
+            )
+            .is_err()
+        );
+        let err = run(&[
+            "--listen".into(),
+            "0.0.0.0:9911".into(),
+            "--model".into(),
+            "unqualified=missing.gguf".into(),
+        ])
+        .unwrap_err();
+        assert_eq!(err.code, "remote_bind_unsupported");
+        let unsupported = run(&["--batch-mode".into(), "fused".into()]).unwrap_err();
+        assert_eq!(unsupported.code, "unsupported_batch_mode");
+        // Refuse uncompiled GPU backend before initializing a device.
+        #[cfg(not(any(feature = "vulkan", feature = "cuda", feature = "hip")))]
+        {
+            let missing = run(&["--backend".into(), "cuda".into()]).unwrap_err();
+            assert_eq!(missing.code, "backend_not_compiled");
+        }
+    }
+}

````

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 66829; SHA256 `ab3f408af3bf4917964a5d4fe365a79f02997a100861435347b9544e031e5082`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- accepted-activation-before-after/measure_same_engine.py
+++ native-prefix-before-after/measure_same_engine.py
@@ -1,5 +1,5 @@
 #!/usr/bin/env python3
-"""ROOT ONLY: bounded accepted-B7/candidate Zerogravity CUDA observations.
+"""ROOT ONLY: bounded accepted-cold-preparation/native-prefix Zerogravity CUDA observations.
 
 Both providers are Zerogravity. This never invokes serve-compare, supplies a
 vLLM slot, evicts caches, or infers source/model/performance qualification.
@@ -26,7 +26,7 @@
 ROOT = Path(__file__).resolve().parents[3]
 OWN = Path(__file__).resolve().parent
 ROOT_RUN = ROOT / ".agents/runs/serving-and-speculation-parity-root-2026-10-19"
-LABELS = {"accepted_b7_zerogravity", "candidate_zerogravity"}
+LABELS = {"accepted_d0prep_zerogravity", "native_prefix_zerogravity"}
 TOOL_PINS = {
     "serve-bench.py": "e119af67a0b3dd384c76b4d2745f8ebf6807cb27832b132ad2341580e9c5373f",
     "serve-lifecycle.py": "b3f9c76f872711edbe8eea5731f6f85c3ba119e86386b8dd50798b507fae9cd3",
@@ -38,13 +38,18 @@
            "config": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"},
 }
 TOKENIZER_SHA = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
-ACCEPTED_SHA = "d7975d1da2eb7392df947a0034b646a1e091e7604a5d5deece22a75b277820c1"
+ACCEPTED_SHA = "a896a455ec5d9093e88706751a9450170cb68b277658268ba30082f85e7ce598"
+ACCEPTED_RECEIPT_SHA = "148f990a06da8a94661304e2b89834c631fc6fc2662bd130d9e2f8dda9358ffb"
+PHASE_COUNTERS = ["gpu_fused_state_only_batches", "gpu_fused_state_only_rows", "gpu_fused_greedy_batches"]
+CURRENT_SOURCE_FIELDS = ["crates/zerogravity-cli/src/serve/gpu_engine.rs",
+    "crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs", "crates/zerogravity-cli/src/serve/mod.rs",
+    "crates/zerogravity-gpu/src/resident_inference/fused.rs"]
 COUNTERS = ["completed_forward_kernel_dispatches", "gpu_forward_calls_completed",
             "completed_model_token_steps", "committed_gpu_roots",
             "native_projection_dispatches_completed", "native_sampler_dispatches_completed",
             "accepted_output_tokens", "gpu_fused_batches", "gpu_fused_rows_completed",
             "native_admission_refusals", "discarded_ready_cohorts", "failed_gpu_quanta",
-            "cancelled_requests", "state_cohort_publications"]
+            "cancelled_requests", "state_cohort_publications", "state_cohort_rows"]
 GAUGES = ["queued_requests", "active_sessions", "output_credit_parked_requests"]
 PROMPT = "Continue this list of integers, separated by commas, for many more terms: 1, 2, 3, 4, 5, 6, 7, 8,"
 
@@ -94,7 +99,7 @@
     return json.loads(raw)
 
 
-def health(url):
+def health(url, label):
     value = get_json(url + "/health")
     require(value.get("status") == "ok" and value.get("per_request_metrics_enabled") is True,
             "actual per-request metrics not enabled")
@@ -104,17 +109,23 @@
             "actual GPU fused B8 configuration differs")
     for name in COUNTERS + GAUGES:
         require(type(admission.get(name)) is int and admission[name] >= 0, f"health field absent/invalid: {name}")
+    visibility = [field in admission for field in PHASE_COUNTERS]
+    require(all(visibility) if label == "native_prefix_zerogravity" else not any(visibility),
+            "actual phase-counter visibility differs from frozen source: legacy absence is not observed zero")
+    for field in PHASE_COUNTERS:
+        if field in admission:
+            require(type(admission[field]) is int and admission[field] >= 0, "invalid observed phase counter")
     bands = admission.get("gpu_fused_batches_by_rows")
     require(isinstance(bands, dict) and all(type(bands.get(str(width))) is int and bands[str(width)] >= 0
                                           for width in [2, 4, 8]), "actual native row-band counters absent")
     return value
 
 
-def quiescent(url):
+def quiescent(url, label):
     deadline = time.monotonic() + 10
     prior = None
     while time.monotonic() < deadline:
-        current = health(url)
+        current = health(url, label)
         admission = current["admission"]
         if admission["queued_requests"] == admission["active_sessions"] == 0:
             if current == prior:
@@ -132,6 +143,16 @@
     for name in COUNTERS:
         require(right[name] >= left[name], f"counter decreased during owned sample: {name}")
         result[name] = right[name] - left[name]
+    for field in PHASE_COUNTERS:
+        require((field in left) == (field in right), "phase-counter visibility changed during sample")
+        if field in left:
+            require(right[field] >= left[field], "phase counter decreased")
+            result[field] = right[field] - left[field]
+        else:
+            result[field] = None
+    result["phase_counter_observability"] = (
+        "observed_native_prefix_phase_counters" if all(field in left for field in PHASE_COUNTERS)
+        else "legacy_absent_not_observed")
     result["native_batches_by_rows"] = {}
     for width in ["2", "4", "8"]:
         a, b = left["gpu_fused_batches_by_rows"][width], right["gpu_fused_batches_by_rows"][width]
@@ -254,15 +275,46 @@
     return SameEnginePlan
 
 
-def workload(max_tokens):
+def workload(max_tokens, count):
     return [dict(request_id=f"request-{index}", prompt=PROMPT, arrival_seconds=0.0,
-                 tenant=f"workload-label-{index % 2}", max_tokens=max_tokens) for index in range(8)]
-
-
-def signature(result):
-    require(result["offered_requests"] == result["completed_requests"] == 8
-            and result["unsuccessful_requests"] == 0 and result["completions_without_usage"] == 0,
-            "all eight offered requests must complete with terminal usage")
+                 tenant=f"workload-label-{index % 2}", max_tokens=max_tokens) for index in range(count)]
+
+
+def settled_result(result, count, max_tokens):
+    require(all(type(result.get(field)) is int and result[field] == count for field in
+                ["offered_requests", "completed_requests", "http_accepted_requests"])
+            and all(type(result.get(field)) is int and result[field] == 0 for field in
+                ["unsuccessful_requests", "completions_without_usage", "http_429_requests", "http_503_requests",
+                 "rejected_requests", "cancelled_requests", "timed_out_requests", "client_overload_requests", "launch_deadline_requests"])
+            and result["classifications"] == {"completed": count} and result["errors"] == {}
+            and type(result["usage_reported_completion_tokens"]) is int
+            and result["usage_reported_completion_tokens"] == count * max_tokens
+            and len(result["requests"]) == count,
+            "all offers/HTTP outcomes/terminal usage accounted; no excluded or failed requests")
+    identities, providers = set(), set()
+    for row in result["requests"]:
+        identity, provider = row["request_id"], row["provider_response_id"]
+        require(identity not in identities and row["wire_request_id"] == result["run_id"] + ":" + identity
+                and isinstance(provider, str) and 0 < len(provider) <= 256 and provider not in providers,
+                "unique bound actual offered/wire/provider response identities")
+        identities.add(identity)
+        providers.add(provider)
+        require(row["completed"] is True and row["classification"] == "completed" and row["http_status"] == 200
+                and row["http_accepted"] is True and row["http_attempted"] is True and row["error"] is None
+                and row["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
+                and type(row["max_tokens"]) is int and row["max_tokens"] == max_tokens
+                and type(row["usage_reported_completion_tokens"]) is int
+                and row["usage_reported_completion_tokens"] == max_tokens and row["finish_reason"] == "length",
+                "same actual prompt and complete HTTP terminal/usage; EOS-shortened samples refuse")
+    require(identities == {f"request-{i}" for i in range(count)}, "every offered identity retained")
+
+
+def signature(result, count):
+    settled_result(result, count, 8)
+    metrics = result["wire_token_metrics"]
+    require(metrics["requested"] is True and metrics["completed_requests_with_individual_token_delivery"] == count
+            and metrics["completed_requests_with_provider_commit_observations"] == count,
+            "all offers have individually observed IDs and distinct provider commit observations")
     rows = {}
     for row in result["requests"]:
         observed = row.get("wire_token_observation") or {}
@@ -270,16 +322,147 @@
         require(observed.get("status") == "provider_bound_complete_ids"
                 and observed.get("client_delivery_status") == "individual_token_ids_observed"
                 and observed.get("provider_commit_observation_status") == "provider_bound_complete_commit_observations"
-                and isinstance(ids, list) and 2 <= len(ids) <= row["max_tokens"]
+                and isinstance(ids, list) and len(ids) == row["max_tokens"] == 8
+                and all(type(token) is int and 0 <= token < 248320 for token in ids)
                 and len(ids) == row["usage_reported_completion_tokens"]
-                and isinstance(row["output_sha256"], str), "actual bound individual token IDs/ITL/commit evidence required")
+                and isinstance(row["output_sha256"], str) and re.fullmatch(r"[a-f0-9]{64}", row["output_sha256"]),
+                "complete actual IDs/delivery/commit/usage required")
+        records = observed["provider_token_records"]
+        require(len(records) == 8 and all(isinstance(token.get("gpu"), dict) for token in records),
+                "all exact GPU packet-producing committed frontiers required")
+        frontiers = [token["gpu"]["producer_frontier"] for token in records]
+        require(type(frontiers[0]) is int and 1 < frontiers[0] <= 504
+                and all(type(frontier) is int for frontier in frontiers)
+                and frontiers == list(range(frontiers[0], frontiers[0]+8)), "actual consecutive producer roots")
         identity = row["request_id"]
         require(identity not in rows, "duplicate measured request identity")
         rows[identity] = dict(prompt_sha256=row["prompt_sha256"], max_tokens=row["max_tokens"],
             token_ids=ids, usage_completion_tokens=row["usage_reported_completion_tokens"],
-            output_sha256=row["output_sha256"], finish_reason=row["finish_reason"])
-    require(set(rows) == {f"request-{i}" for i in range(8)}, "actual offered identities")
+            producer_frontiers=frontiers, output_sha256=row["output_sha256"], finish_reason=row["finish_reason"])
+    require(set(rows) == {f"request-{i}" for i in range(count)}, "all offered identities retained")
+    require(len({tuple(row["producer_frontiers"]) for row in rows.values()}) == 1, "same actual teacher length")
+    require(all(row == next(iter(rows.values())) for row in rows.values()),
+            "same-prompt greedy IDs/text/frontiers/usage/finish agree across all offered requests")
     return rows
+
+
+def output_cohorts(result, count):
+    cohorts = {}
+    for row in result["requests"]:
+        for index, token in enumerate(row["wire_token_observation"]["provider_token_records"]):
+            cohort = token["gpu"].get("native_cohort")
+            if count == 1:
+                require(cohort is None, "B1 packet must not be relabeled a native multirow cohort")
+                continue
+            require(isinstance(cohort, dict) and type(cohort["width"]) is int and cohort["width"] == 8,
+                    "actual measured B8 greedy packet required")
+            sequence = cohort["publication_sequence"]
+            require(type(sequence) is int and 0 < sequence < 2**64, "actual bounded native publication sequence")
+            by_request = cohorts.setdefault(sequence, {})
+            require(row["request_id"] not in by_request, "duplicate packet in one cohort")
+            by_request[row["request_id"]] = index
+    if count == 8:
+        require(len(cohorts) == 8 and all(set(members) == {f"request-{i}" for i in range(8)}
+                and len(set(members.values())) == 1 for members in cohorts.values()),
+                "eight actual width8 greedy cohorts across64 accepted output IDs")
+    return {"observed_greedy_cohorts": len(cohorts), "cohort_members": cohorts,
+            "scope": "actual post-publication provider packet observations, not an independent math/state proof"}
+
+
+def counters(result, label, count):
+    require(label in LABELS and count in [1, 8], "closed same-engine source/workload scope")
+    outputs = signature(result, count)
+    observed = output_cohorts(result, count)
+    delta = result["same_engine_quiescent_health"]["delta"]
+    require(all(type(delta.get(field)) is int and delta[field] >= 0 for field in COUNTERS)
+            and set(delta["native_batches_by_rows"]) == {"2", "4", "8"}
+            and all(type(number) is int and number >= 0 for number in delta["native_batches_by_rows"].values()),
+            "actual finite nonnegative completed-work counters")
+    if label == "native_prefix_zerogravity":
+        require(delta["phase_counter_observability"] == "observed_native_prefix_phase_counters"
+                and all(type(delta.get(field)) is int and delta[field] >= 0 for field in PHASE_COUNTERS),
+                "new phase counters observed, not inferred")
+    else:
+        require(delta["phase_counter_observability"] == "legacy_absent_not_observed"
+                and all(delta.get(field) is None for field in PHASE_COUNTERS), "legacy phase absence remains unobserved")
+    require(all(delta[field] == 0 for field in ["failed_gpu_quanta", "cancelled_requests",
+            "native_admission_refusals", "discarded_ready_cohorts"]), "failed/refused/discarded work in timing sample")
+    p = next(iter(outputs.values()))["producer_frontiers"][0]
+    model_steps = count * (p+7)
+    require(delta["accepted_output_tokens"] == 8*count and delta["completed_model_token_steps"] == model_steps
+            and delta["completed_forward_kernel_dispatches"] > 0, "actual committed output/model token accounting")
+    bands = delta["native_batches_by_rows"]
+    fused = sum(bands.values())
+    fused_rows = sum(int(width)*number for width,number in bands.items())
+    require(fused == delta["gpu_fused_batches"] and fused_rows == delta["gpu_fused_rows_completed"]
+            and delta["state_cohort_publications"] == fused and delta["state_cohort_rows"] == fused_rows,
+            "all observed native band/batch/atomic-publication counters close")
+    require(delta["completed_forward_kernel_dispatches"] >= delta["native_projection_dispatches_completed"]
+            + delta["native_sampler_dispatches_completed"], "actual full kernel count covers both disjoint subsets")
+    if count == 8:
+        require(delta["native_projection_dispatches_completed"] > 0, "actual native projection receipts required")
+    if count == 1:
+        require(fused == fused_rows == observed["observed_greedy_cohorts"] == delta["native_sampler_dispatches_completed"]
+                == delta["native_projection_dispatches_completed"] == 0, "B1 has no multirow-native receipts")
+        prefix_passes = p-1 if label == "native_prefix_zerogravity" else (p-1+31)//32
+        expected_calls = prefix_passes + 8
+        require(delta["gpu_forward_calls_completed"] == delta["committed_gpu_roots"] == expected_calls,
+                "actual one-teacher versus max32 teacher API/publication scheduling receipts")
+        if label == "native_prefix_zerogravity":
+            require(all(delta[field] == 0 for field in PHASE_COUNTERS), "observed B1 native phase counters")
+        observed["B1_teacher_scheduling"] = dict(actual_teacher_frontier=p,
+            declared_policy="one actual teacher token per scheduled pass" if label == "native_prefix_zerogravity" else "legacy max32 teacher tokens per scheduled pass, last prompt tail separate",
+            observed_forward_API_calls=delta["gpu_forward_calls_completed"],
+            observed_committed_root_publications=delta["committed_gpu_roots"],
+            expected_prefix_API_calls=prefix_passes, completed_model_token_steps=model_steps,
+            physical_forward_kernel_dispatches=delta["completed_forward_kernel_dispatches"],
+            scope="single actual request in fused-enabled server; original B1 GPU kernels, no B>1 kernel claim")
+    elif label == "native_prefix_zerogravity":
+        require(delta["gpu_fused_greedy_batches"] == observed["observed_greedy_cohorts"] == 8
+                and delta["gpu_fused_state_only_batches"] > 0
+                and delta["gpu_fused_state_only_rows"] > 0
+                and delta["gpu_fused_state_only_batches"] + 8 == fused
+                and delta["gpu_fused_state_only_rows"] + 64 == fused_rows
+                and bands["8"] >= 8
+                and delta["native_sampler_dispatches_completed"] == 4*delta["gpu_fused_greedy_batches"]
+                and delta["committed_gpu_roots"] == model_steps
+                and delta["gpu_forward_calls_completed"] == fused + model_steps-fused_rows,
+                "actual native-prefix/greedy/sampler/B1 remainder accounting")
+        observed["derived_prefix_native_bands"] = {"2": bands["2"], "4": bands["4"], "8": bands["8"]-8}
+    else:
+        require(delta["phase_counter_observability"] == "legacy_absent_not_observed"
+                and all(delta[field] is None for field in PHASE_COUNTERS)
+                and bands == {"2": 0, "4": 0, "8": 8}
+                and fused == 8 and fused_rows == 64
+                and delta["native_sampler_dispatches_completed"] == 4*observed["observed_greedy_cohorts"]
+                and delta["gpu_forward_calls_completed"] == 8+8*((p-1+31)//32)
+                and delta["committed_gpu_roots"] == 64+8*((p-1+31)//32),
+                "legacy actual eight greedy cohorts plus source-bound max32 prefix API accounting")
+    require(delta["native_sampler_dispatches_completed"] == 4*observed["observed_greedy_cohorts"],
+            "native sampler4-kernel subset closes against actual greedy output cohorts")
+    return outputs, observed
+
+
+def build_receipt(path, expected_sha, binary, current=False):
+    require(sha(path) == expected_sha, "exact reviewed build receipt SHA")
+    receipt = json_read(path)
+    artifact = receipt["artifacts"]["server"]
+    frozen = Path(artifact["frozen_path"])
+    frozen = frozen if frozen.is_absolute() else ROOT / frozen
+    require(frozen.resolve() == binary and artifact["sha256"] == sha(binary)
+            and artifact["bytes"] == binary.stat().st_size, "actual frozen build artifact binding")
+    require(re.fullmatch(r"[a-f0-9]{40}", receipt["source_head"]) is not None
+            and isinstance(receipt.get("source_hashes"), dict) and receipt["source_hashes"]
+            and all(isinstance(name,str) and not Path(name).is_absolute() and ".." not in Path(name).parts
+                and re.fullmatch(r"[a-f0-9]{64}", value) is not None
+                for name,value in receipt["source_hashes"].items()), "actual source head/overlay hash attestation")
+    if current:
+        require(set(CURRENT_SOURCE_FIELDS) <= set(receipt["source_hashes"]), "current prefix owner source hashes required")
+    else:
+        require(receipt["source_head"] == "c5eb1f8e717c79146937e585a145a0052eff747b"
+                and artifact["sha256"] == ACCEPTED_SHA, "accepted cold-preparation compiled source/overlay receipt")
+    return {"receipt_sha256": expected_sha, "receipt": receipt,
+            "scope": "reviewed source/build attestation; not an independent numerical or reproducible-build proof"}
 
 
 def main():
@@ -288,13 +471,16 @@
     parser.add_argument("--current-server", type=Path, required=True)
     parser.add_argument("--current-sha256", required=True)
     parser.add_argument("--current-source-id", required=True, help="root actual source/build receipt identity, never inferred from binary name")
+    parser.add_argument("--current-receipt", type=Path, required=True)
+    parser.add_argument("--current-receipt-sha256", required=True)
+    parser.add_argument("--workloads", default="b8,b1", help="bounded b8 and/or single actual request b1")
     parser.add_argument("--gpu-uuid", required=True)
-    parser.add_argument("--accepted-server", type=Path, default=ROOT_RUN / "zerogravity-b7-cuda-serve")
-    parser.add_argument("--accepted-receipt", type=Path, default=ROOT_RUN / "b7-frozen-server.json")
+    parser.add_argument("--accepted-server", type=Path, default=ROOT_RUN / "activation-cold-final-server")
+    parser.add_argument("--accepted-receipt", type=Path, default=ROOT_RUN / "activation-cold-final-serving-artifacts.json")
     parser.add_argument("--models", default="0.8b,2b")
     parser.add_argument("--order", choices=["ABBA", "BAAB"], default="ABBA")
     parser.add_argument("--policy", choices=["cold_process", "warmed_process"], default="warmed_process")
-    parser.add_argument("--max-tokens", type=int, choices=[8, 16], default=8)
+    parser.add_argument("--max-tokens", type=int, choices=[8], default=8)
     parser.add_argument("--timeout", type=float, default=90)
     parser.add_argument("--ttft-slo", type=float, default=10)
     parser.add_argument("--itl-slo", type=float, default=2)
@@ -309,9 +495,13 @@
     require(re.fullmatch(r"[A-Za-z0-9_.:/+-]{1,256}", args.current_source_id) is not None, "bounded actual source receipt ID")
     require(re.fullmatch(r"GPU-[a-fA-F0-9-]{16,64}", args.gpu_uuid) is not None, "exact observed GPU UUID required")
     require(math.isfinite(args.timeout) and 1 <= args.timeout <= 180 and all(math.isfinite(value)
-            and .001 <= value <= 180 for value in [args.ttft_slo, args.itl_slo]), "finite measurement/SLO limits")
+            and .001 <= value <= ceiling for value,ceiling in [(args.ttft_slo,10), (args.itl_slo,2)]), "finite predeclared SLO limits; no widening")
     models = args.models.split(",")
     require(1 <= len(models) <= 2 and len(set(models)) == len(models) and all(name in MODELS for name in models), "approved source models only")
+    workloads = args.workloads.split(",")
+    require(1 <= len(workloads) <= 2 and len(set(workloads)) == len(workloads)
+            and all(name in {"b8", "b1"} for name in workloads), "explicit bounded workload cases")
+    require(re.fullmatch(r"[a-f0-9]{64}", args.current_receipt_sha256) is not None, "pinned current build receipt SHA")
     ports = [int(value) for value in args.ports.split(",")]
     require(len(ports) == 2 and len(set(ports)) == 2 and all(1024 <= value <= 65535 for value in ports), "two bounded loopback ports")
     allowed = os.sched_getaffinity(0)
@@ -320,9 +510,8 @@
     bench, lifecycle = module("serve-bench.py"), module("serve-lifecycle.py")
     accepted = args.accepted_server.resolve(strict=True)
     current = args.current_server.resolve(strict=True)
-    receipt = json_read(args.accepted_receipt)
-    require(receipt.get("sha256") == ACCEPTED_SHA and receipt.get("source_checkpoint") == "dbb5f4f",
-            "accepted B7 source receipt changed")
+    accepted_build = build_receipt(args.accepted_receipt, ACCEPTED_RECEIPT_SHA, accepted)
+    current_build = build_receipt(args.current_receipt, args.current_receipt_sha256, current, True)
     require(accepted != current and sha(accepted) == ACCEPTED_SHA and sha(current) == args.current_sha256,
             "actual separately frozen accepted/current server hashes")
     root_models = {}
@@ -337,20 +526,33 @@
            "LD_LIBRARY_PATH": "/nix/store/cs50fahgnln1k5cx7zs0dnyikzqzf4iq-cuda12.9-cuda_nvrtc-12.9.86-lib/lib:/run/opengl-driver/lib",
            "RAYON_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
     require(Path(env["CUDA_PATH"]).is_dir(), "approved existing CUDA view missing")
+    args.output_parent = args.output_parent.resolve()
+    require(OWN in args.output_parent.parents, "all new run artifacts must stay in this owned ignored directory")
     args.output_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
     directory = Path(tempfile.mkdtemp(prefix="same-zg-", dir=args.output_parent))
-    report = dict(schema="zerogravity.same-engine.before-after.v1", status="RUNNING",
-        scope="accepted B7 Zerogravity versus candidate Zerogravity only; never a vLLM comparison or source/model qualification",
-        providers={"A": "accepted_b7_zerogravity", "B": "candidate_zerogravity"},
-        binary_sha256={"accepted_b7_zerogravity": ACCEPTED_SHA, "candidate_zerogravity": args.current_sha256},
-        source_ids={"accepted_b7_zerogravity": "dbb5f4f", "candidate_zerogravity": args.current_source_id},
-        order=args.order, process_policy=args.policy, cpus=cpus, samples=[],
+    report = dict(schema="zerogravity.same-engine.native-prefix.before-after.v1", status="RUNNING",
+        scope="accepted cold-preparation Zerogravity versus candidate Zerogravity only; never a vLLM comparison or source/model qualification",
+        providers={"A": "accepted_d0prep_zerogravity", "B": "native_prefix_zerogravity"},
+        binary_sha256={"accepted_d0prep_zerogravity": ACCEPTED_SHA, "native_prefix_zerogravity": args.current_sha256},
+        source_ids={"accepted_d0prep_zerogravity": {"compiled_head": accepted_build["receipt"]["source_head"],
+            "reviewed_build_receipt_sha256": ACCEPTED_RECEIPT_SHA, "landed_reference": "d0a6084",
+            "overlay_sha256": accepted_build["receipt"]["source_hashes"]},
+            "native_prefix_zerogravity": {"caller_receipt_id": args.current_source_id,
+                "compiled_head": current_build["receipt"]["source_head"],
+                "reviewed_build_receipt_sha256": args.current_receipt_sha256,
+                "overlay_sha256": current_build["receipt"]["source_hashes"]}},
+        actual_build_receipts={"accepted_d0prep_zerogravity": accepted_build, "native_prefix_zerogravity": current_build},
+        order=args.order, process_policy=args.policy, workloads=workloads, cpus=cpus, samples=[],
         tools=TOOL_PINS, measurement_driver_sha256=sha(Path(__file__)),
         source_pins={name: MODELS[name] for name in models}, tokenizer_sha256=TOKENIZER_SHA,
         temperature=0, top_p=1, n=1, max_tokens=args.max_tokens, context_capacity=512, active_rows_max=8,
         logical_caps_mib={"host": 4096, "provider": 4096, "device": 5120},
         OS_storage_device_caches_evicted=False, physical_RAM_admission_qualified=False,
-        kernel_counter_scope="successful source forward API receipts; initialization/DMA/unknown-completion excluded; native projection counter is final projection-dot subset",
+        kernel_counter_scope="actual completed source-forward kernel receipts, including native prefix+greedy bodies; initialization/DMA/unknown-completion excluded; projection counter is projection-dot subset; sampler counter is 4-kernel B>1 greedy subset, not all B1 samplers",
+        counter_visibility={"accepted_d0prep_zerogravity": "legacy phase fields absent: unobserved, never zero",
+                            "native_prefix_zerogravity": "all three phase fields required observed"},
+        B1_policy="one actual request, same prompt/caps/fused-enabled configuration: old max32 teacher API chunks versus new one-teacher scheduled passes; original B1 GPU math",
+        predeclared_SLO_limits={"TTFT_seconds":args.ttft_slo,"actual_single_ID_delivery_ITL_seconds":args.itl_slo},
         memory_scope="nvidia-smi and proc observations, not ResourceArbiter reservation telemetry; provider health does not currently expose GPU budget bytes",
         http_transport="private loopback; proxies disabled and redirects refused for benchmark and health requests",
         started_utc=datetime.now(timezone.utc).isoformat())
@@ -361,93 +563,129 @@
         idle = gpu_idle(args.gpu_uuid)
         report["initial_GPU_idle_observations"] = idle
         idle_ceiling = idle[-1]["memory_used_mib"] + 64
+        canonical_by_model = {}
         for name in models:
-            alias = "qwen35-" + name + "-same-source"
-            endpoints = dict(zip(["accepted_b7_zerogravity", "candidate_zerogravity"],
-                                [f"http://127.0.0.1:{port}" for port in ports]))
-            providers = {}
-            for label, binary in [("accepted_b7_zerogravity", accepted), ("candidate_zerogravity", current)]:
-                command = [str(binary), "serve", "--model", alias + "=" + str(root_models[name] / "model.gguf"),
-                    "--backend", "cuda", "--workers", "1", "--active-per-worker", "8", "--max-context", "512",
-                    "--max-output", str(args.max_tokens), "--host-mib", "4096", "--provider-mib", "4096", "--device-mib", "5120",
-                    "--batch-mode", "fused", "--batch-rows", "8", "--enable-per-request-metrics", "--listen",
-                    "127.0.0.1:" + str(ports[0 if label == "accepted_b7_zerogravity" else 1])]
-                providers[label] = dict(argv=command, cwd=str(ROOT), env=env,
-                    declarations=dict(runtime="Zerogravity " + label + " binarySHA " + report["binary_sha256"][label],
-                        source_sha256=MODELS[name]["source"], tokenizer_sha256=TOKENIZER_SHA,
-                        numerical_profile="accepted original GGUF source kernels; actual measured outputs checked separately",
-                        quantization="unchanged original mixed GGUF source bytes", generation_policy="greedy temperature0 top_p1 n1; source EOS; finite output cap",
-                        hardware="caller-pinned NVIDIA UUID " + args.gpu_uuid,
-                        resource_limits=dict(context=512, rows=8, host_mib=4096, provider_mib=4096, device_mib=5120,
-                            logical_CPU_affinity=cpus, Rayon_threads=4, OS_CPU_or_physical_RAM_quota_claimed=False)))
-            plan = plan_type(lifecycle)(providers, endpoints, args.policy, directory / ("owned-" + name))
-            reference = None
-            for sequence, letter in enumerate(args.order):
-                label = report["providers"][letter]
-                url, binary = endpoints[label], accepted if letter == "A" else current
-                require(sha(binary) == report["binary_sha256"][label], "frozen provider changed before spawn")
-                before_idle = gpu_idle(args.gpu_uuid, idle_ceiling)
-                sampler = Sampler(args.gpu_uuid)
-                sample = None
-                try:
-                    def ready():
-                        require({item["id"] for item in get_json(url + "/v1/models")["data"]} == {alias}, "exact ready model alias")
-                        health(url)
-                    with plan.sample(label, ready) as owned:
-                        sampler.pid = owned.process.pid
-                        require(Path(os.readlink(f"/proc/{owned.process.pid}/exe")).resolve() == binary,
-                                "owned foreground executable differs from frozen binary")
-                        if args.policy == "warmed_process":
-                            warm = bench.bench(url, alias, PROMPT, 8, 8, 1, args.timeout, None,
-                                workload=workload(1), snapshot_url=url + "/health", return_token_ids=True,
-                                max_pending=8, run_id=f"same-zg-warm-{name}-{sequence}")
-                            owned.record_warmup(warm)
-                            require(warm["offered_requests"] == warm["completed_requests"] == 8,
-                                    "bounded warmup incomplete")
-                        before = quiescent(url)
-                        measured = bench.bench(url, alias, PROMPT, 8, 8, args.max_tokens, args.timeout, None,
-                            workload=workload(args.max_tokens), ttft_slo_seconds=args.ttft_slo,
-                            itl_slo_seconds=args.itl_slo, snapshot_url=url + "/health", return_token_ids=True,
-                            max_pending=8, run_id=f"same-zg-{name}-{sequence}")
-                        after = quiescent(url)
-                        measured["same_engine_quiescent_health"] = dict(before=before, after=after,
-                            delta=deltas(before, after), gauges=gauge_observations(before, after))
-                        owned.record_measurement(measured)
-                        sample = dict(model=name, sequence=sequence, provider=label,
-                            measurement=measured, lifecycle_receipt_path=str(owned.receipt_path),
-                            GPU_idle_before_spawn=before_idle)
-                finally:
-                    memory = sampler.finish()
-                # Every owner has exited and its original cleanup has completed
-                # before any post-measurement validation or receipt write.
-                require(sample is not None and not plan.blocked and plan.current is None, "owned lifecycle did not finish")
-                sample["memory_observations"] = memory
-                sample["lifecycle_receipt"] = json_read(sample["lifecycle_receipt_path"])
-                owned_pid = sample["lifecycle_receipt"]["pid"]
-                owned_gpu_observed = False
-                for observed in memory["samples"]:
-                    for application in observed["compute_apps_raw"].splitlines():
-                        process = application.split(",", 1)[0].strip()
-                        require(process.isdecimal() and int(process) == owned_pid,
-                                "another GPU compute process appeared during owned sample; preserve it and stop")
-                        owned_gpu_observed = True
-                require(owned_gpu_observed, "selected GPU owned compute process was not physically observed")
-                sample["GPU_idle_after_owned_cleanup"] = gpu_idle(args.gpu_uuid, idle_ceiling)
-                require(sha(binary) == report["binary_sha256"][label], "frozen provider changed during sample")
-                report["samples"].append(sample)
-                write()
-                outputs = signature(sample["measurement"])
-                delta = sample["measurement"]["same_engine_quiescent_health"]["delta"]
-                require(delta["completed_forward_kernel_dispatches"] > 0
-                        and delta["completed_model_token_steps"] > 0 and delta["native_batches_by_rows"]["8"] > 0,
-                        "actual measured native B8 forward dispatch evidence absent")
-                require(all(delta[field] == 0 for field in ["failed_gpu_quanta", "cancelled_requests",
-                        "native_admission_refusals", "discarded_ready_cohorts"]), "failed/refused/discarded work in performance sample")
-                if reference is None:
-                    reference = outputs
-                require(outputs == reference, "actual per-request token IDs/usage/output hashes/finish reason changed; no further sample")
+            for workload_name in workloads:
+                count = 8 if workload_name == "b8" else 1
+                alias = "qwen35-" + name + "-same-source"
+                endpoints = dict(zip(["accepted_d0prep_zerogravity", "native_prefix_zerogravity"],
+                                    [f"http://127.0.0.1:{port}" for port in ports]))
+                providers = {}
+                for label, binary in [("accepted_d0prep_zerogravity", accepted), ("native_prefix_zerogravity", current)]:
+                    command = [str(binary), "serve", "--model", alias + "=" + str(root_models[name] / "model.gguf"),
+                        "--backend", "cuda", "--workers", "1", "--active-per-worker", "8", "--max-context", "512",
+                        "--max-output", "8", "--host-mib", "4096", "--provider-mib", "4096", "--device-mib", "5120",
+                        "--batch-mode", "fused", "--batch-rows", "8", "--enable-per-request-metrics", "--listen",
+                        "127.0.0.1:" + str(ports[0 if label == "accepted_d0prep_zerogravity" else 1])]
+                    providers[label] = dict(argv=command, cwd=str(ROOT), env=env,
+                        declarations=dict(runtime="Zerogravity " + label + " binarySHA " + report["binary_sha256"][label],
+                            source_sha256=MODELS[name]["source"], tokenizer_sha256=TOKENIZER_SHA,
+                            numerical_profile="accepted original GGUF source kernels; measured token/text/usage checked, independent mathematical acceptance remains separate",
+                            quantization="unchanged original mixed GGUF source bytes", generation_policy="greedy temperature0 top_p1 n1; same source EOS; finite max8",
+                            hardware="caller-pinned NVIDIA UUID " + args.gpu_uuid,
+                            resource_limits=dict(context=512, configured_rows=8, offered_requests=count,
+                                host_mib=4096, provider_mib=4096, device_mib=5120,
+                                logical_CPU_affinity=cpus, Rayon_threads=4, OS_CPU_or_physical_RAM_quota_claimed=False)))
+                plan = plan_type(lifecycle)(providers, endpoints, args.policy, directory / ("owned-" + name + "-" + workload_name))
+                reference = None
+                for sequence, letter in enumerate(args.order):
+                    label = report["providers"][letter]
+                    url, binary = endpoints[label], accepted if letter == "A" else current
+                    require(sha(binary) == report["binary_sha256"][label], "frozen provider changed before spawn")
+                    require(sha(root_models[name] / "model.gguf") == MODELS[name]["source"]
+                            and sha(root_models[name] / "config.json") == MODELS[name]["config"]
+                            and sha(root_models[name] / "tokenizer.json") == TOKENIZER_SHA,
+                            "source/config/tokenizer changed before owned provider")
+                    before_idle = gpu_idle(args.gpu_uuid, idle_ceiling)
+                    sampler = Sampler(args.gpu_uuid)
+                    sample = None
+                    try:
+                        def ready():
+                            require({item["id"] for item in get_json(url + "/v1/models")["data"]} == {alias}, "exact ready model alias")
+                            health(url, label)
+                        with plan.sample(label, ready) as owned:
+                            sampler.pid = owned.process.pid
+                            require(Path(os.readlink(f"/proc/{owned.process.pid}/exe")).resolve() == binary,
+                                    "owned foreground executable differs from frozen binary")
+                            if args.policy == "warmed_process":
+                                warm = bench.bench(url, alias, PROMPT, count, count, 1, args.timeout, None,
+                                    workload=workload(1, count), snapshot_url=url + "/health", return_token_ids=True,
+                                    max_pending=count, run_id=f"prefix-zg-warm-{name}-{workload_name}-{sequence}")
+                                owned.record_warmup(warm)
+                                settled_result(warm, count, 1)
+                            before = quiescent(url, label)
+                            measured = bench.bench(url, alias, PROMPT, count, count, 8, args.timeout, None,
+                                workload=workload(8, count), ttft_slo_seconds=args.ttft_slo,
+                                itl_slo_seconds=args.itl_slo, snapshot_url=url + "/health", return_token_ids=True,
+                                max_pending=count, run_id=f"prefix-zg-{name}-{workload_name}-{sequence}")
+                            after = quiescent(url, label)
+                            measured["same_engine_quiescent_health"] = dict(before=before, after=after,
+                                delta=deltas(before, after), gauges=gauge_observations(before, after))
+                            owned.record_measurement(measured)
+                            sample = dict(model=name, workload=workload_name, offered_count=count,
+                                sequence=sequence, provider=label, measurement=measured,
+                                lifecycle_receipt_path=str(owned.receipt_path), GPU_idle_before_spawn=before_idle)
+                    finally:
+                        memory = sampler.finish()
+                    # Original owner closes and proves cleanup before any receipt validation/next provider.
+                    require(sample is not None and not plan.blocked and plan.current is None, "owned lifecycle did not finish")
+                    sample["memory_observations"] = memory
+                    sample["lifecycle_receipt"] = json_read(sample["lifecycle_receipt_path"])
+                    owned_pid = sample["lifecycle_receipt"]["pid"]
+                    owned_gpu_observed = False
+                    for observed_sample in memory["samples"]:
+                        for application in observed_sample["compute_apps_raw"].splitlines():
+                            process = application.split(",", 1)[0].strip()
+                            require(process.isdecimal() and int(process) == owned_pid,
+                                    "another GPU compute process appeared; preserve it and stop")
+                            owned_gpu_observed = True
+                    require(owned_gpu_observed, "selected GPU owned compute process was not physically observed")
+                    sample["GPU_idle_after_owned_cleanup"] = gpu_idle(args.gpu_uuid, idle_ceiling)
+                    require(sha(binary) == report["binary_sha256"][label], "frozen provider changed during sample")
+                    require(sha(root_models[name] / "model.gguf") == MODELS[name]["source"]
+                            and sha(root_models[name] / "config.json") == MODELS[name]["config"]
+                            and sha(root_models[name] / "tokenizer.json") == TOKENIZER_SHA,
+                            "source/config/tokenizer changed during owned provider")
+                    require(sha(args.accepted_receipt) == ACCEPTED_RECEIPT_SHA
+                            and sha(args.current_receipt) == args.current_receipt_sha256,
+                            "reviewed build receipt changed during comparison")
+                    report["samples"].append(sample)
+                    write()
+                    outputs, closed = counters(sample["measurement"], label, count)
+                    sample["actual_output_and_counter_closure"] = closed
+                    canonical = next(iter(outputs.values()))
+                    if name not in canonical_by_model:
+                        canonical_by_model[name] = canonical
+                    require(canonical == canonical_by_model[name],
+                            "same model/prompt B1 and B8 exact IDs/text/frontiers/usage/finish differ")
+                    if reference is None:
+                        reference = outputs
+                    require(outputs == reference, "actual offered token IDs/frontiers/text/usage/finish changed; no next sample")
+                    sample["same_source_outputs_equal"] = True
+                    write()
+        report["comparison_observations"] = []
+        for model_name in models:
+            for workload_name in workloads:
+                selected = [sample for sample in report["samples"]
+                            if sample["model"] == model_name and sample["workload"] == workload_name]
+                require([sample["provider"] for sample in selected] == [report["providers"][letter] for letter in args.order],
+                        "actual serialized sample order equals predeclared ABBA/BAAB")
+                report["comparison_observations"].append({"model":model_name,"workload":workload_name,
+                    "actual_order":[sample["provider"] for sample in selected],
+                    "samples":[{"provider":sample["provider"],"sequence":sample["sequence"],
+                        "wall_seconds":sample["measurement"]["wall_seconds"],
+                        "text_TTFT_p50_seconds":sample["measurement"]["ttft_p50_seconds"],
+                        "text_TTFT_p95_seconds":sample["measurement"]["ttft_p95_seconds"],
+                        "actual_ID_delivery_and_provider_commit_clocks":sample["measurement"]["wire_token_metrics"],
+                        "all_offered_denominators_and_workload_label_fairness":sample["measurement"]["fairness_by_workload_tenant"],
+                        "physical_forward_and_scheduling_receipts":sample["measurement"]["same_engine_quiescent_health"],
+                        "memory_sampled_peak_device_mib":sample["memory_observations"]["peak_gpu_device_mib"],
+                        "memory_owned_process_RSS_peak_kib":sample["memory_observations"]["peak_owned_process_rss_kib"]}
+                        for sample in selected],
+                    "scope":"bounded observations only; B1 records explicit one-teacher versus max32 scheduling cost alongside actual kernels; no automatic causal speed/SLO/vLLM eligibility"})
         report["status"] = "PASS_BOUNDED_SAME_ZEROGRAVITY_OBSERVATIONS_ONLY"
-        report["actual_outputs_identical_across_samples_per_model"] = True
+        report["actual_outputs_identical_across_samples_per_model_and_workload"] = True
+        report["same_prompt_B1_and_B8_outputs_identical_per_model"] = len(workloads) == 2
         report["source_correct_SLO_goodput_qualified"] = False
         report["speed_or_fullmodel_eligibility_inferred"] = False
     except BaseException as error:
--- /dev/null
+++ native-prefix-before-after/test_driver.py
@@ -0,0 +1,131 @@
+#!/usr/bin/env python3
+"""ROOT ONLY source-free accounting fixtures; no HTTP/GPU/process/model calls."""
+import copy
+import hashlib
+import importlib.util
+from pathlib import Path
+import unittest
+
+SPEC = importlib.util.spec_from_file_location('prefix_driver', Path(__file__).with_name('measure_same_engine.py'))
+DRIVER = importlib.util.module_from_spec(SPEC)
+SPEC.loader.exec_module(DRIVER)
+A, B = 'accepted_d0prep_zerogravity', 'native_prefix_zerogravity'
+
+
+def fixture(label, count):
+    p = 35
+    requests = []
+    for request in range(count):
+        tokens = [{'gpu': {'producer_frontier':p+index, 'native_cohort':
+            {'publication_sequence':100+index,'width':8} if count == 8 else None}} for index in range(8)]
+        requests.append({'request_id':f'request-{request}','max_tokens':8,'usage_reported_completion_tokens':8,
+            'wire_request_id':f'fixture:request-{request}','provider_response_id':f'chatcmpl-fixture-{request}',
+            'completed':True,'classification':'completed','http_status':200,'http_accepted':True,
+            'http_attempted':True,'error':None,
+            'prompt_sha256':hashlib.sha256(DRIVER.PROMPT.encode()).hexdigest(),
+            'output_sha256':'2'*64,'finish_reason':'length',
+            'wire_token_observation': {'status':'provider_bound_complete_ids','token_ids':list(range(8)),
+                'client_delivery_status':'individual_token_ids_observed',
+                'provider_commit_observation_status':'provider_bound_complete_commit_observations',
+                'provider_token_records':tokens}})
+    delta = {field:0 for field in DRIVER.COUNTERS}
+    delta.update(completed_forward_kernel_dispatches=1000, completed_model_token_steps=count*(p+7),
+                 accepted_output_tokens=8*count, phase_counter_observability=
+                 'observed_native_prefix_phase_counters' if label == B else 'legacy_absent_not_observed')
+    for field in DRIVER.PHASE_COUNTERS:
+        delta[field] = 0 if label == B else None
+    if count == 8:
+        fused = p+7 if label == B else 8
+        delta.update(gpu_fused_batches=fused, gpu_fused_rows_completed=8*fused,
+                     state_cohort_publications=fused,state_cohort_rows=8*fused,
+                     native_projection_dispatches_completed=100,native_sampler_dispatches_completed=32,
+                     gpu_forward_calls_completed=fused if label == B else 24,
+                     committed_gpu_roots=8*fused if label == B else 80)
+        if label == B:
+            delta.update(gpu_fused_state_only_batches=p-1,gpu_fused_state_only_rows=8*(p-1),gpu_fused_greedy_batches=8)
+    else:
+        fused = 0
+        delta.update(gpu_forward_calls_completed=p+7 if label == B else 10,
+                     committed_gpu_roots=p+7 if label == B else 10)
+    delta['native_batches_by_rows'] = {'2':0,'4':0,'8':fused}
+    return {'run_id':'fixture','offered_requests':count,'completed_requests':count,'http_accepted_requests':count,
+            'usage_reported_completion_tokens':8*count,'unsuccessful_requests':0,'completions_without_usage':0,
+            'http_429_requests':0,'http_503_requests':0,'rejected_requests':0,'cancelled_requests':0,
+            'timed_out_requests':0,'client_overload_requests':0,'launch_deadline_requests':0,
+            'classifications':{'completed':count},'errors':{},'requests':requests,
+            'wire_token_metrics':{'requested':True,'completed_requests_with_individual_token_delivery':count,
+                                  'completed_requests_with_provider_commit_observations':count},
+            'same_engine_quiescent_health':{'delta':delta}}
+
+
+class Accounting(unittest.TestCase):
+    def test_b8_native_prefix_is_not_counted_as_sampler_or_only8total_batches(self):
+        old, new = fixture(A,8), fixture(B,8)
+        self.assertEqual(DRIVER.counters(old,A,8)[0],DRIVER.counters(new,B,8)[0])
+        self.assertEqual(new['same_engine_quiescent_health']['delta']['gpu_fused_batches'],42)
+        self.assertEqual(DRIVER.counters(new,B,8)[1]['observed_greedy_cohorts'],8)
+
+    def test_b1_same_teacher_tokens_distinct_actual_scheduling_API_roots(self):
+        old, new = fixture(A,1), fixture(B,1)
+        left, a = DRIVER.counters(old,A,1)
+        right, b = DRIVER.counters(new,B,1)
+        self.assertEqual(left,right)
+        self.assertEqual(a['B1_teacher_scheduling']['observed_forward_API_calls'],10)
+        self.assertEqual(b['B1_teacher_scheduling']['observed_forward_API_calls'],42)
+
+    def test_legacy_absence_retained_as_unobserved_and_zeros_cannot_stand_in(self):
+        value=fixture(A,8)
+        self.assertTrue(all(value['same_engine_quiescent_health']['delta'][field] is None for field in DRIVER.PHASE_COUNTERS))
+        for field in DRIVER.PHASE_COUNTERS:value['same_engine_quiescent_health']['delta'][field]=0
+        with self.assertRaises(ValueError):DRIVER.counters(value,A,8)
+
+    def test_sampler_counts_greedy_only_and_bands_and_rows_must_close(self):
+        for field,value in [('native_sampler_dispatches_completed',168),('gpu_fused_state_only_rows',1),
+                            ('gpu_fused_greedy_batches',42),('state_cohort_rows',64)]:
+            bad=fixture(B,8);bad['same_engine_quiescent_health']['delta'][field]=value
+            with self.assertRaises(ValueError):DRIVER.counters(bad,B,8)
+
+    def test_actual_output_membership_must_be8fullwidth8cohorts(self):
+        bad=fixture(B,8)
+        bad['requests'][0]['wire_token_observation']['provider_token_records'][0]['gpu']['native_cohort']['width']=4
+        with self.assertRaises(ValueError):DRIVER.counters(bad,B,8)
+
+    def test_offered_denominator_and_usage_and_frontiers_cannot_be_dropped(self):
+        for change in ['offer','usage','frontier']:
+            bad=fixture(B,8)
+            if change=='offer':bad['offered_requests']=9
+            elif change=='usage':bad['requests'][0]['usage_reported_completion_tokens']=7
+            else:bad['requests'][0]['wire_token_observation']['provider_token_records'][0]['gpu']['producer_frontier']=True
+            with self.assertRaises(ValueError):DRIVER.counters(bad,B,8)
+
+    def test_no_b1_multirow_label_or_invented_one_teacher_legacy(self):
+        bad=fixture(A,1);bad['same_engine_quiescent_health']['delta']['gpu_forward_calls_completed']=42
+        with self.assertRaises(ValueError):DRIVER.counters(bad,A,1)
+        bad=fixture(B,1);bad['requests'][0]['wire_token_observation']['provider_token_records'][0]['gpu']['native_cohort']={'publication_sequence':2,'width':8}
+        with self.assertRaises(ValueError):DRIVER.counters(bad,B,1)
+
+    def test_http_outcomes_and_global_usage_cannot_be_excluded(self):
+        for field,value in [('http_accepted_requests',7),('http_429_requests',1),('http_503_requests',1),
+                            ('timed_out_requests',1),('usage_reported_completion_tokens',63),
+                            ('classifications',{'completed':7,'client_overload':1})]:
+            bad=fixture(B,8);bad[field]=value
+            with self.subTest(field=field),self.assertRaises(ValueError):DRIVER.counters(bad,B,8)
+
+    def test_provider_binding_prompt_and_all_request_identity_required(self):
+        for field,value in [('wire_request_id','foreign'),('provider_response_id',''),
+                            ('provider_response_id','chatcmpl-fixture-1'),('prompt_sha256','0'*64),
+                            ('completed',False),('http_accepted',False),('finish_reason','stop')]:
+            bad=fixture(B,8);bad['requests'][0][field]=value
+            with self.subTest(field=field),self.assertRaises(ValueError):DRIVER.counters(bad,B,8)
+
+    def test_actual_individual_ids_and_commit_coverage_required(self):
+        bad=fixture(B,8)
+        bad['requests'][0]['wire_token_observation']['client_delivery_status']='unresolved_token_groups_or_binding'
+        with self.assertRaises(ValueError):DRIVER.counters(bad,B,8)
+        bad=fixture(B,8)
+        bad['wire_token_metrics']['completed_requests_with_provider_commit_observations']=7
+        with self.assertRaises(ValueError):DRIVER.counters(bad,B,8)
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ native-prefix-before-after/source-pins.json
@@ -0,0 +1,40 @@
+{
+  "scope": "Source-only measurement preparation; no provider/model/operator/GPU execution by author; no current-server build inferred",
+  "base_driver": {
+    "path": ".agents/runs/serving-and-speculation-activation-before-after-20261020/measure_same_engine.py",
+    "bytes": 28023,
+    "sha256": "f06a088c6b6abdc4bc7272cc50eecd3935f693b87a774f0c7bbd74ca86aa2124"
+  },
+  "accepted_server": {
+    "path": ".agents/runs/serving-and-speculation-parity-root-2026-10-19/activation-cold-final-server",
+    "bytes": 36365496,
+    "sha256": "a896a455ec5d9093e88706751a9450170cb68b277658268ba30082f85e7ce598",
+    "receipt": {
+      "path": ".agents/runs/serving-and-speculation-parity-root-2026-10-19/activation-cold-final-serving-artifacts.json",
+      "sha256": "148f990a06da8a94661304e2b89834c631fc6fc2662bd130d9e2f8dda9358ffb"
+    },
+    "compiled_source_head": "c5eb1f8e717c79146937e585a145a0052eff747b",
+    "landed_reference": "d0a6084",
+    "scope": "exact preserved actual receipt: c5 HEAD plus six source overlays; not a bare d0 build"
+  },
+  "old_prefill_policy_source": {
+    "revision": "c5eb1f8e717c79146937e585a145a0052eff747b",
+    "path": "crates/zerogravity-cli/src/serve/gpu_engine.rs",
+    "bytes": 34762,
+    "sha256": "e03886e60c1ea8426b11fe20a827454ea51295aa93fd40811a18ac8c6b073c86",
+    "prefill_count_body_excerpt": "&self) -> usize {\n        let remaining = self.prompt.len() - self.cursor;\n        if self.native_rows != 0 && remaining > 1 {\n            (remaining - 1).min(PREFILL_QUANTUM)\n        } else {\n            remaining.min(PREFILL_QUANTUM)\n        }\n    }\n    /// Presentation precedes consumption. A Ready-cohort retry retains this\n    /// mark and the original packet; only a newly committed packet resets it."
+  },
+  "tools": {
+    "serve-bench.py": {
+      "path": "tools/serve-bench.py",
+      "bytes": 55663,
+      "sha256": "e119af67a0b3dd384c76b4d2745f8ebf6807cb27832b132ad2341580e9c5373f"
+    },
+    "serve-lifecycle.py": {
+      "path": "tools/serve-lifecycle.py",
+      "bytes": 40831,
+      "sha256": "b3f9c76f872711edbe8eea5731f6f85c3ba119e86386b8dd50798b507fae9cd3"
+    }
+  },
+  "current_server": "ROOT must supply actual frozen binary/source receipt path+hash after physical acceptance; no placeholder qualifies"
+}
--- /dev/null
+++ native-prefix-before-after/README.md
@@ -0,0 +1,128 @@
+# Same-engine native-prefix before/after — ROOT EXECUTES
+
+Parent: .agents/plans/serving-and-speculation/TODO.md. Owner: spine_review.
+This ignored source-only preparation changes no production/tools/shared plans.
+ROOT owns all builds, tests, CPU/operator/model/container/GPU/provider execution.
+Author performed source/AST/small metadata inspection only; no benchmark/test run.
+
+Both slots ARE Zerogravity; there is no vLLM slot or matched-vLLM claim.
+
+- A `accepted_d0prep_zerogravity`: exact preserved CUDA server SHA
+  a896a455ec5d9093e88706751a9450170cb68b277658268ba30082f85e7ce598.
+  Actual build receipt SHA148f990a06da8a94661304e2b89834c631fc6fc2662bd130d9e2f8dda9358ffb:
+  compiled c5eb1f8e717c79146937e585a145a0052eff747b plus its six accepted
+  source overlays, subsequently landed d0a6084. Never called a bare-d0 build.
+- B `native_prefix_zerogravity`: ROOT must supply separately frozen actual
+  binary SHA and reviewed source/build receipt path+SHA. Receipt must bind its
+  server bytes/path, actual compiled40hex HEAD and source overlay hashes incl
+  CLI worker/cohort/health plus resident fused implementation. Caller source-ID
+  is labelled caller-supplied; receipt is attestation, not independent math or
+  reproducible-build proof. No current binary or performance is invented here.
+- Adapted reviewed measure_same_engine.py; pinned canonical serve-bench and
+  serve-lifecycle modules unchanged. SameEnginePlan adapts initialization only;
+  original sample/OwnedProvider/log/PGID/receipt-I/O/finite cleanup remain reused.
+  Source pins record the exact original driver, accepted receipt and old worker
+  source with max32 prefill scheduling. Old scripts/receipts stay untouched.
+
+## Predeclared finite measurements
+
+- Explicit ABBA or BAAB actual order, fresh owned process/sample. Run cold_process
+  and warmed_process separately; repeat reverse order separately. Each invocation
+  offers only selected b8 and/or b1 workloads, selected0.8B/2B, four samples each
+  (maximum16 owned providers, never concurrent). No automatic retry after a
+  failed sample, row-band miss, output mismatch, cleanup failure or SLO miss.
+- Same approved original GGUF bytes, config/tokenizer hashes, model alias,
+  prompt, source EOS, greedy temperature0/top_p1/n1 and output8 for A/B.
+  No ignore-EOS policy. Early EOS refuses this bounded8-output comparison.
+- Same selected GPU UUID; one worker, active8, configured native rows8/context512;
+  host4096MiB/provider4096MiB/device5120MiB. Provider4GiB/device5GiB is the already
+  physically qualified service envelope. Logical CPU affinity exactly4 allowed
+  CPUs and Rayon/OMP/MKL/OpenBLAS4; not an OS CPU quota/physical RAM proof.
+- b8 = eight actual HTTP offers/concurrency8, b1 = one actual offer/concurrency1,
+  same prompt and fused-enabled server configuration. Workload arrivals all0,
+  max_pending equals offers; no synthetic server barrier or dropped offers.
+  Alternating local tenant labels are workload labels, not authenticated tenants.
+- warmed_process uses one same-offer/max1 warmup wave, records every result, then
+  one measured max8 wave. cold_process has no warmup. These names describe process
+  policy only: OS/storage/device caches are untouched, not claimed cold/all warm.
+  Source hashing occurs outside measured requests and can warm storage caches.
+- Initial and post-owned-cleanup GPU idle: three no-compute samples, utilization
+  <=5%, post memory<=initial+64MiB. Any other compute PID refuses without touching
+  it. Owned executable must match frozen bytes/path and selected GPU must actually
+  observe owned PID. Read-only .5s nvidia-smi/proc sampling can miss true peaks;
+  device/RSS/HWM observations are not ResourceArbiter/exact allocator memory proof.
+- Original bounded lifecycle: startup240s, shutdown30s, logs4MiB; request timeout
+  <=180s, GPU-idle30s, health quiescence10s, sampler join8s. Cleanup precedes
+  validation/evidence I/O/next spawn. Private unique outputs remain inside this run.
+  Proxy-disabled and redirect-refusing loopback HTTP opener is installed before
+  benchmark threads; no external service adoption or arbitrary stop commands.
+
+## Evidence and accounting closure
+
+- All offered, HTTP accepted/completed,429/503/rejected/cancelled/timeout/client
+  overload/launch-deadline outcomes, exact per-request terminal usage and total
+  usage must account. Every request must be actual HTTP200/completed/no error,
+  retain exact offered/wire/provider identities, prompt SHA and final length.
+- All8 actual token IDs/request require provider-bound complete ID status,
+  individual client delivery and complete distinct provider commit observations.
+  Same prompt must yield exact IDs/order/text SHA/usage/finish/frontiers across
+  every request/sample and both B1/B8 workloads for each model. This is limited
+  output evidence, not independent full-vocabulary/42-state mathematical proof.
+- Actual packet-producing committed frontier P comes from provider observations,
+  never tokenizer-length inference. Each request must report P..P+7, same P.
+- b8 must physically observe exactly8 width8 greedy publication groups across
+  all64 output IDs, each group containing every offered request exactly once at
+  the same output index. Row-band miss refuses; it is not relabelled B8.
+- New health requires all three phase counters: state-only batches/rows and
+  greedy batches. Total fused batches/rows/weighted2/4/8 bands and actual atomic
+  cohort publications/rows must close. There must be genuine state-only prefix
+  cohorts plus8 greedy cohorts; sampler dispatches=4*greedy, never4*all prefix
+  cohorts. New API calls=fused_batches+model_token_steps-fused_rows; committed
+  roots=model_token_steps. B1 remainder work remains counted.
+- Old phase fields MUST remain absent: report None/legacy_absent_not_observed,
+  never fabricate observed zero. Its actual eight greedy cohorts/64 rows and
+ 4*8 sampler launches close separately; original max32 prefix API/root receipts
+  are retained. Common counters, including state_cohort_rows, are observed in
+  both accepted old and new health.
+- b1 must have zero actual native multirow cohorts/projections/samplers in both.
+  With actual P and8 outputs, model steps=P+7. Old prefix API count is
+  ceil((P-1)/32), plus8 greedy/API/publication calls; new one-teacher policy is
+  P-1 prefix+8=P+7. Report this explicit scheduling cost beside actual physical
+  forward-kernel receipts, TTFT/ITL, wall/memory; do not infer a causal speedup.
+- Actual completed full-forward kernels cover disjoint native projection/sampler
+  subsets; projection means final-dot subset, sampler counts four native B>1
+  kernels. B1 samplers, initialization/DMA/unknown completion have the original
+  counter scope, not a false graph-as-one-kernel interpretation.
+- No failed/refused/discarded/cancelled GPU work accepted in timing samples.
+  Queued/active/parked fields remain before/after gauges; zero endpoints do not
+  establish zero parking incidents. Full raw health and per-request records stay.
+- TTFT<=10s and actual single-ID delivery ITL<=2s are predeclared maxima; CLI
+  may tighten them, never widen. SLO misses remain recorded, not hidden/retried.
+  Actual ID delivery ITL and provider commit-observation clocks stay distinct.
+  Text TTFT/SSE chunk intervals are not true token ITL; no source-correct SLO
+  goodput or vLLM equivalence is inferred from this same-engine experiment.
+
+## ROOT commands after physical acceptance
+
+Ten source-free accounting fixture methods are prepared, AST-only by author.
+ROOT first executes fixtures, reviews the current actual source/build receipt,
+freezes its binary, and waits for all CPU/build/GPU measurement holds to lift.
+
+```sh
+TASK_PREFIX_AB=.agents/runs/serving-and-speculation-native-prefix-before-after-20261020
+python3 "$TASK_PREFIX_AB/test_driver.py"
+python3 "$TASK_PREFIX_AB/measure_same_engine.py" --run \
+  --current-server /ABSOLUTE/OWNED/FROZEN/NATIVE-PREFIX-SERVER \
+  --current-sha256 ACTUAL_64HEX_BINARY_SHA \
+  --current-source-id ACTUAL_SOURCE_BUILD_RECEIPT_ID \
+  --current-receipt /ABSOLUTE/OWNED/ACTUAL_BUILD_RECEIPT.json \
+  --current-receipt-sha256 ACTUAL_64HEX_RECEIPT_SHA \
+  --gpu-uuid ACTUAL_GPU_UUID --models 0.8b --workloads b8,b1 \
+  --order ABBA --policy warmed_process
+```
+
+Inspect first0.8B report before2B, reverse order and cold controls. Each invocation
+creates private actual-measurements/same-zg-* with full raw observations and
+original lifecycle receipts. Success means bounded same-Zerogravity observations
+only. Unchanged outputs, physical prefix fusion and source/math/full-serving
+acceptance remain distinct claims; no fabricated ratio or matched-vLLM result.
--- /dev/null
+++ native-prefix-before-after/TODO.md
@@ -0,0 +1,19 @@
+# Native-prefix before/after source preparation
+
+Parent: .agents/plans/serving-and-speculation/TODO.md. Owner: spine_review.
+ROOT alone executes all build/test/provider/CPU/model/operator/GPU work.
+
+- [x] Preserve original accepted driver/actual receipts; identify exact a896
+  accepted cold-preparation server and c5+six-overlay build attestation.
+- [x] Adapt original owned lifecycle with honest same-engine labels, same caps,
+  ABBA/BAAB order, cold/warm policies, actual8-offer and1-offer workloads.
+- [x] Separate observed prefix/greedy metrics from absent legacy phase fields;
+  close actual greedy output memberships, rows/bands/sampler/kernel/root counts.
+- [x] Keep all offered outcomes, true ID delivery/provider commit clocks, exact
+  outputs/frontiers/usage across samples and B1/B8; no wider SLO or vLLM claim.
+- [x] Prepare10 source-free accounting fixture methods; Python AST only by author.
+- [ ] Independent source review, ROOT fixture execution and fresh current-server
+  actual artifact/source receipt after physical prefix gates.
+- [ ] ROOT only actual serialized bounded cold/warm ABBA/BAAB after all competing
+  build/CPU/GPU holds lift; report measured B1 scheduling cost and B8 gains/limits.
+- [ ] Genuine matched pinned-vLLM/source-correct SLO gate remains separate/open.

````

