# Draft Patches — volume 26

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/frozen-prepared/NEW_DELTA.patch

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

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 58414; SHA256 `56845b51e9170304c5355e2fa96dbf9289d0c152adedab9d7602f771bfd1c595`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_batch/chunk.rs
@@ -0,0 +1,186 @@
+//! Closed teacher chunk metadata, never a root issuer or execution authority.
+use super::*;
+
+#[derive(Clone, Debug)]
+pub(crate) struct ResidentBatchChunk {
+    roots: ResidentBatchRows,
+    teachers: Vec<Vec<u32>>,
+    vocabulary: u64,
+    capacity: usize,
+}
+impl ResidentBatchChunk {
+    pub(crate) fn new(
+        bindings: &[RootBinding],
+        teachers: &[&[u32]],
+        vocabulary: u64,
+        capacity: usize,
+    ) -> Result<Self> {
+        if !matches!(bindings.len(), 2 | 4 | 8)
+            || teachers.len() != bindings.len()
+            || vocabulary == 0
+            || vocabulary > u64::from(u32::MAX) + 1
+            || capacity == 0
+            || capacity > MAX_PREFIX as usize
+        {
+            return Err(Error::Unsupported(
+                "native teacher chunk source/band/capacity".into(),
+            ));
+        }
+        let count = teachers.first().map_or(0, |row| row.len());
+        if !matches!(count, 2 | 4) || teachers.iter().any(|row| row.len() != count) {
+            return Err(Error::Unsupported(
+                "native teacher chunk requires equal T2/T4 rows".into(),
+            ));
+        }
+        let roots = ResidentBatchRows::new(bindings, 1)?;
+        if roots.frontiers().any(|position| {
+            position
+                .checked_add(count as u64)
+                .is_none_or(|end| end > capacity as u64)
+        }) {
+            return Err(Error::Limit(
+                "native teacher chunk exceeds original context".into(),
+            ));
+        }
+        if teachers
+            .iter()
+            .flat_map(|row| row.iter())
+            .any(|&token| u64::from(token) >= vocabulary)
+        {
+            return Err(Error::Malformed(
+                "native teacher chunk contains invalid source token".into(),
+            ));
+        }
+        Ok(Self {
+            roots,
+            teachers: teachers.iter().map(|row| row.to_vec()).collect(),
+            vocabulary,
+            capacity,
+        })
+    }
+    pub(crate) fn tokens_per_row(&self) -> usize {
+        self.teachers[0].len()
+    }
+    pub(crate) fn rows(&self) -> usize {
+        self.teachers.len()
+    }
+    pub(crate) fn vocabulary(&self) -> u64 {
+        self.vocabulary
+    }
+    pub(crate) fn capacity(&self) -> usize {
+        self.capacity
+    }
+    pub(crate) fn require_current(&self, bindings: &[RootBinding]) -> Result<()> {
+        self.roots.require_current(bindings)
+    }
+    pub(crate) fn token(&self, row: usize, step: usize) -> Result<u32> {
+        self.teachers
+            .get(row)
+            .and_then(|tokens| tokens.get(step))
+            .copied()
+            .ok_or_else(|| Error::State("native teacher chunk row/step changed".into()))
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use zerogravity_core::session_root::SessionAuthority;
+
+    #[test]
+    fn equal_teacher_lengths_preserve_independent_frontiers_and_one_root_advance() {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        let mut owners = (0..8)
+            .map(|_| SessionAuthority::new(source).unwrap())
+            .collect::<Vec<_>>();
+        for (owner, position) in owners.iter_mut().zip([0, 1, 30, 255, 256, 508, 100, 17]) {
+            if position > 0 {
+                let attempt = owner.begin_advance(position).unwrap();
+                owner.prepare_publish(attempt).unwrap().commit();
+            }
+        }
+        for rows in [2, 4, 8] {
+            for count in [2, 4] {
+                let old = owners
+                    .iter()
+                    .take(rows)
+                    .map(SessionAuthority::binding)
+                    .collect::<Vec<_>>();
+                let teacher = (0..rows)
+                    .map(|row| {
+                        (0..count)
+                            .map(|step| (row * 17 + step + 11) as u32)
+                            .collect::<Vec<_>>()
+                    })
+                    .collect::<Vec<_>>();
+                let refs = teacher.iter().map(Vec::as_slice).collect::<Vec<_>>();
+                let chunk = ResidentBatchChunk::new(&old, &refs, 248320, 512).unwrap();
+                assert_eq!(chunk.rows(), rows);
+                assert_eq!(chunk.tokens_per_row(), count);
+                for (row, tokens) in teacher.iter().enumerate() {
+                    for (step, &token) in tokens.iter().enumerate() {
+                        assert_eq!(chunk.token(row, step).unwrap(), token);
+                    }
+                }
+                chunk.require_current(&old).unwrap();
+                assert!(chunk.token(rows, 0).is_err());
+                assert!(chunk.token(0, count).is_err());
+                let attempt = owners[0].begin_advance(count as u64).unwrap();
+                let published = owners[0].prepare_publish(attempt).unwrap().commit();
+                assert_eq!(published.frontier(), old[0].frontier() + count as u64);
+                assert_eq!(published.root(), old[0].root() + 1);
+                assert!(
+                    chunk
+                        .require_current(
+                            &owners
+                                .iter()
+                                .take(rows)
+                                .map(SessionAuthority::binding)
+                                .collect::<Vec<_>>()
+                        )
+                        .is_err()
+                );
+            }
+        }
+    }
+
+    #[test]
+    fn cold_refusals_never_mutate_roots_or_invent_padding() {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        let a = SessionAuthority::new(source).unwrap();
+        let b = SessionAuthority::new(source).unwrap();
+        let mut exhausted = SessionAuthority::new(source).unwrap();
+        let advance = exhausted.begin_advance(511).unwrap();
+        exhausted.prepare_publish(advance).unwrap().commit();
+        let roots = [a.binding(), b.binding()];
+        let original = roots;
+        for teachers in [
+            vec![&[11u32][..], &[353u32][..]],
+            vec![&[11, 12][..], &[353, 354, 355, 356][..]],
+            vec![&[11, 248320][..], &[353, 354][..]],
+        ] {
+            assert!(ResidentBatchChunk::new(&roots, &teachers, 248320, 512).is_err());
+        }
+        assert!(
+            ResidentBatchChunk::new(
+                &[roots[0], roots[0]],
+                &[&[11, 12], &[353, 354]],
+                248320,
+                512
+            )
+            .is_err()
+        );
+        assert!(
+            ResidentBatchChunk::new(
+                &[roots[0], exhausted.binding()],
+                &[&[11, 12], &[353, 354]],
+                248320,
+                512
+            )
+            .is_err()
+        );
+        assert!(ResidentBatchChunk::new(&roots, &[&[11, 12], &[353, 354]], 0, 512).is_err());
+        assert!(ResidentBatchChunk::new(&roots, &[&[11, 12], &[353, 354]], 248320, 513).is_err());
+        assert_eq!([a.binding(), b.binding()], original);
+    }
+}
--- a/crates/zerogravity-gpu/src/resident_batch.rs
+++ b/crates/zerogravity-gpu/src/resident_batch.rs
@@ -225,3 +225,8 @@
         }
     }
 }
+
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+mod chunk;
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+pub(crate) use chunk::ResidentBatchChunk;
--- a/crates/zerogravity-gpu/src/resident_inference/fused.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/fused.rs
@@ -3,6 +3,8 @@
 
 pub(super) enum FusedExecution {
     Native,
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    NativeChunk(crate::resident_batch::ResidentBatchChunk),
     #[cfg(all(feature = "qualification", feature = "cuda"))]
     Graph(Arc<super::graph::CapturedResidentGraph>),
 }
@@ -58,6 +60,39 @@
             .await
     }
 
+    /// Closed qualification-only teacher chunk. Each row consumes the same T2/T4
+    /// length with independent positions, one terminal Ready and one root advance.
+    /// Callers must reserve the prompt tail; no packet/EOS/output is presented.
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    pub async fn prepare_state_fused_chunk_admitted(
+        self: &Arc<Self>,
+        rows: &[(&ResidentSession, &[u32])],
+    ) -> Result<Option<PreparedResidentCohort>> {
+        let bases = rows
+            .iter()
+            .map(|(session, _)| session.binding())
+            .collect::<Vec<_>>();
+        let teachers = rows.iter().map(|(_, tokens)| *tokens).collect::<Vec<_>>();
+        let chunk = crate::resident_batch::ResidentBatchChunk::new(
+            &bases,
+            &teachers,
+            self.vocab,
+            self.capacity,
+        )?;
+        let first = rows
+            .iter()
+            .enumerate()
+            .map(|(row, (session, _))| Ok((*session, chunk.token(row, 0)?)))
+            .collect::<Result<Vec<_>>>()?;
+        self.prepare_fused_execution(
+            &first,
+            None,
+            Readout::None,
+            FusedExecution::NativeChunk(chunk),
+        )
+        .await
+    }
+
     /// One native full-model batch and source-exact device greedy selection.
     /// Each independent output packet gains its producer only at cohort commit.
     pub async fn prepare_greedy_fused_cohort(
@@ -142,6 +177,21 @@
             // attempts, candidates, packet/workspace admission or queue work.
             graph_readout(rows.len(), graph.output(), readout, device_tokens.is_some())?;
         }
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        if matches!(&execution, FusedExecution::NativeChunk(_))
+            && (!matches!(readout, Readout::None) || device_tokens.is_some())
+        {
+            return Err(Error::Unsupported(
+                "teacher chunk requires state-only host tokens".into(),
+            ));
+        }
+        let tokens_per_row = match &execution {
+            FusedExecution::Native => 1,
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            FusedExecution::Graph(_) => 1,
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            FusedExecution::NativeChunk(chunk) => chunk.tokens_per_row(),
+        };
         let output = if matches!(readout, Readout::None) {
             ResidentOutput::StateOnly
         } else {
@@ -154,6 +204,23 @@
         }
         let bases = rows.iter().map(|(s, _)| s.binding()).collect::<Vec<_>>();
         let roots = ResidentBatchRows::new(&bases, 1)?;
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        if let FusedExecution::NativeChunk(chunk) = &execution {
+            chunk.require_current(&bases)?;
+            for (row, (_, token)) in rows.iter().enumerate() {
+                if chunk.token(row, 0)? != *token {
+                    return Err(Error::State(
+                        "teacher chunk first source token changed".into(),
+                    ));
+                }
+            }
+            if chunk.rows() != rows.len()
+                || chunk.vocabulary() != self.vocab
+                || chunk.capacity() != self.capacity
+            {
+                return Err(Error::State("teacher chunk source geometry changed".into()));
+            }
+        }
         if !matches!(rows.len(), 2 | 4 | 8) {
             return Err(Error::Unsupported(
                 "resident native batch rows must be 2/4/8".into(),
@@ -188,7 +255,11 @@
                     .map_err(|_| Error::Limit("native resident batch frontier overflow".into()))
             })
             .collect::<Result<Vec<_>>>()?;
-        if positions.iter().any(|&position| position >= self.capacity) {
+        if positions.iter().any(|&position| {
+            position
+                .checked_add(tokens_per_row)
+                .is_none_or(|end| end > self.capacity)
+        }) {
             return Err(Error::Limit(
                 "native resident batch capacity exhausted".into(),
             ));
@@ -205,7 +276,7 @@
                 ));
             }
             self.validate_domains(&session.domains, true)?;
-            attempts.push(session.authority.begin_advance(1)?);
+            attempts.push(session.authority.begin_advance(tokens_per_row as u64)?);
         }
         let staging_bytes = self
             .state_bytes
@@ -223,6 +294,17 @@
         {
             let mut workspace = self.workspace.lock().await;
             self.service.ensure_usable()?;
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            if matches!(&execution, FusedExecution::NativeChunk(_))
+                && workspace
+                    .batch
+                    .as_ref()
+                    .is_some_and(|batch| batch.program.graph_pinned())
+            {
+                return Err(Error::Unsupported(
+                    "teacher chunk refuses a graph-pinned workspace before allocation".into(),
+                ));
+            }
             let same_band = workspace
                 .batch
                 .as_ref()
@@ -309,6 +391,31 @@
                 valid_tokens: position + 1,
             })
             .collect::<Vec<_>>();
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let chunk_parameters = if let FusedExecution::NativeChunk(chunk) = &execution {
+            Some(
+                (0..tokens_per_row)
+                    .map(|step| {
+                        positions
+                            .iter()
+                            .enumerate()
+                            .map(|(row, &position)| {
+                                let position = position.checked_add(step).ok_or_else(|| {
+                                    Error::Limit("teacher chunk position overflow".into())
+                                })?;
+                                Ok(RuntimeParameters {
+                                    token: chunk.token(row, step)?,
+                                    position,
+                                    valid_tokens: position + 1,
+                                })
+                            })
+                            .collect::<Result<Vec<_>>>()
+                    })
+                    .collect::<Result<Vec<_>>>()?,
+            )
+        } else {
+            None
+        };
         // State initialization uses allocation -> queue. Preserve that order
         // while admitting a cold batch workspace, before selecting a slot.
         let packets = if matches!(readout, Readout::Greedy) {
@@ -431,6 +538,12 @@
         batch
             .program
             .validate_runtime(self.service.client(), &parameters, &bindings)?;
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        if let Some(steps) = &chunk_parameters {
+            batch
+                .program
+                .validate_state_chunk(self.service.client(), steps, &bindings)?;
+        }
         let device_buffers = device_tokens.map(|tokens| {
             tokens
                 .iter()
@@ -522,6 +635,14 @@
         let mut submitted = match &execution {
             #[cfg(all(feature = "qualification", feature = "cuda"))]
             FusedExecution::Graph(graph) => graph.launch(program, &parameters, &bindings)?,
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            FusedExecution::NativeChunk(_) => program.launch_state_chunk(
+                self.service.client(),
+                chunk_parameters
+                    .as_ref()
+                    .ok_or_else(|| Error::State("teacher chunk parameters absent".into()))?,
+                &bindings,
+            )?,
             FusedExecution::Native => {
                 if let Some(tokens) = &device_buffers {
                     program.launch_device_tokens(
@@ -687,7 +808,11 @@
                     } else {
                         Vec::new()
                     },
-                    state_only_steps: usize::from(output == ResidentOutput::StateOnly),
+                    state_only_steps: if output == ResidentOutput::StateOnly {
+                        tokens_per_row
+                    } else {
+                        0
+                    },
                     logit_steps: usize::from(output == ResidentOutput::Logits),
                     bytes: row_bytes + usize::from(row == 0) * 4,
                     chunk_bytes: row_bytes + usize::from(row == 0) * 4,
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk.rs
@@ -0,0 +1,675 @@
+//! ROOT-only full-model temporal proof; original CPU/B1/root helpers unchanged.
+use super::*;
+
+async fn chunk(
+    model: &Arc<ResidentModel>,
+    sessions: &[ResidentSession],
+    teachers: &[Vec<u32>],
+) -> Result<Option<PreparedResidentCohort>> {
+    assert_eq!(sessions.len(), teachers.len());
+    let rows = sessions
+        .iter()
+        .zip(teachers)
+        .map(|(session, tokens)| (session, tokens.as_slice()))
+        .collect::<Vec<_>>();
+    model.prepare_state_fused_chunk_admitted(&rows).await
+}
+fn chunk_reads(reads: &[InferenceReadback], rows: usize, steps: usize, dispatches: usize) {
+    assert_eq!(reads.len(), rows);
+    assert_eq!(reads.iter().map(|read| read.bytes).sum::<usize>(), 4);
+    assert_eq!(reads.iter().map(|read| read.chunk_bytes).sum::<usize>(), 4);
+    assert_eq!(
+        reads.iter().map(|read| read.dispatches).sum::<usize>(),
+        dispatches
+    );
+    for read in reads {
+        assert_eq!(read.state_only_steps, steps);
+        assert_eq!(read.logit_steps, 0);
+        assert!(read.logits.is_empty() && read.device_token.is_none());
+    }
+}
+async fn oracle(
+    source: &Arc<LoadedQwenModel>,
+    model: &Arc<ResidentModel>,
+    rows: usize,
+) -> Result<(
+    Vec<ResidentSession>,
+    Vec<ResidentSession>,
+    Vec<QwenDenseCpu>,
+)> {
+    let mut native = Vec::new();
+    let mut single = Vec::new();
+    let mut cpu = Vec::new();
+    for row in 0..rows {
+        let mut reference = QwenDenseCpu::from_shared(Arc::clone(source))?;
+        let mut b1 = model.session().await?;
+        let mut lane = model.session().await?;
+        for step in 0..row % 3 + 1 {
+            let id = IDS[(row + step) % IDS.len()];
+            let expected = reference.forward(id)?;
+            exact(
+                &expected,
+                &b1.forward(&[id], false).await?.logits,
+                "chunk seed B1 fullV",
+            )?;
+            exact(
+                &expected,
+                &lane.forward(&[id], false).await?.logits,
+                "chunk seed native B1 fullV",
+            )?;
+        }
+        native.push(lane);
+        single.push(b1);
+        cpu.push(reference);
+    }
+    state(&native, &cpu).await?;
+    state(&single, &cpu).await?;
+    Ok((native, single, cpu))
+}
+async fn advance_oracle(
+    single: &mut [ResidentSession],
+    cpu: &mut [QwenDenseCpu],
+    teachers: &[Vec<u32>],
+) -> Result<()> {
+    assert_eq!(single.len(), teachers.len());
+    assert_eq!(cpu.len(), teachers.len());
+    for step in 0..teachers[0].len() {
+        for row in 0..teachers.len() {
+            let expected = cpu[row].forward(teachers[row][step])?;
+            assert_eq!(expected.len(), 248320);
+            exact(
+                &expected,
+                &single[row]
+                    .forward(&[teachers[row][step]], false)
+                    .await?
+                    .logits,
+                "chunk each original teacher CPU/B1 fullV",
+            )?;
+        }
+        // Every teacher has independent accepted complete-domain evidence, not
+        // a reconstructed intermediate state from the temporal implementation.
+        state(single, cpu).await?;
+    }
+    Ok(())
+}
+async fn future(
+    model: &Arc<ResidentModel>,
+    native: &mut [ResidentSession],
+    single: &mut [ResidentSession],
+    cpu: &mut [QwenDenseCpu],
+) -> Result<()> {
+    let rows = native.len();
+    let ids = (0..rows)
+        .map(|row| IDS[(row + 4) % IDS.len()])
+        .collect::<Vec<_>>();
+    let mut expected = Vec::new();
+    for row in 0..rows {
+        let logits = cpu[row].forward(ids[row])?;
+        exact(
+            &logits,
+            &single[row].forward(&[ids[row]], false).await?.logits,
+            "chunk later CPU/B1 fullV",
+        )?;
+        expected.push(logits);
+    }
+    let inputs = native
+        .iter()
+        .zip(&ids)
+        .map(|(session, &id)| (session, id))
+        .collect::<Vec<_>>();
+    let candidate = model.prepare_logits_fused_cohort(&inputs).await?;
+    let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+    assert_eq!(reads.len(), rows);
+    for (read, wanted) in reads.iter().zip(&expected) {
+        assert_eq!(read.logits.len(), 248320);
+        assert_eq!(read.state_only_steps, 0);
+        assert_eq!(read.logit_steps, 1);
+        exact(
+            wanted,
+            &read.logits,
+            "chunk future complete native vocabulary",
+        )?;
+    }
+    state(native, cpu).await?;
+    state(single, cpu).await?;
+    drop(reads);
+    drop(expected);
+    // Actual accepted B1 greedy packets, plus CPU-selected full logits, remain
+    // independent of the new temporal path and its unchanged native sampler.
+    let ids = (0..rows)
+        .map(|row| IDS[(row + 6) % IDS.len()])
+        .collect::<Vec<_>>();
+    let mut selected = Vec::new();
+    for row in 0..rows {
+        let logits = cpu[row].forward(ids[row])?;
+        let saved = single[row].snapshot();
+        exact(
+            &logits,
+            &single[row].forward(&[ids[row]], false).await?.logits,
+            "chunk greedy independent fullV",
+        )?;
+        let index = (0..logits.len())
+            .max_by(|a, b| logits[*a].total_cmp(&logits[*b]).then_with(|| b.cmp(a)))
+            .unwrap();
+        let wanted = (index as u32, logits[index].to_bits());
+        single[row].restore(&saved)?;
+        let read = single[row].forward_greedy(&[ids[row]]).await?;
+        let packet = read.device_token.as_ref().unwrap();
+        assert_eq!((packet.id, packet.max_bits), wanted);
+        selected.push(wanted);
+    }
+    let inputs = native
+        .iter()
+        .zip(&ids)
+        .map(|(session, &id)| (session, id))
+        .collect::<Vec<_>>();
+    let candidate = model.prepare_greedy_fused_cohort(&inputs).await?;
+    assert_eq!(candidate.native_greedy_dispatches(), 4);
+    let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+    assert_eq!(reads.len(), rows);
+    for ((read, session), wanted) in reads.iter().zip(native.iter()).zip(selected) {
+        let packet = read.device_token.as_ref().unwrap();
+        assert_eq!((packet.id, packet.max_bits), wanted);
+        assert_eq!(packet.producer, Some(session.binding()));
+    }
+    state(native, cpu).await?;
+    state(single, cpu).await?;
+    Ok(())
+}
+async fn qualify(name: &str, rows: usize, steps: usize) -> Result<()> {
+    let (source, model, eos) = load(name).await?;
+    let (mut native, mut single, mut cpu) = oracle(&source, &model, rows).await?;
+    let teachers = (0..rows)
+        .map(|row| {
+            (0..steps)
+                .map(|step| {
+                    if step == 1 {
+                        eos
+                    } else {
+                        IDS[(row + step + 2) % IDS.len()]
+                    }
+                })
+                .collect::<Vec<_>>()
+        })
+        .collect::<Vec<_>>();
+    let before = bindings(&native);
+    let domains = domain_values(&native).await?;
+    let queue = model.service.queue().stats()?;
+    let mut invalid = teachers.clone();
+    invalid[rows - 1][steps - 1] = u32::MAX;
+    assert!(chunk(&model, &native, &invalid).await.is_err());
+    let mut ragged = teachers.clone();
+    ragged[rows - 1].pop();
+    assert!(chunk(&model, &native, &ragged).await.is_err());
+    let duplicate = vec![(&native[0], teachers[0].as_slice()); rows];
+    assert!(
+        model
+            .prepare_state_fused_chunk_admitted(&duplicate)
+            .await
+            .is_err()
+    );
+    assert_eq!(model.service.queue().stats()?, queue);
+    unchanged(&native, &before)?;
+    unchanged_domains(&native, &domains).await?;
+    let uploads = model.service.uploaded_value_bytes();
+    let abandoned = chunk(&model, &native, &teachers)
+        .await?
+        .ok_or_else(|| Error::Limit("chunk qualifier unexpected capacity refusal".into()))?;
+    state_receipt(&abandoned, &before);
+    unchanged(&native, &before)?;
+    drop(abandoned);
+    unchanged_domains(&native, &domains).await?;
+    let warm_bytes = model.service.resident_bytes()?;
+    drop(chunk(&model, &native, &teachers).await?.unwrap());
+    idle(&model, warm_bytes)?;
+    let saved = native[rows - 1].snapshot();
+    let stale = chunk(&model, &native, &teachers).await?.unwrap();
+    native[rows - 1].restore(&saved)?;
+    let restored = bindings(&native);
+    assert!(
+        stale
+            .commit(&mut native.iter_mut().collect::<Vec<_>>())
+            .is_err()
+    );
+    unchanged(&native, &restored)?;
+    unchanged_domains(&native, &domains).await?;
+    drop(saved);
+    let before = bindings(&native);
+    let candidate = chunk(&model, &native, &teachers).await?.unwrap();
+    state_receipt(&candidate, &before);
+    let dispatches = candidate.dispatches();
+    let projections = candidate.fused_dispatches();
+    assert_eq!(projections % steps, 0);
+    unchanged(&native, &before)?;
+    advance_oracle(&mut single, &mut cpu, &teachers).await?;
+    let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+    chunk_reads(&reads, rows, steps, dispatches);
+    for (session, old) in native.iter().zip(&before) {
+        assert_eq!(session.binding().root(), old.root() + 1);
+        assert_eq!(session.binding().frontier(), old.frontier() + steps as u64);
+        assert_eq!(session.binding().session_epoch(), old.session_epoch());
+    }
+    state(&native, &cpu).await?;
+    state(&single, &cpu).await?;
+    assert_eq!(model.service.pending_events()?, 0);
+    assert_eq!(
+        model.service.uploaded_value_bytes(),
+        uploads,
+        "warm temporal chunk must not upload model/activation/state data"
+    );
+    drop(reads);
+    future(&model, &mut native, &mut single, &mut cpu).await?;
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    reclaim(model).await?;
+    println!(
+        "P12_NATIVE_TEACHER_CHUNK_PASS source={name} B={rows} T={steps} original_CPU_B1_each_teacher_fullV_all42=true native_final_all42_exact=true future_fullV_and_greedy_exact=true one_terminal_status_bytes=4 one_root_advance=true teacher_EOS_no_output=true original_kernel_dispatches={dispatches} projection_subset={projections} final_reserved=0"
+    );
+    Ok(())
+}
+
+macro_rules! case {
+    ($test:ident,$name:literal,$rows:literal,$steps:literal) => {
+        #[test]
+        #[ignore = "physical original CUDA teacher chunk; ROOT --release --test-threads=1"]
+        fn $test() {
+            futures_lite::future::block_on(qualify($name, $rows, $steps)).unwrap();
+        }
+    };
+}
+case!(qwen08_b2_t2, "qwen35-0.8b", 2, 2);
+case!(qwen08_b2_t4, "qwen35-0.8b", 2, 4);
+case!(qwen08_b4_t2, "qwen35-0.8b", 4, 2);
+case!(qwen08_b4_t4, "qwen35-0.8b", 4, 4);
+case!(qwen08_b8_t2, "qwen35-0.8b", 8, 2);
+case!(qwen08_b8_t4, "qwen35-0.8b", 8, 4);
+case!(qwen2_b2_t2, "qwen35-2b", 2, 2);
+case!(qwen2_b2_t4, "qwen35-2b", 2, 4);
+case!(qwen2_b4_t2, "qwen35-2b", 4, 2);
+case!(qwen2_b4_t4, "qwen35-2b", 4, 4);
+case!(qwen2_b8_t2, "qwen35-2b", 8, 2);
+case!(qwen2_b8_t4, "qwen35-2b", 8, 4);
+
+async fn pressure_advance_chunk(
+    model: &Arc<ResidentModel>,
+    native: &mut [ResidentSession],
+    single: &mut [ResidentSession],
+    cpu: &mut [QwenDenseCpu],
+    teachers: &[Vec<u32>],
+) -> Result<()> {
+    advance_oracle(single, cpu, teachers).await?;
+    let before = bindings(native);
+    let candidate = chunk(model, native, teachers)
+        .await?
+        .ok_or_else(|| Error::Limit("explicit smaller chunk refused".into()))?;
+    state_receipt(&candidate, &before);
+    let dispatches = candidate.dispatches();
+    let rows = native.len();
+    let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+    chunk_reads(&reads, rows, teachers[0].len(), dispatches);
+    state(native, cpu).await?;
+    state(single, cpu).await?;
+    Ok(())
+}
+async fn qualify_pressure(name: &str) -> Result<()> {
+    let (source, model, eos) = load(name).await?;
+    let (mut native, mut single, mut cpu) = oracle(&source, &model, 8).await?;
+    let teachers = (0..8)
+        .map(|row| vec![IDS[row], eos, IDS[(row + 1) % 8], IDS[(row + 2) % 8]])
+        .collect::<Vec<_>>();
+    let (need4, need8) = {
+        let workspace = model.workspace.lock().await;
+        assert!(workspace.batch.is_none());
+        let need = |rows| -> Result<usize> {
+            BatchProgram::preflight(&workspace.program, rows)?
+                .owned_bytes()?
+                .checked_add(model.state_bytes * rows)
+                .ok_or_else(|| Error::Limit("chunk pressure footprint".into()))
+        };
+        (need(4)?, need(8)?)
+    };
+    let available = model.service.remaining_capacity()?;
+    assert!(need4 < need8 && need8 <= available);
+    let pressure = model
+        .service
+        .allocate_uninitialized(available - need4, BufferRole::Scratch)?;
+    let before = bindings(&native);
+    let domains = domain_values(&native).await?;
+    let bytes = model.service.resident_bytes()?;
+    let queue = model.service.queue().stats()?;
+    let uploads = model.service.uploaded_value_bytes();
+    assert!(chunk(&model, &native, &teachers).await?.is_none());
+    unchanged(&native, &before)?;
+    unchanged_domains(&native, &domains).await?;
+    assert_eq!(model.service.resident_bytes()?, bytes);
+    assert_eq!(model.service.queue().stats()?, queue);
+    assert_eq!(model.service.uploaded_value_bytes(), uploads);
+    assert_eq!(model.service.pending_events()?, 0);
+    assert!(model.workspace.lock().await.batch.is_none());
+    pressure_advance_chunk(
+        &model,
+        &mut native[..4],
+        &mut single[..4],
+        &mut cpu[..4],
+        &teachers[..4],
+    )
+    .await?;
+    pressure_advance_chunk(
+        &model,
+        &mut native[4..],
+        &mut single[4..],
+        &mut cpu[4..],
+        &teachers[4..],
+    )
+    .await?;
+    let before = bindings(&native);
+    let bytes = model.service.resident_bytes()?;
+    let queue = model.service.queue().stats()?;
+    assert!(chunk(&model, &native, &teachers).await?.is_none());
+    unchanged(&native, &before)?;
+    assert_eq!(model.service.resident_bytes()?, bytes);
+    assert_eq!(model.service.queue().stats()?, queue);
+    drop(pressure);
+    pressure_advance_chunk(&model, &mut native, &mut single, &mut cpu, &teachers).await?;
+    future(&model, &mut native, &mut single, &mut cpu).await?;
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    reclaim(model).await?;
+    println!(
+        "P12_NATIVE_TEACHER_CHUNK_PRESSURE_PASS source={name} T=4 B8_None_before_launch=true explicit_B4_then_B8_after_release=CPU_B1_all42_exact one_terminal_status_bytes=4 future_fullV_and_greedy_exact=true final_reserved=0"
+    );
+    Ok(())
+}
+#[test]
+#[ignore = "physical original CUDA chunk pressure; ROOT --release --test-threads=1"]
+fn qwen08_chunk_pressure() {
+    futures_lite::future::block_on(qualify_pressure("qwen35-0.8b")).unwrap();
+}
+#[test]
+#[ignore = "physical original CUDA chunk pressure; ROOT --release --test-threads=1"]
+fn qwen2_chunk_pressure() {
+    futures_lite::future::block_on(qualify_pressure("qwen35-2b")).unwrap();
+}
+
+#[cfg(feature = "hardware-abort-probe")]
+async fn qualify_abort(name: &str) -> Result<()> {
+    use std::sync::atomic::Ordering;
+    let (source, model, eos) = load(name).await?;
+    let (mut native, single, cpu) = oracle(&source, &model, 2).await?;
+    let teachers = vec![vec![11, eos, 55, 101], vec![353, eos, 78, 135]];
+    let before = bindings(&native);
+    let saved = native
+        .iter()
+        .map(ResidentSession::snapshot)
+        .collect::<Vec<_>>();
+    let domains = domain_values(&native).await?;
+    let ready = chunk(&model, &native, &teachers).await?.unwrap();
+    state_receipt(&ready, &before);
+    let client = model.service.client().clone();
+    let old_quarantine = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    model.launch_seen.store(0, Ordering::Release);
+    model.abort_after_launch.store(1, Ordering::Release);
+    let mut pending = Box::pin(chunk(&model, &native, &teachers));
+    loop {
+        if let Some(result) = futures_lite::future::poll_once(pending.as_mut()).await {
+            return Err(Error::State(format!(
+                "chunk abort completed before original hook: success={}",
+                result.is_ok()
+            )));
+        }
+        if model.launch_seen.load(Ordering::Acquire) != 0 {
+            break;
+        }
+        futures_lite::future::yield_now().await;
+    }
+    drop(pending);
+    let handles = model.launch_seen.load(Ordering::Acquire);
+    let quarantined = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    assert!(handles > 0 && quarantined >= old_quarantine + handles);
+    assert!(model.service.ensure_usable().is_err());
+    assert!(
+        ready
+            .commit(&mut native.iter_mut().collect::<Vec<_>>())
+            .is_err()
+    );
+    unchanged(&native, &before)?;
+    assert!(chunk(&model, &native, &teachers).await.is_err());
+    assert!(native[0].prepare(&[11], false).await.is_err());
+    assert!(
+        ResidentService::new(
+            Device::Cuda(cubecl::device::CudaDevice::new(0)),
+            source.source_identity(),
+            ResourceOwner::new("other-model", "after-chunk-abort")?,
+            1024 * 1024,
+            DEVICE_CAP
+        )
+        .is_err()
+    );
+    client.sync().await.map_err(|error| {
+        Error::State(format!(
+            "chunk abort independent prior readback barrier:{error}"
+        ))
+    })?;
+    for (snapshot, expected) in saved.iter().zip(&domains) {
+        physical_prior_after_abort(&client, snapshot, expected).await?;
+    }
+    unchanged(&native, &before)?;
+    let service = Arc::clone(&model.service);
+    let key = service.physical_key();
+    drop(saved);
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    drop(model);
+    let budget = ResourceArbiter::process_shared().snapshot(&key)?;
+    assert!(budget.fenced);
+    assert_eq!(budget.reserved, PROVIDER as u64);
+    assert!(service.ensure_usable().is_err());
+    assert!(crate::physical_quarantine::quarantined_handles_for(&client)? >= quarantined);
+    println!(
+        "P12_NATIVE_TEACHER_CHUNK_ABORT_PASS source={name} B=2 T=4 actual_original_queued_chunk=true one_terminal_Ready_not_reached=true all42_prior_bits_exact=true original_handles={handles} quarantine_retained=true service_fenced=true recovery=false"
+    );
+    Ok(())
+}
+#[cfg(feature = "hardware-abort-probe")]
+#[test]
+#[ignore = "unknown completion fences actual card; ROOT separate --release process"]
+fn qwen08_chunk_abort() {
+    futures_lite::future::block_on(qualify_abort("qwen35-0.8b")).unwrap();
+}
+#[cfg(feature = "hardware-abort-probe")]
+#[test]
+#[ignore = "unknown completion fences actual card; ROOT separate --release process"]
+fn qwen2_chunk_abort() {
+    futures_lite::future::block_on(qualify_abort("qwen35-2b")).unwrap();
+}
+
+#[cfg(feature = "portable-checkpoint")]
+async fn qualify_boundary(name: &str, steps: usize) -> Result<()> {
+    let (source, model, _) = load(name).await?;
+    for (scope, positions) in [
+        ("cross256", [257 - steps, 258 - steps]),
+        ("capacity512", [512 - steps, 512 - steps]),
+    ] {
+        let mut cpu = Vec::new();
+        let mut single = Vec::new();
+        let mut native = Vec::new();
+        let mut teachers = Vec::new();
+        for (row, frontier) in positions.into_iter().enumerate() {
+            let stream = (0..512)
+                .map(|position| IDS[(position + row * 3) % IDS.len()])
+                .collect::<Vec<_>>();
+            let mut reference = QwenDenseCpu::from_shared(Arc::clone(&source))?;
+            reference.prefill_chunk_without_logits(&stream[..frontier])?;
+            let checkpoint = reference.checkpoint()?;
+            let domains = reference.snapshot().domain_values();
+            let rebuilt = source.checkpoint_from_domains(frontier, &domains)?;
+            compare_domains(&domains, &checkpoint.domain_values())?;
+            compare_domains(&domains, &rebuilt.domain_values())?;
+            let mut batch = model.session().await?;
+            let mut b1 = model.session().await?;
+            batch.import_checkpoint(&rebuilt).await?;
+            b1.import_checkpoint(&checkpoint).await?;
+            compare_domains(&domains, &batch.snapshot().read_domains().await?)?;
+            compare_domains(&domains, &b1.snapshot().read_domains().await?)?;
+            teachers.push(stream[frontier..frontier + steps].to_vec());
+            cpu.push(reference);
+            single.push(b1);
+            native.push(batch);
+        }
+        let before = bindings(&native);
+        let uploads = model.service.uploaded_value_bytes();
+        advance_oracle(&mut single, &mut cpu, &teachers).await?;
+        let candidate = chunk(&model, &native, &teachers).await?.unwrap();
+        state_receipt(&candidate, &before);
+        let dispatches = candidate.dispatches();
+        let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+        chunk_reads(&reads, 2, steps, dispatches);
+        state(&native, &cpu).await?;
+        state(&single, &cpu).await?;
+        for (session, old) in native.iter().zip(&before) {
+            assert_eq!(session.binding().frontier(), old.frontier() + steps as u64);
+            assert_eq!(session.binding().root(), old.root() + 1);
+        }
+        assert_eq!(model.service.uploaded_value_bytes(), uploads);
+        if scope == "cross256" {
+            future(&model, &mut native, &mut single, &mut cpu).await?;
+        } else {
+            assert!(
+                native
+                    .iter()
+                    .all(|session| session.binding().frontier() == 512)
+            );
+            let before = bindings(&native);
+            let domains = domain_values(&native).await?;
+            let queue = model.service.queue().stats()?;
+            let bytes = model.service.resident_bytes()?;
+            assert!(chunk(&model, &native, &teachers).await.is_err());
+            unchanged(&native, &before)?;
+            unchanged_domains(&native, &domains).await?;
+            assert_eq!(model.service.queue().stats()?, queue);
+            assert_eq!(model.service.resident_bytes()?, bytes);
+        }
+        println!(
+            "P12_NATIVE_TEACHER_CHUNK_BOUNDARY_SCOPE source={name} B=2 T={steps} scope={scope} actual_CPU_hydrated_prefix=true GPU_complete_prefill_claim=false each_teacher_CPU_B1_fullV_all42_exact=true native_terminal_all42_exact=true status_bytes=4"
+        );
+    }
+    reclaim(model).await?;
+    Ok(())
+}
+#[cfg(feature = "portable-checkpoint")]
+#[test]
+#[ignore = "source-real CPU hydrated CUDA boundary; ROOT --release --test-threads=1"]
+fn qwen08_chunk_boundary_t2() {
+    futures_lite::future::block_on(qualify_boundary("qwen35-0.8b", 2)).unwrap();
+}
+#[cfg(feature = "portable-checkpoint")]
+#[test]
+#[ignore = "source-real CPU hydrated CUDA boundary; ROOT --release --test-threads=1"]
+fn qwen08_chunk_boundary_t4() {
+    futures_lite::future::block_on(qualify_boundary("qwen35-0.8b", 4)).unwrap();
+}
+#[cfg(feature = "portable-checkpoint")]
+#[test]
+#[ignore = "source-real CPU hydrated CUDA boundary; ROOT --release --test-threads=1"]
+fn qwen2_chunk_boundary_t2() {
+    futures_lite::future::block_on(qualify_boundary("qwen35-2b", 2)).unwrap();
+}
+#[cfg(feature = "portable-checkpoint")]
+#[test]
+#[ignore = "source-real CPU hydrated CUDA boundary; ROOT --release --test-threads=1"]
+fn qwen2_chunk_boundary_t4() {
+    futures_lite::future::block_on(qualify_boundary("qwen35-2b", 4)).unwrap();
+}
+
+async fn qualify_certificate_refusal(name: &str) -> Result<()> {
+    let (source, model, eos) = load(name).await?;
+    let (mut native, single, cpu) = oracle(&source, &model, 2).await?;
+    let teachers = vec![vec![11, eos, 55, 101], vec![353, eos, 78, 135]];
+    let before = bindings(&native);
+    let saved = native
+        .iter()
+        .map(ResidentSession::snapshot)
+        .collect::<Vec<_>>();
+    let domains = domain_values(&native).await?;
+    let ready = chunk(&model, &native, &teachers).await?.unwrap();
+    state_receipt(&ready, &before);
+    let client = model.service.client().clone();
+    let old_quarantine = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    // Synthetic one-shot NaN in intermediate inactive KV only. The following
+    // teacher erases it; an omitted intermediate certificate/reset OR would pass.
+    BatchProgram::arm_chunk_inactive_nan_probe();
+    assert!(
+        matches!(chunk(&model,&native,&teachers).await,Err(Error::State(ref message))if message.contains("status refused"))
+    );
+    let quarantined = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    assert!(quarantined > old_quarantine);
+    assert!(model.service.ensure_usable().is_err());
+    unchanged(&native, &before)?;
+    assert!(
+        ready
+            .commit(&mut native.iter_mut().collect::<Vec<_>>())
+            .is_err()
+    );
+    // Read only the original retained handles after the actual failed readback.
+    // This diagnostic does not promote an event, make Ready, recover or publish.
+    let (status, padding) = {
+        let workspace = model.workspace.lock().await;
+        let program = &workspace.batch.as_ref().unwrap().program;
+        (program.status(), program.chunk_even_final_padding_view()?)
+    };
+    let raw = client
+        .read_async(vec![status, padding])
+        .await
+        .map_err(|error| Error::State(format!("chunk certificate raw proof:{error}")))?;
+    assert_eq!(raw.len(), 2);
+    assert_eq!(raw[0].len(), 4);
+    assert_eq!(raw[1].len(), 4);
+    assert_eq!(
+        u32::from_ne_bytes(raw[0][..].try_into().unwrap()),
+        crate::program_kernels::NONFINITE_VALUE
+    );
+    assert_eq!(
+        u32::from_ne_bytes(raw[1][..].try_into().unwrap()),
+        0,
+        "later append must heal final padding; rejection proves earlier certificate latch"
+    );
+    for (snapshot, expected) in saved.iter().zip(&domains) {
+        physical_prior_after_abort(&client, snapshot, expected).await?;
+    }
+    assert!(chunk(&model, &native, &teachers).await.is_err());
+    unchanged(&native, &before)?;
+    let service = Arc::clone(&model.service);
+    let key = service.physical_key();
+    drop(raw);
+    drop(saved);
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    drop(model);
+    let budget = ResourceArbiter::process_shared().snapshot(&key)?;
+    assert!(budget.fenced);
+    assert_eq!(budget.reserved, PROVIDER as u64);
+    assert!(service.ensure_usable().is_err());
+    assert!(crate::physical_quarantine::quarantined_handles_for(&client)? >= quarantined);
+    println!(
+        "P12_NATIVE_TEACHER_CHUNK_CERTIFICATE_REFUSAL source={name} B=2 T=4 synthetic_intermediate_inactive_KV_nan=true final_padding_raw_positive_zero=true actual_latched_status=2 original_roots_all42_unchanged=true quarantine_retained=true service_fenced=true recovery=false"
+    );
+    Ok(())
+}
+#[test]
+#[ignore = "actual numerical-status refusal fences card; ROOT fresh --release process"]
+fn qwen08_chunk_certificate_refusal() {
+    futures_lite::future::block_on(qualify_certificate_refusal("qwen35-0.8b")).unwrap();
+}
+#[test]
+#[ignore = "actual numerical-status refusal fences card; ROOT fresh --release process"]
+fn qwen2_chunk_certificate_refusal() {
+    futures_lite::future::block_on(qualify_certificate_refusal("qwen35-2b")).unwrap();
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/batch/chunk.rs
@@ -0,0 +1,307 @@
+//! Qualifier-only sequential original state prefix with one gather/scatter/Ready.
+use super::*;
+
+#[cfg(test)]
+std::thread_local! {
+    static INACTIVE_NAN_PROBE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
+}
+
+/// RAII restores persistent workspace metadata on every ordinary error/unwind.
+/// Both original physical banks remain leased by the enclosing one submission.
+struct PackedOrientation<'a> {
+    program: &'a mut BatchProgram,
+    flipped: bool,
+}
+impl PackedOrientation<'_> {
+    fn flip(&mut self) {
+        for state in &mut self.program.packed {
+            std::mem::swap(&mut state.prior, &mut state.candidate);
+        }
+        self.flipped = !self.flipped;
+    }
+}
+impl Drop for PackedOrientation<'_> {
+    fn drop(&mut self) {
+        if self.flipped {
+            self.flip();
+        }
+    }
+}
+
+impl BatchProgram {
+    /// Synthetic one-shot proof only: poison an inactive intermediate KV cell
+    /// after teacher zero. Teacher one overwrites padding back to canonical zero;
+    /// the real original intermediate certificate must keep the attempt refused.
+    #[cfg(test)]
+    pub(crate) fn arm_chunk_inactive_nan_probe() {
+        INACTIVE_NAN_PROBE
+            .with(|flag| assert!(!flag.replace(true), "chunk fault probe already armed"));
+    }
+
+    /// Diagnostic raw view only, after an even-T refusal and restored metadata.
+    /// Caller retains the original program/lease and cannot obtain a Ready here.
+    #[cfg(test)]
+    pub(crate) fn chunk_even_final_padding_view(&self) -> Result<Handle> {
+        let keys = self
+            .packed
+            .iter()
+            .find_map(|state| match &state.prior {
+                StateBuffers::Kv { keys, .. } => Some(keys),
+                _ => None,
+            })
+            .ok_or_else(|| Error::State("chunk padding proof requires KV".into()))?;
+        let offset = keys
+            .elements
+            .checked_sub(1)
+            .and_then(|n| n.checked_mul(4))
+            .ok_or_else(|| Error::Limit("chunk padding proof offset".into()))?;
+        let start = keys
+            .handle
+            .offset_start
+            .unwrap_or(0)
+            .checked_add(offset as u64)
+            .ok_or_else(|| Error::Limit("chunk padding proof start".into()))?;
+        let end = start
+            .checked_add(4)
+            .ok_or_else(|| Error::Limit("chunk padding proof end".into()))?;
+        let mut view = keys.handle.clone();
+        view.offset_start = Some(start);
+        view.offset_end = Some(
+            view.size()
+                .checked_sub(end)
+                .ok_or_else(|| Error::State("chunk padding proof extent".into()))?,
+        );
+        BufferRange::new(&view)?;
+        Ok(view)
+    }
+
+    /// Entire temporal span, original first priors, and exact final scatter bank
+    /// are cold checked before original submission. No future Ready is invented.
+    pub(crate) fn validate_state_chunk(
+        &self,
+        client: &Client,
+        steps: &[Vec<RuntimeParameters>],
+        bindings: &[Vec<StateBinding>],
+    ) -> Result<()> {
+        if !matches!(steps.len(), 2 | 4) || self.graph_pinned() {
+            return Err(Error::Unsupported(
+                "teacher chunk T2/T4 requires unpinned native workspace".into(),
+            ));
+        }
+        let first = &steps[0];
+        self.validate_runtime(client, first, bindings)?;
+        for (step, parameters) in steps.iter().enumerate() {
+            if parameters.len() != self.rows {
+                return Err(Error::Malformed("native teacher chunk row coverage".into()));
+            }
+            for (row, parameter) in parameters.iter().enumerate() {
+                if first[row].position.checked_add(step) != Some(parameter.position) {
+                    return Err(Error::State(
+                        "native teacher chunk position progression".into(),
+                    ));
+                }
+                // Runtime validation here is physical geometry only. The actual
+                // future prior is produced into an owned packed bank in-stream.
+                let mut shape = bindings[row].clone();
+                for state in &mut shape {
+                    state.prior_tokens = parameter.position;
+                }
+                self.inner.validate_runtime(client, *parameter, &shape)?;
+                for node in self.inner.plan.nodes() {
+                    if let PlanOperation::Embed { tensor, .. } = &node.operation
+                        && parameter.token as usize >= tensor.rows
+                    {
+                        return Err(Error::Malformed(
+                            "teacher chunk token exceeds source vocabulary".into(),
+                        ));
+                    }
+                }
+            }
+        }
+        // T2/T4 finish in the originally prior packed bank. Validate its exact
+        // certified scatter bindings before any kernel is submitted.
+        self.prepare_state_transports_for_bank(
+            client,
+            bindings,
+            true,
+            steps.len().is_multiple_of(2),
+        )?;
+        Ok(())
+    }
+
+    pub(crate) fn launch_state_chunk(
+        &mut self,
+        client: &Client,
+        steps: &[Vec<RuntimeParameters>],
+        bindings: &[Vec<StateBinding>],
+    ) -> Result<ResidentSubmission> {
+        self.validate_state_chunk(client, steps, bindings)?;
+        let mut orientation = PackedOrientation {
+            program: self,
+            flipped: false,
+        };
+        // Only this original entry clears status; later teachers preserve its OR.
+        let prefix = orientation
+            .program
+            .launch_inputs(client, &steps[0], bindings, None)?;
+        let mut body = BatchBodyReceipt {
+            dispatches: 0,
+            fused_dispatches: 0,
+        };
+        for (step, parameters) in steps.iter().enumerate() {
+            if step > 0 {
+                body.dispatches += orientation
+                    .program
+                    .stage_teacher_parameters(client, parameters);
+            }
+            let current =
+                orientation
+                    .program
+                    .launch_body(client, parameters, ResidentOutput::StateOnly)?;
+            body.dispatches += current.dispatches;
+            body.fused_dispatches += current.fused_dispatches;
+            #[cfg(test)]
+            if step == 0 && INACTIVE_NAN_PROBE.with(|flag| flag.replace(false)) {
+                let keys = orientation
+                    .program
+                    .packed
+                    .iter()
+                    .find_map(|state| match &state.candidate {
+                        StateBuffers::Kv { keys, .. } => Some(keys),
+                        _ => None,
+                    })
+                    .ok_or_else(|| {
+                        Error::State("chunk fault probe requires an actual KV domain".into())
+                    })?;
+                if steps.iter().flatten().any(|parameter| {
+                    parameter.valid_tokens >= orientation.program.inner.plan.prefix_capacity()
+                }) {
+                    return Err(Error::State(
+                        "chunk fault probe requires untouched capacity padding".into(),
+                    ));
+                }
+                let byte_offset = keys
+                    .elements
+                    .checked_sub(1)
+                    .and_then(|n| n.checked_mul(4))
+                    .ok_or_else(|| Error::Limit("chunk fault probe offset".into()))?;
+                let start = keys
+                    .handle
+                    .offset_start
+                    .unwrap_or(0)
+                    .checked_add(byte_offset as u64)
+                    .ok_or_else(|| Error::Limit("chunk fault probe view start".into()))?;
+                let end = start
+                    .checked_add(4)
+                    .ok_or_else(|| Error::Limit("chunk fault probe view end".into()))?;
+                let mut view = keys.handle.clone();
+                view.offset_start = Some(start);
+                view.offset_end = Some(
+                    view.size()
+                        .checked_sub(end)
+                        .ok_or_else(|| Error::State("chunk fault probe range".into()))?,
+                );
+                BufferRange::new(&view)?;
+                unsafe {
+                    batch_tokens::launch_unchecked(
+                        client,
+                        grid(1),
+                        threads(),
+                        arg(&view, 1),
+                        0x7fc00001u32,
+                        0,
+                        0,
+                        0,
+                        0,
+                        0,
+                        0,
+                        0,
+                        1,
+                    );
+                }
+                body.dispatches += 1;
+            }
+            if step + 1 < steps.len() {
+                // Every original candidate word (including capacity KV padding)
+                // is certified before a packed bank is consumed/reused. This is
+                // the same finite predicate and persistent OR as final scatter.
+                for state in &orientation.program.packed {
+                    match &state.candidate {
+                        StateBuffers::Vector(value) => {
+                            body.dispatches +=
+                                orientation
+                                    .program
+                                    .certify(client, &value.handle, value.elements);
+                        }
+                        StateBuffers::Kv { keys, values } => {
+                            body.dispatches +=
+                                orientation
+                                    .program
+                                    .certify(client, &keys.handle, keys.elements);
+                            body.dispatches += orientation.program.certify(
+                                client,
+                                &values.handle,
+                                values.elements,
+                            );
+                        }
+                    }
+                }
+                orientation.flip();
+            }
+        }
+        // Exactly one final certified scatter into the independent candidates.
+        // Original completion/publisher remains the enclosing physical owner's job.
+        orientation.program.launch_outputs(
+            client,
+            bindings,
+            ResidentOutput::StateOnly,
+            prefix,
+            body,
+        )
+    }
+
+    /// Same two scalar-token/position kernels as original launch_inputs; no
+    /// clear, gather, upload, device-token feedback, sampler or graph enqueue.
+    fn stage_teacher_parameters(&self, client: &Client, parameters: &[RuntimeParameters]) -> usize {
+        let mut tokens = [0u32; 8];
+        let mut positions = [0u32; 8];
+        for ((token, position), parameter) in tokens.iter_mut().zip(&mut positions).zip(parameters)
+        {
+            *token = parameter.token;
+            *position = parameter.position as u32;
+        }
+        unsafe {
+            batch_tokens::launch_unchecked(
+                client,
+                grid(self.rows),
+                threads(),
+                arg(&self.inner.token, self.rows),
+                tokens[0],
+                tokens[1],
+                tokens[2],
+                tokens[3],
+                tokens[4],
+                tokens[5],
+                tokens[6],
+                tokens[7],
+                self.rows,
+            );
+            batch_tokens::launch_unchecked(
+                client,
+                grid(self.rows),
+                threads(),
+                arg(&self.positions, self.rows),
+                positions[0],
+                positions[1],
+                positions[2],
+                positions[3],
+                positions[4],
+                positions[5],
+                positions[6],
+                positions[7],
+                self.rows,
+            );
+        }
+        2
+    }
+}
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -61,6 +61,9 @@
 pub(crate) use graph::{BatchGraphProfile, BatchGraphReplay};
 #[cfg(all(test, feature = "qualification", feature = "cuda"))]
 mod state_transport_qualification;
+
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+mod chunk;
 
 impl BatchProgram {
     /// New allocations only: the original source handles remain shared.
@@ -719,12 +722,22 @@
         bindings: &[Vec<StateBinding>],
         scatter: bool,
     ) -> Result<Vec<PreparedStateTransport>> {
+        self.prepare_state_transports_for_bank(client, bindings, scatter, false)
+    }
+
+    fn prepare_state_transports_for_bank(
+        &self,
+        client: &Client,
+        bindings: &[Vec<StateBinding>],
+        scatter: bool,
+        swap_banks: bool,
+    ) -> Result<Vec<PreparedStateTransport>> {
         if bindings.len() != self.rows || self.packed.is_empty() || self.packed.len() > 256 {
             return Err(Error::State("state transport domain/row coverage".into()));
         }
         let mut prepared = Vec::with_capacity(self.packed.len());
         for packed in &self.packed {
-            let dense = if scatter {
+            let dense = if scatter != swap_banks {
                 &packed.candidate
             } else {
                 &packed.prior

````

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/ROOT_WIRING.patch

Original UTF-8 bytes: 327; SHA256 `31560bb5d9cbd084eaf76c77e7862c7ea28489dcd04246e2d84c3d66ffdf31a2`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_inference/prefix_qualification.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/prefix_qualification.rs
@@ -577,3 +577,5 @@
 fn original_qwen35_2b_native_prefix_pressure() -> Result<()> {
     futures_lite::future::block_on(qualify_prefix_pressure("qwen35-2b"))
 }
+
+mod chunk;

````

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/frozen/NEW_DELTA.patch

Original UTF-8 bytes: 58414; SHA256 `56845b51e9170304c5355e2fa96dbf9289d0c152adedab9d7602f771bfd1c595`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_batch/chunk.rs
@@ -0,0 +1,186 @@
+//! Closed teacher chunk metadata, never a root issuer or execution authority.
+use super::*;
+
+#[derive(Clone, Debug)]
+pub(crate) struct ResidentBatchChunk {
+    roots: ResidentBatchRows,
+    teachers: Vec<Vec<u32>>,
+    vocabulary: u64,
+    capacity: usize,
+}
+impl ResidentBatchChunk {
+    pub(crate) fn new(
+        bindings: &[RootBinding],
+        teachers: &[&[u32]],
+        vocabulary: u64,
+        capacity: usize,
+    ) -> Result<Self> {
+        if !matches!(bindings.len(), 2 | 4 | 8)
+            || teachers.len() != bindings.len()
+            || vocabulary == 0
+            || vocabulary > u64::from(u32::MAX) + 1
+            || capacity == 0
+            || capacity > MAX_PREFIX as usize
+        {
+            return Err(Error::Unsupported(
+                "native teacher chunk source/band/capacity".into(),
+            ));
+        }
+        let count = teachers.first().map_or(0, |row| row.len());
+        if !matches!(count, 2 | 4) || teachers.iter().any(|row| row.len() != count) {
+            return Err(Error::Unsupported(
+                "native teacher chunk requires equal T2/T4 rows".into(),
+            ));
+        }
+        let roots = ResidentBatchRows::new(bindings, 1)?;
+        if roots.frontiers().any(|position| {
+            position
+                .checked_add(count as u64)
+                .is_none_or(|end| end > capacity as u64)
+        }) {
+            return Err(Error::Limit(
+                "native teacher chunk exceeds original context".into(),
+            ));
+        }
+        if teachers
+            .iter()
+            .flat_map(|row| row.iter())
+            .any(|&token| u64::from(token) >= vocabulary)
+        {
+            return Err(Error::Malformed(
+                "native teacher chunk contains invalid source token".into(),
+            ));
+        }
+        Ok(Self {
+            roots,
+            teachers: teachers.iter().map(|row| row.to_vec()).collect(),
+            vocabulary,
+            capacity,
+        })
+    }
+    pub(crate) fn tokens_per_row(&self) -> usize {
+        self.teachers[0].len()
+    }
+    pub(crate) fn rows(&self) -> usize {
+        self.teachers.len()
+    }
+    pub(crate) fn vocabulary(&self) -> u64 {
+        self.vocabulary
+    }
+    pub(crate) fn capacity(&self) -> usize {
+        self.capacity
+    }
+    pub(crate) fn require_current(&self, bindings: &[RootBinding]) -> Result<()> {
+        self.roots.require_current(bindings)
+    }
+    pub(crate) fn token(&self, row: usize, step: usize) -> Result<u32> {
+        self.teachers
+            .get(row)
+            .and_then(|tokens| tokens.get(step))
+            .copied()
+            .ok_or_else(|| Error::State("native teacher chunk row/step changed".into()))
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use zerogravity_core::session_root::SessionAuthority;
+
+    #[test]
+    fn equal_teacher_lengths_preserve_independent_frontiers_and_one_root_advance() {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        let mut owners = (0..8)
+            .map(|_| SessionAuthority::new(source).unwrap())
+            .collect::<Vec<_>>();
+        for (owner, position) in owners.iter_mut().zip([0, 1, 30, 255, 256, 508, 100, 17]) {
+            if position > 0 {
+                let attempt = owner.begin_advance(position).unwrap();
+                owner.prepare_publish(attempt).unwrap().commit();
+            }
+        }
+        for rows in [2, 4, 8] {
+            for count in [2, 4] {
+                let old = owners
+                    .iter()
+                    .take(rows)
+                    .map(SessionAuthority::binding)
+                    .collect::<Vec<_>>();
+                let teacher = (0..rows)
+                    .map(|row| {
+                        (0..count)
+                            .map(|step| (row * 17 + step + 11) as u32)
+                            .collect::<Vec<_>>()
+                    })
+                    .collect::<Vec<_>>();
+                let refs = teacher.iter().map(Vec::as_slice).collect::<Vec<_>>();
+                let chunk = ResidentBatchChunk::new(&old, &refs, 248320, 512).unwrap();
+                assert_eq!(chunk.rows(), rows);
+                assert_eq!(chunk.tokens_per_row(), count);
+                for (row, tokens) in teacher.iter().enumerate() {
+                    for (step, &token) in tokens.iter().enumerate() {
+                        assert_eq!(chunk.token(row, step).unwrap(), token);
+                    }
+                }
+                chunk.require_current(&old).unwrap();
+                assert!(chunk.token(rows, 0).is_err());
+                assert!(chunk.token(0, count).is_err());
+                let attempt = owners[0].begin_advance(count as u64).unwrap();
+                let published = owners[0].prepare_publish(attempt).unwrap().commit();
+                assert_eq!(published.frontier(), old[0].frontier() + count as u64);
+                assert_eq!(published.root(), old[0].root() + 1);
+                assert!(
+                    chunk
+                        .require_current(
+                            &owners
+                                .iter()
+                                .take(rows)
+                                .map(SessionAuthority::binding)
+                                .collect::<Vec<_>>()
+                        )
+                        .is_err()
+                );
+            }
+        }
+    }
+
+    #[test]
+    fn cold_refusals_never_mutate_roots_or_invent_padding() {
+        let source = SourceIdentity::for_new_allocation().unwrap();
+        let a = SessionAuthority::new(source).unwrap();
+        let b = SessionAuthority::new(source).unwrap();
+        let mut exhausted = SessionAuthority::new(source).unwrap();
+        let advance = exhausted.begin_advance(511).unwrap();
+        exhausted.prepare_publish(advance).unwrap().commit();
+        let roots = [a.binding(), b.binding()];
+        let original = roots;
+        for teachers in [
+            vec![&[11u32][..], &[353u32][..]],
+            vec![&[11, 12][..], &[353, 354, 355, 356][..]],
+            vec![&[11, 248320][..], &[353, 354][..]],
+        ] {
+            assert!(ResidentBatchChunk::new(&roots, &teachers, 248320, 512).is_err());
+        }
+        assert!(
+            ResidentBatchChunk::new(
+                &[roots[0], roots[0]],
+                &[&[11, 12], &[353, 354]],
+                248320,
+                512
+            )
+            .is_err()
+        );
+        assert!(
+            ResidentBatchChunk::new(
+                &[roots[0], exhausted.binding()],
+                &[&[11, 12], &[353, 354]],
+                248320,
+                512
+            )
+            .is_err()
+        );
+        assert!(ResidentBatchChunk::new(&roots, &[&[11, 12], &[353, 354]], 0, 512).is_err());
+        assert!(ResidentBatchChunk::new(&roots, &[&[11, 12], &[353, 354]], 248320, 513).is_err());
+        assert_eq!([a.binding(), b.binding()], original);
+    }
+}
--- a/crates/zerogravity-gpu/src/resident_batch.rs
+++ b/crates/zerogravity-gpu/src/resident_batch.rs
@@ -225,3 +225,8 @@
         }
     }
 }
+
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+mod chunk;
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+pub(crate) use chunk::ResidentBatchChunk;
--- a/crates/zerogravity-gpu/src/resident_inference/fused.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/fused.rs
@@ -3,6 +3,8 @@
 
 pub(super) enum FusedExecution {
     Native,
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    NativeChunk(crate::resident_batch::ResidentBatchChunk),
     #[cfg(all(feature = "qualification", feature = "cuda"))]
     Graph(Arc<super::graph::CapturedResidentGraph>),
 }
@@ -58,6 +60,39 @@
             .await
     }
 
+    /// Closed qualification-only teacher chunk. Each row consumes the same T2/T4
+    /// length with independent positions, one terminal Ready and one root advance.
+    /// Callers must reserve the prompt tail; no packet/EOS/output is presented.
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    pub async fn prepare_state_fused_chunk_admitted(
+        self: &Arc<Self>,
+        rows: &[(&ResidentSession, &[u32])],
+    ) -> Result<Option<PreparedResidentCohort>> {
+        let bases = rows
+            .iter()
+            .map(|(session, _)| session.binding())
+            .collect::<Vec<_>>();
+        let teachers = rows.iter().map(|(_, tokens)| *tokens).collect::<Vec<_>>();
+        let chunk = crate::resident_batch::ResidentBatchChunk::new(
+            &bases,
+            &teachers,
+            self.vocab,
+            self.capacity,
+        )?;
+        let first = rows
+            .iter()
+            .enumerate()
+            .map(|(row, (session, _))| Ok((*session, chunk.token(row, 0)?)))
+            .collect::<Result<Vec<_>>>()?;
+        self.prepare_fused_execution(
+            &first,
+            None,
+            Readout::None,
+            FusedExecution::NativeChunk(chunk),
+        )
+        .await
+    }
+
     /// One native full-model batch and source-exact device greedy selection.
     /// Each independent output packet gains its producer only at cohort commit.
     pub async fn prepare_greedy_fused_cohort(
@@ -142,6 +177,21 @@
             // attempts, candidates, packet/workspace admission or queue work.
             graph_readout(rows.len(), graph.output(), readout, device_tokens.is_some())?;
         }
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        if matches!(&execution, FusedExecution::NativeChunk(_))
+            && (!matches!(readout, Readout::None) || device_tokens.is_some())
+        {
+            return Err(Error::Unsupported(
+                "teacher chunk requires state-only host tokens".into(),
+            ));
+        }
+        let tokens_per_row = match &execution {
+            FusedExecution::Native => 1,
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            FusedExecution::Graph(_) => 1,
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            FusedExecution::NativeChunk(chunk) => chunk.tokens_per_row(),
+        };
         let output = if matches!(readout, Readout::None) {
             ResidentOutput::StateOnly
         } else {
@@ -154,6 +204,23 @@
         }
         let bases = rows.iter().map(|(s, _)| s.binding()).collect::<Vec<_>>();
         let roots = ResidentBatchRows::new(&bases, 1)?;
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        if let FusedExecution::NativeChunk(chunk) = &execution {
+            chunk.require_current(&bases)?;
+            for (row, (_, token)) in rows.iter().enumerate() {
+                if chunk.token(row, 0)? != *token {
+                    return Err(Error::State(
+                        "teacher chunk first source token changed".into(),
+                    ));
+                }
+            }
+            if chunk.rows() != rows.len()
+                || chunk.vocabulary() != self.vocab
+                || chunk.capacity() != self.capacity
+            {
+                return Err(Error::State("teacher chunk source geometry changed".into()));
+            }
+        }
         if !matches!(rows.len(), 2 | 4 | 8) {
             return Err(Error::Unsupported(
                 "resident native batch rows must be 2/4/8".into(),
@@ -188,7 +255,11 @@
                     .map_err(|_| Error::Limit("native resident batch frontier overflow".into()))
             })
             .collect::<Result<Vec<_>>>()?;
-        if positions.iter().any(|&position| position >= self.capacity) {
+        if positions.iter().any(|&position| {
+            position
+                .checked_add(tokens_per_row)
+                .is_none_or(|end| end > self.capacity)
+        }) {
             return Err(Error::Limit(
                 "native resident batch capacity exhausted".into(),
             ));
@@ -205,7 +276,7 @@
                 ));
             }
             self.validate_domains(&session.domains, true)?;
-            attempts.push(session.authority.begin_advance(1)?);
+            attempts.push(session.authority.begin_advance(tokens_per_row as u64)?);
         }
         let staging_bytes = self
             .state_bytes
@@ -223,6 +294,17 @@
         {
             let mut workspace = self.workspace.lock().await;
             self.service.ensure_usable()?;
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            if matches!(&execution, FusedExecution::NativeChunk(_))
+                && workspace
+                    .batch
+                    .as_ref()
+                    .is_some_and(|batch| batch.program.graph_pinned())
+            {
+                return Err(Error::Unsupported(
+                    "teacher chunk refuses a graph-pinned workspace before allocation".into(),
+                ));
+            }
             let same_band = workspace
                 .batch
                 .as_ref()
@@ -309,6 +391,31 @@
                 valid_tokens: position + 1,
             })
             .collect::<Vec<_>>();
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        let chunk_parameters = if let FusedExecution::NativeChunk(chunk) = &execution {
+            Some(
+                (0..tokens_per_row)
+                    .map(|step| {
+                        positions
+                            .iter()
+                            .enumerate()
+                            .map(|(row, &position)| {
+                                let position = position.checked_add(step).ok_or_else(|| {
+                                    Error::Limit("teacher chunk position overflow".into())
+                                })?;
+                                Ok(RuntimeParameters {
+                                    token: chunk.token(row, step)?,
+                                    position,
+                                    valid_tokens: position + 1,
+                                })
+                            })
+                            .collect::<Result<Vec<_>>>()
+                    })
+                    .collect::<Result<Vec<_>>>()?,
+            )
+        } else {
+            None
+        };
         // State initialization uses allocation -> queue. Preserve that order
         // while admitting a cold batch workspace, before selecting a slot.
         let packets = if matches!(readout, Readout::Greedy) {
@@ -431,6 +538,12 @@
         batch
             .program
             .validate_runtime(self.service.client(), &parameters, &bindings)?;
+        #[cfg(all(feature = "qualification", feature = "cuda"))]
+        if let Some(steps) = &chunk_parameters {
+            batch
+                .program
+                .validate_state_chunk(self.service.client(), steps, &bindings)?;
+        }
         let device_buffers = device_tokens.map(|tokens| {
             tokens
                 .iter()
@@ -522,6 +635,14 @@
         let mut submitted = match &execution {
             #[cfg(all(feature = "qualification", feature = "cuda"))]
             FusedExecution::Graph(graph) => graph.launch(program, &parameters, &bindings)?,
+            #[cfg(all(feature = "qualification", feature = "cuda"))]
+            FusedExecution::NativeChunk(_) => program.launch_state_chunk(
+                self.service.client(),
+                chunk_parameters
+                    .as_ref()
+                    .ok_or_else(|| Error::State("teacher chunk parameters absent".into()))?,
+                &bindings,
+            )?,
             FusedExecution::Native => {
                 if let Some(tokens) = &device_buffers {
                     program.launch_device_tokens(
@@ -687,7 +808,11 @@
                     } else {
                         Vec::new()
                     },
-                    state_only_steps: usize::from(output == ResidentOutput::StateOnly),
+                    state_only_steps: if output == ResidentOutput::StateOnly {
+                        tokens_per_row
+                    } else {
+                        0
+                    },
                     logit_steps: usize::from(output == ResidentOutput::Logits),
                     bytes: row_bytes + usize::from(row == 0) * 4,
                     chunk_bytes: row_bytes + usize::from(row == 0) * 4,
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk.rs
@@ -0,0 +1,675 @@
+//! ROOT-only full-model temporal proof; original CPU/B1/root helpers unchanged.
+use super::*;
+
+async fn chunk(
+    model: &Arc<ResidentModel>,
+    sessions: &[ResidentSession],
+    teachers: &[Vec<u32>],
+) -> Result<Option<PreparedResidentCohort>> {
+    assert_eq!(sessions.len(), teachers.len());
+    let rows = sessions
+        .iter()
+        .zip(teachers)
+        .map(|(session, tokens)| (session, tokens.as_slice()))
+        .collect::<Vec<_>>();
+    model.prepare_state_fused_chunk_admitted(&rows).await
+}
+fn chunk_reads(reads: &[InferenceReadback], rows: usize, steps: usize, dispatches: usize) {
+    assert_eq!(reads.len(), rows);
+    assert_eq!(reads.iter().map(|read| read.bytes).sum::<usize>(), 4);
+    assert_eq!(reads.iter().map(|read| read.chunk_bytes).sum::<usize>(), 4);
+    assert_eq!(
+        reads.iter().map(|read| read.dispatches).sum::<usize>(),
+        dispatches
+    );
+    for read in reads {
+        assert_eq!(read.state_only_steps, steps);
+        assert_eq!(read.logit_steps, 0);
+        assert!(read.logits.is_empty() && read.device_token.is_none());
+    }
+}
+async fn oracle(
+    source: &Arc<LoadedQwenModel>,
+    model: &Arc<ResidentModel>,
+    rows: usize,
+) -> Result<(
+    Vec<ResidentSession>,
+    Vec<ResidentSession>,
+    Vec<QwenDenseCpu>,
+)> {
+    let mut native = Vec::new();
+    let mut single = Vec::new();
+    let mut cpu = Vec::new();
+    for row in 0..rows {
+        let mut reference = QwenDenseCpu::from_shared(Arc::clone(source))?;
+        let mut b1 = model.session().await?;
+        let mut lane = model.session().await?;
+        for step in 0..row % 3 + 1 {
+            let id = IDS[(row + step) % IDS.len()];
+            let expected = reference.forward(id)?;
+            exact(
+                &expected,
+                &b1.forward(&[id], false).await?.logits,
+                "chunk seed B1 fullV",
+            )?;
+            exact(
+                &expected,
+                &lane.forward(&[id], false).await?.logits,
+                "chunk seed native B1 fullV",
+            )?;
+        }
+        native.push(lane);
+        single.push(b1);
+        cpu.push(reference);
+    }
+    state(&native, &cpu).await?;
+    state(&single, &cpu).await?;
+    Ok((native, single, cpu))
+}
+async fn advance_oracle(
+    single: &mut [ResidentSession],
+    cpu: &mut [QwenDenseCpu],
+    teachers: &[Vec<u32>],
+) -> Result<()> {
+    assert_eq!(single.len(), teachers.len());
+    assert_eq!(cpu.len(), teachers.len());
+    for step in 0..teachers[0].len() {
+        for row in 0..teachers.len() {
+            let expected = cpu[row].forward(teachers[row][step])?;
+            assert_eq!(expected.len(), 248320);
+            exact(
+                &expected,
+                &single[row]
+                    .forward(&[teachers[row][step]], false)
+                    .await?
+                    .logits,
+                "chunk each original teacher CPU/B1 fullV",
+            )?;
+        }
+        // Every teacher has independent accepted complete-domain evidence, not
+        // a reconstructed intermediate state from the temporal implementation.
+        state(single, cpu).await?;
+    }
+    Ok(())
+}
+async fn future(
+    model: &Arc<ResidentModel>,
+    native: &mut [ResidentSession],
+    single: &mut [ResidentSession],
+    cpu: &mut [QwenDenseCpu],
+) -> Result<()> {
+    let rows = native.len();
+    let ids = (0..rows)
+        .map(|row| IDS[(row + 4) % IDS.len()])
+        .collect::<Vec<_>>();
+    let mut expected = Vec::new();
+    for row in 0..rows {
+        let logits = cpu[row].forward(ids[row])?;
+        exact(
+            &logits,
+            &single[row].forward(&[ids[row]], false).await?.logits,
+            "chunk later CPU/B1 fullV",
+        )?;
+        expected.push(logits);
+    }
+    let inputs = native
+        .iter()
+        .zip(&ids)
+        .map(|(session, &id)| (session, id))
+        .collect::<Vec<_>>();
+    let candidate = model.prepare_logits_fused_cohort(&inputs).await?;
+    let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+    assert_eq!(reads.len(), rows);
+    for (read, wanted) in reads.iter().zip(&expected) {
+        assert_eq!(read.logits.len(), 248320);
+        assert_eq!(read.state_only_steps, 0);
+        assert_eq!(read.logit_steps, 1);
+        exact(
+            wanted,
+            &read.logits,
+            "chunk future complete native vocabulary",
+        )?;
+    }
+    state(native, cpu).await?;
+    state(single, cpu).await?;
+    drop(reads);
+    drop(expected);
+    // Actual accepted B1 greedy packets, plus CPU-selected full logits, remain
+    // independent of the new temporal path and its unchanged native sampler.
+    let ids = (0..rows)
+        .map(|row| IDS[(row + 6) % IDS.len()])
+        .collect::<Vec<_>>();
+    let mut selected = Vec::new();
+    for row in 0..rows {
+        let logits = cpu[row].forward(ids[row])?;
+        let saved = single[row].snapshot();
+        exact(
+            &logits,
+            &single[row].forward(&[ids[row]], false).await?.logits,
+            "chunk greedy independent fullV",
+        )?;
+        let index = (0..logits.len())
+            .max_by(|a, b| logits[*a].total_cmp(&logits[*b]).then_with(|| b.cmp(a)))
+            .unwrap();
+        let wanted = (index as u32, logits[index].to_bits());
+        single[row].restore(&saved)?;
+        let read = single[row].forward_greedy(&[ids[row]]).await?;
+        let packet = read.device_token.as_ref().unwrap();
+        assert_eq!((packet.id, packet.max_bits), wanted);
+        selected.push(wanted);
+    }
+    let inputs = native
+        .iter()
+        .zip(&ids)
+        .map(|(session, &id)| (session, id))
+        .collect::<Vec<_>>();
+    let candidate = model.prepare_greedy_fused_cohort(&inputs).await?;
+    assert_eq!(candidate.native_greedy_dispatches(), 4);
+    let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+    assert_eq!(reads.len(), rows);
+    for ((read, session), wanted) in reads.iter().zip(native.iter()).zip(selected) {
+        let packet = read.device_token.as_ref().unwrap();
+        assert_eq!((packet.id, packet.max_bits), wanted);
+        assert_eq!(packet.producer, Some(session.binding()));
+    }
+    state(native, cpu).await?;
+    state(single, cpu).await?;
+    Ok(())
+}
+async fn qualify(name: &str, rows: usize, steps: usize) -> Result<()> {
+    let (source, model, eos) = load(name).await?;
+    let (mut native, mut single, mut cpu) = oracle(&source, &model, rows).await?;
+    let teachers = (0..rows)
+        .map(|row| {
+            (0..steps)
+                .map(|step| {
+                    if step == 1 {
+                        eos
+                    } else {
+                        IDS[(row + step + 2) % IDS.len()]
+                    }
+                })
+                .collect::<Vec<_>>()
+        })
+        .collect::<Vec<_>>();
+    let before = bindings(&native);
+    let domains = domain_values(&native).await?;
+    let queue = model.service.queue().stats()?;
+    let mut invalid = teachers.clone();
+    invalid[rows - 1][steps - 1] = u32::MAX;
+    assert!(chunk(&model, &native, &invalid).await.is_err());
+    let mut ragged = teachers.clone();
+    ragged[rows - 1].pop();
+    assert!(chunk(&model, &native, &ragged).await.is_err());
+    let duplicate = vec![(&native[0], teachers[0].as_slice()); rows];
+    assert!(
+        model
+            .prepare_state_fused_chunk_admitted(&duplicate)
+            .await
+            .is_err()
+    );
+    assert_eq!(model.service.queue().stats()?, queue);
+    unchanged(&native, &before)?;
+    unchanged_domains(&native, &domains).await?;
+    let uploads = model.service.uploaded_value_bytes();
+    let abandoned = chunk(&model, &native, &teachers)
+        .await?
+        .ok_or_else(|| Error::Limit("chunk qualifier unexpected capacity refusal".into()))?;
+    state_receipt(&abandoned, &before);
+    unchanged(&native, &before)?;
+    drop(abandoned);
+    unchanged_domains(&native, &domains).await?;
+    let warm_bytes = model.service.resident_bytes()?;
+    drop(chunk(&model, &native, &teachers).await?.unwrap());
+    idle(&model, warm_bytes)?;
+    let saved = native[rows - 1].snapshot();
+    let stale = chunk(&model, &native, &teachers).await?.unwrap();
+    native[rows - 1].restore(&saved)?;
+    let restored = bindings(&native);
+    assert!(
+        stale
+            .commit(&mut native.iter_mut().collect::<Vec<_>>())
+            .is_err()
+    );
+    unchanged(&native, &restored)?;
+    unchanged_domains(&native, &domains).await?;
+    drop(saved);
+    let before = bindings(&native);
+    let candidate = chunk(&model, &native, &teachers).await?.unwrap();
+    state_receipt(&candidate, &before);
+    let dispatches = candidate.dispatches();
+    let projections = candidate.fused_dispatches();
+    assert_eq!(projections % steps, 0);
+    unchanged(&native, &before)?;
+    advance_oracle(&mut single, &mut cpu, &teachers).await?;
+    let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+    chunk_reads(&reads, rows, steps, dispatches);
+    for (session, old) in native.iter().zip(&before) {
+        assert_eq!(session.binding().root(), old.root() + 1);
+        assert_eq!(session.binding().frontier(), old.frontier() + steps as u64);
+        assert_eq!(session.binding().session_epoch(), old.session_epoch());
+    }
+    state(&native, &cpu).await?;
+    state(&single, &cpu).await?;
+    assert_eq!(model.service.pending_events()?, 0);
+    assert_eq!(
+        model.service.uploaded_value_bytes(),
+        uploads,
+        "warm temporal chunk must not upload model/activation/state data"
+    );
+    drop(reads);
+    future(&model, &mut native, &mut single, &mut cpu).await?;
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    reclaim(model).await?;
+    println!(
+        "P12_NATIVE_TEACHER_CHUNK_PASS source={name} B={rows} T={steps} original_CPU_B1_each_teacher_fullV_all42=true native_final_all42_exact=true future_fullV_and_greedy_exact=true one_terminal_status_bytes=4 one_root_advance=true teacher_EOS_no_output=true original_kernel_dispatches={dispatches} projection_subset={projections} final_reserved=0"
+    );
+    Ok(())
+}
+
+macro_rules! case {
+    ($test:ident,$name:literal,$rows:literal,$steps:literal) => {
+        #[test]
+        #[ignore = "physical original CUDA teacher chunk; ROOT --release --test-threads=1"]
+        fn $test() {
+            futures_lite::future::block_on(qualify($name, $rows, $steps)).unwrap();
+        }
+    };
+}
+case!(qwen08_b2_t2, "qwen35-0.8b", 2, 2);
+case!(qwen08_b2_t4, "qwen35-0.8b", 2, 4);
+case!(qwen08_b4_t2, "qwen35-0.8b", 4, 2);
+case!(qwen08_b4_t4, "qwen35-0.8b", 4, 4);
+case!(qwen08_b8_t2, "qwen35-0.8b", 8, 2);
+case!(qwen08_b8_t4, "qwen35-0.8b", 8, 4);
+case!(qwen2_b2_t2, "qwen35-2b", 2, 2);
+case!(qwen2_b2_t4, "qwen35-2b", 2, 4);
+case!(qwen2_b4_t2, "qwen35-2b", 4, 2);
+case!(qwen2_b4_t4, "qwen35-2b", 4, 4);
+case!(qwen2_b8_t2, "qwen35-2b", 8, 2);
+case!(qwen2_b8_t4, "qwen35-2b", 8, 4);
+
+async fn pressure_advance_chunk(
+    model: &Arc<ResidentModel>,
+    native: &mut [ResidentSession],
+    single: &mut [ResidentSession],
+    cpu: &mut [QwenDenseCpu],
+    teachers: &[Vec<u32>],
+) -> Result<()> {
+    advance_oracle(single, cpu, teachers).await?;
+    let before = bindings(native);
+    let candidate = chunk(model, native, teachers)
+        .await?
+        .ok_or_else(|| Error::Limit("explicit smaller chunk refused".into()))?;
+    state_receipt(&candidate, &before);
+    let dispatches = candidate.dispatches();
+    let rows = native.len();
+    let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+    chunk_reads(&reads, rows, teachers[0].len(), dispatches);
+    state(native, cpu).await?;
+    state(single, cpu).await?;
+    Ok(())
+}
+async fn qualify_pressure(name: &str) -> Result<()> {
+    let (source, model, eos) = load(name).await?;
+    let (mut native, mut single, mut cpu) = oracle(&source, &model, 8).await?;
+    let teachers = (0..8)
+        .map(|row| vec![IDS[row], eos, IDS[(row + 1) % 8], IDS[(row + 2) % 8]])
+        .collect::<Vec<_>>();
+    let (need4, need8) = {
+        let workspace = model.workspace.lock().await;
+        assert!(workspace.batch.is_none());
+        let need = |rows| -> Result<usize> {
+            BatchProgram::preflight(&workspace.program, rows)?
+                .owned_bytes()?
+                .checked_add(model.state_bytes * rows)
+                .ok_or_else(|| Error::Limit("chunk pressure footprint".into()))
+        };
+        (need(4)?, need(8)?)
+    };
+    let available = model.service.remaining_capacity()?;
+    assert!(need4 < need8 && need8 <= available);
+    let pressure = model
+        .service
+        .allocate_uninitialized(available - need4, BufferRole::Scratch)?;
+    let before = bindings(&native);
+    let domains = domain_values(&native).await?;
+    let bytes = model.service.resident_bytes()?;
+    let queue = model.service.queue().stats()?;
+    let uploads = model.service.uploaded_value_bytes();
+    assert!(chunk(&model, &native, &teachers).await?.is_none());
+    unchanged(&native, &before)?;
+    unchanged_domains(&native, &domains).await?;
+    assert_eq!(model.service.resident_bytes()?, bytes);
+    assert_eq!(model.service.queue().stats()?, queue);
+    assert_eq!(model.service.uploaded_value_bytes(), uploads);
+    assert_eq!(model.service.pending_events()?, 0);
+    assert!(model.workspace.lock().await.batch.is_none());
+    pressure_advance_chunk(
+        &model,
+        &mut native[..4],
+        &mut single[..4],
+        &mut cpu[..4],
+        &teachers[..4],
+    )
+    .await?;
+    pressure_advance_chunk(
+        &model,
+        &mut native[4..],
+        &mut single[4..],
+        &mut cpu[4..],
+        &teachers[4..],
+    )
+    .await?;
+    let before = bindings(&native);
+    let bytes = model.service.resident_bytes()?;
+    let queue = model.service.queue().stats()?;
+    assert!(chunk(&model, &native, &teachers).await?.is_none());
+    unchanged(&native, &before)?;
+    assert_eq!(model.service.resident_bytes()?, bytes);
+    assert_eq!(model.service.queue().stats()?, queue);
+    drop(pressure);
+    pressure_advance_chunk(&model, &mut native, &mut single, &mut cpu, &teachers).await?;
+    future(&model, &mut native, &mut single, &mut cpu).await?;
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    reclaim(model).await?;
+    println!(
+        "P12_NATIVE_TEACHER_CHUNK_PRESSURE_PASS source={name} T=4 B8_None_before_launch=true explicit_B4_then_B8_after_release=CPU_B1_all42_exact one_terminal_status_bytes=4 future_fullV_and_greedy_exact=true final_reserved=0"
+    );
+    Ok(())
+}
+#[test]
+#[ignore = "physical original CUDA chunk pressure; ROOT --release --test-threads=1"]
+fn qwen08_chunk_pressure() {
+    futures_lite::future::block_on(qualify_pressure("qwen35-0.8b")).unwrap();
+}
+#[test]
+#[ignore = "physical original CUDA chunk pressure; ROOT --release --test-threads=1"]
+fn qwen2_chunk_pressure() {
+    futures_lite::future::block_on(qualify_pressure("qwen35-2b")).unwrap();
+}
+
+#[cfg(feature = "hardware-abort-probe")]
+async fn qualify_abort(name: &str) -> Result<()> {
+    use std::sync::atomic::Ordering;
+    let (source, model, eos) = load(name).await?;
+    let (mut native, single, cpu) = oracle(&source, &model, 2).await?;
+    let teachers = vec![vec![11, eos, 55, 101], vec![353, eos, 78, 135]];
+    let before = bindings(&native);
+    let saved = native
+        .iter()
+        .map(ResidentSession::snapshot)
+        .collect::<Vec<_>>();
+    let domains = domain_values(&native).await?;
+    let ready = chunk(&model, &native, &teachers).await?.unwrap();
+    state_receipt(&ready, &before);
+    let client = model.service.client().clone();
+    let old_quarantine = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    model.launch_seen.store(0, Ordering::Release);
+    model.abort_after_launch.store(1, Ordering::Release);
+    let mut pending = Box::pin(chunk(&model, &native, &teachers));
+    loop {
+        if let Some(result) = futures_lite::future::poll_once(pending.as_mut()).await {
+            return Err(Error::State(format!(
+                "chunk abort completed before original hook: success={}",
+                result.is_ok()
+            )));
+        }
+        if model.launch_seen.load(Ordering::Acquire) != 0 {
+            break;
+        }
+        futures_lite::future::yield_now().await;
+    }
+    drop(pending);
+    let handles = model.launch_seen.load(Ordering::Acquire);
+    let quarantined = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    assert!(handles > 0 && quarantined >= old_quarantine + handles);
+    assert!(model.service.ensure_usable().is_err());
+    assert!(
+        ready
+            .commit(&mut native.iter_mut().collect::<Vec<_>>())
+            .is_err()
+    );
+    unchanged(&native, &before)?;
+    assert!(chunk(&model, &native, &teachers).await.is_err());
+    assert!(native[0].prepare(&[11], false).await.is_err());
+    assert!(
+        ResidentService::new(
+            Device::Cuda(cubecl::device::CudaDevice::new(0)),
+            source.source_identity(),
+            ResourceOwner::new("other-model", "after-chunk-abort")?,
+            1024 * 1024,
+            DEVICE_CAP
+        )
+        .is_err()
+    );
+    client.sync().await.map_err(|error| {
+        Error::State(format!(
+            "chunk abort independent prior readback barrier:{error}"
+        ))
+    })?;
+    for (snapshot, expected) in saved.iter().zip(&domains) {
+        physical_prior_after_abort(&client, snapshot, expected).await?;
+    }
+    unchanged(&native, &before)?;
+    let service = Arc::clone(&model.service);
+    let key = service.physical_key();
+    drop(saved);
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    drop(model);
+    let budget = ResourceArbiter::process_shared().snapshot(&key)?;
+    assert!(budget.fenced);
+    assert_eq!(budget.reserved, PROVIDER as u64);
+    assert!(service.ensure_usable().is_err());
+    assert!(crate::physical_quarantine::quarantined_handles_for(&client)? >= quarantined);
+    println!(
+        "P12_NATIVE_TEACHER_CHUNK_ABORT_PASS source={name} B=2 T=4 actual_original_queued_chunk=true one_terminal_Ready_not_reached=true all42_prior_bits_exact=true original_handles={handles} quarantine_retained=true service_fenced=true recovery=false"
+    );
+    Ok(())
+}
+#[cfg(feature = "hardware-abort-probe")]
+#[test]
+#[ignore = "unknown completion fences actual card; ROOT separate --release process"]
+fn qwen08_chunk_abort() {
+    futures_lite::future::block_on(qualify_abort("qwen35-0.8b")).unwrap();
+}
+#[cfg(feature = "hardware-abort-probe")]
+#[test]
+#[ignore = "unknown completion fences actual card; ROOT separate --release process"]
+fn qwen2_chunk_abort() {
+    futures_lite::future::block_on(qualify_abort("qwen35-2b")).unwrap();
+}
+
+#[cfg(feature = "portable-checkpoint")]
+async fn qualify_boundary(name: &str, steps: usize) -> Result<()> {
+    let (source, model, _) = load(name).await?;
+    for (scope, positions) in [
+        ("cross256", [257 - steps, 258 - steps]),
+        ("capacity512", [512 - steps, 512 - steps]),
+    ] {
+        let mut cpu = Vec::new();
+        let mut single = Vec::new();
+        let mut native = Vec::new();
+        let mut teachers = Vec::new();
+        for (row, frontier) in positions.into_iter().enumerate() {
+            let stream = (0..512)
+                .map(|position| IDS[(position + row * 3) % IDS.len()])
+                .collect::<Vec<_>>();
+            let mut reference = QwenDenseCpu::from_shared(Arc::clone(&source))?;
+            reference.prefill_chunk_without_logits(&stream[..frontier])?;
+            let checkpoint = reference.checkpoint()?;
+            let domains = reference.snapshot().domain_values();
+            let rebuilt = source.checkpoint_from_domains(frontier, &domains)?;
+            compare_domains(&domains, &checkpoint.domain_values())?;
+            compare_domains(&domains, &rebuilt.domain_values())?;
+            let mut batch = model.session().await?;
+            let mut b1 = model.session().await?;
+            batch.import_checkpoint(&rebuilt).await?;
+            b1.import_checkpoint(&checkpoint).await?;
+            compare_domains(&domains, &batch.snapshot().read_domains().await?)?;
+            compare_domains(&domains, &b1.snapshot().read_domains().await?)?;
+            teachers.push(stream[frontier..frontier + steps].to_vec());
+            cpu.push(reference);
+            single.push(b1);
+            native.push(batch);
+        }
+        let before = bindings(&native);
+        let uploads = model.service.uploaded_value_bytes();
+        advance_oracle(&mut single, &mut cpu, &teachers).await?;
+        let candidate = chunk(&model, &native, &teachers).await?.unwrap();
+        state_receipt(&candidate, &before);
+        let dispatches = candidate.dispatches();
+        let reads = candidate.commit(&mut native.iter_mut().collect::<Vec<_>>())?;
+        chunk_reads(&reads, 2, steps, dispatches);
+        state(&native, &cpu).await?;
+        state(&single, &cpu).await?;
+        for (session, old) in native.iter().zip(&before) {
+            assert_eq!(session.binding().frontier(), old.frontier() + steps as u64);
+            assert_eq!(session.binding().root(), old.root() + 1);
+        }
+        assert_eq!(model.service.uploaded_value_bytes(), uploads);
+        if scope == "cross256" {
+            future(&model, &mut native, &mut single, &mut cpu).await?;
+        } else {
+            assert!(
+                native
+                    .iter()
+                    .all(|session| session.binding().frontier() == 512)
+            );
+            let before = bindings(&native);
+            let domains = domain_values(&native).await?;
+            let queue = model.service.queue().stats()?;
+            let bytes = model.service.resident_bytes()?;
+            assert!(chunk(&model, &native, &teachers).await.is_err());
+            unchanged(&native, &before)?;
+            unchanged_domains(&native, &domains).await?;
+            assert_eq!(model.service.queue().stats()?, queue);
+            assert_eq!(model.service.resident_bytes()?, bytes);
+        }
+        println!(
+            "P12_NATIVE_TEACHER_CHUNK_BOUNDARY_SCOPE source={name} B=2 T={steps} scope={scope} actual_CPU_hydrated_prefix=true GPU_complete_prefill_claim=false each_teacher_CPU_B1_fullV_all42_exact=true native_terminal_all42_exact=true status_bytes=4"
+        );
+    }
+    reclaim(model).await?;
+    Ok(())
+}
+#[cfg(feature = "portable-checkpoint")]
+#[test]
+#[ignore = "source-real CPU hydrated CUDA boundary; ROOT --release --test-threads=1"]
+fn qwen08_chunk_boundary_t2() {
+    futures_lite::future::block_on(qualify_boundary("qwen35-0.8b", 2)).unwrap();
+}
+#[cfg(feature = "portable-checkpoint")]
+#[test]
+#[ignore = "source-real CPU hydrated CUDA boundary; ROOT --release --test-threads=1"]
+fn qwen08_chunk_boundary_t4() {
+    futures_lite::future::block_on(qualify_boundary("qwen35-0.8b", 4)).unwrap();
+}
+#[cfg(feature = "portable-checkpoint")]
+#[test]
+#[ignore = "source-real CPU hydrated CUDA boundary; ROOT --release --test-threads=1"]
+fn qwen2_chunk_boundary_t2() {
+    futures_lite::future::block_on(qualify_boundary("qwen35-2b", 2)).unwrap();
+}
+#[cfg(feature = "portable-checkpoint")]
+#[test]
+#[ignore = "source-real CPU hydrated CUDA boundary; ROOT --release --test-threads=1"]
+fn qwen2_chunk_boundary_t4() {
+    futures_lite::future::block_on(qualify_boundary("qwen35-2b", 4)).unwrap();
+}
+
+async fn qualify_certificate_refusal(name: &str) -> Result<()> {
+    let (source, model, eos) = load(name).await?;
+    let (mut native, single, cpu) = oracle(&source, &model, 2).await?;
+    let teachers = vec![vec![11, eos, 55, 101], vec![353, eos, 78, 135]];
+    let before = bindings(&native);
+    let saved = native
+        .iter()
+        .map(ResidentSession::snapshot)
+        .collect::<Vec<_>>();
+    let domains = domain_values(&native).await?;
+    let ready = chunk(&model, &native, &teachers).await?.unwrap();
+    state_receipt(&ready, &before);
+    let client = model.service.client().clone();
+    let old_quarantine = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    // Synthetic one-shot NaN in intermediate inactive KV only. The following
+    // teacher erases it; an omitted intermediate certificate/reset OR would pass.
+    BatchProgram::arm_chunk_inactive_nan_probe();
+    assert!(
+        matches!(chunk(&model,&native,&teachers).await,Err(Error::State(ref message))if message.contains("status refused"))
+    );
+    let quarantined = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    assert!(quarantined > old_quarantine);
+    assert!(model.service.ensure_usable().is_err());
+    unchanged(&native, &before)?;
+    assert!(
+        ready
+            .commit(&mut native.iter_mut().collect::<Vec<_>>())
+            .is_err()
+    );
+    // Read only the original retained handles after the actual failed readback.
+    // This diagnostic does not promote an event, make Ready, recover or publish.
+    let (status, padding) = {
+        let workspace = model.workspace.lock().await;
+        let program = &workspace.batch.as_ref().unwrap().program;
+        (program.status(), program.chunk_even_final_padding_view()?)
+    };
+    let raw = client
+        .read_async(vec![status, padding])
+        .await
+        .map_err(|error| Error::State(format!("chunk certificate raw proof:{error}")))?;
+    assert_eq!(raw.len(), 2);
+    assert_eq!(raw[0].len(), 4);
+    assert_eq!(raw[1].len(), 4);
+    assert_eq!(
+        u32::from_ne_bytes(raw[0][..].try_into().unwrap()),
+        crate::program_kernels::NONFINITE_VALUE
+    );
+    assert_eq!(
+        u32::from_ne_bytes(raw[1][..].try_into().unwrap()),
+        0,
+        "later append must heal final padding; rejection proves earlier certificate latch"
+    );
+    for (snapshot, expected) in saved.iter().zip(&domains) {
+        physical_prior_after_abort(&client, snapshot, expected).await?;
+    }
+    assert!(chunk(&model, &native, &teachers).await.is_err());
+    unchanged(&native, &before)?;
+    let service = Arc::clone(&model.service);
+    let key = service.physical_key();
+    drop(raw);
+    drop(saved);
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    drop(model);
+    let budget = ResourceArbiter::process_shared().snapshot(&key)?;
+    assert!(budget.fenced);
+    assert_eq!(budget.reserved, PROVIDER as u64);
+    assert!(service.ensure_usable().is_err());
+    assert!(crate::physical_quarantine::quarantined_handles_for(&client)? >= quarantined);
+    println!(
+        "P12_NATIVE_TEACHER_CHUNK_CERTIFICATE_REFUSAL source={name} B=2 T=4 synthetic_intermediate_inactive_KV_nan=true final_padding_raw_positive_zero=true actual_latched_status=2 original_roots_all42_unchanged=true quarantine_retained=true service_fenced=true recovery=false"
+    );
+    Ok(())
+}
+#[test]
+#[ignore = "actual numerical-status refusal fences card; ROOT fresh --release process"]
+fn qwen08_chunk_certificate_refusal() {
+    futures_lite::future::block_on(qualify_certificate_refusal("qwen35-0.8b")).unwrap();
+}
+#[test]
+#[ignore = "actual numerical-status refusal fences card; ROOT fresh --release process"]
+fn qwen2_chunk_certificate_refusal() {
+    futures_lite::future::block_on(qualify_certificate_refusal("qwen35-2b")).unwrap();
+}
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/batch/chunk.rs
@@ -0,0 +1,307 @@
+//! Qualifier-only sequential original state prefix with one gather/scatter/Ready.
+use super::*;
+
+#[cfg(test)]
+std::thread_local! {
+    static INACTIVE_NAN_PROBE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
+}
+
+/// RAII restores persistent workspace metadata on every ordinary error/unwind.
+/// Both original physical banks remain leased by the enclosing one submission.
+struct PackedOrientation<'a> {
+    program: &'a mut BatchProgram,
+    flipped: bool,
+}
+impl PackedOrientation<'_> {
+    fn flip(&mut self) {
+        for state in &mut self.program.packed {
+            std::mem::swap(&mut state.prior, &mut state.candidate);
+        }
+        self.flipped = !self.flipped;
+    }
+}
+impl Drop for PackedOrientation<'_> {
+    fn drop(&mut self) {
+        if self.flipped {
+            self.flip();
+        }
+    }
+}
+
+impl BatchProgram {
+    /// Synthetic one-shot proof only: poison an inactive intermediate KV cell
+    /// after teacher zero. Teacher one overwrites padding back to canonical zero;
+    /// the real original intermediate certificate must keep the attempt refused.
+    #[cfg(test)]
+    pub(crate) fn arm_chunk_inactive_nan_probe() {
+        INACTIVE_NAN_PROBE
+            .with(|flag| assert!(!flag.replace(true), "chunk fault probe already armed"));
+    }
+
+    /// Diagnostic raw view only, after an even-T refusal and restored metadata.
+    /// Caller retains the original program/lease and cannot obtain a Ready here.
+    #[cfg(test)]
+    pub(crate) fn chunk_even_final_padding_view(&self) -> Result<Handle> {
+        let keys = self
+            .packed
+            .iter()
+            .find_map(|state| match &state.prior {
+                StateBuffers::Kv { keys, .. } => Some(keys),
+                _ => None,
+            })
+            .ok_or_else(|| Error::State("chunk padding proof requires KV".into()))?;
+        let offset = keys
+            .elements
+            .checked_sub(1)
+            .and_then(|n| n.checked_mul(4))
+            .ok_or_else(|| Error::Limit("chunk padding proof offset".into()))?;
+        let start = keys
+            .handle
+            .offset_start
+            .unwrap_or(0)
+            .checked_add(offset as u64)
+            .ok_or_else(|| Error::Limit("chunk padding proof start".into()))?;
+        let end = start
+            .checked_add(4)
+            .ok_or_else(|| Error::Limit("chunk padding proof end".into()))?;
+        let mut view = keys.handle.clone();
+        view.offset_start = Some(start);
+        view.offset_end = Some(
+            view.size()
+                .checked_sub(end)
+                .ok_or_else(|| Error::State("chunk padding proof extent".into()))?,
+        );
+        BufferRange::new(&view)?;
+        Ok(view)
+    }
+
+    /// Entire temporal span, original first priors, and exact final scatter bank
+    /// are cold checked before original submission. No future Ready is invented.
+    pub(crate) fn validate_state_chunk(
+        &self,
+        client: &Client,
+        steps: &[Vec<RuntimeParameters>],
+        bindings: &[Vec<StateBinding>],
+    ) -> Result<()> {
+        if !matches!(steps.len(), 2 | 4) || self.graph_pinned() {
+            return Err(Error::Unsupported(
+                "teacher chunk T2/T4 requires unpinned native workspace".into(),
+            ));
+        }
+        let first = &steps[0];
+        self.validate_runtime(client, first, bindings)?;
+        for (step, parameters) in steps.iter().enumerate() {
+            if parameters.len() != self.rows {
+                return Err(Error::Malformed("native teacher chunk row coverage".into()));
+            }
+            for (row, parameter) in parameters.iter().enumerate() {
+                if first[row].position.checked_add(step) != Some(parameter.position) {
+                    return Err(Error::State(
+                        "native teacher chunk position progression".into(),
+                    ));
+                }
+                // Runtime validation here is physical geometry only. The actual
+                // future prior is produced into an owned packed bank in-stream.
+                let mut shape = bindings[row].clone();
+                for state in &mut shape {
+                    state.prior_tokens = parameter.position;
+                }
+                self.inner.validate_runtime(client, *parameter, &shape)?;
+                for node in self.inner.plan.nodes() {
+                    if let PlanOperation::Embed { tensor, .. } = &node.operation
+                        && parameter.token as usize >= tensor.rows
+                    {
+                        return Err(Error::Malformed(
+                            "teacher chunk token exceeds source vocabulary".into(),
+                        ));
+                    }
+                }
+            }
+        }
+        // T2/T4 finish in the originally prior packed bank. Validate its exact
+        // certified scatter bindings before any kernel is submitted.
+        self.prepare_state_transports_for_bank(
+            client,
+            bindings,
+            true,
+            steps.len().is_multiple_of(2),
+        )?;
+        Ok(())
+    }
+
+    pub(crate) fn launch_state_chunk(
+        &mut self,
+        client: &Client,
+        steps: &[Vec<RuntimeParameters>],
+        bindings: &[Vec<StateBinding>],
+    ) -> Result<ResidentSubmission> {
+        self.validate_state_chunk(client, steps, bindings)?;
+        let mut orientation = PackedOrientation {
+            program: self,
+            flipped: false,
+        };
+        // Only this original entry clears status; later teachers preserve its OR.
+        let prefix = orientation
+            .program
+            .launch_inputs(client, &steps[0], bindings, None)?;
+        let mut body = BatchBodyReceipt {
+            dispatches: 0,
+            fused_dispatches: 0,
+        };
+        for (step, parameters) in steps.iter().enumerate() {
+            if step > 0 {
+                body.dispatches += orientation
+                    .program
+                    .stage_teacher_parameters(client, parameters);
+            }
+            let current =
+                orientation
+                    .program
+                    .launch_body(client, parameters, ResidentOutput::StateOnly)?;
+            body.dispatches += current.dispatches;
+            body.fused_dispatches += current.fused_dispatches;
+            #[cfg(test)]
+            if step == 0 && INACTIVE_NAN_PROBE.with(|flag| flag.replace(false)) {
+                let keys = orientation
+                    .program
+                    .packed
+                    .iter()
+                    .find_map(|state| match &state.candidate {
+                        StateBuffers::Kv { keys, .. } => Some(keys),
+                        _ => None,
+                    })
+                    .ok_or_else(|| {
+                        Error::State("chunk fault probe requires an actual KV domain".into())
+                    })?;
+                if steps.iter().flatten().any(|parameter| {
+                    parameter.valid_tokens >= orientation.program.inner.plan.prefix_capacity()
+                }) {
+                    return Err(Error::State(
+                        "chunk fault probe requires untouched capacity padding".into(),
+                    ));
+                }
+                let byte_offset = keys
+                    .elements
+                    .checked_sub(1)
+                    .and_then(|n| n.checked_mul(4))
+                    .ok_or_else(|| Error::Limit("chunk fault probe offset".into()))?;
+                let start = keys
+                    .handle
+                    .offset_start
+                    .unwrap_or(0)
+                    .checked_add(byte_offset as u64)
+                    .ok_or_else(|| Error::Limit("chunk fault probe view start".into()))?;
+                let end = start
+                    .checked_add(4)
+                    .ok_or_else(|| Error::Limit("chunk fault probe view end".into()))?;
+                let mut view = keys.handle.clone();
+                view.offset_start = Some(start);
+                view.offset_end = Some(
+                    view.size()
+                        .checked_sub(end)
+                        .ok_or_else(|| Error::State("chunk fault probe range".into()))?,
+                );
+                BufferRange::new(&view)?;
+                unsafe {
+                    batch_tokens::launch_unchecked(
+                        client,
+                        grid(1),
+                        threads(),
+                        arg(&view, 1),
+                        0x7fc00001u32,
+                        0,
+                        0,
+                        0,
+                        0,
+                        0,
+                        0,
+                        0,
+                        1,
+                    );
+                }
+                body.dispatches += 1;
+            }
+            if step + 1 < steps.len() {
+                // Every original candidate word (including capacity KV padding)
+                // is certified before a packed bank is consumed/reused. This is
+                // the same finite predicate and persistent OR as final scatter.
+                for state in &orientation.program.packed {
+                    match &state.candidate {
+                        StateBuffers::Vector(value) => {
+                            body.dispatches +=
+                                orientation
+                                    .program
+                                    .certify(client, &value.handle, value.elements);
+                        }
+                        StateBuffers::Kv { keys, values } => {
+                            body.dispatches +=
+                                orientation
+                                    .program
+                                    .certify(client, &keys.handle, keys.elements);
+                            body.dispatches += orientation.program.certify(
+                                client,
+                                &values.handle,
+                                values.elements,
+                            );
+                        }
+                    }
+                }
+                orientation.flip();
+            }
+        }
+        // Exactly one final certified scatter into the independent candidates.
+        // Original completion/publisher remains the enclosing physical owner's job.
+        orientation.program.launch_outputs(
+            client,
+            bindings,
+            ResidentOutput::StateOnly,
+            prefix,
+            body,
+        )
+    }
+
+    /// Same two scalar-token/position kernels as original launch_inputs; no
+    /// clear, gather, upload, device-token feedback, sampler or graph enqueue.
+    fn stage_teacher_parameters(&self, client: &Client, parameters: &[RuntimeParameters]) -> usize {
+        let mut tokens = [0u32; 8];
+        let mut positions = [0u32; 8];
+        for ((token, position), parameter) in tokens.iter_mut().zip(&mut positions).zip(parameters)
+        {
+            *token = parameter.token;
+            *position = parameter.position as u32;
+        }
+        unsafe {
+            batch_tokens::launch_unchecked(
+                client,
+                grid(self.rows),
+                threads(),
+                arg(&self.inner.token, self.rows),
+                tokens[0],
+                tokens[1],
+                tokens[2],
+                tokens[3],
+                tokens[4],
+                tokens[5],
+                tokens[6],
+                tokens[7],
+                self.rows,
+            );
+            batch_tokens::launch_unchecked(
+                client,
+                grid(self.rows),
+                threads(),
+                arg(&self.positions, self.rows),
+                positions[0],
+                positions[1],
+                positions[2],
+                positions[3],
+                positions[4],
+                positions[5],
+                positions[6],
+                positions[7],
+                self.rows,
+            );
+        }
+        2
+    }
+}
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -61,6 +61,9 @@
 pub(crate) use graph::{BatchGraphProfile, BatchGraphReplay};
 #[cfg(all(test, feature = "qualification", feature = "cuda"))]
 mod state_transport_qualification;
+
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+mod chunk;
 
 impl BatchProgram {
     /// New allocations only: the original source handles remain shared.
@@ -719,12 +722,22 @@
         bindings: &[Vec<StateBinding>],
         scatter: bool,
     ) -> Result<Vec<PreparedStateTransport>> {
+        self.prepare_state_transports_for_bank(client, bindings, scatter, false)
+    }
+
+    fn prepare_state_transports_for_bank(
+        &self,
+        client: &Client,
+        bindings: &[Vec<StateBinding>],
+        scatter: bool,
+        swap_banks: bool,
+    ) -> Result<Vec<PreparedStateTransport>> {
         if bindings.len() != self.rows || self.packed.is_empty() || self.packed.len() > 256 {
             return Err(Error::State("state transport domain/row coverage".into()));
         }
         let mut prepared = Vec::with_capacity(self.packed.len());
         for packed in &self.packed {
-            let dense = if scatter {
+            let dense = if scatter != swap_banks {
                 &packed.candidate
             } else {
                 &packed.prior

````

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/frozen/ROOT_WIRING.patch

Original UTF-8 bytes: 327; SHA256 `31560bb5d9cbd084eaf76c77e7862c7ea28489dcd04246e2d84c3d66ffdf31a2`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_inference/prefix_qualification.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/prefix_qualification.rs
@@ -577,3 +577,5 @@
 fn original_qwen35_2b_native_prefix_pressure() -> Result<()> {
     futures_lite::future::block_on(qualify_prefix_pressure("qwen35-2b"))
 }
+
+mod chunk;

````

## Source: .agents/runs/serving-and-speculation-native-solo-temporal-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 65291; SHA256 `4ffbd09a8105c8b360d12addd621a9ddca95331ba9d7f700bd9a2b1774e5aa03`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_inference.rs
+++ b/crates/zerogravity-gpu/src/resident_inference.rs
@@ -24,6 +24,8 @@
 mod fused_qualification;
 #[cfg(all(feature = "qualification", feature = "cuda"))]
 mod graph;
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+mod solo;
 #[cfg(all(feature = "qualification", feature = "cuda"))]
 pub(crate) use graph::qualify as qualify_graph;
 #[cfg(all(
@@ -105,6 +107,8 @@
     state_accesses: Vec<(ResidentLease, DeviceAccess)>,
     logit_accesses: Vec<(ResidentLease, DeviceAccess)>,
     batch: Option<BatchWorkspace>,
+    #[cfg(all(feature = "qualification", feature = "cuda"))]
+    solo: Option<solo::SoloWorkspace>,
 }
 struct BatchWorkspace {
     program: BatchProgram,
@@ -512,6 +516,8 @@
                 state_accesses,
                 logit_accesses,
                 batch: None,
+                #[cfg(all(feature = "qualification", feature = "cuda"))]
+                solo: None,
             }),
             allocation: Mutex::new(()),
         }))
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk/solo.rs
@@ -0,0 +1,598 @@
+//! ROOT-only native SOLO temporal proof; no HTTP/throughput/profile eligibility.
+use super::*;
+use crate::resident_program::SoloTemporalProgram;
+
+async fn loaded(name: &str) -> Result<(Arc<LoadedQwenModel>, Arc<ResidentModel>, u32)> {
+    if cfg!(debug_assertions) {
+        return Err(Error::Unsupported(
+            "solo temporal qualifier requires release".into(),
+        ));
+    }
+    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
+        .join("../../target/models")
+        .join(name);
+    let source = LoadedQwenModel::open(&path.join("model.gguf"), &path.join("config.json"))?;
+    let eos = QwenTokenizer::from_file(&path.join("tokenizer.json"))?.eos();
+    let model = ResidentModel::load_inner(
+        Device::Cuda(cubecl::device::CudaDevice::new(0)),
+        source.clone(),
+        source.forward_plan(512, true, true)?,
+        PROVIDER,
+        DEVICE_CAP,
+        true,
+    )
+    .await?;
+    assert_eq!(model.states.len(), 42);
+    assert_eq!(model.vocab, 248320);
+    model.service.pinned_cuda_queue()?;
+    Ok((source, model, eos))
+}
+fn receipt(session: &ResidentSession, prepared: &PreparedInference) -> Result<()> {
+    session.validate_prepared(prepared)?;
+    assert_eq!(prepared.attempt.base(), session.binding());
+    assert_eq!(prepared.ready.event_ids().len(), 1);
+    let read = &prepared.readback;
+    assert_eq!(read.state_only_steps, 2);
+    assert_eq!(read.logit_steps, 0);
+    assert_eq!(read.bytes, 4);
+    assert_eq!(read.chunk_bytes, 4);
+    assert!(read.dispatches > 0 && read.logits.is_empty() && read.device_token.is_none());
+    Ok(())
+}
+async fn references(
+    single: &mut [ResidentSession],
+    cpu: &mut [QwenDenseCpu],
+    teachers: &[u32; 2],
+) -> Result<Vec<(StateSlot, Vec<f32>)>> {
+    assert_eq!(single.len(), 1);
+    assert_eq!(cpu.len(), 1);
+    let mut first = Vec::new();
+    for (teacher, &token) in teachers.iter().enumerate() {
+        let logits = cpu[0].forward(token)?;
+        assert_eq!(logits.len(), 248320);
+        exact(
+            &logits,
+            &single[0].forward(&[token], false).await?.logits,
+            "solo each teacher CPU/B1 fullV",
+        )?;
+        state(single, cpu).await?;
+        if teacher == 0 {
+            first = cpu[0].snapshot().domain_values();
+        }
+    }
+    Ok(first)
+}
+async fn first_state(
+    model: &Arc<ResidentModel>,
+    root: u64,
+    frontier: usize,
+    expected: &[(StateSlot, Vec<f32>)],
+) -> Result<()> {
+    // Exact PRIVATE T0 bytes after real terminal Ready. No future RootBinding,
+    // SessionAuthority/checkpoint/Ready is manufactured for this observation.
+    let _allocation = model.allocation.lock().await;
+    let workspace = model.workspace.lock().await;
+    let domains = workspace.solo.as_ref().unwrap().middle.clone();
+    assert_eq!(domains.len(), 42);
+    let leases = domains
+        .iter()
+        .flat_map(|d| &d.leases)
+        .cloned()
+        .collect::<Vec<_>>();
+    let bytes = leases.iter().map(|l| l.bytes() as u64).sum();
+    let reads = leases
+        .iter()
+        .map(|l| l.readback(0, l.bytes()))
+        .collect::<Result<Vec<_>>>()?;
+    let accesses = leases
+        .iter()
+        .cloned()
+        .map(|l| (l, DeviceAccess::Read))
+        .collect::<Vec<_>>();
+    drop(workspace);
+    let ticket = model.service.queue().submit(
+        model.service.budget(),
+        model.service.owner().clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::TransferStaging,
+            bytes,
+        }],
+        None,
+    )?;
+    let mut slot = ticket.wait().await?;
+    let _workspace = model.workspace.lock().await;
+    slot.begin_submission()?;
+    let flight = model.service.submit(root, &accesses)?;
+    let data = flight
+        .complete_read_only_readback(&reads, |data| {
+            for raw in data {
+                if words(raw)?.iter().any(|x| !x.is_finite()) {
+                    return Err(Error::State("solo middle nonfinite".into()));
+                }
+            }
+            Ok(())
+        })
+        .await?;
+    assert_eq!(data.len(), leases.len());
+    let mut cursor = 0;
+    let mut actual = Vec::new();
+    for domain in &domains {
+        let mut flat = Vec::new();
+        for lease in &domain.leases {
+            let values = words(&data[cursor])?;
+            cursor += 1;
+            if matches!(domain.buffers, StateBuffers::Kv { .. }) {
+                let active = lease.bytes() / 4 / 512 * frontier;
+                assert!(
+                    values[active..].iter().all(|x| x.to_bits() == 0),
+                    "solo T0 canonical KV tail"
+                );
+                flat.extend_from_slice(&values[..active]);
+            } else {
+                flat.extend(values);
+            }
+        }
+        actual.push((domain.slot, flat));
+    }
+    assert_eq!(cursor, data.len());
+    compare_domains(expected, &actual)?;
+    drop(data);
+    slot.finish()?;
+    assert_eq!(model.service.pending_events()?, 0);
+    Ok(())
+}
+async fn future(
+    native: &mut [ResidentSession],
+    single: &mut [ResidentSession],
+    cpu: &mut [QwenDenseCpu],
+) -> Result<()> {
+    assert_eq!(native.len(), 1);
+    assert_eq!(single.len(), 1);
+    assert_eq!(cpu.len(), 1);
+    let logits = cpu[0].forward(101)?;
+    exact(
+        &logits,
+        &single[0].forward(&[101], false).await?.logits,
+        "solo future CPU/B1 fullV",
+    )?;
+    exact(
+        &logits,
+        &native[0].forward(&[101], false).await?.logits,
+        "solo future original native fullV",
+    )?;
+    state(native, cpu).await?;
+    state(single, cpu).await?;
+    let logits = cpu[0].forward(353)?;
+    let saved = single[0].snapshot();
+    exact(
+        &logits,
+        &single[0].forward(&[353], false).await?.logits,
+        "solo greedy independent fullV",
+    )?;
+    let selected = (0..logits.len())
+        .max_by(|a, b| logits[*a].total_cmp(&logits[*b]).then_with(|| b.cmp(a)))
+        .unwrap();
+    let expected = (selected as u32, logits[selected].to_bits());
+    single[0].restore(&saved)?;
+    let independent = single[0].forward_greedy(&[353]).await?;
+    let packet = independent.device_token.as_ref().unwrap();
+    assert_eq!((packet.id, packet.max_bits), expected);
+    let actual = native[0].forward_greedy(&[353]).await?;
+    let packet = actual.device_token.as_ref().unwrap();
+    assert_eq!((packet.id, packet.max_bits), expected);
+    assert_eq!(packet.producer, Some(native[0].binding()));
+    state(native, cpu).await?;
+    state(single, cpu).await?;
+    Ok(())
+}
+async fn normal(name: &str) -> Result<()> {
+    let (source, model, eos) = loaded(name).await?;
+    let (mut native, mut single, mut cpu) = oracle(&source, &model, 1).await?;
+    assert!(native[0].binding().frontier() > 0);
+    let teachers = [55, eos];
+    let before = bindings(&native);
+    let domains = domain_values(&native).await?;
+    let queue = model.service.queue().stats()?;
+    for invalid in [&[55u32][..], &[55, eos, 101][..], &[55, u32::MAX][..]] {
+        assert!(
+            native[0]
+                .prepare_state_temporal_b1_t2_admitted(invalid)
+                .await
+                .is_err()
+        );
+    }
+    assert_eq!(model.service.queue().stats()?, queue);
+    unchanged(&native, &before)?;
+    let (cold, p1) = {
+        let workspace = model.workspace.lock().await;
+        assert!(workspace.solo.is_none());
+        (
+            SoloTemporalProgram::preflight(&workspace.program)?.owned_bytes()? + model.state_bytes,
+            workspace.program.budget().owned_bytes()?,
+        )
+    };
+    let bytes = model.service.resident_bytes()?;
+    let uploads = model.service.uploaded_value_bytes();
+    let abandoned = native[0]
+        .prepare_state_temporal_b1_t2_admitted(&teachers)
+        .await?
+        .unwrap();
+    receipt(&native[0], &abandoned)?;
+    drop(abandoned);
+    unchanged_domains(&native, &domains).await?;
+    assert_eq!(model.service.resident_bytes()?, bytes + cold);
+    let warm = model.service.resident_bytes()?;
+    drop(
+        native[0]
+            .prepare_state_temporal_b1_t2_admitted(&teachers)
+            .await?
+            .unwrap(),
+    );
+    idle(&model, warm)?;
+    let snapshot = native[0].snapshot();
+    let stale = native[0]
+        .prepare_state_temporal_b1_t2_admitted(&teachers)
+        .await?
+        .unwrap();
+    native[0].restore(&snapshot)?;
+    let restored = bindings(&native);
+    assert!(native[0].commit(stale).is_err());
+    unchanged(&native, &restored)?;
+    unchanged_domains(&native, &domains).await?;
+    drop(snapshot);
+    let before = native[0].binding();
+    let prepared = native[0]
+        .prepare_state_temporal_b1_t2_admitted(&teachers)
+        .await?
+        .unwrap();
+    receipt(&native[0], &prepared)?;
+    let dispatches = prepared.readback.dispatches;
+    let projections = {
+        let workspace = model.workspace.lock().await;
+        let solo = workspace.solo.as_ref().unwrap();
+        assert_eq!(solo.owned_bytes()?, cold);
+        assert_eq!(workspace.program.budget().owned_bytes()?, p1);
+        assert_eq!(
+            solo.last_native_projections,
+            solo.program.expected_native_projections()
+        );
+        assert!(solo.last_native_projections > 0);
+        solo.last_native_projections
+    };
+    let first = references(&mut single, &mut cpu, &teachers).await?;
+    let read = native[0].commit(prepared)?;
+    assert_eq!(native[0].binding().root(), before.root() + 1);
+    assert_eq!(native[0].binding().frontier(), before.frontier() + 2);
+    assert_eq!(native[0].binding().session_epoch(), before.session_epoch());
+    assert_eq!(
+        (
+            read.bytes,
+            read.chunk_bytes,
+            read.state_only_steps,
+            read.logit_steps
+        ),
+        (4, 4, 2, 0)
+    );
+    assert!(read.logits.is_empty() && read.device_token.is_none());
+    first_state(
+        &model,
+        native[0].binding().root(),
+        before.frontier() as usize + 1,
+        &first,
+    )
+    .await?;
+    state(&native, &cpu).await?;
+    state(&single, &cpu).await?;
+    assert_eq!(model.service.uploaded_value_bytes(), uploads);
+    drop(read);
+    future(&mut native, &mut single, &mut cpu).await?;
+    drop(first);
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    reclaim(model).await?;
+    println!(
+        "P12_SOLO_B1_T2_PASS source={name} authority=1 native_stateless_rows=2 original_P1_causal=true each_teacher_CPU_B1_fullV_all42=true native_T0_T1_all42_exact=true future_fullV_greedy=true EOS_teacher_no_output=true terminal_status_bytes=4 advance2_once=true source_launch_sites={dispatches} source_projection_sites={projections} final_reserved=0"
+    );
+    Ok(())
+}
+async fn pressure(name: &str) -> Result<()> {
+    let (source, model, eos) = loaded(name).await?;
+    let (mut native, mut single, mut cpu) = oracle(&source, &model, 1).await?;
+    let teachers = [55, eos];
+    let need = {
+        let workspace = model.workspace.lock().await;
+        assert!(workspace.solo.is_none());
+        SoloTemporalProgram::preflight(&workspace.program)?.owned_bytes()? + 2 * model.state_bytes
+    };
+    let available = model.service.remaining_capacity()?;
+    assert!(need > 0 && available >= need);
+    let pressure = model
+        .service
+        .allocate_uninitialized(available - (need - 1), BufferRole::Scratch)?;
+    let before = bindings(&native);
+    let domains = domain_values(&native).await?;
+    let bytes = model.service.resident_bytes()?;
+    let queue = model.service.queue().stats()?;
+    assert!(
+        native[0]
+            .prepare_state_temporal_b1_t2_admitted(&teachers)
+            .await?
+            .is_none()
+    );
+    assert_eq!(model.service.resident_bytes()?, bytes);
+    assert_eq!(model.service.queue().stats()?, queue);
+    assert!(model.workspace.lock().await.solo.is_none());
+    unchanged(&native, &before)?;
+    unchanged_domains(&native, &domains).await?;
+    drop(pressure);
+    let prepared = native[0]
+        .prepare_state_temporal_b1_t2_admitted(&teachers)
+        .await?
+        .unwrap();
+    references(&mut single, &mut cpu, &teachers).await?;
+    receipt(&native[0], &prepared)?;
+    drop(native[0].commit(prepared)?);
+    state(&native, &cpu).await?;
+    state(&single, &cpu).await?;
+    future(&mut native, &mut single, &mut cpu).await?;
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    reclaim(model).await?;
+    println!(
+        "P12_SOLO_B1_T2_PRESSURE_PASS source={name} full_P2_middle_final_None_before_queue=true original_P1_already_charged=true no_fallback=true final_reserved=0"
+    );
+    Ok(())
+}
+#[cfg(feature = "portable-checkpoint")]
+async fn boundary(name: &str) -> Result<()> {
+    let (source, model, _) = loaded(name).await?;
+    for frontier in [255usize, 510] {
+        let stream = (0..512).map(|i| IDS[i % IDS.len()]).collect::<Vec<_>>();
+        let mut cpu = QwenDenseCpu::from_shared(source.clone())?;
+        cpu.prefill_chunk_without_logits(&stream[..frontier])?;
+        let checkpoint = cpu.checkpoint()?;
+        let expected = cpu.snapshot().domain_values();
+        let rebuilt = source.checkpoint_from_domains(frontier, &expected)?;
+        compare_domains(&expected, &checkpoint.domain_values())?;
+        compare_domains(&expected, &rebuilt.domain_values())?;
+        let mut native = model.session().await?;
+        let mut single = model.session().await?;
+        native.import_checkpoint(&rebuilt).await?;
+        single.import_checkpoint(&checkpoint).await?;
+        compare_domains(&expected, &native.snapshot().read_domains().await?)?;
+        compare_domains(&expected, &single.snapshot().read_domains().await?)?;
+        let mut native = vec![native];
+        let mut single = vec![single];
+        let mut cpu = vec![cpu];
+        let teachers = [stream[frontier], stream[frontier + 1]];
+        let before = native[0].binding();
+        let uploads = model.service.uploaded_value_bytes();
+        let prepared = native[0]
+            .prepare_state_temporal_b1_t2_admitted(&teachers)
+            .await?
+            .unwrap();
+        let first = references(&mut single, &mut cpu, &teachers).await?;
+        receipt(&native[0], &prepared)?;
+        drop(native[0].commit(prepared)?);
+        first_state(&model, native[0].binding().root(), frontier + 1, &first).await?;
+        state(&native, &cpu).await?;
+        state(&single, &cpu).await?;
+        assert_eq!(native[0].binding().root(), before.root() + 1);
+        assert_eq!(native[0].binding().frontier(), (frontier + 2) as u64);
+        assert_eq!(model.service.uploaded_value_bytes(), uploads);
+        if frontier == 255 {
+            future(&mut native, &mut single, &mut cpu).await?;
+        } else {
+            let before = bindings(&native);
+            let domains = domain_values(&native).await?;
+            let queue = model.service.queue().stats()?;
+            let bytes = model.service.resident_bytes()?;
+            assert!(
+                native[0]
+                    .prepare_state_temporal_b1_t2_admitted(&teachers)
+                    .await
+                    .is_err()
+            );
+            unchanged(&native, &before)?;
+            unchanged_domains(&native, &domains).await?;
+            assert_eq!(model.service.queue().stats()?, queue);
+            assert_eq!(model.service.resident_bytes()?, bytes);
+        }
+        println!(
+            "P12_SOLO_B1_T2_BOUNDARY source={name} frontier={frontier} CPU_hydrated_source_real=true GPU_complete_prefill_claim=false native_T0_T1_all42=true each_teacher_CPU_B1_fullV=true"
+        );
+    }
+    reclaim(model).await?;
+    Ok(())
+}
+#[cfg(feature = "hardware-abort-probe")]
+async fn abort(name: &str) -> Result<()> {
+    use std::sync::atomic::Ordering;
+    let (source, model, eos) = loaded(name).await?;
+    let (mut native, single, cpu) = oracle(&source, &model, 1).await?;
+    let teachers = [55, eos];
+    let before = bindings(&native);
+    let snapshot = native[0].snapshot();
+    let domains = domain_values(&native).await?;
+    let ready = native[0]
+        .prepare_state_temporal_b1_t2_admitted(&teachers)
+        .await?
+        .unwrap();
+    let client = model.service.client().clone();
+    let old = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    model.launch_seen.store(0, Ordering::Release);
+    model.abort_after_launch.store(1, Ordering::Release);
+    let mut pending = Box::pin(native[0].prepare_state_temporal_b1_t2_admitted(&teachers));
+    loop {
+        if let Some(result) = futures_lite::future::poll_once(pending.as_mut()).await {
+            return Err(Error::State(format!(
+                "solo abort completed before original hook:{}",
+                result.is_ok()
+            )));
+        }
+        if model.launch_seen.load(Ordering::Acquire) != 0 {
+            break;
+        }
+        futures_lite::future::yield_now().await;
+    }
+    drop(pending);
+    let handles = model.launch_seen.load(Ordering::Acquire);
+    let retained = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    assert!(handles > 0 && retained >= old + handles && model.service.ensure_usable().is_err());
+    assert!(native[0].commit(ready).is_err());
+    unchanged(&native, &before)?;
+    assert!(
+        native[0]
+            .prepare_state_temporal_b1_t2_admitted(&teachers)
+            .await
+            .is_err()
+    );
+    assert!(native[0].prepare(&[55], false).await.is_err());
+    assert!(
+        ResidentService::new(
+            Device::Cuda(cubecl::device::CudaDevice::new(0)),
+            source.source_identity(),
+            ResourceOwner::new("other-model", "after-solo-abort")?,
+            1024 * 1024,
+            DEVICE_CAP
+        )
+        .is_err()
+    );
+    client
+        .sync()
+        .await
+        .map_err(|e| Error::State(format!("solo prior diagnostic barrier:{e}")))?;
+    physical_prior_after_abort(&client, &snapshot, &domains[0]).await?;
+    unchanged(&native, &before)?;
+    let service = model.service.clone();
+    let key = service.physical_key();
+    drop(snapshot);
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    drop(model);
+    assert!(ResourceArbiter::process_shared().snapshot(&key)?.fenced);
+    assert_eq!(
+        ResourceArbiter::process_shared().snapshot(&key)?.reserved,
+        PROVIDER as u64
+    );
+    assert!(service.ensure_usable().is_err());
+    assert!(crate::physical_quarantine::quarantined_handles_for(&client)? >= retained);
+    println!(
+        "P12_SOLO_B1_T2_ABORT_PASS source={name} original_hook=true one_terminal_Ready_not_reached=true P1_P2_middle_final_source_retained=true original_prior_all42_exact=true fenced=true recovery=false"
+    );
+    Ok(())
+}
+async fn certificate(name: &str) -> Result<()> {
+    let (source, model, eos) = loaded(name).await?;
+    let (mut native, single, cpu) = oracle(&source, &model, 1).await?;
+    let teachers = [55, eos];
+    let before = bindings(&native);
+    let snapshot = native[0].snapshot();
+    let domains = domain_values(&native).await?;
+    let ready = native[0]
+        .prepare_state_temporal_b1_t2_admitted(&teachers)
+        .await?
+        .unwrap();
+    let client = model.service.client().clone();
+    let old = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    SoloTemporalProgram::arm_inactive_nan_probe();
+    assert!(
+        matches!(native[0].prepare_state_temporal_b1_t2_admitted(&teachers).await,
+        Err(Error::State(ref message)) if message.contains("status refused"))
+    );
+    let retained = crate::physical_quarantine::quarantined_handles_for(&client)?;
+    assert!(retained > old && model.service.ensure_usable().is_err());
+    let (status, final_kv) = {
+        let workspace = model.workspace.lock().await;
+        (
+            SoloTemporalProgram::diagnostic_original_status(&workspace.program),
+            workspace
+                .solo
+                .as_ref()
+                .unwrap()
+                .last_fault_final_kv
+                .as_ref()
+                .unwrap()
+                .clone(),
+        )
+    };
+    // Explicit post-refusal raw proof only; no promotion/recovery/session Ready.
+    let tail = final_kv
+        .handle()
+        .offset_start((final_kv.bytes() - 4) as u64);
+    let raw = client
+        .read_async(vec![status, tail])
+        .await
+        .map_err(|e| Error::State(format!("solo raw fault proof:{e}")))?;
+    assert_eq!(raw.len(), 2);
+    assert_eq!(raw[0].len(), 4);
+    assert_eq!(raw[1].len(), 4);
+    assert_eq!(
+        u32::from_ne_bytes(raw[0][..].try_into().unwrap()),
+        crate::program_kernels::NONFINITE_VALUE
+    );
+    assert_eq!(
+        u32::from_ne_bytes(raw[1][..].try_into().unwrap()),
+        0,
+        "actual T1 final padding heals NaN"
+    );
+    assert!(native[0].commit(ready).is_err());
+    unchanged(&native, &before)?;
+    physical_prior_after_abort(&client, &snapshot, &domains[0]).await?;
+    assert!(
+        native[0]
+            .prepare_state_temporal_b1_t2_admitted(&teachers)
+            .await
+            .is_err()
+    );
+    let service = model.service.clone();
+    let key = service.physical_key();
+    drop(raw);
+    drop(final_kv);
+    drop(snapshot);
+    drop(domains);
+    drop(cpu);
+    drop(single);
+    drop(native);
+    drop(model);
+    assert!(ResourceArbiter::process_shared().snapshot(&key)?.fenced);
+    assert_eq!(
+        ResourceArbiter::process_shared().snapshot(&key)?.reserved,
+        PROVIDER as u64
+    );
+    assert!(service.ensure_usable().is_err());
+    assert!(crate::physical_quarantine::quarantined_handles_for(&client)? >= retained);
+    println!(
+        "P12_SOLO_B1_T2_CERTIFICATE_PASS source={name} synthetic_inactive_middle_NaN=true actual_final_padding_positive_zero=true persistent_OR_status=2 original_prior_all42_exact=true quarantine=true recovery=false"
+    );
+    Ok(())
+}
+macro_rules! case {
+    ($test:ident,$body:ident,$name:literal) => {
+        #[test]
+        #[ignore = "ROOT actual original SOLO B1/T2; --release --test-threads=1"]
+        fn $test() {
+            futures_lite::future::block_on($body($name)).unwrap();
+        }
+    };
+}
+case!(qwen08_solo_b1_t2, normal, "qwen35-0.8b");
+case!(qwen2_solo_b1_t2, normal, "qwen35-2b");
+case!(qwen08_solo_pressure, pressure, "qwen35-0.8b");
+case!(qwen2_solo_pressure, pressure, "qwen35-2b");
+case!(qwen08_solo_certificate, certificate, "qwen35-0.8b");
+case!(qwen2_solo_certificate, certificate, "qwen35-2b");
+#[cfg(feature = "portable-checkpoint")]
+case!(qwen08_solo_boundary, boundary, "qwen35-0.8b");
+#[cfg(feature = "portable-checkpoint")]
+case!(qwen2_solo_boundary, boundary, "qwen35-2b");
+#[cfg(feature = "hardware-abort-probe")]
+case!(qwen08_solo_abort, abort, "qwen35-0.8b");
+#[cfg(feature = "hardware-abort-probe")]
+case!(qwen2_solo_abort, abort, "qwen35-2b");
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_inference/solo.rs
@@ -0,0 +1,375 @@
+//! Qualifier-only original single-session owner for closed native B1/T2.
+use super::*;
+use crate::resident_program::SoloTemporalProgram;
+
+pub(super) struct SoloWorkspace {
+    pub(super) program: SoloTemporalProgram,
+    leases: Vec<(ResidentLease, DeviceAccess)>,
+    // Private dense P1 T0 bank; never a SessionAuthority/StateRoot/Ready output.
+    pub(super) middle: Vec<Domain>,
+    #[cfg(test)]
+    pub(super) last_native_projections: usize,
+    #[cfg(test)]
+    pub(super) last_fault_final_kv: Option<ResidentLease>,
+}
+impl SoloWorkspace {
+    pub(super) fn owned_bytes(&self) -> Result<usize> {
+        self.middle.iter().flat_map(|d| &d.leases).try_fold(
+            self.program.budget().owned_bytes()?,
+            |bytes, lease| {
+                bytes
+                    .checked_add(lease.bytes())
+                    .ok_or_else(|| Error::Limit("solo middle/flat footprint".into()))
+            },
+        )
+    }
+}
+
+impl ResidentSession {
+    /// ONE authority advances2 only on original commit. Neither private teacher
+    /// nor the P2 activation axis is registered as another request or root.
+    pub async fn prepare_state_temporal_b1_t2_admitted(
+        &self,
+        teachers: &[u32],
+    ) -> Result<Option<PreparedInference>> {
+        let model = &self.model;
+        model.service.pinned_cuda_queue()?;
+        if teachers.len() != 2 || !model.native_batch || model.capacity != 512 {
+            return Err(Error::Unsupported(
+                "solo temporal closed B1/T2/cap512".into(),
+            ));
+        }
+        let base = self.binding();
+        if base.source() != model.service.source_identity() {
+            return Err(Error::State("solo original source changed".into()));
+        }
+        let position =
+            usize::try_from(base.frontier()).map_err(|_| Error::Limit("solo frontier".into()))?;
+        if position
+            .checked_add(2)
+            .is_none_or(|end| end > model.capacity)
+        {
+            return Err(Error::Limit(
+                "solo temporal source context exhausted".into(),
+            ));
+        }
+        if teachers
+            .iter()
+            .any(|token| u64::from(*token) >= model.vocab)
+        {
+            return Err(Error::Malformed(
+                "solo temporal invalid original teacher".into(),
+            ));
+        }
+        model.validate_domains(&self.domains, true)?;
+        let attempt = self.authority.begin_advance(2)?;
+        let steps = std::array::from_fn(|teacher| RuntimeParameters {
+            token: teachers[teacher],
+            position: position + teacher,
+            valid_tokens: position + teacher + 1,
+        });
+        let allocation = model.allocation.lock().await;
+        let state_guard = model
+            .state_bytes
+            .checked_mul(2)
+            .ok_or_else(|| Error::Limit("solo state envelope".into()))?;
+        {
+            let workspace = model.workspace.lock().await;
+            model.service.ensure_usable()?;
+            if workspace
+                .batch
+                .as_ref()
+                .is_some_and(|batch| batch.program.graph_pinned())
+            {
+                return Err(Error::Unsupported(
+                    "solo refuses graph-pinned batch workspace".into(),
+                ));
+            }
+            let fresh = if workspace.solo.is_none() {
+                SoloTemporalProgram::preflight(&workspace.program)?
+                    .owned_bytes()?
+                    .checked_add(model.state_bytes)
+                    .ok_or_else(|| Error::Limit("solo cold P2/middle budget".into()))?
+            } else {
+                0
+            };
+            // Existing P1 is fully charged at model load. Complete NEW P2,
+            // all42 middle and all42 final state are admitted before any own
+            // allocation or queue ticket. Persistent unused P2 state is charged.
+            let required = fresh
+                .checked_add(model.state_bytes)
+                .ok_or_else(|| Error::Limit("solo complete new admission".into()))?;
+            if required > model.service.remaining_capacity()?
+                || state_guard as u64 > model.service.budget().remaining()?
+            {
+                return Ok(None);
+            }
+        }
+        let candidate = model.allocate_uninitialized_state()?;
+        let submit_ticket = || {
+            model.service.queue().submit(
+                model.service.budget(),
+                ResourceOwner {
+                    model: model.service.owner().model.clone(),
+                    session: format!("solo-temporal-{}", base.session().get()),
+                },
+                &[AllocationRequest {
+                    kind: AllocationClass::Scratch,
+                    bytes: state_guard as u64,
+                }],
+                None,
+            )
+        };
+        let mut slot = submit_ticket()?.wait().await?;
+        let mut workspace = model.workspace.lock().await;
+        if workspace.solo.is_none() {
+            let budget = SoloTemporalProgram::preflight(&workspace.program)?;
+            let cold = budget
+                .owned_bytes()?
+                .checked_add(model.state_bytes)
+                .ok_or_else(|| Error::Limit("solo cold footprint".into()))?;
+            if cold > model.service.remaining_capacity()? {
+                return Err(Error::Limit(
+                    "solo cold capacity changed after preflight".into(),
+                ));
+            }
+            slot.begin_submission()?;
+            let mut initialization = model.service.begin_initialization(base.root())?;
+            model
+                .service
+                .client()
+                .memory_cleanup()
+                .map_err(|e| Error::State(format!("solo cold cleanup:{e}")))?;
+            let mut leases = Vec::new();
+            let program = SoloTemporalProgram::prepare(
+                model.service.client(),
+                &workspace.program,
+                |role, count, data| {
+                    let role = match role {
+                        ResidentBufferRole::Source => BufferRole::ImmutableWeights,
+                        ResidentBufferRole::Value | ResidentBufferRole::Parameter => {
+                            BufferRole::Activation
+                        }
+                        ResidentBufferRole::Workspace | ResidentBufferRole::Status => {
+                            BufferRole::Scratch
+                        }
+                    };
+                    let lease = match data {
+                        AllocationInit::Empty => initialization.allocate_empty(count, role)?,
+                        AllocationInit::Upload(bytes) => {
+                            if bytes.len() != count {
+                                return Err(Error::State("solo initializer length".into()));
+                            }
+                            initialization.upload_bytes(bytes, role)?
+                        }
+                    };
+                    let handle = lease.handle();
+                    leases.push((lease, DeviceAccess::Write));
+                    Ok(handle)
+                },
+            )?;
+            for handle in program.owned_handles() {
+                lease_for(&leases, &handle)?;
+            }
+            let actual = leases.iter().try_fold(0usize, |n, (lease, _)| {
+                n.checked_add(lease.bytes())
+                    .ok_or_else(|| Error::Limit("solo allocation sum".into()))
+            })?;
+            if actual != budget.owned_bytes()?
+                || program.budget().owned_bytes()? != budget.owned_bytes()?
+            {
+                return Err(Error::State(
+                    "solo exact complete P2 allocation coverage".into(),
+                ));
+            }
+            let middle = model
+                .states
+                .iter()
+                .map(|state| {
+                    let mut allocate =
+                        |words| initialization.allocate_empty(words * 4, BufferRole::Scratch);
+                    make_domain(state, &mut allocate)
+                })
+                .collect::<Result<Vec<_>>>()?;
+            let middle_bytes = middle
+                .iter()
+                .flat_map(|d| &d.leases)
+                .map(ResidentLease::bytes)
+                .sum::<usize>();
+            let allocated = leases.len() + middle.iter().map(|d| d.leases.len()).sum::<usize>();
+            if middle_bytes != model.state_bytes {
+                return Err(Error::State("solo all-domain middle allocation".into()));
+            }
+            let initialized = initialization.complete_sync().await?;
+            if initialized.outputs().len() != allocated {
+                return Err(Error::State("solo complete cold initialization".into()));
+            }
+            workspace.solo = Some(SoloWorkspace {
+                program,
+                leases,
+                middle,
+                #[cfg(test)]
+                last_native_projections: 0,
+                #[cfg(test)]
+                last_fault_final_kv: None,
+            });
+            if workspace.solo.as_ref().unwrap().owned_bytes()? != cold {
+                return Err(Error::State("solo exact cold footprint changed".into()));
+            }
+            slot.finish()?;
+            drop(workspace);
+            slot = submit_ticket()?.wait().await?;
+            workspace = model.workspace.lock().await;
+        }
+        let solo = workspace
+            .solo
+            .as_ref()
+            .ok_or_else(|| Error::State("solo workspace absent".into()))?;
+        model.validate_domains(&solo.middle, true)?;
+        model.validate_domains(&candidate, false)?;
+        let bindings = [
+            bind_domains(&self.domains, &solo.middle, position)?,
+            bind_domains(&solo.middle, &candidate, position + 1)?,
+        ];
+        solo.program.validate(
+            model.service.client(),
+            &workspace.program,
+            &steps,
+            &bindings,
+        )?;
+        let p1 = workspace
+            .program
+            .execution_handles(ResidentOutput::StateOnly);
+        let p2 = solo.program.execution_handles();
+        let handles = p1.into_iter().chain(p2).collect::<Vec<_>>();
+        let mut accesses = workspace
+            .leases
+            .iter()
+            .chain(&solo.leases)
+            .filter(|(lease, _)| {
+                handles
+                    .iter()
+                    .any(|handle| same_handle(handle, &lease.handle()))
+            })
+            .cloned()
+            .collect::<Vec<_>>();
+        for handle in handles {
+            lease_for(&accesses, &handle)?;
+        }
+        accesses.extend(
+            self.domains
+                .iter()
+                .flat_map(|d| &d.leases)
+                .cloned()
+                .map(|lease| (lease, DeviceAccess::Read)),
+        );
+        accesses.extend(
+            solo.middle
+                .iter()
+                .chain(&candidate)
+                .flat_map(|d| &d.leases)
+                .cloned()
+                .map(|lease| (lease, DeviceAccess::Write)),
+        );
+        drop(allocation);
+        #[cfg(test)]
+        let capture_fault = SoloTemporalProgram::probe_is_armed();
+        slot.begin_submission()?;
+        let flight = model.service.submit(base.root(), &accesses)?;
+        let submitted = solo.program.launch(
+            model.service.client(),
+            &workspace.program,
+            &steps,
+            &bindings,
+        )?;
+        if submitted.logits.is_some()
+            || submitted.proposals.len() != model.states.len()
+            || submitted.fused_dispatches == 0
+        {
+            return Err(Error::State(
+                "solo original complete state-only receipt".into(),
+            ));
+        }
+        for ((version, buffers), expected) in submitted
+            .proposals
+            .iter()
+            .zip(workspace.program.plan().proposed_domains())
+        {
+            let domain = candidate
+                .iter()
+                .find(|d| d.slot == version.slot)
+                .ok_or_else(|| Error::State("solo final foreign domain".into()))?;
+            let a = buffers.handles();
+            let b = domain.buffers.handles();
+            if version != expected
+                || a.len() != b.len()
+                || a.iter().zip(&b).any(|(a, b)| !same_handle(a, b))
+            {
+                return Err(Error::State(
+                    "solo original final version/physical candidate".into(),
+                ));
+            }
+        }
+        #[cfg(test)]
+        {
+            let solo = workspace.solo.as_mut().unwrap();
+            solo.last_native_projections = submitted.fused_dispatches;
+            if capture_fault {
+                solo.last_fault_final_kv = Some(
+                    candidate
+                        .iter()
+                        .find(|d| matches!(d.buffers, StateBuffers::Kv { .. }))
+                        .ok_or_else(|| Error::State("solo fault needs original KV".into()))?
+                        .leases[0]
+                        .clone(),
+                );
+            }
+        }
+        #[cfg(feature = "hardware-abort-probe")]
+        if model
+            .abort_after_launch
+            .fetch_update(
+                std::sync::atomic::Ordering::AcqRel,
+                std::sync::atomic::Ordering::Acquire,
+                |remaining| remaining.checked_sub(1),
+            )
+            .is_ok_and(|remaining| remaining == 1)
+        {
+            model
+                .launch_seen
+                .store(flight.handles().len(), std::sync::atomic::Ordering::Release);
+            std::future::pending::<()>().await;
+        }
+        let status = lease_for(&accesses, &submitted.status)?.readback(0, 4)?;
+        let promoted = candidate
+            .iter()
+            .flat_map(|d| &d.leases)
+            .cloned()
+            .collect::<Vec<_>>();
+        let (ready, read) = flight
+            .complete_with_readback(&promoted, &[status], |data| {
+                if data.len() != 1 || data[0].len() != 4 || !all_zero(&data[0]) {
+                    return Err(Error::State("solo temporal status refused".into()));
+                }
+                Ok(())
+            })
+            .await?;
+        slot.finish()?;
+        if read.len() != 1 || read[0].len() != 4 {
+            return Err(Error::State("solo original terminal4B".into()));
+        }
+        Ok(Some(PreparedInference {
+            attempt,
+            domains: candidate,
+            ready,
+            readback: InferenceReadback {
+                dispatches: submitted.dispatches,
+                state_only_steps: 2,
+                logit_steps: 0,
+                bytes: 4,
+                chunk_bytes: 4,
+                ..Default::default()
+            },
+        }))
+    }
+}
--- a/crates/zerogravity-gpu/src/resident_program.rs
+++ b/crates/zerogravity-gpu/src/resident_program.rs
@@ -40,7 +40,9 @@
 mod batch;
 pub(crate) use batch::BatchProgram;
 #[cfg(all(feature = "qualification", feature = "cuda"))]
-pub(crate) use batch::{BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay, TemporalProgram};
+pub(crate) use batch::{
+    BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay, SoloTemporalProgram, TemporalProgram,
+};
 
 #[cfg(feature = "qualification")]
 #[path = "resident_projection_qualification.rs"]
@@ -1249,36 +1251,8 @@
         }
         dispatches += 2;
         let nodes = output.nodes(&self.plan);
-        for (index, node) in nodes.iter().enumerate() {
-            dispatches += self.launch_node(client, index, node, p, bindings)?;
-            // A preflight-validated View fully writes its dense destination and
-            // certifies every copied word in the same kernel. All other output
-            // certificates (including Contiguous aliases) remain separate.
-            if matches!(node.operation, PlanOperation::View(_)) {
-                continue;
-            }
-            for (out_index, id) in node.outputs.iter().enumerate() {
-                let value = &self.values[id.0 as usize];
-                if let PlanOperation::AttentionQk { geometry, .. } = &node.operation
-                    && out_index == 1
-                {
-                    unsafe {
-                        crate::resident_bindings::certify_causal::launch_unchecked(
-                            client,
-                            grid(value.elements),
-                            threads(),
-                            arg(&value.handle, value.elements),
-                            arg(&self.status, 1),
-                            p.valid_tokens as u32,
-                            geometry.query_heads,
-                            self.plan.prefix_capacity(),
-                        );
-                    }
-                } else {
-                    self.certify(client, &value.handle, value.elements);
-                }
-                dispatches += 1;
-            }
+        for index in 0..nodes.len() {
+            dispatches += self.launch_checked_node(client, index, p, bindings)?;
         }
         let logits = self
             .plan
@@ -1346,6 +1320,52 @@
             dispatches,
             fused_dispatches: 0,
         })
+    }
+    /// Original one-node emitter and output certificates, factored without
+    /// changing the ordinary ordered launch body's arithmetic/order/count.
+    fn launch_checked_node(
+        &self,
+        client: &Client,
+        index: usize,
+        p: RuntimeParameters,
+        bindings: &[StateBinding],
+    ) -> Result<usize> {
+        let node = self
+            .plan
+            .nodes()
+            .get(index)
+            .ok_or_else(|| Error::State("original P1 node index".into()))?;
+        let mut dispatches = 0;
+        dispatches += self.launch_node(client, index, node, p, bindings)?;
+        // A preflight-validated View fully writes its dense destination and
+        // certifies every copied word in the same kernel. All other output
+        // certificates (including Contiguous aliases) remain separate.
+        if matches!(node.operation, PlanOperation::View(_)) {
+            return Ok(dispatches);
+        }
+        for (out_index, id) in node.outputs.iter().enumerate() {
+            let value = &self.values[id.0 as usize];
+            if let PlanOperation::AttentionQk { geometry, .. } = &node.operation
+                && out_index == 1
+            {
+                unsafe {
+                    crate::resident_bindings::certify_causal::launch_unchecked(
+                        client,
+                        grid(value.elements),
+                        threads(),
+                        arg(&value.handle, value.elements),
+                        arg(&self.status, 1),
+                        p.valid_tokens as u32,
+                        geometry.query_heads,
+                        self.plan.prefix_capacity(),
+                    );
+                }
+            } else {
+                self.certify(client, &value.handle, value.elements);
+            }
+            dispatches += 1;
+        }
+        Ok(dispatches)
     }
     #[allow(clippy::too_many_arguments)] // Explicit typed source/destination ranges.
     fn copy<const CERTIFY_FINITE: bool>(
--- a/crates/zerogravity-gpu/src/resident_program/batch.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch.rs
@@ -68,6 +68,10 @@
 mod temporal;
 #[cfg(all(feature = "qualification", feature = "cuda"))]
 pub(crate) use temporal::TemporalProgram;
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+mod solo;
+#[cfg(all(feature = "qualification", feature = "cuda"))]
+pub(crate) use solo::SoloTemporalProgram;
 
 impl BatchProgram {
     /// New allocations only: the original source handles remain shared.
--- /dev/null
+++ b/crates/zerogravity-gpu/src/resident_program/batch/solo.rs
@@ -0,0 +1,470 @@
+//! Closed one-authority/two-teacher lowering, never two fictitious requests.
+use super::temporal::{Dependencies, Stage};
+use super::*;
+
+#[cfg(test)]
+std::thread_local! {
+    static INACTIVE_NAN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
+}
+
+/// P1 values/quantizers remain the original ResidentProgram. Only stateless
+/// activations use the original P2 lowering; P2 state buffers are never read.
+pub(crate) struct SoloTemporalProgram {
+    flat: BatchProgram,
+    dependencies: Dependencies,
+}
+impl SoloTemporalProgram {
+    pub(crate) fn preflight(original: &ResidentProgram) -> Result<ResidentPreflight> {
+        Dependencies::new(&original.plan)?;
+        if original.plan.prefix_capacity() != 512
+            || original
+                .plan
+                .values()
+                .iter()
+                .any(|shape| match shape.capacity_elements() {
+                    Ok(words) => words > crate::program_kernels::MAX_WORD_OPS,
+                    Err(_) => true,
+                })
+            || original
+                .preflight
+                .states
+                .iter()
+                .any(|state| state.bytes / 4 > crate::program_kernels::MAX_WORD_OPS)
+        {
+            return Err(Error::Unsupported(
+                "solo temporal original P1/cap512 bounds".into(),
+            ));
+        }
+        BatchProgram::preflight(original, 2)
+    }
+    pub(crate) fn prepare(
+        client: &Client,
+        original: &ResidentProgram,
+        allocate: impl FnMut(ResidentBufferRole, usize, AllocationInit<'_>) -> Result<Handle>,
+    ) -> Result<Self> {
+        Self::preflight(original)?;
+        Ok(Self {
+            flat: BatchProgram::prepare(client, original, 2, allocate)?,
+            dependencies: Dependencies::new(&original.plan)?,
+        })
+    }
+    pub(crate) fn budget(&self) -> &ResidentPreflight {
+        self.flat.budget()
+    }
+    pub(crate) fn owned_handles(&self) -> Vec<Handle> {
+        self.flat.owned_handles()
+    }
+    pub(crate) fn execution_handles(&self) -> Vec<Handle> {
+        self.flat.execution_handles(ResidentOutput::StateOnly)
+    }
+    #[cfg(test)]
+    pub(crate) fn expected_native_projections(&self) -> usize {
+        self.dependencies
+            .flat
+            .iter()
+            .filter(|&&index| {
+                matches!(
+                    self.flat.nodes[index].operation,
+                    PlanOperation::Project { .. }
+                )
+            })
+            .count()
+    }
+    #[cfg(test)]
+    pub(crate) fn arm_inactive_nan_probe() {
+        INACTIVE_NAN.with(|flag| assert!(!flag.replace(true), "solo NaN probe already armed"));
+    }
+    #[cfg(test)]
+    pub(crate) fn probe_is_armed() -> bool {
+        INACTIVE_NAN.with(std::cell::Cell::get)
+    }
+    #[cfg(test)]
+    pub(crate) fn diagnostic_original_status(original: &ResidentProgram) -> Handle {
+        original.status.clone()
+    }
+
+    /// Entire source/value/physical state chain is checked before owner's flight.
+    /// The middle prior is a private producer in this flight, never a future Root.
+    pub(crate) fn validate(
+        &self,
+        client: &Client,
+        original: &ResidentProgram,
+        steps: &[RuntimeParameters; 2],
+        bindings: &[Vec<StateBinding>; 2],
+    ) -> Result<()> {
+        if self.flat.rows != 2
+            || self.flat.graph_pinned()
+            || !Arc::ptr_eq(&original.plan_lineage, &self.flat.inner.plan_lineage)
+            || !Arc::ptr_eq(&original._source_owner, &self.flat.inner._source_owner)
+            || self.dependencies.count != original.plan.state_nodes().len()
+            || self.dependencies.flat.is_empty()
+            || self.dependencies.causal.is_empty()
+            || steps[0].position.checked_add(1) != Some(steps[1].position)
+        {
+            return Err(Error::State(
+                "solo temporal original lineage/teacher layout".into(),
+            ));
+        }
+        Self::preflight(original)?;
+        for (parameter, states) in steps.iter().zip(bindings) {
+            original.validate_runtime(client, *parameter, states)?;
+            for node in original.plan.nodes() {
+                if let PlanOperation::Embed { tensor, .. } = &node.operation
+                    && parameter.token as usize >= tensor.rows
+                {
+                    return Err(Error::Malformed(
+                        "solo teacher exceeds original vocabulary".into(),
+                    ));
+                }
+            }
+        }
+        if bindings[0].len() != bindings[1].len() {
+            return Err(Error::State("solo middle coverage".into()));
+        }
+        let mut priors = Vec::new();
+        let mut middle = Vec::new();
+        let mut finals = Vec::new();
+        for (first, second) in bindings[0].iter().zip(&bindings[1]) {
+            if first.slot != second.slot {
+                return Err(Error::State("solo middle domain order".into()));
+            }
+            let a = first.candidate.buffers();
+            let b = second.prior.buffers();
+            if a.len() != b.len() {
+                return Err(Error::State(
+                    "solo actual middle producer/consumer binding".into(),
+                ));
+            }
+            for (a, b) in a.iter().zip(&b) {
+                let x = BufferRange::new(&a.handle)?;
+                let y = BufferRange::new(&b.handle)?;
+                if a.elements != b.elements
+                    || x.service != y.service
+                    || x.memory != y.memory
+                    || x.bytes != y.bytes
+                {
+                    return Err(Error::State(
+                        "solo actual middle producer/consumer binding".into(),
+                    ));
+                }
+            }
+            for (state, ranges) in [
+                (&first.prior, &mut priors),
+                (&first.candidate, &mut middle),
+                (&second.candidate, &mut finals),
+            ] {
+                for value in state.buffers() {
+                    ranges.push(BufferRange::new(&value.handle)?);
+                }
+            }
+        }
+        // No middle/final producer aliases a committed input, sibling domain,
+        // or either original mathematical workspace/source allocation.
+        let p1 = original
+            .execution_handles(ResidentOutput::StateOnly)
+            .into_iter()
+            .map(|h| BufferRange::new(&h))
+            .collect::<Result<Vec<_>>>()?;
+        let p2 = self
+            .flat
+            .owned_handles()
+            .into_iter()
+            .map(|h| BufferRange::new(&h))
+            .collect::<Result<Vec<_>>>()?;
+        if p2
+            .iter()
+            .any(|a| a.service != client.service_id() || p1.iter().any(|b| a.overlaps(b)))
+        {
+            return Err(Error::State(
+                "solo private P1/P2 allocations alias/service".into(),
+            ));
+        }
+        let writes = middle.iter().chain(&finals).collect::<Vec<_>>();
+        for (index, write) in writes.iter().enumerate() {
+            if write.service != client.service_id()
+                || priors
+                    .iter()
+                    .chain(&p1)
+                    .chain(&p2)
+                    .any(|r| write.overlaps(r))
+                || writes[..index].iter().any(|r| write.overlaps(r))
+            {
+                return Err(Error::State(
+                    "solo middle/final nonaliasing physical chain".into(),
+                ));
+            }
+        }
+        for (index, shape) in original.plan.values().iter().enumerate() {
+            let words = shape.capacity_elements()?;
+            original.values[index].check(client, words)?;
+            self.flat.inner.values[index].check(
+                client,
+                words
+                    .checked_mul(2)
+                    .ok_or_else(|| Error::Limit("solo P2 value bound".into()))?,
+            )?;
+            for teacher in 0..2 {
+                Self::bridge_span(
+                    &original.values[index],
+                    &self.flat.inner.values[index],
+                    teacher,
+                )?;
+            }
+        }
+        DeviceBuffer::new(original.status.clone(), 1).check(client, 1)?;
+        DeviceBuffer::new(original.token.clone(), 1).check(client, 1)?;
+        DeviceBuffer::new(self.flat.inner.status.clone(), 1).check(client, 1)?;
+        DeviceBuffer::new(self.flat.inner.token.clone(), 2).check(client, 2)?;
+        DeviceBuffer::new(self.flat.positions.clone(), 2).check(client, 2)?;
+        for &index in &self.dependencies.flat {
+            let node = &self.flat.nodes[index];
+            if node.state.is_some() || !Dependencies::reads(node).is_empty() {
+                return Err(Error::State("solo stateless dependency changed".into()));
+            }
+        }
+        Ok(())
+    }
+    fn bridge_span(small: &DeviceBuffer, large: &DeviceBuffer, teacher: usize) -> Result<usize> {
+        if teacher >= 2 || small.elements.checked_mul(2) != Some(large.elements) {
+            return Err(Error::State("solo original/P2 activation geometry".into()));
+        }
+        let offset = teacher
+            .checked_mul(small.elements)
+            .ok_or_else(|| Error::Limit("solo bridge offset".into()))?;
+        let a = BufferRange::new(&small.handle)?;
+        let b = BufferRange::new(&large.handle)?;
+        if offset
+            .checked_add(small.elements)
+            .is_none_or(|end| end > large.elements)
+            || a.service != b.service
+            || a.overlaps(&b)
+            || a.bytes.end - a.bytes.start < bytes(small.elements, 4)? as u64
+            || b.bytes.end - b.bytes.start < bytes(large.elements, 4)? as u64
+        {
+            return Err(Error::State("solo bridge physical range/alias".into()));
+        }
+        Ok(offset)
+    }
+    fn bridge(
+        &self,
+        client: &Client,
+        original: &ResidentProgram,
+        id: ValueId,
+        teacher: usize,
+        to_flat: bool,
+    ) -> Result<usize> {
+        let small = &original.values[id.0 as usize];
+        let large = &self.flat.inner.values[id.0 as usize];
+        let offset = Self::bridge_span(small, large, teacher)?;
+        if to_flat {
+            original.copy::<false>(
+                client,
+                &small.handle,
+                small.elements,
+                &large.handle,
+                large.elements,
+                TensorView::vector(0, small.elements),
+                offset,
+            );
+        } else {
+            original.copy::<false>(
+                client,
+                &large.handle,
+                large.elements,
+                &small.handle,
+                small.elements,
+                TensorView::vector(offset, small.elements),
+                0,
+            );
+        }
+        Ok(1)
+    }
+    fn certify_state(client: &Client, original: &ResidentProgram, state: &StateBuffers) -> usize {
+        for value in state.buffers() {
+            original.certify(client, &value.handle, value.elements);
+        }
+        state.buffers().len()
+    }
+    #[cfg(test)]
+    fn poison_inactive(client: &Client, state: &StateBuffers) -> Result<usize> {
+        let StateBuffers::Kv { keys, .. } = state else {
+            return Ok(0);
+        };
+        let start = keys
+            .handle
+            .offset_start
+            .unwrap_or(0)
+            .checked_add(bytes(
+                keys.elements
+                    .checked_sub(1)
+                    .ok_or_else(|| Error::State("solo NaN empty KV".into()))?,
+                4,
+            )? as u64)
+            .ok_or_else(|| Error::Limit("solo NaN start".into()))?;
+        let end = start
+            .checked_add(4)
+            .ok_or_else(|| Error::Limit("solo NaN end".into()))?;
+        let mut view = keys.handle.clone();
+        view.offset_start = Some(start);
+        view.offset_end = Some(
+            view.size()
+                .checked_sub(end)
+                .ok_or_else(|| Error::State("solo NaN extent".into()))?,
+        );
+        BufferRange::new(&view)?;
+        unsafe {
+            crate::resident_bindings::write_token::launch_unchecked(
+                client,
+                grid(1),
+                threads(),
+                arg(&view, 1),
+                0x7fc00001u32,
+            );
+        }
+        Ok(1)
+    }
+    pub(crate) fn launch(
+        &self,
+        client: &Client,
+        original: &ResidentProgram,
+        steps: &[RuntimeParameters; 2],
+        bindings: &[Vec<StateBinding>; 2],
+    ) -> Result<ResidentSubmission> {
+        self.validate(client, original, steps, bindings)?;
+        unsafe {
+            crate::program_kernels::clear_step_status::launch_unchecked(
+                client,
+                grid(1),
+                threads(),
+                arg(&original.status, 1),
+            );
+            crate::program_kernels::clear_step_status::launch_unchecked(
+                client,
+                grid(1),
+                threads(),
+                arg(&self.flat.inner.status, 1),
+            );
+        }
+        let mut dispatches = 2 + self.flat.stage_teacher_parameters(client, steps);
+        let mut fused_dispatches = 0;
+        for stage in &self.dependencies.stages {
+            match stage {
+                Stage::Flat(index) => {
+                    let receipt = self.flat.launch_checked_node(client, *index, steps)?;
+                    dispatches += receipt.dispatches;
+                    fused_dispatches += receipt.fused_dispatches;
+                }
+                Stage::Causal { nodes, .. } => {
+                    for (teacher, parameter) in steps.iter().enumerate() {
+                        unsafe {
+                            crate::resident_bindings::write_token::launch_unchecked(
+                                client,
+                                grid(1),
+                                threads(),
+                                arg(&original.token, 1),
+                                parameter.token,
+                            );
+                        }
+                        dispatches += 1;
+                        for index in nodes.clone() {
+                            let node = &original.plan.nodes()[index];
+                            for id in &node.inputs {
+                                dispatches += self.bridge(client, original, *id, teacher, false)?;
+                            }
+                            dispatches += original.launch_checked_node(
+                                client,
+                                index,
+                                *parameter,
+                                &bindings[teacher],
+                            )?;
+                            if let Some(effect) = node.state {
+                                let state = &bindings[teacher]
+                                    .iter()
+                                    .find(|state| state.slot == effect.write.slot)
+                                    .ok_or_else(|| {
+                                        Error::State("solo original effect domain".into())
+                                    })?
+                                    .candidate;
+                                #[cfg(test)]
+                                if teacher == 0
+                                    && matches!(effect.write.slot, StateSlot::Kv(_))
+                                    && INACTIVE_NAN.with(|flag| flag.replace(false))
+                                {
+                                    if steps.iter().any(|p| p.valid_tokens >= 512) {
+                                        return Err(Error::State(
+                                            "solo NaN needs inactive KV".into(),
+                                        ));
+                                    }
+                                    dispatches += Self::poison_inactive(client, state)?;
+                                }
+                                // All original state words, including inactive KV, are
+                                // certified BEFORE this producer can feed teacher1.
+                                dispatches += Self::certify_state(client, original, state);
+                            }
+                            for id in &node.outputs {
+                                dispatches += self.bridge(client, original, *id, teacher, true)?;
+                            }
+                        }
+                    }
+                }
+            }
+        }
+        unsafe {
+            crate::resident_bindings::merge_status::launch_unchecked(
+                client,
+                grid(1),
+                threads(),
+                arg(&self.flat.inner.status, 1),
+                arg(&original.status, 1),
+                1,
+            );
+        }
+        dispatches += 1;
+        let mut proposals = Vec::new();
+        for version in original.plan.proposed_domains() {
+            let state = &bindings[1]
+                .iter()
+                .find(|state| state.slot == version.slot)
+                .ok_or_else(|| Error::State("solo final domain closure".into()))?
+                .candidate;
+            dispatches += Self::certify_state(client, original, state);
+            proposals.push((*version, state.clone()));
+        }
+        let mut quant_status = Vec::new();
+        let mut embedding_status = Vec::new();
+        for (program, selected, rows) in [
+            (&self.flat.inner, &self.dependencies.flat, 2),
+            (original, &self.dependencies.causal, 1),
+        ] {
+            for &index in selected {
+                match &program.plan.nodes()[index].operation {
+                    PlanOperation::Project { tensor } => {
+                        if let Some(layout) = program.tensors[&tensor.name].layout {
+                            let flags = &program.node_status[index];
+                            quant_status.push((
+                                flags.block.as_ref().unwrap().clone(),
+                                rows * layout.blocks,
+                                flags.row.as_ref().unwrap().clone(),
+                                rows * layout.rows,
+                            ));
+                        }
+                    }
+                    PlanOperation::Embed { tensor, .. } => embedding_status.push((
+                        program.node_status[index].block.as_ref().unwrap().clone(),
+                        rows * tensor.columns,
+                    )),
+                    _ => {}
+                }
+            }
+        }
+        Ok(ResidentSubmission {
+            logits: None,
+            proposals,
+            status: original.status.clone(),
+            quant_status,
+            embedding_status,
+            dispatches,
+            fused_dispatches,
+        })
+    }
+}
--- a/crates/zerogravity-gpu/src/resident_program/batch/temporal.rs
+++ b/crates/zerogravity-gpu/src/resident_program/batch/temporal.rs
@@ -9,7 +9,7 @@
 }
 
 #[derive(Clone, Debug, PartialEq, Eq)]
-enum Stage {
+pub(super) enum Stage {
     Flat(usize),
     Causal {
         nodes: Range<usize>,
@@ -19,14 +19,14 @@
 
 /// Derived from the SAME immutable plan; no token/layer schedule is copied.
 #[derive(Clone, Debug)]
-struct Dependencies {
-    stages: Vec<Stage>,
-    flat: BTreeSet<usize>,
-    causal: BTreeSet<usize>,
-    count: usize,
+pub(super) struct Dependencies {
+    pub(super) stages: Vec<Stage>,
+    pub(super) flat: BTreeSet<usize>,
+    pub(super) causal: BTreeSet<usize>,
+    pub(super) count: usize,
 }
 impl Dependencies {
-    fn new(plan: &ForwardPlan) -> Result<Self> {
+    pub(super) fn new(plan: &ForwardPlan) -> Result<Self> {
         ResidentProgram::require_legacy_contract(plan)?;
         let result = Self::nodes(plan.nodes(), plan.state_nodes().len(), plan.values().len())?;
         let actual = result
@@ -215,7 +215,7 @@
         }
         Ok(effect.read.slot)
     }
-    fn reads(node: &PlanNode) -> Vec<StateVersion> {
+    pub(super) fn reads(node: &PlanNode) -> Vec<StateVersion> {
         let mut reads = node
             .state
             .map(|effect| vec![effect.read])

````

## Source: .agents/runs/serving-and-speculation-native-solo-temporal-20261020/ROOT_WIRING.patch

Original UTF-8 bytes: 216; SHA256 `f78f97ac271103c3616d0a4ece86ddd65fc0084c49cb5b3a7fe457f11030d9b8`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk.rs
@@ -675,3 +675,4 @@
 }
 
 mod temporal;
+
mod solo;

````

## Source: .agents/runs/serving-and-speculation-native-solo-temporal-20261020/followup-v2/CORRECT_ROOT_WIRING.patch

Original UTF-8 bytes: 217; SHA256 `0dd483406e90295ff1a0e415e9d6118dc2e54db9cad82d4933ad03260596c2b0`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk.rs
@@ -675,3 +675,5 @@
 }
 
 mod temporal;
+
+mod solo;

````

## Source: .agents/runs/serving-and-speculation-native-solo-temporal-20261020/followup-v2/frozen/CORRECT_ROOT_WIRING.patch

Original UTF-8 bytes: 217; SHA256 `0dd483406e90295ff1a0e415e9d6118dc2e54db9cad82d4933ad03260596c2b0`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk.rs
+++ b/crates/zerogravity-gpu/src/resident_inference/prefix_qualification/chunk.rs
@@ -675,3 +675,5 @@
 }
 
 mod temporal;
+
+mod solo;

````

