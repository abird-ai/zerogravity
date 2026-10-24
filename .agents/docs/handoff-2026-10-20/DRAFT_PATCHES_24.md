# Draft Patches — volume 24

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/finalization-v2/FINALIZATION_FOLLOWUP.patch

Original UTF-8 bytes: 10172; SHA256 `2896b710d03bd9e676432db7cb3a1b54ea40754140c0bd0df2eac1d293d02fd9`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ completion_v2.py
@@ -0,0 +1,52 @@
+"""Additive bounded final receipts; no process, activity or policy change."""
+import json
+from pathlib import Path
+
+from container_contracts import HERE, TREE_LIMIT, check_file, frozen, read, require, sha, tree_bound
+
+PARENT = '07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee'
+MANIFEST = HERE / 'finalization-v2/prepared-manifest.json'
+COMPLETION_BYTES = 4 * 1024**2
+COMPLETION_SLOTS = 2
+RESERVED_BYTES = COMPLETION_BYTES * COMPLETION_SLOTS
+
+
+def verify_preparation():
+    frozen(PARENT)
+    manifest = read(MANIFEST)
+    require(manifest['parent_freeze_sha256'] == PARENT and
+            manifest['max_completion_bytes'] == COMPLETION_BYTES and
+            manifest['reserved_completion_slots'] == COMPLETION_SLOTS,
+            'exact additive finalization policy')
+    for item in manifest['files']:
+        check_file(HERE / item['file'], item)
+    return sha(MANIFEST)
+
+
+def reserve(snapshot):
+    require(snapshot['limit'] == TREE_LIMIT and
+            type(snapshot['bytes']) is int and 0 <= snapshot['bytes'] <= TREE_LIMIT - RESERVED_BYTES and
+            type(snapshot['files']) is int and 0 <= snapshot['files'] <= 256 - COMPLETION_SLOTS,
+            'reserve two bounded completion receipts and file slots before final write')
+
+
+def encode(value):
+    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
+    require(len(raw) <= COMPLETION_BYTES, 'bounded complete receipt before any final write')
+    return raw
+
+
+def write_completion(directory, name, value):
+    require(name in ('report.json', 'validation.json'), 'only reviewed final receipt names')
+    before_sha = verify_preparation()
+    raw = encode(value)
+    before = tree_bound(directory)
+    reserve(before)
+    path = Path(directory) / name
+    with path.open('xb') as stream:
+        stream.write(raw)
+    after = tree_bound(directory)
+    require(after['bytes'] == before['bytes'] + len(raw) and after['files'] == before['files'] + 1,
+            'complete owned tree after final write; no concurrent uncounted artifact')
+    require(verify_preparation() == before_sha, 'unchanged finalization sources across final write')
+    return after
--- /dev/null
+++ test_completion_v2.py
@@ -0,0 +1,44 @@
+"""Prepared source-free boundary fixtures; ROOT alone executes."""
+from pathlib import Path
+import tempfile
+import unittest
+from unittest.mock import patch
+
+from completion_v2 import COMPLETION_BYTES, RESERVED_BYTES, TREE_LIMIT, encode, reserve, write_completion
+
+
+class Completion(unittest.TestCase):
+    def test_exact_reserved_byte_and_slot_boundary(self):
+        reserve(dict(limit=TREE_LIMIT, bytes=TREE_LIMIT-RESERVED_BYTES, files=254))
+        for extra in (dict(bytes=TREE_LIMIT-RESERVED_BYTES+1), dict(files=255), dict(files=True)):
+            value=dict(limit=TREE_LIMIT, bytes=TREE_LIMIT-RESERVED_BYTES, files=254)
+            value.update(extra)
+            with self.assertRaises(ValueError): reserve(value)
+
+    def test_metadata_limit_refuses_before_file_creation(self):
+        with tempfile.TemporaryDirectory() as root:
+            with patch('completion_v2.verify_preparation', return_value='a'*64):
+                with self.assertRaises(ValueError):
+                    write_completion(root, 'report.json', {'too_large':'x'*COMPLETION_BYTES})
+            self.assertEqual(list(Path(root).iterdir()), [])
+
+    def test_full_postwrite_tree_and_exclusive_names(self):
+        with tempfile.TemporaryDirectory() as root:
+            with patch('completion_v2.verify_preparation', return_value='a'*64):
+                result=write_completion(root,'report.json',{'status':'fixture-only'})
+                self.assertEqual(result['files'],1)
+                self.assertEqual(result['bytes'],len(encode({'status':'fixture-only'})))
+                with self.assertRaises(FileExistsError):
+                    write_completion(root,'report.json',{'status':'fixture-only'})
+
+    def test_postwrite_unaccounted_file_is_fatal(self):
+        with tempfile.TemporaryDirectory() as root:
+            snapshots=[dict(limit=TREE_LIMIT,bytes=0,files=0), dict(limit=TREE_LIMIT,bytes=999,files=2)]
+            with patch('completion_v2.verify_preparation',return_value='a'*64), \
+                    patch('completion_v2.tree_bound',side_effect=snapshots):
+                with self.assertRaises(ValueError):
+                    write_completion(root,'validation.json',{'status':'fixture-only'})
+
+
+if __name__ == '__main__':
+    unittest.main()
--- run_container.py
+++ run_container_v2.py
@@ -19,6 +19,7 @@
     read,require,sha,tree_bound,write)
 from launch_derivative import derive
 from namespace_control import Controller
+from completion_v2 import verify_preparation as verify_finalization, write_completion
 
 
 def idle(bench,url):
@@ -71,6 +72,7 @@
     parser.add_argument('--output',type=Path,required=True)
     args=parser.parse_args()
     require(args.run,'ROOT reviewed explicit actual execution only')
+    verify_finalization()
     parent=frozen(args.own_freeze); host_parents(parent)
     require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),'host controller unprofiled')
     benchmark=Path(parent['parents']['benchmark'])
@@ -124,6 +126,7 @@
         performance_timing_usable=False,matched_hardware_counts_qualified=False,
         source_correct_goodput_qualified=False,numerical_profiles_identical=False,
         native_or_stock_math_changes=False,cold_process_has_unmeasured_collector_request=False)
+    report['artifact_finalization_followup_sha256']=verify_finalization()
     allowed=os.sched_getaffinity(0)
     require(set(protocol['cpu_affinity'])<=allowed,'same actual four available client/provider CPUs')
     urllib.request.install_opener(opener())
@@ -217,7 +220,7 @@
         except BaseException as error: failures.append('post-run complete payload closure: '+repr(error))
         report['finalization_errors']=failures
         if failures: report['status']='UNQUALIFIED'
-        write(output/'report.json',report)
+        write_completion(output,'report.json',report)
         if failures:
             raise RuntimeError('diagnostic finalization failed; primary='+str(report.get('failure'))+'; '+repr(failures))
 
--- validate_container.py
+++ validate_container_v2.py
@@ -12,6 +12,7 @@
 from container_activity import summarize
 from container_contracts import HERE,FILE_LIMIT,TREE_LIMIT,build_binding,check_file,fingerprint,frozen,host_parents,read,require,sha,tree_bound,write
 from launch_derivative import derive
+from completion_v2 import verify_preparation as verify_finalization, write_completion
 
 
 def successful_owner(receipt,parent):
@@ -117,10 +118,13 @@
     parser.add_argument('--own-freeze',required=True)
     parser.add_argument('--output',type=Path,required=True)
     args=parser.parse_args()
+    verify_finalization()
     parent=frozen(args.own_freeze); host_parents(parent)
     output=args.output.resolve(strict=True)
     require(output.parent==HERE,'owned diagnostic output')
     report=read(output/'report.json')
+    require(report.get('artifact_finalization_followup_sha256')==verify_finalization(),
+            'actual report used reviewed final receipt reservation')
     require(report['status']=='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
             and report['own_freeze_sha256']==args.own_freeze
             and report['benchmark_v8_sha256']==parent['benchmark_v8_manifest']['sha256']
@@ -203,13 +207,14 @@
     require(native_summary is not None and stock_summary,'both actual canonical workload activity scopes present')
     result=dict(status='PASS_CONTAINER_RECORDED_WORKLOAD_ACTIVITY_DIAGNOSTIC_ONLY',
         own_freeze_sha256=args.own_freeze,report_sha256=sha(output/'report.json'),
-        native=native_summary,stock_recorded=stock_summary,artifact_bounds=tree_bound(output),
+        native=native_summary,stock_recorded=stock_summary,artifact_bounds_before_validation_receipt=tree_bound(output),
         same_canonical_workload_policy_and_caps_checked=True,plain_observed_wire_outputs_exact=True,
         source_profiles_numerically_equal=False,source_correct_goodput_qualified=False,
         matched_complete_hardware_kernel_counts_qualified=False,stock_collector_losslessness_verified=False,
         timing_usable=False,device_profiler_allocation_admission_qualified=False)
     frozen(args.own_freeze); host_parents(parent); verify_preparation(report['benchmark_v8_sha256'])
-    write(output/'validation.json',result)
+    result['artifact_finalization_followup_sha256']=verify_finalization()
+    write_completion(output,'validation.json',result)
 
 
 if __name__=='__main__': main()
--- /dev/null
+++ finalization-v2/README.md
@@ -0,0 +1,20 @@
+# Additive final receipt artifact bound
+
+Author: spine_review. Original container07a86bf6 and capacity4337ad63 immutable.
+
+- New run_container_v2.py and validate_container_v2.py retain original CLI,
+  activity/source/window/model/capacity/cgroup/workload/lifecycle/math behavior.
+- Before EACH final receipt write: serialize finite JSON <=4MiB; full existing
+  tree <=1GiB-8MiB and <=254files, reserving two4MiB receipt/file slots. Then
+  exclusive write and full final tree assertion, including exact observed
+  byte/file increments. No post-limit receipt can become accepted evidence.
+- Aggregate1GiB/per-owner768MiB/per-file576MiB and raw512MiB/4M remain unchanged.
+  Reservation is acceptance bookkeeping, not filesystem quota or device/RAM cap.
+- New reports bind this additive manifest; validator refuses old unreserved
+  reports. Native bootstrap/collector still attest unchanged07a/4337, correctly.
+- ROOT first runs test_completion_v2.py (four prepared synthetic methods), then
+  uses v2 runner/validator with identical original arguments. New capacity C++
+  still requires actual ROOT build; no author execution occurred.
+
+Author validation: four Python ASTs parse only, small source/frozen hashes.
+No fixtures/build/runtime/provider/GPU/model or payload hashing.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/finalization-v2/pre-review/FINALIZATION_FOLLOWUP.patch

Original UTF-8 bytes: 10121; SHA256 `7562dc5eef71be69824d0fd1d5ce3e46fa164380f49c1fa3ba378bb39f08b7d6`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ completion_v2.py
@@ -0,0 +1,52 @@
+"""Additive bounded final receipts; no process, activity or policy change."""
+import json
+from pathlib import Path
+
+from container_contracts import HERE, TREE_LIMIT, check_file, frozen, read, require, sha, tree_bound
+
+PARENT = '07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee'
+MANIFEST = HERE / 'finalization-v2/prepared-manifest.json'
+COMPLETION_BYTES = 4 * 1024**2
+COMPLETION_SLOTS = 2
+RESERVED_BYTES = COMPLETION_BYTES * COMPLETION_SLOTS
+
+
+def verify_preparation():
+    frozen(PARENT)
+    manifest = read(MANIFEST)
+    require(manifest['parent_freeze_sha256'] == PARENT and
+            manifest['max_completion_bytes'] == COMPLETION_BYTES and
+            manifest['reserved_completion_slots'] == COMPLETION_SLOTS,
+            'exact additive finalization policy')
+    for item in manifest['files']:
+        check_file(HERE / item['file'], item)
+    return sha(MANIFEST)
+
+
+def reserve(snapshot):
+    require(snapshot['limit'] == TREE_LIMIT and
+            type(snapshot['bytes']) is int and 0 <= snapshot['bytes'] <= TREE_LIMIT - RESERVED_BYTES and
+            type(snapshot['files']) is int and 0 <= snapshot['files'] <= 256 - COMPLETION_SLOTS,
+            'reserve two bounded completion receipts and file slots before final write')
+
+
+def encode(value):
+    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
+    require(len(raw) <= COMPLETION_BYTES, 'bounded complete receipt before any final write')
+    return raw
+
+
+def write_completion(directory, name, value):
+    require(name in ('report.json', 'validation.json'), 'only reviewed final receipt names')
+    before_sha = verify_preparation()
+    raw = encode(value)
+    before = tree_bound(directory)
+    reserve(before)
+    path = Path(directory) / name
+    with path.open('xb') as stream:
+        stream.write(raw)
+    after = tree_bound(directory)
+    require(after['bytes'] == before['bytes'] + len(raw) and after['files'] == before['files'] + 1,
+            'complete owned tree after final write; no concurrent uncounted artifact')
+    require(verify_preparation() == before_sha, 'unchanged finalization sources across final write')
+    return after
--- /dev/null
+++ test_completion_v2.py
@@ -0,0 +1,44 @@
+"""Prepared source-free boundary fixtures; ROOT alone executes."""
+from pathlib import Path
+import tempfile
+import unittest
+from unittest.mock import patch
+
+from completion_v2 import COMPLETION_BYTES, RESERVED_BYTES, TREE_LIMIT, encode, reserve, write_completion
+
+
+class Completion(unittest.TestCase):
+    def test_exact_reserved_byte_and_slot_boundary(self):
+        reserve(dict(limit=TREE_LIMIT, bytes=TREE_LIMIT-RESERVED_BYTES, files=254))
+        for extra in (dict(bytes=TREE_LIMIT-RESERVED_BYTES+1), dict(files=255), dict(files=True)):
+            value=dict(limit=TREE_LIMIT, bytes=TREE_LIMIT-RESERVED_BYTES, files=254)
+            value.update(extra)
+            with self.assertRaises(ValueError): reserve(value)
+
+    def test_metadata_limit_refuses_before_file_creation(self):
+        with tempfile.TemporaryDirectory() as root:
+            with patch('completion_v2.verify_preparation', return_value='a'*64):
+                with self.assertRaises(ValueError):
+                    write_completion(root, 'report.json', {'too_large':'x'*COMPLETION_BYTES})
+            self.assertEqual(list(Path(root).iterdir()), [])
+
+    def test_full_postwrite_tree_and_exclusive_names(self):
+        with tempfile.TemporaryDirectory() as root:
+            with patch('completion_v2.verify_preparation', return_value='a'*64):
+                result=write_completion(root,'report.json',{'status':'fixture-only'})
+                self.assertEqual(result['files'],1)
+                self.assertEqual(result['bytes'],len(encode({'status':'fixture-only'})))
+                with self.assertRaises(FileExistsError):
+                    write_completion(root,'report.json',{'status':'fixture-only'})
+
+    def test_postwrite_unaccounted_file_is_fatal(self):
+        with tempfile.TemporaryDirectory() as root:
+            snapshots=[dict(limit=TREE_LIMIT,bytes=0,files=0), dict(limit=TREE_LIMIT,bytes=999,files=2)]
+            with patch('completion_v2.verify_preparation',return_value='a'*64), \
+                    patch('completion_v2.tree_bound',side_effect=snapshots):
+                with self.assertRaises(ValueError):
+                    write_completion(root,'validation.json',{'status':'fixture-only'})
+
+
+if __name__ == '__main__':
+    unittest.main()
--- run_container.py
+++ run_container_v2.py
@@ -19,6 +19,7 @@
     read,require,sha,tree_bound,write)
 from launch_derivative import derive
 from namespace_control import Controller
+from completion_v2 import verify_preparation, write_completion
 
 
 def idle(bench,url):
@@ -71,6 +72,7 @@
     parser.add_argument('--output',type=Path,required=True)
     args=parser.parse_args()
     require(args.run,'ROOT reviewed explicit actual execution only')
+    verify_preparation()
     parent=frozen(args.own_freeze); host_parents(parent)
     require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),'host controller unprofiled')
     benchmark=Path(parent['parents']['benchmark'])
@@ -124,6 +126,7 @@
         performance_timing_usable=False,matched_hardware_counts_qualified=False,
         source_correct_goodput_qualified=False,numerical_profiles_identical=False,
         native_or_stock_math_changes=False,cold_process_has_unmeasured_collector_request=False)
+    report['artifact_finalization_followup_sha256']=verify_preparation()
     allowed=os.sched_getaffinity(0)
     require(set(protocol['cpu_affinity'])<=allowed,'same actual four available client/provider CPUs')
     urllib.request.install_opener(opener())
@@ -217,7 +220,7 @@
         except BaseException as error: failures.append('post-run complete payload closure: '+repr(error))
         report['finalization_errors']=failures
         if failures: report['status']='UNQUALIFIED'
-        write(output/'report.json',report)
+        write_completion(output,'report.json',report)
         if failures:
             raise RuntimeError('diagnostic finalization failed; primary='+str(report.get('failure'))+'; '+repr(failures))
 
--- validate_container.py
+++ validate_container_v2.py
@@ -12,6 +12,7 @@
 from container_activity import summarize
 from container_contracts import HERE,FILE_LIMIT,TREE_LIMIT,build_binding,check_file,fingerprint,frozen,host_parents,read,require,sha,tree_bound,write
 from launch_derivative import derive
+from completion_v2 import verify_preparation, write_completion
 
 
 def successful_owner(receipt,parent):
@@ -117,10 +118,13 @@
     parser.add_argument('--own-freeze',required=True)
     parser.add_argument('--output',type=Path,required=True)
     args=parser.parse_args()
+    verify_preparation()
     parent=frozen(args.own_freeze); host_parents(parent)
     output=args.output.resolve(strict=True)
     require(output.parent==HERE,'owned diagnostic output')
     report=read(output/'report.json')
+    require(report.get('artifact_finalization_followup_sha256')==verify_preparation(),
+            'actual report used reviewed final receipt reservation')
     require(report['status']=='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
             and report['own_freeze_sha256']==args.own_freeze
             and report['benchmark_v8_sha256']==parent['benchmark_v8_manifest']['sha256']
@@ -203,13 +207,14 @@
     require(native_summary is not None and stock_summary,'both actual canonical workload activity scopes present')
     result=dict(status='PASS_CONTAINER_RECORDED_WORKLOAD_ACTIVITY_DIAGNOSTIC_ONLY',
         own_freeze_sha256=args.own_freeze,report_sha256=sha(output/'report.json'),
-        native=native_summary,stock_recorded=stock_summary,artifact_bounds=tree_bound(output),
+        native=native_summary,stock_recorded=stock_summary,artifact_bounds_before_validation_receipt=tree_bound(output),
         same_canonical_workload_policy_and_caps_checked=True,plain_observed_wire_outputs_exact=True,
         source_profiles_numerically_equal=False,source_correct_goodput_qualified=False,
         matched_complete_hardware_kernel_counts_qualified=False,stock_collector_losslessness_verified=False,
         timing_usable=False,device_profiler_allocation_admission_qualified=False)
     frozen(args.own_freeze); host_parents(parent); verify_preparation(report['benchmark_v8_sha256'])
-    write(output/'validation.json',result)
+    result['artifact_finalization_followup_sha256']=verify_preparation()
+    write_completion(output,'validation.json',result)
 
 
 if __name__=='__main__': main()
--- /dev/null
+++ finalization-v2/README.md
@@ -0,0 +1,20 @@
+# Additive final receipt artifact bound
+
+Author: spine_review. Original container07a86bf6 and capacity4337ad63 immutable.
+
+- New run_container_v2.py and validate_container_v2.py retain original CLI,
+  activity/source/window/model/capacity/cgroup/workload/lifecycle/math behavior.
+- Before EACH final receipt write: serialize finite JSON <=4MiB; full existing
+  tree <=1GiB-8MiB and <=254files, reserving two4MiB receipt/file slots. Then
+  exclusive write and full final tree assertion, including exact observed
+  byte/file increments. No post-limit receipt can become accepted evidence.
+- Aggregate1GiB/per-owner768MiB/per-file576MiB and raw512MiB/4M remain unchanged.
+  Reservation is acceptance bookkeeping, not filesystem quota or device/RAM cap.
+- New reports bind this additive manifest; validator refuses old unreserved
+  reports. Native bootstrap/collector still attest unchanged07a/4337, correctly.
+- ROOT first runs test_completion_v2.py (four prepared synthetic methods), then
+  uses v2 runner/validator with identical original arguments. New capacity C++
+  still requires actual ROOT build; no author execution occurred.
+
+Author validation: four Python ASTs parse only, small source/frozen hashes.
+No fixtures/build/runtime/provider/GPU/model or payload hashing.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/frozen/NEW_FILES.review.patch

Original UTF-8 bytes: 73289; SHA256 `f17bb18a8bb5ab5e5a57499aa1089d889fa1d1df6b21d9ef824f6d161ae3e05b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ container-v1/README.md
@@ -0,0 +1,148 @@
+# Native container activity diagnostic — source only
+
+Author: batch_design_review. Parent plan: serving-and-speculation.
+
+Own scope: this ignored child only. Immutable parents: CUPTI V1 f9f5c522, V2
+932e471e and genuine matched-benchmark V8 b0c50e8a. No MAIN/parent edits, engine,
+math, queue, source/model, timing workload or production profile changes.
+
+## Files / ROOT execution
+
+- `capacity-v1/`: separate bounded C++ revision + streamed validator + six
+  synthetic fixtures. Capacity freeze4337ad63; new actual ROOT build required.
+- `container_contracts.py`, `parent-bindings.json`: exact additive closure.
+- `launch_derivative.py`: exact V8 argv + diagnostic-only mounts/env/bootstrap
+  transport and finite file-size cap. Original model flags untouched.
+- `native_exec.py`: run unchanged V8 bootstrap; its one final native exec uses
+  `os.execve` with explicit collector env. Outer Python/Podman LD_PRELOAD remains
+  empty. Stock always runs original bootstrap unchanged.
+- `namespace_control.py`: original owner nonce/name/CID -> public inspect init
+  HPID -> pidfd/start time/namespace readbacks -> host SO_PEERCRED. No signal,
+  reap, setns, new privilege, foreign PID adoption, context or stream.
+- `run_container.py`: four fresh original V8/53ee-owned processes: native plain,
+  native observed, stock plain, stock public Torch-profiler observed.
+- `container_activity.py`, `capacity_v1_loader.py`, `validate_container.py`:
+  independent full raw activity/wire/namespace/owner/canonical-argv validation.
+- `test_contracts.py`: seven prepared synthetic transport/control negatives.
+  No synthetic fixture establishes physical collection or model mathematics.
+
+ROOT first reviews/freezes, then executes only after timing quiet + reservation:
+
+```text
+python3 capacity-v1/test_activity.py
+python3 test_contracts.py
+```
+
+Build `capacity-v1/collector_capacity.cpp` with immutable parent ROOT_BUILD.md
+procedure and original approved headers/compiler. Record actual dependency/ELF
+closure, argv/source/build/artifact hashes. Fresh artifact directory contains
+`collector.so` and actual `build.json` with:
+
+```json
+{
+  "status": "ROOT_COMPILED_COLLECTOR_ONLY",
+  "freeze_sha256": "4337ad630552e0b46644cfb39d8198c6aad94cb3b48d85c8402a3a315a247bdd",
+  "collector_source_sha256": "ACTUAL_CAPACITY_CPP_SHA256",
+  "collector_sha256": "ACTUAL_COMPILED_ARTIFACT_SHA256",
+  "exit_code": 0
+}
+```
+
+Do not prewrite success. The accepted e918ff2f host V1 artifact is intentionally
+refused for this revision. Build can preserve original callback/API/math bytes;
+only fixed limits128→512MiB, records1M→4M, schema and explicit capacity metadata
+change. No runtime environment override can raise limits.
+
+Use an existing genuine V8 launch directory (exact protocol/workload/launches/
+input-attestation), not a shortened host-control prompt. Set ROOT explicitly:
+
+```text
+ZG_BENCH_HOST_ROOT=/home/pvl/spaces/abird/src/zerogravity
+python3 run_container.py --run --own-freeze OWN_FREEZE_SHA256
+  --launch-directory ABSOLUTE_ORIGINAL_V8_LAUNCH_DIRECTORY
+  --collector-artifact-directory ABSOLUTE_FRESH_ROOT_CAPACITY_BUILD_DIRECTORY
+  --concurrency ORIGINAL_PROTOCOL_CONCURRENCY
+  --output ABSOLUTE_THIS_DIRECTORY/root-container-1
+python3 validate_container.py --own-freeze OWN_FREEZE_SHA256
+  --output ABSOLUTE_THIS_DIRECTORY/root-container-1
+```
+
+Invocation is direct argv under ROOT's existing original process supervisor,
+not shell interpolation; export the explicit host-root variable. Author has
+not imported, run fixtures, compiled or executed any provider/container/workload.
+Only Python AST parsing and small source/metadata hashing/freezing occurred.
+
+## Identity, window, policy
+
+- Actual container init HPID comes from nonce/name/CID-bound public inspect;
+  Podman Popen PID remains its original cleanup owner, not the collector PID.
+- Exact host peer HPID + ordinary host UID must match SO_PEERCRED. pidfd remains
+  live; public container binding, proc start time and namespaces match before/
+  after each START/STOP. Output retains actual UID/GID maps and namespace IDs.
+- Reported `getpid()` is namespacePID1. Raw CUPTI driver IDs are never rewritten:
+  exactly one ID, equal to authenticated HPID OR namespacePID1; reject mixed or
+  foreign IDs and preserve which representation actually occurred. Metadata and
+  driver identities use separate validator arguments.
+- Original `--network=host` supplies abstract UNIX reachability; actual net-ns
+  equality and user-namespace peer translation must pass. Any hidepid/proc,
+  pidfd, mapped-file, permission or provider denial refuses; no privilege repair.
+- Actual native/collector/CUPTI mapped inode/dev/full SHA are checked through
+  `/proc/HPID/root`; CUPTI C++ still resolves only official versioned symbols
+  from exact original provider. Remaining original native/provider dependencies
+  retain existing V8/ROOT build closure; no substitute DSO.
+- Both native modes inherit exact V8 bootstrap/source/affinity/cgroup checks.
+  Stock uses original profiler-config and public `/start_profile`/`stop_profile`;
+  no stock math/module/queue patches or CUPTI preload.
+- `cold_process`: no collector warmup request. `warmed_process`: ONE unchanged
+  V8 unmeasured request with first original prompt and original max_tokens.
+  OS/storage/device caches are not evicted or claimed cold. Official collector
+  initialization overhead itself is diagnostic, not a canonical timing sample.
+- Native START follows original terminal/idle health; all canonical requests
+  then complete; native terminal/idle health precedes STOP/forced flush.
+  All source native dispatch counters remain distinct from CUPTI records.
+- Same canonical request count/prompts/arrivals/concurrency/max_tokens, CPU4,
+  RAM12GiB/swap-total12GiB, device, model/provider limits, stock eager/default
+  route and generation policy. No one-token collector warmup or workload edit.
+- Four actual owner receipts/monitor/containers/wholegroups/readers/EOF must
+  retire under unchanged reviewed53ee WNOWAIT cleanup before next process.
+  Primary/stop/close/post-source failures remain fatal with original receipts.
+
+## Capacity and remaining physical gates
+
+512MiB raw /4,000,000 records are finite, not a guarantee every canonical wave
+fits. Any buffer refusal/drop/overflow/partial record/API/timestamp mismatch
+refuses all acceptance. Entire native file is streamed twice, exact duplicate/
+correlation/totals checked. Validation sets are finite but may use several GiB
+of separate host RAM: ROOT reserves headroom before validation; not engine
+provider admission. Kernel count, successful direct launch APIs, graph launch
+APIs and kernels executed from graphs stay separate.
+
+Diagnostic-only per-file soft=hard RLIMIT_FSIZE is576MiB. Each owned sample tree
+must be<=768MiB (capacity README's native-sample bound); four-sample aggregate is
+<=1GiB to hold512MiB native raw + original stock trace<=256MiB + metadata/logs.
+Tree limits are post-write acceptance checks, not a kernel filesystem quota.
+File limit is enforced by original Podman public ulimit; native wrapper records
+actual readback. Original CPU/RAM/model limits are unchanged. Stock trace retains
+original256MiB compressed/expanded bounds; exceeding any bound is refusal.
+
+Before claiming a method: ROOT must qualify actual new build/13fixtures,
+namespace/user-peer/mapped closure, exact plain/observed wire IDs/text/finish/
+usage, real workload window and no activity loss, actual caps, final process/
+container/group/pipe/monitor closure. Negative tests must include foreign peer,
+wrong nonce/namespace/stale PID, provider permissions/cold refusal, failed STOP,
+interruption/receipt I/O, actual overflow/drop behavior and no accepted partial
+trace. Synthetic corrupted metadata fixtures validate refusal only; they do not
+establish actual callback overflow/CUPTI drop behavior.
+
+Root source estimate236270 dispatches for C1/48prompt/output8 suggests old128MiB
+cap too small. This is source metadata, NOT a canonical hardware measurement or
+speed result. Never run old undersized artifact and accept a prefix of records.
+
+Actual small host V2 PASS64328 kernels remains scoped to that old method. No
+container, larger-capacity, matched counts or new performance claim follows.
+Stock public Torch profiler currently does NOT prove losslessness/all-worker/
+complete initialization coverage; validator deliberately returns
+`matched_complete_hardware_kernel_counts_qualified=false` even if both recorded
+workload scopes pass. These diagnostic data do not establish numerical profile
+parity, source-correct goodput or a timing speedup. Opaque CUPTI internal device
+allocation admission remains unqualified and observed resource cost is separate.
--- /dev/null
+++ container-v1/STATIC_AUDIT.json
@@ -0,0 +1,62 @@
+{
+  "CXX_changes_normalize_byte_exact_to_immutable_V1": true,
+  "actual_new_collector_build_or_runtime_qualified": false,
+  "container_summarizer_changes_only_explicit_namespace_and_driver_arguments": true,
+  "critical_V8_source_pins": [
+    {
+      "bytes": 11893,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/prepare_launch_v8.py",
+      "sha256": "abaa0dfe32bb4aff3c1df0062fc54e7ebb0e73be1676ae290f5e5adb2725dc2d"
+    },
+    {
+      "bytes": 4706,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/server_boot_v8.py",
+      "sha256": "2288dfbe131b88b6c4fa6860d9eb3b070f4223f262fb2352a1f7f7cd0ce76baf"
+    },
+    {
+      "bytes": 9968,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/run_compare_v8.py",
+      "sha256": "fa08b592357217d58dd438014f6986976cb591e36fad8fd601c3fb83bf388adb"
+    },
+    {
+      "bytes": 8621,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/run_diagnostic_v8.py",
+      "sha256": "7c002f4803aca6daee0b772756395388cd5704c49fc5e3602c3309207bd27c7f"
+    },
+    {
+      "bytes": 9685,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/owned_plan_v8.py",
+      "sha256": "346f8f98753018e6ddacaff291dec48def8ce024be6e9c4670c154fc028bedcb"
+    },
+    {
+      "bytes": 11840,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/owned_container_v4.py",
+      "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+    },
+    {
+      "bytes": 10387,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/protocol_v8.py",
+      "sha256": "61de8d581e9353c685bec518eab5f974df10de8baa57353e867606887b9ff98e"
+    },
+    {
+      "bytes": 5708,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/observations_v8.py",
+      "sha256": "23a783abbead3da819f9d0b046643010b669a6f14f2abb1e0752bd4ae35e4a9d"
+    },
+    {
+      "bytes": 13235,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/cgroup_readback_v3.py",
+      "sha256": "e29e76e06eb8477ffc279f1e3bf4277007c3f539e20053eddd76a3fda1df9a7b"
+    },
+    {
+      "bytes": 3387,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/audit_cuda_trace_v8.py",
+      "sha256": "15449b7fec9045d88beecf4cfb2025af39aa36083bc156015c75a5f1354ce747"
+    }
+  ],
+  "fixtures_prepared_not_executed": 13,
+  "original_CUPTI_V1_V2_and_capacity_frozen_bytes_unchanged": true,
+  "parent_manifests_exact": true,
+  "python_AST_files": 11,
+  "scope": "SMALL_SOURCE_ONLY_NO_IMPORTS_TESTS_BUILDS_RUNTIME"
+}
--- /dev/null
+++ container-v1/TODO.md
@@ -0,0 +1,25 @@
+# CUPTI container diagnostic / capacity revision
+
+Parent: serving-and-speculation. Owner: batch_design_review.
+Scope: ONLY this ignored child; no production/parent/plan/README changes.
+
+- [x] Read immutable CUPTI V1/V2 + exact V8 canonical launch/bootstrap/owner/
+  diagnostic/trace/cgroup source; keep original53ee lifecycle unchanged.
+- [x] Prepare separate512MiB/4M C++ cap revision and six source-only negatives;
+  exact callback/API/math unchanged, new compiled artifact mandatory.
+- [x] Prepare final-native-exec-only seam; host PID/namespace PID/raw driver PID
+  distinct; original owner/caps/source/full canonical workload/cold-warm policy.
+- [x] Prepare four-process plain/observed native+stock transparency and independent
+  complete activity/argv/owner validator; seven container transport fixtures.
+- [x] AST parse only; no imports, fixtures, compilation or runtime by author.
+- [ ] Independent source review / ROOT actual fixture+strict C++build+dependency
+  closure / actual process namespace+permissions/metadata/STOP/loss qualification.
+- [ ] ROOT actual canonical workload/negative failure+overflow/retirement tests.
+- [ ] Stock collector loss/completeness gate; no matched complete HW claim yet.
+- [ ] Any actual measured gain/numerical equivalence/fullmodel claim is external;
+  none established by this source proposal.
+
+Failures: no executed test/build/runtime. Source-only inspection corrected a
+post-START receipt-I/O cleanup gap before freeze; STOP flag now arms immediately
+on acknowledged START. Post-run source/affinity errors downgrade report before
+independent validation rather than leaving an executable-success status.
--- /dev/null
+++ container-v1/capacity_v1_loader.py
@@ -0,0 +1,10 @@
+"""Load own frozen capacity helper without changing its bytes or sys.path."""
+import importlib.util
+from pathlib import Path
+import sys
+
+path=Path(__file__).resolve().parent/'capacity-v1/activity.py'
+spec=importlib.util.spec_from_file_location('zg_cupti_capacity_v1_activity',path)
+activity=importlib.util.module_from_spec(spec)
+sys.modules[spec.name]=activity
+spec.loader.exec_module(activity)
--- /dev/null
+++ container-v1/container_activity.py
@@ -0,0 +1,72 @@
+"""Capacity validator with separate authenticated namespace/driver process IDs.
+
+Raw rows and metadata are never rewritten. Apart from these explicit identity
+arguments, same source algorithm as the frozen capacity validator.
+"""
+import collections
+from capacity_v1_loader import activity
+MAX_RECORDS=activity.MAX_RECORDS
+KERNEL_CBIDS=activity.KERNEL_CBIDS
+GRAPH_CBIDS=activity.GRAPH_CBIDS
+require=activity.require
+validate_capacity=activity.validate_capacity
+
+def summarize(rows, metadata, namespace_pid, driver_pid):
+    validate_capacity(metadata)
+    require(metadata['phase'] == 'STOPPED' and metadata['api_version'] == 28
+            and metadata['pid'] == namespace_pid and 0 < metadata['begin'] < metadata['end'],
+            'real complete native CUPTI window')
+    require(metadata['failures'] == metadata['dropped'] == metadata['first_cupti_error'] == 0,
+            'no parse/buffer/timestamp/activity loss or CUPTI error')
+    require(metadata['timing_sample'] is False
+            and metadata['actual_device_allocation_bound_qualified'] is False,
+            'diagnostic-only explicit allocation/timing scope')
+    require(metadata['host_callback_pool_bytes'] == 16 * 1024**2
+            and metadata['device_buffer_requested_bytes'] == 1024**2
+            and metadata['device_pool_limit'] == 8 and metadata['device_preallocate'] == 2,
+            'actual requested/read-back buffer controls')
+    kernel_count = api_count = graph_kernels = direct_calls = graph_calls = 0
+    kernel_correlations, launch_correlations, devices, unique = set(), set(), set(), set()
+    names = collections.Counter()
+    for row in rows:
+        require(metadata['begin'] <= row['start'] <= row['end'] <= metadata['end']
+                and row['start'] > 0, 'complete activity within declared workload window')
+        require(type(row['correlation']) is int and row['correlation'] > 0,
+                'actual nonempty driver/kernel correlation')
+        if row['kind'] == 'kernel':
+            require(row['name'] and len(row['name'].encode()) < 512
+                    and all(type(n) is int and n > 0 for n in row['grid'] + row['block'])
+                    and len(row['grid']) == len(row['block']) == 3, 'actual complete kernel geometry/name')
+            key = ('kernel', row['context'], row['stream'], row['grid_id'], row['start'], row['end'])
+            kernel_count += 1
+            graph_kernels += row['graph'] != 0
+            kernel_correlations.add(row['correlation'])
+            devices.add(row['device'])
+            names[row['name']] += 1
+            require(len(names) <= 2048, 'bounded actual kernel-name inventory')
+        else:
+            require(row['kind'] == 'driver' and row['process'] == driver_pid, 'actual original-process driver API')
+            key = ('driver', row['process'], row['thread'], row['correlation'], row['cbid'], row['start'], row['end'])
+            api_count += 1
+            if row['cbid'] in KERNEL_CBIDS | GRAPH_CBIDS:
+                require(row['result'] == 0, 'actual successful public launch API')
+                launch_correlations.add(row['correlation'])
+                direct_calls += row['cbid'] in KERNEL_CBIDS
+                graph_calls += row['cbid'] in GRAPH_CBIDS
+        require(key not in unique, 'no duplicated CUPTI activity record')
+        unique.add(key)
+        require(len(unique) <= MAX_RECORDS, 'bounded retained activity')
+    require(kernel_count == metadata['kernels'] > 0 and api_count == metadata['driver_apis'] > 0
+            and len(unique) == metadata['records'], 'all retained activity records/totals, not startup only')
+    require(launch_correlations and kernel_correlations <= launch_correlations,
+            'all actual native GPU kernels have retained successful public launch correlations')
+    require(len(devices) == 1, 'one declared physical GPU')
+    return {
+        'actual_gpu_kernel_records': kernel_count,
+        'driver_kernel_launch_api_calls': direct_calls,
+        'driver_graph_launch_api_calls': graph_calls,
+        'kernel_records_from_graphs': graph_kernels,
+        'kernel_names': dict(names),
+        'scope': 'official concurrent-kernel activity records in this native process/window; API calls separate',
+    }
+
--- /dev/null
+++ container-v1/container_contracts.py
@@ -0,0 +1,95 @@
+"""ROOT-only diagnostic closure. No process/GPU work on import."""
+import hashlib
+import json
+from pathlib import Path
+
+HERE=Path(__file__).resolve().parent
+FILE_LIMIT=576*1024**2
+TREE_LIMIT=1024*1024**2
+
+
+def require(ok,why):
+    if not ok: raise ValueError(why)
+
+
+def sha(path):
+    with Path(path).open('rb') as stream:
+        return hashlib.file_digest(stream,'sha256').hexdigest()
+
+
+def read(path):
+    with Path(path).open('rb') as stream: raw=stream.read(4*1024**2+1)
+    require(len(raw)<=4*1024**2,'bounded diagnostic metadata')
+    return json.loads(raw)
+
+
+def write(path,value):
+    with Path(path).open('x') as stream:
+        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
+        stream.write('\n')
+
+
+def check_file(path,item):
+    path=Path(path)
+    require(path.is_file() and not path.is_symlink() and path.stat().st_size==item['bytes']
+            and sha(path)==item['sha256'],'exact prepared file: '+str(path))
+
+
+def frozen(expected):
+    require(sha(HERE/'freeze-manifest.json')==expected,'explicit own prepared source identity')
+    manifest=read(HERE/'freeze-manifest.json')
+    for item in manifest['files']: check_file(HERE/item['file'],item)
+    parent=read(HERE/'parent-bindings.json')
+    check_file(HERE/'capacity-v1/freeze-manifest.json',parent['capacity_manifest'])
+    capacity=read(HERE/'capacity-v1/freeze-manifest.json')
+    for item in capacity['files']: check_file(HERE/'capacity-v1'/item['file'],item)
+    return parent
+
+
+def host_parents(parent):
+    for key in ('cupti_v1_manifest','cupti_v2_manifest','benchmark_v8_manifest'):
+        item=parent[key]; check_file(item['path'],item)
+    for key in ('cupti_v1_manifest','cupti_v2_manifest'):
+        item=parent[key]; root=Path(item['path']).parent
+        manifest=read(item['path'])
+        for row in manifest['files']: check_file(root/row['file'],row)
+        for row in manifest.get('source_pins',[]): check_file(row['path'],row)
+    helper=Path(parent['parents']['benchmark'])/'owned_container_v4.py'
+    require(sha(helper)==parent['owned_helper_sha256'],'same reviewed reserved-leader helper')
+
+
+def build_binding(path,artifact,parent):
+    value=read(path)
+    require(value['status']=='ROOT_COMPILED_COLLECTOR_ONLY' and value['exit_code']==0
+            and value['freeze_sha256']==parent['capacity_manifest']['sha256']
+            and value['collector_source_sha256']==sha(HERE/'capacity-v1/collector_capacity.cpp')
+            and value['collector_sha256']==sha(artifact),
+            'actual ROOT new capacity source/build/artifact binding; old V1 binary cannot substitute')
+    return value
+
+
+def fingerprint(result):
+    require(result['completed_requests']==result['offered_requests']>0
+            and result['unsuccessful_requests']==0 and result['errors']=={},'complete real HTTP denominator')
+    out=[]
+    require(len(result['requests'])==result['offered_requests'],'all request records')
+    for item in result['requests']:
+        ids=item['wire_token_observation']['token_ids']
+        require(item['completed'] is True and item['usage_reported_completion_tokens']>0
+                and len(ids)==item['usage_reported_completion_tokens']
+                and all(type(v) is int and v>=0 for v in ids),'complete wire IDs and usage')
+        out.append((item['input_index'],item['prompt_sha256'],item['output_sha256'],
+                    item['finish_reason'],item['usage_reported_completion_tokens'],tuple(ids)))
+    return out
+
+
+def tree_bound(path,limit=TREE_LIMIT):
+    require(limit in (768*1024**2,TREE_LIMIT),'fixed reviewed per-owner/aggregate artifact limits')
+    total=count=0
+    for entry in Path(path).rglob('*'):
+        require(not entry.is_symlink(),'owned diagnostic tree without symlinks')
+        if entry.is_dir(): continue
+        require(entry.is_file(),'regular diagnostic artifacts only')
+        size=entry.stat().st_size; count+=1; total+=size
+        require(size<=FILE_LIMIT and total<=limit and count<=256,'bounded full diagnostic artifact tree')
+    return dict(bytes=total,files=count,limit=limit,per_file_limit=FILE_LIMIT)
--- /dev/null
+++ container-v1/launch_derivative.py
@@ -0,0 +1,27 @@
+"""Pure exact-V8 command derivative; only diagnostic transport/artifact controls."""
+from pathlib import Path
+
+from container_contracts import FILE_LIMIT, HERE, require
+
+
+def derive(command,provider,mode,own_freeze,artifact):
+    require(provider in ('zerogravity','vllm') and mode in ('plain','observed'), 'finite declared sample')
+    command=list(command)
+    old='fsize=268435456:268435456'
+    require(command.count(old)==1 and command.count('/probe/server_boot_v8.py')==1,
+            'exact known V8 diagnostic file control and bootstrap')
+    # Only diagnostic artifact budget changes; model/cgroup/device/workload unchanged.
+    command[command.index(old)]=f'fsize={FILE_LIMIT}:{FILE_LIMIT}'
+    require('--network=host' in command and 'LD_PRELOAD=' in command and 'LD_AUDIT=' in command,
+            'original host network and empty outer interposition')
+    if provider=='vllm':
+        return command
+    require(command[command.index('--provider')+1]=='zerogravity','native-only wrapper')
+    position=command.index('--entrypoint')
+    command[position:position]=['-v',str(HERE)+':/collector-probe:ro',
+        '-v',str(Path(artifact))+':/collector-artifact:ro',
+        '-e','ZG_CONTAINER_DIAGNOSTIC_FREEZE='+own_freeze,
+        '-e','ZG_CONTAINER_DIAGNOSTIC_MODE='+mode,
+        '-e','ZG_CONTAINER_DIAGNOSTIC_NONCE={owned_nonce}']
+    command[command.index('/probe/server_boot_v8.py')]='/collector-probe/native_exec.py'
+    return command
--- /dev/null
+++ container-v1/namespace_control.py
@@ -0,0 +1,143 @@
+"""Host controller: authenticate nonce/CID -> actual init HPID -> SO_PEERCRED.
+
+Does not signal, reap, setns, or adopt foreign processes. Existing V8/53ee owner
+alone retires its original Podman group/container. Namespace PID and driver PID
+remain separate observed identities; neither is rewritten into a host PID.
+"""
+import json
+import os
+from pathlib import Path
+import re
+import select
+import socket
+import struct
+
+from container_contracts import read, require, sha
+
+
+def text(path,limit):
+    with Path(path).open() as stream: raw=stream.read(limit+1)
+    require(len(raw)<=limit,'bounded original process metadata')
+    return raw
+
+
+def stat_start(raw):
+    # comm may include spaces and parentheses; the last ')' ends field2.
+    at=raw.rfind(')')
+    require(at>1 and len(raw[at+2:].split())>=20,'complete Linux proc stat')
+    return int(raw[at+2:].split()[19])
+
+
+def inspect(owner,owned):
+    owner.live()
+    cid=owned.owned_container(owner.name,owner.nonce,owner.cidfile)
+    require(cid is not None,'live exact nonce/name/CID binding')
+    result=owned.command(owned.PODMAN+['container','inspect',cid],timeout=10)
+    require(result.returncode==0 and len(result.stdout)<=1024**2,'bounded public container inspect')
+    value=json.loads(result.stdout)
+    require(type(value) is list and len(value)==1,'one inspected original container')
+    row=value[0]
+    require(row['Id']==cid and row['Name'].lstrip('/')==owner.name
+            and row['Config']['Labels'][owned.LABEL]==owner.nonce
+            and row['State']['Running'] is True and type(row['State']['Pid']) is int
+            and row['State']['Pid']>0,'same original live init host PID, never Podman Popen PID')
+    return dict(cid=cid,owned_nonce=owner.nonce,owned_name=owner.name,host_init_pid=row['State']['Pid'])
+
+
+def identities(pid):
+    root=Path('/proc')/str(pid)
+    raw=text(root/'status',65536)
+    fields={key:value.strip() for line in raw.splitlines() if ':' in line for key,value in [line.split(':',1)]}
+    nspid=[int(v) for v in fields['NSpid'].split()]
+    require(nspid[0]==pid and nspid[-1]==1,'actual direct native namespace PID1')
+    return dict(start_time=stat_start(text(root/'stat',16384)), NSpid=nspid,
+        namespace_inodes={name:os.readlink(root/'ns'/name) for name in ('pid','net','user','mnt')})
+
+
+def mapped_files(pid,exec_attestation,parent):
+    root=Path('/proc')/str(pid)
+    lines=text(root/'maps',2*1024**2).splitlines()
+    expected={'native':'/runtime/zerogravity','collector':'/collector-artifact/collector.so',
+              'cupti':parent['cupti_library']['container_path']}
+    output={}
+    for key,path in expected.items():
+        actual=root/'root'/path.lstrip('/')
+        info=actual.stat()
+        rows=[]
+        for line in lines:
+            parts=line.split(None,5)
+            if len(parts)==6 and parts[5]==path:
+                major,minor=(int(v,16) for v in parts[3].split(':'))
+                require(int(parts[4])==info.st_ino and os.makedev(major,minor)==info.st_dev,
+                        'actual mapped inode/dev matches exact original file')
+                rows.append(line)
+        require(rows and actual.is_file(),'all declared executable/collector/CUPTI mappings present')
+        digest=sha(actual)
+        expected_sha=exec_attestation['native_sha256' if key=='native' else key+'_sha256']
+        require(digest==expected_sha,'full actual mapped artifact hash: '+key)
+        output[key]=dict(container_path=path,bytes=info.st_size,sha256=digest,
+            inode=info.st_ino,device=info.st_dev,mappings=rows)
+    return output
+
+
+class Controller:
+    def __init__(self,owner,owned,parent):
+        self.owner,self.owned,self.parent=owner,owned,parent
+        self.fd=None
+        self.receipt=dict(schema='zg.cupti.container.controller.v1',events=[],pidfd_closed=False)
+        try:
+            self.binding=inspect(owner,owned)
+            self.pid=self.binding['host_init_pid']
+            self.fd=os.pidfd_open(self.pid,0)
+            self.before=identities(self.pid)
+            self.live()
+            actual=read(owner.directory/'native-exec-attestation.json')
+            require(actual['mode']=='observed' and actual['pid_namespace_pid']==1
+                    and actual['nonce']==owner.nonce and actual['collector_on_outer_python'] is False
+                    and actual['collector_requested_only_at_this_exec'] is True,
+                    'actual unchanged final-exec receipt bound to original owner nonce')
+            require(actual['namespace_inodes']==self.before['namespace_inodes']
+                    and self.before['namespace_inodes']['net']==os.readlink('/proc/self/ns/net'),
+                    'actual host-network namespace shared; no setns/privilege assumption')
+            self.nonce=actual['nonce']
+            self.receipt.update(binding=self.binding,host_before=self.before,exec_attestation=actual,
+                mapped_files=mapped_files(self.pid,actual,parent),
+                host_controller_uid=os.geteuid(),reported_namespace_pid=1,
+                driver_process_id_policy='preserve raw CUPTI IDs; require namespace PID1 independently')
+        except BaseException:
+            self.close()
+            raise
+
+    def live(self):
+        require(self.fd is not None and not select.select([self.fd],[],[],0)[0],
+                'original init pidfd remains live, no recycled PID accepted')
+        require(inspect(self.owner,self.owned)==self.binding and identities(self.pid)==self.before,
+                'original container init identity and namespace binding stable')
+
+    def command(self,verb):
+        require(verb in ('START','STOP') and len(self.receipt['events'])==(0 if verb=='START' else 1),
+                'one ordered START/STOP, no repeated activity windows')
+        self.live()
+        with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as connection:
+            connection.settimeout(30)
+            connection.connect('\0zg-cupti-'+self.nonce)
+            peer=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
+            require(peer[0]==self.pid and peer[1]==os.geteuid(),
+                    'SO_PEERCRED actual host init PID/UID, independently of reported namespace PID')
+            connection.sendall((verb+' '+self.nonce).encode('ascii'))
+            raw,_,flags,_=connection.recvmsg(8192)
+            require(not flags&socket.MSG_TRUNC and len(raw)<8192,'complete bounded control reply')
+            value=json.loads(raw)
+        require(value['pid']==1 and value['nonce']==self.nonce and value['api_version']==28
+                and value['phase']==('STARTED' if verb=='START' else 'STOPPED'),
+                'actual namespace PID1/API/nonce/ordered phase')
+        require(value['failures']==value['dropped']==value['first_cupti_error']==0,'no lost/partial CUPTI result')
+        self.live()
+        self.receipt['events'].append(dict(verb=verb,host_peer_pid=peer[0],host_peer_uid=peer[1],
+            reported_namespace_pid=value['pid'],reply=value))
+        return value
+
+    def close(self):
+        if self.fd is not None:
+            os.close(self.fd); self.fd=None
+        self.receipt['pidfd_closed']=True
--- /dev/null
+++ container-v1/native_exec.py
@@ -0,0 +1,81 @@
+"""Final-native-exec-only diagnostic seam around exact immutable V8 bootstrap.
+
+Outer Python and Podman never receive LD_PRELOAD. No engine argv/model math is
+changed. Stock always uses the unchanged original server_boot_v8.py directly.
+"""
+import os
+import resource
+from pathlib import Path
+import sys
+
+from container_contracts import FILE_LIMIT, HERE, build_binding, check_file, frozen, read, require, sha, write
+
+
+def bounded_text(path,limit=8192):
+    with Path(path).open() as stream: value=stream.read(limit+1)
+    require(len(value)<=limit,'bounded namespace identity')
+    return value
+
+
+def main():
+    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
+            'collector absent from outer Python/Podman process')
+    expected=os.environ['ZG_CONTAINER_DIAGNOSTIC_FREEZE']
+    parent=frozen(expected)
+    require(Path('/probe/prepared-manifest-v8.json').is_file(), 'original V8 mounted separately')
+    check_file('/probe/prepared-manifest-v8.json',parent['benchmark_v8_manifest'])
+    sys.path.insert(0,'/probe')
+    import server_boot_v8 as original
+    require(Path(original.__file__).resolve()==Path('/probe/server_boot_v8.py'), 'exact original bootstrap path')
+    require(os.getpid()==1,'direct original container-init process, no extra supervisor/engine')
+    require(resource.getrlimit(resource.RLIMIT_FSIZE)==(FILE_LIMIT,FILE_LIMIT),
+            'actual diagnostic-only finite file-size soft/hard cap')
+    mode=os.environ['ZG_CONTAINER_DIAGNOSTIC_MODE']
+    require(mode in ('plain','observed'),'declared diagnostic mode')
+    saved=os.execv
+    def final_exec(path,argv):
+        require(path=='/runtime/zerogravity' and argv[0]==path and argv[1]=='serve'
+                and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
+                'single exact final native exec; no outer preload or stock interception')
+        attestation=dict(schema='zg.cupti.container.exec.v1', mode=mode, own_freeze_sha256=expected,
+            benchmark_v8_sha256=parent['benchmark_v8_manifest']['sha256'], argv=argv,
+            native_sha256=sha(path), pid_namespace_pid=os.getpid(), uid=os.geteuid(), gid=os.getegid(),
+            uid_map=bounded_text('/proc/self/uid_map'), gid_map=bounded_text('/proc/self/gid_map'),
+            namespace_inodes={name:os.readlink('/proc/self/ns/'+name) for name in ('pid','net','user','mnt')},
+            CPU_affinity=sorted(os.sched_getaffinity(0)), file_size_limits=list(resource.getrlimit(resource.RLIMIT_FSIZE)),
+            collector_on_outer_python=False,
+            exec_transport_only=True, math_or_queue_changes=False)
+        require(attestation['native_sha256']==os.environ['ZG_NATIVE_BINARY_SHA256'], 'same V8 actual native artifact')
+        env=dict(os.environ)
+        if mode=='observed':
+            collector='/collector-artifact/collector.so'
+            build=build_binding('/collector-artifact/build.json',collector,parent)
+            library=parent['cupti_library']
+            check_file(library['container_path'],library)
+            activity=Path('/output/observed-activity')
+            activity.mkdir(mode=0o700)
+            require(activity.stat().st_uid==os.geteuid() and (activity.stat().st_mode&0o777)==0o700,
+                    'original CUPTI directory owner/mode in actual user namespace')
+            nonce=os.environ['ZG_CONTAINER_DIAGNOSTIC_NONCE']
+            require(len(nonce)==32 and all(v in '0123456789abcdef' for v in nonce),'exact controller nonce')
+            env.update(LD_PRELOAD=collector, ZG_CUPTI_OUTPUT=str(activity), ZG_CUPTI_NONCE=nonce,
+                ZG_CUPTI_LIBRARY=library['container_path'], ZG_CUPTI_TARGET_EXE=path)
+            attestation.update(collector_sha256=sha(collector), collector_build_sha256=sha('/collector-artifact/build.json'),
+                collector_capacity_freeze_sha256=parent['capacity_manifest']['sha256'],
+                cupti_sha256=library['sha256'], nonce=nonce,
+                collector_requested_only_at_this_exec=True, ROOT_build_status=build['status'])
+        write('/output/native-exec-attestation.json',attestation)
+        os.execve(path,argv,env)
+        raise AssertionError('successful exec cannot return')
+    # Scoped control transfer only; original bootstrap retains all source, affinity,
+    # cgroup and native-binary checks before its ONE final exec site.
+    os.execv=final_exec
+    try:
+        original.main()
+        raise AssertionError('original native bootstrap must exec')
+    finally:
+        os.execv=saved
+
+
+if __name__=='__main__':
+    main()
--- /dev/null
+++ container-v1/parent-bindings.json
@@ -0,0 +1,32 @@
+{
+  "benchmark_v8_manifest": {
+    "bytes": 27185,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/prepared-manifest-v8.json",
+    "sha256": "b0c50e8a9d8d647464d3837748db0ba958a7c8294ee18140474703068fca0b33"
+  },
+  "capacity_manifest": {
+    "bytes": 1195,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/freeze-manifest.json",
+    "sha256": "4337ad630552e0b46644cfb39d8198c6aad94cb3b48d85c8402a3a315a247bdd"
+  },
+  "cupti_library": {
+    "bytes": 7604800,
+    "container_path": "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/lib/libcupti.so.12",
+    "sha256": "2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a"
+  },
+  "cupti_v1_manifest": {
+    "bytes": 12997,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/freeze-manifest.json",
+    "sha256": "f9f5c52234ff95386daab29ac1f4435834541e08815891dc7a8bfea29a1406c5"
+  },
+  "cupti_v2_manifest": {
+    "bytes": 2799,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/freeze-manifest-v2.json",
+    "sha256": "932e471e37ce21fc6588bb5bd67423945aae5ba8ac6b14aa5185df7b194653d3"
+  },
+  "owned_helper_sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d",
+  "parents": {
+    "benchmark": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020",
+    "cupti": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020"
+  }
+}
--- /dev/null
+++ container-v1/run_container.py
@@ -0,0 +1,230 @@
+"""ROOT ONLY four fresh diagnostic processes; no canonical timing samples.
+
+Native plain/CUPTI observed, stock plain/public Torch profiler observed. Each
+uses exact V8 requests/cold-warm policy/CPU/RAM/engine limits and original owner.
+"""
+import argparse
+from dataclasses import asdict
+import gzip
+import hashlib
+import json
+import os
+from pathlib import Path
+import signal
+import sys
+import time
+import urllib.request
+
+from container_contracts import (HERE,TREE_LIMIT,build_binding,fingerprint,frozen,host_parents,
+    read,require,sha,tree_bound,write)
+from launch_derivative import derive
+from namespace_control import Controller
+
+
+def idle(bench,url):
+    deadline=time.monotonic()+30
+    previous=None
+    while time.monotonic()<deadline:
+        raw=bench.snapshot(url+'/health',None)
+        require(raw['ok'],'actual original native health')
+        state=raw['value']['admission']
+        require(state['kind']=='resident_gpu' and state['failed_gpu_quanta']==0,'healthy original GPU engine')
+        if state['active_sessions']==state['queued_requests']==state['output_credit_parked_requests']==0:
+            if previous==raw['value']: return raw
+            previous=raw['value']
+        else: previous=None
+        time.sleep(.1)
+    raise TimeoutError('native physical work did not become terminal before window boundary')
+
+
+def stock_traces(owner,audit):
+    output=[]
+    directory=owner.directory/'traces'
+    require(directory.is_dir() and not directory.is_symlink(),'actual original public profiler output')
+    paths=sorted(directory.rglob('*'))
+    require(len(paths)<=32,'bounded actual profiler artifact count')
+    for path in paths:
+        require(not path.is_symlink(),'owned original profiler files')
+        if path.is_dir(): continue
+        require(path.is_file() and path.stat().st_size<=256*1024**2,'bounded public profiler artifact')
+        if not path.name.endswith(('.pt.trace.json','.pt.trace.json.gz')): continue
+        with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as stream:
+            raw=stream.read(256*1024**2+1)
+        require(len(raw)<=256*1024**2,'bounded entire decoded public trace')
+        value=json.loads(raw)
+        if any(event.get('cat')=='kernel' for event in value.get('traceEvents',[])):
+            result=audit(value)
+            result.update(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
+                decoded_sha256=hashlib.sha256(raw).hexdigest())
+            output.append(result)
+    require(len(output)==1,'one actual TP1 CUDA worker trace, no CPU-only substitution')
+    return output
+
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--run',action='store_true')
+    parser.add_argument('--own-freeze',required=True)
+    parser.add_argument('--launch-directory',type=Path,required=True)
+    parser.add_argument('--collector-artifact-directory',type=Path,required=True)
+    parser.add_argument('--concurrency',type=int,choices=(1,2,4,8),required=True)
+    parser.add_argument('--output',type=Path,required=True)
+    args=parser.parse_args()
+    require(args.run,'ROOT reviewed explicit actual execution only')
+    parent=frozen(args.own_freeze); host_parents(parent)
+    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),'host controller unprofiled')
+    benchmark=Path(parent['parents']['benchmark'])
+    sys.path.insert(0,str(benchmark))
+    from protocol_v8 import verify_preparation,verify_small_sources,validate_protocol
+    from prepare_launch_v8 import launch
+    from run_compare_v8 import lifecycle_spec,load_tool,opener
+    from run_diagnostic_v8 import post
+    from audit_cuda_trace_v8 import audit
+    from support_v3 import source_check
+    from owned_plan_v8 import install
+    import owned_container_v4 as owned
+    v8=parent['benchmark_v8_manifest']['sha256']
+    pins=verify_preparation(v8); verify_small_sources()
+    require(sha(owned.__file__)==parent['owned_helper_sha256'],'actual original53ee lifecycle helper import')
+    launch_dir=args.launch_directory.resolve(strict=True)
+    require(launch_dir.parent==benchmark and not args.launch_directory.is_symlink(),'original V8 owned launch directory')
+    protocol=validate_protocol(read(launch_dir/'protocol.json'))
+    require(args.concurrency in protocol['concurrencies'] and protocol['source_sha256']==pins['models'][protocol['model']]['gguf']['sha256']
+            and protocol['tokenizer_sha256']==pins['tokenizer_sha256'],'exact V8 workload/concurrency/source identity')
+    project=read(benchmark/'projector-pins.json')['models'][protocol['model']]
+    conversion=Path(project['conversion_directory'])
+    commands={label:launch(protocol,pins,v8,label,conversion) for label in ('zerogravity','vllm')}
+    require(read(launch_dir/'launches.json')==commands,'exact original canonical engine commands')
+    diagnostics=read(launch_dir/'diagnostic-launches.json')
+    stock_observed=launch(protocol,pins,v8,'vllm',conversion,profiler=True)
+    require(diagnostics['cuda_activity']==stock_observed,'exact existing public stock profiler command')
+    source=pins['models'][protocol['model']]
+    source_check(pins,protocol['model'],source['source_path'],source['gguf']['path'])
+    require(sha(conversion/'mmproj-bf16.gguf')==project['projector_sha256']
+            and sha(protocol['zerogravity_binary'])==protocol['zerogravity_binary_sha256'],
+            'actual unchanged projector/native executable before diagnostic')
+    artifact=args.collector_artifact_directory.resolve(strict=True)
+    require(not args.collector_artifact_directory.is_symlink(),'original ROOT artifact directory')
+    build=build_binding(artifact/'build.json',artifact/'collector.so',parent)
+    output=args.output.resolve()
+    require(output.parent==HERE and not output.exists(),'fresh exclusive owned diagnostic output')
+    output.mkdir(mode=0o700)
+    compare,bench=load_tool('serve-compare.py'),load_tool('serve-bench.py')
+    install(compare.LIFECYCLE)
+    urls={label:f'http://127.0.0.1:{port}' for label,port in zip(('zerogravity','vllm'),protocol['ports'])}
+    spec=lifecycle_spec(protocol,commands)
+    plan=compare.LIFECYCLE['LifecyclePlan'](spec,urls,str(output/'owned'))
+    alias='qwen35-'+protocol['model']+'-same-gguf'
+    report=dict(status='UNQUALIFIED',schema='zg.cupti.container.diagnostic.v1',own_freeze_sha256=args.own_freeze,
+        benchmark_v8_sha256=v8,capacity_freeze_sha256=parent['capacity_manifest']['sha256'],
+        protocol=protocol,protocol_path=str(launch_dir/'protocol.json'),
+        protocol_sha256=sha(launch_dir/'protocol.json'),concurrency=args.concurrency,
+        collector_build_sha256=sha(artifact/'build.json'),collector_sha256=build['collector_sha256'],
+        owned_cleanup_source_sha256=parent['owned_helper_sha256'],samples=[],
+        performance_timing_usable=False,matched_hardware_counts_qualified=False,
+        source_correct_goodput_qualified=False,numerical_profiles_identical=False,
+        native_or_stock_math_changes=False,cold_process_has_unmeasured_collector_request=False)
+    allowed=os.sched_getaffinity(0)
+    require(set(protocol['cpu_affinity'])<=allowed,'same actual four available client/provider CPUs')
+    urllib.request.install_opener(opener())
+    try:
+        os.sched_setaffinity(0,protocol['cpu_affinity'])
+        for label,mode in (('zerogravity','plain'),('zerogravity','observed'),('vllm','plain'),('vllm','observed')):
+            base=stock_observed if label=='vllm' and mode=='observed' else commands[label]
+            spec_command=derive(base,label,mode,args.own_freeze,artifact)
+            plan.spec['providers'][label]['argv']=spec_command
+            sample=dict(provider=label,mode=mode,policy=protocol['policy'],warmup=None)
+            with plan.sample(label,lambda:compare.preflight(urls[label],alias,None)) as owner:
+                if protocol['policy']=='warmed_process':
+                    warm=bench.stream_one(urls[label],alias,protocol['requests'][0]['prompt'],
+                        protocol['max_tokens'],180,None,request_id='matched-diag-warmup',return_token_ids=True)
+                    owner.record_warmup(asdict(warm)); sample['warmup']=asdict(warm)
+                    require(warm.completed,'exact canonical unmeasured warmed_process request')
+                before=idle(bench,urls[label]) if label=='zerogravity' else None
+                if before is not None: write(owner.directory/'health-before.json',before)
+                controller=None; started=False; primary=None; cleanup=[]
+                try:
+                    if mode=='observed':
+                        if label=='zerogravity':
+                            controller=Controller(owner,owned,parent)
+                            start=controller.command('START')
+                            started=True
+                            write(owner.directory/'window-start.json',start)
+                        else:
+                            sample['profile_start']=post(urls[label],'/start_profile')
+                            started=True
+                    measured=bench.bench(urls[label],alias,protocol['requests'][0]['prompt'],len(protocol['requests']),
+                        args.concurrency,protocol['max_tokens'],180,None,workload=protocol['requests'],
+                        return_token_ids=True,max_pending=protocol['max_pending'],
+                        launch_deadline_seconds=protocol['launch_deadline_seconds'],run_id='matched-cuda-activity')
+                    fingerprint(measured); owner.record_measurement(measured)
+                    sample['observation']=measured
+                    after=idle(bench,urls[label]) if label=='zerogravity' else None
+                    if after is not None: write(owner.directory/'health-after.json',after)
+                except BaseException as error:
+                    primary=error
+                finally:
+                    if started:
+                        try:
+                            if controller is not None:
+                                stop=controller.command('STOP')
+                                write(owner.directory/'window-stop.json',stop)
+                            else: sample['profile_stop']=post(urls[label],'/stop_profile')
+                        except BaseException as error: cleanup.append('collector stop: '+repr(error))
+                    if controller is not None:
+                        try: controller.close()
+                        except BaseException as error: cleanup.append('controller close: '+repr(error))
+                        try: write(owner.directory/'namespace-controller.json',controller.receipt)
+                        except BaseException as error: cleanup.append('controller receipt: '+repr(error))
+                    require(not cleanup and primary is None,
+                            'diagnostic primary='+repr(primary)+'; collector_cleanup='+repr(cleanup))
+            # All original owner/process/group/container/monitor/pipe cleanup finished.
+            require(not plan.blocked,'complete original retirement before next provider')
+            sample['owned_receipt_path']=str(owner.receipt_path)
+            if mode=='observed' and label=='vllm': sample['actual_recorded_CUDA_activity']=stock_traces(owner,audit)
+            report['samples'].append(sample)
+            tree_bound(owner.directory,768*1024**2)
+            tree_bound(output)
+        for label in ('zerogravity','vllm'):
+            pair=[item for item in report['samples'] if item['provider']==label]
+            require(fingerprint(pair[0]['observation'])==fingerprint(pair[1]['observation']),
+                    'complete plain/observed wire IDs/text/finish/usage exact for '+label)
+        report['output_transparency_exact']=True
+        report['status']='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        report['failure']=repr(error)[:4096]
+        raise
+    finally:
+        # All process/container owners already retired; independent evidence stages
+        # must still run if affinity restoration or a post-run source scan fails.
+        failures=[]
+        try: os.sched_setaffinity(0,allowed)
+        except BaseException as error: failures.append('restore client affinity: '+repr(error))
+        report['lifecycle_manifest']=str(plan.directory/'lifecycle.json')
+        try:
+            report['lifecycle_manifest_sha256']=sha(plan.directory/'lifecycle.json')
+        except BaseException as error: failures.append('lifecycle digest: '+repr(error))
+        try:
+            frozen(args.own_freeze); host_parents(parent); verify_preparation(v8); verify_small_sources()
+        except BaseException as error: failures.append('post-run immutable source closure: '+repr(error))
+        try:
+            source_check(pins,protocol['model'],source['source_path'],source['gguf']['path'])
+            require(sha(conversion/'mmproj-bf16.gguf')==project['projector_sha256']
+                    and sha(protocol['zerogravity_binary'])==protocol['zerogravity_binary_sha256']
+                    and sha(artifact/'collector.so')==build['collector_sha256']
+                    and sha(artifact/'build.json')==report['collector_build_sha256'],
+                    'unchanged actual model/projector/binary/collector/build after retirement')
+        except BaseException as error: failures.append('post-run complete payload closure: '+repr(error))
+        report['finalization_errors']=failures
+        if failures: report['status']='UNQUALIFIED'
+        write(output/'report.json',report)
+        if failures:
+            raise RuntimeError('diagnostic finalization failed; primary='+str(report.get('failure'))+'; '+repr(failures))
+
+
+
+if __name__=='__main__':
+    def interrupted(_signum,_frame): raise KeyboardInterrupt('ROOT container diagnostic interrupted')
+    saved=signal.signal(signal.SIGTERM,interrupted)
+    try: main()
+    finally: signal.signal(signal.SIGTERM,saved)
--- /dev/null
+++ container-v1/test_contracts.py
@@ -0,0 +1,92 @@
+"""ROOT-only mocked control/transport negatives; no process/CUDA/model proof."""
+import copy
+import json
+from pathlib import Path
+import struct
+import unittest
+from unittest.mock import patch
+
+from capacity_v1_loader import activity
+from container_activity import summarize
+from container_contracts import FILE_LIMIT
+from launch_derivative import derive
+from namespace_control import Controller,stat_start
+from validate_container import driver_identity
+
+
+def metadata():
+    return dict(schema='zg.cupti.window.capacity.v1',max_bytes=activity.MAX_BYTES,max_records=activity.MAX_RECORDS,
+        bytes=1,records=2,failures=0,dropped=0,first_cupti_error=0,phase='STOPPED',api_version=28,
+        pid=1,begin=10,end=100,timing_sample=False,actual_device_allocation_bound_qualified=False,
+        host_callback_pool_bytes=16*1024**2,device_buffer_requested_bytes=1024**2,
+        device_pool_limit=8,device_preallocate=2,kernels=1,driver_apis=1)
+
+
+def records(pid):
+    return [dict(kind='kernel',start=30,end=40,correlation=17,name='synthetic-only',grid=[1,1,1],
+        block=[32,1,1],context=1,stream=3,grid_id=5,graph=0,device=0),
+        dict(kind='driver',start=20,end=25,correlation=17,process=pid,thread=2,cbid=307,result=0)]
+
+
+def command():
+    return ['podman','run','--network=host','--ulimit','fsize=268435456:268435456',
+        '-e','LD_PRELOAD=','-e','LD_AUDIT=','--memory=12g','--cpus=4','--entrypoint','python3','IMAGE',
+        '/probe/server_boot_v8.py','--provider','zerogravity','--model','0.8b','--prepared-sha256','PARENT',
+        '--','serve','--batch-rows','8']
+
+
+class Contracts(unittest.TestCase):
+    def test_native_derivative_never_preloads_outer(self):
+        source=command();actual=derive(source,'zerogravity','observed','OWN',Path('/artifact'))
+        self.assertEqual(source,command())
+        self.assertIn('LD_PRELOAD=',actual)
+        self.assertNotIn('LD_PRELOAD=/collector-artifact/collector.so',actual)
+        self.assertIn('/collector-probe/native_exec.py',actual)
+        self.assertEqual(actual[actual.index('--')+1:],source[source.index('--')+1:])
+        self.assertIn(f'fsize={FILE_LIMIT}:{FILE_LIMIT}',actual)
+        self.assertIn('--memory=12g',actual);self.assertIn('--cpus=4',actual)
+
+    def test_stock_exact_bootstrap_unchanged_except_artifact_fsize(self):
+        source=command();source[source.index('zerogravity')]='vllm'
+        actual=derive(source,'vllm','observed','OWN',Path('/artifact'))
+        changed=list(source);changed[changed.index('fsize=268435456:268435456')]=f'fsize={FILE_LIMIT}:{FILE_LIMIT}'
+        self.assertEqual(actual,changed)
+
+    def test_unknown_original_boot_or_cap_refuses(self):
+        for old,new in (('/probe/server_boot_v8.py','other.py'),('fsize=268435456:268435456','fsize=1:1')):
+            source=command();source[source.index(old)]=new
+            with self.assertRaises(ValueError): derive(source,'zerogravity','plain','OWN',Path('/artifact'))
+
+    def test_namespace_and_raw_driver_ids_remain_separate(self):
+        for pid in (1,32100):
+            actual=records(pid);saved=copy.deepcopy(actual)
+            self.assertEqual(summarize(actual,metadata(),1,pid)['actual_gpu_kernel_records'],1)
+            self.assertEqual(actual,saved)
+            with self.assertRaises(ValueError): summarize(actual,metadata(),9,pid)
+            with self.assertRaises(ValueError): summarize(actual,metadata(),1,pid+1)
+
+    def test_driver_process_namespace_does_not_admit_foreign_or_mixed(self):
+        self.assertEqual(driver_identity(records(1),32100),1)
+        self.assertEqual(driver_identity(records(32100),32100),32100)
+        for raw in (records(7),records(1)+records(32100),[]):
+            with self.assertRaises(ValueError): driver_identity(raw,32100)
+
+    def test_proc_start_parser_handles_comm_spaces_parentheses(self):
+        fields=['S']+['0']*18+['123456']+['0']*8
+        self.assertEqual(stat_start('81 (name with (nested) parens) '+' '.join(fields)),123456)
+
+    def test_control_rejects_wrong_host_peer_even_with_namespace_reply(self):
+        owner=object.__new__(Controller)
+        owner.pid=32100;owner.nonce='a'*32;owner.live=lambda:None;owner.receipt={'events':[]}
+        class Socket:
+            def __enter__(self): return self
+            def __exit__(self,*args): pass
+            def settimeout(self,*args): pass
+            def connect(self,*args): pass
+            def getsockopt(self,*args): return struct.pack('3i',1,1000,1000)
+        with patch('namespace_control.socket.socket',return_value=Socket()),patch('namespace_control.os.geteuid',return_value=1000):
+            with self.assertRaises(ValueError): owner.command('START')
+        self.assertEqual(owner.receipt['events'],[])
+
+
+if __name__=='__main__': unittest.main()
--- /dev/null
+++ container-v1/validate_container.py
@@ -0,0 +1,215 @@
+"""ROOT-only independent complete-workload/namespace/activity/owner validator.
+
+Stock public Torch profiler loss completeness remains explicitly UNQUALIFIED;
+recorded kernel counts cannot establish source-correct goodput or timing speed.
+"""
+import argparse
+import hashlib
+from pathlib import Path
+import sys
+
+from capacity_v1_loader import activity
+from container_activity import summarize
+from container_contracts import HERE,FILE_LIMIT,TREE_LIMIT,build_binding,check_file,fingerprint,frozen,host_parents,read,require,sha,tree_bound,write
+from launch_derivative import derive
+
+
+def successful_owner(receipt,parent):
+    require(receipt['status']=='OWNERS_CLOSED' and receipt['cleanup_errors']==[]
+            and receipt['primary_failure'] is None and receipt['ready'] is True
+            and receipt['owned_container_absent_after_reap'] is True,'actual complete successful owned retirement')
+    cleanup=receipt['foreground_cleanup']
+    require(cleanup['pgid']==receipt['owned_cli_pid']>0 and cleanup['leader_reaped'] is True
+            and cleanup['group_absent'] is True and cleanup['all_nonzero_signals_before_reap'] is True
+            and cleanup['kill_attempts_before_reap']==2 and cleanup['leader_reap_attempts'] in (1,2)
+            and cleanup['stage_errors']==[],'all independent group signals before reap and final absence')
+    log=receipt['log_cleanup']
+    require(log['reader_complete'] is True and log['reader_error'] is None,'actual complete original log/pipe EOF')
+    observations=receipt['physical_observations']
+    require(observations['failures']==[] and observations['samples']
+            and observations['exact_peak'] is False
+            and observations['complete_physical_allocator_accounting'] is False,
+            'actual bounded original physical observations, no invented allocator peak')
+    for row in observations['samples']:
+        if row['compute_apps']:
+            require(all(app['pid'] in row['authenticated_container_host_pids'] for app in row['compute_apps']),
+                    'every actual GPU PID bound to original public container numeric HPIDs')
+    require(receipt['GPU_quiet_after'] and all(not row['compute_apps'] for row in receipt['GPU_quiet_after'][-3:]),
+            'actual final card quiet after original owner cleanup')
+
+
+def workload(result,protocol,concurrency):
+    require(result['offered_requests']==len(protocol['requests']) and result['concurrency']==concurrency
+            and result['max_pending']==protocol['max_pending']
+            and result['launch_deadline_seconds']==protocol['launch_deadline_seconds']
+            and result['run_id']=='matched-cuda-activity','exact canonical workload denominator and scheduling policy')
+    fingerprint(result)
+    require(len(result['requests'])==len(protocol['requests']),'all original canonical offers')
+    for index,(actual,expected) in enumerate(zip(result['requests'],protocol['requests'])):
+        require(actual['input_index']==index and actual['request_id']==expected['request_id']
+                and actual['prompt_sha256']==hashlib.sha256(expected['prompt'].encode()).hexdigest()
+                and actual['max_tokens']==expected['max_tokens'], 'actual full original prompt/max-token/request identity')
+
+
+def driver_identity(rows,host_pid):
+    ids=set()
+    for row in rows:
+        if row['kind']=='driver':
+            require(type(row['process']) is int,'actual raw driver process ID')
+            ids.add(row['process'])
+            require(len(ids)==1,'single native process driver identity; never merge namespaces')
+    require(len(ids)==1 and ids<={1,host_pid},'raw driver ID equals one authenticated process representation')
+    return next(iter(ids))
+
+
+def native_activity(directory,receipt,parent,report):
+    control=read(directory/'namespace-controller.json')
+    actual=read(directory/'native-exec-attestation.json')
+    require(control['pidfd_closed'] is True and control['reported_namespace_pid']==1
+            and control['exec_attestation']==actual and actual['nonce']==receipt['owned_nonce']
+            and control['binding']['owned_nonce']==receipt['owned_nonce']
+            and control['binding']['owned_name']==receipt['owned_name'], 'original owner/exec/controller binding')
+    hp=control['binding']['host_init_pid']
+    require(hp>0 and actual['pid_namespace_pid']==1 and actual['namespace_inodes']==control['host_before']['namespace_inodes']
+            and control['host_before']['NSpid'][0]==hp and control['host_before']['NSpid'][-1]==1,
+            'host init PID and actual namespace PID preserved independently')
+    require(actual['file_size_limits']==[FILE_LIMIT,FILE_LIMIT], 'actual finite diagnostic file-size readback')
+    for key,field in (('native','native_sha256'),('collector','collector_sha256'),('cupti','cupti_sha256')):
+        require(control['mapped_files'][key]['sha256']==actual[field] and control['mapped_files'][key]['mappings'],
+                'actual complete mapped artifact identity retained')
+    require(actual['collector_on_outer_python'] is False and actual['collector_requested_only_at_this_exec'] is True
+            and actual['native_sha256']==report['protocol']['zerogravity_binary_sha256']
+            and actual['collector_sha256']==report['collector_sha256']
+            and actual['collector_capacity_freeze_sha256']==report['capacity_freeze_sha256']
+            and actual['cupti_sha256']==parent['cupti_library']['sha256'],'only final-exec exact artifact/provider closure')
+    require(len(control['events'])==2,'one complete original native activity window')
+    start,stop=read(directory/'window-start.json'),read(directory/'window-stop.json')
+    for event,verb,value in zip(control['events'],('START','STOP'),(start,stop)):
+        require(event['verb']==verb and event['reply']==value and event['host_peer_pid']==hp
+                and event['host_peer_uid']==control['host_controller_uid'] and event['reported_namespace_pid']==1,
+                'actual host SO_PEERCRED authenticated independently from CUPTI namespace identity')
+    require(start['nonce']==stop['nonce']==receipt['owned_nonce']
+            and start['begin']==stop['begin'] and start['end']==0
+            and start['start_monotonic']==stop['start_monotonic']<stop['stop_monotonic'], 'one actual timestamped window')
+    path=directory/'observed-activity/activity.jsonl'
+    require(read(directory/'observed-activity/collector-stop.json')==stop,'persisted stop exactly acknowledged')
+    digest=sha(path)
+    driver_pid=driver_identity(activity.rows(path,stop['bytes']),hp)
+    driver_ids={driver_pid}
+    result=summarize(activity.rows(path,stop['bytes']),stop,1,driver_pid)
+    require(sha(path)==digest,'immutable entire raw trace across independent complete passes')
+    before=read(directory/'health-before.json')['value']['admission']
+    after=read(directory/'health-after.json')['value']['admission']
+    for value in (before,after):
+        require(value['active_sessions']==value['queued_requests']==value['output_credit_parked_requests']==0
+                and value['failed_gpu_quanta']==0,'actual terminal original GPU work at window boundaries')
+    require(after['accepted_output_tokens']>before['accepted_output_tokens'],'actual canonical output within window')
+    result.update(raw_activity_sha256=digest,raw_bytes=path.stat().st_size,namespace_pid=1,
+        authenticated_init_host_pid=hp,actual_raw_driver_process_ids=sorted(driver_ids),
+        driver_process_representation='namespace' if driver_pid==1 else 'host',
+        source_dispatch_delta=after['completed_forward_kernel_dispatches']-before['completed_forward_kernel_dispatches'],
+        source_dispatches_are_not_driver_measurements=True,all_retained_records_lossless=True)
+    return result
+
+
+def main():
+    parser=argparse.ArgumentParser()
+    parser.add_argument('--own-freeze',required=True)
+    parser.add_argument('--output',type=Path,required=True)
+    args=parser.parse_args()
+    parent=frozen(args.own_freeze); host_parents(parent)
+    output=args.output.resolve(strict=True)
+    require(output.parent==HERE,'owned diagnostic output')
+    report=read(output/'report.json')
+    require(report['status']=='EXECUTED_CONTAINER_ACTIVITY_REQUIRES_INDEPENDENT_VALIDATION'
+            and report['own_freeze_sha256']==args.own_freeze
+            and report['benchmark_v8_sha256']==parent['benchmark_v8_manifest']['sha256']
+            and report['capacity_freeze_sha256']==parent['capacity_manifest']['sha256']
+            and report['output_transparency_exact'] is True and report['finalization_errors']==[]
+            and report['performance_timing_usable'] is False
+            and report['native_or_stock_math_changes'] is False,'executed bounded diagnostic-only scope')
+    benchmark=Path(parent['parents']['benchmark']); sys.path.insert(0,str(benchmark))
+    from protocol_v8 import verify_preparation,verify_small_sources,validate_protocol
+    from prepare_launch_v8 import launch
+    from run_diagnostic_v8 import post
+    from audit_cuda_trace_v8 import audit
+    from cgroup_readback_v3 import validate_caps
+    from run_container import stock_traces
+    pins=verify_preparation(report['benchmark_v8_sha256']); verify_small_sources()
+    protocol=validate_protocol(report['protocol'])
+    protocol_path=Path(report['protocol_path'])
+    require(protocol_path.parent.parent==benchmark and protocol_path.name=='protocol.json'
+            and sha(protocol_path)==report['protocol_sha256'] and read(protocol_path)==protocol,
+            'actual immutable original V8 launch protocol file and complete value')
+    project=read(benchmark/'projector-pins.json')['models'][protocol['model']]
+    require(sha(report['lifecycle_manifest'])==report['lifecycle_manifest_sha256'],'complete original lifecycle manifest')
+    lifecycle=read(report['lifecycle_manifest'])
+    require(lifecycle['owned_cleanup_source_sha256']==parent['owned_helper_sha256']
+            and lifecycle['adapter_source_sha256']==sha(benchmark/'owned_plan_v8.py')
+            and len(lifecycle['receipts'])==len(report['samples'])==4,'all four original owner receipts')
+    order=(('zerogravity','plain'),('zerogravity','observed'),('vllm','plain'),('vllm','observed'))
+    native_summary=stock_summary=None
+    for sample,receipt_path,expected in zip(report['samples'],lifecycle['receipts'],order):
+        label,mode=expected
+        require((sample['provider'],sample['mode'])==expected and sample['owned_receipt_path']==receipt_path,
+                'all four actual distinct ordered samples')
+        receipt=read(receipt_path); directory=Path(receipt_path).parent
+        require(output in directory.parents,'original owned sample stays in this exclusive output')
+        tree_bound(directory,768*1024**2)
+        successful_owner(receipt,parent)
+        bootstrap=read(directory/'bootstrap.json')
+        validate_caps(bootstrap['cgroup_readback'],protocol['cpu_affinity'])
+        require(bootstrap['provider']==label and bootstrap['preparation_sha256']==report['benchmark_v8_sha256']
+                and bootstrap['stock_math_modified'] is False
+                and bootstrap['actual_cpu_affinity']==sorted(protocol['cpu_affinity']), 'original runtime caps/source bootstrap')
+        base=launch(protocol,pins,report['benchmark_v8_sha256'],label,Path(project['conversion_directory']),
+            profiler=label=='vllm' and mode=='observed')
+        # Receipt's mounted artifact path is authenticated by actual exec SHA/build
+        # below; all other launch bytes must be exact derivation of immutable V8.
+        artifact=Path(next(receipt['argv'][i+1].removesuffix(':/collector-artifact:ro')
+            for i,v in enumerate(receipt['argv'][:-1]) if v=='-v' and receipt['argv'][i+1].endswith(':/collector-artifact:ro'))) if label=='zerogravity' else HERE
+        if label=='zerogravity':
+            build=build_binding(artifact/'build.json',artifact/'collector.so',parent)
+            require(sha(artifact/'build.json')==report['collector_build_sha256']
+                    and build['collector_sha256']==report['collector_sha256'], 'independent actual ROOT build binding')
+        command=derive(base,label,mode,args.own_freeze,artifact)
+        replacements={'{owned_name}':receipt['owned_name'],'{owned_nonce}':receipt['owned_nonce'],
+            '{cidfile}':str(directory/'cid'),'{sample_directory}':str(directory)}
+        for key,value in replacements.items(): command=[part.replace(key,value) for part in command]
+        require(receipt['argv']==command,'every actual launch argument equals reviewed exact canonical derivative')
+        workload(sample['observation'],protocol,report['concurrency'])
+        require(read(directory/'measurement.json')==sample['observation'],'actual original measured HTTP observations')
+        if protocol['policy']=='cold_process':
+            require(sample['warmup'] is None and receipt['warmup'] is None,'no collector-only warmup in cold_process')
+        else:
+            require(sample['warmup']==receipt['warmup'] and sample['warmup']['completed'] is True
+                    and sample['warmup']['tokens']>0,'actual exact canonical warmed_process request')
+        if label=='zerogravity':
+            actual=read(directory/'native-exec-attestation.json')
+            require(actual['mode']==mode and actual['argv']==['/runtime/zerogravity']+bootstrap['engine_argv']
+                    and actual['own_freeze_sha256']==args.own_freeze,'same original final exec and own closure')
+            if mode=='observed': native_summary=native_activity(directory,receipt,parent,report)
+        elif mode=='observed':
+            for key,route in (('profile_start','/start_profile'),('profile_stop','/stop_profile')):
+                require(sample[key]['route']==route and sample[key]['HTTP_status']==200,'actual original public profiler controls')
+            # Reparse complete original trace after owner is gone, not stored report count.
+            class Owner: pass
+            owner=Owner(); owner.directory=directory
+            stock_summary=stock_traces(owner,audit)
+            require(stock_summary==sample['actual_recorded_CUDA_activity'],'independent entire actual stock trace replay')
+    for label in ('zerogravity','vllm'):
+        pair=[item for item in report['samples'] if item['provider']==label]
+        require(fingerprint(pair[0]['observation'])==fingerprint(pair[1]['observation']),'exact all wire token/text/finish/usage transparency')
+    require(native_summary is not None and stock_summary,'both actual canonical workload activity scopes present')
+    result=dict(status='PASS_CONTAINER_RECORDED_WORKLOAD_ACTIVITY_DIAGNOSTIC_ONLY',
+        own_freeze_sha256=args.own_freeze,report_sha256=sha(output/'report.json'),
+        native=native_summary,stock_recorded=stock_summary,artifact_bounds=tree_bound(output),
+        same_canonical_workload_policy_and_caps_checked=True,plain_observed_wire_outputs_exact=True,
+        source_profiles_numerically_equal=False,source_correct_goodput_qualified=False,
+        matched_complete_hardware_kernel_counts_qualified=False,stock_collector_losslessness_verified=False,
+        timing_usable=False,device_profiler_allocation_admission_qualified=False)
+    frozen(args.own_freeze); host_parents(parent); verify_preparation(report['benchmark_v8_sha256'])
+    write(output/'validation.json',result)
+
+
+if __name__=='__main__': main()

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/NEW_CONTROL_V2.review.patch

Original UTF-8 bytes: 25917; SHA256 `94493f65e64b810d411fa253d7507480eab5349920b06b8a09735c729e09e257`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ native_ab_v2.py
@@ -0,0 +1,17 @@
+"""Additive source binding only; all original native/native contracts unchanged."""
+import native_ab_common as original
+from native_ab_common import *
+
+V1 = '6e5fd796b695e1a3c662502319ba1a7608ff1c5584cea3e3cc4f7e3aa7fc3bf2'
+
+
+def source_freeze():
+    require(original.source_freeze() == V1, 'immutable original native/native preparation')
+    path = HERE / 'followup-v2/freeze-manifest-v2.json'
+    value = load(path)
+    for row in value['files']:
+        pin(dict(row, path=str(HERE / row['file'])))
+        pin(dict(row, path=str(HERE / 'followup-v2/frozen' / row['file'])))
+    for row in value['original_wrapper_and_support']:
+        pin(row)
+    return sha(path)
--- /dev/null
+++ harmless_http_v2.py
@@ -0,0 +1,58 @@
+"""Explicit CPU-only cancellation target. NOT a model/native serving engine."""
+import argparse
+from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
+import json
+import signal
+import threading
+import time
+
+
+class Handler(BaseHTTPRequestHandler):
+    def do_GET(self):
+        value = dict(control_only=True, model_or_GPU_loaded=False)
+        if self.path == '/phase':
+            value['post_open'] = self.server.post_open.is_set()
+        raw = json.dumps(value).encode()
+        # Original native readiness must NOT accept this target.
+        self.send_response(503 if self.path == '/v1/models' else 200)
+        self.send_header('Content-Length', str(len(raw)))
+        self.end_headers()
+        self.wfile.write(raw)
+
+    def do_POST(self):
+        length = int(self.headers.get('Content-Length', '0'))
+        if not 0 < length <= 4096:
+            self.send_error(400)
+            return
+        self.rfile.read(length)
+        self.send_response(200)
+        self.send_header('Content-Type', 'text/event-stream')
+        self.end_headers()
+        self.wfile.flush()
+        self.server.post_open.set()
+        print('CONTROL_POST_OPEN', flush=True)
+        # No IDs/output/model arithmetic. Parent uses short canonical timeout;
+        # an independently retained owner kills this harmless target afterward.
+        time.sleep(20)
+
+    def log_message(self, *_):
+        pass
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument('--port', type=int, required=True)
+    args = parser.parse_args()
+    if not 1024 <= args.port <= 65535:
+        raise SystemExit('bounded control port')
+    # Exercise original owned TERM->KILL path. Only this fresh owned container
+    # is affected; this process has no GPU device/model/source mounts.
+    signal.signal(signal.SIGTERM, signal.SIG_IGN)
+    print('HARMLESS_CONTROL_READY_NO_MODEL_NO_GPU', flush=True)
+    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
+    server.post_open = threading.Event()
+    server.serve_forever()
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ test_owner_process_v2.py
@@ -0,0 +1,250 @@
+"""ROOT ONLY real harmless owners; NEVER GPU/model/native-engine qualification."""
+import argparse
+import os
+from pathlib import Path
+import signal
+import subprocess
+import threading
+import time
+from types import SimpleNamespace
+
+from native_ab_v2 import (HERE, http_json, install_opener, load, parents, require,
+                         save, sha, source_freeze, tree)
+from native_ab_owner import NativeOwner
+from run_native_ab_v2 import private_namespace
+from sigterm_cancellation import SigtermCancellation
+
+
+def control_observation(*_):
+    # Deliberate fixture stub, visibly unlike real point-memory samples. Never
+    # call nvidia-smi, attach a device or imply GPU quietness in this fixture.
+    return [dict(memory_used_mib=0, control_only=True, GPU_observation_performed=False)]
+
+
+class NoGpuMonitor:
+    def __init__(self, *_):
+        pass
+
+    def finish(self):
+        return dict(control_only=True, GPU_observation_performed=False, samples=[])
+
+
+def wait_control(owner, stop, failures):
+    deadline = time.monotonic() + 60
+    while not stop.is_set() and time.monotonic() < deadline:
+        try:
+            require(owner.owned.owned_container(owner.name, owner.nonce, owner.cidfile) is not None,
+                    'actual harmless nonce/CID ownership')
+            value = http_json(owner.endpoint() + '/health')
+            if value == dict(control_only=True, model_or_GPU_loaded=False):
+                return True
+        except Exception:
+            pass
+        stop.wait(.05)
+    if not stop.is_set():
+        failures.append('harmless target readiness deadline')
+    return False
+
+
+def signal_when_ready(owner, signum, stop, failures):
+    if wait_control(owner, stop, failures) and not stop.is_set():
+        # Only our still-running supervisor receives the actual signal. The
+        # original guard records it; no guessed/exited external PID is used.
+        os.kill(os.getpid(), signum)
+
+
+def signal_during_request(owner, signum, stop, failures):
+    deadline = time.monotonic() + 10
+    while not stop.is_set() and time.monotonic() < deadline:
+        try:
+            value = http_json(owner.endpoint() + '/phase')
+            if value == dict(control_only=True, model_or_GPU_loaded=False, post_open=True):
+                require(owner.owned.owned_container(owner.name, owner.nonce, owner.cidfile) is not None,
+                        'actual nonce/CID during open harmless request')
+                require(owner.owned.leader_status_unreaped(owner.process) is None,
+                        'original foreground live and unreaped during request')
+                if stop.is_set():
+                    return
+                owner.receipt['signal_at_open_control_POST'] = value
+                os.kill(os.getpid(), signum)
+                return
+        except Exception:
+            pass
+        stop.wait(.02)
+    if not stop.is_set():
+        failures.append('actual open harmless POST not observed before signal deadline')
+
+
+def start_control(owner):
+    """Fixture-only transport start; native/model readiness is never forged."""
+    require(not owner.lifecycle.listening(('127.0.0.1', owner.protocol['ports'][0])),
+            'exclusive private harmless endpoint')
+    owner.receipt['GPU_quiet_before'] = control_observation()
+    replacements = {'{owned_name}': owner.name, '{owned_nonce}': owner.nonce,
+                    '{cidfile}': str(owner.cidfile), '{sample_directory}': str(owner.directory)}
+    command = owner.command
+    for key, value in replacements.items():
+        command = [part.replace(key, value) for part in command]
+    owner.receipt['argv'] = command
+    owner.cancel.check()
+    owner.process = subprocess.Popen(command, cwd=str(owner.protocol_root()),
+        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
+        start_new_session=True, bufsize=0)
+    owner.receipt['owned_cli_pid'] = owner.process.pid
+    owner.log = owner.lifecycle.BoundedLog(owner.process.stdout, owner.directory / 'execution.log', 8 * 1024**2)
+    owner.save()
+
+
+def case(root, index, phase, signum, dependencies):
+    protocol, _, owned, _, _, pins, bench, _ = dependencies
+    port = 18520 + index
+    p = dict(model='CONTROL_ONLY_NOT_A_MODEL', zerogravity_binary_sha256='NOT_A_NATIVE_BINARY',
+             ports=[port, port + 100], gpu_uuid='NO_GPU_IN_CONTROL_FIXTURE', gpu_total_mib=0,
+             cpu_affinity=sorted(os.sched_getaffinity(0)))
+    # Genuine pinned local image/Podman, no model/source/native binary/device
+    # mounts and no acquisition. The only executed payload is this stdlib stub.
+    command = owned.PODMAN + ['run', '--name', '{owned_name}', '--label', owned.LABEL + '={owned_nonce}',
+        '--cidfile', '{cidfile}', '--rm', '--pull=never', '--network=host', '--read-only',
+        '--cap-drop=all', '--security-opt=no-new-privileges', '--memory=256m', '--memory-swap=256m',
+        '--cpus=1', '--pids-limit=32', '--ulimit', 'fsize=4194304:4194304',
+        '--tmpfs', '/tmp:rw,nosuid,nodev,size=16m', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
+        '-e', 'CUDA_VISIBLE_DEVICES=-1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
+        '-v', str(HERE / 'harmless_http_v2.py') + ':/control.py:ro',
+        '--entrypoint', 'python3', pins['image'], '/control.py', '--port', str(port)]
+    # Original native adapter cleanup/log owner, only the GPU observer is an
+    # explicit control stub. Its empty samples cannot pass native validation.
+    deps = list(dependencies)
+    deps[3] = SimpleNamespace(quiet=control_observation, Monitor=NoGpuMonitor)
+    directory = root / f'{index:02d}-{phase}-{signum or 0}'
+    stop, failures = threading.Event(), []
+    thread = None
+    expected, outcome, close_failure = None, None, None
+    old_cleanup = None
+    with SigtermCancellation() as cancellation:
+        owner = NativeOwner('HARMLESS_CONTROL', directory, command, p, deps, cancellation, root)
+        owner.receipt.update(native_implementation='HARMLESS_CONTROL_NOT_NATIVE_ENGINE',
+            control_only=True, model_GPU_native_math_or_benchmark_qualified=False,
+            original_native_readiness_not_forged=True, phase=phase, signal=signum)
+        try:
+            if phase == 'before_popen':
+                os.kill(os.getpid(), signum)
+                expected = 'InterruptedError'
+                owner.start()  # Must refuse before ANY Popen/device/model work.
+            elif phase == 'readiness':
+                thread = threading.Thread(target=signal_when_ready,
+                    args=(owner, signum, stop, failures))
+                thread.start()
+                expected = 'InterruptedError'
+                owner.start()  # Actual stdlib target returns503, never native readiness.
+            else:
+                start_control(owner)
+                wait_control(owner, stop, failures)
+                require(not failures, 'actual harmless target startup')
+                if phase == 'request':
+                    thread = threading.Thread(target=signal_during_request,
+                        args=(owner, signum, stop, failures))
+                    thread.start()
+                    observed = bench.stream_one(owner.endpoint(), 'HARMLESS_CONTROL_NOT_A_MODEL',
+                        'control-only cancellation transport', 1, 2, None,
+                        request_id='harmless-control', return_token_ids=True)
+                    require(observed.completed is False, 'control stub emits no fake token/model result')
+                    expected = 'InterruptedError'
+                    cancellation.check()
+                elif phase == 'cleanup':
+                    old_cleanup = owned.cleanup
+                    def signalled_cleanup(*args):
+                        os.kill(os.getpid(), signum)
+                        return old_cleanup(*args)  # Delegate unchanged genuine53ee stages.
+                    owned.cleanup = signalled_cleanup
+                else:
+                    require(phase == 'receipt_io', 'closed fixture phase')
+                    expected = 'OSError'
+                    def fail_receipt():
+                        raise OSError('injected postspawn receipt failure')
+                    owner.save = fail_receipt
+                    owner.save()
+        except BaseException as error:
+            outcome = type(error).__name__
+            owner.receipt['expected_control_failure'] = repr(error)
+        finally:
+            try:
+                owner.close()
+            except BaseException as error:
+                close_failure = type(error).__name__
+            finally:
+                if old_cleanup is not None:
+                    owned.cleanup = old_cleanup
+                stop.set()
+                if thread is not None:
+                    # One in-flight original owned inspect (10s) plus HTTP(2s)
+                    # must retire; no signal thread survives guard restoration.
+                    thread.join(15)
+                    require(not thread.is_alive(), 'own signal thread retired')
+        require(not failures, 'control readiness failed, never accepted as negative proof')
+        if phase == 'cleanup':
+            require(cancellation.received is True and outcome is None and close_failure is None,
+                    'recorded signal did not interrupt original cleanup')
+        elif phase == 'receipt_io':
+            require(outcome == expected and close_failure == 'OSError', 'actual expected postspawn IO failure')
+        else:
+            require(outcome == expected and close_failure is None, 'actual expected recorded signal')
+            if phase == 'request':
+                require(owner.receipt.get('signal_at_open_control_POST', {}).get('post_open') is True,
+                        'request cancellation requires actual open transport evidence')
+        require(owner.receipt.get('ready') is not True, 'fixture never accepted native/model readiness')
+        if phase == 'before_popen':
+            require(owner.process is None, 'no child after observed prelaunch cancellation')
+        else:
+            fg, log = owner.receipt['foreground_cleanup'], owner.receipt['log_cleanup']
+            require(owner.receipt['owned_container_absent_after_reap'] is True and
+                    all(fg[k] is True for k in ('leader_reaped', 'group_absent', 'all_nonzero_signals_before_reap')) and
+                    fg['stage_errors'] == [] and log['reader_complete'] is True and
+                    log['reader_error'] is None and log['truncated'] is False and owner.receipt['cleanup_errors'] == [],
+                    'real harmless group/container/EOF closed even after signal/IO failure')
+            require(owned.owned_container(owner.name, owner.nonce, owner.cidfile) is None,
+                    'actual final nonce-bound container absence')
+        # Restore only fixture-injected method, AFTER real owner retirement, so
+        # expected receiptIO failure can retain its independent cleanup evidence.
+        owner.save = lambda: save(directory / 'receipt.json', owner.receipt, root)
+        owner.save()
+        value = dict(status='PASS_HARMLESS_CONTROL_OWNER_ONLY', phase=phase, signal=signum,
+            receipt=str(directory / 'receipt.json'), receipt_sha256=sha(directory / 'receipt.json'),
+            expected_failure=expected, observed_failure=outcome, close_failure=close_failure,
+            real_owned_child=owner.process is not None, GPU_model_native_math_or_benchmark_qualified=False)
+        save(directory / 'control-validation.json', value, root)
+        return value
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--run-control-only', action='store_true')
+    parser.add_argument('--new-output', type=Path, required=True)
+    args = parser.parse_args()
+    require(args.run_control_only, 'ROOT explicitly executes harmless owners only after source review')
+    freeze = source_freeze()
+    dependencies = parents()
+    install_opener()
+    namespace = private_namespace()
+    root = args.new_output.resolve()
+    require(root.parent == HERE and not root.exists(), 'fresh own control artifacts')
+    root.mkdir(mode=0o700)
+    report = dict(status='RUNNING_HARMLESS_CONTROL_ONLY', source_freeze_sha256=freeze,
+                  namespace=namespace, cases=[], GPU_model_native_math_or_benchmark_qualified=False)
+    try:
+        cases = [(phase, sig) for phase in ('before_popen', 'readiness', 'request', 'cleanup')
+                 for sig in (signal.SIGTERM, signal.SIGINT)] + [('receipt_io', None)]
+        for index, (phase, sig) in enumerate(cases):
+            report['cases'].append(case(root, index, phase, sig, dependencies))
+            save(root / 'control-report.json', report, root)
+        require(len(report['cases']) == 9 and source_freeze() == freeze, 'all harmless cases and original sources')
+        report['status'] = 'PASS_NINE_HARMLESS_CONTROL_CASES_NOT_NATIVE_OR_GPU_QUALIFICATION'
+    except BaseException as error:
+        report['failure'] = repr(error)[:2048]
+        raise
+    finally:
+        save(root / 'control-report.json', report, root)
+        tree(root)
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ test_namespace_v2.py
@@ -0,0 +1,66 @@
+"""Prepared pure namespace-contract fixtures; no namespace/container execution."""
+from contextlib import ExitStack
+import os
+from pathlib import Path
+import struct
+from types import SimpleNamespace
+import unittest
+from unittest.mock import patch
+
+import run_native_ab_v2 as runner
+
+
+class Namespace(unittest.TestCase):
+    def namespace(self, mapping='0 1000 1\n1 100000 65536\n', configured='done',
+                  uid='1000', network_owner=9, interfaces=None, up=True):
+        interfaces = [(1, 'lo')] if interfaces is None else interfaces
+        def ioctl(_descriptor, request, *_):
+            if request == 0xB701:
+                return 11
+            self.assertEqual(request, 0x8913)
+            return struct.pack('16sH22x', b'lo', int(up))
+        with ExitStack() as stack:
+            stack.enter_context(patch.object(Path, 'read_text', return_value=mapping))
+            stack.enter_context(patch.dict(os.environ, {
+                '_CONTAINERS_USERNS_CONFIGURED': configured,
+                '_CONTAINERS_ROOTLESS_UID': uid}, clear=True))
+            stack.enter_context(patch.object(runner.os, 'open', return_value=10))
+            stack.enter_context(patch.object(runner.os, 'close'))
+            stack.enter_context(patch.object(runner.os, 'fstat', return_value=SimpleNamespace(st_ino=network_owner)))
+            stack.enter_context(patch.object(runner.os, 'stat', return_value=SimpleNamespace(st_ino=9)))
+            stack.enter_context(patch.object(runner.os, 'readlink', side_effect=lambda path: 'mock-only:' + path))
+            stack.enter_context(patch.object(runner.fcntl, 'ioctl', side_effect=ioctl))
+            stack.enter_context(patch.object(runner.socket, 'if_nameindex', return_value=interfaces))
+            stack.enter_context(patch.object(runner.socket, 'socket'))
+            return runner.private_namespace()
+
+    def test_exact_existing_full_rootless_map(self):
+        value = self.namespace()
+        self.assertEqual(value['uid_map'].split(), ['0', '1000', '1', '1', '100000', '65536'])
+        self.assertEqual(value['rootless_userns_configured'], 'done')
+        self.assertEqual(value['rootless_uid'], '1000')
+
+    def test_known_bad_single_uid_route_refused(self):
+        with self.assertRaises(ValueError):
+            self.namespace(mapping='0 1000 1\n')
+
+    def test_wrong_container_bootstrap_refused(self):
+        for fields in ({'configured': ''}, {'uid': '0'}):
+            with self.subTest(fields=fields), self.assertRaises(ValueError):
+                self.namespace(**fields)
+
+    def test_foreign_network_owner_refused(self):
+        with self.assertRaises(ValueError):
+            self.namespace(network_owner=8)
+
+    def test_non_loopback_interface_refused(self):
+        with self.assertRaises(ValueError):
+            self.namespace(interfaces=[(1, 'lo'), (2, 'eth0')])
+
+    def test_loopback_must_already_be_up(self):
+        with self.assertRaises(ValueError):
+            self.namespace(up=False)
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ README_V2.md
@@ -0,0 +1,86 @@
+# Native baseline/candidate V2 namespace correction
+
+Author: batch_design_review. Parent: serving-and-speculation P12.
+SOURCE ONLY; no author imports/tests, process/container/model/GPU execution,
+builds or binary/model/DSO payload reads/hashes.
+
+- Original12-file freeze6e5fd796 and original8 fixtures remain immutable.
+  V2 adds exact full rootless Podman UID mapping/bootstrap and current-user-owned
+  private network/loopback checks from accepted `root-private-loopback-exec-2.py`.
+  Its full mapping preserves the custom store's overlay ownership. The original
+  single-UID `native-gpu.py` invocation is unsuitable for this Podman driver.
+- `prepare_native_ab_v2.py` changes only its source-binding import.
+  `run_native_ab_v2.py` changes that import and namespace checks only.
+  `validate_native_ab_v2.py` changes its binding/runner imports only.
+  `native_ab_v2.py` checks both immutable V1 and additive V2 live/frozen bytes,
+  plus exact accepted wrapper/support bytes. Original workload, launcher, caps,
+  owner, cancellation recorder, numerical/accounting validation are unchanged.
+- Candidate gate now physically exists: ROOT worker14/HTTP12 receipt18625960,
+  source456, server87e1d727. Preparation still requires its explicit full SHA;
+  no pending/partial gate is accepted. This derivative contains no timing result.
+- Both native labels/models, warmed_process C1 and C8, ABBA+BAAB remain32 fresh
+  measured processes /144 offered requests /32 ordinary canonical warmups.
+  The canonical first prompt has47 IDs, max8. Neither label is a vLLM slot.
+  Original README metrics/limitations still apply.
+
+Prepared additional fixtures, ROOT executes only after source review:
+
+- `test_namespace_v2.py`: six pure/mocked tests; full mapping accepted, known
+  bad single UID/missing container bootstrap/foreign network/non-loopback/down
+  loopback refused. These are contract tests, not actual namespace evidence.
+- `test_owner_process_v2.py`: nine actual harmless-owner cases: TERM+INT before
+  Popen, during readiness, during a real open HTTP POST, during original owner
+  cleanup; one postspawn receipt-I/O failure. Only stdlib `harmless_http_v2.py`
+  runs in the already-pinned local image, no pulls, devices, model/source/native
+  binary mounts. Container256MiB/CPU1/pids32; no provider/native math.
+- Target `/v1/models` ALWAYS503; POST emits no ID/token/text/usage. Fixture
+  readiness is a separately labelled control health endpoint. Request signal
+  requires actual POST-open marker and authenticated live nonce/CID/foreground;
+  no delay-based inference. Original unchanged `serve-bench.stream_one` is used
+  only as bounded transport (fixture timeout2s/max1); this is NOT a canonical
+  benchmark sample. Its failure is mandatory.
+- Original NativeOwner and53ee WNOWAIT/container/group/EOF cleanup are exercised.
+  Signal goes only to the current supervisor, whose original recorder cannot
+  raise across handoff/cleanup. Actual container cleanup signals only original
+  authenticated owners. Stop/join precedes handler restoration; no next fixture
+  after any unexpected failure. The target ignores TERM to exercise owner kill.
+- GPU observation is an explicit `control_only` stub. Empty point samples and
+  `ready!=true`, HARMLESS labels/non-native binary prevent native acceptance.
+  No GPU quietness, engine/bootstrap/cgroup qualification follows these tests.
+  Receipt-I/O failure cannot skip cleanup; actual closure is retained afterward
+  via a restored fixture writer, with expected failures still explicit.
+
+ROOT invocation (not executed here):
+
+```bash
+repo=/home/pvl/spaces/abird/src/zerogravity
+own="$repo/.agents/runs/serving-and-speculation-native-before-after-20261020"
+parent="$repo/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020"
+export ZG_BENCH_HOST_ROOT="$repo" PYTHONDONTWRITEBYTECODE=1
+python3 -m unittest discover -s "$own" -p 'test_namespace_v2.py'
+
+# Exact accepted custom-store rootless wrapper2 chain, fresh private net/mount.
+podman --root /home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers \
+  --runroot /run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers \
+  unshare unshare --net --mount --propagation private \
+  python3 "$parent/root-private-loopback-exec-2.py" \
+  python3 "$own/test_owner_process_v2.py" --run-control-only \
+  --new-output "$own/root-harmless-v2-1"
+```
+
+After all fixtures/review and ROOT's exclusive GPU reservation:
+
+1. Use `prepare_native_ab_v2.py` with unchanged original preparation arguments;
+   fresh own plan directory and exact accepted V8 hardware-protocol SHA, plus
+   `--accepted-serving-sha256
+   186259602e1debf69df208c5f12be0f60687a6fb81c43a699c34f5a55dcd0431`.
+   ROOT performs original source/binary/weight binding scans outside samples.
+2. Repeat the exact Podman/unshare/wrapper2 chain above, replacing ONLY its final
+   Python command with `run_native_ab_v2.py --run --plan ABS_PLAN/plan.json
+   --plan-sha256 ACTUAL_SHA --new-output ABS_OWN/root-native-v2-1`.
+3. `validate_native_ab_v2.py --output ABS_OWN/root-native-v2-1`; review all raw
+   samples/owners/ITL coverage. Failures do not permit another sample/provider.
+
+Own V2 sources only were AST parsed. Real fixtures, preparation, actual timings
+and independent full receipt validation remain ROOT gates. No speed, hardware
+count, source-correct goodput, full HF or vLLM parity claim.
--- /dev/null
+++ TODO_V2.md
@@ -0,0 +1,13 @@
+# P12 native/native before-after V2
+
+Parent: `.agents/plans/serving-and-speculation/TODO.md`; owner batch_design_review.
+
+- [x] Preserve original6e5fd796 live/frozen12 files and ROOT original8 fixture PASS.
+- [x] Full accepted rootless UID/bootstrap/network/loopback guard; V2 bound imports.
+- [x] Prepare6 pure namespace and9 real harmless owner fixtures (not executed).
+- [x] Bind actual POST-open evidence, original owner/EOF, explicit no-GPU stub.
+- [x] Bounded review actual worker14/HTTP12 receipt/logs: no scope mismatch.
+- [x] Source-only AST/read/freeze; no author runtime or payload scans.
+- [ ] Independent V2 source review; ROOT pure and actual harmless fixture execution.
+- [ ] ROOT preparation and actual32-process canonical warmed native/native run.
+- [ ] Independent all samples/owners/numerics/accounting validation; no timing claim yet.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/NEW_FILES.review.patch

Original UTF-8 bytes: 78518; SHA256 `bc90626d98768eddf94920d518ac51cff80b03cb22b6d128c9b69003b0c8a654`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ README.md
@@ -0,0 +1,96 @@
+# Native baseline/candidate latency preparation
+
+Author: batch_design_review. Parent: serving-and-speculation P12.
+SOURCE ONLY; no author imports/tests, child/provider/model/GPU execution, builds,
+binary/model/DSO payload scans or timing results.
+
+- Explicit labels `native_baseline` (4a745527...) and `native_candidate`
+  (87e1d727...). Both are Zerogravity. Neither occupies a vLLM comparison slot;
+  no `serve-compare` labels, legacy ratios or `measure_same_engine` counters.
+- Direct unchanged canonical `tools/serve-bench.py`. The immutable V8 native
+  launch/bootstrap branch remains `provider=zerogravity`, its genuine ENGINE
+  identity, for both new result labels. Stock/vLLM is never launched. Paired
+  launch command normalization requires identical flags/mounts/environment
+  except the immutable native executable and its expected hash.
+- Both sources, warmed_process, C1 canonical solo and C8 canonical eight-offer
+  wave, max8. Exact existing47 prompt IDs/common greedy/EOS policy. Each of four
+  model/band cases runs ABBA then BAAB, eight fresh owned processes:32 measured
+  samples,144 measured offers (16solo+128wave),72 offers/native label, plus32
+  independent unmeasured warmup requests. ONE normal canonical prompt/max8
+  warmup per process; no extra shape/full-wave warmup.
+- Same exact four CPUs, CPU4 quota, RAM12GiB/no extra swap, original UUID/context,
+  active8/native4GiB host/provider+5GiB device, read-only inputs and private user/
+  network namespace; same V8 cgroup readback/point device observation. Native
+  graph remains disabled in the accepted source. This is neither cold-process
+  nor fully warmed shape/graph nor persistent-server steady-state measurement.
+
+Gates and ownership:
+
+- `bindings.json` pins the old actual accepted native metadata, candidate456
+  source freeze, strict/release build, operator10/model46/full isolated+both
+  approved routes and exact future worker14/HTTP12 receipt location. ROOT must
+  supply its completed receipt SHA. Missing/RUNNING/partial/failed/unclean or
+  unexpected-count receipt refuses preparation AND execution AND validation.
+  ROOT should inspect all gate evidence before supplying the SHA; marker alone
+  is insufficient. Later source drift or another runner version needs a fresh
+  derivative, never a silently widened acceptance predicate.
+- Original53ee nonce/name/CID container + reserved-leader WNOWAIT whole-group
+  TERM/KILL-before-any-reap cleanup; original bounded log reader/EOF and V8
+  ownership-authenticated point-memory monitor. Exact00d1 TERM+INT record-only
+  guard spans provider lifetime, with explicit checks outside Popen handoff and
+  retirement. No asynchronous exception crosses an unassigned child handle.
+- Canonical HTTP retains its existing180s deadline/socket-timeout controls. A
+  recorded cancel is checked after canonical return; this is no new hard
+  asynchronous interrupt bound. Owner remains retained. Cleanup ignores recorded
+  signals until all original stages close. No subsequent sample after any
+  numerical, counter, transport, owner, log, memory-observer or receipt failure.
+- Source/weights/binary binding scans run only in ROOT preparation/before+after
+  the whole run, outside timed samples. Logs8MiB/process with truncation refused;
+  whole output512MiB/1024files, every metadata write and final validation checks
+  the tree INCLUDING its newly written receipt. These are artifact acceptance
+  bounds, not a parent process physical-memory admission claim.
+
+Metrics and acceptance:
+
+- Every sample retains full raw benchmark BEFORE numerical/accounting checks.
+  Exact full IDs/text SHA/usage/finish match all rows/versions/bands, including
+  each ordinary warmup. No EOS/usage normalization or minimum-token workaround.
+- Quiescent health before/after excludes warmup and cleanup. Every required
+  counter is present, typed, monotonic; error/refusal/discard/cancel deltas0;
+  weighted2/4/8 bands close actual fused rows, phases and sampler4/cohort. These
+  are source launchsites, never CUDA/CUPTI enumerated kernel counts.
+- C8 is client concurrency, not a promised nativeB8 cohort. Preserve actual row
+  band deltas and packet cohort widths; no padding or forced cohort schedule.
+  Actual packet frontiers47+index bind independent output progression. State/
+  prefix cohorts have separate counters and are not reconstructed from SSE.
+- Text TTFT and actual single-ID TTFT remain distinct. Keep offered/client-
+  backlog TTFT and service TTFT. Individual-token ITL requires complete actual
+  single-ID delivery; coalesced/unresolved rows are counted, never replaced by
+  SSE chunk or meanITL proxies. Zero/one output ID has no ITL interval. Native
+  commit observations remain a separate host clock, not device timestamps.
+- Report offered/completed/usage throughput and all raw paired latencies plus
+  local-label fairness, sampled memory, actual source counters. Four samples per
+  label/band yield descriptive medians, no confidence-certified performance win.
+  Eight finite simultaneous offers measure a burst, not sustained capacity.
+  No source-correct SLO goodput, physical committed-token, fullHF, production
+  graph or vLLM performance/parity claim follows.
+
+ROOT sequence (not executed):
+
+1. Complete/review current14worker/12HTTP source-frozen gate. Run the four pure
+   `test_contracts.py` and four MOCK-only `test_owner_control.py` fixtures, then
+   actual harmless-owner TERM/INT-before-Popen/during-readiness/during-
+   request/during-cleanup failure-injection fixtures. Prior shadow/root evidence
+   is not a test of this new adapter. Review all new source and bindings.
+2. Set `ZG_BENCH_HOST_ROOT` to the original repository. Run
+   `prepare_native_ab.py --hardware-protocol ABS_ACCEPTED_V8_PROTOCOL
+   --hardware-protocol-sha256 SHA --accepted-serving-sha256 SHA
+   --new-output ABS_THIS_RUN/root-plan-1`. That performs ROOT-only source scans,
+   exact launch normalization and completed gate checks; no engine starts.
+3. Under the original native-gpu user/network isolation, invoke Python with the
+   original reviewed private-loopback helper (which execs Python), then
+   `run_native_ab.py --run --plan ABS/root-plan-1/plan.json
+   --plan-sha256 SHA --new-output ABS_THIS_RUN/root-native-1`.
+4. `validate_native_ab.py --output ABS_THIS_RUN/root-native-1`. Independent ROOT
+   review of full raw results/actual owner/point-memory/ITL coverage is required.
+   Profiling/hardware counts and unprofiled timing remain separate runs.
--- /dev/null
+++ TODO.md
@@ -0,0 +1,16 @@
+# Native before/after preparation
+
+Parent: serving-and-speculation; owner: batch_design_review.
+Owned: this new ignored derivative only; V8/old baseline/current456 unchanged.
+
+- [x] Explicit native/native identities, exact existing launch/workload/caps.
+- [x] Same canonical benchmark, complete32-sample/144-offer/32-warmup accounting.
+- [x] Require completed actual candidate worker14/HTTP12 + prior source gates.
+- [x] Prepare original owner cleanup +record-only TERM/INT checkpoints.
+- [x] Full exact output/counter/band/ITL coverage and independent validation.
+- [ ] ROOT/peer source review; actual pure/ownership failure fixtures.
+- [ ] ROOT actual preparation +complete unprofiled native/native run+validation.
+- [ ] Independent physical kernel diagnostic and measured performance acceptance.
+
+Source-only AST and small source/metadata hashes by author. No tests, imports,
+builds, native/provider/model/GPU execution or payload hashes/results.
--- /dev/null
+++ bindings.json
@@ -0,0 +1,170 @@
+{
+  "binaries_models_DSO_payloads_hashed_by_author": false,
+  "canonical_prompt_token_ids": [
+    248045,
+    846,
+    198,
+    22791,
+    411,
+    1103,
+    314,
+    24959,
+    11,
+    18101,
+    539,
+    73982,
+    11,
+    364,
+    1599,
+    777,
+    3665,
+    25,
+    220,
+    16,
+    11,
+    220,
+    17,
+    11,
+    220,
+    18,
+    11,
+    220,
+    19,
+    11,
+    220,
+    20,
+    11,
+    220,
+    21,
+    11,
+    220,
+    22,
+    11,
+    220,
+    23,
+    11,
+    248046,
+    198,
+    248045,
+    74455,
+    198
+  ],
+  "completed_prerequisites": [
+    {
+      "bytes": 86320,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-composed-source-freeze-1.json",
+      "sha256": "c516760006fcefb85f1a2374c101a0eece8893b87e8d0a3da32b60967fe82d68"
+    },
+    {
+      "bytes": 3898,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-serving-build-1.json",
+      "sha256": "0412830ff345ea9b7bc5e3847c0205441929ba94202ad8db4d813c309aece2c8"
+    },
+    {
+      "bytes": 49679,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-full-model-matrix-1.json",
+      "sha256": "f32eb4f473673ab6375b6a24cb4b7d02da221b8cd59c03ff4511b48746c644ac"
+    },
+    {
+      "bytes": 13270,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-operator-matrix-1.json",
+      "sha256": "630cab9c32c2fc22a8e23ee1b000eada9016222bde8a0a04aa2b333b9572fc4f"
+    },
+    {
+      "bytes": 1015,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-full-check-result-1.json",
+      "sha256": "780af4da6d5e18950a422de3b1677c4076adddd5ec2c3ea95f67215cc9481a1a"
+    },
+    {
+      "bytes": 1755,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/accepted-native-baseline.json",
+      "sha256": "26553886c58b204b4807c954fd157d272e7f59df09e05f7142e0de9d3d93c3ec"
+    },
+    {
+      "bytes": 986,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-tokenizer-08b-solo-1.json",
+      "sha256": "a90cb27c8e7a5a40e4aa9eb914644eec8784d9566b3ad74a36b655abfd6b48eb"
+    },
+    {
+      "bytes": 494,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-tokenizer-2b-solo-v8-1.json",
+      "sha256": "5bf9c727ec018f66c52713d09e4cda7da2895a896655235583fe185fe7e4c334"
+    },
+    {
+      "bytes": 838,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-parity-root-2026-10-19/graph-rope-final-http-server-artifacts.json",
+      "sha256": "0ca8aa6c01915e3bcd493cf8677662a6bb79e65eed94da38ed4d882793321756"
+    },
+    {
+      "bytes": 1018,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-parity-root-2026-10-19/graph-rope-composed-final-serving-result.json",
+      "sha256": "e76248e80361efe53d935074cb09f421c093713102ed72db58ba66b2ebc83fcc"
+    },
+    {
+      "bytes": 5014,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-parity-root-2026-10-19/graph-rope-attention-fixed-main-source.json",
+      "sha256": "d89cd668fee704c1f9ae7d9a36dfc4b2e899e2d73bf43139f653c994f27969cc"
+    }
+  ],
+  "executables": {
+    "native_baseline": {
+      "bytes": 36398264,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-parity-root-2026-10-19/graph-rope-final-frozen-server",
+      "sha256": "4a7455277b89aa48e92bd9d30b15a939935017368918c0e304a4f699687aae3f",
+      "source_id": "ROOT-source-freeze:d89cd668fee704c1f9ae7d9a36dfc4b2e899e2d73bf43139f653c994f27969cc"
+    },
+    "native_candidate": {
+      "bytes": 36566200,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-frozen-server-1",
+      "sha256": "87e1d727f908368f508eeb1e4dd53456f14348b8c72c4519cbee5e52b977c61e",
+      "source_id": "ROOT-source-freeze:c516760006fcefb85f1a2374c101a0eece8893b87e8d0a3da32b60967fe82d68"
+    }
+  },
+  "full_gate": {
+    "bytes": 1015,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-full-check-result-1.json",
+    "sha256": "780af4da6d5e18950a422de3b1677c4076adddd5ec2c3ea95f67215cc9481a1a"
+  },
+  "full_model_matrix": {
+    "bytes": 49679,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-full-model-matrix-1.json",
+    "sha256": "f32eb4f473673ab6375b6a24cb4b7d02da221b8cd59c03ff4511b48746c644ac"
+  },
+  "future_serving_gate_path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-serving-physical-2.json",
+  "future_serving_gate_sha256": "ROOT_MUST_SUPPLY_ACTUAL_COMPLETED_14_WORKER_12_HTTP_RECEIPT_SHA",
+  "models": {
+    "0.8b": {
+      "source_sha256": "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
+      "tokenizer_receipt": {
+        "bytes": 986,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-tokenizer-08b-solo-1.json",
+        "sha256": "a90cb27c8e7a5a40e4aa9eb914644eec8784d9566b3ad74a36b655abfd6b48eb"
+      }
+    },
+    "2b": {
+      "source_sha256": "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6",
+      "tokenizer_receipt": {
+        "bytes": 494,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-tokenizer-2b-solo-v8-1.json",
+        "sha256": "5bf9c727ec018f66c52713d09e4cda7da2895a896655235583fe185fe7e4c334"
+      }
+    }
+  },
+  "operator_matrix": {
+    "bytes": 13270,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-operator-matrix-1.json",
+    "sha256": "630cab9c32c2fc22a8e23ee1b000eada9016222bde8a0a04aa2b333b9572fc4f"
+  },
+  "schema": "zerogravity.native-before-after.antecedents.v1",
+  "serving_build": {
+    "bytes": 3898,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-serving-build-1.json",
+    "sha256": "0412830ff345ea9b7bc5e3847c0205441929ba94202ad8db4d813c309aece2c8"
+  },
+  "source_freeze": {
+    "bytes": 86320,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-composed-source-freeze-1.json",
+    "sha256": "c516760006fcefb85f1a2374c101a0eece8893b87e8d0a3da32b60967fe82d68"
+  },
+  "tokenizer_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
+}
--- /dev/null
+++ native_ab_common.py
@@ -0,0 +1,284 @@
+"""ROOT-only native/native contracts; no engine, operator or child on import."""
+import hashlib
+import importlib.util
+import json
+import math
+import os
+from pathlib import Path
+import stat
+import sys
+import urllib.request
+
+HERE = Path(__file__).resolve().parent
+ROOT = HERE.parents[2]
+PARENT = ROOT / '.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020'
+LABELS = ('native_baseline', 'native_candidate')
+MODELS = ('0.8b', '2b')
+BANDS = (('solo', 1), ('wave', 8))
+ORDER = (('native_baseline', 'native_candidate', 'native_candidate', 'native_baseline'),
+         ('native_candidate', 'native_baseline', 'native_baseline', 'native_candidate'))
+BASELINE = '4a7455277b89aa48e92bd9d30b15a939935017368918c0e304a4f699687aae3f'
+CANDIDATE = '87e1d727f908368f508eeb1e4dd53456f14348b8c72c4519cbee5e52b977c61e'
+V8 = 'b0c50e8a9d8d647464d3837748db0ba958a7c8294ee18140474703068fca0b33'
+SOURCE = 'c516760006fcefb85f1a2374c101a0eece8893b87e8d0a3da32b60967fe82d68'
+COUNTERS = ('completed_forward_kernel_dispatches', 'completed_model_token_steps',
+    'committed_gpu_roots', 'native_projection_dispatches_completed',
+    'native_sampler_dispatches_completed', 'accepted_output_tokens',
+    'gpu_fused_state_only_batches', 'gpu_fused_state_only_rows',
+    'gpu_fused_greedy_batches', 'gpu_fused_rows_completed', 'gpu_fused_batches',
+    'native_admission_refusals', 'discarded_ready_cohorts', 'failed_gpu_quanta',
+    'cancelled_requests')
+ZERO_COUNTERS = ('native_admission_refusals', 'discarded_ready_cohorts',
+                 'failed_gpu_quanta', 'cancelled_requests')
+MAX_TREE_BYTES = 512 * 1024**2
+MAX_TREE_FILES = 1024
+
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+
+def sha(path):
+    with Path(path).open('rb') as source:
+        return hashlib.file_digest(source, 'sha256').hexdigest()
+
+
+def read(path, cap=8 * 1024**2):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as source:
+        before = os.fstat(source.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular input')
+        raw = source.read(cap + 1)
+        after = os.fstat(source.fileno())
+    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
+    require(identity(before) == identity(after) and len(raw) == before.st_size, 'unchanged bounded input')
+    return raw
+
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON field')
+        result[key] = value
+    return result
+
+
+def load(path):
+    return json.loads(read(path), object_pairs_hook=unique)
+
+
+def pin(row):
+    raw = read(row['path'])
+    require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
+            'exact source/metadata pin: ' + row['path'])
+
+
+def file_row(path):
+    path = Path(path)
+    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))
+
+
+def tree(root):
+    total, files = 0, 0
+    for path in Path(root).rglob('*'):
+        require(not path.is_symlink(), 'no artifact symlinks')
+        if path.is_file():
+            total += path.stat().st_size
+            files += 1
+    require(total <= MAX_TREE_BYTES and files <= MAX_TREE_FILES, 'bounded complete artifact tree')
+    return dict(bytes=total, files=files, limit_bytes=MAX_TREE_BYTES, limit_files=MAX_TREE_FILES)
+
+
+def save(path, value, root=None):
+    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
+    require(len(raw) <= 8 * 1024**2, 'bounded receipt')
+    path = Path(path)
+    temporary = path.with_name(path.name + '.writing')
+    with temporary.open('xb') as output:
+        output.write(raw)
+    temporary.replace(path)
+    if root is not None:
+        tree(root)  # Includes THIS final receipt; no pre-write-only artifact claim.
+
+
+def install_opener():
+    class NoRedirect(urllib.request.HTTPRedirectHandler):
+        def redirect_request(self, *_args, **_kwargs):
+            raise ValueError('native redirects refused')
+    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()))
+
+
+def http_json(url):
+    with urllib.request.urlopen(url, timeout=2) as response:
+        raw = response.read(1024**2 + 1)
+        require(response.status == 200 and len(raw) <= 1024**2, 'bounded original native HTTP response')
+    return json.loads(raw, object_pairs_hook=unique)
+
+
+def source_freeze():
+    document = load(HERE / 'freeze-manifest.json')
+    for row in document['files']:
+        pin(dict(row, path=str(HERE / row['file'])))
+        pin(dict(row, path=str(HERE / 'frozen' / row['file'])))
+    for row in document['parent_sources_and_metadata']:
+        pin(row)
+    return sha(HERE / 'freeze-manifest.json')
+
+
+def load_path(name, path):
+    require(name not in sys.modules, 'exclusive helper module name')
+    spec = importlib.util.spec_from_file_location(name, path)
+    module = importlib.util.module_from_spec(spec)
+    sys.modules[name] = module
+    spec.loader.exec_module(module)
+    return module
+
+
+def parents():
+    # Exact immutable V8 modules are reused, never rewritten or relabeled as a
+    # second implementation. Native result labels belong only to this driver.
+    source_freeze()
+    require(str(ROOT.resolve()) == os.environ.get('ZG_BENCH_HOST_ROOT'), 'explicit original host repository')
+    sys.path.insert(0, str(PARENT))
+    import protocol_v8 as protocol
+    import prepare_launch_v8 as launch
+    import owned_container_v4 as owned
+    import observations_v8 as observations
+    import cgroup_readback_v3 as cgroup
+    for module in (protocol, launch, owned, observations, cgroup):
+        require(Path(module.__file__).resolve().parent == PARENT, 'exact parent module namespace')
+    pins = protocol.verify_preparation(V8)
+    protocol.verify_small_sources()
+    bench = load_path('native_ab_canonical_bench', ROOT / 'tools/serve-bench.py')
+    lifecycle = load_path('native_ab_canonical_log', ROOT / 'tools/serve-lifecycle.py')
+    return protocol, launch, owned, observations, cgroup, pins, bench, lifecycle
+
+
+def closed_case(row):
+    c, log = row['foreground_cleanup'], row['log']
+    require(row['status'].startswith('PASS_') and row['exit_code'] == 0 and row['cleanup_errors'] == [] and 'failure' not in row and
+            c['exit_code'] == 0 and c['stage_errors'] == [] and
+            all(c[k] is True for k in ('leader_reaped', 'group_absent', 'all_nonzero_signals_before_reap')) and
+            log['pipe_eof'] is True and log['closed'] is True, 'actual original owner retirement')
+
+
+def accepted_candidate(serving_sha):
+    binding = load(HERE / 'bindings.json')
+    for row in binding['completed_prerequisites']:
+        pin(row)
+    freeze_path = Path(binding['source_freeze']['path'])
+    pin(binding['source_freeze'])
+    source = load(freeze_path)
+    require(len(source['files']) == 456, 'complete candidate source closure')
+    for row in source['files']:
+        pin(dict(row, path=str(ROOT / row['file'])))
+    for name, count in (('operator_matrix', 10), ('full_model_matrix', 46)):
+        value = load(binding[name]['path'])
+        require(value['source_freeze_sha256'] == SOURCE and value['case_count'] == len(value['cases']) == count
+                and len({r['case'] for r in value['cases']}) == count and value['status'].startswith('PASS_'),
+                'complete candidate numerical matrix')
+        for row in value['cases']:
+            closed_case(row)
+    full = load(binding['full_gate']['path'])
+    require(full['status'] == 'PASS_FULL_ISOLATED_AND_BOTH_APPROVED_ORIGINAL_ROUTES' and
+            full['source_freeze_sha256'] == SOURCE and full['all456_sources_post_gate_exact'] is True,
+            'candidate full gate and both original routes')
+    build = load(binding['serving_build']['path'])
+    require(build['source_freeze_sha256'] == SOURCE and all(s['exit_code'] == 0 for s in build['steps']) and
+            build['artifacts']['server']['sha256'] == CANDIDATE, 'candidate strict/frozen release build')
+    gate_path = Path(binding['future_serving_gate_path'])
+    require(len(serving_sha) == 64 and sha(gate_path) == serving_sha, 'ROOT binds completed worker/HTTP receipt')
+    gate = load(gate_path)
+    require(gate['status'] == 'PASS_FROZEN_WORKER14_HTTP12_REQUIRES_ROOT_REVIEW' and
+            gate['source_freeze_sha256'] == SOURCE and len(gate['cases']) == 2 and
+            gate['artifacts']['server'] == CANDIDATE and gate['performance_benchmark'] is False,
+            'pending/partial serving gate is refused')
+    for row, label, count in zip(gate['cases'], ('worker', 'http'), (14, 12)):
+        closed_case(row)
+        require(row['case'] == label and row['expected_tests'] == count and
+                row['status'] == 'PASS_ORIGINAL_PHYSICAL_SERVING_REGRESSION', 'all worker14/HTTP12 cases')
+        log = gate_path.parent / f'root-serving-{label}-physical-2.log'
+        raw = read(log)
+        require(len(raw) == row['log']['bytes'] and hashlib.sha256(raw).hexdigest() == row['log']['sha256'] and
+                f'{count} passed; 0 failed; 0 ignored' in raw.decode(), 'actual complete serving test summary')
+    return dict(serving_gate=file_row(gate_path), source_freeze=binding['source_freeze'],
+                completed_prerequisites=binding['completed_prerequisites'])
+
+
+def counters(before, after):
+    values = []
+    for value in (before, after):
+        a = value['admission']
+        require(a['kind'] == 'resident_gpu' and a['batch_mode'] == 'fused' and
+                a['native_batch_rows_max'] == 8 and a['active_sessions'] == a['queued_requests'] == 0,
+                'same native engine/band and quiescent endpoints')
+        row = {name: a[name] for name in COUNTERS}
+        row.update({'band' + key: a['gpu_fused_batches_by_rows'][key] for key in ('2', '4', '8')})
+        require(all(type(v) is int and v >= 0 for v in row.values()), 'actual native integer counters, no absent defaults')
+        values.append(row)
+    left, right = values
+    require(all(right[k] >= left[k] for k in left), 'monotonic same-process counters')
+    delta = {key: right[key] - left[key] for key in left}
+    require(all(delta[k] == 0 for k in ZERO_COUNTERS), 'failed/refused/discarded/cancelled work is not accepted')
+    require(delta['gpu_fused_batches'] == sum(delta['band' + k] for k in ('2', '4', '8')) ==
+            delta['gpu_fused_state_only_batches'] + delta['gpu_fused_greedy_batches'] and
+            delta['gpu_fused_rows_completed'] == sum(int(k) * delta['band' + k] for k in ('2', '4', '8')) and
+            delta['native_sampler_dispatches_completed'] == 4 * delta['gpu_fused_greedy_batches'],
+            'actual native phase/band/row/sampler conservation')
+    require(2 * delta['gpu_fused_state_only_batches'] <= delta['gpu_fused_state_only_rows'] <=
+            8 * delta['gpu_fused_state_only_batches'] and
+            2 * delta['gpu_fused_greedy_batches'] <= delta['gpu_fused_rows_completed'] -
+            delta['gpu_fused_state_only_rows'] <= 8 * delta['gpu_fused_greedy_batches'],
+            'native phase rows stay within actual supported bands')
+    require(delta['completed_forward_kernel_dispatches'] > 0 and delta['completed_model_token_steps'] > 0 and
+            delta['committed_gpu_roots'] > 0, 'genuine complete GPU workload, not counter presence')
+    return delta
+
+
+def signature(sample, expected):
+    require(sample['offered_requests'] == sample['completed_requests'] == len(expected) and
+            sample['unsuccessful_requests'] == sample['completions_without_usage'] == 0 and
+            len(sample['requests']) == len(expected), 'all canonical offers completed with usage')
+    rows = {}
+    for row in sample['requests']:
+        require(row['request_id'] not in rows, 'unique complete workload membership')
+        original = next((x for x in expected if x['request_id'] == row['request_id']), None)
+        require(original is not None and row['prompt_sha256'] == hashlib.sha256(original['prompt'].encode()).hexdigest()
+                and row['max_tokens'] == original['max_tokens'] and row['completed'] is True,
+                'original prompt/cap/request identity')
+        obs = row['wire_token_observation']
+        require(obs['status'] == 'provider_bound_complete_ids' and type(row['usage_reported_completion_tokens']) is int
+                and len(obs['token_ids']) == row['usage_reported_completion_tokens'] <= row['max_tokens']
+                and all(type(token) is int and 0 <= token < 248320 for token in obs['token_ids'])
+                and row['finish_reason'] in ('stop', 'length') and type(row['output_sha256']) is str and
+                len(row['output_sha256']) == 64 and set(row['output_sha256']) <= set('0123456789abcdef'),
+                'full response-bound IDs/usage/finish/text hash')
+        rows[row['request_id']] = {key: row[key] for key in ('prompt_sha256', 'max_tokens',
+            'usage_reported_completion_tokens', 'finish_reason', 'output_sha256')}
+        rows[row['request_id']]['token_ids'] = obs['token_ids']
+    return rows
+
+
+def finite(value):
+    return type(value) in (int, float) and math.isfinite(value) and value >= 0
+
+
+def model_sources(pins):
+    # ROOT-only complete scans, outside every timed sample; same V8 source path.
+    from support_v3 import source_check
+    for model in MODELS:
+        row = pins['models'][model]
+        source_check(pins, model, row['source_path'], row['gguf']['path'])
+
+
+def validate_context(value, parent):
+    require(value['schema'] == 'zerogravity.native-before-after.launch-context.v1' and
+            (value['workload_kind'], value['concurrencies']) in (('solo', [1]), ('wave', [8])),
+            'actual native/native C1 or C8 population, no inherited C2/C4 claim')
+    # Parent's fixed multirow comparison population is irrelevant to launch().
+    # Reuse its unchanged policy/resource/whole-workload validation only, with
+    # its own comparison-population fields in a temporary validation object.
+    checked = dict(value, schema=parent.SCHEMA,
+                   concurrencies=[1] if value['workload_kind'] == 'solo' else [2, 4, 8])
+    parent.validate_protocol(checked)
+    return value
--- /dev/null
+++ native_ab_owner.py
@@ -0,0 +1,133 @@
+"""One original native container/foreground group; no engine or alternate queue."""
+import os
+import secrets
+import subprocess
+import time
+
+from native_ab_common import http_json, require, save
+
+
+class NativeOwner:
+    def __init__(self, label, directory, command, protocol, dependencies, cancellation, artifact_root):
+        self.label, self.directory, self.command, self.protocol = label, directory, command, protocol
+        _, _, self.owned, self.observations, self.cgroup, _, self.bench, self.lifecycle = dependencies
+        self.cancel, self.artifact_root = cancellation, artifact_root
+        self.nonce = secrets.token_hex(16)
+        self.name = 'zg-native-ab-' + label.replace('_', '-') + '-' + self.nonce
+        self.cidfile = directory / 'cid'
+        self.process = self.log = self.monitor = None
+        self.directory.mkdir(mode=0o700)
+        self.receipt = dict(label=label, native_implementation='zerogravity',
+            status='UNQUALIFIED', owned_name=self.name, owned_nonce=self.nonce,
+            executable_sha256=protocol['zerogravity_binary_sha256'], cleanup_errors=[],
+            policy='warmed_process', primary_failure=None, profiler_enabled=False)
+        self.closed = False
+        self.save()
+
+    def save(self):
+        save(self.directory / 'receipt.json', self.receipt, self.artifact_root)
+
+    def live(self):
+        self.cancel.check()
+        require(self.process is not None and self.owned.leader_status_unreaped(self.process) is None,
+                'original owned foreground remains live and unreaped')
+        require(self.log is None or self.log.error is None, 'bounded native log remains operational')
+
+    def start(self):
+        address = ('127.0.0.1', self.protocol['ports'][0])
+        require(not self.lifecycle.listening(address), 'fresh native endpoint; never adopt an existing server')
+        self.receipt['GPU_quiet_before'] = self.observations.quiet(
+            self.protocol['gpu_uuid'], self.protocol['gpu_total_mib'])
+        replacements = {'{owned_name}': self.name, '{owned_nonce}': self.nonce,
+                        '{cidfile}': str(self.cidfile), '{sample_directory}': str(self.directory)}
+        command = self.command
+        for key, value in replacements.items():
+            command = [part.replace(key, value) for part in command]
+        self.receipt['argv'] = command
+        # Record-only TERM/INT handler cannot raise across this ownership handoff.
+        self.cancel.check()
+        self.process = subprocess.Popen(command, cwd=str(self.protocol_root()),
+            env={**os.environ, 'ZG_EXPECTED_GPU_UUID': self.protocol['gpu_uuid']},
+            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
+            start_new_session=True, bufsize=0)
+        self.receipt['owned_cli_pid'] = self.process.pid
+        self.log = self.lifecycle.BoundedLog(self.process.stdout, self.directory / 'execution.log', 8 * 1024**2)
+        self.monitor = self.observations.Monitor(self.protocol['gpu_uuid'], self.name, self.nonce, self.cidfile)
+        self.save()
+        deadline = time.monotonic() + 600
+        while time.monotonic() < deadline:
+            self.live()
+            try:
+                models = http_json(self.endpoint() + '/v1/models')
+                require(self.alias() in [row['id'] for row in models['data']], 'actual original native model alias')
+            except Exception as error:
+                self.receipt['last_readiness_error'] = repr(error)[:512]
+                time.sleep(.1)
+                continue
+            self.live()
+            require(self.owned.owned_container(self.name, self.nonce, self.cidfile) is not None,
+                    'original authenticated nonce/CID/container before readiness')
+            from native_ab_common import load
+            bootstrap = load(self.directory / 'bootstrap.json')
+            require(bootstrap['provider'] == 'zerogravity' and bootstrap['model'] == self.protocol['model'] and
+                    bootstrap['stock_math_modified'] is False and bootstrap['policy_observer'] is False,
+                    'genuine native bootstrap, not a vLLM slot')
+            self.cgroup.validate_caps(bootstrap['cgroup_readback'], self.protocol['cpu_affinity'])
+            self.receipt['actual_bootstrap'] = bootstrap
+            self.receipt['ready'] = True
+            self.save()
+            return
+        raise TimeoutError('finite original native startup; no fallback')
+
+    def protocol_root(self):
+        from native_ab_common import ROOT
+        return ROOT
+
+    def endpoint(self):
+        return 'http://127.0.0.1:' + str(self.protocol['ports'][0])
+
+    def alias(self):
+        return 'qwen35-' + self.protocol['model'] + '-same-gguf'
+
+    def close(self):
+        if self.closed:
+            require(self.receipt['status'] == 'OWNERS_CLOSED', 'prior native cleanup failed')
+            return
+        self.closed = True
+        errors = self.receipt['cleanup_errors']
+        if self.monitor is not None:
+            try:
+                self.receipt['physical_observations'] = self.monitor.finish()
+            except BaseException as error:
+                errors.append('physical observer: ' + repr(error))
+        if self.process is not None:
+            try:
+                errors += self.owned.cleanup(self.process, self.name, self.nonce, self.cidfile, self.receipt)
+            except BaseException as error:
+                errors.append('original container/group cleanup: ' + repr(error))
+        if self.log is not None:
+            try:
+                result = self.log.finish(5)
+                self.receipt['log_cleanup'] = result
+                require(result['reader_complete'] and result['reader_error'] is None and
+                        result['truncated'] is False and result['bytes_seen'] <= 8 * 1024**2,
+                        'actual complete bounded log and pipe EOF')
+            except BaseException as error:
+                errors.append('log cleanup: ' + repr(error))
+        elif self.process is not None and self.process.stdout is not None:
+            try:
+                self.process.stdout.close()
+            except BaseException as error:
+                errors.append('pipe close: ' + repr(error))
+        if self.process is not None:
+            try:
+                ceiling = self.receipt['GPU_quiet_before'][-1]['memory_used_mib'] + 64
+                self.receipt['GPU_quiet_after'] = self.observations.quiet(
+                    self.protocol['gpu_uuid'], self.protocol['gpu_total_mib'], ceiling)
+            except BaseException as error:
+                errors.append('GPU retirement: ' + repr(error))
+        if not errors:
+            self.receipt['status'] = 'OWNERS_CLOSED'
+        # This I/O cannot skip any preceding independent cleanup stage.
+        self.save()
+        require(not errors, 'native cleanup/evidence failed; no next sample')
--- /dev/null
+++ prepare_native_ab.py
@@ -0,0 +1,98 @@
+"""ROOT ONLY: new native/native plan; does not start a provider."""
+import argparse
+import copy
+import os
+from pathlib import Path
+
+from native_ab_common import (HERE, LABELS, MODELS, BANDS, BASELINE, CANDIDATE,
+    V8, SOURCE, accepted_candidate, file_row, load, parents, require, save, sha, source_freeze, validate_context)
+
+
+def protocols(reference, bindings, protocol):
+    result = []
+    for model in MODELS:
+        for workload, concurrency in BANDS:
+            pair = {}
+            for label in LABELS:
+                value = copy.deepcopy(reference)
+                value.update(schema='zerogravity.native-before-after.launch-context.v1',
+                    model=model, route='eager', policy='warmed_process',
+                    workload_kind=workload, max_tokens=8, concurrencies=[concurrency],
+                    requests=protocol.workload(workload, 8), max_pending=1 if workload == 'solo' else 8,
+                    source_sha256=bindings['models'][model]['source_sha256'],
+                    tokenizer_sha256=bindings['tokenizer_sha256'],
+                    zerogravity_binary=bindings['executables'][label]['path'],
+                    zerogravity_binary_sha256=bindings['executables'][label]['sha256'],
+                    zerogravity_source_id=bindings['executables'][label]['source_id'])
+                validate_context(value, protocol)
+                pair[label] = value
+            result.append(dict(model=model, workload=workload, concurrency=concurrency, protocols=pair))
+    return result
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--hardware-protocol', type=Path, required=True)
+    parser.add_argument('--hardware-protocol-sha256', required=True)
+    parser.add_argument('--accepted-serving-sha256', required=True)
+    parser.add_argument('--new-output', type=Path, required=True)
+    args = parser.parse_args()
+    freeze = source_freeze()
+    protocol, launcher, _, _, _, pins, _, _ = parents()
+    candidate = accepted_candidate(args.accepted_serving_sha256)
+    require(sha(args.hardware_protocol) == args.hardware_protocol_sha256, 'ROOT binds actual hardware protocol')
+    reference = protocol.validate_protocol(load(args.hardware_protocol))
+    require(reference['zerogravity_binary_sha256'] == BASELINE and reference['route'] == 'eager' and
+            set(reference['cpu_affinity']) <= os.sched_getaffinity(0), 'accepted native baseline hardware and exact four CPUs')
+    binding = load(HERE / 'bindings.json')
+    for label, digest in zip(LABELS, (BASELINE, CANDIDATE)):
+        row = binding['executables'][label]
+        require(row['sha256'] == digest and Path(row['path']).stat().st_size == row['bytes'] and
+                sha(row['path']) == digest, 'ROOT verifies actual frozen native executable')
+    from support_v3 import source_check
+    projects = protocol.read_json(protocol.HERE / 'projector-pins.json')['models']
+    for model in MODELS:
+        source = pins['models'][model]
+        require(source['gguf']['sha256'] == binding['models'][model]['source_sha256'], 'original source weights')
+        source_check(pins, model, source['source_path'], source['gguf']['path'])
+        require(sha(Path(projects[model]['conversion_directory']) / 'mmproj-bf16.gguf') ==
+                projects[model]['projector_sha256'], 'unchanged existing V8 readonly mount')
+        for name in ('config.json', 'tokenizer.json'):
+            expected = next(r['sha256'] for r in source['source_files'] if r['name'] == name)
+            require(sha(protocol.host_root() / 'target/models' / ('qwen35-' + model) / name) == expected,
+                    'same original native sidecar')
+        tokenizer = load(binding['models'][model]['tokenizer_receipt']['path'])
+        require(tokenizer['tokenizer_sha256'] == pins['tokenizer_sha256'] and tokenizer['eos_token_id'] == 248046 and
+                len(tokenizer['records']) == 1 and tokenizer['records'][0]['prompt_token_ids'] ==
+                binding['canonical_prompt_token_ids'] and len(binding['canonical_prompt_token_ids']) == 47,
+                'actual original canonical47 IDs; no new rendering or tokenizer math')
+    cases = protocols(reference, binding, protocol)
+    for case in cases:
+        case['commands'] = {}
+        for label, value in case['protocols'].items():
+            case['commands'][label] = launcher.launch(value, pins, V8, 'zerogravity',
+                Path(projects[case['model']]['conversion_directory']))
+        def normalized(label):
+            executable = binding['executables'][label]
+            return [part.replace(executable['path'], '<NATIVE_EXECUTABLE>')
+                    .replace(executable['sha256'], '<NATIVE_EXECUTABLE_SHA256>')
+                    for part in case['commands'][label]]
+        require(normalized(LABELS[0]) == normalized(LABELS[1]), 'same V8 launch except native executable and hash')
+    output = args.new_output.resolve()
+    require(output.parent == HERE and not output.exists(), 'fresh own plan only')
+    output.mkdir(mode=0o700)
+    plan = dict(schema='zerogravity.native-before-after.plan.v1', source_freeze_sha256=freeze,
+        parent_v8_sha256=V8, candidate_gate=candidate, hardware_reference=file_row(args.hardware_protocol),
+        labels=list(LABELS), policy='warmed_process', route='eager', max_tokens=8,
+        models=list(MODELS), cases=cases, repetitions=2, measured_samples=32,
+        measured_offers=144, warmup_requests=32, profile='same accepted native GGUF SourceV1',
+        canonical_prompt_token_ids=binding['canonical_prompt_token_ids'],
+        native_source_freeze_sha256=SOURCE, profiler_enabled=False, vllm_provider_present=False,
+        performance_or_source_correct_goodput_qualified=False)
+    save(output / 'plan.json', plan, output)
+    require(source_freeze() == freeze, 'source preparation immutable')
+    print(sha(output / 'plan.json'))
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ run_native_ab.py
@@ -0,0 +1,214 @@
+"""ROOT ONLY explicit native baseline/candidate; never a vLLM comparison slot."""
+import argparse
+from dataclasses import asdict
+import fcntl
+import os
+from pathlib import Path
+import socket
+import struct
+import time
+
+from native_ab_common import (HERE, LABELS, MODELS, BANDS, ORDER, V8, accepted_candidate,
+    counters, file_row, http_json, install_opener, load, model_sources, parents, require, save, sha,
+    signature, source_freeze, tree, validate_context)
+from native_ab_owner import NativeOwner
+from sigterm_cancellation import SigtermCancellation
+
+
+def private_namespace():
+    mapping = Path('/proc/self/uid_map').read_text().split()
+    require(len(mapping) == 3 and mapping[0] == '0' and mapping[1] != '0' and mapping[2] == '1',
+            'original native-gpu private mapped user namespace required')
+    network = os.open('/proc/self/ns/net', os.O_RDONLY)
+    try:
+        owner = fcntl.ioctl(network, 0xB701)
+        try:
+            require(os.fstat(owner).st_ino == os.stat('/proc/self/ns/user').st_ino,
+                    'network namespace belongs to this private user namespace')
+        finally:
+            os.close(owner)
+    finally:
+        os.close(network)
+    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
+        flags = fcntl.ioctl(control, 0x8913, struct.pack('16sH22x', b'lo', 0))
+        require(struct.unpack_from('H', flags, 16)[0] & 1, 'use original reviewed private-loopback helper first')
+    return dict(uid_map=Path('/proc/self/uid_map').read_text(),
+        user_namespace=os.readlink('/proc/self/ns/user'), network_namespace=os.readlink('/proc/self/ns/net'))
+
+
+def quiescent(owner):
+    deadline, previous = time.monotonic() + 15, None
+    while time.monotonic() < deadline:
+        owner.live()
+        value = http_json(owner.endpoint() + '/health')
+        admission = value['admission']
+        if admission['active_sessions'] == admission['queued_requests'] == 0:
+            if value == previous:
+                return value
+            previous = value
+        else:
+            previous = None
+        time.sleep(.1)
+    raise TimeoutError('original GPU worker failed bounded quiescence')
+
+
+def validate_plan(plan, protocol, launcher, pins, binding):
+    require(plan['schema'] == 'zerogravity.native-before-after.plan.v1' and
+            plan['source_freeze_sha256'] == source_freeze() and plan['parent_v8_sha256'] == V8 and
+            plan['labels'] == list(LABELS) and plan['policy'] == 'warmed_process' and
+            plan['route'] == 'eager' and plan['max_tokens'] == 8 and plan['models'] == list(MODELS) and
+            plan['repetitions'] == 2 and plan['measured_samples'] == 32 and
+            plan['measured_offers'] == 144 and plan['warmup_requests'] == 32 and
+            plan['profiler_enabled'] is False and plan['vllm_provider_present'] is False and
+            plan['canonical_prompt_token_ids'] == binding['canonical_prompt_token_ids'] and
+            plan['performance_or_source_correct_goodput_qualified'] is False,
+            'complete declared native/native plan')
+    require(len(plan['cases']) == 4 and [(c['model'], c['workload'], c['concurrency']) for c in plan['cases']] ==
+            [(model, workload, concurrency) for model in MODELS for workload, concurrency in BANDS],
+            'both models and C1/C8 exactly once')
+    projects = protocol.read_json(protocol.HERE / 'projector-pins.json')['models']
+    for case in plan['cases']:
+        require(set(case['commands']) == set(case['protocols']) == set(LABELS), 'no foreign provider slots')
+        for label in LABELS:
+            p = validate_context(case['protocols'][label], protocol)
+            e = binding['executables'][label]
+            require(p['model'] == case['model'] and p['workload_kind'] == case['workload'] and
+                    p['policy'] == 'warmed_process' and p['route'] == 'eager' and p['max_tokens'] == 8 and
+                    p['zerogravity_binary'] == e['path'] and p['zerogravity_binary_sha256'] == e['sha256'] and
+                    p['source_sha256'] == binding['models'][case['model']]['source_sha256'] and
+                    p['tokenizer_sha256'] == binding['tokenizer_sha256'] and
+                    case['commands'][label] == launcher.launch(p, pins, V8, 'zerogravity',
+                        Path(projects[case['model']]['conversion_directory'])),
+                    'exact source-reviewed native V8 launch, no flag/math substitution')
+        left, right = [dict(case['protocols'][label]) for label in LABELS]
+        for value in (left, right):
+            for key in ('zerogravity_binary', 'zerogravity_binary_sha256', 'zerogravity_source_id'):
+                value.pop(key)
+        require(left == right, 'only native executable/source identity differs within a pair')
+    return plan
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--plan', type=Path, required=True)
+    parser.add_argument('--plan-sha256', required=True)
+    parser.add_argument('--new-output', type=Path, required=True)
+    parser.add_argument('--run', action='store_true')
+    args = parser.parse_args()
+    require(args.run, 'ROOT must explicitly run after review and serving gates')
+    freeze = source_freeze()
+    require(sha(args.plan) == args.plan_sha256 and args.plan.resolve().parent.parent == HERE,
+            'own exclusive prepared plan and actual SHA')
+    deps = parents()
+    protocol, launcher, _, _, _, pins, bench, _ = deps
+    binding = load(HERE / 'bindings.json')
+    plan = validate_plan(load(args.plan), protocol, launcher, pins, binding)
+    gate_sha = plan['candidate_gate']['serving_gate']['sha256']
+    require(accepted_candidate(gate_sha) == plan['candidate_gate'], 'all accepted candidate gates unchanged')
+    for row in binding['executables'].values():
+        require(sha(row['path']) == row['sha256'], 'actual immutable native executable before run')
+    model_sources(pins)
+    output = args.new_output.resolve()
+    require(output.parent == HERE and not output.exists(), 'fresh whole native/native run')
+    output.mkdir(mode=0o700)
+    install_opener()
+    namespace = private_namespace()
+    allowed = os.sched_getaffinity(0)
+    cpus = plan['cases'][0]['protocols'][LABELS[0]]['cpu_affinity']
+    require(set(cpus) <= allowed, 'same four available logical CPUs')
+    os.sched_setaffinity(0, cpus)
+    report = dict(schema='zerogravity.native-before-after.run.v1', status='RUNNING_UNQUALIFIED',
+        plan=file_row(args.plan), source_freeze_sha256=freeze, samples=[],
+        namespace=namespace, native_only=True, main_timing_profiled=False,
+        performance_or_vllm_or_source_correct_goodput_qualified=False)
+    save(output / 'report.json', report, output)
+    try:
+        with SigtermCancellation() as cancellation:
+            exact_output_by_model = {}
+            for case in plan['cases']:
+                expected = protocol.workload(case['workload'], 8)
+                for repetition, order in enumerate(ORDER):
+                    for position, label in enumerate(order):
+                        cancellation.check()
+                        index = len(report['samples'])
+                        directory = output / f'{index:03d}-{case["model"]}-c{case["concurrency"]}-{label}'
+                        p = case['protocols'][label]
+                        owner = NativeOwner(label, directory, case['commands'][label], p, deps, cancellation, output)
+                        primary = None
+                        try:
+                            owner.start()
+                            owner.live()
+                            # Exactly ONE ordinary canonical request, no width-specific warmup.
+                            warm = bench.stream_one(owner.endpoint(), owner.alias(), expected[0]['prompt'], 8, 180,
+                                None, request_id='native-ab-warmup-' + str(index), return_token_ids=True)
+                            owner.receipt['warmup'] = asdict(warm)
+                            owner.save()
+                            require(warm.completed, 'ordinary exact canonical warmup completed')
+                            owner.live()
+                            before = quiescent(owner)
+                            measured = bench.bench(owner.endpoint(), owner.alias(), expected[0]['prompt'],
+                                len(expected), case['concurrency'], 8, 180, None, workload=expected,
+                                ttft_slo_seconds=10, itl_slo_seconds=2, max_pending=len(expected),
+                                launch_deadline_seconds=180, return_token_ids=True,
+                                snapshot_url=owner.endpoint() + '/health')
+                            save(directory / 'raw-measurement.json', measured, output)
+                            owner.live()
+                            after = quiescent(owner)
+                            save(directory / 'quiescent.json', dict(before=before, after=after), output)
+                            sig = signature(measured, expected)
+                            delta = counters(before, after)
+                            require(delta['accepted_output_tokens'] == sum(len(v['token_ids']) for v in sig.values()),
+                                    'native accepted packet output conservation; warmup excluded')
+                            if case['concurrency'] == 1:
+                                require(delta['gpu_fused_batches'] == delta['native_projection_dispatches_completed'] ==
+                                        delta['native_sampler_dispatches_completed'] == 0,
+                                        'single independent request has no padded native batch')
+                            measurement = dict(label=label, model=case['model'], workload=case['workload'],
+                                concurrency=case['concurrency'], repetition=repetition, order_position=position,
+                                paired_order='ABBA' if repetition == 0 else 'BAAB',
+                                benchmark=measured, quiescent_before=before, quiescent_after=after,
+                                source_counter_delta=delta, source_counter_unit='completed source launchsites, not CUDA activity',
+                                signature=sig)
+                            save(directory / 'measurement.json', measurement, output)
+                            owner.receipt['measurement'] = file_row(directory / 'measurement.json')
+                            owner.save()
+                            # Same source/prompt/policy: preserve first actual full result,
+                            # reject any later per-row/band/version discrepancy immediately.
+                            exact = exact_output_by_model.setdefault(case['model'], sig['request-0'])
+                            require(all(value == exact for value in sig.values()),
+                                    'native before/after full IDs/text/usage/finish regression; raw result retained')
+                        except BaseException as error:
+                            primary = repr(error)
+                            owner.receipt['primary_failure'] = primary
+                            raise
+                        finally:
+                            # No later sample starts after primary, cleanup or receipt I/O failure.
+                            try:
+                                owner.close()
+                            except BaseException as error:
+                                raise RuntimeError('native owner cleanup/evidence failed; primary=' + repr(primary) +
+                                                   '; cleanup=' + repr(error)) from error
+                            finally:
+                                report['samples'].append(dict(label=label, model=case['model'],
+                                    concurrency=case['concurrency'], repetition=repetition, order_position=position,
+                                    receipt=file_row(directory / 'receipt.json')))
+                                save(output / 'report.json', report, output)
+            cancellation.check()
+            require(len(report['samples']) == 32, 'whole planned32 processes completed')
+            require(source_freeze() == freeze and accepted_candidate(gate_sha) == plan['candidate_gate'],
+                    'all source/gates unchanged after complete native run')
+            for row in binding['executables'].values():
+                require(sha(row['path']) == row['sha256'], 'immutable native binary after run')
+            model_sources(pins)
+            report['status'] = 'RAW_NATIVE_BASELINE_CANDIDATE_COMPLETE_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        report['failure'] = repr(error)[:2048]
+        raise
+    finally:
+        os.sched_setaffinity(0, allowed)
+        save(output / 'report.json', report, output)
+        tree(output)
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ sigterm_cancellation.py
@@ -0,0 +1,47 @@
+"""SIGTERM/SIGINT checkpoints around the original bounded log drain, not new ownership."""
+import signal
+
+
+class SigtermCancellation:
+    def __init__(self):
+        self.received = False
+        self.previous = {}
+
+    def _record(self, signum, frame):
+        # Never asynchronously raise across Popen assignment or retirement.
+        self.received = True
+
+    def __enter__(self):
+        for signum in (signal.SIGTERM, signal.SIGINT):
+            self.previous[signum] = signal.signal(signum, self._record)
+        return self
+
+    def __exit__(self, kind, value, traceback):
+        for signum, previous in self.previous.items():
+            signal.signal(signum, previous)
+        return False
+
+    def exception(self):
+        return InterruptedError("interrupted: retire original owned serving regression before failure")
+
+    def check(self):
+        if self.received:
+            raise self.exception()
+
+    def observe(self, log, process, seconds):
+        original_drain = log.drain
+
+        def checked_drain(timeout):
+            self.check()
+            original_drain(timeout)
+            self.check()
+
+        self.check()
+        log.drain = checked_drain
+        try:
+            result = log.observe(process, seconds)
+            self.check()
+            return result
+        finally:
+            # Original cleanup/finish drain proceeds even after repeated SIGTERM.
+            log.drain = original_drain
--- /dev/null
+++ test_contracts.py
@@ -0,0 +1,79 @@
+"""ROOT-only prepared source-free contracts; no children/engines on import."""
+import copy
+import hashlib
+import unittest
+
+from native_ab_common import BANDS, COUNTERS, LABELS, MODELS, ORDER, counters, signature
+
+
+def health():
+    admission = {name: 0 for name in COUNTERS}
+    admission.update(kind='resident_gpu', batch_mode='fused', native_batch_rows_max=8,
+        active_sessions=0, queued_requests=0, gpu_fused_batches_by_rows={'2': 0, '4': 0, '8': 0})
+    return dict(admission=admission)
+
+
+def advanced():
+    before, after = health(), health()
+    a = after['admission']
+    a.update(completed_forward_kernel_dispatches=100, completed_model_token_steps=12,
+        committed_gpu_roots=12, gpu_fused_batches=2, gpu_fused_state_only_batches=1,
+        gpu_fused_greedy_batches=1, gpu_fused_state_only_rows=4, gpu_fused_rows_completed=12,
+        native_sampler_dispatches_completed=4, native_projection_dispatches_completed=9,
+        accepted_output_tokens=3)
+    a['gpu_fused_batches_by_rows'] = {'2': 0, '4': 1, '8': 1}
+    return before, after
+
+
+class Contracts(unittest.TestCase):
+    def test_explicit_native_order_and_complete_denominators(self):
+        self.assertEqual(LABELS, ('native_baseline', 'native_candidate'))
+        values = [(model, c, label) for model in MODELS for _, c in BANDS for order in ORDER for label in order]
+        self.assertEqual(len(values), 32)
+        self.assertEqual(sum(1 if c == 1 else 8 for _, c, _ in values), 144)
+        for model in MODELS:
+            for _, c in BANDS:
+                self.assertEqual([v[2] for v in values if v[:2] == (model, c)], list(ORDER[0] + ORDER[1]))
+
+    def test_actual_counter_phase_band_row_and_sampler_conservation(self):
+        self.assertEqual(counters(*advanced())['gpu_fused_rows_completed'], 12)
+        for key in ('gpu_fused_rows_completed', 'gpu_fused_batches', 'native_sampler_dispatches_completed'):
+            before, after = advanced()
+            after['admission'][key] += 1
+            with self.assertRaises(ValueError):
+                counters(before, after)
+
+    def test_absent_stale_failed_or_only_present_counters_refuse(self):
+        for key in ('failed_gpu_quanta', 'cancelled_requests', 'native_admission_refusals', 'discarded_ready_cohorts'):
+            before, after = advanced()
+            after['admission'][key] = 1
+            with self.assertRaises(ValueError):
+                counters(before, after)
+        with self.assertRaises(ValueError):
+            counters(health(), health())
+        before, after = advanced()
+        del after['admission']['gpu_fused_batches_by_rows']['8']
+        with self.assertRaises(KeyError):
+            counters(before, after)
+
+    def test_every_offer_and_full_ID_usage_are_required(self):
+        original = dict(request_id='request-0', prompt='literal transport fixture', max_tokens=8)
+        row = dict(request_id='request-0', prompt_sha256=hashlib.sha256(original['prompt'].encode()).hexdigest(),
+            max_tokens=8, completed=True, usage_reported_completion_tokens=2, finish_reason='length',
+            output_sha256='a' * 64, wire_token_observation=dict(status='provider_bound_complete_ids', token_ids=[11, 353]))
+        sample = dict(offered_requests=1, completed_requests=1, unsuccessful_requests=0,
+            completions_without_usage=0, requests=[row])
+        self.assertEqual(signature(sample, [original])['request-0']['token_ids'], [11, 353])
+        for field, bad in [('completed_requests', 0), ('offered_requests', 2)]:
+            value = copy.deepcopy(sample)
+            value[field] = bad
+            with self.assertRaises(ValueError):
+                signature(value, [original])
+        value = copy.deepcopy(sample)
+        value['requests'][0]['wire_token_observation']['token_ids'] = [11]
+        with self.assertRaises(ValueError):
+            signature(value, [original])
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ test_owner_control.py
@@ -0,0 +1,99 @@
+"""Prepared MOCK control fixtures only; no process/provider/physical claims."""
+from pathlib import Path
+import tempfile
+from types import SimpleNamespace
+import unittest
+from unittest.mock import patch
+
+from native_ab_owner import NativeOwner
+from sigterm_cancellation import SigtermCancellation
+
+
+class Control(unittest.TestCase):
+    def fixture(self, root):
+        events = []
+        cancel = SigtermCancellation()
+        process = SimpleNamespace(pid=123, stdout=SimpleNamespace(close=lambda: events.append('pipe')))
+        def cleanup(*_args):
+            events.append('cleanup')
+            return []
+        owned = SimpleNamespace(leader_status_unreaped=lambda _: None,
+            owned_container=lambda *_: 'a' * 64, cleanup=cleanup)
+        class Log:
+            error = None
+            def __init__(self, *_):
+                events.append('log-created')
+            def finish(self, _):
+                events.append('log-finished')
+                return dict(reader_complete=True, reader_error=None, truncated=False, bytes_seen=0)
+        class Monitor:
+            def __init__(self, *_):
+                events.append('monitor-created')
+            def finish(self):
+                events.append('monitor-finished')
+                return dict(failures=[], samples=[{}])
+        observations = SimpleNamespace(quiet=lambda *_: [dict(memory_used_mib=0)], Monitor=Monitor)
+        lifecycle = SimpleNamespace(listening=lambda _: False, BoundedLog=Log)
+        cgroup = SimpleNamespace(validate_caps=lambda *_: None)
+        deps = (None, None, owned, observations, cgroup, None, None, lifecycle)
+        protocol = dict(model='0.8b', zerogravity_binary_sha256='a' * 64,
+            ports=[18460, 18461], gpu_uuid='synthetic-control-only', gpu_total_mib=1, cpu_affinity=[0, 1, 2, 3])
+        owner = NativeOwner('native_baseline', root / 'sample', ['literal-mock'], protocol, deps, cancel, root)
+        return owner, cancel, process, events
+
+    def test_recorded_signal_across_Popen_handoff_still_retires_owned_child(self):
+        with tempfile.TemporaryDirectory() as value:
+            owner, cancel, process, events = self.fixture(Path(value))
+            def handoff(*_args, **_kwargs):
+                cancel._record(None, None)  # No asynchronous raise/real signal.
+                return process
+            with patch('native_ab_owner.subprocess.Popen', side_effect=handoff):
+                with self.assertRaises(InterruptedError):
+                    owner.start()
+            owner.close()
+            self.assertIs(owner.process, process)
+            self.assertEqual(events[-3:], ['monitor-finished', 'cleanup', 'log-finished'])
+
+    def test_postspawn_receipt_failure_cannot_skip_cleanup(self):
+        with tempfile.TemporaryDirectory() as value:
+            owner, _, process, events = self.fixture(Path(value))
+            with patch('native_ab_owner.subprocess.Popen', return_value=process), \
+                 patch.object(owner, 'save', side_effect=OSError('injected receipt failure')):
+                with self.assertRaises(OSError):
+                    owner.start()
+                with self.assertRaises(OSError):
+                    owner.close()
+            self.assertIn('cleanup', events)
+            self.assertIn('log-finished', events)
+
+    def test_monitor_failure_does_not_skip_group_or_log_cleanup(self):
+        with tempfile.TemporaryDirectory() as value:
+            owner, _, process, events = self.fixture(Path(value))
+            owner.process = process
+            owner.monitor = SimpleNamespace(finish=lambda: (_ for _ in ()).throw(ValueError('injected monitor')))
+            owner.log = SimpleNamespace(finish=lambda _: events.append('log-finished') or
+                dict(reader_complete=True, reader_error=None, truncated=False, bytes_seen=0))
+            owner.receipt['GPU_quiet_before'] = [dict(memory_used_mib=0)]
+            with self.assertRaises(ValueError):
+                owner.close()
+            self.assertIn('cleanup', events)
+            self.assertIn('log-finished', events)
+            self.assertEqual(owner.receipt['status'], 'UNQUALIFIED')
+
+    def test_group_failure_does_not_skip_log_retirement_or_allow_reuse(self):
+        with tempfile.TemporaryDirectory() as value:
+            owner, _, process, events = self.fixture(Path(value))
+            owner.process = process
+            owner.owned.cleanup = lambda *_: (_ for _ in ()).throw(ValueError('injected group'))
+            owner.log = SimpleNamespace(finish=lambda _: events.append('log-finished') or
+                dict(reader_complete=True, reader_error=None, truncated=False, bytes_seen=0))
+            owner.receipt['GPU_quiet_before'] = [dict(memory_used_mib=0)]
+            with self.assertRaises(ValueError):
+                owner.close()
+            self.assertIn('log-finished', events)
+            with self.assertRaises(ValueError):
+                owner.close()
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ validate_native_ab.py
@@ -0,0 +1,163 @@
+"""ROOT ONLY independent native/native accounting, no performance promotion."""
+import argparse
+import json
+from pathlib import Path
+import statistics
+
+from native_ab_common import (HERE, LABELS, ORDER, accepted_candidate, counters, finite,
+    load, parents, pin, require, save, sha, signature, source_freeze, tree)
+from run_native_ab import validate_plan
+
+
+def validate(output):
+    report = load(output / 'report.json')
+    require(report['schema'] == 'zerogravity.native-before-after.run.v1' and
+            report['status'] == 'RAW_NATIVE_BASELINE_CANDIDATE_COMPLETE_REQUIRES_INDEPENDENT_VALIDATION' and
+            report['source_freeze_sha256'] == source_freeze() and report['native_only'] is True and
+            report['main_timing_profiled'] is False and
+            report['performance_or_vllm_or_source_correct_goodput_qualified'] is False and
+            'failure' not in report, 'whole native/native execution, never a successful subset')
+    pin(report['plan'])
+    protocol, launcher, _, _, _, pins, _, _ = parents()
+    binding = load(HERE / 'bindings.json')
+    plan = validate_plan(load(report['plan']['path']), protocol, launcher, pins, binding)
+    require(accepted_candidate(plan['candidate_gate']['serving_gate']['sha256']) == plan['candidate_gate'],
+            'candidate actual serving gate and all antecedents unchanged')
+    expected = [(case, repetition, position, label) for case in plan['cases']
+                for repetition, order in enumerate(ORDER) for position, label in enumerate(order)]
+    require(len(report['samples']) == len(expected) == 32, 'all32 fresh owned processes')
+    stable, rows, offer_count, warm_count = {}, [], 0, 0
+    directories = set()
+    for item, (case, repetition, position, label) in zip(report['samples'], expected):
+        require((item['label'], item['model'], item['concurrency'], item['repetition'], item['order_position']) ==
+                (label, case['model'], case['concurrency'], repetition, position), 'complete ABBA/BAAB membership/order')
+        pin(item['receipt'])
+        path = Path(item['receipt']['path'])
+        require(path.name == 'receipt.json' and path.parent.parent == output and path.parent not in directories,
+                'exclusive per-process receipt, no reused sample')
+        directories.add(path.parent)
+        receipt = load(path)
+        require(receipt['label'] == label and receipt['native_implementation'] == 'zerogravity' and
+                receipt['status'] == 'OWNERS_CLOSED' and receipt['primary_failure'] is None and
+                receipt['cleanup_errors'] == [] and receipt['profiler_enabled'] is False and receipt['ready'] is True and
+                receipt['executable_sha256'] == binding['executables'][label]['sha256'] and
+                receipt['owned_container_absent_after_reap'] is True,
+                'genuine labeled native and original nonce/CID owner closed')
+        foreground = receipt['foreground_cleanup']
+        require(all(foreground[k] is True for k in ('leader_reaped', 'group_absent', 'all_nonzero_signals_before_reap')) and
+                foreground['stage_errors'] == [], 'all actual original group stages and final reap')
+        log = receipt['log_cleanup']
+        require(log['reader_complete'] is True and log['reader_error'] is None and log['truncated'] is False and
+                log['bytes_seen'] == log['bytes_retained'] <= 8 * 1024**2 and
+                Path(log['path']).parent == path.parent and sha(log['path']) == log['sha256_retained'],
+                'complete bounded log/pipe EOF evidence')
+        bootstrap = receipt['actual_bootstrap']
+        require(bootstrap['provider'] == 'zerogravity' and bootstrap['model'] == case['model'] and
+                bootstrap['stock_math_modified'] is False and bootstrap['policy_observer'] is False and
+                bootstrap == load(path.parent / 'bootstrap.json'), 'actual unchanged original native bootstrap')
+        from cgroup_readback_v3 import validate_caps
+        validate_caps(bootstrap['cgroup_readback'], case['protocols'][label]['cpu_affinity'])
+        warm = receipt['warmup']
+        require(warm['completed'] is True and warm['tokens'] is not None and
+                warm['token_observation']['status'] == 'provider_bound_complete_ids' and
+                len(warm['token_observation']['token_ids']) == warm['tokens'] <= 8,
+                'one complete ordinary unmeasured canonical warmup')
+        warm_count += 1
+        pin(receipt['measurement'])
+        measurement = load(receipt['measurement']['path'])
+        require(Path(receipt['measurement']['path']) == path.parent / 'measurement.json' and
+                (measurement['label'], measurement['model'], measurement['concurrency'],
+                 measurement['repetition'], measurement['order_position']) ==
+                (label, case['model'], case['concurrency'], repetition, position), 'bound actual measurement identity')
+        sample = measurement['benchmark']
+        require(sample == load(path.parent / 'raw-measurement.json') and
+                sample['concurrency'] == case['concurrency'] and sample['arrival_mode'] == 'open_loop_scheduled_arrivals',
+                'raw canonical measurement unchanged')
+        work = protocol.workload(case['workload'], 8)
+        sig = signature(sample, work)
+        reference = stable.setdefault(case['model'], sig['request-0'])
+        require(all(value == reference for value in sig.values()), 'all versions/bands complete exact outputs')
+        require(warm['output_sha256'] == reference['output_sha256'] and
+                warm['tokens'] == reference['usage_reported_completion_tokens'] and
+                warm['finish_reason'] == reference['finish_reason'] and
+                warm['token_observation']['token_ids'] == reference['token_ids'],
+                'same exact ordinary warmup output, not a special shorter request')
+        before, after = measurement['quiescent_before'], measurement['quiescent_after']
+        require(load(path.parent / 'quiescent.json') == dict(before=before, after=after), 'actual quiescent window')
+        delta = counters(before, after)
+        require(delta == measurement['source_counter_delta'] and delta['accepted_output_tokens'] ==
+                sum(len(value['token_ids']) for value in sig.values()), 'actual native packet/counter conservation')
+        if case['concurrency'] == 1:
+            require(delta['gpu_fused_batches'] == delta['native_projection_dispatches_completed'] ==
+                    delta['native_sampler_dispatches_completed'] == 0, 'no B1 request padding')
+        delivered, intervals, commit_intervals, packet_bands = 0, [], [], []
+        for request in sample['requests']:
+            obs = request['wire_token_observation']
+            ids = obs['token_ids']
+            if obs['client_delivery_status'] == 'individual_token_ids_observed':
+                require(len(obs['client_token_delivery_intervals_seconds']) == max(0, len(ids) - 1) and
+                        all(finite(v) for v in obs['client_token_delivery_intervals_seconds']), 'actual token intervals')
+                delivered += 1
+                intervals.extend(obs['client_token_delivery_intervals_seconds'])
+            require(obs['provider_commit_observation_status'] == 'provider_bound_complete_commit_observations' and
+                    len(obs['provider_token_records']) == len(ids), 'enabled original native per-request commit records')
+            commit_intervals.extend(obs['provider_commit_observation_intervals_seconds'])
+            for index, packet in enumerate(obs['provider_token_records']):
+                require(packet['gpu']['producer_frontier'] == 47 + index, 'actual independent canonical prompt/output frontier')
+                cohort = packet['gpu']['native_cohort']
+                if cohort is not None:
+                    require(cohort['width'] in (2, 4, 8), 'actual accepted row width')
+                    packet_bands.append(cohort['width'])
+        require(sample['wire_token_metrics']['completed_requests_with_individual_token_delivery'] == delivered,
+                'honest token ITL/coalescing coverage')
+        observations = receipt['physical_observations']
+        require(observations['failures'] == [] and observations['samples'] and observations['exact_peak'] is False and
+                all(v['uuid'] == case['protocols'][label]['gpu_uuid'] for v in observations['samples']),
+                'actual same-card point memory samples, not exact peaks')
+        rows.append(dict(label=label, model=case['model'], concurrency=case['concurrency'],
+            repetition=repetition, order_position=position, offered=sample['offered_requests'],
+            completed=sample['completed_requests'], usage_tokens=sample['usage_reported_completion_tokens'],
+            offered_window_seconds=sample['offered_window_seconds'],
+            reported_tokens_per_offered_second=sample['reported_tokens_per_offered_second'],
+            ttft_text_p50=sample['ttft_p50_seconds'], ttft_text_p95=sample['ttft_p95_seconds'],
+            actual_wire_token_metrics=sample['wire_token_metrics'], individual_ID_requests=delivered,
+            token_intervals=len(intervals), coalesced_or_unresolved_ID_requests=len(work)-delivered,
+            actual_packet_native_widths=sorted(set(packet_bands)), source_counter_delta=delta,
+            native_fused_work_observed=delta['gpu_fused_batches'] > 0,
+            fairness_by_local_label=sample['fairness_by_workload_tenant'],
+            point_sampled_total_device_peak_mib=observations['sampled_total_device_peak_mib'],
+            source_launchsites_not_hardware_kernel_counts=True))
+        offer_count += sample['offered_requests']
+    require(offer_count == plan['measured_offers'] == 144 and warm_count == plan['warmup_requests'] == 32,
+            'no warmup/measured offer denominator mixing')
+    summaries = []
+    for case in plan['cases']:
+        group = [r for r in rows if r['model'] == case['model'] and r['concurrency'] == case['concurrency']]
+        rates = {label: [r['reported_tokens_per_offered_second'] for r in group if r['label'] == label] for label in LABELS}
+        require(all(len(values) == 4 and all(finite(v) for v in values) for values in rates.values()), 'all four provider samples')
+        summaries.append(dict(model=case['model'], concurrency=case['concurrency'],
+            median_usage_throughput_by_native_label={label: statistics.median(values) for label, values in rates.items()},
+            four_samples_per_label=True, confidence_certified_win=False))
+    return dict(status='PASS_NATIVE_BASELINE_CANDIDATE_RAW_ACCOUNTING_ONLY', samples=rows, summaries=summaries,
+        measured_offers=offer_count, warmup_requests=warm_count, source_freeze_sha256=source_freeze(),
+        all_full_ID_text_usage_finish_exact=True, main_timing_profiled=False,
+        actual_hardware_kernel_counts_qualified=False, source_correct_slo_goodput_qualified=False,
+        vllm_comparison_or_parity_qualified=False, graph_or_full_HF_eligibility=False,
+        complete_phase_acceptance_requires_ROOT_review=True)
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--output', type=Path, required=True)
+    args = parser.parse_args()
+    output = args.output.resolve()
+    require(output.parent == HERE, 'own exact native/native output only')
+    value = validate(output)
+    save(output / 'validation.json', value, output)
+    tree(output)
+    print(json.dumps(dict(status=value['status'], measured_offers=value['measured_offers'],
+                          warmup_requests=value['warmup_requests']), sort_keys=True))
+
+
+if __name__ == '__main__':
+    main()

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/followup-v2/frozen/NEW_CONTROL_V2.review.patch

Original UTF-8 bytes: 25917; SHA256 `94493f65e64b810d411fa253d7507480eab5349920b06b8a09735c729e09e257`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ native_ab_v2.py
@@ -0,0 +1,17 @@
+"""Additive source binding only; all original native/native contracts unchanged."""
+import native_ab_common as original
+from native_ab_common import *
+
+V1 = '6e5fd796b695e1a3c662502319ba1a7608ff1c5584cea3e3cc4f7e3aa7fc3bf2'
+
+
+def source_freeze():
+    require(original.source_freeze() == V1, 'immutable original native/native preparation')
+    path = HERE / 'followup-v2/freeze-manifest-v2.json'
+    value = load(path)
+    for row in value['files']:
+        pin(dict(row, path=str(HERE / row['file'])))
+        pin(dict(row, path=str(HERE / 'followup-v2/frozen' / row['file'])))
+    for row in value['original_wrapper_and_support']:
+        pin(row)
+    return sha(path)
--- /dev/null
+++ harmless_http_v2.py
@@ -0,0 +1,58 @@
+"""Explicit CPU-only cancellation target. NOT a model/native serving engine."""
+import argparse
+from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
+import json
+import signal
+import threading
+import time
+
+
+class Handler(BaseHTTPRequestHandler):
+    def do_GET(self):
+        value = dict(control_only=True, model_or_GPU_loaded=False)
+        if self.path == '/phase':
+            value['post_open'] = self.server.post_open.is_set()
+        raw = json.dumps(value).encode()
+        # Original native readiness must NOT accept this target.
+        self.send_response(503 if self.path == '/v1/models' else 200)
+        self.send_header('Content-Length', str(len(raw)))
+        self.end_headers()
+        self.wfile.write(raw)
+
+    def do_POST(self):
+        length = int(self.headers.get('Content-Length', '0'))
+        if not 0 < length <= 4096:
+            self.send_error(400)
+            return
+        self.rfile.read(length)
+        self.send_response(200)
+        self.send_header('Content-Type', 'text/event-stream')
+        self.end_headers()
+        self.wfile.flush()
+        self.server.post_open.set()
+        print('CONTROL_POST_OPEN', flush=True)
+        # No IDs/output/model arithmetic. Parent uses short canonical timeout;
+        # an independently retained owner kills this harmless target afterward.
+        time.sleep(20)
+
+    def log_message(self, *_):
+        pass
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument('--port', type=int, required=True)
+    args = parser.parse_args()
+    if not 1024 <= args.port <= 65535:
+        raise SystemExit('bounded control port')
+    # Exercise original owned TERM->KILL path. Only this fresh owned container
+    # is affected; this process has no GPU device/model/source mounts.
+    signal.signal(signal.SIGTERM, signal.SIG_IGN)
+    print('HARMLESS_CONTROL_READY_NO_MODEL_NO_GPU', flush=True)
+    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
+    server.post_open = threading.Event()
+    server.serve_forever()
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ test_owner_process_v2.py
@@ -0,0 +1,250 @@
+"""ROOT ONLY real harmless owners; NEVER GPU/model/native-engine qualification."""
+import argparse
+import os
+from pathlib import Path
+import signal
+import subprocess
+import threading
+import time
+from types import SimpleNamespace
+
+from native_ab_v2 import (HERE, http_json, install_opener, load, parents, require,
+                         save, sha, source_freeze, tree)
+from native_ab_owner import NativeOwner
+from run_native_ab_v2 import private_namespace
+from sigterm_cancellation import SigtermCancellation
+
+
+def control_observation(*_):
+    # Deliberate fixture stub, visibly unlike real point-memory samples. Never
+    # call nvidia-smi, attach a device or imply GPU quietness in this fixture.
+    return [dict(memory_used_mib=0, control_only=True, GPU_observation_performed=False)]
+
+
+class NoGpuMonitor:
+    def __init__(self, *_):
+        pass
+
+    def finish(self):
+        return dict(control_only=True, GPU_observation_performed=False, samples=[])
+
+
+def wait_control(owner, stop, failures):
+    deadline = time.monotonic() + 60
+    while not stop.is_set() and time.monotonic() < deadline:
+        try:
+            require(owner.owned.owned_container(owner.name, owner.nonce, owner.cidfile) is not None,
+                    'actual harmless nonce/CID ownership')
+            value = http_json(owner.endpoint() + '/health')
+            if value == dict(control_only=True, model_or_GPU_loaded=False):
+                return True
+        except Exception:
+            pass
+        stop.wait(.05)
+    if not stop.is_set():
+        failures.append('harmless target readiness deadline')
+    return False
+
+
+def signal_when_ready(owner, signum, stop, failures):
+    if wait_control(owner, stop, failures) and not stop.is_set():
+        # Only our still-running supervisor receives the actual signal. The
+        # original guard records it; no guessed/exited external PID is used.
+        os.kill(os.getpid(), signum)
+
+
+def signal_during_request(owner, signum, stop, failures):
+    deadline = time.monotonic() + 10
+    while not stop.is_set() and time.monotonic() < deadline:
+        try:
+            value = http_json(owner.endpoint() + '/phase')
+            if value == dict(control_only=True, model_or_GPU_loaded=False, post_open=True):
+                require(owner.owned.owned_container(owner.name, owner.nonce, owner.cidfile) is not None,
+                        'actual nonce/CID during open harmless request')
+                require(owner.owned.leader_status_unreaped(owner.process) is None,
+                        'original foreground live and unreaped during request')
+                if stop.is_set():
+                    return
+                owner.receipt['signal_at_open_control_POST'] = value
+                os.kill(os.getpid(), signum)
+                return
+        except Exception:
+            pass
+        stop.wait(.02)
+    if not stop.is_set():
+        failures.append('actual open harmless POST not observed before signal deadline')
+
+
+def start_control(owner):
+    """Fixture-only transport start; native/model readiness is never forged."""
+    require(not owner.lifecycle.listening(('127.0.0.1', owner.protocol['ports'][0])),
+            'exclusive private harmless endpoint')
+    owner.receipt['GPU_quiet_before'] = control_observation()
+    replacements = {'{owned_name}': owner.name, '{owned_nonce}': owner.nonce,
+                    '{cidfile}': str(owner.cidfile), '{sample_directory}': str(owner.directory)}
+    command = owner.command
+    for key, value in replacements.items():
+        command = [part.replace(key, value) for part in command]
+    owner.receipt['argv'] = command
+    owner.cancel.check()
+    owner.process = subprocess.Popen(command, cwd=str(owner.protocol_root()),
+        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
+        start_new_session=True, bufsize=0)
+    owner.receipt['owned_cli_pid'] = owner.process.pid
+    owner.log = owner.lifecycle.BoundedLog(owner.process.stdout, owner.directory / 'execution.log', 8 * 1024**2)
+    owner.save()
+
+
+def case(root, index, phase, signum, dependencies):
+    protocol, _, owned, _, _, pins, bench, _ = dependencies
+    port = 18520 + index
+    p = dict(model='CONTROL_ONLY_NOT_A_MODEL', zerogravity_binary_sha256='NOT_A_NATIVE_BINARY',
+             ports=[port, port + 100], gpu_uuid='NO_GPU_IN_CONTROL_FIXTURE', gpu_total_mib=0,
+             cpu_affinity=sorted(os.sched_getaffinity(0)))
+    # Genuine pinned local image/Podman, no model/source/native binary/device
+    # mounts and no acquisition. The only executed payload is this stdlib stub.
+    command = owned.PODMAN + ['run', '--name', '{owned_name}', '--label', owned.LABEL + '={owned_nonce}',
+        '--cidfile', '{cidfile}', '--rm', '--pull=never', '--network=host', '--read-only',
+        '--cap-drop=all', '--security-opt=no-new-privileges', '--memory=256m', '--memory-swap=256m',
+        '--cpus=1', '--pids-limit=32', '--ulimit', 'fsize=4194304:4194304',
+        '--tmpfs', '/tmp:rw,nosuid,nodev,size=16m', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
+        '-e', 'CUDA_VISIBLE_DEVICES=-1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
+        '-v', str(HERE / 'harmless_http_v2.py') + ':/control.py:ro',
+        '--entrypoint', 'python3', pins['image'], '/control.py', '--port', str(port)]
+    # Original native adapter cleanup/log owner, only the GPU observer is an
+    # explicit control stub. Its empty samples cannot pass native validation.
+    deps = list(dependencies)
+    deps[3] = SimpleNamespace(quiet=control_observation, Monitor=NoGpuMonitor)
+    directory = root / f'{index:02d}-{phase}-{signum or 0}'
+    stop, failures = threading.Event(), []
+    thread = None
+    expected, outcome, close_failure = None, None, None
+    old_cleanup = None
+    with SigtermCancellation() as cancellation:
+        owner = NativeOwner('HARMLESS_CONTROL', directory, command, p, deps, cancellation, root)
+        owner.receipt.update(native_implementation='HARMLESS_CONTROL_NOT_NATIVE_ENGINE',
+            control_only=True, model_GPU_native_math_or_benchmark_qualified=False,
+            original_native_readiness_not_forged=True, phase=phase, signal=signum)
+        try:
+            if phase == 'before_popen':
+                os.kill(os.getpid(), signum)
+                expected = 'InterruptedError'
+                owner.start()  # Must refuse before ANY Popen/device/model work.
+            elif phase == 'readiness':
+                thread = threading.Thread(target=signal_when_ready,
+                    args=(owner, signum, stop, failures))
+                thread.start()
+                expected = 'InterruptedError'
+                owner.start()  # Actual stdlib target returns503, never native readiness.
+            else:
+                start_control(owner)
+                wait_control(owner, stop, failures)
+                require(not failures, 'actual harmless target startup')
+                if phase == 'request':
+                    thread = threading.Thread(target=signal_during_request,
+                        args=(owner, signum, stop, failures))
+                    thread.start()
+                    observed = bench.stream_one(owner.endpoint(), 'HARMLESS_CONTROL_NOT_A_MODEL',
+                        'control-only cancellation transport', 1, 2, None,
+                        request_id='harmless-control', return_token_ids=True)
+                    require(observed.completed is False, 'control stub emits no fake token/model result')
+                    expected = 'InterruptedError'
+                    cancellation.check()
+                elif phase == 'cleanup':
+                    old_cleanup = owned.cleanup
+                    def signalled_cleanup(*args):
+                        os.kill(os.getpid(), signum)
+                        return old_cleanup(*args)  # Delegate unchanged genuine53ee stages.
+                    owned.cleanup = signalled_cleanup
+                else:
+                    require(phase == 'receipt_io', 'closed fixture phase')
+                    expected = 'OSError'
+                    def fail_receipt():
+                        raise OSError('injected postspawn receipt failure')
+                    owner.save = fail_receipt
+                    owner.save()
+        except BaseException as error:
+            outcome = type(error).__name__
+            owner.receipt['expected_control_failure'] = repr(error)
+        finally:
+            try:
+                owner.close()
+            except BaseException as error:
+                close_failure = type(error).__name__
+            finally:
+                if old_cleanup is not None:
+                    owned.cleanup = old_cleanup
+                stop.set()
+                if thread is not None:
+                    # One in-flight original owned inspect (10s) plus HTTP(2s)
+                    # must retire; no signal thread survives guard restoration.
+                    thread.join(15)
+                    require(not thread.is_alive(), 'own signal thread retired')
+        require(not failures, 'control readiness failed, never accepted as negative proof')
+        if phase == 'cleanup':
+            require(cancellation.received is True and outcome is None and close_failure is None,
+                    'recorded signal did not interrupt original cleanup')
+        elif phase == 'receipt_io':
+            require(outcome == expected and close_failure == 'OSError', 'actual expected postspawn IO failure')
+        else:
+            require(outcome == expected and close_failure is None, 'actual expected recorded signal')
+            if phase == 'request':
+                require(owner.receipt.get('signal_at_open_control_POST', {}).get('post_open') is True,
+                        'request cancellation requires actual open transport evidence')
+        require(owner.receipt.get('ready') is not True, 'fixture never accepted native/model readiness')
+        if phase == 'before_popen':
+            require(owner.process is None, 'no child after observed prelaunch cancellation')
+        else:
+            fg, log = owner.receipt['foreground_cleanup'], owner.receipt['log_cleanup']
+            require(owner.receipt['owned_container_absent_after_reap'] is True and
+                    all(fg[k] is True for k in ('leader_reaped', 'group_absent', 'all_nonzero_signals_before_reap')) and
+                    fg['stage_errors'] == [] and log['reader_complete'] is True and
+                    log['reader_error'] is None and log['truncated'] is False and owner.receipt['cleanup_errors'] == [],
+                    'real harmless group/container/EOF closed even after signal/IO failure')
+            require(owned.owned_container(owner.name, owner.nonce, owner.cidfile) is None,
+                    'actual final nonce-bound container absence')
+        # Restore only fixture-injected method, AFTER real owner retirement, so
+        # expected receiptIO failure can retain its independent cleanup evidence.
+        owner.save = lambda: save(directory / 'receipt.json', owner.receipt, root)
+        owner.save()
+        value = dict(status='PASS_HARMLESS_CONTROL_OWNER_ONLY', phase=phase, signal=signum,
+            receipt=str(directory / 'receipt.json'), receipt_sha256=sha(directory / 'receipt.json'),
+            expected_failure=expected, observed_failure=outcome, close_failure=close_failure,
+            real_owned_child=owner.process is not None, GPU_model_native_math_or_benchmark_qualified=False)
+        save(directory / 'control-validation.json', value, root)
+        return value
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--run-control-only', action='store_true')
+    parser.add_argument('--new-output', type=Path, required=True)
+    args = parser.parse_args()
+    require(args.run_control_only, 'ROOT explicitly executes harmless owners only after source review')
+    freeze = source_freeze()
+    dependencies = parents()
+    install_opener()
+    namespace = private_namespace()
+    root = args.new_output.resolve()
+    require(root.parent == HERE and not root.exists(), 'fresh own control artifacts')
+    root.mkdir(mode=0o700)
+    report = dict(status='RUNNING_HARMLESS_CONTROL_ONLY', source_freeze_sha256=freeze,
+                  namespace=namespace, cases=[], GPU_model_native_math_or_benchmark_qualified=False)
+    try:
+        cases = [(phase, sig) for phase in ('before_popen', 'readiness', 'request', 'cleanup')
+                 for sig in (signal.SIGTERM, signal.SIGINT)] + [('receipt_io', None)]
+        for index, (phase, sig) in enumerate(cases):
+            report['cases'].append(case(root, index, phase, sig, dependencies))
+            save(root / 'control-report.json', report, root)
+        require(len(report['cases']) == 9 and source_freeze() == freeze, 'all harmless cases and original sources')
+        report['status'] = 'PASS_NINE_HARMLESS_CONTROL_CASES_NOT_NATIVE_OR_GPU_QUALIFICATION'
+    except BaseException as error:
+        report['failure'] = repr(error)[:2048]
+        raise
+    finally:
+        save(root / 'control-report.json', report, root)
+        tree(root)
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ test_namespace_v2.py
@@ -0,0 +1,66 @@
+"""Prepared pure namespace-contract fixtures; no namespace/container execution."""
+from contextlib import ExitStack
+import os
+from pathlib import Path
+import struct
+from types import SimpleNamespace
+import unittest
+from unittest.mock import patch
+
+import run_native_ab_v2 as runner
+
+
+class Namespace(unittest.TestCase):
+    def namespace(self, mapping='0 1000 1\n1 100000 65536\n', configured='done',
+                  uid='1000', network_owner=9, interfaces=None, up=True):
+        interfaces = [(1, 'lo')] if interfaces is None else interfaces
+        def ioctl(_descriptor, request, *_):
+            if request == 0xB701:
+                return 11
+            self.assertEqual(request, 0x8913)
+            return struct.pack('16sH22x', b'lo', int(up))
+        with ExitStack() as stack:
+            stack.enter_context(patch.object(Path, 'read_text', return_value=mapping))
+            stack.enter_context(patch.dict(os.environ, {
+                '_CONTAINERS_USERNS_CONFIGURED': configured,
+                '_CONTAINERS_ROOTLESS_UID': uid}, clear=True))
+            stack.enter_context(patch.object(runner.os, 'open', return_value=10))
+            stack.enter_context(patch.object(runner.os, 'close'))
+            stack.enter_context(patch.object(runner.os, 'fstat', return_value=SimpleNamespace(st_ino=network_owner)))
+            stack.enter_context(patch.object(runner.os, 'stat', return_value=SimpleNamespace(st_ino=9)))
+            stack.enter_context(patch.object(runner.os, 'readlink', side_effect=lambda path: 'mock-only:' + path))
+            stack.enter_context(patch.object(runner.fcntl, 'ioctl', side_effect=ioctl))
+            stack.enter_context(patch.object(runner.socket, 'if_nameindex', return_value=interfaces))
+            stack.enter_context(patch.object(runner.socket, 'socket'))
+            return runner.private_namespace()
+
+    def test_exact_existing_full_rootless_map(self):
+        value = self.namespace()
+        self.assertEqual(value['uid_map'].split(), ['0', '1000', '1', '1', '100000', '65536'])
+        self.assertEqual(value['rootless_userns_configured'], 'done')
+        self.assertEqual(value['rootless_uid'], '1000')
+
+    def test_known_bad_single_uid_route_refused(self):
+        with self.assertRaises(ValueError):
+            self.namespace(mapping='0 1000 1\n')
+
+    def test_wrong_container_bootstrap_refused(self):
+        for fields in ({'configured': ''}, {'uid': '0'}):
+            with self.subTest(fields=fields), self.assertRaises(ValueError):
+                self.namespace(**fields)
+
+    def test_foreign_network_owner_refused(self):
+        with self.assertRaises(ValueError):
+            self.namespace(network_owner=8)
+
+    def test_non_loopback_interface_refused(self):
+        with self.assertRaises(ValueError):
+            self.namespace(interfaces=[(1, 'lo'), (2, 'eth0')])
+
+    def test_loopback_must_already_be_up(self):
+        with self.assertRaises(ValueError):
+            self.namespace(up=False)
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ README_V2.md
@@ -0,0 +1,86 @@
+# Native baseline/candidate V2 namespace correction
+
+Author: batch_design_review. Parent: serving-and-speculation P12.
+SOURCE ONLY; no author imports/tests, process/container/model/GPU execution,
+builds or binary/model/DSO payload reads/hashes.
+
+- Original12-file freeze6e5fd796 and original8 fixtures remain immutable.
+  V2 adds exact full rootless Podman UID mapping/bootstrap and current-user-owned
+  private network/loopback checks from accepted `root-private-loopback-exec-2.py`.
+  Its full mapping preserves the custom store's overlay ownership. The original
+  single-UID `native-gpu.py` invocation is unsuitable for this Podman driver.
+- `prepare_native_ab_v2.py` changes only its source-binding import.
+  `run_native_ab_v2.py` changes that import and namespace checks only.
+  `validate_native_ab_v2.py` changes its binding/runner imports only.
+  `native_ab_v2.py` checks both immutable V1 and additive V2 live/frozen bytes,
+  plus exact accepted wrapper/support bytes. Original workload, launcher, caps,
+  owner, cancellation recorder, numerical/accounting validation are unchanged.
+- Candidate gate now physically exists: ROOT worker14/HTTP12 receipt18625960,
+  source456, server87e1d727. Preparation still requires its explicit full SHA;
+  no pending/partial gate is accepted. This derivative contains no timing result.
+- Both native labels/models, warmed_process C1 and C8, ABBA+BAAB remain32 fresh
+  measured processes /144 offered requests /32 ordinary canonical warmups.
+  The canonical first prompt has47 IDs, max8. Neither label is a vLLM slot.
+  Original README metrics/limitations still apply.
+
+Prepared additional fixtures, ROOT executes only after source review:
+
+- `test_namespace_v2.py`: six pure/mocked tests; full mapping accepted, known
+  bad single UID/missing container bootstrap/foreign network/non-loopback/down
+  loopback refused. These are contract tests, not actual namespace evidence.
+- `test_owner_process_v2.py`: nine actual harmless-owner cases: TERM+INT before
+  Popen, during readiness, during a real open HTTP POST, during original owner
+  cleanup; one postspawn receipt-I/O failure. Only stdlib `harmless_http_v2.py`
+  runs in the already-pinned local image, no pulls, devices, model/source/native
+  binary mounts. Container256MiB/CPU1/pids32; no provider/native math.
+- Target `/v1/models` ALWAYS503; POST emits no ID/token/text/usage. Fixture
+  readiness is a separately labelled control health endpoint. Request signal
+  requires actual POST-open marker and authenticated live nonce/CID/foreground;
+  no delay-based inference. Original unchanged `serve-bench.stream_one` is used
+  only as bounded transport (fixture timeout2s/max1); this is NOT a canonical
+  benchmark sample. Its failure is mandatory.
+- Original NativeOwner and53ee WNOWAIT/container/group/EOF cleanup are exercised.
+  Signal goes only to the current supervisor, whose original recorder cannot
+  raise across handoff/cleanup. Actual container cleanup signals only original
+  authenticated owners. Stop/join precedes handler restoration; no next fixture
+  after any unexpected failure. The target ignores TERM to exercise owner kill.
+- GPU observation is an explicit `control_only` stub. Empty point samples and
+  `ready!=true`, HARMLESS labels/non-native binary prevent native acceptance.
+  No GPU quietness, engine/bootstrap/cgroup qualification follows these tests.
+  Receipt-I/O failure cannot skip cleanup; actual closure is retained afterward
+  via a restored fixture writer, with expected failures still explicit.
+
+ROOT invocation (not executed here):
+
+```bash
+repo=/home/pvl/spaces/abird/src/zerogravity
+own="$repo/.agents/runs/serving-and-speculation-native-before-after-20261020"
+parent="$repo/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020"
+export ZG_BENCH_HOST_ROOT="$repo" PYTHONDONTWRITEBYTECODE=1
+python3 -m unittest discover -s "$own" -p 'test_namespace_v2.py'
+
+# Exact accepted custom-store rootless wrapper2 chain, fresh private net/mount.
+podman --root /home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers \
+  --runroot /run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers \
+  unshare unshare --net --mount --propagation private \
+  python3 "$parent/root-private-loopback-exec-2.py" \
+  python3 "$own/test_owner_process_v2.py" --run-control-only \
+  --new-output "$own/root-harmless-v2-1"
+```
+
+After all fixtures/review and ROOT's exclusive GPU reservation:
+
+1. Use `prepare_native_ab_v2.py` with unchanged original preparation arguments;
+   fresh own plan directory and exact accepted V8 hardware-protocol SHA, plus
+   `--accepted-serving-sha256
+   186259602e1debf69df208c5f12be0f60687a6fb81c43a699c34f5a55dcd0431`.
+   ROOT performs original source/binary/weight binding scans outside samples.
+2. Repeat the exact Podman/unshare/wrapper2 chain above, replacing ONLY its final
+   Python command with `run_native_ab_v2.py --run --plan ABS_PLAN/plan.json
+   --plan-sha256 ACTUAL_SHA --new-output ABS_OWN/root-native-v2-1`.
+3. `validate_native_ab_v2.py --output ABS_OWN/root-native-v2-1`; review all raw
+   samples/owners/ITL coverage. Failures do not permit another sample/provider.
+
+Own V2 sources only were AST parsed. Real fixtures, preparation, actual timings
+and independent full receipt validation remain ROOT gates. No speed, hardware
+count, source-correct goodput, full HF or vLLM parity claim.
--- /dev/null
+++ TODO_V2.md
@@ -0,0 +1,13 @@
+# P12 native/native before-after V2
+
+Parent: `.agents/plans/serving-and-speculation/TODO.md`; owner batch_design_review.
+
+- [x] Preserve original6e5fd796 live/frozen12 files and ROOT original8 fixture PASS.
+- [x] Full accepted rootless UID/bootstrap/network/loopback guard; V2 bound imports.
+- [x] Prepare6 pure namespace and9 real harmless owner fixtures (not executed).
+- [x] Bind actual POST-open evidence, original owner/EOF, explicit no-GPU stub.
+- [x] Bounded review actual worker14/HTTP12 receipt/logs: no scope mismatch.
+- [x] Source-only AST/read/freeze; no author runtime or payload scans.
+- [ ] Independent V2 source review; ROOT pure and actual harmless fixture execution.
+- [ ] ROOT preparation and actual32-process canonical warmed native/native run.
+- [ ] Independent all samples/owners/numerics/accounting validation; no timing claim yet.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/frozen/NEW_FILES.review.patch

Original UTF-8 bytes: 78518; SHA256 `bc90626d98768eddf94920d518ac51cff80b03cb22b6d128c9b69003b0c8a654`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ README.md
@@ -0,0 +1,96 @@
+# Native baseline/candidate latency preparation
+
+Author: batch_design_review. Parent: serving-and-speculation P12.
+SOURCE ONLY; no author imports/tests, child/provider/model/GPU execution, builds,
+binary/model/DSO payload scans or timing results.
+
+- Explicit labels `native_baseline` (4a745527...) and `native_candidate`
+  (87e1d727...). Both are Zerogravity. Neither occupies a vLLM comparison slot;
+  no `serve-compare` labels, legacy ratios or `measure_same_engine` counters.
+- Direct unchanged canonical `tools/serve-bench.py`. The immutable V8 native
+  launch/bootstrap branch remains `provider=zerogravity`, its genuine ENGINE
+  identity, for both new result labels. Stock/vLLM is never launched. Paired
+  launch command normalization requires identical flags/mounts/environment
+  except the immutable native executable and its expected hash.
+- Both sources, warmed_process, C1 canonical solo and C8 canonical eight-offer
+  wave, max8. Exact existing47 prompt IDs/common greedy/EOS policy. Each of four
+  model/band cases runs ABBA then BAAB, eight fresh owned processes:32 measured
+  samples,144 measured offers (16solo+128wave),72 offers/native label, plus32
+  independent unmeasured warmup requests. ONE normal canonical prompt/max8
+  warmup per process; no extra shape/full-wave warmup.
+- Same exact four CPUs, CPU4 quota, RAM12GiB/no extra swap, original UUID/context,
+  active8/native4GiB host/provider+5GiB device, read-only inputs and private user/
+  network namespace; same V8 cgroup readback/point device observation. Native
+  graph remains disabled in the accepted source. This is neither cold-process
+  nor fully warmed shape/graph nor persistent-server steady-state measurement.
+
+Gates and ownership:
+
+- `bindings.json` pins the old actual accepted native metadata, candidate456
+  source freeze, strict/release build, operator10/model46/full isolated+both
+  approved routes and exact future worker14/HTTP12 receipt location. ROOT must
+  supply its completed receipt SHA. Missing/RUNNING/partial/failed/unclean or
+  unexpected-count receipt refuses preparation AND execution AND validation.
+  ROOT should inspect all gate evidence before supplying the SHA; marker alone
+  is insufficient. Later source drift or another runner version needs a fresh
+  derivative, never a silently widened acceptance predicate.
+- Original53ee nonce/name/CID container + reserved-leader WNOWAIT whole-group
+  TERM/KILL-before-any-reap cleanup; original bounded log reader/EOF and V8
+  ownership-authenticated point-memory monitor. Exact00d1 TERM+INT record-only
+  guard spans provider lifetime, with explicit checks outside Popen handoff and
+  retirement. No asynchronous exception crosses an unassigned child handle.
+- Canonical HTTP retains its existing180s deadline/socket-timeout controls. A
+  recorded cancel is checked after canonical return; this is no new hard
+  asynchronous interrupt bound. Owner remains retained. Cleanup ignores recorded
+  signals until all original stages close. No subsequent sample after any
+  numerical, counter, transport, owner, log, memory-observer or receipt failure.
+- Source/weights/binary binding scans run only in ROOT preparation/before+after
+  the whole run, outside timed samples. Logs8MiB/process with truncation refused;
+  whole output512MiB/1024files, every metadata write and final validation checks
+  the tree INCLUDING its newly written receipt. These are artifact acceptance
+  bounds, not a parent process physical-memory admission claim.
+
+Metrics and acceptance:
+
+- Every sample retains full raw benchmark BEFORE numerical/accounting checks.
+  Exact full IDs/text SHA/usage/finish match all rows/versions/bands, including
+  each ordinary warmup. No EOS/usage normalization or minimum-token workaround.
+- Quiescent health before/after excludes warmup and cleanup. Every required
+  counter is present, typed, monotonic; error/refusal/discard/cancel deltas0;
+  weighted2/4/8 bands close actual fused rows, phases and sampler4/cohort. These
+  are source launchsites, never CUDA/CUPTI enumerated kernel counts.
+- C8 is client concurrency, not a promised nativeB8 cohort. Preserve actual row
+  band deltas and packet cohort widths; no padding or forced cohort schedule.
+  Actual packet frontiers47+index bind independent output progression. State/
+  prefix cohorts have separate counters and are not reconstructed from SSE.
+- Text TTFT and actual single-ID TTFT remain distinct. Keep offered/client-
+  backlog TTFT and service TTFT. Individual-token ITL requires complete actual
+  single-ID delivery; coalesced/unresolved rows are counted, never replaced by
+  SSE chunk or meanITL proxies. Zero/one output ID has no ITL interval. Native
+  commit observations remain a separate host clock, not device timestamps.
+- Report offered/completed/usage throughput and all raw paired latencies plus
+  local-label fairness, sampled memory, actual source counters. Four samples per
+  label/band yield descriptive medians, no confidence-certified performance win.
+  Eight finite simultaneous offers measure a burst, not sustained capacity.
+  No source-correct SLO goodput, physical committed-token, fullHF, production
+  graph or vLLM performance/parity claim follows.
+
+ROOT sequence (not executed):
+
+1. Complete/review current14worker/12HTTP source-frozen gate. Run the four pure
+   `test_contracts.py` and four MOCK-only `test_owner_control.py` fixtures, then
+   actual harmless-owner TERM/INT-before-Popen/during-readiness/during-
+   request/during-cleanup failure-injection fixtures. Prior shadow/root evidence
+   is not a test of this new adapter. Review all new source and bindings.
+2. Set `ZG_BENCH_HOST_ROOT` to the original repository. Run
+   `prepare_native_ab.py --hardware-protocol ABS_ACCEPTED_V8_PROTOCOL
+   --hardware-protocol-sha256 SHA --accepted-serving-sha256 SHA
+   --new-output ABS_THIS_RUN/root-plan-1`. That performs ROOT-only source scans,
+   exact launch normalization and completed gate checks; no engine starts.
+3. Under the original native-gpu user/network isolation, invoke Python with the
+   original reviewed private-loopback helper (which execs Python), then
+   `run_native_ab.py --run --plan ABS/root-plan-1/plan.json
+   --plan-sha256 SHA --new-output ABS_THIS_RUN/root-native-1`.
+4. `validate_native_ab.py --output ABS_THIS_RUN/root-native-1`. Independent ROOT
+   review of full raw results/actual owner/point-memory/ITL coverage is required.
+   Profiling/hardware counts and unprofiled timing remain separate runs.
--- /dev/null
+++ TODO.md
@@ -0,0 +1,16 @@
+# Native before/after preparation
+
+Parent: serving-and-speculation; owner: batch_design_review.
+Owned: this new ignored derivative only; V8/old baseline/current456 unchanged.
+
+- [x] Explicit native/native identities, exact existing launch/workload/caps.
+- [x] Same canonical benchmark, complete32-sample/144-offer/32-warmup accounting.
+- [x] Require completed actual candidate worker14/HTTP12 + prior source gates.
+- [x] Prepare original owner cleanup +record-only TERM/INT checkpoints.
+- [x] Full exact output/counter/band/ITL coverage and independent validation.
+- [ ] ROOT/peer source review; actual pure/ownership failure fixtures.
+- [ ] ROOT actual preparation +complete unprofiled native/native run+validation.
+- [ ] Independent physical kernel diagnostic and measured performance acceptance.
+
+Source-only AST and small source/metadata hashes by author. No tests, imports,
+builds, native/provider/model/GPU execution or payload hashes/results.
--- /dev/null
+++ bindings.json
@@ -0,0 +1,170 @@
+{
+  "binaries_models_DSO_payloads_hashed_by_author": false,
+  "canonical_prompt_token_ids": [
+    248045,
+    846,
+    198,
+    22791,
+    411,
+    1103,
+    314,
+    24959,
+    11,
+    18101,
+    539,
+    73982,
+    11,
+    364,
+    1599,
+    777,
+    3665,
+    25,
+    220,
+    16,
+    11,
+    220,
+    17,
+    11,
+    220,
+    18,
+    11,
+    220,
+    19,
+    11,
+    220,
+    20,
+    11,
+    220,
+    21,
+    11,
+    220,
+    22,
+    11,
+    220,
+    23,
+    11,
+    248046,
+    198,
+    248045,
+    74455,
+    198
+  ],
+  "completed_prerequisites": [
+    {
+      "bytes": 86320,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-composed-source-freeze-1.json",
+      "sha256": "c516760006fcefb85f1a2374c101a0eece8893b87e8d0a3da32b60967fe82d68"
+    },
+    {
+      "bytes": 3898,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-serving-build-1.json",
+      "sha256": "0412830ff345ea9b7bc5e3847c0205441929ba94202ad8db4d813c309aece2c8"
+    },
+    {
+      "bytes": 49679,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-full-model-matrix-1.json",
+      "sha256": "f32eb4f473673ab6375b6a24cb4b7d02da221b8cd59c03ff4511b48746c644ac"
+    },
+    {
+      "bytes": 13270,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-operator-matrix-1.json",
+      "sha256": "630cab9c32c2fc22a8e23ee1b000eada9016222bde8a0a04aa2b333b9572fc4f"
+    },
+    {
+      "bytes": 1015,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-full-check-result-1.json",
+      "sha256": "780af4da6d5e18950a422de3b1677c4076adddd5ec2c3ea95f67215cc9481a1a"
+    },
+    {
+      "bytes": 1755,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/accepted-native-baseline.json",
+      "sha256": "26553886c58b204b4807c954fd157d272e7f59df09e05f7142e0de9d3d93c3ec"
+    },
+    {
+      "bytes": 986,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-tokenizer-08b-solo-1.json",
+      "sha256": "a90cb27c8e7a5a40e4aa9eb914644eec8784d9566b3ad74a36b655abfd6b48eb"
+    },
+    {
+      "bytes": 494,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-tokenizer-2b-solo-v8-1.json",
+      "sha256": "5bf9c727ec018f66c52713d09e4cda7da2895a896655235583fe185fe7e4c334"
+    },
+    {
+      "bytes": 838,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-parity-root-2026-10-19/graph-rope-final-http-server-artifacts.json",
+      "sha256": "0ca8aa6c01915e3bcd493cf8677662a6bb79e65eed94da38ed4d882793321756"
+    },
+    {
+      "bytes": 1018,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-parity-root-2026-10-19/graph-rope-composed-final-serving-result.json",
+      "sha256": "e76248e80361efe53d935074cb09f421c093713102ed72db58ba66b2ebc83fcc"
+    },
+    {
+      "bytes": 5014,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-parity-root-2026-10-19/graph-rope-attention-fixed-main-source.json",
+      "sha256": "d89cd668fee704c1f9ae7d9a36dfc4b2e899e2d73bf43139f653c994f27969cc"
+    }
+  ],
+  "executables": {
+    "native_baseline": {
+      "bytes": 36398264,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-parity-root-2026-10-19/graph-rope-final-frozen-server",
+      "sha256": "4a7455277b89aa48e92bd9d30b15a939935017368918c0e304a4f699687aae3f",
+      "source_id": "ROOT-source-freeze:d89cd668fee704c1f9ae7d9a36dfc4b2e899e2d73bf43139f653c994f27969cc"
+    },
+    "native_candidate": {
+      "bytes": 36566200,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-frozen-server-1",
+      "sha256": "87e1d727f908368f508eeb1e4dd53456f14348b8c72c4519cbee5e52b977c61e",
+      "source_id": "ROOT-source-freeze:c516760006fcefb85f1a2374c101a0eece8893b87e8d0a3da32b60967fe82d68"
+    }
+  },
+  "full_gate": {
+    "bytes": 1015,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-full-check-result-1.json",
+    "sha256": "780af4da6d5e18950a422de3b1677c4076adddd5ec2c3ea95f67215cc9481a1a"
+  },
+  "full_model_matrix": {
+    "bytes": 49679,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-full-model-matrix-1.json",
+    "sha256": "f32eb4f473673ab6375b6a24cb4b7d02da221b8cd59c03ff4511b48746c644ac"
+  },
+  "future_serving_gate_path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-serving-physical-2.json",
+  "future_serving_gate_sha256": "ROOT_MUST_SUPPLY_ACTUAL_COMPLETED_14_WORKER_12_HTTP_RECEIPT_SHA",
+  "models": {
+    "0.8b": {
+      "source_sha256": "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
+      "tokenizer_receipt": {
+        "bytes": 986,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-tokenizer-08b-solo-1.json",
+        "sha256": "a90cb27c8e7a5a40e4aa9eb914644eec8784d9566b3ad74a36b655abfd6b48eb"
+      }
+    },
+    "2b": {
+      "source_sha256": "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6",
+      "tokenizer_receipt": {
+        "bytes": 494,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-tokenizer-2b-solo-v8-1.json",
+        "sha256": "5bf9c727ec018f66c52713d09e4cda7da2895a896655235583fe185fe7e4c334"
+      }
+    }
+  },
+  "operator_matrix": {
+    "bytes": 13270,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-operator-matrix-1.json",
+    "sha256": "630cab9c32c2fc22a8e23ee1b000eada9016222bde8a0a04aa2b333b9572fc4f"
+  },
+  "schema": "zerogravity.native-before-after.antecedents.v1",
+  "serving_build": {
+    "bytes": 3898,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-serving-build-1.json",
+    "sha256": "0412830ff345ea9b7bc5e3847c0205441929ba94202ad8db4d813c309aece2c8"
+  },
+  "source_freeze": {
+    "bytes": 86320,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-composed-source-freeze-1.json",
+    "sha256": "c516760006fcefb85f1a2374c101a0eece8893b87e8d0a3da32b60967fe82d68"
+  },
+  "tokenizer_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
+}
--- /dev/null
+++ native_ab_common.py
@@ -0,0 +1,284 @@
+"""ROOT-only native/native contracts; no engine, operator or child on import."""
+import hashlib
+import importlib.util
+import json
+import math
+import os
+from pathlib import Path
+import stat
+import sys
+import urllib.request
+
+HERE = Path(__file__).resolve().parent
+ROOT = HERE.parents[2]
+PARENT = ROOT / '.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020'
+LABELS = ('native_baseline', 'native_candidate')
+MODELS = ('0.8b', '2b')
+BANDS = (('solo', 1), ('wave', 8))
+ORDER = (('native_baseline', 'native_candidate', 'native_candidate', 'native_baseline'),
+         ('native_candidate', 'native_baseline', 'native_baseline', 'native_candidate'))
+BASELINE = '4a7455277b89aa48e92bd9d30b15a939935017368918c0e304a4f699687aae3f'
+CANDIDATE = '87e1d727f908368f508eeb1e4dd53456f14348b8c72c4519cbee5e52b977c61e'
+V8 = 'b0c50e8a9d8d647464d3837748db0ba958a7c8294ee18140474703068fca0b33'
+SOURCE = 'c516760006fcefb85f1a2374c101a0eece8893b87e8d0a3da32b60967fe82d68'
+COUNTERS = ('completed_forward_kernel_dispatches', 'completed_model_token_steps',
+    'committed_gpu_roots', 'native_projection_dispatches_completed',
+    'native_sampler_dispatches_completed', 'accepted_output_tokens',
+    'gpu_fused_state_only_batches', 'gpu_fused_state_only_rows',
+    'gpu_fused_greedy_batches', 'gpu_fused_rows_completed', 'gpu_fused_batches',
+    'native_admission_refusals', 'discarded_ready_cohorts', 'failed_gpu_quanta',
+    'cancelled_requests')
+ZERO_COUNTERS = ('native_admission_refusals', 'discarded_ready_cohorts',
+                 'failed_gpu_quanta', 'cancelled_requests')
+MAX_TREE_BYTES = 512 * 1024**2
+MAX_TREE_FILES = 1024
+
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+
+def sha(path):
+    with Path(path).open('rb') as source:
+        return hashlib.file_digest(source, 'sha256').hexdigest()
+
+
+def read(path, cap=8 * 1024**2):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as source:
+        before = os.fstat(source.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular input')
+        raw = source.read(cap + 1)
+        after = os.fstat(source.fileno())
+    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
+    require(identity(before) == identity(after) and len(raw) == before.st_size, 'unchanged bounded input')
+    return raw
+
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON field')
+        result[key] = value
+    return result
+
+
+def load(path):
+    return json.loads(read(path), object_pairs_hook=unique)
+
+
+def pin(row):
+    raw = read(row['path'])
+    require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
+            'exact source/metadata pin: ' + row['path'])
+
+
+def file_row(path):
+    path = Path(path)
+    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))
+
+
+def tree(root):
+    total, files = 0, 0
+    for path in Path(root).rglob('*'):
+        require(not path.is_symlink(), 'no artifact symlinks')
+        if path.is_file():
+            total += path.stat().st_size
+            files += 1
+    require(total <= MAX_TREE_BYTES and files <= MAX_TREE_FILES, 'bounded complete artifact tree')
+    return dict(bytes=total, files=files, limit_bytes=MAX_TREE_BYTES, limit_files=MAX_TREE_FILES)
+
+
+def save(path, value, root=None):
+    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
+    require(len(raw) <= 8 * 1024**2, 'bounded receipt')
+    path = Path(path)
+    temporary = path.with_name(path.name + '.writing')
+    with temporary.open('xb') as output:
+        output.write(raw)
+    temporary.replace(path)
+    if root is not None:
+        tree(root)  # Includes THIS final receipt; no pre-write-only artifact claim.
+
+
+def install_opener():
+    class NoRedirect(urllib.request.HTTPRedirectHandler):
+        def redirect_request(self, *_args, **_kwargs):
+            raise ValueError('native redirects refused')
+    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()))
+
+
+def http_json(url):
+    with urllib.request.urlopen(url, timeout=2) as response:
+        raw = response.read(1024**2 + 1)
+        require(response.status == 200 and len(raw) <= 1024**2, 'bounded original native HTTP response')
+    return json.loads(raw, object_pairs_hook=unique)
+
+
+def source_freeze():
+    document = load(HERE / 'freeze-manifest.json')
+    for row in document['files']:
+        pin(dict(row, path=str(HERE / row['file'])))
+        pin(dict(row, path=str(HERE / 'frozen' / row['file'])))
+    for row in document['parent_sources_and_metadata']:
+        pin(row)
+    return sha(HERE / 'freeze-manifest.json')
+
+
+def load_path(name, path):
+    require(name not in sys.modules, 'exclusive helper module name')
+    spec = importlib.util.spec_from_file_location(name, path)
+    module = importlib.util.module_from_spec(spec)
+    sys.modules[name] = module
+    spec.loader.exec_module(module)
+    return module
+
+
+def parents():
+    # Exact immutable V8 modules are reused, never rewritten or relabeled as a
+    # second implementation. Native result labels belong only to this driver.
+    source_freeze()
+    require(str(ROOT.resolve()) == os.environ.get('ZG_BENCH_HOST_ROOT'), 'explicit original host repository')
+    sys.path.insert(0, str(PARENT))
+    import protocol_v8 as protocol
+    import prepare_launch_v8 as launch
+    import owned_container_v4 as owned
+    import observations_v8 as observations
+    import cgroup_readback_v3 as cgroup
+    for module in (protocol, launch, owned, observations, cgroup):
+        require(Path(module.__file__).resolve().parent == PARENT, 'exact parent module namespace')
+    pins = protocol.verify_preparation(V8)
+    protocol.verify_small_sources()
+    bench = load_path('native_ab_canonical_bench', ROOT / 'tools/serve-bench.py')
+    lifecycle = load_path('native_ab_canonical_log', ROOT / 'tools/serve-lifecycle.py')
+    return protocol, launch, owned, observations, cgroup, pins, bench, lifecycle
+
+
+def closed_case(row):
+    c, log = row['foreground_cleanup'], row['log']
+    require(row['status'].startswith('PASS_') and row['exit_code'] == 0 and row['cleanup_errors'] == [] and 'failure' not in row and
+            c['exit_code'] == 0 and c['stage_errors'] == [] and
+            all(c[k] is True for k in ('leader_reaped', 'group_absent', 'all_nonzero_signals_before_reap')) and
+            log['pipe_eof'] is True and log['closed'] is True, 'actual original owner retirement')
+
+
+def accepted_candidate(serving_sha):
+    binding = load(HERE / 'bindings.json')
+    for row in binding['completed_prerequisites']:
+        pin(row)
+    freeze_path = Path(binding['source_freeze']['path'])
+    pin(binding['source_freeze'])
+    source = load(freeze_path)
+    require(len(source['files']) == 456, 'complete candidate source closure')
+    for row in source['files']:
+        pin(dict(row, path=str(ROOT / row['file'])))
+    for name, count in (('operator_matrix', 10), ('full_model_matrix', 46)):
+        value = load(binding[name]['path'])
+        require(value['source_freeze_sha256'] == SOURCE and value['case_count'] == len(value['cases']) == count
+                and len({r['case'] for r in value['cases']}) == count and value['status'].startswith('PASS_'),
+                'complete candidate numerical matrix')
+        for row in value['cases']:
+            closed_case(row)
+    full = load(binding['full_gate']['path'])
+    require(full['status'] == 'PASS_FULL_ISOLATED_AND_BOTH_APPROVED_ORIGINAL_ROUTES' and
+            full['source_freeze_sha256'] == SOURCE and full['all456_sources_post_gate_exact'] is True,
+            'candidate full gate and both original routes')
+    build = load(binding['serving_build']['path'])
+    require(build['source_freeze_sha256'] == SOURCE and all(s['exit_code'] == 0 for s in build['steps']) and
+            build['artifacts']['server']['sha256'] == CANDIDATE, 'candidate strict/frozen release build')
+    gate_path = Path(binding['future_serving_gate_path'])
+    require(len(serving_sha) == 64 and sha(gate_path) == serving_sha, 'ROOT binds completed worker/HTTP receipt')
+    gate = load(gate_path)
+    require(gate['status'] == 'PASS_FROZEN_WORKER14_HTTP12_REQUIRES_ROOT_REVIEW' and
+            gate['source_freeze_sha256'] == SOURCE and len(gate['cases']) == 2 and
+            gate['artifacts']['server'] == CANDIDATE and gate['performance_benchmark'] is False,
+            'pending/partial serving gate is refused')
+    for row, label, count in zip(gate['cases'], ('worker', 'http'), (14, 12)):
+        closed_case(row)
+        require(row['case'] == label and row['expected_tests'] == count and
+                row['status'] == 'PASS_ORIGINAL_PHYSICAL_SERVING_REGRESSION', 'all worker14/HTTP12 cases')
+        log = gate_path.parent / f'root-serving-{label}-physical-2.log'
+        raw = read(log)
+        require(len(raw) == row['log']['bytes'] and hashlib.sha256(raw).hexdigest() == row['log']['sha256'] and
+                f'{count} passed; 0 failed; 0 ignored' in raw.decode(), 'actual complete serving test summary')
+    return dict(serving_gate=file_row(gate_path), source_freeze=binding['source_freeze'],
+                completed_prerequisites=binding['completed_prerequisites'])
+
+
+def counters(before, after):
+    values = []
+    for value in (before, after):
+        a = value['admission']
+        require(a['kind'] == 'resident_gpu' and a['batch_mode'] == 'fused' and
+                a['native_batch_rows_max'] == 8 and a['active_sessions'] == a['queued_requests'] == 0,
+                'same native engine/band and quiescent endpoints')
+        row = {name: a[name] for name in COUNTERS}
+        row.update({'band' + key: a['gpu_fused_batches_by_rows'][key] for key in ('2', '4', '8')})
+        require(all(type(v) is int and v >= 0 for v in row.values()), 'actual native integer counters, no absent defaults')
+        values.append(row)
+    left, right = values
+    require(all(right[k] >= left[k] for k in left), 'monotonic same-process counters')
+    delta = {key: right[key] - left[key] for key in left}
+    require(all(delta[k] == 0 for k in ZERO_COUNTERS), 'failed/refused/discarded/cancelled work is not accepted')
+    require(delta['gpu_fused_batches'] == sum(delta['band' + k] for k in ('2', '4', '8')) ==
+            delta['gpu_fused_state_only_batches'] + delta['gpu_fused_greedy_batches'] and
+            delta['gpu_fused_rows_completed'] == sum(int(k) * delta['band' + k] for k in ('2', '4', '8')) and
+            delta['native_sampler_dispatches_completed'] == 4 * delta['gpu_fused_greedy_batches'],
+            'actual native phase/band/row/sampler conservation')
+    require(2 * delta['gpu_fused_state_only_batches'] <= delta['gpu_fused_state_only_rows'] <=
+            8 * delta['gpu_fused_state_only_batches'] and
+            2 * delta['gpu_fused_greedy_batches'] <= delta['gpu_fused_rows_completed'] -
+            delta['gpu_fused_state_only_rows'] <= 8 * delta['gpu_fused_greedy_batches'],
+            'native phase rows stay within actual supported bands')
+    require(delta['completed_forward_kernel_dispatches'] > 0 and delta['completed_model_token_steps'] > 0 and
+            delta['committed_gpu_roots'] > 0, 'genuine complete GPU workload, not counter presence')
+    return delta
+
+
+def signature(sample, expected):
+    require(sample['offered_requests'] == sample['completed_requests'] == len(expected) and
+            sample['unsuccessful_requests'] == sample['completions_without_usage'] == 0 and
+            len(sample['requests']) == len(expected), 'all canonical offers completed with usage')
+    rows = {}
+    for row in sample['requests']:
+        require(row['request_id'] not in rows, 'unique complete workload membership')
+        original = next((x for x in expected if x['request_id'] == row['request_id']), None)
+        require(original is not None and row['prompt_sha256'] == hashlib.sha256(original['prompt'].encode()).hexdigest()
+                and row['max_tokens'] == original['max_tokens'] and row['completed'] is True,
+                'original prompt/cap/request identity')
+        obs = row['wire_token_observation']
+        require(obs['status'] == 'provider_bound_complete_ids' and type(row['usage_reported_completion_tokens']) is int
+                and len(obs['token_ids']) == row['usage_reported_completion_tokens'] <= row['max_tokens']
+                and all(type(token) is int and 0 <= token < 248320 for token in obs['token_ids'])
+                and row['finish_reason'] in ('stop', 'length') and type(row['output_sha256']) is str and
+                len(row['output_sha256']) == 64 and set(row['output_sha256']) <= set('0123456789abcdef'),
+                'full response-bound IDs/usage/finish/text hash')
+        rows[row['request_id']] = {key: row[key] for key in ('prompt_sha256', 'max_tokens',
+            'usage_reported_completion_tokens', 'finish_reason', 'output_sha256')}
+        rows[row['request_id']]['token_ids'] = obs['token_ids']
+    return rows
+
+
+def finite(value):
+    return type(value) in (int, float) and math.isfinite(value) and value >= 0
+
+
+def model_sources(pins):
+    # ROOT-only complete scans, outside every timed sample; same V8 source path.
+    from support_v3 import source_check
+    for model in MODELS:
+        row = pins['models'][model]
+        source_check(pins, model, row['source_path'], row['gguf']['path'])
+
+
+def validate_context(value, parent):
+    require(value['schema'] == 'zerogravity.native-before-after.launch-context.v1' and
+            (value['workload_kind'], value['concurrencies']) in (('solo', [1]), ('wave', [8])),
+            'actual native/native C1 or C8 population, no inherited C2/C4 claim')
+    # Parent's fixed multirow comparison population is irrelevant to launch().
+    # Reuse its unchanged policy/resource/whole-workload validation only, with
+    # its own comparison-population fields in a temporary validation object.
+    checked = dict(value, schema=parent.SCHEMA,
+                   concurrencies=[1] if value['workload_kind'] == 'solo' else [2, 4, 8])
+    parent.validate_protocol(checked)
+    return value
--- /dev/null
+++ native_ab_owner.py
@@ -0,0 +1,133 @@
+"""One original native container/foreground group; no engine or alternate queue."""
+import os
+import secrets
+import subprocess
+import time
+
+from native_ab_common import http_json, require, save
+
+
+class NativeOwner:
+    def __init__(self, label, directory, command, protocol, dependencies, cancellation, artifact_root):
+        self.label, self.directory, self.command, self.protocol = label, directory, command, protocol
+        _, _, self.owned, self.observations, self.cgroup, _, self.bench, self.lifecycle = dependencies
+        self.cancel, self.artifact_root = cancellation, artifact_root
+        self.nonce = secrets.token_hex(16)
+        self.name = 'zg-native-ab-' + label.replace('_', '-') + '-' + self.nonce
+        self.cidfile = directory / 'cid'
+        self.process = self.log = self.monitor = None
+        self.directory.mkdir(mode=0o700)
+        self.receipt = dict(label=label, native_implementation='zerogravity',
+            status='UNQUALIFIED', owned_name=self.name, owned_nonce=self.nonce,
+            executable_sha256=protocol['zerogravity_binary_sha256'], cleanup_errors=[],
+            policy='warmed_process', primary_failure=None, profiler_enabled=False)
+        self.closed = False
+        self.save()
+
+    def save(self):
+        save(self.directory / 'receipt.json', self.receipt, self.artifact_root)
+
+    def live(self):
+        self.cancel.check()
+        require(self.process is not None and self.owned.leader_status_unreaped(self.process) is None,
+                'original owned foreground remains live and unreaped')
+        require(self.log is None or self.log.error is None, 'bounded native log remains operational')
+
+    def start(self):
+        address = ('127.0.0.1', self.protocol['ports'][0])
+        require(not self.lifecycle.listening(address), 'fresh native endpoint; never adopt an existing server')
+        self.receipt['GPU_quiet_before'] = self.observations.quiet(
+            self.protocol['gpu_uuid'], self.protocol['gpu_total_mib'])
+        replacements = {'{owned_name}': self.name, '{owned_nonce}': self.nonce,
+                        '{cidfile}': str(self.cidfile), '{sample_directory}': str(self.directory)}
+        command = self.command
+        for key, value in replacements.items():
+            command = [part.replace(key, value) for part in command]
+        self.receipt['argv'] = command
+        # Record-only TERM/INT handler cannot raise across this ownership handoff.
+        self.cancel.check()
+        self.process = subprocess.Popen(command, cwd=str(self.protocol_root()),
+            env={**os.environ, 'ZG_EXPECTED_GPU_UUID': self.protocol['gpu_uuid']},
+            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
+            start_new_session=True, bufsize=0)
+        self.receipt['owned_cli_pid'] = self.process.pid
+        self.log = self.lifecycle.BoundedLog(self.process.stdout, self.directory / 'execution.log', 8 * 1024**2)
+        self.monitor = self.observations.Monitor(self.protocol['gpu_uuid'], self.name, self.nonce, self.cidfile)
+        self.save()
+        deadline = time.monotonic() + 600
+        while time.monotonic() < deadline:
+            self.live()
+            try:
+                models = http_json(self.endpoint() + '/v1/models')
+                require(self.alias() in [row['id'] for row in models['data']], 'actual original native model alias')
+            except Exception as error:
+                self.receipt['last_readiness_error'] = repr(error)[:512]
+                time.sleep(.1)
+                continue
+            self.live()
+            require(self.owned.owned_container(self.name, self.nonce, self.cidfile) is not None,
+                    'original authenticated nonce/CID/container before readiness')
+            from native_ab_common import load
+            bootstrap = load(self.directory / 'bootstrap.json')
+            require(bootstrap['provider'] == 'zerogravity' and bootstrap['model'] == self.protocol['model'] and
+                    bootstrap['stock_math_modified'] is False and bootstrap['policy_observer'] is False,
+                    'genuine native bootstrap, not a vLLM slot')
+            self.cgroup.validate_caps(bootstrap['cgroup_readback'], self.protocol['cpu_affinity'])
+            self.receipt['actual_bootstrap'] = bootstrap
+            self.receipt['ready'] = True
+            self.save()
+            return
+        raise TimeoutError('finite original native startup; no fallback')
+
+    def protocol_root(self):
+        from native_ab_common import ROOT
+        return ROOT
+
+    def endpoint(self):
+        return 'http://127.0.0.1:' + str(self.protocol['ports'][0])
+
+    def alias(self):
+        return 'qwen35-' + self.protocol['model'] + '-same-gguf'
+
+    def close(self):
+        if self.closed:
+            require(self.receipt['status'] == 'OWNERS_CLOSED', 'prior native cleanup failed')
+            return
+        self.closed = True
+        errors = self.receipt['cleanup_errors']
+        if self.monitor is not None:
+            try:
+                self.receipt['physical_observations'] = self.monitor.finish()
+            except BaseException as error:
+                errors.append('physical observer: ' + repr(error))
+        if self.process is not None:
+            try:
+                errors += self.owned.cleanup(self.process, self.name, self.nonce, self.cidfile, self.receipt)
+            except BaseException as error:
+                errors.append('original container/group cleanup: ' + repr(error))
+        if self.log is not None:
+            try:
+                result = self.log.finish(5)
+                self.receipt['log_cleanup'] = result
+                require(result['reader_complete'] and result['reader_error'] is None and
+                        result['truncated'] is False and result['bytes_seen'] <= 8 * 1024**2,
+                        'actual complete bounded log and pipe EOF')
+            except BaseException as error:
+                errors.append('log cleanup: ' + repr(error))
+        elif self.process is not None and self.process.stdout is not None:
+            try:
+                self.process.stdout.close()
+            except BaseException as error:
+                errors.append('pipe close: ' + repr(error))
+        if self.process is not None:
+            try:
+                ceiling = self.receipt['GPU_quiet_before'][-1]['memory_used_mib'] + 64
+                self.receipt['GPU_quiet_after'] = self.observations.quiet(
+                    self.protocol['gpu_uuid'], self.protocol['gpu_total_mib'], ceiling)
+            except BaseException as error:
+                errors.append('GPU retirement: ' + repr(error))
+        if not errors:
+            self.receipt['status'] = 'OWNERS_CLOSED'
+        # This I/O cannot skip any preceding independent cleanup stage.
+        self.save()
+        require(not errors, 'native cleanup/evidence failed; no next sample')
--- /dev/null
+++ prepare_native_ab.py
@@ -0,0 +1,98 @@
+"""ROOT ONLY: new native/native plan; does not start a provider."""
+import argparse
+import copy
+import os
+from pathlib import Path
+
+from native_ab_common import (HERE, LABELS, MODELS, BANDS, BASELINE, CANDIDATE,
+    V8, SOURCE, accepted_candidate, file_row, load, parents, require, save, sha, source_freeze, validate_context)
+
+
+def protocols(reference, bindings, protocol):
+    result = []
+    for model in MODELS:
+        for workload, concurrency in BANDS:
+            pair = {}
+            for label in LABELS:
+                value = copy.deepcopy(reference)
+                value.update(schema='zerogravity.native-before-after.launch-context.v1',
+                    model=model, route='eager', policy='warmed_process',
+                    workload_kind=workload, max_tokens=8, concurrencies=[concurrency],
+                    requests=protocol.workload(workload, 8), max_pending=1 if workload == 'solo' else 8,
+                    source_sha256=bindings['models'][model]['source_sha256'],
+                    tokenizer_sha256=bindings['tokenizer_sha256'],
+                    zerogravity_binary=bindings['executables'][label]['path'],
+                    zerogravity_binary_sha256=bindings['executables'][label]['sha256'],
+                    zerogravity_source_id=bindings['executables'][label]['source_id'])
+                validate_context(value, protocol)
+                pair[label] = value
+            result.append(dict(model=model, workload=workload, concurrency=concurrency, protocols=pair))
+    return result
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--hardware-protocol', type=Path, required=True)
+    parser.add_argument('--hardware-protocol-sha256', required=True)
+    parser.add_argument('--accepted-serving-sha256', required=True)
+    parser.add_argument('--new-output', type=Path, required=True)
+    args = parser.parse_args()
+    freeze = source_freeze()
+    protocol, launcher, _, _, _, pins, _, _ = parents()
+    candidate = accepted_candidate(args.accepted_serving_sha256)
+    require(sha(args.hardware_protocol) == args.hardware_protocol_sha256, 'ROOT binds actual hardware protocol')
+    reference = protocol.validate_protocol(load(args.hardware_protocol))
+    require(reference['zerogravity_binary_sha256'] == BASELINE and reference['route'] == 'eager' and
+            set(reference['cpu_affinity']) <= os.sched_getaffinity(0), 'accepted native baseline hardware and exact four CPUs')
+    binding = load(HERE / 'bindings.json')
+    for label, digest in zip(LABELS, (BASELINE, CANDIDATE)):
+        row = binding['executables'][label]
+        require(row['sha256'] == digest and Path(row['path']).stat().st_size == row['bytes'] and
+                sha(row['path']) == digest, 'ROOT verifies actual frozen native executable')
+    from support_v3 import source_check
+    projects = protocol.read_json(protocol.HERE / 'projector-pins.json')['models']
+    for model in MODELS:
+        source = pins['models'][model]
+        require(source['gguf']['sha256'] == binding['models'][model]['source_sha256'], 'original source weights')
+        source_check(pins, model, source['source_path'], source['gguf']['path'])
+        require(sha(Path(projects[model]['conversion_directory']) / 'mmproj-bf16.gguf') ==
+                projects[model]['projector_sha256'], 'unchanged existing V8 readonly mount')
+        for name in ('config.json', 'tokenizer.json'):
+            expected = next(r['sha256'] for r in source['source_files'] if r['name'] == name)
+            require(sha(protocol.host_root() / 'target/models' / ('qwen35-' + model) / name) == expected,
+                    'same original native sidecar')
+        tokenizer = load(binding['models'][model]['tokenizer_receipt']['path'])
+        require(tokenizer['tokenizer_sha256'] == pins['tokenizer_sha256'] and tokenizer['eos_token_id'] == 248046 and
+                len(tokenizer['records']) == 1 and tokenizer['records'][0]['prompt_token_ids'] ==
+                binding['canonical_prompt_token_ids'] and len(binding['canonical_prompt_token_ids']) == 47,
+                'actual original canonical47 IDs; no new rendering or tokenizer math')
+    cases = protocols(reference, binding, protocol)
+    for case in cases:
+        case['commands'] = {}
+        for label, value in case['protocols'].items():
+            case['commands'][label] = launcher.launch(value, pins, V8, 'zerogravity',
+                Path(projects[case['model']]['conversion_directory']))
+        def normalized(label):
+            executable = binding['executables'][label]
+            return [part.replace(executable['path'], '<NATIVE_EXECUTABLE>')
+                    .replace(executable['sha256'], '<NATIVE_EXECUTABLE_SHA256>')
+                    for part in case['commands'][label]]
+        require(normalized(LABELS[0]) == normalized(LABELS[1]), 'same V8 launch except native executable and hash')
+    output = args.new_output.resolve()
+    require(output.parent == HERE and not output.exists(), 'fresh own plan only')
+    output.mkdir(mode=0o700)
+    plan = dict(schema='zerogravity.native-before-after.plan.v1', source_freeze_sha256=freeze,
+        parent_v8_sha256=V8, candidate_gate=candidate, hardware_reference=file_row(args.hardware_protocol),
+        labels=list(LABELS), policy='warmed_process', route='eager', max_tokens=8,
+        models=list(MODELS), cases=cases, repetitions=2, measured_samples=32,
+        measured_offers=144, warmup_requests=32, profile='same accepted native GGUF SourceV1',
+        canonical_prompt_token_ids=binding['canonical_prompt_token_ids'],
+        native_source_freeze_sha256=SOURCE, profiler_enabled=False, vllm_provider_present=False,
+        performance_or_source_correct_goodput_qualified=False)
+    save(output / 'plan.json', plan, output)
+    require(source_freeze() == freeze, 'source preparation immutable')
+    print(sha(output / 'plan.json'))
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ run_native_ab.py
@@ -0,0 +1,214 @@
+"""ROOT ONLY explicit native baseline/candidate; never a vLLM comparison slot."""
+import argparse
+from dataclasses import asdict
+import fcntl
+import os
+from pathlib import Path
+import socket
+import struct
+import time
+
+from native_ab_common import (HERE, LABELS, MODELS, BANDS, ORDER, V8, accepted_candidate,
+    counters, file_row, http_json, install_opener, load, model_sources, parents, require, save, sha,
+    signature, source_freeze, tree, validate_context)
+from native_ab_owner import NativeOwner
+from sigterm_cancellation import SigtermCancellation
+
+
+def private_namespace():
+    mapping = Path('/proc/self/uid_map').read_text().split()
+    require(len(mapping) == 3 and mapping[0] == '0' and mapping[1] != '0' and mapping[2] == '1',
+            'original native-gpu private mapped user namespace required')
+    network = os.open('/proc/self/ns/net', os.O_RDONLY)
+    try:
+        owner = fcntl.ioctl(network, 0xB701)
+        try:
+            require(os.fstat(owner).st_ino == os.stat('/proc/self/ns/user').st_ino,
+                    'network namespace belongs to this private user namespace')
+        finally:
+            os.close(owner)
+    finally:
+        os.close(network)
+    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
+        flags = fcntl.ioctl(control, 0x8913, struct.pack('16sH22x', b'lo', 0))
+        require(struct.unpack_from('H', flags, 16)[0] & 1, 'use original reviewed private-loopback helper first')
+    return dict(uid_map=Path('/proc/self/uid_map').read_text(),
+        user_namespace=os.readlink('/proc/self/ns/user'), network_namespace=os.readlink('/proc/self/ns/net'))
+
+
+def quiescent(owner):
+    deadline, previous = time.monotonic() + 15, None
+    while time.monotonic() < deadline:
+        owner.live()
+        value = http_json(owner.endpoint() + '/health')
+        admission = value['admission']
+        if admission['active_sessions'] == admission['queued_requests'] == 0:
+            if value == previous:
+                return value
+            previous = value
+        else:
+            previous = None
+        time.sleep(.1)
+    raise TimeoutError('original GPU worker failed bounded quiescence')
+
+
+def validate_plan(plan, protocol, launcher, pins, binding):
+    require(plan['schema'] == 'zerogravity.native-before-after.plan.v1' and
+            plan['source_freeze_sha256'] == source_freeze() and plan['parent_v8_sha256'] == V8 and
+            plan['labels'] == list(LABELS) and plan['policy'] == 'warmed_process' and
+            plan['route'] == 'eager' and plan['max_tokens'] == 8 and plan['models'] == list(MODELS) and
+            plan['repetitions'] == 2 and plan['measured_samples'] == 32 and
+            plan['measured_offers'] == 144 and plan['warmup_requests'] == 32 and
+            plan['profiler_enabled'] is False and plan['vllm_provider_present'] is False and
+            plan['canonical_prompt_token_ids'] == binding['canonical_prompt_token_ids'] and
+            plan['performance_or_source_correct_goodput_qualified'] is False,
+            'complete declared native/native plan')
+    require(len(plan['cases']) == 4 and [(c['model'], c['workload'], c['concurrency']) for c in plan['cases']] ==
+            [(model, workload, concurrency) for model in MODELS for workload, concurrency in BANDS],
+            'both models and C1/C8 exactly once')
+    projects = protocol.read_json(protocol.HERE / 'projector-pins.json')['models']
+    for case in plan['cases']:
+        require(set(case['commands']) == set(case['protocols']) == set(LABELS), 'no foreign provider slots')
+        for label in LABELS:
+            p = validate_context(case['protocols'][label], protocol)
+            e = binding['executables'][label]
+            require(p['model'] == case['model'] and p['workload_kind'] == case['workload'] and
+                    p['policy'] == 'warmed_process' and p['route'] == 'eager' and p['max_tokens'] == 8 and
+                    p['zerogravity_binary'] == e['path'] and p['zerogravity_binary_sha256'] == e['sha256'] and
+                    p['source_sha256'] == binding['models'][case['model']]['source_sha256'] and
+                    p['tokenizer_sha256'] == binding['tokenizer_sha256'] and
+                    case['commands'][label] == launcher.launch(p, pins, V8, 'zerogravity',
+                        Path(projects[case['model']]['conversion_directory'])),
+                    'exact source-reviewed native V8 launch, no flag/math substitution')
+        left, right = [dict(case['protocols'][label]) for label in LABELS]
+        for value in (left, right):
+            for key in ('zerogravity_binary', 'zerogravity_binary_sha256', 'zerogravity_source_id'):
+                value.pop(key)
+        require(left == right, 'only native executable/source identity differs within a pair')
+    return plan
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--plan', type=Path, required=True)
+    parser.add_argument('--plan-sha256', required=True)
+    parser.add_argument('--new-output', type=Path, required=True)
+    parser.add_argument('--run', action='store_true')
+    args = parser.parse_args()
+    require(args.run, 'ROOT must explicitly run after review and serving gates')
+    freeze = source_freeze()
+    require(sha(args.plan) == args.plan_sha256 and args.plan.resolve().parent.parent == HERE,
+            'own exclusive prepared plan and actual SHA')
+    deps = parents()
+    protocol, launcher, _, _, _, pins, bench, _ = deps
+    binding = load(HERE / 'bindings.json')
+    plan = validate_plan(load(args.plan), protocol, launcher, pins, binding)
+    gate_sha = plan['candidate_gate']['serving_gate']['sha256']
+    require(accepted_candidate(gate_sha) == plan['candidate_gate'], 'all accepted candidate gates unchanged')
+    for row in binding['executables'].values():
+        require(sha(row['path']) == row['sha256'], 'actual immutable native executable before run')
+    model_sources(pins)
+    output = args.new_output.resolve()
+    require(output.parent == HERE and not output.exists(), 'fresh whole native/native run')
+    output.mkdir(mode=0o700)
+    install_opener()
+    namespace = private_namespace()
+    allowed = os.sched_getaffinity(0)
+    cpus = plan['cases'][0]['protocols'][LABELS[0]]['cpu_affinity']
+    require(set(cpus) <= allowed, 'same four available logical CPUs')
+    os.sched_setaffinity(0, cpus)
+    report = dict(schema='zerogravity.native-before-after.run.v1', status='RUNNING_UNQUALIFIED',
+        plan=file_row(args.plan), source_freeze_sha256=freeze, samples=[],
+        namespace=namespace, native_only=True, main_timing_profiled=False,
+        performance_or_vllm_or_source_correct_goodput_qualified=False)
+    save(output / 'report.json', report, output)
+    try:
+        with SigtermCancellation() as cancellation:
+            exact_output_by_model = {}
+            for case in plan['cases']:
+                expected = protocol.workload(case['workload'], 8)
+                for repetition, order in enumerate(ORDER):
+                    for position, label in enumerate(order):
+                        cancellation.check()
+                        index = len(report['samples'])
+                        directory = output / f'{index:03d}-{case["model"]}-c{case["concurrency"]}-{label}'
+                        p = case['protocols'][label]
+                        owner = NativeOwner(label, directory, case['commands'][label], p, deps, cancellation, output)
+                        primary = None
+                        try:
+                            owner.start()
+                            owner.live()
+                            # Exactly ONE ordinary canonical request, no width-specific warmup.
+                            warm = bench.stream_one(owner.endpoint(), owner.alias(), expected[0]['prompt'], 8, 180,
+                                None, request_id='native-ab-warmup-' + str(index), return_token_ids=True)
+                            owner.receipt['warmup'] = asdict(warm)
+                            owner.save()
+                            require(warm.completed, 'ordinary exact canonical warmup completed')
+                            owner.live()
+                            before = quiescent(owner)
+                            measured = bench.bench(owner.endpoint(), owner.alias(), expected[0]['prompt'],
+                                len(expected), case['concurrency'], 8, 180, None, workload=expected,
+                                ttft_slo_seconds=10, itl_slo_seconds=2, max_pending=len(expected),
+                                launch_deadline_seconds=180, return_token_ids=True,
+                                snapshot_url=owner.endpoint() + '/health')
+                            save(directory / 'raw-measurement.json', measured, output)
+                            owner.live()
+                            after = quiescent(owner)
+                            save(directory / 'quiescent.json', dict(before=before, after=after), output)
+                            sig = signature(measured, expected)
+                            delta = counters(before, after)
+                            require(delta['accepted_output_tokens'] == sum(len(v['token_ids']) for v in sig.values()),
+                                    'native accepted packet output conservation; warmup excluded')
+                            if case['concurrency'] == 1:
+                                require(delta['gpu_fused_batches'] == delta['native_projection_dispatches_completed'] ==
+                                        delta['native_sampler_dispatches_completed'] == 0,
+                                        'single independent request has no padded native batch')
+                            measurement = dict(label=label, model=case['model'], workload=case['workload'],
+                                concurrency=case['concurrency'], repetition=repetition, order_position=position,
+                                paired_order='ABBA' if repetition == 0 else 'BAAB',
+                                benchmark=measured, quiescent_before=before, quiescent_after=after,
+                                source_counter_delta=delta, source_counter_unit='completed source launchsites, not CUDA activity',
+                                signature=sig)
+                            save(directory / 'measurement.json', measurement, output)
+                            owner.receipt['measurement'] = file_row(directory / 'measurement.json')
+                            owner.save()
+                            # Same source/prompt/policy: preserve first actual full result,
+                            # reject any later per-row/band/version discrepancy immediately.
+                            exact = exact_output_by_model.setdefault(case['model'], sig['request-0'])
+                            require(all(value == exact for value in sig.values()),
+                                    'native before/after full IDs/text/usage/finish regression; raw result retained')
+                        except BaseException as error:
+                            primary = repr(error)
+                            owner.receipt['primary_failure'] = primary
+                            raise
+                        finally:
+                            # No later sample starts after primary, cleanup or receipt I/O failure.
+                            try:
+                                owner.close()
+                            except BaseException as error:
+                                raise RuntimeError('native owner cleanup/evidence failed; primary=' + repr(primary) +
+                                                   '; cleanup=' + repr(error)) from error
+                            finally:
+                                report['samples'].append(dict(label=label, model=case['model'],
+                                    concurrency=case['concurrency'], repetition=repetition, order_position=position,
+                                    receipt=file_row(directory / 'receipt.json')))
+                                save(output / 'report.json', report, output)
+            cancellation.check()
+            require(len(report['samples']) == 32, 'whole planned32 processes completed')
+            require(source_freeze() == freeze and accepted_candidate(gate_sha) == plan['candidate_gate'],
+                    'all source/gates unchanged after complete native run')
+            for row in binding['executables'].values():
+                require(sha(row['path']) == row['sha256'], 'immutable native binary after run')
+            model_sources(pins)
+            report['status'] = 'RAW_NATIVE_BASELINE_CANDIDATE_COMPLETE_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        report['failure'] = repr(error)[:2048]
+        raise
+    finally:
+        os.sched_setaffinity(0, allowed)
+        save(output / 'report.json', report, output)
+        tree(output)
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ sigterm_cancellation.py
@@ -0,0 +1,47 @@
+"""SIGTERM/SIGINT checkpoints around the original bounded log drain, not new ownership."""
+import signal
+
+
+class SigtermCancellation:
+    def __init__(self):
+        self.received = False
+        self.previous = {}
+
+    def _record(self, signum, frame):
+        # Never asynchronously raise across Popen assignment or retirement.
+        self.received = True
+
+    def __enter__(self):
+        for signum in (signal.SIGTERM, signal.SIGINT):
+            self.previous[signum] = signal.signal(signum, self._record)
+        return self
+
+    def __exit__(self, kind, value, traceback):
+        for signum, previous in self.previous.items():
+            signal.signal(signum, previous)
+        return False
+
+    def exception(self):
+        return InterruptedError("interrupted: retire original owned serving regression before failure")
+
+    def check(self):
+        if self.received:
+            raise self.exception()
+
+    def observe(self, log, process, seconds):
+        original_drain = log.drain
+
+        def checked_drain(timeout):
+            self.check()
+            original_drain(timeout)
+            self.check()
+
+        self.check()
+        log.drain = checked_drain
+        try:
+            result = log.observe(process, seconds)
+            self.check()
+            return result
+        finally:
+            # Original cleanup/finish drain proceeds even after repeated SIGTERM.
+            log.drain = original_drain
--- /dev/null
+++ test_contracts.py
@@ -0,0 +1,79 @@
+"""ROOT-only prepared source-free contracts; no children/engines on import."""
+import copy
+import hashlib
+import unittest
+
+from native_ab_common import BANDS, COUNTERS, LABELS, MODELS, ORDER, counters, signature
+
+
+def health():
+    admission = {name: 0 for name in COUNTERS}
+    admission.update(kind='resident_gpu', batch_mode='fused', native_batch_rows_max=8,
+        active_sessions=0, queued_requests=0, gpu_fused_batches_by_rows={'2': 0, '4': 0, '8': 0})
+    return dict(admission=admission)
+
+
+def advanced():
+    before, after = health(), health()
+    a = after['admission']
+    a.update(completed_forward_kernel_dispatches=100, completed_model_token_steps=12,
+        committed_gpu_roots=12, gpu_fused_batches=2, gpu_fused_state_only_batches=1,
+        gpu_fused_greedy_batches=1, gpu_fused_state_only_rows=4, gpu_fused_rows_completed=12,
+        native_sampler_dispatches_completed=4, native_projection_dispatches_completed=9,
+        accepted_output_tokens=3)
+    a['gpu_fused_batches_by_rows'] = {'2': 0, '4': 1, '8': 1}
+    return before, after
+
+
+class Contracts(unittest.TestCase):
+    def test_explicit_native_order_and_complete_denominators(self):
+        self.assertEqual(LABELS, ('native_baseline', 'native_candidate'))
+        values = [(model, c, label) for model in MODELS for _, c in BANDS for order in ORDER for label in order]
+        self.assertEqual(len(values), 32)
+        self.assertEqual(sum(1 if c == 1 else 8 for _, c, _ in values), 144)
+        for model in MODELS:
+            for _, c in BANDS:
+                self.assertEqual([v[2] for v in values if v[:2] == (model, c)], list(ORDER[0] + ORDER[1]))
+
+    def test_actual_counter_phase_band_row_and_sampler_conservation(self):
+        self.assertEqual(counters(*advanced())['gpu_fused_rows_completed'], 12)
+        for key in ('gpu_fused_rows_completed', 'gpu_fused_batches', 'native_sampler_dispatches_completed'):
+            before, after = advanced()
+            after['admission'][key] += 1
+            with self.assertRaises(ValueError):
+                counters(before, after)
+
+    def test_absent_stale_failed_or_only_present_counters_refuse(self):
+        for key in ('failed_gpu_quanta', 'cancelled_requests', 'native_admission_refusals', 'discarded_ready_cohorts'):
+            before, after = advanced()
+            after['admission'][key] = 1
+            with self.assertRaises(ValueError):
+                counters(before, after)
+        with self.assertRaises(ValueError):
+            counters(health(), health())
+        before, after = advanced()
+        del after['admission']['gpu_fused_batches_by_rows']['8']
+        with self.assertRaises(KeyError):
+            counters(before, after)
+
+    def test_every_offer_and_full_ID_usage_are_required(self):
+        original = dict(request_id='request-0', prompt='literal transport fixture', max_tokens=8)
+        row = dict(request_id='request-0', prompt_sha256=hashlib.sha256(original['prompt'].encode()).hexdigest(),
+            max_tokens=8, completed=True, usage_reported_completion_tokens=2, finish_reason='length',
+            output_sha256='a' * 64, wire_token_observation=dict(status='provider_bound_complete_ids', token_ids=[11, 353]))
+        sample = dict(offered_requests=1, completed_requests=1, unsuccessful_requests=0,
+            completions_without_usage=0, requests=[row])
+        self.assertEqual(signature(sample, [original])['request-0']['token_ids'], [11, 353])
+        for field, bad in [('completed_requests', 0), ('offered_requests', 2)]:
+            value = copy.deepcopy(sample)
+            value[field] = bad
+            with self.assertRaises(ValueError):
+                signature(value, [original])
+        value = copy.deepcopy(sample)
+        value['requests'][0]['wire_token_observation']['token_ids'] = [11]
+        with self.assertRaises(ValueError):
+            signature(value, [original])
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ test_owner_control.py
@@ -0,0 +1,99 @@
+"""Prepared MOCK control fixtures only; no process/provider/physical claims."""
+from pathlib import Path
+import tempfile
+from types import SimpleNamespace
+import unittest
+from unittest.mock import patch
+
+from native_ab_owner import NativeOwner
+from sigterm_cancellation import SigtermCancellation
+
+
+class Control(unittest.TestCase):
+    def fixture(self, root):
+        events = []
+        cancel = SigtermCancellation()
+        process = SimpleNamespace(pid=123, stdout=SimpleNamespace(close=lambda: events.append('pipe')))
+        def cleanup(*_args):
+            events.append('cleanup')
+            return []
+        owned = SimpleNamespace(leader_status_unreaped=lambda _: None,
+            owned_container=lambda *_: 'a' * 64, cleanup=cleanup)
+        class Log:
+            error = None
+            def __init__(self, *_):
+                events.append('log-created')
+            def finish(self, _):
+                events.append('log-finished')
+                return dict(reader_complete=True, reader_error=None, truncated=False, bytes_seen=0)
+        class Monitor:
+            def __init__(self, *_):
+                events.append('monitor-created')
+            def finish(self):
+                events.append('monitor-finished')
+                return dict(failures=[], samples=[{}])
+        observations = SimpleNamespace(quiet=lambda *_: [dict(memory_used_mib=0)], Monitor=Monitor)
+        lifecycle = SimpleNamespace(listening=lambda _: False, BoundedLog=Log)
+        cgroup = SimpleNamespace(validate_caps=lambda *_: None)
+        deps = (None, None, owned, observations, cgroup, None, None, lifecycle)
+        protocol = dict(model='0.8b', zerogravity_binary_sha256='a' * 64,
+            ports=[18460, 18461], gpu_uuid='synthetic-control-only', gpu_total_mib=1, cpu_affinity=[0, 1, 2, 3])
+        owner = NativeOwner('native_baseline', root / 'sample', ['literal-mock'], protocol, deps, cancel, root)
+        return owner, cancel, process, events
+
+    def test_recorded_signal_across_Popen_handoff_still_retires_owned_child(self):
+        with tempfile.TemporaryDirectory() as value:
+            owner, cancel, process, events = self.fixture(Path(value))
+            def handoff(*_args, **_kwargs):
+                cancel._record(None, None)  # No asynchronous raise/real signal.
+                return process
+            with patch('native_ab_owner.subprocess.Popen', side_effect=handoff):
+                with self.assertRaises(InterruptedError):
+                    owner.start()
+            owner.close()
+            self.assertIs(owner.process, process)
+            self.assertEqual(events[-3:], ['monitor-finished', 'cleanup', 'log-finished'])
+
+    def test_postspawn_receipt_failure_cannot_skip_cleanup(self):
+        with tempfile.TemporaryDirectory() as value:
+            owner, _, process, events = self.fixture(Path(value))
+            with patch('native_ab_owner.subprocess.Popen', return_value=process), \
+                 patch.object(owner, 'save', side_effect=OSError('injected receipt failure')):
+                with self.assertRaises(OSError):
+                    owner.start()
+                with self.assertRaises(OSError):
+                    owner.close()
+            self.assertIn('cleanup', events)
+            self.assertIn('log-finished', events)
+
+    def test_monitor_failure_does_not_skip_group_or_log_cleanup(self):
+        with tempfile.TemporaryDirectory() as value:
+            owner, _, process, events = self.fixture(Path(value))
+            owner.process = process
+            owner.monitor = SimpleNamespace(finish=lambda: (_ for _ in ()).throw(ValueError('injected monitor')))
+            owner.log = SimpleNamespace(finish=lambda _: events.append('log-finished') or
+                dict(reader_complete=True, reader_error=None, truncated=False, bytes_seen=0))
+            owner.receipt['GPU_quiet_before'] = [dict(memory_used_mib=0)]
+            with self.assertRaises(ValueError):
+                owner.close()
+            self.assertIn('cleanup', events)
+            self.assertIn('log-finished', events)
+            self.assertEqual(owner.receipt['status'], 'UNQUALIFIED')
+
+    def test_group_failure_does_not_skip_log_retirement_or_allow_reuse(self):
+        with tempfile.TemporaryDirectory() as value:
+            owner, _, process, events = self.fixture(Path(value))
+            owner.process = process
+            owner.owned.cleanup = lambda *_: (_ for _ in ()).throw(ValueError('injected group'))
+            owner.log = SimpleNamespace(finish=lambda _: events.append('log-finished') or
+                dict(reader_complete=True, reader_error=None, truncated=False, bytes_seen=0))
+            owner.receipt['GPU_quiet_before'] = [dict(memory_used_mib=0)]
+            with self.assertRaises(ValueError):
+                owner.close()
+            self.assertIn('log-finished', events)
+            with self.assertRaises(ValueError):
+                owner.close()
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ validate_native_ab.py
@@ -0,0 +1,163 @@
+"""ROOT ONLY independent native/native accounting, no performance promotion."""
+import argparse
+import json
+from pathlib import Path
+import statistics
+
+from native_ab_common import (HERE, LABELS, ORDER, accepted_candidate, counters, finite,
+    load, parents, pin, require, save, sha, signature, source_freeze, tree)
+from run_native_ab import validate_plan
+
+
+def validate(output):
+    report = load(output / 'report.json')
+    require(report['schema'] == 'zerogravity.native-before-after.run.v1' and
+            report['status'] == 'RAW_NATIVE_BASELINE_CANDIDATE_COMPLETE_REQUIRES_INDEPENDENT_VALIDATION' and
+            report['source_freeze_sha256'] == source_freeze() and report['native_only'] is True and
+            report['main_timing_profiled'] is False and
+            report['performance_or_vllm_or_source_correct_goodput_qualified'] is False and
+            'failure' not in report, 'whole native/native execution, never a successful subset')
+    pin(report['plan'])
+    protocol, launcher, _, _, _, pins, _, _ = parents()
+    binding = load(HERE / 'bindings.json')
+    plan = validate_plan(load(report['plan']['path']), protocol, launcher, pins, binding)
+    require(accepted_candidate(plan['candidate_gate']['serving_gate']['sha256']) == plan['candidate_gate'],
+            'candidate actual serving gate and all antecedents unchanged')
+    expected = [(case, repetition, position, label) for case in plan['cases']
+                for repetition, order in enumerate(ORDER) for position, label in enumerate(order)]
+    require(len(report['samples']) == len(expected) == 32, 'all32 fresh owned processes')
+    stable, rows, offer_count, warm_count = {}, [], 0, 0
+    directories = set()
+    for item, (case, repetition, position, label) in zip(report['samples'], expected):
+        require((item['label'], item['model'], item['concurrency'], item['repetition'], item['order_position']) ==
+                (label, case['model'], case['concurrency'], repetition, position), 'complete ABBA/BAAB membership/order')
+        pin(item['receipt'])
+        path = Path(item['receipt']['path'])
+        require(path.name == 'receipt.json' and path.parent.parent == output and path.parent not in directories,
+                'exclusive per-process receipt, no reused sample')
+        directories.add(path.parent)
+        receipt = load(path)
+        require(receipt['label'] == label and receipt['native_implementation'] == 'zerogravity' and
+                receipt['status'] == 'OWNERS_CLOSED' and receipt['primary_failure'] is None and
+                receipt['cleanup_errors'] == [] and receipt['profiler_enabled'] is False and receipt['ready'] is True and
+                receipt['executable_sha256'] == binding['executables'][label]['sha256'] and
+                receipt['owned_container_absent_after_reap'] is True,
+                'genuine labeled native and original nonce/CID owner closed')
+        foreground = receipt['foreground_cleanup']
+        require(all(foreground[k] is True for k in ('leader_reaped', 'group_absent', 'all_nonzero_signals_before_reap')) and
+                foreground['stage_errors'] == [], 'all actual original group stages and final reap')
+        log = receipt['log_cleanup']
+        require(log['reader_complete'] is True and log['reader_error'] is None and log['truncated'] is False and
+                log['bytes_seen'] == log['bytes_retained'] <= 8 * 1024**2 and
+                Path(log['path']).parent == path.parent and sha(log['path']) == log['sha256_retained'],
+                'complete bounded log/pipe EOF evidence')
+        bootstrap = receipt['actual_bootstrap']
+        require(bootstrap['provider'] == 'zerogravity' and bootstrap['model'] == case['model'] and
+                bootstrap['stock_math_modified'] is False and bootstrap['policy_observer'] is False and
+                bootstrap == load(path.parent / 'bootstrap.json'), 'actual unchanged original native bootstrap')
+        from cgroup_readback_v3 import validate_caps
+        validate_caps(bootstrap['cgroup_readback'], case['protocols'][label]['cpu_affinity'])
+        warm = receipt['warmup']
+        require(warm['completed'] is True and warm['tokens'] is not None and
+                warm['token_observation']['status'] == 'provider_bound_complete_ids' and
+                len(warm['token_observation']['token_ids']) == warm['tokens'] <= 8,
+                'one complete ordinary unmeasured canonical warmup')
+        warm_count += 1
+        pin(receipt['measurement'])
+        measurement = load(receipt['measurement']['path'])
+        require(Path(receipt['measurement']['path']) == path.parent / 'measurement.json' and
+                (measurement['label'], measurement['model'], measurement['concurrency'],
+                 measurement['repetition'], measurement['order_position']) ==
+                (label, case['model'], case['concurrency'], repetition, position), 'bound actual measurement identity')
+        sample = measurement['benchmark']
+        require(sample == load(path.parent / 'raw-measurement.json') and
+                sample['concurrency'] == case['concurrency'] and sample['arrival_mode'] == 'open_loop_scheduled_arrivals',
+                'raw canonical measurement unchanged')
+        work = protocol.workload(case['workload'], 8)
+        sig = signature(sample, work)
+        reference = stable.setdefault(case['model'], sig['request-0'])
+        require(all(value == reference for value in sig.values()), 'all versions/bands complete exact outputs')
+        require(warm['output_sha256'] == reference['output_sha256'] and
+                warm['tokens'] == reference['usage_reported_completion_tokens'] and
+                warm['finish_reason'] == reference['finish_reason'] and
+                warm['token_observation']['token_ids'] == reference['token_ids'],
+                'same exact ordinary warmup output, not a special shorter request')
+        before, after = measurement['quiescent_before'], measurement['quiescent_after']
+        require(load(path.parent / 'quiescent.json') == dict(before=before, after=after), 'actual quiescent window')
+        delta = counters(before, after)
+        require(delta == measurement['source_counter_delta'] and delta['accepted_output_tokens'] ==
+                sum(len(value['token_ids']) for value in sig.values()), 'actual native packet/counter conservation')
+        if case['concurrency'] == 1:
+            require(delta['gpu_fused_batches'] == delta['native_projection_dispatches_completed'] ==
+                    delta['native_sampler_dispatches_completed'] == 0, 'no B1 request padding')
+        delivered, intervals, commit_intervals, packet_bands = 0, [], [], []
+        for request in sample['requests']:
+            obs = request['wire_token_observation']
+            ids = obs['token_ids']
+            if obs['client_delivery_status'] == 'individual_token_ids_observed':
+                require(len(obs['client_token_delivery_intervals_seconds']) == max(0, len(ids) - 1) and
+                        all(finite(v) for v in obs['client_token_delivery_intervals_seconds']), 'actual token intervals')
+                delivered += 1
+                intervals.extend(obs['client_token_delivery_intervals_seconds'])
+            require(obs['provider_commit_observation_status'] == 'provider_bound_complete_commit_observations' and
+                    len(obs['provider_token_records']) == len(ids), 'enabled original native per-request commit records')
+            commit_intervals.extend(obs['provider_commit_observation_intervals_seconds'])
+            for index, packet in enumerate(obs['provider_token_records']):
+                require(packet['gpu']['producer_frontier'] == 47 + index, 'actual independent canonical prompt/output frontier')
+                cohort = packet['gpu']['native_cohort']
+                if cohort is not None:
+                    require(cohort['width'] in (2, 4, 8), 'actual accepted row width')
+                    packet_bands.append(cohort['width'])
+        require(sample['wire_token_metrics']['completed_requests_with_individual_token_delivery'] == delivered,
+                'honest token ITL/coalescing coverage')
+        observations = receipt['physical_observations']
+        require(observations['failures'] == [] and observations['samples'] and observations['exact_peak'] is False and
+                all(v['uuid'] == case['protocols'][label]['gpu_uuid'] for v in observations['samples']),
+                'actual same-card point memory samples, not exact peaks')
+        rows.append(dict(label=label, model=case['model'], concurrency=case['concurrency'],
+            repetition=repetition, order_position=position, offered=sample['offered_requests'],
+            completed=sample['completed_requests'], usage_tokens=sample['usage_reported_completion_tokens'],
+            offered_window_seconds=sample['offered_window_seconds'],
+            reported_tokens_per_offered_second=sample['reported_tokens_per_offered_second'],
+            ttft_text_p50=sample['ttft_p50_seconds'], ttft_text_p95=sample['ttft_p95_seconds'],
+            actual_wire_token_metrics=sample['wire_token_metrics'], individual_ID_requests=delivered,
+            token_intervals=len(intervals), coalesced_or_unresolved_ID_requests=len(work)-delivered,
+            actual_packet_native_widths=sorted(set(packet_bands)), source_counter_delta=delta,
+            native_fused_work_observed=delta['gpu_fused_batches'] > 0,
+            fairness_by_local_label=sample['fairness_by_workload_tenant'],
+            point_sampled_total_device_peak_mib=observations['sampled_total_device_peak_mib'],
+            source_launchsites_not_hardware_kernel_counts=True))
+        offer_count += sample['offered_requests']
+    require(offer_count == plan['measured_offers'] == 144 and warm_count == plan['warmup_requests'] == 32,
+            'no warmup/measured offer denominator mixing')
+    summaries = []
+    for case in plan['cases']:
+        group = [r for r in rows if r['model'] == case['model'] and r['concurrency'] == case['concurrency']]
+        rates = {label: [r['reported_tokens_per_offered_second'] for r in group if r['label'] == label] for label in LABELS}
+        require(all(len(values) == 4 and all(finite(v) for v in values) for values in rates.values()), 'all four provider samples')
+        summaries.append(dict(model=case['model'], concurrency=case['concurrency'],
+            median_usage_throughput_by_native_label={label: statistics.median(values) for label, values in rates.items()},
+            four_samples_per_label=True, confidence_certified_win=False))
+    return dict(status='PASS_NATIVE_BASELINE_CANDIDATE_RAW_ACCOUNTING_ONLY', samples=rows, summaries=summaries,
+        measured_offers=offer_count, warmup_requests=warm_count, source_freeze_sha256=source_freeze(),
+        all_full_ID_text_usage_finish_exact=True, main_timing_profiled=False,
+        actual_hardware_kernel_counts_qualified=False, source_correct_slo_goodput_qualified=False,
+        vllm_comparison_or_parity_qualified=False, graph_or_full_HF_eligibility=False,
+        complete_phase_acceptance_requires_ROOT_review=True)
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--output', type=Path, required=True)
+    args = parser.parse_args()
+    output = args.output.resolve()
+    require(output.parent == HERE, 'own exact native/native output only')
+    value = validate(output)
+    save(output / 'validation.json', value, output)
+    tree(output)
+    print(json.dumps(dict(status=value['status'], measured_offers=value['measured_offers'],
+                          warmup_requests=value['warmup_requests']), sort_keys=True))
+
+
+if __name__ == '__main__':
+    main()

````

