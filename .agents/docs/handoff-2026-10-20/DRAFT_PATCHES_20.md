# Draft Patches — volume 20

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/BACKEND_DIAGNOSTIC_V3.patch

Original UTF-8 bytes: 19455; SHA256 `68cc55b17c9bddfa0159e2f375bacaca96568477a67706a66ba6b1fbca0d62c0`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- immutable-v2/native_ref_control_v2.cpp
+++ additive-v3/native_ref_control_v3.cpp
@@ -128,9 +128,40 @@
     const auto device=get_device(backend);checked(device!=nullptr,"actual backend has public device");
     const auto reg=get_reg(device);checked(reg!=nullptr,"actual backend has public registry");return reg;
 }
-static void set_reference(ggml_backend_t backend,bool initialization) {
+// Public metadata only, reached solely at the original pre-setter refusal.
+[[noreturn]] static void backend_identity_refused(ggml_backend_t backend,bool initialization,
+        ggml_backend_dev_t input_device,bool actual_cpu) {
+    const int saved=errno;
+    static auto backend_name=symbol<decltype(&ggml_backend_name)>(BASE,BASE_BYTES,"ggml_backend_name");
+    static auto get_device=symbol<decltype(&ggml_backend_get_device)>(BASE,BASE_BYTES,"ggml_backend_get_device");
+    static auto device_name=symbol<decltype(&ggml_backend_dev_name)>(BASE,BASE_BYTES,"ggml_backend_dev_name");
+    static auto device_type=symbol<decltype(&ggml_backend_dev_type)>(BASE,BASE_BYTES,"ggml_backend_dev_type");
+    static auto device_reg=symbol<decltype(&ggml_backend_dev_backend_reg)>(BASE,BASE_BYTES,"ggml_backend_dev_backend_reg");
+    static auto reg_name=symbol<decltype(&ggml_backend_reg_name)>(BASE,BASE_BYTES,"ggml_backend_reg_name");
+    const auto device=backend?get_device(backend):nullptr;
+    const auto reg=device?device_reg(device):nullptr;
+    const char *bn=backend?backend_name(backend):"(null)";
+    const char *dn=device?device_name(device):"(null)";
+    const char *rn=reg?reg_name(reg):"(null)";
+    const int type=device?static_cast<int>(device_type(device)):-1;
+    const bool input_known=initialization && input_device;
+    char message[4096]={};
+    const int size=std::snprintf(message,sizeof(message),
+        "backend_identity_diagnostic_v3 origin=%.64s cpu_is_cpu=%s backend=%p backend_name=%.128s "
+        "device=%p device_name=%.128s device_type=%d registry=%p registry_name=%.128s "
+        "input_device=%p input_device_known=%s input_device_equals_returned=%s "
+        "setter_called_for_this_attempt=false compute_continued_after_refusal=false errno=%d",
+        initialization?"public_dev_init":"existing_public_abort_setter_callback",actual_cpu?"true":"false",
+        static_cast<void *>(backend),bn?bn:"(null)",static_cast<void *>(device),dn?dn:"(null)",type,
+        static_cast<void *>(reg),rn?rn:"(null)",static_cast<void *>(input_device),input_known?"true":"false",
+        input_known && input_device==device?"true":"false",saved);
+    if(size<0 || size>=static_cast<int>(sizeof(message)))fatal("backend_identity_diagnostic_v3 formatting refused");
+    errno=saved;fatal(message); // SAME exit125; never skip/set/disable or continue computation.
+}
+static void set_reference(ggml_backend_t backend,bool initialization,ggml_backend_dev_t input_device=nullptr) {
     static IsCpu is_cpu=symbol<IsCpu>(CPU,CPU_BYTES,"ggml_backend_is_cpu");
-    checked(backend && is_cpu(backend),"nonCPU backend in CPU-only control");
+    const bool actual_cpu=backend && is_cpu(backend);
+    if(!actual_cpu)backend_identity_refused(backend,initialization,input_device,actual_cpu);
     void *raw=lookup()(registry(backend),"ggml_backend_cpu_set_use_ref");
     attest(raw,CPU,CPU_BYTES,"ggml_backend_cpu_set_use_ref");
     reinterpret_cast<Ref>(raw)(backend,true); // Sole intentional policy change.
@@ -155,7 +186,7 @@
     static ref_control::Init original=ref_control::symbol<ref_control::Init>(ref_control::BASE,
         ref_control::BASE_BYTES,"ggml_backend_dev_init");
     errno=saved;auto backend=original(device,params);const int after=errno;
-    if(ref_control::armed){ref_control::checked(backend!=nullptr,"native backend initialization failed");ref_control::set_reference(backend,true);}
+    if(ref_control::armed){ref_control::checked(backend!=nullptr,"native backend initialization failed");ref_control::set_reference(backend,true,device);}
     errno=after;return backend;
 }
 extern "C" void *ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg,const char *name) {
@@ -259,7 +290,7 @@
         write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
         std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
         report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
-            <<"\"control_source_revision\":\"public_use_ref_canonical_provider_attestation_v2\","
+            <<"\"control_source_revision\":\"public_use_ref_backend_identity_diagnostic_v3\","
             <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\",\"CPU_backend_count\":2,"
             <<"\"public_setter_calls\":4,\"existing_abort_setter_calls\":2,\"actual_public_plan_calls\":"<<ref_control::graph_calls
             <<",\"actual_successful_public_plan_calls\":"<<ref_control::graph_success
--- immutable-v2/build-argv-v2.json
+++ additive-v3/build-argv-v3.json
@@ -11,7 +11,7 @@
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include",
     "-I",
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v2.cpp",
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v3.cpp",
     "-L",
     "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib",
     "-L",
@@ -29,10 +29,11 @@
     "-ldl",
     "-pthread",
     "-o",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v2"
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v3"
   ],
   "compiler_identity": "ROOT must record real compiler and own finite build receipt before execution",
   "native_math_source_unchanged": true,
-  "source_revision": "public_use_ref_canonical_provider_attestation_v2",
-  "generic_runner_v1_freeze_scope": "unchanged generic launcher/helper sources only; not proof that v1 CPP was compiled into the v2 binary"
+  "source_revision": "public_use_ref_backend_identity_diagnostic_v3",
+  "generic_runner_v1_freeze_scope": "unchanged generic helper only; actual V3 CPP/binary/build binding separately mandatory",
+  "scope": "DIAGNOSTIC_ONLY same pre-setter fatal125; no nonCPU policy allowance"
 }
--- /dev/null
+++ additive-v3/bind_v3.py
@@ -0,0 +1,98 @@
+"""ROOT ONLY: bind actual V3 compile/run receipts; no model/operator calls."""
+import argparse
+from pathlib import Path
+import re
+
+from common import HERE, REPO, digest, jread, require, write
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--manifest-sha256", required=True)
+    parser.add_argument("--build", required=True, type=Path)
+    parser.add_argument("--run", required=True, type=Path)
+    parser.add_argument("--new-binding", required=True, type=Path)
+    args = parser.parse_args()
+    require(re.fullmatch("[a-f0-9]{64}", args.manifest_sha256) is not None,
+            "independently reviewed manifest SHA")
+    path = HERE / "freeze-manifest-v3.json"
+    require(digest(path)["sha256"] == args.manifest_sha256, "exact reviewed V3 freeze")
+    manifest = jread(path)
+    for version, directory in (("original_freeze", "frozen-prepared"), ("v2_freeze", "frozen-v2")):
+        pin = manifest[version]
+        parent = HERE / pin["file"]
+        require(digest(parent) == {key: pin[key] for key in ("bytes", "sha256")}, "unchanged prior manifest")
+        for name, expected in jread(parent)["files"].items():
+            require(digest(HERE / name) == digest(HERE / directory / name) == expected,
+                    "all original source and frozen copies unchanged: " + name)
+    for name, expected in manifest["files"].items():
+        require(digest(HERE / name) == digest(HERE / "frozen-v3" / name) == expected,
+                "new frozen source unchanged: " + name)
+    for row in manifest["preserved_failed_metadata"]:
+        require(digest(row["path"]) == {key: row[key] for key in ("bytes", "sha256")},
+                "failed case evidence unchanged")
+    build, run = args.build.resolve(strict=True), args.run.resolve(strict=True)
+    destination = args.new_binding.resolve()
+    require(build.is_relative_to(HERE) and run.is_relative_to(HERE)
+            and destination.is_relative_to(HERE) and not destination.exists(),
+            "owned actual build/run and fresh binding only")
+    requested, result = jread(build / "requested.json"), jread(build / "result.json")
+    compiler = manifest["root_actual_compiler_pin"]
+    prepared = jread(HERE / "build-argv-v3.json")
+    actual_inputs = {row["path"]: digest(row["path"]) for row in manifest["compile_inputs"]}
+    require(actual_inputs == {row["path"]: {key: row[key] for key in ("bytes", "sha256")}
+                              for row in manifest["compile_inputs"]}, "actual exact compiler inputs")
+    for receipt in (requested, result):
+        require(receipt["compiler"] == compiler["path"]
+                and receipt["compiler_sha256"] == compiler["sha256"]
+                and digest(compiler["path"])["sha256"] == compiler["sha256"], "actual pinned compiler")
+        require(receipt["cwd"] == str(REPO)
+                and receipt["argv"] == [compiler["path"], *prepared["argv"][1:]]
+                and receipt["source_freeze_sha256"] == args.manifest_sha256
+                and receipt["source_inputs"] == actual_inputs
+                and receipt["runtime_or_math_qualified"] is False,
+                "actual relative-argv CWD/new CPP+included source pre/post build binding")
+    require(type(result["exit_code"]) is int and result["exit_code"] == 0, "actual successful fresh build")
+    binary = (HERE / "native_ref_control_v3").resolve(strict=True)
+    identity = digest(binary)
+    require(result["binary_path"] == str(binary)
+            and identity == {"bytes": result["binary_bytes"], "sha256": result["binary_sha256"]},
+            "exact actual new binary output")
+    invocation = jread(run / "invocation.json")
+    require(invocation["argv"][0] == str(binary) and invocation["inputs"][str(binary)] == identity,
+            "actual generic runner executed this newly compiled V3 binary")
+    require(invocation["freeze_manifest"] == digest(HERE / "freeze-manifest.json"),
+            "generic V1 helper receipt remains separate from V3 CPP build evidence")
+    native_log = run / "native.log"
+    require(native_log.is_file() and not native_log.is_symlink()
+            and 0 < native_log.stat().st_size == invocation["log_bytes"] <= 8 * 1024 * 1024,
+            "complete bounded original stdout/stderr diagnostic log")
+    log_identity = digest(native_log)
+    control_path = run / "capture/use-ref-control.json"
+    control = None
+    if control_path.exists():
+        actual = jread(control_path)
+        require(actual["control_source_revision"] == "public_use_ref_backend_identity_diagnostic_v3"
+                and actual["provider_policy"] == "same_canonical_path_regular_device_inode_length_exact_export",
+                "actual V3 source marker with unchanged V2 attestation policy")
+        control = digest(control_path)
+    else:
+        require(invocation["status"] == "UNQUALIFIED", "successful run missing actual V3 control report")
+    write(destination, {
+        "status": "SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE",
+        "new_source_freeze": digest(path), "generic_V1_helper_freeze": digest(HERE / "freeze-manifest.json"),
+        "V2_attestation_freeze": digest(HERE / "freeze-manifest-v2.json"),
+        "new_cpp": {"path": str(HERE / "native_ref_control_v3.cpp"), **digest(HERE / "native_ref_control_v3.cpp")},
+        "included_unchanged_token_loop": digest(HERE / "native_raw_original.cpp"),
+        "build_requested": digest(build / "requested.json"), "build_result": digest(build / "result.json"),
+        "actual_binary": {"path": str(binary), **identity}, "actual_invocation": digest(run / "invocation.json"),
+        "actual_diagnostic_log": {"path": str(native_log), **log_identity},
+        "actual_invocation_status": invocation["status"], "actual_control_report": control,
+        "diagnostic_only_no_nonCPU_policy_allowance": True, "runtime_or_math_qualified": False,
+        "next": "read actual bounded backend-identity refusal; do not infer or enable ACCEL policy",
+    })
+    print("SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE")
+
+
+if __name__ == "__main__":
+    main()
--- /dev/null
+++ additive-v3/SOURCE_AUDIT_V3.md
@@ -0,0 +1,111 @@
+# Additive V3 backend identity diagnostic — ROOT only
+
+Owner: baseline_review, root-authorized additive files only. All original14,
+V2 source/manifest/frozen copies and both failed cases remain unchanged.
+No build, test, model/operator, container/GPU runtime or large payload hash by
+author. No source/profile/math/StateRoot/long-context acceptance.
+
+## Actual refusal and source inference
+
+Fresh `root-0.8b-8192-ref64-2` used actual V2 binary42b98af8… and failed125
+after context geometry n_ctx8448/n_outputs64, before any accepted capture:
+records.jsonl is0B, only raw teacher file exists, no public setter/plan trace.
+Its log explicitly loads BLAS and Haswell CPU. Cleanup completed without errors.
+
+Pinned d812350 source:
+
+- llama-context.cpp334–366 initializes model devices, then every ACCEL backend
+  (explicit example BLAS), then CPU. n_gpu_layers0/offload flags do not suppress
+  the ACCEL construction loop.
+- ggml-blas.cpp214/322/478 public metadata names BLAS;356 returns device type
+  ACCEL;281–307 constructs its distinct BLAS backend GUID.
+- ggml-cpu.cpp249 recognizes CPU GUID specifically. Thus BLAS legitimately
+  returns false. Current control applies set_reference to EVERY dev_init return.
+- CPU_REPACK is an extra CPU buffer type (repack.cpp5162/5238), not a separately
+  initialized backend. It cannot itself identify this failed backend predicate.
+
+BLAS is a source-supported hypothesis, not actual backend identity. No v2
+diagnostic printed the returned backend/device/registry; do not close the cause.
+
+## V3 observation only
+
+`native_ref_control_v3.cpp` adds one fatal metadata helper, only on the original
+failed backend&&is_cpu check. Public BASE symbol lookup retains exact V2
+NOLOAD/canonical regular-file dev/inode/length/export attestation. No new DSO,
+RTLD_GLOBAL, fallback, private structure read or backend/provider replacement.
+
+The <=4095B named diagnostic prints origin, actual is_cpu boolean, opaque
+backend/device/registry identities, public names/device type, original input
+device identity, whether it is known and equals the returned backend device.
+Public getters resolve from the original BASE; existing predicate remains the
+same exact Haswell public is_cpu. Known actual backend is read through public
+APIs only. Null metadata is marked rather than dereferenced.
+
+Then SAME exit125, BEFORE setter invocation for that attempt. It never skips or
+disables a backend, never sets a CPU flag on a nonCPU object, and never proceeds
+to computation after refusal. Successful CPU predicate path retains the same
+setter/callback forwarding/plan checks/token loop; only the source revision
+marker changes for build/run binding. No new accelerator policy is implemented.
+
+## ROOT actual compile and invocation binding
+
+Build V3 with build-argv-v3.json argv from MAIN repository CWD using the exact
+pinned compiler. Preserve old binaries/build directories. Fresh build receipts
+requested.json and result.json must both include actual `cwd`, `compiler`,
+`compiler_sha256`, `argv`, reviewed V3 `source_freeze_sha256`, exact `source_inputs`
+mapping from manifest compile_inputs, and runtime_or_math_qualified=false.
+Record source_inputs before/after compilation. Result also records real exit,
+binary_path/bytes/SHA. This closes the relative-argv CWD gap explicitly.
+
+Use unchanged run_ref_control.py with new binary path/digest and a fresh output
+name. The generic invocation correctly pins V1 launcher/helpers; it does not
+claim that V1 CPP was compiled into this V3 binary. After actual invocation:
+
+```sh
+python3 .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/bind_v3.py \
+  --manifest-sha256 REVIEWED_V3_SHA \
+  --build ABS_FRESH_BUILD --run ABS_FRESH_CASE --new-binding ABS_FRESH_BINDING_JSON
+```
+
+Recipe authenticates all prior/new frozen sources, failed metadata, compiler,
+exact new CPP/includes/CWD/argv/build exit/binary and actual runner inputs. A
+failed diagnostic may bind source/build/invocation while remaining UNQUALIFIED.
+Actual diagnostic interpretation is ROOT review; it is not a numeric verdict.
+
+## Future policy feasibility only — NOT implemented
+
+Pinned public scheduler graph routes:
+
+- ggml-backend.cpp595: sync graph_compute delegates graph_compute_async.
+- 601–604: graph_compute_async dispatches the selected backend iface directly.
+- 1955/1977: both ordinary split and observed graph-view scheduler paths call
+  that public async entry. 588–592: public graph_plan_compute is another direct
+  iface route; checking only inner CPU ggml_graph_compute does not cover it.
+- CPU cpp139–153 stores source cplan.use_ref at plan creation;162–190 both CPU
+  plan/ordinary callbacks converge on ggml_graph_compute, whose current wrapper
+  checks actual cplan.use_ref before/after. Old plans must not be silently fixed.
+- BLAS cpp226–263 directly invokes CBLAS and bypasses inner CPU graph_compute.
+
+If actual identity later proves a genuine ACCEL is initialized but unused,
+unmodified construction could be recorded separately while a NEW reviewed
+control rejects every nonCPU public async/plan compute BEFORE delegation.
+Unknown/foreign backends remain cold refusals. Known CPU public calls could
+carry a thread-local scope through original forwarding, bind actual inner CPU
+plan observations to that backend, and require every original delegated plan
+use_ref=true. Nested sync→async must count one compute rather than two math
+invocations; capture every entry/return/success and all initialization/retirement.
+Opaque plan handles need public create/free association if plan route is used;
+no private plan/backend layout reinterpretation. Existing two-CPU counts cannot
+be relabeled as all-backend coverage; constructed ACCELs require explicit records.
+
+This is source feasibility, not compiled route closure. ROOT must prove actual
+public interposition paths/DSO closure for both plain and observed modes before
+allowing construction to continue. Direct/internal bypass or any unmatched
+compute must refuse. Numerical all730/fullV/all42 and observation guards remain.
+
+Crucially, op_offload=false does NOT prove ACCEL unused: backend.cpp1028 chooses
+the highest priority compatible buffer/op; pass3 around1380 upgrades compatible
+buffer nodes independently of op_offload. BLAS supports host-buffer contiguous
+F32 GEMM with all relevant dimensions>=32 (blas.cpp397–459). Bulk64 QK/AV may
+qualify. A future guard may correctly refuse; do not disable BLAS, alter buffers,
+or silently downgrade math merely to make the control pass.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/BINDING_VALIDATOR_V6.patch

Original UTF-8 bytes: 11233; SHA256 `ccc3049cd78c033efef8577f57104ab2f2fb5ba4633eda97a5dd0273d26f2a94`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- build-argv-v4.json
+++ build-argv-v6.json
@@ -11,7 +11,7 @@
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include",
     "-I",
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v4.cpp",
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v6.cpp",
     "-L",
     "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib",
     "-L",
@@ -34,11 +34,11 @@
     "-ldl",
     "-pthread",
     "-o",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v4"
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v6"
   ],
   "compiler_identity": "ROOT must record real compiler and own finite build receipt before execution",
   "native_math_source_unchanged": true,
-  "source_revision": "public_use_ref_exact_blas_lifecycle_cpu_compute_v4",
+  "source_revision": "public_use_ref_exact_blas_cpu_compute_legal_null_free_v6",
   "generic_runner_v1_freeze_scope": "unchanged generic helper; V4 source/build/compiled-route/invocation binding independently mandatory",
-  "scope": "SOURCE ONLY exact BLAS construction/retirement; refuse EVERY nonCPU compute before delegation; CPU public-to-inner TLS/use_ref guard; no arithmetic acceptance"
+  "scope": "Same V4 nonnull backend/BLAS/use_ref policy; only original legal free(NULL) forwarded without pre/post owner tracking after actualV5 NULL proof"
 }
--- bind_v4.py
+++ bind_v6.py
@@ -1,10 +1,30 @@
-"""ROOT ONLY: bind actual V4 compile/run/static-provider receipts; no native entry calls."""
+"""ROOT ONLY: bind actual V6 compile/run/static-provider receipts; no native entry calls."""
 import argparse
 from pathlib import Path
 import re
 
 from common import HERE, REPO, digest, jread, require, write
-from v4_sources import REVISION, verify_sources, verify_runtime_files
+from v4_sources import verify_runtime_files
+from bind_v5 import verify_sources as verify_v5_sources
+
+REVISION = "public_use_ref_exact_blas_cpu_compute_legal_null_free_v6"
+
+def verify_sources(expected_manifest=None):
+    path = HERE / "freeze-manifest-v6.json"
+    if expected_manifest is not None:
+        require(digest(path)["sha256"] == expected_manifest, "independently reviewed V6 manifest")
+    manifest = jread(path)
+    parent = manifest["v5_freeze"]
+    require(digest(HERE / parent["file"]) == {key: parent[key] for key in ["bytes", "sha256"]},
+            "V5 immutable diagnostic parent")
+    verify_v5_sources(parent["sha256"])
+    for name, expected in manifest["files"].items():
+        require(digest(HERE / name) == digest(HERE / "frozen-v6" / name) == expected,
+                "V6 live/frozen source")
+    for row in manifest["preserved_v5_failed_metadata"]:
+        require(digest(row["path"]) == {key: row[key] for key in ["bytes", "sha256"]},
+                "V5 failed primitive proof unchanged")
+    return manifest
 
 
 def main():
@@ -16,7 +36,7 @@
     args = parser.parse_args()
     require(re.fullmatch("[a-f0-9]{64}", args.manifest_sha256) is not None,
             "independently reviewed manifest SHA")
-    path = HERE / "freeze-manifest-v4.json"
+    path = HERE / "freeze-manifest-v6.json"
     manifest = verify_sources(args.manifest_sha256)
     extra_providers = verify_runtime_files()
     build, run = args.build.resolve(strict=True), args.run.resolve(strict=True)
@@ -26,7 +46,7 @@
             "owned actual build/run and fresh binding only")
     requested, result = jread(build / "requested.json"), jread(build / "result.json")
     compiler = manifest["root_actual_compiler_pin"]
-    prepared = jread(HERE / "build-argv-v4.json")
+    prepared = jread(HERE / "build-argv-v6.json")
     actual_inputs = {row["path"]: digest(row["path"]) for row in manifest["compile_inputs"]}
     require(actual_inputs == {row["path"]: {key: row[key] for key in ("bytes", "sha256")}
                               for row in manifest["compile_inputs"]}, "actual exact compiler inputs")
@@ -41,16 +61,16 @@
                 and receipt["runtime_or_math_qualified"] is False,
                 "actual relative-argv CWD/new CPP+included source pre/post build binding")
     require(type(result["exit_code"]) is int and result["exit_code"] == 0, "actual successful fresh build")
-    binary = (HERE / "native_ref_control_v4").resolve(strict=True)
+    binary = (HERE / "native_ref_control_v6").resolve(strict=True)
     identity = digest(binary)
     require(result["binary_path"] == str(binary)
             and identity == {"bytes": result["binary_bytes"], "sha256": result["binary_sha256"]},
             "exact actual new binary output")
     invocation = jread(run / "invocation.json")
     require(invocation["argv"][0] == str(binary) and invocation["inputs"][str(binary)] == identity,
-            "actual generic runner executed this newly compiled V4 binary")
+            "actual generic runner executed this newly compiled V6 binary")
     require(invocation["freeze_manifest"] == digest(HERE / "freeze-manifest.json"),
-            "generic V1 helper receipt remains separate from V4 CPP build evidence")
+            "generic V1 helper receipt remains separate from V6 CPP build evidence")
     native_log = run / "native.log"
     require(native_log.is_file() and not native_log.is_symlink()
             and 0 < native_log.stat().st_size == invocation["log_bytes"] <= 8 * 1024 * 1024,
@@ -62,10 +82,10 @@
         actual = jread(control_path)
         require(actual["control_source_revision"] == REVISION
                 and actual["provider_policy"] == "same_canonical_path_regular_device_inode_length_exact_export",
-                "actual V4 source marker with unchanged V2 attestation policy")
+                "actual V6 source marker with unchanged V2 attestation policy")
         control = digest(control_path)
     else:
-        require(invocation["status"] == "UNQUALIFIED", "successful run missing actual V4 control report")
+        require(invocation["status"] == "UNQUALIFIED", "successful run missing actual V6 control report")
     trace_path = run / "capture/use-ref-public-trace.jsonl"
     maps_path = run / "capture/use-ref-loaded-maps.txt"
     trace = digest(trace_path) if trace_path.exists() else None
@@ -79,7 +99,9 @@
         "new_source_freeze": digest(path), "generic_V1_helper_freeze": digest(HERE / "freeze-manifest.json"),
         "V2_attestation_freeze": digest(HERE / "freeze-manifest-v2.json"),
         "V3_diagnostic_freeze": digest(HERE / "freeze-manifest-v3.json"),
-        "new_cpp": {"path": str(HERE / "native_ref_control_v4.cpp"), **digest(HERE / "native_ref_control_v4.cpp")},
+        "V4_policy_freeze": digest(HERE / "freeze-manifest-v4.json"),
+        "V5_null_diagnostic_freeze": digest(HERE / "freeze-manifest-v5.json"),
+        "new_cpp": {"path": str(HERE / "native_ref_control_v6.cpp"), **digest(HERE / "native_ref_control_v6.cpp")},
         "included_unchanged_token_loop": digest(HERE / "native_raw_original.cpp"),
         "build_requested": digest(build / "requested.json"), "build_result": digest(build / "result.json"),
         "actual_binary": {"path": str(binary), **identity}, "actual_invocation": digest(run / "invocation.json"),
@@ -87,9 +109,10 @@
         "actual_invocation_status": invocation["status"], "actual_control_report": control,
         "actual_control_trace": trace, "actual_loaded_maps": maps,
         "exact_BLAS_construction_retirement_only": True, "nonCPU_compute_delegation_permitted": False,
+        "same_V4_nonnull_policy_only_legal_NULL_free_restored": True,
         "runtime_or_math_qualified": False, "actual_extra_provider_and_static_scope": extra_providers,
         "selected_static_route_pin": digest(HERE / "static-route-pins-v4.json"),
-        "next": "validate_control_v4.py then separate raw730/fullV/all42 comparators; any BLAS compute stays UNQUALIFIED",
+        "next": "validate_control_v6.py then separate raw730/fullV/all42 comparators; any BLAS compute stays UNQUALIFIED",
     })
     print("SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE")
 
--- validate_control_v4.py
+++ validate_control_v6.py
@@ -7,7 +7,8 @@
 import re
 from common import HERE, REPO, TEACHER_SHA, authenticate, bounded, digest, jread, require, write
 from trace_v4 import validate_trace
-from v4_sources import REVISION, verify_sources, verify_runtime_files
+from v4_sources import verify_runtime_files
+from bind_v6 import REVISION, verify_sources
 
 
 def cpulist(text):
@@ -110,23 +111,23 @@
     blas_path = extra_providers['BLAS']['canonical']
     require(blas_path in paths and not any('libggml-blas' in path and path != blas_path for path in paths),
             'actual exact pinned BLAS mapped, no alternate BLAS provider')
-    binding = jread(output / 'source-build-binding-v4.json')
+    binding = jread(output / 'source-build-binding-v6.json')
     require(binding['status'] == 'SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE'
             and binding['actual_invocation'] == digest(output / 'invocation.json')
-            and binding['new_source_freeze'] == digest(HERE / 'freeze-manifest-v4.json')
+            and binding['new_source_freeze'] == digest(HERE / 'freeze-manifest-v6.json')
             and binding['actual_control_report'] == digest(capture / 'use-ref-control.json')
             and binding['actual_control_trace'] == digest(capture / 'use-ref-public-trace.jsonl')
             and binding['actual_loaded_maps'] == digest(capture / 'use-ref-loaded-maps.txt')
             and binding['actual_binary'] == dict(path=invocation['argv'][0], **invocation['inputs'][invocation['argv'][0]]),
-            'independent ROOT compile/actual V4 binary/generic runner binding required')
+            'independent ROOT compile/actual V6 binary/generic runner binding required')
     verify_sources()
     verify_runtime_files()
-    return dict(status='PASS_PUBLIC_USE_REF_V4_CONTROL_CAPTURE_ONLY_COMPARISONS_REQUIRED',
+    return dict(status='PASS_PUBLIC_USE_REF_V6_CONTROL_CAPTURE_ONLY_COMPARISONS_REQUIRED',
                 source_model_sha256=case['model_identity']['sha256'], raw_teacher_sha256=TEACHER_SHA,
                 cut=8192, prefix_batch=invocation['prefix'], CPU_affinity=cpus, public_route=observed,
                 invocation=digest(output / 'invocation.json'), complete=digest(capture / 'complete.json'),
                 control=digest(capture / 'use-ref-control.json'),
-                extra_provider=extra_providers, source_build_binding=digest(output / 'source-build-binding-v4.json'),
+                extra_provider=extra_providers, source_build_binding=digest(output / 'source-build-binding-v6.json'),
                 scope='exact BLAS lifecycle with no BLAS math; CPU public-route/TLS/actual-use_ref policy and '
                       'original selected-tail transparency only; selected static entry evidence is not universal runtime closure; '
                       'no raw730/cross-provider all42 result or historical/whole-cause conclusion')

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/FREE_REFUSAL_DIAGNOSTIC_V5.patch

Original UTF-8 bytes: 2720; SHA256 `66827cbdce3a6b7d1f24cc814f30492746d21e916c69fdc5725e5132c151f71b`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- native_ref_control_v4.cpp
+++ native_ref_control_v5.cpp
@@ -403,7 +403,23 @@
         ref_control::BASE_BYTES,"ggml_backend_free");
     ref_control::Kind kind=ref_control::Kind::Cpu;
     if(ref_control::armed) {
-        std::lock_guard<std::mutex> guard(ref_control::mutex);auto &record=ref_control::known(backend);
+        std::lock_guard<std::mutex> guard(ref_control::mutex);
+        const auto found=ref_control::backends.find(backend);
+        const bool tracked=found!=ref_control::backends.end();
+        const bool retired=tracked && found->second.freed;
+        if(!backend || !tracked || retired) {
+            // Diagnostic only at SAME refusal. Never dereference unknown/null/retired backend.
+            char message[768]={};
+            const int size=std::snprintf(message,sizeof(message),
+                "free_refusal_diagnostic_v5 api=ggml_backend_free backend=%p is_null=%s tracked=%s retired=%s "
+                "tracked_backend_count=%zu actual_public_compute_calls=%zu actual_inner_calls=%zu "
+                "original_free_delegated=false policy_continued_after_refusal=false",
+                static_cast<void *>(backend),backend?"false":"true",tracked?"true":"false",retired?"true":"false",
+                ref_control::backends.size(),ref_control::public_calls,ref_control::graph_calls);
+            ref_control::checked(size>0 && size<static_cast<int>(sizeof(message)),"bounded free-refusal diagnostic");
+            ref_control::fatal(message); // SAME exit125; no no-op/constructor policy admitted yet.
+        }
+        auto &record=ref_control::known(backend);
         ref_control::checked(ref_control::scope==nullptr && ref_control::sync_scope==nullptr
             && (record.kind==ref_control::Kind::Cpu?(record.abort==1 && record.set==2):(record.set==0 && record.abort==0)),
             "only exact configured CPU or unmodified BLAS retirement outside math");
@@ -488,7 +504,7 @@
         write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
         std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
         report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
-            <<"\"control_source_revision\":\"public_use_ref_exact_blas_lifecycle_cpu_compute_v4\","
+            <<"\"control_source_revision\":\"public_use_ref_unknown_free_diagnostic_v5\","
             <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\","
             <<"\"CPU_backend_count\":2,\"BLAS_backend_count\":2,\"public_setter_calls\":4,"
             <<"\"existing_abort_setter_calls\":2,\"BLAS_compute_delegations\":0,"

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 96374; SHA256 `027a911641491355e2b99aa0fa799af5a74d5d26df448a06d9e3ead5d4e01fae`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/README.md .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/README.md
new file mode 100644
index 0000000..4aa6d15
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/README.md
@@ -0,0 +1,126 @@
+# Public native CPU use_ref bulk control
+
+Parent: `.agents/plans/numerics-and-context/TODO.md`. Owner: `/root/batch_design_review`.
+PREPARED SOURCE ONLY; ROOT alone compiles, tests and runs. No production change.
+
+Fresh8192 teachers, approved original GGUF0.8B/2B, CPU8, F32 KV, FA off,
+prefix=batch=ubatch64 or512. One explicit case per invocation. Existing native
+AR1/Rust evidence is preserved; this control is not yet executed or accepted.
+
+## Public binding
+
+- `native_raw_original.cpp` is byte-identical to the accepted16560B client
+  (`387e0fdf...`). Its token loop, model/context construction, selected-token
+  observer,18 actual GDN callback guards,730 records, full248320 logits and
+  whole serialized plain/observed-state checks are unchanged.
+- New executable exports four public symbols. Original public device init
+  creates each real CPU backend, then the documented CPU registry extension
+  `ggml_backend_cpu_set_use_ref(backend,true)` configures it. Context's existing
+  documented `ggml_backend_set_abort_callback` callback is delegated unchanged,
+  then sets the same reference policy on each real backend again.
+- Pinned `llama-context.cpp:335–375` constructs CPU via public dev-init and
+  calls that callback; `:1206–1219` visits every context backend. No private
+  `llama_context`, `ggml_backend` or CPU context fields are read or written.
+- Original public `ggml_graph_compute(graph,cplan)` receives the original
+  public `ggml_cplan`; every observed plan must already have `use_ref=true`
+  before delegation and retain it after. Both backend callbacks, four setter
+  calls, all plans and two final frees are actual receipts, not predictions.
+  The minimum plan-count bound includes each real prefix decode plus both
+  selected-token decodes; extra scheduler split computations remain counted.
+- CPU backend has **no public use_ref getter**. Plan-to-backend ownership is
+  also unavailable publicly. Receipt says so. Actual plan flag readback plus
+  both genuine setter/callback routes is the proposed policy proof; unseen
+  routes, non-CPU backends, unexpected lifetimes or false flags refuse.
+- Original implementations resolve only through exact already-mapped absolute
+  `NOLOAD|NOW|LOCAL` providers; exact public symbol/path/inode/length checked.
+  Source/runtime/provider files are independently hashed by ROOT before/after.
+  Initial small ELF audit found preemptible public exports/PLT routes; that is
+  feasibility evidence only. Actual interposition/count checks must pass.
+
+## Numerical boundaries
+
+`use_ref` is the sole intentional native execution-policy change. It bypasses
+some generic CPU fusions/tinyBLAS/Hadamard/tiled paths according to pinned source.
+The original extra-buffer dispatch precedes ordinary op selection; CPU_REPACK
+codec/weight packing/dispatch remains untouched. No opaque packed data is
+decoded, disabled, replaced or supplied to a new operator. Fused GDN remains
+the original graph operation. This may still differ from AR1; no bit result is
+assumed and no tolerance changes are allowed.
+
+Plain/observed equality covers the original selected tail token and complete
+post-state after the same plain8191-token bulk prefix. It is not a second
+independent observed8192-token bulk run or an every-frontier observation.
+Independent accepted comparisons are still required for all730 final-token
+stages/full vocabulary and all42 complete occupied causal state domains.
+Gain/scale records keep their existing separate-probe labels. Neither fresh8K
+comparison recovers the lost historical teacher stream or closes every cause.
+
+## Ownership and bounds
+
+- Fresh exclusive output, original native deadline≤7200s, outer deadline+60s,
+  address space12GiB, CPU time≤8×wall+64s, file cap512MiB, log cap8MiB.
+- Explicit eight available CPU IDs; affinity installed before exec/DSO load,
+  checked before/after native execution and every actual public compute call.
+  Actual context thread settings remain8; individual plans may need1..8 tasks.
+- Exact reviewed `supervisor.py` retained unchanged. Public pidfd/WNOWAIT,
+  process-local subreaper, authenticated fresh SID, independent TERM/grace/
+  two KILL stages before any reap, first/final reap, group absence and pipe EOF.
+  Scope is owned SID only, not containment of a descendant creating a new SID.
+- All post-Popen receipt/input checks run inside cleanup `finally`. Failure,
+  partial incremental trace and original outputs remain; no retry/overwrite.
+- This is a bounded diagnostic host process, not physical ResourceArbiter
+  admission or any GPU/profile/serving/performance qualification.
+
+## ROOT execution, after quiet measurements
+
+Inspect `build-argv.json`; compile its exact argv under ROOT's bounded build
+coordinator. It links the original providers, adding only `dl`, `pthread` and
+four executable public exports. Retain compiler/argv/source/binary identity;
+the caller-supplied binary SHA alone cannot prove compilation provenance.
+No author build, linker, native/operator/model or lifecycle test was run.
+
+```sh
+run=.agents/runs/numerics-and-context-use-ref-bulk-control-20261020
+# ROOT only: prepared source-free parser fixtures first.
+python3 "$run/test_control.py"
+python3 "$run/test_supervisor.py"
+# ROOT authenticates the fresh compiled binary and chooses eight available CPUs.
+python3 "$run/run_ref_control.py" --model 0.8b --prefix 64 \
+  --cpus 0,1,2,3,4,5,6,7 --seconds 7200 \
+  --binary "$run/native_ref_control" --binary-sha256 ROOT_FRESH_BINARY_SHA \
+  --new-output "$run/root-0.8b-8192-ref64-1"
+python3 "$run/validate_control.py" --capture-run "$run/root-0.8b-8192-ref64-1" \
+  --new-report "$run/root-0.8b-8192-ref64-1/control-validation.json"
+```
+
+ROOT reviews actual policy/ownership/provider evidence before any next case.
+Use fresh names for2B or512; no automatic ladder. An unobserved route is a
+refusal to investigate, never authorization to fake a getter or force the flag
+inside compute. Numeric differences are recorded, not suppressed.
+
+Existing exact independent comparisons, ROOT only:
+
+```sh
+raw=.agents/runs/numerics-and-context-long-context-audit-20261020
+codec=.agents/runs/numerics-and-context-state-codec-20261020
+ids=.agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le
+ids_sha=44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d
+case="$run/root-0.8b-8192-ref64-1"
+model=target/models/qwen35-0.8b/model.gguf
+config=target/models/qwen35-0.8b/config.json
+model_sha=57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf
+# Supply the already accepted original Rust8192 capture directory; do not rebuild/relabel it.
+python3 "$raw/compare_raw.py" --rust ROOT_ACCEPTED_RUST_0_8B_8192_CAPTURE \
+  --native "$case/capture" --model "$model" --model-sha256 "$model_sha" \
+  --raw-ids "$ids" --raw-sha256 "$ids_sha" --cut 8192 --output "$case/raw-comparison.json"
+python3 "$codec/native_state.py" --run --capture "$case/capture" --model "$model" \
+  --config "$config" --raw-ids "$ids" --raw-sha256 "$ids_sha" --cut 8192 \
+  --output "$case/native-state"
+python3 "$codec/compare_state.py" --native "$case/native-state" \
+  --rust "$codec/root-rust-0.8b-8192" --output "$case/state-comparison.json"
+```
+
+Raw/state comparator exit2 means actual bit differences; malformed/refused
+evidence remains distinct. Both reports must be inspected and bound to this
+new capture before any control-result claim. No production arithmetic change
+follows automatically from either outcome.
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/TODO.md .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/TODO.md
new file mode 100644
index 0000000..99c2ab1
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/TODO.md
@@ -0,0 +1,27 @@
+# Public CPU use_ref fresh8K bulk control
+
+Parent: `.agents/plans/numerics-and-context/TODO.md`. Owner: `/root/batch_design_review`.
+Scope: this new ignored diagnostic run only; no MAIN source/shared-plan edits.
+
+- [x] Source-audit public actual context backend creation/abort extension and
+  reference setter; actual public cplan flag exists, no backend getter claimed.
+- [x] Small ELF metadata confirms candidate exported/PLT routes; no runtime-route claim.
+- [x] Prepare additive wrapper over byte-identical original16560B native client;
+  only policy=true, original operator/token schedule/CPU_REPACK unchanged.
+- [x] Prepare explicitCPU8/source-pinned finite one-case launcher using unchanged
+  reviewed pidfd/WNOWAIT/subreaper owned-SID supervisor; original failures retained.
+- [x] Prepare strict actual route/lifetime/count/affinity/capture validator and
+  source-free negative fixtures. Python AST only; no fixtures executed by author.
+- [x] Source/own-artifact freeze and ROOT-only build/capture/comparison instructions.
+- [ ] Independent source review and ROOT strict C++ build/ELF export inspection.
+- [ ] ROOT prepared parser/lifecycle fixtures; this launcher's cleanup unexecuted.
+- [ ] ROOT genuine new8192 refbulk capture: actual setter/plan flags, provider pins,
+  original730 records + plain/observed fullV/whole serialized state, allcleanup.
+- [ ] ROOT independent unchanged raw730/full248320 and all42 causal-state comparisons
+  against the accepted exact Rust AR1 capture/checkpoint, each source/scope explicit.
+- [ ] Only after reviewing the first result choose another model/64or512 case.
+
+Unqualified: all builds/runtimes/arithmetic results, full bulk cause closure,
+historical teacher recovery, CPU_REPACK layout decoding, GPU/HF/serving support.
+No existing source or accepted receipt was modified; original client/supervisor
+copies were compared byte-for-byte. No model/DSO payload hashes by author.
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/build-argv.json .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/build-argv.json
new file mode 100644
index 0000000..a95ad29
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/build-argv.json
@@ -0,0 +1,36 @@
+{
+  "status": "PREPARED_ROOT_BUILD_ONLY_UNEXECUTED",
+  "argv": [
+    "c++",
+    "-std=c++17",
+    "-O2",
+    "-Wall",
+    "-Wextra",
+    "-Werror",
+    "-I",
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include",
+    "-I",
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include",
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control.cpp",
+    "-L",
+    "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib",
+    "-L",
+    "/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib",
+    "-Wl,-rpath,/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib:/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib",
+    "-Wl,--dynamic-linker,/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib/ld-linux-x86-64.so.2",
+    "-Wl,--export-dynamic-symbol=ggml_backend_dev_init",
+    "-Wl,--export-dynamic-symbol=ggml_backend_reg_get_proc_address",
+    "-Wl,--export-dynamic-symbol=ggml_graph_compute",
+    "-Wl,--export-dynamic-symbol=ggml_backend_free",
+    "-lllama",
+    "-lggml",
+    "-lggml-base",
+    "-lm",
+    "-ldl",
+    "-pthread",
+    "-o",
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control"
+  ],
+  "compiler_identity": "ROOT must record real compiler and own finite build receipt before execution",
+  "native_math_source_unchanged": true
+}
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/cases.json .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/cases.json
new file mode 100644
index 0000000..6613657
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/cases.json
@@ -0,0 +1,35 @@
+{
+  "teachers": {
+    "path": ".agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le",
+    "bytes": 32768,
+    "sha256": "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d"
+  },
+  "models": {
+    "0.8b": {
+      "model": "target/models/qwen35-0.8b/model.gguf",
+      "config": "target/models/qwen35-0.8b/config.json",
+      "hidden": 1024,
+      "model_identity": {
+        "bytes": 563036064,
+        "sha256": "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf"
+      },
+      "config_identity": {
+        "bytes": 2907,
+        "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
+      }
+    },
+    "2b": {
+      "model": "target/models/qwen35-2b/model.gguf",
+      "config": "target/models/qwen35-2b/config.json",
+      "hidden": 2048,
+      "model_identity": {
+        "bytes": 1296764000,
+        "sha256": "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6"
+      },
+      "config_identity": {
+        "bytes": 2908,
+        "sha256": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
+      }
+    }
+  }
+}
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/common.py .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/common.py
new file mode 100644
index 0000000..c795f9e
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/common.py
@@ -0,0 +1,71 @@
+"""ROOT ONLY source/control validation; importing performs no work."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+
+HERE = Path(__file__).resolve().parent
+REPO = HERE.parents[2]
+TEACHER_SHA = '44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d'
+
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+
+def digest(path):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
+        before = os.fstat(stream.fileno())
+        require(stat.S_ISREG(before.st_mode), 'regular pinned input: ' + str(path))
+        result = hashlib.sha256()
+        for block in iter(lambda: stream.read(1024 * 1024), b''):
+            result.update(block)
+        after = os.fstat(stream.fileno())
+        require(all(getattr(before, field) == getattr(after, field)
+                    for field in ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns']),
+                'input changed during hash: ' + str(path))
+    return {'bytes': before.st_size, 'sha256': result.hexdigest()}
+
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON key')
+        result[key] = value
+    return result
+
+
+def bounded(path, cap):
+    path = Path(path)
+    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap,
+            'bounded regular artifact: ' + str(path))
+    with path.open('rb') as stream:
+        raw = stream.read(cap + 1)
+    require(len(raw) <= cap, 'artifact exceeds read bound')
+    return raw
+
+
+def jread(path, cap=1024 * 1024):
+    return json.loads(bounded(path, cap), object_pairs_hook=unique)
+
+
+def write(path, value):
+    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
+    require(len(raw) <= 1024 * 1024, 'bounded metadata receipt')
+    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
+    with os.fdopen(fd, 'wb') as stream:
+        stream.write(raw)
+
+
+def authenticate():
+    # Full model/provider hashing is ROOT execution-time work, never author prep.
+    frozen = jread(HERE / 'freeze-manifest.json')
+    for name, expected in frozen['files'].items():
+        require(digest(HERE / name) == expected, 'frozen own source differs: ' + name)
+    for name in ['original-source-pins.json', 'control-source-pins.json']:
+        for row in jread(HERE / name)['files']:
+            require(digest(row['path']) == {key: row[key] for key in ['bytes', 'sha256']},
+                    'pinned source/provider differs: ' + row['path'])
+    return jread(HERE / 'cases.json')
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/control-source-pins.json .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/control-source-pins.json
new file mode 100644
index 0000000..b074007
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/control-source-pins.json
@@ -0,0 +1,70 @@
+{
+  "scope": "source-derived public ABI/control only; arithmetic unexecuted by author",
+  "files": [
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/src/llama-context.cpp",
+      "bytes": 160236,
+      "sha256": "904c41cb688900936db135159f37c1b9803bdb79ce04a25c326400078a82532f"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml-cpu.h",
+      "bytes": 7753,
+      "sha256": "316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend-reg.cpp",
+      "bytes": 18555,
+      "sha256": "0f63c69ad083e0744872f57d59d049b069283303f5bbc646222907d779afe4d1"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend.cpp",
+      "bytes": 105561,
+      "sha256": "cdb79f2937aa147084a76f949adc0eb760ad1883e0ccc015bb4e0f599c89bd5f"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ggml-cpu.cpp",
+      "bytes": 25005,
+      "sha256": "ccf382d3fbf126b5ddb5850a0a970656fb24e00587da4003419439277c91cd01"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ggml-cpu.c",
+      "bytes": 132440,
+      "sha256": "45b22f8c83b2cbccabfc2f81a38ef8a5810223e1ce366b7c6f77d42faa5cc6f3"
+    },
+    {
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-long-context-audit-20261020/native_raw.cpp",
+      "bytes": 16560,
+      "sha256": "387e0fdf2d16849ecfe574cd6d808e53c8ae4c2b0975254404a5ef88268eaea6"
+    },
+    {
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-compiled-f32-routes-20261020/supervisor.py",
+      "bytes": 10131,
+      "sha256": "f422e0f8b385494a7c28be775a1eac1f3338e634096bfc8fd6137a1316547cc6"
+    },
+    {
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-long-context-audit-20261020/compare_raw.py",
+      "bytes": 18504,
+      "sha256": "8ecd7a0b0bf5a4f31e63886362825e56051c22a91337f57e3ded4c85b2a6ffd2"
+    },
+    {
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-state-codec-20261020/native_state.py",
+      "bytes": 13859,
+      "sha256": "771a7937efc4fdc8de1f96462c7aaaeb6d247be24715fe2ea216c5860a836430"
+    },
+    {
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-state-codec-20261020/compare_state.py",
+      "bytes": 6530,
+      "sha256": "3077b1c46fe36c2964f8c447d191c296fb02e67d222621804c4fe1270e210f68"
+    },
+    {
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-state-codec-20261020/source-profile.json",
+      "bytes": 5482,
+      "sha256": "778ff148ad1094ae341555b027c64de0808e607cdb63eca9482c93867f849da1"
+    },
+    {
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-compiled-f32-routes-20261020/test_supervisor.py",
+      "bytes": 3938,
+      "sha256": "6e3f9e9b42892edf9f0aa939bfd4fd446afee4314413f69537d4a9a9dfecdff4"
+    }
+  ]
+}
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_raw_original.cpp .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_raw_original.cpp
new file mode 100644
index 0000000..9d64f8e
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_raw_original.cpp
@@ -0,0 +1,272 @@
+// Test-only pinned llama.cpp client. Never a production execution dependency.
+// Model/source hashes are independently checked by compare_raw.py.
+// argv: GGUF ids.u32le cut prefix_batch ubatch threads seconds output_directory
+#include "llama.h"
+#include "ggml-backend.h"
+#include <algorithm>
+#include <chrono>
+#include <cmath>
+#include <cstdio>
+#include <cstring>
+#include <filesystem>
+#include <fstream>
+#include <iomanip>
+#include <iostream>
+#include <limits>
+#include <memory>
+#include <set>
+#include <sstream>
+#include <stdexcept>
+#include <string>
+#include <vector>
+#include <sys/stat.h>
+#include <gnu/libc-version.h>
+
+namespace fs = std::filesystem;
+constexpr size_t CAPTURE_CAP = 64 * 1024 * 1024;
+constexpr size_t STATE_CAP = 512 * 1024 * 1024;
+static fs::path folder;
+static std::ofstream records;
+static std::set<std::pair<int,std::string>> seen;
+static size_t captured_bytes = 0;
+static int cut = 0;
+static int active_gdn_count = 0;
+static bool enabled = false;
+static std::chrono::steady_clock::time_point deadline;
+
+static void require(bool good, const std::string &what) {
+    if (!good) throw std::runtime_error(what);
+}
+static void check_time() {
+    require(std::chrono::steady_clock::now() < deadline, "bounded reference deadline exceeded");
+}
+static std::string quote(const std::string &s) {
+    std::ostringstream out; out << '"';
+    for (unsigned char c : s) {
+        if (c=='"' || c=='\\') out << '\\' << c;
+        else if (c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
+        else out << c;
+    }
+    return out.str()+'"';
+}
+static int integer(const char *text, int maximum) {
+    std::string s(text); size_t at=0;
+    long n=std::stol(s,&at);
+    require(at==s.size() && n>=1 && n<=maximum,"invalid bounded positive argument");
+    return static_cast<int>(n);
+}
+static std::vector<unsigned char> read_bounded(const fs::path &p, size_t cap) {
+    std::ifstream in(p,std::ios::binary|std::ios::ate);
+    require(bool(in),"cannot open bounded input");
+    const auto end=in.tellg();
+    require(end>=0 && uint64_t(end)<=cap,"input exceeds bound");
+    std::vector<unsigned char> bytes(static_cast<size_t>(end));
+    in.seekg(0); in.read(reinterpret_cast<char *>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
+    require(bool(in) && in.peek()==std::char_traits<char>::eof(),"input read changed");
+    return bytes;
+}
+static void write_bytes(const fs::path &p, const void *data, size_t bytes) {
+    require(!fs::exists(p),"refusing artifact overwrite");
+    std::ofstream out(p,std::ios::binary); require(bool(out),"artifact open");
+    out.write(static_cast<const char *>(data),static_cast<std::streamsize>(bytes));
+    out.flush(); require(bool(out),"artifact write");
+    require(chmod(p.c_str(),0600)==0,"artifact mode");
+}
+static void emit(int layer,const std::string &stage,const std::vector<float> &values,
+                 const char *origin="live_native_tensor",const ggml_tensor *tensor=nullptr) {
+    require(!values.empty() && values.size()<=2*1024*1024,"capture element cap");
+    require(captured_bytes+values.size()*4<=CAPTURE_CAP,"aggregate capture cap");
+    require(seen.emplace(layer,stage).second,"duplicate native semantic record: "+stage);
+    for(float x:values) require(std::isfinite(x),"nonfinite native capture");
+    const std::string file="record-"+std::to_string(seen.size())+".f32le";
+    write_bytes(folder/file,values.data(),values.size()*4);
+    captured_bytes+=values.size()*4;
+    records << "{\"position\":" << cut-1 << ",\"layer\":" << layer
+            << ",\"stage\":" << quote(stage) << ",\"elements\":" << values.size()
+            << ",\"file\":" << quote(file) << ",\"origin\":" << quote(origin);
+    if (tensor) {
+        records << ",\"native_name\":" << quote(tensor->name) << ",\"native_ne\":[";
+        for(int i=0;i<4;i++)records<<(i?",":"")<<tensor->ne[i];
+        records << "],\"native_nb\":[";
+        for(int i=0;i<4;i++)records<<(i?",":"")<<tensor->nb[i];
+        records << "]";
+    }
+    records << "}\n"; require(bool(records),"record write");
+}
+static std::vector<float> dense(const ggml_tensor *t) {
+    require(t && t->type==GGML_TYPE_F32,"native tensor dtype");
+    const int64_t count=ggml_nelements(t);
+    const size_t bytes=ggml_nbytes(t);
+    require(count>0 && count<=2*1024*1024 && bytes<=16*1024*1024,"native tensor span cap");
+    std::vector<unsigned char> raw(bytes);
+    ggml_backend_tensor_get(t,raw.data(),0,raw.size());
+    std::vector<float> result; result.reserve(static_cast<size_t>(count));
+    for(int64_t d=0;d<t->ne[3];d++)for(int64_t c=0;c<t->ne[2];c++)
+    for(int64_t b=0;b<t->ne[1];b++)for(int64_t a=0;a<t->ne[0];a++) {
+        const size_t offset=a*t->nb[0]+b*t->nb[1]+c*t->nb[2]+d*t->nb[3];
+        require(offset+4<=raw.size(),"native tensor stride outside span");
+        float x; std::memcpy(&x,raw.data()+offset,4); result.push_back(x);
+    }
+    require(result.size()==static_cast<size_t>(count),"native dense coverage");
+    return result;
+}
+static int named_layer(const std::string &name,const std::string &base) {
+    const std::string prefix=base+"-";
+    if(name.rfind(prefix,0)!=0)return -1;
+    const std::string suffix=name.substr(prefix.size());
+    if(suffix.empty() || suffix.find_first_not_of("0123456789")!=std::string::npos)return -1;
+    const int layer=std::stoi(suffix); return layer>=0 && layer<24 ? layer : -1;
+}
+static std::pair<int,std::string> semantic(const ggml_tensor *t) {
+    const std::string name=t->name;
+    if(name=="model.input_embed")return {-1,"embedding"};
+    if(name=="result_norm")return {-1,"final_norm"};
+    for(const auto &pair:std::vector<std::pair<std::string,std::string>>{
+        {"attn_norm","attention_input_norm"},{"linear_attn_out","attention_output"},
+        {"attn_residual","post_attention_residual"},{"attn_post_norm","ffn_input_norm"},
+        {"ffn_out","ffn_output"},{"l_out","post_ffn_residual"},
+        {"linear_attn_qkv_mixed","gdn_qkv_raw"},{"z","gdn_z_raw"},
+        {"beta","gdn_beta_raw"},{"beta_sigmoid","gdn_beta"},
+        {"alpha","gdn_alpha"},{"a_softplus","gdn_alpha_softplus"},
+        {"gate","gdn_gate_raw"},{"conv_output_raw","gdn_conv_raw"},
+        {"conv_output_silu","gdn_conv_silu"},{"conv_states_reshaped","gdn_conv_prior"},
+        {"final_output","gdn_gated_output"}}) {
+        const int layer=named_layer(name,pair.first);
+        if(layer>=0)return {layer,pair.second};
+    }
+    const int layer=named_layer(name,"attn_output");
+    if(layer>=0 && layer%4==3)return {layer,"attention_output"};
+    return {-2,""};
+}
+static bool observe(ggml_tensor *t,bool ask,void*) {
+    if(!enabled || !t)return false;
+    const auto key=semantic(t);
+    const bool projection=t->op==GGML_OP_MUL_MAT && t->src[0] &&
+        std::string(t->src[0]->name).find(".weight")!=std::string::npos;
+    const int conv_copy=t->op==GGML_OP_CPY && t->src[0]
+        ? named_layer(t->src[0]->name,"conv_state_last") : -1;
+    const bool wanted=key.first!=-2 || t->op==GGML_OP_GATED_DELTA_NET || projection || conv_copy>=0;
+    if(ask)return wanted;
+    require(wanted,"unexpected native callback");
+    check_time();
+    if(projection)emit(-1,"projection_input::"+std::string(t->src[0]->name),dense(t->src[1]),"live_native_projection_operand",t->src[1]);
+    if(conv_copy>=0)emit(conv_copy,"gdn_conv_state",dense(t),"live_native_cache_copy",t);
+    if(key.first!=-2) {
+        const auto values=dense(t); emit(key.first,key.second,values,"live_native_tensor",t);
+        if(key.second=="gdn_alpha_softplus") {
+            require(t->src[0] && t->src[0]->op==GGML_OP_ADD && t->src[0]->src[1],"softplus bias source binding");
+            emit(key.first,"gdn_dt",dense(t->src[0]->src[1]),"live_native_parameter_operand",t->src[0]->src[1]);
+        }
+        if(key.second=="gdn_gate_raw") {
+            require(t->op==GGML_OP_MUL && t->src[1],"A source binding");
+            emit(key.first,"gdn_a",dense(t->src[1]),"live_native_parameter_operand",t->src[1]);
+            std::vector<float> gain; for(float gate:values)gain.push_back(::expf(gate));
+            emit(key.first,"gdn_gain",gain,"separate_pinned_libm_expf_probe_not_live_gain");
+        }
+    }
+    if(t->op==GGML_OP_GATED_DELTA_NET) {
+        for(int i=0;i<6;i++)require(t->src[i],"native GDN required operand missing");
+        const int layer=named_layer(t->src[5]->name,"state_predelta");
+        require(layer>=0 && layer%4!=3,"native GDN source layer binding");
+        require(t->src[2]->ne[0]==128 && t->src[2]->ne[1]==16 && t->src[2]->ne[2]==1 && t->src[2]->ne[3]==1,
+                "selected native GDN is not one real causal token/16x128");
+        int32_t snapshots=0; std::memcpy(&snapshots,t->op_params,4);
+        require(snapshots==1,"native GDN snapshot policy changed");
+        for(int i=0;i<3;i++)emit(layer,std::vector<std::string>{"gdn_q_norm","gdn_k_norm","gdn_v"}[i],dense(t->src[i]),"live_native_GDN_operand",t->src[i]);
+        emit(layer,"gdn_prior_state",dense(t->src[5]),"live_native_GDN_prior_operand",t->src[5]);
+        emit(layer,"gdn_scale",std::vector<float>(16,1.0f/::sqrtf(128.0f)),"separate_pinned_source_width_scale_not_live_tensor");
+        const auto packed=dense(t);
+        constexpr size_t core=16*128, state=16*128*128;
+        require(packed.size()==core+state,"native fused GDN output coverage");
+        emit(layer,"gdn_core_output",std::vector<float>(packed.begin(),packed.begin()+core),"live_native_GDN_output",t);
+        emit(layer,"gdn_state",std::vector<float>(packed.begin()+core,packed.end()),"live_native_GDN_proposed_state",t);
+        active_gdn_count++;
+    }
+    return true;
+}
+struct Batch {
+    llama_batch batch;
+    explicit Batch(int n):batch(llama_batch_init(n,0,1)) {}
+    ~Batch(){llama_batch_free(batch);}
+};
+static void decode(llama_context *ctx,const std::vector<llama_token> &ids,int start,int count,bool logits) {
+    check_time(); Batch holder(count); auto &batch=holder.batch; batch.n_tokens=count;
+    for(int i=0;i<count;i++) {
+        batch.token[i]=ids[start+i]; batch.pos[i]=start+i;
+        batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=logits && i+1==count;
+    }
+    require(llama_decode(ctx,batch)==0,"native decode failed");
+    check_time();
+}
+static std::vector<unsigned char> state(llama_context *ctx) {
+    const size_t n=llama_state_get_size(ctx);require(n>0 && n<=STATE_CAP,"native opaque state cap");
+    std::vector<unsigned char> out(n);
+    require(llama_state_get_data(ctx,out.data(),out.size())==n,"native opaque state serialization");
+    return out;
+}
+int main(int argc,char **argv) {
+    try {
+        require(argc==9,"usage: native_raw GGUF ids.u32le cut prefix_batch ubatch threads seconds output_directory");
+        const uint32_t endian=1;require(*reinterpret_cast<const unsigned char *>(&endian)==1 && sizeof(float)==4,"reference requires little-endian f32 host");
+        cut=integer(argv[3],8192);const int prefix=integer(argv[4],512),ubatch=integer(argv[5],512),threads=integer(argv[6],8),seconds=integer(argv[7],7200);
+        deadline=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
+        folder=argv[8];require(fs::create_directory(folder),"output directory must be new");
+        require(chmod(folder.c_str(),0700)==0,"output directory mode");
+        records.open(folder/"records.jsonl");require(bool(records),"record stream open");
+        const auto input=read_bounded(argv[2],8192*4);
+        require(!input.empty() && input.size()%4==0 && size_t(cut)*4<=input.size(),"bounded raw ID stream");
+        std::vector<llama_token> ids(input.size()/4);
+        for(size_t i=0;i<ids.size();i++){uint32_t id;std::memcpy(&id,input.data()+i*4,4);require(id<248320,"raw source token outside vocabulary");ids[i]=static_cast<llama_token>(id);}
+        write_bytes(folder/"input_ids.u32le",input.data(),input.size());
+        llama_backend_init();
+        auto mp=llama_model_default_params();mp.n_gpu_layers=0;
+        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
+        require(bool(model),"pinned native model load");
+        require(llama_model_n_layer(model.get())==24 && llama_vocab_n_tokens(llama_model_get_vocab(model.get()))==248320,"accepted original model geometry");
+        const int hidden=llama_model_n_embd(model.get());require(hidden==1024 || hidden==2048,"accepted hidden width");
+        auto params=llama_context_default_params();
+        params.n_ctx=8320;params.n_batch=prefix;params.n_ubatch=ubatch;
+        params.n_threads=threads;params.n_threads_batch=threads;
+        params.type_k=GGML_TYPE_F32;params.type_v=GGML_TYPE_F32;
+        params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
+        params.offload_kqv=false;params.op_offload=false;params.no_perf=true;
+        params.cb_eval=nullptr;
+        std::unique_ptr<llama_context,decltype(&llama_free)> plain(llama_init_from_model(model.get(),params),llama_free);
+        require(bool(plain),"plain native context");
+        for(int at=0;at<cut-1;) {const int n=std::min(prefix,cut-1-at);decode(plain.get(),ids,at,n,false);at+=n;}
+        const auto prior=state(plain.get());
+        decode(plain.get(),ids,cut-1,1,true);
+        const auto *raw_logits=llama_get_logits_ith(plain.get(),-1);require(raw_logits,"plain full logits missing");
+        std::vector<float> logits(raw_logits,raw_logits+248320);
+        const auto after=state(plain.get());
+        params.cb_eval=observe;
+        std::unique_ptr<llama_context,decltype(&llama_free)> observed(llama_init_from_model(model.get(),params),llama_free);
+        require(bool(observed),"observed native context");
+        require(llama_state_set_data(observed.get(),prior.data(),prior.size())==prior.size(),"native same-source prior restore");
+        enabled=true;decode(observed.get(),ids,cut-1,1,true);enabled=false;
+        const auto *observed_logits=llama_get_logits_ith(observed.get(),-1);require(observed_logits,"observed full logits missing");
+        require(std::memcmp(logits.data(),observed_logits,logits.size()*4)==0,"instrumentation changed original native full logits");
+        const auto observed_state=state(observed.get());
+        require(after==observed_state,"instrumentation changed original native serialized state");
+        require(active_gdn_count==18,"missing actual native fused GDN callbacks");
+        emit(-1,"logits",logits,"unobserved_native_full_vocabulary");
+        write_bytes(folder/"native.prior-state.bin",prior.data(),prior.size());
+        write_bytes(folder/"native.post-state.bin",after.data(),after.size());
+        records.flush();require(bool(records),"final records flush");records.close();
+        std::ofstream done(folder/"complete.json");require(bool(done),"complete report open");
+        done << "{\"status\":\"CAPTURE_COMPLETE_NOT_COMPARISON\",\"provider\":\"pinned_native_llama_cpp\",\"model_path\":" << quote(fs::canonical(argv[1]).string())
+             << ",\"cut\":" << cut << ",\"prefix_batch_requested\":" << prefix << ",\"ubatch_requested\":" << ubatch
+             << ",\"hidden_width\":" << hidden << ",\"actual_shared_model_tokens\":" << cut+1
+             << ",\"n_ctx_actual\":" << llama_n_ctx(plain.get()) << ",\"n_batch_actual\":" << llama_n_batch(plain.get())
+             << ",\"n_ubatch_actual\":" << llama_n_ubatch(plain.get()) << ",\"threads_actual\":" << llama_n_threads(plain.get())
+             << ",\"threads_batch_actual\":" << llama_n_threads_batch(plain.get()) << ",\"F32_KV\":true,\"flash_attention\":false,\"CPU_only\":true"
+             << ",\"actual_selected_token_GDN_ops\":18,\"instrumentation_full_logits_exact\":true,\"instrumentation_whole_native_state_exact\":true"
+             << ",\"captured_bytes\":" << captured_bytes << ",\"record_count\":" << seen.size()
+             << ",\"native_prior_state_bytes\":" << prior.size() << ",\"native_post_state_bytes\":" << after.size()
+             << ",\"fresh_prefix_distinct_from_lost_historical_prefix\":true,\"glibc_version\":" << quote(gnu_get_libc_version())
+             << ",\"system_info\":" << quote(llama_print_system_info()) << "}\n";
+        done.flush();require(bool(done),"complete report write");
+        std::cerr << "NATIVE_RAW_CAPTURE_COMPLETE cut=" << cut << " records=" << seen.size() << " selected_token_only=true source_equality=not_yet_compared\n";
+        observed.reset();plain.reset();model.reset();llama_backend_free();return 0;
+    } catch(const std::exception &error) {std::cerr<<"NATIVE_RAW_UNQUALIFIED "<<error.what()<<'\n';return 2;}
+}
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control.cpp .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control.cpp
new file mode 100644
index 0000000..b26ad98
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control.cpp
@@ -0,0 +1,247 @@
+// Diagnostic-only public CPU reference-policy control. Original token loop included unchanged.
+#ifndef _GNU_SOURCE
+#define _GNU_SOURCE
+#endif
+#include "ggml-cpu.h"
+#include <dlfcn.h>
+#include <algorithm>
+#include <fcntl.h>
+#include <sched.h>
+#include <sys/stat.h>
+#include <unistd.h>
+#include <cerrno>
+#include <cstdint>
+#include <cstdio>
+#include <cstdlib>
+#include <cstring>
+#include <fstream>
+#include <map>
+#include <mutex>
+#include <string>
+#include <vector>
+
+namespace ref_control {
+constexpr const char *BASE="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml-base.so.0.26.0";
+constexpr const char *CPU="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so";
+constexpr size_t BASE_BYTES=972584, CPU_BYTES=1459208, MAX_RECORDS=65536;
+static bool armed=false;
+struct Backend {size_t init=0,abort=0,set=0;bool freed=false;};
+static std::map<ggml_backend_t,Backend> backends;
+static std::mutex mutex;
+static size_t trace_records=0,trace_bytes=0;
+static int trace_fd=-1;
+static std::string trace_path;
+static size_t graph_calls=0,graph_success=0;
+static int thread_min=9,thread_max=0;
+static cpu_set_t expected_affinity;
+static std::vector<int> declared_cpus;
+using Init=ggml_backend_t(*)(ggml_backend_dev_t,const char *);
+using Lookup=void *(*)(ggml_backend_reg_t,const char *);
+using Free=void(*)(ggml_backend_t);
+using IsCpu=bool(*)(ggml_backend_t);
+using Ref=void(*)(ggml_backend_t,bool);
+using Compute=ggml_status(*)(ggml_cgraph *,ggml_cplan *);
+
+[[noreturn]] static void fatal(const char *message) {
+    const int saved=errno;
+    const char prefix[]="USE_REF_CONTROL_UNQUALIFIED ";
+    const ssize_t a=write(STDERR_FILENO,prefix,sizeof(prefix)-1);
+    const ssize_t b=write(STDERR_FILENO,message,strnlen(message,512));
+    const ssize_t c=write(STDERR_FILENO,"\n",1);
+    (void)a;(void)b;(void)c;errno=saved;_exit(125);
+}
+static void checked(bool value,const char *message) {if(!value)fatal(message);}
+static void check_affinity() {
+    cpu_set_t actual;CPU_ZERO(&actual);
+    checked(sched_getaffinity(0,sizeof(actual),&actual)==0
+        && CPU_EQUAL(&actual,&expected_affinity),"actual calling thread CPU8 affinity");
+}
+static void append(const std::string &line) {
+    checked(trace_records<MAX_RECORDS && line.size()<2048
+        && trace_bytes+line.size()+1<=8*1024*1024,"bounded public trace");
+    if(trace_fd<0) {
+        trace_fd=open(trace_path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
+        checked(trace_fd>=0,"exclusive public trace open");
+    }
+    const auto record=line+"\n";
+    size_t offset=0;int interruptions=0;
+    while(offset<record.size()) {
+        const ssize_t written=write(trace_fd,record.data()+offset,record.size()-offset);
+        if(written<0 && errno==EINTR && ++interruptions<=8)continue;
+        checked(written>0,"public trace write");offset+=static_cast<size_t>(written);
+    }
+    ++trace_records;trace_bytes+=record.size();
+}
+static std::string pointer(const void *value) {
+    char text[2+sizeof(void*)*2+1]={};
+    const int size=std::snprintf(text,sizeof(text),"%p",value);
+    checked(size>0 && size<static_cast<int>(sizeof(text)),"opaque identity formatting");
+    return std::string(text);
+}
+static void attest(void *address,const char *path,size_t length,const char *name) {
+    Dl_info info{};struct stat actual{},expected{};
+    checked(address && dladdr(address,&info)!=0 && info.dli_fname,"public symbol lookup");
+    checked(std::strcmp(info.dli_fname,path)==0,"public symbol exact loaded provider path");
+    checked(stat(path,&expected)==0 && stat(info.dli_fname,&actual)==0
+        && expected.st_dev==actual.st_dev && expected.st_ino==actual.st_ino
+        && expected.st_size==static_cast<off_t>(length),"public symbol provider inode/length");
+    checked(info.dli_sname && std::strcmp(info.dli_sname,name)==0,"public symbol exact exported name");
+}
+template<class F> static F symbol(const char *path,size_t bytes,const char *name) {
+    // NOLOAD never acquires/replaces an unregistered CPU implementation. Hold
+    // the original module through backend_free and the final observation.
+    void *handle=dlopen(path,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
+    checked(handle!=nullptr,"exact original provider is not already loaded");
+    dlerror();void *address=dlsym(handle,name);const char *error=dlerror();
+    checked(error==nullptr,"public symbol resolution failed");attest(address,path,bytes,name);
+    return reinterpret_cast<F>(address);
+}
+static Lookup lookup() {static Lookup f=symbol<Lookup>(BASE,BASE_BYTES,"ggml_backend_reg_get_proc_address");return f;}
+static ggml_backend_reg_t registry(ggml_backend_t backend) {
+    auto get_device=symbol<decltype(&ggml_backend_get_device)>(BASE,BASE_BYTES,"ggml_backend_get_device");
+    auto get_reg=symbol<decltype(&ggml_backend_dev_backend_reg)>(BASE,BASE_BYTES,"ggml_backend_dev_backend_reg");
+    const auto device=get_device(backend);checked(device!=nullptr,"actual backend has public device");
+    const auto reg=get_reg(device);checked(reg!=nullptr,"actual backend has public registry");return reg;
+}
+static void set_reference(ggml_backend_t backend,bool initialization) {
+    static IsCpu is_cpu=symbol<IsCpu>(CPU,CPU_BYTES,"ggml_backend_is_cpu");
+    checked(backend && is_cpu(backend),"nonCPU backend in CPU-only control");
+    void *raw=lookup()(registry(backend),"ggml_backend_cpu_set_use_ref");
+    attest(raw,CPU,CPU_BYTES,"ggml_backend_cpu_set_use_ref");
+    reinterpret_cast<Ref>(raw)(backend,true); // Sole intentional policy change.
+    std::lock_guard<std::mutex> guard(mutex);
+    auto &record=backends[backend];checked(!record.freed && backends.size()<=2,"exact two live context CPU backends");
+    (initialization?record.init:record.abort)++;record.set++;
+    append("{\"kind\":\"public_cpu_reference_setter\",\"backend\":\""+pointer(backend)
+        +"\",\"origin\":\""+(initialization?"public_dev_init":"existing_public_abort_setter_callback")
+        +"\",\"requested_use_ref\":true}");
+}
+static void abort_setter(ggml_backend_t backend,ggml_abort_callback callback,void *data) {
+    const int saved=errno;
+    void *raw=lookup()(registry(backend),"ggml_backend_set_abort_callback");
+    attest(raw,CPU,CPU_BYTES,"ggml_backend_cpu_set_abort_callback");
+    errno=saved;reinterpret_cast<ggml_backend_set_abort_callback_t>(raw)(backend,callback,data);
+    const int after=errno;set_reference(backend,false);errno=after;
+}
+}
+
+extern "C" ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device,const char *params) {
+    const int saved=errno;
+    static ref_control::Init original=ref_control::symbol<ref_control::Init>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_dev_init");
+    errno=saved;auto backend=original(device,params);const int after=errno;
+    if(ref_control::armed){ref_control::checked(backend!=nullptr,"native backend initialization failed");ref_control::set_reference(backend,true);}
+    errno=after;return backend;
+}
+extern "C" void *ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg,const char *name) {
+    const int saved=errno;auto original=ref_control::lookup();errno=saved;
+    void *value=original(reg,name);const int after=errno;
+    if(ref_control::armed && value && name && std::strcmp(name,"ggml_backend_set_abort_callback")==0) {
+        ref_control::attest(value,ref_control::CPU,ref_control::CPU_BYTES,"ggml_backend_cpu_set_abort_callback");
+        value=reinterpret_cast<void *>(&ref_control::abort_setter);
+    }
+    errno=after;return value;
+}
+extern "C" enum ggml_status ggml_graph_compute(struct ggml_cgraph *graph,struct ggml_cplan *plan) {
+    const int saved=errno;
+    static ref_control::Compute original=ref_control::symbol<ref_control::Compute>(ref_control::CPU,
+        ref_control::CPU_BYTES,"ggml_graph_compute");
+    if(ref_control::armed) {
+        ref_control::check_affinity();
+        ref_control::checked(graph && plan && plan->use_ref,"actual PUBLIC cplan.use_ref is false/missing");
+        ref_control::checked(plan->n_threads>=1 && plan->n_threads<=8,"actual public plan thread bound");
+        std::lock_guard<std::mutex> guard(ref_control::mutex);
+        ++ref_control::graph_calls;
+        ref_control::thread_min=std::min(ref_control::thread_min,plan->n_threads);
+        ref_control::thread_max=std::max(ref_control::thread_max,plan->n_threads);
+        ref_control::append("{\"kind\":\"actual_public_graph_plan\",\"sequence\":"+std::to_string(ref_control::graph_calls)
+            +",\"use_ref\":true,\"n_threads\":"+std::to_string(plan->n_threads)+"}");
+    }
+    errno=saved;const auto status=original(graph,plan);const int after=errno;
+    if(ref_control::armed) {
+        ref_control::checked(plan->use_ref,"actual public compute changed reference flag");
+        std::lock_guard<std::mutex> guard(ref_control::mutex);
+        if(status==GGML_STATUS_SUCCESS)++ref_control::graph_success;
+        else ref_control::fatal("actual public compute returned failure");
+    }
+    errno=after;return status;
+}
+extern "C" void ggml_backend_free(ggml_backend_t backend) {
+    const int saved=errno;
+    static ref_control::Free original=ref_control::symbol<ref_control::Free>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_free");
+    if(ref_control::armed && backend) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);
+        auto found=ref_control::backends.find(backend);
+        ref_control::checked(found!=ref_control::backends.end() && !found->second.freed,"unknown/duplicate backend retirement");
+        found->second.freed=true;
+        ref_control::append("{\"kind\":\"public_backend_free\",\"backend\":\""+ref_control::pointer(backend)+"\"}");
+    }
+    errno=saved;original(backend);
+}
+
+#define main zg_unchanged_native_raw_main
+#include "native_raw_original.cpp"
+#undef main
+
+int main(int argc,char **argv) {
+    try {
+        require(argc==9,"same original native_raw arguments required");
+        require(std::string(argv[3])=="8192" && std::string(argv[6])=="8",
+            "closed fresh8192/CPU8 control");
+        require((std::string(argv[4])=="64" || std::string(argv[4])=="512")
+            && std::string(argv[5])==argv[4],"predeclared same bulk batch/ubatch64 or512");
+        const char *cpu_env=std::getenv("ZG_USE_REF_CPU_SET");
+        require(cpu_env && std::strlen(cpu_env)<=128,"explicit ROOT CPU8 affinity declaration");
+        CPU_ZERO(&ref_control::expected_affinity);
+        std::istringstream cpu_input(cpu_env);std::string cpu_text;
+        while(std::getline(cpu_input,cpu_text,',')) {
+            require(!cpu_text.empty() && std::all_of(cpu_text.begin(),cpu_text.end(),
+                [](unsigned char c){return c>='0' && c<='9';}),"canonical CPU number");
+            const unsigned long cpu=std::stoul(cpu_text);
+            require(cpu<CPU_SETSIZE && !CPU_ISSET(cpu,&ref_control::expected_affinity),"unique bounded CPU number");
+            CPU_SET(cpu,&ref_control::expected_affinity);ref_control::declared_cpus.push_back(static_cast<int>(cpu));
+        }
+        require(ref_control::declared_cpus.size()==8,"exact eight declared CPUs");
+        ref_control::check_affinity();
+        ref_control::trace_path=std::string(argv[8])+"/use-ref-public-trace.jsonl";
+        ref_control::armed=true;
+        const int result=zg_unchanged_native_raw_main(argc,argv);
+        require(result==0,"unchanged native capture failed");
+        std::lock_guard<std::mutex> guard(ref_control::mutex);
+        require(ref_control::backends.size()==2,"both actual plain/observed CPU contexts observed");
+        for(const auto &item:ref_control::backends) {
+            require(item.second.init==1 && item.second.abort==1 && item.second.set==2 && item.second.freed,
+                "every actual CPU context initialized/configured/retired through public APIs");
+        }
+        const size_t minimum_calls=(8191+std::stoul(argv[4])-1)/std::stoul(argv[4])+2;
+        require(ref_control::graph_calls>=minimum_calls && ref_control::graph_calls==ref_control::graph_success,
+            "actual public cplan readback required, no missing/failed route accepted");
+        ref_control::check_affinity();
+        require(ref_control::trace_fd>=0 && fsync(ref_control::trace_fd)==0,"public trace complete");
+        require(close(ref_control::trace_fd)==0,"public trace close");ref_control::trace_fd=-1;
+        std::ifstream maps("/proc/self/maps",std::ios::binary);
+        std::vector<char> text(2*1024*1024+1);
+        maps.read(text.data(),static_cast<std::streamsize>(text.size()));
+        const auto count=maps.gcount();
+        require(maps.eof() && count>0 && count<=2*1024*1024,"bounded actual loaded mappings");
+        write_bytes(folder/"use-ref-loaded-maps.txt",text.data(),static_cast<size_t>(count));
+        std::ifstream status("/proc/self/status",std::ios::binary);
+        std::vector<char> status_text(64*1024+1);
+        status.read(status_text.data(),static_cast<std::streamsize>(status_text.size()));
+        const auto status_count=status.gcount();
+        require(status.eof() && status_count>0 && status_count<=64*1024,"bounded actual process status");
+        write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
+        std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
+        report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\",\"CPU_backend_count\":2,"
+            <<"\"public_setter_calls\":4,\"existing_abort_setter_calls\":2,\"actual_public_plan_calls\":"<<ref_control::graph_calls
+            <<",\"actual_successful_public_plan_calls\":"<<ref_control::graph_success
+            <<",\"actual_plan_threads_min\":"<<ref_control::thread_min<<",\"actual_plan_threads_max\":"<<ref_control::thread_max
+            <<",\"actual_public_plan_use_ref_all_true\":true,\"actual_CPU_affinity\":[";
+        for(size_t i=0;i<ref_control::declared_cpus.size();++i)report<<(i?",":"")<<ref_control::declared_cpus[i];
+        report<<"],\"backend_private_getter_available\":false,"
+            <<"\"plan_to_backend_identity_publicly_available\":false,\"CPU_REPACK_policy_unchanged\":true,"
+            <<"\"native_raw_token_loop_unchanged\":true,\"historical_prefix_or_cause_closed\":false}\n";
+        report.flush();require(bool(report),"control report write");return 0;
+    }catch(const std::exception &error){std::cerr<<"USE_REF_CONTROL_UNQUALIFIED "<<error.what()<<'\n';return 2;}
+}
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/original-source-pins.json .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/original-source-pins.json
new file mode 100644
index 0000000..da07d29
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/original-source-pins.json
@@ -0,0 +1,66 @@
+{
+  "scope": "source/binary identity inspection only; no native or model execution",
+  "pinned_source_receipt": ".agents/runs/numerics-and-context-root-20261020/source-acquisition.json",
+  "native_revision": "d812350",
+  "native_build": 11429,
+  "separate_glibc": "/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib",
+  "historical_raw_prefix_recovered": false,
+  "fresh_inputs": ".agents/runs/numerics-and-context-root-20261020/fresh-prefix-cases.json",
+  "files": [
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include/llama.h",
+      "bytes": 89978,
+      "sha256": "c9a5f025ad7dbf07eda6660c8ff2c32cd80c07ba440d80728707a9c9d13c1bc3"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml.h",
+      "bytes": 116144,
+      "sha256": "12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml-backend.h",
+      "bytes": 25776,
+      "sha256": "5791edae1fb7622027621d15bc7a160198bd3b75ed8f96d06cb7b85e2b873756"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/src/models/qwen35.cpp",
+      "bytes": 28742,
+      "sha256": "deec1f546ed1562a8fb8811fe21ce823ef3ed66bae2dde7a3c791068e89a9efd"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/src/models/delta-net-base.cpp",
+      "bytes": 23706,
+      "sha256": "5e3b318953f854b3825dd7cfe4b483ca7935cd5f6680438a3ece4be9b4d5428f"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ops.cpp",
+      "bytes": 425708,
+      "sha256": "215d16e7ebbb4066920e2d7a5f7ea6c46636d298aedd553e00f0b3ebb0d59305"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/vec.h",
+      "bytes": 68484,
+      "sha256": "885a50e405f20f75ea70cb7fb062934812beffc9229ddb56a003aa150163d2fa"
+    },
+    {
+      "path": "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libllama.so.0.6.0",
+      "bytes": 4927568,
+      "sha256": "81130cf7c14a3ed10db157a6764baf340e1bd7c2696d7b12de91baf04d334ee5"
+    },
+    {
+      "path": "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml.so.0.26.0",
+      "bytes": 58424,
+      "sha256": "1df83a5f445c119d10cd4fcac260c150a32d1b2ddf01dd49c9686c8b958f5c70"
+    },
+    {
+      "path": "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml-base.so.0.26.0",
+      "bytes": 972584,
+      "sha256": "dc4856a2ad667979352c9f270274b84f8135575c7394bd3b00e415e9e371e903"
+    },
+    {
+      "path": "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so",
+      "bytes": 1459208,
+      "sha256": "ca75af8b6281c57114f411d8883120a4ee595d8cb05c998a0b5e40921417c935"
+    }
+  ]
+}
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/run_ref_control.py .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/run_ref_control.py
new file mode 100644
index 0000000..5bbfa3d
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/run_ref_control.py
@@ -0,0 +1,157 @@
+#!/usr/bin/env python3
+"""ROOT ONLY: one fresh CPU8 public-use_ref control, no automatic case ladder."""
+import argparse
+import os
+from pathlib import Path
+import re
+import selectors
+import subprocess
+import time
+from common import HERE, REPO, TEACHER_SHA, authenticate, digest, require, write
+from supervisor import Session, identity, subreaper
+
+
+def main():
+    p = argparse.ArgumentParser(description=__doc__)
+    p.add_argument('--model', choices=['0.8b', '2b'], required=True)
+    p.add_argument('--prefix', type=int, choices=[64, 512], required=True)
+    p.add_argument('--cpus', required=True, help='explicit eight available CPU IDs, comma separated')
+    p.add_argument('--seconds', type=int, default=7200)
+    p.add_argument('--binary', type=Path, required=True)
+    p.add_argument('--binary-sha256', required=True, help='ROOT authenticated fresh build SHA')
+    p.add_argument('--new-output', type=Path, required=True)
+    a = p.parse_args()
+    require(re.fullmatch(r'[0-9]+(?:,[0-9]+){7}', a.cpus) is not None, 'exact CPU8 declaration')
+    cpus = list(map(int, a.cpus.split(',')))
+    require(cpus == sorted(set(cpus)) and cpus[-1] < 1024
+            and set(cpus) <= os.sched_getaffinity(0), 'eight distinct available ordered CPU IDs')
+    require(1 <= a.seconds <= 7200, 'finite original native wall deadline')
+    require(re.fullmatch('[a-f0-9]{64}', a.binary_sha256) is not None, 'ROOT build digest syntax')
+    cases = authenticate()
+    case = cases['models'][a.model]
+    model = (REPO / case['model']).resolve(strict=True)
+    config = (REPO / case['config']).resolve(strict=True)
+    teachers = (REPO / cases['teachers']['path']).resolve(strict=True)
+    binary = a.binary.resolve(strict=True)
+    require(binary.is_relative_to(HERE) and os.access(binary, os.X_OK), 'own ROOT-built executable only')
+    require(digest(binary)['sha256'] == a.binary_sha256, 'fresh executable changed')
+    inputs = {str(model): digest(model), str(config): digest(config), str(teachers): digest(teachers),
+              str(binary): digest(binary)}
+    require(inputs[str(model)] == case['model_identity'] and inputs[str(config)] == case['config_identity']
+            and inputs[str(teachers)] == {'bytes': 32768, 'sha256': TEACHER_SHA}, 'original source/teacher binding')
+    output = a.new_output.resolve()
+    require(output.is_relative_to(HERE) and output != HERE and not output.exists(), 'fresh own output directory')
+    sr = subreaper()  # Fail before ANY child if the reviewed lifecycle primitive is unavailable.
+    output.mkdir(mode=0o700)
+    capture = output / 'capture'
+    argv = [str(binary), str(model), str(teachers), '8192', str(a.prefix), str(a.prefix), '8',
+            str(a.seconds), str(capture)]
+    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'CUDA_VISIBLE_DEVICES': '-1',
+           'NVIDIA_VISIBLE_DEVICES': 'void', 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8',
+           'ZG_USE_REF_CPU_SET': a.cpus}
+    receipt = dict(status='UNQUALIFIED', argv=argv, model=a.model, prefix=a.prefix, cut=8192,
+                   CPU_affinity=cpus, inputs=inputs, parent=identity(os.getpid()), subreaper=sr,
+                   original_deadline_seconds=a.seconds, outer_wall_seconds=a.seconds + 60,
+                   address_space_bytes=12 * 1024**3, max_log_bytes=8 * 1024**2,
+                   benchmark=False, process_scope='owned SID only; no escaped-session containment claim',
+                   freeze_manifest=digest(HERE / 'freeze-manifest.json'))
+    child = owner = selector = pipe = None
+    primary = None
+    errors = []
+    retained = 0
+    eof = False
+    started = time.monotonic()
+    with (output / 'native.log').open('xb') as log:
+        try:
+            def limits():
+                import resource
+                os.sched_setaffinity(0, cpus)  # Before exec/DSO initialization or any native model work.
+                resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
+                resource.setrlimit(resource.RLIMIT_CPU, (a.seconds * 8 + 64, a.seconds * 8 + 64))
+                resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))
+                resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
+            child = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                                     stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=limits)
+            pipe = child.stdout
+            owner = Session(child)
+            receipt['owned_native'] = owner.leader
+            os.set_blocking(pipe.fileno(), False)
+            selector = selectors.DefaultSelector()
+            selector.register(pipe, selectors.EVENT_READ)
+
+            def drain(seconds):
+                nonlocal retained, eof
+                for key, _ in selector.select(seconds):
+                    try:
+                        block = os.read(key.fd, 65536)
+                    except BlockingIOError:
+                        continue
+                    if not block:
+                        selector.unregister(key.fileobj)
+                        eof = True
+                        continue
+                    retained += len(block)
+                    require(retained <= 8 * 1024**2, 'bounded native log')
+                    log.write(block)
+
+            next_scan = started
+            while True:
+                require(time.monotonic() - started < a.seconds + 60, 'outer native deadline')
+                drain(.1)
+                if time.monotonic() >= next_scan:
+                    owner.scan()
+                    next_scan = time.monotonic() + 1
+                code = owner.unreaped()
+                if code is not None:
+                    receipt['native_exit_code'] = code
+                    break
+            require(receipt['native_exit_code'] == 0, 'original native control refused; preserve all evidence')
+            # Recheck source while all postspawn paths remain inside cleanup finally.
+            for name, expected in inputs.items():
+                require(digest(name) == expected, 'input changed during genuine native run: ' + name)
+            authenticate()
+            receipt['status'] = 'PUBLIC_USE_REF_NATIVE_CAPTURE_CONTROL_VALIDATOR_REQUIRED'
+        except BaseException as error:
+            primary = repr(error)
+        finally:
+            cleanup = {}
+            if owner is not None:
+                try:
+                    owner.cleanup(cleanup, errors)
+                except BaseException as error:
+                    errors.append('unexpected owned cleanup: ' + repr(error))
+            elif child is not None:
+                errors.append('session identity construction failed; no unbound process signal')
+                try:
+                    child.wait(timeout=5)
+                except BaseException as error:
+                    errors.append('unbound native wait: ' + repr(error))
+            if selector is not None:
+                try:
+                    deadline = time.monotonic() + 2
+                    while not eof and time.monotonic() < deadline:
+                        drain(.05)
+                    require(eof, 'all owned stdout writers retired/EOF')
+                    cleanup['pipe_eof'] = True
+                except BaseException as error:
+                    errors.append('final owned drain: ' + repr(error))
+            for label, resource in [('selector_closed', selector), ('pipe_closed', pipe)]:
+                if resource is not None:
+                    try:
+                        resource.close()
+                        cleanup[label] = True
+                    except BaseException as error:
+                        errors.append(label + ': ' + repr(error))
+            receipt.update(cleanup=cleanup, primary_error=primary, cleanup_errors=errors,
+                           log_bytes=retained, elapsed_seconds_not_benchmark=time.monotonic() - started,
+                           native_reaped_exit_code=child.returncode if child is not None else None)
+            if primary is not None or errors:
+                receipt['status'] = 'UNQUALIFIED'
+            write(output / 'invocation.json', receipt)
+    require(receipt['status'] == 'PUBLIC_USE_REF_NATIVE_CAPTURE_CONTROL_VALIDATOR_REQUIRED',
+            'retained unqualified native control; inspect invocation and native.log')
+    print(receipt['status'], str(output))
+
+
+if __name__ == '__main__':
+    main()
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/supervisor.py .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/supervisor.py
new file mode 100644
index 0000000..12b96ab
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/supervisor.py
@@ -0,0 +1,220 @@
+"""ROOT ONLY process-local Linux supervision; import launches/changes nothing."""
+import ctypes
+import errno
+import os
+from pathlib import Path
+import signal
+import time
+
+MAX_MEMBERS = 32
+
+
+def require(ok, message):
+    if not ok:
+        raise RuntimeError(message)
+
+
+def identity(pid):
+    raw = Path(f'/proc/{pid}/stat').read_text()
+    fields = raw[raw.rfind(')')+2:].split()
+    status = Path(f'/proc/{pid}/status').read_text()
+    uid = next(line.split()[1:] for line in status.splitlines() if line.startswith('Uid:'))
+    return dict(pid=pid,state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),sid=int(fields[3]),
+                starttime=int(fields[19]),uids=list(map(int,uid)))
+
+
+def same(left, right):
+    return all(left[key] == right[key] for key in ['pid','starttime','sid','uids'])
+
+
+def subreaper():
+    require(all(hasattr(os,name) for name in ['pidfd_open','waitid','WNOWAIT'])
+            and hasattr(signal,'pidfd_send_signal'),'public Linux pidfd/WNOWAIT availability')
+    probe=os.pidfd_open(os.getpid(),0)
+    try:
+        signal.pidfd_send_signal(probe,0,None,0)  # Read-only feature probe on this supervisor only.
+    finally:
+        os.close(probe)
+    # Linux public prctl(int option, unsigned long x4), no numerical-provider ABI.
+    libc = ctypes.CDLL(None,use_errno=True)
+    libc.prctl.argtypes = [ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
+    libc.prctl.restype = ctypes.c_int
+    old = ctypes.c_int(-1)
+    require(libc.prctl(37,ctypes.addressof(old),0,0,0)==0 and old.value in [0,1],'PR_GET_CHILD_SUBREAPER')
+    require(libc.prctl(36,1,0,0,0)==0,'PR_SET_CHILD_SUBREAPER process-local')
+    actual = ctypes.c_int(-1)
+    require(libc.prctl(37,ctypes.addressof(actual),0,0,0)==0 and actual.value==1,'actual subreaper readback')
+    return {'public_PR_GET':37,'public_PR_SET':36,'previous':old.value,'actual':actual.value,'process_local_only':True}
+
+
+class Session:
+    def __init__(self, child):
+        self.child = child
+        self.parent = identity(os.getpid())
+        self.leader = identity(child.pid)
+        require(self.leader['pid']==self.leader['pgid']==self.leader['sid']
+                and self.leader['ppid']==self.parent['pid'] and self.leader['uids']==self.parent['uids'],
+                'fresh exclusively owned debugger SID')
+        self.known = {child.pid:self.leader}
+        self.groups = {child.pid}
+        self.actions = []
+        self.reap_started = False
+        self.inferior = None
+
+    def unreaped(self):
+        require(not self.reap_started and self.child.returncode is None,'owned leader held until all signals finish')
+        current = identity(self.child.pid)
+        require(same(current,self.leader) and current['ppid']==self.parent['pid'],'live reserved debugger PID identity')
+        result = os.waitid(os.P_PID,self.child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
+        if result is None:
+            return None
+        require(result.si_pid==self.child.pid,'owned WNOWAIT identity')
+        return result.si_status if result.si_code==os.CLD_EXITED else -result.si_status
+
+    def scan(self):
+        self.unreaped()
+        members = []
+        for number,path in enumerate(Path('/proc').iterdir()):
+            require(number<262144,'bounded proc directory enumeration')
+            if not path.name.isdecimal():
+                continue
+            try:
+                row = identity(int(path.name))
+            except (FileNotFoundError,ProcessLookupError,PermissionError):
+                continue
+            if row['sid']!=self.leader['pid']:
+                continue
+            require(row['uids']==self.parent['uids'],'owned SID changed UID; refuse')
+            require(row['ppid'] in self.known or row['ppid']==self.parent['pid'] or row['pid']==self.leader['pid'],
+                    'only authenticated fresh SID descendants/adopted children')
+            if row['pid'] in self.known:
+                require(same(row,self.known[row['pid']]),'owned PID starttime changed; no retargeting')
+            else:
+                require(len(self.known)<MAX_MEMBERS,'finite exclusively owned process membership')
+                self.known[row['pid']]=row
+            self.groups.add(row['pgid'])
+            members.append(row)
+        return members
+
+    def bind_inferior(self, row):
+        require(row['ppid']==self.leader['pid'] and row['sid']==self.leader['pid']
+                and row['uids']==self.parent['uids'],'issued actual inferior belongs to fresh owned debugger')
+        if self.inferior is not None:
+            require(same(row,self.inferior),'same actually issued inferior')
+        self.inferior=row
+        # It may already have been normally reaped by GDB; never retarget absent/reused numeric PID.
+        try:
+            actual=identity(row['pid'])
+        except (FileNotFoundError,ProcessLookupError):
+            return
+        if not same(row,actual):
+            return
+        require(actual['ppid'] in [self.leader['pid'],self.parent['pid']],'known direct/adopted original inferior')
+        self.known[row['pid']]=row
+        self.groups.add(row['pgid'])
+
+    def send(self, row, sig):
+        require(not self.reap_started and sig in [signal.SIGTERM,signal.SIGKILL],'nonzero signals before ANY supervisor reap')
+        self.unreaped()
+        try:
+            before=identity(row['pid'])
+        except (FileNotFoundError,ProcessLookupError):
+            return
+        require(same(row,before) and before['uids']==self.parent['uids']
+                and before['ppid'] in [self.leader['pid'],self.parent['pid'],*self.known]
+                and before['pgid']==row['pgid'],'exact owned process/PGID binding before pidfd')
+        if before['state']=='Z':
+            return
+        fd=None
+        try:
+            fd=os.pidfd_open(row['pid'],0)
+            after=identity(row['pid'])
+            require(same(before,after) and after['ppid']==before['ppid'] and after['pgid']==before['pgid'],
+                    'same starttime/PPID/SID/UID/PGID after pidfd open')
+            signal.pidfd_send_signal(fd,sig,None,0)
+            self.actions.append({'pid':row['pid'],'starttime':row['starttime'],'sid':row['sid'],
+                                 'pgid':row['pgid'],'signal':int(sig),'transport':'public bound pidfd'})
+        except (FileNotFoundError,ProcessLookupError):
+            pass  # Closed original handle; never retry a reused numeric PID.
+        finally:
+            if fd is not None:
+                os.close(fd)
+
+    def cleanup(self, record, errors):
+        """All signal stages independent; reap only known identities afterwards."""
+        def stage(sig,label):
+            try:
+                members=self.scan()
+            except BaseException as error:
+                errors.append(label+' scan: '+repr(error));members=[]
+            # Retire inferior descendants before their debugger parent.
+            for row in sorted(members,key=lambda item:item['pid']==self.leader['pid']):
+                try:
+                    self.send(row,sig)
+                except BaseException as error:
+                    errors.append(label+' pid'+str(row['pid'])+': '+repr(error))
+        stage(signal.SIGTERM,'owned TERM')
+        try:
+            deadline=time.monotonic()+.5
+            while time.monotonic()<deadline:
+                self.scan()
+                time.sleep(.05)
+        except BaseException as error:
+            errors.append('owned grace: '+repr(error))
+        stage(signal.SIGKILL,'owned KILL1')
+        stage(signal.SIGKILL,'owned KILL2')
+        # No nonzero signal from here, even if a wait/absence check fails.
+        self.reap_started=True
+        try:
+            self.child.wait(timeout=5)
+            record['leader_reaped']=True
+        except BaseException as error:
+            errors.append('owned first leader reap: '+repr(error))
+        finally:
+            if self.child.returncode is None:
+                try:
+                    self.child.wait(timeout=5)
+                    record['leader_reaped']=True
+                except BaseException as error:
+                    errors.append('owned final leader reap: '+repr(error))
+        adopted=[]
+        for pid,old in self.known.items():
+            if pid==self.leader['pid']:
+                continue
+            try:
+                actual=identity(pid)
+            except (FileNotFoundError,ProcessLookupError):
+                continue  # Normally reaped original; no wait/signal on numeric reuse.
+            if not same(old,actual):
+                continue
+            try:
+                require(actual['ppid']==self.parent['pid'],'known inferior was not adopted by own subreaper')
+                deadline=time.monotonic()+5
+                while True:
+                    got,status=os.waitpid(pid,os.WNOHANG)
+                    if got==pid:
+                        adopted.append({'pid':pid,'starttime':old['starttime'],'wait_status':status})
+                        break
+                    require(time.monotonic()<deadline,'known adopted inferior reap deadline')
+                    time.sleep(.05)
+            except BaseException as error:
+                errors.append('owned adopted reap pid'+str(pid)+': '+repr(error))
+        record.update(actions=self.actions,groups=sorted(self.groups),known=list(self.known.values()),adopted_reaped=adopted,
+                      all_nonzero_signals_before_reap=True)
+        try:
+            deadline=time.monotonic()+5
+            while True:
+                present=[]
+                for group in self.groups:
+                    try:
+                        os.killpg(group,0)  # Read-only absence check only, never a post-reap signal.
+                        present.append(group)
+                    except ProcessLookupError:
+                        pass
+                if not present:
+                    record['all_owned_groups_absent']=True
+                    break
+                require(time.monotonic()<deadline,'owned PGID absence deadline')
+                time.sleep(.05)
+        except BaseException as error:
+            errors.append('owned groups absence: '+repr(error))
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/test_control.py .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/test_control.py
new file mode 100644
index 0000000..99f9114
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/test_control.py
@@ -0,0 +1,85 @@
+"""ROOT ONLY prepared source-free parser guards; no native/model work."""
+import copy
+import json
+import unittest
+from validate_control import cpulist, validate_trace
+
+
+def fixture():
+    rows = []
+    for backend in ['0x100', '0x200']:
+        for origin in ['public_dev_init', 'existing_public_abort_setter_callback']:
+            rows.append(dict(kind='public_cpu_reference_setter', backend=backend,
+                             origin=origin, requested_use_ref=True))
+    for sequence in range(1, 19):
+        rows.append(dict(kind='actual_public_graph_plan', sequence=sequence, use_ref=True, n_threads=8))
+    for backend in ['0x200', '0x100']:
+        rows.append(dict(kind='public_backend_free', backend=backend))
+    report = dict(actual_public_plan_calls=18, actual_successful_public_plan_calls=18,
+                  actual_plan_threads_min=8, actual_plan_threads_max=8)
+    return rows, report
+
+
+def encode(rows):
+    return b''.join(json.dumps(row).encode() + b'\n' for row in rows)
+
+
+class ControlGuards(unittest.TestCase):
+    def test_complete_and_ordered(self):
+        rows, report = fixture()
+        self.assertEqual(validate_trace(encode(rows), report, 512)['actual_plan_calls'], 18)
+
+    def test_false_actual_plan_refused(self):
+        rows, report = fixture()
+        rows[4]['use_ref'] = False
+        with self.assertRaises(ValueError):
+            validate_trace(encode(rows), report, 512)
+
+    def test_missing_route_cannot_reuse_counter(self):
+        rows, report = fixture()
+        del rows[5]
+        with self.assertRaises(ValueError):
+            validate_trace(encode(rows), report, 512)
+
+    def test_retired_backend_and_duplicate_identity(self):
+        for position in [0, 4]:
+            rows, report = fixture()
+            if position == 0:
+                rows[2]['backend'] = '0x100'
+            else:
+                rows.insert(4, dict(kind='public_backend_free', backend='0x100'))
+            with self.assertRaises(ValueError):
+                validate_trace(encode(rows), report, 512)
+
+    def test_missing_context_callback(self):
+        rows, report = fixture()
+        del rows[3]
+        with self.assertRaises(ValueError):
+            validate_trace(encode(rows), report, 512)
+
+    def test_failed_plan_not_counted_success(self):
+        rows, report = fixture()
+        report['actual_successful_public_plan_calls'] -= 1
+        with self.assertRaises(ValueError):
+            validate_trace(encode(rows), report, 512)
+
+    def test_truncation_duplicate_key_and_thread_bounds(self):
+        rows, report = fixture()
+        for raw in [encode(rows)[:-1], encode(rows).replace(b'"use_ref": true', b'"use_ref": true, "use_ref": true', 1)]:
+            with self.assertRaises(ValueError):
+                validate_trace(raw, report, 512)
+        for value in [True, 0, 9]:
+            invalid = copy.deepcopy(rows)
+            invalid[4]['n_threads'] = value
+            with self.assertRaises(ValueError):
+                validate_trace(encode(invalid), report, 512)
+
+    def test_cpu_range_and_exact_cardinality(self):
+        self.assertEqual(cpulist('0-3,8-11'), [0, 1, 2, 3, 8, 9, 10, 11])
+        for value in ['0-7,7', '0-6', '0-8', '0-7,', '0-1024', '7-0']:
+            with self.assertRaises(ValueError):
+                cpulist(value)
+
+
+if __name__ == '__main__':
+    unittest.main()
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/test_supervisor.py .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/test_supervisor.py
new file mode 100644
index 0000000..14370ef
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/test_supervisor.py
@@ -0,0 +1,86 @@
+"""ROOT ONLY real owned-SID cleanup fixtures; import spawns nothing, no model/GDB."""
+import json
+import os
+import select
+import signal
+import subprocess
+import sys
+import unittest
+from unittest import mock
+from supervisor import Session,subreaper,identity
+
+CHILD="import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(60)"
+LEADER="""import json,signal,subprocess,sys,time
+signal.signal(signal.SIGTERM,signal.SIG_IGN)
+child=subprocess.Popen([sys.executable,'-c',sys.argv[1]],stdout=subprocess.PIPE,process_group=0)
+assert child.stdout.readline()==b'ready\\n'
+print(json.dumps({'inferior':child.pid}),flush=True)
+time.sleep(60)
+"""
+
+
+class Supervision(unittest.TestCase):
+    @classmethod
+    def setUpClass(cls):
+        subreaper()
+
+    def case(self,injection):
+        child=subprocess.Popen([sys.executable,'-c',LEADER,CHILD],stdin=subprocess.DEVNULL,
+            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,start_new_session=True)
+        owner=None
+        try:
+            owner=Session(child)
+            self.assertTrue(select.select([child.stdout],[],[],5)[0],'bounded fixture startup')
+            row=json.loads(child.stdout.readline(4096))
+            owner.bind_inferior(identity(row['inferior']))
+            owner.scan()
+            errors=[];record={};failed=False
+            send,scan,wait=owner.send,owner.scan,child.wait
+            scan_count=0
+            def sending(member,sig):
+                nonlocal failed
+                self.assertFalse(owner.reap_started)
+                if injection in ['term','kill'] and not failed and sig==(signal.SIGTERM if injection=='term' else signal.SIGKILL):
+                    failed=True
+                    raise RuntimeError('injected single owned signal-stage failure')
+                return send(member,sig)
+            def scanning():
+                nonlocal failed,scan_count
+                scan_count+=1
+                if injection=='grace' and scan_count==2:
+                    failed=True
+                    raise RuntimeError('injected owned grace scan failure')
+                return scan()
+            def waiting(timeout):
+                nonlocal failed
+                self.assertTrue(owner.reap_started)
+                if injection=='reap' and not failed:
+                    failed=True
+                    raise RuntimeError('injected first owned reap failure')
+                return wait(timeout=timeout)
+            with mock.patch.object(owner,'send',side_effect=sending),mock.patch.object(owner,'scan',side_effect=scanning),mock.patch.object(child,'wait',side_effect=waiting):
+                owner.cleanup(record,errors)
+            self.assertTrue(record.get('leader_reaped'))
+            self.assertTrue(record.get('all_owned_groups_absent'))
+            self.assertTrue(record.get('all_nonzero_signals_before_reap'))
+            self.assertTrue(record['adopted_reaped'])
+            self.assertTrue(select.select([child.stdout],[],[],5)[0],'bounded fixture pipe retirement')
+            self.assertEqual(child.stdout.read(1),b'')
+            if injection:
+                self.assertTrue(failed)
+                self.assertTrue(errors,'first error retained after later successful cleanup')
+            else:
+                self.assertEqual(errors,[])
+        finally:
+            if child.returncode is None and owner is not None and not owner.reap_started:
+                owner.cleanup({},[])
+            child.stdout.close()
+
+    def test_separate_inferior_group_real_pidfd_adoption_reap(self):self.case(None)
+    def test_term_failure_cannot_skip_kill_and_all_reaps(self):self.case('term')
+    def test_grace_failure_cannot_skip_kill_and_all_reaps(self):self.case('grace')
+    def test_first_kill_failure_has_independent_owned_retry(self):self.case('kill')
+    def test_first_reap_failure_has_independent_final_reap(self):self.case('reap')
+
+
+if __name__=='__main__':unittest.main()
diff --git .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/validate_control.py .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/validate_control.py
new file mode 100644
index 0000000..949f4a3
--- /dev/null
+++ .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/validate_control.py
@@ -0,0 +1,179 @@
+#!/usr/bin/env python3
+"""ROOT ONLY public-policy evidence validation; arithmetic acceptance is separate."""
+import argparse
+import importlib.util
+import json
+from pathlib import Path
+import re
+from common import HERE, REPO, TEACHER_SHA, authenticate, bounded, digest, jread, require, unique, write
+
+
+def validate_trace(raw, report, prefix):
+    require(type(prefix) is int and prefix in [64, 512], 'closed actual prefix geometry')
+    require(0 < len(raw) <= 8 * 1024**2 and raw.endswith(b'\n'), 'complete bounded public trace')
+    lines = raw.splitlines()
+    require(len(lines) <= 65536, 'bounded actual public call count')
+    backends = {}
+    plans = 0
+    threads = []
+    for line in lines:
+        require(0 < len(line) < 2048, 'bounded actual public trace line')
+        row = json.loads(line, object_pairs_hook=unique)
+        kind = row.get('kind')
+        if kind == 'public_cpu_reference_setter':
+            require(set(row) == {'kind', 'backend', 'origin', 'requested_use_ref'}
+                    and row['requested_use_ref'] is True, 'exact original setter trace')
+            name = row['backend']
+            require(isinstance(name, str) and re.fullmatch(r'0x[0-9a-f]{1,16}', name)
+                    and int(name, 16) > 0, 'nonnull opaque identity only')
+            origin = row['origin']
+            if origin == 'public_dev_init':
+                require(name not in backends and len(backends) < 2, 'unique two actual CPU context initializations')
+                backends[name] = {'abort': False, 'freed': False}
+            else:
+                require(origin == 'existing_public_abort_setter_callback' and name in backends
+                        and backends[name] == {'abort': False, 'freed': False}, 'actual callback exactly once')
+                backends[name]['abort'] = True
+        elif kind == 'actual_public_graph_plan':
+            plans += 1
+            require(set(row) == {'kind', 'sequence', 'use_ref', 'n_threads'}
+                    and type(row['sequence']) is int and row['sequence'] == plans
+                    and row['use_ref'] is True, 'every actual delegated plan has exact reference policy')
+            require(backends and all(b['abort'] and not b['freed'] for b in backends.values()),
+                    'every actually seen context configured before computation, no retired context computation')
+            require(type(row['n_threads']) is int and 1 <= row['n_threads'] <= 8, 'actual thread bound')
+            threads.append(row['n_threads'])
+        elif kind == 'public_backend_free':
+            require(set(row) == {'kind', 'backend'} and row['backend'] in backends, 'only known CPU backend retirement')
+            backend = backends[row['backend']]
+            require(backend == {'abort': True, 'freed': False}, 'exact once original backend retirement')
+            backend['freed'] = True
+        else:
+            raise ValueError('unexpected public trace record')
+    minimum = (8191 + prefix - 1) // prefix + 2
+    require(all(type(report[field]) is int for field in ['actual_public_plan_calls',
+                'actual_successful_public_plan_calls', 'actual_plan_threads_min', 'actual_plan_threads_max']),
+            'actual integer counters and thread bounds')
+    require(len(backends) == 2 and all(b == {'abort': True, 'freed': True} for b in backends.values()),
+            'both actual contexts fully configured and retired')
+    require(plans >= minimum and len(lines) == plans + 6
+            and report['actual_public_plan_calls'] == plans
+            and report['actual_successful_public_plan_calls'] == plans
+            and report['actual_plan_threads_min'] == min(threads)
+            and report['actual_plan_threads_max'] == max(threads), 'no missing compute route or fabricated counts')
+    return {'actual_cpu_contexts': 2, 'actual_setter_calls': 4,
+            'actual_original_abort_setter_calls': 2, 'actual_plan_calls': plans,
+            'actual_public_plan_use_ref_all_true': True, 'actual_threads_min': min(threads),
+            'actual_threads_max': max(threads)}
+
+
+def cpulist(text):
+    result = []
+    require(len(text) <= 128 and re.fullmatch(r'[0-9,-]+', text), 'bounded process CPU mask')
+    for item in text.split(','):
+        ends = list(map(int, item.split('-')))
+        require(len(ends) in [1, 2] and 0 <= ends[0] <= ends[-1] < 1024, 'process CPU range')
+        result.extend(range(ends[0], ends[-1] + 1))
+    require(len(result) == 8 and result == sorted(set(result)), 'exact CPU8 process mask')
+    return result
+
+
+def validate(output):
+    invocation = jread(output / 'invocation.json')
+    require(invocation['status'] == 'PUBLIC_USE_REF_NATIVE_CAPTURE_CONTROL_VALIDATOR_REQUIRED'
+            and invocation['primary_error'] is None and invocation['cleanup_errors'] == []
+            and invocation['native_exit_code'] == invocation['native_reaped_exit_code'] == 0,
+            'actual owned control completed with clean reap')
+    require(invocation['freeze_manifest'] == digest(HERE / 'freeze-manifest.json'), 'executed reviewed source closure')
+    for key in ['leader_reaped', 'all_owned_groups_absent', 'all_nonzero_signals_before_reap',
+                'pipe_eof', 'selector_closed', 'pipe_closed']:
+        require(invocation['cleanup'][key] is True, 'complete owned lifecycle: ' + key)
+    require(invocation['benchmark'] is False and invocation['cut'] == 8192
+            and invocation['prefix'] in [64, 512], 'closed fresh bulk control scope')
+    cpus = invocation['CPU_affinity']
+    require(type(cpus) is list and len(cpus) == 8 and all(type(c) is int for c in cpus)
+            and cpus == sorted(set(cpus)) and 0 <= cpus[0] <= cpus[-1] < 1024, 'declared CPU8 list')
+    cases = authenticate()
+    case = cases['models'][invocation['model']]
+    model = str((REPO / case['model']).resolve(strict=True))
+    config = str((REPO / case['config']).resolve(strict=True))
+    teachers = str((REPO / cases['teachers']['path']).resolve(strict=True))
+    require(invocation['inputs'][model] == case['model_identity']
+            and invocation['inputs'][config] == case['config_identity']
+            and invocation['inputs'][teachers] == {'bytes': 32768, 'sha256': TEACHER_SHA}, 'exact original inputs')
+    for path, expected in invocation['inputs'].items():
+        require(digest(path) == expected, 'executed source/binary changed: ' + path)
+    capture = output / 'capture'
+    require(invocation['argv'] == [invocation['argv'][0], model, teachers, '8192',
+            str(invocation['prefix']), str(invocation['prefix']), '8', str(invocation['original_deadline_seconds']),
+            str(capture)], 'unaltered original native CLI and explicit policy wrapper')
+    require(Path(invocation['argv'][0]).is_relative_to(HERE)
+            and invocation['argv'][0] in invocation['inputs'], 'owned authenticated executable')
+    require(bounded(capture / 'input_ids.u32le', 32768) == bounded(teachers, 32768), 'all8192 authentic teacher words')
+    complete = jread(capture / 'complete.json')
+    loader_path = REPO / '.agents/runs/numerics-and-context-long-context-audit-20261020/compare_raw.py'
+    spec = importlib.util.spec_from_file_location('unchanged_native_capture_metadata', loader_path)
+    require(spec is not None and spec.loader is not None, 'pinned original capture validator')
+    loader = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(loader)  # Inert definitions only; no main/provider/operator execution.
+    expected = loader.expected_geometry(case['hidden'], 3584 if case['hidden'] == 1024 else 6144)
+    validated_complete, records = loader.load_capture(capture, 'pinned_native_llama_cpp', expected, 8192)
+    require(complete == validated_complete and len(records) == 730, 'unchanged730-stage full metadata/file coverage')
+    require(complete['status'] == 'CAPTURE_COMPLETE_NOT_COMPARISON'
+            and complete['provider'] == 'pinned_native_llama_cpp' and complete['cut'] == 8192
+            and complete['model_path'] == model and complete['hidden_width'] == case['hidden']
+            and complete['record_count'] == 730 and complete['actual_selected_token_GDN_ops'] == 18
+            and complete['actual_shared_model_tokens'] == 8193
+            and 8320 <= complete['n_ctx_actual'] <= 8704
+            and complete['instrumentation_full_logits_exact'] is True
+            and complete['instrumentation_whole_native_state_exact'] is True
+            and complete['F32_KV'] is True and complete['CPU_only'] is True
+            and complete['flash_attention'] is False, 'unchanged original full capture guards')
+    for field in ['prefix_batch_requested', 'ubatch_requested', 'n_batch_actual', 'n_ubatch_actual']:
+        require(complete[field] == invocation['prefix'], 'actual bulk geometry: ' + field)
+    require(complete['threads_actual'] == complete['threads_batch_actual'] == 8, 'original actual CPU8 contexts')
+    control = jread(capture / 'use-ref-control.json')
+    require(control['status'] == 'PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED'
+            and control['CPU_backend_count'] == 2 and control['public_setter_calls'] == 4
+            and control['existing_abort_setter_calls'] == 2
+            and control['actual_public_plan_use_ref_all_true'] is True
+            and control['backend_private_getter_available'] is False
+            and control['plan_to_backend_identity_publicly_available'] is False
+            and control['CPU_REPACK_policy_unchanged'] is True
+            and control['native_raw_token_loop_unchanged'] is True
+            and control['historical_prefix_or_cause_closed'] is False
+            and control['actual_CPU_affinity'] == cpus, 'honest public API closure')
+    observed = validate_trace(bounded(capture / 'use-ref-public-trace.jsonl', 8 * 1024**2), control, invocation['prefix'])
+    status = bounded(capture / 'use-ref-process-status.txt', 65536).decode()
+    masks = [line.split(':', 1)[1].strip() for line in status.splitlines() if line.startswith('Cpus_allowed_list:')]
+    require(len(masks) == 1 and cpulist(masks[0]) == cpus, 'actual process affinity readback')
+    maps = bounded(capture / 'use-ref-loaded-maps.txt', 2 * 1024**2).decode()
+    require(len(maps.splitlines()) <= 16384, 'bounded actual mapping records')
+    paths = {line.split(maxsplit=5)[5] for line in maps.splitlines() if len(line.split(maxsplit=5)) == 6}
+    providers = [row for row in jread(HERE / 'original-source-pins.json')['files'] if row['path'].startswith('/nix/store/')]
+    require(all(row['path'] in paths for row in providers), 'all original pinned providers actually mapped')
+    require(not any('libggml-cpu-' in path and not path.endswith('/libggml-cpu-haswell.so') for path in paths),
+            'no unobserved alternate CPU implementation')
+    return dict(status='PASS_PUBLIC_USE_REF_CONTROL_CAPTURE_ONLY_COMPARISONS_REQUIRED',
+                source_model_sha256=case['model_identity']['sha256'], raw_teacher_sha256=TEACHER_SHA,
+                cut=8192, prefix_batch=invocation['prefix'], CPU_affinity=cpus, public_route=observed,
+                invocation=digest(output / 'invocation.json'), complete=digest(capture / 'complete.json'),
+                control=digest(capture / 'use-ref-control.json'),
+                scope='actual setter+public-plan policy and original selected-tail observation transparency only; '
+                      'no raw730/cross-provider all42 result or historical/whole-cause conclusion')
+
+
+def main():
+    p = argparse.ArgumentParser(description=__doc__)
+    p.add_argument('--capture-run', type=Path, required=True)
+    p.add_argument('--new-report', type=Path, required=True)
+    a = p.parse_args()
+    output = a.capture_run.resolve(strict=True)
+    require(output.is_relative_to(HERE) and output != HERE, 'own fresh control run only')
+    report = validate(output)
+    write(a.new_report, report)
+    print(report['status'])
+
+
+if __name__ == '__main__':
+    main()

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/NULL_FREE_FORWARDING_V6.patch

Original UTF-8 bytes: 1940; SHA256 `454ac749d4bf5a91b3ef782f169d6aef9f7ce05175a0a396fe55ed2753110cf9`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- native_ref_control_v4.cpp
+++ native_ref_control_v6.cpp
@@ -402,7 +402,7 @@
     static auto original=ref_control::symbol<decltype(&ggml_backend_free)>(ref_control::BASE,
         ref_control::BASE_BYTES,"ggml_backend_free");
     ref_control::Kind kind=ref_control::Kind::Cpu;
-    if(ref_control::armed) {
+    if(ref_control::armed && backend) {
         std::lock_guard<std::mutex> guard(ref_control::mutex);auto &record=ref_control::known(backend);
         ref_control::checked(ref_control::scope==nullptr && ref_control::sync_scope==nullptr
             && (record.kind==ref_control::Kind::Cpu?(record.abort==1 && record.set==2):(record.set==0 && record.abort==0)),
@@ -411,7 +411,7 @@
         record.freed=true;kind=record.kind;
     }
     errno=saved;original(backend);const int after=errno;
-    if(ref_control::armed) {
+    if(ref_control::armed && backend) {
         std::lock_guard<std::mutex> guard(ref_control::mutex);
         ref_control::append("{\"kind\":\"public_backend_free\",\"backend\":\""+ref_control::pointer(backend)
             +"\",\"backend_kind\":\""+ref_control::kind_name(kind)+"\"}");
@@ -488,7 +488,7 @@
         write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
         std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
         report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
-            <<"\"control_source_revision\":\"public_use_ref_exact_blas_lifecycle_cpu_compute_v4\","
+            <<"\"control_source_revision\":\"public_use_ref_exact_blas_cpu_compute_legal_null_free_v6\","
             <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\","
             <<"\"CPU_backend_count\":2,\"BLAS_backend_count\":2,\"public_setter_calls\":4,"
             <<"\"existing_abort_setter_calls\":2,\"BLAS_compute_delegations\":0,"

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/POLICY_V4.patch

Original UTF-8 bytes: 92046; SHA256 `d54613e2ceae169b19d4be4683fadb2ef3a743838c108d17ba0c6bdb517a1bc8`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v4.cpp
@@ -0,0 +1,516 @@
+// Qualifier-only public CPU use_ref control; exact pinned BLAS may construct/retire, never compute.
+#ifndef _GNU_SOURCE
+#define _GNU_SOURCE
+#endif
+#include "ggml-cpu.h"
+#include "ggml-blas.h"
+#include <dlfcn.h>
+#include <algorithm>
+#include <fcntl.h>
+#include <limits.h>
+#include <sched.h>
+#include <sys/stat.h>
+#include <unistd.h>
+#include <cerrno>
+#include <cstdint>
+#include <cstdio>
+#include <cstdlib>
+#include <cstring>
+#include <fstream>
+#include <map>
+#include <mutex>
+#include <string>
+#include <vector>
+
+namespace ref_control {
+constexpr const char *BASE="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml-base.so.0.26.0";
+constexpr const char *CPU="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so";
+constexpr const char *BLAS="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-blas.so";
+constexpr size_t BASE_BYTES=972584, CPU_BYTES=1459208, BLAS_BYTES=64136, MAX_RECORDS=65536;
+static bool armed=false;
+enum class Kind { Cpu, Blas };
+struct Backend {
+    Kind kind=Kind::Cpu;size_t id=0,abort=0,set=0,async=0,plans=0,inner=0,sync=0;
+    ggml_backend_dev_t device=nullptr;ggml_backend_reg_t registry=nullptr;bool freed=false;
+};
+struct Plan {ggml_backend_t backend=nullptr;size_t id=0;};
+static std::map<ggml_backend_t,Backend> backends;
+static std::map<ggml_backend_graph_plan_t,Plan> plans;
+static size_t next_plan=0,public_calls=0,public_success=0,async_calls=0,plan_calls=0,sync_calls=0;
+static size_t plan_creates=0,plan_frees=0,inflight=0;
+struct Scope {
+    ggml_backend_t backend=nullptr;ggml_cgraph *graph=nullptr;ggml_backend_graph_plan_t plan=nullptr;
+    size_t sequence=0,inner=0;const char *api=nullptr;bool in_inner=false;
+};
+static thread_local Scope *scope=nullptr;
+struct SyncScope {ggml_backend_t backend=nullptr;ggml_cgraph *graph=nullptr;size_t async=0,sequence=0;};
+static thread_local SyncScope *sync_scope=nullptr;
+static std::mutex mutex;
+static size_t trace_records=0,trace_bytes=0;
+static int trace_fd=-1;
+static std::string trace_path;
+static size_t graph_calls=0,graph_success=0;
+static int thread_min=9,thread_max=0;
+static cpu_set_t expected_affinity;
+static std::vector<int> declared_cpus;
+using Init=ggml_backend_t(*)(ggml_backend_dev_t,const char *);
+using Lookup=void *(*)(ggml_backend_reg_t,const char *);
+using Free=void(*)(ggml_backend_t);
+using IsCpu=bool(*)(ggml_backend_t);
+using Ref=void(*)(ggml_backend_t,bool);
+using Compute=ggml_status(*)(ggml_cgraph *,ggml_cplan *);
+
+[[noreturn]] static void fatal(const char *message) {
+    const int saved=errno;
+    const char prefix[]="USE_REF_CONTROL_UNQUALIFIED ";
+    const ssize_t a=write(STDERR_FILENO,prefix,sizeof(prefix)-1);
+    const ssize_t b=write(STDERR_FILENO,message,strnlen(message,4096));
+    const ssize_t c=write(STDERR_FILENO,"\n",1);
+    (void)a;(void)b;(void)c;errno=saved;_exit(125);
+}
+static void checked(bool value,const char *message) {if(!value)fatal(message);}
+[[noreturn]] static void provider_failure(const char *stage,const char *name,
+        const char *expected,const char *actual,const char *canonical_expected,
+        const char *canonical_actual,const char *error,const char *actual_symbol=nullptr) {
+    const int saved=errno;char message[4096]={};
+    const int size=std::snprintf(message,sizeof(message),
+        "provider_attestation_v2 stage=%.128s symbol=%.192s expected=%.512s actual=%.512s "
+        "canonical_expected=%.512s canonical_actual=%.512s actual_symbol=%.192s dlerror=%.512s errno=%d",
+        stage?stage:"(null)",name?name:"(null)",expected?expected:"(null)",actual?actual:"(null)",
+        canonical_expected?canonical_expected:"(unresolved)",canonical_actual?canonical_actual:"(unresolved)",
+        actual_symbol?actual_symbol:"(unavailable)",error?error:"(none)",saved);
+    if(size<0)fatal("provider_attestation_v2 diagnostic formatting failed");
+    errno=saved;fatal(message);
+}
+static void check_affinity() {
+    cpu_set_t actual;CPU_ZERO(&actual);
+    checked(sched_getaffinity(0,sizeof(actual),&actual)==0
+        && CPU_EQUAL(&actual,&expected_affinity),"actual calling thread CPU8 affinity");
+}
+static void append(const std::string &line) {
+    checked(trace_records<MAX_RECORDS && line.size()<2048
+        && trace_bytes+line.size()+1<=8*1024*1024,"bounded public trace");
+    if(trace_fd<0) {
+        trace_fd=open(trace_path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
+        checked(trace_fd>=0,"exclusive public trace open");
+    }
+    const auto record=line+"\n";
+    size_t offset=0;int interruptions=0;
+    while(offset<record.size()) {
+        const ssize_t written=write(trace_fd,record.data()+offset,record.size()-offset);
+        if(written<0 && errno==EINTR && ++interruptions<=8)continue;
+        checked(written>0,"public trace write");offset+=static_cast<size_t>(written);
+    }
+    ++trace_records;trace_bytes+=record.size();
+}
+static std::string pointer(const void *value) {
+    if(!value)return "0x0";
+    char text[2+sizeof(void*)*2+1]={};
+    const int size=std::snprintf(text,sizeof(text),"%p",value);
+    checked(size>0 && size<static_cast<int>(sizeof(text)),"opaque identity formatting");
+    return std::string(text);
+}
+static void attest(void *address,const char *path,size_t length,const char *name) {
+    Dl_info info{};struct stat actual{},expected{};
+    if(!address || dladdr(address,&info)==0 || !info.dli_fname)
+        provider_failure("dladdr",name,path,info.dli_fname,nullptr,nullptr,dlerror());
+    char expected_real[PATH_MAX]={},actual_real[PATH_MAX]={};
+    if(!realpath(path,expected_real))
+        provider_failure("canonical_expected",name,path,info.dli_fname,nullptr,nullptr,nullptr);
+    if(!realpath(info.dli_fname,actual_real))
+        provider_failure("canonical_actual",name,path,info.dli_fname,expected_real,nullptr,nullptr);
+    if(std::strcmp(actual_real,expected_real)!=0)
+        provider_failure("canonical_identity",name,path,info.dli_fname,expected_real,actual_real,nullptr);
+    if(stat(path,&expected)!=0 || stat(info.dli_fname,&actual)!=0
+        || !S_ISREG(expected.st_mode) || !S_ISREG(actual.st_mode)
+        || expected.st_dev!=actual.st_dev || expected.st_ino!=actual.st_ino
+        || expected.st_size!=static_cast<off_t>(length) || actual.st_size!=static_cast<off_t>(length))
+        provider_failure("regular_inode_length",name,path,info.dli_fname,expected_real,actual_real,nullptr);
+    if(std::strcmp(path,BLAS)==0 && (expected.st_dev!=38 || expected.st_ino!=2939302343ULL))
+        provider_failure("pinned_blas_device_inode",name,path,info.dli_fname,expected_real,actual_real,nullptr);
+    if(!info.dli_sname || std::strcmp(info.dli_sname,name)!=0)
+        provider_failure("exact_exported_symbol",name,path,info.dli_fname,expected_real,actual_real,nullptr,info.dli_sname);
+}
+template<class F> static F symbol(const char *path,size_t bytes,const char *name) {
+    // NOLOAD never acquires/replaces an unregistered CPU implementation. Hold
+    // the original module through backend_free and the final observation.
+    dlerror();void *handle=dlopen(path,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
+    if(!handle)provider_failure("dlopen_noload",name,path,nullptr,nullptr,nullptr,dlerror());
+    dlerror();void *address=dlsym(handle,name);const char *error=dlerror();
+    if(error)provider_failure("dlsym",name,path,nullptr,nullptr,nullptr,error);
+    attest(address,path,bytes,name);
+    return reinterpret_cast<F>(address);
+}
+static Lookup lookup() {static Lookup f=symbol<Lookup>(BASE,BASE_BYTES,"ggml_backend_reg_get_proc_address");return f;}
+static const char *kind_name(Kind kind) {return kind==Kind::Cpu?"CPU":"BLAS";}
+static Backend &known(ggml_backend_t backend) {
+    auto found=backends.find(backend);
+    checked(backend && found!=backends.end() && !found->second.freed,"unknown or retired public backend");
+    return found->second; // Caller holds mutex; no private backend dereference.
+}
+struct Identity {Kind kind;ggml_backend_dev_t device;ggml_backend_reg_t registry;};
+static Identity identify(ggml_backend_t backend,ggml_backend_dev_t input) {
+    static auto is_cpu=symbol<decltype(&ggml_backend_is_cpu)>(CPU,CPU_BYTES,"ggml_backend_is_cpu");
+    static auto get_device=symbol<decltype(&ggml_backend_get_device)>(BASE,BASE_BYTES,"ggml_backend_get_device");
+    static auto get_reg=symbol<decltype(&ggml_backend_dev_backend_reg)>(BASE,BASE_BYTES,"ggml_backend_dev_backend_reg");
+    static auto get_name=symbol<decltype(&ggml_backend_name)>(BASE,BASE_BYTES,"ggml_backend_name");
+    static auto dev_name=symbol<decltype(&ggml_backend_dev_name)>(BASE,BASE_BYTES,"ggml_backend_dev_name");
+    static auto reg_name=symbol<decltype(&ggml_backend_reg_name)>(BASE,BASE_BYTES,"ggml_backend_reg_name");
+    static auto dev_type=symbol<decltype(&ggml_backend_dev_type)>(BASE,BASE_BYTES,"ggml_backend_dev_type");
+    static auto reg_dev=symbol<decltype(&ggml_backend_reg_dev_get)>(BASE,BASE_BYTES,"ggml_backend_reg_dev_get");
+    checked(backend && input,"nonnull original initialized backend/device");
+    const auto device=get_device(backend);const auto registry=device?get_reg(device):nullptr;
+    checked(device && registry && device==input,"original input device equals public returned device");
+    Kind kind;ggml_backend_reg_t expected=nullptr;
+    const bool cpu=is_cpu(backend);
+    if(cpu) {
+        static auto reg=symbol<decltype(&ggml_backend_cpu_reg)>(CPU,CPU_BYTES,"ggml_backend_cpu_reg");
+        kind=Kind::Cpu;expected=reg();
+    } else {
+        // NOLOAD after actual construction only. Never load/force/disable a provider.
+        static auto is_blas=symbol<decltype(&ggml_backend_is_blas)>(BLAS,BLAS_BYTES,"ggml_backend_is_blas");
+        static auto reg=symbol<decltype(&ggml_backend_blas_reg)>(BLAS,BLAS_BYTES,"ggml_backend_blas_reg");
+        checked(is_blas(backend),"nonCPU backend is not exact pinned public BLAS");
+        kind=Kind::Blas;expected=reg();
+    }
+    const char *name=kind_name(kind);
+    const char *bn=get_name(backend),*dn=dev_name(device),*rn=reg_name(registry);
+    checked(registry==expected && reg_dev(registry,0)==device && bn && dn && rn
+        && std::strcmp(bn,name)==0 && std::strcmp(dn,name)==0 && std::strcmp(rn,name)==0
+        && dev_type(device)==(cpu?GGML_BACKEND_DEVICE_TYPE_CPU:GGML_BACKEND_DEVICE_TYPE_ACCEL),
+        "exact pinned public backend/device/registry singleton, names, type");
+    return {kind,device,registry};
+}
+static void initialized(ggml_backend_t backend,ggml_backend_dev_t input) {
+    const auto identity=identify(backend,input);
+    std::lock_guard<std::mutex> guard(mutex);
+    checked(backends.size()<4 && backends.find(backend)==backends.end(),"four distinct original context backends");
+    Backend record;record.id=backends.size()+1;record.kind=identity.kind;
+    record.device=identity.device;record.registry=identity.registry;backends.emplace(backend,record);
+    append("{\"kind\":\"public_backend_init\",\"backend\":\""+pointer(backend)+"\",\"id\":"+std::to_string(record.id)
+        +",\"backend_kind\":\""+kind_name(record.kind)+"\",\"device\":\""+pointer(identity.device)
+        +"\",\"registry\":\""+pointer(identity.registry)+"\",\"input_device\":\""+pointer(input)
+        +"\",\"backend_name\":\""+kind_name(record.kind)+"\",\"device_name\":\""+kind_name(record.kind)
+        +"\",\"registry_name\":\""+kind_name(record.kind)+"\",\"device_type\":"+(record.kind==Kind::Cpu?"0":"3")
+        +",\"cpu_is_cpu\":"+(record.kind==Kind::Cpu?"true":"false")
+        +",\"input_device_equals_returned\":true,\"exact_provider\":true}");
+}
+static void set_reference(ggml_backend_t backend,bool initialization) {
+    ggml_backend_reg_t reg=nullptr;
+    {
+        std::lock_guard<std::mutex> guard(mutex);auto &record=known(backend);
+        checked(record.kind==Kind::Cpu && record.set==(initialization?0:1)
+            && record.abort==0,"reference setter only exact CPU, original init/abort order");
+        reg=record.registry;
+    }
+    void *raw=lookup()(reg,"ggml_backend_cpu_set_use_ref");
+    attest(raw,CPU,CPU_BYTES,"ggml_backend_cpu_set_use_ref");
+    reinterpret_cast<Ref>(raw)(backend,true); // Sole numerical policy change, same original V1 CPU setter.
+    std::lock_guard<std::mutex> guard(mutex);auto &record=known(backend);
+    if(!initialization)++record.abort;
+    ++record.set;
+    append("{\"kind\":\"public_cpu_reference_setter\",\"backend\":\""+pointer(backend)
+        +"\",\"origin\":\""+(initialization?"public_dev_init":"existing_public_abort_setter_callback")
+        +"\",\"requested_use_ref\":true}");
+}
+static void abort_setter(ggml_backend_t backend,ggml_abort_callback callback,void *data) {
+    const int saved=errno;ggml_backend_reg_t reg=nullptr;
+    {std::lock_guard<std::mutex> guard(mutex);auto &record=known(backend);
+        checked(record.kind==Kind::Cpu,"abort setter cannot target BLAS");reg=record.registry;}
+    void *raw=lookup()(reg,"ggml_backend_set_abort_callback");
+    attest(raw,CPU,CPU_BYTES,"ggml_backend_cpu_set_abort_callback");
+    errno=saved;reinterpret_cast<ggml_backend_set_abort_callback_t>(raw)(backend,callback,data);
+    const int after=errno;set_reference(backend,false);errno=after;
+}
+static void cpu_compute_only(ggml_backend_t backend,const char *api) {
+    auto &record=known(backend);
+    if(record.kind!=Kind::Cpu) {
+        append("{\"kind\":\"nonCPU_compute_refused\",\"backend\":\""+pointer(backend)
+            +"\",\"backend_kind\":\"BLAS\",\"api\":\""+api+"\",\"delegated\":false}");
+        fatal("V4 pinned BLAS actual compute refused BEFORE delegation; capture UNQUALIFIED");
+    }
+    checked(record.set==2 && record.abort==1,"actual CPU configured before public compute");
+}
+static void begin(Scope &current,ggml_backend_t backend,ggml_cgraph *graph,
+        ggml_backend_graph_plan_t plan,const char *api) {
+    check_affinity();checked(scope==nullptr && (graph || plan),"one nonnested public CPU compute scope");
+    std::lock_guard<std::mutex> guard(mutex);cpu_compute_only(backend,api);
+    checked(inflight==0,"one globally serialized original public compute");auto &record=known(backend);
+    size_t plan_id=0;
+    if(plan) {auto found=plans.find(plan);checked(found!=plans.end() && found->second.backend==backend,
+        "opaque public plan belongs to same live CPU backend");plan_id=found->second.id;}
+    if(sync_scope) {checked(!plan && sync_scope->backend==backend && sync_scope->graph==graph
+        && sync_scope->async==0,"sync delegates once to same public async graph");++sync_scope->async;}
+    current.backend=backend;current.graph=graph;current.plan=plan;current.api=api;current.sequence=++public_calls;
+    if(plan){++plan_calls;++record.plans;}else{++async_calls;++record.async;}
+    if(sync_scope)sync_scope->sequence=current.sequence;
+    scope=&current;inflight=current.sequence;
+    append("{\"kind\":\"public_compute_begin\",\"backend\":\""+pointer(backend)+"\",\"sequence\":"
+        +std::to_string(current.sequence)+",\"api\":\""+api+"\",\"graph\":\""+pointer(graph)
+        +"\",\"opaque_plan\":\""+pointer(plan)+"\",\"plan_id\":"+std::to_string(plan_id)+"}");
+}
+static void finish(Scope &current,ggml_status status) {
+    checked(scope==&current && !current.in_inner && current.inner==1 && status==GGML_STATUS_SUCCESS,
+        "each successful public CPU compute has exactly one successful actual inner call");
+    std::lock_guard<std::mutex> guard(mutex);checked(inflight==current.sequence,"exact active original compute sequence");
+    ++public_success;inflight=0;
+    append("{\"kind\":\"public_compute_return\",\"backend\":\""+pointer(current.backend)
+        +"\",\"sequence\":"+std::to_string(current.sequence)+",\"status\":0,\"inner_calls\":1}");
+    scope=nullptr;
+}
+}
+
+extern "C" ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device,const char *params) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_dev_init)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_dev_init");
+    errno=saved;auto backend=original(device,params);const int after=errno;
+    if(ref_control::armed) {
+        ref_control::initialized(backend,device);
+        bool cpu=false;
+        {std::lock_guard<std::mutex> guard(ref_control::mutex);
+            cpu=ref_control::known(backend).kind==ref_control::Kind::Cpu;}
+        if(cpu)ref_control::set_reference(backend,true);
+    }
+    errno=after;return backend;
+}
+extern "C" void *ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg,const char *name) {
+    const int saved=errno;auto original=ref_control::lookup();errno=saved;
+    void *value=original(reg,name);const int after=errno;
+    if(ref_control::armed && value && name && std::strcmp(name,"ggml_backend_set_abort_callback")==0) {
+        ref_control::attest(value,ref_control::CPU,ref_control::CPU_BYTES,"ggml_backend_cpu_set_abort_callback");
+        static auto cpu_reg=ref_control::symbol<decltype(&ggml_backend_cpu_reg)>(ref_control::CPU,
+            ref_control::CPU_BYTES,"ggml_backend_cpu_reg");
+        ref_control::checked(reg==cpu_reg(),"exact CPU registry supplies original abort callback");
+        value=reinterpret_cast<void *>(&ref_control::abort_setter);
+    }
+    errno=after;return value;
+}
+extern "C" enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend,struct ggml_cgraph *graph) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_graph_compute_async)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_graph_compute_async");
+    ref_control::Scope current;
+    if(ref_control::armed)ref_control::begin(current,backend,graph,nullptr,"async");
+    errno=saved;const auto status=original(backend,graph);const int after=errno;
+    if(ref_control::armed)ref_control::finish(current,status);
+    errno=after;return status;
+}
+extern "C" enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend,struct ggml_cgraph *graph) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_graph_compute)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_graph_compute");
+    ref_control::SyncScope current;
+    if(ref_control::armed) {
+        ref_control::checked(ref_control::sync_scope==nullptr && ref_control::scope==nullptr && graph,
+            "nonnested public sync route");
+        {std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::cpu_compute_only(backend,"sync");}
+        current.backend=backend;current.graph=graph;ref_control::sync_scope=&current;
+    }
+    errno=saved;const auto status=original(backend,graph);const int after=errno;
+    if(ref_control::armed) {
+        ref_control::checked(current.async==1 && status==GGML_STATUS_SUCCESS && ref_control::scope==nullptr,
+            "actual public sync delegates exactly once through guarded async");
+        std::lock_guard<std::mutex> guard(ref_control::mutex);++ref_control::sync_calls;++ref_control::known(backend).sync;
+        ref_control::append("{\"kind\":\"public_sync_return\",\"backend\":\""+ref_control::pointer(backend)
+            +"\",\"sequence\":"+std::to_string(current.sequence)+",\"status\":0}");
+        ref_control::sync_scope=nullptr;
+    }
+    errno=after;return status;
+}
+extern "C" ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend,struct ggml_cgraph *graph) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_graph_plan_create)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_graph_plan_create");
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::cpu_compute_only(backend,"plan_create");
+        ref_control::checked(graph && ref_control::scope==nullptr && ref_control::plans.size()<64,
+            "bounded nonnested public opaque plan creation");
+    }
+    errno=saved;auto plan=original(backend,graph);const int after=errno;
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);
+        ref_control::checked(plan && ref_control::plans.find(plan)==ref_control::plans.end()
+            && ref_control::next_plan<65536,"unique bounded live opaque plan");
+        ref_control::Plan record;record.backend=backend;record.id=++ref_control::next_plan;
+        ref_control::plans.emplace(plan,record);++ref_control::plan_creates;
+        ref_control::append("{\"kind\":\"public_plan_create\",\"backend\":\""+ref_control::pointer(backend)
+            +"\",\"opaque_plan\":\""+ref_control::pointer(plan)+"\",\"plan_id\":"+std::to_string(record.id)+"}");
+    }
+    errno=after;return plan;
+}
+extern "C" void ggml_backend_graph_plan_free(ggml_backend_t backend,ggml_backend_graph_plan_t plan) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_graph_plan_free)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_graph_plan_free");
+    size_t id=0;
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::cpu_compute_only(backend,"plan_free");
+        const auto found=ref_control::plans.find(plan);
+        ref_control::checked(ref_control::scope==nullptr && found!=ref_control::plans.end()
+            && found->second.backend==backend,"public plan freed once by original CPU owner");id=found->second.id;
+    }
+    errno=saved;original(backend,plan);const int after=errno;
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::plans.erase(plan);++ref_control::plan_frees;
+        ref_control::append("{\"kind\":\"public_plan_free\",\"backend\":\""+ref_control::pointer(backend)
+            +"\",\"opaque_plan\":\""+ref_control::pointer(plan)+"\",\"plan_id\":"+std::to_string(id)+"}");
+    }
+    errno=after;
+}
+extern "C" enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend,ggml_backend_graph_plan_t plan) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_graph_plan_compute)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_graph_plan_compute");
+    ref_control::Scope current;
+    if(ref_control::armed)ref_control::begin(current,backend,nullptr,plan,"plan");
+    errno=saved;const auto status=original(backend,plan);const int after=errno;
+    if(ref_control::armed)ref_control::finish(current,status);
+    errno=after;return status;
+}
+extern "C" enum ggml_status ggml_graph_compute(struct ggml_cgraph *graph,struct ggml_cplan *plan) {
+    const int saved=errno;
+    static ref_control::Compute original=ref_control::symbol<ref_control::Compute>(ref_control::CPU,
+        ref_control::CPU_BYTES,"ggml_graph_compute");
+    int actual_threads=0;
+    if(ref_control::armed) {
+        ref_control::check_affinity();auto *current=ref_control::scope;
+        ref_control::checked(current && !current->in_inner && current->inner==0 && graph && plan && plan->use_ref
+            && (!current->graph || current->graph==graph),"actual CPU inner compute has exact public TLS owner/use_ref");
+        ref_control::checked(plan->n_threads>=1 && plan->n_threads<=8,"actual public plan thread bound");
+        actual_threads=plan->n_threads;current->in_inner=true;
+    }
+    errno=saved;const auto status=original(graph,plan);const int after=errno;
+    if(ref_control::armed) {
+        auto *current=ref_control::scope;
+        ref_control::checked(current && current->in_inner && plan->use_ref && plan->n_threads==actual_threads
+            && status==GGML_STATUS_SUCCESS,
+            "actual inner CPU compute successful with reference flag unchanged");
+        std::lock_guard<std::mutex> guard(ref_control::mutex);++ref_control::graph_calls;++ref_control::graph_success;
+        ++ref_control::known(current->backend).inner;++current->inner;current->in_inner=false;
+        ref_control::thread_min=std::min(ref_control::thread_min,plan->n_threads);
+        ref_control::thread_max=std::max(ref_control::thread_max,plan->n_threads);
+        ref_control::append("{\"kind\":\"actual_public_graph_plan\",\"backend\":\""+ref_control::pointer(current->backend)
+            +"\",\"sequence\":"+std::to_string(current->sequence)+",\"graph\":\""+ref_control::pointer(graph)
+            +"\",\"cplan\":\""+ref_control::pointer(plan)+"\",\"use_ref_before\":true,\"use_ref_after\":true,\"n_threads\":"
+            +std::to_string(plan->n_threads)+",\"status\":0}");
+    }
+    errno=after;return status;
+}
+extern "C" void ggml_backend_free(ggml_backend_t backend) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_free)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_free");
+    ref_control::Kind kind=ref_control::Kind::Cpu;
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);auto &record=ref_control::known(backend);
+        ref_control::checked(ref_control::scope==nullptr && ref_control::sync_scope==nullptr
+            && (record.kind==ref_control::Kind::Cpu?(record.abort==1 && record.set==2):(record.set==0 && record.abort==0)),
+            "only exact configured CPU or unmodified BLAS retirement outside math");
+        for(const auto &item:ref_control::plans)ref_control::checked(item.second.backend!=backend,"backend retains live opaque plan");
+        record.freed=true;kind=record.kind;
+    }
+    errno=saved;original(backend);const int after=errno;
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);
+        ref_control::append("{\"kind\":\"public_backend_free\",\"backend\":\""+ref_control::pointer(backend)
+            +"\",\"backend_kind\":\""+ref_control::kind_name(kind)+"\"}");
+    }
+    errno=after;
+}
+
+#define main zg_unchanged_native_raw_main
+#include "native_raw_original.cpp"
+#undef main
+
+int main(int argc,char **argv) {
+    try {
+        require(argc==9,"same original native_raw arguments required");
+        require(std::string(argv[3])=="8192" && std::string(argv[6])=="8",
+            "closed fresh8192/CPU8 control");
+        require((std::string(argv[4])=="64" || std::string(argv[4])=="512")
+            && std::string(argv[5])==argv[4],"predeclared same bulk batch/ubatch64 or512");
+        const char *cpu_env=std::getenv("ZG_USE_REF_CPU_SET");
+        require(cpu_env && std::strlen(cpu_env)<=128,"explicit ROOT CPU8 affinity declaration");
+        CPU_ZERO(&ref_control::expected_affinity);
+        std::istringstream cpu_input(cpu_env);std::string cpu_text;
+        while(std::getline(cpu_input,cpu_text,',')) {
+            require(!cpu_text.empty() && std::all_of(cpu_text.begin(),cpu_text.end(),
+                [](unsigned char c){return c>='0' && c<='9';}),"canonical CPU number");
+            const unsigned long cpu=std::stoul(cpu_text);
+            require(cpu<CPU_SETSIZE && !CPU_ISSET(cpu,&ref_control::expected_affinity),"unique bounded CPU number");
+            CPU_SET(cpu,&ref_control::expected_affinity);ref_control::declared_cpus.push_back(static_cast<int>(cpu));
+        }
+        require(ref_control::declared_cpus.size()==8,"exact eight declared CPUs");
+        ref_control::check_affinity();
+        ref_control::trace_path=std::string(argv[8])+"/use-ref-public-trace.jsonl";
+        ref_control::armed=true;
+        const int result=zg_unchanged_native_raw_main(argc,argv);
+        require(result==0,"unchanged native capture failed");
+        std::lock_guard<std::mutex> guard(ref_control::mutex);
+        require(ref_control::backends.size()==4 && ref_control::plans.empty()
+            && ref_control::scope==nullptr && ref_control::sync_scope==nullptr && ref_control::inflight==0,
+            "all original public owners retired");
+        size_t cpu=0,blas=0;
+        for(const auto &item:ref_control::backends) {
+            const auto &r=item.second;
+            require(r.freed,"every original CPU/BLAS context retired");
+            if(r.kind==ref_control::Kind::Cpu) {
+                ++cpu;require(r.abort==1 && r.set==2 && r.inner>0 && r.inner==r.async+r.plans,
+                    "each original CPU nonvacuous with complete public/inner use_ref coverage");
+            } else {
+                ++blas;require(r.abort==0 && r.set==0 && r.async==0 && r.plans==0 && r.inner==0 && r.sync==0,
+                    "pinned BLAS initialized/retired unmodified and never computed");
+            }
+        }
+        require(cpu==2 && blas==2,"exact original two CPU and two pinned BLAS context backends");
+        const size_t minimum_calls=(8191+std::stoul(argv[4])-1)/std::stoul(argv[4])+2;
+        require(ref_control::graph_calls>=minimum_calls && ref_control::graph_calls==ref_control::graph_success
+            && ref_control::graph_calls==ref_control::public_calls
+            && ref_control::public_calls==ref_control::public_success
+            && ref_control::public_calls==ref_control::async_calls+ref_control::plan_calls
+            && ref_control::plan_creates==ref_control::plan_frees,
+            "actual public/inner conservation and all opaque plans retired");
+        ref_control::check_affinity();
+        require(ref_control::trace_fd>=0 && fsync(ref_control::trace_fd)==0,"public trace complete");
+        require(close(ref_control::trace_fd)==0,"public trace close");ref_control::trace_fd=-1;
+        std::ifstream maps("/proc/self/maps",std::ios::binary);
+        std::vector<char> text(2*1024*1024+1);
+        maps.read(text.data(),static_cast<std::streamsize>(text.size()));
+        const auto count=maps.gcount();
+        require(maps.eof() && count>0 && count<=2*1024*1024,"bounded actual loaded mappings");
+        write_bytes(folder/"use-ref-loaded-maps.txt",text.data(),static_cast<size_t>(count));
+        std::ifstream status("/proc/self/status",std::ios::binary);
+        std::vector<char> status_text(64*1024+1);
+        status.read(status_text.data(),static_cast<std::streamsize>(status_text.size()));
+        const auto status_count=status.gcount();
+        require(status.eof() && status_count>0 && status_count<=64*1024,"bounded actual process status");
+        write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
+        std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
+        report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
+            <<"\"control_source_revision\":\"public_use_ref_exact_blas_lifecycle_cpu_compute_v4\","
+            <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\","
+            <<"\"CPU_backend_count\":2,\"BLAS_backend_count\":2,\"public_setter_calls\":4,"
+            <<"\"existing_abort_setter_calls\":2,\"BLAS_compute_delegations\":0,"
+            <<"\"actual_public_plan_calls\":"<<ref_control::graph_calls
+            <<",\"actual_successful_public_plan_calls\":"<<ref_control::graph_success
+            <<",\"public_compute_calls\":"<<ref_control::public_calls
+            <<",\"public_async_calls\":"<<ref_control::async_calls
+            <<",\"public_opaque_plan_compute_calls\":"<<ref_control::plan_calls
+            <<",\"public_sync_calls\":"<<ref_control::sync_calls
+            <<",\"public_opaque_plan_creates\":"<<ref_control::plan_creates
+            <<",\"public_opaque_plan_frees\":"<<ref_control::plan_frees
+            <<",\"opaque_plan_route_observed\":"<<(ref_control::plan_calls?"true":"false")
+            <<",\"actual_plan_threads_min\":"<<ref_control::thread_min
+            <<",\"actual_plan_threads_max\":"<<ref_control::thread_max
+            <<",\"actual_public_plan_use_ref_all_true\":true,\"public_inner_TLS_scope_exact\":true,"
+            <<"\"BLAS_construction_retirement_only\":true,\"no_BLAS_disable_or_offload_change\":true,"
+            <<"\"actual_CPU_affinity\":[";
+        for(size_t i=0;i<ref_control::declared_cpus.size();++i)report<<(i?",":"")<<ref_control::declared_cpus[i];
+        report<<"],\"backend_private_getter_available\":false,"
+            <<"\"plan_to_backend_identity_publicly_available\":false,\"opaque_plan_public_association_checked\":true,"
+            <<"\"CPU_REPACK_policy_unchanged\":true,\"native_raw_token_loop_unchanged\":true,"
+            <<"\"historical_prefix_or_cause_closed\":false}\n";
+        report.flush();require(bool(report),"control report write");return 0;
+    }catch(const std::exception &error){std::cerr<<"USE_REF_CONTROL_UNQUALIFIED "<<error.what()<<'\n';return 2;}
+}
--- /dev/null
+++ b/.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/build-argv-v4.json
@@ -0,0 +1,44 @@
+{
+  "status": "PREPARED_ROOT_BUILD_ONLY_UNEXECUTED",
+  "argv": [
+    "c++",
+    "-std=c++17",
+    "-O2",
+    "-Wall",
+    "-Wextra",
+    "-Werror",
+    "-I",
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include",
+    "-I",
+    "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include",
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v4.cpp",
+    "-L",
+    "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib",
+    "-L",
+    "/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib",
+    "-Wl,-rpath,/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib:/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib",
+    "-Wl,--dynamic-linker,/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib/ld-linux-x86-64.so.2",
+    "-Wl,--export-dynamic-symbol=ggml_backend_dev_init",
+    "-Wl,--export-dynamic-symbol=ggml_backend_reg_get_proc_address",
+    "-Wl,--export-dynamic-symbol=ggml_graph_compute",
+    "-Wl,--export-dynamic-symbol=ggml_backend_free",
+    "-Wl,--export-dynamic-symbol=ggml_backend_graph_compute_async",
+    "-Wl,--export-dynamic-symbol=ggml_backend_graph_compute",
+    "-Wl,--export-dynamic-symbol=ggml_backend_graph_plan_create",
+    "-Wl,--export-dynamic-symbol=ggml_backend_graph_plan_free",
+    "-Wl,--export-dynamic-symbol=ggml_backend_graph_plan_compute",
+    "-lllama",
+    "-lggml",
+    "-lggml-base",
+    "-lm",
+    "-ldl",
+    "-pthread",
+    "-o",
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v4"
+  ],
+  "compiler_identity": "ROOT must record real compiler and own finite build receipt before execution",
+  "native_math_source_unchanged": true,
+  "source_revision": "public_use_ref_exact_blas_lifecycle_cpu_compute_v4",
+  "generic_runner_v1_freeze_scope": "unchanged generic helper; V4 source/build/compiled-route/invocation binding independently mandatory",
+  "scope": "SOURCE ONLY exact BLAS construction/retirement; refuse EVERY nonCPU compute before delegation; CPU public-to-inner TLS/use_ref guard; no arithmetic acceptance"
+}
--- /dev/null
+++ b/.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/trace_v4.py
@@ -0,0 +1,184 @@
+"""Pure bounded V4 public-route parser. Synthetic fixtures do not qualify native routing."""
+import json
+import re
+from common import require, unique
+
+
+def opaque(value, nullable=False):
+    require(isinstance(value, str) and re.fullmatch(r'0x[0-9a-f]{1,16}', value)
+            and (nullable or int(value, 16) > 0), 'bounded opaque public identity')
+    return value
+
+
+def integer(value, minimum=0):
+    require(type(value) is int and minimum <= value <= 65536, 'bounded exact integer')
+    return value
+
+
+def validate_trace(raw, report, prefix):
+    require(type(prefix) is int and prefix in [64, 512], 'closed actual prefix geometry')
+    require(0 < len(raw) <= 8 * 1024**2 and raw.endswith(b'\n'), 'complete bounded public trace')
+    lines = raw.splitlines()
+    require(len(lines) <= 65536, 'bounded actual public call count')
+    backends, plans, singletons = {}, {}, {}
+    active = None
+    completed, creates, frees, syncs = 0, 0, 0, set()
+    counts = {'async': 0, 'plan': 0}
+    threads = []
+    last_completed = None
+
+    def cpu(name):
+        opaque(name)
+        require(name in backends, 'known original backend')
+        owner = backends[name]
+        require(not owner['freed'] and owner['backend_kind'] == 'CPU'
+                and owner['setters'] == ['public_dev_init', 'existing_public_abort_setter_callback'],
+                'only live original fully configured CPU computes')
+        return owner
+
+    for line in lines:
+        require(0 < len(line) < 2048, 'bounded actual trace line')
+        row = json.loads(line, object_pairs_hook=unique)
+        require(type(row) is dict, 'actual public trace object')
+        kind = row.get('kind')
+        if kind == 'public_backend_init':
+            require(set(row) == {'kind', 'backend', 'id', 'backend_kind', 'device', 'registry',
+                                'input_device_equals_returned', 'exact_provider', 'input_device', 'backend_name',
+                                'device_name', 'registry_name', 'device_type', 'cpu_is_cpu'}
+                    and active is None and row['input_device_equals_returned'] is True
+                    and row['exact_provider'] is True and row['backend_kind'] in ['CPU', 'BLAS'],
+                    'exact source-attested public construction record')
+            name = opaque(row['backend'])
+            require(name not in backends and len(backends) < 4
+                    and integer(row['id'], 1) == len(backends) + 1, 'four unique original construction identities')
+            singleton = (opaque(row['device']), opaque(row['registry']))
+            family = row['backend_kind']
+            require(opaque(row['input_device']) == singleton[0]
+                    and all(row[field] == family for field in ['backend_name', 'device_name', 'registry_name'])
+                    and type(row['device_type']) is int and row['device_type'] == (0 if family == 'CPU' else 3)
+                    and row['cpu_is_cpu'] is (family == 'CPU'), 'actual public type/names/input/CPU predicate')
+            if family in singletons:
+                require(singletons[family] == singleton, 'same exact public provider singleton')
+            else:
+                require(all(a != singleton[0] and b != singleton[1] for a, b in singletons.values()),
+                        'CPU and BLAS have distinct public device/registry singletons')
+                singletons[family] = singleton
+            backends[name] = dict(backend_kind=family, setters=[], freed=False, async_calls=0,
+                                  plan_calls=0, inner=0, sync=0)
+        elif kind == 'public_cpu_reference_setter':
+            require(set(row) == {'kind', 'backend', 'origin', 'requested_use_ref'}
+                    and active is None and row['requested_use_ref'] is True, 'exact CPU setter record')
+            name = opaque(row['backend'])
+            require(name in backends and not backends[name]['freed']
+                    and backends[name]['backend_kind'] == 'CPU', 'no BLAS/unknown/retired setter')
+            setters = backends[name]['setters']
+            expected = ['public_dev_init', 'existing_public_abort_setter_callback']
+            require(len(setters) < 2 and row['origin'] == expected[len(setters)], 'original setter/callback order')
+            setters.append(row['origin'])
+        elif kind == 'public_plan_create':
+            require(set(row) == {'kind', 'backend', 'opaque_plan', 'plan_id'} and active is None,
+                    'exact public opaque-plan construction record')
+            cpu(row['backend'])
+            name = opaque(row['opaque_plan'])
+            creates += 1
+            require(name not in plans and len(plans) < 64 and integer(row['plan_id'], 1) == creates,
+                    'bounded live public opaque-plan association/generation')
+            plans[name] = (row['backend'], creates)
+        elif kind == 'public_plan_free':
+            require(set(row) == {'kind', 'backend', 'opaque_plan', 'plan_id'} and active is None,
+                    'exact public opaque-plan retirement record')
+            cpu(row['backend'])
+            name = opaque(row['opaque_plan'])
+            require(name in plans and plans[name] == (row['backend'], integer(row['plan_id'], 1)),
+                    'same public owner retires original opaque plan once')
+            del plans[name]
+            frees += 1
+        elif kind == 'public_compute_begin':
+            require(set(row) == {'kind', 'backend', 'sequence', 'api', 'graph', 'opaque_plan', 'plan_id'}
+                    and active is None and integer(row['sequence'], 1) == completed + 1
+                    and row['api'] in ['async', 'plan'], 'one ordered actual public compute scope')
+            cpu(row['backend'])
+            graph, plan = opaque(row['graph'], True), opaque(row['opaque_plan'], True)
+            plan_id = integer(row['plan_id'])
+            if row['api'] == 'async':
+                require(int(graph, 16) > 0 and plan == '0x0' and plan_id == 0, 'actual async graph only')
+            else:
+                require(graph == '0x0' and plan in plans and plans[plan] == (row['backend'], plan_id),
+                        'actual opaque plan uses original same-backend public association')
+            active = dict(row, inner=False)
+        elif kind == 'actual_public_graph_plan':
+            require(set(row) == {'kind', 'backend', 'sequence', 'graph', 'cplan', 'use_ref_before',
+                                'use_ref_after', 'n_threads', 'status'} and active is not None,
+                    'actual CPU public cplan must be inside one original public scope')
+            require(row['backend'] == active['backend'] and integer(row['sequence'], 1) == active['sequence']
+                    and not active['inner'] and row['use_ref_before'] is True and row['use_ref_after'] is True
+                    and type(row['status']) is int and row['status'] == 0, 'exact TLS owner and pre/post reference policy')
+            opaque(row['graph'])
+            opaque(row['cplan'])
+            require(active['api'] == 'plan' or row['graph'] == active['graph'], 'same async graph pointer')
+            require(type(row['n_threads']) is int and 1 <= row['n_threads'] <= 8, 'actual CPU thread bound')
+            cpu(row['backend'])['inner'] += 1
+            threads.append(row['n_threads'])
+            active['inner'] = True
+        elif kind == 'public_compute_return':
+            require(set(row) == {'kind', 'backend', 'sequence', 'status', 'inner_calls'} and active is not None,
+                    'actual public return has one active original scope')
+            require(row['backend'] == active['backend'] and integer(row['sequence'], 1) == active['sequence']
+                    and active['inner'] and type(row['status']) is int and row['status'] == 0
+                    and type(row['inner_calls']) is int and row['inner_calls'] == 1,
+                    'one successful inner compute for one successful public compute')
+            owner = cpu(row['backend'])
+            owner[active['api'] + '_calls'] += 1
+            counts[active['api']] += 1
+            completed += 1
+            last_completed = (row['backend'], completed, active['api'])
+            active = None
+        elif kind == 'public_sync_return':
+            require(set(row) == {'kind', 'backend', 'sequence', 'status'} and active is None
+                    and type(row['status']) is int and row['status'] == 0, 'exact sync wrapper return')
+            cpu(row['backend'])['sync'] += 1
+            sequence = integer(row['sequence'], 1)
+            require(last_completed == (row['backend'], sequence, 'async') and sequence not in syncs,
+                    'sync references exactly one just-completed same-backend async, no double math count')
+            syncs.add(sequence)
+        elif kind == 'public_backend_free':
+            require(set(row) == {'kind', 'backend', 'backend_kind'} and active is None, 'exact lifecycle retirement')
+            name = opaque(row['backend'])
+            require(name in backends and not backends[name]['freed']
+                    and backends[name]['backend_kind'] == row['backend_kind']
+                    and all(owner != name for owner, _ in plans.values()), 'known original owner retires once without plans')
+            owner = backends[name]
+            if owner['backend_kind'] == 'CPU':
+                cpu(name)
+                require(owner['inner'] > 0 and owner['inner'] == owner['async_calls'] + owner['plan_calls'],
+                        'nonvacuous per-CPU route conservation before retirement')
+            else:
+                require(owner['setters'] == [] and all(owner[key] == 0 for key in
+                        ['async_calls', 'plan_calls', 'inner', 'sync']), 'BLAS lifecycle only, never configured/computed')
+            owner['freed'] = True
+        else:
+            raise ValueError('unaccepted public record (including actual nonCPU compute refusal)')
+    minimum = (8191 + prefix - 1) // prefix + 2
+    require(active is None and not plans and creates == frees and len(backends) == 4
+            and sum(b['backend_kind'] == 'CPU' for b in backends.values()) == 2
+            and sum(b['backend_kind'] == 'BLAS' for b in backends.values()) == 2
+            and all(b['freed'] for b in backends.values()) and completed >= minimum,
+            'complete two CPU/two BLAS lifecycle and nonvacuous public math coverage')
+    expected = {'CPU_backend_count': 2, 'BLAS_backend_count': 2, 'public_setter_calls': 4,
+                'existing_abort_setter_calls': 2, 'BLAS_compute_delegations': 0,
+                'actual_public_plan_calls': completed, 'actual_successful_public_plan_calls': completed,
+                'public_compute_calls': completed, 'public_async_calls': counts['async'],
+                'public_opaque_plan_compute_calls': counts['plan'], 'public_sync_calls': len(syncs),
+                'public_opaque_plan_creates': creates, 'public_opaque_plan_frees': frees,
+                'actual_plan_threads_min': min(threads), 'actual_plan_threads_max': max(threads)}
+    require(all(type(report.get(key)) is int and report[key] == value for key, value in expected.items()),
+            'all actual per-route counters reconcile without sync double counting')
+    require(report.get('opaque_plan_route_observed') is (counts['plan'] > 0),
+            'unobserved opaque-plan route is never claimed physically qualified')
+    return dict(actual_cpu_contexts=2, actual_blas_lifecycle_contexts=2, actual_blas_math_calls=0,
+                actual_setter_calls=4, actual_original_abort_setter_calls=2, actual_plan_calls=completed,
+                actual_public_async_calls=counts['async'], actual_opaque_plan_calls=counts['plan'],
+                actual_public_sync_calls=len(syncs), actual_opaque_plan_creates=creates,
+                actual_opaque_plan_route_observed=counts['plan'] > 0,
+                actual_public_plan_use_ref_all_true=True, actual_public_inner_TLS_scope_exact=True,
+                actual_threads_min=min(threads), actual_threads_max=max(threads))
--- /dev/null
+++ b/.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/v4_sources.py
@@ -0,0 +1,67 @@
+"""ROOT ONLY V4 source/static-provider prerequisites; never calls a native entry point."""
+from pathlib import Path
+import re
+from common import HERE, bounded, digest, jread, require
+
+REVISION = 'public_use_ref_exact_blas_lifecycle_cpu_compute_v4'
+
+
+def verify_sources(expected_manifest=None):
+    manifest_path = HERE / 'freeze-manifest-v4.json'
+    if expected_manifest is not None:
+        require(digest(manifest_path)['sha256'] == expected_manifest, 'independently reviewed V4 manifest')
+    manifest = jread(manifest_path)
+    for label, directory in [('original_freeze', 'frozen-prepared'), ('v2_freeze', 'frozen-v2'),
+                             ('v3_freeze', 'frozen-v3')]:
+        pin = manifest[label]
+        path = HERE / pin['file']
+        require(digest(path) == {key: pin[key] for key in ['bytes', 'sha256']}, 'unchanged prior version manifest')
+        for name, expected in jread(path)['files'].items():
+            require(digest(HERE / name) == digest(HERE / directory / name) == expected,
+                    'immutable original/prior source: ' + name)
+    for name, expected in manifest['files'].items():
+        require(digest(HERE / name) == digest(HERE / 'frozen-v4' / name) == expected,
+                'reviewed V4 live/frozen source: ' + name)
+    for row in manifest['preserved_failed_metadata']:
+        require(digest(row['path']) == {key: row[key] for key in ['bytes', 'sha256']},
+                'immutable failed case evidence')
+    routes = jread(HERE / 'static-route-pins-v4.json')
+    for row in routes['files']:
+        require(digest(HERE / row['path']) == {key: row[key] for key in ['bytes', 'sha256']},
+                'actual ROOT static excerpt differs: ' + row['path'])
+    for row in routes['primary_source_files']:
+        require(digest(row['path']) == {key: row[key] for key in ['bytes', 'sha256']},
+                'pinned source route/header differs: ' + row['path'])
+    root = HERE / 'root-v4-elf-audit'
+    selected = [('base-scheduler.txt', r'3cb18:.*call.*18840.*<ggml_backend_graph_compute_async@plt>'),
+                ('base-scheduler.txt', r'3d12b:.*call.*18840.*<ggml_backend_graph_compute_async@plt>'),
+                ('base-sync.txt', r'375fc:.*call.*18840.*<ggml_backend_graph_compute_async@plt>'),
+                ('cpu-entry.txt', r'1f34a:.*call.*15a40.*<ggml_graph_compute@plt>'),
+                ('cpu-entry.txt', r'1f384:.*jmp.*15a40.*<ggml_graph_compute@plt>'),
+                ('base-relocs.txt', r'00000000000dbb18\s+.*R_X86_64_JUMP_SLOT\s+.*ggml_backend_graph_compute_async'),
+                ('cpu-relocs.txt', r'000000000014de28\s+.*R_X86_64_JUMP_SLOT\s+.*ggml_graph_compute')]
+    for name, expression in selected:
+        require(re.search(expression, bounded(root / name, 256 * 1024).decode()) is not None,
+                'actual selected compiled route missing: ' + name)
+    for name in ['base-dynamic.txt', 'cpu-dynamic.txt']:
+        require('SYMBOLIC' not in bounded(root / name, 65536).decode(), 'selected provider lacks SYMBOLIC')
+    return manifest
+
+
+def verify_runtime_files():
+    # Deliberate ROOT-time hashes, never author preparation. No dlopen/native execution.
+    blas = jread(HERE / 'root-blas-provider-pin.json')
+    path = Path(blas['path'])
+    require(not path.is_symlink() and str(path.resolve(strict=True)) == blas['canonical'],
+            'exact immutable regular BLAS path')
+    st = path.stat()
+    require(st.st_dev == blas['dev'] and st.st_ino == blas['inode']
+            and digest(path) == {key: blas[key] for key in ['bytes', 'sha256']},
+            'actual pinned BLAS inode/content')
+    for name in ['base', 'cpu']:
+        row = jread(HERE / 'root-v4-elf-audit' / (name + '.json'))
+        require(str(Path(row['input']).resolve(strict=True)) == row['canonical']
+                and digest(row['canonical']) == {key: row[key] for key in ['bytes', 'sha256']},
+                'actual selected compiled provider content: ' + name)
+    return dict(BLAS=blas, selected_ELF_scope='two scheduler->async sites, sync->async, '
+                'CPU ordinary+opaque-plan->inner sites; no universal all-call/runtime completeness claim')
--- /dev/null
+++ b/.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/bind_v4.py
@@ -0,0 +1,98 @@
+"""ROOT ONLY: bind actual V4 compile/run/static-provider receipts; no native entry calls."""
+import argparse
+from pathlib import Path
+import re
+
+from common import HERE, REPO, digest, jread, require, write
+from v4_sources import REVISION, verify_sources, verify_runtime_files
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--manifest-sha256", required=True)
+    parser.add_argument("--build", required=True, type=Path)
+    parser.add_argument("--run", required=True, type=Path)
+    parser.add_argument("--new-binding", required=True, type=Path)
+    args = parser.parse_args()
+    require(re.fullmatch("[a-f0-9]{64}", args.manifest_sha256) is not None,
+            "independently reviewed manifest SHA")
+    path = HERE / "freeze-manifest-v4.json"
+    manifest = verify_sources(args.manifest_sha256)
+    extra_providers = verify_runtime_files()
+    build, run = args.build.resolve(strict=True), args.run.resolve(strict=True)
+    destination = args.new_binding.resolve()
+    require(build.is_relative_to(HERE) and run.is_relative_to(HERE)
+            and destination.is_relative_to(HERE) and not destination.exists(),
+            "owned actual build/run and fresh binding only")
+    requested, result = jread(build / "requested.json"), jread(build / "result.json")
+    compiler = manifest["root_actual_compiler_pin"]
+    prepared = jread(HERE / "build-argv-v4.json")
+    actual_inputs = {row["path"]: digest(row["path"]) for row in manifest["compile_inputs"]}
+    require(actual_inputs == {row["path"]: {key: row[key] for key in ("bytes", "sha256")}
+                              for row in manifest["compile_inputs"]}, "actual exact compiler inputs")
+    for receipt in (requested, result):
+        require(receipt["compiler"] == compiler["path"]
+                and receipt["compiler_sha256"] == compiler["sha256"]
+                and digest(compiler["path"])["sha256"] == compiler["sha256"], "actual pinned compiler")
+        require(receipt["cwd"] == str(REPO)
+                and receipt["argv"] == [compiler["path"], *prepared["argv"][1:]]
+                and receipt["source_freeze_sha256"] == args.manifest_sha256
+                and receipt["source_inputs"] == actual_inputs
+                and receipt["runtime_or_math_qualified"] is False,
+                "actual relative-argv CWD/new CPP+included source pre/post build binding")
+    require(type(result["exit_code"]) is int and result["exit_code"] == 0, "actual successful fresh build")
+    binary = (HERE / "native_ref_control_v4").resolve(strict=True)
+    identity = digest(binary)
+    require(result["binary_path"] == str(binary)
+            and identity == {"bytes": result["binary_bytes"], "sha256": result["binary_sha256"]},
+            "exact actual new binary output")
+    invocation = jread(run / "invocation.json")
+    require(invocation["argv"][0] == str(binary) and invocation["inputs"][str(binary)] == identity,
+            "actual generic runner executed this newly compiled V4 binary")
+    require(invocation["freeze_manifest"] == digest(HERE / "freeze-manifest.json"),
+            "generic V1 helper receipt remains separate from V4 CPP build evidence")
+    native_log = run / "native.log"
+    require(native_log.is_file() and not native_log.is_symlink()
+            and 0 < native_log.stat().st_size == invocation["log_bytes"] <= 8 * 1024 * 1024,
+            "complete bounded original stdout/stderr diagnostic log")
+    log_identity = digest(native_log)
+    control_path = run / "capture/use-ref-control.json"
+    control = None
+    if control_path.exists():
+        actual = jread(control_path)
+        require(actual["control_source_revision"] == REVISION
+                and actual["provider_policy"] == "same_canonical_path_regular_device_inode_length_exact_export",
+                "actual V4 source marker with unchanged V2 attestation policy")
+        control = digest(control_path)
+    else:
+        require(invocation["status"] == "UNQUALIFIED", "successful run missing actual V4 control report")
+    trace_path = run / "capture/use-ref-public-trace.jsonl"
+    maps_path = run / "capture/use-ref-loaded-maps.txt"
+    trace = digest(trace_path) if trace_path.exists() else None
+    maps = digest(maps_path) if maps_path.exists() else None
+    require((trace is None or trace["bytes"] <= 8 * 1024**2)
+            and (maps is None or maps["bytes"] <= 2 * 1024**2), "bounded actual control trace/maps")
+    verify_sources(args.manifest_sha256)
+    verify_runtime_files()
+    write(destination, {
+        "status": "SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE",
+        "new_source_freeze": digest(path), "generic_V1_helper_freeze": digest(HERE / "freeze-manifest.json"),
+        "V2_attestation_freeze": digest(HERE / "freeze-manifest-v2.json"),
+        "V3_diagnostic_freeze": digest(HERE / "freeze-manifest-v3.json"),
+        "new_cpp": {"path": str(HERE / "native_ref_control_v4.cpp"), **digest(HERE / "native_ref_control_v4.cpp")},
+        "included_unchanged_token_loop": digest(HERE / "native_raw_original.cpp"),
+        "build_requested": digest(build / "requested.json"), "build_result": digest(build / "result.json"),
+        "actual_binary": {"path": str(binary), **identity}, "actual_invocation": digest(run / "invocation.json"),
+        "actual_diagnostic_log": {"path": str(native_log), **log_identity},
+        "actual_invocation_status": invocation["status"], "actual_control_report": control,
+        "actual_control_trace": trace, "actual_loaded_maps": maps,
+        "exact_BLAS_construction_retirement_only": True, "nonCPU_compute_delegation_permitted": False,
+        "runtime_or_math_qualified": False, "actual_extra_provider_and_static_scope": extra_providers,
+        "selected_static_route_pin": digest(HERE / "static-route-pins-v4.json"),
+        "next": "validate_control_v4.py then separate raw730/fullV/all42 comparators; any BLAS compute stays UNQUALIFIED",
+    })
+    print("SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE")
+
+
+if __name__ == "__main__":
+    main()
--- /dev/null
+++ b/.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/validate_control_v4.py
@@ -0,0 +1,148 @@
+#!/usr/bin/env python3
+"""ROOT ONLY public-policy evidence validation; arithmetic acceptance is separate."""
+import argparse
+import importlib.util
+import json
+from pathlib import Path
+import re
+from common import HERE, REPO, TEACHER_SHA, authenticate, bounded, digest, jread, require, write
+from trace_v4 import validate_trace
+from v4_sources import REVISION, verify_sources, verify_runtime_files
+
+
+def cpulist(text):
+    result = []
+    require(len(text) <= 128 and re.fullmatch(r'[0-9,-]+', text), 'bounded process CPU mask')
+    for item in text.split(','):
+        ends = list(map(int, item.split('-')))
+        require(len(ends) in [1, 2] and 0 <= ends[0] <= ends[-1] < 1024, 'process CPU range')
+        result.extend(range(ends[0], ends[-1] + 1))
+    require(len(result) == 8 and result == sorted(set(result)), 'exact CPU8 process mask')
+    return result
+
+
+def validate(output):
+    verify_sources()
+    extra_providers = verify_runtime_files()
+    invocation = jread(output / 'invocation.json')
+    require(invocation['status'] == 'PUBLIC_USE_REF_NATIVE_CAPTURE_CONTROL_VALIDATOR_REQUIRED'
+            and invocation['primary_error'] is None and invocation['cleanup_errors'] == []
+            and invocation['native_exit_code'] == invocation['native_reaped_exit_code'] == 0,
+            'actual owned control completed with clean reap')
+    require(invocation['freeze_manifest'] == digest(HERE / 'freeze-manifest.json'), 'executed reviewed source closure')
+    for key in ['leader_reaped', 'all_owned_groups_absent', 'all_nonzero_signals_before_reap',
+                'pipe_eof', 'selector_closed', 'pipe_closed']:
+        require(invocation['cleanup'][key] is True, 'complete owned lifecycle: ' + key)
+    require(invocation['benchmark'] is False and invocation['cut'] == 8192
+            and invocation['prefix'] in [64, 512], 'closed fresh bulk control scope')
+    cpus = invocation['CPU_affinity']
+    require(type(cpus) is list and len(cpus) == 8 and all(type(c) is int for c in cpus)
+            and cpus == sorted(set(cpus)) and 0 <= cpus[0] <= cpus[-1] < 1024, 'declared CPU8 list')
+    cases = authenticate()
+    case = cases['models'][invocation['model']]
+    model = str((REPO / case['model']).resolve(strict=True))
+    config = str((REPO / case['config']).resolve(strict=True))
+    teachers = str((REPO / cases['teachers']['path']).resolve(strict=True))
+    require(invocation['inputs'][model] == case['model_identity']
+            and invocation['inputs'][config] == case['config_identity']
+            and invocation['inputs'][teachers] == {'bytes': 32768, 'sha256': TEACHER_SHA}, 'exact original inputs')
+    for path, expected in invocation['inputs'].items():
+        require(digest(path) == expected, 'executed source/binary changed: ' + path)
+    capture = output / 'capture'
+    require(invocation['argv'] == [invocation['argv'][0], model, teachers, '8192',
+            str(invocation['prefix']), str(invocation['prefix']), '8', str(invocation['original_deadline_seconds']),
+            str(capture)], 'unaltered original native CLI and explicit policy wrapper')
+    require(Path(invocation['argv'][0]).is_relative_to(HERE)
+            and invocation['argv'][0] in invocation['inputs'], 'owned authenticated executable')
+    require(bounded(capture / 'input_ids.u32le', 32768) == bounded(teachers, 32768), 'all8192 authentic teacher words')
+    complete = jread(capture / 'complete.json')
+    loader_path = REPO / '.agents/runs/numerics-and-context-long-context-audit-20261020/compare_raw.py'
+    spec = importlib.util.spec_from_file_location('unchanged_native_capture_metadata', loader_path)
+    require(spec is not None and spec.loader is not None, 'pinned original capture validator')
+    loader = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(loader)  # Inert definitions only; no main/provider/operator execution.
+    expected = loader.expected_geometry(case['hidden'], 3584 if case['hidden'] == 1024 else 6144)
+    validated_complete, records = loader.load_capture(capture, 'pinned_native_llama_cpp', expected, 8192)
+    require(complete == validated_complete and len(records) == 730, 'unchanged730-stage full metadata/file coverage')
+    require(complete['status'] == 'CAPTURE_COMPLETE_NOT_COMPARISON'
+            and complete['provider'] == 'pinned_native_llama_cpp' and complete['cut'] == 8192
+            and complete['model_path'] == model and complete['hidden_width'] == case['hidden']
+            and complete['record_count'] == 730 and complete['actual_selected_token_GDN_ops'] == 18
+            and complete['actual_shared_model_tokens'] == 8193
+            and 8320 <= complete['n_ctx_actual'] <= 8704
+            and complete['instrumentation_full_logits_exact'] is True
+            and complete['instrumentation_whole_native_state_exact'] is True
+            and complete['F32_KV'] is True and complete['CPU_only'] is True
+            and complete['flash_attention'] is False, 'unchanged original full capture guards')
+    for field in ['prefix_batch_requested', 'ubatch_requested', 'n_batch_actual', 'n_ubatch_actual']:
+        require(complete[field] == invocation['prefix'], 'actual bulk geometry: ' + field)
+    require(complete['threads_actual'] == complete['threads_batch_actual'] == 8, 'original actual CPU8 contexts')
+    control = jread(capture / 'use-ref-control.json')
+    require(control['status'] == 'PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED'
+            and control['control_source_revision'] == REVISION
+            and control['provider_policy'] == 'same_canonical_path_regular_device_inode_length_exact_export'
+            and control['CPU_backend_count'] == 2 and control['BLAS_backend_count'] == 2
+            and control['public_setter_calls'] == 4
+            and control['existing_abort_setter_calls'] == 2
+            and control['BLAS_compute_delegations'] == 0
+            and control['public_inner_TLS_scope_exact'] is True
+            and control['BLAS_construction_retirement_only'] is True
+            and control['no_BLAS_disable_or_offload_change'] is True
+            and control['opaque_plan_public_association_checked'] is True
+            and control['actual_public_plan_use_ref_all_true'] is True
+            and control['backend_private_getter_available'] is False
+            and control['plan_to_backend_identity_publicly_available'] is False
+            and control['CPU_REPACK_policy_unchanged'] is True
+            and control['native_raw_token_loop_unchanged'] is True
+            and control['historical_prefix_or_cause_closed'] is False
+            and control['actual_CPU_affinity'] == cpus, 'honest public API closure')
+    observed = validate_trace(bounded(capture / 'use-ref-public-trace.jsonl', 8 * 1024**2), control, invocation['prefix'])
+    status = bounded(capture / 'use-ref-process-status.txt', 65536).decode()
+    masks = [line.split(':', 1)[1].strip() for line in status.splitlines() if line.startswith('Cpus_allowed_list:')]
+    require(len(masks) == 1 and cpulist(masks[0]) == cpus, 'actual process affinity readback')
+    maps = bounded(capture / 'use-ref-loaded-maps.txt', 2 * 1024**2).decode()
+    require(len(maps.splitlines()) <= 16384, 'bounded actual mapping records')
+    paths = {line.split(maxsplit=5)[5] for line in maps.splitlines() if len(line.split(maxsplit=5)) == 6}
+    providers = [row for row in jread(HERE / 'original-source-pins.json')['files'] if row['path'].startswith('/nix/store/')]
+    require(all(row['path'] in paths for row in providers), 'all original pinned providers actually mapped')
+    require(not any('libggml-cpu-' in path and not path.endswith('/libggml-cpu-haswell.so') for path in paths),
+            'no unobserved alternate CPU implementation')
+    blas_path = extra_providers['BLAS']['canonical']
+    require(blas_path in paths and not any('libggml-blas' in path and path != blas_path for path in paths),
+            'actual exact pinned BLAS mapped, no alternate BLAS provider')
+    binding = jread(output / 'source-build-binding-v4.json')
+    require(binding['status'] == 'SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE'
+            and binding['actual_invocation'] == digest(output / 'invocation.json')
+            and binding['new_source_freeze'] == digest(HERE / 'freeze-manifest-v4.json')
+            and binding['actual_control_report'] == digest(capture / 'use-ref-control.json')
+            and binding['actual_control_trace'] == digest(capture / 'use-ref-public-trace.jsonl')
+            and binding['actual_loaded_maps'] == digest(capture / 'use-ref-loaded-maps.txt')
+            and binding['actual_binary'] == dict(path=invocation['argv'][0], **invocation['inputs'][invocation['argv'][0]]),
+            'independent ROOT compile/actual V4 binary/generic runner binding required')
+    verify_sources()
+    verify_runtime_files()
+    return dict(status='PASS_PUBLIC_USE_REF_V4_CONTROL_CAPTURE_ONLY_COMPARISONS_REQUIRED',
+                source_model_sha256=case['model_identity']['sha256'], raw_teacher_sha256=TEACHER_SHA,
+                cut=8192, prefix_batch=invocation['prefix'], CPU_affinity=cpus, public_route=observed,
+                invocation=digest(output / 'invocation.json'), complete=digest(capture / 'complete.json'),
+                control=digest(capture / 'use-ref-control.json'),
+                extra_provider=extra_providers, source_build_binding=digest(output / 'source-build-binding-v4.json'),
+                scope='exact BLAS lifecycle with no BLAS math; CPU public-route/TLS/actual-use_ref policy and '
+                      'original selected-tail transparency only; selected static entry evidence is not universal runtime closure; '
+                      'no raw730/cross-provider all42 result or historical/whole-cause conclusion')
+
+
+def main():
+    p = argparse.ArgumentParser(description=__doc__)
+    p.add_argument('--capture-run', type=Path, required=True)
+    p.add_argument('--new-report', type=Path, required=True)
+    a = p.parse_args()
+    output = a.capture_run.resolve(strict=True)
+    require(output.is_relative_to(HERE) and output != HERE, 'own fresh control run only')
+    report = validate(output)
+    write(a.new_report, report)
+    print(report['status'])
+
+
+if __name__ == '__main__':
+    main()
--- /dev/null
+++ b/.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/test_control_v4.py
@@ -0,0 +1,171 @@
+"""ROOT ONLY prepared pure parser fixtures; author has not executed them."""
+import copy
+import json
+import unittest
+from trace_v4 import validate_trace
+
+
+def encode(rows):
+    return b''.join(json.dumps(row).encode() + b'\n' for row in rows)
+
+
+def fixture(plan=False, sync=False):
+    rows = []
+    for index, (name, kind) in enumerate([('0x10', 'BLAS'), ('0x20', 'CPU'),
+                                          ('0x30', 'BLAS'), ('0x40', 'CPU')], 1):
+        device, registry = ('0x500', '0x600') if kind == 'BLAS' else ('0x700', '0x800')
+        rows.append(dict(kind='public_backend_init', backend=name, id=index, backend_kind=kind,
+                         device=device, registry=registry, input_device=device, backend_name=kind, device_name=kind,
+                         registry_name=kind, device_type=0 if kind == 'CPU' else 3, cpu_is_cpu=kind == 'CPU',
+                         input_device_equals_returned=True, exact_provider=True))
+        if kind == 'CPU':
+            for origin in ['public_dev_init', 'existing_public_abort_setter_callback']:
+                rows.append(dict(kind='public_cpu_reference_setter', backend=name,
+                                 origin=origin, requested_use_ref=True))
+    if plan:
+        rows.append(dict(kind='public_plan_create', backend='0x20', opaque_plan='0x900', plan_id=1))
+    for sequence in range(1, 19):
+        owner = '0x20' if sequence < 18 else '0x40'
+        api = 'plan' if plan and sequence == 1 else 'async'
+        graph, opaque_plan, plan_id = ('0x0', '0x900', 1) if api == 'plan' else ('0xa00', '0x0', 0)
+        rows.append(dict(kind='public_compute_begin', backend=owner, sequence=sequence,
+                         api=api, graph=graph, opaque_plan=opaque_plan, plan_id=plan_id))
+        rows.append(dict(kind='actual_public_graph_plan', backend=owner, sequence=sequence,
+                         graph='0xa00', cplan='0xb00', use_ref_before=True, use_ref_after=True,
+                         n_threads=8, status=0))
+        rows.append(dict(kind='public_compute_return', backend=owner, sequence=sequence, status=0, inner_calls=1))
+        if sync and sequence == 2:
+            rows.append(dict(kind='public_sync_return', backend=owner, sequence=sequence, status=0))
+    if plan:
+        rows.append(dict(kind='public_plan_free', backend='0x20', opaque_plan='0x900', plan_id=1))
+    for owner, kind in [('0x30', 'BLAS'), ('0x40', 'CPU'), ('0x10', 'BLAS'), ('0x20', 'CPU')]:
+        rows.append(dict(kind='public_backend_free', backend=owner, backend_kind=kind))
+    report = dict(CPU_backend_count=2, BLAS_backend_count=2, public_setter_calls=4,
+                  existing_abort_setter_calls=2, BLAS_compute_delegations=0,
+                  actual_public_plan_calls=18, actual_successful_public_plan_calls=18,
+                  public_compute_calls=18, public_async_calls=18 - int(plan),
+                  public_opaque_plan_compute_calls=int(plan), public_sync_calls=int(sync),
+                  public_opaque_plan_creates=int(plan), public_opaque_plan_frees=int(plan),
+                  actual_plan_threads_min=8, actual_plan_threads_max=8, opaque_plan_route_observed=plan)
+    return rows, report
+
+
+def first(rows, kind):
+    return next(row for row in rows if row['kind'] == kind)
+
+
+class V4Routes(unittest.TestCase):
+    def refuse(self, rows, report):
+        with self.assertRaises(ValueError):
+            validate_trace(encode(rows), report, 512)
+
+    def test_async_complete_lifecycle_and_scope(self):
+        rows, report = fixture()
+        result = validate_trace(encode(rows), report, 512)
+        self.assertEqual(result['actual_blas_math_calls'], 0)
+        self.assertFalse(result['actual_opaque_plan_route_observed'])
+
+    def test_opaque_plan_and_sync_are_separate_from_math_count(self):
+        rows, report = fixture(plan=True, sync=True)
+        result = validate_trace(encode(rows), report, 512)
+        self.assertEqual(result['actual_plan_calls'], 18)
+        self.assertEqual(result['actual_opaque_plan_calls'], 1)
+        self.assertEqual(result['actual_public_sync_calls'], 1)
+
+    def test_any_blas_compute_or_setter_refuses(self):
+        for kind in ['public_compute_begin', 'public_cpu_reference_setter']:
+            rows, report = fixture()
+            first(rows, kind)['backend'] = '0x10'
+            self.refuse(rows, report)
+        rows, report = fixture()
+        rows.insert(8, dict(kind='nonCPU_compute_refused', backend='0x10', backend_kind='BLAS',
+                            api='async', delegated=False))
+        self.refuse(rows, report)
+
+    def test_missing_inner_missing_scope_and_duplicate_inner(self):
+        for operation in ['missing_inner', 'missing_scope', 'duplicate']:
+            rows, report = fixture()
+            index = next(i for i, row in enumerate(rows) if row['kind'] == 'actual_public_graph_plan')
+            if operation == 'missing_inner':
+                del rows[index]
+            elif operation == 'missing_scope':
+                del rows[index - 1]
+            else:
+                rows.insert(index, copy.deepcopy(rows[index]))
+            self.refuse(rows, report)
+
+    def test_pre_and_post_actual_reference_and_thread_status(self):
+        for field, value in [('use_ref_before', False), ('use_ref_after', False),
+                             ('n_threads', True), ('n_threads', 0), ('n_threads', 9), ('status', 1)]:
+            rows, report = fixture()
+            first(rows, 'actual_public_graph_plan')[field] = value
+            self.refuse(rows, report)
+
+    def test_wrong_backend_graph_sequence_and_plan_owner(self):
+        for field, value in [('backend', '0x40'), ('graph', '0xfff'), ('sequence', 2)]:
+            rows, report = fixture()
+            first(rows, 'actual_public_graph_plan')[field] = value
+            self.refuse(rows, report)
+        rows, report = fixture(plan=True)
+        first(rows, 'public_plan_create')['backend'] = '0x40'
+        self.refuse(rows, report)
+
+    def test_opaque_plan_lifetime_and_generation(self):
+        for operation in ['missing_free', 'wrong_free', 'duplicate_free', 'missing_create']:
+            rows, report = fixture(plan=True)
+            index = next(i for i, row in enumerate(rows) if row['kind'] == 'public_plan_free')
+            if operation == 'missing_free':
+                del rows[index]
+            elif operation == 'wrong_free':
+                rows[index]['plan_id'] = 2
+            elif operation == 'duplicate_free':
+                rows.insert(index, copy.deepcopy(rows[index]))
+            else:
+                del rows[next(i for i, row in enumerate(rows) if row['kind'] == 'public_plan_create')]
+            self.refuse(rows, report)
+
+    def test_original_lifecycle_and_nonvacuous_second_cpu(self):
+        for operation in ['extra', 'wrong_device', 'wrong_kind', 'missing_callback', 'no_second_cpu']:
+            rows, report = fixture()
+            if operation == 'extra':
+                rows.insert(0, copy.deepcopy(rows[0]))
+            elif operation == 'wrong_device':
+                rows[4]['device'] = '0x123'
+            elif operation == 'wrong_kind':
+                rows[-1]['backend_kind'] = 'BLAS'
+            elif operation == 'missing_callback':
+                del rows[3]
+            else:
+                for row in rows:
+                    if row.get('sequence') == 18:
+                        row['backend'] = '0x20'
+            self.refuse(rows, report)
+
+    def test_sync_duplicate_or_forged_math_count(self):
+        rows, report = fixture(sync=True)
+        at = next(i for i, row in enumerate(rows) if row['kind'] == 'public_sync_return')
+        rows.insert(at, copy.deepcopy(rows[at]))
+        self.refuse(rows, report)
+        rows, report = fixture(sync=True)
+        report['public_compute_calls'] += 1
+        self.refuse(rows, report)
+
+    def test_report_reconciliation_and_unobserved_plan_claim(self):
+        for field, value in [('BLAS_compute_delegations', 1), ('public_async_calls', 19),
+                             ('actual_successful_public_plan_calls', 17), ('public_sync_calls', True),
+                             ('opaque_plan_route_observed', True)]:
+            rows, report = fixture()
+            report[field] = value
+            self.refuse(rows, report)
+
+    def test_truncated_duplicate_key_and_unknown_fields(self):
+        rows, report = fixture()
+        for raw in [encode(rows)[:-1], encode(rows).replace(b'"id": 1', b'"id": 1, "id": 1', 1)]:
+            with self.assertRaises(ValueError):
+                validate_trace(raw, report, 512)
+        first(rows, 'public_compute_return')['extra'] = True
+        self.refuse(rows, report)
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ b/.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/static-route-pins-v4.json
@@ -0,0 +1,114 @@
+{
+  "status": "SOURCE_ONLY_SELECTED_COMPILED_SITE_METADATA_NOT_RUNTIME_CLOSURE",
+  "files": [
+    {
+      "path": "root-v4-elf-audit/base-async.txt",
+      "bytes": 1032,
+      "sha256": "c1dac9f48b901da50110bd1d2422d99a52d5c01411ba96d7f75faafbd3a3e3ed"
+    },
+    {
+      "path": "root-v4-elf-audit/base-dynamic.txt",
+      "bytes": 2088,
+      "sha256": "98e36b5d94893163fa9004743bb370ec84a7001bc202c785ccfe47492dabf15a"
+    },
+    {
+      "path": "root-v4-elf-audit/base-plan.txt",
+      "bytes": 1633,
+      "sha256": "6b465efd21940ce20e5cc05f9dafaa4b7539614ccc9050c0f5b7d7c10e33bc04"
+    },
+    {
+      "path": "root-v4-elf-audit/base-relocs.txt",
+      "bytes": 98396,
+      "sha256": "c22b631a19e0262c00f853a4064d2872d9fb60af419d83c272c1f3d37abf54ba"
+    },
+    {
+      "path": "root-v4-elf-audit/base-scheduler.txt",
+      "bytes": 48875,
+      "sha256": "84cf8c71ebdab4f8e44d7aa020a27ab1042695bca44e80f83599a79107d9aaef"
+    },
+    {
+      "path": "root-v4-elf-audit/base-symbols.txt",
+      "bytes": 87815,
+      "sha256": "1d05789fe7a112d40de01862852ff2da68001a4bb4fb91b4a72666dfe21e9743"
+    },
+    {
+      "path": "root-v4-elf-audit/base-sync.txt",
+      "bytes": 1021,
+      "sha256": "04d27a6c886cfbe92ddb871523246c3666954ba874ca3408f8b785ccf9fb33bb"
+    },
+    {
+      "path": "root-v4-elf-audit/base.json",
+      "bytes": 303,
+      "sha256": "4d1379403a344587065e9052cd735e5079b80dbd369340576bc336a7bc957017"
+    },
+    {
+      "path": "root-v4-elf-audit/cpu-dynamic.txt",
+      "bytes": 2151,
+      "sha256": "07c516070396534ea99381895a6f5a88e0289e28782c5d7302a6f4721c7629d3"
+    },
+    {
+      "path": "root-v4-elf-audit/cpu-entry.txt",
+      "bytes": 6151,
+      "sha256": "4f5ec710cdfd24e14441703305ceddc3c7bd3adb4f2641f3352aa246ce4211e5"
+    },
+    {
+      "path": "root-v4-elf-audit/cpu-relocs.txt",
+      "bytes": 94636,
+      "sha256": "ba31c85163badd29b876fd09114076a3afc457fab5451b35f7dbea7d553beb2b"
+    },
+    {
+      "path": "root-v4-elf-audit/cpu-symbols.txt",
+      "bytes": 97716,
+      "sha256": "5bdc932873b3114ba518d3f9b417b892238931015643e5ca2618f4fbd9dfc5d9"
+    },
+    {
+      "path": "root-v4-elf-audit/cpu.json",
+      "bytes": 309,
+      "sha256": "a74d7a5b0a1fe836ea31c48e72eed6c1eb76681fc3326fd403cc30fd076c2180"
+    },
+    {
+      "path": "root-blas-provider-pin.json",
+      "bytes": 578,
+      "sha256": "3f2bc7f3cca28452609afbd68c35616c3df6ce929fac0768a3862b6db4f2ae91"
+    }
+  ],
+  "primary_source_files": [
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include/ggml-blas.h",
+      "bytes": 586,
+      "sha256": "e76cb8594dc9866d99e7d63191b9fff6ab8b50244b4a28ce3af0f962964a6b5b"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend.cpp",
+      "bytes": 105561,
+      "sha256": "cdb79f2937aa147084a76f949adc0eb760ad1883e0ccc015bb4e0f599c89bd5f"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ggml-cpu.cpp",
+      "bytes": 25005,
+      "sha256": "ccf382d3fbf126b5ddb5850a0a970656fb24e00587da4003419439277c91cd01"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-blas/ggml-blas.cpp",
+      "bytes": 17633,
+      "sha256": "6dc134386348b01749e8b7d403ed63f8662ffe53cb526269191fec2c6c4db24b"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/src/llama-context.cpp",
+      "bytes": 160236,
+      "sha256": "904c41cb688900936db135159f37c1b9803bdb79ce04a25c326400078a82532f"
+    },
+    {
+      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-backend-dl.cpp",
+      "bytes": 1051,
+      "sha256": "acd76df8b83bcdef6807bc7b0520f57b379d61b4ff1027c04aa4d140d16ad07f"
+    }
+  ],
+  "actual_provider_metadata": [
+    "root-v4-elf-audit/base.json",
+    "root-v4-elf-audit/cpu.json",
+    "root-blas-provider-pin.json"
+  ],
+  "ROOT_must_hash_actual_providers_before_and_after": true,
+  "complete_all_call_or_runtime_coverage_qualified": false
+}
--- /dev/null
+++ b/.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/SOURCE_AUDIT_V4.md
@@ -0,0 +1,49 @@
+# V4: exact BLAS lifecycle; CPU-only `use_ref` computation
+
+- Owner: baseline_review. Parent: numerics-and-context.
+- Additive source preparation only; no author build/test/native/model/runtime.
+- V1/V2/V3, original token loop, generic runner, failures and source pins unchanged.
+- V3 actually identified BLAS (backend/device/registry, ACCEL=3, `is_cpu=false`, original input device equality). This proves the refusing backend identity; it does not prove BLAS unused.
+
+## Closed policy
+
+- Same public `ggml_backend_dev_init` forwarded once. Classify returned backend using exact BASE public metadata, original input-device equality, CPU `is_cpu` or BLAS `is_blas`, exact public provider registry singleton and registry device0. Require original names/type. Constructor trace records actual checked identities, names, type and predicate.
+- Same canonical-file/regular device+inode/length/exact-export V2 attestation; only `RTLD_LOCAL|RTLD_NOLOAD|RTLD_NOW`. BLAS adds actual ROOT pin: regular non-symlink64136B, dev38/inode2939302343, SHA `ab748c85dadb66ff128c1eda20a506bd6de94646cd06a7fbfd5b7a3881225e64`. ROOT hashes before/after; C++ checks immutable same-file identity and pinned inode/size before obtaining original public symbols. No new provider load or global promotion.
+- Exactly two original CPU and two BLAS contexts; original loop keeps both contexts alive simultaneously. CPU init and original abort callback each apply the same V1 public `use_ref=true` setter. BLAS receives neither setter nor changed arguments/policy. Record constructor and completed retirement separately. Both CPU owners must actually compute; BLAS owners must only construct/retire.
+- Public async, sync and opaque-plan compute wrappers reject every BLAS/unknown/retired target BEFORE original delegation. A real BLAS assignment causes exit125 with `nonCPU_compute_refused`, never a successful capture. Plan creation/free also require CPU: only BLAS construction/retirement is admitted.
+- CPU async/opaque-plan call creates one thread-local public owner scope. Actual public CPU `ggml_graph_compute` must occur once inside that scope, with `cplan.use_ref=true` before/after, unchanged1..8 threads and successful status. Async binds exact graph pointer. Opaque plans use actual public create/free owner+generation associations; no private plan/backend layout is read. CPU implementation's graph copy is deliberately not confused with original public graph pointer.
+- Sync must delegate once through same-backend/same-graph guarded async; its count is separate from math. Actual public/inner counts conserve globally and per CPU, with nonvacuous coverage. Missing/nested/duplicate routes refuse. Opaque-plan route is explicitly UNOBSERVED if count0; successful async evidence does not qualify it.
+- Callback arguments, graph pointers, buffers, offload/repacking policies and original compute remain unchanged. No disableBLAS, wrong-backend setter, CPU forcing, math fallback, signed-zero exception, tolerance change or alternate token loop.
+
+## Source and actual selected compiled routes
+
+Pinned d812350 source in original acquisition receipt:
+
+- `ggml-backend.cpp`: scheduler ordinary split and callback graph-view split call public `ggml_backend_graph_compute_async`; public sync calls async; public plan compute dispatches opaque backend callback. Device/registry getters are public metadata.
+- `ggml-cpu.cpp`: ordinary callback creates public cplan with context `use_ref`, then calls public inner compute; opaque create stores same policy and opaque compute calls same public inner compute. Opaque internals are source-audited only, not accessed by V4.
+- `ggml-blas.cpp`: BLAS ordinary compute invokes CBLAS directly, bypassing inner CPU compute. Thus old inner-only control was insufficient once BLAS exists. Device type ACCEL, names BLAS, actual public `is_blas`/registry functions available.
+- `llama-context.cpp`: constructs ACCEL providers in addition to CPU. `op_offload=false` does not forbid all scheduler assignment: backend/buffer compatibility and scheduler upgrade rules still apply. V4 preserves them and refuses if they actually compute.
+
+ROOT-produced `root-v4-elf-audit/` excerpts are pinned in `static-route-pins-v4.json`:
+
+- BASE `ggml_backend_sched_graph_compute_async`: actual selected calls at `3cb18` (callback graph view) and `3d12b` (ordinary split) -> public async PLT `18840`.
+- BASE sync: `375fc` -> same public async PLT. BASE async/opaque-plan entries dispatch original interface callback.
+- CPU ordinary `1f34a` CALL and opaque-plan `1f384` TAILJMP -> public inner PLT `15a40`.
+- Matching `R_X86_64_JUMP_SLOT`: BASE `0xdbb18` async; CPU `0x14de28` inner. Both selected providers lack dynamic SYMBOLIC.
+- Pinned loader source `ggml-backend-dl.cpp` uses `RTLD_NOW|RTLD_LOCAL`, without DEEPBIND. This source observation does not substitute for actual runtime routing.
+- These are selected-site static evidence for these exact compiled providers. They are **not** proof of every call in every DSO, complete dynamic binding, or runtime acceptance. ROOT must review them, compile actual exported wrappers and observe nonvacuous public-to-inner conservation. No private function offsets are called; no GOT patch, callback replacement, dispatch forcing or LD-preload arithmetic.
+
+## ROOT build / run / validation
+
+1. Independently review exact V4 manifest plus original freezes and selected compiled-route evidence. Do not execute old or newly broadened policy by inference.
+2. Fresh finite strict build from MAIN CWD using `build-argv-v4.json` with actual pinned g++ path from manifest. Record requested/result JSON as prior V3: `cwd`, `compiler`, `compiler_sha256`, exactargv, THIS `source_freeze_sha256`, full seven-entry `source_inputs` before/after, `runtime_or_math_qualified=false`; result adds exit_code and actual output absolute path/bytes/SHA. Preserve all old builds/binaries.
+3. Same unchanged `run_ref_control.py`, fresh output, actual new binary+SHA, `.8B` then2B, fresh8192 IDs, prefix/ubatch64 or512 and explicitCPU8/caps. This generic helper still binds its original V1 freeze; it is never relabeled as V4 source proof.
+4. `bind_v4.py --manifest-sha256 REVIEWED_V4_SHA --build ABS_NEW_BUILD --run ABS_NEW_CASE --new-binding ABS_NEW_CASE/source-build-binding-v4.json`. Records actual new source, seven inputs/build/output, invocation/native log, selected static scope and extra provider. Can bind a failure; does not convert it into PASS. Unknown or actual BLAS compute remains exit125/UNQUALIFIED, even if its construction was legitimate.
+5. ROOT-only source-free `python3 -m unittest test_control_v4.py` from this run; eleven prepared cases. Author AST only. Then `validate_control_v4.py --capture-run ABS_NEW_CASE --new-report ABS_NEW_REPORT` for a successful capture. New validator duplicates old capture/lifecycle/source guards, adds strict V4 trace reconciliation and required separate source/build binding, hashes actual original/extra providers before/after and verifies final exact BLAS mappings.
+6. A control PASS establishes policy/route evidence and original plain-vs-observed full-logit/whole-serialized-state transparency only. Separate authentic raw730, full vocabulary and all42 domain comparisons against Rust remain mandatory. Historical prefix unrecovered; no historical/whole-cause/performance conclusion.
+
+## Bounds
+
+- Same native12GiB address/CPU8/finite deadline, teacher8192, batch64/512, original state/capture caps, owned supervisor cleanup.
+- New maps: <=4 backends, <=64 simultaneous opaque plans, <=65536 total public plan generations; trace <=65536 records, <2048B per record, <=8MiB total. Record buffers/metadata are bounded; no learned matrices/copies/new native allocator.
+- No production engine/profile/serving change, no historical8K divergence closure, no inference support claim.
--- /dev/null
+++ b/.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/TODO_V4.md
@@ -0,0 +1,16 @@
+# baseline_review V4 working checklist
+
+- Parent: `.agents/plans/numerics-and-context/`; root sole execution/merge coordinator.
+- Scope: additive ignored public use_ref diagnostic/control only; no production changes.
+- [x] Read V3 actual BLAS identity and ROOT actual selected ELF excerpts.
+- [x] Preserve original14, V2/V3 freezes, original helper/runner/validators/tests and failed cases.
+- [x] Prepare exact pinned BLAS constructor/retirement policy; refuse every actual nonCPU compute before delegation.
+- [x] Prepare CPU public/TLS/inner pre/post use_ref conservation, opaque public plan owner/lifecycle and nonvacuous per-CPU checks.
+- [x] Prepare strict source-only parser fixtures and source/build/invocation/static-provider binding.
+- [x] AST parse new Python; small-source/excerpt hashes only. NO author tests/build/model/native/operator/GPU/runtime.
+- [ ] Independent ROOT/peer source review.
+- [ ] ROOT strict fresh C++ build and explicit new binary/source/actual build binding.
+- [ ] ROOT execute eleven pure parser fixtures.
+- [ ] ROOT fresh bounded actual model case; if any BLAS compute, preserve exit125/UNQUALIFIED, no policy forcing.
+- [ ] If actual capture succeeds, ROOT V4 validator, exact raw730/fullV/all42 comparison and explicit scope disposition for both models.
+- Unresolved: complete dynamic route closure until actual observation; opaque-plan runtime eligibility if unused; historical prefix and full historical8K cause. No inference/performance claim.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/PROVIDER_ATTESTATION_V2.patch

Original UTF-8 bytes: 7249; SHA256 `365ae456714c2cacd2196078304f09d5d4627d8258a3ea85435c6b8abcb58198`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- original-immutable/native_ref_control.cpp
+++ additive-new/native_ref_control_v2.cpp
@@ -6,6 +6,7 @@
 #include <dlfcn.h>
 #include <algorithm>
 #include <fcntl.h>
+#include <limits.h>
 #include <sched.h>
 #include <sys/stat.h>
 #include <unistd.h>
@@ -46,11 +47,24 @@
     const int saved=errno;
     const char prefix[]="USE_REF_CONTROL_UNQUALIFIED ";
     const ssize_t a=write(STDERR_FILENO,prefix,sizeof(prefix)-1);
-    const ssize_t b=write(STDERR_FILENO,message,strnlen(message,512));
+    const ssize_t b=write(STDERR_FILENO,message,strnlen(message,4096));
     const ssize_t c=write(STDERR_FILENO,"\n",1);
     (void)a;(void)b;(void)c;errno=saved;_exit(125);
 }
 static void checked(bool value,const char *message) {if(!value)fatal(message);}
+[[noreturn]] static void provider_failure(const char *stage,const char *name,
+        const char *expected,const char *actual,const char *canonical_expected,
+        const char *canonical_actual,const char *error,const char *actual_symbol=nullptr) {
+    const int saved=errno;char message[4096]={};
+    const int size=std::snprintf(message,sizeof(message),
+        "provider_attestation_v2 stage=%.128s symbol=%.192s expected=%.512s actual=%.512s "
+        "canonical_expected=%.512s canonical_actual=%.512s actual_symbol=%.192s dlerror=%.512s errno=%d",
+        stage?stage:"(null)",name?name:"(null)",expected?expected:"(null)",actual?actual:"(null)",
+        canonical_expected?canonical_expected:"(unresolved)",canonical_actual?canonical_actual:"(unresolved)",
+        actual_symbol?actual_symbol:"(unavailable)",error?error:"(none)",saved);
+    if(size<0)fatal("provider_attestation_v2 diagnostic formatting failed");
+    errno=saved;fatal(message);
+}
 static void check_affinity() {
     cpu_set_t actual;CPU_ZERO(&actual);
     checked(sched_getaffinity(0,sizeof(actual),&actual)==0
@@ -80,20 +94,31 @@
 }
 static void attest(void *address,const char *path,size_t length,const char *name) {
     Dl_info info{};struct stat actual{},expected{};
-    checked(address && dladdr(address,&info)!=0 && info.dli_fname,"public symbol lookup");
-    checked(std::strcmp(info.dli_fname,path)==0,"public symbol exact loaded provider path");
-    checked(stat(path,&expected)==0 && stat(info.dli_fname,&actual)==0
-        && expected.st_dev==actual.st_dev && expected.st_ino==actual.st_ino
-        && expected.st_size==static_cast<off_t>(length),"public symbol provider inode/length");
-    checked(info.dli_sname && std::strcmp(info.dli_sname,name)==0,"public symbol exact exported name");
+    if(!address || dladdr(address,&info)==0 || !info.dli_fname)
+        provider_failure("dladdr",name,path,info.dli_fname,nullptr,nullptr,dlerror());
+    char expected_real[PATH_MAX]={},actual_real[PATH_MAX]={};
+    if(!realpath(path,expected_real))
+        provider_failure("canonical_expected",name,path,info.dli_fname,nullptr,nullptr,nullptr);
+    if(!realpath(info.dli_fname,actual_real))
+        provider_failure("canonical_actual",name,path,info.dli_fname,expected_real,nullptr,nullptr);
+    if(std::strcmp(actual_real,expected_real)!=0)
+        provider_failure("canonical_identity",name,path,info.dli_fname,expected_real,actual_real,nullptr);
+    if(stat(path,&expected)!=0 || stat(info.dli_fname,&actual)!=0
+        || !S_ISREG(expected.st_mode) || !S_ISREG(actual.st_mode)
+        || expected.st_dev!=actual.st_dev || expected.st_ino!=actual.st_ino
+        || expected.st_size!=static_cast<off_t>(length) || actual.st_size!=static_cast<off_t>(length))
+        provider_failure("regular_inode_length",name,path,info.dli_fname,expected_real,actual_real,nullptr);
+    if(!info.dli_sname || std::strcmp(info.dli_sname,name)!=0)
+        provider_failure("exact_exported_symbol",name,path,info.dli_fname,expected_real,actual_real,nullptr,info.dli_sname);
 }
 template<class F> static F symbol(const char *path,size_t bytes,const char *name) {
     // NOLOAD never acquires/replaces an unregistered CPU implementation. Hold
     // the original module through backend_free and the final observation.
-    void *handle=dlopen(path,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
-    checked(handle!=nullptr,"exact original provider is not already loaded");
+    dlerror();void *handle=dlopen(path,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
+    if(!handle)provider_failure("dlopen_noload",name,path,nullptr,nullptr,nullptr,dlerror());
     dlerror();void *address=dlsym(handle,name);const char *error=dlerror();
-    checked(error==nullptr,"public symbol resolution failed");attest(address,path,bytes,name);
+    if(error)provider_failure("dlsym",name,path,nullptr,nullptr,nullptr,error);
+    attest(address,path,bytes,name);
     return reinterpret_cast<F>(address);
 }
 static Lookup lookup() {static Lookup f=symbol<Lookup>(BASE,BASE_BYTES,"ggml_backend_reg_get_proc_address");return f;}
@@ -233,7 +258,9 @@
         require(status.eof() && status_count>0 && status_count<=64*1024,"bounded actual process status");
         write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
         std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
-        report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\",\"CPU_backend_count\":2,"
+        report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
+            <<"\"control_source_revision\":\"public_use_ref_canonical_provider_attestation_v2\","
+            <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\",\"CPU_backend_count\":2,"
             <<"\"public_setter_calls\":4,\"existing_abort_setter_calls\":2,\"actual_public_plan_calls\":"<<ref_control::graph_calls
             <<",\"actual_successful_public_plan_calls\":"<<ref_control::graph_success
             <<",\"actual_plan_threads_min\":"<<ref_control::thread_min<<",\"actual_plan_threads_max\":"<<ref_control::thread_max
--- original-immutable/build-argv.json
+++ additive-new/build-argv-v2.json
@@ -11,7 +11,7 @@
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include",
     "-I",
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control.cpp",
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v2.cpp",
     "-L",
     "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib",
     "-L",
@@ -29,8 +29,10 @@
     "-ldl",
     "-pthread",
     "-o",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control"
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v2"
   ],
   "compiler_identity": "ROOT must record real compiler and own finite build receipt before execution",
-  "native_math_source_unchanged": true
+  "native_math_source_unchanged": true,
+  "source_revision": "public_use_ref_canonical_provider_attestation_v2",
+  "generic_runner_v1_freeze_scope": "unchanged generic launcher/helper sources only; not proof that v1 CPP was compiled into the v2 binary"
 }

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/REFUSAL_GRAPH_DIAGNOSTIC_V7.patch

Original UTF-8 bytes: 6780; SHA256 `ab616fc21eb874dc12baf3425226ed841193ea23ceafdc14f231cf9b79317034`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/native_ref_control_v6.cpp
+++ b/native_ref_control_v7.cpp
@@ -222,11 +222,60 @@
     errno=saved;reinterpret_cast<ggml_backend_set_abort_callback_t>(raw)(backend,callback,data);
     const int after=errno;set_reference(backend,false);errno=after;
 }
-static void cpu_compute_only(ggml_backend_t backend,const char *api) {
+// V7 diagnostic only: actual public graph arguments/public tensor headers.
+// No payload, backend/plan private fields, graph edits or delegation.
+static std::string quoted(const char *text,size_t cap) {
+    std::string result="\"";
+    if(text)for(size_t i=0;i<cap && text[i]!='\0';++i) {
+        const auto value=static_cast<unsigned char>(text[i]);
+        if(value=='"' || value=='\\') {result+='\\';result+=static_cast<char>(value);}
+        else if(value<32 || value>=127) {
+            char escaped[7]={};
+            const int size=std::snprintf(escaped,sizeof(escaped),"\\u%04x",static_cast<unsigned>(value));
+            checked(size==6,"bounded diagnostic JSON escape");result+=escaped;
+        }else result+=static_cast<char>(value);
+    }
+    result+='"';return result;
+}
+static std::string tensor_metadata(const ggml_tensor *tensor) {
+    if(!tensor)return "\"tensor\":\"0x0\",\"available\":false";
+    static auto type_name=symbol<decltype(&ggml_type_name)>(BASE,BASE_BYTES,"ggml_type_name");
+    static auto op_name=symbol<decltype(&ggml_op_name)>(BASE,BASE_BYTES,"ggml_op_name");
+    const int type=static_cast<int>(tensor->type),op=static_cast<int>(tensor->op);
+    std::string result="\"tensor\":\""+pointer(tensor)+"\",\"available\":true,\"name\":"
+        +quoted(tensor->name,GGML_MAX_NAME)+",\"type_id\":"+std::to_string(type)
+        +",\"type\":"+quoted(type>=0 && type<GGML_TYPE_COUNT?type_name(tensor->type):"INVALID",64)
+        +",\"op_id\":"+std::to_string(op)
+        +",\"op\":"+quoted(op>=0 && op<GGML_OP_COUNT?op_name(tensor->op):"INVALID",64)+",\"ne\":[";
+    for(int d=0;d<GGML_MAX_DIMS;++d)result+=(d?",":"")+std::to_string(tensor->ne[d]);
+    return result+"]";
+}
+static void refused_graph_diagnostic(ggml_backend_t backend,const char *api,ggml_cgraph *graph) {
+    static auto count=symbol<decltype(&ggml_graph_n_nodes)>(BASE,BASE_BYTES,"ggml_graph_n_nodes");
+    static auto node=symbol<decltype(&ggml_graph_node)>(BASE,BASE_BYTES,"ggml_graph_node");
+    constexpr int cap=128;
+    const int actual=graph?count(graph):0;
+    const int retained=std::max(0,std::min(actual,cap));
+    append("{\"kind\":\"V7_nonCPU_graph_diagnostic\",\"backend\":\""+pointer(backend)
+        +"\",\"api\":"+quoted(api,32)+",\"graph\":\""+pointer(graph)+"\",\"public_graph_available\":"
+        +(graph?"true":"false")+",\"actual_node_count\":"+std::to_string(actual)
+        +",\"retained_nodes\":"+std::to_string(retained)+",\"node_cap\":128,\"metadata_only\":true,"
+        "\"payload_read\":false,\"original_delegated\":false,\"policy_continued\":false}");
+    for(int i=0;i<retained;++i) {
+        const auto *current=node(graph,i);
+        append("{\"kind\":\"V7_nonCPU_graph_node\",\"node_index\":"+std::to_string(i)+","+tensor_metadata(current)+"}");
+        if(!current)continue;
+        for(int input=0;input<GGML_MAX_SRC;++input)if(current->src[input])
+            append("{\"kind\":\"V7_nonCPU_graph_input\",\"node_index\":"+std::to_string(i)
+                +",\"input_slot\":"+std::to_string(input)+","+tensor_metadata(current->src[input])+"}");
+    }
+}
+static void cpu_compute_only(ggml_backend_t backend,const char *api,ggml_cgraph *graph=nullptr) {
     auto &record=known(backend);
     if(record.kind!=Kind::Cpu) {
         append("{\"kind\":\"nonCPU_compute_refused\",\"backend\":\""+pointer(backend)
             +"\",\"backend_kind\":\"BLAS\",\"api\":\""+api+"\",\"delegated\":false}");
+        refused_graph_diagnostic(backend,api,graph);
         fatal("V4 pinned BLAS actual compute refused BEFORE delegation; capture UNQUALIFIED");
     }
     checked(record.set==2 && record.abort==1,"actual CPU configured before public compute");
@@ -234,7 +283,7 @@
 static void begin(Scope &current,ggml_backend_t backend,ggml_cgraph *graph,
         ggml_backend_graph_plan_t plan,const char *api) {
     check_affinity();checked(scope==nullptr && (graph || plan),"one nonnested public CPU compute scope");
-    std::lock_guard<std::mutex> guard(mutex);cpu_compute_only(backend,api);
+    std::lock_guard<std::mutex> guard(mutex);cpu_compute_only(backend,api,graph);
     checked(inflight==0,"one globally serialized original public compute");auto &record=known(backend);
     size_t plan_id=0;
     if(plan) {auto found=plans.find(plan);checked(found!=plans.end() && found->second.backend==backend,
@@ -304,7 +353,7 @@
     if(ref_control::armed) {
         ref_control::checked(ref_control::sync_scope==nullptr && ref_control::scope==nullptr && graph,
             "nonnested public sync route");
-        {std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::cpu_compute_only(backend,"sync");}
+        {std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::cpu_compute_only(backend,"sync",graph);}
         current.backend=backend;current.graph=graph;ref_control::sync_scope=&current;
     }
     errno=saved;const auto status=original(backend,graph);const int after=errno;
@@ -323,7 +372,7 @@
     static auto original=ref_control::symbol<decltype(&ggml_backend_graph_plan_create)>(ref_control::BASE,
         ref_control::BASE_BYTES,"ggml_backend_graph_plan_create");
     if(ref_control::armed) {
-        std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::cpu_compute_only(backend,"plan_create");
+        std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::cpu_compute_only(backend,"plan_create",graph);
         ref_control::checked(graph && ref_control::scope==nullptr && ref_control::plans.size()<64,
             "bounded nonnested public opaque plan creation");
     }
@@ -488,7 +537,7 @@
         write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
         std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
         report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
-            <<"\"control_source_revision\":\"public_use_ref_exact_blas_cpu_compute_legal_null_free_v6\","
+            <<"\"control_source_revision\":\"public_use_ref_exact_blas_cpu_compute_refusal_graph_diagnostic_v7\","
             <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\","
             <<"\"CPU_backend_count\":2,\"BLAS_backend_count\":2,\"public_setter_calls\":4,"
             <<"\"existing_abort_setter_calls\":2,\"BLAS_compute_delegations\":0,"

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/REVIEW_DIFFS_V4.patch

Original UTF-8 bytes: 50955; SHA256 `820eb10c2fe653f169201fb98a825d573b3cc35c3b6db52219780e288287d98c`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- native_ref_control_v3.cpp
+++ native_ref_control_v4.cpp
@@ -1,8 +1,9 @@
-// Diagnostic-only public CPU reference-policy control. Original token loop included unchanged.
+// Qualifier-only public CPU use_ref control; exact pinned BLAS may construct/retire, never compute.
 #ifndef _GNU_SOURCE
 #define _GNU_SOURCE
 #endif
 #include "ggml-cpu.h"
+#include "ggml-blas.h"
 #include <dlfcn.h>
 #include <algorithm>
 #include <fcntl.h>
@@ -24,10 +25,26 @@
 namespace ref_control {
 constexpr const char *BASE="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib/libggml-base.so.0.26.0";
 constexpr const char *CPU="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-cpu-haswell.so";
-constexpr size_t BASE_BYTES=972584, CPU_BYTES=1459208, MAX_RECORDS=65536;
+constexpr const char *BLAS="/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/bin/libggml-blas.so";
+constexpr size_t BASE_BYTES=972584, CPU_BYTES=1459208, BLAS_BYTES=64136, MAX_RECORDS=65536;
 static bool armed=false;
-struct Backend {size_t init=0,abort=0,set=0;bool freed=false;};
+enum class Kind { Cpu, Blas };
+struct Backend {
+    Kind kind=Kind::Cpu;size_t id=0,abort=0,set=0,async=0,plans=0,inner=0,sync=0;
+    ggml_backend_dev_t device=nullptr;ggml_backend_reg_t registry=nullptr;bool freed=false;
+};
+struct Plan {ggml_backend_t backend=nullptr;size_t id=0;};
 static std::map<ggml_backend_t,Backend> backends;
+static std::map<ggml_backend_graph_plan_t,Plan> plans;
+static size_t next_plan=0,public_calls=0,public_success=0,async_calls=0,plan_calls=0,sync_calls=0;
+static size_t plan_creates=0,plan_frees=0,inflight=0;
+struct Scope {
+    ggml_backend_t backend=nullptr;ggml_cgraph *graph=nullptr;ggml_backend_graph_plan_t plan=nullptr;
+    size_t sequence=0,inner=0;const char *api=nullptr;bool in_inner=false;
+};
+static thread_local Scope *scope=nullptr;
+struct SyncScope {ggml_backend_t backend=nullptr;ggml_cgraph *graph=nullptr;size_t async=0,sequence=0;};
+static thread_local SyncScope *sync_scope=nullptr;
 static std::mutex mutex;
 static size_t trace_records=0,trace_bytes=0;
 static int trace_fd=-1;
@@ -87,6 +104,7 @@
     ++trace_records;trace_bytes+=record.size();
 }
 static std::string pointer(const void *value) {
+    if(!value)return "0x0";
     char text[2+sizeof(void*)*2+1]={};
     const int size=std::snprintf(text,sizeof(text),"%p",value);
     checked(size>0 && size<static_cast<int>(sizeof(text)),"opaque identity formatting");
@@ -108,6 +126,8 @@
         || expected.st_dev!=actual.st_dev || expected.st_ino!=actual.st_ino
         || expected.st_size!=static_cast<off_t>(length) || actual.st_size!=static_cast<off_t>(length))
         provider_failure("regular_inode_length",name,path,info.dli_fname,expected_real,actual_real,nullptr);
+    if(std::strcmp(path,BLAS)==0 && (expected.st_dev!=38 || expected.st_ino!=2939302343ULL))
+        provider_failure("pinned_blas_device_inode",name,path,info.dli_fname,expected_real,actual_real,nullptr);
     if(!info.dli_sname || std::strcmp(info.dli_sname,name)!=0)
         provider_failure("exact_exported_symbol",name,path,info.dli_fname,expected_real,actual_real,nullptr,info.dli_sname);
 }
@@ -122,71 +142,136 @@
     return reinterpret_cast<F>(address);
 }
 static Lookup lookup() {static Lookup f=symbol<Lookup>(BASE,BASE_BYTES,"ggml_backend_reg_get_proc_address");return f;}
-static ggml_backend_reg_t registry(ggml_backend_t backend) {
-    auto get_device=symbol<decltype(&ggml_backend_get_device)>(BASE,BASE_BYTES,"ggml_backend_get_device");
-    auto get_reg=symbol<decltype(&ggml_backend_dev_backend_reg)>(BASE,BASE_BYTES,"ggml_backend_dev_backend_reg");
-    const auto device=get_device(backend);checked(device!=nullptr,"actual backend has public device");
-    const auto reg=get_reg(device);checked(reg!=nullptr,"actual backend has public registry");return reg;
-}
-// Public metadata only, reached solely at the original pre-setter refusal.
-[[noreturn]] static void backend_identity_refused(ggml_backend_t backend,bool initialization,
-        ggml_backend_dev_t input_device,bool actual_cpu) {
-    const int saved=errno;
-    static auto backend_name=symbol<decltype(&ggml_backend_name)>(BASE,BASE_BYTES,"ggml_backend_name");
+static const char *kind_name(Kind kind) {return kind==Kind::Cpu?"CPU":"BLAS";}
+static Backend &known(ggml_backend_t backend) {
+    auto found=backends.find(backend);
+    checked(backend && found!=backends.end() && !found->second.freed,"unknown or retired public backend");
+    return found->second; // Caller holds mutex; no private backend dereference.
+}
+struct Identity {Kind kind;ggml_backend_dev_t device;ggml_backend_reg_t registry;};
+static Identity identify(ggml_backend_t backend,ggml_backend_dev_t input) {
+    static auto is_cpu=symbol<decltype(&ggml_backend_is_cpu)>(CPU,CPU_BYTES,"ggml_backend_is_cpu");
     static auto get_device=symbol<decltype(&ggml_backend_get_device)>(BASE,BASE_BYTES,"ggml_backend_get_device");
-    static auto device_name=symbol<decltype(&ggml_backend_dev_name)>(BASE,BASE_BYTES,"ggml_backend_dev_name");
-    static auto device_type=symbol<decltype(&ggml_backend_dev_type)>(BASE,BASE_BYTES,"ggml_backend_dev_type");
-    static auto device_reg=symbol<decltype(&ggml_backend_dev_backend_reg)>(BASE,BASE_BYTES,"ggml_backend_dev_backend_reg");
+    static auto get_reg=symbol<decltype(&ggml_backend_dev_backend_reg)>(BASE,BASE_BYTES,"ggml_backend_dev_backend_reg");
+    static auto get_name=symbol<decltype(&ggml_backend_name)>(BASE,BASE_BYTES,"ggml_backend_name");
+    static auto dev_name=symbol<decltype(&ggml_backend_dev_name)>(BASE,BASE_BYTES,"ggml_backend_dev_name");
     static auto reg_name=symbol<decltype(&ggml_backend_reg_name)>(BASE,BASE_BYTES,"ggml_backend_reg_name");
-    const auto device=backend?get_device(backend):nullptr;
-    const auto reg=device?device_reg(device):nullptr;
-    const char *bn=backend?backend_name(backend):"(null)";
-    const char *dn=device?device_name(device):"(null)";
-    const char *rn=reg?reg_name(reg):"(null)";
-    const int type=device?static_cast<int>(device_type(device)):-1;
-    const bool input_known=initialization && input_device;
-    char message[4096]={};
-    const int size=std::snprintf(message,sizeof(message),
-        "backend_identity_diagnostic_v3 origin=%.64s cpu_is_cpu=%s backend=%p backend_name=%.128s "
-        "device=%p device_name=%.128s device_type=%d registry=%p registry_name=%.128s "
-        "input_device=%p input_device_known=%s input_device_equals_returned=%s "
-        "setter_called_for_this_attempt=false compute_continued_after_refusal=false errno=%d",
-        initialization?"public_dev_init":"existing_public_abort_setter_callback",actual_cpu?"true":"false",
-        static_cast<void *>(backend),bn?bn:"(null)",static_cast<void *>(device),dn?dn:"(null)",type,
-        static_cast<void *>(reg),rn?rn:"(null)",static_cast<void *>(input_device),input_known?"true":"false",
-        input_known && input_device==device?"true":"false",saved);
-    if(size<0 || size>=static_cast<int>(sizeof(message)))fatal("backend_identity_diagnostic_v3 formatting refused");
-    errno=saved;fatal(message); // SAME exit125; never skip/set/disable or continue computation.
-}
-static void set_reference(ggml_backend_t backend,bool initialization,ggml_backend_dev_t input_device=nullptr) {
-    static IsCpu is_cpu=symbol<IsCpu>(CPU,CPU_BYTES,"ggml_backend_is_cpu");
-    const bool actual_cpu=backend && is_cpu(backend);
-    if(!actual_cpu)backend_identity_refused(backend,initialization,input_device,actual_cpu);
-    void *raw=lookup()(registry(backend),"ggml_backend_cpu_set_use_ref");
+    static auto dev_type=symbol<decltype(&ggml_backend_dev_type)>(BASE,BASE_BYTES,"ggml_backend_dev_type");
+    static auto reg_dev=symbol<decltype(&ggml_backend_reg_dev_get)>(BASE,BASE_BYTES,"ggml_backend_reg_dev_get");
+    checked(backend && input,"nonnull original initialized backend/device");
+    const auto device=get_device(backend);const auto registry=device?get_reg(device):nullptr;
+    checked(device && registry && device==input,"original input device equals public returned device");
+    Kind kind;ggml_backend_reg_t expected=nullptr;
+    const bool cpu=is_cpu(backend);
+    if(cpu) {
+        static auto reg=symbol<decltype(&ggml_backend_cpu_reg)>(CPU,CPU_BYTES,"ggml_backend_cpu_reg");
+        kind=Kind::Cpu;expected=reg();
+    } else {
+        // NOLOAD after actual construction only. Never load/force/disable a provider.
+        static auto is_blas=symbol<decltype(&ggml_backend_is_blas)>(BLAS,BLAS_BYTES,"ggml_backend_is_blas");
+        static auto reg=symbol<decltype(&ggml_backend_blas_reg)>(BLAS,BLAS_BYTES,"ggml_backend_blas_reg");
+        checked(is_blas(backend),"nonCPU backend is not exact pinned public BLAS");
+        kind=Kind::Blas;expected=reg();
+    }
+    const char *name=kind_name(kind);
+    const char *bn=get_name(backend),*dn=dev_name(device),*rn=reg_name(registry);
+    checked(registry==expected && reg_dev(registry,0)==device && bn && dn && rn
+        && std::strcmp(bn,name)==0 && std::strcmp(dn,name)==0 && std::strcmp(rn,name)==0
+        && dev_type(device)==(cpu?GGML_BACKEND_DEVICE_TYPE_CPU:GGML_BACKEND_DEVICE_TYPE_ACCEL),
+        "exact pinned public backend/device/registry singleton, names, type");
+    return {kind,device,registry};
+}
+static void initialized(ggml_backend_t backend,ggml_backend_dev_t input) {
+    const auto identity=identify(backend,input);
+    std::lock_guard<std::mutex> guard(mutex);
+    checked(backends.size()<4 && backends.find(backend)==backends.end(),"four distinct original context backends");
+    Backend record;record.id=backends.size()+1;record.kind=identity.kind;
+    record.device=identity.device;record.registry=identity.registry;backends.emplace(backend,record);
+    append("{\"kind\":\"public_backend_init\",\"backend\":\""+pointer(backend)+"\",\"id\":"+std::to_string(record.id)
+        +",\"backend_kind\":\""+kind_name(record.kind)+"\",\"device\":\""+pointer(identity.device)
+        +"\",\"registry\":\""+pointer(identity.registry)+"\",\"input_device\":\""+pointer(input)
+        +"\",\"backend_name\":\""+kind_name(record.kind)+"\",\"device_name\":\""+kind_name(record.kind)
+        +"\",\"registry_name\":\""+kind_name(record.kind)+"\",\"device_type\":"+(record.kind==Kind::Cpu?"0":"3")
+        +",\"cpu_is_cpu\":"+(record.kind==Kind::Cpu?"true":"false")
+        +",\"input_device_equals_returned\":true,\"exact_provider\":true}");
+}
+static void set_reference(ggml_backend_t backend,bool initialization) {
+    ggml_backend_reg_t reg=nullptr;
+    {
+        std::lock_guard<std::mutex> guard(mutex);auto &record=known(backend);
+        checked(record.kind==Kind::Cpu && record.set==(initialization?0:1)
+            && record.abort==0,"reference setter only exact CPU, original init/abort order");
+        reg=record.registry;
+    }
+    void *raw=lookup()(reg,"ggml_backend_cpu_set_use_ref");
     attest(raw,CPU,CPU_BYTES,"ggml_backend_cpu_set_use_ref");
-    reinterpret_cast<Ref>(raw)(backend,true); // Sole intentional policy change.
-    std::lock_guard<std::mutex> guard(mutex);
-    auto &record=backends[backend];checked(!record.freed && backends.size()<=2,"exact two live context CPU backends");
-    (initialization?record.init:record.abort)++;record.set++;
+    reinterpret_cast<Ref>(raw)(backend,true); // Sole numerical policy change, same original V1 CPU setter.
+    std::lock_guard<std::mutex> guard(mutex);auto &record=known(backend);
+    if(!initialization)++record.abort;
+    ++record.set;
     append("{\"kind\":\"public_cpu_reference_setter\",\"backend\":\""+pointer(backend)
         +"\",\"origin\":\""+(initialization?"public_dev_init":"existing_public_abort_setter_callback")
         +"\",\"requested_use_ref\":true}");
 }
 static void abort_setter(ggml_backend_t backend,ggml_abort_callback callback,void *data) {
-    const int saved=errno;
-    void *raw=lookup()(registry(backend),"ggml_backend_set_abort_callback");
+    const int saved=errno;ggml_backend_reg_t reg=nullptr;
+    {std::lock_guard<std::mutex> guard(mutex);auto &record=known(backend);
+        checked(record.kind==Kind::Cpu,"abort setter cannot target BLAS");reg=record.registry;}
+    void *raw=lookup()(reg,"ggml_backend_set_abort_callback");
     attest(raw,CPU,CPU_BYTES,"ggml_backend_cpu_set_abort_callback");
     errno=saved;reinterpret_cast<ggml_backend_set_abort_callback_t>(raw)(backend,callback,data);
     const int after=errno;set_reference(backend,false);errno=after;
 }
+static void cpu_compute_only(ggml_backend_t backend,const char *api) {
+    auto &record=known(backend);
+    if(record.kind!=Kind::Cpu) {
+        append("{\"kind\":\"nonCPU_compute_refused\",\"backend\":\""+pointer(backend)
+            +"\",\"backend_kind\":\"BLAS\",\"api\":\""+api+"\",\"delegated\":false}");
+        fatal("V4 pinned BLAS actual compute refused BEFORE delegation; capture UNQUALIFIED");
+    }
+    checked(record.set==2 && record.abort==1,"actual CPU configured before public compute");
+}
+static void begin(Scope &current,ggml_backend_t backend,ggml_cgraph *graph,
+        ggml_backend_graph_plan_t plan,const char *api) {
+    check_affinity();checked(scope==nullptr && (graph || plan),"one nonnested public CPU compute scope");
+    std::lock_guard<std::mutex> guard(mutex);cpu_compute_only(backend,api);
+    checked(inflight==0,"one globally serialized original public compute");auto &record=known(backend);
+    size_t plan_id=0;
+    if(plan) {auto found=plans.find(plan);checked(found!=plans.end() && found->second.backend==backend,
+        "opaque public plan belongs to same live CPU backend");plan_id=found->second.id;}
+    if(sync_scope) {checked(!plan && sync_scope->backend==backend && sync_scope->graph==graph
+        && sync_scope->async==0,"sync delegates once to same public async graph");++sync_scope->async;}
+    current.backend=backend;current.graph=graph;current.plan=plan;current.api=api;current.sequence=++public_calls;
+    if(plan){++plan_calls;++record.plans;}else{++async_calls;++record.async;}
+    if(sync_scope)sync_scope->sequence=current.sequence;
+    scope=&current;inflight=current.sequence;
+    append("{\"kind\":\"public_compute_begin\",\"backend\":\""+pointer(backend)+"\",\"sequence\":"
+        +std::to_string(current.sequence)+",\"api\":\""+api+"\",\"graph\":\""+pointer(graph)
+        +"\",\"opaque_plan\":\""+pointer(plan)+"\",\"plan_id\":"+std::to_string(plan_id)+"}");
+}
+static void finish(Scope &current,ggml_status status) {
+    checked(scope==&current && !current.in_inner && current.inner==1 && status==GGML_STATUS_SUCCESS,
+        "each successful public CPU compute has exactly one successful actual inner call");
+    std::lock_guard<std::mutex> guard(mutex);checked(inflight==current.sequence,"exact active original compute sequence");
+    ++public_success;inflight=0;
+    append("{\"kind\":\"public_compute_return\",\"backend\":\""+pointer(current.backend)
+        +"\",\"sequence\":"+std::to_string(current.sequence)+",\"status\":0,\"inner_calls\":1}");
+    scope=nullptr;
+}
 }
 
 extern "C" ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device,const char *params) {
     const int saved=errno;
-    static ref_control::Init original=ref_control::symbol<ref_control::Init>(ref_control::BASE,
+    static auto original=ref_control::symbol<decltype(&ggml_backend_dev_init)>(ref_control::BASE,
         ref_control::BASE_BYTES,"ggml_backend_dev_init");
     errno=saved;auto backend=original(device,params);const int after=errno;
-    if(ref_control::armed){ref_control::checked(backend!=nullptr,"native backend initialization failed");ref_control::set_reference(backend,true,device);}
+    if(ref_control::armed) {
+        ref_control::initialized(backend,device);
+        bool cpu=false;
+        {std::lock_guard<std::mutex> guard(ref_control::mutex);
+            cpu=ref_control::known(backend).kind==ref_control::Kind::Cpu;}
+        if(cpu)ref_control::set_reference(backend,true);
+    }
     errno=after;return backend;
 }
 extern "C" void *ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg,const char *name) {
@@ -194,46 +279,144 @@
     void *value=original(reg,name);const int after=errno;
     if(ref_control::armed && value && name && std::strcmp(name,"ggml_backend_set_abort_callback")==0) {
         ref_control::attest(value,ref_control::CPU,ref_control::CPU_BYTES,"ggml_backend_cpu_set_abort_callback");
+        static auto cpu_reg=ref_control::symbol<decltype(&ggml_backend_cpu_reg)>(ref_control::CPU,
+            ref_control::CPU_BYTES,"ggml_backend_cpu_reg");
+        ref_control::checked(reg==cpu_reg(),"exact CPU registry supplies original abort callback");
         value=reinterpret_cast<void *>(&ref_control::abort_setter);
     }
     errno=after;return value;
+}
+extern "C" enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend,struct ggml_cgraph *graph) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_graph_compute_async)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_graph_compute_async");
+    ref_control::Scope current;
+    if(ref_control::armed)ref_control::begin(current,backend,graph,nullptr,"async");
+    errno=saved;const auto status=original(backend,graph);const int after=errno;
+    if(ref_control::armed)ref_control::finish(current,status);
+    errno=after;return status;
+}
+extern "C" enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend,struct ggml_cgraph *graph) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_graph_compute)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_graph_compute");
+    ref_control::SyncScope current;
+    if(ref_control::armed) {
+        ref_control::checked(ref_control::sync_scope==nullptr && ref_control::scope==nullptr && graph,
+            "nonnested public sync route");
+        {std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::cpu_compute_only(backend,"sync");}
+        current.backend=backend;current.graph=graph;ref_control::sync_scope=&current;
+    }
+    errno=saved;const auto status=original(backend,graph);const int after=errno;
+    if(ref_control::armed) {
+        ref_control::checked(current.async==1 && status==GGML_STATUS_SUCCESS && ref_control::scope==nullptr,
+            "actual public sync delegates exactly once through guarded async");
+        std::lock_guard<std::mutex> guard(ref_control::mutex);++ref_control::sync_calls;++ref_control::known(backend).sync;
+        ref_control::append("{\"kind\":\"public_sync_return\",\"backend\":\""+ref_control::pointer(backend)
+            +"\",\"sequence\":"+std::to_string(current.sequence)+",\"status\":0}");
+        ref_control::sync_scope=nullptr;
+    }
+    errno=after;return status;
+}
+extern "C" ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend,struct ggml_cgraph *graph) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_graph_plan_create)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_graph_plan_create");
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::cpu_compute_only(backend,"plan_create");
+        ref_control::checked(graph && ref_control::scope==nullptr && ref_control::plans.size()<64,
+            "bounded nonnested public opaque plan creation");
+    }
+    errno=saved;auto plan=original(backend,graph);const int after=errno;
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);
+        ref_control::checked(plan && ref_control::plans.find(plan)==ref_control::plans.end()
+            && ref_control::next_plan<65536,"unique bounded live opaque plan");
+        ref_control::Plan record;record.backend=backend;record.id=++ref_control::next_plan;
+        ref_control::plans.emplace(plan,record);++ref_control::plan_creates;
+        ref_control::append("{\"kind\":\"public_plan_create\",\"backend\":\""+ref_control::pointer(backend)
+            +"\",\"opaque_plan\":\""+ref_control::pointer(plan)+"\",\"plan_id\":"+std::to_string(record.id)+"}");
+    }
+    errno=after;return plan;
+}
+extern "C" void ggml_backend_graph_plan_free(ggml_backend_t backend,ggml_backend_graph_plan_t plan) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_graph_plan_free)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_graph_plan_free");
+    size_t id=0;
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::cpu_compute_only(backend,"plan_free");
+        const auto found=ref_control::plans.find(plan);
+        ref_control::checked(ref_control::scope==nullptr && found!=ref_control::plans.end()
+            && found->second.backend==backend,"public plan freed once by original CPU owner");id=found->second.id;
+    }
+    errno=saved;original(backend,plan);const int after=errno;
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);ref_control::plans.erase(plan);++ref_control::plan_frees;
+        ref_control::append("{\"kind\":\"public_plan_free\",\"backend\":\""+ref_control::pointer(backend)
+            +"\",\"opaque_plan\":\""+ref_control::pointer(plan)+"\",\"plan_id\":"+std::to_string(id)+"}");
+    }
+    errno=after;
+}
+extern "C" enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend,ggml_backend_graph_plan_t plan) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_graph_plan_compute)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_graph_plan_compute");
+    ref_control::Scope current;
+    if(ref_control::armed)ref_control::begin(current,backend,nullptr,plan,"plan");
+    errno=saved;const auto status=original(backend,plan);const int after=errno;
+    if(ref_control::armed)ref_control::finish(current,status);
+    errno=after;return status;
 }
 extern "C" enum ggml_status ggml_graph_compute(struct ggml_cgraph *graph,struct ggml_cplan *plan) {
     const int saved=errno;
     static ref_control::Compute original=ref_control::symbol<ref_control::Compute>(ref_control::CPU,
         ref_control::CPU_BYTES,"ggml_graph_compute");
-    if(ref_control::armed) {
-        ref_control::check_affinity();
-        ref_control::checked(graph && plan && plan->use_ref,"actual PUBLIC cplan.use_ref is false/missing");
+    int actual_threads=0;
+    if(ref_control::armed) {
+        ref_control::check_affinity();auto *current=ref_control::scope;
+        ref_control::checked(current && !current->in_inner && current->inner==0 && graph && plan && plan->use_ref
+            && (!current->graph || current->graph==graph),"actual CPU inner compute has exact public TLS owner/use_ref");
         ref_control::checked(plan->n_threads>=1 && plan->n_threads<=8,"actual public plan thread bound");
-        std::lock_guard<std::mutex> guard(ref_control::mutex);
-        ++ref_control::graph_calls;
+        actual_threads=plan->n_threads;current->in_inner=true;
+    }
+    errno=saved;const auto status=original(graph,plan);const int after=errno;
+    if(ref_control::armed) {
+        auto *current=ref_control::scope;
+        ref_control::checked(current && current->in_inner && plan->use_ref && plan->n_threads==actual_threads
+            && status==GGML_STATUS_SUCCESS,
+            "actual inner CPU compute successful with reference flag unchanged");
+        std::lock_guard<std::mutex> guard(ref_control::mutex);++ref_control::graph_calls;++ref_control::graph_success;
+        ++ref_control::known(current->backend).inner;++current->inner;current->in_inner=false;
         ref_control::thread_min=std::min(ref_control::thread_min,plan->n_threads);
         ref_control::thread_max=std::max(ref_control::thread_max,plan->n_threads);
-        ref_control::append("{\"kind\":\"actual_public_graph_plan\",\"sequence\":"+std::to_string(ref_control::graph_calls)
-            +",\"use_ref\":true,\"n_threads\":"+std::to_string(plan->n_threads)+"}");
-    }
-    errno=saved;const auto status=original(graph,plan);const int after=errno;
-    if(ref_control::armed) {
-        ref_control::checked(plan->use_ref,"actual public compute changed reference flag");
+        ref_control::append("{\"kind\":\"actual_public_graph_plan\",\"backend\":\""+ref_control::pointer(current->backend)
+            +"\",\"sequence\":"+std::to_string(current->sequence)+",\"graph\":\""+ref_control::pointer(graph)
+            +"\",\"cplan\":\""+ref_control::pointer(plan)+"\",\"use_ref_before\":true,\"use_ref_after\":true,\"n_threads\":"
+            +std::to_string(plan->n_threads)+",\"status\":0}");
+    }
+    errno=after;return status;
+}
+extern "C" void ggml_backend_free(ggml_backend_t backend) {
+    const int saved=errno;
+    static auto original=ref_control::symbol<decltype(&ggml_backend_free)>(ref_control::BASE,
+        ref_control::BASE_BYTES,"ggml_backend_free");
+    ref_control::Kind kind=ref_control::Kind::Cpu;
+    if(ref_control::armed) {
+        std::lock_guard<std::mutex> guard(ref_control::mutex);auto &record=ref_control::known(backend);
+        ref_control::checked(ref_control::scope==nullptr && ref_control::sync_scope==nullptr
+            && (record.kind==ref_control::Kind::Cpu?(record.abort==1 && record.set==2):(record.set==0 && record.abort==0)),
+            "only exact configured CPU or unmodified BLAS retirement outside math");
+        for(const auto &item:ref_control::plans)ref_control::checked(item.second.backend!=backend,"backend retains live opaque plan");
+        record.freed=true;kind=record.kind;
+    }
+    errno=saved;original(backend);const int after=errno;
+    if(ref_control::armed) {
         std::lock_guard<std::mutex> guard(ref_control::mutex);
-        if(status==GGML_STATUS_SUCCESS)++ref_control::graph_success;
-        else ref_control::fatal("actual public compute returned failure");
-    }
-    errno=after;return status;
-}
-extern "C" void ggml_backend_free(ggml_backend_t backend) {
-    const int saved=errno;
-    static ref_control::Free original=ref_control::symbol<ref_control::Free>(ref_control::BASE,
-        ref_control::BASE_BYTES,"ggml_backend_free");
-    if(ref_control::armed && backend) {
-        std::lock_guard<std::mutex> guard(ref_control::mutex);
-        auto found=ref_control::backends.find(backend);
-        ref_control::checked(found!=ref_control::backends.end() && !found->second.freed,"unknown/duplicate backend retirement");
-        found->second.freed=true;
-        ref_control::append("{\"kind\":\"public_backend_free\",\"backend\":\""+ref_control::pointer(backend)+"\"}");
-    }
-    errno=saved;original(backend);
+        ref_control::append("{\"kind\":\"public_backend_free\",\"backend\":\""+ref_control::pointer(backend)
+            +"\",\"backend_kind\":\""+ref_control::kind_name(kind)+"\"}");
+    }
+    errno=after;
 }
 
 #define main zg_unchanged_native_raw_main
@@ -265,14 +448,29 @@
         const int result=zg_unchanged_native_raw_main(argc,argv);
         require(result==0,"unchanged native capture failed");
         std::lock_guard<std::mutex> guard(ref_control::mutex);
-        require(ref_control::backends.size()==2,"both actual plain/observed CPU contexts observed");
+        require(ref_control::backends.size()==4 && ref_control::plans.empty()
+            && ref_control::scope==nullptr && ref_control::sync_scope==nullptr && ref_control::inflight==0,
+            "all original public owners retired");
+        size_t cpu=0,blas=0;
         for(const auto &item:ref_control::backends) {
-            require(item.second.init==1 && item.second.abort==1 && item.second.set==2 && item.second.freed,
-                "every actual CPU context initialized/configured/retired through public APIs");
+            const auto &r=item.second;
+            require(r.freed,"every original CPU/BLAS context retired");
+            if(r.kind==ref_control::Kind::Cpu) {
+                ++cpu;require(r.abort==1 && r.set==2 && r.inner>0 && r.inner==r.async+r.plans,
+                    "each original CPU nonvacuous with complete public/inner use_ref coverage");
+            } else {
+                ++blas;require(r.abort==0 && r.set==0 && r.async==0 && r.plans==0 && r.inner==0 && r.sync==0,
+                    "pinned BLAS initialized/retired unmodified and never computed");
+            }
         }
+        require(cpu==2 && blas==2,"exact original two CPU and two pinned BLAS context backends");
         const size_t minimum_calls=(8191+std::stoul(argv[4])-1)/std::stoul(argv[4])+2;
-        require(ref_control::graph_calls>=minimum_calls && ref_control::graph_calls==ref_control::graph_success,
-            "actual public cplan readback required, no missing/failed route accepted");
+        require(ref_control::graph_calls>=minimum_calls && ref_control::graph_calls==ref_control::graph_success
+            && ref_control::graph_calls==ref_control::public_calls
+            && ref_control::public_calls==ref_control::public_success
+            && ref_control::public_calls==ref_control::async_calls+ref_control::plan_calls
+            && ref_control::plan_creates==ref_control::plan_frees,
+            "actual public/inner conservation and all opaque plans retired");
         ref_control::check_affinity();
         require(ref_control::trace_fd>=0 && fsync(ref_control::trace_fd)==0,"public trace complete");
         require(close(ref_control::trace_fd)==0,"public trace close");ref_control::trace_fd=-1;
@@ -290,16 +488,29 @@
         write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
         std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
         report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
-            <<"\"control_source_revision\":\"public_use_ref_backend_identity_diagnostic_v3\","
-            <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\",\"CPU_backend_count\":2,"
-            <<"\"public_setter_calls\":4,\"existing_abort_setter_calls\":2,\"actual_public_plan_calls\":"<<ref_control::graph_calls
+            <<"\"control_source_revision\":\"public_use_ref_exact_blas_lifecycle_cpu_compute_v4\","
+            <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\","
+            <<"\"CPU_backend_count\":2,\"BLAS_backend_count\":2,\"public_setter_calls\":4,"
+            <<"\"existing_abort_setter_calls\":2,\"BLAS_compute_delegations\":0,"
+            <<"\"actual_public_plan_calls\":"<<ref_control::graph_calls
             <<",\"actual_successful_public_plan_calls\":"<<ref_control::graph_success
-            <<",\"actual_plan_threads_min\":"<<ref_control::thread_min<<",\"actual_plan_threads_max\":"<<ref_control::thread_max
-            <<",\"actual_public_plan_use_ref_all_true\":true,\"actual_CPU_affinity\":[";
+            <<",\"public_compute_calls\":"<<ref_control::public_calls
+            <<",\"public_async_calls\":"<<ref_control::async_calls
+            <<",\"public_opaque_plan_compute_calls\":"<<ref_control::plan_calls
+            <<",\"public_sync_calls\":"<<ref_control::sync_calls
+            <<",\"public_opaque_plan_creates\":"<<ref_control::plan_creates
+            <<",\"public_opaque_plan_frees\":"<<ref_control::plan_frees
+            <<",\"opaque_plan_route_observed\":"<<(ref_control::plan_calls?"true":"false")
+            <<",\"actual_plan_threads_min\":"<<ref_control::thread_min
+            <<",\"actual_plan_threads_max\":"<<ref_control::thread_max
+            <<",\"actual_public_plan_use_ref_all_true\":true,\"public_inner_TLS_scope_exact\":true,"
+            <<"\"BLAS_construction_retirement_only\":true,\"no_BLAS_disable_or_offload_change\":true,"
+            <<"\"actual_CPU_affinity\":[";
         for(size_t i=0;i<ref_control::declared_cpus.size();++i)report<<(i?",":"")<<ref_control::declared_cpus[i];
         report<<"],\"backend_private_getter_available\":false,"
-            <<"\"plan_to_backend_identity_publicly_available\":false,\"CPU_REPACK_policy_unchanged\":true,"
-            <<"\"native_raw_token_loop_unchanged\":true,\"historical_prefix_or_cause_closed\":false}\n";
+            <<"\"plan_to_backend_identity_publicly_available\":false,\"opaque_plan_public_association_checked\":true,"
+            <<"\"CPU_REPACK_policy_unchanged\":true,\"native_raw_token_loop_unchanged\":true,"
+            <<"\"historical_prefix_or_cause_closed\":false}\n";
         report.flush();require(bool(report),"control report write");return 0;
     }catch(const std::exception &error){std::cerr<<"USE_REF_CONTROL_UNQUALIFIED "<<error.what()<<'\n';return 2;}
 }
--- build-argv-v3.json
+++ build-argv-v4.json
@@ -11,7 +11,7 @@
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include",
     "-I",
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v3.cpp",
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v4.cpp",
     "-L",
     "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib",
     "-L",
@@ -22,6 +22,11 @@
     "-Wl,--export-dynamic-symbol=ggml_backend_reg_get_proc_address",
     "-Wl,--export-dynamic-symbol=ggml_graph_compute",
     "-Wl,--export-dynamic-symbol=ggml_backend_free",
+    "-Wl,--export-dynamic-symbol=ggml_backend_graph_compute_async",
+    "-Wl,--export-dynamic-symbol=ggml_backend_graph_compute",
+    "-Wl,--export-dynamic-symbol=ggml_backend_graph_plan_create",
+    "-Wl,--export-dynamic-symbol=ggml_backend_graph_plan_free",
+    "-Wl,--export-dynamic-symbol=ggml_backend_graph_plan_compute",
     "-lllama",
     "-lggml",
     "-lggml-base",
@@ -29,11 +34,11 @@
     "-ldl",
     "-pthread",
     "-o",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v3"
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v4"
   ],
   "compiler_identity": "ROOT must record real compiler and own finite build receipt before execution",
   "native_math_source_unchanged": true,
-  "source_revision": "public_use_ref_backend_identity_diagnostic_v3",
-  "generic_runner_v1_freeze_scope": "unchanged generic helper only; actual V3 CPP/binary/build binding separately mandatory",
-  "scope": "DIAGNOSTIC_ONLY same pre-setter fatal125; no nonCPU policy allowance"
+  "source_revision": "public_use_ref_exact_blas_lifecycle_cpu_compute_v4",
+  "generic_runner_v1_freeze_scope": "unchanged generic helper; V4 source/build/compiled-route/invocation binding independently mandatory",
+  "scope": "SOURCE ONLY exact BLAS construction/retirement; refuse EVERY nonCPU compute before delegation; CPU public-to-inner TLS/use_ref guard; no arithmetic acceptance"
 }
--- bind_v3.py
+++ bind_v4.py
@@ -1,9 +1,10 @@
-"""ROOT ONLY: bind actual V3 compile/run receipts; no model/operator calls."""
+"""ROOT ONLY: bind actual V4 compile/run/static-provider receipts; no native entry calls."""
 import argparse
 from pathlib import Path
 import re
 
 from common import HERE, REPO, digest, jread, require, write
+from v4_sources import REVISION, verify_sources, verify_runtime_files
 
 
 def main():
@@ -15,22 +16,9 @@
     args = parser.parse_args()
     require(re.fullmatch("[a-f0-9]{64}", args.manifest_sha256) is not None,
             "independently reviewed manifest SHA")
-    path = HERE / "freeze-manifest-v3.json"
-    require(digest(path)["sha256"] == args.manifest_sha256, "exact reviewed V3 freeze")
-    manifest = jread(path)
-    for version, directory in (("original_freeze", "frozen-prepared"), ("v2_freeze", "frozen-v2")):
-        pin = manifest[version]
-        parent = HERE / pin["file"]
-        require(digest(parent) == {key: pin[key] for key in ("bytes", "sha256")}, "unchanged prior manifest")
-        for name, expected in jread(parent)["files"].items():
-            require(digest(HERE / name) == digest(HERE / directory / name) == expected,
-                    "all original source and frozen copies unchanged: " + name)
-    for name, expected in manifest["files"].items():
-        require(digest(HERE / name) == digest(HERE / "frozen-v3" / name) == expected,
-                "new frozen source unchanged: " + name)
-    for row in manifest["preserved_failed_metadata"]:
-        require(digest(row["path"]) == {key: row[key] for key in ("bytes", "sha256")},
-                "failed case evidence unchanged")
+    path = HERE / "freeze-manifest-v4.json"
+    manifest = verify_sources(args.manifest_sha256)
+    extra_providers = verify_runtime_files()
     build, run = args.build.resolve(strict=True), args.run.resolve(strict=True)
     destination = args.new_binding.resolve()
     require(build.is_relative_to(HERE) and run.is_relative_to(HERE)
@@ -38,7 +26,7 @@
             "owned actual build/run and fresh binding only")
     requested, result = jread(build / "requested.json"), jread(build / "result.json")
     compiler = manifest["root_actual_compiler_pin"]
-    prepared = jread(HERE / "build-argv-v3.json")
+    prepared = jread(HERE / "build-argv-v4.json")
     actual_inputs = {row["path"]: digest(row["path"]) for row in manifest["compile_inputs"]}
     require(actual_inputs == {row["path"]: {key: row[key] for key in ("bytes", "sha256")}
                               for row in manifest["compile_inputs"]}, "actual exact compiler inputs")
@@ -53,16 +41,16 @@
                 and receipt["runtime_or_math_qualified"] is False,
                 "actual relative-argv CWD/new CPP+included source pre/post build binding")
     require(type(result["exit_code"]) is int and result["exit_code"] == 0, "actual successful fresh build")
-    binary = (HERE / "native_ref_control_v3").resolve(strict=True)
+    binary = (HERE / "native_ref_control_v4").resolve(strict=True)
     identity = digest(binary)
     require(result["binary_path"] == str(binary)
             and identity == {"bytes": result["binary_bytes"], "sha256": result["binary_sha256"]},
             "exact actual new binary output")
     invocation = jread(run / "invocation.json")
     require(invocation["argv"][0] == str(binary) and invocation["inputs"][str(binary)] == identity,
-            "actual generic runner executed this newly compiled V3 binary")
+            "actual generic runner executed this newly compiled V4 binary")
     require(invocation["freeze_manifest"] == digest(HERE / "freeze-manifest.json"),
-            "generic V1 helper receipt remains separate from V3 CPP build evidence")
+            "generic V1 helper receipt remains separate from V4 CPP build evidence")
     native_log = run / "native.log"
     require(native_log.is_file() and not native_log.is_symlink()
             and 0 < native_log.stat().st_size == invocation["log_bytes"] <= 8 * 1024 * 1024,
@@ -72,24 +60,36 @@
     control = None
     if control_path.exists():
         actual = jread(control_path)
-        require(actual["control_source_revision"] == "public_use_ref_backend_identity_diagnostic_v3"
+        require(actual["control_source_revision"] == REVISION
                 and actual["provider_policy"] == "same_canonical_path_regular_device_inode_length_exact_export",
-                "actual V3 source marker with unchanged V2 attestation policy")
+                "actual V4 source marker with unchanged V2 attestation policy")
         control = digest(control_path)
     else:
-        require(invocation["status"] == "UNQUALIFIED", "successful run missing actual V3 control report")
+        require(invocation["status"] == "UNQUALIFIED", "successful run missing actual V4 control report")
+    trace_path = run / "capture/use-ref-public-trace.jsonl"
+    maps_path = run / "capture/use-ref-loaded-maps.txt"
+    trace = digest(trace_path) if trace_path.exists() else None
+    maps = digest(maps_path) if maps_path.exists() else None
+    require((trace is None or trace["bytes"] <= 8 * 1024**2)
+            and (maps is None or maps["bytes"] <= 2 * 1024**2), "bounded actual control trace/maps")
+    verify_sources(args.manifest_sha256)
+    verify_runtime_files()
     write(destination, {
         "status": "SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE",
         "new_source_freeze": digest(path), "generic_V1_helper_freeze": digest(HERE / "freeze-manifest.json"),
         "V2_attestation_freeze": digest(HERE / "freeze-manifest-v2.json"),
-        "new_cpp": {"path": str(HERE / "native_ref_control_v3.cpp"), **digest(HERE / "native_ref_control_v3.cpp")},
+        "V3_diagnostic_freeze": digest(HERE / "freeze-manifest-v3.json"),
+        "new_cpp": {"path": str(HERE / "native_ref_control_v4.cpp"), **digest(HERE / "native_ref_control_v4.cpp")},
         "included_unchanged_token_loop": digest(HERE / "native_raw_original.cpp"),
         "build_requested": digest(build / "requested.json"), "build_result": digest(build / "result.json"),
         "actual_binary": {"path": str(binary), **identity}, "actual_invocation": digest(run / "invocation.json"),
         "actual_diagnostic_log": {"path": str(native_log), **log_identity},
         "actual_invocation_status": invocation["status"], "actual_control_report": control,
-        "diagnostic_only_no_nonCPU_policy_allowance": True, "runtime_or_math_qualified": False,
-        "next": "read actual bounded backend-identity refusal; do not infer or enable ACCEL policy",
+        "actual_control_trace": trace, "actual_loaded_maps": maps,
+        "exact_BLAS_construction_retirement_only": True, "nonCPU_compute_delegation_permitted": False,
+        "runtime_or_math_qualified": False, "actual_extra_provider_and_static_scope": extra_providers,
+        "selected_static_route_pin": digest(HERE / "static-route-pins-v4.json"),
+        "next": "validate_control_v4.py then separate raw730/fullV/all42 comparators; any BLAS compute stays UNQUALIFIED",
     })
     print("SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE")
 
--- validate_control.py
+++ validate_control_v4.py
@@ -5,66 +5,9 @@
 import json
 from pathlib import Path
 import re
-from common import HERE, REPO, TEACHER_SHA, authenticate, bounded, digest, jread, require, unique, write
-
-
-def validate_trace(raw, report, prefix):
-    require(type(prefix) is int and prefix in [64, 512], 'closed actual prefix geometry')
-    require(0 < len(raw) <= 8 * 1024**2 and raw.endswith(b'\n'), 'complete bounded public trace')
-    lines = raw.splitlines()
-    require(len(lines) <= 65536, 'bounded actual public call count')
-    backends = {}
-    plans = 0
-    threads = []
-    for line in lines:
-        require(0 < len(line) < 2048, 'bounded actual public trace line')
-        row = json.loads(line, object_pairs_hook=unique)
-        kind = row.get('kind')
-        if kind == 'public_cpu_reference_setter':
-            require(set(row) == {'kind', 'backend', 'origin', 'requested_use_ref'}
-                    and row['requested_use_ref'] is True, 'exact original setter trace')
-            name = row['backend']
-            require(isinstance(name, str) and re.fullmatch(r'0x[0-9a-f]{1,16}', name)
-                    and int(name, 16) > 0, 'nonnull opaque identity only')
-            origin = row['origin']
-            if origin == 'public_dev_init':
-                require(name not in backends and len(backends) < 2, 'unique two actual CPU context initializations')
-                backends[name] = {'abort': False, 'freed': False}
-            else:
-                require(origin == 'existing_public_abort_setter_callback' and name in backends
-                        and backends[name] == {'abort': False, 'freed': False}, 'actual callback exactly once')
-                backends[name]['abort'] = True
-        elif kind == 'actual_public_graph_plan':
-            plans += 1
-            require(set(row) == {'kind', 'sequence', 'use_ref', 'n_threads'}
-                    and type(row['sequence']) is int and row['sequence'] == plans
-                    and row['use_ref'] is True, 'every actual delegated plan has exact reference policy')
-            require(backends and all(b['abort'] and not b['freed'] for b in backends.values()),
-                    'every actually seen context configured before computation, no retired context computation')
-            require(type(row['n_threads']) is int and 1 <= row['n_threads'] <= 8, 'actual thread bound')
-            threads.append(row['n_threads'])
-        elif kind == 'public_backend_free':
-            require(set(row) == {'kind', 'backend'} and row['backend'] in backends, 'only known CPU backend retirement')
-            backend = backends[row['backend']]
-            require(backend == {'abort': True, 'freed': False}, 'exact once original backend retirement')
-            backend['freed'] = True
-        else:
-            raise ValueError('unexpected public trace record')
-    minimum = (8191 + prefix - 1) // prefix + 2
-    require(all(type(report[field]) is int for field in ['actual_public_plan_calls',
-                'actual_successful_public_plan_calls', 'actual_plan_threads_min', 'actual_plan_threads_max']),
-            'actual integer counters and thread bounds')
-    require(len(backends) == 2 and all(b == {'abort': True, 'freed': True} for b in backends.values()),
-            'both actual contexts fully configured and retired')
-    require(plans >= minimum and len(lines) == plans + 6
-            and report['actual_public_plan_calls'] == plans
-            and report['actual_successful_public_plan_calls'] == plans
-            and report['actual_plan_threads_min'] == min(threads)
-            and report['actual_plan_threads_max'] == max(threads), 'no missing compute route or fabricated counts')
-    return {'actual_cpu_contexts': 2, 'actual_setter_calls': 4,
-            'actual_original_abort_setter_calls': 2, 'actual_plan_calls': plans,
-            'actual_public_plan_use_ref_all_true': True, 'actual_threads_min': min(threads),
-            'actual_threads_max': max(threads)}
+from common import HERE, REPO, TEACHER_SHA, authenticate, bounded, digest, jread, require, write
+from trace_v4 import validate_trace
+from v4_sources import REVISION, verify_sources, verify_runtime_files
 
 
 def cpulist(text):
@@ -79,6 +22,8 @@
 
 
 def validate(output):
+    verify_sources()
+    extra_providers = verify_runtime_files()
     invocation = jread(output / 'invocation.json')
     require(invocation['status'] == 'PUBLIC_USE_REF_NATIVE_CAPTURE_CONTROL_VALIDATOR_REQUIRED'
             and invocation['primary_error'] is None and invocation['cleanup_errors'] == []
@@ -134,8 +79,16 @@
     require(complete['threads_actual'] == complete['threads_batch_actual'] == 8, 'original actual CPU8 contexts')
     control = jread(capture / 'use-ref-control.json')
     require(control['status'] == 'PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED'
-            and control['CPU_backend_count'] == 2 and control['public_setter_calls'] == 4
+            and control['control_source_revision'] == REVISION
+            and control['provider_policy'] == 'same_canonical_path_regular_device_inode_length_exact_export'
+            and control['CPU_backend_count'] == 2 and control['BLAS_backend_count'] == 2
+            and control['public_setter_calls'] == 4
             and control['existing_abort_setter_calls'] == 2
+            and control['BLAS_compute_delegations'] == 0
+            and control['public_inner_TLS_scope_exact'] is True
+            and control['BLAS_construction_retirement_only'] is True
+            and control['no_BLAS_disable_or_offload_change'] is True
+            and control['opaque_plan_public_association_checked'] is True
             and control['actual_public_plan_use_ref_all_true'] is True
             and control['backend_private_getter_available'] is False
             and control['plan_to_backend_identity_publicly_available'] is False
@@ -154,12 +107,28 @@
     require(all(row['path'] in paths for row in providers), 'all original pinned providers actually mapped')
     require(not any('libggml-cpu-' in path and not path.endswith('/libggml-cpu-haswell.so') for path in paths),
             'no unobserved alternate CPU implementation')
-    return dict(status='PASS_PUBLIC_USE_REF_CONTROL_CAPTURE_ONLY_COMPARISONS_REQUIRED',
+    blas_path = extra_providers['BLAS']['canonical']
+    require(blas_path in paths and not any('libggml-blas' in path and path != blas_path for path in paths),
+            'actual exact pinned BLAS mapped, no alternate BLAS provider')
+    binding = jread(output / 'source-build-binding-v4.json')
+    require(binding['status'] == 'SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE'
+            and binding['actual_invocation'] == digest(output / 'invocation.json')
+            and binding['new_source_freeze'] == digest(HERE / 'freeze-manifest-v4.json')
+            and binding['actual_control_report'] == digest(capture / 'use-ref-control.json')
+            and binding['actual_control_trace'] == digest(capture / 'use-ref-public-trace.jsonl')
+            and binding['actual_loaded_maps'] == digest(capture / 'use-ref-loaded-maps.txt')
+            and binding['actual_binary'] == dict(path=invocation['argv'][0], **invocation['inputs'][invocation['argv'][0]]),
+            'independent ROOT compile/actual V4 binary/generic runner binding required')
+    verify_sources()
+    verify_runtime_files()
+    return dict(status='PASS_PUBLIC_USE_REF_V4_CONTROL_CAPTURE_ONLY_COMPARISONS_REQUIRED',
                 source_model_sha256=case['model_identity']['sha256'], raw_teacher_sha256=TEACHER_SHA,
                 cut=8192, prefix_batch=invocation['prefix'], CPU_affinity=cpus, public_route=observed,
                 invocation=digest(output / 'invocation.json'), complete=digest(capture / 'complete.json'),
                 control=digest(capture / 'use-ref-control.json'),
-                scope='actual setter+public-plan policy and original selected-tail observation transparency only; '
+                extra_provider=extra_providers, source_build_binding=digest(output / 'source-build-binding-v4.json'),
+                scope='exact BLAS lifecycle with no BLAS math; CPU public-route/TLS/actual-use_ref policy and '
+                      'original selected-tail transparency only; selected static entry evidence is not universal runtime closure; '
                       'no raw730/cross-provider all42 result or historical/whole-cause conclusion')
 
 

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/frozen-v2/PROVIDER_ATTESTATION_V2.patch

Original UTF-8 bytes: 7249; SHA256 `365ae456714c2cacd2196078304f09d5d4627d8258a3ea85435c6b8abcb58198`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- original-immutable/native_ref_control.cpp
+++ additive-new/native_ref_control_v2.cpp
@@ -6,6 +6,7 @@
 #include <dlfcn.h>
 #include <algorithm>
 #include <fcntl.h>
+#include <limits.h>
 #include <sched.h>
 #include <sys/stat.h>
 #include <unistd.h>
@@ -46,11 +47,24 @@
     const int saved=errno;
     const char prefix[]="USE_REF_CONTROL_UNQUALIFIED ";
     const ssize_t a=write(STDERR_FILENO,prefix,sizeof(prefix)-1);
-    const ssize_t b=write(STDERR_FILENO,message,strnlen(message,512));
+    const ssize_t b=write(STDERR_FILENO,message,strnlen(message,4096));
     const ssize_t c=write(STDERR_FILENO,"\n",1);
     (void)a;(void)b;(void)c;errno=saved;_exit(125);
 }
 static void checked(bool value,const char *message) {if(!value)fatal(message);}
+[[noreturn]] static void provider_failure(const char *stage,const char *name,
+        const char *expected,const char *actual,const char *canonical_expected,
+        const char *canonical_actual,const char *error,const char *actual_symbol=nullptr) {
+    const int saved=errno;char message[4096]={};
+    const int size=std::snprintf(message,sizeof(message),
+        "provider_attestation_v2 stage=%.128s symbol=%.192s expected=%.512s actual=%.512s "
+        "canonical_expected=%.512s canonical_actual=%.512s actual_symbol=%.192s dlerror=%.512s errno=%d",
+        stage?stage:"(null)",name?name:"(null)",expected?expected:"(null)",actual?actual:"(null)",
+        canonical_expected?canonical_expected:"(unresolved)",canonical_actual?canonical_actual:"(unresolved)",
+        actual_symbol?actual_symbol:"(unavailable)",error?error:"(none)",saved);
+    if(size<0)fatal("provider_attestation_v2 diagnostic formatting failed");
+    errno=saved;fatal(message);
+}
 static void check_affinity() {
     cpu_set_t actual;CPU_ZERO(&actual);
     checked(sched_getaffinity(0,sizeof(actual),&actual)==0
@@ -80,20 +94,31 @@
 }
 static void attest(void *address,const char *path,size_t length,const char *name) {
     Dl_info info{};struct stat actual{},expected{};
-    checked(address && dladdr(address,&info)!=0 && info.dli_fname,"public symbol lookup");
-    checked(std::strcmp(info.dli_fname,path)==0,"public symbol exact loaded provider path");
-    checked(stat(path,&expected)==0 && stat(info.dli_fname,&actual)==0
-        && expected.st_dev==actual.st_dev && expected.st_ino==actual.st_ino
-        && expected.st_size==static_cast<off_t>(length),"public symbol provider inode/length");
-    checked(info.dli_sname && std::strcmp(info.dli_sname,name)==0,"public symbol exact exported name");
+    if(!address || dladdr(address,&info)==0 || !info.dli_fname)
+        provider_failure("dladdr",name,path,info.dli_fname,nullptr,nullptr,dlerror());
+    char expected_real[PATH_MAX]={},actual_real[PATH_MAX]={};
+    if(!realpath(path,expected_real))
+        provider_failure("canonical_expected",name,path,info.dli_fname,nullptr,nullptr,nullptr);
+    if(!realpath(info.dli_fname,actual_real))
+        provider_failure("canonical_actual",name,path,info.dli_fname,expected_real,nullptr,nullptr);
+    if(std::strcmp(actual_real,expected_real)!=0)
+        provider_failure("canonical_identity",name,path,info.dli_fname,expected_real,actual_real,nullptr);
+    if(stat(path,&expected)!=0 || stat(info.dli_fname,&actual)!=0
+        || !S_ISREG(expected.st_mode) || !S_ISREG(actual.st_mode)
+        || expected.st_dev!=actual.st_dev || expected.st_ino!=actual.st_ino
+        || expected.st_size!=static_cast<off_t>(length) || actual.st_size!=static_cast<off_t>(length))
+        provider_failure("regular_inode_length",name,path,info.dli_fname,expected_real,actual_real,nullptr);
+    if(!info.dli_sname || std::strcmp(info.dli_sname,name)!=0)
+        provider_failure("exact_exported_symbol",name,path,info.dli_fname,expected_real,actual_real,nullptr,info.dli_sname);
 }
 template<class F> static F symbol(const char *path,size_t bytes,const char *name) {
     // NOLOAD never acquires/replaces an unregistered CPU implementation. Hold
     // the original module through backend_free and the final observation.
-    void *handle=dlopen(path,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
-    checked(handle!=nullptr,"exact original provider is not already loaded");
+    dlerror();void *handle=dlopen(path,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
+    if(!handle)provider_failure("dlopen_noload",name,path,nullptr,nullptr,nullptr,dlerror());
     dlerror();void *address=dlsym(handle,name);const char *error=dlerror();
-    checked(error==nullptr,"public symbol resolution failed");attest(address,path,bytes,name);
+    if(error)provider_failure("dlsym",name,path,nullptr,nullptr,nullptr,error);
+    attest(address,path,bytes,name);
     return reinterpret_cast<F>(address);
 }
 static Lookup lookup() {static Lookup f=symbol<Lookup>(BASE,BASE_BYTES,"ggml_backend_reg_get_proc_address");return f;}
@@ -233,7 +258,9 @@
         require(status.eof() && status_count>0 && status_count<=64*1024,"bounded actual process status");
         write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
         std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
-        report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\",\"CPU_backend_count\":2,"
+        report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
+            <<"\"control_source_revision\":\"public_use_ref_canonical_provider_attestation_v2\","
+            <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\",\"CPU_backend_count\":2,"
             <<"\"public_setter_calls\":4,\"existing_abort_setter_calls\":2,\"actual_public_plan_calls\":"<<ref_control::graph_calls
             <<",\"actual_successful_public_plan_calls\":"<<ref_control::graph_success
             <<",\"actual_plan_threads_min\":"<<ref_control::thread_min<<",\"actual_plan_threads_max\":"<<ref_control::thread_max
--- original-immutable/build-argv.json
+++ additive-new/build-argv-v2.json
@@ -11,7 +11,7 @@
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include",
     "-I",
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control.cpp",
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v2.cpp",
     "-L",
     "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib",
     "-L",
@@ -29,8 +29,10 @@
     "-ldl",
     "-pthread",
     "-o",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control"
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v2"
   ],
   "compiler_identity": "ROOT must record real compiler and own finite build receipt before execution",
-  "native_math_source_unchanged": true
+  "native_math_source_unchanged": true,
+  "source_revision": "public_use_ref_canonical_provider_attestation_v2",
+  "generic_runner_v1_freeze_scope": "unchanged generic launcher/helper sources only; not proof that v1 CPP was compiled into the v2 binary"
 }

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/frozen-v3/BACKEND_DIAGNOSTIC_V3.patch

Original UTF-8 bytes: 19455; SHA256 `68cc55b17c9bddfa0159e2f375bacaca96568477a67706a66ba6b1fbca0d62c0`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- immutable-v2/native_ref_control_v2.cpp
+++ additive-v3/native_ref_control_v3.cpp
@@ -128,9 +128,40 @@
     const auto device=get_device(backend);checked(device!=nullptr,"actual backend has public device");
     const auto reg=get_reg(device);checked(reg!=nullptr,"actual backend has public registry");return reg;
 }
-static void set_reference(ggml_backend_t backend,bool initialization) {
+// Public metadata only, reached solely at the original pre-setter refusal.
+[[noreturn]] static void backend_identity_refused(ggml_backend_t backend,bool initialization,
+        ggml_backend_dev_t input_device,bool actual_cpu) {
+    const int saved=errno;
+    static auto backend_name=symbol<decltype(&ggml_backend_name)>(BASE,BASE_BYTES,"ggml_backend_name");
+    static auto get_device=symbol<decltype(&ggml_backend_get_device)>(BASE,BASE_BYTES,"ggml_backend_get_device");
+    static auto device_name=symbol<decltype(&ggml_backend_dev_name)>(BASE,BASE_BYTES,"ggml_backend_dev_name");
+    static auto device_type=symbol<decltype(&ggml_backend_dev_type)>(BASE,BASE_BYTES,"ggml_backend_dev_type");
+    static auto device_reg=symbol<decltype(&ggml_backend_dev_backend_reg)>(BASE,BASE_BYTES,"ggml_backend_dev_backend_reg");
+    static auto reg_name=symbol<decltype(&ggml_backend_reg_name)>(BASE,BASE_BYTES,"ggml_backend_reg_name");
+    const auto device=backend?get_device(backend):nullptr;
+    const auto reg=device?device_reg(device):nullptr;
+    const char *bn=backend?backend_name(backend):"(null)";
+    const char *dn=device?device_name(device):"(null)";
+    const char *rn=reg?reg_name(reg):"(null)";
+    const int type=device?static_cast<int>(device_type(device)):-1;
+    const bool input_known=initialization && input_device;
+    char message[4096]={};
+    const int size=std::snprintf(message,sizeof(message),
+        "backend_identity_diagnostic_v3 origin=%.64s cpu_is_cpu=%s backend=%p backend_name=%.128s "
+        "device=%p device_name=%.128s device_type=%d registry=%p registry_name=%.128s "
+        "input_device=%p input_device_known=%s input_device_equals_returned=%s "
+        "setter_called_for_this_attempt=false compute_continued_after_refusal=false errno=%d",
+        initialization?"public_dev_init":"existing_public_abort_setter_callback",actual_cpu?"true":"false",
+        static_cast<void *>(backend),bn?bn:"(null)",static_cast<void *>(device),dn?dn:"(null)",type,
+        static_cast<void *>(reg),rn?rn:"(null)",static_cast<void *>(input_device),input_known?"true":"false",
+        input_known && input_device==device?"true":"false",saved);
+    if(size<0 || size>=static_cast<int>(sizeof(message)))fatal("backend_identity_diagnostic_v3 formatting refused");
+    errno=saved;fatal(message); // SAME exit125; never skip/set/disable or continue computation.
+}
+static void set_reference(ggml_backend_t backend,bool initialization,ggml_backend_dev_t input_device=nullptr) {
     static IsCpu is_cpu=symbol<IsCpu>(CPU,CPU_BYTES,"ggml_backend_is_cpu");
-    checked(backend && is_cpu(backend),"nonCPU backend in CPU-only control");
+    const bool actual_cpu=backend && is_cpu(backend);
+    if(!actual_cpu)backend_identity_refused(backend,initialization,input_device,actual_cpu);
     void *raw=lookup()(registry(backend),"ggml_backend_cpu_set_use_ref");
     attest(raw,CPU,CPU_BYTES,"ggml_backend_cpu_set_use_ref");
     reinterpret_cast<Ref>(raw)(backend,true); // Sole intentional policy change.
@@ -155,7 +186,7 @@
     static ref_control::Init original=ref_control::symbol<ref_control::Init>(ref_control::BASE,
         ref_control::BASE_BYTES,"ggml_backend_dev_init");
     errno=saved;auto backend=original(device,params);const int after=errno;
-    if(ref_control::armed){ref_control::checked(backend!=nullptr,"native backend initialization failed");ref_control::set_reference(backend,true);}
+    if(ref_control::armed){ref_control::checked(backend!=nullptr,"native backend initialization failed");ref_control::set_reference(backend,true,device);}
     errno=after;return backend;
 }
 extern "C" void *ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg,const char *name) {
@@ -259,7 +290,7 @@
         write_bytes(folder/"use-ref-process-status.txt",status_text.data(),static_cast<size_t>(status_count));
         std::ofstream report(folder/"use-ref-control.json");require(bool(report),"control report open");
         report<<"{\"status\":\"PUBLIC_USE_REF_CONTROL_CAPTURE_COMPARISON_REQUIRED\","
-            <<"\"control_source_revision\":\"public_use_ref_canonical_provider_attestation_v2\","
+            <<"\"control_source_revision\":\"public_use_ref_backend_identity_diagnostic_v3\","
             <<"\"provider_policy\":\"same_canonical_path_regular_device_inode_length_exact_export\",\"CPU_backend_count\":2,"
             <<"\"public_setter_calls\":4,\"existing_abort_setter_calls\":2,\"actual_public_plan_calls\":"<<ref_control::graph_calls
             <<",\"actual_successful_public_plan_calls\":"<<ref_control::graph_success
--- immutable-v2/build-argv-v2.json
+++ additive-v3/build-argv-v3.json
@@ -11,7 +11,7 @@
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/include",
     "-I",
     "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/include",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v2.cpp",
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v3.cpp",
     "-L",
     "/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib",
     "-L",
@@ -29,10 +29,11 @@
     "-ldl",
     "-pthread",
     "-o",
-    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v2"
+    ".agents/runs/numerics-and-context-use-ref-bulk-control-20261020/native_ref_control_v3"
   ],
   "compiler_identity": "ROOT must record real compiler and own finite build receipt before execution",
   "native_math_source_unchanged": true,
-  "source_revision": "public_use_ref_canonical_provider_attestation_v2",
-  "generic_runner_v1_freeze_scope": "unchanged generic launcher/helper sources only; not proof that v1 CPP was compiled into the v2 binary"
+  "source_revision": "public_use_ref_backend_identity_diagnostic_v3",
+  "generic_runner_v1_freeze_scope": "unchanged generic helper only; actual V3 CPP/binary/build binding separately mandatory",
+  "scope": "DIAGNOSTIC_ONLY same pre-setter fatal125; no nonCPU policy allowance"
 }
--- /dev/null
+++ additive-v3/bind_v3.py
@@ -0,0 +1,98 @@
+"""ROOT ONLY: bind actual V3 compile/run receipts; no model/operator calls."""
+import argparse
+from pathlib import Path
+import re
+
+from common import HERE, REPO, digest, jread, require, write
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--manifest-sha256", required=True)
+    parser.add_argument("--build", required=True, type=Path)
+    parser.add_argument("--run", required=True, type=Path)
+    parser.add_argument("--new-binding", required=True, type=Path)
+    args = parser.parse_args()
+    require(re.fullmatch("[a-f0-9]{64}", args.manifest_sha256) is not None,
+            "independently reviewed manifest SHA")
+    path = HERE / "freeze-manifest-v3.json"
+    require(digest(path)["sha256"] == args.manifest_sha256, "exact reviewed V3 freeze")
+    manifest = jread(path)
+    for version, directory in (("original_freeze", "frozen-prepared"), ("v2_freeze", "frozen-v2")):
+        pin = manifest[version]
+        parent = HERE / pin["file"]
+        require(digest(parent) == {key: pin[key] for key in ("bytes", "sha256")}, "unchanged prior manifest")
+        for name, expected in jread(parent)["files"].items():
+            require(digest(HERE / name) == digest(HERE / directory / name) == expected,
+                    "all original source and frozen copies unchanged: " + name)
+    for name, expected in manifest["files"].items():
+        require(digest(HERE / name) == digest(HERE / "frozen-v3" / name) == expected,
+                "new frozen source unchanged: " + name)
+    for row in manifest["preserved_failed_metadata"]:
+        require(digest(row["path"]) == {key: row[key] for key in ("bytes", "sha256")},
+                "failed case evidence unchanged")
+    build, run = args.build.resolve(strict=True), args.run.resolve(strict=True)
+    destination = args.new_binding.resolve()
+    require(build.is_relative_to(HERE) and run.is_relative_to(HERE)
+            and destination.is_relative_to(HERE) and not destination.exists(),
+            "owned actual build/run and fresh binding only")
+    requested, result = jread(build / "requested.json"), jread(build / "result.json")
+    compiler = manifest["root_actual_compiler_pin"]
+    prepared = jread(HERE / "build-argv-v3.json")
+    actual_inputs = {row["path"]: digest(row["path"]) for row in manifest["compile_inputs"]}
+    require(actual_inputs == {row["path"]: {key: row[key] for key in ("bytes", "sha256")}
+                              for row in manifest["compile_inputs"]}, "actual exact compiler inputs")
+    for receipt in (requested, result):
+        require(receipt["compiler"] == compiler["path"]
+                and receipt["compiler_sha256"] == compiler["sha256"]
+                and digest(compiler["path"])["sha256"] == compiler["sha256"], "actual pinned compiler")
+        require(receipt["cwd"] == str(REPO)
+                and receipt["argv"] == [compiler["path"], *prepared["argv"][1:]]
+                and receipt["source_freeze_sha256"] == args.manifest_sha256
+                and receipt["source_inputs"] == actual_inputs
+                and receipt["runtime_or_math_qualified"] is False,
+                "actual relative-argv CWD/new CPP+included source pre/post build binding")
+    require(type(result["exit_code"]) is int and result["exit_code"] == 0, "actual successful fresh build")
+    binary = (HERE / "native_ref_control_v3").resolve(strict=True)
+    identity = digest(binary)
+    require(result["binary_path"] == str(binary)
+            and identity == {"bytes": result["binary_bytes"], "sha256": result["binary_sha256"]},
+            "exact actual new binary output")
+    invocation = jread(run / "invocation.json")
+    require(invocation["argv"][0] == str(binary) and invocation["inputs"][str(binary)] == identity,
+            "actual generic runner executed this newly compiled V3 binary")
+    require(invocation["freeze_manifest"] == digest(HERE / "freeze-manifest.json"),
+            "generic V1 helper receipt remains separate from V3 CPP build evidence")
+    native_log = run / "native.log"
+    require(native_log.is_file() and not native_log.is_symlink()
+            and 0 < native_log.stat().st_size == invocation["log_bytes"] <= 8 * 1024 * 1024,
+            "complete bounded original stdout/stderr diagnostic log")
+    log_identity = digest(native_log)
+    control_path = run / "capture/use-ref-control.json"
+    control = None
+    if control_path.exists():
+        actual = jread(control_path)
+        require(actual["control_source_revision"] == "public_use_ref_backend_identity_diagnostic_v3"
+                and actual["provider_policy"] == "same_canonical_path_regular_device_inode_length_exact_export",
+                "actual V3 source marker with unchanged V2 attestation policy")
+        control = digest(control_path)
+    else:
+        require(invocation["status"] == "UNQUALIFIED", "successful run missing actual V3 control report")
+    write(destination, {
+        "status": "SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE",
+        "new_source_freeze": digest(path), "generic_V1_helper_freeze": digest(HERE / "freeze-manifest.json"),
+        "V2_attestation_freeze": digest(HERE / "freeze-manifest-v2.json"),
+        "new_cpp": {"path": str(HERE / "native_ref_control_v3.cpp"), **digest(HERE / "native_ref_control_v3.cpp")},
+        "included_unchanged_token_loop": digest(HERE / "native_raw_original.cpp"),
+        "build_requested": digest(build / "requested.json"), "build_result": digest(build / "result.json"),
+        "actual_binary": {"path": str(binary), **identity}, "actual_invocation": digest(run / "invocation.json"),
+        "actual_diagnostic_log": {"path": str(native_log), **log_identity},
+        "actual_invocation_status": invocation["status"], "actual_control_report": control,
+        "diagnostic_only_no_nonCPU_policy_allowance": True, "runtime_or_math_qualified": False,
+        "next": "read actual bounded backend-identity refusal; do not infer or enable ACCEL policy",
+    })
+    print("SOURCE_BUILD_INVOCATION_BOUND_NOT_ARITHMETIC_ACCEPTANCE")
+
+
+if __name__ == "__main__":
+    main()
--- /dev/null
+++ additive-v3/SOURCE_AUDIT_V3.md
@@ -0,0 +1,111 @@
+# Additive V3 backend identity diagnostic — ROOT only
+
+Owner: baseline_review, root-authorized additive files only. All original14,
+V2 source/manifest/frozen copies and both failed cases remain unchanged.
+No build, test, model/operator, container/GPU runtime or large payload hash by
+author. No source/profile/math/StateRoot/long-context acceptance.
+
+## Actual refusal and source inference
+
+Fresh `root-0.8b-8192-ref64-2` used actual V2 binary42b98af8… and failed125
+after context geometry n_ctx8448/n_outputs64, before any accepted capture:
+records.jsonl is0B, only raw teacher file exists, no public setter/plan trace.
+Its log explicitly loads BLAS and Haswell CPU. Cleanup completed without errors.
+
+Pinned d812350 source:
+
+- llama-context.cpp334–366 initializes model devices, then every ACCEL backend
+  (explicit example BLAS), then CPU. n_gpu_layers0/offload flags do not suppress
+  the ACCEL construction loop.
+- ggml-blas.cpp214/322/478 public metadata names BLAS;356 returns device type
+  ACCEL;281–307 constructs its distinct BLAS backend GUID.
+- ggml-cpu.cpp249 recognizes CPU GUID specifically. Thus BLAS legitimately
+  returns false. Current control applies set_reference to EVERY dev_init return.
+- CPU_REPACK is an extra CPU buffer type (repack.cpp5162/5238), not a separately
+  initialized backend. It cannot itself identify this failed backend predicate.
+
+BLAS is a source-supported hypothesis, not actual backend identity. No v2
+diagnostic printed the returned backend/device/registry; do not close the cause.
+
+## V3 observation only
+
+`native_ref_control_v3.cpp` adds one fatal metadata helper, only on the original
+failed backend&&is_cpu check. Public BASE symbol lookup retains exact V2
+NOLOAD/canonical regular-file dev/inode/length/export attestation. No new DSO,
+RTLD_GLOBAL, fallback, private structure read or backend/provider replacement.
+
+The <=4095B named diagnostic prints origin, actual is_cpu boolean, opaque
+backend/device/registry identities, public names/device type, original input
+device identity, whether it is known and equals the returned backend device.
+Public getters resolve from the original BASE; existing predicate remains the
+same exact Haswell public is_cpu. Known actual backend is read through public
+APIs only. Null metadata is marked rather than dereferenced.
+
+Then SAME exit125, BEFORE setter invocation for that attempt. It never skips or
+disables a backend, never sets a CPU flag on a nonCPU object, and never proceeds
+to computation after refusal. Successful CPU predicate path retains the same
+setter/callback forwarding/plan checks/token loop; only the source revision
+marker changes for build/run binding. No new accelerator policy is implemented.
+
+## ROOT actual compile and invocation binding
+
+Build V3 with build-argv-v3.json argv from MAIN repository CWD using the exact
+pinned compiler. Preserve old binaries/build directories. Fresh build receipts
+requested.json and result.json must both include actual `cwd`, `compiler`,
+`compiler_sha256`, `argv`, reviewed V3 `source_freeze_sha256`, exact `source_inputs`
+mapping from manifest compile_inputs, and runtime_or_math_qualified=false.
+Record source_inputs before/after compilation. Result also records real exit,
+binary_path/bytes/SHA. This closes the relative-argv CWD gap explicitly.
+
+Use unchanged run_ref_control.py with new binary path/digest and a fresh output
+name. The generic invocation correctly pins V1 launcher/helpers; it does not
+claim that V1 CPP was compiled into this V3 binary. After actual invocation:
+
+```sh
+python3 .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/bind_v3.py \
+  --manifest-sha256 REVIEWED_V3_SHA \
+  --build ABS_FRESH_BUILD --run ABS_FRESH_CASE --new-binding ABS_FRESH_BINDING_JSON
+```
+
+Recipe authenticates all prior/new frozen sources, failed metadata, compiler,
+exact new CPP/includes/CWD/argv/build exit/binary and actual runner inputs. A
+failed diagnostic may bind source/build/invocation while remaining UNQUALIFIED.
+Actual diagnostic interpretation is ROOT review; it is not a numeric verdict.
+
+## Future policy feasibility only — NOT implemented
+
+Pinned public scheduler graph routes:
+
+- ggml-backend.cpp595: sync graph_compute delegates graph_compute_async.
+- 601–604: graph_compute_async dispatches the selected backend iface directly.
+- 1955/1977: both ordinary split and observed graph-view scheduler paths call
+  that public async entry. 588–592: public graph_plan_compute is another direct
+  iface route; checking only inner CPU ggml_graph_compute does not cover it.
+- CPU cpp139–153 stores source cplan.use_ref at plan creation;162–190 both CPU
+  plan/ordinary callbacks converge on ggml_graph_compute, whose current wrapper
+  checks actual cplan.use_ref before/after. Old plans must not be silently fixed.
+- BLAS cpp226–263 directly invokes CBLAS and bypasses inner CPU graph_compute.
+
+If actual identity later proves a genuine ACCEL is initialized but unused,
+unmodified construction could be recorded separately while a NEW reviewed
+control rejects every nonCPU public async/plan compute BEFORE delegation.
+Unknown/foreign backends remain cold refusals. Known CPU public calls could
+carry a thread-local scope through original forwarding, bind actual inner CPU
+plan observations to that backend, and require every original delegated plan
+use_ref=true. Nested sync→async must count one compute rather than two math
+invocations; capture every entry/return/success and all initialization/retirement.
+Opaque plan handles need public create/free association if plan route is used;
+no private plan/backend layout reinterpretation. Existing two-CPU counts cannot
+be relabeled as all-backend coverage; constructed ACCELs require explicit records.
+
+This is source feasibility, not compiled route closure. ROOT must prove actual
+public interposition paths/DSO closure for both plain and observed modes before
+allowing construction to continue. Direct/internal bypass or any unmatched
+compute must refuse. Numerical all730/fullV/all42 and observation guards remain.
+
+Crucially, op_offload=false does NOT prove ACCEL unused: backend.cpp1028 chooses
+the highest priority compatible buffer/op; pass3 around1380 upgrades compatible
+buffer nodes independently of op_offload. BLAS supports host-buffer contiguous
+F32 GEMM with all relevant dimensions>=32 (blas.cpp397–459). Bulk64 QK/AV may
+qualify. A future guard may correctly refuse; do not disable BLAS, alter buffers,
+or silently downgrade math merely to make the control pass.

````

