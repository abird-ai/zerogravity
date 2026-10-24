# Draft Patches — volume 16

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 253195; SHA256 `25734119722b61ca58ff4a77dc5b2d421d5155e114471d0e1bb190973210fb42`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/Cargo.toml
@@ -0,0 +1,24 @@
+[package]
+name = "qwen38-window-pure-roles-replay"
+version = "0.0.0"
+edition = "2024"
+publish = false
+
+[workspace]
+
+[dependencies]
+zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts", features = ["cache"] }
+zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core" }
+zerogravity-cpu = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu", features = ["cache"] }
+zerogravity-model-qwen35 = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35", features = ["cpu", "original-hf-binding"] }
+serde_json = "=1.0.151"
+sha2 = "=0.10.9"
+libc = "=0.2.190"
+
+[profile.release]
+opt-level = 3
+codegen-units = 1
+
+[[bin]]
+name = "qwen38-window-pure-roles-replay"
+path = "main.rs"
--- /dev/null
+++ b/ISOLATED_ROOT.md
@@ -0,0 +1,113 @@
+Author: spine_review
+
+- Additive source-only relocation. MAIN, original qualification/V2/public oracle
+  and isolated ROOT worktree unchanged by preparation.
+- Target ROOT521 source freeze7 `036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615`.
+  The prior actual selected strict +237 focused checks are pinned; no new native
+  pure-wide execution is claimed.
+- Original thirteen-file base `16809776089e1c3063629a0e0df556512a66b9ccd04d6812d56e47b744f4cb40`;
+  metadata V2 `e9581cf982e3c83fa85413a4eb707afef001290c6137563a34a9c679a54514e6`.
+  Every original live/frozen file and parent pin remains authenticated.
+- Four absolute Cargo crate roots relocate. Original main.rs/factory.rs,
+  source-plan.json, actual7058B source-pin4c74fa28, public capture/validator,
+  raw serializer and 53ee owner are byte-identical. support.py changes only its
+  frozen() source dispatch. run_native.py changes only source/oracle-path build
+  attestation dispatch and safe record-only TERM/INT observation.
+- Native numerical code/arguments/512MiB AS/4CPU/1800s wall/4MiB logs/128MiB
+  configured host ledger, source/admission/pressure/cancel/retention/full-word
+  comparisons/final0 remain exact original bytes. No source model copy, fallback,
+  new engine, extra projection/state operator or higher cap.
+
+## Original oracle and source exceptions
+
+- Use the existing independently accepted original directory, not a copied or
+  regenerated archive:
+  `/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/root-reference-1`.
+- Its original validation/manifest/invocation metadata is pinned. Original
+  validation retains freeze16809776; the relocated native bundle has its own
+  preparation identity. These identities are explicitly separate, never retagged.
+- Public original32-case/16-role input/output reference:881856 words,1763712B.
+  Native unchanged main.rs verifies each complete actual payload against its
+  original digest and all words. This is independent public expected data; source
+  attestation alone is neither compiler nor numerical evidence.
+- Original 8 implementation +6 dependency assertions remain explicit. Seven
+  children are byte-identical: pure.rs, window/pure.rs, weights.rs,
+  window_projection.rs, norm/ordinary.rs, artifact tensor_window.rs,
+  original_binding.rs. Seven shared/test children use exact reviewed ROOT
+  composition: forward.rs, host_math.rs, operator_slice.rs,
+  operator_slice_window_roles_tests.rs, operator_slice_window_tests.rs,
+  original_binding/window.rs and core program.rs. Their source-only line deltas
+  are fully bound in relocation.json; this is not a claim of semantic equivalence
+  from a diff. ROOT521, composition journals and recorded corrections close the
+  intended implementation identity.
+- Eight original ROOT correction exceptions2–7 remain exact: coefficient output
+  size adapter; CPU/model coefficient Vec→array test spelling; H48 fixture budget/
+  visibility/import/schedule/lifetime; closed Embedding role negative; distinct
+  fresh SourceIdentity assertion. CPU row math and ordinary norm stay byte-identical.
+
+## Coordinator sequence
+
+1. Review this bundle and source-only `REVIEW.diff`. Run the original four and new
+   thirteen source-free fixture methods only after the timing hold:
+   `python3 -m unittest -v test_contracts.py test_relocation.py` from this directory.
+   The author parsed Python AST only; no imports/tests/runtime were run.
+2. Review ROOT521 source closure and build this local Cargo.toml using ROOT's
+   existing controlled/offline packages and serialized target owner. Four crate
+   paths point at the isolated ROOT worktree; MAIN must remain untouched.
+3. Preserve actual compile argv/log/exit/binary/effective-lock proof. Prepare the
+   isolated build attestation below after the actual build. All source files must
+   still match ROOT036c521. A fresh local Cargo.lock is an additional actual input,
+   and external package normalization against ROOT's accepted lock remains the
+   coordinator's build gate. Caller labels/hashes alone do not prove compilation.
+4. Execute the relocated run_native.py below with the original accepted oracle.
+   The original validator is preserved byte-identically, but rerunning the copied
+   public capture/validator is unnecessary: it would be a fresh capture with a
+   separate preparation identity. Never edit or relabel original accepted data.
+5. Reconcile only private pure-operator qualification after the actual881856-word
+   result, cancellation/real-pressure/retained3035648B/final0/cleanup checks pass.
+
+```sh
+python3 run_native.py --binary /ABS/ROOT/RELEASE/BINARY \
+  --binary-sha256 ROOT_ACTUAL_BINARY_SHA \
+  --reference-directory /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/root-reference-1 \
+  --build-attestation /ABS/ROOT/ACTUAL_BUILD_ATTESTATION.json \
+  --new-output /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/root-native-1
+```
+
+Required actual build receipt fields:
+
+```json
+{
+  "status": "ROOT_COMPILED_ISOLATED_ORIGINAL27B_PURE_ROLES_ONLY",
+  "build_exit_code": 0,
+  "binary": "/ABS/ROOT/RELEASE/BINARY",
+  "binary_sha256": "ACTUAL_BINARY_SHA",
+  "prepared_manifest_sha256": "THIS_PREPARED_MANIFEST_SHA",
+  "qualification_manifest_sha256": "16809776089e1c3063629a0e0df556512a66b9ccd04d6812d56e47b744f4cb40",
+  "qualification_metadata_followup_manifest_sha256": "e9581cf982e3c83fa85413a4eb707afef001290c6137563a34a9c679a54514e6",
+  "root_source_freeze_sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615",
+  "root_worktree": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
+  "qualification_directory": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020",
+  "MAIN_merged": false,
+  "full_model_StateRoot_GPU_or_performance_qualified": false,
+  "files": ["exact absolute path/bytes/sha256 source rows below"]
+}
+```
+
+`files` = all521 final ROOT source rows with absolute ROOT paths, plus this
+directory's exact Cargo.toml/main.rs/factory.rs/source-pin.json/source-plan.json;
+plus local Cargo.lock if created. Exactly526 or527 unique entries. The original
+V2 four compile bindings remain authenticated; the fifth is necessary because
+unchanged main.rs actually `include_bytes!`s source-plan.json. Missing, duplicate,
+foreign, MAIN/old status and unexpected source deltas refuse cold.
+
+TERM+INT use exact accepted00d1efc1 record-only guard; no asynchronous exception
+across Popen assignment or retirement. `cancel.observe` runs only after original
+child/log handoff, restores original log.drain before unchanged53ee cleanup.
+Cleanup still runs before synchronous cancellation is reported. No GPU/provider
+stub, mocked process ownership or hidden retry is introduced.
+
+Scope remains original16 pure roles/32 external probes only. Some role inputs are
+literal BF16 external patterns, not authenticated trained model producers. No
+full GDN/exp/initial/cache/112 StateRoots/session/full27B inference, GPU eligibility
+or throughput claim.
--- /dev/null
+++ b/REVIEW.diff
@@ -0,0 +1,155 @@
+--- original-v2/Cargo.toml
++++ isolated/Cargo.toml
+@@ -7,10 +7,10 @@
+ [workspace]
+ 
+ [dependencies]
+-zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-artifacts", features = ["cache"] }
+-zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core" }
+-zerogravity-cpu = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu", features = ["cache"] }
+-zerogravity-model-qwen35 = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35", features = ["cpu", "original-hf-binding"] }
++zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts", features = ["cache"] }
++zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core" }
++zerogravity-cpu = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu", features = ["cache"] }
++zerogravity-model-qwen35 = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35", features = ["cpu", "original-hf-binding"] }
+ serde_json = "=1.0.151"
+ sha2 = "=0.10.9"
+ libc = "=0.2.190"
+--- original-v2/support.py
++++ isolated/support.py
+@@ -111,7 +111,5 @@
+             'native_ResourceArbiter_admission_qualified':False}
+ 
+ def frozen():
+-    manifest=metadata(HERE/'prepared-manifest.json')
+-    for row in manifest['files']:read(HERE/row['file'],pin=row)
+-    for row in manifest['parent_metadata']:read(row['path'],pin=row)
+-    return sha(HERE/'prepared-manifest.json')
++    from relocated_source import frozen as checked_frozen
++    return checked_frozen(HERE, metadata, read, require, sha)
+--- original-v2/run_native.py
++++ isolated/run_native.py
+@@ -8,45 +8,30 @@
+ import signal
+ import subprocess
+ import time
+-from support import HERE,frozen,metadata,require,sha,write
++from support import HERE,frozen,metadata,read,require,sha,write
++from relocated_source import implementation, reference
++from sigterm_cancellation import SigtermCancellation
+ 
+ 
+-def main():
++def main(cancel):
+     parser=argparse.ArgumentParser()
+     parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--binary-sha256',required=True)
+     parser.add_argument('--reference-directory',type=Path,required=True)
+     parser.add_argument('--build-attestation',type=Path,required=True);parser.add_argument('--new-output',type=Path,required=True)
+     args=parser.parse_args()
++    cancel.check()
+     freeze=frozen()
+     require(args.binary.is_absolute()and not args.binary.is_symlink()and sha(args.binary)==args.binary_sha256,
+             'actual ROOT-built immutable same-engine executable')
++    source_attestation=implementation(HERE,args.build_attestation,metadata,read,require,sha)
+     build=metadata(args.build_attestation)
+-    require(build['status']=='ROOT_COMPILED_ORIGINAL27B_PURE_ROLES_ONLY'and build['build_exit_code']==0 and
+-            build['binary_sha256']==args.binary_sha256 and Path(build['binary']).resolve()==args.binary.resolve()and
+-            build['candidate_manifest_sha256']=='0f9f2a8ff4ff28f5f3284063aecff35508ca61cdd083dac454400e7aec896d3e'and
+-            build['exact_hidden_followup_manifest_sha256']=='65cd436471e26679d4ff596538fda31c9b1c6f28cef6d9a55f7c99e9d28ca82b'and
+-            build['completed_cancel_followup_manifest_sha256']=='7759599f51215de539167a3f71b0efb39aace80fcf817b3433fbe5eba6dfdc6c',
+-            'ROOT actual build/candidate/followup binding; marker alone is not compilation proof')
+-    actual_files=build['actual_source_files'];require(len(actual_files)==8,'all eight actual implementation source paths')
+-    worktree=Path(build['root_worktree']);require(worktree.is_absolute()and worktree.is_dir(),'explicit ROOT build worktree')
+-    expected=metadata(HERE.parent/'freeze-manifest.json')['files']
+-    fixes={row['path']:row for row in metadata(HERE.parent/'followup-v2/freeze-manifest.json')['files']}
+-    fixes.update({row['path']:row for row in metadata(HERE.parent/'followup-v3/freeze-manifest.json')['files']})
+-    for row in expected:
+-        pin=fixes.get(row['path'],row);actual=[v for v in actual_files if v['path']==row['path']]
+-        require(len(actual)==1 and actual[0]['sha256']==pin['sha256']and actual[0]['bytes']==pin['bytes'],
+-                'actual compiled source attestation matches exact reviewed implementation')
+-        require((worktree/row['path']).stat().st_size==pin['bytes']and sha(worktree/row['path'])==pin['sha256'],
+-                'actual ROOT implementation source bytes still exact')
+-    for pin in metadata(HERE.parent/'freeze-manifest.json')['unchanged_dependency_source_pins']:
+-        require((worktree/pin['path']).stat().st_size==pin['bytes']and sha(worktree/pin['path'])==pin['sha256'],
+-                'unchanged original CPU math/core/source closure bytes')
+-    require(args.reference_directory.is_absolute()and args.reference_directory.parent.resolve()==HERE and
+-            args.new_output.is_absolute()and args.new_output.parent.resolve()==HERE,'owned reference/native outputs')
+-    validation=metadata(args.reference_directory/'validation.json')
+-    require(validation['status']=='PASS_ORIGINAL27B_PURE_OPERATOR_REFERENCE_ONLY'and
+-            validation['freeze_sha256']==freeze and validation['manifest_sha256']==sha(args.reference_directory/'captured/manifest.json'),
+-            'independently validated genuine public raw reference first')
++    require(build['binary_sha256']==args.binary_sha256 and
++            Path(build['binary']).resolve()==args.binary.resolve(),
++            'ROOT actual isolated executable binding; source markers alone are not compiler proof')
++    require(args.new_output.is_absolute()and args.new_output.parent.resolve()==HERE,
++            'owned relocated native output only')
++    validation=reference(HERE,args.reference_directory,metadata,read,require)
++    worktree=Path(build['root_worktree'])
+     cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus)
+     args.new_output.mkdir(mode=0o700)
+     require(sha(HERE/'owned_container_v4.py')=='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
+@@ -57,7 +42,9 @@
+     child=log=None;failure=None;started=time.monotonic()
+     receipt={'status':'UNQUALIFIED','freeze_sha256':freeze,'binary_sha256':args.binary_sha256,
+              'build_attestation_sha256':sha(args.build_attestation),'expected_cpus':cpus,
+-             'scope':'same-engine private original27B pure roles only; no full-model/state/GPU'}
++             'scope':'same-engine private original27B pure roles only; no full-model/state/GPU',
++             'MAIN_merged':False,'root_source_freeze_sha256':build['root_source_freeze_sha256'],
++             'original_oracle_preparation_sha256':validation['freeze_sha256']}
+     try:
+         with os.fdopen(os.dup(fd),'rb')as opened:
+             import hashlib
+@@ -73,15 +60,14 @@
+             resource.setrlimit(resource.RLIMIT_CORE,(0,0))
+             resource.setrlimit(resource.RLIMIT_CPU,(1801,1801))
+             resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,4*1024**2))
+-        def cancelled(_signal,_frame):raise InterruptedError('owned native pure replay cancelled')
+-        signal.signal(signal.SIGTERM,cancelled)
++        cancel.check() # Record-only signals cannot interrupt original Popen ownership assignment.
+         try:
+             child=subprocess.Popen(argv,executable=f'/proc/self/fd/{fd}',pass_fds=(fd,),
+                 stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
+                 env=environment,start_new_session=True,preexec_fn=limits)
+             receipt['owned_cli_pid']=child.pid
+             log=helper.Log(child.stdout,args.new_output/'execution.log')
+-            receipt['exit_code']=log.observe(child,1800)
++            receipt['exit_code']=cancel.observe(log,child,1800)
+         except BaseException as error:failure=error
+         finally:
+             receipt['cleanup_errors']=[]
+@@ -95,6 +81,7 @@
+             elif child is not None and child.stdout is not None:
+                 try:child.stdout.close()
+                 except BaseException as error:receipt['cleanup_errors'].append('uninitialized pipe:'+repr(error))
++        cancel.check() # Original cleanup completed before synchronous cancellation is raised.
+         require(failure is None and receipt.get('exit_code')==0 and receipt['cleanup_errors']==[],
+                 'owned native replay/cleanup failed; no automatic retry')
+         report=metadata(args.new_output/'report.json')
+@@ -102,15 +89,14 @@
+                 report['final_reserved']==0 and report['final_fenced']is False and
+                 report['proof']['result']['compared_words']==validation['words'],
+                 'all actual native raw outputs/input words exact and final original ledger0')
+-        for row in expected:
+-            pin=fixes.get(row['path'],row)
+-            require(sha(worktree/row['path'])==pin['sha256'],'actual ROOT implementation unchanged after native replay')
+-        for pin in metadata(HERE.parent/'freeze-manifest.json')['unchanged_dependency_source_pins']:
+-            require(sha(worktree/pin['path'])==pin['sha256'],'original math/core/source dependencies unchanged after native replay')
++        require(implementation(HERE,args.build_attestation,metadata,read,require,sha)==source_attestation and
++                reference(HERE,args.reference_directory,metadata,read,require)==validation,
++                'whole isolated ROOT source/build and accepted oracle remain exact')
+         require(frozen()==freeze and sha(args.binary)==args.binary_sha256 and
+                 sha(args.reference_directory/'captured/manifest.json')==manifest_sha and
+                 sha(args.reference_directory/'validation.json')==validation_sha,
+                 'all preparation/binary/independent raw reference unchanged')
++        cancel.check()
+         receipt.update(status='PASS_OWNED_ORIGINAL27B_PURE_ROLE_REPLAY_ONLY',report_sha256=sha(args.new_output/'report.json'))
+     except BaseException as error:
+         if failure is None:failure=error
+@@ -124,4 +110,6 @@
+     require(receipt['status']=='PASS_OWNED_ORIGINAL27B_PURE_ROLE_REPLAY_ONLY','native proof unqualified; retained original attempt')
+ 
+ 
+-if __name__=='__main__':main()
++if __name__=='__main__':
++    with SigtermCancellation() as cancellation:
++        main(cancellation)
--- /dev/null
+++ b/ROOT.md
@@ -0,0 +1,81 @@
+# ROOT-only genuine original27B pure-role gate
+
+Author: spine_review. Prepared source only; no fixtures/build/model/operator run.
+
+## Scope and provenance
+
+- Original snapshot98cea13c/revision1d4bf0f2, all18 acquired weight shards
+  55,563,006,776B; actual index1199/ordinary-text851 closure. Reused accepted
+  `factory.rs` byte-identical from the original27B three-operator replay.
+- Public PyTorch original BF16 F.linear: GDN QKV/Z/A/B/out; attention full Q+gate,
+  K/V/out; MLP gate/up/down; real untied lm_head. Three original RMSNorm calls:
+  input0, post-attention0 and final. No nonlinearities, residuals, positions,
+  caches, token/layer traversal or state publishers are added.
+- Two external teacher probes11/353. Projection hidden5120 inputs are exact
+  accepted archived original InputNorm0 BF16 words; norm inputs use accepted
+  original embedding words. Only the GDN input prefix has its actual trained
+  producer here. Other hidden-role inputs remain external probes. Nonhidden
+  widths6144/17408 use explicitly literal finite nonzero BF16 patterns, never
+  reconstructed or declared trained producer activations.
+- Sixteen actual source/header role shapes, unchanged raw BF16, three original
+  retained shard FDs. Each parameter mapping is exact page-aligned range plus
+  leading alignment, never a whole shard/full55GB mapping. Original head alone
+  is about2.543GB virtual. All parameter FD/map/tensor owners survive to exit;
+  full source/hash/version/readset checks before/after actual public operators.
+- Public capture calls each genuine operator twice, all input/output bits
+  identical. This is a repeat control, no newly invented instrumentation route.
+  Pinned original image/Torch2.13cf30153/Transformers5.17 source, AVX2/threads4;
+  CPU loader/activation profile is distinct from vLLM CUDA/FLA.
+- Exact parameter VA3,519,025,784B +6GiB runtime VA estimate +2GiB operator
+  estimate +4MiB capture <12GiB RLIMIT_AS. RAM/totalRAM+swap12GiB,CPU quota4,
+  noGPU/network, read-only source/probe/image, finite600s child/2400s parent,
+  32MiB whole artifacts. Estimates/cgroup containment do not prove complete
+  physical host/provider admission. Mapping VA is not resident RSS.
+
+## Execution, after review and timing hold
+
+1. ROOT run `python3 test_contracts.py` here (four prepared stdlib bounds tests).
+   Run the separate original27B GDN V3 actual-runtime raw serializer fixtures
+   first; tensor-bit helper body is the same equal-size integer view/C-order
+   transport with only its local support import changed.
+2. ROOT `python3 run_reference.py --new-output /ABSOLUTE/QUAL/root-reference-1`.
+   Actual full55GB source hash scans happen only under ROOT's quiet scheduling.
+3. ROOT `python3 validate_reference.py --output /ABSOLUTE/QUAL/root-reference-1`.
+   Independent stdlib full raw/header/readset/teacher/count/lifetime validator.
+   All32 cases/all inputs/all output words must pass; no tolerance.
+4. ROOT merges reviewed role candidate0f9 plus exact-hidden65cd and completed-cancel7759 followups, runs
+   strict/focused/default/source-window regressions; builds this Cargo.toml only
+   under its one-coordinator/offline actual-source isolation. No author Cargo.
+5. ROOT writes actual build attestation (not fabricated by these preparations):
+   status `ROOT_COMPILED_ORIGINAL27B_PURE_ROLES_ONLY`, actual binary/path/hash,
+   build_exit_code0, explicit absolute root_worktree, actual_source_files eight
+   path/bytes/SHA entries matching original+followup source, candidate_manifest
+   0f9f2a8f…,exact_hidden_followup_manifest65cd4364…,completed_cancel_followup_manifest7759599f…. Preserve actual compile
+   command/log/effective lock/artifact-source proof. These labels/caller hashes
+   alone are not independent evidence of compilation or correct arithmetic.
+6. ROOT `python3 run_native.py --binary ABS --binary-sha256 SHA \
+   --reference-directory /ABSOLUTE/QUAL/root-reference-1 \
+   --build-attestation ABS --new-output /ABSOLUTE/QUAL/root-native-1`.
+
+## Native proof boundary
+
+- SAME public OriginalWindowOperators pure APIs; SAME accepted source factory,
+  SourceWeights windows, HostMath/CPU64/RNE/ordinary norm. No replacement engine.
+- Full32-case input/output bit comparisons. Full248320-row untied head and
+ 17408-wide down included; every signed-zero bit is compared without tolerance.
+- Real configured original ResourceArbiter128MiB host domain, source metadata
+ 64MiB, qualifier-control8MiB, model metadata4MiB, actual per-op output/scratch
+  and row-window data+offset tickets. No claim128MiB is physical free host RAM.
+- Actual pre-cancel and real pool exhaustion refuse; every temporary owner
+  releases, no device fence. Last actual head result holds source/input/output
+  leases after caller/source drop; final all payload/source owner drop0.
+- Native process512MiB AS,CPU4/noGPU/no provider, bounded1800s/4MiB logs.
+  Exact accepted owned-v4 generic WNOWAIT group cleanup function reused, two
+  independent KILL attempts before any reap, complete group absence/EOF.
+- PASS is only private source-authentic pure-operator numerical/transport scope.
+  No all-role trained activation producer, GDN BF16 gamma/Alog/exp/initial/cached
+  math,112 StateRoots, session/full27B model, GPU eligibility or speed claim.
+
+Source-only author validation: Python ASTs/Rustfmt/small source/metadata pin and
+header reads. No weight/DSO payload hashes, fixture/Torch/operator/model/native/
+container/build/GPU execution. All failed/interrupted ROOT attempts stay intact.
--- /dev/null
+++ b/TODO.md
@@ -0,0 +1,18 @@
+# Working checklist
+
+- Parent model-execution; ROOT reconciles shared plan only.
+- Own additive ignored relocation; preserve all original qualified source/data.
+- [x] Inspect original13-file16809776 + metadata-only V2e9581cf9.
+- [x] Pin ROOT036c521, prior actual strict/focused receipt and correction journals.
+- [x] Conserve all14 source-child assertions:7 unchanged,7 explicit ROOT composition.
+- [x] Preserve native Rust/factory/source plan/source pin/public oracle/validator.
+- [x] Prepare Cargo4-root and source-attestation dispatch only.
+- [x] Replace unsafe async TERM handler with exact record-only TERM+INT checkpoints.
+- [x] Bind all5 actual local build inputs; preserve original V2 four bindings.
+- [x] Prepare13 relocation negatives; source reads/AST only.
+- [ ] ROOT + independent source review.
+- [ ] ROOT fixtures/strict/actual standalone release compile and source attestation.
+- [ ] ROOT actual881856-word native replay/lifetime/final0, no full-model promotion.
+
+Author: no tests, imports, builds, operators/models/providers/GPU/runtime or
+archive/model/DSO/binary payload reads/hashes. Original files and MAIN untouched.
--- /dev/null
+++ b/capture.py
@@ -0,0 +1,146 @@
+"""ROOT ONLY original-weight public pure operators; never a model constructor."""
+import importlib.metadata
+import json
+import mmap
+import os
+from pathlib import Path
+import resource
+import stat
+import struct
+import sys
+from support import HERE,MODEL_SHA,TORCH_GIT,bounds,digest,fd_sha,fingerprint,metadata,read,require,sha,source_header,unique,write
+from tensor_bits import tensor_bits
+
+OWNER=None
+
+class SourceOwner:
+    def __init__(self,plan):
+        self.plan=plan;self.fds={};self.headers={};self.before={};self.maps={};self.values={};self.versions={};self.readsets={}
+        for name,pin in plan['metadata'].items():read(Path('/source')/name,pin=pin)
+        config=metadata(Path('/source/config.json'));text=config['text_config']
+        require(config['architectures']==['Qwen3_5ForConditionalGeneration']and
+                text['hidden_size']==5120 and text['intermediate_size']==17408 and
+                text['vocab_size']==248320 and text['rms_norm_eps']==1e-6 and
+                text['tie_word_embeddings']is False,'exact original27B pure-role config/epsilon')
+        index=metadata(Path('/source/model.safetensors.index.json'))['weight_map']
+        for shard,pin in plan['selected_shards'].items():
+            fd=os.open(Path('/source')/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW);self.fds[shard]=fd
+            actual=os.fstat(fd)
+            require(stat.S_ISREG(actual.st_mode)and actual.st_size==pin['file']['bytes'],
+                    'exact regular selected shard before any full scan')
+            self.before[shard]=fingerprint(actual)
+            require(fd_sha(fd,0,pin['file']['bytes'])==pin['file']['sha256'],'same original full selected shard')
+            n,raw=source_header(fd,pin)
+            require(fingerprint(os.fstat(fd))==self.before[shard],'selected FD immutable before imports')
+            self.headers[shard]=(8+n,json.loads(raw,object_pairs_hook=unique))
+        for role in plan['roles']:
+            shard=role['shard'];start,header=self.headers[shard];row=header[role['name']]
+            require(index[role['name']]==shard and row['shape']==role['shape'] and row['dtype']=='BF16' and
+                    start+row['data_offsets'][0]==role['absolute_offset'] and
+                    row['data_offsets'][1]-row['data_offsets'][0]==role['bytes'],'exact full original role/header')
+    def map(self,torch):
+        for role in self.plan['roles']:
+            name=role['name'];mapping=mmap.mmap(self.fds[role['shard']],role['mapping_bytes'],
+                access=mmap.ACCESS_COPY,offset=role['page_aligned_offset']);self.maps[name]=mapping
+            value=torch.frombuffer(mapping,dtype=torch.bfloat16,count=role['bytes']//2,
+                                   offset=role['view_offset']).reshape(role['shape'])
+            require(value.is_contiguous() and value.device.type=='cpu','authentic original role tensor')
+            self.values[name]=value;self.versions[name]=value._version
+            self.readsets[name]=fd_sha(self.fds[role['shard']],role['absolute_offset'],role['bytes'])
+    def finish(self):
+        for role in self.plan['roles']:
+            name=role['name'];require(self.values[name]._version==self.versions[name],'original source version unchanged')
+            h=__import__('hashlib').sha256();view=memoryview(self.maps[name])
+            try:
+                start=role['view_offset']
+                for offset in range(0,role['bytes'],1024**2):h.update(view[start+offset:start+min(role['bytes'],offset+1024**2)])
+            finally:view.release()
+            require(h.hexdigest()==self.readsets[name],'every original mapped role byte unchanged')
+        for shard,fd in self.fds.items():
+            pin=self.plan['selected_shards'][shard]['file']
+            require(fingerprint(os.fstat(fd))==self.before[shard] and fd_sha(fd,0,pin['bytes'])==pin['sha256'],
+                    'same retained source FD/full bytes unchanged')
+
+
+def main():
+    global OWNER
+    output=Path('/output/captured');require(not output.exists(),'exclusive actual public reference');output.mkdir(mode=0o700)
+    report={'status':'UNQUALIFIED','cases':[],'whole_model_or_state_or_GPU_qualified':False}
+    try:
+        cpus=[int(v)for v in os.environ['ZG_ALLOWED_CPUS'].split(',')]
+        require(len(cpus)==len(set(cpus))==4,'exact four CPUs');os.sched_setaffinity(0,cpus)
+        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void' and
+                os.environ.get('ATEN_CPU_CAPABILITY')=='avx2' and not os.environ.get('LD_PRELOAD') and
+                not os.environ.get('LD_AUDIT'),'pinned public CPU policy')
+        require(sha(HERE/'source-plan.json')==os.environ['ZG_SOURCE_PLAN_SHA'],'bound independent source plan')
+        plan=metadata(HERE/'source-plan.json');derived=bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
+        OWNER=SourceOwner(plan)
+        import torch
+        from torch.nn import functional as F
+        from transformers.models.qwen3_5 import modeling_qwen3_5 as original
+        require(torch.__version__=='2.13.0+cu129' and torch.version.git_version==TORCH_GIT and
+                importlib.metadata.version('transformers')=='5.17.0' and sha(original.__file__)==MODEL_SHA and
+                not torch.cuda.is_available(),'exact pinned original public CPU implementation')
+        torch.set_num_threads(4);torch.set_num_interop_threads(1)
+        require(torch.backends.cpu.get_cpu_capability()=='AVX2','actual CPU AVX2 target')
+        OWNER.map(torch)
+        prefix=metadata(Path('/prefix-reference/reference.json'),pin=plan['prior_public_prefix'])
+        require(prefix['source_repo']==plan['repository'] and prefix['source_revision']==plan['revision'] and
+                prefix['snapshot']==plan['snapshot'],'authentic accepted original27B input archive')
+        total=0
+        def dump(name,raw):
+            nonlocal total
+            total+=len(raw);require(total<=4*1024**2 and len(raw)<=2*1024**2,'bounded complete operator payload')
+            with (output/name).open('xb')as file:file.write(raw)
+            return {'file':name,'bytes':len(raw),'sha256':digest(raw)}
+        with torch.inference_mode():
+            for step,token in enumerate((11,353)):
+                require(prefix['steps'][step]['token']==token,'actual external teacher binding')
+                for i,role in enumerate(plan['roles']):
+                    count=role['input_elements']
+                    if count==5120:
+                        prior=prefix['steps'][step][role['input_stage']]
+                        full=read(Path('/prefix-reference')/prior['file'],pin=prior)
+                        require(len(full)==5120*4 and all(word&0xffff==0 for word, in struct.iter_unpack('<I',full)),
+                                'exact BF16-widened accepted input words')
+                        raw=b''.join((word>>16).to_bytes(2,'little')for word, in struct.iter_unpack('<I',full))
+                        require(digest(raw)==prior['bf16_raw_sha256'],'all accepted genuine raw BF16 input bits')
+                    else:
+                        prior=None
+                        raw=b''.join(((0xbf00 if (j+token)%2 else 0x3f00)|((j*13+token)%128)).to_bytes(2,'little')for j in range(count))
+                    x=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(1,count)
+                    weight=OWNER.values[role['name']]
+                    if role['kind']=='projection':
+                        evaluate=lambda:F.linear(x,weight)
+                    else:
+                        with torch.device('meta'):norm=original.Qwen3_5RMSNorm(count,eps=1e-6)
+                        norm.load_state_dict({'weight':weight},strict=True,assign=True);norm.eval()
+                        require(norm.weight.data_ptr()==weight.data_ptr()and norm.weight.dtype==torch.bfloat16,'raw original norm, no learned cast/cache')
+                        evaluate=lambda:norm(x)
+                    first=evaluate();second=evaluate()
+                    require(first.dtype==second.dtype==torch.bfloat16 and list(first.shape)==list(second.shape)==[1,role['output_elements']],
+                            'actual complete BF16 public operator output')
+                    firstraw=tensor_bits(first,torch);secondraw=tensor_bits(second,torch)
+                    require(firstraw==secondraw and tensor_bits(x,torch)==raw,'two genuine calls/input unchanged all bits')
+                    require(all(word&0x7f80!=0x7f80 for word, in struct.iter_unpack('<H',firstraw)),'finite public output raw words')
+                    stem=f'step-{step}-role-{i:02}'
+                    report['cases'].append({'step':step,'teacher_token':token,'role':role['role'],'layer':role['layer'],
+                        'kind':role['kind'],'source_tensor':role['name'],'source_shape':role['shape'],'source_dtype':'BF16',
+                        'input_provenance':role['input_provenance'],'input_prior':prior,
+                        'input':dump(stem+'.input.bf16le.bin',raw),'output':dump(stem+'.output.bf16le.bin',firstraw),
+                        'original_public_operator_calls':2,'both_calls_full_bits_exact':True})
+        OWNER.finish()
+        report.update(status='CAPTURED_ORIGINAL27B_PURE_OPERATOR_REFERENCE_REQUIRES_VALIDATION',
+            source_plan_sha256=sha(HERE/'source-plan.json'),source_repo=plan['repository'],source_revision=plan['revision'],
+            snapshot=plan['snapshot'],pre_library_bounds=derived,source_FDs=OWNER.before,source_readsets=OWNER.readsets,
+            all_original_source_and_mapped_bytes_unchanged=True,source_owner_retained_until_process_exit=True,
+            payload_bytes=total,torch=torch.__version__,torch_git=torch.version.git_version,modeling_sha256=MODEL_SHA,
+            affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
+            scope=plan['scope'])
+    except BaseException as error:
+        report['failure']=repr(error);raise
+    finally:write(output/'manifest.json',report)
+
+
+if __name__=='__main__':main()
--- /dev/null
+++ b/factory.rs
@@ -0,0 +1,852 @@
+#![allow(dead_code)]
+//! Root-only actual-source byte/window harness. No model or GPU execution.
+use serde_json::{Value, json};
+use sha2::{Digest, Sha256};
+use std::collections::BTreeMap;
+use std::error::Error;
+use std::fs::{File, OpenOptions};
+use std::io::Read;
+use std::ops::Range;
+use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
+use std::path::{Path, PathBuf};
+use std::sync::atomic::AtomicBool;
+use std::time::Instant;
+use zerogravity_artifacts::Error as ZgError;
+use zerogravity_artifacts::artifact::Artifact;
+use zerogravity_artifacts::cache::TransferAdmission;
+use zerogravity_artifacts::source_weights::{
+    ExpectedSourceShard, SourceWeights, VerifiedMatrix, VerifiedTensorSource, VerifiedWindowPlan,
+    WindowOpenLimits,
+};
+use zerogravity_core::accelerator::program::MatrixAxes;
+use zerogravity_core::quant::QuantKind;
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
+};
+
+type Proof<T> = Result<T, Box<dyn Error>>;
+const SNAPSHOT: &str = "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e";
+const REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
+const REPORT_SHA: &str = "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a";
+const INDEX_SHA: &str = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df";
+const BUNDLE_SHA: &str = "7e6a80bc4fc06b6237914007bde59a48a6e4fb0ea570bd5078b3101eb169faf2";
+const SELECTED_BYTES: u64 = 55_586_113_293;
+const WEIGHT_BYTES: u64 = 55_563_006_776;
+const PAYLOAD_BYTES: u64 = 55_562_855_904;
+const META: u64 = 64 * 1024 * 1024;
+const HOST: u64 = 128 * 1024 * 1024;
+const SCAN: usize = 4 * 1024 * 1024;
+const SOURCE_DIR: &str = "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files";
+
+struct Shard {
+    name: String,
+    expected_bytes: u64,
+    expected_sha: String,
+    // Both Artifact and independent raw reference derive from this one open FD.
+    file: File,
+}
+
+fn ensure(condition: bool, message: impl Into<String>) -> Proof<()> {
+    if !condition {
+        let message: String = message.into();
+        return Err(message.into());
+    }
+    Ok(())
+}
+
+fn sha(bytes: &[u8]) -> String {
+    format!("{:x}", Sha256::digest(bytes))
+}
+
+fn bounded_file(path: &Path, maximum: usize) -> Proof<Vec<u8>> {
+    let file = OpenOptions::new()
+        .read(true)
+        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+        .open(path)?;
+    ensure(file.metadata()?.is_file(), "control source is not regular")?;
+    ensure(
+        file.metadata()?.len() <= maximum as u64,
+        "control source byte cap",
+    )?;
+    let mut bytes = Vec::new();
+    file.take(maximum as u64 + 1).read_to_end(&mut bytes)?;
+    ensure(bytes.len() <= maximum, "control source grew beyond cap")?;
+    Ok(bytes)
+}
+
+fn reserved(host: &TransferAdmission) -> Proof<u64> {
+    Ok(host.arbiter.snapshot(&host.resource)?.reserved)
+}
+
+fn ledger(host: &TransferAdmission, metadata: u64, staging: u64, other: u64) -> Proof<()> {
+    let current = host.arbiter.snapshot(&host.resource)?;
+    ensure(!current.fenced, "host window pool unexpectedly fenced")?;
+    ensure(current.limit == HOST, "configured pool limit changed")?;
+    ensure(
+        current.reserved == metadata + staging + other,
+        "live total ledger mismatch",
+    )?;
+    for (class, expected) in [
+        (AllocationClass::Scratch, metadata),
+        (AllocationClass::TransferStaging, staging),
+        (AllocationClass::Other, other),
+    ] {
+        ensure(
+            current.per_class.get(&class).copied().unwrap_or(0) == expected,
+            format!("live allocation-class mismatch: {class:?}"),
+        )?;
+    }
+    ensure(
+        current.remaining + current.reserved == HOST,
+        "remaining ledger mismatch",
+    )?;
+    Ok(())
+}
+
+fn limits() -> WindowOpenLimits {
+    WindowOpenLimits {
+        max_source_bytes: 64 * 1024 * 1024 * 1024,
+        max_metadata_bytes: META,
+        max_shards: 18,
+        max_tensors: 1199,
+        scan_chunk_bytes: SCAN,
+    }
+}
+
+fn expected_shards(shards: &[Shard]) -> Proof<Vec<ExpectedSourceShard>> {
+    shards
+        .iter()
+        .map(|shard| {
+            let clone = shard.file.try_clone()?;
+            let original = shard.file.metadata()?;
+            let same = clone.metadata()?;
+            ensure(
+                (same.dev(), same.ino(), same.len())
+                    == (original.dev(), original.ino(), original.len()),
+                "descriptor clone changed source",
+            )?;
+            Ok(ExpectedSourceShard::new(
+                shard.name.clone(),
+                Artifact::from_file(clone, Path::new(&shard.name))?,
+                shard.expected_bytes,
+                shard.expected_sha.clone(),
+            )?)
+        })
+        .collect()
+}
+
+fn plan(index: &[u8], shards: &[Shard], host: &TransferAdmission) -> Proof<VerifiedWindowPlan> {
+    Ok(SourceWeights::prepare_verified_windows(
+        index,
+        INDEX_SHA,
+        expected_shards(shards)?,
+        limits(),
+        host,
+    )?)
+}
+
+fn expected_limit<T>(result: zerogravity_artifacts::Result<T>, label: &str) -> Proof<String> {
+    match result {
+        Err(ZgError::Limit(message)) => Ok(message),
+        Err(error) => Err(format!("{label}: wrong refusal: {error}").into()),
+        Ok(value) => {
+            drop(value);
+            Err(format!("{label}: operation unexpectedly succeeded").into())
+        }
+    }
+}
+
+fn expected_cancel<T>(result: zerogravity_artifacts::Result<T>, label: &str) -> Proof<String> {
+    match result {
+        Err(ZgError::State(message)) if message.contains("cancelled") => Ok(message),
+        Err(error) => Err(format!("{label}: wrong refusal: {error}").into()),
+        Ok(value) => {
+            drop(value);
+            Err(format!("{label}: operation unexpectedly succeeded").into())
+        }
+    }
+}
+
+fn guards(index: &[u8], shards: &[Shard], host: &TransferAdmission) -> Proof<Value> {
+    let mut small_metadata = limits();
+    small_metadata.max_metadata_bytes = 1;
+    let metadata = expected_limit(
+        SourceWeights::prepare_verified_windows(
+            index,
+            INDEX_SHA,
+            expected_shards(shards)?,
+            small_metadata,
+            host,
+        ),
+        "metadata envelope",
+    )?;
+    ledger(host, 0, 0, 0)?;
+
+    let mut source_limit = limits();
+    source_limit.max_source_bytes = WEIGHT_BYTES - 1;
+    let source = expected_limit(
+        SourceWeights::prepare_verified_windows(
+            index,
+            INDEX_SHA,
+            expected_shards(shards)?,
+            source_limit,
+            host,
+        ),
+        "whole-source file bound",
+    )?;
+    ledger(host, 0, 0, 0)?;
+
+    let held_bytes = HOST - META + 1;
+    let held = host.arbiter.admit(
+        &host.resource,
+        host.owner.clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Other,
+            bytes: held_bytes,
+        }],
+    )?;
+    let unavailable = expected_limit(
+        SourceWeights::prepare_verified_windows(
+            index,
+            INDEX_SHA,
+            expected_shards(shards)?,
+            limits(),
+            host,
+        ),
+        "metadata live-ledger admission",
+    )?;
+    ledger(host, 0, 0, held_bytes)?;
+    drop(held);
+    ledger(host, 0, 0, 0)?;
+
+    let pending = plan(index, shards, host)?;
+    ledger(host, META, 0, 0)?;
+    let cancelled = expected_cancel(
+        pending.open_cancellable(&AtomicBool::new(true)),
+        "pre-cancelled factory open",
+    )?;
+    ledger(host, 0, 0, 0)?;
+    Ok(json!({
+        "metadata_envelope_refused":metadata,
+        "source_file_bound_refused":source,
+        "metadata_live_ledger_refused":unavailable,
+        "pre_cancelled_factory_refused":cancelled,
+        "scope":"real headers inspected; factory pre-cancel before whole payload hash scan; no source published",
+        "post_guard_reserved_bytes":0,
+    }))
+}
+
+fn expected_staging(shard_bytes: u64, offset: u64, bytes: usize) -> Proof<u64> {
+    let end = offset
+        .checked_add(bytes as u64)
+        .ok_or("reference range overflow")?;
+    ensure(
+        bytes != 0 && end <= shard_bytes,
+        "reference range outside source",
+    )?;
+    let first = offset / SCAN as u64;
+    let last = (end - 1) / SCAN as u64;
+    let staged = ((last + 1) * SCAN as u64).min(shard_bytes) - first * SCAN as u64;
+    let offsets = (last - first + 1) * std::mem::size_of::<(usize, Range<usize>)>() as u64;
+    Ok(staged + offsets)
+}
+
+fn compare_rows(
+    source: &VerifiedTensorSource,
+    matrix: &VerifiedMatrix,
+    raw: &Shard,
+    rows: Range<usize>,
+    label: &str,
+    host: &TransferAdmission,
+) -> Proof<Value> {
+    ledger(host, META, 0, 0)?;
+    let offset = matrix.tensor().byte_offset + rows.start as u64 * matrix.columns() as u64 * 2;
+    let bytes = (rows.end - rows.start) * matrix.columns() * 2;
+    ensure(bytes <= 256 * 1024, "raw reference buffer cap")?;
+    let window = source.read_rows(matrix, rows.clone())?;
+    let charge = expected_staging(raw.expected_bytes, offset, bytes)?;
+    ensure(
+        window.charged_staging_bytes() as u64 == charge,
+        "exact staged payload+offset charge",
+    )?;
+    ledger(host, META, charge, 0)?;
+    ensure(
+        window.row_range() == rows && window.source_offset() == offset,
+        "window row/offset mismatch",
+    )?;
+    ensure(
+        window.as_slice().len() == bytes,
+        "window omitted full-width source row",
+    )?;
+    ensure(
+        window.matrix().source_identity() == source.source_identity(),
+        "window source identity changed",
+    )?;
+    ensure(
+        window.verified_bundle_sha256() == BUNDLE_SHA,
+        "window bundle changed",
+    )?;
+    ensure(
+        window.shard_basename() == raw.name && window.shard_sha256() == raw.expected_sha,
+        "window original shard binding changed",
+    )?;
+    ensure(
+        window.extent_index_sha256().len() == 64,
+        "extent receipt absent",
+    )?;
+
+    // The harness's independent comparison copy has a separate, explicit charge.
+    let reference_lease = host.arbiter.admit(
+        &host.resource,
+        host.owner.clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Other,
+            bytes: bytes as u64,
+        }],
+    )?;
+    let mut reference = vec![0u8; bytes];
+    raw.file.read_exact_at(&mut reference, offset)?;
+    ledger(host, META, charge, bytes as u64)?;
+    ensure(
+        reference.as_slice() == window.as_slice(),
+        "actual source row byte mismatch",
+    )?;
+    let record = json!({
+        "matrix":matrix.tensor().name,
+        "case":label,
+        "rows":[rows.start,rows.end],
+        "shape":[matrix.rows(),matrix.columns()],
+        "dtype":"BF16",
+        "axes":"OutputInput",
+        "source_identity":format!("{:?}",source.source_identity()),
+        "shard":raw.name,
+        "shard_sha256":raw.expected_sha,
+        "source_offset":offset,
+        "requested_bytes":bytes,
+        "charged_staging_bytes":charge,
+        "crosses_scan_extent":offset / SCAN as u64 != (offset + bytes as u64 - 1) / SCAN as u64,
+        "byte_exact":true,
+        "raw_and_window_sha256":sha(&reference),
+        "extent_index_sha256":window.extent_index_sha256(),
+    });
+    drop(reference);
+    drop(reference_lease);
+    ledger(host, META, charge, 0)?;
+    drop(window);
+    ledger(host, META, 0, 0)?;
+    Ok(record)
+}
+
+fn exercise(
+    source: VerifiedTensorSource,
+    shards: &[Shard],
+    host: &TransferAdmission,
+) -> Proof<Value> {
+    ensure(
+        source.source_identity().is_bound(),
+        "factory did not mint bound source identity",
+    )?;
+    let identity = source.source_identity();
+    ensure(
+        source.source_bytes() == WEIGHT_BYTES,
+        "whole weight byte count",
+    )?;
+    ensure(
+        source.metadata_reserved_bytes() == META,
+        "source metadata charge",
+    )?;
+    ensure(
+        source.verified_shard_index_sha256() == INDEX_SHA,
+        "source index pin",
+    )?;
+    ensure(
+        source.verified_bundle_sha256() == BUNDLE_SHA,
+        "source bundle pin",
+    )?;
+    ledger(host, META, 0, 0)?;
+    let clone = source.clone();
+    ensure(
+        clone.source_identity() == identity,
+        "source clone minted another identity",
+    )?;
+    drop(clone);
+
+    let specimens = [
+        (
+            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
+            10_240,
+            5_120,
+            1_120_792,
+            "model-00001-of-00018.safetensors",
+        ),
+        (
+            "model.language_model.embed_tokens.weight",
+            248_320,
+            5_120,
+            152,
+            "model-00003-of-00018.safetensors",
+        ),
+        (
+            "model.language_model.layers.63.mlp.down_proj.weight",
+            5_120,
+            17_408,
+            1_364_259_120,
+            "model-00017-of-00018.safetensors",
+        ),
+        (
+            "lm_head.weight",
+            248_320,
+            5_120,
+            1_760,
+            "model-00018-of-00018.safetensors",
+        ),
+    ];
+    let mut matrices = Vec::new();
+    let mut samples = Vec::new();
+    for (name, rows, columns, absolute_offset, basename) in specimens {
+        let matrix = source.matrix(name, MatrixAxes::OutputInput)?;
+        ensure(
+            matrix.source_identity() == identity,
+            "matrix changed source identity",
+        )?;
+        ensure(
+            matrix.axes() == MatrixAxes::OutputInput,
+            "matrix axes changed",
+        )?;
+        ensure(
+            (matrix.rows(), matrix.columns()) == (rows, columns),
+            "actual matrix shape differs from pinned header",
+        )?;
+        ensure(
+            matrix.tensor().shape == [rows as u64, columns as u64],
+            "raw source axes mismatch",
+        )?;
+        ensure(
+            matrix.tensor().kind == QuantKind::Bf16,
+            "raw source is not BF16",
+        )?;
+        ensure(
+            matrix.tensor().byte_len == (rows * columns * 2) as u64,
+            "raw source byte geometry",
+        )?;
+        ensure(
+            matrix.tensor().byte_offset == absolute_offset,
+            "actual source offset differs from independently inspected original header",
+        )?;
+        let raw = shards
+            .iter()
+            .find(|shard| shard.name == basename)
+            .ok_or("missing raw source FD")?;
+        samples.push(compare_rows(
+            &source,
+            &matrix,
+            raw,
+            0..1,
+            "first_full_width_row",
+            host,
+        )?);
+        samples.push(compare_rows(
+            &source,
+            &matrix,
+            raw,
+            rows - 1..rows,
+            "last_full_width_row",
+            host,
+        )?);
+        let start = matrix.tensor().byte_offset;
+        let boundary = (start / SCAN as u64 + 1) * SCAN as u64;
+        let split_row = ((boundary - start) / (columns * 2) as u64) as usize;
+        let split = split_row.saturating_sub(1)..(split_row + 1).min(rows);
+        let requested_start = start + split.start as u64 * (columns * 2) as u64;
+        let requested_end = start + split.end as u64 * (columns * 2) as u64;
+        ensure(
+            requested_start < boundary && requested_end > boundary,
+            "fixture does not straddle actual scan boundary",
+        )?;
+        samples.push(compare_rows(
+            &source,
+            &matrix,
+            raw,
+            split,
+            "scan_extent_split_full_width_rows",
+            host,
+        )?);
+        matrices.push(matrix);
+    }
+
+    let matrix = &matrices[0];
+    let cancelled = expected_cancel(
+        source.read_rows_cancellable(matrix, 0..1, &AtomicBool::new(true)),
+        "pre-cancelled row",
+    )?;
+    ledger(host, META, 0, 0)?;
+    ensure(
+        source
+            .matrix(&matrix.tensor().name, MatrixAxes::InputOutput)
+            .is_err(),
+        "wrong axes admitted",
+    )?;
+    ensure(
+        source
+            .matrix(
+                "model.language_model.layers.0.input_layernorm.weight",
+                MatrixAxes::OutputInput,
+            )
+            .is_err(),
+        "actual rank-one parameter admitted as matrix",
+    )?;
+    ensure(
+        source
+            .matrix("__not_in_the_original_index__", MatrixAxes::OutputInput)
+            .is_err(),
+        "unknown matrix admitted",
+    )?;
+    for rows in [
+        0..0,
+        matrix.rows()..matrix.rows() + 1,
+        usize::MAX..usize::MAX,
+    ] {
+        ensure(
+            source.read_rows(matrix, rows).is_err(),
+            "invalid row range admitted",
+        )?;
+        ledger(host, META, 0, 0)?;
+    }
+    let oversized = expected_limit(
+        source.read_rows(&matrices[1], 0..matrices[1].rows()),
+        "whole embedding tensor materialization",
+    )?;
+    ledger(host, META, 0, 0)?;
+
+    let pin = matrices[0].clone();
+    let last = &matrices[3];
+    let retained = source.read_rows(last, last.rows() - 1..last.rows())?;
+    let retained_sha = sha(retained.as_slice());
+    let charge = retained.charged_staging_bytes() as u64;
+    ledger(host, META, charge, 0)?;
+    drop(matrices);
+    drop(source);
+    ensure(
+        pin.source_identity() == identity && retained.matrix().source_identity() == identity,
+        "descriptor/window lost original source identity",
+    )?;
+    ledger(host, META, charge, 0)?;
+    drop(pin);
+    // Only the owned immutable window retains source metadata and its live FD set.
+    ledger(host, META, charge, 0)?;
+    ensure(
+        sha(retained.as_slice()) == retained_sha,
+        "window bytes changed after all caller source handles dropped",
+    )?;
+    drop(retained);
+    ledger(host, 0, 0, 0)?;
+    Ok(json!({
+        "source_identity":format!("{identity:?}"),
+        "source_bytes":WEIGHT_BYTES,
+        "source_bundle_sha256":BUNDLE_SHA,
+        "index_sha256":INDEX_SHA,
+        "matrix_count":4,
+        "samples":samples,
+        "pre_cancelled_row_refused":cancelled,
+        "wrong_axes_missing_name_rank_one_bad_rows_refused":true,
+        "whole_tensor_materialization_refused":oversized,
+        "only_window_retains_source_metadata_and_stage_after_all_caller_sources_drop":true,
+        "window_after_source_drop_sha256":retained_sha,
+        "final_reserved_bytes":0,
+    }))
+}
+
+fn run(host: &TransferAdmission) -> Proof<Value> {
+    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
+    let report_bytes = bounded_file(&root.join("source-pin.json"), 64 * 1024)?;
+    ensure(
+        sha(&report_bytes) == REPORT_SHA,
+        "acquired source report pin mismatch",
+    )?;
+    let report: Value = serde_json::from_slice(&report_bytes)?;
+    ensure(
+        report["snapshot"] == SNAPSHOT
+            && report["repository"] == "Qwen/Qwen3.8-27B"
+            && report["revision"] == REVISION
+            && report["reference_directory"] == SOURCE_DIR,
+        "acquired snapshot provenance differs",
+    )?;
+    ensure(
+        report["selected_bytes"] == SELECTED_BYTES
+            && report["weight_bytes"] == WEIGHT_BYTES
+            && report["payload_bytes"] == PAYLOAD_BYTES
+            && report["tensor_count"] == 1199,
+        "acquired byte/count pin differs",
+    )?;
+    let files = report["files"]
+        .as_array()
+        .ok_or("source report files absent")?;
+    ensure(files.len() == 31, "selected source file count")?;
+    let mut pinned = BTreeMap::new();
+    let mut total = 0u64;
+    for entry in files {
+        let name = entry["name"].as_str().ok_or("entry name absent")?;
+        let bytes = entry["bytes"].as_u64().ok_or("entry bytes absent")?;
+        let hash = entry["sha256"].as_str().ok_or("entry hash absent")?;
+        total = total
+            .checked_add(bytes)
+            .ok_or("selected byte sum overflow")?;
+        ensure(
+            pinned
+                .insert(name.to_string(), (bytes, hash.to_string()))
+                .is_none(),
+            "duplicate source report entry",
+        )?;
+    }
+    ensure(
+        total == SELECTED_BYTES,
+        "selected report bytes do not close",
+    )?;
+    let index = bounded_file(
+        &Path::new(SOURCE_DIR).join("model.safetensors.index.json"),
+        1024 * 1024,
+    )?;
+    ensure(
+        sha(&index) == INDEX_SHA,
+        "actual original index SHA256 mismatch",
+    )?;
+    let index_info = zerogravity_artifacts::sharded::SafetensorsIndex::from_json(&index)?;
+    ensure(
+        index_info.declared_total_size() == PAYLOAD_BYTES
+            && index_info.tensors().len() == 1199
+            && index_info.shards().len() == 18,
+        "actual original index closure differs",
+    )?;
+    let mut shards = Vec::with_capacity(18);
+    let mut descriptors = Vec::new();
+    for number in 1..=18 {
+        let name = format!("model-{number:05}-of-00018.safetensors");
+        let (bytes, hash) = pinned.get(&name).ok_or("pinned shard absent")?;
+        ensure(
+            index_info.shards().contains(name.as_str()),
+            "pin has extra shard",
+        )?;
+        let file = OpenOptions::new()
+            .read(true)
+            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+            .open(Path::new(SOURCE_DIR).join(&name))?;
+        let stat = file.metadata()?;
+        ensure(
+            stat.is_file() && stat.len() == *bytes,
+            "opened original shard type/length differs",
+        )?;
+        descriptors.push(json!({"shard":name,"dev":stat.dev(),"ino":stat.ino(),"bytes":stat.len(),"sha256":hash}));
+        shards.push(Shard {
+            name,
+            expected_bytes: *bytes,
+            expected_sha: hash.clone(),
+            file,
+        });
+    }
+    ensure(
+        shards.iter().map(|shard| shard.expected_bytes).sum::<u64>() == WEIGHT_BYTES,
+        "whole original weight-file byte sum differs",
+    )?;
+    eprintln!("phase=real_header_only_guards source_weight_bytes={WEIGHT_BYTES}");
+    let refusals = guards(&index, &shards, host)?;
+    let pending = plan(&index, &shards, host)?;
+    ensure(
+        pending.source_bytes() == WEIGHT_BYTES && pending.metadata_reserved_bytes() == META,
+        "prepared whole-source bounds differ",
+    )?;
+    ledger(host, META, 0, 0)?;
+    eprintln!(
+        "phase=whole_source_sha256_scan_start shard_count=18 scan_buffer_bytes={SCAN} metadata_reserved_bytes={META}"
+    );
+    let started = Instant::now();
+    let source = pending.open()?;
+    let scan_milliseconds = started.elapsed().as_millis();
+    eprintln!("phase=whole_source_sha256_scan_ready elapsed_ms={scan_milliseconds}");
+    let windows = exercise(source, &shards, host)?;
+    ledger(host, 0, 0, 0)?;
+    Ok(json!({
+        "schema":"zerogravity.original-owned-window-qualification.v1",
+        "status":"PASS_SOURCE_WINDOWS_ONLY",
+        "repository":"Qwen/Qwen3.8-27B",
+        "revision":REVISION,
+        "snapshot":SNAPSHOT,
+        "selected_artifact_bytes":SELECTED_BYTES,
+        "weight_file_bytes_scanned":WEIGHT_BYTES,
+        "tensor_payload_bytes":PAYLOAD_BYTES,
+        "whole_source_sha256_verified":true,
+        "retained_raw_reference_fds":descriptors,
+        "scan_chunk_bytes":SCAN,
+        "whole_source_scan_milliseconds":scan_milliseconds,
+        "configured_metadata_reserved_bytes":META,
+        "configured_host_pool_limit_bytes":HOST,
+        "refusals":refusals,
+        "windows":windows,
+        "physical_host_ram_admission_qualified":false,
+        "trained_math_or_model_inference_qualified":false,
+        "ssd_gpu_streaming_inference_qualified":false,
+        "gpu_or_state_root_publication_performed":false,
+        "scope":"one source issuer; whole weight SHA/index/header closure; owned bounded raw BF16 row bytes; configured host ledger only",
+    }))
+}
+
+fn main() {
+    let arbiter = ResourceArbiter::process_shared();
+    let resource = PhysicalDeviceKey::new("host", "explicit-owned-window-qualification-pool")
+        .expect("fixed resource label");
+    arbiter
+        .register_matching(resource.clone(), HOST)
+        .expect("fresh explicit pool registration");
+    let host = TransferAdmission {
+        arbiter,
+        resource,
+        owner: ResourceOwner::new(
+            "Qwen/Qwen3.8-27B",
+            "actual-original-source-window-qualification",
+        )
+        .expect("fixed source owner"),
+    };
+    let result = run(&host);
+    let final_state = host.arbiter.snapshot(&host.resource);
+    let clean = matches!(&final_state, Ok(state) if state.reserved == 0 && state.reservations == 0 && !state.fenced);
+    let success = result.is_ok() && clean;
+    let mut report = match result {
+        Ok(report) => report,
+        Err(error) => json!({"status":"FAIL_UNQUALIFIED","error":error.to_string()}),
+    };
+    report["final_owned_ledger_clean"] = json!(clean);
+    report["final_owned_ledger_reserved_bytes"] =
+        json!(final_state.as_ref().ok().map(|state| state.reserved));
+    if !clean {
+        report["status"] = json!("FAIL_UNQUALIFIED");
+        report["ledger_cleanup_error"] = json!(format!("{final_state:?}"));
+    }
+    println!(
+        "{}",
+        serde_json::to_string_pretty(&report).expect("bounded report serialization")
+    );
+    if !success {
+        std::process::exit(2);
+    }
+}
+
+/// Only this additive wrapper is called by the new replay. Original factory
+/// source below remains retained verbatim; its old main/exercise are not called.
+pub struct SourceReady {
+    pub source: VerifiedTensorSource,
+    pub config: Vec<u8>,
+    pub descriptors: Vec<Value>,
+    pub scan_milliseconds: u128,
+}
+pub fn open_source(host: &TransferAdmission) -> Proof<SourceReady> {
+    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
+    let report_bytes = bounded_file(&root.join("source-pin.json"), 64 * 1024)?;
+    ensure(
+        sha(&report_bytes) == REPORT_SHA,
+        "acquired source report pin mismatch",
+    )?;
+    let report: Value = serde_json::from_slice(&report_bytes)?;
+    ensure(
+        report["snapshot"] == SNAPSHOT
+            && report["repository"] == "Qwen/Qwen3.8-27B"
+            && report["revision"] == REVISION
+            && report["reference_directory"] == SOURCE_DIR,
+        "acquired snapshot provenance differs",
+    )?;
+    ensure(
+        report["selected_bytes"] == SELECTED_BYTES
+            && report["weight_bytes"] == WEIGHT_BYTES
+            && report["payload_bytes"] == PAYLOAD_BYTES
+            && report["tensor_count"] == 1199,
+        "acquired byte/count pin differs",
+    )?;
+    let files = report["files"]
+        .as_array()
+        .ok_or("source report files absent")?;
+    ensure(files.len() == 31, "selected source file count")?;
+    let mut pinned = BTreeMap::new();
+    let mut total = 0u64;
+    for entry in files {
+        let name = entry["name"].as_str().ok_or("entry name absent")?;
+        let bytes = entry["bytes"].as_u64().ok_or("entry bytes absent")?;
+        let hash = entry["sha256"].as_str().ok_or("entry hash absent")?;
+        total = total
+            .checked_add(bytes)
+            .ok_or("selected byte sum overflow")?;
+        ensure(
+            pinned
+                .insert(name.to_string(), (bytes, hash.to_string()))
+                .is_none(),
+            "duplicate source report entry",
+        )?;
+    }
+    ensure(
+        total == SELECTED_BYTES,
+        "selected report bytes do not close",
+    )?;
+    let index = bounded_file(
+        &Path::new(SOURCE_DIR).join("model.safetensors.index.json"),
+        1024 * 1024,
+    )?;
+    ensure(
+        sha(&index) == INDEX_SHA,
+        "actual original index SHA256 mismatch",
+    )?;
+    let index_info = zerogravity_artifacts::sharded::SafetensorsIndex::from_json(&index)?;
+    ensure(
+        index_info.declared_total_size() == PAYLOAD_BYTES
+            && index_info.tensors().len() == 1199
+            && index_info.shards().len() == 18,
+        "actual original index closure differs",
+    )?;
+    let mut shards = Vec::with_capacity(18);
+    let mut descriptors = Vec::new();
+    for number in 1..=18 {
+        let name = format!("model-{number:05}-of-00018.safetensors");
+        let (bytes, hash) = pinned.get(&name).ok_or("pinned shard absent")?;
+        ensure(
+            index_info.shards().contains(name.as_str()),
+            "pin has extra shard",
+        )?;
+        let file = OpenOptions::new()
+            .read(true)
+            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+            .open(Path::new(SOURCE_DIR).join(&name))?;
+        let stat = file.metadata()?;
+        ensure(
+            stat.is_file() && stat.len() == *bytes,
+            "opened original shard type/length differs",
+        )?;
+        descriptors.push(json!({"shard":name,"dev":stat.dev(),"ino":stat.ino(),"bytes":stat.len(),"sha256":hash}));
+        shards.push(Shard {
+            name,
+            expected_bytes: *bytes,
+            expected_sha: hash.clone(),
+            file,
+        });
+    }
+    ensure(
+        shards.iter().map(|shard| shard.expected_bytes).sum::<u64>() == WEIGHT_BYTES,
+        "whole original weight-file byte sum differs",
+    )?;
+    let config = bounded_file(&Path::new(SOURCE_DIR).join("config.json"), 64 * 1024)?;
+    ensure(
+        sha(&config) == "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
+        "original config SHA",
+    )?;
+    let pending = plan(&index, &shards, host)?;
+    let started = Instant::now();
+    let source = pending.open()?;
+    ensure(
+        source.source_bytes() == WEIGHT_BYTES
+            && source.metadata_reserved_bytes() == META
+            && source.verified_shard_index_sha256() == INDEX_SHA
+            && source.verified_bundle_sha256() == BUNDLE_SHA,
+        "source factory closure",
+    )?;
+    Ok(SourceReady {
+        source,
+        config,
+        descriptors,
+        scan_milliseconds: started.elapsed().as_millis(),
+    })
+}
--- /dev/null
+++ b/main.rs
@@ -0,0 +1,339 @@
+//! ROOT only: source-authentic pure roles; no layer/token/session/state engine.
+mod factory;
+use serde_json::{Value, json};
+use sha2::{Digest, Sha256};
+use std::error::Error;
+use std::fs::OpenOptions;
+use std::io::{Read, Write};
+use std::os::unix::fs::OpenOptionsExt;
+use std::path::Path;
+use std::sync::atomic::AtomicBool;
+use zerogravity_artifacts::cache::TransferAdmission;
+use zerogravity_core::accelerator::program::ProgramNumericalProfile;
+use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
+use zerogravity_model_qwen35::operator_slice::pure::{
+    PureSliceInput, PureSliceOutput, PureSliceRole,
+};
+use zerogravity_model_qwen35::original_binding::{
+    OriginalWindowOperators, QwenMatrixRole as M, QwenNumericalProfile, QwenParameterRole as P,
+};
+
+type Proof<T> = Result<T, Box<dyn Error>>;
+const HOST: u64 = 128 * 1024 * 1024;
+const CONTROL: usize = 8 * 1024 * 1024;
+fn need(ok: bool, why: &str) -> Proof<()> {
+    if ok { Ok(()) } else { Err(why.into()) }
+}
+fn sha(raw: &[u8]) -> String {
+    format!("{:x}", Sha256::digest(raw))
+}
+fn read(path: &Path, cap: usize) -> Proof<Vec<u8>> {
+    let mut file = OpenOptions::new()
+        .read(true)
+        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+        .open(path)?;
+    let before = file.metadata()?;
+    need(
+        before.is_file() && before.len() <= cap as u64,
+        "bounded actual regular reference",
+    )?;
+    let mut raw = Vec::new();
+    Read::by_ref(&mut file)
+        .take(cap as u64 + 1)
+        .read_to_end(&mut raw)?;
+    use std::os::unix::fs::MetadataExt;
+    let after = file.metadata()?;
+    need(
+        raw.len() <= cap
+            && raw.len() as u64 == before.len()
+            && before.dev() == after.dev()
+            && before.ino() == after.ino()
+            && before.len() == after.len()
+            && before.mtime_nsec() == after.mtime_nsec()
+            && before.ctime_nsec() == after.ctime_nsec()
+            && before.mtime() == after.mtime()
+            && before.ctime() == after.ctime(),
+        "same immutable actual reference FD",
+    )?;
+    Ok(raw)
+}
+fn field<'a>(value: &'a Value, key: &str) -> Proof<&'a str> {
+    value[key]
+        .as_str()
+        .ok_or_else(|| format!("missing string:{key}").into())
+}
+fn payload(root: &Path, row: &Value) -> Proof<Vec<u8>> {
+    let name = field(row, "file")?;
+    need(
+        Path::new(name).file_name().and_then(|v| v.to_str()) == Some(name),
+        "contained reference filename",
+    )?;
+    let raw = read(&root.join(name), 2 * 1024 * 1024)?;
+    need(
+        row["bytes"] == raw.len() && row["sha256"] == sha(&raw) && raw.len().is_multiple_of(2),
+        "full original reference raw bytes",
+    )?;
+    Ok(raw)
+}
+fn compare(actual: &[f32], raw: &[u8]) -> Proof<usize> {
+    need(
+        actual.len().checked_mul(2) == Some(raw.len()),
+        "complete raw word coverage",
+    )?;
+    for (i, (value, word)) in actual.iter().zip(raw.chunks_exact(2)).enumerate() {
+        let expected = u32::from(u16::from_le_bytes([word[0], word[1]])) << 16;
+        if value.to_bits() != expected {
+            return Err(format!(
+                "UNQUALIFIED first exact mismatch word{i}: {:08x}!={expected:08x}",
+                value.to_bits()
+            )
+            .into());
+        }
+    }
+    Ok(actual.len())
+}
+fn role(row: &Value) -> Proof<PureSliceRole> {
+    let layer = || -> Proof<usize> {
+        row["layer"]
+            .as_u64()
+            .and_then(|n| usize::try_from(n).ok())
+            .ok_or_else(|| "actual layer".into())
+    };
+    Ok(match (field(row, "kind")?, field(row, "role")?) {
+        ("projection", "GdnQkv") => PureSliceRole::Projection(M::GdnQkv(layer()?)),
+        ("projection", "GdnGate") => PureSliceRole::Projection(M::GdnGate(layer()?)),
+        ("projection", "GdnAlpha") => PureSliceRole::Projection(M::GdnAlpha(layer()?)),
+        ("projection", "GdnBeta") => PureSliceRole::Projection(M::GdnBeta(layer()?)),
+        ("projection", "GdnOutput") => PureSliceRole::Projection(M::GdnOutput(layer()?)),
+        ("projection", "Query") => PureSliceRole::Projection(M::Query(layer()?)),
+        ("projection", "Key") => PureSliceRole::Projection(M::Key(layer()?)),
+        ("projection", "Value") => PureSliceRole::Projection(M::Value(layer()?)),
+        ("projection", "AttentionOutput") => {
+            PureSliceRole::Projection(M::AttentionOutput(layer()?))
+        }
+        ("projection", "FfnGate") => PureSliceRole::Projection(M::FfnGate(layer()?)),
+        ("projection", "FfnUp") => PureSliceRole::Projection(M::FfnUp(layer()?)),
+        ("projection", "FfnDown") => PureSliceRole::Projection(M::FfnDown(layer()?)),
+        ("projection", "Output") => PureSliceRole::Projection(M::Output),
+        ("ordinary_rms", "InputNorm") => PureSliceRole::OrdinaryRms(P::InputNorm(layer()?)),
+        ("ordinary_rms", "FfnNorm") => PureSliceRole::OrdinaryRms(P::FfnNorm(layer()?)),
+        ("ordinary_rms", "FinalNorm") => PureSliceRole::OrdinaryRms(P::FinalNorm),
+        _ => return Err("unknown pure role".into()),
+    })
+}
+fn run(
+    host: &TransferAdmission,
+    directory: &Path,
+    reference_sha: &str,
+    validation_sha: &str,
+) -> Proof<Value> {
+    let ready = factory::open_source(host)?;
+    let source = ready.source;
+    let control = source.reserve_workspace(CONTROL)?;
+    let plan: Value = serde_json::from_slice(include_bytes!("source-plan.json"))?;
+    let refraw = read(&directory.join("captured/manifest.json"), 4 * 1024 * 1024)?;
+    let validraw = read(&directory.join("validation.json"), 4 * 1024 * 1024)?;
+    need(
+        sha(&refraw) == reference_sha && sha(&validraw) == validation_sha,
+        "ROOT bound independently validated public artifacts",
+    )?;
+    let reference: Value = serde_json::from_slice(&refraw)?;
+    let validated: Value = serde_json::from_slice(&validraw)?;
+    need(
+        validated["status"] == "PASS_ORIGINAL27B_PURE_OPERATOR_REFERENCE_ONLY"
+            && validated["manifest_sha256"] == reference_sha
+            && validated["source_plan_sha256"] == sha(include_bytes!("source-plan.json"))
+            && reference["source_plan_sha256"] == validated["source_plan_sha256"],
+        "genuine public reference/profile binding, no caller hash as independent math proof",
+    )?;
+    let operators = OriginalWindowOperators::bind(
+        source.clone(),
+        &ready.config,
+        "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
+        "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )?;
+    need(
+        operators.binding().text_tensor_count() == 851 && source.source_bytes() == 55_563_006_776,
+        "actual complete original27B851 role source closure",
+    )?;
+    let cases = reference["cases"].as_array().ok_or("public cases")?;
+    let roles = plan["roles"].as_array().ok_or("frozen roles")?;
+    need(
+        cases.len() == 32 && roles.len() == 16,
+        "full16 roles/two probes",
+    )?;
+    let steady = host.arbiter.snapshot(&host.resource)?.reserved;
+    let first = payload(&directory.join("captured"), &cases[0]["input"])?;
+    let input = PureSliceInput::from_bf16_bytes(
+        source.source_identity(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        &first,
+    )?;
+    need(
+        matches!(operators.execute_projection_slice(M::GdnQkv(0),input,Some(&AtomicBool::new(true))),
+          Err(zerogravity_core::Error::State(ref message)) if message.contains("cancelled")),
+        "actual pre-cancel refuses",
+    )?;
+    let mut pressure = Vec::new();
+    loop {
+        let remaining = host.arbiter.snapshot(&host.resource)?.remaining;
+        if remaining <= 1 {
+            break;
+        }
+        pressure.push(source.reserve_workspace((remaining - 1).min(32 * 1024 * 1024) as usize)?);
+    }
+    let held_bytes = host.arbiter.snapshot(&host.resource)?.reserved;
+    need(
+        matches!(operators.execute_projection_slice(M::GdnQkv(0),input,None),
+          Err(zerogravity_core::Error::Limit(ref message)) if message=="physical device memory admission refused"),
+        "actual arbiter capacity refusal",
+    )?;
+    need(
+        host.arbiter.snapshot(&host.resource)?.reserved == held_bytes,
+        "refusal leaves prior real charge exact",
+    )?;
+    drop(pressure);
+    need(
+        host.arbiter.snapshot(&host.resource)?.reserved == steady,
+        "pressure owners restore exact ledger",
+    )?;
+    let mut compared = 0usize;
+    let mut reports = Vec::new();
+    let mut retained: Option<PureSliceOutput> = None;
+    for (position, row) in cases.iter().enumerate() {
+        let (step, index) = (position / 16, position % 16);
+        let expected = &roles[index];
+        need(
+            row["step"] == step
+                && row["teacher_token"] == [11, 353][step]
+                && row["role"] == expected["role"]
+                && row["layer"] == expected["layer"]
+                && row["kind"] == expected["kind"]
+                && row["source_tensor"] == expected["name"]
+                && row["source_shape"] == expected["shape"],
+            "ordered source/role/teacher closure",
+        )?;
+        let original = source.tensor_info(field(row, "source_tensor")?)?;
+        let shape = expected["shape"]
+            .as_array()
+            .ok_or("original shape")?
+            .iter()
+            .map(|v| v.as_u64().ok_or("shape axis"))
+            .collect::<Result<Vec<_>, _>>()?;
+        need(
+            original.shape == shape && original.kind == zerogravity_core::quant::QuantKind::Bf16,
+            "actual sealed original source shape/codec",
+        )?;
+        let x = payload(&directory.join("captured"), &row["input"])?;
+        let y = payload(&directory.join("captured"), &row["output"])?;
+        let transport = PureSliceInput::from_bf16_bytes(
+            source.source_identity(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            &x,
+        )?;
+        let target = role(row)?;
+        let output = match target {
+            PureSliceRole::Projection(role) => {
+                operators.execute_projection_slice(role, transport, None)?
+            }
+            PureSliceRole::OrdinaryRms(role) => {
+                operators.execute_ordinary_rms_slice(role, transport, None)?
+            }
+        };
+        need(
+            output.role() == target && output.source_identity() == source.source_identity(),
+            "exact native output identity",
+        )?;
+        compared += compare(output.input()?, &x)?;
+        compared += compare(output.output()?, &y)?;
+        reports.push(json!({"position":position,"role":row["role"],"source_tensor":row["source_tensor"],"output_words":output.output()?.len(),"all_bits_exact":true}));
+        if target == PureSliceRole::Projection(M::Output) && step == 1 {
+            retained = Some(output);
+        }
+    }
+    need(
+        compared
+            == validated["words"]
+                .as_u64()
+                .ok_or("complete independent words")? as usize,
+        "all actual public/native words compared",
+    )?;
+    let result = retained.ok_or("retained actual untied head output")?;
+    let source_meta = source.metadata_reserved_bytes();
+    drop(operators);
+    drop(source);
+    drop(ready.config);
+    let retained_ledger = host.arbiter.snapshot(&host.resource)?;
+    need(
+        !retained_ledger.fenced
+            && retained_ledger.reserved == source_meta + CONTROL as u64 + 3_035_648,
+        "actual head input/output/source/control retained after caller drop",
+    )?;
+    need(
+        result.output()?.len() == 248320 && result.input()?.len() == 5120,
+        "full untied head remains borrowable",
+    )?;
+    drop(result);
+    drop(first);
+    drop(refraw);
+    drop(validraw);
+    drop(reference);
+    drop(validated);
+    drop(plan);
+    drop(control);
+    need(
+        host.arbiter.snapshot(&host.resource)?.reserved == 0,
+        "last private/source owner releases exact budget",
+    )?;
+    Ok(
+        json!({"cases":reports,"compared_words":compared,"all_exact":true,"actual_pre_cancel_and_pressure_refused":true,
+        "retained_head_bytes":3035648,"whole_model_StateRoot_GPU_or_performance_qualified":false}),
+    )
+}
+fn main() {
+    let args = std::env::args().collect::<Vec<_>>();
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "original27-pure-role-replay").expect("key");
+    arbiter
+        .register(key.clone(), HOST)
+        .expect("isolated bounded ledger");
+    let host = TransferAdmission {
+        arbiter,
+        resource: key,
+        owner: ResourceOwner::new("original27", "pure-role-root-replay").expect("owner"),
+    };
+    let result = (|| -> Proof<Value> {
+        need(
+            args.len() == 5,
+            "REFERENCE_DIR MANIFEST_SHA VALIDATION_SHA OUTPUT",
+        )?;
+        run(&host, Path::new(&args[1]), &args[2], &args[3])
+    })();
+    let final_ledger = host.arbiter.snapshot(&host.resource).expect("final ledger");
+    let okay = result.is_ok() && final_ledger.reserved == 0 && !final_ledger.fenced;
+    let report = match result {
+        Ok(result) => {
+            json!({"status":if okay{"PASS_PRIVATE_ORIGINAL27B_PURE_ROLES_ONLY"}else{"UNQUALIFIED"},"result":result})
+        }
+        Err(error) => json!({"status":"UNQUALIFIED","error":error.to_string()}),
+    };
+    let report = json!({"proof":report,"final_reserved":final_ledger.reserved,"final_fenced":final_ledger.fenced,
+                      "configured_host_limit":HOST,"whole_host_physical_availability_qualified":false});
+    if let Some(path) = args.get(4) {
+        let write = (|| -> Proof<()> {
+            let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
+            f.write_all(&serde_json::to_vec_pretty(&report)?)?;
+            f.sync_all()?;
+            Ok(())
+        })();
+        if let Err(error) = write {
+            eprintln!("REPORT_FAILURE:{error}");
+            std::process::exit(1);
+        }
+    }
+    if !okay {
+        eprintln!("{report}");
+        std::process::exit(2);
+    }
+}
--- /dev/null
+++ b/owned_container_v4.py
@@ -0,0 +1,280 @@
+"""ROOT ONLY finite owned lifetime. No execution occurs by importing this file.
+
+Additive v4: original nonce/CID binding and commands preserved. WNOWAIT and
+selector cleanup follow the reviewed AV runner, with whole-group KILL strictly
+before leader reap: its reserved PID prevents accidental process-group reuse.
+"""
+import hashlib
+import json
+import os
+from pathlib import Path
+import re
+import selectors
+import signal
+import subprocess
+import time
+
+PODMAN = ["podman", "--root", "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers", "--runroot", "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers"]
+LABEL = "abird.zerogravity.vllm-gpu-oracle-owner"
+CAP = 4 * 1024 * 1024
+
+
+def require(value, message):
+    if not value:
+        raise ValueError(message)
+
+
+def sha(path):
+    with Path(path).open("rb") as file:
+        return hashlib.file_digest(file, "sha256").hexdigest()
+
+
+def command(argv, timeout=10):
+    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                            stderr=subprocess.PIPE, timeout=timeout, check=False)
+    require(len(result.stdout) + len(result.stderr) <= 1024 * 1024, "bounded command receipt")
+    return result
+
+
+def leader_status_unreaped(process):
+    require(process.returncode is None, "owned leader already reaped; never signal a recycled PGID")
+    observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if observed is None:
+        return None
+    require(observed.si_pid == process.pid and observed.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
+            "owned leader exit observation")
+    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
+
+
+class Log:
+    """Main-thread bounded drain; no thread/future can survive cleanup."""
+    def __init__(self, pipe, path):
+        self.pipe, self.path = pipe, Path(path)
+        self.selector = self.file = None
+        self.retained = self.total = 0
+        self.eof = self.closed = False
+        self.errors = []
+        try:
+            self.file = self.path.open("xb")
+            os.chmod(self.path, 0o600)
+            os.set_blocking(pipe.fileno(), False)
+            self.selector = selectors.DefaultSelector()
+            self.selector.register(pipe, selectors.EVENT_READ)
+        except BaseException:
+            if self.selector is not None:
+                self.selector.close()
+            if self.file is not None:
+                self.file.close()
+            raise
+
+    def drain(self, timeout):
+        for key, _ in self.selector.select(timeout):
+            # One bounded block per wake; the exit/deadline remains observable
+            # even when an owned writer continuously produces output.
+            try:
+                raw = os.read(key.fd, 65536)
+            except BlockingIOError:
+                continue
+            if not raw:
+                self.selector.unregister(key.fileobj)
+                self.eof = True
+                continue
+            self.total += len(raw)
+            keep = raw[:max(0, CAP - self.retained)]
+            if keep:
+                try:
+                    self.file.write(keep)
+                    self.retained += len(keep)
+                except OSError as error:
+                    self.errors.append(repr(error))
+
+    def observe(self, process, seconds):
+        deadline = time.monotonic() + seconds
+        while time.monotonic() < deadline:
+            self.drain(0.05)
+            require(self.total <= CAP and not self.errors, "owned log failed or exceeded4MiB")
+            status = leader_status_unreaped(process)
+            if status is not None:
+                return status
+        raise TimeoutError("owned foreground exit deadline (leader remains unreaped)")
+
+    def finish(self):
+        failures = []
+        try:
+            deadline = time.monotonic() + 3
+            while not self.eof and time.monotonic() < deadline:
+                self.drain(0.05)
+            require(self.eof, "owned stdout pipe EOF deadline")
+            require(self.total <= CAP and not self.errors, "owned log evidence failed/truncated")
+            self.file.flush()
+            os.fsync(self.file.fileno())
+        except BaseException as error:
+            failures.append("drain/evidence: " + repr(error))
+        finally:
+            # Close all three even when EOF, evidence I/O or one close fails.
+            for name, close in (("selector", self.selector.close), ("log", self.file.close),
+                                ("pipe", self.pipe.close)):
+                try:
+                    close()
+                except BaseException as error:
+                    failures.append(name + ": " + repr(error))
+            self.closed = self.pipe.closed and self.file.closed and not self.selector.get_map()
+        if failures:
+            raise RuntimeError("owned log cleanup: " + "; ".join(failures))
+
+
+def owned_container(name, nonce, cidfile):
+    target, recorded_cid = name, None
+    if cidfile.exists():
+        cid = cidfile.read_text().strip()
+        if cid:
+            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
+            target = recorded_cid = cid
+    exists = command(PODMAN + ["container", "exists", target])
+    if exists.returncode == 1:
+        return None
+    require(exists.returncode == 0, "owned container existence check failed")
+    inspected = command(PODMAN + ["container", "inspect", target])
+    require(inspected.returncode == 0, "owned container inspect failed")
+    value, = json.loads(inspected.stdout)
+    require(value["Name"] == name and value["Config"].get("Labels", {}).get(LABEL) == nonce,
+            "container name/nonce mismatch; no signal")
+    cid = value["Id"]
+    require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
+    if recorded_cid is not None:
+        require(recorded_cid == cid, "owned CID changed; no signal")
+    return cid
+
+
+def cleanup_container(name, nonce, cidfile):
+    errors = []
+    try:
+        cid = owned_container(name, nonce, cidfile)
+    except BaseException as error:
+        return ["container owner binding: " + repr(error)]
+    if cid is None:
+        return errors
+    try:
+        stopped = command(PODMAN + ["stop", "--time", "5", cid], 15)
+        if stopped.returncode != 0:
+            errors.append("owned stop returned nonzero")
+    except BaseException as error:
+        errors.append("container stop: " + repr(error))
+    try:
+        # Fresh exact owner binding is required even after a failed/timedout
+        # stop. A stop failure must not skip the bounded force-remove attempt.
+        if owned_container(name, nonce, cidfile) is not None:
+            removed = command(PODMAN + ["rm", "--force", cid], 15)
+            require(removed.returncode == 0, "owned container force cleanup failed")
+    except BaseException as error:
+        errors.append("container force cleanup: " + repr(error))
+    try:
+        require(owned_container(name, nonce, cidfile) is None, "owned container survived cleanup")
+    except BaseException as error:
+        errors.append("container final absence: " + repr(error))
+    return errors
+
+
+def group_exists(pgid):
+    try:
+        os.killpg(pgid, 0)
+        return True
+    except ProcessLookupError:
+        return False
+
+
+def signal_group_unreaped(process, sig):
+    # waitid authenticates that this is still our unreaped child. start_new_session
+    # made it the session/group leader; no foreign group is inferred from a PID.
+    leader_status_unreaped(process)
+    require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
+            "owned leader session/PGID binding")
+    os.killpg(process.pid, sig)
+
+
+def cleanup_group(process, receipt, *, observe=leader_status_unreaped, pause=time.sleep):
+    """Independent bounded stages; no nonzero signal after any reap attempt."""
+    state = receipt["foreground_cleanup"] = {"pgid": process.pid, "term_sent": False,
+        "kill_sent_before_reap": False, "leader_reaped": False, "group_absent": False,
+        "kill_attempts_before_reap": 0, "leader_reap_attempts": 0}
+    errors = []
+    try:
+        signal_group_unreaped(process, signal.SIGTERM)
+        state["term_sent"] = True
+    except ProcessLookupError:
+        state["term_group_absent_before_reap"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
+    try:
+        deadline = time.monotonic() + 1
+        while time.monotonic() < deadline:
+            observe(process)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # Both independently authenticated KILL attempts precede ANY wait/reap.
+    for attempt in range(2):
+        state["kill_attempts_before_reap"] += 1
+        try:
+            signal_group_unreaped(process, signal.SIGKILL)
+            state["kill_sent_before_reap"] = True
+        except ProcessLookupError:
+            state["kill_group_absent_before_reap"] = True
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    state["all_nonzero_signals_before_reap"] = True
+    # Popen.wait targets only this child. A failed first wait cannot skip final wait.
+    try:
+        state["leader_reap_attempts"] += 1
+        state["exit_code"] = process.wait(timeout=5)
+        state["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned first leader reap: " + repr(error))
+    finally:
+        if process.returncode is None:
+            try:
+                state["leader_reap_attempts"] += 1
+                state["exit_code"] = process.wait(timeout=5)
+                state["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final leader reap: " + repr(error))
+        else:
+            state["exit_code"] = process.returncode
+            state["leader_reaped"] = True
+    try:
+        require(state["leader_reaped"], "owned leader was not reaped")
+        deadline = time.monotonic() + 5
+        while group_exists(process.pid) and time.monotonic() < deadline:
+            pause(0.05)  # group_exists uses signal0 only; never signal after reap.
+        state["group_absent"] = not group_exists(process.pid)
+        require(state["group_absent"], "owned foreground group remains after final KILL/reap")
+    except BaseException as error:
+        errors.append("owned group absence: " + repr(error))
+    state["stage_errors"] = list(errors)
+    if errors:
+        raise RuntimeError("independent owned foreground cleanup: " + "; ".join(errors))
+
+
+def cleanup(process, name, nonce, cidfile, receipt):
+    errors = []
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("initial owned container cleanup: " + repr(error))
+    try:
+        cleanup_group(process, receipt)
+    except BaseException as error:
+        errors.append("owned foreground cleanup: " + repr(error))
+    # Rediscover startup races using only the original nonce/name/CID binding.
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("final owned container cleanup: " + repr(error))
+    try:
+        require(receipt.get("foreground_cleanup", {}).get("leader_reaped"), "owned leader not reaped")
+        require(receipt["foreground_cleanup"]["group_absent"], "owned group absence not established")
+        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
+        receipt["owned_container_absent_after_reap"] = True
+    except BaseException as error:
+        errors.append("final owned absence: " + repr(error))
+    return errors
--- /dev/null
+++ b/relocated_source.py
@@ -0,0 +1,277 @@
+"""Explicit pure-wide source relocation; not compilation or numerical proof."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+
+def _raw(path, row, require):
+    path = Path(path)
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as opened:
+        before = os.fstat(opened.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 2*1024**2,
+                'bounded regular source/metadata only')
+        raw = opened.read(2*1024**2+1)
+        after = os.fstat(opened.fileno())
+    identity = lambda v: (v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
+    require(identity(before) == identity(after) and len(raw) == before.st_size == row['bytes'] and
+            hashlib.sha256(raw).hexdigest() == row['sha256'], 'exact same-FD immutable source pin')
+    return raw
+
+def remap(path, config, require):
+    path = Path(path)
+    old = Path(config['original_harness'])
+    root = Path(config['main_root'])
+    require(path.is_absolute() and path == Path(os.path.normpath(path)), 'canonical absolute build input')
+    if path.is_relative_to(old):
+        relative = path.relative_to(old)
+        require(len(relative.parts) == 1 and relative.name in
+                ('Cargo.toml','main.rs','factory.rs','source-pin.json','source-plan.json'),
+                'only exact original local build inputs may move')
+        return Path(config['relocated_harness'])/relative
+    require(path.is_relative_to(root), 'original build input belongs to original source tree')
+    relative = path.relative_to(root)
+    require(relative.parts[0] == 'crates' or str(relative) in ('Cargo.toml','Cargo.lock'),
+            'only MAIN crate/manifests remap; source artifacts remain original')
+    return Path(config['workspace_root'])/relative
+
+def normalized(before, replacements, require):
+    value = before.decode()
+    for edit in replacements:
+        require(edit['before'] and type(edit['expected_occurrences']) is int and
+                edit['expected_occurrences'] == 1 and value.count(edit['before']) == 1,
+                'exact unique ROOT-recorded source-only adapter/test replacement')
+        value = value.replace(edit['before'],edit['after'],1)
+    return value.encode()
+
+def workspace(config, read_json, require):
+    root = Path(config['workspace_root'])
+    require(root.is_absolute() and root.resolve() == root and root != Path(config['main_root']),
+            'explicit distinct canonical isolated ROOT source tree')
+    pin = config['source_freeze']
+    raw = _raw(pin['path'],pin,require)
+    document = json.loads(raw)
+    require(document['full_model_or_GPU_qualified'] is False and
+            type(config['source_count']) is int and len(document['files']) == config['source_count'] == 521,
+            'exact ROOT-reviewed521 Rust/manifests closure; not runtime/full-model proof')
+    by_path = {}
+    for row in document['files']:
+        relative = Path(row['file'])
+        require(not relative.is_absolute() and relative.parts and
+                all(v not in ('.','..') for v in relative.parts) and
+                (relative.suffix == '.rs' or relative.name in ('Cargo.toml','Cargo.lock')),
+                'finite source-only relative manifest entry')
+        path = root/relative
+        require(path.resolve() == path and str(path) not in by_path, 'no source symlink/duplicate')
+        _raw(path,row,require)
+        by_path[str(path)] = dict(path=str(path),bytes=row['bytes'],sha256=row['sha256'])
+    actual = {str(root/'Cargo.toml'),str(root/'Cargo.lock')}
+    actual.update(str(p) for p in (root/'crates').rglob('*') if p.is_file() and
+                  (p.suffix == '.rs' or p.name == 'Cargo.toml'))
+    require(actual == set(by_path), 'complete current ROOT Rust/manifests inventory, no unlisted code')
+    for row in (config['ROOT_source_patch_pins']+config['ROOT_composition_journal_pins']+
+                [config['ROOT_focused_check_pin']]):
+        _raw(row['path'],row,require)
+    initial_pin = config['ROOT_original_source_freeze']
+    initial = json.loads(_raw(initial_pin['path'],initial_pin,require))
+    require(len(initial['files']) == 521 and initial['full_model_or_GPU_qualified'] is False,
+            'original ROOT-reviewed source closure before recorded corrections')
+    for exception in config['normalization_exceptions']:
+        require(exception['classification'] in
+                ('adapter_only_two_head_count_arguments','test_only_two_vec_array_substitutions',
+                 'test_only_same_H48_budget_valid_schedule_parameter_drop',
+                 'test_only_unused_import_removal','test_only_array_and_real_pressure_chunks',
+                 'test_only_valid_layer_schedule','test_only_explicit_closed_embedding_role_negative',
+                 'test_only_distinct_fresh_source_identity'), 'only named ROOT adapter/test corrections')
+        require(exception['isolated']['path'] == str(root/exception['file']) and
+                by_path.get(exception['isolated']['path']) == exception['isolated'],
+                'exception must match exact ROOT source-freeze row')
+        before = _raw(exception['original']['path'],exception['original'],require)
+        if exception['classification'] == 'test_only_explicit_closed_embedding_role_negative':
+            original_rows = [row for row in initial['files'] if row['file'] == exception['file']]
+            require(len(original_rows) == 1 and
+                    (original_rows[0]['bytes'],original_rows[0]['sha256']) ==
+                    (exception['original']['bytes'],exception['original']['sha256']),
+                    'retained exact original ROOT attention fixture; no MAIN dependency')
+        after = _raw(exception['isolated']['path'],exception['isolated'],require)
+        require(normalized(before,exception['replacements'],require) == after,
+                'only declared ROOT adapter/test changes; no arithmetic or unreviewed body change')
+    return by_path
+
+
+def _original(config, metadata, read, require):
+    """All original13 + additive V2 files remain exact at their old paths."""
+    old = Path(config['original_harness'])
+    pin = config['original_preparation']
+    read(pin['path'],pin=pin)
+    original = metadata(pin['path'])
+    require(pin['sha256'] == '16809776089e1c3063629a0e0df556512a66b9ccd04d6812d56e47b744f4cb40' and
+            len(original['files']) == 13,'exact original wide qualification preparation')
+    for row in original['files']:
+        read(old/row['file'],pin=row)
+        read(old/'frozen-prepared'/row['file'],pin=row)
+    for row in original['parent_metadata']:
+        read(row['path'],pin=row)
+    pin = config['original_metadata_followup']
+    read(pin['path'],pin=pin)
+    follow = metadata(pin['path'])
+    require(pin['sha256'] == 'e9581cf982e3c83fa85413a4eb707afef001290c6137563a34a9c679a54514e6' and
+            follow['original_freeze_unchanged'] is True and len(follow['files']) == 4,
+            'exact additive original7058B source metadata layer')
+    for row in follow['files']:
+        read(old/row['file'],pin=row)
+        read(old/'metadata-v2/frozen'/row['file'],pin=row)
+    for row in follow['parent_metadata']:
+        read(row['path'],pin=row)
+    for row in follow['native_compile_sources']:
+        read(old/row['file'],pin=row)
+    return original,follow
+
+
+def normalized_lines(raw, edits, require):
+    lines = raw.decode().splitlines(True)
+    cursor,result = 0,[]
+    for edit in edits:
+        lo,hi = edit['original_lines']
+        require(type(lo) is type(hi) is int and cursor <= lo <= hi <= len(lines) and
+                edit['tag'] in ('insert','replace','delete') and
+                ''.join(lines[lo:hi]) == edit['before'],
+                'exact ordered original source block, never inferred arithmetic equivalence')
+        result += lines[cursor:lo]
+        result.append(edit['after'])
+        cursor = hi
+    result += lines[cursor:]
+    return ''.join(result).encode()
+
+
+def conserve_children(config, metadata, require):
+    old = Path(config['original_harness']).parent
+    source = metadata(old/'freeze-manifest.json')
+    pins = {r['path']:r for r in source['files']}
+    for follow in ('followup-v2','followup-v3'):
+        pins.update({r['path']:r for r in metadata(old/follow/'freeze-manifest.json')['files']})
+    pins.update({r['path']:r for r in source['unchanged_dependency_source_pins']})
+    require(len(pins) == len(config['original_child_map']) == 14,
+            'all8 original implementation and6 original dependency assertions conserved')
+    seen = set()
+    for row in config['original_child_map']:
+        rel = row['file']
+        require(rel in pins and rel not in seen and
+                (row['original']['bytes'],row['original']['sha256']) ==
+                (pins[rel]['bytes'],pins[rel]['sha256']) and
+                row['isolated']['path'] == str(Path(config['workspace_root'])/rel),
+                'exact original/final typed source child binding')
+        seen.add(rel)
+        before = _raw(row['original']['path'],row['original'],require)
+        after = _raw(row['isolated']['path'],row['isolated'],require)
+        if row['classification'] == 'BYTE_IDENTICAL_ROOT_REMAP':
+            require(not row['line_deltas'] and before == after,'original math/source child byte identity')
+        else:
+            require(row['classification'] == 'ROOT_REVIEWED_SOURCE_COMPOSITION_OR_TEST_CORRECTION' and
+                    normalized_lines(before,row['line_deltas'],require) == after,
+                    'only explicit ROOT-reviewed composition/test source deltas')
+    return pins
+
+
+def _copies(here, config, require):
+    require(str(here) == config['relocated_harness'] and here.resolve() == here and
+            here != Path(config['original_harness']),'explicit distinct original/relocated harness')
+    require(len(config['copy_map']) == 14 and len({r['file'] for r in config['copy_map']}) == 14,
+            'original13 plus real source-pin metadata all conserved')
+    edits = config['native_source_replacements']
+    for row in config['copy_map']:
+        name = row['file']
+        require(name == Path(name).name and row['original']['path'] == str(Path(config['original_harness'])/name)
+                and row['relocated']['path'] == str(here/name),'exact copied source filename/path')
+        before = _raw(row['original']['path'],row['original'],require)
+        after = _raw(row['relocated']['path'],row['relocated'],require)
+        policy = row['policy']
+        if policy == 'BYTE_IDENTICAL':
+            require(before == after,'original Rust/validator/oracle/owner source bytes unchanged')
+        elif policy == 'ONLY_LOCAL_DEPENDENCY_ROOT_REMAP':
+            a,b = (config['main_root']+'/crates/').encode(),(config['workspace_root']+'/crates/').encode()
+            require(name == 'Cargo.toml' and before.count(a) == 4 and after == before.replace(a,b),
+                    'only four absolute Cargo crate roots relocate')
+        elif policy == 'ONLY_FROZEN_SOURCE_DISPATCH':
+            text = before.decode();pos = text.index('def frozen():')
+            require(name == 'support.py' and after == (text[:pos]+config['support_frozen_dispatch']).encode(),
+                    'only source freeze dispatch changes in bounded transport helper')
+        else:
+            require(name == 'run_native.py' and policy == 'ONLY_SOURCE_ATTESTATION_AND_SAFE_SIGNAL_DISPATCH' and
+                    normalized(before,edits,require) == after,
+                    'only enumerated source/path attestation and safe cancellation dispatch')
+
+
+def frozen(here, metadata, read, require, sha):
+    manifest = metadata(here/'prepared-manifest.json')
+    rows = manifest['files']
+    require(type(rows) is list and len({r['file'] for r in rows}) == len(rows),
+            'finite unique relocated source inventory')
+    for row in rows:
+        rel = Path(row['file'])
+        require(not rel.is_absolute() and all(p not in ('.','..') for p in rel.parts),
+                'contained source/metadata preparation files')
+        read(here/rel,pin=row)
+        read(here/'frozen-prepared'/rel,pin=row)
+    config = metadata(here/'relocation.json')
+    require(config['MAIN_merged'] is False and config['full_model_StateRoot_GPU_or_performance_qualified'] is False,
+            'isolated private-wide-only source scope')
+    _original(config,metadata,read,require)
+    _copies(here,config,require)
+    workspace(config,metadata,require)
+    conserve_children(config,metadata,require)
+    for row in config['accepted_reference_metadata']:
+        read(row['path'],pin=row)
+    return sha(here/'prepared-manifest.json')
+
+
+def validate_build(value, config, expected, here, freeze, require):
+    require(value['status'] == config['required_isolated_status'] and value['MAIN_merged'] is False and
+            value['full_model_StateRoot_GPU_or_performance_qualified'] is False and value['build_exit_code'] == 0 and
+            value['prepared_manifest_sha256'] == freeze and
+            value['qualification_manifest_sha256'] == config['original_preparation']['sha256'] and
+            value['qualification_metadata_followup_manifest_sha256'] == config['original_metadata_followup']['sha256'] and
+            value['root_source_freeze_sha256'] == config['source_freeze']['sha256'] and
+            value['root_worktree'] == config['workspace_root'] and value['qualification_directory'] == str(here),
+            'actual explicitly isolated source/build receipt; never MAIN merged or marker-only proof')
+    actual = value['files']
+    require(type(actual) is list and 526 <= len(actual) == len(expected) <= 527 and
+            len({r['path'] for r in actual}) == len(actual) and {r['path']:r for r in actual} == expected,
+            'all521 ROOT plus local5(+lock) source inputs, no aliases/missing/extras')
+
+
+def implementation(here, path, metadata, read, require, sha):
+    freeze = frozen(here,metadata,read,require,sha)
+    config = metadata(here/'relocation.json')
+    value = metadata(path)
+    expected = workspace(config,metadata,require)
+    require(config['local_build_names'] == ['Cargo.toml','main.rs','factory.rs','source-pin.json','source-plan.json'],
+            'complete five local compile inputs, including actual embedded source plan')
+    for name in config['local_build_names']:
+        row = next(r for r in config['copy_map'] if r['file'] == name)['relocated']
+        expected[row['path']] = row
+    lock = here/'Cargo.lock'
+    if lock.exists():
+        raw = read(lock,cap=2*1024**2)
+        expected[str(lock)] = dict(path=str(lock),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
+    validate_build(value,config,expected,here,freeze,require)
+    actual = value['files']
+    for row in actual:
+        read(row['path'],pin=row)
+    return sha(path)
+
+
+def reference(here, directory, metadata, read, require):
+    config = metadata(here/'relocation.json')
+    require(directory.resolve() == directory == Path(config['accepted_reference_directory']),
+            'exact existing independently accepted oracle path; no source/payload copy')
+    for row in config['accepted_reference_metadata']:
+        read(row['path'],pin=row)
+    valid = metadata(directory/'validation.json')
+    require(valid['status'] == 'PASS_ORIGINAL27B_PURE_OPERATOR_REFERENCE_ONLY' and
+            valid['freeze_sha256'] == config['original_preparation']['sha256'] and
+            valid['cases'] == 32 and valid['roles'] == 16 and valid['words'] == 881856 and
+            valid['native_CPU_parity_full_model_StateRoot_GPU_or_performance_qualified'] is False,
+            'unchanged original public32-case oracle, not this relocated native preparation')
+    return valid
+
--- /dev/null
+++ b/relocation.json
@@ -0,0 +1,1174 @@
+{
+  "MAIN_merged": false,
+  "ROOT_composition_journal_pins": [
+    {
+      "bytes": 894,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-parent-source-verification-1.json",
+      "sha256": "a2b255f50a037c42f32a3d831afc7f9c65157cd76fc0c25d70a4f2e319c59616"
+    },
+    {
+      "bytes": 6676,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/coefficient-core-composition-1.json",
+      "sha256": "251b74b71ec6ec150292784af7b90b2595fdb9fe38f65e96992c50bd438d4a8e"
+    },
+    {
+      "bytes": 229,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/coefficient-cold-refusal-composition-1.json",
+      "sha256": "426ef7301495c43a836bf7a3683101c61c5a11746245a39da1663f427a8f8901"
+    },
+    {
+      "bytes": 2732,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/gamma-composition-1.json",
+      "sha256": "4ce2b9c34f6a370a05e1592cc3bb772707ea43e8067685e1cd765227282c871f"
+    },
+    {
+      "bytes": 1344,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/gamma-composition-2.json",
+      "sha256": "5a613cfb590d2f1b4557b050c0284fcc1ae844d370176e4610d3beaaffc37def"
+    },
+    {
+      "bytes": 367,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/gamma-composition-3.json",
+      "sha256": "3d137e5fc87d304cc6a53f06ca1c1544dd1a36bdf40d74a8b94ab476ff84f24d"
+    },
+    {
+      "bytes": 379,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/gamma-composition-4.json",
+      "sha256": "17cd79987b3ef158eb494606a7f8ba80753f2c4523d9abd0d288bb1225863bdb"
+    },
+    {
+      "bytes": 361,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/gamma-composition-5.json",
+      "sha256": "15f4b9ddc5fb8c2dbadd8d09bdfbd042ff27ae493463cac09f0195b158ce9b6c"
+    }
+  ],
+  "ROOT_focused_check_pin": {
+    "bytes": 1738,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-cpu-focused-7.json",
+    "sha256": "7d72b5c08e45316e5ab8ec29a8e605392abca1cc1e5fd835aeb97b69ef82a8aa"
+  },
+  "ROOT_original_source_freeze": {
+    "bytes": 98702,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-1.json",
+    "sha256": "a227fa210ed042be140b3c3a88a5672dbb3ed0a7fced8357e10f9f780506bac1"
+  },
+  "ROOT_source_patch_pins": [
+    {
+      "bytes": 988,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-coefficient-output-size-adapter-2.patch",
+      "sha256": "035c17b5db5559b473249561dd5cf99428ef2f7239778f77ee4a59fd3f1ece8f"
+    },
+    {
+      "bytes": 805,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-coefficient-test-array-style-3.patch",
+      "sha256": "30130924e17c8d8c08e0582c77dc2e8f46ec3e99342aa3a938036a0264588e3a"
+    },
+    {
+      "bytes": 1201,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-window-test-visibility-import-4.patch",
+      "sha256": "11e8fe4b19d5f18873a289db0dabd5a69a02bd2fa66ae8c3a10faf73e65e236f"
+    },
+    {
+      "bytes": 362,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-coefficient-test-array-style-5.patch",
+      "sha256": "10c3b4eec7d42cca192f9972c05d20e2ef88988c1f22a84d7d456395da015c6a"
+    },
+    {
+      "bytes": 3747,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-fixture-contract-reconciliation-6.patch",
+      "sha256": "8ad1b36e65f521621630de8018edea13857e74a75e7a2ee97e351e8f870788f7"
+    },
+    {
+      "bytes": 558,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-fixture-format-6.patch",
+      "sha256": "252c4e4d6bfffdc2267086341fdeeaf877be8489003fd68449599ce9dc987c19"
+    },
+    {
+      "bytes": 1527,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-model-tied-role-parameter-lifetime-fixture-7.patch",
+      "sha256": "d83a5e9564e7420c400422f69ae86b07fd9e7ad015046c91d6811be3c50d1632"
+    }
+  ],
+  "accepted_reference_directory": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/root-reference-1",
+  "accepted_reference_metadata": [
+    {
+      "bytes": 644,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/root-reference-1/validation.json",
+      "sha256": "030b7fa0487da820fa74a6698329cef4adc87f5cb9bcbfa2fb2a9533bca8fc3a"
+    },
+    {
+      "bytes": 22473,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/root-reference-1/invocation.json",
+      "sha256": "536b090b48d3e7cdd7e633c77dc7bd1713afe567aecbd8df04892b81f05f3984"
+    },
+    {
+      "bytes": 49157,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/root-reference-1/captured/manifest.json",
+      "sha256": "7289c8b8599bc64d316974ca33298e8dd02e0c843dff4f2e2c99523c4ce6904e"
+    }
+  ],
+  "copy_map": [
+    {
+      "file": "Cargo.toml",
+      "original": {
+        "bytes": 799,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/Cargo.toml",
+        "sha256": "af3da360933dc1dfa0e99da8ddd0ff4676d12ed700edc1400c81fdf64de7de84"
+      },
+      "policy": "ONLY_LOCAL_DEPENDENCY_ROOT_REMAP",
+      "relocated": {
+        "bytes": 963,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/Cargo.toml",
+        "sha256": "0688aa2b13e9836b2b1987fa5329278643d97450312592ff26c8669e882f7da1"
+      }
+    },
+    {
+      "file": "ROOT.md",
+      "original": {
+        "bytes": 5511,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/ROOT.md",
+        "sha256": "17bf7285ebaa0eb3f761d107fe7d4fcc16ceef5e5a2e333bbb73f192df6c5495"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 5511,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/ROOT.md",
+        "sha256": "17bf7285ebaa0eb3f761d107fe7d4fcc16ceef5e5a2e333bbb73f192df6c5495"
+      }
+    },
+    {
+      "file": "capture.py",
+      "original": {
+        "bytes": 10129,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/capture.py",
+        "sha256": "29c7bff1c992aaa2172e429ee5988cd7f3ecf44da2d9d78eac996436da518f57"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 10129,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/capture.py",
+        "sha256": "29c7bff1c992aaa2172e429ee5988cd7f3ecf44da2d9d78eac996436da518f57"
+      }
+    },
+    {
+      "file": "factory.rs",
+      "original": {
+        "bytes": 29779,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/factory.rs",
+        "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 29779,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/factory.rs",
+        "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
+      }
+    },
+    {
+      "file": "main.rs",
+      "original": {
+        "bytes": 14126,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/main.rs",
+        "sha256": "30d4507a35c6709d3fd20896ddbfe978ddb14850d2e5363396c551adc512315a"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 14126,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/main.rs",
+        "sha256": "30d4507a35c6709d3fd20896ddbfe978ddb14850d2e5363396c551adc512315a"
+      }
+    },
+    {
+      "file": "owned_container_v4.py",
+      "original": {
+        "bytes": 11840,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/owned_container_v4.py",
+        "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 11840,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/owned_container_v4.py",
+        "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+      }
+    },
+    {
+      "file": "run_native.py",
+      "original": {
+        "bytes": 9011,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/run_native.py",
+        "sha256": "142884baae210920f07aca50b5116cb7d8168f3d0b1143b339dcca55df697919"
+      },
+      "policy": "ONLY_SOURCE_ATTESTATION_AND_SAFE_SIGNAL_DISPATCH",
+      "relocated": {
+        "bytes": 7289,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/run_native.py",
+        "sha256": "bee65075d182d258ea1c0ec9746ec6a6f448fdd35ce018eaa4b6762723183db8"
+      }
+    },
+    {
+      "file": "run_reference.py",
+      "original": {
+        "bytes": 8272,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/run_reference.py",
+        "sha256": "3c3acf0fa623ff015a94f4cbf3845d59bbfc53921218e1aa26d936f794986ada"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 8272,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/run_reference.py",
+        "sha256": "3c3acf0fa623ff015a94f4cbf3845d59bbfc53921218e1aa26d936f794986ada"
+      }
+    },
+    {
+      "file": "source-plan.json",
+      "original": {
+        "bytes": 20634,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/source-plan.json",
+        "sha256": "c241955aa8859e295af65186ccfb136b5f78b277156219b6583556b8f8ad711f"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 20634,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/source-plan.json",
+        "sha256": "c241955aa8859e295af65186ccfb136b5f78b277156219b6583556b8f8ad711f"
+      }
+    },
+    {
+      "file": "support.py",
+      "original": {
+        "bytes": 5511,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/support.py",
+        "sha256": "3f8f4140a038e67e3620bd71492b2e1f5dc8159c84b160824dd0144b527cb11a"
+      },
+      "policy": "ONLY_FROZEN_SOURCE_DISPATCH",
+      "relocated": {
+        "bytes": 5399,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/support.py",
+        "sha256": "dac4c61d737ad1961b17e84b30bfc6bbbd1534adb7a43959bc126d24832f1a5e"
+      }
+    },
+    {
+      "file": "tensor_bits.py",
+      "original": {
+        "bytes": 1052,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/tensor_bits.py",
+        "sha256": "9f0d007e1b3e8c536609361071e3a7815fd3d3a0c5af26f16a5d601d093bc7a7"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 1052,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/tensor_bits.py",
+        "sha256": "9f0d007e1b3e8c536609361071e3a7815fd3d3a0c5af26f16a5d601d093bc7a7"
+      }
+    },
+    {
+      "file": "test_contracts.py",
+      "original": {
+        "bytes": 1569,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/test_contracts.py",
+        "sha256": "6d262a20430696d530bdd9ddbc9745f3e09e3ced29d302fec51b0db6efcbf4a8"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 1569,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/test_contracts.py",
+        "sha256": "6d262a20430696d530bdd9ddbc9745f3e09e3ced29d302fec51b0db6efcbf4a8"
+      }
+    },
+    {
+      "file": "validate_reference.py",
+      "original": {
+        "bytes": 7363,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/validate_reference.py",
+        "sha256": "1b8472c728fdea2ccb0bf3dc568db4e61a10432b92e5003a2e9b9df732e6964d"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 7363,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/validate_reference.py",
+        "sha256": "1b8472c728fdea2ccb0bf3dc568db4e61a10432b92e5003a2e9b9df732e6964d"
+      }
+    },
+    {
+      "file": "source-pin.json",
+      "original": {
+        "bytes": 7058,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/source-pin.json",
+        "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+      },
+      "policy": "BYTE_IDENTICAL",
+      "relocated": {
+        "bytes": 7058,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/source-pin.json",
+        "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+      }
+    }
+  ],
+  "full_model_StateRoot_GPU_or_performance_qualified": false,
+  "local_build_names": [
+    "Cargo.toml",
+    "main.rs",
+    "factory.rs",
+    "source-pin.json",
+    "source-plan.json"
+  ],
+  "main_root": "/home/pvl/spaces/abird/src/zerogravity",
+  "native_source_replacements": [
+    {
+      "after": "from support import HERE,frozen,metadata,read,require,sha,write\nfrom relocated_source import implementation, reference\nfrom sigterm_cancellation import SigtermCancellation",
+      "before": "from support import HERE,frozen,metadata,require,sha,write",
+      "expected_occurrences": 1
+    },
+    {
+      "after": "def main(cancel):",
+      "before": "def main():",
+      "expected_occurrences": 1
+    },
+    {
+      "after": "    cancel.check()\n    freeze=frozen()",
+      "before": "    freeze=frozen()",
+      "expected_occurrences": 1
+    },
+    {
+      "after": "    source_attestation=implementation(HERE,args.build_attestation,metadata,read,require,sha)\n    build=metadata(args.build_attestation)\n    require(build['binary_sha256']==args.binary_sha256 and\n            Path(build['binary']).resolve()==args.binary.resolve(),\n            'ROOT actual isolated executable binding; source markers alone are not compiler proof')\n    require(args.new_output.is_absolute()and args.new_output.parent.resolve()==HERE,\n            'owned relocated native output only')\n    validation=reference(HERE,args.reference_directory,metadata,read,require)\n    worktree=Path(build['root_worktree'])\n",
+      "before": "    build=metadata(args.build_attestation)\n    require(build['status']=='ROOT_COMPILED_ORIGINAL27B_PURE_ROLES_ONLY'and build['build_exit_code']==0 and\n            build['binary_sha256']==args.binary_sha256 and Path(build['binary']).resolve()==args.binary.resolve()and\n            build['candidate_manifest_sha256']=='0f9f2a8ff4ff28f5f3284063aecff35508ca61cdd083dac454400e7aec896d3e'and\n            build['exact_hidden_followup_manifest_sha256']=='65cd436471e26679d4ff596538fda31c9b1c6f28cef6d9a55f7c99e9d28ca82b'and\n            build['completed_cancel_followup_manifest_sha256']=='7759599f51215de539167a3f71b0efb39aace80fcf817b3433fbe5eba6dfdc6c',\n            'ROOT actual build/candidate/followup binding; marker alone is not compilation proof')\n    actual_files=build['actual_source_files'];require(len(actual_files)==8,'all eight actual implementation source paths')\n    worktree=Path(build['root_worktree']);require(worktree.is_absolute()and worktree.is_dir(),'explicit ROOT build worktree')\n    expected=metadata(HERE.parent/'freeze-manifest.json')['files']\n    fixes={row['path']:row for row in metadata(HERE.parent/'followup-v2/freeze-manifest.json')['files']}\n    fixes.update({row['path']:row for row in metadata(HERE.parent/'followup-v3/freeze-manifest.json')['files']})\n    for row in expected:\n        pin=fixes.get(row['path'],row);actual=[v for v in actual_files if v['path']==row['path']]\n        require(len(actual)==1 and actual[0]['sha256']==pin['sha256']and actual[0]['bytes']==pin['bytes'],\n                'actual compiled source attestation matches exact reviewed implementation')\n        require((worktree/row['path']).stat().st_size==pin['bytes']and sha(worktree/row['path'])==pin['sha256'],\n                'actual ROOT implementation source bytes still exact')\n    for pin in metadata(HERE.parent/'freeze-manifest.json')['unchanged_dependency_source_pins']:\n        require((worktree/pin['path']).stat().st_size==pin['bytes']and sha(worktree/pin['path'])==pin['sha256'],\n                'unchanged original CPU math/core/source closure bytes')\n    require(args.reference_directory.is_absolute()and args.reference_directory.parent.resolve()==HERE and\n            args.new_output.is_absolute()and args.new_output.parent.resolve()==HERE,'owned reference/native outputs')\n    validation=metadata(args.reference_directory/'validation.json')\n    require(validation['status']=='PASS_ORIGINAL27B_PURE_OPERATOR_REFERENCE_ONLY'and\n            validation['freeze_sha256']==freeze and validation['manifest_sha256']==sha(args.reference_directory/'captured/manifest.json'),\n            'independently validated genuine public raw reference first')\n",
+      "expected_occurrences": 1
+    },
+    {
+      "after": "             'scope':'same-engine private original27B pure roles only; no full-model/state/GPU',\n             'MAIN_merged':False,'root_source_freeze_sha256':build['root_source_freeze_sha256'],\n             'original_oracle_preparation_sha256':validation['freeze_sha256']}",
+      "before": "             'scope':'same-engine private original27B pure roles only; no full-model/state/GPU'}",
+      "expected_occurrences": 1
+    },
+    {
+      "after": "        cancel.check() # Record-only signals cannot interrupt original Popen ownership assignment.",
+      "before": "        def cancelled(_signal,_frame):raise InterruptedError('owned native pure replay cancelled')\n        signal.signal(signal.SIGTERM,cancelled)",
+      "expected_occurrences": 1
+    },
+    {
+      "after": "            receipt['exit_code']=cancel.observe(log,child,1800)",
+      "before": "            receipt['exit_code']=log.observe(child,1800)",
+      "expected_occurrences": 1
+    },
+    {
+      "after": "        cancel.check() # Original cleanup completed before synchronous cancellation is raised.\n        require(failure is None and receipt.get('exit_code')==0",
+      "before": "        require(failure is None and receipt.get('exit_code')==0",
+      "expected_occurrences": 1
+    },
+    {
+      "after": "        require(implementation(HERE,args.build_attestation,metadata,read,require,sha)==source_attestation and\n                reference(HERE,args.reference_directory,metadata,read,require)==validation,\n                'whole isolated ROOT source/build and accepted oracle remain exact')\n",
+      "before": "        for row in expected:\n            pin=fixes.get(row['path'],row)\n            require(sha(worktree/row['path'])==pin['sha256'],'actual ROOT implementation unchanged after native replay')\n        for pin in metadata(HERE.parent/'freeze-manifest.json')['unchanged_dependency_source_pins']:\n            require(sha(worktree/pin['path'])==pin['sha256'],'original math/core/source dependencies unchanged after native replay')\n",
+      "expected_occurrences": 1
+    },
+    {
+      "after": "        cancel.check()\n        receipt.update(status='PASS_OWNED_ORIGINAL27B_PURE_ROLE_REPLAY_ONLY'",
+      "before": "        receipt.update(status='PASS_OWNED_ORIGINAL27B_PURE_ROLE_REPLAY_ONLY'",
+      "expected_occurrences": 1
+    },
+    {
+      "after": "if __name__=='__main__':\n    with SigtermCancellation() as cancellation:\n        main(cancellation)",
+      "before": "if __name__=='__main__':main()",
+      "expected_occurrences": 1
+    }
+  ],
+  "normalization_exceptions": [
+    {
+      "classification": "adapter_only_two_head_count_arguments",
+      "file": "crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs",
+      "isolated": {
+        "bytes": 6351,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs",
+        "sha256": "c4cbb9c57cb1b39db15aae4533fa3baf3724974f41b74a9b842abb572aca8d60"
+      },
+      "original": {
+        "bytes": 6323,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/proposed/crates/zerogravity-model-qwen35/src/host_math/coefficient_preparation.rs",
+        "sha256": "23929aec9bfc531b8cb0d1259017358f706454c1ae0636c09ff442950c51efcf"
+      },
+      "replacements": [
+        {
+          "after": "            ));\n        }\n        #[cfg(all(feature = \"original-hf-binding\", target_os = \"linux\"))]\n        let workspace = self.prepare_output(None, 2 * heads, 1)?; // BEFORE both private copies.\n        let decode = |raw: &[u8]| {\n            raw.chunks_exact(2)\n                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))\n",
+          "before": "            ));\n        }\n        #[cfg(all(feature = \"original-hf-binding\", target_os = \"linux\"))]\n        let workspace = self.prepare_output(None)?; // BEFORE both private copies.\n        let decode = |raw: &[u8]| {\n            raw.chunks_exact(2)\n                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))\n",
+          "expected_occurrences": 1
+        },
+        {
+          "after": "                    ));\n                }\n            }\n            self.prepare_output(Some(alpha), 3 * heads, 1)? // Shared aggregate for ALL3 outputs.\n        };\n        let prepared = match &self.source {\n            HostSource::Loaded(owner) => {\n",
+          "before": "                    ));\n                }\n            }\n            self.prepare_output(Some(alpha))? // Shared aggregate for ALL3 outputs.\n        };\n        let prepared = match &self.source {\n            HostSource::Loaded(owner) => {\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_two_vec_array_substitutions",
+      "file": "crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs",
+      "isolated": {
+        "bytes": 7234,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs",
+        "sha256": "a3032c27074f9a49545052f3d96062a02539616f45c1d0d8e3c9d578ed75e572"
+      },
+      "original": {
+        "bytes": 7242,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/proposed/crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs",
+        "sha256": "031299ebca94fd5a4d1ea81ba2e8447e2d94bebe83de9076f877e6d7ded3dcd5"
+      },
+      "replacements": [
+        {
+          "after": "    #[test]\n    fn raw_f32_dt_is_not_bf16_rounded_or_adjusted() {\n        let word = 0x3380_0001u32;\n        let raw = [word.to_le_bytes(); 16].concat();\n        let zero = [0.0; 16];\n        let out = prepare_gdn_coefficients(\n            source(),\n",
+          "before": "    #[test]\n    fn raw_f32_dt_is_not_bf16_rounded_or_adjusted() {\n        let word = 0x3380_0001u32;\n        let raw = vec![word.to_le_bytes(); 16].concat();\n        let zero = [0.0; 16];\n        let out = prepare_gdn_coefficients(\n            source(),\n",
+          "expected_occurrences": 1
+        },
+        {
+          "after": "        )\n        .unwrap();\n        assert!(out.biased_alpha().iter().all(|v| v.to_bits() == word));\n        assert_eq!(raw, [word.to_le_bytes(); 16].concat());\n    }\n    #[test]\n    fn source_raw_dtype_span_and_nonfinite_refuse() {\n",
+          "before": "        )\n        .unwrap();\n        assert!(out.biased_alpha().iter().all(|v| v.to_bits() == word));\n        assert_eq!(raw, vec![word.to_le_bytes(); 16].concat());\n    }\n    #[test]\n    fn source_raw_dtype_span_and_nonfinite_refuse() {\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_same_H48_budget_valid_schedule_parameter_drop",
+      "file": "crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs",
+      "isolated": {
+        "bytes": 13508,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs",
+        "sha256": "b6f7fcb6d8828051b6004b90f10c7b549303d94c9c1bf358ca701d3fda524b40"
+      },
+      "original": {
+        "bytes": 13482,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/test-style-v2/frozen/crates/zerogravity-model-qwen35/src/gated_rms_window_tests.rs",
+        "sha256": "1ee2170fceaebb0bbc0a6f48daf4d73687f6429eb48ca692a65c15540d798763"
+      },
+      "replacements": [
+        {
+          "after": ") -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {\n    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();\n    let text = &mut cfg[\"text_config\"];\n    text[\"num_hidden_layers\"] = json!(2);\n    text[\"full_attention_interval\"] = json!(2);\n    text[\"layer_types\"] = json!([\"linear_attention\", \"full_attention\"]);\n    text[\"hidden_size\"] = json!(256);\n    text[\"linear_num_key_heads\"] = json!(16);\n    text[\"linear_num_value_heads\"] = json!(48);\n",
+          "before": ") -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {\n    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();\n    let text = &mut cfg[\"text_config\"];\n    text[\"num_hidden_layers\"] = json!(1);\n    text[\"layer_types\"] = json!([\"linear_attention\"]);\n    text[\"hidden_size\"] = json!(256);\n    text[\"linear_num_key_heads\"] = json!(16);\n    text[\"linear_num_value_heads\"] = json!(48);\n",
+          "expected_occurrences": 1
+        },
+        {
+          "after": "            .is_err()\n    );\n    assert_eq!(reserved(&host), before);\n    // Independent declared H48 payloads and conservative scratch floors.\n    let input_need = (2 * 48 * 128 * 4 + 64 * 1024) as u64;\n    let output_need = (48 * 128 * 4 + 960 * 1024) as u64;\n    let pressure = host\n        .arbiter\n        .admit(\n",
+          "before": "            .is_err()\n    );\n    assert_eq!(reserved(&host), before);\n    let input_need = (crate::host_math::gated_rms::INPUT_BYTES\n        + crate::host_math::gated_rms::INPUT_SCRATCH) as u64;\n    let output_need = (crate::host_math::gated_rms::OUTPUT_BYTES\n        + crate::host_math::gated_rms::OUTPUT_SCRATCH) as u64;\n    let pressure = host\n        .arbiter\n        .admit(\n",
+          "expected_occurrences": 1
+        },
+        {
+          "after": "    same(expected.output_bf16(), result.output_bf16().unwrap());\n    drop(expected);\n    drop(gamma);\n    drop(parameter);\n    let bits = result\n        .output_bf16()\n        .unwrap()\n",
+          "before": "    same(expected.output_bf16(), result.output_bf16().unwrap());\n    drop(expected);\n    drop(gamma);\n    let bits = result\n        .output_bf16()\n        .unwrap()\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_unused_import_removal",
+      "file": "crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs",
+      "isolated": {
+        "bytes": 11198,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs",
+        "sha256": "c2a0115da9308799aacf450c9c1c12f7b05141b9f57d32479aef2e581866a0d4"
+      },
+      "original": {
+        "bytes": 11219,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/proposed/crates/zerogravity-model-qwen35/src/cached_gdn_window_tests.rs",
+        "sha256": "d255862bbc9e97b2867c12418df0fac628caaedb0099b5d5e23a55d08506fa84"
+      },
+      "replacements": [
+        {
+          "after": "mod source_gamma;\n\nuse super::*;\nuse crate::accelerator::program::CachedGdnContract;\nuse crate::operator_slice::cached_gdn::CachedGdnSliceInput;\nuse zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};\n\n",
+          "before": "mod source_gamma;\n\nuse super::*;\nuse crate::accelerator::program::{CachedGdnContract, CachedGdnOperands};\nuse crate::operator_slice::cached_gdn::CachedGdnSliceInput;\nuse zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};\n\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_array_and_real_pressure_chunks",
+      "file": "crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
+      "isolated": {
+        "bytes": 8890,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
+        "sha256": "74b8a311402e42c04cd3060dd34a8c4dfa05dd2d061f4d2ae89f896eecb7fec8"
+      },
+      "original": {
+        "bytes": 8682,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/proposed/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
+        "sha256": "36e443b77bd8641c4005d88ee51316196a0507905a7b5cf47bdb30b3356fcfcf"
+      },
+      "replacements": [
+        {
+          "after": "    )\n}\nfn raw(value: u16) -> Vec<u8> {\n    [value.to_le_bytes(); 16].concat()\n}\nfn input<'a>(\n    source: crate::accelerator::SourceIdentity,\n",
+          "before": "    )\n}\nfn raw(value: u16) -> Vec<u8> {\n    vec![value.to_le_bytes(); 16].concat()\n}\nfn input<'a>(\n    source: crate::accelerator::SourceIdentity,\n",
+          "expected_occurrences": 1
+        },
+        {
+          "after": "            .is_err()\n    );\n    assert_eq!(reserved(&admission), before);\n    let mut pressure = Vec::new();\n    loop {\n        let remaining = admission\n            .arbiter\n            .snapshot(&admission.resource)\n            .unwrap()\n            .remaining;\n        if remaining == 0 {\n            break;\n        }\n        pressure.push(\n            source\n                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)\n                .unwrap(),\n        );\n    }\n    let pressured = reserved(&admission);\n    assert!(\n        operators\n",
+          "before": "            .is_err()\n    );\n    assert_eq!(reserved(&admission), before);\n    let remaining = admission\n        .arbiter\n        .snapshot(&admission.resource)\n        .unwrap()\n        .remaining;\n    let pressure = source.reserve_workspace(remaining as usize).unwrap();\n    let pressured = reserved(&admission);\n    assert!(\n        operators\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_valid_layer_schedule",
+      "file": "crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs",
+      "isolated": {
+        "bytes": 10274,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs",
+        "sha256": "8a70bd9f52f299ac1cbac9df3e420f1096edd0a5acd77779bcae4d5aac99dcbc"
+      },
+      "original": {
+        "bytes": 10208,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/proposed/crates/zerogravity-model-qwen35/src/cached_gdn_h48_tests.rs",
+        "sha256": "7e47e38f7b50277261de04cad6567cfbae4120015aa6b68d95db0433003e2db1"
+      },
+      "replacements": [
+        {
+          "after": "fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {\n    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();\n    let text = &mut cfg[\"text_config\"];\n    text[\"num_hidden_layers\"] = json!(2);\n    text[\"full_attention_interval\"] = json!(2);\n    text[\"layer_types\"] = json!([\"linear_attention\", \"full_attention\"]);\n    text[\"hidden_size\"] = json!(256);\n    text[\"linear_num_key_heads\"] = json!(16);\n    text[\"linear_num_value_heads\"] = json!(48);\n",
+          "before": "fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {\n    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();\n    let text = &mut cfg[\"text_config\"];\n    text[\"num_hidden_layers\"] = json!(1);\n    text[\"layer_types\"] = json!([\"linear_attention\"]);\n    text[\"hidden_size\"] = json!(256);\n    text[\"linear_num_key_heads\"] = json!(16);\n    text[\"linear_num_value_heads\"] = json!(48);\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_explicit_closed_embedding_role_negative",
+      "file": "crates/zerogravity-model-qwen35/src/attention_slice_tests.rs",
+      "isolated": {
+        "bytes": 16955,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/attention_slice_tests.rs",
+        "sha256": "f1cc417f41b62acb368161396ff703154c1fcce9ad480ff5c14c9c09518fdf9b"
+      },
+      "original": {
+        "bytes": 16627,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/baseline-attention-slice-tests.rs",
+        "sha256": "57e3abc73a91cf7f661b64db2865c47c21e9eb9c2ca4f6483546d60643b2730e"
+      },
+      "replacements": [
+        {
+          "after": "        math.ordinary_rms_profiled(gdn_gamma, &imported, OrdinaryRmsContract::TorchCpuBf16Avx2)\n            .is_err()\n    );\n    // Tied output and embedding share a descriptor; the explicit role API\n    // rejects Embedding before importing another private value.\n    assert!(\n        operators\n            .execute_projection_slice(\n                QwenMatrixRole::Embedding,\n                crate::operator_slice::pure::PureSliceInput::from_bf16_bytes(\n                    source.source_identity(),\n                    ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,\n                    &bytes,\n                )\n                .unwrap(),\n                None,\n            )\n            .is_err()\n    );\n    assert!(\n        math.ordinary_rms_profiled(\n",
+          "before": "        math.ordinary_rms_profiled(gdn_gamma, &imported, OrdinaryRmsContract::TorchCpuBf16Avx2)\n            .is_err()\n    );\n    assert!(\n        math.project_profiled(\n            operators\n                .matrix_descriptor(QwenMatrixRole::FfnGate(3))\n                .unwrap(),\n            &imported,\n            ProjectionContract::TorchCpuBf16Avx2\n        )\n        .is_err()\n    );\n    assert!(\n        math.ordinary_rms_profiled(\n",
+          "expected_occurrences": 1
+        }
+      ]
+    },
+    {
+      "classification": "test_only_distinct_fresh_source_identity",
+      "file": "crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs",
+      "isolated": {
+        "bytes": 16480,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs",
+        "sha256": "9769cf2902bf6fda572a4a52010cfbedb5bbc06e9869f3f6306043290389df8f"
+      },
+      "original": {
+        "bytes": 16477,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/followup-v3/proposed/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs",
+        "sha256": "7108060b361abe6d52680e161b75e3ec16e2057a019ace4c6455400b5067617d"
+      },
+      "replacements": [
+        {
+          "after": "}\n\n#[test]\nfn separately_opened_same_content_source_cannot_rebind_a_retained_input() {\n    let fixture = Fixture::new();\n    let loaded = source_with_options(&fixture, false, 256, true, |_| {});\n    let (source_a, admission_a) = windows(&fixture, &loaded);\n    let (source_b, admission_b) = windows(&fixture, &loaded);\n    assert_ne!(source_a.source_identity(), source_b.source_identity());\n    assert!(!source_a.same_issuer(&source_b));\n    let operators_a = bind_windows(source_a.clone(), &loaded);\n    let operators_b = bind_windows(source_b.clone(), &loaded);\n",
+          "before": "}\n\n#[test]\nfn same_content_foreign_window_issuer_cannot_rebind_a_retained_input() {\n    let fixture = Fixture::new();\n    let loaded = source_with_options(&fixture, false, 256, true, |_| {});\n    let (source_a, admission_a) = windows(&fixture, &loaded);\n    let (source_b, admission_b) = windows(&fixture, &loaded);\n    assert_eq!(source_a.source_identity(), source_b.source_identity());\n    assert!(!source_a.same_issuer(&source_b));\n    let operators_a = bind_windows(source_a.clone(), &loaded);\n    let operators_b = bind_windows(source_b.clone(), &loaded);\n",
+          "expected_occurrences": 1
+        }
+      ]
+    }
+  ],
+  "original_child_map": [
+    {
+      "classification": "ROOT_REVIEWED_SOURCE_COMPOSITION_OR_TEST_CORRECTION",
+      "file": "crates/zerogravity-model-qwen35/src/forward.rs",
+      "isolated": {
+        "bytes": 31222,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/forward.rs",
+        "sha256": "f755b2d92c089194aad6ccb38a308914ebe882adb90690071d8e671e672e1839"
+      },
+      "line_deltas": [
+        {
+          "after": "}\n\n/// Pure coefficient preparation only; no exp, g/decay, scaling or state effects.\npub(crate) fn gdn_coefficient_preparation_slice<B: MathematicalBackend>(\n    source: &(impl crate::operator_slice::OperatorSliceSource + ?Sized),\n    math: &mut B,\n    layer: usize,\n    alpha: &B::Value,\n    beta_input: &B::Value,\n) -> Result<crate::accelerator::program::GdnCoefficientPreparationValues<B::Value>> {\n    use crate::accelerator::program::{ActivationPrecision, GdnCoefficientPreparationContract};\n    let roles = crate::operator_slice::coefficient_preparation::coefficient_roles(source, layer)?;\n    for input in [alpha, beta_input] {\n        expect_elements(\n            math,\n            input,\n            roles.heads,\n            \"coefficient preparation actual head count\",\n        )?;\n        if math.precision(input) != ActivationPrecision::Bf16WidenedF32 {\n            return Err(Error::Unsupported(\n                \"coefficient preparation requires original BF16 a/b\".into(),\n            ));\n        }\n    }\n    let values = math.coefficient_preparation_profiled(\n        roles.dt,\n        alpha,\n        beta_input,\n        GdnCoefficientPreparationContract::TorchCpuBf16Avx2,\n    )?;\n    for (value, precision) in [\n        (&values.biased_alpha, ActivationPrecision::F32),\n        (&values.softplus, ActivationPrecision::F32),\n        (&values.beta, ActivationPrecision::Bf16WidenedF32),\n    ] {\n        expect_elements(\n            math,\n            value,\n            roles.heads,\n            \"coefficient preparation output head count\",\n        )?;\n        if math.precision(value) != precision {\n            return Err(Error::State(\n                \"coefficient preparation output precision\".into(),\n            ));\n        }\n    }\n    Ok(values)\n",
+          "before": "",
+          "isolated_lines": [
+            270,
+            319
+          ],
+          "original_lines": [
+            270,
+            270
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "\n/// Pure cached-token arithmetic over explicit inputs; no StateEffect or exp.\npub(crate) fn cached_gdn_slice<B: MathematicalBackend>(\n    source: &impl OperatorSliceSource,\n    math: &mut B,\n    layer: usize,\n    input: &crate::accelerator::program::CachedGdnOperands<B::Value>,\n) -> Result<crate::accelerator::program::CachedGdnValues<B::Value>> {\n    use crate::accelerator::program::{ActivationPrecision, CachedGdnContract};\n    let heads = crate::operator_slice::cached_gdn::cached_heads(source, layer)?;\n    let operands = [\n        &input.query,\n        &input.key,\n        &input.value,\n        &input.beta,\n        &input.decay,\n        &input.previous_state,\n    ];\n    let contract = CachedGdnContract::TorchCpuBf16Avx2Width128;\n    contract.validate(heads, &operands.map(|v| math.elements(v)))?;\n    for (index, value) in operands.iter().enumerate() {\n        let expected = if index < 4 {\n            ActivationPrecision::Bf16WidenedF32\n        } else {\n            ActivationPrecision::F32\n        };\n        if math.precision(value) != expected {\n            return Err(Error::State(\"cached GDN source operand precision\".into()));\n        }\n    }\n    let values = math.cached_gdn_profiled(source.source_identity(), heads, input, contract)?;\n    for (value, elements, precision) in [\n        (&values.core_f32, heads * 128, ActivationPrecision::F32),\n        (\n            &values.output_bf16,\n            heads * 128,\n            ActivationPrecision::Bf16WidenedF32,\n        ),\n        (\n            &values.next_state,\n            heads * 128 * 128,\n            ActivationPrecision::F32,\n        ),\n    ] {\n        if math.elements(value) != elements || math.precision(value) != precision {\n            return Err(Error::State(\n                \"cached GDN explicit result geometry/precision\".into(),\n            ));\n        }\n    }\n    Ok(values)\n}\n\n/// Original layer0 H48 source gamma BF16, never a full-GDN/session capability.\npub(crate) fn gated_rms_bf16_gamma_slice<B: MathematicalBackend>(\n    source: &impl crate::operator_slice::OperatorSliceSource,\n    math: &mut B,\n    layer: usize,\n    core: &B::Value,\n    gate: &B::Value,\n) -> Result<B::Value> {\n    let parameter = crate::operator_slice::gated_rms::gated_rms_role(source, layer)?;\n    if math.elements(core) != 6144\n        || math.elements(gate) != 6144\n        || math.precision(core) != crate::accelerator::program::ActivationPrecision::Bf16WidenedF32\n        || math.precision(gate) != crate::accelerator::program::ActivationPrecision::Bf16WidenedF32\n    {\n        return Err(Error::Unsupported(\"gated RMS H48 BF16 core/gate\".into()));\n    }\n    let result = math.gated_rms_gamma_bf16_profiled(\n        parameter,\n        core,\n        gate,\n        crate::accelerator::program::GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128,\n    )?;\n    if math.elements(&result) != 6144\n        || math.precision(&result)\n            != crate::accelerator::program::ActivationPrecision::Bf16WidenedF32\n    {\n        return Err(Error::State(\n            \"gated RMS returned wrong precision/width\".into(),\n        ));\n    }\n    Ok(result)\n}\n",
+          "before": "",
+          "isolated_lines": [
+            769,
+            854
+          ],
+          "original_lines": [
+            720,
+            720
+          ],
+          "tag": "insert"
+        }
+      ],
+      "original": {
+        "bytes": 26341,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/followup-v2/proposed/crates/zerogravity-model-qwen35/src/forward.rs",
+        "sha256": "3ab4de28aad4d0a60f5cda93c19869aeae3d80661950065f09a3368949167c37"
+      }
+    },
+    {
+      "classification": "ROOT_REVIEWED_SOURCE_COMPOSITION_OR_TEST_CORRECTION",
+      "file": "crates/zerogravity-model-qwen35/src/host_math.rs",
+      "isolated": {
+        "bytes": 45787,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/host_math.rs",
+        "sha256": "d928c0d6de2ea7982044527d16cfd00af97e39c11de48c055a265e0784bea415"
+      },
+      "line_deltas": [
+        {
+          "after": "\n#[path = \"host_math/coefficient_preparation.rs\"]\nmod coefficient_preparation;\n",
+          "before": "",
+          "isolated_lines": [
+            21,
+            24
+          ],
+          "original_lines": [
+            21,
+            21
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "    fn cached_gdn_profiled(\n        &mut self,\n        source: crate::accelerator::SourceIdentity,\n        heads: usize,\n        input: &crate::accelerator::program::CachedGdnOperands<HostValue>,\n        contract: crate::accelerator::program::CachedGdnContract,\n    ) -> Result<crate::accelerator::program::CachedGdnValues<HostValue>> {\n        #[cfg(all(feature = \"original-hf-binding\", target_os = \"linux\"))]\n        {\n            self.execute_cached_gdn(source, heads, input, contract)\n        }\n        #[cfg(not(all(feature = \"original-hf-binding\", target_os = \"linux\")))]\n        {\n            let _ = (source, heads, input, contract);\n            Err(Error::Unsupported(\n                \"cached GDN requires leased original windows\".into(),\n            ))\n        }\n    }\n",
+          "before": "",
+          "isolated_lines": [
+            620,
+            639
+          ],
+          "original_lines": [
+            617,
+            617
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "    }\n    fn gated_rms_gamma_bf16_profiled(\n        &mut self,\n        parameter: SourceParameter,\n        core: &HostValue,\n        gate: &HostValue,\n        contract: crate::accelerator::program::GatedRmsGammaBf16Contract,\n    ) -> Result<HostValue> {\n        #[cfg(all(feature = \"original-hf-binding\", target_os = \"linux\"))]\n        {\n            self.execute_gated_rms_bf16_gamma(parameter, core, gate, contract)\n        }\n        #[cfg(not(all(feature = \"original-hf-binding\", target_os = \"linux\")))]\n        {\n            let _ = (parameter, core, gate, contract);\n            Err(Error::Unsupported(\n                \"gated RMS requires leased original windows\".into(),\n            ))\n        }\n    }\n    fn coefficient_preparation_profiled(\n        &mut self,\n        dt: SourceParameter,\n        alpha: &HostValue,\n        beta: &HostValue,\n        contract: crate::accelerator::program::GdnCoefficientPreparationContract,\n    ) -> Result<crate::accelerator::program::GdnCoefficientPreparationValues<HostValue>> {\n        self.coefficient_preparation_closed(dt, alpha, beta, contract)\n",
+          "before": "",
+          "isolated_lines": [
+            657,
+            685
+          ],
+          "original_lines": [
+            635,
+            635
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "\n#[cfg(all(feature = \"original-hf-binding\", target_os = \"linux\"))]\npub(crate) mod cached_gdn;\n\n#[cfg(all(feature = \"original-hf-binding\", target_os = \"linux\"))]\nmod gated_rms;\n",
+          "before": "",
+          "isolated_lines": [
+            1212,
+            1218
+          ],
+          "original_lines": [
+            1162,
+            1162
+          ],
+          "tag": "insert"
+        }
+      ],
+      "original": {
+        "bytes": 43660,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/proposed/crates/zerogravity-model-qwen35/src/host_math.rs",
+        "sha256": "095d6c0ff721864c81e11e05e3c97bab372b27b9af776b19f711dc90ecda5d21"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "file": "crates/zerogravity-model-qwen35/src/operator_slice/pure.rs",
+      "isolated": {
+        "bytes": 6526,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice/pure.rs",
+        "sha256": "9a2f8d7130f77e3244fe6aedc2b04cb09cfd9474800267e7dd7af22295ad5e4d"
+      },
+      "line_deltas": [],
+      "original": {
+        "bytes": 6526,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/followup-v3/proposed/crates/zerogravity-model-qwen35/src/operator_slice/pure.rs",
+        "sha256": "9a2f8d7130f77e3244fe6aedc2b04cb09cfd9474800267e7dd7af22295ad5e4d"
+      }
+    },
+    {
+      "classification": "ROOT_REVIEWED_SOURCE_COMPOSITION_OR_TEST_CORRECTION",
+      "file": "crates/zerogravity-model-qwen35/src/operator_slice.rs",
+      "isolated": {
+        "bytes": 8985,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice.rs",
+        "sha256": "5b7af46a274a88e5e58dd7fa614de575ce662f19822ec8030df8e1a95ca7c273"
+      },
+      "line_deltas": [
+        {
+          "after": "/// Pure source coefficient preparation; no g/decay/exp or state capability.\n#[path = \"operator_slice/coefficient_preparation.rs\"]\npub mod coefficient_preparation;\n\n",
+          "before": "",
+          "isolated_lines": [
+            230,
+            234
+          ],
+          "original_lines": [
+            230,
+            230
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "\n#[path = \"gdn_cached_slice.rs\"]\npub mod cached_gdn;\n\n#[path = \"gdn_gated_rms_slice.rs\"]\npub mod gated_rms;\n",
+          "before": "",
+          "isolated_lines": [
+            244,
+            250
+          ],
+          "original_lines": [
+            240,
+            240
+          ],
+          "tag": "insert"
+        }
+      ],
+      "original": {
+        "bytes": 8712,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/proposed/crates/zerogravity-model-qwen35/src/operator_slice.rs",
+        "sha256": "9e6ce542c753e6521210594be46f7d6b21bbed1d41ae445770e7c16185b66f1e"
+      }
+    },
+    {
+      "classification": "ROOT_REVIEWED_SOURCE_COMPOSITION_OR_TEST_CORRECTION",
+      "file": "crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs",
+      "isolated": {
+        "bytes": 16480,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs",
+        "sha256": "9769cf2902bf6fda572a4a52010cfbedb5bbc06e9869f3f6306043290389df8f"
+      },
+      "line_deltas": [
+        {
+          "after": "fn separately_opened_same_content_source_cannot_rebind_a_retained_input() {\n",
+          "before": "fn same_content_foreign_window_issuer_cannot_rebind_a_retained_input() {\n",
+          "isolated_lines": [
+            330,
+            331
+          ],
+          "original_lines": [
+            330,
+            331
+          ],
+          "tag": "replace"
+        },
+        {
+          "after": "    assert_ne!(source_a.source_identity(), source_b.source_identity());\n",
+          "before": "    assert_eq!(source_a.source_identity(), source_b.source_identity());\n",
+          "isolated_lines": [
+            335,
+            336
+          ],
+          "original_lines": [
+            335,
+            336
+          ],
+          "tag": "replace"
+        }
+      ],
+      "original": {
+        "bytes": 16477,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/followup-v3/proposed/crates/zerogravity-model-qwen35/src/operator_slice_window_roles_tests.rs",
+        "sha256": "7108060b361abe6d52680e161b75e3ec16e2057a019ace4c6455400b5067617d"
+      }
+    },
+    {
+      "classification": "ROOT_REVIEWED_SOURCE_COMPOSITION_OR_TEST_CORRECTION",
+      "file": "crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs",
+      "isolated": {
+        "bytes": 13321,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs",
+        "sha256": "eebdb97649fc69b7f960e666a7cd66d258ede55288b7bb80e8c4f7a345cb26c0"
+      },
+      "line_deltas": [
+        {
+          "after": "\n#[path = \"cached_gdn_window_tests.rs\"]\nmod cached_gdn;\n\n#[path = \"coefficient_preparation_tests.rs\"]\nmod coefficient_preparation;\n\n#[path = \"gated_rms_window_tests.rs\"]\nmod gated_rms;\n",
+          "before": "",
+          "isolated_lines": [
+            387,
+            396
+          ],
+          "original_lines": [
+            387,
+            387
+          ],
+          "tag": "insert"
+        }
+      ],
+      "original": {
+        "bytes": 13136,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/proposed/crates/zerogravity-model-qwen35/src/operator_slice_window_tests.rs",
+        "sha256": "4f87fa6c4e6824be2400d27eee2395a62ef697b6ff686cb1019a9dede431ac0d"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "file": "crates/zerogravity-model-qwen35/src/original_binding/window/pure.rs",
+      "isolated": {
+        "bytes": 4613,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/original_binding/window/pure.rs",
+        "sha256": "8ac493e822e7c7e804526ea75231d8d0c361e5bed490a06c3255a0c9f4290077"
+      },
+      "line_deltas": [],
+      "original": {
+        "bytes": 4613,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/proposed/crates/zerogravity-model-qwen35/src/original_binding/window/pure.rs",
+        "sha256": "8ac493e822e7c7e804526ea75231d8d0c361e5bed490a06c3255a0c9f4290077"
+      }
+    },
+    {
+      "classification": "ROOT_REVIEWED_SOURCE_COMPOSITION_OR_TEST_CORRECTION",
+      "file": "crates/zerogravity-model-qwen35/src/original_binding/window.rs",
+      "isolated": {
+        "bytes": 13767,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/original_binding/window.rs",
+        "sha256": "4841fa591204f0f815331038d2255f19501cc2c5bbfeb5882541a9b59fd84360"
+      },
+      "line_deltas": [
+        {
+          "after": "\n#[cfg(feature = \"cpu\")]\nmod gated_rms;\n",
+          "before": "",
+          "isolated_lines": [
+            383,
+            386
+          ],
+          "original_lines": [
+            383,
+            383
+          ],
+          "tag": "insert"
+        }
+      ],
+      "original": {
+        "bytes": 13727,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/proposed/crates/zerogravity-model-qwen35/src/original_binding/window.rs",
+        "sha256": "616aa13302fcbfe11dddce415cd6ab514a43e6367ff409ef0bf7f34733036993"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "file": "crates/zerogravity-cpu/src/weights.rs",
+      "isolated": {
+        "bytes": 41744,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/weights.rs",
+        "sha256": "8282a25b231c5d126b5cba1100764919ca6fd2bcff503c08d5e6119e8dafc74a"
+      },
+      "line_deltas": [],
+      "original": {
+        "bytes": 41744,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/original-dependency-source/crates/zerogravity-cpu/src/weights.rs",
+        "sha256": "8282a25b231c5d126b5cba1100764919ca6fd2bcff503c08d5e6119e8dafc74a"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "file": "crates/zerogravity-cpu/src/original_hf/window_projection.rs",
+      "isolated": {
+        "bytes": 8664,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/window_projection.rs",
+        "sha256": "c5907cba6082c0d2ec004fd9e9d399e8db0207a33233a0c9eade0aa7247fbeee"
+      },
+      "line_deltas": [],
+      "original": {
+        "bytes": 8664,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/original-dependency-source/crates/zerogravity-cpu/src/original_hf/window_projection.rs",
+        "sha256": "c5907cba6082c0d2ec004fd9e9d399e8db0207a33233a0c9eade0aa7247fbeee"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "file": "crates/zerogravity-cpu/src/original_hf/norm/ordinary.rs",
+      "isolated": {
+        "bytes": 8092,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/norm/ordinary.rs",
+        "sha256": "513503b486cb5eb95055a330f3e520102de3daea718aef7d6299094471a7a4b2"
+      },
+      "line_deltas": [],
+      "original": {
+        "bytes": 8092,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/original-dependency-source/crates/zerogravity-cpu/src/original_hf/norm/ordinary.rs",
+        "sha256": "513503b486cb5eb95055a330f3e520102de3daea718aef7d6299094471a7a4b2"
+      }
+    },
+    {
+      "classification": "ROOT_REVIEWED_SOURCE_COMPOSITION_OR_TEST_CORRECTION",
+      "file": "crates/zerogravity-core/src/accelerator/program.rs",
+      "isolated": {
+        "bytes": 83050,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core/src/accelerator/program.rs",
+        "sha256": "4332a3712ffbb3e91502808c53073f9adec890d7fc647a2e3ad5e7d2f3ef4787"
+      },
+      "line_deltas": [
+        {
+          "after": "mod cached_gdn;\nmod gated_rms;\npub use cached_gdn::{CachedGdnContract, CachedGdnOperands, CachedGdnValues};\n",
+          "before": "",
+          "isolated_lines": [
+            14,
+            17
+          ],
+          "original_lines": [
+            14,
+            14
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "pub use gated_rms::GatedRmsGammaBf16Contract;\n",
+          "before": "",
+          "isolated_lines": [
+            18,
+            19
+          ],
+          "original_lines": [
+            15,
+            15
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "mod coefficient_preparation;\npub use coefficient_preparation::{\n    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues,\n};\n",
+          "before": "",
+          "isolated_lines": [
+            20,
+            24
+          ],
+          "original_lines": [
+            16,
+            16
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "    /// Cached single-token arithmetic only; decay is an explicit external input.\n    fn cached_gdn_profiled(\n        &mut self,\n        _source: SourceIdentity,\n        _heads: usize,\n        _input: &CachedGdnOperands<Self::Value>,\n        _contract: CachedGdnContract,\n    ) -> Result<CachedGdnValues<Self::Value>> {\n        Err(Error::Unsupported(\n            \"backend does not implement explicit cached GDN contract\".into(),\n        ))\n    }\n",
+          "before": "",
+          "isolated_lines": [
+            292,
+            304
+          ],
+          "original_lines": [
+            284,
+            284
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "        ))\n    }\n    fn gated_rms_gamma_bf16_profiled(\n        &mut self,\n        _parameter: SourceParameter,\n        _core: &Self::Value,\n        _gate: &Self::Value,\n        _contract: GatedRmsGammaBf16Contract,\n    ) -> Result<Self::Value> {\n        Err(Error::Unsupported(\n            \"backend does not implement explicit BF16 gamma gated RMS contract\".into(),\n        ))\n    }\n    fn coefficient_preparation_profiled(\n        &mut self,\n        _dt: SourceParameter,\n        _alpha: &Self::Value,\n        _beta: &Self::Value,\n        _contract: GdnCoefficientPreparationContract,\n    ) -> Result<GdnCoefficientPreparationValues<Self::Value>> {\n        Err(Error::Unsupported(\n            \"backend does not implement original GDN coefficient preparation\".into(),\n",
+          "before": "",
+          "isolated_lines": [
+            314,
+            336
+          ],
+          "original_lines": [
+            294,
+            294
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "    TypedCachedGdnCore {\n        source: SourceIdentity,\n        heads: usize,\n        contract: CachedGdnContract,\n    },\n    TypedCachedGdnBf16Return {\n        source: SourceIdentity,\n        heads: usize,\n        contract: CachedGdnContract,\n    },\n",
+          "before": "",
+          "isolated_lines": [
+            511,
+            521
+          ],
+          "original_lines": [
+            469,
+            469
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "    },\n    TypedGatedRmsGammaBf16 {\n        parameter: SourceParameter,\n        contract: GatedRmsGammaBf16Contract,\n    },\n    TypedGdnAlphaPreparation {\n        dt: SourceParameter,\n        contract: GdnCoefficientPreparationContract,\n    },\n    TypedGdnBetaPreparation {\n        heads: usize,\n        contract: GdnCoefficientPreparationContract,\n",
+          "before": "",
+          "isolated_lines": [
+            528,
+            540
+          ],
+          "original_lines": [
+            476,
+            476
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "            PlanOperation::TypedCachedGdnCore {\n                source,\n                heads,\n                contract,\n            } => cached_gdn::result_precision(self, *source, *heads, *contract, inputs, false),\n            PlanOperation::TypedCachedGdnBf16Return {\n                source,\n                heads,\n                contract,\n            } => cached_gdn::result_precision(self, *source, *heads, *contract, inputs, true),\n",
+          "before": "",
+          "isolated_lines": [
+            884,
+            894
+          ],
+          "original_lines": [
+            820,
+            820
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "            PlanOperation::TypedGatedRmsGammaBf16 {\n                parameter,\n                contract,\n            } => gated_rms::result_precision(self, parameter, *contract, inputs),\n            PlanOperation::TypedGdnAlphaPreparation { dt, contract } => {\n                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1\n                    || dt.source != self.source\n                    || inputs.len() != 1\n                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32\n                {\n                    return Err(Error::Unsupported(\n                        \"GDN preparation alpha profile/source/precision\".into(),\n                    ));\n                }\n                contract.validate(dt, inputs[0].elements, inputs[0].elements)?;\n                Ok(ActivationPrecision::F32)\n            }\n            PlanOperation::TypedGdnBetaPreparation { heads, .. } => {\n                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1\n                    || !matches!(*heads, 16 | 48)\n                    || inputs.len() != 1\n                    || inputs[0].elements != *heads\n                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32\n                {\n                    return Err(Error::Unsupported(\n                        \"GDN preparation beta profile/span/precision\".into(),\n                    ));\n                }\n                Ok(ActivationPrecision::Bf16WidenedF32)\n            }\n",
+          "before": "",
+          "isolated_lines": [
+            898,
+            928
+          ],
+          "original_lines": [
+            824,
+            824
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "                        | PlanOperation::TypedGatedRmsGammaBf16 { .. }\n                        | PlanOperation::TypedGdnAlphaPreparation { .. }\n                        | PlanOperation::TypedGdnBetaPreparation { .. }\n                        | PlanOperation::TypedCachedGdnCore { .. }\n                        | PlanOperation::TypedCachedGdnBf16Return { .. }\n",
+          "before": "",
+          "isolated_lines": [
+            945,
+            950
+          ],
+          "original_lines": [
+            841,
+            841
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "    fn validate_record_inputs(\n        &self,\n        operation: &PlanOperation,\n",
+          "before": "    fn record(\n        &mut self,\n        name: &str,\n        operation: PlanOperation,\n",
+          "isolated_lines": [
+            959,
+            962
+          ],
+          "original_lines": [
+            850,
+            854
+          ],
+          "tag": "replace"
+        },
+        {
+          "after": "    ) -> Result<()> {\n",
+          "before": "        shapes: &[ValueShape],\n        state: Option<StateEffect>,\n    ) -> Result<Vec<PlanValue>> {\n",
+          "isolated_lines": [
+            963,
+            964
+          ],
+          "original_lines": [
+            855,
+            858
+          ],
+          "tag": "replace"
+        },
+        {
+          "after": "            if !input.contiguous && !matches!(operation, PlanOperation::Contiguous) {\n",
+          "before": "            if !input.contiguous && !matches!(&operation, PlanOperation::Contiguous) {\n",
+          "isolated_lines": [
+            974,
+            975
+          ],
+          "original_lines": [
+            868,
+            869
+          ],
+          "tag": "replace"
+        },
+        {
+          "after": "        Ok(())\n    }\n    fn record(\n        &mut self,\n        name: &str,\n        operation: PlanOperation,\n        inputs: &[PlanValue],\n        shapes: &[ValueShape],\n        state: Option<StateEffect>,\n    ) -> Result<Vec<PlanValue>> {\n        self.validate_record_inputs(&operation, inputs)?;\n",
+          "before": "",
+          "isolated_lines": [
+            980,
+            991
+          ],
+          "original_lines": [
+            874,
+            874
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "    fn cached_gdn_profiled(\n        &mut self,\n        source: SourceIdentity,\n        heads: usize,\n        input: &CachedGdnOperands<PlanValue>,\n        contract: CachedGdnContract,\n    ) -> Result<CachedGdnValues<PlanValue>> {\n        cached_gdn::record(self, source, heads, input, contract)\n    }\n",
+          "before": "",
+          "isolated_lines": [
+            1225,
+            1234
+          ],
+          "original_lines": [
+            1108,
+            1108
+          ],
+          "tag": "insert"
+        },
+        {
+          "after": "    }\n    fn gated_rms_gamma_bf16_profiled(\n        &mut self,\n        parameter: SourceParameter,\n        core: &PlanValue,\n        gate: &PlanValue,\n        contract: GatedRmsGammaBf16Contract,\n    ) -> Result<PlanValue> {\n        gated_rms::record(self, parameter, core, gate, contract)\n    }\n    fn coefficient_preparation_profiled(\n        &mut self,\n        dt: SourceParameter,\n        alpha: &PlanValue,\n        beta: &PlanValue,\n        contract: GdnCoefficientPreparationContract,\n    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {\n        let heads = contract.validate(&dt, alpha.elements, beta.elements)?;\n        if self\n            .values\n            .len()\n            .checked_add(3)\n            .is_none_or(|count| count > u32::MAX as usize)\n        {\n            return Err(Error::Limit(\"GDN preparation value count overflow\".into()));\n        }\n        let alpha_op = PlanOperation::TypedGdnAlphaPreparation {\n            dt: dt.clone(),\n            contract,\n        };\n        let beta_op = PlanOperation::TypedGdnBetaPreparation { heads, contract };\n        // Validate BOTH original values and every cold guard before recording.\n        self.validate_record_inputs(&alpha_op, &[*alpha])?;\n        self.validate_record_inputs(&beta_op, &[*beta])?;\n        self.result_precision(&alpha_op, &[*alpha])?;\n        self.result_precision(&beta_op, &[*beta])?;\n        let beta = self.one(\n            &format!(\"{}.beta-preparation\", dt.name),\n            beta_op,\n            &[*beta],\n            ValueShape::Vector(heads),\n            None,\n        )?;\n        let values = self.record(\n            &format!(\"{}.alpha-preparation\", dt.name),\n            alpha_op,\n            &[*alpha],\n            &[ValueShape::Vector(heads); 2],\n            None,\n        )?;\n        Ok(GdnCoefficientPreparationValues {\n            biased_alpha: values[0],\n            softplus: values[1],\n            beta,\n        })\n",
+          "before": "",
+          "isolated_lines": [
+            1242,
+            1297
+          ],
+          "original_lines": [
+            1116,
+            1116
+          ],
+          "tag": "insert"
+        }
+      ],
+      "original": {
+        "bytes": 76201,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/original-dependency-source/crates/zerogravity-core/src/accelerator/program.rs",
+        "sha256": "2ee7db7698fe0c65330cdf2e074c8335fd2cef36033c291b9aa553764695452f"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "file": "crates/zerogravity-artifacts/src/source_weights/tensor_window.rs",
+      "isolated": {
+        "bytes": 29533,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs",
+        "sha256": "c700495b9e252f972b7dfeeb41368dc64413b869cc73cc9a8eb0868bcfe60b45"
+      },
+      "line_deltas": [],
+      "original": {
+        "bytes": 29533,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/original-dependency-source/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs",
+        "sha256": "c700495b9e252f972b7dfeeb41368dc64413b869cc73cc9a8eb0868bcfe60b45"
+      }
+    },
+    {
+      "classification": "BYTE_IDENTICAL_ROOT_REMAP",
+      "file": "crates/zerogravity-model-qwen35/src/original_binding.rs",
+      "isolated": {
+        "bytes": 16645,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35/src/original_binding.rs",
+        "sha256": "d80b100feb85b4247a7237e647972df3795653f7feed3c2c8263828a36cefb1d"
+      },
+      "line_deltas": [],
+      "original": {
+        "bytes": 16645,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/original-dependency-source/crates/zerogravity-model-qwen35/src/original_binding.rs",
+        "sha256": "d80b100feb85b4247a7237e647972df3795653f7feed3c2c8263828a36cefb1d"
+      }
+    }
+  ],
+  "original_harness": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification",
+  "original_metadata_followup": {
+    "bytes": 2160,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/metadata-v2/prepared-manifest.json",
+    "sha256": "e9581cf982e3c83fa85413a4eb707afef001290c6137563a34a9c679a54514e6"
+  },
+  "original_preparation": {
+    "bytes": 3644,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/prepared-manifest.json",
+    "sha256": "16809776089e1c3063629a0e0df556512a66b9ccd04d6812d56e47b744f4cb40"
+  },
+  "relocated_harness": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020",
+  "required_isolated_status": "ROOT_COMPILED_ISOLATED_ORIGINAL27B_PURE_ROLES_ONLY",
+  "schema": "zerogravity.pure-wide.isolated-source-relocation.v1",
+  "source_count": 521,
+  "source_freeze": {
+    "bytes": 99248,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-7.json",
+    "sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615"
+  },
+  "support_frozen_dispatch": "def frozen():\n    from relocated_source import frozen as checked_frozen\n    return checked_frozen(HERE, metadata, read, require, sha)\n",
+  "workspace_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020"
+}
--- /dev/null
+++ b/run_native.py
@@ -0,0 +1,115 @@
+"""ROOT ONLY owned same-engine pure-role child; no model/runtime on import."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import resource
+import signal
+import subprocess
+import time
+from support import HERE,frozen,metadata,read,require,sha,write
+from relocated_source import implementation, reference
+from sigterm_cancellation import SigtermCancellation
+
+
+def main(cancel):
+    parser=argparse.ArgumentParser()
+    parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--binary-sha256',required=True)
+    parser.add_argument('--reference-directory',type=Path,required=True)
+    parser.add_argument('--build-attestation',type=Path,required=True);parser.add_argument('--new-output',type=Path,required=True)
+    args=parser.parse_args()
+    cancel.check()
+    freeze=frozen()
+    require(args.binary.is_absolute()and not args.binary.is_symlink()and sha(args.binary)==args.binary_sha256,
+            'actual ROOT-built immutable same-engine executable')
+    source_attestation=implementation(HERE,args.build_attestation,metadata,read,require,sha)
+    build=metadata(args.build_attestation)
+    require(build['binary_sha256']==args.binary_sha256 and
+            Path(build['binary']).resolve()==args.binary.resolve(),
+            'ROOT actual isolated executable binding; source markers alone are not compiler proof')
+    require(args.new_output.is_absolute()and args.new_output.parent.resolve()==HERE,
+            'owned relocated native output only')
+    validation=reference(HERE,args.reference_directory,metadata,read,require)
+    worktree=Path(build['root_worktree'])
+    cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus)
+    args.new_output.mkdir(mode=0o700)
+    require(sha(HERE/'owned_container_v4.py')=='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
+            'exact reviewed original WNOWAIT cleanup')
+    spec=importlib.util.spec_from_file_location('pure_native_owned',HERE/'owned_container_v4.py')
+    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
+    fd=os.open(args.binary,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+    child=log=None;failure=None;started=time.monotonic()
+    receipt={'status':'UNQUALIFIED','freeze_sha256':freeze,'binary_sha256':args.binary_sha256,
+             'build_attestation_sha256':sha(args.build_attestation),'expected_cpus':cpus,
+             'scope':'same-engine private original27B pure roles only; no full-model/state/GPU',
+             'MAIN_merged':False,'root_source_freeze_sha256':build['root_source_freeze_sha256'],
+             'original_oracle_preparation_sha256':validation['freeze_sha256']}
+    try:
+        with os.fdopen(os.dup(fd),'rb')as opened:
+            import hashlib
+            require(hashlib.file_digest(opened,'sha256').hexdigest()==args.binary_sha256,'exact retained exec FD')
+        manifest_sha=sha(args.reference_directory/'captured/manifest.json');validation_sha=sha(args.reference_directory/'validation.json')
+        argv=[str(args.binary),str(args.reference_directory),manifest_sha,validation_sha,str(args.new_output/'report.json')]
+        receipt['argv']=argv
+        environment={k:v for k,v in os.environ.items()if k in ('PATH','HOME','LANG','LC_ALL','TZ','XDG_CACHE_HOME')}
+        environment.update(RAYON_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void')
+        def limits():
+            os.sched_setaffinity(0,cpus)
+            resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
+            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
+            resource.setrlimit(resource.RLIMIT_CPU,(1801,1801))
+            resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,4*1024**2))
+        cancel.check() # Record-only signals cannot interrupt original Popen ownership assignment.
+        try:
+            child=subprocess.Popen(argv,executable=f'/proc/self/fd/{fd}',pass_fds=(fd,),
+                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
+                env=environment,start_new_session=True,preexec_fn=limits)
+            receipt['owned_cli_pid']=child.pid
+            log=helper.Log(child.stdout,args.new_output/'execution.log')
+            receipt['exit_code']=cancel.observe(log,child,1800)
+        except BaseException as error:failure=error
+        finally:
+            receipt['cleanup_errors']=[]
+            if child is not None:
+                try:helper.cleanup_group(child,receipt)
+                except BaseException as error:receipt['cleanup_errors'].append('group:'+repr(error))
+            if log is not None:
+                try:log.finish()
+                except BaseException as error:receipt['cleanup_errors'].append('log:'+repr(error))
+                finally:receipt['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
+            elif child is not None and child.stdout is not None:
+                try:child.stdout.close()
+                except BaseException as error:receipt['cleanup_errors'].append('uninitialized pipe:'+repr(error))
+        cancel.check() # Original cleanup completed before synchronous cancellation is raised.
+        require(failure is None and receipt.get('exit_code')==0 and receipt['cleanup_errors']==[],
+                'owned native replay/cleanup failed; no automatic retry')
+        report=metadata(args.new_output/'report.json')
+        require(report['proof']['status']=='PASS_PRIVATE_ORIGINAL27B_PURE_ROLES_ONLY'and
+                report['final_reserved']==0 and report['final_fenced']is False and
+                report['proof']['result']['compared_words']==validation['words'],
+                'all actual native raw outputs/input words exact and final original ledger0')
+        require(implementation(HERE,args.build_attestation,metadata,read,require,sha)==source_attestation and
+                reference(HERE,args.reference_directory,metadata,read,require)==validation,
+                'whole isolated ROOT source/build and accepted oracle remain exact')
+        require(frozen()==freeze and sha(args.binary)==args.binary_sha256 and
+                sha(args.reference_directory/'captured/manifest.json')==manifest_sha and
+                sha(args.reference_directory/'validation.json')==validation_sha,
+                'all preparation/binary/independent raw reference unchanged')
+        cancel.check()
+        receipt.update(status='PASS_OWNED_ORIGINAL27B_PURE_ROLE_REPLAY_ONLY',report_sha256=sha(args.new_output/'report.json'))
+    except BaseException as error:
+        if failure is None:failure=error
+    finally:
+        try:os.close(fd)
+        except BaseException as error:
+            if failure is None:failure=error
+        receipt['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        if failure is not None:receipt.update(status='FAIL_UNQUALIFIED',failure=repr(failure))
+        write(args.new_output/'invocation.json',receipt)
+    require(receipt['status']=='PASS_OWNED_ORIGINAL27B_PURE_ROLE_REPLAY_ONLY','native proof unqualified; retained original attempt')
+
+
+if __name__=='__main__':
+    with SigtermCancellation() as cancellation:
+        main(cancellation)
--- /dev/null
+++ b/run_reference.py
@@ -0,0 +1,134 @@
+"""ROOT ONLY owned CPU reference, whole source before/after, no model on import."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import signal
+import stat
+import subprocess
+import time
+import uuid
+from support import HERE, IMAGE, IMAGE_DIGEST, bounds, fd_sha, fingerprint, frozen, metadata, read, require, sha, write
+
+def source_open(plan):
+    """ROOT-only complete source hashes, bounded chunks; same FDs kept open."""
+    handles, rows = {}, []
+    try:
+        for row in plan['source_files']:
+            require(Path(row['name']).name==row['name'],'contained full source filename')
+            fd=os.open(Path(plan['source_directory'])/row['name'],os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            handles[row['name']]=fd
+            before=os.fstat(fd)
+            require(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'] and
+                    fd_sha(fd,0,row['bytes'])==row['sha256'] and fingerprint(before)==fingerprint(os.fstat(fd)),
+                    'actual full original source bytes/FD immutable before reference')
+            rows.append({'name':row['name'],'bytes':row['bytes'],'sha256':row['sha256'],
+                         'fd_identity':fingerprint(before)})
+        require(sum(row['bytes'] for row in plan['source_files'] if row['name'].endswith('.safetensors'))
+                ==plan['weight_bytes']==55563006776,'all18 authentic original weight shards')
+        return handles,rows
+    except BaseException:
+        for fd in handles.values():os.close(fd)
+        raise
+
+def source_after(handles,rows):
+    after=[]
+    for row in rows:
+        fd=handles[row['name']]
+        require(fingerprint(os.fstat(fd))==row['fd_identity'] and
+                fd_sha(fd,0,row['bytes'])==row['sha256'],'same full original source FD after reference')
+        after.append(dict(row))
+    return after
+
+def tree_size(root):
+    total=0
+    for path in root.rglob('*'):
+        require(not path.is_symlink(),'no output artifact symlink')
+        if path.is_file():
+            require(stat.S_ISREG(path.stat().st_mode),'regular actual artifact')
+            total+=path.stat().st_size
+    require(total<=32*1024**2-128*1024,'bounded whole attempt with reserved completion metadata')
+    return total
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--new-output',type=Path,required=True)
+    args=parser.parse_args()
+    def deadline(_signal,_frame):raise TimeoutError('finite whole source/reference/lifecycle deadline')
+    signal.signal(signal.SIGALRM,deadline);signal.alarm(2400)
+    def cancelled(_signal,_frame):raise InterruptedError('owned reference cancelled; retire only owned container/process')
+    signal.signal(signal.SIGTERM,cancelled)
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+    require(args.new_output.is_absolute() and args.new_output.parent.resolve()==HERE,
+            'exclusive owned attempt directory');args.new_output.mkdir(mode=0o700)
+    cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus)
+    require(sha(HERE/'owned_container_v4.py')=='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
+            'byte-identical reviewed owned container lifecycle')
+    spec=importlib.util.spec_from_file_location('roles27_owned',HERE/'owned_container_v4.py')
+    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
+    nonce=uuid.uuid4().hex;owned_name='zg-original27-roles-'+nonce;cid=args.new_output/'owned.cid'
+    receipt={'status':'UNQUALIFIED','freeze_manifest_sha256':freeze,'source_plan_sha256':sha(HERE/'source-plan.json'),
+             'expected_cpus':cpus,'owned_name':owned_name,'owned_nonce':nonce,'image_id':IMAGE,'image_digest':IMAGE_DIGEST,
+             'no_GPU_or_full_model_or_native_provider_admission':True}
+    handles={};child=log=None;failure=None;started=time.monotonic()
+    try:
+        handles,before=source_open(plan);receipt['source_before']=before
+        inspected=helper.command(helper.PODMAN+['image','inspect',IMAGE],15)
+        require(inspected.returncode==0,'existing approved image; no pull')
+        (args.new_output/'image-inspect.json').write_bytes(inspected.stdout)
+        image,=json.loads(inspected.stdout)
+        require(image['Id']==IMAGE.removeprefix('sha256:') and image['Digest']==IMAGE_DIGEST,'pinned actual image')
+        prefix=Path(plan['prior_public_prefix']['path']).parent
+        env={'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'void','ATEN_CPU_CAPABILITY':'avx2',
+             'ZG_ALLOWED_CPUS':','.join(map(str,cpus)),'ZG_SOURCE_PLAN_SHA':sha(HERE/'source-plan.json'),
+             'USE_HUB_KERNELS':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
+             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE',
+             'PYTHONPATH':'/probe','PYTHONDONTWRITEBYTECODE':'1','LD_PRELOAD':'','LD_AUDIT':'','VLLM_PLUGINS':''}
+        argv=helper.PODMAN+['run','--name',owned_name,'--label',helper.LABEL+'='+nonce,'--cidfile',str(cid),
+            '--rm','--pull=never','--network=none','--read-only','--cap-drop=all','--security-opt=no-new-privileges',
+            '--memory=12g','--memory-swap=12g','--cpus=4','--pids-limit=256',
+            '--ulimit','fsize=134217728:134217728','--tmpfs','/tmp:rw,nosuid,nodev,size=1g']
+        for key,value in env.items():argv+=['-e',key+'='+value]
+        argv+=['-v',str(HERE)+':/probe:ro','-v',plan['source_directory']+':/source:ro',
+               '-v',str(prefix)+':/prefix-reference:ro','-v',str(args.new_output)+':/output:rw',
+               '--entrypoint','python3',IMAGE,'/probe/capture.py']
+        receipt['argv']=argv;write(args.new_output/'prepared-invocation.json',receipt)
+        try:
+            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
+            receipt['owned_cli_pid']=child.pid
+            log=helper.Log(child.stdout,args.new_output/'execution.log')
+            receipt['exit_code']=log.observe(child,600)
+        except BaseException as error:failure=error
+        finally:
+            receipt['cleanup_errors']=helper.cleanup(child,owned_name,nonce,cid,receipt) if child is not None else []
+            if log is not None:
+                try:log.finish()
+                except BaseException as error:receipt['cleanup_errors'].append('log cleanup: '+repr(error))
+                finally:receipt['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
+            elif child is not None and child.stdout is not None:
+                try:child.stdout.close()
+                except BaseException as error:receipt['cleanup_errors'].append('uninitialized pipe close: '+repr(error))
+        require(failure is None and receipt.get('exit_code')==0 and receipt['cleanup_errors']==[],
+                'owned original public reference/cleanup failed; preserve this attempt')
+        receipt['source_after']=source_after(handles,before)
+        require(receipt['source_after']==before and frozen()==freeze,'all source/preparation files unchanged')
+        receipt['precompletion_artifact_bytes']=tree_size(args.new_output)
+        receipt['status']='EXECUTED_ORIGINAL27B_PURE_ROLES_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        if failure is None:failure=error
+    finally:
+        failures=[]
+        for fd in handles.values():
+            try:os.close(fd)
+            except BaseException as error:failures.append(repr(error))
+        receipt['source_FD_cleanup_errors']=failures
+        receipt['source_FDs_closed']=not failures
+        receipt['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        if failure is not None:receipt['failure']=repr(failure)
+        if failure is not None or failures:receipt['status']='FAIL_UNQUALIFIED'
+        write(args.new_output/'invocation.json',receipt)
+    require(receipt['status']=='EXECUTED_ORIGINAL27B_PURE_ROLES_REQUIRES_INDEPENDENT_VALIDATION',
+            'reference attempt unqualified; retained original evidence')
+
+if __name__=='__main__':main()
--- /dev/null
+++ b/sigterm_cancellation.py
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
+++ b/source-pin.json
@@ -0,0 +1,209 @@
+{
+  "classification": "authenticated_cache_source_only_not_model_execution",
+  "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
+  "cache": "/home/pvl/.cache/abird/zerogravity/models",
+  "reference_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
+  "repository": "Qwen/Qwen3.8-27B",
+  "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
+  "selected_bytes": 55586113293,
+  "weight_bytes": 55563006776,
+  "payload_bytes": 55562855904,
+  "files": [
+    {
+      "name": "LICENSE",
+      "bytes": 11544,
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
+      "git_blob_verified": true
+    },
+    {
+      "name": "README.md",
+      "bytes": 65012,
+      "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641",
+      "git_blob_verified": true
+    },
+    {
+      "name": "chat_template.jinja",
+      "bytes": 8952,
+      "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
+      "git_blob_verified": true
+    },
+    {
+      "name": "config.json",
+      "bytes": 4312,
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
+      "git_blob_verified": true
+    },
+    {
+      "name": "crc32.txt",
+      "bytes": 238,
+      "sha256": "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83",
+      "git_blob_verified": true
+    },
+    {
+      "name": "generation_config.json",
+      "bytes": 202,
+      "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
+      "git_blob_verified": true
+    },
+    {
+      "name": "merges.txt",
+      "bytes": 3353259,
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
+      "git_blob_verified": true
+    },
+    {
+      "name": "model-00001-of-00018.safetensors",
+      "bytes": 3966730552,
+      "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00002-of-00018.safetensors",
+      "bytes": 3043080328,
+      "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00003-of-00018.safetensors",
+      "bytes": 2542796952,
+      "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00004-of-00018.safetensors",
+      "bytes": 3988973152,
+      "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00005-of-00018.safetensors",
+      "bytes": 2099339864,
+      "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00006-of-00018.safetensors",
+      "bytes": 3979553696,
+      "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00007-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00008-of-00018.safetensors",
+      "bytes": 3979553696,
+      "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00009-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00010-of-00018.safetensors",
+      "bytes": 3979553696,
+      "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00011-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00012-of-00018.safetensors",
+      "bytes": 3979553696,
+      "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00013-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00014-of-00018.safetensors",
+      "bytes": 3979553696,
+      "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00015-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00016-of-00018.safetensors",
+      "bytes": 3979564040,
+      "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00017-of-00018.safetensors",
+      "bytes": 2108759344,
+      "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model-00018-of-00018.safetensors",
+      "bytes": 3392197344,
+      "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
+      "git_blob_verified": false
+    },
+    {
+      "name": "model.safetensors.index.json",
+      "bytes": 112216,
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
+      "git_blob_verified": true
+    },
+    {
+      "name": "preprocessor_config.json",
+      "bytes": 390,
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
+      "git_blob_verified": true
+    },
+    {
+      "name": "tokenizer.json",
+      "bytes": 12809320,
+      "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
+      "git_blob_verified": false
+    },
+    {
+      "name": "tokenizer_config.json",
+      "bytes": 17928,
+      "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
+      "git_blob_verified": true
+    },
+    {
+      "name": "video_preprocessor_config.json",
+      "bytes": 385,
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
+      "git_blob_verified": true
+    },
+    {
+      "name": "vocab.json",
+      "bytes": 6722759,
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
+      "git_blob_verified": true
+    }
+  ],
+  "tensor_count": 1199,
+  "ordinary_text_tensors": 851,
+  "other_tensors": 348,
+  "ordinary_text_bytes": 53791996928,
+  "full_file_sha256s_independent": true,
+  "ordinary_source_shapes_match_config": true,
+  "headers_match_preflight": true,
+  "buffer_bytes": 1048576,
+  "statvfs_free_after": 544125792256,
+  "finished_utc": "2026-10-19T15:50:57Z"
+}
--- /dev/null
+++ b/source-plan.json
@@ -0,0 +1,622 @@
+{
+  "bounds": {
+    "CPU_threads": 4,
+    "artifact_bytes": 33554432,
+    "container_RAM_bytes": 12884901888,
+    "mapped_bytes": 3519025784,
+    "operator_working_allowance_bytes": 2147483648,
+    "page_size": 4096,
+    "payload_bytes": 4194304,
+    "process_AS_bytes": 12884901888,
+    "runtime_VA_allowance_bytes": 6442450944,
+    "whole_model_native_admission_qualified": false
+  },
+  "metadata": {
+    "config.json": {
+      "bytes": 4312,
+      "git_blob_verified": true,
+      "name": "config.json",
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+    },
+    "model.safetensors.index.json": {
+      "bytes": 112216,
+      "git_blob_verified": true,
+      "name": "model.safetensors.index.json",
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+    }
+  },
+  "prior_full_source_verification": {
+    "bytes": 7058,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-owned-window-20261020/source-pin.json",
+    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+  },
+  "prior_public_prefix": {
+    "bytes": 65417,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-typed-oracle-20261020/root-reference-1/captured/reference.json",
+    "sha256": "794765650ff40a52b821cebf85889cb46593b337ea79b94b50f4bad4d244e0aa"
+  },
+  "repository": "Qwen/Qwen3.8-27B",
+  "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
+  "roles": [
+    {
+      "absolute_offset": 1120792,
+      "bytes": 104857600,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 104860184,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
+      "output_elements": 10240,
+      "page_aligned_offset": 1118208,
+      "role": "GdnQkv",
+      "shape": [
+        10240,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    {
+      "absolute_offset": 105978392,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 62917144,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_z.weight",
+      "output_elements": 6144,
+      "page_aligned_offset": 105975808,
+      "role": "GdnGate",
+      "shape": [
+        6144,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    {
+      "absolute_offset": 137752,
+      "bytes": 491520,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 494104,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_a.weight",
+      "output_elements": 48,
+      "page_aligned_offset": 135168,
+      "role": "GdnAlpha",
+      "shape": [
+        48,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    {
+      "absolute_offset": 629272,
+      "bytes": 491520,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 494104,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_b.weight",
+      "output_elements": 48,
+      "page_aligned_offset": 626688,
+      "role": "GdnBeta",
+      "shape": [
+        48,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    {
+      "absolute_offset": 168893208,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "input_elements": 6144,
+      "input_provenance": "external_literal_BF16_pattern_v1_no_trained_producer_claim",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 62917400,
+      "name": "model.language_model.layers.0.linear_attn.out_proj.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 168890368,
+      "role": "GdnOutput",
+      "shape": [
+        5120,
+        6144
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    },
+    {
+      "absolute_offset": 2907880088,
+      "bytes": 125829120,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 3,
+      "mapping_bytes": 125831832,
+      "name": "model.language_model.layers.3.self_attn.q_proj.weight",
+      "output_elements": 12288,
+      "page_aligned_offset": 2907877376,
+      "role": "Query",
+      "shape": [
+        12288,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2712
+    },
+    {
+      "absolute_offset": 2834479256,
+      "bytes": 10485760,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 3,
+      "mapping_bytes": 10487960,
+      "name": "model.language_model.layers.3.self_attn.k_proj.weight",
+      "output_elements": 1024,
+      "page_aligned_offset": 2834477056,
+      "role": "Key",
+      "shape": [
+        1024,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2200
+    },
+    {
+      "absolute_offset": 3033709208,
+      "bytes": 10485760,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 3,
+      "mapping_bytes": 10488472,
+      "name": "model.language_model.layers.3.self_attn.v_proj.weight",
+      "output_elements": 1024,
+      "page_aligned_offset": 3033706496,
+      "role": "Value",
+      "shape": [
+        1024,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2712
+    },
+    {
+      "absolute_offset": 2844965016,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "input_elements": 6144,
+      "input_provenance": "external_literal_BF16_pattern_v1_no_trained_producer_claim",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 3,
+      "mapping_bytes": 62916760,
+      "name": "model.language_model.layers.3.self_attn.o_proj.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 2844962816,
+      "role": "AttentionOutput",
+      "shape": [
+        5120,
+        6144
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2200
+    },
+    {
+      "absolute_offset": 410065688,
+      "bytes": 178257920,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 178260760,
+      "name": "model.language_model.layers.0.mlp.gate_proj.weight",
+      "output_elements": 17408,
+      "page_aligned_offset": 410062848,
+      "role": "FfnGate",
+      "shape": [
+        17408,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    },
+    {
+      "absolute_offset": 588323608,
+      "bytes": 178257920,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 178260760,
+      "name": "model.language_model.layers.0.mlp.up_proj.weight",
+      "output_elements": 17408,
+      "page_aligned_offset": 588320768,
+      "role": "FfnUp",
+      "shape": [
+        17408,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    },
+    {
+      "absolute_offset": 231807768,
+      "bytes": 178257920,
+      "dtype": "BF16",
+      "input_elements": 17408,
+      "input_provenance": "external_literal_BF16_pattern_v1_no_trained_producer_claim",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": 0,
+      "mapping_bytes": 178260760,
+      "name": "model.language_model.layers.0.mlp.down_proj.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 231804928,
+      "role": "FfnDown",
+      "shape": [
+        5120,
+        17408
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    },
+    {
+      "absolute_offset": 1760,
+      "bytes": 2542796800,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_normalized_teacher_external_probe",
+      "input_stage": "normalized",
+      "kind": "projection",
+      "layer": null,
+      "mapping_bytes": 2542798560,
+      "name": "lm_head.weight",
+      "output_elements": 248320,
+      "page_aligned_offset": 0,
+      "role": "Output",
+      "shape": [
+        248320,
+        5120
+      ],
+      "shard": "model-00018-of-00018.safetensors",
+      "view_offset": 1760
+    },
+    {
+      "absolute_offset": 45400,
+      "bytes": 10240,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_embedding_teacher_external_probe",
+      "input_stage": "embedding",
+      "kind": "ordinary_rms",
+      "layer": 0,
+      "mapping_bytes": 10584,
+      "name": "model.language_model.layers.0.input_layernorm.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 45056,
+      "role": "InputNorm",
+      "shape": [
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 344
+    },
+    {
+      "absolute_offset": 766581528,
+      "bytes": 10240,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_embedding_teacher_external_probe",
+      "input_stage": "embedding",
+      "kind": "ordinary_rms",
+      "layer": 0,
+      "mapping_bytes": 13080,
+      "name": "model.language_model.layers.0.post_attention_layernorm.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 766578688,
+      "role": "FfnNorm",
+      "shape": [
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    },
+    {
+      "absolute_offset": 3979553800,
+      "bytes": 10240,
+      "dtype": "BF16",
+      "input_elements": 5120,
+      "input_provenance": "accepted_original27B_embedding_teacher_external_probe",
+      "input_stage": "embedding",
+      "kind": "ordinary_rms",
+      "layer": null,
+      "mapping_bytes": 13320,
+      "name": "model.language_model.norm.weight",
+      "output_elements": 5120,
+      "page_aligned_offset": 3979550720,
+      "role": "FinalNorm",
+      "shape": [
+        5120
+      ],
+      "shard": "model-00016-of-00018.safetensors",
+      "view_offset": 3080
+    }
+  ],
+  "runtime_source": {
+    "cache_utils.py": {
+      "bytes": 104677,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/cache_utils.py",
+      "sha256": "702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c"
+    },
+    "configuration_qwen3_5.py": {
+      "bytes": 8395,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/configuration_qwen3_5.py",
+      "sha256": "19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da"
+    },
+    "hub_kernels.py": {
+      "bytes": 44974,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/hub_kernels.py",
+      "sha256": "c1eafefe0cbdf7f0deaa21b1ef6cb6d7c6082b3f486651d069ff38fcf3e53c95"
+    },
+    "modeling_qwen3_5.py": {
+      "bytes": 95685,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/modeling_qwen3_5.py",
+      "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+    }
+  },
+  "schema": "zerogravity.original27B.pure-window-roles.public-oracle-plan.v1",
+  "scope": "Thirteen source-authentic BF16 pure projections and three original ordinary norms, two teacher-bound external probes. Hidden input from accepted actual original27B InputNorm0; nonhidden input literal nonzero finite BF16 probe, never a trained activation producer. No full model/session/state/exp/GPU/throughput claim.",
+  "selected_shards": {
+    "model-00001-of-00018.safetensors": {
+      "file": {
+        "bytes": 3966730552,
+        "git_blob_verified": false,
+        "name": "model-00001-of-00018.safetensors",
+        "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+      },
+      "header_bytes": 45392,
+      "header_sha256": "7bdeb03df566aa804dca2cb48e420908cb1864e5222858ad04918299883287e0"
+    },
+    "model-00016-of-00018.safetensors": {
+      "file": {
+        "bytes": 3979564040,
+        "git_blob_verified": false,
+        "name": "model-00016-of-00018.safetensors",
+        "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"
+      },
+      "header_bytes": 10048,
+      "header_sha256": "19bf5bf58f5b46c36817fb179af3d226d7394466a7905c8464a9c119581f18e0"
+    },
+    "model-00018-of-00018.safetensors": {
+      "file": {
+        "bytes": 3392197344,
+        "git_blob_verified": false,
+        "name": "model-00018-of-00018.safetensors",
+        "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"
+      },
+      "header_bytes": 1752,
+      "header_sha256": "2e6b71178b57a98de637fa669dedfaf35cbed1388914ffc93238bddd169eab54"
+    }
+  },
+  "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
+  "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
+  "source_files": [
+    {
+      "bytes": 11544,
+      "git_blob_verified": true,
+      "name": "LICENSE",
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
+    },
+    {
+      "bytes": 65012,
+      "git_blob_verified": true,
+      "name": "README.md",
+      "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
+    },
+    {
+      "bytes": 8952,
+      "git_blob_verified": true,
+      "name": "chat_template.jinja",
+      "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
+    },
+    {
+      "bytes": 4312,
+      "git_blob_verified": true,
+      "name": "config.json",
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+    },
+    {
+      "bytes": 238,
+      "git_blob_verified": true,
+      "name": "crc32.txt",
+      "sha256": "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83"
+    },
+    {
+      "bytes": 202,
+      "git_blob_verified": true,
+      "name": "generation_config.json",
+      "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
+    },
+    {
+      "bytes": 3353259,
+      "git_blob_verified": true,
+      "name": "merges.txt",
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
+    },
+    {
+      "bytes": 3966730552,
+      "git_blob_verified": false,
+      "name": "model-00001-of-00018.safetensors",
+      "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+    },
+    {
+      "bytes": 3043080328,
+      "git_blob_verified": false,
+      "name": "model-00002-of-00018.safetensors",
+      "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb"
+    },
+    {
+      "bytes": 2542796952,
+      "git_blob_verified": false,
+      "name": "model-00003-of-00018.safetensors",
+      "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"
+    },
+    {
+      "bytes": 3988973152,
+      "git_blob_verified": false,
+      "name": "model-00004-of-00018.safetensors",
+      "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a"
+    },
+    {
+      "bytes": 2099339864,
+      "git_blob_verified": false,
+      "name": "model-00005-of-00018.safetensors",
+      "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00006-of-00018.safetensors",
+      "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00007-of-00018.safetensors",
+      "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00008-of-00018.safetensors",
+      "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00009-of-00018.safetensors",
+      "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00010-of-00018.safetensors",
+      "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00011-of-00018.safetensors",
+      "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00012-of-00018.safetensors",
+      "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00013-of-00018.safetensors",
+      "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00014-of-00018.safetensors",
+      "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00015-of-00018.safetensors",
+      "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22"
+    },
+    {
+      "bytes": 3979564040,
+      "git_blob_verified": false,
+      "name": "model-00016-of-00018.safetensors",
+      "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00017-of-00018.safetensors",
+      "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8"
+    },
+    {
+      "bytes": 3392197344,
+      "git_blob_verified": false,
+      "name": "model-00018-of-00018.safetensors",
+      "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"
+    },
+    {
+      "bytes": 112216,
+      "git_blob_verified": true,
+      "name": "model.safetensors.index.json",
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+    },
+    {
+      "bytes": 390,
+      "git_blob_verified": true,
+      "name": "preprocessor_config.json",
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
+    },
+    {
+      "bytes": 12809320,
+      "git_blob_verified": false,
+      "name": "tokenizer.json",
+      "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
+    },
+    {
+      "bytes": 17928,
+      "git_blob_verified": true,
+      "name": "tokenizer_config.json",
+      "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
+    },
+    {
+      "bytes": 385,
+      "git_blob_verified": true,
+      "name": "video_preprocessor_config.json",
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
+    },
+    {
+      "bytes": 6722759,
+      "git_blob_verified": true,
+      "name": "vocab.json",
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
+    }
+  ],
+  "weight_bytes": 55563006776
+}
--- /dev/null
+++ b/support.py
@@ -0,0 +1,115 @@
+"""Bounded metadata/source transport; no Torch/provider/model import."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+import struct
+
+HERE = Path(__file__).resolve().parent
+IMAGE = 'sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe'
+IMAGE_DIGEST = 'sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634'
+MODEL_SHA = '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'
+TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+def digest(raw):
+    return hashlib.sha256(raw).hexdigest()
+
+def sha(path):
+    with Path(path).open('rb') as file:
+        return hashlib.file_digest(file, 'sha256').hexdigest()
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON key')
+        result[key] = value
+    return result
+
+def read(path, cap=4*1024*1024, pin=None):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as file:
+        before = os.fstat(file.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
+        raw = file.read(cap+1)
+        after = os.fstat(file.fileno())
+    require(len(raw) <= cap and fingerprint(before) == fingerprint(after), 'immutable metadata FD')
+    if pin is not None:
+        require(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'exact metadata pin')
+    return raw
+
+def metadata(path, cap=4*1024*1024, pin=None):
+    return json.loads(read(path, cap, pin), object_pairs_hook=unique)
+
+def write(path, value):
+    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
+    require(len(raw) <= 4*1024*1024, 'bounded output metadata')
+    with Path(path).open('xb') as file:
+        os.fchmod(file.fileno(),0o600)
+        file.write(raw)
+
+def fingerprint(meta):
+    return [meta.st_dev,meta.st_ino,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns]
+
+def fd_sha(fd, begin, count):
+    value = hashlib.sha256()
+    for offset in range(0,count,1024*1024):
+        raw = os.pread(fd,min(1024*1024,count-offset),begin+offset)
+        require(len(raw) == min(1024*1024,count-offset), 'short source FD read')
+        value.update(raw)
+    return value.hexdigest()
+
+def source_header(fd, pin):
+    """Validate bounded original header length before the body pread."""
+    before = os.fstat(fd)
+    require(stat.S_ISREG(before.st_mode) and before.st_size == pin['file']['bytes'],
+            'exact regular selected source FD')
+    prefix = os.pread(fd, 8, 0)
+    require(len(prefix) == 8, 'complete source header length word')
+    n, = struct.unpack('<Q', prefix)
+    expected = pin['header_bytes']
+    require(type(expected) is int and 1 < expected <= 8*1024*1024 and
+            n == expected and 8+n <= before.st_size,
+            'bounded frozen source header length before body read')
+    raw = os.pread(fd, n, 8)
+    require(len(raw) == n and digest(raw) == pin['header_sha256'] and
+            fingerprint(before) == fingerprint(os.fstat(fd)),
+            'complete exact immutable original source header')
+    return n, raw
+
+def bounds(plan, page):
+    limits=plan['bounds']
+    require(page==limits['page_size']==4096 and len(plan['roles'])==16, 'fixed actual role/page closure')
+    require(len({v['name']for v in plan['roles']})==16,'unique complete original role names')
+    mapped=0
+    for item in plan['roles']:
+        require(item['dtype']=='BF16' and item['kind'] in ('projection','ordinary_rms'), 'original role codec/kind')
+        size=1
+        for dim in item['shape']:
+            require(type(dim)is int and 0<dim<=248320,'bounded original role dimension');size*=dim
+        shape=item['shape']
+        require((item['kind']=='projection' and len(shape)==2 and item['input_elements']==shape[1] and
+                 item['output_elements']==shape[0] and 0<shape[1]<=17408)or
+                (item['kind']=='ordinary_rms' and shape==[5120] and
+                 item['input_elements']==item['output_elements']==5120),
+                'exact operation input/output geometry before library initialization')
+        require(item['bytes']==2*size and item['absolute_offset']>=8 and
+                item['absolute_offset']+item['bytes']<=plan['selected_shards'][item['shard']]['file']['bytes'],
+                'full exact original role extent')
+        begin=item['absolute_offset'];aligned=begin-begin%page
+        require(item['page_aligned_offset']==aligned and item['view_offset']==begin-aligned and
+                item['mapping_bytes']==item['bytes']+begin-aligned,'bounded role-only mapping')
+        mapped+=item['mapping_bytes']
+    require(mapped==limits['mapped_bytes'] and mapped+limits['runtime_VA_allowance_bytes']+
+            limits['operator_working_allowance_bytes']+limits['payload_bytes']<limits['process_AS_bytes'] and
+            limits['process_AS_bytes']==limits['container_RAM_bytes']==12*1024**3 and
+            limits['payload_bytes']==4*1024**2,'finite explicit containment, not native physical admission')
+    return {'mapped_bytes':mapped,'process_AS_bytes':limits['process_AS_bytes'],
+            'native_ResourceArbiter_admission_qualified':False}
+
+def frozen():
+    from relocated_source import frozen as checked_frozen
+    return checked_frozen(HERE, metadata, read, require, sha)
--- /dev/null
+++ b/tensor_bits.py
@@ -0,0 +1,19 @@
+"""Bounded logical raw-bit transport only; no tensor/provider import on import."""
+import sys
+from support import require
+
+
+def tensor_bits(value, torch):
+    require(sys.byteorder == 'little', 'pinned little-endian raw-bit transport')
+    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
+            value.layout == torch.strided, 'raw-bit transport requires a CPU strided tensor')
+    require(value.dtype in (torch.bfloat16, torch.float32),
+            'raw-bit transport only original BF16/F32')
+    require(value.numel()*value.element_size() <= 128*1024**2,
+            'bounded raw-bit transport before view/copy')
+    integer = torch.int16 if value.dtype == torch.bfloat16 else torch.int32
+    # Equal element size keeps singleton/nonunit strides legal. NumPy writes
+    # logical C order; no floating conversion or source/storage mutation.
+    raw = value.detach().view(integer).numpy().tobytes(order='C')
+    require(len(raw) == value.numel()*value.element_size(), 'complete logical raw-bit transport')
+    return raw
--- /dev/null
+++ b/test_contracts.py
@@ -0,0 +1,34 @@
+"""ROOT-only source-free bounds fixtures; no tensor/model imports."""
+import copy
+import unittest
+from support import HERE,bounds,metadata
+
+
+class BoundsTests(unittest.TestCase):
+    def setUp(self):
+        self.plan=metadata(HERE/'source-plan.json')
+
+    def test_actual_16_header_role_plan_is_containment_only(self):
+        result=bounds(self.plan,4096)
+        self.assertEqual(result['mapped_bytes'],3519025784)
+        self.assertIs(result['native_ResourceArbiter_admission_qualified'],False)
+
+    def test_mapping_and_role_extent_corruption_refuses_before_libraries(self):
+        for key in ('mapping_bytes','view_offset','bytes'):
+            changed=copy.deepcopy(self.plan);changed['roles'][0][key]+=1
+            with self.subTest(key=key),self.assertRaises(ValueError):bounds(changed,4096)
+
+    def test_input_output_geometry_and_duplicate_role_refuse(self):
+        for key in ('input_elements','output_elements'):
+            changed=copy.deepcopy(self.plan);changed['roles'][0][key]+=1
+            with self.subTest(key=key),self.assertRaises(ValueError):bounds(changed,4096)
+        changed=copy.deepcopy(self.plan);changed['roles'][1]['name']=changed['roles'][0]['name']
+        with self.assertRaises(ValueError):bounds(changed,4096)
+
+    def test_unsupported_AS_and_page_geometry_refuse(self):
+        with self.assertRaises(ValueError):bounds(self.plan,8192)
+        changed=copy.deepcopy(self.plan);changed['bounds']['process_AS_bytes']-=1
+        with self.assertRaises(ValueError):bounds(changed,4096)
+
+
+if __name__=='__main__':unittest.main()
--- /dev/null
+++ b/test_relocation.py
@@ -0,0 +1,105 @@
+"""Prepared synthetic source-only relocation negatives; no operator/process."""
+import copy
+from pathlib import Path
+import unittest
+from support import require
+from relocated_source import normalized_lines, remap, validate_build
+from sigterm_cancellation import SigtermCancellation
+
+
+class Relocation(unittest.TestCase):
+    def fixture(self):
+        here = Path('/owned/isolated-pure-wide')
+        config = dict(required_isolated_status='ROOT_COMPILED_ISOLATED_ORIGINAL27B_PURE_ROLES_ONLY',
+            original_preparation=dict(sha256='1'*64),original_metadata_followup=dict(sha256='2'*64),
+            source_freeze=dict(sha256='3'*64),workspace_root='/owned/ROOT521')
+        rows = [dict(path=f'/owned/source-{i}.rs',bytes=i+1,sha256=f'{i:064x}') for i in range(526)]
+        expected = {r['path']:r for r in rows}
+        value = dict(status=config['required_isolated_status'],MAIN_merged=False,
+            full_model_StateRoot_GPU_or_performance_qualified=False,build_exit_code=0,
+            prepared_manifest_sha256='4'*64,qualification_manifest_sha256='1'*64,
+            qualification_metadata_followup_manifest_sha256='2'*64,
+            root_source_freeze_sha256='3'*64,root_worktree=config['workspace_root'],
+            qualification_directory=str(here),files=copy.deepcopy(rows))
+        return value,config,expected,here
+
+    def check(self,value,config,expected,here):
+        validate_build(value,config,expected,here,'4'*64,require)
+
+    def test_complete526_source_rows(self):
+        self.check(*self.fixture())
+
+    def test_optional_local_lock_is527th_actual_input(self):
+        value,config,expected,here = self.fixture()
+        lock = dict(path=str(here/'Cargo.lock'),bytes=123,sha256='5'*64)
+        expected[lock['path']] = lock
+        value['files'].append(lock)
+        self.check(value,config,expected,here)
+
+    def test_old_MAIN_build_status_or_merged_claim_refuses(self):
+        for field,new in [('status','ROOT_COMPILED_ORIGINAL27B_PURE_ROLES_ONLY'),('MAIN_merged',True)]:
+            value,config,expected,here = self.fixture()
+            value[field] = new
+            with self.assertRaises(ValueError):self.check(value,config,expected,here)
+
+    def test_missing_and_duplicate_source_plan_inputs_refuse(self):
+        for duplicate in (False,True):
+            value,config,expected,here = self.fixture()
+            if duplicate:value['files'][-1] = value['files'][0]
+            else:value['files'].pop()
+            with self.assertRaises(ValueError):self.check(value,config,expected,here)
+
+    def test_changed_source_or_forged_ROOT_freeze_refuses(self):
+        for freeze in (False,True):
+            value,config,expected,here = self.fixture()
+            if freeze:value['root_source_freeze_sha256'] = 'f'*64
+            else:value['files'][-1]['sha256'] = 'f'*64
+            with self.assertRaises(ValueError):self.check(value,config,expected,here)
+
+    def test_foreign_compile_directory_refuses(self):
+        value,config,expected,here = self.fixture()
+        value['qualification_directory'] = '/foreign'
+        with self.assertRaises(ValueError):self.check(value,config,expected,here)
+
+    def test_missing_additive_original_metadata_layer_refuses(self):
+        value,config,expected,here = self.fixture()
+        value['qualification_metadata_followup_manifest_sha256'] = '0'*64
+        with self.assertRaises(ValueError):self.check(value,config,expected,here)
+
+    def test_fullmodel_claim_refuses(self):
+        value,config,expected,here = self.fixture()
+        value['full_model_StateRoot_GPU_or_performance_qualified'] = True
+        with self.assertRaises(ValueError):self.check(value,config,expected,here)
+
+    def test_manifest_and_local_root_remap_are_exact(self):
+        config = dict(original_harness='/old/harness',main_root='/old/MAIN',
+                      relocated_harness='/new/harness',workspace_root='/new/ROOT')
+        self.assertEqual(remap('/old/MAIN/crates/a/src/b.rs',config,require),Path('/new/ROOT/crates/a/src/b.rs'))
+        self.assertEqual(remap('/old/harness/source-plan.json',config,require),Path('/new/harness/source-plan.json'))
+
+    def test_unlisted_local_alias_and_foreign_path_refuse(self):
+        config = dict(original_harness='/old/harness',main_root='/old/MAIN',
+                      relocated_harness='/new/harness',workspace_root='/new/ROOT')
+        for path in ['/old/harness/source-bindings.json','/foreign/x.rs','/old/MAIN/README.md']:
+            with self.assertRaises(ValueError):remap(path,config,require)
+
+    def test_exact_composition_delta_and_wrong_old_block(self):
+        delta = dict(tag='replace',original_lines=[1,2],before='old\n',after='new\n')
+        self.assertEqual(normalized_lines(b'prefix\nold\nsuffix\n',[delta],require),b'prefix\nnew\nsuffix\n')
+        delta['before'] = 'different\n'
+        with self.assertRaises(ValueError):normalized_lines(b'prefix\nold\nsuffix\n',[delta],require)
+
+    def test_overlapping_composition_edits_refuse(self):
+        edits = [dict(tag='replace',original_lines=[0,2],before='a\nb\n',after='c\n'),
+                 dict(tag='delete',original_lines=[1,2],before='b\n',after='')]
+        with self.assertRaises(ValueError):normalized_lines(b'a\nb\n',edits,require)
+
+    def test_record_only_signal_waits_for_safe_explicit_checkpoint(self):
+        guard = SigtermCancellation()
+        guard._record(15,None)
+        self.assertTrue(guard.received)
+        with self.assertRaises(InterruptedError):guard.check()
+
+
+if __name__ == '__main__':
+    unittest.main()
--- /dev/null
+++ b/validate_reference.py
@@ -0,0 +1,98 @@
+"""ROOT ONLY exact public raw-output/source/lifecycle validation, stdlib only."""
+import argparse
+import json
+import os
+from pathlib import Path
+import struct
+from support import HERE,MODEL_SHA,TORCH_GIT,bounds,digest,fd_sha,fingerprint,frozen,metadata,read,require,sha,source_header,unique,write
+
+
+def main():
+    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
+    require(args.output.is_absolute()and args.output.parent.resolve()==HERE,'owned original-reference directory')
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');inv=metadata(args.output/'invocation.json')
+    r=metadata(args.output/'captured/manifest.json')
+    require(inv['status']=='EXECUTED_ORIGINAL27B_PURE_ROLES_REQUIRES_INDEPENDENT_VALIDATION' and
+            inv['freeze_manifest_sha256']==freeze and inv['source_plan_sha256']==sha(HERE/'source-plan.json') and
+            inv['source_before']==inv['source_after'] and inv['source_FDs_closed'] is True and
+            inv['source_FD_cleanup_errors']==[] and inv['cleanup_errors']==[] and inv['exit_code']==0 and
+            inv['owned_container_absent_after_reap'] is True,'all original source/process lifetime complete')
+    g=inv['foreground_cleanup'];log=inv['log_cleanup']
+    require(g['leader_reaped'] is True and g['group_absent'] is True and g['all_nonzero_signals_before_reap'] is True and
+            g['stage_errors']==[] and g['kill_attempts_before_reap']==2 and
+            log['pipe_eof'] is log['all_closed'] is True and log['total_bytes']==log['retained_bytes']<=4*1024**2,
+            'complete owned leader/group/EOF cleanup, no truncated evidence')
+    expected={row['name']:(row['bytes'],row['sha256'])for row in plan['source_files']}
+    require(len(inv['source_before'])==len(expected) and
+            {row['name']:(row['bytes'],row['sha256'])for row in inv['source_before']}==expected,
+            'all original files/full18-shard provenance')
+    require(r['status']=='CAPTURED_ORIGINAL27B_PURE_OPERATOR_REFERENCE_REQUIRES_VALIDATION' and
+            r['source_plan_sha256']==sha(HERE/'source-plan.json') and r['source_repo']==plan['repository'] and
+            r['source_revision']==plan['revision'] and r['snapshot']==plan['snapshot'] and
+            r['torch']=='2.13.0+cu129' and r['torch_git']==TORCH_GIT and r['modeling_sha256']==MODEL_SHA and
+            r['affinity']==inv['expected_cpus'] and len(set(r['affinity']))==4 and
+            r['threads']==4 and r['interop_threads']==1 and r['pre_library_bounds']==bounds(plan,4096) and
+            r['all_original_source_and_mapped_bytes_unchanged'] is True and
+            r['source_owner_retained_until_process_exit'] is True and
+            r['whole_model_or_state_or_GPU_qualified'] is False,'actual original CPU/profile/role-only scope')
+    prefix=metadata(plan['prior_public_prefix']['path'],pin=plan['prior_public_prefix'])
+    require(len(r['cases'])==32 and len(plan['roles'])==16,'all16 roles/two teacher probes')
+    total=0;words=0;files={'manifest.json'}
+    for position,row in enumerate(r['cases']):
+        step,i=divmod(position,16);role=plan['roles'][i];token=(11,353)[step]
+        require(row['step']==step and row['teacher_token']==token and
+                all(row[key]==role[key]for key in ('role','layer','kind','input_provenance')) and
+                row['source_tensor']==role['name'] and row['source_shape']==role['shape'] and row['source_dtype']=='BF16' and
+                row['original_public_operator_calls']==2 and row['both_calls_full_bits_exact'] is True,
+                'exact original role/source/teacher/full-call closure')
+        payload={}
+        for key,count in [('input',role['input_elements']),('output',role['output_elements'])]:
+            pin=row[key];name=pin['file']
+            require(Path(name).name==name and name not in files and pin['bytes']==count*2,
+                    'unique complete BF16 payload extent')
+            raw=read(args.output/'captured'/name,2*1024**2,pin)
+            require(all(word&0x7f80!=0x7f80 for word, in struct.iter_unpack('<H',raw)),'all finite BF16 raw words')
+            payload[key]=raw;files.add(name);total+=len(raw);words+=count
+        if role['input_elements']==5120:
+            prior=prefix['steps'][step][role['input_stage']]
+            require(prefix['steps'][step]['token']==token and row['input_prior']==prior,'accepted external input provenance')
+            full=read(Path(plan['prior_public_prefix']['path']).parent/prior['file'],pin=prior)
+            require(len(full)==5120*4 and all(word&0xffff==0 for word, in struct.iter_unpack('<I',full)),
+                    'all accepted source input BF16-exact words')
+            actual=b''.join((word>>16).to_bytes(2,'little')for word, in struct.iter_unpack('<I',full))
+            require(actual==payload['input'] and digest(actual)==prior['bf16_raw_sha256'],
+                    'exact independent archived actual original input bytes')
+        else:
+            raw=b''.join(((0xbf00 if (j+token)%2 else 0x3f00)|((j*13+token)%128)).to_bytes(2,'little')
+                         for j in range(role['input_elements']))
+            require(row['input_prior'] is None and raw==payload['input'],'literal external BF16 probe, no trained producer')
+    require(total==r['payload_bytes']<=4*1024**2 and
+            {p.name for p in (args.output/'captured').iterdir()}==files,'all payload bytes/files accountable')
+    require(set(r['source_readsets'])=={v['name']for v in plan['roles']},'all16 original learned readsets')
+    for shard,pin in plan['selected_shards'].items():
+        fd=os.open(Path(plan['source_directory'])/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+        try:
+            before=fingerprint(os.fstat(fd));n,raw=source_header(fd,pin);head=json.loads(raw,object_pairs_hook=unique)
+            for role in plan['roles']:
+                if role['shard']!=shard:continue
+                header=head[role['name']]
+                require(header['shape']==role['shape'] and header['dtype']=='BF16' and
+                        8+n+header['data_offsets'][0]==role['absolute_offset'] and
+                        header['data_offsets'][1]-header['data_offsets'][0]==role['bytes'] and
+                        fd_sha(fd,role['absolute_offset'],role['bytes'])==r['source_readsets'][role['name']],
+                        'independent full original role/header/raw source-byte identity')
+            require(fingerprint(os.fstat(fd))==before,'original validation FD immutable')
+        finally:os.close(fd)
+    tree=0
+    for path in args.output.rglob('*'):
+        require(not path.is_symlink(),'no owned output symlink')
+        if path.is_file():tree+=path.stat().st_size
+    require(tree<=32*1024**2 and frozen()==freeze,'whole artifact and source/preparation closure')
+    write(args.output/'validation.json',{'status':'PASS_ORIGINAL27B_PURE_OPERATOR_REFERENCE_ONLY',
+        'manifest_sha256':sha(args.output/'captured/manifest.json'),'invocation_sha256':sha(args.output/'invocation.json'),
+        'freeze_sha256':freeze,'source_plan_sha256':sha(HERE/'source-plan.json'),'cases':32,'roles':16,
+        'words':words,'payload_bytes':total,'trained_activation_producer_for_all_roles_qualified':False,
+        'native_CPU_parity_full_model_StateRoot_GPU_or_performance_qualified':False})
+
+
+if __name__=='__main__':main()

````

