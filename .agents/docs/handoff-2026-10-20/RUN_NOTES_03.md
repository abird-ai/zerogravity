# Run Notes — volume 3

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/frozen/TODO.md

Original UTF-8 bytes: 1375; SHA256 `49e85486edfce85ab664f348cb9acf5ef990f851e8d1460c0b49d70f6e1ba15f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist — coefficient preparation only

Parent: model-execution/shared original-HF engine; owner batch_design_review.

- [x] Read accepted sigmoid/softplus, actual .8B coefficient frames, official
  source617/619, actual 2B restored-source metadata and native ABI root-actual-3.
- [x] Identify exp boundary: direct vmsExp four-call replay PASS; resource
  admission/provider and portable source schedule remain unqualified.
- [x] Prepare same-engine CPU helper, two typed plan nodes, HostMath adapter,
  OriginalWindowOperators raw dt staging, exact-role a/b imports and outputs.
  No exp/g/decay or StateRoot/model eligibility changes.
- [x] Prepare source-free CPU/core/window ownership/pressure/cancel/provider
  tests. All tests unexecuted by author; rustfmt only.
- [x] Prepare bounded ROOT-only selected public stage oracle, inputs exporter,
  actual .8B stage checks and explicit 2B public-fragment provenance.
- [x] Prepare genuine two-source Rust raw-word replay/lease/final-zero harness.
- [x] Source-review complete delta, AST/rustfmt, freeze hashes/copies/commands.
- [ ] ROOT independent review, compilation/strict tests, public oracle and
  actual Rust two-source all-head comparison.
- [ ] Full coefficients still need legitimate admitted exact exp provider/math;
  initial chunk versus cached core/FLA/full-HF/profile/performance remain closed.

````

## Source: .agents/runs/model-execution-pinned-native-provider-feasibility-20261020/FEASIBILITY.md

Original UTF-8 bytes: 15419; SHA256 `41527c2ec2569c6bf56019875d173bc335534128345e8a3ab66a9fad95729c6b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Optional pinned CPU BLAS/VML operator provider — source-only assessment

Owner: `/root/baseline_review`; parent: model-execution. Research only; no provider implementation, library load, build, runtime/model/GPU launch or source/profile promotion.

**Feasible as an optional, explicitly pinned per-operator provider in the existing Rust engine. Not production-qualified.** First qualify direct public-ABI calls from Rust inside the acquired CPU container; retain the original-HF execution guard. A host `dlopen` of this wheel is not evidence of the pinned container's math.

## Existing seams and required extension

- `core::accelerator::program::MathematicalBackend` owns value/private state operations; `qwen35::forward` owns the sole layer schedule. `HostMath` already binds typed values to source identity, numerical profile and precision.
- `MatVecKernel` provides optional regions, private completed proposals, and error propagation. Providers cannot publish a `StateRoot`. The current GDN coefficient/recurrent regions describe accepted GGUF math; they do not encode original-HF parameter dtype, VML mode, initial padded64 chunk versus cached recurrence, or exact batched matrix layout.
- Current typed HostMath embedding/projection rejects an attached untyped physical provider; CPU64 projection also refuses an unaware provider before invocation. Keep these checks and `LoadedQwenModel::require_execution_profile` closed. No existing hook can silently admit the proposed provider.
- Smallest extension: closed, source/profile-bound primitive contracts for FP32 exp, batched FP32 matrix multiplication, and unit triangular FP32 solve. Default trait methods return Unsupported; a caller with no qualified implementation refuses execution. Do not reinterpret legacy regions or replace CPU64 learned projection with GEMM.
- Contract metadata: exact `SourceIdentity` and profile, operation/site identity, FP32 dtype, dimensions/layout/strides/storage bounds, transpose/triangle/side/unit-diagonal, alpha/beta bits, batch grouping, and exp span/mode. Provider capability separately binds the immutable library/dependency closure and execution policy. Avoid treating a source ID as a runtime-library ID.
- Rust retains initial-chunk padding, FP32 materializations, source-order pointwise operations, casts, transforms and state codecs. The provider receives borrowed read sets and admitted private destinations only. It has no model graph, weights owner, token loop, session, checkpoint or publication authority.

## Acquired ABI and actual reference evidence

Static receipt: `static-evidence.json`, `public-exports.txt`, `torch-dynamic.txt`; exact artifact paths/hashes are retained there. No MKL implementation disassembly was performed for this audit.

- Actual reference image: `docker.io/vllm/vllm-openai@sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f`; Torch `2.13.0+cu129`, commit `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- `libtorch_cpu.so`: 438,418,713 B, SHA256 `1b4ed646ce51a4c341808552426a0f9e2e2c04961955759701066d335c1631ab`. MKL is bundled here; no separate MKL runtime has been established.
- Actual source public-op replay records six `SGEMM_BATCH` invocations and 32 `STRSM` calls (two solves ×16 heads), `lp64 gnu_thread`, four threads, `CNR:OFF`, `Dyn:0`, `FastMM:1`. Observed version string is **oneMKL 2024.0 Update 2 Product build 20240605**; do not infer a different release name from documentation URLs.
- Exported public symbols: `vmsExp`, `vsExp`, `sgemm_batch_`/`sgemm_batch`, `strsm_`/`strsm`, `cblas_sgemm_batch`. **`cblas_strsm` is absent** from this DSO's dynamic exports. The pinned ATen float batch wrapper actually calls public `cblas_sgemm_batch`; SGEMM_BATCH is its observed vendor verbose name. STRSM uses the Fortran symbol. Preserve the exact original entry, column-major argument swap and transpose values; see the prepared diagnostic `ABI.md` for primary caller links.
- Intel documents LP64 integer dimensions as 32-bit; vector-math mode remains 64-bit. `vmsExp` takes an explicit per-call mode. The installed Torch VML caller specifies `0x140102` (HA | FTZDAZ_OFF | ERRMODE_IGNORE). Header ABI, pointers and signed conversions must be checked, including all grouping and leading dimensions. [LP64/ILP64](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/using-the-ilp64-interface-vs-lp64-interface.html), [exp](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/v-exp.html), [batched GEMM](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-2/gemm-batch.html).
- Wheel include inventory contains none of `mkl.h`, `mkl_blas.h`, `mkl_cblas.h`, `mkl_service.h`, `mkl_types.h`, `mkl_vml.h`, `i_malloc.h`. Exact primary ATen declarations/calls plus official Intel API types establish the three narrow diagnostic signatures without an SDK download (prepared probe `ABI.md`); other service/allocator interfaces still require separate exact declarations. Thread-service aliases have distinct ELF addresses/sizes (`mkl_set_num_threads_local` versus `MKL_Set_Num_Threads_Local`); do not infer by-value C versus by-reference Fortran calling convention from a symbol's spelling.
- Original solves are `STRSM(L,U,T,U,64,128,1, A,64,B,64)` per head. The six actual GEMM transpose pairs are `ut(T,N)`, `intra(T,N)`, `old_read(N,T)`, `inter(N,N)`, `intra_values(T,N)`, `state_update(T,T)`. The public replay's exact shape/stride/input/output metadata is the contract source; reshape-contiguous guesses would invalidate it.
- Public ABI output parity would qualify use of that pinned provider, not reveal or implement its internal microkernel. HA accuracy and CNR controls are not proof of agreement with this observed CNR-off reference. Intel documents ISA/thread-dependent dispatch and CNR conditions. [Reproducibility](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-1/obtaining-numerically-reproducible-results.html).

## Same Rust executable inside the pinned container

Statically promising; **not executed**. `rust-elf.json` pins the inspected `target/release/zerogravity` artifact (SHA256 `9e2d3325685d09350600fe3673d2518e1d520db5dc8c91a843f8dceb6a647bf3`, 4,691,536 B). Its build source/features are not inferred from the filename.

- DT_NEEDED: `libc.so.6`, `libm.so.6`, `libgcc_s.so.1`; maximum required GLIBC symbol version is **2.34**, below container glibc2.39. This is necessary ABI feasibility, not a successful execution or complete version/dependency closure proof.
- PT_INTERP points to host Nix glibc2.42; DT_RUNPATH points to Nix glibc/GCC. Normal in-container execution cannot be assumed to work. A root-only probe can invoke the **container's original GNU loader directly**, with explicit container-only library paths overriding RUNPATH, the binary mounted readonly, and no host `/nix/store` mounts. GNU documents direct invocation and `--library-path`. [Loader interface](https://man7.org/linux/man-pages/man8/ld.so.8.html).
- Proposed packaging remains one Rust process/forward schedule. An outer rootless container process is launch isolation, not an operator RPC or foreign model engine. No Python/Torch tensor API is called. Direct FFI uses the acquired DSO's public entry points only.
- Actual `libtorch_cpu.so` also needs `libc10`, GNU OpenMP, C++/GCC runtime, CUDA runtime and CUPTI DSOs even for these CPU symbols. Its RPATH spans bundled NVIDIA directories; INIT_ARRAY is 3,040 B (~380 function pointers). Cold loading can run constructors before a numerical call. This is a heavy Torch wheel dependency, not a standalone CPU BLAS artifact.
- Pin the complete resolved loader/libc/libm/libgcc/libstdc++/libgomp/libc10/CUDA closure and image manifest; require mapped-file identities and public entry provenance in the root-only qualification. The reference recorded libm SHA `fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0` and libgomp SHA `59badd7c550c340a23d217c37232a4bac6e66a99f47aa273a0a0393bc7c68ddb`.
- Preserve actual CPU/ISA dispatch, four-thread policy, MXCSR/rounding/FTZ/DAZ and environment. Invoking the pinned loader removes the obvious host-libm substitution route; it does not itself prove selected IFUNC/BLAS/VML dispatch or output bits. Host `dlopen`, `RTLD_DEEPBIND`, or a new `dlmopen` namespace is a more complex unqualified alternative, especially with duplicate libc/OpenMP/TLS/allocators.
- Do not turn on CNR, replace GNU OpenMP, change FastMM, or substitute a standalone MKL release and call it the same reference. Such changes require their own declared provider/profile qualification.

## Physical memory, lifetime and cancellation gate

- Existing source owner and raw parameters remain borrowed and source-validated. No duplicated learned-matrix cache, preconversion, hidden source loading or whole55GB staging. Admit checked sizes for all materializations, pointer arrays, matrix outputs, packing/scratch and state proposals before calls; retain leases through actual synchronous return. A small scratch/output allocation is not a licence to copy the full source.
- **Tensor leases alone do not account for foreign allocations.** This DSO's cold mappings/constructors, C++ metadata, GNU OpenMP workers/stacks, BLAS packing and retained FastMM buffers need an explicit admitted lifetime and independently measured coverage. Library/allocator lifetime may exceed one call/session; do not release its charge at model drop while pools survive.
- Public allocator-pointer objects `i_malloc`, `i_free`, `i_calloc`, `i_realloc` are exported. Intel documents installing replacements before the first library call. This is a possible admission seam, not proof that all Torch/loader/OpenMP allocations traverse it. Installation is process-global, cannot race other users, and exact existing-owner/free/realloc/alignment/OOM behavior must be established. No internal `mkl_serv_*` symbols are proposed as public ABI. [Allocator replacement](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/redefining-memory-functions.html).
- Requested public `mkl_mem_stat`, `mkl_peak_mem_usage`, `mkl_free_buffers`, `mkl_thread_free_buffers`, `mkl_disable_fast_mm`, `mkl_set_memory_limit`, `mkl_malloc`, `mkl_free` names are absent from this DSO's dynamic exports. Internal similarly named exports do not confer a supported ABI. A matching official header might map service names to other public wrappers; verify instead of guessing.
- `mkl_peak_mem_usage` is retrospective application-wide allocator telemetry, not pre-admission. Documented `mkl_set_memory_limit` controls MCDRAM, not a general host heap cap. Neither establishes ResourceArbiter coverage. A process/cgroup/RLIMIT cap is additional containment; it cannot replace admission or turn native OOM/abort into safe Rust `Result`. [Peak memory](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-1/mkl-peak-mem-usage.html), [MCDRAM limit](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/mkl-set-memory-limit.html).
- Foreign synchronous calls are not cooperatively interruptible. Bound dimensions/call duration; check cancellation before entry, retain all read/write/runtime leases while in flight, discard outputs after return if canceled, and publish nothing. Dropping a future must not free FFI buffers. Native fatal/OOM behavior is not catchable Rust unwinding. Never unload an active DSO or pool; a first implementation should retain its handle/runtime charge through process lifetime unless complete quiescence is proven.
- Reserve one fixed provider execution policy under TaskCoordinator/CPU budget; prevent nested OpenMP/Rayon oversubscription and global setting races. Use verified public thread-local save/restore if needed; Intel documents settings persist on reused OpenMP threads. [Thread-local control](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/mkl-set-num-threads-local.html).

## License and packaging boundary

- The acquired wheel supplies PyTorch BSD and 98 license files; targeted wheel-license inventory found oneDNN/LLVM OpenMP notices but no identified MKL-specific packaged agreement. This is not a whole-image legal audit. Do not label bundled Intel code BSD, extract/repackage its implementation as open Rust code, or assume redistribution clearance from an export table.
- Intel's FAQ permits redistribution subject to ISSL. The official page lists oneMKL under ISSL and says the bundled agreement controls; its public ISSL includes notice/binary-use conditions and reverse-engineering restrictions. Obtain/preserve the exact matching packaged agreement/notices before redistributing a derived runtime. This research neither distributes nor modifies the binary. [FAQ](https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html), [Intel licence page](https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html#inpage-nav-1).
- Preferred diagnostic package: already acquired immutable image + original DSO closure + readonly Rust artifact. Preferred future distributable candidate: optional independently pinned licensed CPU-only MKL closure, if separately obtained/authorized and requalified against actual source outputs. It may avoid Torch/CUDA dependencies but is not established equivalent today. No mandatory SDK, system packages, host linker configuration or driver changes.

## Recommended next decision and bounded root-only gates

1. Approve only a diagnostic public-ABI adapter/probe first, under the explicit original-HF CPU reference policy; no automatic implementation of complete HF execution. Root alone builds/runs. Pin official matching ABI declarations and all resolved files before calls.
2. Prove same Rust artifact starts through the pinned container loader with exact dependency resolution. Existing image, CPU-only/no network, ≤four CPUs, finite startup/runtime/RAM caps; no Python/model load, no GPU. Report mappings/versions/dispatch conditions, constructor and worker memory separately.
3. Replay actual original A_log and decay exp spans with exact mode, including noncontiguous source gather/scatter, then the six GEMM and 32 STRSM shapes using only authentic clean input archives. Compare every output word, signed zero and nonfinite status; include independent bounded layout/alias/overflow/cancel/refusal cases. Do not embed expected outputs in the provider or generalize a head16 replay to head48/27B geometry.
4. Establish pre-admitted cold/runtime/internal allocation behavior, fixed-thread concurrent exclusion, refusal before FFI, cancellation retention and final charge accounting. If full admission cannot be demonstrated, keep it diagnostic even when numerical words match.
5. Only after those gates: wire typed primitive contracts into the same HostMath/recorded plan and original-source state codecs; compare original full logits/all state domains and immutable source across initial/cached/mixed histories. Remaining operators and full-HF execution guard stay closed until qualified. CUDA/FLA, GGUF, serving and vLLM performance eligibility are separate.

Current verdict: **PASS_STATIC_ABI_FEASIBILITY_ONLY**. Runtime compatibility, exact FFI outputs, memory admission, redistribution closure and full-model inference remain **UNQUALIFIED**.

````

## Source: .agents/runs/model-execution-pinned-native-provider-feasibility-20261020/TODO.md

Original UTF-8 bytes: 901; SHA256 `7e01afb63e99a85e551c142db8602eae84a7f3bb821c3c9f9c5e21d681c7d437`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pinned native CPU operator feasibility

- Parent: `.agents/plans/model-execution/TODO.md`; owner `/root/baseline_review`.
- Scope: own ignored run only; source/static ELF/official API research. Root owns implementation and all builds/runtime/model/GPU qualification.
- [x] Read current MathematicalBackend/HostMath/MatVecKernel and original-HF guards.
- [x] Verify acquired Torch public exports/dependencies and pin original chunk observations.
- [x] Inspect current Rust ELF's container ABI feasibility; no binary execution.
- [x] Identify internal allocation, cancellation, runtime closure and licence boundaries.
- [x] Write recommendation and exact static receipts; no runtime capability claim.
- [ ] Root decision: optional diagnostic FFI adapter/probe, official ABI declarations and closure pins.
- [ ] Root-only loader/public-ABI exact numerical/admission qualification; remains unqualified.

````

## Source: .agents/runs/model-execution-pinned-native-provider-feasibility-20261020/initial-frozen/FEASIBILITY.md

Original UTF-8 bytes: 15117; SHA256 `4dbcf433273df9fc6f59d8940e6968e8431827ba6ee1e6307d8c8ea57a9543a5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Optional pinned CPU BLAS/VML operator provider — source-only assessment

Owner: `/root/baseline_review`; parent: model-execution. Research only; no provider implementation, library load, build, runtime/model/GPU launch or source/profile promotion.

**Feasible as an optional, explicitly pinned per-operator provider in the existing Rust engine. Not production-qualified.** First qualify direct public-ABI calls from Rust inside the acquired CPU container; retain the original-HF execution guard. A host `dlopen` of this wheel is not evidence of the pinned container's math.

## Existing seams and required extension

- `core::accelerator::program::MathematicalBackend` owns value/private state operations; `qwen35::forward` owns the sole layer schedule. `HostMath` already binds typed values to source identity, numerical profile and precision.
- `MatVecKernel` provides optional regions, private completed proposals, and error propagation. Providers cannot publish a `StateRoot`. The current GDN coefficient/recurrent regions describe accepted GGUF math; they do not encode original-HF parameter dtype, VML mode, initial padded64 chunk versus cached recurrence, or exact batched matrix layout.
- Current typed HostMath embedding/projection rejects an attached untyped physical provider; CPU64 projection also refuses an unaware provider before invocation. Keep these checks and `LoadedQwenModel::require_execution_profile` closed. No existing hook can silently admit the proposed provider.
- Smallest extension: closed, source/profile-bound primitive contracts for FP32 exp, batched FP32 matrix multiplication, and unit triangular FP32 solve. Default trait methods return Unsupported; a caller with no qualified implementation refuses execution. Do not reinterpret legacy regions or replace CPU64 learned projection with GEMM.
- Contract metadata: exact `SourceIdentity` and profile, operation/site identity, FP32 dtype, dimensions/layout/strides/storage bounds, transpose/triangle/side/unit-diagonal, alpha/beta bits, batch grouping, and exp span/mode. Provider capability separately binds the immutable library/dependency closure and execution policy. Avoid treating a source ID as a runtime-library ID.
- Rust retains initial-chunk padding, FP32 materializations, source-order pointwise operations, casts, transforms and state codecs. The provider receives borrowed read sets and admitted private destinations only. It has no model graph, weights owner, token loop, session, checkpoint or publication authority.

## Acquired ABI and actual reference evidence

Static receipt: `static-evidence.json`, `public-exports.txt`, `torch-dynamic.txt`; exact artifact paths/hashes are retained there. No MKL implementation disassembly was performed for this audit.

- Actual reference image: `docker.io/vllm/vllm-openai@sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f`; Torch `2.13.0+cu129`, commit `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- `libtorch_cpu.so`: 438,418,713 B, SHA256 `1b4ed646ce51a4c341808552426a0f9e2e2c04961955759701066d335c1631ab`. MKL is bundled here; no separate MKL runtime has been established.
- Actual source public-op replay records six `SGEMM_BATCH` invocations and 32 `STRSM` calls (two solves ×16 heads), `lp64 gnu_thread`, four threads, `CNR:OFF`, `Dyn:0`, `FastMM:1`. Observed version string is **oneMKL 2024.0 Update 2 Product build 20240605**; do not infer a different release name from documentation URLs.
- Exported public symbols: `vmsExp`, `vsExp`, `sgemm_batch_`/`sgemm_batch`, `strsm_`/`strsm`, `cblas_sgemm_batch`. **`cblas_strsm` is absent** from this DSO's dynamic exports. The original source path uses the Fortran BLAS symbols; preserve its column-major interpretation and transpose arguments before considering any wrapper.
- Intel documents LP64 integer dimensions as 32-bit; vector-math mode remains 64-bit. `vmsExp` takes an explicit per-call mode. The installed Torch VML caller specifies `0x140102` (HA | FTZDAZ_OFF | ERRMODE_IGNORE). Header ABI, pointers and signed conversions must be checked, including all grouping and leading dimensions. [LP64/ILP64](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/using-the-ilp64-interface-vs-lp64-interface.html), [exp](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/v-exp.html), [batched GEMM](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-2/gemm-batch.html).
- Wheel include inventory contains none of `mkl.h`, `mkl_blas.h`, `mkl_cblas.h`, `mkl_service.h`, `mkl_types.h`, `mkl_vml.h`, `i_malloc.h`. Official matching ABI declarations still need binding/verification before an FFI implementation. Thread-service aliases have distinct ELF addresses/sizes (`mkl_set_num_threads_local` versus `MKL_Set_Num_Threads_Local`); do not infer by-value C versus by-reference Fortran calling convention from a symbol's spelling.
- Original solves are `STRSM(L,U,T,U,64,128,1, A,64,B,64)` per head. The six actual GEMM transpose pairs are `ut(T,N)`, `intra(T,N)`, `old_read(N,T)`, `inter(N,N)`, `intra_values(T,N)`, `state_update(T,T)`. The public replay's exact shape/stride/input/output metadata is the contract source; reshape-contiguous guesses would invalidate it.
- Public ABI output parity would qualify use of that pinned provider, not reveal or implement its internal microkernel. HA accuracy and CNR controls are not proof of agreement with this observed CNR-off reference. Intel documents ISA/thread-dependent dispatch and CNR conditions. [Reproducibility](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-1/obtaining-numerically-reproducible-results.html).

## Same Rust executable inside the pinned container

Statically promising; **not executed**. `rust-elf.json` pins the inspected `target/release/zerogravity` artifact (SHA256 `9e2d3325685d09350600fe3673d2518e1d520db5dc8c91a843f8dceb6a647bf3`, 4,691,536 B). Its build source/features are not inferred from the filename.

- DT_NEEDED: `libc.so.6`, `libm.so.6`, `libgcc_s.so.1`; maximum required GLIBC symbol version is **2.34**, below container glibc2.39. This is necessary ABI feasibility, not a successful execution or complete version/dependency closure proof.
- PT_INTERP points to host Nix glibc2.42; DT_RUNPATH points to Nix glibc/GCC. Normal in-container execution cannot be assumed to work. A root-only probe can invoke the **container's original GNU loader directly**, with explicit container-only library paths overriding RUNPATH, the binary mounted readonly, and no host `/nix/store` mounts. GNU documents direct invocation and `--library-path`. [Loader interface](https://man7.org/linux/man-pages/man8/ld.so.8.html).
- Proposed packaging remains one Rust process/forward schedule. An outer rootless container process is launch isolation, not an operator RPC or foreign model engine. No Python/Torch tensor API is called. Direct FFI uses the acquired DSO's public entry points only.
- Actual `libtorch_cpu.so` also needs `libc10`, GNU OpenMP, C++/GCC runtime, CUDA runtime and CUPTI DSOs even for these CPU symbols. Its RPATH spans bundled NVIDIA directories; INIT_ARRAY is 3,040 B (~380 function pointers). Cold loading can run constructors before a numerical call. This is a heavy Torch wheel dependency, not a standalone CPU BLAS artifact.
- Pin the complete resolved loader/libc/libm/libgcc/libstdc++/libgomp/libc10/CUDA closure and image manifest; require mapped-file identities and public entry provenance in the root-only qualification. The reference recorded libm SHA `fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0` and libgomp SHA `59badd7c550c340a23d217c37232a4bac6e66a99f47aa273a0a0393bc7c68ddb`.
- Preserve actual CPU/ISA dispatch, four-thread policy, MXCSR/rounding/FTZ/DAZ and environment. Invoking the pinned loader removes the obvious host-libm substitution route; it does not itself prove selected IFUNC/BLAS/VML dispatch or output bits. Host `dlopen`, `RTLD_DEEPBIND`, or a new `dlmopen` namespace is a more complex unqualified alternative, especially with duplicate libc/OpenMP/TLS/allocators.
- Do not turn on CNR, replace GNU OpenMP, change FastMM, or substitute a standalone MKL release and call it the same reference. Such changes require their own declared provider/profile qualification.

## Physical memory, lifetime and cancellation gate

- Existing source owner and raw parameters remain borrowed and source-validated. No duplicated learned-matrix cache, preconversion, hidden source loading or whole55GB staging. Admit checked sizes for all materializations, pointer arrays, matrix outputs, packing/scratch and state proposals before calls; retain leases through actual synchronous return. A small scratch/output allocation is not a licence to copy the full source.
- **Tensor leases alone do not account for foreign allocations.** This DSO's cold mappings/constructors, C++ metadata, GNU OpenMP workers/stacks, BLAS packing and retained FastMM buffers need an explicit admitted lifetime and independently measured coverage. Library/allocator lifetime may exceed one call/session; do not release its charge at model drop while pools survive.
- Public allocator-pointer objects `i_malloc`, `i_free`, `i_calloc`, `i_realloc` are exported. Intel documents installing replacements before the first library call. This is a possible admission seam, not proof that all Torch/loader/OpenMP allocations traverse it. Installation is process-global, cannot race other users, and exact existing-owner/free/realloc/alignment/OOM behavior must be established. No internal `mkl_serv_*` symbols are proposed as public ABI. [Allocator replacement](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/redefining-memory-functions.html).
- Requested public `mkl_mem_stat`, `mkl_peak_mem_usage`, `mkl_free_buffers`, `mkl_thread_free_buffers`, `mkl_disable_fast_mm`, `mkl_set_memory_limit`, `mkl_malloc`, `mkl_free` names are absent from this DSO's dynamic exports. Internal similarly named exports do not confer a supported ABI. A matching official header might map service names to other public wrappers; verify instead of guessing.
- `mkl_peak_mem_usage` is retrospective application-wide allocator telemetry, not pre-admission. Documented `mkl_set_memory_limit` controls MCDRAM, not a general host heap cap. Neither establishes ResourceArbiter coverage. A process/cgroup/RLIMIT cap is additional containment; it cannot replace admission or turn native OOM/abort into safe Rust `Result`. [Peak memory](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-1/mkl-peak-mem-usage.html), [MCDRAM limit](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/mkl-set-memory-limit.html).
- Foreign synchronous calls are not cooperatively interruptible. Bound dimensions/call duration; check cancellation before entry, retain all read/write/runtime leases while in flight, discard outputs after return if canceled, and publish nothing. Dropping a future must not free FFI buffers. Native fatal/OOM behavior is not catchable Rust unwinding. Never unload an active DSO or pool; a first implementation should retain its handle/runtime charge through process lifetime unless complete quiescence is proven.
- Reserve one fixed provider execution policy under TaskCoordinator/CPU budget; prevent nested OpenMP/Rayon oversubscription and global setting races. Use verified public thread-local save/restore if needed; Intel documents settings persist on reused OpenMP threads. [Thread-local control](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/mkl-set-num-threads-local.html).

## License and packaging boundary

- The acquired wheel supplies PyTorch BSD and 98 license files; targeted wheel-license inventory found oneDNN/LLVM OpenMP notices but no identified MKL-specific packaged agreement. This is not a whole-image legal audit. Do not label bundled Intel code BSD, extract/repackage its implementation as open Rust code, or assume redistribution clearance from an export table.
- Intel's FAQ permits redistribution subject to ISSL. The official page lists oneMKL under ISSL and says the bundled agreement controls; its public ISSL includes notice/binary-use conditions and reverse-engineering restrictions. Obtain/preserve the exact matching packaged agreement/notices before redistributing a derived runtime. This research neither distributes nor modifies the binary. [FAQ](https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html), [Intel licence page](https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html#inpage-nav-1).
- Preferred diagnostic package: already acquired immutable image + original DSO closure + readonly Rust artifact. Preferred future distributable candidate: optional independently pinned licensed CPU-only MKL closure, if separately obtained/authorized and requalified against actual source outputs. It may avoid Torch/CUDA dependencies but is not established equivalent today. No mandatory SDK, system packages, host linker configuration or driver changes.

## Recommended next decision and bounded root-only gates

1. Approve only a diagnostic public-ABI adapter/probe first, under the explicit original-HF CPU reference policy; no automatic implementation of complete HF execution. Root alone builds/runs. Pin official matching ABI declarations and all resolved files before calls.
2. Prove same Rust artifact starts through the pinned container loader with exact dependency resolution. Existing image, CPU-only/no network, ≤four CPUs, finite startup/runtime/RAM caps; no Python/model load, no GPU. Report mappings/versions/dispatch conditions, constructor and worker memory separately.
3. Replay actual original A_log and decay exp spans with exact mode, including noncontiguous source gather/scatter, then the six GEMM and 32 STRSM shapes using only authentic clean input archives. Compare every output word, signed zero and nonfinite status; include independent bounded layout/alias/overflow/cancel/refusal cases. Do not embed expected outputs in the provider or generalize a head16 replay to head48/27B geometry.
4. Establish pre-admitted cold/runtime/internal allocation behavior, fixed-thread concurrent exclusion, refusal before FFI, cancellation retention and final charge accounting. If full admission cannot be demonstrated, keep it diagnostic even when numerical words match.
5. Only after those gates: wire typed primitive contracts into the same HostMath/recorded plan and original-source state codecs; compare original full logits/all state domains and immutable source across initial/cached/mixed histories. Remaining operators and full-HF execution guard stay closed until qualified. CUDA/FLA, GGUF, serving and vLLM performance eligibility are separate.

Current verdict: **PASS_STATIC_ABI_FEASIBILITY_ONLY**. Runtime compatibility, exact FFI outputs, memory admission, redistribution closure and full-model inference remain **UNQUALIFIED**.

````

## Source: .agents/runs/model-execution-pinned-native-provider-probe-20261020/ABI.md

Original UTF-8 bytes: 5651; SHA256 `9898e1ad9e95de82d9510083d44843d7d89b9e25649f0b18eec1f0e614e92f39`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed public ABI — source derivation

Research/prepared diagnostic only. No vendor implementation-source derivation,
native build/load/call by author, model engine, state owner or profile promotion.

- Exact Torch source commit: `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Actual captured source .8B, layer0, source36-F32-restored, tokens11/353;
  fixture metadata retains original byte ranges, source names/strides and hashes.
- Expected output files contain independent original public-Torch outputs. Rust
  first opens expected-output bytes **after** native completion. Provider calls
  receive only captured inputs and private initialized/poisoned destinations.

## Three public entries

1. `vmsExp`: Intel's [2024.2 exp API](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/v-exp.html)
   specifies LP64 `MKL_INT` count and explicit `MKL_INT64` mode. Installed pinned
   `ATen/cpu/vml.h` invokes the mode-aware FP32 function with HA|FTZDAZ_OFF|
   ERRMODE_IGNORE, value0x140102. Probe signature uses int32 count by value,
   FP32 input/output pointers and int64 mode by value. No default global VML mode.
   Original [UnaryOpsKernel.cpp iterator](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp#L724)
   gathers noncontiguous spans to a contiguous local buffer. Probe reconstructs
   actual source view, gathers16 original words without arithmetic and calls once.
   Actual internal caller alignment/selected VML microkernel is unobserved; this
   diagnostic can establish only output parity for these four captured calls.
2. **`cblas_sgemm_batch`**, not direct Fortran `sgemm_batch_`: pinned
   [ATen mkl/LinearAlgebra.cpp66–74](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/mkl/LinearAlgebra.cpp#L66)
   supplies column-major layout, by-value group count1 and group-size16 pointer;
   matrices/scalars/dimensions/leading dimensions are per-group arrays. Official
   [C interface enum values](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/c-interface-conventions-for-blas-routines.html)
   establish ColMajor102, NoTrans111, Trans112, and
   [grouped C API](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-gemm-batch.html)
   supplies the exact signature. No hidden character lengths exist in this C API.
   Actual MKL_VERBOSE calls are named SGEMM_BATCH; that name alone does not
   identify the original public wrapper. Public dynamic symbol is exported.
3. `strsm_`: pinned [ATen declaration452](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/BatchLinearAlgebra.cpp#L452)
   explicitly gives11 pointer arguments: four char*, two int*, float*alpha,
   float*A, int*lda, float*B, int*ldb. Its
   [FP32 call1436–1438](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/BatchLinearAlgebra.cpp#L1436)
   passes exactly those11 and alpha1. No hidden string lengths are inferred or
   added. [Kernel716–740](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp#L716)
   loops heads with matrix strides and scalar flags. Existing source capture
   independently observes all32 STRSM(L,U,T,U,64,128,...64,...64) calls.

LP64 int32 is independently established by actual source MKL_VERBOSE;
[official integer contract](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/using-the-ilp64-interface-vs-lp64-interface.html)
keeps VML mode64-bit even with32-bit dimensions. Exact primary callers and public
ABI docs suffice for these three diagnostic signatures; no missing SDK header is
guessed. No thread-service/allocator/internal MKL ABI is used.

## Captured matrix transport

All six families batch16, alpha1/beta0. Original row-major output means ATen swaps
logical operands: native A=logical RHS, B=logical LHS. Buffer reconstruction uses
the actual observed full shape/stride; C holds the original row-major output.
Original source-independent layout analysis in the parent batch agent's ignored
`p12-hf-ordinary-rms-20261020/.../INITIAL_CHUNK_AUDIT.md` is corroboration; actual
report metadata and pinned original source are the binding evidence.

| Family | transA/B | M,N,K | lda/ldb/ldc |
|---|---|---|---|
| ut | T,N |64,64,128|128/128/64|
| intra | T,N |64,64,128|128/128/64|
| old_read | N,T |128,64,128|128/64/128|
| inter | N,N |128,64,128|128/128/128|
| intra_values | T,N |128,64,64|64/64/128|
| state_update | T,T |128,128,64|64/128/128|

Solves each use16 heads. A remains original row-major raw storage, interpreted
as column-major upper + transpose. B is copied from captured RHS logical words
to the private original output's column-major storage before the in-place call.
The copied bytes are RHS **inputs**, never expected results. No algebraic
elimination of initial zeros/identity-like solves, no tolerance or signed-zero
exception. Redzones and unchanged raw read sets checked after each group.

## ABI/provider provenance gate

`dlsym`+`dladdr` must bind each public entry to the hash-pinned original
libtorch_cpu.so and public ELF offsets vmsExp0x08ba7c10,
cblas_sgemm_batch0x08bd4a40, strsm_0x08bc9250. These are exported public entry
identities, **not internal kernel provenance**. Source parameter metadata and
profile guards are untouched. PyTorch BSD notice is retained alongside this
diagnostic. No Intel implementation code was copied or disassembled.

````

## Source: .agents/runs/model-execution-pinned-native-provider-probe-20261020/README.md

Original UTF-8 bytes: 5379; SHA256 `6f25c7445955450bcbe51f49acd6ccc33a8939ba0c37bc4318eb8417248efef3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only Rust public BLAS/VML diagnostic

Parent: model-execution; owner `/root/baseline_review`. **PREPARED ONLY**: author
performed byte/header/ELF source inspection and Python syntax checks, no Rust/
Cargo/C/native/container/Torch/math/GPU execution. Production is untouched.

The smallest diagnostic calls genuine acquired public native entry points from
one standalone Rust process. There is no Python computation, foreign model,
forward/token/session loop, RPC, state publication, vendor arithmetic port or
learned matrix copy. This does not yet wire an operator provider into HostMath.

- `fixture-manifest.json` binds12 cases: four exp16 calls from actual source
  A_log/g at tokens11/353; six grouped CBLAS GEMM calls; two16-head STRSM groups.
  Byte-only copied payload12,845,568B, total diagnostic cap32MiB.
- Raw source archive `original-source-operators.safetensors` is pinned SHA
  65992c168c90762f48fee1523441e6154e41f15fd049e4320f4dea6499d9c423.
  The independent original public outputs are preserved, not reconstructed.
- Exact public signatures and shape/argument derivation: `ABI.md`.
- `static-closure-pins.json` pins14 candidate ELF dependency files from the exact
  original image's40-layer chain. This is static evidence, not a claim all these
  files are or are the only files mapped. Rust requires actual mapped-file
  paths/hashes match these candidates (plus its frozen binary); any unknown
  constructor-loaded DSO refuses. All candidates are hashed before dlopen.
- Original container GNU loader is the entrypoint with explicit library paths,
  no host `/nix/store` mounts. The binary's required GLIBC symbols must≤2.39;
  PT_INTERP/RUNPATH are recorded, never assumed compatible from filenames.
  Direct-loader `/proc/self/exe` points to the loader; the Rust artifact's mapped
  identity is separately checked at `/binary/public-abi-probe`.
- Exact image digest b18abb2d…; `--pull=never --network=none`, no GPU/device/driver
  mounts, readonly root/run/binary, private output only; twelveGiB memory/swap,
  four logical CPUs +fixed four threads,256pids,300s deadline,64MiB file bound,
  fourMiB continuously drained log. CPU32/FMA/ISA/library dispatch remains the
  genuine same-host runtime, not a claimed portable Rust schedule.
- Rust writes staged proc-map/status/thread/MXCSR and public-entry receipts,
  each actual output, full raw-word difference counts/first16 differences.
  Input redzones/read-set immutability and MXCSR control are checked. No DSO
  unload; foreign pools survive only until this owned process ends.
- Launcher verifies actual6 SGEMM_BATCH/32 STRSM verbose lines, transpose/flags,
  LP64/GNU-thread productbuild20240605, four threads/CNRoff/Dyn0/FastMM1.
  These are public dispatch observations; internal microkernel remains
  **UNOBSERVED**. VML has no assumed MKL_VERBOSE support; exact public entry,
  actual call count/mode and output bits are recorded in Rust.
- OMP worker/pool/internal native allocation is **not ResourceArbiter admitted**.
  Container limits/proc RSS are containment/observation only. Even a successful
  run leaves production memory admission/profile/full-model/serving/vLLM gates
  **UNQUALIFIED**. No library/source-profile promotion follows automatically.

Root may build offline in its serialized Cargo queue after reviewing source:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 cargo build --offline --locked --release \
  --manifest-path .agents/runs/model-execution-pinned-native-provider-probe-20261020/Cargo.toml \
  --target-dir .agents/runs/model-execution-pinned-native-provider-probe-20261020/target
```

The standalone lock reuses exact repository lock blocks. If Cargo requires a
feature/dependency lock normalization, root must inspect and retain that actual
offline change; do not enable fetching. Copy the resulting artifact to a stable
owned path and record its real SHA before launch; shared target path is mutable.

```sh
# ROOT ONLY, actual artifact/hash values required; new output directory only.
python3 .agents/runs/model-execution-pinned-native-provider-probe-20261020/run_probe.py \
  --run --binary /ABSOLUTE/OWNED/FROZEN/public-abi-probe \
  --binary-sha256 ACTUAL_64_LOWERCASE_HEX_SHA256 \
  --output .agents/runs/model-execution-pinned-native-provider-probe-20261020/root-actual-1
```

No SDK/headers/model download is needed for this prepared ABI slice. Source
pins, fixtures and static closure must be reviewed before root compiles/runs.
Invocation cleanup owns a unique nonce-label/name/CID and foreground PGID only;
no arbitrary existing container/process signals. Ownership begins immediately
after spawn, before receipt writes; cleanup/log drains precede final evidence
validation/I/O. Failure/truncation/unknown library/cleanup refusal stays
UNQUALIFIED; no automatic rerun or next provider. `prepared-manifest.json`
retains exact source/input hashes. Licence/redistribution closure is still the
source-only audit's open boundary; no library is redistributed here.

Root review follow-up preserves initial `frozen-prepared/` bytes: explicit
actual/expected output-length equality before comparison, and a second
nonce/name/CID-bound container cleanup/absence check after foreground CLI reap.
An empty CID file during owned startup uses the independently nonce-bound name;
nonempty malformed/foreign ownership is refused without container signals.

````

## Source: .agents/runs/model-execution-pinned-native-provider-probe-20261020/TODO.md

Original UTF-8 bytes: 1402; SHA256 `078813c4323bbdc1b8329314dd6a502f11fecce46e3e42f93429b9e3d802b742`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native public-ABI diagnostic preparation

- Parent: `.agents/plans/model-execution/TODO.md`; owner `/root/baseline_review`.
- Own ignored run only. Root alone builds, loads libraries or executes probes.
- [x] Derive exact CBLAS grouped signature and STRSM11pointer/no-length call from pinned ATen primary source; VML count/mode from original header +Intel API.
- [x] Byte-copy authentic clean inputs/independent outputs; preserve full shape/stride/source/hash binding, no arithmetic expectations reconstructed.
- [x] Prepare one Rust public-operator diagnostic with bounded private aligned buffers, raw-word comparison and actual mapped closure/entry/read-set receipts.
- [x] Prepare exact GNU-loader/container launcher with static ELF version guard, fixed finite limits and owned cleanup before final evidence I/O.
- [x] Python AST/syntax only; no Cargo/Rust/native/container/model/runtime execution.
- [x] Root review repairs: exact output-length equality and post-CLI-reap independent owned-container cleanup/absence; empty startup CID handled by nonce-bound name. First frozen bytes retained.
- [ ] Independent source/ABI/layout/cleanup review and root build/strict source checks.
- [ ] Root-only bounded actual native output/closure/dispatch qualification; no inference claim.
- [ ] Native allocation/admission/production packaging and same-engine typed wiring remain separate unqualified work.

````

## Source: .agents/runs/model-execution-pinned-native-provider-probe-20261020/frozen-prepared/ABI.md

Original UTF-8 bytes: 5651; SHA256 `9898e1ad9e95de82d9510083d44843d7d89b9e25649f0b18eec1f0e614e92f39`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed public ABI — source derivation

Research/prepared diagnostic only. No vendor implementation-source derivation,
native build/load/call by author, model engine, state owner or profile promotion.

- Exact Torch source commit: `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Actual captured source .8B, layer0, source36-F32-restored, tokens11/353;
  fixture metadata retains original byte ranges, source names/strides and hashes.
- Expected output files contain independent original public-Torch outputs. Rust
  first opens expected-output bytes **after** native completion. Provider calls
  receive only captured inputs and private initialized/poisoned destinations.

## Three public entries

1. `vmsExp`: Intel's [2024.2 exp API](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/v-exp.html)
   specifies LP64 `MKL_INT` count and explicit `MKL_INT64` mode. Installed pinned
   `ATen/cpu/vml.h` invokes the mode-aware FP32 function with HA|FTZDAZ_OFF|
   ERRMODE_IGNORE, value0x140102. Probe signature uses int32 count by value,
   FP32 input/output pointers and int64 mode by value. No default global VML mode.
   Original [UnaryOpsKernel.cpp iterator](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp#L724)
   gathers noncontiguous spans to a contiguous local buffer. Probe reconstructs
   actual source view, gathers16 original words without arithmetic and calls once.
   Actual internal caller alignment/selected VML microkernel is unobserved; this
   diagnostic can establish only output parity for these four captured calls.
2. **`cblas_sgemm_batch`**, not direct Fortran `sgemm_batch_`: pinned
   [ATen mkl/LinearAlgebra.cpp66–74](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/mkl/LinearAlgebra.cpp#L66)
   supplies column-major layout, by-value group count1 and group-size16 pointer;
   matrices/scalars/dimensions/leading dimensions are per-group arrays. Official
   [C interface enum values](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/c-interface-conventions-for-blas-routines.html)
   establish ColMajor102, NoTrans111, Trans112, and
   [grouped C API](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-gemm-batch.html)
   supplies the exact signature. No hidden character lengths exist in this C API.
   Actual MKL_VERBOSE calls are named SGEMM_BATCH; that name alone does not
   identify the original public wrapper. Public dynamic symbol is exported.
3. `strsm_`: pinned [ATen declaration452](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/BatchLinearAlgebra.cpp#L452)
   explicitly gives11 pointer arguments: four char*, two int*, float*alpha,
   float*A, int*lda, float*B, int*ldb. Its
   [FP32 call1436–1438](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/BatchLinearAlgebra.cpp#L1436)
   passes exactly those11 and alpha1. No hidden string lengths are inferred or
   added. [Kernel716–740](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp#L716)
   loops heads with matrix strides and scalar flags. Existing source capture
   independently observes all32 STRSM(L,U,T,U,64,128,...64,...64) calls.

LP64 int32 is independently established by actual source MKL_VERBOSE;
[official integer contract](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/using-the-ilp64-interface-vs-lp64-interface.html)
keeps VML mode64-bit even with32-bit dimensions. Exact primary callers and public
ABI docs suffice for these three diagnostic signatures; no missing SDK header is
guessed. No thread-service/allocator/internal MKL ABI is used.

## Captured matrix transport

All six families batch16, alpha1/beta0. Original row-major output means ATen swaps
logical operands: native A=logical RHS, B=logical LHS. Buffer reconstruction uses
the actual observed full shape/stride; C holds the original row-major output.
Original source-independent layout analysis in the parent batch agent's ignored
`p12-hf-ordinary-rms-20261020/.../INITIAL_CHUNK_AUDIT.md` is corroboration; actual
report metadata and pinned original source are the binding evidence.

| Family | transA/B | M,N,K | lda/ldb/ldc |
|---|---|---|---|
| ut | T,N |64,64,128|128/128/64|
| intra | T,N |64,64,128|128/128/64|
| old_read | N,T |128,64,128|128/64/128|
| inter | N,N |128,64,128|128/128/128|
| intra_values | T,N |128,64,64|64/64/128|
| state_update | T,T |128,128,64|64/128/128|

Solves each use16 heads. A remains original row-major raw storage, interpreted
as column-major upper + transpose. B is copied from captured RHS logical words
to the private original output's column-major storage before the in-place call.
The copied bytes are RHS **inputs**, never expected results. No algebraic
elimination of initial zeros/identity-like solves, no tolerance or signed-zero
exception. Redzones and unchanged raw read sets checked after each group.

## ABI/provider provenance gate

`dlsym`+`dladdr` must bind each public entry to the hash-pinned original
libtorch_cpu.so and public ELF offsets vmsExp0x08ba7c10,
cblas_sgemm_batch0x08bd4a40, strsm_0x08bc9250. These are exported public entry
identities, **not internal kernel provenance**. Source parameter metadata and
profile guards are untouched. PyTorch BSD notice is retained alongside this
diagnostic. No Intel implementation code was copied or disassembled.

````

## Source: .agents/runs/model-execution-pinned-native-provider-probe-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 4997; SHA256 `af55727a6d0c1f3888564f18cd520fbe5e1ffaa09be1ccf20421704a9cfac88a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only Rust public BLAS/VML diagnostic

Parent: model-execution; owner `/root/baseline_review`. **PREPARED ONLY**: author
performed byte/header/ELF source inspection and Python syntax checks, no Rust/
Cargo/C/native/container/Torch/math/GPU execution. Production is untouched.

The smallest diagnostic calls genuine acquired public native entry points from
one standalone Rust process. There is no Python computation, foreign model,
forward/token/session loop, RPC, state publication, vendor arithmetic port or
learned matrix copy. This does not yet wire an operator provider into HostMath.

- `fixture-manifest.json` binds12 cases: four exp16 calls from actual source
  A_log/g at tokens11/353; six grouped CBLAS GEMM calls; two16-head STRSM groups.
  Byte-only copied payload12,845,568B, total diagnostic cap32MiB.
- Raw source archive `original-source-operators.safetensors` is pinned SHA
  65992c168c90762f48fee1523441e6154e41f15fd049e4320f4dea6499d9c423.
  The independent original public outputs are preserved, not reconstructed.
- Exact public signatures and shape/argument derivation: `ABI.md`.
- `static-closure-pins.json` pins14 candidate ELF dependency files from the exact
  original image's40-layer chain. This is static evidence, not a claim all these
  files are or are the only files mapped. Rust requires actual mapped-file
  paths/hashes match these candidates (plus its frozen binary); any unknown
  constructor-loaded DSO refuses. All candidates are hashed before dlopen.
- Original container GNU loader is the entrypoint with explicit library paths,
  no host `/nix/store` mounts. The binary's required GLIBC symbols must≤2.39;
  PT_INTERP/RUNPATH are recorded, never assumed compatible from filenames.
  Direct-loader `/proc/self/exe` points to the loader; the Rust artifact's mapped
  identity is separately checked at `/binary/public-abi-probe`.
- Exact image digest b18abb2d…; `--pull=never --network=none`, no GPU/device/driver
  mounts, readonly root/run/binary, private output only; twelveGiB memory/swap,
  four logical CPUs +fixed four threads,256pids,300s deadline,64MiB file bound,
  fourMiB continuously drained log. CPU32/FMA/ISA/library dispatch remains the
  genuine same-host runtime, not a claimed portable Rust schedule.
- Rust writes staged proc-map/status/thread/MXCSR and public-entry receipts,
  each actual output, full raw-word difference counts/first16 differences.
  Input redzones/read-set immutability and MXCSR control are checked. No DSO
  unload; foreign pools survive only until this owned process ends.
- Launcher verifies actual6 SGEMM_BATCH/32 STRSM verbose lines, transpose/flags,
  LP64/GNU-thread productbuild20240605, four threads/CNRoff/Dyn0/FastMM1.
  These are public dispatch observations; internal microkernel remains
  **UNOBSERVED**. VML has no assumed MKL_VERBOSE support; exact public entry,
  actual call count/mode and output bits are recorded in Rust.
- OMP worker/pool/internal native allocation is **not ResourceArbiter admitted**.
  Container limits/proc RSS are containment/observation only. Even a successful
  run leaves production memory admission/profile/full-model/serving/vLLM gates
  **UNQUALIFIED**. No library/source-profile promotion follows automatically.

Root may build offline in its serialized Cargo queue after reviewing source:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 cargo build --offline --locked --release \
  --manifest-path .agents/runs/model-execution-pinned-native-provider-probe-20261020/Cargo.toml \
  --target-dir .agents/runs/model-execution-pinned-native-provider-probe-20261020/target
```

The standalone lock reuses exact repository lock blocks. If Cargo requires a
feature/dependency lock normalization, root must inspect and retain that actual
offline change; do not enable fetching. Copy the resulting artifact to a stable
owned path and record its real SHA before launch; shared target path is mutable.

```sh
# ROOT ONLY, actual artifact/hash values required; new output directory only.
python3 .agents/runs/model-execution-pinned-native-provider-probe-20261020/run_probe.py \
  --run --binary /ABSOLUTE/OWNED/FROZEN/public-abi-probe \
  --binary-sha256 ACTUAL_64_LOWERCASE_HEX_SHA256 \
  --output .agents/runs/model-execution-pinned-native-provider-probe-20261020/root-actual-1
```

No SDK/headers/model download is needed for this prepared ABI slice. Source
pins, fixtures and static closure must be reviewed before root compiles/runs.
Invocation cleanup owns a unique nonce-label/name/CID and foreground PGID only;
no arbitrary existing container/process signals. Ownership begins immediately
after spawn, before receipt writes; cleanup/log drains precede final evidence
validation/I/O. Failure/truncation/unknown library/cleanup refusal stays
UNQUALIFIED; no automatic rerun or next provider. `prepared-manifest.json`
retains exact source/input hashes. Licence/redistribution closure is still the
source-only audit's open boundary; no library is redistributed here.

````

## Source: .agents/runs/model-execution-pinned-native-provider-probe-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1207; SHA256 `997f17f3483df1b379c0f3e3f9dfe9c71ab708142932937ad966912ad5037fd6`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native public-ABI diagnostic preparation

- Parent: `.agents/plans/model-execution/TODO.md`; owner `/root/baseline_review`.
- Own ignored run only. Root alone builds, loads libraries or executes probes.
- [x] Derive exact CBLAS grouped signature and STRSM11pointer/no-length call from pinned ATen primary source; VML count/mode from original header +Intel API.
- [x] Byte-copy authentic clean inputs/independent outputs; preserve full shape/stride/source/hash binding, no arithmetic expectations reconstructed.
- [x] Prepare one Rust public-operator diagnostic with bounded private aligned buffers, raw-word comparison and actual mapped closure/entry/read-set receipts.
- [x] Prepare exact GNU-loader/container launcher with static ELF version guard, fixed finite limits and owned cleanup before final evidence I/O.
- [x] Python AST/syntax only; no Cargo/Rust/native/container/model/runtime execution.
- [ ] Independent source/ABI/layout/cleanup review and root build/strict source checks.
- [ ] Root-only bounded actual native output/closure/dispatch qualification; no inference claim.
- [ ] Native allocation/admission/production packaging and same-engine typed wiring remain separate unqualified work.

````

## Source: .agents/runs/model-execution-pinned-native-provider-probe-20261020/frozen-reviewed/ABI.md

Original UTF-8 bytes: 5651; SHA256 `9898e1ad9e95de82d9510083d44843d7d89b9e25649f0b18eec1f0e614e92f39`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed public ABI — source derivation

Research/prepared diagnostic only. No vendor implementation-source derivation,
native build/load/call by author, model engine, state owner or profile promotion.

- Exact Torch source commit: `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Actual captured source .8B, layer0, source36-F32-restored, tokens11/353;
  fixture metadata retains original byte ranges, source names/strides and hashes.
- Expected output files contain independent original public-Torch outputs. Rust
  first opens expected-output bytes **after** native completion. Provider calls
  receive only captured inputs and private initialized/poisoned destinations.

## Three public entries

1. `vmsExp`: Intel's [2024.2 exp API](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/v-exp.html)
   specifies LP64 `MKL_INT` count and explicit `MKL_INT64` mode. Installed pinned
   `ATen/cpu/vml.h` invokes the mode-aware FP32 function with HA|FTZDAZ_OFF|
   ERRMODE_IGNORE, value0x140102. Probe signature uses int32 count by value,
   FP32 input/output pointers and int64 mode by value. No default global VML mode.
   Original [UnaryOpsKernel.cpp iterator](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp#L724)
   gathers noncontiguous spans to a contiguous local buffer. Probe reconstructs
   actual source view, gathers16 original words without arithmetic and calls once.
   Actual internal caller alignment/selected VML microkernel is unobserved; this
   diagnostic can establish only output parity for these four captured calls.
2. **`cblas_sgemm_batch`**, not direct Fortran `sgemm_batch_`: pinned
   [ATen mkl/LinearAlgebra.cpp66–74](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/mkl/LinearAlgebra.cpp#L66)
   supplies column-major layout, by-value group count1 and group-size16 pointer;
   matrices/scalars/dimensions/leading dimensions are per-group arrays. Official
   [C interface enum values](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/c-interface-conventions-for-blas-routines.html)
   establish ColMajor102, NoTrans111, Trans112, and
   [grouped C API](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-gemm-batch.html)
   supplies the exact signature. No hidden character lengths exist in this C API.
   Actual MKL_VERBOSE calls are named SGEMM_BATCH; that name alone does not
   identify the original public wrapper. Public dynamic symbol is exported.
3. `strsm_`: pinned [ATen declaration452](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/BatchLinearAlgebra.cpp#L452)
   explicitly gives11 pointer arguments: four char*, two int*, float*alpha,
   float*A, int*lda, float*B, int*ldb. Its
   [FP32 call1436–1438](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/BatchLinearAlgebra.cpp#L1436)
   passes exactly those11 and alpha1. No hidden string lengths are inferred or
   added. [Kernel716–740](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp#L716)
   loops heads with matrix strides and scalar flags. Existing source capture
   independently observes all32 STRSM(L,U,T,U,64,128,...64,...64) calls.

LP64 int32 is independently established by actual source MKL_VERBOSE;
[official integer contract](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/using-the-ilp64-interface-vs-lp64-interface.html)
keeps VML mode64-bit even with32-bit dimensions. Exact primary callers and public
ABI docs suffice for these three diagnostic signatures; no missing SDK header is
guessed. No thread-service/allocator/internal MKL ABI is used.

## Captured matrix transport

All six families batch16, alpha1/beta0. Original row-major output means ATen swaps
logical operands: native A=logical RHS, B=logical LHS. Buffer reconstruction uses
the actual observed full shape/stride; C holds the original row-major output.
Original source-independent layout analysis in the parent batch agent's ignored
`p12-hf-ordinary-rms-20261020/.../INITIAL_CHUNK_AUDIT.md` is corroboration; actual
report metadata and pinned original source are the binding evidence.

| Family | transA/B | M,N,K | lda/ldb/ldc |
|---|---|---|---|
| ut | T,N |64,64,128|128/128/64|
| intra | T,N |64,64,128|128/128/64|
| old_read | N,T |128,64,128|128/64/128|
| inter | N,N |128,64,128|128/128/128|
| intra_values | T,N |128,64,64|64/64/128|
| state_update | T,T |128,128,64|64/128/128|

Solves each use16 heads. A remains original row-major raw storage, interpreted
as column-major upper + transpose. B is copied from captured RHS logical words
to the private original output's column-major storage before the in-place call.
The copied bytes are RHS **inputs**, never expected results. No algebraic
elimination of initial zeros/identity-like solves, no tolerance or signed-zero
exception. Redzones and unchanged raw read sets checked after each group.

## ABI/provider provenance gate

`dlsym`+`dladdr` must bind each public entry to the hash-pinned original
libtorch_cpu.so and public ELF offsets vmsExp0x08ba7c10,
cblas_sgemm_batch0x08bd4a40, strsm_0x08bc9250. These are exported public entry
identities, **not internal kernel provenance**. Source parameter metadata and
profile guards are untouched. PyTorch BSD notice is retained alongside this
diagnostic. No Intel implementation code was copied or disassembled.

````

## Source: .agents/runs/model-execution-pinned-native-provider-probe-20261020/frozen-reviewed/README.md

Original UTF-8 bytes: 5379; SHA256 `6f25c7445955450bcbe51f49acd6ccc33a8939ba0c37bc4318eb8417248efef3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only Rust public BLAS/VML diagnostic

Parent: model-execution; owner `/root/baseline_review`. **PREPARED ONLY**: author
performed byte/header/ELF source inspection and Python syntax checks, no Rust/
Cargo/C/native/container/Torch/math/GPU execution. Production is untouched.

The smallest diagnostic calls genuine acquired public native entry points from
one standalone Rust process. There is no Python computation, foreign model,
forward/token/session loop, RPC, state publication, vendor arithmetic port or
learned matrix copy. This does not yet wire an operator provider into HostMath.

- `fixture-manifest.json` binds12 cases: four exp16 calls from actual source
  A_log/g at tokens11/353; six grouped CBLAS GEMM calls; two16-head STRSM groups.
  Byte-only copied payload12,845,568B, total diagnostic cap32MiB.
- Raw source archive `original-source-operators.safetensors` is pinned SHA
  65992c168c90762f48fee1523441e6154e41f15fd049e4320f4dea6499d9c423.
  The independent original public outputs are preserved, not reconstructed.
- Exact public signatures and shape/argument derivation: `ABI.md`.
- `static-closure-pins.json` pins14 candidate ELF dependency files from the exact
  original image's40-layer chain. This is static evidence, not a claim all these
  files are or are the only files mapped. Rust requires actual mapped-file
  paths/hashes match these candidates (plus its frozen binary); any unknown
  constructor-loaded DSO refuses. All candidates are hashed before dlopen.
- Original container GNU loader is the entrypoint with explicit library paths,
  no host `/nix/store` mounts. The binary's required GLIBC symbols must≤2.39;
  PT_INTERP/RUNPATH are recorded, never assumed compatible from filenames.
  Direct-loader `/proc/self/exe` points to the loader; the Rust artifact's mapped
  identity is separately checked at `/binary/public-abi-probe`.
- Exact image digest b18abb2d…; `--pull=never --network=none`, no GPU/device/driver
  mounts, readonly root/run/binary, private output only; twelveGiB memory/swap,
  four logical CPUs +fixed four threads,256pids,300s deadline,64MiB file bound,
  fourMiB continuously drained log. CPU32/FMA/ISA/library dispatch remains the
  genuine same-host runtime, not a claimed portable Rust schedule.
- Rust writes staged proc-map/status/thread/MXCSR and public-entry receipts,
  each actual output, full raw-word difference counts/first16 differences.
  Input redzones/read-set immutability and MXCSR control are checked. No DSO
  unload; foreign pools survive only until this owned process ends.
- Launcher verifies actual6 SGEMM_BATCH/32 STRSM verbose lines, transpose/flags,
  LP64/GNU-thread productbuild20240605, four threads/CNRoff/Dyn0/FastMM1.
  These are public dispatch observations; internal microkernel remains
  **UNOBSERVED**. VML has no assumed MKL_VERBOSE support; exact public entry,
  actual call count/mode and output bits are recorded in Rust.
- OMP worker/pool/internal native allocation is **not ResourceArbiter admitted**.
  Container limits/proc RSS are containment/observation only. Even a successful
  run leaves production memory admission/profile/full-model/serving/vLLM gates
  **UNQUALIFIED**. No library/source-profile promotion follows automatically.

Root may build offline in its serialized Cargo queue after reviewing source:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 cargo build --offline --locked --release \
  --manifest-path .agents/runs/model-execution-pinned-native-provider-probe-20261020/Cargo.toml \
  --target-dir .agents/runs/model-execution-pinned-native-provider-probe-20261020/target
```

The standalone lock reuses exact repository lock blocks. If Cargo requires a
feature/dependency lock normalization, root must inspect and retain that actual
offline change; do not enable fetching. Copy the resulting artifact to a stable
owned path and record its real SHA before launch; shared target path is mutable.

```sh
# ROOT ONLY, actual artifact/hash values required; new output directory only.
python3 .agents/runs/model-execution-pinned-native-provider-probe-20261020/run_probe.py \
  --run --binary /ABSOLUTE/OWNED/FROZEN/public-abi-probe \
  --binary-sha256 ACTUAL_64_LOWERCASE_HEX_SHA256 \
  --output .agents/runs/model-execution-pinned-native-provider-probe-20261020/root-actual-1
```

No SDK/headers/model download is needed for this prepared ABI slice. Source
pins, fixtures and static closure must be reviewed before root compiles/runs.
Invocation cleanup owns a unique nonce-label/name/CID and foreground PGID only;
no arbitrary existing container/process signals. Ownership begins immediately
after spawn, before receipt writes; cleanup/log drains precede final evidence
validation/I/O. Failure/truncation/unknown library/cleanup refusal stays
UNQUALIFIED; no automatic rerun or next provider. `prepared-manifest.json`
retains exact source/input hashes. Licence/redistribution closure is still the
source-only audit's open boundary; no library is redistributed here.

Root review follow-up preserves initial `frozen-prepared/` bytes: explicit
actual/expected output-length equality before comparison, and a second
nonce/name/CID-bound container cleanup/absence check after foreground CLI reap.
An empty CID file during owned startup uses the independently nonce-bound name;
nonempty malformed/foreign ownership is refused without container signals.

````

## Source: .agents/runs/model-execution-pinned-native-provider-probe-20261020/frozen-reviewed/TODO.md

Original UTF-8 bytes: 1402; SHA256 `078813c4323bbdc1b8329314dd6a502f11fecce46e3e42f93429b9e3d802b742`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native public-ABI diagnostic preparation

- Parent: `.agents/plans/model-execution/TODO.md`; owner `/root/baseline_review`.
- Own ignored run only. Root alone builds, loads libraries or executes probes.
- [x] Derive exact CBLAS grouped signature and STRSM11pointer/no-length call from pinned ATen primary source; VML count/mode from original header +Intel API.
- [x] Byte-copy authentic clean inputs/independent outputs; preserve full shape/stride/source/hash binding, no arithmetic expectations reconstructed.
- [x] Prepare one Rust public-operator diagnostic with bounded private aligned buffers, raw-word comparison and actual mapped closure/entry/read-set receipts.
- [x] Prepare exact GNU-loader/container launcher with static ELF version guard, fixed finite limits and owned cleanup before final evidence I/O.
- [x] Python AST/syntax only; no Cargo/Rust/native/container/model/runtime execution.
- [x] Root review repairs: exact output-length equality and post-CLI-reap independent owned-container cleanup/absence; empty startup CID handled by nonce-bound name. First frozen bytes retained.
- [ ] Independent source/ABI/layout/cleanup review and root build/strict source checks.
- [ ] Root-only bounded actual native output/closure/dispatch qualification; no inference claim.
- [ ] Native allocation/admission/production packaging and same-engine typed wiring remain separate unqualified work.

````

## Source: .agents/runs/model-execution-qwen38-arithmetic-2026-10-18/TODO.md

Original UTF-8 bytes: 1736; SHA256 `d68b0b9d8d184748b8456a0e39b5b86c443c6a093cbcb61a357551bc9471279d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Qwen original analytic numerical layer — current agent

Provenance: b796a2e
Plan: [model-execution](../../plans/model-execution/TODO.md)
Scope: original 27B hybrid layer GDN/full-attention/FFN source-based
analytical CPU operators only. No trained weights, GPU, SSD or new
model/session authority.

- [x] Reuse original Qwen verified-index/source/tensor binder and
  existing Rust Qwen GGUF mathematical primitives.
- [x] HF full Q/gate per-head ordering, 24Q/4KV mapping, source
  partial RoPE, QK/softmax/AV/sigmoid/out proposal in source layer.
- [x] GDN Q/K 16→48 repeat, four-tap conv, L2 normalization,
  original -exp(A_log) and source GDN coeff/recurrent/output; all
  state remains independently source/layer-labelled and tentative.
- [x] FFN zero-centered norm, SwiGLU residual, dimensional/finite
  refusal, tiny independent analytical numerical tests.
- [x] Strict lint default/composed CPU/cache/Qwen/acquisition
  `--all-targets -- -D warnings`, full frozen
  `scripts/check.sh` 1,111 Rust passes/0 failures/69 ignored
  across 238 suites, 18 Python fixtures and actual offline Nix source
  integrity passed on final source.
- [x] Original Qwen3.5 .8B/2B trained reference release direct-cache
  and HF-import parity independently passed on final source:
  complete 248320 logits, all42 StateRoot domains, 2 tokens,
  independent sessions and mutation/reclamation exact.
  Direct 17.36s; HF import 22.49s.
- [x] Signed scoped local commit `b11e58c`, exactly five
  reviewed source/tests/plan paths; no push or model download.
  Post-commit source integrity is separately verified.
- [ ] Deferred: original trained BF16 rounding/FLA GDN, complete
  StateRoot/task model run, official tokenizer, SSD and physical GPU.

````

## Source: .agents/runs/model-execution-qwen38-bf16-2026-10-18/TODO.md

Original UTF-8 bytes: 1020; SHA256 `4abe7f75fdfea9c61821b98ff1034f9c39c4a860aecc90c01164fec5997bb0d9`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ME-P0 original Qwen BF16 source dtype diagnostics

Base: 7e32841
Parent: [model execution](../../plans/model-execution/TODO.md)

- [x] IEEE BF16 RNE finite source conversion, signed zero,
  tie-even and overflow guards. No new deps/models.
- [x] BF16 source RMS, projections, GDN/attention/FFN stages
  and distinct session state schemas, FP32 baseline unchanged.
- [x] Independent NumPy analytical BF16 goldens for ordinary
  and gated RMS; two-step FP32 state/BF16 GDN recurrent core.
- [x] Full synthetic 4-layer/two-token BF16 Qwen, mode/refusal.
- [x] Complete isolated check.sh 1128/0, 69 ignored,
  238 Rust suites plus 18 Python fixtures; default/composed
  strict Clippy, Nix precommit source integrity pass.
- [x] Original trained Qwen3.5 0.8B/2B direct cache and HF
  import release parity pass: exact full logits/all42 state domains,
  two tokens/source mutation/lease/reclamation.
- [ ] Scoped signed commit (no push/download/device).
- [ ] Deferred: source-real trained HF/FLA BF16 and physical GPU/SSD.

````

## Source: .agents/runs/model-execution-qwen38-closed-domains-audit-20261020/AUDIT.md

Original UTF-8 bytes: 8241; SHA256 `8d348750bf8e31aae429521d263fbff6fd281dff728d8aa421f3276b7768af05`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original 27B: concrete closed-domain gaps

Author: baseline_review. Source-only; no operator/provider/model/build/test execution or payload hashing. No shared-plan changes.

## Authentic geometry

- Repository `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; snapshot `98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e`.
- Read only config/index and bounded Safetensors headers from the existing snapshot. Actual index/header closure: **1199 tensors, 851 text tensors, all851 BF16**. Existing acquisition/owned-window receipts authenticate complete payloads; this audit did not rescan them.
- 64 layers: 48 GDN,16 full attention. Hidden5120; MLP17408; vocab248320; head256; rotary64; trained context262144. Embedding and output head are separate (`tie_word_embeddings=false`).

| Original role | Raw [output,input] / parameter shape | Count |
|---|---|---:|
| Embedding / separate lm_head | [248320,5120] each | 1+1 |
| Input/post-attention norm | [5120] | 64+64 |
| Final ordinary norm | [5120] | 1 |
| GDN QKV | [10240,5120] | 48 |
| GDN Z | [6144,5120] | 48 |
| GDN alpha / beta | [48,5120] each | 48+48 |
| GDN output | [5120,6144] | 48 |
| GDN conv | [10240,1,4] | 48 |
| GDN A_log / dt_bias | [48] each | 48+48 |
| GDN norm gamma | [128] | 48 |
| FA Q + output gate | [12288,5120] | 16 |
| FA K / V | [1024,5120] each | 16+16 |
| FA output | [5120,6144] | 16 |
| FA Q / K norm | [256] each | 16+16 |
| MLP gate / up | [17408,5120] each | 64+64 |
| MLP down | [5120,17408] | 64 |

GDN original Q/K16×128 and V48×128; repeat Q/K heads3× AFTER convolution/coefficient operations. FA Q24×256 and KV4×256; repeat KV6× for attention. Source matrix axes remain OutputInput.

## Actual blockers and minimum cuts

1. **Original dtype contracts.** Pending coefficient V2 already accepts BF16/F32 dt_bias and heads16/48 (`followup-v2/proposed/.../program/coefficient_preparation.rs:20`, CPU child57/72). Only its *unevaluated* A_log descriptor requires F32/4H bytes (`operator_slice/coefficient_preparation.rs:63`). Smallest future metadata extension: permit raw BF1648/96B with exact same issuer/name/shape; preserve actual `.float()` BEFORE public exp. No new exp eligibility. Current `original_hf/norm.rs:65` accepts only F32 gamma. Pinned modeling230 multiplies raw gamma by normalized BF16: BF16 gamma causes an additional BF16 product rounding before FP32 SiLU gate; simply widening gamma and reusing the F32-product branch changes math. Add an explicit source-kind branch plus genuine27B public-stage oracle; retain old F32 branch unchanged.

2. **Closed window role coverage and output admission.** `original_binding/window.rs:264` permits only GdnQkv(0) and FA Q/K/V projections; normalize permits InputNorm(0) and FA Q/K norm only. Other GDN layers, Z/A/B/output, all MLP, post/final norms and separate lm_head currently refuse. Extend only the next source-role family through SAME resolver/PlanBuilder/HostMath after operator proof. `workspace()`126 reserves Output64KiB+Scratch960KiB for the existing small fragments. MLP17408 alone needs69632B of retained F32 backing; full vocab248320 needs993280B, plus private raw/final arrays/scratch. Derive actual aggregate reservations from descriptor/operation BEFORE allocation; do not only enlarge role allowlists.

3. **Projection bounds are backend-specific.** CPU `weights.rs:659` uses all ordered64-column blocks; CPU window wrapper has no8192 column cap and full-row limit1,000,000 (vocab248320 fits). CUDA CPU-profile helper `cpu_bf16_projection.rs:39` has transported-range rows≤16384, full columns≤8192, weight range≤32MiB. Its full descriptor may retain >16384 rows (`Bf16Window::bind`139); output-row windows can address gate/up17408 and full vocabulary without changing the dot. **They cannot fix down-projection width17408**. A narrow CUDA cap extension must preserve the full17408-column64-lane reduction (272 ordered blocks), source identity, checked products, exact profile and actual trained oracle. CPU widths6144/17408 also require actual source/operator qualification beyond the accepted5120 QKV fragment. With128-row transport, widest row window is4,456,448B, within32MiB data cap; full-column splitting/reassociation is forbidden.

4. **GDN schedule and missing operators.** Current full `forward.rs:287` assumes equal Q/K/V width6144 and requires18432 QKV values; authentic27B QKV has10240. Closed original path needs source key/value geometry, split2048/2048/6144 then head3× replication. Existing conv channel10240≤16384 and K4 source BF16 fit its contract; no channel-cap increase needed. Cached H48 state786432F32 words fits current≤1,048,576 reducer/core bound; head norm48≤64 also fits. These are domain compatibility facts, not trained27B proofs. Initial padded chunk64 remains a distinct missing FLA operator; it cannot be replaced by cached recurrence from zero. Public exp/decay and SGEMM_BATCH/STRSM arithmetic proof is not physical provider admission: `root-actual-3/rust-report.json` explicitly leaves native internal memory UNQUALIFIED. Preserve that gate.

5. **Attention, residual/MLP and readout remain incomplete.** Current typed attention slice stops after Q/K/V and QK norms;27B dimensions fit (Q+gate12288≤16384, KV1024, width256, hidden5120). It does not establish rotary, KV cache, BF16 QK/AV BMM, softmax, sigmoid output gate, output projection or residual. Softmax closed helper maxwidth512 supports an initial context512 gate, not trained262144. MLP requires exact BF16 SiLU/product/down and BF16 residual materialization in the same schedule, rather than GGUF F32 math. Unary≤1M bounds fit B1 widths17408; a simultaneous64-token MLP span1,114,112 exceeds1M, so a future bulk path needs separately reviewed bounded elementwise row partitioning (and independently qualified GEMM rather than B1 GEMV substitution).

6. **Source/session/state integration is still intentionally closed.** Existing complete SourceWeights4GiB materialization cannot load55,563,006,776B. Sealed windows already verify source with explicit64GiB limit; default window4GiB must be overridden, not confused with admission to materialize55GB. Actual same-engine27B three-node40960-word PASS proves only embedding→InputNorm0→QKV0. Full traversal must retain this source owner and use existing SessionAuthority/Ready/publication, not fake a LoadedQwenModel/TensorSource from partial windows. Original full execution and checkpoint guards must remain until all operators compose exactly. Canonical state is **112 domains**: 48 conv+48 recurrent+16 KV. At context512, source geometry gives5,898,240B conv+150,994,944B recurrent+67,108,864B KV =224,002,048B; per recurrent domain3MiB. Current1GiB CPU session cap therefore fits512 but rejects8192 (1GiB KV plus149.625MiB fixed state). Trained262144 would require32GiB KV plus fixed state; no silent cap increase or full-context claim. Existing configured source-window host ledger does not itself prove physical RAM admission.

## Ordered next work

1. Finish current coefficient/cached physical gates for qualified small models; separately admit authentic27B BF16 A_log metadata and dtype-correct gated gamma branch with actual H48 public references.
2. Add same-resolver per-layer GDN Z/A/B/output and raw post/final norm roles with operation-sized Output/Scratch leases. Qualify full-column5120/6144/17408 CPU dots and MLP17408 source stages; only then allow their typed append operations.
3. Admit exact public exp/FLA primitives physically and compose separate initial/cached GDN through existing generic layer schedule, using original K16→V48 replication and canonical↔key-major state transport.
4. Complete typed FA rotary/cache/BMM/softmax/gate, BF16 residual/MLP and untied full-vocab readout. Begin bounded context512; each layer/stage/all112 domains/fullV against authentic source before any full-session permit.
5. Extend CUDA full-column17408 only as an explicit same-profile closed kernel gate. All-source residency on6GiB is impossible; production streamed weight leasing/completion must be qualified separately from this CPU window path.

No metadata-only/profile-wide enablement, no default/GGUF change, no benchmark/performance claim.

````

## Source: .agents/runs/model-execution-qwen38-gdn-next-cut-audit-20261020/NEXT.md

Original UTF-8 bytes: 3728; SHA256 `a63bff6fa572e76ba51b3398fcf45e00e95e03a77b7b924a9157afb2a79b776d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Smallest next27B GDN cuts

Owner baseline_review; parent model-execution. Source/metadata audit only.

**Next implement exact initial query MUL +width64 prefix sum in a new CPU
original_hf child. In parallel, close ONE admitted native EXP call through the
existing provider-startup boundary. Neither requires a new model/token engine.**

| Disjoint cut | Exact existing seam | Actual prerequisite / result |
|---|---|---|
|1. Initial scale/prefix|CPU `original_hf/reduction.rs` reused; new small child|H48×128 source L2 output ×rounded0x3db504f3, then padded decay64 F64 carry/+0 with F32 store each prefix. Cached directDIVsqrt128 unchanged. V3 already supplies genuine source intermediates for ROOT exact replay.|
|2. Source metadata +bit transport|Pending coefficientV2 `operator_slice/coefficient_preparation.rs:58–70`; existing SourceParameter/window machinery|dt already permitsBF16/F32. ONLY unevaluated A_log requires separate rawBF16[48]/96B acceptance. Add explicit Q/K BF16 head-repeat16→48; single TensorView row stride cannot represent repeat-interleave. No exp/profile permit.|
|3. Live coefficients|Existing accepted sigmoid/softplus; future narrow typed public native-exp callback beneath SAME MathematicalBackend|Source order betaBF16→A_logBF16→F32 exp/neg→aF32+rawdt/softplus→separateF32g→exp(g). Native actual3 proves EXP16 arithmetic only; startup6 proves ownership with providerUNLOADED. Physical provider/runtime/pool/thread lifetime is still open. First actual EXP16 admission, then authenticEXP48; never substitute SLEEF/libm.|
|4. Initial chunk64 core|New closed typed core in SAME HostMath; no StateEffect initially|True one-token pad64, triangular mask -Inf, six BMM families +two48-head unit-lower solve groups (96STRSM), exact row/column-major layouts and materialized products/adds. Requires admitted native exp/BLAS +H48 qualification; old16-head ABI receipt is insufficient.|
|5. Layer0 composition|Existing forward/operator_slice/HostMath +OriginalWindowOperators851-role source|Genuine QKV/Z/a/b source producers→conv/SiLU→repeat→coefficients→initial/cached core→distinct BF16-gamma norm→out projection. Pass private HostValues directly; preserve all reservations/source issuer. Return private nextconv/state, then compare complete initial→cached V3 pair. No full-model/root enablement.|

Three concrete traps:

- Cached H48 native PASS imports observed exp(g) and already-repeated Q/K; it
  does not close those producers. Its initial state may be zero without making
  it the source chunk64 schedule.
- Ordinary typed F32 rejects Inf; initial strict-upper -Inf mask is intentional.
  Keep it inside a dedicated closed masked-exp/chunk contract. Do not weaken
  global finite validation. Preserve all zero/identity-like native operations.
- Recurrent source is `[48,key128,value128]`, canonical state is value-major;
  original conv cacheK4 differs from canonical newest3. Metadata is not a live
  state codec/currentness permit. Full HF/session/checkpoint guards stay CLOSED.

Actual receipts distinguish native cached+gamma PASS from source/public-only
wide/coefficient targets. Layer0 raw projections fit oldK<=8192/M<=16384;
17408-column FFN-down is later full-model work. New native wide-role/27B
conv/coef producer replays remain required where no receipt exists.

Details +exact metadata/source paths: SOURCE-MAP.md and EXISTING_METADATA.json.
The prefix schedule cites the retained pinned primary-source audit; reread its
two cumsum/accumulator source files before implementation (scope note in map).
No builds/tests/operators/providers/models/GPU, payload/DSO scans/hashes or
MAIN/shared-plan changes by this audit. ROOT owns implementation/execution.

````

## Source: .agents/runs/model-execution-qwen38-gdn-next-cut-audit-20261020/SOURCE-MAP.md

Original UTF-8 bytes: 12434; SHA256 `b0d2a9037fffb4261994058f9dba6010bdff8f2bbb643047adc510730ca19d52`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B layer0 GDN: minimum remaining cuts

Author: baseline_review. Parent: model-execution. Source/metadata audit only.
No MAIN/shared-plan/frozen-source changes, build/tests/imported provider/model/
operator/GPU execution, source payload or DSO/binary scans/hashes.

## What is actually available

- Current model-execution TODO12–33 and actual `model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/validation.json`: genuine original27B@1d4bf0f… layer0 initial chunk64 +true cached token, teachers11/353. Raw source is ALL BF16. This is a public target, not native whole-GDN execution.
- Actual H48 cached private slice (`model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/root-native-1/validation.json`):804960 imported +798720 output words exact, including all786432 recurrent state words. It imports already-repeated Q/K and independently observed exp(g). It proves neither those producers nor initial chunk64.
- Actual BF16-gamma gated RMS private slice (`model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/root-native-1/validation.json`):61632 stage +12288 typed final +24576 retained input words. Keep its BF16 weighted-product cast; former raw-F32-gamma promotion differs.
- Wide-role target has32 public reference cases/881856 words,16 roles; its actual validation explicitly leaves native CPU parity and trained producers for all roles false. No native wide-role receipt was found in that qualification directory. CoefficientV2 actual public target covers4 calls, heads16; its native/H48 live-producer qualification is not implied by that target. These distinctions survive source acceptance.
- Native actual3 (`model-execution-pinned-native-provider-probe-20261020/root-actual-3/{rust-report,invocation}.json`):4EXP16 +6 batch-GEMMs +32 STRSM,1,048,640 words0diff on original .8B operands. Provider-internal memory/admission, portable implementation and full profile all explicitly UNQUALIFIED. Startup6 actual cases (`model-execution-native-cpu-provider-envelope-20261020/startup-v1/root-native-v3-attempt-1/validation.json`) pass with PROVIDER UNLOADED. Neither is an admitted math provider.

## Exact layer0 geometry and source

V3 captured manifest `parameter_audit`, `steps[].tensors` (metadata read only):

| Role | Raw BF16 shape | Bytes |
|---|---|---:|
| QKV |[10240,5120]|104857600|
| Z |[6144,5120]|62914560|
| a,b each |[48,5120]|491520|
| output |[5120,6144]|62914560|
| conv |[10240,1,4]|81920|
| A_log,dt_bias each |[48]|96|
| gated gamma |[128]|256|

Split Q/K/V widths2048/2048/6144, then repeat each key head3×:16→48, width128. Recurrent math state is FP32 `[48,key128,value128]` (786432 words); existing canonical transport is `[head,value,key]`. Original conv cache is BF16 `[10240,4]`; current canonical history is exact newest3 `[10240,3]`, oldest→newest. Do not pretend either layout is the other or reconstruct discarded original cache entries as zeros.

All layer0 projection K/M dimensions fit the existing8192-column/16384-row CPU64 bounds; the17408-column FFN-down issue is later full-model work, not a layer0 blocker. Conv10240 fits its<=16384/multiple32 contract, but geometry fit alone does not qualify its27B original layout/SiLU route.

## Reuse and missing seams

| Existing seam | Reuse | Missing minimal addition |
|---|---|---|
| `forward.rs:embedding_input_norm_projection_slice`, `original_binding/window.rs:project`, CPU `window_projection.rs` | Same sealed851-role source, full-K row windows, CPU64 BF16 output, raw InputNorm5120 | Add genuine GDN Z/a/b/out role calls in ONE HostMath fragment, retaining actual producers/leases. Run the existing pure-role native gate; external role probes are not trained producers. |
| `gdn_conv_slice.rs`, `host_math/gdn_conv.rs`, core `program/gdn_conv.rs` | Existing original conv4→BF16→SiLU→BF16 +private next3, raw BF16 parameter window | Qualify27B initial4-frame/cached-last-frame source routes; connect actual QKV HostValue and canonical3 history, rather than re-import observed bytes. |
| coefficientV2 `operator_slice/coefficient_preparation.rs:58–70` | beta BF16 +a.float()+raw dt→FP32 softplus, one call heads48 already representable | The UNEVALUATED A_log descriptor is F32-only. Admit genuine `(BF16,96)` in a separate exact role/codec gate; dt already supports BF16/F32. Actual A_log evaluation must explicitly widen raw BF16 toF32, never retag bytes. |
| CPU `original_hf/reduction::{l2_norm_128,sum_key_128}` and `recurrent::cached_step_128`; cached typed core+return nodes | Cached direct query DIVsqrt128, materialized products/key reduction/separate update, H48 within1M state bound | Add source-sealed BF16 Q/K repeat16→48 producer +live FP32 exp(g) producer. A single TensorView row stride cannot express repeat-interleave; use a checked bit-copy operator, not normalization-before-repeat reassociation. |
| BF16-gamma typed gated RMS | Existing exact BF16 weighted product, SiLU F32 product, finalBF16; SourceParameter raw gamma | Connect cached/initial BF16 core and genuine Z HostValues directly, then existing BF16 out projection. Keep raw-F32 gamma path separate. |
| `OriginalStateDescriptor` / `SessionAuthority` | Source/phase/layout metadata and original publication owner | Pure layer0 can return private next-state values. Metadata does not prove currentness; full-model effects/phase/current root/codec/version must be qualified before registering112-domain execution. |

Current `original_binding.rs:require_execution_profile` and typed nodes' operator-slice-only guards MUST remain. `crates/zerogravity/src/original_qwen_math.rs:propose_gdn` is the unqualified diagnostic, using legacy grouped RMS/conv/host exp/recurrence; it is not a shortcut around the closed helpers. No new model/token loop or partial TensorSource/LoadedQwenModel is needed.

## Missing exact native-exp cut

Pinned modeling source762feb6c… line617/619:

1. `beta=b.sigmoid()` returns BF16. Preserve one48-element call: vector32 prefix +scalar16 tail; merging teachers changes iterator schedules.
2. Raw BF16 A_log[48]→FP32 cast→`Tensor.exp`→unary negation.
3. BF16 a→FP32, add original BF16 dt_bias after promotion→FP32 softplus.
4. Separate FP32 multiply gives g; cached source line483 separately computes exp(g).

Pinned Torchcf30153 local `UnaryOpsKernel.cpp:722–764` +`vml.h:109–135`: FP32 Tensor.exp calls MKL `vmsExp`, LP64 int32 count/int64 mode0x140102. Iterator grain2048; noncontiguous input gathers in an8KiB local buffer. Existing SLEEF exp inside sigmoid/softplus/SiLU is a DIFFERENT closed operator, not a standalone Tensor.exp implementation. Do not use Rust `.exp()`, host libm or a recorded-output lookup.

Schedule seam: coefficientV2 exposes distinct beta/alpha recorder nodes but its HostMath callback computes both preparation sides together. For complete source-order composition, keep beta→A_log cast/exp/neg→a cast/add/softplus→g. Factor/reuse the unchanged accepted preparation stages or introduce a closed complete coefficient operator; do not infer that the standalone preparation callback already contains that missing interleaving.

Shortest provider gate: extend the existing startup/service admission boundary to ONE synchronous actual EXP16, then authentic27B EXP48. Before native load/call: retain original source/raw window, admitted caller input/output and persistent provider owner; prove the real domain/envelope, mapped closure/constructors/OpenMP threads/pools/control lifetime and fixed runtime/FP environment. Startup currently denies new threads; native4-thread use is a new policy/lifetime gate. Unknown completion must keep owners/fence, cancellation discards only after known return. Same Rust executable/schedule inside the pinned GNU-loader closure remains valid; no Python/foreign model/session/RPC engine is needed. Existing4GiB diagnostic envelope/AS/cgroup observations do not themselves close physical admission.

## Initial chunk64 is a distinct operator

Pinned modeling source300–435 and retained local `p12-hf-ordinary-rms-20261020/.../INITIAL_CHUNK_AUDIT.md`:

Source verification limit: this pass reread the local modeling source and prior
audit; the latter pins `ReduceOpsKernel.cpp:72` and `AccumulateType.h:146` at
cf30153 for F64 cumsum. Those two primary files were not locally available in
the scoped audit directories or freshly reread here. Pin/read them before
implementing the prefix helper, then compare the genuine V3 intermediates.

- Normalize widened Q/K via existing source L2; initial query MULTIPLIES rounded `128**-0.5` (`0x3db504f3`), whereas cached query DIVIDES by rounded sqrt128. Preserve both contracts.
- Pad one valid token to64 with source +0, materialize beta*K/V; F32 decay cumsum uses an F64 carry from +0 and F32 store after EACH prefix. A first-prefix copy or final-only cast changes the source schedule.
- Prefix differences→strict upper mask -Inf→exp, two Gram BMMs +separate pairwise multiply, exp(prefix)*betaK, two unit-lower solves. Keep actual Gram diagonal; TRSM unitdiagonal flag handles it.
- Then decayedQ/K/chunk-decay; six total BMM families and two solve groups with materialized subtraction/product/add boundaries. Initial state is true source zero but ZERO/identity-like operators cannot be algebraically removed from the target.
- Solve outputs/newV retain column-major64×128 storage. Returning core slices ONE valid row then castsBF16; state remains FP32 key-major. Existing cached recurrence on zero state is not initial chunk64.

Authentic27B exp metadata: A_log48 contiguous; pairwise196608 contiguous; prefix3072 contiguous at source386/413; last-minus-prefix3072 contiguous414; lastprefix48 stride64 at415. Cached483 is48 contiguous. Actual3 qualifies only four16-word exp calls, not these shapes/call partitions/masked -Inf values. V3 has exact operands/results for ROOT to replay; no need recapture a full55GB model.

The six old ABI BMM layouts/leading dimensions remain per-head128/64, but group-size becomes48, not16. Two solves mean96 public STRSM calls. Each largest H48 output786432 words fits the existing1M per-array bound; combined live scratch must be newly calculated/admitted, not hidden behind that per-array limit. Actual16-head ABI parity is not48-head parity.

Important type trap: ordinary typed F32 validates FINITE values; pairwise -Inf mask is intentional. Keep masked exp operands inside a dedicated closed chunk/mask contract (or explicit masked domain), validating -Inf ONLY at proven strict-upper indices. Never weaken general F32 finite validation or reuse unrelated attention softmax arithmetic.

## Ordered next implementation recommendation

1. **Disjoint, no-native-provider arithmetic:** new CPU original_hf child for exact initial query MUL and width64 F64-carry/F32-store prefix sum. Reuse existing L2; preserve cached DIV untouched. ROOT compare genuine H48 source V3 stages +signedzero/subnormal/cancellation adversaries before typing/composition. This is the smallest independently closable numerical cut while provider admission remains open.
2. **Disjoint metadata/bit transport:** narrowly admit unused raw BF16 A_log descriptor +new BF16 repeat16→48, same issuer/role/full geometry, pre-admitted backing and cancellation. Qualify existing coefficient prep on actual27B a/b/dt targets and native wide role producer outputs. No exp/profile permit follows.
3. **Admitted public native exp** as above; then seal live `A_log→negativeExp→g→decay` as private per-op contract under existing MathematicalBackend. Cached H48 core→BF16 gamma norm→out projection can then be composed and checked against actual cached V3 output/cache, with true independently sourced prior state explicitly labeled.
4. **Closed initial chunk64 core:** same typed backend with exact phase/axis/precision contract, accepted scale/prefix/pointwise ops +admitted exp and six public BMM/two TRSM families. ROOT H48 replay every captured intermediate/core/state, followed by initial result→cached continuation and layer0 outputs. No general profile enablement or synthetic expected outputs.
5. Only after complete layer/model parity: attach original typed effects/layout codec to SAME SessionAuthority/StateRoot/ResourceArbiter. Reconcile all112 domains/full logits and complete-source streaming/physical bounds. Existing private slice source acceptance is not that gate.

````

## Source: .agents/runs/model-execution-qwen38-gdn-next-cut-audit-20261020/TODO.md

Original UTF-8 bytes: 723; SHA256 `e34b956c0fe5fcbbe0a2201e87d3d035c838a16465d27e7b332f6f8df5e502ec`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B next GDN cut audit

Parent model-execution; owner baseline_review. Source/metadata only.

- [x] Read current live priority, pinned layer0 source, actual public/native cached/gamma/provider/startup metadata and current/pending typed contracts.
- [x] Separate exact EXP48/initial chunk64 arithmetic+admission gaps from representation/source-only acceptance.
- [x] Map smallest disjoint scale/prefix, BF16 A_log/repeat, admitted exp, chunk and existing-engine composition cuts.
- [ ] ROOT future implementation/review/actual qualification; no source/model/profile promotion from this audit.

No shared plan/MAIN/other freezes changed; no build/tests/operators/provider/model/GPU/runtime or payload scans/hashes.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/FOLLOWUP_V2.md

Original UTF-8 bytes: 2075; SHA256 `74f4f245af89c305c18e8ad16f5a96d70148c35d727644f9dcd664820ff50270`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Header-read bound followup

Author: spine_review. Source-only additive V2.

- Original freeze `09461af63adc9e53a279a40d6608efcabeaa3d8a405f04d9a55b220bf73ef0f4`
  and every original source/observer/plan byte remain unchanged.
- Peer review found capture read the declared header body before validating its
  size. Independent validation also needed an explicit pre-read cap.
- Shared `support_v2.source_header`: retained exact-size regular FD; exact 8-byte
  length read; pinned length, `1 < n <= 8 MiB`, and in-file extent before body
  read; exact body length/hash and unchanged FD fingerprint afterward.
- Capture also checks the FD fingerprint still equals its pre-whole-hash value.
  Independent validator retains its original per-shard FD through all raw reads.
- `run_reference_v2.py` selects `capture_v2.py`; all V2 entrypoints bind both
  original frozen source/metadata and the additive V2 manifest.
- All public math, original operator calls, hooks, source roles, dtypes, mapping
  geometry, bounds, archives, expected values, cleanup and eligibility unchanged.
- Nine prepared stdlib reader fixtures: valid bounds, oversized/mismatched/short
  length, short body, wrong hash, FD mutation, wrong file size and nonregular FD.
  Synthetic fixtures establish only predicates when ROOT executes them.

ROOT commands from this directory (not run by author):

```bash
python3 test_header_bounds_v2.py
python3 run_reference_v2.py --new-output /ABSOLUTE/RUN/root-reference-v2-1
python3 validate_reference_v2.py --output /ABSOLUTE/RUN/root-reference-v2-1
```

Original three schema fixtures passed under ROOT, before this followup. No
public model/reference execution has occurred. Full H48 capture, finite resource
containment and independent archive validation remain pending. No native
provider/exp admission, full-model StateRoot, GPU or profile promotion follows.

Author checks: five Python ASTs; original eleven bytes/frozen hashes; static
review diff. No fixture, model, Torch, operator, provider, container, build or
GPU execution; no weight/DSO payload hashes.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/FOLLOWUP_V3.md

Original UTF-8 bytes: 3013; SHA256 `80e2601b898f4a8df4b4f280ddf3a3bc9a1bb75886b7e478e81d5d6a9eab6014`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Singleton-stride raw-bit transport

Author: spine_review. Additive source-only V3; original V1/V2 remain immutable.

ROOT V2 attempt `root-reference-v2-1` reached actual original operators but
failed after saving the first observed archive: BF16 byte reinterpretation
requires last stride1 when the element size changes. A singleton dimension
can be logically contiguous while keeping last stride10240. That attempt and
all original sources/archives stay preserved; no complete reference passed.

`tensor_bits_v3.tensor_bits` accepts only bounded original CPU strided BF16/F32.
It reinterprets BF16 as int16 and F32 as int32 (same element size), then NumPy
`tobytes(order='C')` serializes logical row order. There is no numerical cast,
float arithmetic, storage mutation or replacement operator. Explicit little
endianness/dtype/device/layout/128MiB bounds precede views and copies. Signed
zero, subnormal, Inf and NaN payload words must remain unchanged.

`capture_v3.py` replaces only the local bits helper/import. Reader V2 guard,
original operators/weights/dtypes/hooks/call topology/safetensors writes and all
expected/source/state checks stay unchanged. No other live original observer
contains the byte-view hazard. `run_reference_v3.py` selects that entrypoint;
`validate_reference_v3.py` only changes support import. `support_v3.frozen`
validates original V1, exact V2, and every additive V3 file before/after use.

ROOT-only `transport_fixtures_v3.py` uses the exact existing public Torch/runtime
without model/source weights: two dtypes × six layouts (normal, singleton-last,
singleton-middle, transpose, nonzero offset, stride2),192 literal raw words,
three unsupported dtype refusals, all underlying integer storage/canaries exact.
Fixture literals include signed zero, +/-subnormal, finite extrema, +/-Inf and
quiet/signaling NaN payloads. Expected bytes are literal integer words and
explicit index permutations, never candidate-generated floating results.

`run_transport_fixtures_v3.py` uses byte-identical owned-v4 lifecycle from the
original freeze: no network/GPU/model mount, read-only probe/image, fresh output,
12GiB memory+swap12GiB, quota4/explicit four-CPU affinity,12GiB AS before Torch,
120s process deadline, complete bounded logs/nonce-CID/PGID/reap/EOF cleanup.
This is a transport-only control, not model math/profile admission.

ROOT commands from this directory, after source review and quiet-run scheduling:

```bash
python3 run_transport_fixtures_v3.py --new-output /ABSOLUTE/RUN/root-transport-v3-1
python3 run_reference_v3.py --new-output /ABSOLUTE/RUN/root-reference-v3-1
python3 validate_reference_v3.py --output /ABSOLUTE/RUN/root-reference-v3-1
```

Author checks only seven Python ASTs, narrow diff and original V1/V2 frozen-byte
checks. No fixture/Torch/operator/model/container/provider/build/GPU execution;
no source-weight or DSO payload hashes. ROOT executes all new controls and
full reference; H48/112-StateRoot/full-model/native-exp eligibility stays closed.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/README.md

Original UTF-8 bytes: 5605; SHA256 `a2120a75dbbc3d62065c3a1a6c2f2809a36d562d673476ebaaa6769b0476094d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B layer0 GDN public reference — prepared only

Author: spine_review. Parent plan: model-execution.

- Actual original `Qwen/Qwen3.8-27B` snapshot98cea13c config explicitly declares
  Qwen3_5ForConditionalGeneration/qwen3_5/qwen3_5_text. Use that exact pinned
  original module; no architecture inference from the model name.
- Original F.embedding →original Qwen3_5RMSNorm →original Qwen3_5GatedDeltaNet
  layer0. Only two teachers:11 enters real padded initial chunk64;353 enters
  real cached recurrence using that first return through original DynamicCache.
  No whole-model constructor/traversal,55GB materialization or112-domain claim.
- Exact nine GDN parameters +inputnorm +embedding bind by original index, shape,
  raw BF16 kind and same opened source FD/range. All parameters assigned strictly
  into meta-constructed ORIGINAL classes; no random/unfilled parameter may run.
  A_log,dt_bias and gated norm gamma remain ORIGINAL BF16. No source36F32 restore,
  official-loader blanket cast or conversion from GGUF learned values.
- Genuine original source public forward called unchanged once per invocation.
  Public USE_HUB_KERNELS=0 declares CPU reference profile; optional package
  fallback must identify original Torch code or refuse. No function replacement,
  scalar arithmetic port, reconstructed state or native exp eligibility.
- Independent fresh plain/observed caches, same immutable weights. Every token's
  complete output5120, conv10240×4, recurrent48×128×128, embedding and inputnorm
  must be bit-identical. True initial return must equal cached prior in all words.
  Real source Q/K16→V48 repeat3 is compared directly from captured actual bytes.
- Original call/line/return frame hooks reuse accepted source line maps. Capture
  q/k/v/beta/g, initial/prior/proposed/returned states, all old GDN intermediates,
  actual conv operands/return, projection input/output, and gated RMS stages.
  Scoped TorchDispatchMode records original coefficient operations, five real
  initial exps, cached exp(g) and norm args/results; forwards each original op
  once and returns its actual result. Observer clones are suppressed.
- Gated gamma product must be actual BF16 BEFORE FP32 SiLU multiplication. This
  differs from the `.8B` source36F32 gamma profile. No new native dtype eligibility.
- Legitimate initial pairwise exp input contains canonical negative Inf ONLY in
  the exact strictly-upper64×64 source mask. Preserve those raw bits; all other
  captured values and published outputs must be finite. No tolerance widening.
- Six actual embedding/inputnorm/QKV outputs also match the independently
  accepted original27B public prefix reference in every word.

## Bounds and source ownership

- BEFORE importing Torch or creating maps: page4096, eleven exact parameter
  windows, selected GDN payload231,752,128B; touched source incl inputnorm and
  two embedding rows231,782,848B; aggregate mapped VA2,774,582,792B (<3GiB).
- Each COW window maps page-aligned offset and only leading alignment+complete
  original tensor bytes. Full embedding raw view2.543GB VA retains its original
  geometry; only trained rows11/353 are read. No whole-shard/18-shard mmap.
- Before library initialization, derive separate virtual and physical-envelope
  estimates:6GiB runtime VA allowance,2GiB operator working estimate,128MiB
  captured payload,2GiB runtime resident allowance,64MiB control allowance.
  Finite RLIMIT_AS12GiB +configured container12GiB RAM/totalRAM+swap,4CPUs,
 600s foreground/2400s outer wall,128MiB per-file,256MiB whole artifact cap.
  These estimates/OS containment do NOT prove native ResourceArbiter admission
  or opaque allocator bounds. Any actual cap/refusal remains UNQUALIFIED.
- ROOT alone hashes ALL18 original shards/all source files before and after via
  retained FDs and bounded1MiB reads. Inside, selected original FDs are fully
  verified before importing Torch and rechecked after; COW parameter readsets,
  exact pointers/dtypes/shapes and versions are unchanged. Source FD/map/tensor
  owners remain until the owned process exits; no premature mmap close.
- Exact accepted owned_container_v4.py53ee4270, fresh nonce/CID, networknone,
  readonly source/probe/root image, no GPU/CDI, inside affinity before Torch.
  All postspawn paths use original independent cleanup stages and EOF proof.
  No next provider/process may start after cleanup/evidence failure.
- Configured OS caps and raw proc/cgroup/mount metadata are retained; no claim
  that a configured limit or historical RSS is native physical admission.

## ROOT execution, after independent source review

All directories fresh. ROOT owns every test/container/operator execution.

```bash
python3 "$RUN/test_contracts.py"
python3 "$RUN/run_reference.py" --new-output "$RUN/root-reference-1"
python3 "$RUN/validate_reference.py" --output "$RUN/root-reference-1"
```

Output: untouched plain archive +two prevalidation observed archives, full
source/runtime/parameter/cleanup records. Failure preserves actual metadata and
raw frames before semantic validation. Final validator compares all raw inputs,
intermediates, initial/cached prior, source gamma rounding and transparency.
H48 cached inputs804960 words and outputs798720 words become reference artifacts
only after actual ROOT PASS. Existing model slice F32 gamma guard stays CLOSED;
future source-kind admission and native H48 replay require separate review.

No author new source payload hashes, model/provider/operator/library/container,
Cargo/GPU execution, MAIN/shared-plan edits or commits. AST/metadata only.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/TODO.md

Original UTF-8 bytes: 1035; SHA256 `427ce9395f019bd5bed06a197f92e538e9252ecdcb6be2733638304bef898894`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: `.agents/plans/model-execution/`. Owner: spine_review.

- [x] Read actual original config, index and bounded selected source headers.
- [x] Verify declared Qwen3_5 architecture and all11 raw BF16 role descriptors.
- [x] Derive exact parameter-window VA/touched-payload bounds before libraries.
- [x] Prepare original module/meta strict assignment, real DynamicCache, two
  teacher plain/observed reference, original intermediate/dispatch observers.
- [x] Prepare whole-source/FD/map/readset/lifetime checks, original owned runner
  and independent raw/state/source/provider/cleanup validator; no math replacement.
- [x] Complete author static self-review/AST/freeze; no runtime or tests.
- [ ] ROOT independent full source review.
- [ ] ROOT source-free fixtures, actual original public reference and validator.
- [ ] Separate H48 source-kind/native cached slice qualification after reference.

No author runtime/tests/new payload hashes. No full-model/112 StateRoot/native
exp/GPU admission from this source-only preparation.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 5605; SHA256 `a2120a75dbbc3d62065c3a1a6c2f2809a36d562d673476ebaaa6769b0476094d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B layer0 GDN public reference — prepared only

Author: spine_review. Parent plan: model-execution.

- Actual original `Qwen/Qwen3.8-27B` snapshot98cea13c config explicitly declares
  Qwen3_5ForConditionalGeneration/qwen3_5/qwen3_5_text. Use that exact pinned
  original module; no architecture inference from the model name.
- Original F.embedding →original Qwen3_5RMSNorm →original Qwen3_5GatedDeltaNet
  layer0. Only two teachers:11 enters real padded initial chunk64;353 enters
  real cached recurrence using that first return through original DynamicCache.
  No whole-model constructor/traversal,55GB materialization or112-domain claim.
- Exact nine GDN parameters +inputnorm +embedding bind by original index, shape,
  raw BF16 kind and same opened source FD/range. All parameters assigned strictly
  into meta-constructed ORIGINAL classes; no random/unfilled parameter may run.
  A_log,dt_bias and gated norm gamma remain ORIGINAL BF16. No source36F32 restore,
  official-loader blanket cast or conversion from GGUF learned values.
- Genuine original source public forward called unchanged once per invocation.
  Public USE_HUB_KERNELS=0 declares CPU reference profile; optional package
  fallback must identify original Torch code or refuse. No function replacement,
  scalar arithmetic port, reconstructed state or native exp eligibility.
- Independent fresh plain/observed caches, same immutable weights. Every token's
  complete output5120, conv10240×4, recurrent48×128×128, embedding and inputnorm
  must be bit-identical. True initial return must equal cached prior in all words.
  Real source Q/K16→V48 repeat3 is compared directly from captured actual bytes.
- Original call/line/return frame hooks reuse accepted source line maps. Capture
  q/k/v/beta/g, initial/prior/proposed/returned states, all old GDN intermediates,
  actual conv operands/return, projection input/output, and gated RMS stages.
  Scoped TorchDispatchMode records original coefficient operations, five real
  initial exps, cached exp(g) and norm args/results; forwards each original op
  once and returns its actual result. Observer clones are suppressed.
- Gated gamma product must be actual BF16 BEFORE FP32 SiLU multiplication. This
  differs from the `.8B` source36F32 gamma profile. No new native dtype eligibility.
- Legitimate initial pairwise exp input contains canonical negative Inf ONLY in
  the exact strictly-upper64×64 source mask. Preserve those raw bits; all other
  captured values and published outputs must be finite. No tolerance widening.
- Six actual embedding/inputnorm/QKV outputs also match the independently
  accepted original27B public prefix reference in every word.

## Bounds and source ownership

- BEFORE importing Torch or creating maps: page4096, eleven exact parameter
  windows, selected GDN payload231,752,128B; touched source incl inputnorm and
  two embedding rows231,782,848B; aggregate mapped VA2,774,582,792B (<3GiB).
- Each COW window maps page-aligned offset and only leading alignment+complete
  original tensor bytes. Full embedding raw view2.543GB VA retains its original
  geometry; only trained rows11/353 are read. No whole-shard/18-shard mmap.
- Before library initialization, derive separate virtual and physical-envelope
  estimates:6GiB runtime VA allowance,2GiB operator working estimate,128MiB
  captured payload,2GiB runtime resident allowance,64MiB control allowance.
  Finite RLIMIT_AS12GiB +configured container12GiB RAM/totalRAM+swap,4CPUs,
 600s foreground/2400s outer wall,128MiB per-file,256MiB whole artifact cap.
  These estimates/OS containment do NOT prove native ResourceArbiter admission
  or opaque allocator bounds. Any actual cap/refusal remains UNQUALIFIED.
- ROOT alone hashes ALL18 original shards/all source files before and after via
  retained FDs and bounded1MiB reads. Inside, selected original FDs are fully
  verified before importing Torch and rechecked after; COW parameter readsets,
  exact pointers/dtypes/shapes and versions are unchanged. Source FD/map/tensor
  owners remain until the owned process exits; no premature mmap close.
- Exact accepted owned_container_v4.py53ee4270, fresh nonce/CID, networknone,
  readonly source/probe/root image, no GPU/CDI, inside affinity before Torch.
  All postspawn paths use original independent cleanup stages and EOF proof.
  No next provider/process may start after cleanup/evidence failure.
- Configured OS caps and raw proc/cgroup/mount metadata are retained; no claim
  that a configured limit or historical RSS is native physical admission.

## ROOT execution, after independent source review

All directories fresh. ROOT owns every test/container/operator execution.

```bash
python3 "$RUN/test_contracts.py"
python3 "$RUN/run_reference.py" --new-output "$RUN/root-reference-1"
python3 "$RUN/validate_reference.py" --output "$RUN/root-reference-1"
```

Output: untouched plain archive +two prevalidation observed archives, full
source/runtime/parameter/cleanup records. Failure preserves actual metadata and
raw frames before semantic validation. Final validator compares all raw inputs,
intermediates, initial/cached prior, source gamma rounding and transparency.
H48 cached inputs804960 words and outputs798720 words become reference artifacts
only after actual ROOT PASS. Existing model slice F32 gamma guard stays CLOSED;
future source-kind admission and native H48 replay require separate review.

No author new source payload hashes, model/provider/operator/library/container,
Cargo/GPU execution, MAIN/shared-plan edits or commits. AST/metadata only.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1035; SHA256 `427ce9395f019bd5bed06a197f92e538e9252ecdcb6be2733638304bef898894`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: `.agents/plans/model-execution/`. Owner: spine_review.

- [x] Read actual original config, index and bounded selected source headers.
- [x] Verify declared Qwen3_5 architecture and all11 raw BF16 role descriptors.
- [x] Derive exact parameter-window VA/touched-payload bounds before libraries.
- [x] Prepare original module/meta strict assignment, real DynamicCache, two
  teacher plain/observed reference, original intermediate/dispatch observers.
- [x] Prepare whole-source/FD/map/readset/lifetime checks, original owned runner
  and independent raw/state/source/provider/cleanup validator; no math replacement.
- [x] Complete author static self-review/AST/freeze; no runtime or tests.
- [ ] ROOT independent full source review.
- [ ] ROOT source-free fixtures, actual original public reference and validator.
- [ ] Separate H48 source-kind/native cached slice qualification after reference.

No author runtime/tests/new payload hashes. No full-model/112 StateRoot/native
exp/GPU admission from this source-only preparation.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/frozen-v2/FOLLOWUP_V2.md

Original UTF-8 bytes: 2075; SHA256 `74f4f245af89c305c18e8ad16f5a96d70148c35d727644f9dcd664820ff50270`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Header-read bound followup

Author: spine_review. Source-only additive V2.

- Original freeze `09461af63adc9e53a279a40d6608efcabeaa3d8a405f04d9a55b220bf73ef0f4`
  and every original source/observer/plan byte remain unchanged.
- Peer review found capture read the declared header body before validating its
  size. Independent validation also needed an explicit pre-read cap.
- Shared `support_v2.source_header`: retained exact-size regular FD; exact 8-byte
  length read; pinned length, `1 < n <= 8 MiB`, and in-file extent before body
  read; exact body length/hash and unchanged FD fingerprint afterward.
- Capture also checks the FD fingerprint still equals its pre-whole-hash value.
  Independent validator retains its original per-shard FD through all raw reads.
- `run_reference_v2.py` selects `capture_v2.py`; all V2 entrypoints bind both
  original frozen source/metadata and the additive V2 manifest.
- All public math, original operator calls, hooks, source roles, dtypes, mapping
  geometry, bounds, archives, expected values, cleanup and eligibility unchanged.
- Nine prepared stdlib reader fixtures: valid bounds, oversized/mismatched/short
  length, short body, wrong hash, FD mutation, wrong file size and nonregular FD.
  Synthetic fixtures establish only predicates when ROOT executes them.

ROOT commands from this directory (not run by author):

```bash
python3 test_header_bounds_v2.py
python3 run_reference_v2.py --new-output /ABSOLUTE/RUN/root-reference-v2-1
python3 validate_reference_v2.py --output /ABSOLUTE/RUN/root-reference-v2-1
```

Original three schema fixtures passed under ROOT, before this followup. No
public model/reference execution has occurred. Full H48 capture, finite resource
containment and independent archive validation remain pending. No native
provider/exp admission, full-model StateRoot, GPU or profile promotion follows.

Author checks: five Python ASTs; original eleven bytes/frozen hashes; static
review diff. No fixture, model, Torch, operator, provider, container, build or
GPU execution; no weight/DSO payload hashes.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/frozen-v3/FOLLOWUP_V3.md

Original UTF-8 bytes: 3013; SHA256 `80e2601b898f4a8df4b4f280ddf3a3bc9a1bb75886b7e478e81d5d6a9eab6014`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Singleton-stride raw-bit transport

Author: spine_review. Additive source-only V3; original V1/V2 remain immutable.

ROOT V2 attempt `root-reference-v2-1` reached actual original operators but
failed after saving the first observed archive: BF16 byte reinterpretation
requires last stride1 when the element size changes. A singleton dimension
can be logically contiguous while keeping last stride10240. That attempt and
all original sources/archives stay preserved; no complete reference passed.

`tensor_bits_v3.tensor_bits` accepts only bounded original CPU strided BF16/F32.
It reinterprets BF16 as int16 and F32 as int32 (same element size), then NumPy
`tobytes(order='C')` serializes logical row order. There is no numerical cast,
float arithmetic, storage mutation or replacement operator. Explicit little
endianness/dtype/device/layout/128MiB bounds precede views and copies. Signed
zero, subnormal, Inf and NaN payload words must remain unchanged.

`capture_v3.py` replaces only the local bits helper/import. Reader V2 guard,
original operators/weights/dtypes/hooks/call topology/safetensors writes and all
expected/source/state checks stay unchanged. No other live original observer
contains the byte-view hazard. `run_reference_v3.py` selects that entrypoint;
`validate_reference_v3.py` only changes support import. `support_v3.frozen`
validates original V1, exact V2, and every additive V3 file before/after use.

ROOT-only `transport_fixtures_v3.py` uses the exact existing public Torch/runtime
without model/source weights: two dtypes × six layouts (normal, singleton-last,
singleton-middle, transpose, nonzero offset, stride2),192 literal raw words,
three unsupported dtype refusals, all underlying integer storage/canaries exact.
Fixture literals include signed zero, +/-subnormal, finite extrema, +/-Inf and
quiet/signaling NaN payloads. Expected bytes are literal integer words and
explicit index permutations, never candidate-generated floating results.

`run_transport_fixtures_v3.py` uses byte-identical owned-v4 lifecycle from the
original freeze: no network/GPU/model mount, read-only probe/image, fresh output,
12GiB memory+swap12GiB, quota4/explicit four-CPU affinity,12GiB AS before Torch,
120s process deadline, complete bounded logs/nonce-CID/PGID/reap/EOF cleanup.
This is a transport-only control, not model math/profile admission.

ROOT commands from this directory, after source review and quiet-run scheduling:

```bash
python3 run_transport_fixtures_v3.py --new-output /ABSOLUTE/RUN/root-transport-v3-1
python3 run_reference_v3.py --new-output /ABSOLUTE/RUN/root-reference-v3-1
python3 validate_reference_v3.py --output /ABSOLUTE/RUN/root-reference-v3-1
```

Author checks only seven Python ASTs, narrow diff and original V1/V2 frozen-byte
checks. No fixture/Torch/operator/model/container/provider/build/GPU execution;
no source-weight or DSO payload hashes. ROOT executes all new controls and
full reference; H48/112-StateRoot/full-model/native-exp eligibility stays closed.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/internal-draft-1/README.md

Original UTF-8 bytes: 5605; SHA256 `a2120a75dbbc3d62065c3a1a6c2f2809a36d562d673476ebaaa6769b0476094d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B layer0 GDN public reference — prepared only

Author: spine_review. Parent plan: model-execution.

- Actual original `Qwen/Qwen3.8-27B` snapshot98cea13c config explicitly declares
  Qwen3_5ForConditionalGeneration/qwen3_5/qwen3_5_text. Use that exact pinned
  original module; no architecture inference from the model name.
- Original F.embedding →original Qwen3_5RMSNorm →original Qwen3_5GatedDeltaNet
  layer0. Only two teachers:11 enters real padded initial chunk64;353 enters
  real cached recurrence using that first return through original DynamicCache.
  No whole-model constructor/traversal,55GB materialization or112-domain claim.
- Exact nine GDN parameters +inputnorm +embedding bind by original index, shape,
  raw BF16 kind and same opened source FD/range. All parameters assigned strictly
  into meta-constructed ORIGINAL classes; no random/unfilled parameter may run.
  A_log,dt_bias and gated norm gamma remain ORIGINAL BF16. No source36F32 restore,
  official-loader blanket cast or conversion from GGUF learned values.
- Genuine original source public forward called unchanged once per invocation.
  Public USE_HUB_KERNELS=0 declares CPU reference profile; optional package
  fallback must identify original Torch code or refuse. No function replacement,
  scalar arithmetic port, reconstructed state or native exp eligibility.
- Independent fresh plain/observed caches, same immutable weights. Every token's
  complete output5120, conv10240×4, recurrent48×128×128, embedding and inputnorm
  must be bit-identical. True initial return must equal cached prior in all words.
  Real source Q/K16→V48 repeat3 is compared directly from captured actual bytes.
- Original call/line/return frame hooks reuse accepted source line maps. Capture
  q/k/v/beta/g, initial/prior/proposed/returned states, all old GDN intermediates,
  actual conv operands/return, projection input/output, and gated RMS stages.
  Scoped TorchDispatchMode records original coefficient operations, five real
  initial exps, cached exp(g) and norm args/results; forwards each original op
  once and returns its actual result. Observer clones are suppressed.
- Gated gamma product must be actual BF16 BEFORE FP32 SiLU multiplication. This
  differs from the `.8B` source36F32 gamma profile. No new native dtype eligibility.
- Legitimate initial pairwise exp input contains canonical negative Inf ONLY in
  the exact strictly-upper64×64 source mask. Preserve those raw bits; all other
  captured values and published outputs must be finite. No tolerance widening.
- Six actual embedding/inputnorm/QKV outputs also match the independently
  accepted original27B public prefix reference in every word.

## Bounds and source ownership

- BEFORE importing Torch or creating maps: page4096, eleven exact parameter
  windows, selected GDN payload231,752,128B; touched source incl inputnorm and
  two embedding rows231,782,848B; aggregate mapped VA2,774,582,792B (<3GiB).
- Each COW window maps page-aligned offset and only leading alignment+complete
  original tensor bytes. Full embedding raw view2.543GB VA retains its original
  geometry; only trained rows11/353 are read. No whole-shard/18-shard mmap.
- Before library initialization, derive separate virtual and physical-envelope
  estimates:6GiB runtime VA allowance,2GiB operator working estimate,128MiB
  captured payload,2GiB runtime resident allowance,64MiB control allowance.
  Finite RLIMIT_AS12GiB +configured container12GiB RAM/totalRAM+swap,4CPUs,
 600s foreground/2400s outer wall,128MiB per-file,256MiB whole artifact cap.
  These estimates/OS containment do NOT prove native ResourceArbiter admission
  or opaque allocator bounds. Any actual cap/refusal remains UNQUALIFIED.
- ROOT alone hashes ALL18 original shards/all source files before and after via
  retained FDs and bounded1MiB reads. Inside, selected original FDs are fully
  verified before importing Torch and rechecked after; COW parameter readsets,
  exact pointers/dtypes/shapes and versions are unchanged. Source FD/map/tensor
  owners remain until the owned process exits; no premature mmap close.
- Exact accepted owned_container_v4.py53ee4270, fresh nonce/CID, networknone,
  readonly source/probe/root image, no GPU/CDI, inside affinity before Torch.
  All postspawn paths use original independent cleanup stages and EOF proof.
  No next provider/process may start after cleanup/evidence failure.
- Configured OS caps and raw proc/cgroup/mount metadata are retained; no claim
  that a configured limit or historical RSS is native physical admission.

## ROOT execution, after independent source review

All directories fresh. ROOT owns every test/container/operator execution.

```bash
python3 "$RUN/test_contracts.py"
python3 "$RUN/run_reference.py" --new-output "$RUN/root-reference-1"
python3 "$RUN/validate_reference.py" --output "$RUN/root-reference-1"
```

Output: untouched plain archive +two prevalidation observed archives, full
source/runtime/parameter/cleanup records. Failure preserves actual metadata and
raw frames before semantic validation. Final validator compares all raw inputs,
intermediates, initial/cached prior, source gamma rounding and transparency.
H48 cached inputs804960 words and outputs798720 words become reference artifacts
only after actual ROOT PASS. Existing model slice F32 gamma guard stays CLOSED;
future source-kind admission and native H48 replay require separate review.

No author new source payload hashes, model/provider/operator/library/container,
Cargo/GPU execution, MAIN/shared-plan edits or commits. AST/metadata only.

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/internal-draft-1/TODO.md

Original UTF-8 bytes: 1035; SHA256 `427ce9395f019bd5bed06a197f92e538e9252ecdcb6be2733638304bef898894`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: `.agents/plans/model-execution/`. Owner: spine_review.

- [x] Read actual original config, index and bounded selected source headers.
- [x] Verify declared Qwen3_5 architecture and all11 raw BF16 role descriptors.
- [x] Derive exact parameter-window VA/touched-payload bounds before libraries.
- [x] Prepare original module/meta strict assignment, real DynamicCache, two
  teacher plain/observed reference, original intermediate/dispatch observers.
- [x] Prepare whole-source/FD/map/readset/lifetime checks, original owned runner
  and independent raw/state/source/provider/cleanup validator; no math replacement.
- [x] Complete author static self-review/AST/freeze; no runtime or tests.
- [ ] ROOT independent full source review.
- [ ] ROOT source-free fixtures, actual original public reference and validator.
- [ ] Separate H48 source-kind/native cached slice qualification after reference.

No author runtime/tests/new payload hashes. No full-model/112 StateRoot/native
exp/GPU admission from this source-only preparation.

````

## Source: .agents/runs/model-execution-qwen38-owned-window-20261020/README.md

Original UTF-8 bytes: 4817; SHA256 `29c1c8fd3a59b23b2d1da4c54db1ee7490fa630459d99919a8634af7c6f4ab01`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only original27B source-window qualification

Prepared, **not compiled or executed**. Merge/review the repaired artifact
window API first; sole MAIN coordinator builds/runs. No downloads or GPU APIs.

Inputs: genuine `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; acquired snapshot
`98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e`.
`source-pin.json` is an exact copy of the prior independently verified source
report, SHA256 `4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a`.
The harness pins the report SHA, original index SHA, selected/weight/payload
byte counts, all18 shard SHA/lengths and the existing source-bundle digest.

One full **55,563,006,776-byte weight-file scan** authenticates all18 shards;
the selected artifact snapshot is55,586,113,293 bytes including other files.
No complete owned55GB allocation, mmap or `SourceWeights::materialize` occurs.
The harness deliberately avoids `CacheStore::open_snapshot`, which would first
rehash the entire snapshot and cause a second whole-source scan in the factory.
It opens each exact original file once read-only `O_NOFOLLOW|O_CLOEXEC`, checks
regular type/length and retains descriptor clones. The same opened inode feeds
Artifact→ExpectedSourceShard→the existing source-window issuer and independent
positioned raw-reference reads; no raw comparison reopens a pathname.

Factory open uses bounded4MiB scan/staging chunks and a64MiB retained metadata
reservation inside an explicitly configured128MiB host pool. This exercises
the existing ResourceArbiter control-plane ledger; **it does not independently
attest free RAM, enforce RSS/cgroup admission or count all harness/allocator/
kernel page-cache memory**. Caller-owned report/index/header descriptors and
generic arbiter bookkeeping are outside the window payload accounting. Raw
comparison copies have a separate `Other` lease, at most256KiB.

Every sample compares exact owned BF16 bytes, with no decoding or tolerance:

| Matrix | Original shape | Shard | Absolute source byte offset |
| --- | --- | --- | ---: |
| layer0 linear_attn.in_proj_qkv.weight | [10240,5120] | 1 | 1120792 |
| embed_tokens.weight | [248320,5120] | 3 | 152 |
| layer63 mlp.down_proj.weight | [5120,17408] | 17 | 1364259120 |
| lm_head.weight | [248320,5120] | 18 | 1760 |

Names are original `model.language_model.*` paths except `lm_head.weight`.
Shapes/kinds/absolute offsets were independently read from pinned original
headers during preparation; the returned sealed descriptor must match them.
For each matrix: first row, last row and full-width row(s) crossing an actual
4MiB authenticated scan-extent boundary. The live lease must equal complete
selected chunk bytes plus retained offset-array bytes, not requested row bytes.
Every read/drop checks exact Scratch/TransferStaging/Other class totals.

Before payload scanning: actual-header preparation followed by metadata bound,
whole-source bound and occupied host-pool refusal; prepared-plan pre-cancel must
drop its metadata reservation. This is pre-cancel coverage only, not an
asynchronous mid-read cancellation latency claim. After scanning: pre-cancelled
row, wrong axes, actual rank-one source, missing name, invalid rows and attempted
whole embedding materialization all refuse without a retained staging charge.
The final owned window alone keeps original identity/metadata/staging live after
all caller source/matrix handles drop; final window drop must release all leases.

Parent commands (not run by child), from MAIN, using the existing isolation
wrapper and offline dependencies. The local manifest declares an empty workspace
and exact already-pinned dependency versions; root may retain its generated lock:

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- cargo build --offline --release --manifest-path .agents/runs/model-execution-qwen38-owned-window-20261020/Cargo.toml
```

Execute the built CPU-only harness once, retaining stdout as actual-report.json,
stderr as actual-run.log and process exit status. A finite timeout and `/usr/bin/time
-v` RSS observation may be applied by the coordinator. Default local binary:
`.agents/runs/model-execution-qwen38-owned-window-20261020/target/release/qwen38-owned-window-qualification`.
No counter/runtime state must be inferred from these unexecuted commands.

Success label is only `PASS_SOURCE_WINDOWS_ONLY`: whole-file SHA/index/header
closure, actual original bounded owned rows, same source identity, byte equality
and configured host-ledger lifetime. It does not qualify trained math, full
inference, TensorSource/ForwardPlan streaming, physical RAM attestation, GPU
readiness/StateRoot publication or vLLM performance. Failure prints
`FAIL_UNQUALIFIED`, actual error/final ledger and exits2.

````

## Source: .agents/runs/model-execution-qwen38-owned-window-20261020/TODO.md

Original UTF-8 bytes: 1559; SHA256 `2d09c0c45b3a3996fe66fd4c399ee197121c9b87b65861fa44d7d71f441ff851`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Actual Qwen3.8 27B owned-window harness

Parent: `.agents/plans/memory-and-streaming/TODO.md`, model-execution.
Owner: `/root/baseline_review`; parent executes/accepts.
Scope: ignored harness + instructions only; no production source changes.

- [x] Inspect repaired seven-path TensorWindow patch `0872794d...` and existing
  exact27B acquired snapshot/source-verification report.
- [x] Prepare root-only bounded whole-weight scan + actual raw-row qualifier.
- [x] Pin original index, 18 SHA/lengths, same-FD raw references, four actual
  BF16 matrix shapes/absolute source offsets, bundle and snapshot identity.
- [x] Include pre-cancel/metadata/source/live-cap refusal, wrong axes/rank/name/
  range/whole-tensor refusal, exact live staging+offset charge, source/window
  lifetime and final ledger zero.
- [x] Parent compiled current repaired artifacts API via locked offline isolated
  manifest after an initial direct-rustc LTO link failure (no payload scan).
- [x] Parent actual scan PASS_SOURCE_WINDOWS_ONLY: all18 shards /
  55,563,006,776B verified in82.847s;12 byte-exact raw BF16 samples including
  four real extent crossings; max staged charge8,388,656B; final ledger0.
  Actual run exit0; peakRSS17,328KiB under512MiB address cap.
  Receipts: actual-report.json, actual-run.log, actual-run-exit.json.
  Physical RAM admission / trained inference / GPU execution remain unqualified.

Child performed no Cargo, compiled harness execution, GPU/model/provider launch
  or source payload scan. Parent executed the qualified source-window run above.

````

## Source: .agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/ISOLATED_ROOT.md

Original UTF-8 bytes: 7183; SHA256 `239fb5193dbc8c24a8af839b7fe43329f1cefb62df2849580415e0514cf5870d`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: spine_review

- Additive source-only relocation. MAIN, original qualification/V2/public oracle
  and isolated ROOT worktree unchanged by preparation.
- Target ROOT521 source freeze7 `036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615`.
  The prior actual selected strict +237 focused checks are pinned; no new native
  pure-wide execution is claimed.
- Original thirteen-file base `16809776089e1c3063629a0e0df556512a66b9ccd04d6812d56e47b744f4cb40`;
  metadata V2 `e9581cf982e3c83fa85413a4eb707afef001290c6137563a34a9c679a54514e6`.
  Every original live/frozen file and parent pin remains authenticated.
- Four absolute Cargo crate roots relocate. Original main.rs/factory.rs,
  source-plan.json, actual7058B source-pin4c74fa28, public capture/validator,
  raw serializer and 53ee owner are byte-identical. support.py changes only its
  frozen() source dispatch. run_native.py changes only source/oracle-path build
  attestation dispatch and safe record-only TERM/INT observation.
- Native numerical code/arguments/512MiB AS/4CPU/1800s wall/4MiB logs/128MiB
  configured host ledger, source/admission/pressure/cancel/retention/full-word
  comparisons/final0 remain exact original bytes. No source model copy, fallback,
  new engine, extra projection/state operator or higher cap.

## Original oracle and source exceptions

- Use the existing independently accepted original directory, not a copied or
  regenerated archive:
  `/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/root-reference-1`.
- Its original validation/manifest/invocation metadata is pinned. Original
  validation retains freeze16809776; the relocated native bundle has its own
  preparation identity. These identities are explicitly separate, never retagged.
- Public original32-case/16-role input/output reference:881856 words,1763712B.
  Native unchanged main.rs verifies each complete actual payload against its
  original digest and all words. This is independent public expected data; source
  attestation alone is neither compiler nor numerical evidence.
- Original 8 implementation +6 dependency assertions remain explicit. Seven
  children are byte-identical: pure.rs, window/pure.rs, weights.rs,
  window_projection.rs, norm/ordinary.rs, artifact tensor_window.rs,
  original_binding.rs. Seven shared/test children use exact reviewed ROOT
  composition: forward.rs, host_math.rs, operator_slice.rs,
  operator_slice_window_roles_tests.rs, operator_slice_window_tests.rs,
  original_binding/window.rs and core program.rs. Their source-only line deltas
  are fully bound in relocation.json; this is not a claim of semantic equivalence
  from a diff. ROOT521, composition journals and recorded corrections close the
  intended implementation identity.
- Eight original ROOT correction exceptions2–7 remain exact: coefficient output
  size adapter; CPU/model coefficient Vec→array test spelling; H48 fixture budget/
  visibility/import/schedule/lifetime; closed Embedding role negative; distinct
  fresh SourceIdentity assertion. CPU row math and ordinary norm stay byte-identical.

## Coordinator sequence

1. Review this bundle and source-only `REVIEW.diff`. Run the original four and new
   thirteen source-free fixture methods only after the timing hold:
   `python3 -m unittest -v test_contracts.py test_relocation.py` from this directory.
   The author parsed Python AST only; no imports/tests/runtime were run.
2. Review ROOT521 source closure and build this local Cargo.toml using ROOT's
   existing controlled/offline packages and serialized target owner. Four crate
   paths point at the isolated ROOT worktree; MAIN must remain untouched.
3. Preserve actual compile argv/log/exit/binary/effective-lock proof. Prepare the
   isolated build attestation below after the actual build. All source files must
   still match ROOT036c521. A fresh local Cargo.lock is an additional actual input,
   and external package normalization against ROOT's accepted lock remains the
   coordinator's build gate. Caller labels/hashes alone do not prove compilation.
4. Execute the relocated run_native.py below with the original accepted oracle.
   The original validator is preserved byte-identically, but rerunning the copied
   public capture/validator is unnecessary: it would be a fresh capture with a
   separate preparation identity. Never edit or relabel original accepted data.
5. Reconcile only private pure-operator qualification after the actual881856-word
   result, cancellation/real-pressure/retained3035648B/final0/cleanup checks pass.

```sh
python3 run_native.py --binary /ABS/ROOT/RELEASE/BINARY \
  --binary-sha256 ROOT_ACTUAL_BINARY_SHA \
  --reference-directory /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/root-reference-1 \
  --build-attestation /ABS/ROOT/ACTUAL_BUILD_ATTESTATION.json \
  --new-output /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/root-native-1
```

Required actual build receipt fields:

```json
{
  "status": "ROOT_COMPILED_ISOLATED_ORIGINAL27B_PURE_ROLES_ONLY",
  "build_exit_code": 0,
  "binary": "/ABS/ROOT/RELEASE/BINARY",
  "binary_sha256": "ACTUAL_BINARY_SHA",
  "prepared_manifest_sha256": "THIS_PREPARED_MANIFEST_SHA",
  "qualification_manifest_sha256": "16809776089e1c3063629a0e0df556512a66b9ccd04d6812d56e47b744f4cb40",
  "qualification_metadata_followup_manifest_sha256": "e9581cf982e3c83fa85413a4eb707afef001290c6137563a34a9c679a54514e6",
  "root_source_freeze_sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615",
  "root_worktree": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
  "qualification_directory": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020",
  "MAIN_merged": false,
  "full_model_StateRoot_GPU_or_performance_qualified": false,
  "files": ["exact absolute path/bytes/sha256 source rows below"]
}
```

`files` = all521 final ROOT source rows with absolute ROOT paths, plus this
directory's exact Cargo.toml/main.rs/factory.rs/source-pin.json/source-plan.json;
plus local Cargo.lock if created. Exactly526 or527 unique entries. The original
V2 four compile bindings remain authenticated; the fifth is necessary because
unchanged main.rs actually `include_bytes!`s source-plan.json. Missing, duplicate,
foreign, MAIN/old status and unexpected source deltas refuse cold.

TERM+INT use exact accepted00d1efc1 record-only guard; no asynchronous exception
across Popen assignment or retirement. `cancel.observe` runs only after original
child/log handoff, restores original log.drain before unchanged53ee cleanup.
Cleanup still runs before synchronous cancellation is reported. No GPU/provider
stub, mocked process ownership or hidden retry is introduced.

Scope remains original16 pure roles/32 external probes only. Some role inputs are
literal BF16 external patterns, not authenticated trained model producers. No
full GDN/exp/initial/cache/112 StateRoots/session/full27B inference, GPU eligibility
or throughput claim.

````

## Source: .agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/ROOT.md

Original UTF-8 bytes: 5511; SHA256 `17bf7285ebaa0eb3f761d107fe7d4fcc16ceef5e5a2e333bbb73f192df6c5495`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only genuine original27B pure-role gate

Author: spine_review. Prepared source only; no fixtures/build/model/operator run.

## Scope and provenance

- Original snapshot98cea13c/revision1d4bf0f2, all18 acquired weight shards
  55,563,006,776B; actual index1199/ordinary-text851 closure. Reused accepted
  `factory.rs` byte-identical from the original27B three-operator replay.
- Public PyTorch original BF16 F.linear: GDN QKV/Z/A/B/out; attention full Q+gate,
  K/V/out; MLP gate/up/down; real untied lm_head. Three original RMSNorm calls:
  input0, post-attention0 and final. No nonlinearities, residuals, positions,
  caches, token/layer traversal or state publishers are added.
- Two external teacher probes11/353. Projection hidden5120 inputs are exact
  accepted archived original InputNorm0 BF16 words; norm inputs use accepted
  original embedding words. Only the GDN input prefix has its actual trained
  producer here. Other hidden-role inputs remain external probes. Nonhidden
  widths6144/17408 use explicitly literal finite nonzero BF16 patterns, never
  reconstructed or declared trained producer activations.
- Sixteen actual source/header role shapes, unchanged raw BF16, three original
  retained shard FDs. Each parameter mapping is exact page-aligned range plus
  leading alignment, never a whole shard/full55GB mapping. Original head alone
  is about2.543GB virtual. All parameter FD/map/tensor owners survive to exit;
  full source/hash/version/readset checks before/after actual public operators.
- Public capture calls each genuine operator twice, all input/output bits
  identical. This is a repeat control, no newly invented instrumentation route.
  Pinned original image/Torch2.13cf30153/Transformers5.17 source, AVX2/threads4;
  CPU loader/activation profile is distinct from vLLM CUDA/FLA.
- Exact parameter VA3,519,025,784B +6GiB runtime VA estimate +2GiB operator
  estimate +4MiB capture <12GiB RLIMIT_AS. RAM/totalRAM+swap12GiB,CPU quota4,
  noGPU/network, read-only source/probe/image, finite600s child/2400s parent,
  32MiB whole artifacts. Estimates/cgroup containment do not prove complete
  physical host/provider admission. Mapping VA is not resident RSS.

## Execution, after review and timing hold

1. ROOT run `python3 test_contracts.py` here (four prepared stdlib bounds tests).
   Run the separate original27B GDN V3 actual-runtime raw serializer fixtures
   first; tensor-bit helper body is the same equal-size integer view/C-order
   transport with only its local support import changed.
2. ROOT `python3 run_reference.py --new-output /ABSOLUTE/QUAL/root-reference-1`.
   Actual full55GB source hash scans happen only under ROOT's quiet scheduling.
3. ROOT `python3 validate_reference.py --output /ABSOLUTE/QUAL/root-reference-1`.
   Independent stdlib full raw/header/readset/teacher/count/lifetime validator.
   All32 cases/all inputs/all output words must pass; no tolerance.
4. ROOT merges reviewed role candidate0f9 plus exact-hidden65cd and completed-cancel7759 followups, runs
   strict/focused/default/source-window regressions; builds this Cargo.toml only
   under its one-coordinator/offline actual-source isolation. No author Cargo.
5. ROOT writes actual build attestation (not fabricated by these preparations):
   status `ROOT_COMPILED_ORIGINAL27B_PURE_ROLES_ONLY`, actual binary/path/hash,
   build_exit_code0, explicit absolute root_worktree, actual_source_files eight
   path/bytes/SHA entries matching original+followup source, candidate_manifest
   0f9f2a8f…,exact_hidden_followup_manifest65cd4364…,completed_cancel_followup_manifest7759599f…. Preserve actual compile
   command/log/effective lock/artifact-source proof. These labels/caller hashes
   alone are not independent evidence of compilation or correct arithmetic.
6. ROOT `python3 run_native.py --binary ABS --binary-sha256 SHA \
   --reference-directory /ABSOLUTE/QUAL/root-reference-1 \
   --build-attestation ABS --new-output /ABSOLUTE/QUAL/root-native-1`.

## Native proof boundary

- SAME public OriginalWindowOperators pure APIs; SAME accepted source factory,
  SourceWeights windows, HostMath/CPU64/RNE/ordinary norm. No replacement engine.
- Full32-case input/output bit comparisons. Full248320-row untied head and
 17408-wide down included; every signed-zero bit is compared without tolerance.
- Real configured original ResourceArbiter128MiB host domain, source metadata
 64MiB, qualifier-control8MiB, model metadata4MiB, actual per-op output/scratch
  and row-window data+offset tickets. No claim128MiB is physical free host RAM.
- Actual pre-cancel and real pool exhaustion refuse; every temporary owner
  releases, no device fence. Last actual head result holds source/input/output
  leases after caller/source drop; final all payload/source owner drop0.
- Native process512MiB AS,CPU4/noGPU/no provider, bounded1800s/4MiB logs.
  Exact accepted owned-v4 generic WNOWAIT group cleanup function reused, two
  independent KILL attempts before any reap, complete group absence/EOF.
- PASS is only private source-authentic pure-operator numerical/transport scope.
  No all-role trained activation producer, GDN BF16 gamma/Alog/exp/initial/cached
  math,112 StateRoots, session/full27B model, GPU eligibility or speed claim.

Source-only author validation: Python ASTs/Rustfmt/small source/metadata pin and
header reads. No weight/DSO payload hashes, fixture/Torch/operator/model/native/
container/build/GPU execution. All failed/interrupted ROOT attempts stay intact.

````

## Source: .agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/TODO.md

Original UTF-8 bytes: 1139; SHA256 `f5ff60117da6e2ca9078ed6f64b2b9a2a4e197d453396904d8ef27de69b44463`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent model-execution; ROOT reconciles shared plan only.
- Own additive ignored relocation; preserve all original qualified source/data.
- [x] Inspect original13-file16809776 + metadata-only V2e9581cf9.
- [x] Pin ROOT036c521, prior actual strict/focused receipt and correction journals.
- [x] Conserve all14 source-child assertions:7 unchanged,7 explicit ROOT composition.
- [x] Preserve native Rust/factory/source plan/source pin/public oracle/validator.
- [x] Prepare Cargo4-root and source-attestation dispatch only.
- [x] Replace unsafe async TERM handler with exact record-only TERM+INT checkpoints.
- [x] Bind all5 actual local build inputs; preserve original V2 four bindings.
- [x] Prepare13 relocation negatives; source reads/AST only.
- [ ] ROOT + independent source review.
- [ ] ROOT fixtures/strict/actual standalone release compile and source attestation.
- [ ] ROOT actual881856-word native replay/lifetime/final0, no full-model promotion.

Author: no tests, imports, builds, operators/models/providers/GPU/runtime or
archive/model/DSO/binary payload reads/hashes. Original files and MAIN untouched.

````

## Source: .agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/frozen-prepared/ISOLATED_ROOT.md

Original UTF-8 bytes: 7183; SHA256 `239fb5193dbc8c24a8af839b7fe43329f1cefb62df2849580415e0514cf5870d`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: spine_review

- Additive source-only relocation. MAIN, original qualification/V2/public oracle
  and isolated ROOT worktree unchanged by preparation.
- Target ROOT521 source freeze7 `036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615`.
  The prior actual selected strict +237 focused checks are pinned; no new native
  pure-wide execution is claimed.
- Original thirteen-file base `16809776089e1c3063629a0e0df556512a66b9ccd04d6812d56e47b744f4cb40`;
  metadata V2 `e9581cf982e3c83fa85413a4eb707afef001290c6137563a34a9c679a54514e6`.
  Every original live/frozen file and parent pin remains authenticated.
- Four absolute Cargo crate roots relocate. Original main.rs/factory.rs,
  source-plan.json, actual7058B source-pin4c74fa28, public capture/validator,
  raw serializer and 53ee owner are byte-identical. support.py changes only its
  frozen() source dispatch. run_native.py changes only source/oracle-path build
  attestation dispatch and safe record-only TERM/INT observation.
- Native numerical code/arguments/512MiB AS/4CPU/1800s wall/4MiB logs/128MiB
  configured host ledger, source/admission/pressure/cancel/retention/full-word
  comparisons/final0 remain exact original bytes. No source model copy, fallback,
  new engine, extra projection/state operator or higher cap.

## Original oracle and source exceptions

- Use the existing independently accepted original directory, not a copied or
  regenerated archive:
  `/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/root-reference-1`.
- Its original validation/manifest/invocation metadata is pinned. Original
  validation retains freeze16809776; the relocated native bundle has its own
  preparation identity. These identities are explicitly separate, never retagged.
- Public original32-case/16-role input/output reference:881856 words,1763712B.
  Native unchanged main.rs verifies each complete actual payload against its
  original digest and all words. This is independent public expected data; source
  attestation alone is neither compiler nor numerical evidence.
- Original 8 implementation +6 dependency assertions remain explicit. Seven
  children are byte-identical: pure.rs, window/pure.rs, weights.rs,
  window_projection.rs, norm/ordinary.rs, artifact tensor_window.rs,
  original_binding.rs. Seven shared/test children use exact reviewed ROOT
  composition: forward.rs, host_math.rs, operator_slice.rs,
  operator_slice_window_roles_tests.rs, operator_slice_window_tests.rs,
  original_binding/window.rs and core program.rs. Their source-only line deltas
  are fully bound in relocation.json; this is not a claim of semantic equivalence
  from a diff. ROOT521, composition journals and recorded corrections close the
  intended implementation identity.
- Eight original ROOT correction exceptions2–7 remain exact: coefficient output
  size adapter; CPU/model coefficient Vec→array test spelling; H48 fixture budget/
  visibility/import/schedule/lifetime; closed Embedding role negative; distinct
  fresh SourceIdentity assertion. CPU row math and ordinary norm stay byte-identical.

## Coordinator sequence

1. Review this bundle and source-only `REVIEW.diff`. Run the original four and new
   thirteen source-free fixture methods only after the timing hold:
   `python3 -m unittest -v test_contracts.py test_relocation.py` from this directory.
   The author parsed Python AST only; no imports/tests/runtime were run.
2. Review ROOT521 source closure and build this local Cargo.toml using ROOT's
   existing controlled/offline packages and serialized target owner. Four crate
   paths point at the isolated ROOT worktree; MAIN must remain untouched.
3. Preserve actual compile argv/log/exit/binary/effective-lock proof. Prepare the
   isolated build attestation below after the actual build. All source files must
   still match ROOT036c521. A fresh local Cargo.lock is an additional actual input,
   and external package normalization against ROOT's accepted lock remains the
   coordinator's build gate. Caller labels/hashes alone do not prove compilation.
4. Execute the relocated run_native.py below with the original accepted oracle.
   The original validator is preserved byte-identically, but rerunning the copied
   public capture/validator is unnecessary: it would be a fresh capture with a
   separate preparation identity. Never edit or relabel original accepted data.
5. Reconcile only private pure-operator qualification after the actual881856-word
   result, cancellation/real-pressure/retained3035648B/final0/cleanup checks pass.

```sh
python3 run_native.py --binary /ABS/ROOT/RELEASE/BINARY \
  --binary-sha256 ROOT_ACTUAL_BINARY_SHA \
  --reference-directory /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-window-roles-20261020/qualification/root-reference-1 \
  --build-attestation /ABS/ROOT/ACTUAL_BUILD_ATTESTATION.json \
  --new-output /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/root-native-1
```

Required actual build receipt fields:

```json
{
  "status": "ROOT_COMPILED_ISOLATED_ORIGINAL27B_PURE_ROLES_ONLY",
  "build_exit_code": 0,
  "binary": "/ABS/ROOT/RELEASE/BINARY",
  "binary_sha256": "ACTUAL_BINARY_SHA",
  "prepared_manifest_sha256": "THIS_PREPARED_MANIFEST_SHA",
  "qualification_manifest_sha256": "16809776089e1c3063629a0e0df556512a66b9ccd04d6812d56e47b744f4cb40",
  "qualification_metadata_followup_manifest_sha256": "e9581cf982e3c83fa85413a4eb707afef001290c6137563a34a9c679a54514e6",
  "root_source_freeze_sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615",
  "root_worktree": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
  "qualification_directory": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020",
  "MAIN_merged": false,
  "full_model_StateRoot_GPU_or_performance_qualified": false,
  "files": ["exact absolute path/bytes/sha256 source rows below"]
}
```

`files` = all521 final ROOT source rows with absolute ROOT paths, plus this
directory's exact Cargo.toml/main.rs/factory.rs/source-pin.json/source-plan.json;
plus local Cargo.lock if created. Exactly526 or527 unique entries. The original
V2 four compile bindings remain authenticated; the fifth is necessary because
unchanged main.rs actually `include_bytes!`s source-plan.json. Missing, duplicate,
foreign, MAIN/old status and unexpected source deltas refuse cold.

TERM+INT use exact accepted00d1efc1 record-only guard; no asynchronous exception
across Popen assignment or retirement. `cancel.observe` runs only after original
child/log handoff, restores original log.drain before unchanged53ee cleanup.
Cleanup still runs before synchronous cancellation is reported. No GPU/provider
stub, mocked process ownership or hidden retry is introduced.

Scope remains original16 pure roles/32 external probes only. Some role inputs are
literal BF16 external patterns, not authenticated trained model producers. No
full GDN/exp/initial/cache/112 StateRoots/session/full27B inference, GPU eligibility
or throughput claim.

````

## Source: .agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/frozen-prepared/ROOT.md

Original UTF-8 bytes: 5511; SHA256 `17bf7285ebaa0eb3f761d107fe7d4fcc16ceef5e5a2e333bbb73f192df6c5495`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only genuine original27B pure-role gate

Author: spine_review. Prepared source only; no fixtures/build/model/operator run.

## Scope and provenance

- Original snapshot98cea13c/revision1d4bf0f2, all18 acquired weight shards
  55,563,006,776B; actual index1199/ordinary-text851 closure. Reused accepted
  `factory.rs` byte-identical from the original27B three-operator replay.
- Public PyTorch original BF16 F.linear: GDN QKV/Z/A/B/out; attention full Q+gate,
  K/V/out; MLP gate/up/down; real untied lm_head. Three original RMSNorm calls:
  input0, post-attention0 and final. No nonlinearities, residuals, positions,
  caches, token/layer traversal or state publishers are added.
- Two external teacher probes11/353. Projection hidden5120 inputs are exact
  accepted archived original InputNorm0 BF16 words; norm inputs use accepted
  original embedding words. Only the GDN input prefix has its actual trained
  producer here. Other hidden-role inputs remain external probes. Nonhidden
  widths6144/17408 use explicitly literal finite nonzero BF16 patterns, never
  reconstructed or declared trained producer activations.
- Sixteen actual source/header role shapes, unchanged raw BF16, three original
  retained shard FDs. Each parameter mapping is exact page-aligned range plus
  leading alignment, never a whole shard/full55GB mapping. Original head alone
  is about2.543GB virtual. All parameter FD/map/tensor owners survive to exit;
  full source/hash/version/readset checks before/after actual public operators.
- Public capture calls each genuine operator twice, all input/output bits
  identical. This is a repeat control, no newly invented instrumentation route.
  Pinned original image/Torch2.13cf30153/Transformers5.17 source, AVX2/threads4;
  CPU loader/activation profile is distinct from vLLM CUDA/FLA.
- Exact parameter VA3,519,025,784B +6GiB runtime VA estimate +2GiB operator
  estimate +4MiB capture <12GiB RLIMIT_AS. RAM/totalRAM+swap12GiB,CPU quota4,
  noGPU/network, read-only source/probe/image, finite600s child/2400s parent,
  32MiB whole artifacts. Estimates/cgroup containment do not prove complete
  physical host/provider admission. Mapping VA is not resident RSS.

## Execution, after review and timing hold

1. ROOT run `python3 test_contracts.py` here (four prepared stdlib bounds tests).
   Run the separate original27B GDN V3 actual-runtime raw serializer fixtures
   first; tensor-bit helper body is the same equal-size integer view/C-order
   transport with only its local support import changed.
2. ROOT `python3 run_reference.py --new-output /ABSOLUTE/QUAL/root-reference-1`.
   Actual full55GB source hash scans happen only under ROOT's quiet scheduling.
3. ROOT `python3 validate_reference.py --output /ABSOLUTE/QUAL/root-reference-1`.
   Independent stdlib full raw/header/readset/teacher/count/lifetime validator.
   All32 cases/all inputs/all output words must pass; no tolerance.
4. ROOT merges reviewed role candidate0f9 plus exact-hidden65cd and completed-cancel7759 followups, runs
   strict/focused/default/source-window regressions; builds this Cargo.toml only
   under its one-coordinator/offline actual-source isolation. No author Cargo.
5. ROOT writes actual build attestation (not fabricated by these preparations):
   status `ROOT_COMPILED_ORIGINAL27B_PURE_ROLES_ONLY`, actual binary/path/hash,
   build_exit_code0, explicit absolute root_worktree, actual_source_files eight
   path/bytes/SHA entries matching original+followup source, candidate_manifest
   0f9f2a8f…,exact_hidden_followup_manifest65cd4364…,completed_cancel_followup_manifest7759599f…. Preserve actual compile
   command/log/effective lock/artifact-source proof. These labels/caller hashes
   alone are not independent evidence of compilation or correct arithmetic.
6. ROOT `python3 run_native.py --binary ABS --binary-sha256 SHA \
   --reference-directory /ABSOLUTE/QUAL/root-reference-1 \
   --build-attestation ABS --new-output /ABSOLUTE/QUAL/root-native-1`.

## Native proof boundary

- SAME public OriginalWindowOperators pure APIs; SAME accepted source factory,
  SourceWeights windows, HostMath/CPU64/RNE/ordinary norm. No replacement engine.
- Full32-case input/output bit comparisons. Full248320-row untied head and
 17408-wide down included; every signed-zero bit is compared without tolerance.
- Real configured original ResourceArbiter128MiB host domain, source metadata
 64MiB, qualifier-control8MiB, model metadata4MiB, actual per-op output/scratch
  and row-window data+offset tickets. No claim128MiB is physical free host RAM.
- Actual pre-cancel and real pool exhaustion refuse; every temporary owner
  releases, no device fence. Last actual head result holds source/input/output
  leases after caller/source drop; final all payload/source owner drop0.
- Native process512MiB AS,CPU4/noGPU/no provider, bounded1800s/4MiB logs.
  Exact accepted owned-v4 generic WNOWAIT group cleanup function reused, two
  independent KILL attempts before any reap, complete group absence/EOF.
- PASS is only private source-authentic pure-operator numerical/transport scope.
  No all-role trained activation producer, GDN BF16 gamma/Alog/exp/initial/cached
  math,112 StateRoots, session/full27B model, GPU eligibility or speed claim.

Source-only author validation: Python ASTs/Rustfmt/small source/metadata pin and
header reads. No weight/DSO payload hashes, fixture/Torch/operator/model/native/
container/build/GPU execution. All failed/interrupted ROOT attempts stay intact.

````

## Source: .agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1139; SHA256 `f5ff60117da6e2ca9078ed6f64b2b9a2a4e197d453396904d8ef27de69b44463`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent model-execution; ROOT reconciles shared plan only.
- Own additive ignored relocation; preserve all original qualified source/data.
- [x] Inspect original13-file16809776 + metadata-only V2e9581cf9.
- [x] Pin ROOT036c521, prior actual strict/focused receipt and correction journals.
- [x] Conserve all14 source-child assertions:7 unchanged,7 explicit ROOT composition.
- [x] Preserve native Rust/factory/source plan/source pin/public oracle/validator.
- [x] Prepare Cargo4-root and source-attestation dispatch only.
- [x] Replace unsafe async TERM handler with exact record-only TERM+INT checkpoints.
- [x] Bind all5 actual local build inputs; preserve original V2 four bindings.
- [x] Prepare13 relocation negatives; source reads/AST only.
- [ ] ROOT + independent source review.
- [ ] ROOT fixtures/strict/actual standalone release compile and source attestation.
- [ ] ROOT actual881856-word native replay/lifetime/final0, no full-model promotion.

Author: no tests, imports, builds, operators/models/providers/GPU/runtime or
archive/model/DSO/binary payload reads/hashes. Original files and MAIN untouched.

````

## Source: .agents/runs/model-execution-qwen38-pure-wide-peer-review-batch-20261020/REVIEW.md

Original UTF-8 bytes: 3977; SHA256 `8f42a4f7ac2ddbad0dfa713e2b2ff7ccc87dfc95393cdaee214b61b0ab723fdb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B pure-wide actual proof — independent review

- Verdict: **no blocker** in retained source/build/ownership/numerical-proof closure. Review only; no execution, imports, builds, binary/DSO/model/raw-payload reads or hashes.
- Parent plan: model-execution; owned scope: this ignored review directory only. MAIN/shared plans/root README untouched.

## Actual receipt closure

- Prepared `44de7b7a`; isolated ROOT521 `036ce0dc`; actual implementation attestation `66c51e47`; build stages lock-normalize/metadata/strict/release all exit 0. All four small logs match declared bytes/SHA.
- Exact attestation tuple set = all521 isolated source rows + five local compile inputs (including embedded source-plan) + local lock =527. Binary digest/argv agree across build, implementation and invocation; binary bytes intentionally not re-read.
- All27 prepared live/frozen source rows checked. All14 original implementation/dependency children conserve recorded bytes: seven byte-identical, seven exact explicit composition/test deltas. Fourteen relocated harness copies retain original comparison/oracle/math bodies; only documented source/ownership/path adapters differ.
- ROOT retained fixture log reports17/17 and matches its pinned hash. This is source/contract evidence; it is not numerical execution.
- Native invocation: exit0, no cleanup errors, owned group absent, leader reaped once, both nonzero-signal stages before reap, pipe EOF/all handles closed and zero execution-log bytes. Original public invocation also exit0 with group/container/FD/EOF retirement and no cleanup errors; all31 source-before/after metadata rows equal.
- Original public validation/manifests remain exact `030b7fa0` / `7289c8b8`; original invocation `536b090b`. These retain original preparation `16809776`, not the relocated preparation identity.

## Numerical and physical scope

-32 native cases =16 roles × two teacher-bound external probes. Source comparison checks every imported BF16 input and every final BF16 output through SAME HostMath/window operations. Actual result: **881,856 total input+output words**, comprising192,512 inputs +689,344 outputs; all raw bits exact, including complete248,320-word untied-head projection in both probes.
- Authentic source roles/full geometries, BF16 raw source bytes and sealed source identity remain unchanged. Thirteen projections + three ordinary norms cover the listed GDN, attention, FFN and head/norm operators.
- Actual pre-cancel and physical source-pool pressure refusal occurred; comparison code requires pressure ledger unchanged after refusal. Retained head remains readable after operators/source/config caller drops.
- Retained private input/output/scratch charge =3,035,648 bytes (1,048,576 input workspace +993,280 output +993,792 output scratch). Source metadata and8MiB control remain additional charges in the exact ledger assertion. Final reserved=0 and unfenced after final owners drop.

## Explicit limits

- External pure probes, **not all roles' trained activation producers**: hidden probes reuse accepted InputNorm0, ordinary norms use accepted embedding, nonhidden widths use explicit finite BF16 literals. Reusing InputNorm0 for layer3/head/FFN probes does not produce their actual trained hidden states.
- Case positions0..31 are case indices, not StateRoot/token frontiers. The complete untied head projection is not full-model logits.
- No residual/attention/RoPE/GDN state execution, initial/cached schedule, full model, StateRoot parity, GPU, serving, performance, opaque provider or whole-host physical availability qualification. Runtime111.17s is a diagnostic duration.
-128MiB is the original ResourceArbiter limit for this native harness; it does not claim all process/OS memory is admitted. Public oracle's own12GiB AS containment is distinct from native ResourceArbiter admission.
- Implementation remains in the isolated ROOT tree, **MAIN_merged=false**. ROOT alone decides phase acceptance/merge.

````

## Source: .agents/runs/model-execution-qwen38-pure-wide-peer-review-batch-20261020/TODO.md

Original UTF-8 bytes: 804; SHA256 `3d7342ca1b1501bda3e74348f8ab17a5204fca4e4bd60f939e2db6e6903f2e1a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pure-wide peer review

Parent: model-execution. Owner: batch_design_review. Scope: ignored review artifacts only.

- [x] Read retained actual native/reference/build/fixture metadata and comparison/ownership sources.
- [x] Check small metadata/source hashes,527-source build closure,14 original child comparisons and ownership EOF/retirement.
- [x] Distinguish881,856 combined input/output words from outputs, external probes from trained producers and complete head projection from full-model logits.
- [x] Report no blocker within the bounded pure-operator scope.
- [ ] ROOT acceptance/merge and further full-model/StateRoot/GPU/serving qualification remain outside this review.

No test/build/provider/model/GPU execution by reviewer. No payload/binary/DSO reads/hashes; no MAIN/shared-plan changes.

````

## Source: .agents/runs/model-execution-qwen38-reference-2026-10-18/TODO.md

Original UTF-8 bytes: 1243; SHA256 `70ace0d25ee6d1ddf255a6818be49cb80cf560655e05a97ff6077f7f2110524c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Qwen3.8 reference gate — current implementation

Base: aaf9ba6
Owner plan: [model execution](../../plans/model-execution/TODO.md)

- [x] Indexed immutable source-bundle digest authenticates all whole shards;
  order independent, changed same-index learned bytes change digest.
- [x] Opt-in bounded source/epoch/coordinator/token-specific numeric trace
  with selected single GDN FP32 recurrent-state layer.
- [x] Pinned typed SHA256 JSON reference, exact BF16 comparison and
  distinct bounded FP32 tolerance, first stage/element mismatch.
- [x] Source-free analytical/adversarial reference and state tests,
  independent synthetic final logit=8 and BF16 result-cast repair.
- [x] Strict Clippy default/composed all-targets -D warnings,
  isolated scripts/check.sh 1,135 Rust pass/0 failure/69 ignored,
  238 suites and 18 Python fixtures; Nix source-integrity passed.
- [x] Approved Qwen3.5 .8B/2B direct-cache and HF-import independent
  release source parity pass, full 248320 logits/42 state domains
  and source/lease/session correctness.
- [ ] Scoped SSH-signed commit, no push/download/device launch.
- [ ] Later: trained Qwen3.8 checkpoint and independent real BF16/FLA
  captures + numerical/StateRoot comparison on capable host.

````

## Source: .agents/runs/model-execution-qwen38-statebridge-2026-10-18/TODO.md

Original UTF-8 bytes: 1343; SHA256 `5d29de4f05c91162ef6758aa9b5c901ac79a1d5e0dc4ed29d79254d02a4851d7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Qwen original source-bound StateRoot diagnostic — agent run

Base: 631e59f.
Plan: [model execution](../../plans/model-execution/TODO.md).
Scope: the existing TaskCoordinator admits a single source-verified
analytical GDN or full-attention layer; no new session owner or
multi-layer token loop.

- [x] Source-bound single-writer coordinator identity, with
  prior generic publisher unchanged.
- [x] One original Qwen layer maps typed ForwardPlan state slots to
  fixed-version state schemas, source and causal frontier.
- [x] Positive GDN 2-domain and full KV publication; immutable
  snapshots, competing/stale/foreign owner refusal, invalid stored
  bytes, truncated/nonfinite data and next-KV budget preflight.
- [x] Default and full CPU/cache/Qwen/acquisition Clippy -D warnings
  passed.
- [x] scripts/check.sh isolated frozen gate completed:
  1,116 Rust tests passed, 0 failed, 69 ignored across 238 suites
  plus release/installer, CPU comparators, 18 Python fixtures.
- [x] Independent Qwen3.5 .8B and 2B trained model
  release cache/direct 17.11s and HF-import 22.27s: full logits,
  all42 StateRoot domains, exact source/lease/session regression.
- [ ] Exact-path signed commit and post-commit Nix source integrity.
- [ ] Full original Qwen token schedule/BF16 model parity, SSD,
  memory placement and devices are future work.

````

## Source: .agents/runs/model-execution-qwen38-typed-oracle-20261020/README.md

Original UTF-8 bytes: 5981; SHA256 `371bcd6a4384d3d50722d6c554a70f46c9083f257715358f46004d47ec4f8b48`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B bounded public-operator reference

Prepared source only. Root alone reviews/runs; no Torch/operator/model/Cargo/GPU
execution by author. Parent:`.agents/plans/model-execution/TODO.md`.

Exact original `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
snapshot `98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e`.
The repository's config explicitly uses the existing Qwen3.5 mathematical
architecture. The repository/revision identity stays Qwen3.8; no substitution.

| Original role | Raw BF16 shape | Bytes | Source shard/absolute offset |
| --- | --- | ---: | --- |
| Embedding | [248320,5120] | 2542796800 | 3 /152 |
| InputNorm(0) | [5120] | 10240 | 1 /45400 |
| GdnQkv(0) | [10240,5120] | 104857600 | 1 /1120792 |

`source-pins.json` binds actual index/config SHA, exact original header SHA,
full shard length/SHA from prior independent whole-source verification, selected
raw matrix/norm SHA and original embedding rows11/353 SHA. Child inspected
bounded real headers/index/config and hashed selected100MiB matrix+10KiB norm
with1MiB buffers plus two10KiB rows. No full embedding/55GB payload scan.
Existing root source-window qualification scanned all55,563,006,776 weight-file
bytes, sampled actual owned rows and ended at ledger0; its report is linked and
hashed. That source-only PASS does not qualify mathematics or model execution.

The **original27B learned norm remains BF16**. The .8B restored36-F32-parameter
capture is a different source policy. This oracle never restores/upcasts learned
parameters before binding. Original RMS's own runtime `.float()` and
`1.0 + weight.float()` remain inside its unchanged forward implementation.

## Root-only oracle

- Genuine pinned stock0.31 image/Torch2.13/Transformers5.17/AVX2, CPU4/interop1.
  Import only after inner process affinity is set to four allowed CPUs; no GPU
  CDI/device/runtime exposure, network, model loader or foreign token engine.
- Open only original shards1/3 `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`; retain same FDs.
  Full SHA both selected shards before/after, exact original index/header/name/
  kind/shape/offset checks. A private demand-paged mmap of each original full
  shard supplies original contiguous raw tensors via `torch.frombuffer`.
  Full matrix geometry is retained; no fake header, selected-row embedding
  alias, matrix transpose, learned copy or dtype rewrite.
- For independent IDs11 and353, call actual `F.embedding` → unchanged original
  `Qwen3_5RMSNorm` → actual `F.linear` using original100MiB QKV matrix. Only an
  ordinary RMS module is instantiated; never the27B model. Original module file
  SHA and `_norm`/`forward` source entrypoints must match the pinned source;
  substituted hub kernels refuse.
- Each fragment runs plain then with metadata-only TorchDispatchMode; all three
  complete raw BF16 outputs must match exactly. No expectation fitting or
  intercepted arithmetic replacement. Save each complete actual output as
  exact BF16-widened F32LE, with raw/F32 SHA. No intermediate expected generator.
- Original raw matrix/norm/selected embedding mappings rehash unchanged; tensor
  mutation versions unchanged. Same original FD fingerprint and full source SHA
  rechecked after operators. Private tensor owners drop before normal mmap close;
  process exit owns mappings on failure. Completion metadata is written last.
- Runtime records actual CPU affinity, threads, source/compiled DSO hashes,
  actual mapped shared-library closure, RSS and elapsed diagnostic time. These
  observations are reference provenance, not production RAM/throughput proof.

Only two original shards are mapped:6,509,527,504 virtual bytes. No complete
55GB mapping or RAM load. Full embedding mapping does not materialize its2.54GB
payload as a Tensor copy; row selection touches two rows. Shard hashing scans
6.51GB sequentially twice through1MiB buffers; page cache is outside Rust
ResourceArbiter accounting and can affect observed cgroup/RSS behavior.
Only104.86MB selected QKV matrix/10KiB norm are numerical inputs beyond rows.
Complete output is40,960 words/163,840B, bounded1MiB. DSO-hash I/O has a finite
16GiB aggregate cap; one library<=2GiB and<=256 libraries. No library is changed.

Launcher:rootless owned nonce/name/CID; no shell; fixed exact image, pull never;
network none; memory12GiB/swap12GiB, CPU quota4, pids256, read-only root/source,
64MiB tmpfs,4MiB per-file/log cap, deadline default600s (30..900). Owned PGID
and independently verified owned container are cleaned/reaped before final
receipt I/O, including a final ownership/absence check after CLI reap. No
existing/unowned process is signalled. CPU affinity is enforced again inside
Python because OCI may reset inherited affinity; no cpuset controller changes.

```sh
# ROOT ONLY, after source review and coordinating host memory/CPU work.
python3 .agents/runs/model-execution-qwen38-typed-oracle-20261020/run_probe.py \
  --output .agents/runs/model-execution-qwen38-typed-oracle-20261020/root-reference-1 \
  --seconds 600
```

Success remains `REFERENCE_CAPTURE_ONLY_UNCOMPARED`. No Rust replay is prepared
against a fabricated complete TensorSource. The next legitimate integration
seams are in `WINDOW-INTEGRATION.md`; current guards remain closed.

Root acceptance next:independently check frozen script/source pins, run bounded
oracle, inspect actual resource/closure/source/output receipts and failures,
then implement/review/test sealed window operator adapters. Compare all10,240
embedding words +10,240 normalized words +20,480 QKV words exactly for both IDs
through the same Rust typed private fragment. Source-free foreign owner/role/
range/dtype/cancel/lease fixtures precede trained replay. No tolerance widening.
That eventual PASS would qualify these three original operators only. All64
layers, initial/cached GDN/FLA, full logits/all StateRoot domains, streaming
inference/physical RAM/GPU admission and matched serving benchmark remain open.

````

## Source: .agents/runs/model-execution-qwen38-typed-oracle-20261020/TODO.md

Original UTF-8 bytes: 1123; SHA256 `48fa43f211e316701db9c87e03037776761b43c09ece2d1a6c33a9b1bc77c323`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B bounded typed operator oracle

Parent:`.agents/plans/model-execution/TODO.md`; owner `/root/baseline_review`.

- [x] Read actual source-window PASS/source pins and current typed/binding/math APIs.
- [x] Verify bounded actual original config/index/headers and selected raw payload/row SHA.
- [x] Identify genuine BF16 dtype/OutputInput geometry and source36F32 distinction.
- [x] Document current rank1/whole-source/HostMath guard blockers and same-engine seams.
- [x] Prepare bounded CPU public F.embedding→unchanged original RMS→F.linear oracle.
- [x] Prepare exact-image finite owned launcher/cleanup and raw artifact validator.
- [ ] Independent frozen-source review.
- [ ] ROOT ONLY actual oracle execution and source/closure/resource/output inspection.
- [ ] Future reviewed sealed parameter/window resolver/row-range adapters +same-engine replay.
- [ ] Full27B logits/StateRoot/trained initial/cached GDN/FLA/streaming/GPU remain unqualified.

Author checks:source/byte/hash inspection and Python AST parse only; no Cargo,
rustc/C++compile/Torch/model/operator/GPU/container/runtime/test execution.

````

## Source: .agents/runs/model-execution-qwen38-typed-oracle-20261020/WINDOW-INTEGRATION.md

Original UTF-8 bytes: 6884; SHA256 `0d37357933c9110c5c987bc2bb03f6530ff42f1d39c683160cce5d9fd93a9b9e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Same-engine physical-window seam: design only

No production changes or admission. Current source at this preparation does
**not** execute this private27B fragment through windows.

## Existing seams and exact blockers

- `VerifiedTensorSource` owns authenticated original index/header descriptors,
  original SourceIdentity, opened ExtentSources and retained metadata admission.
  Root actual27B scan/row-byte test passed with final ledger0. Window getters
  expose original name/kind/full shape, axes, row range, source offsets and
  shard/extent digests. Owned `TensorWindow` retains source+stage until Drop.
- Its `matrix(name,OutputInput)` deliberately requires rank2. Therefore it
  currently refuses genuine InputNorm0 rank1; it is **not** a TensorSource and
  offers no complete payload borrow. Do not represent norm as a1×5120 matrix or
  sliced QKV as a complete tensor with a fake reduced original shape.
- `OriginalQwenBinding::bind` currently requires complete `SourceWeights` /
  `QwenOriginalText`. `LoadedQwenModel::from_original_source` preserves the4GiB
  owned-source guard and full original-HF execution/session/checkpoint refusal.
  HostMath directly borrows that loaded owner; `SourceParameter::validate_source`
  and matrix validation require actual full TensorSource references.
- Typed PlanBuilder can record metadata-only embedding/projection and pending
  ordinary RMS contracts. Recording does not resolve source bytes or issue a
  model execution permit. Current `output_first_projection_with_profile`
  validates a complete matrix payload; a partial window cannot honestly satisfy
  that API. Its shared row kernel is already the correct arithmetic reuse point.

## Small disjoint implementation slices after closed-op reference

1. **Artifact raw parameter descriptor/window.** Extend the existing source
   owner with a sealed rank1 parameter handle/accessor (name/kind/raw shape/
   exact bytes/original issuer); bounded owned parameter window shares the same
   staging/read/hash/cancel implementation and metadata lifetime. No converted
   gamma cache. Authentic SourceParameter validation compares full original
   metadata against this issuer before exposing10KiB raw BF16. Existing matrix
   row API and GGUF/owned-source APIs keep their contracts.
2. **Original operator-role metadata.** Reuse the original config/role binder
   against verified original metadata, preserving same source/index/bundle,
   all851 text-role dtype/shape closure and explicit profile. Existing loaded
   binding remains a complete-source variant; a window-backed variant is a
   private operator-source capability only. No new model/session/StateRoot owner
   or constructor that claims complete HF execution. Factor only metadata
   resolution, not a second layer/token schedule.
3. **Same MathematicalBackend/HostMath resolver.** Borrow either existing
   loaded source or sealed window operator-source through one private resolver.
   Reuse the exact shared forward embedding→InputNorm0→GdnQkv0 fragment; its
   role/profile/precision/epsilon checks remain. Keep existing loaded constructor
   and default GGUF paths unchanged. Window mode exposes only the qualified
   typed pure ops; unaware physical providers and unqualified state ops refuse
   before callbacks. Do not emulate a TensorSource with missing/fake payloads.
4. **Row-range adapter to the existing CPU kernel.** Carry the full immutable
   original SourceTensor descriptor+axes and explicit row range alongside the
   authentic TensorWindow. Validate same issuer/name/kind/full raw geometry,
   source byte offset/window range and packed row length. Add a closed row-range
   wrapper around existing `source_reference_projection_with_profile` /
   `dot_bf16_cpu_avx2`; no new arithmetic, transpose or altered full-width5120
   reduction. Ordinary MatVecRegion currently lacks original row offset/full
   descriptor, so do not publish a partial-row region as an unchanged whole
   original tensor. Preserve full10240-row identity at the adapter boundary.
5. **Private buffers/admission and qualification.** Reserve bounded private
   output/activation/scratch before allocation; retain stage while CPU reads it.
   Assemble complete10240-word output privately, final BF16 rounding in existing
   order, no publication. One forward fragment, no speculative/second engine.
   Root first runs source-free binding/geometry/lease/cancel/default refusal
   tests, then both complete original-token operator comparisons. The pending
   typed ordinary RMS patch must pass its own .8B authentic replay first.

## Concrete bounded lifetime for the selected fragment

With existing4MiB scan chunks and64MiB metadata envelope, keep one source issuer
for all tensors. Embedding row11 then353 each requests one full-width10,240B
original row; stage charges authenticated intersecting whole chunks, not just
10KiB. Drop each stage only after the synchronous exact BF16 widening finishes.
Norm10KiB can remain a separate small authentic stage while processing one row;
raw BF16→F32 widening/runtime+1 occurs inside accepted ordinary RMS, no adjusted
learned cache. One norm frame and BF16 output remain private.

Project QKV in fixed128-output-row ranges:1,310,720 requested bytes per range,
at most two4MiB chunks/8,388,656 charged stage bytes (includes retained offset
records). Preserve **all5120 input columns per row**; only output rows are
partitioned. After a synchronous chunk computation, private output words remain
and that chunk stage drops before the next read.80 ranges cover10240 rows.
This partitions transport only; exact equality to native full-matrix F.linear
must be demonstrated, not inferred from the formula. No full100MiB QKV copy.

Conservative configured pool proposal128MiB:64MiB retained metadata, up to8MiB+
QKV stage, up to4MiB+norm stage, and separately admitted<=1MiB private math
buffers; leave bounded headroom. Use actual `charged_staging_bytes` and class
snapshots, not requested-byte estimates. Generic ResourceArbiter control-plane,
allocator/Rayon/thread/runtime memory and OS page cache remain outside this
payload ledger. A configured pool is not independent physical free-RAM proof.

Pre-cancel before stage/read and between row ranges; synchronous IO must retire
before stage release. Cancel/error drops the complete private proposed output,
parameter/matrix windows, source metadata last; final all-owned ledger0. Tests
must retain a window after caller handles drop, reject foreign same-label issuer,
swapped role/shape/dtype/axes/row range, hash corruption and pool exhaustion.

Any future GPU adapter uses original queue/submission/BufferLease/Ready and
quarantine; retain incoming host window through actual upload completion. Unknown
completion cannot release source/staging or device lease. This preparation
implements or qualifies none of that GPU path.

````

## Source: .agents/runs/model-execution-qwen38-typed-oracle-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 5981; SHA256 `371bcd6a4384d3d50722d6c554a70f46c9083f257715358f46004d47ec4f8b48`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B bounded public-operator reference

Prepared source only. Root alone reviews/runs; no Torch/operator/model/Cargo/GPU
execution by author. Parent:`.agents/plans/model-execution/TODO.md`.

Exact original `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
snapshot `98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e`.
The repository's config explicitly uses the existing Qwen3.5 mathematical
architecture. The repository/revision identity stays Qwen3.8; no substitution.

| Original role | Raw BF16 shape | Bytes | Source shard/absolute offset |
| --- | --- | ---: | --- |
| Embedding | [248320,5120] | 2542796800 | 3 /152 |
| InputNorm(0) | [5120] | 10240 | 1 /45400 |
| GdnQkv(0) | [10240,5120] | 104857600 | 1 /1120792 |

`source-pins.json` binds actual index/config SHA, exact original header SHA,
full shard length/SHA from prior independent whole-source verification, selected
raw matrix/norm SHA and original embedding rows11/353 SHA. Child inspected
bounded real headers/index/config and hashed selected100MiB matrix+10KiB norm
with1MiB buffers plus two10KiB rows. No full embedding/55GB payload scan.
Existing root source-window qualification scanned all55,563,006,776 weight-file
bytes, sampled actual owned rows and ended at ledger0; its report is linked and
hashed. That source-only PASS does not qualify mathematics or model execution.

The **original27B learned norm remains BF16**. The .8B restored36-F32-parameter
capture is a different source policy. This oracle never restores/upcasts learned
parameters before binding. Original RMS's own runtime `.float()` and
`1.0 + weight.float()` remain inside its unchanged forward implementation.

## Root-only oracle

- Genuine pinned stock0.31 image/Torch2.13/Transformers5.17/AVX2, CPU4/interop1.
  Import only after inner process affinity is set to four allowed CPUs; no GPU
  CDI/device/runtime exposure, network, model loader or foreign token engine.
- Open only original shards1/3 `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`; retain same FDs.
  Full SHA both selected shards before/after, exact original index/header/name/
  kind/shape/offset checks. A private demand-paged mmap of each original full
  shard supplies original contiguous raw tensors via `torch.frombuffer`.
  Full matrix geometry is retained; no fake header, selected-row embedding
  alias, matrix transpose, learned copy or dtype rewrite.
- For independent IDs11 and353, call actual `F.embedding` → unchanged original
  `Qwen3_5RMSNorm` → actual `F.linear` using original100MiB QKV matrix. Only an
  ordinary RMS module is instantiated; never the27B model. Original module file
  SHA and `_norm`/`forward` source entrypoints must match the pinned source;
  substituted hub kernels refuse.
- Each fragment runs plain then with metadata-only TorchDispatchMode; all three
  complete raw BF16 outputs must match exactly. No expectation fitting or
  intercepted arithmetic replacement. Save each complete actual output as
  exact BF16-widened F32LE, with raw/F32 SHA. No intermediate expected generator.
- Original raw matrix/norm/selected embedding mappings rehash unchanged; tensor
  mutation versions unchanged. Same original FD fingerprint and full source SHA
  rechecked after operators. Private tensor owners drop before normal mmap close;
  process exit owns mappings on failure. Completion metadata is written last.
- Runtime records actual CPU affinity, threads, source/compiled DSO hashes,
  actual mapped shared-library closure, RSS and elapsed diagnostic time. These
  observations are reference provenance, not production RAM/throughput proof.

Only two original shards are mapped:6,509,527,504 virtual bytes. No complete
55GB mapping or RAM load. Full embedding mapping does not materialize its2.54GB
payload as a Tensor copy; row selection touches two rows. Shard hashing scans
6.51GB sequentially twice through1MiB buffers; page cache is outside Rust
ResourceArbiter accounting and can affect observed cgroup/RSS behavior.
Only104.86MB selected QKV matrix/10KiB norm are numerical inputs beyond rows.
Complete output is40,960 words/163,840B, bounded1MiB. DSO-hash I/O has a finite
16GiB aggregate cap; one library<=2GiB and<=256 libraries. No library is changed.

Launcher:rootless owned nonce/name/CID; no shell; fixed exact image, pull never;
network none; memory12GiB/swap12GiB, CPU quota4, pids256, read-only root/source,
64MiB tmpfs,4MiB per-file/log cap, deadline default600s (30..900). Owned PGID
and independently verified owned container are cleaned/reaped before final
receipt I/O, including a final ownership/absence check after CLI reap. No
existing/unowned process is signalled. CPU affinity is enforced again inside
Python because OCI may reset inherited affinity; no cpuset controller changes.

```sh
# ROOT ONLY, after source review and coordinating host memory/CPU work.
python3 .agents/runs/model-execution-qwen38-typed-oracle-20261020/run_probe.py \
  --output .agents/runs/model-execution-qwen38-typed-oracle-20261020/root-reference-1 \
  --seconds 600
```

Success remains `REFERENCE_CAPTURE_ONLY_UNCOMPARED`. No Rust replay is prepared
against a fabricated complete TensorSource. The next legitimate integration
seams are in `WINDOW-INTEGRATION.md`; current guards remain closed.

Root acceptance next:independently check frozen script/source pins, run bounded
oracle, inspect actual resource/closure/source/output receipts and failures,
then implement/review/test sealed window operator adapters. Compare all10,240
embedding words +10,240 normalized words +20,480 QKV words exactly for both IDs
through the same Rust typed private fragment. Source-free foreign owner/role/
range/dtype/cancel/lease fixtures precede trained replay. No tolerance widening.
That eventual PASS would qualify these three original operators only. All64
layers, initial/cached GDN/FLA, full logits/all StateRoot domains, streaming
inference/physical RAM/GPU admission and matched serving benchmark remain open.

````

## Source: .agents/runs/model-execution-qwen38-typed-oracle-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1123; SHA256 `48fa43f211e316701db9c87e03037776761b43c09ece2d1a6c33a9b1bc77c323`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B bounded typed operator oracle

Parent:`.agents/plans/model-execution/TODO.md`; owner `/root/baseline_review`.

- [x] Read actual source-window PASS/source pins and current typed/binding/math APIs.
- [x] Verify bounded actual original config/index/headers and selected raw payload/row SHA.
- [x] Identify genuine BF16 dtype/OutputInput geometry and source36F32 distinction.
- [x] Document current rank1/whole-source/HostMath guard blockers and same-engine seams.
- [x] Prepare bounded CPU public F.embedding→unchanged original RMS→F.linear oracle.
- [x] Prepare exact-image finite owned launcher/cleanup and raw artifact validator.
- [ ] Independent frozen-source review.
- [ ] ROOT ONLY actual oracle execution and source/closure/resource/output inspection.
- [ ] Future reviewed sealed parameter/window resolver/row-range adapters +same-engine replay.
- [ ] Full27B logits/StateRoot/trained initial/cached GDN/FLA/streaming/GPU remain unqualified.

Author checks:source/byte/hash inspection and Python AST parse only; no Cargo,
rustc/C++compile/Torch/model/operator/GPU/container/runtime/test execution.

````

## Source: .agents/runs/model-execution-qwen38-typed-oracle-20261020/frozen-prepared/WINDOW-INTEGRATION.md

Original UTF-8 bytes: 6884; SHA256 `0d37357933c9110c5c987bc2bb03f6530ff42f1d39c683160cce5d9fd93a9b9e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Same-engine physical-window seam: design only

No production changes or admission. Current source at this preparation does
**not** execute this private27B fragment through windows.

## Existing seams and exact blockers

- `VerifiedTensorSource` owns authenticated original index/header descriptors,
  original SourceIdentity, opened ExtentSources and retained metadata admission.
  Root actual27B scan/row-byte test passed with final ledger0. Window getters
  expose original name/kind/full shape, axes, row range, source offsets and
  shard/extent digests. Owned `TensorWindow` retains source+stage until Drop.
- Its `matrix(name,OutputInput)` deliberately requires rank2. Therefore it
  currently refuses genuine InputNorm0 rank1; it is **not** a TensorSource and
  offers no complete payload borrow. Do not represent norm as a1×5120 matrix or
  sliced QKV as a complete tensor with a fake reduced original shape.
- `OriginalQwenBinding::bind` currently requires complete `SourceWeights` /
  `QwenOriginalText`. `LoadedQwenModel::from_original_source` preserves the4GiB
  owned-source guard and full original-HF execution/session/checkpoint refusal.
  HostMath directly borrows that loaded owner; `SourceParameter::validate_source`
  and matrix validation require actual full TensorSource references.
- Typed PlanBuilder can record metadata-only embedding/projection and pending
  ordinary RMS contracts. Recording does not resolve source bytes or issue a
  model execution permit. Current `output_first_projection_with_profile`
  validates a complete matrix payload; a partial window cannot honestly satisfy
  that API. Its shared row kernel is already the correct arithmetic reuse point.

## Small disjoint implementation slices after closed-op reference

1. **Artifact raw parameter descriptor/window.** Extend the existing source
   owner with a sealed rank1 parameter handle/accessor (name/kind/raw shape/
   exact bytes/original issuer); bounded owned parameter window shares the same
   staging/read/hash/cancel implementation and metadata lifetime. No converted
   gamma cache. Authentic SourceParameter validation compares full original
   metadata against this issuer before exposing10KiB raw BF16. Existing matrix
   row API and GGUF/owned-source APIs keep their contracts.
2. **Original operator-role metadata.** Reuse the original config/role binder
   against verified original metadata, preserving same source/index/bundle,
   all851 text-role dtype/shape closure and explicit profile. Existing loaded
   binding remains a complete-source variant; a window-backed variant is a
   private operator-source capability only. No new model/session/StateRoot owner
   or constructor that claims complete HF execution. Factor only metadata
   resolution, not a second layer/token schedule.
3. **Same MathematicalBackend/HostMath resolver.** Borrow either existing
   loaded source or sealed window operator-source through one private resolver.
   Reuse the exact shared forward embedding→InputNorm0→GdnQkv0 fragment; its
   role/profile/precision/epsilon checks remain. Keep existing loaded constructor
   and default GGUF paths unchanged. Window mode exposes only the qualified
   typed pure ops; unaware physical providers and unqualified state ops refuse
   before callbacks. Do not emulate a TensorSource with missing/fake payloads.
4. **Row-range adapter to the existing CPU kernel.** Carry the full immutable
   original SourceTensor descriptor+axes and explicit row range alongside the
   authentic TensorWindow. Validate same issuer/name/kind/full raw geometry,
   source byte offset/window range and packed row length. Add a closed row-range
   wrapper around existing `source_reference_projection_with_profile` /
   `dot_bf16_cpu_avx2`; no new arithmetic, transpose or altered full-width5120
   reduction. Ordinary MatVecRegion currently lacks original row offset/full
   descriptor, so do not publish a partial-row region as an unchanged whole
   original tensor. Preserve full10240-row identity at the adapter boundary.
5. **Private buffers/admission and qualification.** Reserve bounded private
   output/activation/scratch before allocation; retain stage while CPU reads it.
   Assemble complete10240-word output privately, final BF16 rounding in existing
   order, no publication. One forward fragment, no speculative/second engine.
   Root first runs source-free binding/geometry/lease/cancel/default refusal
   tests, then both complete original-token operator comparisons. The pending
   typed ordinary RMS patch must pass its own .8B authentic replay first.

## Concrete bounded lifetime for the selected fragment

With existing4MiB scan chunks and64MiB metadata envelope, keep one source issuer
for all tensors. Embedding row11 then353 each requests one full-width10,240B
original row; stage charges authenticated intersecting whole chunks, not just
10KiB. Drop each stage only after the synchronous exact BF16 widening finishes.
Norm10KiB can remain a separate small authentic stage while processing one row;
raw BF16→F32 widening/runtime+1 occurs inside accepted ordinary RMS, no adjusted
learned cache. One norm frame and BF16 output remain private.

Project QKV in fixed128-output-row ranges:1,310,720 requested bytes per range,
at most two4MiB chunks/8,388,656 charged stage bytes (includes retained offset
records). Preserve **all5120 input columns per row**; only output rows are
partitioned. After a synchronous chunk computation, private output words remain
and that chunk stage drops before the next read.80 ranges cover10240 rows.
This partitions transport only; exact equality to native full-matrix F.linear
must be demonstrated, not inferred from the formula. No full100MiB QKV copy.

Conservative configured pool proposal128MiB:64MiB retained metadata, up to8MiB+
QKV stage, up to4MiB+norm stage, and separately admitted<=1MiB private math
buffers; leave bounded headroom. Use actual `charged_staging_bytes` and class
snapshots, not requested-byte estimates. Generic ResourceArbiter control-plane,
allocator/Rayon/thread/runtime memory and OS page cache remain outside this
payload ledger. A configured pool is not independent physical free-RAM proof.

Pre-cancel before stage/read and between row ranges; synchronous IO must retire
before stage release. Cancel/error drops the complete private proposed output,
parameter/matrix windows, source metadata last; final all-owned ledger0. Tests
must retain a window after caller handles drop, reject foreign same-label issuer,
swapped role/shape/dtype/axes/row range, hash corruption and pool exhaustion.

Any future GPU adapter uses original queue/submission/BufferLease/Ready and
quarantine; retain incoming host window through actual upload completion. Unknown
completion cannot release source/staging or device lease. This preparation
implements or qualifies none of that GPU path.

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/API.md

Original UTF-8 bytes: 3645; SHA256 `cf6c65a4695a3e3748293ba6d73073efa0357bc9ae1135aa73402dc0789febee`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original window pure roles

Author: spine_review. Source-only candidate; uncompiled/unexecuted.

- Same `OriginalWindowOperators`/full config-index-header closure, original issuer,
  same HostMath/PlanBuilder/CPU helpers. No alternate model/layer/token loop.
- `append_projection_slice(builder, QwenMatrixRole, existing PlanValue)`;
  `append_ordinary_rms_slice(builder, QwenParameterRole, existing PlanValue)`.
  Existing same-builder/source typed producer is required; no external plan node.
- `execute_projection_slice(role, PureSliceInput, cancelled)` and
  `execute_ordinary_rms_slice(role, PureSliceInput, cancelled)` return retained
  `PureSliceOutput` with source/profile/role, borrowed input/output slices only.
- Input is external finite raw BF16 bytes; identity/profile/representation/width
  checked. It does not authenticate its caller's producer/archive provenance.
- Projections: every originally bound matrix role except Embedding, including
  GDN QKV/Z/A/B/out, attention full Q+gate/K/V/out, MLP gate/up/down and tied or
  untied Output. Exact original name/axes/kind/full shape comes from the sealed
  config/index/header issuer. CPU columns<=17408/rows<=248320; no GPU guard change.
- Ordinary hidden norms: exact InputNorm/FfnNorm/FinalNorm roles, raw BF16/F32
  gamma and original runtime(1+weight), existing epsilon1e-6 and widths only.
  Existing attention Q/K norms retain their accepted separate shape checks.
- Other GDN parameters, gated norm BF16-gamma composition, coefficient/exp,
  initial/cached state,112 StateRoots, sessions/full HF execution stay closed.

## Payload admission

- Same source host domain/owner admits before each input/output allocation.
  Retained source issuer checked before an input can feed a second operator.
- Existing64KiB Output/960KiB Scratch conservative floors retained; output grows
  by checked `4*actual_elements`. Projection Scratch grows by checked
  `4*(rows+128)` for full raw output plus row accumulator while final output
  remains allocated. Source row-window data/offsets keep their separate leases.
- Ordinary RMS Scratch derives5 full vectors+2 weight vectors+4 row vectors from
  the unchanged diagnostic frame, then uses the existing conservative floor.
- Wide raw/final projection arrays explicitly reserve exact capacity and refuse
  larger capacity. RNE uses the SAME helper/row order in an explicit loop;
  Result<Vec> implicit capacity growth is not assumed to fit the ticket.
- Untied248320-row head: Output993280B/Scratch993792B, plus independent input and
  actual staged row-window leases. It is not capped at the old64KiB output floor.
- Returned backing retains source/ticket through final drop; no Vec escape.
  This is the existing payload accounting contract, not whole-host availability.

## ROOT gates

- Eight source-free tests prepared: every role/real complete fixture, raw norm
  roles/closed GDN parameters, existing producer plan chain/no domains, actual
  ticket growth/overflow, bad input/role/pool/cancel, final-owner retention,
  unaware provider before callback, same-content foreign issuer refusal.
- ROOT compile/strict default+CPU+original binding/GPU cold refusal regressions,
  existing window/attention/conv and source parity, full isolated gate required.
- Genuine original27B F.linear/ordinary norm public oracle + native full output
  replay still needed, especially17408-wide down and248320-row untied head.
  A trained role's source bytes do not establish its activation's graph producer.
- No runtime, numerical qualification, full source/model support or throughput
  follows from source review or these synthetic fixtures.

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/TODO.md

Original UTF-8 bytes: 1718; SHA256 `41a3dbe65b5935be24749de9bb9ef78355da2eb325ed2b681464ef437ab90fe0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B window role continuation

Parent: existing model-execution/shared-engine plan. Owner: spine_review.
Scope: own ignored source-only candidate/preparation; no MAIN/plans/commits.

- [x] Audit accepted full-header role closure, HostMath/backend, window CPU math.
- [x] Preserve exact five existing source baselines; add only eight owned paths.
- [x] Extend pure projection/ordinary hidden-norm slices on the same schedule.
- [x] Derive/grow actual output/scratch payload before allocations; retain owners.
- [x] Prepare eight initial meaningful source-free ownership/admission/type/role fixtures.
- [x] Rustfmt proposed source only; no compile/test/runtime/math execution.
- [ ] Independent source review and ROOT composition/compile/strict checks.
- [x] Freeze genuine original27B bounded public oracle/native replay preparation.
- [x] Add exacthidden recorder guard + valid2hidden producer negative (V2).
- [x] Add completed-output cancel boundary + actual private-result lastdrop test (V3).
- [ ] ROOT source/focused/strict review of latest V3 (ten prepared Rust tests).
- [ ] ROOT actual trained-source wide-role public/native bit qualification.
- [ ] ROOT accepted old window/attention/conv/source parity/full isolated gates.
- [ ] Full27B GDN math/112 StateRoots/session/model qualification remains closed.

Failures: initial own ignored window-child write lacked its new directory; mkdir
then wrote it. No MAIN bytes touched. Static self-audit removed a private core
method call and placed explicit input-capacity guard in its intended new helper.
No candidate code or fixtures executed. V3 raw transport followup was prepared
separately under the preexisting original27B GDN run at ROOT's priority.

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/qualification/ROOT.md

Original UTF-8 bytes: 5511; SHA256 `17bf7285ebaa0eb3f761d107fe7d4fcc16ceef5e5a2e333bbb73f192df6c5495`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only genuine original27B pure-role gate

Author: spine_review. Prepared source only; no fixtures/build/model/operator run.

## Scope and provenance

- Original snapshot98cea13c/revision1d4bf0f2, all18 acquired weight shards
  55,563,006,776B; actual index1199/ordinary-text851 closure. Reused accepted
  `factory.rs` byte-identical from the original27B three-operator replay.
- Public PyTorch original BF16 F.linear: GDN QKV/Z/A/B/out; attention full Q+gate,
  K/V/out; MLP gate/up/down; real untied lm_head. Three original RMSNorm calls:
  input0, post-attention0 and final. No nonlinearities, residuals, positions,
  caches, token/layer traversal or state publishers are added.
- Two external teacher probes11/353. Projection hidden5120 inputs are exact
  accepted archived original InputNorm0 BF16 words; norm inputs use accepted
  original embedding words. Only the GDN input prefix has its actual trained
  producer here. Other hidden-role inputs remain external probes. Nonhidden
  widths6144/17408 use explicitly literal finite nonzero BF16 patterns, never
  reconstructed or declared trained producer activations.
- Sixteen actual source/header role shapes, unchanged raw BF16, three original
  retained shard FDs. Each parameter mapping is exact page-aligned range plus
  leading alignment, never a whole shard/full55GB mapping. Original head alone
  is about2.543GB virtual. All parameter FD/map/tensor owners survive to exit;
  full source/hash/version/readset checks before/after actual public operators.
- Public capture calls each genuine operator twice, all input/output bits
  identical. This is a repeat control, no newly invented instrumentation route.
  Pinned original image/Torch2.13cf30153/Transformers5.17 source, AVX2/threads4;
  CPU loader/activation profile is distinct from vLLM CUDA/FLA.
- Exact parameter VA3,519,025,784B +6GiB runtime VA estimate +2GiB operator
  estimate +4MiB capture <12GiB RLIMIT_AS. RAM/totalRAM+swap12GiB,CPU quota4,
  noGPU/network, read-only source/probe/image, finite600s child/2400s parent,
  32MiB whole artifacts. Estimates/cgroup containment do not prove complete
  physical host/provider admission. Mapping VA is not resident RSS.

## Execution, after review and timing hold

1. ROOT run `python3 test_contracts.py` here (four prepared stdlib bounds tests).
   Run the separate original27B GDN V3 actual-runtime raw serializer fixtures
   first; tensor-bit helper body is the same equal-size integer view/C-order
   transport with only its local support import changed.
2. ROOT `python3 run_reference.py --new-output /ABSOLUTE/QUAL/root-reference-1`.
   Actual full55GB source hash scans happen only under ROOT's quiet scheduling.
3. ROOT `python3 validate_reference.py --output /ABSOLUTE/QUAL/root-reference-1`.
   Independent stdlib full raw/header/readset/teacher/count/lifetime validator.
   All32 cases/all inputs/all output words must pass; no tolerance.
4. ROOT merges reviewed role candidate0f9 plus exact-hidden65cd and completed-cancel7759 followups, runs
   strict/focused/default/source-window regressions; builds this Cargo.toml only
   under its one-coordinator/offline actual-source isolation. No author Cargo.
5. ROOT writes actual build attestation (not fabricated by these preparations):
   status `ROOT_COMPILED_ORIGINAL27B_PURE_ROLES_ONLY`, actual binary/path/hash,
   build_exit_code0, explicit absolute root_worktree, actual_source_files eight
   path/bytes/SHA entries matching original+followup source, candidate_manifest
   0f9f2a8f…,exact_hidden_followup_manifest65cd4364…,completed_cancel_followup_manifest7759599f…. Preserve actual compile
   command/log/effective lock/artifact-source proof. These labels/caller hashes
   alone are not independent evidence of compilation or correct arithmetic.
6. ROOT `python3 run_native.py --binary ABS --binary-sha256 SHA \
   --reference-directory /ABSOLUTE/QUAL/root-reference-1 \
   --build-attestation ABS --new-output /ABSOLUTE/QUAL/root-native-1`.

## Native proof boundary

- SAME public OriginalWindowOperators pure APIs; SAME accepted source factory,
  SourceWeights windows, HostMath/CPU64/RNE/ordinary norm. No replacement engine.
- Full32-case input/output bit comparisons. Full248320-row untied head and
 17408-wide down included; every signed-zero bit is compared without tolerance.
- Real configured original ResourceArbiter128MiB host domain, source metadata
 64MiB, qualifier-control8MiB, model metadata4MiB, actual per-op output/scratch
  and row-window data+offset tickets. No claim128MiB is physical free host RAM.
- Actual pre-cancel and real pool exhaustion refuse; every temporary owner
  releases, no device fence. Last actual head result holds source/input/output
  leases after caller/source drop; final all payload/source owner drop0.
- Native process512MiB AS,CPU4/noGPU/no provider, bounded1800s/4MiB logs.
  Exact accepted owned-v4 generic WNOWAIT group cleanup function reused, two
  independent KILL attempts before any reap, complete group absence/EOF.
- PASS is only private source-authentic pure-operator numerical/transport scope.
  No all-role trained activation producer, GDN BF16 gamma/Alog/exp/initial/cached
  math,112 StateRoots, session/full27B model, GPU eligibility or speed claim.

Source-only author validation: Python ASTs/Rustfmt/small source/metadata pin and
header reads. No weight/DSO payload hashes, fixture/Torch/operator/model/native/
container/build/GPU execution. All failed/interrupted ROOT attempts stay intact.

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/qualification/frozen-prepared/ROOT.md

Original UTF-8 bytes: 5511; SHA256 `17bf7285ebaa0eb3f761d107fe7d4fcc16ceef5e5a2e333bbb73f192df6c5495`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only genuine original27B pure-role gate

Author: spine_review. Prepared source only; no fixtures/build/model/operator run.

## Scope and provenance

- Original snapshot98cea13c/revision1d4bf0f2, all18 acquired weight shards
  55,563,006,776B; actual index1199/ordinary-text851 closure. Reused accepted
  `factory.rs` byte-identical from the original27B three-operator replay.
- Public PyTorch original BF16 F.linear: GDN QKV/Z/A/B/out; attention full Q+gate,
  K/V/out; MLP gate/up/down; real untied lm_head. Three original RMSNorm calls:
  input0, post-attention0 and final. No nonlinearities, residuals, positions,
  caches, token/layer traversal or state publishers are added.
- Two external teacher probes11/353. Projection hidden5120 inputs are exact
  accepted archived original InputNorm0 BF16 words; norm inputs use accepted
  original embedding words. Only the GDN input prefix has its actual trained
  producer here. Other hidden-role inputs remain external probes. Nonhidden
  widths6144/17408 use explicitly literal finite nonzero BF16 patterns, never
  reconstructed or declared trained producer activations.
- Sixteen actual source/header role shapes, unchanged raw BF16, three original
  retained shard FDs. Each parameter mapping is exact page-aligned range plus
  leading alignment, never a whole shard/full55GB mapping. Original head alone
  is about2.543GB virtual. All parameter FD/map/tensor owners survive to exit;
  full source/hash/version/readset checks before/after actual public operators.
- Public capture calls each genuine operator twice, all input/output bits
  identical. This is a repeat control, no newly invented instrumentation route.
  Pinned original image/Torch2.13cf30153/Transformers5.17 source, AVX2/threads4;
  CPU loader/activation profile is distinct from vLLM CUDA/FLA.
- Exact parameter VA3,519,025,784B +6GiB runtime VA estimate +2GiB operator
  estimate +4MiB capture <12GiB RLIMIT_AS. RAM/totalRAM+swap12GiB,CPU quota4,
  noGPU/network, read-only source/probe/image, finite600s child/2400s parent,
  32MiB whole artifacts. Estimates/cgroup containment do not prove complete
  physical host/provider admission. Mapping VA is not resident RSS.

## Execution, after review and timing hold

1. ROOT run `python3 test_contracts.py` here (four prepared stdlib bounds tests).
   Run the separate original27B GDN V3 actual-runtime raw serializer fixtures
   first; tensor-bit helper body is the same equal-size integer view/C-order
   transport with only its local support import changed.
2. ROOT `python3 run_reference.py --new-output /ABSOLUTE/QUAL/root-reference-1`.
   Actual full55GB source hash scans happen only under ROOT's quiet scheduling.
3. ROOT `python3 validate_reference.py --output /ABSOLUTE/QUAL/root-reference-1`.
   Independent stdlib full raw/header/readset/teacher/count/lifetime validator.
   All32 cases/all inputs/all output words must pass; no tolerance.
4. ROOT merges reviewed role candidate0f9 plus exact-hidden65cd and completed-cancel7759 followups, runs
   strict/focused/default/source-window regressions; builds this Cargo.toml only
   under its one-coordinator/offline actual-source isolation. No author Cargo.
5. ROOT writes actual build attestation (not fabricated by these preparations):
   status `ROOT_COMPILED_ORIGINAL27B_PURE_ROLES_ONLY`, actual binary/path/hash,
   build_exit_code0, explicit absolute root_worktree, actual_source_files eight
   path/bytes/SHA entries matching original+followup source, candidate_manifest
   0f9f2a8f…,exact_hidden_followup_manifest65cd4364…,completed_cancel_followup_manifest7759599f…. Preserve actual compile
   command/log/effective lock/artifact-source proof. These labels/caller hashes
   alone are not independent evidence of compilation or correct arithmetic.
6. ROOT `python3 run_native.py --binary ABS --binary-sha256 SHA \
   --reference-directory /ABSOLUTE/QUAL/root-reference-1 \
   --build-attestation ABS --new-output /ABSOLUTE/QUAL/root-native-1`.

## Native proof boundary

- SAME public OriginalWindowOperators pure APIs; SAME accepted source factory,
  SourceWeights windows, HostMath/CPU64/RNE/ordinary norm. No replacement engine.
- Full32-case input/output bit comparisons. Full248320-row untied head and
 17408-wide down included; every signed-zero bit is compared without tolerance.
- Real configured original ResourceArbiter128MiB host domain, source metadata
 64MiB, qualifier-control8MiB, model metadata4MiB, actual per-op output/scratch
  and row-window data+offset tickets. No claim128MiB is physical free host RAM.
- Actual pre-cancel and real pool exhaustion refuse; every temporary owner
  releases, no device fence. Last actual head result holds source/input/output
  leases after caller/source drop; final all payload/source owner drop0.
- Native process512MiB AS,CPU4/noGPU/no provider, bounded1800s/4MiB logs.
  Exact accepted owned-v4 generic WNOWAIT group cleanup function reused, two
  independent KILL attempts before any reap, complete group absence/EOF.
- PASS is only private source-authentic pure-operator numerical/transport scope.
  No all-role trained activation producer, GDN BF16 gamma/Alog/exp/initial/cached
  math,112 StateRoots, session/full27B model, GPU eligibility or speed claim.

Source-only author validation: Python ASTs/Rustfmt/small source/metadata pin and
header reads. No weight/DSO payload hashes, fixture/Torch/operator/model/native/
container/build/GPU execution. All failed/interrupted ROOT attempts stay intact.

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/qualification/metadata-v2/ROOT.md

Original UTF-8 bytes: 1748; SHA256 `64afc576056761f54db128a3229c453e6334c94b36bdca143f58f783df05b360`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive native source metadata binding

Author: spine_review. Source only; original thirteen-file16809776 unchanged.

- Added qualification/source-pin.json: exact accepted7058B source-report4c74fa28.
  Its path now matches unchanged factory.rs `CARGO_MANIFEST_DIR/source-pin.json`;
  no source/model/weights/role or math mutation. Original factory REPORT_SHA stays.
- Build the existing qualification/Cargo.toml from that exact directory. Include
  four entries Cargo.toml/main.rs/factory.rs/source-pin.json in actual build
  attestation qualification_source_files, plus qualification_directory,
  qualification_manifest_sha25616809776 and
  qualification_metadata_followup_manifest_sha256 from this layer. Preserve all
  original implementation/compile/log/lock/binary evidence; labels are not proof.
- Execute run_native_v2.py with the original run_native.py arguments. It validates
  this additive layer and all four compile inputs before/after delegating the
  unchanged original owner/ledger/full-word replay. ROOT runs three prepared
  test_native_metadata_v2.py tests first; author AST only, no fixtures/runtime.

The original12GiB AS/RAM cap stays. Mapped3,519,025,784B is exact;6GiB runtime and
2GiB operator allowances are estimates, not a source-backed peak bound. Pinned
Torch cf30153 Linear.cpp→LinearAlgebra.cpp maps this contiguous BF16[1,5120]
head to CPUBlas(T,N,248320,1,5120). The noAVX512BF16 CPU64 route uses per-row
BF16 inputs/FP32 register reduction, not a full5,085,593,600B F32 weight clone.
Actual compiled route/provider/library/allocator/thread overhead and head peak
are unqualified until ROOT executes under unchanged containment. No cap widening
or public-operator replacement is authorized by these estimates.

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/qualification/metadata-v2/TODO.md

Original UTF-8 bytes: 645; SHA256 `044a096fd331078dec8f7bea4df51f8076cb664bf050ccb9f0eedde317eb4426`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native metadata followup

Parent model-execution; owner spine_review; ignored source-only scope.

- [x] Exact accepted7058B4c74 report now at unchanged factory's required localpath.
- [x] Freeze additive e958 four-input/build-directory/original+newmanifest binding.
- [x] Baseline peer review no blocker; original13/1680 unchanged.
- [x] Audit pinned BF16 CPU head path; no full-F32weight cast in TN CPU64 source.
- [x] Keep12GiB containment; runtime/operator allowances remain unqualified estimates.
- [ ] ROOT three metadata fixtures, actual build, public/native32case/fullgate.

No author fixture/build/runtime/operator or payload hashing.

````

## Source: .agents/runs/model-execution-qwen38-window-roles-20261020/qualification/metadata-v2/frozen/metadata-v2/ROOT.md

Original UTF-8 bytes: 1748; SHA256 `64afc576056761f54db128a3229c453e6334c94b36bdca143f58f783df05b360`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive native source metadata binding

Author: spine_review. Source only; original thirteen-file16809776 unchanged.

- Added qualification/source-pin.json: exact accepted7058B source-report4c74fa28.
  Its path now matches unchanged factory.rs `CARGO_MANIFEST_DIR/source-pin.json`;
  no source/model/weights/role or math mutation. Original factory REPORT_SHA stays.
- Build the existing qualification/Cargo.toml from that exact directory. Include
  four entries Cargo.toml/main.rs/factory.rs/source-pin.json in actual build
  attestation qualification_source_files, plus qualification_directory,
  qualification_manifest_sha25616809776 and
  qualification_metadata_followup_manifest_sha256 from this layer. Preserve all
  original implementation/compile/log/lock/binary evidence; labels are not proof.
- Execute run_native_v2.py with the original run_native.py arguments. It validates
  this additive layer and all four compile inputs before/after delegating the
  unchanged original owner/ledger/full-word replay. ROOT runs three prepared
  test_native_metadata_v2.py tests first; author AST only, no fixtures/runtime.

The original12GiB AS/RAM cap stays. Mapped3,519,025,784B is exact;6GiB runtime and
2GiB operator allowances are estimates, not a source-backed peak bound. Pinned
Torch cf30153 Linear.cpp→LinearAlgebra.cpp maps this contiguous BF16[1,5120]
head to CPUBlas(T,N,248320,1,5120). The noAVX512BF16 CPU64 route uses per-row
BF16 inputs/FP32 register reduction, not a full5,085,593,600B F32 weight clone.
Actual compiled route/provider/library/allocator/thread overhead and head peak
are unqualified until ROOT executes under unchanged containment. No cap widening
or public-operator replacement is authorized by these estimates.

````

## Source: .agents/runs/model-execution-root-bf16-composition-20261020/TODO.md

Original UTF-8 bytes: 2739; SHA256 `81b1e8f4a577033986596939a76a94ca60e7649cce7b0da5583dcba487ab45b4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT original BF16 composition

- Parent: model-execution. Owner: ROOT.
- Scope: compose previously peer-reviewed pure wide roles, coefficients, cached H16/H48 and BF16-gamma operator fragments in an isolated ROOT-owned worktree.
- MAIN456-source frozen while physical worker/HTTP regressions run. No MAIN edits or GPU runtime here. ROOT remains sole Cargo/test/GPU coordinator.
- [ ] Verify original/additive source pins and compose narrow deltas, preserving F32/default paths.
- [ ] Focused CPU/core/model original-HF tests +strict checks.
- [ ] Authentic original reference replays with owned child lifetimes/source windows/admission/cancel/retention/final0.
- [ ] Review, merge only owned bytes with fresh timestamps after current GPU phase closes; full original gates before acceptance.
- Fullmodel27B/112domains/GPU/nativeprovider not qualified by fragments.

## Actual ROOT continuation 2026-10-20 04:28 UTC

- Detached own worktree created at FFF1842; MAIN456-source GPU worker/HTTP closure unchanged.
- Composed wide0f9f+65cd+7759, cached57a4+stride71e5+H48metadata26d9, coefficientd744, gamma43c4+7f26. Parent86local source files verified; pure new arithmetic retained exact original bytes. Core/module EOF/cold-refusal intersections reconciled explicitly; journals retained.
- Source freezes1–7 and all failed actual attempts retained. Failure1: two old prepare_output callers after wide API gained sizes; ROOT adapter uses2*heads/3*heads/1scratch under same old64KiB/960KiB floors. Failures2–4: strict test-only Vec→array, private-test constant refs, unused import. Attempt5 strictPASS/237suite partial: eight modelfixture failures, fixed valid2-layer hybrid schedules, real<=32MiB pressure chunks, supported-FFN oldnegative supersession and correct minted source identity. Attempt6 strictPASS/model69pass2fail: tiedEmbedding/Output shared descriptor must reject explicitrole API; diagnosticVerifiedParameter stillowned32MiB, explicitdropbeforefinal0. No production arithmetic or acceptance limits changed.
- FINAL actual19636/root-cpu-focused-7.json EXIT0: strict core/CPU/model alltargets originalHF PASS;237passed/0failed/2ignored (100core/66CPU/71model), all521postsource exact. Freeze7 036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615. ROOT read shared diffs; Spine independently reviewed fullcomposition through5 and6fixes;7narrowreview requested.
- Relocated explicit ISOLATED-source qualification harness preparation bySpine pending; original bundles retain MAIN-path contracts and cannot be relabeled merged. Real raw source replays, default/GPU strict checks, full isolatedgate and original parityroutes remain before merge/acceptance. No MAIN model changes/commit or full27B inference claim.

````

## Source: .agents/runs/model-execution-slow-gdn-observation-2026-10-20/OPERATOR_GAPS.md

Original UTF-8 bytes: 7658; SHA256 `b48fb97ea4d00f5c63da3f777301376f8c37a1f593afda28c7d3d0de9be09b6b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Actual pinned slow-GDN operator evidence

Owner: batch_design_review. Parent: model-execution.
No production/worktree edits, Cargo, GPU, arithmetic replacement or rounding
oracle. All files belong to this ignored run.

## Executed evidence

- Stock image `docker.io/vllm/vllm-openai@sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f`.
- Torch2.13.0+cu129, Transformers5.17.0; all5 installed runtime source hashes
  match the prior pin. Modeling SHA256
  `762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`.
- Original Qwen/Qwen3.5-0.8B revision
  `2fc06364715b967f1860aea9cf38778875588b17`, all source-receipt files hashed
  before/after;473 loaded ordinary parameters, exact36originalF32 restored,
  remaining437BF16, authentic tied head.
- `run_probe.py`: network none, no GPU devices, CUDA_VISIBLE_DEVICES=-1,
  read-only source and reference mounts,12GiB memory,4CPU/threads4, interop1.
- Successful invocation exit0,17.18s diagnostic elapsed, maxRSS2,318,584KiB.
  Tokens11,353 produce actual argmax198,599. All49 raw prior-capture tensors
  per step (complete248320 logits +all raw cache tensors) bit-identical:
  98 checks. Existing reference manifest unchanged.
- `captured-gdn/step-{0,1}/actual-gdn-frames.safetensors`:50/40 observed
  tensors,12,448,160/6,468,192 payload bytes, total18,916,352 (<64MiB).
  Every tensor retains actual dtype, shape, original stride and raw hash.
  Archive SHA256s:
  - step0: `a493dd31bf1c511edee9198ac4e8eb30f7141f79207d2ce836dfb8ef02ad0450`.
  - step1: `c8b7c72d0644c418dfa5c74e974678f7ee76e3d7cc9e83a7c157c3a9f836c0fa`.
- Observer uses selective `sys.settrace` call/line/return events. Functions,
  inputs and results are untouched; only existing tensors are cloned. Line
  events record locals before their named source line. No standalone
  recurrent beta*v exists in this source; actual chunk v_beta is captured.
- `validate_capture.py` exit0: archive/metadata/hash closure, all words finite,
  actual l2 returns match parent normalized q/k,63pad positions canonical+0,
  recurrent token2 input-state bits equal chunk token1 return-state bits.
- First attempt retained: forward/prior comparisons passed, then raw byte
  serialization failed on a singleton tensor's non-unit last stride. Fixed
  observer serialization by flattening its copy; no model arithmetic changed.
  All retained tensor archives including failure total31,379,200B (<64MiB).

## Observed operator contract

- Layer0 input q/k/v BF16 `[1,1,16,128]`; beta BF16 `[1,1,16]`, g FP32
  `[1,1,16]`. First token inputs are strided actual conv views
  `[8192,8192,512,4]`; second token is contiguous. Original function transposes
  and converts all five inputs to contiguous FP32 before arithmetic.
- Actual normalized q/k FP32 `[1,16,1,128]`, from separate l2norm calls;
  actual inv_norm and returned normalized vectors captured.
- First token: query scaling precedes pad1→64; padded q/k/v/beta/decay,
  v_beta/k_beta, cumulative and pairwise decay, UT/intra-chunk products,
  triangular solutions, decayed q/k, prior zero state, v_new, FP32 core and
  final return are captured. `chunk_size=64`, `pad_size=63`, `num_chunks=1`.
- Second token: scaled query, exp(g), original/decayed state, q_t/k_t/v_t,
  beta_t, actual memory prediction, delta, proposed state, FP32 core and final
  return captured. No synthetic reconstruction of any of these tensors.
- Returned core BF16 `[1,1,16,128]`; recurrent state FP32
  `[1,16,128,128]`, key-major. Rust portable semantic domain is value-major;
  compare by explicit layout transport, never by changing numerical values.

## Concrete differences from existing Rust

These are source-arithmetic differences, not an attribution of every current
end-to-end error. Existing Rust QKV/conv inputs already differ upstream; replay
the captured exact operator inputs before assigning individual numerical causes.

| Boundary | Actual pinned slow source | Existing Rust |
| --- | --- | --- |
| Q/K L2 | modeling294–297: FP32 square, FP32 sum, +eps1e-6, rsqrt, FP32 multiply | original_qwen_math800–803 routes through GGUF grouped RMS; model-qwen35 lib259–265 accumulates rounded squares inFP64, divides to mean, adds eps/width, sqrt+reciprocal, then multiplies1/sqrt(width). Algebraic equivalence does not preserve operation order. |
| Query scale | chunk347–348 multiplies query by head_dim**-0.5; recurrent468 divides query by head_dim**0.5 BEFORE state read | source_gdn_recurrent_step1496 scales AFTER source_dot_f32(state,q). Even the two slow source phases intentionally express distinct FP32 operations. |
| First-token route | chunk352–425 pads to64, forms beta*K/beta*V, triangular solves and matmuls, then computes output and final state | original_qwen_math815–825 uses the GGUF recurrent one-token operator for both first and later tokens. A scalar recurrence is not the same instruction sequence as the actual initial chunk path. |
| Cached memory/output reductions | recurrent487/491 materializes FP32 products then `.sum(dim=-2)` across key axis of key-major state | source_gdn_recurrent_step1491/1496 uses GGUF source_dot_f32: four8-lane FMA accumulators and Haswell reduction tree. State layout/reduction source must be explicit. |
| State update | recurrent489 separately rounded `k*delta`, then separately rounded `state+product` | source_gdn_recurrent_step1494 uses `ki.mul_add(delta,state)`. Fused update can differ even with identical input bits. |
| Coefficients | modeling617–619 BF16 sigmoid(b); FP32 softplus(a.float()+dt_bias), exp(A_log.float()), then exp(g) inside actual core | Rust original_gdn_coefficients reuses GGUF scalar sigmoid/softplus/exp. Captured input beta/g and actual decay_t/chunk_decay provide independent targets; exact transcendental implementation remains unqualified. |
| Gated RMS with restored F32 gamma | modeling225–234 casts normalized FP32 core to BF16, multiplies by learned weight(F32) promoting result toFP32, multiplies FP32 SiLU gate, then final BF16 cast | original_qwen_precision130 unconditionally BF16-rounds weighted value before gate, inserting a cast absent from restored36F32 source. Source-level conclusion from pinned code+verified parameter dtype; this observer did not capture gatednorm internal frames. |

## Minimal continuation through the existing engine

1. Preserve the existing GGUF source profile and accepted numerics. Bind the
   pinned Torch CPU operator profile to source/runtime identity explicitly;
   dtype alone cannot select a reduction tree or chunk algorithm.
2. Reuse original source projection, convolution, recurrence-state and
   SessionAuthority owners. Add the exact source L2/FP32-reduction and query
   scale order as closed mathematical operators, replaying the captured
   q/k/normalized/scaled vectors without any token loop.
3. Qualify initial chunk and cached recurrent operations separately on these
   actual inputs. Keep source key-major↔canonical value-major conversion a
   bit transport. Preserve materialized products/separate add versus FMA per
   source profile; do not alter the accepted GGUF recurrence globally.
4. Bind source parameter dtype at gated RMS multiplication; gammaF32 must not
   acquire an invented BF16 cast. Keep official all-BF16 loader a distinct
   profile with separately qualified casts and results.
5. Reuse the same ForwardPlan/math interfaces for production binding after
   exact per-op replay passes. Then repeat full248320/all42 source acceptance.
   These CPU observations do not qualify CUDA FLA, vLLM serving, quantization,
   throughput,27B, or Rust parity; no numerical tolerance has been introduced.

````

## Source: .agents/runs/model-execution-slow-gdn-observation-2026-10-20/TODO.md

Original UTF-8 bytes: 1325; SHA256 `5dda767a3fc3e48500719e63321d5107a17bd29ad837fc6f82ec1addcc3bcceb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pinned slow-GDN observation

Parent: model-execution / serving-and-speculation. Owner: batch_design_review.
Scope: this ignored run only; no MAIN/worktree source, Cargo, GPU or reference
artifact edits. Stock pinned runtime, offline, source/reference read-only,
12GiB memory, 4 CPUs, original restored36F32 .8B profile, tokens11,353.

- [x] Inspect pinned modeling lines300–497 and existing Rust math/precision.
- [x] Capture actual layer0 chunk/recurrent inputs, selected line/return locals,
  original dtypes/layout/order/padding; tensor payload bound64MiB.
- [x] Prove observed full logits/all raw cache tensors match prior captures,
  argmax198,599, source/runtime hashes and unchanged existing manifest.
- [x] Report concrete operator differences and minimal shared per-op seams in
  `OPERATOR_GAPS.md`.

Executed: capture exit0; argmax198,599;98 complete raw reference comparisons
exact;18,916,352B final tensor payload; maxRSS2,318,584KiB. Archive validator
exit0; preserved failed singleton byte-view attempt, total retained archives
31,379,200B. Original pinned reference manifest unchanged. No Cargo/GPU or
production/worktree edits; no Rust/FLA/vLLM inference support claim.

No altered function input, operator, dtype, output or oracle rounding; observer
only clones tensors already present in actual frames.

````

## Source: .agents/runs/model-execution-tied-head/TODO.md

Original UTF-8 bytes: 853; SHA256 `e98696ac1c885b974600535d329a7191f055fb87d0a410a8100a61df270d3d49`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original Qwen tied head

Parent: `.agents/plans/model-execution/TODO.md`
Owner: `/root/batch_design_review`
Scope: `original_qwen_math.rs`, `tests/qwen_original_cpu.rs`; no Cargo/GPU.

## Completed

- [x] Select the authenticated embedding or independent head according to
  verified `tie_word_embeddings`; preserve original projection arithmetic.
- [x] Source-free fixture omits the tied head; untied head has opposite-sign
  rows to detect accidental embedding fallback. Exact analytical logits in
  F32 and BF16 diagnostics, unchanged source identity and lease reclamation.
- [x] Rustfmt and diff check.

## Pending

- [ ] MAIN: focused source-free tests, strict Clippy and full applicable gates.
- [ ] Original trained BF16/FLA and production ForwardPlan binding remain
  independently unqualified; this change only corrects final head selection.

````

## Source: .agents/runs/model-execution-torch-gdn-reductions-2026-10-20/TODO.md

Original UTF-8 bytes: 1213; SHA256 `39226e96c6f6f03b96dbcf848086c98519a48bbe9ab4ad8d8faf5a45d17a702e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pinned Torch GDN reductions

Parent: model-execution. Owner: batch_design_review.
Scope: this ignored run only; source/pinned capture read-only, no MAIN/worktree
edits, Cargo/GPU, operator replacement or oracle reinterpretation.

- [x] Inspect and pin actual CPU sum/rsqrt source/installed headers.
- [x] Replay untouched l2norm and actual public FP32 products/sums/rsqrt
  against captured layer0 inputs/inv_norm/normalized vectors.
- [x] Replay actual materialized recurrent products and sum(keyaxis),
  preserving original strided/key-major layout and source outputs as oracle.
- [x] Derive minimal source-qualified operator proposal and independent
  source-based candidate topology checks; no tolerance changes.
- [x] Parent authorized isolated helpers, now frozen in
  `p12-torch-gdn-reduce-20261020` with3paths and compact8412Bactual fixture.
- [ ] MAIN executes Rust checks and accepts helper integration.

Actual public/original replay14 checks exact; candidate14 checks/12,416F32
values exact, serial128 contrast differs1674memory/1646output values.
See TOPOLOGY.md for source pins, artifact hashes, implementation boundary and
remaining operators. No Rust inference/Cargo/GPU executed by this agent.

````

## Source: .agents/runs/model-execution-torch-gdn-reductions-2026-10-20/TOPOLOGY.md

Original UTF-8 bytes: 4622; SHA256 `0fccb4180f9140646714202b54a7d1da5d83dfd886b097c23eb1efcafb259bd9`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pinned FP32 CPU reduction topology

Owner: batch_design_review. Parent: model-execution. Ignored research only.

## Executed actual-source replay

- Stock pinned vLLM image, no network/GPU,12GiB/4CPU, Torch
  `cf30153c4c131c8164ee7798e5022d810682e2cb`, AVX2,4threads/interop1.
- No model reload. Untouched original `l2norm` and public Torch
  multiply/sum/rsqrt operations replayed on authentic observed layer0 inputs.
-14 checks bit-exact against actual first/second-token frames: four L2 input
  vectors, original normalized output/inv_norm and two recurrent FP32 product
  reductions. Output archive4,328,616B (payload4,325,888B), SHA256
  `1ed8cfd748f7f501a7179a49efb0aaec83a2c55f68a5e8f13ec559c5d9b6c401`.
- `replay-report.json` retains installed-header hashes, CPU capability/build,
  actual shapes/strides, dispatcher provenance and all comparison hashes.
- Actual modeling hash remains762feb6c...412c3b. Original GDN capture and prior
  complete reference artifacts are mounted read-only and unchanged.

## Source-pinned derivation

- Official immutable source commit above, `pinned-torch-source/manifest.json`:
  SumKernel.cpp SHA256
  `9b88aa14e626dd708e50816a5ca89acfc66798226dc82501cf1857caebe96f95`;
  UnaryOpsKernel.cpp SHA256
  `e018483843480dfa2e83fd02ce254dc62afffbb72c86330d46551cb99e2bfc11`.
  Only these small source texts were fetched outside the offline container;
  no model/SDK acquired. Installed Reduce/vec headers copied from pinned image.
- L2 contiguous FP32 width128: cascade_sum selects vectorized_inner_sum.
  AVX2vec8 →16vectors; row_sum ILP4 →multi_row_sum(size4,nrows4).
  Each of32 partial scalar lanes accumulates four inputs, stride32. Combine
  four8-lane vectors sequentially; then sum eight lanes sequentially from+0.
- The captured l2norm squares are materialized FP32 before that sum. Add
  epsilon1e-6 inFP32, perform FP32 sqrt then divide1, then FP32 input*inverse.
  Actual installed AVX2 rsqrt uses sqrt/div, not an approximate rsqrt estimate.
- Recurrent materialized product shape `[1,16,128,128]`, strides
  `[262144,16384,128,1]`, reduction along key=-2. cascade_sum selects contiguous
  outer path. Vector lanes represent independent values, not key partials.
  multi_row_sum(size128) accumulates16-key subchunks into a second-level carry,
  eight times. No horizontal vector reduction; no fused product/sum.

## Independent candidate comparison, not source replacement

- `candidate_topologies.py` retains actual public-op outputs as the oracle.
  NumPyFP32 candidate only follows source-derived topology; it never replaces
  the source function, redefines captured outputs or adjusts rounding.
- All14 candidate checks /12,416F32 values bit-exact:64 squared sums,
  64 inverses,8192normalized Q/K values,2048memory +2048output sums.
- Contrast: a sequential128-key sum differs1674/2048memory values and
  1646/2048output values. Shape and source topology are observable.
- These are pinned CPU per-op observations, not Rust execution, full-model
  equivalence, CUDA FLA, a benchmark or27B acceptance.

## Authorized isolated implementation

- Tree `/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-torch-gdn-reduce-20261020`,
  branch `agent/p12-torch-gdn-reduce-20261020`, clean base6bfcdbe; no overlays.
- New `zerogravity_cpu::original_hf::reduction` exports source-bound finite
  `sum_inner_128`, `l2_norm_128`, `sum_key_128`. Max1,048,576input elements;
  last operator explicitly receives materialized `[head,key128,value128]`.
  Only parent module declaration changes; existing GGUF/source/CPU64/facade/
  state/trace semantics untouched. No token loop or provider dispatch.
- Five Rust tests written, not executed here: adversarial contiguous order,
 16-key cascade with two independent heads/value coordinates, signedzero,
  finite/source/layout/size/overflow guards, compact actual measured goldens.
- Fixture8412B, SHA256
  `9ee2e4e97fc8a2b2a2c48db4e9bec2fed9d0350f1e6b319031b879274c62edec`:
  head0 Q/K ×2tokens plus recurrent actual product columns0,7,63,127/head0.
  Full4.3MiB observed replay stays ignored. Extraction copies raw words only.
- Frozen `reduction-only.patch` contains exactly3 paths; rustfmt, diffcheck
  and reverse-applycheck pass. MAIN alone runs Cargo/strictClippy and accepts
  arithmetic, then coordinates typed ForwardPlan/math/profile binding.
- Open: actual first-token chunk64 solve/matmuls, cached query scale before
  read, coefficients, separately rounded state update and F32-gamma gated RMS.
  Do not map these wholesale to existing GGUF recurrent operators or infer
  runtime eligibility from BF16 dtype alone.

````

## Source: .agents/runs/model-execution-vllm-gemm-operands-20261020/README.md

Original UTF-8 bytes: 6665; SHA256 `e54890a34107b9e05d86efdf06dd634e2af0f269bfdb1f1b24db3ee0e417bceb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Authentic original-HF CUDA GEMM operand extract

Parent: `.agents/plans/model-execution/TODO.md`. Owner: `/root/batch_design_review`.
SOURCE PREPARATION ONLY. ROOT executes tests/export/validation. No tracked source,
old oracle, native provider or shared plan change. No author Torch/model/GPU/
operator/container/runtime execution or actual raw/DSO payload read/hash.

The accepted genuine vLLM0.31.0 eager-B1 capture already retains all required
original public-hook BF16 operands/results for teachers11→353. No new model run,
observer rebuild, transfer or tensor kernel is needed. New extraction is stdlib
file-byte transport over that immutable artifact, not a numerical oracle.

## Authority and preserved gates

Exact accepted parent: `root-probe-v4-1` under the original GEMM observation run.
Its actual `gemm-validation.json` SHA is
`55d68c7a90ac766b97743d2fa8c936d9044c04df8f7611207736d3703c5588be`.
`bindings.json` pins49 existing source files (including all four source freezes),
20 actual metadata/trace files, and expected raw/observer identities copied from
the genuine reports. ROOT hashes the raw/build payloads; author did not read them.

- Re-execute **unchanged source** of both accepted stdlib validators. Definitions
  load from pinned source bytes, not bytecode caches. Only their two metadata
  `write` bindings are replaced with a bounded collector: requested paths must be
  exactly the old two report paths; each result must equal the authenticated
  accepted report. No old path is opened for writing. All original guards,
  comparisons, constants and returned results remain unchanged.
- Original full496640 raw pre-sampler logit words stay plain/observed byte-exact;
  all320 actual source/cast audits before/after, actual runtime/source/image,
  four genuine GEMM metadata/pointer/stream calls, full mapped provider hashes,
  trace health, original public settings, and repaired owned-container cleanup
  must still pass. Existing raw and trace archives are verified before/after.
- All four call keys/phase/order bind `qkvz0,ba0,qkvz1,ba1`, with actual input,
  learned matrix and returned output pointers/storage plus original PID/TID/
  stream. This closed extract additionally requires the authentic observed Ex
  settings: transA1/B0, BF16 A/B/C14, compute68, requested algo99, F32 alpha1/
  beta0, math/atomics/pointer0, stream0, explicit workspace8519680B. No hidden
  selected kernel or algorithm is inferred from that public requested enum.
- Original raw archive record **index + label + call + source metadata + byte
  offset/count/hash** closes each extracted row to these actual hook instances.
  Metadata labels or source-free fixtures alone never authenticate actual values.

## Actual hook ordering

Pinned `gemm_observer.py::ZgGemmExtension.zg_oracle_install` calls the original
base install first. Base registers its raw forward hook; GEMM extension then
registers the public metadata pre-hook and a later forward hook. For each genuine
call: metadata pre-hook records original args[0]/weight pointers and begins the
public phase; stock projection executes unchanged; the earlier base hook copies
that exact args[0]/returned output[0]; the later GEMM hook ends the phase and
records that same output tuple pointer. Every hook returns None. No arithmetic,
tensor/module forward, sampler, learned weight or route was replaced.

The original raw transfer uses detach→blocking `.to(cpu)`→contiguous→uint8 bytes.
Its synchronization/transfer overhead is diagnostic and already covered by the
plain/observed raw-logit guard; it is not a timing/default-graph qualification.
The new extract performs **zero new GPU/model/operator calls or D2H transfers**.
The original C interposer only observes public metadata/pointer extents; it never
reads device-pointer contents. No interposer/worker/launcher bytes are modified.

## Compact values and learned row layout

Six files, exactly36992B:

- Each teacher: one2048B authentic BF16 input shared by QKVZ/BA, one16384B QKVZ
  result, one64B BA result. Signed zero/subnormal and all other bits preserved;
  require every value finite, without dtype conversions or reconstruction.
- QKVZ full raw learned matrix8192×1024: source `in_proj_qkv.weight` occupies
  rows0..6144; source `in_proj_z.weight` rows6144..8192.
- BA full raw learned matrix32×1024: source `in_proj_b.weight` rows0..16;
  source `in_proj_a.weight` rows16..32. It is **b,a**, not lexical source order.
- Full original source descriptors/names/headers, actual packed rows/strides,
  public mapper chain and source/loaded audit hashes remain attached. Stream-hash
  each existing packed-weight archive subrange against its corresponding actual
  source audit. Do not copy/export the existing16MiB QKVZ weight again.

Load profile is the actual CUDA vLLM profile:284 sourceBF16→loadedBF16;
18 A_log sourceF32→loadedF32;18 GDN gamma sourceF32→loadedBF16 explicitly. This
is distinct from the accepted restored-F32 TorchCPU profile. No silent cast36,
CPU reducer inference or wider source/profile eligibility follows.

## ROOT-only commands

Run only after independent source review and the CPU/GPU quiet phase. Six Python
files are AST-parsed only by author; ten source-free test methods are prepared,
not executed. Tests exercise file range bits, signed zero/subnormals/nonfinite,
source/load profile, packed order, raw keys, call/stream/pointer/settings changes,
and actual read-only writer interception using explicitly synthetic definitions.
They are not model/operator/raw-value oracle evidence.

```sh
run=.agents/runs/model-execution-vllm-gemm-operands-20261020
python3 "$run/test_contracts.py"
python3 "$run/export_operands.py" --new-output "$PWD/$run/root-extract-1"
python3 "$run/validate_extract.py" --extract "$PWD/$run/root-extract-1" \
  --new-report "$PWD/$run/root-extract-validation-1.json"
```

Export and validator restrict their own process to four available CPUs,
512MiB address space,120CPU-seconds and180s alarm. They spawn no children, load
no provider/runtime, and only read originals. New outputs are exclusive0600 in
fresh0700 directories; failures never authorize overwrite/retry or old receipt
rewrites. A validator report belongs outside the closed extract directory.

Remaining gates: ROOT source review/test/export + independent raw binding
validation; native original-actor provider/internal allocation/workspace admission;
real native BA first, then QKVZ operator replay. Source/queue/leases/completion/
quarantine must remain original. Full all42/logits native-HF, cached/initial FLA,
wider/mixed batches/default graphs and matched performance remain unqualified.

````

## Source: .agents/runs/model-execution-vllm-gemm-operands-20261020/TODO.md

Original UTF-8 bytes: 1663; SHA256 `8f727c9821d3123896858e8431a107c9fab1a46529ce7c1d3c6c1811ba63adbb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Authentic CUDA GEMM operand capsule

Parent: `.agents/plans/model-execution/TODO.md`. Owner: `/root/batch_design_review`.
Owned scope: this new ignored run only; no original source/captures or MAIN edits.

- [x] Inspect pinned original hook order and actual accepted four-call metadata.
  Existing archive already contains genuine BF16 input/output rows; no rerun.
- [x] Prepare strict49-source/20-metadata/3-payload identity closure; payload
  identities copied from original genuine reports, not read/hashed by author.
- [x] Prepare unchanged accepted full validator delegation with only metadata
  write interception; require both results equal authenticated originals.
- [x] Prepare36992B compact row export, actual call/pointer/stream/settings/raw
  key/index binding; source-audit packed-weight subrange hashes, no weight copy.
- [x] Preserve actual284/18/18 CUDA load-cast profile, BA b,a/QKVZ qkv,z layout;
  all old source/runtime/fullV/320audit/provider/cleanup gates stay mandatory.
- [x] Prepare10 source-free tests; Python AST only, no tests executed by author.
- [x] Freeze source/patch/docs for independent review. Originals stay unchanged.
- [ ] ROOT independent review and source-free tests.
- [ ] ROOT fresh bounded read-only export and independent validation.
- [ ] Original-actor native provider/internal workspace/allocation admission.
- [ ] ROOT physical native BA then QKVZ replay against authentic capsule words.

No native provider/math implementation, full-HF/session/profile/graph/wider-batch
or performance eligibility. No author Torch/model/GPU/operator/runtime/container
execution or actual source/model/DSO/raw payload hash/read.

````

## Source: .agents/runs/model-execution-vllm-gemm-operands-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 6665; SHA256 `e54890a34107b9e05d86efdf06dd634e2af0f269bfdb1f1b24db3ee0e417bceb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Authentic original-HF CUDA GEMM operand extract

Parent: `.agents/plans/model-execution/TODO.md`. Owner: `/root/batch_design_review`.
SOURCE PREPARATION ONLY. ROOT executes tests/export/validation. No tracked source,
old oracle, native provider or shared plan change. No author Torch/model/GPU/
operator/container/runtime execution or actual raw/DSO payload read/hash.

The accepted genuine vLLM0.31.0 eager-B1 capture already retains all required
original public-hook BF16 operands/results for teachers11→353. No new model run,
observer rebuild, transfer or tensor kernel is needed. New extraction is stdlib
file-byte transport over that immutable artifact, not a numerical oracle.

## Authority and preserved gates

Exact accepted parent: `root-probe-v4-1` under the original GEMM observation run.
Its actual `gemm-validation.json` SHA is
`55d68c7a90ac766b97743d2fa8c936d9044c04df8f7611207736d3703c5588be`.
`bindings.json` pins49 existing source files (including all four source freezes),
20 actual metadata/trace files, and expected raw/observer identities copied from
the genuine reports. ROOT hashes the raw/build payloads; author did not read them.

- Re-execute **unchanged source** of both accepted stdlib validators. Definitions
  load from pinned source bytes, not bytecode caches. Only their two metadata
  `write` bindings are replaced with a bounded collector: requested paths must be
  exactly the old two report paths; each result must equal the authenticated
  accepted report. No old path is opened for writing. All original guards,
  comparisons, constants and returned results remain unchanged.
- Original full496640 raw pre-sampler logit words stay plain/observed byte-exact;
  all320 actual source/cast audits before/after, actual runtime/source/image,
  four genuine GEMM metadata/pointer/stream calls, full mapped provider hashes,
  trace health, original public settings, and repaired owned-container cleanup
  must still pass. Existing raw and trace archives are verified before/after.
- All four call keys/phase/order bind `qkvz0,ba0,qkvz1,ba1`, with actual input,
  learned matrix and returned output pointers/storage plus original PID/TID/
  stream. This closed extract additionally requires the authentic observed Ex
  settings: transA1/B0, BF16 A/B/C14, compute68, requested algo99, F32 alpha1/
  beta0, math/atomics/pointer0, stream0, explicit workspace8519680B. No hidden
  selected kernel or algorithm is inferred from that public requested enum.
- Original raw archive record **index + label + call + source metadata + byte
  offset/count/hash** closes each extracted row to these actual hook instances.
  Metadata labels or source-free fixtures alone never authenticate actual values.

## Actual hook ordering

Pinned `gemm_observer.py::ZgGemmExtension.zg_oracle_install` calls the original
base install first. Base registers its raw forward hook; GEMM extension then
registers the public metadata pre-hook and a later forward hook. For each genuine
call: metadata pre-hook records original args[0]/weight pointers and begins the
public phase; stock projection executes unchanged; the earlier base hook copies
that exact args[0]/returned output[0]; the later GEMM hook ends the phase and
records that same output tuple pointer. Every hook returns None. No arithmetic,
tensor/module forward, sampler, learned weight or route was replaced.

The original raw transfer uses detach→blocking `.to(cpu)`→contiguous→uint8 bytes.
Its synchronization/transfer overhead is diagnostic and already covered by the
plain/observed raw-logit guard; it is not a timing/default-graph qualification.
The new extract performs **zero new GPU/model/operator calls or D2H transfers**.
The original C interposer only observes public metadata/pointer extents; it never
reads device-pointer contents. No interposer/worker/launcher bytes are modified.

## Compact values and learned row layout

Six files, exactly36992B:

- Each teacher: one2048B authentic BF16 input shared by QKVZ/BA, one16384B QKVZ
  result, one64B BA result. Signed zero/subnormal and all other bits preserved;
  require every value finite, without dtype conversions or reconstruction.
- QKVZ full raw learned matrix8192×1024: source `in_proj_qkv.weight` occupies
  rows0..6144; source `in_proj_z.weight` rows6144..8192.
- BA full raw learned matrix32×1024: source `in_proj_b.weight` rows0..16;
  source `in_proj_a.weight` rows16..32. It is **b,a**, not lexical source order.
- Full original source descriptors/names/headers, actual packed rows/strides,
  public mapper chain and source/loaded audit hashes remain attached. Stream-hash
  each existing packed-weight archive subrange against its corresponding actual
  source audit. Do not copy/export the existing16MiB QKVZ weight again.

Load profile is the actual CUDA vLLM profile:284 sourceBF16→loadedBF16;
18 A_log sourceF32→loadedF32;18 GDN gamma sourceF32→loadedBF16 explicitly. This
is distinct from the accepted restored-F32 TorchCPU profile. No silent cast36,
CPU reducer inference or wider source/profile eligibility follows.

## ROOT-only commands

Run only after independent source review and the CPU/GPU quiet phase. Six Python
files are AST-parsed only by author; ten source-free test methods are prepared,
not executed. Tests exercise file range bits, signed zero/subnormals/nonfinite,
source/load profile, packed order, raw keys, call/stream/pointer/settings changes,
and actual read-only writer interception using explicitly synthetic definitions.
They are not model/operator/raw-value oracle evidence.

```sh
run=.agents/runs/model-execution-vllm-gemm-operands-20261020
python3 "$run/test_contracts.py"
python3 "$run/export_operands.py" --new-output "$PWD/$run/root-extract-1"
python3 "$run/validate_extract.py" --extract "$PWD/$run/root-extract-1" \
  --new-report "$PWD/$run/root-extract-validation-1.json"
```

Export and validator restrict their own process to four available CPUs,
512MiB address space,120CPU-seconds and180s alarm. They spawn no children, load
no provider/runtime, and only read originals. New outputs are exclusive0600 in
fresh0700 directories; failures never authorize overwrite/retry or old receipt
rewrites. A validator report belongs outside the closed extract directory.

Remaining gates: ROOT source review/test/export + independent raw binding
validation; native original-actor provider/internal allocation/workspace admission;
real native BA first, then QKVZ operator replay. Source/queue/leases/completion/
quarantine must remain original. Full all42/logits native-HF, cached/initial FLA,
wider/mixed batches/default graphs and matched performance remain unqualified.

````

## Source: .agents/runs/model-execution-vllm-gemm-operands-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1663; SHA256 `8f727c9821d3123896858e8431a107c9fab1a46529ce7c1d3c6c1811ba63adbb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Authentic CUDA GEMM operand capsule

Parent: `.agents/plans/model-execution/TODO.md`. Owner: `/root/batch_design_review`.
Owned scope: this new ignored run only; no original source/captures or MAIN edits.

- [x] Inspect pinned original hook order and actual accepted four-call metadata.
  Existing archive already contains genuine BF16 input/output rows; no rerun.
- [x] Prepare strict49-source/20-metadata/3-payload identity closure; payload
  identities copied from original genuine reports, not read/hashed by author.
- [x] Prepare unchanged accepted full validator delegation with only metadata
  write interception; require both results equal authenticated originals.
- [x] Prepare36992B compact row export, actual call/pointer/stream/settings/raw
  key/index binding; source-audit packed-weight subrange hashes, no weight copy.
- [x] Preserve actual284/18/18 CUDA load-cast profile, BA b,a/QKVZ qkv,z layout;
  all old source/runtime/fullV/320audit/provider/cleanup gates stay mandatory.
- [x] Prepare10 source-free tests; Python AST only, no tests executed by author.
- [x] Freeze source/patch/docs for independent review. Originals stay unchanged.
- [ ] ROOT independent review and source-free tests.
- [ ] ROOT fresh bounded read-only export and independent validation.
- [ ] Original-actor native provider/internal workspace/allocation admission.
- [ ] ROOT physical native BA then QKVZ replay against authentic capsule words.

No native provider/math implementation, full-HF/session/profile/graph/wider-batch
or performance eligibility. No author Torch/model/GPU/operator/runtime/container
execution or actual source/model/DSO/raw payload hash/read.

````

## Source: .agents/runs/numerical-divergence-planning-review-20261020/REVIEW.md

Original UTF-8 bytes: 14275; SHA256 `802ca18878cbb40dd69d11314cba4e55d605531485d1ae77fb0eb9bc2fe76ee2`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Numerical divergence: planning-only review

Owner baseline_review; ROOT consolidates the existing plans. Live TODOs/source
and actual retained result metadata reread. No code/plan changes, builds, tests,
imports, operators, providers, models, payload hashes or runtime execution.

**Not all causes are known.** Preserve the exact GGUF AR1 regressions. Treat
native-default bulk, pinned Torch CPU and vLLM CUDA as distinct numerical targets.
Per ROOT's direction, genuine vLLM CUDA is the primary new serving target; the
CPU Torch schedule remains a separately named reference, not a mandatory CUDA
implementation or an oracle to overwrite when a GPU result differs.

| Scope | Cause/status and classification | Smallest fix / dependency | Fast decisive experiment |
|---|---|---|---|
| Historical .8B/2B 8K | Old final-token boundaries were .8B GDN4/2B GDN1. Exact historical IDs/serialization not recovered. This is unresolved historical reproduction, not proof those are today's first causal operators. Recovered240549 artifact hash has unknown encoding;14 negative serialization checks do not prove a different token sequence. | Keep historical issue explicit; use a recovered original ID file/generator+format if found. Don't block current arithmetic diagnosis on unbounded artifact searches. | No new full8K run solely for recovery. Reuse fresh fixed IDs for numerical work; label them fresh. Source: numerics TODO10–26/135–137; bulk-f32-primitive/HISTORICAL-RECOVERY.json. |
| Fresh GGUF AR1, both models | Actual8K matches all730 selected records/full248320 logits and separately all42 causal state domains/55,382,016 words/model, zero bits including signed zero. Declared use_ref64 controls also match;2B additionally requires explicit CPU_Mapped intermediates to avoid BLAS math. These are qualified schedules, not default bulk. | Keep current four-Vec8 source math as the regression target. No blanket GDN rewrite is justified. | Reuse existing AR1/control comparisons as regression; rerun them only after relevant code changes. Sources: numerics TODO39–77; actual long-context/root-compare-*-8192-ar1.json +state-codec/root-compare-*-8192.json. |
| .8B native-default bulk | First observed unchanged first64 gap: layer3 attention AV, position2,530/2048 words. Probabilities/all causal V and earlier captured layers match. Public replay binds default outputs; use_ref bulk equals AR1. Compiled first entries select tinyBLAS vs ggml_vec_dot_f32. Demonstrated route-dependent arithmetic; not a proven sole/full8K cause. | Qualify the isolated single-Vec8 tinyBLAS primitive/typed slice against existing expected outputs; preserve AR1 four-Vec8. Full default equivalence then needs actual temporal-N ForwardPlan and correct per-row causal/padded KV, not token-major prefill relabeled bulk. | Existing AV+alpha132,096-word raw replay first, then unchanged first64 full intermediate/state comparison. Avoid another costly8K run until the first block closes. Sources: numerics TODO78–92/111–125; lib.rs1305/1423; isolated root-default TODO. |
| 2B native-default bulk F32 | First observed gap is alpha0 position0,13/16 words; preceding compared rows, all131072 input words and exact raw F32 weight agree. Beta also differs before GDN. Public alpha replay/control and compiled first-entry observations support different reduction schedules. | Same explicit temporal tinyBLAS contract, not a global F32 reducer replacement. Candidate remains uncompiled/unqualified; one primitive cannot establish every native bulk route. | Reuse genuine65 alpha cases and all1024 bulk output words. After exactness, compare the unchanged first64 layer0 trajectory; stop at next difference. Sources: numerics TODO93–125; CPU weights.rs592; model lib.rs1175–1215 still loops forward_token. |
| 2B native-default Q8→BLAS | V7 actually refuses `linear_attn_out-0`: Q8_0[2048,2048] ×F32[2048,64]. Default BLAS dequantizes weights to F32 and calls SGEMM; current packed-dot route quantizes activations. This is a separate numerical-route mismatch, not evidence alpha caused the BLAS split. CPU_Mapped is a declared control, not a production fix. | Derive/qualify this exact default dequantization+SGEMM route or explicitly retain a distinct AR1 profile. Keep original backends/source values; don't disable BLAS to claim default parity. | Closed replay of that actual operator with identical input/raw weight first; bind dequantized words and complete outputs. No new wholemodel8K run needed to isolate it. Sources: use-ref/root-2b-8192-ref64-7 public trace; pinned ggml-blas.cpp59–145/396–432. |
| HF loader profiles, .8B/2B | Official BF16 loading and source-36-F32-restored loading are intentionally different. .8B first argmax321 vs198;2B differs47/49 raw tensors per step despite equal argmax. This is a profile/source-transformation difference, not by itself a Rust bug. | Declare the actual target loader/loaded parameter dtypes and keep raw source dtype metadata intact. For vLLM target follow its authenticated loaded profile; don't silently restore/cast36 parameters to make another target pass. | Reuse existing two-profile two-token captures and320-role vLLM loader audits. Compare exact target tensors only. Source: model-execution TODO37–65; accepted pinned-reference/HF2B captures. |
| HF .8B diagnostic vs Torch CPU | Old full diagnostic still uses GGUF grouped norm/conv/host exp/fused recurrence and prematurely BF16-rounds a raw-F32 gamma product. Those are genuine implementation gaps relative to the explicitly selected CPU target. The accepted closed norm/conv/reducer/cached helpers fix specific operators, but are not yet a complete forward. Old41/42-domain/full-logit failure is not a fresh composed-helper result. | Use existing typed HostMath operations in one schedule, preserving original dtype/cast/state order. Do not modify accepted GGUF math or infer that fragment acceptance fixed the full diagnostic. | Two teachers11/353 through one genuine layer0 initial→cached chain, first-difference comparison at real intermediates, then whole .8B fullV/all42. Source: original_qwen_math.rs174/705–865; precision.rs112–149; live model TODO55–65. |
| HF27B initial/cached | Actual reference provides initial64+true cached targets. Isolated H48 cached core+BF16-gamma RMS+wide roles pass, but cached imports observed exp(g) and repeated Q/K. Initial64 remains missing. This is incomplete implementation/producer qualification, not a demonstrated error in the accepted cached primitive. | Complete the phase-specific producers and initial solver below; keep full-HF execution/checkpoint guards closed. Source851-role windows and width17408/head248320 slices now work in isolated pure-role proof, not a whole model. | Reuse existing V3 archives, not another55GB model load. Initial scale/prefix and H48 coefficient preparation can be qualified separately using combined525; then exact whole layer0 initial→cached. Source: model TODO12–35; actual H48/gamma validators; original_binding.rs307. |
| Primary new CUDA target, .8B/2B | Existing `cpu_bf16_projection.rs` deliberately implements CPU64 numerics on CUDA. That is not vLLM's cuBLAS/FLA profile. GGUF stock also uses uniform Q8_1 vs ZG's source-specific Q8_0/Q8_1/Q8_K preparation plus distinct casts/reducers. Same weights/tokenizer or matching text does not prove equivalent math. No evidence establishes a unique cross-engine cause yet. | Add an explicit target numerical contract, using original queue/source/leases. Keep CPU-target kernels/regressions distinct. Prioritize the actual pinned vLLM CUDA route being compared; its accepted public GEMM observation is a target, not a replacement engine. | Start .8B with its authenticated36992B BF16 QKVZ/BA operands +cuBLAS settings and compare raw native operator results first; then capture/compare genuine vLLM layer0 conv/coefficient/FLA/state boundaries for teachers11/353. Stop at first mismatch, including cast boundaries. FullV/all42 follows actual layer closure, not CPU64 acceptance. Sources: model TODO240–275; gpu cpu_bf16_projection.rs43; matched-bench PROTOCOL.md10. |
| Context/state/resource | GPU qualification is bounded512 consumed tokens. Complete original HF2B/27B source admission/streaming, all42/112 typed effects and full-model checkpoint/profile binding remain open. These are qualification/ownership blockers, not observed numerical bugs. | Reuse original source windows, SessionAuthority/StateRoot and arbiter; no partial TensorSource/fullmodel permit.2B's4.548GB shard exceeds complete-source4GiB guard;27B cannot be whole-RAM-loaded through that route. | After short whole-layer/model exactness, extend context/boundary/source-state tests. Do not start16K/32K or report27B inference from pure-window operators. Sources: live numerics TODO20–21; model TODO12–71/278–300; original_binding.rs307. |

## Exact remaining original-HF CPU cuts

These are the pinned CPU target's requirements. vLLM CUDA may use different
qualified cuBLAS/FLA algorithms; it should be compared to its own actual outputs.

1. **Cached:** genuine QKV producer→source conv4/SiLU→BF16 Q/K repeat16→48
   (for27B), accepted FP32 L2 and direct query DIVsqrt128; actual betaBF16,
   raw A_log BF16/F32→F32 exp/neg, `a.float()+dt`→softplus, separate F32 g and
   exp(g). Accepted core then materializes decay/products, key-axis sums and
   separate update multiply/add; state is key-major FP32, return BF16. Connect
   correct F32-gamma or BF16-gamma gated RMS and output projection. Imported
   decay/prior/QK proves no producer. Never substitute GGUF FMA recurrence.
2. **Initial64:** preserve initial MUL0x3db504f3, true pad64+0, sequential
   F64-carry/F32-store cumsum, strict-upper -Inf masked exp, six real batch-GEMM
   families and two per-head unit-triangular solve groups. For H16 that's32
   STRSM calls; H48 requires96. Preserve original column-major solve outputs,
   all materialized boundaries and BF16 return; zero initial state does not
   permit replacing chunk64 with cached recurrence. New scale/prefix leaf is
   source-only, not qualified. Initial-prefix/H48 coefficient replay scripts bind the source521+overlays
   through the preserved combined525 derivative, still unexecuted. The cached
   core replay itself already passed in the separate isolated composition.
3. **Standalone exp target:** pinned Torch Tensor.exp calls public MKL vmsExp
   (LP64 count/int64 mode0x140102), not SLEEF exp used within accepted SiLU/
   sigmoid/softplus. Actual3 proves four EXP16 calls, six group16 GEMMs and32
   STRSM,1,048,640 exact words. It does not prove H48 calls, masked -Inf,
   strided exp views, general provider admission or complete GDN. Existing V3
   operands suffice for decisive new operator replays; don't synthesize targets.
4. **Resource blocker:** six startup tests pass with the provider UNLOADED.
   Current child seccomp rejects clone/threads, so MKL/OpenMP4-thread math is
   outside that policy. Source/output/scratch admission alone doesn't account
   native DSO constructors, stacks, internal pools/caches or opaque allocator
   lifetime. Keep owners through known return/retirement; unknown completion
   fences/retains. A configured4GiB envelope/AS/cgroup observation is not an
   admitted physical bound. Prove a single real exp call under a reviewed
   owner/envelope before extending CPU GEMM/TRSM. CUDA public workspace identity
   likewise does not by itself qualify every internal native allocation.

Source anchors: pinned modeling_qwen3_5.py300–435/438–497/617–642;
CPU original_hf/recurrent.rs99–225; native-provider-probe/ABI.md;
root-actual-3/rust-report.json; startup-v1/POLICY.md and actual validation.json.

## Shortest ordered experiments, when execution resumes

1. **New primary CUDA parity:** use existing pinned vLLM outputs, identical
   loaded source/profile/teacher IDs, smallest real operator and then layer0
   initial/cached intermediate+state comparison. Keep original CPU/AR1 regressions.
2. **CPU default-bulk issue:** first qualify existing132096-word tinyBLAS slice,
   then unchanged first64. Isolate default2B Q8→BLAS separately. Only then run
   512→1K→2K→4K→8K with identical actual partitions/fullV/all42, retaining tails.
3. **CPU original-HF closure:** reuse two-token .8B/2B/27B captures and the
   accepted isolated fragments; qualify scale/prefix/coefficients and ONE
   admitted native exp, then initial BMM/TRSM and complete layer0 chain. The
   CPU target can advance in parallel; its ISA schedule is not CUDA's target.
4. **Whole models:** compose the same ForwardPlan/HostMath/state effects only
   after all missing operators are exact for their named profile, then full
   logits/state and source/resource gates. No tolerance widening, zero-sign
   exception, expected-output fitting, second engine or new receipt framework.


## ROOT draft review

Read all507 lines of
`.agents/plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md`.
C1 is explicitly a bounded CPU side task; C0/C2 correctly prioritize vLLM CUDA.
No all-causes-found or automatic fullmodel claim. No shared document edited.

Concrete additions/clarifications sent ROOT:

- Add stock GGUF uniformQ8_1 vs ZG source-specific Q8_0/Q8_1/Q8_K to the
  inventory; N4's CPU BLAS route is different. Select GGUF-plugin vs HF-BF16
  lane at C0. The accepted BF16 cuBLAS operands cannot stand in for GGUF inputs.
- Compare actual initial/continued-prefill/cached phases with identical target
  partitions; do not assert different B/chunk partitions have equal bits by name.
- N5–N7 selected-kernel source-profile differences need pinned source/caller/
  backend anchors and actual route evidence before being identified as causes
  of current cross-engine output differences. Their source-level differences
  alone do not localize every observed failure.
- Credit qualified isolated17408-column CPU down/window parity; trained producer,
  fullmodel and native GPU wide integration remain. CPU native provider admission
  is still open and must not block independent CUDA progress.

No other concrete sequencing blocker found by this review. It does not claim
complete cause discovery or independently re-execute ROOT's retained results.

````

## Source: .agents/runs/numerical-divergence-planning-review-20261020/TODO.md

Original UTF-8 bytes: 670; SHA256 `c76fa71699d4349acacf7d9f9c28b1fa83af49e5718ef16fb97c9a92be6a09fe`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: numerics-and-context/model-execution. Own ignored planning review only.

- [x] Read live queue/plans, original GGUF/HF source seams and actual result metadata.
- [x] Separate historical recovery, fresh AR1/reference controls and native-default.
- [x] Separate confirmed implementation gaps, profile differences and missing gates.
- [x] Prioritize vLLM CUDA as new target; preserve named Torch CPU/AR1 regressions.
- [x] Give bounded next experiments and actual initial/cached/provider blockers.

No code/plans/worktree edits, builds/tests/imports/provider/model/runtime or hashes.
ROOT owns consolidated planning; report makes no all-causes-found/closure claim.

````

## Source: .agents/runs/numerics-and-context-2b-alpha-replay-20261020/CLEANUP-V2.md

Original UTF-8 bytes: 1437; SHA256 `976c596c0931480ebcd52acfca6527a347f8ad65601b94f5cfb75957c66b2b60`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive owned-child cleanup v2 — ROOT ONLY

Original ee78cf57 preparation/frozen sources are unchanged. Do not execute the
old run_replay.py entrypoint. Use run_replay_v2.py with IDENTICAL CLI arguments
for both plain and observed runs. Original C++/observer/prepare/validator,
actual math, inputs, ABI, conditions and outputs stay unchanged.

Exact reviewed 27B cleanup helper copied: WNOWAIT+unreaped Popen/session/PGID
binding, independent TERM/grace/two bound KILL attempts BEFORE ANY wait;
independent first/final wait, no nonzero signal after any reap attempt.
Post-reap group absence uses signal0 only. Every stage error remains visible;
pipe draining and closers run independently and retain all cleanup errors plus
original primary exception. Any cleanup failure makes receipt UNQUALIFIED and
blocks successful wrapper exit, even if a later cleanup stage succeeds.

Original receipt status/argv/cleanup fields are retained so validate.py uses
unchanged arguments. Additive runner SHA and cleanup-v2-manifest pin identify
actual v2 execution. ROOT must inspect these bindings in both invocation receipts.
Main authenticates every unchanged original prepared source and all v2 files.
Four copied real-owned-child TERM/grace/first-KILL/first-reap failure fixtures
are prepared for ROOT: python3 test_cleanup_v2.py. Author ran AST only; no child,
fixture, operator/model/native/build/GPU execution. Root performs actual tests.

````

## Source: .agents/runs/numerics-and-context-2b-alpha-replay-20261020/README.md

Original UTF-8 bytes: 8090; SHA256 `ab4144bbac7bceb08301dc8134e796ea160e44dc7e6a5ac38d544572fe66af0c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original2B alpha projection public replay — ROOT EXECUTES

Parent: numerics-and-context. Owner: spine_review. New ignored diagnostic only.
No MAIN/plan/math edits, build, native/operator/model/container/GPU execution by
this author. Preparation used source, metadata and bounded raw-byte comparison.
Original capture/helpers/frozen attempts remain untouched.

## Earliest actual input closure

Accepted root-first64-repack-transport capture independently retains plain versus
observed whole post-state and next real AR1 full248320/whole-state transparency.
Its first observed gap is layer0 projection_output::blk.0.ssm_alpha.weight at
position0:13/16 words, first bulkbf1ef412 versus AR1bf1ef414, before bias/softplus.

- All960 preceding compared rows (15 captured stage families x64) are exact:
  embedding, actual RMS/weighted norm, QKV, conv four-tap inputs/results, Q/K/V,
  and every alpha input. This is captured-scope order, not an unobserved global
  first-cause claim.
- Independently read all131072 actual F32 alpha-input words. Bulk [2048,64,1,1]
  contiguous bytes equal every corresponding64 AR1 [2048,1,1,1] input exactly;
  SHA731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6.
  Actual source tensorID5 attn_norm-0/MUL matches the captured producer and each
  actual projection src1, checked before/after. No tokenizer/state reconstruction.
- Full32768 F32 learned alpha words, shape[2048,16,1,1],131072B, SHA
  0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876,
  independently equal original approved GGUF bytes [459622304,459753376).
  Actual source buffer is ordinary CPU_Mapped/GGUF_RAW; opaque CPU_REPACK bytes
  from other model operators do not enter this standalone F32 projection.
- Exact original GGUF SHA91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6,
  full source/header/capture metadata and relevant original source/DSO files are
  pinned. Author read only bounded alpha source range; ROOT authenticate performs
  whole-source before/after hashing. INPUT-CLOSURE.json records this audit.

## Genuine public computation and observation

- Same pinned original llama.cpp0.6.0 Haswell CPU backend and public
  ggml_mul_mat(F32 weight[2048,16],F32 input[2048,64 or1]); all ne/nb/dtypes checked.
  CPU payloads are populated with unchanged actual raw bits. No model load,
  dequantization, custom dot loop, kernel replacement or mathematical fallback.
  Input tensor/buffer ownership is independent diagnostic storage, not an alias
  or substitute model/source identity.
- Original64-row operator plus64 original AR1 operators,8 backend threads.
  Every default output must bit-match its entire authentic capture:1024 bulk
  words+64x16 AR1words=2048. Any default mismatch refuses BEFORE any control.
- Only after all65 defaults pass, repeat65 calls using the explicitly public
  ggml_backend_cpu_set_use_ref control. This is a named diagnostic condition,
  not original model parity or permission to change Rust/production kernels.
  Record every control output and complete refbulk versus defaultAR1 comparison.
- Plain no-observer run first; separate same binary/inputs run with source-only
  observer. ALL130 full default/control outputs in each run must be identical
  across plain/observed before route evidence is accepted (4096words/run).
- Optional LD_PRELOAD observer uses exact public sgemm.h13-argument ABI and
  unchanged opaque params. Original public target is exact already-loaded
  Haswell DSO via NOLOAD+dlsym+dladdr at authenticated exported offset0x10c8f0.
  Call original exactly once, forward all pointers/scalars and return original
  bool unchanged. No private params layout read, arithmetic or substitution.
- Integer-only bounded2048-call ring;8 expected source entry calls/bulk and each
  AR1 call, no control calls. Source-derived intended population:8 bulk64 true,
  64x8 AR1 false,65 use_ref zero =520. Exact m16/n64-or1/k2048, lda/ldb2048,
  ldc16, F32 types, zero bound operand offsets. ROOT must actually establish it.
- Primary source: qwen35.cpp379-381/build_lora_mm uses this exact projection;
  tiled.cpp rejects F32 source; ggml-cpu.c1308+ gates the public sgemm branch;
  sgemm.cpp3911 returns false for n<2 and its F32 AVX branch selects tinyBLAS;
  use_ref skips sgemm. These are source branch facts, not observed internal
  instruction/microkernel routes. The wrapper records exported entry only.
- Actual maps before/after must show only pinned Haswell CPU backend. Exact
  relevant source and ggml DSO pins do not establish entire host runtime closure.
  Internal microkernel/long-context cause/profile/model eligibility remain false.
  Existing Rust declared AR1 kernels are untouched; no fitting nativebulk math.

## Bounds and ROOT procedure

Fresh private fixtures<4MiB (actual payload1,187,840B), expected values copied only
from authentic actual captures, never candidate math. C++ opens expected payloads
for comparison only after genuine public computation; outer source authentication
hashes fixture files before/after. Every input is re-read in CPU storage and file
form after each operator. F32 finite checks preserve all raw bits/signedzero.

No network/GPU/model loading. Process AS<=8GiB, CPU deadline<=300x8+1s, wall300s,
8 backend threads, backing<=2MiB/case, graph metadata2MiB/case. Owned PGID launcher
is copied from accepted AV wrapper: WNOWAIT preserves leader identity, unconditional
TERM/KILL/reap/group-absence and pipeEOF/closure happen before finalreceipt I/O,
even if leader already exited. Logs<=8MiB/run; binary<=16MiB, observer<=8MiB.
Entire new ignored-run artifact tree<=512MiB including reserved bounded report.
No old capture/failed attempt is overwritten. Python AST only by author; no tests.

ROOT verifies freeze and source pins, then builds with its accepted GCC/glibc2.44
closure and strict warnings. No fast-math or upstream build. Root alone runs this
after competing model/CPU/GPU work permits; placeholders require actual hashes.

```sh
TASK_ALPHA=.agents/runs/numerics-and-context-2b-alpha-replay-20261020
TASK_NATIVE=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
TASK_C=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
python3 "$TASK_ALPHA/prepare.py" "$TASK_ALPHA/root-fixtures"
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "$TASK_NATIVE/ggml/include" "$TASK_ALPHA/main.cpp" \
  -L "$TASK_LIB" -L "$TASK_C" -Wl,-rpath,"$TASK_LIB:$TASK_C" \
  -Wl,--dynamic-linker,"$TASK_C/ld-linux-x86-64.so.2" \
  -lggml -lggml-base -ldl -lm -o "$TASK_ALPHA/root-replay"
c++ -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC \
  -I "$TASK_NATIVE/ggml/src/ggml-cpu/llamafile" "$TASK_ALPHA/observer.cpp" \
  -L "$TASK_C" -Wl,-rpath,"$TASK_C" -ldl -o "$TASK_ALPHA/root-observer.so"
python3 "$TASK_ALPHA/run_replay.py" --binary "$TASK_ALPHA/root-replay" \
  --binary-sha256 ACTUAL_REPLAY_SHA --fixtures "$TASK_ALPHA/root-fixtures" \
  --new-output "$TASK_ALPHA/root-plain" --new-log "$TASK_ALPHA/root-plain.log" \
  --new-receipt "$TASK_ALPHA/root-plain.invocation.json"
python3 "$TASK_ALPHA/run_replay.py" --binary "$TASK_ALPHA/root-replay" \
  --binary-sha256 ACTUAL_REPLAY_SHA --fixtures "$TASK_ALPHA/root-fixtures" \
  --preload "$TASK_ALPHA/root-observer.so" --preload-sha256 ACTUAL_OBSERVER_SHA \
  --new-output "$TASK_ALPHA/root-observed" --new-log "$TASK_ALPHA/root-observed.log" \
  --new-receipt "$TASK_ALPHA/root-observed.invocation.json"
python3 "$TASK_ALPHA/validate.py" --plain-output "$TASK_ALPHA/root-plain" \
  --plain-receipt "$TASK_ALPHA/root-plain.invocation.json" \
  --observed-output "$TASK_ALPHA/root-observed" \
  --observed-receipt "$TASK_ALPHA/root-observed.invocation.json" \
  --new-report "$TASK_ALPHA/root-validation.json"
```

Successful output means full public-operator replay/observer transparency and
exported-entry diagnostic evidence. It does not close full runtime/internal
kernel causality, historical8K corpus equivalence, HF/vLLM math or model support.

````

## Source: .agents/runs/numerics-and-context-2b-alpha-replay-20261020/TODO.md

Original UTF-8 bytes: 1253; SHA256 `63a95580a37987be9f8939905f247d6261a7b69599e3e1500eca93107da787d8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original2B alpha first-gap public replay

Parent: numerics-and-context. Owner: spine_review; ignored new run only.
ROOT owns all builds/tests/native/operator/model/GPU/container execution.

- [x] Independently inspect actual first64 capture/comparison;960 preceding rows
  exact, full131072 alpha input words exact and actual producer binding closed.
- [x] Independently byte-match all32768 alpha F32 weights to original GGUF
  range[459622304,459753376); distinguish ordinary raw from opaque otherweights.
- [x] Prepare genuine public ggml full64+64AR default replay, gated publicuse_ref
  controls, unchanged ABI/return entry observer and full plain/observed validator.
- [x] Preserve existing helpers/attempts; pin all relevant capture/source/DSO files,
  bounded owned lifecycle, exact finite shapes/outputs and no internalcause claim.
- [ ] Independent source review, ROOT strict build and byte-only fixture creation.
- [ ] ROOT actual all65 default2048 words exact BEFORE65 controls; all130
  plain/observed4096 words exact BEFORE520 exported-entry route population accepted.
- [ ] Separately interpret actual control results/source branch proof. Do not alter
  declared Rust AR1 math by fitting nativebulk; no automatic full-longcontext cause.

````

## Source: .agents/runs/numerics-and-context-2b-alpha-replay-20261020/frozen-cleanup-v2/CLEANUP-V2.md

Original UTF-8 bytes: 1437; SHA256 `976c596c0931480ebcd52acfca6527a347f8ad65601b94f5cfb75957c66b2b60`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive owned-child cleanup v2 — ROOT ONLY

Original ee78cf57 preparation/frozen sources are unchanged. Do not execute the
old run_replay.py entrypoint. Use run_replay_v2.py with IDENTICAL CLI arguments
for both plain and observed runs. Original C++/observer/prepare/validator,
actual math, inputs, ABI, conditions and outputs stay unchanged.

Exact reviewed 27B cleanup helper copied: WNOWAIT+unreaped Popen/session/PGID
binding, independent TERM/grace/two bound KILL attempts BEFORE ANY wait;
independent first/final wait, no nonzero signal after any reap attempt.
Post-reap group absence uses signal0 only. Every stage error remains visible;
pipe draining and closers run independently and retain all cleanup errors plus
original primary exception. Any cleanup failure makes receipt UNQUALIFIED and
blocks successful wrapper exit, even if a later cleanup stage succeeds.

Original receipt status/argv/cleanup fields are retained so validate.py uses
unchanged arguments. Additive runner SHA and cleanup-v2-manifest pin identify
actual v2 execution. ROOT must inspect these bindings in both invocation receipts.
Main authenticates every unchanged original prepared source and all v2 files.
Four copied real-owned-child TERM/grace/first-KILL/first-reap failure fixtures
are prepared for ROOT: python3 test_cleanup_v2.py. Author ran AST only; no child,
fixture, operator/model/native/build/GPU execution. Root performs actual tests.

````

## Source: .agents/runs/numerics-and-context-2b-alpha-replay-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 8090; SHA256 `ab4144bbac7bceb08301dc8134e796ea160e44dc7e6a5ac38d544572fe66af0c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original2B alpha projection public replay — ROOT EXECUTES

Parent: numerics-and-context. Owner: spine_review. New ignored diagnostic only.
No MAIN/plan/math edits, build, native/operator/model/container/GPU execution by
this author. Preparation used source, metadata and bounded raw-byte comparison.
Original capture/helpers/frozen attempts remain untouched.

## Earliest actual input closure

Accepted root-first64-repack-transport capture independently retains plain versus
observed whole post-state and next real AR1 full248320/whole-state transparency.
Its first observed gap is layer0 projection_output::blk.0.ssm_alpha.weight at
position0:13/16 words, first bulkbf1ef412 versus AR1bf1ef414, before bias/softplus.

- All960 preceding compared rows (15 captured stage families x64) are exact:
  embedding, actual RMS/weighted norm, QKV, conv four-tap inputs/results, Q/K/V,
  and every alpha input. This is captured-scope order, not an unobserved global
  first-cause claim.
- Independently read all131072 actual F32 alpha-input words. Bulk [2048,64,1,1]
  contiguous bytes equal every corresponding64 AR1 [2048,1,1,1] input exactly;
  SHA731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6.
  Actual source tensorID5 attn_norm-0/MUL matches the captured producer and each
  actual projection src1, checked before/after. No tokenizer/state reconstruction.
- Full32768 F32 learned alpha words, shape[2048,16,1,1],131072B, SHA
  0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876,
  independently equal original approved GGUF bytes [459622304,459753376).
  Actual source buffer is ordinary CPU_Mapped/GGUF_RAW; opaque CPU_REPACK bytes
  from other model operators do not enter this standalone F32 projection.
- Exact original GGUF SHA91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6,
  full source/header/capture metadata and relevant original source/DSO files are
  pinned. Author read only bounded alpha source range; ROOT authenticate performs
  whole-source before/after hashing. INPUT-CLOSURE.json records this audit.

## Genuine public computation and observation

- Same pinned original llama.cpp0.6.0 Haswell CPU backend and public
  ggml_mul_mat(F32 weight[2048,16],F32 input[2048,64 or1]); all ne/nb/dtypes checked.
  CPU payloads are populated with unchanged actual raw bits. No model load,
  dequantization, custom dot loop, kernel replacement or mathematical fallback.
  Input tensor/buffer ownership is independent diagnostic storage, not an alias
  or substitute model/source identity.
- Original64-row operator plus64 original AR1 operators,8 backend threads.
  Every default output must bit-match its entire authentic capture:1024 bulk
  words+64x16 AR1words=2048. Any default mismatch refuses BEFORE any control.
- Only after all65 defaults pass, repeat65 calls using the explicitly public
  ggml_backend_cpu_set_use_ref control. This is a named diagnostic condition,
  not original model parity or permission to change Rust/production kernels.
  Record every control output and complete refbulk versus defaultAR1 comparison.
- Plain no-observer run first; separate same binary/inputs run with source-only
  observer. ALL130 full default/control outputs in each run must be identical
  across plain/observed before route evidence is accepted (4096words/run).
- Optional LD_PRELOAD observer uses exact public sgemm.h13-argument ABI and
  unchanged opaque params. Original public target is exact already-loaded
  Haswell DSO via NOLOAD+dlsym+dladdr at authenticated exported offset0x10c8f0.
  Call original exactly once, forward all pointers/scalars and return original
  bool unchanged. No private params layout read, arithmetic or substitution.
- Integer-only bounded2048-call ring;8 expected source entry calls/bulk and each
  AR1 call, no control calls. Source-derived intended population:8 bulk64 true,
  64x8 AR1 false,65 use_ref zero =520. Exact m16/n64-or1/k2048, lda/ldb2048,
  ldc16, F32 types, zero bound operand offsets. ROOT must actually establish it.
- Primary source: qwen35.cpp379-381/build_lora_mm uses this exact projection;
  tiled.cpp rejects F32 source; ggml-cpu.c1308+ gates the public sgemm branch;
  sgemm.cpp3911 returns false for n<2 and its F32 AVX branch selects tinyBLAS;
  use_ref skips sgemm. These are source branch facts, not observed internal
  instruction/microkernel routes. The wrapper records exported entry only.
- Actual maps before/after must show only pinned Haswell CPU backend. Exact
  relevant source and ggml DSO pins do not establish entire host runtime closure.
  Internal microkernel/long-context cause/profile/model eligibility remain false.
  Existing Rust declared AR1 kernels are untouched; no fitting nativebulk math.

## Bounds and ROOT procedure

Fresh private fixtures<4MiB (actual payload1,187,840B), expected values copied only
from authentic actual captures, never candidate math. C++ opens expected payloads
for comparison only after genuine public computation; outer source authentication
hashes fixture files before/after. Every input is re-read in CPU storage and file
form after each operator. F32 finite checks preserve all raw bits/signedzero.

No network/GPU/model loading. Process AS<=8GiB, CPU deadline<=300x8+1s, wall300s,
8 backend threads, backing<=2MiB/case, graph metadata2MiB/case. Owned PGID launcher
is copied from accepted AV wrapper: WNOWAIT preserves leader identity, unconditional
TERM/KILL/reap/group-absence and pipeEOF/closure happen before finalreceipt I/O,
even if leader already exited. Logs<=8MiB/run; binary<=16MiB, observer<=8MiB.
Entire new ignored-run artifact tree<=512MiB including reserved bounded report.
No old capture/failed attempt is overwritten. Python AST only by author; no tests.

ROOT verifies freeze and source pins, then builds with its accepted GCC/glibc2.44
closure and strict warnings. No fast-math or upstream build. Root alone runs this
after competing model/CPU/GPU work permits; placeholders require actual hashes.

```sh
TASK_ALPHA=.agents/runs/numerics-and-context-2b-alpha-replay-20261020
TASK_NATIVE=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
TASK_C=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
python3 "$TASK_ALPHA/prepare.py" "$TASK_ALPHA/root-fixtures"
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "$TASK_NATIVE/ggml/include" "$TASK_ALPHA/main.cpp" \
  -L "$TASK_LIB" -L "$TASK_C" -Wl,-rpath,"$TASK_LIB:$TASK_C" \
  -Wl,--dynamic-linker,"$TASK_C/ld-linux-x86-64.so.2" \
  -lggml -lggml-base -ldl -lm -o "$TASK_ALPHA/root-replay"
c++ -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC \
  -I "$TASK_NATIVE/ggml/src/ggml-cpu/llamafile" "$TASK_ALPHA/observer.cpp" \
  -L "$TASK_C" -Wl,-rpath,"$TASK_C" -ldl -o "$TASK_ALPHA/root-observer.so"
python3 "$TASK_ALPHA/run_replay.py" --binary "$TASK_ALPHA/root-replay" \
  --binary-sha256 ACTUAL_REPLAY_SHA --fixtures "$TASK_ALPHA/root-fixtures" \
  --new-output "$TASK_ALPHA/root-plain" --new-log "$TASK_ALPHA/root-plain.log" \
  --new-receipt "$TASK_ALPHA/root-plain.invocation.json"
python3 "$TASK_ALPHA/run_replay.py" --binary "$TASK_ALPHA/root-replay" \
  --binary-sha256 ACTUAL_REPLAY_SHA --fixtures "$TASK_ALPHA/root-fixtures" \
  --preload "$TASK_ALPHA/root-observer.so" --preload-sha256 ACTUAL_OBSERVER_SHA \
  --new-output "$TASK_ALPHA/root-observed" --new-log "$TASK_ALPHA/root-observed.log" \
  --new-receipt "$TASK_ALPHA/root-observed.invocation.json"
python3 "$TASK_ALPHA/validate.py" --plain-output "$TASK_ALPHA/root-plain" \
  --plain-receipt "$TASK_ALPHA/root-plain.invocation.json" \
  --observed-output "$TASK_ALPHA/root-observed" \
  --observed-receipt "$TASK_ALPHA/root-observed.invocation.json" \
  --new-report "$TASK_ALPHA/root-validation.json"
```

Successful output means full public-operator replay/observer transparency and
exported-entry diagnostic evidence. It does not close full runtime/internal
kernel causality, historical8K corpus equivalence, HF/vLLM math or model support.

````

## Source: .agents/runs/numerics-and-context-2b-alpha-replay-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1253; SHA256 `63a95580a37987be9f8939905f247d6261a7b69599e3e1500eca93107da787d8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original2B alpha first-gap public replay

Parent: numerics-and-context. Owner: spine_review; ignored new run only.
ROOT owns all builds/tests/native/operator/model/GPU/container execution.

- [x] Independently inspect actual first64 capture/comparison;960 preceding rows
  exact, full131072 alpha input words exact and actual producer binding closed.
- [x] Independently byte-match all32768 alpha F32 weights to original GGUF
  range[459622304,459753376); distinguish ordinary raw from opaque otherweights.
- [x] Prepare genuine public ggml full64+64AR default replay, gated publicuse_ref
  controls, unchanged ABI/return entry observer and full plain/observed validator.
- [x] Preserve existing helpers/attempts; pin all relevant capture/source/DSO files,
  bounded owned lifecycle, exact finite shapes/outputs and no internalcause claim.
- [ ] Independent source review, ROOT strict build and byte-only fixture creation.
- [ ] ROOT actual all65 default2048 words exact BEFORE65 controls; all130
  plain/observed4096 words exact BEFORE520 exported-entry route population accepted.
- [ ] Separately interpret actual control results/source branch proof. Do not alter
  declared Rust AR1 math by fitting nativebulk; no automatic full-longcontext cause.

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/README.md

Original UTF-8 bytes: 9128; SHA256 `63071037fff64668667324f49ff4de3d2868e478515e2ab420022f18e7cbf1e3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original 2B unchanged bulk64/AR1 causal observer — ROOT EXECUTES

Parent: numerics-and-context. Owner: spine_review. Source-only preparation.
No build, operator, model, native-library/container or GPU execution by author.
No MAIN tracked/plans/math edits; old helpers/frozen captures unchanged.

## Scope

- Existing approved GGUF2B SHA91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6,
  1,296,764,000B. Same actual8192 u32LE teacher file SHA44c93512…ddd7d;
  same original n_ctx8320→actual8448, private seq0, CPU8threads, F32KV,
  flash-attention/offload disabled. No quantization/profile/source substitution.
- First try original first64 rows from zero. Optional start64/128/…/960
  preserves earlier64-row prefix calls. Every selected block has actual64 rows;
  this is an early block of original8191-prefix127×64+63, not an arbitrary-cut
  bisection or a modified final selected-token execution. No automatic ladder.
  If incoming WHOLE native bulk/AR1 states differ, refuse and inspect an earlier
  aligned block. Beyond this bounded early1K region requires another reviewed
  capture budget; no unsupported8K snapshot cap escalation.
- Observe at least layers0/1 before historical selected-token GDN1 incoming-state
  differences. First *observed* tensor gap remains diagnostic; no automatic first
  cause, Rust parity or long-context eligibility claim.

## Actual operands/results

- Full embedding, all actual projection inputs and results, norm producers,
  weighted norm results, residuals, FFN up/gate/SwiGLU/final results; full-row
  original GDN coefficients/conv/Q/K/V/core, initial prior and final next state.
  GDN original K=1 snapshot policy unchanged: no intermediate states invented.
- Actual original RMS `norm` outputs selected at their real producer boundary
  (three/layer). Selecting producer boundaries prevents pretending a possibly
  overwritten in-place intermediate is the earlier operand. Per-call tensor and
  source IDs preserve original graph links; source result files retain exact
  original bits before later consumers. No arithmetic or mathematical fallback.
- Full actual SSM-conv input includes raw3-history +all64 current projections,
  ne[67,6144,1,1]. Every AR1 input ne[4,6144,1,1]. Comparator byte-selects each
  actual four-tap causal slice and compares all4×6144 raw words for every row.
  It computes no convolution or expected numerical result.
- All28 learned inputs retain actual original tensor/type/shape/byte metadata,
  separately labelled `GGUF_RAW` or `CPU_REPACK_UNDECODED`. Header-only source
  geometry covers 72,582,400B, largest7,864,320B. Actual repacked bytes are NOT
  interpreted/dequantized as the metadata's GGUF codec. Ordinary CPU/CPU_Mapped
  payloads still must equal independently pinned original GGUF bytes exactly;
  opaque repack source-codec equality remains UNQUALIFIED even after a passing
  diagnostic. Full model hash/header binding is preserved for both transports.
- Exact CPU_REPACK type pointer is selected through the public CPU registry's
  documented extra-bufts callback, dladdr-bound to the pinned Haswell DSO. Known
  raw CPU and CPU_Mapped singleton types are also checked. CPU_Mapped's private
  singleton is obtained through a public temporary descriptor over an aligned
  local64B array, immediately freed; it never aliases a model allocation.
  Unknown buffers refuse before any data access. No private iface access.
- The pinned source allocates CPU_REPACK using an ordinary CPU allocation,
  retaining get_base/free/size while replacing set_tensor and clearing get_tensor
  (repack.cpp5168–5180). The original public tensor_get would call NULL. Opaque
  reads therefore use original public buffer base+size, nonview data uintptr
  and complete ggml_nbytes bounds, then <=1MiB memcpy. No native repack setting,
  operation, arithmetic or callback is changed. Raw CPU uses public tensor_get.
- Each role must retain the same tensor/data/buffer/type/base/size/offset through
  bulk and all64 AR1 calls. Copy once in <=1MiB chunks; re-read/compare all28
  complete payloads after bulk and again after AR1. These are actual captured
  start/boundary readsets, not per-FMA observations. Numerical transparency
  against each own plain context remains mandatory. Physical-owner addresses
  are diagnostic identity within this one process, not cross-run identities.
- No attention3/cache-capacity read is selected in this two-layer slice. F32
  copier retains exact ne/nb/checked physical span, reads only occupied inner
  runs and refuses spans>16MiB. No private strided-cache exception/cap increase.

## Mandatory transparency and bounds

- Four original public contexts share one original model. Independent plain
  bulk64/AR1 prefixes; exact entire common prior required. Each observed context
  restores its own actual prior, repeats unchanged call width, then must match
  its own plain complete serialized post-state bit-for-bit. Both observed/plain
  pairs consume the next actual AR1 teacher token and must match full248320
  logits AND whole serialized state. Observation that changes math refuses.
- Both C++ and Python require every scope/stage family, all3 norm occurrences,
  all projections/learned outputs, all5 GDN operands/core and initial/final state,
  and all28 learned roles. Missing/duplicate/unpaired field refuses. No missing
  attention is represented as captured; it is outside this two-layer scope.
- Original bulk128→112MiB and AR192→112MiB caps both reduced. Field≤2M F32 words
  (8MiB), metadata≤16MiB/mode, raw learned≤80MiB in≤1MiB chunks plus128KiB
  metadata, three snapshots combined≤136MiB, two next-full-logits<2MiB.
  Worst summed capture ceilings<475MiB including IDs/metadata/completion; final own
  recursive capture tree independently checked≤480MiB with reserved completion
  space. Remaining32MiB reserves comparison≤16MiB +external log≤8MiB +
  bounded source/receipt metadata≤8MiB, whole attempt≤512MiB.
  No symlinks/nonregular artifacts, no overwrite. Failure retains partial data.
- ROOT uses existing finite owned-process supervisor (CPU/RAM/wall/log caps).
  Four contexts+one1.3GB model fit configured8GiB address ceiling as an estimate;
  only actual RSS/limits establish fit. No ResourceArbiter or physical-memory
  claim from this diagnostic. No devices/driver privileges/recovery changes.

## Root commands

Authenticate freeze-manifest.json SHA first, then verify_preparation.py. Root
reuses accepted strict GCC15/glibc2.44 public-header/backend closure, no upstream
build/fast-math. Original helper text is included with unused main renamed.

```bash
TASK_2B_PROBE=.agents/runs/numerics-and-context-2b-bulk-causality-20261020
TASK_LLAMA_SRC=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LLAMA_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
TASK_GLIBC=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
python3 "$TASK_2B_PROBE/verify_preparation.py"
python3 "$TASK_2B_PROBE/test_transport_validation.py"
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "$TASK_LLAMA_SRC/include" -I "$TASK_LLAMA_SRC/ggml/include" \
  "$TASK_2B_PROBE/native_window.cpp" -L "$TASK_LLAMA_LIB" -L "$TASK_GLIBC" \
  -Wl,-rpath,"$TASK_LLAMA_LIB:$TASK_GLIBC" \
  -Wl,--dynamic-linker,"$TASK_GLIBC/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -ldl -o "$TASK_2B_PROBE/native_window"
# Only ROOT under existing bounded supervisor; fresh output and binary hash.
"$TASK_2B_PROBE/native_window" target/models/qwen35-2b/model.gguf \
  .agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le \
  0 8 900 "$TASK_2B_PROBE/root-first64-repack-transport"
python3 "$TASK_2B_PROBE/compare_window.py" --capture "$TASK_2B_PROBE/root-first64-repack-transport" \
  --output "$TASK_2B_PROBE/root-first64-repack-transport-comparison.json"
python3 "$TASK_2B_PROBE/verify_preparation.py"
```

C++ completion means only numerical observation transparency. Comparator exit0
means a complete diagnostic was emitted, even if result rows differ. Source or
internal operator dispatch, then source-correct shared-engine implementation and
full independent42-state/logit long cases remain separate required work.

## Preserved failure and additive transport follow-up

The first actual ROOT attempt exited SIGSEGV(-11), before complete observation.
Root GDB/source isolated the NULL CPU_REPACK get_tensor callback; that attempt is
UNQUALIFIED. Original frozen-prepared/, compiled binary, partial root-first64/
and failure/GDB logs are retained unchanged. frozen-before-repack-transport/
retains the exact live pre-repair sources (including earlier row-binding fix).
The additive transport patch must be reviewed and freshly compiled by ROOT.
No compilation, native call, operator/model/container/GPU execution by author.
Prepared six descriptor-validation fixture methods have not been executed here.
Passing transport/transparency establishes diagnostic read/owner scope only;
original repack codec equality and first numerical causal operator remain open.

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/TODO.md

Original UTF-8 bytes: 1480; SHA256 `f8e37c42401f972945e54e63c673b3b752ba5beb0b27a0282d4b46a07f0d1669`.
Resolve embedded relative paths against the original path, not this volume.

````text
# numerics-and-context — original2B bulk causal observer

Owner: spine_review. This ignored run only; ROOT owns all build/runtime/plans.

- [x] Read unchanged accepted first64 observer/source/caps and original2B
  GGUF header geometry (metadata only), actual native8K teacher identity.
- [x] Prepare disjoint derivative layers0/1, raw packed learned operands,
  original norm/FFN/GDN producers, four-tap raw conv input comparison and all
  prior/post/next-full-logit/whole-state transparency gates.
- [x] Fixed512MiB attempt budget; lowered both mode caps; no math/operator replacement.
- [ ] Independent complete source review and ROOT strict build.
- [ ] ROOT actual first64 complete observation/source readset comparison.
- [ ] Locate first observed gap; require every actual causal input/dispatch
  before attributing an operator or implementing a source-correct fix.
- [ ] Full long-context independent42-domain/logit acceptance remains open.

- [x] Preserve first failed actual attempt and pre-repair bytes; source-isolate
  public CPU_REPACK get_tensor=NULL; prepare closed opaque physical transport,
  explicit codec-UNQUALIFIED boundary, owner/full-readset checks after both modes.
- [ ] Independent additive transport review, descriptor fixtures, strict fresh
  ROOT build and actual complete unchanged-window capture.
- [ ] Any opaque repack-to-original codec proof requires a separate reviewed
  unchanged original repack control; this diagnostic does not claim it.

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-before-repack-transport/README.md

Original UTF-8 bytes: 6922; SHA256 `29f0e1c078229b8df5168a7772c2992c699232aa55fe64c1d8d5452b2783fdeb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original 2B unchanged bulk64/AR1 causal observer — ROOT EXECUTES

Parent: numerics-and-context. Owner: spine_review. Source-only preparation.
No build, operator, model, native-library/container or GPU execution by author.
No MAIN tracked/plans/math edits; old helpers/frozen captures unchanged.

## Scope

- Existing approved GGUF2B SHA91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6,
  1,296,764,000B. Same actual8192 u32LE teacher file SHA44c93512…ddd7d;
  same original n_ctx8320→actual8448, private seq0, CPU8threads, F32KV,
  flash-attention/offload disabled. No quantization/profile/source substitution.
- First try original first64 rows from zero. Optional start64/128/…/960
  preserves earlier64-row prefix calls. Every selected block has actual64 rows;
  this is an early block of original8191-prefix127×64+63, not an arbitrary-cut
  bisection or a modified final selected-token execution. No automatic ladder.
  If incoming WHOLE native bulk/AR1 states differ, refuse and inspect an earlier
  aligned block. Beyond this bounded early1K region requires another reviewed
  capture budget; no unsupported8K snapshot cap escalation.
- Observe at least layers0/1 before historical selected-token GDN1 incoming-state
  differences. First *observed* tensor gap remains diagnostic; no automatic first
  cause, Rust parity or long-context eligibility claim.

## Actual operands/results

- Full embedding, all actual projection inputs and results, norm producers,
  weighted norm results, residuals, FFN up/gate/SwiGLU/final results; full-row
  original GDN coefficients/conv/Q/K/V/core, initial prior and final next state.
  GDN original K=1 snapshot policy unchanged: no intermediate states invented.
- Actual original RMS `norm` outputs selected at their real producer boundary
  (three/layer). Selecting producer boundaries prevents pretending a possibly
  overwritten in-place intermediate is the earlier operand. Per-call tensor and
  source IDs preserve original graph links; source result files retain exact
  original bits before later consumers. No arithmetic or mathematical fallback.
- Full actual SSM-conv input includes raw3-history +all64 current projections,
  ne[67,6144,1,1]. Every AR1 input ne[4,6144,1,1]. Comparator byte-selects each
  actual four-tap causal slice and compares all4×6144 raw words for every row.
  It computes no convolution or expected numerical result.
- All28 original learned inputs for these two layers are captured as their exact
  packed source bytes, including real Q4_0/Q4_1/Q8_0 matrix encodings and F32
  norms/kernel/A/dt. Header-only inspection independently records actual source
  ne/type/relative+absolute offsets: total72,582,400B, largest7,864,320B.
  One shared source owner/tensor/data/buffer identity must remain the same for
  every context/callback. Copy once in≤1MiB chunks; re-read/compare all bytes
  after both observed computations. Comparator checks all28 raw operands against
  the full-hash-pinned original GGUF payload, not dequantized/candidate values.
  Learned source invariance is checked at captured start/end; no per-FMA trace.
- No attention3/cache-capacity read is selected in this two-layer slice. F32
  copier retains exact ne/nb/checked physical span, reads only occupied inner
  runs and refuses spans>16MiB. No private strided-cache exception/cap increase.

## Mandatory transparency and bounds

- Four original public contexts share one original model. Independent plain
  bulk64/AR1 prefixes; exact entire common prior required. Each observed context
  restores its own actual prior, repeats unchanged call width, then must match
  its own plain complete serialized post-state bit-for-bit. Both observed/plain
  pairs consume the next actual AR1 teacher token and must match full248320
  logits AND whole serialized state. Observation that changes math refuses.
- Both C++ and Python require every scope/stage family, all3 norm occurrences,
  all projections/learned outputs, all5 GDN operands/core and initial/final state,
  and all28 learned roles. Missing/duplicate/unpaired field refuses. No missing
  attention is represented as captured; it is outside this two-layer scope.
- Original bulk128→112MiB and AR192→112MiB caps both reduced. Field≤2M F32 words
  (8MiB), metadata≤16MiB/mode, raw learned≤80MiB in≤1MiB chunks plus128KiB
  metadata, three snapshots combined≤136MiB, two next-full-logits<2MiB.
  Worst summed capture ceilings<475MiB including IDs/metadata/completion; final own
  recursive capture tree independently checked≤480MiB with reserved completion
  space. Remaining32MiB reserves comparison≤16MiB +external log≤8MiB +
  bounded source/receipt metadata≤8MiB, whole attempt≤512MiB.
  No symlinks/nonregular artifacts, no overwrite. Failure retains partial data.
- ROOT uses existing finite owned-process supervisor (CPU/RAM/wall/log caps).
  Four contexts+one1.3GB model fit configured8GiB address ceiling as an estimate;
  only actual RSS/limits establish fit. No ResourceArbiter or physical-memory
  claim from this diagnostic. No devices/driver privileges/recovery changes.

## Root commands

Authenticate freeze-manifest.json SHA first, then verify_preparation.py. Root
reuses accepted strict GCC15/glibc2.44 public-header/backend closure, no upstream
build/fast-math. Original helper text is included with unused main renamed.

```bash
TASK_2B_PROBE=.agents/runs/numerics-and-context-2b-bulk-causality-20261020
TASK_LLAMA_SRC=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LLAMA_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
TASK_GLIBC=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
python3 "$TASK_2B_PROBE/verify_preparation.py"
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "$TASK_LLAMA_SRC/include" -I "$TASK_LLAMA_SRC/ggml/include" \
  "$TASK_2B_PROBE/native_window.cpp" -L "$TASK_LLAMA_LIB" -L "$TASK_GLIBC" \
  -Wl,-rpath,"$TASK_LLAMA_LIB:$TASK_GLIBC" \
  -Wl,--dynamic-linker,"$TASK_GLIBC/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -o "$TASK_2B_PROBE/native_window"
# Only ROOT under existing bounded supervisor; fresh output and binary hash.
"$TASK_2B_PROBE/native_window" target/models/qwen35-2b/model.gguf \
  .agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le \
  0 8 900 "$TASK_2B_PROBE/root-first64"
python3 "$TASK_2B_PROBE/compare_window.py" --capture "$TASK_2B_PROBE/root-first64" \
  --output "$TASK_2B_PROBE/root-first64-comparison.json"
python3 "$TASK_2B_PROBE/verify_preparation.py"
```

C++ completion means only numerical observation transparency. Comparator exit0
means a complete diagnostic was emitted, even if result rows differ. Source or
internal operator dispatch, then source-correct shared-engine implementation and
full independent42-state/logit long cases remain separate required work.

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-before-repack-transport/TODO.md

Original UTF-8 bytes: 950; SHA256 `cc8319fe3e018bd162635c8ad32e318562582f90ff1f925d84102c3ce54d8c18`.
Resolve embedded relative paths against the original path, not this volume.

````text
# numerics-and-context — original2B bulk causal observer

Owner: spine_review. This ignored run only; ROOT owns all build/runtime/plans.

- [x] Read unchanged accepted first64 observer/source/caps and original2B
  GGUF header geometry (metadata only), actual native8K teacher identity.
- [x] Prepare disjoint derivative layers0/1, raw packed learned operands,
  original norm/FFN/GDN producers, four-tap raw conv input comparison and all
  prior/post/next-full-logit/whole-state transparency gates.
- [x] Fixed512MiB attempt budget; lowered both mode caps; no math/operator replacement.
- [ ] Independent complete source review and ROOT strict build.
- [ ] ROOT actual first64 complete observation/source readset comparison.
- [ ] Locate first observed gap; require every actual causal input/dispatch
  before attributing an operator or implementing a source-correct fix.
- [ ] Full long-context independent42-domain/logit acceptance remains open.

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 6922; SHA256 `29f0e1c078229b8df5168a7772c2992c699232aa55fe64c1d8d5452b2783fdeb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original 2B unchanged bulk64/AR1 causal observer — ROOT EXECUTES

Parent: numerics-and-context. Owner: spine_review. Source-only preparation.
No build, operator, model, native-library/container or GPU execution by author.
No MAIN tracked/plans/math edits; old helpers/frozen captures unchanged.

## Scope

- Existing approved GGUF2B SHA91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6,
  1,296,764,000B. Same actual8192 u32LE teacher file SHA44c93512…ddd7d;
  same original n_ctx8320→actual8448, private seq0, CPU8threads, F32KV,
  flash-attention/offload disabled. No quantization/profile/source substitution.
- First try original first64 rows from zero. Optional start64/128/…/960
  preserves earlier64-row prefix calls. Every selected block has actual64 rows;
  this is an early block of original8191-prefix127×64+63, not an arbitrary-cut
  bisection or a modified final selected-token execution. No automatic ladder.
  If incoming WHOLE native bulk/AR1 states differ, refuse and inspect an earlier
  aligned block. Beyond this bounded early1K region requires another reviewed
  capture budget; no unsupported8K snapshot cap escalation.
- Observe at least layers0/1 before historical selected-token GDN1 incoming-state
  differences. First *observed* tensor gap remains diagnostic; no automatic first
  cause, Rust parity or long-context eligibility claim.

## Actual operands/results

- Full embedding, all actual projection inputs and results, norm producers,
  weighted norm results, residuals, FFN up/gate/SwiGLU/final results; full-row
  original GDN coefficients/conv/Q/K/V/core, initial prior and final next state.
  GDN original K=1 snapshot policy unchanged: no intermediate states invented.
- Actual original RMS `norm` outputs selected at their real producer boundary
  (three/layer). Selecting producer boundaries prevents pretending a possibly
  overwritten in-place intermediate is the earlier operand. Per-call tensor and
  source IDs preserve original graph links; source result files retain exact
  original bits before later consumers. No arithmetic or mathematical fallback.
- Full actual SSM-conv input includes raw3-history +all64 current projections,
  ne[67,6144,1,1]. Every AR1 input ne[4,6144,1,1]. Comparator byte-selects each
  actual four-tap causal slice and compares all4×6144 raw words for every row.
  It computes no convolution or expected numerical result.
- All28 original learned inputs for these two layers are captured as their exact
  packed source bytes, including real Q4_0/Q4_1/Q8_0 matrix encodings and F32
  norms/kernel/A/dt. Header-only inspection independently records actual source
  ne/type/relative+absolute offsets: total72,582,400B, largest7,864,320B.
  One shared source owner/tensor/data/buffer identity must remain the same for
  every context/callback. Copy once in≤1MiB chunks; re-read/compare all bytes
  after both observed computations. Comparator checks all28 raw operands against
  the full-hash-pinned original GGUF payload, not dequantized/candidate values.
  Learned source invariance is checked at captured start/end; no per-FMA trace.
- No attention3/cache-capacity read is selected in this two-layer slice. F32
  copier retains exact ne/nb/checked physical span, reads only occupied inner
  runs and refuses spans>16MiB. No private strided-cache exception/cap increase.

## Mandatory transparency and bounds

- Four original public contexts share one original model. Independent plain
  bulk64/AR1 prefixes; exact entire common prior required. Each observed context
  restores its own actual prior, repeats unchanged call width, then must match
  its own plain complete serialized post-state bit-for-bit. Both observed/plain
  pairs consume the next actual AR1 teacher token and must match full248320
  logits AND whole serialized state. Observation that changes math refuses.
- Both C++ and Python require every scope/stage family, all3 norm occurrences,
  all projections/learned outputs, all5 GDN operands/core and initial/final state,
  and all28 learned roles. Missing/duplicate/unpaired field refuses. No missing
  attention is represented as captured; it is outside this two-layer scope.
- Original bulk128→112MiB and AR192→112MiB caps both reduced. Field≤2M F32 words
  (8MiB), metadata≤16MiB/mode, raw learned≤80MiB in≤1MiB chunks plus128KiB
  metadata, three snapshots combined≤136MiB, two next-full-logits<2MiB.
  Worst summed capture ceilings<475MiB including IDs/metadata/completion; final own
  recursive capture tree independently checked≤480MiB with reserved completion
  space. Remaining32MiB reserves comparison≤16MiB +external log≤8MiB +
  bounded source/receipt metadata≤8MiB, whole attempt≤512MiB.
  No symlinks/nonregular artifacts, no overwrite. Failure retains partial data.
- ROOT uses existing finite owned-process supervisor (CPU/RAM/wall/log caps).
  Four contexts+one1.3GB model fit configured8GiB address ceiling as an estimate;
  only actual RSS/limits establish fit. No ResourceArbiter or physical-memory
  claim from this diagnostic. No devices/driver privileges/recovery changes.

## Root commands

Authenticate freeze-manifest.json SHA first, then verify_preparation.py. Root
reuses accepted strict GCC15/glibc2.44 public-header/backend closure, no upstream
build/fast-math. Original helper text is included with unused main renamed.

```bash
TASK_2B_PROBE=.agents/runs/numerics-and-context-2b-bulk-causality-20261020
TASK_LLAMA_SRC=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LLAMA_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
TASK_GLIBC=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
python3 "$TASK_2B_PROBE/verify_preparation.py"
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "$TASK_LLAMA_SRC/include" -I "$TASK_LLAMA_SRC/ggml/include" \
  "$TASK_2B_PROBE/native_window.cpp" -L "$TASK_LLAMA_LIB" -L "$TASK_GLIBC" \
  -Wl,-rpath,"$TASK_LLAMA_LIB:$TASK_GLIBC" \
  -Wl,--dynamic-linker,"$TASK_GLIBC/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -o "$TASK_2B_PROBE/native_window"
# Only ROOT under existing bounded supervisor; fresh output and binary hash.
"$TASK_2B_PROBE/native_window" target/models/qwen35-2b/model.gguf \
  .agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le \
  0 8 900 "$TASK_2B_PROBE/root-first64"
python3 "$TASK_2B_PROBE/compare_window.py" --capture "$TASK_2B_PROBE/root-first64" \
  --output "$TASK_2B_PROBE/root-first64-comparison.json"
python3 "$TASK_2B_PROBE/verify_preparation.py"
```

C++ completion means only numerical observation transparency. Comparator exit0
means a complete diagnostic was emitted, even if result rows differ. Source or
internal operator dispatch, then source-correct shared-engine implementation and
full independent42-state/logit long cases remain separate required work.

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 950; SHA256 `cc8319fe3e018bd162635c8ad32e318562582f90ff1f925d84102c3ce54d8c18`.
Resolve embedded relative paths against the original path, not this volume.

````text
# numerics-and-context — original2B bulk causal observer

Owner: spine_review. This ignored run only; ROOT owns all build/runtime/plans.

- [x] Read unchanged accepted first64 observer/source/caps and original2B
  GGUF header geometry (metadata only), actual native8K teacher identity.
- [x] Prepare disjoint derivative layers0/1, raw packed learned operands,
  original norm/FFN/GDN producers, four-tap raw conv input comparison and all
  prior/post/next-full-logit/whole-state transparency gates.
- [x] Fixed512MiB attempt budget; lowered both mode caps; no math/operator replacement.
- [ ] Independent complete source review and ROOT strict build.
- [ ] ROOT actual first64 complete observation/source readset comparison.
- [ ] Locate first observed gap; require every actual causal input/dispatch
  before attributing an operator or implementing a source-correct fix.
- [ ] Full long-context independent42-domain/logit acceptance remains open.

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-repack-transport-followup/README.md

Original UTF-8 bytes: 9128; SHA256 `63071037fff64668667324f49ff4de3d2868e478515e2ab420022f18e7cbf1e3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original 2B unchanged bulk64/AR1 causal observer — ROOT EXECUTES

Parent: numerics-and-context. Owner: spine_review. Source-only preparation.
No build, operator, model, native-library/container or GPU execution by author.
No MAIN tracked/plans/math edits; old helpers/frozen captures unchanged.

## Scope

- Existing approved GGUF2B SHA91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6,
  1,296,764,000B. Same actual8192 u32LE teacher file SHA44c93512…ddd7d;
  same original n_ctx8320→actual8448, private seq0, CPU8threads, F32KV,
  flash-attention/offload disabled. No quantization/profile/source substitution.
- First try original first64 rows from zero. Optional start64/128/…/960
  preserves earlier64-row prefix calls. Every selected block has actual64 rows;
  this is an early block of original8191-prefix127×64+63, not an arbitrary-cut
  bisection or a modified final selected-token execution. No automatic ladder.
  If incoming WHOLE native bulk/AR1 states differ, refuse and inspect an earlier
  aligned block. Beyond this bounded early1K region requires another reviewed
  capture budget; no unsupported8K snapshot cap escalation.
- Observe at least layers0/1 before historical selected-token GDN1 incoming-state
  differences. First *observed* tensor gap remains diagnostic; no automatic first
  cause, Rust parity or long-context eligibility claim.

## Actual operands/results

- Full embedding, all actual projection inputs and results, norm producers,
  weighted norm results, residuals, FFN up/gate/SwiGLU/final results; full-row
  original GDN coefficients/conv/Q/K/V/core, initial prior and final next state.
  GDN original K=1 snapshot policy unchanged: no intermediate states invented.
- Actual original RMS `norm` outputs selected at their real producer boundary
  (three/layer). Selecting producer boundaries prevents pretending a possibly
  overwritten in-place intermediate is the earlier operand. Per-call tensor and
  source IDs preserve original graph links; source result files retain exact
  original bits before later consumers. No arithmetic or mathematical fallback.
- Full actual SSM-conv input includes raw3-history +all64 current projections,
  ne[67,6144,1,1]. Every AR1 input ne[4,6144,1,1]. Comparator byte-selects each
  actual four-tap causal slice and compares all4×6144 raw words for every row.
  It computes no convolution or expected numerical result.
- All28 learned inputs retain actual original tensor/type/shape/byte metadata,
  separately labelled `GGUF_RAW` or `CPU_REPACK_UNDECODED`. Header-only source
  geometry covers 72,582,400B, largest7,864,320B. Actual repacked bytes are NOT
  interpreted/dequantized as the metadata's GGUF codec. Ordinary CPU/CPU_Mapped
  payloads still must equal independently pinned original GGUF bytes exactly;
  opaque repack source-codec equality remains UNQUALIFIED even after a passing
  diagnostic. Full model hash/header binding is preserved for both transports.
- Exact CPU_REPACK type pointer is selected through the public CPU registry's
  documented extra-bufts callback, dladdr-bound to the pinned Haswell DSO. Known
  raw CPU and CPU_Mapped singleton types are also checked. CPU_Mapped's private
  singleton is obtained through a public temporary descriptor over an aligned
  local64B array, immediately freed; it never aliases a model allocation.
  Unknown buffers refuse before any data access. No private iface access.
- The pinned source allocates CPU_REPACK using an ordinary CPU allocation,
  retaining get_base/free/size while replacing set_tensor and clearing get_tensor
  (repack.cpp5168–5180). The original public tensor_get would call NULL. Opaque
  reads therefore use original public buffer base+size, nonview data uintptr
  and complete ggml_nbytes bounds, then <=1MiB memcpy. No native repack setting,
  operation, arithmetic or callback is changed. Raw CPU uses public tensor_get.
- Each role must retain the same tensor/data/buffer/type/base/size/offset through
  bulk and all64 AR1 calls. Copy once in <=1MiB chunks; re-read/compare all28
  complete payloads after bulk and again after AR1. These are actual captured
  start/boundary readsets, not per-FMA observations. Numerical transparency
  against each own plain context remains mandatory. Physical-owner addresses
  are diagnostic identity within this one process, not cross-run identities.
- No attention3/cache-capacity read is selected in this two-layer slice. F32
  copier retains exact ne/nb/checked physical span, reads only occupied inner
  runs and refuses spans>16MiB. No private strided-cache exception/cap increase.

## Mandatory transparency and bounds

- Four original public contexts share one original model. Independent plain
  bulk64/AR1 prefixes; exact entire common prior required. Each observed context
  restores its own actual prior, repeats unchanged call width, then must match
  its own plain complete serialized post-state bit-for-bit. Both observed/plain
  pairs consume the next actual AR1 teacher token and must match full248320
  logits AND whole serialized state. Observation that changes math refuses.
- Both C++ and Python require every scope/stage family, all3 norm occurrences,
  all projections/learned outputs, all5 GDN operands/core and initial/final state,
  and all28 learned roles. Missing/duplicate/unpaired field refuses. No missing
  attention is represented as captured; it is outside this two-layer scope.
- Original bulk128→112MiB and AR192→112MiB caps both reduced. Field≤2M F32 words
  (8MiB), metadata≤16MiB/mode, raw learned≤80MiB in≤1MiB chunks plus128KiB
  metadata, three snapshots combined≤136MiB, two next-full-logits<2MiB.
  Worst summed capture ceilings<475MiB including IDs/metadata/completion; final own
  recursive capture tree independently checked≤480MiB with reserved completion
  space. Remaining32MiB reserves comparison≤16MiB +external log≤8MiB +
  bounded source/receipt metadata≤8MiB, whole attempt≤512MiB.
  No symlinks/nonregular artifacts, no overwrite. Failure retains partial data.
- ROOT uses existing finite owned-process supervisor (CPU/RAM/wall/log caps).
  Four contexts+one1.3GB model fit configured8GiB address ceiling as an estimate;
  only actual RSS/limits establish fit. No ResourceArbiter or physical-memory
  claim from this diagnostic. No devices/driver privileges/recovery changes.

## Root commands

Authenticate freeze-manifest.json SHA first, then verify_preparation.py. Root
reuses accepted strict GCC15/glibc2.44 public-header/backend closure, no upstream
build/fast-math. Original helper text is included with unused main renamed.

```bash
TASK_2B_PROBE=.agents/runs/numerics-and-context-2b-bulk-causality-20261020
TASK_LLAMA_SRC=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LLAMA_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
TASK_GLIBC=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
python3 "$TASK_2B_PROBE/verify_preparation.py"
python3 "$TASK_2B_PROBE/test_transport_validation.py"
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "$TASK_LLAMA_SRC/include" -I "$TASK_LLAMA_SRC/ggml/include" \
  "$TASK_2B_PROBE/native_window.cpp" -L "$TASK_LLAMA_LIB" -L "$TASK_GLIBC" \
  -Wl,-rpath,"$TASK_LLAMA_LIB:$TASK_GLIBC" \
  -Wl,--dynamic-linker,"$TASK_GLIBC/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -ldl -o "$TASK_2B_PROBE/native_window"
# Only ROOT under existing bounded supervisor; fresh output and binary hash.
"$TASK_2B_PROBE/native_window" target/models/qwen35-2b/model.gguf \
  .agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le \
  0 8 900 "$TASK_2B_PROBE/root-first64-repack-transport"
python3 "$TASK_2B_PROBE/compare_window.py" --capture "$TASK_2B_PROBE/root-first64-repack-transport" \
  --output "$TASK_2B_PROBE/root-first64-repack-transport-comparison.json"
python3 "$TASK_2B_PROBE/verify_preparation.py"
```

C++ completion means only numerical observation transparency. Comparator exit0
means a complete diagnostic was emitted, even if result rows differ. Source or
internal operator dispatch, then source-correct shared-engine implementation and
full independent42-state/logit long cases remain separate required work.

## Preserved failure and additive transport follow-up

The first actual ROOT attempt exited SIGSEGV(-11), before complete observation.
Root GDB/source isolated the NULL CPU_REPACK get_tensor callback; that attempt is
UNQUALIFIED. Original frozen-prepared/, compiled binary, partial root-first64/
and failure/GDB logs are retained unchanged. frozen-before-repack-transport/
retains the exact live pre-repair sources (including earlier row-binding fix).
The additive transport patch must be reviewed and freshly compiled by ROOT.
No compilation, native call, operator/model/container/GPU execution by author.
Prepared six descriptor-validation fixture methods have not been executed here.
Passing transport/transparency establishes diagnostic read/owner scope only;
original repack codec equality and first numerical causal operator remain open.

````

## Source: .agents/runs/numerics-and-context-2b-bulk-causality-20261020/frozen-repack-transport-followup/TODO.md

Original UTF-8 bytes: 1480; SHA256 `f8e37c42401f972945e54e63c673b3b752ba5beb0b27a0282d4b46a07f0d1669`.
Resolve embedded relative paths against the original path, not this volume.

````text
# numerics-and-context — original2B bulk causal observer

Owner: spine_review. This ignored run only; ROOT owns all build/runtime/plans.

- [x] Read unchanged accepted first64 observer/source/caps and original2B
  GGUF header geometry (metadata only), actual native8K teacher identity.
- [x] Prepare disjoint derivative layers0/1, raw packed learned operands,
  original norm/FFN/GDN producers, four-tap raw conv input comparison and all
  prior/post/next-full-logit/whole-state transparency gates.
- [x] Fixed512MiB attempt budget; lowered both mode caps; no math/operator replacement.
- [ ] Independent complete source review and ROOT strict build.
- [ ] ROOT actual first64 complete observation/source readset comparison.
- [ ] Locate first observed gap; require every actual causal input/dispatch
  before attributing an operator or implementing a source-correct fix.
- [ ] Full long-context independent42-domain/logit acceptance remains open.

- [x] Preserve first failed actual attempt and pre-repair bytes; source-isolate
  public CPU_REPACK get_tensor=NULL; prepare closed opaque physical transport,
  explicit codec-UNQUALIFIED boundary, owner/full-readset checks after both modes.
- [ ] Independent additive transport review, descriptor fixtures, strict fresh
  ROOT build and actual complete unchanged-window capture.
- [ ] Any opaque repack-to-original codec proof requires a separate reviewed
  unchanged original repack control; this diagnostic does not claim it.

````

## Source: .agents/runs/numerics-and-context-av-dispatch-20261020/README.md

Original UTF-8 bytes: 6621; SHA256 `4e781c20c07e80c8bfa2778ceafe3a18c3075351c90f5969610e834b9b772bbe`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Transparent AV public-entry observer — ROOT EXECUTES

Parent: numerics-and-context. Owner: spine_review. Source-only preparation;
no build, library load, public operator, model or GPU execution by this agent.
Original AV helpers/captures remain unchanged. ROOT first reviews/builds/runs.

## Boundary

- Original public AV replay already reproduced all65 default outputs exactly;
  public use_ref bulk matched all64 original AR1 outputs, while default bulk
  differed. This run observes the exported dispatch entry, without replacing
  arithmetic or treating that prior result as a complete long-context cause.
- Exact installed Haswell ELF exports GLOBAL/DEFAULT `llamafile_sgemm`, has
  an R_X86_64_JUMP_SLOT call relocation and no SYMBOLIC binding. Its exact
  original `sgemm.h` C declaration is included. `ggml_compute_params` stays
  opaque: no private ABI fields are read. All13 arguments and original bool
  result pass through unchanged, exactly once.
- Backend loader uses RTLD_NOW|RTLD_LOCAL. Resolve the real function through
  an exact-path RTLD_NOLOAD handle; require dladdr path and symbol0x10c8f0.
  RTLD_NEXT alone cannot establish ownership of a locally loaded backend.
- Ring has2048 integer-only records;130 ordered compute scopes. Resolve before
  computation; no per-call allocation, logging, symbol resolution, float math
  or output inspection. Main closes a scope only after original synchronous
  graph completion. New trace is written once after all original computations.
  Unexpected/overflow observations refuse; they never return an invented bool
  that could choose a different mathematical path.
- Public controls65 run only after all65 defaults match. The adapter only
  adds begin/end/save calls, and keeps original views/strides/readsets/output
  and expected-read-after-compute order. Plain adapter is run first. ALL130
  outputs from BOTH new runs must equal the authenticated original130 outputs
  before any route observation is accepted. Full signed-zero/raw-bit equality.

## Source route hypothesis, validated only after actual trace

- `ggml-cpu.c1308`: without use_ref, contiguous F32 P tries llamafile per head.
  F32 vec_dot_type isF32, so the second conversion gate1387 is inapplicable.
- F32 V is rejected by tiled.cpp's quant-only supported-type gate. The sole
  fusion in this source is RMS_NORM+MUL; isolated AV graph has neither pair.
- Default bulk64:8 workers×8 query heads=64 calls, all true. Shape m/n/k is
  256/64/256, lda/ldb/ldc8448/256/256. Head h uses A offset(h/4)*8650752,
  B/C offseth*64*256*4; exact GQA2→8.
- Default AR1:8 calls at first head; original `sgemm.cpp` n<2 gate returns
  false, falling through to original vec_dot. All64 cases retained.
- Public use_ref: all65 cases bypass this entry, zero calls.
- Under source AVX2 branch, accepted F32 bulk uses tinyBLAS Vec8 and fallback
  uses four Vec8 accumulators. **This observer does not see internal kernels**,
  ISA instructions or all libc/libm dispatch. No compiled-schedule/math port,
  production fix, full runtime closure or model/profile eligibility claim.

## Root commands

Use accepted strict GCC15 / glibc2.44 environment. No fast-math or upstream
rebuild. Authenticate source-pins.json/freeze-manifest.json before building.

```bash
TASK_AV_OBS=.agents/runs/numerics-and-context-av-dispatch-20261020
TASK_AV_BASE=.agents/runs/numerics-and-context-av-replay-20261020
TASK_LLAMA_SRC=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LLAMA_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
c++ -std=c++20 -O2 -Wall -Wextra -Werror -fPIC -shared \
  -I "$TASK_LLAMA_SRC/ggml/src/ggml-cpu/llamafile" \
  "$TASK_AV_OBS/observer.cpp" -ldl -Wl,-z,defs -o "$TASK_AV_OBS/observer.so"
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I "$TASK_LLAMA_SRC/ggml/include" "$TASK_AV_OBS/main.cpp" \
  -L "$TASK_LLAMA_LIB" -Wl,-rpath,"$TASK_LLAMA_LIB" \
  -lggml -lggml-base -ldl -o "$TASK_AV_OBS/replay"
sha256sum "$TASK_AV_OBS/replay" "$TASK_AV_OBS/observer.so"
python3 "$TASK_AV_OBS/run_replay.py" \
  --binary "$TASK_AV_OBS/replay" --binary-sha256 FRESH_REPLAY_SHA \
  --fixtures "$TASK_AV_BASE/root-fixtures" \
  --new-output "$TASK_AV_OBS/root-plain" --new-log "$TASK_AV_OBS/root-plain.log" \
  --new-receipt "$TASK_AV_OBS/root-plain.invocation.json"
python3 "$TASK_AV_OBS/run_replay.py" \
  --binary "$TASK_AV_OBS/replay" --binary-sha256 FRESH_REPLAY_SHA \
  --preload "$TASK_AV_OBS/observer.so" --preload-sha256 FRESH_OBSERVER_SHA \
  --fixtures "$TASK_AV_BASE/root-fixtures" \
  --new-output "$TASK_AV_OBS/root-observed" --new-log "$TASK_AV_OBS/root-observed.log" \
  --new-receipt "$TASK_AV_OBS/root-observed.invocation.json"
python3 "$TASK_AV_OBS/validate.py" \
  --plain-output "$TASK_AV_OBS/root-plain" --plain-receipt "$TASK_AV_OBS/root-plain.invocation.json" \
  --observed-output "$TASK_AV_OBS/root-observed" --observed-receipt "$TASK_AV_OBS/root-observed.invocation.json"
```

- ROOT source-pin checks include original accepted sources, captures, DSOs,
  loader, baseline130-output receipt and unchanged old helpers. Existing
  accepted fixtures reused; no copy/overwrite. Runner authenticates unchanged
  helper before import and all input hashes before/after execution.
- Same original8threads,300s wall,8GiB address-space cap, native backing≤32MiB,
  no GPU/model loading. These are configured limits, not host physical-RAM or
  ResourceArbiter qualification. Two runs are serial. New combined outputs
  ≤512MiB; original fixtures~36MiB, actual130-output payload2MiB/run,
  trace≤1MiB, ring≤2048 records; per-run external log≤8MiB.
- Exact clean environment; preload path/hash explicit, plain run has neither
  preload nor enable variable. Original GNU loader and library search used.
- Owned new-session PGID cleanup is unchanged from reviewed followup: leader
  held unreaped via WNOWAIT, unconditional TERM/KILL including after leader
  exit, bounded group absence and pipe EOF, separate failure receipts. No
  arbitrary external process/container cleanup.
- Upstream header remains external/unmodified under copied MIT notice.
  Own observer/adapter Apache2.0. No upstream arithmetic is copied here.

## Remaining

- Independent source review; root strict build/actual plain+observed runs.
- All130 outputs exact before accepting all130 route scopes/576 entries.
- Internal kernel/full compiled dependency closure remains unobserved.
- Full independent long-context42-state/logit/model cause and source-correct
  shared-engine fix remain separate gates. No tolerance/profile workaround.

````

## Source: .agents/runs/numerics-and-context-av-dispatch-20261020/TODO.md

Original UTF-8 bytes: 769; SHA256 `d1a74db2c3b81af2046db6388bd9dd1c6908ee5e3502adc0992a33d1d7e70fb4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# numerics-and-context — AV public-entry route

Owner: spine_review. Scope: this ignored run only. Root is sole build/runtime
coordinator. No MAIN/plans/model math edits; old helpers/captures preserved.

- [x] Source/ELF-only exported ABI and PLT/local-loader feasibility audit.
- [x] Opaque passthrough observer, unchanged public-AV adapter, bounded owned
  runner and fail-closed130-output transparency/130-scope route validator.
- [x] Python AST-only parse and complete source delta review. No execution.
- [ ] Independent source review + ROOT strict build.
- [ ] ROOT plain then observed full130 raw output equality + route receipts.
- [ ] Future independent long-context/full42 cause closure and source fix;
  this diagnostic alone does not mark those complete.

````

## Source: .agents/runs/numerics-and-context-av-dispatch-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 6621; SHA256 `4e781c20c07e80c8bfa2778ceafe3a18c3075351c90f5969610e834b9b772bbe`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Transparent AV public-entry observer — ROOT EXECUTES

Parent: numerics-and-context. Owner: spine_review. Source-only preparation;
no build, library load, public operator, model or GPU execution by this agent.
Original AV helpers/captures remain unchanged. ROOT first reviews/builds/runs.

## Boundary

- Original public AV replay already reproduced all65 default outputs exactly;
  public use_ref bulk matched all64 original AR1 outputs, while default bulk
  differed. This run observes the exported dispatch entry, without replacing
  arithmetic or treating that prior result as a complete long-context cause.
- Exact installed Haswell ELF exports GLOBAL/DEFAULT `llamafile_sgemm`, has
  an R_X86_64_JUMP_SLOT call relocation and no SYMBOLIC binding. Its exact
  original `sgemm.h` C declaration is included. `ggml_compute_params` stays
  opaque: no private ABI fields are read. All13 arguments and original bool
  result pass through unchanged, exactly once.
- Backend loader uses RTLD_NOW|RTLD_LOCAL. Resolve the real function through
  an exact-path RTLD_NOLOAD handle; require dladdr path and symbol0x10c8f0.
  RTLD_NEXT alone cannot establish ownership of a locally loaded backend.
- Ring has2048 integer-only records;130 ordered compute scopes. Resolve before
  computation; no per-call allocation, logging, symbol resolution, float math
  or output inspection. Main closes a scope only after original synchronous
  graph completion. New trace is written once after all original computations.
  Unexpected/overflow observations refuse; they never return an invented bool
  that could choose a different mathematical path.
- Public controls65 run only after all65 defaults match. The adapter only
  adds begin/end/save calls, and keeps original views/strides/readsets/output
  and expected-read-after-compute order. Plain adapter is run first. ALL130
  outputs from BOTH new runs must equal the authenticated original130 outputs
  before any route observation is accepted. Full signed-zero/raw-bit equality.

## Source route hypothesis, validated only after actual trace

- `ggml-cpu.c1308`: without use_ref, contiguous F32 P tries llamafile per head.
  F32 vec_dot_type isF32, so the second conversion gate1387 is inapplicable.
- F32 V is rejected by tiled.cpp's quant-only supported-type gate. The sole
  fusion in this source is RMS_NORM+MUL; isolated AV graph has neither pair.
- Default bulk64:8 workers×8 query heads=64 calls, all true. Shape m/n/k is
  256/64/256, lda/ldb/ldc8448/256/256. Head h uses A offset(h/4)*8650752,
  B/C offseth*64*256*4; exact GQA2→8.
- Default AR1:8 calls at first head; original `sgemm.cpp` n<2 gate returns
  false, falling through to original vec_dot. All64 cases retained.
- Public use_ref: all65 cases bypass this entry, zero calls.
- Under source AVX2 branch, accepted F32 bulk uses tinyBLAS Vec8 and fallback
  uses four Vec8 accumulators. **This observer does not see internal kernels**,
  ISA instructions or all libc/libm dispatch. No compiled-schedule/math port,
  production fix, full runtime closure or model/profile eligibility claim.

## Root commands

Use accepted strict GCC15 / glibc2.44 environment. No fast-math or upstream
rebuild. Authenticate source-pins.json/freeze-manifest.json before building.

```bash
TASK_AV_OBS=.agents/runs/numerics-and-context-av-dispatch-20261020
TASK_AV_BASE=.agents/runs/numerics-and-context-av-replay-20261020
TASK_LLAMA_SRC=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LLAMA_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
c++ -std=c++20 -O2 -Wall -Wextra -Werror -fPIC -shared \
  -I "$TASK_LLAMA_SRC/ggml/src/ggml-cpu/llamafile" \
  "$TASK_AV_OBS/observer.cpp" -ldl -Wl,-z,defs -o "$TASK_AV_OBS/observer.so"
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I "$TASK_LLAMA_SRC/ggml/include" "$TASK_AV_OBS/main.cpp" \
  -L "$TASK_LLAMA_LIB" -Wl,-rpath,"$TASK_LLAMA_LIB" \
  -lggml -lggml-base -ldl -o "$TASK_AV_OBS/replay"
sha256sum "$TASK_AV_OBS/replay" "$TASK_AV_OBS/observer.so"
python3 "$TASK_AV_OBS/run_replay.py" \
  --binary "$TASK_AV_OBS/replay" --binary-sha256 FRESH_REPLAY_SHA \
  --fixtures "$TASK_AV_BASE/root-fixtures" \
  --new-output "$TASK_AV_OBS/root-plain" --new-log "$TASK_AV_OBS/root-plain.log" \
  --new-receipt "$TASK_AV_OBS/root-plain.invocation.json"
python3 "$TASK_AV_OBS/run_replay.py" \
  --binary "$TASK_AV_OBS/replay" --binary-sha256 FRESH_REPLAY_SHA \
  --preload "$TASK_AV_OBS/observer.so" --preload-sha256 FRESH_OBSERVER_SHA \
  --fixtures "$TASK_AV_BASE/root-fixtures" \
  --new-output "$TASK_AV_OBS/root-observed" --new-log "$TASK_AV_OBS/root-observed.log" \
  --new-receipt "$TASK_AV_OBS/root-observed.invocation.json"
python3 "$TASK_AV_OBS/validate.py" \
  --plain-output "$TASK_AV_OBS/root-plain" --plain-receipt "$TASK_AV_OBS/root-plain.invocation.json" \
  --observed-output "$TASK_AV_OBS/root-observed" --observed-receipt "$TASK_AV_OBS/root-observed.invocation.json"
```

- ROOT source-pin checks include original accepted sources, captures, DSOs,
  loader, baseline130-output receipt and unchanged old helpers. Existing
  accepted fixtures reused; no copy/overwrite. Runner authenticates unchanged
  helper before import and all input hashes before/after execution.
- Same original8threads,300s wall,8GiB address-space cap, native backing≤32MiB,
  no GPU/model loading. These are configured limits, not host physical-RAM or
  ResourceArbiter qualification. Two runs are serial. New combined outputs
  ≤512MiB; original fixtures~36MiB, actual130-output payload2MiB/run,
  trace≤1MiB, ring≤2048 records; per-run external log≤8MiB.
- Exact clean environment; preload path/hash explicit, plain run has neither
  preload nor enable variable. Original GNU loader and library search used.
- Owned new-session PGID cleanup is unchanged from reviewed followup: leader
  held unreaped via WNOWAIT, unconditional TERM/KILL including after leader
  exit, bounded group absence and pipe EOF, separate failure receipts. No
  arbitrary external process/container cleanup.
- Upstream header remains external/unmodified under copied MIT notice.
  Own observer/adapter Apache2.0. No upstream arithmetic is copied here.

## Remaining

- Independent source review; root strict build/actual plain+observed runs.
- All130 outputs exact before accepting all130 route scopes/576 entries.
- Internal kernel/full compiled dependency closure remains unobserved.
- Full independent long-context42-state/logit/model cause and source-correct
  shared-engine fix remain separate gates. No tolerance/profile workaround.

````

## Source: .agents/runs/numerics-and-context-av-dispatch-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 769; SHA256 `d1a74db2c3b81af2046db6388bd9dd1c6908ee5e3502adc0992a33d1d7e70fb4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# numerics-and-context — AV public-entry route

Owner: spine_review. Scope: this ignored run only. Root is sole build/runtime
coordinator. No MAIN/plans/model math edits; old helpers/captures preserved.

- [x] Source/ELF-only exported ABI and PLT/local-loader feasibility audit.
- [x] Opaque passthrough observer, unchanged public-AV adapter, bounded owned
  runner and fail-closed130-output transparency/130-scope route validator.
- [x] Python AST-only parse and complete source delta review. No execution.
- [ ] Independent source review + ROOT strict build.
- [ ] ROOT plain then observed full130 raw output equality + route receipts.
- [ ] Future independent long-context/full42 cause closure and source fix;
  this diagnostic alone does not mark those complete.

````

## Source: .agents/runs/numerics-and-context-av-replay-20261020/README.md

Original UTF-8 bytes: 5193; SHA256 `3a22a43eb9c870c5a6b4adcf8e4e2a8e160f1cf4dd2b7b20f729d47f9a7ff98f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Prepared public AV replay — ROOT EXECUTES

- Parent:numerics-and-context. Owner:spine_review. Source-only preparation;
  no compilation, operator, model or GPU execution by this agent.
- Actual first64 capture is immutable. All64 full probability rows match,
  131,072 words; all1,064,960 causal V-word comparisons match; all114,432
  future probabilities are positive zero. First reported AV output mismatch:
  position2,530/2048 words. This is operator-boundary evidence, not an
  observed internal dispatch or a full long-context cause proof.
- Source candidate:ggml-cpu.c1308 tries llamafile only without use_ref;
  sgemm.cpp3911 refuses n<2. Haswell tinyBLAS627 uses one8-lane FMA
  accumulator/output; vec.cpp104–121 uses four8-lane accumulators with a
  different final reduction. No arithmetic from either is implemented here.

## Actual public calls

1. Authenticate all source/DSO/capture inputs through pins.json. Read-only
   byte-copy actual V/P/output fixtures into a new directory. Raw expected
   outputs come exclusively from the genuine capture.
2. Load exact installed libggml-cpu-haswell.so via ggml_backend_load; use its
   public registry/device owner and public thread/abort/ref-control functions.
   No model load or other backend registration. Record actual maps.
3. Recreate get_v's public4D view and attention's(0,2,1,3) permutation:
   F32 ne[256,256,2,1],nb[4,33792,8650752,17301504]. Allocate original8448
   capacity, byte-copy only actual logical rows. Uncaptured storage gaps are
   zeroed; they are not claimed as original captured bytes. Preserve8 query
   heads/2 KV heads and contiguous P/output ne[256,64or1,8,1].
4. Execute ggml_mul_mat. Require complete default bulk131,072 outputs and
   every64×2048 AR1 output to equal their own unchanged actual captures.
   Expected file is first opened by C++ after its public computation.
   Re-read all logical inputs; require unchanged bytes.
5. Only after all65 defaults pass, separately execute the public use_ref
   control for the same65 original operand pairs. Retain every control output
   and mismatch count. Compare ref-bulk against all actual/default AR1 rows.
   Control equality is diagnostic; no cause/profile eligibility is automatic.

## Root commands

Use the already accepted strict GCC15/native build environment and unchanged
installed0.6.0 headers/libraries. Do not build upstream or add fast-math.

```bash
TASK_AV_RUN=.agents/runs/numerics-and-context-av-replay-20261020
TASK_LLAMA_SRC=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LLAMA_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I "$TASK_LLAMA_SRC/ggml/include" "$TASK_AV_RUN/main.cpp" \
  -L "$TASK_LLAMA_LIB" -Wl,-rpath,"$TASK_LLAMA_LIB" \
  -lggml -lggml-base -ldl -o "$TASK_AV_RUN/replay"
python3 "$TASK_AV_RUN/prepare.py" "$TASK_AV_RUN/root-fixtures"
sha256sum "$TASK_AV_RUN/replay"
```

Pass that fresh binary hash explicitly to run_replay.py:

```bash
python3 "$TASK_AV_RUN/run_replay.py" \
  --binary "$TASK_AV_RUN/replay" --binary-sha256 ACTUAL_FRESH_SHA256 \
  --fixtures "$TASK_AV_RUN/root-fixtures" \
  --new-output "$TASK_AV_RUN/root-replay" \
  --new-log "$TASK_AV_RUN/root-replay.log" \
  --new-receipt "$TASK_AV_RUN/root-replay.invocation.json"
python3 "$TASK_AV_RUN/validate.py" \
  --fixtures "$TASK_AV_RUN/root-fixtures" --output "$TASK_AV_RUN/root-replay"
```

- Runner uses the already accepted glibc2.44 GNU loader,8 threads matching
  actual capture,300s wall,8GiB address-space limit, finite CPU/file/log caps,
  no NVIDIA exposure and only owned child-process-group cleanup. These are
  configured process ceilings, not ResourceArbiter/physical-RAM qualification.
  The leader is inspected without reaping before cleanup. Every spawned group
  receives bounded TERM/KILL cleanup even after leader exit; receipt requires
  leader reaped, group absent, stdout EOF and closed selector/pipe. Cleanup has
  separate3s TERM/5s leader-KILL/5s group/1s pipe deadlines; failures stay
  UNQUALIFIED and retain both primary and cleanup errors.
- Fixtures/output≤512MiB combined, native live backing≤32MiB/case, separate
  bounded8MiB log. Default and control payloads total2MiB. Python validates
  original descriptor/byte binding, full raw words and before/after pins.
  Repeated default AR1 reads require all8192bytes and the previously validated
  content hash; accepted completion explicitly requires8threads.
- CPU backend maps and public control origin are observed. Internal
  microkernel dispatch remains unobserved; no automatic claim that use_ref
  equality proves every long-context failure or a production fix.
- Upstream public headers/source retain MIT license; exact LICENSE copied as
  UPSTREAM_LICENSE. Harness itself Apache2.0; no upstream arithmetic copied.

## Remaining

- Root strict compile and source-only independent review.
- Root default full-output replay, then separately qualified control evidence.
- Any production reducer selection/fix requires source-derived shared-engine
  implementation, prior-context/full42/logit replay and regression gates.

````

## Source: .agents/runs/numerics-and-context-av-replay-20261020/TODO.md

Original UTF-8 bytes: 1020; SHA256 `665107ff5c0b27403e9fd306ead9e965f9bd2aeea76b495c726758bb697ca8d6`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Public AV diagnostic preparation

- Parent:numerics-and-context; owner:spine_review.
- Own:new ignored run only; preserve all old captures/helpers/frozens.
- [x] Read actual first64 complete metadata/report; independently byte-compare
  full probability operands, causal V and positive-zero future columns.
- [x] Source-audit original Haswell public dispatch and reduction candidates.
- [x] Prepare bounded original public ggml AV default/ref-control replay;
  no model load or arithmetic replacement.
- [x] Independent source review; repair owned-group cleanup after leader exit,
  re-read output coverage/hash and explicit8thread validation. Original frozen
  preparation retained; follow-up unexecuted.
- [ ] Root compile/run/validator and independent source review.
- [ ] Full long-context causal closure/production implementation remains open.

No Cargo/build/operator/model/GPU execution, production source, shared plan,
root README, SDK/model acquisition, device privilege, commit or publication
by this agent.

````

## Source: .agents/runs/numerics-and-context-av-replay-20261020/frozen-cleanup-followup/README.md

Original UTF-8 bytes: 5193; SHA256 `3a22a43eb9c870c5a6b4adcf8e4e2a8e160f1cf4dd2b7b20f729d47f9a7ff98f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Prepared public AV replay — ROOT EXECUTES

- Parent:numerics-and-context. Owner:spine_review. Source-only preparation;
  no compilation, operator, model or GPU execution by this agent.
- Actual first64 capture is immutable. All64 full probability rows match,
  131,072 words; all1,064,960 causal V-word comparisons match; all114,432
  future probabilities are positive zero. First reported AV output mismatch:
  position2,530/2048 words. This is operator-boundary evidence, not an
  observed internal dispatch or a full long-context cause proof.
- Source candidate:ggml-cpu.c1308 tries llamafile only without use_ref;
  sgemm.cpp3911 refuses n<2. Haswell tinyBLAS627 uses one8-lane FMA
  accumulator/output; vec.cpp104–121 uses four8-lane accumulators with a
  different final reduction. No arithmetic from either is implemented here.

## Actual public calls

1. Authenticate all source/DSO/capture inputs through pins.json. Read-only
   byte-copy actual V/P/output fixtures into a new directory. Raw expected
   outputs come exclusively from the genuine capture.
2. Load exact installed libggml-cpu-haswell.so via ggml_backend_load; use its
   public registry/device owner and public thread/abort/ref-control functions.
   No model load or other backend registration. Record actual maps.
3. Recreate get_v's public4D view and attention's(0,2,1,3) permutation:
   F32 ne[256,256,2,1],nb[4,33792,8650752,17301504]. Allocate original8448
   capacity, byte-copy only actual logical rows. Uncaptured storage gaps are
   zeroed; they are not claimed as original captured bytes. Preserve8 query
   heads/2 KV heads and contiguous P/output ne[256,64or1,8,1].
4. Execute ggml_mul_mat. Require complete default bulk131,072 outputs and
   every64×2048 AR1 output to equal their own unchanged actual captures.
   Expected file is first opened by C++ after its public computation.
   Re-read all logical inputs; require unchanged bytes.
5. Only after all65 defaults pass, separately execute the public use_ref
   control for the same65 original operand pairs. Retain every control output
   and mismatch count. Compare ref-bulk against all actual/default AR1 rows.
   Control equality is diagnostic; no cause/profile eligibility is automatic.

## Root commands

Use the already accepted strict GCC15/native build environment and unchanged
installed0.6.0 headers/libraries. Do not build upstream or add fast-math.

```bash
TASK_AV_RUN=.agents/runs/numerics-and-context-av-replay-20261020
TASK_LLAMA_SRC=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LLAMA_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I "$TASK_LLAMA_SRC/ggml/include" "$TASK_AV_RUN/main.cpp" \
  -L "$TASK_LLAMA_LIB" -Wl,-rpath,"$TASK_LLAMA_LIB" \
  -lggml -lggml-base -ldl -o "$TASK_AV_RUN/replay"
python3 "$TASK_AV_RUN/prepare.py" "$TASK_AV_RUN/root-fixtures"
sha256sum "$TASK_AV_RUN/replay"
```

Pass that fresh binary hash explicitly to run_replay.py:

```bash
python3 "$TASK_AV_RUN/run_replay.py" \
  --binary "$TASK_AV_RUN/replay" --binary-sha256 ACTUAL_FRESH_SHA256 \
  --fixtures "$TASK_AV_RUN/root-fixtures" \
  --new-output "$TASK_AV_RUN/root-replay" \
  --new-log "$TASK_AV_RUN/root-replay.log" \
  --new-receipt "$TASK_AV_RUN/root-replay.invocation.json"
python3 "$TASK_AV_RUN/validate.py" \
  --fixtures "$TASK_AV_RUN/root-fixtures" --output "$TASK_AV_RUN/root-replay"
```

- Runner uses the already accepted glibc2.44 GNU loader,8 threads matching
  actual capture,300s wall,8GiB address-space limit, finite CPU/file/log caps,
  no NVIDIA exposure and only owned child-process-group cleanup. These are
  configured process ceilings, not ResourceArbiter/physical-RAM qualification.
  The leader is inspected without reaping before cleanup. Every spawned group
  receives bounded TERM/KILL cleanup even after leader exit; receipt requires
  leader reaped, group absent, stdout EOF and closed selector/pipe. Cleanup has
  separate3s TERM/5s leader-KILL/5s group/1s pipe deadlines; failures stay
  UNQUALIFIED and retain both primary and cleanup errors.
- Fixtures/output≤512MiB combined, native live backing≤32MiB/case, separate
  bounded8MiB log. Default and control payloads total2MiB. Python validates
  original descriptor/byte binding, full raw words and before/after pins.
  Repeated default AR1 reads require all8192bytes and the previously validated
  content hash; accepted completion explicitly requires8threads.
- CPU backend maps and public control origin are observed. Internal
  microkernel dispatch remains unobserved; no automatic claim that use_ref
  equality proves every long-context failure or a production fix.
- Upstream public headers/source retain MIT license; exact LICENSE copied as
  UPSTREAM_LICENSE. Harness itself Apache2.0; no upstream arithmetic copied.

## Remaining

- Root strict compile and source-only independent review.
- Root default full-output replay, then separately qualified control evidence.
- Any production reducer selection/fix requires source-derived shared-engine
  implementation, prior-context/full42/logit replay and regression gates.

````

## Source: .agents/runs/numerics-and-context-av-replay-20261020/frozen-cleanup-followup/TODO.md

Original UTF-8 bytes: 1020; SHA256 `665107ff5c0b27403e9fd306ead9e965f9bd2aeea76b495c726758bb697ca8d6`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Public AV diagnostic preparation

- Parent:numerics-and-context; owner:spine_review.
- Own:new ignored run only; preserve all old captures/helpers/frozens.
- [x] Read actual first64 complete metadata/report; independently byte-compare
  full probability operands, causal V and positive-zero future columns.
- [x] Source-audit original Haswell public dispatch and reduction candidates.
- [x] Prepare bounded original public ggml AV default/ref-control replay;
  no model load or arithmetic replacement.
- [x] Independent source review; repair owned-group cleanup after leader exit,
  re-read output coverage/hash and explicit8thread validation. Original frozen
  preparation retained; follow-up unexecuted.
- [ ] Root compile/run/validator and independent source review.
- [ ] Full long-context causal closure/production implementation remains open.

No Cargo/build/operator/model/GPU execution, production source, shared plan,
root README, SDK/model acquisition, device privilege, commit or publication
by this agent.

````

## Source: .agents/runs/numerics-and-context-av-replay-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 4680; SHA256 `6a33484f263d9a3b6a6cbbd2a404841d3bcb8e37dc8c340827276b3fc85b1b27`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Prepared public AV replay — ROOT EXECUTES

- Parent:numerics-and-context. Owner:spine_review. Source-only preparation;
  no compilation, operator, model or GPU execution by this agent.
- Actual first64 capture is immutable. All64 full probability rows match,
  131,072 words; all1,064,960 causal V-word comparisons match; all114,432
  future probabilities are positive zero. First reported AV output mismatch:
  position2,530/2048 words. This is operator-boundary evidence, not an
  observed internal dispatch or a full long-context cause proof.
- Source candidate:ggml-cpu.c1308 tries llamafile only without use_ref;
  sgemm.cpp3911 refuses n<2. Haswell tinyBLAS627 uses one8-lane FMA
  accumulator/output; vec.cpp104–121 uses four8-lane accumulators with a
  different final reduction. No arithmetic from either is implemented here.

## Actual public calls

1. Authenticate all source/DSO/capture inputs through pins.json. Read-only
   byte-copy actual V/P/output fixtures into a new directory. Raw expected
   outputs come exclusively from the genuine capture.
2. Load exact installed libggml-cpu-haswell.so via ggml_backend_load; use its
   public registry/device owner and public thread/abort/ref-control functions.
   No model load or other backend registration. Record actual maps.
3. Recreate get_v's public4D view and attention's(0,2,1,3) permutation:
   F32 ne[256,256,2,1],nb[4,33792,8650752,17301504]. Allocate original8448
   capacity, byte-copy only actual logical rows. Uncaptured storage gaps are
   zeroed; they are not claimed as original captured bytes. Preserve8 query
   heads/2 KV heads and contiguous P/output ne[256,64or1,8,1].
4. Execute ggml_mul_mat. Require complete default bulk131,072 outputs and
   every64×2048 AR1 output to equal their own unchanged actual captures.
   Expected file is first opened by C++ after its public computation.
   Re-read all logical inputs; require unchanged bytes.
5. Only after all65 defaults pass, separately execute the public use_ref
   control for the same65 original operand pairs. Retain every control output
   and mismatch count. Compare ref-bulk against all actual/default AR1 rows.
   Control equality is diagnostic; no cause/profile eligibility is automatic.

## Root commands

Use the already accepted strict GCC15/native build environment and unchanged
installed0.6.0 headers/libraries. Do not build upstream or add fast-math.

```bash
TASK_AV_RUN=.agents/runs/numerics-and-context-av-replay-20261020
TASK_LLAMA_SRC=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
TASK_LLAMA_LIB=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I "$TASK_LLAMA_SRC/ggml/include" "$TASK_AV_RUN/main.cpp" \
  -L "$TASK_LLAMA_LIB" -Wl,-rpath,"$TASK_LLAMA_LIB" \
  -lggml -lggml-base -ldl -o "$TASK_AV_RUN/replay"
python3 "$TASK_AV_RUN/prepare.py" "$TASK_AV_RUN/root-fixtures"
sha256sum "$TASK_AV_RUN/replay"
```

Pass that fresh binary hash explicitly to run_replay.py:

```bash
python3 "$TASK_AV_RUN/run_replay.py" \
  --binary "$TASK_AV_RUN/replay" --binary-sha256 ACTUAL_FRESH_SHA256 \
  --fixtures "$TASK_AV_RUN/root-fixtures" \
  --new-output "$TASK_AV_RUN/root-replay" \
  --new-log "$TASK_AV_RUN/root-replay.log" \
  --new-receipt "$TASK_AV_RUN/root-replay.invocation.json"
python3 "$TASK_AV_RUN/validate.py" \
  --fixtures "$TASK_AV_RUN/root-fixtures" --output "$TASK_AV_RUN/root-replay"
```

- Runner uses the already accepted glibc2.44 GNU loader,8 threads matching
  actual capture,300s wall,8GiB address-space limit, finite CPU/file/log caps,
  no NVIDIA exposure and only owned child-process-group cleanup. These are
  configured process ceilings, not ResourceArbiter/physical-RAM qualification.
- Fixtures/output≤512MiB combined, native live backing≤32MiB/case, separate
  bounded8MiB log. Default and control payloads total2MiB. Python validates
  original descriptor/byte binding, full raw words and before/after pins.
- CPU backend maps and public control origin are observed. Internal
  microkernel dispatch remains unobserved; no automatic claim that use_ref
  equality proves every long-context failure or a production fix.
- Upstream public headers/source retain MIT license; exact LICENSE copied as
  UPSTREAM_LICENSE. Harness itself Apache2.0; no upstream arithmetic copied.

## Remaining

- Root strict compile and source-only independent review.
- Root default full-output replay, then separately qualified control evidence.
- Any production reducer selection/fix requires source-derived shared-engine
  implementation, prior-context/full42/logit replay and regression gates.

````

## Source: .agents/runs/numerics-and-context-av-replay-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 816; SHA256 `c0ad1b473ac43350afa7624f9324a71ea0918277e1ee3de4696784ffc5d1d6f0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Public AV diagnostic preparation

- Parent:numerics-and-context; owner:spine_review.
- Own:new ignored run only; preserve all old captures/helpers/frozens.
- [x] Read actual first64 complete metadata/report; independently byte-compare
  full probability operands, causal V and positive-zero future columns.
- [x] Source-audit original Haswell public dispatch and reduction candidates.
- [x] Prepare bounded original public ggml AV default/ref-control replay;
  no model load or arithmetic replacement.
- [ ] Root compile/run/validator and independent source review.
- [ ] Full long-context causal closure/production implementation remains open.

No Cargo/build/operator/model/GPU execution, production source, shared plan,
root README, SDK/model acquisition, device privilege, commit or publication
by this agent.

````

## Source: .agents/runs/numerics-and-context-bulk-causality-20261020/AUDIT.md

Original UTF-8 bytes: 9196; SHA256 `87a2ccef1168264e9fd95ecb35a45563210f0b49f49e8c98bcbff26b60fa80f8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Fresh0.8B1024 bulk64 causality audit

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/spine_review`. Source-only preparation; root alone builds/runs.
No production/frozen-helper/capture modifications. No Cargo/C++/native/model/
operator/runtime/GPU execution by author.

First64 root observation refused the old16MiB backing-span cap before completion;
see `SPAN_FIX.md` for preserved failure and exact V-cache logical-run follow-up.

## Current actual evidence

- Parent fresh AR1 controls512/1024 both originals compare730 records exactly.
- Fresh0.8B1024 prefix_batch64/ubatch64 has543 differing stages/9,230,360 words.
  First *observed* selected-token gap: layer4 incoming recurrent state171,058
  words, maxabs0.00047385692596435547. All current layer4 projection/conv/
  coefficient/q/k/v inputs and conv prior/proposal exact; current core/state/
  later layers then differ. Layers0–2 whole incoming/proposed recurrence exact;
  layer3 selected-token broad results exact. This does not identify an earlier
  causal token/operator or prove all preceding row outputs exact.
- Capture helper used original1023-prefix segmentation15×64+63, then actual
  selected token1023 as AR1. Its callback was disabled for the prefix. Last-token
  records do not observe the shape64 execution that produced the inherited gap.

## Source-derived candidates, not runtime attribution

- `llama-context.cpp:235–237` sets fused_gdn_ar/ch=true, auto_fgdn=false.
  `models/delta-net-base.cpp:373–446` dispatches both widths to the same
  GGML_OP_GATED_DELTA_NET, K=1 final snapshot. The CH label does **not** establish
  the separate chunked-FLA graph. Native CPU `ops.cpp:11107–11275` copies prior
  state then loops actual token time, scalar expf(g), vector decay, dot, delta,
  FMA update and output dot. Head/thread partitioning is outside that time loop.
- `ggml-cpu.c:1269` tries tiled quant kernels first. `tiled/tiled.cpp:605–632`
  excludes F32 src0, so this cannot supply F32 attention QK/AV.
- `ggml-cpu.c:1309–1327` tries llamafile for contiguous src1. F32 src1/KV requires
  no conversion. `llamafile/sgemm.cpp:3909–3934` rejects n<2, while F32 AVX2 n≥2
  may use tinyBLAS if its geometry gates pass. QK query views may be strided;
  do not assume contiguity. AV softmax output is contiguous. Capture actual
  ne/nb and operands instead of claiming a backend from requested ubatch.
- tinyBLAS (`sgemm.cpp:559–670`) uses one eight-lane F32 accumulator per output
  for each k+=8, then hsum. Native fallback `vec.cpp:107–135` uses four eight-lane
  accumulators, k+=32, pairwise accumulator reduction and lane reduction. These
  are distinct source rounding schedules, not proof which one produced a bit.
- Attention graph `llama-graph.cpp:2755–2808` is F32 QK→masked scaled softmax→AV.
  `ops.cpp:5813–5910` softmax runs each query row independently; `vec.cpp:531+`
  exp lanes/reduction depend on padded KV width. Native KV get_n_kv (`kv-cache
  :1284`) pads to256. Batch rows also see look-ahead cache values hidden by the
  causal mask; comparing those unmasked future QK columns as causal results
  would be wrong. Retain all raw bits, classify future columns explicitly.
- Sigmoid/SiLU/softplus and quant projection paths can also depend on actual
  spans/layout/provider. Their row inputs/results are captured. No HF helper,
  new tolerance, profile switch or fallback is proposed from this audit.

## Minimal actual continuation

1. First try **original first64-token block from zero**. It preserves the actual
   width64 arithmetic and has a common exact initial state. No prefix search is
   needed if its actual row evidence already isolates an upstream operation.
2. If first block remains exact, inspect subsequent aligned blocks0,64,…,960
   preserving64 rows, with final original block63 rows. Existing helper cut65,
   129,… can expose the original corresponding prefix boundary in its *prior*
   state; use aligned boundaries only. Do not call arbitrary-cut binary search
   causal bisection: it changes the batch shape, and bit divergence need not be
   monotone. Sequential first-boundary scan avoids that assumption.
3. At chosen aligned block, new probe computes independent native bulk/AR1
   prefixes using original segmentation. It **requires their entire serialized
   incoming state equal**. A differing inherited prior refuses and directs root
   earlier. Bulk plain64 (or63) and observed64 restore identical actual prior;
   AR1 plain/observed independently replay the same actual teacher rows.
4. Both observation paths must match their own plain path's complete post-state
   exactly. Then both plain/observed paths consume the same next actual teacher
   token as AR1; require full248320 logits and whole serialized state exact.
   Those continuation guards do not replace the captured original block.
5. Capture actual layers0–4 full rows, all projection operands/results, attention3
   Q/K/V/projection/norm/RoPE/cache operands/QK/mask/softmax/AV/gate/output, and
   GDN0/1/2/4 actual initial state, q/k/v/g/b, core rows and final proposed state.
   No exp(g) reconstruction or coefficient expected generation. Original K=1
   snapshot policy unchanged: intermediate native recurrent states unobserved.
6. Compare matching actual row slices. Preserve full raw padded caches; compare
   whole-cache operands at final AR1 frontier. Report future unmasked QK raw
   differences separately from causal positions<=query. This is exact bit
   coverage selection by actual causal mask, not tolerance/zero normalization.
7. Causal attribution still requires matching complete actual inputs and prior
   to the *specific* first differing result. Then root may replay that operator
   through existing shared engine/public native path, observe dispatch if
   necessary and derive a source-correct implementation. No production change
   or long-context/model eligibility follows automatically from this probe.

## New files and bounds

- `native_window.cpp` reuses byte-pinned old public-client helpers by textual
  include with renamed unused main; old bytes remain untouched. Actual source
  ne/nb and exported dense-view shape/slice offset are recorded separately.
- `compare_window.py` compares native bulk/AR1 actual observations only. It
  validates complete metadata/file shapes, repeated reads and byte hashes;
  records first *observed* field difference, never auto-labels a cause or Rust
  parity. Exit3=malformed/failed guard; exit0=diagnostic emitted, not parity.
- Original0.8B geometry only; teacher stream exactly8192 u32LE IDs, vocabulary
  bound248320; selected aligned start0..960; original case cut1024. CPU threads
  1..8; cooperative≤7200s deadline; original n_ctx8320/F32 KV/FAoff/CPU settings.
- ≤128MiB tensor payload+16MiB metadata/provider, field≤2M F32 words, records
  ≤16384/provider; no full graph dump/learned weight copies. Three raw snapshots
  at≤1023 causal tokens add<136MiB; two full-next-logit files<2MiB. Combined
  artifacts<512MiB excluding bounded external process logs. Root applies finite
  actual process RAM/CPU/wall/log caps. Four contexts share one original model;
  these bounds are not ResourceArbiter/physical RAM qualification.
- Completion file is written only after all transparency guards. Failure keeps
  an incomplete exclusive directory and root's failure log. No existing output
  is overwritten; no batch width/backend/operator/privilege is changed.

## Root-only preparation/execution

Root must verify `prepared-manifest.json`/`source-pins.json`, original model SHA
57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf and common raw-ID
SHA44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d independently.
Use same compiler/include/link/glibc closure as existing native_raw; freeze/hash
the fresh executable and retain actual argv/library/source receipt. No downloads.

```sh
# ROOT ONLY; existing root supervisor enforces finite process/resource/log caps.
probe=.agents/runs/numerics-and-context-bulk-causality-20261020
src=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
lib=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
glibc=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I "$src/include" -I "$src/ggml/include" \
  "$probe/native_window.cpp" -L "$lib" -L "$glibc" \
  -Wl,-rpath,"$lib:$glibc" -Wl,--dynamic-linker,"$glibc/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -o "$probe/native_window"
# ROOT review first, then run one original first64-row block only.
"$probe/native_window" target/models/qwen35-0.8b/model.gguf \
  .agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le \
  0 8 900 "$probe/root-first64"
python3 "$probe/compare_window.py" --capture "$probe/root-first64" \
  --output "$probe/root-first64-comparison.json"
```

Do not automatically launch the complete block ladder or call these timings a
serving benchmark. Instrumentation controls establish numerical transparency,
not performance transparency or internal microkernel observation.

````

## Source: .agents/runs/numerics-and-context-bulk-causality-20261020/SPAN_FIX.md

Original UTF-8 bytes: 3046; SHA256 `407900e18043551c033aeb8fb6afd601acb6535b445c2222394f115e4959f2dc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# First64 observation span refusal: narrow follow-up

Initial frozen manifest564ab0e0/helper639fc525 and root-first64 output remain
unchanged. Root compiled that helper strictly; four source-free comparator
fixtures passed. Actual run exited3 before transparency completion, therefore
establishes no accepted native window/model inference result.

- Last completed field152: actual `kqv-3`, ne[256,64,8,1]. Next field reads its
  V-cache input. Previously inherited `dense()` refused any backing span>16MiB.
- Pinned `llama-context.cpp:292` pads requested8320 context to256→8448. Actual
  common contexts and earlier captured K-cache nb3=17301504 independently agree.
- `llama-kv-cache.cpp:1329/:1514`: transposed V storage is
  `[capacity,head*dimension]`; view+permute gives ne[visibleKV,256,2,1] with
  nb[4,8448*4,8448*256*4,8448*512*4]. At visibleKV256, exact minimum backing
  span is `((512-1)*8448+256)*4=17268736B`, while logical data is524288B.
  `ggml.c:1298` computes exactly4+sum((ne-1)*nb). No larger tensor was requested.

`window_dense()` is local to this new diagnostic only:

- Existing native_raw.cpp/dense() bytes and all frozen/captured attempts unchanged.
- F32 innerstride4; positive checked dimensions/product≤2M words; checked stride
  products/span additions; computed span must equal actual public ggml_nbytes.
- Ordinary spans retain≤16MiB policy. The only larger accepted span is exactly
  the named attention3 transposed V-cache operand with actual context8448,
  visibleKV256/512/768/1024, heads2×dim256 and exact above strides/formula.
  Its range remains inside the actual full cache allocation17301504B.
- Allocate only logical≤8MiB result. Public `ggml_backend_tensor_get` copies each
  inner contiguous run into old dense order; read offsets/counts checked against
  the exact tensor span, no capacity-gap read/copy and no scalar math operation.
- Refusals now print bounded call/site/name/op/ne/nb/context/logical/span details.
  Original exact full-state/next-token guards, limits and source math unchanged.

Independent review also found mandatory-field coverage should be explicit.
New C++ checks each actual call; comparator independently requires every call's
embedding, first5-layer broad boundaries, exact original projection role
inputs/outputs, attention3 QK/mask/softmax/AV/cache inputs and all GDN0/1/2/4
q/k/v/g/b/core plus initial/final states at their actual boundaries. No omitted
attention field may silently reduce scope. Runtime callback labels still must
match actual source; no guessed expected value or numerical exception.

Conv source history/kernel inputs remain unobserved. A convolution-result gap
would still need a narrow actual operand capture before causal attribution.
The comparator stays diagnostic-only and never auto-promotes a cause/profile.

Author performed source/hash/AST checks only. Root independently reviews,
compiles a fresh binary and runs a **new** exclusive attempt directory; no
native/model/operator/runtime/Cargo/C++compile/GPU execution by author.

````

## Source: .agents/runs/numerics-and-context-bulk-causality-20261020/TODO.md

Original UTF-8 bytes: 1298; SHA256 `bd53e75b091c2ef655fa42d4a1e0d1dceb989c6d0dba4b02d71b3f4e313fc59e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/spine_review`; root owns execution/shared-plan reconciliation.

- [x] Audit current1024 bulk64 report and pinned graph/CPU dispatch source.
- [x] Separate inherited layer4 prior-state difference from causal attribution.
- [x] Preserve original15×64+63; document why arbitrary-cut binary search changes
  source arithmetic and cannot identify the original first divergent row.
- [x] Prepare source-only unchanged-block/native-AR1 observer + bit comparator;
  all input/result rows and independent state/next-logit transparency guards.
- [x] Preserve root first64 span refusal; prepare local logical-run V-cache
  copier with exact context/head/stride/span guards and refusal context.
- [x] Require projection/attention/GDN semantic families per actual call in
  both emitter and comparator; no silently partial observation scope.
- [ ] Root/independent review, compile and actual first64-row bounded control.
- [ ] Actual first causal row/operator with complete equal inputs/prior.
- [ ] Source-correct existing-engine fix; actual operator/full-state/long gates.

No native/model/operator/runtime/Cargo/C++compile/GPU execution by author.
Original frozen helpers/captures and all production source unchanged.

````

