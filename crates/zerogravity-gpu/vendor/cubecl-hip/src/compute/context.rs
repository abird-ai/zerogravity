//! The device context: its compiled kernels, and its open profiles.
//!
//! Compilation is memoized here rather than per stream, because a module is
//! loaded into the context and every stream sharing it can launch from the
//! same one. What a compiled kernel answers for beyond its entry point is
//! which of its bindings it writes, which is what a launch stages its write
//! scope from.

use super::storage::gpu::GpuResource;
use crate::compiler::{HipBackend, HipCompilationOptions, HipCompiler, HipRepresentation};
use crate::compute::events::EventProfiler;
use crate::compute::status::checked;
use crate::compute::stream::Stream;
use cubecl_core::hash::StableHasher;
use cubecl_core::{hash::StableHash, ir::DeviceProperties, prelude::*, server::ResourceLimitError};
use cubecl_cpp::formatter::format_cpp;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::persistence::Store;
use cubecl_hip_sys::get_hip_include_path;
use cubecl_server::compiler::{
    CompilationCache, CompilationRecording, build_id_hash, compilation_store, store_compiled,
};
use cubecl_server::kernel::BufferIOAttr;
use cubecl_server::kernel::DebugInformation;
use cubecl_server::{
    compiler::CompilationError,
    validation::{validate_cube_dim, validate_units},
};
use cubecl_server::{
    compiler::KernelCacheKey,
    kernel::{CompiledKernel, CubeKernel},
    logging::ServerLogger,
};
use serde::Deserialize;
use serde::Serialize;
use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct HipContext {
    /// The modules loaded on the device, in front of
    /// [`Self::compilation_cache`].
    ///
    /// An environment switch drops these, and nothing unloads the modules they
    /// name: see [`HipContext::is_loaded`].
    modules: CompilationCache<KernelId, HipCompiledKernel>,
    pub(crate) native: crate::prepared_native::NativeSets,
    pub(crate) native_context: Option<crate::prepared_native::PersistentNativeOwner>,
    pub(crate) legacy_loaded: usize,
    pub profiler: EventProfiler,
    pub compilation_options: HipCompilationOptions,
    pub properties: DeviceProperties,
    pub compilation_cache: Option<Store<KernelCacheKey, CompilationCacheEntry>>,
    /// Cache mapping C++ code hashes to the key that first compiled them. We can skip the slow HIP
    /// compiler if we already have a compiled artifact for the same code.
    ///
    /// C++ only: the LLVM backend emits a code object directly, so there is no
    /// intermediate source to key a second line on.
    pub second_line_compilation_cache: Option<Store<StableHash, KernelCacheKey>>,
    build_id: StableHash,
}

#[derive(Debug)]
pub struct HipCompiledKernel {
    pub(crate) module: cubecl_hip_sys::hipModule_t,
    _native: Option<crate::prepared_native::LoadedNativeOwner>,
    func: cubecl_hip_sys::hipFunction_t,
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    /// What the kernel does with each buffer binding, by buffer position --
    /// the compiler's answer, carried here because on a cache hit nothing
    /// else of the compilation survives. `None` for entries persisted before
    /// the answer existed, which the launch path reads as every buffer both
    /// read and written.
    io: Option<Arc<[BufferIOAttr]>>,
}

impl HipCompiledKernel {
    pub(crate) fn mark_native_unloaded(&mut self) {
        if let Some(owner) = self._native.as_mut() {
            owner.mark_unloaded();
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct CompilationCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    binary: Vec<i8>,
    /// See [`HipCompiledKernel::io`]; defaulted for entries persisted before
    /// the field existed.
    #[serde(default)]
    io: Option<Vec<BufferIOAttr>>,
}

/// The namespace a backend's compiled artifacts live under.
///
/// Both backends emit AMD code objects, so without the backend in the key a stale
/// artifact from one would load and run happily under the other — tests passing
/// while measuring nothing.
fn cache_namespace(fingerprint: &str, backend: HipBackend) -> String {
    let backend = match backend {
        HipBackend::Cpp => "cpp",
        #[cfg(feature = "llvm")]
        HipBackend::Llvm => "llvm",
    };
    format!("{fingerprint}-{backend}")
}

impl HipContext {
    /// `fingerprint` is the one the runtime already published on
    /// [`DeviceProperties::identity`], rather than one rebuilt here: the
    /// namespace a kernel is cached under and the identity a bundle is stamped
    /// with have to be the same string, and the only way to guarantee that is
    /// for there to be one string.
    ///
    /// `backend` is which one compiles here; see [`cache_namespace`].
    pub fn new(
        compilation_options: HipCompilationOptions,
        properties: DeviceProperties,
        fingerprint: String,
        backend: HipBackend,
    ) -> Self {
        let fingerprint = cache_namespace(&fingerprint, backend);
        let compilation_cache = compilation_store("hip", &fingerprint);
        let second_line_compilation_cache = compilation_store("hip-second-line", fingerprint);

        Self {
            modules: CompilationCache::mirroring(&compilation_cache),
            native: crate::prepared_native::NativeSets::default(),
            native_context: None,
            legacy_loaded: 0,
            profiler: EventProfiler::default(),
            compilation_options,
            compilation_cache,
            second_line_compilation_cache,
            properties,
            build_id: build_id_hash(),
        }
    }

    /// Whether `kernel_id` is already loaded on the device.
    ///
    /// An environment switch drops the whole cache, so the new environment's
    /// store is filled rather than bypassed. The modules those entries named
    /// stay resident: nothing calls `hipModuleUnload`, here or anywhere else in
    /// this context, and unloading one a stream still has queued work against
    /// would be unsound. A process that switches environments a handful of
    /// times at startup pays a bounded price; one that switches repeatedly
    /// grows its resident modules without bound — see
    /// [`cubecl_environment::environment::activate`].
    pub fn is_loaded(&mut self, kernel_id: &KernelId) -> bool {
        if self.native.active() {
            return self
                .native
                .prepared(kernel_id)
                .is_some_and(|p| !p.failed && !p.retired && p.kernels.contains_key(kernel_id));
        }
        self.modules.contains(kernel_id)
    }

    fn loaded(&mut self, kernel_id: &KernelId) -> Option<&HipCompiledKernel> {
        if self.native.active() {
            let p = self.native.prepared(kernel_id)?;
            return (p.sealed && !p.failed && !p.retired && p.admission.require_usable().is_ok())
                .then(|| p.kernels.get(kernel_id))
                .flatten();
        }
        self.modules.get(kernel_id)
    }

    fn try_load_cached(
        &mut self,
        kernel_id: &KernelId,
    ) -> Result<Result<(), Option<KernelCacheKey>>, CompilationError> {
        let key = if let Some(cache) = self.compilation_cache.as_mut() {
            let key = KernelCacheKey::new(kernel_id, self.build_id);

            if let Some(entry) = cache.remove(&key) {
                log::trace!("Using compilation cache");

                self.load_compiled_binary(
                    &entry.binary,
                    kernel_id.clone(),
                    entry.entrypoint_name,
                    kernel_id.cube_dim.into(),
                    entry.shared_mem_bytes,
                    entry.io.map(Arc::from),
                )?;
                return Ok(Ok(()));
            }
            Some(key)
        } else {
            None
        };
        Ok(Err(key))
    }

    /// Compiles a kernel.
    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        cube_kernel: Box<dyn CubeKernel>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        if self.native.active()
            && self.native.prepared(kernel_id).is_none_or(|p| {
                p.sealed || p.failed || p.retired || p.admission.require_usable().is_err()
            })
        {
            return Err(CompilationError::Generic {
                reason: "HIP prepared native set is sealed/failed/retired; no fallback compilation"
                    .into(),
                backtrace: BackTrace::capture(),
            }
            .into());
        }
        let mut recording = CompilationRecording::new(kernel_id);
        let key = match self.try_load_cached(kernel_id)? {
            Ok(()) => {
                recording.loaded();
                return Ok(());
            }
            Err(key) => key,
        };

        validate_cube_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        // CubeCL compilation
        // jitc = just-in-time compiled
        let definition = cubecl_core::define_kernel(&*cube_kernel)?;
        recording.defined(&definition);
        let jitc_kernel = CompiledKernel::compile(
            &*cube_kernel,
            definition,
            &mut HipCompiler::default(),
            &self.compilation_options,
        )?;

        self.validate_shared(&jitc_kernel.repr)?;
        recording.source(&jitc_kernel.source);

        self.load_jit_kernel(kernel_id, key, jitc_kernel, logger, recording)
    }

    /// Loads what the compiler produced, by the route that backend's output takes, and closes
    /// `recording` with how the artifact was obtained.
    fn load_jit_kernel(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        jitc_kernel: CompiledKernel<HipCompiler>,
        logger: Arc<ServerLogger>,
        recording: CompilationRecording,
    ) -> Result<(), LaunchError> {
        match &jitc_kernel.repr {
            Some(HipRepresentation::Cpp(_)) => {
                self.load_transpiled(kernel_id, key, jitc_kernel, logger, recording)
            }
            #[cfg(feature = "llvm")]
            Some(HipRepresentation::Llvm(_)) => {
                self.load_code_object(kernel_id, key, jitc_kernel, logger, recording)
            }
            // A precompiled kernel: its text passed the language check in
            // `CompiledKernel::compile`, so it is whatever the default backend
            // reads. HIP C++ goes through HIP RTC like a transpiled kernel;
            // the LLVM backend produces linked code objects and has no route
            // for text.
            None => match HipBackend::default() {
                HipBackend::Cpp => self.load_transpiled(kernel_id, key, jitc_kernel, logger, recording),
                #[cfg(feature = "llvm")]
                HipBackend::Llvm => Err(CompilationError::Generic {
                    reason: "the LLVM backend cannot load a precompiled kernel: it has no text to compile from"
                        .to_string(),
                    backtrace: BackTrace::capture(),
                }
                .into()),
            },
        }
    }

    /// Turns a kernel the LLVM backend just compiled into a loaded module.
    ///
    /// What it hands back is already a linked `ET_DYN` code object, so no `hiprtc*`
    /// call belongs here: the bytes go straight to [`Self::load_compiled_binary`],
    /// exactly as a cache hit would.
    #[cfg(feature = "llvm")]
    fn load_code_object(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        mut jitc_kernel: CompiledKernel<HipCompiler>,
        logger: Arc<ServerLogger>,
        recording: CompilationRecording,
    ) -> Result<(), LaunchError> {
        let Some(HipRepresentation::Llvm(module)) = &jitc_kernel.repr else {
            unreachable!("dispatched on the representation");
        };

        if logger.compilation_source_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("ll", kernel_id.clone()));
        }
        logger.log_compilation(&jitc_kernel);

        let code = to_signed(&module.code_object);
        let shared_mem_bytes = module.shared_memory_size;
        let io = jitc_kernel.io.take();
        let entrypoint_name = jitc_kernel.entrypoint_name.clone();

        self.load_compiled_binary(
            &code,
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            jitc_kernel.cube_dim,
            shared_mem_bytes,
            io.clone().map(Arc::from),
        )?;

        // Cached after the load, so a code object the driver rejects is not handed back on
        // the next run, and the bytes move rather than being copied a third time.
        // `try_load_cached` hands back a key exactly when there is a cache to put it in.
        let stored = match self.compilation_cache.as_mut().zip(key) {
            Some((cache, key)) => store_compiled(
                cache,
                key,
                CompilationCacheEntry {
                    entrypoint_name,
                    shared_mem_bytes,
                    binary: code,
                    io,
                },
            ),
            None => false,
        };
        recording.compiled(stored);
        Ok(())
    }

    /// Turns a kernel the C++ backend just transpiled into a loaded module, by
    /// running its source through HIP RTC first.
    fn load_transpiled(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        mut jitc_kernel: CompiledKernel<HipCompiler>,
        logger: Arc<ServerLogger>,
        recording: CompilationRecording,
    ) -> Result<(), LaunchError> {
        if logger.compilation_source_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&jitc_kernel.source) {
                jitc_kernel.source = formatted;
            }
        }
        logger.log_compilation(&jitc_kernel);

        // `try_load_cached` hands back a key exactly when there is a cache to put it in, and
        // both stores are opened together in `HipContext::new`.
        let cpp_hash = if let Some(key) = key
            && let Some(cache) = self.compilation_cache.as_mut()
        {
            let second_line_cache = self.second_line_compilation_cache.as_mut().unwrap();
            let cpp_hash = StableHasher::hash_one(&jitc_kernel.source);

            if let Some(old_key) = second_line_cache.purge_key(&cpp_hash)
                && let Some(entry) = cache.purge_key(&old_key)
            {
                log::trace!("Using second-line compilation cache");
                let stored = store_compiled(cache, key, entry);
                store_compiled(second_line_cache, cpp_hash, key);
                self.try_load_cached(kernel_id)?
                    .expect("Should be cached now");
                recording.rekeyed(stored);
                return Ok(());
            }
            Some(cpp_hash)
        } else {
            None
        };

        let compiler_owner = self
            .native
            .prepared(kernel_id)
            .map(|p| (p._preparation.clone(), p.admission.clone()));
        let code = compile_to_binary(&jitc_kernel.source, compiler_owner)?;

        let io = jitc_kernel.io.take();
        // A precompiled kernel has no representation to read the size from:
        // it declares its shared memory statically, so the launch reserves none.
        let shared_mem_bytes = jitc_kernel
            .repr
            .as_ref()
            .map(|repr| repr.shared_memory_size())
            .unwrap_or(0);
        let entrypoint_name = jitc_kernel.entrypoint_name.clone();

        self.load_compiled_binary(
            &code,
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            jitc_kernel.cube_dim,
            shared_mem_bytes,
            io.clone().map(Arc::from),
        )?;

        // Cached after the load, so a binary the driver rejects is not handed back on the
        // next run, and the bytes move rather than being copied.
        let stored = match self.compilation_cache.as_mut().zip(key) {
            Some((cache, key)) => {
                let second_line_cache = self.second_line_compilation_cache.as_mut().unwrap();
                let stored = store_compiled(
                    cache,
                    key,
                    CompilationCacheEntry {
                        entrypoint_name,
                        shared_mem_bytes,
                        binary: code,
                        io,
                    },
                );
                store_compiled(second_line_cache, cpp_hash.unwrap(), key);
                stored
            }
            None => false,
        };
        recording.compiled(stored);
        Ok(())
    }

    fn load_compiled_binary(
        &mut self,
        code: &[i8],
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
        io: Option<Arc<[BufferIOAttr]>>,
    ) -> Result<(), CompilationError> {
        let func_name = CString::new(entrypoint_name.clone()).unwrap();
        if self.native.active() && self.native.prepared(&kernel_id).is_none() {
            return Err(CompilationError::Generic {
                reason: "HIP native module load missing exact original set route".into(),
                backtrace: BackTrace::capture(),
            });
        }
        let native = if let Some(prepared) = self.native.prepared_mut(&kernel_id) {
            let inventory = crate::prepared_native::code_inventory(code).map_err(|e| {
                CompilationError::Generic {
                    reason: e.to_string(),
                    backtrace: BackTrace::capture(),
                }
            })?;
            let owner = prepared
                .admission
                .reserve_module(inventory, code)
                .map_err(|e| CompilationError::Generic {
                    reason: e.to_string(),
                    backtrace: BackTrace::capture(),
                })?;
            prepared.inventory =
                prepared
                    .inventory
                    .add(inventory)
                    .map_err(|e| CompilationError::Generic {
                        reason: e.to_string(),
                        backtrace: BackTrace::capture(),
                    })?;
            Some(crate::prepared_native::LoadedNativeOwner::new(code, owner))
        } else {
            None
        };

        let mut module: cubecl_hip_sys::hipModule_t = std::ptr::null_mut();
        let mut func: cubecl_hip_sys::hipFunction_t = std::ptr::null_mut();
        // SAFETY: actual HIP code object; this call remains on its original actor.
        let load = unsafe {
            checked(
                "hipModuleLoadData",
                cubecl_hip_sys::hipModuleLoadData(&mut module, code.as_ptr().cast()),
            )
        };
        let result = load.and_then(|()| {
            // SAFETY: successful original module load and NUL-terminated entry.
            unsafe {
                checked(
                    "hipModuleGetFunction",
                    cubecl_hip_sys::hipModuleGetFunction(&mut func, module, func_name.as_ptr()),
                )
            }
        });
        let loaded = HipCompiledKernel {
            module,
            func,
            cube_dim,
            shared_mem_bytes,
            io,
            _native: native,
        };
        if let Some(prepared) = self.native.prepared_mut(&kernel_id) {
            // Retain even failed loads, including a null returned handle: HIP may
            // have consumed opaque allocations that no successful unload proves.
            prepared.kernels.insert(kernel_id, loaded);
            if result.is_err() {
                self.native.fence_all();
            }
        } else {
            if let Err(error) = result {
                return Err(error.into());
            }
            self.legacy_loaded += 1;
            self.modules.insert(kernel_id, loaded);
            return Ok(());
        }
        result?;

        Ok(())
    }

    /// What the compiled kernel does with each buffer binding, by buffer
    /// position — `None` when the kernel is not loaded or predates the
    /// answer, which the launch path reads as every buffer both read and
    /// written.
    pub fn kernel_io(&mut self, kernel_id: &KernelId) -> Option<Arc<[BufferIOAttr]>> {
        self.loaded(kernel_id).and_then(|kernel| kernel.io.clone())
    }

    /// Executes a task on the given stream.
    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: &[GpuResource],
    ) -> Result<(), LaunchError> {
        if let Some(prepared) = self.native.prepared(&kernel_id) {
            if !prepared.sealed || prepared.failed {
                return Err(LaunchError::Unknown {
                    reason: "HIP native set is unsealed or failed before launch".into(),
                    backtrace: BackTrace::capture(),
                });
            }
            prepared
                .admission
                .require_usable()
                .map_err(|error| LaunchError::Unknown {
                    reason: format!("HIP native original admission before launch: {error}"),
                    backtrace: BackTrace::capture(),
                })?;
        }
        let mut bindings = resources
            .iter()
            .map(|memory| memory.binding)
            .collect::<Vec<_>>();

        let kernel = self
            .loaded(&kernel_id)
            .ok_or_else(|| LaunchError::Unknown {
                reason: "HIP prepared kernel unavailable/failed/retired".into(),
                backtrace: BackTrace::capture(),
            })?;
        let cube_dim = kernel.cube_dim;

        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid HIP stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        unsafe {
            let status = cubecl_hip_sys::hipModuleLaunchKernel(
                kernel.func,
                dispatch_count.0,
                dispatch_count.1,
                dispatch_count.2,
                cube_dim.x,
                cube_dim.y,
                cube_dim.z,
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                bindings.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            if status != 0 {
                // Actual submission failure is unknown for the SAME physical actor.
                self.native.fence_all();
            }
            match checked("hipModuleLaunchKernel", status) {
                Ok(()) => Ok(()),
                Err(_)
                    if status == cubecl_hip_sys::hipError_t_hipErrorOutOfMemory
                        && !self.native.active() =>
                {
                    Err(LaunchError::OutOfMemory {
                        reason: format!("out of memory launching kernel {kernel_id:?}"),
                        backtrace: BackTrace::capture(),
                    })
                }
                Err(err) => Err(LaunchError::Unknown {
                    reason: format!("{err}, launching kernel {kernel_id:?}"),
                    backtrace: BackTrace::capture(),
                }),
            }
        }
    }

    fn validate_shared(&self, repr: &Option<HipRepresentation>) -> Result<(), LaunchError> {
        let requested = repr.as_ref().map(|repr| repr.shared_memory_size());
        let max = self.properties.hardware.max_shared_memory_size;
        if let Some(requested) = requested
            && requested > max
        {
            Err(ResourceLimitError::SharedMemory {
                requested,
                max,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}

/// RTC owner keeps actual preparation admission through checked destruction.
struct RtcProgram {
    handle: cubecl_hip_sys::hiprtcProgram,
    owner: Option<(
        crate::prepared_native::NativeRetention,
        Arc<dyn crate::prepared_native::HipNativeAdmission>,
    )>,
}
impl RtcProgram {
    fn retain_unknown(&mut self) {
        if let Some((preparation, admission)) = self.owner.take() {
            admission.fence_unknown();
            std::mem::forget(preparation);
            std::mem::forget(admission);
        }
    }
    fn destroy(&mut self) -> Result<(), CompilationError> {
        if self.handle.is_null() {
            return Ok(());
        }
        // SAFETY: this guard alone owns the original RTC handle. Failure does
        // not prove destruction; retain its charged preparation for process life.
        let result = unsafe {
            checked(
                "hiprtcDestroyProgram",
                cubecl_hip_sys::hiprtcDestroyProgram(&mut self.handle),
            )
        };
        if let Err(error) = result {
            self.retain_unknown();
            // No second destruction attempt after an indeterminate result.
            self.handle = std::ptr::null_mut();
            return Err(error.into());
        }
        self.handle = std::ptr::null_mut();
        Ok(())
    }
}
impl Drop for RtcProgram {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

/// Compile `source` to a device binary with HIP RTC.
///
/// # Errors
///
/// [`CompilationError::Generic`] carrying the compiler's own log, and the
/// source that produced it, so a kernel the driver refuses says why.
fn compile_to_binary(
    source: &str,
    owner: Option<(
        crate::prepared_native::NativeRetention,
        Arc<dyn crate::prepared_native::HipNativeAdmission>,
    )>,
) -> Result<Vec<i8>, CompilationError> {
    let source = CString::new(source).map_err(|err| CompilationError::Generic {
        reason: format!("The generated source is not a valid C string: {err}"),
        backtrace: BackTrace::capture(),
    })?;

    // SAFETY: `source` is null-terminated and outlives the call. The returned
    // handle is valid on success and owned by the guard from here on.
    let mut program = unsafe {
        let mut program: cubecl_hip_sys::hiprtcProgram = std::ptr::null_mut();
        let status = cubecl_hip_sys::hiprtcCreateProgram(
            &mut program,
            source.as_ptr(),
            std::ptr::null(), // program name seems unnecessary
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        let mut program = RtcProgram {
            handle: program,
            owner,
        };
        if let Err(error) = checked("hiprtcCreateProgram", status) {
            // Even a null failed result cannot prove no opaque allocation.
            program.retain_unknown();
            return Err(error.into());
        }
        program
    };

    let include_path = get_hip_include_path().map_err(|err| CompilationError::Generic {
        reason: format!("Unable to locate the HIP headers to compile against: {err}"),
        backtrace: BackTrace::capture(),
    })?;
    let include_option =
        CString::new(format!("-I{include_path}")).map_err(|err| CompilationError::Generic {
            reason: format!("The HIP include path is not a valid C string: {err}"),
            backtrace: BackTrace::capture(),
        })?;
    // needed for rocWMMA extension to compile
    let cpp_std_option = c"--std=c++17";
    let optimization_level = c"-O3";
    let mut options = [
        cpp_std_option.as_ptr(),
        include_option.as_ptr(),
        optimization_level.as_ptr(),
    ];

    // SAFETY: `program.handle` is the handle created above, and `options` holds
    // null-terminated pointers that outlive the call.
    let status = unsafe {
        cubecl_hip_sys::hiprtcCompileProgram(
            program.handle,
            options.len() as i32,
            options.as_mut_ptr(),
        )
    };
    if checked("hiprtcCompileProgram", status).is_err() {
        return Err(CompilationError::Generic {
            reason: format!(
                "{}\n[Source]  \n{}",
                compilation_log(&program),
                source.to_string_lossy()
            ),
            backtrace: BackTrace::capture(),
        });
    }

    // SAFETY: `program.handle` compiled successfully above, so it has code to
    // report the size of and to copy out into a buffer of exactly that size.
    unsafe {
        let mut code_size: usize = 0;
        let status = cubecl_hip_sys::hiprtcGetCodeSize(program.handle, &mut code_size);
        checked("hiprtcGetCodeSize", status)?;
        let mut code = vec![0; code_size];
        let status = cubecl_hip_sys::hiprtcGetCode(program.handle, code.as_mut_ptr());
        checked("hiprtcGetCode", status)?;
        program.destroy()?;
        Ok(code)
    }
}

/// The compiler's log for a program it refused, indented under a heading.
///
/// Reports why the log itself is missing rather than failing on it: this runs
/// on a path that already has an error to report, and losing that error to a
/// second one would leave the caller with nothing.
fn compilation_log(program: &RtcProgram) -> String {
    let mut message = "[Compilation Error] ".to_string();
    // SAFETY: `program.handle` is a valid handle; the log buffer is sized by the
    // call that reports its length, and read back as a C string.
    let log = unsafe {
        let mut log_size: usize = 0;
        let status =
            cubecl_hip_sys::hiprtcGetProgramLogSize(program.handle, &mut log_size as *mut usize);
        if let Err(err) = checked("hiprtcGetProgramLogSize", status) {
            return message + &format!("\n the log's length is unavailable: {err}");
        }
        if log_size == 0 {
            return message + "\n No compilation logs found!";
        }
        let mut log_buffer = vec![0; log_size];
        let status = cubecl_hip_sys::hiprtcGetProgramLog(program.handle, log_buffer.as_mut_ptr());
        if let Err(err) = checked("hiprtcGetProgramLog", status) {
            return message + &format!("\n the log itself is unavailable: {err}");
        }
        CStr::from_ptr(log_buffer.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    for line in log.split('\n').filter(|line| !line.is_empty()) {
        message += format!("\n    {line}").as_str();
    }
    message
}

/// A code object as the `c_char` slice the cache and `hipModuleLoadData` are written in.
///
/// `i8` and `u8` have the same layout, so this is the copy out of the compiler's buffer and
/// nothing more.
#[cfg(feature = "llvm")]
fn to_signed(bytes: &[u8]) -> Vec<i8> {
    bytes.iter().map(|byte| *byte as i8).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn cpp_cache_namespace_is_stable() {
        assert_eq!(
            super::cache_namespace("gfx1151-abc", crate::compiler::HipBackend::Cpp),
            "gfx1151-abc-cpp"
        );
    }

    /// See [`super::cache_namespace`] for why this must hold.
    #[test]
    #[cfg(feature = "llvm")]
    fn cache_namespace_separates_backends() {
        assert_ne!(
            super::cache_namespace("gfx1201-abc", crate::compiler::HipBackend::Cpp),
            super::cache_namespace("gfx1201-abc", crate::compiler::HipBackend::Llvm),
        );
    }
}
