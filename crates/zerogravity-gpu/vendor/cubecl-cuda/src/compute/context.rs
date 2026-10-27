use cubecl_cpp::cuda::arch::CudaArchitecture;
use cubecl_cpp::formatter::format_cpp;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::kernel::BufferIOAttr;
use cubecl_server::kernel::DebugInformation;
use cubecl_server::{
    compiler::{CompilationError, build_id_hash},
    validation::{validate_cube_dim, validate_units},
};

use crate::compiler::{CudaBackend, CudaCompilationOptions, CudaCompiler, CudaRepresentation};
use crate::compute::events::{EventProfiler, driver_error, poisons_device};
use crate::compute::stream::Stream;
use cubecl_core::{
    hash::{StableHash, StableHasher},
    ir::DeviceProperties,
    prelude::*,
    server::ResourceLimitError,
};
use cubecl_environment::persistence::Store;
#[cfg(feature = "llvm")]
use cubecl_llvm::nvptx::ptx_version::PtxVersion;
use cubecl_server::{
    compiler::KernelCacheKey,
    kernel::{CompiledKernel, CubeKernel},
    logging::ServerLogger,
};
use cudarc::driver::sys::CUfunc_st;
use cudarc::driver::sys::{CUctx_st, CUfunction_attribute, CUstream};
use std::ffi::CString;
use std::ffi::c_char;
use std::sync::Arc;
use std::{ffi::CStr, os::raw::c_void};

use cubecl_server::compiler::{
    CompilationCache, CompilationRecording, compilation_store, store_compiled,
};

#[cfg(feature = "precompiled-shared")]
mod precompiled_shared;

mod compiler_context;
mod compiler_limits;
mod nvrtc;
mod program_lifetime;
use compiler_context::CompilerContextState;
pub use compiler_limits::{CompilerContextUsage, CudaCompilerContextPolicy};

#[derive(Debug)]
pub(crate) struct CudaContext {
    pub context: *mut CUctx_st,
    /// The stream collectives run on. Kept on the context so a relocation —
    /// which reaches the context, not the server — can wait on it.
    pub comm_stream: CUstream,
    /// The modules loaded on the device, in front of [`Self::ptx_cache`].
    ///
    /// An environment switch drops these, and nothing unloads the modules they
    /// name: see [`CudaContext::is_loaded`].
    modules: CompilationCache<KernelId, CudaCompiledKernel>,
    ptx_cache: Option<Store<KernelCacheKey, PtxCacheEntry>>,
    /// Cache mapping C++ code hashes to the key that first compiled them. We can skip the slow CUDA
    /// compiler if we already have a compiled artifact for the same code.
    ///
    /// C++ only: the LLVM backend emits PTX directly, so there is no intermediate source to
    /// key a second line on.
    second_line_ptx_cache: Option<Store<StableHash, KernelCacheKey>>,
    pub profiler: EventProfiler,
    pub arch: CudaArchitecture,
    pub compilation_options: CudaCompilationOptions,
    pub properties: DeviceProperties,
    build_id: StableHash,
    /// Failed program retirement/module load forbids reuse and healthy sync.
    compiler_unknown: bool,
    compiler_context: Option<CompilerContextState>,
    compilation_attempts: usize,
}

#[derive(Debug)]
pub struct CudaCompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
    /// What the kernel does with each buffer binding, by buffer position --
    /// the compiler's answer, carried here because on a cache hit nothing
    /// else of the compilation survives. `None` for entries persisted before
    /// the answer existed, which the launch path reads as every buffer both
    /// read and written.
    io: Option<Arc<[BufferIOAttr]>>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    ptx: Vec<std::ffi::c_char>,
    /// See [`CudaCompiledKernel::io`]; defaulted for entries persisted before the
    /// field existed.
    #[serde(default)]
    io: Option<Vec<BufferIOAttr>>,
}

/// The namespace a backend's compiled artifacts live under.
///
/// Both backends emit PTX, so without the backend in the key a stale artifact from one would
/// load and run happily under the other -- tests passing while measuring nothing. The LLVM
/// backend's PTX version follows the driver, so it is in the key too: after a driver downgrade,
/// PTX newer than the driver loads would otherwise be read back and refused.
fn cache_namespace(
    fingerprint: &str,
    backend: CudaBackend,
    #[cfg(feature = "llvm")] ptx_version: Option<PtxVersion>,
) -> String {
    match backend {
        CudaBackend::Cpp => format!("{fingerprint}-cpp"),
        #[cfg(feature = "llvm")]
        CudaBackend::Llvm => match ptx_version {
            None => format!("{fingerprint}-llvm"),
            Some(ptx_version) => format!("{fingerprint}-llvm-{ptx_version}"),
        },
    }
}

impl CudaContext {
    /// `backend` is which one compiles here; see [`cache_namespace`].
    pub fn new(
        compilation_options: CudaCompilationOptions,
        properties: DeviceProperties,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
        backend: CudaBackend,
        comm_stream: CUstream,
    ) -> Self {
        let fingerprint = cache_namespace(
            &format!("ptx_sm{}", arch.version),
            backend,
            #[cfg(feature = "llvm")]
            compilation_options.ptx_version,
        );
        let ptx_cache = compilation_store("cuda", &fingerprint);
        let second_line_ptx_cache = compilation_store("cuda-second-line", fingerprint);

        Self {
            context,
            comm_stream,
            modules: CompilationCache::mirroring(&ptx_cache),
            ptx_cache,
            second_line_ptx_cache,
            arch,
            profiler: EventProfiler::default(),
            compilation_options,
            properties,
            build_id: build_id_hash(),
            compiler_unknown: false,
            compiler_context: None,
            compilation_attempts: 0,
        }
    }

    /// Whether `kernel_id` is already loaded on the device.
    ///
    /// An environment switch drops the whole cache, so the new environment's
    /// PTX store is filled rather than bypassed. The modules those entries
    /// named stay resident: nothing calls `cuModuleUnload`, here or anywhere
    /// else in this context, and unloading one a stream still has queued work
    /// against would be unsound. A process that switches environments a handful
    /// of times at startup pays a bounded price; one that switches repeatedly
    /// grows its resident modules without bound — see
    /// [`cubecl_environment::environment::activate`].
    pub fn is_loaded(&mut self, kernel_id: &KernelId) -> bool {
        !self.compiler_unknown && self.modules.contains(kernel_id)
    }

    /// Switches the current CUDA context to this context.
    pub fn unsafe_set_current(&self) -> Result<(), cubecl_core::server::ServerError> {
        self.check_compiler()?;
        // SAFETY: `self.context` is a valid CUDA context obtained from `primary_ctx::retain`
        // during server initialization and remains valid for the server's lifetime.
        unsafe { cudarc::driver::result::ctx::set_current(self.context) }
            .map_err(|err| driver_error("cuCtxSetCurrent", err).into())
    }

    fn try_load_cached(
        &mut self,
        kernel_id: &KernelId,
    ) -> Result<Result<(), Option<KernelCacheKey>>, CompilationError> {
        let key = if let Some(cache) = self.ptx_cache.as_mut() {
            let key = KernelCacheKey::new(kernel_id, self.build_id);

            if let Some(entry) = cache.remove(&key) {
                log::trace!("Using PTX cache");

                self.load_ptx(
                    entry.ptx,
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
        kernel: Box<dyn CubeKernel>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        self.check_compiler()?;
        self.compilation_attempts = self
            .compilation_attempts
            .checked_add(1)
            .ok_or_else(|| self.compiler_limit("CUDA compilation attempt overflow"))?;
        if let Some(context) = self.compiler_context.as_mut() {
            context.begin().map_err(|e| self.compiler_limit(e))?;
        }
        let mut recording = CompilationRecording::new(kernel_id);
        let key = match self.try_load_cached(kernel_id)? {
            Ok(()) => {
                recording.loaded();
                return Ok(());
            }
            Err(key) => key,
        };

        log::trace!("Compiling kernel");

        validate_cube_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        let definition = cubecl_core::define_kernel(&*kernel)?;
        recording.defined(&definition);
        let jitc_kernel = CompiledKernel::compile(
            &*kernel,
            definition,
            &mut CudaCompiler::default(),
            &self.compilation_options,
        )?;

        self.validate_shared(&jitc_kernel.repr)?;
        #[cfg(feature = "precompiled-shared")]
        if jitc_kernel.repr.is_none() {
            precompiled_shared::bytes(
                &jitc_kernel.source,
                self.properties.hardware.max_shared_memory_size,
            )?;
        }
        recording.source(&jitc_kernel.source);

        self.load_jit_kernel(kernel_id, key, jitc_kernel, logger, recording)
    }

    /// Loads what the compiler produced, by the route that backend's output takes, and closes
    /// `recording` with how the artifact was obtained.
    fn load_jit_kernel(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        jitc_kernel: CompiledKernel<CudaCompiler>,
        logger: Arc<ServerLogger>,
        recording: CompilationRecording,
    ) -> Result<(), LaunchError> {
        match &jitc_kernel.repr {
            Some(CudaRepresentation::Cpp(_)) => {
                self.load_transpiled(kernel_id, key, jitc_kernel, logger, recording)
            }
            #[cfg(feature = "llvm")]
            Some(CudaRepresentation::Llvm(_)) => {
                self.load_emitted_ptx(kernel_id, key, jitc_kernel, logger, recording)
            }
            // A precompiled kernel: its text passed the language check in
            // `CompiledKernel::compile`, so it is whatever the default backend reads. CUDA
            // C++ goes through NVRTC like a transpiled kernel; the LLVM backend produces PTX
            // from the dialect and has no route for text.
            None => match CudaBackend::default() {
                CudaBackend::Cpp => {
                    self.load_transpiled(kernel_id, key, jitc_kernel, logger, recording)
                }
                #[cfg(feature = "llvm")]
                CudaBackend::Llvm => Err(CompilationError::Generic {
                    reason: "the LLVM backend cannot load a precompiled kernel: it has no text to \
                         compile from"
                        .to_string(),
                    backtrace: BackTrace::capture(),
                }
                .into()),
            },
        }
    }

    /// Turns a kernel the LLVM backend just compiled into a loaded module.
    ///
    /// What it hands back is already PTX, so no `nvrtc*` call belongs here: the bytes go
    /// straight to [`Self::load_ptx`], exactly as a cache hit would. The driver still JITs
    /// them when the module loads, which is what it does with NVRTC's output too.
    #[cfg(feature = "llvm")]
    fn load_emitted_ptx(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        mut jitc_kernel: CompiledKernel<CudaCompiler>,
        logger: Arc<ServerLogger>,
        recording: CompilationRecording,
    ) -> Result<(), LaunchError> {
        let Some(CudaRepresentation::Llvm(module)) = &jitc_kernel.repr else {
            unreachable!("dispatched on the representation");
        };

        if logger.compilation_source_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("ll", kernel_id.clone()));
        }
        logger.log_compilation(&jitc_kernel);

        let ptx = module.ptx.clone();
        let shared_mem_bytes = module.shared_memory_size;
        let io = jitc_kernel.io.take();
        let entrypoint_name = jitc_kernel.entrypoint_name.clone();

        self.load_ptx(
            ptx.clone(),
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            jitc_kernel.cube_dim,
            shared_mem_bytes,
            io.clone().map(Arc::from),
        )?;

        // Cached after the load, so PTX the driver rejects is not handed back on the next run.
        // `try_load_cached` hands back a key exactly when there is a cache to put it in. No
        // second-line entry: that cache is keyed on generated C++ source, which this backend
        // never produces.
        let stored = match self.ptx_cache.as_mut().zip(key) {
            Some((cache, key)) => store_compiled(
                cache,
                key,
                PtxCacheEntry {
                    entrypoint_name,
                    shared_mem_bytes,
                    ptx,
                    io,
                },
            ),
            None => false,
        };
        recording.compiled(stored);
        Ok(())
    }

    /// Turns a kernel the C++ backend just transpiled into a loaded module, by running its
    /// source through NVRTC first.
    fn load_transpiled(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        mut jitc_kernel: CompiledKernel<CudaCompiler>,
        logger: Arc<ServerLogger>,
        recording: CompilationRecording,
    ) -> Result<(), LaunchError> {
        if self.compiler_context.is_some() {
            CudaCompilerContextPolicy::validate_compiled_metadata(
                jitc_kernel.entrypoint_name.len(),
                jitc_kernel.io.as_ref().map(Vec::len),
            )
            .map_err(|e| self.compiler_limit(e))?;
        }
        // Pin this before optional formatting; persisted PTX carries the same
        // admitted launch metadata on both ordinary and second-line cache hits.
        let shared_mem_bytes = match jitc_kernel.repr.as_ref() {
            Some(repr) => repr.shared_memory_size(),
            None => {
                #[cfg(feature = "precompiled-shared")]
                {
                    precompiled_shared::bytes(
                        &jitc_kernel.source,
                        self.properties.hardware.max_shared_memory_size,
                    )?
                }
                #[cfg(not(feature = "precompiled-shared"))]
                {
                    0
                }
            }
        };
        if logger.compilation_source_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&jitc_kernel.source) {
                jitc_kernel.source = formatted;
            }
        }

        let cpp_hash = if let Some(cache) = self.ptx_cache.as_mut() {
            let key = key.unwrap();
            let second_line_cache = self.second_line_ptx_cache.as_mut().unwrap();
            let cpp_hash = StableHasher::hash_one(&jitc_kernel.source);

            if let Some(old_key) = second_line_cache.purge_key(&cpp_hash)
                && let Some(entry) = cache.purge_key(&old_key)
            {
                log::trace!("Using second-line PTX cache");
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

        logger.log_compilation(&jitc_kernel);

        if let Some(context) = self.compiler_context.as_mut() {
            context
                .source(jitc_kernel.source.len())
                .map_err(|e| self.compiler_limit(e))?;
        }
        let ptx = self.compile_to_ptx(&jitc_kernel.source)?;

        let cube_dim = jitc_kernel.cube_dim;
        let io = jitc_kernel.io.take();

        let stored = match &mut self.ptx_cache {
            Some(cache) => {
                let second_line_cache = self.second_line_ptx_cache.as_mut().unwrap();
                let key = key.unwrap();
                let stored = store_compiled(
                    cache,
                    key,
                    PtxCacheEntry {
                        entrypoint_name: jitc_kernel.entrypoint_name.clone(),
                        shared_mem_bytes,
                        ptx: ptx.clone(),
                        io: io.clone(),
                    },
                );
                store_compiled(second_line_cache, cpp_hash.unwrap(), key);
                stored
            }
            None => false,
        };

        self.load_ptx(
            ptx,
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            cube_dim,
            shared_mem_bytes,
            io.map(Arc::from),
        )?;
        recording.compiled(stored);
        Ok(())
    }

    fn load_ptx(
        &mut self,
        ptx: Vec<c_char>,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
        io: Option<Arc<[BufferIOAttr]>>,
    ) -> Result<(), CompilationError> {
        if shared_mem_bytes > self.properties.hardware.max_shared_memory_size {
            return Err(CompilationError::Generic {
                reason: "cached CUDA dynamic shared memory exceeds reported hardware".into(),
                backtrace: BackTrace::capture(),
            });
        }
        dump_ptx(&kernel_id, &ptx);

        let func_name = CString::new(entrypoint_name).map_err(|err| CompilationError::Generic {
            reason: format!("CUDA entry point: {err}"),
            backtrace: BackTrace::capture(),
        })?;
        // SAFETY: `ptx` is a valid null-terminated PTX binary from NVRTC. `func_name` is a
        // null-terminated `CString` matching the kernel entry point in the compiled module.
        let func = unsafe {
            let module = cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _)
                .map_err(|err| self.compiler_failure(format!(
                    "cuModuleLoadData failed with uncertain module lifetime: {err}; original submitting owners must remain retained"
                )))?;

            cudarc::driver::result::module::get_function(module, func_name).map_err(|err| {
                // Loading already changed the original context. No function
                // or cache entry does not establish retirement of that module.
                self.compiler_failure(format!(
                    "cuModuleGetFunction failed after module load: {err}; original submitting owners must remain retained"
                ))
            })?
        };

        self.modules.insert(
            kernel_id.clone(),
            CudaCompiledKernel {
                cube_dim,
                shared_mem_bytes,
                func,
                io,
            },
        );

        Ok(())
    }

    /// What the compiled kernel does with each buffer binding, by buffer
    /// position — `None` when the kernel is not loaded or predates the
    /// answer, which the launch path reads as every buffer both read and
    /// written.
    pub fn kernel_io(&mut self, kernel_id: &KernelId) -> Option<Arc<[BufferIOAttr]>> {
        self.modules
            .get(kernel_id)
            .and_then(|kernel| kernel.io.clone())
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn kernel_parameter_layout(
        &mut self,
        kernel_id: &KernelId,
        parameter_count: usize,
    ) -> Result<Vec<(usize, usize)>, CompilationError> {
        self.check_compiler()?;
        let function = self
            .modules
            .get(kernel_id)
            .map(|kernel| kernel.func)
            .ok_or_else(|| self.compiler_limit("CUDA parameter diagnostic kernel is not loaded"))?;
        let mut layout = Vec::with_capacity(parameter_count);
        for index in 0..parameter_count {
            let mut offset = usize::MAX;
            let mut bytes = usize::MAX;
            let status = unsafe {
                cudarc::driver::sys::cuFuncGetParamInfo(
                    function,
                    index,
                    &mut offset,
                    &mut bytes,
                )
            };
            if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(self.compiler_limit(format!(
                    "cuFuncGetParamInfo parameter {index}: {status:?}"
                )));
            }
            layout.push((offset, bytes));
        }
        let mut offset = 0;
        let mut bytes = 0;
        let status = unsafe {
            cudarc::driver::sys::cuFuncGetParamInfo(
                function,
                parameter_count,
                &mut offset,
                &mut bytes,
            )
        };
        if status != cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE {
            return Err(self.compiler_limit(format!(
                "CUDA parameter diagnostic count differs: {status:?}"
            )));
        }
        Ok(layout)
    }

    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: &mut [*mut c_void],
    ) -> Result<(), LaunchError> {
        self.check_compiler()?;
        let kernel = self.modules.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;
        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid CUDA stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                kernel.func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kernel.shared_mem_bytes as i32,
            )
            .map_err(|err| launch_failed("cuFuncSetAttribute", err))?;
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                resources,
            )
            .map_err(|err| launch_failed("cuLaunchKernel", err))?;
        };

        Ok(())
    }

    fn validate_shared(&self, repr: &Option<CudaRepresentation>) -> Result<(), LaunchError> {
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

/// Writes the exact PTX handed to the driver only when the explicit dump switch is set.
/// Each attempt has a unique create-new path and full compilation id sidecar;
/// truncated display suffixes cannot overwrite settings-distinct witnesses.
fn dump_ptx(kernel_id: &KernelId, ptx: &[c_char]) {
    let Some(dir) = std::env::var_os("CUBECL_CUDA_DUMP_PTX") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }

    // Same complete Debug id as the full compilation logger. The hash is an
    // index, never identity proof; sequence + create_new prevent overwriting.
    let id = format!("{kernel_id:#?}");
    // SAFETY: the PTX handed to the driver is a null-terminated C string.
    let text = unsafe { CStr::from_ptr(ptx.as_ptr()) };
    // Diagnostic I/O remains best effort; a witness validator must reject any
    // missing, partial or unpaired artifact instead of inferring coverage.
    let _ = write_ptx_dump(&dir, &id, text.to_bytes());
}

static PTX_DUMP_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn write_ptx_dump(
    dir: &std::path::Path,
    kernel_id: &str,
    ptx: &[u8],
) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let hash = StableHasher::hash_one(&kernel_id);
    for _ in 0..64 {
        let sequence = PTX_DUMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let stem = format!("ptx-p{}-s{sequence:016x}-k{hash:032x}", std::process::id());
        let path = dir.join(format!("{stem}.ptx"));
        let mut output = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        };
        // Reserve both files before writing. Never overwrite an existing id or
        // PTX, even on hash collision, sequence wrap, pid reuse or repeated id.
        let mut id = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.with_extension("kernel-id.txt"))?;
        id.write_all(kernel_id.as_bytes())?;
        output.write_all(ptx)?;
        return Ok(path);
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "CUDA PTX diagnostic path collisions exhausted",
    ))
}

/// A refused launch error : maps to a poisoned-device error if that call happened right after
/// a fault.
fn launch_failed(op: &'static str, err: cudarc::driver::DriverError) -> LaunchError {
    match poisons_device(err.0) {
        true => driver_error(op, err).into(),
        false => LaunchError::Unknown {
            reason: format!("{err}"),
            backtrace: BackTrace::capture(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::cache_namespace;
    use crate::compiler::CudaBackend;

    #[test]
    fn ptx_dumps_keep_suffix_collisions_and_repeated_ids_distinct() {
        struct OwnedDirectory(std::path::PathBuf);
        impl Drop for OwnedDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let sequence = super::PTX_DUMP_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "cubecl-ptx-dump-test-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        let directory = OwnedDirectory(path);
        // Synthetic transport fixtures, never compiler or numerical witnesses.
        let suffix = "same settings suffix ".repeat(20);
        let first = format!("full-first-id {suffix}");
        let second = format!("full-second-id {suffix}");
        let samples = [
            (first.as_str(), b"synthetic ptx one\n".as_slice()),
            (second.as_str(), b"synthetic ptx two\n".as_slice()),
            (first.as_str(), b"synthetic repeated id\n".as_slice()),
        ];
        let mut paths = Vec::new();
        for (id, ptx) in samples {
            let path = super::write_ptx_dump(&directory.0, id, ptx).unwrap();
            assert!(!paths.contains(&path));
            assert!(path.file_name().unwrap().len() < 255);
            assert_eq!(std::fs::read(&path).unwrap(), ptx);
            assert_eq!(
                std::fs::read_to_string(path.with_extension("kernel-id.txt")).unwrap(),
                id
            );
            paths.push(path);
        }
        assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 6);
    }

    #[test]
    fn cpp_cache_namespace_is_stable() {
        assert_eq!(
            cache_namespace(
                "ptx_sm86",
                CudaBackend::Cpp,
                #[cfg(feature = "llvm")]
                None
            ),
            "ptx_sm86-cpp"
        );
    }
    #[cfg(feature = "llvm")]
    use cubecl_llvm::nvptx::ptx_version::PtxVersion;

    /// See [`super::cache_namespace`] for why this must hold.
    #[test]
    #[cfg(feature = "llvm")]
    fn cache_namespace_separates_backends() {
        assert_ne!(
            cache_namespace("ptx_sm86", CudaBackend::Cpp, None),
            cache_namespace("ptx_sm86", CudaBackend::Llvm, None),
        );
    }

    /// See [`super::cache_namespace`] for why this must hold.
    #[test]
    #[cfg(feature = "llvm")]
    fn cache_namespace_separates_the_llvm_backends_ptx_versions() {
        assert_ne!(
            cache_namespace("ptx_sm86", CudaBackend::Llvm, PtxVersion::for_driver(12080)),
            cache_namespace("ptx_sm86", CudaBackend::Llvm, PtxVersion::for_driver(12090)),
        );
    }
}
