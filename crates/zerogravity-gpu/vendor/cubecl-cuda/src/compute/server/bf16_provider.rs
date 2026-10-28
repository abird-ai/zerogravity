//! Pinned cuBLAS entry points and native handles never leave CudaServer.
use super::*;
use crate::OwnedBlasAllocationPool;
use crate::owned_command::{Bf16BlasSources, Bf16GemmBindings, Bf16GemmOperation};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dladdr(pointer: *const c_void, info: *mut DlInfo) -> c_int;
}
#[repr(C)]
struct DlInfo {
    filename: *const c_char,
    base: *mut c_void,
    symbol: *const c_char,
    address: *mut c_void,
}
type Create = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type SetStream = unsafe extern "C" fn(*mut c_void, CUstream) -> c_int;
type SetWorkspace = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> c_int;
type SetMode = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type Gemm = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
#[derive(Clone, Copy)]
struct Api {
    create: Create,
    destroy: Destroy,
    stream: SetStream,
    workspace: SetWorkspace,
    pointer: SetMode,
    math: SetMode,
    atomics: SetMode,
    gemm: Gemm,
}
/// No automatic destructor: unknown completion keeps native state quarantined.
pub(super) struct BlasProvider {
    libraries: [usize; 2],
    api: Option<Api>,
    handle: usize,
    ready: bool,
    sources: Arc<Bf16BlasSources>,
    pool: Arc<OwnedBlasAllocationPool>,
}
impl std::fmt::Debug for BlasProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlasProvider")
            .field("ready", &self.ready)
            .finish_non_exhaustive()
    }
}
fn failure(message: impl Into<String>) -> ServerError {
    ServerError::graph_state(message)
}
fn status(code: c_int, operation: &str) -> Result<(), ServerError> {
    if code == 0 {
        Ok(())
    } else {
        Err(failure(format!("pinned cuBLAS {operation} status {code}")))
    }
}
unsafe fn symbol(
    library: usize,
    name: &CStr,
    sources: &Bf16BlasSources,
    index: usize,
) -> Result<*mut c_void, ServerError> {
    let pointer = unsafe { dlsym(library as *mut c_void, name.as_ptr()) };
    if pointer.is_null() {
        return Err(failure(format!("pinned cuBLAS symbol {name:?}")));
    }
    let mut info = DlInfo {
        filename: ptr::null(),
        base: ptr::null_mut(),
        symbol: ptr::null(),
        address: ptr::null_mut(),
    };
    if unsafe { dladdr(pointer, &mut info) } == 0 || info.filename.is_null() {
        return Err(failure("pinned cuBLAS loaded symbol origin absent"));
    }
    let path = Path::new(std::ffi::OsStr::from_bytes(
        unsafe { CStr::from_ptr(info.filename) }.to_bytes(),
    ));
    sources.require_loaded_file(index, path)?;
    Ok(pointer)
}
impl CudaServer {
    pub(crate) fn create_owned_bf16_blas(
        &mut self,
        stream: StreamId,
        sources: Arc<Bf16BlasSources>,
        pool: Arc<OwnedBlasAllocationPool>,
    ) -> Result<(), ServerError> {
        if self.blas_provider.is_some() {
            return Err(failure("one pinned cuBLAS provider per original actor"));
        }
        let native_stream = self.profiled_stream(stream, "diagnostic cuBLAS create")?;
        pool.begin_provider()?;
        // Install ownership before dlopen/constructors. Errors retain all partial
        // state for the caller's original initialization quarantine.
        self.blas_provider = Some(BlasProvider {
            libraries: [0; 2],
            api: None,
            handle: 0,
            ready: false,
            sources,
            pool,
        });
        let provider = self.blas_provider.as_mut().unwrap();
        let paths = provider.sources.fd_paths();
        for (index, path) in paths.iter().enumerate() {
            let path = CString::new(path.as_os_str().as_bytes())
                .map_err(|_| failure("pinned cuBLAS FD path NUL"))?;
            // Exact retained LT is loaded first; its verified SONAME satisfies
            // BLAS's dependency. No alternate directory/provider fallback.
            let library = unsafe { dlopen(path.as_ptr(), 2 | 0x100) }; // NOW|GLOBAL
            if library.is_null() {
                return Err(failure("pinned cuBLAS retained-FD dlopen failed"));
            }
            provider.libraries[index] = library as usize;
            let probe = if index == 0 {
                c"cublasLtGetVersion"
            } else {
                c"cublasGetVersion_v2"
            };
            unsafe {
                symbol(library as usize, probe, &provider.sources, index)?;
            }
        }
        let library = provider.libraries[1];
        macro_rules! get {
            ($name:literal, $ty:ty) => {
                unsafe {
                    std::mem::transmute::<*mut c_void, $ty>(symbol(
                        library,
                        $name,
                        &provider.sources,
                        1,
                    )?)
                }
            };
        }
        let api = Api {
            create: get!(c"cublasCreate_v2", Create),
            destroy: get!(c"cublasDestroy_v2", Destroy),
            stream: get!(c"cublasSetStream_v2", SetStream),
            workspace: get!(c"cublasSetWorkspace_v2", SetWorkspace),
            pointer: get!(c"cublasSetPointerMode_v2", SetMode),
            math: get!(c"cublasSetMathMode", SetMode),
            atomics: get!(c"cublasSetAtomicsMode", SetMode),
            gemm: get!(c"cublasGemmEx", Gemm),
        };
        provider.api = Some(api);
        let mut handle = ptr::null_mut();
        let created = unsafe { (api.create)(&mut handle) };
        provider.handle = handle as usize;
        status(created, "create")?;
        if handle.is_null() {
            return Err(failure("pinned cuBLAS null successful handle"));
        }
        status(
            unsafe { (api.stream)(handle, native_stream) },
            "set original stream",
        )?;
        status(unsafe { (api.pointer)(handle, 0) }, "host pointer mode")?;
        status(unsafe { (api.math)(handle, 0) }, "default math mode")?;
        status(unsafe { (api.atomics)(handle, 0) }, "atomics mode")?;
        if provider.pool.snapshot()?.poisoned {
            return Err(failure("cuBLAS allocator poisoned during initialization"));
        }
        provider.ready = true;
        Ok(())
    }
    pub(crate) fn enqueue_owned_bf16_gemm(
        &mut self,
        stream: StreamId,
        operation: Bf16GemmOperation,
        bindings: Bf16GemmBindings,
    ) -> Result<(), ServerError> {
        self.inspect_owned_bf16_gemm(
            stream,
            operation,
            Bf16GemmBindings {
                weights: bindings.weights.clone(),
                input: bindings.input.clone(),
                output: bindings.output.clone(),
                workspace: bindings.workspace.clone(),
            },
        )?;
        let provider = self
            .blas_provider
            .as_ref()
            .ok_or_else(|| failure("original cuBLAS provider absent"))?;
        if !provider.ready || provider.pool.snapshot()?.poisoned {
            return Err(failure("original cuBLAS provider unavailable"));
        }
        let api = provider
            .api
            .ok_or_else(|| failure("original cuBLAS API absent"))?;
        let handle = provider.handle;
        let mut written = self.write_set();
        written.push(bindings.output.clone().binding());
        written.push(bindings.workspace.clone().binding());
        ExecuteScope::over(self, stream, written)
            .execute(move |server| {
                server.streams.ensure_written(
                    [
                        &bindings.weights.clone().binding(),
                        &bindings.input.clone().binding(),
                    ]
                    .into_iter(),
                )?;
                let native = bindings.handles().map(|handle| handle.clone().binding());
                let mut command = server.command(stream, native.iter())?;
                if command.stream().capturing.is_recording() {
                    return Err(failure("diagnostic cuBLAS cannot enter graph capture"));
                }
                let weights = command.resource(native[0].clone())?.ptr;
                let input = command.resource(native[1].clone())?.ptr;
                let output = command.resource(native[2].clone())?.ptr;
                let workspace = command.resource(native[3].clone())?.ptr;
                let handle = handle as *mut c_void;
                // SetStream resets workspace; explicit workspace MUST follow it.
                status(
                    unsafe { (api.stream)(handle, command.stream().sys) },
                    "set original stream",
                )?;
                status(
                    unsafe {
                        (api.workspace)(
                            handle,
                            workspace as *mut c_void,
                            operation.workspace_bytes(),
                        )
                    },
                    "set admitted workspace",
                )?;
                let contract = operation.api();
                let alpha = f32::from_bits(contract.alpha_bits);
                let beta = f32::from_bits(contract.beta_bits);
                status(
                    unsafe {
                        (api.gemm)(
                            handle,
                            contract.transpose_a,
                            contract.transpose_b,
                            operation.m() as c_int,
                            operation.n() as c_int,
                            operation.k() as c_int,
                            (&alpha as *const f32).cast(),
                            weights as *const c_void,
                            contract.data_type,
                            operation.k() as c_int,
                            input as *const c_void,
                            contract.data_type,
                            operation.k() as c_int,
                            (&beta as *const f32).cast(),
                            output as *mut c_void,
                            contract.data_type,
                            operation.m() as c_int,
                            contract.compute_type,
                            contract.requested_algorithm,
                        )
                    },
                    "GemmEx enqueue",
                )
            })
            .into_result()
    }
    pub(crate) fn destroy_owned_bf16_blas(&mut self) -> Result<(), ServerError> {
        self.ctx
            .unsafe_set_current()
            .map_err(|e| failure(format!("cuBLAS destroy original context: {e:?}")))?;
        let provider = self
            .blas_provider
            .as_mut()
            .ok_or_else(|| failure("original cuBLAS provider absent"))?;
        let api = provider
            .api
            .ok_or_else(|| failure("original cuBLAS partial provider cannot safely retire"))?;
        status(
            unsafe { (api.destroy)(provider.handle as *mut c_void) },
            "destroy after original completion",
        )?;
        provider.handle = 0;
        provider.ready = false;
        for index in [1, 0] {
            if unsafe { dlclose(provider.libraries[index] as *mut c_void) } != 0 {
                return Err(failure(
                    "pinned cuBLAS unload failed; original ownership retained",
                ));
            }
            provider.libraries[index] = 0;
        }
        provider.pool.finish_provider()?;
        self.blas_provider = None;
        Ok(())
    }
}

impl CudaServer {
    /// Bind identity while the original actor's primary reference is already
    /// alive. No context retain/create or raw pointer crosses the public seam.
    pub(crate) fn bind_owned_blas_pool(&mut self) -> Result<(), ServerError> {
        self.ctx
            .unsafe_set_current()
            .map_err(|e| failure(format!("bind original BLAS context: {e:?}")))?;
        let mut device = -1;
        let result = unsafe { cudarc::driver::sys::cuCtxGetDevice(&mut device) };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(failure("original BLAS current device query"));
        }
        let mut flags = 0;
        let mut active = 0;
        let result = unsafe {
            cudarc::driver::sys::cuDevicePrimaryCtxGetState(device, &mut flags, &mut active)
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS || active != 1 {
            return Err(failure(
                "original BLAS primary context is not already active",
            ));
        }
        crate::owned_blas_allocator::bind_original_context(device, self.ctx.context as usize)
    }
}
