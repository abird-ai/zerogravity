//! Native GEMM API, independent of source/model semantics.
//! The server owns the handle; the caller owns the original physical envelope.
use crate::prepared_native::failure;
use cubecl_core::server::ServerError;
use libloading::Library;
use std::ffi::{c_int, c_void};

pub const ROCBLAS_WORKSPACE_BYTES: usize = 64 * 1024 * 1024;

/// Row-major X[requests,K] times W[rows,K]^T; packed BF16 inputs and output,
/// FP32 accumulation. Geometry is data, not a model-specific recipe.
#[derive(Clone, Copy, Debug)]
pub struct RocBlasGemm {
    rows: i32,
    columns: i32,
    requests: i32,
}
impl RocBlasGemm {
    pub fn new(rows: usize, columns: usize, requests: usize) -> Result<Self, ServerError> {
        let dim = |n| {
            i32::try_from(n)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| failure("rocBLAS positive i32 matrix geometry"))
        };
        let op = Self {
            rows: dim(rows)?,
            columns: dim(columns)?,
            requests: dim(requests)?,
        };
        op.extents()?;
        Ok(op)
    }
    pub fn extents(self) -> Result<[usize; 4], ServerError> {
        let extent = |a: i32, b: i32| {
            (a as usize)
                .checked_mul(b as usize)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| failure("rocBLAS BF16 extent"))
        };
        Ok([
            extent(self.rows, self.columns)?,
            extent(self.requests, self.columns)?,
            extent(self.requests, self.rows)?,
            ROCBLAS_WORKSPACE_BYTES,
        ])
    }
}

type Create = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type SetStream = unsafe extern "C" fn(*mut c_void, cubecl_hip_sys::hipStream_t) -> c_int;
type SetWorkspace = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> c_int;
type SetPointerMode = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
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
    *const c_void,
    c_int,
    c_int,
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    u32,
) -> c_int;

#[derive(Clone, Copy)]
pub(crate) struct Api {
    destroy: Destroy,
    stream: SetStream,
    workspace: SetWorkspace,
    gemm: Gemm,
}
fn checked(status: c_int, name: &'static str) -> Result<(), ServerError> {
    if status == 0 {
        Ok(())
    } else {
        Err(failure(format!("rocBLAS {name}: {status}")))
    }
}

pub(crate) struct Provider {
    library: Option<Library>,
    pub handle: usize,
    pub api: Api,
    create: Create,
    unresolved: bool,
    pointer_mode: SetPointerMode,
}
impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocBlasProvider")
            .field("live", &(self.handle != 0))
            .finish()
    }
}
impl Provider {
    pub(crate) fn open() -> Result<Self, ServerError> {
        // SAFETY: symbols have the published rocBLAS C ABI; retain the library
        // until the original actor has actually destroyed the handle.
        unsafe {
            let library = Library::new("librocblas.so.5")
                .map_err(|e| failure(format!("rocBLAS library: {e}")))?;
            fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, ServerError> {
                unsafe { library.get::<T>(name) }
                    .map(|s| *s)
                    .map_err(|e| failure(format!("rocBLAS symbol: {e}")))
            }
            let api = Api {
                destroy: symbol(&library, b"rocblas_destroy_handle\0")?,
                stream: symbol(&library, b"rocblas_set_stream\0")?,
                workspace: symbol(&library, b"rocblas_set_workspace\0")?,
                gemm: symbol(&library, b"rocblas_gemm_ex\0")?,
            };
            let create = symbol(&library, b"rocblas_create_handle\0")?;
            let pointer_mode = symbol(&library, b"rocblas_set_pointer_mode\0")?;
            Ok(Self {
                library: Some(library),
                handle: 0,
                api,
                create,
                pointer_mode,
                unresolved: false,
            })
        }
    }
    pub(crate) fn initialize(&mut self) -> Result<(), ServerError> {
        self.unresolved = true;
        let mut handle = std::ptr::null_mut();
        let status = unsafe { (self.create)(&mut handle) };
        self.handle = handle as usize;
        checked(status, "create handle")?;
        if self.handle == 0 {
            return Err(failure("rocBLAS null handle"));
        }
        checked(
            unsafe { (self.pointer_mode)(handle, 0) },
            "host scalar mode",
        )
    }
    /// The original actor has synchronized its actual streams before this.
    pub(crate) fn retire(&mut self) -> Result<(), ServerError> {
        if self.handle == 0 {
            return Err(failure("rocBLAS unresolved handle"));
        }
        checked(
            unsafe { (self.api.destroy)(self.handle as *mut c_void) },
            "destroy handle",
        )?;
        self.handle = 0;
        self.unresolved = false;
        Ok(())
    }
}
impl Drop for Provider {
    fn drop(&mut self) {
        if self.unresolved {
            // An unresolved provider cannot unload its code while work may use it.
            if let Some(lib) = self.library.take() {
                std::mem::forget(lib);
            }
        }
    }
}
impl Api {
    /// Exact C ABI: transposed weights, contiguous input/output; alpha1 beta0.
    pub(crate) unsafe fn enqueue(
        self,
        handle: usize,
        stream: cubecl_hip_sys::hipStream_t,
        op: RocBlasGemm,
        ptrs: [*mut c_void; 4],
    ) -> Result<(), ServerError> {
        let handle = handle as *mut c_void;
        let [weights, input, output, workspace] = ptrs;
        let (alpha, beta) = (1.0f32, 0.0f32);
        unsafe {
            checked((self.stream)(handle, stream), "original stream")?;
            checked(
                (self.workspace)(handle, workspace, ROCBLAS_WORKSPACE_BYTES),
                "original workspace",
            )?;
            checked(
                (self.gemm)(
                    handle,
                    112,
                    111,
                    op.rows,
                    op.requests,
                    op.columns,
                    (&alpha as *const f32).cast(),
                    weights,
                    168,
                    op.columns,
                    input,
                    168,
                    op.columns,
                    (&beta as *const f32).cast(),
                    output,
                    168,
                    op.rows,
                    output,
                    168,
                    op.rows,
                    151,
                    0,
                    0,
                    0,
                ),
                "gemm_ex",
            )
        }
    }
}
