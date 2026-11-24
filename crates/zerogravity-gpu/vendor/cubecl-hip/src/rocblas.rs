//! Native GEMM API, independent of source/model semantics.
//! The server owns the handle; the caller owns the original physical envelope.
use crate::prepared_native::failure;
use cubecl_core::server::ServerError;
use libloading::Library;
use std::ffi::{c_int, c_void};

pub const ROCBLAS_WORKSPACE_BYTES: usize = 64 * 1024 * 1024;

/// Row-major dense matrix product. The backend performs the transposed
/// column-major operation without copying either input.
#[derive(Clone, Copy, Debug)]
pub struct RocBlasGemm {
    rows: i32,     // output columns
    columns: i32,  // inner
    requests: i32, // output rows
    batches: i32,
    transpose_left: bool,
    transpose_right: bool,
    bf16: bool,
}
impl RocBlasGemm {
    fn dim(n: usize) -> Result<i32, ServerError> {
        i32::try_from(n)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| failure("rocBLAS positive i32 matrix geometry"))
    }
    /// BF16 X[requests,K] * W[rows,K]^T, with packed BF16 output.
    pub fn new(rows: usize, columns: usize, requests: usize) -> Result<Self, ServerError> {
        let op = Self {
            rows: Self::dim(rows)?,
            columns: Self::dim(columns)?,
            requests: Self::dim(requests)?,
            batches: 1,
            transpose_left: false,
            transpose_right: true,
            bf16: true,
        };
        op.extents()?;
        Ok(op)
    }
    /// Contiguous independent row-major F32 products, no broadcast or padding.
    pub fn f32_batched(
        m: usize,
        n: usize,
        k: usize,
        batches: usize,
        transpose_left: bool,
        transpose_right: bool,
    ) -> Result<Self, ServerError> {
        let op = Self {
            rows: Self::dim(n)?,
            columns: Self::dim(k)?,
            requests: Self::dim(m)?,
            batches: Self::dim(batches)?,
            transpose_left,
            transpose_right,
            bf16: false,
        };
        op.extents()?;
        Ok(op)
    }
    pub(crate) fn element_bytes(self) -> usize {
        if self.bf16 { 2 } else { 4 }
    }
    /// Binding order: right, left, output, workspace.
    pub fn extents(self) -> Result<[usize; 4], ServerError> {
        let extent = |a: i32, b: i32| {
            (a as usize)
                .checked_mul(b as usize)
                .and_then(|n| n.checked_mul(self.batches as usize))
                .and_then(|n| n.checked_mul(self.element_bytes()))
                .ok_or_else(|| failure("rocBLAS matrix extent"))
        };
        Ok([
            extent(self.rows, self.columns)?,
            extent(self.requests, self.columns)?,
            extent(self.requests, self.rows)?,
            ROCBLAS_WORKSPACE_BYTES,
        ])
    }
}

/// Compact row-major FP32 L*X=B, unit lower, independent batches.
#[derive(Clone, Copy, Debug)]
pub struct RocBlasUnitLowerSolve {
    order: i32,
    rhs_columns: i32,
    batches: i32,
}
impl RocBlasUnitLowerSolve {
    pub fn new(order: usize, rhs_columns: usize, batches: usize) -> Result<Self, ServerError> {
        let op = Self {
            order: RocBlasGemm::dim(order)?,
            rhs_columns: RocBlasGemm::dim(rhs_columns)?,
            batches: RocBlasGemm::dim(batches)?,
        };
        op.extents()?;
        Ok(op)
    }
    pub fn extents(self) -> Result<[usize; 3], ServerError> {
        let extent = |a: i32, b: i32| {
            (a as usize)
                .checked_mul(b as usize)
                .and_then(|n| n.checked_mul(self.batches as usize))
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| failure("rocBLAS solve extent"))
        };
        Ok([
            extent(self.order, self.order)?,
            extent(self.order, self.rhs_columns)?,
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

type SgemmBatched = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    i64,
    *const f32,
    c_int,
    i64,
    *const f32,
    *mut f32,
    c_int,
    i64,
    c_int,
) -> c_int;

type StrsmBatched = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    i64,
    *mut f32,
    c_int,
    i64,
    c_int,
) -> c_int;

#[derive(Clone, Copy)]
pub(crate) struct Api {
    destroy: Destroy,
    stream: SetStream,
    workspace: SetWorkspace,
    gemm: Gemm,
    sgemm_batched: SgemmBatched,
    strsm_batched: StrsmBatched,
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
                sgemm_batched: symbol(&library, b"rocblas_sgemm_strided_batched\0")?,
                strsm_batched: symbol(&library, b"rocblas_strsm_strided_batched\0")?,
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
    unsafe fn bind(
        self,
        handle: *mut c_void,
        stream: cubecl_hip_sys::hipStream_t,
        workspace: *mut c_void,
    ) -> Result<(), ServerError> {
        unsafe {
            checked((self.stream)(handle, stream), "original stream")?;
            checked(
                (self.workspace)(handle, workspace, ROCBLAS_WORKSPACE_BYTES),
                "original workspace",
            )
        }
    }

    /// Native contiguous products; alpha1 beta0.
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
            self.bind(handle, stream, workspace)?;
            if op.bf16 {
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
            } else {
                let trans = |yes| if yes { 112 } else { 111 };
                let lda = if op.transpose_right {
                    op.columns
                } else {
                    op.rows
                };
                let ldb = if op.transpose_left {
                    op.requests
                } else {
                    op.columns
                };
                checked(
                    (self.sgemm_batched)(
                        handle,
                        trans(op.transpose_right),
                        trans(op.transpose_left),
                        op.rows,
                        op.requests,
                        op.columns,
                        &alpha,
                        weights.cast(),
                        lda,
                        i64::from(op.rows) * i64::from(op.columns),
                        input.cast(),
                        ldb,
                        i64::from(op.requests) * i64::from(op.columns),
                        &beta,
                        output.cast(),
                        op.rows,
                        i64::from(op.requests) * i64::from(op.rows),
                        op.batches,
                    ),
                    "sgemm_strided_batched",
                )
            }
        }
    }

    pub(crate) unsafe fn solve(
        self,
        handle: usize,
        stream: cubecl_hip_sys::hipStream_t,
        op: RocBlasUnitLowerSolve,
        ptrs: [*mut c_void; 3],
    ) -> Result<(), ServerError> {
        let handle = handle as *mut c_void;
        let [lower, rhs, workspace] = ptrs;
        let alpha = 1.0f32;
        // Row-major L*X=B becomes X^T*L^T=B^T: right, upper,
        // no transpose, unit diagonal in the column-major vendor ABI.
        unsafe {
            self.bind(handle, stream, workspace)?;
            checked(
                (self.strsm_batched)(
                    handle,
                    142,
                    121,
                    111,
                    132,
                    op.rhs_columns,
                    op.order,
                    &alpha,
                    lower.cast(),
                    op.order,
                    i64::from(op.order) * i64::from(op.order),
                    rhs.cast(),
                    op.rhs_columns,
                    i64::from(op.order) * i64::from(op.rhs_columns),
                    op.batches,
                ),
                "strsm_strided_batched",
            )
        }
    }
}
