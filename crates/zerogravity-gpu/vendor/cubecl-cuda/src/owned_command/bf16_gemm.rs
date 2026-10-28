//! Closed observed cuBLAS metadata on the original actor.
use super::OwnedCudaQueue;
use cubecl_core::server::{Handle, ServerError};
use sha2::{Digest, Sha256};
use std::fs::{File, Metadata};
use std::io::Read;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_operations_lower_to_identical_generic_contracts() {
        use ObservedBf16Gemm::*;
        for observed in [
            Qwen08Layer0QkvzDecode,
            Qwen08Layer0BaDecode,
            Qwen08Layer0QkvzPrefill,
            Qwen08Layer0BaPrefill,
            Qwen08Layer0OutProjectionDecode,
            Qwen08Layer0OutProjectionPrefill,
            Qwen08Layer0QkvzContinuedPrefill,
            Qwen08Layer0BaContinuedPrefill,
            Qwen08Layer0OutProjectionContinuedPrefill,
        ] {
            let operation = observed.operation();
            assert_eq!(
                (operation.m(), operation.n(), operation.k()),
                (observed.m(), observed.n(), observed.k())
            );
            assert_eq!(operation.api(), observed.observed_api());
            assert_eq!(operation.weight_bytes().unwrap(), observed.weight_bytes());
            assert_eq!(operation.input_bytes().unwrap(), observed.input_bytes());
            assert_eq!(operation.output_bytes().unwrap(), observed.output_bytes());
            assert_eq!(operation.workspace_bytes(), observed.workspace_bytes());
        }
    }

    #[test]
    fn generic_reference_geometry_is_dynamic_but_api_is_closed() {
        for rows in 1..=64 {
            let operation = Bf16GemmOperation::reference(8192, rows, 1024).unwrap();
            assert_eq!(operation.n(), rows);
            assert_eq!(operation.api(), REFERENCE_BF16_GEMM_API);
            assert_eq!(
                operation.sizes().unwrap(),
                [
                    8192 * 1024 * 2,
                    1024 * rows * 2,
                    8192 * rows * 2,
                    REFERENCE_BF16_GEMM_WORKSPACE_BYTES,
                ]
            );
        }
        assert!(Bf16GemmOperation::reference(0, 1, 1).is_err());
        assert!(Bf16GemmOperation::reference(1, 0, 1).is_err());
        assert!(Bf16GemmOperation::reference(1, 1, 0).is_err());
        assert!(Bf16GemmOperation::reference(i32::MAX as usize + 1, 1, 1).is_err());
        assert!(
            Bf16GemmOperation::new(1, 1, 1, 4, REFERENCE_BF16_GEMM_API).is_err()
        );
        let mut wrong = REFERENCE_BF16_GEMM_API;
        wrong.beta_bits = 1;
        assert!(
            Bf16GemmOperation::new(
                1,
                1,
                1,
                REFERENCE_BF16_GEMM_WORKSPACE_BYTES,
                wrong
            )
            .is_err()
        );
    }
}

#[cfg(all(
    feature = "owned-cublas-allocator",
    target_arch = "x86_64",
    target_env = "gnu"
))]
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "owned-cubin")]
mod captured_cubin;
#[cfg(feature = "owned-cubin")]
mod cubin_source;
#[cfg(feature = "owned-cubin")]
mod derived_cubin;
#[cfg(feature = "owned-cubin")]
pub use captured_cubin::{CapturedBf16CubinRecord, PinnedCapturedBf16Cubin};
#[cfg(feature = "owned-cubin")]
pub use derived_cubin::{PinnedBf16Cubin, PinnedBf16CubinDerivation};

/// Actual pinned layer0 calls; phase semantics remain the caller's contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservedBf16Gemm {
    Qwen08Layer0QkvzDecode,
    Qwen08Layer0BaDecode,
    Qwen08Layer0QkvzPrefill,
    Qwen08Layer0BaPrefill,
    Qwen08Layer0OutProjectionDecode,
    Qwen08Layer0OutProjectionPrefill,
    Qwen08Layer0QkvzContinuedPrefill,
    Qwen08Layer0BaContinuedPrefill,
    Qwen08Layer0OutProjectionContinuedPrefill,
}

impl ObservedBf16Gemm {
    pub const fn operation(self) -> Bf16GemmOperation {
        Bf16GemmOperation {
            m: self.m(),
            n: self.n(),
            k: self.k(),
            workspace_bytes: REFERENCE_BF16_GEMM_WORKSPACE_BYTES,
            api: REFERENCE_BF16_GEMM_API,
        }
    }

    pub const fn m(self) -> usize {
        match self {
            Self::Qwen08Layer0QkvzDecode
            | Self::Qwen08Layer0QkvzPrefill
            | Self::Qwen08Layer0QkvzContinuedPrefill => 8192,
            Self::Qwen08Layer0BaDecode
            | Self::Qwen08Layer0BaPrefill
            | Self::Qwen08Layer0BaContinuedPrefill => 32,
            _ => 1024,
        }
    }
    pub const fn n(self) -> usize {
        match self {
            Self::Qwen08Layer0QkvzDecode
            | Self::Qwen08Layer0BaDecode
            | Self::Qwen08Layer0OutProjectionDecode => 1,
            Self::Qwen08Layer0QkvzPrefill
            | Self::Qwen08Layer0BaPrefill
            | Self::Qwen08Layer0OutProjectionPrefill => 64,
            _ => 2,
        }
    }
    pub const fn k(self) -> usize {
        match self {
            Self::Qwen08Layer0OutProjectionDecode
            | Self::Qwen08Layer0OutProjectionPrefill
            | Self::Qwen08Layer0OutProjectionContinuedPrefill => 2048,
            _ => 1024,
        }
    }
    pub const fn weight_bytes(self) -> usize {
        self.m() * self.k() * 2
    }
    pub const fn input_bytes(self) -> usize {
        self.k() * self.n() * 2
    }
    pub const fn output_bytes(self) -> usize {
        self.m() * self.n() * 2
    }
    pub const fn workspace_bytes(self) -> usize {
        8_519_680
    }
    /// Public cublasGemmEx request only, not the hidden selected kernel.
    pub const fn observed_api(self) -> Bf16GemmApi {
        REFERENCE_BF16_GEMM_API
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16GemmApi {
    pub transpose_a: i32,
    pub transpose_b: i32,
    pub data_type: i32,
    pub compute_type: i32,
    pub requested_algorithm: i32,
    pub alpha_bits: u32,
    pub beta_bits: u32,
    pub pointer_mode: i32,
    pub math_mode: i32,
    pub atomics_mode: i32,
}

pub const REFERENCE_BF16_GEMM_API: Bf16GemmApi = Bf16GemmApi {
    transpose_a: 1,
    transpose_b: 0,
    data_type: 14,
    compute_type: 68,
    requested_algorithm: 99,
    alpha_bits: 0x3f80_0000,
    beta_bits: 0,
    pointer_mode: 0,
    math_mode: 0,
    atomics_mode: 0,
};
pub const REFERENCE_BF16_GEMM_WORKSPACE_BYTES: usize = 8_519_680;

/// Exact public GemmEx operation. Geometry is source/request derived; the
/// numerical API profile is closed and cannot be caller-modified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16GemmOperation {
    m: usize,
    n: usize,
    k: usize,
    workspace_bytes: usize,
    api: Bf16GemmApi,
}
impl From<ObservedBf16Gemm> for Bf16GemmOperation {
    fn from(observed: ObservedBf16Gemm) -> Self {
        observed.operation()
    }
}

impl Bf16GemmOperation {
    pub fn reference(m: usize, n: usize, k: usize) -> Result<Self, ServerError> {
        Self::new(
            m,
            n,
            k,
            REFERENCE_BF16_GEMM_WORKSPACE_BYTES,
            REFERENCE_BF16_GEMM_API,
        )
    }

    fn new(
        m: usize,
        n: usize,
        k: usize,
        workspace_bytes: usize,
        api: Bf16GemmApi,
    ) -> Result<Self, ServerError> {
        if m == 0
            || n == 0
            || k == 0
            || m > i32::MAX as usize
            || n > i32::MAX as usize
            || k > i32::MAX as usize
            || workspace_bytes != REFERENCE_BF16_GEMM_WORKSPACE_BYTES
            || api != REFERENCE_BF16_GEMM_API
        {
            return Err(failure("closed BF16 GemmEx geometry/API contract"));
        }
        let operation = Self {
            m,
            n,
            k,
            workspace_bytes,
            api,
        };
        operation.sizes()?;
        Ok(operation)
    }

    pub const fn m(self) -> usize {
        self.m
    }
    pub const fn n(self) -> usize {
        self.n
    }
    pub const fn k(self) -> usize {
        self.k
    }
    pub const fn workspace_bytes(self) -> usize {
        self.workspace_bytes
    }
    pub const fn api(self) -> Bf16GemmApi {
        self.api
    }
    pub fn weight_bytes(self) -> Result<usize, ServerError> {
        checked_bf16_bytes(self.m, self.k, "weights")
    }
    pub fn input_bytes(self) -> Result<usize, ServerError> {
        checked_bf16_bytes(self.k, self.n, "input")
    }
    pub fn output_bytes(self) -> Result<usize, ServerError> {
        checked_bf16_bytes(self.m, self.n, "output")
    }
    pub fn sizes(self) -> Result<[usize; 4], ServerError> {
        Ok([
            self.weight_bytes()?,
            self.input_bytes()?,
            self.output_bytes()?,
            self.workspace_bytes,
        ])
    }
}

fn checked_bf16_bytes(a: usize, b: usize, role: &str) -> Result<usize, ServerError> {
    a.checked_mul(b)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| failure(format!("BF16 GemmEx {role} extent overflow")))
}

/// Original handles only. Caller retains the physical ResidentLease owners.
pub struct Bf16GemmBindings {
    pub weights: Handle,
    pub input: Handle,
    pub output: Handle,
    pub workspace: Handle,
}
impl Bf16GemmBindings {
    pub(crate) fn handles(&self) -> [&Handle; 4] {
        [&self.weights, &self.input, &self.output, &self.workspace]
    }
}

const CUBLAS_HASH: &str = "d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048";
const LT_HASH: &str = "96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bf16BlasSourceIdentity {
    pub cublas_path: PathBuf,
    pub cublas_sha256: &'static str,
    pub cublas_lt_path: PathBuf,
    pub cublas_lt_sha256: &'static str,
}

struct PinnedFile {
    file: File,
    stamp: [u64; 7],
    path: PathBuf,
}
fn stamp(metadata: &Metadata) -> [u64; 7] {
    [
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime() as u64,
        metadata.mtime_nsec() as u64,
        metadata.ctime() as u64,
        metadata.ctime_nsec() as u64,
    ]
}
fn failure(message: impl Into<String>) -> ServerError {
    ServerError::graph_state(message)
}
impl PinnedFile {
    fn read(path: &Path, soname: &str, bytes: u64, expected: &str) -> Result<Self, ServerError> {
        if !path.is_absolute() || path.file_name().and_then(|name| name.to_str()) != Some(soname) {
            return Err(failure(
                "explicit pinned cuBLAS absolute DSO path/SONAME required",
            ));
        }
        let path = path
            .canonicalize()
            .map_err(|e| failure(format!("cuBLAS source path: {e}")))?;
        let mut file =
            File::open(&path).map_err(|e| failure(format!("cuBLAS source open: {e}")))?;
        let metadata = file
            .metadata()
            .map_err(|e| failure(format!("cuBLAS source metadata: {e}")))?;
        if !metadata.is_file() || metadata.len() != bytes {
            return Err(failure("pinned cuBLAS original DSO extent"));
        }
        let mut hash = Sha256::new();
        let mut buffer = [0_u8; 65536];
        let mut count = 0_u64;
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|e| failure(format!("cuBLAS source read: {e}")))?;
            if read == 0 {
                break;
            }
            count = count
                .checked_add(read as u64)
                .ok_or_else(|| failure("cuBLAS source extent overflow"))?;
            if count > bytes {
                return Err(failure("cuBLAS source grew during retained-FD read"));
            }
            hash.update(&buffer[..read]);
        }
        let after = file
            .metadata()
            .map_err(|e| failure(format!("cuBLAS source metadata: {e}")))?;
        if count != bytes
            || stamp(&metadata) != stamp(&after)
            || format!("{:x}", hash.finalize()) != expected
        {
            return Err(failure("pinned cuBLAS source SHA/EOF/retained-FD race"));
        }
        Ok(Self {
            file,
            stamp: stamp(&after),
            path,
        })
    }
    fn unchanged(&self) -> Result<(), ServerError> {
        let current = self
            .file
            .metadata()
            .map_err(|e| failure(format!("cuBLAS retained source metadata: {e}")))?;
        if stamp(&current) != self.stamp {
            return Err(failure("cuBLAS retained source changed"));
        }
        Ok(())
    }
}

/// Retains original FDs and actor identity, not loaded libraries or admission.
pub struct Bf16BlasSources {
    queue: Arc<OwnedCudaQueue>,
    cublas: PinnedFile,
    lt: PinnedFile,
}
impl Bf16BlasSources {
    #[cfg(all(
        feature = "owned-cublas-allocator",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    pub(crate) fn fd_paths(&self) -> [PathBuf; 2] {
        [
            PathBuf::from(format!("/proc/self/fd/{}", self.lt.file.as_raw_fd())),
            PathBuf::from(format!("/proc/self/fd/{}", self.cublas.file.as_raw_fd())),
        ]
    }
    #[cfg(all(
        feature = "owned-cublas-allocator",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    pub(crate) fn require_loaded_file(&self, index: usize, path: &Path) -> Result<(), ServerError> {
        let actual =
            std::fs::metadata(path).map_err(|e| failure(format!("loaded cuBLAS source: {e}")))?;
        let expected = if index == 0 { &self.lt } else { &self.cublas };
        if stamp(&actual) != expected.stamp {
            return Err(failure("loaded cuBLAS DSO differs from retained FD"));
        }
        expected.unchanged()
    }
    pub fn identity(&self) -> Bf16BlasSourceIdentity {
        Bf16BlasSourceIdentity {
            cublas_path: self.cublas.path.clone(),
            cublas_sha256: CUBLAS_HASH,
            cublas_lt_path: self.lt.path.clone(),
            cublas_lt_sha256: LT_HASH,
        }
    }
    pub(crate) fn require_queue(&self, queue: &Arc<OwnedCudaQueue>) -> Result<(), ServerError> {
        if !Arc::ptr_eq(&self.queue, queue) {
            return Err(failure("cuBLAS source belongs to another original queue"));
        }
        self.cublas.unchanged()?;
        self.lt.unchanged()
    }
}

/// No raw pointer/physical stream, loaded provider or Ready escapes inspection.
#[derive(Debug)]
pub struct Bf16GemmInspection {
    pub operation: Bf16GemmOperation,
    pub bytes: [usize; 4],
    pub provider_allocations_admitted: bool,
}

impl OwnedCudaQueue {
    /// Cold source-only read. This deliberately does NOT call dlopen/Create.
    pub fn verify_bf16_blas_sources(
        self: &Arc<Self>,
        cublas: &Path,
        cublas_lt: &Path,
    ) -> Result<Arc<Bf16BlasSources>, ServerError> {
        let cublas = PinnedFile::read(cublas, "libcublas.so.12", 105_140_976, CUBLAS_HASH)?;
        let lt = PinnedFile::read(cublas_lt, "libcublasLt.so.12", 749_205_904, LT_HASH)?;
        Ok(Arc::new(Bf16BlasSources {
            queue: self.clone(),
            cublas,
            lt,
        }))
    }
    /// Resolve actual original buffers inside the pinned actor; performs no GEMM.
    ///
    /// # Safety
    /// Caller has admitted/retained all original physical buffer leases and an
    /// initialization/submission guard before any binding can materialize.
    pub unsafe fn inspect_bf16_gemm(
        self: &Arc<Self>,
        sources: &Bf16BlasSources,
        operation: impl Into<Bf16GemmOperation>,
        bindings: Bf16GemmBindings,
    ) -> Result<Bf16GemmInspection, ServerError> {
        let operation = operation.into();
        sources.require_queue(self)?;
        let sizes = operation.sizes()?;
        if bindings
            .handles()
            .iter()
            .zip(sizes)
            .any(|(handle, bytes)| {
                handle.service != self.service_id() || handle.size() != bytes as u64
            })
        {
            return Err(failure("cuBLAS exact original service/binding extent"));
        }
        self.actor
            .submit_blocking(move |server| {
                server.inspect_owned_bf16_gemm(self.stream, operation, bindings)
            })
            .map_err(|e| failure(format!("original cuBLAS actor inspection: {e:?}")))?
    }
}

#[cfg(all(
    feature = "owned-cublas-allocator",
    target_arch = "x86_64",
    target_env = "gnu"
))]
mod provider;
#[cfg(all(
    feature = "owned-cublas-allocator",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub use provider::OwnedBf16BlasProvider;
