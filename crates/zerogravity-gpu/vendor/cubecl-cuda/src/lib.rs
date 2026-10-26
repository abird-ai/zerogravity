#[macro_use]
extern crate derive_new;
extern crate alloc;

pub mod compiler;
mod compute;
mod device;
mod observation;
pub use observation::{CudaDeviceMetadata, observe_cuda_devices};
#[cfg(all(
    feature = "owned-cublas-allocator",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
mod owned_blas_allocator;
mod owned_command;
#[cfg(all(
    feature = "owned-cublas-allocator",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub use owned_blas_allocator::{BlasAllocationSnapshot, OwnedBlasAllocationPool};
mod runtime;

pub use compiler::{CudaCompilationOptions, CudaCompiler, CudaRepresentation};
pub use device::*;
pub use owned_command::OwnedCudaQueue;
#[cfg(target_os = "linux")]
pub use owned_command::SourceWeightLayout;
#[cfg(target_os = "linux")]
pub use owned_command::{
    Bf16BlasSourceIdentity, Bf16BlasSources, Bf16GemmApi, Bf16GemmBindings, Bf16GemmInspection,
    ObservedBf16Gemm,
};
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub use owned_command::{
    Bf16VocabularyHead, CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes,
    CubinParameter, CubinPointerFixup, CubinProvider, CubinWorkspaceClear, CubinWorkspaceView,
    OwnedCudaBinding, OwnedCudaKernel, OwnedCudaModuleSet, OwnedCudaTransferRetirement,
    VerifiedCubin, VocabularyHeadSelection,
};
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub use owned_command::{
    CapturedBf16CubinRecord, PinnedBf16Cubin, PinnedBf16CubinDerivation, PinnedCapturedBf16Cubin,
};
pub use runtime::*;

pub mod install {
    use std::path::PathBuf;

    pub fn include_path() -> PathBuf {
        let mut path = cuda_path().expect("
        CUDA installation not found.
        Please ensure that CUDA is installed and the CUDA_PATH environment variable is set correctly.
        Note: Default paths are used for Linux (/usr/local/cuda) and Windows (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/), which may not be correct.
    ");
        path.push("include");
        path
    }

    pub fn cccl_include_path() -> PathBuf {
        let mut path = include_path();
        path.push("cccl");
        path
    }

    pub fn cuda_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("CUDA_PATH") {
            return Some(PathBuf::from(path));
        }

        #[cfg(target_os = "linux")]
        {
            // If it is installed as part of the distribution
            return if std::fs::exists("/usr/local/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/usr/local/cuda"))
            } else if std::fs::exists("/opt/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/opt/cuda"))
            } else if std::fs::exists("/usr/bin/nvcc").is_ok_and(|exists| exists) {
                // Maybe the compiler was installed within the user path.
                Some(PathBuf::from("/usr"))
            } else {
                None
            };
        }

        #[cfg(target_os = "windows")]
        {
            return Some(PathBuf::from(
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/",
            ));
        }

        #[allow(unreachable_code)]
        None
    }
}

#[cfg(all(
    feature = "owned-cublas-allocator",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub use owned_command::OwnedBf16BlasProvider;

#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub use owned_command::{Bf16MatrixLayout, TextEmbeddingWidth};

pub use compute::context::{CompilerContextUsage, CudaCompilerContextPolicy};

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::CudaRuntime;

    pub use half::bf16;
    pub use half::f16;

    // The complex types are the C++ backend's: the LLVM backend does not lower them, so it does
    // not advertise them (see `restrict_features`) and the tests for them are not generated.
    // `testgen_complex_validation` runs either way -- it asserts that an unadvertised complex
    // operation is refused, which is what the LLVM backend must do and what the C++ backend
    // never reaches.
    cubecl_core::testgen_all!(f32: [f16, bf16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);

    cubecl_core::testgen_launch_dynamic_count!();
    cubecl_std::testgen!();

    cubecl_std::testgen_tensor_identity!([f16, bf16, f32, u32]);

    cubecl_std::testgen_quantized_view!(f16);

    #[cfg(feature = "cpp")]
    cubecl_core::testgen_complex_core!();
    #[cfg(feature = "cpp")]
    cubecl_core::testgen_complex_compare!();
    #[cfg(feature = "cpp")]
    cubecl_core::testgen_complex_math!();
    cubecl_core::testgen_complex_validation!();
    cubecl_core::testgen_profiling!();
    cubecl_core::testgen_profiling_nested!();
}
