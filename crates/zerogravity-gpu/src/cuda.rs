//! CUDA source operations on the existing physical runtime.
// Former modules: vllm_cuda_gdn, vllm_cuda_input_rms,
// vllm_cuda_qualification, vllm_ptx_inline.

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[allow(dead_code)] // Internal candidate; no production selection before physical qualification.
pub(crate) mod bf16_attention;
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[allow(dead_code)] // Internal execution candidate; physical qualification remains required.
pub(crate) mod bf16_gemm;
pub mod compiler;
#[cfg(feature = "cuda-native")]
pub mod gdn;
#[cfg(feature = "cuda-native")]
#[cfg_attr(not(feature = "cuda-native-qualification"), allow(dead_code))]
pub(crate) mod kernel;
#[cfg(feature = "cuda-native-qualification")]
pub mod native;
#[cfg(feature = "cuda-native")]
#[allow(dead_code)] // Internal candidate API; no production selector yet.
pub mod ordinary_text;
#[cfg(feature = "cuda-native-qualification")]
pub mod rms;

#[cfg(all(
    test,
    feature = "qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
pub(crate) mod test_support;
