//! CUDA source operations on the existing physical runtime.
// Former modules: vllm_cuda_gdn, vllm_cuda_input_rms,
// vllm_cuda_qualification, vllm_ptx_inline.

pub mod compiler;
#[cfg(feature = "cuda-native")]
pub mod gdn;
#[cfg(feature = "cuda-native")]
#[cfg_attr(not(feature = "cuda-native-qualification"), allow(dead_code))]
pub(crate) mod kernel;
#[cfg(feature = "cuda-native-qualification")]
pub mod native;
#[cfg(feature = "cuda-native")]
pub mod ordinary_text;
#[cfg(feature = "cuda-native-qualification")]
pub mod rms;
