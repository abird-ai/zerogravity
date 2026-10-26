//! CUDA SDK metadata checks; no client/context creation or execution permit.
use zerogravity_core::{Error, Result};

/// Refuse invalid SDK metadata before intentional CUDA Client bootstrap.
// Previous function: validate_cuda_compiler_environment.
pub fn validate_environment() -> Result<()> {
    cubecl::cuda::CudaCompilerContextPolicy::validate_sdk_configuration()
        .map_err(|s| Error::Unsupported(s.into()))
}
