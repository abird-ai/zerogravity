//! Cold serving-source selection. Captured profiles are explicit diagnostics;
//! their original finite guards remain distinct from ordinary GGUF capability.
#[cfg(all(
    feature = "text",
    any(feature = "vulkan", feature = "cuda", feature = "hip")
))]
use crate::devices::GpuBackend;
use crate::runtime::text::{Router, ServiceError};
use std::path::PathBuf;

/// Source/scheduler policy only; no environment, credentials, signals or bind IO.
#[derive(Clone, Debug)]
pub struct TextProfile {
    pub workers: usize,
    pub active_per_worker: usize,
    pub max_context: usize,
    pub max_output: usize,
    pub host_mib: usize,
    pub backend: String,
    pub source_profile: String,
    pub native_artifacts: Option<PathBuf>,
    pub provider_mib: usize,
    pub device_mib: usize,
    pub batch_mode: String,
    pub batch_rows: usize,
    pub native_batch_rows: usize,
    pub max_pending_requests: Option<usize>,
}
impl TextProfile {
    /// Cold validation. A valid description is not source or execution authority.
    pub fn validate(&self) -> Result<(), ServiceError> {
        if self.max_pending_requests == Some(0) {
            return Err(ServiceError::new(
                400,
                "invalid_pending_limit",
                "--max-pending-requests must be a positive count or none",
            ));
        }
        validate_batch_mode(
            &self.batch_mode,
            &self.backend,
            self.active_per_worker,
            self.batch_rows,
        )?;
        validate_source_profile(self)
    }
    pub(crate) fn load(self, declared: Vec<(String, PathBuf)>) -> Result<Router, ServiceError> {
        let options = self;
        let backend = options.backend.as_str();
        match backend {
            "cpu" => {
                #[cfg(all(feature = "cpu", feature = "text", feature = "cache"))]
                {
                    crate::builtins::text::load_router(
                        declared,
                        options.workers,
                        options.active_per_worker,
                        options.max_context,
                        options.max_output,
                        options.host_mib,
                        options.max_pending_requests,
                    )
                }
                #[cfg(not(all(feature = "cpu", feature = "text", feature = "cache")))]
                {
                    let _ = declared;
                    Err(ServiceError::new(
                        400,
                        "backend_not_compiled",
                        "CPU text loading requires the cpu, text and cache features",
                    ))
                }
            }
            "vulkan-nvidia" | "vulkan-amd" | "cuda" | "rocm" => {
                #[cfg(all(
                    feature = "text",
                    any(feature = "vulkan", feature = "cuda", feature = "hip")
                ))]
                {
                    let selected = match backend {
                        "vulkan-nvidia" => GpuBackend::VulkanNvidia,
                        "vulkan-amd" => GpuBackend::VulkanAmd,
                        "cuda" => GpuBackend::Cuda,
                        _ => GpuBackend::Hip,
                    };
                    if !selected.compiled() {
                        return Err(ServiceError::new(
                            400,
                            "backend_not_compiled",
                            "selected GPU backend was not compiled into this binary",
                        ));
                    }
                    let profile = crate::builtins::text::GpuProfile {
                        workers: options.workers,
                        active: options.active_per_worker,
                        context: options.max_context,
                        output: options.max_output,
                        provider_mib: options.provider_mib,
                        device_mib: options.device_mib,
                        cohort_width: if options.source_profile == "qwen35-cuda-bf16" {
                            options.native_batch_rows
                        } else if options.batch_mode == "fused" {
                            options.batch_rows
                        } else {
                            0
                        },
                        host_mib: options.host_mib,
                        max_pending: options.max_pending_requests,
                    };
                    if options.source_profile == "fixed-prompt-bf16" {
                        #[cfg(all(
                            feature = "qwen35-cuda-serving",
                            feature = "qwen35-cuda-qualification",
                            target_os = "linux"
                        ))]
                        {
                            crate::diagnostics::text_gpu::load_fixed_prompt(
                                declared,
                                selected,
                                profile,
                                options.host_mib,
                                options.native_artifacts.as_deref(),
                            )
                        }
                        #[cfg(not(all(
                            feature = "qwen35-cuda-serving",
                            feature = "qwen35-cuda-qualification",
                            target_os = "linux"
                        )))]
                        {
                            Err(ServiceError::new(
                                400,
                                "backend_not_compiled",
                                "fixed-prompt serving requires explicit qwen35-cuda-qualification",
                            ))
                        }
                    } else if options.source_profile == "qwen35-cuda-bf16" {
                        #[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
                        {
                            crate::diagnostics::text_gpu::load_qwen35_cuda(
                                declared,
                                selected,
                                profile,
                                options.host_mib,
                                options.native_artifacts.as_deref(),
                            )
                        }
                        #[cfg(not(all(feature = "qwen35-cuda-serving", target_os = "linux")))]
                        {
                            Err(ServiceError::new(
                                400,
                                "backend_not_compiled",
                                "qwen35-cuda-serving feature required",
                            ))
                        }
                    } else {
                        crate::builtins::text::load_gpu_router(declared, selected, profile)
                    }
                }
                #[cfg(not(all(
                    feature = "text",
                    any(feature = "vulkan", feature = "cuda", feature = "hip")
                )))]
                {
                    Err(ServiceError::new(
                        400,
                        "backend_not_compiled",
                        "compile with --features serve,vulkan or serve,cuda or serve,hip",
                    ))
                }
            }
            _ => Err(ServiceError::new(
                400,
                "invalid_backend",
                "backend must be cpu, vulkan-nvidia, vulkan-amd, cuda or rocm",
            )),
        }
    }
}

fn validate_batch_mode(
    mode: &str,
    backend: &str,
    active: usize,
    rows: usize,
) -> Result<(), ServiceError> {
    match mode {
        "interleaved" => Ok(()),
        "fused" if backend == "cuda" && matches!(rows, 2 | 4 | 8) && active >= rows => Ok(()),
        _ => Err(ServiceError::new(
            400,
            "unsupported_batch_mode",
            "use interleaved, or fused with CUDA, rows=2/4/8 and active>=rows",
        )),
    }
}

fn fixed_prompt_owner_profile(active: usize, host: usize, provider: usize) -> bool {
    matches!(
        (provider, host, active),
        (2304, 3072..=16384, 1..=2) | (4096, 4096, 1..=8)
    )
}

fn qwen35_cuda_host_profile(host_mib: usize) -> bool {
    matches!(host_mib, 4096 | 8192)
}

fn validate_source_profile(options: &TextProfile) -> Result<(), ServiceError> {
    if options.native_batch_rows != 0
        && (options.source_profile != "qwen35-cuda-bf16"
            || options.backend != "cuda"
            || options.batch_mode != "interleaved"
            || options.native_batch_rows != 4
            || options.active_per_worker < 4)
    {
        return Err(ServiceError::new(
            400,
            "unsupported_native_batch_profile",
            "native-batch-rows=4 requires qwen35-cuda-bf16/CUDA/interleaved/active>=4; default0",
        ));
    }
    match options.source_profile.as_str() {
        "gguf" if options.native_artifacts.is_none() => Ok(()),
        "fixed-prompt-bf16" => {
            if options.backend != "cuda"
                || options.workers != 1
                || options.max_context != 512
                || !(1..=8).contains(&options.max_output)
                || options.batch_mode != "interleaved"
                || options.device_mib != 5120
                || !fixed_prompt_owner_profile(
                    options.active_per_worker,
                    options.host_mib,
                    options.provider_mib,
                )
            {
                return Err(ServiceError::new(
                    400,
                    "unsupported_fixed-prompt_profile",
                    "fixed-prompt requires CUDA/1worker/context512/maxoutput1..8/interleaved/device5120 and finite2 or exact4GiB active8 owner profile",
                ));
            }
            if !cfg!(all(
                feature = "qwen35-cuda-serving",
                feature = "qwen35-cuda-qualification",
                target_os = "linux"
            )) {
                return Err(ServiceError::new(
                    400,
                    "backend_not_compiled",
                    "fixed-prompt serving requires explicit qwen35-cuda-qualification",
                ));
            }
            Ok(())
        }
        "qwen35-cuda-bf16" => {
            if options.backend != "cuda"
                || options.workers != 1
                || options.max_context != 512
                || !(1..=8).contains(&options.max_output)
                || !(1..=8).contains(&options.active_per_worker)
                || options.batch_mode != "interleaved"
                || !qwen35_cuda_host_profile(options.host_mib)
                || options.provider_mib == 0
                || options.provider_mib > options.device_mib
                || options.device_mib != 5120
                || options.native_artifacts.is_none()
            {
                return Err(ServiceError::new(
                    400,
                    "unsupported_qwen35_cuda_profile",
                    "Qwen CUDA requires CUDA/1worker/context512/maxoutput1..8/active1..8/interleaved/4096or8192host/provider1..5120/5120device and actual native-artifacts",
                ));
            }
            if !cfg!(all(feature = "qwen35-cuda-serving", target_os = "linux")) {
                return Err(ServiceError::new(
                    400,
                    "backend_not_compiled",
                    "compile Linux qwen35-cuda-serving",
                ));
            }
            Ok(())
        }
        _ => Err(ServiceError::new(
            400,
            "unsupported_source_profile",
            "source-profile is gguf, fixed-prompt-bf16 or qwen35-cuda-bf16; native-artifacts is HF-only",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_eight_profile_needs_all_three_exact_owner_settings() {
        for active in 1..=8 {
            assert!(super::fixed_prompt_owner_profile(active, 4096, 4096));
        }
        for (active, host, provider) in [
            (0, 4096, 4096),
            (9, 4096, 4096),
            (8, 3072, 4096),
            (8, 4095, 4096),
            (8, 4097, 4096),
            (8, 4096, 2304),
            (8, 4096, 4095),
            (8, 4096, 4097),
            (3, 4096, 2304),
        ] {
            assert!(!super::fixed_prompt_owner_profile(active, host, provider));
        }
        assert!(super::fixed_prompt_owner_profile(1, 3072, 2304));
        assert!(super::fixed_prompt_owner_profile(2, 16384, 2304));
    }

    #[test]
    fn original_batch_modes_refuse_without_source_or_device_access() {
        for backend in ["cpu", "cuda", "vulkan-nvidia", "vulkan-amd", "rocm"] {
            assert!(validate_batch_mode("interleaved", backend, 1, 2).is_ok());
            if backend != "cuda" {
                assert_eq!(
                    validate_batch_mode("fused", backend, 8, 4)
                        .unwrap_err()
                        .code,
                    "unsupported_batch_mode"
                );
            }
        }
        for rows in [2, 4, 8] {
            assert!(validate_batch_mode("fused", "cuda", rows, rows).is_ok());
        }
        for rows in [0, 1, 3, 5, 6, 7, 9, 16] {
            assert!(validate_batch_mode("fused", "cuda", 8, rows).is_err());
        }
        assert!(validate_batch_mode("fused", "cuda", 4, 8).is_err());
        for mode in ["fused", "fused-b4", "unknown"] {
            assert!(validate_batch_mode(mode, "cuda", 1, 2).is_err());
        }
    }
}
