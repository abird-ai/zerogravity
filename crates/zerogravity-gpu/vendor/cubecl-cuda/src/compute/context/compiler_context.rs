//! Finite source-neutral compiler policy on the original CUDA context.
use std::{any::Any, sync::Arc};

use super::compiler_limits::CompilerContextUsage;

#[derive(Debug, PartialEq, Eq)]
struct SdkConfiguration {
    include: std::path::PathBuf,
    canonical: std::path::PathBuf,
    cccl: bool,
}
impl SdkConfiguration {
    fn capture() -> Result<Self, &'static str> {
        let root = crate::install::cuda_path().ok_or("admitted CUDA SDK path absent")?;
        if root.as_os_str().as_encoded_bytes().len() > 4096 {
            return Err("admitted CUDA SDK path bound");
        }
        let include = root.join("include");
        let canonical = include
            .canonicalize()
            .map_err(|_| "admitted CUDA include path unavailable")?;
        if canonical.as_os_str().as_encoded_bytes().len() > 4096 {
            return Err("admitted CUDA canonical include path bound");
        }
        let cccl = include.join("cccl").exists();
        Ok(Self {
            include,
            canonical,
            cccl,
        })
    }
}

pub(crate) struct CompilerContextState {
    pub(crate) owner: Arc<dyn Any + Send + Sync>,
    generation: u32,
    sdk: SdkConfiguration,
    pub(crate) usage: CompilerContextUsage,
}
impl core::fmt::Debug for CompilerContextState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CompilerContextState")
            .field("generation", &self.generation)
            .field("usage", &self.usage)
            .finish()
    }
}
impl CompilerContextState {
    pub(crate) fn new(owner: Arc<dyn Any + Send + Sync>) -> Result<Self, &'static str> {
        Ok(Self {
            owner,
            generation: cubecl_environment::environment::generation(),
            sdk: SdkConfiguration::capture()?,
            usage: CompilerContextUsage::default(),
        })
    }
    pub(crate) fn check_environment(&self) -> Result<(), &'static str> {
        if cubecl_environment::environment::generation() != self.generation {
            Err("admitted CUDA compiler environment changed; no cache eviction/reload credit")
        } else {
            Ok(())
        }
    }
    pub(super) fn include_paths(&self) -> (std::path::PathBuf, Option<std::path::PathBuf>) {
        (
            self.sdk.include.clone(),
            self.sdk.cccl.then(|| self.sdk.include.join("cccl")),
        )
    }
    pub(crate) fn begin(&mut self) -> Result<(), &'static str> {
        self.check_environment()?;
        if SdkConfiguration::capture()? != self.sdk {
            return Err("admitted CUDA include/options configuration changed");
        }
        self.usage.begin()
    }
    pub(crate) fn source(&mut self, bytes: usize) -> Result<(), &'static str> {
        self.usage.source(bytes)
    }
    pub(crate) fn ptx(&mut self, bytes: usize) -> Result<(), &'static str> {
        self.usage.ptx(bytes)
    }
}

impl super::CudaContext {
    pub(super) fn compiler_limit(
        &self,
        reason: impl Into<String>,
    ) -> cubecl_server::compiler::CompilationError {
        cubecl_server::compiler::CompilationError::Generic {
            reason: reason.into(),
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        }
    }
    pub(crate) fn compiler_context_usage(&self) -> Option<super::CompilerContextUsage> {
        self.compiler_context.as_ref().map(|c| c.usage)
    }
    pub(crate) fn compiler_context_owner(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.compiler_context.as_ref().map(|c| c.owner.clone())
    }
    pub(crate) fn require_compiler_context(
        &self,
    ) -> Result<(), cubecl_server::compiler::CompilationError> {
        self.check_compiler()?;
        self.compiler_context
            .as_ref()
            .ok_or_else(|| {
                self.compiler_limit("CUDA compile-only requires original admitted context")
            })?
            .check_environment()
            .map_err(|e| self.compiler_limit(e))
    }
    pub(crate) fn install_compiler_context(
        &mut self,
        owner: Arc<dyn Any + Send + Sync>,
        adopted: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), cubecl_server::compiler::CompilationError> {
        self.check_compiler()?;
        if self.compiler_context.is_some()
            || self.compilation_attempts != 0
            || self.ptx_cache.is_some()
            || self.second_line_ptx_cache.is_some()
            || self.arch.version != 86
            || !matches!(
                crate::compiler::CudaBackend::default(),
                crate::compiler::CudaBackend::Cpp
            )
        {
            return Err(self.compiler_limit(
                "CUDA finite compiler policy requires fresh SM86 context, persistence disabled",
            ));
        }
        let context = CompilerContextState::new(owner).map_err(|e| self.compiler_limit(e))?;
        adopted.store(true, std::sync::atomic::Ordering::Release);
        self.compiler_context = Some(context);
        Ok(())
    }
}

impl super::CudaCompilerContextPolicy {
    /// Same pinned SDK-path observation, before any Client/context/compiler
    /// effect. The actor still captures and revalidates its own environment.
    pub fn validate_sdk_configuration() -> Result<(), &'static str> {
        let sdk = SdkConfiguration::capture()?;
        for header in ["cuda_runtime.h", "cuda_fp16.h"] {
            if !sdk.include.join(header).is_file() {
                return Err("CUDA SDK required runtime/FP16 header absent");
            }
        }
        if (sdk.cccl && !sdk.include.join("cccl/cuda/std").is_dir())
            || (!sdk.cccl && !sdk.include.join("cuda/std").is_dir())
        {
            return Err("CUDA SDK CCCL headers absent");
        }
        Ok(())
    }
}
