//! Private source/session variants; every route uses the original worker.
use super::*;
use zerogravity::session_root::RootBinding;

pub(super) enum ModelEngine {
    Gguf(Arc<ResidentModel>),
    #[cfg(all(
        feature = "qwen35-cuda-serving",
        feature = "qwen35-cuda-qualification",
        target_os = "linux"
    ))]
    FixedPrompt(Arc<zerogravity::qwen35_cuda::fixed_prompt::Model>),
    #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
    Qwen35Cuda(Arc<zerogravity::qwen35_cuda::Model>),
}
pub(super) enum RequestEngine {
    Gguf {
        resident: Arc<ResidentModel>,
        session: ResidentSession,
    },
    #[cfg(all(
        feature = "qwen35-cuda-serving",
        feature = "qwen35-cuda-qualification",
        target_os = "linux"
    ))]
    FixedPrompt {
        model: Arc<zerogravity::qwen35_cuda::fixed_prompt::Model>,
        session: zerogravity::qwen35_cuda::fixed_prompt::Session,
        selected: Option<u32>,
    },
    #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
    Qwen35Cuda {
        model: Arc<zerogravity::qwen35_cuda::Model>,
        session: zerogravity::qwen35_cuda::Session,
        selected: Option<u32>,
    },
}
impl ModelEngine {
    pub(super) fn session(&self, cancelled: &AtomicBool) -> Result<RequestEngine, ServiceError> {
        if cancelled.load(Ordering::Acquire) {
            return Err(ServiceError::new(499, "cancelled", "request cancelled"));
        }
        match self {
            Self::Gguf(resident) => Ok(RequestEngine::Gguf {
                resident: resident.clone(),
                session: block_on(resident.session()).map_err(ServiceError::inference)?,
            }),
            #[cfg(all(
                feature = "qwen35-cuda-serving",
                feature = "qwen35-cuda-qualification",
                target_os = "linux"
            ))]
            Self::FixedPrompt(model) => Ok(RequestEngine::FixedPrompt {
                model: model.clone(),
                session: block_on(model.session(Some(cancelled)))
                    .map_err(|e| fixed_prompt::inference_error(e, model, cancelled))?,
                selected: None,
            }),
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            Self::Qwen35Cuda(model) => Ok(RequestEngine::Qwen35Cuda {
                model: model.clone(),
                session: block_on(model.session(Some(cancelled)))
                    .map_err(|e| qwen35_cuda::inference_error(e, model, cancelled))?,
                selected: None,
            }),
        }
    }
    #[cfg(all(test, feature = "cuda"))]
    pub(super) fn legacy_model(&self) -> Option<&Arc<ResidentModel>> {
        match self {
            Self::Gguf(m) => Some(m),
            #[cfg(all(
                feature = "qwen35-cuda-serving",
                feature = "qwen35-cuda-qualification",
                target_os = "linux"
            ))]
            Self::FixedPrompt(_) => None,
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            Self::Qwen35Cuda(_) => None,
        }
    }
    pub(super) fn validate_request(
        &self,
        tokenizer: &QwenTokenizer,
        request: &Request,
    ) -> Result<(), ServiceError> {
        let _ = (tokenizer, request);
        #[cfg(all(
            feature = "qwen35-cuda-serving",
            feature = "qwen35-cuda-qualification",
            target_os = "linux"
        ))]
        if let Self::FixedPrompt(model) = self {
            let ids = tokenizer
                .prompt_tokens(&request.input, request.chat)
                .map_err(ServiceError::inference)?;
            model
                .validate_prompt(&ids, request.max_tokens)
                .map_err(|e| ServiceError::new(400, "unsupported_source_request", e.to_string()))?;
        }
        #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
        if let Self::Qwen35Cuda(model) = self {
            let ids = tokenizer
                .prompt_tokens(&request.input, request.chat)
                .map_err(ServiceError::inference)?;
            model
                .validate_prompt(&ids, request.max_tokens)
                .map_err(|e| ServiceError::new(400, "unsupported_source_request", e.to_string()))?;
        }
        Ok(())
    }
}
impl RequestEngine {
    pub(super) fn binding(&self) -> RootBinding {
        match self {
            Self::Gguf { session, .. } => session.binding(),
            #[cfg(all(
                feature = "qwen35-cuda-serving",
                feature = "qwen35-cuda-qualification",
                target_os = "linux"
            ))]
            Self::FixedPrompt { session, .. } => session.binding(),
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            Self::Qwen35Cuda { session, .. } => session.binding(),
        }
    }
    pub(super) fn is_canonical(&self) -> bool {
        match self {
            Self::Gguf { .. } => false,
            #[cfg(all(
                feature = "qwen35-cuda-serving",
                feature = "qwen35-cuda-qualification",
                target_os = "linux"
            ))]
            Self::FixedPrompt { .. } => true,
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            Self::Qwen35Cuda { .. } => false,
        }
    }
    pub(super) fn is_original_hf(&self) -> bool {
        if self.is_canonical() {
            return true;
        }
        self.is_qwen35_cuda()
    }
    pub(super) fn is_qwen35_cuda(&self) -> bool {
        #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
        if matches!(self, Self::Qwen35Cuda { .. }) {
            return true;
        }
        false
    }
    pub(super) fn legacy_model(&self) -> Option<&Arc<ResidentModel>> {
        match self {
            Self::Gguf { resident, .. } => Some(resident),
            #[cfg(all(
                feature = "qwen35-cuda-serving",
                feature = "qwen35-cuda-qualification",
                target_os = "linux"
            ))]
            Self::FixedPrompt { .. } => None,
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            Self::Qwen35Cuda { .. } => None,
        }
    }
    pub(super) fn legacy(&self) -> Option<&ResidentSession> {
        match self {
            Self::Gguf { session, .. } => Some(session),
            #[cfg(all(
                feature = "qwen35-cuda-serving",
                feature = "qwen35-cuda-qualification",
                target_os = "linux"
            ))]
            Self::FixedPrompt { .. } => None,
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            Self::Qwen35Cuda { .. } => None,
        }
    }
    pub(super) fn legacy_mut(&mut self) -> Option<&mut ResidentSession> {
        match self {
            Self::Gguf { session, .. } => Some(session),
            #[cfg(all(
                feature = "qwen35-cuda-serving",
                feature = "qwen35-cuda-qualification",
                target_os = "linux"
            ))]
            Self::FixedPrompt { .. } => None,
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            Self::Qwen35Cuda { .. } => None,
        }
    }
    pub(super) fn presentation_id(
        &self,
        legacy: Option<&ResidentToken>,
    ) -> Result<u32, ServiceError> {
        match self {
            Self::Gguf { .. } => legacy.map(|t| t.id),
            #[cfg(all(
                feature = "qwen35-cuda-serving",
                feature = "qwen35-cuda-qualification",
                target_os = "linux"
            ))]
            Self::FixedPrompt { selected, .. } => *selected,
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            Self::Qwen35Cuda { selected, .. } => *selected,
        }
        .ok_or_else(|| {
            ServiceError::new(
                500,
                "missing_device_token",
                "computed committed packet absent",
            )
        })
    }
}
