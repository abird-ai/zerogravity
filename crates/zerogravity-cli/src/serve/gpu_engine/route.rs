//! Private source/session variants; every route uses the original worker.
use super::*;
use zerogravity::session_root::RootBinding;

pub(super) enum ModelEngine {
    Gguf(Arc<ResidentModel>),
    #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
    Canonical(Arc<zerogravity::canonical47::Canonical47Model>),
    #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
    ShortHf(Arc<zerogravity::short_hf::ShortHfModel>),
}
pub(super) enum RequestEngine {
    Gguf {
        resident: Arc<ResidentModel>,
        session: ResidentSession,
    },
    #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
    Canonical {
        model: Arc<zerogravity::canonical47::Canonical47Model>,
        session: zerogravity::gpu::vllm_cuda_qualification::Canonical47Session,
        selected: Option<u32>,
    },
    #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
    ShortHf {
        model: Arc<zerogravity::short_hf::ShortHfModel>,
        session: zerogravity::gpu::vllm_cuda_qualification::ShortHfSession,
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
            #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
            Self::Canonical(model) => Ok(RequestEngine::Canonical {
                model: model.clone(),
                session: block_on(model.resident().session(Some(cancelled)))
                    .map_err(|e| canonical::inference_error(e, model, cancelled))?,
                selected: None,
            }),
            #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
            Self::ShortHf(model) => Ok(RequestEngine::ShortHf {
                model: model.clone(),
                session: block_on(model.resident().session(Some(cancelled)))
                    .map_err(|e| short_hf::inference_error(e, model, cancelled))?,
                selected: None,
            }),
        }
    }
    #[cfg(all(test, feature = "cuda"))]
    pub(super) fn legacy_model(&self) -> Option<&Arc<ResidentModel>> {
        match self {
            Self::Gguf(m) => Some(m),
            #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
            Self::Canonical(_) => None,
            #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
            Self::ShortHf(_) => None,
        }
    }
    #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
    pub(super) fn validate_request(
        &self,
        tokenizer: &QwenTokenizer,
        request: &Request,
    ) -> Result<(), ServiceError> {
        if let Self::Canonical(model) = self {
            let ids = tokenizer
                .prompt_tokens(&request.input, request.chat)
                .map_err(ServiceError::inference)?;
            model
                .resident()
                .validate_prompt(&ids, request.max_tokens)
                .map_err(|e| ServiceError::new(400, "unsupported_source_request", e.to_string()))?;
        }
        #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
        if let Self::ShortHf(model) = self {
            let ids = tokenizer
                .prompt_tokens(&request.input, request.chat)
                .map_err(ServiceError::inference)?;
            model
                .resident()
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
            #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
            Self::Canonical { session, .. } => session.binding(),
            #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
            Self::ShortHf { session, .. } => session.binding(),
        }
    }
    pub(super) fn is_canonical(&self) -> bool {
        match self {
            Self::Gguf { .. } => false,
            #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
            Self::Canonical { .. } => true,
            #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
            Self::ShortHf { .. } => false,
        }
    }
    pub(super) fn is_original_hf(&self) -> bool {
        if self.is_canonical() {
            return true;
        }
        self.is_short_hf()
    }
    pub(super) fn is_short_hf(&self) -> bool {
        #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
        if matches!(self, Self::ShortHf { .. }) {
            return true;
        }
        false
    }
    pub(super) fn legacy_model(&self) -> Option<&Arc<ResidentModel>> {
        match self {
            Self::Gguf { resident, .. } => Some(resident),
            #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
            Self::Canonical { .. } => None,
            #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
            Self::ShortHf { .. } => None,
        }
    }
    pub(super) fn legacy(&self) -> Option<&ResidentSession> {
        match self {
            Self::Gguf { session, .. } => Some(session),
            #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
            Self::Canonical { .. } => None,
            #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
            Self::ShortHf { .. } => None,
        }
    }
    pub(super) fn legacy_mut(&mut self) -> Option<&mut ResidentSession> {
        match self {
            Self::Gguf { session, .. } => Some(session),
            #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
            Self::Canonical { .. } => None,
            #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
            Self::ShortHf { .. } => None,
        }
    }
    pub(super) fn presentation_id(
        &self,
        legacy: Option<&ResidentToken>,
    ) -> Result<u32, ServiceError> {
        match self {
            Self::Gguf { .. } => legacy.map(|t| t.id),
            #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
            Self::Canonical { selected, .. } => *selected,
            #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
            Self::ShortHf { selected, .. } => *selected,
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
