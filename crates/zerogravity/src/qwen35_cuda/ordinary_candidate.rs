//! Explicit qualification-only adapter for the ordinary Qwen CUDA path.
//!
//! This is deliberately not the public backend selector. It exists so the
//! integrated ordinary path can be qualified on real source/model/root data
//! before ordinary capability is allowed to succeed.

use super::ordinary_source;
use crate::{
    Error, Result,
    accelerator::program::SourceCallMode,
    gpu::{InferenceReadback, OrdinaryCudaProviderPaths, ResidentModel, ResidentSession},
    qwen35::plan::{OrdinaryQwenPlan, QwenSourceProfile},
    resource_arbiter::ResourceOwner,
};
use std::{
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

#[derive(Clone, Debug)]
pub struct SourceMetadata {
    pub family: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub index_sha256: &'static str,
    pub hidden_width: usize,
}

pub struct Model {
    plan: OrdinaryQwenPlan,
    resident: Arc<ResidentModel>,
    metadata: SourceMetadata,
}
impl Model {
    #[allow(clippy::too_many_arguments)]
    pub async fn load(
        device: crate::gpu::Device,
        hf_directory: &Path,
        context_capacity_tokens: usize,
        providers: OrdinaryCudaProviderPaths,
        host_mib: usize,
        provider_capacity_bytes: usize,
        device_capacity_bytes: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let host = ordinary_source::host_admission_mib(
            host_mib,
            ResourceOwner::new("qwen35-cuda-ordinary-candidate", "ordinary-source")?,
        )?;
        let opened = ordinary_source::open(hf_directory, &host, cancelled)?;
        let profile = opened.seal.model_profile();
        let plan = opened.seal.build_plan(context_capacity_tokens)?;
        if plan.model_profile() != profile
            || plan.source().source_identity() != opened.seal.source_identity()
            || plan.context_capacity_tokens() != context_capacity_tokens
        {
            return Err(Error::State(
                "ordinary Qwen candidate plan/source/profile differs".into(),
            ));
        }
        let resident = ResidentModel::load_ordinary_cuda(
            device,
            opened.seal.source().clone(),
            plan.plan().clone(),
            providers,
            provider_capacity_bytes,
            device_capacity_bytes,
            cancelled,
        )
        .await?;
        Ok(Arc::new(Self {
            plan,
            resident,
            metadata: SourceMetadata {
                family: opened.metadata.family,
                repository: opened.metadata.repository,
                revision: opened.metadata.revision,
                index_sha256: opened.metadata.index_sha256,
                hidden_width: opened.metadata.hidden_width,
            },
        }))
    }

    pub fn model_profile(&self) -> QwenSourceProfile {
        self.plan.model_profile()
    }

    pub fn context_capacity_tokens(&self) -> usize {
        self.plan.context_capacity_tokens()
    }

    pub fn source_metadata(&self) -> &SourceMetadata {
        &self.metadata
    }

    pub fn plan(&self) -> &OrdinaryQwenPlan {
        &self.plan
    }

    pub fn resident(&self) -> &Arc<ResidentModel> {
        &self.resident
    }

    pub async fn session(self: &Arc<Self>) -> Result<Session> {
        Ok(Session {
            model: self.clone(),
            resident: self.resident.session().await?,
        })
    }
}

pub struct Session {
    model: Arc<Model>,
    resident: ResidentSession,
}
impl Session {
    pub fn binding(&self) -> crate::session_root::RootBinding {
        self.resident.binding()
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub async fn forward(
        &mut self,
        mode: SourceCallMode,
        ids: &[u32],
        emit_logits: bool,
    ) -> Result<InferenceReadback> {
        if ids.is_empty() || ids.len() > self.model.plan.max_chunk_rows() {
            return Err(Error::Malformed(
                "ordinary Qwen candidate request row count".into(),
            ));
        }
        let base = self.resident.binding();
        if base.source() != self.model.plan.source().source_identity() {
            return Err(Error::State(
                "ordinary Qwen candidate session source differs".into(),
            ));
        }
        let invocation = self.model.plan.bind_invocation(
            mode,
            usize::try_from(base.frontier())
                .map_err(|_| Error::Limit("ordinary Qwen candidate frontier".into()))?,
            Arc::from(ids),
            emit_logits,
        )?;
        let request = invocation.materialize_source_plan()?;
        if emit_logits {
            self.resident.forward_ordinary_cuda(request).await
        } else {
            self.resident
                .forward_ordinary_cuda_state_only(request)
                .await
        }
    }
}
