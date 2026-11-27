use super::{
    TextTokenizer,
    local::{AdmittedGguf, SourceDeclarations},
};
use crate::qwen35::{LoadedQwenModel, QwenDenseCpu, StateSpec};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use zerogravity_engine::model_instance::{LogitModel, PreparedModel, TextSession};
use zerogravity_engine::router::{Router, ServiceError, pending::ServingHost, validate_config};

#[path = "budget.rs"]
mod budget;
#[path = "load.rs"]
mod load;
use budget::session_budget;

struct Model {
    weights: Arc<AdmittedGguf>,
    tokenizer: Arc<dyn TextTokenizer>,
    max_context: usize,
    max_session_bytes: u64,
    max_token_bytes: usize,
}
struct Session {
    session: QwenDenseCpu,
    // Retire actual session readers before the original immutable-source lease.
    _source: Arc<Model>,
}
impl TextSession for Session {
    fn prefill_chunk_without_logits(&mut self, tokens: &[u32]) -> crate::Result<()> {
        self.session.prefill_chunk_without_logits(tokens)
    }
    fn prefill_transaction(&mut self, tokens: &[u32]) -> crate::Result<Vec<f32>> {
        self.session.prefill_transaction(tokens)
    }
    fn forward(&mut self, token: u32) -> crate::Result<Vec<f32>> {
        self.session.forward(token)
    }
}
impl PreparedModel for Model {
    fn source(&self) -> crate::identity::SourceIdentity {
        self.weights.model().source_identity()
    }
    fn tokenizer(&self) -> Arc<dyn TextTokenizer> {
        self.tokenizer.clone()
    }
    fn max_context(&self) -> usize {
        self.max_context
    }

    fn max_token_bytes(&self) -> usize {
        self.max_token_bytes
    }
    fn vocabulary_size(&self) -> u64 {
        self.weights.model().geometry().vocab_size
    }
    fn public_realtime_text(&self) -> bool {
        true
    }
    fn describe(&self) -> crate::Result<zerogravity_engine::diagnostics::ModelDetails> {
        use zerogravity_core::provider::{
            ExecutionDescription, ExecutionMode, ProviderKind, SelectedMode,
        };
        use zerogravity_engine::diagnostics::{LayerDevice, LayerPlacement, ModelDetails};
        Ok(ModelDetails {
            artifact: Some(super::local::artifact_description(
                self.weights.cache_snapshot().map(str::to_owned),
            )),
            execution: Some(ExecutionDescription {
                selected: SelectedMode {
                    provider: ProviderKind::Cpu,
                    mode: ExecutionMode::Native,
                },
                // This direct CPU adapter has no explicit prepared ForwardPlan.
                numerical_profile: None,
                text_calls: None,
                program: None,
                region: None,
            }),
            layer_placement: LayerPlacement::single(
                self.weights.model().geometry().layers.len(),
                LayerDevice::HostCpu,
            ),
            prefix_sharing: Some(false),
            source_host_reserved_bytes: Some(self.weights.host_reserved_bytes()?),
            ..Default::default()
        })
    }
}
impl LogitModel for Model {
    fn max_session_bytes(&self) -> u64 {
        self.max_session_bytes
    }
    fn session_scratch_bytes(&self) -> u64 {
        // Original projection credit plus the now-boxed session's allocation.
        crate::cpu::runtime::RAYON_PROJECTION_INJECTOR_BYTES
            + std::mem::size_of::<Session>() as u64
            + 64
    }
    fn new_session(self: Arc<Self>) -> crate::Result<Box<dyn TextSession>> {
        Ok(Box::new(Session {
            session: self.weights.model().new_session()?,
            _source: self,
        }))
    }
}

/// Load exact local Qwen sources, sharing canonical paths across aliases, then
/// transfer their admitted owners to the model-neutral scheduler.
pub fn load_router(
    declared: Vec<(String, PathBuf)>,
    workers: usize,
    active_per_worker: usize,
    max_context: usize,
    max_output: usize,
    host_mib: usize,
    max_pending: Option<usize>,
) -> Result<Router, ServiceError> {
    load_router_declared(
        SourceDeclarations::Paths(declared),
        crate::runtime::TextConfig {
            workers,
            active_per_worker,
            max_context,
            max_output,
            host_mib,
            max_pending,
        },
    )
}
pub(crate) fn load_router_declared(
    declared: SourceDeclarations<'_>,
    config: crate::runtime::TextConfig,
) -> Result<Router, ServiceError> {
    let crate::runtime::TextConfig {
        workers,
        active_per_worker,
        max_context,
        max_output,
        host_mib,
        max_pending,
    } = config;
    validate_config(
        declared.len(),
        workers,
        active_per_worker,
        max_context,
        max_output,
        max_pending,
    )?;
    let declared = declared.resolve()?;
    let host = ServingHost::new(host_mib)?;
    let models = load::load_models(declared, &host, max_context)?;
    // The same process-lifetime Rayon owner is admitted before any worker.
    let alias = models
        .keys()
        .next()
        .ok_or_else(|| ServiceError::new(400, "invalid_model", "no model routes"))?;
    let admission = host.transfer_admission(alias, "global-rayon")?;
    crate::cpu::runtime::admit_global_rayon(&admission)?;
    Router::new(
        models,
        workers,
        active_per_worker,
        max_context,
        max_output,
        host,
        max_pending,
    )
}
