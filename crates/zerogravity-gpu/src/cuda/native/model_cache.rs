//! Defined native ABI scratch per original attempt/layer, never a StateRoot.
use crate::resident_inference::native_fragment::model_state::BoundModelState;
use crate::resident_leases::{ResidentLease, ResidentService};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::SourceCallMode;
use zerogravity_core::device_events::BufferRole;
use zerogravity_core::session_root::{RootAttempt, RootBinding};
use zerogravity_core::{Error, Result};

const BYTES: usize = 5_570_560;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum History {
    Reference { frontier: usize },
    FixedPromptPrefix,
    FixedPromptAttentionKv,
    FixedPromptWhole,
}
struct ZeroSource {
    bytes: Vec<u8>,
    _source: VerifiedTensorSource,
    // Original bytes, client H2D copy and bounded staging/metadata admission.
    _host: WindowWorkspace,
}
pub(crate) struct ModelCache {
    lease: ResidentLease,
    source: VerifiedTensorSource,
    base: RootBinding,
    layer: usize,
    history: History,
    fixed_prompt: Option<Arc<crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding>>,
    initial: Arc<BoundModelState>,
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(
            "full24 defined native cache initialization cancelled".into(),
        ));
    }
    Ok(())
}
impl ModelCache {
    /// Caller holds its SAME original fair preparation slot; one bounded zero
    /// window is retired before the next cache/source-W allocation. No native
    /// module is active while these ordinary initialized resources are created.
    pub(crate) async fn prepare(
        source: VerifiedTensorSource,
        service: Arc<ResidentService>,
        attempt: &RootAttempt,
        initial: Arc<BoundModelState>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let first = initial.call();
        let history = match (
            first.mode(),
            first.rows().base(),
            first.rows().len(),
            attempt.frontier(),
        ) {
            (SourceCallMode::Prefill, 0, 1, 2) => History::Reference { frontier: 2 },
            (SourceCallMode::Prefill, 0, 64, 67) => History::Reference { frontier: 67 },
            _ => {
                return Err(Error::Unsupported(
                    "closed original full-model native-cache request histories".into(),
                ));
            }
        };
        Self::prepare_checked(source, service, attempt, initial, history, None, cancelled).await
    }
    pub(crate) async fn prepare_fixed_prompt_prefix(
        scope: Arc<crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding>,
        service: Arc<ResidentService>,
        attempt: &RootAttempt,
        initial: Arc<BoundModelState>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        initial.require_fixed_prompt_model_prefix(&scope)?;
        if scope.kind() != crate::cuda::native::fixed_prompt_model::FixedPromptCallKind::Prefill
            || attempt.base().frontier() != 0
            || attempt.frontier() != 47
        {
            return Err(Error::Unsupported(
                "fixed_prompt defined cache is initial first3 only".into(),
            ));
        }
        Self::prepare_checked(
            scope.source().clone(),
            service,
            attempt,
            initial,
            History::FixedPromptPrefix,
            Some(scope),
            cancelled,
        )
        .await
    }
    pub(crate) async fn prepare_fixed_prompt_attention_kv(
        cut: &Arc<super::fixed_prompt_model_prefix::FixedPromptPrefixCut>,
        service: Arc<ResidentService>,
        attempt: &RootAttempt,
        initial: Arc<BoundModelState>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        if !cut.is_attention_cut() || attempt.base().frontier() != 0 || attempt.frontier() != 47 {
            return Err(Error::Unsupported(
                "fixed_prompt exact35-node Kv3 cache".into(),
            ));
        }
        initial.require_fixed_prompt_attention_kv(cut.scope())?;
        Self::prepare_checked(
            cut.scope().source().clone(),
            service,
            attempt,
            initial,
            History::FixedPromptAttentionKv,
            Some(cut.scope().clone()),
            cancelled,
        )
        .await
    }
    async fn prepare_checked(
        source: VerifiedTensorSource,
        service: Arc<ResidentService>,
        attempt: &RootAttempt,
        initial: Arc<BoundModelState>,
        history: History,
        fixed_prompt: Option<Arc<crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let first = initial.call();
        let layer = initial.layer();
        if first.source() != source.source_identity()
            || service.source_identity() != source.source_identity()
            || attempt.base().source() != source.source_identity()
            || initial.base() != attempt.base()
            || attempt.base().frontier() != 0
            || layer >= 24
        {
            return Err(Error::State(
                "full24 native cache same source/attempt/layer".into(),
            ));
        }
        let host = source.reserve_workspace(BYTES * 3 + 64 * 1024)?;
        let owner = Arc::new(ZeroSource {
            bytes: vec![0; BYTES],
            _source: source.clone(),
            _host: host,
        });
        let mut init = service.begin_initialization(attempt.base().root())?;
        init.retain_completion_owner(owner.clone())?;
        let lease = init.upload_bytes(&owner.bytes, BufferRole::RecurrentState)?;
        not_cancelled(cancelled)?;
        let initialized = init.complete_sync().await?;
        if !lease.is_initialized()
            || lease.bytes() != BYTES
            || !Arc::ptr_eq(lease.service(), &service)
        {
            return Err(Error::State(
                "full24 original native cache physical initialization".into(),
            ));
        }
        // Genuine H2D has completed before releasing its full host admission.
        drop(initialized);
        drop(owner);
        not_cancelled(cancelled)?;
        Ok(Arc::new(Self {
            lease,
            source,
            base: attempt.base(),
            layer,
            history,
            fixed_prompt,
            initial,
        }))
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn require_binding(&self, binding: &BoundModelState) -> Result<()> {
        let call = binding.call();
        let base = call.rows().base();
        let allowed = match self.history {
            History::Reference { frontier: 2 } => matches!(
                (call.mode(), base, call.rows().len()),
                (SourceCallMode::Prefill, 0, 1) | (SourceCallMode::Decode, 1, 1)
            ),
            History::Reference { frontier: 67 } => matches!(
                (call.mode(), base, call.rows().len()),
                (SourceCallMode::Prefill, 0, 64)
                    | (SourceCallMode::Prefill, 64, 2)
                    | (SourceCallMode::Decode, 66, 1)
            ),
            History::Reference { .. } => false,
            History::FixedPromptWhole => {
                let cut = self.initial.whole_cut()?;
                binding.require_fixed_prompt_whole(cut)?;
                call.mode() == SourceCallMode::Prefill && base == 0 && call.rows().len() == 47
            }
            History::FixedPromptPrefix => {
                let scope = self.fixed_prompt.as_ref().ok_or_else(|| {
                    Error::State("fixed_prompt defined cache scope absent".into())
                })?;
                binding.require_fixed_prompt_model_prefix(scope)?;
                call.mode() == SourceCallMode::Prefill && base == 0 && call.rows().len() == 47
            }
            History::FixedPromptAttentionKv => {
                let scope = self.fixed_prompt.as_ref().ok_or_else(|| {
                    Error::State("fixed_prompt Kv3 cache source scope absent".into())
                })?;
                binding.require_fixed_prompt_attention_kv(scope)?;
                call.mode() == SourceCallMode::Prefill && base == 0 && call.rows().len() == 47
            }
        };
        self.lease.service().ensure_usable()?;
        if !allowed
            || !binding.same_request(&self.initial)
            || binding.base() != self.base
            || binding.layer() != self.layer
            || binding.call().source() != self.source.source_identity()
            || !self.lease.is_initialized()
            || self.lease.bytes() != BYTES
        {
            return Err(Error::State(
                "full24 native cache cannot cross original layer/request/history".into(),
            ));
        }
        Ok(())
    }
}

mod whole;

mod natural;
pub(crate) use natural::SharedModelCache;
