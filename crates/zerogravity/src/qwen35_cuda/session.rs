//! Family policy composes common execution; feedback remains engine-owned.
use super::*;
use crate::session_root::RootBinding;
pub struct Session {
    engine: native::TextSession,
    pub(super) templates: Arc<templates::Templates>,
}
impl Session {
    pub(super) fn new(engine: native::TextSession, templates: Arc<templates::Templates>) -> Self {
        Self { engine, templates }
    }
    pub fn engine(&self) -> &native::TextSession {
        &self.engine
    }
    pub fn engine_mut(&mut self) -> &mut native::TextSession {
        &mut self.engine
    }
    pub fn binding(&self) -> RootBinding {
        self.engine.binding()
    }
    pub fn snapshot_empty(&self) -> Result<native::TextEmptySnapshot> {
        self.engine.snapshot_empty()
    }
    pub fn restore_empty(
        &mut self,
        snapshot: &native::TextEmptySnapshot,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        self.engine.restore_empty(snapshot, cancelled)
    }
    pub fn cached_cohort_eligible(&self) -> bool {
        self.engine.cached_cohort_eligible()
    }
    pub fn cohort_feedback_eligible(&self) -> bool {
        self.engine.cohort_feedback_eligible()
    }
    pub async fn prepare_prompt_with<F>(
        &self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<native::TextPending>
    where
        F: for<'a> FnMut(native::TextReadback<'a>) -> Result<()>,
    {
        not_cancelled(cancelled)?;
        let (binding, implementation) = self.templates.bind_prompt(ids)?;
        self.engine
            .prepare_prompt_with(binding, implementation, cancelled, validate)
            .await
    }
    pub async fn prepare_next_with<F>(
        &self,
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<native::TextPending>
    where
        F: for<'a> FnMut(native::TextReadback<'a>) -> Result<()>,
    {
        not_cancelled(cancelled)?;
        let request = self.engine.feedback_request()?;
        let (binding, implementation) = self.templates.bind_cached_metadata(
            request.prompt_tokens(),
            request.base(),
            request.selected_id(),
        )?;
        self.engine
            .prepare_next_with(request, binding, implementation, cancelled, validate)
            .await
    }
    pub fn commit(
        &mut self,
        pending: native::TextPending,
        cancelled: Option<&AtomicBool>,
    ) -> Result<native::TextCallReceipt> {
        self.engine.commit(pending, cancelled)
    }
    pub async fn infer_prompt_with<F>(
        &mut self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<native::TextCallReceipt>
    where
        F: for<'a> FnMut(native::TextReadback<'a>) -> Result<()>,
    {
        let pending = self.prepare_prompt_with(ids, cancelled, validate).await?;
        self.commit(pending, cancelled)
    }
    pub async fn infer_next_with<F>(
        &mut self,
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<native::TextCallReceipt>
    where
        F: for<'a> FnMut(native::TextReadback<'a>) -> Result<()>,
    {
        let pending = self.prepare_next_with(cancelled, validate).await?;
        self.commit(pending, cancelled)
    }
    pub async fn infer_prompt(
        &mut self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<native::TextCallReceipt> {
        self.infer_prompt_with(ids, cancelled, |_| Ok(())).await
    }
    pub async fn infer_next(
        &mut self,
        cancelled: Option<&AtomicBool>,
    ) -> Result<native::TextCallReceipt> {
        self.infer_next_with(cancelled, |_| Ok(())).await
    }
    pub async fn probe_cached(
        self,
        fault: native::TextFault,
        cancelled: Option<&AtomicBool>,
    ) -> Result<native::TextFaultOutcome> {
        let request = self.engine.feedback_request()?;
        let (binding, implementation) = self.templates.bind_cached_metadata(
            request.prompt_tokens(),
            request.base(),
            request.selected_id(),
        )?;
        let Self { engine, templates } = self;
        // The observation must not be satisfied by the app's own source cache.
        drop(templates);
        engine
            .probe_cached(fault, binding, implementation, cancelled)
            .await
    }
}
