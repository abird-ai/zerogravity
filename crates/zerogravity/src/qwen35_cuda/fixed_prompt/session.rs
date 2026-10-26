//! Fixed model policy over the common execution and feedback owners.
use super::*;
pub struct Session {
    engine: native::FixedPromptSession,
    templates: Arc<bindings::Templates>,
}
impl Session {
    pub(super) fn new(
        engine: native::FixedPromptSession,
        templates: Arc<bindings::Templates>,
    ) -> Self {
        Self { engine, templates }
    }
    pub fn binding(&self) -> crate::session_root::RootBinding {
        self.engine.binding()
    }
    pub fn snapshot_empty(&self) -> Result<native::FixedPromptEmptySnapshot> {
        self.engine.snapshot_empty()
    }
    pub fn restore_empty(
        &mut self,
        snapshot: &native::FixedPromptEmptySnapshot,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        self.engine.restore_empty(snapshot, cancelled)
    }
    pub async fn prepare_prompt_with<F>(
        &self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<native::FixedPromptPending>
    where
        F: for<'a> FnMut(native::FixedPromptNaturalReadback<'a>) -> Result<()>,
    {
        not_cancelled(cancelled)?;
        OriginalWindowModelSource::validate_fixed_prompt_request(ids, 1)?;
        self.engine
            .prepare_prompt_with(self.templates.bind(0, ids)?, cancelled, validate)
            .await
    }
    pub async fn prepare_next_with<F>(
        &self,
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<native::FixedPromptPending>
    where
        F: for<'a> FnMut(native::FixedPromptNaturalReadback<'a>) -> Result<()>,
    {
        not_cancelled(cancelled)?;
        let request = self.engine.feedback_request()?;
        let record = self
            .templates
            .bind(request.base(), &[request.selected_id()])?;
        self.engine
            .prepare_next_with(request, record, cancelled, validate)
            .await
    }
    pub fn commit(
        &mut self,
        pending: native::FixedPromptPending,
        cancelled: Option<&AtomicBool>,
    ) -> Result<native::FixedPromptCallReceipt> {
        self.engine.commit(pending, cancelled)
    }
    pub async fn infer_prompt_with<F>(
        &mut self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<native::FixedPromptCallReceipt>
    where
        F: for<'a> FnMut(native::FixedPromptNaturalReadback<'a>) -> Result<()>,
    {
        let pending = self.prepare_prompt_with(ids, cancelled, validate).await?;
        self.commit(pending, cancelled)
    }
    pub async fn infer_next_with<F>(
        &mut self,
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<native::FixedPromptCallReceipt>
    where
        F: for<'a> FnMut(native::FixedPromptNaturalReadback<'a>) -> Result<()>,
    {
        let pending = self.prepare_next_with(cancelled, validate).await?;
        self.commit(pending, cancelled)
    }
    pub async fn infer_prompt(
        &mut self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<native::FixedPromptCallReceipt> {
        self.infer_prompt_with(ids, cancelled, |_| Ok(())).await
    }
    pub async fn infer_next(
        &mut self,
        cancelled: Option<&AtomicBool>,
    ) -> Result<native::FixedPromptCallReceipt> {
        self.infer_next_with(cancelled, |_| Ok(())).await
    }
    pub async fn probe_cached(
        self,
        fault: native::FixedPromptNaturalFault,
        cancelled: Option<&AtomicBool>,
    ) -> Result<native::FixedPromptNaturalFaultOutcome> {
        let request = self.engine.feedback_request()?;
        let record = self
            .templates
            .bind(request.base(), &[request.selected_id()])?;
        let Self { engine, templates } = self;
        drop(templates);
        engine.probe_cached(fault, record, cancelled).await
    }
}
