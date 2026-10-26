//! Model calls for independent batch rows; the GPU owns grouped publication.
use super::*;
type Validation<'a> = &'a mut dyn for<'b> FnMut(native::TextCohortReadback<'b>) -> Result<()>;
impl Model {
    pub async fn prepare_fixed_prompt_cohort(
        &self,
        sessions: &[&Session],
        ids: &[u32],
        cancelled: &[&AtomicBool],
        validate: Option<Validation<'_>>,
    ) -> Result<Option<native::TextPendingCohort>> {
        prepared(
            self.prepare_fixed_prompt_cohort_status(sessions, ids, cancelled, validate)
                .await?,
        )
    }
    pub async fn prepare_feedback_cohort(
        &self,
        sessions: &[&Session],
        cancelled: &[&AtomicBool],
        validate: Option<Validation<'_>>,
    ) -> Result<Option<native::TextPendingCohort>> {
        prepared(
            self.prepare_feedback_cohort_status(sessions, cancelled, validate)
                .await?,
        )
    }
    pub async fn probe_feedback_cohort_fault(
        &self,
        sessions: &[&Session],
        cancelled: &[&AtomicBool],
        fault: native::TextCohortFault,
    ) -> Result<native::TextCohortFaultOutcome> {
        if sessions.len() != 4 {
            return Err(Error::Malformed(
                "text cached batch requires four requests".into(),
            ));
        }
        let _metadata = self
            .templates
            .source()
            .source()
            .source()
            .reserve_workspace(4096)?;
        let engines = sessions
            .iter()
            .map(|session| {
                if !Arc::ptr_eq(&self.templates, &session.templates) {
                    return Err(Error::State("text batch belongs to another model".into()));
                }
                Ok(session.engine())
            })
            .collect::<Result<Vec<_>>>()?;
        let requests = engines
            .iter()
            .map(|s| s.feedback_request())
            .collect::<Result<Vec<_>>>()?;
        let calls = requests
            .iter()
            .map(|request| {
                self.templates.bind_cached_metadata(
                    request.prompt_tokens(),
                    request.base(),
                    request.selected_id(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.resident
            .probe_feedback_cohort_fault(&engines, &requests, &calls, cancelled, fault)
            .await
    }
    pub async fn prepare_fixed_prompt_cohort_status(
        &self,
        sessions: &[&Session],
        ids: &[u32],
        cancelled: &[&AtomicBool],
        validate: Option<Validation<'_>>,
    ) -> Result<native::TextCohortPreparation> {
        if sessions.len() != 4 || ids.len() != 47 {
            return Err(Error::Malformed(
                "text initial batch requires four prompt47 requests".into(),
            ));
        }
        let _metadata = self
            .templates
            .source()
            .source()
            .source()
            .reserve_workspace(4096)?;
        let calls = sessions
            .iter()
            .map(|session| {
                if !Arc::ptr_eq(&self.templates, &session.templates) {
                    return Err(Error::State("text batch belongs to another model".into()));
                }
                self.templates.bind_prompt(ids)
            })
            .collect::<Result<Vec<_>>>()?;
        let engines = sessions.iter().map(|s| s.engine()).collect::<Vec<_>>();
        self.resident
            .prepare_fixed_prompt_cohort_status(&engines, &calls, cancelled, validate)
            .await
    }
    pub async fn prepare_feedback_cohort_status(
        &self,
        sessions: &[&Session],
        cancelled: &[&AtomicBool],
        validate: Option<Validation<'_>>,
    ) -> Result<native::TextCohortPreparation> {
        if sessions.len() != 4 {
            return Err(Error::Malformed(
                "text cached batch requires four requests".into(),
            ));
        }
        let _metadata = self
            .templates
            .source()
            .source()
            .source()
            .reserve_workspace(4096)?;
        let engines = sessions
            .iter()
            .map(|session| {
                if !Arc::ptr_eq(&self.templates, &session.templates) {
                    return Err(Error::State("text batch belongs to another model".into()));
                }
                Ok(session.engine())
            })
            .collect::<Result<Vec<_>>>()?;
        let requests = engines
            .iter()
            .map(|s| s.feedback_request())
            .collect::<Result<Vec<_>>>()?;
        let calls = requests
            .iter()
            .map(|request| {
                self.templates.bind_cached_metadata(
                    request.prompt_tokens(),
                    request.base(),
                    request.selected_id(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.resident
            .prepare_feedback_cohort_status(&engines, &requests, &calls, cancelled, validate)
            .await
    }
}

fn prepared(status: native::TextCohortPreparation) -> Result<Option<native::TextPendingCohort>> {
    match status {
        native::TextCohortPreparation::Ready(pending) => Ok(Some(pending)),
        native::TextCohortPreparation::Capacity => Ok(None),
        native::TextCohortPreparation::CancelledBeforeLaunch => {
            Err(Error::State("short HF request cancelled".into()))
        }
    }
}
