//! New short-HF caller adapter only. Existing workspace/launch loop stays sole.
use super::*;
use crate::cuda::rms::{FullModelRmsScratch, RmsScratchRun};
use crate::resident_leases::ResidentSubmission;

/// Constructor stays in this original-workspace module; no caller-built token.
pub(crate) struct RmsScratchRegistration<'a, 'flight> {
    flight: &'a mut ResidentSubmission<'flight>,
    scope: &'a Arc<FullModelRmsScratch>,
}
impl RmsScratchRegistration<'_, '_> {
    pub(crate) fn require_scope(&self, scope: &Arc<FullModelRmsScratch>) -> Result<()> {
        if !Arc::ptr_eq(self.scope, scope) {
            return Err(Error::State(
                "RMS workspace registration belongs to another record".into(),
            ));
        }
        self.flight.require_rms_scratch_scope(scope)
    }
    pub(crate) fn require_value_storage_accesses(
        &self,
        required: &[(
            crate::resident_leases::ResidentLease,
            zerogravity_core::device_events::DeviceAccess,
        )],
    ) -> Result<()> {
        self.flight.require_rms_actor_accesses(required)
    }
    pub(crate) fn retain_value_storage(
        &mut self,
        run: Arc<crate::cuda::native::model_values::bounded::ValueStorageRun>,
    ) -> Result<()> {
        self.flight.retain_completion_owner(run)
    }
    pub(crate) fn retain_streamed_head(
        &mut self,
        run: Arc<crate::cuda::native::text_head::Run>,
    ) -> Result<()> {
        self.flight.retain_completion_owner(run)
    }
    pub(crate) fn retain(&mut self, run: Arc<RmsScratchRun>) -> Result<()> {
        // Attach first; original unknown Drop owns the exact same run.
        self.flight.retain_completion_owner(run)
    }
}

/// Caller reaches this only after its original fair slot, current-attempt
/// validation and real submit; `workspace` is the live original Mutex guard.
/// Keep that guard and flight through original completion/late readbacks.
pub(in crate::resident_inference) fn enqueue_with_rms_scratch<T: Send + 'static>(
    workspace: &mut async_lock::MutexGuard<'_, Workspace>,
    flight: &mut ResidentSubmission<'_>,
    scope: &Arc<FullModelRmsScratch>,
    enqueue: impl FnOnce(&mut ResidentProgram) -> Result<T> + Send,
) -> Result<T> {
    let placement = workspace
        .program
        .full_model_body()?
        .text_enqueue_placement()?;
    match placement {
        crate::resident_leases::EnqueuePlacement::Caller => {
            register_and_enqueue(workspace, flight, scope, enqueue)
        }
        crate::resident_leases::EnqueuePlacement::DeviceActor => {
            flight.ordered_enqueue(|flight| register_and_enqueue(workspace, flight, scope, enqueue))
        }
    }
}

/// Entire original registration/run scope shares the execution thread.
fn register_and_enqueue<T>(
    workspace: &mut async_lock::MutexGuard<'_, Workspace>,
    flight: &mut ResidentSubmission<'_>,
    scope: &Arc<FullModelRmsScratch>,
    enqueue: impl FnOnce(&mut ResidentProgram) -> Result<T>,
) -> Result<T> {
    let body = workspace.program.full_model_body()?.clone();
    let required = body.require_rms_scratch(scope)?;
    flight.require_rms_actor_accesses(&required)?;
    flight.retain_completion_owner(body.completion_owner())?;
    let mut registration = RmsScratchRegistration { flight, scope };
    let values_run = body.values().start_storage_run(&mut registration, scope)?;
    body.register_text_head(&mut registration)?;
    let run = scope.start_run(&mut registration)?;
    // Active run is held locally only during launch, then by original flight.
    // Body/actors/PreparedInference keep only the pool's Weak run lookup.
    let submitted = enqueue(&mut workspace.program)?;
    run.require_finished()?;
    if let Some(run) = values_run {
        run.require_finished()?;
    }
    Ok(submitted)
}
