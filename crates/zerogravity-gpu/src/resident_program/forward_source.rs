//! Shared resident text-execution dispatch, retaining each backend's original body.
use super::*;
use crate::resident_leases::ResidentSubmission as Flight;
impl ResidentProgram {
    pub(crate) fn is_forward_source(&self) -> bool {
        match &self.body {
            #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
            ResidentProgramBody::CudaForward(_) => true,
            #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
            ResidentProgramBody::TextExecution(_) => true,
            _ => false,
        }
    }
    pub(crate) fn text_execution_waits_on_cancel(&self) -> bool {
        #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
        if let ResidentProgramBody::CudaForward(body) = &self.body {
            return body.waits_on_cancel();
        }

        #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
        if matches!(self.body, ResidentProgramBody::TextExecution(_)) {
            return true;
        }
        false
    }
    pub(crate) fn validate_forward_source_request(&self, request: &ForwardPlan) -> Result<()> {
        match &self.body {
            #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
            ResidentProgramBody::CudaForward(body) => body.validate_request(request).map(|_| ()),
            #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
            ResidentProgramBody::TextExecution(body) => body.validate_request(request).map(|_| ()),
            _ => Err(Error::State(
                "text execution reached another resident body".into(),
            )),
        }
    }
    pub(crate) fn validate_text_execution_states(
        &self,
        request: &ForwardPlan,
        states: &[StateBinding],
    ) -> Result<()> {
        #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
        if let ResidentProgramBody::TextExecution(body) = &self.body {
            return body
                .states(body.validate_request(request)?, states)
                .map(|_| ());
        }
        let _ = (request, states);
        Ok(())
    }
    #[allow(clippy::too_many_arguments)] // Shared backend seam keeps execution ownership explicit.
    pub(crate) async fn launch_forward_source(
        &self,
        flight: &mut Flight<'_>,
        request: &ForwardPlan,
        states: &[StateBinding],
        output: ResidentOutput,
        compact_greedy: bool,
        device_token: Option<cubecl::server::Handle>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
        observer: Option<&mut dyn super::text_observation::TextObserver>,
    ) -> Result<ResidentSubmission> {
        let _ = cancelled;
        match &self.body {
            #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
            ResidentProgramBody::CudaForward(body) => {
                if observer.is_some() {
                    return Err(Error::Unsupported(
                        "CUDA text-execution observer is not implemented".into(),
                    ));
                }
                body.execute(
                    flight,
                    request,
                    states,
                    output,
                    compact_greedy,
                    device_token.as_ref(),
                    cancelled,
                )
                .await
            }
            #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
            ResidentProgramBody::TextExecution(body) => {
                if compact_greedy || device_token.is_some() {
                    return Err(Error::Unsupported(
                        "HIP text execution has no compact CUDA greedy readout".into(),
                    ));
                }
                body.execute(flight, request, states, output, observer)
                    .await
            }
            _ => Err(Error::State(
                "text execution reached another resident body".into(),
            )),
        }
    }

    #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
    pub(crate) fn text_execution_cohort_buffers(
        &self,
        requests: &[ForwardPlan],
        emit_logits: bool,
    ) -> Result<Option<super::text_execution::CohortBuffers>> {
        match &self.body {
            ResidentProgramBody::TextExecution(body) => body.cohort_buffers(requests, emit_logits),
            _ => Err(Error::Unsupported(
                "cohort requires HIP text execution".into(),
            )),
        }
    }

    #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
    pub(crate) async fn launch_text_execution_cohort(
        &self,
        flight: &mut Flight<'_>,
        requests: &[ForwardPlan],
        states: &[Vec<StateBinding>],
        buffers: &super::text_execution::CohortBuffers,
    ) -> Result<super::text_execution::CohortSubmission> {
        match &self.body {
            ResidentProgramBody::TextExecution(body) => {
                body.execute_cohort(flight, requests, states, buffers).await
            }
            _ => Err(Error::Unsupported(
                "cohort requires HIP text execution".into(),
            )),
        }
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub(crate) async fn launch_forward_source_prefill_cohort(
        &self,
        flight: &mut Flight<'_>,
        requests: &[ForwardPlan],
        states: &[Vec<StateBinding>],
        logits: Option<crate::resident_leases::ResidentLease>,
        values: &super::cuda_forward::CudaForwardCohortValues,
    ) -> Result<super::cuda_forward::CudaForwardCohort> {
        match &self.body {
            ResidentProgramBody::CudaForward(body) => {
                body.execute_prefill_cohort(flight, requests, states, logits, values)
                    .await
            }
            _ => Err(Error::Unsupported(
                "forward source prefill cohort requires CUDA".into(),
            )),
        }
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub(crate) async fn launch_forward_source_decode_cohort(
        &self,
        flight: &mut Flight<'_>,
        requests: &[ForwardPlan],
        states: &[Vec<StateBinding>],
        logits: crate::resident_leases::ResidentLease,
        values: &super::cuda_forward::CudaForwardCohortValues,
    ) -> Result<super::cuda_forward::CudaForwardCohort> {
        match &self.body {
            ResidentProgramBody::CudaForward(body) => {
                body.execute_decode_cohort(flight, requests, states, logits, values)
                    .await
            }
            _ => Err(Error::Unsupported(
                "forward source decode cohort requires CUDA".into(),
            )),
        }
    }
}

#[cfg(all(
    feature = "hip-source",
    feature = "hip",
    feature = "hardware-abort-probe"
))]
impl ResidentProgram {
    #[cfg(feature = "qwen-source-qualification")]
    pub(crate) async fn text_execution_diagnostic_trig(
        &self,
        angles: &[f32],
        owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<Vec<f32>> {
        match &self.body {
            ResidentProgramBody::TextExecution(body) => body.diagnostic_trig(angles, owner).await,
            _ => Err(Error::State(
                "source trig reached another resident body".into(),
            )),
        }
    }

    pub(crate) fn text_execution_native_probe(
        &self,
    ) -> Result<crate::hip::native::FinalNativeProbe> {
        match &self.body {
            ResidentProgramBody::TextExecution(body) => Ok(body.native_probe()),
            _ => Err(Error::State(
                "text-execution probe reached another resident body".into(),
            )),
        }
    }
}
