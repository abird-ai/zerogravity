//! Shared resident source-call dispatch, retaining each backend's original body.
use super::*;
use crate::resident_leases::ResidentSubmission as Flight;
impl ResidentProgram {
    pub(crate) fn is_ordinary_source(&self) -> bool {
        match &self.body {
            #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
            ResidentProgramBody::OrdinaryCuda(_) => true,
            #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
            ResidentProgramBody::SourceCall(_) => true,
            _ => false,
        }
    }
    pub(crate) fn source_call_waits_on_cancel(&self) -> bool {
        #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
        if matches!(self.body, ResidentProgramBody::SourceCall(_)) {
            return true;
        }
        false
    }
    pub(crate) fn validate_ordinary_source_request(&self, request: &ForwardPlan) -> Result<()> {
        match &self.body {
            #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
            ResidentProgramBody::OrdinaryCuda(body) => body.validate_request(request).map(|_| ()),
            #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
            ResidentProgramBody::SourceCall(body) => body.validate_request(request).map(|_| ()),
            _ => Err(Error::State(
                "source call reached another resident body".into(),
            )),
        }
    }
    pub(crate) fn validate_source_call_states(
        &self,
        request: &ForwardPlan,
        states: &[StateBinding],
    ) -> Result<()> {
        #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
        if let ResidentProgramBody::SourceCall(body) = &self.body {
            return body
                .states(body.validate_request(request)?, states)
                .map(|_| ());
        }
        let _ = (request, states);
        Ok(())
    }
    pub(crate) async fn launch_ordinary_source(
        &self,
        flight: &mut Flight<'_>,
        request: &ForwardPlan,
        states: &[StateBinding],
        output: ResidentOutput,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
        observer: Option<&mut dyn super::source_observation::SourceCallObserver>,
    ) -> Result<ResidentSubmission> {
        let _ = cancelled;
        match &self.body {
            #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
            ResidentProgramBody::OrdinaryCuda(body) => {
                if observer.is_some() {
                    return Err(Error::Unsupported(
                        "CUDA source-call observer is not implemented".into(),
                    ));
                }
                body.execute(flight, request, states, output, cancelled)
            }
            #[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
            ResidentProgramBody::SourceCall(body) => {
                body.execute(flight, request, states, output, observer)
                    .await
            }
            _ => Err(Error::State(
                "source call reached another resident body".into(),
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
    pub(crate) async fn source_call_diagnostic_trig(
        &self,
        angles: &[f32],
        owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<Vec<f32>> {
        match &self.body {
            ResidentProgramBody::SourceCall(body) => body.diagnostic_trig(angles, owner).await,
            _ => Err(Error::State(
                "source trig reached another resident body".into(),
            )),
        }
    }

    pub(crate) fn source_call_native_probe(&self) -> Result<crate::hip::native::FinalNativeProbe> {
        match &self.body {
            ResidentProgramBody::SourceCall(body) => Ok(body.native_probe()),
            _ => Err(Error::State(
                "source-call probe reached another resident body".into(),
            )),
        }
    }
}
