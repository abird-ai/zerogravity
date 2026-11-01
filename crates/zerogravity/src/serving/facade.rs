//! Facade view of the original engine service; no registry, queue or transport IO.
use super::{
    TextService,
    service::{Request, RequestEvents, ServiceError, ServingCharge},
};
type Result<T> = std::result::Result<T, ServiceError>;

impl TextService for crate::Runtime {
    fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents> {
        self.engine.request_admitted(request, ingress)
    }
    fn charge_ingress(&self, session: &str, bytes: u64) -> Result<ServingCharge> {
        self.engine.charge_ingress(session, bytes)
    }
    fn model_ids(&self) -> Result<Vec<String>> {
        self.engine.model_ids()
    }
    fn max_output(&self) -> Result<usize> {
        self.engine.max_output()
    }
    fn admission(&self) -> Result<serde_json::Value> {
        self.engine.admission()
    }
    fn require_open(&self) -> crate::Result<()> {
        self.engine.require_open()
    }
}
