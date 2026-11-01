//! Model-free transport fixtures only; no model arithmetic qualification.
use crate::{
    TextService,
    service::{Request, RequestEvents, ServiceError, ServingCharge},
};
use std::sync::{Arc, Mutex, MutexGuard};
use zerogravity_core::resource_arbiter::AllocationClass;
use zerogravity_engine::router::pending::{CancelToken, PendingQueue, ServingHost};

static LOCK: Mutex<()> = Mutex::new(());
pub(crate) fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
pub(crate) fn host() -> ServingHost {
    ServingHost::new(256).unwrap()
}
pub(crate) trait HostBytes {
    fn serving_bytes(&self) -> u64;
}
impl HostBytes for ServingHost {
    fn serving_bytes(&self) -> u64 {
        self.snapshot().unwrap().3
    }
}
pub(crate) fn cancel(host: &ServingHost) -> CancelToken {
    let queue = PendingQueue::<()>::new(host.clone(), Some(1)).unwrap();
    queue.cancel_token(None)
}
#[derive(Clone)]
pub(crate) struct IdleService(pub(crate) ServingHost);
impl TextService for IdleService {
    fn request_admitted(
        &self,
        _: Request,
        _: Vec<ServingCharge>,
    ) -> Result<RequestEvents, ServiceError> {
        Err(ServiceError::new(
            404,
            "model_not_found",
            "no model in transport fixture",
        ))
    }
    fn charge_ingress(&self, session: &str, bytes: u64) -> Result<ServingCharge, ServiceError> {
        self.0
            .charge("transport-test", session, AllocationClass::Scratch, bytes)
    }
    fn model_ids(&self) -> Result<Vec<String>, ServiceError> {
        Ok(Vec::new())
    }
    fn max_output(&self) -> Result<usize, ServiceError> {
        Ok(8)
    }
    fn admission(&self) -> Result<serde_json::Value, ServiceError> {
        Ok(
            serde_json::json!({"kind":"transport_test_only", "serving_host_reserved_bytes":self.0.serving_bytes()}),
        )
    }
}
pub(crate) fn idle(host: &ServingHost) -> Arc<dyn TextService> {
    Arc::new(IdleService(host.clone()))
}
