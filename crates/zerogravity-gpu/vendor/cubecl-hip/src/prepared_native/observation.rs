//! Read-only original-actor observations; no work/admission or lookup authority.
use super::*;
#[derive(Clone, Debug)]
pub struct HipScopedSetObservation {
    pub set_key: u64,
    pub logical_stream: u64,
    pub catalogue_keys: usize,
    pub non_null_modules: usize,
    pub admitted_native: HipNativeInventory,
    pub scoped_metadata_bytes: usize,
    pub sealed: bool,
    pub usable: bool,
    pub sealed_miss_refused: bool,
}
#[derive(Clone, Debug)]
pub struct HipScopedObservation {
    pub sets: Vec<HipScopedSetObservation>,
    pub original_key_overlap: usize,
    pub default_stream_refused: bool,
}
impl RegisteredHipNative {
    /// No allocation/enqueue: inspect exact existing actor routes and preflights.
    pub fn inspect_scoped(client: &Client) -> Result<HipScopedObservation, ServerError> {
        let service = client.service_id();
        if service != ServiceId::of::<HipServer>(service.device) {
            return Err(failure("HIP scoped observation requires original service"));
        }
        DeviceHandle::<HipServer>::new(service.device)
            .submit_blocking(|server| server.inspect_scoped_native())
            .map_err(|e| failure(format!("HIP scoped observation actor: {e:?}")))?
    }
    /// The SAME ordinary stream admission guard, without allocating/enqueuing.
    pub fn scoped_stream_usable(client: &Client, logical: u64) -> Result<bool, ServerError> {
        let service = client.service_id();
        if service != ServiceId::of::<HipServer>(service.device) {
            return Err(failure("HIP stream observation requires original service"));
        }
        DeviceHandle::<HipServer>::new(service.device)
            .submit_blocking(move |server| server.scoped_stream_usable(logical))
            .map_err(|e| failure(format!("HIP stream observation actor: {e:?}")))?
    }
}
