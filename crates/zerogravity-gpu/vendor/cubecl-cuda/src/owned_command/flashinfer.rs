//! Single-request FlashInfer decode command on the original CUDA actor.
//! Raw pointers and the physical CUDA stream never cross this module boundary.

use super::OwnedCudaQueue;
use cubecl_core::server::{Handle, ServerError};

pub const FLASHINFER_DECODE_SCRATCH_BYTES: usize = 32 * 1024 * 1024;

fn failure(message: impl Into<String>) -> ServerError {
    ServerError::graph_state(message)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16FlashInferDecode {
    kv_len: usize,
    capacity: usize,
    query_heads: usize,
    kv_heads: usize,
    scale: f32,
}

impl Bf16FlashInferDecode {
    pub fn new(
        kv_len: usize,
        capacity: usize,
        query_heads: usize,
        kv_heads: usize,
        scale: f32,
    ) -> Result<Self, ServerError> {
        if kv_len == 0
            || capacity == 0
            || kv_len > capacity
            || query_heads == 0
            || kv_heads == 0
            || !query_heads.is_multiple_of(kv_heads)
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err(failure("FlashInfer BF16/head-256 decode geometry"));
        }
        Ok(Self {
            kv_len,
            capacity,
            query_heads,
            kv_heads,
            scale,
        })
    }

    pub(crate) fn kv_len(self) -> usize {
        self.kv_len
    }
    pub(crate) fn query_heads(self) -> usize {
        self.query_heads
    }
    pub(crate) fn kv_heads(self) -> usize {
        self.kv_heads
    }
    pub(crate) fn scale(self) -> f32 {
        self.scale
    }

    pub(crate) fn sizes(self) -> Result<[usize; 5], ServerError> {
        const WIDTH: usize = 256;
        let q = self
            .query_heads
            .checked_mul(WIDTH)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| failure("FlashInfer query byte extent"))?;
        let kv = self
            .capacity
            .checked_mul(self.kv_heads)
            .and_then(|n| n.checked_mul(WIDTH))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| failure("FlashInfer KV byte extent"))?;
        Ok([q, kv, kv, q, FLASHINFER_DECODE_SCRATCH_BYTES])
    }
}

#[derive(Clone, Debug)]
pub struct Bf16FlashInferDecodeBindings {
    pub query: Handle,
    pub keys: Handle,
    pub values: Handle,
    pub output: Handle,
    pub scratch: Handle,
}

impl Bf16FlashInferDecodeBindings {
    pub(crate) fn handles(&self) -> [&Handle; 5] {
        [
            &self.query,
            &self.keys,
            &self.values,
            &self.output,
            &self.scratch,
        ]
    }
}

impl OwnedCudaQueue {
    /// Enqueue one BF16/head-256 NHD FlashInfer decode on this queue.
    ///
    /// # Safety
    ///
    /// The caller retains every original allocation and its submission owner
    /// through actual CUDA completion/quarantine. Query/K/V are initialized
    /// reads; output/scratch are exclusive writable views for this submission.
    pub unsafe fn enqueue_bf16_flashinfer_decode(
        &self,
        operation: Bf16FlashInferDecode,
        bindings: Bf16FlashInferDecodeBindings,
    ) -> Result<(), ServerError> {
        let sizes = operation.sizes()?;
        for (index, (handle, bytes)) in bindings.handles().into_iter().zip(sizes).enumerate() {
            if handle.service != self.service_id() || handle.size_in_used() != bytes as u64 {
                return Err(failure(format!(
                    "FlashInfer binding {index} service/extent differs"
                )));
            }
        }
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.enqueue_owned_flashinfer_decode(stream, operation, bindings)
            })
            .map_err(|error| failure(format!("FlashInfer original actor: {error:?}")))?
    }
}
