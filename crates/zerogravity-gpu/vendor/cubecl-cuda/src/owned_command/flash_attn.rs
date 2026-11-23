//! Forward-only FlashAttention FA2 command on the original CUDA actor.
//! Raw pointers and the physical CUDA stream never cross this module boundary.

use super::OwnedCudaQueue;
use cubecl_core::server::{Handle, ServerError};

fn failure(message: impl Into<String>) -> ServerError {
    ServerError::graph_state(message)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16FlashAttention {
    rows: usize,
    base: usize,
    capacity: usize,
    query_heads: usize,
    kv_heads: usize,
    scale: f32,
}

impl Bf16FlashAttention {
    pub fn new(
        rows: usize,
        base: usize,
        capacity: usize,
        query_heads: usize,
        kv_heads: usize,
        scale: f32,
    ) -> Result<Self, ServerError> {
        if rows == 0
            || capacity == 0
            || base.checked_add(rows).is_none_or(|end| end > capacity)
            || query_heads == 0
            || kv_heads == 0
            || !query_heads.is_multiple_of(kv_heads)
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err(failure("FlashAttention BF16/head-256 geometry"));
        }
        Ok(Self {
            rows,
            base,
            capacity,
            query_heads,
            kv_heads,
            scale,
        })
    }

    pub(crate) fn rows(self) -> usize {
        self.rows
    }
    pub(crate) fn base(self) -> usize {
        self.base
    }
    pub(crate) fn capacity(self) -> usize {
        self.capacity
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
            .rows
            .checked_mul(self.query_heads)
            .and_then(|n| n.checked_mul(WIDTH))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| failure("FlashAttention query byte extent"))?;
        let kv = self
            .capacity
            .checked_mul(self.kv_heads)
            .and_then(|n| n.checked_mul(WIDTH))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| failure("FlashAttention KV byte extent"))?;
        let lse = self
            .rows
            .checked_mul(self.query_heads)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| failure("FlashAttention LSE byte extent"))?;
        Ok([q, kv, kv, q, lse])
    }
}

#[derive(Clone, Debug)]
pub struct Bf16FlashAttentionBindings {
    pub query: Handle,
    pub keys: Handle,
    pub values: Handle,
    pub output: Handle,
    pub softmax_lse: Handle,
}

impl Bf16FlashAttentionBindings {
    pub(crate) fn handles(&self) -> [&Handle; 5] {
        [
            &self.query,
            &self.keys,
            &self.values,
            &self.output,
            &self.softmax_lse,
        ]
    }
}

impl OwnedCudaQueue {
    /// Enqueue one dense BF16 causal FlashAttention FA2 forward on this queue.
    ///
    /// # Safety
    ///
    /// The caller retains every original allocation and its submission owner
    /// through actual CUDA completion/quarantine. Query/K/V are initialized
    /// reads; output/LSE are exclusive writable views for this submission.
    pub unsafe fn enqueue_bf16_flash_attention(
        &self,
        operation: Bf16FlashAttention,
        bindings: Bf16FlashAttentionBindings,
    ) -> Result<(), ServerError> {
        let sizes = operation.sizes()?;
        for (index, (handle, bytes)) in bindings.handles().into_iter().zip(sizes).enumerate() {
            if handle.service != self.service_id() || handle.size_in_used() != bytes as u64 {
                return Err(failure(format!(
                    "FlashAttention binding {index} service/extent differs"
                )));
            }
        }
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.enqueue_owned_flash_attention(stream, operation, bindings)
            })
            .map_err(|error| failure(format!("FlashAttention original actor: {error:?}")))?
    }
}
