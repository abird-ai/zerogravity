//! Bounded finite encoder work in the original model registry and worker FIFO.
//! No decoder StateRoot, token presenter, new execution queue, or HTTP math.
use crate::{
    model_instance::EmbeddingEncoder,
    router::{
        ModelRead, RequestEvents, ServiceError, Work,
        event_channel::{self, mpsc},
        pending::{self, CancelToken, PendingQueue, ServingCharge, ServingHost},
        statistics::WorkStats,
    },
};
use std::sync::{Arc, atomic::Ordering};
use zerogravity_core::{identity::SourceIdentity, resource_arbiter::AllocationClass};

pub const MAX_INPUTS: usize = 16;
pub const MAX_DIMENSIONS: usize = 4096;
type Result<T> = std::result::Result<T, ServiceError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmbeddingInput {
    Text(String),
    /// Exact input IDs; no implied special tokens, mask or truncation.
    Tokens(Vec<u32>),
}
pub struct EmbeddingRequest {
    pub model: String,
    pub inputs: Vec<EmbeddingInput>,
}
fn overflow() -> ServiceError {
    ServiceError::new(400, "size_overflow", "embedding allocation extent overflow")
}
impl EmbeddingRequest {
    pub(crate) fn validate(&self, max_bytes: usize, context: usize, vocabulary: u64) -> Result<()> {
        if !(1..=MAX_INPUTS).contains(&self.inputs.len()) {
            return Err(ServiceError::new(
                400,
                "invalid_input",
                "embedding batch must contain 1..16 inputs",
            ));
        }
        let mut bytes = 0usize;
        for input in &self.inputs {
            let size = match input {
                EmbeddingInput::Text(text) if !text.is_empty() => text.len(),
                EmbeddingInput::Tokens(ids)
                    if !ids.is_empty()
                        && ids.len() <= context
                        && ids.iter().all(|&id| u64::from(id) < vocabulary) =>
                {
                    ids.len()
                        .checked_mul(std::mem::size_of::<u32>())
                        .ok_or_else(overflow)?
                }
                _ => {
                    return Err(ServiceError::new(
                        400,
                        "invalid_input",
                        "empty or invalid embedding token input",
                    ));
                }
            };
            bytes = bytes.checked_add(size).ok_or_else(overflow)?;
        }
        if bytes > max_bytes {
            return Err(ServiceError::new(
                413,
                "input_too_large",
                "embedding batch exceeds configured input bytes",
            ));
        }
        Ok(())
    }
    fn backing_bytes(&self) -> Result<u64> {
        let mut bytes = pending::checked_sum(&[
            pending::heap_strings(&[&self.model])?,
            pending::vec_bytes::<EmbeddingInput>(self.inputs.capacity())?,
        ])?;
        for input in &self.inputs {
            let size = match input {
                EmbeddingInput::Text(text) => pending::heap_strings(&[text])?,
                EmbeddingInput::Tokens(ids) => pending::vec_bytes::<u32>(ids.capacity())?,
            };
            bytes = bytes.checked_add(size).ok_or_else(overflow)?;
        }
        Ok(bytes)
    }
}

/// Complete ordered batch, with the original admission attached to its
/// backing. Holding a result does not keep an encoder/model reader alive.
pub struct EmbeddingBatch {
    values: Vec<f32>,
    dimensions: usize,
    input_tokens: usize,
    source: SourceIdentity,
    _charge: ServingCharge,
}
impl EmbeddingBatch {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub fn len(&self) -> usize {
        self.values.len() / self.dimensions
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        let begin = index.checked_mul(self.dimensions)?;
        self.values.get(begin..begin.checked_add(self.dimensions)?)
    }
    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
}
/// The original request ID/cancellation/charged event-channel receipt,
/// instantiated with a single typed terminal rather than fake text tokens.
pub type EmbeddingReceipt = RequestEvents<Result<EmbeddingBatch>>;

pub(crate) struct EmbeddingJob {
    request: EmbeddingRequest,
    model: Arc<dyn EmbeddingEncoder>,
    result: Option<EmbeddingBatch>,
    context: usize,
    cursor: usize,
    events: mpsc::SyncSender<Result<EmbeddingBatch>>,
    cancel: CancelToken,
    _request: ServingCharge,
    // The last field retires the actual registration reader after all
    // private input/output/model owners have been destroyed.
    reader: ModelRead,
}
impl EmbeddingJob {
    pub(crate) fn prepare(
        request: EmbeddingRequest,
        model: Arc<dyn EmbeddingEncoder>,
        reader: ModelRead,
        host: &ServingHost,
        queue: &PendingQueue<Work>,
        id: u64,
        context: usize,
    ) -> Result<(Box<Self>, EmbeddingReceipt)> {
        let reader = reader;
        let model = model;
        let request_bytes = pending::checked_sum(&[
            request.backing_bytes()?,
            crate::router::fixed_size::<Self>()?,
            64, // boxed job backing/rounding
        ])?;
        let charge = host.charge(
            &request.model,
            "embedding-request",
            AllocationClass::Scratch,
            request_bytes,
        )?;
        let receipt_charge = host.charge(
            &request.model,
            "embedding-receipt",
            AllocationClass::Scratch,
            pending::heap_strings(&[&request.model])?,
        )?;
        let dimensions = model.dimensions();
        let count = request
            .inputs
            .len()
            .checked_mul(dimensions)
            .ok_or_else(overflow)?;
        let output = host.charge(
            &request.model,
            "embedding-output",
            AllocationClass::Output,
            pending::vec_bytes::<f32>(count)?,
        )?;
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(|_| {
            ServiceError::new(503, "queue_allocation", "embedding allocation refused")
        })?;
        // An adapter must write every coordinate; never publish untouched
        // output storage as though it were real model output.
        values.resize(count, f32::NAN);
        let result = EmbeddingBatch {
            values,
            dimensions,
            input_tokens: 0,
            source: model.source(),
            _charge: output,
        };
        let (events, receiver) =
            mpsc::try_sync_channel(host, &request.model, "embedding-terminal", 1)?;
        let cancel_charge = host.charge(
            &request.model,
            "cancel-token",
            AllocationClass::Scratch,
            event_channel::cancel_bytes()?,
        )?;
        let cancel = queue.cancel_token(Some(Arc::new(cancel_charge)));
        let receipt = RequestEvents::new(
            id,
            request.model.clone(),
            receiver,
            cancel.clone(),
            Some(receipt_charge),
            None,
        );
        Ok((
            Box::new(Self {
                request,
                model,
                result: Some(result),
                context,
                cursor: 0,
                events,
                cancel,
                _request: charge,
                reader,
            }),
            receipt,
        ))
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// A single unpadded sequence is one finite scheduling quantum. Keep the
    /// entire result private until all sequences succeed. No replay or new
    /// request on another worker; cancellation drops the whole batch.
    pub(crate) fn step(&mut self, stats: &WorkStats) -> bool {
        if self.cancel.is_cancelled() {
            stats.cancelled.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let batch = self.result.as_mut().expect("private encoder batch");
        let input = &self.request.inputs[self.cursor];
        let begin = self.cursor * batch.dimensions;
        let row = &mut batch.values[begin..begin + batch.dimensions];
        let result = self
            .model
            .embed_into(input, self.context, row, self.cancel.flag());
        if self.cancel.is_cancelled() {
            stats.cancelled.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let result = result.map_err(ServiceError::inference).and_then(|tokens| {
            if tokens == 0
                || tokens > self.context
                || matches!(input, EmbeddingInput::Tokens(ids) if ids.len() != tokens)
                || row.iter().any(|v| !v.is_finite())
            {
                return Err(ServiceError::new(
                    500,
                    "invalid_embedding",
                    "encoder returned invalid vector or usage",
                ));
            }
            batch.input_tokens = batch
                .input_tokens
                .checked_add(tokens)
                .ok_or_else(overflow)?;
            Ok(())
        });
        if let Err(error) = result {
            stats.failed.fetch_add(1, Ordering::Relaxed);
            let _ = self.events.try_send(Err(error));
            return false;
        }
        self.cursor += 1;
        if self.cursor != self.request.inputs.len() {
            return true;
        }
        let result = self.result.take().expect("complete private encoder batch");
        let tokens = result.input_tokens as u64;
        // One terminal, one receiver slot; no output-blocked encoder state.
        if self.events.try_send(Ok(result)).is_ok() {
            stats.embedding_requests.fetch_add(1, Ordering::Relaxed);
            stats
                .embedding_input_tokens
                .fetch_add(tokens, Ordering::Relaxed);
            self.reader.complete();
        } else {
            stats.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        false
    }
}
