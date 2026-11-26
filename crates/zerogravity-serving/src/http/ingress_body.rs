//! Decoded body frames, with one deadline and one cumulative byte bound.
//! HTTP framing stays with Hyper. Retained bytes belong to the caller's
//! serving ledger; this reader does not own a scheduler or upload store.
use crate::TextService;
use bytes::Bytes;
use hyper::body::{Body, Incoming};
use std::{future::poll_fn, pin::Pin, time::Duration};
use tokio::time::{self, Instant};
use zerogravity_engine::router::{
    ServiceError,
    pending::{ServingCharge, allocate_bytes},
};

pub(super) struct Reader {
    incoming: Incoming,
    declared: Option<usize>,
    limit: usize,
    received: usize,
    deadline: Instant,
}
impl Reader {
    pub(super) fn new(
        incoming: Incoming,
        declared: Option<usize>,
        limit: usize,
        timeout: Duration,
    ) -> Result<Self, ServiceError> {
        if declared.is_some_and(|length| length > limit) {
            return Err(ServiceError::new(
                413,
                "body_too_large",
                "HTTP body too large",
            ));
        }
        Ok(Self {
            incoming,
            declared,
            limit,
            received: 0,
            deadline: Instant::now() + timeout,
        })
    }

    /// The consumer must use a frame before reading the next one or charge
    /// its own retained copy. Hyper's bounded receive buffer is connection
    /// headroom, not another application allocation.
    pub(super) async fn next(&mut self) -> Result<Option<Bytes>, ServiceError> {
        if Instant::now() >= self.deadline {
            return Err(ServiceError::new(
                408,
                "timeout",
                "HTTP body deadline reached",
            ));
        }
        let frame = time::timeout_at(
            self.deadline,
            poll_fn(|cx| Pin::new(&mut self.incoming).poll_frame(cx)),
        )
        .await
        .map_err(|_| ServiceError::new(408, "timeout", "HTTP body deadline reached"))?;
        let Some(frame) = frame else {
            if self.declared.is_some_and(|length| length != self.received) {
                return Err(ServiceError::new(
                    400,
                    "incomplete_body",
                    "short Content-Length",
                ));
            }
            return Ok(None);
        };
        let frame =
            frame.map_err(|_| ServiceError::new(400, "incomplete_body", "incomplete HTTP body"))?;
        let data = frame.into_data().map_err(|_| {
            ServiceError::new(400, "unsupported_transfer", "request trailers unsupported")
        })?;
        let end = self
            .received
            .checked_add(data.len())
            .ok_or_else(|| ServiceError::new(413, "body_too_large", "HTTP body too large"))?;
        if self.declared.is_some_and(|length| end > length) {
            return Err(ServiceError::new(
                400,
                "bad_length",
                "body exceeds Content-Length",
            ));
        }
        if end > self.limit {
            return Err(ServiceError::new(
                413,
                "body_too_large",
                "HTTP body too large",
            ));
        }
        self.received = end;
        Ok(Some(data))
    }
}

/// A contiguous JSON/input staging owner. Known lengths allocate once;
/// unknown lengths grow geometrically under the cap. Both old and new
/// allocations are charged until the old backing has actually been dropped.
pub(super) struct Collected {
    backing: Box<[u8]>,
    used: usize,
    charge: ServingCharge,
}
impl Collected {
    pub(super) fn into_parts(self) -> (Box<[u8]>, usize, ServingCharge) {
        (self.backing, self.used, self.charge)
    }
}

pub(super) async fn collect(
    mut reader: Reader,
    service: &dyn TextService,
) -> Result<Collected, ServiceError> {
    let initial = reader.declared.unwrap_or(0);
    let charge = service.charge_ingress("http-body", initial as u64)?;
    let mut body = Collected {
        backing: allocate_bytes(initial)?,
        used: 0,
        charge,
    };
    while let Some(data) = reader.next().await? {
        let end = reader.received;
        if end > body.backing.len() {
            // The reader has already checked the cumulative cap.
            let capacity = end
                .checked_next_power_of_two()
                .unwrap_or(reader.limit)
                .max(4096)
                .min(reader.limit);
            let charge = service.charge_ingress("http-body", capacity as u64)?;
            let mut backing = allocate_bytes(capacity)?;
            backing[..body.used].copy_from_slice(&body.backing[..body.used]);
            body.backing = backing;
            body.charge = charge;
        }
        body.backing[body.used..end].copy_from_slice(&data);
        body.used = end;
    }
    Ok(body)
}

#[cfg(test)]
mod tests;
