//! Bounded observations owned by one HTTP response; no inference authority.
use super::engine::{
    GpuTokenObservation, MAX_TOKEN_OBSERVATIONS, ServiceError, TokenMetricsRequest,
    TokenObservation,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Instant;

pub(super) fn validate_wire_id(value: &str) -> Result<(), ServiceError> {
    if value.is_empty()
        || value.len() > 256
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
    {
        return Err(ServiceError::new(
            400,
            "invalid_request_id",
            "X-Request-ID must contain 1..256 ASCII letters, digits or ._:-",
        ));
    }
    Ok(())
}

#[derive(Serialize)]
struct Record {
    index: usize,
    token_id: u32,
    commit_observed_at_seconds: f64,
    output_received_at_seconds: f64,
    http_write_completed_at_seconds: Option<f64>,
    gpu: Option<GpuTokenObservation>,
}

pub(super) struct TokenMetrics {
    request: TokenMetricsRequest,
    provider_response_id: String,
    records: Vec<Record>,
    finished: bool,
}
impl TokenMetrics {
    pub(super) fn new(request: TokenMetricsRequest, provider_response_id: String) -> Self {
        Self {
            request,
            provider_response_id,
            records: Vec::new(),
            finished: false,
        }
    }
    pub(super) fn wire_request_id(&self) -> &str {
        self.request
            .wire_request_id
            .as_deref()
            .unwrap_or(&self.provider_response_id)
    }
    pub(super) fn record(&mut self, token: TokenObservation) -> Result<(), ServiceError> {
        let received = Instant::now();
        if self.finished
            || self.records.len() >= MAX_TOKEN_OBSERVATIONS
            || token.index != self.records.len()
            || token.commit_observed_at < self.request.started
            || token.commit_observed_at > received
            || token.gpu.is_some_and(|gpu| {
                gpu.producer_frontier == 0
                    || gpu.native_cohort.is_some_and(|cohort| {
                        cohort.publication_sequence == 0 || !matches!(cohort.width, 2 | 4 | 8)
                    })
            })
        {
            return Err(ServiceError::new(
                500,
                "invalid_token_observation",
                "invalid token sequence, clock or bounded output observation",
            ));
        }
        let committed = token
            .commit_observed_at
            .duration_since(self.request.started)
            .as_secs_f64();
        if self
            .records
            .last()
            .is_some_and(|last| committed < last.commit_observed_at_seconds)
        {
            return Err(ServiceError::new(
                500,
                "invalid_token_observation",
                "token publication clock moved backwards",
            ));
        }
        if let Some(last) = self.records.last()
            && match (last.gpu, token.gpu) {
                (Some(a), Some(b)) => {
                    a.producer_frontier.checked_add(1) != Some(b.producer_frontier)
                }
                (None, None) => false,
                _ => true,
            }
        {
            return Err(ServiceError::new(
                500,
                "invalid_token_observation",
                "GPU producer frontier differs from consecutive committed outputs",
            ));
        }
        self.records.push(Record {
            index: token.index,
            token_id: token.id,
            commit_observed_at_seconds: committed,
            output_received_at_seconds: received.duration_since(self.request.started).as_secs_f64(),
            http_write_completed_at_seconds: None,
            gpu: token.gpu,
        });
        Ok(())
    }
    pub(super) fn mark_written(&mut self) {
        if let Some(last) = self.records.last_mut() {
            last.http_write_completed_at_seconds =
                Some(self.request.started.elapsed().as_secs_f64());
        }
    }
    pub(super) fn token_ids(&self) -> Vec<u32> {
        self.records.iter().map(|record| record.token_id).collect()
    }
    pub(super) fn finish(&mut self, count: usize) -> Result<(Value, Value), ServiceError> {
        if self.finished || count != self.records.len() {
            return Err(ServiceError::new(
                500,
                "invalid_token_usage",
                "terminal token usage differs from once-only observations",
            ));
        }
        self.finished = true;
        let first = self
            .records
            .first()
            .map(|record| record.commit_observed_at_seconds);
        let generation = self
            .records
            .first()
            .zip(self.records.last())
            .map(|(a, b)| b.commit_observed_at_seconds - a.commit_observed_at_seconds);
        let mean = (count > 1).then(|| generation.expect("nonempty records") / (count - 1) as f64);
        // vLLM's named TTFT starts at scheduled_ts, which this observer does
        // not measure. Leave that field null; request-origin timing is below.
        let aggregate = json!({"time_to_first_token_ms":null, "generation_time_ms":generation.map(|x| x * 1000.0),
            "queue_time_ms":null, "mean_itl_ms":mean.map(|x| x * 1000.0), "tokens_per_second":null});
        let full = json!({"schema":"zerogravity.serving.request-token-metrics.v1", "provider_response_id":self.provider_response_id,
            "wire_request_id":self.wire_request_id(), "clock":"request_monotonic_seconds", "complete":true, "completion_tokens":count,
            "first_commit_observed_from_parsed_request_seconds":first,
            "commit_semantics":"observer immediately after existing source-root publication; not a GPU event clock or independent numerical proof",
            "delivery_semantics":"successful HTTP token-event write/flush to socket; not client receipt; absent for nonstream responses",
            "aggregate_semantics":"generation/mean ITL use provider commit-observed intervals; scheduled-start TTFT, queue time and scheduled throughput are unavailable; request-origin first commit is separate",
            "gpu_semantics":"optional exact committed producer frontier; native cohort sequence/width from the existing successful atomic-publication counter; absent native cohort for B1/interleaved; observation is not independent state or mathematics proof",
            "tokens":self.records});
        Ok((aggregate, full))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn observations_bind_once_only_ids_clocks_usage_and_delivery() {
        let started = Instant::now() - Duration::from_secs(1);
        let mut metrics = TokenMetrics::new(
            TokenMetricsRequest {
                started,
                wire_request_id: Some("run:request-0".into()),
            },
            "chatcmpl-zg-1".into(),
        );
        for (index, id) in [4, 5].into_iter().enumerate() {
            metrics
                .record(TokenObservation {
                    index,
                    id,
                    commit_observed_at: started + Duration::from_millis(10 + index as u64),
                    gpu: None,
                })
                .unwrap();
            metrics.mark_written();
        }
        assert_eq!(metrics.token_ids(), [4, 5]);
        let (aggregate, full) = metrics.finish(2).unwrap();
        assert_eq!(full["wire_request_id"], "run:request-0");
        assert_eq!(full["provider_response_id"], "chatcmpl-zg-1");
        assert!(
            full["tokens"][1]["http_write_completed_at_seconds"]
                .as_f64()
                .unwrap()
                >= 0.011
        );
        assert!((aggregate["mean_itl_ms"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert!(metrics.finish(2).is_err());
    }

    #[test]
    fn foreign_indices_clocks_counts_and_request_labels_fail_closed() {
        let start = Instant::now() - Duration::from_secs(1);
        let mut metrics = TokenMetrics::new(
            TokenMetricsRequest {
                started: start,
                wire_request_id: None,
            },
            "response-1".into(),
        );
        assert_eq!(metrics.wire_request_id(), "response-1");
        assert!(
            metrics
                .record(TokenObservation {
                    index: 1,
                    id: 3,
                    commit_observed_at: start,
                    gpu: None,
                })
                .is_err()
        );
        assert!(
            metrics
                .record(TokenObservation {
                    index: 0,
                    id: 3,
                    commit_observed_at: start - Duration::from_millis(1),
                    gpu: None,
                })
                .is_err()
        );
        assert!(
            metrics
                .record(TokenObservation {
                    index: 0,
                    id: 3,
                    commit_observed_at: Instant::now() + Duration::from_secs(1),
                    gpu: None,
                })
                .is_err()
        );
        metrics
            .record(TokenObservation {
                index: 0,
                id: 4,
                commit_observed_at: start,
                gpu: None,
            })
            .unwrap();
        assert!(
            metrics
                .record(TokenObservation {
                    index: 0,
                    id: 4,
                    commit_observed_at: start,
                    gpu: None,
                })
                .is_err()
        );
        assert!(metrics.finish(2).is_err());
        for label in ["", "not a label", "x\r\nInjected:yes", &"a".repeat(257)] {
            assert!(validate_wire_id(label).is_err());
        }
        assert!(validate_wire_id("run-1:request_2.3").is_ok());
    }

    #[test]
    fn observation_capacity_is_exact_and_zero_token_eos_has_no_invented_timing() {
        let started = Instant::now() - Duration::from_secs(1);
        let request = TokenMetricsRequest {
            started,
            wire_request_id: None,
        };
        let mut empty = TokenMetrics::new(request.clone(), "empty-eos".into());
        let (aggregate, full) = empty.finish(0).unwrap();
        assert!(aggregate["time_to_first_token_ms"].is_null());
        assert!(aggregate["mean_itl_ms"].is_null());
        assert_eq!(full["tokens"].as_array().unwrap().len(), 0);
        let mut bounded = TokenMetrics::new(request, "bounded".into());
        for index in 0..MAX_TOKEN_OBSERVATIONS {
            bounded
                .record(TokenObservation {
                    index,
                    id: 4,
                    commit_observed_at: started,
                    gpu: None,
                })
                .unwrap();
        }
        assert!(
            bounded
                .record(TokenObservation {
                    index: MAX_TOKEN_OBSERVATIONS,
                    id: 4,
                    commit_observed_at: started,
                    gpu: None,
                })
                .is_err()
        );
        assert_eq!(bounded.records.len(), MAX_TOKEN_OBSERVATIONS);
        assert!(bounded.finish(MAX_TOKEN_OBSERVATIONS).is_ok());
    }
    #[test]
    fn gpu_observations_preserve_b1_and_native_producer_frontiers() {
        use super::super::engine::NativeCohortObservation;
        let started = Instant::now() - Duration::from_secs(1);
        let mut metrics = TokenMetrics::new(
            TokenMetricsRequest {
                started,
                wire_request_id: Some("gpu:request".into()),
            },
            "chatcmpl-gpu".into(),
        );
        for (index, native_cohort) in [
            None,
            Some(NativeCohortObservation {
                publication_sequence: 7,
                width: 2,
            }),
        ]
        .into_iter()
        .enumerate()
        {
            metrics
                .record(TokenObservation {
                    index,
                    id: 4 + index as u32,
                    commit_observed_at: started,
                    gpu: Some(GpuTokenObservation {
                        producer_frontier: 11 + index as u64,
                        native_cohort,
                    }),
                })
                .unwrap();
            metrics.mark_written();
        }
        assert!(
            metrics
                .record(TokenObservation {
                    index: 2,
                    id: 6,
                    commit_observed_at: started,
                    gpu: Some(GpuTokenObservation {
                        producer_frontier: 12,
                        native_cohort: None
                    })
                })
                .is_err()
        );
        let (_, full) = metrics.finish(2).unwrap();
        assert_eq!(full["tokens"][0]["gpu"]["producer_frontier"], 11);
        assert!(full["tokens"][0]["gpu"]["native_cohort"].is_null());
        assert_eq!(
            full["tokens"][1]["gpu"]["native_cohort"]["publication_sequence"],
            7
        );
        assert_eq!(full["tokens"][1]["gpu"]["native_cohort"]["width"], 2);
        for gpu in [
            GpuTokenObservation {
                producer_frontier: 0,
                native_cohort: None,
            },
            GpuTokenObservation {
                producer_frontier: 1,
                native_cohort: Some(NativeCohortObservation {
                    publication_sequence: 0,
                    width: 2,
                }),
            },
            GpuTokenObservation {
                producer_frontier: 1,
                native_cohort: Some(NativeCohortObservation {
                    publication_sequence: 1,
                    width: 3,
                }),
            },
        ] {
            let mut invalid = TokenMetrics::new(
                TokenMetricsRequest {
                    started,
                    wire_request_id: None,
                },
                "invalid".into(),
            );
            assert!(
                invalid
                    .record(TokenObservation {
                        index: 0,
                        id: 4,
                        commit_observed_at: started,
                        gpu: Some(gpu)
                    })
                    .is_err()
            );
            assert!(invalid.records.is_empty());
        }
    }
}
