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
        || value.len() > MAX_WIRE_ID_BYTES
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

// Fixed semantic strings written once per metrics response. Their lengths are
// part of the charged response envelope below.
const SCHEMA: &str = "zerogravity.serving.request-token-metrics.v1";
const CLOCK: &str = "request_monotonic_seconds";
const COMMIT_SEMANTICS: &str = "observer immediately after existing source-root publication; not a GPU event clock or independent numerical proof";
const DELIVERY_SEMANTICS: &str = "successful HTTP token-event write/flush to socket; not client receipt; absent for nonstream responses";
const AGGREGATE_SEMANTICS: &str = "generation/mean ITL use provider commit-observed intervals; scheduled-start TTFT, queue time and scheduled throughput are unavailable; request-origin first commit is separate";
const GPU_SEMANTICS: &str = "optional exact committed producer frontier; native cohort sequence/width from the existing successful atomic-publication counter; absent native cohort for B1/interleaved; observation is not independent state or mathematics proof";

/// Largest JSON number/identifier text this module can emit (`u64` decimal,
/// `f64` shortest-round-trip and a JSON `null`).
const JSON_SCALAR_TEXT: u64 = 32;
const MAX_WIRE_ID_BYTES: usize = 256;
const PROVIDER_RESPONSE_ID_BOUND: u64 = "chatcmpl-zg-".len() as u64 + 20;
/// Heap text owned by the six fixed semantic strings in the full Value tree.
/// All six are source literals with no JSON quote, backslash or control byte,
/// so their serialized length equals their source length.
const FIXED_LITERALS: u64 = (SCHEMA.len()
    + CLOCK.len()
    + COMMIT_SEMANTICS.len()
    + DELIVERY_SEMANTICS.len()
    + AGGREGATE_SEMANTICS.len()
    + GPU_SEMANTICS.len()) as u64;
/// Response-level provider ID Strings. The nonstream path can overlap a
/// `format!` owner, the `TokenMetrics` holder and Value-tree copies; the
/// streaming path keeps the format owner, its `TokenMetrics` clone, the SSE
/// framed copy and a Value copy. Six bounded texts cover the 2x format/SSE
/// growth and the clone/tree owners without claiming an exact phase peak.
const PROVIDER_ID_STRINGS: u64 = 6 * PROVIDER_RESPONSE_ID_BOUND;
/// Wire ID Strings: the request-event owner, the `TokenMetrics` clone and the
/// full Value-tree copy. `validate_wire_id` restricts every byte to unescaped
/// ASCII, so each allocation is bounded by `MAX_WIRE_ID_BYTES`.
const WIRE_ID_STRINGS: u64 = 3 * MAX_WIRE_ID_BYTES as u64;
/// The streaming writer formats the bounded `X-Request-ID` line into a fresh
/// String; the same String growth bound applies to that response-level owner.
const WIRE_HEADER_OWNER: u64 = 2 * ("X-Request-ID: ".len() as u64 + MAX_WIRE_ID_BYTES as u64 + 2);
/// SSE framing around one serialized `Value` copy.
const SSE_COPY_OVERHEAD: u64 = ("data: ".len() + 2) as u64;

/// One `serde_json::Map<String, Value>` leaf node, priced from the pinned std
/// layout rather than a per-field guess: `BTreeMap` allocates the full 11-key
/// leaf (16-byte header, 11 inline key `String` headers, 11 inline `Value`
/// slots), so only the key text is variable. `producer_root` (3 keys, 27 text
/// bytes) is therefore exactly 659 B.
const MAP_LEAF: u64 =
    (16 + 11 * (std::mem::size_of::<String>() + std::mem::size_of::<Value>())) as u64;
const _: () = assert!(std::mem::size_of::<Value>() == 32);
const _: () = assert!(std::mem::size_of::<String>() == 24);
const _: () = assert!(MAP_LEAF == 632);

/// A split root embeds the 11-slot leaf data plus 12 child edges. A 12-field
/// map therefore owns one internal node and two leaves, not one leaf.
const MAP_INTERNAL: u64 = MAP_LEAF + 12 * std::mem::size_of::<usize>() as u64;
const _: () = assert!(MAP_INTERNAL == 728);

fn map_nodes(fields: u64) -> u64 {
    debug_assert!(fields <= 2 * 11 + 1);
    if fields <= 11 {
        MAP_LEAF
    } else {
        MAP_INTERNAL + 2 * MAP_LEAF
    }
}

/// Old+new backing bound for one serialization buffer. The retained capacity
/// starts at `minimum_capacity`; at the final reallocation the old capacity is
/// at most the final text length and the new one at most twice that length.
fn output_buffer_peak(text: u64, minimum_capacity: u64) -> u64 {
    if text <= minimum_capacity {
        minimum_capacity
    } else {
        text.saturating_mul(3)
    }
}

/// One JSON object plus its key text.
fn json_object<I>(keys: I) -> u64
where
    I: IntoIterator<Item = u64>,
{
    let mut fields = 0u64;
    let mut total = 0u64;
    for key in keys {
        fields += 1;
        total += key;
    }
    total + map_nodes(fields)
}

/// `serde_json::Value::Array` of `len` values: pinned `Vec` doubling holds the
/// old backing during regrowth.
fn json_array(len: u64, element: u64) -> u64 {
    2 * len * element
}

/// Serialized text of one object field: key quotes, colon, comma and a scalar.
fn json_scalar_text(key: &str, value: u64) -> u64 {
    key.len() as u64 + 4 + value
}

/// Serialized text of one object field: key quotes, colon, comma, string quotes
/// and the already-unescaped string body.
fn json_string_text(key: &str, value: u64) -> u64 {
    key.len() as u64 + 6 + value
}

/// Serialized text of one object field: key quotes, colon, comma and the scalar.
fn json_member(key: u64) -> u64 {
    key + 4 + JSON_SCALAR_TEXT
}

/// Per-record envelope for the typed metric owner and its serialized JSON
/// twin. The combined response buffer contains every record, so charge three
/// times each record's serialized text (old + new backing at the final
/// growth). The response-level wrapper is priced separately by
/// `response_envelope`; the additive sum bounds the one combined buffer.
pub(super) fn record_envelope() -> u64 {
    let producer_root = json_object([
        "session_id".len() as u64,
        "session_epoch".len() as u64,
        "root".len() as u64,
    ]);
    let native_cohort = json_object(["publication_sequence".len() as u64, "width".len() as u64]);
    let gpu = json_object([
        "producer_frontier".len() as u64,
        "producer_root".len() as u64,
        "native_cohort".len() as u64,
    ]) + producer_root
        + native_cohort;
    let record_keys = [
        "index".len() as u64,
        "token_id".len() as u64,
        "commit_observed_at_seconds".len() as u64,
        "output_received_at_seconds".len() as u64,
        "http_write_completed_at_seconds".len() as u64,
        "gpu".len() as u64,
    ];
    let tree = json_object(record_keys) + gpu;
    let serialized = json_member("index".len() as u64)
        + json_member("token_id".len() as u64)
        + json_member("commit_observed_at_seconds".len() as u64)
        + json_member("output_received_at_seconds".len() as u64)
        + json_member("http_write_completed_at_seconds".len() as u64)
        + json_member("gpu".len() as u64)
        + json_member("producer_frontier".len() as u64)
        + json_member("producer_root".len() as u64)
        + json_member("session_id".len() as u64)
        + json_member("session_epoch".len() as u64)
        + json_member("root".len() as u64)
        + json_member("native_cohort".len() as u64)
        + json_member("publication_sequence".len() as u64)
        + json_member("width".len() as u64)
        + 32; // object/array punctuation
    // Typed record, its Value tree, and the record text's share of the one
    // combined serializer/SSE buffer (old + new backing at final growth).
    (std::mem::size_of::<Record>() as u64) + tree + 3 * serialized
}

/// Once-per-response growth overlap of the bounded `Vec<Record>`. The final
/// record backing is already charged once per emitted token by
/// `record_envelope`; only the old backing live during the last doubling is
/// added here.
pub(super) fn records_growth() -> u64 {
    (MAX_TOKEN_OBSERVATIONS as u64 / 2) * std::mem::size_of::<Record>() as u64
}

/// Response-level JSON objects written once: both aggregate/full metric
/// objects, their field keys, the fixed semantic String heaps and serialized
/// text, the bounded provider/wire identifier owners, and the serializer or
/// SSE output buffer.
pub(super) fn response_envelope() -> u64 {
    let aggregate_keys = [
        "time_to_first_token_ms",
        "generation_time_ms",
        "queue_time_ms",
        "mean_itl_ms",
        "tokens_per_second",
    ];
    let full_keys = [
        "schema",
        "provider_response_id",
        "wire_request_id",
        "clock",
        "complete",
        "completion_tokens",
        "first_commit_observed_from_parsed_request_seconds",
        "commit_semantics",
        "delivery_semantics",
        "aggregate_semantics",
        "gpu_semantics",
        "tokens",
    ];
    let aggregate = json_object(aggregate_keys.iter().map(|key| key.len() as u64));
    let full = json_object(full_keys.iter().map(|key| key.len() as u64));
    let tokens = json_array(
        MAX_TOKEN_OBSERVATIONS as u64,
        std::mem::size_of::<Value>() as u64,
    );
    // Serialized text, including fixed values and bounded identifiers. Each
    // number/null uses the same bounded scalar text as `json_member`; every
    // string field includes both quote pairs and the field comma.
    let aggregate_text = aggregate_keys
        .iter()
        .map(|key| json_scalar_text(key, JSON_SCALAR_TEXT))
        .sum::<u64>()
        + 2;
    let full_text = json_string_text("schema", SCHEMA.len() as u64)
        + json_string_text("provider_response_id", PROVIDER_RESPONSE_ID_BOUND)
        + json_string_text("wire_request_id", MAX_WIRE_ID_BYTES as u64)
        + json_string_text("clock", CLOCK.len() as u64)
        + json_scalar_text("complete", 4)
        + json_scalar_text("completion_tokens", JSON_SCALAR_TEXT)
        + json_scalar_text(
            "first_commit_observed_from_parsed_request_seconds",
            JSON_SCALAR_TEXT,
        )
        + json_string_text("commit_semantics", COMMIT_SEMANTICS.len() as u64)
        + json_string_text("delivery_semantics", DELIVERY_SEMANTICS.len() as u64)
        + json_string_text("aggregate_semantics", AGGREGATE_SEMANTICS.len() as u64)
        + json_string_text("gpu_semantics", GPU_SEMANTICS.len() as u64)
        + json_scalar_text("tokens", 0)
        + 2;
    // Per-record text is already charged by `record_envelope`; only the array
    // brackets/commas are response-level here.
    let tokens_text = MAX_TOKEN_OBSERVATIONS as u64 + 2;
    // The two parent keys that place these objects in the completion/usage
    // response are source-derived punctuation on the serializer copy.
    let parent_keys_text =
        json_scalar_text("metrics", 0) + json_scalar_text("zerogravity_token_metrics", 0);
    let text = aggregate_text + full_text + tokens_text + parent_keys_text;
    // `write_json` uses one `to_vec` buffer; the SSE caller builds one
    // `format!` String copy. They are alternate response paths, so price the
    // larger actual old+new backing peak instead of assuming two exact text
    // buffers.
    let output_buffer =
        output_buffer_peak(text, 128).max(output_buffer_peak(text + SSE_COPY_OVERHEAD, 8));
    // Value-tree owners, response-level String owners, then the serializer or
    // SSE copy buffer.
    (aggregate + full + tokens)
        + FIXED_LITERALS
        + PROVIDER_ID_STRINGS
        + WIRE_ID_STRINGS
        + WIRE_HEADER_OWNER
        + output_buffer
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
                    || gpu
                        .producer_root
                        .is_some_and(|root| root.session_id == 0 || root.root == 0)
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
                        || match (a.producer_root, b.producer_root) {
                            (Some(a), Some(b)) => {
                                a.session_id != b.session_id
                                    || a.session_epoch != b.session_epoch
                                    || a.root.checked_add(1) != Some(b.root)
                            }
                            (None, None) => false,
                            _ => true,
                        }
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
        let full = json!({"schema":SCHEMA, "provider_response_id":self.provider_response_id,
            "wire_request_id":self.wire_request_id(), "clock":CLOCK, "complete":true, "completion_tokens":count,
            "first_commit_observed_from_parsed_request_seconds":first,
            "commit_semantics":COMMIT_SEMANTICS,
            "delivery_semantics":DELIVERY_SEMANTICS,
            "aggregate_semantics":AGGREGATE_SEMANTICS,
            "gpu_semantics":GPU_SEMANTICS,
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
                        producer_root: None,
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
                        producer_root: None,
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
                producer_root: None,
                native_cohort: None,
            },
            GpuTokenObservation {
                producer_frontier: 1,
                producer_root: None,
                native_cohort: Some(NativeCohortObservation {
                    publication_sequence: 0,
                    width: 2,
                }),
            },
            GpuTokenObservation {
                producer_frontier: 1,
                producer_root: None,
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

    #[test]
    fn response_envelope_prices_the_split_twelve_field_root() {
        let aggregate = json_object(
            [
                "time_to_first_token_ms",
                "generation_time_ms",
                "queue_time_ms",
                "mean_itl_ms",
                "tokens_per_second",
            ]
            .iter()
            .map(|key| key.len() as u64),
        );
        let full = json_object(
            [
                "schema",
                "provider_response_id",
                "wire_request_id",
                "clock",
                "complete",
                "completion_tokens",
                "first_commit_observed_from_parsed_request_seconds",
                "commit_semantics",
                "delivery_semantics",
                "aggregate_semantics",
                "gpu_semantics",
                "tokens",
            ]
            .iter()
            .map(|key| key.len() as u64),
        );
        assert_eq!(MAP_LEAF, 632);
        assert_eq!(MAP_INTERNAL, 728);
        assert_eq!(map_nodes(12), MAP_INTERNAL + 2 * MAP_LEAF);
        assert!(full >= MAP_INTERNAL + 2 * MAP_LEAF);
        let tokens = json_array(
            MAX_TOKEN_OBSERVATIONS as u64,
            std::mem::size_of::<Value>() as u64,
        );
        assert!(
            response_envelope()
                >= aggregate
                    + full
                    + tokens
                    + FIXED_LITERALS
                    + PROVIDER_ID_STRINGS
                    + WIRE_ID_STRINGS
                    + WIRE_HEADER_OWNER
        );
        // Four leaves: record, gpu, producer_root and native_cohort.
        assert!(record_envelope() >= std::mem::size_of::<Record>() as u64 + 4 * MAP_LEAF);
        assert!(
            records_growth()
                >= (MAX_TOKEN_OBSERVATIONS as u64 / 2) * std::mem::size_of::<Record>() as u64
        );
    }

    #[test]
    fn response_envelope_prices_literal_and_id_heaps_punctuation_and_output_growth() {
        assert_eq!(json_scalar_text("x", 1), 1 + 4 + 1);
        assert_eq!(json_string_text("x", 1), 1 + 6 + 1);
        assert_eq!(output_buffer_peak(128, 128), 128);
        assert_eq!(output_buffer_peak(129, 128), 3 * 129);
        assert_eq!(output_buffer_peak(8, 8), 8);
        assert_eq!(output_buffer_peak(9, 8), 3 * 9);
        const { assert!(FIXED_LITERALS > 0) };
        const { assert!(PROVIDER_ID_STRINGS >= PROVIDER_RESPONSE_ID_BOUND) };
        const { assert!(WIRE_ID_STRINGS >= MAX_WIRE_ID_BYTES as u64) };
        const { assert!(WIRE_HEADER_OWNER > MAX_WIRE_ID_BYTES as u64) };
        // The fixed semantic strings and the validated wire-ID alphabet need
        // no JSON escaping beyond the two quote bytes already priced.
        for literal in [
            SCHEMA,
            CLOCK,
            COMMIT_SEMANTICS,
            DELIVERY_SEMANTICS,
            AGGREGATE_SEMANTICS,
            GPU_SEMANTICS,
        ] {
            assert!(!literal.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20));
        }
        assert!(validate_wire_id("chatcmpl-zg-0").is_ok());
        assert!(validate_wire_id("run-1:request_2.3").is_ok());
    }
}
