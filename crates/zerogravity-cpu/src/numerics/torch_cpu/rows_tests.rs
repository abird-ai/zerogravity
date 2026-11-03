//! Independent original operator data; never runtime inputs from expected outputs.
use super::*;
use crate::accelerator::SourceIdentity;
use zerogravity_core::{
    encoding::{ScalarDType, StoredTensorEncoding},
    source::SourceTensorRef,
};

use super::rows_reference::{Archive, number};
fn source() -> SourceIdentity {
    SourceIdentity::for_new_allocation().unwrap()
}

#[test]
#[ignore = "requires independently captured original source Conv4 and attention archive"]
fn original_convolution_calls() {
    let archive = Archive::open();
    let mut words = 0;
    let mut cases = 0;
    for call in archive.doc["calls"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["operation"] == "conv")
    {
        let case = number(call, "case");
        let rows = number(call, "rows");
        let channels = number(call, "channels");
        let input = archive.values(case, "input");
        let prior = archive.values(case, "prior");
        let shape = [channels as u64, 1, 4];
        let source = source();
        let done = conv::rows::causal_bf16(conv::rows::Inputs {
            source,
            tensor_name: "original.conv.weight",
            weight: SourceTensorRef {
                kind: StoredTensorEncoding::Scalar(ScalarDType::Bf16),
                shape: &shape,
                packed_bytes: archive.bytes(case, "weight"),
            },
            activation: &input,
            rows,
            previous: call["initialized"]
                .as_bool()
                .unwrap()
                .then_some(prior.as_slice()),
        })
        .unwrap();
        words += archive.compare(case, "raw", &done.raw_bf16);
        words += archive.compare(case, "after", &done.state);
        let activated = transcendentals::silu_bf16_widened_f32_contiguous16(source, &done.raw_bf16)
            .unwrap()
            .into_iter()
            .map(|v| norm::round_bf16(v).unwrap())
            .collect::<Vec<_>>();
        words += archive.compare(case, "activated", &activated);
        cases += 1;
        eprintln!(
            "SOURCE_CONV_ROWS_CASE case={case} rows={rows} initialized={} exact=true",
            call["initialized"]
        );
    }
    assert_eq!(cases, 12);
    eprintln!("SOURCE_CONV_ROWS_EXACT cases={cases} words={words} trained_model=false gpu=false");
}

#[test]
#[ignore = "requires independently captured original source Conv4 and attention archive"]
fn original_full_call_attention() {
    let archive = Archive::open();
    let mut words = 0;
    let mut cases = 0;
    for call in archive.doc["calls"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["operation"] == "attention")
    {
        let case = number(call, "case");
        let rows = number(call, "rows");
        let tokens = number(call, "tokens");
        let query = archive.values(case, "query");
        let key = archive.values(case, "key");
        let value = archive.values(case, "value");
        let done = attention::rows::causal_bf16(
            attention::rows::Inputs {
                source: source(),
                query: &query,
                keys: &key,
                values: &value,
                rows,
                prior_tokens: number(call, "base"),
                query_heads: number(call, "heads"),
                kv_heads: number(call, "kv_heads"),
                width: number(call, "width"),
            },
            None,
        )
        .unwrap();
        for (label, data) in [
            ("dots", &done.dots),
            ("scores", &done.scores),
            ("probabilities", &done.probabilities),
            ("context", &done.context),
        ] {
            words += archive.compare(case, label, data);
        }
        cases += 1;
        eprintln!("SOURCE_ATTENTION_ROWS_CASE case={case} rows={rows} tokens={tokens} exact=true");
    }
    assert_eq!(cases, 20);
    eprintln!(
        "SOURCE_ATTENTION_ROWS_EXACT cases={cases} words={words} trained_model=false gpu=false"
    );
}

#[test]
fn causal_rows_mask_future_values_but_keep_the_complete_contraction() {
    let query = vec![0.0; 512];
    let keys = vec![0.0; 512];
    let mut values = vec![1.0; 512];
    values[256..].fill(3.0);
    let input = || attention::rows::Inputs {
        source: source(),
        query: &query,
        keys: &keys,
        values: &values,
        rows: 2,
        prior_tokens: 0,
        query_heads: 1,
        kv_heads: 1,
        width: 256,
    };
    let done = attention::rows::causal_bf16(input(), None).unwrap();
    assert_eq!(done.probabilities, [1.0, 0.0, 0.5, 0.5]);
    assert!(
        done.context[..256]
            .iter()
            .all(|v| v.to_bits() == 1.0f32.to_bits())
    );
    assert!(
        done.context[256..]
            .iter()
            .all(|v| v.to_bits() == 2.0f32.to_bits())
    );
    let cancelled = std::sync::atomic::AtomicBool::new(true);
    assert!(attention::rows::causal_bf16(input(), Some(&cancelled)).is_err());
    let mut invalid = input();
    invalid.prior_tokens = usize::MAX;
    assert!(attention::rows::causal_bf16(invalid, None).is_err());
    let mut invalid = input();
    invalid.rows = 65;
    assert!(attention::rows::causal_bf16(invalid, None).is_err());
    let mut invalid = input();
    invalid.kv_heads = 0;
    assert!(attention::rows::causal_bf16(invalid, None).is_err());
    let mut invalid = input();
    invalid.source = SourceIdentity::unbound();
    assert!(attention::rows::causal_bf16(invalid, None).is_err());
    let mut wrong = values.clone();
    wrong[511] = f32::from_bits(0x3f80_0001);
    let mut invalid = input();
    invalid.values = &wrong;
    assert!(attention::rows::causal_bf16(invalid, None).is_err());
}
