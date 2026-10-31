use super::super::*;
use crate::accelerator::SourceIdentity;

#[test]
fn chunk_zero_updates_preserve_real_prior_and_refuse_invalid_inputs() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let zeros = vec![0.0; 16 * 17 * 128];
    let rates = vec![0.0; 16 * 17];
    let state = vec![3.5; 16 * 128 * 128];
    let input = || ChunkInputs {
        source,
        rows: 17,
        heads: 16,
        query: &zeros,
        key: &zeros,
        value: &zeros,
        beta: &rates,
        log_decay: &rates,
        previous_state: Some(&state),
    };
    let output = core(input()).unwrap();
    assert!(output.output_bf16.iter().all(|v| v.to_bits() == 0));
    assert_eq!(output.state, state);
    let mut bad = input();
    bad.rows = 65;
    assert!(core(bad).is_err());
    let mut bad = input();
    bad.heads = 0;
    assert!(core(bad).is_err());
    let mut bad = input();
    bad.source = SourceIdentity::unbound();
    assert!(core(bad).is_err());
    let mut bad = input();
    bad.previous_state = Some(&state[..state.len() - 1]);
    assert!(core(bad).is_err());
    let invalid = vec![f32::INFINITY; rates.len()];
    let mut bad = input();
    bad.log_decay = &invalid;
    assert!(core(bad).is_err());
    let invalid = vec![0.125; rates.len()];
    let mut bad = input();
    bad.log_decay = &invalid;
    assert!(core(bad).is_err());
    let invalid = vec![f32::from_bits(0x3f80_0001); zeros.len()];
    let mut bad = input();
    bad.query = &invalid;
    assert!(core(bad).is_err());
    let mut bad = input();
    bad.beta = &[0.0];
    assert!(core(bad).is_err());
    let invalid = vec![2.0; rates.len()];
    let mut bad = input();
    bad.beta = &invalid;
    assert!(core(bad).is_err());
    let invalid = vec![-f32::MAX; rates.len()];
    let mut bad = input();
    bad.log_decay = &invalid;
    assert!(core(bad).is_err()); // Finite inputs cannot publish overflowing cumsum.
    let mut visits = 0;
    let mut reject = |stage, _: &[f32]| {
        assert_eq!(stage, ChunkStage::ScaledQuery);
        visits += 1;
        Err(Error::State("observer refuses private computation".into()))
    };
    assert!(core_observed(input(), Some(&mut reject)).is_err());
    assert_eq!(visits, 1);
    assert!(state.iter().all(|v| *v == 3.5));
    assert!(zeros.iter().chain(&rates).all(|v| v.to_bits() == 0));
}

fn label(stage: ChunkStage) -> &'static str {
    match stage {
        ChunkStage::ScaledQuery => "scaled_query",
        ChunkStage::NormalizedKey => "normalized_key",
        ChunkStage::BetaKey => "k_beta",
        ChunkStage::BetaValue => "v_beta",
        ChunkStage::CumulativeDecay => "cum_decay",
        ChunkStage::PairwiseDecay => "pairwise_decay",
        ChunkStage::TriangularSystem => "ut_system",
        ChunkStage::IntraChunkAttention => "intra_chunk_attn",
        ChunkStage::DecayedBetaKey => "decayed_k_beta",
        ChunkStage::NewValues => "new_values",
        ChunkStage::CumulativeKey => "k_cumdecay",
        ChunkStage::DecayedQuery => "decayed_query",
        ChunkStage::DecayedKey => "decayed_key",
        ChunkStage::ChunkDecay => "chunk_decay",
        ChunkStage::ValueCorrection => "v_new",
        ChunkStage::InterChunkAttention => "inter_chunk_attn",
        ChunkStage::CoreF32 => "core_f32",
        ChunkStage::FinalState => "final_state",
        ChunkStage::Output => "output",
    }
}

#[test]
#[ignore = "requires independently captured original HF padded GDN stages"]
fn original_chunk_stages_all_words() {
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        fs::File,
        io::{Read, Seek, SeekFrom},
        path::PathBuf,
    };
    let root = PathBuf::from(std::env::var("ZG_TORCH_CPU_GDN_CHUNK_REFERENCE").unwrap());
    let file = File::open(root.join("gdn.json")).unwrap();
    assert_eq!(file.metadata().unwrap().len(), 189_791);
    let mut json = Vec::new();
    file.take(189_792).read_to_end(&mut json).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&json)),
        "889aeceb1ab2b6861442b6e28fde6443394f58ca94879b2fdf5f12ef7460dff2"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&json).unwrap();
    assert_eq!(manifest["trained_model"], false);
    assert_eq!(manifest["rust_or_gpu_parity"], false);
    let mut raw = File::open(root.join("gdn.bin")).unwrap();
    let before = raw.metadata().unwrap();
    assert_eq!(before.len(), 280_669_088);
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut total = 0;
    loop {
        let n = raw.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        total += n;
    }
    assert_eq!(total, 280_669_088);
    assert_eq!(
        format!("{:x}", hash.finalize()),
        "953da106117d85cbcd4310863a92ec30b39320011443c35f8c4ccf35355ec92b"
    );
    let records = manifest["records"].as_array().unwrap();
    assert_eq!(records.len(), 484);
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 19);
    let mut all_words = 0usize;
    let mut all_records = 0;
    let mut failures = 0usize;
    for (case, settings) in cases.iter().enumerate() {
        let mut expected = BTreeMap::new();
        for record in records
            .iter()
            .filter(|r| r["case"].as_u64().unwrap() == case as u64)
        {
            let n = record["bytes"].as_u64().unwrap() as usize;
            assert!(n <= 16 * 1024 * 1024);
            raw.seek(SeekFrom::Start(record["offset"].as_u64().unwrap()))
                .unwrap();
            let mut bytes = vec![0u8; n];
            raw.read_exact(&mut bytes).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                record["sha256"].as_str().unwrap()
            );
            let values = match record["dtype"].as_str().unwrap() {
                "BF16" => bytes
                    .chunks_exact(2)
                    .map(|b| {
                        f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16)
                    })
                    .collect::<Vec<_>>(),
                "F32" => bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                _ => panic!("original dtype"),
            };
            let label = record["label"].as_str().unwrap().to_owned();
            assert!(expected.insert(label, values).is_none());
        }
        assert_eq!(
            expected.len(),
            settings["records"].as_u64().unwrap() as usize
        );
        let query = expected.remove("input.query").unwrap();
        let key = expected.remove("input.key").unwrap();
        let value = expected.remove("input.value").unwrap();
        let beta = expected.remove("input.beta").unwrap();
        let log_decay = expected.remove("input.decay").unwrap();
        let state = expected.remove("input.state");
        assert_eq!(state.is_some(), settings["initialized"].as_bool().unwrap());
        let mut words = 0;
        let mut fields = 0;
        let mut mismatches = 0;
        let mut compare = |name: &str, actual: &[f32]| {
            let values = expected.remove(name).unwrap();
            assert_eq!(actual.len(), values.len(), "{name}");
            let mut bad = 0;
            let mut first = None;
            for (word, (a, b)) in actual.iter().zip(&values).enumerate() {
                if a.to_bits() != b.to_bits() {
                    bad += 1;
                    first.get_or_insert((word, a.to_bits(), b.to_bits()));
                }
            }
            if bad != 0 {
                println!(
                    "GDN_CHUNK_FIELD case={case} field={name} words={} mismatches={bad} first={first:?}",
                    actual.len()
                );
            }
            words += actual.len();
            fields += 1;
            mismatches += bad;
        };
        let mut observe = |stage, values: &[f32]| {
            compare(label(stage), values);
            Ok(())
        };
        let result = core_observed(
            ChunkInputs {
                source: SourceIdentity::for_new_allocation().unwrap(),
                rows: settings["rows"].as_u64().unwrap() as usize,
                heads: settings["heads"].as_u64().unwrap() as usize,
                query: &query,
                key: &key,
                value: &value,
                beta: &beta,
                log_decay: &log_decay,
                previous_state: state.as_deref(),
            },
            Some(&mut observe),
        );
        match result {
            Ok(result) => compare("returned_state", &result.state),
            Err(error) => {
                println!("GDN_CHUNK_ERROR case={case} error={error}");
                failures += 1;
            }
        }
        println!(
            "GDN_CHUNK_CASE case={case} fields={fields} words={words} mismatches={mismatches} unobserved={}",
            expected.len()
        );
        if !expected.is_empty() || mismatches != 0 {
            failures += 1;
        }
        all_words += words;
        all_records += fields;
    }
    let after = raw.metadata().unwrap();
    assert_eq!(before.len(), after.len());
    assert_eq!(before.modified().unwrap(), after.modified().unwrap());
    println!(
        "GDN_CHUNK_ORIGINAL_ALL cases=19 fields={all_records} words={all_words} failures={failures} trained_model=false native=false"
    );
    assert_eq!(failures, 0);
    assert_eq!(all_records, 380);
}
