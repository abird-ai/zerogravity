use super::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    sync::atomic::{AtomicU64, Ordering},
};
use zerogravity_artifacts::cache::{
    CacheOptions, CacheStore, CommitOutcome, FileRole, Manifest, ManifestEntry, ManifestSource,
};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

fn config() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("config.json",json!({"model_type":"bert","architectures":["BertModel"],"hidden_act":"gelu",
            "hidden_size":4,"intermediate_size":6,"num_attention_heads":2,"num_hidden_layers":2,
            "vocab_size":7,"max_position_embeddings":8,"type_vocab_size":2,"layer_norm_eps":1e-5,
            "pad_token_id":0,"position_embedding_type":"absolute","use_cache":true}).to_string().into_bytes()),
        ("sentence_bert_config.json",json!({"max_seq_length":5,"do_lower_case":false}).to_string().into_bytes()),
        ("modules.json",json!([
            {"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
            {"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"},
            {"idx":2,"name":"2","path":"2_Normalize","type":"sentence_transformers.models.Normalize"}
        ]).to_string().into_bytes()),
        ("vocab.txt",b"[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n##s\n".to_vec()),
        ("tokenizer_config.json",json!({
            "tokenizer_class":"BertTokenizer","do_lower_case":true,"do_basic_tokenize":true,
            "tokenize_chinese_chars":true,"strip_accents":null,"never_split":null,
            "model_max_length":8,"unk_token":"[UNK]","sep_token":"[SEP]","pad_token":"[PAD]",
            "cls_token":"[CLS]","mask_token":"[MASK]","name_or_path":"fixture"
        }).to_string().into_bytes()),
        ("1_Pooling/config.json",json!({"word_embedding_dimension":4,"pooling_mode_cls_token":false,"pooling_mode_mean_tokens":true,
            "pooling_mode_max_tokens":false,"pooling_mode_mean_sqrt_len_tokens":false}).to_string().into_bytes()),
    ]
}
struct Tensors {
    header: serde_json::Map<String, serde_json::Value>,
    payload: Vec<u8>,
    seed: usize,
}
impl Tensors {
    fn new() -> Self {
        Self {
            header: Default::default(),
            payload: Vec::new(),
            seed: 1,
        }
    }
    fn add(&mut self, name: &str, shape: &[usize], scale: f32, offset: f32) {
        let start = self.payload.len();
        for i in 0..shape.iter().product() {
            let value = offset + scale * (((i * 17 + self.seed * 13) % 29) as f32 - 14.);
            self.payload.extend(value.to_le_bytes());
        }
        self.seed += 1;
        self.header.insert(
            name.to_owned(),
            json!({"dtype":"F32","shape":shape,"data_offsets":[start,self.payload.len()]}),
        );
    }
    fn linear(&mut self, name: &str, input: usize, out: usize) {
        self.add(&format!("{name}.weight"), &[out, input], 0.03, 0.);
        self.add(&format!("{name}.bias"), &[out], 0.01, 0.);
    }
    fn norm(&mut self, name: &str) {
        self.add(&format!("{name}.weight"), &[4], 0.01, 1.);
        self.add(&format!("{name}.bias"), &[4], 0.01, 0.);
    }
    fn encode(self) -> Vec<u8> {
        let mut header = serde_json::to_vec(&self.header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(self.payload);
        file
    }
}
fn weights(bad_index: bool, missing: bool) -> Vec<u8> {
    let mut t = Tensors::new();
    t.add("embeddings.word_embeddings.weight", &[7, 4], 0.05, 0.);
    t.add("embeddings.position_embeddings.weight", &[8, 4], 0.02, 0.);
    t.add("embeddings.token_type_embeddings.weight", &[2, 4], 0.01, 0.);
    t.norm("embeddings.LayerNorm");
    for i in 0..2 {
        let p = format!("encoder.layer.{i}");
        for role in ["query", "key", "value"] {
            t.linear(&format!("{p}.attention.self.{role}"), 4, 4);
        }
        t.linear(&format!("{p}.attention.output.dense"), 4, 4);
        t.norm(&format!("{p}.attention.output.LayerNorm"));
        t.linear(&format!("{p}.intermediate.dense"), 4, 6);
        t.linear(&format!("{p}.output.dense"), 6, 4);
        t.norm(&format!("{p}.output.LayerNorm"));
    }
    let start = t.payload.len();
    for i in 0..8i64 {
        t.payload
            .extend((if bad_index && i == 4 { 0 } else { i }).to_le_bytes());
    }
    t.header.insert(
        "embeddings.position_ids".into(),
        json!({"dtype":"I64","shape":[1,8],"data_offsets":[start,t.payload.len()]}),
    );
    // Change name while preserving full byte closure.
    if missing {
        let old = t
            .header
            .remove("encoder.layer.1.output.dense.bias")
            .unwrap();
        t.header.insert("unrecognized.bias".into(), old);
    }
    t.encode()
}

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        use std::os::unix::fs::DirBuilderExt;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = std::env::temp_dir().join(format!(
            "zg-bert-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
        Self(path)
    }
    fn snapshot(
        &self,
        files: Vec<(&'static str, Vec<u8>)>,
        source: Vec<u8>,
        admission: &TransferAdmission,
    ) -> (CacheStore, SnapshotLease) {
        let mut files = files;
        files.push(("model.safetensors", source));
        let store = CacheStore::open(CacheOptions::new(
            Some(self.0.join("cache")),
            8 * 1024 * 1024,
        ))
        .unwrap();
        let manifest = Manifest::new(
            ManifestSource::new("local", "synthetic-bert", "v1").unwrap(),
            files
                .iter()
                .map(|(name, bytes)| {
                    ManifestEntry::new(
                        *name,
                        format!("{:x}", Sha256::digest(bytes)),
                        bytes.len() as u64,
                        if *name == "model.safetensors" {
                            FileRole::Weights
                        } else if matches!(*name, "vocab.txt" | "tokenizer_config.json") {
                            FileRole::Tokenizer
                        } else {
                            FileRole::Config
                        },
                    )
                    .unwrap()
                })
                .collect(),
            &Default::default(),
        )
        .unwrap();
        let digest = manifest.digest().unwrap();
        let mut transaction = store
            .begin(manifest, admission, Arc::new(AtomicBool::new(false)))
            .unwrap();
        for (name, bytes) in &files {
            transaction
                .import_object(name, &mut bytes.as_slice())
                .unwrap();
        }
        assert!(matches!(
            transaction.commit(None).unwrap(),
            CommitOutcome::Durable(_)
        ));
        let lease = store.open_snapshot(&digest).unwrap();
        (store, lease)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn admission() -> TransferAdmission {
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("test-host", "bert").unwrap();
    arbiter
        .register(resource.clone(), 256 * 1024 * 1024)
        .unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("fixture", "encoder").unwrap(),
    }
}
fn reserved(a: &TransferAdmission) -> u64 {
    a.arbiter.snapshot(&a.resource).unwrap().reserved
}

#[test]
fn complete_source_bound_forward_and_result_retirement() {
    let a = admission();
    let f = Fixture::new();
    let (_store, snapshot) = f.snapshot(config(), weights(false, false), &a);
    let baseline = reserved(&a);
    let model = BertSentenceEncoder::load(&snapshot, &a).unwrap();
    assert_eq!(model.config().dimensions(), 4);
    assert!(model.source().is_bound());
    assert!(model.source.verified_single_sha256().is_some());
    assert!(model.source.verified_bundle_sha256().is_none());
    assert!(model.source.verified_shard_index_sha256().is_none());
    let retained = reserved(&a);
    let text = model.embed_text("HÉLLOS", None).unwrap();
    let ids = model.embed_tokens(&[2, 5, 6, 3], None).unwrap();
    assert_eq!(text.tokens(), 4);
    assert_eq!(text.values(), ids.values());
    drop((text, ids));
    assert_eq!(reserved(&a), retained);
    assert!(model.embed_text("hello hello hello hello", None).is_err());
    assert!(model.embed_text(" ", None).is_err());
    let first = model.embed_tokens(&[1, 2, 3], None).unwrap();
    let second = model.embed_tokens(&[1, 2, 4], None).unwrap();
    assert_eq!(first.tokens(), 3);
    assert_eq!(first.source(), model.source());
    assert_eq!(first.values().len(), 4);
    // Independent numeric oracle, generated using Python's scalar
    // math/erf implementation of the original BERT forward graph.
    let expected1 = [0.33089855f64, -0.59064984, 0.56863701, -0.46721610];
    let expected2 = [0.69847023f64, -0.07389471, -0.71059227, 0.04168299];
    for (actual, expected) in [(first.values(), expected1), (second.values(), expected2)] {
        for (&a, b) in actual.iter().zip(expected) {
            assert!(
                (f64::from(a) - b).abs() < 2e-5,
                "{actual:?} expected {expected:?}"
            );
        }
        assert!((actual.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-6);
    }
    assert_ne!(first.values(), second.values());
    assert_eq!(reserved(&a), retained + 2 * 4 * 4);
    for input in [&[][..], &[7][..], &[1, 2, 3, 4, 5, 6][..]] {
        assert!(model.embed_tokens(input, None).is_err());
    }
    let cancelled = AtomicBool::new(true);
    assert!(model.embed_tokens(&[1], Some(&cancelled)).is_err());
    assert_eq!(reserved(&a), retained + 2 * 4 * 4);
    let pressure = reserve(
        &a,
        AllocationClass::Scratch,
        a.arbiter.snapshot(&a.resource).unwrap().remaining,
    )
    .unwrap();
    assert!(matches!(
        model.embed_tokens(&[1], None),
        Err(Error::Resource(_))
    ));
    drop(pressure);
    drop(model);
    assert_eq!(reserved(&a), baseline + 2 * 4 * 4);
    assert_eq!(first.source(), second.source());
    drop((first, second));
    assert_eq!(reserved(&a), baseline);
}

#[test]
fn refuse_bad_source_and_pooling_without_retained_charges() {
    for (bad_index, missing, bad_pool) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        let a = admission();
        let f = Fixture::new();
        let mut files = config();
        if bad_pool {
            let p = files.last_mut().unwrap();
            let mut value: serde_json::Value = serde_json::from_slice(&p.1).unwrap();
            value["pooling_mode_mean_tokens"] = json!(false);
            value["pooling_mode_cls_token"] = json!(true);
            p.1 = value.to_string().into_bytes();
        }
        let (_store, snapshot) = f.snapshot(files, weights(bad_index, missing), &a);
        let baseline = reserved(&a);
        assert!(BertSentenceEncoder::load(&snapshot, &a).is_err());
        assert_eq!(reserved(&a), baseline);
    }
}
