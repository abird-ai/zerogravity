//! Unit-test-only independent original records and pins.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{io::Read, path::PathBuf};
use zerogravity_core::encoding::{ScalarDType, StoredTensorEncoding};

pub(super) struct Archive {
    pub doc: Value,
    raw: Vec<u8>,
}
impl Archive {
    pub fn open() -> Self {
        Self::pinned(
            "ZG_TORCH_CPU_ROWS_REFERENCE",
            "zg.original-torch-cpu-row-operators.v1",
            32,
            "247643fa78ef89df4be1a9db3c26cec4db9c85b6f6829ea213dc9c526bed86ba",
            155_294_128,
            "aff694769db1a4e1c369d00ca9230dfd8b099801affe2cdc406e830e64172637",
        )
    }
    pub fn gdn() -> Self {
        Self::pinned(
            "ZG_TORCH_CPU_GDN_ROWS_REFERENCE",
            "zg.original-torch-cpu-gdn-layer-rows.v1",
            6,
            "d474adf31d971fcfeee17ee1851e4dec2215afdcad90bb7451bba6f969da871e",
            25_501_888,
            "6aa2f7164e68290ab083a750245e65cf1a57031e75ef787a65dcc5b356172686",
        )
    }
    fn pinned(
        env: &str,
        schema: &str,
        calls: usize,
        manifest_hash: &str,
        raw_bytes: usize,
        raw_hash: &str,
    ) -> Self {
        let dir = PathBuf::from(std::env::var(env).expect("independent original operator archive"));
        let mut manifest = Vec::new();
        std::fs::File::open(dir.join("operators.json"))
            .unwrap()
            .take(256 * 1024 + 1)
            .read_to_end(&mut manifest)
            .unwrap();
        assert!(manifest.len() <= 256 * 1024);
        assert_eq!(format!("{:x}", Sha256::digest(&manifest)), manifest_hash);
        let doc: Value = serde_json::from_slice(&manifest).unwrap();
        assert_eq!(doc["schema"], schema);
        assert_eq!(doc["status"], "ORIGINAL_REFERENCE_ONLY");
        assert_eq!(doc["rust_or_gpu_parity"], false);
        assert_eq!(
            doc["modeling_sha256"],
            "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
        );
        assert_eq!(doc["calls"].as_array().unwrap().len(), calls);
        let file = std::fs::File::open(dir.join("operators.bin")).unwrap();
        assert_eq!(file.metadata().unwrap().len(), raw_bytes as u64);
        let mut raw = Vec::new();
        file.take(raw_bytes as u64 + 1)
            .read_to_end(&mut raw)
            .unwrap();
        assert_eq!(raw.len(), raw_bytes);
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), raw_hash);
        let mut end = 0;
        for r in doc["records"].as_array().unwrap() {
            assert_eq!(number(r, "offset"), end);
            let words: usize = r["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .product();
            let bytes = match r["dtype"].as_str().unwrap() {
                "torch.bfloat16" => 2,
                "torch.float32" => 4,
                other => panic!("unknown original dtype {other}"),
            };
            assert_eq!(number(r, "bytes"), words * bytes);
            let data = &raw[end..end + words * bytes];
            assert_eq!(
                format!("{:x}", Sha256::digest(data)),
                r["sha256"].as_str().unwrap()
            );
            end += data.len();
        }
        assert_eq!(end, raw.len());
        Self { doc, raw }
    }
    fn record(&self, case: usize, label: &str) -> &Value {
        let records = self.doc["records"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| number(v, "case") == case && v["label"] == label)
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 1, "case={case} label={label}");
        records[0]
    }
    pub fn bytes(&self, case: usize, label: &str) -> &[u8] {
        let r = self.record(case, label);
        let at = number(r, "offset");
        &self.raw[at..at + number(r, "bytes")]
    }
    pub fn shape(&self, case: usize, label: &str) -> Vec<u64> {
        self.record(case, label)["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect()
    }
    pub fn kind(&self, case: usize, label: &str) -> StoredTensorEncoding {
        StoredTensorEncoding::Scalar(match self.record(case, label)["dtype"].as_str().unwrap() {
            "torch.bfloat16" => ScalarDType::Bf16,
            "torch.float32" => ScalarDType::F32,
            _ => unreachable!(),
        })
    }
    pub fn values(&self, case: usize, label: &str) -> Vec<f32> {
        if self.kind(case, label) == StoredTensorEncoding::Scalar(ScalarDType::Bf16) {
            self.bytes(case, label)
                .chunks_exact(2)
                .map(|v| f32::from_bits(u32::from(u16::from_le_bytes(v.try_into().unwrap())) << 16))
                .collect()
        } else {
            self.bytes(case, label)
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        }
    }
    pub fn compare(&self, case: usize, label: &str, actual: &[f32]) -> usize {
        let expected = self.values(case, label);
        assert_eq!(actual.len(), expected.len());
        for (word, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "case={case} stage={label} word={word}"
            );
        }
        actual.len()
    }
}
pub(super) fn number(v: &Value, key: &str) -> usize {
    usize::try_from(v[key].as_u64().unwrap()).unwrap()
}
