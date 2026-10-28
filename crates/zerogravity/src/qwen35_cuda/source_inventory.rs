//! Immutable approved Qwen3.5 source inventories shared by ordinary and
//! qualification loaders. These identify source bytes only; they grant no
//! device, execution, capture, root or publication authority.

use super::*;
use sha2::{Digest, Sha256};

const MIB: usize = 1024 * 1024;

pub(super) struct Spec {
    pub(super) family: &'static str,
    pub(super) repository: &'static str,
    pub(super) revision: &'static str,
    pub(super) config_sha: &'static str,
    pub(super) index_sha: &'static str,
    pub(super) manifest_sha: &'static str,
    pub(super) inventory: &'static str,
    pub(super) hidden: usize,
    pub(super) max_source: u64,
}

static SPECS: [Spec; 2] = [
    Spec {
        family: "0.8b",
        repository: "Qwen/Qwen3.5-0.8B",
        revision: "2fc06364715b967f1860aea9cf38778875588b17",
        config_sha: "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        index_sha: "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
        manifest_sha: "c4c1d349743c0e0ba51ab07f769348e6b13f9896c354303c7af5e839c11dbf09",
        inventory: include_str!("source08.json"),
        hidden: 1024,
        max_source: 2 * 1024 * MIB as u64,
    },
    Spec {
        family: "2b",
        repository: "Qwen/Qwen3.5-2B",
        revision: "15852e8c16360a2fea060d615a32b45270f8a8fc",
        config_sha: "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
        index_sha: "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9",
        manifest_sha: "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
        inventory: include_str!("source2b.json"),
        hidden: 2048,
        max_source: 5 * 1024 * MIB as u64,
    },
];

pub(super) fn select(config: &[u8]) -> Result<&'static Spec> {
    let sha = Sha256::digest(config);
    SPECS
        .iter()
        .find(|spec| sha[..] == digest(spec.config_sha))
        .ok_or_else(|| Error::Unsupported("Qwen CUDA exact approved config/source required".into()))
}

pub(super) fn member(row: &serde_json::Value, small: bool) -> Result<(&str, usize, &str)> {
    let text = |key| {
        row[key]
            .as_str()
            .ok_or_else(|| Error::Malformed("compiled source member".into()))
    };
    let name = text(if small { "name" } else { "path" })?;
    let sha = text(if small { "sha256" } else { "digest" })?;
    let bytes = row["bytes"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| Error::Limit("compiled source member bytes".into()))?;
    Ok((name, bytes, sha))
}

pub(crate) fn digest(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64, "compiled SHA256 literal");
    let mut out = [0; 32];
    for (i, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let digit = |c: u8| match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("compiled SHA256 literal"),
        };
        out[i] = digit(pair[0]) * 16 + digit(pair[1]);
    }
    out
}
