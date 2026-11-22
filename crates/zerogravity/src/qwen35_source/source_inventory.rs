//! Immutable approved Qwen3.5 source inventories shared by ordinary and
//! qualification loaders. These identify source bytes only; they grant no
//! device, execution, capture, root or publication authority.

use crate::{Error, Result};
use sha2::{Digest, Sha256};

const MIB: usize = 1024 * 1024;

pub(crate) struct Spec {
    pub(crate) family: &'static str,
    pub(crate) repository: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) config_sha: &'static str,
    pub(crate) index_sha: &'static str,
    pub(crate) manifest_sha: &'static str,
    pub(crate) inventory: &'static str,
    pub(crate) hidden: usize,
    pub(crate) member_count: usize,
    pub(crate) shard_count: usize,
    pub(crate) tensor_limit: usize,
    pub(crate) named_members: bool,
    pub(crate) max_source: u64,
    #[cfg(any(
        all(feature = "model-qwen35", feature = "hip"),
        all(feature = "model-qwen35", feature = "cuda")
    ))]
    pub(crate) snapshot_sha: &'static str,
}

static SPECS: &[Spec] = &[
    Spec {
        family: "0.8b",
        #[cfg(any(
            all(feature = "model-qwen35", feature = "hip"),
            all(feature = "model-qwen35", feature = "cuda")
        ))]
        snapshot_sha: "d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c",
        repository: "Qwen/Qwen3.5-0.8B",
        revision: "2fc06364715b967f1860aea9cf38778875588b17",
        config_sha: "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        index_sha: "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
        manifest_sha: "c4c1d349743c0e0ba51ab07f769348e6b13f9896c354303c7af5e839c11dbf09",
        inventory: include_str!("../qwen35_cuda/source08.json"),
        hidden: 1024,
        member_count: 11,
        shard_count: 1,
        tensor_limit: 1024,
        named_members: true,
        max_source: 2 * 1024 * MIB as u64,
    },
    Spec {
        family: "2b",
        #[cfg(any(
            all(feature = "model-qwen35", feature = "hip"),
            all(feature = "model-qwen35", feature = "cuda")
        ))]
        snapshot_sha: "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
        repository: "Qwen/Qwen3.5-2B",
        revision: "15852e8c16360a2fea060d615a32b45270f8a8fc",
        config_sha: "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
        index_sha: "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9",
        manifest_sha: "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
        inventory: include_str!("../qwen35_cuda/source2b.json"),
        hidden: 2048,
        member_count: 12,
        shard_count: 1,
        tensor_limit: 1024,
        named_members: false,
        max_source: 5 * 1024 * MIB as u64,
    },
    Spec {
        family: "27b",
        #[cfg(any(
            all(feature = "model-qwen35", feature = "hip"),
            all(feature = "model-qwen35", feature = "cuda")
        ))]
        snapshot_sha: "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
        repository: "Qwen/Qwen3.8-27B",
        revision: "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        config_sha: "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
        index_sha: "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
        manifest_sha: "9b52f365bc061fd9a57223f551eac586b83dad2d6a9690efbd6c0bb136a039ce",
        inventory: include_str!("../qwen27b_gpu/source.json"),
        hidden: 5120,
        member_count: 31,
        shard_count: 18,
        tensor_limit: 2048,
        named_members: true,
        max_source: 64 * 1024 * MIB as u64,
    },
];

#[cfg(any(
    all(feature = "model-qwen35", feature = "hip"),
    all(feature = "model-qwen35", feature = "cuda")
))]
pub(crate) fn snapshot_for_target(target: &str) -> Option<&'static str> {
    let family = match target {
        "qwen3.5-0.8b" => "0.8b",
        "qwen3.5-2b" => "2b",
        "qwen3.8-27b" => "27b",
        _ => return None,
    };
    SPECS
        .iter()
        .find(|spec| spec.family == family)
        .map(|spec| spec.snapshot_sha)
}

pub(crate) fn select(config: &[u8]) -> Result<&'static Spec> {
    let sha = Sha256::digest(config);
    SPECS
        .iter()
        .find(|spec| sha[..] == digest(spec.config_sha))
        .ok_or_else(|| Error::Unsupported("Qwen exact approved config/source required".into()))
}

pub(crate) fn member(row: &serde_json::Value, named: bool) -> Result<(&str, usize, &str)> {
    let text = |key| {
        row[key]
            .as_str()
            .ok_or_else(|| Error::Malformed("compiled source member".into()))
    };
    let name = text(if named { "name" } else { "path" })?;
    let sha = text(if named { "sha256" } else { "digest" })?;
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
