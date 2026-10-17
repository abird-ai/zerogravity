//! Xet protocol1.1 hash namespace, not a raw content SHA256.
//! Reference: xet-core c3c726bed5cf54ded92a63fab892cfb7857c751a,
//! merklehash/{data_hash,aggregated_hashes}.rs. No platform endianness casts.
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) const MAX_HASH_CHUNKS: usize = 1_048_576;
pub(crate) const MAX_XORB_CHUNKS: usize = 8192;
const MAX_XORB_RAW_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CHUNK_BYTES: usize = 128 * 1024;
const DATA_KEY: [u8; 32] = [
    102, 151, 245, 119, 91, 149, 80, 222, 49, 53, 203, 172, 165, 151, 24, 28, 157, 228, 33, 16,
    155, 235, 43, 88, 180, 208, 176, 75, 147, 173, 242, 41,
];
const INTERNAL_KEY: [u8; 32] = [
    1, 126, 197, 199, 165, 71, 41, 150, 253, 148, 102, 102, 180, 138, 2, 230, 93, 221, 83, 111, 55,
    199, 109, 210, 248, 99, 82, 230, 74, 83, 113, 63,
];
const VERIFICATION_KEY: [u8; 32] = [
    127, 24, 87, 214, 206, 86, 237, 102, 18, 127, 249, 19, 231, 165, 195, 243, 164, 205, 38, 213,
    181, 219, 73, 230, 65, 36, 152, 127, 40, 251, 148, 195,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct XetHash([u8; 32]);
impl XetHash {
    pub(crate) const ZERO: Self = Self([0; 32]);
    pub(crate) const fn from_bytes(raw: [u8; 32]) -> Self {
        Self(raw)
    }
    pub(crate) fn raw(&self) -> &[u8; 32] {
        &self.0
    }
    pub(crate) fn from_text(value: &str) -> Result<Self> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::Malformed("invalid canonical Xet hash".into()));
        }
        let mut raw = [0u8; 32];
        for (word, text) in raw
            .chunks_exact_mut(8)
            .zip(value.as_bytes().chunks_exact(16))
        {
            for i in 0..8 {
                let hex = &text[i * 2..i * 2 + 2];
                word[7 - i] = (nibble(hex[0]) << 4) | nibble(hex[1]);
            }
        }
        Ok(Self(raw))
    }
    fn text_bytes(self) -> [u8; 64] {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = [0u8; 64];
        for (word, text) in self.0.chunks_exact(8).zip(output.chunks_exact_mut(16)) {
            for (i, byte) in word.iter().rev().enumerate() {
                text[i * 2] = HEX[(byte >> 4) as usize];
                text[i * 2 + 1] = HEX[(byte & 15) as usize];
            }
        }
        output
    }
    pub(crate) fn text(self) -> String {
        self.text_bytes().iter().map(|b| char::from(*b)).collect()
    }
    fn natural_cut(self) -> bool {
        // Reference DataHash %4 uses the fourth little-endian u64 limb.
        self.0[24] & 3 == 0
    }
}
fn nibble(byte: u8) -> u8 {
    if byte.is_ascii_digit() {
        byte - b'0'
    } else {
        byte - b'a' + 10
    }
}
fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::State("Xet hashing canceled".into()))
    } else {
        Ok(())
    }
}

pub(crate) fn chunk_hash(raw: &[u8], cancel: &AtomicBool) -> Result<XetHash> {
    if raw.len() > MAX_CHUNK_BYTES {
        return Err(Error::Limit("Xet raw chunk bound".into()));
    }
    let mut hasher = blake3::Hasher::new_keyed(&DATA_KEY);
    for chunk in raw.chunks(4096) {
        check_cancel(cancel)?;
        hasher.update(chunk);
    }
    check_cancel(cancel)?;
    Ok(XetHash(*hasher.finalize().as_bytes()))
}

/// Reuses the caller's admitted hash/length vector in place; no second vector.
/// All input entries are consumed and can no longer be used as original leaves.
pub(crate) fn xorb_hash(chunks: &mut [(XetHash, u64)], cancel: &AtomicBool) -> Result<XetHash> {
    if chunks.len() > MAX_XORB_CHUNKS {
        return Err(Error::Limit("Xet xorb chunk count bound".into()));
    }
    let mut total = 0u64;
    for (_, size) in chunks.iter() {
        check_cancel(cancel)?;
        total = total
            .checked_add(*size)
            .filter(|total| *total <= MAX_XORB_RAW_BYTES)
            .ok_or_else(|| Error::Limit("Xet xorb raw byte bound".into()))?;
    }
    aggregate(chunks, cancel)
}
fn aggregate(chunks: &mut [(XetHash, u64)], cancel: &AtomicBool) -> Result<XetHash> {
    if chunks.len() > MAX_HASH_CHUNKS {
        return Err(Error::Limit("Xet chunk hash count bound".into()));
    }
    let mut total = 0u64;
    for (_, size) in chunks.iter() {
        check_cancel(cancel)?;
        if *size > MAX_CHUNK_BYTES as u64 {
            return Err(Error::Limit("Xet raw chunk length bound".into()));
        }
        total = total
            .checked_add(*size)
            .ok_or_else(|| Error::Limit("Xet length overflow".into()))?;
    }
    let mut used = chunks.len();
    while used > 1 {
        let mut read = 0;
        let mut written = 0;
        while read < used {
            check_cancel(cancel)?;
            let remaining = &chunks[read..used];
            let end = remaining.len().min(9);
            let cut = (2..end)
                .find(|i| remaining[*i].0.natural_cut())
                .map_or(end, |i| i + 1);
            chunks[written] = merge(&remaining[..cut])?;
            written += 1;
            read += cut;
        }
        used = written;
    }
    check_cancel(cancel)?;
    let _ = total; // Checked even when a single input node bypasses merging.
    Ok(chunks.first().map_or(XetHash::ZERO, |entry| entry.0))
}
fn merge(children: &[(XetHash, u64)]) -> Result<(XetHash, u64)> {
    let mut hasher = blake3::Hasher::new_keyed(&INTERNAL_KEY);
    let mut total = 0u64;
    for (hash, size) in children {
        hasher.update(&hash.text_bytes());
        hasher.update(b" : ");
        let mut decimal = [0u8; 20];
        let mut at = decimal.len();
        let mut value = *size;
        loop {
            at -= 1;
            decimal[at] = b'0' + (value % 10) as u8;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        hasher.update(&decimal[at..]);
        hasher.update(b"\n");
        total = total
            .checked_add(*size)
            .ok_or_else(|| Error::Limit("Xet aggregate length overflow".into()))?;
    }
    Ok((XetHash(*hasher.finalize().as_bytes()), total))
}
pub(crate) fn file_hash(chunks: &mut [(XetHash, u64)], cancel: &AtomicBool) -> Result<XetHash> {
    let empty = chunks.is_empty();
    let root = aggregate(chunks, cancel)?;
    check_cancel(cancel)?;
    Ok(if empty {
        XetHash::ZERO
    } else {
        XetHash(*blake3::keyed_hash(&[0; 32], root.raw()).as_bytes())
    })
}

/// Shard/upload verification primitive. Download responses do not contain an
/// expected value, so merely calculating it does not authenticate a term.
pub(crate) fn verification_hash(chunks: &[XetHash], cancel: &AtomicBool) -> Result<XetHash> {
    if chunks.len() > MAX_HASH_CHUNKS {
        return Err(Error::Limit("Xet verification hash count bound".into()));
    }
    let mut hasher = blake3::Hasher::new_keyed(&VERIFICATION_KEY);
    for hash in chunks {
        check_cancel(cancel)?;
        hasher.update(hash.raw());
    }
    check_cancel(cancel)?;
    Ok(XetHash(*hasher.finalize().as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    const A: &str = "cfc5d07f6f03c29bbf424132963fe08d19a37d5757aaf520bf08119f05cd56d6";
    const B: &str = "c3e67584b5c4fc2a89837ec39e40f2c8a6bb0b2987ac94cd4b31e5fbdd210a72";
    const C: &str = "0d2beb91b9196929a5ddec9f6e306924ddf4a24268e3e59fd8464738d525af37";
    const D: &str = "adf8773496a9b7319b2e50dc98093f344053b17d8ad37100b9c07d9805988784";
    const E: &str = "4ac202caf347fc1e9c874b1ef6a1c5e619141eb775a6f43f0f0124ccd0060d9e";
    const F: &str = "b3b28636f65c149ea52eb1f94669466f70f033b54cea792824c696ba6ef3c389";
    const G: &str = "0e2c1a002aae913d2c0fc8ddfa4e9e14b7b311b3b0d458726d5d9f6a6318013c";
    fn entry(hash: &str, size: u64) -> (XetHash, u64) {
        (XetHash::from_text(hash).unwrap(), size)
    }

    #[test]
    fn official_text_limb_endianness_is_separate_from_raw_digest_hex() {
        let raw = std::array::from_fn(|i| i as u8);
        let hash = XetHash::from_bytes(raw);
        assert_eq!(
            hash.text(),
            "07060504030201000f0e0d0c0b0a090817161514131211101f1e1d1c1b1a1918"
        );
        assert_eq!(XetHash::from_text(&hash.text()).unwrap().raw(), &raw);
        // Saved data_hash.rs reference vector, distinct from the docs' short
        // 0..31 text example which accidentally omits byte0x10.
        let saved = XetHash::from_bytes([
            22, 175, 58, 132, 4, 75, 131, 214, 190, 153, 138, 66, 226, 3, 153, 242, 204, 86, 80,
            234, 249, 153, 80, 99, 159, 80, 65, 138, 236, 231, 149, 78,
        ]);
        assert_eq!(
            saved.text(),
            "d6834b04843aaf16f29903e2428a99be635099f9ea5056cc4e95e7ec8a41509f"
        );
        for bad in [
            "a".repeat(63),
            "A".repeat(64),
            "g".repeat(64),
            "é".repeat(32),
        ] {
            assert!(XetHash::from_text(&bad).is_err());
        }
    }
    #[test]
    fn frozen_independent_keyed_chunk_file_and_verification_namespaces() {
        // Frozen BEFORE Rust implementation by the independent scalar Python
        // one-chunk reference; its four official BLAKE3 vector assertions pass.
        for (raw, chunk, file) in [
            (
                b"".as_slice(),
                "e0f2cf784e7e5f10c34f84af150e9a5ff9664216debad915364d741049870f67",
                "c5b8746ae7e55c74d71e80fd99d34bf9a2c5666c690f1c113dec6d292790488a",
            ),
            (
                b"abc",
                "a8803506f7aff014f19aa9e9b79dc6455c49672570f96156071329698dd0284e",
                "55b9df318a7c45f470d57b04d8ca79a15b0637ffe94b25041a58d00c3528afff",
            ),
            (
                b"hello world",
                "4e39378b9d359f118190557a1f44130219a54e5fcfa07bd96cf50b466fe651b1",
                "f15edd43b7e85cfd25e2758ffe13b99fdd09dfa817e8d9f6548722a7a126db7f",
            ),
        ] {
            let hash = chunk_hash(raw, &AtomicBool::new(false)).unwrap();
            assert_eq!(hash.text(), chunk);
            assert_eq!(
                file_hash(&mut [(hash, raw.len() as u64)], &AtomicBool::new(false))
                    .unwrap()
                    .text(),
                file
            );
        }
        assert_eq!(
            verification_hash(&[], &AtomicBool::new(false))
                .unwrap()
                .text(),
            "b81ce0cc7ce434c719c5f31801bc0825ead2a9ebae7419605afc00726961e5d9"
        );
        let raw0 = XetHash::from_bytes(std::array::from_fn(|i| i as u8));
        let raw1 = XetHash::from_bytes(std::array::from_fn(|i| i as u8 + 32));
        assert_eq!(
            verification_hash(&[raw0], &AtomicBool::new(false))
                .unwrap()
                .text(),
            "bdd672a821699fbd0eb1e8a00c01af5d119af5bc63603caab0690320cdfa0d06"
        );
        assert_eq!(
            verification_hash(&[raw0, raw1], &AtomicBool::new(false))
                .unwrap()
                .text(),
            "480e1ea1efee72b9a12da972653300195ef445830f37af2389bb9db00d7e8c09"
        );
    }
    #[test]
    fn official_saved_merkle_vectors_cover_empty_single_repeated_cuts_and_tail() {
        // Literal reference cases from aggregated_hashes.rs at the pinned commit.
        let cases = [
            (
                vec![],
                "0000000000000000000000000000000000000000000000000000000000000000",
                "0000000000000000000000000000000000000000000000000000000000000000",
            ),
            (
                vec![entry(A, 100)],
                A,
                "8e16257caa3fe079d484d872a8975264b2ff683b0d6db9028cc7c0f968a45661",
            ),
            (
                vec![entry(A, 100), entry(B, 200), entry(C, 300)],
                "71ec1275fca074724e2dd666921b3277c7cee603e4d025bcab2d4050015be2bc",
                "54e55dccc6653c612bdb5576c5d3cb34bb31bc4e100248abccf4c908b3ef7715",
            ),
            (
                vec![entry(A, 100); 4],
                "89f2ada89ff8c96763c6b25010e6dd76a4c05b1466207633ea559acf2093211b",
                "2cdba690d0e09563596e0cda626d43eb4c96ef1e994fe72d9b2f5a83cfcd36dd",
            ),
            (
                vec![entry(A, 100), entry(B, 200), entry(A, 100), entry(B, 200)],
                "90f8313ef12df385d237a067aded02562c35ded12116e32eba401dbc86c38031",
                "284ea045e5a579e99c21ec597c20de1fc0c09e7168162aac00db8f61b3d82dbb",
            ),
            (
                vec![entry(C, 300), entry(D, 400), entry(E, 500), entry(C, 300)],
                "73b9dc802ab03a0855c66823a18d95c2a785af594535756abf405038b6270cb8",
                "c7d5be75ef5ec12888027ad305141e4109e599494477661177475c84e12de48b",
            ),
            (
                vec![
                    entry(C, 300),
                    entry(D, 400),
                    entry(E, 500),
                    entry(E, 500),
                    entry(E, 500),
                ],
                "2c1fd1eca47e284512fcb40fcf45a1874939f9b91c774d46aeba5d004d9d263d",
                "0799ba7ee2c3fed173a8e8d832cd13970ddc16070ffbc8aa6dcd77476f66dc73",
            ),
            (
                vec![entry(A, 100); 32],
                "0a0123c1617921883b7e13902095fcb86676e77c49120c33b233003b0af0e0a6",
                "53af4711fd1d5e5bdc7f931b6be932314d8d673cb16ad2482f6f5222eaf9e63d",
            ),
        ];
        let start = vec![
            (XetHash::ZERO, 0),
            entry(A, 100),
            entry(B, 200),
            entry(C, 300),
            entry(D, 400),
            entry(E, 500),
            entry(F, 600),
            entry(G, 700),
        ];
        let mut tail = start.clone();
        tail.extend([entry(A, 100); 4]);
        let extra = [
            (
                start,
                "f62abe77e3fb9c954fe52b0028027ddc90c064c45951a4fd2211d87e5c0011db",
                "d1b068be5bbdb38992269e8efe61f601881e39f7a7585fd76883cc6ea5c23b44",
            ),
            (
                tail,
                "e7fc30b73ec9930593f80fe777b334e9d3bb73e45e6a79fa5784b2603446158b",
                "69e1af2fd6023ce133e0d1b90e32d1d42ef79f5a710f2b0accfc3a735e98de9d",
            ),
        ];
        for (mut entries, xorb, file) in cases.into_iter().chain(extra) {
            assert_eq!(
                xorb_hash(&mut entries.clone(), &AtomicBool::new(false))
                    .unwrap()
                    .text(),
                xorb
            );
            assert_eq!(
                file_hash(&mut entries, &AtomicBool::new(false))
                    .unwrap()
                    .text(),
                file
            );
        }
    }
    #[test]
    fn hashing_refuses_unadmitted_shapes_and_observes_cancellation() {
        assert!(chunk_hash(&vec![0; MAX_CHUNK_BYTES + 1], &AtomicBool::new(false)).is_err());
        assert!(
            xorb_hash(
                &mut vec![(XetHash::ZERO, 0); MAX_XORB_CHUNKS + 1],
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(
            xorb_hash(
                &mut vec![(XetHash::ZERO, MAX_CHUNK_BYTES as u64); 513],
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(chunk_hash(b"", &AtomicBool::new(true)).is_err());
        assert!(xorb_hash(&mut [], &AtomicBool::new(true)).is_err());
        assert!(verification_hash(&[], &AtomicBool::new(true)).is_err());
        assert!(
            xorb_hash(
                &mut [(XetHash::ZERO, MAX_CHUNK_BYTES as u64 + 1)],
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
}
