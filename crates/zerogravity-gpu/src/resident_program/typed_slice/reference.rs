//! Test-only original Torch archive. Opened solely after actual completion.
use super::*;
use std::{io::Read, os::unix::fs::MetadataExt, path::Path};

const STAGES: [&str; 11] = [
    "embedding",
    "raw_gamma",
    "runtime_gamma",
    "squares",
    "mean",
    "denominator",
    "inverse",
    "normalized",
    "weighted",
    "rms_bf16",
    "projection_bf16",
];
const MAX: usize = 1024 * 1024;

fn counts(width: usize, rows: usize) -> [usize; 11] {
    [
        width, width, width, width, 1, 1, 1, width, width, width, rows,
    ]
}
fn parse<'a>(
    raw: &'a [u8],
    source: &str,
    width: usize,
    rows: usize,
    token: u32,
) -> Result<Vec<(&'static str, &'a [u8])>> {
    if !matches!(width, 1024 | 2048)
        || rows == 0
        || rows > 16384
        || raw.len() < 56
        || raw.len() > MAX
        || &raw[..8] != b"ZGBF16V1"
        || source.len() != 64
        || raw[8..40]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
            != source
    {
        return Err(Error::State(
            "portable original reference header/source".into(),
        ));
    }
    let number = |at| u32::from_le_bytes(raw[at..at + 4].try_into().expect("bounded header"));
    if [number(40), number(44), number(48), number(52)] != [width as u32, rows as u32, token, 11] {
        return Err(Error::State(
            "portable original reference geometry/token/stages".into(),
        ));
    }
    let mut at = 56usize;
    let mut result = Vec::with_capacity(11);
    for (name, words) in STAGES.into_iter().zip(counts(width, rows)) {
        let length = raw
            .get(at..at + 4)
            .ok_or_else(|| Error::State("reference length absent".into()))?;
        let bytes = u32::from_le_bytes(length.try_into().expect("bounded length")) as usize;
        at += 4;
        if bytes != words * 4 {
            return Err(Error::State("reference full defined extent".into()));
        }
        let payload = raw
            .get(at..at + bytes)
            .ok_or_else(|| Error::State("reference payload absent".into()))?;
        for word in payload.chunks_exact(4) {
            let bits = u32::from_le_bytes(word.try_into().expect("whole F32 word"));
            if !f32::from_bits(bits).is_finite()
                || matches!(name, "embedding" | "rms_bf16" | "projection_bf16")
                    && bits & 0xffff != 0
            {
                return Err(Error::State(
                    "reference finite/source BF16 precision".into(),
                ));
            }
        }
        result.push((name, payload));
        at += bytes;
    }
    if at != raw.len() {
        return Err(Error::State("reference trailing bytes".into()));
    }
    Ok(result)
}
fn stamp(m: &fs::Metadata) -> [u64; 7] {
    [
        m.dev(),
        m.ino(),
        m.size(),
        m.mtime() as u64,
        m.mtime_nsec() as u64,
        m.ctime() as u64,
        m.ctime_nsec() as u64,
    ]
}
pub(super) fn compare(
    done: &qualification::CompletedSlice<'_>,
    source: &str,
    width: usize,
    rows: usize,
) -> Result<()> {
    let file = std::env::var(format!("ZG_ARITHMETIC_REFERENCE_{width}")).map_err(|_| {
        Error::State("pinned independently accepted original reference required".into())
    })?;
    let sha = std::env::var(format!("ZG_ARITHMETIC_REFERENCE_{width}_SHA256"))
        .map_err(|_| Error::State("whole original reference SHA required".into()))?;
    let path = Path::new(&file);
    if !path.is_absolute()
        || sha.len() != 64
        || !sha
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(Error::State("reference absolute path/lowercase SHA".into()));
    }
    let mut held = File::open(path)?;
    let before = held.metadata()?;
    if !before.is_file()
        || before.len() as usize > MAX
        || stamp(&before) != stamp(&fs::symlink_metadata(path)?)
    {
        return Err(Error::State("reference bounded regular held file".into()));
    }
    let mut raw = vec![0u8; before.len() as usize];
    held.read_exact(&mut raw)?;
    let mut tail = [0u8; 1];
    if held.read(&mut tail)? != 0
        || raw.len() != before.len() as usize
        || raw.capacity() > MAX
        || format!("{:x}", Sha256::digest(&raw)) != sha
        || stamp(&before) != stamp(&held.metadata()?)
        || stamp(&before) != stamp(&fs::symlink_metadata(path)?)
    {
        return Err(Error::State(
            "reference exact content/EOF/current identity".into(),
        ));
    }
    for (name, expected) in parse(&raw, source, width, rows, done.token())? {
        let actual = done.value(name)?;
        if actual.len() != expected.len() {
            return Err(Error::State(format!("original {name} full extent")));
        }
        if let Some((i, (a, b))) = actual
            .chunks_exact(4)
            .zip(expected.chunks_exact(4))
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            return Err(Error::State(format!(
                "original {name} first differing word{i}: actual={:08x} expected={:08x}",
                u32::from_le_bytes(a.try_into().unwrap()),
                u32::from_le_bytes(b.try_into().unwrap())
            )));
        }
    }
    Ok(())
}

#[test]
fn expected_protocol_refuses_source_geometry_extent_precision_and_trailing_data() {
    let source = "ab".repeat(32);
    let mut raw = Vec::new();
    raw.extend_from_slice(b"ZGBF16V1");
    raw.extend_from_slice(&[0xab; 32]);
    for n in [1024u32, 97, 16, 11] {
        raw.extend_from_slice(&n.to_le_bytes());
    }
    for n in counts(1024, 97) {
        raw.extend_from_slice(&(n as u32 * 4).to_le_bytes());
        raw.resize(raw.len() + n * 4, 0);
    }
    assert_eq!(parse(&raw, &source, 1024, 97, 16).unwrap().len(), 11);
    assert!(parse(&raw, &"cd".repeat(32), 1024, 97, 16).is_err());
    assert!(parse(&raw, &source, 2048, 97, 16).is_err());
    assert!(parse(&raw, &source, 1024, 98, 16).is_err());
    assert!(parse(&raw, &source, 1024, 97, 15).is_err());
    let mut hostile = raw.clone();
    hostile[56..60].copy_from_slice(&4u32.to_le_bytes());
    assert!(parse(&hostile, &source, 1024, 97, 16).is_err());
    hostile = raw.clone();
    hostile[60..64].copy_from_slice(&0x3f800001u32.to_le_bytes());
    assert!(parse(&hostile, &source, 1024, 97, 16).is_err());
    hostile = raw.clone();
    hostile[60..64].copy_from_slice(&f32::INFINITY.to_bits().to_le_bytes());
    assert!(parse(&hostile, &source, 1024, 97, 16).is_err());
    assert!(parse(&raw[..raw.len() - 1], &source, 1024, 97, 16).is_err());
    raw.push(0);
    assert!(parse(&raw, &source, 1024, 97, 16).is_err());
}
