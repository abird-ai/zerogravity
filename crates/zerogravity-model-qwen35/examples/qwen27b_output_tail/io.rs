use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    path::Path,
};
pub(super) type Proof<T> = Result<T, Box<dyn std::error::Error>>;
pub(super) fn need(ok: bool, why: impl Into<String>) -> Proof<()> {
    if ok { Ok(()) } else { Err(why.into().into()) }
}
pub(super) fn stamp(file: &File) -> Proof<[u64; 7]> {
    let m = file.metadata()?;
    need(m.is_file(), "held regular reference/source")?;
    Ok([
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime() as u64,
        m.mtime_nsec() as u64,
        m.ctime() as u64,
        m.ctime_nsec() as u64,
    ])
}
pub(super) fn open(path: &Path) -> Proof<File> {
    Ok(OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)?)
}
pub(super) fn hash(file: &File, bytes: u64) -> Proof<String> {
    let before = stamp(file)?;
    need(before[2] == bytes, "complete held-file extent")?;
    let mut h = Sha256::new();
    let mut chunk = [0; 65536];
    let mut at = 0;
    while at < bytes {
        let n = (bytes - at).min(chunk.len() as u64) as usize;
        file.read_exact_at(&mut chunk[..n], at)?;
        h.update(&chunk[..n]);
        at += n as u64;
    }
    let mut tail = [0];
    need(
        file.read_at(&mut tail, bytes)? == 0 && stamp(file)? == before,
        "same held SHA/EOF/identity",
    )?;
    Ok(format!("{:x}", h.finalize()))
}
pub(super) fn read(path: &Path, cap: usize, pin: Option<&Value>) -> Proof<Vec<u8>> {
    let mut file = open(path)?;
    let before = stamp(&file)?;
    let count = usize::try_from(before[2])?;
    need(count > 0 && count <= cap, "bounded regular JSON/raw")?;
    let mut out = Vec::new();
    out.try_reserve_exact(count)?;
    out.resize(count, 0);
    file.read_exact(&mut out)?;
    let mut tail = [0];
    need(
        file.read(&mut tail)? == 0 && stamp(&file)? == before,
        "complete file/no truncated bytes",
    )?;
    if let Some(p) = pin {
        need(
            p["bytes"] == count && p["sha256"] == sha(&out),
            "accepted complete reference pin",
        )?;
    }
    Ok(out)
}
pub(super) fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
pub(super) fn json(path: &Path, cap: usize, pin: Option<&Value>) -> Proof<Value> {
    Ok(serde_json::from_slice(&read(path, cap, pin)?)?)
}
pub(super) fn string<'a>(v: &'a Value, key: &str) -> Proof<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("missing string:{key}").into())
}
pub(super) fn write_new(path: &Path, value: &Value) -> Proof<()> {
    let raw = serde_json::to_vec_pretty(value)?;
    need(raw.len() <= 1024 * 1024, "bounded replay report")?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&raw)?;
    file.write_all(b"\n")?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn held_fixture_checks_complete_sha_and_refuses_symlink_and_short_pin() {
        let root = std::env::temp_dir().join(format!("zg27-tail-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("data");
        std::fs::write(&path, b"fixture").unwrap();
        let pin = serde_json::json!({"bytes":7,"sha256":sha(b"fixture")});
        assert_eq!(read(&path, 32, Some(&pin)).unwrap(), b"fixture");
        let mut bad = pin.clone();
        bad["bytes"] = 6.into();
        assert!(read(&path, 32, Some(&bad)).is_err());
        bad = pin;
        bad["sha256"] = "wrong".into();
        assert!(read(&path, 32, Some(&bad)).is_err());
        assert!(read(&path, 6, None).is_err());
        std::os::unix::fs::symlink("data", root.join("link")).unwrap();
        assert!(read(&root.join("link"), 32, None).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
