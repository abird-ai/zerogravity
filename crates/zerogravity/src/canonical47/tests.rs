use super::*;
use sha2::{Digest, Sha256};
use std::{ffi::OsStr, fs, os::unix::fs::symlink, sync::atomic::AtomicUsize};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "zg-native-loader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn native_paths_follow_only_absolute_xdg_or_home() {
    let suffix = "abird/zerogravity/native/canonical47-v1";
    assert_eq!(
        paths::native_root(None, Some(OsStr::new("/cache")), None).unwrap(),
        Path::new("/cache").join(suffix)
    );
    for x in [None, Some(OsStr::new("relative")), Some(OsStr::new(""))] {
        assert_eq!(
            paths::native_root(None, x, Some(OsStr::new("/home/person"))).unwrap(),
            Path::new("/home/person/.cache").join(suffix)
        );
    }
    assert!(paths::native_root(None, None, Some(OsStr::new("relative"))).is_err());
    assert!(paths::native_root(Some(Path::new("relative")), None, None).is_err());
    assert_eq!(
        paths::native_root(Some(Path::new("/explicit")), None, None).unwrap(),
        Path::new("/explicit")
    );
}
#[test]
fn held_files_check_full_sha_extent_eof_and_cancel() {
    let f = Fixture::new();
    fs::write(f.0.join("input"), b"tiny authentic fixture").unwrap();
    let dir = paths::Directory::open(&f.0).unwrap();
    let sha: [u8; 32] = Sha256::digest(b"tiny authentic fixture").into();
    assert_eq!(
        dir.read("input", 22, Some(&sha), None).unwrap(),
        b"tiny authentic fixture"
    );
    assert!(dir.read("input", 21, Some(&sha), None).is_err());
    assert!(dir.read("input", 23, Some(&sha), None).is_err());
    assert!(dir.read("input", 22, Some(&[0; 32]), None).is_err());
    assert!(
        dir.read("input", 22, Some(&sha), Some(&AtomicBool::new(true)))
            .is_err()
    );
    assert!(dir.bounded("input", 21, None).is_err());
}
#[test]
fn safe_relative_files_refuse_symlinks_ancestors_and_special_files() {
    let f = Fixture::new();
    fs::create_dir(f.0.join("inside")).unwrap();
    fs::write(f.0.join("inside/data"), b"x").unwrap();
    symlink("inside", f.0.join("redirect")).unwrap();
    symlink("inside/data", f.0.join("link")).unwrap();
    let dir = paths::Directory::open(&f.0).unwrap();
    for name in [
        "../escape",
        "/absolute",
        "inside/../data",
        "redirect/data",
        "link",
        "inside",
        "a//b",
    ] {
        assert!(dir.file(name).is_err(), "{name}");
    }
    assert_eq!(dir.read("inside/data", 1, None, None).unwrap(), b"x");
    fs::rename(f.0.join("inside"), f.0.join("old")).unwrap();
    symlink("old", f.0.join("inside")).unwrap();
    assert!(dir.file("inside/data").is_err());
}
fn manifest() -> serde_json::Value {
    serde_json::json!({"schema":"zerogravity-canonical47-native-v1",
        "images":registry::MODEL.iter().map(|k|serde_json::json!({"kind":format!("{k:?}"),"path":format!("images/{k:?}.cubin")})).collect::<Vec<_>>(),
        "rotary":registry::ROTARY.iter().map(|k|serde_json::json!({"kind":format!("{k:?}"),"path":format!("rotary/{k:?}.cubin")})).collect::<Vec<_>>(),
        "activation":"activation.cubin","cublas":"cublas.so","cublas_lt":"cublasLt.so"})
}
#[test]
fn manifest_is_only_a_closed_registry_path_map() {
    let f = Fixture::new();
    for file in ["cublas.so", "cublasLt.so"] {
        fs::write(f.0.join(file), b"fixture; never loaded").unwrap();
    }
    let good = manifest();
    let write = |v: &serde_json::Value| {
        fs::write(
            f.0.join("canonical47-native.json"),
            serde_json::to_vec(v).unwrap(),
        )
        .unwrap()
    };
    write(&good);
    assert!(manifest::Bundle::open(&f.0, None).is_ok());
    for mutate in 0..7 {
        let mut v = good.clone();
        match mutate {
            0 => v["schema"] = "other".into(),
            1 => {
                v["images"].as_array_mut().unwrap().pop();
            }
            2 => v["images"][0]["kind"] = "InventedKind".into(),
            3 => v["images"][0]["sha256"] = "caller hash has no authority".into(),
            4 => v["activation"] = "../outside".into(),
            5 => v["images"][0] = v["images"][1].clone(),
            6 => v["rotary"][0]["path"] = v["images"][0]["path"].clone(),
            _ => unreachable!(),
        }
        write(&v);
        assert!(
            manifest::Bundle::open(&f.0, None).is_err(),
            "mutant{mutate}"
        );
    }
}
#[test]
fn native_payload_read_never_accepts_a_synthetic_image() {
    let f = Fixture::new();
    fs::write(f.0.join("image"), b"fixture").unwrap();
    let dir = paths::Directory::open(&f.0).unwrap();
    let kind = registry::MODEL[0];
    assert!(
        dir.read("image", kind.bytes(), Some(&kind.sha256()), None)
            .is_err()
    );
    assert_eq!(registry::MODEL.len(), 14);
    assert_eq!(registry::ROTARY.len(), 9);
    assert!(registry::MODEL.contains(&native::CubinImageKind::AttentionMultiplySm86));
}
