use super::*;
use std::{fs, io::Write, path::PathBuf};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "zg-gemma-tokenizer-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn host(bytes: u64) -> TransferAdmission {
    let h = TransferAdmission {
        arbiter: ResourceArbiter::new(),
        resource: PhysicalDeviceKey::new("host", "tokenizer fixture").unwrap(),
        owner: ResourceOwner::new("Gemma", "tokenizer fixture").unwrap(),
    };
    h.arbiter.register(h.resource.clone(), bytes).unwrap();
    h
}
fn invalid_files() -> Gemma4TokenizerFiles {
    Gemma4TokenizerFiles {
        tokenizer: File::open("/dev/null").unwrap(),
        tokenizer_config: File::open("/dev/null").unwrap(),
        generation_config: File::open("/dev/null").unwrap(),
        model_config: File::open("/dev/null").unwrap(),
    }
}
#[test]
fn source_refill_refuses_same_size_mutation_before_delivering_untrusted_bytes() {
    let tmp = Temp::new();
    let p = tmp.0.join("source");
    fs::write(&p, b"trusted").unwrap();
    let file = File::open(&p).unwrap();
    let s = stamp(&file).unwrap();
    let held = Held {
        file,
        stamp: s,
        pin: (
            7,
            "a9a089195c68d2adeee23beaa2c3a93b1d40d980d9db9920f2c39bc96668d1c4",
        ),
    };
    let mut reader = Reader {
        source: &held,
        at: 0,
        cancelled: None,
    };
    let mut first = [0; 3];
    reader.read_exact(&mut first).unwrap();
    assert_eq!(first, b"tru"[..]);
    let mut changed = fs::OpenOptions::new().write(true).open(&p).unwrap();
    changed.write_all(b"changed").unwrap();
    changed.sync_all().unwrap();
    let mut rest = [0x7f; 4];
    assert!(reader.read(&mut rest).is_err());
    assert_eq!(rest, [0x7f; 4]);
    assert!(held.verify(None).is_err());
}
#[test]
fn cancellation_and_real_phase_pressure_refuse_before_source_parse_and_retire_all() {
    let h = host(CONSTRUCTION_BYTES + CONTROL_BYTES - 1);
    let flag = AtomicBool::new(true);
    assert!(matches!(
        Gemma4TextTokenizer::prepare(invalid_files(), &h, Some(&flag)),
        Err(Error::State(_))
    ));
    assert_eq!(h.arbiter.snapshot(&h.resource).unwrap().reserved, 0);
    assert!(matches!(
        Gemma4TextTokenizer::prepare(invalid_files(), &h, None),
        Err(Error::Limit(_))
    ));
    assert_eq!(h.arbiter.snapshot(&h.resource).unwrap().reserved, 0);
}
#[test]
fn complete_source_sha_and_role_length_are_required_without_compatibility_fallback() {
    let tmp = Temp::new();
    let p = tmp.0.join("wrong");
    fs::write(&p, b"other").unwrap();
    let file = File::open(&p).unwrap();
    let s = stamp(&file).unwrap();
    let held = Held {
        file,
        stamp: s,
        pin: (
            5,
            "0000000000000000000000000000000000000000000000000000000000000000",
        ),
    };
    assert!(matches!(held.verify(None), Err(Error::State(_))));
    let h = host(128 * MIB);
    assert!(matches!(
        Gemma4TextTokenizer::prepare(invalid_files(), &h, None),
        Err(Error::Malformed(_))
    ));
    assert_eq!(h.arbiter.snapshot(&h.resource).unwrap().reserved, 0);
}

#[test]
fn control_scratch_has_one_nonblocking_operation_and_returns_on_refusal() {
    let busy = AtomicBool::new(false);
    let first = operation(&busy).unwrap();
    std::thread::scope(|scope| {
        scope
            .spawn(|| assert!(matches!(operation(&busy), Err(Error::State(_)))))
            .join()
            .unwrap();
    });
    assert!(operation(&busy).is_err());
    drop(first);
    assert!(operation(&busy).is_ok());
    assert!(!busy.load(Ordering::Acquire));
}
