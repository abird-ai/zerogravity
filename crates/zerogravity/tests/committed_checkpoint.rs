//! Real local-source continuation through the opt-in production SDK feature.
#![cfg(all(
    feature = "cpu",
    feature = "model-qwen35",
    feature = "state-checkpoint"
))]
use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use zerogravity::{
    Error,
    qwen35::LoadedQwenModel,
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner, TransferAdmission},
};

struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn admission(bytes: u64) -> TransferAdmission {
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "checkpoint-test").unwrap();
    arbiter.register(resource.clone(), bytes).unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("qwen", "checkpoint").unwrap(),
    }
}
fn reserved(host: &TransferAdmission) -> u64 {
    host.arbiter.snapshot(&host.resource).unwrap().reserved
}

#[test]
#[ignore = "actual approved local Qwen GGUF 0.8B/2B files; no download"]
fn committed_cpu_checkpoint_restores_exact_continuation_and_retains_import_charge() {
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/models"));
    let directory = std::env::temp_dir().join(format!("zg-committed-cpu-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let _cleanup = Temp(directory.clone());
    let host = admission(2 * 1024 * 1024 * 1024);
    let short = admission(1024 * 1024);
    let path = directory.join("session.zg");
    let open = |name: &str| {
        let folder = root.join(name);
        LoadedQwenModel::open(&folder.join("model.gguf"), &folder.join("config.json")).unwrap()
    };
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let source = open(name);
        let mut original = source.new_session().unwrap();
        original.prefill_chunk_without_logits(&[11, 353]).unwrap();
        let committed = original.state_fingerprint();
        // This uncommitted candidate must not enter the persisted state.
        let (_, pending) = original.next_logits(17).unwrap();
        original.save_committed_checkpoint(&path, &host).unwrap();
        assert_eq!(reserved(&host), 0);
        assert_eq!(original.state_fingerprint(), committed);
        let reloaded = open(name);
        assert_ne!(source.source_identity(), reloaded.source_identity());
        let mut receiving = reloaded.new_session().unwrap();
        let (_, stale) = receiving.next_logits(17).unwrap();
        let before = receiving.state_fingerprint();
        assert!(matches!(
            receiving.restore_committed_checkpoint(&path, &short),
            Err(Error::Resource(_))
        ));
        assert_eq!(reserved(&short), 0);
        assert_eq!(
            (
                receiving.root_id(),
                receiving.position(),
                receiving.state_fingerprint()
            ),
            (0, 0, before)
        );

        // Corrupt only the trailer: the full candidate must remain private,
        // and all temporary/retained charges must retire on failure.
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        let mut last = [0u8; 1];
        file.read_exact(&mut last).unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[last[0] ^ 1]).unwrap();
        drop(file);
        assert!(
            receiving
                .restore_committed_checkpoint(&path, &host)
                .is_err()
        );
        assert_eq!(reserved(&host), 0);
        assert_eq!(
            (
                receiving.root_id(),
                receiving.position(),
                receiving.state_fingerprint()
            ),
            (0, 0, before)
        );
        original.save_committed_checkpoint(&path, &host).unwrap();

        if name == "qwen35-0.8b" {
            let foreign = open("qwen35-2b");
            let mut other = foreign.new_session().unwrap();
            assert!(matches!(
                other.restore_committed_checkpoint(&path, &host),
                Err(Error::State(_))
            ));
            assert_eq!(other.root_id(), 0);
            assert_eq!(reserved(&host), 0);
        }

        receiving
            .restore_committed_checkpoint(&path, &host)
            .unwrap();
        assert_eq!(
            (
                receiving.root_id(),
                receiving.position(),
                receiving.state_fingerprint()
            ),
            (1, 2, committed)
        );
        assert!(receiving.commit(stale).is_err());
        let imported = receiving.snapshot();
        let charge = reserved(&host);
        assert!(charge >= receiving.state_bytes() as u64);
        let expected = original.forward(17).unwrap();
        let actual = receiving.forward(17).unwrap();
        assert_eq!(expected.len(), actual.len());
        assert!(
            expected
                .iter()
                .zip(&actual)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name} continuation logits"
        );
        assert_eq!(receiving.state_fingerprint(), original.state_fingerprint());
        assert!(original.commit(pending).is_err());
        assert_eq!(reserved(&host), charge);
        receiving.reset().unwrap();
        assert_eq!(
            reserved(&host),
            charge,
            "last immutable reader retains import"
        );
        drop(imported);
        assert_eq!(reserved(&host), 0);
        println!(
            "COMMITTED_CHECKPOINT_PASS model={name} vocab={} frontier=3",
            actual.len()
        );
    }
}
