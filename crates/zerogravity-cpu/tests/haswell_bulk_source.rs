//! Captured DEFAULT operators only; no model, session, GPU or timing qualification.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity_cpu::numerics::haswell_bulk_f32::{
    BulkF32Admission, BulkF32Profile, BulkF32Request, F32View, execute,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const STAGING: u64 = 32 * 1024 * 1024;
const LIMIT: u64 = 128 * 1024 * 1024;

fn fp_control() -> u32 {
    let mut value = 0u32;
    // SAFETY: read-only control instruction writes one valid local word.
    unsafe {
        std::arch::asm!("stmxcsr [{pointer}]", pointer = in(reg) &mut value,
            options(nostack, preserves_flags));
    }
    value
}

fn read_pinned(path: &Path, bytes: usize, digest: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let before = file.metadata()?;
    assert!(before.is_file());
    assert_eq!(before.len(), bytes as u64);
    let mut raw = Vec::with_capacity(bytes + 1);
    (&mut file).take(bytes as u64 + 1).read_to_end(&mut raw)?;
    let after = file.metadata()?;
    assert_eq!(raw.len(), bytes, "complete file and terminal EOF: {path:?}");
    assert_eq!(
        (
            before.dev(),
            before.ino(),
            before.len(),
            before.mtime(),
            before.mtime_nsec(),
            before.ctime(),
            before.ctime_nsec()
        ),
        (
            after.dev(),
            after.ino(),
            after.len(),
            after.mtime(),
            after.mtime_nsec(),
            after.ctime(),
            after.ctime_nsec()
        ),
        "fixture changed during read: {path:?}"
    );
    assert_eq!(format!("{:x}", Sha256::digest(&raw)), digest, "{path:?}");
    Ok(raw)
}

fn words(raw: &[u8]) -> Vec<f32> {
    assert!(raw.len().is_multiple_of(4));
    raw.chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("four bytes")))
        .collect()
}

fn output_digest(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for word in values {
        hash.update(word.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn check_ledger(
    arbiter: &ResourceArbiter,
    key: &PhysicalDeviceKey,
    staging: u64,
    output: u64,
) -> Result<()> {
    let snapshot = arbiter.snapshot(key)?;
    let scratch = if output == 0 { 0 } else { 256 };
    assert_eq!(snapshot.reserved, staging + output + scratch);
    assert_eq!(
        snapshot.reservations,
        usize::from(staging != 0) + usize::from(output != 0)
    );
    assert!(!snapshot.fenced);
    for (class, bytes) in [
        (AllocationClass::TransferStaging, staging),
        (AllocationClass::Output, output),
        (AllocationClass::Scratch, scratch),
    ] {
        assert_eq!(snapshot.per_class.get(&class).copied().unwrap_or(0), bytes);
    }
    Ok(())
}

fn replay(
    root: &Path,
    arbiter: &ResourceArbiter,
    key: &PhysicalDeviceKey,
    owner: &ResourceOwner,
    av: bool,
) -> Result<usize> {
    let staging = arbiter.admit(
        key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: STAGING,
        }],
    )?;
    check_ledger(arbiter, key, STAGING, 0)?;
    let folder = root.join(if av {
        ".agents/runs/numerics-and-context-av-replay-20261020/root-fixtures"
    } else {
        ".agents/runs/numerics-and-context-2b-alpha-replay-20261020/root-fixtures"
    });
    let (left, right, a_shape, a_strides, b_shape, b_strides, output_shape, expected_hash) = if av {
        let dense = words(&read_pinned(
            &folder.join("bulk64.v.f32le"),
            524_288,
            "b97493ac1b6f2b6f935c65d9627f2581c912e3e48c1afea1919c2e7c2e5c5b29",
        )?);
        // Original strided view; uncaptured gaps are transport padding.
        let mut left = vec![0.0; 4_317_184];
        for head in 0..2 {
            for row in 0..256 {
                let at = head * 2_162_688 + row * 8_448;
                let source = (head * 256 + row) * 256;
                left[at..at + 256].copy_from_slice(&dense[source..source + 256]);
            }
        }
        let right = words(&read_pinned(
            &folder.join("bulk64.p.f32le"),
            524_288,
            "2457d3616026d4307e61d832ebd22a612716efee3714345bfa6a740cf380d074",
        )?);
        (
            left,
            right,
            [256, 256, 2, 1],
            [1, 8_448, 2_162_688, 4_325_376],
            [256, 64, 8, 1],
            [1, 256, 16_384, 131_072],
            [256, 64, 8, 1],
            "91adb9f1d4a55e57a068106d195f928205f1031a998a5b9d4cb3fdc633e507b9",
        )
    } else {
        let left = words(&read_pinned(
            &folder.join("weight.f32le"),
            131_072,
            "0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876",
        )?);
        let right = words(&read_pinned(
            &folder.join("bulk64.x.f32le"),
            524_288,
            "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6",
        )?);
        (
            left,
            right,
            [2_048, 16, 1, 1],
            [1, 2_048, 32_768, 32_768],
            [2_048, 64, 1, 1],
            [1, 2_048, 131_072, 131_072],
            [16, 64, 1, 1],
            "448f5626f2f06de0e5fe0f375a92589bd87b81833fcbb9daca530c20f8424b3e",
        )
    };
    let source = SourceIdentity::for_new_allocation()?;
    let request = BulkF32Request {
        profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
        source,
        left: F32View {
            source,
            name: if av {
                "cache_v_l3 (view) (permuted)"
            } else {
                "blk.0.ssm_alpha.weight"
            },
            values: &left,
            offset: 0,
            shape: a_shape,
            strides: a_strides,
        },
        right: F32View {
            source,
            name: if av { "kq_soft_max-3" } else { "attn_norm-0" },
            values: &right,
            offset: 0,
            shape: b_shape,
            strides: b_strides,
        },
    };
    let admission = || BulkF32Admission {
        arbiter,
        resource: key,
        owner,
    };
    let output_words = output_shape.iter().product::<usize>();
    let output_bytes = (output_words * 4) as u64;
    let pressure = arbiter.admit(
        key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: LIMIT - STAGING - output_bytes - 256 + 1,
        }],
    )?;
    let reserved = arbiter.snapshot(key)?.reserved;
    assert!(matches!(
        execute(request, admission(), None),
        Err(zerogravity_core::Error::Limit(_))
    ));
    assert_eq!(arbiter.snapshot(key)?.reserved, reserved);
    drop(pressure);
    let result = execute(request, admission(), None)?;
    assert_eq!(result.shape(), output_shape);
    assert_eq!(result.source(), source);
    assert_eq!(result.values().len(), output_words);
    check_ledger(arbiter, key, STAGING, output_bytes)?;

    // First expected-output open occurs after execute returns.
    let expected = read_pinned(
        &folder.join("bulk64.expected.f32le"),
        output_words * 4,
        expected_hash,
    )?;
    let unequal = result
        .values()
        .iter()
        .zip(expected.chunks_exact(4))
        .filter(|(actual, bytes)| {
            actual.to_bits() != u32::from_le_bytes((*bytes).try_into().expect("four bytes"))
        })
        .count();
    assert_eq!(
        unequal,
        0,
        "{} complete DEFAULT output bits",
        if av { "AV" } else { "alpha" }
    );
    drop(expected);
    drop(left);
    drop(right);
    drop(staging);
    check_ledger(arbiter, key, 0, output_bytes)?;
    assert_eq!(
        output_digest(result.values()),
        expected_hash,
        "retained output after input release"
    );
    drop(result);
    check_ledger(arbiter, key, 0, 0)?;
    Ok(output_words)
}

#[test]
#[ignore = "requires existing authentic DEFAULT captures via absolute ZG_REFERENCE_ROOT"]
fn captured_default_av_and_alpha_match_all_source_bits() -> Result<()> {
    let root = PathBuf::from(std::env::var_os("ZG_REFERENCE_ROOT").expect("set ZG_REFERENCE_ROOT"));
    assert!(root.is_absolute() && root.is_dir());
    assert_eq!(fp_control() & 0xffc0, 0x1f80, "RNE/masked/noFTZ/noDAZ");
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "captured-default-f32")?;
    arbiter.register(key.clone(), LIMIT)?;
    let owner = ResourceOwner::new("captured-default-f32", "source-only-test")?;
    check_ledger(&arbiter, &key, 0, 0)?;
    let compared = replay(&root, &arbiter, &key, &owner, true)?
        + replay(&root, &arbiter, &key, &owner, false)?;
    assert_eq!(compared, 132_096);
    check_ledger(&arbiter, &key, 0, 0)?;
    assert_eq!(fp_control() & 0xffc0, 0x1f80, "unchanged FP controls");
    Ok(())
}
